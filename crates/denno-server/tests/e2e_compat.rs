//! P821: 両 API (Anthropic `/v1/messages` / OpenAI `/v1/chat/completions`) を
//! **同一サーバー上で混在**させた end-to-end 互換性・並行負荷テスト。
//!
//! 既存の個別テスト (openai_chat / openai_tools / tools / scheduler /
//! client_disconnect / ops_endpoints) は片方の API を単独で見る。ここでは
//! 「同じ Scheduler・同じ KV・同じ chat template を通る」という設計契約を、
//! 混在負荷の下で固定する。カバー項目:
//!
//! 1. ストリーミング / 非ストリーミングを交互に並行実行しても、各リクエストの
//!    出力が単独実行の基準と一致する (KV 切替え = インターリーブ exactness)。
//!    Anthropic と OpenAI の同じ会話は同じ本文を返す。
//! 2. tool calling の往復 (tool_use / tool_calls → tool_result / role=tool) を
//!    両 API で行う。ストリーミングの組み立ても両方で確認する。
//! 3. 生成中のクライアント切断で生成が止まり、スロットが解放される (両 API)。
//! 4. キュー満杯のバックプレッシャ。
//!    実装の契約は Anthropic が **503** `overloaded_error`、OpenAI が
//!    **429** `rate_limit_exceeded` (`max_queue-depth` 超過時、P873 で
//!    プロトコル別に分かれた)。どちらも `Retry-After` を伴う。
//!    429 `rate_limit_error` の「KV スロット枯渇」経路は別で、
//!    `tests/admit_error_message.rs` が単一スロット構成で実 HTTP 検証している。
//! 5. 壊れた JSON のエラー封筒 (両 API で形が異なる) と、その直後の回復。
//!
//! すべて CPU のみ (ダミーモデル / ダミートークナイザ)。GPU は使わない。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{
    dummy_tokenizer, tool_tokenizer, KvSensitiveModel, DUMMY_EOS, DUMMY_VOCAB, TOOL_VOCAB,
};
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------

/// 決定的で **状態を持たない** モデル。peak = 直前トークン + 1。
/// 同一プロンプトなら、どの API・どの並行度でも同じ本文になる (比較の基準)。
/// `delay` は並行進行 (インターリーブ) を確実に起こすため。
struct StepModel {
    delay: Duration,
    /// EOS を出さない (切断テストで生成を長引かせる)
    endless: bool,
}

impl Model for StepModel {
    fn context(&self) -> usize {
        512
    }

    fn eos_ids(&self) -> &[u32] {
        if self.endless {
            &[]
        } else {
            &[DUMMY_EOS]
        }
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let mut peak = (last + 1) % DUMMY_VOCAB;
        if self.endless && peak == DUMMY_EOS as usize {
            peak = (peak + 1) % DUMMY_VOCAB;
        }
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, 512)
    }
}

/// 固定トークン列を 1 forward ずつ再生する決定論モデル。
///
/// `testutil::ScriptModel` は「call 数 = スクリプト位置」の**単一セッション専用**
/// なので、同一サーバーへ複数リクエストを送ると位置が混ざる。ここでは
/// 「直前トークンが自分が最後に出したトークンなら続き、そうでなければ先頭から」
/// という規則でカーソルを進める (prefill は必ず先頭に戻る)。1 リクエストずつ
/// 順に送る tool calling 往復テストで使う。
struct ReplayModel {
    script: Vec<u32>,
    cursor: usize,
}

impl ReplayModel {
    fn new(script: Vec<u32>) -> Self {
        Self { script, cursor: 0 }
    }
}

impl Model for ReplayModel {
    fn context(&self) -> usize {
        256
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        // 継続 = decode (1 トークン) かつ、それがスクリプトの直前の出力と一致する。
        let continues = self.cursor > 0
            && tokens.len() == 1
            && self.script.get(self.cursor - 1).copied() == tokens.first().copied();
        if !continues {
            self.cursor = 0;
        }
        let idx = self.cursor.min(self.script.len().saturating_sub(1));
        let peak = self.script[idx];
        self.cursor = self.cursor.saturating_add(1);
        let mut logits = vec![0.0f32; 48];
        logits[peak as usize] = 5.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, 256)
    }
}

/// TOOL_VOCAB の piece → id。
fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// 完全な tool_call block を吐くスクリプト (名前 + `city=Tokyo`)。
fn block_script(name: &str) -> Vec<u32> {
    [
        "Checking.",
        "<tool_call>",
        "Ċ",
        "<function=",
        name,
        ">",
        "Ċ",
        "<parameter=",
        "city",
        ">",
        "Ċ",
        "Tokyo",
        "Ċ",
        "</parameter>",
        "Ċ",
        "</function>",
        "Ċ",
        "</tool_call>",
        "eos",
    ]
    .iter()
    .map(|p| tid(p))
    .collect()
}

// ---------------------------------------------------------------------------
// App / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn state_with(model: Box<dyn Model>, tok: DennoTokenizer, max_queue: usize) -> Arc<AppState> {
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), max_queue);
    Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

fn step_state(delay_ms: u64, endless: bool, max_queue: usize) -> Arc<AppState> {
    state_with(
        Box::new(StepModel { delay: Duration::from_millis(delay_ms), endless }),
        dummy_tokenizer().unwrap(),
        max_queue,
    )
}

fn replay_state(script: Vec<u32>) -> Arc<AppState> {
    state_with(Box::new(ReplayModel::new(script)), tool_tokenizer().unwrap(), 8)
}

fn kv_state(delay_ms: u64, max_queue: usize) -> Arc<AppState> {
    state_with(
        Box::new(KvSensitiveModel { delay: Duration::from_millis(delay_ms) }),
        dummy_tokenizer().unwrap(),
        max_queue,
    )
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

/// P873: 応答ヘッダも見たいテスト用 (過負荷応答の `Retry-After` 確認)。
/// 戻り値は (status, Retry-After があるか, body)。
async fn post_with_retry_header(app: &Router, uri: &str, body: Value) -> (StatusCode, bool, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let has_retry_after = resp
        .headers()
        .contains_key(axum::http::header::RETRY_AFTER);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, has_retry_after, v)
}

async fn post_raw_body(app: &Router, uri: &str, body: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

async fn post_sse(app: &Router, uri: &str, body: Value) -> String {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// SSE 本文の `data: ...` 行 (JSON) を順に返す ([DONE] は除く)。
fn sse_jsons(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str(d).ok())
        .collect()
}

/// Anthropic SSE の text_delta を連結する。
fn anthropic_stream_text(text: &str) -> String {
    sse_jsons(text)
        .iter()
        .filter(|v| v["type"] == "content_block_delta")
        .filter_map(|v| v["delta"]["text"].as_str())
        .collect()
}

/// OpenAI SSE の delta.content を連結する。
fn openai_stream_text(text: &str) -> String {
    sse_jsons(text)
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

fn user_msg(text: &str) -> Value {
    json!([{"role": "user", "content": text}])
}

fn anthropic_body(text: &str, max_tokens: usize, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": user_msg(text),
    })
}

fn openai_body(text: &str, max_tokens: usize, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": text}],
    })
}

// ---------------------------------------------------------------------------
// 1. 混在 (Anthropic / OpenAI × stream / non-stream) の決定性
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn interleaved_streaming_and_nonstreaming_are_deterministic() {
    let state = kv_state(6, 8);
    let app = denno_server::router(state.clone());

    // 基準: 同じ app へ順に 1 本ずつ送った本文 (KV 依存モデルなので
    // 切替えが壊れると値が変わる)。max_tokens は KV 依存の argmax が
    // 動き続ける長さにする。
    let prompts = ["t15", "t9"];
    let mut refs = Vec::new();
    for p in prompts {
        let (s, v) = post(&app, "/v1/messages", anthropic_body(p, 20, false)).await;
        assert_eq!(s, StatusCode::OK, "body = {v}");
        refs.push(v["content"][0]["text"].as_str().unwrap().to_string());
    }
    assert_ne!(refs[0], refs[1], "テストの前提: プロンプトが違えば本文も違うこと");

    // 並行度の監視 (実際に 2 本以上が同時進行していることを確かめる)
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let monitor = {
        let state = state.clone();
        let max_in_flight = max_in_flight.clone();
        let done = done.clone();
        tokio::spawn(async move {
            while !done.load(Ordering::SeqCst) {
                max_in_flight.fetch_max(state.scheduler.in_flight(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    // 8 本を交互に投げる: index 偶数 = Anthropic、奇数 = OpenAI。
    // stream は index % 4 < 2 で有効 (= 4 通りの組合せがすべて出る)。
    let mut handles = Vec::new();
    for i in 0..8usize {
        let app = app.clone();
        let prompt = prompts[i % prompts.len()].to_string();
        handles.push(tokio::spawn(async move {
            let stream = i % 4 < 2;
            if i % 2 == 0 {
                let body = anthropic_body(&prompt, 20, stream);
                if stream {
                    let text = post_sse(&app, "/v1/messages", body).await;
                    assert!(text.contains("event: message_stop\n"), "SSE 不完全: {text}");
                    Ok::<String, String>(anthropic_stream_text(&text))
                } else {
                    let (s, v) = post(&app, "/v1/messages", body).await;
                    if s != StatusCode::OK {
                        return Err(format!("anthropic 非ストリーム status={s} body={v}"));
                    }
                    Ok(v["content"][0]["text"].as_str().unwrap().to_string())
                }
            } else {
                let body = openai_body(&prompt, 20, stream);
                if stream {
                    let text = post_sse(&app, "/v1/chat/completions", body).await;
                    assert!(
                        text.trim_end().ends_with("data: [DONE]"),
                        "OpenAI SSE の終端が [DONE] でない: {text}"
                    );
                    Ok::<String, String>(openai_stream_text(&text))
                } else {
                    let (s, v) = post(&app, "/v1/chat/completions", body).await;
                    if s != StatusCode::OK {
                        return Err(format!("openai 非ストリーム status={s} body={v}"));
                    }
                    Ok(v["choices"][0]["message"]["content"].as_str().unwrap().to_string())
                }
            }
        }));
    }

    for (i, h) in handles.into_iter().enumerate() {
        let got = h
            .await
            .unwrap()
            .unwrap_or_else(|e| panic!("request {i}: {e}"));
        let want = &refs[i % prompts.len()];
        assert_eq!(
            &got, want,
            "request {i} (prompt={}) の本文が単独実行と不一致 \
             (混在負荷で KV 切替え / ストリーム組み立てが壊れた)",
            prompts[i % prompts.len()]
        );
    }
    done.store(true, Ordering::SeqCst);
    let _ = monitor.await;

    assert!(
        max_in_flight.load(Ordering::SeqCst) >= 2,
        "2 本以上が並行進行していない (混在負荷を検証できていない): max = {}",
        max_in_flight.load(Ordering::SeqCst)
    );
    assert_eq!(state.scheduler.in_flight(), 0);
    assert_eq!(state.scheduler.pending(), 0);
}

// ---------------------------------------------------------------------------
// 2. tool calling の往復 (両 API)
// ---------------------------------------------------------------------------

fn anthropic_tools() -> Value {
    json!([{
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }])
}

fn openai_tools() -> Value {
    json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }])
}

#[tokio::test]
async fn tool_call_roundtrip_anthropic_and_openai() {
    let state = replay_state(block_script("get_weather"));
    let app = denno_server::router(state.clone());

    // ---- Anthropic: 非ストリーミング → tool_result で 2 ターン目 ----
    let body = json!({
        "model": "denno-test",
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": user_msg("weather in tokyo?"),
        "tools": anthropic_tools(),
    });
    let (s, first) = post(&app, "/v1/messages", body).await;
    assert_eq!(s, StatusCode::OK, "body = {first}");
    assert_eq!(first["stop_reason"], "tool_use", "body = {first}");
    assert_eq!(first["content"][0]["type"], "text");
    assert_eq!(first["content"][0]["text"], "Checking.");
    let tu = first["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap_or_else(|| panic!("tool_use block が無い: {first}"));
    assert_eq!(tu["name"], "get_weather");
    assert_eq!(tu["input"], json!({"city": "Tokyo"}));
    let tu_id = tu["id"].as_str().unwrap().to_string();
    assert!(tu_id.starts_with("toolu_"), "id = {tu_id}");

    // 2 ターン目: 返した assistant メッセージ + tool_result を履歴に積む
    let mut follow = json!({
        "model": "denno-test",
        "max_tokens": 16,
        "temperature": 0.0,
        "tools": anthropic_tools(),
        "messages": [
            {"role": "user", "content": "weather in tokyo?"},
            {"role": "assistant", "content": first["content"].clone()},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tu_id, "content": "sunny 25C"
            }]},
        ],
    });
    let (s, second) = post(&app, "/v1/messages", follow.clone()).await;
    assert_eq!(s, StatusCode::OK, "tool_result 往復が通らない: {second}");
    assert_eq!(second["type"], "message");

    // 未知の tool_use_id は 400 (往復の整合が検証されている証拠)
    follow["messages"][2]["content"][0]["tool_use_id"] = json!("toolu_does_not_exist");
    let (s, bad) = post(&app, "/v1/messages", follow).await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "body = {bad}");
    assert_eq!(bad["error"]["type"], "invalid_request_error");

    // ---- OpenAI: 非ストリーミング → role=tool で 2 ターン目 ----
    let body = json!({
        "model": "denno-test",
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "weather in tokyo?"}],
        "tools": openai_tools(),
    });
    let (s, first) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(s, StatusCode::OK, "body = {first}");
    assert_eq!(first["choices"][0]["finish_reason"], "tool_calls", "body = {first}");
    let calls = first["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "calls = {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"city": "Tokyo"}));
    let call_id = calls[0]["id"].as_str().unwrap().to_string();

    // 同じ会話を Anthropic に投げても同じ tool 名 / 引数になる
    let (s, an) = post(&app, "/v1/messages", json!({
        "model": "denno-test",
        "max_tokens": 64,
        "temperature": 0.0,
        "messages": user_msg("weather in tokyo?"),
        "tools": anthropic_tools(),
    }))
    .await;
    assert_eq!(s, StatusCode::OK, "body = {an}");
    let an_tu = an["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .unwrap();
    assert_eq!(an_tu["name"], calls[0]["function"]["name"]);
    assert_eq!(an_tu["input"], args);

    let (s, second) = post(&app, "/v1/chat/completions", json!({
        "model": "denno-test",
        "max_tokens": 16,
        "temperature": 0.0,
        "tools": openai_tools(),
        "messages": [
            {"role": "user", "content": "weather in tokyo?"},
            {"role": "assistant", "content": null, "tool_calls": calls},
            {"role": "tool", "tool_call_id": call_id, "content": "sunny 25C"},
        ],
    }))
    .await;
    assert_eq!(s, StatusCode::OK, "role=tool 往復が通らない: {second}");
    assert_eq!(second["object"], "chat.completion");
}

#[tokio::test]
async fn tool_calls_streaming_assembly_both_apis() {
    let state = replay_state(block_script("get_weather"));
    let app = denno_server::router(state);

    // ---- Anthropic: content_block_start(tool_use) + input_json_delta ----
    let text = post_sse(
        &app,
        "/v1/messages",
        json!({
            "model": "denno-test",
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": true,
            "messages": user_msg("weather in tokyo?"),
            "tools": anthropic_tools(),
        }),
    )
    .await;
    let evs = sse_jsons(&text);
    let start = evs
        .iter()
        .find(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .unwrap_or_else(|| panic!("tool_use の content_block_start が無い:\n{text}"));
    assert_eq!(start["content_block"]["name"], "get_weather");
    let partial: String = evs
        .iter()
        .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "input_json_delta")
        .map(|e| e["delta"]["partial_json"].as_str().unwrap_or("").to_string())
        .collect();
    let parsed: Value = serde_json::from_str(&partial).expect("input_json_delta が JSON でない");
    assert_eq!(parsed, json!({"city": "Tokyo"}));
    assert!(
        evs.iter().any(|e| e["type"] == "message_delta"
            && e["delta"]["stop_reason"] == "tool_use"),
        "stop_reason=tool_use の message_delta が無い:\n{text}"
    );

    // ---- OpenAI: delta.tool_calls (index / id / name / 増分 arguments) ----
    let text = post_sse(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "denno-test",
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": true,
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": openai_tools(),
        }),
    )
    .await;
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");
    let chunks = sse_jsons(&text);
    let mut name = String::new();
    let mut args = String::new();
    let mut id = String::new();
    for c in &chunks {
        let Some(tcs) = c["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for tc in tcs {
            assert_eq!(tc["index"], 0);
            if let Some(n) = tc["function"]["name"].as_str() {
                name.push_str(n);
                id = tc["id"].as_str().unwrap_or_default().to_string();
                assert_eq!(tc["function"]["arguments"], "");
            } else if let Some(a) = tc["function"]["arguments"].as_str() {
                args.push_str(a);
                assert!(tc.get("id").is_none(), "id が 2 回載っている: {tc}");
            }
        }
    }
    assert_eq!(name, "get_weather");
    assert!(id.starts_with("call_"), "id = {id}");
    let parsed: Value = serde_json::from_str(&args).expect("増分 arguments が JSON でない");
    assert_eq!(parsed, json!({"city": "Tokyo"}));
    let last = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
}

// ---------------------------------------------------------------------------
// 3. 生成中のクライアント切断 (両 API、実 TCP)
// ---------------------------------------------------------------------------

async fn serve_tcp(state: Arc<AppState>) -> std::net::SocketAddr {
    let app: Router = denno_server::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

async fn post_raw(addr: std::net::SocketAddr, path: &str, body: &str) -> tokio::net::TcpStream {
    use tokio::io::AsyncWriteExt;
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s.flush().await.unwrap();
    s
}

async fn read_until(
    stream: &mut tokio::net::TcpStream,
    buf: &mut String,
    marker: &str,
    secs: u64,
) -> bool {
    use tokio::io::AsyncReadExt;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    let mut tmp = [0u8; 4096];
    while !buf.contains(marker) {
        let remain = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remain.is_zero() {
            return false;
        }
        match tokio::time::timeout(remain, stream.read(&mut tmp)).await {
            Ok(Ok(0)) | Ok(Err(_)) | Err(_) => return false,
            Ok(Ok(n)) => buf.push_str(&String::from_utf8_lossy(&tmp[..n])),
        }
    }
    true
}

async fn wait_in_flight(state: &AppState, want: usize, secs: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    while tokio::time::Instant::now() < deadline {
        if state.scheduler.in_flight() == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!(
        "in_flight が {want} にならない (現在 {})",
        state.scheduler.in_flight()
    );
}

/// ストリーム途中で切断 → Run が打ち切られ、スロットが解放され、次の要求が通る。
/// 両 API を同じ形で検証する (raw チャネル/SSE 送信の失敗が拾われている証拠)。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn midstream_disconnect_releases_slot_on_both_apis() {
    for (path, body, marker) in [
        (
            "/v1/messages",
            anthropic_body("t0", 300, true).to_string(),
            "event: message_start",
        ),
        (
            "/v1/chat/completions",
            openai_body("t0", 300, true).to_string(),
            "chat.completion.chunk",
        ),
    ] {
        let state = step_state(20, true, 8);
        let addr = serve_tcp(state.clone()).await;

        let mut stream = post_raw(addr, path, &body).await;
        let mut buf = String::new();
        assert!(
            read_until(&mut stream, &mut buf, marker, 10).await,
            "{path}: 最初のイベントが届かない: {buf}"
        );
        wait_in_flight(&state, 1, 5).await;
        drop(stream); // 生成中の切断

        wait_in_flight(&state, 0, 5).await;

        // 後続 (非ストリーム) が完了する = ワーカー / KV スロットは健常
        let app = denno_server::router(state.clone());
        let (s, v) = post(&app, path, json!({
            "model": "denno-test",
            "max_tokens": 4,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
        }))
        .await;
        assert_eq!(s, StatusCode::OK, "{path}: 切断後の要求が失敗: {v}");
        wait_in_flight(&state, 0, 5).await;
    }
}

// ---------------------------------------------------------------------------
// 4. キュー満杯のバックプレッシャ (503)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_full_backpressure_is_retryable_on_both_apis() {
    // forward 20ms × 数トークン。キュー容量 2 に対し 8 本をほぼ同時に投げる。
    let state = step_state(20, false, 2);
    let app = denno_server::router(state.clone());

    let (s, reference) = post(&app, "/v1/messages", anthropic_body("t0", 8, false)).await;
    assert_eq!(s, StatusCode::OK, "body = {reference}");
    let expected = reference["content"][0]["text"].as_str().unwrap().to_string();
    assert!(!expected.is_empty());

    let mut tasks = Vec::new();
    for i in 0..8usize {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            let anthropic = i % 2 == 0;
            let body = if anthropic {
                anthropic_body("t0", 8, false)
            } else {
                openai_body("t0", 8, false)
            };
            let uri = if anthropic { "/v1/messages" } else { "/v1/chat/completions" };
            let (status, has_retry_after, v) = post_with_retry_header(&app, uri, body).await;
            (i, anthropic, status, has_retry_after, v)
        }));
    }

    let mut ok = 0usize;
    let mut overloaded = 0usize;
    for t in tasks {
        let (i, anthropic, status, has_retry_after, v) = t.await.unwrap();
        match status {
            StatusCode::OK => {
                ok += 1;
                let text = if anthropic {
                    v["content"][0]["text"].as_str().unwrap_or("").to_string()
                } else {
                    v["choices"][0]["message"]["content"]
                        .as_str()
                        .unwrap_or("")
                        .to_string()
                };
                assert_eq!(text, expected, "request {i}: 受理済みの本文が基準と不一致");
            }
            // 実装の契約 (P873 でプロトコル別に分かれた): キュー満杯は
            // Anthropic が 503 overloaded_error、OpenAI が 429 rate_limit_exceeded。
            // どちらも `Retry-After` を伴う。
            StatusCode::SERVICE_UNAVAILABLE | StatusCode::TOO_MANY_REQUESTS => {
                overloaded += 1;
                if anthropic {
                    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {v}");
                    assert_eq!(v["type"], "error", "body = {v}");
                    assert_eq!(v["error"]["type"], "overloaded_error", "body = {v}");
                } else {
                    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "body = {v}");
                    assert_eq!(v["error"]["type"], "rate_limit_exceeded", "body = {v}");
                }
                let msg = v["error"]["message"].as_str().unwrap_or("");
                assert!(
                    msg.contains("queue is full"),
                    "再試行可能な理由が本文に無い: {v}"
                );
                assert!(has_retry_after, "request {i}: Retry-After が無い");
            }
            other => panic!("unexpected status {other} (request {i}): {v}"),
        }
    }
    assert_eq!(ok + overloaded, 8);
    assert!(
        overloaded >= 1,
        "過負荷応答が 1 件も無い (キュー上限が効いていない: ok={ok})"
    );
    assert!(ok >= 2, "受理数が少なすぎる: ok={ok}");

    // 満杯で拒否された直後でもサーバーは通常応答に戻る
    let (s, v) = post(&app, "/v1/chat/completions", openai_body("t0", 8, false)).await;
    assert_eq!(s, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["message"]["content"], expected);
    assert_eq!(state.scheduler.in_flight(), 0);
    assert_eq!(state.scheduler.pending(), 0);
}

// ---------------------------------------------------------------------------
// 5. 壊れた JSON のエラー封筒 + 回復
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_json_envelopes_and_recovery() {
    let state = step_state(0, false, 8);
    let app = denno_server::router(state);

    // Anthropic: {"type":"error","error":{"type","message"}}
    let (s, v) = post_raw_body(&app, "/v1/messages", "{\"messages\": ").await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("invalid request body"));

    // OpenAI: {"error":{"type","message","param"}}
    let (s, v) = post_raw_body(&app, "/v1/chat/completions", "{\"messages\": ").await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].is_string());

    // 空ボディも 400 (どちらも panic せず封筒を返す)
    for uri in ["/v1/messages", "/v1/chat/completions"] {
        let (s, v) = post_raw_body(&app, uri, "").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: body = {v}");
        assert!(v["error"]["type"].is_string(), "{uri}: body = {v}");
    }

    // 不正 JSON の直後でも正常要求は通る
    let (s, v) = post(&app, "/v1/messages", anthropic_body("t0", 4, false)).await;
    assert_eq!(s, StatusCode::OK, "body = {v}");
    assert!(!v["content"][0]["text"].as_str().unwrap().is_empty());
    let (s, v) = post(&app, "/v1/chat/completions", openai_body("t0", 4, false)).await;
    assert_eq!(s, StatusCode::OK, "body = {v}");
    assert!(!v["choices"][0]["message"]["content"].as_str().unwrap().is_empty());
}
