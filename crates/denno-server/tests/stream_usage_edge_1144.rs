//! task 1144: `stream_options.include_usage` の境界 (edge) と tool-call パリティ。
//!
//! 936 が仕様の基本形 (true の usage chunk / false・未指定の非出現) を押さえて
//! いるのに対し、ここでは同じ 4 状態 (true / false / null / 未指定) を
//! chat・legacy completions・tool-call ストリームの全経路で表駆動に確認し、
//! さらに JSON の null が 2 か所 (stream_options 自体 / include_usage) で
//! 意味が違うことを固定する:
//!
//! - `"stream_options": null` は Option の None と同じ = 未指定扱い (usage 無し)。
//! - `"stream_options": {"include_usage": null}` は bool として不正なので
//!   OpenAI 形式の 400 invalid_request_error (SSE を開始しない)。
//! - `"stream_options": {}` は既定 false。
//!
//! また非ストリーミング要求では `include_usage: true` が chunk 契約を変えず、
//! 通常の `usage` オブジェクトを 1 個だけ返すことを確認する。
//!
//! 方式は tests/openai_stream_usage_936.rs と同じ (ダミートークナイザ +
//! 決定的モデルで HTTP ワイヤを見る)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{
    dummy_tokenizer, tool_tokenizer, ScriptModel, DUMMY_EOS, DUMMY_VOCAB, TOOL_VOCAB,
};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 512;

/// 決定的ダミーモデル (peak = 直前トークン + 1、EOS の位置だけ避ける)。
struct SeqModel;

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = (last + 1) % DUMMY_VOCAB;
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn app_from(engine: Engine) -> Router {
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

fn test_app() -> Router {
    let engine = Engine::from_parts(
        Box::new(SeqModel),
        dummy_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    app_from(engine)
}

/// ScriptModel は forward 呼び出し回数を持つので、リクエストごとに新しい
/// Engine を作る (同じ script なら同じ生成列)。
fn tool_app(name: &str) -> Router {
    let model = ScriptModel::new(block_script(name), tid("t1"));
    let engine = Engine::from_parts(
        Box::new(model),
        tool_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    app_from(engine)
}

fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// `get_weather(city=Tokyo)` の完全な `<tool_call>` block を吐くスクリプト。
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

/// 生の応答 (status / content-type / 本文文字列)。
struct Resp {
    status: StatusCode,
    ctype: String,
    text: String,
}

async fn call(app: &Router, uri: &str, body: &Value) -> Resp {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp {
        status,
        ctype,
        text: String::from_utf8(bytes.to_vec()).unwrap(),
    }
}

/// `data: ...` 行の JSON を到着順に返す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn completion_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "prompt": "t0 t1 t2",
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn tool_body(stream: bool, include: Option<Value>) -> Value {
    let mut b = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "weather in tokyo?"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": stream,
        "tools": [{
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
        }],
    });
    if let Some(so) = include {
        b["stream_options"] = so;
    }
    b
}

/// `usage` の 3 フィールドが自己整合であることを確かめ (p, c) を返す。
fn assert_usage_complete(usage: &Value) -> (u64, u64) {
    let p = usage["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("prompt_tokens が数値でない: {usage}"));
    let c = usage["completion_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("completion_tokens が数値でない: {usage}"));
    let t = usage["total_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("total_tokens が数値でない: {usage}"));
    assert_eq!(t, p + c, "total_tokens != prompt + completion: {usage}");
    (p, c)
}

/// usage chunk が末尾 1 個で、直後が `[DONE]` であることを確かめる。
fn assert_usage_chunk_is_last(chunks: &[Value], text: &str) -> Value {
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
    let last = chunks.last().unwrap();
    assert_eq!(
        last["choices"],
        json!([]),
        "usage chunk の choices が空配列でない: {last}"
    );
    assert!(
        last["usage"].is_object(),
        "末尾 chunk が usage でない: {last}"
    );
    let n = chunks.iter().filter(|c| c["usage"].is_object()).count();
    assert_eq!(n, 1, "usage chunk が {n} 個ある: {chunks:?}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない:\n{text}"
    );
    assert!(
        !text.contains("[DONE]\n\ndata:"),
        "[DONE] の後に data 行がある:\n{text}"
    );
    last.clone()
}

/// 非最終 chunk がすべて `usage: null` (フィールド欠落ではない) であること。
fn assert_null_slots_except_last(chunks: &[Value]) {
    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(
            c.get("usage"),
            Some(&Value::Null),
            "非最終 chunk の usage が明示 null でない: {c}"
        );
    }
}

/// `usage` フィールド自体が本文に一度も現れないこと。
fn assert_no_usage_at_all(text: &str) {
    assert!(!text.contains("\"usage\""), "usage が出ている:\n{text}");
}

/// id / created だけを潰した chunk 列 (2 リクエストの形を比較するため)。
fn normalized(mut chunks: Vec<Value>) -> Vec<Value> {
    for c in &mut chunks {
        if let Some(o) = c.as_object_mut() {
            o.insert("id".to_string(), json!("ID"));
            o.insert("created".to_string(), json!(0));
        }
    }
    chunks
}

/// stream_options に入れる値。4 状態 + JSON null の 2 変種。
#[derive(Clone, Copy)]
enum So {
    Omitted,
    EmptyObject,
    False,
    True,
    /// `"stream_options": null` (Option の None = 未指定扱い)
    NullObject,
    /// `"stream_options": {"include_usage": null}` (bool 不正 → 400)
    NullField,
}

fn apply_so(body: &mut Value, so: So) {
    match so {
        So::Omitted => {}
        So::EmptyObject => body["stream_options"] = json!({}),
        So::False => body["stream_options"] = json!({"include_usage": false}),
        So::True => body["stream_options"] = json!({"include_usage": true}),
        So::NullObject => body["stream_options"] = Value::Null,
        So::NullField => body["stream_options"] = json!({"include_usage": null}),
    }
}

const NO_USAGE_CASES: [So; 4] = [So::Omitted, So::EmptyObject, So::False, So::NullObject];

fn assert_null_field_400(resp: &Resp) {
    assert_eq!(
        resp.status,
        StatusCode::BAD_REQUEST,
        "include_usage: null が 400 でない: {} {}",
        resp.status,
        resp.text
    );
    assert!(
        !resp.ctype.starts_with("text/event-stream"),
        "400 なのに SSE を開始している: {}",
        resp.ctype
    );
    let v: Value = serde_json::from_str(&resp.text).expect("エラー本文が JSON でない");
    let msg = v["error"]["message"]
        .as_str()
        .unwrap_or_else(|| panic!("error.message が文字列でない: {v}"));
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    assert!(
        msg.contains("null") && msg.contains("boolean"),
        "include_usage の型エラーを示す本文でない: {msg}"
    );
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_usage_state_matrix() {
    let app = test_app();
    let (_, base) = {
        let r = call(&app, "/v1/chat/completions", &chat_body(false)).await;
        assert_eq!(r.status, StatusCode::OK, "{}", r.text);
        let v: Value = serde_json::from_str(&r.text).unwrap();
        (r.status, v)
    };
    let baseline = assert_usage_complete(&base["usage"]);

    for so in NO_USAGE_CASES {
        let mut body = chat_body(true);
        apply_so(&mut body, so);
        let r = call(&app, "/v1/chat/completions", &body).await;
        assert_eq!(r.status, StatusCode::OK, "so = {body}");
        assert!(
            r.ctype.starts_with("text/event-stream"),
            "content-type が SSE でない: {}",
            r.ctype
        );
        let chunks = sse_chunks(&r.text);
        assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
        assert_no_usage_at_all(&r.text);
        assert!(
            chunks.iter().all(|c| c.get("usage").is_none()),
            "usage キーがある: {chunks:?}"
        );
        assert!(r.text.trim_end().ends_with("data: [DONE]"), "{}", r.text);
    }

    // 明示 false と未指定は同じバイト列 (id/created を除く)。
    let mut b1 = chat_body(true);
    apply_so(&mut b1, So::Omitted);
    let mut b2 = chat_body(true);
    apply_so(&mut b2, So::False);
    let (r1, r2) = (
        call(&app, "/v1/chat/completions", &b1).await,
        call(&app, "/v1/chat/completions", &b2).await,
    );
    assert_eq!(r1.status, StatusCode::OK);
    assert_eq!(r2.status, StatusCode::OK);
    assert_eq!(
        normalized(sse_chunks(&r1.text)),
        normalized(sse_chunks(&r2.text)),
        "include_usage: false が未指定と同形でない"
    );

    // include_usage: null は 400。
    let mut body = chat_body(true);
    apply_so(&mut body, So::NullField);
    assert_null_field_400(&call(&app, "/v1/chat/completions", &body).await);

    // include_usage: true は末尾 usage chunk。値は非ストリームと一致。
    let mut body = chat_body(true);
    apply_so(&mut body, So::True);
    let r = call(&app, "/v1/chat/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK);
    let chunks = sse_chunks(&r.text);
    assert_null_slots_except_last(&chunks);
    let usage = assert_usage_chunk_is_last(&chunks, &r.text);
    assert_eq!(usage["object"], "chat.completion.chunk");
    assert_eq!(usage["id"], chunks[0]["id"]);
    assert_eq!(assert_usage_complete(&usage["usage"]), baseline);
    assert_ne!(baseline, (0, 0), "usage が 0 のまま: {baseline:?}");
    let last_choice = chunks[..chunks.len() - 1]
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last_choice["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_non_stream_ignores_include_usage() {
    // 非ストリーム + include_usage: true でも通常の chat.completion 1 個。
    let mut body = chat_body(false);
    body["stream_options"] = json!({"include_usage": true});
    let r = call(&test_app(), "/v1/chat/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(
        !r.ctype.starts_with("text/event-stream"),
        "非ストリームが SSE になった: {}",
        r.ctype
    );
    let v: Value = serde_json::from_str(&r.text).expect("JSON でない");
    assert_eq!(v["object"], "chat.completion");
    assert!(v.get("usage").is_some(), "usage が無い: {v}");
    let (p, c) = assert_usage_complete(&v["usage"]);
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");
    // `stream_options` 自体が不正 (null フィールド) なら非ストリームでも 400。
    let mut bad = chat_body(false);
    bad["stream_options"] = json!({"include_usage": null});
    assert_null_field_400(&call(&test_app(), "/v1/chat/completions", &bad).await);
}

// ---------------------------------------------------------------------------
// /v1/completions (legacy)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_stream_usage_state_matrix() {
    let app = test_app();
    let r = call(&app, "/v1/completions", &completion_body(false)).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    let base: Value = serde_json::from_str(&r.text).unwrap();
    let baseline = assert_usage_complete(&base["usage"]);
    // 4 トークン上限なので stop ではなく length で切れることもある。
    // 非ストリームと同じ値であることを基準にする。
    let baseline_finish = base["choices"][0]["finish_reason"].clone();

    for so in NO_USAGE_CASES {
        let mut body = completion_body(true);
        apply_so(&mut body, so);
        let r = call(&app, "/v1/completions", &body).await;
        assert_eq!(r.status, StatusCode::OK, "so = {body}");
        assert!(r.ctype.starts_with("text/event-stream"), "{}", r.ctype);
        assert_no_usage_at_all(&r.text);
        let chunks = sse_chunks(&r.text);
        assert!(!chunks.is_empty(), "chunk が無い");
        assert!(
            chunks.iter().all(|c| c.get("usage").is_none()),
            "usage キーがある: {chunks:?}"
        );
        for c in &chunks {
            assert_eq!(c["object"], "text_completion");
        }
        assert!(r.text.trim_end().ends_with("data: [DONE]"), "{}", r.text);
    }

    let mut body = completion_body(true);
    apply_so(&mut body, So::NullField);
    assert_null_field_400(&call(&app, "/v1/completions", &body).await);

    let mut body = completion_body(true);
    apply_so(&mut body, So::True);
    let r = call(&app, "/v1/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK);
    let chunks = sse_chunks(&r.text);
    assert_null_slots_except_last(&chunks);
    let usage = assert_usage_chunk_is_last(&chunks, &r.text);
    assert_eq!(usage["object"], "text_completion");
    assert_eq!(usage["id"], chunks[0]["id"]);
    assert_eq!(assert_usage_complete(&usage["usage"]), baseline);
    assert_ne!(baseline, (0, 0), "usage が 0 のまま: {baseline:?}");
    // 最終 choices chunk に finish_reason がある。
    let last_choice = chunks[..chunks.len() - 1]
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(
        last_choice["choices"][0]["finish_reason"], baseline_finish,
        "finish_reason が非ストリームと違う"
    );
}

#[tokio::test]
async fn completions_non_stream_ignores_include_usage() {
    let mut body = completion_body(false);
    body["stream_options"] = json!({"include_usage": true});
    let r = call(&test_app(), "/v1/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK);
    assert!(!r.ctype.starts_with("text/event-stream"), "{}", r.ctype);
    let v: Value = serde_json::from_str(&r.text).expect("JSON でない");
    assert_eq!(v["object"], "text_completion");
    assert!(v.get("usage").is_some(), "usage が無い: {v}");
    let (p, c) = assert_usage_complete(&v["usage"]);
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");

    let mut bad = completion_body(false);
    bad["stream_options"] = json!({"include_usage": null});
    assert_null_field_400(&call(&test_app(), "/v1/completions", &bad).await);
}

// ---------------------------------------------------------------------------
// tool-call SSE ストリーム
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_stream_usage_state_matrix() {
    // usage 無しの 4 状態: tool_calls で終わり usage フィールドが無い。
    for so in NO_USAGE_CASES {
        let app = tool_app("get_weather");
        let mut body = tool_body(true, None);
        apply_so(&mut body, so);
        let r = call(&app, "/v1/chat/completions", &body).await;
        assert_eq!(r.status, StatusCode::OK, "{body}");
        assert_no_usage_at_all(&r.text);
        let chunks = sse_chunks(&r.text);
        assert!(
            chunks
                .iter()
                .any(|c| c["choices"][0]["delta"]["tool_calls"].is_array()),
            "tool_calls delta が無い: {chunks:?}"
        );
        let last_choice = chunks
            .iter()
            .rev()
            .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
            .expect("choices 付き chunk が無い");
        assert_eq!(last_choice["choices"][0]["finish_reason"], "tool_calls");
    }

    // include_usage: null → 400。
    let app = tool_app("get_weather");
    let body = tool_body(true, Some(json!({"include_usage": null})));
    assert_null_field_400(&call(&app, "/v1/chat/completions", &body).await);

    // include_usage: true → tool_calls の後に usage chunk が 1 個。
    let app = tool_app("get_weather");
    let body = tool_body(true, Some(json!({"include_usage": true})));
    let r = call(&app, "/v1/chat/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK);
    let chunks = sse_chunks(&r.text);
    assert!(
        chunks
            .iter()
            .any(|c| c["choices"][0]["delta"]["tool_calls"].is_array()),
        "tool_calls delta が無い: {chunks:?}"
    );
    assert_null_slots_except_last(&chunks);
    let usage = assert_usage_chunk_is_last(&chunks, &r.text);
    assert_eq!(usage["object"], "chat.completion.chunk");
    let (p, c) = assert_usage_complete(&usage["usage"]);
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");
    let last_choice = chunks[..chunks.len() - 1]
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last_choice["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn tool_stream_usage_matches_non_stream_tool_usage() {
    // 同じ script の新しい Engine 同士なので生成列が同じ。usage が一致する。
    let nonstream = call(
        &tool_app("get_weather"),
        "/v1/chat/completions",
        &tool_body(false, None),
    )
    .await;
    assert_eq!(nonstream.status, StatusCode::OK, "{}", nonstream.text);
    let nv: Value = serde_json::from_str(&nonstream.text).unwrap();
    assert_eq!(nv["choices"][0]["finish_reason"], "tool_calls");
    assert!(
        nv["choices"][0]["message"]["tool_calls"].is_array(),
        "非ストリームに tool_calls が無い: {nv}"
    );
    let baseline = assert_usage_complete(&nv["usage"]);

    let stream = call(
        &tool_app("get_weather"),
        "/v1/chat/completions",
        &tool_body(true, Some(json!({"include_usage": true}))),
    )
    .await;
    assert_eq!(stream.status, StatusCode::OK, "{}", stream.text);
    let chunks = sse_chunks(&stream.text);
    let usage = assert_usage_chunk_is_last(&chunks, &stream.text);
    let streamed = assert_usage_complete(&usage["usage"]);
    assert_eq!(
        streamed, baseline,
        "tool ストリームと非ストリームの usage が違う (stream = {streamed:?}, non-stream = {baseline:?})"
    );
    assert_ne!(baseline, (0, 0), "usage が 0 のまま: {baseline:?}");
}
