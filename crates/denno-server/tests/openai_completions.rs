//! P837: OpenAI 旧 Completions (`POST /v1/completions`) の統合テスト。
//!
//! denno-engine のダミートークナイザ + 決定的ダミーモデル (tests/openai_chat.rs と
//! 同一方式) で HTTP ワイヤを検証する。カバー項目:
//!
//! - 非ストリーミング応答の形 (`id: cmpl-...` / `object: text_completion` /
//!   choices / usage)
//! - **chat template を通さない** こと (プロンプト `t0 t1 t2` の続きが
//!   ちょうど `t3 t4 ...` になる = 役割ラベルのトークンが挟まらない)
//! - `prompt` の文字列 / 1 要素配列、複数要素とトークン ID 配列は 400
//! - SSE (`text_completion` chunk / `finish_reason` / `include_usage` / `[DONE]`)
//! - `stop` (文字列 / 配列) の打ち切り、`echo` の前置、`max_tokens` の length
//! - エラーの形 (400 / 404)

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 512;

/// 決定的ダミーモデル。peak = 直前トークン + 1。
struct SeqModel {
    /// 1 forward あたりの sleep (切断テストで生成を長引かせる)
    delay: Duration,
    /// EOS を出さない (max_tokens まで必ず続く)
    endless: bool,
}

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
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
        let peak = (last + 1) % DUMMY_VOCAB;
        let peak = if self.endless && peak == DUMMY_EOS as usize { peak + 1 } else { peak };
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn test_state() -> Arc<AppState> {
    test_state_with(Duration::ZERO, false)
}

fn test_state_with(delay: Duration, endless: bool) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(SeqModel { delay, endless }), tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
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

fn test_app() -> Router {
    denno_server::router(test_state())
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

async fn post_sse(app: &Router, uri: &str, body: Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn sse_text(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["text"].as_str())
        .collect()
}

/// 生成テキストの語列 (デトークナイザの空白の付け方に依存しない比較用)。
fn words(v: &Value) -> Vec<&str> {
    v.as_str()
        .unwrap_or_default()
        .split_whitespace()
        .collect()
}

fn completion_body(prompt: Value, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": stream,
    })
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_streaming_shape_and_raw_prompt() {
    let app = test_app();
    let (status, v) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "text_completion");
    assert!(
        v["id"].as_str().unwrap().starts_with("cmpl-"),
        "id = {}",
        v["id"]
    );
    assert_eq!(v["id"].as_str().unwrap().len(), "cmpl-".len() + 24);
    assert!(v["created"].as_u64().unwrap() > 0);
    assert_eq!(v["model"], "denno-test");
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["logprobs"], Value::Null);
    // chat template を通さない: プロンプトの続きはちょうど t3 t4 t5 t6
    // (テンプレートが挟まれば末尾トークンが変わり、この列にはならない)
    assert_eq!(
        words(&v["choices"][0]["text"]),
        ["t3", "t4", "t5", "t6"],
        "raw prompt になっていない: {v}"
    );
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    // usage
    let p = v["usage"]["prompt_tokens"].as_u64().unwrap();
    let c = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(p, 3, "プロンプトは 3 トークン (t0 t1 t2): {v}");
    assert_eq!(c, 4);
    assert_eq!(v["usage"]["total_tokens"].as_u64().unwrap(), p + c);
}

#[tokio::test]
async fn prompt_array_of_one_matches_string() {
    let app = test_app();
    let (_, a) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    let (status, b) = post(
        &app,
        "/v1/completions",
        completion_body(json!(["t0 t1 t2"]), false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {b}");
    assert_eq!(a["choices"][0]["text"], b["choices"][0]["text"]);
    assert_eq!(a["usage"]["prompt_tokens"], b["usage"]["prompt_tokens"]);
}

#[tokio::test]
async fn streaming_chunks_usage_and_done() {
    let app = test_app();
    let (_, full) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    let want = full["choices"][0]["text"].as_str().unwrap().to_string();

    let mut b = completion_body(json!("t0 t1 t2"), true);
    b["stream_options"] = json!({"include_usage": true});
    let (status, text) = post_sse(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!text.contains("event: "), "event: 行が混ざっている:\n{text}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない:\n{text}"
    );
    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 3, "chunk が少なすぎる: {chunks:?}");
    for c in &chunks {
        assert_eq!(c["object"], "text_completion");
        assert_eq!(c["id"], chunks[0]["id"]);
    }
    assert!(chunks[0]["id"].as_str().unwrap().starts_with("cmpl-"));

    // text の結合 = 非ストリーミング本文 (finish_reason chunk の空文字を含む)
    assert_eq!(sse_text(&chunks), want);

    // finish_reason は最後の choices 付き chunk にだけ入る
    let last_with_choice = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .unwrap();
    assert!(
        last_with_choice["choices"][0]["finish_reason"].is_string(),
        "final chunk に finish_reason が無い: {last_with_choice}"
    );

    // usage chunk (choices が空)
    let usage_chunk = chunks
        .iter()
        .find(|c| c["usage"].is_object())
        .expect("usage chunk が無い");
    assert_eq!(usage_chunk["choices"], json!([]));
    let p = usage_chunk["usage"]["prompt_tokens"].as_u64().unwrap();
    let c = usage_chunk["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(usage_chunk["usage"]["total_tokens"].as_u64().unwrap(), p + c);

    // include_usage 無しでは usage chunk が出ない
    let (status, text) = post_sse(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), true)).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);
    assert!(
        !chunks.iter().any(|c| c["usage"].is_object()),
        "include_usage 無しで usage が出ている: {chunks:?}"
    );
    assert_eq!(sse_text(&chunks), want);
}

#[tokio::test]
async fn stop_truncates_and_echo_prefixes_prompt() {
    let app = test_app();
    let (_, full) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    let full_text = full["choices"][0]["text"].as_str().unwrap().to_string();
    let last_word = full_text.split_whitespace().last().unwrap().to_string();

    for stop in [json!(last_word), json!([last_word])] {
        let mut b = completion_body(json!("t0 t1 t2"), false);
        b["stop"] = stop;
        let (status, v) = post(&app, "/v1/completions", b).await;
        assert_eq!(status, StatusCode::OK, "body = {v}");
        let text = v["choices"][0]["text"].as_str().unwrap();
        assert!(
            !text.contains(&last_word),
            "stop 語 {last_word:?} が残っている: {text:?}"
        );
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }

    // echo: true はプロンプトを前置する (生成部分は同じ)
    let mut b = completion_body(json!("t0 t1 t2"), false);
    b["echo"] = json!(true);
    let (status, v) = post(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let echoed = v["choices"][0]["text"].as_str().unwrap();
    assert!(
        echoed.starts_with("t0 t1 t2"),
        "echo の前置が無い: {echoed:?}"
    );
    assert_eq!(&echoed["t0 t1 t2".len()..], full_text);

    // echo 未指定なら前置しない
    let (_, v) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    let plain = v["choices"][0]["text"].as_str().unwrap();
    assert!(!plain.starts_with("t0 t1 t2"), "echo 無しで前置された: {plain:?}");

    // ストリームでも echo は最初の chunk に載る
    let mut b = completion_body(json!("t0 t1 t2"), true);
    b["echo"] = json!(true);
    let (status, text) = post_sse(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);
    assert_eq!(chunks[0]["choices"][0]["text"], "t0 t1 t2", "{chunks:?}");
    assert_eq!(sse_text(&chunks), format!("t0 t1 t2{full_text}"));
}

#[tokio::test]
async fn max_tokens_default_and_stop_reason() {
    let app = test_app();
    // max_tokens 未指定は既定 (1024 clamp) ではなく、EOS か max まで。
    // ダミーモデルは t12 が EOS なので 9 トークン (t3..=t11) で止まる。
    let mut b = completion_body(json!("t0 t1 t2"), false);
    b["max_tokens"] = Value::Null;
    let (status, v) = post(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        words(&v["choices"][0]["text"]),
        ["t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11"]
    );
}

#[tokio::test]
async fn errors_are_openai_shaped() {
    let app = test_app();

    // 空プロンプト
    let (status, v) = post(&app, "/v1/completions", completion_body(json!(""), false)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "prompt");

    // 複数プロンプト (batch 非対応)
    let (status, v) = post(
        &app,
        "/v1/completions",
        completion_body(json!(["t0", "t1"]), false),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "prompt");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("batch of 2 prompts"));

    // トークン ID 配列は非対応 (1 要素でも中身が文字列でなければ 400)
    let (status, v) = post(&app, "/v1/completions", completion_body(json!([1]), false)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "prompt[0]");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("expected a string"));
    // prompt 自体が数値 (トークン ID 列の直書き) も 400
    let (status, v) = post(&app, "/v1/completions", completion_body(json!(3), false)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "prompt");

    // prompt 欠落
    let (status, v) = post(
        &app,
        "/v1/completions",
        json!({"model": "denno-test", "max_tokens": 4}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["type"], "invalid_request_error");

    // 未知モデル
    let mut b = completion_body(json!("t0 t1 t2"), false);
    b["model"] = json!("nope");
    let (status, v) = post(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("nope"));

    // 壊れた JSON
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("{\"prompt\": "))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // stop の型不正 (数値)
    let mut b = completion_body(json!("t0 t1 t2"), false);
    b["stop"] = json!(3);
    let (status, _) = post(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_endpoint_still_uses_template() {
    // 同じ語彙で chat 経路はテンプレートを通す (completion との差の固定)
    let app = test_app();
    let (status, v) = post(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "t0 t1 t2"}],
            "max_tokens": 4,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let chat_text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        !chat_text.starts_with(" t3"),
        "chat 経路が raw prompt になっている: {chat_text:?}"
    );
}

/// 切断耐性の簡易確認 (completion の SSE でもクライアント切断で生成が止まり、
/// 後続リクエストが通る)。openai_chat.rs と同じ実 TCP 経路。
#[tokio::test]
async fn client_disconnect_does_not_wedge_worker() {
    let state = test_state_with(Duration::from_millis(40), true);
    let app = denno_server::router(state.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = app.clone();
    tokio::spawn(async move {
        axum::serve(listener, served).await.unwrap();
    });

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let b = json!({
        "model": "denno-test",
        "prompt": "t0 t1 t2",
        "max_tokens": 400,
        "temperature": 0.0,
        "stream": true,
    })
    .to_string();
    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        b.len(),
        b
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s.flush().await.unwrap();
    // HTTP ヘッダを跨いで最初の chunk が届くまで読む
    let mut buf = String::new();
    let mut tmp = [0u8; 4096];
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while !buf.contains("text_completion") {
        assert!(
            tokio::time::Instant::now() < deadline,
            "最初の chunk が completion 形式でない: {buf}"
        );
        match tokio::time::timeout(Duration::from_secs(5), s.read(&mut tmp)).await {
            Ok(Ok(n)) if n > 0 => buf.push_str(&String::from_utf8_lossy(&tmp[..n])),
            other => panic!("最初の chunk を読めない: {other:?} / {buf}"),
        }
    }
    wait_in_flight(&state, 1, 5).await;
    drop(s); // 切断

    wait_in_flight(&state, 0, 5).await;

    // 後続 (非ストリーム) が通る = ワーカー / KV スロットは健常
    let (status, v) = post(&app, "/v1/completions", completion_body(json!("t0"), false)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
}

async fn wait_in_flight(state: &AppState, want: usize, secs: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    while tokio::time::Instant::now() < deadline {
        if state.scheduler.in_flight() == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "in_flight が {want} にならない (現在 {})",
        state.scheduler.in_flight()
    );
}
