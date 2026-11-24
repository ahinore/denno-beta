//! P793: OpenAI 互換 `POST /v1/chat/completions` の統合テスト。
//!
//! denno-engine のダミートークナイザ + 決定的ダミーモデル (tests/scheduler.rs と
//! 同一方式) を使い、HTTP ワイヤレベル (axum Router + tower oneshot) と
//! 実 TCP ソケットで検証する。カバー項目:
//!
//! - 非ストリーミング応答の形 (id/object/created/model/choices/usage)
//! - `/v1/messages` との数値・本文一致 (同じ Scheduler・テンプレート・KV を通る)
//! - SSE (chat.completion.chunk) の delta 結合 = 非ストリーミング本文、
//!   finish_reason の確定 chunk、`data: [DONE]`
//! - `stream_options.include_usage` の usage chunk
//! - stop (文字列 / 配列) の打ち切り
//! - エラーのステータスコード (400 / 404) と OpenAI 形式の error 本文
//! - クライアント切断で生成が打ち切られ、後続リクエストが通る
//! - `/v1/models` が OpenAI 形式 (object/data[].object) と従来フィールドを併記する

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

/// 決定的ダミーモデル (dummy_tokenizer の語彙に合わせる)。
/// peak = 直前トークン + 1、EOS の位置だけ避ける。
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

fn app_with(delay: Duration, endless: bool) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(SeqModel { delay, endless }),
        tok,
        "denno-test",
        None,
    );
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
    denno_server::router(app_with(Duration::ZERO, false))
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

/// SSE 本文から `data: ...` 行の JSON を取り出す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

/// chunk 列の delta.content を連結する。
fn sse_content(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": stream,
    })
}

/// Anthropic 側の同じ会話 (thinking 既定 = brief を揃える)。
fn messages_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "temperature": 0.0,
    })
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_streaming_shape_and_matches_messages() {
    let app = test_app();
    let (status, oa) = post(&app, "/v1/chat/completions", chat_body(false)).await;
    assert_eq!(status, StatusCode::OK, "body = {oa}");
    assert_eq!(oa["object"], "chat.completion");
    assert!(
        oa["id"].as_str().unwrap().starts_with("chatcmpl-"),
        "id = {}",
        oa["id"]
    );
    assert!(oa["created"].as_u64().unwrap() > 0);
    assert_eq!(oa["model"], "denno-test");
    assert_eq!(oa["choices"][0]["index"], 0);
    assert_eq!(oa["choices"][0]["message"]["role"], "assistant");
    let text = oa["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    assert!(!text.is_empty(), "content が空: {oa}");

    // finish_reason / usage の整合
    let fr = oa["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(fr == "stop" || fr == "length", "finish_reason = {fr}");
    let prompt = oa["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion = oa["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(prompt > 0);
    assert_eq!(
        oa["usage"]["total_tokens"].as_u64().unwrap(),
        prompt + completion
    );

    // /v1/messages と本文・usage が一致する (同じ経路を通っている証拠)
    let (status, an) = post(&app, "/v1/messages", messages_body()).await;
    assert_eq!(status, StatusCode::OK, "body = {an}");
    assert_eq!(an["content"][0]["text"], text);
    assert_eq!(an["usage"]["input_tokens"].as_u64().unwrap(), prompt);
    assert_eq!(an["usage"]["output_tokens"].as_u64().unwrap(), completion);
    if fr == "length" {
        assert_eq!(an["stop_reason"], "max_tokens");
    } else {
        assert_eq!(an["stop_reason"], "end_turn");
    }
}

#[tokio::test]
async fn content_parts_and_developer_role_are_accepted() {
    let app = test_app();
    let body = json!({
        "model": "denno-test",
        "messages": [
            {"role": "developer", "content": [{"type": "text", "text": "dev"}]},
            {"role": "user", "content": [
                {"type": "text", "text": "hel"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                {"type": "text", "text": "lo"},
            ]},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    });
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "chat.completion");
}

#[tokio::test]
async fn streaming_chunks_deltas_and_done() {
    let app = test_app();
    // まず非ストリーミングで本文を取る (同一入力・greedy なので一致するはず)
    let (_, oa) = post(&app, "/v1/chat/completions", chat_body(false)).await;
    let want = oa["choices"][0]["message"]["content"].as_str().unwrap().to_string();

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body(true).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    // OpenAI 形式: event: 行は無く、data: 行だけ + [DONE]
    assert!(!text.contains("event: "), "event: 行が混ざっている:\n{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "終端が [DONE] でない:\n{text}");

    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
    // 全 chunk が chat.completion.chunk で id が同じ
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], chunks[0]["id"]);
    }
    assert!(chunks[0]["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    // delta の結合が非ストリーミング本文と一致する
    assert_eq!(sse_content(&chunks), want);
    // finish_reason は最後の choices 付き chunk だけに入る
    let last_with_choice = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .unwrap();
    assert!(
        last_with_choice["choices"][0]["finish_reason"].is_string(),
        "final chunk に finish_reason が無い: {last_with_choice}"
    );
    let fr = last_with_choice["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(fr == "stop" || fr == "length", "finish_reason = {fr}");
}

#[tokio::test]
async fn streaming_include_usage_adds_usage_chunk() {
    let app = test_app();
    let mut body = chat_body(true);
    body["stream_options"] = json!({"include_usage": true});
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let chunks = sse_chunks(&text);
    let usage_chunk = chunks
        .iter()
        .find(|c| c["usage"].is_object())
        .expect("usage chunk が無い");
    assert_eq!(usage_chunk["choices"], json!([]));
    let p = usage_chunk["usage"]["prompt_tokens"].as_u64().unwrap();
    let c = usage_chunk["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(p > 0);
    assert_eq!(usage_chunk["usage"]["total_tokens"].as_u64().unwrap(), p + c);
}

#[tokio::test]
async fn stop_string_and_array_truncate() {
    let app = test_app();
    let (_, oa) = post(&app, "/v1/chat/completions", chat_body(false)).await;
    let full = oa["choices"][0]["message"]["content"].as_str().unwrap().to_string();
    // 生成本文の最後の語を stop にする (greedy なので再実行しても同じ列)
    let last_word = full
        .split_whitespace()
        .last()
        .expect("生成本文が空")
        .to_string();

    for stop in [json!(last_word), json!([last_word])] {
        let mut body = chat_body(false);
        body["stop"] = stop;
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "body = {v}");
        let text = v["choices"][0]["message"]["content"].as_str().unwrap();
        assert!(
            !text.contains(&last_word),
            "stop 語 {last_word:?} が本文に残っている: {text:?}"
        );
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
    }
}

#[tokio::test]
async fn error_status_codes_are_openai_shaped() {
    let app = test_app();

    // messages 空 → 400
    let mut body = chat_body(false);
    body["messages"] = json!([]);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("messages"));
    assert_eq!(v["error"]["param"], "messages");

    // 未知ロール → 400 (param に位置)
    let mut body = chat_body(false);
    body["messages"] = json!([{"role": "wizard", "content": "x"}]);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "messages[0].role");

    // 未知モデル → 404 相当 (error.type は invalid_request_error)
    let mut body = chat_body(false);
    body["model"] = json!("nope");
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"].as_str().unwrap().contains("nope"));

    // 壊れた JSON → 400
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from("{\"messages\": "))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn max_tokens_gives_length_finish_reason() {
    let app = test_app();
    let mut body = chat_body(false);
    body["max_completion_tokens"] = json!(2);
    body["max_tokens"] = Value::Null;
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 2);
}

#[tokio::test]
async fn models_endpoint_is_openai_and_anthropic_compatible() {
    let app = test_app();
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    // OpenAI 形式
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["object"], "model");
    assert_eq!(v["data"][0]["id"], "denno-test");
    assert!(v["data"][0]["owned_by"].is_string());
    // 従来 (Anthropic クライアント向け) フィールドも残る
    assert_eq!(v["data"][0]["type"], "model");
    assert_eq!(v["data"][0]["display_name"], "denno-test");
    assert_eq!(v["has_more"], false);
    assert!(v["data"][0]["context_length"].as_u64().unwrap() > 0);
}

// ---------- 実 TCP: クライアント切断 ----------

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
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "in_flight が {want} にならない (現在 {})",
        state.scheduler.in_flight()
    );
}

/// 生成中のクライアント切断で Run が打ち切られ、後続リクエストが通る
/// (= raw チャネルの閉鎖が client_gone に拾われている証拠)。
#[tokio::test]
async fn client_disconnect_aborts_openai_stream() {
    let state = app_with(Duration::from_millis(40), true);
    let addr = serve_tcp(state.clone()).await;
    let body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 400,
        "temperature": 0.0,
        "stream": true,
    })
    .to_string();

    let mut stream = post_raw(addr, "/v1/chat/completions", &body).await;
    let mut buf = String::new();
    assert!(
        read_until(&mut stream, &mut buf, "chat.completion.chunk", 10).await,
        "最初の chunk が届かない: {buf}"
    );
    // 最初の chunk は scheduler の admit 前に送られる (handler 側) ため、
    // ワーカーが pull して in_flight が 1 になるのを待ってから切断する。
    wait_in_flight(&state, 1, 5).await;
    drop(stream); // 切断

    wait_in_flight(&state, 0, 5).await;

    // 後続 (非ストリーム) が即座に完了する = ワーカー / KV スロットは健常
    let mut s2 = post_raw(
        addr,
        "/v1/chat/completions",
        &json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "t0"}],
            "max_tokens": 4,
            "temperature": 0.0,
        })
        .to_string(),
    )
    .await;
    let mut buf2 = String::new();
    assert!(
        read_until(&mut s2, &mut buf2, "\"object\":\"chat.completion\"", 20).await,
        "後続リクエストの応答が届かない: {buf2}"
    );
    assert_eq!(state.scheduler.in_flight(), 0);
}
