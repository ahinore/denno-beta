//! task 911: リクエスト ID と計測ヘッダの統合テスト。
//!
//! - クライアントの `x-request-id` をそのまま返す (Anthropic / OpenAI /
//!   Ollama / 制御エンドポイント / SSE の全部)
//! - 無害化 (印字可能 ASCII 以外を捨てる・128 byte で切る)
//! - 未指定なら生成 ID (プロセス内で一意・単調増加)
//! - 完了した非ストリーミング生成応答に `x-denno-ttft-ms` /
//!   `x-denno-decode-tps` が付く (ストリーミングと非生成応答には付かない)
//! - JSON アクセスログ行に `request_id` が入る

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
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

/// 応答のヘッダと本文 (テキスト) を返す。
async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap()
}

fn post_json(uri: &str, body: &Value, rid: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(rid) = rid {
        b = b.header("x-request-id", rid);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

fn rid(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn header_f64(headers: &HeaderMap, name: &str) -> Option<f64> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
}

fn messages_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

// ---------------------------------------------------------------------------
// 1. 送った ID をそのまま返す (全系統)

#[tokio::test]
async fn echoes_custom_request_id_on_all_protocols() {
    let app = app();
    let cases: Vec<(&str, Request<Body>)> = vec![
        ("POST /v1/messages", post_json("/v1/messages", &messages_body(false), Some("cli-abc-123"))),
        (
            "POST /v1/chat/completions",
            post_json("/v1/chat/completions", &chat_body(false), Some("cli-abc-123")),
        ),
        (
            "POST /v1/completions",
            post_json("/v1/completions", &json!({"model": "denno-test", "prompt": "hi", "max_tokens": 8}), Some("cli-abc-123")),
        ),
        (
            "POST /api/chat",
            post_json(
                "/api/chat",
                &json!({"model": "denno-test", "messages": [{"role": "user", "content": "hi"}], "stream": false, "options": {"num_predict": 8}}),
                Some("cli-abc-123"),
            ),
        ),
        (
            "POST /api/generate",
            post_json(
                "/api/generate",
                &json!({"model": "denno-test", "prompt": "hi", "stream": false, "options": {"num_predict": 8}}),
                Some("cli-abc-123"),
            ),
        ),
        ("GET /health", {
            let mut r = get("/health");
            r.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_static("cli-abc-123"),
            );
            r
        }),
        ("GET /v1/models", {
            let mut r = get("/v1/models");
            r.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_static("cli-abc-123"),
            );
            r
        }),
    ];
    for (label, req) in cases {
        let (status, headers, body) = send(&app, req).await;
        assert!(status.is_success(), "{label}: status = {status}, body = {body}");
        assert_eq!(rid(&headers).as_deref(), Some("cli-abc-123"), "{label}");
    }
}

/// SSE ストリーミングの応答ヘッダにも ID が付く (本文は流れる前のヘッダで見る)。
#[tokio::test]
async fn echoes_custom_request_id_on_sse() {
    let app = app();
    let req = post_json("/v1/messages", &messages_body(true), Some("sse-42"));
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(rid(resp.headers()).as_deref(), Some("sse-42"));
    // SSE はストリームなので計測ヘッダは付けない。
    assert!(!resp.headers().contains_key("x-denno-ttft-ms"));
    assert!(!resp.headers().contains_key("x-denno-decode-tps"));
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes);
    assert!(text.contains("message_start") || text.contains("content_block"), "{text}");
}

// ---------------------------------------------------------------------------
// 2. 無害化

#[tokio::test]
async fn sanitizes_non_ascii_and_control_bytes() {
    let app = app();
    // HeaderValue が通す範囲では TAB / SPACE / obs-text (0xff) が混じり得る。
    // 印字可能 ASCII だけが残る (先頭の `cli`、`id`、`X` は残す)。
    let raw = HeaderValue::from_bytes(b"cli\xff\tid X").unwrap();
    let mut req = get("/health");
    req.headers_mut().insert("x-request-id", raw);
    let (status, headers, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(rid(&headers).as_deref(), Some("cliidX"));
}

#[tokio::test]
async fn truncates_request_id_to_128_bytes() {
    let app = app();
    let long = "a".repeat(300);
    let mut req = get("/health");
    req.headers_mut()
        .insert("x-request-id", HeaderValue::from_str(&long).unwrap());
    let (status, headers, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK);
    let got = rid(&headers).expect("x-request-id が無い");
    assert_eq!(got.len(), denno_server::MAX_REQUEST_ID_LEN);
    assert_eq!(got, "a".repeat(denno_server::MAX_REQUEST_ID_LEN));
}

#[tokio::test]
async fn unusable_request_id_is_replaced_by_generated_one() {
    let app = app();
    // 非表示 byte (obs-text) だけ → 有効な文字が残らないので生成 ID になる。
    let raw = HeaderValue::from_bytes(b"\xff\xfe").unwrap();
    let mut req = get("/health");
    req.headers_mut().insert("x-request-id", raw);
    let (_, headers, _) = send(&app, req).await;
    let got = rid(&headers).expect("x-request-id が無い");
    assert!(got.starts_with("req-"), "{got}");
}

// ---------------------------------------------------------------------------
// 3. 生成 ID

#[tokio::test]
async fn generates_unique_monotonic_ids_when_absent() {
    let app = app();
    let (_, h1, _) = send(&app, get("/health")).await;
    let (_, h2, _) = send(&app, get("/health")).await;
    let (_, h3, _) = send(&app, get("/v1/models")).await;
    let ids: Vec<String> = [h1, h2, h3]
        .iter()
        .map(|h| rid(h).expect("x-request-id が無い"))
        .collect();
    for id in &ids {
        assert!(id.starts_with("req-"), "{id}");
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit() || c == '-' || c == 'r' || c == 'e' || c == 'q'),
            "ID が ASCII hex でない: {id}"
        );
    }
    assert_ne!(ids[0], ids[1]);
    assert_ne!(ids[1], ids[2]);
    // 単調増加 (連番が末尾にある)。
    assert!(ids[0] < ids[1], "{} < {}", ids[0], ids[1]);
    assert!(ids[1] < ids[2], "{} < {}", ids[1], ids[2]);
}

#[tokio::test]
async fn sanitize_and_generate_helpers_are_consistent() {
    assert_eq!(denno_server::sanitize_request_id("abc-123").as_deref(), Some("abc-123"));
    assert_eq!(denno_server::sanitize_request_id("a\x01b").as_deref(), Some("ab"));
    assert_eq!(denno_server::sanitize_request_id("\u{3042}\u{3044}"), None);
    assert_eq!(denno_server::sanitize_request_id(""), None);
    let a = denno_server::generate_request_id();
    let b = denno_server::generate_request_id();
    assert_ne!(a, b);
    assert!(a < b);
}

// ---------------------------------------------------------------------------
// 4. 計測ヘッダ (完了した非ストリーミング生成応答)

#[tokio::test]
async fn timing_headers_on_non_streaming_generations() {
    let app = app();
    for (label, req) in [
        ("/v1/messages", post_json("/v1/messages", &messages_body(false), None)),
        ("/v1/chat/completions", post_json("/v1/chat/completions", &chat_body(false), None)),
        (
            "/v1/completions",
            post_json("/v1/completions", &json!({"model": "denno-test", "prompt": "hi", "max_tokens": 8}), None),
        ),
        (
            "/api/chat",
            post_json(
                "/api/chat",
                &json!({"model": "denno-test", "messages": [{"role": "user", "content": "hi"}], "stream": false, "options": {"num_predict": 8}}),
                None,
            ),
        ),
        (
            "/api/generate",
            post_json(
                "/api/generate",
                &json!({"model": "denno-test", "prompt": "hi", "stream": false, "options": {"num_predict": 8}}),
                None,
            ),
        ),
    ] {
        let (status, headers, body) = send(&app, req).await;
        assert!(status.is_success(), "{label}: {status} {body}");
        let ttft = header_f64(&headers, "x-denno-ttft-ms")
            .unwrap_or_else(|| panic!("{label}: x-denno-ttft-ms が無い"));
        let tps = header_f64(&headers, "x-denno-decode-tps")
            .unwrap_or_else(|| panic!("{label}: x-denno-decode-tps が無い"));
        assert!(ttft >= 0.0, "{label}: ttft = {ttft}");
        assert!(tps >= 0.0, "{label}: tps = {tps}");
        assert!(tps > 0.0, "{label}: 出力トークンがあるのに tps = 0");
        // 応答本文は計測ヘッダのために書き換わらない (JSON として読める)。
        let v: Value = serde_json::from_str(&body).expect("本文が JSON ではない");
        assert!(v["usage"].is_object() || v["eval_count"].is_number(), "{label}: {body}");
    }
}

#[tokio::test]
async fn no_timing_headers_on_non_generation_or_streaming_responses() {
    let app = app();
    let (_, h, _) = send(&app, get("/health")).await;
    assert!(!h.contains_key("x-denno-ttft-ms"));
    assert!(!h.contains_key("x-denno-decode-tps"));
    assert!(rid(&h).is_some());

    let (_, h, _) = send(&app, get("/v1/models")).await;
    assert!(!h.contains_key("x-denno-ttft-ms"));

    // Ollama のストリーミング (既定) は NDJSON なので計測ヘッダを付けない。
    let (_, h, _) = send(
        &app,
        post_json(
            "/api/generate",
            &json!({"model": "denno-test", "prompt": "hi", "options": {"num_predict": 4}}),
            None,
        ),
    )
    .await;
    assert!(rid(&h).is_some());
    assert!(!h.contains_key("x-denno-ttft-ms"));
}

// ---------------------------------------------------------------------------
// 5. 構造化アクセスログ

#[test]
fn json_access_log_line_carries_request_id() {
    let line = denno_server::access_log_line("POST", "/v1/messages", 200, 3.5, "req-abc");
    let v: Value = serde_json::from_str(&line).unwrap();
    assert_eq!(v["request_id"], "req-abc");
    assert_eq!(v["method"], "POST");
    assert_eq!(v["path"], "/v1/messages");
    assert_eq!(v["status"], 200);
}
