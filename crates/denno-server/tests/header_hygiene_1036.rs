//! task 1036: HTTP ヘッダ衛生 (キャッシュ / MIME スニッフィング / SSE / charset)。
//!
//! `header_hygiene` 層 (`denno_server::INFERENCE_PATHS` / `is_inference_path`) が
//! 全応答に付けるヘッダを固定する。
//!
//!   - 全応答: `x-content-type-options: nosniff`
//!   - 推論系 (`/v1/messages` 等) と `/metrics`: `cache-control: no-store`
//!   - SSE: `cache-control: no-store` / `x-accel-buffering: no`
//!     (content-type は `text/event-stream` のまま。`drain_edge_1274` が完全一致で固定)
//!   - 2xx の JSON / text: `charset=utf-8` 付きの content-type
//!   - エラー封筒 (404 / 405 / 429) の content-type は既存の完全一致テストに
//!     合わせて `application/json` のまま (charset を足さない)

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::{RateLimitConfig, RateLimiter};
use denno_server::{
    is_inference_path, AppState, CorsConfig, EngineInfo, Scheduler, INFERENCE_PATHS,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn state() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
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

fn app() -> Router {
    denno_server::router(state())
}

/// バースト 2・回復 1 rpm (テスト中は実質回復しない) のレート制限付きルーター。
fn app_rate_limited() -> Router {
    let limiter = RateLimiter::from_config(RateLimitConfig::new(1, Some(2)));
    denno_server::router_with_rate_limit(state(), CorsConfig::default(), limiter)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn hdr(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn ct(headers: &HeaderMap) -> String {
    hdr(headers, "content-type").unwrap_or_default()
}

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn post_json(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn anthropic_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
    })
}

fn openai_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": "hello"}],
    })
}

/// nosniff は全応答に付く。
fn assert_nosniff(headers: &HeaderMap, what: &str) {
    assert_eq!(
        hdr(headers, "x-content-type-options").as_deref(),
        Some("nosniff"),
        "{what}: x-content-type-options"
    );
}

// ---------------------------------------------------------------------------
// パスの定義
// ---------------------------------------------------------------------------

#[test]
fn inference_paths_match_the_registered_routes() {
    for p in INFERENCE_PATHS {
        assert!(is_inference_path(p), "{p} が推論扱いでない");
    }
    for p in ["/health", "/ready", "/metrics", "/v1/models", "/v1/messages/x"] {
        assert!(!is_inference_path(p), "{p} を推論扱いにしてはいけない");
    }
}

// ---------------------------------------------------------------------------
// 非ストリーミング推論の JSON 応答
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_json_response_has_no_store_and_charset() {
    let (s, h, body) = send(&app(), post_json("/v1/messages", &anthropic_body(false))).await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(ct(&h), "application/json; charset=utf-8", "content-type");
    assert_eq!(hdr(&h, "cache-control").as_deref(), Some("no-store"));
    assert_nosniff(&h, "/v1/messages");
    assert!(hdr(&h, "x-request-id").is_some(), "x-request-id が無い");
    let v: Value = serde_json::from_str(&body).expect("JSON でない");
    assert!(v.get("content").is_some(), "{body}");
}

#[tokio::test]
async fn openai_chat_json_response_has_no_store_and_charset() {
    let (s, h, body) = send(
        &app(),
        post_json("/v1/chat/completions", &openai_body(false)),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(ct(&h), "application/json; charset=utf-8", "content-type");
    assert_eq!(hdr(&h, "cache-control").as_deref(), Some("no-store"));
    assert_nosniff(&h, "/v1/chat/completions");
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_sse_has_clean_stream_headers() {
    let (s, h, body) = send(&app(), post_json("/v1/messages", &anthropic_body(true))).await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(ct(&h), "text/event-stream", "content-type");
    assert_eq!(hdr(&h, "cache-control").as_deref(), Some("no-store"));
    assert_eq!(hdr(&h, "x-accel-buffering").as_deref(), Some("no"));
    assert_nosniff(&h, "/v1/messages (stream)");
    // ヘッダを付けてもイベント列は壊れていない。
    assert!(
        body.contains("event: message_start"),
        "SSE 本文に message_start が無い: {body}"
    );
}

#[tokio::test]
async fn openai_sse_has_clean_stream_headers() {
    let (s, h, body) = send(
        &app(),
        post_json("/v1/chat/completions", &openai_body(true)),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{body}");
    assert_eq!(ct(&h), "text/event-stream", "content-type");
    assert_eq!(hdr(&h, "cache-control").as_deref(), Some("no-store"));
    assert_eq!(hdr(&h, "x-accel-buffering").as_deref(), Some("no"));
    assert_nosniff(&h, "/v1/chat/completions (stream)");
    assert!(body.contains("data: "), "SSE 本文に data 行が無い: {body}");
}

/// SSE に 1 度も no-cache (axum の既定) を残さない。
#[tokio::test]
async fn sse_cache_control_is_no_store_not_no_cache() {
    let (_, h, _) = send(&app(), post_json("/v1/messages", &anthropic_body(true))).await;
    let cc = hdr(&h, "cache-control").unwrap_or_default();
    assert_eq!(cc, "no-store", "cache-control = {cc}");
}

// ---------------------------------------------------------------------------
// 運用エンドポイント
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metrics_has_no_store_and_charset() {
    let (s, h, body) = send(&app(), get("/metrics")).await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        body.contains("denno_http_requests_total"),
        "/metrics の本文が想定外"
    );
    assert_eq!(
        ct(&h),
        "text/plain; version=0.0.4; charset=utf-8",
        "content-type"
    );
    assert_eq!(hdr(&h, "cache-control").as_deref(), Some("no-store"));
    assert_nosniff(&h, "/metrics");
}

#[tokio::test]
async fn health_gets_nosniff_and_charset_without_forcing_no_store() {
    let (s, h, _) = send(&app(), get("/health")).await;
    assert_eq!(s, StatusCode::OK);
    assert_nosniff(&h, "/health");
    assert_eq!(ct(&h), "application/json; charset=utf-8", "content-type");
    // 推論でも /metrics でもないので no-store は付けない。
    assert_eq!(hdr(&h, "cache-control"), None, "/health に cache-control");
}

#[tokio::test]
async fn models_list_gets_nosniff_and_charset() {
    let (s, h, _) = send(&app(), get("/v1/models")).await;
    assert_eq!(s, StatusCode::OK);
    assert_nosniff(&h, "/v1/models");
    assert_eq!(ct(&h), "application/json; charset=utf-8", "content-type");
}

// ---------------------------------------------------------------------------
// エラー応答 (既存の完全一致テストを壊さない)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn error_envelopes_keep_bare_json_media_type() {
    for (req, want) in [
        (
            get("/definitely-not-a-route"),
            StatusCode::NOT_FOUND,
        ),
        (
            Request::builder()
                .method("DELETE")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
            StatusCode::METHOD_NOT_ALLOWED,
        ),
    ] {
        let (s, h, body) = send(&app(), req).await;
        assert_eq!(s, want, "{body}");
        assert_eq!(ct(&h), "application/json", "content-type は charset 無し");
        assert_nosniff(&h, "error envelope");
        assert_eq!(hdr(&h, "cache-control"), None, "エラーに cache-control");
    }
}

#[tokio::test]
async fn rate_limited_429_still_carries_hygiene_headers() {
    let app = app_rate_limited();
    for i in 0..2 {
        let (s, _, _) = send(&app, post_json("/v1/messages", &anthropic_body(false))).await;
        assert_eq!(s, StatusCode::OK, "{i} 回目は通る");
    }
    let (s, h, body) = send(&app, post_json("/v1/messages", &anthropic_body(false))).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{body}");
    assert_nosniff(&h, "429");
    // 推論パスなので no-store は付く。
    assert_eq!(hdr(&h, "cache-control").as_deref(), Some("no-store"));
    // 封筒の content-type は既存テストどおり charset 無し。
    assert_eq!(ct(&h), "application/json", "429 の content-type");
}
