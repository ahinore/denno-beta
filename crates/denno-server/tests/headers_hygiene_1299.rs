//! task 1299: denno-server の応答ヘッダ衛生の契約テスト。
//!
//! 対象は「どの経路でも同じ規則でヘッダが付くか」:
//!
//! - `content-type` は経路ごとに 1 個だけ・具体的な型 (JSON / SSE / NDJSON /
//!   Prometheus text)。HTML 等の sniff 依存の型は出さない。
//! - `x-request-id` は全応答 (制御 / JSON / SSE / NDJSON / エラー) に 1 個だけ。
//! - `cache-control` は「キャッシュ可能」を示さない。SSE は axum の
//!   `Sse` が `no-cache` を付けるので、それを上書き・重複させない。
//! - `x-content-type-options` を出すなら `nosniff` だけ (別の値を混ぜない)。
//! - `x-denno-ttft-ms` / `x-denno-decode-tps` は完了した非ストリーミング
//!   JSON 生成応答にだけ 1 個ずつ。
//!
//! 実装 (`src/lib.rs` の `log_requests` / `add_timing_headers` /
//! `cors_middleware` と axum 0.7 の `Sse`) に合わせた契約で、新しいヘッダを
//! 要求はしない。ヘッダが増減したときに黙って壊れないための回帰網。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::{RateLimitConfig, RateLimiter};
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// fixture

fn state_with(api_key: Option<&str>, ready: bool) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = if ready {
        Scheduler::spawn(engine.clone(), 8)
    } else {
        Scheduler::spawn_not_ready(engine.clone(), 8)
    };
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
    Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

fn app() -> Router {
    denno_server::router(state_with(None, true))
}

fn app_auth() -> Router {
    denno_server::router(state_with(Some("secret-key"), true))
}

fn app_not_ready() -> Router {
    denno_server::router(state_with(None, false))
}

fn app_small_body() -> Router {
    denno_server::router_with_body_limit(
        state_with(None, true),
        CorsConfig::default(),
        RateLimiter::disabled(),
        64,
    )
}

fn app_rate_limited() -> Router {
    denno_server::router_with_rate_limit(
        state_with(None, true),
        CorsConfig::default(),
        RateLimiter::from_config(RateLimitConfig::new(1, Some(1))),
    )
}

// ---------------------------------------------------------------------------
// リクエスト組み立て

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn get_rid(uri: &str, rid: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("x-request-id", rid)
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

fn post_json_rid(uri: &str, body: &Value, rid: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .header("x-request-id", rid)
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn messages_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn ollama_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "prompt": "hi",
        "stream": stream,
        "options": {"num_predict": 4},
    })
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, Vec<u8>) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, bytes.to_vec())
}

// ---------------------------------------------------------------------------
// ヘッダ検査ヘルパ

/// 指定ヘッダの値を全部返す (大文字小文字は HeaderMap が吸収する)。
fn all<'a>(headers: &'a HeaderMap, name: &str) -> Vec<&'a str> {
    headers
        .get_all(name)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .collect()
}

/// 1 個だけ存在することを確かめて値を返す。
fn one<'a>(headers: &'a HeaderMap, name: &str, label: &str) -> &'a str {
    let vals = all(headers, name);
    assert_eq!(vals.len(), 1, "{label}: {name} が {} 個ある: {vals:?}", vals.len());
    vals[0]
}

fn assert_no_duplicate_headers(headers: &HeaderMap, label: &str) {
    for name in [
        "content-type",
        "x-request-id",
        "cache-control",
        "x-content-type-options",
        "content-length",
        "x-denno-ttft-ms",
        "x-denno-decode-tps",
    ] {
        let n = headers.get_all(name).iter().count();
        assert!(n <= 1, "{label}: {name} が {n} 個重複している");
    }
}

/// キャッシュ可能を示すディレクティブが無いこと。
///
/// `no-store` / `no-cache` / `private` は許可する (むしろ望ましい)。
/// `public` / `max-age` / `s-maxage` / `immutable` は「共有キャッシュに
/// 残してよい」という意味なので、推論 API の応答には出さない契約。
fn assert_not_cacheable(headers: &HeaderMap, label: &str) {
    for v in all(headers, "cache-control") {
        let lc = v.to_ascii_lowercase();
        for bad in ["public", "max-age", "s-maxage", "immutable", "stale-while-revalidate"] {
            assert!(
                !lc.contains(bad),
                "{label}: キャッシュ可能な cache-control = {lc:?}"
            );
        }
    }
}

/// `x-content-type-options` を出すなら `nosniff` だけ。
fn assert_sniff_policy(headers: &HeaderMap, label: &str) {
    for v in all(headers, "x-content-type-options") {
        assert_eq!(v, "nosniff", "{label}: x-content-type-options は nosniff のみ");
    }
}

/// content-type が具体的で、sniff に頼らない型であること。
fn assert_concrete_content_type(headers: &HeaderMap, label: &str) -> String {
    let ct = one(headers, "content-type", label).to_string();
    let lc = ct.to_ascii_lowercase();
    assert!(
        lc.starts_with("application/json")
            || lc.starts_with("text/event-stream")
            || lc.starts_with("application/x-ndjson")
            || lc.starts_with("text/plain"),
        "{label}: 想定外の content-type = {ct:?}"
    );
    assert!(!lc.contains("text/html"), "{label}: HTML を返していない: {ct:?}");
    ct
}

fn assert_no_timing(headers: &HeaderMap, label: &str) {
    assert!(
        headers.get("x-denno-ttft-ms").is_none(),
        "{label}: ストリーミング/非生成に x-denno-ttft-ms が付いた"
    );
    assert!(
        headers.get("x-denno-decode-tps").is_none(),
        "{label}: ストリーミング/非生成に x-denno-decode-tps が付いた"
    );
}

fn assert_timing(headers: &HeaderMap, label: &str) {
    let ttft = one(headers, "x-denno-ttft-ms", label);
    let tps = one(headers, "x-denno-decode-tps", label);
    let t: f64 = ttft.parse().unwrap_or_else(|_| panic!("{label}: ttft = {ttft:?}"));
    let p: f64 = tps.parse().unwrap_or_else(|_| panic!("{label}: tps = {tps:?}"));
    assert!(t >= 0.0, "{label}: ttft = {t}");
    assert!(p > 0.0, "{label}: 出力があるのに tps = {p}");
}

fn parse_json(bytes: &[u8], label: &str) -> Value {
    serde_json::from_slice(bytes).unwrap_or_else(|e| {
        panic!(
            "{label}: JSON として読めない ({e}): {}",
            String::from_utf8_lossy(bytes)
        )
    })
}

// ---------------------------------------------------------------------------
// 1. 制御エンドポイント (JSON)

#[tokio::test]
async fn control_endpoints_use_plain_json_and_request_id() {
    let app = app();
    let cases = [
        ("GET /health", get("/health")),
        ("GET /ready", get("/ready")),
        ("GET /v1/models", get("/v1/models")),
        (
            "POST /v1/messages/count_tokens",
            post_json(
                "/v1/messages/count_tokens",
                &json!({
                    "model": "denno-test",
                    "messages": [{"role": "user", "content": "hi"}],
                }),
            ),
        ),
    ];
    for (label, req) in cases {
        let (status, headers, body) = send(&app, req).await;
        assert!(status.is_success(), "{label}: {status} {}", String::from_utf8_lossy(&body));
        assert_eq!(
            assert_concrete_content_type(&headers, label),
            "application/json; charset=utf-8",
            "{label}: JSON 経路は application/json; charset=utf-8 固定 (P1036)"
        );
        assert!(one(&headers, "x-request-id", label).starts_with("req-"), "{label}");
        assert_not_cacheable(&headers, label);
        assert_sniff_policy(&headers, label);
        assert_no_timing(&headers, label);
        assert_no_duplicate_headers(&headers, label);
        parse_json(&body, label);
    }
}

#[tokio::test]
async fn metrics_is_prometheus_text_with_charset() {
    let (status, headers, body) = send(&app(), get("/metrics")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        one(&headers, "content-type", "GET /metrics"),
        "text/plain; version=0.0.4; charset=utf-8",
        "Prometheus の exposition format を名乗る"
    );
    assert!(one(&headers, "x-request-id", "GET /metrics").starts_with("req-"));
    assert_not_cacheable(&headers, "GET /metrics");
    assert_sniff_policy(&headers, "GET /metrics");
    assert_no_timing(&headers, "GET /metrics");
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("# TYPE"), "/metrics が exposition format でない: {text}");
}

// ---------------------------------------------------------------------------
// 2. 非ストリーミング生成 (JSON + 計測ヘッダ)

#[tokio::test]
async fn non_streaming_generations_keep_json_headers_and_timing() {
    let app = app();
    let cases = [
        ("POST /v1/messages", post_json("/v1/messages", &messages_body(false))),
        (
            "POST /v1/chat/completions",
            post_json("/v1/chat/completions", &chat_body(false)),
        ),
        (
            "POST /v1/completions",
            post_json(
                "/v1/completions",
                &json!({"model": "denno-test", "prompt": "hi", "max_tokens": 8}),
            ),
        ),
        (
            "POST /api/chat",
            post_json(
                "/api/chat",
                &json!({
                    "model": "denno-test",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": false,
                    "options": {"num_predict": 4},
                }),
            ),
        ),
        (
            "POST /api/generate",
            post_json("/api/generate", &ollama_body(false)),
        ),
    ];
    for (label, req) in cases {
        let (status, headers, body) = send(&app, req).await;
        assert!(status.is_success(), "{label}: {status} {}", String::from_utf8_lossy(&body));
        assert_eq!(
            assert_concrete_content_type(&headers, label),
            "application/json; charset=utf-8",
            "{label}"
        );
        assert!(one(&headers, "x-request-id", label).starts_with("req-"), "{label}");
        assert_timing(&headers, label);
        assert_not_cacheable(&headers, label);
        assert_sniff_policy(&headers, label);
        assert_no_duplicate_headers(&headers, label);
        parse_json(&body, label);
    }
}

// ---------------------------------------------------------------------------
// 3. SSE

#[tokio::test]
async fn sse_sets_event_stream_and_no_cache() {
    let app = app();
    for (label, req) in [
        ("POST /v1/messages stream", post_json("/v1/messages", &messages_body(true))),
        (
            "POST /v1/chat/completions stream",
            post_json("/v1/chat/completions", &chat_body(true)),
        ),
    ] {
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::OK, "{label}");
        assert_eq!(
            one(&headers, "content-type", label),
            "text/event-stream",
            "{label}: SSE の content-type"
        );
        // P1036 header_hygiene が SSE の cache-control を no-store へ統一する。
        assert_eq!(
            one(&headers, "cache-control", label),
            "no-store",
            "{label}: SSE は中間キャッシュ禁止 (P1036 no-store)"
        );
        assert!(
            headers.get("content-length").is_none(),
            "{label}: ストリームに content-length を付けない"
        );
        assert!(one(&headers, "x-request-id", label).starts_with("req-"), "{label}");
        assert_no_timing(&headers, label);
        assert_sniff_policy(&headers, label);
        assert_no_duplicate_headers(&headers, label);

        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event:") || text.contains("data:"),
            "{label}: SSE フレームが無い: {text}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Ollama NDJSON ストリーム

#[tokio::test]
async fn ollama_stream_is_ndjson_not_sse() {
    let (status, headers, body) = send(&app(), post_json("/api/generate", &ollama_body(true))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        one(&headers, "content-type", "POST /api/generate stream"),
        "application/x-ndjson",
        "Ollama は NDJSON。SSE と取り違えない"
    );
    assert_eq!(
        one(&headers, "cache-control", "POST /api/generate stream"),
        "no-store",
        "NDJSON 推論経路は P1036 の no-store (SSE 用の no-cache を混ぜない)"
    );
    assert!(one(&headers, "x-request-id", "POST /api/generate stream").starts_with("req-"));
    assert_no_timing(&headers, "POST /api/generate stream");
    assert_sniff_policy(&headers, "POST /api/generate stream");
    assert_no_duplicate_headers(&headers, "POST /api/generate stream");

    let text = String::from_utf8_lossy(&body);
    let mut lines = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        serde_json::from_str::<Value>(line)
            .unwrap_or_else(|e| panic!("NDJSON 行が JSON でない ({e}): {line}"));
        lines += 1;
    }
    assert!(lines >= 2, "1 行 1 JSON のストリームでない: {text}");
}

// ---------------------------------------------------------------------------
// 5. エラー封筒 (404 / 405 / 401 / 413 / 429)

#[tokio::test]
async fn error_responses_keep_json_headers_and_request_id() {
    // 404
    let (s, h, b) = send(&app(), get("/definitely-not-a-route")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(assert_concrete_content_type(&h, "404"), "application/json");
    assert!(one(&h, "x-request-id", "404").starts_with("req-"));
    assert_not_cacheable(&h, "404");
    assert_sniff_policy(&h, "404");
    parse_json(&b, "404");

    // 405
    let (s, h, b) = send(&app(), get("/v1/messages")).await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(assert_concrete_content_type(&h, "405"), "application/json");
    assert!(one(&h, "x-request-id", "405").starts_with("req-"));
    assert_not_cacheable(&h, "405");
    parse_json(&b, "405");

    // 401 (認証ありのルーター)
    let (s, h, b) = send(&app_auth(), get("/v1/models")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_eq!(assert_concrete_content_type(&h, "401"), "application/json");
    assert!(one(&h, "x-request-id", "401").starts_with("req-"));
    assert_not_cacheable(&h, "401");
    assert_sniff_policy(&h, "401");
    parse_json(&b, "401");

    // 413 (ボディ上限 64 byte)
    let big = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "x".repeat(256)}],
    });
    let (s, h, b) = send(&app_small_body(), post_json("/v1/messages", &big)).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    assert_eq!(assert_concrete_content_type(&h, "413"), "application/json");
    assert!(one(&h, "x-request-id", "413").starts_with("req-"));
    assert_not_cacheable(&h, "413");
    parse_json(&b, "413");

    // 429 (rpm=1 / burst=1 を使い切る)
    let app = app_rate_limited();
    let (s1, _, _) = send(&app, post_json("/v1/messages", &messages_body(false))).await;
    assert_eq!(s1, StatusCode::OK);
    let (s2, h2, b2) = send(&app, post_json("/v1/messages", &messages_body(false))).await;
    assert_eq!(s2, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(assert_concrete_content_type(&h2, "429"), "application/json");
    assert!(one(&h2, "x-request-id", "429").starts_with("req-"));
    assert_not_cacheable(&h2, "429");
    assert!(h2.get("retry-after").is_some(), "429 には Retry-After");
    parse_json(&b2, "429");
}

#[tokio::test]
async fn ready_503_has_the_same_header_hygiene() {
    let (status, headers, body) = send(&app_not_ready(), get("/ready")).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        assert_concrete_content_type(&headers, "/ready 503"),
        "application/json"
    );
    assert!(one(&headers, "x-request-id", "/ready 503").starts_with("req-"));
    assert_not_cacheable(&headers, "/ready 503");
    assert_sniff_policy(&headers, "/ready 503");
    assert_no_timing(&headers, "/ready 503");
    let v = parse_json(&body, "/ready 503");
    assert_eq!(v["ready"], false);
    assert_eq!(v["status"], "loading");
}

// ---------------------------------------------------------------------------
// 6. クライアント指定 ID はトランスポートが違っても同じ扱い

#[tokio::test]
async fn custom_request_id_is_echoed_on_every_transport() {
    let app = app();
    let rid = "hygiene-1299";
    let cases = [
        ("GET /health", get_rid("/health", rid)),
        ("GET /metrics", get_rid("/metrics", rid)),
        ("GET /v1/models", get_rid("/v1/models", rid)),
        (
            "POST /v1/messages SSE",
            post_json_rid("/v1/messages", &messages_body(true), rid),
        ),
        (
            "POST /v1/chat/completions SSE",
            post_json_rid("/v1/chat/completions", &chat_body(true), rid),
        ),
        (
            "POST /api/generate NDJSON",
            post_json_rid("/api/generate", &ollama_body(true), rid),
        ),
        (
            "POST /v1/completions JSON",
            post_json_rid(
                "/v1/completions",
                &json!({"model": "denno-test", "prompt": "hi", "max_tokens": 4}),
                rid,
            ),
        ),
        ("GET 404", get_rid("/nope-1299", rid)),
    ];
    for (label, req) in cases {
        let (_s, headers, _b) = send(&app, req).await;
        assert_eq!(one(&headers, "x-request-id", label), rid, "{label}");
        assert_no_duplicate_headers(&headers, label);
        assert_sniff_policy(&headers, label);
        assert_not_cacheable(&headers, label);
    }
}

// ---------------------------------------------------------------------------
// 8. ヘッダ名・値の妥当性 (注入・制御文字の混入防止)

#[tokio::test]
async fn header_names_and_values_are_ascii_tokens() {
    let app = app();
    let cases = [
        ("GET /health", get("/health")),
        ("GET /metrics", get("/metrics")),
        ("POST /v1/messages", post_json("/v1/messages", &messages_body(false))),
        ("POST /v1/messages SSE", post_json("/v1/messages", &messages_body(true))),
        ("POST /api/generate NDJSON", post_json("/api/generate", &ollama_body(true))),
        ("GET /nope", get("/nope-1299")),
    ];
    for (label, req) in cases {
        let (_s, headers, _b) = send(&app, req).await;
        for (name, value) in headers.iter() {
            let n = name.as_str();
            assert!(
                n.chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
                "{label}: ヘッダ名が token でない: {n:?}"
            );
            let v = value
                .to_str()
                .unwrap_or_else(|_| panic!("{label}: {n} の値が ASCII でない"));
            assert!(
                !v.contains('\r') && !v.contains('\n'),
                "{label}: {n} の値に改行が混じる"
            );
            assert!(
                v.chars().all(|c| (' '..='~').contains(&c)),
                "{label}: {n} の値に制御文字: {v:?}"
            );
        }
    }
}


// ---------------------------------------------------------------------------
// 7. 横断: ヘッダの重複・余計な露出が無い

#[tokio::test]
async fn no_leaky_or_duplicated_headers_across_the_surface() {
    let app = app();
    let cases = [
        ("GET /health", get("/health")),
        ("GET /ready", get("/ready")),
        ("GET /metrics", get("/metrics")),
        ("GET /v1/models", get("/v1/models")),
        ("POST /v1/messages", post_json("/v1/messages", &messages_body(false))),
        ("POST /v1/messages SSE", post_json("/v1/messages", &messages_body(true))),
        (
            "POST /v1/chat/completions",
            post_json("/v1/chat/completions", &chat_body(false)),
        ),
        ("POST /api/generate NDJSON", post_json("/api/generate", &ollama_body(true))),
        ("GET /nope", get("/nope-1299")),
    ];
    for (label, req) in cases {
        let (_s, headers, _b) = send(&app, req).await;
        assert_no_duplicate_headers(&headers, label);
        // サーバ実装を晒すヘッダは出さない (axum も付けない契約を固定)。
        assert!(headers.get("server").is_none(), "{label}: server ヘッダ");
        assert!(headers.get("x-powered-by").is_none(), "{label}: x-powered-by ヘッダ");
        // content-type は必ず 1 個・具体的。
        assert_concrete_content_type(&headers, label);
        // リクエスト ID は全経路に 1 個。
        one(&headers, "x-request-id", label);
    }
}
