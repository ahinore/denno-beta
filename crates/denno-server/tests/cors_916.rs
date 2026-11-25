//! task 916: CORS (DENNO_CORS_ORIGIN / --cors-origin) と
//! 404 / 405 の JSON エラー封筒の統合テスト。
//!
//! - 既定はループバック (localhost / 127.0.0.1 / [::1]) の任意ポートだけ許可
//! - `*` は任意オリジン
//! - カンマ区切りの一覧はそのオリジンだけ許可 (大文字小文字・末尾スラッシュを正規化)
//! - プリフライト (OPTIONS + Access-Control-Request-Method) は
//!   204 で Allow-Origin / Methods / Headers / Max-Age を返す
//! - 未知ルートは 404、未対応メソッドは 405 を OpenAI 形式の JSON で返す

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{
    parse_cors_origin, resolve_cors_origin, AppState, CorsConfig, EngineInfo, Scheduler,
    CORS_ORIGIN_ENV,
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

/// 既定 CORS (ループバックのみ) のルーター。
fn app() -> Router {
    denno_server::router(state())
}

/// CORS 設定を明示したルーター。
fn app_with(cors: CorsConfig) -> Router {
    denno_server::router_with_cors(state(), cors)
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

fn get_with_origin(uri: &str, origin: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("origin", origin)
        .body(Body::empty())
        .unwrap()
}

fn preflight(uri: &str, origin: &str, method: &str, req_headers: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .header("origin", origin)
        .header("access-control-request-method", method);
    if let Some(h) = req_headers {
        b = b.header("access-control-request-headers", h);
    }
    b.body(Body::empty()).unwrap()
}

fn post_json(uri: &str, body: &Value, origin: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(o) = origin {
        b = b.header("origin", o);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

// ---------- 設定の解決 ----------

#[test]
fn config_default_allows_loopback_any_port() {
    let c = parse_cors_origin(None);
    assert!(!c.allow_any, "既定で任意オリジンを開けない");
    assert!(c.allow_loopback, "既定でローカルブラウザは通す");
    assert!(c.allows("http://localhost:5173"));
    assert!(c.allows("http://127.0.0.1:8080"));
    assert!(c.allows("https://[::1]:3000"));
    assert!(c.allows("http://localhost"));
    assert!(!c.allows("http://localhost.evil.example:5173"));
    assert!(!c.allows("http://192.168.1.10:8080"));
    assert!(!c.allows("https://example.com"));
    assert!(!c.allows("file://"));
    // 空文字は既定と同じ。
    assert_eq!(parse_cors_origin(Some("   ")), CorsConfig::default());
}

#[test]
fn config_star_opens_everything() {
    let c = parse_cors_origin(Some("*"));
    assert!(c.allow_any);
    assert!(!c.allow_loopback);
    assert!(c.allows("https://example.com"));
    assert!(c.allows("http://localhost:5173"));
}

#[test]
fn config_list_is_normalized_and_exclusive() {
    let c = parse_cors_origin(Some(
        " https://App.Example.com/ , http://localhost:5173 ,, ",
    ));
    assert!(!c.allow_any);
    assert!(!c.allow_loopback, "明示一覧ではループバック既定を切り、一覧だけ許可");
    assert_eq!(
        c.origins,
        vec![
            "https://app.example.com".to_string(),
            "http://localhost:5173".to_string()
        ]
    );
    assert!(c.allows("https://app.example.com"));
    assert!(c.allows("HTTPS://APP.EXAMPLE.COM/"));
    assert!(c.allows("http://localhost:5173"));
    assert!(!c.allows("http://localhost:5174"), "一覧に無いポートは不可");
    assert!(!c.allows("https://other.example.com"));
}

#[test]
fn env_var_is_used_when_cli_is_absent_and_cli_wins() {
    // この env を読むテストはこの 1 本だけ (テスト間の競合を避ける)。
    std::env::set_var(CORS_ORIGIN_ENV, "https://from-env.example");
    let c = resolve_cors_origin(None);
    assert_eq!(c.origins, vec!["https://from-env.example".to_string()]);
    assert!(c.allows("https://from-env.example"));

    let c = resolve_cors_origin(Some("*"));
    assert!(c.allow_any, "CLI が env より優先する");

    std::env::remove_var(CORS_ORIGIN_ENV);
    let c = resolve_cors_origin(None);
    assert_eq!(c, CorsConfig::default());
}

// ---------- 通常リクエストのヘッダ ----------
#[tokio::test]
async fn loopback_origin_gets_allow_origin_by_default() {
    let (s, h, _) = send(&app(), get_with_origin("/health", "http://localhost:5173")).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173")
    );
    assert_eq!(hdr(&h, "vary").as_deref(), Some("Origin"));
    // 計測ヘッダを JS から読めるように expose している。
    let expose = hdr(&h, "access-control-expose-headers").unwrap_or_default();
    assert!(expose.contains("x-request-id"), "expose = {expose}");
    assert!(expose.contains("x-denno-ttft-ms"), "expose = {expose}");
}

#[tokio::test]
async fn foreign_origin_gets_no_cors_headers_by_default() {
    let (s, h, _) = send(&app(), get_with_origin("/health", "https://evil.example")).await;
    assert_eq!(s, StatusCode::OK, "CORS は応答自体を拒否しない (ブラウザが止める)");
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
    assert_eq!(hdr(&h, "access-control-expose-headers"), None);
}

#[tokio::test]
async fn allowlist_origin_is_echoed_and_others_not() {
    let a = app_with(parse_cors_origin(Some("https://app.example.com")));
    let (_, h, _) = send(&a, get_with_origin("/health", "https://app.example.com")).await;
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("https://app.example.com")
    );

    let (_, h, _) = send(&a, get_with_origin("/health", "http://localhost:5173")).await;
    assert_eq!(
        hdr(&h, "access-control-allow-origin"),
        None,
        "明示一覧に無いオリジン (ループバック含む) は不可"
    );
}

#[tokio::test]
async fn star_config_returns_wildcard() {
    let a = app_with(parse_cors_origin(Some("*")));
    let (_, h, _) = send(&a, get_with_origin("/health", "https://anything.example")).await;
    assert_eq!(hdr(&h, "access-control-allow-origin").as_deref(), Some("*"));
    // `*` では資格情報付きアクセスを許さない (Vary: Origin も不要)。
    assert_eq!(hdr(&h, "vary"), None);
}

// ---------- プリフライト ----------

#[tokio::test]
async fn preflight_for_loopback_origin() {
    let (s, h, body) = send(
        &app(),
        preflight("/v1/messages", "http://localhost:5173", "POST", None),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT, "プリフライトは 204");
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173")
    );
    let methods = hdr(&h, "access-control-allow-methods").unwrap_or_default();
    assert!(methods.contains("POST"), "methods = {methods}");
    assert!(methods.contains("OPTIONS"), "methods = {methods}");
    let allow_headers = hdr(&h, "access-control-allow-headers").unwrap_or_default();
    assert!(allow_headers.contains("content-type"), "headers = {allow_headers}");
    assert!(allow_headers.contains("x-api-key"), "headers = {allow_headers}");
    assert_eq!(hdr(&h, "access-control-max-age").as_deref(), Some("86400"));
    assert!(body.is_empty(), "プリフライトにボディは無い");
}

#[tokio::test]
async fn preflight_echoes_requested_headers() {
    let (s, h, _) = send(
        &app(),
        preflight(
            "/v1/messages",
            "http://127.0.0.1:3000",
            "POST",
            Some("x-custom-header,content-type"),
        ),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(
        hdr(&h, "access-control-allow-headers").as_deref(),
        Some("x-custom-header,content-type")
    );
}

#[tokio::test]
async fn preflight_from_foreign_origin_has_no_allow_origin() {
    let (s, h, _) = send(
        &app(),
        preflight("/v1/messages", "https://evil.example", "POST", None),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
}

#[tokio::test]
async fn preflight_for_unknown_path_still_204_when_allowed() {
    // ブラウザは失敗したプリフライトをキャッシュしないので、未知パスでも
    // 許可オリジンなら 204 を返す (実リクエスト側で 404 になる)。
    let (s, h, _) = send(
        &app(),
        preflight("/not-a-route", "http://localhost:5173", "GET", None),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173")
    );
}

#[tokio::test]
async fn plain_options_without_request_method_is_405_json() {
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/v1/messages")
        .body(Body::empty())
        .unwrap();
    let (s, _, body) = send(&app(), req).await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    let v: Value = serde_json::from_str(&body).expect("405 は JSON");
    assert_eq!(v["error"]["code"], "method_not_allowed");
}

// ---------- 404 / 405 の封筒 ----------

#[tokio::test]
async fn unknown_route_returns_json_envelope() {
    let (s, h, body) = send(&app(), get_with_origin("/definitely-not-a-route", "http://localhost:5173"))
        .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(hdr(&h, "content-type").as_deref(), Some("application/json"));
    let v: Value = serde_json::from_str(&body).expect("404 は JSON");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["code"], "unknown_route");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("/definitely-not-a-route"),
        "{v}"
    );
    assert!(v["error"].get("param").is_some(), "param キーは null で存在する");
    // 404 も log_requests の層を通る (x-request-id が付く)。
    assert!(hdr(&h, "x-request-id").is_some(), "404 に x-request-id が無い");
}

#[tokio::test]
async fn method_not_allowed_returns_json_envelope() {
    let (s, _, body) = send(&app(), get_with_origin("/v1/messages", "http://localhost:5173")).await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    let v: Value = serde_json::from_str(&body).expect("405 は JSON");
    assert_eq!(v["error"]["code"], "method_not_allowed");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("/v1/messages"),
        "{v}"
    );
}

// ---------- 生成経路でも CORS が効く ----------

#[tokio::test]
async fn generation_response_carries_cors_headers() {
    let body = json!({
        "model": "denno-test",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "t0"}],
    });
    let (s, h, _) = send(
        &app(),
        post_json("/v1/messages", &body, Some("http://localhost:5173")),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173")
    );
}

#[tokio::test]
async fn cors_is_applied_to_error_responses_from_handlers() {
    // 未知モデルの 404 (ハンドラ内の ApiError) にも Allow-Origin が付く。
    let body = json!({
        "model": "nope",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "t0"}],
    });
    let (s, h, _) = send(
        &app(),
        post_json("/v1/messages", &body, Some("http://localhost:5173")),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173")
    );
}
