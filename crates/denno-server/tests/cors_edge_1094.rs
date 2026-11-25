//! task 1094: denno-server の CORS 正規化・ルックアライク判定・
//! ワイルドカード / 明示一覧 / 既定ループバックの差、および
//! 404 / 405 / プリフライトの封筒まわりのエッジケース。
//!
//! task 916 (`cors_916.rs`) が「主経路」を固定しているのに対し、本ファイルは
//! - scheme / host の大文字小文字と末尾スラッシュの正規化
//! - 見せかけたドメイン (`localhost.evil.example` 等) の拒否
//! - `*` / カンマ一覧 / 既定での `Vary: Origin` と `Expose-Headers` の差
//! - 未知パス 404・未対応メソッド 405 の OpenAI 封筒のキー集合
//! を追加で固定する。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{
    parse_cors_origin, AppState, CorsConfig, EngineInfo, Scheduler,
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

fn simple(method: &str, uri: &str, origin: Option<&str>) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some(o) = origin {
        b = b.header("origin", o);
    }
    b.body(Body::empty()).unwrap()
}

fn preflight(uri: &str, origin: &str, method: &str) -> Request<Body> {
    Request::builder()
        .method("OPTIONS")
        .uri(uri)
        .header("origin", origin)
        .header("access-control-request-method", method)
        .body(Body::empty())
        .unwrap()
}

fn envelope(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("JSON でない: {e}: {body}"))
}

/// OpenAI エラー封筒のキー集合がちょうど 4 つであること。
fn assert_envelope_shape(v: &Value, code: &str) {
    let err = v["error"].as_object().expect("error はオブジェクト");
    let mut keys: Vec<&str> = err.keys().map(|k| k.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(keys, ["code", "message", "param", "type"], "封筒のキー: {v}");
    assert_eq!(err["type"], "invalid_request_error", "{v}");
    assert_eq!(err["code"], code, "{v}");
    assert!(err["param"].is_null(), "param は null で存在する: {v}");
    assert!(
        err["message"].as_str().is_some_and(|m| !m.is_empty()),
        "message は空でない文字列: {v}"
    );
}

// ---------- 正規化: scheme / host の大小と末尾スラッシュ ----------

#[test]
fn default_config_normalizes_case_and_trailing_slashes() {
    let c = CorsConfig::default();
    assert!(c.allows("HTTP://LOCALHOST:8080"), "scheme は大小無視");
    assert!(c.allows("Https://LocalHost:5173/"), "host は大小無視 + 末尾スラッシュ");
    assert!(c.allows("http://127.0.0.1:8080//"), "末尾スラッシュは複数でも可");
    assert!(c.allows("HTTPS://[::1]:3000/"), "IPv6 ループバックも大小無視");
    assert!(c.allows("  http://localhost:5173  "), "前後の空白は無視");
    assert!(!c.allows("http://localhost:abc"), "非数値ポートは不可");
    assert!(!c.allows("http://localhost:"), "空ポートは不可");
    assert!(!c.allows("ftp://localhost"), "http(s) 以外の scheme は不可");
}

#[test]
fn allowlist_normalizes_entries_and_lookups() {
    let c = parse_cors_origin(Some("HTTPS://APP.EXAMPLE.COM/, HTTP://LOCALHOST:5173///"));
    assert!(!c.allow_loopback, "明示一覧はループバック既定を切る");
    assert_eq!(
        c.origins,
        vec![
            "https://app.example.com".to_string(),
            "http://localhost:5173".to_string()
        ],
        "保存時に正規化される"
    );
    assert!(c.allows("https://app.example.com"));
    assert!(c.allows("HTTPS://App.Example.COM/"));
    assert!(c.allows("http://localhost:5173/"));
    assert!(c.allows("HTTP://LOCALHOST:5173"));
    assert!(!c.allows("http://localhost:5174"), "一覧に無いポート");
}

// ---------- ルックアライクのドメイン拒否 ----------

#[test]
fn lookalike_domains_are_rejected_by_default() {
    let c = CorsConfig::default();
    for bad in [
        "http://localhost.evil.example",
        "http://localhost.evil.example:8080",
        "http://127.0.0.1.evil.example",
        "http://127.0.0.1.evil.example:80",
        "http://localhost-evil.example",
        "https://[::1].evil.example",
        "https://[::1].evil.example:443",
        "http://evil.example/localhost",
        "http://evil@localhost:5173",
        "http://localhost:5173.evil.example",
    ] {
        assert!(!c.allows(bad), "見せかけのオリジンを通してはいけない: {bad}");
    }
}

#[test]
fn lookalike_domains_are_not_matched_by_allowlist_entries() {
    let c = parse_cors_origin(Some("http://localhost, https://[::1]:3000"));
    assert!(c.allows("http://localhost"));
    assert!(!c.allows("http://localhost.evil.example"));
    assert!(!c.allows("http://localhost.evil.example:80"));
    assert!(c.allows("https://[::1]:3000"));
    assert!(!c.allows("https://[::1]:3000.evil.example"));
}

#[tokio::test]
async fn lookalike_origin_gets_no_cors_headers_on_real_requests() {
    for bad in [
        "http://localhost.evil.example",
        "http://127.0.0.1.evil.example:8080",
    ] {
        let (s, h, _) = send(&app(), simple("GET", "/health", Some(bad))).await;
        assert_eq!(s, StatusCode::OK, "CORS は応答自体を拒否しない");
        assert_eq!(
            hdr(&h, "access-control-allow-origin"),
            None,
            "見せかけのオリジンに Allow-Origin を出してはいけない: {bad}"
        );
        assert_eq!(hdr(&h, "access-control-expose-headers"), None, "{bad}");
        assert_eq!(hdr(&h, "vary"), None, "{bad}");
    }
}

#[tokio::test]
async fn lookalike_origin_preflight_is_204_without_allow_headers() {
    let (s, h, body) = send(
        &app(),
        preflight("/v1/messages", "http://localhost.evil.example", "POST"),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
    assert_eq!(hdr(&h, "access-control-allow-methods"), None);
    assert_eq!(hdr(&h, "access-control-allow-headers"), None);
    assert_eq!(hdr(&h, "access-control-max-age"), None);
    assert!(body.is_empty());
}

// ---------- 既定 (ループバックのみ) ----------

#[tokio::test]
async fn default_loopback_echoes_raw_origin_and_exposes_metrics() {
    // 大文字 + 末尾スラッシュでも、応答は受け取った生の Origin を返す。
    let raw = "HTTP://LocalHost:5173/";
    let (s, h, _) = send(&app(), simple("GET", "/health", Some(raw))).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some(raw),
        "Allow-Origin は生の値をそのまま返す"
    );
    assert_eq!(hdr(&h, "vary").as_deref(), Some("Origin"));
    let expose = hdr(&h, "access-control-expose-headers").unwrap_or_default();
    for name in ["x-request-id", "x-denno-ttft-ms", "x-denno-decode-tps"] {
        assert!(expose.contains(name), "expose = {expose} に {name} が無い");
    }
    // プリフライト用のヘッダは通常応答には出さない。
    assert_eq!(hdr(&h, "access-control-allow-methods"), None);
    assert_eq!(hdr(&h, "access-control-max-age"), None);
}

#[tokio::test]
async fn default_config_rejects_lan_and_public_origins() {
    for bad in ["http://192.168.1.10:8080", "https://example.com", "file://"] {
        let (_, h, _) = send(&app(), simple("GET", "/health", Some(bad))).await;
        assert_eq!(
            hdr(&h, "access-control-allow-origin"),
            None,
            "既定で開けてはいけない: {bad}"
        );
    }
}

// ---------- ワイルドカード ----------

#[tokio::test]
async fn wildcard_sets_star_without_vary_but_still_exposes() {
    let a = app_with(parse_cors_origin(Some("*")));
    let (s, h, _) = send(&a, simple("GET", "/health", Some("https://anything.example"))).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(hdr(&h, "access-control-allow-origin").as_deref(), Some("*"));
    assert_eq!(hdr(&h, "vary"), None, "`*` では Vary: Origin を出さない");
    let expose = hdr(&h, "access-control-expose-headers").unwrap_or_default();
    assert!(expose.contains("x-request-id"), "expose = {expose}");
}

#[tokio::test]
async fn wildcard_preflight_is_star_without_vary() {
    let a = app_with(parse_cors_origin(Some("*")));
    for origin in ["https://anything.example", "http://localhost.evil.example"] {
        let (s, h, body) = send(&a, preflight("/v1/messages", origin, "POST")).await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        assert_eq!(
            hdr(&h, "access-control-allow-origin").as_deref(),
            Some("*"),
            "ワイルドカードでは常に `*`: {origin}"
        );
        assert_eq!(hdr(&h, "vary"), None);
        assert_eq!(hdr(&h, "access-control-max-age").as_deref(), Some("86400"));
        let methods = hdr(&h, "access-control-allow-methods").unwrap_or_default();
        assert!(methods.contains("POST"), "methods = {methods}");
        assert!(body.is_empty());
    }
}

// ---------- 明示カンマ一覧 ----------

#[tokio::test]
async fn allowlist_echoes_listed_origin_only_and_varies() {
    let a = app_with(parse_cors_origin(Some(
        " https://App.Example.com/ , http://localhost:5173 ",
    )));

    // 一覧内 (大小 + 末尾スラッシュ違い) は生の値をそのまま返す。
    let raw = "HTTPS://APP.EXAMPLE.COM/";
    let (_, h, _) = send(&a, simple("GET", "/health", Some(raw))).await;
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some(raw)
    );
    assert_eq!(hdr(&h, "vary").as_deref(), Some("Origin"));

    // 一覧に無いループバックは不可。
    let (_, h, _) = send(&a, simple("GET", "/health", Some("http://127.0.0.1:5173"))).await;
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
    assert_eq!(hdr(&h, "access-control-expose-headers"), None);
    assert_eq!(hdr(&h, "vary"), None);

    // 見せかけのドメインも不可。
    let (_, h, _) = send(
        &a,
        simple("GET", "/health", Some("https://app.example.com.evil.example")),
    )
    .await;
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
}

#[tokio::test]
async fn allowlist_preflight_varies_and_skips_foreign_origin() {
    let a = app_with(parse_cors_origin(Some("https://app.example.com")));

    let (s, h, _) = send(
        &a,
        preflight("/v1/chat/completions", "https://app.example.com", "POST"),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("https://app.example.com")
    );
    assert_eq!(hdr(&h, "vary").as_deref(), Some("Origin"));

    let (s, h, body) = send(
        &a,
        preflight("/v1/chat/completions", "https://other.example", "POST"),
    )
    .await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
    assert_eq!(hdr(&h, "vary"), None);
    assert!(body.is_empty());
}

// ---------- 404 封筒 ----------

#[tokio::test]
async fn unknown_nested_path_returns_clean_404_envelope() {
    for path in ["/v1/messages/extra", "/definitely-not-a-route", "/v1/models/x/y"] {
        let (s, h, body) = send(&app(), simple("GET", path, Some("http://localhost:5173"))).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{path}");
        assert_eq!(
            hdr(&h, "content-type").as_deref(),
            Some("application/json"),
            "{path}"
        );
        let v = envelope(&body);
        assert_envelope_shape(&v, "unknown_route");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(msg.contains(path), "message にパスが無い: {msg}");
        assert!(msg.starts_with("Unknown route: GET "), "message = {msg}");
        assert!(hdr(&h, "x-request-id").is_some(), "404 にも x-request-id");
        assert_eq!(
            hdr(&h, "access-control-allow-origin").as_deref(),
            Some("http://localhost:5173"),
            "404 にも CORS が掛かる"
        );
    }
}

#[tokio::test]
async fn unknown_path_404_has_no_cors_headers_for_foreign_origin() {
    let (s, h, body) = send(
        &app(),
        simple("GET", "/not-a-route", Some("https://evil.example")),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
    assert_envelope_shape(&envelope(&body), "unknown_route");
}

// ---------- 405 封筒 ----------

#[tokio::test]
async fn wrong_methods_return_clean_405_envelopes() {
    for (method, path) in [("DELETE", "/v1/models"), ("GET", "/v1/messages")] {
        let (s, h, body) =
            send(&app(), simple(method, path, Some("http://localhost:5173"))).await;
        assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED, "{method} {path}");
        assert_eq!(
            hdr(&h, "content-type").as_deref(),
            Some("application/json"),
            "{method} {path}"
        );
        let v = envelope(&body);
        assert_envelope_shape(&v, "method_not_allowed");
        let msg = v["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains(method) && msg.contains(path),
            "message = {msg}"
        );
        assert!(
            hdr(&h, "x-request-id").is_some(),
            "405 にも x-request-id が要る"
        );
        assert_eq!(
            hdr(&h, "access-control-allow-origin").as_deref(),
            Some("http://localhost:5173"),
            "405 にも CORS が掛かる"
        );
    }
}

#[tokio::test]
async fn wrong_method_405_has_no_cors_headers_for_foreign_origin() {
    let (s, h, body) = send(
        &app(),
        simple("DELETE", "/v1/models", Some("https://evil.example")),
    )
    .await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    assert_eq!(hdr(&h, "access-control-allow-origin"), None);
    assert_envelope_shape(&envelope(&body), "method_not_allowed");
}

#[tokio::test]
async fn plain_options_without_request_method_is_405_json_not_preflight() {
    // `Access-Control-Request-Method` が無い OPTIONS はプリフライトではなく
    // 通常の未対応メソッド (405) として JSON で返る。
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/v1/models")
        .header("origin", "http://localhost:5173")
        .body(Body::empty())
        .unwrap();
    let (s, h, body) = send(&app(), req).await;
    assert_eq!(s, StatusCode::METHOD_NOT_ALLOWED);
    assert!(!body.is_empty(), "405 は JSON ボディを持つ");
    assert_envelope_shape(&envelope(&body), "method_not_allowed");
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173"),
        "405 にも CORS が掛かる"
    );
}

// ---------- プリフライトはボディを読まずに 204 ----------

#[tokio::test]
async fn preflight_is_204_and_ignores_body_and_query() {
    let req = Request::builder()
        .method("OPTIONS")
        .uri("/v1/chat/completions?x=1")
        .header("origin", "http://127.0.0.1:3000/")
        .header("access-control-request-method", "POST")
        .header("content-type", "application/json")
        .body(Body::from("{\"not\":\"parsed\"}"))
        .unwrap();
    let (s, h, body) = send(&app(), req).await;
    assert_eq!(s, StatusCode::NO_CONTENT);
    assert!(body.is_empty());
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://127.0.0.1:3000/")
    );
    assert_eq!(hdr(&h, "vary").as_deref(), Some("Origin"));
}

#[tokio::test]
async fn generation_still_works_with_normalized_origin() {
    let body = json!({
        "model": "denno-test",
        "max_tokens": 4,
        "messages": [{"role": "user", "content": "t0"}],
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .header("origin", "HTTPS://LOCALHOST:5173/")
        .body(Body::from(body.to_string()))
        .unwrap();
    let (s, h, _) = send(&app(), req).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("HTTPS://LOCALHOST:5173/")
    );
    assert_eq!(hdr(&h, "vary").as_deref(), Some("Origin"));
}
