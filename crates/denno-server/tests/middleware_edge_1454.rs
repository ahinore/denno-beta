//! task 1454: ミドルウェアの優先順位と相互作用の契約テスト。
//!
//! `src/lib.rs` の層は外側から
//! `header_hygiene` → `log_requests` → CORS → レート制限 →
//! `body_limit_middleware` / `DefaultBodyLimit` → リクエスト期限 → ルート、
//! の順に重なる。認証 (`check_auth`) はルート (ハンドラ) の内側にある。
//! 本ファイルはこの順序が外部契約として観測できることを固定する:
//!
//! - `OPTIONS` プリフライトは CORS 層で 204 を返し、認証もボディ読み込みも
//!   通らない (API キー設定済みでも 401 にしない)。
//! - `Content-Length` が上限を超えるボディはボディ上限層で 413 になり、
//!   ハンドラの認証 (401) より先に返る。
//! - `Access-Control-Request-Method` の無い `OPTIONS` はプリフライトではなく
//!   405 (OpenAI 封筒) で、これも認証より先。
//! - `x-request-id` は全応答 (204 / 401 / 404 / 405 / 413) に付き、
//!   クライアント値を無害化して引き継ぐ。無ければ生成する。
//! - セキュリティヘッダ (`x-content-type-options: nosniff`) と CORS ヘッダは
//!   エラー応答にも付く。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::RateLimiter;
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler, MAX_REQUEST_ID_LEN};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// fixture

const KEY: &str = "secret-key-1454";
const LOOPBACK_ORIGIN: &str = "http://localhost:5173";
const BODY_LIMIT: usize = 64;

fn state(api_key: Option<&str>) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
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

/// 認証あり + ボディ上限 64 byte + 既定 CORS (ループバック許可)。
fn app_auth() -> Router {
    denno_server::router_with_body_limit(
        state(Some(KEY)),
        CorsConfig::default(),
        RateLimiter::disabled(),
        BODY_LIMIT,
    )
}

/// 認証なし + ボディ上限 64 byte + 既定 CORS。
fn app_open() -> Router {
    denno_server::router_with_body_limit(
        state(None),
        CorsConfig::default(),
        RateLimiter::disabled(),
        BODY_LIMIT,
    )
}

// ---------------------------------------------------------------------------
// リクエスト組み立て / 応答読み取り

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

fn preflight(path: &str, origin: &str, method: Option<&str>, req_headers: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("OPTIONS")
        .uri(path)
        .header("origin", origin);
    if let Some(m) = method {
        b = b.header("access-control-request-method", m);
    }
    if let Some(h) = req_headers {
        b = b.header("access-control-request-headers", h);
    }
    b.body(Body::empty()).unwrap()
}

fn post_messages(body: Vec<u8>, content_length: Option<usize>, key: Option<&str>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json");
    if let Some(n) = content_length {
        b = b.header("content-length", n.to_string());
    }
    if let Some(k) = key {
        b = b.header("x-api-key", k);
    }
    b.body(Body::from(body)).unwrap()
}

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or_else(|e| panic!("JSON でない: {e}: {body}"))
}

/// Anthropic 形式のエラー封筒 (`{"type":"error","error":{"type","message"}}`)。
fn assert_anthropic_error(v: &Value, kind: &str) {
    assert_eq!(v["type"], "error", "{v}");
    let err = v["error"].as_object().expect("error はオブジェクト");
    assert_eq!(err["type"], kind, "{v}");
    assert!(
        err["message"].as_str().is_some_and(|m| !m.is_empty()),
        "message は空でない: {v}"
    );
}

/// OpenAI 形式のエラー封筒 (`{"error":{"message","type","param","code"}}`)。
fn assert_openai_error(v: &Value, code: &str) {
    let err = v["error"].as_object().expect("error はオブジェクト");
    assert_eq!(err["code"], code, "{v}");
    assert!(err["param"].is_null(), "param は null: {v}");
    assert!(
        err["message"].as_str().is_some_and(|m| !m.is_empty()),
        "message は空でない: {v}"
    );
}

/// 全応答に付くはずのヘッダ (ヘッダ衛生 + リクエスト ID)。
fn assert_common_headers(headers: &HeaderMap, ctx: &str) {
    assert_eq!(
        hdr(headers, "x-content-type-options").as_deref(),
        Some("nosniff"),
        "{ctx}: nosniff"
    );
    let rid = hdr(headers, "x-request-id").unwrap_or_else(|| panic!("{ctx}: x-request-id が無い"));
    assert!(!rid.is_empty(), "{ctx}: x-request-id が空");
    assert!(
        rid.len() <= MAX_REQUEST_ID_LEN,
        "{ctx}: x-request-id が上限超過 ({})",
        rid.len()
    );
}

// ---------------------------------------------------------------------------
// プリフライト vs 認証

#[test]
fn preflight_bypasses_api_key_auth() {
    let app = app_auth();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // 認証必須のサーバでも、プリフライトは 204 で完結する。
        let req = preflight(
            "/v1/messages",
            LOOPBACK_ORIGIN,
            Some("POST"),
            Some("content-type, x-api-key"),
        );
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::NO_CONTENT, "本文: {body}");
        assert!(body.is_empty(), "204 に本文は無い: {body}");
        assert_eq!(
            hdr(&headers, "access-control-allow-origin").as_deref(),
            Some(LOOPBACK_ORIGIN)
        );
        assert_eq!(
            hdr(&headers, "access-control-allow-methods").as_deref(),
            Some("GET, POST, OPTIONS")
        );
        assert_eq!(
            hdr(&headers, "access-control-allow-headers").as_deref(),
            Some("content-type, x-api-key"),
            "要求ヘッダをそのまま返す"
        );
        assert_eq!(
            hdr(&headers, "access-control-max-age").as_deref(),
            Some("86400")
        );
        assert_eq!(hdr(&headers, "vary").as_deref(), Some("Origin"));
        assert_common_headers(&headers, "preflight");

        // 同じサーバへの通常 POST は認証が要る (プリフライトだけが素通り)。
        let req = post_messages(b"{}".to_vec(), None, None);
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_anthropic_error(&json_of(&body), "authentication_error");
        assert_common_headers(&headers, "401");
    });
}

#[test]
fn preflight_without_request_headers_uses_allowlist_default() {
    let app = app_open();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // `Access-Control-Request-Headers` が無いときは既定の許可一覧を返す。
        let req = preflight("/v1/chat/completions", LOOPBACK_ORIGIN, Some("POST"), None);
        let (status, headers, _) = send(&app, req).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        let allow = hdr(&headers, "access-control-allow-headers").unwrap_or_default();
        for name in ["content-type", "x-api-key", "authorization", "x-request-id"] {
            assert!(allow.contains(name), "既定許可に {name} が無い: {allow}");
        }
    });
}

#[test]
fn preflight_from_disallowed_origin_has_no_cors_headers() {
    let app = app_open();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let req = preflight("/v1/messages", "https://app.example.com", Some("POST"), None);
        let (status, headers, _) = send(&app, req).await;
        // プリフライト自体は 204 で返るが、許可していないオリジンにはヘッダを出さない。
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert!(hdr(&headers, "access-control-allow-origin").is_none());
        assert!(hdr(&headers, "access-control-allow-methods").is_none());
        assert_common_headers(&headers, "preflight 拒否");
    });
}

#[test]
fn options_without_access_control_request_method_hits_405_not_401() {
    let app = app_auth();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // プリフライトの条件 (ACRM) を満たさない OPTIONS は通常リクエストとして
        // ルーティングされ、405 になる。認証 (401) よりルーティングが先。
        let req = preflight("/v1/messages", LOOPBACK_ORIGIN, None, None);
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "本文: {body}");
        assert_openai_error(&json_of(&body), "method_not_allowed");
        // 許可オリジンなので CORS ヘッダは付く (エラーでも CORS 層は外側)。
        assert_eq!(
            hdr(&headers, "access-control-allow-origin").as_deref(),
            Some(LOOPBACK_ORIGIN)
        );
        assert_common_headers(&headers, "405");
    });
}

#[test]
fn preflight_to_unknown_path_still_returns_204() {
    let app = app_open();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // CORS 層はルーティングより外側なので、未知パスでもプリフライトは 204。
        let req = preflight("/v1/no-such-route", LOOPBACK_ORIGIN, Some("GET"), None);
        let (status, headers, _) = send(&app, req).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert_eq!(
            hdr(&headers, "access-control-allow-origin").as_deref(),
            Some(LOOPBACK_ORIGIN)
        );
        assert_common_headers(&headers, "未知パス プリフライト");
    });
}

// ---------------------------------------------------------------------------
// ボディ上限 (413) vs 認証 (401)

#[test]
fn oversized_content_length_is_413_before_auth_regardless_of_key() {
    let app = app_auth();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let big = vec![b'a'; BODY_LIMIT + 1];
        for key in [None, Some(KEY)] {
            let req = post_messages(big.clone(), Some(big.len()), key);
            let (status, headers, body) = send(&app, req).await;
            assert_eq!(
                status,
                StatusCode::PAYLOAD_TOO_LARGE,
                "key={key:?}: 413 が 401 より先"
            );
            let v = json_of(&body);
            assert_anthropic_error(&v, "invalid_request_error");
            let msg = v["error"]["message"].as_str().unwrap_or_default();
            assert!(
                msg.contains(&format!("{BODY_LIMIT} bytes")),
                "上限 byte 数を明示: {msg}"
            );
            assert_eq!(
                hdr(&headers, "content-type").as_deref(),
                Some("application/json"),
                "エラー封筒の content-type は charset 無し"
            );
            assert_common_headers(&headers, "413");
        }
    });
}

#[test]
fn declared_length_over_limit_with_small_body_is_413() {
    let app = app_auth();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // 実際のボディは小さくても、宣言された Content-Length が上限超過なら
        // 読み込み前に 413 (早期判定)。認証は評価されない。
        let req = post_messages(b"{}".to_vec(), Some(BODY_LIMIT + 100), None);
        let (status, _, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "本文: {body}");
        assert_anthropic_error(&json_of(&body), "invalid_request_error");
    });
}

#[test]
fn body_without_content_length_over_limit_is_413_json_envelope() {
    let app = app_open();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // Content-Length 無し: 抽出器 (Bytes) の DefaultBodyLimit が打ち切り、
        // 外側の body_limit_middleware が平文 413 を JSON 封筒へ包む。
        let big = vec![b'{'; BODY_LIMIT * 4];
        let req = post_messages(big, None, None);
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_anthropic_error(&json_of(&body), "invalid_request_error");
        assert_eq!(
            hdr(&headers, "content-type").as_deref(),
            Some("application/json"),
            "平文 413 を JSON 封筒へ置換する: {body}"
        );
        assert_common_headers(&headers, "chunked 413");
    });
}

#[test]
fn body_at_limit_reaches_auth_so_401_wins_over_413() {
    let app = app_auth();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // ちょうど上限のボディは 413 にしない。認証が先に走って 401 になる。
        let body = vec![b'a'; BODY_LIMIT];
        let req = post_messages(body.clone(), Some(body.len()), None);
        let (status, _, resp_body) = send(&app, req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "本文: {resp_body}");
        assert_anthropic_error(&json_of(&resp_body), "authentication_error");

        // 鍵があれば認証を通過し、不正 JSON の 400 に到達する (413 ではない)。
        let req = post_messages(body.clone(), Some(body.len()), Some(KEY));
        let (status, _, resp_body) = send(&app, req).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "本文: {resp_body}");
        assert_anthropic_error(&json_of(&resp_body), "invalid_request_error");
    });
}

// ---------------------------------------------------------------------------
// x-request-id の伝播 / 生成

#[test]
fn client_request_id_is_sanitized_and_echoed_on_204_and_413() {
    let app = app_open();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // 引用符・バックスラッシュ・空白は落ち、印字可能 ASCII だけが残る。
        let mut req = preflight("/v1/messages", LOOPBACK_ORIGIN, Some("POST"), None);
        req.headers_mut().insert(
            "x-request-id",
            "ab\"c\\d e".parse().unwrap(),
        );
        let (status, headers, _) = send(&app, req).await;
        assert_eq!(status, StatusCode::NO_CONTENT);
        assert_eq!(hdr(&headers, "x-request-id").as_deref(), Some("abcde"));

        // 413 でも同じ ID が返る (log_requests はボディ上限より外側)。
        let big = vec![b'a'; BODY_LIMIT + 1];
        let mut req = post_messages(big.clone(), Some(big.len()), None);
        req.headers_mut()
            .insert("x-request-id", "client-rid-1454".parse().unwrap());
        let (status, headers, _) = send(&app, req).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            hdr(&headers, "x-request-id").as_deref(),
            Some("client-rid-1454")
        );
    });
}

#[test]
fn missing_or_unusable_request_id_is_generated_on_errors() {
    let app = app_open();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let cases: Vec<(&str, Request<Body>)> = vec![
            (
                "404",
                Request::builder()
                    .method("GET")
                    .uri("/nope-1454")
                    .body(Body::empty())
                    .unwrap(),
            ),
            (
                "413",
                post_messages(vec![b'a'; BODY_LIMIT + 1], Some(BODY_LIMIT + 1), None),
            ),
            (
                "401",
                // 認証付きサーバで鍵なし POST。ID 未指定 → 生成。
                post_messages(b"{}".to_vec(), None, None),
            ),
        ];
        for (ctx, req) in cases {
            let target = if ctx == "401" { app_auth() } else { app.clone() };
            let (_, headers, _) = send(&target, req).await;
            let rid = hdr(&headers, "x-request-id")
                .unwrap_or_else(|| panic!("{ctx}: 生成 ID が無い"));
            assert!(
                rid.starts_with("req-") && rid.len() > 4,
                "{ctx}: 生成 ID の形が違う: {rid}"
            );
            assert!(rid.len() <= MAX_REQUEST_ID_LEN, "{ctx}: {rid}");
        }
    });
}

// ---------------------------------------------------------------------------
// セキュリティヘッダ / CORS ヘッダのエラー応答への付与

#[test]
fn security_and_cors_headers_are_present_on_error_responses() {
    let app = app_auth();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        // 404 (未知ルート): nosniff / x-request-id は付くが CORS は Origin 無し
        // なので付かない。
        let req = Request::builder()
            .method("GET")
            .uri("/nope-1454")
            .body(Body::empty())
            .unwrap();
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_openai_error(&json_of(&body), "unknown_route");
        assert_common_headers(&headers, "404");
        assert!(hdr(&headers, "access-control-allow-origin").is_none());

        // 401: 許可オリジン付きなら CORS ヘッダ (Allow-Origin / Expose-Headers) も
        // 付く。推論パスなので cache-control: no-store。
        let mut req = post_messages(b"{}".to_vec(), None, None);
        req.headers_mut()
            .insert("origin", LOOPBACK_ORIGIN.parse().unwrap());
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_anthropic_error(&json_of(&body), "authentication_error");
        assert_eq!(
            hdr(&headers, "access-control-allow-origin").as_deref(),
            Some(LOOPBACK_ORIGIN)
        );
        let expose = hdr(&headers, "access-control-expose-headers").unwrap_or_default();
        assert!(expose.contains("x-request-id"), "expose: {expose}");
        assert_eq!(hdr(&headers, "cache-control").as_deref(), Some("no-store"));
        assert_common_headers(&headers, "401+cors");

        // 413: ボディ上限層より外側の CORS / ヘッダ衛生がエラーにも掛かる。
        let big = vec![b'a'; BODY_LIMIT + 1];
        let mut req = post_messages(big.clone(), Some(big.len()), Some(KEY));
        req.headers_mut()
            .insert("origin", LOOPBACK_ORIGIN.parse().unwrap());
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_anthropic_error(&json_of(&body), "invalid_request_error");
        assert_eq!(
            hdr(&headers, "access-control-allow-origin").as_deref(),
            Some(LOOPBACK_ORIGIN)
        );
        assert_eq!(hdr(&headers, "cache-control").as_deref(), Some("no-store"));
        assert_common_headers(&headers, "413+cors");
    });
}
