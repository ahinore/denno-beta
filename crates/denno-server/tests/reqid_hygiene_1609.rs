//! task 1609: `x-request-id` の無害化・長さ制限・セキュリティヘッダの網羅テスト。
//!
//! 対象は `src/lib.rs` の `sanitize_request_id_bytes` と `log_requests` /
//! `header_hygiene` の契約:
//!
//! - 制御文字 (0x00-0x1F, 0x7F)・空白・非 ASCII は落ちる。
//! - CRLF 注入 (`\r\n` で別ヘッダ/リクエストを偽装する試み) は改行が消える。
//! - 二重引用符とバックスラッシュは落ちる。
//! - [`MAX_REQUEST_ID_LEN`] byte で切る (切る判定は「残した文字数」)。
//! - 有効文字が残らない (空ヘッダ含む) 場合は `req-` 形式の生成 ID に切り替わる。
//! - 200/400/401/404/413/429/501/503 のすべてで `x-content-type-options: nosniff`
//!   と `x-request-id` が 1 個ずつ付き、`server` / `x-powered-by` を漏らさない。
//!
//! 実装 (`src/lib.rs`) は変更しない。CPU のみ・決定論的。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::{RateLimitConfig, RateLimiter};
use denno_server::{
    sanitize_request_id, sanitize_request_id_bytes, AppState, CorsConfig, EngineInfo, Scheduler,
    MAX_REQUEST_ID_LEN,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// fixture (既存テストと同じ形)

fn build_state(api_key: Option<&str>, ready: bool) -> Arc<AppState> {
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
    denno_server::router(build_state(None, true))
}

fn app_auth() -> Router {
    denno_server::router(build_state(Some("secret-key"), true))
}

fn app_not_ready() -> Router {
    denno_server::router(build_state(None, false))
}

fn app_small_body() -> Router {
    denno_server::router_with_body_limit(
        build_state(None, true),
        CorsConfig::default(),
        RateLimiter::disabled(),
        64,
    )
}

fn app_rate_limited() -> Router {
    denno_server::router_with_rate_limit(
        build_state(None, true),
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

fn get_with_rid(uri: &str, rid: HeaderValue) -> Request<Body> {
    let mut req = get(uri);
    req.headers_mut().insert("x-request-id", rid);
    req
}

fn post_raw(uri: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn post_json(uri: &str, body: &Value) -> Request<Body> {
    post_raw(uri, &body.to_string())
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

fn all<'a>(headers: &'a HeaderMap, name: &str) -> Vec<&'a str> {
    headers
        .get_all(name)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .collect()
}

fn one<'a>(headers: &'a HeaderMap, name: &str, label: &str) -> &'a str {
    let vals = all(headers, name);
    assert_eq!(
        vals.len(),
        1,
        "{label}: {name} が {} 個ある: {vals:?}",
        vals.len()
    );
    vals[0]
}

/// 生成 ID (`req-` + 12 hex + 8 hex) の形か。
fn assert_generated_rid(id: &str, label: &str) {
    assert!(id.starts_with("req-"), "{label}: 生成 ID の形でない: {id:?}");
    assert_eq!(id.len(), 24, "{label}: 生成 ID の長さ: {id:?}");
    assert!(
        id["req-".len()..].chars().all(|c| c.is_ascii_hexdigit()),
        "{label}: 生成 ID が hex でない: {id:?}"
    );
}

/// 応答の `x-request-id` がヘッダ値として安全 (可視 ASCII のみ) か。
fn assert_safe_rid(id: &str, label: &str) {
    assert!(!id.is_empty(), "{label}: x-request-id が空");
    assert!(
        id.len() <= MAX_REQUEST_ID_LEN,
        "{label}: x-request-id が {} byte (上限 {MAX_REQUEST_ID_LEN})",
        id.len()
    );
    assert!(
        id.bytes().all(|b| (0x21..=0x7e).contains(&b)),
        "{label}: x-request-id に可視 ASCII 以外が混じる: {id:?}"
    );
    assert!(
        !id.contains('\r') && !id.contains('\n') && !id.contains('"') && !id.contains('\\'),
        "{label}: x-request-id に危険な文字: {id:?}"
    );
}

/// 全応答に共通で期待するヘッダ衛生。
fn assert_security_headers(headers: &HeaderMap, label: &str) {
    assert_eq!(
        one(headers, "x-content-type-options", label),
        "nosniff",
        "{label}: x-content-type-options は nosniff のみ"
    );
    assert_safe_rid(one(headers, "x-request-id", label), label);
    for name in [
        "content-type",
        "x-request-id",
        "cache-control",
        "x-content-type-options",
        "x-accel-buffering",
        "content-length",
        "x-denno-ttft-ms",
        "x-denno-decode-tps",
    ] {
        let n = headers.get_all(name).iter().count();
        assert!(n <= 1, "{label}: {name} が {n} 個重複している");
    }
    for v in all(headers, "cache-control") {
        let lc = v.to_ascii_lowercase();
        for bad in [
            "public",
            "max-age",
            "s-maxage",
            "immutable",
            "stale-while-revalidate",
        ] {
            assert!(
                !lc.contains(bad),
                "{label}: キャッシュ可能な cache-control = {lc:?}"
            );
        }
    }
    let ct = one(headers, "content-type", label).to_ascii_lowercase();
    assert!(
        ct.starts_with("application/json") || ct.starts_with("text/plain"),
        "{label}: 想定外の content-type = {ct:?}"
    );
    assert!(headers.get("server").is_none(), "{label}: server ヘッダ");
    assert!(
        headers.get("x-powered-by").is_none(),
        "{label}: x-powered-by ヘッダ"
    );
    // `x-request-id` の生バイトが ASCII として読めること (obs-text でない)。
    headers
        .get("x-request-id")
        .unwrap()
        .to_str()
        .unwrap_or_else(|_| panic!("{label}: x-request-id が ASCII でない"));
}

// ---------------------------------------------------------------------------
// 1. 無害化の純関数: 制御文字・CRLF・引用符・バックスラッシュ

#[test]
fn sanitize_removes_all_c0_controls_del_and_space() {
    // 0x00-0x1F と 0x7F は全て落ちる。
    for b in 0x00u8..=0x1fu8 {
        assert_eq!(
            sanitize_request_id_bytes(&[b'a', b, b'b']).as_deref(),
            Some("ab"),
            "制御文字 0x{b:02x} が残っている"
        );
    }
    assert_eq!(
        sanitize_request_id_bytes(&[b'a', 0x7f, b'b']).as_deref(),
        Some("ab"),
        "DEL (0x7f) が残っている"
    );
    // 空白 (0x20) は落ちる。
    assert_eq!(
        sanitize_request_id(" a b\tc ").as_deref(),
        Some("abc"),
        "空白は落ちる"
    );
    // 制御文字だけなら None (呼び側が生成 ID を振る)。
    assert_eq!(sanitize_request_id_bytes(&[0x00, 0x01, 0x1f, 0x7f]), None);
    assert_eq!(sanitize_request_id("\t\r\n"), None);
}

#[test]
fn sanitize_neutralizes_crlf_injection_payloads() {
    // 生の CRLF は HTTP ヘッダ層が拒否するが、防御のため無害化関数も改行を落とす。
    let payloads: [&str; 5] = [
        "abc\r\nX-Injected: yes",
        "abc\r\nSet-Cookie: sid=1",
        "abc\rdef",
        "abc\ndef",
        "abc\r\n\r\nGET /admin HTTP/1.1",
    ];
    for raw in payloads {
        let got = sanitize_request_id(raw)
            .unwrap_or_else(|| panic!("可視文字が残るはず: {raw:?}"));
        assert!(
            !got.contains('\r') && !got.contains('\n'),
            "CRLF が残った: raw={raw:?} got={got:?}"
        );
        assert_safe_rid(&got, "crlf");
    }
    // 改行だけ落ち、可視の `X-Injected: yes` は連結されて残る (注入にはならない)。
    assert_eq!(
        sanitize_request_id("abc\r\nX-Injected: yes").as_deref(),
        Some("abcX-Injected:yes")
    );
}

#[test]
fn sanitize_strips_quotes_backslashes_and_non_ascii() {
    // 二重引用符・バックスラッシュは落ちる。
    assert_eq!(sanitize_request_id(r#"a"b\c"#).as_deref(), Some("abc"));
    assert_eq!(sanitize_request_id("\"\"\"\\\\\\"), None);
    assert_eq!(sanitize_request_id("a\"\"b\\\\c").as_deref(), Some("abc"));
    // 非 ASCII (日本語) は全て落ちる。
    assert_eq!(sanitize_request_id("\u{65e5}\u{672c}\u{8a9e}"), None);
    // obs-text (0x80-0xFF) は byte 版で落ちる。
    assert_eq!(sanitize_request_id_bytes(&[0x80, 0xfe, 0xff]), None);
    assert_eq!(sanitize_request_id_bytes(&[b'a', 0x80, b'b']).as_deref(), Some("ab"));
    // 全 256 byte を一括で流しても、残るのは 0x21..=0x7E から `"` `\` を除いたものだけ。
    let all_bytes: Vec<u8> = (0u8..=255).collect();
    let got = sanitize_request_id_bytes(&all_bytes).expect("可視文字は残る");
    assert!(got.len() <= MAX_REQUEST_ID_LEN);
    assert!(got
        .bytes()
        .all(|b| (0x21..=0x7e).contains(&b) && b != b'"' && b != b'\\'));
}

#[test]
fn sanitize_clamps_to_max_len_counting_kept_bytes_only() {
    // 200 byte の可視文字は 128 byte で切る。
    let long = "a".repeat(200);
    assert_eq!(
        sanitize_request_id(&long),
        Some("a".repeat(MAX_REQUEST_ID_LEN))
    );
    // 落とす文字 (引用符) を挟んでも、残した文字数で 128 に収める。
    let mixed = format!("{}{}{}", "a".repeat(100), "\"".repeat(50), "b".repeat(100));
    let got = sanitize_request_id(&mixed).expect("some");
    assert_eq!(got.len(), MAX_REQUEST_ID_LEN);
    assert_eq!(got, format!("{}{}", "a".repeat(100), "b".repeat(28)));
    // 落とす文字だけで 128 を超えても生成 ID 側へ回る。
    assert_eq!(sanitize_request_id(&"\"".repeat(300)), None);
}

// ---------------------------------------------------------------------------
// 2. ヘッダ層: CRLF はそもそも HeaderValue にできない

#[test]
fn header_value_rejects_raw_crlf() {
    for raw in [&b"abc\r\ndef"[..], &b"abc\ndef"[..], &b"abc\rdef"[..]] {
        assert!(
            HeaderValue::from_bytes(raw).is_err(),
            "CR/LF を含むヘッダ値が通ってしまった: {raw:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. ミドルウェア: 空ヘッダ・空白のみ・obs-text のみは生成 ID に切り替わる

#[tokio::test]
async fn empty_and_stripped_header_values_are_regenerated() {
    let app = app();
    let cases: [(&str, HeaderValue); 4] = [
        ("空ヘッダ", HeaderValue::from_static("")),
        ("空白のみ", HeaderValue::from_static("   \t ")),
        (
            "引用符のみ",
            HeaderValue::from_str(&"\"".repeat(32)).unwrap(),
        ),
        (
            "obs-text のみ",
            HeaderValue::from_bytes(&[0x80, 0xff, 0xfe]).unwrap(),
        ),
    ];
    for (label, value) in cases {
        let (status, headers, _) = send(&app, get_with_rid("/health", value)).await;
        assert_eq!(status, StatusCode::OK, "{label}");
        let id = one(&headers, "x-request-id", label);
        assert_generated_rid(id, label);
        assert_safe_rid(id, label);
    }
}

// ---------------------------------------------------------------------------
// 4. ミドルウェア: 引用符・バックスラッシュ・空白は落として引き継ぐ

#[tokio::test]
async fn messy_but_salvageable_header_is_sanitized_and_echoed() {
    let app = app();
    let raw = r#"  "abc" \def\  "#;
    let (status, headers, _) =
        send(&app, get_with_rid("/health", HeaderValue::from_str(raw).unwrap())).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(one(&headers, "x-request-id", "sanitize"), "abcdef");
    assert_safe_rid(one(&headers, "x-request-id", "sanitize"), "sanitize");
}

// ---------------------------------------------------------------------------
// 5. ミドルウェア: 128 byte 超はクランプする

#[tokio::test]
async fn overlong_header_is_clamped_at_max_len() {
    let app = app();
    let raw = "z".repeat(MAX_REQUEST_ID_LEN + 64);
    let (status, headers, _) =
        send(&app, get_with_rid("/health", HeaderValue::from_str(&raw).unwrap())).await;
    assert_eq!(status, StatusCode::OK);
    let id = one(&headers, "x-request-id", "clamp");
    assert_eq!(id.len(), MAX_REQUEST_ID_LEN);
    assert_eq!(id, "z".repeat(MAX_REQUEST_ID_LEN));
    assert_safe_rid(id, "clamp");
}

// ---------------------------------------------------------------------------
// 6. セキュリティヘッダを 200/400/401/404/413/429/501/503 で確認

#[tokio::test]
async fn security_headers_hold_across_200_and_error_statuses() {
    // 200
    let (s, h, _) = send(&app(), get("/health")).await;
    assert_eq!(s, StatusCode::OK);
    assert_security_headers(&h, "200 /health");

    // 400 (JSON 構文エラー)
    let (s, h, _) = send(&app(), post_raw("/v1/messages", "{not json")).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_security_headers(&h, "400 malformed json");

    // 401 (認証ありのルーター)
    let (s, h, _) = send(&app_auth(), get("/v1/models")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    assert_security_headers(&h, "401 unauthorized");

    // 404 (未知ルート)
    let (s, h, _) = send(&app(), get("/definitely-not-a-route-1609")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert_security_headers(&h, "404 unknown route");

    // 413 (ボディ上限 64 byte)
    let big = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "x".repeat(256)}],
    });
    let (s, h, _) = send(&app_small_body(), post_json("/v1/messages", &big)).await;
    assert_eq!(s, StatusCode::PAYLOAD_TOO_LARGE);
    assert_security_headers(&h, "413 payload too large");

    // 429 (rpm=1 / burst=1 を使い切る)
    let rl_app = app_rate_limited();
    let body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "temperature": 0.0,
    });
    let (s1, _, _) = send(&rl_app, post_json("/v1/messages", &body)).await;
    assert_eq!(s1, StatusCode::OK);
    let (s, h, _) = send(&rl_app, post_json("/v1/messages", &body)).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_security_headers(&h, "429 rate limited");

    // 501 (未実装 OpenAI 経路)
    let (s, h, _) = send(&app(), get("/v1/embeddings")).await;
    assert_eq!(s, StatusCode::NOT_IMPLEMENTED);
    assert_security_headers(&h, "501 unimplemented");

    // 503 (not ready)
    let (s, h, _) = send(&app_not_ready(), get("/ready")).await;
    assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE);
    assert_security_headers(&h, "503 not ready");
}

// ---------------------------------------------------------------------------
// 7. エラー応答でもクライアント指定 ID は同じ規則で無害化される

#[tokio::test]
async fn request_id_is_sanitized_on_error_responses() {
    let raw = r#" "client-1609" \ok "#;
    let expected = "client-1609ok";

    for (label, app, uri) in [
        ("404", app(), "/nope-1609"),
        ("501", app(), "/v1/embeddings"),
        ("401", app_auth(), "/v1/models"),
        ("503", app_not_ready(), "/ready"),
    ] {
        let (_, headers, _) =
            send(&app, get_with_rid(uri, HeaderValue::from_str(raw).unwrap())).await;
        let id = one(&headers, "x-request-id", label);
        assert_eq!(id, expected, "{label}: 無害化結果が違う");
        assert_safe_rid(id, label);
    }

    // 400: 不正ボディでもヘッダは同じ扱い。
    let mut req = post_raw("/v1/messages", "{not json");
    req.headers_mut()
        .insert("x-request-id", HeaderValue::from_str(raw).unwrap());
    let (s, headers, _) = send(&app(), req).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(
        one(&headers, "x-request-id", "400 sanitize"),
        expected,
        "400: 無害化結果が違う"
    );
    assert_safe_rid(one(&headers, "x-request-id", "400 sanitize"), "400 sanitize");
}
