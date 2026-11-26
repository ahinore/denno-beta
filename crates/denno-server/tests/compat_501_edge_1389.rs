//! task 1389: 未実装 OpenAI 経路 (`/v1/embeddings` / `/v1/responses`) の
//! 501 封筒について、HTTP メソッド・末尾スラッシュ・クエリ文字列・認証・CORS・
//! ヘッダ衛生の相互作用を追加で固定する。
//!
//! task 1051 (`compat_501_1051.rs`) が「封筒の形」と「代表的な経路一致」を
//! 固定しているのに対し、本ファイルは組み合わせを広げる:
//!
//! - GET / POST / PUT / DELETE のすべてで 501 (405 にしない)。
//! - クエリ文字列はルーティングに影響しない (`?x=1` 付きでも 501、
//!   message にはクエリを含めない)。末尾スラッシュ 1 個 + クエリも同じ経路。
//! - `/v1/embeddings/foo` `/v2/embeddings` などの紛らわしい経路は
//!   クエリ付きでも 404 `unknown_route` のまま。
//! - API キー設定時でも 501 (401 にしない)。認証が有効なことは別経路
//!   (`/v1/models`) の 401 で確かめる (テストが空振りしていないことの確認)。
//! - CORS: 許可オリジンの単純リクエストには `Allow-Origin` が付き、
//!   プリフライトは 501 より先に 204 で完結する。拒否オリジンには付かない。
//! - ヘッダ衛生: 501 / 404 の双方に `nosniff` と `x-request-id` が付き、
//!   エラー封筒の `content-type` は charset 無しの `application/json`。
//! - 501 の message にリクエスト本文やクエリ値が混ざらない。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler, MAX_REQUEST_ID_LEN};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

/// 501 を返す 2 経路 (実装側の一覧と一致することは 1051 が固定済み)。
const UNIMPLEMENTED: [&str; 2] = ["/v1/embeddings", "/v1/responses"];

const CANONICAL: &str = "denno-test";

/// ループバック既定 CORS が許可するオリジン。
const ALLOWED_ORIGIN: &str = "http://localhost:5173";

/// ループバックでも明示許可でもないオリジン (127.0.0.1 以外のループバック)。
const DENIED_ORIGIN: &str = "http://127.0.0.2:8080";

fn app_with_key(api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, CANONICAL, None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: CANONICAL.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

fn app() -> Router {
    app_with_key(None)
}

struct Resp {
    status: StatusCode,
    headers: HeaderMap,
    body: Value,
    raw: String,
}

impl Resp {
    fn hdr(&self, name: &str) -> Option<&str> {
        self.headers.get(name).and_then(|v| v.to_str().ok())
    }

    fn err(&self) -> &Value {
        &self.body["error"]
    }

    fn message(&self) -> &str {
        self.err()["message"].as_str().unwrap_or("")
    }
}

async fn send_with_headers(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<&str>,
    extra: &[(&str, &str)],
) -> Resp {
    let mut req = Request::builder().method(method).uri(uri);
    for (name, value) in extra {
        req = req.header(*name, *value);
    }
    let body = match body {
        None => Body::empty(),
        Some(b) => {
            req = req.header(header::CONTENT_TYPE, "application/json");
            Body::from(b.to_string())
        }
    };
    let resp = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let raw = String::from_utf8_lossy(&bytes).to_string();
    let body: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Resp { status, headers, body, raw }
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<&str>) -> Resp {
    send_with_headers(app, method, uri, body, &[]).await
}

/// 501 封筒の形を検査する。`expected_path` は message に入るべき
/// クエリ抜きのパス (末尾スラッシュはそのまま含める)。
fn assert_501(resp: &Resp, method: &str, expected_path: &str) {
    assert_eq!(
        resp.status,
        StatusCode::NOT_IMPLEMENTED,
        "{method} {expected_path} が {} を返した: {}",
        resp.status.as_u16(),
        resp.raw
    );
    assert_eq!(
        resp.hdr("content-type"),
        Some("application/json"),
        "{method} {expected_path} の content-type が封筒の形と違う"
    );
    let err = resp.err();
    assert!(err.is_object(), "{method} {expected_path} の本文が封筒でない: {}", resp.raw);
    assert_eq!(err["type"], "api_error", "{method} {expected_path}: {err}");
    assert_eq!(err["code"], "not_implemented", "{method} {expected_path}: {err}");
    assert_eq!(err["param"], Value::Null, "{method} {expected_path}: {err}");
    let msg = resp.message();
    assert!(
        msg.starts_with(&format!("{method} {expected_path} is not implemented")),
        "{method} {expected_path} の message がメソッド + パスで始まらない: {msg}"
    );
}

/// 404 `unknown_route` 封筒を検査する。
fn assert_404(resp: &Resp, method: &str, expected_path: &str) {
    assert_eq!(
        resp.status,
        StatusCode::NOT_FOUND,
        "{method} {expected_path} が {} を返した: {}",
        resp.status.as_u16(),
        resp.raw
    );
    assert_eq!(resp.err()["code"], "unknown_route", "{method} {expected_path}: {}", resp.raw);
    assert_eq!(resp.err()["type"], "invalid_request_error", "{method} {expected_path}");
    assert_eq!(
        resp.message(),
        format!("Unknown route: {method} {expected_path}"),
        "{method} {expected_path} の 404 message が変わった"
    );
}

fn assert_hygiene(resp: &Resp, label: &str) {
    assert_eq!(
        resp.hdr("x-content-type-options"),
        Some("nosniff"),
        "{label} に nosniff が無い"
    );
    let rid = resp.hdr("x-request-id").unwrap_or_else(|| panic!("{label} に x-request-id が無い"));
    assert!(!rid.is_empty(), "{label} の x-request-id が空");
}

// ---------------------------------------------------------------------------
// HTTP メソッド
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_post_put_delete_are_all_501_not_405() {
    let app = app();
    for path in UNIMPLEMENTED {
        for method in ["GET", "POST", "PUT", "DELETE"] {
            let resp = send(&app, method, path, None).await;
            assert_501(&resp, method, path);
            // 経路が無いので 405 は立てない (未対応メソッドの封筒と混同しない)。
            assert_ne!(resp.status, StatusCode::METHOD_NOT_ALLOWED, "{method} {path}");
        }
    }
}

#[tokio::test]
async fn body_is_not_required_nor_parsed_for_501() {
    let app = app();
    for path in UNIMPLEMENTED {
        for body in [None, Some("{}"), Some("not json at all"), Some("{\"stream\":true}")] {
            let resp = send(&app, "POST", path, body).await;
            assert_501(&resp, "POST", path);
        }
    }
}

// ---------------------------------------------------------------------------
// クエリ文字列と末尾スラッシュ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn query_strings_do_not_change_routing_or_the_message() {
    let app = app();
    for path in UNIMPLEMENTED {
        for query in ["?x=1", "?model=text-embedding-3-small&input=hello", "?"] {
            let uri = format!("{path}{query}");
            let resp = send(&app, "GET", &uri, None).await;
            // message にはクエリを混ぜない (ログ/UI に値が流れない)。
            assert_501(&resp, "GET", path);
            assert!(
                !resp.message().contains('?'),
                "501 の message にクエリが混ざった: {}",
                resp.message()
            );
        }
    }
    // 本文つき POST + クエリでも同じ。
    for path in UNIMPLEMENTED {
        let resp = send(&app, "POST", &format!("{path}?trace=1"), Some("{}")).await;
        assert_501(&resp, "POST", path);
    }
}

#[tokio::test]
async fn trailing_slash_one_is_501_two_is_404_even_with_query() {
    let app = app();
    for path in UNIMPLEMENTED {
        let one = format!("{path}/");
        let resp = send(&app, "POST", &one, Some("{}")).await;
        assert_501(&resp, "POST", &one);

        let with_query = format!("{path}/?x=1");
        let resp = send(&app, "GET", &with_query, None).await;
        assert_501(&resp, "GET", &one);

        let two = format!("{path}//");
        let resp = send(&app, "POST", &two, Some("{}")).await;
        assert_404(&resp, "POST", &two);
    }
}

// ---------------------------------------------------------------------------
// 紛らわしい兄弟経路は 404 のまま
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_siblings_stay_404_with_and_without_query() {
    let app = app();
    for path in [
        "/v1/embedding",
        "/v1/embeddings/foo",
        "/v1/embeddings/foo/bar",
        "/v1/responses/foo",
        "/v2/embeddings",
        "/v2/responses",
        "/api/embeddings",
    ] {
        let resp = send(&app, "POST", path, Some("{}")).await;
        assert_404(&resp, "POST", path);

        let with_query = format!("{path}?x=1");
        let resp = send(&app, "GET", &with_query, None).await;
        assert_404(&resp, "GET", path);
        assert!(
            !resp.message().contains('?'),
            "404 の message にクエリが混ざった: {}",
            resp.message()
        );
    }
}

// ---------------------------------------------------------------------------
// 認証との相互作用
// ---------------------------------------------------------------------------

#[tokio::test]
async fn api_key_setup_keeps_501_and_does_not_leak_into_401() {
    let app = app_with_key(Some("secret-key"));

    // 認証が本当に有効なこと: 実装済み経路は鍵なしで 401。
    let models = send(&app, "GET", "/v1/models", None).await;
    assert_eq!(
        models.status,
        StatusCode::UNAUTHORIZED,
        "認証が有効でないと、このテストは 501 の確認として空振りする: {}",
        models.raw
    );

    let wrong_bearer = [("authorization", "Bearer wrong-key")];
    let wrong_x_api_key = [("x-api-key", "wrong-key")];
    let right_bearer = [("authorization", "Bearer secret-key")];
    let right_x_api_key = [("x-api-key", "secret-key")];

    for path in UNIMPLEMENTED {
        for method in ["GET", "POST", "PUT", "DELETE"] {
            // 資格情報なし / 誤り / 正しい、のどれでも経路が無いことが先。
            let resp = send(&app, method, path, None).await;
            assert_501(&resp, method, path);
            for headers in [&wrong_bearer[..], &wrong_x_api_key[..], &right_bearer[..], &right_x_api_key[..]] {
                let resp = send_with_headers(&app, method, path, None, headers).await;
                assert_501(&resp, method, path);
                assert_ne!(resp.status, StatusCode::UNAUTHORIZED, "{method} {path} {headers:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

#[tokio::test]
async fn allowed_origin_gets_cors_headers_on_501_and_denied_origin_does_not() {
    let app = app();
    for path in UNIMPLEMENTED {
        let resp = send_with_headers(&app, "POST", path, Some("{}"), &[("origin", ALLOWED_ORIGIN)]).await;
        assert_501(&resp, "POST", path);
        assert_eq!(
            resp.hdr("access-control-allow-origin"),
            Some(ALLOWED_ORIGIN),
            "{path} の 501 に CORS が掛かっていない"
        );
        assert_eq!(resp.hdr("vary"), Some("Origin"), "{path}: Vary が無い");
        let expose = resp.hdr("access-control-expose-headers").unwrap_or("");
        assert!(
            expose.contains("x-request-id"),
            "{path} の expose-headers に x-request-id が無い: {expose:?}"
        );

        let denied = send_with_headers(&app, "POST", path, Some("{}"), &[("origin", DENIED_ORIGIN)]).await;
        assert_501(&denied, "POST", path);
        assert_eq!(
            denied.hdr("access-control-allow-origin"),
            None,
            "{path} が拒否オリジンに Allow-Origin を返した"
        );
    }
}

#[tokio::test]
async fn preflight_is_answered_before_the_501_fallback() {
    let app = app();
    for path in UNIMPLEMENTED {
        let resp = send_with_headers(
            &app,
            "OPTIONS",
            path,
            None,
            &[
                ("origin", ALLOWED_ORIGIN),
                ("access-control-request-method", "POST"),
                ("access-control-request-headers", "content-type, x-api-key"),
            ],
        )
        .await;
        assert_eq!(
            resp.status,
            StatusCode::NO_CONTENT,
            "{path} のプリフライトが {} を返した: {}",
            resp.status.as_u16(),
            resp.raw
        );
        assert_eq!(resp.hdr("access-control-allow-origin"), Some(ALLOWED_ORIGIN));
        let methods = resp.hdr("access-control-allow-methods").unwrap_or("");
        assert!(methods.contains("POST"), "{path} の allow-methods: {methods:?}");
        assert_eq!(
            resp.hdr("access-control-allow-headers"),
            Some("content-type, x-api-key"),
            "{path} の allow-headers が要求を反映していない"
        );
        assert!(resp.hdr("access-control-max-age").is_some(), "{path} に max-age が無い");
        assert_eq!(resp.hdr("x-content-type-options"), Some("nosniff"));
        assert!(resp.hdr("x-request-id").is_some(), "{path} の 204 に x-request-id が無い");
        assert!(resp.raw.is_empty(), "{path} のプリフライトに本文がある: {}", resp.raw);
    }

    // 拒否オリジンのプリフライトは 204 のまま CORS ヘッダを付けない。
    let denied = send_with_headers(
        &app,
        "OPTIONS",
        "/v1/embeddings",
        None,
        &[("origin", DENIED_ORIGIN), ("access-control-request-method", "POST")],
    )
    .await;
    assert_eq!(denied.status, StatusCode::NO_CONTENT);
    assert_eq!(denied.hdr("access-control-allow-origin"), None);
}

// ---------------------------------------------------------------------------
// ヘッダ衛生
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nosniff_and_request_id_cover_501_and_404() {
    let app = app();
    for path in UNIMPLEMENTED {
        let resp = send(&app, "POST", path, Some("{}")).await;
        assert_501(&resp, "POST", path);
        assert_hygiene(&resp, path);
        let rid = resp.hdr("x-request-id").unwrap();
        // 生成 ID の形 (`req-` + 12 hex + 8 hex) を緩く固定する。
        assert!(rid.starts_with("req-"), "{path} の生成 ID 形が違う: {rid}");
        assert_eq!(rid.len(), 24, "{path} の生成 ID 長が違う: {rid}");
        assert!(
            rid[4..].chars().all(|c| c.is_ascii_hexdigit()),
            "{path} の生成 ID に hex 以外が混ざる: {rid}"
        );

        let near = format!("{path}/foo");
        let resp = send(&app, "POST", &near, Some("{}")).await;
        assert_404(&resp, "POST", &near);
        assert_hygiene(&resp, &near);
    }
}

#[tokio::test]
async fn client_request_id_is_echoed_sanitized_and_truncated() {
    let app = app();

    let resp = send_with_headers(
        &app,
        "POST",
        "/v1/embeddings",
        Some("{}"),
        &[("x-request-id", "client-rid-42")],
    )
    .await;
    assert_501(&resp, "POST", "/v1/embeddings");
    assert_eq!(resp.hdr("x-request-id"), Some("client-rid-42"));

    // 空白・引用符・バックスラッシュは落とす。
    let resp = send_with_headers(
        &app,
        "POST",
        "/v1/responses",
        Some("{}"),
        &[("x-request-id", "a b\"c\\d")],
    )
    .await;
    assert_501(&resp, "POST", "/v1/responses");
    assert_eq!(resp.hdr("x-request-id"), Some("abcd"));

    // 長すぎる ID は MAX_REQUEST_ID_LEN byte で切る。
    let long = "z".repeat(MAX_REQUEST_ID_LEN + 32);
    let resp = send_with_headers(
        &app,
        "POST",
        "/v1/embeddings",
        Some("{}"),
        &[("x-request-id", long.as_str())],
    )
    .await;
    assert_501(&resp, "POST", "/v1/embeddings");
    assert_eq!(resp.hdr("x-request-id"), Some("z".repeat(MAX_REQUEST_ID_LEN).as_str()));
}

#[tokio::test]
async fn error_envelope_has_no_charset_but_success_json_does() {
    let app = app();
    for path in UNIMPLEMENTED {
        let resp = send(&app, "GET", path, None).await;
        assert_501(&resp, "GET", path);
        assert_eq!(
            resp.hdr("content-type"),
            Some("application/json"),
            "{path}: エラー封筒に charset が付いている"
        );
    }
    // 対照: 2xx の JSON は charset 補完の対象。
    let ok = send(&app, "GET", "/v1/models", None).await;
    assert_eq!(ok.status, StatusCode::OK, "{}", ok.raw);
    assert_eq!(
        ok.hdr("content-type"),
        Some("application/json; charset=utf-8"),
        "2xx の JSON に charset が付いていない"
    );
}

#[tokio::test]
async fn error_501_message_does_not_echo_the_body_or_query_value() {
    let app = app();
    let secret = "SUPER_SECRET_INPUT_VALUE";
    let body = format!(r#"{{"input":"{secret}"}}"#);
    for path in UNIMPLEMENTED {
        let resp = send(&app, "POST", &format!("{path}?token={secret}"), Some(&body)).await;
        assert_501(&resp, "POST", path);
        assert!(
            !resp.message().contains(secret),
            "{path} の message がリクエスト値を反響している: {}",
            resp.message()
        );
        assert!(
            resp.err().get("param").is_some(),
            "{path} の封筒から param キーが消えた"
        );
    }
}
