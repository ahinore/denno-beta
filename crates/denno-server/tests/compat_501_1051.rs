//! Task 1051: 未実装の OpenAI 経路 (`/v1/embeddings` / `/v1/responses`) の
//! 501 封筒と互換性の監査。
//!
//! 背景: OpenAI SDK / LangChain / 各種ゲートウェイは起動時に `/v1/embeddings`
//! などを probe する。denno は生成モデル専用で embedding も Responses API も
//! 持たないので、これらは **経路としては既知・実装は無い** と答える必要がある。
//! 未知ルートの 404 (`unknown_route`) だと「URL の誤り」と解釈されるため、
//! `lib.rs` の fallback がこの 2 経路だけ 501 (`not_implemented`) に振り替える。
//!
//! ここで固定するもの:
//!
//! - 501 / `application/json` / `error.type = "api_error"` /
//!   `error.code = "not_implemented"` / `param` キーが `null` で存在
//! - メッセージに経路と代替 (`/v1/chat/completions`) が入る
//! - メソッド・本文 (`stream: true` を含む)・認証設定に依らず同じ封筒
//! - 末尾スラッシュ 1 個は同じ経路、それ以外の紛らわしいパスは 404 のまま
//! - ルータに `.route(...)` を足していない = `openapi_contract` の
//!   「ルート ⇔ 仕様」1:1 突き合わせを壊さない (仕様にも載っていない)
//! - ミドルウェア (x-request-id / nosniff / CORS) が 501 にも掛かる

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler, UNIMPLEMENTED_OPENAI_PATHS};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

/// テスト対象の経路 (この 2 本だけが 501 になる)。
const UNIMPLEMENTED: [&str; 2] = ["/v1/embeddings", "/v1/responses"];

const CANONICAL: &str = "denno-test";

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
    headers: axum::http::HeaderMap,
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
}

async fn send(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<&str>,
    origin: Option<&str>,
) -> Resp {
    let mut req = Request::builder().method(method).uri(uri);
    if let Some(o) = origin {
        req = req.header(header::ORIGIN, o);
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

/// 501 封筒の形をまとめて検査する。
fn assert_not_implemented(resp: &Resp, path: &str) {
    assert_eq!(
        resp.status,
        StatusCode::NOT_IMPLEMENTED,
        "{path} が {} を返した: {}",
        resp.status.as_u16(),
        resp.raw
    );
    let ct = resp.hdr("content-type").unwrap_or("");
    assert!(
        ct.starts_with("application/json"),
        "{path} の content-type = {ct:?}"
    );
    let err = resp.err();
    assert!(err.is_object(), "{path} の本文が封筒でない: {}", resp.raw);
    assert_eq!(err["type"], "api_error", "{path}: {err}");
    assert_eq!(err["code"], "not_implemented", "{path}: {err}");
    // `param` はキーとして必ず存在し、値は null。
    assert!(
        err.get("param").is_some(),
        "{path} の封筒に param キーが無い: {err}"
    );
    assert_eq!(err["param"], Value::Null, "{path}: {err}");
    let msg = err["message"].as_str().unwrap_or("");
    assert!(msg.contains(path), "{path} の message にパスが無い: {msg}");
    assert!(
        msg.contains("not implemented"),
        "{path} の message が未実装と書いていない: {msg}"
    );
    assert!(
        msg.contains("/v1/chat/completions"),
        "{path} の message に代替経路が無い: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 単体の封筒
// ---------------------------------------------------------------------------

#[tokio::test]
async fn embeddings_probe_returns_501_envelope() {
    let app = app();
    let resp = send(
        &app,
        "POST",
        "/v1/embeddings",
        Some(r#"{"model":"text-embedding-3-small","input":"hello"}"#),
        None,
    )
    .await;
    assert_not_implemented(&resp, "/v1/embeddings");
    // OpenAI SDK のエラー型として最低限必要な 4 キーだけを持つ。
    let err = resp.err();
    let keys: Vec<&String> = err.as_object().unwrap().keys().collect();
    assert_eq!(keys.len(), 4, "余計なキーがある: {err}");
    for k in ["message", "type", "param", "code"] {
        assert!(err.get(k).is_some(), "{k} が無い: {err}");
    }
}

#[tokio::test]
async fn responses_probe_returns_501_envelope() {
    let app = app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(r#"{"model":"denno-test","input":"hi"}"#),
        None,
    )
    .await;
    assert_not_implemented(&resp, "/v1/responses");
    let msg = resp.err()["message"].as_str().unwrap_or("");
    assert!(
        msg.starts_with("POST /v1/responses"),
        "message の先頭がメソッド + パスでない: {msg}"
    );
}

#[tokio::test]
async fn all_methods_are_501_not_405() {
    let app = app();
    for path in UNIMPLEMENTED {
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"] {
            let resp = send(&app, method, path, None, None).await;
            assert_not_implemented(&resp, path);
            let msg = resp.err()["message"].as_str().unwrap_or("");
            assert!(
                msg.starts_with(&format!("{method} {path}")),
                "{method} {path} の message = {msg}"
            );
        }
    }
}

#[tokio::test]
async fn streaming_probe_does_not_get_an_sse_stream() {
    let app = app();
    let resp = send(
        &app,
        "POST",
        "/v1/responses",
        Some(r#"{"model":"denno-test","input":"hi","stream":true}"#),
        None,
    )
    .await;
    assert_not_implemented(&resp, "/v1/responses");
    let ct = resp.hdr("content-type").unwrap_or("");
    assert!(
        !ct.contains("event-stream"),
        "stream:true の probe に SSE を返している: {ct:?}"
    );
    assert!(resp.body.is_object(), "本文が JSON でない: {}", resp.raw);
}

#[tokio::test]
async fn malformed_body_still_gets_501_not_400() {
    let app = app();
    // fallback は本文を読まないので、壊れた JSON でも 400 にはならない
    // (経路が無いことが先に確定する)。
    for path in UNIMPLEMENTED {
        let resp = send(&app, "POST", path, Some("{\"input\":"), None).await;
        assert_not_implemented(&resp, path);
    }
}

#[tokio::test]
async fn auth_key_does_not_turn_the_probe_into_401() {
    // 経路自体が無いので、キー設定時も 401 ではなく 501 を返す
    // (クライアントに「認証の問題」と誤解させない)。
    let app = app_with_key(Some("secret"));
    for path in UNIMPLEMENTED {
        let resp = send(&app, "POST", path, Some("{}"), None).await;
        assert_not_implemented(&resp, path);
    }
}

// ---------------------------------------------------------------------------
// 経路の一致規則
// ---------------------------------------------------------------------------

#[tokio::test]
async fn one_trailing_slash_is_the_same_route() {
    let app = app();
    for path in UNIMPLEMENTED {
        let resp = send(&app, "POST", &format!("{path}/"), Some("{}"), None).await;
        assert_eq!(
            resp.status,
            StatusCode::NOT_IMPLEMENTED,
            "{path}/ が {} を返した: {}",
            resp.status.as_u16(),
            resp.raw
        );
        assert_eq!(resp.err()["code"], "not_implemented");
    }
}

#[tokio::test]
async fn near_miss_paths_stay_404_unknown_route() {
    let app = app();
    // 部分一致で 501 にしない (実装の無い別経路を「既知」と誤って答えない)。
    for path in [
        "/v1/embedding",
        "/v1/embeddings/extra",
        "/v1/embeddings//",
        "/v1/responses/input",
        "/v2/embeddings",
    ] {
        let resp = send(&app, "POST", path, Some("{}"), None).await;
        assert_eq!(
            resp.status,
            StatusCode::NOT_FOUND,
            "{path} が {} を返した: {}",
            resp.status.as_u16(),
            resp.raw
        );
        assert_eq!(resp.err()["code"], "unknown_route", "{path}: {}", resp.raw);
        assert_eq!(resp.err()["type"], "invalid_request_error", "{path}");
    }
}

#[tokio::test]
async fn unrelated_unknown_route_is_still_404() {
    let app = app();
    let resp = send(&app, "GET", "/definitely-not-a-route", None, None).await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND);
    assert_eq!(resp.err()["code"], "unknown_route", "{}", resp.raw);
    assert!(
        resp.err()["message"]
            .as_str()
            .unwrap_or("")
            .starts_with("Unknown route: GET "),
        "404 の message 形が変わった: {}",
        resp.raw
    );
}

// ---------------------------------------------------------------------------
// ミドルウェアとの噛み合わせ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_id_nosniff_and_cors_still_apply_to_501() {
    let app = app();
    let resp = send(
        &app,
        "POST",
        "/v1/embeddings",
        Some("{}"),
        Some("http://localhost:5173"),
    )
    .await;
    assert_not_implemented(&resp, "/v1/embeddings");
    assert!(resp.hdr("x-request-id").is_some(), "x-request-id が無い");
    assert_eq!(
        resp.hdr("x-content-type-options"),
        Some("nosniff"),
        "ヘッダ衛生が 501 に掛かっていない"
    );
    assert_eq!(
        resp.hdr("access-control-allow-origin"),
        Some("http://localhost:5173"),
        "501 に CORS が掛かっていない"
    );
}

// ---------------------------------------------------------------------------
// 互換性の監査: 仕様とルータに現れないこと
// ---------------------------------------------------------------------------

/// `router()` と `docs/openapi.yaml` の突き合わせ (`openapi_contract`) を
/// 壊していないことをソースレベルで固定する。501 は fallback が受けるので、
/// ルータにも仕様にも載らない.
#[test]
fn unimplemented_paths_are_absent_from_router_and_spec() {
    let router_src = include_str!("../src/lib.rs");
    let spec = include_str!("../../../docs/openapi.yaml");

    for path in UNIMPLEMENTED {
        // ルータ本体 (`router_with_body_limit`) に `.route("<path>"` が無い。
        let route_line = format!(".route(\"{path}\"");
        assert!(
            !router_src.contains(&route_line),
            "{path} を `.route` で登録している (fallback の 501 と二重になる)"
        );
        // 仕様の paths: にも無い (`every_router_route_is_documented` の前提)。
        let spec_line = format!("\n  {path}:");
        assert!(
            !spec.contains(&spec_line),
            "docs/openapi.yaml に {path} が載っている (仕様とルータの 1:1 が崩れる)"
        );
    }

    // 501 の一覧とこのテストの表がずれていないこと。
    let listed: Vec<&str> = UNIMPLEMENTED_OPENAI_PATHS.iter().map(|(p, _)| *p).collect();
    assert_eq!(listed, UNIMPLEMENTED, "lib.rs の一覧とテストの表がずれている");
}

#[test]
fn hints_are_distinct_and_mention_an_alternative() {
    for (path, hint) in UNIMPLEMENTED_OPENAI_PATHS {
        assert!(
            hint.contains("/v1/chat/completions"),
            "{path} の説明に代替経路が無い: {hint}"
        );
        assert!(
            !hint.to_lowercase().contains("error:"),
            "{path} の説明がエラー文になっている: {hint}"
        );
    }
    a_public_hint_lookup_covers_both();
}

/// `unimplemented_openai_hint` が公開 API として両方を引けること
/// (テスト専用の抜け道ではなく、実装と同じ関数で判定していることの確認)。
fn a_public_hint_lookup_covers_both() {
    for path in UNIMPLEMENTED {
        assert!(
            denno_server::unimplemented_openai_hint(path).is_some(),
            "{path} が hint 表から引けない"
        );
        assert!(
            denno_server::unimplemented_openai_hint(&format!("{path}/")).is_some(),
            "{path}/ が hint 表から引けない"
        );
    }
    assert!(
        denno_server::unimplemented_openai_hint("/v1/embeddings/extra").is_none(),
        "部分一致で hint を返している"
    );
    assert!(denno_server::unimplemented_openai_hint("/v1/models").is_none());
}
