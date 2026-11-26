//! task 1269: Ollama 照会系 (`POST /api/show` / `GET /api/ps`) の
//! 契約エッジテスト。
//!
//! `tests/ollama_show_ps_996.rs` が「正常系 + 認証」を見るのに対し、こちらは
//! 入力の境界とワイヤ契約を固定する:
//!
//! - モデル名の照合: 前後空白は trim、`denno-test:latest` のようなタグ付きは
//!   完全一致しないので 404、`model` が `name` より優先。
//! - 空ボディ / 空文字 / 空白のみ / JSON でないボディ / 非オブジェクト JSON は
//!   すべて 400 の Ollama 封筒 (`{"error": "..."}`)。
//! - content-type: `/api/show` はボディを `Bytes` で受けるので要求側の
//!   content-type を検査しない (text/plain でも JSON として解釈する)。
//!   応答は常に `application/json`。
//! - `/api/ps` の JSON スキーマ不変条件 (キー集合・型・digest の形・
//!   `expires_at` が将来・`modified_at` を持たない)。
//!
//! ここで固定する挙動は `src/ollama.rs` の現実装 (`check_model` /
//! `ShowRequest::requested_model` / `show` / `ps`) を読み取ったもので、
//! テストは実装を変更せずに契約だけを検査する。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "denno-test";

fn state_with_key(api_key: Option<&str>) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, MODEL, None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: MODEL.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

fn app() -> Router {
    let state = state_with_key(None);
    denno_server::router(state)
}

struct Resp {
    status: StatusCode,
    content_type: String,
    body: String,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("本文が JSON ではない ({e}): {:?}", self.body))
    }

    fn error_message(&self) -> String {
        self.json()["error"]
            .as_str()
            .unwrap_or_else(|| panic!("error キーが文字列でない: {:?}", self.body))
            .to_string()
    }
}

/// 要求の content-type を明示できる送信ヘルパ。`content_type = None` は
/// ヘッダ無し。
async fn send(
    app: &Router,
    method: &str,
    uri: &str,
    content_type: Option<&str>,
    body: &str,
) -> Resp {
    let mut builder = Request::builder().method(method).uri(uri);
    if let Some(ct) = content_type {
        builder = builder.header(header::CONTENT_TYPE, ct);
    }
    let req = builder.body(Body::from(body.to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp { status, content_type, body: String::from_utf8_lossy(&bytes).to_string() }
}

async fn post_show(app: &Router, body: &str) -> Resp {
    send(app, "POST", "/api/show", Some("application/json"), body).await
}

async fn get_ps(app: &Router) -> Resp {
    send(app, "GET", "/api/ps", None, "").await
}

fn assert_json_content_type(r: &Resp) {
    assert!(
        r.content_type.starts_with("application/json"),
        "content-type = {:?}",
        r.content_type
    );
}

fn assert_rfc3339_utc_secs(s: &str, what: &str) {
    let b = s.as_bytes();
    assert_eq!(s.len(), 20, "{what} の長さが違う: {s}");
    assert_eq!(b[4], b'-', "{what} の形が違う: {s}");
    assert_eq!(b[7], b'-', "{what} の形が違う: {s}");
    assert_eq!(b[10], b'T', "{what} の形が違う: {s}");
    assert_eq!(b[13], b':', "{what} の形が違う: {s}");
    assert_eq!(b[16], b':', "{what} の形が違う: {s}");
    assert_eq!(b[19], b'Z', "{what} が UTC でない: {s}");
}

// ---------------------------------------------------------------------------
// /api/show: モデル名の照合
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_accepts_exact_model_name_and_trims_whitespace() {
    let app = app();

    for model in [MODEL, " denno-test ", "\tdenno-test\n"] {
        let r = post_show(&app, &json!({ "model": model }).to_string()).await;
        assert_eq!(r.status, StatusCode::OK, "model = {model:?}: {}", r.body);
        assert_json_content_type(&r);
        assert_eq!(r.json()["model_info"]["general.name"], MODEL);
    }
}

#[tokio::test]
async fn show_unknown_names_are_404_including_ollama_tag_form() {
    let app = app();

    // Ollama クライアントは `name:tag` を送るが、denno は完全一致なので
    // `denno-test:latest` は未知モデル (推測で別名解決しない)。
    for model in ["nope", "denno-test:latest", "DENNO-TEST", "denno"] {
        let r = post_show(&app, &json!({ "model": model }).to_string()).await;
        assert_eq!(r.status, StatusCode::NOT_FOUND, "model = {model:?}: {}", r.body);
        assert_json_content_type(&r);
        let msg = r.error_message();
        assert!(msg.contains("not found"), "model = {model:?} の文言: {msg}");
    }
}

#[tokio::test]
async fn show_prefers_model_over_legacy_name() {
    let app = app();

    // name だけなら受理。
    let r = post_show(&app, &json!({ "name": MODEL }).to_string()).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);

    // model が有効なら name が未知でも model 側で照合する。
    let r = post_show(
        &app,
        &json!({ "model": MODEL, "name": "some-other-model" }).to_string(),
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert_eq!(r.json()["model_info"]["general.name"], MODEL);

    // model が未知なら name が有効でも 404 (name へフォールバックしない)。
    let r = post_show(&app, &json!({ "model": "nope", "name": MODEL }).to_string()).await;
    assert_eq!(r.status, StatusCode::NOT_FOUND, "{}", r.body);
}

// ---------------------------------------------------------------------------
// /api/show: ボディ境界
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_empty_or_modelless_bodies_are_400_with_error_envelope() {
    let app = app();

    for body in [
        "",
        "{}",
        r#"{"model":""}"#,
        r#"{"model":"   "}"#,
        r#"{"name":""}"#,
        r#"{"name":"\t\n"}"#,
        r#"{"model":null}"#,
        // model が空文字なら name が有効でも 400 (`model` が優先され、
        // 空に trim した結果 None になる)。
        r#"{"model":"","name":"denno-test"}"#,
    ] {
        let r = post_show(&app, body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body:?}: {}", r.body);
        assert_json_content_type(&r);
        assert_eq!(
            r.json().as_object().map(|o| o.len()),
            Some(1),
            "封筒に余計なキー: {}",
            r.body
        );
        assert!(!r.error_message().is_empty(), "body = {body:?}");
    }
}

#[tokio::test]
async fn show_malformed_and_non_object_json_are_400() {
    let app = app();

    for body in ["{not json", "null", "[]", "42", "\"denno-test\"", "true"] {
        let r = post_show(&app, body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body:?}: {}", r.body);
        assert_json_content_type(&r);
        let msg = r.error_message();
        assert!(
            msg.contains("invalid request body") || msg.contains("model"),
            "body = {body:?} の文言: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// content-type
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_does_not_require_request_content_type() {
    let app = app();
    let body = json!({ "model": MODEL }).to_string();

    // ヘッダ無し / 別の content-type でもボディは JSON として解釈する
    // (`Bytes` 抽出なので content-type を検査しない)。
    for ct in [None, Some("text/plain"), Some("application/x-www-form-urlencoded")] {
        let r = send(&app, "POST", "/api/show", ct, &body).await;
        assert_eq!(r.status, StatusCode::OK, "content-type = {ct:?}: {}", r.body);
        assert_json_content_type(&r);
    }

    // charset 付きも同じ。
    let r = send(
        &app,
        "POST",
        "/api/show",
        Some("application/json; charset=utf-8"),
        &body,
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
}

#[tokio::test]
async fn show_and_ps_responses_are_json_with_expected_top_level_shape() {
    let app = app();

    let show = post_show(&app, &json!({ "model": MODEL }).to_string()).await;
    assert_eq!(show.status, StatusCode::OK);
    assert_json_content_type(&show);
    let v = show.json();
    let obj = v.as_object().expect("/api/show の応答がオブジェクトでない");
    for key in [
        "modelfile",
        "parameters",
        "template",
        "details",
        "model_info",
        "capabilities",
        "modified_at",
    ] {
        assert!(obj.contains_key(key), "/api/show に {key} が無い: {v}");
    }

    let ps = get_ps(&app).await;
    assert_eq!(ps.status, StatusCode::OK, "{}", ps.body);
    assert_json_content_type(&ps);
    assert_eq!(ps.body.lines().count(), 1, "ps の本文が 1 行でない: {:?}", ps.body);
    assert!(ps.json().is_object(), "ps の応答がオブジェクトでない");
}

// ---------------------------------------------------------------------------
// メソッド不整合
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_is_post_only_and_ps_is_get_only() {
    let app = app();

    let r = send(&app, "GET", "/api/show", None, "").await;
    assert_eq!(r.status, StatusCode::METHOD_NOT_ALLOWED, "{}", r.body);

    let r = send(
        &app,
        "POST",
        "/api/ps",
        Some("application/json"),
        r#"{"model":"denno-test"}"#,
    )
    .await;
    assert_eq!(r.status, StatusCode::METHOD_NOT_ALLOWED, "{}", r.body);
}

// ---------------------------------------------------------------------------
// /api/ps: JSON スキーマ不変条件
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ps_schema_invariants_hold() {
    let app = app();
    let r = get_ps(&app).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let v = r.json();

    let top = v.as_object().expect("/api/ps の応答がオブジェクトでない");
    assert_eq!(top.len(), 1, "トップレベルのキーが models だけでない: {v}");
    let models = v["models"].as_array().expect("models が配列でない");
    assert_eq!(models.len(), 1, "常駐モデルは 1 本: {v}");

    let m = &models[0];
    let obj = m.as_object().expect("models[0] がオブジェクトでない");
    // Ollama の /api/ps 要素のキー集合。増減はワイヤ契約の変更なので固定する。
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec![
            "details",
            "digest",
            "expires_at",
            "model",
            "name",
            "size",
            "size_vram"
        ],
        "models[0] のキー集合が違う: {m}"
    );
    // /api/tags の要素と違い、ps は modified_at を持たない。
    assert!(obj.get("modified_at").is_none(), "ps に modified_at がある: {m}");

    assert_eq!(m["name"], MODEL);
    assert_eq!(m["model"], MODEL);

    for key in ["size", "size_vram"] {
        assert!(m[key].is_u64(), "{key} が非負整数でない: {m}");
        assert_eq!(m[key], 0, "{key} は engine 非公開なので 0: {m}");
    }

    let digest = m["digest"].as_str().expect("digest が文字列でない");
    assert_eq!(digest.len(), 64, "digest = {digest}");
    assert!(
        digest.chars().all(|c| c.is_ascii_hexdigit()),
        "digest が hex でない: {digest}"
    );

    let expires = m["expires_at"].as_str().expect("expires_at が文字列でない");
    assert_rfc3339_utc_secs(expires, "expires_at");
    // 常駐 (アンロードしない) を表す無期限側の値。少なくとも 2100 年より後。
    assert!(expires > "2100", "expires_at が無期限側でない: {expires}");

    let d = m["details"].as_object().expect("details がオブジェクトでない");
    for key in [
        "parent_model",
        "format",
        "family",
        "families",
        "parameter_size",
        "quantization_level",
    ] {
        assert!(d.contains_key(key), "details に {key} が無い: {m}");
    }
    assert_eq!(d["parent_model"], "");
    assert_eq!(d["format"], "gguf");
    assert_eq!(d["family"], "denno");
    assert_eq!(d["families"], json!(["denno"]));
    assert_eq!(d["parameter_size"], "");
    assert!(d["quantization_level"].is_string(), "quantization_level: {m}");
}

#[tokio::test]
async fn ps_digest_matches_show_and_tags() {
    let app = app();

    let ps = get_ps(&app).await.json();
    let ps_digest = ps["models"][0]["digest"].as_str().unwrap().to_string();

    let tags = send(&app, "GET", "/api/tags", None, "").await.json();
    assert_eq!(tags["models"][0]["digest"], json!(ps_digest), "tags と ps の digest が違う");

    // digest は決定論的: 2 回呼んでも同じ。
    let again = get_ps(&app).await.json();
    assert_eq!(again["models"][0]["digest"], json!(ps_digest));
}
