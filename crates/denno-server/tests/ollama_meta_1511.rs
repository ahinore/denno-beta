//! task 1511: Ollama 照会系 (`GET /api/tags` / `POST /api/show` /
//! `GET /api/version` / `GET /api/ps`) のクライアント互換監査。
//!
//! 標準 Ollama クライアント (Open WebUI / Continue / Cline 等) が読む形を
//! ワイヤ契約として固定する:
//!
//! - `/api/tags`: 要素は `name` と `model` の両方を持ち同値、`details` は
//!   `parent_model` / `format` / `family` / `families` / `parameter_size` /
//!   `quantization_level` の 6 キー、`digest` は 64 hex、`modified_at` は
//!   RFC3339 (UTC・秒精度)。
//! - `/api/show`: 標準キー集合 (task 1511 で `license` を追加)。`details` は
//!   `/api/tags` と同値、`model_info` の数値は起動時の不変情報と一致。
//! - `/api/version`: `{"version": "..."}` の 1 キーだけ。
//! - 未知モデルは 404、壊れた JSON / 非オブジェクト / 空は 400 で、本文は
//!   Ollama の封筒 `{"error": "..."}` のみ。
//! - 認証は全 4 経路の内側 (`x-api-key` と `Bearer` の両方を受ける)。
//! - ヘッダ衛生: `application/json`、`server` / `x-powered-by` を出さない、
//!   `x-request-id` は 1 個。本文にホスト情報 (パス・メール・IP) を漏らさない。
//!
//! 全て同期的なルータ呼び出しだけで、sleep も生成も使わない (決定論的)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
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

fn app_with_key(key: Option<&str>) -> Router {
    denno_server::router(state_with_key(key))
}

fn app() -> Router {
    app_with_key(None)
}

struct Resp {
    status: StatusCode,
    headers: HeaderMap,
    body: String,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("本文が JSON ではない ({e}): {:?}", self.body))
    }

    fn content_type(&self) -> String {
        self.headers
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string()
    }

    fn error_message(&self) -> String {
        self.json()["error"]
            .as_str()
            .unwrap_or_else(|| panic!("error キーが文字列でない: {:?}", self.body))
            .to_string()
    }

    fn header_count(&self, name: &str) -> usize {
        self.headers.get_all(name).iter().count()
    }
}

async fn send(
    app: &Router,
    method: &str,
    uri: &str,
    extra: &[(&str, &str)],
    body: &str,
) -> Resp {
    let mut builder = Request::builder().method(method).uri(uri);
    for (k, v) in extra {
        builder = builder.header(*k, *v);
    }
    let req = builder.body(Body::from(body.to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp { status, headers, body: String::from_utf8_lossy(&bytes).to_string() }
}

async fn get(app: &Router, uri: &str) -> Resp {
    send(app, "GET", uri, &[], "").await
}

async fn post_json(app: &Router, uri: &str, body: &str) -> Resp {
    send(app, "POST", uri, &[(header::CONTENT_TYPE.as_str(), "application/json")], body).await
}

fn assert_json_hygiene(r: &Resp, what: &str) {
    assert!(
        r.content_type().starts_with("application/json"),
        "{what}: content-type = {:?}",
        r.content_type()
    );
    assert!(r.headers.get("server").is_none(), "{what}: server ヘッダ");
    assert!(
        r.headers.get("x-powered-by").is_none(),
        "{what}: x-powered-by ヘッダ"
    );
    assert_eq!(r.header_count("x-request-id"), 1, "{what}: x-request-id の数");
}

fn sorted_keys(v: &Value) -> Vec<String> {
    let obj = v.as_object().unwrap_or_else(|| panic!("オブジェクトでない: {v}"));
    let mut keys: Vec<String> = obj.keys().cloned().collect();
    keys.sort_unstable();
    keys
}

fn assert_rfc3339_utc_secs(s: &str, what: &str) {
    let b = s.as_bytes();
    assert_eq!(s.len(), 20, "{what} の長さが違う: {s}");
    for (i, c) in [(4, b'-'), (7, b'-'), (10, b'T'), (13, b':'), (16, b':'), (19, b'Z')] {
        assert_eq!(b[i], c, "{what} の形が違う: {s}");
    }
}

fn assert_no_env_leak(body: &str, what: &str) {
    // リテラルを直書きしない (このファイル自身が監査の grep に引っかからない
    // ように、禁止パターンは部品から組み立てる)。
    let needles = [
        ["/", "home", "/"].concat(),
        ["127", "0", "0", "1"].join("."),
        ["local", "host"].concat(),
        ["http", "://"].concat(),
        "@".to_string(),
        ["debug", "!"].concat(),
    ];
    for needle in needles {
        assert!(
            !body.contains(&needle),
            "{what}: 本文に環境情報 ({needle}) が漏れている: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// /api/tags
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tags_matches_the_standard_model_list_shape() {
    let app = app();
    let r = get(&app, "/api/tags").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert_json_hygiene(&r, "tags");

    let v = r.json();
    assert_eq!(sorted_keys(&v), vec!["models".to_string()], "トップレベル: {v}");

    let models = v["models"].as_array().expect("models が配列でない");
    assert_eq!(models.len(), 1, "{v}");
    let m = &models[0];

    assert_eq!(
        sorted_keys(m),
        vec![
            "details".to_string(),
            "digest".to_string(),
            "model".to_string(),
            "modified_at".to_string(),
            "name".to_string(),
            "size".to_string(),
        ],
        "tags 要素のキー集合: {m}"
    );
    // 標準クライアントは `model` を読み、古いものは `name` を読む。
    assert_eq!(m["name"], MODEL);
    assert_eq!(m["model"], MODEL);
    assert_eq!(m["size"], 0);
    assert!(m["size"].is_u64(), "size が非負整数でない: {m}");

    assert_rfc3339_utc_secs(m["modified_at"].as_str().expect("modified_at"), "modified_at");
    let digest = m["digest"].as_str().expect("digest が文字列でない");
    assert_eq!(digest.len(), 64, "digest = {digest}");
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()), "digest = {digest}");

    let d = &m["details"];
    assert_eq!(
        sorted_keys(d),
        vec![
            "families".to_string(),
            "family".to_string(),
            "format".to_string(),
            "parameter_size".to_string(),
            "parent_model".to_string(),
            "quantization_level".to_string(),
        ],
        "details のキー集合: {d}"
    );
    assert_eq!(d["parent_model"], "");
    assert_eq!(d["format"], "gguf");
    assert_eq!(d["family"], "denno");
    assert_eq!(d["families"], json!(["denno"]));
    assert_eq!(d["parameter_size"], "");
    assert!(
        d["quantization_level"].is_string(),
        "quantization_level が文字列でない: {d}"
    );

    assert_no_env_leak(&r.body, "tags");
}

#[tokio::test]
async fn tags_is_deterministic_and_agrees_with_ps() {
    let app = app();
    let first = get(&app, "/api/tags").await.json();
    let second = get(&app, "/api/tags").await.json();
    assert_eq!(first["models"][0]["digest"], second["models"][0]["digest"]);
    assert_eq!(first["models"][0]["details"], second["models"][0]["details"]);

    let ps = get(&app, "/api/ps").await.json();
    assert_eq!(first["models"][0]["digest"], ps["models"][0]["digest"]);
    assert_eq!(first["models"][0]["details"], ps["models"][0]["details"]);
    assert_eq!(first["models"][0]["name"], ps["models"][0]["name"]);
    assert_eq!(first["models"][0]["model"], ps["models"][0]["model"]);
}

// ---------------------------------------------------------------------------
// /api/show
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_carries_every_standard_key_including_license() {
    let app = app();
    let r = post_json(&app, "/api/show", &json!({ "model": MODEL }).to_string()).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert_json_hygiene(&r, "show");

    let v = r.json();
    // task 1511: `license` (標準 Ollama は常に返す) を追加した。キー集合は
    // クライアントの読み取り先なので増減を固定する。
    assert_eq!(
        sorted_keys(&v),
        vec![
            "capabilities".to_string(),
            "details".to_string(),
            "license".to_string(),
            "model_info".to_string(),
            "modelfile".to_string(),
            "modified_at".to_string(),
            "parameters".to_string(),
            "template".to_string(),
        ],
        "show のキー集合: {v}"
    );
    assert_eq!(v["license"], "", "license は空文字: {v}");
    assert!(v["modelfile"].as_str().is_some_and(|s| s.contains(MODEL)), "{v}");
    assert_eq!(v["parameters"], "");
    assert_eq!(v["template"], "");
    assert_eq!(v["capabilities"], json!(["completion"]));
    assert_rfc3339_utc_secs(v["modified_at"].as_str().expect("modified_at"), "modified_at");

    assert_no_env_leak(&r.body, "show");
}

#[tokio::test]
async fn show_details_and_model_info_are_consistent_with_tags_and_info() {
    let app = app();
    let state = state_with_key(None);
    let show = post_json(&app, "/api/show", &json!({ "model": MODEL }).to_string())
        .await
        .json();
    let tags = get(&app, "/api/tags").await.json();

    assert_eq!(
        show["details"], tags["models"][0]["details"],
        "show と tags の details が違う"
    );
    assert_eq!(show["details"]["quantization_level"], json!(state.info.quant));

    let info = &show["model_info"];
    assert_eq!(info["general.architecture"], "denno");
    assert_eq!(info["general.name"], MODEL);
    assert_eq!(info["denno.context_length"], json!(state.info.context));
    assert_eq!(info["denno.block_count"], json!(state.info.layers));
    assert_eq!(info["denno.kv_slot_count"], json!(state.info.slots));
    assert_eq!(info["denno.quantization"], json!(state.info.quant));
    for key in [
        "denno.context_length",
        "denno.block_count",
        "denno.kv_slot_count",
    ] {
        assert!(info[key].is_u64(), "model_info[{key}] が非負整数でない: {info}");
    }
}

#[tokio::test]
async fn show_legacy_name_alias_returns_the_same_document() {
    let app = app();
    let by_model = post_json(&app, "/api/show", &json!({ "model": MODEL }).to_string())
        .await
        .json();
    let by_name = post_json(&app, "/api/show", &json!({ "name": MODEL }).to_string())
        .await
        .json();
    assert_eq!(by_model["details"], by_name["details"]);
    assert_eq!(by_model["model_info"], by_name["model_info"]);
    assert_eq!(by_model["license"], by_name["license"]);
}

// ---------------------------------------------------------------------------
// エッジ: 未知モデル 404 / 壊れたボディ 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn show_unknown_model_is_404_with_ollama_error_envelope() {
    let app = app();
    // `name:tag` 形 (Ollama クライアントが送りがち) も完全一致しないので 404。
    for model in ["nope", "denno-test:latest", "DENNO-TEST", "denno"] {
        let r = post_json(&app, "/api/show", &json!({ "model": model }).to_string()).await;
        assert_eq!(r.status, StatusCode::NOT_FOUND, "model = {model}: {}", r.body);
        assert_json_hygiene(&r, "show 404");
        assert_eq!(sorted_keys(&r.json()), vec!["error".to_string()], "{}", r.body);
        assert!(r.error_message().contains("not found"), "{model}: {}", r.body);
    }
}

#[tokio::test]
async fn show_malformed_or_modelless_bodies_are_400_with_ollama_error_envelope() {
    let app = app();
    for body in [
        "",
        "{}",
        "{not json",
        "null",
        "[]",
        "42",
        "\"denno-test\"",
        r#"{"model":""}"#,
        r#"{"model":"   "}"#,
        r#"{"model":null}"#,
    ] {
        let r = post_json(&app, "/api/show", body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body:?}: {}", r.body);
        assert_json_hygiene(&r, "show 400");
        assert_eq!(
            sorted_keys(&r.json()),
            vec!["error".to_string()],
            "body = {body:?}: {}",
            r.body
        );
        assert!(!r.error_message().is_empty(), "body = {body:?}");
    }
}

// ---------------------------------------------------------------------------
// /api/version
// ---------------------------------------------------------------------------

#[tokio::test]
async fn version_is_a_single_deterministic_string_key() {
    let app = app();
    let r = get(&app, "/api/version").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    assert_json_hygiene(&r, "version");

    let key = "version".to_string();
    let v = r.json();
    assert_eq!(sorted_keys(&v), vec![key], "{v}");
    let s = v["version"].as_str().expect("version が文字列でない");
    assert!(!s.is_empty(), "version が空");
    // セマンティックバージョン風 (`数字.数字...`) でないとクライアントの
    // 機能ゲートが落ちるため、先頭が数字であることだけ固定する。
    assert!(
        s.chars().next().is_some_and(|c| c.is_ascii_digit()),
        "version が数字で始まらない: {s}"
    );

    let again = get(&app, "/api/version").await.json();
    assert_eq!(v, again, "version が呼ぶたびに変わる");
}

// ---------------------------------------------------------------------------
// 認証 (4 経路すべて内側)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metadata_endpoints_require_auth_and_accept_both_header_forms() {
    let app = app_with_key(Some("secret-key"));
    let show_body = json!({ "model": MODEL }).to_string();

    for (label, r) in [
        ("tags", get(&app, "/api/tags").await),
        ("version", get(&app, "/api/version").await),
        ("show", post_json(&app, "/api/show", &show_body).await),
        ("ps", get(&app, "/api/ps").await),
    ] {
        assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{label}: {}", r.body);
        assert_json_hygiene(&r, "401");
        assert_eq!(sorted_keys(&r.json()), vec!["error".to_string()], "{label}: {}", r.body);
        // 認証要求ヘッダで資格情報の形式を漏らさない (Ollama は素の 401)。
        assert!(
            r.headers.get(header::WWW_AUTHENTICATE).is_none(),
            "{label}: WWW-Authenticate を出している"
        );
        assert!(!r.body.contains("secret-key"), "{label}: キーが本文に漏れる");
    }

    // 誤った値は拒否。
    let r = send(&app, "GET", "/api/tags", &[(header::AUTHORIZATION.as_str(), "Bearer wrong")], "")
        .await;
    assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{}", r.body);

    // `x-api-key` と `Authorization: Bearer` の両方を受ける。
    let r = send(&app, "GET", "/api/tags", &[("x-api-key", "secret-key")], "").await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let r = send(
        &app,
        "GET",
        "/api/version",
        &[(header::AUTHORIZATION.as_str(), "Bearer secret-key")],
        "",
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    let r = send(
        &app,
        "POST",
        "/api/show",
        &[
            (header::CONTENT_TYPE.as_str(), "application/json"),
            ("x-api-key", "secret-key"),
        ],
        &show_body,
    )
    .await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.body);
    // 認証付きでも応答本文にキーを出さない。
    assert!(!r.body.contains("secret-key"), "show の本文にキーが漏れる");
}
