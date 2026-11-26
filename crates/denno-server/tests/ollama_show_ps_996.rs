//! task 996: Ollama 互換の照会系 (`POST /api/show` / `GET /api/ps`) の
//! 統合テスト。
//!
//! 目的:
//!
//! - `/api/show` がモデルのメタデータ (`details` / `model_info` /
//!   `capabilities`) を返し、未指定は 400・未知モデルは 404 になること。
//! - `/api/ps` が常駐モデル 1 本を Ollama の形 (`expires_at` / `size_vram`) で
//!   返すこと。
//! - 両方が認証の内側にあること。
//! - ルートが `ollama.rs` のサブルータ側にあり、`lib.rs` の `.route("` 行を
//!   増やしていないこと (`tests/openapi_contract.rs` の逆方向検査を守る)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
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

fn app_with_key(api_key: Option<&str>) -> (Router, Arc<AppState>) {
    let state = state_with_key(api_key);
    (denno_server::router(state.clone()), state)
}

async fn post_json(app: &Router, uri: &str, body: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    send(app, req).await
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    send(app, req).await
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

#[tokio::test]
async fn show_returns_model_metadata() {
    let (app, state) = app_with_key(None);
    let (status, body) = post_json(&app, "/api/show", &json!({"model": MODEL}).to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {body}");

    // details は /api/tags と同じ形。
    assert_eq!(body["details"]["family"], "denno");
    assert_eq!(body["details"]["format"], "gguf");
    assert_eq!(body["details"]["quantization_level"], json!(state.info.quant));

    // model_info は engine の不変情報と一致する。
    assert_eq!(body["model_info"]["general.architecture"], "denno");
    assert_eq!(body["model_info"]["general.name"], MODEL);
    assert_eq!(body["model_info"]["denno.context_length"], json!(state.info.context));
    assert_eq!(body["model_info"]["denno.block_count"], json!(state.info.layers));
    assert_eq!(body["model_info"]["denno.kv_slot_count"], json!(state.info.slots));

    assert_eq!(body["capabilities"], json!(["completion"]));
    assert!(body.get("modelfile").is_some());
    // modified_at は RFC3339 (UTC・秒精度) の形。
    let modified = body["modified_at"].as_str().expect("modified_at が文字列でない");
    assert!(
        modified.len() == 20 && modified.ends_with('Z') && modified.as_bytes()[10] == b'T',
        "modified_at の形が違う: {modified}"
    );
}

#[tokio::test]
async fn show_accepts_legacy_name_field() {
    let (app, _state) = app_with_key(None);
    let (status, body) = post_json(&app, "/api/show", &json!({"name": MODEL}).to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["details"]["family"], "denno");
}

#[tokio::test]
async fn show_rejects_missing_model_with_400() {
    let (app, _state) = app_with_key(None);
    for body in ["", "{}", r#"{"model":""}"#] {
        let (status, v) = post_json(&app, "/api/show", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body = {body:?} -> {v}");
        // Ollama 形式のエラー封筒。
        assert!(v["error"].is_string(), "error キーが無い: {v}");
    }
}

#[tokio::test]
async fn show_rejects_invalid_json_with_400() {
    let (app, _state) = app_with_key(None);
    let (status, v) = post_json(&app, "/api/show", "{not json").await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"].is_string(), "{v}");
}

#[tokio::test]
async fn show_unknown_model_is_404() {
    let (app, _state) = app_with_key(None);
    let (status, v) = post_json(&app, "/api/show", &json!({"model": "nope"}).to_string()).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{v}");
    let msg = v["error"].as_str().unwrap_or_default();
    assert!(msg.contains("not found"), "error 文言が違う: {msg}");
}

#[tokio::test]
async fn ps_lists_the_resident_model() {
    let (app, state) = app_with_key(None);
    let (status, body) = get(&app, "/api/ps").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");

    let models = body["models"].as_array().expect("models が配列でない");
    assert_eq!(models.len(), 1, "常駐モデルは 1 本: {body}");
    let m = &models[0];
    assert_eq!(m["name"], MODEL);
    assert_eq!(m["model"], MODEL);
    assert_eq!(m["size"], 0);
    assert_eq!(m["size_vram"], 0);
    assert_eq!(m["details"]["quantization_level"], json!(state.info.quant));

    // digest は /api/tags と同じ決定論的な 64 hex。
    let digest = m["digest"].as_str().expect("digest が文字列でない");
    assert_eq!(digest.len(), 64);
    assert!(digest.chars().all(|c| c.is_ascii_hexdigit()), "digest が hex でない: {digest}");
    let (tags_status, tags) = get(&app, "/api/tags").await;
    assert_eq!(tags_status, StatusCode::OK);
    assert_eq!(tags["models"][0]["digest"], json!(digest), "tags と ps の digest が違う");

    // expires_at は将来の RFC3339。denno は常駐なので無期限側 (100 年後)。
    let expires = m["expires_at"].as_str().expect("expires_at が文字列でない");
    assert!(
        expires.len() == 20 && expires.ends_with('Z') && expires.as_bytes()[10] == b'T',
        "expires_at の形が違う: {expires}"
    );
    assert!(expires > "2100", "expires_at が無期限側でない: {expires}");
}

#[tokio::test]
async fn show_and_ps_are_behind_auth() {
    let (app, _state) = app_with_key(Some("secret-key"));

    let (status, v) = post_json(&app, "/api/show", &json!({"model": MODEL}).to_string()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{v}");
    assert!(v["error"].is_string(), "Ollama 形式のエラーでない: {v}");

    let (status, v) = get(&app, "/api/ps").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "{v}");

    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header("content-type", "application/json")
        .header("authorization", "Bearer secret-key")
        .body(Body::from(json!({"model": MODEL}).to_string()))
        .unwrap();
    let (status, v) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK, "認証付きの show が通らない: {v}");

    let req = Request::builder()
        .method("GET")
        .uri("/api/ps")
        .header("x-api-key", "secret-key")
        .body(Body::empty())
        .unwrap();
    let (status, v) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK, "認証付きの ps が通らない: {v}");
}

/// サブルータ経由であることの回帰テスト。`lib.rs` に `.route("/api/show")` /
/// `.route("/api/ps")` を書くと `tests/openapi_contract.rs` の
/// `every_router_route_is_documented` が落ちる (`docs/openapi.yaml` は今回の
/// スコープ外なので仕様に足せない)。登録先を固定して事故を防ぐ。
#[test]
fn new_routes_live_in_the_ollama_subrouter_not_lib_rs() {
    let lib = include_str!("../src/lib.rs");
    let ollama = include_str!("../src/ollama.rs");
    for path in ["/api/show", "/api/ps"] {
        let route_line = format!(".route(\"{path}\"");
        assert!(
            !lib.contains(&route_line),
            "lib.rs に {route_line} がある (サブルータに移すこと)"
        );
        assert!(
            ollama.contains(&route_line),
            "ollama.rs に {route_line} が無い"
        );
    }
    assert!(
        lib.contains(".merge(ollama::admin_router())"),
        "lib.rs が ollama::admin_router() を merge していない"
    );
}
