//! Task 901: `/v1/models` のメタデータ拡充・`GET /v1/models/:id`・
//! 生成 3 経路 (`/v1/chat/completions` / `/v1/completions` / `/v1/messages`) の
//! 別名解決のテスト。
//!
//! ダミーエンジンは他の denno-server 統合テストと同じ (`DummyModel` +
//! `dummy_tokenizer`)。canonical 名は `denno-test`、`model_id` は
//! `Qwen/Qwen2.5-0.5B-Instruct`、`model_dir` は `qwen-test-dir`。
//!
//! 別名の環境変数 `DENNO_MODEL_ALIASES` はプロセス全体に効くため、
//! 設定するテストは 1 本だけにし、他は「含まれること」だけを見る。

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

const CANONICAL: &str = "denno-test";
const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const MODEL_DIR: &str = "qwen-test-dir";

fn app_with_key(api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, CANONICAL, None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = MODEL_DIR.to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: CANONICAL.to_string(),
        model_id: MODEL_ID.to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

fn app() -> Router {
    app_with_key(None)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    send(app, req).await
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    send(app, req).await
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

// ---------------------------------------------------------------------------
// /v1/models のメタデータ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_list_carries_rich_metadata() {
    let app = app();
    let (status, body) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["object"], "list");
    assert_eq!(body["has_more"], false);

    let m = &body["data"][0];
    assert_eq!(m["id"], CANONICAL);
    assert_eq!(m["object"], "model");
    assert_eq!(m["type"], "model");
    assert_eq!(m["display_name"], CANONICAL);
    assert_eq!(m["owned_by"], "denno");
    // created は 0 ではなく実時刻 (epoch 秒)。2020 年以降であることだけ固定する。
    assert!(
        m["created"].as_u64().unwrap_or(0) > 1_577_836_800,
        "created が実時刻でない: m = {m}"
    );
    // context_window は 3 つの名前で同じ値。
    let cw = m["context_window"].as_u64().unwrap_or(0);
    assert!(cw > 0, "m = {m}");
    assert_eq!(m["context_length"].as_u64(), Some(cw));
    assert_eq!(m["max_context_length"].as_u64(), Some(cw));
    // 運用メタデータは維持。
    assert_eq!(m["model_dir"], MODEL_DIR);
    assert_eq!(m["model_id"], MODEL_ID);
    assert_eq!(m["capabilities"]["openai"], true);
    assert_eq!(m["capabilities"]["anthropic"], true);
}

#[tokio::test]
async fn models_list_reports_default_aliases() {
    let app = app();
    let (status, body) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    let aliases: Vec<String> = body["data"][0]["aliases"]
        .as_array()
        .expect("aliases が配列でない")
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    // model_id 全文とその最終セグメント、model_dir。
    for want in [MODEL_ID, "Qwen2.5-0.5B-Instruct", MODEL_DIR] {
        assert!(aliases.iter().any(|a| a == want), "別名 {want} が無い: {aliases:?}");
    }
    // canonical 名は別名に含めない (重複を reporting しない)。
    assert!(!aliases.iter().any(|a| a == CANONICAL), "aliases = {aliases:?}");
}

// ---------------------------------------------------------------------------
// GET /v1/models/:id
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_by_id_accepts_canonical_and_alias() {
    let app = app();
    // `model_id` は `/` を含むので、URL ではパーセントエンコードして 1 セグメント
    // として渡す (axum の `:id` はデコード後の値を見る)。
    for id in [CANONICAL, "Qwen%2FQwen2.5-0.5B-Instruct", MODEL_DIR] {
        let (status, body) = get(&app, &format!("/v1/models/{id}")).await;
        assert_eq!(status, StatusCode::OK, "id = {id}, body = {body}");
        // 応答の id は常に canonical (別名は同じ重みの別名でしかない)。
        assert_eq!(body["id"], CANONICAL, "id = {id}, body = {body}");
        assert_eq!(body["object"], "model");
        assert_eq!(body["owned_by"], "denno");
        assert_eq!(body["context_window"], body["context_length"]);
    }
}

#[tokio::test]
async fn model_by_id_unknown_is_clean_404() {
    let app = app();
    let (status, body) = get(&app, "/v1/models/nope").await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body = {body}");
    // OpenAI 形式のエラー本文 (code は null)。
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"].as_str().unwrap_or("").contains("nope"),
        "body = {body}"
    );
    assert!(body["error"].get("code").is_some(), "body = {body}");
}

#[tokio::test]
async fn model_by_id_requires_auth_when_key_is_set() {
    let app = app_with_key(Some("secret"));
    let (status, _) = get(&app, "/v1/models/denno-test").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// 生成 3 経路の別名解決
// ---------------------------------------------------------------------------

/// `DENNO_MODEL_ALIASES` は 1 プロセス 1 回しか読まないわけではないので、
/// このテストだけで設定し、他のテストは「含まれる」ことだけを見る。
#[tokio::test]
async fn configurable_aliases_are_accepted_across_generation_routes() {
    // `alias` (裸) と `alias=canonical` は採用、`alias=他モデル` は無視。
    std::env::set_var(
        "DENNO_MODEL_ALIASES",
        "swift-15,fast=denno-test,wrong=some-other-model",
    );

    let app = app();

    // 一覧と単体取得に反映される。
    let (status, body) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    let aliases: Vec<&str> = body["data"][0]["aliases"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    assert!(aliases.contains(&"swift-15"), "aliases = {aliases:?}");
    assert!(aliases.contains(&"fast"), "aliases = {aliases:?}");
    assert!(!aliases.contains(&"wrong"), "aliases = {aliases:?}");

    let (status, body) = get(&app, "/v1/models/fast").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["id"], CANONICAL);

    // 1) /v1/chat/completions
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "swift-15",
            "max_tokens": 4,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    // 応答の model は canonical に正規化される。
    assert_eq!(body["model"], CANONICAL, "body = {body}");

    // 2) /v1/completions
    let (status, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "fast", "max_tokens": 4, "temperature": 0.0, "prompt": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["model"], CANONICAL, "body = {body}");

    // 3) /v1/messages
    let (status, body) = post(
        &app,
        "/v1/messages",
        json!({
            "model": "swift-15",
            "max_tokens": 4,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["model"], CANONICAL, "body = {body}");

    // 未登録の別名は従来どおり 404 (chat / completions / messages)。
    let (status, body) = post(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "not-registered",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body = {body}");
    let (status, body) = post(
        &app,
        "/v1/completions",
        json!({"model": "not-registered", "max_tokens": 4, "prompt": "hi"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body = {body}");
    let (status, body) = post(
        &app,
        "/v1/messages",
        json!({
            "model": "not-registered",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body = {body}");

    std::env::remove_var("DENNO_MODEL_ALIASES");
}
