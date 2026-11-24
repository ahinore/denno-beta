//! P817: 運用エンドポイントの統合テスト。
//!
//! - `GET /ready` — 200 は「モデルロード済み + ワーカー生存 + drain 中でない」だけ。
//!   未ロード (503 loading) / drain 中 (503 draining) を別々に踏む。
//! - `GET /ready` は `/health` と同じく認証を要求しない (プローブはトークンを持たない)。
//! - `GET /v1/models` のメタデータ (context_length / quant / model_dir /
//!   capabilities)。既存フィールドは残る。
//! - drain 中は `POST /v1/messages` と `POST /v1/chat/completions` が
//!   503 overloaded_error で**受付しない** (キューに積まれない = pending 0)。
//! - drain 開始時に既に受付済みの要求は打ち切られず完走する
//!   (「進行中の生成を短い上限まで待つ」の土台)。

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn app_with(
    spawn: fn(Arc<Mutex<Engine>>, usize) -> Scheduler,
    api_key: Option<&str>,
    model_dir: &str,
) -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = model_dir.to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    (denno_server::router(state.clone()), state)
}

/// 通常起動の state (ワーカー ready)。
fn ready_state() -> (Router, Arc<AppState>) {
    app_with(Scheduler::spawn, None, "qwen-test-dir")
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn chat_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

// ---------------------------------------------------------------------------
// /ready
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ready_returns_200_when_model_loaded_and_worker_alive() {
    let (app, state) = ready_state();
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["ready"], true);
    assert_eq!(body["draining"], false);
    assert_eq!(body["model_loaded"], true);
    assert_eq!(body["worker_alive"], true);
    assert_eq!(body["model"], "denno-test");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    assert!(body["context_length"].as_u64().unwrap_or(0) > 0, "body = {body}");
    // /health は従来どおり (liveness は drain 中でも 200)。
    let (hs, hb) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
    assert_eq!(hb["status"], "ok");
    drop(state);
}

#[tokio::test]
async fn ready_returns_503_while_loading() {
    let (app, _state) = app_with(Scheduler::spawn_not_ready, None, "");

    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "loading");
    assert_eq!(body["ready"], false);
    assert_eq!(body["model_loaded"], false);
    // liveness は 200 のまま (プロセスは生きている)
    let (hs, _) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
}

#[tokio::test]
async fn ready_returns_503_while_draining() {
    let (app, state) = ready_state();
    state.scheduler.lifecycle().begin_drain();

    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["draining"], true);
    assert_eq!(body["ready"], false);
    // liveness は drain 中でも 200 (プロセスは生きている)
    let (hs, _) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);

    // drain を取り消す手段は無い (新しい state でしか ready に戻らない) ことも
    // ここで固定しておく: フラグは立ったまま。
    assert!(state.scheduler.lifecycle().is_draining());
}

#[tokio::test]
async fn ready_does_not_require_auth() {
    let (app, _state) = app_with(Scheduler::spawn, Some("secret"), "");

    // トークン無しでも 200 (プローブ用)
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    // 一方 /v1/models は認証が要る (既存挙動は変えない)
    let (ms, _) = get(&app, "/v1/models").await;
    assert_eq!(ms, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// /v1/models のメタデータ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_exposes_operational_metadata() {
    let (app, _state) = ready_state();
    let (status, body) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["object"], "list");
    let m = &body["data"][0];
    // 既存フィールドは残る
    assert_eq!(m["id"], "denno-test");
    assert_eq!(m["object"], "model");
    assert_eq!(m["type"], "model");
    assert_eq!(m["display_name"], "denno-test");
    assert_eq!(m["owned_by"], "denno");
    assert!(m["context_length"].as_u64().unwrap_or(0) > 0, "m = {m}");
    // P817 で足したもの
    assert_eq!(m["max_context_length"], m["context_length"]);
    assert_eq!(m["model_dir"], "qwen-test-dir");
    assert!(!m["quant"].as_str().unwrap_or("").is_empty(), "m = {m}");
    assert!(m["slots"].as_u64().is_some(), "m = {m}");
    assert!(m["layers"].as_u64().is_some(), "m = {m}");
    let caps = &m["capabilities"];
    assert_eq!(caps["anthropic"], true);
    assert_eq!(caps["openai"], true);
    assert_eq!(caps["tools"], true);
    assert_eq!(caps["streaming"], true);
    assert_eq!(caps["count_tokens"], true);
    // telemetry はゲート (env) 依存なので boolean であることだけ固定する
    assert!(caps["telemetry"].is_boolean(), "caps = {caps}");
}

// ---------------------------------------------------------------------------
// drain 中の受付拒否
// ---------------------------------------------------------------------------

#[tokio::test]
async fn messages_are_rejected_while_draining() {
    let (app, state) = ready_state();
    state.scheduler.lifecycle().begin_drain();

    let (status, body) = post(&app, "/v1/messages", chat_body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "overloaded_error");
    assert!(
        body["error"]["message"].as_str().unwrap_or("").contains("draining"),
        "body = {body}"
    );
    // キューに積まれていない (drain が終わらない原因を作らない)
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
}

#[tokio::test]
async fn openai_chat_is_rejected_while_draining() {
    let (app, state) = ready_state();
    state.scheduler.lifecycle().begin_drain();

    let body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
    });
    let (status, resp) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {resp}");
    // OpenAI のエラー型では overloaded_error を server_error に寄せる既存規則
    // (`impl From<ApiError> for OaError`) を通る。メッセージで drain と分かる。
    assert_eq!(resp["error"]["type"], "server_error", "body = {resp}");
    assert!(
        resp["error"]["message"].as_str().unwrap_or("").contains("draining"),
        "body = {resp}"
    );
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
}

// ---------------------------------------------------------------------------
// drain は受付済みの生成を打ち切らない
// ---------------------------------------------------------------------------

#[tokio::test]
async fn accepted_request_finishes_after_drain_starts() {
    let (app, state) = ready_state();

    // エンジンのロックを握ってワーカーを止め、要求が「受付済み・未実行」の
    // 状態を作る (drain の待ち対象 = キュー + 進行中)。
    let guard = state.engine.lock().unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(chat_body().to_string()))
        .unwrap();
    let app2 = app.clone();
    let task = tokio::spawn(async move { app2.oneshot(req).await });

    // ハンドラが受付 (submit) するまで待つ
    let start = Instant::now();
    while state.scheduler.pending() == 0 && state.scheduler.in_flight() == 0 {
        assert!(start.elapsed() < Duration::from_secs(10), "要求が受付されない");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    state.scheduler.lifecycle().begin_drain();
    // ここで新規要求は 503 (受付を止めた証拠)
    let (rej, _) = post(&app, "/v1/messages", chat_body()).await;
    assert_eq!(rej, StatusCode::SERVICE_UNAVAILABLE);

    // ワーカーを解放 → 受付済みの要求は打ち切られず完走する
    drop(guard);
    let resp = tokio::time::timeout(Duration::from_secs(30), task)
        .await
        .expect("drain 中に受付済みの要求が完了しない")
        .unwrap()
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "受付済みの要求が打ち切られた");

    // 進行中 + キューが 0 に戻る (run の drain 待ちと同じ条件)
    let start = Instant::now();
    while state.scheduler.pending() != 0 || state.scheduler.in_flight() != 0 {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "drain 待ちの条件が 10s 以内に満たされない: pending={} in_flight={}",
            state.scheduler.pending(),
            state.scheduler.in_flight()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
