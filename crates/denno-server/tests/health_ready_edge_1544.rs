//! Task 1544: `/health` / `/ready` / `begin_drain` の状態遷移の境界テスト。
//!
//! 契約は `src/lib.rs` と `src/lifecycle.rs` の実装に合わせる:
//!
//! - `/health` は liveness。常に 200 `{"status":"ok"}` で、認証も drain も見ない。
//! - `/ready` は `ready && worker_alive && !draining` のときだけ 200。それ以外は
//!   503 で、本文の `status` が理由 (`loading` / `draining`) を自己記述する。
//! - `begin_drain` は一方向かつ冪等。何度呼んでも状態は変わらず、`pending` /
//!   `in_flight` の会計も動かさない。`set_ready(true)` でも 200 には戻らない。
//! - 両エンドポイントは推論系ではないので `cache-control` を付けず、全応答に
//!   `x-content-type-options: nosniff` が付く。2xx の JSON だけ charset が付く。
//!
//! 同期方針: 固定 sleep で状態を作らず、prefill を Condvar のゲートで止めて
//! 「受付済みだが未完了」を決定的に作る。状態待ちは期限付きポーリング。

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DummyModel, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// deadlock / 状態待ちの期限。これを超えたら失敗にする。
const WAIT: Duration = Duration::from_secs(30);
/// 状態待ちのポーリング間隔。
const POLL: Duration = Duration::from_millis(5);

// ---------------------------------------------------------------------------
// アプリ生成
// ---------------------------------------------------------------------------

fn build_state(scheduler: Scheduler, api_key: Option<&str>) -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let mut info = EngineInfo::from_engine(&engine);
    info.model_dir = "qwen-test-dir".to_string();
    let engine = Arc::new(Mutex::new(engine));
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

/// モデルロード済み・ワーカー生存・非 drain。
fn ready_app(api_key: Option<&str>) -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine, 8);
    build_state(scheduler, api_key)
}

/// 起動途中 (モデル未ロード)。
fn loading_app() -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn_not_ready(engine, 8);
    build_state(scheduler, None)
}

// ---------------------------------------------------------------------------
// ゲート付きモデル (prefill を止めて in_flight を決定的に作る)
// ---------------------------------------------------------------------------

struct GatedModel {
    gate: Arc<(Mutex<bool>, Condvar)>,
}

impl GatedModel {
    fn new() -> (Self, Arc<(Mutex<bool>, Condvar)>) {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        (Self { gate: gate.clone() }, gate)
    }
}

fn release(gate: &Arc<(Mutex<bool>, Condvar)>) {
    let (lock, cv) = &**gate;
    *lock.lock().unwrap() = true;
    cv.notify_all();
}

impl Model for GatedModel {
    fn context(&self) -> usize {
        256
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        // prefill (最初の forward = cache が空) だけ止める。decode は進める。
        if !cache.is_empty() {
            let (lock, cv) = &*self.gate;
            let mut open = lock.lock().unwrap();
            while !*open {
                open = cv.wait(open).unwrap();
            }
        }
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.context())
    }
}

fn gated_app() -> (Router, Arc<AppState>, Arc<(Mutex<bool>, Condvar)>) {
    let (model, gate) = GatedModel::new();
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(model), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine, 8);
    let (app, state) = build_state(scheduler, None);
    (app, state, gate)
}

// ---------------------------------------------------------------------------
// HTTP ヘルパ
// ---------------------------------------------------------------------------

fn hdr(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

async fn request(
    app: &Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, HeaderMap, Vec<u8>) {
    let mut builder = Request::builder().method(method).uri(uri);
    let payload = match body {
        Some(v) => {
            builder = builder.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let req = builder.body(payload).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, bytes.to_vec())
}

async fn get_full(app: &Router, uri: &str) -> (StatusCode, HeaderMap, Value) {
    let (status, headers, bytes) = request(app, "GET", uri, None).await;
    (status, headers, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let (status, _headers, bytes) = request(app, "POST", uri, Some(body)).await;
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn messages_body() -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": false,
    })
}

/// `pred` が真になるまで期限付きで待つ。
async fn wait_until<F: Fn() -> bool>(what: &str, pred: F) {
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if pred() {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
    panic!("{what}: {} 秒以内に条件が満たされない", WAIT.as_secs());
}

/// 全応答に付くヘッダ衛生 (`nosniff`)。
fn assert_nosniff(headers: &HeaderMap, what: &str) {
    assert_eq!(
        hdr(headers, "x-content-type-options").as_deref(),
        Some("nosniff"),
        "{what}: x-content-type-options"
    );
}

/// 運用エンドポイントは推論系ではないので `cache-control` を付けない。
fn assert_no_store_absent(headers: &HeaderMap, what: &str) {
    assert_eq!(
        hdr(headers, "cache-control"),
        None,
        "{what}: cache-control を付けた"
    );
    assert_eq!(
        hdr(headers, "x-accel-buffering"),
        None,
        "{what}: x-accel-buffering を付けた"
    );
}

/// 503 のプローブ応答は Retry-After ではなく本文で理由を示す。
fn assert_no_retry_after(headers: &HeaderMap, what: &str) {
    assert_eq!(hdr(headers, "retry-after"), None, "{what}: Retry-After");
}

/// `/ready` の本文全体の形 (aborts / limits を含む) を固定する。
fn assert_ready_payload(body: &Value, status: &str, ready: bool, draining: bool) {
    assert_eq!(body["status"], status, "body = {body}");
    assert_eq!(body["ready"], ready, "body = {body}");
    assert_eq!(body["draining"], draining, "body = {body}");
    assert_eq!(body["model"], "denno-test", "body = {body}");
    assert!(
        body["context_length"].as_u64().unwrap_or(0) > 0,
        "context_length が正でない: {body}"
    );
    assert_eq!(
        body["limits"]["max_queue_depth"], 8,
        "max_queue_depth: {body}"
    );
    assert!(
        body["limits"]["request_timeout_secs"].is_u64(),
        "request_timeout_secs が数値でない: {body}"
    );
    for key in ["completed", "timeout", "disconnected", "error"] {
        assert!(
            body["aborts"][key].is_u64(),
            "aborts.{key} が数値でない: {body}"
        );
    }
    assert!(
        body["pending"].as_u64().is_some(),
        "pending が数値でない: {body}"
    );
    assert!(
        body["in_flight"].as_u64().is_some(),
        "in_flight が数値でない: {body}"
    );
}

// ---------------------------------------------------------------------------
// 1. アイドル: /health と /ready の 200 契約
// ---------------------------------------------------------------------------

#[tokio::test]
async fn idle_health_and_ready_return_200_with_security_headers() {
    // 認証を有効にしてもプローブはトークン無しで通る。
    let (app, _state) = ready_app(Some("secret"));

    let (status, headers, body) = get_full(&app, "/health").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body, json!({"status": "ok"}), "/health の本文");
    assert_eq!(
        body.as_object().map(|m| m.len()),
        Some(1),
        "/health は status だけを返す: {body}"
    );
    assert_eq!(
        hdr(&headers, "content-type").as_deref(),
        Some("application/json; charset=utf-8"),
        "/health content-type"
    );
    assert_nosniff(&headers, "/health");
    assert_no_store_absent(&headers, "/health");
    assert_no_retry_after(&headers, "/health");

    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_ready_payload(&body, "ready", true, false);
    assert_eq!(body["model_loaded"], true);
    assert_eq!(body["worker_alive"], true);
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    for key in ["completed", "timeout", "disconnected", "error"] {
        assert_eq!(body["aborts"][key], 0, "aborts.{key}: {body}");
    }
    assert_eq!(
        hdr(&headers, "content-type").as_deref(),
        Some("application/json; charset=utf-8"),
        "/ready content-type"
    );
    assert_nosniff(&headers, "/ready");
    assert_no_store_absent(&headers, "/ready");
    assert_no_retry_after(&headers, "/ready");
}

#[tokio::test]
async fn head_health_is_200_with_empty_body() {
    let (app, _state) = ready_app(None);
    let (status, headers, bytes) = request(&app, "HEAD", "/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(bytes.is_empty(), "HEAD /health に本文がある");
    assert_nosniff(&headers, "HEAD /health");
}

// ---------------------------------------------------------------------------
// 2. 起動途中: 503 "loading" とロード完了後の 200 遷移
// ---------------------------------------------------------------------------

#[tokio::test]
async fn loading_state_returns_503_then_becomes_ready() {
    let (app, state) = loading_app();

    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_ready_payload(&body, "loading", false, false);
    assert_eq!(body["model_loaded"], false);
    assert_eq!(body["worker_alive"], true);
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    // 失敗応答には charset を付けない (成功時だけ補う既存規則)。
    assert_eq!(
        hdr(&headers, "content-type").as_deref(),
        Some("application/json"),
        "/ready 503 content-type"
    );
    assert_nosniff(&headers, "/ready 503");
    assert_no_store_absent(&headers, "/ready 503");
    assert_no_retry_after(&headers, "/ready 503");

    // liveness はロード中でも 200。
    let (hs, _, hb) = get_full(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
    assert_eq!(hb["status"], "ok");

    // ロード完了 -> 200 "ready"。状態は一方向に進む。
    state.scheduler.lifecycle().set_ready(true);
    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_ready_payload(&body, "ready", true, false);
    assert_eq!(body["model_loaded"], true);
    assert_eq!(
        hdr(&headers, "content-type").as_deref(),
        Some("application/json; charset=utf-8")
    );
}

// ---------------------------------------------------------------------------
// 3. ワーカー死亡: worker_alive=false は drain ではない 503
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dead_worker_returns_503_without_draining() {
    let (app, state) = ready_app(None);
    state.scheduler.lifecycle().mark_worker_dead();

    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    // モデルは載っているが受付不可。drain ではないので理由は "loading"。
    assert_ready_payload(&body, "loading", false, false);
    assert_eq!(body["model_loaded"], true, "body = {body}");
    assert_eq!(body["worker_alive"], false, "body = {body}");
    assert_eq!(body["draining"], false, "body = {body}");
    assert_nosniff(&headers, "/ready");
    assert_no_retry_after(&headers, "/ready");

    // ワーカーが死んでも liveness は 200 (プロセスは生きている)。
    let (hs, _, hb) = get_full(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
    assert_eq!(hb["status"], "ok");
}

// ---------------------------------------------------------------------------
// 4. 活動中 -> drain: 会計を保ったまま 503 へ、begin_drain は冪等
// ---------------------------------------------------------------------------

#[tokio::test]
async fn active_request_ready_to_draining_is_idempotent() {
    let (app, state, gate) = gated_app();

    // prefill を止めて 1 本を in_flight に固定する。
    let a = app.clone();
    let held = tokio::spawn(async move { post(&a, "/v1/messages", messages_body()).await });
    wait_until("1 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 1
    })
    .await;

    // 活動中でも ready は 200 で、会計が本文に出る。
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_ready_payload(&body, "ready", true, false);
    assert_eq!(body["in_flight"], 1, "body = {body}");
    assert_eq!(body["pending"], 0, "body = {body}");

    // 1 回目の drain: 503 draining。受付済みの会計は保つ。
    state.scheduler.lifecycle().begin_drain();
    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_ready_payload(&body, "draining", false, true);
    assert_eq!(body["in_flight"], 1, "drain が会計を動かした: {body}");
    assert_eq!(body["pending"], 0, "body = {body}");
    assert_eq!(body["model_loaded"], true);
    assert_eq!(body["worker_alive"], true);
    assert_nosniff(&headers, "/ready 503");
    assert_no_retry_after(&headers, "/ready 503");
    assert_no_store_absent(&headers, "/ready 503");

    // 2 回目・3 回目の begin_drain は状態も会計も変えない (冪等)。
    let before = body.clone();
    state.scheduler.lifecycle().begin_drain();
    state.scheduler.lifecycle().begin_drain();
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body, before, "begin_drain の再呼び出しで本文が変わった");
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 1);

    // drain 中も liveness は 200。
    let (hs, _, hb) = get_full(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
    assert_eq!(hb["status"], "ok");

    // 受付済みの要求は完走し、会計が 0 に戻っても draining のまま。
    release(&gate);
    let (status, body) = tokio::time::timeout(WAIT, held)
        .await
        .expect("drain 中に受付済みの要求が完了しない")
        .unwrap();
    assert_eq!(status, StatusCode::OK, "受付済みの要求が打ち切られた: {body}");
    wait_until("完了後に会計が 0 になる", || {
        state.scheduler.pending() == 0 && state.scheduler.in_flight() == 0
    })
    .await;

    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_ready_payload(&body, "draining", false, true);
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    assert_eq!(body["model_loaded"], true);
    // 受付済みの要求は正常完了として 1 回だけ計上される。
    assert_eq!(body["aborts"]["completed"], 1, "body = {body}");
    assert_eq!(body["aborts"]["timeout"], 0, "body = {body}");
    assert_eq!(body["aborts"]["disconnected"], 0, "body = {body}");
    assert_eq!(body["aborts"]["error"], 0, "body = {body}");
}

// ---------------------------------------------------------------------------
// 5. 未ロードのまま drain: 一方向・冪等で、ready を立てても戻らない
// ---------------------------------------------------------------------------

#[tokio::test]
async fn drain_before_ready_is_permanent_and_idempotent() {
    let (app, state) = loading_app();

    // drain 前は 503 "loading"。
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "loading");

    state.scheduler.lifecycle().begin_drain();
    state.scheduler.lifecycle().begin_drain();
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    // 未ロードでも drain が優先して理由になる。
    assert_ready_payload(&body, "draining", false, true);
    assert_eq!(body["model_loaded"], false);
    assert_eq!(body["worker_alive"], true);
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);

    // ロード完了を記録しても drain は解除されない (一方向)。
    state.scheduler.lifecycle().set_ready(true);
    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_ready_payload(&body, "draining", false, true);
    assert_eq!(body["model_loaded"], true);
    assert_nosniff(&headers, "/ready");
    assert_no_retry_after(&headers, "/ready");

    // 追加の begin_drain でも本文は変わらない (冪等)。
    let before = body.clone();
    state.scheduler.lifecycle().begin_drain();
    let (_, _, body) = get_full(&app, "/ready").await;
    assert_eq!(body, before, "begin_drain の再呼び出しで本文が変わった");

    // drain 中の新規生成要求は受付しない (会計を増やさない)。
    let (status, resp) = post(&app, "/v1/messages", messages_body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {resp}");
    assert_eq!(resp["type"], "error");
    assert_eq!(resp["error"]["type"], "overloaded_error", "body = {resp}");
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);

    // liveness は drain 中でも 200 のまま。
    let (hs, _, hb) = get_full(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
    assert_eq!(hb["status"], "ok");
}
