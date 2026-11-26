//! Task 1274: drain の境界 (上限 0 / 進行中 SSE の完走 / `/ready` の 503 遷移)。
//!
//! 契約は `src/lifecycle.rs` と `src/lib.rs` の実装に合わせる:
//!
//! - `wait_for_idle` は `is_idle` が真なら上限 0 でも即 `true`、そうでなければ
//!   上限 0 では待たずに即 `false`。
//! - drain 中も**受付済み**のストリーミング要求は打ち切られない。SSE は最後まで
//!   イベントを流し切り、`message_stop` で終わり、その完了で `in_flight` が 0 に
//!   戻ってから drain 待ちが `true` で終わる (上限には張り付かない)。
//! - `/ready` は `ready && worker_alive && !draining` のときだけ 200。drain 開始で
//!   503 `status == "draining"` に落ち、以後 `set_ready(true)` でも 200 に戻らない
//!   (一方向)。drain 中は新規生成要求が 503 `overloaded_error` で拒否され、
//!   `pending` / `in_flight` を増やさない。`/health` (liveness) は drain 中も 200。
//!
//! 同期方針: 固定 sleep で状態を作らない。prefill を Condvar のゲートで止めて
//! 「受付済みだが未完了」を決定的に作り、待ちは期限付きポーリングにする。

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{wait_for_idle, AppState, EngineInfo, Lifecycle, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// deadlock を失敗として表面化させる期限。
const WAIT: Duration = Duration::from_secs(30);
/// 状態待ちのポーリング間隔。
const POLL: Duration = Duration::from_millis(5);

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

// ---------------------------------------------------------------------------
// app / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn gated_state() -> (Router, Arc<AppState>, Arc<(Mutex<bool>, Condvar)>) {
    let (model, gate) = GatedModel::new();
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(model), tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    (denno_server::router(state.clone()), state, gate)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let (status, _headers, body) = get_full(app, uri).await;
    (status, body)
}

/// ヘッダ込みの GET (503 に Retry-After が付かないことの確認用)。
async fn get_full(app: &Router, uri: &str) -> (StatusCode, HeaderMap, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, body)
}

fn messages_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

async fn post(app: &Router, body: Value) -> (StatusCode, HeaderMap, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).into_owned())
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

/// run の drain 待ちと同じ条件 (`pending == 0 && in_flight == 0`)。
fn scheduler_idle(state: &AppState) -> bool {
    state.scheduler.pending() == 0 && state.scheduler.in_flight() == 0
}

fn sse_events(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str::<Value>(d).expect("SSE data 行が JSON でない"))
        .collect()
}

/// SSE 応答が「正常に流し切られた」ことを固定する。
fn assert_sse_completed(events: &[Value], body: &str) {
    assert!(!events.is_empty(), "SSE イベントが 1 つも無い: {body}");
    assert_eq!(events.first().unwrap()["type"], "message_start");
    assert_eq!(events.last().unwrap()["type"], "message_stop");
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "message_stop")
            .count(),
        1,
        "message_stop はちょうど 1 回"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "message_start")
            .count(),
        1,
        "message_start はちょうど 1 回"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "message_delta")
            .count(),
        1,
        "message_delta はちょうど 1 回"
    );
    assert!(
        events
            .iter()
            .any(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta"),
        "本文に text_delta が無い: {body}"
    );
    assert!(
        !events.iter().any(|e| e["type"] == "error"),
        "drain で途中終了した (error イベント): {body}"
    );
}

// ---------------------------------------------------------------------------
// 1. 上限 0 の drain は待たない (進行中でも即 false / 空なら即 true)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn zero_timeout_drain_never_waits_with_active_sse() {
    let (app, state, gate) = gated_state();

    // ストリーミング要求を 1 本 in_flight に固定する。
    let a = app.clone();
    let held = tokio::spawn(async move { post(&a, messages_body(true)).await });
    wait_until("1 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 1
    })
    .await;

    state.scheduler.lifecycle().begin_drain();

    // 上限 0: 進行中がある場合は待たずに即 false。
    let start = Instant::now();
    let idle = wait_for_idle(|| scheduler_idle(&state), Duration::ZERO, POLL).await;
    let elapsed = start.elapsed();
    assert!(!idle, "上限 0 で進行中があるのに待った");
    assert!(
        elapsed < Duration::from_millis(500),
        "上限 0 なのに待った: {elapsed:?}"
    );

    // drain 待ちは打ち切っても受付済みの SSE は完走する。
    release(&gate);
    let (status, headers, body) = tokio::time::timeout(WAIT, held)
        .await
        .expect("上限 0 の打ち切りで要求が失われた")
        .unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "text/event-stream"
    );
    assert_sse_completed(&sse_events(&body), &body);

    // 完了すれば上限 0 でも idle は true。readiness は draining のまま。
    wait_until("完了後に会計が 0 になる", || scheduler_idle(&state)).await;
    assert!(
        wait_for_idle(|| scheduler_idle(&state), Duration::ZERO, POLL).await,
        "アイドルなら上限 0 でも完了"
    );
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
}

// ---------------------------------------------------------------------------
// 2. 進行中の SSE は上限より前に完走し、その完了で drain が終わる
// ---------------------------------------------------------------------------

#[tokio::test]
async fn active_sse_completes_before_timeout() {
    let (app, state, gate) = gated_state();

    // drain 前は /ready は 200 (ready) であること。
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["draining"], false);

    let a = app.clone();
    let held = tokio::spawn(async move { post(&a, messages_body(true)).await });
    wait_until("1 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 1
    })
    .await;

    // 停止要求: /ready は 503 draining、進行中は残る。
    let lc: &Lifecycle = state.scheduler.lifecycle();
    lc.begin_drain();
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["in_flight"], 1, "body = {body}");
    assert_eq!(body["pending"], 0, "body = {body}");

    // まだ完了していないので、短い上限では false (打ち切り経路)。
    let not_yet = wait_for_idle(
        || scheduler_idle(&state),
        Duration::from_millis(100),
        POLL,
    )
    .await;
    assert!(!not_yet, "進行中があるのに idle と判定された");

    // 解放 -> SSE は完走し、drain は上限より十分早く true で終わる。
    release(&gate);
    let timeout = Duration::from_secs(10);
    let start = Instant::now();
    let idle = tokio::time::timeout(WAIT, wait_for_idle(|| scheduler_idle(&state), timeout, POLL))
        .await
        .expect("drain の待ちが期限を超えた");
    let elapsed = start.elapsed();
    assert!(idle, "SSE が完了したのに drain が終わらない");
    assert!(
        elapsed < timeout,
        "完了した drain が上限に張り付いた: {elapsed:?}"
    );

    // 受付済みの SSE は最後まで流れた (message_stop で終わる)。
    let (status, headers, body) = tokio::time::timeout(WAIT, held)
        .await
        .expect("drain 中に SSE が完了しない")
        .unwrap();
    assert_eq!(status, StatusCode::OK, "drain で SSE が打ち切られた");
    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    assert_sse_completed(&sse_events(&body), &body);

    // 会計は 0、readiness は draining のまま、liveness は 200。
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    assert_eq!(get(&app, "/health").await.0, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// 3. /ready の 200 -> 503 遷移はきれいで一方向
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ready_transition_to_503_is_clean_and_permanent() {
    let (app, state, _gate) = gated_state();

    // 起動直後: 200 ready。
    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["ready"], true);
    assert_eq!(body["draining"], false);
    assert_eq!(body["model_loaded"], true);
    assert_eq!(body["worker_alive"], true);
    assert!(
        headers.get("retry-after").is_none(),
        "/ready の 200 に Retry-After を付けない"
    );

    // drain 開始: 503 draining。理由フィールドが自己記述で、Retry-After は無い。
    state.scheduler.lifecycle().begin_drain();
    let (status, headers, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["ready"], false);
    assert_eq!(body["draining"], true);
    assert_eq!(body["model_loaded"], true, "モデルは載ったまま: {body}");
    assert_eq!(body["worker_alive"], true, "ワーカーは生きている: {body}");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    assert!(
        headers.get("retry-after").is_none(),
        "503 は Retry-After ではなく本文で理由を示す"
    );

    // drain は一方向: ready を立て直しても 200 に戻らない。
    state.scheduler.lifecycle().set_ready(true);
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["model_loaded"], true);

    // liveness は drain 中も 200。
    let (status, _, body) = get_full(&app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");

    // drain 中の新規生成要求は 503 overloaded_error で拒否し、キューを増やさない。
    let (status, headers, body) = post(&app, messages_body(false)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    let v: Value = serde_json::from_str(&body).expect("エラー本文が JSON でない");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "overloaded_error", "body = {body}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("draining"),
        "拒否理由が本文に無い: {body}"
    );
    assert!(
        headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .starts_with("application/json"),
        "拒否は SSE ではなく JSON 本文"
    );

    // ストリーミング要求も同様に受付しない (SSE を開かない)。
    let (status, headers, body) = post(&app, messages_body(true)).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_ne!(
        headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default(),
        "text/event-stream",
        "drain 中に SSE を開いた"
    );

    // 拒否しても /ready の会計は 0 のまま (キューに積んでいない)。
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
    let (status, _, body) = get_full(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
}
