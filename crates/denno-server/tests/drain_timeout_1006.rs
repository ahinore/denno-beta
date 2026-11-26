//! Task 1006: 停止時の drain 上限 `DENNO_DRAIN_TIMEOUT_SECS` の解決と、
//! 「要求が早く終わる」「上限に達する」の両方で drain が正しく終わること。
//!
//! 契約 (`src/lifecycle.rs` の実装に合わせる):
//!
//! - 優先順は `DENNO_DRAIN_TIMEOUT_SECS` (秒) > `DENNO_DRAIN_TIMEOUT_MS` (ms)
//!   > 既定 10 s。どちらも 600 s にクランプし、0 は「待たない」として保持する。
//! - 判定は `drain_step(is_idle, elapsed, timeout)` の純関数に集約:
//!   アイドルなら常に完了 (`Idle`)、そうでなく `elapsed >= timeout` なら
//!   打ち切り (`TimedOut`)、それ以外は継続 (`Wait`)。
//! - 待ち (`wait_for_idle`) は `pending == 0 && in_flight == 0` で完了し、
//!   上限に達したら false で戻る。打ち切っても受付済みの要求は完走する。
//!
//! 同期方針: 固定 sleep で状態を作らない。テスト用モデルの prefill を
//! Condvar のゲートで止め、「受付済みだが未完了」を時間に依存せず作る。

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{
    drain_step, drain_timeout, resolve_drain_timeout, wait_for_idle, AppState, DrainStep,
    EngineInfo, Lifecycle, Scheduler, DEFAULT_DRAIN_TIMEOUT_SECS, DRAIN_TIMEOUT_SECS_ENV,
    MAX_DRAIN_TIMEOUT_SECS,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// deadlock を失敗として表面化させるための期限。
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
        // prefill (最初の forward) だけ止める。decode は素直に進める。
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
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn messages_request() -> Request<Body> {
    let body = json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    });
    Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
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

// ---------------------------------------------------------------------------
// 1. env 解決 (純関数・決定的)
// ---------------------------------------------------------------------------

#[test]
fn resolve_prefers_secs_env_then_ms_env_then_default() {
    // secs が最優先。ms があっても無視する。
    assert_eq!(
        resolve_drain_timeout(Some("3"), Some("9000")),
        Duration::from_secs(3)
    );
    // secs が無効なら ms。
    assert_eq!(
        resolve_drain_timeout(Some("abc"), Some("250")),
        Duration::from_millis(250)
    );
    assert_eq!(
        resolve_drain_timeout(None, Some(" 1500 ")),
        Duration::from_millis(1500)
    );
    // どちらも無効なら既定 10 s。
    assert_eq!(
        resolve_drain_timeout(None, None),
        Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
    );
    assert_eq!(
        resolve_drain_timeout(Some(""), Some("-5")),
        Duration::from_secs(DEFAULT_DRAIN_TIMEOUT_SECS)
    );
}

#[test]
fn resolve_clamps_to_max_and_keeps_explicit_zero() {
    assert_eq!(
        resolve_drain_timeout(Some("100000"), None),
        Duration::from_secs(MAX_DRAIN_TIMEOUT_SECS)
    );
    assert_eq!(
        resolve_drain_timeout(None, Some("100000000")),
        Duration::from_millis(MAX_DRAIN_TIMEOUT_SECS * 1_000)
    );
    // 0 は「待たない」の明示。既定に化けない。
    assert_eq!(resolve_drain_timeout(Some("0"), None), Duration::ZERO);
    assert_eq!(resolve_drain_timeout(None, Some("0")), Duration::ZERO);
}

/// env を実際に読む経路も 1 本だけ固定する (env を触るので直列化する)。
#[test]
fn drain_timeout_reads_secs_env_with_precedence() {
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let old_secs = std::env::var(DRAIN_TIMEOUT_SECS_ENV).ok();
    let old_ms = std::env::var("DENNO_DRAIN_TIMEOUT_MS").ok();

    std::env::set_var(DRAIN_TIMEOUT_SECS_ENV, "2");
    std::env::set_var("DENNO_DRAIN_TIMEOUT_MS", "9000");
    assert_eq!(drain_timeout(), Duration::from_secs(2), "secs が ms に勝つ");

    std::env::set_var(DRAIN_TIMEOUT_SECS_ENV, "not-a-number");
    assert_eq!(
        drain_timeout(),
        Duration::from_millis(9000),
        "secs が無効なら ms に落ちる"
    );

    // 後始末 (このバイナリの他のテストを汚さない)。
    match old_secs {
        Some(v) => std::env::set_var(DRAIN_TIMEOUT_SECS_ENV, v),
        None => std::env::remove_var(DRAIN_TIMEOUT_SECS_ENV),
    }
    match old_ms {
        Some(v) => std::env::set_var("DENNO_DRAIN_TIMEOUT_MS", v),
        None => std::env::remove_var("DENNO_DRAIN_TIMEOUT_MS"),
    }
}

// ---------------------------------------------------------------------------
// 2. 判定の純関数 (3 分岐を数えて発火を証明する)
// ---------------------------------------------------------------------------

#[test]
fn drain_step_covers_idle_wait_and_timeout() {
    let timeout = Duration::from_millis(100);
    let mut idle = 0usize;
    let mut wait = 0usize;
    let mut timed_out = 0usize;
    for (is_idle, elapsed_ms, want) in [
        (true, 0u64, DrainStep::Idle),
        (true, 100, DrainStep::Idle),
        (false, 0, DrainStep::Wait),
        (false, 99, DrainStep::Wait),
        (false, 100, DrainStep::TimedOut),
        (false, 101, DrainStep::TimedOut),
    ] {
        let got = drain_step(is_idle, Duration::from_millis(elapsed_ms), timeout);
        assert_eq!(got, want, "idle={is_idle} elapsed={elapsed_ms}");
        assert_eq!(got.is_final(), want != DrainStep::Wait);
        match got {
            DrainStep::Idle => idle += 1,
            DrainStep::Wait => wait += 1,
            DrainStep::TimedOut => timed_out += 1,
        }
    }
    assert_eq!((idle, wait, timed_out), (2, 2, 2), "3 分岐すべて発火");
    // 上限 0: アイドルでなければ即打ち切り、アイドルなら成功。
    assert_eq!(
        drain_step(false, Duration::ZERO, Duration::ZERO),
        DrainStep::TimedOut
    );
    assert_eq!(
        drain_step(true, Duration::ZERO, Duration::ZERO),
        DrainStep::Idle
    );
}

// ---------------------------------------------------------------------------
// 3. 要求が早く終わる drain は完了する (上限に張り付かない)
// ---------------------------------------------------------------------------

/// 1 本を in_flight に固定し、drain を開始してからゲートを開ける。
/// `wait_for_idle` は true で戻り、上限 (10 s) より十分早く終わる。
#[tokio::test]
async fn drain_completes_when_requests_finish_early() {
    let (app, state, gate) = gated_state();
    let a = app.clone();
    let held = tokio::spawn(async move { a.oneshot(messages_request()).await.unwrap() });

    wait_until("1 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 1
    })
    .await;

    // 停止要求: 以後の新規は 503。
    state.scheduler.lifecycle().begin_drain();
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["in_flight"], 1, "進行中は残っている: {body}");

    // まだ完了していないので、短い上限では false (打ち切り経路の前段)。
    let not_yet = wait_for_idle(
        || scheduler_idle(&state),
        Duration::from_millis(100),
        POLL,
    )
    .await;
    assert!(!not_yet, "進行中があるのに idle と判定された");

    // 解放 -> 要求は完走し、drain は true で終わる。
    release(&gate);
    let timeout = Duration::from_secs(10);
    let start = Instant::now();
    let idle = tokio::time::timeout(WAIT, wait_for_idle(|| scheduler_idle(&state), timeout, POLL))
        .await
        .expect("drain の待ちが期限を超えた");
    let elapsed = start.elapsed();
    assert!(idle, "要求が終わったのに drain が完了しない");
    assert!(
        elapsed < timeout,
        "完了した drain が上限に張り付いた: {elapsed:?}"
    );

    let resp = tokio::time::timeout(WAIT, held)
        .await
        .expect("受付済みの要求が完了しない")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "drain で要求が打ち切られた");

    // 会計は 0、readiness は draining のまま、liveness は 200。
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    assert_eq!(get(&app, "/health").await.0, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// 4. 上限が来たら drain は false で打ち切る (要求は生かしたまま)
// ---------------------------------------------------------------------------

/// ゲートを閉じたまま (in_flight=1 を維持) `wait_for_idle` を短い上限で回す。
/// false で戻り、経過が上限以上。打ち切り後も要求は完走する (殺さない)。
#[tokio::test]
async fn drain_times_out_while_request_still_in_flight() {
    let (app, state, gate) = gated_state();
    let a = app.clone();
    let held = tokio::spawn(async move { a.oneshot(messages_request()).await.unwrap() });

    wait_until("1 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 1
    })
    .await;

    state.scheduler.lifecycle().begin_drain();

    // 解決した上限が「要求より短い」ことを固定する (env 経路の意味)。
    let timeout = resolve_drain_timeout(Some("0"), None);
    assert_eq!(timeout, Duration::ZERO, "0 = 待たない");
    let timeout = Duration::from_millis(120);

    let start = Instant::now();
    let idle = wait_for_idle(|| scheduler_idle(&state), timeout, Duration::from_millis(10)).await;
    let elapsed = start.elapsed();
    assert!(!idle, "進行中が残っているのに idle と判定された");
    assert!(
        elapsed >= timeout,
        "上限より前に打ち切った: timeout={timeout:?} elapsed={elapsed:?}"
    );
    assert!(
        elapsed < timeout + Duration::from_secs(5),
        "打ち切りが上限から大きく遅れた: {elapsed:?}"
    );
    // 打ち切りは状態を書き換えない (会計は進行中のまま)。
    assert_eq!(state.scheduler.in_flight(), 1);
    assert_eq!(state.scheduler.pending(), 0);

    // 打ち切っても受付済みの要求は完走する。
    release(&gate);
    let resp = tokio::time::timeout(WAIT, held)
        .await
        .expect("打ち切りで要求が失われた")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    wait_until("完了後に会計が 0 になる", || scheduler_idle(&state)).await;
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining", "打ち切り後も受付は再開しない");
}

// ---------------------------------------------------------------------------
// 5. 上限 0 の drain は待たずに即 false / アイドルなら即 true
// ---------------------------------------------------------------------------

#[tokio::test]
async fn zero_timeout_returns_immediately() {
    let (app, state, gate) = gated_state();
    let a = app.clone();
    let held = tokio::spawn(async move { a.oneshot(messages_request()).await.unwrap() });
    wait_until("1 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 1
    })
    .await;
    state.scheduler.lifecycle().begin_drain();

    let start = Instant::now();
    let idle = wait_for_idle(|| scheduler_idle(&state), Duration::ZERO, POLL).await;
    assert!(!idle, "上限 0 で進行中があるのに待った");
    assert!(start.elapsed() < Duration::from_millis(500));

    release(&gate);
    let resp = tokio::time::timeout(WAIT, held)
        .await
        .expect("要求が完了しない")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    wait_until("完了後に会計が 0 になる", || scheduler_idle(&state)).await;
    assert!(
        wait_for_idle(|| scheduler_idle(&state), Duration::ZERO, POLL).await,
        "アイドルなら上限 0 でも完了"
    );
}

// ---------------------------------------------------------------------------
// 6. Lifecycle の drain 状態は wait_for_idle と独立に一方向
// ---------------------------------------------------------------------------

#[test]
fn lifecycle_drain_flag_is_one_way() {
    let lc = Lifecycle::new(true);
    assert!(lc.serves_requests());
    lc.begin_drain();
    assert!(lc.is_draining());
    assert!(!lc.serves_requests());
    lc.set_ready(true);
    assert!(!lc.serves_requests(), "drain は取り消せない");
}
