//! Task 944: ライフサイクル (/ready・drain・graceful shutdown) の統合テスト。
//!
//! 固定する契約 (`src/lifecycle.rs` と `src/lib.rs` の実装に合わせる):
//!
//! - `/health` は liveness。常に 200 (`{"status":"ok"}`)。drain 中も 200。
//! - `/ready` は 200 になる条件が `ready && worker_alive && !draining` の
//!   3 つだけ。それ以外は 503 で、本文の `status` が理由を示す
//!   (`loading` = 未ロードまたはワーカー死亡 / `draining` = 停止中)。
//! - 503 に `Retry-After` は付けない。代わりに本文が自己記述
//!   (`status` / `ready` / `draining` / `model_loaded` / `worker_alive` /
//!   `pending` / `in_flight`) なので、クライアントは本文で再試行可否を判断する。
//!   `Retry-After` を「有る」と仮定するテストは書かない。
//! - 停止要求 (`begin_drain`) は一方向。以後 `set_ready(true)` でも 200 に戻らない。
//! - drain 中、新規の生成要求は 503 overloaded で**受付しない** (キューに積まない)。
//!   受付済みの要求は打ち切られず完走する。
//! - drain 完了条件は `pending() == 0 && in_flight() == 0` (run の待ちと同じ)。
//!
//! 同期方針: 固定 sleep での待ち合わせはしない。モデルの prefill を
//! Condvar のゲートで止め、テスト側は「状態が目的の値になるまで」を
//! 期限付きでポーリングする。待ち行列/スロットは観測値 (pending/in_flight/
//! 後続要求の成否) で判定する。

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// テストが待つ上限。deadlock を「失敗」として表面化させるための期限。
const WAIT: Duration = Duration::from_secs(30);
/// ポーリング間隔 (待ち合わせの最小単位。固定 sleep ではない)。
const POLL: Duration = Duration::from_millis(5);

// ---------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------

/// prefill の forward をゲートで止めるモデル。
///
/// ゲートが開くまで worker スレッドは forward の中でブロックする。これで
/// 「受付済み (in_flight に計上済み) だが生成は未完了」の状態を、時間に
/// 依存せず作れる。ゲートは一度開くと戻らない (全待ち手が同時に進む)。
struct GatedModel {
    gate: Arc<(Mutex<bool>, Condvar)>,
    ctx: usize,
}

impl GatedModel {
    fn new() -> (Self, Arc<(Mutex<bool>, Condvar)>) {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        (
            Self {
                gate: gate.clone(),
                ctx: 256,
            },
            gate,
        )
    }

    fn wait_open(&self) {
        let (lock, cv) = &*self.gate;
        let mut open = lock.lock().unwrap();
        while !*open {
            open = cv.wait(open).unwrap();
        }
    }
}

/// ゲートを開ける (テスト側)。
fn release(gate: &Arc<(Mutex<bool>, Condvar)>) {
    let (lock, cv) = &**gate;
    *lock.lock().unwrap() = true;
    cv.notify_all();
}

impl Model for GatedModel {
    fn context(&self) -> usize {
        self.ctx
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        // prefill (最初の forward) だけ止める。decode は素直に進める。
        if !cache.is_empty() {
            self.wait_open();
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
        KvCache::new(1, 4, self.ctx)
    }
}

// ---------------------------------------------------------------------------
// app / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn state_with(spawn: fn(Arc<Mutex<Engine>>, usize) -> Scheduler, model: Box<dyn Model>) -> Arc<AppState> {
    state_with_queue(spawn, model, 8)
}

fn state_with_queue(
    spawn: fn(Arc<Mutex<Engine>>, usize) -> Scheduler,
    model: Box<dyn Model>,
    max_queue: usize,
) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = spawn(engine.clone(), max_queue);
    Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

fn ready_state() -> (Router, Arc<AppState>) {
    let state = state_with(Scheduler::spawn, Box::new(denno_engine::testutil::DummyModel));
    (denno_server::router(state.clone()), state)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

fn chat_body() -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

fn messages_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

fn post_request(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(post_request(uri, body)).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// `pred` が真になるまで期限付きで待つ。満たさなければ `what` を添えて panic。
async fn wait_until<F: Fn() -> bool>(what: &str, pred: F) {
    let deadline = Instant::now() + WAIT;
    while Instant::now() < deadline {
        if pred() {
            return;
        }
        tokio::time::sleep(POLL).await;
    }
    panic!("{what}: {} 以内に条件が満たされない", WAIT.as_secs_f64());
}

/// run の drain 待ちと同じ条件 (`pending == 0 && in_flight == 0`) を、
/// 上限付きで待つ。`lifecycle::wait_for_idle` は crate 内部 (非公開) なので、
/// 統合テスト側で同じ条件・同じ上限意味論を再現する。
async fn wait_idle_bounded(state: &AppState, timeout: Duration, poll: Duration) -> bool {
    let start = Instant::now();
    loop {
        if state.scheduler.pending() == 0 && state.scheduler.in_flight() == 0 {
            return true;
        }
        if start.elapsed() >= timeout {
            return false;
        }
        let left = timeout.saturating_sub(start.elapsed());
        tokio::time::sleep(poll.min(left)).await;
    }
}

/// `/ready` 本文が契約どおりの形かを検証する (全レスポンス共通)。
fn assert_ready_shape(body: &Value) {
    for key in [
        "status",
        "ready",
        "draining",
        "model_loaded",
        "worker_alive",
        "model",
        "context_length",
        "pending",
        "in_flight",
    ] {
        assert!(body.get(key).is_some(), "/ready 本文に {key} が無い: {body}");
    }
    assert!(body["status"].is_string(), "status は文字列: {body}");
    for key in ["ready", "draining", "model_loaded", "worker_alive"] {
        assert!(body[key].is_boolean(), "{key} は boolean: {body}");
    }
    for key in ["context_length", "pending", "in_flight"] {
        assert!(body[key].is_u64(), "{key} は非負整数: {body}");
    }
    let consistent = body["status"] == "ready";
    assert_eq!(
        body["ready"].as_bool(),
        Some(consistent),
        "200 のときだけ ready=true (status と整合): {body}"
    );
    if consistent {
        assert_eq!(body["draining"], false, "200 で draining=true は矛盾: {body}");
        assert_eq!(body["model_loaded"], true, "200 で model_loaded=false は矛盾: {body}");
        assert_eq!(body["worker_alive"], true, "200 で worker_alive=false は矛盾: {body}");
    }
}

// ---------------------------------------------------------------------------
// 1. /ready の状態遷移
// ---------------------------------------------------------------------------

/// starting (loading) -> ready -> draining -> 503、の全段を 1 本で踏む。
/// 各段で /health は 200 のまま (liveness と readiness の分離)。
#[tokio::test]
async fn ready_transitions_loading_to_ready_to_draining() {
    // starting: モデル未ロードの state (起動直後と同じ)。
    let (app, state) = {
        let s = state_with(
            Scheduler::spawn_not_ready,
            Box::new(denno_engine::testutil::DummyModel),
        );
        (denno_server::router(s.clone()), s)
    };

    let (status, body) = get(&app, "/ready").await;
    assert_ready_shape(&body);
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "loading");
    assert_eq!(body["ready"], false);
    assert_eq!(body["model_loaded"], false);
    assert_eq!(body["worker_alive"], true, "ワーカーは起動済み: {body}");
    assert_eq!(body["draining"], false);
    let (hs, hb) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);
    assert_eq!(hb["status"], "ok");

    // ready: 起動経路がモデルロード完了を記録するのと同じ操作。
    state.scheduler.lifecycle().set_ready(true);
    let (status, body) = get(&app, "/ready").await;
    assert_ready_shape(&body);
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["ready"], true);
    assert_eq!(body["model_loaded"], true);
    assert_eq!(body["worker_alive"], true);
    assert_eq!(body["draining"], false);
    assert_eq!(body["model"], "denno-test");

    // draining: 停止要求。以後 503 + status=draining で、本文で理由が分かる。
    state.scheduler.lifecycle().begin_drain();
    let (status, body) = get(&app, "/ready").await;
    assert_ready_shape(&body);
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["ready"], false);
    assert_eq!(body["draining"], true);
    assert_eq!(body["worker_alive"], true, "停止要求はワーカーを殺さない: {body}");

    // liveness は drain 中も 200。
    let (hs, _) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);

    // 一方向: ready を立て直しても 200 には戻らない (draining が勝つ)。
    state.scheduler.lifecycle().set_ready(true);
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "drain は取り消せない: {body}");
    assert_eq!(body["status"], "draining");
}

/// ワーカー死亡は loading 扱いの 503 (draining とは別の理由)。
#[tokio::test]
async fn ready_returns_503_when_worker_died() {
    let (app, state) = ready_state();
    assert_eq!(get(&app, "/ready").await.0, StatusCode::OK);

    state.scheduler.lifecycle().mark_worker_dead();
    let (status, body) = get(&app, "/ready").await;
    assert_ready_shape(&body);
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["worker_alive"], false);
    assert_eq!(body["draining"], false, "異常終了は drain ではない: {body}");
    assert_eq!(body["status"], "loading");
    assert_eq!(body["model_loaded"], true, "モデルは載っている: {body}");

    // liveness は 200 のまま (プロセス自体は生きている)。
    assert_eq!(get(&app, "/health").await.0, StatusCode::OK);
}

/// 503 の本文は自己記述で、`Retry-After` は付けない (実装の契約を固定)。
#[tokio::test]
async fn ready_503_body_is_self_describing_without_retry_after() {
    let (app, state) = ready_state();
    state.scheduler.lifecycle().begin_drain();

    let req = Request::builder()
        .method("GET")
        .uri("/ready")
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert!(
        resp.headers().get("retry-after").is_none(),
        "503 に Retry-After は付けない (本文の status で判断する契約)"
    );
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "application/json",
        "本文は JSON"
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_ready_shape(&body);
    assert_eq!(body["status"], "draining");
}

// ---------------------------------------------------------------------------
// 2. drain 中の受付拒否 + 受付済みは完走
// ---------------------------------------------------------------------------

/// 3 本を「受付済み・未完了」のまま同時に保持し、drain を挟んで
/// 新規は 503・既存は 200 で完走することを確かめる。
///
/// 完了待ちは run と同じ
/// `pending() == 0 && in_flight() == 0` を上限付きで待つ。
#[tokio::test]
async fn in_flight_finish_during_drain_and_new_requests_are_rejected() {
    let (model, gate) = GatedModel::new();
    let state = state_with(Scheduler::spawn, Box::new(model));
    let app: Router = denno_server::router(state.clone());

    // 3 本を同時に投げる (MAX_ACTIVE_SESSIONS = 4 以内なので全部 admit される)。
    let mut tasks = Vec::new();
    for _ in 0..3 {
        let a = app.clone();
        tasks.push(tokio::spawn(async move {
            a.oneshot(post_request("/v1/messages", messages_body()))
                .await
                .unwrap()
        }));
    }

    wait_until("3 本が in_flight に計上される", || {
        state.scheduler.in_flight() == 3 && state.scheduler.pending() == 0
    })
    .await;

    // drain 前は受付可能。
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["in_flight"], 3, "本文の in_flight が実測と一致: {body}");

    // 停止要求 → /ready は draining。
    state.scheduler.lifecycle().begin_drain();
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["in_flight"], 3, "進行中はそのまま: {body}");

    // 新規は Anthropic / OpenAI とも受付しない。キューは増えない。
    let (s1, b1) = post(&app, "/v1/messages", messages_body()).await;
    assert_eq!(s1, StatusCode::SERVICE_UNAVAILABLE, "body = {b1}");
    assert_eq!(b1["error"]["type"], "overloaded_error");
    assert!(
        b1["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("draining"),
        "body = {b1}"
    );
    let (s2, b2) = post(&app, "/v1/chat/completions", chat_body()).await;
    assert_eq!(s2, StatusCode::SERVICE_UNAVAILABLE, "body = {b2}");
    assert!(
        b2["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("draining"),
        "body = {b2}"
    );
    assert_eq!(state.scheduler.pending(), 0, "拒否した要求はキューに積まない");
    assert_eq!(state.scheduler.in_flight(), 3, "進行中は減らない");

    // drain 待ちが終わらないことを、短い上限で「まだ idle でない」と確認する
    // (固定 sleep ではなく、ゲートが閉じている事実の確認)。
    let not_yet = wait_idle_bounded(&state, Duration::from_millis(30), Duration::from_millis(5)).await;
    assert!(!not_yet, "進行中があるのに idle と判定された");

    // 進行中を解放 → 3 本とも打ち切られず 200 で完走。
    release(&gate);
    for (i, t) in tasks.into_iter().enumerate() {
        let resp = tokio::time::timeout(WAIT, t)
            .await
            .unwrap_or_else(|_| panic!("受付済みの要求 {i} が完了しない"))
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "要求 {i} が drain で打ち切られた");
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["type"], "message", "要求 {i} の本文が壊れた: {body}");
    }

    // drain 完了条件 (run と同じ) が満たされる。
    let idle = tokio::time::timeout(WAIT, wait_idle_bounded(&state, WAIT, POLL))
        .await
        .expect("drain の待ちが期限を超えた");
    assert!(idle, "完了後も pending/in_flight が 0 にならない");

    // 停止中も liveness は 200、readiness は draining のまま。
    assert_eq!(get(&app, "/health").await.0, StatusCode::OK);
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["status"], "draining");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
}

/// drain 中の拒否を連打してもキュー・スロットが増えない (受付を素通りしない)。
#[tokio::test]
async fn repeated_rejections_during_drain_do_not_queue_or_leak_slots() {
    let (model, gate) = GatedModel::new();
    let state = state_with(Scheduler::spawn, Box::new(model));
    let app: Router = denno_server::router(state.clone());

    // 1 本だけ受付済みで止めておく (drain の待ち対象)。
    let a = app.clone();
    let held = tokio::spawn(async move {
        a.oneshot(post_request("/v1/messages", messages_body()))
            .await
            .unwrap()
    });
    wait_until("受付済みの要求が 1 本になる", || {
        state.scheduler.in_flight() == 1
    })
    .await;

    state.scheduler.lifecycle().begin_drain();
    for i in 0..100u32 {
        let (status, body) = post(&app, "/v1/messages", messages_body()).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "反復 {i}: body = {body}");
        assert_eq!(body["error"]["type"], "overloaded_error", "反復 {i}");
    }
    assert_eq!(state.scheduler.pending(), 0, "拒否 100 回でキューが増えた");
    assert_eq!(state.scheduler.in_flight(), 1, "拒否で進行中が変わった");

    release(&gate);
    let resp = tokio::time::timeout(WAIT, held)
        .await
        .expect("受付済みの要求が完了しない")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // 停止後も後続要求は 503 のまま (スロットが解放されても受付は再開しない)。
    let (status, _) = post(&app, "/v1/messages", messages_body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}

// ---------------------------------------------------------------------------
// 3. 負荷中の /health・/ready 連打 (deadlock / slot leak 無し)
// ---------------------------------------------------------------------------

/// 生成 12 本 (MAX_ACTIVE_SESSIONS = 4 を超えるので必ず同時実行窓ができる) と
/// 並行に /health・/ready を連打する。全体を timeout で包み、deadlock を失敗に
/// する。完了後に (a) pending/in_flight が 0、(b) スロット数 + 2 本の逐次要求が
/// 全て 200、を確認する = スロット漏れ無し。
#[tokio::test]
async fn concurrent_status_checks_under_load_do_not_deadlock_or_leak_slots() {
    // 12 本を同時に積むので待ち行列は広めに取る (キュー溢れの 503 を
    // 観測したいわけではない)。
    let state = state_with_queue(
        Scheduler::spawn,
        Box::new(denno_engine::testutil::DummyModel),
        32,
    );
    let app: Router = denno_server::router(state.clone());
    let slots = state.engine.lock().unwrap().caches.len();
    assert!(slots > 0, "KV スロット数が 0");

    let mut gens = Vec::new();
    for _ in 0..12 {
        let a = app.clone();
        gens.push(tokio::spawn(async move {
            a.oneshot(post_request("/v1/messages", messages_body()))
                .await
                .unwrap()
        }));
    }

    let mut probers = Vec::new();
    for _ in 0..4 {
        let a = app.clone();
        probers.push(tokio::spawn(async move {
            for _ in 0..40 {
                let (hs, hb) = get(&a, "/health").await;
                assert_eq!(hs, StatusCode::OK, "/health が 200 でない: {hb}");
                assert_eq!(hb["status"], "ok");

                let req = Request::builder()
                    .method("GET")
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap();
                let resp = a.clone().oneshot(req).await.unwrap();
                let code = resp.status();
                let bytes = resp.into_body().collect().await.unwrap().to_bytes();
                let body: Value = serde_json::from_slice(&bytes).unwrap();
                assert_ready_shape(&body);
                match code {
                    StatusCode::OK => assert_eq!(body["status"], "ready"),
                    StatusCode::SERVICE_UNAVAILABLE => {
                        assert!(
                            body["status"] == "loading" || body["status"] == "draining",
                            "503 の理由が不明: {body}"
                        );
                        assert_eq!(body["ready"], false);
                    }
                    other => panic!("想定外の /ready ステータス {other}: {body}"),
                }
                assert!(
                    body["in_flight"].as_u64().unwrap() <= 4,
                    "同時実行が上限 4 を超えた: {body}"
                );
            }
        }));
    }

    // 生成 12 本 + プローブ 160 回。deadlock ならここで失敗する。
    tokio::time::timeout(WAIT, async {
        for (i, t) in gens.into_iter().enumerate() {
            let resp = t.await.expect("生成タスクが panic した");
            let status = resp.status();
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(status, StatusCode::OK, "生成 {i} が失敗: {body}");
            assert_eq!(body["type"], "message", "生成 {i} の本文: {body}");
        }
        for p in probers {
            p.await.expect("プローブタスクが panic した");
        }
    })
    .await
    .expect("負荷中に status 連打が完了しない (deadlock の疑い)");

    // 会計が戻る。
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
    assert!(state.scheduler.lifecycle().worker_alive());

    // スロット漏れの挙動確認: スロット数 + 2 本を逐次流し、全て 200 であること。
    for i in 0..slots + 2 {
        let (status, body) = post(&app, "/v1/messages", messages_body()).await;
        assert_eq!(
            status,
            StatusCode::OK,
            "負荷後の逐次要求 {i} が失敗 (スロット漏れの疑い): {body}"
        );
    }
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
}

/// drain の待ちに上限があること (「進まない drain を無限に待たない」) の保証。
/// ゲートを閉じたまま短い上限を渡すと false で戻る。
#[tokio::test]
async fn drain_wait_is_bounded_when_generation_never_finishes() {
    let (model, gate) = GatedModel::new();
    let state = state_with(Scheduler::spawn, Box::new(model));
    let app: Router = denno_server::router(state.clone());

    let a = app.clone();
    let held = tokio::spawn(async move {
        a.oneshot(post_request("/v1/messages", messages_body()))
            .await
            .unwrap()
    });
    wait_until("受付済みの要求が 1 本になる", || {
        state.scheduler.in_flight() == 1
    })
    .await;
    state.scheduler.lifecycle().begin_drain();

    let start = Instant::now();
    let idle = wait_idle_bounded(&state, Duration::from_millis(60), Duration::from_millis(5)).await;
    let elapsed = start.elapsed();
    assert!(!idle, "進行中があるのに idle と判定された");
    assert!(elapsed >= Duration::from_millis(60), "上限より早く戻った: {elapsed:?}");
    assert!(elapsed < WAIT, "上限を大きく超えて待った: {elapsed:?}");

    // 後始末: 解放してタスクを回収 (テスト終了時のハングを避ける)。
    release(&gate);
    let resp = tokio::time::timeout(WAIT, held)
        .await
        .expect("後始末の要求が完了しない")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}
