//! Task 1074: ライフサイクル (readiness / drain) の単体寄りのエッジケース。
//!
//! `lifecycle_stress_944.rs` が「進行中あり・負荷あり」の経路を厚く見るのに対し、
//! ここは境界の縮退ケースを固定する:
//!
//! - drain 開始の直前と直後で `/ready` が 200 -> 503 に切り替わり、
//!   `/health` (liveness) は 200 のまま。待ち合わせ無しで即座に観測する。
//! - `begin_drain` の繰り返し呼び出しは冪等 (状態が揺れない)。
//!   進行中 0 の drain はその場で完了し、会計 (`pending`/`in_flight`) は 0 のまま。
//! - drain 中の新規推論要求は Anthropic / OpenAI とも 503 で受付拒否され、
//!   拒否を何度繰り返しても進行中カウンタを汚さない。
//!
//! 契約は `src/lifecycle.rs` と `src/lib.rs` の実装に合わせる:
//! `/ready` の 200 条件は `ready && worker_alive && !draining` の 3 つだけ。
//! 503 に `Retry-After` は付けない (本文の `status` で理由が分かる)。
//! drain 完了条件は run と同じ `pending() == 0 && in_flight() == 0`。

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::dummy_tokenizer;
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// 待つ上限。縮退ケースなので短くてよいが、CI の揺れを吸収する余裕は残す。
const WAIT: Duration = Duration::from_secs(10);
/// アイドル待ちのポーリング間隔 (実装の DRAIN_POLL_MS と同じ意味)。
const POLL: Duration = Duration::from_millis(5);

// ---------------------------------------------------------------------------
// app / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn ready_state() -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(denno_engine::testutil::DummyModel),
        tok,
        "denno-test",
        None,
    );
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
    (denno_server::router(state.clone()), state)
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

fn messages_body() -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

fn chat_body() -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

/// run の drain 待ちと同じ条件 (`pending == 0 && in_flight == 0`) を上限付きで待つ。
/// `lifecycle::wait_for_idle` は crate 内部なので、同じ条件・同じ上限意味論を再現する。
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

// ---------------------------------------------------------------------------
// 1. drain 直前 200 / 直後 503、/health は 200 のまま
// ---------------------------------------------------------------------------

/// 待ち合わせ無しで境界を踏む。`begin_drain` の直後の `/ready` が
/// 最初の応答から 503 になっていること (ポーリングで遅れて切り替わるのでは
/// なく、状態遷移が同期的に効いていること) を固定する。
#[tokio::test]
async fn ready_is_200_before_drain_and_503_immediately_after() {
    let (app, state) = ready_state();

    // drain 前: 200 / ready。
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["status"], "ready");
    assert_eq!(body["ready"], true);
    assert_eq!(body["draining"], false);
    assert_eq!(body["worker_alive"], true);

    // drain 開始。sleep も再試行も挟まず、次の 1 リクエストで 503 になる。
    state.scheduler.lifecycle().begin_drain();

    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["ready"], false);
    assert_eq!(body["draining"], true);
    assert_eq!(body["model_loaded"], true, "モデルは載ったまま: {body}");
    assert_eq!(body["worker_alive"], true, "drain はワーカーを殺さない: {body}");

    // liveness と readiness の分離: drain 中も /health は 200。
    let (hs, hb) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK, "/health = {hb}");
    assert_eq!(hb["status"], "ok");

    // 進行中は無いので会計は 0 のまま。
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
}

// ---------------------------------------------------------------------------
// 2. begin_drain の冪等性 + 進行中 0 の drain は即完了
// ---------------------------------------------------------------------------

/// `begin_drain` を繰り返しても状態が揺れないこと。
/// 進行中 0 なら drain の待ちは即座に true で戻り、会計は 0 のまま。
#[tokio::test]
async fn repeated_drain_calls_are_idempotent_and_idle_drain_completes() {
    let (app, state) = ready_state();
    let lc = state.scheduler.lifecycle();

    // 前提: 受付可能・進行中 0。
    assert!(lc.serves_requests());
    assert!(!lc.is_draining());
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);

    // 同じ停止要求を 100 回。毎回の観測値が同じであること。
    for i in 0..100u32 {
        lc.begin_drain();
        assert!(lc.is_draining(), "反復 {i}: draining が立たない");
        assert!(!lc.serves_requests(), "反復 {i}: drain 中に受付可能と判定された");
        assert!(lc.is_ready(), "反復 {i}: ready フラグを巻き戻した");
        assert!(lc.worker_alive(), "反復 {i}: ワーカー生存を変えた");
        assert_eq!(state.scheduler.pending(), 0, "反復 {i}: pending が動いた");
        assert_eq!(state.scheduler.in_flight(), 0, "反復 {i}: in_flight が動いた");
    }

    // 進行中 0 の drain 完了: 待ちは即 true、会計は 0。
    let start = Instant::now();
    let idle = wait_idle_bounded(&state, WAIT, POLL).await;
    let elapsed = start.elapsed();
    assert!(idle, "進行中 0 なのに idle にならない");
    assert!(elapsed < WAIT, "進行中 0 の drain 待ちが上限に張り付いた: {elapsed:?}");

    // drain 後も /ready は 503 / draining、/health は 200。
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["pending"], 0);
    assert_eq!(body["in_flight"], 0);
    let (hs, _) = get(&app, "/health").await;
    assert_eq!(hs, StatusCode::OK);

    // 一方向: ready を立て直しても受付には戻らない。
    lc.set_ready(true);
    assert!(!lc.serves_requests());
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "drain は取り消せない: {body}");
    assert_eq!(body["status"], "draining");
}

// ---------------------------------------------------------------------------
// 3. drain 中の新規推論要求の拒否と会計の非汚染
// ---------------------------------------------------------------------------

/// 進行中 0 のまま drain し、Anthropic / OpenAI 双方の新規要求を連打する。
/// すべて 503 overloaded_error で、拒否のたびに `pending`/`in_flight` が
/// 0 のままであること (拒否要求をキューやスロットに計上しない)。
#[tokio::test]
async fn new_requests_are_rejected_during_drain_without_touching_counters() {
    let (app, state) = ready_state();
    state.scheduler.lifecycle().begin_drain();

    for i in 0..50u32 {
        let (s1, b1) = post(&app, "/v1/messages", messages_body()).await;
        assert_eq!(s1, StatusCode::SERVICE_UNAVAILABLE, "反復 {i}: body = {b1}");
        assert_eq!(b1["error"]["type"], "overloaded_error", "反復 {i}: body = {b1}");
        assert!(
            b1["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("draining"),
            "反復 {i}: 拒否理由が本文に出ていない: {b1}"
        );

        // OpenAI 互換の外側は overloaded_error を server_error に写像する
        // (openai.rs の error_type 変換)。拒否の事実は本文の message で示す。
        let (s2, b2) = post(&app, "/v1/chat/completions", chat_body()).await;
        assert_eq!(s2, StatusCode::SERVICE_UNAVAILABLE, "反復 {i}: body = {b2}");
        assert_eq!(b2["error"]["type"], "server_error", "反復 {i}: body = {b2}");
        assert!(
            b2["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("draining"),
            "反復 {i}: 拒否理由が本文に出ていない: {b2}"
        );

        // 拒否は会計に触らない。
        assert_eq!(state.scheduler.pending(), 0, "反復 {i}: 拒否がキューに積まれた");
        assert_eq!(state.scheduler.in_flight(), 0, "反復 {i}: 拒否が進行中に計上された");
    }

    // /ready 本文の会計も 0 のまま (ハンドラ越しの観測)。
    let (status, body) = get(&app, "/ready").await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body = {body}");
    assert_eq!(body["status"], "draining");
    assert_eq!(body["pending"], 0, "拒否連打で pending が増えた: {body}");
    assert_eq!(body["in_flight"], 0, "拒否連打で in_flight が増えた: {body}");

    // 拒否後も liveness は 200、readiness は draining のまま。
    assert_eq!(get(&app, "/health").await.0, StatusCode::OK);

    // スロットを消費していないことの傍証: drain を解かない限り受付は 503 のまま。
    let (status, _) = post(&app, "/v1/messages", messages_body()).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
}
