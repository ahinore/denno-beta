//! Task 1009: スケジューラのキュー深度・期限・キャンセルの境界テスト。
//!
//! カバー項目:
//! 1. キュー深度の厳密な上限 (N 件受理・N+1 件目は `QueueFull` で拒否) を
//!    エンジンロックでワーカーを止めて決定論的に観測する。
//! 2. 同じ上限が HTTP 層では 503 / `overloaded_error` / `Retry-After` として
//!    現れること (受理分はロック解放後に完走する)。
//! 3. リクエスト期限の満了 (生成中): 500 で打ち切り、KV スロットを漏らさず、
//!    直後の通常リクエストが通ること。
//! 4. キュー待ちのまま期限が満了したジョブ: `admit_budget` 経路で打ち切られ、
//!    スロットを取得しないこと (待ち行列に滞留したまま実行開始しない)。
//! 5. 同時バースト + クライアント切断: `aborted_disconnected` と `completed` が
//!    件数どおりに増え、`in_flight` / `pending` が 0 に戻り、直後に通常リクエストが
//!    通ること。
//!
//! すべて CPU のみ (ダミーモデル / ダミートークナイザ)。実モデルは使わない。

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_server::{
    AppState, EngineInfo, Limits, QueueFull, Scheduler, MAX_ACTIVE_SESSIONS, QUEUE_RETRY_AFTER_SECS,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// モデル (決定論的 logits。`eos` を切ると max_tokens まで生成が続く)
// ---------------------------------------------------------------------------

struct ProbeModel {
    delay: Duration,
    eos: bool,
}

impl Model for ProbeModel {
    fn context(&self) -> usize {
        denno_engine::testutil::DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        if self.eos {
            &[DUMMY_EOS]
        } else {
            &[]
        }
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
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
        KvCache::new(1, 4, denno_engine::testutil::DUMMY_CONTEXT)
    }
}

// ---------------------------------------------------------------------------
// ヘルパ
// ---------------------------------------------------------------------------

fn engine_with(model: Box<dyn Model>) -> Arc<Mutex<Engine>> {
    let tok = dummy_tokenizer().unwrap();
    Arc::new(Mutex::new(Engine::from_parts(model, tok, "denno-test", None)))
}

fn state_with(model: Box<dyn Model>, max_queue: usize, limits: Limits) -> Arc<AppState> {
    let engine = engine_with(model);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
    let scheduler = Scheduler::spawn_with_limits(engine.clone(), max_queue, limits);
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

fn params(max_tokens: usize) -> GenParams {
    GenParams {
        max_tokens,
        temperature: Some(0.0),
        ..Default::default()
    }
}

fn user_msg() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".into(),
        content: "t0".into(),
    }]
}

fn body(max_tokens: usize, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": "t0"}]
    })
}

async fn post(app: &Router, b: Value) -> (StatusCode, Value) {
    let (status, _headers, v) = post_with_headers(app, b).await;
    (status, v)
}

async fn post_with_headers(
    app: &Router,
    b: Value,
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(b.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(30), app.clone().oneshot(req))
        .await
        .expect("30s 以内に応答が返らない")
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = tokio::time::timeout(Duration::from_secs(30), resp.into_body().collect())
        .await
        .expect("30s 以内にボディが終わらない")
        .unwrap()
        .to_bytes();
    (
        status,
        headers,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

/// `cond` が真になるまで最大 `secs` 待つ。真になれば true。
fn wait_until(mut cond: impl FnMut() -> bool, secs: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    cond()
}

/// スケジューラが空 (実行中 0・待ち 0) になるまで待つ。
fn wait_idle(sched: &Scheduler, secs: u64) -> bool {
    wait_until(|| sched.in_flight() == 0 && sched.pending() == 0, secs)
}

// ---------------------------------------------------------------------------
// 1. キュー深度の厳密な上限 (スケジューラ直)
// ---------------------------------------------------------------------------

/// `max_queue = N` のとき、ワーカーが 1 件を pull して停止している間に
/// キューへはちょうど N 件入り、N+1 件目は `QueueFull` で拒否される。
/// 拒否は `pending()` に計上されず、ロック解放後は受理分がすべて完走する。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_depth_accepts_exactly_n_and_rejects_n_plus_one() {
    const N: usize = 3;
    let engine = engine_with(Box::new(ProbeModel {
        delay: Duration::ZERO,
        eos: true,
    }));
    let sched = Scheduler::spawn(engine.clone(), N);
    assert_eq!(sched.max_queue(), N);

    let mut rxs = Vec::new();
    {
        // エンジンロックを握ってワーカーを build_run のロック待ちで止める。
        // この間 `in_flight` は 1、`pending` は 0 のまま安定する。
        let guard = engine.lock().unwrap();
        rxs.push(
            sched
                .submit_chat(None, user_msg(), params(4), None, None, None)
                .expect("1 件目は受理されるべき"),
        );
        assert!(
            wait_until(|| sched.in_flight() == 1 && sched.pending() == 0, 5),
            "ワーカーが 1 件目を pull して停止しない (in_flight={}, pending={})",
            sched.in_flight(),
            sched.pending()
        );

        // キュー深度ぶんだけ受理される。
        for i in 0..N {
            rxs.push(
                sched
                    .submit_chat(None, user_msg(), params(4), None, None, None)
                    .unwrap_or_else(|_| panic!("キュー内 {i} 件目が拒否された")),
            );
        }
        assert_eq!(sched.pending(), N, "pending がキュー深度と一致しない");

        // N+1 件目は即座に拒否される。
        let err = sched
            .submit_chat(None, user_msg(), params(4), None, None, None)
            .expect_err("N+1 件目は QueueFull であるべき");
        assert!(matches!(err, QueueFull), "拒否が QueueFull でない");
        assert_eq!(sched.pending(), N, "拒否されたジョブが pending に残っている");

        drop(guard);
    }

    // 受理された N+1 件はすべて完走する。
    for (i, rx) in rxs.into_iter().enumerate() {
        let out = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .unwrap_or_else(|_| panic!("受理 {i} 件目が時間内に完了しない"))
            .unwrap_or_else(|_| panic!("受理 {i} 件目の応答チャネルが落ちた"))
            .unwrap_or_else(|e| panic!("受理 {i} 件目がエラー: {e:#}"));
        assert_eq!(out.output_tokens, 4, "受理 {i} 件目の生成数が違う");
    }

    assert!(wait_idle(&sched, 5), "完了後に in_flight/pending が 0 に戻らない");
    let st = sched.stats();
    assert_eq!(st.completed, N + 1, "完了計数が受理数と一致しない: {st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");
    assert_eq!(st.aborted_error, 0, "{st:?}");
}

// ---------------------------------------------------------------------------
// 2. 同じ上限の HTTP 表現 (503 / overloaded_error / Retry-After)
// ---------------------------------------------------------------------------

/// ワーカーをロックで止めた状態で、キュー深度 2 に 3 件を同時投入する。
/// ちょうど 1 件が 503 `overloaded_error` + `Retry-After` で拒否され、
/// 残り (pull 済み 1 件 + キュー 2 件) はロック解放後に 200 で完走する。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_full_over_http_returns_503_overloaded_with_retry_after() {
    let state = state_with(
        Box::new(ProbeModel {
            delay: Duration::ZERO,
            eos: true,
        }),
        2,
        Limits::default(),
    );
    let app = denno_server::router(state.clone());

    let guard = state.engine.lock().unwrap();
    let first = {
        let app = app.clone();
        tokio::spawn(async move { post_with_headers(&app, body(4, false)).await })
    };
    assert!(
        wait_until(|| state.scheduler.in_flight() == 1 && state.scheduler.pending() == 0, 5),
        "1 件目が pull されない (in_flight={}, pending={})",
        state.scheduler.in_flight(),
        state.scheduler.pending()
    );

    let mut rest = Vec::new();
    for _ in 0..3 {
        let app = app.clone();
        rest.push(tokio::spawn(async move { post_with_headers(&app, body(4, false)).await }));
    }
    // ハンドラの submit が終わるまで待つ (この間ワーカーはロック待ちで動かない)。
    std::thread::sleep(Duration::from_millis(300));
    drop(guard);

    let mut ok = 0usize;
    let mut rejected = 0usize;
    let results = {
        let mut v = vec![first.await.unwrap()];
        for h in rest {
            v.push(h.await.unwrap());
        }
        v
    };
    for (i, (status, headers, v)) in results.into_iter().enumerate() {
        match status {
            StatusCode::OK => {
                ok += 1;
                assert_eq!(v["type"], "message", "{i} 件目: {v}");
                assert_eq!(v["usage"]["output_tokens"], 4, "{i} 件目: {v}");
            }
            StatusCode::SERVICE_UNAVAILABLE => {
                rejected += 1;
                assert_eq!(v["error"]["type"], "overloaded_error", "{i} 件目: {v}");
                assert!(
                    v["error"]["message"]
                        .as_str()
                        .unwrap_or("")
                        .contains("queue is full"),
                    "{i} 件目: {v}"
                );
                let retry = headers
                    .get(axum::http::header::RETRY_AFTER)
                    .unwrap_or_else(|| panic!("{i} 件目に Retry-After が無い: {v}"))
                    .to_str()
                    .unwrap();
                assert_eq!(retry, QUEUE_RETRY_AFTER_SECS.to_string(), "{i} 件目");
            }
            other => panic!("{i} 件目の想定外ステータス {other}: {v}"),
        }
    }
    assert_eq!(rejected, 1, "キュー深度 2 に対し拒否が 1 件でない: ok={ok}");
    assert_eq!(ok, 3, "受理数が pull 済み 1 + キュー 2 でない: ok={ok}");

    assert!(
        wait_idle(&state.scheduler, 5),
        "in_flight/pending が 0 に戻らない (in_flight={}, pending={})",
        state.scheduler.in_flight(),
        state.scheduler.pending()
    );
    assert_eq!(state.scheduler.stats().completed, 3);
    assert_eq!(state.engine.lock().unwrap().slot_busy_count(), 0);
}

// ---------------------------------------------------------------------------
// 3. 生成中のリクエスト期限 (wall clock) → 500 + スロット解放
// ---------------------------------------------------------------------------

/// 期限 80ms に対し 1 forward 5ms で 400 トークン生成させる。期限で打ち切られ
/// 500 (api_error) が返り、KV スロットが漏れない (busy 0)。直後の短い要求は
/// 期限内に完走する。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_deadline_mid_generation_aborts_and_releases_slots() {
    let limits = Limits {
        request: Duration::from_millis(80),
        ..Limits::default()
    };
    let state = state_with(
        Box::new(ProbeModel {
            delay: Duration::from_millis(5),
            eos: false,
        }),
        8,
        limits,
    );
    let app = denno_server::router(state.clone());

    let (status, v) = post(&app, body(400, false)).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "body={v}");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("timeout") && msg.contains("request wall clock"),
        "期限超過を説明していない: {v}"
    );

    assert!(wait_idle(&state.scheduler, 5), "打ち切り後に idle に戻らない");
    let st = state.scheduler.stats();
    assert_eq!(st.aborted_timeout, 1, "{st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");
    assert_eq!(
        state.engine.lock().unwrap().slot_busy_count(),
        0,
        "期限打ち切りで KV スロットが漏れている"
    );

    // 同じスケジューラで短い要求は通る (スロット解放の実証)。
    let (status, v) = post(&app, body(3, false)).await;
    assert_eq!(status, StatusCode::OK, "打ち切り後に通らない: {v}");
    assert_eq!(v["usage"]["output_tokens"], 3);
    assert!(wait_idle(&state.scheduler, 5));
    assert_eq!(state.scheduler.stats().completed, 1);
}

// ---------------------------------------------------------------------------
// 4. キュー待ちのまま期限が満了したジョブ (admit_budget 経路)
// ---------------------------------------------------------------------------

/// ワーカーを止めた状態で 1 件を pull させ、さらに 1 件をキューに積む。
/// 2 期限ぶん待ってから解放すると、どちらも `admit` 予算超過で打ち切られ、
/// スロットを取得しない (busy 0)。待ち行列に滞留したジョブは実行開始しない。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queued_job_past_deadline_is_rejected_without_taking_a_slot() {
    let limit = Duration::from_millis(80);
    let engine = engine_with(Box::new(ProbeModel {
        delay: Duration::from_millis(3),
        eos: false,
    }));
    let sched = Scheduler::spawn_with_limits(
        engine.clone(),
        4,
        Limits {
            request: limit,
            ..Limits::default()
        },
    );

    let mut rxs = Vec::new();
    {
        let guard = engine.lock().unwrap();
        rxs.push(
            sched
                .submit_chat(None, user_msg(), params(400), None, None, None)
                .expect("pull される 1 件目は受理されるべき"),
        );
        assert!(
            wait_until(|| sched.in_flight() == 1 && sched.pending() == 0, 5),
            "1 件目が pull されない"
        );
        rxs.push(
            sched
                .submit_chat(None, user_msg(), params(400), None, None, None)
                .expect("キューに積む 2 件目は受理されるべき"),
        );
        assert_eq!(sched.pending(), 1, "2 件目が待ち行列に入っていない");

        // 両方の期限が確実に過ぎるまで待つ (期限の 2 倍)。
        std::thread::sleep(limit * 2);
        drop(guard);
    }

    for (i, rx) in rxs.into_iter().enumerate() {
        let err = tokio::time::timeout(Duration::from_secs(10), rx)
            .await
            .unwrap_or_else(|_| panic!("{i} 件目が時間内に打ち切られない"))
            .unwrap_or_else(|_| panic!("{i} 件目の応答チャネルが落ちた"))
            .expect_err("期限超過なのに成功が返った");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("timeout"),
            "{i} 件目のエラーが timeout でない: {msg}"
        );
    }

    assert!(wait_idle(&sched, 5), "打ち切り後に idle に戻らない");
    let st = sched.stats();
    assert_eq!(st.aborted_timeout, 2, "期限打ち切りの計数が 2 でない: {st:?}");
    assert_eq!(st.completed, 0, "{st:?}");
    assert_eq!(
        engine.lock().unwrap().slot_busy_count(),
        0,
        "キュー待ち期限切れで KV スロットが漏れている"
    );

    // 待ち行列が空になった後、通常ジョブは完走する。
    let rx = sched
        .submit_chat(None, user_msg(), params(2), None, None, None)
        .expect("回復後のジョブが受理されない");
    let out = tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .expect("回復後のジョブが完了しない")
        .unwrap()
        .expect("回復後のジョブがエラー");
    assert_eq!(out.output_tokens, 2);
    assert!(wait_idle(&sched, 5));
}

// ---------------------------------------------------------------------------
// 5. 同時バースト + クライアント切断 → 計数と回復
// ---------------------------------------------------------------------------

/// SSE ストリーム 8 本を同時に送り、半数は応答ボディを即座に drop (切断)、
/// 半数は最後まで読む。ワーカーは 4 本ずつ捌くため、切断分は pull 時または
/// ステップ境界で打ち切られる。計数 (completed / aborted_disconnected) が
/// 件数どおりで、in_flight / pending が 0 に戻り、直後に通常要求が通ること。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_burst_with_disconnects_counts_and_recovers() {
    let state = state_with(
        Box::new(ProbeModel {
            delay: Duration::from_millis(10),
            eos: false,
        }),
        8,
        Limits::default(),
    );
    let app = denno_server::router(state.clone());

    const TOTAL: usize = 8;
    const SURVIVORS: usize = 4;
    let mut handles = Vec::new();
    for i in 0..TOTAL {
        let app = app.clone();
        let drop_body = i % 2 == 0; // 4 本を切断
        let max_tokens = if drop_body { 400 } else { 8 };
        handles.push(tokio::spawn(async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body(max_tokens, true).to_string()))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            let status = resp.status();
            if drop_body {
                drop(resp); // 応答ボディ drop = クライアント切断相当
                return (status, None);
            }
            let bytes = tokio::time::timeout(Duration::from_secs(20), resp.into_body().collect())
                .await
                .expect("SSE が 20s 以内に終わらない")
                .unwrap()
                .to_bytes();
            (status, Some(String::from_utf8_lossy(&bytes).to_string()))
        }));
    }

    let mut completed = 0usize;
    let mut dropped = 0usize;
    for h in handles {
        let (status, text) = h.await.unwrap();
        assert_eq!(status, StatusCode::OK, "SSE が 200 で始まらない");
        match text {
            Some(t) => {
                assert!(t.contains("event: message_stop\n"), "SSE が完了していない: {t}");
                completed += 1;
            }
            None => dropped += 1,
        }
    }
    assert_eq!(dropped, TOTAL - SURVIVORS, "切断した本数が想定と違う");
    assert_eq!(completed, SURVIVORS, "完走した本数が想定と違う");

    assert!(
        wait_idle(&state.scheduler, 10),
        "バースト後に idle に戻らない (in_flight={}, pending={})",
        state.scheduler.in_flight(),
        state.scheduler.pending()
    );
    let st = state.scheduler.stats();
    assert_eq!(st.completed, SURVIVORS, "完了計数が違う: {st:?}");
    assert_eq!(
        st.aborted_disconnected,
        TOTAL - SURVIVORS,
        "切断打ち切りの計数が違う: {st:?}"
    );
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(st.aborted_error, 0, "{st:?}");
    assert_eq!(MAX_ACTIVE_SESSIONS, 4, "同時実行上限の前提が変わった");
    assert_eq!(
        state.engine.lock().unwrap().slot_busy_count(),
        0,
        "切断バーストで KV スロットが漏れている"
    );

    // 回復: 直後の通常要求が完走する。
    let (status, v) = post(&app, body(3, false)).await;
    assert_eq!(status, StatusCode::OK, "バースト後に通常要求が通らない: {v}");
    assert!(wait_idle(&state.scheduler, 5));
    assert_eq!(state.scheduler.stats().completed, SURVIVORS + 1);
}

/// `QUEUE_RETRY_AFTER_SECS` が正の値であること (HTTP テストの前提)。
#[test]
fn retry_after_constant_is_positive() {
    assert!(QUEUE_RETRY_AFTER_SECS >= 1);
}
