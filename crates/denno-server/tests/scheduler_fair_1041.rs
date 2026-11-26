//! Task 1041: スケジューラの受付公平性 (FIFO)・飢餓防止・飽和時の
//! プロトコル別エラー伝播の監査テスト。
//!
//! 監査した不変条件 (実装は `src/scheduler.rs` の worker / admit_job):
//!
//! 1. **FIFO 公平性**: 受付キューは `mpsc::sync_channel` (FIFO)。worker は
//!    「空きスロットがある限り先頭から pull し、pull 順に prefill する」ため、
//!    受付順 = prefill 開始順 になる。プロトコル (Anthropic SSE / OpenAI SSE /
//!    Ollama 非ストリーム = Job::Stream / Job::OpenaiStream / Job::Chat) は
//!    受付経路が違うだけで同じ 1 本のキューに載る。
//! 2. **飢餓防止**: 同時実行は `MAX_ACTIVE_SESSIONS` (KV スロット数) で頭打ちに
//!    なるが、ジョブは完了までに必ず順番が回る。待ちには `Limits::admit` の
//!    上限があり、期限を過ぎたジョブは worker を塞がずにエラーで返る。
//! 3. **飽和時のエラー伝播**: キュー満杯はプロトコルごとの形
//!    (Anthropic = 503 overloaded_error / OpenAI = 429 rate_limit_exceeded /
//!    Ollama = 503) + `Retry-After` で返り、受理済みジョブは取りこぼさない。
//!
//! すべて CPU のみ (ダミーモデル / ダミートークナイザ)。実モデルは使わない。

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_CONTEXT, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_server::{
    AppState, EngineInfo, Limits, Scheduler, MAX_ACTIVE_SESSIONS, QUEUE_RETRY_AFTER_SECS,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// モデル: 決定論的 logits + prefill (入力長 > 1 の forward) の記録
// ---------------------------------------------------------------------------

/// prefill のたびに入力トークン列を `prefills` へ追記する。
/// decode の forward は入力長 1 なので記録されない — つまり `prefills` の並びが
/// そのまま「prefill 開始順」になる。
struct AuditModel {
    delay: Duration,
    eos: bool,
    prefills: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl Model for AuditModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        if self.eos {
            &[DUMMY_EOS]
        } else {
            &[]
        }
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        if tokens.len() > 1 {
            self.prefills.lock().unwrap().push(tokens.to_vec());
        }
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        // P17e: 需要ベース確保 (書き込み前に容量確保) のダミー側契約
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

// ---------------------------------------------------------------------------
// ヘルパ
// ---------------------------------------------------------------------------

fn recording_model(delay: Duration) -> (Box<dyn Model>, Arc<Mutex<Vec<Vec<u32>>>>) {
    let prefills = Arc::new(Mutex::new(Vec::new()));
    let model = AuditModel { delay, eos: false, prefills: prefills.clone() };
    (Box::new(model), prefills)
}

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
    GenParams { max_tokens, temperature: Some(0.0), ..Default::default() }
}

fn user_msg(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: text.into() }]
}

// プロトコル別ボディ (content は prefill 記録から job を識別するための印)
fn anthropic_body(content: &str, max_tokens: usize, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": content}]
    })
}

fn openai_body(content: &str, max_tokens: usize, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": content}]
    })
}

fn ollama_body(content: &str, max_tokens: usize) -> Value {
    json!({
        "model": "denno-test",
        "stream": false,
        "messages": [{"role": "user", "content": content}],
        "options": {"num_predict": max_tokens, "temperature": 0.0}
    })
}

/// 1 リクエスト送って (status, headers, body 文字列) を返す。SSE も最後まで読む。
async fn send(app: &Router, uri: &str, body: Value) -> (StatusCode, header::HeaderMap, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(30), app.clone().oneshot(req))
        .await
        .expect("30s 以内に応答ヘッダが返らない (サーバ無応答)")
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = tokio::time::timeout(Duration::from_secs(30), resp.into_body().collect())
        .await
        .expect("30s 以内にボディが終わらない")
        .expect("ボディ読み取り失敗")
        .to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn retry_after(headers: &header::HeaderMap) -> Option<String> {
    headers.get(header::RETRY_AFTER).and_then(|v| v.to_str().ok()).map(str::to_string)
}

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

fn wait_idle(sched: &Scheduler, secs: u64) -> bool {
    wait_until(|| sched.in_flight() == 0 && sched.pending() == 0, secs)
}

/// prefill 記録から content トークン (t13 / t14 / t15 = id 13/14/15) を抜き出し、
/// 出現順に並べる。
fn prefill_markers(prefills: &Arc<Mutex<Vec<Vec<u32>>>>) -> Vec<u32> {
    let seen = prefills.lock().unwrap();
    seen.iter()
        .filter_map(|v| v.iter().copied().find(|t| (13..=15).contains(t)))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. FIFO 公平性: 受付順 = prefill 開始順 (プロトコル混在)
// ---------------------------------------------------------------------------

/// 3 プロトコルを 1 本のキューへ「Anthropic SSE → OpenAI SSE → Ollama 非ストリーム」
/// の順に投入し、prefill が同じ順で始まること (= FIFO) を検証する。
///
/// 決定論のため、エンジンロックを握って worker を 1 件目の admit で止め、
/// 3 件の受付順を確定させてからロックを解放する。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fair_admission_prefill_order_is_fifo_across_protocols() {
    let (model, prefills) = recording_model(Duration::ZERO);
    // 3 ジョブすべてが同時に active へ入る (MAX_ACTIVE_SESSIONS = 4)
    let state = state_with(model, 8, Limits::default());
    let app = denno_server::router(state.clone());

    let guard = state.engine.lock().unwrap();

    // 1 件目: Anthropic SSE (Job::Stream)
    let h1 = tokio::spawn({
        let app = app.clone();
        async move { send(&app, "/v1/messages", anthropic_body("t13", 2, true)).await }
    });
    assert!(
        wait_until(|| state.scheduler.in_flight() == 1, 5),
        "1 件目が pull されない (in_flight={})",
        state.scheduler.in_flight()
    );

    // 2 件目: OpenAI SSE (Job::OpenaiStream)
    let h2 = tokio::spawn({
        let app = app.clone();
        async move { send(&app, "/v1/chat/completions", openai_body("t14", 2, true)).await }
    });
    assert!(
        wait_until(|| state.scheduler.pending() == 1, 5),
        "2 件目が待ち行列に入らない (pending={})",
        state.scheduler.pending()
    );

    // 3 件目: Ollama 非ストリーム (Job::Chat)
    let h3 = tokio::spawn({
        let app = app.clone();
        async move { send(&app, "/api/chat", ollama_body("t15", 2)).await }
    });
    assert!(
        wait_until(|| state.scheduler.pending() == 2, 5),
        "3 件目が待ち行列に入らない (pending={})",
        state.scheduler.pending()
    );

    drop(guard);

    let (s1, _, b1) = h1.await.unwrap();
    let (s2, _, b2) = h2.await.unwrap();
    let (s3, _, b3) = h3.await.unwrap();
    assert_eq!(s1, StatusCode::OK, "Anthropic SSE: {b1}");
    assert_eq!(s2, StatusCode::OK, "OpenAI SSE: {b2}");
    assert_eq!(s3, StatusCode::OK, "Ollama: {b3}");
    assert!(wait_idle(&state.scheduler, 10), "完了後に in_flight/pending が 0 に戻らない");

    let order = prefill_markers(&prefills);
    assert_eq!(
        order,
        vec![13, 14, 15],
        "prefill の開始順が受付順 (Anthropic → OpenAI → Ollama) と違う (FIFO 公平性の破れ)"
    );
    assert_eq!(state.scheduler.stats().completed, 3, "3 件すべて完了していない");
}

// ---------------------------------------------------------------------------
// 2. 飢餓防止: 同時実行上限を超える件数でも全件が順番に完了する
// ---------------------------------------------------------------------------

/// 受け付け可能数 (10 件) が同時実行上限 (MAX_ACTIVE_SESSIONS = 4) を超えても、
/// 全件が拒否されずに完了し、進行中が上限を超えない。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_starvation_all_queued_requests_complete() {
    const N: usize = 10;
    let (model, _prefills) = recording_model(Duration::from_millis(1));
    let sched = Arc::new(Scheduler::spawn(engine_with(model), 16));

    let done = Arc::new(AtomicBool::new(false));
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let monitor = {
        let done = done.clone();
        let max_in_flight = max_in_flight.clone();
        let sched = sched.clone();
        tokio::spawn(async move {
            while !done.load(Ordering::SeqCst) {
                max_in_flight.fetch_max(sched.in_flight(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    let mut rxs = Vec::new();
    for i in 0..N {
        let rx = sched
            .submit_chat(None, user_msg("t13"), params(3), None, None, None)
            .unwrap_or_else(|_| panic!("{i} 件目が QueueFull で拒否された (max_queue=16)"));
        rxs.push(rx);
    }

    for (i, rx) in rxs.into_iter().enumerate() {
        let out = tokio::time::timeout(Duration::from_secs(30), rx)
            .await
            .unwrap_or_else(|_| panic!("{i} 件目が 30s 以内に完了しない (飢餓)"))
            .unwrap_or_else(|_| panic!("{i} 件目の応答チャネルが落ちた"))
            .unwrap_or_else(|e| panic!("{i} 件目がエラー: {e:#}"));
        assert_eq!(out.output_tokens, 3, "{i} 件目の生成数が違う");
    }
    done.store(true, Ordering::SeqCst);
    let _ = monitor.await;

    assert!(wait_idle(&sched, 10), "完了後に in_flight/pending が 0 に戻らない");
    let st = sched.stats();
    assert_eq!(st.completed, N, "完了計数が受付数と一致しない: {st:?}");
    assert_eq!(st.aborted_timeout, 0, "{st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");
    assert_eq!(st.aborted_error, 0, "{st:?}");
    let max = max_in_flight.load(Ordering::SeqCst);
    assert!(max <= MAX_ACTIVE_SESSIONS, "進行中が上限を超えた: max={max}");
    assert!(max >= 2, "並行進行が観測できない (max={max}) — テストの前提が崩れている");
}

// ---------------------------------------------------------------------------
// 3. 飽和時のエラー伝播 (3 プロトコル) と受理済みジョブの取りこぼし無し
// ---------------------------------------------------------------------------

/// ワーカーを停止させ、キュー深度 1 を埋めた状態で各プロトコルのリクエストを
/// 1 本ずつ送る。すべて「そのプロトコルの形」の過負荷応答 + `Retry-After` で
/// 即座に返り、受理済み 2 件はロック解放後に 200 で完走する。
/// その後、3 プロトコルすべてが通常応答に戻る (回復) ことも見る。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn saturation_errors_are_protocol_shaped_and_do_not_drop_accepted_jobs() {
    let (model, _prefills) = recording_model(Duration::ZERO);
    let state = state_with(model, 1, Limits::default());
    let app = denno_server::router(state.clone());

    let guard = state.engine.lock().unwrap();
    // 受理済み 1 件目 (worker が pull してロック待ち) + 2 件目 (キュー)
    let first = tokio::spawn({
        let app = app.clone();
        async move { send(&app, "/v1/messages", anthropic_body("t13", 2, false)).await }
    });
    assert!(
        wait_until(|| state.scheduler.in_flight() == 1 && state.scheduler.pending() == 0, 5),
        "1 件目が pull されない (in_flight={}, pending={})",
        state.scheduler.in_flight(),
        state.scheduler.pending()
    );
    let second = tokio::spawn({
        let app = app.clone();
        async move { send(&app, "/v1/messages", anthropic_body("t14", 2, false)).await }
    });
    assert!(
        wait_until(|| state.scheduler.pending() == 1, 5),
        "2 件目が待ち行列に入らない (pending={})",
        state.scheduler.pending()
    );

    let want_retry_after = QUEUE_RETRY_AFTER_SECS.to_string();

    // Anthropic: 503 overloaded_error
    let (st, hd, body) = send(&app, "/v1/messages", anthropic_body("t13", 2, false)).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "anthropic: {body}");
    assert_eq!(
        retry_after(&hd).as_deref(),
        Some(want_retry_after.as_str()),
        "anthropic に Retry-After が無い: {body}"
    );
    let v: Value = serde_json::from_str(&body).expect("anthropic エラーが JSON でない");
    assert_eq!(v["error"]["type"], "overloaded_error", "anthropic: {body}");
    assert!(
        v["error"]["message"].as_str().unwrap_or("").contains("queue is full"),
        "anthropic 本文が飽和を説明していない: {body}"
    );

    // OpenAI: 429 rate_limit_exceeded
    let (st, hd, body) = send(&app, "/v1/chat/completions", openai_body("t13", 2, false)).await;
    assert_eq!(st, StatusCode::TOO_MANY_REQUESTS, "openai: {body}");
    assert_eq!(
        retry_after(&hd).as_deref(),
        Some(want_retry_after.as_str()),
        "openai に Retry-After が無い: {body}"
    );
    let v: Value = serde_json::from_str(&body).expect("openai エラーが JSON でない");
    assert_eq!(v["error"]["type"], "rate_limit_exceeded", "openai: {body}");
    assert!(
        v["error"]["message"].as_str().unwrap_or("").contains("queue is full"),
        "openai 本文が飽和を説明していない: {body}"
    );

    // Ollama: 503 + Retry-After
    let (st, hd, body) = send(&app, "/api/chat", ollama_body("t13", 2)).await;
    assert_eq!(st, StatusCode::SERVICE_UNAVAILABLE, "ollama: {body}");
    assert_eq!(
        retry_after(&hd).as_deref(),
        Some(want_retry_after.as_str()),
        "ollama に Retry-After が無い: {body}"
    );
    assert!(body.contains("queue is full"), "ollama 本文が飽和を説明していない: {body}");

    // 拒否はキュー容量を消費していない (受理済み 2 件は残っている)
    assert_eq!(state.scheduler.pending(), 1, "拒否が待ち行列を消費した");
    assert_eq!(state.scheduler.in_flight(), 1, "{:?}", state.scheduler.stats());

    drop(guard);

    let (s1, _, b1) = first.await.unwrap();
    let (s2, _, b2) = second.await.unwrap();
    assert_eq!(s1, StatusCode::OK, "受理済み 1 件目が完走しない: {b1}");
    assert_eq!(s2, StatusCode::OK, "受理済み 2 件目が完走しない: {b2}");
    assert!(wait_idle(&state.scheduler, 10), "完了後に in_flight/pending が 0 に戻らない");

    let st_stats = state.scheduler.stats();
    assert_eq!(st_stats.completed, 2, "受理済み 2 件が完了計数に載っていない: {st_stats:?}");
    assert_eq!(st_stats.aborted_timeout, 0, "{st_stats:?}");
    assert_eq!(st_stats.aborted_error, 0, "{st_stats:?}");
    assert_eq!(st_stats.aborted_disconnected, 0, "{st_stats:?}");

    // 回復: 3 プロトコルすべて通常応答に戻る
    let (s, _, b) = send(&app, "/v1/messages", anthropic_body("t13", 2, false)).await;
    assert_eq!(s, StatusCode::OK, "anthropic 回復失敗: {b}");
    let (s, _, b) = send(&app, "/v1/chat/completions", openai_body("t13", 2, false)).await;
    assert_eq!(s, StatusCode::OK, "openai 回復失敗: {b}");
    let (s, _, b) = send(&app, "/api/chat", ollama_body("t13", 2)).await;
    assert_eq!(s, StatusCode::OK, "ollama 回復失敗: {b}");
}

// ---------------------------------------------------------------------------
// 4. 飢餓防止の上限: 期限内に始められないジョブは worker を塞がずエラーで返る
// ---------------------------------------------------------------------------

/// `Limits::admit` を短くし、エンジンロックで worker を止めたまま期限を過ぎさせる。
/// 先頭ジョブはタイムアウトで返り (ハングしない)、直後のリクエストは通常どおり
/// 完走する (KV スロットが漏れない)。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn expired_head_of_line_job_times_out_without_wedging_the_worker() {
    let (model, _prefills) = recording_model(Duration::ZERO);
    let limits = Limits { admit: Duration::from_millis(200), ..Limits::default() };
    let engine = engine_with(model);
    let sched = Scheduler::spawn_with_limits(engine.clone(), 4, limits);

    let guard = engine.lock().unwrap();
    let rx = sched
        .submit_chat(None, user_msg("t13"), params(2), None, None, None)
        .expect("1 件目は受理されるべき");
    assert!(
        wait_until(|| sched.in_flight() == 1, 5),
        "1 件目が pull されない (in_flight={})",
        sched.in_flight()
    );
    // admit 予算 (200ms) を確実に超えさせる
    std::thread::sleep(Duration::from_millis(600));
    drop(guard);

    let err = tokio::time::timeout(Duration::from_secs(10), rx)
        .await
        .expect("期限切れジョブが 10s 以内に返らない (worker が塞がっている)")
        .expect("応答チャネルが落ちた")
        .expect_err("期限切れジョブが成功してしまった");
    let msg = format!("{err:#}");
    assert!(msg.contains("queue/slot wait"), "打ち切り理由が想定外: {msg}");

    assert!(wait_idle(&sched, 5), "打ち切り後に in_flight/pending が 0 に戻らない");
    assert_eq!(sched.stats().aborted_timeout, 1, "{:?}", sched.stats());
    assert_eq!(sched.stats().completed, 0, "{:?}", sched.stats());

    // 後続は通る (スロット / ワーカーが健全)
    let rx2 = sched
        .submit_chat(None, user_msg("t13"), params(2), None, None, None)
        .expect("後続リクエストが QueueFull で拒否された");
    let out = tokio::time::timeout(Duration::from_secs(10), rx2)
        .await
        .expect("後続リクエストが 10s 以内に完了しない")
        .expect("応答チャネルが落ちた")
        .expect("後続リクエストがエラー");
    assert_eq!(out.output_tokens, 2);
    assert!(wait_idle(&sched, 10), "後続完了後に in_flight/pending が 0 に戻らない");
    assert_eq!(sched.stats().completed, 1, "{:?}", sched.stats());
}
