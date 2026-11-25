//! P873: キュー深度の上限とリクエスト期限 (タイムアウト) の統合テスト。
//!
//! カバー項目:
//! 1. `--max-queue-depth` / `--request-timeout-secs` の解決順 (CLI > env > 既定)
//!    とクランプ。
//! 2. キュー飽和時に **プロトコルごとに適切な過負荷応答 + `Retry-After`** を
//!    返すこと (Anthropic 503 overloaded_error / OpenAI 429 rate_limit_exceeded /
//!    Ollama 503)。
//! 3. リクエスト期限 (`Limits::request`) を超えた生成を打ち切り、KV スロットを
//!    解放して `in_flight` が 0 に戻ること (非ストリーム / SSE の両方)。
//! 4. 打ち切り計数 ([`denno_server::RunStatsSnapshot`]) が理由別に増えること。
//!
//! すべて CPU のみ (ダミーモデル / ダミートークナイザ)。実モデルは使わない。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_CONTEXT, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{
    resolve_max_queue_depth, resolve_request_timeout_secs, AppState, EngineInfo, Limits,
    Scheduler, DEFAULT_MAX_QUEUE_DEPTH, MAX_QUEUE_DEPTH_LIMIT, MAX_REQUEST_TIMEOUT_SECS,
    MAX_QUEUE_DEPTH_ENV, QUEUE_RETRY_AFTER_SECS, REQUEST_TIMEOUT_SECS_ENV,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------

/// 1 forward あたり `delay` だけ眠る決定論モデル。`endless` のときは EOS を
/// 出さない (期限まで生成を続けさせる)。
struct TimerModel {
    delay: Duration,
    endless: bool,
}

impl Model for TimerModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        if self.endless {
            &[]
        } else {
            &[denno_engine::testutil::DUMMY_EOS]
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
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

// ---------------------------------------------------------------------------
// App / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn state_with(model: Box<dyn Model>, max_queue: usize, limits: Limits) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
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

fn anthropic_body(max_tokens: usize, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": "t0"}]
    })
}

fn openai_body(max_tokens: usize) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}]
    })
}

fn ollama_body(max_tokens: usize) -> Value {
    json!({
        "model": "denno-test",
        "stream": false,
        "messages": [{"role": "user", "content": "t0"}],
        "options": {"num_predict": max_tokens, "temperature": 0.0}
    })
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, axum::http::HeaderMap, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(30), app.clone().oneshot(req))
        .await
        .expect("30s 以内に応答が返らない (サーバが無応答)")
        .unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, v)
}

/// SSE のボディを最後まで読む (ストリームは期限打ち切りで閉じる)。
async fn post_stream_all(app: &Router, uri: &str, body: Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(30), app.clone().oneshot(req))
        .await
        .expect("30s 以内に SSE が始まらない")
        .unwrap();
    let status = resp.status();
    let bytes = tokio::time::timeout(Duration::from_secs(30), resp.into_body().collect())
        .await
        .expect("30s 以内に SSE が終わらない (期限打ち切りが効いていない)")
        .expect("SSE ボディの読み取りに失敗")
        .to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

async fn wait_in_flight(state: &Arc<AppState>, want: usize) -> bool {
    for _ in 0..2000 {
        if state.scheduler.in_flight() == want && state.scheduler.pending() == 0 {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    false
}

// ---------------------------------------------------------------------------
// 1. 解決順 (CLI > env > 既定) とクランプ
// ---------------------------------------------------------------------------

/// env / CLI の解決は起動時に 1 回だけなので、ここでは関数を直接叩く。
/// env を汚すテストはこの 1 本だけに閉じる (並列実行でも他テストに影響しない
/// よう、最後に必ず unset する)。
#[test]
fn queue_depth_and_timeout_resolution_precedence() {
    // CLI が最優先 (env があっても CLI が勝つ)
    assert_eq!(resolve_max_queue_depth(Some(3), DEFAULT_MAX_QUEUE_DEPTH), 3);
    assert_eq!(resolve_request_timeout_secs(Some(7)), 7);
    // 0 / 過大値はクランプされる
    assert_eq!(resolve_max_queue_depth(Some(0), DEFAULT_MAX_QUEUE_DEPTH), 1);
    assert_eq!(
        resolve_max_queue_depth(Some(usize::MAX), DEFAULT_MAX_QUEUE_DEPTH),
        MAX_QUEUE_DEPTH_LIMIT
    );
    assert_eq!(
        resolve_request_timeout_secs(Some(u64::MAX)),
        MAX_REQUEST_TIMEOUT_SECS
    );
    // 既定: 待ち行列は呼び出し側の fallback、期限は 0 (無効)
    assert_eq!(resolve_max_queue_depth(None, 7), 7);
    assert_eq!(resolve_request_timeout_secs(None), 0);

    // env は CLI 未指定のときだけ効く
    std::env::set_var(MAX_QUEUE_DEPTH_ENV, "5");
    std::env::set_var(REQUEST_TIMEOUT_SECS_ENV, "12");
    assert_eq!(resolve_max_queue_depth(None, DEFAULT_MAX_QUEUE_DEPTH), 5);
    assert_eq!(resolve_max_queue_depth(Some(2), DEFAULT_MAX_QUEUE_DEPTH), 2);
    assert_eq!(resolve_request_timeout_secs(None), 12);
    assert_eq!(resolve_request_timeout_secs(Some(1)), 1);
    // 解析できない env は無視して既定へ落ちる
    std::env::set_var(MAX_QUEUE_DEPTH_ENV, "not-a-number");
    std::env::set_var(REQUEST_TIMEOUT_SECS_ENV, "");
    assert_eq!(resolve_max_queue_depth(None, DEFAULT_MAX_QUEUE_DEPTH), DEFAULT_MAX_QUEUE_DEPTH);
    assert_eq!(resolve_request_timeout_secs(None), 0);
    std::env::remove_var(MAX_QUEUE_DEPTH_ENV);
    std::env::remove_var(REQUEST_TIMEOUT_SECS_ENV);
}

// ---------------------------------------------------------------------------
// 2. キュー飽和 → プロトコルごとの過負荷応答 + Retry-After
// ---------------------------------------------------------------------------

/// `max_queue = 1` で 3 プロトコルに同時に投げ、あふれた要求が
/// Anthropic 503 / OpenAI 429 / Ollama 503 で、いずれも `Retry-After` を
/// 持つことを確かめる。最後に in_flight / pending が 0 に戻る。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_saturation_returns_retry_after_on_all_protocols() {
    let state = state_with(
        Box::new(TimerModel { delay: Duration::from_millis(25), endless: false }),
        1,
        Limits::default(),
    );
    let app = denno_server::router(state.clone());

    let mut tasks = Vec::new();
    for i in 0..14usize {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            match i % 3 {
                0 => {
                    let (s, h, v) = post(&app, "/v1/messages", anthropic_body(4, false)).await;
                    (0u8, s, h, v)
                }
                1 => {
                    let (s, h, v) = post(&app, "/v1/chat/completions", openai_body(4)).await;
                    (1u8, s, h, v)
                }
                _ => {
                    let (s, h, v) = post(&app, "/api/chat", ollama_body(4)).await;
                    (2u8, s, h, v)
                }
            }
        }));
    }

    let mut ok = 0usize;
    let mut saturated = 0usize;
    for t in tasks {
        let (proto, status, headers, v) = t.await.unwrap();
        if status == StatusCode::OK {
            ok += 1;
            continue;
        }
        let retry = headers.get(axum::http::header::RETRY_AFTER);
        let retry = retry
            .unwrap_or_else(|| panic!("過負荷応答に Retry-After が無い: proto={proto} body={v}"))
            .to_str()
            .unwrap()
            .parse::<u64>()
            .expect("Retry-After が秒数でない");
        assert_eq!(retry, QUEUE_RETRY_AFTER_SECS, "proto={proto}");
        let msg = v["error"]["message"]
            .as_str()
            .or_else(|| v["error"].as_str())
            .unwrap_or("");
        assert!(
            msg.contains("queue is full"),
            "再試行可能な理由が本文に無い: proto={proto} body={v}"
        );
        match proto {
            // Anthropic: 503 overloaded_error
            0 => {
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body={v}");
                assert_eq!(v["error"]["type"], "overloaded_error", "body={v}");
            }
            // OpenAI: 429 rate_limit_exceeded
            1 => {
                assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "body={v}");
                assert_eq!(v["error"]["type"], "rate_limit_exceeded", "body={v}");
            }
            // Ollama: 503 + {"error": "..."}
            _ => {
                assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body={v}");
                assert!(v["error"].is_string(), "Ollama の本文が文字列でない: {v}");
            }
        }
        saturated += 1;
    }
    assert!(
        saturated >= 1,
        "過負荷応答が 1 件も無い (キュー上限が効いていない: ok={ok})"
    );
    assert!(ok >= 2, "受理数が少なすぎる: ok={ok}");

    assert!(
        wait_in_flight(&state, 0).await,
        "飽和後に in_flight が 0 に戻らない (in_flight={}, pending={})",
        state.scheduler.in_flight(),
        state.scheduler.pending()
    );
    let st = state.scheduler.stats();
    assert_eq!(st.completed, ok, "完了計数が受理数と一致しない: {st:?}");

    // 飽和の直後でもサーバーは通常応答に戻る
    let (s, _, v) = post(&app, "/v1/chat/completions", openai_body(2)).await;
    assert_eq!(s, StatusCode::OK, "body = {v}");
}

// ---------------------------------------------------------------------------
// 3. リクエスト期限 → 打ち切り + スロット解放 (非ストリーム)
// ---------------------------------------------------------------------------

/// 期限 100ms に対し 1 forward 5ms で 200 トークン生成させる。途中で期限を
/// 超えるので 500 (api_error) で返り、in_flight が 0 に戻る。その後、期限内に
/// 収まる短い要求は通る (スロットが解放されている証拠)。
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_timeout_aborts_generation_and_releases_slot() {
    let limits = Limits {
        request: Duration::from_millis(100),
        ..Limits::default()
    };
    let state = state_with(
        Box::new(TimerModel { delay: Duration::from_millis(5), endless: true }),
        8,
        limits,
    );
    let app = denno_server::router(state.clone());

    let (s, _, v) = post(&app, "/v1/messages", anthropic_body(200, false)).await;
    assert_eq!(
        s,
        StatusCode::INTERNAL_SERVER_ERROR,
        "期限超過が 500 で返らない: {v}"
    );
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("timeout") && msg.contains("request wall clock"),
        "本文がリクエスト期限の超過を説明していない: {v}"
    );
    assert!(
        wait_in_flight(&state, 0).await,
        "打ち切り後に in_flight が 0 に戻らない (in_flight={})",
        state.scheduler.in_flight()
    );
    let st = state.scheduler.stats();
    assert_eq!(st.aborted_timeout, 1, "期限打ち切りの計数が増えていない: {st:?}");
    assert_eq!(st.aborted_disconnected, 0, "{st:?}");

    // 同じスケジューラで、期限内に収まる要求は通る (スロット解放の確認)
    let (s, _, v) = post(&app, "/v1/messages", anthropic_body(2, false)).await;
    assert_eq!(s, StatusCode::OK, "打ち切りのあとスロットが解放されていない: {v}");
    assert!(wait_in_flight(&state, 0).await);
    assert_eq!(state.scheduler.stats().completed, 1);
}

// ---------------------------------------------------------------------------
// 4. リクエスト期限 → SSE ではエラーイベントで通知される
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn request_timeout_reports_error_event_on_sse() {
    let limits = Limits {
        request: Duration::from_millis(120),
        ..Limits::default()
    };
    let state = state_with(
        Box::new(TimerModel { delay: Duration::from_millis(5), endless: true }),
        8,
        limits,
    );
    let app = denno_server::router(state.clone());

    let (s, body) = post_stream_all(&app, "/v1/messages", anthropic_body(400, true)).await;
    assert_eq!(s, StatusCode::OK, "SSE は 200 で始まる: {body}");
    assert!(
        body.contains("\"type\":\"error\"") && body.contains("request wall clock"),
        "SSE に期限超過の error イベントが無い: {body}"
    );
    assert!(
        wait_in_flight(&state, 0).await,
        "SSE 打ち切り後に in_flight が 0 に戻らない (in_flight={})",
        state.scheduler.in_flight()
    );
    assert_eq!(state.scheduler.stats().aborted_timeout, 1);
}

// ---------------------------------------------------------------------------
// 5. 期限を無効 (0) にしたときは従来どおり完走する
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn timeout_disabled_by_default_lets_generation_finish() {
    let counter = Arc::new(AtomicUsize::new(0));
    let limits = Limits::default();
    assert!(limits.request.is_zero(), "既定のリクエスト期限は 0 (無効) であるべき");
    let state = state_with(
        Box::new(CountingModel(counter.clone())),
        8,
        limits,
    );
    let app = denno_server::router(state.clone());
    let (s, _, v) = post(&app, "/v1/messages", anthropic_body(16, false)).await;
    assert_eq!(s, StatusCode::OK, "body = {v}");
    assert_eq!(state.scheduler.stats().aborted_timeout, 0);
    assert!(counter.load(Ordering::SeqCst) > 0, "forward が 1 回も呼ばれていない");
}

/// 呼び出し回数を数えるだけの小さなモデル (期限無効の経路の確認用)。
struct CountingModel(Arc<AtomicUsize>);

impl Model for CountingModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[denno_engine::testutil::DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}
