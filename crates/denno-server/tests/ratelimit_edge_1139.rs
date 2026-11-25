//! task 1139: レート制限の境界 (バースト使い切り / Retry-After 書式 /
//! 監視パスの素通し / 複数クライアント同時押し) を突く追加テスト。
//!
//! task 931 (`ratelimit_931.rs`) が「基本挙動」を押さえているので、ここでは
//! 境界値だけを見る。バケット単体のテストは時刻を注入するので決定論的。
//! ルータのテストは `rpm` を小さく (回復量がテスト中ほぼ 0) して、
//! バースト数を超えた回数だけが 429 になる性質に依存する。

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::{
    is_bypass_path, is_inference_path, too_many_requests_response, Decision, RateLimitConfig,
    RateLimiter, TokenBucket, RATE_LIMIT_INFERENCE_PATHS,
};
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------- 共通ヘルパ ----------

fn state() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
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

fn app_with(rpm: u32, burst: Option<u32>) -> Router {
    let limiter = RateLimiter::from_config(RateLimitConfig::new(rpm, burst));
    denno_server::router_with_rate_limit(state(), CorsConfig::default(), limiter)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn hdr(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn generation() -> Request<Body> {
    let body = json!({
        "model": "denno-test",
        "max_tokens": 2,
        "messages": [{"role": "user", "content": "t0"}],
    });
    Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap()
}

// ---------- バースト使い切りの境界 ----------

#[test]
fn burst_is_exhausted_after_exactly_capacity_requests() {
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(4)), t0).expect("有効");
    for i in 0..4 {
        assert!(b.try_acquire_at(t0).is_ok(), "{i} 個目 (容量内) は通る");
        assert!(
            (b.available_tokens(t0) - (3 - i) as f64).abs() < 1e-9,
            "残トークンが 1 個ずつ減る"
        );
    }
    assert!((b.available_tokens(t0) - 0.0).abs() < 1e-9, "使い切りで 0");
    // 5 個目以降は何度でも拒否。拒否がトークンを負にしない。
    for _ in 0..10 {
        assert!(b.try_acquire_at(t0).is_err());
        assert!((b.available_tokens(t0) - 0.0).abs() < 1e-9, "0 未満にならない");
    }
}

#[test]
fn burst_can_exceed_the_per_minute_rate() {
    // 60 rpm (1/秒) でも、バーストが 120 なら瞬間的に 120 個通る。
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(120)), t0).expect("有効");
    let mut allowed = 0u32;
    for _ in 0..121 {
        if b.try_acquire_at(t0).is_ok() {
            allowed += 1;
        }
    }
    assert_eq!(allowed, 120, "バースト上限ちょうどまで通る");
    assert!(b.try_acquire_at(t0).is_err(), "121 個目は拒否");
}

#[test]
fn burst_of_one_denies_immediately_after_first_request() {
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(1)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    assert_eq!(b.try_acquire_at(t0), Err(1));
    // 1 秒経過で 1 トークン回復し、また 1 回だけ通る。
    assert!(b.try_acquire_at(t0 + Duration::from_secs(1)).is_ok());
    assert_eq!(b.try_acquire_at(t0 + Duration::from_secs(1)), Err(1));
}

#[test]
fn partial_refill_below_one_token_still_denies() {
    // 120 rpm = 2/秒、バースト 2 を使い切る。
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(120, Some(2)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    assert!(b.try_acquire_at(t0).is_ok());
    // 0.25 秒で 0.5 トークン。まだ足りない。
    let t1 = t0 + Duration::from_millis(250);
    assert!(b.available_tokens(t1) < 1.0);
    assert_eq!(b.try_acquire_at(t1), Err(1), "端数では通さない");
    // 0.5 秒で 1 トークンちょうど → 1 回通る、次は拒否。
    let t2 = t0 + Duration::from_millis(500);
    assert!(b.try_acquire_at(t2).is_ok());
    assert_eq!(b.try_acquire_at(t2), Err(1));
}

#[test]
fn refill_never_exceeds_capacity_after_idle_period() {
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(6000, Some(3)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    // 1 時間アイドルでも容量 3 で頭打ち。
    let t1 = t0 + Duration::from_secs(3600);
    assert!((b.available_tokens(t1) - 3.0).abs() < 1e-9);
    let mut allowed = 0u32;
    for _ in 0..5 {
        if b.try_acquire_at(t1).is_ok() {
            allowed += 1;
        }
    }
    assert_eq!(allowed, 3, "容量を超えて回復しない");
}

// ---------- Retry-After の値 (切り上げ・整数秒) ----------

#[test]
fn retry_after_is_whole_seconds_rounded_up_for_exact_rates() {
    // per_second が 2 進で正確になる rpm だけを使い、境界値を厳密に比較する。
    // 60 rpm = 1/秒。
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(1)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    assert_eq!(b.try_acquire_at(t0), Err(1), "不足 1 token / 1 = 1 秒");
    // 0.5 秒後は 0.5 token 不足 → 切り上げ 1 秒。
    assert_eq!(b.try_acquire_at(t0 + Duration::from_millis(500)), Err(1));

    // 30 rpm = 0.5/秒 → 不足 1 token は 2 秒。
    let b = TokenBucket::new_at(RateLimitConfig::new(30, Some(1)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    assert_eq!(b.try_acquire_at(t0), Err(2), "不足 1 / 0.5 = 2 秒");
    // 1 秒後は 0.5 token 回復 → 不足 0.5 / 0.5 = 1 秒。
    assert_eq!(b.try_acquire_at(t0 + Duration::from_secs(1)), Err(1));
    // 2 秒後は 1 token 回復して通る。
    assert!(b.try_acquire_at(t0 + Duration::from_secs(2)).is_ok());
}

#[test]
fn retry_after_uses_the_remaining_deficit_not_full_one() {
    // 30 rpm = 0.5/秒、バースト 2。残 0.5 token の不足は「1 秒」であって
    // 「満タンまでの 2 秒」ではない。
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(30, Some(2)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    assert!(b.try_acquire_at(t0).is_ok());
    let t1 = t0 + Duration::from_secs(1);
    assert!((b.available_tokens(t1) - 0.5).abs() < 1e-9, "1 秒で 0.5 token");
    assert_eq!(b.try_acquire_at(t1), Err(1), "残不足 0.5 / 0.5 = 1 秒");
    // さらに 0.5 秒で 0.75 token → まだ足りない、切り上げて 1 秒。
    assert_eq!(b.try_acquire_at(t1 + Duration::from_millis(500)), Err(1));
    // t0 + 2 秒で 1.0 token に達して通る。
    assert!(b.try_acquire_at(t0 + Duration::from_secs(2)).is_ok());
}

#[test]
fn retry_after_response_never_reports_zero() {
    // 0 秒はクライアントを即時リトライのループに落とすので 1 に丸める。
    let resp = too_many_requests_response(0);
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    let v = resp.headers().get("retry-after").expect("Retry-After が無い");
    assert_eq!(v.to_str().unwrap(), "1", "0 は 1 に丸める");
}

#[test]
fn retry_after_header_is_bare_integer_seconds() {
    let resp = too_many_requests_response(7);
    let all: Vec<&str> = resp
        .headers()
        .get_all("retry-after")
        .iter()
        .map(|v| v.to_str().unwrap())
        .collect();
    assert_eq!(all, vec!["7"], "Retry-After は 1 個で単位なしの整数秒");
    assert!(
        all[0].chars().all(|c| c.is_ascii_digit()),
        "Retry-After に 's' や空白を入れない: {:?}",
        all[0]
    );
    assert_eq!(all[0].parse::<u64>().unwrap(), 7);
}

// ---------- 監視パスの素通し ----------

#[test]
fn bypass_paths_do_not_consume_tokens_when_exhausted() {
    let t0 = Instant::now();
    let l = RateLimiter::from_config_at(RateLimitConfig::new(60, Some(1)), t0);
    assert_eq!(l.check_at("/v1/messages", t0), Decision::Allow);
    assert!(matches!(
        l.check_at("/v1/messages", t0),
        Decision::Deny { .. }
    ));
    // 素通しパスを何度叩いても Allow のまま、推論は拒否のまま。
    for _ in 0..100 {
        for p in ["/health", "/ready", "/metrics"] {
            assert_eq!(l.check_at(p, t0), Decision::Allow, "{p} は素通し");
        }
    }
    assert_eq!(
        l.check_at("/v1/messages", t0),
        Decision::Deny { retry_after_secs: 1 },
        "素通しはトークンを補充しない"
    );
}

#[test]
fn bypass_and_inference_paths_are_disjoint_with_trailing_slash() {
    for p in ["/health", "/ready", "/metrics"] {
        assert!(is_bypass_path(p));
        assert!(is_bypass_path(&format!("{p}/")), "末尾スラッシュも素通し");
        assert!(!is_inference_path(p));
    }
    for p in RATE_LIMIT_INFERENCE_PATHS {
        assert!(is_inference_path(p));
        assert!(is_inference_path(&format!("{p}/")));
        assert!(!is_bypass_path(p));
    }
}

#[tokio::test]
async fn exhausted_bucket_still_serves_health_ready_metrics_repeatedly() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    // バケットは空。監視系は何度でも 429 にならない (末尾スラッシュは
    // 別ルートで 404 になり得るが、レート制限は掛からない)。
    for _ in 0..5 {
        for uri in ["/health", "/health/", "/metrics", "/metrics/"] {
            let (s, _, _) = send(&app, get(uri)).await;
            assert_ne!(s, StatusCode::TOO_MANY_REQUESTS, "{uri} は素通し");
        }
        let (s, _, _) = send(&app, get("/health")).await;
        assert_eq!(s, StatusCode::OK, "/health は 200");
    }
    // /ready は状態依存 (200 か 503) だが、429 にはしない。
    let (s, _, _) = send(&app, get("/ready")).await;
    assert_ne!(s, StatusCode::TOO_MANY_REQUESTS, "/ready はレート制限対象外");
    // クエリ文字列付きでもパス判定は /health。
    let (s, _, _) = send(&app, get("/health?probe=1")).await;
    assert_eq!(s, StatusCode::OK, "クエリ付き /health も素通し");
    // バケットは空のまま。
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn bypass_requests_do_not_refill_the_shared_bucket() {
    // 1 rpm = 1/60 per sec。監視トラフィックを大量に流しても推論の回復は早まらない。
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    for _ in 0..200 {
        let (s, _, _) = send(&app, get("/metrics")).await;
        assert_eq!(s, StatusCode::OK);
    }
    // 実経過は数 ms なので、まだ 429。
    let (s, h, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert!(hdr(&h, "retry-after").is_some());
}

// ---------- 429 のレスポンス書式 (ルータ経由) ----------

#[tokio::test]
async fn router_429_retry_after_matches_body_message() {
    // 30 rpm = 0.5/秒、バースト 1 → 2 個目は不足 1 token = 2 秒。
    let app = app_with(30, Some(1));
    let (s, h, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    assert!(hdr(&h, "retry-after").is_none(), "200 に Retry-After は付けない");

    let (s, h, body) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    let retry = hdr(&h, "retry-after").expect("Retry-After が無い");
    assert!(retry.chars().all(|c| c.is_ascii_digit()), "整数秒のみ: {retry}");
    let secs: u64 = retry.parse().expect("Retry-After は数値");
    assert_eq!(secs, 2, "不足 1 token / 0.5 per sec = 2 秒");
    assert_eq!(h.get_all("retry-after").iter().count(), 1, "重複ヘッダなし");

    let v: Value = serde_json::from_str(&body).expect("429 は JSON");
    let msg = v["error"]["message"].as_str().expect("message は文字列");
    assert!(
        msg.contains(&format!("retry after {secs}s")),
        "ボディの秒数がヘッダと一致しない: {msg}"
    );
    assert_eq!(v["error"]["code"], "rate_limit_exceeded");
}

#[tokio::test]
async fn repeated_denials_keep_retry_after_positive_and_stable() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    let mut seen = Vec::new();
    for _ in 0..5 {
        let (s, h, _) = send(&app, generation()).await;
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
        let secs: u64 = hdr(&h, "retry-after").expect("Retry-After").parse().unwrap();
        assert!(secs >= 1, "Retry-After は常に 1 以上: {secs}");
        seen.push(secs);
    }
    assert!(
        seen.windows(2).all(|w| w[0].abs_diff(w[1]) <= 1),
        "同一時刻の連続拒否で値が暴れない: {seen:?}"
    );
}

// ---------- 複数クライアントの同時押し ----------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_clients_share_one_bucket_across_paths() {
    // バースト 6、回復 1 rpm (テスト中は実質 0)。同時に 6 クライアントが
    // 別々の推論パスへ押しても、通るのは合計 6 件ちょうど。
    let burst = 6u32;
    let clients = 6usize;
    let per_client = 4usize;
    let limiter = RateLimiter::from_config(RateLimitConfig::new(1, Some(burst)));
    let ok = Arc::new(AtomicUsize::new(0));
    let denied = Arc::new(AtomicUsize::new(0));
    // 非同期バリアを使う (std の Barrier は worker を塞いでデッドロックする)。
    let barrier = Arc::new(tokio::sync::Barrier::new(clients));

    let mut handles = Vec::new();
    for c in 0..clients {
        let limiter = limiter.clone();
        let ok = ok.clone();
        let denied = denied.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            for i in 0..per_client {
                let path = RATE_LIMIT_INFERENCE_PATHS[(c + i) % RATE_LIMIT_INFERENCE_PATHS.len()];
                match limiter.check(path) {
                    Decision::Allow => {
                        ok.fetch_add(1, Ordering::SeqCst);
                    }
                    Decision::Deny { retry_after_secs } => {
                        assert!(retry_after_secs >= 1);
                        denied.fetch_add(1, Ordering::SeqCst);
                    }
                }
                tokio::task::yield_now().await;
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    let ok = ok.load(Ordering::SeqCst);
    let denied = denied.load(Ordering::SeqCst);
    assert_eq!(ok, burst as usize, "バースト数を超えて通してはいけない");
    assert_eq!(ok + denied, clients * per_client, "全リクエストが判定された");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_router_clients_are_limited_at_http_layer() {
    // ルータ経由 (HTTP 層) で 4 クライアントが同時に推論を叩く。
    // バースト 3、回復 1 rpm なので、200 はちょうど 3 件、残りは 429。
    let app = app_with(1, Some(3));
    let clients = 4usize;
    let per_client = 3usize;
    let ok = Arc::new(AtomicUsize::new(0));
    let limited = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(tokio::sync::Barrier::new(clients));

    let mut handles = Vec::new();
    for _ in 0..clients {
        let app = app.clone();
        let ok = ok.clone();
        let limited = limited.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier.wait().await;
            for _ in 0..per_client {
                let (s, h, _) = send(&app, generation()).await;
                match s {
                    StatusCode::OK => {
                        assert!(hdr(&h, "retry-after").is_none());
                        ok.fetch_add(1, Ordering::SeqCst);
                    }
                    StatusCode::TOO_MANY_REQUESTS => {
                        let secs: u64 = hdr(&h, "retry-after")
                            .expect("429 には Retry-After")
                            .parse()
                            .unwrap();
                        assert!(secs >= 1);
                        limited.fetch_add(1, Ordering::SeqCst);
                    }
                    other => panic!("想定外のステータス: {other}"),
                }
            }
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    assert_eq!(ok.load(Ordering::SeqCst), 3, "200 はバースト数ちょうど");
    assert_eq!(
        limited.load(Ordering::SeqCst),
        clients * per_client - 3,
        "残りはすべて 429"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_bypass_traffic_never_gets_429() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    // バケットは空。並行する監視アクセスは全部 200。
    let mut handles = Vec::new();
    for i in 0..24usize {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let uri = ["/health", "/ready", "/metrics"][i % 3];
            let (s, _, _) = send(&app, get(uri)).await;
            assert_ne!(s, StatusCode::TOO_MANY_REQUESTS, "{uri} が 429 になった");
        }));
    }
    for h in handles {
        h.await.unwrap();
    }
    // 並行監視の後もバケットは空のまま。
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn concurrent_threads_never_exceed_burst_across_paths() {
    // std スレッド + Barrier で開始を揃え、別パスへ同時に押しても
    // 合計はバースト 5 ちょうど。
    let limiter = RateLimiter::from_config(RateLimitConfig::new(1, Some(5)));
    let barrier = Arc::new(Barrier::new(8));
    let ok = Arc::new(AtomicUsize::new(0));
    let mut handles = Vec::new();
    for t in 0..8usize {
        let limiter = limiter.clone();
        let barrier = barrier.clone();
        let ok = ok.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            for i in 0..5usize {
                let path = RATE_LIMIT_INFERENCE_PATHS[(t + i) % RATE_LIMIT_INFERENCE_PATHS.len()];
                if let Decision::Allow = limiter.check(path) {
                    ok.fetch_add(1, Ordering::SeqCst);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(ok.load(Ordering::SeqCst), 5);
}
