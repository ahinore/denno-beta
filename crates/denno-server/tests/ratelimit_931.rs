//! task 931: オプトインのトークンバケット式レート制限の単体 + ルータテスト。
//!
//! 単位テストは時刻を `Instant` で注入するので実時間に依存しない (決定論的)。
//! ルータテストは `rpm = 1` (1 分に 1 トークン回復 = 数 ms では実質 0) の
//! 設定を使うので、バースト数を超えた回数だけが 429 になる。

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::{
    is_bypass_path, is_inference_path, Decision, RateLimitConfig, RateLimiter, TokenBucket,
    RATE_LIMIT_BURST_ENV, RATE_LIMIT_ERROR_CODE, RATE_LIMIT_ERROR_TYPE, RATE_LIMIT_INFERENCE_PATHS,
    RATE_LIMIT_RPM_ENV,
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

/// 明示設定のレート制限器を持つルーター。
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

fn post(uri: &str) -> Request<Body> {
    let body = json!({
        "model": "denno-test",
        "max_tokens": 2,
        "messages": [{"role": "user", "content": "t0"}],
        "prompt": "t0",
        "stream": false,
    });
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

// ---------- 設定の解決 ----------

#[test]
fn config_default_is_disabled() {
    let c = RateLimitConfig::parse(None, None);
    assert_eq!(c, RateLimitConfig::DISABLED);
    assert!(!c.is_enabled());
    // 明示 0 も無効。
    assert!(!RateLimitConfig::parse(Some("0"), Some("8")).is_enabled());
    assert!(!RateLimitConfig::new(0, Some(8)).is_enabled());
}

#[test]
fn config_parses_rpm_and_burst() {
    let c = RateLimitConfig::parse(Some("600"), None);
    assert_eq!(c.rpm, 600);
    assert_eq!(c.burst, 10, "省略時は 1 秒ぶん (600/60)");
    assert!(c.is_enabled());
    assert!((c.per_second() - 10.0).abs() < 1e-9);

    let c = RateLimitConfig::parse(Some(" 120 "), Some(" 3 "));
    assert_eq!((c.rpm, c.burst), (120, 3));

    // 不正 / 空は無効。バースト 0 は既定へ。
    assert!(!RateLimitConfig::parse(Some("abc"), Some("3")).is_enabled());
    assert!(!RateLimitConfig::parse(Some(""), None).is_enabled());
    let c = RateLimitConfig::parse(Some("1"), Some("0"));
    assert_eq!(c.burst, 1, "rpm=1 の既定バーストは 1");

    // 小さい rpm でも既定バーストは最低 1。
    assert_eq!(RateLimitConfig::parse(Some("30"), None).burst, 1);
}

#[test]
fn env_names_are_stable() {
    assert_eq!(RATE_LIMIT_RPM_ENV, "DENNO_RATE_LIMIT_RPM");
    assert_eq!(RATE_LIMIT_BURST_ENV, "DENNO_RATE_LIMIT_BURST");
    assert_eq!(RATE_LIMIT_ERROR_TYPE, "rate_limit_error");
    assert_eq!(RATE_LIMIT_ERROR_CODE, "rate_limit_exceeded");
}

// ---------- パスの分類 ----------

#[test]
fn path_classification() {
    for p in RATE_LIMIT_INFERENCE_PATHS {
        assert!(is_inference_path(p), "{p} は推論エンドポイント");
        assert!(!is_bypass_path(p));
    }
    assert!(is_inference_path("/v1/messages/"), "末尾スラッシュは同じ扱い");
    for p in ["/health", "/ready", "/metrics"] {
        assert!(is_bypass_path(p));
        assert!(!is_inference_path(p), "{p} は素通し");
    }
    for p in ["/v1/models", "/v1/cache/stats", "/v1/telemetry/ws", "/", "/nope"] {
        assert!(!is_inference_path(p), "{p} は対象外");
    }
}

// ---------- バケットの決定論的挙動 ----------

#[test]
fn bucket_allows_burst_then_denies_with_retry_after() {
    let t0 = Instant::now();
    // 60 rpm = 1 token/sec、バースト 3。
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(3)), t0).expect("有効");
    for i in 0..3 {
        assert!(b.try_acquire_at(t0).is_ok(), "{i} 個目は通る");
    }
    // 4 個目は拒否。不足 1 トークン / 1 token/sec = 1 秒。
    assert_eq!(b.try_acquire_at(t0), Err(1));
    // 0.5 秒後: 0.5 トークンでは足りない (切り上げて 1 秒)。
    assert_eq!(b.try_acquire_at(t0 + Duration::from_millis(500)), Err(1));
    // 1 秒後: 1 トークン回復して通る。
    assert!(b.try_acquire_at(t0 + Duration::from_secs(1)).is_ok());
    assert_eq!(b.try_acquire_at(t0 + Duration::from_secs(1)), Err(1));
}

#[test]
fn refill_caps_at_burst_capacity() {
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(600, Some(2)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    assert!(b.try_acquire_at(t0).is_ok());
    assert!(b.try_acquire_at(t0).is_err());
    // 1 時間放置しても容量 2 を超えない。
    let t1 = t0 + Duration::from_secs(3600);
    assert!((b.available_tokens(t1) - 2.0).abs() < 1e-9);
    assert!(b.try_acquire_at(t1).is_ok());
    assert!(b.try_acquire_at(t1).is_ok());
    assert!(b.try_acquire_at(t1).is_err(), "3 個目は容量超過");
}

#[test]
fn retry_after_rounds_up_to_whole_seconds_and_is_never_zero() {
    let t0 = Instant::now();
    // 600 rpm = 10 token/sec。バースト 1 を使い切る。
    let b = TokenBucket::new_at(RateLimitConfig::new(600, Some(1)), t0).expect("有効");
    assert!(b.try_acquire_at(t0).is_ok());
    // 0.05 秒で 0.5 トークン。不足 0.5 / 10 = 0.05 秒 → 切り上げ 1 秒。
    assert_eq!(b.try_acquire_at(t0 + Duration::from_millis(50)), Err(1));
    // 0.095 秒: 0.95 トークン。不足 0.05 / 10 = 0.005 秒 → 1 秒。
    assert_eq!(b.try_acquire_at(t0 + Duration::from_millis(95)), Err(1));
    assert!(b.try_acquire_at(t0 + Duration::from_millis(100)).is_ok());
}

#[test]
fn token_bucket_rejects_disabled_config() {
    assert!(TokenBucket::new(RateLimitConfig::DISABLED).is_none());
    let l = RateLimiter::from_config(RateLimitConfig::DISABLED);
    assert!(!l.is_enabled());
}

#[test]
fn limiter_skips_paths_outside_inference_set_without_consuming() {
    let t0 = Instant::now();
    let l = RateLimiter::from_config_at(RateLimitConfig::new(60, Some(1)), t0);
    assert!(l.is_enabled());
    // 対象外パスは何度でも Allow で、トークンを減らさない。
    for _ in 0..50 {
        assert_eq!(l.check_at("/health", t0), Decision::Allow);
        assert_eq!(l.check_at("/v1/models", t0), Decision::Allow);
        assert_eq!(l.check_at("/metrics", t0), Decision::Allow);
    }
    // まだ 1 トークン残っているので推論パスは 1 回通る。
    assert_eq!(l.check_at("/v1/messages", t0), Decision::Allow);
    assert_eq!(
        l.check_at("/v1/messages", t0),
        Decision::Deny { retry_after_secs: 1 }
    );
    // 429 になった後でも /health は通る。
    assert_eq!(l.check_at("/health", t0), Decision::Allow);
}

#[test]
fn all_inference_paths_are_limited() {
    let t0 = Instant::now();
    for path in RATE_LIMIT_INFERENCE_PATHS {
        let l = RateLimiter::from_config_at(RateLimitConfig::new(1, Some(1)), t0);
        assert_eq!(l.check_at(path, t0), Decision::Allow, "{path} 1 回目");
        assert!(
            matches!(l.check_at(path, t0), Decision::Deny { .. }),
            "{path} 2 回目は拒否"
        );
    }
}

#[test]
fn clone_shares_bucket_state() {
    let t0 = Instant::now();
    let l = RateLimiter::from_config_at(RateLimitConfig::new(1, Some(2)), t0);
    let c = l.clone();
    assert_eq!(l.check_at("/v1/messages", t0), Decision::Allow);
    assert_eq!(c.check_at("/v1/messages", t0), Decision::Allow);
    assert!(matches!(
        l.check_at("/v1/messages", t0),
        Decision::Deny { .. }
    ));
}

#[test]
fn concurrent_acquires_never_exceed_burst() {
    // 8 スレッド x 8 回、バースト 5。実時間の回復量 (1 rpm = 1/60 token/sec) は
    // 無視できるので、ちょうど 5 回だけ通る。
    let l = RateLimiter::from_config(RateLimitConfig::new(1, Some(5)));
    let allowed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut handles = Vec::new();
    for _ in 0..8 {
        let l = l.clone();
        let allowed = allowed.clone();
        handles.push(std::thread::spawn(move || {
            for _ in 0..8 {
                if let Decision::Allow = l.check("/v1/messages") {
                    allowed.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert_eq!(
        allowed.load(std::sync::atomic::Ordering::SeqCst),
        5,
        "バースト数を超えて通してはいけない"
    );
}

// ---------- env の解決 (このファイルで env を触るのはこの 1 本だけ) ----------

#[test]
fn from_env_reads_both_variables() {
    std::env::remove_var(RATE_LIMIT_RPM_ENV);
    std::env::remove_var(RATE_LIMIT_BURST_ENV);
    assert!(!RateLimiter::from_env().is_enabled(), "既定は無効");

    std::env::set_var(RATE_LIMIT_RPM_ENV, "120");
    let l = RateLimiter::from_env();
    assert!(l.is_enabled());
    assert_eq!(l.config().rpm, 120);
    assert_eq!(l.config().burst, 2);

    std::env::set_var(RATE_LIMIT_BURST_ENV, "5");
    let l = RateLimiter::from_env();
    assert_eq!(l.config().burst, 5, "バーストは env が上書きする");

    std::env::remove_var(RATE_LIMIT_RPM_ENV);
    std::env::remove_var(RATE_LIMIT_BURST_ENV);
    assert!(!RateLimiter::from_env().is_enabled());
}

// ---------- ルータ: 無効時は従来どおり ----------

#[tokio::test]
async fn disabled_limiter_never_throttles() {
    let app = app_with(0, None);
    for i in 0..5 {
        let (s, _, _) = send(&app, generation()).await;
        assert_eq!(s, StatusCode::OK, "{i} 回目が 429 になった");
    }
}

// ---------- ルータ: 有効時の 429 ----------

#[tokio::test]
async fn exceeded_requests_get_429_with_retry_after_and_json() {
    // バースト 2、回復 1 rpm (テスト中は実質 0)。
    let app = app_with(1, Some(2));
    for i in 0..2 {
        let (s, _, _) = send(&app, generation()).await;
        assert_eq!(s, StatusCode::OK, "{i} 回目は通る");
    }
    let (s, h, body) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(hdr(&h, "content-type").as_deref(), Some("application/json"));
    let retry = hdr(&h, "retry-after").expect("Retry-After が無い");
    let secs: u64 = retry.parse().expect("Retry-After は整数秒");
    assert!(secs >= 1, "Retry-After = {secs}");
    let v: Value = serde_json::from_str(&body).expect("429 は JSON");
    assert_eq!(v["error"]["type"], RATE_LIMIT_ERROR_TYPE);
    assert_eq!(v["error"]["code"], RATE_LIMIT_ERROR_CODE);
    assert!(v["error"].get("param").is_some(), "param キーは null で存在する");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("Rate limit"),
        "{v}"
    );
    // 429 も log 層を通る。
    assert!(hdr(&h, "x-request-id").is_some());
}

#[tokio::test]
async fn health_ready_metrics_bypass_even_when_exhausted() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    // バケットは空。監視系は 429 にならない。
    for _ in 0..3 {
        let (s, _, _) = send(&app, get("/health")).await;
        assert_eq!(s, StatusCode::OK, "/health は素通し");
    }
    let (s, _, body) = send(&app, get("/metrics")).await;
    assert_eq!(s, StatusCode::OK, "/metrics は素通し");
    assert!(body.contains("denno_"), "metrics のボディが空: {body}");
    let (s, _, _) = send(&app, get("/ready")).await;
    assert_ne!(s, StatusCode::TOO_MANY_REQUESTS, "/ready は素通し");
    // まだ空のまま: 推論は 429。
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn non_inference_endpoints_are_not_limited() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    for _ in 0..3 {
        let (s, _, _) = send(&app, get("/v1/models")).await;
        assert_eq!(s, StatusCode::OK, "/v1/models は対象外");
    }
    // 未知ルートも 404 のまま (429 にしない)。
    let (s, _, _) = send(&app, get("/definitely-not-a-route")).await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    // 推論は空のバケットで 429。
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn openai_and_ollama_inference_paths_share_the_bucket() {
    let app = app_with(1, Some(2));
    let (s, _, _) = send(&app, post("/v1/chat/completions")).await;
    assert_eq!(s, StatusCode::OK);
    // 別パスでも同じバケットを消費する (2 個目)。
    let (s, _, _) = send(&app, post("/api/chat")).await;
    assert_ne!(s, StatusCode::TOO_MANY_REQUESTS, "2 個目は通る");
    let (s, h, _) = send(&app, post("/v1/completions")).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "3 個目は 429");
    assert!(hdr(&h, "retry-after").is_some());
}

#[tokio::test]
async fn preflight_does_not_consume_tokens() {
    let app = app_with(1, Some(1));
    // プリフライトは CORS 層が 204 で返す (トークンは減らない)。
    for _ in 0..3 {
        let req = Request::builder()
            .method("OPTIONS")
            .uri("/v1/messages")
            .header("origin", "http://localhost:5173")
            .header("access-control-request-method", "POST")
            .body(Body::empty())
            .unwrap();
        let (s, _, _) = send(&app, req).await;
        assert_eq!(s, StatusCode::NO_CONTENT);
    }
    // 1 トークンは残っている。
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn rate_limited_response_carries_cors_headers() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    let mut req = generation();
    req.headers_mut()
        .insert("origin", "http://localhost:5173".parse().unwrap());
    let (s, h, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        hdr(&h, "access-control-allow-origin").as_deref(),
        Some("http://localhost:5173"),
        "429 にも CORS ヘッダが付く"
    );
}

#[tokio::test]
async fn recovery_after_wait_allows_again() {
    // 60000 rpm = 1000 token/sec、バースト 1。数 ms 待てば 1 トークン回復する。
    let app = app_with(60_000, Some(1));
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    tokio::time::sleep(Duration::from_millis(20)).await;
    let (s, _, _) = send(&app, generation()).await;
    assert_eq!(s, StatusCode::OK, "待てば回復する");
}
