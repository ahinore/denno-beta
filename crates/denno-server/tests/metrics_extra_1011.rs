//! task 1011: `/metrics` の prefix cache 計器とレート制限 429 拒否カウンタ。
//!
//! 契約として固定するもの:
//!   1. `denno_prefix_cache_hits_total` / `_misses_total` / `_reused_tokens_total`
//!      / `_hit_rate` が exposition format 0.0.4 で出る。
//!   2. `/metrics` と `/v1/cache/stats` が同じ計器を読む (値が一致する)。
//!   3. トークンバケットが 429 で拒否すると
//!      `denno_ratelimit_rejections_total{reason="token_bucket"}` が 1 増える。
//!      429 応答も通常の `denno_http_requests_total{status="429"}` に載る。
//!   4. レート制限が枯れていても `/metrics` は 200 で読める (素通しパス)。
//!
//! 計器はプロセス全体で共有されるので、値は絶対値ではなく差分で見る。
//! このテストバイナリ内で直列化する (`SERIAL`)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::metrics;
use denno_server::ratelimit::{RateLimitConfig, RateLimiter};
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// 共有計器を触るテストを 1 本ずつ実行するためのロック。
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn state() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
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

async fn get_text(app: &Router, uri: &str) -> (StatusCode, String) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

async fn scrape(app: &Router) -> String {
    let (status, body) = get_text(app, "/metrics").await;
    assert_eq!(status, StatusCode::OK, "/metrics が 200 でない");
    body
}

async fn post(app: &Router, uri: &str, body: Value) -> StatusCode {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(req).await.unwrap().status()
}

/// `name{labels}` / `name value` の 1 サンプル (ラベルは部分一致)。無ければ None。
fn sample(text: &str, name: &str, labels: &str) -> Option<f64> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .find_map(|l| {
            let (series, val) = l.rsplit_once(' ')?;
            let matches = series == name
                || (series.starts_with(&format!("{name}{{"))
                    && (labels.is_empty() || series.contains(labels)));
            if matches {
                val.parse::<f64>().ok()
            } else {
                None
            }
        })
}

fn value_or_zero(text: &str, name: &str, labels: &str) -> f64 {
    sample(text, name, labels).unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// 1. prefix cache 計器
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prefix_cache_counters_are_exposed_and_shared_with_cache_stats() {
    let _g = SERIAL.lock().await;
    let app = app_with(0, None); // レート制限なし

    let before = scrape(&app).await;
    for family in [
        "denno_prefix_cache_hits_total",
        "denno_prefix_cache_misses_total",
        "denno_prefix_cache_reused_tokens_total",
        "denno_prefix_cache_hit_rate",
        "denno_ratelimit_rejections_total",
    ] {
        assert!(
            before.contains(&format!("# TYPE {family} ")),
            "{family} が /metrics に無い:\n{before}"
        );
    }
    // ラベル付き counter は 0 でも系列を出す (TYPE だけの空ファミリを作らない)。
    assert!(
        before.contains("denno_ratelimit_rejections_total{reason=\"token_bucket\"}"),
        "token_bucket の系列が無い"
    );

    let hits_before = value_or_zero(&before, "denno_prefix_cache_hits_total", "");
    let misses_before = value_or_zero(&before, "denno_prefix_cache_misses_total", "");
    let reused_before = value_or_zero(&before, "denno_prefix_cache_reused_tokens_total", "");

    // 完了 1 本ぶんの再利用を記録する (計器の記録関数そのもの)。
    metrics::record_prefix_cache_reuse(7);
    metrics::record_prefix_cache_reuse(0);

    let after = scrape(&app).await;
    let hits_after = value_or_zero(&after, "denno_prefix_cache_hits_total", "");
    let misses_after = value_or_zero(&after, "denno_prefix_cache_misses_total", "");
    let reused_after = value_or_zero(&after, "denno_prefix_cache_reused_tokens_total", "");

    assert_eq!(hits_after - hits_before, 1.0, "hits の増分");
    assert_eq!(misses_after - misses_before, 1.0, "misses の増分");
    assert_eq!(reused_after - reused_before, 7.0, "reused_tokens の増分");

    // hit_rate は累積の hits / (hits + misses) と一致する。
    let expected = hits_after / (hits_after + misses_after);
    let got = sample(&after, "denno_prefix_cache_hit_rate", "").expect("hit_rate が無い");
    assert!(
        (got - expected).abs() < 1e-6,
        "hit_rate = {got}, 期待 {expected}"
    );

    // /v1/cache/stats も同じ計器を読む (値が一致する)。
    let (status, body) = get_text(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK, "/v1/cache/stats が 200 でない");
    let v: Value = serde_json::from_str(&body).expect("cache stats が JSON でない");
    assert_eq!(v["hits"].as_u64().unwrap() as f64, hits_after, "hits が一致しない");
    assert_eq!(
        v["misses"].as_u64().unwrap() as f64,
        misses_after,
        "misses が一致しない"
    );
    assert_eq!(
        v["total_reused_tokens"].as_u64().unwrap() as f64,
        reused_after,
        "total_reused_tokens が一致しない"
    );
}

#[test]
fn prefix_cache_hit_rate_is_pure() {
    assert_eq!(metrics::prefix_cache_hit_rate(0, 0), 0.0);
    assert_eq!(metrics::prefix_cache_hit_rate(3, 1), 0.75);
    assert_eq!(metrics::prefix_cache_hit_rate(0, 5), 0.0);
}

// ---------------------------------------------------------------------------
// 2. レート制限 429 の拒否カウンタ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limit_429_rejection_is_counted() {
    let _g = SERIAL.lock().await;
    // rpm = 1 / burst = 1: 数 ms では回復しないので 2 本目は必ず 429。
    let app = app_with(1, Some(1));

    let before = scrape(&app).await;
    let rejections_before =
        value_or_zero(&before, "denno_ratelimit_rejections_total", "reason=\"token_bucket\"");
    let http429_before = value_or_zero(
        &before,
        "denno_http_requests_total",
        "endpoint=\"/v1/messages\",status=\"429\"",
    );

    let body = json!({
        "model": "denno-test",
        "max_tokens": 2,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    });
    let first = post(&app, "/v1/messages", body.clone()).await;
    assert!(
        first != StatusCode::TOO_MANY_REQUESTS,
        "1 本目がレート制限で落ちた: {first}"
    );
    let second = post(&app, "/v1/messages", body).await;
    assert_eq!(second, StatusCode::TOO_MANY_REQUESTS, "2 本目が 429 でない");

    let after = scrape(&app).await;
    let rejections_after =
        value_or_zero(&after, "denno_ratelimit_rejections_total", "reason=\"token_bucket\"");
    assert_eq!(
        rejections_after - rejections_before,
        1.0,
        "429 の拒否カウンタが 1 増えていない"
    );

    // 429 応答も HTTP カウンタに載る (log_requests が外側の層)。
    let http429_after = value_or_zero(
        &after,
        "denno_http_requests_total",
        "endpoint=\"/v1/messages\",status=\"429\"",
    );
    assert_eq!(http429_after - http429_before, 1.0, "http 429 が 1 増えていない");

    // 枯れていても /metrics は素通しで 200 (上の scrape がそれ)。
    let (status, _) = get_text(&app, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
}
