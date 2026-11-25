//! P877: Prometheus `/metrics` の統合テスト。
//!
//! - `GET /metrics` が Prometheus text format 0.0.4 の Content-Type で返る。
//! - リクエストを `/v1/messages` と `/v1/chat/completions` に送ると、
//!   `denno_http_requests_total{endpoint=...,status="200"}` が 1 ずつ増える。
//! - 完走した生成がトークンカウンタ / TTFT ヒストグラムに載る。
//! - ヒストグラムは累積で、`le="+Inf"` が `_count` と一致する。
//!
//! 計器はプロセス全体で共有されるので、値は絶対値ではなく差分で見る。
//! このテストバイナリは他の統合テストと別プロセスなので、他ファイルの
//! リクエストは混ざらない。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
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
    denno_server::router(state)
}

async fn get_text(app: &Router, uri: &str) -> (StatusCode, String, String) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ct, String::from_utf8_lossy(&bytes).to_string())
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

/// `name{labels}` またはラベル無し `name` の 1 サンプルを引く
/// (ラベル部分は部分一致)。無ければ None。
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

/// ヒストグラムのバケットが累積で、+Inf が count と一致することを確かめる。
fn assert_histogram(text: &str, name: &str) {
    let mut buckets = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(&format!("{name}_bucket{{le=\"")) {
            let (le, val) = rest.split_once("\"} ").expect("bucket line");
            buckets.push((le.to_string(), val.parse::<u64>().unwrap()));
        }
    }
    assert!(!buckets.is_empty(), "{name}_bucket が無い");
    let mut prev = 0u64;
    for (le, v) in &buckets {
        assert!(*v >= prev, "{name} のバケットが累積でない (le={le})");
        prev = *v;
    }
    let count = sample(text, &format!("{name}_count"), "").expect("count が無い");
    let inf = buckets
        .iter()
        .find(|(le, _)| le == "+Inf")
        .map(|(_, v)| *v as f64)
        .expect("+Inf バケットが無い");
    assert_eq!(inf, count, "{name} の +Inf と _count が不一致");
}

#[tokio::test]
async fn metrics_counts_requests_and_tokens() {
    let app = app();

    // 1. 初期スクレイプ: Content-Type と系列の存在。
    let (status, ct, before) = get_text(&app, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        ct.starts_with("text/plain; version=0.0.4"),
        "Content-Type = {ct}"
    );
    for family in [
        "denno_http_requests_total",
        "denno_prompt_tokens_total",
        "denno_completion_tokens_total",
        "denno_kv_cache_read_tokens_total",
        "denno_kv_reuse_ratio",
        "denno_requests_in_flight",
        "denno_ttft_seconds",
        "denno_tpot_seconds",
    ] {
        assert!(
            before.contains(&format!("# TYPE {family} ")),
            "{family} が /metrics に無い:\n{before}"
        );
    }
    let msg_before = sample(&before, "denno_http_requests_total", "/v1/messages").unwrap_or(0.0);
    let chat_before =
        sample(&before, "denno_http_requests_total", "/v1/chat/completions").unwrap_or(0.0);
    let prompt_before = sample(&before, "denno_prompt_tokens_total", "").unwrap_or(0.0);
    let comp_before = sample(&before, "denno_completion_tokens_total", "").unwrap_or(0.0);
    let ttft_before = sample(&before, "denno_ttft_seconds_count", "").unwrap_or(0.0);

    // 2. Anthropic 互換と OpenAI 互換の生成を 1 本ずつ通す。
    let s = post(
        &app,
        "/v1/messages",
        json!({
            "model": "denno-test",
            "system": "You are a test.",
            "messages": [{"role": "user", "content": "t0"}],
            "max_tokens": 8,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    let s = post(
        &app,
        "/v1/chat/completions",
        json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "t1"}],
            "max_tokens": 8,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(s, StatusCode::OK);

    // 3. 再スクレイプ: カウンタが 1 ずつ増え、トークン / TTFT が載っている。
    let (status, ct, after) = get_text(&app, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.starts_with("text/plain; version=0.0.4"), "Content-Type = {ct}");

    let msg_after = sample(&after, "denno_http_requests_total", "/v1/messages").unwrap();
    let chat_after = sample(&after, "denno_http_requests_total", "/v1/chat/completions").unwrap();
    assert_eq!(msg_after - msg_before, 1.0, "messages のカウンタ");
    assert_eq!(chat_after - chat_before, 1.0, "chat/completions のカウンタ");
    assert!(
        sample(&after, "denno_http_requests_total", "status=\"200\"").unwrap() > 0.0,
        "status ラベルが 200 で出ていない"
    );

    let prompt_after = sample(&after, "denno_prompt_tokens_total", "").unwrap();
    let comp_after = sample(&after, "denno_completion_tokens_total", "").unwrap();
    assert!(prompt_after > prompt_before, "prompt_tokens が増えていない");
    assert!(comp_after > comp_before, "completion_tokens が増えていない");
    let ttft_after = sample(&after, "denno_ttft_seconds_count", "").unwrap();
    assert_eq!(ttft_after - ttft_before, 2.0, "TTFT は 2 リクエストぶん観測される");

    assert_eq!(sample(&after, "denno_requests_in_flight", ""), Some(0.0));
    assert_histogram(&after, "denno_ttft_seconds");
    assert_histogram(&after, "denno_tpot_seconds");
    // モデル名のラベルが載っている。
    assert!(after.contains("denno_model_info{model=\"denno-test\"} 1"));
}
