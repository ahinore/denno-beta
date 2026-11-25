//! P966: OpenAI サンプリングパラメータの範囲検証。
//!
//! `/v1/chat/completions` と `/v1/completions` の両方で、範囲外 / 型不正な
//! パラメータが param 名付きの 400 invalid_request_error で決定的に返り、
//! 境界値と未指定 (既存の既定) は従来どおり受理されることを HTTP ワイヤで
//! 検証する。
//!
//! 対象:
//! - temperature: 0..=2 (非有限・負値・上限超は 400)
//! - top_p: (0, 1] (0 以下・1 超・非有限は 400)
//! - n: 1 のみ (未実装の n != 1 は 400)
//! - max_tokens / max_completion_tokens: 負値は 400 (上限超は従来どおり clamp)
//! - serve 側 temperature cap は範囲判定より先に効く (cap 内なら 400 にしない)

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler, MAX_TOKENS_CLAMP};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 8192;

/// openai_logprobs.rs と同一の決定的ダミーモデル (peak = 直前トークン + 1)。
struct SeqModel;

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let mut peak = (last + 1) % DUMMY_VOCAB;
        if peak == DUMMY_EOS as usize {
            peak += 1;
        }
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn test_app_with_cap(temperature_cap: Option<f32>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(SeqModel), tok, "denno-test", None);
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
        temperature_cap,
    });
    denno_server::router(state)
}

fn test_app() -> Router {
    test_app_with_cap(None)
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
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn chat_body() -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4,
        "temperature": 0.0,
    })
}

fn completion_body() -> Value {
    json!({
        "model": "denno-test",
        "prompt": "hello",
        "max_tokens": 4,
        "temperature": 0.0,
    })
}

/// 400 invalid_request_error の形と param を固定する。
fn assert_invalid(status: StatusCode, v: &Value, param: &str, ctx: &str) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "{ctx}: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{ctx}: {v}");
    assert_eq!(v["error"]["param"], param, "{ctx}: {v}");
    assert!(
        v["error"]["message"].as_str().map(|m| !m.is_empty()).unwrap_or(false),
        "{ctx}: message が空: {v}"
    );
}

/// 両エンドポイントで同じ body 変形を流す。
async fn post_both(app: &Router, mutate: impl Fn(&mut Value)) -> [(StatusCode, Value); 2] {
    let mut chat = chat_body();
    mutate(&mut chat);
    let mut cmpl = completion_body();
    mutate(&mut cmpl);
    [
        post(app, "/v1/chat/completions", chat).await,
        post(app, "/v1/completions", cmpl).await,
    ]
}

// ---------- temperature ----------

#[tokio::test]
async fn temperature_boundaries_are_accepted() {
    let app = test_app();
    for t in [0.0f64, 0.5, 1.0, 2.0] {
        let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["temperature"] = json!(t)).await;
        assert_eq!(s1, StatusCode::OK, "chat temperature={t}: {v1}");
        assert_eq!(s2, StatusCode::OK, "completions temperature={t}: {v2}");
    }
}

#[tokio::test]
async fn temperature_out_of_range_is_400() {
    let app = test_app();
    for t in [-0.1f64, -100.0, 2.0001, 3.0, 1e30] {
        let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["temperature"] = json!(t)).await;
        assert_invalid(s1, &v1, "temperature", &format!("chat temperature={t}"));
        assert_invalid(s2, &v2, "temperature", &format!("completions temperature={t}"));
    }
}

#[tokio::test]
async fn temperature_non_finite_is_400() {
    let app = test_app();
    // f32 の上限を超える JSON 数値は serde が inf にする (3.5e38 -> inf)。
    // ここでは finite チェックが param 名付き 400 を返すことを固定する。
    for raw in ["3.5e38", "-3.5e38"] {
        let chat = format!(
            "{{\"model\":\"denno-test\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],\
             \"max_tokens\":4,\"temperature\":{raw}}}"
        );
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(chat))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        assert_invalid(status, &v, "temperature", &format!("temperature={raw}"));
    }

    // NaN / inf を JSON の数値として直接書いた場合 (1e400) は serde が
    // body 自体を拒否する。それでも 400 invalid_request_error であることは保つ。
    let chat = "{\"model\":\"denno-test\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\
                \"max_tokens\":4,\"temperature\":1e400}";
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
}

// ---------- top_p ----------

#[tokio::test]
async fn top_p_boundaries_are_accepted() {
    let app = test_app();
    for p in [1.0f64, 0.9, 1e-30] {
        let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["top_p"] = json!(p)).await;
        assert_eq!(s1, StatusCode::OK, "chat top_p={p}: {v1}");
        assert_eq!(s2, StatusCode::OK, "completions top_p={p}: {v2}");
    }
}

#[tokio::test]
async fn top_p_out_of_range_is_400() {
    let app = test_app();
    for p in [0.0f64, -0.5, 1.0001, 1.5, 100.0] {
        let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["top_p"] = json!(p)).await;
        assert_invalid(s1, &v1, "top_p", &format!("chat top_p={p}"));
        assert_invalid(s2, &v2, "top_p", &format!("completions top_p={p}"));
    }
}

// ---------- n ----------

#[tokio::test]
async fn n_only_one_is_supported() {
    let app = test_app();
    // n = 1 は明示しても可
    let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["n"] = json!(1)).await;
    assert_eq!(s1, StatusCode::OK, "chat n=1: {v1}");
    assert_eq!(s2, StatusCode::OK, "completions n=1: {v2}");

    // 未実装の n != 1 (0・複数・負) は 400
    for n in [0i64, 2, 10, -1] {
        let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["n"] = json!(n)).await;
        assert_invalid(s1, &v1, "n", &format!("chat n={n}"));
        assert_invalid(s2, &v2, "n", &format!("completions n={n}"));
    }
}

// ---------- max_tokens ----------

#[tokio::test]
async fn negative_max_tokens_is_400_with_param_name() {
    let app = test_app();
    let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["max_tokens"] = json!(-1)).await;
    assert_invalid(s1, &v1, "max_tokens", "chat max_tokens=-1");
    assert_invalid(s2, &v2, "max_tokens", "completions max_tokens=-1");

    // 別名の負値も別名で報告する
    let mut b = chat_body();
    b["max_tokens"] = Value::Null;
    b.as_object_mut().unwrap().remove("max_tokens");
    b["max_completion_tokens"] = json!(-4);
    let (s, v) = post(&app, "/v1/chat/completions", b).await;
    assert_invalid(s, &v, "max_completion_tokens", "max_completion_tokens=-4");
}

#[tokio::test]
async fn max_tokens_zero_and_clamp_are_preserved() {
    let app = test_app();
    // 0 は従来どおり 0 として尊重 (生成なし)
    let [(s1, v1), (s2, v2)] = post_both(&app, |b| b["max_tokens"] = json!(0)).await;
    assert_eq!(s1, StatusCode::OK, "chat max_tokens=0: {v1}");
    assert_eq!(v1["usage"]["completion_tokens"], 0, "{v1}");
    assert_eq!(s2, StatusCode::OK, "completions max_tokens=0: {v2}");
    assert_eq!(v2["usage"]["completion_tokens"], 0, "{v2}");

    // 上限超は従来どおり clamp (400 にしない)
    let mut b = chat_body();
    b["max_tokens"] = json!(100_000);
    let (s, v) = post(&app, "/v1/chat/completions", b).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], MAX_TOKENS_CLAMP as u64, "{v}");
}

// ---------- 既存の既定 / serve 側 cap ----------

#[tokio::test]
async fn unspecified_params_keep_defaults() {
    let app = test_app();
    for (uri, body) in [
        (
            "/v1/chat/completions",
            json!({"model": "denno-test", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 4}),
        ),
        (
            "/v1/completions",
            json!({"model": "denno-test", "prompt": "hi", "max_tokens": 4}),
        ),
    ] {
        let (s, v) = post(&app, uri, body).await;
        assert_eq!(s, StatusCode::OK, "{uri}: {v}");
        assert_eq!(v["usage"]["completion_tokens"], 4, "{uri}: {v}");
    }
}

#[tokio::test]
async fn serve_temperature_cap_applies_before_range_check() {
    // cap = 0.5 のとき、範囲外 (2.5) の要求は cap で丸められて受理される
    // (cap は「クライアントの機械的な temperature=1 を抑える」既存機能なので、
    //  cap 内に収まる値は 400 にしない)。
    let app = test_app_with_cap(Some(0.5));
    let (s, v) = post(&app, "/v1/chat/completions", {
        let mut b = chat_body();
        b["temperature"] = json!(2.5);
        b
    })
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");

    // cap が無ければ同じ 2.5 は 400 (上のテストと同じ経路)
    let app = test_app();
    let (s, v) = post(&app, "/v1/chat/completions", {
        let mut b = chat_body();
        b["temperature"] = json!(2.5);
        b
    })
    .await;
    assert_invalid(s, &v, "temperature", "no cap, temperature=2.5");
}

#[tokio::test]
async fn server_stays_healthy_after_rejections() {
    let app = test_app();
    // 400 を出した後に同じ app で通常要求が通る (スロット漏れ・状態汚染なし)
    let (s, v) = post(&app, "/v1/chat/completions", {
        let mut b = chat_body();
        b["temperature"] = json!(-1.0);
        b
    })
    .await;
    assert_invalid(s, &v, "temperature", "reject first");
    let (s, v) = post(&app, "/v1/chat/completions", chat_body()).await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["usage"]["completion_tokens"], 4, "{v}");
}
