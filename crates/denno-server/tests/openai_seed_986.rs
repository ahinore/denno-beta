//! P986: OpenAI `POST /v1/chat/completions` の `system_fingerprint` と
//! `seed` の決定性テスト。
//!
//! 検証する内容 (ワイヤレベル):
//!
//! - 非ストリーム応答に `system_fingerprint` (`fp_` + 16hex) が載り、
//!   同一構成の全リクエスト / 別プロセス相当 (別 AppState) で同じ値になる
//! - seed / temperature / stream の別に依存しない (fingerprint は構成由来)
//! - 構成 (model_name / model_id) が変われば fingerprint も変わる
//! - ストリームの全 chunk (role / content / logprobs / finish / usage) に
//!   同じ fingerprint が載る
//! - 同一 seed + 同一プロンプトは非ストリーム / ストリームで同一本文
//!   (temperature > 0 でも決定的)、seed が違えば本文が変わる
//! - seed の型違いは 400 `invalid_request_error` で `error.param = "seed"`
//!
//! ダミーモデルは語彙 0..=11 を一様 (logit 0)、EOS と未定義語 (12..=15) を
//! -1e30 にして、seed 無しでも EOS で早期停止しないようにする。

use std::sync::{Arc, Mutex};

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

const TEST_CONTEXT: usize = 512;

/// 12 候補の一様分布モデル (EOS / 未定義語は -1e30)。
struct UniformModel;

impl Model for UniformModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let mut logits = vec![-1e30f32; DUMMY_VOCAB];
        for l in logits.iter_mut().take(12) {
            *l = 0.0;
        }
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn test_app_with(model_name: &str, model_id: &str) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(UniformModel), tok, model_name, None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: model_name.to_string(),
        model_id: model_id.to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

fn test_app() -> Router {
    test_app_with("denno-test", "Qwen/Qwen2.5-0.5B-Instruct")
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

/// SSE 応答を (content-type, 本文) で返す。ステータスは 200 前提。
async fn post_sse(app: &Router, body: Value) -> (String, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (ctype, String::from_utf8(bytes.to_vec()).unwrap())
}

/// SSE 本文から `data: ...` 行の JSON を取り出す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn chat_body(max_tokens: usize) -> Value {
    chat_body_for("denno-test", max_tokens)
}

fn chat_body_for(model: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
}

fn fingerprint_of(v: &Value) -> &str {
    let fp = v["system_fingerprint"]
        .as_str()
        .unwrap_or_else(|| panic!("system_fingerprint が無い / 文字列でない: {v}"));
    assert!(fp.starts_with("fp_"), "system_fingerprint の形: {fp}");
    assert_eq!(fp.len(), 19, "fp_ + 16hex でない: {fp}");
    assert!(
        fp[3..].bytes().all(|b| b.is_ascii_hexdigit()),
        "16 桁 hex でない: {fp}"
    );
    fp
}

/// ストリームの delta.content を連結して本文にする。
fn stream_text(chunks: &[Value]) -> String {
    let mut out = String::new();
    for c in chunks {
        if let Some(t) = c["choices"][0]["delta"]["content"].as_str() {
            out.push_str(t);
        }
    }
    out
}

// ---------------------------------------------------------------------------

/// 非ストリーム: フィールドが載り、同一構成では毎回同じ値。
#[tokio::test]
async fn non_stream_fingerprint_present_and_stable() {
    let app = test_app();
    let (s0, a) = post(&app, chat_body(4)).await;
    assert_eq!(s0, StatusCode::OK, "body = {a}");
    let (s1, b) = post(&app, chat_body(4)).await;
    assert_eq!(s1, StatusCode::OK, "body = {b}");

    let fa = fingerprint_of(&a).to_string();
    assert_eq!(fa, fingerprint_of(&b), "同一構成で fingerprint が変わる");
    assert_eq!(a["object"], "chat.completion");

    // 別 AppState (= 別プロセス相当) でも同じ値 (乱数や時刻に依存しない)
    let app2 = test_app();
    let (s2, c) = post(&app2, chat_body(2)).await;
    assert_eq!(s2, StatusCode::OK, "body = {c}");
    assert_eq!(fa, fingerprint_of(&c), "インスタンス間で fingerprint が違う");
}

/// fingerprint は seed / temperature / stream に依存しない。
#[tokio::test]
async fn fingerprint_independent_of_seed_and_stream() {
    let app = test_app();
    let (_, base) = post(&app, chat_body(4)).await;
    let want = fingerprint_of(&base).to_string();

    for seed in [json!(0), json!(1), json!(-5), json!(i64::MAX / 2), Value::Null] {
        let mut body = chat_body(4);
        body["seed"] = seed.clone();
        body["temperature"] = json!(0.7);
        let (status, v) = post(&app, body).await;
        assert_eq!(status, StatusCode::OK, "seed={seed}: {v}");
        assert_eq!(fingerprint_of(&v), want, "seed={seed} で fingerprint が変わる");
    }

    let mut body = chat_body(4);
    body["stream"] = json!(true);
    let (ctype, text) = post_sse(&app, body).await;
    assert!(ctype.starts_with("text/event-stream"), "content-type = {ctype}");
    let chunks = sse_chunks(&text);
    assert!(!chunks.is_empty(), "chunk が無い: {text}");
    for c in &chunks {
        assert_eq!(fingerprint_of(c), want, "stream chunk の fingerprint: {c}");
    }
}

/// ストリームの全 chunk (role / content / logprobs / finish / usage) に載る。
#[tokio::test]
async fn stream_every_chunk_carries_fingerprint() {
    let app = test_app();
    let mut body = chat_body(5);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    body["logprobs"] = json!(true);
    body["seed"] = json!(77);
    let (_, text) = post_sse(&app, body).await;
    assert!(text.trim_end().ends_with("data: [DONE]"), "終端が [DONE] でない");
    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 4, "chunk が少なすぎる: {text}");

    let fp = fingerprint_of(&chunks[0]).to_string();
    let mut saw_usage_chunk = false;
    let mut saw_logprob_chunk = false;
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(fingerprint_of(c), fp, "chunk 間で fingerprint が違う: {c}");
        if c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(false) {
            saw_usage_chunk = true;
            assert!(c["usage"].is_object(), "usage chunk に usage が無い: {c}");
        }
        if c["choices"][0]["logprobs"].is_object() {
            saw_logprob_chunk = true;
        }
    }
    assert!(saw_usage_chunk, "usage chunk が無い: {text}");
    assert!(saw_logprob_chunk, "logprobs chunk が無い: {text}");

    // 非ストリームと同じ fingerprint
    let mut nb = chat_body(5);
    nb["seed"] = json!(77);
    let (_, non_stream) = post(&app, nb).await;
    assert_eq!(fingerprint_of(&non_stream), fp);
}

/// 構成が変われば fingerprint も変わる (定数でないことの確認)。
#[tokio::test]
async fn fingerprint_depends_on_model_config() {
    let base = test_app();
    let (_, a) = post(&base, chat_body(2)).await;
    let other_name = test_app_with("denno-other", "Qwen/Qwen2.5-0.5B-Instruct");
    let (_, b) = post(&other_name, chat_body_for("denno-other", 2)).await;
    let other_id = test_app_with("denno-test", "Qwen/Qwen2.5-1.5B-Instruct");
    let (_, c) = post(&other_id, chat_body_for("denno-test", 2)).await;

    assert_ne!(fingerprint_of(&a), fingerprint_of(&b), "model_name を変えても同じ");
    assert_ne!(fingerprint_of(&a), fingerprint_of(&c), "model_id を変えても同じ");
}

/// 同一 seed は非ストリーム / ストリームで同一本文。seed 違いは本文が変わる。
#[tokio::test]
async fn seed_makes_generation_deterministic_and_distinct() {
    let app = test_app();
    let max_tokens = 24usize;

    let mut body = chat_body(max_tokens);
    body["temperature"] = json!(0.9);
    body["seed"] = json!(1234);
    let (s0, a) = post(&app, body.clone()).await;
    assert_eq!(s0, StatusCode::OK, "body = {a}");
    let (s1, b) = post(&app, body.clone()).await;
    assert_eq!(s1, StatusCode::OK, "body = {b}");
    let text_a = a["choices"][0]["message"]["content"]
        .as_str()
        .expect("content が文字列でない")
        .to_string();
    assert_eq!(a["usage"]["completion_tokens"].as_u64(), Some(max_tokens as u64));
    assert_eq!(text_a, b["choices"][0]["message"]["content"], "同一 seed で本文が変わる");

    // ストリームでも同一 seed → 同一本文
    let mut sbody = body.clone();
    sbody["stream"] = json!(true);
    let (_, stext) = post_sse(&app, sbody).await;
    let streamed = stream_text(&sse_chunks(&stext));
    assert_eq!(streamed, text_a, "seed 固定で stream と non-stream の本文が違う");

    // seed 違いは本文が変わる (12 候補 x 24 トークンの一致確率は無視できる)
    let mut other = chat_body(max_tokens);
    other["temperature"] = json!(0.9);
    other["seed"] = json!(4321);
    let (s2, c) = post(&app, other).await;
    assert_eq!(s2, StatusCode::OK, "body = {c}");
    assert_ne!(
        text_a, c["choices"][0]["message"]["content"],
        "seed を変えても本文が同じ (seed が engine に伝わっていない可能性)"
    );
}

/// seed の型違いは param 付き 400。null / 未指定は従来どおり 200。
#[tokio::test]
async fn seed_type_errors_carry_param() {
    let app = test_app();
    for bad in [json!("7"), json!(7.5), json!(true), json!([7]), json!({"v": 7})] {
        let mut body = chat_body(2);
        body["seed"] = bad.clone();
        let (status, v) = post(&app, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "seed={bad}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "seed={bad}: {v}");
        assert_eq!(v["error"]["param"], "seed", "seed={bad}: {v}");
        assert!(v["choices"].is_null(), "400 に choices が混ざっている: {v}");
    }

    // null と未指定はどちらも 200 (seed 無し = engine 既定)
    let mut body = chat_body(2);
    body["seed"] = Value::Null;
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "seed=null: {v}");
}
