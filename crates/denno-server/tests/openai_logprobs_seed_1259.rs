//! P1259: OpenAI `POST /v1/chat/completions` の `logprobs` / `top_logprobs`
//! 境界と、`seed` 併用時の決定性をワイヤレベルで追加検証する。
//!
//! 既存の P981 (`openai_logprobs_981.rs`) と P986 (`openai_seed_986.rs`) を
//! 補完する位置づけで、次を確認する:
//!
//! - `top_logprobs` の境界 0 / 1 / 20 / 21。0 は空配列、1 は 1 件、
//!   20 は語彙長 (16) でクランプ、21 は 400 `invalid_request_error`
//!   (`error.param = "top_logprobs"`)
//! - `logprobs: false` / null / 未指定 + `top_logprobs` は 400 で
//!   `top_logprobs` 側に param が付く (0 / 20 の境界値でも同じ)
//! - `system_fingerprint` が同時実行 (並行リクエスト) でも全リクエストで
//!   同一値になる (構成由来で、競合・乱数・時刻に依存しない)
//! - 同一 seed + logprobs 有効の同時実行は本文・logprobs 配列・
//!   fingerprint まで一致する
//!
//! ダミーモデルは P981 の決定モデル (peak = 直前トークン + 1、greedy) と
//! P986 の一様モデル (0..=11 が同値、EOS / 未定義語は -1e30) を使う。

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

/// P981 と同じ決定的モデル。greedy で peak が必ず選ばれる。
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

/// P986 と同じ一様モデル (0..=11 が同値、EOS / 未定義語は -1e30)。
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

fn app_with(model: Box<dyn Model>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
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
    denno_server::router(state)
}

fn seq_app() -> Router {
    app_with(Box::new(SeqModel))
}

fn uniform_app() -> Router {
    app_with(Box::new(UniformModel))
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

fn chat_body(max_tokens: usize) -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
}

/// logits [5, 1, 0 x (V-2)] の log-softmax 解析値。
fn log_denom() -> f64 {
    let v = DUMMY_VOCAB as f64;
    5f64.exp() + 1f64.exp() + (v - 2.0)
}

fn expected_peak_logprob() -> f64 {
    5f64 - log_denom().ln()
}

/// `fp_` + 16 hex を検証して返す。
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

fn content_entries(v: &Value, want: usize) -> Vec<Value> {
    let content = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("logprobs.content が配列でない: {v}"))
        .clone();
    assert_eq!(content.len(), want, "content 数 != 生成トークン数: {v}");
    content
}

// ---------------------------------------------------------------------------

/// `top_logprobs` の境界 0 / 1 / 20 / 21。
#[tokio::test]
async fn top_logprobs_boundary_0_1_20_21() {
    let app = seq_app();

    // 0: 有効・空配列。1: 1 件。20: 語彙長 16 でクランプ。
    for (n, want_len) in [(0i64, 0usize), (1, 1), (20, DUMMY_VOCAB)] {
        let mut body = chat_body(4);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "top_logprobs={n}: {v}");
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["logprobs"]["refusal"], Value::Null);

        for e in content_entries(&v, 4) {
            let top = e["top_logprobs"]
                .as_array()
                .unwrap_or_else(|| panic!("top_logprobs={n} で配列でない: {e}"));
            assert_eq!(top.len(), want_len, "top_logprobs={n} の件数: {e}");
            let mut prev = f64::INFINITY;
            for t in top {
                let l = t["logprob"].as_f64().expect("top[].logprob が数値でない");
                assert!(l <= prev + 1e-6, "top_logprobs={n} が降順でない: {e}");
                prev = l;
            }
            if want_len > 0 {
                assert_eq!(top[0]["token"], e["token"], "top[0] が選択トークンでない: {e}");
                let l0 = top[0]["logprob"].as_f64().unwrap();
                assert!(
                    (l0 - expected_peak_logprob()).abs() < 1e-4,
                    "top_logprobs={n} の top[0].logprob {l0} != 解析値"
                );
                assert!(
                    (e["logprob"].as_f64().unwrap() - l0).abs() < 1e-6,
                    "content.logprob != top[0].logprob: {e}"
                );
            }
        }
    }

    // 21: 範囲外は 400 (choices を混ぜない)。
    let mut body = chat_body(4);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(21);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "top_logprobs=21: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "top_logprobs");
    assert!(v["choices"].is_null(), "400 に choices が混ざっている: {v}");
}

/// `logprobs: false` / null / 未指定 + `top_logprobs` は境界値でも 400。
#[tokio::test]
async fn logprobs_false_with_top_logprobs_is_rejected() {
    let app = seq_app();

    for n in [0i64, 1, 20] {
        // false
        let mut body = chat_body(2);
        body["logprobs"] = json!(false);
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "false + {n}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "false + {n}: {v}");
        assert_eq!(v["error"]["param"], "top_logprobs", "false + {n}: {v}");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("requires logprobs=true"),
            "false + {n} のメッセージ: {v}"
        );

        // null
        let mut body = chat_body(2);
        body["logprobs"] = Value::Null;
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "null + {n}: {v}");
        assert_eq!(v["error"]["param"], "top_logprobs", "null + {n}: {v}");

        // 未指定
        let mut body = chat_body(2);
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "absent + {n}: {v}");
        assert_eq!(v["error"]["param"], "top_logprobs", "absent + {n}: {v}");
    }

    // logprobs=true + top_logprobs=0 は引き続き 200 (false と取り違えない)。
    let mut body = chat_body(2);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(0);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "true + 0: {v}");
}

/// `system_fingerprint` は並行リクエストでも全件同一 (構成由来)。
#[tokio::test]
async fn fingerprint_is_deterministic_across_concurrent_calls() {
    let app = seq_app();

    // 逐次ベースライン。
    let mut base_body = chat_body(2);
    base_body["seed"] = json!(7);
    let (s0, base) = post(&app, "/v1/chat/completions", base_body).await;
    assert_eq!(s0, StatusCode::OK, "baseline: {base}");
    let want = fingerprint_of(&base).to_string();

    // 同時に 6 本。seed / temperature / logprobs / top_logprobs をばらす。
    let mut handles = Vec::new();
    for i in 0..6i64 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let mut body = chat_body(3);
            body["seed"] = json!(i * 101 - 50);
            body["temperature"] = json!(if i % 2 == 0 { 0.0 } else { 0.8 });
            if i % 3 != 2 {
                body["logprobs"] = json!(true);
                body["top_logprobs"] = json!(i % 21);
            }
            post(&app, "/v1/chat/completions", body).await
        }));
    }
    let mut ids = std::collections::HashSet::new();
    for h in handles {
        let (status, v) = h.await.expect("同時リクエストの task が panic");
        assert_eq!(status, StatusCode::OK, "同時リクエストが失敗: {v}");
        assert_eq!(
            fingerprint_of(&v),
            want,
            "同時リクエストで fingerprint が変わった: {v}"
        );
        let id = v["id"].as_str().expect("id が無い").to_string();
        assert!(ids.insert(id), "同時リクエストで id が重複: {v}");
    }
}

/// 同一 seed + logprobs 有効の同時実行は本文・logprobs・fingerprint 一致。
#[tokio::test]
async fn same_seed_logprobs_identical_across_concurrent_calls() {
    let app = uniform_app();
    let max_tokens = 6usize;

    let mut handles = Vec::new();
    for _ in 0..4 {
        let app = app.clone();
        handles.push(tokio::spawn(async move {
            let mut body = chat_body(max_tokens);
            body["temperature"] = json!(0.9);
            body["seed"] = json!(4242);
            body["logprobs"] = json!(true);
            body["top_logprobs"] = json!(20);
            post(&app, "/v1/chat/completions", body).await
        }));
    }

    let mut want_text: Option<String> = None;
    let mut want_lp: Option<Vec<Value>> = None;
    let mut want_fp: Option<String> = None;
    for h in handles {
        let (status, v) = h.await.expect("同時リクエストの task が panic");
        assert_eq!(status, StatusCode::OK, "同時リクエストが失敗: {v}");
        assert_eq!(
            v["usage"]["completion_tokens"].as_u64(),
            Some(max_tokens as u64),
            "生成数が max_tokens でない: {v}"
        );

        let text = v["choices"][0]["message"]["content"]
            .as_str()
            .expect("content が文字列でない")
            .to_string();
        let lp = content_entries(&v, max_tokens);
        // 一様 12 候補なので全トークンの logprob は -ln(12)、top は語彙長 16 件
        // (0..=11 が -ln(12)、EOS / 未定義語が -1e30 で後ろに並ぶ)。
        for (i, e) in lp.iter().enumerate() {
            let l = e["logprob"].as_f64().expect("logprob が数値でない");
            assert!(
                (l - -(12f64).ln()).abs() < 1e-4,
                "content[{i}].logprob {l} != -ln(12)"
            );
            let top = e["top_logprobs"].as_array().unwrap();
            assert_eq!(top.len(), DUMMY_VOCAB, "top_logprobs が語彙長でない: {e}");
            for t in &top[..12] {
                let tl = t["logprob"].as_f64().unwrap();
                assert!(
                    (tl - -(12f64).ln()).abs() < 1e-4,
                    "content[{i}] の有限候補 logprob {tl} != -ln(12)"
                );
            }
        }

        let fp = fingerprint_of(&v).to_string();
        match (&want_text, &want_lp, &want_fp) {
            (Some(t), Some(l), Some(f)) => {
                assert_eq!(&text, t, "同一 seed で本文が違う: {v}");
                assert_eq!(&lp, l, "同一 seed で logprobs が違う: {v}");
                assert_eq!(&fp, f, "同一 seed で fingerprint が違う: {v}");
            }
            _ => {
                want_text = Some(text);
                want_lp = Some(lp);
                want_fp = Some(fp);
            }
        }
    }
}
