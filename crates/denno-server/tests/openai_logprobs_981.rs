//! P981: OpenAI `POST /v1/chat/completions` の `logprobs` / `top_logprobs`
//! エッジ硬化テスト。
//!
//! openai_logprobs.rs (P869) と同じ決定的ダミーモデル (peak = 直前トークン+1、
//! logits = [5.0 (peak), 1.0 (peak+1), 0.0 x (V-2)]) を使い、ワイヤレベルで
//! 次を検証する:
//!
//! - 非ストリーム `choices[0].logprobs` のスキーマ (キー集合・bytes = token の
//!   UTF-8・refusal = null) と解析値 (log-softmax) の一致
//! - `top_logprobs` の件数 (0 / 20 / 語彙超えは語彙長でクランプ)・降順・
//!   先頭 = 選択トークン・2 位の解析値
//! - 検証: 0..=20 の境界、`logprobs: true` 必須、範囲外 / 単独指定は
//!   400 `invalid_request_error` (`error.param` 付き)。ストリーム指定でも
//!   生成前に JSON の 400 (SSE を開かない)
//! - 型違い (`"3"`, `3.5`, `1e30`, `true`, `1`) は 400 で `param` が付く
//! - SSE: chunk スキーマ (id / object / choices[0].index / delta {} /
//!   logprobs.refusal = null / finish_reason)、logprobs chunk 数 = 生成数、
//!   usage chunk、[DONE]、非ストリームとの値の一致
//! - `logprobs` 未指定 / false は従来どおり (logprobs: null、本文も同一)

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

/// 決定的ダミーモデル (openai_logprobs.rs と同一)。peak = 直前トークン + 1、
/// EOS の位置だけ避ける。logits は各ステップ [5.0 (peak), 1.0 (peak+1), 0.0 ...]。
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

fn test_app() -> Router {
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
        temperature_cap: None,
    });
    denno_server::router(state)
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

fn expected_second_logprob() -> f64 {
    1f64 - log_denom().ln()
}

fn expected_zero_logprob() -> f64 {
    -(log_denom().ln())
}

fn entry_keys(e: &Value) -> Vec<String> {
    e.as_object()
        .expect("logprobs.content[] が object でない")
        .keys()
        .cloned()
        .collect()
}

/// 1 件の content[] エントリのスキーマと解析値を検証する。
fn assert_content_entry(e: &Value, index: usize, top_n: usize) {
    let mut keys = entry_keys(e);
    keys.sort();
    assert_eq!(
        keys,
        vec!["bytes", "logprob", "token", "top_logprobs"],
        "content[{index}] のキー集合: {e}"
    );

    let token = e["token"].as_str().expect("token が文字列でない");
    assert!(!token.is_empty(), "content[{index}].token が空");

    let bytes: Vec<u8> = e["bytes"]
        .as_array()
        .expect("bytes が配列でない")
        .iter()
        .map(|b| b.as_u64().expect("byte が数値でない") as u8)
        .collect();
    assert_eq!(bytes, token.as_bytes(), "content[{index}].bytes != token の UTF-8");

    let l = e["logprob"].as_f64().expect("logprob が数値でない");
    assert!(l.is_finite() && l <= 0.0, "content[{index}].logprob = {l}");
    assert!(
        (l - expected_peak_logprob()).abs() < 1e-4,
        "content[{index}].logprob {l} != 解析値 {}",
        expected_peak_logprob()
    );

    let top = e["top_logprobs"]
        .as_array()
        .unwrap_or_else(|| panic!("content[{index}].top_logprobs が配列でない: {e}"));
    assert_eq!(top.len(), top_n, "content[{index}].top_logprobs の件数: {e}");

    let mut prev = f64::INFINITY;
    for t in top {
        let mut tk = t.as_object().expect("top[] が object でない").keys().cloned().collect::<Vec<_>>();
        tk.sort();
        assert_eq!(tk, vec!["bytes", "logprob", "token"], "top[] のキー集合: {t}");
        let tl = t["logprob"].as_f64().expect("top[].logprob が数値でない");
        assert!(tl <= prev + 1e-6, "top_logprobs が降順でない: {top:?}");
        prev = tl;
    }
    assert_eq!(top[0]["token"], e["token"], "top[0] が選択トークンでない: {e}");
    assert!(
        (top[0]["logprob"].as_f64().unwrap() - l).abs() < 1e-6,
        "top[0].logprob != content.logprob: {e}"
    );
}

// ---------------------------------------------------------------------------

/// 非ストリームのスキーマと解析値。top_logprobs 3 で 2 位の値まで確認する。
#[tokio::test]
async fn non_stream_schema_and_analytic_values() {
    let app = test_app();
    let mut body = chat_body(8);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(3);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");

    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "denno-test");
    assert_eq!(v["choices"][0]["index"], 0);
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    let completion = v["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    assert_eq!(completion, 8, "dummy モデルは max_tokens まで生成する: {v}");

    let lp = &v["choices"][0]["logprobs"];
    let mut lp_keys = lp
        .as_object()
        .expect("logprobs が object でない")
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    lp_keys.sort();
    assert_eq!(lp_keys, vec!["content", "refusal"]);
    assert_eq!(lp["refusal"], Value::Null);

    let content = lp["content"].as_array().expect("content が配列でない");
    assert_eq!(content.len(), completion, "content 数 != 生成トークン数: {v}");
    for (i, e) in content.iter().enumerate() {
        assert_content_entry(e, i, 3);
        let top = e["top_logprobs"].as_array().unwrap();
        let second = top[1]["logprob"].as_f64().unwrap();
        assert!(
            (second - expected_second_logprob()).abs() < 1e-4,
            "content[{i}] top[1].logprob {second} != 解析値 {}",
            expected_second_logprob()
        );
        let third = top[2]["logprob"].as_f64().unwrap();
        assert!(
            (third - expected_zero_logprob()).abs() < 1e-4,
            "content[{i}] top[2].logprob {third} != 解析値"
        );
    }
}

/// `logprobs: true` で top_logprobs 未指定は空配列 (null ではない)。
#[tokio::test]
async fn top_logprobs_default_is_empty_array() {
    let app = test_app();
    let mut body = chat_body(4);
    body["logprobs"] = json!(true);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    for e in content {
        assert_eq!(
            e["top_logprobs"],
            json!([]),
            "top_logprobs 未指定は空配列: {e}"
        );
    }
}

/// `top_logprobs: 20` (語彙 16 超え) は語彙長でクランプされ、降順で並ぶ。
#[tokio::test]
async fn top_logprobs_clamped_to_vocab_and_sorted() {
    let app = test_app();
    let mut body = chat_body(4);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(20);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content.len(), 4);
    for e in content {
        let top = e["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), DUMMY_VOCAB, "語彙長でクランプされていない: {e}");
        let mut prev = f64::INFINITY;
        for t in top {
            let l = t["logprob"].as_f64().unwrap();
            assert!(l <= prev + 1e-6, "降順でない: {top:?}");
            prev = l;
        }
        assert_eq!(top[0]["token"], e["token"], "top[0] が選択トークンでない");
    }
}

/// 0..=20 の境界と `logprobs: true` 必須。エラーは param 付き 400。
#[tokio::test]
async fn validation_bounds_and_requires_logprobs() {
    let app = test_app();

    // 境界: 0 と 20 は 200
    for n in [0, 20] {
        let mut body = chat_body(2);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "top_logprobs={n}: {v}");
        let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
        let want_n = n.min(DUMMY_VOCAB);
        for e in content {
            assert_eq!(e["top_logprobs"].as_array().unwrap().len(), want_n);
        }
    }

    // 範囲外: 21 / -1 / i64::MIN
    for n in [21i64, -1, i64::MIN] {
        let mut body = chat_body(2);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "top_logprobs={n}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "top_logprobs");
        assert_eq!(v["error"]["code"], Value::Null);
        assert!(v["error"]["message"].as_str().unwrap().contains("top_logprobs"));
        assert!(v["choices"].is_null(), "400 に choices が混ざっている: {v}");
    }

    // 単独指定は 400 (logprobs 未指定 / false の両方)
    let mut body = chat_body(2);
    body["top_logprobs"] = json!(5);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "top_logprobs");

    let mut body = chat_body(2);
    body["logprobs"] = json!(false);
    body["top_logprobs"] = json!(0);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "top_logprobs");
}

/// ストリーム指定でも検証は生成前に走り、SSE ではなく JSON の 400 を返す。
#[tokio::test]
async fn validation_error_is_json_even_with_stream() {
    let app = test_app();
    let mut body = chat_body(2);
    body["stream"] = json!(true);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(21);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    assert!(ctype.starts_with("application/json"), "content-type = {ctype}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(!text.contains("data:"), "400 が SSE 形式になっている: {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "top_logprobs");
}

/// 型違いは 400 で param が付く (P981 の硬化)。
#[tokio::test]
async fn type_errors_carry_param() {
    let app = test_app();

    // top_logprobs が整数でない
    for bad in [json!("3"), json!(3.5), json!(true), json!(1e30), json!([3])] {
        let mut body = chat_body(2);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = bad.clone();
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "top_logprobs={bad}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
        assert_eq!(v["error"]["param"], "top_logprobs", "top_logprobs={bad}: {v}");
    }

    // logprobs が bool でない
    for bad in [json!("true"), json!(1), json!(0), json!([])] {
        let mut body = chat_body(2);
        body["logprobs"] = bad.clone();
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "logprobs={bad}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
        assert_eq!(v["error"]["param"], "logprobs", "logprobs={bad}: {v}");
    }

    // null は「無し」= 200 (従来経路)
    let mut body = chat_body(2);
    body["logprobs"] = Value::Null;
    body["top_logprobs"] = Value::Null;
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["logprobs"], Value::Null);
}

/// `logprobs` 未指定 / false は従来どおり null で、本文も変わらない。
#[tokio::test]
async fn disabled_logprobs_keeps_default_path() {
    let app = test_app();
    let (s0, base) = post(&app, "/v1/chat/completions", chat_body(6)).await;
    assert_eq!(s0, StatusCode::OK);
    assert_eq!(base["choices"][0]["logprobs"], Value::Null);

    for off in [json!(false), Value::Null] {
        let mut body = chat_body(6);
        body["logprobs"] = off.clone();
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "logprobs={off}: {v}");
        assert_eq!(v["choices"][0]["logprobs"], Value::Null);
        assert_eq!(
            v["choices"][0]["message"]["content"], base["choices"][0]["message"]["content"],
            "logprobs={off} で本文が変わる"
        );
    }

    // true でも本文は同一 (生成経路は同じ)
    let mut body = chat_body(6);
    body["logprobs"] = json!(true);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(
        v["choices"][0]["message"]["content"], base["choices"][0]["message"]["content"],
        "logprobs=true で本文が変わる"
    );
    assert_eq!(v["usage"], base["usage"]);
}

/// SSE: chunk スキーマ・logprobs chunk 数・usage chunk・[DONE]。
#[tokio::test]
async fn sse_chunk_schema_and_counts() {
    let app = test_app();
    let mut body = chat_body(6);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let (ctype, text) = post_sse(&app, body).await;
    assert!(ctype.starts_with("text/event-stream"), "content-type = {ctype}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない: {text}"
    );
    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {text}");

    let id = chunks[0]["id"].as_str().expect("id が無い").to_string();
    let created = chunks[0]["created"].as_u64().expect("created が無い");
    for c in &chunks {
        assert_eq!(c["id"], id, "chunk 間で id が違う: {c}");
        assert_eq!(c["object"], "chat.completion.chunk", "object: {c}");
        assert_eq!(c["created"], created, "chunk 間で created が違う: {c}");
        assert_eq!(c["model"], "denno-test", "model: {c}");
        // include_usage=true なので全 chunk に usage (null) が載る
        assert!(c.get("usage").is_some(), "usage スロットが無い: {c}");
    }

    // 先頭 chunk は role
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["choices"][0]["logprobs"], Value::Null);

    // logprobs を持つ chunk
    let lp_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"][0]["logprobs"].is_object())
        .collect();
    assert_eq!(lp_chunks.len(), 6, "logprobs chunk 数 != 生成トークン数");
    for (i, c) in lp_chunks.iter().enumerate() {
        assert_eq!(
            c["choices"].as_array().map(|a| a.len()),
            Some(1),
            "choices が 1 件でない: {c}"
        );
        assert_eq!(c["choices"][0]["index"], 0);
        assert_eq!(c["choices"][0]["delta"], json!({}), "delta が空でない: {c}");
        assert_eq!(c["choices"][0]["finish_reason"], Value::Null, "途中 chunk に finish_reason: {c}");
        let lp = &c["choices"][0]["logprobs"];
        assert_eq!(lp["refusal"], Value::Null);
        let content = lp["content"].as_array().unwrap();
        assert_eq!(content.len(), 1, "1 chunk = 1 トークン: {c}");
        assert_content_entry(&content[0], i, 2);
    }

    // 最終 chunk (finish_reason) は logprobs null
    let fin = chunks
        .iter()
        .find(|c| c["choices"][0]["finish_reason"].is_string())
        .expect("finish_reason chunk が無い");
    assert_eq!(fin["choices"][0]["finish_reason"], "length");
    assert_eq!(fin["choices"][0]["logprobs"], Value::Null);

    // usage chunk は choices が空
    let usage_chunk = chunks
        .iter()
        .find(|c| c["usage"].is_object())
        .expect("usage chunk が無い");
    assert_eq!(usage_chunk["choices"], json!([]));
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 6);
}

/// SSE の logprobs が非ストリームと値まで一致する (同じ greedy 経路)。
#[tokio::test]
async fn sse_logprobs_match_non_stream_values() {
    let app = test_app();
    let mut body = chat_body(5);
    body["stream"] = json!(true);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(3);
    let (_, text) = post_sse(&app, body).await;
    let chunks = sse_chunks(&text);
    let streamed: Vec<Value> = chunks
        .iter()
        .filter(|c| c["choices"][0]["logprobs"].is_object())
        .map(|c| c["choices"][0]["logprobs"]["content"][0].clone())
        .collect();
    assert_eq!(streamed.len(), 5, "logprobs chunk 数 != 生成数");

    let mut plain = chat_body(5);
    plain["logprobs"] = json!(true);
    plain["top_logprobs"] = json!(3);
    let (status, v) = post(&app, "/v1/chat/completions", plain).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(content, &streamed, "stream と非ストリームの logprobs が違う");
}

/// ストリームで logprobs 無効なら logprobs chunk は流れない (全部 null)。
#[tokio::test]
async fn sse_without_logprobs_has_no_logprob_chunks() {
    let app = test_app();
    let mut body = chat_body(4);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (_, text) = post_sse(&app, body).await;
    let chunks = sse_chunks(&text);
    for c in &chunks {
        if let Some(ch) = c["choices"].get(0) {
            assert_eq!(ch["logprobs"], Value::Null, "logprobs 無効なのに載っている: {c}");
        }
    }
    // 本文は非ストリームと同一
    let (status, v) = post(&app, "/v1/chat/completions", chat_body(4)).await;
    assert_eq!(status, StatusCode::OK);
    let want = v["choices"][0]["message"]["content"].as_str().unwrap();
    let got: String = chunks
        .iter()
        .filter_map(|c| c["choices"].get(0))
        .filter_map(|ch| ch["delta"]["content"].as_str())
        .collect();
    assert_eq!(got, want, "ストリーム本文が非ストリームと違う");
}
