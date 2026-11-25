//! P869: OpenAI `POST /v1/chat/completions` の `logprobs` / `top_logprobs` / `seed`。
//!
//! openai_chat.rs と同じ決定的ダミーモデル (peak = 直前トークン+1) を使い、
//! HTTP ワイヤレベルで次を検証する:
//!
//! - `logprobs: true` で `choices[0].logprobs.content[]` が生成トークン数だけ並び、
//!   `{token, logprob, bytes, top_logprobs}` を持つ
//! - logprob の値がサンプル直前 logits の log-softmax と一致する
//!   (ダミーモデルは各ステップ [5, 1, 0, ...] なので解析値が決まる)
//! - `top_logprobs` の件数・降順・先頭が選ばれたトークン (greedy)
//! - `logprobs` / `top_logprobs` の検証 (0..=20、単独指定は 400)
//! - 未指定は従来どおり `logprobs: null` (既定経路不変)
//! - SSE では logprobs が専用 chunk で 1 トークンずつ流れる
//! - `seed` を受理し、同一 seed + 同一入力で出力が一致する

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

/// 決定的ダミーモデル (openai_chat.rs と同一)。peak = 直前トークン + 1、
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

/// SSE 本文から `data: ...` 行の JSON を取り出す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

async fn post_sse(app: &Router, body: Value) -> String {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
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

/// ダミーモデルの 1 ステップ分の log-softmax 解析値 (peak = 5.0 のトークン)。
/// [5, 1, 0, ...] (語彙 DUMMY_VOCAB) なので ln(e^5 / (e^5 + e^1 + (V-2)))。
fn expected_peak_logprob() -> f64 {
    let v = DUMMY_VOCAB as f64;
    let denom = 5f64.exp() + 1f64.exp() + (v - 2.0);
    5f64.exp().ln() - denom.ln()
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_streaming_logprobs_content_and_top() {
    let app = test_app();
    let mut body = chat_body(8);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(3);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");

    let completion = v["usage"]["completion_tokens"].as_u64().unwrap() as usize;
    assert_eq!(completion, 8, "dummy モデルは max_tokens まで生成する: {v}");
    let lp = &v["choices"][0]["logprobs"];
    assert!(lp.is_object(), "logprobs が object でない: {v}");
    let content = lp["content"].as_array().expect("logprobs.content が配列でない");
    assert_eq!(content.len(), completion, "content 数 != 生成トークン数: {v}");

    let want = expected_peak_logprob();
    for (i, e) in content.iter().enumerate() {
        let token = e["token"].as_str().expect("token が文字列でない");
        assert!(!token.is_empty(), "content[{i}].token が空");
        // bytes は token の UTF-8 バイト列
        let bytes: Vec<u8> = e["bytes"]
            .as_array()
            .expect("bytes が配列でない")
            .iter()
            .map(|b| b.as_u64().expect("byte が数値でない") as u8)
            .collect();
        assert_eq!(bytes, token.as_bytes(), "content[{i}].bytes != token の UTF-8");

        let l = e["logprob"].as_f64().expect("logprob が数値でない");
        assert!(l <= 0.0 && l.is_finite(), "content[{i}].logprob = {l}");
        assert!(
            (l - want).abs() < 1e-4,
            "content[{i}].logprob {l} != 解析値 {want}"
        );

        // top_logprobs: 件数・降順・先頭は greedy の選択 (= このトークン)
        let top = e["top_logprobs"].as_array().expect("top_logprobs が配列でない");
        assert_eq!(top.len(), 3, "content[{i}].top_logprobs の件数: {e}");
        let mut prev = f64::INFINITY;
        for t in top {
            assert!(t["token"].is_string(), "top の token が文字列でない: {t}");
            assert!(t["bytes"].is_array(), "top の bytes が配列でない: {t}");
            let tl = t["logprob"].as_f64().unwrap();
            assert!(tl <= prev + 1e-6, "top_logprobs が降順でない: {top:?}");
            prev = tl;
        }
        assert_eq!(top[0]["token"], e["token"], "top[0] が選択トークンでない: {e}");
        assert!(
            (top[0]["logprob"].as_f64().unwrap() - l).abs() < 1e-6,
            "top[0].logprob != content.logprob: {e}"
        );
    }
}

#[tokio::test]
async fn logprobs_absent_is_null() {
    let app = test_app();
    let (status, v) = post(&app, "/v1/chat/completions", chat_body(4)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["logprobs"], Value::Null);
}

#[tokio::test]
async fn top_logprobs_validation() {
    let app = test_app();

    // 21 は範囲外
    let mut body = chat_body(4);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(21);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "top_logprobs");

    // 負値も範囲外
    let mut body = chat_body(4);
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(-1);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "top_logprobs");

    // logprobs 無しの top_logprobs は 400
    let mut body = chat_body(4);
    body["top_logprobs"] = json!(5);
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "top_logprobs");

    // 境界 0 と 20 は 200
    for n in [0, 20] {
        let mut body = chat_body(4);
        body["logprobs"] = json!(true);
        body["top_logprobs"] = json!(n);
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "top_logprobs={n}: {v}");
        let content = v["choices"][0]["logprobs"]["content"].as_array().unwrap();
        assert!(!content.is_empty());
        for e in content {
            // 語彙数を超える要求は語彙長で頭打ちになる (OpenAI 実装と同じ)
            let want_n = (n as usize).min(DUMMY_VOCAB);
            let top = e["top_logprobs"].as_array().unwrap();
            assert_eq!(top.len(), want_n, "top_logprobs={n}: {e}");
        }
    }
}

#[tokio::test]
async fn streaming_logprobs_chunks() {
    let app = test_app();
    let mut body = chat_body(5);
    body["stream"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    body["logprobs"] = json!(true);
    body["top_logprobs"] = json!(2);
    let text = post_sse(&app, body).await;
    assert!(text.trim_end().ends_with("data: [DONE]"), "終端が [DONE] でない");
    let chunks = sse_chunks(&text);

    // logprobs を持つ chunk を集める (delta は空 = content とは別 chunk)
    let lp_chunks: Vec<&Value> = chunks
        .iter()
        .filter(|c| c["choices"][0]["logprobs"].is_object())
        .collect();
    assert_eq!(lp_chunks.len(), 5, "logprobs chunk 数 != 生成トークン数");
    for c in &lp_chunks {
        assert_eq!(c["choices"][0]["delta"], json!({}), "delta が空でない: {c}");
        let content = c["choices"][0]["logprobs"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 1, "1 chunk = 1 トークン: {c}");
        assert_eq!(content[0]["top_logprobs"].as_array().unwrap().len(), 2);
        assert!(content[0]["logprob"].as_f64().unwrap() <= 0.0);
    }

    // usage chunk の completion_tokens と logprobs の総数が一致する
    let usage_chunk = chunks
        .iter()
        .find(|c| c["usage"].is_object())
        .expect("usage chunk が無い");
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 5);

    // 非ストリーミングの本文と同じであること (同じ greedy 経路)
    let mut plain = chat_body(5);
    plain["logprobs"] = json!(true);
    plain["top_logprobs"] = json!(2);
    let (status, v) = post(&app, "/v1/chat/completions", plain).await;
    assert_eq!(status, StatusCode::OK);
    let want: String = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["token"].as_str().unwrap().to_string())
        .collect();
    let got: String = lp_chunks
        .iter()
        .map(|c| {
            c["choices"][0]["logprobs"]["content"][0]["token"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(got, want, "stream の logprobs 列が非ストリームと違う");
}

#[tokio::test]
async fn seed_is_accepted_and_deterministic() {
    let app = test_app();
    let mk = || {
        let mut b = chat_body(16);
        b["temperature"] = json!(1.0);
        b["seed"] = json!(12345);
        b
    };
    let (s1, v1) = post(&app, "/v1/chat/completions", mk()).await;
    let (s2, v2) = post(&app, "/v1/chat/completions", mk()).await;
    assert_eq!(s1, StatusCode::OK, "body = {v1}");
    assert_eq!(s2, StatusCode::OK, "body = {v2}");
    assert_eq!(
        v1["choices"][0]["message"]["content"],
        v2["choices"][0]["message"]["content"],
        "同一 seed + 同一入力で出力が違う"
    );

    // 負の seed も受理する (OpenAI は任意の整数)
    let mut b = chat_body(4);
    b["seed"] = json!(-7);
    let (status, v) = post(&app, "/v1/chat/completions", b).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
}
