//! Task 1044: `/v1/models` と旧 `/v1/completions` の契約エッジケース。
//!
//! denno-engine の決定的ダミーモデル (`tests/openai_completions.rs` と同一方式) を
//! 使い、HTTP ワイヤだけを見る。カバー範囲:
//!
//! - `GET /v1/models` の一覧スキーマ (`object="list"`、`data[]` の
//!   `id` / `object` / `owned_by`) と認証
//! - `/v1/completions` の非ストリーミング応答 (`text_completion` / `cmpl-` id /
//!   `choices[]` / `usage`)
//! - SSE (`text_completion` chunk 列 + 終端 `[DONE]`)
//! - `prompt` の文字列 vs 1 要素配列、空文字 / batch の 400
//! - `stop` (文字列 / 配列) の打ち切り
//! - `seed` / `logprobs` (旧 Completions では未対応 = 受理して無視し、
//!   `logprobs` は常に null) の扱い
//! - 壊れた JSON が OpenAI 形式の 4xx JSON で返ること

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
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
const CANONICAL: &str = "denno-test";

/// 決定的ダミーモデル。peak = 直前トークン + 1。
struct SeqModel {
    /// EOS を出さない (max_tokens まで必ず続く)
    endless: bool,
}

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        if self.endless {
            &[]
        } else {
            &[DUMMY_EOS]
        }
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = (last + 1) % DUMMY_VOCAB;
        let peak = if self.endless && peak == DUMMY_EOS as usize {
            peak + 1
        } else {
            peak
        };
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn app_with_key(api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(SeqModel { endless: false }),
        tok,
        CANONICAL,
        None,
    );
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: CANONICAL.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

fn app() -> Router {
    app_with_key(None)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
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

async fn post_sse(app: &Router, uri: &str, body: Value) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ct, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn sse_text(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["text"].as_str())
        .collect()
}

fn words(v: &Value) -> Vec<&str> {
    v.as_str()
        .unwrap_or_default()
        .split_whitespace()
        .collect()
}

fn completion_body(prompt: Value, stream: bool) -> Value {
    json!({
        "model": CANONICAL,
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": stream,
    })
}

// ---------------------------------------------------------------------------
// GET /v1/models
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_list_schema() {
    let app = app();
    let (status, body) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["object"], "list", "body = {body}");
    assert_eq!(body["has_more"], false, "body = {body}");

    let data = body["data"].as_array().expect("data が配列でない");
    assert_eq!(data.len(), 1, "data = {data:?}");
    let m = &data[0];
    // 一覧の各要素は id / object / owned_by を持つ。
    assert_eq!(m["id"], CANONICAL, "m = {m}");
    assert_eq!(m["object"], "model", "m = {m}");
    assert_eq!(m["owned_by"], "denno", "m = {m}");
    assert!(
        m["created"].as_u64().unwrap_or(0) > 1_577_836_800,
        "created が実時刻でない: m = {m}"
    );
    assert_eq!(m["capabilities"]["openai"], true, "m = {m}");
}

#[tokio::test]
async fn models_list_requires_auth_when_key_is_set() {
    let app = app_with_key(Some("secret"));
    let (status, body) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "body = {body}");
    assert!(body["error"].is_object(), "body = {body}");
}

// ---------------------------------------------------------------------------
// /v1/completions: 非ストリーミング
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_streaming_text_completion_shape() {
    let app = app();
    let (status, v) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");

    assert_eq!(v["object"], "text_completion", "v = {v}");
    let id = v["id"].as_str().expect("id が文字列でない");
    assert!(id.starts_with("cmpl-"), "id = {id}");
    assert_eq!(id.len(), "cmpl-".len() + 24, "id = {id}");
    assert!(v["created"].as_u64().unwrap_or(0) > 0, "v = {v}");
    assert_eq!(v["model"], CANONICAL, "v = {v}");
    assert_eq!(v["choices"].as_array().map(|a| a.len()), Some(1), "v = {v}");

    let c = &v["choices"][0];
    assert_eq!(c["index"], 0, "c = {c}");
    assert_eq!(c["logprobs"], Value::Null, "c = {c}");
    assert_eq!(c["finish_reason"], "length", "c = {c}");
    // chat template を通さない: t0 t1 t2 の続きはちょうど t3..t6。
    assert_eq!(words(&c["text"]), ["t3", "t4", "t5", "t6"], "v = {v}");

    let p = v["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens");
    let g = v["usage"]["completion_tokens"].as_u64().expect("completion_tokens");
    assert_eq!(p, 3, "v = {v}");
    assert_eq!(g, 4, "v = {v}");
    assert_eq!(v["usage"]["total_tokens"].as_u64(), Some(p + g), "v = {v}");
}

#[tokio::test]
async fn prompt_string_and_single_element_array_agree() {
    let app = app();
    let (status, a) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    assert_eq!(status, StatusCode::OK, "body = {a}");
    let (status, b) = post(
        &app,
        "/v1/completions",
        completion_body(json!(["t0 t1 t2"]), false),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {b}");
    assert_eq!(a["choices"][0]["text"], b["choices"][0]["text"]);
    assert_eq!(a["usage"]["prompt_tokens"], b["usage"]["prompt_tokens"]);
    assert_eq!(a["usage"]["completion_tokens"], b["usage"]["completion_tokens"]);
}

#[tokio::test]
async fn invalid_prompts_are_openai_4xx() {
    let app = app();

    // 空文字は生成不能。
    let (status, v) = post(&app, "/v1/completions", completion_body(json!(""), false)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "v = {v}");
    assert_eq!(v["error"]["param"], "prompt", "v = {v}");

    // 複数プロンプト (batch) は非対応。
    let (status, v) = post(
        &app,
        "/v1/completions",
        completion_body(json!(["t0", "t1"]), false),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "prompt", "v = {v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("batch of 2 prompts"),
        "v = {v}"
    );

    // トークン ID 配列 (1 要素でも中身が文字列でない) は 400。
    let (status, v) = post(&app, "/v1/completions", completion_body(json!([1]), false)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "prompt[0]", "v = {v}");
}

// ---------------------------------------------------------------------------
// /v1/completions: SSE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_sse_text_completion_chunks() {
    let app = app();
    let (_, full) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    let want = full["choices"][0]["text"].as_str().unwrap().to_string();

    let (status, ct, text) =
        post_sse(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), true)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        ct.starts_with("text/event-stream"),
        "content-type = {ct:?}"
    );
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない:\n{text}"
    );
    assert!(!text.contains("event: "), "event: 行が混ざっている:\n{text}");

    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
    for c in &chunks {
        assert_eq!(c["object"], "text_completion", "c = {c}");
        assert_eq!(c["id"], chunks[0]["id"], "c = {c}");
        assert_eq!(c["choices"][0]["logprobs"], Value::Null, "c = {c}");
    }
    assert!(chunks[0]["id"].as_str().unwrap().starts_with("cmpl-"));
    // chunk の text 結合 = 非ストリーミング本文。
    assert_eq!(sse_text(&chunks), want, "chunks = {chunks:?}");

    // finish_reason は最後の choices 付き chunk にだけ入る。
    let last = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert!(
        last["choices"][0]["finish_reason"].is_string(),
        "final chunk に finish_reason が無い: {last}"
    );
    // include_usage 未指定では usage chunk を出さない。
    assert!(
        !chunks.iter().any(|c| c["usage"].is_object()),
        "include_usage 無しで usage が出ている: {chunks:?}"
    );
}

#[tokio::test]
async fn streaming_with_include_usage_emits_usage_chunk_before_done() {
    let app = app();
    let mut b = completion_body(json!("t0 t1 t2"), true);
    b["stream_options"] = json!({"include_usage": true});
    let (status, _, text) = post_sse(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::OK);

    let chunks = sse_chunks(&text);
    let usage = chunks
        .iter()
        .find(|c| c["usage"].is_object())
        .expect("usage chunk が無い");
    assert_eq!(usage["choices"], json!([]), "usage = {usage}");
    let p = usage["usage"]["prompt_tokens"].as_u64().unwrap();
    let g = usage["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(usage["usage"]["total_tokens"].as_u64(), Some(p + g));
    // usage chunk は [DONE] より前にある。
    let done_at = text.rfind("data: [DONE]").expect("[DONE] が無い");
    let usage_at = text.find("\"usage\"").expect("usage 本文が無い");
    assert!(usage_at < done_at, "usage が [DONE] より後にある:\n{text}");
}

// ---------------------------------------------------------------------------
// stop / seed / logprobs
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_sequences_string_and_array_truncate() {
    let app = app();
    let (_, full) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;
    let full_text = full["choices"][0]["text"].as_str().unwrap().to_string();
    let last_word = full_text.split_whitespace().last().unwrap().to_string();

    for stop in [json!(last_word), json!([last_word])] {
        let mut b = completion_body(json!("t0 t1 t2"), false);
        b["stop"] = stop;
        let (status, v) = post(&app, "/v1/completions", b).await;
        assert_eq!(status, StatusCode::OK, "body = {v}");
        let text = v["choices"][0]["text"].as_str().unwrap();
        assert!(
            !text.contains(&last_word),
            "stop 語 {last_word:?} が残っている: {text:?}"
        );
        assert_eq!(v["choices"][0]["finish_reason"], "stop", "v = {v}");
    }
}

#[tokio::test]
async fn seed_and_logprobs_are_accepted_but_not_applied() {
    let app = app();
    let (_, plain) = post(&app, "/v1/completions", completion_body(json!("t0 t1 t2"), false)).await;

    // 旧 Completions の実装は seed / logprobs / top_logprobs を持たない。
    // 未知フィールドとして受理し (400 にせず)、生成結果は変えない。
    let mut b = completion_body(json!("t0 t1 t2"), false);
    b["seed"] = json!(42);
    b["logprobs"] = json!(true);
    b["top_logprobs"] = json!(5);
    let (status, v) = post(&app, "/v1/completions", b.clone()).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(
        v["choices"][0]["text"], plain["choices"][0]["text"],
        "seed/logprobs で生成結果が変わった: v = {v}"
    );
    // logprobs は要求しても常に null (未対応であることをワイヤで固定)。
    assert_eq!(v["choices"][0]["logprobs"], Value::Null, "v = {v}");

    // ストリームでも同様に受理され、chunk の logprobs は null。
    b["stream"] = json!(true);
    let (status, _, text) = post_sse(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);
    assert!(!chunks.is_empty(), "chunk が無い:\n{text}");
    for c in &chunks {
        assert_eq!(c["choices"][0]["logprobs"], Value::Null, "c = {c}");
    }
}

// ---------------------------------------------------------------------------
// 壊れた JSON
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_json_returns_clean_openai_error() {
    let app = app();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(Body::from("{\"prompt\": "))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).expect("エラー本文が JSON ではない");

    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert!(ct.starts_with("application/json"), "content-type = {ct:?}");
    // OpenAI 形式: error.message / error.type / error.param / error.code
    assert!(v["error"].is_object(), "body = {v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("invalid request body"),
        "body = {v}"
    );
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    assert!(v["error"].get("param").is_some(), "body = {v}");
    assert!(v["error"].get("code").is_some(), "body = {v}");
}

#[tokio::test]
async fn stop_of_wrong_type_is_clean_400() {
    let app = app();
    let mut b = completion_body(json!("t0 t1 t2"), false);
    b["stop"] = json!(3);
    let (status, v) = post(&app, "/v1/completions", b).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert!(v["error"].is_object(), "body = {v}");
}

/// `[DONE]` の後に余計な data chunk が続かないこと (一部クライアントは最初の
/// `[DONE]` で読み取りを止めるため、順序を固定する)。
#[tokio::test]
async fn done_is_the_last_sse_frame() {
    let app = app();
    let (status, _, text) = post_sse(
        &app,
        "/v1/completions",
        completion_body(json!("t0 t1 t2"), true),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let mut seen_done = false;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        if seen_done {
            assert!(
                line.starts_with(':'),
                "[DONE] の後に data が続く: {line:?}\n{text}"
            );
        }
        if line == "data: [DONE]" {
            seen_done = true;
        }
    }
    assert!(seen_done, "[DONE] が無い:\n{text}");
}

/// 生成そのものは時間を要さない (ダミーモデル) が、SSE タスクが終わるまで
/// 本文を読み切る経路を明示的に通す (無限に待たないことの回帰)。
#[tokio::test]
async fn streaming_completes_within_timeout() {
    let app = app();
    let fut = post_sse(
        &app,
        "/v1/completions",
        completion_body(json!("t0 t1 t2"), true),
    );
    let (status, _, text) = tokio::time::timeout(Duration::from_secs(10), fut)
        .await
        .expect("SSE が 10 秒以内に終わらない");
    assert_eq!(status, StatusCode::OK);
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");
}
