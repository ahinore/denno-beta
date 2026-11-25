//! task 936: OpenAI `stream_options.include_usage` の完全準拠テスト。
//!
//! 検証する仕様 (OpenAI API reference / chat completions & legacy completions):
//!
//! - `stream_options` 未指定 / `include_usage: false` のストリームには
//!   `usage` フィールド自体が現れない。
//! - `include_usage: true` のときは `data: [DONE]` の直前に、
//!   `choices: []` + 完全な `usage` を持つ chunk が 1 個だけ流れる。
//! - `include_usage: true` のそれ以外の chunk は `"usage": null` を持つ
//!   ("All other chunks will also include a usage field, but with a null value")。
//! - tool call ストリームでも `finish_reason: "tool_calls"` の chunk の後に
//!   同じ usage chunk が来る。
//!
//! 方式は既存の統合テスト (tests/openai_chat.rs / tests/openai_tools.rs) と
//! 同じ — denno-engine のダミートークナイザ + 決定的モデルで HTTP ワイヤを見る。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{
    dummy_tokenizer, tool_tokenizer, DUMMY_EOS, DUMMY_VOCAB, ScriptModel, TOOL_VOCAB,
};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 512;

/// 決定的ダミーモデル (peak = 直前トークン + 1、EOS の位置だけ避ける)。
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
        let peak = (last + 1) % DUMMY_VOCAB;
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn app_from(engine: Engine) -> Router {
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

fn test_app() -> Router {
    let engine = Engine::from_parts(
        Box::new(SeqModel),
        dummy_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    app_from(engine)
}

/// TOOL_VOCAB の piece → id。
fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// `get_weather(city=Tokyo)` の完全な `<tool_call>` block を吐くスクリプト。
fn block_script(name: &str) -> Vec<u32> {
    [
        "Checking.",
        "<tool_call>",
        "Ċ",
        "<function=",
        name,
        ">",
        "Ċ",
        "<parameter=",
        "city",
        ">",
        "Ċ",
        "Tokyo",
        "Ċ",
        "</parameter>",
        "Ċ",
        "</function>",
        "Ċ",
        "</tool_call>",
        "eos",
    ]
    .iter()
    .map(|p| tid(p))
    .collect()
}

async fn post(app: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// SSE 本文をそのまま返す (順序と `[DONE]` を見るため文字列でも扱う)。
async fn post_sse(app: &Router, uri: &str, body: &Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// `data: ...` 行の JSON を到着順に返す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn completion_body(prompt: Value, stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "prompt": prompt,
        "max_tokens": 4,
        "temperature": 0.0,
        "stream": stream,
    })
}

/// `usage` の 3 フィールドが自己整合であることを確かめる。
fn assert_usage_complete(usage: &Value) -> (u64, u64) {
    let p = usage["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("prompt_tokens が数値でない: {usage}"));
    let c = usage["completion_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("completion_tokens が数値でない: {usage}"));
    let t = usage["total_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("total_tokens が数値でない: {usage}"));
    assert_eq!(t, p + c, "total_tokens != prompt + completion: {usage}");
    (p, c)
}

/// usage chunk が「最後の chunk」で、その直後が `[DONE]` であることを確かめる。
fn assert_usage_chunk_is_last(chunks: &[Value], text: &str) -> Value {
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
    let last = chunks.last().unwrap();
    assert_eq!(
        last["choices"],
        json!([]),
        "usage chunk の choices が空配列でない: {last}"
    );
    assert!(last["usage"].is_object(), "最後の chunk が usage でない: {last}");
    // usage chunk より前に usage オブジェクトを持つ chunk は無い (1 個だけ)。
    let n = chunks.iter().filter(|c| c["usage"].is_object()).count();
    assert_eq!(n, 1, "usage chunk が {n} 個ある: {chunks:?}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない:\n{text}"
    );
    // [DONE] の直前の data 行が usage chunk (間に空行/コメント以外を挟まない)。
    let before_done = text
        .trim_end()
        .strip_suffix("data: [DONE]")
        .expect("[DONE] が無い")
        .trim_end();
    let last_data = before_done
        .rfind("data: ")
        .map(|i| &before_done[i + "data: ".len()..])
        .expect("data 行が無い");
    let parsed: Value = serde_json::from_str(last_data.trim()).expect("直前 data が JSON でない");
    assert!(
        parsed["usage"].is_object() && parsed["choices"] == json!([]),
        "usage chunk が [DONE] の直前でない: {parsed}"
    );
    last.clone()
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_usage_absent_when_omitted_or_false() {
    let app = test_app();
    // 非ストリーミングの usage を基準に取る。
    let (status, base) = post(&app, "/v1/chat/completions", &chat_body(false)).await;
    assert_eq!(status, StatusCode::OK, "body = {base}");

    for body in [
        chat_body(true), // stream_options 未指定
        {
            let mut b = chat_body(true);
            b["stream_options"] = json!({"include_usage": false});
            b
        },
    ] {
        let (status, text) = post_sse(&app, "/v1/chat/completions", &body).await;
        assert_eq!(status, StatusCode::OK, "body = {body}");
        let chunks = sse_chunks(&text);
        assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
        for c in &chunks {
            assert!(
                c.get("usage").is_none(),
                "include_usage 無効なのに usage フィールドがある: {c}"
            );
        }
        // 本数の増減が無い (usage chunk も null slot も出ていない)。
        assert!(!text.contains("\"usage\""), "usage 文字列が混ざる:\n{text}");
    }
}

#[tokio::test]
async fn chat_include_usage_emits_final_usage_chunk_before_done() {
    let app = test_app();
    let (_, base) = post(&app, "/v1/chat/completions", &chat_body(false)).await;
    let (bp, bc) = assert_usage_complete(&base["usage"]);

    let mut body = chat_body(true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, text) = post_sse(&app, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);

    // 途中の chunk は usage: null を持つ (フィールド欠落ではなく明示 null)。
    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(
            c.get("usage"),
            Some(&Value::Null),
            "非最終 chunk の usage が明示 null でない: {c}"
        );
    }
    let usage = assert_usage_chunk_is_last(&chunks, &text);
    let (p, c) = assert_usage_complete(&usage["usage"]);
    // 非ストリーミングと同じ数 (同じ入力・同じ生成)。
    assert_eq!((p, c), (bp, bc), "stream と non-stream の usage が違う");
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");
    // usage chunk も chat.completion.chunk / 同じ id。
    assert_eq!(usage["object"], "chat.completion.chunk");
    assert_eq!(usage["id"], chunks[0]["id"]);
    // finish_reason は usage chunk より前の最後の choices 付き chunk にある。
    let last_choice = chunks[..chunks.len() - 1]
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last_choice["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_tool_call_stream_usage_follows_tool_calls_finish() {
    let model = ScriptModel::new(block_script("get_weather"), tid("t1"));
    let engine = Engine::from_parts(
        Box::new(model),
        tool_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    let app = app_from(engine);

    let body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "weather in tokyo?"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": true,
        "stream_options": {"include_usage": true},
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
    });
    let (status, text) = post_sse(&app, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);

    // tool_call の delta が実際に流れている (この経路の証拠)。
    let tool_delta = chunks
        .iter()
        .any(|c| c["choices"][0]["delta"]["tool_calls"].is_array());
    assert!(tool_delta, "tool_calls delta が無い: {chunks:?}");

    let usage = assert_usage_chunk_is_last(&chunks, &text);
    let (p, c) = assert_usage_complete(&usage["usage"]);
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");

    // finish_reason = tool_calls の chunk が usage より前にある。
    let last_choice = chunks[..chunks.len() - 1]
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last_choice["choices"][0]["finish_reason"], "tool_calls");
    // それ以外の chunk は usage: null。
    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(
            c.get("usage"),
            Some(&Value::Null),
            "非最終 chunk の usage: {c}"
        );
    }
}

#[tokio::test]
async fn chat_tool_call_stream_without_include_usage_has_no_usage() {
    let model = ScriptModel::new(block_script("get_weather"), tid("t1"));
    let engine = Engine::from_parts(
        Box::new(model),
        tool_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    let app = app_from(engine);

    let body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "weather in tokyo?"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": true,
        "tools": [{
            "type": "function",
            "function": {"name": "get_weather", "description": "Get current weather"}
        }],
    });
    let (status, text) = post_sse(&app, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);
    assert!(chunks.iter().all(|c| c.get("usage").is_none()), "{chunks:?}");
    let last_choice = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last_choice["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn chat_stream_options_wrong_type_is_openai_shaped_400() {
    let app = test_app();
    let mut body = chat_body(true);
    body["stream_options"] = json!({"include_usage": "yes"});
    let (status, v) = post(&app, "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert!(
        v["error"]["message"].is_string(),
        "OpenAI 形式の error 本文でない: {v}"
    );
}

// ---------------------------------------------------------------------------
// /v1/completions (legacy text completion)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_usage_absent_when_omitted_or_false() {
    let app = test_app();
    for body in [
        completion_body(json!("t0 t1 t2"), true),
        {
            let mut b = completion_body(json!("t0 t1 t2"), true);
            b["stream_options"] = json!({"include_usage": false});
            b
        },
    ] {
        let (status, text) = post_sse(&app, "/v1/completions", &body).await;
        assert_eq!(status, StatusCode::OK, "body = {body}");
        assert!(!text.contains("\"usage\""), "usage が出ている:\n{text}");
        for c in sse_chunks(&text) {
            assert_eq!(c["object"], "text_completion");
        }
    }
}

#[tokio::test]
async fn completions_include_usage_emits_final_usage_chunk_before_done() {
    let app = test_app();
    let (_, base) = post(&app, "/v1/completions", &completion_body(json!("t0 t1 t2"), false)).await;
    let (bp, bc) = assert_usage_complete(&base["usage"]);

    let mut body = completion_body(json!("t0 t1 t2"), true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, text) = post_sse(&app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);

    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(
            c.get("usage"),
            Some(&Value::Null),
            "非最終 chunk の usage: {c}"
        );
        assert_eq!(c["object"], "text_completion");
    }
    let usage = assert_usage_chunk_is_last(&chunks, &text);
    assert_eq!(usage["object"], "text_completion");
    assert_eq!(usage["id"], chunks[0]["id"]);
    let (p, c) = assert_usage_complete(&usage["usage"]);
    assert_eq!((p, c), (bp, bc), "stream と non-stream の usage が違う");
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");
}

#[tokio::test]
async fn completions_echo_include_usage_marks_all_chunks() {
    let app = test_app();
    let mut body = completion_body(json!("t0 t1 t2"), true);
    body["echo"] = json!(true);
    body["stream_options"] = json!({"include_usage": true});
    let (status, text) = post_sse(&app, "/v1/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_chunks(&text);
    // echo chunk (最初) にも usage: null が付く。
    assert_eq!(
        chunks[0].get("usage"),
        Some(&Value::Null),
        "echo chunk に usage: null が無い: {chunks:?}"
    );
    let usage = assert_usage_chunk_is_last(&chunks, &text);
    let (p, c) = assert_usage_complete(&usage["usage"]);
    assert!(p > 0 && c > 0, "p = {p}, c = {c}");
}
