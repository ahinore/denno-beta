//! P797: OpenAI 互換 `POST /v1/chat/completions` の tool calling 統合テスト。
//!
//! `denno-engine` の tool 用ダミー (ByteLevel デコーダ + `ScriptModel`) を使い、
//! モデルが実際に `<tool_call>` block を出力する状況を作って HTTP ワイヤで検証する。
//! カバー項目:
//!
//! - 非ストリーミング: `message.tool_calls` (id/type/function.name/arguments) と
//!   `finish_reason="tool_calls"`、複数 tool call
//! - tools 無し / テキストのみ出力のときは `tool_calls` を出さない (従来経路)
//! - 履歴の assistant `tool_calls` と `role="tool"` を受理して生成まで通る
//! - 不正な履歴 (`arguments` が JSON オブジェクトでない) は 400
//! - ストリーミング: `delta.tool_calls` (index / id / type / function.name /
//!   増分 arguments) の組み立てと `finish_reason="tool_calls"`
//! - `tool_choice` (`required` = 拘束デコード / 名前指定 / 未定義名 400 /
//!   tools 無しで指定 400)
//! - `response_format` (`text` / `json_object` 受理、未対応 type は 400)

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, tool_tokenizer, DummyModel, ScriptModel, TOOL_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// TOOL_VOCAB の piece → id。
fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// 完全な tool_call block を吐くスクリプト (名前 + `city=Tokyo`)。
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

/// スクリプト再生モデル (`vocab` 48) で app を作る。
fn script_app(script: Vec<u32>) -> Router {
    let model = ScriptModel::new(script, tid("t1"));
    let engine = Engine::from_parts(Box::new(model), tool_tokenizer().unwrap(), "denno-test", None);
    state_app(engine)
}

/// 通常のダミーモデル (tool call を出さない) で app を作る。
fn plain_app() -> Router {
    let engine = Engine::from_parts(Box::new(DummyModel), dummy_tokenizer().unwrap(), "denno-test", None);
    state_app(engine)
}

fn state_app(engine: Engine) -> Router {
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

/// SSE 本文の data 行 (JSON) を順に返す。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn tools_json() -> Value {
    json!([{
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
    }, {
        "type": "function",
        "function": {"name": "run_tool", "description": "Run the tool"}
    }])
}

fn chat_body(stream: bool, tools: bool) -> Value {
    let mut body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": stream,
    });
    if tools {
        body["tools"] = tools_json();
    }
    body
}

// ---------------------------------------------------------------------------
// 非ストリーミング
// ---------------------------------------------------------------------------

#[tokio::test]
async fn non_streaming_single_tool_call() {
    let app = script_app(block_script("get_weather"));
    let (status, v) = post(&app, chat_body(false, true)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");

    let msg = &v["choices"][0]["message"];
    assert_eq!(msg["role"], "assistant");
    // tool_call block を除いた残りテキストが content に入る
    assert_eq!(msg["content"], "Checking.");
    let calls = msg["tool_calls"].as_array().expect("tool_calls が配列でない");
    assert_eq!(calls.len(), 1, "calls = {calls:?}");
    assert_eq!(calls[0]["type"], "function");
    let id = calls[0]["id"].as_str().unwrap();
    assert!(id.starts_with("call_"), "id = {id}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args = calls[0]["function"]["arguments"]
        .as_str()
        .expect("arguments が文字列でない");
    let parsed: Value = serde_json::from_str(args).expect("arguments が JSON でない");
    assert_eq!(parsed, json!({"city": "Tokyo"}));

    // usage は従来どおり入る
    let p = v["usage"]["prompt_tokens"].as_u64().unwrap();
    let c = v["usage"]["completion_tokens"].as_u64().unwrap();
    assert!(p > 0 && c > 0);
    assert_eq!(v["usage"]["total_tokens"].as_u64().unwrap(), p + c);
}

#[tokio::test]
async fn non_streaming_multiple_tool_calls() {
    let mut script = block_script("get_weather");
    script.pop(); // eos を一旦外す
    script.extend(block_script("run_tool"));
    let app = script_app(script);
    let (status, v) = post(&app, chat_body(false, true)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 2, "calls = {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(calls[1]["function"]["name"], "run_tool");
    assert_ne!(calls[0]["id"], calls[1]["id"]);
    let a0: Value = serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(a0, json!({"city": "Tokyo"}));
}

#[tokio::test]
async fn non_streaming_text_only_with_tools_declared() {
    // tools は宣言されているがモデルは tool_call を出さない → 従来の stop 応答
    let app = script_app(vec![tid("Checking."), tid("t2"), tid("eos")]);
    let (status, v) = post(&app, chat_body(false, true)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    let msg = &v["choices"][0]["message"];
    assert!(msg.get("tool_calls").is_none(), "message = {msg}");
    let text = msg["content"].as_str().unwrap();
    assert!(text.starts_with("Checking."), "content = {text:?}");
    assert!(text.ends_with("t2"), "content = {text:?}");
}

#[tokio::test]
async fn tools_not_declared_keeps_text_verbatim() {
    // tools 未指定なら抽出しない (本文は P793 と同一 = 生のまま)
    let app = plain_app();
    let (status, v) = post(&app, chat_body(false, false)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert!(v["choices"][0]["message"].get("tool_calls").is_none());
    assert!(v["choices"][0]["message"]["content"].as_str().unwrap().len() > 0);
}

// ---------------------------------------------------------------------------
// 履歴 (multi-turn)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_tool_history_is_accepted() {
    let app = script_app(vec![tid("t3"), tid("eos")]);
    let body = json!({
        "model": "denno-test",
        "messages": [
            {"role": "user", "content": "weather in tokyo?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}
            }]},
            {"role": "tool", "tool_call_id": "call_abc", "content": "sunny 25C"},
            {"role": "user", "content": "thanks"}
        ],
        "tools": tools_json(),
        "max_tokens": 16,
        "temperature": 0.0,
    });
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["content"], "t3");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn assistant_tool_call_arguments_must_be_json_object() {
    let app = plain_app();
    let body = json!({
        "model": "denno-test",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": "not json"}
            }]},
        ],
        "max_tokens": 8,
    });
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "messages[1].tool_calls[0]");
    assert!(v["error"]["message"].as_str().unwrap().contains("arguments"));
}

// ---------------------------------------------------------------------------
// ストリーミング
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_tool_calls_assembly() {
    let app = script_app(block_script("get_weather"));
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body(true, true).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");

    let chunks = sse_chunks(&text);
    // delta.content の結合 = tool_call block を除いた本文
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Checking.", "content = {content:?}");

    // delta.tool_calls: 最初の chunk に id/type/name、次の chunk に増分 arguments
    let mut starts = 0usize;
    let mut args = String::new();
    let mut name = String::new();
    let mut id = String::new();
    let mut index_seen: Vec<u64> = Vec::new();
    for c in &chunks {
        let Some(tcs) = c["choices"][0]["delta"]["tool_calls"].as_array() else { continue };
        for tc in tcs {
            index_seen.push(tc["index"].as_u64().unwrap());
            if let Some(n) = tc["function"]["name"].as_str() {
                starts += 1;
                name.push_str(n);
                id = tc["id"].as_str().unwrap_or_default().to_string();
                assert_eq!(tc["type"], "function");
                assert_eq!(tc["function"]["arguments"], "");
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                args.push_str(a);
            }
            // id は最初の chunk のみ (仕様)
            if tc["function"]["name"].is_null() {
                assert!(tc.get("id").is_none(), "id が 2 回載っている: {tc}");
            }
        }
    }
    assert_eq!(starts, 1, "tool_call start chunk が 1 個でない: {chunks:?}");
    assert_eq!(name, "get_weather");
    assert!(id.starts_with("call_"), "id = {id}");
    assert!(index_seen.iter().all(|&i| i == 0), "index = {index_seen:?}");
    let parsed: Value = serde_json::from_str(&args).expect("arguments が JSON でない");
    assert_eq!(parsed, json!({"city": "Tokyo"}));

    // finish_reason は最後の choices 付き chunk に tool_calls
    let last = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    // [DONE] の前に usage は出さない (include_usage 未指定)
    assert!(chunks.iter().all(|c| c.get("usage").is_none()));
}

#[tokio::test]
async fn streaming_text_only_with_tools_declared() {
    let app = script_app(vec![tid("Checking."), tid("t2"), tid("eos")]);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(chat_body(true, true).to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let chunks = sse_chunks(&text);
    let content: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(content, "Checking.t2");
    assert!(chunks
        .iter()
        .all(|c| c["choices"][0]["delta"].get("tool_calls").is_none()));
    let last = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

// ---------------------------------------------------------------------------
// tool_choice / tools 検証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_choice_required_forces_tool_call() {
    let app = script_app(block_script("get_weather"));
    let mut body = chat_body(false, true);
    body["tool_choice"] = json!("required");
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

#[tokio::test]
async fn tool_choice_named_function_is_accepted() {
    let app = script_app(block_script("get_weather"));
    let mut body = chat_body(false, true);
    body["tool_choice"] = json!({"type": "function", "function": {"name": "get_weather"}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
}

#[tokio::test]
async fn tool_choice_undefined_name_is_rejected() {
    let app = plain_app();
    let mut body = chat_body(false, true);
    body["tool_choice"] = json!({"type": "function", "function": {"name": "nope"}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "tool_choice");
    assert!(v["error"]["message"].as_str().unwrap().contains("nope"));
}

#[tokio::test]
async fn tool_choice_without_tools_is_rejected() {
    let app = plain_app();
    let mut body = chat_body(false, false);
    body["tool_choice"] = json!("auto");
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "tool_choice");
}

#[tokio::test]
async fn tool_choice_bogus_value_is_rejected() {
    let app = plain_app();
    let mut body = chat_body(false, true);
    body["tool_choice"] = json!("sometimes");
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "tool_choice");
}

#[tokio::test]
async fn invalid_tool_definition_is_rejected() {
    let app = plain_app();
    // name 空
    let mut body = chat_body(false, false);
    body["tools"] = json!([{"type": "function", "function": {"name": ""}}]);
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");

    // parameters がオブジェクトでない
    let mut body = chat_body(false, false);
    body["tools"] = json!([{
        "type": "function",
        "function": {"name": "f", "parameters": [1, 2]}
    }]);
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "tools[0].function.parameters");
}

// ---------------------------------------------------------------------------
// response_format
// ---------------------------------------------------------------------------

#[tokio::test]
async fn response_format_text_and_json_object_are_accepted() {
    let app = plain_app();
    for (rf, ok) in [(json!({"type": "text"}), true), (json!({"type": "json_object"}), true)] {
        let mut body = chat_body(false, false);
        body["response_format"] = rf;
        let (status, v) = post(&app, body).await;
        assert_eq!(status, StatusCode::OK, "body = {v}");
        assert_eq!(v["object"], "chat.completion");
        let _ = ok;
    }
}

#[tokio::test]
async fn response_format_unsupported_type_is_rejected() {
    let app = plain_app();
    let mut body = chat_body(false, false);
    body["response_format"] = json!({"type": "json_schema"});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "response_format.type");
}
