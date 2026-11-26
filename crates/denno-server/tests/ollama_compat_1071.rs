//! task 1071: Ollama 互換 `/api/chat` / `/api/generate` の
//! 任意フィールド (keep_alive / format / think / tools) の互換テスト。
//!
//! 人気クライアント (Open WebUI / Continue / Cline) は Ollama へ
//! こちらが解釈しないフィールドを常に送る。ここで固定する契約は:
//!
//! - `keep_alive` (文字列 / 整数 / null / object) を受理し、生成結果を変えない。
//! - `format` (`"json"` / JSON Schema オブジェクト / 未知の値) を受理する
//!   (未知の値でも 400 にしない)。
//! - `think` (bool / `"low"|"medium"|"high"` / 未知の文字列) を受理する。
//! - メッセージ側の `images` / `tool_calls` / `thinking` などの未知キーを受理する。
//! - `tools` を渡すと、モデル出力の `<tool_call>` block が Ollama 形式の
//!   `message.tool_calls` (arguments は **オブジェクト**) へ変換され、
//!   ストリームでも block の断片を chunk に漏らさない。
//! - `tools` を渡さなければ従来どおり本文に残る (分岐が発火した証拠)。
//! - 形の不正な `tools` 要素 / 配列でない `tools` は読み飛ばす (400 にしない)。
//!
//! 実装: `crates/denno-server/src/ollama.rs`
//! (`format_hint` / `thinking_from` / `ollama_tool_defs` / `NdjsonStream`)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, tool_tokenizer, DummyModel, ScriptModel, TOOL_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "denno-test";

/// `TOOL_VOCAB` の piece → id。
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

fn state_app(engine: Engine) -> Router {
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: MODEL.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

/// 通常のダミーモデル (tool call を出さない)。
fn plain_app() -> Router {
    state_app(Engine::from_parts(
        Box::new(DummyModel),
        dummy_tokenizer().unwrap(),
        MODEL,
        None,
    ))
}

/// スクリプト再生モデル (tool call を出す)。
fn script_app(script: Vec<u32>) -> Router {
    state_app(Engine::from_parts(
        Box::new(ScriptModel::new(script, tid("t1"))),
        tool_tokenizer().unwrap(),
        MODEL,
        None,
    ))
}

async fn post_uri(app: &Router, uri: &str, body: &Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// NDJSON 本文を行ごとの JSON にする。
fn ndjson(text: &str) -> Vec<Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str(l).unwrap_or_else(|e| panic!("NDJSON 行が JSON でない ({e}): {l:?}"))
        })
        .collect()
}

fn chat_body() -> Value {
    json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "options": {"temperature": 0, "num_predict": 64},
    })
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
    }])
}

// ---------------------------------------------------------------------------
// keep_alive
// ---------------------------------------------------------------------------

#[tokio::test]
async fn keep_alive_variants_are_accepted_and_do_not_change_output() {
    let app = plain_app();
    let baseline = post_uri(&app, "/api/chat", &chat_body()).await;
    assert_eq!(baseline.0, StatusCode::OK, "baseline = {}", baseline.1);
    let baseline_text = {
        let v: Value = serde_json::from_str(&baseline.1).unwrap();
        v["message"]["content"].as_str().unwrap_or_default().to_string()
    };

    let variants = [
        json!("5m"),
        json!("0"),
        json!("-1"),
        json!(0),
        json!(300),
        json!(-1),
        Value::Null,
        json!({"num": 1}),
    ];
    for ka in variants {
        let mut body = chat_body();
        body["keep_alive"] = ka.clone();
        let (status, text) = post_uri(&app, "/api/chat", &body).await;
        assert_eq!(status, StatusCode::OK, "keep_alive = {ka}: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["done"], true, "keep_alive = {ka}: {v}");
        assert_eq!(
            v["message"]["content"].as_str().unwrap_or_default(),
            baseline_text,
            "keep_alive = {ka} が生成結果を変えた"
        );
    }
}

// ---------------------------------------------------------------------------
// format / think / メッセージ側の未知キー
// ---------------------------------------------------------------------------

#[tokio::test]
async fn format_and_think_are_accepted_in_every_form() {
    let app = plain_app();
    // format: json / schema / 未知の値 (どれも 200)
    for fmt in [json!("json"), json!({"type": "object"}), json!("xml"), json!(7)] {
        let mut body = chat_body();
        body["format"] = fmt.clone();
        let (status, text) = post_uri(&app, "/api/chat", &body).await;
        assert_eq!(status, StatusCode::OK, "format = {fmt}: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["done"], true, "format = {fmt}: {v}");
    }
    // think: bool / 効力文字列 / 未知 (どれも 200)
    for think in [json!(true), json!(false), json!("high"), json!("off"), json!(3)] {
        let mut body = chat_body();
        body["think"] = think.clone();
        let (status, text) = post_uri(&app, "/api/chat", &body).await;
        assert_eq!(status, StatusCode::OK, "think = {think}: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["done"], true, "think = {think}: {v}");
    }
    // メッセージ側の未知キー (images / tool_calls / thinking) も 400 にしない
    let mut body = chat_body();
    body["messages"] = json!([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello", "images": ["aGVsbG8="]},
        {"role": "assistant", "content": "", "thinking": "hmm",
         "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}]},
        {"role": "user", "content": [{"type": "text", "text": "hi"},
                                     {"type": "image", "image": "aGVsbG8="}]}
    ]);
    body["keep_alive"] = json!("10m");
    body["format"] = json!("json");
    body["think"] = json!("low");
    let (status, text) = post_uri(&app, "/api/chat", &body).await;
    assert_eq!(status, StatusCode::OK, "extras: {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["done"], true, "extras: {v}");
}

// ---------------------------------------------------------------------------
// tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tools_turn_tool_call_block_into_ollama_tool_calls() {
    let app = script_app(block_script("get_weather"));
    let mut body = chat_body();
    body["tools"] = tools_json();
    let (status, text) = post_uri(&app, "/api/chat", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["done"], true, "v = {v}");
    assert_eq!(v["done_reason"], "stop");

    let msg = &v["message"];
    assert_eq!(msg["role"], "assistant");
    // block を除いた残り本文
    assert_eq!(msg["content"], "Checking.");
    let calls = msg["tool_calls"].as_array().expect("tool_calls が配列でない");
    assert_eq!(calls.len(), 1, "calls = {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    // Ollama の arguments はオブジェクト
    assert_eq!(calls[0]["function"]["arguments"], json!({"city": "Tokyo"}));
}

#[tokio::test]
async fn streaming_tools_emit_one_line_and_leak_no_block_fragments() {
    let app = script_app(block_script("get_weather"));
    let mut body = chat_body();
    body["stream"] = json!(true);
    body["tools"] = tools_json();
    let (status, text) = post_uri(&app, "/api/chat", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let lines = ndjson(&text);
    assert!(!lines.is_empty(), "NDJSON が空");

    let mut tool_lines = 0;
    for l in &lines {
        let content = l["message"]["content"].as_str().unwrap_or_default();
        assert!(
            !content.contains("<tool_call>") && !content.contains("</tool_call>"),
            "tool_call block の断片が chunk に漏れた: {l}"
        );
    }
    for l in &lines {
        if l["message"]["tool_calls"].is_array() {
            tool_lines += 1;
            assert_eq!(l["done"], false, "tool_calls は done=false の行: {l}");
            let calls = l["message"]["tool_calls"].as_array().unwrap();
            assert_eq!(calls.len(), 1, "calls = {calls:?}");
            assert_eq!(calls[0]["function"]["name"], "get_weather");
            assert_eq!(calls[0]["function"]["arguments"], json!({"city": "Tokyo"}));
            assert_eq!(l["message"]["content"], "Checking.");
        }
    }
    assert_eq!(tool_lines, 1, "tool_calls 行が 1 本でない: {lines:?}");
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true, "最終行が done=true でない: {last}");
    assert_eq!(last["done_reason"], "stop", "last = {last}");
}

#[tokio::test]
async fn without_tools_the_block_stays_in_text() {
    // 分岐の証拠: tools を渡さなければ抽出しない (従来動作)。
    let app = script_app(block_script("get_weather"));
    let (status, text) = post_uri(&app, "/api/chat", &chat_body()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    let content = v["message"]["content"].as_str().unwrap_or_default();
    assert!(content.contains("<tool_call>"), "content = {content:?}");
    assert!(v["message"]["tool_calls"].is_null(), "v = {v}");
}

#[tokio::test]
async fn malformed_tools_are_skipped_not_rejected() {
    let app = plain_app();
    let baseline = post_uri(&app, "/api/chat", &chat_body()).await;
    assert_eq!(baseline.0, StatusCode::OK);
    let baseline_text = {
        let v: Value = serde_json::from_str(&baseline.1).unwrap();
        v["message"]["content"].as_str().unwrap_or_default().to_string()
    };
    // 配列でない / 形が不正な要素だけの tools は「無し」と同じ扱い
    for tools in [
        json!("nope"),
        json!({"function": {"name": "x"}}),
        json!([]),
        json!([{}, {"function": {}}, {"type": "function"}, 5, "s"]),
    ] {
        let mut body = chat_body();
        body["tools"] = tools.clone();
        let (status, text) = post_uri(&app, "/api/chat", &body).await;
        assert_eq!(status, StatusCode::OK, "tools = {tools}: {text}");
        let v: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            v["message"]["content"].as_str().unwrap_or_default(),
            baseline_text,
            "tools = {tools} が生成結果を変えた"
        );
        assert!(v["message"]["tool_calls"].is_null(), "tools = {tools}: {v}");
    }
}

// ---------------------------------------------------------------------------
// /api/generate
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_accepts_benign_extra_fields() {
    let app = plain_app();
    let body = json!({
        "model": MODEL,
        "prompt": "hello",
        "system": "be brief",
        "stream": false,
        "keep_alive": "5m",
        "format": "json",
        "think": true,
        "suffix": "tail",
        "template": "{{ .Prompt }}",
        "options": {"temperature": 0, "num_predict": 64, "num_ctx": 4096},
    });
    let (status, text) = post_uri(&app, "/api/generate", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["done"], true, "v = {v}");
    assert!(v["response"].is_string(), "v = {v}");
}
