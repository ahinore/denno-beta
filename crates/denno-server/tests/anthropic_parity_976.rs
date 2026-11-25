//! Task 976: Anthropic `/v1/messages` の stop_reason / usage パリティテスト
//! (非ストリーミング vs SSE)。
//!
//! fixture は `tests/anthropic_edge_974.rs` / `tests/tools.rs` と同一
//! (dummy モデル + dummy トークナイザ / ScriptModel + tool トークナイザ。GPU 不要)。
//!
//! 対象:
//! - stop_reason: end_turn / max_tokens / stop_sequence (stop_sequence 文字列入り) / tool_use
//! - usage: input_tokens / output_tokens / cache_read_input_tokens
//!   (SSE は message_start に input/cache、message_delta に output が分かれるため
//!    Anthropic の意味論どおり畳んで比較する。キャッシュ再利用ありの連続リクエストも含む)
//!
//! 注意: ScriptModel は 1 セッション専用 (forward 呼び出し回数 = スクリプト位置) なので、
//! パリティ比較では非ストリーミング用とストリーミング用に **別々の app** を作る。
//! dummy モデルも同じ理由 (先に走らせた側が prefix cache を温める) で分ける。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_engine, tool_tokenizer_with, ScriptModel, TOOL_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn app_from_engine(engine: Engine) -> Router {
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

fn dummy_app() -> Router {
    app_from_engine(dummy_engine("denno-test", None).unwrap())
}

/// ScriptModel + tool トークナイザで app を作る。`extra` は追加語彙。
fn script_app(script: Vec<u32>, extra: &[&str]) -> Router {
    let tok = tool_tokenizer_with(extra).unwrap();
    let engine = Engine::from_parts(
        Box::new(ScriptModel::new(script, tid("t0"))),
        tok,
        "denno-test",
        None,
    );
    app_from_engine(engine)
}

async fn post_raw(app: &Router, body: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// `data:` 行の JSON を順に返す (keep-alive の ping は除く)。
fn sse_events(text: &str) -> Vec<Value> {
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            let v: Value = serde_json::from_str(data).expect("SSE data が JSON ではない");
            if v["type"] != "ping" {
                out.push(v);
            }
        }
    }
    out
}

async fn nonstream(app: &Router, mut body: Value) -> Value {
    body["stream"] = json!(false);
    let (status, text) = post_raw(app, &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    serde_json::from_str(&text).expect("JSON 応答でない")
}

async fn stream(app: &Router, mut body: Value) -> Vec<Value> {
    body["stream"] = json!(true);
    let (status, text) = post_raw(app, &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    sse_events(&text)
}

fn find<'a>(events: &'a [Value], ty: &str) -> &'a Value {
    events
        .iter()
        .find(|e| e["type"] == ty)
        .unwrap_or_else(|| panic!("{ty} イベントが無い: {events:?}"))
}

/// SSE の usage は message_start (input/cache) + message_delta (output) に分かれる。
/// Anthropic の意味論どおり 1 個の usage に畳んでから非ストリーミングと比べる。
fn merged_stream_usage(events: &[Value]) -> Value {
    let mut merged = find(events, "message_start")["message"]["usage"].clone();
    for (k, v) in find(events, "message_delta")["usage"].as_object().unwrap() {
        merged[k] = v.clone();
    }
    merged
}

fn stream_text(events: &[Value]) -> String {
    let mut out = String::new();
    for e in events {
        if e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta" {
            out.push_str(e["delta"]["text"].as_str().unwrap());
        }
    }
    out
}

fn tools_json() -> Value {
    json!([{
        "name": "get_weather",
        "description": "Get current weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }])
}

fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// 完全な tool_call block を吐くスクリプト (`</tool_call>` 付き = 抽出対象)
fn block_script(name: &str, tail: &[u32]) -> Vec<u32> {
    let head: Vec<u32> = [
        "Checking.", "<tool_call>", "Ċ", "<function=", name, ">", "city", "Tokyo",
        "</parameter>", "</function>", "</tool_call>",
    ]
    .iter()
    .map(|p| tid(p))
    .collect();
    let mut script = head;
    script.extend_from_slice(tail);
    script
}

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    })
}

// ---------------------------------------------------------------------------
// stop_reason parity
// ---------------------------------------------------------------------------

/// message_start は stop_reason / stop_sequence を null で載せる (Anthropic 互換)。
#[tokio::test]
async fn message_start_has_null_stop_reason() {
    let app = dummy_app();
    let ev = stream(&app, base_body()).await;
    let msg = &find(&ev, "message_start")["message"];
    assert_eq!(msg["stop_reason"], Value::Null);
    assert_eq!(msg["stop_sequence"], Value::Null);
}

#[tokio::test]
async fn end_turn_parity() {
    let ns = nonstream(&dummy_app(), base_body()).await;
    let ev = stream(&dummy_app(), base_body()).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(ns["stop_reason"], "end_turn");
    assert_eq!(delta["delta"]["stop_reason"], "end_turn");
    assert_eq!(ns["stop_sequence"], Value::Null);
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
}

#[tokio::test]
async fn max_tokens_parity() {
    let mut body = base_body();
    body["max_tokens"] = json!(3);
    let ns = nonstream(&dummy_app(), body.clone()).await;
    let ev = stream(&dummy_app(), body).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(ns["stop_reason"], "max_tokens", "ns = {ns}");
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(ns["stop_sequence"], Value::Null);
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
    assert_eq!(ns["usage"]["output_tokens"], 3);
    assert_eq!(delta["usage"]["output_tokens"], 3);
    assert_eq!(stream_text(&ev), ns["content"][0]["text"]);
}

#[tokio::test]
async fn stop_sequence_parity_populates_string() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t7"]);
    let ns = nonstream(&dummy_app(), body.clone()).await;
    let ev = stream(&dummy_app(), body).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(ns["stop_reason"], "stop_sequence", "ns = {ns}");
    assert_eq!(delta["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(ns["stop_sequence"], "t7");
    assert_eq!(delta["delta"]["stop_sequence"], "t7");
    // usage は打ち切り位置に依存せず一致する
    assert_eq!(delta["usage"]["output_tokens"], ns["usage"]["output_tokens"]);
}

#[tokio::test]
async fn tool_use_parity() {
    let script = block_script("get_weather", &[tid("eos")]);
    let mut body = base_body();
    body["max_tokens"] = json!(64);
    body["tools"] = tools_json();

    let ns = nonstream(&script_app(script.clone(), &[]), body.clone()).await;
    let ev = stream(&script_app(script, &[]), body).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(ns["stop_reason"], "tool_use", "ns = {ns}");
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(ns["stop_sequence"], Value::Null);
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);

    // 同一テキスト・同一 tool_use block (name / input)
    assert_eq!(stream_text(&ev), ns["content"][0]["text"]);
    let tool_ns = ns["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|b| b["type"] == "tool_use")
        .expect("非ストリーミングに tool_use block が無い");
    let tool_ev = ev
        .iter()
        .find(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .expect("ストリーミングに tool_use block が無い");
    assert_eq!(tool_ns["name"], tool_ev["content_block"]["name"]);
    let input_ev: Value = ev
        .iter()
        .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "input_json_delta")
        .map(|e| e["delta"]["partial_json"].as_str().unwrap().to_string())
        .collect::<Vec<_>>()
        .join("")
        .parse()
        .expect("input_json_delta が JSON でない");
    assert_eq!(tool_ns["input"], input_ev);
}

#[tokio::test]
async fn multiple_tool_use_parity() {
    let mut script = block_script("get_weather", &[]);
    script.extend_from_slice(&block_script("get_weather", &[tid("eos")]));
    let mut body = base_body();
    body["max_tokens"] = json!(128);
    body["tools"] = tools_json();

    let ns = nonstream(&script_app(script.clone(), &[]), body.clone()).await;
    let ev = stream(&script_app(script, &[]), body).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(ns["stop_reason"], "tool_use", "ns = {ns}");
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    let ns_tools: Vec<&Value> = ns["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .collect();
    let ev_tools: Vec<&Value> = ev
        .iter()
        .filter(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .map(|e| &e["content_block"])
        .collect();
    assert_eq!(ns_tools.len(), 2, "ns = {ns}");
    assert_eq!(ev_tools.len(), 2, "ev = {ev:?}");
    for (a, b) in ns_tools.iter().zip(ev_tools.iter()) {
        assert_eq!(a["name"], b["name"]);
    }
}

/// tool_use と stop_sequence が同時に成立するケース (block の後に通常テキストを
/// 挟んで stop させる)。Anthropic は stop_reason = tool_use のとき
/// stop_sequence を null にする — 非ストリーミングの message_json と同じ規則を
/// SSE の message_delta でも満たすこと。
#[tokio::test]
async fn tool_use_with_stop_sequence_nulls_stop_sequence() {
    // tool_tokenizer_with は追加語彙を 40 から順に割り当てる
    const HELLO: u32 = 40;
    const END: u32 = 41;
    // block の最終行を `\n` で閉じてから stop 語を含む行を続ける
    // (engine の行頭 holdback が `<function=` 行を保持するため、改行が要る)
    let script = block_script("get_weather", &[tid("Ċ"), HELLO, END, tid("eos")]);
    let mut body = base_body();
    body["max_tokens"] = json!(64);
    body["stop_sequences"] = json!(["END"]);
    body["tools"] = tools_json();

    let ns = nonstream(&script_app(script.clone(), &["hello", "END"]), body.clone()).await;
    let ev = stream(&script_app(script, &["hello", "END"]), body).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(ns["stop_reason"], "tool_use", "ns = {ns}");
    assert_eq!(delta["delta"]["stop_reason"], "tool_use", "ev = {ev:?}");
    assert_eq!(ns["stop_sequence"], Value::Null, "ns = {ns}");
    assert_eq!(
        delta["delta"]["stop_sequence"], Value::Null,
        "stream delta = {delta}"
    );

    let names_ns: Vec<&Value> = ns["content"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|b| b["type"] == "tool_use")
        .map(|b| &b["name"])
        .collect();
    let names_ev: Vec<&Value> = ev
        .iter()
        .filter(|e| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .map(|e| &e["content_block"]["name"])
        .collect();
    assert_eq!(names_ns, vec![&json!("get_weather")], "ns = {ns}");
    assert_eq!(names_ev, names_ns, "ev = {ev:?}");

    let merged = merged_stream_usage(&ev);
    assert_eq!(merged["input_tokens"], ns["usage"]["input_tokens"]);
    assert_eq!(merged["output_tokens"], ns["usage"]["output_tokens"]);
    assert_eq!(merged["cache_read_input_tokens"], ns["usage"]["cache_read_input_tokens"]);
}

// ---------------------------------------------------------------------------
// usage parity
// ---------------------------------------------------------------------------

#[tokio::test]
async fn usage_parity_end_turn() {
    let mut body = base_body();
    body["system"] = json!("You are a test.");
    let ns = nonstream(&dummy_app(), body.clone()).await;
    let ev = stream(&dummy_app(), body).await;
    let merged = merged_stream_usage(&ev);

    assert!(ns["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(merged["input_tokens"], ns["usage"]["input_tokens"]);
    assert_eq!(merged["output_tokens"], ns["usage"]["output_tokens"]);
    assert_eq!(merged["cache_read_input_tokens"], ns["usage"]["cache_read_input_tokens"]);
    assert_eq!(
        merged["cache_creation_input_tokens"],
        ns["usage"]["cache_creation_input_tokens"]
    );
    // 冷えた状態は全量 prefill
    assert_eq!(merged["cache_read_input_tokens"], 0);
    assert_eq!(
        merged["cache_read_input_tokens"].as_u64().unwrap()
            + merged["cache_creation_input_tokens"].as_u64().unwrap(),
        merged["input_tokens"].as_u64().unwrap()
    );
}

/// 同一プロンプト連続: 2 回目以降は prefix 再利用 (cache_read > 0)。
/// 非ストリーミング / ストリーミングで同じ usage を返すこと。
#[tokio::test]
async fn usage_parity_prefix_cache_reuse() {
    let mut body = base_body();
    body["max_tokens"] = json!(8);
    body["system"] = json!("You are a test.");
    body["messages"] = json!([{"role": "user", "content": "t0 repeat"}]);

    // 1 回目で prefill してキャッシュを温める
    let first = nonstream(&dummy_app(), body.clone()).await;
    // 同じ app で温め直してから測る (app を分けるとキャッシュが無い)
    let warm_app = dummy_app();
    let warm = nonstream(&warm_app, body.clone()).await;
    assert_eq!(warm["usage"]["cache_read_input_tokens"], 0);
    assert_eq!(first["usage"]["cache_read_input_tokens"], 0);

    let ns = nonstream(&warm_app, body.clone()).await;
    let ev = stream(&warm_app, body).await;
    let merged = merged_stream_usage(&ev);

    assert!(
        ns["usage"]["cache_read_input_tokens"].as_u64().unwrap() > 0,
        "prefix 再利用が起きていない: {ns}"
    );
    assert_eq!(merged["cache_read_input_tokens"], ns["usage"]["cache_read_input_tokens"]);
    assert_eq!(merged["input_tokens"], ns["usage"]["input_tokens"]);
    assert_eq!(merged["output_tokens"], ns["usage"]["output_tokens"]);
    assert_eq!(
        merged["cache_creation_input_tokens"],
        ns["usage"]["cache_creation_input_tokens"]
    );
}

#[tokio::test]
async fn usage_parity_tool_use() {
    let script = block_script("get_weather", &[tid("eos")]);
    let mut body = base_body();
    body["max_tokens"] = json!(64);
    body["tools"] = tools_json();

    let ns = nonstream(&script_app(script.clone(), &[]), body.clone()).await;
    let ev = stream(&script_app(script, &[]), body).await;
    let merged = merged_stream_usage(&ev);

    assert_eq!(merged["input_tokens"], ns["usage"]["input_tokens"]);
    assert_eq!(merged["output_tokens"], ns["usage"]["output_tokens"]);
    assert_eq!(merged["cache_read_input_tokens"], ns["usage"]["cache_read_input_tokens"]);
    assert_eq!(
        merged["cache_creation_input_tokens"],
        ns["usage"]["cache_creation_input_tokens"]
    );
}
