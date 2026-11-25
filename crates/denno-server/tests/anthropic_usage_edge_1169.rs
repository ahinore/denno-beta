//! Task 1169: Anthropic `/v1/messages` の stop_reason 別 usage 会計の境界テスト。
//!
//! `anthropic_parity_976.rs` は「非ストリーミングと SSE が同じ値を返すか」を
//! 見るパリティテスト。本ファイルはその補完で、**fixture が決定的に決める
//! 具体値**を固定する (exact accounting):
//!
//! - end_turn / max_tokens (0・1・2 境界) / stop_sequence / tool_use の
//!   `usage.input_tokens` / `output_tokens` / `cache_read_input_tokens` /
//!   `cache_creation_input_tokens` の実数
//! - `/v1/messages/count_tokens` と生成応答 `usage.input_tokens` の一致
//!   (同一 `prepare_chat` 経路 = 契約の一貫性)
//! - SSE の `message_start` (input/cache, output=0) と `message_delta`
//!   (output のみ) の分割が Anthropic の意味論どおりであること
//!
//! fixture (dummy_engine / dummy トークナイザ) の決定性:
//! - プロンプト `t0` + 既定 system の描画結果は 11 トークン
//! - greedy 生成は `t5 t6 t7 t8 t9 t10 t11` の 7 トークン + EOS で end_turn
//!   (`output_tokens` は EOS を含む 8)
//! - `stop_sequences: ["t7"]` は 3 トークン生成 (`t5 t6 t7`) で打ち切られ、
//!   テキストは `t5 t6 ` になる (終端トークンは output_tokens に計上されるが
//!   テキストには含まれない)
//! - tool script は block 15 トークン + EOS = output_tokens 16
//!
//! したがって本ファイルの数値はすべて固定値で、環境差は入らない (GPU 不要)。

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

/// ScriptModel + tool トークナイザで app を作る。
fn script_app(script: Vec<u32>) -> Router {
    let tok = tool_tokenizer_with(&[]).unwrap();
    let engine = Engine::from_parts(
        Box::new(ScriptModel::new(script, tid("t0"))),
        tok,
        "denno-test",
        None,
    );
    app_from_engine(engine)
}

fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

async fn post_raw(app: &Router, uri: &str, body: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// `data:` 行の JSON を順に返す (ping は除く)。
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
    let (status, text) = post_raw(app, "/v1/messages", &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    serde_json::from_str(&text).expect("JSON 応答でない")
}

async fn stream(app: &Router, mut body: Value) -> Vec<Value> {
    body["stream"] = json!(true);
    let (status, text) = post_raw(app, "/v1/messages", &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    sse_events(&text)
}

async fn count_tokens(app: &Router, mut body: Value) -> u64 {
    // count_tokens は stream / max_tokens を参照しない (同じ正規化だけを使う)
    body.as_object_mut().unwrap().remove("stream");
    let (status, text) = post_raw(app, "/v1/messages/count_tokens", &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let v: Value = serde_json::from_str(&text).expect("JSON 応答でない");
    v["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が数値でない: {v}"))
}

fn find<'a>(events: &'a [Value], ty: &str) -> &'a Value {
    events
        .iter()
        .find(|e| e["type"] == ty)
        .unwrap_or_else(|| panic!("{ty} イベントが無い: {events:?}"))
}

/// SSE の usage は message_start (input/cache) + message_delta (output)。
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

fn text_delta_count(events: &[Value]) -> usize {
    events
        .iter()
        .filter(|e| e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta")
        .count()
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

/// 完全な tool_call block (XML 形式・引数 1 個) + EOS を吐くスクリプト。
fn tool_script() -> Vec<u32> {
    [
        "Checking.",
        "<tool_call>",
        "Ċ",
        "<function=",
        "get_weather",
        ">",
        "Ċ",
        "<parameter=",
        "city",
        ">",
        "Tokyo",
        "</parameter>",
        "Ċ",
        "</function>",
        "</tool_call>",
        "eos",
    ]
    .iter()
    .map(|p| tid(p))
    .collect()
}

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    })
}

/// fixture の基準値: プロンプト 11 トークン、greedy 7 トークン + EOS。
const BASE_INPUT_TOKENS: u64 = 11;
const BASE_TEXT: &str = "t5 t6 t7 t8 t9 t10 t11";
/// EOS を含む生成トークン数 (end_turn)
const BASE_OUTPUT_TOKENS: u64 = 8;

fn assert_usage_exact(usage: &Value, input: u64, output: u64, read: u64, creation: u64) {
    assert_eq!(usage["input_tokens"], input, "usage = {usage}");
    assert_eq!(usage["output_tokens"], output, "usage = {usage}");
    assert_eq!(usage["cache_read_input_tokens"], read, "usage = {usage}");
    assert_eq!(
        usage["cache_creation_input_tokens"], creation,
        "usage = {usage}"
    );
    // Anthropic の契約: input は read + creation に分解される
    assert_eq!(read + creation, input, "usage = {usage}");
}

// ---------------------------------------------------------------------------
// end_turn: 具体値
// ---------------------------------------------------------------------------

#[tokio::test]
async fn end_turn_nonstream_exact_counts() {
    let body = nonstream(&dummy_app(), base_body()).await;

    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["stop_sequence"], Value::Null);
    assert_eq!(body["content"].as_array().unwrap().len(), 1);
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], BASE_TEXT);
    assert_usage_exact(
        &body["usage"],
        BASE_INPUT_TOKENS,
        BASE_OUTPUT_TOKENS,
        0,
        BASE_INPUT_TOKENS,
    );
}

#[tokio::test]
async fn end_turn_sse_exact_counts_and_split() {
    let ev = stream(&dummy_app(), base_body()).await;
    let start = find(&ev, "message_start");
    let delta = find(&ev, "message_delta");

    // message_start: input/cache だけ。output は 0 固定。
    assert_usage_exact(
        &start["message"]["usage"],
        BASE_INPUT_TOKENS,
        0,
        0,
        BASE_INPUT_TOKENS,
    );
    // message_delta: output だけを載せる (input/cache は再送しない)。
    let usage = delta["usage"].as_object().expect("usage が object でない");
    assert_eq!(usage.len(), 1, "delta usage = {delta}");
    assert_eq!(usage["output_tokens"], BASE_OUTPUT_TOKENS);

    assert_eq!(delta["delta"]["stop_reason"], "end_turn");
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
    assert_eq!(stream_text(&ev), BASE_TEXT);
    // 7 text トークン + EOS を 8 として計上する
    assert_eq!(text_delta_count(&ev), 7);

    assert_usage_exact(
        &merged_stream_usage(&ev),
        BASE_INPUT_TOKENS,
        BASE_OUTPUT_TOKENS,
        0,
        BASE_INPUT_TOKENS,
    );
}

#[tokio::test]
async fn count_tokens_matches_usage_input_tokens() {
    let app = dummy_app();
    let body = base_body();
    assert_eq!(count_tokens(&app, body.clone()).await, BASE_INPUT_TOKENS);
    let ns = nonstream(&app, body.clone()).await;
    assert_eq!(ns["usage"]["input_tokens"], BASE_INPUT_TOKENS);

    // max_tokens / stream は input_tokens に影響しない (同じ正規化経路)。
    let mut tuned = body.clone();
    tuned["max_tokens"] = json!(1);
    tuned["stream"] = json!(true);
    assert_eq!(count_tokens(&app, tuned).await, BASE_INPUT_TOKENS);
}

// ---------------------------------------------------------------------------
// max_tokens: 0 / 1 / 2 境界
// ---------------------------------------------------------------------------

#[tokio::test]
async fn max_tokens_zero_generates_nothing() {
    let mut body = base_body();
    body["max_tokens"] = json!(0);

    let ns = nonstream(&dummy_app(), body.clone()).await;
    assert_eq!(ns["stop_reason"], "max_tokens", "ns = {ns}");
    assert_eq!(ns["content"].as_array().unwrap().len(), 1);
    assert_eq!(ns["content"][0]["text"], "");
    assert_usage_exact(&ns["usage"], BASE_INPUT_TOKENS, 0, 0, BASE_INPUT_TOKENS);

    let ev = stream(&dummy_app(), body).await;
    let delta = find(&ev, "message_delta");
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(delta["usage"]["output_tokens"], 0);
    assert_eq!(text_delta_count(&ev), 0, "ev = {ev:?}");
    assert_eq!(stream_text(&ev), "");
    assert_usage_exact(&merged_stream_usage(&ev), BASE_INPUT_TOKENS, 0, 0, BASE_INPUT_TOKENS);
}

#[tokio::test]
async fn max_tokens_one_is_an_exact_boundary() {
    let mut body = base_body();
    body["max_tokens"] = json!(1);

    let ns = nonstream(&dummy_app(), body.clone()).await;
    assert_eq!(ns["stop_reason"], "max_tokens", "ns = {ns}");
    assert_eq!(ns["content"][0]["text"], "t5");
    assert_usage_exact(&ns["usage"], BASE_INPUT_TOKENS, 1, 0, BASE_INPUT_TOKENS);

    let ev = stream(&dummy_app(), body).await;
    let delta = find(&ev, "message_delta");
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(delta["usage"]["output_tokens"], 1);
    assert_eq!(stream_text(&ev), "t5");
    assert_usage_exact(&merged_stream_usage(&ev), BASE_INPUT_TOKENS, 1, 0, BASE_INPUT_TOKENS);
}

#[tokio::test]
async fn max_tokens_two_stops_mid_sentence() {
    let mut body = base_body();
    body["max_tokens"] = json!(2);

    let ns = nonstream(&dummy_app(), body.clone()).await;
    assert_eq!(ns["stop_reason"], "max_tokens", "ns = {ns}");
    assert_eq!(ns["content"][0]["text"], "t5 t6");
    assert_usage_exact(&ns["usage"], BASE_INPUT_TOKENS, 2, 0, BASE_INPUT_TOKENS);

    let ev = stream(&dummy_app(), body).await;
    assert_eq!(find(&ev, "message_delta")["usage"]["output_tokens"], 2);
    assert_eq!(stream_text(&ev), "t5 t6");
    assert_usage_exact(&merged_stream_usage(&ev), BASE_INPUT_TOKENS, 2, 0, BASE_INPUT_TOKENS);
}

// ---------------------------------------------------------------------------
// stop_sequence: 終端トークンの計上規則
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_sequence_counts_terminator_token_but_truncates_text() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t7"]);

    // 生成は t5 / t6 / t7 の 3 トークン。テキストは t7 の手前で切れる。
    let ns = nonstream(&dummy_app(), body.clone()).await;
    assert_eq!(ns["stop_reason"], "stop_sequence", "ns = {ns}");
    assert_eq!(ns["stop_sequence"], "t7");
    assert_eq!(ns["content"][0]["text"], "t5 t6 ");
    assert_usage_exact(&ns["usage"], BASE_INPUT_TOKENS, 3, 0, BASE_INPUT_TOKENS);

    let ev = stream(&dummy_app(), body.clone()).await;
    let delta = find(&ev, "message_delta");
    assert_eq!(delta["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(delta["delta"]["stop_sequence"], "t7");
    assert_eq!(delta["usage"]["output_tokens"], 3);
    // SSE は stop sequence の prefix になり得る末尾 1 文字 (空白) を holdback
    // したまま打ち切るため、非ストリーミングの "t5 t6 " より 1 文字短い。
    // usage は同一 (output_tokens = 3) で、テキストに stop sequence は入らない。
    assert_eq!(stream_text(&ev), "t5 t6");
    assert!(!stream_text(&ev).contains("t7"));
    assert_usage_exact(&merged_stream_usage(&ev), BASE_INPUT_TOKENS, 3, 0, BASE_INPUT_TOKENS);

    // stop_sequences はプロンプトを変えない = input_tokens は基準値のまま。
    assert_eq!(count_tokens(&dummy_app(), body).await, BASE_INPUT_TOKENS);
}

// ---------------------------------------------------------------------------
// tool_use: block 抽出時の usage
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_use_nonstream_exact_counts() {
    let app = script_app(tool_script());
    let mut body = base_body();
    body["max_tokens"] = json!(64);
    body["tools"] = tools_json();

    // tool 付きプロンプトの input_tokens は count_tokens と一致する
    let expected_input = count_tokens(&app, body.clone()).await;

    let ns = nonstream(&app, body).await;
    assert_eq!(ns["stop_reason"], "tool_use", "ns = {ns}");
    assert_eq!(ns["stop_sequence"], Value::Null);
    let blocks = ns["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 2, "ns = {ns}");
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "Checking.");
    assert_eq!(blocks[1]["type"], "tool_use");
    assert_eq!(blocks[1]["name"], "get_weather");
    assert_eq!(blocks[1]["input"]["city"], "Tokyo", "ns = {ns}");

    // block 15 トークン + EOS
    assert_usage_exact(&ns["usage"], expected_input, 16, 0, expected_input);
}

#[tokio::test]
async fn tool_use_sse_exact_counts() {
    let mut body = base_body();
    body["max_tokens"] = json!(64);
    body["tools"] = tools_json();

    let ns = nonstream(&script_app(tool_script()), body.clone()).await;
    let ev = stream(&script_app(tool_script()), body).await;
    let delta = find(&ev, "message_delta");

    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
    assert_eq!(delta["usage"]["output_tokens"], 16);
    assert_eq!(stream_text(&ev), "Checking.");
    let tool_starts = ev
        .iter()
        .filter(|e| {
            e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use"
        })
        .count();
    assert_eq!(tool_starts, 1, "ev = {ev:?}");

    let merged = merged_stream_usage(&ev);
    assert_eq!(merged["input_tokens"], ns["usage"]["input_tokens"]);
    assert_eq!(merged["output_tokens"], 16);
    assert_eq!(merged["cache_read_input_tokens"], 0);
    assert_eq!(
        merged["cache_creation_input_tokens"],
        merged["input_tokens"]
    );
}

/// max_tokens が block の途中で尽きた場合は tool_use block を出さない
/// (不完全な block を tool_use として返さない = stop_reason は max_tokens)。
#[tokio::test]
async fn tool_use_truncated_by_max_tokens_is_not_tool_use() {
    let mut body = base_body();
    body["max_tokens"] = json!(5);
    body["tools"] = tools_json();

    let ns = nonstream(&script_app(tool_script()), body.clone()).await;
    assert_eq!(ns["stop_reason"], "max_tokens", "ns = {ns}");
    assert_eq!(ns["stop_sequence"], Value::Null);
    let blocks = ns["content"].as_array().unwrap();
    assert_eq!(blocks.len(), 1, "ns = {ns}");
    assert!(blocks.iter().all(|b| b["type"] != "tool_use"), "ns = {ns}");
    assert_usage_exact(
        &ns["usage"],
        ns["usage"]["input_tokens"].as_u64().unwrap(),
        5,
        0,
        ns["usage"]["input_tokens"].as_u64().unwrap(),
    );

    let ev = stream(&script_app(tool_script()), body).await;
    let delta = find(&ev, "message_delta");
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(delta["usage"]["output_tokens"], 5);
    assert_eq!(
        ev.iter()
            .filter(|e| {
                e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use"
            })
            .count(),
        0,
        "ev = {ev:?}"
    );
}

// ---------------------------------------------------------------------------
// キャッシュ会計: 冷 / 温 の内訳と不変条件
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_accounting_splits_input_exactly() {
    let app = dummy_app();
    let mut body = base_body();
    body["system"] = json!("You are a test.");
    body["max_tokens"] = json!(8);

    // 1 回目 (冷): 全量 prefill
    let cold = nonstream(&app, body.clone()).await;
    let input = cold["usage"]["input_tokens"].as_u64().unwrap();
    assert_eq!(input, 16, "cold = {cold}");
    assert_usage_exact(&cold["usage"], input, BASE_OUTPUT_TOKENS, 0, input);

    // 2 回目 (温): 先頭 15 トークンを再利用し、1 トークンだけ新規 prefill。
    let warm = nonstream(&app, body.clone()).await;
    assert_usage_exact(&warm["usage"], input, BASE_OUTPUT_TOKENS, 15, 1);

    // 3 回目も同じ内訳 (再現性)
    let warm2 = nonstream(&app, body.clone()).await;
    assert_usage_exact(&warm2["usage"], input, BASE_OUTPUT_TOKENS, 15, 1);

    // キャッシュは input_tokens を増やさない (会計の不変条件)
    assert_eq!(warm["usage"]["input_tokens"], cold["usage"]["input_tokens"]);
}

#[tokio::test]
async fn cache_accounting_in_sse_message_start() {
    let app = dummy_app();
    let mut body = base_body();
    body["system"] = json!("You are a test.");
    body["max_tokens"] = json!(8);

    // 冷 1 回 + 温 1 回で prefix cache を安定させる
    let _ = nonstream(&app, body.clone()).await;
    let _ = nonstream(&app, body.clone()).await;

    let ev = stream(&app, body).await;
    let start = &find(&ev, "message_start")["message"]["usage"];
    assert_usage_exact(start, 16, 0, 15, 1);
    assert_eq!(find(&ev, "message_delta")["usage"]["output_tokens"], BASE_OUTPUT_TOKENS);
    assert_usage_exact(&merged_stream_usage(&ev), 16, BASE_OUTPUT_TOKENS, 15, 1);
}
