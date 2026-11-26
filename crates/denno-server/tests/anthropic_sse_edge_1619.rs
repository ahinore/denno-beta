//! Task 1619: Anthropic `/v1/messages` SSE の**イベント列契約**テスト。
//!
//! 既存ファイルとの役割分担:
//! - `anthropic_params_edge_1539.rs`: パラメータ境界と stop_reason ごとの順序不変量
//! - `anthropic_usage_edge_1169.rs`: stop_reason 別 usage の**具体値**
//! - 本ファイル: SSE フレーム列そのものの契約
//!   - `event:` 行と `data.type` の一致 (フレーム単位)
//!   - `message_start` → `content_block_start` → `content_block_delta`* →
//!     `content_block_stop` → `message_delta` → `message_stop` の**厳密な順序**
//!   - block `index` が 0 から始まり 1 ずつ単調増加すること
//!   - 各 delta / stop の `index` が現在 open 中の block を指すこと
//!   - `message_start` (input/cache) と `message_delta` (output) の usage 分割、
//!     `stop_reason` / `stop_sequence` のフィールド
//!
//! fixture は他の Anthropic テストと同一 (dummy / ScriptModel + tool トークナイザ)。
//! 生成は greedy 固定で GPU 不要・決定論的。

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

const MESSAGES: &str = "/v1/messages";

/// greedy 時の全文 (EOS の直前まで)
const GREEDY: &str = "t5 t6 t7 t8 t9 t10 t11";
/// プロンプト `t0` + 既定 system の input_tokens
const BASE_INPUT_TOKENS: u64 = 11;
/// EOS を含む生成トークン数
const BASE_OUTPUT_TOKENS: u64 = 8;

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

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    })
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
    }, {
        "name": "run_tool",
        "description": "Run a tool",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    }])
}

/// 完全な tool_call block + EOS を吐くスクリプト。
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

/// tool_call block 2 個 + EOS を吐くスクリプト (連続 block の index 検証用)。
fn two_tool_script() -> Vec<u32> {
    let mut script: Vec<&str> = vec!["Checking."];
    for name in ["get_weather", "run_tool"] {
        script.extend([
            "<tool_call>",
            "Ċ",
            "<function=",
            name,
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
        ]);
    }
    script.push("eos");
    script.iter().map(|p| tid(p)).collect()
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

async fn nonstream(app: &Router, mut body: Value) -> Value {
    body["stream"] = json!(false);
    let (status, text) = post_raw(app, MESSAGES, &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    serde_json::from_str(&text).expect("JSON 応答でない")
}

async fn count_tokens(app: &Router, mut body: Value) -> u64 {
    body.as_object_mut().unwrap().remove("stream");
    let (status, text) = post_raw(app, "/v1/messages/count_tokens", &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let v: Value = serde_json::from_str(&text).expect("JSON 応答でない");
    v["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が数値でない: {v}"))
}

/// SSE を `(event 行, data JSON)` の列として返す (keep-alive ping は落とす)。
///
/// 各 `data:` 行には直前の `event:` 行が対になることを要求する
/// (フレーム単位の契約検証)。
fn sse_frames(text: &str) -> Vec<(String, Value)> {
    let mut out = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        if let Some(e) = line.strip_prefix("event: ") {
            assert!(
                pending.is_none(),
                "event 行が連続している: pending = {pending:?}"
            );
            pending = Some(e.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            let v: Value = serde_json::from_str(data).expect("SSE data が JSON ではない");
            assert!(
                v["type"].is_string(),
                "data.type が文字列でない: {data}"
            );
            if v["type"] == "ping" {
                pending = None;
                continue;
            }
            let name = pending.take().expect("data 行に対応する event 行が無い");
            assert_eq!(
                name,
                v["type"].as_str().unwrap(),
                "event 行と data.type が不一致: {data}"
            );
            out.push((name, v));
        }
    }
    out
}

async fn stream(app: &Router, mut body: Value) -> Vec<(String, Value)> {
    body["stream"] = json!(true);
    let (status, text) = post_raw(app, MESSAGES, &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert!(!text.contains("event: error"), "error イベントがある: {text}");
    sse_frames(&text)
}

fn types(frames: &[(String, Value)]) -> Vec<&str> {
    frames.iter().map(|(t, _)| t.as_str()).collect()
}

/// 全テスト共通の外枠: 先頭 message_start / 末尾 message_stop / 各 1 回。
/// さらに `message_delta` が最後の `content_block_stop` より後であることを見る。
fn assert_skeleton(frames: &[(String, Value)]) -> (&Value, &Value) {
    let ty = types(frames);
    assert!(!ty.is_empty(), "イベントが空");
    assert_eq!(ty[0], "message_start", "先頭が message_start でない: {ty:?}");
    assert_eq!(
        *ty.last().unwrap(),
        "message_stop",
        "末尾が message_stop でない: {ty:?}"
    );
    for one in ["message_start", "message_delta", "message_stop"] {
        assert_eq!(
            ty.iter().filter(|t| **t == one).count(),
            1,
            "{one} はちょうど 1 回: {ty:?}"
        );
    }
    assert!(!ty.contains(&"error"), "error イベントがある: {ty:?}");

    let delta_pos = ty.iter().position(|t| *t == "message_delta").unwrap();
    let stop_positions: Vec<usize> = ty
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == "content_block_stop")
        .map(|(i, _)| i)
        .collect();
    if let Some(last_stop) = stop_positions.iter().max() {
        assert!(
            *last_stop < delta_pos,
            "message_delta より後に content_block_stop がある: {ty:?}"
        );
    }
    let start_positions: Vec<usize> = ty
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == "content_block_start")
        .map(|(i, _)| i)
        .collect();
    if let Some(last_start) = start_positions.iter().max() {
        assert!(
            *last_start < delta_pos,
            "message_delta より後に content_block_start がある: {ty:?}"
        );
    }
    (frames[0].1.get("message").expect("message_start.message が無い"),
     &frames[delta_pos].1)
}

/// content block の index 契約を検証し、(index, content_block 種別) を返す。
///
/// - start の index は 0 から 1 ずつ単調増加
/// - delta / stop の index は現在 open 中の block のみ
/// - 開いた block はすべてちょうど 1 回 stop される
fn assert_block_indices(frames: &[(String, Value)]) -> Vec<(u64, String)> {
    let mut next_expected: u64 = 0;
    let mut open: Vec<(u64, String)> = Vec::new();
    let mut blocks: Vec<(u64, String)> = Vec::new();
    for (ty, ev) in frames {
        match ty.as_str() {
            "content_block_start" => {
                let idx = ev["index"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("start.index が数値でない: {ev}"));
                assert_eq!(
                    idx, next_expected,
                    "block index が単調増加でない: {idx} (期待 {next_expected})"
                );
                next_expected += 1;
                let btype = ev["content_block"]["type"]
                    .as_str()
                    .unwrap_or_else(|| panic!("content_block.type が無い: {ev}"))
                    .to_string();
                if btype == "tool_use" {
                    assert!(
                        ev["content_block"]["id"]
                            .as_str()
                            .map_or(false, |s| s.starts_with("toolu_")),
                        "tool_use block の id が不正: {ev}"
                    );
                    assert!(
                        ev["content_block"]["name"].is_string(),
                        "tool_use block の name が無い: {ev}"
                    );
                    assert!(
                        ev["content_block"]["input"].is_object(),
                        "tool_use block の input が object でない: {ev}"
                    );
                }
                assert!(
                    !open.iter().any(|(i, _)| *i == idx),
                    "同じ index の block が二重に open: {ev}"
                );
                open.push((idx, btype.clone()));
                blocks.push((idx, btype));
            }
            "content_block_delta" => {
                let idx = ev["index"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("delta.index が数値でない: {ev}"));
                let (_, btype) = open
                    .iter()
                    .find(|(i, _)| *i == idx)
                    .unwrap_or_else(|| panic!("open 中でない index の delta: {ev}"));
                let expected = if btype == "tool_use" {
                    "input_json_delta"
                } else {
                    "text_delta"
                };
                assert_eq!(
                    ev["delta"]["type"], expected,
                    "{btype} block の delta 種別が違う: {ev}"
                );
                if btype == "text" {
                    assert!(
                        ev["delta"]["text"].is_string(),
                        "text_delta.text が文字列でない: {ev}"
                    );
                } else {
                    assert!(
                        ev["delta"]["partial_json"].is_string(),
                        "input_json_delta.partial_json が文字列でない: {ev}"
                    );
                }
            }
            "content_block_stop" => {
                let idx = ev["index"]
                    .as_u64()
                    .unwrap_or_else(|| panic!("stop.index が数値でない: {ev}"));
                let pos = open
                    .iter()
                    .position(|(i, _)| *i == idx)
                    .unwrap_or_else(|| panic!("open 中でない index の stop: {ev}"));
                open.remove(pos);
            }
            _ => {}
        }
    }
    assert!(open.is_empty(), "stop されていない block が残った: {open:?}");
    blocks
}

fn stream_text(frames: &[(String, Value)]) -> String {
    let mut out = String::new();
    for (ty, ev) in frames {
        if ty == "content_block_delta" && ev["delta"]["type"] == "text_delta" {
            out.push_str(ev["delta"]["text"].as_str().unwrap());
        }
    }
    out
}

// ---------------------------------------------------------------------------
// テキストのみ: 厳密な順序と usage 分割
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_stream_exact_event_sequence() {
    let frames = stream(&dummy_app(), base_body()).await;
    let ty = types(&frames);

    // 厳密な列: message_start, content_block_start(0), text_delta*, stop(0),
    // message_delta, message_stop。中間は text_delta だけで埋まる。
    assert_eq!(ty[0], "message_start", "order = {ty:?}");
    assert_eq!(ty[1], "content_block_start", "order = {ty:?}");
    let n = ty.len();
    assert_eq!(
        &ty[n - 3..],
        ["content_block_stop", "message_delta", "message_stop"],
        "末尾の順序が違う: {ty:?}"
    );
    assert!(
        ty[2..n - 3].iter().all(|t| *t == "content_block_delta"),
        "block 区間に delta 以外がある: {ty:?}"
    );
    assert!(n > 5, "delta が 1 個も無い: {ty:?}");

    let (message, delta) = assert_skeleton(&frames);
    let blocks = assert_block_indices(&frames);
    assert_eq!(blocks, vec![(0u64, "text".to_string())]);

    // message_start: 空 content / stop_reason null / input+cache のみ
    assert_eq!(message["type"], "message");
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["model"], "denno-test");
    assert_eq!(message["content"], json!([]));
    assert_eq!(message["stop_reason"], Value::Null);
    assert_eq!(message["stop_sequence"], Value::Null);
    let usage = &message["usage"];
    assert_eq!(usage["input_tokens"], BASE_INPUT_TOKENS);
    assert_eq!(usage["output_tokens"], 0);
    assert_eq!(usage["cache_read_input_tokens"], 0);
    assert_eq!(usage["cache_creation_input_tokens"], BASE_INPUT_TOKENS);
    assert_eq!(
        usage["cache_read_input_tokens"].as_u64().unwrap()
            + usage["cache_creation_input_tokens"].as_u64().unwrap(),
        BASE_INPUT_TOKENS,
        "input が read + creation に分解されていない: {usage}"
    );

    // message_delta: stop_reason / stop_sequence / output のみ
    assert_eq!(delta["delta"]["stop_reason"], "end_turn");
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
    let du = delta["usage"].as_object().expect("usage が object でない");
    assert_eq!(du.len(), 1, "message_delta usage は output のみ: {delta}");
    assert_eq!(du["output_tokens"], BASE_OUTPUT_TOKENS);

    assert_eq!(stream_text(&frames), GREEDY);
    // 各 text トークンが 1 delta ずつ (7 トークン)
    let deltas = ty.iter().filter(|t| **t == "content_block_delta").count();
    assert_eq!(deltas, 7, "text_delta 数: {ty:?}");
}

#[tokio::test]
async fn text_stream_sequence_is_stable_across_repeats() {
    // 同じ app (prefix cache が温まる) でも順序と index は不変。
    let app = dummy_app();
    let _ = nonstream(&app, base_body()).await;
    let frames = stream(&app, base_body()).await;
    let (message, delta) = assert_skeleton(&frames);
    assert_eq!(
        types(&frames)[1],
        "content_block_start",
        "2 回目で先頭 block が変わった"
    );
    assert_eq!(assert_block_indices(&frames), vec![(0u64, "text".to_string())]);
    // 温まり: input の一部が cache_read に移り、残りが creation。
    // 具体値は再利用長に依存するため、分解の不変条件だけを固定する。
    let usage = &message["usage"];
    let read = usage["cache_read_input_tokens"].as_u64().unwrap();
    let creation = usage["cache_creation_input_tokens"].as_u64().unwrap();
    assert_eq!(usage["input_tokens"], BASE_INPUT_TOKENS);
    assert!(read > 0, "温まったのに cache_read が 0: {usage}");
    assert_eq!(read + creation, BASE_INPUT_TOKENS, "usage = {usage}");
    assert_eq!(delta["usage"]["output_tokens"], BASE_OUTPUT_TOKENS);
}

// ---------------------------------------------------------------------------
// 生成 0 トークン: block は open/stop のみで delta 無し
// ---------------------------------------------------------------------------

#[tokio::test]
async fn zero_token_stream_has_no_delta_but_full_skeleton() {
    let mut body = base_body();
    body["max_tokens"] = json!(0);
    let frames = stream(&dummy_app(), body).await;
    let ty = types(&frames);

    assert_eq!(
        ty,
        vec![
            "message_start",
            "content_block_start",
            "content_block_stop",
            "message_delta",
            "message_stop"
        ],
        "0 トークン時の列: {ty:?}"
    );
    let (message, delta) = assert_skeleton(&frames);
    assert_eq!(assert_block_indices(&frames), vec![(0u64, "text".to_string())]);
    assert_eq!(message["usage"]["output_tokens"], 0);
    assert_eq!(delta["delta"]["stop_reason"], "max_tokens");
    assert_eq!(delta["usage"]["output_tokens"], 0);
    assert_eq!(stream_text(&frames), "");
}

// ---------------------------------------------------------------------------
// stop_sequence: 順序は不変、フィールドだけ変わる
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_sequence_stream_keeps_sequence_and_sets_fields() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t7"]);
    let frames = stream(&dummy_app(), body).await;
    let ty = types(&frames);

    assert_eq!(ty[0], "message_start");
    assert_eq!(ty[1], "content_block_start");
    assert_eq!(
        &ty[ty.len() - 3..],
        ["content_block_stop", "message_delta", "message_stop"]
    );
    let (_, delta) = assert_skeleton(&frames);
    assert_eq!(assert_block_indices(&frames), vec![(0u64, "text".to_string())]);
    assert_eq!(delta["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(delta["delta"]["stop_sequence"], "t7");
    assert_eq!(delta["usage"]["output_tokens"], 3);
    assert!(!stream_text(&frames).contains("t7"), "stop sequence が本文に残った");
}

// ---------------------------------------------------------------------------
// tool_use: block index の単調増加と delta/stop の対応
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_use_stream_block_indices_are_monotonic() {
    let mut body = base_body();
    body["max_tokens"] = json!(64);
    body["tools"] = tools_json();

    let app = script_app(tool_script());
    let expected_input = count_tokens(&app, body.clone()).await;
    let frames = stream(&app, body.clone()).await;
    let ty = types(&frames);
    let (message, delta) = assert_skeleton(&frames);

    // 列: message_start, cbs(text,0), deltas(0)..., cbs(tool_use,1),
    // delta(json,1), stop(1), [finish で stop(0)], message_delta, message_stop
    assert_eq!(ty[0], "message_start");
    assert_eq!(ty[1], "content_block_start");
    let blocks = assert_block_indices(&frames);
    assert_eq!(
        blocks,
        vec![(0u64, "text".to_string()), (1u64, "tool_use".to_string())],
        "block 列: {blocks:?}"
    );

    // text block の delta は index 0、tool block は index 1 に閉じている。
    for (t, ev) in &frames {
        if t == "content_block_delta" && ev["delta"]["type"] == "text_delta" {
            assert_eq!(ev["index"], 0, "text_delta の index: {ev}");
        }
        if t == "content_block_delta" && ev["delta"]["type"] == "input_json_delta" {
            assert_eq!(ev["index"], 1, "input_json_delta の index: {ev}");
        }
    }
    let tool_start = frames
        .iter()
        .find(|(_, e)| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .map(|(_, e)| e)
        .expect("tool_use の content_block_start が無い");
    assert_eq!(tool_start["content_block"]["name"], "get_weather");
    let partial: Value = serde_json::from_str(
        frames
            .iter()
            .find(|(_, e)| e["delta"]["type"] == "input_json_delta")
            .map(|(_, e)| e["delta"]["partial_json"].as_str().unwrap())
            .expect("input_json_delta が無い"),
    )
    .expect("partial_json が JSON でない");
    assert_eq!(partial["city"], "Tokyo");

    // usage 分割: start は input/cache のみ (output 0)、delta は output のみ。
    assert_eq!(message["usage"]["input_tokens"], expected_input);
    assert_eq!(message["usage"]["output_tokens"], 0);
    assert_eq!(message["usage"]["cache_read_input_tokens"], 0);
    assert_eq!(
        message["usage"]["cache_creation_input_tokens"],
        expected_input
    );
    let du = delta["usage"].as_object().unwrap();
    assert_eq!(du.len(), 1, "message_delta usage は output のみ: {delta}");
    assert_eq!(du["output_tokens"], 16); // block 15 トークン + EOS

    // stop_reason は tool_use (非ストリーミングと同じ規則)
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
    assert_eq!(stream_text(&frames), "Checking.");
}

#[tokio::test]
async fn two_tool_blocks_get_consecutive_indices() {
    let mut body = base_body();
    body["max_tokens"] = json!(96);
    body["tools"] = tools_json();

    let app = script_app(two_tool_script());
    let expected_input = count_tokens(&app, body.clone()).await;
    let frames = stream(&app, body.clone()).await;

    let blocks = assert_block_indices(&frames);
    assert_eq!(
        blocks,
        vec![
            (0u64, "text".to_string()),
            (1u64, "tool_use".to_string()),
            (2u64, "tool_use".to_string()),
        ],
        "block 列: {blocks:?}"
    );
    let names: Vec<&str> = frames
        .iter()
        .filter(|(_, e)| e["type"] == "content_block_start" && e["content_block"]["type"] == "tool_use")
        .map(|(_, e)| e["content_block"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["get_weather", "run_tool"]);

    let (message, delta) = assert_skeleton(&frames);
    assert_eq!(message["usage"]["input_tokens"], expected_input);
    assert_eq!(message["usage"]["output_tokens"], 0);
    assert_eq!(delta["delta"]["stop_reason"], "tool_use");
    assert!(delta["usage"]["output_tokens"].as_u64().unwrap() > 16);

    // 各 tool_use block の直前で open 中の text block が閉じられるため、
    // stop は開いた順 (0 → 1 → 2) に並ぶ。
    let stop_indices: Vec<u64> = frames
        .iter()
        .filter(|(t, _)| t == "content_block_stop")
        .map(|(_, e)| e["index"].as_u64().unwrap())
        .collect();
    assert_eq!(stop_indices, vec![0, 1, 2], "stop 順: {stop_indices:?}");

    // 非ストリーミングと output_tokens / stop_reason が一致する。
    let ns = nonstream(&script_app(two_tool_script()), body).await;
    assert_eq!(ns["stop_reason"], "tool_use");
    assert_eq!(
        ns["usage"]["output_tokens"],
        delta["usage"]["output_tokens"],
        "stream と非 stream の output_tokens が違う: ns = {ns}"
    );
    assert_eq!(stream_text(&frames), "Checking.");
    assert_eq!(ns["content"][0]["text"], "Checking.");
}

// ---------------------------------------------------------------------------
// 不正リクエストでは SSE を開かない (イベント列契約の前提)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_request_returns_json_error_not_sse() {
    let mut body = base_body();
    body["max_tokens"] = json!(0);
    body["stream"] = json!(true);
    body["messages"] = json!([]);
    let (status, text) = post_raw(&dummy_app(), MESSAGES, &body.to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {text}");
    assert!(!text.contains("event: message_start"), "SSE が始まった: {text}");
    let v: Value = serde_json::from_str(&text).expect("JSON エラーでない");
    assert_eq!(v["type"], "error");
    assert_eq!(v["error"]["type"], "invalid_request_error");
}
