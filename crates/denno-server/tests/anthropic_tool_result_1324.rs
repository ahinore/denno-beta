//! Task 1324: Anthropic `/v1/messages` のマルチターン tool_use / tool_result
//! ブロックと構造化 content blocks の契約テスト。
//!
//! `src/lib.rs` の実装済み契約をワイヤレベルで固定する (`InMessage::to_text` /
//! `system_to_text` / `prepare_chat` / scheduler の SSE イベント列):
//!
//! - user → assistant `tool_use` → user `tool_result` のラウンドトリップが
//!   同期 / SSE の両方で受理され、同じ本文・usage・stop_reason になる
//! - `tool_use` は `<tool_call>` XML、`tool_result` は `<tool_response>` 包みの
//!   テキストへ写り、「同じテキストを明示した履歴」と同じプロンプト
//!   (`usage.input_tokens`) になる
//! - `is_error` は受理されるが履歴テキストには織り込まれない (実装済み挙動)
//! - `tool_result.content` は文字列 / text block 配列 / null / 空配列を受理し、
//!   空 content は空の `<tool_response>` になる (実装済み挙動)
//! - malformed な tool_use / tool_result (id / name / tool_use_id の欠落・空・
//!   未知参照・重複、input / content の型違い) は 400 の Anthropic 封筒
//! - assistant の text + tool_use 混在は text 間に空行 (`\n\n`) を挟んで連結
//! - `system` の text block 配列 + `cache_control` (有り/無し/null/{}/ttl) は
//!   すべて無視され、同じ連結文字列の system と同一プロンプトになる
//!
//! fixture は `tests/anthropic_parity_976.rs` と同一 (dummy モデル + dummy
//! トークナイザ。GPU 不要)。src は変更しない。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::dummy_engine;
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

/// 決定的 dummy モデル (直前トークン + 1 をピークにする)。
fn app() -> Router {
    app_from_engine(dummy_engine("denno-test", None).unwrap())
}

const WIDE_CONTEXT: usize = 512;

/// 長いプロンプト用の決定的モデル (dummy の文脈長 64 では 2 個の tool_use
/// block を含む履歴が入らないため、比較テストだけ文脈長を広げる)。
struct WideModel;

impl denno_engine::model::Model for WideModel {
    fn context(&self) -> usize {
        WIDE_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[denno_engine::testutil::DUMMY_EOS]
    }

    fn forward(
        &mut self,
        tokens: &[u32],
        cache: &mut denno_engine::kv_cache::KvCache,
    ) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; denno_engine::testutil::DUMMY_VOCAB];
        logits[(last + 1) % denno_engine::testutil::DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % denno_engine::testutil::DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> denno_engine::kv_cache::KvCache {
        denno_engine::kv_cache::KvCache::new(1, 4, WIDE_CONTEXT)
    }
}

fn wide_app() -> Router {
    let engine = Engine::from_parts(
        Box::new(WideModel),
        denno_engine::testutil::dummy_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    app_from_engine(engine)
}

async fn post_raw(app: &Router, uri: &str, body: &Value) -> (StatusCode, String) {
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

async fn post_full(app: &Router, body: &Value) -> (StatusCode, HeaderMap, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, v)
}

/// 非ストリーミング応答を要求し、200 を確認して JSON を返す。
async fn nonstream(app: &Router, mut body: Value) -> Value {
    body["stream"] = json!(false);
    let (status, text) = post_raw(app, "/v1/messages", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    serde_json::from_str(&text).expect("JSON 応答でない")
}

/// ストリーミング応答の生テキストと `data:` JSON 列 (ping を除く)。
async fn stream(app: &Router, mut body: Value) -> (String, Vec<Value>) {
    body["stream"] = json!(true);
    let (status, text) = post_raw(app, "/v1/messages", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let mut events = Vec::new();
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            let v: Value = serde_json::from_str(data).expect("SSE data が JSON ではない");
            if v["type"] != "ping" {
                events.push(v);
            }
        }
    }
    (text, events)
}

fn find<'a>(events: &'a [Value], ty: &str) -> &'a Value {
    events
        .iter()
        .find(|e| e["type"] == ty)
        .unwrap_or_else(|| panic!("{ty} イベントが無い: {events:?}"))
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

/// SSE の usage は message_start (input/cache) + message_delta (output) に分かれる。
fn merged_stream_usage(events: &[Value]) -> Value {
    let mut merged = find(events, "message_start")["message"]["usage"].clone();
    for (k, v) in find(events, "message_delta")["usage"].as_object().unwrap() {
        merged[k] = v.clone();
    }
    merged
}

fn text_of(v: &Value) -> &str {
    v["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("content[0].text が文字列ではない: {v}"))
}

fn input_tokens(v: &Value) -> u64 {
    v["usage"]["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("usage.input_tokens が無い: {v}"))
}

fn user(content: Value) -> Value {
    json!({"role": "user", "content": content})
}

fn assistant(content: Value) -> Value {
    json!({"role": "assistant", "content": content})
}

fn tool_use_block(id: &str, name: &str, input: Value) -> Value {
    json!({"type": "tool_use", "id": id, "name": name, "input": input})
}

fn tool_result_block(tool_use_id: &str, content: Value) -> Value {
    json!({"type": "tool_result", "tool_use_id": tool_use_id, "content": content})
}

fn tool_result_block_err(tool_use_id: &str, content: Value, is_error: bool) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    })
}

/// `tool_use_to_xml` と同じ形 (`src/lib.rs` の契約)。パラメータは 1 個前提。
fn expected_xml(name: &str, param: &str, value: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"
    )
}

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

fn body_with(messages: Value) -> Value {
    let mut body = base_body();
    body["messages"] = messages;
    body
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

fn body_with_tools(messages: Value) -> Value {
    let mut body = body_with(messages);
    body["tools"] = tools_json();
    body
}

/// Anthropic の 400 封筒を厳密に検証し、`error.message` を返す。
fn assert_clean_error(
    status: StatusCode,
    headers: &HeaderMap,
    v: &Value,
    must_contain: &[&str],
) -> String {
    assert_eq!(status, StatusCode::BAD_REQUEST, "400 でない: body = {v}");
    let ct = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    assert!(
        ct.starts_with("application/json"),
        "エラー封筒が JSON でない (content-type = {ct:?}): body = {v}"
    );

    let top = v
        .as_object()
        .unwrap_or_else(|| panic!("body が object でない: {v}"));
    let mut top_keys: Vec<&str> = top.keys().map(|s| s.as_str()).collect();
    top_keys.sort_unstable();
    assert_eq!(top_keys, vec!["error", "type"], "封筒のトップレベルが余計: {v}");

    assert_eq!(v["type"], "error", "body = {v}");
    let err = v["error"]
        .as_object()
        .unwrap_or_else(|| panic!("error が object でない: {v}"));
    let mut err_keys: Vec<&str> = err.keys().map(|s| s.as_str()).collect();
    err_keys.sort_unstable();
    assert_eq!(err_keys, vec!["message", "type"], "error のキーが余計: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");

    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "error.message が空: {v}");
    for needle in must_contain {
        assert!(msg.contains(needle), "message に {needle:?} が無い: {msg}");
    }
    msg.to_string()
}

async fn assert_400(body: Value, needles: &[&str]) {
    let (status, headers, v) = post_full(&app(), &body).await;
    assert_clean_error(status, &headers, &v, needles);
}

/// マルチターン履歴 (user → assistant tool_use → user tool_result → user)。
fn multi_turn_messages() -> Value {
    json!([
        user(json!("weather in tokyo?")),
        assistant(json!([tool_use_block(
            "toolu_1",
            "get_weather",
            json!({"city": "Tokyo"})
        )])),
        user(json!([tool_result_block("toolu_1", json!("sunny 25C"))])),
        user(json!("thanks")),
    ])
}

/// 同じ履歴を明示テキストで書いたもの。
fn multi_turn_explicit_text() -> Value {
    json!([
        user(json!("weather in tokyo?")),
        assistant(json!(expected_xml("get_weather", "city", "Tokyo"))),
        user(json!("<tool_response>\nsunny 25C\n</tool_response>")),
        user(json!("thanks")),
    ])
}

// ---------------------------------------------------------------------------
// 1. マルチターン tool_use / tool_result のラウンドトリップ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sync_multi_turn_tool_round_trip_is_accepted() {
    let app = app();
    let v = nonstream(&app, body_with_tools(multi_turn_messages())).await;

    assert_eq!(v["type"], "message");
    assert_eq!(v["role"], "assistant");
    let text = text_of(&v);
    assert!(!text.is_empty());
    let fr = v["stop_reason"].as_str().unwrap();
    assert!(fr == "end_turn" || fr == "max_tokens", "stop_reason = {fr}");
    assert!(input_tokens(&v) > 0);
}

#[tokio::test]
async fn sse_multi_turn_matches_sync() {
    let body = body_with_tools(multi_turn_messages());
    let ns = nonstream(&app(), body.clone()).await;
    let (raw, ev) = stream(&app(), body).await;

    // イベント列: message_start → content_block_start(text, index 0) →
    // text_delta 群 → content_block_stop → message_delta → message_stop。
    let types: Vec<&str> = ev.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert!(types.len() >= 5, "events = {ev:?}");
    assert_eq!(types[0], "message_start");
    assert_eq!(types[1], "content_block_start");
    let n = types.len();
    assert_eq!(
        &types[n - 3..],
        ["content_block_stop", "message_delta", "message_stop"],
        "末尾のイベント順が違う: {types:?}"
    );
    assert!(
        types[2..n - 3].iter().all(|t| *t == "content_block_delta"),
        "中間イベントが text_delta でない: {types:?}"
    );
    assert!(!types.contains(&"error"), "error イベントがある: {ev:?}");
    for ty in &types {
        assert!(
            raw.contains(&format!("event: {ty}")),
            "生 SSE に `event: {ty}` 行が無い: {raw}"
        );
    }

    let start = find(&ev, "content_block_start");
    assert_eq!(start["index"], 0);
    assert_eq!(start["content_block"]["type"], "text");
    assert_eq!(start["content_block"]["text"], "");
    for e in &ev {
        if e["type"] == "content_block_delta" {
            assert_eq!(e["index"], 0);
            assert_eq!(e["delta"]["type"], "text_delta");
            assert!(e["delta"]["text"].is_string(), "delta = {e}");
        }
    }
    let message = &find(&ev, "message_start")["message"];
    assert_eq!(message["type"], "message");
    assert_eq!(message["role"], "assistant");
    assert_eq!(message["model"], "denno-test");
    assert_eq!(message["content"], json!([]));
    assert_eq!(message["stop_reason"], Value::Null);

    // 本文・usage・stop_reason が非ストリーミングと一致する。
    assert_eq!(stream_text(&ev), text_of(&ns));
    let merged = merged_stream_usage(&ev);
    assert_eq!(merged["input_tokens"], ns["usage"]["input_tokens"]);
    assert_eq!(merged["output_tokens"], ns["usage"]["output_tokens"]);
    assert_eq!(
        merged["cache_read_input_tokens"],
        ns["usage"]["cache_read_input_tokens"]
    );
    assert_eq!(
        find(&ev, "message_delta")["delta"]["stop_reason"],
        ns["stop_reason"]
    );
}

#[tokio::test]
async fn tool_history_maps_to_the_same_prompt_as_explicit_text() {
    let app = app();
    let implicit = nonstream(&app, body_with_tools(multi_turn_messages())).await;
    let explicit = nonstream(&app, body_with_tools(multi_turn_explicit_text())).await;

    assert_eq!(
        input_tokens(&implicit),
        input_tokens(&explicit),
        "tool 履歴が明示テキストと同じプロンプトになっていない"
    );
    assert_eq!(text_of(&implicit), text_of(&explicit));
}

#[tokio::test]
async fn tool_use_first_block_has_no_leading_separator() {
    let app = app();
    let implicit = body_with_tools(json!([
        user(json!("weather?")),
        assistant(json!([tool_use_block(
            "toolu_a",
            "get_weather",
            json!({"city": "Tokyo"})
        )])),
        user(json!([tool_result_block("toolu_a", json!("sunny"))])),
        user(json!("thanks")),
    ]));
    let explicit = body_with_tools(json!([
        user(json!("weather?")),
        assistant(json!(expected_xml("get_weather", "city", "Tokyo"))),
        user(json!("<tool_response>\nsunny\n</tool_response>")),
        user(json!("thanks")),
    ]));
    let a = nonstream(&app, implicit).await;
    let b = nonstream(&app, explicit).await;
    assert_eq!(input_tokens(&a), input_tokens(&b));
    assert_eq!(text_of(&a), text_of(&b));
}

// ---------------------------------------------------------------------------
// 2. tool_result の content 形 / is_error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_result_with_is_error_true_is_accepted_and_ignored() {
    let app = app();
    let plain = nonstream(&app, body_with_tools(multi_turn_messages())).await;

    for flag in [true, false] {
        let messages = json!([
            user(json!("weather in tokyo?")),
            assistant(json!([tool_use_block(
                "toolu_1",
                "get_weather",
                json!({"city": "Tokyo"})
            )])),
            user(json!([tool_result_block_err(
                "toolu_1",
                json!("sunny 25C"),
                flag
            )])),
            user(json!("thanks")),
        ]);
        let v = nonstream(&app, body_with_tools(messages)).await;
        assert_eq!(
            input_tokens(&v),
            input_tokens(&plain),
            "is_error = {flag} がプロンプトへ影響している"
        );
        assert_eq!(text_of(&v), text_of(&plain));
    }
}

#[tokio::test]
async fn tool_result_array_of_text_blocks_equals_joined_string() {
    let app = app();
    let string_form = nonstream(&app, body_with_tools(multi_turn_messages())).await;

    let messages = json!([
        user(json!("weather in tokyo?")),
        assistant(json!([tool_use_block(
            "toolu_1",
            "get_weather",
            json!({"city": "Tokyo"})
        )])),
        user(json!([tool_result_block_err(
            "toolu_1",
            json!([
                {"type": "text", "text": "sunny "},
                {"type": "text", "text": "25C"}
            ]),
            true
        )])),
        user(json!("thanks")),
    ]);
    let array_form = nonstream(&app, body_with_tools(messages)).await;
    assert_eq!(
        input_tokens(&array_form),
        input_tokens(&string_form),
        "text block 配列が連結文字列と同じプロンプトになっていない"
    );
    assert_eq!(text_of(&array_form), text_of(&string_form));
}

#[tokio::test]
async fn empty_tool_result_content_is_accepted_as_empty_response() {
    // 実装済み挙動: content の欠落 / null / 空文字列 / 空配列はすべて空本文の
    // `<tool_response>` になり 400 にはならない (`InMessage::to_text`)。
    let app = app();
    let mut baseline: Option<(u64, String)> = None;
    for content in [json!(null), json!(""), json!([])] {
        let messages = json!([
            user(json!("weather?")),
            assistant(json!([tool_use_block(
                "toolu_e",
                "get_weather",
                json!({"city": "Tokyo"})
            )])),
            user(json!([tool_result_block("toolu_e", content.clone())])),
            user(json!("thanks")),
        ]);
        let v = nonstream(&app, body_with_tools(messages)).await;
        let got = (input_tokens(&v), text_of(&v).to_string());
        match &baseline {
            None => baseline = Some(got),
            Some(b) => assert_eq!(&got, b, "空 content のプロンプトが揃っていない: {content}"),
        }
    }
}

#[tokio::test]
async fn multiple_tool_results_are_joined_and_match_explicit_text() {
    let app = wide_app();
    let implicit = body_with_tools(json!([
        user(json!("x")),
        assistant(json!([
            tool_use_block("toolu_t", "f", json!({"c": "a"})),
            tool_use_block("toolu_o", "f", json!({"c": "b"})),
        ])),
        user(json!([
            tool_result_block("toolu_t", json!("a")),
            tool_result_block("toolu_o", json!("b")),
        ])),
        user(json!("y")),
    ]));
    // 2 個の tool_result は改行 1 個で連結される。
    let explicit = body_with_tools(json!([
        user(json!("x")),
        assistant(json!(format!(
            "{}\n\n{}",
            expected_xml("f", "c", "a"),
            expected_xml("f", "c", "b")
        ))),
        user(json!("<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response>")),
        user(json!("y")),
    ]));
    let a = nonstream(&app, implicit).await;
    let b = nonstream(&app, explicit).await;
    assert_eq!(input_tokens(&a), input_tokens(&b));
    assert_eq!(text_of(&a), text_of(&b));
}

// ---------------------------------------------------------------------------
// 3. malformed tool_use / tool_result → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_use_missing_or_empty_id_is_400() {
    for block in [
        json!({"type": "tool_use", "name": "get_weather", "input": {}}),
        json!({"type": "tool_use", "id": "", "name": "get_weather", "input": {}}),
        json!({"type": "tool_use", "id": 123, "name": "get_weather", "input": {}}),
        json!({"type": "tool_use", "id": null, "name": "get_weather", "input": {}}),
    ] {
        assert_400(
            body_with_tools(json!([user(json!("hi")), assistant(json!([block]))])),
            &["tool_use.id must be a non-empty string"],
        )
        .await;
    }
}

#[tokio::test]
async fn tool_use_missing_or_empty_name_is_400() {
    for block in [
        json!({"type": "tool_use", "id": "toolu_1", "input": {}}),
        json!({"type": "tool_use", "id": "toolu_1", "name": "", "input": {}}),
        json!({"type": "tool_use", "id": "toolu_1", "name": 7, "input": {}}),
    ] {
        assert_400(
            body_with_tools(json!([user(json!("hi")), assistant(json!([block]))])),
            &["tool_use.name must be a non-empty string"],
        )
        .await;
    }
}

#[tokio::test]
async fn tool_use_non_object_input_is_400() {
    for input in [json!("city=Tokyo"), json!([1, 2]), json!(3)] {
        let block = json!({
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": input,
        });
        assert_400(
            body_with_tools(json!([user(json!("hi")), assistant(json!([block]))])),
            &["tool_use.input must be an object"],
        )
        .await;
    }
}

#[tokio::test]
async fn tool_use_in_user_message_is_400() {
    assert_400(
        body_with_tools(json!([
            user(json!([tool_use_block(
                "toolu_1",
                "get_weather",
                json!({"city": "Tokyo"})
            )])),
        ])),
        &["tool_use block is only allowed in assistant messages"],
    )
    .await;
}

#[tokio::test]
async fn duplicate_tool_use_id_is_400() {
    let messages = json!([
        user(json!("hi")),
        assistant(json!([tool_use_block(
            "toolu_dup",
            "get_weather",
            json!({"city": "Tokyo"})
        )])),
        user(json!([tool_result_block("toolu_dup", json!("sunny"))])),
        assistant(json!([tool_use_block(
            "toolu_dup",
            "get_weather",
            json!({"city": "Osaka"})
        )])),
    ]);
    assert_400(body_with_tools(messages), &["duplicate id", "toolu_dup"]).await;
}

#[tokio::test]
async fn tool_result_missing_or_empty_tool_use_id_is_400() {
    for block in [
        json!({"type": "tool_result", "content": "x"}),
        json!({"type": "tool_result", "tool_use_id": "", "content": "x"}),
        json!({"type": "tool_result", "tool_use_id": 42, "content": "x"}),
    ] {
        let messages = json!([
            user(json!("hi")),
            assistant(json!([tool_use_block(
                "toolu_1",
                "get_weather",
                json!({"city": "Tokyo"})
            )])),
            user(json!([block])),
        ]);
        assert_400(
            body_with_tools(messages),
            &["tool_result.tool_use_id must be a non-empty string"],
        )
        .await;
    }
}

#[tokio::test]
async fn tool_result_with_unknown_tool_use_id_is_400() {
    let messages = json!([
        user(json!("hi")),
        assistant(json!([tool_use_block(
            "toolu_1",
            "get_weather",
            json!({"city": "Tokyo"})
        )])),
        user(json!([tool_result_block("toolu_missing", json!("sunny"))])),
    ]);
    assert_400(
        body_with_tools(messages),
        &["toolu_missing", "does not match any tool_use block"],
    )
    .await;
}

#[tokio::test]
async fn tool_result_in_assistant_message_is_400() {
    let messages = json!([
        user(json!("hi")),
        assistant(json!([tool_result_block("toolu_1", json!("sunny"))])),
    ]);
    assert_400(
        body_with_tools(messages),
        &["tool_result block is only allowed in user messages"],
    )
    .await;
}

#[tokio::test]
async fn tool_result_non_string_or_non_text_array_content_is_400() {
    let base = |content: Value| {
        json!([
            user(json!("hi")),
            assistant(json!([tool_use_block(
                "toolu_1",
                "get_weather",
                json!({"city": "Tokyo"})
            )])),
            user(json!([tool_result_block("toolu_1", content)])),
        ])
    };
    for content in [json!(42), json!({"text": "x"}), json!(true)] {
        assert_400(
            body_with_tools(base(content)),
            &["tool_result.content must be a string or an array of text blocks"],
        )
        .await;
    }
    for item in [
        json!({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}),
        json!({"text": "missing type"}),
    ] {
        assert_400(
            body_with_tools(base(json!([item]))),
            &["unsupported tool_result content block type"],
        )
        .await;
    }
}

#[tokio::test]
async fn unknown_or_non_object_content_block_is_400() {
    assert_400(
        body_with_tools(json!([
            user(json!("hi")),
            assistant(json!([{"type": "thinking", "thinking": "hmm"}])),
        ])),
        &["unsupported content block type", "thinking"],
    )
    .await;
    assert_400(
        body_with_tools(json!([user(json!([{}]))])),
        &["unsupported content block type"],
    )
    .await;
    assert_400(
        body_with_tools(json!([user(json!(7))])),
        &["messages[].content must be a string or an array of content blocks"],
    )
    .await;
}

// ---------------------------------------------------------------------------
// 4. 混在ブロック / system 配列 + cache_control
// ---------------------------------------------------------------------------

#[tokio::test]
async fn interleaved_text_and_tool_use_blocks_are_joined_with_blank_lines() {
    let app = app();
    let implicit = body_with_tools(json!([
        user(json!("weather in tokyo?")),
        assistant(json!([
            {"type": "text", "text": "Let me check."},
            tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"})),
        ])),
        user(json!([tool_result_block("toolu_1", json!("sunny 25C"))])),
        user(json!("thanks")),
    ]));
    let explicit_text = format!(
        "Let me check.\n\n{}",
        expected_xml("get_weather", "city", "Tokyo")
    );
    let explicit = body_with_tools(json!([
        user(json!("weather in tokyo?")),
        assistant(json!(explicit_text)),
        user(json!("<tool_response>\nsunny 25C\n</tool_response>")),
        user(json!("thanks")),
    ]));
    let a = nonstream(&app, implicit).await;
    let b = nonstream(&app, explicit).await;
    assert_eq!(
        input_tokens(&a),
        input_tokens(&b),
        "混在 block の連結が明示テキストと一致しない"
    );
    assert_eq!(text_of(&a), text_of(&b));
}

#[tokio::test]
async fn tool_use_then_text_appends_without_separator() {
    let app = app();
    let implicit = body_with_tools(json!([
        user(json!("weather?")),
        assistant(json!([
            tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"})),
            {"type": "text", "text": " done."},
        ])),
        user(json!([tool_result_block("toolu_1", json!("sunny"))])),
        user(json!("thanks")),
    ]));
    // tool_use の後に来る text は区切り無しで連結される (実装済み挙動)。
    let explicit_text = format!(
        "{} done.",
        expected_xml("get_weather", "city", "Tokyo")
    );
    let explicit = body_with_tools(json!([
        user(json!("weather?")),
        assistant(json!(explicit_text)),
        user(json!("<tool_response>\nsunny\n</tool_response>")),
        user(json!("thanks")),
    ]));
    let a = nonstream(&app, implicit).await;
    let b = nonstream(&app, explicit).await;
    assert_eq!(input_tokens(&a), input_tokens(&b));
    assert_eq!(text_of(&a), text_of(&b));
}

#[tokio::test]
async fn system_array_with_cache_control_matches_plain_string() {
    let app = app();
    let mut mixed = body_with_tools(multi_turn_messages());
    mixed["system"] = json!([
        {"type": "text", "text": "You are "},
        {"type": "text", "text": "a test.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " Done", "cache_control": null},
        {"type": "text", "text": " Now", "cache_control": {}},
        {
            "type": "text",
            "text": " End",
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        },
    ]);
    let mut plain = body_with_tools(multi_turn_messages());
    plain["system"] = json!("You are a test. Done Now End");

    let a = nonstream(&app, mixed.clone()).await;
    let b = nonstream(&app, plain.clone()).await;
    assert_eq!(
        input_tokens(&a),
        input_tokens(&b),
        "cache_control の混在で input_tokens が変わる"
    );
    assert_eq!(text_of(&a), text_of(&b));

    // count_tokens でも同じ連結結果になる。
    let (s1, t1) = post_raw(&app, "/v1/messages/count_tokens", &mixed).await;
    assert_eq!(s1, StatusCode::OK, "body = {t1}");
    let (s2, t2) = post_raw(&app, "/v1/messages/count_tokens", &plain).await;
    assert_eq!(s2, StatusCode::OK, "body = {t2}");
    let c1: Value = serde_json::from_str(&t1).unwrap();
    let c2: Value = serde_json::from_str(&t2).unwrap();
    assert_eq!(c1["input_tokens"], c2["input_tokens"], "count_tokens 不一致");
}

#[tokio::test]
async fn sse_with_system_array_and_cache_control_matches_sync() {
    let mut body = body_with_tools(multi_turn_messages());
    body["system"] = json!([
        {"type": "text", "text": "You are "},
        {"type": "text", "text": "a test.", "cache_control": {"type": "ephemeral"}},
    ]);
    let ns = nonstream(&app(), body.clone()).await;
    let (_raw, ev) = stream(&app(), body).await;
    assert_eq!(stream_text(&ev), text_of(&ns));
    assert_eq!(
        merged_stream_usage(&ev)["input_tokens"],
        ns["usage"]["input_tokens"]
    );
}
