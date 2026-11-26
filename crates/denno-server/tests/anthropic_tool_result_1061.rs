//! Task 1061: Anthropic `/v1/messages` のマルチターン tool_use / tool_result
//! 監査テスト (is_error / string vs text-block 配列 / tool_use_id 欠落)。
//!
//! 1324 の契約テストを補う角度だけを置く (`src/lib.rs` は task 1061 で
//! is_error と text block の型検証を追加した):
//!
//! - `tool_result.is_error` は省略可能な boolean。bool は受理し、履歴テキスト
//!   には織り込まない (`<tool_response>` の本文だけ)。文字列 / 数値 / 配列 /
//!   オブジェクト / null は 400 (`/v1/messages` と `count_tokens` の両方)
//! - `tool_result.content` の text block は `text` が文字列のみ。欠落 / null は
//!   空文字、型違いは 400
//! - 文字列 content / 1 個の text block / 複数 text block の連結は同じプロンプト
//! - `tool_use_id` の欠落・null・空・非文字列は 400。先行する assistant の
//!   tool_use を指さない参照も 400 (順序も含む)
//! - 2 往復 (tool_use → tool_result → tool_use → tool_result) が同期 / SSE で
//!   同じ本文・usage になり、`count_tokens` が `input_tokens` と一致する
//! - 不正履歴は `stream: true` でも SSE を開かず 400 の JSON 封筒
//!
//! fixture は `tests/anthropic_parity_976.rs` / `anthropic_tool_result_1324.rs`
//! と同一 (dummy モデル + dummy トークナイザ。GPU 不要)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
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

fn app() -> Router {
    app_from_engine(dummy_engine("denno-test", None).unwrap())
}

const WIDE_CONTEXT: usize = 512;

/// dummy の文脈長 64 では 2 往復の履歴が入らないため、比較テストだけ広げる。
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

async fn post_uri(app: &Router, uri: &str, body: &Value) -> (StatusCode, String, String) {
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
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default()
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        ct,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

/// 非ストリーム送信 (200 を要求) して JSON を返す。
async fn nonstream(app: &Router, mut body: Value) -> Value {
    body["stream"] = json!(false);
    let (status, _, text) = post_uri(app, "/v1/messages", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("JSON でない: {e}: {text}"))
}

/// SSE 送信 (200 を要求) してイベント配列を返す。
async fn stream(app: &Router, mut body: Value) -> Vec<Value> {
    body["stream"] = json!(true);
    let (status, ct, text) = post_uri(app, "/v1/messages", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("SSE JSON でない: {e}: {d}")))
        .collect()
}

fn stream_text(events: &[Value]) -> String {
    let mut s = String::new();
    for e in events {
        if e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta" {
            s.push_str(e["delta"]["text"].as_str().unwrap_or_default());
        }
    }
    s
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

fn tool_result(tool_use_id: &str, content: Value) -> Value {
    json!({"type": "tool_result", "tool_use_id": tool_use_id, "content": content})
}

fn tool_result_err(tool_use_id: &str, content: Value, is_error: Value) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
        "is_error": is_error,
    })
}

fn expected_xml(name: &str, param: &str, value: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"
    )
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
    json!({
        "model": "denno-test",
        "max_tokens": 8,
        "temperature": 0.0,
        "tools": tools_json(),
        "messages": messages,
    })
}

/// 400 の Anthropic 封筒であることを検証し、error.message を返す。
fn assert_400(status: StatusCode, ct: &str, text: &str, needles: &[&str]) -> String {
    assert_eq!(status, StatusCode::BAD_REQUEST, "400 でない: body = {text}");
    assert!(ct.starts_with("application/json"), "content-type = {ct}");
    let v: Value = serde_json::from_str(text).unwrap_or_else(|e| panic!("JSON でない: {e}: {text}"));
    assert_eq!(v["type"], "error", "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    let msg = v["error"]["message"].as_str().unwrap_or_default().to_string();
    assert!(!msg.is_empty(), "error.message が空: {v}");
    for needle in needles {
        assert!(msg.contains(needle), "message に {needle:?} が無い: {msg}");
    }
    msg
}

/// 同じ 400 が `/v1/messages` と `/v1/messages/count_tokens` の両方で出る。
async fn assert_400_both(messages: Value, needles: &[&str]) {
    let app = app();
    let body = body_with_tools(messages);
    let (s, ct, t) = post_uri(&app, "/v1/messages", &body).await;
    assert_400(s, &ct, &t, needles);
    let (s, ct, t) = post_uri(&app, "/v1/messages/count_tokens", &body).await;
    assert_400(s, &ct, &t, needles);
}

/// user → assistant tool_use → user tool_result → user。
fn one_round(result_block: Value) -> Value {
    json!([
        user(json!("weather in tokyo?")),
        assistant(json!([tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"}))])),
        user(json!([result_block])),
        user(json!("thanks")),
    ])
}

/// 2 往復 (tool_use → tool_result × 2)。
fn two_rounds() -> Value {
    json!([
        user(json!("weather in tokyo?")),
        assistant(json!([tool_use_block("toolu_a", "get_weather", json!({"city": "Tokyo"}))])),
        user(json!([tool_result("toolu_a", json!("sunny 25C"))])),
        assistant(json!([tool_use_block("toolu_b", "get_weather", json!({"city": "Osaka"}))])),
        user(json!([tool_result("toolu_b", json!("rain 18C"))])),
        user(json!("thanks")),
    ])
}

// ---------------------------------------------------------------------------
// 1. is_error
// ---------------------------------------------------------------------------

#[tokio::test]
async fn is_error_true_and_false_are_accepted_with_identical_prompt() {
    let app = wide_app();
    let none = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", json!("boom"))))).await;
    for flag in [json!(false), json!(true)] {
        let v = nonstream(
            &app,
            body_with_tools(one_round(tool_result_err("toolu_1", json!("boom"), flag.clone()))),
        )
        .await;
        assert_eq!(
            input_tokens(&v),
            input_tokens(&none),
            "is_error = {flag} がプロンプトへ影響している"
        );
        assert_eq!(text_of(&v), text_of(&none), "is_error = {flag}");
    }
}

#[tokio::test]
async fn is_error_true_with_text_block_array_matches_string_form() {
    let app = wide_app();
    let string_form = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", json!("a b"))))).await;
    let array_form = nonstream(
        &app,
        body_with_tools(one_round(tool_result_err(
            "toolu_1",
            json!([{"type": "text", "text": "a "}, {"type": "text", "text": "b"}]),
            json!(true),
        ))),
    )
    .await;
    assert_eq!(input_tokens(&array_form), input_tokens(&string_form));
    assert_eq!(text_of(&array_form), text_of(&string_form));
}

#[tokio::test]
async fn is_error_non_bool_is_rejected_on_messages_and_count_tokens() {
    for bad in [json!("true"), json!(1), json!(0), json!([]), json!({}), json!(null)] {
        assert_400_both(
            one_round(tool_result_err("toolu_1", json!("x"), bad.clone())),
            &["is_error"],
        )
        .await;
    }
}

#[tokio::test]
async fn invalid_history_with_stream_true_is_a_400_json_envelope() {
    let app = app();
    let mut body = body_with_tools(one_round(tool_result_err("toolu_1", json!("x"), json!("yes"))));
    body["stream"] = json!(true);
    let (status, ct, text) = post_uri(&app, "/v1/messages", &body).await;
    assert_400(status, &ct, &text, &["is_error"]);
}

// ---------------------------------------------------------------------------
// 2. content: string vs text-block 配列
// ---------------------------------------------------------------------------

#[tokio::test]
async fn string_and_single_text_block_and_split_blocks_are_equivalent() {
    let app = wide_app();
    let base = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", json!("sunny 25C"))))).await;
    let forms = [
        json!([{"type": "text", "text": "sunny 25C"}]),
        json!([{"type": "text", "text": "sunny"}, {"type": "text", "text": " 25C"}]),
        json!([{"type": "text", "text": "sun"}, {"type": "text", "text": "ny 2"}, {"type": "text", "text": "5C"}]),
    ];
    for content in forms {
        let v = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", content.clone())))).await;
        assert_eq!(
            input_tokens(&v),
            input_tokens(&base),
            "content = {content} が連結文字列と同じプロンプトになっていない"
        );
        assert_eq!(text_of(&v), text_of(&base), "content = {content}");
    }
}

#[tokio::test]
async fn text_block_with_missing_or_null_text_is_empty_but_non_string_is_400() {
    let app = wide_app();
    let empty = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", json!(""))))).await;
    for content in [
        json!([{"type": "text"}]),
        json!([{"type": "text", "text": null}]),
    ] {
        let v = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", content.clone())))).await;
        assert_eq!(input_tokens(&v), input_tokens(&empty), "content = {content}");
        assert_eq!(text_of(&v), text_of(&empty), "content = {content}");
    }
    for bad in [json!(3), json!(true), json!(["x"]), json!({"t": 1})] {
        assert_400_both(
            one_round(tool_result("toolu_1", json!([{"type": "text", "text": bad.clone()}]))),
            &["text"],
        )
        .await;
    }
}

#[tokio::test]
async fn tool_result_array_with_non_text_inner_block_is_400() {
    assert_400_both(
        one_round(tool_result(
            "toolu_1",
            json!([{"type": "image", "source": {"type": "base64", "data": "AAAA"}}]),
        )),
        &["tool_result"],
    )
    .await;
}

#[tokio::test]
async fn empty_tool_result_content_forms_agree() {
    let app = wide_app();
    let mut baseline: Option<(u64, String)> = None;
    for content in [json!(null), json!(""), json!([])] {
        let v = nonstream(&app, body_with_tools(one_round(tool_result("toolu_1", content.clone())))).await;
        let got = (input_tokens(&v), text_of(&v).to_string());
        match &baseline {
            None => baseline = Some(got),
            Some(b) => assert_eq!(&got, b, "空 content のプロンプトが揃っていない: {content}"),
        }
    }
}

// ---------------------------------------------------------------------------
// 3. tool_use_id の欠落 / 順序
// ---------------------------------------------------------------------------

#[tokio::test]
async fn missing_null_empty_or_non_string_tool_use_id_is_400() {
    let cases = [
        json!({"type": "tool_result", "content": "x"}),
        json!({"type": "tool_result", "tool_use_id": null, "content": "x"}),
        json!({"type": "tool_result", "tool_use_id": "", "content": "x"}),
        json!({"type": "tool_result", "tool_use_id": 7, "content": "x"}),
        json!({"type": "tool_result", "tool_use_id": ["toolu_1"], "content": "x"}),
    ];
    for block in cases {
        assert_400_both(one_round(block.clone()), &["tool_use_id"]).await;
    }
}

#[tokio::test]
async fn tool_result_before_the_matching_tool_use_is_400() {
    // 先頭の user メッセージに tool_result (先行する tool_use が無い)。
    assert_400_both(
        json!([
            user(json!([tool_result("toolu_1", json!("x"))])),
            assistant(json!([tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"}))])),
            user(json!("thanks")),
        ]),
        &["toolu_1"],
    )
    .await;
}

#[tokio::test]
async fn tool_result_referencing_a_later_tool_use_is_400() {
    assert_400_both(
        json!([
            user(json!("weather?")),
            user(json!([tool_result("toolu_2", json!("x"))])),
            assistant(json!([tool_use_block("toolu_2", "get_weather", json!({"city": "Tokyo"}))])),
            user(json!("thanks")),
        ]),
        &["toolu_2"],
    )
    .await;
}

#[tokio::test]
async fn tool_result_whitespace_only_id_is_unknown_reference_400() {
    assert_400_both(one_round(tool_result(" ", json!("x"))), &["tool_use_id"]).await;
}

#[tokio::test]
async fn tool_result_may_reference_a_non_adjacent_earlier_tool_use() {
    // 実装は「先行する assistant メッセージなら何個前でも許す」(緩めの契約)。
    let app = wide_app();
    let implicit = body_with_tools(json!([
        user(json!("weather?")),
        assistant(json!([tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"}))])),
        user(json!("hold on")),
        user(json!([tool_result("toolu_1", json!("sunny"))])),
        user(json!("thanks")),
    ]));
    let explicit = body_with_tools(json!([
        user(json!("weather?")),
        assistant(json!(expected_xml("get_weather", "city", "Tokyo"))),
        user(json!("hold on")),
        user(json!("<tool_response>\nsunny\n</tool_response>")),
        user(json!("thanks")),
    ]));
    let a = nonstream(&app, implicit).await;
    let b = nonstream(&app, explicit).await;
    assert_eq!(input_tokens(&a), input_tokens(&b));
    assert_eq!(text_of(&a), text_of(&b));
}

#[tokio::test]
async fn duplicate_tool_result_for_same_id_is_accepted() {
    let app = wide_app();
    let v = nonstream(
        &app,
        body_with_tools(json!([
            user(json!("weather?")),
            assistant(json!([tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"}))])),
            user(json!([tool_result("toolu_1", json!("first")), tool_result("toolu_1", json!("again"))])),
            user(json!("thanks")),
        ])),
    )
    .await;
    assert!(input_tokens(&v) > 0);
}

#[tokio::test]
async fn tool_result_in_assistant_message_is_400_on_both_endpoints() {
    assert_400_both(
        json!([
            user(json!("weather?")),
            assistant(json!([
                tool_use_block("toolu_1", "get_weather", json!({"city": "Tokyo"})),
                tool_result("toolu_1", json!("x")),
            ])),
            user(json!("thanks")),
        ]),
        &["tool_result"],
    )
    .await;
}

// ---------------------------------------------------------------------------
// 4. マルチターン (2 往復) の同期 / SSE / count_tokens パリティ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn two_rounds_sync_and_sse_agree_and_count_tokens_matches() {
    let app = wide_app();
    let body = body_with_tools(two_rounds());

    let ns = nonstream(&app, body.clone()).await;
    let ev = stream(&app, body.clone()).await;
    assert!(!ev.is_empty(), "SSE イベントが空");
    assert_eq!(ev[0]["type"], "message_start");
    assert_eq!(
        stream_text(&ev),
        text_of(&ns),
        "SSE の本文が同期と違う"
    );
    assert_eq!(
        ev[0]["message"]["usage"]["input_tokens"],
        json!(input_tokens(&ns)),
        "SSE の message_start usage が同期と違う"
    );

    let (status, ct, text) = post_uri(&app, "/v1/messages/count_tokens", &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert!(ct.starts_with("application/json"), "content-type = {ct}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["input_tokens"], json!(input_tokens(&ns)), "count_tokens 不一致");
}

#[tokio::test]
async fn two_rounds_match_explicit_text_history() {
    let app = wide_app();
    let implicit = nonstream(&app, body_with_tools(two_rounds())).await;
    let explicit = nonstream(
        &app,
        body_with_tools(json!([
            user(json!("weather in tokyo?")),
            assistant(json!(expected_xml("get_weather", "city", "Tokyo"))),
            user(json!("<tool_response>\nsunny 25C\n</tool_response>")),
            assistant(json!(expected_xml("get_weather", "city", "Osaka"))),
            user(json!("<tool_response>\nrain 18C\n</tool_response>")),
            user(json!("thanks")),
        ])),
    )
    .await;
    assert_eq!(input_tokens(&implicit), input_tokens(&explicit));
    assert_eq!(text_of(&implicit), text_of(&explicit));
}

#[tokio::test]
async fn two_rounds_with_is_error_flags_match_plain_history() {
    let app = wide_app();
    let plain = nonstream(&app, body_with_tools(two_rounds())).await;
    let flagged = nonstream(
        &app,
        body_with_tools(json!([
            user(json!("weather in tokyo?")),
            assistant(json!([tool_use_block("toolu_a", "get_weather", json!({"city": "Tokyo"}))])),
            user(json!([tool_result_err("toolu_a", json!("sunny 25C"), json!(true))])),
            assistant(json!([tool_use_block("toolu_b", "get_weather", json!({"city": "Osaka"}))])),
            user(json!([tool_result_err("toolu_b", json!("rain 18C"), json!(false))])),
            user(json!("thanks")),
        ])),
    )
    .await;
    assert_eq!(input_tokens(&flagged), input_tokens(&plain));
    assert_eq!(text_of(&flagged), text_of(&plain));
}
