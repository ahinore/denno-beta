//! task 1319: OpenAI 互換 `POST /v1/chat/completions` のマルチターン
//! tool-result 履歴と role / content 検証の契約テスト。
//!
//! `src/openai.rs` の実装済み契約をワイヤレベルで固定する:
//!
//! - 妥当な multi-turn (user → assistant `tool_calls` → `tool` `tool_call_id`
//!   → assistant) が受理され、履歴が `/v1/messages` と同じ
//!   `<tool_call>` / `<tool_response>` テキストへ写ること
//!   (`usage.prompt_tokens` が「同じテキストを明示したリクエスト」と一致)
//! - 未知 role は 400 + OpenAI エラー封筒 (`error.type = invalid_request_error`)
//! - `messages: []` / system のみ / `messages` 欠落の 400
//! - 空・空白・null・空 parts の content は受理される (実装済み挙動)
//! - `tool_call_id` は受理されるが履歴には使われない (欠落・不一致・重複でも
//!   同じプロンプトになる)
//! - ストリーム / 非ストリームの本文・finish_reason・usage が一致する
//!
//! `denno-engine` のダミートークナイザ + 決定的ダミーモデル
//! (`tests/openai_chat.rs` と同一方式) を使う。src は変更しない。

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

/// 決定的ダミーモデル (直前トークン + 1 をピークにする)。EOS の位置だけ避ける。
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

fn test_app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(SeqModel), tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    denno_server::router(Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    }))
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v, text)
}

async fn chat(app: &Router, body: Value) -> (StatusCode, Value) {
    let (status, v, _) = post(app, "/v1/chat/completions", body).await;
    (status, v)
}

/// `messages` だけを差し替えた既定リクエスト。
fn body_with(messages: Value) -> Value {
    json!({
        "model": "denno-test",
        "messages": messages,
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

/// `{role, content}` の素朴なメッセージ列。
fn simple(role: &str, content: Value) -> Value {
    json!({"role": role, "content": content})
}

/// 応答の `usage.prompt_tokens` (200 のときだけ)。
fn prompt_tokens(v: &Value) -> u64 {
    v["usage"]["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("usage.prompt_tokens が無い: {v}"))
}

/// 応答本文 (`choices[0].message.content`)。
fn content_of(v: &Value) -> &str {
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("message.content が文字列ではない: {v}"))
}

/// assistant `tool_calls` 1 個分の履歴形。
fn assistant_tool_call(id: &str, name: &str, args: &str) -> Value {
    json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": id,
            "type": "function",
            "function": {"name": name, "arguments": args},
        }],
    })
}

/// `tool_use_to_xml` と同じ形 (`src/lib.rs` の契約)。
fn expected_xml(name: &str, param: &str, value: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"
    )
}

/// マルチターン tool 履歴の本文 (user → assistant tool_calls → tool → user)。
fn multi_turn_messages() -> Value {
    json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        json!({"role": "tool", "tool_call_id": "call_abc", "content": "sunny 25C"}),
        simple("user", json!("thanks")),
    ])
}

/// SSE 本文から `data:` 行の JSON を取り出す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

// ---------------------------------------------------------------------------
// 1. 妥当な multi-turn tool-result 履歴
// ---------------------------------------------------------------------------

#[tokio::test]
async fn valid_multi_turn_tool_result_history_is_accepted() {
    let app = test_app();
    let (status, v) = chat(&app, body_with(multi_turn_messages())).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert!(!content_of(&v).is_empty());
    let fr = v["choices"][0]["finish_reason"].as_str().unwrap();
    assert!(fr == "stop" || fr == "length", "finish_reason = {fr}");
    assert!(prompt_tokens(&v) > 0);
}

#[tokio::test]
async fn tool_history_maps_to_the_same_prompt_as_explicit_text() {
    let app = test_app();
    // assistant `tool_calls` → `<tool_call>` XML、`tool` → `<tool_response>` 包み。
    // 同じテキストを content に明示した履歴と同じプロンプト (同じ prompt_tokens、
    // ダミーモデルはプロンプト最終トークンだけで出力が決まるので本文も一致する)。
    let explicit = json!([
        simple("user", json!("weather in tokyo?")),
        simple("assistant", json!(expected_xml("get_weather", "city", "Tokyo"))),
        simple("user", json!("<tool_response>\nsunny 25C\n</tool_response>")),
        simple("user", json!("thanks")),
    ]);
    let (s1, implicit) = chat(&app, body_with(multi_turn_messages())).await;
    let (s2, explicit) = chat(&app, body_with(explicit)).await;
    assert_eq!(s1, StatusCode::OK, "implicit = {implicit}");
    assert_eq!(s2, StatusCode::OK, "explicit = {explicit}");
    assert_eq!(
        prompt_tokens(&implicit),
        prompt_tokens(&explicit),
        "tool 履歴が明示テキストと同じプロンプトになっていない"
    );
    assert_eq!(content_of(&implicit), content_of(&explicit));
}

#[tokio::test]
async fn multi_turn_tool_history_matches_anthropic_tool_blocks() {
    let app = test_app();
    let anthropic = json!({
        "model": "denno-test",
        "messages": [
            simple("user", json!("weather in tokyo?")),
            json!({"role": "assistant", "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Tokyo"},
            }]}),
            json!({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "sunny 25C",
            }]}),
            simple("user", json!("thanks")),
        ],
        "max_tokens": 8,
        "temperature": 0.0,
    });
    let (so, openai) = chat(&app, body_with(multi_turn_messages())).await;
    let (sa, anth, _) = post(&app, "/v1/messages", anthropic).await;
    assert_eq!(so, StatusCode::OK, "openai = {openai}");
    assert_eq!(sa, StatusCode::OK, "anthropic = {anth}");
    assert_eq!(
        prompt_tokens(&openai),
        anth["usage"]["input_tokens"].as_u64().unwrap(),
        "/v1/messages と同じ履歴テキストになっていない"
    );
    assert_eq!(content_of(&openai), anth["content"][0]["text"]);
}

#[tokio::test]
async fn assistant_text_and_tool_calls_are_joined_with_a_blank_line() {
    let app = test_app();
    let mixed = json!([
        simple("user", json!("weather in tokyo?")),
        json!({
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"},
            }],
        }),
        json!({"role": "tool", "tool_call_id": "call_abc", "content": "sunny 25C"}),
        simple("user", json!("thanks")),
    ]);
    let text_form = format!(
        "Let me check.\n\n{}",
        expected_xml("get_weather", "city", "Tokyo")
    );
    let explicit = json!([
        simple("user", json!("weather in tokyo?")),
        simple("assistant", json!(text_form)),
        simple("user", json!("<tool_response>\nsunny 25C\n</tool_response>")),
        simple("user", json!("thanks")),
    ]);
    let (s1, implicit) = chat(&app, body_with(mixed)).await;
    let (s2, explicit) = chat(&app, body_with(explicit)).await;
    assert_eq!(s1, StatusCode::OK, "implicit = {implicit}");
    assert_eq!(s2, StatusCode::OK, "explicit = {explicit}");
    assert_eq!(prompt_tokens(&implicit), prompt_tokens(&explicit));
}

// ---------------------------------------------------------------------------
// 2. 未知 role の拒否 (OpenAI エラー封筒)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_role_is_rejected_with_openai_error_envelope() {
    let app = test_app();
    let (status, v) = chat(
        &app,
        body_with(json!([
            simple("user", json!("hi")),
            simple("wizard", json!("spells")),
        ])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    let err = &v["error"];
    assert_eq!(err["type"], "invalid_request_error");
    assert_eq!(err["param"], "messages[1].role");
    assert!(err["code"].is_null(), "error.code = {}", err["code"]);
    let msg = err["message"].as_str().unwrap();
    assert!(msg.contains("unsupported role"), "message = {msg}");
    assert!(msg.contains("wizard"), "message = {msg}");
    // 封筒は 4 フィールド固定 (message / type / param / code)
    let keys: Vec<&String> = err.as_object().unwrap().keys().collect();
    assert_eq!(keys.len(), 4, "error = {err}");
}

#[tokio::test]
async fn empty_and_mixed_case_roles_are_rejected() {
    let app = test_app();
    for role in ["", "User", "ASSISTANT", "system\n"] {
        let (status, v) = chat(&app, body_with(json!([simple(role, json!("hi"))]))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "role = {role:?} body = {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "messages[0].role");
    }
}

// ---------------------------------------------------------------------------
// 3. messages 配列 / content のエッジ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_messages_array_is_rejected() {
    let app = test_app();
    let (status, v) = chat(&app, body_with(json!([]))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "messages");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("at least one message"),
        "body = {v}"
    );
}

#[tokio::test]
async fn system_only_messages_are_rejected() {
    let app = test_app();
    for role in ["system", "developer"] {
        let (status, v) = chat(&app, body_with(json!([simple(role, json!("sys"))]))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "role = {role} body = {v}");
        assert_eq!(v["error"]["param"], "messages");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("at least one user or assistant message"),
            "body = {v}"
        );
    }
}

#[tokio::test]
async fn missing_messages_field_is_rejected() {
    let app = test_app();
    let (status, v) = chat(&app, json!({"model": "denno-test"})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("messages"),
        "body = {v}"
    );
}

#[tokio::test]
async fn empty_and_whitespace_content_is_accepted() {
    let app = test_app();
    // 実装済み挙動: content の空 / null / 空文字列 / 空 parts / 非 text part のみは
    // すべて空テキストとして受理される (user メッセージ自体は存在するため)。
    let variants = [
        json!(""),
        json!(null),
        json!("   "),
        json!([]),
        json!([{"type": "image_url", "image_url": {"url": "ignored"}}]),
    ];
    let mut baseline: Option<u64> = None;
    for content in variants {
        let (status, v) = chat(&app, body_with(json!([simple("user", content.clone())]))).await;
        assert_eq!(status, StatusCode::OK, "content = {content} body = {v}");
        let pt = prompt_tokens(&v);
        match baseline {
            None => baseline = Some(pt),
            Some(b) => assert_eq!(pt, b, "空 content のプロンプト長が揃っていない: {content}"),
        }
    }
}

#[tokio::test]
async fn text_parts_and_scalar_content_are_normalized() {
    let app = test_app();
    let (_, plain) = chat(&app, body_with(json!([simple("user", json!("t1"))]))).await;
    let (s1, parts) = chat(
        &app,
        body_with(json!([simple("user", json!([{"type": "text", "text": "t1"}]))])),
    )
    .await;
    assert_eq!(s1, StatusCode::OK, "parts = {parts}");
    assert_eq!(prompt_tokens(&plain), prompt_tokens(&parts));

    // 複数 text part は連結される (区切り無し)
    let (s2, joined) = chat(
        &app,
        body_with(json!([simple(
            "user",
            json!([{"type": "text", "text": "t1"}, {"type": "text", "text": "t2"}])
        )])),
    )
    .await;
    let (s3, flat) = chat(&app, body_with(json!([simple("user", json!("t1t2"))]))).await;
    assert_eq!(s2, StatusCode::OK, "joined = {joined}");
    assert_eq!(s3, StatusCode::OK, "flat = {flat}");
    assert_eq!(prompt_tokens(&joined), prompt_tokens(&flat));

    // 数値 content は文字列化される
    let (s4, num) = chat(&app, body_with(json!([simple("user", json!(42))]))).await;
    let (s5, stringy) = chat(&app, body_with(json!([simple("user", json!("42"))]))).await;
    assert_eq!(s4, StatusCode::OK, "num = {num}");
    assert_eq!(s5, StatusCode::OK, "stringy = {stringy}");
    assert_eq!(prompt_tokens(&num), prompt_tokens(&stringy));
}

// ---------------------------------------------------------------------------
// 4. tool_call_id の扱い (受理のみ、履歴テキストには使わない)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_call_id_is_accepted_but_unused_in_history() {
    let app = test_app();
    let (_, matching) = chat(&app, body_with(multi_turn_messages())).await;
    let base_pt = prompt_tokens(&matching);

    // 欠落 / 不一致 / 空文字 のいずれも同じプロンプト (id は履歴に織り込まれない)
    let variants = [
        json!({"role": "tool", "content": "sunny 25C"}),
        json!({"role": "tool", "tool_call_id": "call_unknown", "content": "sunny 25C"}),
        json!({"role": "tool", "tool_call_id": "", "content": "sunny 25C"}),
    ];
    for tool_msg in variants {
        let messages = json!([
            simple("user", json!("weather in tokyo?")),
            assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
            tool_msg,
            simple("user", json!("thanks")),
        ]);
        let (status, v) = chat(&app, body_with(messages)).await;
        assert_eq!(status, StatusCode::OK, "body = {v}");
        assert_eq!(
            prompt_tokens(&v),
            base_pt,
            "tool_call_id がプロンプトへ影響している"
        );
    }

    // 非文字列の tool_call_id は serde の型エラーで 400 (受理は string のみ)。
    // エラー本文は serde の汎用文言 (フィールド名は含まない) になる。
    let bad = json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        json!({"role": "tool", "tool_call_id": 12345, "content": "sunny 25C"}),
        simple("user", json!("thanks")),
    ]);
    let (status, v) = chat(&app, body_with(bad)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("invalid type"), "message = {msg}");
    assert!(msg.contains("expected a string"), "message = {msg}");
}

#[tokio::test]
async fn duplicate_tool_call_ids_are_accepted() {
    let app = test_app();
    let messages = json!([
        simple("user", json!("two cities")),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call_dup", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}},
                {"id": "call_dup", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Osaka\"}"}},
            ],
        }),
        json!({"role": "tool", "tool_call_id": "call_dup", "content": "sunny"}),
        json!({"role": "tool", "tool_call_id": "call_dup", "content": "rainy"}),
        simple("user", json!("thanks")),
    ]);
    let (status, v) = chat(&app, body_with(messages)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");

    // 2 個の assistant tool_call は順序どおり XML 連結、tool 2 件は 2 個の
    // `<tool_response>` になる (重複 id でも受理)。
    let xml0 = expected_xml("get_weather", "city", "Tokyo");
    let xml1 = expected_xml("get_weather", "city", "Osaka");
    let explicit = json!([
        simple("user", json!("two cities")),
        simple("assistant", json!(format!("{xml0}\n\n{xml1}"))),
        simple("user", json!("<tool_response>\nsunny\n</tool_response>")),
        simple("user", json!("<tool_response>\nrainy\n</tool_response>")),
        simple("user", json!("thanks")),
    ]);
    let (s2, v2) = chat(&app, body_with(explicit)).await;
    assert_eq!(s2, StatusCode::OK, "explicit = {v2}");
    assert_eq!(prompt_tokens(&v), prompt_tokens(&v2));
}

#[tokio::test]
async fn function_role_is_equivalent_to_tool_role() {
    let app = test_app();
    let (s1, tool_v) = chat(&app, body_with(multi_turn_messages())).await;
    let as_function = json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        json!({"role": "function", "tool_call_id": "call_abc", "content": "sunny 25C"}),
        simple("user", json!("thanks")),
    ]);
    let (s2, fn_v) = chat(&app, body_with(as_function)).await;
    assert_eq!(s1, StatusCode::OK, "tool = {tool_v}");
    assert_eq!(s2, StatusCode::OK, "function = {fn_v}");
    assert_eq!(prompt_tokens(&tool_v), prompt_tokens(&fn_v));
}

// ---------------------------------------------------------------------------
// 5. ストリーム / 非ストリームの一致
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_and_non_streaming_parity_for_tool_result_history() {
    let app = test_app();
    let (sn, non_stream) = chat(&app, body_with(multi_turn_messages())).await;
    assert_eq!(sn, StatusCode::OK, "non-stream = {non_stream}");

    let mut stream_body = body_with(multi_turn_messages());
    stream_body["stream"] = json!(true);
    stream_body["stream_options"] = json!({"include_usage": true});
    let (ss, text) = {
        let (status, _, text) = post(&app, "/v1/chat/completions", stream_body).await;
        (status, text)
    };
    assert_eq!(ss, StatusCode::OK, "stream text = {text}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "text = {text}"
    );

    let chunks = sse_chunks(&text);
    assert!(!chunks.is_empty(), "SSE chunk が無い");
    // 全 chunk が chat.completion.chunk
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
    }
    let streamed: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(streamed, content_of(&non_stream), "delta 結合 != 非ストリーム本文");

    // finish_reason は最終 choice chunk で確定する
    let last_finish = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["finish_reason"].as_str())
        .last()
        .expect("finish_reason が無い");
    assert_eq!(
        last_finish,
        non_stream["choices"][0]["finish_reason"].as_str().unwrap()
    );

    // include_usage の usage chunk (choices 空) が usage を一致させる
    let usage_chunk = chunks
        .iter()
        .find(|c| c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(false))
        .expect("usage chunk が無い");
    assert_eq!(
        usage_chunk["usage"]["prompt_tokens"].as_u64(),
        Some(prompt_tokens(&non_stream))
    );
    assert_eq!(
        usage_chunk["usage"]["completion_tokens"],
        non_stream["usage"]["completion_tokens"]
    );
    assert_eq!(
        usage_chunk["usage"]["total_tokens"],
        non_stream["usage"]["total_tokens"]
    );
}

#[tokio::test]
async fn streaming_unknown_role_is_rejected_before_streaming() {
    let app = test_app();
    let mut body = body_with(json!([simple("user", json!("hi")), simple("wizard", json!("x"))]));
    body["stream"] = json!(true);
    let (status, v, text) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "text = {text}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "messages[1].role");
    assert!(!text.contains("data: "), "SSE が流れている: {text}");

    // 空 messages も同じくストリーム前に 400
    let mut empty = body_with(json!([]));
    empty["stream"] = json!(true);
    let (status, v, text) = post(&app, "/v1/chat/completions", empty).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "text = {text}");
    assert_eq!(v["error"]["param"], "messages");
}
