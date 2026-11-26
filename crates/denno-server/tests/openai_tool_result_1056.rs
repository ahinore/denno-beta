//! task 1056: OpenAI 互換 `POST /v1/chat/completions` のマルチターン
//! tool-result (`role="tool"`) 履歴の監査テスト。
//!
//! `src/openai.rs` の `split_messages` 実装を読み、`role="tool"` 経路の
//! 実際の契約を固定する (非ストリーム / SSE の両方):
//!
//! - `tool_call_id` は受理されるが履歴テキストには使われない
//!   (一致・不一致・空・欠落・null で同じプロンプト、非文字列は serde の 400)
//! - `content` の空形 (`""` / null / 欠落 / `[]` / text 以外の part のみ) は
//!   すべて空の `<tool_response>` になり、part 配列は連結される
//! - assistant `tool_calls` の直後に tool 結果が続く形 (1 個 / 並列 2 個) が
//!   `<tool_call>` + `<tool_response>` の明示テキスト履歴と同じプロンプトになる
//! - `arguments` は JSON 文字列とオブジェクト直書きの両方を受ける。
//!   空文字 / null / 欠落は `{}`、非オブジェクトは 400 (`messages[i].tool_calls[j]`)
//! - 上記すべてについて SSE と非ストリームで本文・finish_reason・usage が一致する
//!
//! ダミートークナイザ + 決定的ダミーモデル (`tests/openai_chat.rs` と同一方式)
//! を使い、`src` は変更しない。

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

/// 同じ `messages` をストリームで投げる body。
fn stream_body_with(messages: Value) -> Value {
    let mut b = body_with(messages);
    b["stream"] = json!(true);
    b["stream_options"] = json!({"include_usage": true});
    b
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

/// assistant `tool_calls` 1 個分の履歴形 (content は null)。
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

/// `tool` メッセージの履歴形。
fn tool_msg(id: Value, content: Value) -> Value {
    json!({"role": "tool", "tool_call_id": id, "content": content})
}

/// `tool_use_to_xml` と同じ形。
fn expected_xml_named(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    for (k, v) in params {
        s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
    }
    s.push_str("</function>\n</tool_call>");
    s
}

fn expected_xml(name: &str, param: &str, value: &str) -> String {
    expected_xml_named(name, &[(param, value)])
}

/// `<tool_response>` 包み。
fn wrapped(body: &str) -> String {
    format!("<tool_response>\n{body}\n</tool_response>")
}

/// SSE 本文から `data:` 行の JSON を取り出す ([DONE] は除く)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

/// SSE 本文の delta.content を連結したもの。
fn sse_content(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

/// SSE 本文の usage chunk (`choices: []`)。
fn sse_usage_chunk(chunks: &[Value]) -> &Value {
    chunks
        .iter()
        .find(|c| c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(false))
        .expect("usage chunk が無い")
}

/// user → assistant `tool_calls`(2 個) → tool 2 件 → user。
fn parallel_tool_messages() -> Value {
    json!([
        simple("user", json!("weather in tokyo and osaka?")),
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [
                {"id": "call_t", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}},
                {"id": "call_o", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Osaka\"}"}},
            ],
        },
        tool_msg(json!("call_t"), json!("sunny 25C")),
        tool_msg(json!("call_o"), json!("rainy 18C")),
        simple("user", json!("thanks")),
    ])
}

// ---------------------------------------------------------------------------
// 1. 非ストリーム: 妥当な tool 履歴
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nonstream_tool_message_is_accepted_and_maps_to_tool_response() {
    let app = test_app();
    let messages = json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        tool_msg(json!("call_abc"), json!("sunny 25C")),
        simple("user", json!("thanks")),
    ]);
    let (status, v) = chat(&app, body_with(messages)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "chat.completion");
    assert!(prompt_tokens(&v) > 0);
    assert!(!content_of(&v).is_empty());
}

#[tokio::test]
async fn assistant_tool_call_followed_by_tool_result_matches_explicit_text() {
    let app = test_app();
    let implicit = json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        tool_msg(json!("call_abc"), json!("sunny 25C")),
        simple("user", json!("thanks")),
    ]);
    let explicit = json!([
        simple("user", json!("weather in tokyo?")),
        simple("assistant", json!(expected_xml("get_weather", "city", "Tokyo"))),
        simple("user", json!(wrapped("sunny 25C"))),
        simple("user", json!("thanks")),
    ]);
    let (s1, a) = chat(&app, body_with(implicit)).await;
    let (s2, b) = chat(&app, body_with(explicit)).await;
    assert_eq!(s1, StatusCode::OK, "implicit = {a}");
    assert_eq!(s2, StatusCode::OK, "explicit = {b}");
    assert_eq!(prompt_tokens(&a), prompt_tokens(&b));
    assert_eq!(content_of(&a), content_of(&b));
}

#[tokio::test]
async fn parallel_tool_calls_and_results_map_in_order() {
    let app = test_app();
    let xml_t = expected_xml("get_weather", "city", "Tokyo");
    let xml_o = expected_xml("get_weather", "city", "Osaka");
    let explicit = json!([
        simple("user", json!("weather in tokyo and osaka?")),
        simple("assistant", json!(format!("{xml_t}\n\n{xml_o}"))),
        simple("user", json!(wrapped("sunny 25C"))),
        simple("user", json!(wrapped("rainy 18C"))),
        simple("user", json!("thanks")),
    ]);
    let (s1, a) = chat(&app, body_with(parallel_tool_messages())).await;
    let (s2, b) = chat(&app, body_with(explicit)).await;
    assert_eq!(s1, StatusCode::OK, "implicit = {a}");
    assert_eq!(s2, StatusCode::OK, "explicit = {b}");
    assert_eq!(
        prompt_tokens(&a),
        prompt_tokens(&b),
        "並列 tool_calls / tool_response の順序が明示テキストと一致しない"
    );
    assert_eq!(content_of(&a), content_of(&b));
}

#[tokio::test]
async fn tool_only_history_is_accepted() {
    // tool は履歴上 user 発話へ写るため、tool 単独でも「user または assistant
    // が 1 件以上」の検証を通る (実装済み挙動の固定)。
    let app = test_app();
    let (status, v) = chat(
        &app,
        body_with(json!([tool_msg(json!("call_x"), json!("orphan result"))])),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let (s2, reference) = chat(&app, body_with(json!([simple("user", json!(wrapped("orphan result")))]))).await;
    assert_eq!(s2, StatusCode::OK, "reference = {reference}");
    assert_eq!(prompt_tokens(&v), prompt_tokens(&reference));
}

// ---------------------------------------------------------------------------
// 2. 非ストリーム: tool_call_id の扱い
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_call_id_variants_do_not_change_the_prompt() {
    let app = test_app();
    let (_, base) = chat(&app, body_with(parallel_tool_messages())).await;
    let base_pt = prompt_tokens(&base);

    // 一致 / 不一致 / 空 / 欠落 / null のどれでも同じプロンプト。
    let variants = [
        tool_msg(json!("call_t"), json!("sunny 25C")),
        tool_msg(json!("call_zzz"), json!("sunny 25C")),
        tool_msg(json!(""), json!("sunny 25C")),
        json!({"role": "tool", "content": "sunny 25C"}),
        tool_msg(Value::Null, json!("sunny 25C")),
    ];
    for first in variants {
        let messages = json!([
            simple("user", json!("weather in tokyo and osaka?")),
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {"id": "call_t", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Tokyo\"}"}},
                    {"id": "call_o", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Osaka\"}"}},
                ],
            },
            first,
            tool_msg(json!("call_o"), json!("rainy 18C")),
            simple("user", json!("thanks")),
        ]);
        let (status, v) = chat(&app, body_with(messages)).await;
        assert_eq!(status, StatusCode::OK, "body = {v}");
        assert_eq!(
            prompt_tokens(&v),
            base_pt,
            "tool_call_id が履歴テキストへ影響している"
        );
    }
}

#[tokio::test]
async fn non_string_tool_call_id_is_rejected_by_serde() {
    let app = test_app();
    let messages = json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        json!({"role": "tool", "tool_call_id": 12345, "content": "sunny 25C"}),
    ]);
    let (status, v) = chat(&app, body_with(messages)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("invalid type"), "message = {msg}");
    assert!(msg.contains("expected a string"), "message = {msg}");
}

// ---------------------------------------------------------------------------
// 3. 非ストリーム: tool の content (空 / 配列)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tool_content_empty_shapes_are_equivalent() {
    let app = test_app();
    // "" / null / 欠落 / [] / text 以外の part のみ はどれも空の tool_response。
    let empty_shapes: Vec<Value> = vec![
        json!({"role": "tool", "tool_call_id": "call_abc", "content": ""}),
        json!({"role": "tool", "tool_call_id": "call_abc", "content": null}),
        json!({"role": "tool", "tool_call_id": "call_abc"}),
        json!({"role": "tool", "tool_call_id": "call_abc", "content": []}),
        json!({"role": "tool", "tool_call_id": "call_abc",
               "content": [{"type": "image_url", "image_url": {"url": "ignored"}}]}),
    ];
    let mut baseline: Option<u64> = None;
    for tool in empty_shapes {
        let messages = json!([
            simple("user", json!("weather in tokyo?")),
            assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
            tool,
            simple("user", json!("thanks")),
        ]);
        let (status, v) = chat(&app, body_with(messages.clone())).await;
        assert_eq!(status, StatusCode::OK, "body = {v} messages = {messages}");
        let pt = prompt_tokens(&v);
        match baseline {
            None => {
                // 空の tool_response は明示テキストと同じ長さ
                let explicit = json!([
                    simple("user", json!("weather in tokyo?")),
                    simple("assistant", json!(expected_xml("get_weather", "city", "Tokyo"))),
                    simple("user", json!(wrapped(""))),
                    simple("user", json!("thanks")),
                ]);
                let (sx, ex) = chat(&app, body_with(explicit)).await;
                assert_eq!(sx, StatusCode::OK, "explicit = {ex}");
                assert_eq!(pt, prompt_tokens(&ex), "空 content の tool_response が不一致");
                baseline = Some(pt);
            }
            Some(b) => assert_eq!(pt, b, "空 content の形でプロンプト長が揺れている"),
        }
    }
}

#[tokio::test]
async fn tool_content_array_parts_are_concatenated() {
    let app = test_app();
    let parts = json!({"role": "tool", "tool_call_id": "call_abc", "content": [
        {"type": "text", "text": "sunny"},
        {"type": "text", "text": " 25C"},
    ]});
    let flat = tool_msg(json!("call_abc"), json!("sunny 25C"));
    let mk = |tool: Value| {
        json!([
            simple("user", json!("weather in tokyo?")),
            assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
            tool,
            simple("user", json!("thanks")),
        ])
    };
    let (s1, a) = chat(&app, body_with(mk(parts))).await;
    let (s2, b) = chat(&app, body_with(mk(flat))).await;
    assert_eq!(s1, StatusCode::OK, "parts = {a}");
    assert_eq!(s2, StatusCode::OK, "flat = {b}");
    assert_eq!(prompt_tokens(&a), prompt_tokens(&b));
    assert_eq!(content_of(&a), content_of(&b));
}

// ---------------------------------------------------------------------------
// 4. 非ストリーム: assistant tool_calls の arguments / 形の検証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn arguments_object_form_equals_json_string_form() {
    let app = test_app();
    let as_object = json!([
        simple("user", json!("weather in tokyo?")),
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"city": "Tokyo"}},
            }],
        },
        tool_msg(json!("call_abc"), json!("sunny 25C")),
    ]);
    let as_string = json!([
        simple("user", json!("weather in tokyo?")),
        assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
        tool_msg(json!("call_abc"), json!("sunny 25C")),
    ]);
    let (s1, a) = chat(&app, body_with(as_object)).await;
    let (s2, b) = chat(&app, body_with(as_string)).await;
    assert_eq!(s1, StatusCode::OK, "object = {a}");
    assert_eq!(s2, StatusCode::OK, "string = {b}");
    assert_eq!(prompt_tokens(&a), prompt_tokens(&b));
}

#[tokio::test]
async fn empty_or_missing_arguments_become_empty_object() {
    let app = test_app();
    // 空文字 / 空白 / null / 欠落 → `{}` (= parameter 無しの tool_call XML)
    let arg_shapes = [
        json!(""),
        json!("   "),
        Value::Null,
        json!("{}"),
    ];
    let xml = expected_xml_named("get_weather", &[]);
    let explicit = json!([
        simple("user", json!("weather in tokyo?")),
        simple("assistant", json!(xml)),
        simple("user", json!(wrapped("sunny 25C"))),
    ]);
    let (sx, ex) = chat(&app, body_with(explicit)).await;
    assert_eq!(sx, StatusCode::OK, "explicit = {ex}");
    for args in arg_shapes {
        let messages = json!([
            simple("user", json!("weather in tokyo?")),
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": args.clone()},
                }],
            },
            tool_msg(json!("call_abc"), json!("sunny 25C")),
        ]);
        let (status, v) = chat(&app, body_with(messages)).await;
        assert_eq!(status, StatusCode::OK, "arguments = {args} body = {v}");
        assert_eq!(prompt_tokens(&v), prompt_tokens(&ex), "arguments = {args}");
    }

    // arguments フィールド自体が無い場合も同じ
    let missing = json!([
        simple("user", json!("weather in tokyo?")),
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather"},
            }],
        },
        tool_msg(json!("call_abc"), json!("sunny 25C")),
    ]);
    let (status, v) = chat(&app, body_with(missing)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(prompt_tokens(&v), prompt_tokens(&ex));
}

#[tokio::test]
async fn non_object_arguments_are_rejected_with_tool_calls_param() {
    let app = test_app();
    let cases = [json!("[1,2]"), json!("\"text\""), json!("42"), json!([1, 2]), json!(42)];
    for args in cases {
        let messages = json!([
            simple("user", json!("hi")),
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "f", "arguments": args.clone()},
                }],
            },
        ]);
        let (status, v) = chat(&app, body_with(messages)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "arguments = {args} body = {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(
            v["error"]["param"], "messages[1].tool_calls[0]",
            "arguments = {args} body = {v}"
        );
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("must be a JSON object string"),
            "arguments = {args} body = {v}"
        );
    }
}

#[tokio::test]
async fn malformed_arguments_json_is_rejected() {
    let app = test_app();
    let messages = json!([
        simple("user", json!("hi")),
        assistant_tool_call("call_abc", "f", "{\"city\": "),
    ]);
    let (status, v) = chat(&app, body_with(messages)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "messages[1].tool_calls[0]");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not valid JSON"),
        "body = {v}"
    );
}

#[tokio::test]
async fn unsupported_tool_call_type_and_empty_name_are_rejected() {
    let app = test_app();
    // type != "function"
    let bad_type = json!([
        simple("user", json!("hi")),
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "web_search",
                "function": {"name": "f", "arguments": "{}"},
            }],
        },
    ]);
    let (status, v) = chat(&app, body_with(bad_type)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "messages[1].tool_calls[0]");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("unsupported value"),
        "body = {v}"
    );

    // 空 / 空白の function.name
    for name in ["", "   "] {
        let bad_name = json!([
            simple("user", json!("hi")),
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": name, "arguments": "{}"},
                }],
            },
        ]);
        let (status, v) = chat(&app, body_with(bad_name)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "name = {name:?} body = {v}");
        assert_eq!(v["error"]["param"], "messages[1].tool_calls[0]");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("non-empty string"),
            "name = {name:?} body = {v}"
        );
    }
}

#[tokio::test]
async fn tool_calls_on_non_assistant_messages_are_ignored() {
    let app = test_app();
    // user に付いた tool_calls は履歴に織り込まれない (assistant のみ解釈)。
    let with_user_calls = json!([
        {
            "role": "user",
            "content": "hi",
            "tool_calls": [{
                "id": "call_x",
                "type": "function",
                "function": {"name": "f", "arguments": "{\"a\":1}"},
            }],
        },
        tool_msg(json!("call_x"), json!("ok")),
    ]);
    let plain = json!([
        simple("user", json!("hi")),
        tool_msg(json!("call_x"), json!("ok")),
    ]);
    let (s1, a) = chat(&app, body_with(with_user_calls)).await;
    let (s2, b) = chat(&app, body_with(plain)).await;
    assert_eq!(s1, StatusCode::OK, "with calls = {a}");
    assert_eq!(s2, StatusCode::OK, "plain = {b}");
    assert_eq!(prompt_tokens(&a), prompt_tokens(&b));
}

// ---------------------------------------------------------------------------
// 5. SSE: tool 履歴のパリティ
// ---------------------------------------------------------------------------

/// 非ストリームと SSE を突き合わせる共通検査。
async fn assert_stream_matches_nonstream(app: &Router, messages: Value) {
    let (sn, nonstream) = chat(app, body_with(messages.clone())).await;
    assert_eq!(sn, StatusCode::OK, "non-stream = {nonstream}");

    let (ss, _, text) = post(app, "/v1/chat/completions", stream_body_with(messages)).await;
    assert_eq!(ss, StatusCode::OK, "stream text = {text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");

    let chunks = sse_chunks(&text);
    assert!(!chunks.is_empty(), "SSE chunk が無い");
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
    }
    // 先頭 chunk は delta.role = assistant
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(sse_content(&chunks), content_of(&nonstream));

    let last_finish = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["finish_reason"].as_str())
        .last()
        .expect("finish_reason が無い");
    assert_eq!(
        last_finish,
        nonstream["choices"][0]["finish_reason"].as_str().unwrap()
    );

    let usage = sse_usage_chunk(&chunks);
    assert_eq!(
        usage["usage"]["prompt_tokens"].as_u64(),
        Some(prompt_tokens(&nonstream))
    );
    assert_eq!(
        usage["usage"]["completion_tokens"],
        nonstream["usage"]["completion_tokens"]
    );
    assert_eq!(
        usage["usage"]["total_tokens"],
        nonstream["usage"]["total_tokens"]
    );
}

#[tokio::test]
async fn sse_parallel_tool_results_match_nonstream() {
    let app = test_app();
    assert_stream_matches_nonstream(&app, parallel_tool_messages()).await;
}

#[tokio::test]
async fn sse_tool_message_with_empty_content_matches_nonstream() {
    let app = test_app();
    for content in [json!(""), Value::Null, json!([])] {
        let messages = json!([
            simple("user", json!("weather in tokyo?")),
            assistant_tool_call("call_abc", "get_weather", "{\"city\":\"Tokyo\"}"),
            json!({"role": "tool", "tool_call_id": "call_abc", "content": content}),
            simple("user", json!("thanks")),
        ]);
        assert_stream_matches_nonstream(&app, messages).await;
    }
}

#[tokio::test]
async fn sse_tool_only_history_matches_nonstream() {
    let app = test_app();
    assert_stream_matches_nonstream(
        &app,
        json!([tool_msg(json!("call_x"), json!("orphan result"))]),
    )
    .await;
}

#[tokio::test]
async fn sse_prompt_tokens_equal_nonstream_for_tool_history() {
    let app = test_app();
    let messages = parallel_tool_messages();
    let (_, nonstream) = chat(&app, body_with(messages.clone())).await;
    let (ss, _, text) = post(&app, "/v1/chat/completions", stream_body_with(messages)).await;
    assert_eq!(ss, StatusCode::OK, "text = {text}");
    let chunks = sse_chunks(&text);
    assert_eq!(
        sse_usage_chunk(&chunks)["usage"]["prompt_tokens"].as_u64(),
        Some(prompt_tokens(&nonstream)),
        "SSE と非ストリームで tool 履歴のプロンプト長が違う"
    );
}

// ---------------------------------------------------------------------------
// 6. SSE: tool 履歴の不正形はストリーム開始前に 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sse_invalid_tool_history_is_rejected_before_streaming() {
    let app = test_app();

    // arguments が非オブジェクト
    let bad_args = json!([
        simple("user", json!("hi")),
        {
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "f", "arguments": "not json"},
            }],
        },
    ]);
    let (status, v, text) = post(&app, "/v1/chat/completions", stream_body_with(bad_args)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "text = {text}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert_eq!(v["error"]["param"], "messages[1].tool_calls[0]");
    assert!(!text.contains("data: "), "SSE が流れている: {text}");

    // tool_call_id が非文字列
    let bad_id = json!([
        simple("user", json!("hi")),
        json!({"role": "tool", "tool_call_id": 7, "content": "x"}),
    ]);
    let (status, v, text) = post(&app, "/v1/chat/completions", stream_body_with(bad_id)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "text = {text}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(!text.contains("data: "), "SSE が流れている: {text}");

    // 未知 role
    let bad_role = json!([simple("user", json!("hi")), simple("toolbox", json!("x"))]);
    let (status, v, text) = post(&app, "/v1/chat/completions", stream_body_with(bad_role)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "text = {text}");
    assert_eq!(v["error"]["param"], "messages[1].role");
    assert!(!text.contains("data: "), "SSE が流れている: {text}");
}

// ---------------------------------------------------------------------------
// 7. SSE: tools 宣言あり (アセンブラ経路) の tool 履歴
// ---------------------------------------------------------------------------

/// tool 履歴 + `tools` 宣言。SSE は StreamAssembler 経路を通る。
fn tools_body(messages: Value, tool_choice: Value) -> Value {
    let mut b = body_with(messages);
    b["tools"] = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]);
    b["tool_choice"] = tool_choice;
    b
}

#[tokio::test]
async fn sse_with_declared_tools_matches_nonstream_for_tool_history() {
    let app = test_app();
    let messages = parallel_tool_messages();

    let (sn, nonstream) = chat(&app, tools_body(messages.clone(), json!("auto"))).await;
    assert_eq!(sn, StatusCode::OK, "non-stream = {nonstream}");

    let mut stream = tools_body(messages.clone(), json!("auto"));
    stream["stream"] = json!(true);
    stream["stream_options"] = json!({"include_usage": true});
    let (ss, _, text) = post(&app, "/v1/chat/completions", stream).await;
    assert_eq!(ss, StatusCode::OK, "stream text = {text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");

    let chunks = sse_chunks(&text);
    assert!(!chunks.is_empty(), "SSE chunk が無い");
    assert_eq!(sse_content(&chunks), content_of(&nonstream), "delta 結合 != 非ストリーム本文");
    let last_finish = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["finish_reason"].as_str())
        .last()
        .expect("finish_reason が無い");
    assert_eq!(
        last_finish,
        nonstream["choices"][0]["finish_reason"].as_str().unwrap()
    );
    let usage = sse_usage_chunk(&chunks);
    assert_eq!(
        usage["usage"]["prompt_tokens"].as_u64(),
        Some(prompt_tokens(&nonstream))
    );
    assert_eq!(
        usage["usage"]["completion_tokens"],
        nonstream["usage"]["completion_tokens"]
    );
}

#[tokio::test]
async fn tools_declared_do_not_change_tool_history_prompt() {
    // tools 宣言はプロンプト先頭の tool 定義を足すだけで、履歴の写像は変えない。
    // tool_choice=none は tools_active=false (定義文を足さない) なので、
    // auto との差は宣言分だけになる。
    let app = test_app();
    let messages = parallel_tool_messages();
    let (s1, with_tools) = chat(&app, tools_body(messages.clone(), json!("auto"))).await;
    assert_eq!(s1, StatusCode::OK, "with tools = {with_tools}");
    let (s2, without_tools) = chat(&app, body_with(messages)).await;
    assert_eq!(s2, StatusCode::OK, "without tools = {without_tools}");
    let (s3, none) = chat(&app, tools_body(parallel_tool_messages(), json!("none"))).await;
    assert_eq!(s3, StatusCode::OK, "none = {none}");
    assert_eq!(
        prompt_tokens(&without_tools),
        prompt_tokens(&none),
        "tool_choice=none でも履歴の写像が変わっている"
    );
    assert!(
        prompt_tokens(&with_tools) >= prompt_tokens(&without_tools),
        "tools 宣言でプロンプトが短くなった: {} < {}",
        prompt_tokens(&with_tools),
        prompt_tokens(&without_tools)
    );
}
