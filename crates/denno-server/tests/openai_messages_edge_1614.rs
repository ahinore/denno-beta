//! task 1614: OpenAI `/v1/chat/completions` の `messages` 配列エッジ。
//!
//! `crates/denno-server/src/openai.rs` の [`split_messages`] / [`content_to_text`]
//! が定める受理規則を HTTP ワイヤで固定する:
//!
//! - `messages: []` は 400 `error.param = "messages"`、
//!   system / developer だけの配列も 400 (user / assistant が 1 件も無い)
//! - 複数の system / developer は配列順に 1 本の system へ連結され、
//!   改行で連結した単一 system とプロンプト・出力が一致する
//! - content の parts 配列は text / input_text だけを連結し、image_url 等の
//!   非テキスト part と非文字列要素は無視する (base64 の長さに依存しない)
//! - content が null / 空文字列でも受理され、同じ空プロンプトになる
//! - 連続する同一 role の発話はマージされず、ターンごとに保持される
//! - 未知 role は実際の配列 index 付き 400、既知の別名 (tool / function) は
//!   user の `<tool_response>` 包みとして受理される
//!
//! fixture は他の OpenAI テストと同じ (dummy モデル + dummy トークナイザ、
//! GPU 不要)。temperature 0 の greedy なので同一プロンプトは同一出力になる。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const CHAT: &str = "/v1/chat/completions";

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
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
        .uri(CHAT)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn body(messages: Value) -> Value {
    json!({
        "model": "denno-test",
        "messages": messages,
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

fn prompt_tokens(v: &Value) -> u64 {
    v["usage"]["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("usage.prompt_tokens missing: {v}"))
}

fn completion_text(v: &Value) -> &str {
    v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_else(|| panic!("choices[0].message.content missing: {v}"))
}

fn assert_invalid(v: &Value, param: &str) {
    assert_eq!(v["error"]["type"], "invalid_request_error", "body: {v}");
    assert_eq!(v["error"]["param"], param, "body: {v}");
    assert!(v["error"]["message"].is_string(), "body: {v}");
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_messages_array_is_400_with_messages_param() {
    let (status, v) = post(&app(), body(json!([]))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {v}");
    assert_invalid(&v, "messages");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("at least one message"), "message: {msg}");
}

#[tokio::test]
async fn messages_with_only_system_roles_are_400() {
    // system / developer は system プロンプトへ畳まれるため、会話本体が空になる。
    let (status, v) = post(
        &app(),
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "developer", "content": "t1"},
        ])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {v}");
    assert_invalid(&v, "messages");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("at least one user or assistant"),
        "message: {msg}"
    );
}

#[tokio::test]
async fn multiple_system_and_developer_messages_join_in_order() {
    let app = app();

    // 3 本の system / developer メッセージ (配列順)。
    let (status_split, split) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "developer", "content": "t1"},
            {"role": "system", "content": "t2"},
            {"role": "user", "content": "t3"},
        ])),
    )
    .await;
    assert_eq!(status_split, StatusCode::OK, "body: {split}");

    // 改行 (`"\n"`) で連結した単一 system と同一プロンプト。
    let (status_joined, joined) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0\nt1\nt2"},
            {"role": "user", "content": "t3"},
        ])),
    )
    .await;
    assert_eq!(status_joined, StatusCode::OK, "body: {joined}");

    assert_eq!(prompt_tokens(&split), prompt_tokens(&joined));
    assert_eq!(completion_text(&split), completion_text(&joined));
    assert_eq!(split["usage"], joined["usage"]);
    assert_eq!(split["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn mixed_content_parts_use_only_text_and_input_text() {
    let app = app();

    let parts = json!([
        {"type": "text", "text": "t0"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "input_text", "text": " t1"},
        {"type": "audio", "audio": {"data": "x"}},
        7,
        null,
        {"type": "text"}
    ]);
    let (status_parts, parts_v) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "user", "content": parts},
        ])),
    )
    .await;
    assert_eq!(status_parts, StatusCode::OK, "body: {parts_v}");

    // 非テキスト part を除いた同じ文字列。
    let (status_plain, plain) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "user", "content": "t0 t1"},
        ])),
    )
    .await;
    assert_eq!(status_plain, StatusCode::OK, "body: {plain}");

    assert_eq!(prompt_tokens(&parts_v), prompt_tokens(&plain));
    assert_eq!(completion_text(&parts_v), completion_text(&plain));
}

#[tokio::test]
async fn non_text_part_payload_size_does_not_affect_prompt() {
    let app = app();
    let small = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "t0"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]);
    // 同じ形で base64 だけ巨大化しても text part 以外は捨てられる。
    let long_url = format!("data:image/png;base64,{}", "A".repeat(4096));
    let large = json!([
        {"role": "user", "content": [
            {"type": "text", "text": "t0"},
            {"type": "image_url", "image_url": {"url": long_url}},
        ]},
    ]);

    let (status_small, small_v) = post(&app, body(small)).await;
    let (status_large, large_v) = post(&app, body(large)).await;
    assert_eq!(status_small, StatusCode::OK, "body: {small_v}");
    assert_eq!(status_large, StatusCode::OK, "body: {large_v}");
    assert_eq!(prompt_tokens(&small_v), prompt_tokens(&large_v));
    assert_eq!(completion_text(&small_v), completion_text(&large_v));
}

#[tokio::test]
async fn null_and_empty_string_content_are_equivalent() {
    let app = app();
    let (status_null, null_v) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "user", "content": null},
        ])),
    )
    .await;
    let (status_empty, empty_v) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "user", "content": ""},
        ])),
    )
    .await;
    assert_eq!(status_null, StatusCode::OK, "body: {null_v}");
    assert_eq!(status_empty, StatusCode::OK, "body: {empty_v}");
    assert_eq!(prompt_tokens(&null_v), prompt_tokens(&empty_v));
    assert_eq!(completion_text(&null_v), completion_text(&empty_v));
}

#[tokio::test]
async fn consecutive_same_role_turns_are_kept_separate() {
    let app = app();
    let (status_sep, sep) = post(
        &app,
        body(json!([
            {"role": "user", "content": "t0"},
            {"role": "user", "content": "t1"},
        ])),
    )
    .await;
    let (status_merged, merged) = post(
        &app,
        body(json!([{"role": "user", "content": "t0 t1"}])),
    )
    .await;
    assert_eq!(status_sep, StatusCode::OK, "body: {sep}");
    assert_eq!(status_merged, StatusCode::OK, "body: {merged}");

    // ターンごとに role 見出しが付くため、マージ形より長いプロンプトになる。
    assert!(
        prompt_tokens(&sep) > prompt_tokens(&merged),
        "separate={} merged={}",
        prompt_tokens(&sep),
        prompt_tokens(&merged)
    );
    // 最後のトークンが同じなので生成結果は一致する。
    assert_eq!(completion_text(&sep), completion_text(&merged));

    // assistant の連続も同様に保持される。
    let (status_asst, asst) = post(
        &app,
        body(json!([
            {"role": "user", "content": "t0"},
            {"role": "assistant", "content": "t1"},
            {"role": "assistant", "content": "t2"},
            {"role": "user", "content": "t3"},
        ])),
    )
    .await;
    assert_eq!(status_asst, StatusCode::OK, "body: {asst}");
    assert_eq!(asst["choices"][0]["finish_reason"], "stop");

    // 3 ターンに増やすと 2 ターンよりさらに長くなる (順序どおりに積まれる)。
    let (status_three, three) = post(
        &app,
        body(json!([
            {"role": "user", "content": "t0"},
            {"role": "user", "content": "t1"},
            {"role": "user", "content": "t2"},
        ])),
    )
    .await;
    assert_eq!(status_three, StatusCode::OK, "body: {three}");
    assert!(prompt_tokens(&three) > prompt_tokens(&sep));
}

#[tokio::test]
async fn unknown_role_is_400_with_actual_array_index() {
    let app = app();

    let (status, v) = post(
        &app,
        body(json!([
            {"role": "user", "content": "t0"},
            {"role": "critic", "content": "t1"},
        ])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body: {v}");
    assert_invalid(&v, "messages[1].role");
    let msg = v["error"]["message"].as_str().unwrap();
    assert!(msg.contains("unsupported role"), "message: {msg}");
    assert!(msg.contains("critic"), "message: {msg}");

    // system を挟んでも param の index は元の配列位置を指す。
    let (status2, v2) = post(
        &app,
        body(json!([
            {"role": "system", "content": "t0"},
            {"role": "user", "content": "t1"},
            {"role": "", "content": "t2"},
        ])),
    )
    .await;
    assert_eq!(status2, StatusCode::BAD_REQUEST, "body: {v2}");
    assert_invalid(&v2, "messages[2].role");
}

#[tokio::test]
async fn tool_and_function_roles_are_known_aliases() {
    let app = app();

    // tool / function は user の `<tool_response>` 包みへ写される。
    let (status_alias, alias) = post(
        &app,
        body(json!([
            {"role": "user", "content": "t0"},
            {"role": "tool", "tool_call_id": "call_x", "content": "t1"},
            {"role": "function", "content": "t2"},
        ])),
    )
    .await;
    assert_eq!(status_alias, StatusCode::OK, "body: {alias}");

    let (status_plain, plain) = post(
        &app,
        body(json!([
            {"role": "user", "content": "t0"},
            {"role": "user", "content": "<tool_response>\nt1\n</tool_response>"},
            {"role": "user", "content": "<tool_response>\nt2\n</tool_response>"},
        ])),
    )
    .await;
    assert_eq!(status_plain, StatusCode::OK, "body: {plain}");
    assert_eq!(prompt_tokens(&alias), prompt_tokens(&plain));
    assert_eq!(completion_text(&alias), completion_text(&plain));
}
