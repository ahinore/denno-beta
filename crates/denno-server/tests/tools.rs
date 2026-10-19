//! tool use (Anthropic API tools/tool_choice 対応) の HTTP ワイヤレベル統合テスト。
//!
//! denno-engine のダミーモデル/ダミートークナイザを使用 (models/scheduler.rs と同一方式)。
//! ダミーモデルは `` を出力しないため、tool_use block の応答生成は
//! lib.rs の unit テスト (抽出関数 + message_json) で検証し、ここでは
//! リクエスト検証 (400) と tools 付きリクエストが生成経路まで通ることを検証する。
//!
//! カバー項目:
//! - tools 付きリクエストが受理され従来と同一の content blocks で応答される
//! - tool_choice none (ツール非公開) も受理される
//! - tools + stream=true は SSE で受理される (第27: streaming tool_use 対応。
//!   ダミーモデルは tool_call を出さないため text block のみの SSE を検証。
//!   tool block の組み立ては lib.rs / stream_asm.rs の unit テスト)
//! - tool_choice 単独指定 / 不正な tools 定義 / 不正な tool_choice は 400
//! - tool_result のみの 2 ターン目が受理される (履歴織り込み経路)
//! - 未知 tool_use_id 参照の tool_result は 400

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn test_app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
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
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
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

/// ダミーモデルの greedy 生成の期待値 (models/scheduler.rs と同一)
const EXPECTED_GREEDY_TEXT: &str = "t5 t6 t7 t8 t9 t10 t11";

#[tokio::test]
async fn tools_request_is_accepted_and_returns_legacy_blocks() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
            "tools": tools_json(),
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["type"], "message");
    // ダミーモデルは tool_call を出力しない → 従来どおり text block のみ
    assert_eq!(body["content"], json!([{"type": "text", "text": EXPECTED_GREEDY_TEXT}]));
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["stop_sequence"], Value::Null);
}

#[tokio::test]
async fn tool_choice_none_is_accepted() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
            "tools": tools_json(),
            "tool_choice": "none",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["content"][0]["text"], EXPECTED_GREEDY_TEXT);
}

/// tools + stream=true が受理され、text block のみの完全な SSE を返す。
/// (第27: 旧来は 400 拒否だったが streaming tool_use 対応で解禁。
///  ダミーモデルは tool_call を出さないため delta 結合 = greedy 期待値)
#[tokio::test]
async fn tools_with_stream_is_accepted() {
    let app = test_app();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "denno-test",
                "max_tokens": 20,
                "temperature": 0.0,
                "stream": true,
                "messages": [{"role": "user", "content": "t0"}],
                "tools": tools_json(),
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    // 完全な SSE イベント列 (event: 行必須)
    for ty in ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"] {
        assert!(text.contains(&format!("event: {ty}\n")), "missing event: {ty} in:\n{text}");
    }
    // tool block 無し → tool_use block は出ない
    assert!(!text.contains("\"tool_use\""), "unexpected tool_use block in:\n{text}");
    // delta 結合が greedy 期待値と一致
    let mut out = String::new();
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if v["type"] == "content_block_delta" {
                    if let Some(t) = v["delta"]["text"].as_str() {
                        out.push_str(t);
                    }
                }
            }
        }
    }
    assert_eq!(out, EXPECTED_GREEDY_TEXT, "delta 結合が期待値と不一致:\n{text}");
    assert!(text.contains("\"stop_reason\":\"end_turn\""));
}

#[tokio::test]
async fn tool_choice_without_tools_is_rejected() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "t0"}],
            "tool_choice": "auto",
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn invalid_tool_definition_is_rejected() {
    let app = test_app();
    // input_schema が object 以外
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "t0"}],
            "tools": [{"name": "f", "input_schema": "not an object"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {body}");

    // input_schema 欠落
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "t0"}],
            "tools": [{"name": "f"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {body}");
}

#[tokio::test]
async fn invalid_tool_choice_is_rejected() {
    let app = test_app();
    for choice in [json!("bogus"), json!({"type": "tool"})] {
        let (status, body) = post(
            &app,
            json!({
                "model": "denno-test",
                "max_tokens": 20,
                "messages": [{"role": "user", "content": "t0"}],
                "tools": tools_json(),
                "tool_choice": choice,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "choice = {choice}, body = {body}");
    }
    // 未定義ツールの指定
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "t0"}],
            "tools": tools_json(),
            "tool_choice": {"type": "tool", "name": "no_such_tool"},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {body}");
}

#[tokio::test]
async fn tool_result_second_turn_is_accepted() {
    let app = test_app();
    // 1 ターン目相当の assistant tool_use を履歴に織り込んで tool_result を返す
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "temperature": 0.0,
            "messages": [
                {"role": "user", "content": "t0"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_abc123", "name": "get_weather",
                     "input": {"city": "Tokyo"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc123",
                     "content": "sunny 25C"}
                ]}
            ],
            "tools": tools_json(),
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["content"][0]["text"], EXPECTED_GREEDY_TEXT);
}

#[tokio::test]
async fn tool_result_with_unknown_tool_use_id_is_rejected() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 20,
            "messages": [
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_missing", "content": "x"}
                ]}
            ],
            "tools": tools_json(),
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not match any tool_use"));
}
