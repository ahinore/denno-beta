//! Task 991: `/v1/messages` の `system` ブロック配列とマルチモーダル拒否の統合テスト。
//!
//! fixture は `tests/anthropic_edge_974.rs` / `tests/anthropic_compat.rs` と同一
//! (dummy モデル + dummy トークナイザ。GPU 不要)。カバー項目:
//!
//! - 複数 text ブロックの `system` が全ブロック連結で受理され、同じ文字列の
//!   単一 system と `input_tokens` / `count_tokens` が一致する (連結順も含む)
//! - 空 / 空白のみ / `text` 欠落 / 非文字列 `text` の text ブロックは 400
//!   `invalid_request_error` (Anthropic 封筒) で理由に block index を含む
//! - `image` / `document` ブロック (system と messages の両方) は 400 で拒否
//!   (黙ってプロンプトから落とさない)
//! - `type` 欠落 / 非文字列、ブロックがオブジェクトでない、`system` が
//!   文字列でも配列でもない → 400
//! - `/v1/messages/count_tokens` も同じ検証を通る (同じ 400 を返す)
//! - `stream: true` でも検証エラーは SSE ではなく 400 JSON 封筒
//! - 空配列 `system: []` は従来どおり受理 (既存挙動の固定)

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

fn test_app() -> Router {
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

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 32,
        "temperature": 0.0,
    })
}

/// Anthropic のエラー封筒 (`{"type":"error","error":{"type":..,"message":..}}`) を検証し、
/// `error.message` を返す。
fn assert_invalid_request(status: StatusCode, v: &Value) -> String {
    assert_eq!(status, StatusCode::BAD_REQUEST, "400 でない: body = {v}");
    assert_eq!(v["type"], "error", "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "error.message が空: {v}");
    msg.to_string()
}

fn image_block() -> Value {
    json!({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}
    })
}

fn document_block() -> Value {
    json!({
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": "hello"}
    })
}

// ---------------------------------------------------------------------------
// 複数 text ブロックの system
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_block_system_is_accepted_and_matches_concatenated_string() {
    let app = test_app();

    let mut blocks = base_body();
    blocks["system"] = json!([
        {"type": "text", "text": "You are "},
        {"type": "text", "text": "a test.", "cache_control": {"type": "ephemeral"}},
    ]);
    let mut joined = base_body();
    joined["system"] = json!("You are a test.");

    let (status, b) = post(&app, "/v1/messages", blocks).await;
    assert_eq!(status, StatusCode::OK, "body = {b}");
    let (status, j) = post(&app, "/v1/messages", joined).await;
    assert_eq!(status, StatusCode::OK, "body = {j}");

    // 連結結果が同じ文字列なので、プロンプト長と生成結果が一致する
    assert_eq!(b["usage"]["input_tokens"], j["usage"]["input_tokens"]);
    assert_eq!(b["content"][0]["text"], j["content"][0]["text"]);
}

#[tokio::test]
async fn every_system_block_contributes_tokens() {
    let app = test_app();

    let mut one = base_body();
    one["system"] = json!([{"type": "text", "text": "alpha "}]);
    let mut three = base_body();
    three["system"] = json!([
        {"type": "text", "text": "alpha "},
        {"type": "text", "text": "beta "},
        {"type": "text", "text": "gamma"},
    ]);

    let (s1, a) = post(&app, "/v1/messages/count_tokens", one).await;
    assert_eq!(s1, StatusCode::OK, "body = {a}");
    let (s3, b) = post(&app, "/v1/messages/count_tokens", three).await;
    assert_eq!(s3, StatusCode::OK, "body = {b}");

    let n1 = a["input_tokens"].as_u64().unwrap();
    let n3 = b["input_tokens"].as_u64().unwrap();
    assert!(n3 > n1, "3 ブロック ({n3}) が 1 ブロック ({n1}) 以下: 全ブロックが使われていない");
}

#[tokio::test]
async fn count_tokens_matches_input_tokens_for_multi_block_system() {
    let app = test_app();

    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "You are "},
        {"type": "text", "text": "a test."},
    ]);

    let (status, gen) = post(&app, "/v1/messages", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "body = {gen}");
    let (status, cnt) = post(&app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::OK, "body = {cnt}");
    assert_eq!(gen["usage"]["input_tokens"], cnt["input_tokens"]);
}

#[tokio::test]
async fn empty_system_array_is_accepted() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "空配列は 400 にしない: body = {v}");
}

// ---------------------------------------------------------------------------
// 空 / 不正な text ブロック → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_text_block_in_system_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([{"type": "text", "text": ""}]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].text"), "message = {msg}");
}

#[tokio::test]
async fn whitespace_only_text_block_in_system_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "ok"},
        {"type": "text", "text": "  \n\t "},
    ]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[1].text"), "message = {msg}");
}

#[tokio::test]
async fn text_block_without_text_field_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([{"type": "text"}]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].text"), "message = {msg}");
}

#[tokio::test]
async fn text_block_with_non_string_text_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([{"type": "text", "text": 42}]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].text"), "message = {msg}");
}

#[tokio::test]
async fn system_block_without_type_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([{"text": "hi"}]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].type"), "message = {msg}");
}

#[tokio::test]
async fn non_object_system_block_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!(["plain string"]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0]"), "message = {msg}");
}

#[tokio::test]
async fn scalar_system_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!(123);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system must be a string or an array"), "message = {msg}");
}

// ---------------------------------------------------------------------------
// マルチモーダル block → 400
// ---------------------------------------------------------------------------

#[tokio::test]
async fn image_block_in_system_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([image_block()]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].type"), "message = {msg}");
    assert!(msg.contains("image"), "message = {msg}");
}

#[tokio::test]
async fn document_block_in_system_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([document_block()]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].type"), "message = {msg}");
    assert!(msg.contains("document"), "message = {msg}");
}

#[tokio::test]
async fn image_block_after_valid_text_block_reports_its_index() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "ok"},
        {"type": "text", "text": "also ok"},
        image_block(),
    ]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[2].type"), "message = {msg}");
}

#[tokio::test]
async fn image_block_in_message_content_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["messages"] = json!([{
        "role": "user",
        "content": [{"type": "text", "text": "look"}, image_block()],
    }]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("unsupported content block type"), "message = {msg}");
    assert!(msg.contains("image"), "message = {msg}");
}

#[tokio::test]
async fn document_block_in_message_content_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["messages"] = json!([{
        "role": "user",
        "content": [document_block()],
    }]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("unsupported content block type"), "message = {msg}");
    assert!(msg.contains("document"), "message = {msg}");
}

// ---------------------------------------------------------------------------
// count_tokens / stream も同じ検証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn count_tokens_rejects_image_system_block() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([image_block()]);
    let (status, v) = post(&app, "/v1/messages/count_tokens", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].type"), "message = {msg}");
}

#[tokio::test]
async fn count_tokens_rejects_empty_text_block() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([{"type": "text", "text": " "}]);
    let (status, v) = post(&app, "/v1/messages/count_tokens", body).await;
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].text"), "message = {msg}");
}

#[tokio::test]
async fn streaming_request_with_bad_system_block_is_json_400() {
    let app = test_app();
    let mut body = base_body();
    body["stream"] = json!(true);
    body["system"] = json!([image_block()]);
    let (status, v) = post(&app, "/v1/messages", body).await;
    // SSE ではなく通常の JSON エラー封筒で返る (ストリーム開始前に検証する)
    let msg = assert_invalid_request(status, &v);
    assert!(msg.contains("system[0].type"), "message = {msg}");
}
