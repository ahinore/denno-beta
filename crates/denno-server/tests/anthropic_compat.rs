//! P805: Anthropic Messages API 互換性の統合テスト。
//!
//! denno-engine のダミーモデル / ダミートークナイザで HTTP ワイヤを検証する
//! (tests/tools.rs と同一の fixture 方式)。カバー項目:
//!
//! - `POST /v1/messages/count_tokens` が `{"input_tokens": N}` を返し、値が
//!   同じ本文の `/v1/messages` の `usage.input_tokens` と一致する
//! - count_tokens は prefill しない (KV スロット / in_flight が動かない)
//! - `system` が文字列でも text block 配列でも同じ結果になる
//!   (任意の `cache_control` は黙って無視される)
//! - prefix cache 再利用時に `usage.cache_read_input_tokens` /
//!   `cache_creation_input_tokens` が報告される (非ストリーミング / SSE 両方)
//! - `stop_sequences` の打ち切り (非ストリーミング / SSE 両方)
//! - count_tokens のエラー (未知モデル 404 / messages 空 400 / 壊れた JSON 400)

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

fn app_with_state() -> (Router, Arc<AppState>) {
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
    let app = denno_server::router(state.clone());
    (app, state)
}

fn test_app() -> Router {
    app_with_state().0
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

/// SSE 本文から `data: ...` 行の JSON を取り出す (event: 行は無視)。
fn sse_events(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn sse_text(events: &[Value]) -> String {
    events
        .iter()
        .filter_map(|e| {
            if e["type"] == "content_block_delta" {
                e["delta"]["text"].as_str()
            } else {
                None
            }
        })
        .collect()
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

// ---------------------------------------------------------------------------
// count_tokens
// ---------------------------------------------------------------------------

#[tokio::test]
async fn count_tokens_matches_messages_input_tokens() {
    let app = test_app();
    let body = base_body();

    let (status, counted) = post(&app, "/v1/messages/count_tokens", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "body = {counted}");
    let n = counted["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が無い: {counted}"));
    assert!(n > 0, "input_tokens = {n}");

    // 同じ本文で生成 → usage.input_tokens が一致する (同じ正規化 + 同じ tokenizer)
    let (status, msg) = post(&app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "body = {msg}");
    assert_eq!(msg["usage"]["input_tokens"].as_u64().unwrap(), n);
}

#[tokio::test]
async fn count_tokens_does_not_prefill_or_touch_kv() {
    let (app, state) = app_with_state();
    let body = base_body();

    // 2 回叩いても値は同じ・スロットは動かない
    let (s1, a) = post(&app, "/v1/messages/count_tokens", body.clone()).await;
    let (s2, b) = post(&app, "/v1/messages/count_tokens", body).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(a["input_tokens"], b["input_tokens"]);

    assert_eq!(
        state.scheduler.in_flight(),
        0,
        "count_tokens がスケジューラにジョブを積んでいる"
    );
    let g = state.engine.lock().unwrap();
    assert_eq!(g.slot_busy_count(), 0, "count_tokens が KV スロットを掴んでいる");
    assert_eq!(g.slot_reserved_total(), 0, "count_tokens が KV 予約を取っている");
}

#[tokio::test]
async fn count_tokens_accepts_system_blocks_with_cache_control() {
    let app = test_app();
    let as_string = base_body();

    let mut as_blocks = base_body();
    as_blocks["system"] = json!([
        {"type": "text", "text": "You are a test.", "cache_control": {"type": "ephemeral"}}
    ]);

    let (s1, a) = post(&app, "/v1/messages/count_tokens", as_string).await;
    let (s2, b) = post(&app, "/v1/messages/count_tokens", as_blocks).await;
    assert_eq!(s1, StatusCode::OK, "body = {a}");
    assert_eq!(s2, StatusCode::OK, "body = {b}");
    assert_eq!(a["input_tokens"], b["input_tokens"]);
}

#[tokio::test]
async fn count_tokens_errors_are_anthropic_shaped() {
    let app = test_app();

    // 未知モデル → 404
    let mut body = base_body();
    body["model"] = json!("nope");
    let (status, v) = post(&app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "body = {v}");
    assert_eq!(v["type"], "error");

    // messages 空 → 400
    let mut body = base_body();
    body["messages"] = json!([]);
    let (status, v) = post(&app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["type"], "error");

    // 壊れた JSON → 400
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages/count_tokens")
        .header("content-type", "application/json")
        .body(Body::from("{\"messages\": "))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// system blocks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn system_text_blocks_with_cache_control_are_accepted() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "You are a test.", "cache_control": {"type": "ephemeral"}},
    ]);

    let (status, v) = post(&app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["type"], "message");
    // ダミーモデルの決定的な出力 (t0 → t5 t6 ...)
    assert_eq!(
        v["content"][0]["text"].as_str().unwrap(),
        "t5 t6 t7 t8 t9 t10 t11"
    );
}

#[tokio::test]
async fn system_string_and_blocks_count_identically() {
    // 文字列 system と 1 個の text block は同じプロンプトになる
    let app = test_app();
    let mut blocks = base_body();
    blocks["system"] = json!([{"type": "text", "text": "You are a test."}]);

    let (_, a) = post(&app, "/v1/messages", base_body()).await;
    let (_, b) = post(&app, "/v1/messages", blocks).await;
    assert_eq!(a["usage"]["input_tokens"], b["usage"]["input_tokens"]);
    assert_eq!(a["content"][0]["text"], b["content"][0]["text"]);
}

// ---------------------------------------------------------------------------
// cache metrics
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_metrics_report_prefix_reuse_non_streaming() {
    let app = test_app();

    let (status, first) = post(&app, "/v1/messages", base_body()).await;
    assert_eq!(status, StatusCode::OK, "body = {first}");
    let usage = &first["usage"];
    let input = usage["input_tokens"].as_u64().unwrap();
    assert_eq!(
        usage["cache_read_input_tokens"].as_u64().unwrap(),
        0,
        "初回から prefix を再利用している: {usage}"
    );
    assert_eq!(
        usage["cache_creation_input_tokens"].as_u64().unwrap(),
        input
    );

    // 同一本文の 2 回目 → KV prefix が再利用される
    let (status, second) = post(&app, "/v1/messages", base_body()).await;
    assert_eq!(status, StatusCode::OK, "body = {second}");
    let usage = &second["usage"];
    let input2 = usage["input_tokens"].as_u64().unwrap();
    let read = usage["cache_read_input_tokens"].as_u64().unwrap();
    let created = usage["cache_creation_input_tokens"].as_u64().unwrap();
    assert!(read > 0, "2 回目で cache_read が 0: {usage}");
    assert_eq!(read + created, input2, "read + creation != input: {usage}");
    assert_eq!(input2, input, "同一プロンプトで input_tokens が変わった");
}

#[tokio::test]
async fn cache_metrics_in_streaming_message_start() {
    let app = test_app();

    let mut body = base_body();
    body["stream"] = json!(true);

    // 1 回目 (full prefill) と 2 回目 (prefix 再利用) の message_start を比べる
    let (status, _) = post(&app, "/v1/messages", body.clone()).await;
    assert_eq!(status, StatusCode::OK);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let events = sse_events(&text);

    let start = events
        .iter()
        .find(|e| e["type"] == "message_start")
        .expect("message_start が無い");
    let usage = &start["message"]["usage"];
    let input = usage["input_tokens"].as_u64().unwrap();
    let read = usage["cache_read_input_tokens"].as_u64().unwrap();
    let created = usage["cache_creation_input_tokens"].as_u64().unwrap();
    assert!(read > 0, "message_start で cache_read が 0: {usage}");
    assert_eq!(read + created, input);
}

// ---------------------------------------------------------------------------
// stop_sequences
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_sequences_truncate_non_streaming() {
    let app = test_app();
    let mut body = base_body();
    body["stop_sequences"] = json!(["t8"]);

    let (status, v) = post(&app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let text = v["content"][0]["text"].as_str().unwrap();
    assert!(!text.contains("t8"), "stop 語が本文に残っている: {text:?}");
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "t8");
    // stop 語の直前までの本文 (語間の空白は本文の一部なので残る)
    assert_eq!(text.trim_end(), "t5 t6 t7");
}

#[tokio::test]
async fn stop_sequences_truncate_streaming() {
    let app = test_app();
    let mut body = base_body();
    body["stop_sequences"] = json!(["t8"]);
    body["stream"] = json!(true);

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let events = sse_events(&text);

    let deltas = sse_text(&events);
    assert!(!deltas.contains("t8"), "stop 語が delta に残っている: {deltas:?}");
    assert_eq!(deltas, "t5 t6 t7");

    let delta = events
        .iter()
        .find(|e| e["type"] == "message_delta")
        .expect("message_delta が無い");
    assert_eq!(delta["delta"]["stop_reason"], "stop_sequence");
    assert_eq!(delta["delta"]["stop_sequence"], "t8");
}
