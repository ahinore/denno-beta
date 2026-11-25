//! Task 974: Anthropic Messages / count_tokens / cache_control の edge-case 統合テスト。
//!
//! fixture は `tests/anthropic_compat.rs` / `tests/tools.rs` と同一
//! (dummy モデル + dummy トークナイザ。GPU 不要)。カバー項目:
//!
//! - `/v1/messages/count_tokens` + system prompt / 複数 block の content 配列 /
//!   tools 定義。値が同じ本文の `/v1/messages` の `usage.input_tokens` と一致する
//! - 空 / 不正 JSON / モデル欠落 / 範囲外 max_tokens が Anthropic 形式の 4xx を返す
//! - system と messages 配列の `cache_control` (ephemeral) がエラー無く受理される
//! - SSE のイベント順序不変量
//!   (message_start -> content_block_start -> content_block_delta*
//!    -> content_block_stop -> message_delta -> message_stop)
//! - `anthropic-version` ヘッダの有無 (サーバーは現状このヘッダを要求しない)
//!
//! 注意: 本サーバーは `anthropic-version` を検証せず、usize 範囲内の
//! 過大な `max_tokens` は 400 ではなく `MAX_TOKENS_CLAMP` (4096) へ clamp する。
//! 要件の「4xx」は実際の契約に合わせ、**range 外 (usize で表現不能) の
//! max_tokens** と壊れた JSON で検証する (推測で 400 を期待しない)。

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

/// 生ボディ + 任意ヘッダで POST する (壊れた JSON とヘッダ有無の検証用)。
async fn post_raw(
    app: &Router,
    uri: &str,
    body: &str,
    headers: &[(&str, &str)],
) -> (StatusCode, String) {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let req = builder.body(Body::from(body.to_string())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let (status, text) = post_raw(app, uri, &body.to_string(), &[]).await;
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, v)
}

/// SSE の `event:` / `data:` 行を順に取り出す (`data:` 行のみ JSON)。
/// keep-alive の `event: ping` もそのまま残す (順序不変量の検証側で無視する)。
fn sse_lines(text: &str) -> Vec<(Option<String>, Option<Value>)> {
    let mut out = Vec::new();
    let mut pending_event: Option<String> = None;
    for line in text.lines() {
        if let Some(ev) = line.strip_prefix("event: ") {
            pending_event = Some(ev.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            let v = serde_json::from_str::<Value>(data).expect("SSE data が JSON ではない");
            out.push((pending_event.take(), Some(v)));
        }
    }
    out
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

/// Anthropic のエラー封筒 (`{"type":"error","error":{"type":..,"message":..}}`) を検証する。
fn assert_error_envelope(status: StatusCode, v: &Value, expected_kind: &str) {
    assert!(status.is_client_error(), "4xx でない: {status} body = {v}");
    assert_eq!(v["type"], "error", "body = {v}");
    assert_eq!(v["error"]["type"], expected_kind, "body = {v}");
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "error.message が空: {v}");
}

// ---------------------------------------------------------------------------
// count_tokens: system prompt / multi-block content / tools
// ---------------------------------------------------------------------------

#[tokio::test]
async fn count_tokens_multi_block_content_matches_generation() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "You are a test.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " Answer briefly.", "cache_control": {"type": "ephemeral"}}
    ]);
    body["messages"] = json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": "t0", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": " please answer"}
        ]
    }]);

    let (status, counted) = post(&app, "/v1/messages/count_tokens", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "body = {counted}");
    let n = counted["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が無い: {counted}"));
    assert!(n > 0, "input_tokens = {n}");

    let (status, msg) = post(&app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "body = {msg}");
    assert_eq!(
        msg["usage"]["input_tokens"].as_u64().unwrap(),
        n,
        "count_tokens と生成で input_tokens が不一致: {msg}"
    );
}

#[tokio::test]
async fn count_tokens_with_tools_matches_generation() {
    let app = test_app();
    let plain = base_body();
    let mut with_tools = base_body();
    with_tools["tools"] = tools_json();

    let (s_plain, a) = post(&app, "/v1/messages/count_tokens", plain).await;
    assert_eq!(s_plain, StatusCode::OK, "body = {a}");
    // tools 定義付きでも 200 (`input_tokens` が返る)。dummy fixture の chat
    // template は tools を prompt へ展開しないため、plain と同値になり得る。
    let (s_tools, b) = post(&app, "/v1/messages/count_tokens", with_tools.clone()).await;
    assert_eq!(s_tools, StatusCode::OK, "body = {b}");
    let n_tools = b["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が無い: {b}"));
    assert!(n_tools > 0, "input_tokens = {n_tools}");

    // tools 付きでも count_tokens と生成の input_tokens が一致する
    let (status, msg) = post(&app, "/v1/messages", with_tools).await;
    assert_eq!(status, StatusCode::OK, "body = {msg}");
    assert_eq!(msg["usage"]["input_tokens"].as_u64().unwrap(), n_tools);
}

#[tokio::test]
async fn count_tokens_empty_and_invalid_bodies_are_4xx_envelopes() {
    let app = test_app();

    // messages 空
    let mut body = base_body();
    body["messages"] = json!([]);
    let (status, v) = post(&app, "/v1/messages/count_tokens", body).await;
    assert_error_envelope(status, &v, "invalid_request_error");

    // model 欠落 (デシリアライズ失敗)
    let (status, v) = post(&app, "/v1/messages/count_tokens", json!({"messages": []})).await;
    assert_error_envelope(status, &v, "invalid_request_error");

    // 空ボディ
    let (status, text) = post_raw(&app, "/v1/messages/count_tokens", "", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_error_envelope(status, &v, "invalid_request_error");

    // 壊れた JSON
    let (status, text) = post_raw(&app, "/v1/messages/count_tokens", "{\"messages\": ", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_error_envelope(status, &v, "invalid_request_error");
}

// ---------------------------------------------------------------------------
// cache_control: system / messages 配列で受理される
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_control_in_system_and_messages_arrays_is_accepted() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([{
        "type": "text",
        "text": "You are a test.",
        "cache_control": {"type": "ephemeral"}
    }]);
    body["messages"] = json!([
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "t0",
                "cache_control": {"type": "ephemeral"}
            }]
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "t5",
                "cache_control": {"type": "ephemeral"}
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "again",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }]
        }
    ]);

    let (status, v) = post(&app, "/v1/messages", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "cache_control で失敗: {v}");
    assert_eq!(v["type"], "message");
    assert_eq!(v["content"][0]["type"], "text");

    let (status, counted) = post(&app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::OK, "count_tokens が cache_control を拒否: {counted}");
    assert!(counted["input_tokens"].as_u64().unwrap() > 0);
}

// ---------------------------------------------------------------------------
// SSE イベント順序不変量
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_sse_event_sequence_invariant() {
    let app = test_app();
    let mut body = base_body();
    body["stream"] = json!(true);

    let (status, text) = post_raw(&app, "/v1/messages", &body.to_string(), &[]).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");

    let lines = sse_lines(&text);
    // `event:` 行と data の type が一致する (ping 含む)
    for (event, data) in &lines {
        let data = data.as_ref().unwrap();
        assert_eq!(
            event.as_deref(),
            data["type"].as_str(),
            "event 行と data.type が不一致: {text}"
        );
    }

    // keep-alive の ping を除いたイベント列
    let seq: Vec<&str> = lines
        .iter()
        .filter_map(|(_, d)| d.as_ref())
        .map(|d| d["type"].as_str().unwrap())
        .filter(|t| *t != "ping")
        .collect();

    assert_eq!(seq.first(), Some(&"message_start"), "先頭が message_start でない: {seq:?}");
    assert_eq!(seq.last(), Some(&"message_stop"), "末尾が message_stop でない: {seq:?}");
    assert_eq!(seq.iter().filter(|t| **t == "message_start").count(), 1);
    assert_eq!(seq.iter().filter(|t| **t == "message_stop").count(), 1);
    assert_eq!(seq.iter().filter(|t| **t == "message_delta").count(), 1);
    assert_eq!(seq.iter().filter(|t| **t == "content_block_start").count(), 1);
    assert_eq!(seq.iter().filter(|t| **t == "content_block_stop").count(), 1);

    let pos = |name: &str| seq.iter().position(|t| *t == name).unwrap();
    let start = pos("content_block_start");
    let stop = pos("content_block_stop");
    let delta = pos("message_delta");
    let end = pos("message_stop");
    assert!(start < stop, "content_block_start が stop より後: {seq:?}");
    assert!(stop < delta, "content_block_stop が message_delta より後: {seq:?}");
    assert!(delta < end, "message_delta が message_stop より後: {seq:?}");

    // content_block_start と stop の間に delta が 1 個以上あり、
    // その区間に他の型が混ざらない。
    let deltas: Vec<usize> = seq
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == "content_block_delta")
        .map(|(i, _)| i)
        .collect();
    assert!(!deltas.is_empty(), "content_block_delta が無い: {seq:?}");
    for i in start + 1..stop {
        assert_eq!(seq[i], "content_block_delta", "block 区間に余計なイベント: {seq:?}");
    }

    // delta の結合が dummy モデルの greedy 期待値と一致し、index は 0
    let mut out = String::new();
    for (_, data) in &lines {
        let data = data.as_ref().unwrap();
        if data["type"] == "content_block_delta" {
            assert_eq!(data["index"], 0, "delta index が 0 でない: {data}");
            if let Some(t) = data["delta"]["text"].as_str() {
                out.push_str(t);
            }
        }
    }
    assert_eq!(out, "t5 t6 t7 t8 t9 t10 t11");

    // message_start / message_delta の必須フィールド
    let start_ev = lines
        .iter()
        .find_map(|(_, d)| d.as_ref().filter(|d| d["type"] == "message_start"))
        .unwrap();
    assert_eq!(start_ev["message"]["role"], "assistant");
    assert!(start_ev["message"]["usage"]["input_tokens"].as_u64().unwrap() > 0);
    let delta_ev = lines
        .iter()
        .find_map(|(_, d)| d.as_ref().filter(|d| d["type"] == "message_delta"))
        .unwrap();
    assert_eq!(delta_ev["delta"]["stop_reason"], "end_turn");
    assert!(delta_ev["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

// ---------------------------------------------------------------------------
// バージョンヘッダ / 壊れた JSON / 範囲外 max_tokens
// ---------------------------------------------------------------------------

/// 現行サーバーは `anthropic-version` を要求しない (ヘッダ非依存)。
/// 欠落・未知値のどちらも通常応答することを固定する。
#[tokio::test]
async fn anthropic_version_header_is_optional() {
    let app = test_app();
    let body = base_body().to_string();

    let (status, text) = post_raw(&app, "/v1/messages", &body, &[]).await;
    assert_eq!(status, StatusCode::OK, "ヘッダ無しで失敗: {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "message");

    let (status, text) = post_raw(
        &app,
        "/v1/messages",
        &body,
        &[("anthropic-version", "1999-01-01")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "未知バージョンで失敗: {text}");

    let (status, text) = post_raw(
        &app,
        "/v1/messages/count_tokens",
        &body,
        &[("anthropic-version", "2023-06-01")],
    )
    .await;
    assert_eq!(status, StatusCode::OK, "count_tokens で失敗: {text}");
}

#[tokio::test]
async fn invalid_json_and_out_of_range_max_tokens_are_4xx_envelopes() {
    let app = test_app();

    // 壊れた JSON
    let (status, text) = post_raw(&app, "/v1/messages", "{\"model\":", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_error_envelope(status, &v, "invalid_request_error");

    // 空ボディ
    let (status, text) = post_raw(&app, "/v1/messages", "", &[]).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {text}");

    // usize で表現できない max_tokens (デシリアライズ失敗 → 400)
    let mut body = base_body();
    body["max_tokens"] = json!(1e300);
    let (status, v) = post(&app, "/v1/messages", body).await;
    assert_error_envelope(status, &v, "invalid_request_error");

    // 未知モデル → 404 not_found_error
    let mut body = base_body();
    body["model"] = json!("nope");
    let (status, v) = post(&app, "/v1/messages", body).await;
    assert_error_envelope(status, &v, "not_found_error");
    assert_eq!(status, StatusCode::NOT_FOUND, "body = {v}");
}
