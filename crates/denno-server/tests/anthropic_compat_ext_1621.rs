//! Task 1621: Anthropic `/v1/messages` の `thinking` 設定と `cache_control`
//! (system / message content blocks) の受理監査テスト。
//!
//! Claude Code / Anthropic SDK が送る形の `thinking`
//! (`{"type":"enabled","budget_tokens":N}` / `{"type":"disabled"}`) と、
//! system / message content / tool_use / tool_result の各 block に付く
//! `cache_control` (ephemeral・ttl・null・空オブジェクト) が **400 にならず**、
//! かつプロンプト/生成結果/トークン数に影響しないことを固定する。
//!
//! 監査結果 (実装変更なし):
//! - `thinking` のマップ形式は既存の s52/s53c の規則どおり受理される
//!   (enabled/disabled とも off 扱い、`budget_tokens` / `display` 等は無視)
//! - `cache_control` は system / message content / tool_result 内 block の
//!   いずれでも無視され、値の有無で応答が変わらない
//! - 変更していない既存の拒否 (text 以外の system block / 空 text system block /
//!   `image` 等のメディア block / assistant 履歴の `thinking` block) は
//!   従来どおり 400 (`thinking` 設定そのものではない)
//!
//! fixture は `tests/anthropic_blocks_edge_1264.rs` と同一 (dummy モデル +
//! dummy トークナイザ、GPU 不要、sleep 無し、greedy で決定的)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
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

async fn post_full(app: &Router, uri: &str, body: Value) -> (StatusCode, HeaderMap, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
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

async fn post_text(app: &Router, uri: &str, body: Value) -> (StatusCode, HeaderMap, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
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

/// 生成結果の比較用シグネチャ。`id` と KV prefix 再利用の内訳 (`cache_*`) は
/// リクエスト順に依存するため除き、プロンプトに由来する値だけを取り出す。
fn gen_sig(v: &Value) -> Value {
    json!({
        "content": v["content"],
        "stop_reason": v["stop_reason"],
        "stop_sequence": v["stop_sequence"],
        "input_tokens": v["usage"]["input_tokens"],
        "output_tokens": v["usage"]["output_tokens"],
    })
}

async fn gen_ok(app: &Router, body: Value) -> Value {
    let (status, _headers, v) = post_full(app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "200 でない: body = {v}");
    assert_eq!(v["type"], "message", "body = {v}");
    assert!(v["usage"]["input_tokens"].as_u64().unwrap_or(0) > 0, "body = {v}");
    v
}

/// 400 封筒の形を検証し、`error.message` を返す (トップレベルは type/error のみ)。
fn assert_clean_400(status: StatusCode, headers: &HeaderMap, v: &Value, needle: &str) -> String {
    assert_eq!(status, StatusCode::BAD_REQUEST, "400 でない: body = {v}");
    let ct = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    assert!(ct.starts_with("application/json"), "JSON でない: {ct:?}");
    let top = v.as_object().unwrap_or_else(|| panic!("object でない: {v}"));
    let mut keys: Vec<&str> = top.keys().map(|s| s.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(keys, vec!["error", "type"], "封筒が余計: {v}");
    assert_eq!(v["type"], "error", "body = {v}");
    let err = v["error"].as_object().unwrap_or_else(|| panic!("error が object でない: {v}"));
    let mut ekeys: Vec<&str> = err.keys().map(|s| s.as_str()).collect();
    ekeys.sort_unstable();
    assert_eq!(ekeys, vec!["message", "type"], "error のキーが余計: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    let msg = v["error"]["message"].as_str().unwrap_or_default().to_string();
    assert!(!msg.is_empty(), "message が空: {v}");
    assert!(msg.contains(needle), "message に {needle:?} が無い: {msg}");
    msg
}

/// `cache_control` のバリエーション (SDK が実際に送る形)。
fn cache_variants() -> Vec<Value> {
    vec![
        json!({"type": "ephemeral"}),
        json!({"type": "ephemeral", "ttl": "5m"}),
        json!({"type": "ephemeral", "ttl": "1h"}),
        Value::Null,
        json!({}),
    ]
}

// ---------------------------------------------------------------------------
// thinking 設定
// ---------------------------------------------------------------------------

/// SDK / Claude Code が送る thinking マップは 400 にならず正常に生成する。
#[tokio::test]
async fn thinking_sdk_map_forms_are_accepted() {
    let app = test_app();
    for thinking in [
        json!({"type": "enabled", "budget_tokens": 1024}),
        json!({"type": "enabled", "budget_tokens": 31999}),
        json!({"type": "enabled", "budget_tokens": 8000, "display": "summarized"}),
        json!({"type": "enabled"}),
        json!({"type": "disabled"}),
        json!({"type": "disabled", "budget_tokens": 0}),
    ] {
        let mut body = base_body();
        body["thinking"] = thinking.clone();
        let v = gen_ok(&app, body).await;
        assert_eq!(v["stop_reason"], "end_turn", "thinking={thinking}: {v}");
        assert!(
            v["content"].as_array().map_or(false, |a| !a.is_empty()),
            "thinking={thinking}: {v}"
        );
    }
}

/// enabled / disabled マップは既存の off 扱いと同一 (budget_tokens は無視)。
#[tokio::test]
async fn thinking_map_equals_string_off_regardless_of_budget() {
    let app = test_app();
    let mut off = base_body();
    off["thinking"] = json!("off");
    let off_sig = gen_sig(&gen_ok(&app, off).await);

    for thinking in [
        json!({"type": "enabled", "budget_tokens": 1024}),
        json!({"type": "enabled", "budget_tokens": 31999, "display": "summarized"}),
        json!({"type": "disabled"}),
    ] {
        let mut body = base_body();
        body["thinking"] = thinking.clone();
        let sig = gen_sig(&gen_ok(&app, body).await);
        assert_eq!(sig, off_sig, "thinking={thinking} が off と一致しない");
    }
}

/// `"thinking": null` は未指定と同じ (既定 = brief)。
#[tokio::test]
async fn thinking_null_equals_omitted() {
    let app = test_app();
    let omitted = gen_sig(&gen_ok(&app, base_body()).await);
    let mut body = base_body();
    body["thinking"] = Value::Null;
    let sig = gen_sig(&gen_ok(&app, body).await);
    assert_eq!(sig, omitted, "thinking:null が未指定と一致しない");
}

/// 未知の thinking type は 400 で、封筒が壊れず message に thinking が出る。
#[tokio::test]
async fn thinking_unknown_forms_are_clean_400() {
    let app = test_app();
    for thinking in [
        json!({"type": "mystery"}),
        json!({"budget_tokens": 1024}),
        json!({"type": 7}),
        json!(42),
        json!(true),
        json!(["enabled"]),
        json!("extended"),
    ] {
        let mut body = base_body();
        body["thinking"] = thinking.clone();
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        assert_clean_400(status, &headers, &v, "thinking");
    }
}

// ---------------------------------------------------------------------------
// cache_control: system
// ---------------------------------------------------------------------------

/// system の text block に付く cache_control は無視される (応答が変わらない)。
#[tokio::test]
async fn cache_control_on_system_blocks_is_ignored() {
    let app = test_app();
    let plain = json!([
        {"type": "text", "text": "You are a test."},
        {"type": "text", "text": " Answer briefly."}
    ]);
    let mut base = base_body();
    base["system"] = plain.clone();
    let base_sig = gen_sig(&gen_ok(&app, base).await);

    for cc in cache_variants() {
        // 全 block に付与
        let mut body = base_body();
        body["system"] = json!([
            {"type": "text", "text": "You are a test.", "cache_control": cc.clone()},
            {"type": "text", "text": " Answer briefly.", "cache_control": cc.clone()}
        ]);
        assert_eq!(gen_sig(&gen_ok(&app, body).await), base_sig, "全 block cc={cc}");

        // Anthropic の慣習どおり末尾 block だけに付与
        let mut body = base_body();
        body["system"] = json!([
            {"type": "text", "text": "You are a test."},
            {"type": "text", "text": " Answer briefly.", "cache_control": cc}
        ]);
        assert_eq!(gen_sig(&gen_ok(&app, body).await), base_sig, "末尾 block cc");
    }
}

/// cache_control の有無で system の text 検証は緩まない (空 text は 400 のまま)。
#[tokio::test]
async fn cache_control_does_not_relax_system_text_validation() {
    let app = test_app();
    for cc in [json!({"type": "ephemeral"}), json!({"type": "ephemeral", "ttl": "1h"})] {
        let mut body = base_body();
        body["system"] = json!([{"type": "text", "text": "", "cache_control": cc.clone()}]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        assert_clean_400(status, &headers, &v, "system[0].text");

        let mut body = base_body();
        body["system"] = json!([{"type": "text", "cache_control": cc}]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        assert_clean_400(status, &headers, &v, "system[0].text");
    }
}

// ---------------------------------------------------------------------------
// cache_control: messages
// ---------------------------------------------------------------------------

/// user の text block に付く cache_control は無視され、素の連結と一致する。
#[tokio::test]
async fn cache_control_on_message_text_blocks_is_ignored() {
    let app = test_app();
    let mut plain = base_body();
    plain["messages"] = json!([{"role": "user", "content": "hello world"}]);
    let plain_sig = gen_sig(&gen_ok(&app, plain).await);

    for cc in cache_variants() {
        let mut body = base_body();
        body["messages"] = json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "hello ", "cache_control": cc.clone()},
                {"type": "text", "text": "world", "cache_control": cc}
            ]
        }]);
        assert_eq!(gen_sig(&gen_ok(&app, body).await), plain_sig, "cc 付き text block");
    }

    // 文字列との混在ではなく block 配列のみ (cache_control 無し) とも一致
    let mut blocks = base_body();
    blocks["messages"] = json!([{
        "role": "user",
        "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]
    }]);
    assert_eq!(gen_sig(&gen_ok(&app, blocks).await), plain_sig, "block 配列の連結");
}

/// tool_use / tool_result の block に付く cache_control も無視される
/// (tool_result.content 内の text block の cache_control を含む)。
#[tokio::test]
async fn cache_control_on_tool_blocks_is_ignored() {
    let app = test_app();
    let tools = json!([{
        "name": "get_weather",
        "description": "weather lookup",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
    }]);

    let build = |cc: Option<Value>| {
        let mut tool_use = json!({
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "Tokyo"}
        });
        let mut inner_text = json!({"type": "text", "text": "sunny"});
        let mut tool_result = json!({
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": [inner_text.clone()]
        });
        if let Some(c) = &cc {
            tool_use["cache_control"] = c.clone();
            inner_text["cache_control"] = c.clone();
            tool_result["cache_control"] = c.clone();
            tool_result["content"] = json!([inner_text]);
        }
        let mut body = base_body();
        body["tools"] = tools.clone();
        body["messages"] = json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [tool_result]}
        ]);
        body
    };

    let base_sig = gen_sig(&gen_ok(&app, build(None)).await);
    for cc in cache_variants() {
        assert_eq!(
            gen_sig(&gen_ok(&app, build(Some(cc.clone()))).await),
            base_sig,
            "tool block cc={cc}"
        );
    }
}

/// Claude Code 形のフルペイロード (thinking マップ + cache_control 付き system /
/// message blocks + tools + stream=false) がそのまま 200 で通る。
#[tokio::test]
async fn claude_code_shaped_payload_is_accepted() {
    let app = test_app();
    let body = json!({
        "model": "denno-test",
        "max_tokens": 64,
        "temperature": 0.0,
        "thinking": {"type": "enabled", "budget_tokens": 10000},
        "system": [
            {"type": "text", "text": "You are a test."},
            {"type": "text", "text": " Answer briefly.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
        ],
        "tools": [{
            "name": "get_weather",
            "description": "weather lookup",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            "cache_control": {"type": "ephemeral"}
        }],
        "tool_choice": {"type": "auto"},
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "5m"}}
            ]
        }]
    });
    let v = gen_ok(&app, body).await;
    assert_eq!(v["model"], "denno-test", "body = {v}");
    assert!(v["usage"]["output_tokens"].as_u64().unwrap_or(0) > 0, "body = {v}");
}

/// 同じフルペイロードの stream=true も SSE で開始/終了イベントを返す。
#[tokio::test]
async fn claude_code_shaped_payload_streams() {
    let app = test_app();
    let body = json!({
        "model": "denno-test",
        "max_tokens": 32,
        "temperature": 0.0,
        "stream": true,
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "system": [{"type": "text", "text": "You are a test.", "cache_control": {"type": "ephemeral"}}],
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": "t0", "cache_control": {"type": "ephemeral"}}]
        }]
    });
    let (status, headers, text) = post_text(&app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "SSE が 200 でない: {text}");
    let ct = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct:?}");
    assert!(text.contains("event: message_start"), "SSE = {text}");
    assert!(text.contains("event: message_stop"), "SSE = {text}");
}

// ---------------------------------------------------------------------------
// count_tokens も同じ受理 (Claude Code は送信前に叩く)
// ---------------------------------------------------------------------------

/// count_tokens は thinking マップ + cache_control を受理し、無視フィールドの
/// 有無でトークン数が変わらない (thinking は off 扱いどうしで比較)。
#[tokio::test]
async fn count_tokens_accepts_thinking_and_cache_control() {
    let app = test_app();

    let mut plain = base_body();
    plain["thinking"] = json!("off");
    plain["system"] = json!([
        {"type": "text", "text": "You are a test."},
        {"type": "text", "text": " Answer briefly."}
    ]);
    let (status, _h, v) = post_full(&app, "/v1/messages/count_tokens", plain).await;
    assert_eq!(status, StatusCode::OK, "count_tokens が 200 でない: {v}");
    let base_n = v["input_tokens"].as_u64().expect("input_tokens");

    let mut full = base_body();
    full["thinking"] = json!({"type": "enabled", "budget_tokens": 31999, "display": "summarized"});
    full["system"] = json!([
        {"type": "text", "text": "You are a test.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " Answer briefly.", "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    ]);
    full["messages"] = json!([{
        "role": "user",
        "content": [{"type": "text", "text": "t0", "cache_control": {"type": "ephemeral"}}]
    }]);
    let (status, _h, v) = post_full(&app, "/v1/messages/count_tokens", full).await;
    assert_eq!(status, StatusCode::OK, "count_tokens が 200 でない: {v}");
    assert_eq!(
        v["input_tokens"].as_u64().expect("input_tokens"),
        base_n,
        "無視されるはずのフィールドで input_tokens が変わった"
    );
}
