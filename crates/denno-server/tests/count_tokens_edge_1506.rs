//! Task 1506: `/v1/messages/count_tokens` と `/v1/messages` の
//! `input_tokens` パリティ / 不正ペイロードの 400 封筒パリティ監査。
//!
//! 監査対象:
//! - system (文字列 / text block 配列) の描画差が両端点で同じであること
//! - tools 宣言が count に反映され、生成時の `usage.input_tokens` と一致すること
//! - tool_use / tool_result 履歴が両端点で同じトークン数になること
//! - 不正ペイロードの 400 (と 404) が **同じ status / 同じ JSON 本文** になること
//! - drain 中の新規要求が両端点で同じ 503 overloaded_error になること
//!
//! fixture は `tests/anthropic_compat.rs` と同じ (dummy モデル + テンプレート付き
//! 決定的トークナイザ)。tools 宣言を prompt に反映させるため、QWEN 風ではなく
//! tools を描画する最小テンプレートを使う (テンプレート無しの fallback は tools を
//! 無視するため、パリティ監査の意味が無くなる)。GPU 不要・sleep 無し。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{DummyModel, TOKENIZER_JSON};
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// tools の name / description と messages を素直に並べる最小テンプレート。
/// Whitespace 前処理の WordLevel 語彙なので、区切りは空白で入れる。
const TEMPLATE: &str = "{% for m in messages %}{{ m.role }} {{ m.content }} \
{% endfor %}{% if tools %}{% for t in tools %}{{ t.function.name }} \
{{ t.function.description }} {% endfor %}{% endif %}assistant";

fn test_tokenizer() -> DennoTokenizer {
    DennoTokenizer::from_json_str(TOKENIZER_JSON, Some(TEMPLATE.to_string())).unwrap()
}

fn app_with_state() -> (Router, Arc<AppState>) {
    let engine = Engine::from_parts(
        Box::new(DummyModel),
        test_tokenizer(),
        "denno-test",
        None,
    );
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

async fn post_raw(app: &Router, uri: &str, body: &str) -> (StatusCode, String) {
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

async fn post_json(app: &Router, uri: &str, body: &Value) -> (StatusCode, String) {
    post_raw(app, uri, &body.to_string()).await
}

/// count_tokens の `{"input_tokens": N}` を読む。
async fn count_tokens(app: &Router, body: &Value) -> u64 {
    let (status, text) = post_json(app, "/v1/messages/count_tokens", body).await;
    assert_eq!(status, StatusCode::OK, "count_tokens body = {text}");
    let v: Value = serde_json::from_str(&text).expect("count_tokens が JSON でない");
    v["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が数値でない: {v}"))
}

/// /v1/messages の `usage.input_tokens` を読む。
async fn usage_input_tokens(app: &Router, body: &Value) -> u64 {
    let (status, text) = post_json(app, "/v1/messages", body).await;
    assert_eq!(status, StatusCode::OK, "messages body = {text}");
    let v: Value = serde_json::from_str(&text).expect("/v1/messages が JSON でない");
    v["usage"]["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("usage.input_tokens が数値でない: {v}"))
}

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "t1 t2",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

fn tool_defs() -> Value {
    json!([
        {
            "name": "t3",
            "description": "t4",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "t4",
            "description": "t5",
            "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}
        }
    ])
}

// ---------------------------------------------------------------------------
// 1. system: 文字列と block 配列 (cache_control 付き) のパリティ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn system_string_and_blocks_match_both_endpoints() {
    let (app, _state) = app_with_state();

    let mut blocks = base_body();
    blocks["system"] = json!([
        {"type": "text", "text": "t1 "},
        {"type": "text", "text": "t2", "cache_control": {"type": "ephemeral"}}
    ]);

    let plain = base_body();
    let c_plain = count_tokens(&app, &plain).await;
    let c_blocks = count_tokens(&app, &blocks).await;
    assert_eq!(
        c_plain, c_blocks,
        "system の表現で count_tokens が変わった (string={c_plain}, blocks={c_blocks})"
    );
    assert_eq!(
        usage_input_tokens(&app, &plain).await,
        c_plain,
        "/v1/messages の usage.input_tokens と count_tokens が不一致 (system string)"
    );
    assert_eq!(
        usage_input_tokens(&app, &blocks).await,
        c_blocks,
        "/v1/messages の usage.input_tokens と count_tokens が不一致 (system blocks)"
    );
}

// ---------------------------------------------------------------------------
// 2. tools 宣言
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tools_declarations_are_counted_and_match_usage() {
    let (app, _state) = app_with_state();

    let mut with_tools = base_body();
    with_tools["tools"] = tool_defs();

    let without = count_tokens(&app, &base_body()).await;
    let with = count_tokens(&app, &with_tools).await;
    assert!(
        with > without,
        "tools 宣言が prompt に反映されていない (tools 無し={without}, 有り={with})"
    );
    assert_eq!(
        usage_input_tokens(&app, &with_tools).await,
        with,
        "tools 付きリクエストで count_tokens と usage.input_tokens が不一致"
    );

    // tool_choice=none は tools を非公開にする (両端点で同じだけ減る)。
    let mut choice_none = with_tools.clone();
    choice_none["tool_choice"] = json!("none");
    let c_none = count_tokens(&app, &choice_none).await;
    assert_eq!(
        c_none, without,
        "tool_choice=none が tools を非公開にしていない (count={c_none}, tools 無し={without})"
    );
    assert_eq!(
        usage_input_tokens(&app, &choice_none).await,
        c_none,
        "tool_choice=none で count_tokens と usage.input_tokens が不一致"
    );
}

// ---------------------------------------------------------------------------
// 3. tool_use / tool_result 履歴
// ---------------------------------------------------------------------------

fn tool_history() -> Value {
    json!({
        "model": "denno-test",
        "system": "t1",
        "messages": [
            {"role": "user", "content": "t0"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "t2 "},
                {"type": "tool_use", "id": "call-1", "name": "t3", "input": {"x": "t4"}}
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "t5"}
            ]}
        ],
        "tools": tool_defs(),
        "max_tokens": 8,
        "temperature": 0.0,
    })
}

#[tokio::test]
async fn tool_use_and_result_history_match_usage() {
    let (app, _state) = app_with_state();

    let body = tool_history();
    let counted = count_tokens(&app, &body).await;
    assert_eq!(
        usage_input_tokens(&app, &body).await,
        counted,
        "tool_use/tool_result 履歴で count_tokens と usage.input_tokens が不一致"
    );

    // 履歴の中身が実際に prompt へ入っていること (素の 2 往復とは違う)。
    let plain = json!({
        "model": "denno-test",
        "system": "t1",
        "messages": [
            {"role": "user", "content": "t0"},
            {"role": "assistant", "content": "t2"},
            {"role": "user", "content": "t5"}
        ],
        "tools": tool_defs(),
        "max_tokens": 8,
        "temperature": 0.0,
    });
    assert_ne!(
        counted,
        count_tokens(&app, &plain).await,
        "tool_use/tool_result のテキスト化が count に反映されていない"
    );
}

// ---------------------------------------------------------------------------
// 4. 不正ペイロード: status と本文の完全一致
// ---------------------------------------------------------------------------

async fn assert_parity(app: &Router, body: &Value) {
    let (s_msg, b_msg) = post_json(app, "/v1/messages", body).await;
    let (s_cnt, b_cnt) = post_json(app, "/v1/messages/count_tokens", body).await;
    assert_eq!(
        s_msg, s_cnt,
        "status 不一致 (messages={s_msg}, count_tokens={s_cnt}) body={body}"
    );
    let v_msg: Value = serde_json::from_str(&b_msg).expect("messages のエラーが JSON でない");
    let v_cnt: Value = serde_json::from_str(&b_cnt).expect("count_tokens のエラーが JSON でない");
    assert_eq!(
        v_msg, v_cnt,
        "エラー本文不一致 (messages={b_msg}, count_tokens={b_cnt})"
    );
    assert_eq!(v_msg["type"], "error", "Anthropic 封筒でない: {b_msg}");
    assert!(
        v_msg["error"]["type"].is_string(),
        "error.type が無い: {b_msg}"
    );
}

#[tokio::test]
async fn invalid_payload_error_envelopes_are_identical() {
    let (app, _state) = app_with_state();

    let mut empty_messages = base_body();
    empty_messages["messages"] = json!([]);

    let mut bad_role = base_body();
    bad_role["messages"] = json!([{"role": "system", "content": "t0"}]);

    let mut choice_without_tools = base_body();
    choice_without_tools["tool_choice"] = json!("any");

    let mut choice_undefined = base_body();
    choice_undefined["tools"] = tool_defs();
    choice_undefined["tool_choice"] = json!({"type": "tool", "name": "t9"});

    let mut blank_tool_name = base_body();
    blank_tool_name["tools"] = json!([{"name": "  ", "input_schema": {"type": "object"}}]);

    let mut bad_schema = base_body();
    bad_schema["tools"] =
        json!([{"name": "t3", "input_schema": ["not", "an", "object"]}]);

    let mut bad_system_type = base_body();
    bad_system_type["system"] = json!(42);

    let mut empty_system_block = base_body();
    empty_system_block["system"] = json!([{"type": "text", "text": "   "}]);

    let mut bad_system_block_type = base_body();
    bad_system_block_type["system"] = json!([{"type": "image", "text": "t1"}]);

    let mut tool_use_in_user = base_body();
    tool_use_in_user["messages"] = json!([
        {"role": "user", "content": [
            {"type": "tool_use", "id": "call-1", "name": "t3", "input": {}}
        ]}
    ]);

    let mut tool_result_in_assistant = base_body();
    tool_result_in_assistant["messages"] = json!([
        {"role": "assistant", "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": "t5"}
        ]}
    ]);

    let mut duplicate_tool_use = base_body();
    duplicate_tool_use["messages"] = json!([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call-1", "name": "t3", "input": {}},
            {"type": "tool_use", "id": "call-1", "name": "t4", "input": {}}
        ]}
    ]);

    let mut bad_tool_use_input = base_body();
    bad_tool_use_input["messages"] = json!([
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "call-1", "name": "t3", "input": "t5"}
        ]}
    ]);

    let mut bad_thinking = base_body();
    bad_thinking["thinking"] = json!({"type": "mystery"});

    let mut bad_ngram = base_body();
    bad_ngram["no_repeat_ngram_size"] = json!(1);

    let mut unknown_model = base_body();
    unknown_model["model"] = json!("nope");

    let cases: Vec<(&str, Value)> = vec![
        ("messages 空", empty_messages),
        ("role 不正", bad_role),
        ("tool_choice に tools 無し", choice_without_tools),
        ("未定義ツール名", choice_undefined),
        ("tools[].name 空", blank_tool_name),
        ("input_schema 非 object", bad_schema),
        ("system が数値", bad_system_type),
        ("system text 空", empty_system_block),
        ("system block type 不正", bad_system_block_type),
        ("user の tool_use", tool_use_in_user),
        ("assistant の tool_result", tool_result_in_assistant),
        ("tool_use id 重複", duplicate_tool_use),
        ("tool_use.input 非 object", bad_tool_use_input),
        ("thinking 不正", bad_thinking),
        ("no_repeat_ngram_size=1", bad_ngram),
        ("未知モデル (404)", unknown_model),
    ];
    for (label, body) in cases {
        assert_parity(&app, &body).await;
        // 400/404 のどちらでも、Anthropic 封筒が両端点で同文であることを再確認済み。
        let (status, _) = post_json(&app, "/v1/messages/count_tokens", &body).await;
        assert!(
            status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
            "{label}: 想定外 status {status}"
        );
    }

    // 壊れた JSON は Value を作れないので生ボディで比較する。
    let raw = "{\"model\": \"denno-test\", \"messages\": [";
    let (s1, b1) = post_raw(&app, "/v1/messages", raw).await;
    let (s2, b2) = post_raw(&app, "/v1/messages/count_tokens", raw).await;
    assert_eq!(s1, StatusCode::BAD_REQUEST);
    assert_eq!(s2, StatusCode::BAD_REQUEST);
    assert_eq!(
        serde_json::from_str::<Value>(&b1).unwrap(),
        serde_json::from_str::<Value>(&b2).unwrap(),
        "壊れた JSON の 400 本文が不一致 (messages={b1}, count_tokens={b2})"
    );
}

// ---------------------------------------------------------------------------
// 5. drain 中の新規要求
// ---------------------------------------------------------------------------

#[tokio::test]
async fn draining_rejects_both_endpoints_with_same_503() {
    let (app, state) = app_with_state();
    state.scheduler.lifecycle().begin_drain();

    let body = base_body();
    let (s_msg, b_msg) = post_json(&app, "/v1/messages", &body).await;
    let (s_cnt, b_cnt) = post_json(&app, "/v1/messages/count_tokens", &body).await;
    assert_eq!(s_msg, StatusCode::SERVICE_UNAVAILABLE, "body = {b_msg}");
    assert_eq!(
        s_cnt, s_msg,
        "drain 中の count_tokens が 503 でない (body = {b_cnt})"
    );
    let v_msg: Value = serde_json::from_str(&b_msg).unwrap();
    let v_cnt: Value = serde_json::from_str(&b_cnt).unwrap();
    assert_eq!(v_msg["error"]["type"], "overloaded_error");
    assert_eq!(
        v_msg, v_cnt,
        "drain 中の 503 本文が不一致 (messages={b_msg}, count_tokens={b_cnt})"
    );

    // 拒否された count_tokens はスケジューラ / KV を動かさない。
    assert_eq!(state.scheduler.in_flight(), 0);
    let g = state.engine.lock().unwrap();
    assert_eq!(g.slot_busy_count(), 0);
    assert_eq!(g.slot_reserved_total(), 0);
}
