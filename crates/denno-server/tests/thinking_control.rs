//! s47: thinking 制御 (`thinking`: "off" | "brief" | "full") のテスト。
//!
//! - lib 層: parse_thinking (モード判定・未指定 = brief・不正値 400) と
//!   brief_system (決定的な短答誘導文の system 冒頭注入) の単体検証
//! - wire 層: ThinkingSimModel (qwen_style_tokenizer) で「思考する/しない」を
//!   トークンレベルで観測する (off は空の think block で閉じた generation prompt に
//!   なるため即 EOS・brief/full は開いた think block から思考フィラを生成)。
//!   brief の system 注入は全 prompt トークンに依存する RecurStateModel の出力差分で
//!   観測する (注入の有無で状態が変わり argmax が変わる)。
//! - SSE 経路 (submit_stream) も同一の thinking を受けることを検証する。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{recurrent_qwen_style_engine, thinking_sim_engine};
use denno_server::{
    brief_system, parse_thinking, AppState, Scheduler, Thinking, THINKING_BRIEF_INSTRUCTION,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// テスト用 AppState + Router を構築する (engine を注入できる版)
fn test_app_with(engine: denno_engine::Engine, max_queue: usize) -> Router {
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), max_queue);
    let state = Arc::new(AppState {
        engine,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
    });
    denno_server::router(state)
}

fn request_body(content: &str, max_tokens: usize, stream: bool, thinking: Option<&str>) -> String {
    let mut v = json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": content}]
    });
    if let Some(t) = thinking {
        v["thinking"] = json!(t);
    }
    v.to_string()
}

/// s52: thinking を任意の JSON 値で指定する本文 (マップ形式 = Hermes と同一形状)
fn request_body_thinking_value(
    content: &str,
    max_tokens: usize,
    stream: bool,
    thinking: Value,
) -> String {
    let mut v = json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": content}]
    });
    v["thinking"] = thinking;
    v.to_string()
}

async fn post(app: &Router, body: String) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

/// SSE 用: ステータスと生ボディを返す
async fn post_raw(app: &Router, body: String) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// SSE ボディから content_block_delta の text を結合して生成テキストを復元する
fn sse_text(body: &str) -> String {
    let mut out = String::new();
    for line in body.lines() {
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
    out
}

// ---------- lib 層: parse_thinking / brief_system ----------

/// thinking のモード判定 (未指定 = brief・off/brief/full・不正値はエラー)
/// s52/s53c: Anthropic マップ形式も受理 (enabled = off 扱い / disabled = off 扱い /
/// 未知の type・type 欠落・非文字列 type・文字列/オブジェクト以外はエラー)
#[test]
fn parse_thinking_maps_all_modes_and_defaults_to_brief() {
    assert_eq!(parse_thinking(&None).unwrap(), Thinking::Brief);
    assert_eq!(parse_thinking(&Some(json!("off"))).unwrap(), Thinking::Off);
    assert_eq!(parse_thinking(&Some(json!("brief"))).unwrap(), Thinking::Brief);
    assert_eq!(parse_thinking(&Some(json!("full"))).unwrap(), Thinking::Full);
    // 大文字小文字を含む未知値は拒否 (tool_choice と同一の厳密一致)
    for bad in ["OFF", "Brief", "FULL", "", "on", "true", "none"] {
        assert!(parse_thinking(&Some(json!(bad))).is_err(), "{bad} は拒否されるべき");
    }
    // s52/s53c: Anthropic マップ形式 (enabled = off 扱い・disabled = off 扱い。
    // budget_tokens は無視・決定的)
    assert_eq!(
        parse_thinking(&Some(json!({"type": "enabled", "budget_tokens": 8000}))).unwrap(),
        Thinking::Off
    );
    assert_eq!(parse_thinking(&Some(json!({"type": "enabled"}))).unwrap(), Thinking::Off);
    assert_eq!(parse_thinking(&Some(json!({"type": "disabled"}))).unwrap(), Thinking::Off);
    // マップの不正 type / type 欠落 / 非文字列 type / マップ以外の値型は拒否
    for bad in [
        json!({"type": "bogus"}),
        json!({}),
        json!({"type": 1}),
        json!(1),
        json!(true),
        json!(["off"]),
    ] {
        assert!(parse_thinking(&Some(bad.clone())).is_err(), "{bad} は拒否されるべき");
    }
}

/// brief の system 注入は決定的 (固定文・既存 system は保持)
#[test]
fn brief_system_prepends_fixed_instruction_deterministically() {
    // 同一入力 → 同一出力 (決定的固定文)
    assert_eq!(brief_system(None), brief_system(None));
    assert_eq!(brief_system(None).as_deref(), Some(THINKING_BRIEF_INSTRUCTION));
    // system 未指定/空 → 誘導文単独の system
    assert_eq!(
        brief_system(Some(String::new())).as_deref(),
        Some(THINKING_BRIEF_INSTRUCTION)
    );
    // 既存 system は保持 (誘導文 + 空行 + 元の system)
    let expected = format!("{THINKING_BRIEF_INSTRUCTION}\n\nSYS");
    assert_eq!(brief_system(Some("SYS".into())).as_deref(), Some(expected.as_str()));
}

// ---------- wire 層: off / brief (既定) / full ----------

/// thinking=off と brief (既定) は空の think block で閉じた prompt になるため
/// ThinkingSimModel が即 EOS を出す (output_tokens = 1・テキスト空)。full は開いた
/// think block から思考フィラ 3 トークン + EOS (output_tokens = 4)。
/// s53c: brief は enable_thinking=false + 簡潔指示のため off と同じ即答になる。
#[tokio::test]
async fn thinking_off_and_brief_short_circuit_and_full_thinks() {
    let app = test_app_with(thinking_sim_engine("think-sim").unwrap(), 8);

    let (status, body) = post(&app, request_body("t0", 20, false, Some("off"))).await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["usage"]["output_tokens"], json!(1), "off は即 EOS (1 トークン)");
    assert_eq!(body["content"][0]["text"], "", "off は EOS のみでテキスト空");
    assert_eq!(body["stop_reason"], "end_turn");

    // s53c: 未指定 = brief (既定)・"brief" は off と同じ即答になる
    for thinking in [None, Some("brief")] {
        let (status, body) = post(&app, request_body("t0", 20, false, thinking)).await;
        assert_eq!(status, StatusCode::OK, "thinking={thinking:?} body = {body}");
        assert_eq!(
            body["usage"]["output_tokens"],
            json!(1),
            "thinking={thinking:?} は思考無効で即 EOS"
        );
        assert_eq!(body["content"][0]["text"], "", "thinking={thinking:?}");
        assert_eq!(body["stop_reason"], "end_turn", "thinking={thinking:?}");
    }

    // full のみ思考経路 (フィラ 3 + EOS)
    let (status, body) = post(&app, request_body("t0", 20, false, Some("full"))).await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["usage"]["output_tokens"], json!(4), "full は思考してから答える");
    assert_eq!(body["content"][0]["text"], "t13 t14 t15");
    assert_eq!(body["stop_reason"], "end_turn");
}

/// brief (既定) の system 注入と full の無注入は RecurStateModel (全 prompt トークン
/// に依存する状態) の出力差分で観測する。未指定は brief と同一・off は空 think block
/// 分だけ prompt が変わる。
#[tokio::test]
async fn thinking_brief_injects_system_and_full_is_unchanged() {
    let app = test_app_with(recurrent_qwen_style_engine("qwen-style", None).unwrap(), 8);

    let (_, default) = post(&app, request_body("t0", 8, false, None)).await;
    let (_, brief) = post(&app, request_body("t0", 8, false, Some("brief"))).await;
    let (_, full) = post(&app, request_body("t0", 8, false, Some("full"))).await;
    let (_, off) = post(&app, request_body("t0", 8, false, Some("off"))).await;

    // 未指定 = brief (同一の実効 system → 同一出力)
    assert_eq!(
        default["content"][0]["text"], brief["content"][0]["text"],
        "未指定は brief (既定) と同一であるべき"
    );
    // brief は誘導文注入で system が変わる → 状態依存モデルの出力が変わる
    assert_ne!(
        brief["content"][0]["text"], full["content"][0]["text"],
        "brief の注入が prompt に反映されていない"
    );
    // off は空 think block 分 prompt が変わる
    assert_ne!(
        off["content"][0]["text"], full["content"][0]["text"],
        "off の prompt 変更が反映されていない"
    );
}

/// 不正値は 400 invalid_request_error (tool_choice のバリデーション慣習と同一)
#[tokio::test]
async fn thinking_unknown_value_is_rejected_with_400() {
    let app = test_app_with(thinking_sim_engine("think-sim").unwrap(), 8);
    for bad in ["bogus", "", "OFF"] {
        let (status, body) = post(&app, request_body("t0", 4, false, Some(bad))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "thinking={bad:?} body={body}");
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("thinking"), "message={msg}");
    }
}

/// SSE 経路 (submit_stream) も同一の thinking を受ける
#[tokio::test]
async fn thinking_off_streams_immediately_and_full_streams_thinking() {
    let app = test_app_with(thinking_sim_engine("think-sim").unwrap(), 8);

    let (status, body) = post_raw(&app, request_body("t0", 20, true, Some("off"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sse_text(&body), "", "off はテキスト無しで即終了");

    let (status, body) = post_raw(&app, request_body("t0", 20, true, Some("full"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sse_text(&body), "t13 t14 t15", "full は思考フィラを流す");
}

// ---------- s52: Anthropic マップ形式 (Hermes 互換) ----------

/// マップ形式は enabled = off 扱い (思考無効・即答) / disabled = off 扱い。
/// budget_tokens は無視 (決定的)。s53c: 実測で brief (思考 ON + 誘導文) が
/// 短い system で破綻したため、Hermes が送る enabled は off に写像する。
#[tokio::test]
async fn thinking_map_enabled_maps_to_off_and_disabled_is_off() {
    let app = test_app_with(thinking_sim_engine("think-sim").unwrap(), 8);

    // enabled (Hermes が送る形状・budget_tokens 付き) は off 扱い
    let (status, enabled) = post(
        &app,
        request_body_thinking_value("t0", 20, false, json!({"type": "enabled", "budget_tokens": 8000})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {enabled}");
    assert_eq!(
        enabled["usage"]["output_tokens"],
        json!(1),
        "enabled は off 扱いで即答"
    );
    assert_eq!(enabled["content"][0]["text"], "");

    // 文字列 "off" と同一出力 (同一経路・決定的)
    let (_, off) = post(&app, request_body("t0", 20, false, Some("off"))).await;
    assert_eq!(
        enabled["content"][0]["text"], off["content"][0]["text"],
        "マップ enabled は文字列 off と同一経路であるべき"
    );

    // disabled は off 扱い (即 EOS・1 トークン)
    let (status, disabled) = post(
        &app,
        request_body_thinking_value("t0", 20, false, json!({"type": "disabled"})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {disabled}");
    assert_eq!(
        disabled["usage"]["output_tokens"],
        json!(1),
        "disabled は off 扱いで即答"
    );
    assert_eq!(disabled["content"][0]["text"], "", "disabled は EOS のみでテキスト空");
}

/// マップ形式の不正 type (未知 type・type 欠落・非文字列 type) とマップ以外の
/// 値型は 400 invalid_request_error (tool_choice のバリデーション慣習と同一)
#[tokio::test]
async fn thinking_map_unknown_type_is_rejected_with_400() {
    let app = test_app_with(thinking_sim_engine("think-sim").unwrap(), 8);
    for bad in [json!({"type": "bogus"}), json!({}), json!({"type": 1}), json!(1)] {
        let (status, body) =
            post(&app, request_body_thinking_value("t0", 4, false, bad.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "thinking={bad} body={body}");
        assert_eq!(body["type"], "error");
        assert_eq!(body["error"]["type"], "invalid_request_error");
        let msg = body["error"]["message"].as_str().unwrap();
        assert!(msg.contains("thinking"), "message={msg}");
    }
}

/// s52/s53c E2E: Hermes デスクトップが送る同一形状の本文 (thinking マップ +
/// system/message を text block 配列 + cache_control 付き) で 200 を確認する
/// (CPU 経路・実エンジン注入なしの wire テスト)。enabled は off 写像のため
/// 即答 (テキスト空) になる。
#[tokio::test]
async fn thinking_hermes_shaped_request_returns_200() {
    let app = test_app_with(thinking_sim_engine("think-sim").unwrap(), 8);
    let body = json!({
        "model": "denno-test",
        "max_tokens": 1024,
        "temperature": 0.0,
        "system": [{"type": "text", "text": "You are a helpful assistant.", "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "t0", "cache_control": {"type": "ephemeral"}}]}
        ],
        "thinking": {"type": "enabled", "budget_tokens": 8000},
        "stream": false
    })
    .to_string();
    let (status, resp) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {resp}");
    assert_eq!(resp["type"], "message");
    assert_eq!(resp["stop_reason"], "end_turn");
    assert_eq!(
        resp["content"][0]["text"], "",
        "thinking マップ enabled は off 扱い (即答)"
    );
}
