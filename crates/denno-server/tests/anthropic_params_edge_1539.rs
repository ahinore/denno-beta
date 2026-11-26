//! task 1539: Anthropic `/v1/messages` のパラメータ境界テスト。
//!
//! 既存ファイルとの役割分担:
//! - `anthropic_blocks_edge_1264.rs`: system / content ブロックの**検証** (400 系)
//! - `anthropic_parity_976.rs`: 非ストリーミングと SSE の**パリティ**
//! - `anthropic_usage_edge_1169.rs`: stop_reason 別の usage **具体値**
//! - 本ファイル: 受理側の**意味**の境界
//!   - `system` の文字列 / text block 配列が同一プロンプトになること
//!   - `stop_sequences` の一致位置と `stop_reason`
//!     (`stop_sequence` / `max_tokens` / `end_turn` の切り替わり)
//!   - `metadata` と未知フィールドが受理され、結果に影響しないこと
//!   - `thinking` の文字列 / マップ形式 / null の正規化と 400 系
//!   - SSE のイベント順序が stop_reason によらず不変であること
//!
//! fixture は `dummy_engine` (greedy, 語彙 16, EOS = 12)。プロンプト `t0` +
//! 既定 system (brief 誘導文) は 11 トークン、greedy 生成は
//! `t5 t6 t7 t8 t9 t10 t11` の 7 トークン + EOS。すべて固定値で GPU 不要。

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

const MESSAGES: &str = "/v1/messages";
const COUNT_TOKENS: &str = "/v1/messages/count_tokens";

/// greedy 時の全文 (EOS の直前まで)
const GREEDY: &str = "t5 t6 t7 t8 t9 t10 t11";

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

async fn post_raw(app: &Router, uri: &str, body: &str) -> (StatusCode, HeaderMap, String) {
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
    (status, headers, String::from_utf8_lossy(&bytes).into_owned())
}

async fn post_json(app: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let (status, _, text) = post_raw(app, uri, &body.to_string()).await;
    let v = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, v)
}

/// 新しい app (KV キャッシュが冷えた状態) でリクエストする。
/// usage の cache 内訳まで含めて比較できるようにするため。
async fn fresh(uri: &str, body: &Value) -> (StatusCode, Value) {
    post_json(&app(), uri, body).await
}

async fn nonstream(body: Value) -> Value {
    let mut body = body;
    body["stream"] = json!(false);
    let (status, v) = fresh(MESSAGES, &body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    v
}

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    })
}

async fn count_tokens(body: &Value) -> u64 {
    let mut body = body.clone();
    body.as_object_mut().unwrap().remove("stream");
    let (status, v) = fresh(COUNT_TOKENS, &body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    v["input_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("input_tokens が数値でない: {v}"))
}

fn text_of(v: &Value) -> String {
    v["content"]
        .as_array()
        .unwrap_or_else(|| panic!("content が配列でない: {v}"))
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect()
}

fn assert_invalid_envelope(status: StatusCode, v: &Value) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["type"], "error", "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "error.message が空: {v}");
}

/// `id` 以外の応答が一致すること (同一入力 → 同一応答)。
fn assert_same_message(a: &Value, b: &Value) {
    let strip = |v: &Value| {
        let mut v = v.clone();
        v.as_object_mut().unwrap().remove("id");
        v
    };
    assert_eq!(strip(a), strip(b), "応答が一致しない");
}

// ---------------------------------------------------------------------------
// system: 文字列 vs text block 配列
// ---------------------------------------------------------------------------

#[tokio::test]
async fn system_string_and_single_text_block_are_identical() {
    let mut as_string = base_body();
    as_string["system"] = json!("You are a test.");
    let mut as_array = base_body();
    as_array["system"] = json!([{"type": "text", "text": "You are a test."}]);

    let s1 = nonstream(as_string.clone()).await;
    let s2 = nonstream(as_array.clone()).await;
    assert_same_message(&s1, &s2);
    assert_eq!(text_of(&s2), GREEDY);
    assert_eq!(count_tokens(&as_string).await, count_tokens(&as_array).await);
}

#[tokio::test]
async fn system_multiple_text_blocks_match_concatenated_string() {
    let mut blocks = base_body();
    blocks["system"] = json!([
        {"type": "text", "text": "t1 t2"},
        {"type": "text", "text": " t3 t4"},
    ]);
    let mut concat = base_body();
    concat["system"] = json!("t1 t2 t3 t4");

    // 区切り文字を挟まず出現順に連結する。
    assert_eq!(count_tokens(&blocks).await, count_tokens(&concat).await);
    assert_same_message(&nonstream(blocks).await, &nonstream(concat).await);
}

#[tokio::test]
async fn system_block_order_changes_nothing_but_text_is_ordered() {
    // 同じブロック集合なら順序だけが違い、連結結果は「順序どおり」。
    // 連結が順序どおりであることは、逆順の文字列と一致することで確認する。
    let mut forward = base_body();
    forward["system"] = json!([
        {"type": "text", "text": "t1"},
        {"type": "text", "text": "t2"},
    ]);
    let mut reversed = base_body();
    reversed["system"] = json!([
        {"type": "text", "text": "t2"},
        {"type": "text", "text": "t1"},
    ]);
    let mut plain_forward = base_body();
    plain_forward["system"] = json!("t1t2");
    let mut plain_reversed = base_body();
    plain_reversed["system"] = json!("t2t1");

    assert_eq!(
        count_tokens(&forward).await,
        count_tokens(&plain_forward).await
    );
    assert_eq!(
        count_tokens(&reversed).await,
        count_tokens(&plain_reversed).await
    );
}

#[tokio::test]
async fn system_empty_array_equals_empty_string() {
    let mut empty_array = base_body();
    empty_array["system"] = json!([]);
    empty_array["thinking"] = json!("off");
    let mut empty_string = base_body();
    empty_string["system"] = json!("");
    empty_string["thinking"] = json!("off");

    assert_eq!(
        count_tokens(&empty_array).await,
        count_tokens(&empty_string).await
    );
}

#[tokio::test]
async fn system_null_behaves_like_absent() {
    let mut null = base_body();
    null["system"] = Value::Null;
    null["thinking"] = json!("off");
    let mut absent = base_body();
    absent["thinking"] = json!("off");

    assert_eq!(count_tokens(&null).await, count_tokens(&absent).await);
    assert_same_message(&nonstream(null).await, &nonstream(absent).await);
}

#[tokio::test]
async fn system_top_level_wrong_types_are_400() {
    for bad in [json!({"type": "text", "text": "hi"}), json!(7), json!(true)] {
        let mut body = base_body();
        body["system"] = bad.clone();
        let (status, v) = fresh(MESSAGES, &body).await;
        assert_invalid_envelope(status, &v);
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("system must be a string or an array of text blocks"),
            "system = {bad}, message = {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// stop_sequences: 一致位置と stop_reason
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_or_absent_stop_sequences_end_turn() {
    let absent = nonstream(base_body()).await;
    assert_eq!(absent["stop_reason"], "end_turn");
    assert_eq!(absent["stop_sequence"], Value::Null);
    assert_eq!(text_of(&absent), GREEDY);

    let mut empty = base_body();
    empty["stop_sequences"] = json!([]);
    let empty = nonstream(empty).await;
    assert_eq!(empty["stop_reason"], "end_turn");
    assert_eq!(empty["stop_sequence"], Value::Null);
    assert_eq!(text_of(&empty), GREEDY);
}

#[tokio::test]
async fn empty_string_stop_entries_are_ignored() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["", "never"]);
    let v = nonstream(body).await;
    assert_eq!(v["stop_reason"], "end_turn", "v = {v}");
    assert_eq!(text_of(&v), GREEDY);
}

#[tokio::test]
async fn stop_sequence_at_the_final_token_beats_end_turn() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t11"]);
    let v = nonstream(body).await;

    // 最後の生成トークンで一致する。EOS まで進む前に打ち切るので end_turn ではなく
    // stop_sequence になり、停止語自体は本文に残らない。
    assert_eq!(v["stop_reason"], "stop_sequence", "v = {v}");
    assert_eq!(v["stop_sequence"], "t11");
    assert_eq!(text_of(&v), "t5 t6 t7 t8 t9 t10 ");
    assert_eq!(v["usage"]["output_tokens"], 7);
    assert!(!text_of(&v).contains("t11"));
}

#[tokio::test]
async fn stop_sequence_wins_over_max_tokens_at_the_same_boundary() {
    // 7 トークン目 (t11) で stop と max_tokens の境界が重なる。停止判定が
    // 上限判定より先なので stop_sequence になる。
    let mut stop_and_max = base_body();
    stop_and_max["max_tokens"] = json!(7);
    stop_and_max["stop_sequences"] = json!(["t11"]);
    let v = nonstream(stop_and_max).await;
    assert_eq!(v["stop_reason"], "stop_sequence", "v = {v}");
    assert_eq!(v["usage"]["output_tokens"], 7);

    // 1 トークン手前で上限に達すれば停止語は完成せず max_tokens。
    let mut max_only = base_body();
    max_only["max_tokens"] = json!(6);
    max_only["stop_sequences"] = json!(["t11"]);
    let v = nonstream(max_only).await;
    assert_eq!(v["stop_reason"], "max_tokens", "v = {v}");
    assert_eq!(v["stop_sequence"], Value::Null);
    assert_eq!(text_of(&v), "t5 t6 t7 t8 t9 t10");
    assert_eq!(v["usage"]["output_tokens"], 6);
}

#[tokio::test]
async fn multi_token_stop_phrase_truncates_at_its_start() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t9 t10"]);
    let v = nonstream(body).await;

    assert_eq!(v["stop_reason"], "stop_sequence", "v = {v}");
    assert_eq!(v["stop_sequence"], "t9 t10");
    assert_eq!(text_of(&v), "t5 t6 t7 t8 ");
    // 停止語を構成したトークン (t9, t10) は計上されるが本文には残らない。
    assert_eq!(v["usage"]["output_tokens"], 6);
}

#[tokio::test]
async fn multiple_stop_sequences_stop_at_the_earliest_match() {
    // "t7" は "t10" より前に現れる。指定順を入れ替えても結果は同じ
    // (最長一致ではなく最も早い出現)。
    for stop in [json!(["t10", "t7"]), json!(["t7", "t10"])] {
        let mut body = base_body();
        body["stop_sequences"] = stop.clone();
        let v = nonstream(body).await;
        assert_eq!(v["stop_reason"], "stop_sequence", "stop = {stop}");
        assert_eq!(v["stop_sequence"], "t7", "stop = {stop}");
        assert_eq!(text_of(&v), "t5 t6 ", "stop = {stop}");
        assert_eq!(v["usage"]["output_tokens"], 3, "stop = {stop}");
    }
}

#[tokio::test]
async fn absent_stop_sequence_generates_the_full_sequence() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["never", "zzz"]);
    let v = nonstream(body).await;
    assert_eq!(v["stop_reason"], "end_turn", "v = {v}");
    assert_eq!(v["stop_sequence"], Value::Null);
    assert_eq!(text_of(&v), GREEDY);
}

#[tokio::test]
async fn stop_sequences_wrong_types_are_400() {
    for bad in [
        json!("t7"),
        json!([5]),
        json!([{"seq": "t7"}]),
        json!([null]),
        Value::Null,
    ] {
        let mut body = base_body();
        body["stop_sequences"] = bad.clone();
        let (status, v) = fresh(MESSAGES, &body).await;
        assert_invalid_envelope(status, &v);
        assert_eq!(v["error"]["type"], "invalid_request_error", "bad = {bad}");
    }
}

#[tokio::test]
async fn stop_sequences_do_not_change_the_prompt() {
    let base = count_tokens(&base_body()).await;
    for stop in [json!(["t7"]), json!(["t9 t10"]), json!(["t11"])] {
        let mut body = base_body();
        body["stop_sequences"] = stop.clone();
        assert_eq!(count_tokens(&body).await, base, "stop = {stop}");
    }
}

#[tokio::test]
async fn oversized_stop_sequence_list_is_accepted() {
    let big: Vec<String> = (0..2000).map(|i| format!("zz{i}")).collect();
    let mut body = base_body();
    body["stop_sequences"] = json!(big);
    let v = nonstream(body).await;
    assert_eq!(v["stop_reason"], "end_turn", "v = {v}");
    assert_eq!(text_of(&v), GREEDY);
}

// ---------------------------------------------------------------------------
// metadata / 未知フィールド
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metadata_is_accepted_and_ignored() {
    let mut with_meta = base_body();
    with_meta["metadata"] = json!({"user_id": "user-123"});
    let v = nonstream(with_meta.clone()).await;

    assert_same_message(&v, &nonstream(base_body()).await);
    assert_eq!(text_of(&v), GREEDY);
    assert_eq!(
        count_tokens(&with_meta).await,
        count_tokens(&base_body()).await
    );

    // count_tokens 側でも受理される。
    let (status, ct) = fresh(COUNT_TOKENS, &with_meta).await;
    assert_eq!(status, StatusCode::OK, "ct = {ct}");
}

#[tokio::test]
async fn metadata_wrong_shapes_are_still_ignored() {
    for meta in [
        json!("just-a-string"),
        json!(["a", "b"]),
        json!(42),
        Value::Null,
    ] {
        let mut body = base_body();
        body["metadata"] = meta.clone();
        let v = nonstream(body).await;
        assert_eq!(v["stop_reason"], "end_turn", "meta = {meta}");
        assert_eq!(text_of(&v), GREEDY, "meta = {meta}");
    }
}

#[tokio::test]
async fn unknown_top_level_and_message_fields_are_ignored() {
    let mut body = base_body();
    body["metadata"] = json!({"user_id": "u1"});
    body["service_tier"] = json!("auto");
    body["mcp_servers"] = json!([]);
    body["container"] = Value::Null;
    body["custom_unknown"] = json!({"a": [1, 2, 3]});
    body["messages"] = json!([{
        "role": "user",
        "content": [{
            "type": "text",
            "text": "t0",
            "cache_control": {"type": "ephemeral"},
            "citations": [],
            "unknown_block_field": true
        }],
        "name": "ignored",
        "unknown_message_field": {"x": 1}
    }]);

    let v = nonstream(body.clone()).await;
    assert_eq!(v["stop_reason"], "end_turn", "v = {v}");
    assert_eq!(text_of(&v), GREEDY);
    assert_same_message(&v, &nonstream(base_body()).await);
    assert_eq!(
        count_tokens(&body).await,
        count_tokens(&base_body()).await
    );
}

// ---------------------------------------------------------------------------
// thinking: 文字列 / マップ / null
// ---------------------------------------------------------------------------

#[tokio::test]
async fn thinking_enabled_map_is_off_and_budget_is_ignored() {
    let mut off = base_body();
    off["thinking"] = json!("off");
    let off_body = nonstream(off.clone()).await;

    let variants = [
        json!({"type": "enabled"}),
        json!({"type": "enabled", "budget_tokens": 0}),
        json!({"type": "enabled", "budget_tokens": 1_000_000}),
        json!({"type": "disabled"}),
        json!({"type": "disabled", "budget_tokens": -5}),
    ];
    for thinking in variants {
        let mut body = base_body();
        body["thinking"] = thinking.clone();
        let v = nonstream(body.clone()).await;
        assert_eq!(v["stop_reason"], off_body["stop_reason"], "thinking = {thinking}");
        assert_eq!(text_of(&v), text_of(&off_body), "thinking = {thinking}");
        assert_eq!(
            count_tokens(&body).await,
            count_tokens(&off).await,
            "thinking = {thinking}"
        );
    }
}

#[tokio::test]
async fn thinking_null_defaults_to_brief() {
    let mut null = base_body();
    null["thinking"] = Value::Null;
    let mut brief = base_body();
    brief["thinking"] = json!("brief");
    let mut off = base_body();
    off["thinking"] = json!("off");

    assert_eq!(count_tokens(&null).await, count_tokens(&brief).await);
    assert_same_message(&nonstream(null.clone()).await, &nonstream(brief).await);

    // brief は system 冒頭に誘導文を注入するので off とはプロンプトが異なる。
    assert_ne!(count_tokens(&null).await, count_tokens(&off).await);
}

#[tokio::test]
async fn thinking_full_keeps_system_verbatim() {
    // full は system を書き換えない (brief の誘導文を注入しない)。
    let mut full = base_body();
    full["thinking"] = json!("full");
    let mut off = base_body();
    off["thinking"] = json!("off");

    assert_eq!(count_tokens(&full).await, count_tokens(&off).await);
    assert_same_message(&nonstream(full).await, &nonstream(off).await);
}

#[tokio::test]
async fn thinking_invalid_values_are_400_on_both_endpoints() {
    let bad_values = [
        json!("on"),
        json!("OFF"),
        json!(""),
        json!({}),
        json!({"type": 5}),
        json!({"type": "auto"}),
        json!({"budget_tokens": 100}),
        json!(5),
        json!(true),
        json!([]),
    ];
    for thinking in bad_values {
        let mut body = base_body();
        body["thinking"] = thinking.clone();
        let (status, v) = fresh(MESSAGES, &body).await;
        assert_invalid_envelope(status, &v);
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(
            msg.contains("thinking"),
            "thinking = {thinking}, message = {msg}"
        );

        let (status, v) = fresh(COUNT_TOKENS, &body).await;
        assert_invalid_envelope(status, &v);
    }
}

#[tokio::test]
async fn system_array_and_string_agree_under_thinking_brief() {
    let mut as_string = base_body();
    as_string["system"] = json!("You are a test.");
    let mut as_array = base_body();
    as_array["system"] = json!([{"type": "text", "text": "You are a test."}]);
    // brief が既定 (thinking 未指定)
    assert_eq!(
        count_tokens(&as_string).await,
        count_tokens(&as_array).await
    );
    assert_same_message(&nonstream(as_string).await, &nonstream(as_array).await);
}

// ---------------------------------------------------------------------------
// SSE: イベント順序
// ---------------------------------------------------------------------------

/// `event:` 行と `data.type` の一致を確認しつつ、ping を除いたイベント列を返す。
fn sse_ordered(text: &str) -> Vec<Value> {
    let mut events = Vec::new();
    let mut pending: Option<String> = None;
    for line in text.lines() {
        if let Some(e) = line.strip_prefix("event: ") {
            pending = Some(e.trim().to_string());
        } else if let Some(data) = line.strip_prefix("data: ") {
            let v: Value = serde_json::from_str(data).expect("SSE data が JSON ではない");
            let ty = v["type"].as_str().expect("SSE data.type が無い").to_string();
            if ty == "ping" {
                pending = None;
                continue;
            }
            if let Some(e) = pending.take() {
                assert_eq!(e, ty, "event 行と data.type が不一致: {line}");
            }
            events.push(v);
        }
    }
    events
}

async fn stream_events(body: &Value) -> (Vec<Value>, String) {
    let mut body = body.clone();
    body["stream"] = json!(true);
    let (status, headers, text) = post_raw(&app(), MESSAGES, &body.to_string()).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert_eq!(
        headers.get("content-type").and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    let ev = sse_ordered(&text);
    (ev, text)
}

/// stop_reason によらない SSE の不変量を検証し、本文 delta の結合を返す。
fn assert_sse_shape(ev: &[Value], stop_reason: &str) -> String {
    let types: Vec<&str> = ev
        .iter()
        .map(|e| e["type"].as_str().unwrap())
        .collect();

    assert_eq!(
        types.first().copied(),
        Some("message_start"),
        "先頭が message_start でない: {types:?}"
    );
    assert_eq!(
        types.last().copied(),
        Some("message_stop"),
        "末尾が message_stop でない: {types:?}"
    );
    assert_eq!(types.iter().filter(|t| **t == "message_start").count(), 1);
    assert_eq!(types.iter().filter(|t| **t == "message_stop").count(), 1);
    assert_eq!(types.iter().filter(|t| **t == "message_delta").count(), 1);
    assert_eq!(
        types.iter().filter(|t| **t == "content_block_start").count(),
        1
    );
    assert_eq!(
        types.iter().filter(|t| **t == "content_block_stop").count(),
        1
    );

    let pos = |name: &str| {
        types
            .iter()
            .position(|t| *t == name)
            .unwrap_or_else(|| panic!("{name} が無い: {types:?}"))
    };
    let start = pos("content_block_start");
    let stop = pos("content_block_stop");
    let delta = pos("message_delta");
    let end = pos("message_stop");
    assert!(start < stop, "order = {types:?}");
    assert!(stop < delta, "order = {types:?}");
    assert!(delta < end, "order = {types:?}");
    assert_eq!(types[0], "message_start");
    assert_eq!(types[1], "content_block_start", "order = {types:?}");

    // content_block_start と stop の間は delta だけで埋まる。
    for i in start + 1..stop {
        assert_eq!(
            types[i], "content_block_delta",
            "block 区間に余計なイベント: {types:?}"
        );
        assert_eq!(ev[i]["index"], 0, "delta index が 0 でない: {}", ev[i]);
    }

    let start_ev = &ev[0];
    assert_eq!(start_ev["message"]["stop_reason"], Value::Null);
    assert_eq!(start_ev["message"]["stop_sequence"], Value::Null);
    assert!(
        start_ev["message"]["usage"]["input_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );

    let delta_ev = &ev[delta];
    assert_eq!(
        delta_ev["delta"]["stop_reason"], stop_reason,
        "message_delta = {delta_ev}"
    );

    let mut text = String::new();
    for e in ev {
        if e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta" {
            text.push_str(e["delta"]["text"].as_str().unwrap());
        }
    }
    text
}

#[tokio::test]
async fn sse_event_order_end_turn() {
    let (ev, _) = stream_events(&base_body()).await;
    let text = assert_sse_shape(&ev, "end_turn");
    assert_eq!(text, GREEDY);
    let delta = ev.iter().find(|e| e["type"] == "message_delta").unwrap();
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
}

#[tokio::test]
async fn sse_event_order_stop_sequence() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t7"]);
    let (ev, _) = stream_events(&body).await;

    let text = assert_sse_shape(&ev, "stop_sequence");
    assert_eq!(text, "t5 t6");
    assert!(!text.contains("t7"), "停止語が漏れている: {text}");
    let delta = ev.iter().find(|e| e["type"] == "message_delta").unwrap();
    assert_eq!(delta["delta"]["stop_sequence"], "t7");
    assert_eq!(delta["usage"]["output_tokens"], 3);
}

#[tokio::test]
async fn sse_event_order_max_tokens() {
    let mut body = base_body();
    body["max_tokens"] = json!(2);
    let (ev, _) = stream_events(&body).await;

    let text = assert_sse_shape(&ev, "max_tokens");
    assert_eq!(text, "t5 t6");
    let delta = ev.iter().find(|e| e["type"] == "message_delta").unwrap();
    assert_eq!(delta["delta"]["stop_sequence"], Value::Null);
    assert_eq!(delta["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn sse_phrase_stop_holds_back_the_partial_prefix() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t9 t10"]);
    let (ev, _) = stream_events(&body).await;

    let text = assert_sse_shape(&ev, "stop_sequence");
    // 非ストリーミングと同じ位置までで、停止語の prefix ("t9") は一切漏れない。
    // SSE は停止語を構成するトークンの直前の空白も holdback するため、
    // 非ストリーミングより末尾 1 文字 (空白) が短い。
    assert_eq!(text, "t5 t6 t7 t8");
    assert!(!text.contains("t9"), "停止語の prefix が漏れている: {text}");
    let ns = nonstream(body).await;
    assert_eq!(text_of(&ns), format!("{text} "));
}

#[tokio::test]
async fn sse_stop_at_final_token_matches_nonstream() {
    let mut body = base_body();
    body["stop_sequences"] = json!(["t11"]);
    let (ev, _) = stream_events(&body).await;

    let text = assert_sse_shape(&ev, "stop_sequence");
    let ns = nonstream(body).await;
    // 停止語の直前の空白は SSE では holdback される (非ストリーミングは保持)。
    assert_eq!(text_of(&ns), format!("{text} "));
    assert_eq!(text, "t5 t6 t7 t8 t9 t10");
    assert!(!text.contains("t11"));
}
