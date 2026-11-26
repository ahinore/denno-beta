//! Task 1264: `/v1/messages` のブロック検証エッジケース (system の入れ子 / 混在
//! `cache_control` / 非対応メディア型の 400 封筒) の統合テスト。
//!
//! fixture は `tests/anthropic_blocks_991.rs` と同一 (dummy モデル + dummy
//! トークナイザ。GPU 不要)。task 991 が「単純な空 text / 単一の image block」を
//! 固定したのに対し、本ファイルは *構造の入れ子* と *混在 cache_control*、
//! および *非対応メディア型でも封筒が壊れない* ことを固定する。カバー項目:
//!
//! - `system` の要素がさらに配列 (入れ子) の場合は 400
//!   `system[i]: each block must be an object` (index が実際の位置を指す)
//! - `text` の値がオブジェクト/配列 (入れ子) の場合は 400 `system[i].text`
//! - `cache_control` 付きの空 text ブロックも 400 (cache_control の有無で
//!   検証が緩まない)
//! - `cache_control` の混在 (有り/無し/null/空オブジェクト/ttl 付き) は
//!   すべて無視され、同じ連結文字列の単一 system と `input_tokens` /
//!   生成結果が一致する
//! - 非対応メディア型 (`image` / `document` / `audio` / `video`) は
//!   system / message content / count_tokens / stream のすべてで
//!   Anthropic の 400 封筒 (`{"type":"error","error":{...}}`) を返す
//! - 封筒はトップレベルが `type` と `error` のみ、`error` が `type` と
//!   `message` のみ (内部情報や base64 ペイロードを漏らさない)
//! - `source` を欠いた malformed なメディア block でも 500 ではなく 400

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

fn base_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 32,
        "temperature": 0.0,
    })
}

/// Anthropic のエラー封筒を厳密に検証し、`error.message` を返す。
///
/// - status は 400
/// - トップレベル keys は `type` と `error` のみ
/// - `type` は "error"、`error.type` は "invalid_request_error"
/// - `error` の keys は `type` と `message` のみ、`message` は非空
/// - `message` に panic / backtrace / base64 ペイロード片が混ざらない
fn assert_clean_error(
    status: StatusCode,
    headers: &HeaderMap,
    v: &Value,
    must_contain: &[&str],
) -> String {
    assert_eq!(status, StatusCode::BAD_REQUEST, "400 でない: body = {v}");
    let ct = headers
        .get("content-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or_default();
    assert!(
        ct.starts_with("application/json"),
        "エラー封筒が JSON でない (content-type = {ct:?}): body = {v}"
    );

    let top = v.as_object().unwrap_or_else(|| panic!("body が object でない: {v}"));
    let mut top_keys: Vec<&str> = top.keys().map(|s| s.as_str()).collect();
    top_keys.sort_unstable();
    assert_eq!(top_keys, vec!["error", "type"], "封筒のトップレベルが余計: {v}");

    assert_eq!(v["type"], "error", "body = {v}");
    let err = v["error"]
        .as_object()
        .unwrap_or_else(|| panic!("error が object でない: {v}"));
    let mut err_keys: Vec<&str> = err.keys().map(|s| s.as_str()).collect();
    err_keys.sort_unstable();
    assert_eq!(err_keys, vec!["message", "type"], "error のキーが余計: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");

    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "error.message が空: {v}");
    for bad in ["panicked", "RUST_BACKTRACE", "backtrace", "AAAA"] {
        assert!(
            !msg.contains(bad),
            "error.message に {bad:?} が漏れている: {msg}"
        );
    }
    for needle in must_contain {
        assert!(msg.contains(needle), "message に {needle:?} が無い: {msg}");
    }
    msg.to_string()
}

/// 非対応メディア型の block (`type` と `source` だけを持つ最小形)。
fn media_block(ty: &str, media_type: &str) -> Value {
    json!({
        "type": ty,
        "source": {"type": "base64", "media_type": media_type, "data": "AAAA"}
    })
}

// ---------------------------------------------------------------------------
// system の入れ子ブロック
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nested_array_as_system_block_is_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([[{"type": "text", "text": "hi"}]]);
    let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
    let msg = assert_clean_error(status, &headers, &v, &["system[0]", "each block must be an object"]);
    assert!(msg.contains("system[0]"), "message = {msg}");
}

#[tokio::test]
async fn nested_array_after_valid_block_reports_its_index() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "ok"},
        {"type": "text", "text": "fine"},
        [],
    ]);
    let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
    assert_clean_error(status, &headers, &v, &["system[2]", "each block must be an object"]);
}

#[tokio::test]
async fn nested_text_value_is_400() {
    let app = test_app();
    for nested in [
        json!({"type": "text", "text": {"type": "text", "text": ""}}),
        json!({"type": "text", "text": [{"type": "text", "text": "hi"}]}),
    ] {
        let mut body = base_body();
        body["system"] = json!([nested]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        assert_clean_error(status, &headers, &v, &["system[0].text"]);
    }
}

#[tokio::test]
async fn empty_text_block_with_cache_control_is_still_400() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "ok", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "", "cache_control": {"type": "ephemeral"}},
    ]);
    let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
    assert_clean_error(
        status,
        &headers,
        &v,
        &["system[1].text", "non-empty"],
    );
}

#[tokio::test]
async fn whitespace_text_with_cache_control_inside_nested_array_reports_index() {
    let app = test_app();
    let mut body = base_body();
    body["system"] = json!([
        {"type": "text", "text": "a"},
        {"type": "text", "text": "b"},
        {"type": "text", "text": " \n\t ", "cache_control": {"type": "ephemeral", "ttl": "5m"}},
    ]);
    let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
    assert_clean_error(status, &headers, &v, &["system[2].text"]);
}

// ---------------------------------------------------------------------------
// cache_control の混在 (無視される)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn mixed_cache_control_matches_plain_concatenation() {
    let app = test_app();

    let mut mixed = base_body();
    mixed["system"] = json!([
        {"type": "text", "text": "You are "},
        {"type": "text", "text": "a test.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " Done", "cache_control": null},
        {"type": "text", "text": " Now", "cache_control": {}},
        {
            "type": "text",
            "text": " End",
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        },
    ]);
    let mut plain = base_body();
    plain["system"] = json!("You are a test. Done Now End");

    let (s1, _, a) = post_full(&app, "/v1/messages", mixed.clone()).await;
    assert_eq!(s1, StatusCode::OK, "body = {a}");
    let (s2, _, b) = post_full(&app, "/v1/messages", plain.clone()).await;
    assert_eq!(s2, StatusCode::OK, "body = {b}");

    assert_eq!(
        a["usage"]["input_tokens"], b["usage"]["input_tokens"],
        "cache_control の混在で input_tokens が変わる"
    );
    assert_eq!(
        a["content"][0]["text"], b["content"][0]["text"],
        "cache_control の混在で生成結果が変わる"
    );

    let (s3, _, c) = post_full(&app, "/v1/messages/count_tokens", mixed).await;
    assert_eq!(s3, StatusCode::OK, "body = {c}");
    let (s4, _, d) = post_full(&app, "/v1/messages/count_tokens", plain).await;
    assert_eq!(s4, StatusCode::OK, "body = {d}");
    assert_eq!(c["input_tokens"], d["input_tokens"], "count_tokens 不一致");
}

#[tokio::test]
async fn cache_control_on_message_content_text_block_is_ignored() {
    let app = test_app();

    let mut with_cc = base_body();
    with_cc["messages"] = json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": "hello ", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "world", "cache_control": null}
        ]
    }]);
    let mut without = base_body();
    without["messages"] = json!([{"role": "user", "content": "hello world"}]);

    let (s1, _, a) = post_full(&app, "/v1/messages/count_tokens", with_cc).await;
    assert_eq!(s1, StatusCode::OK, "body = {a}");
    let (s2, _, b) = post_full(&app, "/v1/messages/count_tokens", without).await;
    assert_eq!(s2, StatusCode::OK, "body = {b}");
    assert_eq!(a["input_tokens"], b["input_tokens"], "cache_control が効いている");
}

// ---------------------------------------------------------------------------
// 非対応メディア型 → 400 封筒
// ---------------------------------------------------------------------------

const MEDIA: [(&str, &str); 4] = [
    ("image", "image/png"),
    ("document", "application/pdf"),
    ("audio", "audio/wav"),
    ("video", "video/mp4"),
];

#[tokio::test]
async fn unsupported_media_blocks_in_system_are_400_with_index() {
    let app = test_app();
    for (ty, media) in MEDIA {
        let mut body = base_body();
        body["system"] = json!([
            {"type": "text", "text": "ok"},
            media_block(ty, media),
        ]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        let msg = assert_clean_error(status, &headers, &v, &["system[1].type", ty]);
        assert!(!msg.contains(media), "{media} の media_type が message に出ている: {msg}");
    }
}

#[tokio::test]
async fn unsupported_media_blocks_in_message_content_are_400() {
    let app = test_app();
    for (ty, media) in MEDIA {
        let mut body = base_body();
        body["messages"] = json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                media_block(ty, media),
            ],
        }]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        assert_clean_error(
            status,
            &headers,
            &v,
            &["unsupported content block type", ty],
        );
    }
}

#[tokio::test]
async fn unsupported_media_blocks_rejected_before_empty_text_block_in_content() {
    let app = test_app();
    let mut body = base_body();
    body["messages"] = json!([{
        "role": "user",
        "content": [
            {"type": "text", "text": ""},
            media_block("image", "image/jpeg"),
        ],
    }]);
    let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
    assert_clean_error(status, &headers, &v, &["image"]);
}

#[tokio::test]
async fn count_tokens_returns_same_clean_envelope_for_media_blocks() {
    let app = test_app();
    for (ty, media) in MEDIA {
        let mut sys = base_body();
        sys["system"] = json!([media_block(ty, media)]);
        let (s1, h1, v1) = post_full(&app, "/v1/messages/count_tokens", sys).await;
        assert_clean_error(s1, &h1, &v1, &["system[0].type", ty]);

        let mut cnt = base_body();
        cnt["messages"] = json!([{"role": "user", "content": [media_block(ty, media)]}]);
        let (s2, h2, v2) = post_full(&app, "/v1/messages/count_tokens", cnt).await;
        assert_clean_error(s2, &h2, &v2, &["unsupported content block type", ty]);
    }
}

#[tokio::test]
async fn streaming_request_with_media_block_is_json_400_not_sse() {
    let app = test_app();
    for (ty, media) in MEDIA {
        let mut body = base_body();
        body["stream"] = json!(true);
        body["messages"] = json!([{"role": "user", "content": [media_block(ty, media)]}]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        // content-type は assert_clean_error が application/json を要求する
        assert_clean_error(status, &headers, &v, &["unsupported content block type", ty]);
    }
}

#[tokio::test]
async fn malformed_media_block_without_source_is_400_not_500() {
    let app = test_app();
    for (ty, _) in MEDIA {
        let mut body = base_body();
        body["messages"] = json!([{
            "role": "user",
            "content": [{"type": ty}],
        }]);
        let (status, headers, v) = post_full(&app, "/v1/messages", body).await;
        assert_clean_error(status, &headers, &v, &["unsupported content block type", ty]);
    }
}

#[tokio::test]
async fn unknown_block_type_in_system_and_content_is_400() {
    let app = test_app();
    let mut sys = base_body();
    sys["system"] = json!([{"type": "thinking", "thinking": ""}]);
    let (s1, h1, v1) = post_full(&app, "/v1/messages", sys).await;
    assert_clean_error(s1, &h1, &v1, &["system[0].type", "thinking"]);

    let mut msg = base_body();
    msg["messages"] = json!([{"role": "user", "content": [{"type": "thinking"}]}]);
    let (s2, h2, v2) = post_full(&app, "/v1/messages", msg).await;
    assert_clean_error(s2, &h2, &v2, &["unsupported content block type", "thinking"]);
}
