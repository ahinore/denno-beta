//! task 1616: `response_format` の監査 (`POST /v1/chat/completions`)。
//!
//! 対象は OpenAI 互換クライアントが実際に送る形:
//! `text` / `json_object` / `json_schema` / 未指定 / `null` / 未知 type /
//! 非オブジェクト。非ストリームと SSE の両経路で、
//! 「受理されるものは 200」「不正なものは OpenAI 形式の 400 エンベロープ」
//! を固定する。ダミーモデル (tests/openai_json_schema.rs と同方式) を使い、
//! sleep は使わない。
//!
//! 監査結果の要点:
//! - 検証は `chat_completions` の stream 分岐より前に行われるため、
//!   SSE 要求でも response_format の不正は SSE ではなく JSON の 400 になる。
//! - `null` / 未指定 / `{"type":"text"}` はプロンプト不変 (同一トークン列)。
//! - 未知 type・非オブジェクトは `param = "response_format.type"` の 400
//!   (既存契約。openai_json_schema.rs / json_schema_edge_999.rs と同じ)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 512;

/// 決定的ダミーモデル。peak = 直前トークン + 1。
/// 最初の forward で受け取ったプロンプトのトークン列を記録する。
struct CaptureModel {
    seen: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl Model for CaptureModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        {
            let mut seen = self.seen.lock().unwrap();
            if seen.is_empty() {
                seen.push(tokens.to_vec());
            }
        }
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = (last + 1) % DUMMY_VOCAB;
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn app_with_capture() -> (Router, Arc<Mutex<Vec<Vec<u32>>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(CaptureModel { seen: seen.clone() }),
        tok,
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
    (denno_server::router(state), seen)
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

/// `response_format` 抜きの body に値を挿す (`Value::Null` を渡すと JSON null)。
fn body_with_format(stream: bool, fmt: Value) -> Value {
    let mut b = chat_body(stream);
    b["response_format"] = fmt;
    b
}

async fn post_raw(app: &Router, body: &str) -> (StatusCode, Option<String>, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap_or_default().to_string());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ct, String::from_utf8(bytes.to_vec()).unwrap())
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let (status, _, text) = post_raw(app, &body.to_string()).await;
    let v: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
    (status, v)
}

fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

/// エラーの共通形。`error` は message / type / param / code の 4 キーだけ。
fn assert_error_envelope(v: &Value, label: &str) {
    let err = v
        .get("error")
        .unwrap_or_else(|| panic!("{label}: error が無い: {v}"));
    let obj = err.as_object().unwrap_or_else(|| panic!("{label}: error が object でない: {v}"));
    let mut keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["code", "message", "param", "type"],
        "{label}: error のキー集合: {v}"
    );
    assert_eq!(err["type"], "invalid_request_error", "{label}: {v}");
    assert!(err["code"].is_null(), "{label}: code は null: {v}");
    assert!(
        err["param"].as_str().map(|p| !p.is_empty()).unwrap_or(false),
        "{label}: param が空: {v}"
    );
    let msg = err["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "{label}: message が空: {v}");
    for leak in ["panicked", "unwrap()", "src/"] {
        assert!(!msg.contains(leak), "{label}: message に内部情報: {msg}");
    }
}

fn object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": ["answer", "confidence"],
        "additionalProperties": false,
    })
}

/// OpenAI SDK が送る形 (`json_schema` キーの中に name/strict/schema)。
fn sdk_json_schema_format() -> Value {
    json!({
        "type": "json_schema",
        "json_schema": {
            "name": "structured_answer",
            "description": "answer with confidence",
            "strict": true,
            "schema": object_schema(),
        },
    })
}

// ---------------------------------------------------------------------------
// 受理される形 (200)
// ---------------------------------------------------------------------------

/// 未指定 / null / `{"type":"text"}` は同じプロンプト・同じ出力になる。
#[tokio::test]
async fn text_null_and_omitted_are_prompt_identical() {
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    let mut contents: Vec<String> = Vec::new();
    for fmt in [None, Some(Value::Null), Some(json!({"type": "text"}))] {
        let (app, seen) = app_with_capture();
        let body = match &fmt {
            None => chat_body(false),
            Some(f) => body_with_format(false, f.clone()),
        };
        let (status, v) = post(&app, body).await;
        assert_eq!(status, StatusCode::OK, "fmt={fmt:?}: {v}");
        assert_eq!(v["object"], "chat.completion", "fmt={fmt:?}");
        prompts.push(seen.lock().unwrap()[0].clone());
        contents.push(v["choices"][0]["message"]["content"].as_str().unwrap().to_string());
    }
    assert_eq!(prompts[0], prompts[1], "null がプロンプトを変えている");
    assert_eq!(prompts[0], prompts[2], "text がプロンプトを変えている");
    assert_eq!(contents[0], contents[1]);
    assert_eq!(contents[0], contents[2]);
}

/// `json_object` は指示がプロンプトへ入り、未知のトップレベルキーは無視される。
#[tokio::test]
async fn json_object_accepted_and_hint_reaches_prompt() {
    let (app, seen) = app_with_capture();
    let fmt = json!({"type": "json_object", "note": "client extension"});
    let (status, v) = post(&app, body_with_format(false, fmt)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let with_hint = seen.lock().unwrap()[0].clone();

    let (app2, seen2) = app_with_capture();
    let (status, v2) = post(&app2, chat_body(false)).await;
    assert_eq!(status, StatusCode::OK);
    let plain = seen2.lock().unwrap()[0].clone();
    assert!(
        with_hint.len() > plain.len() + 5,
        "json_object の指示がプロンプトに入っていない: hint={} plain={}",
        with_hint.len(),
        plain.len()
    );
    assert_eq!(v["choices"][0]["message"]["content"], v2["choices"][0]["message"]["content"]);
}

/// `json_object` は SSE でも 200 で chunk が返る。
#[tokio::test]
async fn json_object_stream_accepted() {
    let (app, _) = app_with_capture();
    let body = body_with_format(true, json!({"type": "json_object"})).to_string();
    let (status, ct, text) = post_raw(&app, &body).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert!(
        ct.as_deref().unwrap_or_default().starts_with("text/event-stream"),
        "content-type = {ct:?}"
    );
    assert!(text.trim_end().ends_with("data: [DONE]"), "終端が [DONE] でない");
    let chunks = sse_chunks(&text);
    assert!(!chunks.is_empty(), "chunk が無い");
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
    }
}

/// OpenAI SDK 形の json_schema が非ストリーム / SSE の両方で 200。
#[tokio::test]
async fn json_schema_sdk_payload_accepted_both_routes() {
    let (app, seen) = app_with_capture();
    let (status, v) = post(&app, body_with_format(false, sdk_json_schema_format())).await;
    assert_eq!(status, StatusCode::OK, "non-stream: {v}");
    assert_eq!(v["object"], "chat.completion");
    assert!(!seen.lock().unwrap()[0].is_empty());

    let (app2, _) = app_with_capture();
    let body = body_with_format(true, sdk_json_schema_format()).to_string();
    let (status, ct, text) = post_raw(&app2, &body).await;
    assert_eq!(status, StatusCode::OK, "stream: {text}");
    assert!(
        ct.as_deref().unwrap_or_default().starts_with("text/event-stream"),
        "content-type = {ct:?}"
    );
    assert!(text.trim_end().ends_with("data: [DONE]"));
    assert!(!sse_chunks(&text).is_empty());
}

/// `strict` 省略 (既定 false) と、response_format/json_schema 直下の
/// 未知キーは受理される。非 strict なので oneOf 等も通る。
#[tokio::test]
async fn json_schema_optional_strict_and_unknown_keys_accepted() {
    let (app, _) = app_with_capture();
    let fmt = json!({
        "type": "json_schema",
        "extra": 1,
        "json_schema": {
            "name": "flexible",
            "schema": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
            "unknown_field": {"x": true},
        },
    });
    let (status, v) = post(&app, body_with_format(false, fmt)).await;
    assert_eq!(status, StatusCode::OK, "strict 省略は既定 false: {v}");

    // strict: false を明示しても同じ
    let (app2, _) = app_with_capture();
    let fmt = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "flexible",
            "strict": false,
            "schema": {"oneOf": [{"type": "string"}]},
        },
    });
    let (status, v) = post(&app2, body_with_format(false, fmt)).await;
    assert_eq!(status, StatusCode::OK, "strict=false: {v}");
}

// ---------------------------------------------------------------------------
// 400 になる形 (OpenAI エンベロープ)
// ---------------------------------------------------------------------------

/// 未知 type / type の型違い / type 欠落 / 非オブジェクトは
/// `param = "response_format.type"` の 400 (既存契約の固定)。
/// `response_format: null` は未指定と同じで 400 にならない。
#[tokio::test]
async fn malformed_response_format_is_clean_400() {
    let app = app_with_capture().0;
    let cases: Vec<(&str, Value)> = vec![
        ("未知の type", json!({"type": "yaml"})),
        ("type が null", json!({"type": null})),
        ("type が数値", json!({"type": 7})),
        ("type が bool", json!({"type": true})),
        ("type 欠落", json!({})),
        ("文字列", json!("json_object")),
        ("配列", json!([{"type": "json_object"}])),
        ("数値", json!(3)),
        ("bool", json!(true)),
    ];
    for (label, fmt) in cases {
        let (status, v) = post(&app, body_with_format(false, fmt.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}: {v}");
        assert_error_envelope(&v, label);
        assert_eq!(
            v["error"]["param"], "response_format.type",
            "{label}: param: {v}"
        );
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("response_format"),
            "{label}: message: {v}"
        );
    }

    // null は「未指定」と同じ扱い (400 ではない)
    let (status, v) = post(&app, body_with_format(false, Value::Null)).await;
    assert_eq!(status, StatusCode::OK, "null: {v}");
}

/// `json_schema` 側の形の不正は `param = "response_format.json_schema"` の 400。
#[tokio::test]
async fn malformed_json_schema_is_clean_400() {
    let app = app_with_capture().0;
    let cases: Vec<(&str, Value)> = vec![
        ("json_schema 欠落", json!({"type": "json_schema"})),
        (
            "json_schema が文字列",
            json!({"type": "json_schema", "json_schema": "nope"}),
        ),
        (
            "name 欠落",
            json!({"type": "json_schema", "json_schema": {"schema": {"type": "string"}}}),
        ),
        (
            "name が空",
            json!({"type": "json_schema",
                   "json_schema": {"name": "  ", "schema": {"type": "string"}}}),
        ),
        (
            "schema 欠落",
            json!({"type": "json_schema", "json_schema": {"name": "n"}}),
        ),
        (
            "schema が数値",
            json!({"type": "json_schema",
                   "json_schema": {"name": "n", "schema": 1}}),
        ),
        (
            "strict が文字列",
            json!({"type": "json_schema",
                   "json_schema": {"name": "n", "strict": "yes",
                                   "schema": {"type": "string"}}}),
        ),
        (
            "strict: true で $ref",
            json!({"type": "json_schema",
                   "json_schema": {"name": "n", "strict": true,
                                   "schema": {"$ref": "#/$defs/x"}}}),
        ),
    ];
    for (label, fmt) in cases {
        let (status, v) = post(&app, body_with_format(false, fmt.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}: {v}");
        assert_error_envelope(&v, label);
        let param = v["error"]["param"].as_str().unwrap_or_default();
        assert!(
            param.starts_with("response_format.json_schema"),
            "{label}: param = {param}"
        );
    }
}

/// stream: true でも response_format の検証は先に走り、SSE ではなく
/// JSON の 400 を返す ([DONE] を流さない)。
#[tokio::test]
async fn invalid_format_on_stream_returns_plain_json_400() {
    let app = app_with_capture().0;
    for (label, fmt) in [
        ("未知の type", json!({"type": "yaml"})),
        ("非オブジェクト", json!("json_object")),
        (
            "壊れた json_schema",
            json!({"type": "json_schema", "json_schema": {"name": "n", "schema": 1}}),
        ),
    ] {
        let body = body_with_format(true, fmt).to_string();
        let (status, ct, text) = post_raw(&app, &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}: {text}");
        let ct = ct.unwrap_or_default();
        assert!(
            ct.starts_with("application/json"),
            "{label}: SSE ヘッダが付いている: content-type = {ct}"
        );
        assert!(!text.contains("data: [DONE]"), "{label}: SSE 本文: {text}");
        let v: Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{label}: JSON でない: {e}: {text}"));
        assert_error_envelope(&v, label);
    }
}
