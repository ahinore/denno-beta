//! P837: OpenAI Structured Outputs (`response_format: {"type":"json_schema",...}`) の
//! 統合テスト。
//!
//! denno-engine のダミートークナイザ + 決定的ダミーモデル (tests/openai_chat.rs と
//! 同一方式) で HTTP ワイヤを検証する。カバー項目:
//!
//! - 有効なスキーマで 200 (非ストリーム / SSE)、応答の形
//! - スキーマ指示が **プロンプトへ実際に入る** こと (モデルが受け取った
//!   プロンプトのトークン数と、同一スキーマ 2 回の決定性で確認)
//! - `strict: true` で非対応スキーマ ($ref / oneOf / required 不足) は 400
//! - 壊れたスキーマ (name 無し / schema がオブジェクトでない / strict が bool でない)
//!   は 400
//! - `json_object` / `text` は従来どおり (回帰)

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
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
/// 最初の forward で受け取ったプロンプトのトークン列を記録する
/// (スキーマ指示がプロンプトに入ったことの観測に使う)。
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

/// SSE をそのまま文字列で取り出す。
async fn post_sse(app: &Router, uri: &str, body: Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get("content-type")
        .map(|v| v.to_str().unwrap_or_default().to_string())
        .unwrap_or_default();
    assert!(
        status != StatusCode::OK || ct.starts_with("text/event-stream"),
        "SSE の content-type = {ct}"
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn sse_content(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
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

fn json_schema_body(stream: bool, strict: bool, schema: Value) -> Value {
    let mut b = chat_body(stream);
    b["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "structured_answer",
            "description": "answer with confidence",
            "strict": strict,
            "schema": schema,
        },
    });
    b
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn json_schema_non_streaming_ok_and_hint_reaches_prompt() {
    let (app, seen) = app_with_capture();
    let (status, v) = post(
        &app,
        "/v1/chat/completions",
        json_schema_body(false, true, object_schema()),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["object"], "chat.completion");
    let text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(!text.is_empty(), "content が空: {v}");

    let with_schema = seen.lock().unwrap();
    assert_eq!(with_schema.len(), 1, "forward が記録されていない");
    let prompt_ids = with_schema[0].clone();
    drop(with_schema);

    // 対照: response_format 無しの同じ会話。スキーマ指示ぶんだけプロンプトが
    // 長くなる (指示が実際にプロンプトへ入ったことの証拠)。
    let (app2, seen2) = app_with_capture();
    let (status, _) = post(&app2, "/v1/chat/completions", chat_body(false)).await;
    assert_eq!(status, StatusCode::OK);
    let plain = seen2.lock().unwrap()[0].clone();
    assert!(
        prompt_ids.len() > plain.len() + 10,
        "スキーマ指示がプロンプトに入っていない: schema={} plain={}",
        prompt_ids.len(),
        plain.len()
    );
    assert_ne!(prompt_ids, plain);

    // 同じスキーマを 2 回投げるとプロンプトも出力も同一 (決定的な文字列化)
    let (app3, seen3) = app_with_capture();
    let (status, v3) = post(
        &app3,
        "/v1/chat/completions",
        json_schema_body(false, true, object_schema()),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        v3["choices"][0]["message"]["content"],
        v["choices"][0]["message"]["content"]
    );
    assert_eq!(seen3.lock().unwrap()[0], prompt_ids);
}

#[tokio::test]
async fn json_schema_streaming_ok() {
    let app = app_with_capture().0;
    let (status, text) = post_sse(
        &app,
        "/v1/chat/completions",
        json_schema_body(true, true, object_schema()),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert!(!text.contains("event: "), "event: 行が混ざっている:\n{text}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない:\n{text}"
    );
    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
    }
    assert!(!sse_content(&chunks).is_empty(), "delta が空: {chunks:?}");
}

#[tokio::test]
async fn strict_true_rejects_unsupported_schema() {
    let app = app_with_capture().0;

    // $ref は strict に関係なく非対応
    for strict in [true, false] {
        let schema = json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/a"}},
            "required": ["a"],
            "additionalProperties": false,
        });
        let (status, v) = post(
            &app,
            "/v1/chat/completions",
            json_schema_body(false, strict, schema),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "strict={strict}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(
            v["error"]["param"]
                .as_str()
                .unwrap_or_default()
                .starts_with("response_format.json_schema"),
            "param = {}",
            v["error"]["param"]
        );
        assert!(
            v["error"]["message"].as_str().unwrap().contains("$ref"),
            "message = {}",
            v["error"]["message"]
        );
    }

    // oneOf は strict で 400 / 非 strict では受理
    let one_of = json!({
        "oneOf": [{"type": "string"}, {"type": "integer"}],
    });
    let (status, v) = post(
        &app,
        "/v1/chat/completions",
        json_schema_body(false, true, one_of.clone()),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("oneOf"),
        "message = {}",
        v["error"]["message"]
    );
    let (status, v) = post(
        &app,
        "/v1/chat/completions",
        json_schema_body(false, false, one_of),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "非 strict は受理する: {v}");

    // strict は additionalProperties: false と required の全列挙を要求する
    let no_addl = json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a"],
    });
    let (status, v) = post(
        &app,
        "/v1/chat/completions",
        json_schema_body(false, true, no_addl),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("additionalProperties"),
        "message = {}",
        v["error"]["message"]
    );

    let missing_required = json!({
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
        "additionalProperties": false,
    });
    let (status, v) = post(
        &app,
        "/v1/chat/completions",
        json_schema_body(false, true, missing_required),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert!(
        v["error"]["message"].as_str().unwrap().contains("required"),
        "message = {}",
        v["error"]["message"]
    );
}

#[tokio::test]
async fn malformed_json_schema_is_400() {
    let app = app_with_capture().0;

    // json_schema オブジェクト自体が無い
    let mut body = chat_body(false);
    body["response_format"] = json!({"type": "json_schema"});
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");

    // name 無し
    let mut body = chat_body(false);
    body["response_format"] =
        json!({"type": "json_schema", "json_schema": {"schema": {"type": "string"}}});
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("name"));

    // name が空文字
    let mut body = chat_body(false);
    body["response_format"] =
        json!({"type": "json_schema", "json_schema": {"name": "", "schema": {"type": "string"}}});
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    // schema がオブジェクトでない
    let mut body = chat_body(false);
    body["response_format"] =
        json!({"type": "json_schema", "json_schema": {"name": "n", "schema": "nope"}});
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        v["error"]["message"].as_str().unwrap().contains("schema"),
        "message = {}",
        v["error"]["message"]
    );

    // strict が bool でない
    let mut body = chat_body(false);
    body["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {"name": "n", "strict": "yes", "schema": {"type": "string"}},
    });
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("boolean"));

    // 未知の type は従来どおり 400
    let mut body = chat_body(false);
    body["response_format"] = json!({"type": "yaml"});
    let (status, v) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "response_format.type");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("json_schema"));

    // type が文字列でない
    let mut body = chat_body(false);
    body["response_format"] = json!({"type": 3});
    let (status, _) = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn text_and_json_object_unchanged() {
    let app = app_with_capture().0;
    for fmt in [json!({"type": "text"}), json!({"type": "json_object"})] {
        let mut body = chat_body(false);
        body["response_format"] = fmt.clone();
        let (status, v) = post(&app, "/v1/chat/completions", body).await;
        assert_eq!(status, StatusCode::OK, "fmt={fmt}: {v}");
        assert!(!v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty());
    }
    // text は response_format 無しと同一の入力 (プロンプト不変)
    let (app_a, seen_a) = app_with_capture();
    let mut body = chat_body(false);
    body["response_format"] = json!({"type": "text"});
    let (status, _) = post(&app_a, "/v1/chat/completions", body).await;
    assert_eq!(status, StatusCode::OK);
    let (app_b, seen_b) = app_with_capture();
    let (status, _) = post(&app_b, "/v1/chat/completions", chat_body(false)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(seen_a.lock().unwrap()[0], seen_b.lock().unwrap()[0]);
}

#[tokio::test]
async fn unrelated_endpoints_still_work() {
    // /v1/messages が json_schema 追加の影響を受けていない (回帰)
    let app = app_with_capture().0;
    let (status, v) = post(
        &app,
        "/v1/messages",
        json!({
            "model": "denno-test",
            "system": "You are a test.",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["type"], "message");
}

/// ダミーモデルの 1 forward が「プロンプト全体」であることの前提を固定する
/// (このテストが壊れたら上のプロンプト観測は無意味になる)。
#[tokio::test]
async fn first_forward_receives_whole_prompt() {
    let (app, seen) = app_with_capture();
    let (status, v) = post(&app, "/v1/chat/completions", chat_body(false)).await;
    assert_eq!(status, StatusCode::OK);
    let prompt = seen.lock().unwrap()[0].clone();
    let input_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap() as usize;
    assert_eq!(prompt.len(), input_tokens);
    assert!(prompt.len() > 1);
}
