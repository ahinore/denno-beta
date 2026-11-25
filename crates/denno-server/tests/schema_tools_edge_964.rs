//! Task 964: OpenAI tool calling / JSON Schema のエッジケース統合テスト。
//!
//! `openai_tools.rs` / `openai_json_schema.rs` の正常系を補い、境界だけを見る:
//!
//! - 入れ子 object / array / enum / nullable / optional / 深いネストのスキーマ
//! - 壊れた `response_format.json_schema` が **すべて** きれいな 400 になること
//!   (エラー本文に panic の痕跡や空 message が無いこと)
//! - 複数 tool 定義 + `tool_choice` の none / auto / 関数名指定と、
//!   ストリーミング `delta.tool_calls` の framing 不変条件
//! - tool 引数 / スキーマ内の日本語・Unicode・エスケープされた引用符
//!
//! 生成は denno-engine の ScriptModel (固定トークン列) とダミーモデルだけを使う
//! (モデル DL 不要・決定的)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{
    dummy_tokenizer, tool_tokenizer, tool_tokenizer_with, DummyModel, ScriptModel, TOOL_VOCAB,
};
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::Engine;
use denno_server::json_schema::{JsonSchemaSpec, MAX_SCHEMA_DEPTH};
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// app / HTTP ヘルパ
// ---------------------------------------------------------------------------

/// `DummyModel` (tool_call を出さない) の app。
fn plain_app() -> Router {
    let engine = Engine::from_parts(
        Box::new(DummyModel),
        dummy_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    state_app(engine)
}

/// `DummyModel` と同じ生成規則で、コンテキスト長だけ 512 に広げたモデル。
/// スキーマ指示を含むプロンプトは DUMMY_CONTEXT (64) を超えるため、
/// 受理されるスキーマの HTTP テストはこちらを使う。
struct WideDummyModel(DummyModel);

impl Model for WideDummyModel {
    fn context(&self) -> usize {
        512
    }

    fn eos_ids(&self) -> &[u32] {
        self.0.eos_ids()
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        self.0.forward(tokens, cache)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, 512)
    }
}

/// 広いコンテキストの `DummyModel` app。
fn wide_app() -> Router {
    let engine = Engine::from_parts(
        Box::new(WideDummyModel(DummyModel)),
        dummy_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    state_app(engine)
}

/// `ScriptModel` + 差し替え可能なトークナイザの app。
fn script_app(script: Vec<u32>, tok: DennoTokenizer) -> Router {
    let engine = Engine::from_parts(
        Box::new(ScriptModel::new(script, tid("t1"))),
        tok,
        "denno-test",
        None,
    );
    state_app(engine)
}

fn state_app(engine: Engine) -> Router {
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

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

/// 生の本文 (JSON として壊れているケース用)。
async fn post_raw(app: &Router, body: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// SSE 本文をそのまま取り出す (content-type も検査する)。
async fn post_sse(app: &Router, body: Value) -> (StatusCode, String) {
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

// ---------------------------------------------------------------------------
// tool script ヘルパ
// ---------------------------------------------------------------------------

fn tid(piece: &str) -> u32 {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
        .unwrap_or_else(|| panic!("vocab missing: {piece}"))
}

/// `<tool_call>` block を 1 個吐くスクリプト (本文 + 引数 value トークン列)。
/// `value_pieces` は TOOL_VOCAB の piece 名で、`<parameter=city>` の値になる。
fn block_script(name: &str, value_pieces: &[&str], extra: &dyn Fn(&str) -> u32) -> Vec<u32> {
    let mut pieces = vec![
        "Checking.",
        "<tool_call>",
        "Ċ",
        "<function=",
        name,
        ">",
        "Ċ",
        "<parameter=",
        "city",
        ">",
        "Ċ",
    ];
    pieces.extend_from_slice(value_pieces);
    pieces.extend_from_slice(&[
        "Ċ",
        "</parameter>",
        "Ċ",
        "</function>",
        "Ċ",
        "</tool_call>",
        "eos",
    ]);
    pieces
        .into_iter()
        .map(|p| match tid_opt(p) {
            Some(id) => id,
            None => extra(p),
        })
        .collect()
}

fn tid_opt(piece: &str) -> Option<u32> {
    TOOL_VOCAB
        .iter()
        .find(|(p, _)| *p == piece)
        .map(|(_, id)| *id)
}

fn chat_body(stream: bool, tools: Option<Value>) -> Value {
    let mut body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": stream,
    });
    if let Some(t) = tools {
        body["tools"] = t;
    }
    body
}

/// 複数 tool 定義 (3 個)。tool_choice の各モードで使い回す。
fn three_tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_tool",
                "description": "Run the tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {"type": "string"},
                        "env": {"type": "object", "additionalProperties": {"type": "string"}}
                    },
                    "required": ["cmd"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_docs",
                "description": "Search the documentation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "enum": [5, 10, 20]}
                    },
                    "required": ["query"]
                }
            }
        },
    ])
}

fn json_schema_body(stream: bool, strict: bool, schema: Value) -> Value {
    let mut body = chat_body(stream, None);
    body["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "edge_schema",
            "strict": strict,
            "schema": schema,
        },
    });
    body
}

// ---------------------------------------------------------------------------
// 1. 入れ子 / array / enum / nullable / optional / 深いネスト
// ---------------------------------------------------------------------------

#[test]
fn nested_object_array_enum_schema_is_accepted_by_validator() {
    // strict: true でも通る「全 property が required + additionalProperties:false」の
    // 入れ子 object、array of array、enum、null 型を 1 枚にまとめる。
    let schema = json!({
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string", "enum": ["admin", "user", "guest"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "manager": {"type": "null"}
                },
                "required": ["name", "role", "tags", "manager"],
                "additionalProperties": false
            },
            "scores": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}}
            }
        },
        "required": ["user", "scores"],
        "additionalProperties": false
    });
    let spec = JsonSchemaSpec::parse(&json!({
        "name": "nested",
        "strict": true,
        "schema": schema,
    }))
    .expect("入れ子スキーマが拒否された");
    assert!(spec.strict);
    // hint は決定的で、enum の値をそのまま含む (日本語なし版)
    let h1 = spec.hint();
    assert!(h1.contains("\"enum\":[\"admin\",\"user\",\"guest\"]"), "{h1}");
    assert_eq!(h1, spec.hint());
}

#[test]
fn optional_property_is_ok_non_strict_but_rejected_strict() {
    // optional field は非 strict なら OK、strict は required 全列挙を要求する。
    let schema = json!({
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "nickname": {"type": "string"}
        },
        "required": ["id"],
        "additionalProperties": false
    });
    assert!(JsonSchemaSpec::parse(&json!({"name": "o", "schema": schema.clone()})).is_ok());
    let err = JsonSchemaSpec::parse(&json!({
        "name": "o", "strict": true, "schema": schema
    }))
    .unwrap_err();
    assert!(err.contains("required") && err.contains("nickname"), "{err}");
}

#[test]
fn nullable_union_type_form_is_rejected_with_pointer() {
    // JSON Schema の nullable 表記 type:["string","null"] はこのサーバでは
    // 未対応 (type は文字列のみ)。非 strict でも 400 で、パスが返る。
    let err = JsonSchemaSpec::parse(&json!({
        "name": "n",
        "schema": {
            "type": "object",
            "properties": {"note": {"type": ["string", "null"]}},
            "required": ["note"],
            "additionalProperties": false
        }
    }))
    .unwrap_err();
    assert!(err.contains("$.properties.note.type"), "{err}");
    assert!(err.contains("must be a string"), "{err}");
}

#[test]
fn deeply_nested_schemas_hit_exact_depth_boundary() {
    // depth limit は「root = 0、入れ子 1 段ごとに +1、> MAX で 400」。
    let nested_arrays = |n: usize| {
        let mut s = json!({"type": "string"});
        for _ in 0..n {
            s = json!({"type": "array", "items": s});
        }
        s
    };
    let at_limit = json!({"name": "deep", "schema": nested_arrays(MAX_SCHEMA_DEPTH)});
    assert!(
        JsonSchemaSpec::parse(&at_limit).is_ok(),
        "深度 {MAX_SCHEMA_DEPTH} は受理されるべき"
    );
    let over = json!({"name": "deep", "schema": nested_arrays(MAX_SCHEMA_DEPTH + 1)});
    let err = JsonSchemaSpec::parse(&over).unwrap_err();
    assert!(err.contains("nested too deeply"), "{err}");
}

#[tokio::test]
async fn nested_schema_round_trips_through_http() {
    let app = wide_app();
    let schema = json!({
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}}
                },
                "required": ["inner"],
                "additionalProperties": false
            }
        },
        "required": ["outer"],
        "additionalProperties": false
    });
    let (status, v) = post(&app, json_schema_body(false, true, schema)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
}

// ---------------------------------------------------------------------------
// 2. 壊れた response_format.json_schema はすべて 400
// ---------------------------------------------------------------------------

/// `json_schema` の中身を差し替えたリクエスト。
fn with_json_schema(value: Value) -> Value {
    let mut body = chat_body(false, None);
    body["response_format"] = json!({"type": "json_schema", "json_schema": value});
    body
}

#[tokio::test]
async fn malformed_json_schema_payloads_are_clean_400() {
    let app = plain_app();

    let deep = {
        let mut s = json!({"type": "string"});
        for _ in 0..(MAX_SCHEMA_DEPTH + 1) {
            s = json!({"type": "array", "items": s});
        }
        s
    };
    let long = "x".repeat(16 * 1024);

    let cases: Vec<(&str, Value, &str)> = vec![
        ("json_schema が文字列", json!("nope"), "must be a JSON object"),
        ("json_schema が数値", json!(3), "must be a JSON object"),
        ("json_schema が配列", json!([]), "must be a JSON object"),
        ("name 欠落", json!({"schema": {"type": "string"}}), "name is required"),
        ("name が非文字列", json!({"name": 7, "schema": {"type": "string"}}), "name"),
        ("name が空白のみ", json!({"name": "  ", "schema": {"type": "string"}}), "non-empty"),
        ("description が非文字列", json!({"name": "n", "description": 1, "schema": {"type": "string"}}), "description"),
        ("strict が非 bool", json!({"name": "n", "strict": "yes", "schema": {"type": "string"}}), "boolean"),
        ("schema 欠落", json!({"name": "n"}), "schema is required"),
        ("schema が null", json!({"name": "n", "schema": Value::Null}), "must be a JSON object or a boolean"),
        ("schema が数値", json!({"name": "n", "schema": 3}), "must be a JSON object or a boolean"),
        ("properties が配列", json!({"name": "n", "schema": {"type": "object", "properties": []}}), "properties must be a JSON object"),
        ("type が配列 (nullable 形式)", json!({"name": "n", "schema": {"type": ["string", "null"]}}), "type must be a string"),
        ("未知の type", json!({"name": "n", "schema": {"type": "tuple"}}), "unsupported type"),
        ("items がタプル形式", json!({"name": "n", "schema": {"type": "array", "items": [{"type": "string"}]}}), "tuple form is not supported"),
        ("additionalProperties が文字列", json!({"name": "n", "schema": {"type": "object", "properties": {}, "additionalProperties": "yes"}}), "additionalProperties"),
        ("入れ子の $ref", json!({"name": "n", "schema": {"type": "object", "properties": {"a": {"$ref": "#/x"}}, "required": ["a"], "additionalProperties": false}}), "$ref"),
        ("サイズ超過", json!({"name": "n", "schema": {"type": "string", "description": long}}), "too large"),
        ("深すぎるネスト", json!({"name": "n", "schema": deep}), "nested too deeply"),
        ("strict で additionalProperties 欠落", json!({"name": "n", "strict": true, "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}}), "additionalProperties"),
        ("strict で required 不足", json!({"name": "n", "strict": true, "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "required": [], "additionalProperties": false}}), "required"),
    ];

    for (label, value, needle) in cases {
        let (status, v) = post(&app, with_json_schema(value)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{label}: {v}");
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(!msg.is_empty(), "{label}: 空の message");
        assert!(msg.contains(needle), "{label}: message={msg:?} (期待 {needle:?})");
        let param = v["error"]["param"].as_str().unwrap_or_default();
        assert!(
            param.starts_with("response_format"),
            "{label}: param={param:?}"
        );
        // panic / 内部エラー文字列が漏れていない
        assert!(!msg.contains("panicked"), "{label}: {msg}");
        assert!(!msg.contains("RUST_BACKTRACE"), "{label}: {msg}");
    }
}

#[tokio::test]
async fn strict_rejects_each_unsupported_keyword() {
    // strict: true では下記キーワードすべてが 400、非 strict では受理される。
    let keywords = [
        "oneOf",
        "anyOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "patternProperties",
        "$defs",
        "definitions",
        "prefixItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "dependentSchemas",
    ];
    for kw in keywords {
        let mut schema = json!({"type": "object", "properties": {}});
        schema[kw] = if kw == "not" { json!({"type": "string"}) } else { json!({}) };
        let app = plain_app();
        let (status, v) = post(&app, json_schema_body(false, true, schema.clone())).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{kw}: {v}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains(kw),
            "{kw}: message={}",
            v["error"]["message"]
        );
        let (status, v) = post(&app, json_schema_body(false, false, schema)).await;
        assert_eq!(status, StatusCode::OK, "{kw} は非 strict で受理: {v}");
    }
}

#[tokio::test]
async fn malformed_json_body_and_response_format_shape_are_400() {
    let app = plain_app();
    let (status, v) = post_raw(&app, "{\"messages\": [").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("invalid request body"));

    // response_format 自体がオブジェクトでない
    let mut body = chat_body(false, None);
    body["response_format"] = json!("json_schema");
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "response_format.type");

    // json_schema キー欠落
    let mut body = chat_body(false, None);
    body["response_format"] = json!({"type": "json_schema"});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["param"], "response_format.json_schema");
}

// ---------------------------------------------------------------------------
// 3. 複数 tool + tool_choice (none / auto / 関数名) とストリーム framing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_tools_tool_choice_none_disables_extraction() {
    let script = block_script("get_weather", &["Tokyo"], &|p| tid(p));
    let app = script_app(script, tool_tokenizer().unwrap());
    let mut body = chat_body(false, Some(three_tools()));
    body["tool_choice"] = json!("none");
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let msg = &v["choices"][0]["message"];
    assert!(msg.get("tool_calls").is_none(), "message = {msg}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    let text = msg["content"].as_str().unwrap();
    assert!(text.contains("<tool_call>"), "content = {text:?}");
    assert!(text.contains("get_weather"), "content = {text:?}");
}

#[tokio::test]
async fn multi_tools_tool_choice_auto_and_named() {
    // auto: tools が有効なので block が抽出される
    let app = script_app(
        block_script("get_weather", &["Tokyo"], &|p| tid(p)),
        tool_tokenizer().unwrap(),
    );
    let mut body = chat_body(false, Some(three_tools()));
    body["tool_choice"] = json!("auto");
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );

    // 関数名指定 (3 個のうち 1 個)
    let app = script_app(
        block_script("get_weather", &["Tokyo"], &|p| tid(p)),
        tool_tokenizer().unwrap(),
    );
    let mut body = chat_body(false, Some(three_tools()));
    body["tool_choice"] = json!({"type": "function", "function": {"name": "get_weather"}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "calls = {calls:?}");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: Value = serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap())
        .expect("arguments が JSON でない");
    assert_eq!(args, json!({"city": "Tokyo"}));
}

#[tokio::test]
async fn multi_tools_tool_choice_none_streaming_has_no_tool_deltas() {
    let app = script_app(
        block_script("get_weather", &["Tokyo"], &|p| tid(p)),
        tool_tokenizer().unwrap(),
    );
    let mut body = chat_body(true, Some(three_tools()));
    body["tool_choice"] = json!("none");
    let (status, text) = post_sse(&app, body).await;
    assert_eq!(status, StatusCode::OK, "text = {text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");
    let chunks = sse_chunks(&text);
    assert!(chunks
        .iter()
        .all(|c| c["choices"][0]["delta"].get("tool_calls").is_none()));
    let last = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い");
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn streaming_two_tool_calls_framing_invariants() {
    // 2 連続 block: index 0 → 1、id は start の 1 回だけ、arguments は増分連結。
    let mut script = block_script("get_weather", &["Tokyo"], &|p| tid(p));
    script.pop(); // eos
    script.extend(block_script("run_tool", &["Tokyo"], &|p| tid(p)));
    let app = script_app(script, tool_tokenizer().unwrap());
    let mut body = chat_body(true, Some(three_tools()));
    body["tool_choice"] = json!("auto");
    let (status, text) = post_sse(&app, body).await;
    assert_eq!(status, StatusCode::OK, "text = {text}");
    assert!(!text.contains("event: "), "event: 行が混ざっている:\n{text}");
    assert!(text.trim_end().ends_with("data: [DONE]"), "text = {text}");

    let chunks = sse_chunks(&text);
    assert!(chunks.len() >= 6, "chunks = {chunks:?}");

    let id0 = chunks[0]["id"].as_str().expect("id が無い").to_string();
    assert!(id0.starts_with("chatcmpl-"), "id = {id0}");
    let created0 = chunks[0]["created"].as_u64().unwrap();
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], id0, "chunk ごとに id が違う: {c}");
        assert_eq!(c["created"].as_u64().unwrap(), created0);
        assert_eq!(c["choices"].as_array().unwrap().len(), 1);
    }
    // 先頭 chunk は role のみ
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["choices"][0]["delta"]["content"], "");

    let mut starts: Vec<(u64, String, String)> = Vec::new(); // (index, id, name)
    let mut args: std::collections::BTreeMap<u64, String> = Default::default();
    for c in &chunks {
        let Some(tcs) = c["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        assert_eq!(tcs.len(), 1, "delta.tool_calls が 1 個でない: {c}");
        let tc = &tcs[0];
        let index = tc["index"].as_u64().expect("index が無い");
        if let Some(name) = tc["function"]["name"].as_str() {
            // start chunk: id/type/name/空 arguments がそろう
            assert_eq!(tc["type"], "function");
            assert_eq!(tc["function"]["arguments"], "");
            let id = tc["id"].as_str().expect("start に id が無い").to_string();
            assert!(id.starts_with("call_"), "id = {id}");
            starts.push((index, id, name.to_string()));
        } else {
            // 継続 chunk: id/type を再送しない
            assert!(tc.get("id").is_none(), "id が 2 回載っている: {tc}");
            assert!(tc.get("type").is_none(), "type が再送されている: {tc}");
            let a = tc["function"]["arguments"]
                .as_str()
                .expect("arguments が文字列でない");
            args.entry(index).or_default().push_str(a);
        }
    }
    assert_eq!(starts.len(), 2, "starts = {starts:?}");
    assert_eq!(starts[0].0, 0);
    assert_eq!(starts[0].2, "get_weather");
    assert_eq!(starts[1].0, 1);
    assert_eq!(starts[1].2, "run_tool");
    assert_ne!(starts[0].1, starts[1].1, "tool call id が重複");
    for (index, raw) in &args {
        let v: Value = serde_json::from_str(raw).expect("arguments 増分の連結が JSON でない");
        assert_eq!(v, json!({"city": "Tokyo"}), "index={index}: {raw}");
    }
    assert_eq!(args.len(), 2, "args = {args:?}");

    // finish_reason は最後の choices 付き chunk にだけ tool_calls
    let last = chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(last["choices"][0]["delta"], json!({}));
    // include_usage 未指定なので usage chunk は出ない
    assert!(chunks.iter().all(|c| c.get("usage").is_none()));
}

// ---------------------------------------------------------------------------
// 4. Unicode / 日本語 / エスケープされた引用符
// ---------------------------------------------------------------------------

#[tokio::test]
async fn japanese_in_tool_arguments_is_preserved() {
    // TOOL_VOCAB の 天気 (id 31) を引数値に使う。
    let app = script_app(
        block_script("get_weather", &["天気"], &|p| tid(p)),
        tool_tokenizer().unwrap(),
    );
    let (status, v) = post(&app, chat_body(false, Some(three_tools()))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let args = v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    // serde_json は非 ASCII をそのまま出す (エスケープしない)
    assert!(args.contains("天気"), "arguments = {args}");
    let parsed: Value = serde_json::from_str(args).unwrap();
    assert_eq!(parsed, json!({"city": "天気"}));
}

#[tokio::test]
async fn escaped_quotes_in_tool_arguments_round_trip() {
    // 追加語彙 (id 40..): `"` `\` `hello` ` ` `world`
    const EXTRA: [&str; 5] = ["\"", "\\", "hello", " ", "world"];
    let extra_tid = |p: &str| -> u32 {
        EXTRA
            .iter()
            .position(|e| *e == p)
            .map(|i| 40 + i as u32)
            .unwrap_or_else(|| panic!("extra vocab missing: {p:?}"))
    };
    // 引数値 = `"hello \"world\""` (JSON 文字列として復元されると hello "world")
    let value = ["\"", "hello", " ", "\\", "\"", "world", "\\", "\"", "\""];
    let script = block_script("get_weather", &value, &|p| extra_tid(p));
    let app = script_app(script, tool_tokenizer_with(&EXTRA).unwrap());
    let (status, v) = post(&app, chat_body(false, Some(three_tools()))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let args = v["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        .as_str()
        .unwrap();
    // ワイヤ上は引用符が JSON エスケープされている
    assert!(args.contains("hello \\\"world\\\""), "arguments = {args}");
    let parsed: Value = serde_json::from_str(args).unwrap();
    assert_eq!(parsed, json!({"city": "hello \"world\""}));
}

#[test]
fn japanese_and_quotes_in_schema_are_serialized_deterministically() {
    let spec = JsonSchemaSpec::parse(&json!({
        "name": "天気スキーマ",
        "description": "say \"hi\" in 日本語",
        "strict": false,
        "schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "enum": ["晴れ", "雨", "曇り"]},
                "note": {"type": "string", "description": "引用符 \"x\" を含む"}
            },
            "required": ["summary", "note"]
        }
    }))
    .expect("日本語スキーマが拒否された");
    let h = spec.hint();
    assert!(h.contains("天気スキーマ"), "{h}");
    // description は指示文へ raw で入る (スキーマ本文の JSON だけがエスケープされる)
    assert!(h.contains("say \"hi\" in 日本語"), "{h}");
    assert!(h.contains("晴れ"), "{h}");
    assert!(h.contains("引用符 \\\"x\\\" を含む"), "{h}");
    assert_eq!(h, spec.hint(), "hint が非決定的");
}

#[tokio::test]
async fn japanese_schema_request_succeeds_on_wire() {
    let app = wide_app();
    let schema = json!({
        "type": "object",
        "properties": {
            "summary": {"type": "string", "enum": ["晴れ", "雨"]},
            "detail": {"type": "string", "description": "自由記述 (日本語可)"}
        },
        "required": ["summary", "detail"],
        "additionalProperties": false
    });
    let (status, v) = post(&app, json_schema_body(false, true, schema)).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert!(!v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());
}
