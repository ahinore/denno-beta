//! Task 999: JSON Schema / Structured Output のエッジケース。
//!
//! `openai_json_schema.rs` (正常系) と `schema_tools_edge_964.rs` (境界の一部) を
//! 補い、次の 4 群だけを独立に固定する:
//!
//! 1. 入れ子 object / enum / array / min-max 制約 / required /
//!    additionalProperties の受理と、strict モードでの再帰的な required 検査。
//! 2. 深いネスト・自己参照 ($ref)・型違いのスキーマ入力が **panic せず**
//!    パス付きのエラー文字列を返すこと (深さ上限の境界を含む)。
//! 3. サーバールータ経由の `response_format` (`json_object` / `json_schema`) の
//!    形のエッジ: 余分キー、null、配列、boolean スキーマ、name 欠落。
//! 4. 同じスキーマから生成される指示 (grammar) が反復実行で決定的であること。
//!
//! 生成は denno-engine の決定的ダミーモデル + ダミートークナイザのみを使う
//! (モデル DL / GPU 不要)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::json_schema::{JsonSchemaSpec, ResponseFormat, MAX_SCHEMA_DEPTH};
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// fixture
// ---------------------------------------------------------------------------

/// スキーマ指示ぶん長くなるプロンプトを受けられるようにコンテキストを広げた
/// 決定的ダミーモデル。peak = 直前トークン + 1。最初の forward で受け取った
/// プロンプトのトークン列を 1 回だけ記録する (指示がプロンプトに入った証拠)。
struct CaptureModel {
    seen: Arc<Mutex<Vec<Vec<u32>>>>,
}

impl Model for CaptureModel {
    fn context(&self) -> usize {
        4096
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
        KvCache::new(1, 4, 4096)
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

fn chat_body() -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 12,
        "temperature": 0.0,
    })
}

fn schema_body(strict: bool, schema: Value) -> Value {
    let mut b = chat_body();
    b["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "edge_schema",
            "strict": strict,
            "schema": schema,
        },
    });
    b
}

fn spec(schema: Value, strict: bool) -> JsonSchemaSpec {
    JsonSchemaSpec {
        name: "edge_schema".to_string(),
        description: None,
        schema,
        strict,
    }
}

/// 深さ `levels` の配列スキーマ。最深の葉は深さ `levels` に置かれる。
fn nested_arrays(levels: usize) -> Value {
    let mut s = json!({"type": "string"});
    for _ in 0..levels {
        s = json!({"type": "array", "items": s});
    }
    s
}

// ---------------------------------------------------------------------------
// 1. 入れ子 / enum / array / min-max / required / additionalProperties
// ---------------------------------------------------------------------------

/// 実運用に近い複合スキーマ (入れ子 object + enum + array + 数値/長さ境界)。
/// 検証器は制約の中身までは解釈しないが、**受理して指示に丸ごと載せる**ことが
/// 契約 (捨てない・書き換えない)。strict でも enum / minimum / maxItems は
/// 非対応キーワードではないので通る。
#[test]
fn nested_object_enum_array_bounds_accepted_and_kept_in_hint() {
    let schema = json!({
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "degraded", "failed"]},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "tags": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 32},
                "minItems": 1,
                "maxItems": 8,
            },
            "details": {
                "type": "object",
                "properties": {
                    "retries": {"type": "integer", "minimum": 0, "maximum": 5},
                    "codes": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["retries", "codes"],
                "additionalProperties": false,
            },
        },
        "required": ["status", "score", "tags", "details"],
        "additionalProperties": false,
    });

    for strict in [true, false] {
        spec(schema.clone(), strict)
            .validate()
            .unwrap_or_else(|e| panic!("strict={strict} で受理されるべき: {e}"));
    }

    // 指示 (grammar) には enum の値と境界がそのまま載る。enum は配列なので
    // 値の順序はスキーマに書いた順のまま (ソートされない) — 同じ入力なら同じ並び。
    let h = spec(schema.clone(), true).hint();
    for needle in [
        "\"enum\":[\"ok\",\"degraded\",\"failed\"]",
        "\"maximum\":1.0",
        "\"minimum\":0.0",
        "\"maxItems\":8",
        "\"additionalProperties\":false",
    ] {
        assert!(h.contains(needle), "hint に {needle} が無い: {h}");
    }
}

/// strict の required 全列挙検査は **入れ子の object にも再帰的に** 効き、
/// エラーのパスが該当ノードを指す。
#[test]
fn strict_required_check_recurses_with_node_path() {
    let schema = json!({
        "type": "object",
        "properties": {
            "outer": {"type": "string"},
            "inner": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "string"},
                },
                "required": ["a"],
                "additionalProperties": false,
            },
        },
        "required": ["outer", "inner"],
        "additionalProperties": false,
    });

    // 非 strict は通る (required の不足はモデル任せ)。
    spec(schema.clone(), false).validate().unwrap();

    let e = spec(schema, true).validate().unwrap_err();
    assert!(e.contains("$.properties.inner"), "パスが無い: {e}");
    assert!(e.contains("required"), "{e}");
    assert!(e.contains("\"b\""), "{e}");
}

/// `additionalProperties` がスキーマ (object) のときも辿られ、不正な型は
/// パス付きで弾く。真偽値はそのまま受理。配列などは 400 相当のエラー。
#[test]
fn additional_properties_schema_object_is_traversed() {
    // object 形式 → 内部を検査 (未対応 type を検出)
    let s = json!({
        "type": "object",
        "properties": {},
        "additionalProperties": {"type": "tuple"},
    });
    let e = spec(s, false).validate().unwrap_err();
    assert!(e.contains("$.additionalProperties"), "{e}");
    assert!(e.contains("unsupported type"), "{e}");

    // boolean 形式は葉として受理
    for b in [true, false] {
        let s = json!({"type": "object", "additionalProperties": b});
        spec(s, false).validate().unwrap();
    }

    // 配列 / 文字列は型エラー
    for bad in [json!([]), json!("no"), json!(0)] {
        let s = json!({"type": "object", "additionalProperties": bad});
        let e = spec(s.clone(), false).validate().unwrap_err();
        assert!(
            e.contains("additionalProperties must be a JSON object or a boolean"),
            "{e}"
        );
    }
}

/// `properties` を持たない object は strict でも additionalProperties /
/// required を要求されない (現行の厳格モードの境界を固定する)。
#[test]
fn strict_object_without_properties_needs_no_required() {
    let s = json!({"type": "object", "additionalProperties": true});
    spec(s.clone(), true).validate().unwrap();

    // `type` を省いたノードも同様に required 検査の対象外 (properties だけでは
    // object と判定しない)。既存挙動の固定。
    let s = json!({"properties": {"a": {"type": "string"}}});
    spec(s, true).validate().unwrap();
}

// ---------------------------------------------------------------------------
// 2. 深いネスト / 自己参照 / 型違い → panic しないきれいなエラー
// ---------------------------------------------------------------------------

#[test]
fn depth_boundary_is_exact() {
    // 深さ 0 から数えて MAX_SCHEMA_DEPTH までは受理、+1 で拒否。
    spec(nested_arrays(MAX_SCHEMA_DEPTH), false).validate().unwrap();
    let e = spec(nested_arrays(MAX_SCHEMA_DEPTH + 1), false)
        .validate()
        .unwrap_err();
    assert!(e.contains("nested too deeply"), "{e}");
    // パスは $.items の連鎖になる (先頭がルート)
    assert!(e.contains("$.items"), "{e}");
}

/// 自己参照の $ref は解決機構が無いので、strict/非 strict とも 400 相当。
/// (Value は循環できないので「循環」は $ref の再帰で表現する)
#[test]
fn self_referential_schema_is_rejected_not_hung() {
    let cyclic = json!({
        "type": "object",
        "properties": {
            "child": {"$ref": "#"},
            "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
        },
        "required": ["child", "children"],
        "additionalProperties": false,
    });
    for strict in [true, false] {
        let e = spec(cyclic.clone(), strict).validate().unwrap_err();
        assert!(e.contains("$ref is not supported"), "strict={strict}: {e}");
    }
}

/// 深すぎる (上限のはるか上) スキーマでも再帰が止まり、panic せずエラー。
/// 途中で切れるので構築コストも線形。
#[test]
fn very_deep_schema_errors_without_stack_overflow() {
    let deep = nested_arrays(MAX_SCHEMA_DEPTH * 8);
    let e = spec(deep, false).validate().unwrap_err();
    assert!(e.contains("nested too deeply"), "{e}");

    // properties 側の連鎖でも同じく止まる。
    let mut node = json!({"type": "string"});
    for _ in 0..(MAX_SCHEMA_DEPTH * 4) {
        node = json!({"type": "object", "properties": {"n": node}});
    }
    let e = spec(node, false).validate().unwrap_err();
    assert!(e.contains("nested too deeply"), "{e}");
}

/// 構文的に「JSON Schema ではない」入力はすべて型エラーで返る (panic 無し)。
#[test]
fn non_schema_shapes_are_type_errors() {
    for bad in [json!(1), json!("x"), json!([]), json!([true])] {
        let e = spec(bad.clone(), false).validate().unwrap_err();
        assert!(e.contains("must be a JSON object or a boolean"), "{bad}: {e}");
        // strict でも同じ (strict 検査より前に弾く)
        assert!(spec(bad, true).validate().is_err());
    }

    // `type` が文字列でない / `properties` が object でない / `items` が tuple
    for s in [
        json!({"type": 1}),
        json!({"type": ["string", "null"]}),
        json!({"properties": []}),
        json!({"items": [{"type": "string"}]}),
    ] {
        assert!(spec(s.clone(), false).validate().is_err(), "{s}");
    }
}

// ---------------------------------------------------------------------------
// 3. ルータ経由の response_format エッジ
// ---------------------------------------------------------------------------

/// `json_object` は追加キーを無視して受理し、指示をプロンプトへ入れる。
/// `response_format: null` は未指定と等価 (プロンプト不変)。
#[tokio::test]
async fn json_object_and_null_response_format_via_router() {
    let (app, seen_obj) = app_with_capture();
    let mut body = chat_body();
    body["response_format"] = json!({"type": "json_object", "unexpected": {"k": 1}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let obj_prompt = seen_obj.lock().unwrap()[0].clone();

    let (app2, seen_plain) = app_with_capture();
    let (status, _) = post(&app2, chat_body()).await;
    assert_eq!(status, StatusCode::OK);
    let plain_prompt = seen_plain.lock().unwrap()[0].clone();
    assert!(
        obj_prompt.len() > plain_prompt.len(),
        "json_object の指示がプロンプトに入っていない: {} vs {}",
        obj_prompt.len(),
        plain_prompt.len()
    );

    // null は未指定と同じ
    let (app3, seen_null) = app_with_capture();
    let mut body = chat_body();
    body["response_format"] = Value::Null;
    let (status, v) = post(&app3, body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(seen_null.lock().unwrap()[0], plain_prompt);
    // null の内容も未指定と同じ
    let (status, v2) = post(&app2, chat_body()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        v["choices"][0]["message"]["content"],
        v2["choices"][0]["message"]["content"]
    );
}

/// `response_format` の型違い / 形違いは param=response_format.type の 400。
#[tokio::test]
async fn malformed_response_format_shapes_are_400() {
    let app = app_with_capture().0;
    let cases: Vec<(&str, Value)> = vec![
        ("配列", json!([{"type": "json_object"}])),
        ("文字列", json!("json_object")),
        ("数値", json!(1)),
        ("type が null", json!({"type": null})),
        ("type が数値", json!({"type": 7})),
        ("空オブジェクト", json!({})),
        ("未知の type", json!({"type": "yaml"})),
    ];
    for (label, fmt) in cases {
        let mut body = chat_body();
        body["response_format"] = fmt.clone();
        let (status, v) = post(&app, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{label}: {v}");
        assert_eq!(
            v["error"]["type"], "invalid_request_error",
            "{label}: {v}"
        );
        assert_eq!(v["error"]["param"], "response_format.type", "{label}: {v}");
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(!msg.is_empty(), "{label}: message が空");
    }
}

/// json_schema 側の形のエッジ: boolean スキーマは受理、name 欠落/空 と
/// strict 非 bool は param=response_format.json_schema の 400。
#[tokio::test]
async fn json_schema_shape_edges_via_router() {
    let app = app_with_capture().0;

    // boolean スキーマは両モードで受理される (JSON Schema の真偽値形式)
    for strict in [true, false] {
        let (status, v) = post(&app, schema_body(strict, json!(true))).await;
        assert_eq!(status, StatusCode::OK, "strict={strict}: {v}");
    }

    // name 欠落
    let mut body = chat_body();
    body["response_format"] =
        json!({"type": "json_schema", "json_schema": {"schema": {"type": "object"}}});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "response_format.json_schema");
    assert!(v["error"]["message"].as_str().unwrap().contains("name"));

    // name が空白のみ
    let mut body = chat_body();
    body["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {"name": "   ", "schema": {"type": "object"}},
    });
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("non-empty"));

    // json_schema が object でない
    let mut body = chat_body();
    body["response_format"] = json!({"type": "json_schema", "json_schema": []});
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["param"], "response_format.json_schema");

    // strict が bool でない
    let mut body = chat_body();
    body["response_format"] = json!({
        "type": "json_schema",
        "json_schema": {"name": "n", "strict": 1, "schema": {"type": "object"}},
    });
    let (status, v) = post(&app, body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"]["message"].as_str().unwrap().contains("boolean"));
}

/// 深すぎるスキーマをルータへ投げても 400 で返り、本文が壊れない
/// (エラーの `message` にパスが入り、JSON として読める)。
#[tokio::test]
async fn too_deep_schema_via_router_is_clean_400() {
    let app = app_with_capture().0;
    let (status, v) = post(&app, schema_body(false, nested_arrays(MAX_SCHEMA_DEPTH + 3))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("nested too deeply"), "message = {msg}");
    assert!(msg.contains("$.items"), "message = {msg}");
    assert!(!msg.contains("panicked"), "panic の痕跡: {msg}");
}

/// `parse_response_format` の返り値分類 (Text / JsonObject / JsonSchema) を
/// 直接固定する (ルータの param 振り分けの前提)。
#[test]
fn response_format_kinds_are_classified() {
    let cases = [
        (json!({"type": "text"}), "text"),
        (json!({"type": "json_object"}), "json_object"),
        (
            json!({"type": "json_schema", "json_schema": {"name": "n",
                "schema": {"type": "object", "additionalProperties": true}}}),
            "json_schema",
        ),
    ];
    for (v, want) in cases {
        let got = denno_server::json_schema::parse_response_format(&v).unwrap();
        match (want, got) {
            ("text", ResponseFormat::Text) => {}
            ("json_object", ResponseFormat::JsonObject) => {}
            ("json_schema", ResponseFormat::JsonSchema(s)) => assert_eq!(s.name, "n"),
            (w, _) => panic!("{w} にならない"),
        }
    }
}

// ---------------------------------------------------------------------------
// 4. 決定的な grammar / 制約生成 (反復実行)
// ---------------------------------------------------------------------------

/// 同じスキーマからは、何度呼んでも byte 単位で同一の指示が生成される。
/// キーの挿入順が違っても同一 (serde_json の BTreeMap 正規化)。
#[test]
fn hint_is_byte_identical_across_repeated_runs() {
    let schema = json!({
        "type": "object",
        "properties": {
            "b": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
            "a": {"type": "string", "enum": ["x", "y"]},
        },
        "required": ["a", "b"],
        "additionalProperties": false,
    });
    let first = spec(schema.clone(), true).hint();
    for i in 0..32 {
        assert_eq!(spec(schema.clone(), true).hint(), first, "反復 {i} で揺れた");
    }

    // キー順を逆に組み立てた等価スキーマでも同一
    let mut m = serde_json::Map::new();
    m.insert("additionalProperties".into(), json!(false));
    m.insert("required".into(), json!(["a", "b"]));
    m.insert(
        "properties".into(),
        json!({
            "a": {"type": "string", "enum": ["x", "y"]},
            "b": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
        }),
    );
    m.insert("type".into(), json!("object"));
    assert_eq!(spec(Value::Object(m), true).hint(), first);

    // strict と非 strict はモード表記だけが変わる (スキーマ本文は不変)
    let loose = spec(schema, false).hint();
    assert!(loose.contains("best-effort"), "{loose}");
    assert!(first.contains("strict"), "{first}");
    assert_eq!(
        loose.replace("best-effort", "strict"),
        first,
        "モード表記以外が揺れている"
    );
}

/// 同じ不正スキーマは毎回同じエラー文字列を返す (診断の再現性)。
#[test]
fn validation_errors_are_deterministic() {
    let bad = json!({
        "type": "object",
        "properties": {"a": {"type": "array", "items": {"type": "tuple"}}},
        "required": ["a"],
        "additionalProperties": false,
    });
    let first = spec(bad.clone(), false).validate().unwrap_err();
    for _ in 0..16 {
        assert_eq!(spec(bad.clone(), false).validate().unwrap_err(), first);
    }
}

/// HTTP 経由の反復も決定的: 同じ複合スキーマを 3 回投げると、モデルが受け取る
/// プロンプトのトークン列と応答本文が一致する。
#[tokio::test]
async fn repeated_http_requests_produce_identical_prompt_and_output() {
    let schema = json!({
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok", "failed"]},
            "items": {"type": "array", "items": {"type": "integer"}, "maxItems": 4},
        },
        "required": ["status", "items"],
        "additionalProperties": false,
    });

    let (app1, seen1) = app_with_capture();
    let (status, v1) = post(&app1, schema_body(true, schema.clone())).await;
    assert_eq!(status, StatusCode::OK, "body = {v1}");
    let prompt1 = seen1.lock().unwrap()[0].clone();
    let content1 = v1["choices"][0]["message"]["content"].clone();

    for run in 0..2 {
        let (app, seen) = app_with_capture();
        let (status, v) = post(&app, schema_body(true, schema.clone())).await;
        assert_eq!(status, StatusCode::OK, "run {run}: {v}");
        assert_eq!(
            seen.lock().unwrap()[0],
            prompt1,
            "run {run}: プロンプトが揺れた"
        );
        assert_eq!(
            v["choices"][0]["message"]["content"],
            content1,
            "run {run}: 応答が揺れた"
        );
    }
}
