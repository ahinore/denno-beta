//! task 1086: `GET /v1/models` と `GET /v1/models/:id` のメタデータ一致。
//!
//! 両エンドポイントは `openai::model_object` を共有しているので、ここでは
//! 「共有が将来も保たれること」を外側から固定する:
//!
//! - 一覧の `data[0]` と単体取得の本文が **JSON として完全一致** する
//!   (フィールドの追加・削除・値のずれを 1 本で捕まえる)。
//! - `context_window` / `context_length` / `max_context_length` の 3 名が
//!   同じ値で、0 より大きい。
//! - URL エンコードした id (`%2F` / `%2f` / ハイフン・ピリオド) でも同じ本文。
//! - 未知の id は OpenAI 形式の 404 封筒 (`code` は null、内部情報を漏らさない)。
//! - 認証は一覧と単体で同じ (キー設定時に両方 401)。
//! - `docs/openapi.yaml` の `Model` / `ModelList` スキーマが、実行時の
//!   フィールドを漏れなく記載している (仕様のドリフト検出)。
//!
//! `DENNO_MODEL_ALIASES` などの環境変数は触らない (プロセス全体に効くため)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

const CANONICAL: &str = "denno-test";
const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const MODEL_DIR: &str = "qwen-test-dir";

/// `docs/openapi.yaml` (テストファイルからの相対)。
const SPEC: &str = include_str!("../../../docs/openapi.yaml");

fn app_with_key(api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, CANONICAL, None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = MODEL_DIR.to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: CANONICAL.to_string(),
        model_id: MODEL_ID.to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

fn app() -> Router {
    app_with_key(None)
}

struct Resp {
    status: StatusCode,
    content_type: Option<String>,
    body: Value,
}

async fn get(app: &Router, uri: &str) -> Resp {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    Resp { status, content_type, body }
}

/// 一覧の `data[0]`。
fn list_entry(app: &Router) -> impl std::future::Future<Output = Value> {
    let app = app.clone();
    async move {
        let resp = get(&app, "/v1/models").await;
        assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.body);
        assert_eq!(resp.body["object"], "list", "body = {}", resp.body);
        assert_eq!(resp.body["has_more"], false, "body = {}", resp.body);
        let data = resp.body["data"]
            .as_array()
            .unwrap_or_else(|| panic!("data が配列でない: {}", resp.body));
        assert_eq!(data.len(), 1, "data = {}", resp.body["data"]);
        data[0].clone()
    }
}

fn object_keys(v: &Value) -> Vec<String> {
    let mut keys: Vec<String> = v
        .as_object()
        .expect("model object がオブジェクトでない")
        .keys()
        .cloned()
        .collect();
    keys.sort_unstable();
    keys
}

// ---------------------------------------------------------------------------
// 本文の一致
// ---------------------------------------------------------------------------

#[tokio::test]
async fn list_entry_equals_by_id_body_field_for_field() {
    let app = app();
    let entry = list_entry(&app).await;
    let resp = get(&app, &format!("/v1/models/{CANONICAL}")).await;
    assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.body);
    assert_eq!(
        resp.body, entry,
        "一覧の data[0] と単体取得の本文が一致しない\nlist = {entry}\nby_id = {}",
        resp.body
    );
}

#[tokio::test]
async fn model_object_carries_the_documented_operational_fields() {
    let app = app();
    let entry = list_entry(&app).await;
    // 運用メタデータ (OpenAI 互換 + Anthropic 互換 + denno 固有) を固定する。
    assert_eq!(entry["id"], CANONICAL);
    assert_eq!(entry["object"], "model");
    assert_eq!(entry["type"], "model");
    assert_eq!(entry["display_name"], CANONICAL);
    assert_eq!(entry["owned_by"], "denno");
    assert_eq!(entry["model_id"], MODEL_ID);
    assert_eq!(entry["model_dir"], MODEL_DIR);
    assert!(
        entry["created"].as_u64().unwrap_or(0) > 1_577_836_800,
        "created が実時刻でない: {entry}"
    );
    for key in ["quant", "slots", "layers"] {
        assert!(!entry[key].is_null(), "{key} が無い: {entry}");
    }
    assert!(
        entry["aliases"].is_array(),
        "aliases が配列でない: {entry}"
    );
    for cap in ["openai", "anthropic", "ollama", "tools", "streaming", "count_tokens"] {
        assert_eq!(entry["capabilities"][cap], true, "capabilities.{cap}: {entry}");
    }
    assert!(
        entry["capabilities"]["telemetry"].is_boolean(),
        "capabilities.telemetry が bool でない: {entry}"
    );
    // キー集合そのものも固定する (増減は仕様とテストの追随漏れなので落とす)。
    assert_eq!(
        object_keys(&entry),
        [
            "aliases",
            "capabilities",
            "context_length",
            "context_window",
            "created",
            "display_name",
            "id",
            "layers",
            "max_context_length",
            "model_dir",
            "model_id",
            "object",
            "owned_by",
            "quant",
            "slots",
            "type",
        ],
        "model object のキー集合が変わった: {entry}"
    );
}

#[tokio::test]
async fn context_fields_agree_on_both_endpoints() {
    let app = app();
    let entry = list_entry(&app).await;
    let by_id = get(&app, &format!("/v1/models/{CANONICAL}")).await.body;
    for (label, m) in [("list", &entry), ("by_id", &by_id)] {
        let context_window = m["context_window"].as_u64().unwrap_or(0);
        assert!(context_window > 0, "{label}: context_window = {context_window}");
        assert_eq!(m["context_length"].as_u64(), Some(context_window), "{label}: {m}");
        assert_eq!(
            m["max_context_length"].as_u64(),
            Some(context_window),
            "{label}: {m}"
        );
    }
    assert_eq!(entry["context_length"], by_id["context_length"]);
}

// ---------------------------------------------------------------------------
// URL エンコード
// ---------------------------------------------------------------------------

#[tokio::test]
async fn url_encoded_ids_return_the_same_body_as_the_listing() {
    let app = app();
    let entry = list_entry(&app).await;
    // `model_id` は `/` を含む -> `%2F` (大文字) / `%2f` (小文字)。
    // ハイフン (`%2D`) とピリオド (`%2E`) も同じ別名に解決する。
    for id in [
        "Qwen%2FQwen2.5-0.5B-Instruct",
        "Qwen%2fQwen2.5-0.5B-Instruct",
        "Qwen2.5-0.5B-Instruct",
        "Qwen2%2E5%2D0%2E5B%2DInstruct",
        MODEL_DIR,
    ] {
        let resp = get(&app, &format!("/v1/models/{id}")).await;
        assert_eq!(resp.status, StatusCode::OK, "id = {id}, body = {}", resp.body);
        assert_eq!(resp.body, entry, "id = {id} の本文が一覧と一致しない");
        // 応答の id は常に canonical。
        assert_eq!(resp.body["id"], CANONICAL, "id = {id}");
    }
}

// ---------------------------------------------------------------------------
// 404 封筒
// ---------------------------------------------------------------------------

fn assert_model_not_found(resp: &Resp, id_in_message: &str) {
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.body);
    assert!(
        resp.content_type.as_deref().unwrap_or("").starts_with("application/json"),
        "content-type = {:?}",
        resp.content_type
    );
    let err = &resp.body["error"];
    let mut keys: Vec<String> = err
        .as_object()
        .unwrap_or_else(|| panic!("error がオブジェクトでない: {}", resp.body))
        .keys()
        .cloned()
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["code", "message", "param", "type"], "error = {err}");
    assert_eq!(err["type"], "invalid_request_error");
    assert_eq!(err["code"], Value::Null, "error = {err}");
    assert_eq!(err["param"], Value::Null, "error = {err}");
    let msg = err["message"].as_str().unwrap_or("");
    assert!(msg.contains(id_in_message), "message = {msg}");
    // 内部情報 (ソースパス・panic 文言) を漏らさない。
    for leak in [".rs:", "panicked", "src/"] {
        assert!(!msg.contains(leak), "message に内部情報: {msg}");
    }
}

#[tokio::test]
async fn unknown_ids_use_the_same_404_envelope() {
    let app = app();
    // 平文 / パーセントエンコード / 二重エンコード。メッセージには復号後の
    // 文字列が載る (エンコード形そのものは載らない)。
    let cases = [
        ("nope", "nope"),
        ("Qwen%2Fnope", "Qwen/nope"),
        ("Qwen%252FQwen2.5-0.5B-Instruct", "Qwen%2FQwen2.5-0.5B-Instruct"),
    ];
    for (id, shown) in cases {
        let resp = get(&app, &format!("/v1/models/{id}")).await;
        assert_model_not_found(&resp, shown);
        // 404 本文に model object が混ざっていないこと。
        assert!(resp.body.get("id").is_none(), "body = {}", resp.body);
        assert!(resp.body.get("object").is_none(), "body = {}", resp.body);
    }
}

#[tokio::test]
async fn canonical_lookup_is_case_sensitive() {
    let app = app();
    let resp = get(&app, "/v1/models/DENNO-TEST").await;
    assert_model_not_found(&resp, "DENNO-TEST");
}

// ---------------------------------------------------------------------------
// 認証の一致
// ---------------------------------------------------------------------------

#[tokio::test]
async fn both_endpoints_require_the_same_auth() {
    let app = app_with_key(Some("secret"));
    let list = get(&app, "/v1/models").await;
    let by_id = get(&app, &format!("/v1/models/{CANONICAL}")).await;
    assert_eq!(list.status, StatusCode::UNAUTHORIZED, "body = {}", list.body);
    assert_eq!(by_id.status, StatusCode::UNAUTHORIZED, "body = {}", by_id.body);
    assert_eq!(list.body["error"]["type"], by_id.body["error"]["type"]);
    assert_eq!(list.body["error"]["type"], "authentication_error");
}

// ---------------------------------------------------------------------------
// 仕様書のドリフト検出
// ---------------------------------------------------------------------------

/// `docs/openapi.yaml` の `    NAME:` ブロックを (indent, 名前) で返す。
/// 4 スペース字下げの次のキーで終わる (components/schemas 直下の名前だけ)。
fn schema_block(name: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut inside = false;
    for line in SPEC.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim_start();
        if !inside {
            if indent == 4 && t == format!("{name}:") {
                inside = true;
            }
            continue;
        }
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if indent <= 4 {
            break;
        }
        let key = t.trim_end_matches(':');
        // 値を持つ行 (`type: object` など) と見出し行の区別は呼び出し側で行う。
        out.push((indent, key.to_string()));
    }
    assert!(!out.is_empty(), "仕様にスキーマ {name} が無い");
    out
}

/// スキーマ直下の `properties` にあるキー (8 スペース)。
fn schema_properties(name: &str) -> Vec<String> {
    // 見出し行だけを拾うと `type: object` の `type` も入るため、
    // 「`properties:` の後、次の同レベル見出しまで」の 8 スペース行に限る。
    let block = schema_block(name);
    let mut out = Vec::new();
    let mut in_props = false;
    for (indent, key) in block {
        if indent == 6 {
            in_props = key == "properties";
            continue;
        }
        if in_props && indent == 8 {
            out.push(key);
        }
    }
    out
}

/// `capabilities` の下のキー (12 スペース)。
fn capability_keys(name: &str) -> Vec<String> {
    let block = schema_block(name);
    let mut out = Vec::new();
    let mut in_caps = false;
    let mut in_caps_props = false;
    for (indent, key) in block {
        if indent == 8 {
            in_caps = key == "capabilities";
            in_caps_props = false;
            continue;
        }
        if !in_caps {
            continue;
        }
        if indent == 10 {
            in_caps_props = key == "properties";
            continue;
        }
        if in_caps_props && indent == 12 {
            out.push(key);
        }
    }
    out
}

/// 指定スキーマのブロック内の `required: [...]` (6 スペース)。
fn schema_required(name: &str) -> Vec<String> {
    let mut inside = false;
    for line in SPEC.lines() {
        let indent = line.len() - line.trim_start().len();
        let t = line.trim_start();
        if !inside {
            if indent == 4 && t == format!("{name}:") {
                inside = true;
            }
            continue;
        }
        if !t.is_empty() && indent <= 4 {
            break;
        }
        if indent == 6 {
            if let Some(rest) = t.strip_prefix("required:") {
                let inner = rest.trim().trim_start_matches('[').trim_end_matches(']');
                return inner
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        }
    }
    panic!("仕様のスキーマ {name} に required が無い");
}

#[tokio::test]
async fn spec_model_schema_documents_every_runtime_field() {
    let entry = list_entry(&app()).await;

    let documented = schema_properties("Model");
    for key in object_keys(&entry) {
        assert!(
            documented.iter().any(|d| *d == key),
            "実行時のフィールド {key} が docs/openapi.yaml の Model に無い (記載: {documented:?})"
        );
    }
    // 必須項目は実行時に必ず存在する。
    let required = schema_required("Model");
    assert!(!required.is_empty(), "Model.required が空");
    for key in &required {
        assert!(
            entry.get(key).is_some(),
            "仕様が必須としている {key} が本文に無い: {entry}"
        );
    }
    // capabilities の記載漏れも見る。
    let caps = capability_keys("Model");
    let runtime_caps: Vec<String> = entry["capabilities"]
        .as_object()
        .expect("capabilities がオブジェクトでない")
        .keys()
        .cloned()
        .collect();
    for key in &runtime_caps {
        assert!(
            caps.iter().any(|d| d == key),
            "capabilities.{key} が docs/openapi.yaml に無い (記載: {caps:?})"
        );
    }
}

#[test]
fn spec_list_envelope_documents_has_more_and_data() {
    let documented = schema_properties("ModelList");
    for key in ["object", "data", "has_more"] {
        assert!(
            documented.iter().any(|d| d == key),
            "ModelList に {key} が無い (記載: {documented:?})"
        );
    }
    // `has_more` は一覧の封筒のフィールドで、モデル本体のフィールドではない。
    assert!(
        !schema_properties("Model").iter().any(|d| d == "has_more"),
        "Model に has_more が残っている"
    );
}

#[test]
fn spec_parsers_see_the_expected_shapes() {
    // パーサが無言で 0 件を返していないことの固定 (ドリフト検出の前提)。
    let model = schema_properties("Model");
    assert!(model.len() >= 10, "Model の properties を {} 件しか読めない", model.len());
    assert!(model.iter().any(|d| d == "id"), "Model に id が無い");
    assert!(capability_keys("Model").len() >= 6, "capabilities を読めていない");
    assert!(!schema_required("Model").is_empty(), "Model.required を読めていない");
}
