//! task 1449: `/v1/models` と `/v1/models/:id` の境界・エラー整合テスト。
//!
//! `openai_models_901.rs` (骨格) と `models_alias_edge_1214.rs` (別名解決) の
//! 上に、次の 5 点を **ワイヤ上の完全な形** で固定する:
//!
//! 1. 一覧 (`object=list`) と単体 (`object=model`) の JSON スキーマ。
//!    トップレベル / model object / capabilities のキー集合を exact に固定し、
//!    単体取得の本文が一覧 `data[0]` と **完全一致** することを確かめる。
//! 2. 未知モデルの 404 封筒 (4 フィールド、`code=null`) と、ルータ由来の
//!    404 (`code="unknown_route"`) の区別。
//! 3. 認証: 一覧は Anthropic 形式 (`type=error` + `error{type,message}`)、
//!    単体は OpenAI 形式 (4 フィールド)。`x-api-key` / `Authorization: Bearer`
//!    の両方を受ける。認証はハンドラ内なので、404 / 405 より後 (経路が先)。
//! 4. メソッド: GET のみ (HEAD は GET と同義で本文だけ落ちる)。他の
//!    メソッドは 405 の JSON 封筒。OPTIONS も CORS プリフライト以外は 405。
//! 5. ヘッダ衛生: 2xx の JSON は `charset=utf-8`、エラー封筒は charset 無し、
//!    全応答に `nosniff` と `x-request-id`。制御プレーンの応答には
//!    `cache-control` を付けない (推論パスだけ no-store)。
//!
//! 環境変数は触らない (別名は既定の model_id / 最終セグメント / model_dir)。

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

const CANONICAL: &str = "denno-test";
const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const MODEL_TAIL: &str = "Qwen2.5-0.5B-Instruct";
const MODEL_DIR: &str = "qwen-test-dir";
const SECRET: &str = "secret-key";

/// model object のキー集合 (openai::model_object が組み立てる全フィールド)。
const MODEL_KEYS: &[&str] = &[
    "id",
    "object",
    "type",
    "display_name",
    "created",
    "owned_by",
    "context_window",
    "context_length",
    "max_context_length",
    "model_id",
    "model_dir",
    "quant",
    "slots",
    "layers",
    "aliases",
    "capabilities",
];

const CAPABILITY_KEYS: &[&str] = &[
    "anthropic",
    "openai",
    "ollama",
    "tools",
    "streaming",
    "count_tokens",
    "telemetry",
];

// ---------------------------------------------------------------------------
// fixture (既存テストと同じ決定的ダミーモデル)
// ---------------------------------------------------------------------------

fn app_with_key(api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, CANONICAL, None);
    let mut info = EngineInfo::from_engine(&engine);
    info.model_dir = MODEL_DIR.to_string();
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
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

// ---------------------------------------------------------------------------
// リクエスト / 応答
// ---------------------------------------------------------------------------

struct Resp {
    status: StatusCode,
    headers: HeaderMap,
    raw: Vec<u8>,
    /// ボディが JSON として読めたときだけ Some。HEAD は空ボディなので None。
    body: Option<Value>,
}

impl Resp {
    fn ct(&self) -> String {
        hdr(&self.headers, "content-type").unwrap_or_default()
    }

    fn hdr(&self, name: &str) -> Option<String> {
        hdr(&self.headers, name)
    }

    /// assert_eq 用 (JSON でない応答も本文を見せる)。
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.raw).into_owned()
    }
}

fn hdr(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

async fn send(
    app: &Router,
    method: &str,
    uri: &str,
    headers: &[(&str, &str)],
    body: Option<Value>,
) -> Resp {
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    for (k, v) in headers {
        builder = builder.header(*k, *v);
    }
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let req = builder.body(Body::from(payload)).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let raw = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
    let body = serde_json::from_slice(&raw).ok();
    Resp { status, headers, raw, body }
}

async fn get(app: &Router, uri: &str) -> Resp {
    send(app, "GET", uri, &[], None).await
}

async fn get_auth(app: &Router, uri: &str, headers: &[(&str, &str)]) -> Resp {
    send(app, "GET", uri, headers, None).await
}

fn sorted_keys(v: &Value) -> Vec<String> {
    let mut keys: Vec<String> = v
        .as_object()
        .unwrap_or_else(|| panic!("オブジェクトでない: {v}"))
        .keys()
        .map(|k| k.to_string())
        .collect();
    keys.sort_unstable();
    keys
}

fn json_body(resp: &Resp) -> &Value {
    resp.body.as_ref().unwrap_or_else(|| panic!("JSON ボディが無い: {:?}", resp.text()))
}

// ---------------------------------------------------------------------------
// 1. JSON スキーマ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_list_top_level_schema_is_exact() {
    let resp = get(&app(), "/v1/models").await;
    assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.text());
    let body = json_body(&resp);

    // トップレベルは object / data / has_more の 3 キーだけ。
    let mut expected = vec!["data".to_string(), "has_more".to_string(), "object".to_string()];
    expected.sort_unstable();
    assert_eq!(sorted_keys(body), expected, "body = {body}");
    assert_eq!(body["object"], "list", "body = {body}");
    assert_eq!(body["has_more"], false, "body = {body}");

    let data = body["data"].as_array().expect("data が配列でない");
    assert_eq!(data.len(), 1, "data = {data:?}");
}

#[tokio::test]
async fn model_object_keys_and_types_are_exact() {
    let resp = get(&app(), "/v1/models").await;
    assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.text());
    let body = json_body(&resp);
    let m = &body["data"][0];

    let mut expected: Vec<String> = MODEL_KEYS.iter().map(|s| s.to_string()).collect();
    expected.sort_unstable();
    assert_eq!(sorted_keys(m), expected, "model = {m}");

    // 一覧要素の同一性。
    assert_eq!(m["id"], CANONICAL, "m = {m}");
    assert_eq!(m["object"], "model", "m = {m}");
    assert_eq!(m["type"], "model", "m = {m}");
    assert_eq!(m["display_name"], CANONICAL, "m = {m}");
    assert_eq!(m["owned_by"], "denno", "m = {m}");
    assert_eq!(m["model_id"], MODEL_ID, "m = {m}");
    assert_eq!(m["model_dir"], MODEL_DIR, "m = {m}");

    // 文字列 / 数値 / 配列の型を固定する。
    for key in ["id", "object", "type", "display_name", "owned_by", "model_id", "model_dir", "quant"] {
        assert!(m[key].is_string(), "{key} が文字列でない: m = {m}");
        assert!(!m[key].as_str().unwrap_or("").is_empty(), "{key} が空: m = {m}");
    }
    for key in ["created", "context_window", "context_length", "max_context_length", "slots", "layers"] {
        assert!(m[key].is_u64(), "{key} が u64 でない: m = {m}");
    }
    assert!(
        m["created"].as_u64().unwrap() > 1_577_836_800,
        "created が実時刻でない: m = {m}"
    );
    assert!(m["slots"].as_u64().unwrap() >= 1, "slots = {}", m["slots"]);
    assert!(m["layers"].as_u64().unwrap() >= 1, "layers = {}", m["layers"]);

    // context は 3 つの別名が同じ値 (互換のため併記)。
    assert_eq!(m["context_window"], m["context_length"], "m = {m}");
    assert_eq!(m["context_length"], m["max_context_length"], "m = {m}");
    assert!(m["context_window"].as_u64().unwrap() >= 1, "m = {m}");
}

#[tokio::test]
async fn capabilities_and_aliases_are_exact() {
    let resp = get(&app(), "/v1/models").await;
    assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.text());
    let m = &json_body(&resp)["data"][0];

    let mut expected: Vec<String> = CAPABILITY_KEYS.iter().map(|s| s.to_string()).collect();
    expected.sort_unstable();
    assert_eq!(sorted_keys(&m["capabilities"]), expected, "capabilities = {}", m["capabilities"]);
    for key in CAPABILITY_KEYS {
        assert!(
            m["capabilities"][key].is_boolean(),
            "capabilities.{key} が bool でない: {}",
            m["capabilities"]
        );
    }
    for key in ["anthropic", "openai", "ollama", "tools", "streaming", "count_tokens"] {
        assert_eq!(m["capabilities"][key], true, "capabilities.{key}");
    }

    // 既定別名は model_id / その最終セグメント / model_dir の順、canonical 抜き。
    let aliases: Vec<&str> = m["aliases"]
        .as_array()
        .expect("aliases が配列でない")
        .iter()
        .map(|v| v.as_str().expect("aliases の要素が文字列でない"))
        .collect();
    assert_eq!(aliases, [MODEL_ID, MODEL_TAIL, MODEL_DIR], "aliases = {aliases:?}");
    assert!(!aliases.contains(&CANONICAL), "canonical が別名に混ざった");
}

#[tokio::test]
async fn model_by_id_body_equals_list_entry() {
    let app = app();
    let list = get(&app, "/v1/models").await;
    assert_eq!(list.status, StatusCode::OK, "body = {}", list.text());
    let entry = json_body(&list)["data"][0].clone();

    for id in [CANONICAL, MODEL_ID.replace('/', "%2F").as_str(), MODEL_TAIL, MODEL_DIR] {
        let one = get(&app, &format!("/v1/models/{id}")).await;
        assert_eq!(one.status, StatusCode::OK, "id = {id}, body = {}", one.text());
        let body = json_body(&one);
        // 単体取得は model object そのもの (list/data/has_more で包まない)。
        assert_eq!(sorted_keys(body), sorted_keys(&entry), "id = {id}: {body}");
        assert_eq!(*body, entry, "id = {id}: 単体と一覧 data[0] が不一致");
    }
}

// ---------------------------------------------------------------------------
// 2. 未知モデル 404 / ルータ fallback
// ---------------------------------------------------------------------------

/// ハンドラ由来の 404: 4 フィールド、code/param は null。
fn assert_handler_404(resp: &Resp, id: &str) {
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.text());
    let err = &json_body(resp)["error"];
    let mut expected = vec![
        "code".to_string(),
        "message".to_string(),
        "param".to_string(),
        "type".to_string(),
    ];
    expected.sort_unstable();
    assert_eq!(sorted_keys(err), expected, "error = {err}");
    assert_eq!(err["type"], "invalid_request_error", "error = {err}");
    assert_eq!(err["code"], Value::Null, "error = {err}");
    assert_eq!(err["param"], Value::Null, "error = {err}");
    assert_eq!(err["message"], format!("model: {id} not found"), "error = {err}");
    // 内部情報 (ソースパス・panic 文言) を漏らさない。
    let msg = err["message"].as_str().unwrap_or("");
    for leak in [".rs:", "panicked", "src/", "target/"] {
        assert!(!msg.contains(leak), "message に内部情報: {msg}");
    }
}

#[tokio::test]
async fn unknown_model_404_is_the_handler_envelope() {
    let app = app();
    for id in ["ghost-1449", "definitely-not-a-model", "denno-test-2"] {
        let resp = get(&app, &format!("/v1/models/{id}")).await;
        assert_handler_404(&resp, id);
    }
}

#[tokio::test]
async fn router_fallback_404_is_distinguishable_from_unknown_model() {
    let app = app();
    // 2 セグメント以上は `:id` に届かずルータの fallback。
    let resp = get(&app, "/v1/models/ghost-1449/extra").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.text());
    let err = &json_body(&resp)["error"];
    assert_eq!(err["code"], "unknown_route", "error = {err}");
    assert_eq!(err["type"], "invalid_request_error", "error = {err}");
    assert_eq!(err["param"], Value::Null, "error = {err}");
    let msg = err["message"].as_str().unwrap_or("");
    assert_eq!(msg, "Unknown route: GET /v1/models/ghost-1449/extra", "message = {msg}");

    // クエリは message に混ざらない。
    let resp = get(&app, "/v1/models/ghost-1449?x=1").await;
    assert_handler_404(&resp, "ghost-1449");
}

// ---------------------------------------------------------------------------
// 3. 認証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn models_list_401_uses_anthropic_envelope() {
    let app = app_with_key(Some(SECRET));
    let resp = get(&app, "/v1/models").await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED, "body = {}", resp.text());

    let body = json_body(&resp);
    let mut expected = vec!["error".to_string(), "type".to_string()];
    expected.sort_unstable();
    assert_eq!(sorted_keys(body), expected, "body = {body}");
    assert_eq!(body["type"], "error", "body = {body}");

    let err = &body["error"];
    let mut inner = vec!["message".to_string(), "type".to_string()];
    inner.sort_unstable();
    assert_eq!(sorted_keys(err), inner, "error = {err}");
    assert_eq!(err["type"], "authentication_error", "error = {err}");
    assert_eq!(err["message"], "invalid x-api-key", "error = {err}");
}

#[tokio::test]
async fn model_by_id_401_uses_openai_envelope() {
    let app = app_with_key(Some(SECRET));
    let resp = get(&app, &format!("/v1/models/{CANONICAL}")).await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED, "body = {}", resp.text());

    let err = &json_body(&resp)["error"];
    let mut expected = vec![
        "code".to_string(),
        "message".to_string(),
        "param".to_string(),
        "type".to_string(),
    ];
    expected.sort_unstable();
    assert_eq!(sorted_keys(err), expected, "error = {err}");
    assert_eq!(err["type"], "authentication_error", "error = {err}");
    assert_eq!(err["code"], Value::Null, "error = {err}");
    assert_eq!(err["param"], Value::Null, "error = {err}");
    assert_eq!(err["message"], "invalid x-api-key", "error = {err}");
}

#[tokio::test]
async fn auth_accepts_x_api_key_and_bearer_scheme() {
    let app = app_with_key(Some(SECRET));
    for headers in [
        &[("x-api-key", SECRET)][..],
        &[("authorization", "Bearer secret-key")][..],
        // スキーム名の大文字小文字は無視される。
        &[("authorization", "bearer secret-key")][..],
        &[("authorization", "BEARER secret-key")][..],
        // 空白は trim される。
        &[("x-api-key", "  secret-key  ")][..],
    ] {
        for uri in ["/v1/models", &format!("/v1/models/{CANONICAL}")] {
            let resp = get_auth(&app, uri, headers).await;
            assert_eq!(
                resp.status,
                StatusCode::OK,
                "headers = {headers:?}, uri = {uri}, body = {}",
                resp.text()
            );
        }
    }
}

#[tokio::test]
async fn auth_rejects_wrong_key_and_wrong_scheme() {
    let app = app_with_key(Some(SECRET));
    for headers in [
        &[("x-api-key", "wrong-key")][..],
        &[("authorization", "Bearer wrong-key")][..],
        &[("authorization", "Basic secret-key")][..],
        &[("authorization", "secret-key")][..],
        &[("x-api-key", "")][..],
    ] {
        let resp = get_auth(&app, "/v1/models", headers).await;
        assert_eq!(
            resp.status,
            StatusCode::UNAUTHORIZED,
            "headers = {headers:?}, body = {}",
            resp.text()
        );
    }
    // 正しい鍵と誤った鍵が併存するときは、どれか 1 つが一致すれば通す。
    let resp = get_auth(
        &app,
        "/v1/models",
        &[("x-api-key", "wrong-key"), ("authorization", "Bearer secret-key")],
    )
    .await;
    assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.text());
}

#[tokio::test]
async fn auth_does_not_gate_routing_errors() {
    let app = app_with_key(Some(SECRET));
    // 経路 / メソッドの判定はハンドラより前。鍵が無くても 404 / 405 が先に出る。
    let resp = get(&app, "/v1/models/ghost-1449/extra").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.text());
    assert_eq!(json_body(&resp)["error"]["code"], "unknown_route");

    let resp = send(&app, "POST", "/v1/models", &[], Some(json!({}))).await;
    assert_eq!(resp.status, StatusCode::METHOD_NOT_ALLOWED, "body = {}", resp.text());
    assert_eq!(json_body(&resp)["error"]["code"], "method_not_allowed");

    // 一方、経路が存在すれば認証が先 (未知モデルでも 404 ではなく 401)。
    let resp = get(&app, "/v1/models/ghost-1449").await;
    assert_eq!(resp.status, StatusCode::UNAUTHORIZED, "body = {}", resp.text());
    assert_eq!(json_body(&resp)["error"]["type"], "authentication_error");
}

// ---------------------------------------------------------------------------
// 4. メソッド (405 / HEAD / OPTIONS)
// ---------------------------------------------------------------------------

fn assert_405(resp: &Resp, method: &str, path: &str) {
    assert_eq!(resp.status, StatusCode::METHOD_NOT_ALLOWED, "body = {}", resp.text());
    let err = &json_body(resp)["error"];
    let mut expected = vec![
        "code".to_string(),
        "message".to_string(),
        "param".to_string(),
        "type".to_string(),
    ];
    expected.sort_unstable();
    assert_eq!(sorted_keys(err), expected, "error = {err}");
    assert_eq!(err["type"], "invalid_request_error", "error = {err}");
    assert_eq!(err["code"], "method_not_allowed", "error = {err}");
    assert_eq!(err["param"], Value::Null, "error = {err}");
    assert_eq!(
        err["message"],
        format!("Method {method} not allowed for {path}"),
        "error = {err}"
    );
}

#[tokio::test]
async fn non_get_methods_are_405_on_both_routes() {
    let app = app();
    for path in ["/v1/models", &format!("/v1/models/{CANONICAL}")] {
        for method in ["POST", "PUT", "PATCH", "DELETE"] {
            let resp = send(&app, method, path, &[], Some(json!({}))).await;
            assert_405(&resp, method, path);
        }
    }
}

#[tokio::test]
async fn head_matches_get_with_empty_body() {
    let app = app();
    // 登録済み GET ルートへの HEAD は GET と同義。axum が本文だけ落とす
    // (status / content-type は GET と同じ)。
    for uri in ["/v1/models", &*format!("/v1/models/{CANONICAL}"), "/v1/models/ghost-1449"] {
        let head = send(&app, "HEAD", uri, &[], None).await;
        let get_resp = get(&app, uri).await;
        assert_eq!(head.status, get_resp.status, "HEAD と GET の status が違う: {uri}");
        assert!(head.raw.is_empty(), "HEAD にボディが付いた: uri = {uri}, body = {}", head.text());
        assert_eq!(head.ct(), get_resp.ct(), "uri = {uri}: content-type");
        assert_eq!(
            head.hdr("x-content-type-options"),
            Some("nosniff".to_string()),
            "uri = {uri}"
        );
        assert!(head.hdr("x-request-id").is_some(), "uri = {uri}: x-request-id が無い");
    }
}

#[tokio::test]
async fn head_on_router_fallback_keeps_the_envelope_body() {
    let app = app();
    // fallback はメソッド非依存のハンドラなので、HEAD でも本文が残る
    // (GET ルートの HEAD と違い axum の本文除去が掛からない)。封筒の
    // message には実際のメソッドが載る。
    let resp = send(&app, "HEAD", "/v1/models/ghost-1449/extra", &[], None).await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.text());
    let err = &json_body(&resp)["error"];
    assert_eq!(err["code"], "unknown_route", "error = {err}");
    assert_eq!(
        err["message"],
        "Unknown route: HEAD /v1/models/ghost-1449/extra",
        "error = {err}"
    );
    assert_eq!(resp.ct(), "application/json", "content-type");
    assert_header_hygiene(&resp, "HEAD fallback 404");
}

#[tokio::test]
async fn plain_options_is_405_but_preflight_is_204() {
    let app = app();
    // CORS プリフライトでない OPTIONS はルート未対応 → 405 封筒。
    let resp = send(&app, "OPTIONS", "/v1/models", &[], None).await;
    assert_405(&resp, "OPTIONS", "/v1/models");

    // `Access-Control-Request-Method` 付きは CORS 層が先に 204 で返す。
    let resp = send(
        &app,
        "OPTIONS",
        "/v1/models",
        &[
            ("origin", "http://localhost:3000"),
            ("access-control-request-method", "GET"),
        ],
        None,
    )
    .await;
    assert_eq!(resp.status, StatusCode::NO_CONTENT, "body = {}", resp.text());
    assert!(resp.raw.is_empty(), "プリフライトにボディが付いた");
    assert_eq!(
        resp.hdr("x-content-type-options"),
        Some("nosniff".to_string())
    );
}

// ---------------------------------------------------------------------------
// 5. ヘッダ衛生
// ---------------------------------------------------------------------------

fn assert_header_hygiene(resp: &Resp, what: &str) {
    // 全応答に nosniff。
    assert_eq!(
        resp.hdr("x-content-type-options"),
        Some("nosniff".to_string()),
        "{what}: x-content-type-options"
    );
    // 全応答に x-request-id (生成値は印字可能 ASCII)。
    let rid = resp.hdr("x-request-id").unwrap_or_default();
    assert!(!rid.is_empty(), "{what}: x-request-id が無い");
    assert!(
        rid.bytes().all(|b| b.is_ascii_graphic()),
        "{what}: x-request-id が印字可能 ASCII でない: {rid:?}"
    );
    // 制御プレーン (モデルカタログ) は推論パスではないので no-store を付けない。
    assert_eq!(resp.hdr("cache-control"), None, "{what}: cache-control = {:?}", resp.hdr("cache-control"));
}

#[tokio::test]
async fn success_responses_are_utf8_json_with_hygiene_headers() {
    let app = app();
    for uri in ["/v1/models", &format!("/v1/models/{CANONICAL}")] {
        let resp = get(&app, uri).await;
        assert_eq!(resp.status, StatusCode::OK, "uri = {uri}");
        assert_eq!(resp.ct(), "application/json; charset=utf-8", "uri = {uri}");
        assert_header_hygiene(&resp, uri);
    }
}

#[tokio::test]
async fn error_responses_are_json_without_charset_and_with_hygiene_headers() {
    let app = app_with_key(Some(SECRET));
    for (what, resp) in [
        ("list 401", get(&app, "/v1/models").await),
        ("by_id 401", get(&app, &format!("/v1/models/{CANONICAL}")).await),
        ("by_id 404", get(&app, "/v1/models/ghost-1449").await),
        ("router 404", get(&app, "/v1/models/ghost-1449/extra").await),
        (
            "list 405",
            send(&app, "POST", "/v1/models", &[], Some(json!({}))).await,
        ),
    ] {
        assert_eq!(resp.ct(), "application/json", "{what}: content-type");
        assert!(!resp.ct().contains("charset"), "{what}: エラーに charset が付いた");
        assert_header_hygiene(&resp, what);
        // エラー封筒は JSON として読め、error キーを持つ。
        assert!(json_body(&resp).get("error").is_some(), "{what}: body = {}", resp.text());
    }
}
