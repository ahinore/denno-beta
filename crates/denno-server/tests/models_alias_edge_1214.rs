//! task 1214: `GET /v1/models/:id` の URL エンコード・大文字小文字・
//! 別名の重複・404 エラー封筒のエッジケース。
//!
//! `openai_models_901.rs` が「canonical / 別名で 200、未知は 404」という骨格を
//! 固定しているのに対し、ここでは境界を掘る:
//!
//! - パーセントエンコード: `%2F` と `%2f` は同じ 1 セグメントとして `/` に
//!   デコードされる。二重エンコード (`%252F`) はデコード後 `%2F` の文字列で
//!   あって `/` ではないので一致しない。`%2F` 単体 (`/` 1 文字) も未知。
//! - 大文字小文字: canonical / 別名の照合はバイト一致 (case-sensitive)。
//! - `DENNO_MODEL_ALIASES` の重複: 同じ別名・canonical 名そのもの・
//!   空項目・`alias=` (target 空) は重複排除され、`aliases` に 1 回だけ出る。
//! - 404 封筒: `{"error":{message,type,param,code}}` の 4 フィールドだけ。
//!   ハンドラ由来 (未知モデル) は `code = null`、ルータの fallback 由来
//!   (未知ルート) は `code = "unknown_route"` で区別される。
//!
//! `DENNO_MODEL_ALIASES` はプロセス全体に効くため、環境変数を触るテストは
//! このファイル内で [`ENV_LOCK`] を取って直列化し、終了時に必ず消す。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const CANONICAL: &str = "denno-test";
const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";
const MODEL_DIR: &str = "qwen-test-dir";

/// 環境変数を触るテストを 1 本ずつに絞るためのロック。
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct Resp {
    status: StatusCode,
    content_type: Option<String>,
    body: Value,
}

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, CANONICAL, None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = MODEL_DIR.to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: CANONICAL.to_string(),
        model_id: MODEL_ID.to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

async fn get(app: &Router, uri: &str) -> Resp {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    send(app, req).await
}

async fn send(app: &Router, req: Request<Body>) -> Resp {
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

fn aliases_of(body: &Value) -> Vec<String> {
    body["aliases"]
        .as_array()
        .expect("aliases が配列でない")
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect()
}

/// モデル単体取得の 404 が 4 フィールドの OpenAI 封筒であることを確かめる。
fn assert_model_not_found(resp: &Resp, id_in_message: &str) {
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.body);
    assert!(
        resp.content_type
            .as_deref()
            .unwrap_or("")
            .starts_with("application/json"),
        "content-type = {:?}",
        resp.content_type
    );
    let err = &resp.body["error"];
    let keys: Vec<&str> = err.as_object().expect("error がオブジェクトでない").keys().map(|k| k.as_str()).collect();
    let mut keys = keys;
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

// ---------------------------------------------------------------------------
// パーセントエンコード
// ---------------------------------------------------------------------------

#[tokio::test]
async fn percent_encoded_slash_is_a_single_segment() {
    let app = app();
    // `%2F` (大文字) と `%2f` (小文字) は同じ `/` にデコードされる。
    // `model_id` は `aliases` の 1 件として登録されている。
    for enc in ["Qwen%2FQwen2.5-0.5B-Instruct", "Qwen%2fQwen2.5-0.5B-Instruct"] {
        let resp = get(&app, &format!("/v1/models/{enc}")).await;
        assert_eq!(resp.status, StatusCode::OK, "enc = {enc}, body = {}", resp.body);
        assert_eq!(resp.body["id"], CANONICAL, "enc = {enc}");
        assert_eq!(resp.body["model_id"], MODEL_ID, "enc = {enc}");
    }
}

#[tokio::test]
async fn double_encoded_slash_does_not_resolve() {
    let app = app();
    // `%252F` は復号すると `%2F` という 4 文字の文字列で、`/` ではない。
    let resp = get(&app, "/v1/models/Qwen%252FQwen2.5-0.5B-Instruct").await;
    assert_model_not_found(&resp, "Qwen%2FQwen2.5-0.5B-Instruct");
    // 復号後は `/` を含む 1 セグメントのまま 404 メッセージに載る。
    assert!(
        !resp.body["error"]["message"].as_str().unwrap_or("").contains("%252F"),
        "二重エンコードが素通しされている: {}",
        resp.body
    );
}

#[tokio::test]
async fn bare_encoded_slash_is_unknown() {
    let app = app();
    // 復号結果が `/` 1 文字。既定別名のどれとも一致しない。
    let resp = get(&app, "/v1/models/%2F").await;
    assert_model_not_found(&resp, "model: / not found");
}

#[tokio::test]
async fn encoded_trailing_space_does_not_match_trimmed_alias() {
    let app = app();
    let resp = get(&app, "/v1/models/qwen-test-dir%20").await;
    assert_model_not_found(&resp, "qwen-test-dir ");
}

#[tokio::test]
async fn encoded_alias_equals_plain_alias() {
    let app = app();
    // ハイフン (`%2D`) とピリオド (`%2E`) をエンコードしても同じ別名に解決する。
    let plain = get(&app, "/v1/models/Qwen2.5-0.5B-Instruct").await;
    let encoded = get(&app, "/v1/models/Qwen2%2E5%2D0%2E5B%2DInstruct").await;
    assert_eq!(plain.status, StatusCode::OK, "body = {}", plain.body);
    assert_eq!(encoded.status, StatusCode::OK, "body = {}", encoded.body);
    assert_eq!(plain.body["id"], encoded.body["id"]);
}

#[tokio::test]
async fn malformed_percent_escape_is_passed_through_literally() {
    let app = app();
    // axum 0.7 の `Path` 抽出は不正なパーセント表記を復号せず、そのまま
    // 文字列として渡す (`%zz` はそのまま)。よって 400 ではなく、
    // 未知モデルとしてハンドラの 404 封筒になる。
    let resp = get(&app, "/v1/models/%zz").await;
    assert_model_not_found(&resp, "model: %zz not found");
}

// ---------------------------------------------------------------------------
// 大文字小文字
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_id_lookup_is_case_sensitive() {
    let app = app();
    // canonical は一致、大文字小文字を変えたものは 404。
    let ok = get(&app, &format!("/v1/models/{CANONICAL}")).await;
    assert_eq!(ok.status, StatusCode::OK, "body = {}", ok.body);
    for bad in ["Denno-Test", "DENNO-TEST", "denno-TEST"] {
        let resp = get(&app, &format!("/v1/models/{bad}")).await;
        assert_model_not_found(&resp, bad);
    }
}

#[tokio::test]
async fn alias_lookup_is_case_sensitive() {
    let app = app();
    // `/` を含む別名はパーセントエンコードして 1 セグメントとして渡す
    // (生の `/` だと別ルートになって fallback に落ちる。下の専用テスト参照)。
    // 404 メッセージに載るのは復号後の文字列。
    for (enc, decoded) in [
        ("qwen%2Fqwen2.5-0.5b-instruct", "qwen/qwen2.5-0.5b-instruct"), // 小文字化した model_id
        ("Qwen%2Fqwen2.5-0.5B-Instruct", "Qwen/qwen2.5-0.5B-Instruct"), // セグメントだけ小文字化
        ("QWEN-TEST-DIR", "QWEN-TEST-DIR"),                            // 大文字化した model_dir
        ("qwen2.5-0.5b-instruct", "qwen2.5-0.5b-instruct"),            // 小文字化した最終セグメント
    ] {
        let resp = get(&app, &format!("/v1/models/{enc}")).await;
        assert_model_not_found(&resp, decoded);
    }
}

#[tokio::test]
async fn raw_slash_in_alias_splits_the_route() {
    let app = app();
    // エンコードしない `/` はセグメント区切りなので `:id` が 2 セグメントに
    // 分かれ、モデル照合ではなくルータの fallback (unknown_route) になる。
    let resp = get(&app, "/v1/models/qwen/qwen2.5-0.5B-Instruct").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.body);
    assert_eq!(resp.body["error"]["code"], "unknown_route", "body = {}", resp.body);
}

// ---------------------------------------------------------------------------
// DENNO_MODEL_ALIASES の重複
// ---------------------------------------------------------------------------

#[tokio::test]
async fn duplicate_aliases_are_collapsed_once() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // 同一別名の繰り返し / 重複した `alias=canonical` / canonical 名そのもの /
    // `alias=` (target 空) / 空項目をまとめて流す。
    std::env::set_var(
        "DENNO_MODEL_ALIASES",
        "dup,dup,dup=denno-test,denno-test,denno-test=denno-test,empty=,, ,dup",
    );
    let app = app();

    let list = get(&app, "/v1/models").await;
    assert_eq!(list.status, StatusCode::OK, "body = {}", list.body);
    let aliases = aliases_of(&list.body["data"][0]);
    let dup_count = aliases.iter().filter(|a| *a == "dup").count();
    assert_eq!(dup_count, 1, "aliases = {aliases:?}");
    // canonical 名は別名一覧に出さない (登録済みでも重複報告しない)。
    assert!(!aliases.contains(&CANONICAL.to_string()), "aliases = {aliases:?}");
    // `empty=` は target 空なので採用される (`alias` として扱う)。
    assert_eq!(aliases.iter().filter(|a| *a == "empty").count(), 1, "aliases = {aliases:?}");
    // 空文字の別名は入らない。
    assert!(!aliases.iter().any(|a| a.is_empty()), "aliases = {aliases:?}");
    // 一覧内に重複が無いこと自体を固定する。
    let mut sorted = aliases.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), aliases.len(), "aliases = {aliases:?}");

    // 重複登録された別名でも単体取得は 1 回で 200。
    let one = get(&app, "/v1/models/dup").await;
    assert_eq!(one.status, StatusCode::OK, "body = {}", one.body);
    assert_eq!(one.body["id"], CANONICAL);

    std::env::remove_var("DENNO_MODEL_ALIASES");
}

#[tokio::test]
async fn alias_equal_to_canonical_does_not_duplicate_entry() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("DENNO_MODEL_ALIASES", "denno-test=denno-test");
    let app = app();
    let list = get(&app, "/v1/models").await;
    assert_eq!(list.status, StatusCode::OK, "body = {}", list.body);
    let aliases = aliases_of(&list.body["data"][0]);
    assert!(!aliases.contains(&CANONICAL.to_string()), "aliases = {aliases:?}");
    // canonical 自体は引き続き取得できる。
    let resp = get(&app, &format!("/v1/models/{CANONICAL}")).await;
    assert_eq!(resp.status, StatusCode::OK, "body = {}", resp.body);
    std::env::remove_var("DENNO_MODEL_ALIASES");
}

#[tokio::test]
async fn alias_targeting_other_model_is_ignored_but_name_still_404() {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var("DENNO_MODEL_ALIASES", "ghost=other-model");
    let app = app();
    let list = get(&app, "/v1/models").await;
    assert_eq!(list.status, StatusCode::OK, "body = {}", list.body);
    let aliases = aliases_of(&list.body["data"][0]);
    assert!(!aliases.contains(&"ghost".to_string()), "aliases = {aliases:?}");
    // 別名として採用していないので単体取得も 404。
    let resp = get(&app, "/v1/models/ghost").await;
    assert_model_not_found(&resp, "ghost");
    std::env::remove_var("DENNO_MODEL_ALIASES");
}

// ---------------------------------------------------------------------------
// 404 / 405 のエラー封筒
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unknown_model_404_uses_handler_envelope() {
    let app = app();
    let resp = get(&app, "/v1/models/definitely-not-a-model").await;
    assert_model_not_found(&resp, "model: definitely-not-a-model not found");
    // ハンドラ由来は `code` が null (fallback の `unknown_route` と区別)。
    assert_eq!(resp.body["error"]["code"], Value::Null);
}

#[tokio::test]
async fn empty_model_id_falls_through_to_router_fallback() {
    let app = app();
    // `/v1/models/` は `:id` が空セグメントなのでハンドラに届かず fallback。
    let resp = get(&app, "/v1/models/").await;
    assert_eq!(resp.status, StatusCode::NOT_FOUND, "body = {}", resp.body);
    let err = &resp.body["error"];
    assert_eq!(err["type"], "invalid_request_error", "error = {err}");
    assert_eq!(err["code"], "unknown_route", "error = {err}");
    assert!(err.get("param").is_some(), "error = {err}");
}

#[tokio::test]
async fn post_to_model_by_id_is_405_with_json_envelope() {
    let app = app();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/denno-test")
        .header("content-type", "application/json")
        .body(Body::from(json!({}).to_string()))
        .unwrap();
    let resp = send(&app, req).await;
    assert_eq!(resp.status, StatusCode::METHOD_NOT_ALLOWED, "body = {}", resp.body);
    assert_eq!(resp.body["error"]["type"], "invalid_request_error");
    assert_eq!(resp.body["error"]["code"], "method_not_allowed");
}
