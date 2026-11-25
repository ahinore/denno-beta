//! task 951: 複数 API キーの認証 (定数時間比較) の統合テスト。
//!
//! 検証する契約:
//!
//! - キー未設定なら従来どおり開放 (`--api-key` 無しの既定)。
//! - `DENNO_API_KEYS` / `--api-keys` / `--api-key` はいずれもカンマ区切りを
//!   受け付け、和が実効キー一覧になる (空要素・前後空白・重複は正規化)。
//! - `x-api-key` と `Authorization: Bearer` のどちらでも通り、Bearer の
//!   スキーム名は大文字小文字を無視する。両方あるときはどれか 1 つでよい。
//! - 比較は [`constant_time_eq`] (長さが違っても早期 return しない)。
//! - `/health` / `/ready` / `/metrics` はキー設定後も認証不要。
//! - 保護エンドポイントの 401 はルートに合った JSON 封筒
//!   (Anthropic / OpenAI / Ollama) で、キーの値は本文にもヘッダにも出ない。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{
    constant_time_eq, parse_api_keys, resolve_api_keys, AppState, EngineInfo, Scheduler,
    API_KEYS_ENV,
};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

// ---------- 共通ヘルパ ----------

fn state(api_key: Option<&str>) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
    Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

fn app(api_key: Option<&str>) -> Router {
    denno_server::router(state(api_key))
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

/// 認証ヘッダなしの GET。
fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

/// `x-api-key` 付き GET。
fn get_x_key(uri: &str, key: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("x-api-key", key)
        .body(Body::empty())
        .unwrap()
}

/// `Authorization: Bearer` 付き GET (スキーム表記はそのまま渡す)。
fn get_bearer(uri: &str, scheme_and_key: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("authorization", scheme_and_key)
        .body(Body::empty())
        .unwrap()
}

fn post_json(uri: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from("{}"))
        .unwrap()
}

fn json_body(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or(Value::Null)
}

// ---------- 既定 (キー未設定) は開放 ----------

#[tokio::test]
async fn no_key_configured_keeps_server_open() {
    let app = app(None);
    let (status, _, _) = send(&app, get("/v1/models")).await;
    assert_eq!(status, StatusCode::OK, "キー未設定なら従来どおり開放");
    let (status, _, _) = send(&app, post_json("/v1/messages")).await;
    assert_ne!(status, StatusCode::UNAUTHORIZED, "401 にしてはいけない");
}

// ---------- 単一キー (従来挙動の維持) ----------

#[tokio::test]
async fn single_key_accepts_both_header_styles() {
    let app = app(Some("sk-single"));
    let (status, _, _) = send(&app, get_x_key("/v1/models", "sk-single")).await;
    assert_eq!(status, StatusCode::OK, "x-api-key");
    let (status, _, _) = send(&app, get_bearer("/v1/models", "Bearer sk-single")).await;
    assert_eq!(status, StatusCode::OK, "Authorization: Bearer");
    let (status, _, _) = send(&app, get("/v1/models")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "ヘッダ無しは 401");
    let (status, _, _) = send(&app, get_x_key("/v1/models", "sk-wrong")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "不一致は 401");
}

#[tokio::test]
async fn bearer_scheme_is_case_insensitive() {
    let app = app(Some("sk-single"));
    for v in ["bearer sk-single", "BEARER sk-single", "BeArEr sk-single"] {
        let (status, _, _) = send(&app, get_bearer("/v1/models", v)).await;
        assert_eq!(status, StatusCode::OK, "scheme = {v}");
    }
    // 他のスキームは資格情報として扱わない。
    let (status, _, _) = send(&app, get_bearer("/v1/models", "Basic sk-single")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------- 複数キー ----------

#[tokio::test]
async fn every_key_in_comma_list_is_accepted() {
    let app = app(Some("alpha,beta,gamma"));
    for k in ["alpha", "beta", "gamma"] {
        let (status, _, _) = send(&app, get_x_key("/v1/models", k)).await;
        assert_eq!(status, StatusCode::OK, "key = {k}");
        let (status, _, _) = send(&app, get_bearer("/v1/models", &format!("Bearer {k}"))).await;
        assert_eq!(status, StatusCode::OK, "bearer key = {k}");
    }
    for k in ["", "alph", "alphaa", "Alpha", "delta"] {
        let (status, _, _) = send(&app, get_x_key("/v1/models", k)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "key = {k:?}");
    }
}

#[tokio::test]
async fn either_header_may_supply_the_matching_key() {
    let app = app(Some("alpha,beta"));
    // x-api-key が不一致でも Authorization が一致すれば通す。
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "nope")
        .header("authorization", "Bearer beta")
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::OK);
    // 両方不一致なら 401。
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "nope")
        .header("authorization", "Bearer also-nope")
        .body(Body::empty())
        .unwrap();
    let (status, _, _) = send(&app, req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn whitespace_around_keys_is_normalized() {
    let app = app(Some(" alpha , beta ,, "));
    for k in ["alpha", "beta"] {
        let (status, _, _) = send(&app, get_x_key("/v1/models", k)).await;
        assert_eq!(status, StatusCode::OK, "key = {k}");
    }
    // ヘッダ値の前後空白は HTTP の OWS として落とす (キーの一部にしない)。
    let (status, _, _) = send(&app, get_x_key("/v1/models", " alpha ")).await;
    assert_eq!(status, StatusCode::OK);
    // 内部の空白は落とさない (別のキー)。
    let (status, _, _) = send(&app, get_x_key("/v1/models", "al pha")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------- プローブは認証不要 ----------

#[tokio::test]
async fn probes_stay_open_even_with_keys_configured() {
    let app = app(Some("alpha,beta"));
    for uri in ["/health", "/ready", "/metrics"] {
        let (status, _, body) = send(&app, get(uri)).await;
        assert_eq!(status, StatusCode::OK, "{uri} は認証不要 (body = {body})");
    }
    // 保護エンドポイントは同じアプリで 401 のまま。
    let (status, _, _) = send(&app, get("/v1/models")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------- 401 の封筒 (ルート別) ----------

#[tokio::test]
async fn anthropic_routes_return_anthropic_envelope() {
    let app = app(Some("sk-secret-1,sk-secret-2"));
    for uri in ["/v1/messages", "/v1/messages/count_tokens"] {
        let (status, headers, body) = send(&app, post_json(uri)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri}");
        let v = json_body(&body);
        assert_eq!(v["type"], "error", "{uri}: body = {body}");
        assert_eq!(v["error"]["type"], "authentication_error", "{uri}");
        assert!(
            v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
            "{uri}: message が空"
        );
        assert_eq!(
            headers.get("content-type").and_then(|v| v.to_str().ok()),
            Some("application/json")
        );
    }
    // GET /v1/models も Anthropic 形式の封筒 (lib.rs の ApiError)。
    let (status, _, body) = send(&app, get("/v1/models")).await;
    let v = json_body(&body);
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(v["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn openai_routes_return_openai_envelope() {
    let app = app(Some("sk-secret-1"));
    for uri in ["/v1/chat/completions", "/v1/completions"] {
        let (status, _, body) = send(&app, post_json(uri)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri}");
        let v = json_body(&body);
        // OpenAI: ルートは "error" 1 つ。type / param / code を持つ。
        assert!(v.get("type").is_none(), "{uri}: body = {body}");
        assert_eq!(v["error"]["type"], "authentication_error", "{uri}");
        assert!(v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()));
        assert!(v["error"].get("param").is_some(), "{uri}: param 欠落");
        assert!(v["error"].get("code").is_some(), "{uri}: code 欠落");
    }
}

#[tokio::test]
async fn ollama_routes_return_ollama_envelope() {
    let app = app(Some("sk-secret-1"));
    for uri in ["/api/tags", "/api/version"] {
        let (status, _, body) = send(&app, get(uri)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri}");
        let v = json_body(&body);
        assert!(
            v["error"].as_str().is_some_and(|m| !m.is_empty()),
            "{uri}: body = {body}"
        );
    }
    for uri in ["/api/chat", "/api/generate"] {
        let (status, _, body) = send(&app, post_json(uri)).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{uri}");
        assert!(json_body(&body)["error"].as_str().is_some());
    }
}

// ---------- キー漏れが無い ----------

#[tokio::test]
async fn responses_never_leak_a_configured_key() {
    let keys = "sk-alpha-SECRET,sk-beta-SECRET-2";
    let app = app(Some(keys));
    let wrong = "sk-gamma-SECRET";
    for req in [
        get("/v1/models"),
        get_x_key("/v1/models", wrong),
        get_bearer("/v1/models", &format!("Bearer {wrong}")),
        post_json("/v1/messages"),
        post_json("/v1/chat/completions"),
        post_json("/api/chat"),
    ] {
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        for secret in ["sk-alpha-SECRET", "sk-beta-SECRET-2", "sk-gamma-SECRET"] {
            assert!(!body.contains(secret), "本文にキーが漏れた: {body}");
        }
        for (name, value) in headers.iter() {
            let v = value.to_str().unwrap_or("");
            assert!(
                !v.contains("SECRET"),
                "ヘッダ {name} にキーが漏れた: {v}"
            );
        }
    }
}

// ---------- 定数時間比較と正規化の単体契約 ----------

#[test]
fn constant_time_eq_matches_equality() {
    assert!(constant_time_eq(b"abc", b"abc"));
    assert!(constant_time_eq(b"", b""));
    assert!(!constant_time_eq(b"abc", b"abd"));
    assert!(!constant_time_eq(b"abc", b"abcd"), "長さ違い");
    assert!(!constant_time_eq(b"abc", b"ab"), "長さ違い (短)");
    assert!(!constant_time_eq(b"abc", b""), "空 vs 非空");
    assert!(!constant_time_eq(b"", b"abc"));
    assert!(constant_time_eq(b"\xff\x00", b"\xff\x00"), "非 UTF-8 バイト");
}

#[test]
fn parse_api_keys_trims_dedupes_and_drops_empties() {
    assert_eq!(
        parse_api_keys(" a , b ,a,, "),
        vec!["a".to_string(), "b".to_string()]
    );
    assert!(parse_api_keys("").is_empty());
    assert!(parse_api_keys(" , ,").is_empty());
    assert_eq!(parse_api_keys("solo").len(), 1);
}

#[test]
fn resolve_api_keys_unions_sources_and_keeps_open_default() {
    assert_eq!(resolve_api_keys(&[]), None);
    assert_eq!(resolve_api_keys(&[None, None]), None);
    assert_eq!(resolve_api_keys(&[Some(""), Some("  ")]), None);
    assert_eq!(
        resolve_api_keys(&[Some("a"), None]),
        Some("a".to_string())
    );
    // 単一キー設定 + 環境変数の和 (重複は 1 つに畳む)。
    assert_eq!(
        resolve_api_keys(&[Some("a,b"), Some("b,c")]),
        Some("a,b,c".to_string())
    );
}

#[test]
fn api_keys_env_name_is_stable() {
    assert_eq!(API_KEYS_ENV, "DENNO_API_KEYS");
}
