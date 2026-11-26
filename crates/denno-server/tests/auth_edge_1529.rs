//! task 1529: API キー認証の境界テスト (`auth_edge_1159.rs` の追加網)。
//!
//! `check_auth` の契約を、**単一キー / 複数キー / ヘッダの組み合わせ**の
//! 3 軸で固定する:
//!
//! - 単一キー構成では `x-api-key` と `Authorization: Bearer` のどちらでも通り、
//!   不一致なら 401。複数キー構成 (`api_key` にカンマ区切り) では設定した
//!   全キーが同じように通り、未設定キー・接頭辞・1 文字違いは通らない。
//! - `Bearer` のスキーム名は ASCII 大小を無視する。区切りは空白 1 個以上で、
//!   タブ区切りや資格情報の追加トークンは資格情報として採用しない。
//! - `x-api-key` と `Authorization` は **OR** (どちらか一致で受理)。片方が
//!   不一致でも他方が一致すれば通る — 「x-api-key が優先」ではない。
//! - 空 / 空白のみの資格情報と空のキー設定は「候補なし」として 401。
//! - 長さ違いのキー (接頭辞・延長・同長別値) は一致しない。
//! - `/health` (および `/ready` / `/metrics`) は資格情報の有無・正否に
//!   関わらず素通し。
//! - 401 封筒は Anthropic 形式で、設定キーも送信キーも本文に漏れない。
//!
//! 実装 (`src/lib.rs` の `check_auth` / `bearer_credential` /
//! `constant_time_eq`) に合わせた契約で、`src/` は変更しない。実時間・
//! スレッド競合・環境変数に依存しないので決定的。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

const MODELS: &str = "/v1/models";

// ---------------------------------------------------------------------------
// fixture
// ---------------------------------------------------------------------------

/// `api_key` は本番と同じ「カンマ区切りの 1 本」(None = 認証なし)。
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

fn json_of(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------
// リクエスト組み立て
// ---------------------------------------------------------------------------

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn get_with(uri: &str, name: &'static str, value: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header(name, value)
        .body(Body::empty())
        .unwrap()
}

fn get_x_key(uri: &str, key: &str) -> Request<Body> {
    get_with(uri, "x-api-key", key)
}

fn get_bearer(uri: &str, value: &str) -> Request<Body> {
    get_with(uri, "authorization", value)
}

fn post_json(uri: &str, body: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

fn delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

/// 認証で保護されている代表的なリクエスト (資格情報なし)。
fn protected_requests() -> Vec<(&'static str, Request<Body>)> {
    vec![
        ("GET /v1/models", get(MODELS)),
        ("GET /v1/cache/stats", get("/v1/cache/stats")),
        ("DELETE /v1/cache", delete("/v1/cache")),
        ("POST /v1/messages", post_json("/v1/messages", "{}")),
        (
            "POST /v1/messages/count_tokens",
            post_json("/v1/messages/count_tokens", "{}"),
        ),
    ]
}

// ---------------------------------------------------------------------------
// 単一キー構成
// ---------------------------------------------------------------------------

#[tokio::test]
async fn single_key_accepts_either_header_style() {
    let key = "solo-key-1529";
    let app = app(Some(key));

    let (s, _, _) = send(&app, get_x_key(MODELS, key)).await;
    assert_eq!(s, StatusCode::OK, "x-api-key 完全一致");
    let (s, _, _) = send(&app, get_bearer(MODELS, &format!("Bearer {key}"))).await;
    assert_eq!(s, StatusCode::OK, "Authorization: Bearer 完全一致");

    // 資格情報なし / 不一致は 401。
    let (s, _, _) = send(&app, get(MODELS)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "資格情報なし");
    let (s, _, _) = send(&app, get_x_key(MODELS, "solo-key-1520")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "1 文字違い (x-api-key)");
    let (s, _, _) = send(&app, get_bearer(MODELS, "Bearer solo-key-1520")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "1 文字違い (Bearer)");
}

// ---------------------------------------------------------------------------
// 複数キー構成
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_key_accepts_every_configured_key_and_nothing_else() {
    let app = app(Some("alpha-1529,beta-1529,gamma-1529"));

    for key in ["alpha-1529", "beta-1529", "gamma-1529"] {
        let (s, _, body) = send(&app, get_x_key(MODELS, key)).await;
        assert_eq!(s, StatusCode::OK, "x-api-key = {key} body = {body}");
        let (s, _, body) = send(&app, get_bearer(MODELS, &format!("Bearer {key}"))).await;
        assert_eq!(s, StatusCode::OK, "Bearer = {key} body = {body}");
    }

    for bad in [
        "delta-1529",  // 未設定
        "alpha",       // 接頭辞
        "alpha-152",   // 短い
        "alpha-15290", // 長い
        "ALPHA-1529",  // キーの大小は畳まない
    ] {
        let (s, _, _) = send(&app, get_x_key(MODELS, bad)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad x-api-key = {bad:?}");
        let (s, _, _) = send(&app, get_bearer(MODELS, &format!("Bearer {bad}"))).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad bearer = {bad:?}");
    }
}

#[tokio::test]
async fn multi_key_config_tolerates_surrounding_whitespace() {
    // 起動時は `resolve_api_keys` が正規化するが、`AppState` を直接組む
    // テストでは生の文字列が入り得る。`check_auth` は各キーを trim してから
    // 比較するので、前後空白付きの設定でも一致する。
    let app = app(Some("  alpha-1529 ,\tbeta-1529\t, gamma-1529  "));

    for key in ["alpha-1529", "beta-1529", "gamma-1529"] {
        let (s, _, _) = send(&app, get_x_key(MODELS, key)).await;
        assert_eq!(s, StatusCode::OK, "key = {key}");
    }
    let (s, _, _) = send(&app, get_x_key(MODELS, "alpha-1529 ")).await;
    assert_eq!(s, StatusCode::OK, "送信側の前後空白も trim される");
}

#[tokio::test]
async fn multi_key_prefix_pairs_do_not_cross_match() {
    // 一方が他方の接頭辞でも、短いキーで長いキーには一致しない。
    let app = app(Some("ab-1529,abc-1529"));

    for key in ["ab-1529", "abc-1529"] {
        let (s, _, _) = send(&app, get_x_key(MODELS, key)).await;
        assert_eq!(s, StatusCode::OK, "key = {key}");
    }
    for bad in ["a-1529", "ab-152", "abc-152", "abc-15290"] {
        let (s, _, _) = send(&app, get_x_key(MODELS, bad)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad = {bad:?}");
    }
}

// ---------------------------------------------------------------------------
// Bearer スキームの字句
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bearer_scheme_name_is_case_insensitive() {
    let key = "scheme-key-1529";
    let app = app(Some(key));

    for value in [
        "Bearer scheme-key-1529",
        "bearer scheme-key-1529",
        "BEARER scheme-key-1529",
        "BeArEr scheme-key-1529",
    ] {
        let (s, _, body) = send(&app, get_bearer(MODELS, value)).await;
        assert_eq!(s, StatusCode::OK, "value = {value:?} body = {body}");
    }
}

#[tokio::test]
async fn bearer_requires_a_space_separator_and_a_single_credential() {
    let key = "scheme-key-1529";
    let app = app(Some(key));

    for value in [
        "Bearer",              // 資格情報なし
        "Bearer ",             // 空
        "Bearer   ",           // 空白だけ
        "Bearer\tscheme-key-1529", // 区切りが空白でない (タブ)
        "Bearerx scheme-key-1529", // スキーム名違い
        "Basic scheme-key-1529",   // 未知スキーム
        "scheme-key-1529",         // スキームなし
        "Bearer scheme-key-1529 extra", // 資格情報に余分なトークン
    ] {
        let (s, _, body) = send(&app, get_bearer(MODELS, value)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "value = {value:?} body = {body}");
    }

    // 区切りの空白が複数・資格情報の前後に OWS があっても同じ資格情報。
    for value in [
        "Bearer  scheme-key-1529",
        "Bearer   scheme-key-1529  ",
    ] {
        let (s, _, _) = send(&app, get_bearer(MODELS, value)).await;
        assert_eq!(s, StatusCode::OK, "value = {value:?}");
    }

    // スキーム名の前に空白があると、その空白が区切りになる (スキーム名が空)。
    let (s, _, _) = send(&app, get_bearer(MODELS, "  Bearer scheme-key-1529")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "先頭 OWS は許さない");
}

// ---------------------------------------------------------------------------
// 資格情報の空白
// ---------------------------------------------------------------------------

#[tokio::test]
async fn credential_whitespace_is_trimmed_but_inner_space_is_significant() {
    let key = "ws-key-1529";
    let app = app(Some(key));

    // x-api-key は前後 OWS を落とす。
    for value in ["ws-key-1529", " ws-key-1529", "ws-key-1529 ", "  ws-key-1529  "] {
        let (s, _, _) = send(&app, get_x_key(MODELS, value)).await;
        assert_eq!(s, StatusCode::OK, "x-api-key = {value:?}");
    }

    // 空 / 空白のみは「候補なし」= 401。
    for value in ["", " ", "\t", "   \t  "] {
        let (s, _, _) = send(&app, get_x_key(MODELS, value)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "x-api-key = {value:?}");
    }

    // 内部の空白はキーの一部。分割もされない。
    for value in ["ws-key 1529", "ws-key-1529,ws-key-1529"] {
        let (s, _, _) = send(&app, get_x_key(MODELS, value)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "x-api-key = {value:?}");
    }
}

// ---------------------------------------------------------------------------
// 2 ヘッダの関係は OR (優先順位ではない)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn api_key_headers_are_ored_not_precedence() {
    let app = app(Some("good-1529,best-1529"));

    // x-api-key = 不一致 + Bearer = 一致 → 通る。
    let req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .header("x-api-key", "wrong-1529")
        .header("authorization", "Bearer best-1529")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK, "bearer 側が一致");

    // x-api-key = 一致 + Bearer = 不一致 → 通る (x-api-key を優先して
    // 401 にしない)。
    let req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .header("x-api-key", "good-1529")
        .header("authorization", "Bearer wrong-1529")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK, "x-api-key 側が一致");

    // 片方が空でも、他方が一致すれば通る。
    let req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .header("x-api-key", "")
        .header("authorization", "Bearer good-1529")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK, "空 x-api-key + 正 bearer");

    let req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .header("x-api-key", "good-1529")
        .header("authorization", "Bearer ")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK, "正 x-api-key + 空 bearer");

    // 非 Bearer スキームの Authorization は候補にならないが、x-api-key で通る。
    let req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .header("x-api-key", "good-1529")
        .header("authorization", "Basic good-1529")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK, "Basic は無視して x-api-key で通す");

    // 両方不一致なら 401。
    let req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .header("x-api-key", "wrong-1529")
        .header("authorization", "Bearer also-wrong-1529")
        .body(Body::empty())
        .unwrap();
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn non_utf8_x_api_key_does_not_block_a_valid_bearer() {
    let key = "utf8-bearer-1529";
    let app = app(Some(key));

    // 非 UTF-8 の x-api-key は to_str() 失敗で候補なし。bearer 側で通る。
    let mut req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-api-key", HeaderValue::from_bytes(b"\xff\xfe").unwrap());
    req.headers_mut()
        .insert("authorization", HeaderValue::from_static("Bearer utf8-bearer-1529"));
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::OK);

    // 非 UTF-8 のみなら 401 (panic しない)。
    let mut req = Request::builder()
        .method("GET")
        .uri(MODELS)
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-api-key", HeaderValue::from_bytes(b"\xff\xfe").unwrap());
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// 空のキー設定
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_key_configuration_denies_all_protected_routes() {
    // `Some("")` / 空白・カンマのみ = キー 0 本。None (開放) ではないので全拒否。
    for cfg in ["", " ", "\t", " , ,", "  ,\t,  "] {
        let app = app(Some(cfg));
        for (label, req) in protected_requests() {
            let (s, _, body) = send(&app, req).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "cfg = {cfg:?} {label} body = {body}");
        }
        // どんな資格情報でも通らない。
        let (s, _, _) = send(&app, get_x_key(MODELS, "anything")).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "cfg = {cfg:?} x-api-key");
        let (s, _, _) = send(&app, get_bearer(MODELS, "Bearer anything")).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "cfg = {cfg:?} bearer");
    }
}

#[tokio::test]
async fn no_configuration_stays_open() {
    // `None` = 認証なし (従来どおり開放)。この既定を壊さない。
    let app = app(None);
    for (label, req) in protected_requests() {
        let (s, _, _) = send(&app, req).await;
        assert_ne!(s, StatusCode::UNAUTHORIZED, "{label} は認証で止まらない");
    }
}

// ---------------------------------------------------------------------------
// 長さ違い
// ---------------------------------------------------------------------------

#[tokio::test]
async fn wrong_length_keys_never_match() {
    let secret = "sk-1529-abcdef";
    let app = app(Some(secret));

    let (s, _, _) = send(&app, get_x_key(MODELS, secret)).await;
    assert_eq!(s, StatusCode::OK, "完全一致");

    let shorter = &secret[..secret.len() - 1];
    let longer = format!("{secret}x");
    let prefix = &secret[..5];
    let same_len_diff = "sk-1529-abcdeg";

    assert_eq!(shorter.len() + 1, secret.len());
    assert_eq!(same_len_diff.len(), secret.len());

    for bad in [
        shorter,
        longer.as_str(),
        prefix,
        same_len_diff,
        "",
        "sk-",
        "SK-1529-ABCDEF",
    ] {
        let (s, _, _) = send(&app, get_x_key(MODELS, bad)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad x-api-key = {bad:?}");
        let (s, _, _) = send(&app, get_bearer(MODELS, &format!("Bearer {bad}"))).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad bearer = {bad:?}");
    }

    // 逆向き: 短いキー設定に対し、接頭辞 / 延長 / 同長別値は通らない。
    let short_app = self::app(Some("sk-1529"));
    for bad in ["sk-152", "sk-15299", "sk-1530", "sk-1529x"] {
        let (s, _, _) = send(&short_app, get_x_key(MODELS, bad)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad = {bad:?}");
    }
}

// ---------------------------------------------------------------------------
// /health ほかの免除
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_and_probes_bypass_auth() {
    let key = "probe-key-1529";
    let app = app(Some(key));

    for uri in ["/health", "/ready", "/metrics"] {
        // 資格情報なし。
        let (s, _, body) = send(&app, get(uri)).await;
        assert_eq!(s, StatusCode::OK, "{uri} ヘッダなし body = {body}");

        // 不一致の資格情報でも 401 にしない。
        let (s, _, _) = send(&app, get_x_key(uri, "wrong-1529")).await;
        assert_eq!(s, StatusCode::OK, "{uri} 誤 x-api-key");
        let (s, _, _) = send(&app, get_bearer(uri, "Bearer wrong-1529")).await;
        assert_eq!(s, StatusCode::OK, "{uri} 誤 bearer");
        let (s, _, _) = send(&app, get_bearer(uri, "Basic wrong-1529")).await;
        assert_eq!(s, StatusCode::OK, "{uri} 未知スキーム");

        // 正しい資格情報でも同じ 200。
        let (s, _, _) = send(&app, get_x_key(uri, key)).await;
        assert_eq!(s, StatusCode::OK, "{uri} 正キー");

        // 本文にキーも認証エラーも現れない。
        let (_, _, body) = send(&app, get(uri)).await;
        assert!(!body.contains(key), "{uri} の本文にキーが漏れた: {body}");
        assert!(
            !body.contains("authentication_error"),
            "{uri} の本文に認証エラーが現れた: {body}"
        );
    }

    // 対照: 保護エンドポイントは同じ構成で 401。
    let (s, _, _) = send(&app, get(MODELS)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED, "対照: /v1/models");
}

// ---------------------------------------------------------------------------
// 401 封筒と漏れ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unauthorized_envelope_is_anthropic_and_does_not_echo_credentials() {
    let configured = "envelope-key-1529";
    let provided = "attacker-guess-1529";
    let app = app(Some(configured));

    let (s, headers, body) = send(&app, get_x_key(MODELS, provided)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    assert!(ct.contains("application/json"), "content-type = {ct}");

    let v = json_of(&body);
    assert_eq!(v["type"], "error", "body = {body}");
    assert_eq!(
        v["error"]["type"], "authentication_error",
        "body = {body}"
    );
    assert!(
        v["error"]["message"].is_string(),
        "message は文字列: {body}"
    );
    // 汎用メッセージで、どのヘッダを送ったか・正解に近いかを漏らさない。
    assert_eq!(v["error"]["message"], "invalid x-api-key", "body = {body}");

    assert!(!body.contains(configured), "設定キーが本文に漏れた: {body}");
    assert!(!body.contains(provided), "送信キーが本文に漏れた: {body}");
    for (name, value) in headers.iter() {
        let hv = value.to_str().unwrap_or("");
        assert!(!hv.contains(configured), "ヘッダ {name} に設定キー: {hv}");
        assert!(!hv.contains(provided), "ヘッダ {name} に送信キー: {hv}");
    }
}

#[tokio::test]
async fn unauthorized_body_is_identical_for_missing_and_wrong_credentials() {
    let app = app(Some("envelope-key-1529"));

    let (s1, _, b1) = send(&app, get(MODELS)).await;
    let (s2, _, b2) = send(&app, get_x_key(MODELS, "guess-1529")).await;
    let (s3, _, b3) = send(&app, get_bearer(MODELS, "Bearer guess-1529")).await;

    assert_eq!((s1, s2, s3), (
        StatusCode::UNAUTHORIZED,
        StatusCode::UNAUTHORIZED,
        StatusCode::UNAUTHORIZED
    ));
    // 応答本文から「資格情報を送ったか」「どの方式か」を区別できない。
    assert_eq!(b1, b2, "欠落と誤 x-api-key で本文が違う");
    assert_eq!(b1, b3, "欠落と誤 bearer で本文が違う");
}
