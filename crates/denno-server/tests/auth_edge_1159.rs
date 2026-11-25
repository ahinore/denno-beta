//! task 1159: 複数 API キー認証 (`DENNO_API_KEYS`) の境界テスト。
//!
//! task 951 の統合テスト (`auth_951.rs`) が「正常系の契約」を押さえているので、
//! ここでは **境界と取りこぼし** だけを狙う:
//!
//! - `x-api-key` と `Authorization: Bearer` のどちらを欠いても、片方が一致すれば
//!   通ること / 両方だめなら 401 になること (スキーム名・空白の揺れを含む)。
//! - `DENNO_API_KEYS` の空白要素・空要素・重複・タブ/改行の正規化。
//! - 長さ違い (接頭辞 / 1 文字違い / 前後 1 バイト) のキーが一致しないこと。
//!   定数時間比較は長さも比較対象なので、NUL 埋めで長さ差が消えてはいけない。
//! - `/health` / `/ready` / `/metrics` は資格情報の有無・正否に関わらず素通し。
//! - 401 応答 (本文・ヘッダ) と正常応答に、設定キーも送信キーも漏れないこと。
//!
//! 環境変数を実際に書き換えるテストは並列実行で競合するため行わない。
//! `DENNO_API_KEYS` の意味論は、起動時に同じ関数を通る
//! [`parse_api_keys`] / [`resolve_api_keys`] を直接検証して固定する。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
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

fn get(uri: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn get_x_key(uri: &str, key: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("x-api-key", key)
        .body(Body::empty())
        .unwrap()
}

fn get_bearer(uri: &str, value: &str) -> Request<Body> {
    Request::builder()
        .method("GET")
        .uri(uri)
        .header("authorization", value)
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

fn delete(uri: &str) -> Request<Body> {
    Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap()
}

fn json_body(body: &str) -> Value {
    serde_json::from_str(body).unwrap_or(Value::Null)
}

/// 保護エンドポイントの代表的なリクエスト (資格情報なし)。
fn protected_requests() -> Vec<(&'static str, Request<Body>)> {
    vec![
        ("GET /v1/models", get("/v1/models")),
        ("GET /v1/cache/stats", get("/v1/cache/stats")),
        ("DELETE /v1/cache", delete("/v1/cache")),
        ("POST /v1/messages", post_json("/v1/messages")),
        (
            "POST /v1/messages/count_tokens",
            post_json("/v1/messages/count_tokens"),
        ),
        ("POST /v1/chat/completions", post_json("/v1/chat/completions")),
        ("POST /v1/completions", post_json("/v1/completions")),
        ("GET /api/tags", get("/api/tags")),
        ("GET /api/version", get("/api/version")),
        ("POST /api/chat", post_json("/api/chat")),
        ("POST /api/generate", post_json("/api/generate")),
    ]
}

// ---------- x-api-key と Bearer の組み合わせ ----------

#[tokio::test]
async fn bearer_and_x_api_key_are_interchangeable() {
    let app = app(Some("key-a,key-b"));

    // 単独: どちらのヘッダでも、どちらのキーでも通る。
    for key in ["key-a", "key-b"] {
        let (s, _, _) = send(&app, get_x_key("/v1/models", key)).await;
        assert_eq!(s, StatusCode::OK, "x-api-key = {key}");
        let (s, _, _) = send(&app, get_bearer("/v1/models", &format!("Bearer {key}"))).await;
        assert_eq!(s, StatusCode::OK, "Authorization: Bearer {key}");
    }

    // 片方が不一致でも、もう片方が一致すれば通す。
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "nope")
        .header("authorization", "Bearer key-b")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::OK, "bearer 側が一致");

    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "key-a")
        .header("authorization", "Bearer nope")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::OK, "x-api-key 側が一致");

    // 空の x-api-key は「候補なし」扱いで、bearer 側の一致を潰さない。
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "")
        .header("authorization", "Bearer key-a")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::OK, "空 x-api-key + 正しい bearer");

    // 非 Bearer スキームは資格情報として採用しない (x-api-key 側で通る)。
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "key-a")
        .header("authorization", "Basic key-a")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::OK);

    // 両方だめなら 401。
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .header("x-api-key", "nope")
        .header("authorization", "Bearer also-nope")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn malformed_authorization_values_are_rejected() {
    let app = app(Some("key-a"));

    for value in [
        "Bearer",       // 資格情報なし (空白なし)
        "Bearer ",      // 資格情報が空
        "Bearer    ",   // 空白だけ
        "Bearer\tkey-a", // 区切りが空白でない
        "Bearerx key-a", // スキーム名が別物
        "Token key-a",  // 未知スキーム
        "key-a",        // スキームなし
        "",             // 空文字
    ] {
        let (s, _, body) = send(&app, get_bearer("/v1/models", value)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "value = {value:?} body = {body}");
    }

    // 空白の揺れは許す (スキーム名の大小 + 資格情報前後の OWS)。
    for value in [
        "Bearer key-a",
        "bearer key-a",
        "BEARER key-a",
        "BeArEr key-a",
        "Bearer  key-a",
        "Bearer   key-a  ",
    ] {
        let (s, _, _) = send(&app, get_bearer("/v1/models", value)).await;
        assert_eq!(s, StatusCode::OK, "value = {value:?}");
    }
}

#[tokio::test]
async fn x_api_key_whitespace_and_empty_values() {
    let app = app(Some("key-a"));

    for value in ["", " ", "\t", " key", "key ", " key ", "\tkey\t"] {
        let req = Request::builder()
            .method("GET")
            .uri("/v1/models")
            .header("x-api-key", value)
            .body(Body::empty())
            .unwrap();
        let (s, _, _) = send(&app, req).await;
        // 前後 OWS は落ちるので " key " 系は通る。内部に空白を残した "ke y" は別キー。
        let expect_ok = !value.trim().is_empty() && value.trim() == "key-a";
        let want = if expect_ok {
            StatusCode::OK
        } else {
            StatusCode::UNAUTHORIZED
        };
        assert_eq!(s, want, "value = {value:?}");
    }

    // 内部の空白は落とさない = 別のキー。
    let (s, _, _) = send(&app, get_x_key("/v1/models", "ke y-a")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn duplicate_x_api_key_header_consults_the_first_value() {
    let app = app(Some("good"));
    let mut req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    // HeaderMap::get は先頭値を見る (2 番目だけ一致しても通さない)。
    req.headers_mut().append("x-api-key", HeaderValue::from_static("bad"));
    req.headers_mut().append("x-api-key", HeaderValue::from_static("good"));
    assert_eq!(send(&app, req).await.0, StatusCode::UNAUTHORIZED);

    // 1 本目が一致していれば、後続が不一致でも通る。
    let mut req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut().append("x-api-key", HeaderValue::from_static("good"));
    req.headers_mut().append("x-api-key", HeaderValue::from_static("bad"));
    assert_eq!(send(&app, req).await.0, StatusCode::OK);
}

#[tokio::test]
async fn non_utf8_header_values_do_not_panic_or_bypass() {
    let app = app(Some("key-a"));

    // x-api-key が非 UTF-8 → to_str() 失敗 → 候補なし。
    let mut req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-api-key", HeaderValue::from_bytes(b"\xff\xfe").unwrap());
    let (s, _, _) = send(&app, req).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    // 非 UTF-8 の x-api-key があっても、正しい bearer 側で通る (panic しない)。
    let mut req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    req.headers_mut()
        .insert("x-api-key", HeaderValue::from_bytes(b"\xff\xfe").unwrap());
    req.headers_mut()
        .insert("authorization", HeaderValue::from_static("Bearer key-a"));
    assert_eq!(send(&app, req).await.0, StatusCode::OK);
}

// ---------- DENNO_API_KEYS の正規化 ----------

#[test]
fn denno_api_keys_normalizes_whitespace_and_empties() {
    assert_eq!(API_KEYS_ENV, "DENNO_API_KEYS");

    // タブ / 改行 / CR も trim し、空要素は捨てる。
    assert_eq!(
        parse_api_keys("\t alpha \n, beta \r\n, , "),
        vec!["alpha".to_string(), "beta".to_string()]
    );
    // 空白だけで構成された値は「キーなし」。
    assert!(parse_api_keys("").is_empty());
    assert!(parse_api_keys("   ").is_empty());
    assert!(parse_api_keys("\t\n\r , , \t").is_empty());
    // 重複は前後空白を取り除いた上で畳む (順序は最初の出現順)。
    assert_eq!(
        parse_api_keys("a, a ,a,\ta\t"),
        vec!["a".to_string()]
    );
    // 内部の空白はキーの一部 (落とさない)。
    assert_eq!(
        parse_api_keys("a b,c"),
        vec!["a b".to_string(), "c".to_string()]
    );
    // 非 ASCII キーもそのまま扱う。
    assert_eq!(parse_api_keys("鍵-1,鍵-2"), vec!["鍵-1".to_string(), "鍵-2".to_string()]);
}

#[test]
fn denno_api_keys_union_keeps_open_default() {
    // 空 / 空白だけの出所は「未設定」と同じ (開放)。
    assert_eq!(resolve_api_keys(&[Some("")]), None);
    assert_eq!(resolve_api_keys(&[Some(" \t\n ")]), None);
    assert_eq!(resolve_api_keys(&[Some(" , , "), None]), None);
    // 出所の和。重複は 1 本に畳む。
    assert_eq!(
        resolve_api_keys(&[Some(" alpha ,, beta "), Some("beta, gamma,, \t")]),
        Some("alpha,beta,gamma".to_string())
    );
    // 正規化後は各要素が非空・trim 済みであること。
    let raw = resolve_api_keys(&[Some("\n a \t, , b \r")]).unwrap();
    for part in raw.split(',') {
        assert_eq!(part, part.trim());
        assert!(!part.is_empty());
    }
    assert_eq!(raw, "a,b");
}

#[tokio::test]
async fn empty_or_whitespace_only_key_config_closes_the_server() {
    // 空文字 / 空白だけの api_key は「キー 1 本も無い」なので、開放ではなく全拒否。
    for cfg in ["", " ", "\t", " , ,", "  ,\t, "] {
        let app = app(Some(cfg));
        for (label, req) in protected_requests() {
            let (s, _, _) = send(&app, req).await;
            assert_eq!(s, StatusCode::UNAUTHORIZED, "cfg = {cfg:?} {label}");
        }
        // キーが 1 本も無いので、どんな資格情報でも通らない。
        let (s, _, _) = send(&app, get_x_key("/v1/models", "anything")).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "cfg = {cfg:?}");
        let (s, _, _) = send(&app, get_bearer("/v1/models", "Bearer anything")).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "cfg = {cfg:?}");
    }
}

// ---------- 長さ違いのキー ----------

#[tokio::test]
async fn wrong_length_keys_never_match() {
    let secret = "sk-boundary-abc123";
    let long_app = app(Some(secret));

    let (s, _, _) = send(&long_app, get_x_key("/v1/models", secret)).await;
    assert_eq!(s, StatusCode::OK, "完全一致");

    let shorter = &secret[..secret.len() - 1];
    let longer = format!("{secret}x");
    let prefix = "sk-";
    let same_len_diff = "sk-boundary-abc124";
    assert_eq!(shorter.len() + 1, secret.len());
    assert_eq!(same_len_diff.len(), secret.len());

    for bad in [
        shorter,
        longer.as_str(),
        prefix,
        same_len_diff,
        "",
        "sk-boundary-abc12",
        "sk-boundary-abc1234",
        "xs-boundary-abc123",
    ] {
        let (s, _, _) = send(&long_app, get_x_key("/v1/models", bad)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad = {bad:?}");
        let (s, _, _) = send(&long_app, get_bearer("/v1/models", &format!("Bearer {bad}"))).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bearer bad = {bad:?}");
    }

    // 逆向き: 短いキーを設定したときに、その接頭辞/延長で通らないこと。
    let short_app = app(Some("sk-b"));
    for bad in ["sk", "sk-bb", "sk-b1", "s", ""] {
        let (s, _, _) = send(&short_app, get_x_key("/v1/models", bad)).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "bad = {bad:?}");
    }
}

#[test]
fn constant_time_eq_never_hides_a_length_difference() {
    // NUL 埋めで長さ差が消えないこと (短い方は 0 で埋められる)。
    assert!(!constant_time_eq(b"a", b"a\0"));
    assert!(!constant_time_eq(b"a\0", b"a"));
    assert!(!constant_time_eq(b"abc", b"abc\0\0"));
    assert!(!constant_time_eq(b"", b"\0"));
    assert!(constant_time_eq(b"\0", b"\0"));

    // 小さなアルファベットで全数比較: 結果が通常の == と一致する。
    let alphabet = [0u8, 1, 0xff];
    let mut corpus: Vec<Vec<u8>> = vec![Vec::new()];
    for len in 1..=3 {
        let mut level = vec![Vec::new()];
        for _ in 0..len {
            let mut next = Vec::new();
            for base in &level {
                for b in alphabet {
                    let mut v = base.clone();
                    v.push(b);
                    next.push(v);
                }
            }
            level = next;
        }
        corpus.extend(level);
    }
    for a in &corpus {
        for b in &corpus {
            assert_eq!(
                constant_time_eq(a, b),
                a == b,
                "a = {a:?}, b = {b:?}"
            );
        }
    }
}

// ---------- プローブのバイパス ----------

#[tokio::test]
async fn probes_ignore_credentials_entirely() {
    let key = "sk-probe-secret";
    let app = app(Some(key));

    for uri in ["/health", "/ready", "/metrics"] {
        // ヘッダなし。
        let (s, _, body) = send(&app, get(uri)).await;
        assert_eq!(s, StatusCode::OK, "{uri} はヘッダなしで 200 (body = {body})");

        // 不一致の資格情報でも 401 にしない。
        let (s, _, _) = send(&app, get_x_key(uri, "wrong")).await;
        assert_eq!(s, StatusCode::OK, "{uri} は誤キーでも 200");
        let (s, _, _) = send(&app, get_bearer(uri, "Bearer wrong")).await;
        assert_eq!(s, StatusCode::OK, "{uri} は誤 bearer でも 200");
        let (s, _, _) = send(&app, get_bearer(uri, "Basic wrong")).await;
        assert_eq!(s, StatusCode::OK, "{uri} は未知スキームでも 200");

        // 正しい資格情報でも同じ 200。
        let (s, _, _) = send(&app, get_x_key(uri, key)).await;
        assert_eq!(s, StatusCode::OK, "{uri} は正キーでも 200");

        // プローブの本文にキーが現れない。
        let (_, _, body) = send(&app, get(uri)).await;
        assert!(!body.contains(key), "{uri} の本文にキーが漏れた: {body}");
        assert!(!body.contains("authentication_error"), "{uri}");
    }
}

// ---------- キー漏れが無い ----------

/// `secret` が本文にもヘッダにも含まれていないことを確かめる。
fn assert_no_leak(label: &str, headers: &HeaderMap, body: &str, secrets: &[&str]) {
    for secret in secrets {
        assert!(
            !body.contains(secret),
            "{label}: 本文にキーが漏れた ({secret:?}): {body}"
        );
        for (name, value) in headers.iter() {
            let v = value.to_str().unwrap_or("");
            assert!(
                !v.contains(secret),
                "{label}: ヘッダ {name} にキーが漏れた: {v}"
            );
            // ヘッダ名側に混入する経路は無いが、念のため。
            assert!(!name.as_str().contains(secret), "{label}: ヘッダ名に漏れた");
        }
    }
}

#[tokio::test]
async fn unauthorized_responses_leak_neither_configured_nor_sent_keys() {
    let a = "sk-leak-alpha-7f3";
    let b = "sk-leak-beta-9c1";
    let app = app(Some(&format!("{a},{b}")));
    // 設定キーと同じ長さの「惜しい」送信キー。エラー本文に echo してはいけない。
    let sent = "sk-leak-alpha-000";

    for (label, req) in protected_requests() {
        let (s, headers, body) = send(&app, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{label}");
        assert_no_leak(label, &headers, &body, &[a, b]);
        assert!(
            !body.contains(sent),
            "{label}: 送信されたキーを反射した: {body}"
        );
        // 401 の封筒は JSON で、message は固定文言。
        let v = json_body(&body);
        assert!(v.is_object(), "{label}: JSON でない: {body}");
    }

    // 誤キー付きでも同様。
    for (label, req) in protected_requests() {
        let (s, headers, body) = send_with_key(app.clone(), req, sent).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{label} + sent key");
        assert_no_leak(label, &headers, &body, &[a, b, sent]);
    }
}

/// 任意のリクエストに `x-api-key` を後付けする (メソッド/ボディは保つ)。
async fn send_with_key(
    app: Router,
    mut req: Request<Body>,
    key: &str,
) -> (StatusCode, HeaderMap, String) {
    req.headers_mut()
        .insert("x-api-key", HeaderValue::from_str(key).unwrap());
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test]
async fn successful_responses_do_not_echo_keys() {
    let a = "sk-ok-alpha-2d8";
    let b = "sk-ok-beta-5e0";
    let app = app(Some(&format!("{a},{b}")));

    for key in [a, b] {
        let (s, headers, body) = send(&app, get_x_key("/v1/models", key)).await;
        assert_eq!(s, StatusCode::OK);
        assert_no_leak("GET /v1/models", &headers, &body, &[a, b]);

        let (s, headers, body) = send(&app, get_x_key("/v1/cache/stats", key)).await;
        assert_eq!(s, StatusCode::OK);
        assert_no_leak("GET /v1/cache/stats", &headers, &body, &[a, b]);

        for uri in ["/health", "/ready", "/metrics"] {
            let (s, headers, body) = send(&app, get_x_key(uri, key)).await;
            assert_eq!(s, StatusCode::OK, "{uri}");
            assert_no_leak(uri, &headers, &body, &[a, b]);
        }
    }
}

// ---------- 単一キー時代との互換 ----------

#[tokio::test]
async fn key_matching_is_case_sensitive_and_exact() {
    let app = app(Some("AlphaKey"));
    let (s, _, _) = send(&app, get_x_key("/v1/models", "AlphaKey")).await;
    assert_eq!(s, StatusCode::OK);
    for bad in ["alphakey", "ALPHAKEY", "AlphaKey ", " AlphaKey"] {
        // 前後空白は OWS として落ちるので " AlphaKey" / "AlphaKey " は通る。
        let trimmed = bad.trim();
        let (s, _, _) = send(&app, get_x_key("/v1/models", bad)).await;
        let want = if trimmed == "AlphaKey" {
            StatusCode::OK
        } else {
            StatusCode::UNAUTHORIZED
        };
        assert_eq!(s, want, "bad = {bad:?}");
    }
}

#[tokio::test]
async fn duplicate_keys_in_config_are_harmless() {
    let app = app(Some("dup,dup, dup "));
    let (s, _, _) = send(&app, get_x_key("/v1/models", "dup")).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, _) = send(&app, get_bearer("/v1/models", "Bearer dup")).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, _) = send(&app, get_x_key("/v1/models", "dupp")).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn very_long_keys_compare_exactly() {
    let long = format!("sk-{}", "a".repeat(2048));
    let app = app(Some(&long));

    let (s, _, _) = send(&app, get_x_key("/v1/models", &long)).await;
    assert_eq!(s, StatusCode::OK);

    let mut off_by_one = long.clone();
    off_by_one.pop();
    let (s, _, _) = send(&app, get_x_key("/v1/models", &off_by_one)).await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);

    let (s, _, _) = send(&app, get_bearer("/v1/models", &format!("Bearer {long}"))).await;
    assert_eq!(s, StatusCode::OK);
}
