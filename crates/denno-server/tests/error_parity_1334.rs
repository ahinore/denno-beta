//! task 1334: エラー応答のプロトコル横断パリティ (401 / 413 / 503 / 400)。
//!
//! OpenAI (`/v1/chat/completions`) / Anthropic (`/v1/messages`) / Ollama
//! (`/api/chat`) の 3 経路について、同じ入力条件で
//!
//! - HTTP ステータス
//! - `Content-Type` (JSON 封筒であること)
//! - `Retry-After` の有無 (この 4 種のエラーでは付かない)
//! - エラー封筒の形 (プロトコルごとの流儀は保ちつつ、同一条件では全経路で揃う)
//!
//! が一致することを固定する。対象は次の 4 条件:
//!
//! 1. 401: API キー認証が有効なのに資格情報が無い / 不一致
//! 2. 413: ボディが上限を超える (共通ミドルウェアの封筒)
//! 3. 503: drain 中 (`begin_drain` 後) の新規生成要求
//! 4. 400: JSON 構文エラー
//!
//! すべて決定的: 固定 sleep も競合も使わない。drain はライフサイクルを直接
//! 遷移させ、認証・ボディ上限は設定値で作る。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::RateLimiter;
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// ボディ上限テストで使う上限 (byte)。封筒にこの値が載る。
const BODY_LIMIT: usize = 4096;

// ---------------------------------------------------------------------------
// プロトコル定義
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Proto {
    Anthropic,
    OpenAi,
    Ollama,
}

const PROTOCOLS: [Proto; 3] = [Proto::Anthropic, Proto::OpenAi, Proto::Ollama];

impl Proto {
    fn uri(self) -> &'static str {
        match self {
            Proto::Anthropic => "/v1/messages",
            Proto::OpenAi => "/v1/chat/completions",
            Proto::Ollama => "/api/chat",
        }
    }

    /// 各経路の最小の有効リクエスト (ダミーモデルで完走する設定)。
    fn valid_body(self) -> Value {
        match self {
            Proto::Anthropic => json!({
                "model": "denno-test",
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
            }),
            Proto::OpenAi => json!({
                "model": "denno-test",
                "max_tokens": 2,
                "temperature": 0.0,
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
            }),
            Proto::Ollama => json!({
                "model": "denno-test",
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
                "options": {"temperature": 0.0},
            }),
        }
    }

    /// ボディ上限を超える、構文としては有効なリクエスト。
    fn oversized_body(self) -> Vec<u8> {
        let pad = "a".repeat(BODY_LIMIT * 2);
        let body = match self {
            Proto::Anthropic => json!({
                "model": "denno-test",
                "max_tokens": 2,
                "messages": [{"role": "user", "content": "t0"}],
                "zz_pad_1334": pad,
            }),
            Proto::OpenAi => json!({
                "model": "denno-test",
                "max_tokens": 2,
                "messages": [{"role": "user", "content": "t0"}],
                "zz_pad_1334": pad,
            }),
            Proto::Ollama => json!({
                "model": "denno-test",
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
                "zz_pad_1334": pad,
            }),
        };
        let bytes = serde_json::to_vec(&body).unwrap();
        assert!(bytes.len() > BODY_LIMIT);
        bytes
    }
}

// ---------------------------------------------------------------------------
// App / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn state(api_key: Option<&str>) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
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

/// レート制限を無効化し、ボディ上限を明示したルーター。env に依存しない。
fn app_with_body_limit(state: Arc<AppState>, max_body_bytes: usize) -> Router {
    denno_server::router_with_body_limit(
        state,
        CorsConfig::default(),
        RateLimiter::disabled(),
        max_body_bytes,
    )
}

fn post_bytes(uri: &str, body: Vec<u8>) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn content_type(headers: &HeaderMap) -> String {
    headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn retry_after(headers: &HeaderMap) -> Option<String> {
    headers
        .get(header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

fn parse_json(proto: Proto, text: &str) -> Value {
    serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("{:?}: エラー本文が JSON でない ({e}): {text}", proto))
}

/// すべてのエラー応答に共通の前提: JSON の Content-Type、`Retry-After` なし。
fn assert_common(error_label: &str, proto: Proto, status: StatusCode, headers: &HeaderMap) {
    let ct = content_type(headers);
    assert!(
        ct.starts_with("application/json"),
        "{error_label}/{:?}: Content-Type が JSON でない: {ct:?}",
        proto
    );
    assert!(
        retry_after(headers).is_none(),
        "{error_label}/{:?}: この条件で Retry-After を付けてはいけない: {:?}",
        proto,
        retry_after(headers)
    );
    assert_ne!(status, StatusCode::OK, "エラー条件で 200 が返った");
}

// ---------------------------------------------------------------------------
// 1. 401: 認証失敗の封筒
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unauthorized_envelope_parity() {
    const SECRET: &str = "sk-parity-1334-secret";
    let state = state(Some(SECRET));
    let app = app_with_body_limit(state, BODY_LIMIT);

    for proto in PROTOCOLS {
        let body = serde_json::to_vec(&proto.valid_body()).unwrap();

        // 資格情報なし。
        let (status, headers, text) = send(&app, post_bytes(proto.uri(), body.clone())).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "{:?}: body = {text}",
            proto
        );
        assert_common("401", proto, status, &headers);

        let v = parse_json(proto, &text);
        match proto {
            Proto::Anthropic => {
                assert_eq!(v["type"], "error", "{v}");
                assert_eq!(v["error"]["type"], "authentication_error", "{v}");
                assert!(
                    v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
                    "{v}"
                );
                // 封筒は {type, message} の 2 キーだけ。
                let keys: Vec<&str> = v["error"]
                    .as_object()
                    .expect("error がオブジェクトでない")
                    .keys()
                    .map(String::as_str)
                    .collect();
                assert_eq!(keys.len(), 2, "{:?}: error のキー = {keys:?}", proto);
                assert!(keys.contains(&"type") && keys.contains(&"message"), "{v}");
            }
            Proto::OpenAi => {
                assert_eq!(v["error"]["type"], "authentication_error", "{v}");
                assert!(
                    v["error"]["message"].as_str().is_some_and(|m| !m.is_empty()),
                    "{v}"
                );
                // OpenAI 封筒は param / code を必ず持つ (null でも)。
                assert!(v["error"].get("param").is_some(), "{v}");
                assert!(v["error"]["param"].is_null(), "{v}");
                assert!(v["error"].get("code").is_some(), "{v}");
                assert!(v["error"]["code"].is_null(), "{v}");
                assert!(v.get("type").is_none(), "OpenAI 封筒に top-level type: {v}");
            }
            Proto::Ollama => {
                assert!(
                    v["error"].as_str().is_some_and(|m| !m.is_empty()),
                    "Ollama の本文が非空文字列でない: {v}"
                );
                assert_eq!(
                    v.as_object().map(|o| o.len()),
                    Some(1),
                    "Ollama 封筒は error 1 キーだけ: {v}"
                );
            }
        }

        // キーが本文に漏れていない。
        assert!(!text.contains(SECRET), "{:?}: キーが本文に漏れた: {text}", proto);

        // 不一致の資格情報でも同じ 401。
        let req = Request::builder()
            .method("POST")
            .uri(proto.uri())
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-api-key", "wrong-key")
            .body(Body::from(body.clone()))
            .unwrap();
        let (s, _, t) = send(&app, req).await;
        assert_eq!(s, StatusCode::UNAUTHORIZED, "{:?}: {t}", proto);

        // 正しいキーなら 401 にならない (認証以外の理由で 401 を返していない)。
        let req = Request::builder()
            .method("POST")
            .uri(proto.uri())
            .header(header::CONTENT_TYPE, "application/json")
            .header("x-api-key", SECRET)
            .body(Body::from(body))
            .unwrap();
        let (s, _, t) = send(&app, req).await;
        assert_eq!(s, StatusCode::OK, "{:?}: 正キーで 200 でない: {t}", proto);
    }
}

// ---------------------------------------------------------------------------
// 2. 413: ボディ上限超過の封筒 (プロトコル非依存の共通ミドルウェア)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn payload_too_large_envelope_is_identical_across_protocols() {
    let app = app_with_body_limit(state(None), BODY_LIMIT);

    let mut envelopes: Vec<(Proto, Value)> = Vec::new();
    for proto in PROTOCOLS {
        let (status, headers, text) = send(&app, post_bytes(proto.uri(), proto.oversized_body())).await;
        assert_eq!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{:?}: body = {text}",
            proto
        );
        assert_common("413", proto, status, &headers);
        let v = parse_json(proto, &text);
        assert_eq!(v["type"], "error", "{:?}: {v}", proto);
        assert_eq!(v["error"]["type"], "invalid_request_error", "{:?}: {v}", proto);
        let msg = v["error"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains(&BODY_LIMIT.to_string()),
            "{:?}: メッセージに上限値が無い: {msg}",
            proto
        );
        envelopes.push((proto, v));
    }

    // 3 経路で封筒そのものが一致する (共通ミドルウェアの証拠)。
    let (first_proto, first) = &envelopes[0];
    for (proto, v) in &envelopes[1..] {
        assert_eq!(
            first, v,
            "413 封筒が {:?} と {:?} で異なる",
            first_proto, proto
        );
    }
}

// ---------------------------------------------------------------------------
// 3. 503: drain 中の新規要求の封筒
// ---------------------------------------------------------------------------

#[tokio::test]
async fn draining_envelope_parity() {
    let state = state(None);
    let app = app_with_body_limit(state.clone(), BODY_LIMIT);
    state.scheduler.lifecycle().begin_drain();

    let mut messages: Vec<(Proto, String)> = Vec::new();
    for proto in PROTOCOLS {
        let body = serde_json::to_vec(&proto.valid_body()).unwrap();
        let (status, headers, text) = send(&app, post_bytes(proto.uri(), body)).await;
        assert_eq!(
            status,
            StatusCode::SERVICE_UNAVAILABLE,
            "{:?}: body = {text}",
            proto
        );
        assert_common("503", proto, status, &headers);

        let v = parse_json(proto, &text);
        let msg = match proto {
            Proto::Anthropic => {
                assert_eq!(v["type"], "error", "{v}");
                assert_eq!(v["error"]["type"], "overloaded_error", "{v}");
                v["error"]["message"].as_str().unwrap_or("").to_string()
            }
            Proto::OpenAi => {
                // Anthropic の overloaded_error は OpenAI では server_error に寄る。
                assert_eq!(v["error"]["type"], "server_error", "{v}");
                assert!(v["error"]["param"].is_null(), "{v}");
                assert!(v["error"]["code"].is_null(), "{v}");
                v["error"]["message"].as_str().unwrap_or("").to_string()
            }
            Proto::Ollama => v["error"].as_str().unwrap_or("").to_string(),
        };
        assert!(
            msg.contains("draining"),
            "{:?}: drain 理由が本文に無い: {msg}",
            proto
        );
        assert!(!msg.is_empty(), "{:?}: メッセージが空", proto);
        messages.push((proto, msg));
    }

    // 3 経路で同一の理由文を返す (パリティ)。
    let (first_proto, first_msg) = &messages[0];
    for (proto, msg) in &messages[1..] {
        assert_eq!(
            first_msg, msg,
            "503 の理由文が {:?} と {:?} で異なる",
            first_proto, proto
        );
    }

    // liveness (/health) は drain 中でも 200 のまま。
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    assert_eq!(send(&app, req).await.0, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// 4. 400: JSON 構文エラーの封筒
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_json_envelope_parity() {
    let app = app_with_body_limit(state(None), BODY_LIMIT);

    for proto in PROTOCOLS {
        for malformed in [b"{not json".to_vec(), b"".to_vec()] {
            let (status, headers, text) = send(&app, post_bytes(proto.uri(), malformed)).await;
            assert_eq!(
                status,
                StatusCode::BAD_REQUEST,
                "{:?}: body = {text}",
                proto
            );
            assert_common("400", proto, status, &headers);

            let v = parse_json(proto, &text);
            let msg = match proto {
                Proto::Anthropic => {
                    assert_eq!(v["type"], "error", "{v}");
                    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
                    v["error"]["message"].as_str().unwrap_or("").to_string()
                }
                Proto::OpenAi => {
                    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
                    assert!(v["error"]["param"].is_null(), "{v}");
                    assert!(v["error"]["code"].is_null(), "{v}");
                    v["error"]["message"].as_str().unwrap_or("").to_string()
                }
                Proto::Ollama => v["error"].as_str().unwrap_or("").to_string(),
            };
            assert!(
                msg.starts_with("invalid request body"),
                "{:?}: 構文エラーの理由が揃っていない: {msg}",
                proto
            );
        }
    }
}
