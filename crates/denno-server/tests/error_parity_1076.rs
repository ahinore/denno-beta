//! task 1076: 期限・過負荷エラー (429 / 503 / 504 / 408) の封筒が
//! **プロトコルごとに正しい形**で返ることを固定する。
//!
//! 対象は 3 経路:
//!
//! - Anthropic : `POST /v1/messages`      → `{"type":"error","error":{...}}`
//! - OpenAI    : `POST /v1/chat/completions` → `{"error":{message,type,param,code}}`
//! - Ollama    : `POST /api/chat`         → `{"error":"..."}`
//!
//! 検証する 4 条件 (すべて決定的。固定 sleep は「十分に長い」方向にだけ使う):
//!
//! 1. 429: レート制限 (バケットを使い切る)。共通ミドルウェアの封筒。
//! 2. 503: drain 中の新規生成要求。ハンドラがプロトコル別に返す。
//! 3. 504: ハンドラ期限 (エンジンが期限内に応答しない)。共通ミドルウェア。
//! 4. 408: ボディ読み込み期限 (クライアントがボディを送り切らない)。共通ミドルウェア。
//!
//! 3 / 4 は [`denno_server::RequestTimeouts`] を明示したルータでだけ有効になる
//! (既定は無効 = 従来と同一挙動)。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::ratelimit::{RateLimitConfig, RateLimiter};
use denno_server::{
    AppState, CorsConfig, EngineInfo, RequestTimeouts, Scheduler,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tokio_stream::StreamExt;
use tower::ServiceExt;

/// ボディ上限 (この 4 条件では 413 を踏まない大きさ)。
const BODY_LIMIT: usize = 64 * 1024;

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

    /// 各経路の最小の有効リクエスト (非ストリーミング)。
    fn valid_body(self) -> Value {
        match self {
            Proto::Anthropic | Proto::OpenAi => json!({
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
}

// ---------------------------------------------------------------------------
// モデル
// ---------------------------------------------------------------------------

/// 1 forward ごとに `delay` だけ眠る決定的モデル。
/// ハンドラ期限 (504) を「必ず超える」ために使う。
struct SlowModel {
    delay: Duration,
}

impl Model for SlowModel {
    fn context(&self) -> usize {
        512
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        std::thread::sleep(self.delay);
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = (last + 1) % DUMMY_VOCAB;
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, 512)
    }
}

// ---------------------------------------------------------------------------
// App / HTTP ヘルパ
// ---------------------------------------------------------------------------

fn state_with(model: Box<dyn Model>) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

fn fast_state() -> Arc<AppState> {
    state_with(Box::new(SlowModel { delay: Duration::ZERO }))
}

/// リクエスト期限つきのルータ。
fn app_with(
    state: Arc<AppState>,
    limiter: RateLimiter,
    timeouts: RequestTimeouts,
) -> Router {
    denno_server::router_with_request_timeouts(
        state,
        CorsConfig::default(),
        limiter,
        BODY_LIMIT,
        timeouts,
    )
}

fn post_json(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

/// `delay` 待ってからボディを 1 回だけ流す (408 用)。
fn slow_body(delay: Duration, body: Vec<u8>) -> Body {
    let stream = tokio_stream::iter(vec![body]).then(move |chunk| async move {
        tokio::time::sleep(delay).await;
        Ok::<_, std::io::Error>(chunk)
    });
    Body::from_stream(stream)
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
        .unwrap_or_else(|e| panic!("{proto:?}: エラー本文が JSON でない ({e}): {text}"))
}

/// 封筒の形をプロトコル別に検証し、本文のメッセージを返す。
///
/// `expect_type` は Anthropic / OpenAI の `error.type`。`expect_code` は OpenAI の
/// `error.code` (Anthropic は 429 のときだけ code / param を残す既存契約)。
fn assert_envelope(
    label: &str,
    proto: Proto,
    status: StatusCode,
    headers: &HeaderMap,
    text: &str,
    expect_type: &str,
    expect_code: Option<&str>,
) -> String {
    assert_ne!(status, StatusCode::OK, "{label}/{proto:?}: エラー条件で 200");
    assert!(
        content_type(headers).starts_with("application/json"),
        "{label}/{proto:?}: Content-Type が JSON でない: {:?}",
        content_type(headers)
    );
    let v = parse_json(proto, text);
    match proto {
        Proto::Anthropic => {
            assert_eq!(v["type"], "error", "{label}/Anthropic: 外枠が無い: {v}");
            assert_eq!(v["error"]["type"], expect_type, "{label}/Anthropic: {v}");
            let msg = v["error"]["message"].as_str().unwrap_or("");
            assert!(!msg.is_empty(), "{label}/Anthropic: message が空: {v}");
            let keys = v["error"]
                .as_object()
                .expect("error がオブジェクトでない")
                .keys()
                .count();
            if expect_code.is_some() {
                // 429 の既存契約 (ratelimit_931) が code / param を読む。
                assert_eq!(keys, 4, "{label}/Anthropic: 429 のキー数: {v}");
                assert_eq!(v["error"]["code"], expect_code.unwrap(), "{v}");
                assert!(v["error"]["param"].is_null(), "{v}");
            } else {
                assert_eq!(keys, 2, "{label}/Anthropic: キーは type/message だけ: {v}");
            }
            msg.to_string()
        }
        Proto::OpenAi => {
            assert!(
                v.get("type").is_none(),
                "{label}/OpenAI: top-level type を出してはいけない: {v}"
            );
            assert_eq!(v["error"]["type"], expect_type, "{label}/OpenAI: {v}");
            let msg = v["error"]["message"].as_str().unwrap_or("");
            assert!(!msg.is_empty(), "{label}/OpenAI: message が空: {v}");
            assert!(
                v["error"].get("param").is_some(),
                "{label}/OpenAI: param キーが無い: {v}"
            );
            assert!(v["error"]["param"].is_null(), "{v}");
            assert!(
                v["error"].get("code").is_some(),
                "{label}/OpenAI: code キーが無い: {v}"
            );
            match expect_code {
                Some(c) => assert_eq!(v["error"]["code"], c, "{v}"),
                None => assert!(v["error"]["code"].is_null(), "{v}"),
            }
            let keys = v.as_object().expect("封筒がオブジェクトでない").keys().count();
            assert_eq!(keys, 1, "{label}/OpenAI: 封筒は error 1 キー: {v}");
            msg.to_string()
        }
        Proto::Ollama => {
            let msg = v["error"].as_str().unwrap_or("");
            assert!(!msg.is_empty(), "{label}/Ollama: error が非空文字列でない: {v}");
            let keys = v.as_object().expect("封筒がオブジェクトでない").keys().count();
            assert_eq!(keys, 1, "{label}/Ollama: 封筒は error 1 キー: {v}");
            msg.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// 1. 429: レート制限 (共通ミドルウェア)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limit_429_envelope_is_protocol_native() {
    for proto in PROTOCOLS {
        // バースト 1 / 回復 1 rpm: 2 本目は必ず 429。
        let limiter = RateLimiter::from_config(RateLimitConfig::new(1, Some(1)));
        let app = app_with(
            fast_state(),
            limiter,
            RequestTimeouts::disabled(),
        );

        let (s, _, t) = send(&app, post_json(proto.uri(), &proto.valid_body())).await;
        assert_eq!(s, StatusCode::OK, "{proto:?}: 1 本目が通らない: {t}");

        let (s, h, t) = send(&app, post_json(proto.uri(), &proto.valid_body())).await;
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{proto:?}: {t}");
        assert!(retry_after(&h).is_some(), "{proto:?}: Retry-After が無い");
        let msg = assert_envelope(
            "429",
            proto,
            s,
            &h,
            &t,
            "rate_limit_error",
            Some("rate_limit_exceeded"),
        );
        assert!(msg.contains("Rate limit"), "{proto:?}: 理由が無い: {msg}");
    }
}

// ---------------------------------------------------------------------------
// 2. 503: drain 中の新規要求 (ハンドラがプロトコル別に返す)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn draining_503_envelope_is_protocol_native() {
    for proto in PROTOCOLS {
        let state = fast_state();
        let app = app_with(state.clone(), RateLimiter::disabled(), RequestTimeouts::disabled());
        state.scheduler.lifecycle().begin_drain();

        let (s, h, t) = send(&app, post_json(proto.uri(), &proto.valid_body())).await;
        assert_eq!(s, StatusCode::SERVICE_UNAVAILABLE, "{proto:?}: {t}");
        assert!(
            retry_after(&h).is_none(),
            "{proto:?}: drain の 503 に Retry-After は付けない"
        );
        // Anthropic は overloaded_error、OpenAI は server_error に寄る。
        let expect_type = match proto {
            Proto::Anthropic => "overloaded_error",
            Proto::OpenAi => "server_error",
            Proto::Ollama => "",
        };
        let msg = assert_envelope("503", proto, s, &h, &t, expect_type, None);
        assert!(msg.contains("draining"), "{proto:?}: 理由が無い: {msg}");
    }
}

// ---------------------------------------------------------------------------
// 3. 504: ハンドラ期限 (共通ミドルウェア)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn gateway_timeout_504_envelope_is_protocol_native() {
    // 1 forward 300ms に対し期限 30ms: どの経路も必ず期限を超える。
    let timeouts = RequestTimeouts {
        read_body: None,
        handler: Some(Duration::from_millis(30)),
    };
    for proto in PROTOCOLS {
        let state = state_with(Box::new(SlowModel { delay: Duration::from_millis(300) }));
        let app = app_with(state, RateLimiter::disabled(), timeouts);

        let (s, h, t) = send(&app, post_json(proto.uri(), &proto.valid_body())).await;
        assert_eq!(s, StatusCode::GATEWAY_TIMEOUT, "{proto:?}: {t}");
        let expect_type = match proto {
            Proto::Anthropic => "api_error",
            Proto::OpenAi => "server_error",
            Proto::Ollama => "",
        };
        let msg = assert_envelope("504", proto, s, &h, &t, expect_type, None);
        assert!(
            msg.contains("gateway timeout"),
            "{proto:?}: 理由が無い: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. 408: ボディ読み込み期限 (共通ミドルウェア)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn request_timeout_408_envelope_is_protocol_native() {
    // ボディは 300ms 後にしか流れない。期限は 30ms。
    let timeouts = RequestTimeouts {
        read_body: Some(Duration::from_millis(30)),
        handler: None,
    };
    for proto in PROTOCOLS {
        let app = app_with(fast_state(), RateLimiter::disabled(), timeouts);
        let body = serde_json::to_vec(&proto.valid_body()).unwrap();

        let req = Request::builder()
            .method("POST")
            .uri(proto.uri())
            .header(header::CONTENT_TYPE, "application/json")
            .body(slow_body(Duration::from_millis(300), body))
            .unwrap();
        let (s, h, t) = send(&app, req).await;
        assert_eq!(s, StatusCode::REQUEST_TIMEOUT, "{proto:?}: {t}");
        let expect_type = match proto {
            Proto::Anthropic => "api_error",
            Proto::OpenAi => "server_error",
            Proto::Ollama => "",
        };
        let msg = assert_envelope("408", proto, s, &h, &t, expect_type, None);
        assert!(
            msg.contains("request timeout"),
            "{proto:?}: 理由が無い: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. 期限を有効にしても、期限内に終わる要求は従来どおり 200
// ---------------------------------------------------------------------------

#[tokio::test]
async fn enabled_timeouts_do_not_disturb_fast_requests() {
    let timeouts = RequestTimeouts {
        read_body: Some(Duration::from_secs(5)),
        handler: Some(Duration::from_secs(30)),
    };
    for proto in PROTOCOLS {
        let app = app_with(fast_state(), RateLimiter::disabled(), timeouts);
        let (s, _, t) = send(&app, post_json(proto.uri(), &proto.valid_body())).await;
        assert_eq!(s, StatusCode::OK, "{proto:?}: 期限有効で 200 が壊れた: {t}");
    }
}
