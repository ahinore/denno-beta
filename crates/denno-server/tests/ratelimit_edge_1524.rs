//! task 1524: トークンバケットのバースト消費・補充と、429 封筒のプロトコル
//! 横断パリティ (OpenAI / Anthropic / Ollama) を突く追加テスト。
//!
//! - バケット単体は時刻を [`Instant`] で注入するので実時間に依存しない。
//! - ルータ経由のテストは `rpm` を小さくして補充量を無視できるようにし、
//!   `burst` 数を超えた回数だけが 429 になる性質に依存する。ミリ秒 sleep も
//!   スレッド競合も使わないので 4 コア CI でもフレークしない。
//!
//! task 931 (`ratelimit_931.rs`) が基本挙動、task 1139
//! (`ratelimit_edge_1139.rs`) がバースト境界と Retry-After 書式を見ているので、
//! ここでは「3 プロトコルで 429 封筒が揃うこと」と「拒否されたリクエストの
//! パスで封筒が決まること」を主眼に置く。

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::{
    Decision, RateLimitConfig, RateLimiter, TokenBucket, RATE_LIMIT_BURST_ENV,
    RATE_LIMIT_ERROR_CODE, RATE_LIMIT_ERROR_TYPE, RATE_LIMIT_INFERENCE_PATHS, RATE_LIMIT_RPM_ENV,
};
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// 共通ヘルパ
// ---------------------------------------------------------------------------

fn state() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
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

fn app_with(rpm: u32, burst: Option<u32>) -> Router {
    let limiter = RateLimiter::from_config(RateLimitConfig::new(rpm, burst));
    denno_server::router_with_rate_limit(state(), CorsConfig::default(), limiter)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn hdr(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn content_type(headers: &HeaderMap) -> String {
    hdr(headers, "content-type")
        .unwrap_or_default()
        .to_ascii_lowercase()
}

fn retry_after(headers: &HeaderMap) -> Option<String> {
    hdr(headers, "retry-after")
}

fn post_json(uri: &str, body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

// ---------------------------------------------------------------------------
// プロトコル定義 (封筒の期待形)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Proto {
    Anthropic,
    OpenAi,
    Ollama,
}

const PROTOCOLS: [Proto; 3] = [Proto::Anthropic, Proto::OpenAi, Proto::Ollama];

impl Proto {
    fn label(self) -> &'static str {
        match self {
            Proto::Anthropic => "anthropic",
            Proto::OpenAi => "openai",
            Proto::Ollama => "ollama",
        }
    }

    fn uri(self) -> &'static str {
        match self {
            Proto::Anthropic => "/v1/messages",
            Proto::OpenAi => "/v1/chat/completions",
            Proto::Ollama => "/api/chat",
        }
    }

    /// ダミーモデルで完走する最小の有効リクエスト。
    fn body(self) -> Value {
        match self {
            Proto::Anthropic => json!({
                "model": "denno-test",
                "max_tokens": 2,
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
            }),
            Proto::OpenAi => json!({
                "model": "denno-test",
                "max_tokens": 2,
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
            }),
            Proto::Ollama => json!({
                "model": "denno-test",
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
            }),
        }
    }

    fn request(self) -> Request<Body> {
        post_json(self.uri(), self.body())
    }
}

/// パスから封筒の系統を決める (`protocol_for_path` と同じ規則)。
fn proto_for_path(path: &str) -> Proto {
    if path == "/api" || path.starts_with("/api/") {
        Proto::Ollama
    } else if path == "/v1/messages" || path.starts_with("/v1/messages/") {
        Proto::Anthropic
    } else {
        Proto::OpenAi
    }
}

/// パスごとに受理されうるリクエスト本文。429 判定はルートより手前で起きるので
/// 本文の正しさには依存しないが、1 回目をなるべく 2xx にするために整える。
fn body_for_path(path: &str) -> Value {
    match path {
        "/api/generate" => json!({
            "model": "denno-test",
            "prompt": "t0",
            "stream": false,
        }),
        "/api/chat" => json!({
            "model": "denno-test",
            "stream": false,
            "messages": [{"role": "user", "content": "t0"}],
        }),
        "/v1/messages/count_tokens" => json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "t0"}],
        }),
        "/v1/completions" => json!({
            "model": "denno-test",
            "prompt": "t0",
            "max_tokens": 2,
            "stream": false,
        }),
        _ => json!({
            "model": "denno-test",
            "max_tokens": 2,
            "stream": false,
            "messages": [{"role": "user", "content": "t0"}],
        }),
    }
}

/// 429 封筒がそのプロトコルの流儀に沿っていることを検査し、message を返す。
///
/// - Anthropic … `{"type":"error","error":{"type","message","code","param"}}`
/// - OpenAI    … `{"error":{"message","type","param":null,"code"}}`
/// - Ollama    … `{"error":"<message>"}`
fn assert_429_envelope(proto: Proto, headers: &HeaderMap, text: &str) -> String {
    let ct = content_type(headers);
    assert!(
        ct.starts_with("application/json"),
        "{}: 429 の Content-Type が JSON でない: {ct:?}",
        proto.label()
    );

    // Retry-After は「1 個だけ」「単位なしの整数秒」「1 以上」。
    let all: Vec<&str> = headers
        .get_all(header::RETRY_AFTER)
        .iter()
        .map(|v| v.to_str().expect("Retry-After は ASCII"))
        .collect();
    assert_eq!(
        all.len(),
        1,
        "{}: Retry-After はちょうど 1 個: {all:?}",
        proto.label()
    );
    let raw = all[0];
    assert!(
        !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit()),
        "{}: Retry-After は単位なしの整数秒: {raw:?}",
        proto.label()
    );
    let secs: u64 = raw.parse().expect("Retry-After は数値");
    assert!(secs >= 1, "{}: Retry-After = {secs}", proto.label());

    let v: Value = serde_json::from_str(text)
        .unwrap_or_else(|e| panic!("{}: 429 が JSON でない ({e}): {text}", proto.label()));

    match proto {
        Proto::Anthropic => {
            assert_eq!(v["type"], "error", "anthropic 外枠: {v}");
            assert_eq!(
                v["error"]["type"], RATE_LIMIT_ERROR_TYPE,
                "anthropic error.type: {v}"
            );
            assert_eq!(
                v["error"]["code"], RATE_LIMIT_ERROR_CODE,
                "anthropic error.code: {v}"
            );
            assert!(
                v["error"].get("param").is_some() && v["error"]["param"].is_null(),
                "anthropic error.param は null で存在する: {v}"
            );
            // 外枠の `type` は "error"、内側だけが rate_limit_error。
            assert!(
                v.get("code").is_none(),
                "anthropic は code を外枠に置かない: {v}"
            );
        }
        Proto::OpenAi => {
            assert!(
                v["error"].is_object(),
                "openai は error オブジェクト: {v}"
            );
            assert_eq!(
                v["error"]["type"], RATE_LIMIT_ERROR_TYPE,
                "openai error.type: {v}"
            );
            assert_eq!(
                v["error"]["code"], RATE_LIMIT_ERROR_CODE,
                "openai error.code: {v}"
            );
            assert!(
                v["error"].get("param").is_some() && v["error"]["param"].is_null(),
                "openai error.param は null で存在する: {v}"
            );
            assert!(
                v.get("type").is_none(),
                "openai は外枠に type を置かない: {v}"
            );
        }
        Proto::Ollama => {
            assert!(
                v["error"].is_string(),
                "ollama は error を文字列で返す: {v}"
            );
            assert!(
                v.get("type").is_none(),
                "ollama は外枠に type を置かない: {v}"
            );
        }
    }

    let message = match proto {
        Proto::Anthropic | Proto::OpenAi => v["error"]["message"]
            .as_str()
            .unwrap_or_else(|| panic!("{}: message が文字列でない: {v}", proto.label()))
            .to_string(),
        Proto::Ollama => v["error"].as_str().unwrap().to_string(),
    };
    assert!(
        message.contains("Rate limit exceeded"),
        "{}: message にレート制限の説明が無い: {message:?}",
        proto.label()
    );
    assert!(
        message.contains(RATE_LIMIT_RPM_ENV),
        "{}: message に env 名が無い: {message:?}",
        proto.label()
    );
    assert!(
        message.contains(&format!("retry after {secs}s")),
        "{}: message の秒数がヘッダと一致しない: {message:?} / {secs}",
        proto.label()
    );
    message
}

// ---------------------------------------------------------------------------
// バケット単体: バースト消費 (注入時刻なので決定論的)
// ---------------------------------------------------------------------------

#[test]
fn burst_consumption_is_exact_at_a_fixed_instant() {
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(4)), t0).expect("有効な設定");
    assert!((b.available_tokens(t0) - 4.0).abs() < 1e-9, "初期値はバースト数");

    for i in 0..4u32 {
        assert!(b.try_acquire_at(t0).is_ok(), "{i} 個目は通る");
        let left = (3 - i) as f64;
        assert!(
            (b.available_tokens(t0) - left).abs() < 1e-9,
            "取得のたびに 1 個ずつ減る: 期待 {left}"
        );
    }
    assert!(b.available_tokens(t0).abs() < 1e-9, "使い切りで 0");

    // 拒否はトークンを負にしない (何度叩いても同じ)。
    for _ in 0..8 {
        assert_eq!(b.try_acquire_at(t0), Err(1), "バースト超過は拒否");
        assert!(b.available_tokens(t0).abs() < 1e-9, "0 未満にならない");
    }
}

#[test]
fn refill_is_linear_at_injected_instants_and_caps_at_burst() {
    let t0 = Instant::now();
    // 120 rpm = 2 token/sec、バースト 3。
    let b = TokenBucket::new_at(RateLimitConfig::new(120, Some(3)), t0).expect("有効な設定");
    for _ in 0..3 {
        assert!(b.try_acquire_at(t0).is_ok());
    }

    // +250ms → 0.5 token。1 個には足りない (切り上げで 1 秒)。
    let t1 = t0 + Duration::from_millis(250);
    assert!((b.available_tokens(t1) - 0.5).abs() < 1e-9, "0.25 秒 x 2/秒");
    assert_eq!(b.try_acquire_at(t1), Err(1), "端数では通さない");

    // +500ms → ちょうど 1 token。1 回だけ通る。
    let t2 = t0 + Duration::from_millis(500);
    assert!((b.available_tokens(t2) - 1.0).abs() < 1e-9);
    assert!(b.try_acquire_at(t2).is_ok(), "1 token 回復したので通る");
    assert_eq!(b.try_acquire_at(t2), Err(1), "直後の 2 個目は拒否");

    // +2s → 3 token ぶん回復するが容量 3 で頭打ち。
    let t3 = t0 + Duration::from_secs(2);
    assert!((b.available_tokens(t3) - 3.0).abs() < 1e-9, "容量で頭打ち");
    assert!((b.available_tokens(t0 + Duration::from_secs(3600)) - 3.0).abs() < 1e-9);
}

#[test]
fn retry_after_reflects_the_deficit_and_rounds_up() {
    let t0 = Instant::now();

    // 30 rpm = 0.5 token/sec、バースト 1。不足 1 token は 2 秒。
    let b = TokenBucket::new_at(RateLimitConfig::new(30, Some(1)), t0).expect("有効な設定");
    assert!(b.try_acquire_at(t0).is_ok());
    assert_eq!(b.try_acquire_at(t0), Err(2), "1 / 0.5 = 2 秒");
    // +1 秒で 0.5 token。残不足 0.5 / 0.5 = 1 秒 (満タン待ちの 2 秒ではない)。
    assert_eq!(
        b.try_acquire_at(t0 + Duration::from_secs(1)),
        Err(1),
        "残不足から計算する"
    );
    // +2 秒で 1 token ちょうど。
    assert!(b.try_acquire_at(t0 + Duration::from_secs(2)).is_ok());
    assert_eq!(b.try_acquire_at(t0 + Duration::from_secs(2)), Err(2));

    // 60 rpm = 1 token/sec なら不足 1 token は 1 秒。
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(1)), t0).expect("有効な設定");
    assert!(b.try_acquire_at(t0).is_ok());
    assert_eq!(b.try_acquire_at(t0), Err(1));
    assert_eq!(b.try_acquire_at(t0 + Duration::from_millis(999)), Err(1));
}

#[test]
fn repeated_denials_at_one_instant_are_stable() {
    let t0 = Instant::now();
    let b = TokenBucket::new_at(RateLimitConfig::new(60, Some(1)), t0).expect("有効な設定");
    assert!(b.try_acquire_at(t0).is_ok());
    for _ in 0..10 {
        assert_eq!(b.try_acquire_at(t0), Err(1), "同じ時刻では同じ答え");
        assert!(b.available_tokens(t0).abs() < 1e-9, "拒否は状態を変えない");
    }
}

#[test]
fn disabled_limiter_allows_every_path_without_state() {
    let limiter = RateLimiter::from_config(RateLimitConfig::DISABLED);
    assert!(!limiter.is_enabled());
    assert!(TokenBucket::new(RateLimitConfig::DISABLED).is_none());
    let t0 = Instant::now();
    for _ in 0..100 {
        for p in RATE_LIMIT_INFERENCE_PATHS {
            assert_eq!(
                limiter.check_at(p, t0),
                Decision::Allow,
                "無効時は {p} も素通し"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ルータ: 429 封筒のプロトコル横断パリティ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn burst_is_consumed_exactly_per_protocol_route() {
    for proto in PROTOCOLS {
        // 1 rpm = 1/60 token/sec。テスト中の補充は無視できる。
        let app = app_with(1, Some(3));
        for i in 0..3 {
            let (s, h, text) = send(&app, proto.request()).await;
            assert_eq!(
                s,
                StatusCode::OK,
                "{}: {i} 個目 (バースト内) が 429 になった: {text}",
                proto.label()
            );
            assert!(
                retry_after(&h).is_none(),
                "{}: 成功応答に Retry-After を付けない",
                proto.label()
            );
        }
        let (s, _, text) = send(&app, proto.request()).await;
        assert_eq!(
            s,
            StatusCode::TOO_MANY_REQUESTS,
            "{}: 4 個目 (バースト超過) が 429 でない: {text}",
            proto.label()
        );
    }
}

#[tokio::test]
async fn forty_two_nine_envelope_parity_across_protocols() {
    // 30 rpm = 0.5 token/sec、バースト 1 → 2 個目は不足 1 token = 2 秒。
    // 実時間の経過は数 ms なので Retry-After はどの経路でも 2 になる。
    let mut messages = Vec::new();
    let mut secs_seen = Vec::new();

    for proto in PROTOCOLS {
        let app = app_with(30, Some(1));
        let (s, _, text) = send(&app, proto.request()).await;
        assert_eq!(
            s,
            StatusCode::OK,
            "{}: 1 個目は通る: {text}",
            proto.label()
        );

        let (s, h, text) = send(&app, proto.request()).await;
        assert_eq!(
            s,
            StatusCode::TOO_MANY_REQUESTS,
            "{}: 2 個目は 429: {text}",
            proto.label()
        );
        let message = assert_429_envelope(proto, &h, &text);

        let secs: u64 = retry_after(&h).unwrap().parse().unwrap();
        assert_eq!(
            secs, 2,
            "{}: 不足 1 token / 0.5 per sec = 2 秒",
            proto.label()
        );
        assert!(
            hdr(&h, "x-request-id").is_some(),
            "{}: 429 もログ層を通り x-request-id が付く",
            proto.label()
        );
        messages.push(message);
        secs_seen.push(secs);
    }

    // 3 経路で同じ条件なら message も同じ (プロトコルの外枠だけが違う)。
    assert_eq!(
        messages[0], messages[1],
        "anthropic と openai の message が一致しない: {messages:?}"
    );
    assert_eq!(
        messages[1], messages[2],
        "openai と ollama の message が一致しない: {messages:?}"
    );
    assert!(
        secs_seen.windows(2).all(|w| w[0] == w[1]),
        "Retry-After が経路で揺れる: {secs_seen:?}"
    );
}

#[tokio::test]
async fn shared_bucket_denial_uses_the_denied_routes_protocol() {
    // 3 プロトコルは同じ 1 個のバケットを共有する。バースト 2 を使い切った後、
    // 3 個目 (Anthropic) は Anthropic 封筒で 429、4 個目 (Ollama) は Ollama 封筒。
    let app = app_with(1, Some(2));
    let (s, _, _) = send(&app, Proto::OpenAi.request()).await;
    assert_eq!(s, StatusCode::OK);
    let (s, _, _) = send(&app, Proto::Ollama.request()).await;
    assert_eq!(s, StatusCode::OK, "バースト 2 個目は別パスでも通る");

    let (s, h, text) = send(&app, Proto::Anthropic.request()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_429_envelope(Proto::Anthropic, &h, &text);
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap()["type"],
        "error",
        "拒否されたリクエストのパス (Anthropic) の外枠を使う"
    );

    let (s, h, text) = send(&app, Proto::Ollama.request()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_429_envelope(Proto::Ollama, &h, &text);
    let v: Value = serde_json::from_str(&text).unwrap();
    assert!(
        v["error"].is_string(),
        "拒否されたリクエストのパス (Ollama) の封筒を使う: {v}"
    );

    // OpenAI 経路を拒否した場合も OpenAI 封筒。
    let (s, h, text) = send(&app, Proto::OpenAi.request()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
    assert_429_envelope(Proto::OpenAi, &h, &text);
}

#[tokio::test]
async fn every_inference_path_denies_with_its_protocol_family() {
    for path in RATE_LIMIT_INFERENCE_PATHS {
        let app = app_with(1, Some(1));
        let (s, _, text) = send(&app, post_json(path, body_for_path(path))).await;
        assert_ne!(
            s,
            StatusCode::TOO_MANY_REQUESTS,
            "{path}: 1 個目が 429 になった: {text}"
        );

        let (s, h, text) = send(&app, post_json(path, body_for_path(path))).await;
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS, "{path}: 2 個目は 429: {text}");
        let proto = proto_for_path(path);
        assert_429_envelope(proto, &h, &text);
    }
}

#[tokio::test]
async fn trailing_slash_inference_path_is_also_limited() {
    for path in ["/v1/messages/", "/v1/chat/completions/", "/api/chat/"] {
        let app = app_with(1, Some(1));
        let body = body_for_path(path.trim_end_matches('/'));
        let (s, _, text) = send(&app, post_json(path, body.clone())).await;
        assert_ne!(
            s,
            StatusCode::TOO_MANY_REQUESTS,
            "{path}: 1 個目は 429 にしない: {text}"
        );
        let (s, _, text) = send(&app, post_json(path, body)).await;
        assert_eq!(
            s,
            StatusCode::TOO_MANY_REQUESTS,
            "{path}: 末尾スラッシュでも同じバケットを消費する: {text}"
        );
    }
}

#[tokio::test]
async fn retry_after_is_never_absent_or_zero_on_429() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, Proto::Anthropic.request()).await;
    assert_eq!(s, StatusCode::OK);
    for _ in 0..5 {
        let (s, h, _) = send(&app, Proto::Anthropic.request()).await;
        assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
        let raw = retry_after(&h).expect("429 に Retry-After が無い");
        let secs: u64 = raw.parse().expect("Retry-After は数値");
        assert!(secs >= 1, "Retry-After は 0 にしない: {raw}");
        assert!(
            h.get_all(header::RETRY_AFTER).iter().count() == 1,
            "Retry-After を重複させない"
        );
    }
}

#[tokio::test]
async fn non_inference_and_bypass_paths_do_not_get_429_when_exhausted() {
    let app = app_with(1, Some(1));
    let (s, _, _) = send(&app, Proto::Anthropic.request()).await;
    assert_eq!(s, StatusCode::OK);

    // 監視系は素通し、非推論の /v1/models も対象外。
    for uri in ["/health", "/metrics", "/v1/models"] {
        let req = Request::builder()
            .method("GET")
            .uri(uri)
            .body(Body::empty())
            .unwrap();
        let (s, h, _) = send(&app, req).await;
        assert_ne!(s, StatusCode::TOO_MANY_REQUESTS, "{uri} は 429 にしない");
        assert!(retry_after(&h).is_none(), "{uri} に Retry-After を付けない");
    }

    // バケットは空のまま。
    let (s, _, _) = send(&app, Proto::Anthropic.request()).await;
    assert_eq!(s, StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn error_constants_used_by_the_envelopes_are_stable() {
    // 封筒のキー値が変わるとクライアント互換が壊れるので固定する。
    assert_eq!(RATE_LIMIT_ERROR_TYPE, "rate_limit_error");
    assert_eq!(RATE_LIMIT_ERROR_CODE, "rate_limit_exceeded");
    assert_eq!(RATE_LIMIT_RPM_ENV, "DENNO_RATE_LIMIT_RPM");
    assert_eq!(RATE_LIMIT_BURST_ENV, "DENNO_RATE_LIMIT_BURST");
}
