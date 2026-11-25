//! task 1154: ボディ上限 (413) の境界を全プロトコル横断で固定する。
//!
//! task 946 の `body_limit_946.rs` は `/v1/messages` を中心に
//! 「上限内 / 上限超過 / 封筒の形」を確かめている。ここでは
//! `/v1/messages` / `/v1/chat/completions` / `/api/chat` / `/api/generate`
//! の 4 経路すべてについて、**境界 (ちょうど上限 / 1 byte 超過)** の
//! 受理・拒否が同じ規則で揃っていることを固定する:
//!
//! - ボディ長 == 上限 は受理される (413 にしない)
//! - ボディ長 == 上限 + 1 は 413 で、JSON 封筒に実効 byte 上限を含む
//! - `Content-Length` 付き (早期判定) と無し (`DefaultBodyLimit` が
//!   読み込み中に打ち切る経路) の両方で同じ封筒になる
//! - `Content-Length` だけが上限を超える (実際のボディは小さい) 場合も
//!   早期に 413 になる
//! - 空ボディは 413 ではない (各ハンドラの通常の 4xx に委ねる)

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::RateLimiter;
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

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

/// ボディ上限を明示したルーター (レート制限は無効)。
fn app_with_body_limit(max_body_bytes: usize) -> Router {
    denno_server::router_with_body_limit(
        state(),
        CorsConfig::default(),
        RateLimiter::disabled(),
        max_body_bytes,
    )
}

/// 対象の 4 プロトコル経路。
#[derive(Clone, Copy, Debug)]
enum Route {
    Messages,
    ChatCompletions,
    OllamaChat,
    OllamaGenerate,
}

const ROUTES: [Route; 4] = [
    Route::Messages,
    Route::ChatCompletions,
    Route::OllamaChat,
    Route::OllamaGenerate,
];

impl Route {
    fn uri(self) -> &'static str {
        match self {
            Route::Messages => "/v1/messages",
            Route::ChatCompletions => "/v1/chat/completions",
            Route::OllamaChat => "/api/chat",
            Route::OllamaGenerate => "/api/generate",
        }
    }

    /// 各経路の最小の有効リクエスト。`pad` は未知フィールドで埋めるので
    /// デシリアライズには影響しない (serde は未知フィールドを無視する)。
    fn body_json(self, pad: usize) -> Value {
        let padding = "a".repeat(pad);
        match self {
            Route::Messages => json!({
                "model": "denno-test",
                "messages": [{"role": "user", "content": "t0"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "zz_pad_1154": padding,
            }),
            Route::ChatCompletions => json!({
                "model": "denno-test",
                "messages": [{"role": "user", "content": "t0"}],
                "max_tokens": 4,
                "temperature": 0.0,
                "zz_pad_1154": padding,
            }),
            Route::OllamaChat => json!({
                "model": "denno-test",
                "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
                "options": {"temperature": 0.0},
                "zz_pad_1154": padding,
            }),
            Route::OllamaGenerate => json!({
                "model": "denno-test",
                "stream": false,
                "prompt": "t0",
                "options": {"temperature": 0.0},
                "zz_pad_1154": padding,
            }),
        }
    }

    fn body_with_pad(self, pad: usize) -> Vec<u8> {
        serde_json::to_vec(&self.body_json(pad)).unwrap()
    }

    /// シリアライズ長がちょうど `target` byte になるボディ。
    fn exact_body(self, target: usize) -> Vec<u8> {
        let mut pad = target;
        loop {
            let body = self.body_with_pad(pad);
            match body.len().cmp(&target) {
                std::cmp::Ordering::Equal => return body,
                std::cmp::Ordering::Less => pad += target - body.len(),
                std::cmp::Ordering::Greater => pad -= body.len() - target,
            }
        }
    }
}

fn post(uri: &str, body: Vec<u8>, content_length: Option<usize>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(n) = content_length {
        b = b.header(header::CONTENT_LENGTH, n);
    }
    b.body(Body::from(body)).unwrap()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, String, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
    let text = String::from_utf8_lossy(&bytes).to_string();
    (status, ct, text)
}

fn assert_json_413(status: StatusCode, ct: &str, text: &str, limit: usize, route: &str) {
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{route}: body = {text}");
    assert!(
        ct.to_ascii_lowercase().starts_with("application/json"),
        "{route}: content-type = {ct:?} (JSON 封筒でない): {text}"
    );
    let v: Value = serde_json::from_str(text).unwrap_or_else(|e| {
        panic!("{route}: 413 のボディが JSON でない ({e}): {text}")
    });
    assert_eq!(v["type"], "error", "{route}: 封筒の形が違う: {v}");
    assert_eq!(
        v["error"]["type"], "invalid_request_error",
        "{route}: error.type が違う: {v}"
    );
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains(&limit.to_string()),
        "{route}: メッセージに上限 ({limit}) が無い: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 境界: ちょうど上限はすべての経路で受理される
// ---------------------------------------------------------------------------

#[tokio::test]
async fn exact_limit_is_accepted_on_all_protocols() {
    let limit = 4096;
    let app = app_with_body_limit(limit);
    for route in ROUTES {
        let body = route.exact_body(limit);
        assert_eq!(body.len(), limit, "{route:?}");

        // Content-Length 付き
        let (status, _ct, text) = send(&app, post(route.uri(), body.clone(), Some(limit))).await;
        assert_ne!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{route:?} (CL 有り): ちょうど上限は拒否しない: {text}"
        );
        assert_eq!(status, StatusCode::OK, "{route:?}: body = {text}");

        // Content-Length 無し (chunked 相当)
        let (status, _ct, text) = send(&app, post(route.uri(), body, None)).await;
        assert_ne!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{route:?} (CL 無し): ちょうど上限は拒否しない: {text}"
        );
        assert_eq!(status, StatusCode::OK, "{route:?}: body = {text}");
    }
}

// ---------------------------------------------------------------------------
// 境界: 1 byte 超過はすべての経路で JSON 413
// ---------------------------------------------------------------------------

#[tokio::test]
async fn one_byte_over_limit_is_json_413_on_all_protocols() {
    let limit = 4096;
    let app = app_with_body_limit(limit);
    for route in ROUTES {
        let mut body = route.exact_body(limit);
        body.push(b' ');
        assert_eq!(body.len(), limit + 1, "{route:?}");
        let (status, ct, text) = send(&app, post(route.uri(), body, Some(limit + 1))).await;
        assert_json_413(status, &ct, &text, limit, route.uri());
    }
}

/// `Content-Length` を持たない経路 (`DefaultBodyLimit` が読み込み中に
/// 打ち切る) でも、同じ JSON 封筒になること。
#[tokio::test]
async fn over_limit_without_content_length_is_json_413_on_all_protocols() {
    let limit = 8192;
    let app = app_with_body_limit(limit);
    for route in ROUTES {
        let body = route.body_with_pad(256 * 1024);
        assert!(body.len() > limit);
        let (status, ct, text) = send(&app, post(route.uri(), body, None)).await;
        assert_json_413(status, &ct, &text, limit, route.uri());
    }
}

// ---------------------------------------------------------------------------
// Content-Length と実ボディ長の食い違い
// ---------------------------------------------------------------------------

/// `Content-Length` だけが上限を超える (実ボディは小さい) 場合は、
/// 読み込み前に早期 413 になる。
#[tokio::test]
async fn declared_length_over_limit_is_413_before_read() {
    let limit = 4096;
    let app = app_with_body_limit(limit);
    for route in ROUTES {
        let body = route.body_with_pad(0);
        assert!(body.len() < limit, "{route:?}");
        let (status, ct, text) = send(&app, post(route.uri(), body, Some(limit + 1))).await;
        assert_json_413(status, &ct, &text, limit, route.uri());
    }
}

/// `Content-Length` は上限ちょうどだが実ボディが 1 byte 超える場合も、
/// 読み込み中の打ち切りで 413 + JSON 封筒になる。
#[tokio::test]
async fn understated_content_length_still_413_on_all_protocols() {
    let limit = 4096;
    let app = app_with_body_limit(limit);
    for route in ROUTES {
        let mut body = route.exact_body(limit);
        body.push(b' ');
        // 宣言は上限ちょうど (= 早期判定は通す)、実ボディは +1 byte。
        let (status, ct, text) = send(&app, post(route.uri(), body, Some(limit))).await;
        assert_json_413(status, &ct, &text, limit, route.uri());
    }
}

// ---------------------------------------------------------------------------
// 空ボディは 413 ではない
// ---------------------------------------------------------------------------

/// 空ボディはボディ上限の対象外。各ハンドラの通常の 4xx に委ねる
/// (413 にすり替えないことだけを固定する)。
#[tokio::test]
async fn empty_body_is_not_413_on_all_protocols() {
    let limit = 4096;
    let app = app_with_body_limit(limit);
    for route in ROUTES {
        let (status, _ct, text) = send(&app, post(route.uri(), Vec::new(), Some(0))).await;
        assert_ne!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE,
            "{route:?}: 空ボディを 413 にしてはいけない: {text}"
        );
    }
}
