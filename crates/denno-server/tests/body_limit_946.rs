//! task 946: `DENNO_MAX_BODY_MB` / `--max-body-mb` によるリクエストボディ上限と
//! 413 Payload Too Large の JSON エラー封筒の統合テスト。
//!
//! - 上限内のボディは通常どおり処理される
//! - 上限超過は HTTP 413 を `application/json` の
//!   `{"type":"error","error":{"type":...,"message":...}}` で返す
//! - `Content-Length` がある経路 (早期判定) と無い経路
//!   (`DefaultBodyLimit` が読み込み中に打ち切る) の両方で同じ封筒になる

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::ratelimit::RateLimiter;
use denno_server::{
    resolve_max_body_mb, AppState, CorsConfig, EngineInfo, Scheduler, DEFAULT_MAX_BODY_MB,
    MAX_BODY_MB_ENV, MAX_BODY_MB_LIMIT,
};
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

fn messages_body(padding_bytes: usize) -> Vec<u8> {
    let mut v = json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
    });
    if padding_bytes > 0 {
        v["metadata"] = json!({"pad": "a".repeat(padding_bytes)});
    }
    serde_json::to_vec(&v).unwrap()
}

fn post(body: Vec<u8>, content_length: Option<usize>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(n) = content_length {
        b = b.header(header::CONTENT_LENGTH, n);
    }
    b.body(Body::from(body)).unwrap()
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, String, String, Vec<u8>) {
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
    (status, ct, text, bytes)
}

fn assert_json_413(status: StatusCode, ct: &str, text: &str, limit: usize) {
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "body = {text}");
    assert!(
        ct.to_ascii_lowercase().starts_with("application/json"),
        "content-type = {ct:?} (JSON 封筒でない): {text}"
    );
    let v: Value = serde_json::from_str(text).expect("413 のボディが JSON でない");
    assert_eq!(v["type"], "error", "封筒の形が違う: {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains(&limit.to_string()),
        "メッセージに上限 ({limit}) が無い: {msg}"
    );
}

// ---------------------------------------------------------------------------
// 上限内
// ---------------------------------------------------------------------------

#[tokio::test]
async fn under_limit_body_is_accepted() {
    let limit = 1024 * 1024;
    let app = app_with_body_limit(limit);
    // 長めのプロンプト相当 (数十 KB) は余裕で通る
    let body = messages_body(64 * 1024);
    let (status, _ct, text, _) = send(&app, post(body, None)).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    let v: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["type"], "message", "Anthropic 形式でない: {v}");
}

/// 上限ちょうど (= 境界) は拒否しない。境界の判定は「大きい方だけ」。
#[tokio::test]
async fn body_exactly_at_limit_is_accepted() {
    let limit = 4096;
    let app = app_with_body_limit(limit);
    // padding を調整して JSON の長さを limit に合わせる
    let mut pad = limit;
    let body = loop {
        let body = messages_body(pad);
        match body.len().cmp(&limit) {
            std::cmp::Ordering::Equal => break body,
            std::cmp::Ordering::Less => pad += limit - body.len(),
            std::cmp::Ordering::Greater => pad -= body.len() - limit,
        }
    };
    assert_eq!(body.len(), limit);
    let (status, _ct, text, _) = send(&app, post(body.clone(), Some(limit))).await;
    assert_ne!(status, StatusCode::PAYLOAD_TOO_LARGE, "body = {text}");

    // 1 byte 超えたら 413
    let mut over = body;
    over.push(b' ');
    let (status, ct, text, _) = send(&app, post(over, Some(limit + 1))).await;
    assert_json_413(status, &ct, &text, limit);
}

// ---------------------------------------------------------------------------
// 上限超過
// ---------------------------------------------------------------------------

/// `Content-Length` 付き (早期判定の経路)。
#[tokio::test]
async fn over_limit_with_content_length_is_json_413() {
    let limit = 64 * 1024;
    let app = app_with_body_limit(limit);
    let body = messages_body(2 * 1024 * 1024);
    let n = body.len();
    assert!(n > limit);
    let (status, ct, text, _) = send(&app, post(body, Some(n))).await;
    assert_json_413(status, &ct, &text, limit);
}

/// `Content-Length` 無し (chunked 相当)。`DefaultBodyLimit` が読み込み中に
/// 打ち切った 413 も JSON 封筒に置き換わることを確かめる。
#[tokio::test]
async fn over_limit_without_content_length_is_json_413() {
    let limit = 64 * 1024;
    let app = app_with_body_limit(limit);
    let body = messages_body(2 * 1024 * 1024);
    let (status, ct, text, _) = send(&app, post(body, None)).await;
    assert_json_413(status, &ct, &text, limit);
}

/// OpenAI 互換・Ollama 互換の経路も同じ上限/封筒で止まる。
#[tokio::test]
async fn over_limit_is_json_413_on_all_chat_routes() {
    let limit = 8 * 1024;
    let app = app_with_body_limit(limit);
    for uri in ["/v1/chat/completions", "/v1/completions", "/api/chat"] {
        let mut body = messages_body(256 * 1024);
        body.push(b' ');
        let n = body.len();
        let req = Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CONTENT_LENGTH, n)
            .body(Body::from(body))
            .unwrap();
        let (status, ct, text, _) = send(&app, req).await;
        assert_json_413(status, &ct, &text, limit);
    }
}

// ---------------------------------------------------------------------------
// 解決規則 (CLI > env > 既定)
// ---------------------------------------------------------------------------

#[test]
fn cli_value_wins_over_env_and_default() {
    assert_eq!(resolve_max_body_mb(Some(32)), 32);
    assert_eq!(resolve_max_body_mb(Some(MAX_BODY_MB_LIMIT + 1)), MAX_BODY_MB_LIMIT);
    assert_eq!(resolve_max_body_mb(Some(0)), 1, "0 は 1 MiB に丸める");
    assert_eq!(
        denno_server::resolve_max_body_bytes(Some(2)),
        2 * 1024 * 1024
    );
}

#[test]
fn default_limit_is_16_mib() {
    // env が無い状態での既定 (このテストは env を読む経路を通るので、
    // 他のテストと env を共有しないよう単独で評価する)
    assert_eq!(DEFAULT_MAX_BODY_MB, 16);
    assert_eq!(denno_server::DEFAULT_MAX_BODY_MB, 16);
    if std::env::var(denno_server::MAX_BODY_MB_ENV).is_err() {
        assert_eq!(resolve_max_body_mb(None), DEFAULT_MAX_BODY_MB);
    }
}

/// `DENNO_MAX_BODY_MB` が効き、CLI 値が env に勝つこと。env はプロセス共有
/// なので、このファイル内で env を触るテストを直列化する。
#[tokio::test]
async fn env_sets_the_limit_and_cli_wins() {
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    std::env::set_var(MAX_BODY_MB_ENV, "1");
    assert_eq!(resolve_max_body_mb(None), 1);
    assert_eq!(resolve_max_body_mb(Some(7)), 7, "CLI が env に勝つべき");

    // env 経路で組んだルーター (既定の `router`) が 1 MiB を超えるボディを弾く
    let app = denno_server::router(state());
    let body = messages_body(2 * 1024 * 1024);
    let n = body.len();
    let (status, ct, text, _) = send(&app, post(body, Some(n))).await;
    assert_json_413(status, &ct, &text, 1024 * 1024);
    std::env::remove_var(MAX_BODY_MB_ENV);

    // env を外すと 2 MiB は通る (16 MiB 既定)
    let app = denno_server::router(state());
    let body = messages_body(2 * 1024 * 1024);
    let (status, _ct, text, _) = send(&app, post(body, None)).await;
    assert_ne!(status, StatusCode::PAYLOAD_TOO_LARGE, "body = {text}");
}
