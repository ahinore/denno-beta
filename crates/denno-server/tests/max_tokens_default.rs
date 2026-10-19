//! chat 用 max_tokens 既定上限 (未指定時のみ適用・明示指定は尊重) の統合テスト。
//!
//! denno-engine のダミートークナイザ + EOS を吐かないローカルダミーモデル
//! (EndlessModel・本ファイル内定義。tests/scheduler.rs の SleepModel と同一方式)
//! を使い、HTTP ワイヤレベル (axum Router + tower oneshot) で検証する。
//! EndlessModel は EOS を決して選ばないため生成は max_tokens まで必ず続き、
//! usage.output_tokens / stop_reason で実効 max_tokens を直接観測できる。
//!
//! カバー項目:
//! - resolve_max_tokens の単体テスト (未指定 → 既定 1024・明示 → 尊重・別名・優先順位)
//! - max_tokens 未指定 (非ストリーム) → DEFAULT_MAX_TOKENS (1024) で打ち切られ
//!   stop_reason = "max_tokens" (旧コードでは未指定は必須フィールド欠落で 400)
//! - max_tokens 未指定 (stream=true) → SSE の message_delta で同一の打ち切り
//! - 明示指定 (max_tokens=5 / 2048) は尊重される (既定 1024 に切り下げられない)
//! - 明示が上限 4096 を超える場合は 4096 に clamp される (s53)
//! - OpenAI 互換別名 max_completion_tokens のみの指定も実効値になる
//! - 両方指定時は max_tokens 優先 (Anthropic 形式のネイティブ名優先)
//! - 明示 0 も 0 として尊重する (既定へフォールバックしない)

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{
    resolve_max_tokens, AppState, Scheduler, DEFAULT_MAX_TOKENS, MAX_TOKENS_CLAMP,
};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// テスト用モデルのコンテキスト長 (clamp 上限 4096 + プロンプトが収まる大きさ)
const TEST_CONTEXT: usize = 8192;

/// EOS を決して出力しない決定論的ダミーモデル (peak = 直前トークン + 1、
/// EOS の位置だけ +1 ずらして回避)。生成は max_tokens まで必ず続く。
struct EndlessModel;

impl Model for EndlessModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        // P17e: 需要ベース確保 (書き込み前に容量確保) のダミー側契約
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = (last + 1) % DUMMY_VOCAB;
        let peak = if peak == DUMMY_EOS as usize { peak + 1 } else { peak };
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

/// テスト用 AppState + Router を構築する (tests/scheduler.rs と同一方式)
fn test_app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(EndlessModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
    });
    denno_server::router(state)
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

// ---------- resolve_max_tokens 単体テスト ----------

#[test]
fn resolve_max_tokens_applies_default_only_when_unspecified() {
    // 未指定 → 既定上限 (s53: 1024)
    assert_eq!(resolve_max_tokens(None, None), DEFAULT_MAX_TOKENS);
    assert_eq!(DEFAULT_MAX_TOKENS, 1024);
    // 明示指定は尊重 (0・既定超え・上限まで)
    assert_eq!(resolve_max_tokens(Some(5), None), 5);
    assert_eq!(resolve_max_tokens(Some(0), None), 0);
    assert_eq!(resolve_max_tokens(Some(2048), None), 2048);
    assert_eq!(resolve_max_tokens(Some(MAX_TOKENS_CLAMP), None), MAX_TOKENS_CLAMP);
    // s53: 上限 4096 を超える明示は clamp される (未指定の既定 1024 とは独立)
    assert_eq!(MAX_TOKENS_CLAMP, 4096);
    assert_eq!(resolve_max_tokens(Some(100_000), None), MAX_TOKENS_CLAMP);
    assert_eq!(resolve_max_tokens(None, Some(100_000)), MAX_TOKENS_CLAMP);
    // OpenAI 互換別名のみ → 別名を採用
    assert_eq!(resolve_max_tokens(None, Some(7)), 7);
    // 両方 → max_tokens 優先
    assert_eq!(resolve_max_tokens(Some(5), Some(9)), 5);
}

// ---------- ワイヤレベル (未指定 → 既定上限) ----------

/// 未指定 (max_tokens も max_completion_tokens も無し) → 既定上限で打ち切られる。
/// finish_reason 相当は Anthropic 形式の stop_reason = "max_tokens"
/// (OpenAI の finish_reason = "length" 相当) として正しく出る。
#[tokio::test]
async fn unspecified_max_tokens_is_capped_at_default_non_stream() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["type"], "message");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["usage"]["output_tokens"], DEFAULT_MAX_TOKENS as u64);
}

/// stream=true の未指定も同一 (Job::Stream 経路)。SSE の message_delta に
/// stop_reason = "max_tokens" と usage.output_tokens = 既定上限が流れる。
#[tokio::test]
async fn unspecified_max_tokens_is_capped_at_default_stream() {
    let app = test_app();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "denno-test",
                "temperature": 0.0,
                "stream": true,
                "messages": [{"role": "user", "content": "t0"}],
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    let mut capped = false;
    for line in text.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if let Ok(v) = serde_json::from_str::<Value>(data) {
                if v["type"] == "message_delta" {
                    assert_eq!(v["delta"]["stop_reason"], "max_tokens", "SSE:\n{text}");
                    assert_eq!(
                        v["usage"]["output_tokens"],
                        DEFAULT_MAX_TOKENS as u64,
                        "SSE:\n{text}"
                    );
                    capped = true;
                }
            }
        }
    }
    assert!(capped, "message_delta not found in:\n{text}");
}

// ---------- ワイヤレベル (明示指定は尊重) ----------

/// 明示指定は尊重される (既定 4096 に引き上げられない)。
#[tokio::test]
async fn explicit_max_tokens_is_respected() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["usage"]["output_tokens"], 5);
}

/// OpenAI 互換別名 max_completion_tokens のみの指定も実効値になる。
#[tokio::test]
async fn max_completion_tokens_alias_is_honored() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_completion_tokens": 7,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["usage"]["output_tokens"], 7);
}

/// 両方指定時は max_tokens 優先 (Anthropic 形式のネイティブ名を優先)。
#[tokio::test]
async fn max_tokens_takes_precedence_over_alias() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 5,
            "max_completion_tokens": 9,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["usage"]["output_tokens"], 5);
}

/// 明示 0 も 0 として尊重する (既定へフォールバックしない。応答は空テキスト)。
#[tokio::test]
async fn explicit_zero_is_respected() {
    let app = test_app();
    let (status, body) = post(
        &app,
        json!({
            "model": "denno-test",
            "max_tokens": 0,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "t0"}],
        }),
    )
    .await;

    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(body["stop_reason"], "max_tokens");
    assert_eq!(body["usage"]["output_tokens"], 0);
    assert_eq!(body["content"][0]["text"], "");
}
