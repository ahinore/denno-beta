//! P58-4: 失敗したリクエストの後始末 (KV 予約の解放 + 500 応答 + 復旧)。
//!
//! 背景 (2026-10-25 実測): VRAM 枯渇で prefill が失敗したあと KV 予約が
//! 解放されず、以後のリクエストがすべて
//! `KV 予算超過 (予約 22,173 + 20,058 tokens × 131072 B/token > 4096 MB)`
//! で弾かれた。予約 (`slot_reserved`) はエンジン側の会計で、失敗経路の
//! 解放漏れが原因だった (P58-4 で RAII ガード化)。
//!
//! 本テストが固定する性質 (**GPU 無しで回る**):
//!
//! 1. 予算を「1 本分ちょうど」に絞った状態で prefill を失敗させると、
//!    500 (api_error) が返り、そのリクエストの予約が 0 に戻る。
//! 2. 続く 2 本目が `KV 予算超過` にならずに完走する (サーバが復旧する)。
//! 3. 失敗のあと `/v1/models` が 200 を返す (エンジンロックを保持したまま
//!    スピンしていない)。
//!
//! 数値経路には触れない (モデルはテスト用のダミー)。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// テスト用コンテキスト長。bytes_per_token = kv_dim(4) × 2 × 4 B = 32 B。
const CTX: usize = 4096;
const BPT: usize = 32;

/// 最初の forward だけ Err を返すモデル (VRAM 枯渇で prefill が落ちる状況)。
/// 以後は EOS (12) を出すので 1 トークンで終わる。
struct FailOnceModel {
    failed: Arc<AtomicBool>,
}

impl Model for FailOnceModel {
    fn context(&self) -> usize {
        CTX
    }

    fn eos_ids(&self) -> &[u32] {
        &[12]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        // KV 行を書いて len を進めた**後**に失敗する (実機の「残差チェーン
        // 失敗」と同じ位置 = 部分 prefill の KV が残る状況を作る)。
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        if !self.failed.swap(true, Ordering::SeqCst) {
            anyhow::bail!("injected VRAM exhaustion (P58-4 probe)");
        }
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[12] = 5.0; // EOS
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, CTX)
    }
}

fn app_state() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let model = Box::new(FailOnceModel { failed: Arc::new(AtomicBool::new(false)) });
    let mut engine = Engine::from_parts(model, tok, "denno-test", None);
    // 予算を「1 本分ちょうど」に絞る。need = prompt + max_tokens で max_tokens は
    // ctx - prompt にクランプされるため need は常に CTX トークン =
    // CTX * BPT バイト。予約が漏れていれば 2 本目が KV 予算超過になる。
    engine.set_kv_budget_bytes(Some(CTX * BPT));
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

fn body(max_tokens: usize) -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
        "messages": [{"role": "user", "content": "t0"}]
    })
}

async fn post(app: &Router, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(20), app.clone().oneshot(req))
        .await
        .expect("20s 以内に応答が返らない (サーバが無応答)")
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

async fn get_models(app: &Router) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(5), app.clone().oneshot(req))
        .await
        .expect("/v1/models が返らない (エンジンロックを保持したまま停止している)")
        .unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// 失敗したリクエストの予約が戻り、次のリクエストが通る。
#[tokio::test]
async fn failed_request_returns_500_releases_budget_and_next_request_succeeds() {
    let state = app_state();
    let app = denno_server::router(state.clone());

    // 1 本目: prefill の forward が失敗する。
    let (status, b) = post(&app, body(CTX)).await;
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "失敗したリクエストが 500 で返らない: {b}"
    );
    let text = b.to_string();
    assert!(
        text.contains("injected VRAM exhaustion"),
        "500 の本文にエンジン原因が無い (別の理由で落ちている): {text}"
    );
    assert!(
        !text.contains("KV 予算超過"),
        "失敗の理由が予算超過に化けている: {text}"
    );

    // 失敗直後の会計: 予約・busy・部分 KV がすべて戻っている。
    {
        let g = state.engine.lock().unwrap();
        assert_eq!(g.slot_reserved_total(), 0, "失敗したリクエストの予約が残っている");
        assert_eq!(g.slot_busy_count(), 0, "失敗したリクエストのスロットが busy のまま");
        let live: Vec<usize> = g.caches.iter().map(|c| c.len()).filter(|&l| l > 0).collect();
        assert!(
            live.is_empty(),
            "失敗したリクエストの部分 prefill の KV 行が残っている (len = {live:?})"
        );
    }

    // 制御プレーンは生きている (ロック保持のままスピンしていない)。
    let (s_models, m) = get_models(&app).await;
    assert_eq!(s_models, StatusCode::OK, "/v1/models が 200 を返さない: {m}");

    // 2 本目: 予算は 1 本分しかない。予約が戻っていれば通る。
    let (status, b) = post(&app, body(CTX)).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "失敗のあと次のリクエストが通らない (予約の解放漏れ): {b}"
    );
    assert_eq!(b["type"], "message", "{b}");
    assert_eq!(b["stop_reason"], "end_turn", "{b}");
    assert_eq!(b["usage"]["output_tokens"], 1, "{b}");

    // 3 本目も通る (会計が積み上がっていない)。
    let (status, b) = post(&app, body(CTX)).await;
    assert_eq!(status, StatusCode::OK, "3 本目が通らない: {b}");

    let g = state.engine.lock().unwrap();
    assert_eq!(g.slot_reserved_total(), 0);
    assert_eq!(g.slot_busy_count(), 0);
    assert_eq!(state.scheduler.in_flight(), 0);
}
