//! Task 926: prefix cache 管理エンドポイントの統合テスト。
//!
//! - `GET /v1/cache/stats` の形 (slots_used / slots_capacity / hits / misses /
//!   hit_rate / total_reused_tokens) と、新しいエンジンでの初期値
//! - 同じ会話を 2 回流すと 1 回目 = miss、2 回目 = hit、再利用トークンが増える
//! - `DELETE /v1/cache` で保持中の prefix エントリが消える (カウンタは保持)
//! - `DELETE /v1/cache` は走っているリクエストを止めない
//! - `api_key` 付きサーバでは両方 401 (鍵なし) / 200 (正しい鍵)
//!
//! カウンタはプロセス全体の累積 (Relaxed の原子) なので、このバイナリのテストは
//! `gate()` で直列化し、値は「前後の差分」で見る。

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// このバイナリのテストを直列化する (グローバルカウンタの読み書きのため)。
static COUNTER_GATE: Mutex<()> = Mutex::new(());

fn gate() -> MutexGuard<'static, ()> {
    COUNTER_GATE.lock().unwrap_or_else(|e| e.into_inner())
}

fn app_with(api_key: Option<&str>) -> (Router, Arc<AppState>) {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(|s| s.to_string()),
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    (denno_server::router(state.clone()), state)
}

fn app() -> (Router, Arc<AppState>) {
    app_with(None)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(app, Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap()).await
}

async fn get_with_key(app: &Router, uri: &str, key: &str) -> (StatusCode, Value) {
    send(
        app,
        Request::builder()
            .method("GET")
            .uri(uri)
            .header("x-api-key", key)
            .body(Body::empty())
            .unwrap(),
    )
    .await
}

async fn delete(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(
        app,
        Request::builder().method("DELETE").uri(uri).body(Body::empty()).unwrap(),
    )
    .await
}

async fn delete_with_key(app: &Router, uri: &str, key: &str) -> (StatusCode, Value) {
    send(
        app,
        Request::builder()
            .method("DELETE")
            .uri(uri)
            .header("x-api-key", key)
            .body(Body::empty())
            .unwrap(),
    )
    .await
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(
        app,
        Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
}

fn chat_body(max_tokens: u32) -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    })
}

fn rate(hits: u64, misses: u64) -> f64 {
    let n = hits + misses;
    if n == 0 {
        0.0
    } else {
        hits as f64 / n as f64
    }
}

// ---------------------------------------------------------------------------
// 形と初期値
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stats_reports_required_keys_and_engine_state() {
    let _g = gate();
    let (app, state) = app();
    let (status, body) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    for k in [
        "slots_used",
        "slots_capacity",
        "hits",
        "misses",
        "hit_rate",
        "total_reused_tokens",
    ] {
        assert!(body.get(k).is_some(), "{k} が無い: {body}");
    }
    // 新しいエンジンでは prefix エントリは 0、予算は > 0 (LRU のトークン上限)。
    assert_eq!(body["slots_used"], 0);
    assert!(
        body["slots_capacity"].as_u64().unwrap_or(0) > 0,
        "slot 容量が 0: {body}"
    );
    // カウンタは公開 API と同じ値 (先行テストの分も含む)。
    let (hits, misses, reused) = denno_server::prefix_cache_counters();
    assert_eq!(body["hits"].as_u64().unwrap(), hits);
    assert_eq!(body["misses"].as_u64().unwrap(), misses);
    assert_eq!(body["total_reused_tokens"].as_u64().unwrap(), reused);
    let got = body["hit_rate"].as_f64().unwrap();
    assert!((got - rate(hits, misses)).abs() < 1e-12, "hit_rate = {got}");
    // KV スロット側の占有も併記される (in-flight は 0)。
    assert_eq!(body["kv_slots"].as_u64().unwrap() as usize, state.info.slots);
    assert_eq!(body["kv_slots_busy"], 0);
}

// ---------------------------------------------------------------------------
// 再利用の計上と flush
// ---------------------------------------------------------------------------

#[tokio::test]
async fn reuse_updates_counters_and_delete_flushes_entries() {
    let _g = gate();
    let (app, _state) = app();
    let (h0, m0, r0) = denno_server::prefix_cache_counters();

    // 1 回目: 空のエンジンなのでフル prefill (再利用 0)。
    let (status, first) = post(&app, "/v1/messages", chat_body(8)).await;
    assert_eq!(status, StatusCode::OK, "body = {first}");
    assert_eq!(first["usage"]["cache_read_input_tokens"], 0);

    // 2 回目: 同じ会話 → prefix cache から前置きを使い回す。
    let (status, second) = post(&app, "/v1/messages", chat_body(8)).await;
    assert_eq!(status, StatusCode::OK, "body = {second}");
    let reused = second["usage"]["cache_read_input_tokens"]
        .as_u64()
        .expect("cache_read_input_tokens が数値でない");
    assert!(reused > 0, "2 回目が prefix を再利用していない: {second}");
    // 出力は 1 回目と同じ (再利用で数値経路が変わらない)。
    assert_eq!(first["content"], second["content"]);

    let (status, stats) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(stats["hits"].as_u64().unwrap(), h0 + 1, "{stats}");
    assert_eq!(stats["misses"].as_u64().unwrap(), m0 + 1, "{stats}");
    assert_eq!(
        stats["total_reused_tokens"].as_u64().unwrap(),
        r0 + reused,
        "{stats}"
    );
    assert!(stats["slots_used"].as_u64().unwrap() >= 1, "{stats}");

    // flush: 保持中の prefix エントリが消える。
    let (status, flushed) = delete(&app, "/v1/cache").await;
    assert_eq!(status, StatusCode::OK, "body = {flushed}");
    assert!(flushed["flushed"].as_u64().unwrap() >= 1, "{flushed}");
    assert_eq!(flushed["slots_used"], 0);

    let (status, after) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(after["slots_used"], 0, "{after}");
    // カウンタは flush で消えない (別の意味の累積値)。
    assert_eq!(after["hits"], stats["hits"]);
    assert_eq!(after["misses"], stats["misses"]);
    assert_eq!(after["total_reused_tokens"], stats["total_reused_tokens"]);

    // 空のキャッシュへの flush は 0 を返し、成功する (冪等)。
    let (status, again) = delete(&app, "/v1/cache").await;
    assert_eq!(status, StatusCode::OK, "body = {again}");
    assert_eq!(again["flushed"], 0);
    assert_eq!(again["slots_used"], 0);
}

// ---------------------------------------------------------------------------
// in-flight 不可侵
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_does_not_disturb_in_flight_request() {
    let _g = gate();
    let (app, _state) = app();
    let app_bg = app.clone();
    let gen = tokio::spawn(async move { post(&app_bg, "/v1/messages", chat_body(64)).await });

    // 生成が始まるのを待ってから flush する (flush はエンジンロックを取るが、
    // ワーカーはステップ単位でロックを離すので待たされるだけで止めない)。
    tokio::time::sleep(Duration::from_millis(5)).await;
    let (status, flushed) = delete(&app, "/v1/cache").await;
    assert_eq!(status, StatusCode::OK, "body = {flushed}");
    assert!(flushed.get("flushed").is_some(), "{flushed}");

    let (status, out) = gen.await.expect("生成タスクが panic した");
    assert_eq!(status, StatusCode::OK, "body = {out}");
    assert!(
        out["usage"]["output_tokens"].as_u64().unwrap_or(0) > 0,
        "生成が空: {out}"
    );
}

// ---------------------------------------------------------------------------
// 認証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_endpoints_require_api_key_when_configured() {
    let _g = gate();
    let (app, _state) = app_with(Some("secret"));
    let (status, _) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _) = delete(&app, "/v1/cache").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let (status, body) = get_with_key(&app, "/v1/cache/stats", "secret").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    let (status, body) = delete_with_key(&app, "/v1/cache", "secret").await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
}
