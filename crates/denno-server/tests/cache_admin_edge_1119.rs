//! Task 1119: prefix cache 管理エンドポイント (`GET /v1/cache/stats` /
//! `DELETE /v1/cache`) の異常系・境界の統合テスト。
//!
//! - `api_key` 付きサーバでは、鍵なしは 401、`Authorization: Bearer` と
//!   `x-api-key` のどちらでも 200。誤った鍵は 401 (両エンドポイント)。
//! - 空のキャッシュへの `DELETE /v1/cache` は何度呼んでも `flushed=0` で 200。
//!   エントリを積んだ後の連続 DELETE も「1 回目だけ >0、以降 0」で安定する。
//! - 推論リクエストが走っている最中に `GET /v1/cache/stats` と
//!   `DELETE /v1/cache` を多重に叩いてもデッドロック / panic せず、
//!   `hit_rate` は常に有限で 0.0 以上 1.0 以下。
//!
//! カウンタはプロセス全体の累積 (Relaxed の原子) なので、このバイナリのテストは
//! `gate()` で直列化する (cache_admin_926 と同じ方針)。

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

fn app_with(api_key: Option<&str>) -> Router {
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
    denno_server::router(state)
}

fn app() -> Router {
    app_with(None)
}

async fn send(app: &Router, req: Request<Body>) -> (StatusCode, Value) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

/// リクエストを組み立てる。`auth` は (`ヘッダ名`, `値`) の組。
fn request(method: &str, uri: &str, auth: Option<(&str, &str)>, body: Option<Value>) -> Request<Body> {
    let mut b = Request::builder().method(method).uri(uri);
    if let Some((name, value)) = auth {
        b = b.header(name, value);
    }
    match body {
        Some(v) => b
            .header("content-type", "application/json")
            .body(Body::from(v.to_string()))
            .unwrap(),
        None => b.body(Body::empty()).unwrap(),
    }
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(app, request("GET", uri, None, None)).await
}

async fn delete(app: &Router, uri: &str) -> (StatusCode, Value) {
    send(app, request("DELETE", uri, None, None)).await
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    send(app, request("POST", uri, None, Some(body))).await
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

/// `hit_rate` が「有限・非負・1 以下」であることを検査する。
fn assert_valid_hit_rate(stats: &Value) {
    let r = stats["hit_rate"]
        .as_f64()
        .unwrap_or_else(|| panic!("hit_rate が数値でない: {stats}"));
    assert!(r.is_finite(), "hit_rate が非有限: {r} ({stats})");
    assert!((0.0..=1.0).contains(&r), "hit_rate が範囲外: {r} ({stats})");
}

// ---------------------------------------------------------------------------
// 認証 (401 / Bearer / x-api-key)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn auth_is_enforced_with_either_credential_header() {
    let _g = gate();
    let app = app_with(Some("secret"));

    // 鍵なし: 両エンドポイントとも 401。
    let (status, _) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "鍵なし GET");
    let (status, _) = delete(&app, "/v1/cache").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "鍵なし DELETE");

    // 誤った鍵: 401。
    let (status, _) = send(
        &app,
        request("GET", "/v1/cache/stats", Some(("x-api-key", "wrong")), None),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "誤鍵 GET (x-api-key)");
    let (status, _) = send(
        &app,
        request(
            "DELETE",
            "/v1/cache",
            Some(("authorization", "Bearer wrong")),
            None,
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "誤鍵 DELETE (Bearer)");

    // x-api-key: 200。
    let (status, body) = send(
        &app,
        request("GET", "/v1/cache/stats", Some(("x-api-key", "secret")), None),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "x-api-key GET: {body}");
    assert_valid_hit_rate(&body);
    let (status, body) = send(
        &app,
        request("DELETE", "/v1/cache", Some(("x-api-key", "secret")), None),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "x-api-key DELETE: {body}");

    // Authorization: Bearer: 200 (スキーム名の大小は問わない)。
    for scheme in ["Bearer", "bearer"] {
        let (status, body) = send(
            &app,
            request(
                "GET",
                "/v1/cache/stats",
                Some(("authorization", &format!("{scheme} secret"))),
                None,
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{scheme} GET: {body}");
        let (status, body) = send(
            &app,
            request(
                "DELETE",
                "/v1/cache",
                Some(("authorization", &format!("{scheme} secret"))),
                None,
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{scheme} DELETE: {body}");
    }
}

// ---------------------------------------------------------------------------
// 冪等な DELETE
// ---------------------------------------------------------------------------

#[tokio::test]
async fn repeated_delete_is_idempotent_on_empty_cache() {
    let _g = gate();
    let app = app();

    // 空のキャッシュ: 何度呼んでも 200 / flushed=0 / slots_used=0。
    for i in 0..5 {
        let (status, body) = delete(&app, "/v1/cache").await;
        assert_eq!(status, StatusCode::OK, "{i} 回目: {body}");
        assert_eq!(body["flushed"], 0, "{i} 回目: {body}");
        assert_eq!(body["slots_used"], 0, "{i} 回目: {body}");
        assert!(
            body["slots_capacity"].as_u64().unwrap_or(0) > 0,
            "{i} 回目に予算が 0 に戻っていない: {body}"
        );
    }
    // stats も 0 件のまま。hit_rate は 0 (分母 0 でも NaN にしない)。
    let (status, stats) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK, "{stats}");
    assert_eq!(stats["slots_used"], 0, "{stats}");
    assert_valid_hit_rate(&stats);
}

#[tokio::test]
async fn delete_drains_entries_then_returns_zero() {
    let _g = gate();
    let app = app();

    // 同じ会話を 2 回: 1 回目 miss、2 回目 hit。prefix エントリが 1 本残る。
    for _ in 0..2 {
        let (status, body) = post(&app, "/v1/messages", chat_body(8)).await;
        assert_eq!(status, StatusCode::OK, "body = {body}");
    }
    let (status, stats) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert!(stats["slots_used"].as_u64().unwrap() >= 1, "{stats}");

    // 1 回目の DELETE で捨てる。
    let (status, first) = delete(&app, "/v1/cache").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert!(first["flushed"].as_u64().unwrap() >= 1, "{first}");
    assert_eq!(first["slots_used"], 0, "{first}");

    // 以降は何度呼んでも flushed=0 (累積カウンタは保持される)。
    for i in 0..4 {
        let (status, body) = delete(&app, "/v1/cache").await;
        assert_eq!(status, StatusCode::OK, "{i} 回目: {body}");
        assert_eq!(body["flushed"], 0, "{i} 回目: {body}");
        assert_eq!(body["slots_used"], 0, "{i} 回目: {body}");
    }
    let (status, after) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(after["slots_used"], 0, "{after}");
    assert_eq!(after["hits"], stats["hits"], "{after}");
    assert_eq!(after["misses"], stats["misses"], "{after}");
    assert_valid_hit_rate(&after);
}

// ---------------------------------------------------------------------------
// 推論中の stats / flush 多重呼び出し
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stats_and_flush_stay_sane_during_active_inference() {
    let _g = gate();
    let app = app();

    // 先に prefix cache を温める (flush が実際に何か捨てる状態にする)。
    for _ in 0..2 {
        let (status, body) = post(&app, "/v1/messages", chat_body(8)).await;
        assert_eq!(status, StatusCode::OK, "warmup: {body}");
    }

    // 生成タスクが実際に in-flight だった瞬間を観測できたか (kv_slots_busy > 0)。
    let observed_busy = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // 生成タスク: 各 8 リクエストを順に流す。
    let mut gens = Vec::new();
    for _ in 0..4 {
        let a = app.clone();
        gens.push(tokio::spawn(async move {
            let mut out = Vec::new();
            for _ in 0..8 {
                out.push(post(&a, "/v1/messages", chat_body(32)).await);
            }
            out
        }));
    }

    // 管理タスク: stats と flush を交互に多重で叩く。
    let mut admins = Vec::new();
    for _ in 0..4 {
        let a = app.clone();
        let seen = observed_busy.clone();
        admins.push(tokio::spawn(async move {
            for _ in 0..40 {
                let (status, stats) = get(&a, "/v1/cache/stats").await;
                assert_eq!(status, StatusCode::OK, "stats: {stats}");
                assert_valid_hit_rate(&stats);
                assert!(stats["slots_used"].as_u64().is_some(), "{stats}");
                assert!(stats["slots_capacity"].as_u64().unwrap_or(0) > 0, "{stats}");
                let busy = stats["kv_slots_busy"].as_u64().unwrap_or(0);
                seen.fetch_max(busy as usize, std::sync::atomic::Ordering::Relaxed);
                let (status, flushed) = delete(&a, "/v1/cache").await;
                assert_eq!(status, StatusCode::OK, "flush: {flushed}");
                assert!(flushed.get("flushed").is_some(), "{flushed}");
                assert_eq!(flushed["slots_used"], 0, "{flushed}");
                tokio::task::yield_now().await;
            }
        }));
    }

    // デッドロック検知: 全体を上限付きで待つ。
    let all = async {
        for g in gens {
            let outs = g.await.expect("生成タスクが panic した");
            for (status, body) in outs {
                assert_eq!(status, StatusCode::OK, "生成が失敗: {body}");
                assert!(
                    body["usage"]["output_tokens"].as_u64().unwrap_or(0) > 0,
                    "生成が空: {body}"
                );
            }
        }
        for a in admins {
            a.await.expect("管理タスクが panic した");
        }
    };
    tokio::time::timeout(Duration::from_secs(120), all)
        .await
        .expect("推論中の stats / flush がデッドロックした (120 秒超過)");

    // 管理呼び出しの最中に、推論リクエストが実際にスロットを握っていたこと。
    // (stats と DELETE が生成と重なった状態を観測できた = 「推論中」の実証)
    let seen = observed_busy.load(std::sync::atomic::Ordering::Relaxed);
    assert!(seen > 0, "推論 in-flight の瞬間を観測できなかった (kv_slots_busy が常に 0)");

    // 事後も stats は健全。hit_rate は有限で [0, 1]。
    let (status, stats) = get(&app, "/v1/cache/stats").await;
    assert_eq!(status, StatusCode::OK, "{stats}");
    assert_valid_hit_rate(&stats);
    assert_eq!(stats["slots_used"], 0, "最後の flush 後は 0 件: {stats}");
    let (hits, misses, _reused) = denno_server::prefix_cache_counters();
    assert_eq!(stats["hits"].as_u64().unwrap(), hits, "{stats}");
    assert_eq!(stats["misses"].as_u64().unwrap(), misses, "{stats}");
}
