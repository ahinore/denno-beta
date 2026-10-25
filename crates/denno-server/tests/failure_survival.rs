//! P57: 「1 リクエストの失敗がサーバ全体の永久停止に化ける」欠陥の回帰テスト。
//!
//! 背景 (2026-10-25 本番障害): VRAM 枯渇 → `cuMemAlloc` 失敗 → 残差チェーン失敗
//! (`prefill residual chain: att upload failed`) → GPU_BROKEN ラッチ → 以降の
//! prefill は CPU フォールバックで 1 ステップが数分。その間ワーカーがエンジン
//! Mutex をほぼ連続保持していたため、`/v1/models` はロック待ちで **200 すら
//! 返せず**、プロセスを殺すまで復旧しなかった。
//!
//! 本ファイルが固定する性質 (**GPU 無しで回る**):
//!
//! 1. `models_responds_while_engine_lock_is_held` — エンジン Mutex を他スレッドが
//!    握ったままでも `/v1/models` は 200 を返す (= 直列化と無応答は別物)。
//! 2. `repeated_forward_failures_do_not_leak_slots` — 失敗が続いてもスロットは
//!    解放され、そのあとの正常リクエストが通る (プロセスは生き残る)。
//! 3. `queued_request_beyond_stall_returns_500_and_server_recovers` — 無進捗が
//!    `Limits::stall` を超えたリクエストは 500 で返り、スロットが解放され、
//!    そのあとのリクエストが通る (旧実装は無応答のままだった)。
//! 4. `stalled_sse_reader_does_not_block_the_worker` — 読み手が止まった SSE
//!    クライアントでワーカーが永久ブロックしない (`send_bounded`)。
//!
//! 数値経路には触れない (モデルはすべてテスト用のダミー)。

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::response::sse::Event;
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_CONTEXT, DUMMY_VOCAB};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_server::{AppState, EngineInfo, Limits, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------- テスト用モデル ----------

/// 決定論的な logits を返す最小モデル (EOS 無し)。
struct TinyModel {
    ctx: usize,
}

impl Model for TinyModel {
    fn context(&self) -> usize {
        self.ctx
    }
    fn eos_ids(&self) -> &[u32] {
        &[]
    }
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        Ok(logits_for(tokens))
    }
    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.ctx)
    }
}

/// 最初の `fails` 回の forward が必ず Err になるモデル (P57 の失敗注入)。
/// GPU 失敗 (チェーン失敗) と同じく「prefill の forward が Err」の経路を通す。
struct FailNModel {
    ctx: usize,
    fails: AtomicUsize,
}

impl Model for FailNModel {
    fn context(&self) -> usize {
        self.ctx
    }
    fn eos_ids(&self) -> &[u32] {
        &[]
    }
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        if self.fails.load(Ordering::SeqCst) > 0 {
            self.fails.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("injected forward failure (P57 probe)");
        }
        cache.advance(tokens.len());
        Ok(logits_for(tokens))
    }
    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.ctx)
    }
}

/// 最初の 1 回の forward だけゲートで止まるモデル (テストが open するまで待つ)。
struct GatedModel {
    ctx: usize,
    entered: Arc<AtomicBool>,
    open: Arc<AtomicBool>,
}

impl Model for GatedModel {
    fn context(&self) -> usize {
        self.ctx
    }
    fn eos_ids(&self) -> &[u32] {
        &[]
    }
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        if !self.entered.swap(true, Ordering::SeqCst) {
            let deadline = std::time::Instant::now() + Duration::from_secs(15);
            while !self.open.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
        }
        cache.advance(tokens.len());
        Ok(logits_for(tokens))
    }
    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.ctx)
    }
}

/// forward ごとに `delay` だけ寝るモデル (SSE 送信の詰まりを作る)。
struct SlowModel {
    ctx: usize,
    delay: Duration,
}

impl Model for SlowModel {
    fn context(&self) -> usize {
        self.ctx
    }
    fn eos_ids(&self) -> &[u32] {
        &[]
    }
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        std::thread::sleep(self.delay);
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        Ok(logits_for(tokens))
    }
    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.ctx)
    }
}

/// DummyModel 系と同一の決定論的 logits (最終トークンの次を強く推す)。
fn logits_for(tokens: &[u32]) -> Vec<f32> {
    let last = tokens.last().copied().unwrap_or(0) as usize;
    let mut logits = vec![0.0f32; DUMMY_VOCAB];
    logits[(last + 1) % DUMMY_VOCAB] = 5.0;
    logits[(last + 2) % DUMMY_VOCAB] = 1.0;
    logits
}

/// ゲートがテスト終了時に必ず開くようにする (テストが panic しても
/// ワーカースレッドが 15 秒待つだけで済む)。
struct Gate(Arc<AtomicBool>);
impl Drop for Gate {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

// ---------- AppState / HTTP ----------

fn app_state(model: Box<dyn Model>, max_queue: usize, limits: Limits) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn_with_limits(engine.clone(), max_queue, limits);
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
    // 応答が返らない退行は「ハング」ではなく「失敗」にする (CI を止めない)
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
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

fn msg(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: text.into() }]
}

fn params(max_tokens: usize) -> GenParams {
    GenParams { max_tokens, temperature: Some(0.0), ..Default::default() }
}

// ---------- 1. 生成中でも制御プレーンは応答する ----------

/// `/v1/models` はエンジン Mutex を取らない (P57)。他スレッドが Mutex を
/// 握ったままでも 200 を返す — 「直列化」と「無応答」は別物。
///
/// 注: multi_thread ランタイムで回す。単一スレッドだと退行時 (ハンドラが Mutex で
/// ブロック) に `timeout` 自体が動かず、テストが「失敗」ではなく「ハング」になる。
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn models_responds_while_engine_lock_is_held() {
    let state = app_state(Box::new(TinyModel { ctx: DUMMY_CONTEXT }), 8, Limits::default());
    let app = denno_server::router(state.clone());

    // エンジン Mutex を握り続けるスレッド (生成中のワーカーを模擬)
    let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
    let held = Arc::new(AtomicBool::new(false));
    let engine = state.engine.clone();
    let held_in_thread = held.clone();
    let holder = std::thread::spawn(move || {
        let _g = engine.lock().unwrap();
        held_in_thread.store(true, Ordering::SeqCst);
        let _ = release_rx.recv();
    });

    // ロックが実際に握られるまで待つ (前提の確認)
    for _ in 0..2000 {
        if held.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert!(held.load(Ordering::SeqCst), "ロック保持スレッドが動いていない");
    assert!(
        state.engine.try_lock().is_err(),
        "エンジンロックが握られていない (テストの前提が崩れている)"
    );

    let (status, body) = tokio::time::timeout(Duration::from_millis(500), get_models(&app))
        .await
        .expect("/v1/models が 500ms 以内に返らない (エンジンロック待ちで詰まっている)");
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert_eq!(
        body["data"][0]["context_length"],
        json!(DUMMY_CONTEXT),
        "context_length が起動時キャッシュの値と違う: {body}"
    );

    release_tx.send(()).unwrap();
    holder.join().unwrap();
}

// ---------- 2. 失敗が続いてもプロセスは生き残る ----------

/// prefill の forward が失敗し続けても、スロットは解放され、そのあとの正常な
/// リクエストが通る。旧実装の失敗経路 (install 後の Err) はスロット漏れを
/// 起こし得た (s144 で修正済み — ここで P57 の「プロセス生存」も固定する)。
#[tokio::test]
async fn repeated_forward_failures_do_not_leak_slots() {
    let state = app_state(
        Box::new(FailNModel { ctx: 256, fails: AtomicUsize::new(5) }),
        8,
        Limits::default(),
    );
    let app = denno_server::router(state.clone());

    for i in 0..5 {
        let (status, body) = post(&app, body(4)).await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{i} 回目: {body}");
        let text = body.to_string();
        assert!(
            text.contains("injected forward failure"),
            "{i} 回目の本文にエンジン原因が無い (スロット枯渇に化けていないか): {text}"
        );
    }
    // 6 回目は通る = ワーカーもスロットも生きている
    let (status, body) = post(&app, body(4)).await;
    assert_eq!(status, StatusCode::OK, "失敗のあと正常リクエストが通らない: {body}");
}

// ---------- 3. 無進捗の上限で 500 を返して復旧する ----------

/// 待ち行列に居る間に `Limits::stall` を超えたリクエストは 500 (api_error) で
/// 返り、スロットが解放され、次のリクエストが通る。
#[tokio::test]
async fn queued_request_beyond_stall_returns_500_and_server_recovers() {
    let entered = Arc::new(AtomicBool::new(false));
    let open = Arc::new(AtomicBool::new(false));
    let gate = Gate(open.clone());
    let limits = Limits {
        admit: Duration::from_secs(10),
        stall: Duration::from_millis(150),
        send: Duration::from_secs(2),
    };
    let state = app_state(
        Box::new(GatedModel { ctx: 256, entered: entered.clone(), open: open.clone() }),
        8,
        limits,
    );
    let app = denno_server::router(state.clone());

    // 1 件目: prefill の forward がゲートで止まる (ワーカーはステップ内で待つ)
    let app1 = app.clone();
    let h1 = tokio::spawn(async move { post(&app1, body(4)).await });
    for _ in 0..2000 {
        if entered.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert!(entered.load(Ordering::SeqCst), "1 件目が forward に到達しない");

    // 2 件目: ワーカーがステップ内で止まっている間に待ち行列へ入る
    let app2 = app.clone();
    let h2 = tokio::spawn(async move { post(&app2, body(4)).await });

    // stall (150ms) より長く待たせてからゲートを開く
    tokio::time::sleep(Duration::from_millis(400)).await;
    open.store(true, Ordering::SeqCst);

    let (s1, b1) = tokio::time::timeout(Duration::from_secs(20), h1)
        .await
        .expect("1 件目が返らない (ワーカーが止まっている)")
        .unwrap();
    assert_eq!(s1, StatusCode::OK, "1 件目 (進んでいた方) が落ちた: {b1}");

    let (s2, b2) = tokio::time::timeout(Duration::from_secs(20), h2)
        .await
        .expect("2 件目が返らない (上限が効いていない = 無応答)")
        .unwrap();
    assert_eq!(
        s2,
        StatusCode::INTERNAL_SERVER_ERROR,
        "無進捗の上限を超えたリクエストが 500 で返らない: {b2}"
    );
    let text = b2.to_string();
    assert!(
        text.contains("timeout") && text.contains("generation progress"),
        "500 の本文が上限超過を説明していない: {text}"
    );

    // 3 件目: スロットもワーカーも解放済み = 通る
    let (s3, b3) = post(&app, body(4)).await;
    assert_eq!(s3, StatusCode::OK, "打ち切りのあとサーバが復旧しない: {b3}");
    drop(gate);
}

// ---------- 4. 読み手が止まった SSE でワーカーが止まらない ----------

/// 受信バッファが詰まったまま読まれない SSE では、有界送信 (`Limits::send`) が
/// 諦めて Run を打ち切る。ワーカーは永久ブロックせず、後続のリクエストが通る。
/// (旧実装は `blocking_send` でエンジン Mutex を握ったまま永久ブロックし得た)
#[tokio::test]
async fn stalled_sse_reader_does_not_block_the_worker() {
    let limits = Limits {
        admit: Duration::from_secs(10),
        stall: Duration::from_secs(60),
        send: Duration::from_millis(80),
    };
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(SlowModel { ctx: 256, delay: Duration::from_millis(2) }),
        tok,
        "denno-test",
        None,
    );
    let engine = Arc::new(Mutex::new(engine));
    let sched = Scheduler::spawn_with_limits(engine, 8, limits);

    // 受信バッファ 1 本。プリアンブル (message_start / content_block_start / ping)
    // だけ読んで、そのあとは読まない = 読み手が止まったクライアント。
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(1);
    sched
        .submit_stream(None, msg("t0"), params(8), None, None, None, tx)
        .expect("submit_stream");
    for i in 0..3 {
        let got = tokio::time::timeout(Duration::from_secs(5), rx.recv())
            .await
            .unwrap_or_else(|_| panic!("プリアンブル {i} 本目が来ない"));
        assert!(got.is_some(), "プリアンブル {i} 本目でチャネルが閉じた");
    }

    // ここから読まない。delta の送信が詰まり、send (80ms) で諦めて打ち切られる。
    // 証明: 後続の非ストリーム要求が完走する (ワーカーがブロックしていない)。
    let rx2 = sched
        .submit_chat(None, msg("t0"), params(4), None, None, None)
        .expect("submit_chat");
    let out = tokio::time::timeout(Duration::from_secs(10), rx2)
        .await
        .expect("ワーカーが SSE 送信でブロックしたまま (後続リクエストが返らない)")
        .expect("scheduler が応答を捨てた");
    assert!(out.is_ok(), "後続リクエストが失敗: {:?}", out.err().map(|e| format!("{e:#}")));
    drop(rx);
}
