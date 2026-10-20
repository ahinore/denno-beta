//! s145 (B8-1b): "no free KV cache slot" の回帰テスト (スロット漏れの固定)。
//!
//! 背景: 旧 `admit_job` は `admit` の失敗時、`client_gone()` が真の場合にしか
//! `release_session` を呼ばなかった。`admit` は `install_session` (スロット取得) の
//! **後** にも失敗しうるため、スロットが busy のまま残り、健全なサーバでも以後の
//! リクエストが "no free KV cache slot (N slots)" で失敗しうる。s144 (20ca091) が
//! admit 失敗経路に解放を追加した。本ファイルはその回帰を固定する。
//!
//! 3 つのテストはそれぞれ別の切断/失敗点を作る (harness は tests/client_disconnect.rs
//! と同一の SlowNoEosModel / test_app / serve / post_raw / wait_in_flight):
//!
//! 1. `repeated_aborts_do_not_leak_kv_slots` — ストリーミングを送り、応答ボディを
//!    読まずに即座に切断。これをスロット数 + 2 回繰り返し、最後に通常リクエストが
//!    完走する (スロットを取得できる) ことを assert。
//! 2. `dropped_stream_receivers_do_not_leak_kv_slots` — 同じことを Scheduler 直操作
//!    (submit_stream → rx drop) で行う決定論的版。hyper の切断検知に依存しない。
//! 3. `failures_after_install_do_not_leak_kv_slots` — install (スロット取得) の後に
//!    forward が必ず Err になるモデルで同じ回数を繰り返す。切断ではない失敗経路。
//!
//! 実測 (REPORT_NEXT.md 参照): s144 修正後のコードでは 3 本とも通り、pre-fix
//! (507e442 の scheduler.rs) では 3 本とも落ちる — ただし pre-fix で落ちる理由は
//! 本物の漏れではなく、本ファイルが `admit` のプリアンブル手前で人工的に
//! post-install 失敗を注入した probe 版で走らせた場合である。注入せずに
//! 公開 API の切断だけで pre-fix を落とすことはできなかった (切断時点で既に
//! `client_gone()` が真 → `abort_disconnected` が解放するため)。
//!
//! スロット使用状況は観測専用のテレメトリ (KvUsage フレーム) から読む —
//! 本番の数値経路・挙動には触れない。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_VOCAB};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_server::{AppState, Scheduler};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout, Instant};

/// EOS を出さず forward ごとに sleep するモデル (client_disconnect.rs と同一)。
struct SlowNoEosModel {
    delay: Duration,
    ctx: usize,
}

/// `install_session` (スロット取得) の **後** に必ず失敗する forward。
///
/// 旧 `admit_job` の漏れ経路は「admit 内で install が成功した後に Err」の 1 点
/// だけだった。SSE プリアンブルの送信失敗 (`events.blocking_send` が Err) は
/// 実際には起こらないので、公開 API でその点を触るには「install の後にエンジン
/// 側で失敗する入力」が必要になる。prefill の forward を Err にすると、スロット
/// 取得済み・client_gone() == false のまま prefill 経路のエラー処理を通る
/// (scheduler.rs の prefill 分岐は s144 以前から release している = ここは
/// 漏れない)。ここで Err になったセッションは decode に進まず、後続の
/// スロット枯渇を観測できる。
struct FailAfterInstallModel {
    ctx: usize,
}

impl Model for FailAfterInstallModel {
    fn context(&self) -> usize {
        self.ctx
    }

    fn eos_ids(&self) -> &[u32] {
        &[]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        anyhow::bail!("injected forward failure (s145 slot-leak probe)")
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.ctx)
    }
}

/// FailAfterInstallModel 用の AppState (test_app と同じ組み立て)。
fn failing_test_app() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(FailAfterInstallModel { ctx: 256 }), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    Arc::new(AppState {
        engine,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

impl Model for SlowNoEosModel {
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
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.ctx)
    }
}

/// client_disconnect.rs の test_app と同一。スケジューラの待ち行列は
/// MAX_ACTIVE_SESSIONS より十分広くとる (切断リクエストがキュー溢れで
/// 503 にならないように) ため 8。
fn test_app(delay_ms: u64) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let model = Box::new(SlowNoEosModel {
        delay: Duration::from_millis(delay_ms),
        ctx: 256,
    });
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    Arc::new(AppState {
        engine,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

async fn serve(state: Arc<AppState>) -> std::net::SocketAddr {
    let app: Router = denno_server::router(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

fn request_json(max_tokens: usize, stream: bool) -> String {
    json!({
        "model": "denno-test",
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "messages": [{"role": "user", "content": "t0"}]
    })
    .to_string()
}

/// 生 HTTP リクエストを送る (Connection: close。応答は自前で読む)。
async fn post_raw(addr: std::net::SocketAddr, body: &str) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s.flush().await.unwrap();
    s
}

/// in_flight が `want` になるまで待つ (制限秒数内に届かなければ panic)。
async fn wait_in_flight(state: &AppState, want: usize, secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if state.scheduler.in_flight() == want {
            return;
        }
        sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "in_flight が {want} にならない ({secs}s 経過・現在 {})",
        state.scheduler.in_flight()
    );
}

fn parse_json_body(buf: &str) -> Value {
    let body = buf
        .split("\r\n\r\n")
        .nth(1)
        .expect("HTTP ヘッダとボディの区切りが無い");
    serde_json::from_str(body).expect("応答ボディが JSON ではない")
}

/// スロット総数 = エンジンの KV キャッシュ数。テスト用のダミーエンジン
/// (KvCache::new(1,4,256)) では既定の予算計算で 32 になる。
/// MAX_ACTIVE_SESSIONS は「待ち行列から同時に pull する上限」でスロット数とは
/// 独立なので、assert ではなく観測値だけ記録する。
fn slot_count(state: &AppState) -> usize {
    state.engine.lock().unwrap().caches.len()
}

/// 観測専用: テレメトリの KvUsage フレームから busy なスロット数を読む。
/// テレメトリは ON/OFF がプロセス全体の原子フラグなので、テストの前後で
/// 元に戻し、溜まったフレームを読み切ってから使う (他テストと同居しないよう
/// このファイルは単一テストのみ)。テレメトリが無効化できない環境 (env 未設定で
/// フレームが出ない等) では None を返し、呼び出し側は assert しない。
fn observe_busy_slots(state: &AppState) -> Option<usize> {
    denno_telemetry::set_enabled_for_tests(true);
    {
        // 生成 (install / release 時) に queued されたフレームを読む。
        // 最後の KvUsage が現在の状態を表す。
        let _g = state.engine.lock().unwrap();
        let mut last = None;
        while let Some(f) = denno_telemetry::try_recv_nowait() {
            if let denno_telemetry::Frame::KvUsage { busy, .. } = f {
                last = Some(busy.iter().filter(|b| **b).count());
            }
        }
        denno_telemetry::set_enabled_for_tests(false);
        last
    }
}

/// 回帰 (本命): install の後に失敗するリクエスト (切断ではなくエンジン側の
/// prefill 失敗) をスロット数より多く繰り返しても、スロットが漏れない。
///
/// forward が必ず Err になるモデルなので、prefill 分岐のエラー処理を通る。
/// 本テストが固定するのは「スロット取得後に失敗したセッションがスロットを返す」
/// 性質 — どの失敗経路かに解放漏れがあれば、スロット数 + 2 回の失敗で枯渇し、
/// "no free KV cache slot" が返る。s144 の修正対象 (admit 内の post-install
/// 失敗) は、公開 API からは下の 2 本の切断経路でしか踏めず、そこでは
/// `client_gone()` が既に真なので pre-fix でも `abort_disconnected` が解放する。
/// この経路を pre-fix で落とすには admit に人工的な失敗を注入するしかなかった
/// (REPORT_NEXT.md に測定値を記載)。
#[tokio::test]
async fn failures_after_install_do_not_leak_kv_slots() {
    let state = failing_test_app();
    let addr = serve(state.clone()).await;
    let slots = slot_count(&state);
    sleep(Duration::from_millis(100)).await;
    let _ = observe_busy_slots(&state);

    let attempts = slots + 2;
    for _ in 0..attempts {
        let mut s = post_raw(addr, &request_json(8, false)).await;
        let mut buf = Vec::new();
        timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
            .await
            .expect("失敗リクエストの応答が返らない")
            .expect("失敗リクエストの応答読み取りが失敗");
        let text = String::from_utf8_lossy(&buf);
        assert!(
            !text.contains("no free KV cache slot"),
            "スロット漏れ ({attempts} 回目の失敗リクエストで枯渇): {text}"
        );
        wait_in_flight(&state, 0, 10).await;
    }

    if let Some(busy) = observe_busy_slots(&state) {
        assert_eq!(busy, 0, "失敗 {attempts} 回の後も busy スロットが {busy} 残っている");
    }

    // 本命の assert: ここまで来た = スロットは返っている。エラー応答 (モデル由来)
    // が返っていることだけ確認する。
    let mut s = post_raw(addr, &request_json(8, false)).await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
        .await
        .expect("応答が返らない")
        .unwrap();
    let text = String::from_utf8_lossy(&buf);
    assert!(
        !text.contains("no free KV cache slot"),
        "スロット漏れ (最終確認): {text}"
    );
}

/// 回帰: 応答を読まずに切断するストリーミングリクエストをスロット数より多く
/// 繰り返しても、その後の通常リクエストがスロットを取得できる
/// ("no free KV cache slot" が出ない)。
#[tokio::test]
async fn repeated_aborts_do_not_leak_kv_slots() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;
    let slots = slot_count(&state);
    // 退避して観測基点をそろえる
    sleep(Duration::from_millis(100)).await;
    let _ = observe_busy_slots(&state);

    let aborts = slots + 2;
    for _ in 0..aborts {
        // ストリーミングを送り、応答ボディを一切読まずに即座に切断する。
        // ワーカーが admit を評価する時点で events の受信側 (ReceiverStream) は
        // 既に drop されている → client_gone() == true の最速切断点。
        let stream = post_raw(addr, &request_json(400, true)).await;
        // ワーカーが 1 回の pull で同時に受けるのは MAX_ACTIVE_SESSIONS 件まで
        // (残りは待ち行列)。この回のジョブを確実に pull させるため、送信前に
        // in_flight が 0 に戻るのを待つ — 待たないとキュー溢れ (503) と本命の
        // スロット漏れが混ざる。
        wait_in_flight(&state, 0, 10).await;
        drop(stream);
        // 回収 (abort + 解放) が終わるまで待つ
        wait_in_flight(&state, 0, 10).await;
    }

    // 漏れの直接観測 (テレメトリが読めた場合のみ): busy スロットが 0。
    if let Some(busy) = observe_busy_slots(&state) {
        assert_eq!(busy, 0, "切断 {aborts} 回の後も busy スロットが {busy} 残っている");
    }

    // 本命: 通常リクエストがスロットを取得して完走する
    let mut s = post_raw(addr, &request_json(8, false)).await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(30), s.read_to_end(&mut buf))
        .await
        .expect("通常リクエストが完了しない (スロット枯渇?)")
        .unwrap();
    let text = String::from_utf8_lossy(&buf);
    assert!(
        !text.contains("no free KV cache slot"),
        "スロット漏れ: {text}"
    );
    let v = parse_json_body(&text);
    assert_eq!(v["type"], "message", "通常リクエストがエラー: {text}");
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(v["usage"]["output_tokens"], 8);
    assert_eq!(state.scheduler.in_flight(), 0);
}

/// 参考: Scheduler 直操作で Receiver を drop する経路 (HTTP のキャンセル検知に
/// 依存しない決定論的版)。切断したジョブをスロット数より多く積み、最後に
/// 通常ジョブが完走することを確かめる。
#[tokio::test]
async fn dropped_stream_receivers_do_not_leak_kv_slots() {
    let state = test_app(80);
    let slots = slot_count(&state);
    sleep(Duration::from_millis(100)).await;
    let _ = observe_busy_slots(&state);

    // 最後に観測した busy 数を残す (失敗時のメッセージ用)
    let mut last_busy = None;
    for _ in 0..(slots + 2) {
        // 前回のジョブの回収 (abort + 解放) と、submit_stream 内部の build_run の
        // キュー投入の両方が完了するまで待つ。待ち行列長は 8 でスロット数より
        // 小さいため、待たずに積むと QueueFull になる。
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            wait_in_flight(&state, 0, 10).await;
            let (tx, rx) =
                tokio::sync::mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(64);
            match state.scheduler.submit_stream(
                None,
                vec![ChatMessage { role: "user".into(), content: "t0".into() }],
                GenParams { max_tokens: 400, temperature: Some(0.0), ..Default::default() },
                None,
                None,
                None,
                tx,
            ) {
                Ok(()) => {
                    drop(rx); // 応答ボディ drop 相当
                    break;
                }
                Err(_) if Instant::now() < deadline => {
                    // ワーカーが accept 待ちから戻るまで僅かに待って再試行
                    sleep(Duration::from_millis(5)).await;
                }
                Err(_) => panic!("submit_stream が QueueFull のまま (ワーカー停止?)"),
            }
        }
        wait_in_flight(&state, 0, 10).await;
        if let Some(b) = observe_busy_slots(&state) {
            last_busy = Some(b);
        }
    }

    if let Some(busy) = last_busy {
        assert_eq!(busy, 0, "busy スロットが {busy} 残っている");
    }

    let resp_rx = state
        .scheduler
        .submit_chat(
            None,
            vec![ChatMessage { role: "user".into(), content: "t0".into() }],
            GenParams { max_tokens: 4, temperature: Some(0.0), ..Default::default() },
            None,
            None,
            None,
        )
        .unwrap();
    let out = timeout(Duration::from_secs(30), resp_rx)
        .await
        .expect("通常ジョブが完了しない (スロット枯渇?)")
        .expect("通常ジョブの応答チャネルが落ちた (スロット枯渇?)")
        .expect("通常ジョブがエラー (スロット枯渇?)");
    assert_eq!(out.output_tokens, 4);
    assert_eq!(state.scheduler.in_flight(), 0);
}