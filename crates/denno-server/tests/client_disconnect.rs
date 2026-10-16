//! s38: クライアント切断検知 (SSE ストリームの生成 abort) の統合テスト。
//!
//! 実ソケット (TCP) でサーバーに接続し、生成中に接続を切断して Run が打ち切られる
//! (in_flight が 0 に戻る・後続リクエストが即座に完了する) ことを検証する。
//!
//! 検知経路: クライアントが接続を切ると hyper は SSE イベント / keep-alive ping の
//! 書き込み時に broken pipe / connection reset を検知し応答ボディを drop する →
//! events チャネルが閉鎖 → ワーカーの切断スイープ / decode ステップ後の判定が
//! 該当 Run を abort する (KV スロット解放・ログ 1 行)。
//!
//! 非ストリーム (Job::Chat) はレスポンス送信まで接続状態を観測する手段が無いため
//! 対象外 (HISTORY §3 参照)。

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

/// EOS を出さず forward ごとに sleep するモデル。
/// logits は DummyModel と同一 (決定論) だが eos_ids が空のため
/// max_tokens (コンテキストclamp) まで生成が続く —
/// 「切断しなければ完了しない長さの生成」を作って abort の効果を観測する。
struct SlowNoEosModel {
    delay: Duration,
    ctx: usize,
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
        // P17e: 需要ベース確保 (書き込み前に容量確保) のダミー側契約
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

/// テスト用 AppState を構築する (1 forward = delay_ms の生成で max_tokens を
/// クランプまで伸ばす。ctx 256 × delay 80ms → 切断しないと完了しない長さ)。
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
    })
}

/// 実 TCP ポートで serve する (バックグラウンドタスク)。
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

/// `marker` が現れるまで応答を読む (制限秒数内に現れなければ false)。
async fn read_until(stream: &mut TcpStream, buf: &mut String, marker: &str, secs: u64) -> bool {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut tmp = [0u8; 4096];
    while !buf.contains(marker) {
        let remain = deadline.saturating_duration_since(Instant::now());
        if remain.is_zero() {
            return false;
        }
        match timeout(remain, stream.read(&mut tmp)).await {
            Ok(Ok(0)) | Ok(Err(_)) | Err(_) => return false,
            Ok(Ok(n)) => buf.push_str(&String::from_utf8_lossy(&tmp[..n])),
        }
    }
    true
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

/// 応答バッファから非ストリーム JSON ボディを取り出してパースする。
fn parse_json_body(buf: &str) -> Value {
    let body = buf
        .split("\r\n\r\n")
        .nth(1)
        .expect("HTTP ヘッダとボディの区切りが無い");
    serde_json::from_str(body).expect("応答ボディが JSON ではない")
}

// ---------------------------------------------------------------------------

/// 生成中 (decode ステップ進行中) のクライアント切断で生成が打ち切られる。
/// 修正前は生成完了 (~245 token × delay) まで in_flight が 1 のまま残り、
/// 単一ワーカーを塞いで後続リクエストが全て列待ちになる。
#[tokio::test]
async fn disconnect_during_generation_aborts_run() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    // 接続して生成を開始させ、最初の delta が流れたところで切断する
    let mut stream = post_raw(addr, &request_json(400, true)).await;
    let mut buf = String::new();
    assert!(
        read_until(&mut stream, &mut buf, "event: message_start", 10).await,
        "message_start が届かない: {buf}"
    );
    assert!(
        read_until(&mut stream, &mut buf, "content_block_delta", 10).await,
        "delta が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(stream); // クライアント切断 (ソケット close)

    // 生成が打ち切られて in_flight が 0 に戻る
    wait_in_flight(&state, 0, 5).await;

    // 後続リクエストが即座に完了する (ワーカー / KV スロットが健常)
    let mut s2 = post_raw(addr, &request_json(8, false)).await;
    let mut buf2 = Vec::new();
    timeout(Duration::from_secs(20), s2.read_to_end(&mut buf2))
        .await
        .expect("後続リクエストが完了しない")
        .unwrap();
    let v = parse_json_body(&String::from_utf8_lossy(&buf2));
    assert_eq!(v["type"], "message");
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(state.scheduler.in_flight(), 0);
}

/// 生成開始前 (SSE プリアンブル後・prefill 中) の切断も打ち切られる。
#[tokio::test]
async fn disconnect_before_first_token_aborts_run() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    // message_start (admit 完了 = ジョブ pull 後) を受けてから切断する
    let mut stream = post_raw(addr, &request_json(400, true)).await;
    let mut buf = String::new();
    assert!(
        read_until(&mut stream, &mut buf, "event: message_start", 10).await,
        "message_start が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(stream);

    // 最初の delta が届く前に打ち切られる (prefill 完了まで待たない)
    wait_in_flight(&state, 0, 5).await;

    // サーバー健常の確認 (後続リクエストが完了できる)
    let mut s2 = post_raw(addr, &request_json(8, false)).await;
    let mut buf2 = Vec::new();
    timeout(Duration::from_secs(20), s2.read_to_end(&mut buf2))
        .await
        .expect("後続リクエストが完了しない")
        .unwrap();
    assert_eq!(parse_json_body(&String::from_utf8_lossy(&buf2))["type"], "message");
}

/// HTTP を介さず Scheduler 直操作で「受信側が先に消えたジョブ」の打ち切りを検証
/// (待ち行列 pull 時 / admit 時の切断経路。応答ボディ drop = 受信 drop を直接再現)。
#[tokio::test]
async fn dropped_receiver_aborts_run_without_generation() {
    let tok = dummy_tokenizer().unwrap();
    let model = Box::new(SlowNoEosModel {
        delay: Duration::from_millis(80),
        ctx: 256,
    });
    let engine = Arc::new(Mutex::new(Engine::from_parts(model, tok, "denno-test", None)));
    let scheduler = Scheduler::spawn(engine.clone(), 8);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<axum::response::sse::Event, std::convert::Infallible>>(64);
    scheduler
        .submit_stream(
            None,
            vec![ChatMessage { role: "user".into(), content: "t0".into() }],
            GenParams { max_tokens: 400, temperature: Some(0.0), ..Default::default() },
            None,
            None,
            tx,
        )
        .unwrap();
    drop(rx); // クライアント切断相当 (応答ボディ drop → チャネル閉鎖)

    // ワーカーが pull しても生成は完了しない (400 token × 80ms ≒ 30s 超) —
    // 短時間で in_flight == 0 に戻る = pull/admit 時点で打ち切られた証拠
    sleep(Duration::from_millis(300)).await;
    assert_eq!(scheduler.in_flight(), 0);

    // 打ち切り後もスケジューラは健常 (通常ジョブが完走する)
    let resp_rx = scheduler
        .submit_chat(
            None,
            vec![ChatMessage { role: "user".into(), content: "t0".into() }],
            GenParams { max_tokens: 4, temperature: Some(0.0), ..Default::default() },
            None,
            None,
        )
        .unwrap();
    let out = timeout(Duration::from_secs(20), resp_rx)
        .await
        .expect("通常ジョブが完了しない")
        .unwrap()
        .unwrap();
    assert_eq!(out.output_tokens, 4);
    assert_eq!(scheduler.in_flight(), 0);
}

/// 正常系: 切断スイープが常時動いていても、接続を維持したリクエストは
/// 最後まで完了する (誤 abort しない)。
#[tokio::test]
async fn healthy_stream_request_is_not_aborted() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    let mut stream = post_raw(addr, &request_json(8, true)).await;
    let mut buf = String::new();
    assert!(
        read_until(&mut stream, &mut buf, "event: message_stop", 20).await,
        "message_stop が届かない: {buf}"
    );
    assert!(buf.contains("content_block_start"));
    assert!(buf.contains("content_block_delta"));
    assert!(buf.contains("message_delta"));
    // delta の結合が空でない (生成テキストが欠落していない)
    let text: String = buf
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter(|v| v["type"] == "content_block_delta")
        .filter_map(|v| v["delta"]["text"].as_str().map(str::to_string))
        .collect();
    assert!(!text.is_empty(), "delta テキストが空: {buf}");
    assert_eq!(state.scheduler.in_flight(), 0);
}
