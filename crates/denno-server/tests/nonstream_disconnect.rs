//! s44: 非ストリーム /v1/messages のクライアント切断後処理の堅牢化。
//!
//! 検知経路: hyper は応答 future が pending の間も接続の read 側を監視しており、
//! クライアント切断を検知すると接続タスクごと handler future をキャンセル (drop)
//! する (スクラッチ実験で確定: 切断後に handler の完了フラグが立たない)。
//! handler は respond (oneshot) の Receiver を保持しているため、キャンセルされると
//! Sender 側の is_closed() が true になる → SSE (s38) と同一の sweep で該当 Run を
//! 打ち切る (KV スロット解放・他 Run への影響無し)。
//!
//! 検証する内容:
//! 1. 生成中の切断で非ストリーム Run も打ち切られる (完了を待たず in_flight が 0 に戻る)
//! 2. 受信側が先に消えたジョブ (Receiver drop) も生成無しで打ち切られる
//! 3. 健常な非ストリームリクエストは最後まで完了する (誤 abort しない)

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
/// ctx 256 × delay 80ms で max_tokens をクランプまで伸ばし、
/// 「切断しなければ完了しない長さの生成」を作る。
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

/// 生 HTTP リクエストを送る (Connection: close)。
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

/// /health に GET して 200 {"status":"ok"} が返ることを確認する
/// (新規接続を受け付け続けていることの確認)。
async fn health_ok(addr: std::net::SocketAddr) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    s.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut buf = Vec::new();
    timeout(Duration::from_secs(10), s.read_to_end(&mut buf))
        .await
        .expect("/health が完了しない (接続受付が停止?)")
        .unwrap();
    let text = String::from_utf8_lossy(&buf);
    assert!(text.contains("200"), "/health が 200 でない: {text}");
    assert!(text.contains("\"status\":\"ok\""), "/health 異常: {text}");
}

/// 後続の非ストリームリクエストが完了することを確認する (ワーカー / KV 健常)。
async fn followup_chat_ok(addr: std::net::SocketAddr) {
    let mut s = post_raw(addr, &request_json(8, false)).await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
        .await
        .expect("後続リクエストが完了しない")
        .unwrap();
    let v = parse_json_body(&String::from_utf8_lossy(&buf));
    assert_eq!(v["type"], "message");
    assert_eq!(v["stop_reason"], "max_tokens");
}

/// 生成中 (decode 進行中) のクライアント切断で非ストリーム Run も打ち切られる。
/// 修正前は生成完了 (200 token × 80ms = 16 秒) まで in_flight が 1 のまま残り、
/// 単一ワーカーを塞いで後続リクエストが全て列待ちになった。
#[tokio::test]
async fn disconnect_during_nonstream_generation_aborts_run() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    // 非ストリームで長めの生成を開始させ、進行中に切断する
    // (200 token × 80ms ≒ 16 秒。打ち切られなければ 5 秒では完了しない)
    let mut stream = post_raw(addr, &request_json(200, false)).await;
    sleep(Duration::from_millis(300)).await; // admit + prefill/decode 進行を待つ
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(stream); // クライアント切断 → hyper が handler future をキャンセル

    // 生成完了を待たずに打ち切られる (in_flight が 0 に戻る)
    wait_in_flight(&state, 0, 5).await;

    // サーバー健常の確認 (新規接続受付 + 後続リクエスト)
    health_ok(addr).await;
    followup_chat_ok(addr).await;
    assert_eq!(state.scheduler.in_flight(), 0);
}

/// HTTP を介さず Scheduler 直操作で「受信側が先に消えたジョブ」の打ち切りを検証
/// (決定論的。hyper のキャンセルを待たず Receiver drop を直接再現)。
#[tokio::test]
async fn dropped_chat_receiver_aborts_run_without_generation() {
    let tok = dummy_tokenizer().unwrap();
    let model = Box::new(SlowNoEosModel {
        delay: Duration::from_millis(80),
        ctx: 256,
    });
    let engine = Arc::new(Mutex::new(Engine::from_parts(model, tok, "denno-test", None)));
    let scheduler = Scheduler::spawn(engine.clone(), 8);

    let rx = scheduler
        .submit_chat(
            None,
            vec![ChatMessage { role: "user".into(), content: "t0".into() }],
            GenParams { max_tokens: 400, temperature: Some(0.0), ..Default::default() },
            None,
            None,
            None,
        )
        .unwrap();
    drop(rx); // クライアント切断相当 (hyper による handler キャンセル → Receiver drop)

    // ワーカーが pull しても生成は完了しない (400 token × 80ms ≒ 32s 超) —
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

/// 正常系 (regression): 受信側が存在する非ストリームジョブは最後まで完了し、
/// 誤 abort されない。応答も通常どおり受信できる。
#[tokio::test]
async fn healthy_nonstream_request_completes() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    // Scheduler 直操作: Receiver を保持したまま完了を待つ
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
    let out = timeout(Duration::from_secs(20), resp_rx)
        .await
        .expect("ジョブが完了しない (誤 abort?)")
        .unwrap()
        .unwrap();
    assert_eq!(out.output_tokens, 4);
    assert_eq!(state.scheduler.in_flight(), 0);

    // HTTP 経路でも応答が最後まで受信できる
    let mut s = post_raw(addr, &request_json(8, false)).await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
        .await
        .expect("応答が完了しない")
        .unwrap();
    let v = parse_json_body(&String::from_utf8_lossy(&buf));
    assert_eq!(v["type"], "message");
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(v["usage"]["output_tokens"], 8);
    assert_eq!(state.scheduler.in_flight(), 0);
}
