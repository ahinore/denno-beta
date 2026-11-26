//! task 1604: SSE ストリームのクライアント切断 (途中 drop / 部分 chunk drop) の
//! 境界テスト。OpenAI (`/v1/chat/completions`) と Anthropic (`/v1/messages`) の
//! 両経路で、生成途中の切断が Run を打ち切り、KV スロットを漏らさず、後続の
//! リクエストが正常に完了することを検証する。
//!
//! 検知経路は tests/client_disconnect.rs と同じ — クライアントがソケットを
//! close すると hyper が SSE 書き込み時に切断を検知して応答ボディを drop し、
//! events / raw チャネルが閉鎖 → ワーカーが該当 Run を abort する。
//!
//! 決定論性: 固定 sleep で挙動を作らず、`in_flight()` / `stats()` /
//! `slot_busy_count()` を期限付きでポーリングして状態遷移を観測する。
//! 生成モデルは EOS を出さず 1 forward ごとに短い delay を挟むため、切断しなければ
//! 完了しない長さの生成になる (切断の効果だけを観測できる)。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout, Instant};

/// EOS を出さず forward ごとに sleep するモデル (client_disconnect.rs と同一契約)。
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
        // 需要ベース確保 (書き込み前に容量確保) のダミー側契約
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

/// テスト用 AppState (delay 20ms × ctx 256 ≒ 5s の生成)。
fn test_app() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let model = Box::new(SlowNoEosModel {
        delay: Duration::from_millis(20),
        ctx: 256,
    });
    let engine = Engine::from_parts(model, tok, "denno-test", None);
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
async fn post_raw(addr: std::net::SocketAddr, path: &str, body: &str) -> TcpStream {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
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

/// `marker` まで 1 バイトずつ読み、さらに `n` バイトだけ読んで止める —
/// SSE フレームの途中 (chunk / event が閉じる前) で読みを止めた状態を作る。
/// ソケット読みを 1 バイトに絞ることで、OS バッファの先読みに依存せず
/// 「部分フレーム」を決定的に作れる。成功したら true。
async fn read_marker_then_fragment(
    stream: &mut TcpStream,
    buf: &mut String,
    marker: &str,
    n: usize,
    secs: u64,
) -> bool {
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut byte = [0u8; 1];
    while !buf.contains(marker) {
        let remain = deadline.saturating_duration_since(Instant::now());
        if remain.is_zero() {
            return false;
        }
        match timeout(remain, stream.read(&mut byte)).await {
            Ok(Ok(1)) => buf.push(byte[0] as char),
            _ => return false,
        }
    }
    for _ in 0..n {
        let remain = deadline.saturating_duration_since(Instant::now());
        if remain.is_zero() {
            return false;
        }
        match timeout(remain, stream.read(&mut byte)).await {
            Ok(Ok(1)) => buf.push(byte[0] as char),
            _ => return false,
        }
    }
    true
}

/// in_flight が `want` になるまで期限付きで待つ (届かなければ panic)。
async fn wait_in_flight(state: &AppState, want: usize, secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    while Instant::now() < deadline {
        if state.scheduler.in_flight() == want {
            return;
        }
        sleep(Duration::from_millis(5)).await;
    }
    panic!(
        "in_flight が {want} にならない ({secs}s 経過・現在 {})",
        state.scheduler.in_flight()
    );
}

/// busy KV スロット数が `want` になるまで期限付きで待つ。
async fn wait_slot_busy(state: &AppState, want: usize, secs: u64) {
    let deadline = Instant::now() + Duration::from_secs(secs);
    loop {
        let busy = state.engine.lock().unwrap().slot_busy_count();
        if busy == want {
            return;
        }
        if Instant::now() >= deadline {
            panic!("busy スロットが {want} にならない ({secs}s 経過・現在 {busy})");
        }
        sleep(Duration::from_millis(5)).await;
    }
}

fn parse_json_body(buf: &str) -> Value {
    let body = buf
        .split("\r\n\r\n")
        .nth(1)
        .expect("HTTP ヘッダとボディの区切りが無い");
    serde_json::from_str(body).expect("応答ボディが JSON ではない")
}

/// 後続の非ストリーム OpenAI リクエストが完走することを確認する。
async fn assert_openai_followup_ok(addr: std::net::SocketAddr, state: &AppState) {
    let mut s = post_raw(addr, "/v1/chat/completions", &request_json(8, false)).await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
        .await
        .expect("後続 OpenAI リクエストが完了しない")
        .unwrap();
    let text = String::from_utf8_lossy(&buf);
    assert!(
        !text.contains("no free KV cache slot"),
        "スロット枯渇: {text}"
    );
    let v = parse_json_body(&text);
    assert_eq!(v["object"], "chat.completion", "後続応答がエラー: {text}");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 8);
    wait_in_flight(state, 0, 10).await;
}

/// 後続の非ストリーム Anthropic リクエストが完走することを確認する。
async fn assert_anthropic_followup_ok(addr: std::net::SocketAddr, state: &AppState) {
    let mut s = post_raw(addr, "/v1/messages", &request_json(8, false)).await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
        .await
        .expect("後続 Anthropic リクエストが完了しない")
        .unwrap();
    let text = String::from_utf8_lossy(&buf);
    assert!(
        !text.contains("no free KV cache slot"),
        "スロット枯渇: {text}"
    );
    let v = parse_json_body(&text);
    assert_eq!(v["type"], "message", "後続応答がエラー: {text}");
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(v["usage"]["output_tokens"], 8);
    wait_in_flight(state, 0, 10).await;
}

// ---------------------------------------------------------------------------

/// OpenAI SSE: 最初の chunk を受けた直後に切断すると Run が打ち切られ、
/// KV スロットが返り、後続リクエストが完走する。
#[tokio::test]
async fn openai_client_drop_mid_stream_aborts_and_recovers() {
    let state = test_app();
    let addr = serve(state.clone()).await;
    let aborted_before = state.scheduler.stats().aborted_disconnected;

    let mut s = post_raw(addr, "/v1/chat/completions", &request_json(400, true)).await;
    let mut buf = String::new();
    assert!(
        read_until(&mut s, &mut buf, "chat.completion.chunk", 10).await,
        "OpenAI chunk が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(s); // クライアント切断

    wait_in_flight(&state, 0, 10).await;
    wait_slot_busy(&state, 0, 10).await;
    assert!(
        state.scheduler.stats().aborted_disconnected > aborted_before,
        "切断が aborted_disconnected に計上されていない"
    );

    assert_openai_followup_ok(addr, &state).await;
}

/// Anthropic SSE: 最初の delta を受けた直後の切断でも同様に打ち切られる。
#[tokio::test]
async fn anthropic_client_drop_mid_stream_aborts_and_recovers() {
    let state = test_app();
    let addr = serve(state.clone()).await;
    let aborted_before = state.scheduler.stats().aborted_disconnected;

    let mut s = post_raw(addr, "/v1/messages", &request_json(400, true)).await;
    let mut buf = String::new();
    assert!(
        read_until(&mut s, &mut buf, "content_block_delta", 10).await,
        "Anthropic delta が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(s);

    wait_in_flight(&state, 0, 10).await;
    wait_slot_busy(&state, 0, 10).await;
    assert!(
        state.scheduler.stats().aborted_disconnected > aborted_before,
        "切断が aborted_disconnected に計上されていない"
    );

    assert_anthropic_followup_ok(addr, &state).await;
}

/// OpenAI SSE: SSE フレーム (chunk) が閉じる前に読むのを止めて切断しても
/// 打ち切られる (部分 chunk drop)。
#[tokio::test]
async fn openai_partial_chunk_drop_aborts_and_recovers() {
    let state = test_app();
    let addr = serve(state.clone()).await;

    let mut s = post_raw(addr, "/v1/chat/completions", &request_json(400, true)).await;
    let mut buf = String::new();
    assert!(
        read_marker_then_fragment(&mut s, &mut buf, "data: ", 8, 10).await,
        "chunk の先頭フラグメントが届かない: {buf}"
    );
    // 最初の chunk は閉じていない (JSON の途中で読みを止めた) ことを確認する。
    let after_data = buf.split("data: ").nth(1).unwrap_or("");
    assert!(
        !after_data.contains("}\n\n") && !after_data.contains("}\r\n\r\n"),
        "chunk が閉じるまで読んでしまった (部分 chunk になっていない): {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(s);

    wait_in_flight(&state, 0, 10).await;
    wait_slot_busy(&state, 0, 10).await;

    assert_openai_followup_ok(addr, &state).await;
}

/// Anthropic SSE: イベントフレームの途中で切断しても打ち切られる。
#[tokio::test]
async fn anthropic_partial_event_drop_aborts_and_recovers() {
    let state = test_app();
    let addr = serve(state.clone()).await;

    let mut s = post_raw(addr, "/v1/messages", &request_json(400, true)).await;
    let mut buf = String::new();
    assert!(
        read_marker_then_fragment(&mut s, &mut buf, "event: ", 4, 10).await,
        "イベントの先頭フラグメントが届かない: {buf}"
    );
    // イベントフレームが終端 (\n\n) に達する前に読みを止めたことを確認する。
    let after_event = buf.split("event: ").nth(1).unwrap_or("");
    assert!(
        !after_event.contains("\n\n") && !after_event.contains("\r\n\r\n"),
        "イベントが閉じるまで読んでしまった (部分フレームになっていない): {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(s);

    wait_in_flight(&state, 0, 10).await;
    wait_slot_busy(&state, 0, 10).await;

    assert_anthropic_followup_ok(addr, &state).await;
}

/// 両プロトコルの途中切断をスロット数 + 2 回繰り返しても KV スロットが
/// 漏れない ("no free KV cache slot" が出ず、busy スロットが 0 に戻る)。
#[tokio::test]
async fn cross_protocol_drops_do_not_leak_kv_slots() {
    let state = test_app();
    let addr = serve(state.clone()).await;
    let slots = state.engine.lock().unwrap().slot_count();
    let aborted_before = state.scheduler.stats().aborted_disconnected;

    for i in 0..(slots + 2) {
        let (path, marker) = if i % 2 == 0 {
            ("/v1/chat/completions", "chat.completion.chunk")
        } else {
            ("/v1/messages", "\"type\":\"message_start\"")
        };
        let mut s = post_raw(addr, path, &request_json(400, true)).await;
        let mut buf = String::new();
        assert!(
            read_until(&mut s, &mut buf, marker, 10).await,
            "{path} の最初のイベントが届かない ({i} 回目): {buf}"
        );
        drop(s);
        wait_in_flight(&state, 0, 10).await;
        wait_slot_busy(&state, 0, 10).await;
    }

    let aborted = state.scheduler.stats().aborted_disconnected - aborted_before;
    assert!(
        aborted >= slots + 2,
        "切断 {slots} + 2 回のうち aborted_disconnected が {aborted} 回しか計上されていない"
    );
    assert_eq!(state.engine.lock().unwrap().slot_busy_count(), 0);

    // 両プロトコルで後続リクエストが完走する。
    assert_openai_followup_ok(addr, &state).await;
    assert_anthropic_followup_ok(addr, &state).await;
}
