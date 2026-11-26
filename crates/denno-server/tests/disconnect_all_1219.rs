//! Task 1219: 3 プロトコル共通の SSE / NDJSON クライアント切断後処理の横断テスト。
//!
//! 既存の切断テストは Anthropic `/v1/messages` (tests/client_disconnect.rs) と
//! 非ストリーム (tests/nonstream_disconnect.rs) に限られていた。ここでは
//! OpenAI `/v1/chat/completions` (SSE) と Ollama `/api/chat` (NDJSON) も含め、
//! 「ストリーム途中でクライアントが TCP を切る」と同じ経路で:
//!
//! 1. in_flight が 0 に戻る (Run が打ち切られる)
//! 2. `aborted_disconnected` がちょうど 1 増える (他計数は動かない)
//! 3. KV スロットが漏れない (`slot_busy_count() == 0`)
//! 4. 後続リクエストが即座に完了する (ワーカーが塞がっていない)
//!
//! を検証する。3 プロトコルは別々の `AppState` で実行するので、
//! 各テストの計数は独立している。

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

/// EOS を出さず forward ごとに sleep するモデル (既存切断テストと同一契約)。
/// ctx 256 × delay 80ms で max_tokens をクランプまで伸ばし、
/// 「切断しなければ数秒では完了しない生成」を作る。
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

/// 生 HTTP POST (Connection: close)。応答は自前で読む。
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

/// 述語が応答バッファに対して真になるまで読む (制限秒数内に満たなければ false)。
async fn read_until<F>(stream: &mut TcpStream, buf: &mut String, pred: F, secs: u64) -> bool
where
    F: Fn(&str) -> bool,
{
    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut tmp = [0u8; 4096];
    while !pred(buf) {
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

/// バッファ内の完結した行だけを返す (末尾の途中行は捨てる)。
fn complete_lines(buf: &str) -> impl Iterator<Item = &str> {
    let mut lines: Vec<&str> = buf.split('\n').collect();
    if !buf.ends_with('\n') {
        lines.pop();
    }
    lines.into_iter()
}

/// OpenAI SSE: `choices[0].delta.content` が非空の chunk が来たか。
fn openai_delta_seen(buf: &str) -> bool {
    complete_lines(buf)
        .filter_map(|l| l.trim_end_matches('\r').strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .any(|v| {
            v["choices"][0]["delta"]["content"]
                .as_str()
                .map(|t| !t.is_empty())
                .unwrap_or(false)
        })
}

/// Ollama NDJSON: `done:false` かつ `message.content` が非空の行が来たか。
fn ollama_delta_seen(buf: &str) -> bool {
    complete_lines(buf)
        .filter_map(|l| serde_json::from_str::<Value>(l.trim_end_matches('\r')).ok())
        .filter(|v| v["done"] == json!(false))
        .any(|v| {
            v["message"]["content"]
                .as_str()
                .map(|t| !t.is_empty())
                .unwrap_or(false)
        })
}

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

/// 切断後の共通検査: 計数 (aborted_disconnected == 1・他 0)、スロット漏れ無し、
/// 後続の非ストリームリクエストが完了する。
async fn assert_clean_abort(state: &AppState, addr: std::net::SocketAddr) {
    wait_in_flight(state, 0, 5).await;

    let st = state.scheduler.stats();
    assert_eq!(
        st.aborted_disconnected, 1,
        "切断打ち切りが 1 回でない: {st:?}"
    );
    assert_eq!(st.completed, 0, "切断が完了として数えられた: {st:?}");
    assert_eq!(st.aborted_timeout, 0, "期限超過として数えられた: {st:?}");
    assert_eq!(st.aborted_error, 0, "エラーとして数えられた: {st:?}");
    assert_eq!(state.scheduler.pending(), 0, "キューに残骸がある");
    assert_eq!(
        state.engine.lock().unwrap().slot_busy_count(),
        0,
        "切断で KV スロットが漏れている"
    );

    // 後続リクエスト (Anthropic 非ストリーム) が即座に完走する。
    let mut s = post_raw(
        addr,
        "/v1/messages",
        &json!({
            "model": "denno-test",
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": false,
            "messages": [{"role": "user", "content": "t0"}]
        })
        .to_string(),
    )
    .await;
    let mut buf = Vec::new();
    timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
        .await
        .expect("後続リクエストが完了しない (ワーカーが塞がっている)")
        .unwrap();
    let v = parse_json_body(&String::from_utf8_lossy(&buf));
    assert_eq!(v["type"], "message");
    assert_eq!(v["stop_reason"], "max_tokens");
    assert_eq!(v["usage"]["output_tokens"], 8);
    assert_eq!(state.scheduler.stats().completed, 1);
    assert_eq!(
        state.engine.lock().unwrap().slot_busy_count(),
        0,
        "後続リクエスト完了後もスロットが busy"
    );
}

/// Anthropic SSE (`/v1/messages`, stream=true) の生成途中切断。
#[tokio::test]
async fn anthropic_sse_disconnect_midstream_cleans_up() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    let mut stream = post_raw(
        addr,
        "/v1/messages",
        &json!({
            "model": "denno-test",
            "max_tokens": 400,
            "temperature": 0.0,
            "stream": true,
            "messages": [{"role": "user", "content": "t0"}]
        })
        .to_string(),
    )
    .await;
    let mut buf = String::new();
    assert!(
        read_until(
            &mut stream,
            &mut buf,
            |b| b.contains("event: message_start"),
            10
        )
        .await,
        "message_start が届かない: {buf}"
    );
    assert!(
        read_until(
            &mut stream,
            &mut buf,
            |b| b.contains("content_block_delta"),
            10
        )
        .await,
        "delta が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(stream); // ソケット close = クライアント切断

    assert_clean_abort(&state, addr).await;
}

/// OpenAI SSE (`/v1/chat/completions`, stream=true) の生成途中切断。
/// 最初の delta (非空 content) を受けてから切る。
#[tokio::test]
async fn openai_sse_disconnect_midstream_cleans_up() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    let mut stream = post_raw(
        addr,
        "/v1/chat/completions",
        &json!({
            "model": "denno-test",
            "max_tokens": 400,
            "temperature": 0.0,
            "stream": true,
            "messages": [{"role": "user", "content": "t0"}]
        })
        .to_string(),
    )
    .await;
    let mut buf = String::new();
    assert!(
        read_until(&mut stream, &mut buf, openai_delta_seen, 10).await,
        "content delta が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(stream);

    assert_clean_abort(&state, addr).await;
}

/// Ollama NDJSON (`/api/chat`, stream 既定 true) の生成途中切断。
#[tokio::test]
async fn ollama_ndjson_disconnect_midstream_cleans_up() {
    let state = test_app(80);
    let addr = serve(state.clone()).await;

    let mut stream = post_raw(
        addr,
        "/api/chat",
        &json!({
            "model": "denno-test",
            "stream": true,
            "messages": [{"role": "user", "content": "t0"}],
            "options": {"temperature": 0.0, "num_predict": 400}
        })
        .to_string(),
    )
    .await;
    let mut buf = String::new();
    assert!(
        read_until(&mut stream, &mut buf, ollama_delta_seen, 10).await,
        "NDJSON の delta 行が届かない: {buf}"
    );
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(stream);

    assert_clean_abort(&state, addr).await;
}

/// 3 プロトコルを同じサーバー (別 `AppState` ではなく 1 つ) で連続して切断し、
/// 計数が 3 で止まり・スロットが漏れず・最後に全経路が健常であることを確認する。
#[tokio::test]
async fn three_protocols_sequential_disconnect_no_leak() {
    let state = test_app(60);
    let addr = serve(state.clone()).await;

    // (path, body, delta 判定)
    let cases: [(&str, Value, fn(&str) -> bool); 3] = [
        (
            "/v1/messages",
            json!({
                "model": "denno-test", "max_tokens": 400, "temperature": 0.0, "stream": true,
                "messages": [{"role": "user", "content": "t0"}]
            }),
            |b| b.contains("content_block_delta"),
        ),
        (
            "/v1/chat/completions",
            json!({
                "model": "denno-test", "max_tokens": 400, "temperature": 0.0, "stream": true,
                "messages": [{"role": "user", "content": "t0"}]
            }),
            openai_delta_seen,
        ),
        (
            "/api/chat",
            json!({
                "model": "denno-test", "stream": true,
                "messages": [{"role": "user", "content": "t0"}],
                "options": {"temperature": 0.0, "num_predict": 400}
            }),
            ollama_delta_seen,
        ),
    ];

    for (i, (path, body, pred)) in cases.into_iter().enumerate() {
        let mut stream = post_raw(addr, path, &body.to_string()).await;
        let mut buf = String::new();
        assert!(
            read_until(&mut stream, &mut buf, pred, 10).await,
            "{path}: delta が届かない: {buf}"
        );
        assert_eq!(state.scheduler.in_flight(), 1, "{path}: 実行中でない");
        drop(stream);
        wait_in_flight(&state, 0, 5).await;

        let st = state.scheduler.stats();
        assert_eq!(
            st.aborted_disconnected,
            i + 1,
            "{path}: 切断計数が想定と違う: {st:?}"
        );
        assert_eq!(st.completed, 0, "{path}: 完了計数が動いた: {st:?}");
        assert_eq!(
            state.engine.lock().unwrap().slot_busy_count(),
            0,
            "{path}: スロットが漏れている"
        );
    }

    // 3 経路すべてが引き続き応答できる。
    for path in ["/v1/messages", "/v1/chat/completions", "/api/chat"] {
        let body = if path == "/api/chat" {
            json!({
                "model": "denno-test", "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
                "options": {"temperature": 0.0, "num_predict": 8}
            })
        } else {
            json!({
                "model": "denno-test", "max_tokens": 8, "temperature": 0.0, "stream": false,
                "messages": [{"role": "user", "content": "t0"}]
            })
        };
        let mut s = post_raw(addr, path, &body.to_string()).await;
        let mut buf = Vec::new();
        timeout(Duration::from_secs(20), s.read_to_end(&mut buf))
            .await
            .unwrap_or_else(|_| panic!("{path}: 切断後に応答が完了しない"))
            .unwrap();
        let text = String::from_utf8_lossy(&buf);
        assert!(
            text.starts_with("HTTP/1.1 200"),
            "{path}: 200 でない: {text}"
        );
    }
    let st = state.scheduler.stats();
    assert_eq!(st.aborted_disconnected, 3, "{st:?}");
    assert_eq!(st.completed, 3, "{st:?}");
    assert_eq!(state.scheduler.in_flight(), 0);
    assert_eq!(state.scheduler.pending(), 0);
    assert_eq!(state.engine.lock().unwrap().slot_busy_count(), 0);
}
