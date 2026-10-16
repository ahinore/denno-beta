//! s44: ログ消費者 (stdout パイプの tee 等) が凍結してもサーバが受付を続けることの
//! 統合テスト (実機 wedge 事例 REPORT_NEXT17 §4 の回帰防止)。
//!
//! 事例の再現条件:
//! 1. stdout 消費者が凍結 (旧実装ではログ書き込みスレッドが std::io::stdout() の
//!    グローバルロックを保持したままパイプ書き込みで永久ブロック)
//! 2. 以後の全リクエストのログ書き込みがロック待ちで停止し tokio ワーカーが枯渇
//! 3. /health を含む全リクエストが無応答 (CPU 0%・プロセス生存)
//!
//! 修正後は logging.rs の非同期ライタ (有界キュー + 専用スレッド・満杯時破棄) を
//! グローバルサブスクライバに使い、消費者が永遠に読まなくてもサーバスレッドは
//! ブロックしない。このテストは「凍結シンク + 容量 2 (即ドロップ)」でその契約を検証する。
//!
//! 注意: グローバルサブスクライバはプロセスで 1 度しか設定できないため、
//! このテストファイルは単一テストのみを持つ。

use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_VOCAB};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_server::{AppState, NonBlockingStdout, Scheduler};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{sleep, timeout, Instant};

/// 永遠に読まない消費者 (凍結した tee 相当): write が決して戻らない。
/// 専用ログスレッドはここで止まり続けるが、サーバスレッドには影響しない。
struct FrozenSink;

impl io::Write for FrozenSink {
    fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
        loop {
            std::thread::sleep(Duration::from_secs(3600));
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

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

fn parse_json_body(buf: &str) -> Value {
    let body = buf
        .split("\r\n\r\n")
        .nth(1)
        .expect("HTTP ヘッダとボディの区切りが無い");
    serde_json::from_str(body).expect("応答ボディが JSON ではない")
}

/// 凍結したログ消費者の下で: 切断 → abort → /health → 後続リクエスト、が全て完遂
/// され、ログ行が容量を超えて破棄されていること (ブロックではなく破棄で対処した証拠)。
#[tokio::test]
async fn frozen_log_consumer_does_not_wedge_the_server() {
    // 事象条件の再現: グローバルサブスクライバに「凍結シンク + 容量 2」の
    // 非同期ライタを設定する (以後の全 tracing はここに流れる)
    let writer = NonBlockingStdout::spawn_with(Box::new(FrozenSink), 2);
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new("info"))
        .with_writer(writer.clone())
        .init();

    let tok = dummy_tokenizer().unwrap();
    let model = Box::new(SlowNoEosModel {
        delay: Duration::from_millis(80),
        ctx: 256,
    });
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
    });

    let app: Router = denno_server::router(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // 非ストリームで生成を開始させ、進行中に切断する (200 token × 80ms ≒ 16 秒)
    let mut s = TcpStream::connect(addr).await.unwrap();
    let body = request_json(200, false);
    let req = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s.flush().await.unwrap();
    sleep(Duration::from_millis(300)).await;
    assert_eq!(state.scheduler.in_flight(), 1);
    drop(s);

    // 生成完了を待たずに打ち切られる (logging がブロックしていても sweep は進む)
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        if state.scheduler.in_flight() == 0 {
            break;
        }
        sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(
        state.scheduler.in_flight(),
        0,
        "切断後も Run が残っている (ロギングのブロックがワーカーに波及?)"
    );

    // /health が応答する (旧実装ではここで wedge した)
    let mut h = TcpStream::connect(addr).await.unwrap();
    h.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut buf = Vec::new();
    timeout(Duration::from_secs(10), h.read_to_end(&mut buf))
        .await
        .expect("/health が完了しない (wedge 再現?)")
        .unwrap();
    let text = String::from_utf8_lossy(&buf);
    assert!(text.contains("200") && text.contains("\"status\":\"ok\""), "{text}");

    // 後続の非ストリームリクエストも完了する
    let mut s2 = TcpStream::connect(addr).await.unwrap();
    let body2 = request_json(8, false);
    let req2 = format!(
        "POST /v1/messages HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body2.len(),
        body2
    );
    s2.write_all(req2.as_bytes()).await.unwrap();
    let mut buf2 = Vec::new();
    timeout(Duration::from_secs(20), s2.read_to_end(&mut buf2))
        .await
        .expect("後続リクエストが完了しない")
        .unwrap();
    assert_eq!(parse_json_body(&String::from_utf8_lossy(&buf2))["type"], "message");

    // ログはブロックではなく破棄で対処された証拠 (容量 2 を超えた行が捨てられている)
    assert!(
        writer.dropped() > 0,
        "ログ行が 1 行も破棄されていない = シンクに書けたことになる (想定と不一致)"
    );
}
