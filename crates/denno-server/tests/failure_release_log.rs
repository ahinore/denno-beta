//! P58-4: 失敗したリクエストのスロット解放が `DENNO_PROFILE=1` の 1 行ログに
//! 出ることを固定する (実機でメインが読む行)。
//!
//! ログ本文:
//! `kv slot 0 released after failure (prefill/decode step failed): reserved 4096
//!  tokens cleared, KV mirror rows discarded (epoch 1)`
//!
//! `prof::enabled()` はプロセス内で一度だけ評価される (`OnceLock`)。他のテストと
//! 同居すると「先に評価した方が勝つ」ため、この検証は**専用のテストバイナリ**に
//! 分けてある (env と subscriber の両方がプロセス全体に効くため)。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::json;
use tower::ServiceExt;

const CTX: usize = 4096;
const BPT: usize = 32;

/// 最初の forward だけ Err を返すモデル (KV 行を書いた後に落ちる)。
struct FailOnceModel {
    failed: bool,
}

impl Model for FailOnceModel {
    fn context(&self) -> usize {
        CTX
    }

    fn eos_ids(&self) -> &[u32] {
        &[12]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        if !self.failed {
            self.failed = true;
            anyhow::bail!("injected VRAM exhaustion (P58-4 log probe)");
        }
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[12] = 5.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, CTX)
    }
}

/// tracing の出力を溜める writer。
#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
    type Writer = Capture;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[tokio::test]
async fn failure_release_emits_one_line_under_profile() {
    // prof の有効化は OnceLock なので、エンジンを触る前に設定する。
    std::env::set_var("DENNO_PROFILE", "1");
    let cap = Capture::default();
    tracing_subscriber::fmt()
        .with_writer(cap.clone())
        .with_ansi(false)
        .with_max_level(tracing::Level::INFO)
        .init();

    let tok = dummy_tokenizer().unwrap();
    let mut engine = Engine::from_parts(
        Box::new(FailOnceModel { failed: false }),
        tok,
        "denno-test",
        None,
    );
    engine.set_kv_budget_bytes(Some(CTX * BPT));
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    let app = denno_server::router(state);

    let body = json!({
        "model": "denno-test",
        "max_tokens": CTX,
        "temperature": 0.0,
        "stream": false,
        "messages": [{"role": "user", "content": "t0"}]
    });
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = tokio::time::timeout(Duration::from_secs(20), app.clone().oneshot(req))
        .await
        .expect("20s 以内に応答が返らない")
        .unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let _ = resp.into_body().collect().await;

    let text = String::from_utf8_lossy(&cap.0.lock().unwrap()).to_string();
    // 失敗そのもののログ (scheduler) が出ている = 失敗経路を通った証拠
    assert!(text.contains("chat failed"), "失敗ログが無い: {text}");
    // 本命: 解放の 1 行
    assert!(
        text.contains("released after failure"),
        "解放ログが DENNO_PROFILE=1 で出ていない: {text}"
    );
    assert!(
        text.contains("kv slot 0 released after failure (prefill/decode step failed)"),
        "解放ログの書式が変わっている: {text}"
    );
    assert!(
        text.contains("reserved 4096 tokens cleared"),
        "解放した予約トークン数がログに無い: {text}"
    );
    assert!(
        text.contains("KV mirror rows discarded"),
        "KV ミラー行の破棄がログに無い: {text}"
    );
}
