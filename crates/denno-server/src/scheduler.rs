//! P4: サーバー層のリクエストスケジューラ (キューイング + 直列実行 + 同時受け付け)。
//!
//! 設計 (安全な中間形。エンジン内での prefill/decode インターリーブ =
//! 真の連続バッチングは次ステップ):
//!
//! - HTTP ハンドラは検証が通ったリクエストを直ちにジョブとして有界キューに
//!   入れる (**同時受け付け**)。キューが満杯のとき `QueueFull` を返し、
//!   ハンドラは HTTP 503 (overloaded_error) を返す。
//! - ワーカーは 1 つだけあり、ジョブを 1 件ずつ取り出してエンジンを Mutex
//!   ロック下で逐次実行する (**直列実行、同時実行上限 = 1**)。実行中の
//!   リクエストが完了するまで次のジョブは開始しない。
//! - キューに入ったリクエストは実行開始まで待つ (**サーバー側の待ち時間
//!   タイムアウトは無し**。タイムアウト制御はクライアント側の責務)。
//!   SSE リクエストは応答ストリームを即座に開始するため、待ち行列中も
//!   keep-alive ping (2秒間隔) が流れ続けて接続が切れない。
//! - エンジンは単一 KV キャッシュを持つため実行は必然的に直列であり、
//!   出力は単発実行と完全一致 (**exactness 維持**)。

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::response::sse::Event;
use denno_engine::{ChatMessage, ChatOutput, Engine, GenParams};
use serde_json::json;
use tokio::sync::{mpsc, oneshot};

use crate::{new_msg_id, to_event};

/// スケジューラに積むジョブ。検証済みのパラメータのみが入る。
pub enum Job {
    /// 非ストリーミング生成 (結果を oneshot で返す)
    Chat {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        respond: oneshot::Sender<Result<ChatOutput>>,
    },
    /// SSE ストリーミング生成 (イベントを events チャネルに流す)
    Stream {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        events: mpsc::Sender<Result<Event, Infallible>>,
    },
}

/// キュー満杯 (同時受け付け上限超過)。ハンドラは 503 を返す。
#[derive(Debug)]
pub struct QueueFull;

/// リクエストスケジューラ。有界キュー + 単一ワーカー。
pub struct Scheduler {
    tx: mpsc::Sender<Job>,
    /// 現在実行中のジョブ数 (常時 0 or 1。観測/テスト用)
    in_flight: Arc<AtomicUsize>,
    max_queue: usize,
}

impl Scheduler {
    /// ワーカーを起動する。必ず tokio ランタイム内で呼ぶこと。
    /// `max_queue` は「実行中 1 件の他に待ち行列に入れられる数」。
    pub fn spawn(engine: Arc<Mutex<Engine>>, max_queue: usize) -> Self {
        let (tx, rx) = mpsc::channel(max_queue.max(1));
        let in_flight = Arc::new(AtomicUsize::new(0));
        tokio::spawn(worker(engine, rx, in_flight.clone()));
        Self { tx, in_flight, max_queue }
    }

    pub fn max_queue(&self) -> usize {
        self.max_queue
    }

    /// 現在実行中のジョブ数 (観測/テスト用。単一ワーカーなので 0 or 1)
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// 非ストリーミング生成をキューに入れる。満杯なら `QueueFull`。
    pub fn submit_chat(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
    ) -> Result<oneshot::Receiver<Result<ChatOutput>>, QueueFull> {
        let (respond, rx) = oneshot::channel();
        self.tx
            .try_send(Job::Chat { system, messages, params, respond })
            .map_err(|_| QueueFull)?;
        Ok(rx)
    }

    /// SSE ストリーミング生成をキューに入れる。満杯なら `QueueFull`。
    pub fn submit_stream(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        events: mpsc::Sender<Result<Event, Infallible>>,
    ) -> Result<(), QueueFull> {
        self.tx
            .try_send(Job::Stream { system, messages, params, events })
            .map_err(|_| QueueFull)
    }
}

async fn worker(engine: Arc<Mutex<Engine>>, mut rx: mpsc::Receiver<Job>, in_flight: Arc<AtomicUsize>) {
    while let Some(job) = rx.recv().await {
        in_flight.fetch_add(1, Ordering::SeqCst);
        match job {
            Job::Chat { system, messages, params, respond } => {
                let eng = engine.clone();
                let result = tokio::task::spawn_blocking(move || {
                    let mut g = eng.lock().unwrap();
                    g.chat(system.as_deref(), &messages, &params, |_| {})
                })
                .await
                .map_err(|e| anyhow::anyhow!("join error: {e}"))
                .and_then(|r| r);
                let _ = respond.send(result);
            }
            Job::Stream { system, messages, params, events } => {
                let eng = engine.clone();
                // クライアント切断 (events の受信者が消滅) してもジョブは
                // 最後まで走らせて終了させる (既存の spawn_blocking と同じ挙動)
                let _ = tokio::task::spawn_blocking(move || {
                    run_stream_job(eng, system, messages, params, events)
                })
                .await;
            }
        }
        in_flight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// SSE ジョブ本体 (旧 stream_response の spawn_blocking 部分を移植)。
fn run_stream_job(
    engine: Arc<Mutex<Engine>>,
    system: Option<String>,
    messages: Vec<ChatMessage>,
    params: GenParams,
    tx: mpsc::Sender<Result<Event, Infallible>>,
) {
    tracing::info!("stream job started");
    let input_tokens = {
        // message_start 用に input tokens を事前計算するため、まずテンプレートだけ適用
        let g = engine.lock().unwrap();
        let prompt = g.tok.apply_chat_template(system.as_deref(), &messages);
        prompt.ok().and_then(|p| g.tok.encode(&p).ok()).map(|v| v.len()).unwrap_or(0)
    };

    let send = |v: serde_json::Value| -> bool {
        match tx.blocking_send(Ok(to_event(&v))) {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!("sse send failed: {e}");
                false
            }
        }
    };

    // 1. message_start
    if !send(json!({
        "type": "message_start",
        "message": {
            "id": new_msg_id(),
            "type": "message",
            "role": "assistant",
            "model": engine.lock().unwrap().name.clone(),
            "content": [],
            "stop_reason": serde_json::Value::Null,
            "stop_sequence": serde_json::Value::Null,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        }
    })) {
        return;
    }
    // 2. content_block_start
    if !send(json!({
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""}
    })) {
        return;
    }
    // 3. ping
    let _ = send(json!({"type": "ping"}));

    // 生成
    let run = {
        let mut g = engine.lock().unwrap();
        g.chat(system.as_deref(), &messages, &params, |delta| {
            let _ = send(json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta}
            }));
        })
    };
    let (output_tokens, stop_reason, stop_sequence) = match run {
        Ok(out) => (out.output_tokens, out.stop_reason, out.stop_sequence),
        Err(e) => {
            tracing::error!("chat failed in stream: {e:#}");
            // エラーは error イベントで通知して閉じる
            let _ = send(json!({
                "type": "error",
                "error": {"type": "api_error", "message": format!("{e:#}")}
            }));
            return;
        }
    };

    // 4. content_block_stop
    if !send(json!({"type": "content_block_stop", "index": 0})) {
        return;
    }
    // 5. message_delta
    let _ = send(json!({
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason.as_str(), "stop_sequence": stop_sequence},
        "usage": {"output_tokens": output_tokens}
    }));
    // 6. message_stop
    let _ = send(json!({"type": "message_stop"}));
}
