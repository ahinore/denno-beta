//! P6: 連続バッチングスケジューラ (同時受け付け + prefill/decode インターリーブ)。
//!
//! 設計:
//!
//! - HTTP ハンドラは検証が通ったリクエストを直ちに有界キュー (`sync_channel`)
//!   に入れる (**同時受け付け**)。キューが満杯のとき `QueueFull` を返し、
//!   ハンドラは HTTP 503 (overloaded_error) を返す。
//! - ワーカーは 1 つだけあり、エンジンの Mutex を **1 ステップ単位**で取得して
//!   複数リクエストの生成を進める:
//!   - 各リクエストは [`MAX_ACTIVE_SESSIONS`] 個の KV キャッシュスロットの
//!     いずれかに**完了まで常駐**する (P4 の suspend/restore ローテーション廃止)。
//!   - decode 中の全リクエストは [`Engine::step_chat_batch`] で **1 forward に
//!     まとめて**進める (P6 バッチ並列 decode。線形層で重み読み出しを共有)。
//!   - 新規リクエストの prefill は decode ステップの合間に 1 forward で割り込ませる
//!     (**prefill/decode インターリーブ**。TTFT 改善)。優先度は 実行中 decode > 新規 prefill。
//!   - exactness: `forward_batch` は各セッションの KV/演算列が単独 decode と
//!     ビット一致 (engine 側の契約)。サンプラはセッション毎に独立した RNG を
//!     持つため、バッチの構成は各リクエストの出力に影響しない
//!     (**インターリーブ下でも各リクエストの出力は単独実行と完全一致**)。
//! - 進行中のリクエスト数は [`MAX_ACTIVE_SESSIONS`] (KV キャッシュスロット数) に
//!   制限する。
//! - キューに入ったリクエストは実行開始まで待つ (**サーバー側の待ち時間
//!   タイムアウトは無し**。タイムアウト制御はクライアント側の責務)。
//!   SSE リクエストは応答ストリームを即座に開始するため、待ち行列中も
//!   keep-alive ping (2秒間隔) が流れ続けて接続が切れない。

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use anyhow::{bail, Result};
use axum::response::sse::Event;
use denno_engine::{prof, ChatMessage, ChatOutput, ChatSession, Engine, GenParams};
use serde_json::json;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::{new_msg_id, to_event};

/// ワーカーが同時に進行できるリクエスト数の上限 (= エンジンの KV キャッシュ
/// スロット数と一致させる)。キュー (`max_queue`) とは別の上限。
pub const MAX_ACTIVE_SESSIONS: usize = 4;

/// スケジューラに積むジョブ。検証済みのパラメータのみが入る。
/// `tools` は chat template の `tools` 変数へ渡す値 (None = tools 無し)。
pub enum Job {
    /// 非ストリーミング生成 (結果を oneshot で返す)
    Chat {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        respond: oneshot::Sender<Result<ChatOutput>>,
    },
    /// SSE ストリーミング生成 (イベントを events チャネルに流す)
    Stream {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        events: tokio_mpsc::Sender<Result<Event, Infallible>>,
    },
}

/// キュー満杯 (同時受け付け上限超過)。ハンドラは 503 を返す。
#[derive(Debug)]
pub struct QueueFull;

/// ワーカー内での 1 リクエスト分の実行状態。
struct Run {
    /// begin_chat が成功していれば Some。失敗した場合は error に理由が入る。
    sess: Option<ChatSession>,
    error: Option<anyhow::Error>,
    respond: Option<oneshot::Sender<Result<ChatOutput>>>,
    events: Option<tokio_mpsc::Sender<Result<Event, Infallible>>>,
    /// SSE プリアンブル (message_start 等) を送信済みか
    started: bool,
}

/// リクエストスケジューラ。有界キュー + 単一ワーカースレッド。
pub struct Scheduler {
    tx: mpsc::SyncSender<Job>,
    /// 現在進行中 (pull 済み未完了) のジョブ数。観測/テスト用
    in_flight: Arc<AtomicUsize>,
    max_queue: usize,
}

impl Scheduler {
    /// ワーカースレッドを起動する。
    /// `max_queue` は「実行中 (進行中) の他に待ち行列に入れられる数」。
    pub fn spawn(engine: Arc<Mutex<Engine>>, max_queue: usize) -> Self {
        let (tx, rx) = mpsc::sync_channel(max_queue.max(1));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let worker_in_flight = in_flight.clone();
        std::thread::Builder::new()
            .name("denno-scheduler".into())
            .spawn(move || worker(engine, rx, worker_in_flight))
            .expect("scheduler スレッドの起動に失敗");
        Self { tx, in_flight, max_queue }
    }

    pub fn max_queue(&self) -> usize {
        self.max_queue
    }

    /// 現在進行中のジョブ数 (pull 済み未完了。キュー待ちは含まない)
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// 非ストリーミング生成をキューに入れる。満杯なら `QueueFull`。
    pub fn submit_chat(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
    ) -> Result<oneshot::Receiver<Result<ChatOutput>>, QueueFull> {
        let (respond, rx) = oneshot::channel();
        self.tx
            .try_send(Job::Chat { system, messages, params, tools, respond })
            .map_err(|_| QueueFull)?;
        Ok(rx)
    }

    /// SSE ストリーミング生成をキューに入れる。満杯なら `QueueFull`。
    pub fn submit_stream(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        events: tokio_mpsc::Sender<Result<Event, Infallible>>,
    ) -> Result<(), QueueFull> {
        self.tx
            .try_send(Job::Stream { system, messages, params, tools, events })
            .map_err(|_| QueueFull)
    }
}

/// ワーカーメインループ。エンジン Mutex を 1 ステップ単位で取得し、
/// 複数リクエストを並行して進める (decode は 1 forward にバッチ)。
fn worker(engine: Arc<Mutex<Engine>>, rx: mpsc::Receiver<Job>, in_flight: Arc<AtomicUsize>) {
    // 常駐中 (KV スロット占有中) のセッション群
    let mut active: Vec<Run> = Vec::new();

    loop {
        // --- 新規ジョブの pull (空きスロットがある限り同時受け付け) ---
        while active.len() < MAX_ACTIVE_SESSIONS {
            match rx.try_recv() {
                Ok(job) => {
                    in_flight.fetch_add(1, Ordering::SeqCst);
                    let mut run = build_run(&engine, job);
                    if let Err(e) = admit(&engine, &mut run) {
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        send_error(&mut run, e);
                    } else {
                        active.push(run);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if active.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }

        // --- prefill (優先): 未 prefill のセッションを 1 つ進める ---
        // (prefill はプロンプト全体の 1 forward — バッチ化せず個別実行。
        //  decode の合間に割り込ませることで新規リクエストの TTFT を抑える)
        if let Some(i) = active
            .iter()
            .position(|r| !r.sess.as_ref().unwrap().prefill_done())
        {
            let step = {
                let mut g = engine.lock().unwrap();
                g.step_chat(active[i].sess.as_mut().unwrap(), &mut |_| {})
            };
            if let Err(e) = step {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                let mut run = active.remove(i);
                {
                    let mut g = engine.lock().unwrap();
                    g.release_session(run.sess.as_mut().unwrap());
                }
                send_error(&mut run, e);
            }
            continue;
        }

        // --- decode: 全セッションを 1 ステップまとめて進める ---
        // (SSE 送信者は先に clone しておき、セッションの可変借用と分離する)
        let senders: Vec<Option<tokio_mpsc::Sender<Result<Event, Infallible>>>> =
            active.iter().map(|r| r.events.clone()).collect();
        let mut boxed_cbs: Vec<Box<dyn FnMut(&str)>> = senders
            .into_iter()
            .map(|s| -> Box<dyn FnMut(&str)> {
                match s {
                    Some(events) => Box::new(move |delta: &str| {
                        let _ = events.blocking_send(Ok(to_event(&json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": delta},
                        }))));
                    }),
                    None => Box::new(|_: &str| {}),
                }
            })
            .collect();
        let step = {
            let mut g = engine.lock().unwrap();
            let mut sess_refs: Vec<&mut ChatSession> =
                active.iter_mut().map(|r| r.sess.as_mut().unwrap()).collect();
            g.step_chat_batch(&mut sess_refs, &mut boxed_cbs)
        };
        match step {
            Ok(done) => {
                // 完了したセッションを外して後処理 (後ろから外すことで添字を維持)
                let mut i = done.len();
                while i > 0 {
                    i -= 1;
                    if done[i] {
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        let mut run = active.remove(i);
                        {
                            let mut g = engine.lock().unwrap();
                            g.release_session(run.sess.as_mut().unwrap());
                        }
                        finish_run(run);
                    }
                }
                // P17k: 最後のセッションが完了した時点で 1 回だけ出力する。
                // (旧: active が空なら毎 tick 出力 — ワーカーループはブロックなしで
                // 回るためアイドル時に空 PROFILE ブロックを秒間数十万回ログし、
                // /tmp ログが数時間で数百 GB に膨らむ事故の原因だった)
                if done.iter().any(|&d| d) && active.is_empty() && prof::enabled() {
                    tracing::info!("{}", prof::summary());
                }
            }
            Err(e) => {
                // バッチ全体が失敗 (detok エラー等。実運用では稀)。
                // 全セッションをエラーで終わらせる。
                let mut runs = std::mem::take(&mut active);
                for run in runs.iter_mut() {
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    let mut g = engine.lock().unwrap();
                    g.release_session(run.sess.as_mut().unwrap());
                }
                for mut run in runs {
                    let msg = format!("{e:#}");
                    send_error(&mut run, anyhow::anyhow!(msg));
                }
            }
        }
    }
}

/// ジョブから実行状態を作る (begin_chat まで。KV には触れない)。
fn build_run(engine: &Arc<Mutex<Engine>>, job: Job) -> Run {
    let (system, messages, params, tools, respond, events) = match job {
        Job::Chat { system, messages, params, tools, respond } => {
            (system, messages, params, tools, Some(respond), None)
        }
        Job::Stream { system, messages, params, tools, events } => {
            (system, messages, params, tools, None, Some(events))
        }
    };
    let (sess, error) = {
        let g = engine.lock().unwrap();
        match g.begin_chat_with_tools(system.as_deref(), &messages, &params, tools.as_ref()) {
            Ok(s) => (Some(s), None),
            Err(e) => (None, Some(e)),
        }
    };
    Run { sess, error, respond, events, started: false }
}

/// セッションを live にする (install) + 初回なら SSE プリアンブルを送る。
fn admit(engine: &Arc<Mutex<Engine>>, run: &mut Run) -> Result<()> {
    if let Some(e) = run.error.take() {
        return Err(e);
    }
    {
        let mut g = engine.lock().unwrap();
        g.install_session(run.sess.as_mut().unwrap())
            .map_err(|e| e.context("KVキャッシュスロットの取得に失敗"))?;
    }
    if run.events.is_some() && !run.started {
        run.started = true;
        let events = run.events.as_ref().unwrap();
        let model = engine.lock().unwrap().name.clone();
        let input_tokens = run.sess.as_ref().unwrap().input_tokens();
        let send = |v: serde_json::Value| -> bool {
            events.blocking_send(Ok(to_event(&v))).is_ok()
        };
        // 1. message_start
        if !send(json!({
            "type": "message_start",
            "message": {
                "id": new_msg_id(),
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": serde_json::Value::Null,
                "stop_sequence": serde_json::Value::Null,
                "usage": {"input_tokens": input_tokens, "output_tokens": 0},
            }
        })) {
            bail!("sse send failed (client disconnected)");
        }
        // 2. content_block_start
        if !send(json!({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""}
        })) {
            bail!("sse send failed (client disconnected)");
        }
        // 3. ping
        let _ = send(json!({"type": "ping"}));
    }
    Ok(())
}

/// 完了したセッションの後処理 (respond or SSE 後半イベント)。
fn finish_run(run: Run) {
    let Run { sess, respond, events, .. } = run;
    let Some(sess) = sess else { return };
    let out = sess.into_output();
    if let Some(respond) = respond {
        let _ = respond.send(Ok(out));
        return;
    }
    if let Some(events) = events {
        let send = |v: serde_json::Value| -> bool {
            events.blocking_send(Ok(to_event(&v))).is_ok()
        };
        // 4. content_block_stop
        if !send(json!({"type": "content_block_stop", "index": 0})) {
            return;
        }
        // 5. message_delta
        let _ = send(json!({
            "type": "message_delta",
            "delta": {"stop_reason": out.stop_reason.as_str(), "stop_sequence": out.stop_sequence},
            "usage": {"output_tokens": out.output_tokens}
        }));
        // 6. message_stop
        let _ = send(json!({"type": "message_stop"}));
    }
}

/// セッションをエラーで終わらせる (respond or SSE error イベント)。
fn send_error(run: &mut Run, e: anyhow::Error) {
    tracing::error!("chat failed: {e:#}");
    if let Some(respond) = run.respond.take() {
        let _ = respond.send(Err(e));
    } else if let Some(events) = run.events.take() {
        // エラーは error イベントで通知して閉じる
        let _ = events.blocking_send(Ok(to_event(&json!({
            "type": "error",
            "error": {"type": "api_error", "message": format!("{e:#}")}
        }))));
    }
}
