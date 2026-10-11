//! P4: 連続バッチングスケジューラ (同時受け付け + prefill/decode インターリーブ)。
//!
//! 設計:
//!
//! - HTTP ハンドラは検証が通ったリクエストを直ちに有界キュー (`sync_channel`)
//!   に入れる (**同時受け付け**)。キューが満杯のとき `QueueFull` を返し、
//!   ハンドラは HTTP 503 (overloaded_error) を返す。
//! - ワーカーは 1 つだけあり、エンジンの Mutex を **1 forward ステップ単位**で
//!   取得して複数リクエストの生成を時分割で進める:
//!   - 実行中リクエストの decode 1 ステップの合間に、新規リクエストの prefill
//!     (1 forward) を割り込ませる (**prefill/decode インターリーブ**。TTFT 改善)。
//!     優先度は 実行中 decode > 新規 prefill。
//!   - decode 中の複数リクエストは round-robin で 1 ステップずつ進める (フェアネス)。
//!   - 切替時、単一 KV キャッシュの内容を `suspend_session` / `install_session`
//!     でスナップショット/復元する。KV の各行はトークン接頭辞のみに依存するため
//!     切替えは bit 完全に可逆 (**exactness 維持**: インターリーブ下でも各
//!     リクエストの出力は単独実行と完全一致)。
//! - 同時に live (KV キャッシュ保持者) にできるのは 1 リクエスト。進行中
//!   (live + suspend 済み) のリクエスト数は [`MAX_ACTIVE_SESSIONS`] に制限する
//!   (suspend KV スナップショットのメモリ上限)。
//! - キューに入ったリクエストは実行開始まで待つ (**サーバー側の待ち時間
//!   タイムアウトは無し**。タイムアウト制御はクライアント側の責務)。
//!   SSE リクエストは応答ストリームを即座に開始するため、待ち行列中も
//!   keep-alive ping (2秒間隔) が流れ続けて接続が切れない。

use std::collections::VecDeque;
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use anyhow::{bail, Result};
use axum::response::sse::Event;
use denno_engine::{ChatMessage, ChatOutput, ChatSession, Engine, GenParams};
use serde_json::json;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::{new_msg_id, to_event};

/// ワーカーが同時に進行できるリクエスト数の上限 (live + suspend 済み)。
/// キュー (`max_queue`) とは別の上限で、suspend KV スナップショットの
/// メモリ (1 リクエストあたり最大コンテキスト長分の KV) を抑える。
pub const MAX_ACTIVE_SESSIONS: usize = 4;

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
        events: tokio_mpsc::Sender<Result<Event, Infallible>>,
    ) -> Result<(), QueueFull> {
        self.tx
            .try_send(Job::Stream { system, messages, params, events })
            .map_err(|_| QueueFull)
    }
}

/// ワーカーメインループ。エンジン Mutex を 1 ステップ単位で取得し、
/// 複数リクエストを時分割で進める。
fn worker(engine: Arc<Mutex<Engine>>, rx: mpsc::Receiver<Job>, in_flight: Arc<AtomicUsize>) {
    // live (KV キャッシュ保持者) のセッション
    let mut live: Option<Run> = None;
    // suspend 済み (KV を退避している) のセッション群
    let mut pending: VecDeque<Run> = VecDeque::new();
    // 直前の割り込みが「新規 prefill の pull」だったか。
    // 新規が連続で到着しても pending (decode 中) を飢えさせないため、
    // 1 pull ごとに 1 回 rotate を挟む。
    let mut last_was_pull = false;

    loop {
        // --- live を確保する ---
        if live.is_none() {
            let (mut run, from_rx) = match pending.pop_front() {
                Some(r) => (r, false),
                None => match rx.recv() {
                    Ok(job) => {
                        in_flight.fetch_add(1, Ordering::SeqCst);
                        (build_run(&engine, job), true)
                    }
                    // 全 sender drop → 終了
                    Err(_) => break,
                },
            };
            if let Err(e) = admit(&engine, &mut run) {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                send_error(&mut run, e);
                continue;
            }
            live = Some(run);
            last_was_pull = from_rx;
        }

        // --- 1 ステップ実行 (エンジン Mutex を 1 forward 分だけ取得) ---
        let step = {
            let run = live.as_mut().unwrap();
            let mut g = engine.lock().unwrap();
            match run.events.as_ref() {
                Some(events) => {
                    let events = events.clone();
                    let mut cb = |delta: &str| {
                        let _ = events.blocking_send(Ok(to_event(&json!({
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta", "text": delta},
                        }))));
                    };
                    g.step_chat(&mut run.sess.as_mut().unwrap(), &mut cb)
                }
                None => g.step_chat(&mut run.sess.as_mut().unwrap(), &mut |_| {}),
            }
        };
        match step {
            Ok(false) => {}
            Ok(true) => {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                finish_run(live.take().unwrap());
                continue;
            }
            Err(e) => {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                send_error(&mut live.take().unwrap(), e);
                continue;
            }
        }

        // --- ステップ間のスケジューリング (インターリーブ) ---
        let active = 1 + pending.len();
        let want_pull = active < MAX_ACTIVE_SESSIONS && (!last_was_pull || pending.is_empty());
        let mut scheduled = false;
        if want_pull {
            if let Ok(job) = rx.try_recv() {
                // 新規リクエストの prefill を decode の合間に割り込ませる:
                // 実行中を suspend して pending へ、新規を live (prefill) へ
                in_flight.fetch_add(1, Ordering::SeqCst);
                {
                    let mut g = engine.lock().unwrap();
                    g.suspend_session(&mut live.as_mut().unwrap().sess.as_mut().unwrap());
                }
                pending.push_back(live.take().unwrap());
                let mut nrun = build_run(&engine, job);
                if let Err(e) = admit(&engine, &mut nrun) {
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    send_error(&mut nrun, e);
                } else {
                    live = Some(nrun);
                    last_was_pull = true;
                }
                scheduled = true;
            }
        }
        if !scheduled && !pending.is_empty() {
            // round-robin: 実行中を suspend し、pending の先頭を resume
            {
                let mut g = engine.lock().unwrap();
                g.suspend_session(&mut live.as_mut().unwrap().sess.as_mut().unwrap());
            }
            pending.push_back(live.take().unwrap());
            let mut next = pending.pop_front().unwrap();
            {
                let mut g = engine.lock().unwrap();
                g.install_session(&mut next.sess.as_mut().unwrap());
            }
            live = Some(next);
            last_was_pull = false;
        }
    }
}

/// ジョブから実行状態を作る (begin_chat まで。KV には触れない)。
fn build_run(engine: &Arc<Mutex<Engine>>, job: Job) -> Run {
    let (system, messages, params, respond, events) = match job {
        Job::Chat { system, messages, params, respond } => {
            (system, messages, params, Some(respond), None)
        }
        Job::Stream { system, messages, params, events } => {
            (system, messages, params, None, Some(events))
        }
    };
    let (sess, error) = {
        let g = engine.lock().unwrap();
        match g.begin_chat(system.as_deref(), &messages, &params) {
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
        g.install_session(run.sess.as_mut().unwrap());
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
