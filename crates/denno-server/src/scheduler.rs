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
//! - s38: クライアント切断の検知。SSE ストリームではクライアントが接続を切ると
//!   hyper が書き込み失敗 (broken pipe / connection reset) を検知して応答ボディを
//!   drop し、events チャネルが閉鎖する。ワーカーはステップ境界ごとに閉鎖を確認し、
//!   該当 Run の生成を打ち切る (KV スロット解放 + ログ 1 行。他 Run への影響無し・
//!   数値経路には触れない)。
//! - s44: 非ストリーム (Job::Chat) も同様に検知する。hyper は応答 future が
//!   pending の間も接続の read 側を監視しており、クライアント切断を検知すると
//!   接続タスクごと handler future をキャンセル (drop) する。handler は
//!   respond (oneshot) の Receiver を保持しているため、キャンセルされると
//!   Sender 側の is_closed() が true になる — これを切断の間接観測に使い、
//!   SSE と同一の打ち切り経路 (sweep) に乗せる。切断が応答書き込みの瞬間まで
//!   検知されなかった場合 (EPIPE) は生成が完走し、hyper がその接続タスクだけを
//!   終了させるため、いずれの経路でもジョブは完走/破棄のどちらかで確実に整理され、
//!   ワーカー・KV・受付ループへの残留影響は無い。

use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use anyhow::{bail, Result};
use axum::response::sse::Event;
use denno_engine::{prof, ChatMessage, ChatOutput, ChatSession, Engine, GenParams, ToolConstraint};
use serde_json::json;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::{new_msg_id, to_event};
use crate::stream_asm::StreamAssembler;

/// ワーカーが同時に進行できるリクエスト数の上限 (= エンジンの KV キャッシュ
/// スロット数と一致させる)。キュー (`max_queue`) とは別の上限。
pub const MAX_ACTIVE_SESSIONS: usize = 4;

/// スケジューラに積むジョブ。検証済みのパラメータのみが入る。
/// `tools` は chat template の `tools` 変数へ渡す値 (None = tools 無し)。
/// `enable_thinking` (s47) は chat template の `enable_thinking` 変数として渡すか
/// 否か (None = 変数無し = 従来経路と同一。thinking=off のとき Some(false))。
pub enum Job {
    /// 非ストリーミング生成 (結果を oneshot で返す)
    Chat {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        /// 第21: tool_choice any/Named の拘束デコード (None = 無マスク)
        constraint: Option<ToolConstraint>,
        /// s47: thinking 制御 (off のとき Some(false)・brief/full は None)
        enable_thinking: Option<bool>,
        respond: oneshot::Sender<Result<ChatOutput>>,
    },
    /// SSE ストリーミング生成 (イベントを events チャネルに流す)
    Stream {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        /// 第21: tool_choice any/Named の拘束デコード (None = 無マスク)
        constraint: Option<ToolConstraint>,
        /// s47: thinking 制御 (off のとき Some(false)・brief/full は None)
        enable_thinking: Option<bool>,
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
    /// 第27: streaming tool_use の SSE アセンブラ (Stream ジョブのみ Some)。
    /// decode ステップの間のみ取り出して delta コールバックへ渡す
    /// (ステップ完了後に Run へ戻す。finish_run が残りイベントを出す)
    asm: Option<StreamAssembler>,
}

impl Run {
    /// s38/s44: クライアント切断検知。切断済みクライアントへの応答チャネルが
    /// 閉鎖すると true になる。
    ///
    /// - SSE (Stream): hyper はクライアント切断 (broken pipe / connection reset) を
    ///   書き込み時に検知し応答ボディを drop する → ReceiverStream の Receiver が
    ///   全て消えて `events.is_closed()` が true になる。
    /// - 非ストリーム (Chat): hyper は応答 future pending 中のクライアント切断
    ///   (read 側の EOF / RST) を検知すると接続タスクごと handler future を
    ///   キャンセルする → handler が保持していた respond の Receiver が drop され
    ///   `respond.is_closed()` が true になる (s38 当時は検知経路が無かったが、
    ///   この観測で SSE と同一の sweep に乗せられる)。
    fn client_gone(&self) -> bool {
        if let Some(events) = &self.events {
            return events.is_closed();
        }
        if let Some(respond) = &self.respond {
            return respond.is_closed();
        }
        false
    }
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
        constraint: Option<ToolConstraint>,
        enable_thinking: Option<bool>,
    ) -> Result<oneshot::Receiver<Result<ChatOutput>>, QueueFull> {
        let (respond, rx) = oneshot::channel();
        self.tx
            .try_send(Job::Chat { system, messages, params, tools, constraint, enable_thinking, respond })
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
        constraint: Option<ToolConstraint>,
        enable_thinking: Option<bool>,
        events: tokio_mpsc::Sender<Result<Event, Infallible>>,
    ) -> Result<(), QueueFull> {
        self.tx
            .try_send(Job::Stream { system, messages, params, tools, constraint, enable_thinking, events })
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
        // s37: active が空のときは try_recv ポーリングではなく recv でブロックする。
        // 旧実装はワーカーループ全体がブロック無しで回る設計で (P17k コメントの
        // 「秒間数十万回」参照)、ジョブが無い間も try_recv + engine Mutex + 空
        // step_chat_batch を全速で回し続け、アイドル中のワーカースレッドが
        // 100% CPU を恒久消費していた (実機 8095 の「denno プロセスの 1 スレッド
        // が 100% CPU で永久スピン」観測のスピン源)。recv はジョブ到着で即座に
        // 起床するため応答性は不変・全 Sender drop で終了する (旧終了条件
        // Disconnected && active.is_empty() と同一)。
        if active.is_empty() {
            match rx.recv() {
                Ok(job) => admit_job(&engine, job, &in_flight, &mut active),
                Err(_) => return,
            }
        }
        while active.len() < MAX_ACTIVE_SESSIONS {
            match rx.try_recv() {
                Ok(job) => admit_job(&engine, job, &in_flight, &mut active),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if active.is_empty() {
                        return;
                    }
                    break;
                }
            }
        }

        // --- s38: 切断スイープ (decode / prefill の各ステップ境界で評価) ---
        // 前ステップ以降にクライアントが切断した Run (SSE 応答ボディの drop →
        // events チャネル閉鎖) を打ち切る。待ち行列 pull 直後〜prefill 前の切断も
        // ここで拾うため、切断済みクライアントの分の無駄な prefill を実行しない。
        let mut i = active.len();
        while i > 0 {
            i -= 1;
            if active[i].client_gone() {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                let run = active.remove(i);
                abort_disconnected(&engine, run);
            }
        }
        if active.is_empty() {
            continue;
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
        // (SSE 送信者は先に clone しておき、セッションの可変借用と分離する。
        //  アセンブラも Run から取り出して delta コールバックへ渡す —
        //  コールバックは Box<dyn FnMut> なので &mut StreamAssembler を直接捕まえる
        //  ためには engine 側の Box<dyn FnMut + '_> 寿命拡張が必要。第27)
        let mut asms: Vec<Option<StreamAssembler>> =
            active.iter_mut().map(|r| r.asm.take()).collect();
        // コールバック (asms への &mut 借用を保持する) はこの block 内で drop させる
        // — ステップ完了後に asms を Run へ move し戻す必要があるため、借用と
        // move の区間を重ねない (E0505 対策)
        let step = {
            let mut boxed_cbs: Vec<Box<dyn FnMut(&str) + '_>> = asms
                .iter_mut()
                .zip(active.iter())
                .map(|(asm, run)| -> Box<dyn FnMut(&str) + '_> {
                    let events = run.events.clone();
                    match asm {
                        Some(asm) => Box::new(move |delta: &str| {
                            // 第27: block 境界検出付きの SSE 組み立て (tool_use block も流す)
                            asm.push_delta(delta);
                            for ev in asm.take_events() {
                                if let Some(events) = &events {
                                    let _ = events.blocking_send(Ok(to_event(&ev)));
                                }
                            }
                        }),
                        None => Box::new(move |_: &str| {
                            // 非ストリーム (Job::Chat) は送信先が無い → no-op
                            let _ = &events;
                        }),
                    }
                })
                .collect();
            let mut g = engine.lock().unwrap();
            let mut sess_refs: Vec<&mut ChatSession> =
                active.iter_mut().map(|r| r.sess.as_mut().unwrap()).collect();
            g.step_chat_batch(&mut sess_refs, &mut boxed_cbs)
        };
        match step {
            Ok(done) => {
                // アセンブラを Run へ戻す (done 判定の添字は active の並びと一致)
                for (run, asm) in active.iter_mut().zip(asms) {
                    run.asm = asm;
                }
                // 完了/切断したセッションを外して後処理 (後ろから外すことで添字を維持)
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
                    } else if active[i].client_gone() {
                        // s38/s44: decode ステップ中にクライアントが切断。
                        // Stream: 書き込み失敗 → 応答ボディ drop → チャネル閉鎖。
                        // Chat: hyper が handler future をキャンセル → respond 閉鎖。
                        // 生成を打ち切って KV スロットを解放する (通常完了と同一の
                        // 除去経路なので他 Run / engine 状態への影響は無い)
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        let run = active.remove(i);
                        abort_disconnected(&engine, run);
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
    let (system, messages, params, tools, constraint, enable_thinking, respond, events) = match job {
        Job::Chat { system, messages, params, tools, constraint, enable_thinking, respond } => {
            (system, messages, params, tools, constraint, enable_thinking, Some(respond), None)
        }
        Job::Stream { system, messages, params, tools, constraint, enable_thinking, events } => {
            (system, messages, params, tools, constraint, enable_thinking, None, Some(events))
        }
    };
    let (sess, error) = {
        let g = engine.lock().unwrap();
        match g.begin_chat_with_tools(
            system.as_deref(),
            &messages,
            &params,
            tools.as_ref(),
            constraint.as_ref(),
            enable_thinking,
        ) {
            Ok(s) => (Some(s), None),
            Err(e) => (None, Some(e)),
        }
    };
    // Stream ジョブは SSE アセンブラを持つ (index 0 の text block は admit が開ける)
    let asm = match respond {
        Some(_) => None,
        None => Some(StreamAssembler::new(0)),
    };
    Run { sess, error, respond, events, started: false, asm }
}

/// 新規ジョブの admit (pull 経路 2 箇所の共通化)。成功なら active へ push する。
/// admit 失敗時は send_error で応答するが、s38: 切断由来 (events チャネル閉鎖 —
/// SSE プリアンブル送信が broken pipe で失敗する等 / s44: respond Receiver drop —
/// 待ち行列中にクライアントが切断した場合) の場合はエラーではなく
/// 打ち切りとして扱う (クライアントは既に居ないため送信先が存在しない)。
fn admit_job(
    engine: &Arc<Mutex<Engine>>,
    job: Job,
    in_flight: &AtomicUsize,
    active: &mut Vec<Run>,
) {
    in_flight.fetch_add(1, Ordering::SeqCst);
    let mut run = build_run(engine, job);
    if let Err(e) = admit(engine, &mut run) {
        in_flight.fetch_sub(1, Ordering::SeqCst);
        if run.client_gone() {
            abort_disconnected(engine, run);
        } else {
            send_error(&mut run, e);
        }
    } else {
        active.push(run);
    }
}

/// s38/s44: クライアント切断による Run 打ち切り。KV スロットを解放し 1 行だけ記録する。
/// 送信は行わない (送信先チャネルは既に閉じている)。engine 側は通常完了と同一の
/// `release_session` のみで、KV/Scratch 状態は次のセッションからそのまま使える。
/// install 前 (admit 失敗・未実施) の場合も `release_session` は slot 無しで no-op。
fn abort_disconnected(engine: &Arc<Mutex<Engine>>, mut run: Run) {
    if let Some(sess) = run.sess.as_mut() {
        engine.lock().unwrap().release_session(sess);
    }
    tracing::info!("client disconnected; generation aborted");
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
    let Run { sess, respond, events, mut asm, .. } = run;
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
        // 4. content_block_stop (アセンブラが残りテキストのフラッシュ + open block の
        //    close を行う。tool block を出した場合 stop_reason は "tool_use")
        let tool_use = asm.as_ref().map_or(false, |a| a.blocks_emitted() > 0);
        if let Some(asm) = asm.as_mut() {
            asm.finish();
            for ev in asm.take_events() {
                if !send(ev) {
                    return;
                }
            }
        }
        // 5. message_delta (tool_use block を出した場合は "tool_use" —
        //    非ストリーミング message_json と同一規則)
        let stop_reason = if tool_use { "tool_use" } else { out.stop_reason.as_str() };
        let _ = send(json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": out.stop_sequence},
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
