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
//! - キューに入ったリクエストは実行開始まで待つが、**待ちには上限がある**
//!   ([`Limits`])。P57: 旧実装は上限が無く、GPU チェーン失敗 → CPU フォールバック
//!   (GPU_BROKEN ラッチ後は毎回) で 1 ステップが数分に伸びた際、後続リクエストが
//!   永久に無応答になった。今は「キュー待ち + スロット待ち」が [`Limits::admit`] を
//!   超えたら 500 (api_error) で返してスロットを解放し、無進捗が [`Limits::stall`]
//!   を超えた Run も同じ経路で打ち切る。SSE イベント送信も [`Limits::send`] で
//!   打ち切る (読み手が止まったクライアントでワーカーがブロックしない)。
//!   生成そのものは 1 GPU なので直列化したままだが、**直列化と無応答は別物**である
//!   (制御プレーン — `/v1/models` 等 — はエンジン Mutex を取らない)。
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
//! - P873: リクエスト単位の壁時計上限 ([`Limits::request`]、既定 0 = 無効)。
//!   0 でないときは受付時刻から `request` を超えた Run をステップ境界で打ち切り、
//!   KV スロットを解放してクライアントへエラーを返す (生成そのものは中断できない —
//!   エンジンロックを握った 1 ステップが返ってきた境界で必ず整理する)。
//!   打ち切り理由は [`RunStats`] に数える (切断 / 期限 / エラー)。

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use axum::response::sse::Event;
use denno_engine::{
    prof, ChatMessage, ChatOutput, ChatSession, Engine, GenParams, TokenLogprob, ToolConstraint,
};
use serde_json::json;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};

use crate::{new_msg_id, to_event};
use crate::stream_asm::StreamAssembler;

/// ワーカーが同時に進行できるリクエスト数の上限 (= エンジンの KV キャッシュ
/// スロット数と一致させる)。キュー (`max_queue`) とは別の上限。
pub const MAX_ACTIVE_SESSIONS: usize = 4;

/// P57: 待ちに上限を入れるための値。**上限が無いと「1 リクエストの失敗」が
/// 「サーバ全体の永久無応答」に化ける** (2026-10-25 本番障害: VRAM 枯渇 →
/// `cuMemAlloc` 失敗 → 残差チェーン失敗 → GPU_BROKEN ラッチ → 以降の prefill は
/// CPU フォールバックで 1 ステップが数分。ワーカーがエンジン Mutex をほぼ連続
/// 保持し、`/v1/models` はロック待ちで 200 すら返せなくなった)。
///
/// 既定値の根拠 (推定を含む):
/// - `admit` 300s: 27B の 65k prefill が 1 本 30〜60s 級なので、4 スロット +
///   キューが埋まっても 300s 以内に必ず順番が回る。超えたら「待たせるより 500」。
/// - `stall` 900s: 正常な 1 ステップ (prefill 1 回 / decode 1 トークン) の
///   最悪値の 10 倍以上。これを超えて進まない Run は異常とみなす。
/// - `send` 30s: SSE の受信バッファ (64 本) が 30s 詰まったクライアントは
///   読み手として死んでいる。旧実装は `blocking_send` で**エンジン Mutex を
///   握ったまま**ここで永久ブロックし得た。
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// 受付 (submit) → prefill 開始 (キュー待ち + スロット待ち) の上限
    pub admit: Duration,
    /// 最後に進んだステップからの無進捗時間の上限
    pub stall: Duration,
    /// SSE イベント 1 本の送信待ちの上限
    pub send: Duration,
    /// P873: リクエストの壁時計 (受付 → 完了) の上限。`Duration::ZERO` = 無効
    /// (既定。従来と完全に同一の挙動)。0 でないときは `admit` / `stall` より
    /// 優先して効く — 待ち行列の中でも生成中でも、受付から `request` を超えた
    /// Run はステップ境界で打ち切る。
    pub request: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            admit: Duration::from_secs(300),
            stall: Duration::from_secs(900),
            send: Duration::from_secs(30),
            request: Duration::ZERO,
        }
    }
}

impl Limits {
    /// P873: キュー待ち (admit) に実効的に効く上限。`request` が有効なら
    /// その方が短いときに採用する (待ち行列の中で request 期限を過ぎたジョブを
    /// 生成開始させない)。
    fn admit_budget(&self) -> Duration {
        if self.request.is_zero() {
            self.admit
        } else {
            self.admit.min(self.request)
        }
    }
}

/// P57: 上限超過をクライアントへ返す 1 行の原因。`ApiError::from_anyhow` は
/// 未知の anyhow を 500 (api_error) にするので、追加の型は要らない。
fn timeout_error(what: &str, limit: Duration) -> anyhow::Error {
    anyhow::anyhow!(
        "timeout: {what} exceeded {:.1}s; the request was abandoned and its KV slot released",
        limit.as_secs_f64()
    )
}

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
        /// P837: raw prompt (Some なら chat template を通さず、このテキストを
        /// そのままトークン化する = `/v1/completions` 経路)。Some のとき
        /// system / messages / tools / constraint / enable_thinking は未使用。
        raw_prompt: Option<String>,
        respond: oneshot::Sender<Result<ChatOutput>>,
        /// P869: logprobs を記録するときだけ Some(top_n) (None = 既定・追加コストなし)
        logprobs: Option<usize>,
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
    /// P793: OpenAI 互換ストリーミング生成。Anthropic SSE ではなく
    /// [`RawEvent`] (生の text delta と使用量) を流し、HTTP 側 (openai.rs) が
    /// `chat.completion.chunk` へ組み直す。エンジン・セッション・KV 経路は
    /// `Stream` と完全に同一 (アセンブラだけ持たない)。
    OpenaiStream {
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        constraint: Option<ToolConstraint>,
        enable_thinking: Option<bool>,
        /// P837: raw prompt (`/v1/completions` の stream。意味は `Job::Chat` と同じ)
        raw_prompt: Option<String>,
        events: tokio_mpsc::Sender<RawEvent>,
        /// P869: logprobs を記録するときだけ Some(top_n) (None = 既定・追加コストなし)
        logprobs: Option<usize>,
    },
}

/// P793: OpenAI 互換ストリーム用の生イベント。
///
/// `/v1/messages` の SSE イベントとは別系統で、`Run` に `raw` チャネルが
/// 入っているジョブだけがこの型を受信する (それ以外の経路は従来と同一)。
#[derive(Debug, Clone, PartialEq)]
pub enum RawEvent {
    /// prefill 完了 (KV 取得) 時点の入力トークン数
    Start { input_tokens: usize },
    /// detok 済みの text delta (1 トークン分の piece)
    Delta(String),
    /// P869: 生成 1 トークン分の logprobs (`logprobs=true` のときだけ流れる)。
    /// `Delta` とは独立の chunk で、HTTP 側は `choices[0].logprobs.content` に載せる。
    Logprob(TokenLogprob),
    /// 生成終了。stop_reason は engine の表記 (`end_turn` / `max_tokens` / `stop_sequence`)
    Done {
        input_tokens: usize,
        output_tokens: usize,
        stop_reason: String,
        stop_sequence: Option<String>,
    },
    /// 生成失敗 (HTTP 側は OpenAI 形式の error イベントへ変換する)
    Error(String),
}

/// キュー満杯 (同時受け付け上限超過)。ハンドラは 503 を返す。
#[derive(Debug)]
pub struct QueueFull;

/// P873: キュー満杯時にクライアントへ返す `Retry-After` の秒数。
///
/// 根拠 (推定): 1 スロットあたりの生成は最短でも数百 ms なので、1 秒後に
/// 再試行すれば「同じ瞬間に集中した分」は確実にはけた状態になる。値を伸ばすと
/// クライアント側の体感 (と総スループット) を落とすため、短い固定値にする。
pub const QUEUE_RETRY_AFTER_SECS: u64 = 1;

impl QueueFull {
    /// クライアントへ返す `Retry-After` (秒)。
    pub fn retry_after_secs(&self) -> u64 {
        QUEUE_RETRY_AFTER_SECS
    }
}

/// P873: 打ち切り / 完了の計数。atomic 1 回の読み書きだけで、生成経路の
/// 数値・順序には触れない。運用 (`/ready`) が読む。
#[derive(Debug, Default)]
pub struct RunStats {
    completed: AtomicUsize,
    aborted_timeout: AtomicUsize,
    aborted_disconnected: AtomicUsize,
    aborted_error: AtomicUsize,
}

/// [`RunStats`] のコピー (読み出し用)。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RunStatsSnapshot {
    /// 正常完了したリクエスト数
    pub completed: usize,
    /// 期限超過 (admit / stall / request) で打ち切った数
    pub aborted_timeout: usize,
    /// クライアント切断で打ち切った数
    pub aborted_disconnected: usize,
    /// 生成エラーで終わった数
    pub aborted_error: usize,
}

impl RunStats {
    fn snapshot(&self) -> RunStatsSnapshot {
        RunStatsSnapshot {
            completed: self.completed.load(Ordering::Relaxed),
            aborted_timeout: self.aborted_timeout.load(Ordering::Relaxed),
            aborted_disconnected: self.aborted_disconnected.load(Ordering::Relaxed),
            aborted_error: self.aborted_error.load(Ordering::Relaxed),
        }
    }
}

/// ワーカー内での 1 リクエスト分の実行状態。
struct Run {
    /// begin_chat が成功していれば Some。失敗した場合は error に理由が入る。
    sess: Option<ChatSession>,
    error: Option<anyhow::Error>,
    respond: Option<oneshot::Sender<Result<ChatOutput>>>,
    events: Option<tokio_mpsc::Sender<Result<Event, Infallible>>>,
    /// P793: OpenAI 互換ストリームの生イベント送信先 (OpenaiStream のみ Some)。
    /// Anthropic 経路 (`events`) とは排他で、両方 Some にはならない。
    raw: Option<tokio_mpsc::Sender<RawEvent>>,
    /// SSE プリアンブル (message_start 等) を送信済みか
    started: bool,
    /// 第27: streaming tool_use の SSE アセンブラ (Stream ジョブのみ Some)。
    /// decode ステップの間のみ取り出して delta コールバックへ渡す
    /// (ステップ完了後に Run へ戻す。finish_run が残りイベントを出す)
    asm: Option<StreamAssembler>,
    /// s151 (Task B12): リクエストの壁時計の入口票 (計測 OFF は None)。
    /// admit で `prof::request_begin` を打ち、完了 / 切断 / エラーのどれでも
    /// `prof::request_end` を 1 回だけ呼ぶ。生成の終わりはここ (Run の終端) で、
    /// HTTP ハンドラの戻りではない (SSE は即座に戻るため)。
    req_mark: Option<prof::RequestMarker>,
    /// P57: 最後にステップが成功した時刻 (無進捗スイープの入力)。
    /// 起点は受付時刻 (= prefill 待ちも無進捗として数える)。
    last_progress: Instant,
    /// P873: リクエストの壁時計の期限 (`Limits::request` が 0 のときは None)。
    /// 受付時刻 + request。超えたらステップ境界で打ち切る。
    deadline: Option<Instant>,
    /// P57: SSE イベント送信が詰まった (読み手が止まっている) ことを
    /// delta コールバックから Run へ伝えるフラグ。ステップ境界で読む。
    send_dead: Arc<AtomicBool>,
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
        // P793: OpenAI ストリームも同じ規約 (応答ボディの drop で受信側が消える)
        if let Some(raw) = &self.raw {
            return raw.is_closed();
        }
        if let Some(respond) = &self.respond {
            return respond.is_closed();
        }
        false
    }
}

/// リクエストスケジューラ。有界キュー + 単一ワーカースレッド。
pub struct Scheduler {
    tx: mpsc::SyncSender<(Instant, Job)>,
    /// 現在進行中 (pull 済み未完了) のジョブ数。観測/テスト用
    in_flight: Arc<AtomicUsize>,
    /// P817: 受付済みでワーカーがまだ pull していないジョブ数。
    /// graceful drain は「キュー + 進行中」が 0 になるのを待つ。
    queued: Arc<AtomicUsize>,
    /// P817: 運用の状態 (`/ready` / drain)。
    lifecycle: Arc<crate::lifecycle::Lifecycle>,
    /// P873: 完了 / 打ち切りの計数 (`/ready` が読む)。
    stats: Arc<RunStats>,
    max_queue: usize,
    limits: Limits,
}

/// P817: ワーカースレッドの終了を `Lifecycle` へ伝える RAII ガード。
/// ワーカーは複数の `return` 経路を持つので、Drop で 1 回だけ書く。
struct WorkerExitGuard(Arc<crate::lifecycle::Lifecycle>);

impl Drop for WorkerExitGuard {
    fn drop(&mut self) {
        self.0.mark_worker_dead();
    }
}

impl Scheduler {
    /// ワーカースレッドを起動する。
    /// `max_queue` は「実行中 (進行中) の他に待ち行列に入れられる数」。
    pub fn spawn(engine: Arc<Mutex<Engine>>, max_queue: usize) -> Self {
        Self::spawn_with_limits(engine, max_queue, Limits::default())
    }

    /// P57: 待ちの上限を指定して起動する (テスト・運用チューニング用)。
    pub fn spawn_with_limits(engine: Arc<Mutex<Engine>>, max_queue: usize, limits: Limits) -> Self {
        Self::spawn_inner(engine, max_queue, limits, true)
    }

    /// P817: モデル未ロードの状態で起動する (テストで `/ready` の 503 を踏むため)。
    /// 通常の起動経路 (`run`) はモデルを読み終えてからワーカーを起動するので
    /// [`Scheduler::spawn`] を使う。
    pub fn spawn_not_ready(engine: Arc<Mutex<Engine>>, max_queue: usize) -> Self {
        Self::spawn_inner(engine, max_queue, Limits::default(), false)
    }

    fn spawn_inner(
        engine: Arc<Mutex<Engine>>,
        max_queue: usize,
        limits: Limits,
        ready: bool,
    ) -> Self {
        let (tx, rx) = mpsc::sync_channel(max_queue.max(1));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let queued = Arc::new(AtomicUsize::new(0));
        let lifecycle = crate::lifecycle::Lifecycle::new(ready);
        let stats = Arc::new(RunStats::default());
        let worker_in_flight = in_flight.clone();
        let worker_queued = queued.clone();
        let worker_lifecycle = lifecycle.clone();
        let worker_stats = stats.clone();
        std::thread::Builder::new()
            .name("denno-scheduler".into())
            .spawn(move || {
                worker(
                    engine,
                    rx,
                    worker_in_flight,
                    worker_queued,
                    worker_lifecycle,
                    worker_stats,
                    limits,
                )
            })
            .expect("scheduler スレッドの起動に失敗");
        Self { tx, in_flight, queued, lifecycle, stats, max_queue, limits }
    }

    pub fn max_queue(&self) -> usize {
        self.max_queue
    }

    /// P817: `/ready` と停止経路が読む状態。
    pub fn lifecycle(&self) -> &Arc<crate::lifecycle::Lifecycle> {
        &self.lifecycle
    }

    /// P817: 受付済みでまだ実行が始まっていないジョブ数。
    pub fn pending(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }

    /// P57: 実際に使われている上限 (観測/テスト用)。
    pub fn limits(&self) -> Limits {
        self.limits
    }

    /// 現在進行中のジョブ数 (pull 済み未完了。キュー待ちは含まない)
    pub fn in_flight(&self) -> usize {
        self.in_flight.load(Ordering::SeqCst)
    }

    /// P873: 完了 / 打ち切り (切断・期限・エラー) の計数。読み取りのみ。
    pub fn stats(&self) -> RunStatsSnapshot {
        self.stats.snapshot()
    }

    /// s146 (B8-3): テスト用に「同時実行上限」の値を公開する (分類メッセージと
    /// アサートの出所を 1 箇所にするため。値は不変)。
    pub fn max_active_sessions() -> usize {
        MAX_ACTIVE_SESSIONS
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
        self.submit_chat_ext(
            system,
            messages,
            params,
            tools,
            constraint,
            enable_thinking,
            None,
        )
    }

    /// P869: [`Scheduler::submit_chat`] の logprobs 指定版。
    /// `logprobs = Some(top_n)` のときだけ生成トークンの log 確率を
    /// [`ChatOutput::logprobs`] に載せる (None = 従来と完全同一)。
    #[allow(clippy::too_many_arguments)]
    pub fn submit_chat_ext(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        constraint: Option<ToolConstraint>,
        enable_thinking: Option<bool>,
        logprobs: Option<usize>,
    ) -> Result<oneshot::Receiver<Result<ChatOutput>>, QueueFull> {
        let (respond, rx) = oneshot::channel();
        self.enqueue(Job::Chat {
            system,
            messages,
            params,
            tools,
            constraint,
            enable_thinking,
            raw_prompt: None,
            respond,
            logprobs,
        })?;
        Ok(rx)
    }

    /// P837: raw prompt の非ストリーミング生成 (`POST /v1/completions`)。
    /// chat template を通さない以外は [`Scheduler::submit_chat`] と同一経路。
    pub fn submit_completion(
        &self,
        prompt: String,
        params: GenParams,
    ) -> Result<oneshot::Receiver<Result<ChatOutput>>, QueueFull> {
        let (respond, rx) = oneshot::channel();
        self.enqueue(Job::Chat {
            system: None,
            messages: Vec::new(),
            params,
            tools: None,
            constraint: None,
            enable_thinking: None,
            raw_prompt: Some(prompt),
            respond,
            logprobs: None,
        })?;
        Ok(rx)
    }

    /// P817: キューへ積む唯一の経路。`queued` の増減をここに閉じ込める
    /// (drain の待ち判定が取りこぼさないため)。
    fn enqueue(&self, job: Job) -> Result<(), QueueFull> {
        self.queued.fetch_add(1, Ordering::SeqCst);
        match self.tx.try_send((Instant::now(), job)) {
            Ok(()) => Ok(()),
            Err(_) => {
                self.queued.fetch_sub(1, Ordering::SeqCst);
                Err(QueueFull)
            }
        }
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
        self.enqueue(Job::Stream { system, messages, params, tools, constraint, enable_thinking, events })
    }

    /// P793: OpenAI 互換ストリーミング生成をキューに入れる。
    /// セッション・KV・上限の扱いは [`Scheduler::submit_stream`] と同一で、
    /// 流すイベントだけが [`RawEvent`] (生 delta) になる。
    pub fn submit_openai_stream(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        constraint: Option<ToolConstraint>,
        enable_thinking: Option<bool>,
        events: tokio_mpsc::Sender<RawEvent>,
    ) -> Result<(), QueueFull> {
        self.submit_openai_stream_ext(
            system,
            messages,
            params,
            tools,
            constraint,
            enable_thinking,
            events,
            None,
        )
    }

    /// P869: [`Scheduler::submit_openai_stream`] の logprobs 指定版。
    /// `logprobs = Some(top_n)` のときだけ [`RawEvent::Logprob`] を流す。
    #[allow(clippy::too_many_arguments)]
    pub fn submit_openai_stream_ext(
        &self,
        system: Option<String>,
        messages: Vec<ChatMessage>,
        params: GenParams,
        tools: Option<serde_json::Value>,
        constraint: Option<ToolConstraint>,
        enable_thinking: Option<bool>,
        events: tokio_mpsc::Sender<RawEvent>,
        logprobs: Option<usize>,
    ) -> Result<(), QueueFull> {
        self.enqueue(Job::OpenaiStream {
            system,
            messages,
            params,
            tools,
            constraint,
            enable_thinking,
            raw_prompt: None,
            events,
            logprobs,
        })
    }

    /// P837: raw prompt の OpenAI 互換ストリーミング (`POST /v1/completions`, stream)。
    /// 流すのは [`RawEvent`] なので、HTTP 側 (openai.rs) が `text_completion`
    /// chunk へ組み直す。
    pub fn submit_completion_stream(
        &self,
        prompt: String,
        params: GenParams,
        events: tokio_mpsc::Sender<RawEvent>,
    ) -> Result<(), QueueFull> {
        self.enqueue(Job::OpenaiStream {
            system: None,
            messages: Vec::new(),
            params,
            tools: None,
            constraint: None,
            enable_thinking: None,
            raw_prompt: Some(prompt),
            events,
            logprobs: None,
        })
    }
}

/// ワーカーメインループ。エンジン Mutex を 1 ステップ単位で取得し、
/// 複数リクエストを並行して進める (decode は 1 forward にバッチ)。
///
/// P57: エンジン Mutex を握るのは **このスレッドだけ** (HTTP ハンドラは取らない —
/// `AppState` の不変情報は起動時にキャッシュする)。それでも 1 ステップが異常に
/// 長くなることはあり得る (GPU チェーン失敗 → CPU フォールバック) ので、
/// 握る区間そのものは短くできない代わりに **待ち側に上限**を置く ([`Limits`])。
/// ステップ境界で期限切れの Run を打ち切り、スロットとロックを解放する。
fn worker(
    engine: Arc<Mutex<Engine>>,
    rx: mpsc::Receiver<(Instant, Job)>,
    in_flight: Arc<AtomicUsize>,
    queued: Arc<AtomicUsize>,
    lifecycle: Arc<crate::lifecycle::Lifecycle>,
    stats: Arc<RunStats>,
    limits: Limits,
) {
    // P817: このスレッドが終わったら `Lifecycle` に生存=false を書く
    // (複数の return 経路があるので RAII)。
    let _exit = WorkerExitGuard(lifecycle);
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
                Ok((submitted, job)) => {
                    // P817: 加算 (in_flight) を減算 (queued) より先に行い、
                    // drain の待ち判定に 0/0 の窓を作らない。
                    in_flight.fetch_add(1, Ordering::SeqCst);
                    queued.fetch_sub(1, Ordering::SeqCst);
                    admit_job(&engine, job, submitted, &in_flight, &mut active, &stats, limits)
                }
                Err(_) => return,
            }
        }
        while active.len() < MAX_ACTIVE_SESSIONS {
            match rx.try_recv() {
                Ok((submitted, job)) => {
                    in_flight.fetch_add(1, Ordering::SeqCst);
                    queued.fetch_sub(1, Ordering::SeqCst);
                    admit_job(&engine, job, submitted, &in_flight, &mut active, &stats, limits)
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

        // --- s38: 切断スイープ + P57: 上限超過スイープ (各ステップ境界で評価) ---
        // 前ステップ以降にクライアントが切断した Run (SSE 応答ボディの drop →
        // events チャネル閉鎖) を打ち切る。待ち行列 pull 直後〜prefill 前の切断も
        // ここで拾うため、切断済みクライアントの分の無駄な prefill を実行しない。
        // P57: 併せて「進まない Run」(無進捗が上限超過 / SSE 送信が詰まった) も
        // ここで打ち切る — ワーカーが 1 本の Run に引きずられて永久に回り続ける
        // ことを防ぐ (スロットとエンジンロックはこの境界で必ず解放される)。
        let mut i = active.len();
        while i > 0 {
            i -= 1;
            if active[i].client_gone() {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                stats.aborted_disconnected.fetch_add(1, Ordering::Relaxed);
                let run = active.remove(i);
                abort_disconnected(&engine, run);
            } else if let Some(why) = expired(&active[i], limits) {
                in_flight.fetch_sub(1, Ordering::SeqCst);
                stats.aborted_timeout.fetch_add(1, Ordering::Relaxed);
                let mut run = active.remove(i);
                release_run(&engine, &mut run);
                send_error(&mut run, why, limits.send);
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
            match step {
                Ok(_) => active[i].last_progress = Instant::now(),
                Err(e) => {
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    stats.aborted_error.fetch_add(1, Ordering::Relaxed);
                    let mut run = active.remove(i);
                    release_run(&engine, &mut run);
                    send_error(&mut run, e, limits.send);
                }
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
        let send_timeout = limits.send;
        let step = {
            let mut boxed_cbs: Vec<Box<dyn FnMut(&str) + '_>> = asms
                .iter_mut()
                .zip(active.iter())
                .map(|(asm, run)| -> Box<dyn FnMut(&str) + '_> {
                    let events = run.events.clone();
                    // P57: 読み手が止まったクライアントで**エンジンロックを握ったまま**
                    // `blocking_send` が永久ブロックしないよう、有界送信にして
                    // 詰まったらフラグを立てる (打ち切りはステップ境界で行う)。
                    let dead = run.send_dead.clone();
                    // P793: OpenAI ストリームは生 delta をそのまま流す (アセンブラ無し)
                    let raw = run.raw.clone();
                    let cb: Box<dyn FnMut(&str) + '_> = match asm {
                        Some(asm) => Box::new(move |delta: &str| {
                            // 第27: block 境界検出付きの SSE 組み立て (tool_use block も流す)
                            asm.push_delta(delta);
                            for ev in asm.take_events() {
                                if let Some(events) = &events {
                                    if !send_bounded(events, to_event(&ev), send_timeout) {
                                        dead.store(true, Ordering::Relaxed);
                                        return;
                                    }
                                }
                            }
                        }),
                        None => match raw {
                            Some(raw) => Box::new(move |delta: &str| {
                                if !send_raw_bounded(
                                    &raw,
                                    RawEvent::Delta(delta.to_string()),
                                    send_timeout,
                                ) {
                                    dead.store(true, Ordering::Relaxed);
                                }
                            }),
                            None => Box::new(move |_: &str| {
                                // 非ストリーム (Job::Chat) は送信先が無い → no-op
                                let _ = (&events, &dead);
                            }),
                        },
                    };
                    cb
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
                let now = Instant::now();
                for (run, asm) in active.iter_mut().zip(asms) {
                    run.asm = asm;
                    // P57: ステップが 1 つ進んだ = この Run は生きている
                    run.last_progress = now;
                }
                // 完了/切断したセッションを外して後処理 (後ろから外すことで添字を維持)
                // P869: logprobs は完了処理より先に流す (Done の後に取り残さない)
                for run in active.iter_mut() {
                    flush_raw_logprobs(run, limits.send);
                }
                let mut i = done.len();
                while i > 0 {
                    i -= 1;
                    if done[i] {
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        stats.completed.fetch_add(1, Ordering::Relaxed);
                        let mut run = active.remove(i);
                        release_run(&engine, &mut run);
                        finish_run(run, limits.send);
                    } else if active[i].client_gone() {
                        // s38/s44: decode ステップ中にクライアントが切断。
                        // Stream: 書き込み失敗 → 応答ボディ drop → チャネル閉鎖。
                        // Chat: hyper が handler future をキャンセル → respond 閉鎖。
                        // 生成を打ち切って KV スロットを解放する (通常完了と同一の
                        // 除去経路なので他 Run / engine 状態への影響は無い)
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        stats.aborted_disconnected.fetch_add(1, Ordering::Relaxed);
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
                    stats.aborted_error.fetch_add(1, Ordering::Relaxed);
                    release_run(&engine, run);
                }
                for mut run in runs {
                    let msg = format!("{e:#}");
                    send_error(&mut run, anyhow::anyhow!(msg), limits.send);
                }
            }
        }
    }
}

/// P57: 打ち切り理由 (上限を超えていなければ None)。
fn expired(run: &Run, limits: Limits) -> Option<anyhow::Error> {
    // P873: リクエストの壁時計上限を最優先で見る (admit / stall より厳しい場合が多い)。
    if let Some(deadline) = run.deadline {
        if Instant::now() >= deadline {
            return Some(timeout_error("request wall clock", limits.request));
        }
    }
    if run.send_dead.load(Ordering::Relaxed) {
        return Some(timeout_error(
            "SSE event send (client stopped reading)",
            limits.send,
        ));
    }
    if run.last_progress.elapsed() >= limits.stall {
        return Some(timeout_error("generation progress", limits.stall));
    }
    None
}

/// セッションをエンジンへ返す (`install_session` 前なら no-op)。
fn release_run(engine: &Arc<Mutex<Engine>>, run: &mut Run) {
    if let Some(sess) = run.sess.as_mut() {
        engine.lock().unwrap().release_session(sess);
    }
}

/// ジョブから実行状態を作る (begin_chat まで。KV には触れない)。
fn build_run(engine: &Arc<Mutex<Engine>>, job: Job, submitted: Instant, limits: Limits) -> Run {
    let (system, messages, params, tools, constraint, enable_thinking, raw_prompt, respond, events, raw, logprobs) = match job {
        Job::Chat { system, messages, params, tools, constraint, enable_thinking, raw_prompt, respond, logprobs } => {
            (system, messages, params, tools, constraint, enable_thinking, raw_prompt, Some(respond), None, None, logprobs)
        }
        Job::Stream { system, messages, params, tools, constraint, enable_thinking, events } => {
            (system, messages, params, tools, constraint, enable_thinking, None, None, Some(events), None, None)
        }
        Job::OpenaiStream { system, messages, params, tools, constraint, enable_thinking, raw_prompt, events, logprobs } => {
            (system, messages, params, tools, constraint, enable_thinking, raw_prompt, None, None, Some(events), logprobs)
        }
    };
    let (sess, error) = {
        let g = engine.lock().unwrap();
        // P837: `raw_prompt` があれば chat template を通さない (completion 経路)
        let built = match raw_prompt.as_deref() {
            Some(p) => g.begin_completion(p, &params),
            None => g.begin_chat_with_tools(
                system.as_deref(),
                &messages,
                &params,
                tools.as_ref(),
                constraint.as_ref(),
                enable_thinking,
            ),
        };
        match built {
            Ok(mut s) => {
                // P869: logprobs 要求時だけセッションに記録を指示する
                if let Some(top) = logprobs {
                    s.enable_logprobs(top);
                }
                (Some(s), None)
            }
            Err(e) => (None, Some(e)),
        }
    };
    // Stream ジョブは SSE アセンブラを持つ (index 0 の text block は admit が開ける)。
    // P793: OpenAI ストリームは生 delta をそのまま流すのでアセンブラを持たない。
    let asm = match (respond.is_some(), events.is_some()) {
        (false, true) => Some(StreamAssembler::new(0)),
        _ => None,
    };
    // s151 (B12): リクエストの壁時計はここで開く (セッションの生成区間 = prefill
    // 開始から完了まで)。計測 OFF は None (クロック読みなし)。
    let req_mark = prof::request_begin();
    // P57: 無進捗の起点は「受付時刻」。prefill が始まるまでの待ちもここから測る。
    let last_progress = submitted;
    // P873: リクエストの壁時計上限 (0 = 無効)。受付時刻からの絶対期限にする。
    let deadline = if limits.request.is_zero() {
        None
    } else {
        Some(submitted + limits.request)
    };
    Run {
        sess,
        error,
        respond,
        events,
        raw,
        started: false,
        asm,
        req_mark,
        last_progress,
        deadline,
        send_dead: Arc::new(AtomicBool::new(false)),
    }
}

/// 新規ジョブの admit (pull 経路 2 箇所の共通化)。成功なら active へ push する。
/// admit 失敗時は send_error で応答するが、s38: 切断由来 (events チャネル閉鎖 —
/// SSE プリアンブル送信が broken pipe で失敗する等 / s44: respond Receiver drop —
/// 待ち行列中にクライアントが切断した場合) の場合はエラーではなく
/// 打ち切りとして扱う (クライアントは既に居ないため送信先が存在しない)。
///
/// P57: 受付から [`Limits::admit`] を超えて待たされたジョブは、スロットを
/// 取らずに 500 (api_error) で返す。「待たせ続ける」より「期限を切って返す」。
fn admit_job(
    engine: &Arc<Mutex<Engine>>,
    job: Job,
    submitted: Instant,
    in_flight: &AtomicUsize,
    active: &mut Vec<Run>,
    stats: &RunStats,
    limits: Limits,
) {
    // P817: `in_flight` の加算は呼び出し側 (pull 直後) で `queued` の減算より
    // **先** に行う。ここで加算すると「queued も in_flight も 0」に見える窓が
    // でき、drain が進行中のジョブを見落とす。
    let mut run = build_run(engine, job, submitted, limits);
    // P57: キュー待ち + スロット待ちの上限。スロットはまだ取っていないので
    // 解放は不要 (取得済みでも release_run は no-op)。
    // P873: `request` が有効ならその方が短いときに adopt する (期限切れのジョブを
    // 生成開始させない)。
    let budget = limits.admit_budget();
    let waited = submitted.elapsed();
    if waited >= budget {
        in_flight.fetch_sub(1, Ordering::SeqCst);
        stats.aborted_timeout.fetch_add(1, Ordering::Relaxed);
        let why = timeout_error(
            &format!("queue/slot wait ({:.1}s)", waited.as_secs_f64()),
            budget,
        );
        send_error(&mut run, why, limits.send);
        return;
    }
    // s146: このジョブより先に KV を握っているセッション数 (分類の入力)
    let live = active.len();
    if let Err(e) = admit(engine, &mut run, live, limits.send) {
        in_flight.fetch_sub(1, Ordering::SeqCst);
        if run.client_gone() {
            stats.aborted_disconnected.fetch_add(1, Ordering::Relaxed);
            abort_disconnected(engine, run);
        } else {
            stats.aborted_error.fetch_add(1, Ordering::Relaxed);
            // s144 (B8-1): `admit` は install (スロット取得) の **後** にも失敗しうる
            // (SSE プリアンブルの送信失敗など)。旧コードはこの経路で send_error だけを
            // 行い、取得済みスロットの busy フラグを立てたまま返していた。そのため
            // 中断されたリクエストが数回続くと全スロットが busy で残り、健全なサーバでも
            // 以後の全リクエストが "no free KV cache slot" で失敗した
            // (DENNO_BATCH_SLOTS=4 が回避策として効いたのはこれが理由)。
            // `release_session` は slot 未取得なら no-op なので、全経路で呼んでよい。
            release_run(engine, &mut run);
            send_error(&mut run, e, limits.send);
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
    release_run(engine, &mut run);
    // s151 (B12): 打ち切られたリクエストも壁時計を 1 本だけ閉じる (未計上で
    // `requests` と `wall` がずれると COVERAGE の分母が嘘になる)。
    prof::request_end(run.req_mark.take());
    tracing::info!("client disconnected; generation aborted");
}

/// s146 (B8-3 / P4-9): `install_session` 失敗のクライアント向け分類。
///
/// 旧コードは全失敗を文脈文字列 1 本 ("KVキャッシュスロットの取得に失敗") で
/// 包んでいたため、クライアントは「容量不足 (待てば通る)」と「設定上限」
/// 「内部異常」を区別できなかった。ここでは原因 (エンジン側メッセージ) を
/// 見て 3 種に分ける:
///
/// | 分類 | 条件 | クライアントの取るべき行動 |
/// |---|---|---|
/// | `SlotsBusy` | engine の "no free KV cache slot (N slots)" | 実行中が終わるのを待って再試行 |
/// | `SessionLimit` | 同時実行が MAX_ACTIVE_SESSIONS に達した (防御) | 実行中が終わるのを待って再試行 |
/// | `Internal` | それ以外 (モデル未ロード・予算超過・未知のエラー) | 設定/ログを確認 (再試行で解決しない) |
///
/// 分類は既に Err で返ってきた失敗に対する **読み取りのみ** で、エンジンの
/// 状態・数値経路には触れない。`install_session` は Err 時にスロットを
/// 取得しないため、この時点の slot_busy 計数は「他リクエストが握っている数」
/// と等しい。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdmitKind {
    /// (a) 全 KV スロットが実行中リクエストに占有されている — 一時的な容量不足
    SlotsBusy,
    /// (b) 同時実行数が [`MAX_ACTIVE_SESSIONS`] に達した
    SessionLimit,
    /// (c) モデル未ロード・KV 予算超過など内部要因
    Internal,
}

impl AdmitKind {
    /// Anthropic Messages API の `error.type` に対応する文字列。
    /// 容量系は 429/503 の rate limit 系、内部は api_error。
    pub fn error_type(self) -> &'static str {
        match self {
            AdmitKind::SlotsBusy | AdmitKind::SessionLimit => "rate_limit_error",
            AdmitKind::Internal => "api_error",
        }
    }
}

/// クライアントに返す分類済みの失敗。`Display` = クライアント向け本文、
/// `source` = エンジン側の元エラー (tracing の原因連鎖で保持する)。
///
/// `anyhow::Error` として `send_error` まで運ばれるが、ハンドラは
/// [`ApiError::from_anyhow`] で `downcast_ref` して HTTP ステータスと
/// `error.type` を決める — 文字列マッチでステータスを決めない。
#[derive(Debug)]
pub struct AdmitFailure {
    kind: AdmitKind,
    message: String,
    cause: anyhow::Error,
}

impl AdmitFailure {
    fn new(kind: AdmitKind, message: String, cause: anyhow::Error) -> Self {
        Self { kind, message, cause }
    }

    /// 分類する。`install_session` が返したエラー `e` を読み、エンジンの
    /// 計数 (スロット数 / 使用中スロット数) を添えて本文を作る。
    ///
    /// `live` は「このワーカーが今 KV を握っているセッション数」— スケジューラの
    /// `active.len()`。`slot_busy` は install 失敗時点ではまだ立っていないため
    /// (エンジンは Err のときスロットを取らない)、busy 計数だけでは
    /// 同時実行上限の判定ができない。呼び出し側 (worker) は自分の active 長を
    /// 知っているのでそれを渡す。
    pub fn classify(e: anyhow::Error, engine: &Engine, live: usize) -> Self {
        let text = format!("{e:#}");
        let slots = engine.slot_count();
        // 報告用の使用中数: live (ワーカーが把握している常駐数) と
        // エンジンの busy 計数の大きい方 (防御的に両方の観測を反映する)
        let busy = engine.slot_busy_count().max(live).min(slots);
        // engine.rs の枯渇エラーは "no free KV cache slot (N slots)"。
        // 文字列はエンジン側の公開文言 (テスト slot_leak.rs も同じ語で判定)。
        if text.contains("no free KV cache slot") {
            return Self::new(
                AdmitKind::SlotsBusy,
                format!(
                    "capacity: all {slots} KV cache slots are busy ({busy}/{slots} in use); \
                     this request was not started. action: retry after an in-flight \
                     request finishes, or raise the slot count (`{}={}`) and restart",
                    denno_engine::engine::BATCH_SLOTS_ENV,
                    slots
                ),
                e,
            );
        }
        // 防御: ハンドラとワーカーの間で状態がずれた場合 (スロットが空いていても
        // 同時実行上限に達している)。枯渇文言以外の失敗でも、常駐数が上限に
        // 達していればクライアントの取るべき行動は「待って再試行」で同じ。
        if live >= MAX_ACTIVE_SESSIONS {
            return Self::new(
                AdmitKind::SessionLimit,
                format!(
                    "capacity: concurrent request limit reached \
                     (MAX_ACTIVE_SESSIONS = {MAX_ACTIVE_SESSIONS}, {busy}/{slots} slots in use); \
                     this request was not started. action: retry after an in-flight \
                     request finishes, or raise the slot count (`{}={}`) and restart",
                    denno_engine::engine::BATCH_SLOTS_ENV,
                    slots
                ),
                e,
            );
        }
        let cause_text = text;
        Self::new(
            AdmitKind::Internal,
            format!(
                "internal: request could not be admitted (model loaded = true, \
                 {busy}/{slots} KV slots in use); retrying will not help. \
                 action: check the server log and the KV/session settings \
                 (cause: {cause_text})"
            ),
            e,
        )
    }

    pub fn kind(&self) -> AdmitKind {
        self.kind
    }

    /// クライアントに返す 1 行本文。
    pub fn client_message(&self) -> &str {
        &self.message
    }

    /// `anyhow::Error` へ戻す (スケジューラ / ハンドラと同じ運び方をするため)。
    /// downcast で分類 (kind) を取り出せる状態を保つ。
    pub fn into_anyhow(self) -> anyhow::Error {
        anyhow::Error::new(self)
    }

    /// 元のエラー (エンジン側の原因) の全文。ログ用。`send_error` が
    /// `{e:#}` で記録する内容と同じ連鎖を、分類器の単体テストからも読める。
    pub fn cause_chain(&self) -> String {
        format!("{:#}", self.cause)
    }
}

impl std::fmt::Display for AdmitFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for AdmitFailure {
    /// 原因 (エンジン側エラー) を **そのまま** 返す。`self.cause.source()` を
    /// 返すと、原因自身が source を持たない場合に連鎖が丸ごと消える
    /// (`{e:#}` にエンジン原文が残らなくなる — s146 で実測)。
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.cause.as_ref())
    }
}

/// セッションを live にする (install) + 初回なら SSE プリアンブルを送る。
///
/// P57: プリアンブル送信は [`send_bounded`] にする。旧実装の `blocking_send` は
/// 読み手が止まったクライアントで**ワーカースレッドを永久にブロック**し得た
/// (この時点でエンジンロックは離しているが、ワーカーが止まればサーバ全体が止まる)。
fn admit(engine: &Arc<Mutex<Engine>>, run: &mut Run, live: usize, send_timeout: Duration) -> Result<()> {
    if let Some(e) = run.error.take() {
        return Err(e);
    }
    {
        let mut g = engine.lock().unwrap();
        let sess = run.sess.as_mut().unwrap();
        if let Err(e) = g.install_session(sess) {
            // s146 (B8-3): install_session の失敗を **分類してから** 返す。
            // 旧コードはここで `e.context("KVキャッシュスロットの取得に失敗")` を
            // 付けていたため、クライアントが見る先頭文字列が汎用プレフィクスになり、
            // 本来の原因 (engine.rs の "no free KV cache slot (N slots)") が
            // その後ろに埋もれていた。分類は計数用の再計算のみで、エンジンの
            // 状態・数値には一切触れない (install_session は Err 時に slot を取らない)。
            // `live` はこの時点の active 長 (= 今 KV を握っているセッション数)。
            return Err(AdmitFailure::classify(e, &g, live).into());
        }
    }
    if run.events.is_some() && !run.started {
        run.started = true;
        let events = run.events.as_ref().unwrap();
        let model = engine.lock().unwrap().name.clone();
        let input_tokens = run.sess.as_ref().unwrap().input_tokens();
        // P805: prefix cache から再利用した長さ (Anthropic 互換 usage)。
        let reused = run.sess.as_ref().unwrap().reused_prefix_tokens();
        let send = |v: serde_json::Value| -> bool {
            send_bounded(events, to_event(&v), send_timeout)
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
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "cache_read_input_tokens": reused,
                    "cache_creation_input_tokens": input_tokens.saturating_sub(reused),
                },
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
    // P793: OpenAI 互換ストリームのプリアンブル (入力トークン数だけ先に渡す)。
    // `events` の側とは排他なので、両方送られることはない。
    if run.raw.is_some() && !run.started {
        run.started = true;
        let raw = run.raw.as_ref().unwrap().clone();
        let input_tokens = run.sess.as_ref().unwrap().input_tokens();
        if !send_raw_bounded(&raw, RawEvent::Start { input_tokens }, send_timeout) {
            bail!("sse send failed (client disconnected)");
        }
    }
    Ok(())
}

/// P869: セッションに溜まった logprobs を [`RawEvent::Logprob`] として流す
/// (OpenAI ストリームのみ)。送信に失敗したら `send_dead` を立てる
/// (次のステップ境界で打ち切られる)。
fn flush_raw_logprobs(run: &mut Run, send_timeout: Duration) {
    let Some(raw) = run.raw.clone() else { return };
    let dead = run.send_dead.clone();
    let Some(sess) = run.sess.as_mut() else { return };
    for lp in sess.take_logprobs() {
        if !send_raw_bounded(&raw, RawEvent::Logprob(lp), send_timeout) {
            dead.store(true, Ordering::Relaxed);
            return;
        }
    }
}

/// 完了したセッションの後処理 (respond or SSE 後半イベント)。
fn finish_run(mut run: Run, send_timeout: Duration) {
    // s151 (B12): 生成が終わった点でリクエストの壁時計を閉じる。JSON 応答でも
    // SSE でも同じ 1 本で、Run の終端は全経路でここ (または abort) を通る。
    let mark = run.req_mark.take();
    prof::request_end(mark);
    let Run { sess, respond, events, raw, mut asm, .. } = run;
    let Some(sess) = sess else { return };
    let out = sess.into_output();
    if let Some(respond) = respond {
        let _ = respond.send(Ok(out));
        return;
    }
    if let Some(events) = events {
        // P57: 読み手が止まったクライアントで永久ブロックしない (有界送信)
        let send = |v: serde_json::Value| -> bool {
            send_bounded(&events, to_event(&v), send_timeout)
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
        return;
    }
    // P793: OpenAI 互換ストリームの終端 (使用量 + stop_reason)。
    if let Some(raw) = raw {
        let _ = send_raw_bounded(
            &raw,
            RawEvent::Done {
                input_tokens: out.input_tokens,
                output_tokens: out.output_tokens,
                stop_reason: out.stop_reason.as_str().to_string(),
                stop_sequence: out.stop_sequence.clone(),
            },
            send_timeout,
        );
    }
}

/// セッションをエラーで終わらせる (respond or SSE error イベント)。
///
/// s146 (B8-3): クライアント向け本文 (`{e}`) とログ用の全文 (`{e:#}`) を
/// 分ける。`AdmitFailure` の `Display` は分類済みの 1 行だけを返すため、
/// ログ側は `{e:#}` (cause 連鎖 — エンジン原文 "no free KV cache slot
/// (N slots)" 等) を必ず別途記録し、診断情報を落とさない。
fn send_error(run: &mut Run, e: anyhow::Error, send_timeout: Duration) {
    let detail = format!("{e:#}");
    let client_msg = format!("{e}");
    tracing::error!("chat failed: {detail}");
    // s151 (B12): 失敗で終わるリクエストも壁時計を 1 本だけ閉じる
    // (`request_end` は `take()` 済みなら no-op。完了経路と二重計上しない)。
    prof::request_end(run.req_mark.take());
    if let Some(respond) = run.respond.take() {
        let _ = respond.send(Err(e));
    } else if let Some(events) = run.events.take() {
        // エラーは error イベントで通知して閉じる (読み手が止まっていれば諦める)
        let _ = send_bounded(
            &events,
            to_event(&json!({
                "type": "error",
                "error": {"type": "api_error", "message": client_msg}
            })),
            send_timeout,
        );
    } else if let Some(raw) = run.raw.take() {
        // P793: OpenAI 経路は生イベントで同じ通知をする (HTTP 側が error chunk へ)
        let _ = send_raw_bounded(&raw, RawEvent::Error(client_msg), send_timeout);
    }
}

/// P793: [`send_bounded`] の [`RawEvent`] 版 (OpenAI 互換ストリーム)。
/// 意味論は同一 — 詰まった受信バッファには timeout まで 1ms sleep で再試行し、
/// 諦めたら false (= 呼び出し側が Run を打ち切る)。
fn send_raw_bounded(
    events: &tokio_mpsc::Sender<RawEvent>,
    ev: RawEvent,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    let mut pending = ev;
    loop {
        match events.try_send(pending) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => return false,
            Err(TrySendError::Full(back)) => {
                pending = back;
                if Instant::now() >= deadline {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

/// P57: 有界の SSE 送信。
///
/// 旧実装は `blocking_send` だった — 受信バッファ (64 本) が詰まったクライアント
/// (読み手が止まっている) では**永久にブロック**し、decode の delta コールバックは
/// エンジン Mutex を握ったまま呼ばれるため、そのままサーバ全体が停止した。
///
/// ここでは `try_send` を `timeout` まで **短い sleep を挟んで** 再試行する
/// (ビジーループにしない・CPU を焼かない)。諦めたら false を返し、呼び出し側が
/// その Run を打ち切る。
fn send_bounded(
    events: &tokio_mpsc::Sender<Result<Event, Infallible>>,
    ev: Event,
    timeout: Duration,
) -> bool {
    let deadline = Instant::now() + timeout;
    let mut pending = ev;
    loop {
        match events.try_send(Ok(pending)) {
            Ok(()) => return true,
            Err(TrySendError::Closed(_)) => return false,
            Err(TrySendError::Full(back)) => {
                pending = match back {
                    Ok(ev) => ev,
                    Err(_) => return false, // 型上あり得ない (Ok で送っている)
                };
                if Instant::now() >= deadline {
                    return false;
                }
                // 1ms 未満の sleep は OS 依存でスピンに近づくため 1ms 固定。
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}
