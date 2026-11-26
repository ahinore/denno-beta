//! Anthropic Messages API 互換 HTTP サーバー。
//! POST /v1/messages (非ストリーミング + SSE)、GET /v1/models、GET /health。

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use denno_engine::{ChatMessage, ChatOutput, Engine, GenParams, QuantMode, ToolConstraint};
use denno_telemetry::{self as telem, Frame};
use rand::Rng;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

mod lifecycle;
mod logging;
/// P877: Prometheus `/metrics` (依存クレート無し) の計器。
pub mod metrics;
/// P837: OpenAI Structured Outputs のスキーマ検証 (`response_format: json_schema`)。
pub mod json_schema;
mod openai;

/// task 931: 任意オプトインのレート制限 (env 未設定なら無効)。
pub mod ratelimit;
/// task 971: 起動サマリの整形 (純関数。生の API キーは持たない)。
pub mod startup;
// P841: Ollama 互換 facade (`/api/tags` / `/api/version` / `/api/chat` / `/api/generate`)
mod ollama;
mod scheduler;
mod stream_asm;
pub use lifecycle::{Lifecycle, DEFAULT_DRAIN_TIMEOUT_MS};
pub use logging::{
    access_log_json, access_log_line, init_serve_logging, set_access_log_json, NonBlockingStdout,
};
pub use scheduler::{
    AdmitFailure, AdmitKind, Job, Limits, RunStatsSnapshot, MAX_ACTIVE_SESSIONS,
    QUEUE_RETRY_AFTER_SECS, QueueFull, Scheduler,
};
pub use startup::{
    format_startup_summary, summarize_auth, AuthState, CorsSummary, RateLimitSummary,
    StartupSummary,
};

/// P57: エンジンの **不変情報**。起動時に 1 回だけ読み、以降はハンドラが
/// `engine.lock()` を取らずに応答するために使う。
///
/// 背景 (2026-10-25 本番障害): `/v1/models` は名前と context を読むためだけに
/// エンジン Mutex を取っていた (旧 lib.rs:900)。GPU チェーン失敗 →
/// CPU フォールバックで 1 ステップが数分に伸びた際、ワーカーが Mutex をほぼ連続
/// 保持したため `/v1/models` がロック待ちで **200 すら返せず**、サーバ全体が
/// 無応答に見えた。制御プレーンは生成の直列化に巻き込まない。
#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// コンテキスト長 (トークン)
    pub context: usize,
    /// 量子化モードのラベル (`QuantMode::label`)
    pub quant: String,
    /// KV キャッシュスロット数
    pub slots: usize,
    /// モデルの層数
    pub layers: usize,
    /// P817: `--model-dir` の basename (`/v1/models` の `model_dir`)。
    /// エンジンはディレクトリを知らないので `from_engine` では空、
    /// `run` がロード後に埋める。
    pub model_dir: String,
}

impl EngineInfo {
    /// エンジンから 1 回だけ読む (起動時 / テストの fixture 構築時)。
    pub fn from_engine(e: &Engine) -> Self {
        Self {
            context: e.context(),
            quant: e.quant.label().to_string(),
            slots: e.caches.len(),
            layers: e.caches.first().map(|c| c.n_layers()).unwrap_or(0),
            model_dir: String::new(),
        }
    }
}

pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    /// task 951: 受理する API キーの一覧を **カンマ区切りの 1 本** で持つ
    /// (`--api-key` / `--api-keys` / `DENNO_API_KEYS` の和を
    /// [`resolve_api_keys`] が正規化したもの)。None = 認証なし (従来どおり開放)。
    /// 値は決してログ / 応答に出さない。
    pub api_key: Option<String>,
    /// P4: リクエストスケジューラ (有界キュー + 単一ワーカー)
    pub scheduler: Scheduler,
    /// モデル名 (不変)。ハンドラが engine.lock() を取らずに済むようにキャッシュ。
    /// (ワーカーは mutex をステップ間でほぼ連続保持するため、ハンドラが
    ///  generation 中に lock を奪おうとすると starve し得る)
    pub model_name: String,
    /// 実行中のモデル ID (例: Qwen/Qwen2.5-0.5B-Instruct)。テレメトリ ServerInfo 用。
    pub model_id: String,
    /// P57: 生成中でもロックを取らずに応答するための不変情報
    /// ([`EngineInfo`])。ハンドラは **これを読む** (engine.lock() を取らない)。
    pub info: EngineInfo,
    /// s56b: serve 既定の repetition_penalty。リクエストが未指定のときだけ適用する。
    /// 既定 None (= 従来と完全に同一のサンプリング経路) なので、
    /// フラグを付けない起動では出力ビット不変 (アンカー不変)。
    pub default_repetition_penalty: Option<f32>,
    /// s67b: temperature の上限。Some(c) のとき、実効 temperature を c 以下に
    /// クランプする (リクエスト未指定なら sampler 既定の 1.0 を c で頭打ち)。
    /// 既定 None = クランプ無し = 従来とビット同一。
    /// 目的: thinking 有効時にクライアントが機械的に temperature=1 を送る実装
    /// (Hermes 等) に対し、小型モデルの崩壊を運用側で防ぐ。
    pub temperature_cap: Option<f32>,
}

/// serve 設定
pub struct ServeConfig {
    pub model_dir: std::path::PathBuf,
    pub name: String,
    /// モデル ID (pull 元の repo id)。テレメトリで報告する
    pub model_id: String,
    pub host: String,
    pub port: u16,
    /// 単一キー (従来)。task 951: カンマ区切りも受け付ける。
    pub api_key: Option<String>,
    /// task 951: 複数キー (`--api-keys`)。`api_key` と `DENNO_API_KEYS` の
    /// 和が実効キー一覧になる。None = 未指定。
    pub api_keys: Option<String>,
    pub default_seed: Option<u64>,
    /// マルチエントリ prefix cache のトークン上限 (None なら env/デフォルトのまま)
    pub prefix_cache_tokens: Option<usize>,
    /// 重みの量子化モード (P3。デフォルト fp32。int8 は CLI でのみ有効化)
    pub quant: QuantMode,
    /// P4: リクエストキューの上限 (実行中 1 件の他に待ち行列に入れられる数)。
    /// 同時受け付けがこれを超えると 503 (overloaded_error) を返す。
    /// キューに入ったリクエストは実行開始まで待つが、待ちには上限がある
    /// (P57: `scheduler::Limits` — キュー待ち + スロット待ちが 300s を超えたら
    /// 500 で返してスロットを解放する。旧実装は上限が無かった)。
    ///
    /// P873: 実際の値は [`resolve_max_queue_depth`] が決める (`--max-queue-depth`
    /// > `DENNO_MAX_QUEUE_DEPTH` > このフィールド > 既定 32)。
    pub max_queue: usize,
    /// P873: `--max-queue-depth` (未指定 = env / `max_queue` / 既定に従う)。
    pub max_queue_depth: Option<usize>,
    /// P873: `--request-timeout-secs` (None = env / 既定 0 = 無効)。
    pub request_timeout_secs: Option<u64>,
    /// s56b: serve 既定の repetition_penalty (None = リクエスト指定のみ)。
    /// Qwen3.5-9B int8 は長い system + tools の greedy 経路で崩壊するため、
    /// 運用では 1.05 を指定する (実測: 崩壊解消・短答化)。
    pub repetition_penalty: Option<f32>,
    /// s67b: temperature の上限 (None = クランプ無し)。運用では 0.7 程度を指定する。
    pub temperature_cap: Option<f32>,
    /// task 916: CORS の許可オリジン。None = `DENNO_CORS_ORIGIN` → 既定
    /// (ループバックのみ)。`*` またはカンマ区切りのオリジン一覧。
    pub cors_origin: Option<String>,
    /// task 946: HTTP リクエストボディの上限 (MiB)。None = `DENNO_MAX_BODY_MB`
    /// → 既定 16 MiB。超過は 413 + JSON 封筒。
    pub max_body_mb: Option<u64>,
}

// ---------- Anthropic 形式エラー ----------

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
    /// P873: `Retry-After` ヘッダ (秒)。None = 付けない。
    retry_after: Option<u64>,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), retry_after: None }
    }
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }
    pub fn auth() -> Self {
        Self::new(StatusCode::UNAUTHORIZED, "authentication_error", "invalid x-api-key")
    }
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, "not_found_error", message)
    }
    /// キュー満杯 (同時受け付け上限超過)。Anthropic 流の overloaded_error。
    pub fn overloaded(message: impl Into<String>) -> Self {
        Self::new(StatusCode::SERVICE_UNAVAILABLE, "overloaded_error", message)
    }
    /// P873: 過負荷 + `Retry-After` (秒)。クライアントに「いつ再試行してよいか」を
    /// 明示する (Anthropic / OpenAI / Ollama のいずれも待ちを推奨する応答で使う)。
    pub fn overloaded_retry_after(message: impl Into<String>, secs: u64) -> Self {
        let mut e = Self::overloaded(message);
        e.retry_after = Some(secs);
        e
    }
    fn api(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", message)
    }
    /// s146 (B8-3 / P4-9): admit 失敗 (KV スロット枯渇など) の分類済みエラー。
    ///
    /// ステータスの選択 (ルート・他のエラー経路は不変):
    /// - `SlotsBusy` → **429 Too Many Requests**。全スロットが実行中リクエストに
    ///   占有されている状態は **一時的な容量不足** で、待てば必ず通る。
    ///   429 は「後で再試行」を意味する標準的な信号 (Retry-After なしでも可)。
    ///   `error.type` は Anthropic 互換の `rate_limit_error` — クライアントの
    ///   再試行判定はこの型で行われる想定。
    /// - `SessionLimit` (防御経路: 同時実行上限) → 503 overloaded_error。
    ///   これは同じファイルのキュー満杯 (`queue_full_error`) と同じ「過負荷」の
    ///   意味論なので、既存の 503 overloaded_error に揃える。
    /// - `Internal` → 500 api_error。再試行で解決しない内部要因。
    fn from_admit_failure(f: &scheduler::AdmitFailure) -> Self {
        use scheduler::AdmitKind;
        match f.kind() {
            AdmitKind::SlotsBusy => Self::new(
                StatusCode::TOO_MANY_REQUESTS,
                AdmitKind::SlotsBusy.error_type(),
                f.client_message(),
            ),
            AdmitKind::SessionLimit => Self::overloaded(f.client_message()),
            AdmitKind::Internal => Self::api(f.client_message()),
        }
    }
    /// スケジューラから返った `anyhow::Error` を HTTP エラーへ変換する。
    /// 分類済みの [`scheduler::AdmitFailure`] は種類で、それ以外は従来どおり
    /// 文字列でステータスを決める (既存挙動を変えない)。
    pub(crate) fn from_anyhow(e: anyhow::Error) -> Self {
        if let Some(f) = e.downcast_ref::<scheduler::AdmitFailure>() {
            return Self::from_admit_failure(f);
        }
        let msg = format!("{e:#}");
        if msg.contains("invalid_request_error") {
            Self::invalid_request(msg)
        } else {
            Self::api(msg)
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "type": "error",
            "error": {"type": self.kind, "message": self.message}
        });
        let mut resp = (self.status, Json(body)).into_response();
        // P873: 過負荷のときは再試行までの待ちをヘッダで示す。
        if let Some(secs) = self.retry_after {
            if let Ok(v) = axum::http::HeaderValue::from_str(&secs.to_string()) {
                resp.headers_mut().insert(axum::http::header::RETRY_AFTER, v);
            }
        }
        resp
    }
}

impl ApiError {
    /// P873: このエラーが要求する `Retry-After` (秒)。HTTP 変換とテストが読む。
    pub fn retry_after(&self) -> Option<u64> {
        self.retry_after
    }
}

/// s146: スケジューラ経由の `anyhow::Error` を HTTP エラーへ変換する
/// (ハンドラが使うのと同じ関数。統合テストから admit 失敗のマッピングを
/// 直接検証できるように公開する)。
pub fn error_from_anyhow(e: anyhow::Error) -> ApiError {
    ApiError::from_anyhow(e)
}

/// s146: `ApiError` の中身 (status, error.type, message) をテスト用に取り出す。
/// サーバー挙動には影響しない (読み取りのみ)。
pub fn debug_api_error(e: ApiError) -> (u16, &'static str, String) {
    (e.status.as_u16(), e.kind, e.message)
}

// ---------- リクエストDTO ----------

/// chat 用 max_tokens の既定上限 (リクエストで未指定の場合のみ適用する。
/// 明示指定は尊重 — コンテキストへのクランプは engine 側の既存挙動のまま)。
///
/// 値の根拠: engine の `SESSION_RESERVE_TOKENS` (P17e の 1 セッションあたり
/// KV 予約の目安・engine.rs) と同値で、KV 予算設計の前提と整合する。
/// thinking モデルが max_tokens 未指定のまま超長生成し、単一ワーカー構成の
/// サーバーで後続リクエストを長時間列待ちさせる問題 (8095 運用・s38 の切断
/// 検知はクライアント切断時のみ有効) の対策として、未指定リクエストに上限を
/// 保証する。Anthropic API 仕様上 max_tokens は必須だが、本サーバーは学習用
/// ツールのため未指定を 400 にせず既定上限で受理する。
///
/// s53: 4096 → **1024**。8095 運用で未指定リクエストが 4096 まで走り
/// 単一ワーカーを長時間占有するケースが残ったため、既定を 1/4 に下げる。
/// 明示指定は尊重する (clamp 上限 4096 は `MAX_TOKENS_CLAMP` 側で不変)。
pub const DEFAULT_MAX_TOKENS: usize = 1024;

/// P57: 非ストリーミング (`stream: false`) のハンドラが応答を待つ上限。
///
/// 旧実装は `rx.await` を無期限に待っていた。ワーカーが 1 ステップで詰まる
/// (GPU 失敗 → CPU フォールバック等) と、クライアントは 500 も返らず永久に
/// 無応答になった。上限を超えたら 500 (api_error) を返し、受信側を drop する
/// — ワーカーから見るとクライアント切断と同じなので、次のステップ境界で
/// KV スロットが解放される。
///
/// 値の根拠 (推定): 既定 max_tokens = 1024 の 1 リクエストは 27B で数十秒級。
/// スケジューラ側の `Limits::admit` (300s) と `Limits::stall` (900s) より
/// 短くすると「キュー待ちで 500」が先に出てしまうため、stall と同値の 900s に
/// する (無期限より短ければよい、という設計)。
pub const RESPONSE_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(900);

// ---------- P873: キュー深度 / リクエスト期限の解決 ----------

/// P873: 待ち行列の上限を上書きする env (`--max-queue-depth` が無いとき)。
pub const MAX_QUEUE_DEPTH_ENV: &str = "DENNO_MAX_QUEUE_DEPTH";
/// P873: リクエストの壁時計上限 (秒) を上書きする env。
pub const REQUEST_TIMEOUT_SECS_ENV: &str = "DENNO_REQUEST_TIMEOUT_SECS";
/// P873: 待ち行列の上限の既定値 (旧 `--max-queue` の既定と同じ = 既存挙動)。
pub const DEFAULT_MAX_QUEUE_DEPTH: usize = 32;
/// P873: リクエスト期限の既定値。0 = 無効 (従来と完全に同一の挙動)。
pub const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 0;
/// P873: 待ち行列の上限の上限 (暴走した env / CLI をそのまま確保しない)。
pub const MAX_QUEUE_DEPTH_LIMIT: usize = 65_536;
/// P873: リクエスト期限の上限 (秒)。
pub const MAX_REQUEST_TIMEOUT_SECS: u64 = 86_400;

/// P873: env から非負整数を読む (空白は無視、解析できなければ None)。
fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
}

/// P873: 実効の待ち行列上限。優先順は CLI > env > 既存フィールド > 既定。
/// 0 は 1 に切り上げ、[`MAX_QUEUE_DEPTH_LIMIT`] で頭打ち (sync_channel の長さ)。
pub fn resolve_max_queue_depth(cli: Option<usize>, fallback: usize) -> usize {
    let from_env = env_u64(MAX_QUEUE_DEPTH_ENV).map(|v| v.min(MAX_QUEUE_DEPTH_LIMIT as u64) as usize);
    cli.or(from_env).unwrap_or(fallback).clamp(1, MAX_QUEUE_DEPTH_LIMIT)
}

/// P873: 実効のリクエスト期限 (秒)。優先順は CLI > env > 既定 0 (無効)。
pub fn resolve_request_timeout_secs(cli: Option<u64>) -> u64 {
    cli.or_else(|| env_u64(REQUEST_TIMEOUT_SECS_ENV))
        .unwrap_or(DEFAULT_REQUEST_TIMEOUT_SECS)
        .min(MAX_REQUEST_TIMEOUT_SECS)
}

// ---------- task 946: リクエストボディ上限 ----------

/// `DENNO_MAX_BODY_MB` の env 名 (MiB 単位)。
pub const MAX_BODY_MB_ENV: &str = "DENNO_MAX_BODY_MB";

/// ボディ上限の既定値 (MiB)。長いプロンプト (数万トークン + tools) が
/// 収まる一方で、無制限にはしない。
pub const DEFAULT_MAX_BODY_MB: u64 = 16;

/// ボディ上限の絶対上限 (MiB)。
pub const MAX_BODY_MB_LIMIT: u64 = 4096;

/// task 946: 実効のボディ上限 (MiB)。優先順は CLI > env > 既定 16 MiB。
/// 0 は 1 MiB に、[`MAX_BODY_MB_LIMIT`] を超える値はそこに丸める。
pub fn resolve_max_body_mb(cli: Option<u64>) -> u64 {
    cli.or_else(|| env_u64(MAX_BODY_MB_ENV))
        .unwrap_or(DEFAULT_MAX_BODY_MB)
        .clamp(1, MAX_BODY_MB_LIMIT)
}

/// task 946: [`resolve_max_body_mb`] の byte 版 (`DefaultBodyLimit` 用)。
pub fn resolve_max_body_bytes(cli_mb: Option<u64>) -> usize {
    (resolve_max_body_mb(cli_mb) as usize).saturating_mul(1024 * 1024)
}

/// task 946: 413 の JSON 封筒 (他のエラーと同じ `{"type":"error", ...}` 形)。
pub fn payload_too_large_response(max_body_bytes: usize) -> Response {
    let body = json!({
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": format!(
                "request body too large: the configured limit is {max_body_bytes} bytes"
            )
        }
    });
    (StatusCode::PAYLOAD_TOO_LARGE, Json(body)).into_response()
}

/// task 946: `Content-Length` が上限を超える要求を早期に 413 にし、
/// 抽出器 (`DefaultBodyLimit`) が出す平文 413 を JSON 封筒へ置き換える。
///
/// `Content-Length` を信じない / 持たない chunked ボディは
/// `DefaultBodyLimit` が読み込み中に打ち切り、その 413 をここで包む。
async fn body_limit_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
    max_body_bytes: usize,
) -> Response {
    let declared = req
        .headers()
        .get(axum::http::header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok());
    if let Some(n) = declared {
        if n > max_body_bytes as u64 {
            return payload_too_large_response(max_body_bytes);
        }
    }
    let resp = next.run(req).await;
    if resp.status() == StatusCode::PAYLOAD_TOO_LARGE {
        // ハンドラが自分で JSON の 413 を返した場合はそのまま通す。
        let is_json = resp
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.to_ascii_lowercase().starts_with("application/json"))
            .unwrap_or(false);
        if !is_json {
            return payload_too_large_response(max_body_bytes);
        }
    }
    resp
}

/// P873: ハンドラが応答を待つ上限。リクエスト期限が有効ならその少し後まで
/// 待つ (スケジューラが期限で打ち切ってエラーを返すのを、ハンドラ側の
/// タイムアウトが先回りして握り潰さないため)。無効なら従来の
/// [`RESPONSE_WAIT_TIMEOUT`] (挙動不変)。
pub fn response_wait_timeout(limits: Limits) -> std::time::Duration {
    if limits.request.is_zero() {
        RESPONSE_WAIT_TIMEOUT
    } else {
        limits.request + std::time::Duration::from_secs(30)
    }
}

/// P805: `count_tokens` がエンジンロックを待つ上限。生成中 (ワーカーが Mutex を
/// ほぼ連続保持) でも制御プレーンを永久にブロックしないための上限。
pub const COUNT_TOKENS_LOCK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// s53: 明示指定時の上限 (clamp)。既定 (1024) とは独立。
pub const MAX_TOKENS_CLAMP: usize = 4096;

/// chat 経路の実効 max_tokens を決める (未指定 → `DEFAULT_MAX_TOKENS`)。
/// `max_tokens` を優先し、無ければ OpenAI chat completions 互換の別名
/// `max_completion_tokens` を使う。
///
/// s53: 明示指定は `MAX_TOKENS_CLAMP` (4096) で clamp する。未指定時の既定は
/// 1024 (単一ワーカーの長時間占有を防ぐ)。0 は「prefill のみ」の明示指定として
/// 0 のまま尊重する (clamp は上限側のみ)。
pub fn resolve_max_tokens(
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
) -> usize {
    max_tokens
        .or(max_completion_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS)
        .min(MAX_TOKENS_CLAMP)
}

#[derive(Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    /// 生成上限トークン数。未指定 (max_completion_tokens も未指定) の場合は
    /// 既定上限 `DEFAULT_MAX_TOKENS` を適用する。明示指定は尊重される
    /// (コンテキスト超過分のクランプは engine の既存挙動)。
    pub max_tokens: Option<usize>,
    /// OpenAI chat completions 互換の別名 (max_tokens 未指定時のみ使用。
    /// 両方指定された場合は max_tokens を優先)。
    pub max_completion_tokens: Option<usize>,
    pub messages: Vec<InMessage>,
    pub system: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// A2.5: 反復ペナルティ (denno 拡張・任意。None = 無効)
    pub repetition_penalty: Option<f32>,
    /// s56 (T2): OpenAI 互換 frequency penalty (denno 拡張・任意。None/0 = 無効)。
    /// 適用順は repetition_penalty の直後・presence の直前 (docs/api.md)。
    pub frequency_penalty: Option<f32>,
    /// s56 (T2): OpenAI 互換 presence penalty (denno 拡張・任意。None/0 = 無効)。
    pub presence_penalty: Option<f32>,
    /// s56 (T2): no_repeat_ngram (denno 拡張・任意。None = 無効)。
    /// 1 以下は 400 (文脈 n-gram が作れないため「無効」と区別できない)。
    pub no_repeat_ngram_size: Option<usize>,
    pub stream: Option<bool>,
    /// tool use: ツール定義 (Anthropic 形式)。空配列は tools 無しと同様に扱う
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    /// tool use: "auto" | "none" | "any" | {"type": "auto"|"none"|"any"|"tool", "name"?}
    /// 既定は auto。`any`/`tool` (強制) は拘束デコード (第21) が有効:
    /// モデル生成を tool_call block (公式テンプレート形式) へ強制する。
    pub tool_choice: Option<serde_json::Value>,
    /// s47: thinking 制御。既定は "brief" 相当。
    /// s52: 2 形式を受理する (denno 拡張のトップレベル任意フィールド):
    /// - 文字列: "off" | "brief" | "full"
    /// - Anthropic マップ (Hermes 互換): {"type":"enabled", ...} = brief 扱い /
    ///   {"type":"disabled"} = off 扱い (budget_tokens 等は無視・決定的)。
    ///   未知の type は 400。
    /// 未指定のリクエストは brief が適用される (prompt に短答誘導文が注入される
    /// 意図的変更)。未知フィールドを拒否しない現行パーサの挙動は維持。
    pub thinking: Option<serde_json::Value>,
}

/// tool use: ツール定義 1 件 (Anthropic 形式。type フィールドは受け流す)
#[derive(Debug, Clone, Deserialize)]
pub struct ToolDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema。必須 (欠落は 400)
    pub input_schema: serde_json::Value,
}

/// tool_choice の正規化結果。Named は指定ツールの存在検証と拘束デコード (第21)
/// の許容名に使い、プロンプト/抽出の挙動は Any (実質 auto) と同一。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Any,
    Named(String),
}

/// tool_choice を正規化する (未指定 = Auto。不正値は 400)
pub fn parse_tool_choice(v: &Option<serde_json::Value>) -> Result<ToolChoice, ApiError> {
    let choice = match v {
        None => ToolChoice::Auto,
        Some(serde_json::Value::String(s)) => parse_tool_choice_str(s)?,
        Some(serde_json::Value::Object(m)) => {
            let ty = m.get("type").and_then(|t| t.as_str()).ok_or_else(|| {
                ApiError::invalid_request("tool_choice: object form requires a string \"type\"")
            })?;
            match ty {
                "auto" | "none" | "any" => parse_tool_choice_str(ty)?,
                "tool" => {
                    let name = m.get("name").and_then(|n| n.as_str()).ok_or_else(|| {
                        ApiError::invalid_request(
                            "tool_choice: {\"type\":\"tool\"} requires a string \"name\"",
                        )
                    })?;
                    if name.is_empty() {
                        return Err(ApiError::invalid_request(
                            "tool_choice: \"name\" must be non-empty",
                        ));
                    }
                    ToolChoice::Named(name.to_string())
                }
                other => {
                    return Err(ApiError::invalid_request(format!(
                        "tool_choice.type: unsupported value {:?} (auto/none/any/tool)",
                        other
                    )))
                }
            }
        }
        Some(_) => {
            return Err(ApiError::invalid_request(
                "tool_choice must be a string (auto/none/any) or an object",
            ))
        }
    };
    Ok(choice)
}

fn parse_tool_choice_str(s: &str) -> Result<ToolChoice, ApiError> {
    match s {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "any" => Ok(ToolChoice::Any),
        other => Err(ApiError::invalid_request(format!(
            "tool_choice: unsupported value {:?} (auto/none/any)",
            other
        ))),
    }
}

// ---------- s47: thinking 制御 ----------

/// s47: thinking 制御のモード (`thinking` トップレベル任意フィールド)。
///
/// - `Off`: chat template に `enable_thinking=false` を渡す (実モデルでは空の
///   think block が generation prompt に入り、思考せず即答する)
/// - `Brief` (既定): s53c で **Off と同じく思考を無効化**し、system 冒頭に
///   「回答は簡潔に」という短答誘導文を注入する (実測: 思考 ON + 誘導文では
///   9B が "Thinking Process:" のメタ分析を延々と書き回答に到達しなかったため)
/// - `Full`: 従来経路のまま (注入なし・`enable_thinking` 変数も渡さない =
///   prompt はビット同一)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    Off,
    Brief,
    Full,
}

/// thinking=brief のとき system 冒頭へ注入する誘導文 (決定的・固定文字列。
/// 同一入力→同一出力の契約のため、内容はコードに固定する)。
pub const THINKING_BRIEF_INSTRUCTION: &str = "回答は簡潔に、数文でまとめてください。";

/// thinking をパースする (未指定 = Brief。不正値は 400 invalid_request_error —
/// tool_choice のバリデーション慣習と同一)。
///
/// s52: 2 形式を受理する。
/// - 文字列 (s47 からの現行): "off" | "brief" | "full"
/// - Anthropic マップ (Hermes 互換): `{"type":"enabled", ...}` = **off 扱い**
///   (s53c: 実行時は思考無効 + 簡潔指示。`budget_tokens` 等は無視・決定的) /
///   `{"type":"disabled"}` = off 扱い / 未知の type (欠落・非文字列含む) は 400。
///   文字列/オブジェクト以外も 400。
pub fn parse_thinking(v: &Option<serde_json::Value>) -> Result<Thinking, ApiError> {
    match v {
        None => Ok(Thinking::Brief),
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "off" => Ok(Thinking::Off),
            "brief" => Ok(Thinking::Brief),
            "full" => Ok(Thinking::Full),
            other => Err(ApiError::invalid_request(format!(
                "thinking: unsupported value {:?} (off/brief/full)",
                other
            ))),
        },
        Some(serde_json::Value::Object(m)) => match m.get("type").and_then(|t| t.as_str()) {
            Some("enabled") => Ok(Thinking::Off),
            Some("disabled") => Ok(Thinking::Off),
            other => Err(ApiError::invalid_request(format!(
                "thinking.type: unsupported value {:?} (enabled/disabled)",
                other
            ))),
        },
        Some(other) => Err(ApiError::invalid_request(format!(
            "thinking must be a string (off/brief/full) or an object with a string \"type\" (enabled/disabled), got {:?}",
            other
        ))),
    }
}

/// thinking=brief の実効 system を組む (誘導文を system 冒頭へ注入。
/// 既存 system は保持・未指定/空なら誘導文単独の system になる)。
pub fn brief_system(system: Option<String>) -> Option<String> {
    Some(match system {
        Some(s) if !s.is_empty() => format!("{THINKING_BRIEF_INSTRUCTION}\n\n{s}"),
        _ => THINKING_BRIEF_INSTRUCTION.to_string(),
    })
}

/// s67b: 実効 temperature に serve 側の上限を適用する。
/// `cap` が None なら要求値をそのまま返す (従来と完全に同一)。
/// `cap` が Some(c) のときは、未指定 (sampler 既定 1.0) も含めて c で頭打ちにする。
pub(crate) fn apply_temperature_cap(requested: Option<f32>, cap: Option<f32>) -> Option<f32> {
    match cap {
        None => requested,
        Some(c) => Some(requested.unwrap_or(1.0).min(c)),
    }
}

/// s56 (T2): 反復抑止の追加パラメータを検証して正規化する。
///
/// - `frequency_penalty` / `presence_penalty`: OpenAI 互換。有限値のみ受理
///   (NaN/inf は 400)。符号は OpenAI 同様に許す (−2.0..2.0 を推奨するが、
///   範囲外もサーバーは拒否せず素通しする = 既存 temperature と同じ作法)。
///   `0.0` は無効 (no-op) として `None` に正規化する。
/// - `no_repeat_ngram_size`: **1 以下は 400** — `n == 1` は「直前トークン自身を
///   禁止」という別の意味になり、無効 (None) と区別が付かないため明示的に弾く
///   (HF / llama.cpp はいずれも n>=2 を要求する)。上限 `MAX_NGRAM_SIZE` 超も 400
///   (指定ミスを黙って飲まない)。
/// - 未指定はすべて `None` = 適用なし = 既定経路とビット一致。
pub fn parse_anti_repetition(req: &MessagesRequest) -> Result<GenAntiRepetition, ApiError> {
    let fin = |v: Option<f32>, name: &str| -> Result<Option<f32>, ApiError> {
        match v {
            None => Ok(None),
            Some(x) if !x.is_finite() => Err(ApiError::invalid_request(format!(
                "{name} must be a finite number (got {x})"
            ))),
            Some(x) if x == 0.0 => Ok(None),
            Some(x) => Ok(Some(x)),
        }
    };
    if let Some(p) = req.repetition_penalty {
        if !p.is_finite() {
            return Err(ApiError::invalid_request(format!(
                "repetition_penalty must be a finite number (got {p})"
            )));
        }
    }
    let n = match req.no_repeat_ngram_size {
        None => None,
        // 0 は「n-gram が作れない」ので無効扱い (明示的な 0 は許可しない: 1 と同様 400)
        Some(n) if n < 2 => {
            return Err(ApiError::invalid_request(format!(
                "no_repeat_ngram_size must be >= 2 (got {n}); omit the field to disable"
            )))
        }
        Some(n) if n > MAX_NGRAM_SIZE => {
            return Err(ApiError::invalid_request(format!(
                "no_repeat_ngram_size must be <= {MAX_NGRAM_SIZE} (got {n})"
            )))
        }
        Some(n) => Some(n),
    };
    Ok(GenAntiRepetition {
        frequency: fin(req.frequency_penalty, "frequency_penalty")?,
        presence: fin(req.presence_penalty, "presence_penalty")?,
        no_repeat_ngram_size: n,
    })
}

/// no_repeat_ngram_size の上限 (これより長い n は文脈長でも no-op になり、
/// 悪意/事故で巨大な語彙外サイズを渡されるのを防ぐ)。
pub const MAX_NGRAM_SIZE: usize = 1024;

/// `parse_anti_repetition` の出力 (GenParams へ渡す正規化済み値)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct GenAntiRepetition {
    pub frequency: Option<f32>,
    pub presence: Option<f32>,
    pub no_repeat_ngram_size: Option<usize>,
}

/// tools 定義の検証 (name 非空・input_schema は object)。
/// 正常ならテンプレートへ渡す Qwen (HF) 形式の配列を返す。
pub fn validate_tools(tools: &[ToolDef]) -> Result<serde_json::Value, ApiError> {
    let mut out = Vec::with_capacity(tools.len());
    for t in tools {
        if t.name.trim().is_empty() {
            return Err(ApiError::invalid_request(
                "tools[].name must be a non-empty string",
            ));
        }
        if !t.input_schema.is_object() {
            return Err(ApiError::invalid_request(format!(
                "tools[{}].input_schema must be an object",
                t.name
            )));
        }
        out.push(json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description.clone().unwrap_or_default(),
                "parameters": t.input_schema,
            }
        }));
    }
    Ok(json!(out))
}

#[derive(Deserialize)]
pub struct InMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl InMessage {
    /// content: 文字列 or content blocks のテキスト化 (tools 対応)。
    /// `tool_use_ids` はここまでの会話で現れた tool_use の id 集合 (重複/未知参照の検証用)。
    ///
    /// - assistant の tool_use block → 公式テンプレートと同一形式の
    ///   `<tool_call>` XML テキストへ変換 (モデル出力と同一形式で履歴に織り込む)
    /// - user の tool_result block → `<tool_response>` で包んで user テキストへ
    ///   (公式テンプレートの multi_step_tool 判定は「trim 後の user content が
    ///   `<tool_response>` で始まり `<\/tool_response>` で終わる」ことなので
    ///   tool_result のみのメッセージはその形になる)
    fn to_text(&self, tool_use_ids: &mut HashSet<String>) -> Result<String, ApiError> {
        match &self.content {
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Array(blocks) => {
                let mut out = String::new();
                for b in blocks {
                    match b.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            // 従来経路: text block は無条件で連結 (tools 無しリクエストで
                            // 結果が変わらないように区切り挿入はしない)
                            out.push_str(
                                b.get("text").and_then(|v| v.as_str()).unwrap_or_default(),
                            );
                        }
                        Some("tool_use") => {
                            if self.role != "assistant" {
                                return Err(ApiError::invalid_request(
                                    "tool_use block is only allowed in assistant messages",
                                ));
                            }
                            let id = b
                                .get("id")
                                .and_then(|v| v.as_str())
                                .filter(|s| !s.is_empty())
                                .ok_or_else(|| {
                                    ApiError::invalid_request(
                                        "tool_use.id must be a non-empty string",
                                    )
                                })?;
                            if !tool_use_ids.insert(id.to_string()) {
                                return Err(ApiError::invalid_request(format!(
                                    "tool_use.id: duplicate id {:?}",
                                    id
                                )));
                            }
                            let name = b
                                .get("name")
                                .and_then(|v| v.as_str())
                                .filter(|s| !s.is_empty())
                                .ok_or_else(|| {
                                    ApiError::invalid_request(
                                        "tool_use.name must be a non-empty string",
                                    )
                                })?;
                            let input = match b.get("input") {
                                None | Some(serde_json::Value::Null) => json!({}),
                                Some(v) if v.is_object() => v.clone(),
                                Some(_) => {
                                    return Err(ApiError::invalid_request(
                                        "tool_use.input must be an object",
                                    ))
                                }
                            };
                            if !out.is_empty() {
                                out.push_str("\n\n");
                            }
                            out.push_str(&tool_use_to_xml(name, &input));
                        }
                        Some("tool_result") => {
                            if self.role != "user" {
                                return Err(ApiError::invalid_request(
                                    "tool_result block is only allowed in user messages",
                                ));
                            }
                            let tuid = b
                                .get("tool_use_id")
                                .and_then(|v| v.as_str())
                                .filter(|s| !s.is_empty())
                                .ok_or_else(|| {
                                    ApiError::invalid_request(
                                        "tool_result.tool_use_id must be a non-empty string",
                                    )
                                })?;
                            if !tool_use_ids.contains(tuid) {
                                return Err(ApiError::invalid_request(format!(
                                    "tool_result.tool_use_id {:?} does not match any tool_use block in a previous assistant message",
                                    tuid
                                )));
                            }
                            let body = match b.get("content") {
                                None | Some(serde_json::Value::Null) => String::new(),
                                Some(serde_json::Value::String(s)) => s.clone(),
                                Some(serde_json::Value::Array(items)) => {
                                    let mut s = String::new();
                                    for it in items {
                                        if it.get("type").and_then(|t| t.as_str()) == Some("text")
                                        {
                                            s.push_str(
                                                it.get("text").and_then(|t| t.as_str())
                                                    .unwrap_or_default(),
                                            );
                                        } else {
                                            return Err(ApiError::invalid_request(format!(
                                                "unsupported tool_result content block type: {:?} (text blocks only)",
                                                it.get("type")
                                            )));
                                        }
                                    }
                                    s
                                }
                                Some(_) => {
                                    return Err(ApiError::invalid_request(
                                        "tool_result.content must be a string or an array of text blocks",
                                    ))
                                }
                            };
                            if !out.is_empty() {
                                out.push('\n');
                            }
                            out.push_str(&format!(
                                "<tool_response>\n{}\n</tool_response>",
                                body
                            ));
                        }
                        other => {
                            return Err(ApiError::invalid_request(format!(
                                "unsupported content block type: {:?} (text/tool_use/tool_result blocks only)",
                                other
                            )))
                        }
                    }
                }
                Ok(out)
            }
            _ => Err(ApiError::invalid_request(
                "messages[].content must be a string or an array of content blocks",
            )),
        }
    }
}

/// tool_use block を公式テンプレート (Qwen3.5) がモデルに出力させる
/// `<tool_call>` XML 形式へ変換する (履歴織り込み用。モデル自身の出力と同一形式)。
/// 非文字列値は tojson 相当の JSON 直列化、文字列値は raw のまま (公式テンプレート準拠)。
///
/// P797: OpenAI 経路 (openai.rs) の assistant `tool_calls` 履歴もこの形式へ写す
/// (モデルが見る履歴を `/v1/messages` と同一にするため crate 内に公開)。
pub(crate) fn tool_use_to_xml(name: &str, input: &serde_json::Value) -> String {
    let mut s = format!("<tool_call>\n<function={name}>\n");
    if let Some(obj) = input.as_object() {
        for (k, v) in obj {
            s.push_str(&format!("<parameter={k}>\n"));
            match v {
                serde_json::Value::String(raw) => s.push_str(raw),
                other => s.push_str(&other.to_string()),
            }
            s.push_str("\n</parameter>\n");
        }
    }
    s.push_str("</function>\n</tool_call>");
    s
}

/// system: 文字列 or `[{type:"text", text: "..."}]`
///
/// task 991: ブロック配列の検証を厳格化する (以前は text 以外のブロックを
/// 黙って読み飛ばし、`text` の欠落や空文字も空文字として通していた)。
/// - ブロックはオブジェクトであること (`system[i]: each block must be an object`)
/// - `type` は **"text" のみ**。`image` / `document` などのマルチモーダル
///   ブロックはテキスト専用モデルでは解釈できないため 400 で明示的に拒否する
///   (黙って落とすとプロンプトから静かに内容が消える)。
/// - `text` は文字列で、かつ非空白であること (空 text ブロックは 400)。
/// - 複数ブロックは出現順に連結する (区切り文字は挟まない。単一ブロックの
///   既存経路と同一のプロンプトになり、`cache_control` は従来どおり無視する)。
/// - 空配列は「system 無し」ではなく空文字列の system として扱う (既存挙動)。
///
/// エラーはすべて `400 invalid_request_error` の Anthropic 封筒になる
/// (`ApiError::invalid_request` は `error.type = "invalid_request_error"`)。
fn system_to_text(v: &Option<serde_json::Value>) -> Result<Option<String>, ApiError> {
    match v {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(serde_json::Value::Array(blocks)) => {
            let mut out = String::new();
            for (i, b) in blocks.iter().enumerate() {
                if !b.is_object() {
                    return Err(ApiError::invalid_request(format!(
                        "system[{i}]: each block must be an object"
                    )));
                }
                let ty = b.get("type").and_then(|t| t.as_str()).ok_or_else(|| {
                    ApiError::invalid_request(format!(
                        "system[{i}].type: must be a string (text blocks only)"
                    ))
                })?;
                if ty != "text" {
                    return Err(ApiError::invalid_request(format!(
                        "system[{i}].type: unsupported value {ty:?} (text blocks only)"
                    )));
                }
                let text = b.get("text").and_then(|t| t.as_str()).ok_or_else(|| {
                    ApiError::invalid_request(format!(
                        "system[{i}].text: must be a string"
                    ))
                })?;
                if text.trim().is_empty() {
                    return Err(ApiError::invalid_request(format!(
                        "system[{i}].text: text blocks must be non-empty"
                    )));
                }
                out.push_str(text);
            }
            Ok(Some(out))
        }
        Some(_) => Err(ApiError::invalid_request(
            "system must be a string or an array of text blocks",
        )),
    }
}

// ---------- tool use: モデル出力からの tool_call 抽出 ----------

/// 抽出された 1 件の tool call (input は Anthropic の `input` オブジェクト)
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedToolCall {
    pub name: String,
    pub input: serde_json::Value,
}

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";

/// モデル出力テキストから完全な `&lt;tool_call&gt;` block を抽出する。
///
/// 戻り値は (tool_call block を完全に除去+trim した残りテキスト, 抽出された呼び出し列)。
/// - block 単位で独立にパースし、パースに成功した block だけ除去する
///   (失敗 block はテキストとして残す。モデルが形式例を出した場合への防御)
/// - 未完結 (閉じ tag 無し) は block とみなさずテキストとして残す
/// - 対応形式: 公式テンプレートの XML 形式
///   `&lt;tool_call&gt;\n&lt;function=name&gt;\n&lt;parameter=k&gt;\nvalue\n&lt;/parameter&gt;\n&lt;/function&gt;\n&lt;/tool_call&gt;`
///   と JSON 形式 `{"name":..., "arguments":{...}}` ("parameters"/"input" も受ける)
pub fn extract_tool_calls(text: &str) -> (String, Vec<ExtractedToolCall>) {
    let mut calls = Vec::new();
    let mut keep = String::with_capacity(text.len());
    let mut pos = 0usize;
    while let Some(s) = text[pos..].find(TOOL_CALL_OPEN) {
        let s = pos + s;
        keep.push_str(&text[pos..s]);
        let inner_start = s + TOOL_CALL_OPEN.len();
        let Some(e_rel) = text[inner_start..].find(TOOL_CALL_CLOSE) else {
            // 未完結: 残り全体をテキストとして残す
            keep.push_str(&text[s..]);
            pos = text.len();
            break;
        };
        let inner = &text[inner_start..inner_start + e_rel];
        pos = inner_start + e_rel + TOOL_CALL_CLOSE.len();
        match parse_tool_call_inner(inner) {
            Some(c) => calls.push(c),
            None => keep.push_str(&text[s..pos]),
        }
    }
    keep.push_str(&text[pos..]);
    (keep.trim().to_string(), calls)
}

/// tool_call block の中身 (open/close 間) をパースする。失敗は None。
fn parse_tool_call_inner(inner: &str) -> Option<ExtractedToolCall> {
    let inner = inner.trim();
    // 1) XML 形式: <function=NAME> ... </function>
    if let Some(body) = inner.strip_prefix("<function=").and_then(|r| r.strip_suffix("</function>"))
    {
        let gt = body.find('>')?;
        let name = body[..gt].trim();
        if name.is_empty() {
            return None;
        }
        let mut input = serde_json::Map::new();
        let mut rest = &body[gt + 1..];
        while let Some(p) = rest.find("<parameter=") {
            let after = &rest[p + "<parameter=".len()..];
            let gt = after.find('>')?;
            let key = after[..gt].trim();
            if key.is_empty() {
                return None;
            }
            let vstart = gt + 1;
            let vend_rel = after[vstart..].find("</parameter>")?;
            let raw = &after[vstart..vstart + vend_rel];
            rest = &after[vstart + vend_rel + "</parameter>".len()..];
            input.insert(key.to_string(), parse_param_value(raw));
        }
        return Some(ExtractedToolCall {
            name: name.to_string(),
            input: serde_json::Value::Object(input),
        });
    }
    // 2) JSON 形式: {"name": ..., "arguments"|"parameters"|"input": {...}}
    let v: serde_json::Value = serde_json::from_str(inner).ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?;
    if name.is_empty() {
        return None;
    }
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .or_else(|| obj.get("input"));
    let input = match args {
        None => serde_json::Map::new(),
        Some(a) if a.is_object() => a.as_object().unwrap().clone(),
        Some(_) => return None,
    };
    Some(ExtractedToolCall {
        name: name.to_string(),
        input: serde_json::Value::Object(input),
    })
}

/// `<parameter=k>` の値を復元する。JSON 値 (数値/真偽/null/object/array/引用文字列)
/// としてパースできればその値、できなければ raw 文字列。
/// 公式テンプレートは非文字列値を tojson し文字列値を raw 出力するため、
/// テンプレートに値を渡した際の `tojson | safe` と一致する規則。
fn parse_param_value(raw: &str) -> serde_json::Value {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(raw.trim()) {
        return v;
    }
    // テンプレートが付与する前後の改行 1 つずつを落として生文字列として返す
    let s = raw.strip_prefix('\n').unwrap_or(raw);
    let s = s.strip_suffix('\n').unwrap_or(s);
    serde_json::Value::String(s.to_string())
}

/// tool_use block の id (`toolu_` + 24hex)。msg id と同一の乱数経路。
pub(crate) fn new_tool_id() -> String {
    let mut rng = rand::thread_rng();
    let id: String = (0..24).map(|_| format!("{:x}", rng.gen::<u8>() % 16)).collect();
    format!("toolu_{id}")
}

// ---------- ルーター ----------

pub fn router(state: Arc<AppState>) -> Router {
    // task 916: 既定は「ループバック (localhost / 127.0.0.1 / [::1]) の任意ポート」
    // だけを許可する。既存のローカルブラウザからのアクセスはそのまま通り、
    // 任意の外部オリジンは開かない。
    router_with_cors(state, CorsConfig::default())
}

/// task 916: CORS 設定を明示してルーターを組む。
///
/// task 931: レート制限は env (`DENNO_RATE_LIMIT_RPM` / `DENNO_RATE_LIMIT_BURST`)
/// から解決する。明示的に差し替えたいときは [`router_with_rate_limit`] を使う。
pub fn router_with_cors(state: Arc<AppState>, cors: CorsConfig) -> Router {
    router_with_rate_limit(state, cors, ratelimit::RateLimiter::from_env())
}

/// task 931: レート制限器を明示してルーターを組む。
///
/// 層の順序は外側から `log_requests` → レート制限 → CORS → ルート。429 も
/// CORS ヘッダと `x-request-id` を伴って返る (プリフライトは CORS 層が先に
/// 204 を返すのでトークンを消費しない)。
pub fn router_with_rate_limit(
    state: Arc<AppState>,
    cors: CorsConfig,
    limiter: ratelimit::RateLimiter,
) -> Router {
    // task 946: ボディ上限は env (`DENNO_MAX_BODY_MB`) → 既定 16 MiB。
    // 明示指定は [`router_with_body_limit`] を使う。
    router_with_body_limit(state, cors, limiter, resolve_max_body_bytes(None))
}

/// task 946: ボディ上限 (byte) を明示してルーターを組む。
///
/// 上限を超えるボディは HTTP 413 を `application/json` の
/// `{"type":"error","error":{...}}` 封筒で返す。
pub fn router_with_body_limit(
    state: Arc<AppState>,
    cors: CorsConfig,
    limiter: ratelimit::RateLimiter,
    max_body_bytes: usize,
) -> Router {
    let cors = Arc::new(cors);
    let limiter = Arc::new(limiter);
    Router::new()
        .route("/health", get(health))
        // P877: Prometheus のスクレイプ先 (認証は /health と同じく不要)。
        .route("/metrics", get(metrics_endpoint))
        // P817: readiness。モデルロード済み + ワーカー生存 + drain していない
        // ときだけ 200 (それ以外は 503)。liveness の /health とは別物。
        .route("/ready", get(ready))
        .route("/v1/models", get(models))
        // task 961: `GET /v1/models/:id`。仕様 (`docs/openapi.yaml` の
        // `GET /v1/models/{id}`) と 1:1 なので、他のルートと同じくここで登録する。
        .route("/v1/models/:id", get(openai::model_by_id))
        // task 926: prefix cache の管理 (`/v1/cache/stats` / `DELETE /v1/cache`)。
        // パスは定数なので、task 893 の `.route("` 行の契約テストには現れない。
        .merge(cache_admin_router())
        .route("/v1/messages", post(messages))
        // P805: Anthropic 互換のトークン数見積り (prefill しない)
        .route("/v1/messages/count_tokens", post(count_tokens))
        // P793: OpenAI 互換 (同じ Scheduler / chat template / KV prefix cache を通す)
        .route("/v1/chat/completions", post(openai::chat_completions))
        // P837: OpenAI 旧 Completions (chat template を通さない raw prompt)
        .route("/v1/completions", post(openai::completions))
        // P841: Ollama 互換 (NDJSON。同じ Scheduler / chat template / KV を通す)
        .route("/api/tags", get(ollama::tags))
        .route("/api/version", get(ollama::version))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        // task 996: `/api/show` (POST) と `/api/ps` (GET) は ollama.rs の
        // サブルータ側で登録する。`docs/openapi.yaml` は今回のスコープ外で、
        // `tests/openapi_contract.rs` は lib.rs の `.route("` 行だけを拾うため、
        // ここに直接書くと契約テストが赤になる。
        .merge(ollama::admin_router())
        // P13: 推論テレメトリの WebSocket ストリーム (denno-trace)。
        // Anthropic 互換 API とは別系統で、互換性には影響しない。
        .route("/v1/telemetry/ws", get(telemetry_ws))
        // task 916: 未知ルート (404) と未対応メソッド (405) も空ボディではなく
        // OpenAI 形式の JSON 封筒で返す。`.layer()` より前に置くことで、
        // 下の log_requests / CORS の層がこの 2 つにも掛かる。
        .fallback(route_not_found)
        .method_not_allowed_fallback(method_not_allowed)
        // task 946: ボディ上限。`DefaultBodyLimit` は抽出器 (Bytes / Json) に
        // 効き、413 はすぐ外側の `body_limit_middleware` が JSON 封筒へ包む。
        .layer(axum::extract::DefaultBodyLimit::max(max_body_bytes))
        .layer(axum::middleware::from_fn(move |req, next| async move {
            body_limit_middleware(req, next, max_body_bytes).await
        }))
        // task 931: レート制限。CORS 層より内側 (コード上は先) に置くことで、
        // 429 にも Allow-Origin などの CORS ヘッダが付く。
        .layer(axum::middleware::from_fn(move |req, next| {
            let limiter = limiter.clone();
            async move { rate_limit_middleware(limiter, req, next).await }
        }))
        .layer(axum::middleware::from_fn(move |req, next| {
            let cors = cors.clone();
            async move { cors_middleware(cors, req, next).await }
        }))
        .layer(axum::middleware::from_fn(log_requests))
        .with_state(state)
}

// ---------- task 916: CORS ----------

/// `DENNO_CORS_ORIGIN` の env 名。
pub const CORS_ORIGIN_ENV: &str = "DENNO_CORS_ORIGIN";

/// プリフライトで許可するメソッド。
const CORS_ALLOW_METHODS: &str = "GET, POST, OPTIONS";

/// `Access-Control-Request-Headers` が無いときのプリフライト応答に入れる
/// 許可ヘッダ (denno の API が実際に読むもの)。
const CORS_ALLOW_HEADERS: &str =
    "content-type, x-api-key, authorization, anthropic-version, x-request-id";

/// プリフライト結果のキャッシュ秒数。
const CORS_MAX_AGE_SECS: &str = "86400";

/// 非ストリーミング応答で JS から読ませるヘッダ (task 911 の計測ヘッダ)。
const CORS_EXPOSE_HEADERS: &str = "x-request-id, x-denno-ttft-ms, x-denno-decode-tps";

/// task 916: CORS の解決結果。
///
/// 既定 ([`Default`]) は「ループバックの任意ポートだけ許可」。サーバは
/// 127.0.0.1 で待ち受けるのが既定なので、ローカルブラウザ (denno-view など)
/// からのアクセスはそのままで、外部オリジンへはヘッダを出さない。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorsConfig {
    /// `*`: 任意オリジンを許可する (`Access-Control-Allow-Origin: *`)。
    pub allow_any: bool,
    /// 明示許可するオリジン (小文字・末尾スラッシュ無しに正規化済み)。
    pub origins: Vec<String>,
    /// `http(s)://localhost` / `127.0.0.1` / `[::1]` を任意ポートで許可する。
    pub allow_loopback: bool,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self { allow_any: false, origins: Vec::new(), allow_loopback: true }
    }
}

impl CorsConfig {
    /// この設定で `origin` を許可するか。
    pub fn allows(&self, origin: &str) -> bool {
        if self.allow_any {
            return true;
        }
        let o = normalize_origin(origin);
        if self.origins.iter().any(|x| x == &o) {
            return true;
        }
        self.allow_loopback && is_loopback_origin(&o)
    }
}

/// オリジン文字列の比較用正規化 (小文字化 + 末尾スラッシュ除去)。
/// `<scheme>://<host>` の scheme / host は ASCII 大文字小文字を区別しない。
fn normalize_origin(origin: &str) -> String {
    origin.trim().trim_end_matches('/').to_ascii_lowercase()
}

/// `scheme://host[:port]` から見てループバックのオリジンか。
/// `http`/`https` 以外の scheme と、ホストがループバック名でないものは false。
fn is_loopback_origin(normalized: &str) -> bool {
    let Some(rest) = normalized
        .strip_prefix("http://")
        .or_else(|| normalized.strip_prefix("https://"))
    else {
        return false;
    };
    if let Some(port) = rest.strip_prefix("[::1]") {
        return port.is_empty() || is_port_suffix(port);
    }
    match rest.rsplit_once(':') {
        Some((host, port)) => matches!(host, "localhost" | "127.0.0.1") && is_valid_port(port),
        None => matches!(rest, "localhost" | "127.0.0.1"),
    }
}

fn is_port_suffix(s: &str) -> bool {
    s.strip_prefix(':').is_some_and(is_valid_port)
}

fn is_valid_port(p: &str) -> bool {
    !p.is_empty() && p.len() <= 5 && p.chars().all(|c| c.is_ascii_digit())
}

/// task 916: CORS 設定を解決する。優先順は CLI (`--cors-origin`) > env。
///
/// - 未指定 / 空: 既定 (ループバックのみ許可)
/// - `*`: 任意オリジン
/// - カンマ区切り: その一覧だけを許可 (ループバック既定は無効化される。
///   明示した一覧にループバックを入れれば従来どおりローカルからも使える)
pub fn resolve_cors_origin(cli: Option<&str>) -> CorsConfig {
    let raw = match cli {
        Some(s) => Some(s.to_string()),
        None => std::env::var(CORS_ORIGIN_ENV).ok(),
    };
    parse_cors_origin(raw.as_deref())
}

/// [`resolve_cors_origin`] の純関数部分 (テスト用)。
pub fn parse_cors_origin(raw: Option<&str>) -> CorsConfig {
    let Some(raw) = raw else { return CorsConfig::default() };
    let raw = raw.trim();
    if raw.is_empty() {
        return CorsConfig::default();
    }
    let mut allow_any = false;
    let mut origins = Vec::new();
    for part in raw.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if p == "*" {
            allow_any = true;
        } else {
            origins.push(normalize_origin(p));
        }
    }
    if !allow_any && origins.is_empty() {
        return CorsConfig::default();
    }
    CorsConfig { allow_any, origins, allow_loopback: false }
}

/// task 916: CORS を適用する層。プリフライト (`OPTIONS` +
/// `Access-Control-Request-Method`) は here で完結させ、通常のリクエストは
/// `Access-Control-Allow-Origin` を足してから内側へ渡す。
async fn cors_middleware(
    cfg: Arc<CorsConfig>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    use axum::http::header;
    use axum::http::Method;

    let origin = req
        .headers()
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let allowed_origin = match &origin {
        Some(o) if cfg.allows(o) => Some(o.clone()),
        _ => None,
    };

    let is_preflight = req.method() == Method::OPTIONS
        && req.headers().contains_key(header::ACCESS_CONTROL_REQUEST_METHOD);
    if is_preflight {
        let mut resp = StatusCode::NO_CONTENT.into_response();
        if allowed_origin.is_some() || cfg.allow_any {
            let h = resp.headers_mut();
            let value = if cfg.allow_any {
                "*".to_string()
            } else {
                allowed_origin.clone().unwrap_or_default()
            };
            set_cors_header(h, header::ACCESS_CONTROL_ALLOW_ORIGIN, &value);
            set_cors_header(h, header::ACCESS_CONTROL_ALLOW_METHODS, CORS_ALLOW_METHODS);
            let requested = req
                .headers()
                .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
                .and_then(|v| v.to_str().ok())
                .unwrap_or(CORS_ALLOW_HEADERS);
            set_cors_header(h, header::ACCESS_CONTROL_ALLOW_HEADERS, requested);
            set_cors_header(h, header::ACCESS_CONTROL_MAX_AGE, CORS_MAX_AGE_SECS);
            if !cfg.allow_any {
                set_cors_header(h, header::VARY, "Origin");
            }
        }
        return resp;
    }

    let mut resp = next.run(req).await;
    if allowed_origin.is_some() || cfg.allow_any {
        let value = if cfg.allow_any {
            "*".to_string()
        } else {
            allowed_origin.unwrap_or_default()
        };
        let h = resp.headers_mut();
        set_cors_header(h, header::ACCESS_CONTROL_ALLOW_ORIGIN, &value);
        set_cors_header(h, header::ACCESS_CONTROL_EXPOSE_HEADERS, CORS_EXPOSE_HEADERS);
        if !cfg.allow_any {
            set_cors_header(h, header::VARY, "Origin");
        }
    }
    resp
}

fn set_cors_header(
    headers: &mut axum::http::HeaderMap,
    name: axum::http::header::HeaderName,
    value: &str,
) {
    if let Ok(v) = axum::http::HeaderValue::from_str(value) {
        headers.insert(name, v);
    }
}

/// task 916: 未知ルートの 404 (OpenAI 形式の JSON 封筒)。
async fn route_not_found(req: axum::extract::Request) -> Response {
    let path = req.uri().path().to_string();
    openai_error_response(
        StatusCode::NOT_FOUND,
        "unknown_route",
        format!("Unknown route: {} {}", req.method(), path),
    )
}

/// task 916: 既知ルートへの未対応メソッド (405)。
async fn method_not_allowed(req: axum::extract::Request) -> Response {
    let path = req.uri().path().to_string();
    openai_error_response(
        StatusCode::METHOD_NOT_ALLOWED,
        "method_not_allowed",
        format!("Method {} not allowed for {}", req.method(), path),
    )
}

/// OpenAI 互換のエラー封筒
/// `{"error":{"message","type","param","code"}}` を組み立てる。
fn openai_error_response(status: StatusCode, code: &str, message: String) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": null,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}


/// task 931: 推論エンドポイントのレート制限層。
///
/// 無効時 (既定) は何もせず素通しする。有効時も /health / /ready / /metrics と
/// 推論以外のパス、OPTIONS (プリフライト) は消費しない。
async fn rate_limit_middleware(
    limiter: Arc<ratelimit::RateLimiter>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if limiter.is_enabled() && req.method() != axum::http::Method::OPTIONS {
        if let ratelimit::Decision::Deny { retry_after_secs } = limiter.check(req.uri().path()) {
            return ratelimit::too_many_requests_response(retry_after_secs);
        }
    }
    next.run(req).await
}

async fn log_requests(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let started = std::time::Instant::now();
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    // task 911: クライアントの `x-request-id` を無害化して引き継ぐ。
    // 無効 (空 / 制御文字だけ / 非 ASCII だけ) なら新しい ID を振る。
    let request_id = req
        .headers()
        .get(REQUEST_ID_HEADER)
        .map(|v| v.as_bytes())
        .and_then(sanitize_request_id_bytes)
        .unwrap_or_else(generate_request_id);
    // P877: ラベルは既知ルート名へ正規化する (未知のパスは `other`)。
    let endpoint = metrics::normalize_endpoint(&path);
    if logging::access_log_json() {
        // JSON 形式は 1 リクエスト 1 行 (応答後にまとめて出す)。
    } else {
        tracing::info!("{} {} rid={}", method, path, request_id);
    }
    let mut resp = next.run(req).await;
    let status = resp.status();
    let elapsed = started.elapsed();
    // P877: HTTP カウンタ (endpoint, status)。
    metrics::record_http(endpoint, status.as_u16());
    // task 911: 全応答 (SSE / NDJSON を含む) にリクエスト ID を付ける。
    set_header_str(&mut resp, "x-request-id", &request_id);
    // task 911: 完了した非ストリーミング JSON 応答にだけ計測ヘッダを足す。
    // (SSE / NDJSON はボディが終わらないので対象外)
    resp = add_timing_headers(resp, elapsed).await;
    if logging::access_log_json() {
        let line = logging::access_log_line(
            &method,
            &path,
            status.as_u16(),
            elapsed.as_secs_f64() * 1000.0,
            &request_id,
        );
        tracing::info!(target: "denno_access", "{line}");
    } else {
        tracing::info!("-> {} rid={}", status, request_id);
    }
    resp
}

// ---------- task 911: リクエスト ID と計測ヘッダ ----------

/// `x-request-id` として受理・返却する最大長 (byte。ASCII 前提)。
pub const MAX_REQUEST_ID_LEN: usize = 128;

/// [`add_timing_headers`] が応答ボディを読み込む上限 (byte)。
/// 生成応答は数 KB なので実質的に当たらない。
const TIMING_BODY_LIMIT: usize = 64 * 1024 * 1024;

/// クライアントのリクエスト ID を受け取るヘッダ名。
const REQUEST_ID_HEADER: &str = "x-request-id";

/// 生成 ID の連番 (プロセス内で単調増加)。
static NEXT_REQUEST_SEQ: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(1);

/// 生成 ID の上位ビット (プロセス起動ごとに変わる 48 bit)。
static REQUEST_ID_BASE: OnceLock<u64> = OnceLock::new();

/// 送信元の `x-request-id` を無害化する。
///
/// ASCII の印字可能文字 (0x21..=0x7E) だけを残し、制御文字・空白・非 ASCII は
/// 捨てる (`\"` と `\\` も落とす)。[`MAX_REQUEST_ID_LEN`] byte で切る。
/// 有効な文字が残らない場合は `None` (呼び側が新しい ID を振る)。
pub fn sanitize_request_id(raw: &str) -> Option<String> {
    sanitize_request_id_bytes(raw.as_bytes())
}

/// [`sanitize_request_id`] の byte 版。ヘッダは非表示 byte (obs-text) を
/// 含み得て `str` にならないので、ミドルウェアはこちらを使う。
pub fn sanitize_request_id_bytes(raw: &[u8]) -> Option<String> {
    let mut out = String::with_capacity(raw.len().min(MAX_REQUEST_ID_LEN));
    for &b in raw {
        if out.len() >= MAX_REQUEST_ID_LEN {
            break;
        }
        if (0x21..=0x7e).contains(&b) && b != b'"' && b != b'\\' {
            out.push(b as char);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// 新しいリクエスト ID を振る (`req-` + base 12 hex + 連番 8 hex)。
///
/// base はプロセス起動時に一度だけ作る (時刻 ^ 乱数の下位 48 bit)。連番は
/// プロセス内で単調増加するので、同じプロセスの ID は衝突せず辞書順も
/// 時刻順になる。
pub fn generate_request_id() -> String {
    let base = *REQUEST_ID_BASE.get_or_init(|| {
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        (ns ^ rand::random::<u64>()) & 0x0000_ffff_ffff_ffff
    });
    let seq = NEXT_REQUEST_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as u32;
    format!("req-{base:012x}{seq:08x}")
}

/// 応答ヘッダを 1 個上書きする (値がヘッダとして不正なら何もしない)。
fn set_header_str(resp: &mut Response, name: &'static str, value: &str) {
    if let Ok(v) = axum::http::HeaderValue::from_str(value) {
        resp.headers_mut().insert(axum::http::HeaderName::from_static(name), v);
    }
}

/// JSON 応答から生成トークン数を読む (`usage.output_tokens` (Anthropic) /
/// `usage.completion_tokens` (OpenAI) / `eval_count` (Ollama))。
fn body_output_tokens(v: &serde_json::Value) -> Option<u64> {
    v.get("usage")
        .and_then(|u| u.get("output_tokens").or_else(|| u.get("completion_tokens")))
        .and_then(|n| n.as_u64())
        .or_else(|| v.get("eval_count").and_then(|n| n.as_u64()))
}

/// task 911: 完了した非ストリーミング JSON 応答に計測ヘッダを足す。
///
/// - `x-denno-ttft-ms`: リクエスト受付から応答完成までの壁時計 (ms)。
///   非ストリーミング応答ではボディの最初の byte が生成完了後に出るため、
///   これは HTTP 上の TTFT (= 応答全体の待ち時間) になる。エンジンは
///   サーバ crate へ最初のトークンの時刻を公開していない
///   (`scheduler.rs` の `first_token_at` は非公開・別ファイル) ので、
///   prefill と decode の内訳は測れない。
/// - `x-denno-decode-tps`: `output_tokens / 壁時計` の end-to-end スループット
///   (tokens/s)。prefill を含むので真の decode 速度の下限。Ollama 経路の
///   `eval_duration` 近似と同じ方針。
///
/// JSON でも生成結果でない応答 (usage が無い `/health` 等) には付けない。
/// ボディは読み込んでそのまま返す (内容・順序は変えない)。
async fn add_timing_headers(resp: Response, elapsed: std::time::Duration) -> Response {
    let is_json = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|ct| ct.to_ascii_lowercase().starts_with("application/json"))
        .unwrap_or(false);
    if !is_json {
        return resp;
    }
    let (mut parts, body) = resp.into_parts();
    let bytes = match axum::body::to_bytes(body, TIMING_BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("response body could not be buffered for timing headers: {e}"),
            )
                .into_response();
        }
    };
    let out_tokens = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| body_output_tokens(&v));
    if let Some(out_tokens) = out_tokens {
        let secs = elapsed.as_secs_f64();
        let ttft_ms = secs * 1000.0;
        let tps = if secs > 0.0 { out_tokens as f64 / secs } else { 0.0 };
        let mut hdrs = axum::http::HeaderMap::new();
        if let Ok(v) = axum::http::HeaderValue::from_str(&format!("{ttft_ms:.1}")) {
            hdrs.insert(axum::http::HeaderName::from_static("x-denno-ttft-ms"), v);
        }
        if let Ok(v) = axum::http::HeaderValue::from_str(&format!("{tps:.2}")) {
            hdrs.insert(axum::http::HeaderName::from_static("x-denno-decode-tps"), v);
        }
        for (name, value) in hdrs.iter() {
            parts.headers.insert(name.clone(), value.clone());
        }
    }
    Response::from_parts(parts, axum::body::Body::from(bytes))
}

/// P877: Prometheus text exposition format 0.0.4。
///
/// 認証は要求しない (`/health` / `/ready` と同じ。スクレイパは API キーを持たない)。
/// ゲージは既存のスケジューラ状態を読むだけで、ここでは何も変えない。
async fn metrics_endpoint(State(state): State<Arc<AppState>>) -> Response {
    let body = metrics::render(
        &state.model_name,
        state.scheduler.in_flight(),
        state.scheduler.pending(),
        &state.scheduler.stats(),
    );
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// task 926: prefix cache の管理エンドポイント
//   `GET /v1/cache/stats` — 保持数 / 予算 / ヒット数 / ミス数 / 再利用トークン
//   `DELETE /v1/cache`     — 保持中の prefix エントリを捨てる (in-flight は不可侵)
// ---------------------------------------------------------------------------
//
// prefix cache 本体は engine のマルチエントリ LRU (`denno_engine::prefix_cache`)。
// 推論のホットパスに足すのは Relaxed の原子加算 1 回だけで、ロックは取らない。
// エンジンロックを取るのはこの管理エンドポイント自身だけ (スケジューラのワーカーは
// ステップ単位でロックを離すので、最悪 1 ステップ分だけ待つ)。
// KV / logits / サンプリングの数値経路には一切触れない。

/// `GET /v1/cache/stats` のパス。
///
/// `.route("...")` を直書きしないのは、task 893 の契約テストが lib.rs の
/// `.route("` 行を `docs/openapi.yaml` と突き合わせるため。仕様書はこのタスクの
/// scope 外なので、サブルーター登録にして route 行を増やさない。
const CACHE_STATS_PATH: &str = "/v1/cache/stats";
/// `DELETE /v1/cache` のパス。
const CACHE_FLUSH_PATH: &str = "/v1/cache";

/// prefix 再利用したリクエスト数 (累積)。
static PREFIX_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
/// 再利用なし (フル prefill) だったリクエスト数 (累積)。
static PREFIX_CACHE_MISSES: AtomicU64 = AtomicU64::new(0);
/// 再利用した先頭トークン数の合計 (累積)。
static PREFIX_CACHE_REUSED_TOKENS: AtomicU64 = AtomicU64::new(0);

/// 完了したリクエスト 1 本ぶんの prefix 再利用を数える。
///
/// `reused` は KV prefix cache から使い回した先頭トークン数
/// (`ChatOutput::cache_read_input_tokens`)。ホットパスでは Relaxed の
/// 原子加算 1〜2 回だけで、ロックも確保もしない。失敗 / 打ち切りで完走しなかった
/// リクエストは数えない (再利用の実績ではないため)。
pub fn record_prefix_cache_reuse(reused: usize) {
    if reused > 0 {
        PREFIX_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        PREFIX_CACHE_REUSED_TOKENS.fetch_add(reused as u64, Ordering::Relaxed);
    } else {
        PREFIX_CACHE_MISSES.fetch_add(1, Ordering::Relaxed);
    }
}

/// 累積カウンタ `(hits, misses, total_reused_tokens)` を読む (状態は変えない)。
pub fn prefix_cache_counters() -> (u64, u64, u64) {
    (
        PREFIX_CACHE_HITS.load(Ordering::Relaxed),
        PREFIX_CACHE_MISSES.load(Ordering::Relaxed),
        PREFIX_CACHE_REUSED_TOKENS.load(Ordering::Relaxed),
    )
}

/// `/v1/cache/stats` の本文。
///
/// - `slots_used`: 保持中の prefix エントリ数 (会話スナップショットの本数)
/// - `slots_capacity`: prefix cache の LRU 予算 (トークン)。エントリの KV 重みの
///   合計がこれを超えると古い順に捨てる — 「スロットの容量」は本数ではなく
///   このトークン予算で決まる
/// - `hits` / `misses`: 完了したリクエストのうち prefix を再利用した / しなかった本数
/// - `hit_rate`: `hits / (hits + misses)`。リクエストが 0 なら 0
/// - `total_reused_tokens`: 再利用した先頭トークン数の累計
fn cache_stats_json(entries: usize, budget_tokens: usize) -> serde_json::Value {
    let (hits, misses, reused) = prefix_cache_counters();
    let lookups = hits + misses;
    let hit_rate = if lookups == 0 {
        0.0
    } else {
        hits as f64 / lookups as f64
    };
    json!({
        "slots_used": entries,
        "slots_capacity": budget_tokens,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "total_reused_tokens": reused,
    })
}

/// task 926: `GET /v1/cache/stats`。
///
/// エンジンロックを取るのはここだけ。`spawn_blocking` で tokio のワーカーを
/// 塞がない (スケジューラが prefill 中の 1 ステップを握っている間は待つ)。
async fn cache_stats(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_auth(&state, &headers)?;
    let engine = state.engine.clone();
    let (entries, budget, kv_slots, kv_busy) = tokio::task::spawn_blocking(move || {
        let g = engine.lock().unwrap();
        (
            g.prefix_cache_entries(),
            g.prefix_cache_max_tokens(),
            g.slot_count(),
            g.slot_busy_count(),
        )
    })
    .await
    .map_err(|e| ApiError::api(format!("cache stats task failed: {e}")))?;
    let mut v = cache_stats_json(entries, budget);
    // KV スロット側の占有 (in-flight の本数) も併記する。prefix エントリとは別物。
    v["kv_slots"] = json!(kv_slots);
    v["kv_slots_busy"] = json!(kv_busy);
    Ok(Json(v))
}

/// task 926: `DELETE /v1/cache` — 保持中の prefix エントリを捨てる。
///
/// 触るのはエンジンの prefix cache ストア (会話スナップショット) だけ。走って
/// いるリクエストの KV は各スロットの `caches[slot]` にあり、ここでは触らないので
/// **in-flight の生成は止まらず、出力も変わらない**。次のリクエストは前置きを
/// 使い回さずフル prefill になる。
///
/// 空にするのは既存 API の組み合わせ: 予算 0 で全エントリを LRU eviction させ、
/// 直後に元の予算へ戻す (予算が 0 のときは何もしない)。
async fn cache_flush(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_auth(&state, &headers)?;
    let engine = state.engine.clone();
    let (before, after, budget) = tokio::task::spawn_blocking(move || {
        let mut g = engine.lock().unwrap();
        let before = g.prefix_cache_entries();
        let budget = g.prefix_cache_max_tokens();
        if before > 0 && budget > 0 {
            g.set_prefix_cache_max_tokens(0);
            g.set_prefix_cache_max_tokens(budget);
        }
        (before, g.prefix_cache_entries(), budget)
    })
    .await
    .map_err(|e| ApiError::api(format!("cache flush task failed: {e}")))?;
    Ok(Json(json!({
        "flushed": before.saturating_sub(after),
        "slots_used": after,
        "slots_capacity": budget,
    })))
}

/// task 926: prefix cache 管理ルート (パスは定数。上のコメント参照)。
fn cache_admin_router() -> Router<Arc<AppState>> {
    Router::new()
        .route(CACHE_STATS_PATH, get(cache_stats))
        .route(CACHE_FLUSH_PATH, delete(cache_flush))
}

pub async fn run(cfg: ServeConfig) -> Result<()> {
    let model_dir = cfg.model_dir.clone();
    let name = cfg.name.clone();
    let seed = cfg.default_seed;
    let quant = cfg.quant;
    tracing::info!("loading model from {:?} ...", model_dir);
    let mut engine =
        tokio::task::spawn_blocking(move || Engine::load(&model_dir, name, seed, quant))
            .await
            .map_err(|e| anyhow::anyhow!("join error: {e}"))??;
    let ctx = engine.context();
    // CLI 引数での prefix cache 上限指定 (env より優先)
    if let Some(n) = cfg.prefix_cache_tokens {
        engine.set_prefix_cache_max_tokens(n);
    }
    // s56b: serve 既定 repetition_penalty の妥当性検査 (0 以下/非有限は起動時に拒否)。
    if let Some(p) = cfg.repetition_penalty {
        if !p.is_finite() || p <= 0.0 {
            anyhow::bail!("--repetition-penalty must be a finite positive number (got {p})");
        }
        tracing::info!("serve default repetition_penalty = {p}");
    }
    // s67b: temperature 上限の妥当性検査 (負/非有限は起動時に拒否)。
    if let Some(c) = cfg.temperature_cap {
        if !c.is_finite() || c < 0.0 {
            anyhow::bail!("--temperature-cap must be a finite non-negative number (got {c})");
        }
        tracing::info!("serve temperature cap = {c}");
    }
    tracing::info!(
        "model loaded: {} (context = {}, engine = denno)",
        engine.name,
        ctx
    );

    let model_name = engine.name.clone();
    let model_id = cfg.model_id.clone();
    // P57: 不変情報をここで 1 回だけ読む (ハンドラはエンジンロックを取らない)
    let mut info = EngineInfo::from_engine(&engine);
    // P817: `/v1/models` の `model_dir` (運用で「どの重みか」を判別するため)。
    info.model_dir = cfg
        .model_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    // task 901: `/v1/models` の `created` に重みディレクトリの mtime を使う。
    openai::register_model_dir(&cfg.model_dir);
    let engine = Arc::new(Mutex::new(engine));
    // P873: キュー深度とリクエスト期限を解決する (CLI > env > 既定)。
    let queue_depth = resolve_max_queue_depth(cfg.max_queue_depth, cfg.max_queue);
    let request_timeout_secs = resolve_request_timeout_secs(cfg.request_timeout_secs);
    let limits = Limits {
        request: std::time::Duration::from_secs(request_timeout_secs),
        ..Limits::default()
    };
    // task 951: 認証キーを解決する。実効一覧 = `--api-key` / `--api-keys` /
    // `DENNO_API_KEYS` (いずれもカンマ区切り可) の和。空なら従来どおり開放。
    // ログには **本数だけ** を出し、値は決して出さない (task 971: 起動サマリの
    // `AuthState` が本数だけを持つ)。
    let env_api_keys = std::env::var(API_KEYS_ENV).ok();
    let api_key = resolve_api_keys(&[
        cfg.api_key.as_deref(),
        cfg.api_keys.as_deref(),
        env_api_keys.as_deref(),
    ]);
    let auth_state = summarize_auth(api_key.as_deref());
    let state = Arc::new(AppState {
        engine: engine.clone(),
        api_key,
        scheduler: Scheduler::spawn_with_limits(engine, queue_depth, limits),
        model_name,
        model_id,
        info,
        default_repetition_penalty: cfg.repetition_penalty,
        temperature_cap: cfg.temperature_cap,
    });

    // P13: テレメトリが有効ならハブ → broadcast への転送タスクを起動する。
    // (OFF のままならこのタスクは存在せず、捕捉コストもゼロ)
    if telem::enabled() {
        start_telemetry_relay();
    }

    // task 916: CORS の許可オリジンを解決する (CLI > env > ループバック既定)。
    let cors = resolve_cors_origin(cfg.cors_origin.as_deref());
    // task 946: ボディ上限を解決する (CLI > env > 既定 16 MiB)。
    if cfg.max_body_mb == Some(0) {
        anyhow::bail!("--max-body-mb must be >= 1");
    }
    let max_body_bytes = resolve_max_body_bytes(cfg.max_body_mb);
    let rate_limiter = ratelimit::RateLimiter::from_env();
    let rate_limit_summary = RateLimitSummary::from_config(&rate_limiter.config());
    let cors_summary = CorsSummary::from_config(&cors);
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let app = router_with_body_limit(state.clone(), cors, rate_limiter, max_body_bytes);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{}", addr);
    // task 971: 解決済みの設定を 1 つの起動サマリにまとめて出す。値 (API キー)
    // は AuthState が本数だけを持つので、この行から秘密は出ない。
    tracing::info!(
        "{}",
        format_startup_summary(&StartupSummary {
            host: &cfg.host,
            port: cfg.port,
            model_id: &state.model_id,
            model_name: &state.model_name,
            context_tokens: state.info.context,
            kv_slots: state.info.slots,
            quant: &state.info.quant,
            layers: state.info.layers,
            auth: auth_state,
            metrics_enabled: metrics::enabled(),
            queue_depth,
            request_timeout_secs,
            cors: cors_summary,
            max_body_bytes,
            rate_limit: rate_limit_summary,
        })
    );
    // P817: SIGTERM / SIGINT では即死せず drain する。新規要求は 503 にし、
    // 進行中の生成を短い上限 (`DENNO_DRAIN_TIMEOUT_MS`、既定 10 s) まで待って
    // から axum を止める。expert 使用回数もここで書く。
    axum::serve(listener, app)
        .with_graceful_shutdown(drain_on_signal(state))
        .await?;
    Ok(())
}

/// P817: 停止シグナルを待つ (unix は SIGTERM / SIGINT、それ以外は Ctrl-C)。
#[cfg(unix)]
async fn wait_for_shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("SIGTERM ハンドラを仕掛けられない: {e}");
            // 仕掛けられないなら SIGINT だけを待つ
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = term.recv() => tracing::info!("SIGTERM を受けた"),
        _ = tokio::signal::ctrl_c() => tracing::info!("SIGINT を受けた"),
    }
}

#[cfg(not(unix))]
async fn wait_for_shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("Ctrl-C を受けた");
}

/// P817: graceful drain。シグナルを受けたら
///
/// 1. `Lifecycle::begin_drain` — 以後の生成要求は 503 (ハンドラが弾く)
/// 2. キュー待ち + 進行中が 0 になるのを `drain_timeout()` まで待つ
/// 3. expert 使用回数を書く (`save_now` は未変更なら何もしない)
///
/// をしてから返る。返ると axum の graceful shutdown が走り、受付を止めて
/// 残りの接続が閉じるのを待つ。
async fn drain_on_signal(state: Arc<AppState>) {
    wait_for_shutdown_signal().await;
    state.scheduler.lifecycle().begin_drain();
    let timeout = lifecycle::drain_timeout();
    tracing::info!(
        "drain 開始: 新規リクエストは 503。進行中の生成を最大 {:.1}s 待つ",
        timeout.as_secs_f64()
    );
    let is_idle = || state.scheduler.pending() == 0 && state.scheduler.in_flight() == 0;
    let idle = lifecycle::wait_for_idle(
        is_idle,
        timeout,
        std::time::Duration::from_millis(lifecycle::DRAIN_POLL_MS),
    )
    .await;
    if idle {
        tracing::info!("drain 完了: 進行中の生成は無い");
    } else {
        tracing::warn!(
            "drain 上限 ({:.1}s) 超過: pending={} in_flight={} のまま停止する",
            timeout.as_secs_f64(),
            state.scheduler.pending(),
            state.scheduler.in_flight()
        );
    }
    // expert 使用回数の flush。ファイル書き込みはブロッキングなので
    // spawn_blocking に逃がす (decode は既に止まっている想定)。
    match tokio::task::spawn_blocking(denno_engine::model::exp_stats::save_now).await {
        Ok(Some(line)) => tracing::info!("{line}"),
        Ok(None) => {}
        Err(e) => tracing::warn!("expert 使用回数の保存に失敗: {e}"),
    }
}

// ---------- ハンドラ ----------

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

/// P817: readiness プローブ。liveness (/health) との違い:
///
/// - 200: モデルロード済み・ワーカー生存・drain 中でない (受付可能)
/// - 503: 起動途中 / drain 中 / ワーカーが死んだ
///
/// 認証は /health と同じく要求しない (プローブはトークンを持たない)。
/// 本文はオーケストレータ向けに理由と現在の負荷を返す (200 のときも同じ形)。
async fn ready(State(state): State<Arc<AppState>>) -> Response {
    let lc = state.scheduler.lifecycle();
    let ok = lc.serves_requests();
    let status = if ok { "ready" } else if lc.is_draining() { "draining" } else { "loading" };
    // P873: 打ち切り / 完了の計数 (全 API 共通のスケジューラ計数)。
    let st = state.scheduler.stats();
    let body = json!({
        "status": status,
        "ready": ok,
        "draining": lc.is_draining(),
        "model_loaded": lc.is_ready(),
        "worker_alive": lc.worker_alive(),
        "model": state.model_name,
        "context_length": state.info.context,
        "pending": state.scheduler.pending(),
        "in_flight": state.scheduler.in_flight(),
        "aborts": {
            "completed": st.completed,
            "timeout": st.aborted_timeout,
            "disconnected": st.aborted_disconnected,
            "error": st.aborted_error,
        },
        "limits": {
            "max_queue_depth": state.scheduler.max_queue(),
            "request_timeout_secs": state.scheduler.limits().request.as_secs(),
        },
    });
    let code = if ok { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (code, Json(body)).into_response()
}

/// P817: drain 中の新規生成要求を 503 (overloaded_error) にする。
/// 停止の途中で来た要求を**受けない**ことが drain の主目的 — 受けると
/// 「待ち行列が増え続けて drain が終わらない」状態になる。
pub(crate) fn reject_if_draining(state: &AppState) -> Result<(), ApiError> {
    if state.scheduler.lifecycle().is_draining() {
        return Err(ApiError::overloaded(
            "server is draining (shutdown signal received); the request was not accepted",
        ));
    }
    Ok(())
}

async fn models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_auth(&state, &headers)?;
    // P57: ここで engine.lock() を取らない (旧実装は名前と context のためだけに
    // 取っており、生成中は 200 すら返せなかった)。値は起動時のキャッシュ。
    // P793/P817/task 901: model object の組立は `openai::model_object` に
    // 集約した (`/v1/models/:id` と同じ本文を使う)。
    Ok(Json(json!({
        "object": "list",
        "data": [openai::model_object(&state)],
        "has_more": false,
    })))
}

/// task 951: 複数 API キーの環境変数名 (カンマ区切り)。
pub const API_KEYS_ENV: &str = "DENNO_API_KEYS";

/// task 951: カンマ区切りのキー一覧を正規化する。
/// 前後の空白を落とし、空要素と重複を除く (順序は保つ)。
pub fn parse_api_keys(raw: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for part in raw.split(',') {
        let k = part.trim();
        if !k.is_empty() && !out.iter().any(|e| e == k) {
            out.push(k.to_string());
        }
    }
    out
}

/// task 951: 複数の出所 (CLI フラグ / env) のキーを 1 本のカンマ区切りに
/// まとめる。1 つも無ければ None (= 認証なし = 従来どおり開放)。
pub fn resolve_api_keys(sources: &[Option<&str>]) -> Option<String> {
    let mut keys: Vec<String> = Vec::new();
    for raw in sources.iter().flatten() {
        for k in parse_api_keys(raw) {
            if !keys.iter().any(|e| e == &k) {
                keys.push(k);
            }
        }
    }
    if keys.is_empty() {
        None
    } else {
        Some(keys.join(","))
    }
}

/// task 951: バイト列の定数時間比較。
///
/// 長さが違っても `max(len)` 回だけ走査し、途中で早期 return しない
/// (一致した位置・接頭辞の長さをタイミングで漏らさない)。`black_box` で
/// 最適化による畳み込み (早期打ち切りへの書き換え) を防ぐ。
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff: u32 = (a.len() ^ b.len()) as u32;
    let n = a.len().max(b.len());
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0) as u32;
        let y = b.get(i).copied().unwrap_or(0) as u32;
        diff |= x ^ y;
    }
    std::hint::black_box(diff) == 0
}

/// `Authorization: Bearer <key>` の資格情報部を取り出す (スキーム名の
/// 大文字小文字は無視する)。それ以外のスキームは None。
fn bearer_credential(value: &str) -> Option<&str> {
    let (scheme, rest) = value.split_once(' ')?;
    if scheme.eq_ignore_ascii_case("bearer") {
        Some(rest.trim())
    } else {
        None
    }
}

pub(crate) fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(keys) = &state.api_key else { return Ok(()) };
    // task 951: `x-api-key` と `Authorization: Bearer` のどちらでも通す
    // (両方あるときは、どれか 1 つが一致すれば受理する)。
    let mut provided: [Option<&str>; 2] = [None, None];
    if let Some(v) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        provided[0] = Some(v.trim());
    }
    if let Some(v) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        provided[1] = bearer_credential(v);
    }
    // 全キー × 全候補を走査し、早期 return しない (どのキーで一致したかを
    // タイミングで漏らさない)。個々の比較は定数時間。
    let mut ok = false;
    for key in keys.split(',') {
        let key = key.trim();
        if key.is_empty() {
            continue;
        }
        for cand in provided.iter().flatten() {
            ok |= constant_time_eq(cand.as_bytes(), key.as_bytes());
        }
    }
    if ok {
        Ok(())
    } else {
        Err(ApiError::auth())
    }
}

/// P805: `/v1/messages` と `/v1/messages/count_tokens` が共有する
/// リクエスト正規化の結果 (chat template へ渡す形)。
///
/// 両エンドポイントが **同じ関数** を通ることで、`count_tokens` の値が
/// 生成時の `input_tokens` と一致することを構造的に保証する。
pub(crate) struct PreparedChat {
    pub system: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub params: GenParams,
    pub tools_for_prompt: Option<serde_json::Value>,
    pub tool_constraint: Option<ToolConstraint>,
    pub enable_thinking: Option<bool>,
    pub tools_active: bool,
}

/// Anthropic Messages リクエストを検証し、chat template に渡す形へ正規化する。
/// 不正は 400 (`ApiError::invalid_request`)、未知モデルは 404。
fn prepare_chat(state: &AppState, req: &MessagesRequest) -> Result<PreparedChat, ApiError> {
    // task 901: 登録別名 (canonical 名以外) も同じモデルとして受ける。
    if !openai::model_matches(state, &req.model) {
        return Err(ApiError::not_found(format!(
            "model: {} not found",
            req.model
        )));
    }
    if req.messages.is_empty() {
        return Err(ApiError::invalid_request(
            "messages: at least one message is required",
        ));
    }

    // tool use: 定義と tool_choice の検証 (tools 未指定時の tool_choice は 400)
    let tools = match &req.tools {
        None => Vec::new(),
        Some(t) => t.clone(),
    };
    if req.tool_choice.is_some() && tools.is_empty() {
        return Err(ApiError::invalid_request(
            "tool_choice is only allowed when tools are provided",
        ));
    }
    let tool_choice = parse_tool_choice(&req.tool_choice)?;
    let tools_json = validate_tools(&tools)?;
    if let ToolChoice::Named(n) = &tool_choice {
        if !tools.iter().any(|t| &t.name == n) {
            return Err(ApiError::invalid_request(format!(
                "tool_choice: tool {:?} is not defined in tools",
                n
            )));
        }
    }
    // ツールをモデルに見せるか (none = 非公開・抽出もしない。空配列も実質無し)
    let tools_active = !tools.is_empty() && tool_choice != ToolChoice::None;
    let tools_for_prompt = if tools_active { Some(tools_json) } else { None };
    // 第21: tool_choice any/Named の拘束デコード。any = 全定義ツール、
    // Named = 指定ツールのみを許容名として tool_call block までの生成を強制する
    // (auto/none は None = 無マスク経路とビット同一。any/Named 到達時点で
    // tools 非空は検証済み — tools 無し/未定義ツール指定は 400)。
    let tool_constraint = match &tool_choice {
        ToolChoice::Any => Some(ToolConstraint {
            names: tools.iter().map(|t| t.name.clone()).collect(),
        }),
        ToolChoice::Named(n) => Some(ToolConstraint { names: vec![n.clone()] }),
        ToolChoice::Auto | ToolChoice::None => None,
    };

    let mut tool_use_ids = HashSet::new();
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        if m.role != "user" && m.role != "assistant" {
            return Err(ApiError::invalid_request(format!(
                "messages[].role: must be \"user\" or \"assistant\", got {:?}",
                m.role
            )));
        }
        messages.push(ChatMessage {
            role: m.role.clone(),
            content: m.to_text(&mut tool_use_ids)?,
        });
    }
    let system = system_to_text(&req.system)?;
    // s47/s53c: thinking 制御。brief (既定) は system 冒頭に短答誘導文を注入し、
    // 併せて enable_thinking=false で思考を無効化する (実測: 思考 ON では 9B が
    // "Thinking Process:" を延々と書き回答に到達しない)。off も同じく無効化。
    let thinking = parse_thinking(&req.thinking)?;
    let system = match thinking {
        Thinking::Brief => brief_system(system),
        Thinking::Off | Thinking::Full => system,
    };
    let enable_thinking = match thinking {
        Thinking::Off | Thinking::Brief => Some(false),
        Thinking::Full => None,
    };
    // s39: max_tokens / max_completion_tokens の未指定時は既定上限 (DEFAULT_MAX_TOKENS)
    // を適用する。明示指定は尊重。数値経路への影響は無い (設定値の既定適用のみ)。
    // s56: 反復抑止の追加パラメータ検証 (不正値は 400。未指定 = 無効 = ビット不変)
    let anti = parse_anti_repetition(req)?;
    let params = GenParams {
        max_tokens: resolve_max_tokens(req.max_tokens, req.max_completion_tokens),
        temperature: apply_temperature_cap(req.temperature, state.temperature_cap),
        top_p: req.top_p,
        top_k: req.top_k,
        stop_sequences: req.stop_sequences.clone(),
        seed: None,
        repetition_penalty: req.repetition_penalty.or(state.default_repetition_penalty),
        frequency_penalty: anti.frequency,
        presence_penalty: anti.presence,
        no_repeat_ngram_size: anti.no_repeat_ngram_size,
    };

    Ok(PreparedChat {
        system,
        messages,
        params,
        tools_for_prompt,
        tool_constraint,
        enable_thinking,
        tools_active,
    })
}

async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    check_auth(&state, &headers)?;
    // P817: drain 中は新規の生成要求を受けない (503 overloaded_error)。
    reject_if_draining(&state)?;
    // s148 (Task B10・計測のみ): リクエストの壁時計。COVERAGE 行の分母
    // (「帰属 ms / リクエストの壁時計」) になる。
    // s151 (Task B12): このフックは `/v1/messages` の**同期ハンドラ**にあり、
    // ストリーミング (本番の serve 経路はほぼ SSE) はレスポンス本体を返した時点で
    // 戻るため、生成が終わった時点の壁時計になっていなかった (`requests 0`)。
    // 実際の計上は scheduler の Run (prefill 開始 → セッション完了) が
    // `request_begin` / `request_end` で行う — ここでは二重計上しない。
    let resp = messages_inner(state, headers, body).await;
    resp
}

/// s148: [`messages`] の本体 (リクエスト壁時計の計測を外したもの)。
async fn messages_inner(
    state: Arc<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    let req: MessagesRequest = serde_json::from_slice(&body).map_err(|e| {
        ApiError::invalid_request(format!("invalid request body: {}", e))
    })?;
    let PreparedChat {
        system,
        messages: chat_messages,
        params,
        tools_for_prompt,
        tool_constraint,
        enable_thinking,
        tools_active,
    } = prepare_chat(&state, &req)?;

    if req.stream.unwrap_or(false) {
        // 第27: tools + stream を許可。SSE アセンブラ (stream_asm) が delta ストリームの
        // <tool_call> block を検出し tool_use block (content_block_start /
        // input_json_delta / content_block_stop) として流す
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        state
            .scheduler
            .submit_stream(
                system,
                chat_messages,
                params,
                tools_for_prompt,
                tool_constraint,
                enable_thinking,
                tx,
            )
            .map_err(queue_full_error(&state))?;
        let stream = ReceiverStream::new(rx);
        Ok(Sse::new(stream)
            // prefill 中にイベントが途切れるとクライアント (Hermes 等) がアイドル
            // タイムアウトで接続を切ってしまうため、短い間隔で keep-alive を送る。
            .keep_alive(
                KeepAlive::new()
                    .interval(std::time::Duration::from_secs(2))
                    .text("ping"),
            )
            .into_response())
    } else {
        let rx = state
            .scheduler
            .submit_chat(system, chat_messages, params, tools_for_prompt, tool_constraint, enable_thinking)
            .map_err(queue_full_error(&state))?;
        // P57: 待ちに上限を入れる。旧実装は `rx.await` を無期限に待ったため、
        // ワーカーが 1 ステップで詰まるとクライアントは永久に無応答だった。
        // 期限を超えたら 500 を返し、`rx` を落とす (= ワーカーから見て
        // 「クライアント切断」と同じ扱いになり、次のステップ境界で
        // KV スロットが解放される)。
        // P873: `--request-timeout-secs` が有効ならその少し後まで待つ
        // (スケジューラ側の期限が先に発火して、期限超過を説明するエラーを返す)。
        let wait = response_wait_timeout(state.scheduler.limits());
        let waited = tokio::time::timeout(wait, rx).await;
        let mut output = match waited {
            Ok(Ok(Ok(out))) => out,
            Ok(Ok(Err(e))) => return Err(ApiError::from_anyhow(e)),
            Ok(Err(e)) => {
                return Err(ApiError::api(format!("scheduler dropped: {e}")))
            }
            Err(_) => {
                return Err(ApiError::api(format!(
                    "timeout: no response from the engine within {:.1}s; \
                     the request was abandoned (retry later)",
                    wait.as_secs_f64()
                )))
            }
        };
        // task 926: prefix cache の利用実績を 1 回だけ数える (非ストリーミング経路)。
        // ストリーミングは message_start (usage) を `to_event` が拾う。
        record_prefix_cache_reuse(output.cache_read_input_tokens);
        // tool use: モデル出力から tool_call block を抽出し content blocks を組む
        let mut calls = Vec::new();
        if tools_active {
            let (clean_text, extracted) = extract_tool_calls(&output.text);
            output.text = clean_text;
            calls = extracted;
        }
        Ok(Json(message_json(&state.model_name, &output, &calls)).into_response())
    }
}

/// P805: `POST /v1/messages/count_tokens`。
///
/// Anthropic 互換の `{"input_tokens": N}` を返す。生成経路と同じ正規化
/// ([`prepare_chat`]) と同じトークナイザを使うが、**prefill も KV も触らない**
/// (`Engine::count_prompt_tokens`)。Claude Code は送信前にこの端点を叩く。
async fn count_tokens(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_auth(&state, &headers)?;
    let req: MessagesRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::invalid_request(format!("invalid request body: {}", e)))?;
    let prepared = prepare_chat(&state, &req)?;
    let engine = state.engine.clone();
    let n = tokio::task::spawn_blocking(move || -> Result<usize, ApiError> {
        // P57 と同じ理由でハンドラは生成の直列化に巻き込まない。ロックが
        // 取れない間は短い間隔で再試行し、上限を超えたら 503 で返す
        // (無応答にしない)。
        let deadline = std::time::Instant::now() + COUNT_TOKENS_LOCK_TIMEOUT;
        loop {
            match engine.try_lock() {
                Ok(g) => {
                    return g
                        .count_prompt_tokens(
                            prepared.system.as_deref(),
                            &prepared.messages,
                            prepared.tools_for_prompt.as_ref(),
                            prepared.enable_thinking,
                        )
                        .map_err(ApiError::from_anyhow)
                }
                Err(std::sync::TryLockError::WouldBlock) => {
                    if std::time::Instant::now() >= deadline {
                        return Err(ApiError::overloaded(format!(
                            "engine is busy; count_tokens gave up after {:.1}s (retry later)",
                            COUNT_TOKENS_LOCK_TIMEOUT.as_secs_f64()
                        )));
                    }
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                Err(std::sync::TryLockError::Poisoned(_)) => {
                    return Err(ApiError::api("engine lock poisoned"))
                }
            }
        }
    })
    .await
    .map_err(|e| ApiError::api(format!("count_tokens task failed: {e}")))??;
    Ok(Json(json!({ "input_tokens": n })))
}

/// キュー満杯時の 503 エラー (Anthropic 流 overloaded_error)
///
/// P873: `Retry-After` を付ける (クライアントが待ち時間を知らずに即再試行して
/// 同じ混雑を繰り返すのを避ける)。ステータスは既存契約の 503 のまま
/// (Anthropic の過負荷は overloaded_error)。
fn queue_full_error(state: &AppState) -> impl Fn(QueueFull) -> ApiError + '_ {
    move |e: QueueFull| {
        ApiError::overloaded_retry_after(
            format!(
                "server busy: request queue is full (max_queue = {}, in_flight = 1); retry later",
                state.scheduler.max_queue()
            ),
            e.retry_after_secs(),
        )
    }
}

/// 応答 JSON を組む。`calls` が空でなければ content は
/// [text block (非空なら)] + tool_use blocks、stop_reason は "tool_use"。
fn message_json(
    model: &str,
    out: &ChatOutput,
    calls: &[ExtractedToolCall],
) -> serde_json::Value {
    let mut content = Vec::new();
    if !out.text.is_empty() || calls.is_empty() {
        content.push(json!({"type": "text", "text": out.text}));
    }
    for c in calls {
        content.push(json!({
            "type": "tool_use",
            "id": new_tool_id(),
            "name": c.name,
            "input": c.input,
        }));
    }
    let (stop_reason, stop_sequence) = if calls.is_empty() {
        (out.stop_reason.as_str(), out.stop_sequence.clone())
    } else {
        ("tool_use", None)
    };
    json!({
        "id": new_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
            // P805: KV prefix 再利用の内訳 (Anthropic 互換)。
            // cache_read = 再利用した先頭トークン数、cache_creation = 新規 prefill 分。
            "cache_read_input_tokens": out.cache_read_input_tokens,
            "cache_creation_input_tokens": out.cache_creation_input_tokens,
        }
    })
}

pub(crate) fn new_msg_id() -> String {
    let mut rng = rand::thread_rng();
    let id: String = (0..24).map(|_| format!("{:x}", rng.gen::<u8>() % 16)).collect();
    format!("msg_{id}")
}

pub(crate) fn to_event(v: &serde_json::Value) -> Event {
    // task 926: SSE の `message_start` は scheduler が prefix 再利用長
    // (`usage.cache_read_input_tokens`) を載せて 1 回だけ流す。ここで数えると
    // ストリーミング経路のヒット/ミスをハンドラ側で二重計上せずに済む
    // (比較 1 回だけで、値には触れない)。
    if v.get("type").and_then(|t| t.as_str()) == Some("message_start") {
        if let Some(reused) = v
            .pointer("/message/usage/cache_read_input_tokens")
            .and_then(|n| n.as_u64())
        {
            record_prefix_cache_reuse(reused as usize);
        }
    }
    // task 976: Anthropic は `stop_reason` が `"tool_use"` のとき `stop_sequence`
    // を null にする (非ストリーミング `message_json` と同一規則)。scheduler の
    // message_delta は engine の `out.stop_sequence` をそのまま載せるため、
    // tool_use 上書き時だけ SSE へ出す直前に正規化してパリティを取る
    // (それ以外の stop_reason では何もしない = 既定経路は不変)。
    let mut normalized: Option<serde_json::Value> = None;
    if v.get("type").and_then(|t| t.as_str()) == Some("message_delta")
        && v.pointer("/delta/stop_reason").and_then(|r| r.as_str()) == Some("tool_use")
        && v.pointer("/delta/stop_sequence")
            .map_or(false, |s| !s.is_null())
    {
        let mut n = v.clone();
        if let Some(d) = n.get_mut("delta").and_then(|d| d.as_object_mut()) {
            d.insert("stop_sequence".to_string(), serde_json::Value::Null);
        }
        normalized = Some(n);
    }
    let v = normalized.as_ref().unwrap_or(v);
    // Anthropic の SSE は各イベントに event: <type> 行が必要
    // (例: event: message_start / data: {...})
    let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("message");
    Event::default().event(ty).data(v.to_string())
}

// ---------- P13: 推論テレメトリ WebSocket (denno-trace) ----------

/// ハブ → broadcast の転送経路 (1 プロセス 1 個)。
/// ハブの単一消費者は下の relay スレッドで、複数の WS クライアントには
/// broadcast で配信する (クライアントが追いつかない場合は Lagged → Gap フレーム)。
static TEL_BCAST: OnceLock<tokio::sync::broadcast::Sender<Frame>> = OnceLock::new();

/// ハブの単一消費者スレッドを起動する (telemetry 有効時のみ run() から呼ぶ)。
fn start_telemetry_relay() {
    let (tx, _) = tokio::sync::broadcast::channel::<Frame>(256);
    if TEL_BCAST.set(tx).is_err() {
        return; // 既に起動済み
    }
    let btx = TEL_BCAST.get().unwrap().clone();
    std::thread::Builder::new()
        .name("denno-telemetry".into())
        .spawn(move || {
            let mut last_dropped = telem::dropped();
            loop {
                // ドロップが増えていたら Gap を先に流す (黙って飛ばさない)
                let now_dropped = telem::dropped();
                if now_dropped > last_dropped {
                    let lost = (now_dropped - last_dropped) as u32;
                    last_dropped = now_dropped;
                    let _ = btx.send(Frame::Gap { lost });
                }
                match telem::try_recv_timeout(std::time::Duration::from_millis(50)) {
                    Some(f) => {
                        let _ = btx.send(f);
                    }
                    None => continue,
                }
            }
        })
        .expect("telemetry relay thread の起動に失敗");
}

/// GET /v1/telemetry/ws — 推論テレメトリのバイナリフレームストリーム。
async fn telemetry_ws(
    State(state): State<Arc<AppState>>,
    upgrade: WebSocketUpgrade,
) -> Response {
    if !telem::enabled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "telemetry disabled (set DENNO_TELEMETRY=1)",
        )
            .into_response();
    }
    upgrade.on_upgrade(move |socket| telemetry_socket(socket, state))
}

async fn telemetry_socket(mut socket: WebSocket, state: Arc<AppState>) {
    // 最初に ServerInfo (モデル/モデルID/量子化/スロット数/コンテキスト長/層数)
    // P57: エンジンロックを取らない (生成中でもハンドシェイクが通る)。値は
    // 起動時のキャッシュ (`EngineInfo`) — モデルは実行中に変わらない。
    let (model, model_id, quant, slots, context, layers) = (
        state.model_name.clone(),
        state.model_id.clone(),
        state.info.quant.clone(),
        state.info.slots as u32,
        state.info.context as u32,
        state.info.layers as u32,
    );
    let info = Frame::ServerInfo { model, model_id, quant, slots, context, layers };
    if socket.send(Message::Binary(info.encode())).await.is_err() {
        return;
    }
    let Some(btx) = TEL_BCAST.get() else { return };
    let mut rx = btx.subscribe();
    loop {
        tokio::select! {
            r = rx.recv() => {
                match r {
                    Ok(f) => {
                        if socket.send(Message::Binary(f.encode())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        // クライアントが追いつかなかった分を Gap で明示
                        let gap = Frame::Gap { lost: n as u32 };
                        if socket.send(Message::Binary(gap.encode())).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
            // クライアント→サーバーのコマンド (focus プロトコルは P15)。今は無視。
            m = socket.recv() => {
                match m {
                    Some(Ok(Message::Binary(_))) | Some(Ok(Message::Text(_))) | Some(Ok(_)) => {}
                    _ => break,
                }
            }
        }
    }
}

// ---------- tool use テスト ----------

#[cfg(test)]
mod tool_tests {
    use super::*;
    use denno_engine::StopReason;

    fn tool_defs() -> Vec<ToolDef> {
        vec![ToolDef {
            name: "get_weather".into(),
            description: Some("Get weather".into()),
            input_schema: json!({"type": "object", "properties": {"city": {"type": "string"}}}),
        }]
    }

    fn chat_output(text: impl Into<String>) -> ChatOutput {
        ChatOutput {
            text: text.into(),
            stop_reason: StopReason::EndTurn,
            stop_sequence: None,
            input_tokens: 10,
            output_tokens: 5,
            logprobs: Vec::new(),
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 10,
        }
    }

    // ---- tool_choice パース ----

    #[test]
    fn tool_choice_parse() {
        assert_eq!(parse_tool_choice(&None).unwrap(), ToolChoice::Auto);
        for (v, want) in [
            (json!("auto"), ToolChoice::Auto),
            (json!("none"), ToolChoice::None),
            (json!("any"), ToolChoice::Any),
            (json!({"type": "auto"}), ToolChoice::Auto),
            (json!({"type": "none"}), ToolChoice::None),
            (json!({"type": "any"}), ToolChoice::Any),
            (json!({"type": "tool", "name": "f"}), ToolChoice::Named("f".into())),
        ] {
            assert_eq!(parse_tool_choice(&Some(v)).unwrap(), want);
        }
        assert!(parse_tool_choice(&Some(json!("bogus"))).is_err());
        assert!(parse_tool_choice(&Some(json!(1))).is_err());
        assert!(parse_tool_choice(&Some(json!({"name": "f"}))).is_err());
        assert!(parse_tool_choice(&Some(json!({"type": "tool"}))).is_err());
        assert!(parse_tool_choice(&Some(json!({"type": "tool", "name": ""}))).is_err());
        assert!(parse_tool_choice(&Some(json!({"type": "bogus"}))).is_err());
    }

    // ---- tools 定義の検証 ----

    #[test]
    fn validate_tools_ok_and_err() {
        let v = validate_tools(&tool_defs()).unwrap();
        assert_eq!(
            v,
            json!([{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            }])
        );
        // 空配列 OK
        assert_eq!(validate_tools(&[]).unwrap(), json!([]));
        // name 空
        let bad = vec![ToolDef {
            name: " ".into(),
            description: None,
            input_schema: json!({}),
        }];
        assert!(validate_tools(&bad).is_err());
        // input_schema が object 以外
        let bad = vec![ToolDef {
            name: "f".into(),
            description: None,
            input_schema: json!("not an object"),
        }];
        assert!(validate_tools(&bad).is_err());
    }

    // ---- tool_use (assistant 履歴) → XML 変換 ----

    #[test]
    fn assistant_tool_use_block_becomes_xml() {
        let m = InMessage {
            role: "assistant".into(),
            content: json!([
                {"type": "text", "text": "Let me check."},
                {"type": "tool_use", "id": "toolu_01", "name": "get_weather",
                 "input": {"city": "Tokyo", "days": 3, "rain": true}}
            ]),
        };
        let mut ids = HashSet::new();
        let text = m.to_text(&mut ids).unwrap();
        assert_eq!(
            text,
            "Let me check.\n\n<tool_call>\n<function=get_weather>\n\
             <parameter=city>\nTokyo\n</parameter>\n\
             <parameter=days>\n3\n</parameter>\n\
             <parameter=rain>\ntrue\n</parameter>\n\
             </function>\n</tool_call>"
        );
        assert!(ids.contains("toolu_01"));
    }

    #[test]
    fn assistant_tool_use_validation_errors() {
        let mut ids = HashSet::new();
        // user への tool_use は不可
        let m = InMessage {
            role: "user".into(),
            content: json!([{"type": "tool_use", "id": "x", "name": "f", "input": {}}]),
        };
        assert!(m.to_text(&mut ids).is_err());
        // input が object 以外
        let m = InMessage {
            role: "assistant".into(),
            content: json!([{"type": "tool_use", "id": "x", "name": "f", "input": [1]}]),
        };
        assert!(m.to_text(&mut ids).is_err());
        // id 欠落
        let m = InMessage {
            role: "assistant".into(),
            content: json!([{"type": "tool_use", "name": "f", "input": {}}]),
        };
        assert!(m.to_text(&mut ids).is_err());
        // 重複 id
        let m = InMessage {
            role: "assistant".into(),
            content: json!([{"type": "tool_use", "id": "dup", "name": "f", "input": {}}]),
        };
        assert!(m.to_text(&mut ids).is_ok());
        let m = InMessage {
            role: "assistant".into(),
            content: json!([{"type": "tool_use", "id": "dup", "name": "f", "input": {}}]),
        };
        assert!(m.to_text(&mut ids).is_err());
    }

    // ---- tool_result (user 履歴) → tool_response 変換 ----

    #[test]
    fn user_tool_result_block_becomes_tool_response() {
        let mut ids = HashSet::new();
        let m = InMessage {
            role: "assistant".into(),
            content: json!([{"type": "tool_use", "id": "toolu_09", "name": "f", "input": {}}]),
        };
        m.to_text(&mut ids).unwrap();

        // 文字列 content
        let m = InMessage {
            role: "user".into(),
            content: json!([
                {"type": "tool_result", "tool_use_id": "toolu_09", "content": "sunny 25C"}
            ]),
        };
        let text = m.to_text(&mut ids).unwrap();
        assert_eq!(text, "<tool_response>\nsunny 25C\n</tool_response>");

        // 配列 content (text blocks)
        let m = InMessage {
            role: "user".into(),
            content: json!([
                {"type": "tool_result", "tool_use_id": "toolu_09",
                 "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
            ]),
        };
        let text = m.to_text(&mut ids).unwrap();
        assert_eq!(text, "<tool_response>\nab\n</tool_response>");

        // text + tool_result の混合
        let m = InMessage {
            role: "user".into(),
            content: json!([
                {"type": "text", "text": "result below"},
                {"type": "tool_result", "tool_use_id": "toolu_09", "content": "x"}
            ]),
        };
        let text = m.to_text(&mut ids).unwrap();
        assert_eq!(text, "result below\n<tool_response>\nx\n</tool_response>");
    }

    #[test]
    fn user_tool_result_validation_errors() {
        let mut ids = HashSet::new();
        // 未知の tool_use_id
        let m = InMessage {
            role: "user".into(),
            content: json!([{"type": "tool_result", "tool_use_id": "nope", "content": "x"}]),
        };
        assert!(m.to_text(&mut ids).is_err());
        // assistant への tool_result は不可
        let m = InMessage {
            role: "assistant".into(),
            content: json!([{"type": "tool_result", "tool_use_id": "nope", "content": "x"}]),
        };
        assert!(m.to_text(&mut ids).is_err());
        // content が文字列/配列以外
        let m = InMessage {
            role: "user".into(),
            content: json!([{"type": "tool_result", "tool_use_id": "nope", "content": 3}]),
        };
        assert!(m.to_text(&mut ids).is_err());
    }

    // ---- tool_call 抽出 ----

    fn xml_call(name: &str, params: &[(&str, &str)]) -> String {
        let mut s = format!("<tool_call>\n<function={name}>\n");
        for (k, v) in params {
            s.push_str(&format!("<parameter={k}>\n{v}\n</parameter>\n"));
        }
        s.push_str("</function>\n</tool_call>");
        s
    }

    #[test]
    fn extract_xml_tool_call_with_text() {
        let text = format!(
            "Let me check the weather.\n\n{}\n",
            xml_call("get_weather", &[("city", "Tokyo"), ("days", "3")])
        );
        let (clean, calls) = extract_tool_calls(&text);
        assert_eq!(clean, "Let me check the weather.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].input,
            json!({"city": "Tokyo", "days": 3}) // 文字列は raw・数値は JSON 値
        );
    }

    #[test]
    fn extract_multiple_calls_in_order() {
        let text = format!(
            "{}\nmiddle text\n{}",
            xml_call("f1", &[("a", "1")]),
            xml_call("f2", &[])
        );
        let (clean, calls) = extract_tool_calls(&text);
        assert_eq!(clean, "middle text");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "f1");
        assert_eq!(calls[0].input, json!({"a": 1}));
        assert_eq!(calls[1].name, "f2");
        assert_eq!(calls[1].input, json!({}));
    }

    #[test]
    fn extract_xml_param_value_types() {
        let text = xml_call(
            "f",
            &[
                ("s", "hello world"),              // raw 文字列
                ("n", "42"),                        // 数値
                ("b", "false"),                     // 真偽
                ("arr", "[1, 2]"),                  // 配列
                ("obj", "{\"k\": \"v\"}"),          // object
                ("q", "\"quoted\""),                // 引用文字列
                ("multi", "line1\nline2"),          // 複数行 raw 文字列
            ],
        );
        let (clean, calls) = extract_tool_calls(&text);
        assert_eq!(clean, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].input,
            json!({
                "s": "hello world",
                "n": 42,
                "b": false,
                "arr": [1, 2],
                "obj": {"k": "v"},
                "q": "quoted",
                "multi": "line1\nline2",
            })
        );
    }

    #[test]
    fn extract_json_style_tool_call() {
        let (clean, calls) = extract_tool_calls(
            r#"prefix
<tool_call>
{"name": "get_weather", "arguments": {"city": "Osaka"}}
</tool_call>"#,
        );
        assert_eq!(clean, "prefix");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input, json!({"city": "Osaka"}));

        // parameters / input キーも受ける・arguments 無しは {}
        for args_key in ["parameters", "input"] {
            let body = format!(
                r#"{{"name": "f", "{args_key}": {{"x": 1}}}}"#
            );
            let text = format!("<tool_call>\n{body}\n</tool_call>");
            let (_, calls) = extract_tool_calls(&text);
            assert_eq!(calls.len(), 1);
            assert_eq!(calls[0].input, json!({"x": 1}));
        }
        let (_, calls) = extract_tool_calls("<tool_call>\n{\"name\": \"f\"}\n</tool_call>");
        assert_eq!(calls[0].input, json!({}));
    }

    #[test]
    fn incomplete_or_invalid_blocks_stay_in_text() {
        // 未完結 (閉じ tag 無し)
        let (clean, calls) = extract_tool_calls("<tool_call>\n<function=f>\n");
        assert!(calls.is_empty());
        assert_eq!(clean, "<tool_call>\n<function=f>");

        // 閉じているが中身が不正 (name 欠落 JSON)
        let (clean, calls) = extract_tool_calls("<tool_call>\n{\"arguments\": {}}\n</tool_call>");
        assert!(calls.is_empty());
        assert_eq!(clean, "<tool_call>\n{\"arguments\": {}}\n</tool_call>");

        // name が空の XML
        let (clean, calls) = extract_tool_calls("<tool_call>\n<function=>\n</function>\n</tool_call>");
        assert!(calls.is_empty());
        assert_eq!(clean, "<tool_call>\n<function=>\n</function>\n</tool_call>");

        // tool_call tag 無し
        let (clean, calls) = extract_tool_calls("just text");
        assert!(calls.is_empty());
        assert_eq!(clean, "just text");
    }

    #[test]
    fn roundtrip_history_xml_is_extracted() {
        // 履歴織り込み (tool_use_to_xml) で作ったテキストは抽出関数で復元できる
        let xml = tool_use_to_xml(
            "get_weather",
            &json!({"city": "Tokyo", "days": 3, "rain": true}),
        );
        let (clean, calls) = extract_tool_calls(&xml);
        assert_eq!(clean, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].input, json!({"city": "Tokyo", "days": 3, "rain": true}));
    }

    // ---- 応答 JSON 組み立て ----

    #[test]
    fn message_json_without_calls_is_legacy() {
        let out = chat_output("hello");
        let v = message_json("m", &out, &[]);
        assert_eq!(
            v["content"],
            json!([{"type": "text", "text": "hello"}])
        );
        assert_eq!(v["stop_reason"], "end_turn");
    }

    #[test]
    fn message_json_with_calls_has_tool_use_blocks() {
        let out = chat_output(format!(
            "Checking...\n{}",
            xml_call("get_weather", &[("city", "Tokyo")])
        ));
        let (clean, calls) = extract_tool_calls(&out.text);
        let out = ChatOutput { text: clean, ..out };
        let v = message_json("m", &out, &calls);
        assert_eq!(v["stop_reason"], "tool_use");
        assert_eq!(v["stop_sequence"], serde_json::Value::Null);
        let content = v["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "Checking...");
        assert_eq!(content[1]["type"], "tool_use");
        assert_eq!(content[1]["name"], "get_weather");
        assert_eq!(content[1]["input"], json!({"city": "Tokyo"}));
        let id = content[1]["id"].as_str().unwrap();
        assert!(id.starts_with("toolu_") && id.len() == "toolu_".len() + 24);
    }

    #[test]
    fn message_json_empty_text_with_calls_drops_text_block() {
        // handler は extract_tool_calls の trim 済みテキストを渡すため同じにする
        let (clean, _) = extract_tool_calls("  \n");
        let out = chat_output(clean);
        let v = message_json("m", &out, &[ExtractedToolCall {
            name: "f".into(),
            input: json!({}),
        }]);
        let content = v["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "tool_use");
    }
}

/// s56 (T2): 反復抑止の追加パラメータ (frequency / presence / no_repeat_ngram)
/// の受理・検証・GenParams への伝播。
#[cfg(test)]
mod anti_repetition_tests {
    use super::*;

    fn req(v: serde_json::Value) -> MessagesRequest {
        let mut base = serde_json::json!({
            "model": "m",
            "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}],
        });
        for (k, val) in v.as_object().unwrap() {
            base[k] = val.clone();
        }
        serde_json::from_value(base).expect("リクエストがパースできない")
    }

    /// 未指定 = すべて None (無効)。追加フィールドの既定は挙動不変。
    #[test]
    fn anti_repetition_absent_is_none() {
        let r = req(serde_json::json!({}));
        assert_eq!(r.frequency_penalty, None);
        assert_eq!(r.presence_penalty, None);
        assert_eq!(r.no_repeat_ngram_size, None);
        let a = parse_anti_repetition(&r).unwrap();
        assert_eq!(a, GenAntiRepetition::default());
    }

    /// 指定値がそのまま通る (frequency/presence は負値も許す・n=2 以上)。
    #[test]
    fn anti_repetition_values_are_passed_through() {
        let r = req(serde_json::json!({
            "frequency_penalty": 0.5, "presence_penalty": -0.25, "no_repeat_ngram_size": 4
        }));
        let a = parse_anti_repetition(&r).unwrap();
        assert_eq!(a.frequency, Some(0.5));
        assert_eq!(a.presence, Some(-0.25));
        assert_eq!(a.no_repeat_ngram_size, Some(4));
    }

    /// 0.0 は「無効」= None に正規化される (既定と同じ経路に落ちる)。
    #[test]
    fn anti_repetition_zero_is_normalized_to_none() {
        let r = req(serde_json::json!({"frequency_penalty": 0.0, "presence_penalty": -0.0}));
        let a = parse_anti_repetition(&r).unwrap();
        assert_eq!(a.frequency, None);
        assert_eq!(a.presence, None);
        // 負のゼロも 0 と同じ扱い (符号で分岐しない)
        let r2 = req(serde_json::json!({"frequency_penalty": -0.0}));
        assert_eq!(parse_anti_repetition(&r2).unwrap().frequency, None);
    }

    /// 不正値は 400 (invalid_request_error): n<2 / n>上限 / NaN / inf。
    #[test]
    fn anti_repetition_invalid_values_are_rejected() {
        for v in [
            serde_json::json!({"no_repeat_ngram_size": 0}),
            serde_json::json!({"no_repeat_ngram_size": 1}),
            serde_json::json!({"no_repeat_ngram_size": MAX_NGRAM_SIZE + 1}),
            serde_json::json!({"frequency_penalty": "NaN"}),
            serde_json::json!({"presence_penalty": "inf"}),
            serde_json::json!({"frequency_penalty": 1e40}),
            serde_json::json!({"repetition_penalty": 1e40}),
        ] {
            let r = serde_json::from_value::<MessagesRequest>(serde_json::json!({
                "model": "m", "max_tokens": 8,
                "messages": [{"role": "user", "content": "hi"}],
            }).as_object().map(|o| {
                let mut o = o.clone();
                for (k, val) in v.as_object().unwrap() { o.insert(k.clone(), val.clone()); }
                serde_json::Value::Object(o)
            }).unwrap());
            // パース自体が落ちる場合 (型違い) も不正入力として扱う
            if let Ok(r) = r {
                assert!(parse_anti_repetition(&r).is_err(), "受理してはいけない: {v}");
            }
        }
        // 境界: n = 2 と n = 上限は受理
        for n in [2usize, MAX_NGRAM_SIZE] {
            let r = req(serde_json::json!({"no_repeat_ngram_size": n}));
            assert_eq!(parse_anti_repetition(&r).unwrap().no_repeat_ngram_size, Some(n));
        }
    }

    /// エラーは 400 invalid_request_error (既存の作法と同じ)。
    #[test]
    fn anti_repetition_error_is_400_invalid_request() {
        let r = req(serde_json::json!({"no_repeat_ngram_size": 1}));
        let e = parse_anti_repetition(&r).unwrap_err();
        assert_eq!(e.status, StatusCode::BAD_REQUEST);
        assert_eq!(e.kind, "invalid_request_error");
        assert!(e.message.contains("no_repeat_ngram_size"), "{}", e.message);
    }

    /// ハンドラが組む GenParams と同じ形で 3 フィールドが伝わること
    /// (engine 側の AntiRepetition に落ちて no-op 判定が変わる)。
    #[test]
    fn anti_repetition_reaches_gen_params() {
        let r = req(serde_json::json!({
            "repetition_penalty": 1.1,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.2,
            "no_repeat_ngram_size": 3,
        }));
        let a = parse_anti_repetition(&r).unwrap();
        let params = GenParams {
            max_tokens: resolve_max_tokens(r.max_tokens, r.max_completion_tokens),
            temperature: r.temperature,
            top_p: r.top_p,
            top_k: r.top_k,
            stop_sequences: r.stop_sequences.clone(),
            seed: None,
            repetition_penalty: r.repetition_penalty,
            frequency_penalty: a.frequency,
            presence_penalty: a.presence,
            no_repeat_ngram_size: a.no_repeat_ngram_size,
        };
        assert_eq!(params.repetition_penalty, Some(1.1));
        assert_eq!(params.frequency_penalty, Some(0.3));
        assert_eq!(params.presence_penalty, Some(0.2));
        assert_eq!(params.no_repeat_ngram_size, Some(3));
        let ar = denno_engine::sampler::AntiRepetition {
            repetition_penalty: params.repetition_penalty,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: params.presence_penalty,
            no_repeat_ngram_size: params.no_repeat_ngram_size,
        };
        assert!(!ar.is_noop(), "指定時は no-op ではない (適用される)");
        // すべて未指定なら no-op (= 既定挙動不変)
        let r0 = req(serde_json::json!({}));
        let a0 = parse_anti_repetition(&r0).unwrap();
        let ar0 = denno_engine::sampler::AntiRepetition {
            repetition_penalty: r0.repetition_penalty,
            frequency_penalty: a0.frequency,
            presence_penalty: a0.presence,
            no_repeat_ngram_size: a0.no_repeat_ngram_size,
        };
        assert!(ar0.is_noop());
    }
}

#[cfg(test)]
mod temperature_cap_tests {
    use super::apply_temperature_cap;

    #[test]
    fn no_cap_is_passthrough() {
        // フラグ未指定なら従来と完全に同一 (None は None のまま = sampler 既定)
        assert_eq!(apply_temperature_cap(None, None), None);
        assert_eq!(apply_temperature_cap(Some(1.0), None), Some(1.0));
        assert_eq!(apply_temperature_cap(Some(0.3), None), Some(0.3));
    }

    #[test]
    fn cap_clamps_requested_temperature() {
        assert_eq!(apply_temperature_cap(Some(1.0), Some(0.7)), Some(0.7));
        assert_eq!(apply_temperature_cap(Some(0.9), Some(0.7)), Some(0.7));
        assert_eq!(apply_temperature_cap(Some(0.3), Some(0.7)), Some(0.3));
    }

    #[test]
    fn cap_applies_to_default_when_omitted() {
        // sampler の既定は 1.0 なので、リクエスト未指定でも cap で頭打ちにする
        assert_eq!(apply_temperature_cap(None, Some(0.7)), Some(0.7));
        assert_eq!(apply_temperature_cap(None, Some(0.0)), Some(0.0));
    }

    #[test]
    fn cap_above_request_does_not_raise_it() {
        assert_eq!(apply_temperature_cap(Some(0.2), Some(2.0)), Some(0.2));
    }
}

/// task 991: `system` のブロック配列の正規化 (連結順・空 text・マルチモーダル拒否)。
#[cfg(test)]
mod system_blocks_991_tests {
    use super::system_to_text;
    use serde_json::json;

    fn err_msg(v: serde_json::Value) -> String {
        match system_to_text(&Some(v)) {
            Ok(ok) => panic!("400 を期待したが通った: {ok:?}"),
            Err(e) => format!("{e:?}"),
        }
    }

    #[test]
    fn string_system_is_passthrough() {
        assert_eq!(
            system_to_text(&Some(json!("You are a test."))).unwrap(),
            Some("You are a test.".to_string())
        );
        assert_eq!(system_to_text(&None).unwrap(), None);
        // 空文字列は従来どおり通す (ブロック配列の空 text だけが 400)
        assert_eq!(system_to_text(&Some(json!(""))).unwrap(), Some(String::new()));
    }

    #[test]
    fn multi_blocks_concatenate_in_order() {
        // 連結順は配列順。区切り文字は挟まない (単一ブロック経路と同じ文字列になる)
        let v = json!([
            {"type": "text", "text": "aaa"},
            {"type": "text", "text": "bbb", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "ccc"},
        ]);
        assert_eq!(system_to_text(&Some(v)).unwrap(), Some("aaabbbccc".to_string()));
    }

    #[test]
    fn single_text_block_matches_string_form() {
        let blk = system_to_text(&Some(json!([{"type": "text", "text": "abc"}]))).unwrap();
        let s = system_to_text(&Some(json!("abc"))).unwrap();
        assert_eq!(blk, s);
    }

    #[test]
    fn empty_system_array_is_empty_string() {
        assert_eq!(
            system_to_text(&Some(json!([]))).unwrap(),
            Some(String::new())
        );
    }

    #[test]
    fn empty_or_whitespace_text_block_is_rejected() {
        for text in ["", " ", "\n\t "] {
            let m = err_msg(json!([{"type": "text", "text": text}]));
            assert!(m.contains("system[0].text"), "message = {m}");
            assert!(m.contains("non-empty"), "message = {m}");
        }
    }

    #[test]
    fn missing_or_non_string_text_is_rejected() {
        let m = err_msg(json!([{"type": "text"}]));
        assert!(m.contains("system[0].text: must be a string"), "message = {m}");
        let m = err_msg(json!([{"type": "text", "text": 5}]));
        assert!(m.contains("system[0].text: must be a string"), "message = {m}");
        let m = err_msg(json!([{"type": "text", "text": null}]));
        assert!(m.contains("system[0].text: must be a string"), "message = {m}");
    }

    #[test]
    fn multimodal_blocks_are_rejected() {
        let image = json!([{
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}
        }]);
        let m = err_msg(image);
        assert!(m.contains("system[0].type"), "message = {m}");
        assert!(m.contains("image"), "message = {m}");

        let doc = json!([{
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": "hi"}
        }]);
        let m = err_msg(doc);
        assert!(m.contains("system[0].type"), "message = {m}");
        assert!(m.contains("document"), "message = {m}");

        // 2 個目が image なら index が 1 になる
        let mixed = json!([
            {"type": "text", "text": "ok"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA"}},
        ]);
        let m = err_msg(mixed);
        assert!(m.contains("system[1].type"), "message = {m}");
    }

    #[test]
    fn missing_type_and_non_object_blocks_are_rejected() {
        let m = err_msg(json!([{"text": "hi"}]));
        assert!(m.contains("system[0].type"), "message = {m}");
        let m = err_msg(json!([{"type": 3, "text": "hi"}]));
        assert!(m.contains("system[0].type"), "message = {m}");
        let m = err_msg(json!(["hi"]));
        assert!(m.contains("system[0]: each block must be an object"), "message = {m}");
    }

    #[test]
    fn non_string_non_array_system_is_rejected() {
        let m = err_msg(json!(42));
        assert!(m.contains("system must be a string or an array"), "message = {m}");
        let m = err_msg(json!({"type": "text", "text": "hi"}));
        assert!(m.contains("system must be a string or an array"), "message = {m}");
    }
}
