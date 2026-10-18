//! Anthropic Messages API 互換 HTTP サーバー。
//! POST /v1/messages (非ストリーミング + SSE)、GET /v1/models、GET /health。

use anyhow::Result;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use denno_engine::{ChatMessage, ChatOutput, Engine, GenParams, QuantMode, ToolConstraint};
use denno_telemetry::{self as telem, Frame};
use rand::Rng;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::sync::OnceLock;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

mod logging;
mod scheduler;
mod stream_asm;
pub use logging::{init_serve_logging, NonBlockingStdout};
pub use scheduler::{Job, MAX_ACTIVE_SESSIONS, QueueFull, Scheduler};

pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    pub api_key: Option<String>,
    /// P4: リクエストスケジューラ (有界キュー + 単一ワーカー)
    pub scheduler: Scheduler,
    /// モデル名 (不変)。ハンドラが engine.lock() を取らずに済むようにキャッシュ。
    /// (ワーカーは mutex をステップ間でほぼ連続保持するため、ハンドラが
    ///  generation 中に lock を奪おうとすると starve し得る)
    pub model_name: String,
    /// 実行中のモデル ID (例: Qwen/Qwen2.5-0.5B-Instruct)。テレメトリ ServerInfo 用。
    pub model_id: String,
}

/// serve 設定
pub struct ServeConfig {
    pub model_dir: std::path::PathBuf,
    pub name: String,
    /// モデル ID (pull 元の repo id)。テレメトリで報告する
    pub model_id: String,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub default_seed: Option<u64>,
    /// マルチエントリ prefix cache のトークン上限 (None なら env/デフォルトのまま)
    pub prefix_cache_tokens: Option<usize>,
    /// 重みの量子化モード (P3。デフォルト fp32。int8 は CLI でのみ有効化)
    pub quant: QuantMode,
    /// P4: リクエストキューの上限 (実行中 1 件の他に待ち行列に入れられる数)。
    /// 同時受け付けがこれを超えると 503 (overloaded_error) を返す。
    /// キューに入ったリクエストは実行開始まで待つ (サーバー側の待ち時間
    /// タイムアウトは無し)。
    pub max_queue: usize,
}

// ---------- Anthropic 形式エラー ----------

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into() }
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
    fn api(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, "api_error", message)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({
            "type": "error",
            "error": {"type": self.kind, "message": self.message}
        });
        (self.status, Json(body)).into_response()
    }
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
fn tool_use_to_xml(name: &str, input: &serde_json::Value) -> String {
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

/// system: 文字列 or [{type:"text",...}]
fn system_to_text(v: &Option<serde_json::Value>) -> Result<Option<String>, ApiError> {
    match v {
        None => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s.clone())),
        Some(serde_json::Value::Array(blocks)) => {
            let mut out = String::new();
            for b in blocks {
                if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                    out.push_str(b.get("text").and_then(|t| t.as_str()).unwrap_or_default());
                }
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
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/messages", post(messages))
        // P13: 推論テレメトリの WebSocket ストリーム (denno-trace)。
        // Anthropic 互換 API とは別系統で、互換性には影響しない。
        .route("/v1/telemetry/ws", get(telemetry_ws))
        .layer(axum::middleware::from_fn(log_requests))
        .with_state(state)
}

async fn log_requests(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    tracing::info!("{} {}", req.method(), req.uri().path());
    let resp = next.run(req).await;
    tracing::info!("-> {}", resp.status());
    resp
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
    tracing::info!(
        "model loaded: {} (context = {}, engine = denno)",
        engine.name,
        ctx
    );

    let model_name = engine.name.clone();
    let model_id = cfg.model_id.clone();
    let engine = Arc::new(Mutex::new(engine));
    let state = Arc::new(AppState {
        engine: engine.clone(),
        api_key: cfg.api_key,
        scheduler: Scheduler::spawn(engine, cfg.max_queue.max(1)),
        model_name,
        model_id,
    });

    // P13: テレメトリが有効ならハブ → broadcast への転送タスクを起動する。
    // (OFF のままならこのタスクは存在せず、捕捉コストもゼロ)
    if telem::enabled() {
        start_telemetry_relay();
    }

    let app = router(state);
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

// ---------- ハンドラ ----------

async fn health() -> Json<serde_json::Value> {
    Json(json!({"status": "ok"}))
}

async fn models(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    check_auth(&state, &headers)?;
    let (name, ctx) = {
        let g = state.engine.lock().unwrap();
        (g.name.clone(), g.context())
    };
    Ok(Json(json!({
        "data": [{
            "id": name,
            "type": "model",
            "display_name": name,
            "context_length": ctx,
        }],
        "has_more": false,
    })))
}

fn check_auth(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &state.api_key else { return Ok(()) };
    let provided = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .or_else(|| {
            headers
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.strip_prefix("Bearer "))
        });
    match provided {
        Some(p) if p == key => Ok(()),
        _ => Err(ApiError::auth()),
    }
}

async fn messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, ApiError> {
    check_auth(&state, &headers)?;

    let req: MessagesRequest = serde_json::from_slice(&body).map_err(|e| {
        ApiError::invalid_request(format!("invalid request body: {}", e))
    })?;

    if req.model != state.model_name {
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
    let mut chat_messages = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        if m.role != "user" && m.role != "assistant" {
            return Err(ApiError::invalid_request(format!(
                "messages[].role: must be \"user\" or \"assistant\", got {:?}",
                m.role
            )));
        }
        chat_messages.push(ChatMessage {
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
    let anti = parse_anti_repetition(&req)?;
    let params = GenParams {
        max_tokens: resolve_max_tokens(req.max_tokens, req.max_completion_tokens),
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop_sequences: req.stop_sequences.clone(),
        seed: None,
        repetition_penalty: req.repetition_penalty,
        frequency_penalty: anti.frequency,
        presence_penalty: anti.presence,
        no_repeat_ngram_size: anti.no_repeat_ngram_size,
    };

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
        let mut output = rx
            .await
            .map_err(|e| anyhow::anyhow!("scheduler dropped: {e}"))
            .and_then(|r| r)
            .map_err(|e| {
                let msg = format!("{e:#}");
                if msg.contains("invalid_request_error") {
                    ApiError::invalid_request(msg)
                } else {
                    ApiError::api(msg)
                }
            })?;
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

/// キュー満杯時の 503 エラー (Anthropic 流 overloaded_error)
fn queue_full_error(state: &AppState) -> impl Fn(QueueFull) -> ApiError + '_ {
    move |_| {
        ApiError::overloaded(format!(
            "server busy: request queue is full (max_queue = {}, in_flight = 1); retry later",
            state.scheduler.max_queue()
        ))
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
        }
    })
}

pub(crate) fn new_msg_id() -> String {
    let mut rng = rand::thread_rng();
    let id: String = (0..24).map(|_| format!("{:x}", rng.gen::<u8>() % 16)).collect();
    format!("msg_{id}")
}

pub(crate) fn to_event(v: &serde_json::Value) -> Event {
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
    let (model, model_id, quant, slots, context, layers) = {
        let g = state.engine.lock().unwrap();
        (
            g.name.clone(),
            state.model_id.clone(),
            g.quant.label().to_string(),
            g.caches.len() as u32,
            g.context() as u32,
            g.caches.first().map(|c| c.n_layers()).unwrap_or(0) as u32,
        )
    };
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
