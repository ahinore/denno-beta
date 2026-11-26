//! P841: Ollama 互換 API facade (`/api/tags` / `/api/version` / `/api/chat` /
//! `/api/generate`)。
//!
//! `/v1/messages`・`/v1/chat/completions` と **同じ** Scheduler・chat template・
//! KV prefix cache・engine を通す。違うのはワイヤ形式だけ:
//!
//! - 応答は SSE ではなく **NDJSON** (`application/x-ndjson`)。1 行 1 JSON で、
//!   最終行が `done: true`。
//! - リクエストの `options` は Ollama の名前 (`num_predict` / `stop` / ...) で受ける。
//! - エラー本文は `{"error": "..."}` (Ollama 形式)。
//!
//! ストリームは [`crate::scheduler::RawEvent`] を直接 [`Stream`] として包む
//! (中間タスク無し)。応答ボディが drop される = クライアント切断で `raw` 受信端も
//! 同時に落ちるため、切断検知と KV スロット解放は既存の raw 経路がそのまま効く。
//!
//! 既定値は `/v1/messages` と同じ (`max_tokens` 未指定 1024 / clamp 4096 /
//! temperature 上限 / thinking = brief + `enable_thinking=false`)。
//! `stream` の既定は Ollama 仕様どおり **true** (Anthropic/OpenAI 経路の false と違う)。
//!
//! `/api/chat` の履歴は system / user / assistant のみ (履歴の `tool` ロールは 400)。
//! `/api/generate` は `raw: true` のときだけ chat template を通さない
//! (`Scheduler::submit_completion`)。`raw: false` は system + user の chat 経路。
//!
//! task 1071 (互換の監査): 人気クライアント (Open WebUI / Continue / Cline) が
//! 常に送る任意フィールドを**落とさない**:
//!
//! - `keep_alive` (文字列 `"5m"` / 整数 / null / object) — 受理して無視する
//!   (モデルは常駐なので解放要求には意味を持たせない)。
//! - `format` (`"json"` / JSON Schema オブジェクト) — system に JSON 指示を足す。
//!   それ以外の値 (未知の文字列・数値) は無視する。
//! - `think` (bool / `"low"|"medium"|"high"` 等) — `false` 系は
//!   `enable_thinking=false` (従来既定)、`true` 系は template 既定に任せる。
//! - `tools` (OpenAI 形式の function 定義) — プロンプトへ渡し、出力の
//!   `<tool_call>` block を Ollama 形式の `message.tool_calls` に変換する。
//!   形が不正な要素は読み飛ばす (400 にしない)。
//! - メッセージ側の `images` / `tool_calls` / `thinking` 等の未知キーは
//!   serde が無視する (元から 400 にしない)。

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use denno_engine::{ChatMessage, ChatOutput, GenParams};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::Stream;

use crate::scheduler::RawEvent;
use crate::{
    apply_temperature_cap, brief_system, check_auth, debug_api_error, extract_tool_calls,
    resolve_max_tokens, validate_tools, ApiError, AppState, ToolDef,
};

/// `/api/version` が返す文字列。
pub const OLLAMA_VERSION: &str = "0.1.0-denno";

// ---------- エラー (Ollama 形式) ----------

/// Ollama 互換のエラー本文 `{"error": "..."}`。
#[derive(Debug)]
pub struct OllamaError {
    status: StatusCode,
    message: String,
    /// P873: `Retry-After` ヘッダ (秒)。None = 付けない。
    retry_after: Option<u64>,
}

impl OllamaError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self { status, message: message.into(), retry_after: None }
    }
    /// P873: `Retry-After` を付ける (過負荷の応答)。
    fn with_retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, message)
    }
    fn not_found(message: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, message)
    }
    fn server(message: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, message)
    }
}

impl IntoResponse for OllamaError {
    fn into_response(self) -> Response {
        let mut resp = (self.status, Json(json!({"error": self.message}))).into_response();
        // P873: 過負荷では再試行までの待ちをヘッダで示す。
        if let Some(secs) = self.retry_after {
            if let Ok(v) = axum::http::HeaderValue::from_str(&secs.to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}

/// `ApiError` (Anthropic 形式) → Ollama 形式。分類 (ステータス) はそのまま、
/// 本文の形だけを変える。
impl From<ApiError> for OllamaError {
    fn from(e: ApiError) -> Self {
        let retry_after = e.retry_after();
        let (status, _, message) = debug_api_error(e);
        let mut out = Self::new(
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            message,
        );
        out.retry_after = retry_after;
        out
    }
}

/// P873: キュー満杯 (待ち行列の上限超過)。Ollama は過負荷を 503 で返す慣例なので
/// そのまま 503 + `Retry-After`。
fn queue_full(state: &AppState) -> OllamaError {
    OllamaError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "server busy: request queue is full (max_queue = {}); retry later",
            state.scheduler.max_queue()
        ),
    )
    .with_retry_after(crate::QUEUE_RETRY_AFTER_SECS)
}

// ---------- 時刻 (Ollama の created_at は RFC3339) ----------

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Unix 秒 → `YYYY-MM-DDTHH:MM:SSZ` (UTC・秒精度)。
/// chrono を依存に足さないための小さな civil-from-days 変換。
fn rfc3339_from_unix(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let sod = secs % 86_400;
    let (h, mi, s) = (sod / 3600, (sod % 3600) / 60, sod % 60);
    // Howard Hinnant の civil_from_days
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `/api/tags` の `digest`。Ollama は重みの sha256 を返すが、ここではモデル名から
/// 作る決定論的な 64 hex の **表示用の識別子** (重みの内容のハッシュではない)。
fn digest_for(name: &str) -> String {
    let mut out = String::with_capacity(64);
    for round in 0..4u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.bytes().chain(round.to_le_bytes()) {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        out.push_str(&format!("{h:016x}"));
    }
    out
}

// ---------- リクエスト DTO ----------

/// Ollama の `options`。未知キー (`num_ctx` / `seed` 等) は無視する。
#[derive(Debug, Default, Deserialize)]
pub struct OllamaOptions {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    /// Ollama の生成上限。負値 (-1 = 無制限 / -2 = context 埋め) と 0 は
    /// 「未指定」として扱い、サーバー既定 (`DEFAULT_MAX_TOKENS`) を適用する。
    pub num_predict: Option<i64>,
    /// 停止列。文字列 or 文字列配列 (Ollama は配列が一般的)。
    pub stop: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaMessage {
    pub role: String,
    /// 文字列 or text part 配列 (Ollama の `content`)。画像等の他 part は無視。
    #[serde(default)]
    pub content: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    /// Ollama の既定は **true**。
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: OllamaOptions,
    /// task 1071: 受理して無視する (モデルは常駐。型は検証しない)。
    #[serde(default)]
    #[allow(dead_code)]
    pub keep_alive: Option<Value>,
    /// task 1071: `"json"` か JSON Schema オブジェクトなら JSON 指示を足す。
    #[serde(default)]
    pub format: Option<Value>,
    /// task 1071: bool または `"low"|"medium"|"high"` 等。
    #[serde(default)]
    pub think: Option<Value>,
    /// task 1071: OpenAI 形式の function 定義 (配列)。形が不正な要素は無視。
    #[serde(default)]
    pub tools: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    /// true なら chat template を通さない (生プロンプト)。
    #[serde(default)]
    pub raw: Option<bool>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: OllamaOptions,
    /// task 1071: 受理して無視する (型は検証しない)。
    #[serde(default)]
    #[allow(dead_code)]
    pub keep_alive: Option<Value>,
    /// task 1071: `"json"` か JSON Schema オブジェクトなら JSON 指示を足す。
    #[serde(default)]
    pub format: Option<Value>,
    /// task 1071: bool または効力文字列。
    #[serde(default)]
    pub think: Option<Value>,
}

/// content (文字列 or parts 配列) を 1 本のテキストにする。
fn content_to_text(v: &Option<Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Value::String(s) => out.push_str(s),
                    Value::Object(_) => {
                        if matches!(p.get("type").and_then(|t| t.as_str()), Some("text")) {
                            out.push_str(
                                p.get("text").and_then(|t| t.as_str()).unwrap_or_default(),
                            );
                        }
                    }
                    _ => {}
                }
            }
            out
        }
        Some(other) => other.to_string(),
    }
}

/// Ollama の `messages` を (system, chat_messages) へ写す。
fn split_chat_messages(
    msgs: &[OllamaMessage],
) -> Result<(Option<String>, Vec<ChatMessage>), OllamaError> {
    let mut system: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let text = content_to_text(&m.content);
        match m.role.as_str() {
            "system" => system.push(text),
            "user" => out.push(ChatMessage { role: "user".to_string(), content: text }),
            "assistant" => {
                out.push(ChatMessage { role: "assistant".to_string(), content: text })
            }
            other => {
                return Err(OllamaError::invalid(format!(
                    "messages[{i}].role: unsupported role {other:?} \
                     (expected system, user or assistant; \"tool\" role messages are not \
                      supported on /api/chat)"
                )))
            }
        }
    }
    let system = if system.is_empty() { None } else { Some(system.join("\n")) };
    Ok((system, out))
}

/// `options.stop` を正規化する (文字列 1 本 or 文字列配列)。
fn normalize_stop(v: &Option<Value>) -> Result<Vec<String>, OllamaError> {
    match v {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(if s.is_empty() { Vec::new() } else { vec![s.clone()] }),
        Some(Value::Array(a)) => {
            let mut out = Vec::with_capacity(a.len());
            for x in a {
                match x.as_str() {
                    Some(s) if !s.is_empty() => out.push(s.to_string()),
                    Some(_) => {}
                    None => {
                        return Err(OllamaError::invalid(
                            "options.stop: array elements must be strings",
                        ))
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err(OllamaError::invalid(
            "options.stop: expected a string or an array of strings",
        )),
    }
}

/// task 1071: `format` の解釈結果を system へ足す指示文にする。
///
/// - `"json"` → JSON オブジェクト限定の指示
/// - オブジェクト → JSON Schema として埋め込む (32 KiB 超は無視)
/// - それ以外 (未知の文字列・数値・null) → 何も足さない (400 にしない)
fn format_hint(v: &Option<Value>) -> Option<String> {
    /// 埋め込むスキーマの上限 (これを超える入力は無視して従来どおり)。
    const MAX_SCHEMA_CHARS: usize = 32 * 1024;
    match v {
        Some(Value::String(s)) if s.trim().eq_ignore_ascii_case("json") => {
            Some(JSON_OBJECT_HINT.to_string())
        }
        Some(Value::Object(map)) => {
            let schema = Value::Object(map.clone()).to_string();
            if schema.len() > MAX_SCHEMA_CHARS {
                return None;
            }
            Some(format!(
                "Return a single valid JSON value that conforms to this JSON Schema and nothing \
                 else (no markdown fences, no prose): {schema}"
            ))
        }
        _ => None,
    }
}

/// `format: "json"` のときに system へ足す固定文 (OpenAI の
/// `response_format: {"type":"json_object"}` と同じ文面)。
const JSON_OBJECT_HINT: &str =
    "Return a single valid JSON object and nothing else (no markdown fences, no prose).";

/// system に JSON 指示を足す (既存 system があれば改行 2 つで連結)。
fn append_hint(system: Option<String>, hint: Option<String>) -> Option<String> {
    match (system, hint) {
        (s, None) => s,
        (None, Some(h)) => Some(h),
        (Some(s), Some(h)) if s.trim().is_empty() => Some(h),
        (Some(s), Some(h)) => Some(format!("{s}\n\n{h}")),
    }
}

/// task 1071: `think` を template の `enable_thinking` へ写す。
///
/// - 未指定 / `false` / `"false"` / `"off"` / `"no"` / `"none"` → `Some(false)`
///   (従来既定 = 思考オフ)
/// - `true` / `"true"` / `"on"` / 効力文字列 (`"low"|"medium"|"high"`) → `None`
///   (template 既定に任せる = 思考オン)
/// - それ以外の型 (数値・object 等) → `Some(false)` (安全側 = 従来既定)
fn thinking_from(v: &Option<Value>) -> Option<bool> {
    match v {
        Some(Value::Bool(false)) => Some(false),
        Some(Value::Bool(true)) => None,
        Some(Value::String(s)) => match s.trim().to_ascii_lowercase().as_str() {
            "false" | "off" | "no" | "none" | "0" => Some(false),
            "true" | "on" | "yes" | "low" | "medium" | "high" | "1" => None,
            _ => Some(false),
        },
        _ => Some(false),
    }
}

/// task 1071: Ollama の `tools` (OpenAI 形式。`function` を直接持つ形も許す) を
/// [`ToolDef`] へ写す。形が不正な要素は読み飛ばす (400 にしない)。
fn ollama_tool_defs(v: &Option<Value>) -> Vec<ToolDef> {
    let Some(Value::Array(items)) = v else {
        return Vec::new();
    };
    items
        .iter()
        .filter_map(|it| {
            let f = it.get("function").unwrap_or(it);
            let name = f.get("name")?.as_str()?.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let description = f
                .get("description")
                .and_then(|d| d.as_str())
                .map(str::to_string);
            // `parameters` が無い / object でないツールは引数なしとして受ける
            // (`validate_tools` は object 以外を 400 にするのでここで正規化する)。
            let input_schema = match f.get("parameters") {
                Some(p) if p.is_object() => p.clone(),
                _ => json!({"type": "object", "properties": {}}),
            };
            Some(ToolDef { name, description, input_schema })
        })
        .collect()
}

/// モデル出力の `<tool_call>` block を Ollama 形式の `tool_calls` にする。
/// Ollama の `arguments` は **オブジェクト** (OpenAI の JSON 文字列とは違う)。
fn tool_calls_json(calls: &[crate::ExtractedToolCall]) -> Value {
    Value::Array(
        calls
            .iter()
            .map(|c| {
                json!({
                    "function": {
                        "name": c.name,
                        "arguments": c.input,
                    }
                })
            })
            .collect(),
    )
}

/// 実効 max_tokens。`num_predict` が正のときだけ採用し、`/v1/messages` と同じ
/// 既定 (1024) と clamp (4096) を通す。
fn max_tokens_from(num_predict: Option<i64>) -> usize {
    let n = match num_predict {
        Some(v) if v > 0 => Some(v as usize),
        _ => None,
    };
    resolve_max_tokens(n, None)
}

/// `options` から生成パラメータを作る (既定値は既存ハンドラと同一)。
fn build_params(state: &AppState, opts: &OllamaOptions, stop: Vec<String>) -> GenParams {
    GenParams {
        max_tokens: max_tokens_from(opts.num_predict),
        temperature: apply_temperature_cap(opts.temperature, state.temperature_cap),
        top_p: opts.top_p,
        top_k: None,
        stop_sequences: stop,
        seed: None,
        repetition_penalty: state.default_repetition_penalty,
        frequency_penalty: None,
        presence_penalty: None,
        no_repeat_ngram_size: None,
    }
}

/// モデル名の照合。空 (未指定) は既定モデルとして受理する。
fn check_model(state: &AppState, model: &str) -> Result<(), OllamaError> {
    if !model.is_empty() && model != state.model_name {
        return Err(OllamaError::not_found(format!(
            "model '{model}' not found, try pulling it first"
        )));
    }
    Ok(())
}

/// engine の stop_reason → Ollama の `done_reason`。
fn done_reason(engine_reason: &str) -> &'static str {
    match engine_reason {
        "max_tokens" => "length",
        _ => "stop",
    }
}

// ---------- NDJSON ストリーム ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chat,
    Generate,
}

fn ndjson_line(v: Value) -> Bytes {
    Bytes::from(format!("{v}\n").into_bytes())
}

/// [`RawEvent`] を 1 行 1 JSON (NDJSON) へ変換する [`Stream`]。
///
/// 中間タスクを持たないので、応答ボディが drop されると `raw` 受信端も同時に
/// 落ち、scheduler の送信が失敗して Run が打ち切られる (切断 = abort)。
struct NdjsonStream {
    raw: tokio::sync::mpsc::Receiver<RawEvent>,
    kind: Kind,
    model: String,
    created_at: String,
    started: Instant,
    input_tokens: usize,
    /// 最終行 (`done: true` か error) を出したか。出したら次で None。
    finished: bool,
    pending: VecDeque<Bytes>,
    /// task 1071: tools 有効時は本文をバッファし、Done で 1 行にまとめる
    /// (途中の chunk に `<tool_call>` block の断片を漏らさないため)。
    tools_active: bool,
    buffered: String,
}

impl NdjsonStream {
    fn chunk_line(&self, text: &str) -> Bytes {
        let v = match self.kind {
            Kind::Chat => json!({
                "model": self.model,
                "created_at": self.created_at,
                "message": {"role": "assistant", "content": text},
                "done": false,
            }),
            Kind::Generate => json!({
                "model": self.model,
                "created_at": self.created_at,
                "response": text,
                "done": false,
            }),
        };
        ndjson_line(v)
    }

    /// task 1071: tools 有効時の 1 行 (本文 + `message.tool_calls`)。
    /// Ollama は `arguments` をオブジェクトで返す。
    fn tool_chunk_line(&self, text: &str, calls: &[crate::ExtractedToolCall]) -> Bytes {
        let mut msg = json!({"role": "assistant", "content": text});
        if !calls.is_empty() {
            msg["tool_calls"] = tool_calls_json(calls);
        }
        ndjson_line(json!({
            "model": self.model,
            "created_at": self.created_at,
            "message": msg,
            "done": false,
        }))
    }

    fn final_line(&self, input_tokens: usize, output_tokens: usize, stop_reason: &str) -> Bytes {
        // 時間は ns (Ollama 仕様)。prefill と decode の内訳は engine から
        // 取れないので prompt_eval_duration / load_duration は 0、
        // eval_duration は総時間で近似する (0 除算を避けるため)。
        let total = self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let eval_duration = total;
        let mut v = match self.kind {
            Kind::Chat => json!({
                "model": self.model,
                "created_at": self.created_at,
                "message": {"role": "assistant", "content": ""},
                "done": true,
            }),
            Kind::Generate => json!({
                "model": self.model,
                "created_at": self.created_at,
                "response": "",
                "done": true,
            }),
        };
        v["done_reason"] = json!(done_reason(stop_reason));
        v["total_duration"] = json!(total);
        v["load_duration"] = json!(0);
        v["prompt_eval_count"] = json!(input_tokens);
        v["prompt_eval_duration"] = json!(0);
        v["eval_count"] = json!(output_tokens);
        v["eval_duration"] = json!(eval_duration);
        ndjson_line(v)
    }
}

impl Stream for NdjsonStream {
    type Item = Result<Bytes, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if let Some(b) = this.pending.pop_front() {
                return Poll::Ready(Some(Ok(b)));
            }
            if this.finished {
                return Poll::Ready(None);
            }
            match this.raw.poll_recv(cx) {
                Poll::Pending => return Poll::Pending,
                // Done を待たずに閉じた (abort) 場合はそのまま終わる。
                Poll::Ready(None) => {
                    this.finished = true;
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(ev)) => match ev {
                    RawEvent::Start { input_tokens } => this.input_tokens = input_tokens,
                    RawEvent::Delta(text) => {
                        if !text.is_empty() {
                            if this.tools_active {
                                this.buffered.push_str(&text);
                            } else {
                                let line = this.chunk_line(&text);
                                this.pending.push_back(line);
                            }
                        }
                    }
                    RawEvent::Done { input_tokens, output_tokens, stop_reason, .. } => {
                        this.input_tokens = input_tokens;
                        if this.tools_active {
                            let (text, calls) = extract_tool_calls(&this.buffered);
                            if !text.is_empty() || !calls.is_empty() {
                                let line = this.tool_chunk_line(&text, &calls);
                                this.pending.push_back(line);
                            }
                        }
                        let line = this.final_line(input_tokens, output_tokens, &stop_reason);
                        this.pending.push_back(line);
                        this.finished = true;
                    }
                    // P869: ollama 経路は logprobs を要求しない (常に None)。
                    RawEvent::Logprob(_) => {}
                    RawEvent::Error(msg) => {
                        this.pending.push_back(ndjson_line(json!({"error": msg})));
                        this.finished = true;
                    }
                },
            }
        }
    }
}

/// 非ストリーム応答の共通末尾 (`done: true` 以下のフィールド)。
fn done_fields(started: Instant, input_tokens: usize, output_tokens: usize, reason: &str) -> Value {
    let total = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    json!({
        "done": true,
        "done_reason": done_reason(reason),
        "total_duration": total,
        "load_duration": 0,
        "prompt_eval_count": input_tokens,
        "prompt_eval_duration": 0,
        "eval_count": output_tokens,
        "eval_duration": total,
    })
}

// ---------- ハンドラ: /api/tags, /api/version ----------

/// `/api/tags`・`/api/show`・`/api/ps` が共有する `details` ブロック。
/// `tags` の出力と同一 (`quantization_level` だけ engine の量子化ラベル)。
fn model_details(state: &AppState) -> Value {
    json!({
        "parent_model": "",
        "format": "gguf",
        "family": "denno",
        "families": ["denno"],
        "parameter_size": "",
        "quantization_level": state.info.quant,
    })
}

pub async fn tags(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, OllamaError> {
    check_auth(&state, &headers).map_err(OllamaError::from)?;
    let modified_at = rfc3339_from_unix(now_secs());
    Ok(Json(json!({
        "models": [{
            "name": state.model_name,
            "model": state.model_name,
            "modified_at": modified_at,
            "size": 0,
            "digest": digest_for(&state.model_name),
            "details": model_details(&state),
        }]
    })))
}


pub async fn version(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, OllamaError> {
    check_auth(&state, &headers).map_err(OllamaError::from)?;
    Ok(Json(json!({"version": OLLAMA_VERSION})))
}

// ---------- ハンドラ: /api/show, /api/ps (task 996) ----------

/// `/api/show` のリクエスト。現行の Ollama は `model`、古いクライアントは
/// `name` を送るので両方受ける (両方あるときは `model` を優先)。
#[derive(Debug, Default, Deserialize)]
pub struct ShowRequest {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
}

impl ShowRequest {
    /// 照合に使うモデル名。空 / 未指定は `None`。
    fn requested_model(&self) -> Option<&str> {
        self.model
            .as_deref()
            .or(self.name.as_deref())
            .map(str::trim)
            .filter(|s| !s.is_empty())
    }
}

/// `POST /api/show`: モデルのメタデータ。重みは読まずに [`AppState::info`]
/// (起動時に 1 回取った不変情報) だけを返すので、生成中でも lock を取らない。
///
/// Ollama と同じく、未指定は 400、未知のモデル名は 404 (`{"error": ...}`)。
/// `parameters` / `template` は空: denno はモデルごとのパラメータ上書きも
/// 露出した chat template も持たない (テンプレートは engine 内で固定)。
pub async fn show(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<Value>, OllamaError> {
    check_auth(&state, &headers).map_err(OllamaError::from)?;
    let req: ShowRequest = if body.is_empty() {
        ShowRequest::default()
    } else {
        serde_json::from_slice(&body)
            .map_err(|e| OllamaError::invalid(format!("invalid request body: {e}")))?
    };
    let Some(model) = req.requested_model() else {
        return Err(OllamaError::invalid("model: required"));
    };
    check_model(&state, model)?;

    let info = &state.info;
    Ok(Json(json!({
        "modelfile": format!("# denno model {}\n", state.model_name),
        "parameters": "",
        "template": "",
        "details": model_details(&state),
        "model_info": {
            "general.architecture": "denno",
            "general.name": state.model_name,
            "denno.context_length": info.context,
            "denno.block_count": info.layers,
            "denno.kv_slot_count": info.slots,
            "denno.quantization": info.quant,
        },
        "capabilities": ["completion"],
        "modified_at": rfc3339_from_unix(now_secs()),
    })))
}

/// `GET /api/ps`: 常駐中のモデル一覧。denno は起動時にロードした 1 本を
/// プロセス寿命まで保持する (アンロードしない) ので、常にその 1 本を返す。
/// `expires_at` はその性質を表す無期限側の値 (100 年後) で、実際の
/// アンロード時刻ではない。`size` / `size_vram` は engine が重みの実バイト数を
/// 公開していないため 0 (嘘の値を出さない)。
pub async fn ps(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, OllamaError> {
    check_auth(&state, &headers).map_err(OllamaError::from)?;
    // 100 年 (秒)。u64 の範囲内で桁あふれしない。
    const KEEP_FOREVER_SECS: u64 = 100 * 365 * 24 * 60 * 60;
    let expires_at = rfc3339_from_unix(now_secs().saturating_add(KEEP_FOREVER_SECS));
    Ok(Json(json!({
        "models": [{
            "name": state.model_name,
            "model": state.model_name,
            "size": 0,
            "size_vram": 0,
            "digest": digest_for(&state.model_name),
            "details": model_details(&state),
            "expires_at": expires_at,
        }]
    })))
}

/// task 996: 照会系 (`/api/show` / `/api/ps`) のサブルータ。lib.rs 側は
/// `.merge(ollama::admin_router())` の 1 行だけなので、
/// `tests/openapi_contract.rs` の「router() の `.route(...)` 行 = 仕様」という
/// 逆方向検査 (docs/openapi.yaml は今回のスコープ外) に影響しない。
pub fn admin_router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/show", post(show))
        .route("/api/ps", get(ps))
}

// ---------- 共通の待ち合わせ ----------

async fn await_output(
    rx: tokio::sync::oneshot::Receiver<anyhow::Result<ChatOutput>>,
    limits: crate::Limits,
) -> Result<ChatOutput, OllamaError> {
    // P873: `--request-timeout-secs` が有効ならその少し後まで待つ
    // (スケジューラ側の期限が先に発火して理由を返す)。
    let wait = crate::response_wait_timeout(limits);
    match tokio::time::timeout(wait, rx).await {
        Ok(Ok(Ok(out))) => Ok(out),
        Ok(Ok(Err(e))) => Err(OllamaError::from(crate::error_from_anyhow(e))),
        Ok(Err(e)) => Err(OllamaError::server(format!("scheduler dropped: {e}"))),
        Err(_) => Err(OllamaError::from(ApiError::gateway_timeout(format!(
            "timeout: no response from the engine within {:.1}s; \
             the request was abandoned (retry later)",
            wait.as_secs_f64()
        )))),
    }
}

/// NDJSON 応答を組み立てる。`raw_prompt` が Some なら `/api/generate` の raw 経路。
#[allow(clippy::too_many_arguments)]
fn ndjson_response(
    state: &AppState,
    kind: Kind,
    system: Option<String>,
    messages: Vec<ChatMessage>,
    raw_prompt: Option<String>,
    params: GenParams,
    enable_thinking: Option<bool>,
    tools: Option<Value>,
) -> Result<Response, OllamaError> {
    let (raw_tx, raw_rx) = tokio::sync::mpsc::channel::<RawEvent>(64);
    match raw_prompt {
        Some(p) => state
            .scheduler
            .submit_completion_stream(p, params, raw_tx)
            .map_err(|_| queue_full(state))?,
        None => state
            .scheduler
            .submit_openai_stream(
                system,
                messages,
                params,
                tools.clone(),
                None,
                enable_thinking,
                raw_tx,
            )
            .map_err(|_| queue_full(state))?,
    }
    let stream = NdjsonStream {
        raw: raw_rx,
        kind,
        model: state.model_name.clone(),
        created_at: rfc3339_from_unix(now_secs()),
        started: Instant::now(),
        input_tokens: 0,
        finished: false,
        pending: VecDeque::new(),
        tools_active: tools.is_some(),
        buffered: String::new(),
    };
    Ok((
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(stream),
    )
        .into_response())
}

// ---------- ハンドラ: /api/chat ----------

pub async fn chat(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, OllamaError> {
    check_auth(&state, &headers).map_err(OllamaError::from)?;
    crate::reject_if_draining(&state).map_err(OllamaError::from)?;
    let started = Instant::now();
    let req: ChatRequest = serde_json::from_slice(&body)
        .map_err(|e| OllamaError::invalid(format!("invalid request body: {e}")))?;

    check_model(&state, &req.model)?;
    if req.messages.is_empty() {
        return Err(OllamaError::invalid(
            "messages: at least one message is required",
        ));
    }
    let (system, chat_messages) = split_chat_messages(&req.messages)?;
    if chat_messages.is_empty() {
        return Err(OllamaError::invalid(
            "messages: at least one user or assistant message is required",
        ));
    }
    let stop = normalize_stop(&req.options.stop)?;
    let params = build_params(&state, &req.options, stop);
    // task 1071: tools (OpenAI 形式) を template 用の JSON へ。形が不正な要素は
    // 読み飛ばすので 400 にはならない。
    let tool_defs = ollama_tool_defs(&req.tools);
    let tools_active = !tool_defs.is_empty();
    let tools_for_prompt = if tools_active {
        Some(validate_tools(&tool_defs).map_err(OllamaError::from)?)
    } else {
        None
    };
    // /v1/messages と同じ thinking 既定 (brief + `enable_thinking` は think から)。
    // task 1071: `format` の JSON 指示も system に足す。
    let system = brief_system(system);
    let system = append_hint(system, format_hint(&req.format));
    let enable_thinking = thinking_from(&req.think);

    if req.stream.unwrap_or(true) {
        return ndjson_response(
            &state,
            Kind::Chat,
            system,
            chat_messages,
            None,
            params,
            enable_thinking,
            tools_for_prompt,
        );
    }

    let rx = state
        .scheduler
        .submit_chat(
            system,
            chat_messages,
            params,
            tools_for_prompt,
            None,
            enable_thinking,
        )
        .map_err(|_| queue_full(&state))?;
    let out = await_output(rx, state.scheduler.limits()).await?;
    // task 1071: tools 有効時だけ `<tool_call>` block を Ollama 形式へ変換する。
    let (text, calls) = if tools_active {
        extract_tool_calls(&out.text)
    } else {
        (out.text.clone(), Vec::new())
    };
    let mut message = json!({"role": "assistant", "content": text});
    if !calls.is_empty() {
        message["tool_calls"] = tool_calls_json(&calls);
    }
    let mut v = json!({
        "model": state.model_name,
        "created_at": rfc3339_from_unix(now_secs()),
        "message": message,
    });
    if let Value::Object(ref mut m) = v {
        if let Value::Object(fields) = done_fields(
            started,
            out.input_tokens,
            out.output_tokens,
            out.stop_reason.as_str(),
        ) {
            m.extend(fields);
        }
    }
    Ok(Json(v).into_response())
}

// ---------- ハンドラ: /api/generate ----------

pub async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, OllamaError> {
    check_auth(&state, &headers).map_err(OllamaError::from)?;
    crate::reject_if_draining(&state).map_err(OllamaError::from)?;
    let started = Instant::now();
    let req: GenerateRequest = serde_json::from_slice(&body)
        .map_err(|e| OllamaError::invalid(format!("invalid request body: {e}")))?;

    check_model(&state, &req.model)?;
    let prompt = req.prompt.unwrap_or_default();
    if prompt.is_empty() {
        return Err(OllamaError::invalid(
            "prompt: must be a non-empty string",
        ));
    }
    let stop = normalize_stop(&req.options.stop)?;
    let params = build_params(&state, &req.options, stop);
    let raw = req.raw.unwrap_or(false);
    let stream = req.stream.unwrap_or(true);
    // task 1071: `format` の JSON 指示と `think` の写像 (chat と同じ規則)。
    let hint = format_hint(&req.format);
    let enable_thinking = thinking_from(&req.think);

    if raw {
        // chat template を通さない (`/v1/completions` と同じ raw 経路)。system は無視する。
        if stream {
            return ndjson_response(
                &state,
                Kind::Generate,
                None,
                Vec::new(),
                Some(prompt),
                params,
                None,
                None,
            );
        }
        let rx = state
            .scheduler
            .submit_completion(prompt, params)
            .map_err(|_| queue_full(&state))?;
        let out = await_output(rx, state.scheduler.limits()).await?;
        return Ok(Json(generate_json(&state, started, &out)).into_response());
    }

    // 非 raw: system + user の chat 経路 (chat template を通す)。
    let system = brief_system(req.system.filter(|s| !s.is_empty()));
    let system = append_hint(system, hint);
    let messages = vec![ChatMessage { role: "user".to_string(), content: prompt }];
    if stream {
        return ndjson_response(
            &state,
            Kind::Generate,
            system,
            messages,
            None,
            params,
            enable_thinking,
            None,
        );
    }
    let rx = state
        .scheduler
        .submit_chat(system, messages, params, None, None, enable_thinking)
        .map_err(|_| queue_full(&state))?;
    let out = await_output(rx, state.scheduler.limits()).await?;
    Ok(Json(generate_json(&state, started, &out)).into_response())
}

fn generate_json(state: &AppState, started: Instant, out: &ChatOutput) -> Value {
    let mut v = json!({
        "model": state.model_name,
        "created_at": rfc3339_from_unix(now_secs()),
        "response": out.text,
    });
    if let Value::Object(ref mut m) = v {
        if let Value::Object(fields) = done_fields(
            started,
            out.input_tokens,
            out.output_tokens,
            out.stop_reason.as_str(),
        ) {
            m.extend(fields);
        }
    }
    v
}

#[cfg(test)]
mod compat_1071_tests {
    use super::*;

    fn s(v: Value) -> Option<Value> {
        Some(v)
    }

    #[test]
    fn format_hint_accepts_json_and_schema() {
        assert_eq!(format_hint(&s(json!("json"))), Some(JSON_OBJECT_HINT.to_string()));
        assert_eq!(format_hint(&s(json!(" JSON "))), Some(JSON_OBJECT_HINT.to_string()));
        let schema = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let h = format_hint(&s(schema.clone())).expect("schema が無視された");
        assert!(h.contains("\"properties\""), "hint = {h}");
        assert!(h.contains("\"a\""), "hint = {h}");
        // 未知の値は何も足さない (400 にもしない)
        assert_eq!(format_hint(&s(json!("xml"))), None);
        assert_eq!(format_hint(&s(json!(42))), None);
        assert_eq!(format_hint(&s(Value::Null)), None);
        assert_eq!(format_hint(&None), None);
    }

    #[test]
    fn format_hint_ignores_oversized_schema() {
        let big = "x".repeat(40 * 1024);
        assert_eq!(format_hint(&s(json!({"const": big}))), None);
    }

    #[test]
    fn thinking_from_maps_bool_and_levels() {
        assert_eq!(thinking_from(&None), Some(false));
        assert_eq!(thinking_from(&s(json!(false))), Some(false));
        assert_eq!(thinking_from(&s(json!(true))), None);
        for on in ["true", "on", "high", "medium", "low"] {
            assert_eq!(thinking_from(&s(json!(on))), None, "think = {on}");
        }
        for off in ["false", "off", "no", "none"] {
            assert_eq!(thinking_from(&s(json!(off))), Some(false), "think = {off}");
        }
        // 未知 / 非文字列は安全側 (従来既定)
        assert_eq!(thinking_from(&s(json!("bogus"))), Some(false));
        assert_eq!(thinking_from(&s(json!({"x": 1}))), Some(false));
    }

    #[test]
    fn append_hint_joins_system() {
        assert_eq!(append_hint(None, None), None);
        assert_eq!(append_hint(Some("a".into()), None), Some("a".to_string()));
        assert_eq!(append_hint(None, Some("h".into())), Some("h".to_string()));
        assert_eq!(
            append_hint(Some("a".into()), Some("h".into())),
            Some("a\n\nh".to_string())
        );
        assert_eq!(append_hint(Some("  ".into()), Some("h".into())), Some("h".to_string()));
    }

    #[test]
    fn ollama_tool_defs_skips_malformed_entries() {
        let v = json!([
            {"type": "function", "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }},
            {"function": {"name": "no_params"}},
            {"function": {}},
            {"function": {"name": "   "}},
            {"type": "function"},
            "not an object",
            42
        ]);
        let defs = ollama_tool_defs(&s(v));
        assert_eq!(defs.len(), 2, "defs = {defs:?}");
        assert_eq!(defs[0].name, "get_weather");
        assert_eq!(defs[0].description.as_deref(), Some("Get current weather"));
        assert!(defs[0].input_schema.is_object());
        assert_eq!(defs[1].name, "no_params");
        // parameters 欠落は「引数なし」の object へ正規化される
        assert_eq!(defs[1].input_schema, json!({"type": "object", "properties": {}}));
        // 正規化後は validate_tools が必ず通る (400 の口を作らない)
        validate_tools(&defs).expect("validate_tools が落ちた");
        assert!(ollama_tool_defs(&None).is_empty());
        assert!(ollama_tool_defs(&s(json!({"name": "x"}))).is_empty());
        assert!(ollama_tool_defs(&s(json!("nope"))).is_empty());
    }

    #[test]
    fn tool_calls_json_uses_object_arguments() {
        let calls = vec![crate::ExtractedToolCall {
            name: "get_weather".to_string(),
            input: json!({"city": "Tokyo"}),
        }];
        assert_eq!(
            tool_calls_json(&calls),
            json!([{"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}])
        );
        // Ollama の arguments はオブジェクト (OpenAI の JSON 文字列ではない)
        assert!(tool_calls_json(&calls)[0]["function"]["arguments"].is_object());
    }
}
