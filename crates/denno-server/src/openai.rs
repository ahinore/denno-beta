//! P793/P797: OpenAI Chat Completions 互換エンドポイント (`POST /v1/chat/completions`)。
//!
//! `/v1/messages` と **同じ** Scheduler・chat template・KV prefix cache・engine を
//! 通す。違うのはワイヤ形式だけ:
//!
//! - リクエスト: OpenAI の `messages` / `stop` / `stream_options` / `tools` /
//!   `tool_choice` / `response_format`
//! - 応答: `chat.completion` (非ストリーム) と `chat.completion.chunk` (SSE)
//!
//! 既存の Anthropic 経路 (`/v1/messages`) には一切触らない。ストリームは
//! scheduler の [`crate::scheduler::RawEvent`] (生 delta) を HTTP 側で
//! chunk へ組み直す。
//!
//! P797 (tool calling): ツール定義は engine の [`crate::ToolDef`] へ、履歴の
//! assistant `tool_calls` / `role="tool"` は `/v1/messages` と同一の
//! `<tool_call>` / `<tool_response>` テキストへ写す。モデル出力からの抽出は
//! 既存の [`crate::extract_tool_calls`]、ストリームの block 境界検出は既存の
//! [`crate::stream_asm::StreamAssembler`] をそのまま使い、そのイベント列を
//! OpenAI の `delta.tool_calls` へ翻訳する (抽出/組立の意味論を二重実装しない)。
//!
//! P837: 追加 2 経路。
//! - `response_format: {"type":"json_schema", ...}` (Structured Outputs):
//!   スキーマを検証し (厳格モードの制約は [`crate::json_schema`])、スキーマ本文を
//!   system へ埋め込む。トークン拘束デコードは行わない (docs/api.md に明記)。
//! - `POST /v1/completions`: chat template を通さない raw prompt の続き生成
//!   (scheduler は [`crate::scheduler::Job::Chat`] / `OpenaiStream` の
//!   `raw_prompt` 経路をそのまま使う)。
//!
//! 対応フィールド (docs/api.md §POST /v1/chat/completions):
//! `model` / `messages` (system・developer・user・assistant (+ `tool_calls`)・
//! tool・function、content は文字列 or text part 配列) / `max_tokens` /
//! `max_completion_tokens` / `temperature` / `top_p` / `stop` (文字列 or 配列) /
//! `stream` / `stream_options.include_usage` / `tools` / `tool_choice` /
//! `response_format` (`text` | `json_object` | `json_schema`)。未知フィールドは無視する
//! (serde 既定)。

use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use denno_engine::{ChatMessage, GenParams, TokenLogprob, ToolConstraint};
use rand::Rng;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;

use crate::scheduler::RawEvent;
use crate::stream_asm::StreamAssembler;
use crate::{
    apply_temperature_cap, brief_system, check_auth, debug_api_error, extract_tool_calls,
    parse_tool_choice, resolve_max_tokens, tool_use_to_xml, validate_tools, ApiError, AppState,
    ExtractedToolCall, ToolChoice, ToolDef,
};

// ---------- エラー (OpenAI 形式) ----------

/// OpenAI 互換のエラー本文。`{"error": {"message", "type", "param", "code"}}`。
#[derive(Debug)]
pub struct OaError {
    status: StatusCode,
    kind: &'static str,
    message: String,
    param: Option<String>,
    /// P873: `Retry-After` ヘッダ (秒)。None = 付けない。
    retry_after: Option<u64>,
}

impl OaError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), param: None, retry_after: None }
    }
    /// P873: `Retry-After` を付ける (過負荷 / レート制限の応答)。
    fn with_retry_after(mut self, secs: u64) -> Self {
        self.retry_after = Some(secs);
        self
    }
    fn invalid(message: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "invalid_request_error", message)
    }
    fn with_param(mut self, p: impl Into<String>) -> Self {
        self.param = Some(p.into());
        self
    }
}

impl IntoResponse for OaError {
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "param": self.param,
                "code": Value::Null,
            }
        });
        let mut resp = (self.status, Json(body)).into_response();
        // P873: 過負荷 / レート制限では再試行までの待ちをヘッダで示す。
        if let Some(secs) = self.retry_after {
            if let Ok(v) = axum::http::HeaderValue::from_str(&secs.to_string()) {
                resp.headers_mut().insert(axum::http::header::RETRY_AFTER, v);
            }
        }
        resp
    }
}

/// `ApiError` (Anthropic 形式) → OpenAI 形式。中身は [`debug_api_error`] で読む
/// (値は既存の分類のままで、この変換は表示形式だけを変える)。
impl From<ApiError> for OaError {
    fn from(e: ApiError) -> Self {
        let retry_after = e.retry_after();
        let (status, kind, message) = debug_api_error(e);
        // Anthropic の type 名を OpenAI の type 名へ寄せる (未知はそのまま)
        let kind = match kind {
            "not_found_error" => "invalid_request_error",
            "overloaded_error" => "server_error",
            "api_error" => "server_error",
            other => other,
        };
        let mut out = Self::new(
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            kind,
            message,
        );
        out.retry_after = retry_after;
        out
    }
}

// ---------- モデルカタログ (task 901) ----------
//
// `/v1/models` の 1 件分のメタデータと、`/v1/models/:id` / 各生成経路の
// 別名解決をここに集約する。別名は「同じ重みを指す別の名前」なので、
// 生成経路では canonical 名と区別しない (応答の `model` は canonical を返す)。

/// 環境変数 `DENNO_MODEL_ALIASES` (カンマ区切り)。各項目は `alias` か
/// `alias=target`。`target` が今のモデル名と一致しない項目は無視する
/// (1 プロセス 1 モデルなので、他モデル向けの別名を受けてしまわないようにする)。
fn env_aliases(canonical: &str) -> Vec<String> {
    match std::env::var("DENNO_MODEL_ALIASES") {
        Ok(raw) => parse_aliases(&raw, canonical),
        Err(_) => Vec::new(),
    }
}

/// [`env_aliases`] の本体 (env を読まないので単体テストできる)。
fn parse_aliases(raw: &str, canonical: &str) -> Vec<String> {
    let mut out = Vec::new();
    for item in raw.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (alias, target) = match item.split_once('=') {
            Some((a, t)) => (a.trim(), Some(t.trim())),
            None => (item, None),
        };
        if alias.is_empty() {
            continue;
        }
        if let Some(t) = target {
            if !t.is_empty() && t != canonical {
                continue;
            }
        }
        out.push(alias.to_string());
    }
    out
}

/// 既定の別名: `model_id` (pull 元 repo id) とその最終セグメント、
/// `model_dir` の basename。どれも canonical 名と同じなら出さない。
fn default_aliases(state: &AppState) -> Vec<String> {
    let mut out = Vec::new();
    let id = state.model_id.trim();
    if !id.is_empty() {
        out.push(id.to_string());
        if let Some(tail) = id.rsplit('/').next() {
            if !tail.is_empty() && tail != id {
                out.push(tail.to_string());
            }
        }
    }
    let dir = state.info.model_dir.trim();
    if !dir.is_empty() {
        out.push(dir.to_string());
    }
    out
}

/// サービング中のモデルに登録された別名 (canonical 名は含めない、重複無し)。
pub(crate) fn model_aliases(state: &AppState) -> Vec<String> {
    let canonical = state.model_name.as_str();
    let mut out: Vec<String> = Vec::new();
    for a in default_aliases(state).into_iter().chain(env_aliases(canonical)) {
        let a = a.trim().to_string();
        if a.is_empty() || a == canonical || out.iter().any(|x| *x == a) {
            continue;
        }
        out.push(a);
    }
    out
}

/// `requested` が今サービングしているモデル (canonical 名 or 登録別名) を
/// 指すか。空文字は「未指定」として呼び出し側が扱う。
pub(crate) fn model_matches(state: &AppState, requested: &str) -> bool {
    requested == state.model_name || model_aliases(state).iter().any(|a| a == requested)
}

/// モデルの作成時刻 (epoch 秒) の解決順:
/// 1. `DENNO_MODEL_CREATED` (epoch 秒) — 運用が正確な値を与えたいとき。
/// 2. `register_model_dir` で登録された重みディレクトリの mtime
///    (`serve` 起動時のみ。重みを置いた時刻の近似)。
/// 3. このプロセスが最初にカタログを読んだ時刻。
///
/// 学習日時や repo の公開日時は分からない (`EngineInfo::model_dir` は basename
/// だけ) ので、2 は「重みを配置した時刻」の意味でしかない。
fn model_created() -> u64 {
    static CREATED: OnceLock<u64> = OnceLock::new();
    *CREATED.get_or_init(|| {
        let env = std::env::var("DENNO_MODEL_CREATED")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok());
        created_from(env, MODEL_DIR_PATH.get().map(|p| p.as_path())).unwrap_or_else(now_secs)
    })
}

/// [`model_created`] の優先順位だけを持つ純関数 (env も static も読まない)。
fn created_from(env: Option<u64>, dir: Option<&std::path::Path>) -> Option<u64> {
    if env.is_some() {
        return env;
    }
    let dir = dir?;
    let mtime = std::fs::metadata(dir).ok()?.modified().ok()?;
    Some(mtime.duration_since(UNIX_EPOCH).ok()?.as_secs())
}

/// `serve` 起動時に重みディレクトリを 1 回だけ登録する
/// ([`model_created`] が mtime を読むため)。テストは登録しない。
pub(crate) fn register_model_dir(path: &std::path::Path) {
    let _ = MODEL_DIR_PATH.set(path.to_path_buf());
}

/// [`register_model_dir`] が入れる重みディレクトリ (最初の 1 個だけ)。
static MODEL_DIR_PATH: OnceLock<std::path::PathBuf> = OnceLock::new();

/// `/v1/models` の `data[]` 1 件、および `/v1/models/:id` の本文。
/// Anthropic 側フィールド (`type` / `display_name` / `context_length`) と
/// OpenAI 側 (`object` / `owned_by` / `created`) を併記したまま、
/// `context_window` / `aliases` を足す。
pub(crate) fn model_object(state: &AppState) -> Value {
    let name = &state.model_name;
    json!({
        "id": name,
        "object": "model",
        "type": "model",
        "display_name": name,
        "created": model_created(),
        "owned_by": "denno",
        "context_window": state.info.context,
        "context_length": state.info.context,
        "max_context_length": state.info.context,
        "model_id": state.model_id,
        "model_dir": state.info.model_dir,
        "quant": state.info.quant,
        "slots": state.info.slots,
        "layers": state.info.layers,
        "aliases": model_aliases(state),
        "capabilities": {
            "anthropic": true,
            "openai": true,
            "ollama": true,
            "tools": true,
            "streaming": true,
            "count_tokens": true,
            "telemetry": denno_telemetry::enabled(),
        },
    })
}

/// `GET /v1/models/:id`。canonical 名または登録別名なら model object を返し、
/// それ以外は OpenAI 形式の 404 を返す。
///
/// このルートは router() (`lib.rs`) ではなくここで登録している。task 893 の
/// 契約テストは `lib.rs` の `.route("...")` 行と `docs/openapi.yaml` を
/// 突き合わせるため、仕様書に未記載のパスを lib.rs に足すと落ちる。
/// `docs/openapi.yaml` へ `GET /v1/models/{id}` を足すのは仕様側の作業
/// (スコープ外) なので、それまではここに置く。
pub(crate) fn models_router() -> Router<Arc<AppState>> {
    Router::new().route("/v1/models/:id", get(model_by_id))
}

async fn model_by_id(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, OaError> {
    check_auth(&state, &headers).map_err(OaError::from)?;
    if !model_matches(&state, &id) {
        return Err(OaError::new(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            format!("model: {id} not found"),
        ));
    }
    Ok(Json(model_object(&state)))
}

// ---------- リクエスト DTO ----------

/// `stop` は文字列 or 文字列配列 (OpenAI 仕様)。null は「無し」。
fn de_stop<'de, D>(d: D) -> Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Option::<Value>::deserialize(d)?;
    Ok(match v {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => {
            if s.is_empty() {
                Vec::new()
            } else {
                vec![s]
            }
        }
        Some(Value::Array(a)) => a
            .into_iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .filter(|s| !s.is_empty())
            .collect(),
        Some(_) => {
            return Err(serde::de::Error::custom(
                "stop must be a string or an array of strings",
            ))
        }
    })
}

/// 履歴の assistant `tool_calls[].function`。
/// `arguments` は OpenAI 仕様では **JSON 文字列**。デンノ拡張として
/// オブジェクト直書きも受ける (どちらも同じ input へ正規化する)。
#[derive(Debug, Deserialize)]
pub struct OaToolCallFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: Option<Value>,
}

/// 履歴の assistant `tool_calls[]` 1 件。
///
/// `id` は OpenAI クライアントが `role="tool"` の `tool_call_id` で参照するが、
/// chat template は id を参照しないため履歴テキストには使わない (受理のみ)。
#[derive(Debug, Deserialize)]
pub struct OaToolCall {
    #[serde(default)]
    #[allow(dead_code)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: OaToolCallFunction,
}

#[derive(Debug, Deserialize)]
pub struct OaMessage {
    pub role: String,
    /// 文字列 or content parts 配列 (null 許容)
    #[serde(default)]
    pub content: Option<Value>,
    /// assistant の tool call 履歴 (OpenAI 形式)
    #[serde(default)]
    pub tool_calls: Option<Vec<OaToolCall>>,
    /// role="tool" の応答が指す tool call id (受理するが履歴テキストには使わない)
    #[serde(default)]
    #[allow(dead_code)]
    pub tool_call_id: Option<String>,
}

/// OpenAI の `tools[]` 1 件 (`type="function"` のみ対応)。
#[derive(Debug, Deserialize)]
pub struct OaFunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema。未指定は空オブジェクト schema として扱う。
    #[serde(default)]
    pub parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct OaTool {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: OaFunctionDef,
}

#[derive(Debug, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<OaMessage>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default, deserialize_with = "de_stop")]
    pub stop: Vec<String>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub tools: Option<Vec<OaTool>>,
    pub tool_choice: Option<Value>,
    /// `{"type": "text" | "json_object" | "json_schema"}` (P837 で json_schema を追加)
    pub response_format: Option<Value>,
    /// P869: true で `choices[].logprobs` を埋める (既定 false = 従来と同一)
    pub logprobs: Option<bool>,
    /// P869: 各トークンで返す top_logprobs の件数 (0..=20)。`logprobs: true` が必須
    pub top_logprobs: Option<i64>,
    /// P869: サンプリング seed (OpenAI 互換)。未指定は engine 既定 (乱数)
    pub seed: Option<i64>,
}

/// `response_format` を検証し、モデルへ渡す追加ヒントを返す。
/// `text` は無変換、`json_object` は「JSON オブジェクトだけを返す」指示、
/// `json_schema` (P837) はスキーマ本文を埋め込んだ指示を system へ足す
/// (未指定時はプロンプト不変 = P793 と同一経路)。
fn response_format_hint(v: &Option<Value>) -> Result<Option<String>, OaError> {
    let Some(v) = v else { return Ok(None) };
    let fmt = crate::json_schema::parse_response_format(v).map_err(|msg| {
        let param = if msg.starts_with("response_format.json_schema") {
            "response_format.json_schema"
        } else {
            "response_format.type"
        };
        OaError::invalid(msg).with_param(param)
    })?;
    Ok(match fmt {
        crate::json_schema::ResponseFormat::Text => None,
        crate::json_schema::ResponseFormat::JsonObject => Some(JSON_OBJECT_HINT.to_string()),
        crate::json_schema::ResponseFormat::JsonSchema(spec) => Some(spec.hint()),
    })
}

/// `response_format: {"type":"json_object"}` のときに system へ足す固定文。
const JSON_OBJECT_HINT: &str =
    "Return a single valid JSON object and nothing else (no markdown fences, no prose).";

/// content (文字列 or parts 配列) を 1 本のテキストにする。
/// text 以外の part (image_url 等) は無視する — モデルはテキスト専用。
fn content_to_text(v: &Option<Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            for p in parts {
                match p {
                    Value::String(s) => out.push_str(s),
                    Value::Object(_) => match p.get("type").and_then(|t| t.as_str()) {
                        Some("text") | Some("input_text") => {
                            out.push_str(p.get("text").and_then(|t| t.as_str()).unwrap_or_default());
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            out
        }
        Some(other) => other.to_string(),
    }
}

/// assistant の `tool_calls[]` 1 件を `<tool_call>` XML テキストへ写す。
/// (`/v1/messages` の tool_use block と同一形式 — モデルが見る履歴を揃える)
fn tool_call_to_xml(c: &OaToolCall) -> Result<String, OaError> {
    if let Some(kind) = &c.kind {
        if kind != "function" {
            return Err(OaError::invalid(format!(
                "tool_calls[].type: unsupported value {kind:?} (function)"
            )));
        }
    }
    if c.function.name.trim().is_empty() {
        return Err(OaError::invalid(
            "tool_calls[].function.name must be a non-empty string",
        ));
    }
    let input = match &c.function.arguments {
        None | Some(Value::Null) => json!({}),
        Some(Value::String(s)) if s.trim().is_empty() => json!({}),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(v) if v.is_object() => v,
            Ok(_) => {
                return Err(OaError::invalid(
                    "tool_calls[].function.arguments must be a JSON object string",
                ))
            }
            Err(e) => {
                return Err(OaError::invalid(format!(
                    "tool_calls[].function.arguments is not valid JSON: {e}"
                )))
            }
        },
        Some(v) if v.is_object() => v.clone(),
        Some(_) => {
            return Err(OaError::invalid(
                "tool_calls[].function.arguments must be a JSON object string",
            ))
        }
    };
    Ok(tool_use_to_xml(&c.function.name, &input))
}

/// OpenAI のメッセージ列を (system, chat_messages) へ写す。
///
/// - `system` / `developer` → system
/// - assistant の `tool_calls` → 本文の後ろへ `<tool_call>` XML を連結
///   (`/v1/messages` の tool_use block と同じテキスト)
/// - `tool` / `function` (実行結果) → user 発話の `<tool_response>` 包み
///   (`/v1/messages` の tool_result block と同じテキスト)
fn split_messages(msgs: &[OaMessage]) -> Result<(Option<String>, Vec<ChatMessage>), OaError> {
    let mut system: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let mut text = content_to_text(&m.content);
        match m.role.as_str() {
            "system" | "developer" => system.push(text),
            "user" => out.push(ChatMessage { role: "user".to_string(), content: text }),
            "assistant" => {
                for (j, c) in m.tool_calls.as_deref().unwrap_or(&[]).iter().enumerate() {
                    let xml = tool_call_to_xml(c).map_err(|e| {
                        e.with_param(format!("messages[{i}].tool_calls[{j}]"))
                    })?;
                    if !text.is_empty() {
                        text.push_str("\n\n");
                    }
                    text.push_str(&xml);
                }
                out.push(ChatMessage { role: "assistant".to_string(), content: text });
            }
            "tool" | "function" => {
                // 公式テンプレートの tool 応答形式 (Anthropic 経路の tool_result と同一)。
                // `tool_call_id` は受理するが、テンプレートは id を参照しないため使わない。
                out.push(ChatMessage {
                    role: "user".to_string(),
                    content: format!("<tool_response>\n{text}\n</tool_response>"),
                });
            }
            other => {
                return Err(OaError::invalid(format!(
                    "messages[{i}].role: unsupported role {other:?} \
                     (expected system, developer, user, assistant, tool or function)"
                ))
                .with_param(format!("messages[{i}].role")))
            }
        }
    }
    let system = if system.is_empty() { None } else { Some(system.join("\n")) };
    Ok((system, out))
}

/// OpenAI の `tools[]` を engine の [`ToolDef`] 列へ写す。
fn map_tools(tools: &Option<Vec<OaTool>>) -> Result<Vec<ToolDef>, OaError> {
    let Some(tools) = tools else { return Ok(Vec::new()) };
    let mut out = Vec::with_capacity(tools.len());
    for (i, t) in tools.iter().enumerate() {
        if let Some(kind) = &t.kind {
            if kind != "function" {
                return Err(OaError::invalid(format!(
                    "tools[{i}].type: unsupported value {kind:?} (function)"
                ))
                .with_param(format!("tools[{i}].type")));
            }
        }
        let input_schema = match &t.function.parameters {
            None | Some(Value::Null) => json!({"type": "object", "properties": {}}),
            Some(v) if v.is_object() => v.clone(),
            Some(_) => {
                return Err(OaError::invalid(format!(
                    "tools[{i}].function.parameters must be a JSON object"
                ))
                .with_param(format!("tools[{i}].function.parameters")))
            }
        };
        out.push(ToolDef {
            name: t.function.name.clone(),
            description: t.function.description.clone(),
            input_schema,
        });
    }
    Ok(out)
}

/// OpenAI の `tool_choice` を既存の [`parse_tool_choice`] が読む形へ正規化する。
///
/// - `"auto"` / `"none"` → そのまま
/// - `"required"` → `"any"` (拘束デコードを有効にする)
/// - `{"type":"function","function":{"name":..}}` → `{"type":"tool","name":..}`
fn oa_tool_choice_value(v: &Option<Value>) -> Result<Option<Value>, OaError> {
    let Some(v) = v else { return Ok(None) };
    let normalized = match v {
        Value::String(s) => match s.as_str() {
            "auto" => json!("auto"),
            "none" => json!("none"),
            "required" => json!("any"),
            other => {
                return Err(OaError::invalid(format!(
                    "tool_choice: unsupported value {other:?} (auto/none/required)"
                ))
                .with_param("tool_choice"))
            }
        },
        Value::Object(m) => {
            let ty = m
                .get("type")
                .and_then(|t| t.as_str())
                .ok_or_else(|| {
                    OaError::invalid("tool_choice: object form requires a string \"type\"")
                        .with_param("tool_choice")
                })?;
            match ty {
                "auto" | "none" => json!(ty),
                "required" | "any" => json!("any"),
                "function" => {
                    let name = m
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .ok_or_else(|| {
                            OaError::invalid(
                                "tool_choice: {\"type\":\"function\"} requires a string function.name",
                            )
                            .with_param("tool_choice")
                        })?;
                    if name.is_empty() {
                        return Err(OaError::invalid(
                            "tool_choice: function.name must be non-empty",
                        )
                        .with_param("tool_choice"));
                    }
                    json!({"type": "tool", "name": name})
                }
                other => {
                    return Err(OaError::invalid(format!(
                        "tool_choice.type: unsupported value {other:?} (auto/none/required/function)"
                    ))
                    .with_param("tool_choice"))
                }
            }
        }
        _ => {
            return Err(OaError::invalid(
                "tool_choice must be a string (auto/none/required) or an object",
            )
            .with_param("tool_choice"))
        }
    };
    Ok(Some(normalized))
}

/// engine の stop_reason → OpenAI の finish_reason。
fn finish_reason(engine_reason: &str) -> &'static str {
    match engine_reason {
        "max_tokens" => "length",
        _ => "stop",
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn hex24() -> String {
    let mut rng = rand::thread_rng();
    (0..24).map(|_| format!("{:x}", rng.gen::<u8>() % 16)).collect()
}

/// `chatcmpl-` + 24hex (msg id と同じ乱数経路)。
fn new_chatcmpl_id() -> String {
    format!("chatcmpl-{}", hex24())
}

/// tool call の id (OpenAI の `call_` + 24hex)。
fn new_call_id() -> String {
    format!("call_{}", hex24())
}

fn usage_json(input_tokens: usize, output_tokens: usize) -> Value {
    json!({
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    })
}

// ---------- P869: logprobs / seed ----------

/// `logprobs` / `top_logprobs` を検証し、エンジンへの記録要求
/// (`Some(top_n)`) に正規化する。OpenAI と同じ規則:
/// - `top_logprobs` は 0..=20 (範囲外は 400)
/// - `top_logprobs` は `logprobs: true` が必須 (単独指定は 400)
/// - `logprobs: false` / 未指定なら記録しない (追加コストゼロ)
fn normalize_logprobs(
    logprobs: Option<bool>,
    top_logprobs: Option<i64>,
) -> Result<Option<usize>, OaError> {
    let want = logprobs.unwrap_or(false);
    if let Some(n) = top_logprobs {
        if !(0..=20).contains(&n) {
            return Err(OaError::invalid(format!(
                "top_logprobs: {n} is out of range (0..=20)"
            ))
            .with_param("top_logprobs"));
        }
        if !want {
            return Err(OaError::invalid(
                "top_logprobs: requires logprobs=true",
            )
            .with_param("top_logprobs"));
        }
    }
    Ok(if want { Some(top_logprobs.unwrap_or(0) as usize) } else { None })
}

/// `content[]` 1 件 (`{token, logprob, bytes, top_logprobs}`)。
fn token_logprob_json(t: &TokenLogprob) -> Value {
    json!({
        "token": String::from_utf8_lossy(&t.bytes),
        "logprob": t.logprob,
        "bytes": t.bytes,
        "top_logprobs": t
            .top
            .iter()
            .map(|x| json!({
                "token": String::from_utf8_lossy(&x.bytes),
                "logprob": x.logprob,
                "bytes": x.bytes,
            }))
            .collect::<Vec<Value>>(),
    })
}

/// `choices[].logprobs` (`{content: [...], refusal: null}`)。
fn choice_logprobs_json(entries: &[TokenLogprob]) -> Value {
    json!({
        "content": entries.iter().map(token_logprob_json).collect::<Vec<Value>>(),
        "refusal": Value::Null,
    })
}

/// 非ストリーム応答の `message`。tool call があれば `tool_calls` を足し、
/// 本文が空なら `content: null` (OpenAI 仕様の tool call 応答と同じ)。
fn completion_message(text: &str, calls: &[ExtractedToolCall]) -> Value {
    let content = if calls.is_empty() || !text.is_empty() {
        Value::String(text.to_string())
    } else {
        Value::Null
    };
    let mut m = json!({"role": "assistant", "content": content});
    if !calls.is_empty() {
        let arr: Vec<Value> = calls
            .iter()
            .map(|c| {
                json!({
                    "id": new_call_id(),
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": serde_json::to_string(&c.input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                })
            })
            .collect();
        m["tool_calls"] = Value::Array(arr);
    }
    m
}

// ---------- ハンドラ ----------

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, OaError> {
    check_auth(&state, &headers).map_err(OaError::from)?;
    // P817: drain 中は新規の生成要求を受けない (/v1/messages と同じ 503)。
    crate::reject_if_draining(&state).map_err(OaError::from)?;
    let req: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| OaError::invalid(format!("invalid request body: {e}")))?;

    // model: /v1/messages と同じ規則 (不一致は 400/404 系。OpenAI は 404 ではなく
    // 400 invalid_request_error を返す実装が多いが、既存の分類を尊重する)
    // task 901: 登録別名 (canonical 名以外) も同じモデルとして受ける。
    if !req.model.is_empty() && !model_matches(&state, &req.model) {
        return Err(OaError::from(ApiError::not_found(format!(
            "model: {} not found",
            req.model
        ))));
    }
    if req.messages.is_empty() {
        return Err(OaError::invalid("messages: at least one message is required")
            .with_param("messages"));
    }
    let (system, chat_messages) = split_messages(&req.messages)?;
    if chat_messages.is_empty() {
        return Err(
            OaError::invalid("messages: at least one user or assistant message is required")
                .with_param("messages"),
        );
    }
    // P869: logprobs / top_logprobs の検証 (範囲外・単独指定は 400)
    let lp_req = normalize_logprobs(req.logprobs, req.top_logprobs)?;

    // tools / tool_choice: /v1/messages と同じ検証規則を通す
    let tool_defs = map_tools(&req.tools)?;
    if req.tool_choice.is_some() && tool_defs.is_empty() {
        return Err(OaError::invalid("tool_choice is only allowed when tools are provided")
            .with_param("tool_choice"));
    }
    let tool_choice = parse_tool_choice(&oa_tool_choice_value(&req.tool_choice)?)
        .map_err(OaError::from)?;
    let tools_json = validate_tools(&tool_defs).map_err(OaError::from)?;
    if let ToolChoice::Named(n) = &tool_choice {
        if !tool_defs.iter().any(|t| &t.name == n) {
            return Err(OaError::invalid(format!(
                "tool_choice: tool {n:?} is not defined in tools"
            ))
            .with_param("tool_choice"));
        }
    }
    let tools_active = !tool_defs.is_empty() && tool_choice != ToolChoice::None;
    let tools_for_prompt = if tools_active { Some(tools_json) } else { None };
    let tool_constraint = match &tool_choice {
        ToolChoice::Any => Some(ToolConstraint {
            names: tool_defs.iter().map(|t| t.name.clone()).collect(),
        }),
        ToolChoice::Named(n) => Some(ToolConstraint { names: vec![n.clone()] }),
        ToolChoice::Auto | ToolChoice::None => None,
    };

    // /v1/messages と同じ既定 (thinking = brief) を適用し、同一会話で同一出力にする。
    let mut system = brief_system(system);
    if let Some(hint) = response_format_hint(&req.response_format)? {
        system = Some(match system {
            Some(s) => format!("{s}\n\n{hint}"),
            None => hint.to_string(),
        });
    }
    let enable_thinking = Some(false);
    let params = GenParams {
        max_tokens: resolve_max_tokens(req.max_tokens, req.max_completion_tokens),
        temperature: apply_temperature_cap(req.temperature, state.temperature_cap),
        top_p: req.top_p,
        top_k: None,
        stop_sequences: req.stop.clone(),
        seed: req.seed.map(|s| s as u64),
        repetition_penalty: state.default_repetition_penalty,
        frequency_penalty: None,
        presence_penalty: None,
        no_repeat_ngram_size: None,
    };

    if req.stream.unwrap_or(false) {
        stream_response(
            state,
            system,
            chat_messages,
            params,
            req.stream_options,
            tools_for_prompt,
            tool_constraint,
            enable_thinking,
            tools_active,
            lp_req,
        )
        .await
    } else {
        let rx = state
            .scheduler
            .submit_chat_ext(
                system,
                chat_messages,
                params,
                tools_for_prompt,
                tool_constraint,
                enable_thinking,
                lp_req,
            )
            .map_err(|_| queue_full(&state))?;
        let wait = crate::response_wait_timeout(state.scheduler.limits());
        let output = match tokio::time::timeout(wait, rx).await {
            Ok(Ok(Ok(out))) => out,
            Ok(Ok(Err(e))) => return Err(OaError::from(crate::error_from_anyhow(e))),
            Ok(Err(e)) => {
                return Err(OaError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    format!("scheduler dropped: {e}"),
                ))
            }
            Err(_) => {
                return Err(OaError::new(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "server_error",
                    format!(
                        "timeout: no response from the engine within {:.1}s; \
                         the request was abandoned (retry later)",
                        wait.as_secs_f64()
                    ),
                ))
            }
        };
        // tools 有効時だけ抽出する (none / 未指定なら本文は P793 と同一)
        let mut text = output.text;
        let mut calls: Vec<ExtractedToolCall> = Vec::new();
        if tools_active {
            let (clean, extracted) = extract_tool_calls(&text);
            text = clean;
            calls = extracted;
        }
        let finish = if calls.is_empty() {
            finish_reason(output.stop_reason.as_str())
        } else {
            "tool_calls"
        };
        // P869: logprobs 要求時のみ content[] を埋める (非要求は null で従来と同一)。
        // 注: content は生成トークン列 (EOS 除外) に対応し、tool call 抽出で
        // 本文から落ちたトークンも含む。
        let lp_json = match lp_req {
            Some(_) => choice_logprobs_json(&output.logprobs),
            None => Value::Null,
        };
        let created = now_secs();
        Ok(Json(json!({
            "id": new_chatcmpl_id(),
            "object": "chat.completion",
            "created": created,
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "message": completion_message(&text, &calls),
                "logprobs": lp_json,
                "finish_reason": finish,
            }],
            "usage": usage_json(output.input_tokens, output.output_tokens),
        }))
        .into_response())
    }
}

/// P873: キュー満杯 (待ち行列の上限超過)。
///
/// OpenAI のプロトコルでは「一時的な容量不足 = 後で再試行」は **429**
/// (`rate_limit_exceeded`) で、`Retry-After` を添えるのが慣例。Anthropic 経路
/// (`/v1/messages`) は overloaded_error の 503 のまま (プロトコルごとに
/// 適切な形が違うため、ここで分ける)。
fn queue_full(state: &AppState) -> OaError {
    OaError::new(
        StatusCode::TOO_MANY_REQUESTS,
        "rate_limit_exceeded",
        format!(
            "server busy: request queue is full (max_queue = {}, in_flight = 1); retry later",
            state.scheduler.max_queue()
        ),
    )
    .with_retry_after(crate::QUEUE_RETRY_AFTER_SECS)
}

/// 1 個の delta chunk を送る。送信失敗 (クライアント切断) は false。
async fn send_delta_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
) -> bool {
    let chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "logprobs": Value::Null,
            "finish_reason": Value::Null,
        }],
    });
    tx.send(Ok(sse_data(&chunk))).await.is_ok()
}

/// P869: 1 トークン分の logprobs を載せた chunk を送る (delta は空)。
/// OpenAI は logprobs を任意の chunk に載せられるので、content と分離して流す。
async fn send_logprob_chunk(
    tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
    id: &str,
    created: u64,
    model: &str,
    t: &TokenLogprob,
) -> bool {
    let chunk = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "logprobs": choice_logprobs_json(std::slice::from_ref(t)),
            "finish_reason": Value::Null,
        }],
    });
    tx.send(Ok(sse_data(&chunk))).await.is_ok()
}

/// [`StreamAssembler`] のイベント列を OpenAI の delta へ翻訳する。
///
/// - text_delta → `{"content": ...}`
/// - tool_use start → `{"tool_calls":[{"index":n,"id":..,"type":"function",
///   "function":{"name":..,"arguments":""}}]}`
/// - input_json_delta → `{"tool_calls":[{"index":n,"function":{"arguments":..}}]}`
///
/// tool call の index は送出順 (0 始まり)。id は start の 1 回だけ載せる。
struct OaToolDeltaTranslator {
    /// 送出済み tool call 数 (= 次に割り当てる index)
    emitted: usize,
    /// open 中の tool_use の block index → OpenAI の tool index
    open: Option<usize>,
}

impl OaToolDeltaTranslator {
    fn new() -> Self {
        Self { emitted: 0, open: None }
    }
}

/// SSE 応答。scheduler の生イベントを OpenAI chunk へ組み直す。
///
/// `assemble_tools` が true のときだけ [`StreamAssembler`] を通す
/// (`<tool_call>` block の検出と分割到着の境界処理を既存実装へ委ねる)。
#[allow(clippy::too_many_arguments)]
async fn stream_response(
    state: Arc<AppState>,
    system: Option<String>,
    chat_messages: Vec<ChatMessage>,
    params: GenParams,
    stream_options: Option<StreamOptions>,
    tools: Option<Value>,
    constraint: Option<ToolConstraint>,
    enable_thinking: Option<bool>,
    assemble_tools: bool,
    logprobs: Option<usize>,
) -> Result<Response, OaError> {
    let include_usage = stream_options.map(|o| o.include_usage).unwrap_or(false);
    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::channel::<RawEvent>(64);
    state
        .scheduler
        .submit_openai_stream_ext(
            system,
            chat_messages,
            params,
            tools,
            constraint,
            enable_thinking,
            raw_tx,
            logprobs,
        )
        .map_err(|_| queue_full(&state))?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    let id = new_chatcmpl_id();
    let created = now_secs();
    let model = state.model_name.clone();
    tokio::spawn(async move {
        // 最初の chunk: delta.role (OpenAI の慣習どおり content は空文字)
        let first = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "logprobs": Value::Null,
                "finish_reason": Value::Null,
            }],
        });
        if tx.send(Ok(sse_data(&first))).await.is_err() {
            return; // クライアント切断 → raw_rx を落として生成を打ち切らせる
        }
        let mut reason = "stop";
        let (mut input_tokens, mut output_tokens) = (0usize, 0usize);
        // tools 有効時のみ (P797): block 境界検出は既存アセンブラに任せる
        let mut asm = if assemble_tools { Some(StreamAssembler::new(0)) } else { None };
        let mut tr = OaToolDeltaTranslator::new();
        while let Some(ev) = raw_rx.recv().await {
            match ev {
                RawEvent::Start { input_tokens: n } => input_tokens = n,
                RawEvent::Delta(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    match asm.as_mut() {
                        None => {
                            if !send_delta_chunk(&tx, &id, created, &model, json!({"content": text}))
                                .await
                            {
                                return;
                            }
                        }
                        Some(asm) => {
                            asm.push_delta(&text);
                            for e in asm.take_events() {
                                let delta = match e["type"].as_str() {
                                    Some("content_block_start")
                                        if e["content_block"]["type"] == "tool_use" =>
                                    {
                                        let idx = tr.emitted;
                                        tr.emitted += 1;
                                        tr.open = Some(idx);
                                        Some(json!({"tool_calls": [{
                                            "index": idx,
                                            "id": new_call_id(),
                                            "type": "function",
                                            "function": {
                                                "name": e["content_block"]["name"]
                                                    .as_str()
                                                    .unwrap_or_default(),
                                                "arguments": "",
                                            },
                                        }]}))
                                    }
                                    Some("content_block_delta")
                                        if e["delta"]["type"] == "input_json_delta" =>
                                    {
                                        let idx = tr.open.unwrap_or(tr.emitted.saturating_sub(1));
                                        Some(json!({"tool_calls": [{
                                            "index": idx,
                                            "function": {
                                                "arguments": e["delta"]["partial_json"]
                                                    .as_str()
                                                    .unwrap_or_default(),
                                            },
                                        }]}))
                                    }
                                    Some("content_block_delta")
                                        if e["delta"]["type"] == "text_delta" =>
                                    {
                                        let t = e["delta"]["text"].as_str().unwrap_or_default();
                                        if t.is_empty() {
                                            None
                                        } else {
                                            Some(json!({"content": t}))
                                        }
                                    }
                                    _ => None,
                                };
                                if let Some(d) = delta {
                                    if !send_delta_chunk(&tx, &id, created, &model, d).await {
                                        return;
                                    }
                                }
                            }
                        }
                    }
                }
                RawEvent::Done { input_tokens: i, output_tokens: o, stop_reason, .. } => {
                    input_tokens = i;
                    output_tokens = o;
                    reason = finish_reason(&stop_reason);
                }
                // P869: logprobs (`logprobs=true` のときだけ届く)
                RawEvent::Logprob(t) => {
                    if !send_logprob_chunk(&tx, &id, created, &model, &t).await {
                        return;
                    }
                }
                RawEvent::Error(msg) => {
                    let err = json!({
                        "error": {"message": msg, "type": "server_error", "param": Value::Null, "code": Value::Null}
                    });
                    let _ = tx.send(Ok(sse_data(&err))).await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
            }
        }
        // 残り (ホールドされていた末尾テキスト / 未完結 block) を流し切る
        if let Some(asm) = asm.as_mut() {
            asm.finish();
            for e in asm.take_events() {
                if e["type"] == "content_block_delta" && e["delta"]["type"] == "text_delta" {
                    let t = e["delta"]["text"].as_str().unwrap_or_default();
                    if !t.is_empty()
                        && !send_delta_chunk(&tx, &id, created, &model, json!({"content": t})).await
                    {
                        return;
                    }
                }
            }
            if asm.blocks_emitted() > 0 {
                reason = "tool_calls";
            }
        }
        // 最終 chunk (finish_reason)
        let last = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "logprobs": Value::Null,
                "finish_reason": reason,
            }],
        });
        if tx.send(Ok(sse_data(&last))).await.is_err() {
            return;
        }
        // stream_options.include_usage: choices が空の usage chunk (OpenAI 仕様)
        if include_usage {
            let u = json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage_json(input_tokens, output_tokens),
            });
            if tx.send(Ok(sse_data(&u))).await.is_err() {
                return;
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        // OpenAI クライアントは `data: ping` を JSON として解釈しようとするため、
        // SSE のコメント行 (`: ping`) で keep-alive する。
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(2))
                .event(Event::default().comment("ping")),
        )
        .into_response())
}

fn sse_data(v: &Value) -> Event {
    Event::default().data(v.to_string())
}

// ---------- P837: POST /v1/completions (旧 text completions) ----------

/// `POST /v1/completions` のリクエスト (OpenAI 旧 Completions API)。
///
/// chat template を通さず、`prompt` の文字列をそのままトークン化して続きを
/// 生成する (IDE の FIM / インライン補完のように role 包みを望まないクライアント用)。
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    #[serde(default)]
    pub model: String,
    /// 文字列 or 1 要素の文字列配列。複数プロンプト (batch) とトークン ID 配列は 400。
    pub prompt: Value,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    #[serde(default, deserialize_with = "de_stop")]
    pub stop: Vec<String>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    /// true なら `text` の先頭にプロンプト自身を付ける (OpenAI と同じ)。
    #[serde(default)]
    pub echo: Option<bool>,
}

/// `prompt` を 1 本の文字列へ正規化する。空文字は生成不能なので 400。
fn normalize_prompt(v: &Value) -> Result<String, OaError> {
    let s = match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => {
            if a.len() != 1 {
                return Err(OaError::invalid(format!(
                    "prompt: batch of {} prompts is not supported (send exactly one prompt)",
                    a.len()
                ))
                .with_param("prompt"));
            }
            match &a[0] {
                Value::String(s) => s.clone(),
                other => {
                    return Err(OaError::invalid(format!(
                        "prompt[0]: expected a string, got {}",
                        json_type_name(other)
                    ))
                    .with_param("prompt[0]"))
                }
            }
        }
        other => {
            return Err(OaError::invalid(format!(
                "prompt: expected a string or an array of one string, got {} \
                 (token-id arrays are not supported)",
                json_type_name(other)
            ))
            .with_param("prompt"))
        }
    };
    if s.is_empty() {
        return Err(OaError::invalid("prompt: must be a non-empty string").with_param("prompt"));
    }
    Ok(s)
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// `cmpl-` + 24hex。
fn new_cmpl_id() -> String {
    format!("cmpl-{}", hex24())
}

/// `/v1/completions` の 1 chunk (`object = text_completion`)。
fn completion_chunk(id: &str, created: u64, model: &str, text: &str, finish: Value) -> Value {
    json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": finish,
        }],
    })
}

pub async fn completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, OaError> {
    check_auth(&state, &headers).map_err(OaError::from)?;
    crate::reject_if_draining(&state).map_err(OaError::from)?;
    let req: CompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| OaError::invalid(format!("invalid request body: {e}")))?;

    // task 901: `/v1/chat/completions` と同じく登録別名を受ける。
    if !req.model.is_empty() && !model_matches(&state, &req.model) {
        return Err(OaError::from(ApiError::not_found(format!(
            "model: {} not found",
            req.model
        ))));
    }
    let prompt = normalize_prompt(&req.prompt)?;

    // chat 経路と同じ既定 (max_tokens クランプ・temperature 上限・反復抑止の既定値)
    let params = GenParams {
        max_tokens: resolve_max_tokens(req.max_tokens, req.max_completion_tokens),
        temperature: apply_temperature_cap(req.temperature, state.temperature_cap),
        top_p: req.top_p,
        top_k: None,
        stop_sequences: req.stop.clone(),
        seed: None,
        repetition_penalty: state.default_repetition_penalty,
        frequency_penalty: None,
        presence_penalty: None,
        no_repeat_ngram_size: None,
    };
    let echo = req.echo.unwrap_or(false);

    if req.stream.unwrap_or(false) {
        return completion_stream_response(state, prompt, params, req.stream_options, echo).await;
    }

    let rx = state
        .scheduler
        .submit_completion(prompt.clone(), params)
        .map_err(|_| queue_full(&state))?;
    let wait = crate::response_wait_timeout(state.scheduler.limits());
    let output = match tokio::time::timeout(wait, rx).await {
        Ok(Ok(Ok(out))) => out,
        Ok(Ok(Err(e))) => return Err(OaError::from(crate::error_from_anyhow(e))),
        Ok(Err(e)) => {
            return Err(OaError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!("scheduler dropped: {e}"),
            ))
        }
        Err(_) => {
            return Err(OaError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                format!(
                    "timeout: no response from the engine within {:.1}s; \
                     the request was abandoned (retry later)",
                    wait.as_secs_f64()
                ),
            ))
        }
    };
    let mut text = String::new();
    if echo {
        text.push_str(&prompt);
    }
    text.push_str(&output.text);
    let finish = finish_reason(output.stop_reason.as_str());
    Ok(Json(json!({
        "id": new_cmpl_id(),
        "object": "text_completion",
        "created": now_secs(),
        "model": state.model_name,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": Value::Null,
            "finish_reason": finish,
        }],
        "usage": usage_json(output.input_tokens, output.output_tokens),
    }))
    .into_response())
}

/// `/v1/completions` の SSE (`object = text_completion` の chunk + `[DONE]`)。
async fn completion_stream_response(
    state: Arc<AppState>,
    prompt: String,
    params: GenParams,
    stream_options: Option<StreamOptions>,
    echo: bool,
) -> Result<Response, OaError> {
    let include_usage = stream_options.map(|o| o.include_usage).unwrap_or(false);
    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::channel::<RawEvent>(64);
    state
        .scheduler
        .submit_completion_stream(prompt.clone(), params, raw_tx)
        .map_err(|_| queue_full(&state))?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
    let id = new_cmpl_id();
    let created = now_secs();
    let model = state.model_name.clone();
    tokio::spawn(async move {
        let send = |v: Value| {
            let tx = tx.clone();
            async move { tx.send(Ok(sse_data(&v))).await.is_ok() }
        };
        // echo: プロンプト自身を最初の chunk として流す (OpenAI と同じ)
        if echo && !prompt.is_empty() {
            if !send(completion_chunk(&id, created, &model, &prompt, Value::Null)).await {
                return;
            }
        }
        let mut reason = "stop";
        let (mut input_tokens, mut output_tokens) = (0usize, 0usize);
        while let Some(ev) = raw_rx.recv().await {
            match ev {
                RawEvent::Start { input_tokens: n } => input_tokens = n,
                RawEvent::Delta(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    if !send(completion_chunk(&id, created, &model, &text, Value::Null)).await {
                        return;
                    }
                }
                RawEvent::Done { input_tokens: i, output_tokens: o, stop_reason, .. } => {
                    input_tokens = i;
                    output_tokens = o;
                    reason = finish_reason(&stop_reason);
                }
                // P869: /v1/completions は logprobs を要求しない (常に None)。
                RawEvent::Logprob(_) => {}
                RawEvent::Error(msg) => {
                    let err = json!({
                        "error": {"message": msg, "type": "server_error", "param": Value::Null, "code": Value::Null}
                    });
                    let _ = tx.send(Ok(sse_data(&err))).await;
                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
            }
        }
        // 最終 chunk (finish_reason)
        if !send(completion_chunk(&id, created, &model, "", json!(reason))).await {
            return;
        }
        if include_usage {
            let u = json!({
                "id": id,
                "object": "text_completion",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage_json(input_tokens, output_tokens),
            });
            if !send(u).await {
                return;
            }
        }
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(
            KeepAlive::new()
                .interval(std::time::Duration::from_secs(2))
                .event(Event::default().comment("ping")),
        )
        .into_response())
}

// ---------- 単体テスト (task 901 の別名・created 解決) ----------

#[cfg(test)]
mod catalog_tests {
    use super::{created_from, parse_aliases};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn parse_aliases_keeps_bare_and_matching_targets() {
        let got = parse_aliases(" swift , fast=denno-test ,,bad=other", "denno-test");
        assert_eq!(got, vec!["swift".to_string(), "fast".to_string()]);
    }

    #[test]
    fn parse_aliases_ignores_empty_items() {
        assert!(parse_aliases("", "denno-test").is_empty());
        assert!(parse_aliases(" , =x , ", "denno-test").is_empty());
    }

    #[test]
    fn created_from_prefers_env_over_dir_mtime() {
        let dir = std::env::temp_dir();
        assert_eq!(created_from(Some(1234), Some(dir.as_path())), Some(1234));
    }

    #[test]
    fn created_from_uses_dir_mtime_without_env() {
        let dir = std::env::temp_dir();
        let got = created_from(None, Some(dir.as_path())).expect("temp_dir の mtime が読めない");
        // temp_dir は 2020 年以降に存在する (mtime が実時刻であることの下限)。
        assert!(got > 1_577_836_800, "mtime = {got}");
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        assert!(got <= now, "mtime が未来: {got} > {now}");
    }

    #[test]
    fn created_from_is_none_without_env_and_dir() {
        assert_eq!(created_from(None, None), None);
        // 存在しないディレクトリも None (呼び出し側が now に落とす)。
        let missing = std::path::Path::new("/nonexistent-denno-model-dir-901");
        assert_eq!(created_from(None, Some(missing)), None);
    }
}
