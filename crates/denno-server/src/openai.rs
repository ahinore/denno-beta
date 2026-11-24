//! P793: OpenAI Chat Completions 互換エンドポイント (`POST /v1/chat/completions`)。
//!
//! `/v1/messages` と **同じ** Scheduler・chat template・KV prefix cache・engine を
//! 通す。違うのはワイヤ形式だけ:
//!
//! - リクエスト: OpenAI の `messages` / `stop` / `stream_options`
//! - 応答: `chat.completion` (非ストリーム) と `chat.completion.chunk` (SSE)
//!
//! 既存の Anthropic 経路 (`/v1/messages`) には一切触らない。ストリームは
//! scheduler の [`crate::scheduler::RawEvent`] (生 delta) を HTTP 側で
//! chunk へ組み直す — アセンブラ (tool_use block 分割) は持たない。
//!
//! 対応フィールド (docs/api.md §POST /v1/chat/completions):
//! `model` / `messages` (system・developer・user・assistant、content は
//! 文字列 or text part 配列) / `max_tokens` / `max_completion_tokens` /
//! `temperature` / `top_p` / `stop` (文字列 or 配列) / `stream` /
//! `stream_options.include_usage`。未知フィールドは無視する (serde 既定)。

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use denno_engine::{ChatMessage, GenParams};
use rand::Rng;
use serde::Deserialize;
use serde_json::{json, Value};
use tokio_stream::wrappers::ReceiverStream;

use crate::scheduler::RawEvent;
use crate::{
    apply_temperature_cap, brief_system, check_auth, debug_api_error, resolve_max_tokens, ApiError,
    AppState, RESPONSE_WAIT_TIMEOUT,
};

// ---------- エラー (OpenAI 形式) ----------

/// OpenAI 互換のエラー本文。`{"error": {"message", "type", "param", "code"}}`。
#[derive(Debug)]
pub struct OaError {
    status: StatusCode,
    kind: &'static str,
    message: String,
    param: Option<String>,
}

impl OaError {
    fn new(status: StatusCode, kind: &'static str, message: impl Into<String>) -> Self {
        Self { status, kind, message: message.into(), param: None }
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
        (self.status, Json(body)).into_response()
    }
}

/// `ApiError` (Anthropic 形式) → OpenAI 形式。中身は [`debug_api_error`] で読む
/// (値は既存の分類のままで、この変換は表示形式だけを変える)。
impl From<ApiError> for OaError {
    fn from(e: ApiError) -> Self {
        let (status, kind, message) = debug_api_error(e);
        // Anthropic の type 名を OpenAI の type 名へ寄せる (未知はそのまま)
        let kind = match kind {
            "not_found_error" => "invalid_request_error",
            "overloaded_error" => "server_error",
            "api_error" => "server_error",
            other => other,
        };
        Self::new(
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            kind,
            message,
        )
    }
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

#[derive(Debug, Deserialize)]
pub struct OaMessage {
    pub role: String,
    /// 文字列 or content parts 配列 (null 許容)
    #[serde(default)]
    pub content: Option<Value>,
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
}

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

/// OpenAI のメッセージ列を (system, chat_messages) へ写す。
/// `system` / `developer` は system へ、`tool` / `function` は user の発話として
/// 履歴へ織り込む (どちらのロールも chat template には存在しないため)。
fn split_messages(msgs: &[OaMessage]) -> Result<(Option<String>, Vec<ChatMessage>), OaError> {
    let mut system: Vec<String> = Vec::new();
    let mut out = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        let text = content_to_text(&m.content);
        match m.role.as_str() {
            "system" | "developer" => system.push(text),
            "user" => out.push(ChatMessage { role: "user".to_string(), content: text }),
            "assistant" => out.push(ChatMessage { role: "assistant".to_string(), content: text }),
            "tool" | "function" => out.push(ChatMessage { role: "user".to_string(), content: text }),
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

/// `chatcmpl-` + 24hex (msg id と同じ乱数経路)。
fn new_chatcmpl_id() -> String {
    let mut rng = rand::thread_rng();
    let id: String = (0..24).map(|_| format!("{:x}", rng.gen::<u8>() % 16)).collect();
    format!("chatcmpl-{id}")
}

fn usage_json(input_tokens: usize, output_tokens: usize) -> Value {
    json!({
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    })
}

// ---------- ハンドラ ----------

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Response, OaError> {
    check_auth(&state, &headers).map_err(OaError::from)?;
    let req: ChatCompletionRequest = serde_json::from_slice(&body)
        .map_err(|e| OaError::invalid(format!("invalid request body: {e}")))?;

    // model: /v1/messages と同じ規則 (不一致は 400/404 系。OpenAI は 404 ではなく
    // 400 invalid_request_error を返す実装が多いが、既存の分類を尊重する)
    if !req.model.is_empty() && req.model != state.model_name {
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

    // /v1/messages と同じ既定 (thinking = brief) を適用し、同一会話で同一出力にする。
    let system = brief_system(system);
    let enable_thinking = Some(false);
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

    if req.stream.unwrap_or(false) {
        stream_response(state, system, chat_messages, params, req.stream_options).await
    } else {
        let rx = state
            .scheduler
            .submit_chat(system, chat_messages, params, None, None, enable_thinking)
            .map_err(|_| queue_full(&state))?;
        let output = match tokio::time::timeout(RESPONSE_WAIT_TIMEOUT, rx).await {
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
                        RESPONSE_WAIT_TIMEOUT.as_secs_f64()
                    ),
                ))
            }
        };
        let created = now_secs();
        Ok(Json(json!({
            "id": new_chatcmpl_id(),
            "object": "chat.completion",
            "created": created,
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output.text},
                "logprobs": Value::Null,
                "finish_reason": finish_reason(output.stop_reason.as_str()),
            }],
            "usage": usage_json(output.input_tokens, output.output_tokens),
        }))
        .into_response())
    }
}

fn queue_full(state: &AppState) -> OaError {
    OaError::new(
        StatusCode::SERVICE_UNAVAILABLE,
        "server_error",
        format!(
            "server busy: request queue is full (max_queue = {}, in_flight = 1); retry later",
            state.scheduler.max_queue()
        ),
    )
}

/// SSE 応答。scheduler の生イベントを OpenAI chunk へ組み直す。
async fn stream_response(
    state: Arc<AppState>,
    system: Option<String>,
    chat_messages: Vec<ChatMessage>,
    params: GenParams,
    stream_options: Option<StreamOptions>,
) -> Result<Response, OaError> {
    let include_usage = stream_options.map(|o| o.include_usage).unwrap_or(false);
    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::channel::<RawEvent>(64);
    state
        .scheduler
        .submit_openai_stream(system, chat_messages, params, None, None, Some(false), raw_tx)
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
        while let Some(ev) = raw_rx.recv().await {
            match ev {
                RawEvent::Start { input_tokens: n } => input_tokens = n,
                RawEvent::Delta(text) => {
                    if text.is_empty() {
                        continue;
                    }
                    let chunk = json!({
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "logprobs": Value::Null,
                            "finish_reason": Value::Null,
                        }],
                    });
                    if tx.send(Ok(sse_data(&chunk))).await.is_err() {
                        return;
                    }
                }
                RawEvent::Done { input_tokens: i, output_tokens: o, stop_reason, .. } => {
                    input_tokens = i;
                    output_tokens = o;
                    reason = finish_reason(&stop_reason);
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
