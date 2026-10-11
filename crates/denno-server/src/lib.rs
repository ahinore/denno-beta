//! Anthropic Messages API 互換 HTTP サーバー。
//! POST /v1/messages (非ストリーミング + SSE)、GET /v1/models、GET /health。

use anyhow::Result;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use denno_engine::{ChatMessage, ChatOutput, Engine, GenParams};
use rand::Rng;
use serde::Deserialize;
use serde_json::json;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

pub struct AppState {
    pub engine: Mutex<Engine>,
    pub api_key: Option<String>,
}

/// serve 設定
pub struct ServeConfig {
    pub model_dir: std::path::PathBuf,
    pub name: String,
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub default_seed: Option<u64>,
}

// ---------- Anthropic 形式エラー ----------

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

#[derive(Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: usize,
    pub messages: Vec<InMessage>,
    pub system: Option<serde_json::Value>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    pub stream: Option<bool>,
}

#[derive(Deserialize)]
pub struct InMessage {
    pub role: String,
    pub content: serde_json::Value,
}

impl InMessage {
    /// content: 文字列 or [{type:"text", text:...}] の両対応
    fn to_text(&self) -> Result<String, ApiError> {
        match &self.content {
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Array(blocks) => {
                let mut out = String::new();
                for b in blocks {
                    match b.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            out.push_str(
                                b.get("text").and_then(|v| v.as_str()).unwrap_or_default(),
                            );
                        }
                        other => {
                            return Err(ApiError::invalid_request(format!(
                                "unsupported content block type: {:?} (text blocks only)",
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

// ---------- ルーター ----------

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/messages", post(messages))
        .with_state(state)
}

pub async fn run(cfg: ServeConfig) -> Result<()> {
    let model_dir = cfg.model_dir.clone();
    let name = cfg.name.clone();
    let seed = cfg.default_seed;
    tracing::info!("loading model from {:?} ...", model_dir);
    let engine = tokio::task::spawn_blocking(move || Engine::load(&model_dir, name, seed))
        .await
        .map_err(|e| anyhow::anyhow!("join error: {e}"))??;
    let ctx = engine.context();
    tracing::info!(
        "model loaded: {} (context = {}, engine = denno)",
        engine.name,
        ctx
    );

    let state = Arc::new(AppState {
        engine: Mutex::new(engine),
        api_key: cfg.api_key,
    });

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

    if req.model != state.engine.lock().unwrap().name {
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
            content: m.to_text()?,
        });
    }
    let system = system_to_text(&req.system)?;
    let params = GenParams {
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        stop_sequences: req.stop_sequences.clone(),
        seed: None,
    };

    if req.stream.unwrap_or(false) {
        stream_response(state, chat_messages, system, params).await
    } else {
        let output = run_engine(state.clone(), chat_messages, system, params)
            .await
            .map_err(|e| {
                let msg = format!("{e:#}");
                if msg.contains("invalid_request_error") {
                    ApiError::invalid_request(msg)
                } else {
                    ApiError::api(msg)
                }
            })?;
        let model_name = state.engine.lock().unwrap().name.clone();
        Ok(Json(message_json(&model_name, &output)).into_response())
    }
}

/// エンジン実行(spawn_blocking + Mutex。エンジンは逐次実行)
async fn run_engine(
    state: Arc<AppState>,
    chat_messages: Vec<ChatMessage>,
    system: Option<String>,
    params: GenParams,
) -> Result<ChatOutput, anyhow::Error> {
    tokio::task::spawn_blocking(move || {
        let mut g = state.engine.lock().unwrap();
        g.chat(
            system.as_deref(),
            &chat_messages,
            &params,
            |_| {}, // 非ストリーミング: 差分は捨てる
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("join error: {e}"))?
}

fn message_json(model: &str, out: &ChatOutput) -> serde_json::Value {
    json!({
        "id": new_msg_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": out.text}],
        "stop_reason": out.stop_reason.as_str(),
        "stop_sequence": out.stop_sequence,
        "usage": {
            "input_tokens": out.input_tokens,
            "output_tokens": out.output_tokens,
        }
    })
}

fn new_msg_id() -> String {
    let mut rng = rand::thread_rng();
    let id: String = (0..24).map(|_| format!("{:x}", rng.gen::<u8>() % 16)).collect();
    format!("msg_{id}")
}

// ---------- SSE ストリーミング ----------

async fn stream_response(
    state: Arc<AppState>,
    chat_messages: Vec<ChatMessage>,
    system: Option<String>,
    params: GenParams,
) -> Result<Response, ApiError> {
    let model_name = state.engine.lock().unwrap().name.clone();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::task::spawn_blocking(move || {
        let input_tokens = {
            // message_start 用に input tokens を事前計算するため、まずテンプレートだけ適用
            let mut g = state.engine.lock().unwrap();
            let prompt = g.tok.apply_chat_template(
                system.as_deref(),
                &chat_messages,
            );
            prompt.ok().and_then(|p| g.tok.encode(&p).ok()).map(|v| v.len()).unwrap_or(0)
        };

        let send = |v: serde_json::Value| -> bool { tx.blocking_send(Ok(to_event(&v))).is_ok() };

        // 1. message_start
        if !send(json!({
            "type": "message_start",
            "message": {
                "id": new_msg_id(),
                "type": "message",
                "role": "assistant",
                "model": model_name,
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
        let mut output_tokens = 0usize;
        let mut stop_reason = "max_tokens";
        let mut stop_sequence: Option<String> = None;
        let run = {
            let mut g = state.engine.lock().unwrap();
            g.chat(system.as_deref(), &chat_messages, &params, |delta| {
                let _ = send(json!({
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": delta}
                }));
            })
        };
        match run {
            Ok(out) => {
                output_tokens = out.output_tokens;
                stop_reason = out.stop_reason.as_str();
                stop_sequence = out.stop_sequence;
            }
            Err(e) => {
                // エラーは error イベントで通知して閉じる
                let _ = send(json!({
                    "type": "error",
                    "error": {"type": "api_error", "message": format!("{e:#}")}
                }));
                return;
            }
        }

        // 4. content_block_stop
        if !send(json!({"type": "content_block_stop", "index": 0})) {
            return;
        }
        // 5. message_delta
        let _ = send(json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": output_tokens}
        }));
        // 6. message_stop
        let _ = send(json!({"type": "message_stop"}));
    });

    let stream = ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn to_event(v: &serde_json::Value) -> Event {
    Event::default().data(v.to_string())
}
