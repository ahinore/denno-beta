//! Anthropic Messages API 互換 HTTP サーバー。
//! POST /v1/messages (非ストリーミング + SSE)、GET /v1/models、GET /health。

use anyhow::Result;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use denno_engine::{ChatMessage, ChatOutput, Engine, GenParams, QuantMode};
use rand::Rng;
use serde::Deserialize;
use serde_json::json;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio_stream::wrappers::ReceiverStream;

mod scheduler;
pub use scheduler::{Job, MAX_ACTIVE_SESSIONS, QueueFull, Scheduler};

pub struct AppState {
    pub engine: Arc<Mutex<Engine>>,
    pub api_key: Option<String>,
    /// P4: リクエストスケジューラ (有界キュー + 単一ワーカー)
    pub scheduler: Scheduler,
}

/// serve 設定
pub struct ServeConfig {
    pub model_dir: std::path::PathBuf,
    pub name: String,
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

    let engine = Arc::new(Mutex::new(engine));
    let state = Arc::new(AppState {
        engine: engine.clone(),
        api_key: cfg.api_key,
        scheduler: Scheduler::spawn(engine, cfg.max_queue.max(1)),
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
        // SSE 応答は即座に開始し、ジョブ実行 (キュー待ち含む) 中は
        // keep-alive ping (2秒間隔) が流れ続ける
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        state
            .scheduler
            .submit_stream(system, chat_messages, params, tx)
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
            .submit_chat(system, chat_messages, params)
            .map_err(queue_full_error(&state))?;
        let output = rx
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
        let model_name = state.engine.lock().unwrap().name.clone();
        Ok(Json(message_json(&model_name, &output)).into_response())
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
