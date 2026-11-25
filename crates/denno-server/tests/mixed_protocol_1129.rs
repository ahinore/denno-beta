//! task 1129: 全プロトコル同時混在負荷の統合テスト。
//!
//! 1 つのサーバー (1 つの Scheduler / 1 つの KV) に対し、Anthropic /
//! OpenAI / Ollama / 運用プローブの全系統を **同時に** 投げ、次を固定する:
//!
//! - すべてのリクエストが 200 で完走し、各プロトコルのフレーミングが壊れない
//!   (Anthropic SSE のイベント順 / OpenAI SSE の `[DONE]` / Ollama の NDJSON
//!   / 各 JSON 本文の形)。
//! - 全応答 (SSE / NDJSON / 運用プローブを含む) に `x-request-id` が付き、
//!   生成 ID がリクエスト間で衝突しない。
//! - 混在負荷の間に実際に 2 本以上が同時進行している (同時性の証拠)。
//! - 完了後、スケジューラの in_flight / pending、`/metrics` のゲージ、
//!   `/v1/cache/stats` の busy スロットがすべて厳密に 0 に戻る。
//!
//! モデル / トークナイザは既存の統合テストと同じ決定論ダミー (CPU のみ)。
//! 固定 sleep での待ち合わせはせず、状態は期限付きポーリングで観測する。

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{HeaderMap, Method, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const CONTEXT: usize = 512;
/// 待ち合わせの上限 (deadlock を失敗として表面化させる)。
const WAIT: Duration = Duration::from_secs(60);
const POLL: Duration = Duration::from_millis(5);
/// 1 種別あたりの同時本数。16 種別 × 3 = 48 本を一斉に投げる。
const PER_KIND: usize = 3;

// ---------------------------------------------------------------------------
// 決定的モデル (小さな遅延で並行性を確実にする)
// ---------------------------------------------------------------------------

struct DelayModel {
    delay: Duration,
}

impl Model for DelayModel {
    fn context(&self) -> usize {
        CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = (last + 1) % DUMMY_VOCAB;
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, CONTEXT)
    }
}

/// 48 本を同時に受けるため待ち行列を十分広く取る (MAX_ACTIVE_SESSIONS = 4)。
fn state_with(model: Box<dyn Model>, max_queue: usize) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), max_queue);
    Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    })
}

// ---------------------------------------------------------------------------
// リクエスト定義
// ---------------------------------------------------------------------------

/// 全種別 (16 個)。`spec` が (method, uri, body) を返す。
const KINDS: [&str; 16] = [
    "anthropic_json",
    "anthropic_sse",
    "count_tokens",
    "openai_chat_json",
    "openai_chat_sse",
    "openai_completions_json",
    "models",
    "ollama_chat_json",
    "ollama_chat_stream",
    "ollama_generate_json",
    "ollama_generate_stream",
    "tags",
    "health",
    "ready",
    "metrics",
    "cache_stats",
];

fn anthropic_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn openai_chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn ollama_chat_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "options": {"temperature": 0.0, "num_predict": 8},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

fn ollama_generate_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": "denno-test",
        "prompt": "t0",
        "options": {"temperature": 0.0, "num_predict": 8},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

fn spec(kind: &str) -> (Method, &'static str, Option<Value>) {
    match kind {
        "anthropic_json" => (Method::POST, "/v1/messages", Some(anthropic_body(false))),
        "anthropic_sse" => (Method::POST, "/v1/messages", Some(anthropic_body(true))),
        "count_tokens" => (
            Method::POST,
            "/v1/messages/count_tokens",
            Some(json!({
                "model": "denno-test",
                "messages": [{"role": "user", "content": "t0"}],
            })),
        ),
        "openai_chat_json" => (Method::POST, "/v1/chat/completions", Some(openai_chat_body(false))),
        "openai_chat_sse" => (Method::POST, "/v1/chat/completions", Some(openai_chat_body(true))),
        "openai_completions_json" => (
            Method::POST,
            "/v1/completions",
            Some(json!({
                "model": "denno-test",
                "prompt": "t0 t1 t2",
                "max_tokens": 4,
                "temperature": 0.0,
            })),
        ),
        "models" => (Method::GET, "/v1/models", None),
        "ollama_chat_json" => (Method::POST, "/api/chat", Some(ollama_chat_body(Some(false)))),
        "ollama_chat_stream" => (Method::POST, "/api/chat", Some(ollama_chat_body(None))),
        "ollama_generate_json" => (
            Method::POST,
            "/api/generate",
            Some(ollama_generate_body(Some(false))),
        ),
        "ollama_generate_stream" => (
            Method::POST,
            "/api/generate",
            Some(ollama_generate_body(None)),
        ),
        "tags" => (Method::GET, "/api/tags", None),
        "health" => (Method::GET, "/health", None),
        "ready" => (Method::GET, "/ready", None),
        "metrics" => (Method::GET, "/metrics", None),
        "cache_stats" => (Method::GET, "/v1/cache/stats", None),
        other => panic!("未知の kind: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 1 応答
// ---------------------------------------------------------------------------

struct Resp {
    kind: &'static str,
    status: StatusCode,
    rid: Option<String>,
    content_type: String,
    body: String,
}

fn header_str(headers: &HeaderMap, name: &str) -> String {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string()
}

async fn send(app: &Router, kind: &'static str) -> Resp {
    let (method, uri, body) = spec(kind);
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    let req = builder
        .body(Body::from(body.map(|b| b.to_string()).unwrap_or_default()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let content_type = header_str(&headers, "content-type");
    let rid = headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    // ストリーミング応答も最後まで読む (= 完走を待つ)。
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp {
        kind,
        status,
        rid,
        content_type,
        body: String::from_utf8_lossy(&bytes).to_string(),
    }
}

// ---------------------------------------------------------------------------
// プロトコル別フレーミング検証
// ---------------------------------------------------------------------------

fn json_of(r: &Resp) -> Value {
    serde_json::from_str(&r.body)
        .unwrap_or_else(|e| panic!("[{}] 本文が JSON ではない ({e}): {:?}", r.kind, r.body))
}

fn sse_data_jsons(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| {
            serde_json::from_str(d)
                .unwrap_or_else(|e| panic!("SSE data が JSON ではない ({e}): {d:?}"))
        })
        .collect()
}

fn ndjson_lines(body: &str) -> Vec<Value> {
    assert!(!body.is_empty(), "NDJSON 本文が空");
    assert!(body.ends_with('\n'), "NDJSON が改行で終わっていない: {body:?}");
    assert_eq!(
        body.matches('\n').count(),
        body.lines().count(),
        "NDJSON に空行が混ざる (フレーミング破れ): {body:?}"
    );
    body.lines()
        .map(|l| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("NDJSON 行が JSON ではない ({e}): {l:?}"))
        })
        .collect()
}

fn validate(r: &Resp) {
    assert_eq!(r.status, StatusCode::OK, "[{}] status={} body={}", r.kind, r.status, r.body);
    assert!(
        r.rid.as_deref().map(|s| !s.is_empty()).unwrap_or(false),
        "[{}] x-request-id が無い",
        r.kind
    );

    match r.kind {
        "anthropic_json" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["type"], "message", "[{}] body={v}", r.kind);
            assert!(v["id"].as_str().unwrap_or("").starts_with("msg_"), "[{}] id={v}", r.kind);
            assert!(v["content"].is_array(), "[{}] content={v}", r.kind);
            assert!(v["stop_reason"].is_string(), "[{}] body={v}", r.kind);
            assert!(v["usage"]["input_tokens"].is_u64(), "[{}] usage={v}", r.kind);
            assert!(v["usage"]["output_tokens"].is_u64(), "[{}] usage={v}", r.kind);
        }
        "anthropic_sse" => {
            assert!(r.content_type.starts_with("text/event-stream"), "[{}] ct={}", r.kind, r.content_type);
            let events: Vec<&str> = r
                .body
                .lines()
                .filter_map(|l| l.strip_prefix("event: "))
                .collect();
            assert!(!events.is_empty(), "[{}] SSE イベントが無い: {r:?}", r.kind, r = r.body);
            assert_eq!(events.first().copied(), Some("message_start"), "[{}] 先頭イベント: {events:?}", r.kind);
            assert_eq!(events.last().copied(), Some("message_stop"), "[{}] 末尾イベント: {events:?}", r.kind);
            assert!(events.contains(&"content_block_delta"), "[{}] delta が無い: {events:?}", r.kind);
            // data 行はすべて JSON として読める。
            let data: Vec<&str> = r
                .body
                .lines()
                .filter_map(|l| l.strip_prefix("data: "))
                .collect();
            assert!(!data.is_empty(), "[{}] data 行が無い", r.kind);
            for d in data {
                serde_json::from_str::<Value>(d)
                    .unwrap_or_else(|e| panic!("[{}] SSE data が JSON ではない ({e}): {d:?}", r.kind));
            }
        }
        "count_tokens" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert!(v["input_tokens"].as_u64().unwrap_or(0) > 0, "[{}] body={v}", r.kind);
        }
        "openai_chat_json" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["object"], "chat.completion", "[{}] body={v}", r.kind);
            assert!(v["id"].as_str().unwrap_or("").starts_with("chatcmpl"), "[{}] id={v}", r.kind);
            assert!(v["choices"][0]["message"]["content"].is_string(), "[{}] body={v}", r.kind);
            let usage = &v["usage"];
            let p = usage["prompt_tokens"].as_u64().expect("prompt_tokens");
            let c = usage["completion_tokens"].as_u64().expect("completion_tokens");
            assert_eq!(usage["total_tokens"].as_u64(), Some(p + c), "[{}] usage={usage}", r.kind);
        }
        "openai_chat_sse" => {
            assert!(r.content_type.starts_with("text/event-stream"), "[{}] ct={}", r.kind, r.content_type);
            assert!(r.body.trim_end().ends_with("data: [DONE]"), "[{}] 終端が [DONE] でない", r.kind);
            let chunks = sse_data_jsons(&r.body);
            assert!(chunks.len() >= 2, "[{}] chunk が少なすぎる", r.kind);
            for c in &chunks {
                assert_eq!(c["object"], "chat.completion.chunk", "[{}] chunk={c}", r.kind);
            }
            let finish = chunks
                .iter()
                .find_map(|c| c["choices"][0]["finish_reason"].as_str())
                .expect("finish_reason が無い");
            assert!(!finish.is_empty(), "[{}] finish_reason が空", r.kind);
        }
        "openai_completions_json" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["object"], "text_completion", "[{}] body={v}", r.kind);
            assert!(v["choices"][0]["text"].is_string(), "[{}] body={v}", r.kind);
            assert!(v["usage"]["total_tokens"].is_u64(), "[{}] usage={v}", r.kind);
        }
        "models" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["object"], "list", "[{}] body={v}", r.kind);
            let data = v["data"].as_array().expect("data が配列でない");
            assert_eq!(data.len(), 1, "[{}] data={v}", r.kind);
            assert_eq!(data[0]["id"], "denno-test", "[{}] data={v}", r.kind);
        }
        "ollama_chat_json" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["done"], true, "[{}] body={v}", r.kind);
            assert_eq!(v["message"]["role"], "assistant", "[{}] body={v}", r.kind);
            assert!(v["message"]["content"].is_string(), "[{}] body={v}", r.kind);
            assert!(v["eval_count"].as_u64().unwrap_or(0) > 0, "[{}] body={v}", r.kind);
        }
        "ollama_chat_stream" => {
            assert!(r.content_type.starts_with("application/x-ndjson"), "[{}] ct={}", r.kind, r.content_type);
            let lines = ndjson_lines(&r.body);
            assert!(lines.len() >= 2, "[{}] 行が少なすぎる", r.kind);
            for l in &lines[..lines.len() - 1] {
                assert_eq!(l["done"], false, "[{}] 途中行が done:true: {l}", r.kind);
                assert_eq!(l["message"]["role"], "assistant", "[{}] 途中行: {l}", r.kind);
            }
            let last = lines.last().unwrap();
            assert_eq!(last["done"], true, "[{}] 最終行が done:true でない: {last}", r.kind);
            assert!(last["eval_count"].as_u64().unwrap_or(0) > 0, "[{}] 最終行: {last}", r.kind);
        }
        "ollama_generate_json" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["done"], true, "[{}] body={v}", r.kind);
            assert!(v["response"].is_string(), "[{}] body={v}", r.kind);
            assert!(v["eval_count"].as_u64().unwrap_or(0) > 0, "[{}] body={v}", r.kind);
        }
        "ollama_generate_stream" => {
            assert!(r.content_type.starts_with("application/x-ndjson"), "[{}] ct={}", r.kind, r.content_type);
            let lines = ndjson_lines(&r.body);
            assert!(lines.len() >= 2, "[{}] 行が少なすぎる", r.kind);
            for l in &lines[..lines.len() - 1] {
                assert_eq!(l["done"], false, "[{}] 途中行: {l}", r.kind);
                assert!(l.get("message").is_none(), "[{}] generate 行に message: {l}", r.kind);
            }
            let last = lines.last().unwrap();
            assert_eq!(last["done"], true, "[{}] 最終行: {last}", r.kind);
            assert!(last["eval_count"].as_u64().unwrap_or(0) > 0, "[{}] 最終行: {last}", r.kind);
        }
        "tags" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            let models = v["models"].as_array().expect("models が配列でない");
            assert!(!models.is_empty(), "[{}] models が空", r.kind);
            assert_eq!(models[0]["name"], "denno-test", "[{}] body={v}", r.kind);
        }
        "health" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["status"], "ok", "[{}] body={v}", r.kind);
        }
        "ready" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            assert_eq!(v["status"], "ready", "[{}] body={v}", r.kind);
            assert_eq!(v["ready"], true, "[{}] body={v}", r.kind);
            assert_eq!(v["draining"], false, "[{}] body={v}", r.kind);
            assert!(v["pending"].is_u64() && v["in_flight"].is_u64(), "[{}] body={v}", r.kind);
        }
        "metrics" => {
            assert!(r.content_type.starts_with("text/plain"), "[{}] ct={}", r.kind, r.content_type);
            assert!(r.body.contains("# TYPE denno_requests_in_flight gauge"), "[{}] ゲージが無い", r.kind);
            assert!(r.body.contains("denno_requests_in_flight "), "[{}] ゲージ値が無い", r.kind);
            assert!(r.body.contains("denno_http_requests_total"), "[{}] HTTP カウンタが無い", r.kind);
        }
        "cache_stats" => {
            assert!(r.content_type.starts_with("application/json"), "[{}] ct={}", r.kind, r.content_type);
            let v = json_of(r);
            for key in ["slots_used", "slots_capacity", "hits", "misses", "hit_rate", "kv_slots", "kv_slots_busy"] {
                assert!(v.get(key).is_some(), "[{}] {key} が無い: {v}", r.kind);
            }
            assert!(v["kv_slots"].as_u64().unwrap_or(0) > 0, "[{}] kv_slots={v}", r.kind);
        }
        other => panic!("未知の kind: {other}"),
    }
}

// ---------------------------------------------------------------------------
// 観測ヘルパ
// ---------------------------------------------------------------------------

fn gauge(text: &str, name: &str) -> u64 {
    let prefix = format!("{name} ");
    text.lines()
        .find_map(|l| l.strip_prefix(prefix.as_str()))
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or_else(|| panic!("{name} が /metrics に無い"))
}

async fn get_json(app: &Router, uri: &str) -> Value {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "{uri}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap_or(Value::Null)
}

async fn get_text(app: &Router, uri: &str) -> String {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "{uri}");
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    String::from_utf8_lossy(&bytes).to_string()
}

// ---------------------------------------------------------------------------
// 本体
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn all_protocols_concurrent_mixed_workload_is_clean() {
    let state = state_with(Box::new(DelayModel { delay: Duration::from_millis(1) }), 128);
    let app = denno_server::router(state.clone());

    // 同時性の観測 (2 本以上が本当に同時進行していることを確かめる)。
    let max_in_flight = Arc::new(AtomicUsize::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let monitor = {
        let state = state.clone();
        let max_in_flight = max_in_flight.clone();
        let stop = stop.clone();
        tokio::spawn(async move {
            while !stop.load(Ordering::SeqCst) {
                max_in_flight.fetch_max(state.scheduler.in_flight(), Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
    };

    // 16 種別 × 3 本を一斉に投げる。
    let mut handles = Vec::new();
    for _ in 0..PER_KIND {
        for kind in KINDS {
            let app = app.clone();
            handles.push(tokio::spawn(async move { send(&app, kind).await }));
        }
    }

    let mut responses = Vec::with_capacity(handles.len());
    for (i, h) in handles.into_iter().enumerate() {
        let r = tokio::time::timeout(WAIT, h)
            .await
            .unwrap_or_else(|_| panic!("リクエスト {i} が {WAIT:?} 以内に完了しない"))
            .unwrap();
        validate(&r);
        responses.push(r);
    }
    stop.store(true, Ordering::SeqCst);
    let _ = monitor.await;

    assert_eq!(responses.len(), KINDS.len() * PER_KIND);

    // 全応答の x-request-id が生成 ID で、リクエスト間で衝突しない。
    let ids: Vec<String> = responses
        .iter()
        .map(|r| r.rid.clone().expect("x-request-id"))
        .collect();
    for id in &ids {
        assert!(id.starts_with("req-"), "生成 ID の形ではない: {id}");
    }
    let unique: HashSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), ids.len(), "x-request-id が衝突した: {} / {}", unique.len(), ids.len());

    // 実際に並行していた (単なる直列実行では混在負荷を検証できない)。
    let peak = max_in_flight.load(Ordering::SeqCst);
    assert!(peak >= 2, "2 本以上が同時進行していない: max in_flight = {peak}");

    // 完了後、厳密に idle へ戻る (期限付きポーリング)。
    let deadline = Instant::now() + WAIT;
    loop {
        if state.scheduler.pending() == 0 && state.scheduler.in_flight() == 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "idle に戻らない: pending={} in_flight={}",
            state.scheduler.pending(),
            state.scheduler.in_flight()
        );
        tokio::time::sleep(POLL).await;
    }

    // スケジューラが直接見る値。
    assert_eq!(state.scheduler.in_flight(), 0, "in_flight が 0 でない");
    assert_eq!(state.scheduler.pending(), 0, "pending が 0 でない");

    // /metrics のゲージも 0。完了カウンタは進んでいる。
    let metrics = get_text(&app, "/metrics").await;
    assert_eq!(gauge(&metrics, "denno_requests_in_flight"), 0, "metrics in_flight");
    assert_eq!(gauge(&metrics, "denno_requests_queued"), 0, "metrics queued");
    assert!(
        gauge(&metrics, "denno_requests_completed_total") > 0,
        "完了カウンタが進んでいない (生成経路を通っていない)"
    );

    // busy スロットも 0 (kv_slots は 0 より大きい)。
    let stats = get_json(&app, "/v1/cache/stats").await;
    assert!(stats["kv_slots"].as_u64().unwrap_or(0) > 0, "kv_slots={stats}");
    assert_eq!(stats["kv_slots_busy"].as_u64(), Some(0), "busy スロットが残っている: {stats}");

    // /ready の自己申告も idle。
    let ready = get_json(&app, "/ready").await;
    assert_eq!(ready["status"], "ready", "ready={ready}");
    assert_eq!(ready["pending"], 0, "ready={ready}");
    assert_eq!(ready["in_flight"], 0, "ready={ready}");
}
