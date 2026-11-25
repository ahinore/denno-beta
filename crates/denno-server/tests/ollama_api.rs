//! P841: Ollama 互換 API (`/api/tags` / `/api/version` / `/api/chat` /
//! `/api/generate`) の統合テスト。
//!
//! denno-engine のダミートークナイザ + 決定的ダミーモデル (tests/openai_chat.rs と
//! 同一方式) を使い、HTTP ワイヤレベル (axum Router + tower oneshot) と実 TCP
//! ソケットで検証する。カバー項目:
//!
//! - `/api/tags` / `/api/version` の形
//! - `/api/chat` 非ストリームが `/v1/messages` と本文・トークン数一致
//! - `/api/chat` の NDJSON 既定 (stream 未指定 = true) の delta 結合 = 非ストリーム
//! - `/api/generate` 非ストリーム / ストリーム、`options.num_predict` / `options.stop`
//! - `raw: true` が chat template を通さない
//! - 400 (messages 空 / prompt 空) と 404 (未知モデル) が `{"error": ...}` 形式
//! - 実 TCP のクライアント切断で in_flight が 0 に戻る

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 512;

/// 決定的ダミーモデル (peak = 直前トークン + 1、EOS の位置だけ避ける)。
struct SeqModel {
    delay: Duration,
    endless: bool,
}

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        if self.endless {
            &[]
        } else {
            &[DUMMY_EOS]
        }
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
        let peak = if self.endless && peak == DUMMY_EOS as usize { peak + 1 } else { peak };
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn app_with(delay: Duration, endless: bool) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(
        Box::new(SeqModel { delay, endless }),
        tok,
        "denno-test",
        None,
    );
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
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

fn test_app() -> Router {
    denno_server::router(app_with(Duration::ZERO, false))
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, v)
}

/// NDJSON 応答を取得する (content-type も返す)。
async fn post_ndjson(app: &Router, uri: &str, body: Value) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ct, String::from_utf8_lossy(&bytes).to_string())
}

/// NDJSON 本文を 1 行 1 JSON に分解する。
fn ndjson(text: &str) -> Vec<Value> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("NDJSON 行が JSON ではない"))
        .collect()
}

fn chat_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "options": {"temperature": 0, "num_predict": 32},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

fn messages_body() -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 32,
        "temperature": 0.0,
    })
}

// ---------------------------------------------------------------------------

#[tokio::test]
async fn tags_and_version_have_ollama_shape() {
    let app = test_app();
    let (status, v) = get(&app, "/api/tags").await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let m = &v["models"][0];
    assert_eq!(m["name"], "denno-test");
    assert_eq!(m["model"], "denno-test");
    assert_eq!(m["size"], 0);
    assert!(m["modified_at"].as_str().unwrap().contains('T'));
    assert!(m["modified_at"].as_str().unwrap().ends_with('Z'));
    assert_eq!(m["digest"].as_str().unwrap().len(), 64);
    assert_eq!(m["details"]["format"], "gguf");
    assert_eq!(m["details"]["family"], "denno");

    let (status, v) = get(&app, "/api/version").await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["version"], "0.1.0-denno");
}

#[tokio::test]
async fn chat_non_stream_matches_messages() {
    let app = test_app();
    let (status, v) = post(&app, "/api/chat", chat_body(Some(false))).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["model"], "denno-test");
    assert_eq!(v["done"], true);
    assert_eq!(v["message"]["role"], "assistant");
    let text = v["message"]["content"].as_str().unwrap().to_string();
    assert!(!text.is_empty(), "content が空: {v}");
    let prompt = v["prompt_eval_count"].as_u64().unwrap();
    let eval = v["eval_count"].as_u64().unwrap();
    assert!(prompt > 0);
    assert!(eval > 0);
    assert!(v["done_reason"] == "stop" || v["done_reason"] == "length");
    assert_eq!(v["load_duration"], 0);

    // /v1/messages と同じ scheduler / template / thinking 既定を通るので本文が一致する
    let (status, an) = post(&app, "/v1/messages", messages_body()).await;
    assert_eq!(status, StatusCode::OK, "body = {an}");
    assert_eq!(an["content"][0]["text"], text);
    assert_eq!(an["usage"]["input_tokens"].as_u64().unwrap(), prompt);
    assert_eq!(an["usage"]["output_tokens"].as_u64().unwrap(), eval);
}

#[tokio::test]
async fn chat_streams_ndjson_by_default() {
    let app = test_app();
    let (_, non) = post(&app, "/api/chat", chat_body(Some(false))).await;
    let want = non["message"]["content"].as_str().unwrap().to_string();

    // stream 未指定 = Ollama 既定の true
    let (status, ct, body) = post_ndjson(&app, "/api/chat", chat_body(None)).await;
    assert_eq!(status, StatusCode::OK, "body = {body}");
    assert!(ct.starts_with("application/x-ndjson"), "content-type = {ct}");
    let lines = ndjson(&body);
    assert!(lines.len() >= 2, "行が少なすぎる: {lines:?}");
    // 全行が同じ model / created_at、done は最後の 1 行だけ
    for l in &lines {
        assert_eq!(l["model"], "denno-test");
        assert_eq!(l["created_at"], lines[0]["created_at"]);
    }
    let done: Vec<&Value> = lines.iter().filter(|l| l["done"] == true).collect();
    assert_eq!(done.len(), 1, "done:true 行が 1 本でない: {lines:?}");
    assert_eq!(lines.last().unwrap()["done"], true);
    // delta の連結 = 非ストリーム本文
    let joined: String = lines
        .iter()
        .filter(|l| l["done"] == false)
        .filter_map(|l| l["message"]["content"].as_str())
        .collect();
    assert_eq!(joined, want);
    // 終端行のトークン数が非ストリームと一致
    assert_eq!(done[0]["prompt_eval_count"], non["prompt_eval_count"]);
    assert_eq!(done[0]["eval_count"], non["eval_count"]);
    assert_eq!(done[0]["done_reason"], non["done_reason"]);
}

#[tokio::test]
async fn generate_non_stream_and_stream_agree() {
    let app = test_app();
    let body = json!({
        "model": "denno-test",
        "prompt": "hello",
        "system": "You are a test.",
        "stream": false,
        "options": {"temperature": 0, "num_predict": 32},
    });
    let (status, v) = post(&app, "/api/generate", body.clone()).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["done"], true);
    let full = v["response"].as_str().unwrap().to_string();
    assert!(!full.is_empty(), "response が空: {v}");

    // 同じ system + user の chat 経路なので /api/chat と本文が一致する
    let (_, c) = post(&app, "/api/chat", chat_body(Some(false))).await;
    assert_eq!(c["message"]["content"].as_str().unwrap(), full);

    // ストリーム (既定 true) の連結も一致
    let mut sbody = body.clone();
    sbody["stream"] = Value::Null;
    let (status, ct, text) = post_ndjson(&app, "/api/generate", sbody).await;
    assert_eq!(status, StatusCode::OK, "body = {text}");
    assert!(ct.starts_with("application/x-ndjson"), "content-type = {ct}");
    let lines = ndjson(&text);
    let joined: String = lines
        .iter()
        .filter(|l| l["done"] == false)
        .filter_map(|l| l["response"].as_str())
        .collect();
    assert_eq!(joined, full);
    assert_eq!(lines.last().unwrap()["done"], true);
    assert_eq!(lines.last().unwrap()["eval_count"], v["eval_count"]);
}

#[tokio::test]
async fn generate_raw_bypasses_chat_template() {
    let app = test_app();
    let body = json!({
        "model": "denno-test",
        "prompt": "t0 t1",
        "raw": true,
        "stream": false,
        "options": {"temperature": 0, "num_predict": 4},
    });
    let (status, v) = post(&app, "/api/generate", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let raw = v["response"].as_str().unwrap();
    // chat template を通すと system 包みの前置きで prompt が伸びる。
    // raw はプロンプトをそのままトークン化するので、同じ 4 トークン生成でも
    // prompt_eval_count が chat 経路より小さい。
    let (_, c) = post(&app, "/api/chat", chat_body(Some(false))).await;
    assert!(
        v["prompt_eval_count"].as_u64().unwrap() < c["prompt_eval_count"].as_u64().unwrap(),
        "raw 経路が prompt を包んでいる: {v} vs {c}"
    );
    assert!(!raw.is_empty());

    let (_, an) = post(
        &app,
        "/v1/completions",
        json!({"model": "denno-test", "prompt": "t0 t1", "max_tokens": 4, "temperature": 0.0}),
    )
    .await;
    assert_eq!(an["choices"][0]["text"], raw);
}

#[tokio::test]
async fn options_num_predict_and_stop_are_applied() {
    let app = test_app();
    // num_predict = 2 → eval_count 2 / done_reason length
    let mut body = chat_body(Some(false));
    body["options"]["num_predict"] = json!(2);
    let (status, v) = post(&app, "/api/chat", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(v["eval_count"], 2);
    assert_eq!(v["done_reason"], "length");

    // stop (配列) で生成が切れ、本文に残らない
    let (_, full) = post(&app, "/api/chat", chat_body(Some(false))).await;
    let text = full["message"]["content"].as_str().unwrap().to_string();
    let last_word = text.split_whitespace().last().expect("生成本文が空").to_string();
    let mut body = chat_body(Some(false));
    body["options"]["stop"] = json!([last_word]);
    let (status, v) = post(&app, "/api/chat", body).await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    let stopped = v["message"]["content"].as_str().unwrap();
    assert!(
        !stopped.contains(&last_word),
        "stop 語 {last_word:?} が本文に残っている: {stopped:?}"
    );
    assert_eq!(v["done_reason"], "stop");
}

#[tokio::test]
async fn errors_are_ollama_shaped() {
    let app = test_app();

    // messages 空 → 400
    let mut body = chat_body(Some(false));
    body["messages"] = json!([]);
    let (status, v) = post(&app, "/api/chat", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("messages"), "body = {v}");

    // 未知モデル → 404
    let mut body = chat_body(Some(false));
    body["model"] = json!("nope:latest");
    let (status, v) = post(&app, "/api/chat", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(v["error"].as_str().unwrap().contains("nope"), "body = {v}");

    // prompt 空 → 400
    let (status, v) = post(
        &app,
        "/api/generate",
        json!({"model": "denno-test", "prompt": "", "stream": false}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("prompt"), "body = {v}");

    // 未知ロール → 400
    let (status, v) = post(
        &app,
        "/api/chat",
        json!({"model": "denno-test", "messages": [{"role": "tool", "content": "x"}], "stream": false}),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(v["error"].as_str().unwrap().contains("role"), "body = {v}");

    // 壊れた JSON → 400
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header("content-type", "application/json")
        .body(Body::from("{\"messages\": "))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ---------- 実 TCP: クライアント切断 ----------

async fn serve_tcp(state: Arc<AppState>) -> std::net::SocketAddr {
    let app: Router = denno_server::router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    addr
}

async fn read_until(
    stream: &mut tokio::net::TcpStream,
    buf: &mut String,
    marker: &str,
    secs: u64,
) -> bool {
    use tokio::io::AsyncReadExt;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    let mut tmp = [0u8; 4096];
    while !buf.contains(marker) {
        let remain = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remain.is_zero() {
            return false;
        }
        match tokio::time::timeout(remain, stream.read(&mut tmp)).await {
            Ok(Ok(0)) | Ok(Err(_)) | Err(_) => return false,
            Ok(Ok(n)) => buf.push_str(&String::from_utf8_lossy(&tmp[..n])),
        }
    }
    true
}

async fn wait_in_flight(state: &AppState, want: usize, secs: u64) {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(secs);
    while tokio::time::Instant::now() < deadline {
        if state.scheduler.in_flight() == want {
            return;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!(
        "in_flight が {want} にならない (現在 {})",
        state.scheduler.in_flight()
    );
}

/// NDJSON ストリーム中のクライアント切断で Run が打ち切られる
/// (ボディ drop → raw 受信端 drop → scheduler の送信失敗で abort)。
#[tokio::test]
async fn client_disconnect_aborts_ollama_stream() {
    use tokio::io::AsyncWriteExt;

    let state = app_with(Duration::from_millis(40), true);
    let addr = serve_tcp(state.clone()).await;
    let body = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
        "stream": true,
        "options": {"temperature": 0, "num_predict": 400},
    })
    .to_string();

    let mut s = tokio::net::TcpStream::connect(addr).await.unwrap();
    let req = format!(
        "POST /api/chat HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    s.write_all(req.as_bytes()).await.unwrap();
    s.flush().await.unwrap();

    let mut buf = String::new();
    assert!(
        read_until(&mut s, &mut buf, "\"done\":false", 10).await,
        "最初の NDJSON 行が届かない: {buf}"
    );
    wait_in_flight(&state, 1, 5).await;
    drop(s); // 切断

    wait_in_flight(&state, 0, 5).await;

    // 後続 (非ストリーム) が即座に完了する = ワーカー / KV スロットは健常
    let (status, v) = post(
        &denno_server::router(state.clone()),
        "/api/chat",
        json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "t0"}],
            "stream": false,
            "options": {"temperature": 0, "num_predict": 4},
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body = {v}");
    assert_eq!(state.scheduler.in_flight(), 0);
}
