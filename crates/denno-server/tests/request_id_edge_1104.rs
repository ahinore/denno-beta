//! task 1104: `x-request-id` と計測ヘッダのエッジケース。
//!
//! - 長すぎる ID は無害化と切り詰めを経て [`denno_server::MAX_REQUEST_ID_LEN`]
//!   byte に収まる (切り詰めは「残した文字数」で行う)。
//! - 引用符 / バックスラッシュ / 空白 / 制御 / 非 ASCII は落ちる。
//! - 無害化の結果が空なら生成 ID に切り替わる。
//! - 生成 ID は 64 並列でも衝突しない。
//! - 計測ヘッダは「完了した非ストリーミング JSON 生成応答」にだけ付き、
//!   `/health` `/ready` `/metrics` / SSE / NDJSON には付かない。

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

fn app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let info = EngineInfo::from_engine(&engine.lock().unwrap());
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: "denno-test".to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

/// 応答のヘッダと本文 (テキスト) を返す。ストリーミング応答は最後まで読む。
async fn send(app: &Router, req: Request<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

fn get(uri: &str) -> Request<Body> {
    Request::builder().method("GET").uri(uri).body(Body::empty()).unwrap()
}

fn get_with_rid(uri: &str, rid: HeaderValue) -> Request<Body> {
    let mut req = get(uri);
    req.headers_mut().insert("x-request-id", rid);
    req
}

fn post_json(uri: &str, body: &Value, rid: Option<HeaderValue>) -> Request<Body> {
    let mut b = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json");
    if let Some(rid) = rid {
        b = b.header("x-request-id", rid);
    }
    b.body(Body::from(body.to_string())).unwrap()
}

fn rid(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn header_f64(headers: &HeaderMap, name: &str) -> Option<f64> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<f64>().ok())
}

fn messages_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "system": "You are a test.",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn ollama_chat_body(stream: Option<bool>) -> Value {
    match stream {
        Some(s) => json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": s,
            "options": {"num_predict": 8},
        }),
        // `stream` を省くと Ollama 仕様どおり true (NDJSON)。
        None => json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "hi"}],
            "options": {"num_predict": 8},
        }),
    }
}

fn ollama_generate_body(stream: Option<bool>) -> Value {
    match stream {
        Some(s) => json!({
            "model": "denno-test",
            "prompt": "hi",
            "stream": s,
            "options": {"num_predict": 8},
        }),
        None => json!({
            "model": "denno-test",
            "prompt": "hi",
            "options": {"num_predict": 8},
        }),
    }
}

fn assert_generated_rid(id: &str) {
    assert!(id.starts_with("req-"), "生成 ID の形ではない: {id}");
    assert_eq!(id.len(), 24, "生成 ID の長さが違う: {id}");
    assert!(
        id.strip_prefix("req-")
            .unwrap()
            .chars()
            .all(|c| c.is_ascii_hexdigit()),
        "生成 ID が hex でない: {id}"
    );
}

// ---------------------------------------------------------------------------
// 1. 切り詰め

#[tokio::test]
async fn truncates_long_request_id_at_max_len() {
    let app = app();
    let long = "a".repeat(300);
    let (status, headers, _) =
        send(&app, get_with_rid("/health", HeaderValue::from_str(&long).unwrap())).await;
    assert_eq!(status, StatusCode::OK);
    let got = rid(&headers).expect("x-request-id が無い");
    assert_eq!(got.len(), denno_server::MAX_REQUEST_ID_LEN);
    assert_eq!(got, "a".repeat(denno_server::MAX_REQUEST_ID_LEN));
}

/// 切り詰めは「除去後の残存文字数」で判定する。落とす文字を挟んでも
/// 128 byte を超えず、後続の有効文字が枠を埋める。
#[tokio::test]
async fn truncation_counts_only_kept_bytes() {
    let app = app();
    // 100 文字 + 引用符 50 個 (落ちる) + b 100 個 → a*100 + b*28 が残る。
    let raw = format!("{}{}{}", "a".repeat(100), "\"".repeat(50), "b".repeat(100));
    let (status, headers, _) =
        send(&app, get_with_rid("/health", HeaderValue::from_str(&raw).unwrap())).await;
    assert_eq!(status, StatusCode::OK);
    let got = rid(&headers).expect("x-request-id が無い");
    let mut expected = "a".repeat(100);
    expected.push_str(&"b".repeat(28));
    assert_eq!(got, expected);
    assert_eq!(got.len(), denno_server::MAX_REQUEST_ID_LEN);
    assert!(!got.contains('"'));
}

/// 引用符だけ 300 個 → すべて落ちて生成 ID になる (長さで切られもしない)。
#[tokio::test]
async fn all_stripped_long_header_falls_back() {
    let app = app();
    let raw = "\"".repeat(300);
    let (_, headers, _) =
        send(&app, get_with_rid("/health", HeaderValue::from_str(&raw).unwrap())).await;
    let got = rid(&headers).expect("x-request-id が無い");
    assert_generated_rid(&got);
}

// ---------------------------------------------------------------------------
// 2. 除去

#[tokio::test]
async fn strips_quotes_backslashes_spaces_tabs_and_obs_text() {
    let app = app();
    // 空白 / TAB / 引用符 / バックスラッシュ / obs-text はヘッダ値としては
    // 通るが、ID としては残さない。
    let raw = HeaderValue::from_bytes(b"  ab\"c\\d\t e\xff\xfe").unwrap();
    let (status, headers, _) = send(&app, get_with_rid("/health", raw)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(rid(&headers).as_deref(), Some("abcde"));
}

#[test]
fn sanitize_rejects_control_only_and_non_ascii_only() {
    assert_eq!(denno_server::sanitize_request_id("\x00\x01\x1f\x7f"), None);
    assert_eq!(denno_server::sanitize_request_id(" \t \t"), None);
    assert_eq!(denno_server::sanitize_request_id("\"\\\\\""), None);
    // 非 ASCII (UTF-8 の複数 byte) はすべて落ちる。
    assert_eq!(denno_server::sanitize_request_id("\u{65e5}\u{672c}\u{8a9e}"), None);
    // 有効文字が 1 個でも残ればそれを返す。
    assert_eq!(denno_server::sanitize_request_id("a\x01b c").as_deref(), Some("abc"));
    assert_eq!(denno_server::sanitize_request_id("\"\x01~\"").as_deref(), Some("~"));
}

#[tokio::test]
async fn empty_sanitized_header_is_replaced_by_generated_id() {
    let app = app();
    for raw in [
        HeaderValue::from_bytes(b"\xff\xfe").unwrap(),
        HeaderValue::from_bytes(b"\" \\ \t\xff").unwrap(),
    ] {
        let (status, headers, _) = send(&app, get_with_rid("/health", raw)).await;
        assert_eq!(status, StatusCode::OK);
        let got = rid(&headers).expect("x-request-id が無い");
        assert_generated_rid(&got);
    }
}

// ---------------------------------------------------------------------------
// 3. 並列生成の一意性

#[test]
fn sixty_four_parallel_generated_ids_are_unique() {
    const N: usize = 64;
    let barrier = Arc::new(std::sync::Barrier::new(N));
    let mut handles = Vec::with_capacity(N);
    for _ in 0..N {
        let barrier = barrier.clone();
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            denno_server::generate_request_id()
        }));
    }
    let ids: Vec<String> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    for id in &ids {
        assert_generated_rid(id);
    }
    let unique: HashSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), N, "生成 ID が衝突した: {} / {N} 一意", unique.len());
}

/// 並列生成の ID 群は、同じプロセスの逐次生成と混ざっても単調増加の連番を持つ。
#[test]
fn parallel_ids_share_one_monotonic_sequence() {
    let mut ids = Vec::new();
    std::thread::scope(|s| {
        let mut hs = Vec::new();
        for _ in 0..16 {
            hs.push(s.spawn(|| {
                (0..4).map(|_| denno_server::generate_request_id()).collect::<Vec<_>>()
            }));
        }
        for h in hs {
            ids.extend(h.join().unwrap());
        }
    });
    assert_eq!(ids.len(), 64);
    let unique: HashSet<&String> = ids.iter().collect();
    assert_eq!(unique.len(), 64);
    // 末尾 8 hex は一意な連番なので、重複しないことを hex 値でも確認する。
    let seqs: HashSet<u32> = ids
        .iter()
        .map(|id| u32::from_str_radix(&id[id.len() - 8..], 16).unwrap())
        .collect();
    assert_eq!(seqs.len(), 64);
}

// ---------------------------------------------------------------------------
// 4. 計測ヘッダ: 非ストリーミング JSON 生成応答に付く

#[tokio::test]
async fn timing_headers_present_and_finite_on_non_streaming_json() {
    let app = app();
    let cases: Vec<(&str, Request<Body>)> = vec![
        ("anthropic /v1/messages", post_json("/v1/messages", &messages_body(false), None)),
        ("openai /v1/chat/completions", post_json("/v1/chat/completions", &chat_body(false), None)),
        (
            "openai /v1/completions",
            post_json(
                "/v1/completions",
                &json!({"model": "denno-test", "prompt": "hi", "max_tokens": 8}),
                None,
            ),
        ),
        ("ollama /api/chat", post_json("/api/chat", &ollama_chat_body(Some(false)), None)),
        (
            "ollama /api/generate",
            post_json("/api/generate", &ollama_generate_body(Some(false)), None),
        ),
    ];
    for (label, req) in cases {
        let (status, headers, body) = send(&app, req).await;
        assert!(status.is_success(), "{label}: {status} {body}");
        let ct = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.starts_with("application/json"), "{label}: content-type = {ct}");
        let ttft = header_f64(&headers, "x-denno-ttft-ms")
            .unwrap_or_else(|| panic!("{label}: x-denno-ttft-ms が無い"));
        let tps = header_f64(&headers, "x-denno-decode-tps")
            .unwrap_or_else(|| panic!("{label}: x-denno-decode-tps が無い"));
        assert!(ttft.is_finite(), "{label}: ttft = {ttft}");
        assert!(tps.is_finite(), "{label}: tps = {tps}");
        assert!(ttft >= 0.0, "{label}: ttft = {ttft}");
        assert!(tps > 0.0, "{label}: tps = {tps}");
        // 1 個のリクエスト ID も同時に付く。
        assert!(rid(&headers).is_some(), "{label}: x-request-id が無い");
        // 本文は計測ヘッダのために壊れない (JSON として読める)。
        let v: Value = serde_json::from_str(&body).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(
            v["usage"].is_object() || v["eval_count"].is_number(),
            "{label}: usage も eval_count も無い: {body}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. 計測ヘッダ: 付かない応答

#[tokio::test]
async fn no_timing_headers_on_ops_endpoints() {
    let app = app();
    for uri in ["/health", "/ready", "/metrics"] {
        let (_, headers, _) = send(&app, get(uri)).await;
        assert!(!headers.contains_key("x-denno-ttft-ms"), "{uri}: ttft が付いた");
        assert!(!headers.contains_key("x-denno-decode-tps"), "{uri}: tps が付いた");
        assert!(rid(&headers).is_some(), "{uri}: x-request-id が無い");
    }
}

#[tokio::test]
async fn no_timing_headers_on_sse_streams() {
    let app = app();
    // Anthropic SSE
    let resp = app
        .clone()
        .oneshot(post_json("/v1/messages", &messages_body(true), None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(!resp.headers().contains_key("x-denno-ttft-ms"));
    assert!(!resp.headers().contains_key("x-denno-decode-tps"));
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.starts_with("text/event-stream"), "content-type = {ct}");
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8_lossy(&body);
    assert!(text.contains("data:"), "SSE 本文ではない: {text}");

    // OpenAI SSE
    for (label, req) in [
        (
            "openai /v1/chat/completions",
            post_json("/v1/chat/completions", &chat_body(true), None),
        ),
        (
            "openai /v1/completions",
            post_json(
                "/v1/completions",
                &json!({"model": "denno-test", "prompt": "hi", "max_tokens": 8, "stream": true}),
                None,
            ),
        ),
    ] {
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::OK, "{label}: {body}");
        assert!(!headers.contains_key("x-denno-ttft-ms"), "{label}: ttft が付いた");
        assert!(!headers.contains_key("x-denno-decode-tps"), "{label}: tps が付いた");
        assert!(rid(&headers).is_some(), "{label}: x-request-id が無い");
    }
}

#[tokio::test]
async fn no_timing_headers_on_ollama_ndjson_streams() {
    let app = app();
    // stream 未指定 (既定 true) と stream:true の両方。
    let cases: Vec<(&str, Request<Body>)> = vec![
        ("/api/generate default", post_json("/api/generate", &ollama_generate_body(None), None)),
        (
            "/api/generate stream",
            post_json("/api/generate", &ollama_generate_body(Some(true)), None),
        ),
        ("/api/chat default", post_json("/api/chat", &ollama_chat_body(None), None)),
        ("/api/chat stream", post_json("/api/chat", &ollama_chat_body(Some(true)), None)),
    ];
    for (label, req) in cases {
        let (status, headers, body) = send(&app, req).await;
        assert_eq!(status, StatusCode::OK, "{label}: {body}");
        let ct = headers
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(ct.starts_with("application/x-ndjson"), "{label}: content-type = {ct}");
        assert!(!headers.contains_key("x-denno-ttft-ms"), "{label}: ttft が付いた");
        assert!(!headers.contains_key("x-denno-decode-tps"), "{label}: tps が付いた");
        assert!(rid(&headers).is_some(), "{label}: x-request-id が無い");
        // NDJSON は 1 行 1 JSON。
        let first = body.lines().next().unwrap_or("");
        assert!(
            serde_json::from_str::<Value>(first).is_ok(),
            "{label}: 先頭行が JSON ではない: {first}"
        );
    }
}

/// 計測ヘッダはエラー応答 (JSON でも usage 無し) には付かない。
#[tokio::test]
async fn no_timing_headers_on_json_errors() {
    let app = app();
    let req = post_json(
        "/v1/chat/completions",
        &json!({"model": "denno-test", "messages": [], "max_tokens": 0}),
        None,
    );
    let (status, headers, _) = send(&app, req).await;
    assert!(!status.is_success());
    assert!(!headers.contains_key("x-denno-ttft-ms"));
    assert!(!headers.contains_key("x-denno-decode-tps"));
    assert!(rid(&headers).is_some());
}
