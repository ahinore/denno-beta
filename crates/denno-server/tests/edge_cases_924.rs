//! P924: マルチプロトコルの異常系・境界値テスト。
//!
//! 対象は 5 経路:
//! `/v1/chat/completions`・`/v1/completions` (OpenAI)、`/v1/messages` (Anthropic)、
//! `/api/chat`・`/api/generate` (Ollama)。既存テスト (openai_chat /
//! anthropic_compat / ollama_api / e2e_compat) は正常系と一部の壊れ JSON を
//! 見るが、ここでは「入力の壊れ方が違っても、パニックせず・スロットを漏らさず
//! 戻る」という共通契約を固定する。
//!
//! 検証項目:
//! 1. 壊れた JSON / 必須フィールド欠落 -> 各プロトコルのエラー封筒で 4xx。
//! 2. 空の messages / 空 prompt、max_tokens = 0 / 負 / 巨大、未知モデル、
//!    範囲外 temperature / top_p (拒否はせず clean に処理する既存作法)。
//! 3. OpenAI + Anthropic + Ollama を混在させた並行負荷 (正常 + 異常) の後に
//!    in_flight / pending が 0 へ戻り、後続の正常要求が通る。
//!
//! CPU のみ (ダミーモデル / ダミートークナイザ)。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::tokenizer::DennoTokenizer;
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// モデル / App
// ---------------------------------------------------------------------------

/// 決定的・状態なし。peak = 直前トークン + 1 で、数トークンで EOS に達する。
/// 異常系の後にサーバーが健全であることだけを見るので遅延は入れない。
struct StepModel;

impl Model for StepModel {
    fn context(&self) -> usize {
        512
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
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
        KvCache::new(1, 4, 512)
    }
}

fn test_state(max_queue: usize) -> Arc<AppState> {
    state_with(Box::new(StepModel), dummy_tokenizer().unwrap(), max_queue)
}

fn state_with(model: Box<dyn Model>, tok: DennoTokenizer, max_queue: usize) -> Arc<AppState> {
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
// HTTP ヘルパ
// ---------------------------------------------------------------------------

async fn post_json(app: &Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    post_raw(app, uri, &body.to_string()).await
}

async fn post_raw(app: &Router, uri: &str, body: &str) -> (StatusCode, Value) {
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

fn openai_chat_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 4,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    })
}

fn anthropic_body() -> Value {
    json!({
        "model": "denno-test",
        "max_tokens": 4,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    })
}

fn ollama_chat_body() -> Value {
    json!({
        "model": "denno-test",
        "stream": false,
        "messages": [{"role": "user", "content": "t0"}],
        "options": {"temperature": 0.0},
    })
}

fn ollama_generate_body() -> Value {
    json!({
        "model": "denno-test",
        "stream": false,
        "prompt": "t0",
        "options": {"temperature": 0.0},
    })
}

/// 全経路が「正常に応答できる」ことを 1 回ずつ確かめる (回復確認用)。
async fn assert_all_protocols_healthy(app: &Router) {
    let (s, v) = post_json(app, "/v1/chat/completions", &openai_chat_body()).await;
    assert_eq!(s, StatusCode::OK, "openai chat: {v}");
    assert!(v["choices"][0]["message"]["content"].is_string(), "{v}");

    let (s, v) = post_json(app, "/v1/messages", &anthropic_body()).await;
    assert_eq!(s, StatusCode::OK, "anthropic: {v}");
    assert!(!v["content"][0]["text"].as_str().unwrap_or("").is_empty(), "{v}");

    let (s, v) = post_json(app, "/api/chat", &ollama_chat_body()).await;
    assert_eq!(s, StatusCode::OK, "ollama chat: {v}");
    assert!(v["message"]["content"].is_string(), "{v}");
    assert_eq!(v["done"], true, "{v}");

    let (s, v) = post_json(app, "/api/generate", &ollama_generate_body()).await;
    assert_eq!(s, StatusCode::OK, "ollama generate: {v}");
    assert!(v["response"].is_string(), "{v}");
    assert_eq!(v["done"], true, "{v}");

    let (s, v) = post_json(
        app,
        "/v1/completions",
        &json!({"model": "denno-test", "max_tokens": 4, "temperature": 0.0, "prompt": "t0"}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "openai completions: {v}");
    assert!(v["choices"][0]["text"].is_string(), "{v}");
}

fn assert_no_slots(state: &AppState) {
    assert_eq!(state.scheduler.in_flight(), 0, "in_flight が残っている");
    assert_eq!(state.scheduler.pending(), 0, "pending が残っている");
}

// ---------------------------------------------------------------------------
// 1. 壊れた JSON / 必須フィールド欠落
// ---------------------------------------------------------------------------

/// 壊れた JSON は各プロトコルの封筒で 400。panic も slot 消費もしない。
#[tokio::test]
async fn malformed_json_returns_protocol_envelope() {
    let state = test_state(8);
    let app = denno_server::router(state.clone());

    // OpenAI chat / completions: {"error":{"message","type",...}}。
    for uri in ["/v1/chat/completions", "/v1/completions"] {
        let (s, v) = post_raw(&app, uri, "{\"messages\": ").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: body = {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{uri}: {v}");
        assert!(v["error"]["message"].is_string(), "{uri}: {v}");
        assert_no_slots(&state);
    }

    // Anthropic: {"type":"error","error":{"type","message"}}。
    let (s, v) = post_raw(&app, "/v1/messages", "{\"messages\": ").await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");
    assert_no_slots(&state);

    // Ollama: {"error": "<message>"} (文字列)。
    for uri in ["/api/chat", "/api/generate"] {
        let (s, v) = post_raw(&app, uri, "{\"messages\": ").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: body = {v}");
        assert!(v["error"].is_string(), "{uri}: 封筒が Ollama 形式でない: {v}");
        assert_no_slots(&state);
    }

    // 空ボディも 5 経路すべて 400。
    for uri in [
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/api/chat",
        "/api/generate",
    ] {
        let (s, v) = post_raw(&app, uri, "").await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: body = {v}");
        assert_no_slots(&state);
    }

    // 壊れた JSON の直後でも全経路が正常応答へ戻る。
    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

/// 必須フィールドの欠落・型違いは 400 (panic しない)。
#[tokio::test]
async fn missing_required_fields_are_rejected_cleanly() {
    let state = test_state(8);
    let app = denno_server::router(state.clone());

    let cases: [(&str, Value); 10] = [
        // OpenAI chat: messages / 型違い
        (
            "/v1/chat/completions",
            json!({"model": "denno-test", "max_tokens": 4}),
        ),
        (
            "/v1/chat/completions",
            json!({"model": "denno-test", "messages": "not-an-array"}),
        ),
        // OpenAI completions: prompt 欠落
        ("/v1/completions", json!({"model": "denno-test"})),
        ("/v1/completions", json!({"model": "denno-test", "prompt": null})),
        // Anthropic: messages / model 欠落
        ("/v1/messages", json!({"model": "denno-test", "max_tokens": 4})),
        ("/v1/messages", json!({"max_tokens": 4, "messages": [{"role": "user", "content": "t0"}]})),
        // Ollama chat: messages 欠落 / 型違い
        ("/api/chat", json!({"model": "denno-test", "stream": false})),
        ("/api/chat", json!({"model": "denno-test", "messages": 42})),
        // Ollama generate: prompt 欠落 / 空
        ("/api/generate", json!({"model": "denno-test", "stream": false})),
        ("/api/generate", json!({"model": "denno-test", "stream": false, "prompt": ""})),
    ];

    for (uri, body) in cases {
        let (s, v) = post_json(&app, uri, &body).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri} {body}: body = {v}");
        assert!(!v.is_null(), "{uri} {body}: 空ボディ (封筒なし)");
        assert_no_slots(&state);
    }

    // 400 を返した直後も全経路が使える。
    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

// ---------------------------------------------------------------------------
// 2. 境界値
// ---------------------------------------------------------------------------

/// 空の messages は 3 プロトコルとも 400。
#[tokio::test]
async fn empty_message_arrays_are_rejected() {
    let state = test_state(8);
    let app = denno_server::router(state.clone());

    let (s, v) = post_json(
        &app,
        "/v1/chat/completions",
        &json!({"model": "denno-test", "max_tokens": 4, "messages": []}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");

    let (s, v) = post_json(
        &app,
        "/v1/messages",
        &json!({"model": "denno-test", "max_tokens": 4, "messages": []}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "{v}");

    let (s, v) = post_json(
        &app,
        "/api/chat",
        &json!({"model": "denno-test", "stream": false, "messages": []}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"].is_string(), "{v}");

    // system だけ (user/assistant 無し) も 400。
    let (s, v) = post_json(
        &app,
        "/api/chat",
        &json!({"model": "denno-test", "stream": false,
                "messages": [{"role": "system", "content": "s"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert!(v["error"].is_string(), "{v}");

    assert_no_slots(&state);
    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

/// max_tokens の 0 / 負 / 巨大。
#[tokio::test]
async fn max_tokens_boundaries() {
    let state = test_state(8);
    let app = denno_server::router(state.clone());

    // 0 は明示指定として尊重 (prefill のみ・空本文)。400 にしない。
    let (s, v) = post_json(
        &app,
        "/v1/messages",
        &json!({"model": "denno-test", "max_tokens": 0, "temperature": 0.0,
                "messages": [{"role": "user", "content": "t0"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["stop_reason"], "max_tokens", "{v}");
    assert_eq!(v["content"][0]["text"], "", "{v}");

    // 負値は usize として読めない = 400 (panic しない)。
    for uri in ["/v1/chat/completions", "/v1/messages"] {
        let mut body = if uri == "/v1/messages" {
            anthropic_body()
        } else {
            openai_chat_body()
        };
        body["max_tokens"] = json!(-1);
        let (s, v) = post_json(&app, uri, &body).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: {v}");
        assert_no_slots(&state);
    }
    // OpenAI completions も同じ。
    let (s, v) = post_json(
        &app,
        "/v1/completions",
        &json!({"model": "denno-test", "max_tokens": -1, "prompt": "t0"}),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");

    // 巨大値は clamp されて完了する (パニック / 過大確保をしない)。
    let (s, v) = post_json(
        &app,
        "/v1/messages",
        &json!({"model": "denno-test", "max_tokens": 5_000_000, "temperature": 0.0,
                "messages": [{"role": "user", "content": "t0"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");
    assert!(v["usage"]["output_tokens"].as_u64().unwrap() <= 4096, "{v}");

    // usize に収まらない値は 400。
    let (s, v) = post_raw(
        &app,
        "/v1/chat/completions",
        &openai_chat_body().to_string().replace("\"max_tokens\":4", "\"max_tokens\":1e30"),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");

    // Ollama の num_predict は 0 / 負を「未指定」として既定へ倒す (400 にしない)。
    for n in [0i64, -1, -2] {
        let (s, v) = post_json(
            &app,
            "/api/generate",
            &json!({"model": "denno-test", "stream": false, "prompt": "t0",
                    "options": {"num_predict": n, "temperature": 0.0}}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "num_predict={n}: {v}");
        assert_eq!(v["done"], true, "num_predict={n}: {v}");
    }
    assert_no_slots(&state);
    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

/// 未知モデルは 404。各プロトコルの封筒で返る。
#[tokio::test]
async fn unknown_model_is_404_on_all_protocols() {
    let state = test_state(8);
    let app = denno_server::router(state.clone());

    let (s, v) = post_json(
        &app,
        "/v1/messages",
        &json!({"model": "nope", "max_tokens": 4, "messages": [{"role": "user", "content": "t0"}]}),
    )
    .await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{v}");
    assert_eq!(v["type"], "error", "{v}");
    assert_eq!(v["error"]["type"], "not_found_error", "{v}");

    for uri in ["/v1/chat/completions", "/v1/completions"] {
        let body = if uri == "/v1/chat/completions" {
            json!({"model": "nope", "max_tokens": 4, "messages": [{"role": "user", "content": "t0"}]})
        } else {
            json!({"model": "nope", "max_tokens": 4, "prompt": "t0"})
        };
        let (s, v) = post_json(&app, uri, &body).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{uri}: {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "{uri}: {v}");
        assert_no_slots(&state);
    }

    for uri in ["/api/chat", "/api/generate"] {
        let body = if uri == "/api/chat" {
            json!({"model": "nope", "stream": false, "messages": [{"role": "user", "content": "t0"}]})
        } else {
            json!({"model": "nope", "stream": false, "prompt": "t0"})
        };
        let (s, v) = post_json(&app, uri, &body).await;
        assert_eq!(s, StatusCode::NOT_FOUND, "{uri}: {v}");
        let msg = v["error"].as_str().unwrap_or("");
        assert!(msg.contains("not found"), "{uri}: {v}");
        assert_no_slots(&state);
    }

    // 空の model は「既定モデル」として受理する (Ollama / completions の規則)。
    let (s, v) = post_json(
        &app,
        "/api/generate",
        &json!({"stream": false, "prompt": "t0", "options": {"temperature": 0.0}}),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{v}");

    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

/// 範囲外 temperature / top_p は拒否せず clean に処理する (docs/api.md の作法)。
/// NaN / inf は JSON で表現できないため、文字列 (型違い) は 400 になることも固定。
#[tokio::test]
async fn out_of_range_sampling_params_do_not_panic() {
    let state = test_state(8);
    let app = denno_server::router(state.clone());

    for (temp, top_p) in [
        (-1.0f64, 0.9f64),
        (0.0, 0.0),
        (0.0, -0.5),
        (0.0, 1.5),
        (1e30, 2.0),
        (1e-30, 1e-30),
    ] {
        let (s, v) = post_json(
            &app,
            "/v1/messages",
            &json!({"model": "denno-test", "max_tokens": 4,
                    "temperature": temp, "top_p": top_p,
                    "messages": [{"role": "user", "content": "t0"}]}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "temperature={temp} top_p={top_p}: {v}");
        assert_no_slots(&state);

        let (s, v) = post_json(
            &app,
            "/api/generate",
            &json!({"model": "denno-test", "stream": false, "prompt": "t0",
                    "options": {"temperature": temp, "top_p": top_p}}),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "ollama temperature={temp} top_p={top_p}: {v}");
        assert_no_slots(&state);
    }

    // 型違い (文字列) は 400。
    for uri in ["/v1/chat/completions", "/v1/messages", "/api/chat"] {
        let body = match uri {
            "/v1/chat/completions" => json!({"model": "denno-test",
                "messages": [{"role": "user", "content": "t0"}], "temperature": "hot"}),
            "/v1/messages" => json!({"model": "denno-test", "max_tokens": 4,
                "messages": [{"role": "user", "content": "t0"}], "top_p": "wide"}),
            _ => json!({"model": "denno-test", "stream": false,
                "messages": [{"role": "user", "content": "t0"}],
                "options": {"top_p": "wide"}}),
        };
        let (s, v) = post_json(&app, uri, &body).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "{uri}: {v}");
        assert_no_slots(&state);
    }

    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

// ---------------------------------------------------------------------------
// 3. 混在プロトコルの並行負荷 (正常 + 異常)
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_mixed_protocols_leak_no_slots() {
    let state = test_state(64);
    let app = denno_server::router(state.clone());

    // 正常 5 + 異常 5 の混在を 3 プロトコル分。異常の期待値は 400 か 404。
    let mut tasks = Vec::new();
    for i in 0..30usize {
        let app = app.clone();
        let kind = i % 3;
        let broken = i % 2 == 1;
        tasks.push(tokio::spawn(async move {
            let (uri, body) = match (kind, broken) {
                (0, false) => ("/v1/chat/completions", openai_chat_body()),
                (0, true) => (
                    "/v1/chat/completions",
                    json!({"model": "denno-test", "max_tokens": 4, "messages": []}),
                ),
                (1, false) => ("/v1/messages", anthropic_body()),
                (1, true) => (
                    "/v1/messages",
                    json!({"model": "denno-test", "messages": "broken"}),
                ),
                (2, false) => ("/api/chat", ollama_chat_body()),
                (2, true) => (
                    "/api/generate",
                    json!({"model": "denno-test", "stream": false, "prompt": ""}),
                ),
                _ => unreachable!(),
            };
            let (status, v) = post_json(&app, uri, &body).await;
            (i, uri, broken, status, v)
        }));
    }

    for t in tasks {
        let (i, uri, broken, status, v) = t.await.unwrap();
        if broken {
            assert!(
                status.is_client_error(),
                "request {i} ({uri}) の異常系が 4xx でない: {status} {v}"
            );
        } else {
            assert_eq!(status, StatusCode::OK, "request {i} ({uri}): {v}");
        }
    }

    // 並行負荷の完了後にスロットが漏れていない。
    assert_no_slots(&state);

    // 3 プロトコルとも通常応答へ戻る。
    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}

/// 異常系を大量に連投しても slot が増えない (admit 前後で失敗させる)。
#[tokio::test]
async fn repeated_rejections_do_not_leak_slots() {
    let state = test_state(4);
    let app = denno_server::router(state.clone());

    for _ in 0..50 {
        let (s, _) = post_raw(&app, "/v1/messages", "{").await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
        let (s, _) = post_json(
            &app,
            "/api/chat",
            &json!({"model": "nope", "stream": false,
                    "messages": [{"role": "user", "content": "t0"}]}),
        )
        .await;
        assert_eq!(s, StatusCode::NOT_FOUND);
        let (s, _) = post_json(
            &app,
            "/v1/completions",
            &json!({"model": "denno-test", "prompt": ""}),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST);
    }

    assert_no_slots(&state);
    assert_all_protocols_healthy(&app).await;
    assert_no_slots(&state);
}
