//! task 1329: Ollama `/api/chat` / `/api/generate` の
//! マルチターン・keep_alive・options・NDJSON フレーミング契約テスト。
//!
//! `tests/ollama_api.rs` (正常系の一致) と `tests/ollama_edge_979.rs`
//! (ストリーミング/入力エラーの細部) の後を埋める。ここで固定するのは:
//!
//! - `/api/chat` の system / user / assistant 混在履歴が stream=false と
//!   stream=true の両方で受理され、同じ本文・同じトークン数になること。
//! - `keep_alive` が文字列 (`"5m"` / `"0"`) でも整数 (`0` / `300`) でも
//!   受理されること (現実装は `ChatRequest` / `GenerateRequest` に
//!   `keep_alive` フィールドを持たないので serde の未知キーとして無視される。
//!   エラーにしないことが契約)。
//! - `options` の `temperature` / `top_p` / `seed` / `stop` / `num_predict`。
//!   `num_predict` と `stop` は実行結果に現れること、`seed` は未知キーとして
//!   無視され greedy (temperature=0) の結果を変えないこと。
//! - NDJSON のフレーミング不変条件: 全行が単体 JSON、最終行まで `done=false`、
//!   最終行だけ `done=true` で eval 系フィールドを持つこと。
//! - 未知モデル 404 / 壊れた JSON 400 が Ollama 封筒 `{"error": "..."}` で
//!   返ること (ストリーム要求でも本文検証が先に走る)。
//!
//! 参照実装: `crates/denno-server/src/ollama.rs` (`split_chat_messages` /
//! `OllamaOptions` / `build_params` / `check_model` / `NdjsonStream`)。
//! 実装は変更せず、現行の契約だけを検査する。

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const MODEL: &str = "denno-test";
const TEST_CONTEXT: usize = 512;

/// 決定的ダミーモデル (peak = 直前トークン + 1、EOS の位置だけ避ける)。
struct SeqModel;

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
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
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn test_app() -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(SeqModel), tok, MODEL, None);
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: None,
        scheduler,
        model_name: MODEL.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

struct Resp {
    status: StatusCode,
    content_type: String,
    body: String,
}

impl Resp {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body)
            .unwrap_or_else(|e| panic!("本文が JSON ではない ({e}): {:?}", self.body))
    }

    fn error_message(&self) -> String {
        self.json()["error"]
            .as_str()
            .unwrap_or_else(|| panic!("error キーが文字列でない: {:?}", self.body))
            .to_string()
    }
}

async fn post_bytes(app: &Router, uri: &str, body: &str) -> Resp {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp { status, content_type, body: String::from_utf8_lossy(&bytes).to_string() }
}

async fn post(app: &Router, uri: &str, body: &Value) -> Resp {
    post_bytes(app, uri, &body.to_string()).await
}

/// 1 行 1 JSON を検証して分解する。空行・末尾改行の欠落はフレーミング破れ。
fn ndjson_strict(body: &str) -> Vec<Value> {
    assert!(!body.is_empty(), "NDJSON 本文が空");
    assert!(body.ends_with('\n'), "末尾が改行でない: {body:?}");
    assert_eq!(
        body.matches('\n').count(),
        body.lines().count(),
        "空行が混ざっている: {body:?}"
    );
    body.lines()
        .map(|l| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("NDJSON 行が JSON ではない ({e}): {l:?}"))
        })
        .collect()
}

/// system + user + assistant + user の 4 メッセージ履歴。
fn history_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "again"},
        ],
        "options": {"temperature": 0, "num_predict": 6},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

fn single_turn_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "options": {"temperature": 0, "num_predict": 6},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

fn generate_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": MODEL,
        "prompt": "hello",
        "system": "You are a test.",
        "options": {"temperature": 0, "num_predict": 6},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

// ---------------------------------------------------------------------------
// 1. マルチターン履歴 (/api/chat)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_multi_turn_history_non_stream_uses_full_history() {
    let app = test_app();

    let r = post(&app, "/api/chat", &history_body(Some(false))).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        r.content_type.starts_with("application/json"),
        "content-type = {}",
        r.content_type
    );
    let v = r.json();
    assert_eq!(v["done"], true, "body = {v}");
    assert_eq!(v["message"]["role"], "assistant");
    assert!(
        !v["message"]["content"].as_str().unwrap().is_empty(),
        "content が空: {v}"
    );
    let multi_prompt = v["prompt_eval_count"].as_u64().unwrap();

    // 履歴 (assistant ターンを含む) がプロンプトに載っている = 単一ターンより長い。
    let s = post(&app, "/api/chat", &single_turn_body(Some(false))).await;
    assert_eq!(s.status, StatusCode::OK, "body = {}", s.body);
    let single_prompt = s.json()["prompt_eval_count"].as_u64().unwrap();
    assert!(
        multi_prompt > single_prompt,
        "履歴がプロンプトに反映されていない: multi={multi_prompt} single={single_prompt}"
    );
}

#[tokio::test]
async fn chat_multi_turn_history_stream_matches_non_stream() {
    let app = test_app();

    let non = post(&app, "/api/chat", &history_body(Some(false))).await;
    assert_eq!(non.status, StatusCode::OK, "body = {}", non.body);
    let non = non.json();
    let want = non["message"]["content"].as_str().unwrap().to_string();

    let r = post(&app, "/api/chat", &history_body(Some(true))).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        r.content_type.starts_with("application/x-ndjson"),
        "content-type = {}",
        r.content_type
    );
    let lines = ndjson_strict(&r.body);
    assert!(lines.len() >= 2, "行が少なすぎる: {lines:?}");

    // 中間行の delta を連結すると非ストリーム本文と一致する。
    let mut joined = String::new();
    for (i, l) in lines.iter().take(lines.len() - 1).enumerate() {
        assert_eq!(l["done"], false, "line {i} が done:true: {l}");
        assert_eq!(l["model"], MODEL, "line {i}");
        joined.push_str(l["message"]["content"].as_str().expect("content が文字列でない"));
    }
    assert_eq!(joined, want, "履歴付きストリームの delta 連結が非ストリームと違う");

    // 最終行のトークン数は非ストリームと同じ。
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true, "最終行 = {last}");
    assert_eq!(last["prompt_eval_count"], non["prompt_eval_count"]);
    assert_eq!(last["eval_count"], non["eval_count"]);
    assert_eq!(last["done_reason"], non["done_reason"]);
}

// ---------------------------------------------------------------------------
// 2. keep_alive の型ゆるさ (文字列 / 整数)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn keep_alive_accepts_string_and_integer_forms() {
    let app = test_app();

    // (値, 説明)。現実装は keep_alive を未知キーとして無視するので、
    // どの型でも受理され、結果は baseline と同一になる。
    let cases: Vec<(Value, &str)> = vec![
        (json!("5m"), "string 5m"),
        (json!("0"), "string 0"),
        (json!(0), "int 0"),
        (json!(300), "int 300"),
        (json!(-1), "int -1 (Ollama の永続)"),
    ];

    let base = post(&app, "/api/chat", &{
        let mut b = single_turn_body(Some(false));
        b["messages"] = json!([{"role": "user", "content": "hello"}]);
        b
    })
    .await;
    assert_eq!(base.status, StatusCode::OK, "body = {}", base.body);
    let base = base.json();
    let base_text = base["message"]["content"].as_str().unwrap().to_string();

    for (ka, label) in cases {
        let mut body = single_turn_body(Some(false));
        body["messages"] = json!([{"role": "user", "content": "hello"}]);
        body["keep_alive"] = ka.clone();
        let r = post(&app, "/api/chat", &body).await;
        assert_eq!(r.status, StatusCode::OK, "keep_alive = {label}: {}", r.body);
        let v = r.json();
        assert_eq!(v["done"], true, "keep_alive = {label}: {v}");
        assert_eq!(
            v["message"]["content"].as_str().unwrap(),
            base_text,
            "keep_alive = {label} が生成結果を変えた"
        );

        // ストリーム経路でも同じ (本文検証はストリーム開始前)。
        body["stream"] = json!(true);
        let r = post(&app, "/api/chat", &body).await;
        assert_eq!(r.status, StatusCode::OK, "keep_alive = {label} (stream): {}", r.body);
        let lines = ndjson_strict(&r.body);
        assert_eq!(lines.last().unwrap()["done"], true, "keep_alive = {label}: {lines:?}");

        // /api/generate も同じ DTO 方針 (未知キー無視)。
        let mut g = generate_body(Some(false));
        g["keep_alive"] = ka;
        let r = post(&app, "/api/generate", &g).await;
        assert_eq!(r.status, StatusCode::OK, "keep_alive = {label} (generate): {}", r.body);
        assert_eq!(r.json()["done"], true);
    }
}

// ---------------------------------------------------------------------------
// 3. options
// ---------------------------------------------------------------------------

#[tokio::test]
async fn options_temperature_top_p_seed_are_accepted() {
    let app = test_app();

    let base = post(
        &app,
        "/api/chat",
        &json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "options": {"temperature": 0, "num_predict": 6},
        }),
    )
    .await;
    assert_eq!(base.status, StatusCode::OK, "body = {}", base.body);
    let base_text = base.json()["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();

    // temperature / top_p / seed は受理され、greedy (temperature=0) の
    // 出力を変えない (seed は現実装では未知キー = 無視)。
    for opts in [
        json!({"temperature": 0, "top_p": 0.9, "num_predict": 6}),
        json!({"temperature": 0, "top_p": 1.0, "seed": 42, "num_predict": 6}),
        json!({"temperature": 0, "top_p": 0.5, "seed": 1234, "num_predict": 6}),
    ] {
        let r = post(
            &app,
            "/api/chat",
            &json!({
                "model": MODEL,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": false,
                "options": opts,
            }),
        )
        .await;
        assert_eq!(r.status, StatusCode::OK, "options = {opts}: {}", r.body);
        assert_eq!(
            r.json()["message"]["content"].as_str().unwrap(),
            base_text,
            "options = {opts} が greedy の出力を変えた"
        );
    }
}

#[tokio::test]
async fn options_num_predict_is_respected_in_both_stream_modes() {
    let app = test_app();

    for n in [1u64, 3, 7] {
        let body = json!({
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": false,
            "options": {"temperature": 0, "num_predict": n},
        });
        let r = post(&app, "/api/chat", &body).await;
        assert_eq!(r.status, StatusCode::OK, "num_predict = {n}: {}", r.body);
        let v = r.json();
        assert_eq!(v["eval_count"], n, "num_predict = {n}: {v}");
        assert_eq!(v["done_reason"], "length", "num_predict = {n}: {v}");

        // ストリーム経路でも最終行に同じ上限が効く。
        let mut sbody = body.clone();
        sbody["stream"] = json!(true);
        let r = post(&app, "/api/chat", &sbody).await;
        assert_eq!(r.status, StatusCode::OK, "num_predict = {n} (stream): {}", r.body);
        let lines = ndjson_strict(&r.body);
        let last = lines.last().unwrap();
        assert_eq!(last["done"], true, "num_predict = {n}: {last}");
        assert_eq!(last["eval_count"], n, "num_predict = {n}: {last}");
        assert_eq!(last["done_reason"], "length", "num_predict = {n}: {last}");
    }
}

#[tokio::test]
async fn options_stop_is_respected_string_and_array() {
    let app = test_app();

    let full = post(&app, "/api/chat", &single_turn_body(Some(false))).await;
    assert_eq!(full.status, StatusCode::OK, "body = {}", full.body);
    let text = full.json()["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let last_word = text
        .split_whitespace()
        .last()
        .expect("生成本文が空")
        .to_string();

    for stop in [json!(last_word.clone()), json!([last_word.clone(), "zzz"])] {
        let body = json!({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a test."},
                {"role": "user", "content": "hello"},
            ],
            "stream": false,
            "options": {"temperature": 0, "num_predict": 6, "stop": stop},
        });
        let r = post(&app, "/api/chat", &body).await;
        assert_eq!(r.status, StatusCode::OK, "stop = {stop}: {}", r.body);
        let v = r.json();
        assert!(
            !v["message"]["content"].as_str().unwrap().contains(&last_word),
            "stop = {stop} の語が本文に残る: {v}"
        );
        assert_eq!(v["done_reason"], "stop", "stop = {stop}: {v}");
    }
}

// ---------------------------------------------------------------------------
// 4. NDJSON フレーミング不変条件
// ---------------------------------------------------------------------------

#[tokio::test]
async fn ndjson_every_line_is_json_and_only_final_is_done() {
    let app = test_app();

    for uri in ["/api/chat", "/api/generate"] {
        let body = if uri == "/api/chat" {
            history_body(Some(true))
        } else {
            generate_body(Some(true))
        };
        let r = post(&app, uri, &body).await;
        assert_eq!(r.status, StatusCode::OK, "{uri} body = {}", r.body);
        assert!(
            r.content_type.starts_with("application/x-ndjson"),
            "{uri} content-type = {}",
            r.content_type
        );

        let lines = ndjson_strict(&r.body);
        assert!(lines.len() >= 2, "{uri} 行が少なすぎる: {lines:?}");

        let created_at = lines[0]["created_at"]
            .as_str()
            .expect("created_at が文字列でない")
            .to_string();
        assert_eq!(created_at.len(), 20, "{uri} created_at = {created_at}");
        assert!(created_at.ends_with('Z'), "{uri} created_at が UTC でない");

        for (i, l) in lines.iter().enumerate() {
            assert_eq!(l["model"], MODEL, "{uri} line {i}");
            assert_eq!(l["created_at"], created_at, "{uri} line {i}");
            let is_last = i + 1 == lines.len();
            if !is_last {
                assert_eq!(l["done"], false, "{uri} line {i} が done:true: {l}");
                // 中間行は eval 系フィールドを持たない (最終行だけ)。
                for key in [
                    "eval_count",
                    "prompt_eval_count",
                    "total_duration",
                    "done_reason",
                ] {
                    assert!(l.get(key).is_none(), "{uri} line {i} に {key}: {l}");
                }
            }
        }

        let last = lines.last().unwrap();
        assert_eq!(last["done"], true, "{uri} 最終行 = {last}");
        // 最終行の必須フィールドと型。
        for key in ["eval_count", "prompt_eval_count", "total_duration"] {
            assert!(
                last.get(key).map(Value::is_u64).unwrap_or(false),
                "{uri} 最終行の {key} が非負整数でない: {last}"
            );
        }
        for key in ["eval_duration", "load_duration", "prompt_eval_duration"] {
            assert!(
                last.get(key).map(Value::is_u64).unwrap_or(false),
                "{uri} 最終行の {key} が非負整数でない: {last}"
            );
        }
        assert!(last["eval_count"].as_u64().unwrap() > 0, "{uri}: {last}");
        assert!(
            last["prompt_eval_count"].as_u64().unwrap() > 0,
            "{uri}: {last}"
        );
        assert_eq!(last["load_duration"], 0, "{uri}: {last}");
        assert_eq!(last["prompt_eval_duration"], 0, "{uri}: {last}");
        assert!(
            matches!(last["done_reason"].as_str(), Some("stop") | Some("length")),
            "{uri} done_reason = {last}"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. 4xx エラー封筒 (未知モデル / 壊れた JSON)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn invalid_model_returns_404_error_envelope() {
    let app = test_app();

    // ストリーム既定 (true) でも本文検証が先に走る = 200 のストリームに
    // ならないこと。
    for uri in ["/api/chat", "/api/generate"] {
        let body = if uri == "/api/chat" {
            json!({"model": "nope:latest", "messages": [{"role": "user", "content": "hi"}]})
        } else {
            json!({"model": "nope:latest", "prompt": "hi"})
        };
        let r = post(&app, uri, &body).await;
        assert_eq!(r.status, StatusCode::NOT_FOUND, "{uri} body = {}", r.body);
        assert!(
            r.content_type.starts_with("application/json"),
            "{uri} content-type = {}",
            r.content_type
        );
        let v = r.json();
        assert_eq!(
            v.as_object().map(|o| o.len()),
            Some(1),
            "{uri} 封筒に余計なキー: {v}"
        );
        let msg = r.error_message();
        assert!(msg.contains("nope:latest"), "{uri} の文言: {msg}");
    }
}

#[tokio::test]
async fn malformed_json_returns_400_error_envelope() {
    let app = test_app();

    for uri in ["/api/chat", "/api/generate"] {
        // 構文エラー / 非オブジェクト / 空ボディ。ストリーム指定でも 400 が先。
        for body in [
            "{\"messages\": ",
            "[]",
            "null",
            "42",
            "\"hello\"",
            "",
            "{\"model\": \"denno-test\", \"messages\": \"oops\"}",
        ] {
            let r = post_bytes(&app, uri, body).await;
            assert_eq!(
                r.status,
                StatusCode::BAD_REQUEST,
                "{uri} body = {body:?}: {}",
                r.body
            );
            assert!(
                r.content_type.starts_with("application/json"),
                "{uri} body = {body:?} content-type = {}",
                r.content_type
            );
            let v = r.json();
            assert_eq!(
                v.as_object().map(|o| o.len()),
                Some(1),
                "{uri} body = {body:?} 封筒に余計なキー: {v}"
            );
            assert!(!r.error_message().is_empty(), "{uri} body = {body:?}");
        }
    }
}

#[tokio::test]
async fn chat_empty_messages_and_unsupported_roles_are_400() {
    let app = test_app();

    // messages 無し / 空配列。
    for body in [
        json!({"model": MODEL}),
        json!({"model": MODEL, "messages": []}),
    ] {
        let r = post(&app, "/api/chat", &body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body}: {}", r.body);
        assert!(r.error_message().contains("messages"), "body = {body}: {}", r.body);
    }

    // system だけ (user/assistant が無い) / tool ロールは 400。
    for body in [
        json!({"model": MODEL, "messages": [{"role": "system", "content": "s"}]}),
        json!({"model": MODEL, "messages": [{"role": "tool", "content": "x"}]}),
    ] {
        let r = post(&app, "/api/chat", &body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body}: {}", r.body);
        let msg = r.error_message();
        assert!(!msg.is_empty(), "body = {body}");
    }
}

/// 念のため: テスト用の固定待ちが混ざっていないこと (実時間に依存しない)。
#[tokio::test]
async fn streaming_completes_without_artificial_delay() {
    let app = test_app();
    let started = std::time::Instant::now();
    let r = post(&app, "/api/chat", &history_body(Some(true))).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "ストリームが想定より遅い: {:?}",
        started.elapsed()
    );
}
