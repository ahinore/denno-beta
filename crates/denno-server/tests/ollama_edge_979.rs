//! P979: Ollama 互換 API (`/api/tags` / `/api/version` / `/api/chat` /
//! `/api/generate`) のエッジケース・ストリーミング契約テスト。
//!
//! `tests/ollama_api.rs` と同じダミートークナイザ + 決定的ダミーモデルを使い、
//! HTTP ワイヤレベル (axum Router + tower oneshot) で検証する。`ollama_api.rs`
//! が「正常系の一致」を見るのに対し、こちらは **契約の細部** を見る:
//!
//! - `/api/tags` / `/api/version` の JSON スキーマと content-type
//! - NDJSON の行フレーミング (1 行 1 JSON / 末尾改行 / 最終行だけ `done: true`) と
//!   最終行の eval カウント
//! - `stream: false` が 1 個の JSON オブジェクトであること
//! - 複数ターン履歴、`options` (temperature / num_predict / stop)、空プロンプト、
//!   壊れた JSON
//! - 並行 NDJSON ストリームがスロットを漏らさないこと

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
}

impl Model for SeqModel {
    fn context(&self) -> usize {
        TEST_CONTEXT
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
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn app_with(delay: Duration) -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(SeqModel { delay }), tok, "denno-test", None);
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
    denno_server::router(app_with(Duration::ZERO))
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
}

async fn send(app: &Router, method: &str, uri: &str, body: Option<String>) -> Resp {
    let mut builder = Request::builder().method(method).uri(uri);
    if body.is_some() {
        builder = builder.header("content-type", "application/json");
    }
    let req = builder.body(Body::from(body.unwrap_or_default())).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp { status, content_type, body: String::from_utf8_lossy(&bytes).to_string() }
}

async fn get(app: &Router, uri: &str) -> Resp {
    send(app, "GET", uri, None).await
}

async fn post(app: &Router, uri: &str, body: Value) -> Resp {
    send(app, "POST", uri, Some(body.to_string())).await
}

async fn post_str(app: &Router, uri: &str, body: &str) -> Resp {
    send(app, "POST", uri, Some(body.to_string())).await
}

/// 1 行 1 JSON であることを検証してから分解する。
/// - 本文は空でない
/// - 末尾は必ず改行 (最終行もフレームされる)
/// - 空行が無い (= 行数 == 改行数)
/// - 各行が単体で JSON としてパースできる
fn ndjson_strict(body: &str) -> Vec<Value> {
    assert!(!body.is_empty(), "NDJSON 本文が空");
    assert!(body.ends_with('\n'), "NDJSON 本文が改行で終わっていない: {body:?}");
    assert_eq!(
        body.matches('\n').count(),
        body.lines().count(),
        "空行が混ざっている (フレーミング破れ): {body:?}"
    );
    body.lines()
        .map(|l| {
            serde_json::from_str(l)
                .unwrap_or_else(|e| panic!("NDJSON 行が JSON ではない ({e}): {l:?}"))
        })
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

fn gen_body(stream: Option<bool>) -> Value {
    let mut v = json!({
        "model": "denno-test",
        "prompt": "hello",
        "system": "You are a test.",
        "options": {"temperature": 0, "num_predict": 32},
    });
    if let Some(s) = stream {
        v["stream"] = json!(s);
    }
    v
}

// ---------------------------------------------------------------------------
// /api/tags, /api/version
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tags_schema_and_content_type() {
    let app = test_app();
    let r = get(&app, "/api/tags").await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        r.content_type.starts_with("application/json"),
        "content-type = {}",
        r.content_type
    );

    let v = r.json();
    let models = v["models"].as_array().expect("models が配列でない");
    assert_eq!(models.len(), 1, "models = {v}");
    let m = &models[0];

    for key in ["name", "model", "modified_at", "size", "digest", "details"] {
        assert!(m.get(key).is_some(), "models[0].{key} が無い: {m}");
    }
    assert_eq!(m["name"], "denno-test");
    assert_eq!(m["model"], "denno-test");
    assert_eq!(m["size"], 0);
    assert!(m["size"].is_u64(), "size が整数でない: {m}");

    let modified = m["modified_at"].as_str().expect("modified_at が文字列でない");
    // RFC3339 (UTC・秒精度): YYYY-MM-DDTHH:MM:SSZ
    assert_eq!(modified.len(), 20, "modified_at = {modified}");
    assert!(modified.ends_with('Z'));
    assert_eq!(&modified[4..5], "-");
    assert_eq!(&modified[10..11], "T");

    let digest = m["digest"].as_str().expect("digest が文字列でない");
    assert_eq!(digest.len(), 64, "digest = {digest}");
    assert!(
        digest.chars().all(|c| c.is_ascii_hexdigit()),
        "digest が hex でない: {digest}"
    );

    let d = &m["details"];
    assert_eq!(d["parent_model"], "");
    assert_eq!(d["format"], "gguf");
    assert_eq!(d["family"], "denno");
    assert_eq!(d["families"][0], "denno");
    assert_eq!(d["parameter_size"], "");
    assert!(
        d["quantization_level"].is_string(),
        "quantization_level が文字列でない: {d}"
    );
}

#[tokio::test]
async fn version_schema_and_content_type() {
    let app = test_app();
    let r = get(&app, "/api/version").await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        r.content_type.starts_with("application/json"),
        "content-type = {}",
        r.content_type
    );
    let v = r.json();
    assert_eq!(v.as_object().map(|o| o.len()), Some(1), "余計なキー: {v}");
    assert_eq!(v["version"], "0.1.0-denno");
}

// ---------------------------------------------------------------------------
// ストリーミング (NDJSON) のフレーミング
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_true_frames_ndjson_with_eval_counts() {
    let app = test_app();
    let (_, non) = {
        let r = post(&app, "/api/chat", chat_body(Some(false))).await;
        assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
        (r.status, r.json())
    };
    let want = non["message"]["content"].as_str().unwrap().to_string();

    let r = post(&app, "/api/chat", chat_body(Some(true))).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        r.content_type.starts_with("application/x-ndjson"),
        "content-type = {}",
        r.content_type
    );

    let lines = ndjson_strict(&r.body);
    assert!(lines.len() >= 2, "行が少なすぎる: {lines:?}");

    let created_at = lines[0]["created_at"].as_str().expect("created_at が文字列でない");
    assert_eq!(created_at.len(), 20);
    assert!(created_at.ends_with('Z'));

    // delta 行: done=false、message.role=assistant、content が文字列
    let mut joined = String::new();
    for (i, l) in lines.iter().enumerate() {
        assert_eq!(l["model"], "denno-test", "line {i}");
        assert_eq!(l["created_at"].as_str().unwrap(), created_at, "line {i}");
        let is_last = i == lines.len() - 1;
        if !is_last {
            assert_eq!(l["done"], false, "line {i} が done:true: {l}");
            assert_eq!(l["message"]["role"], "assistant", "line {i}");
            joined.push_str(l["message"]["content"].as_str().expect("content が文字列でない"));
            // 中間行は eval カウントを持たない (最終行のみ)
            assert!(l.get("eval_count").is_none(), "line {i} に eval_count: {l}");
        }
    }
    assert_eq!(joined, want, "delta の連結が非ストリーム本文と違う");

    let last = lines.last().unwrap();
    assert_eq!(last["done"], true, "最終行が done:true でない: {last}");
    assert_eq!(last["message"]["role"], "assistant");
    assert_eq!(last["message"]["content"], "");
    assert_eq!(last["done_reason"], non["done_reason"]);
    assert_eq!(last["prompt_eval_count"], non["prompt_eval_count"]);
    assert_eq!(last["eval_count"], non["eval_count"]);
    assert!(last["prompt_eval_count"].as_u64().unwrap() > 0);
    assert!(last["eval_count"].as_u64().unwrap() > 0);
    // 時間は ns の整数、load_duration / prompt_eval_duration は 0
    assert!(last["total_duration"].is_u64(), "total_duration = {last}");
    assert!(last["eval_duration"].is_u64(), "eval_duration = {last}");
    assert_eq!(last["load_duration"], 0);
    assert_eq!(last["prompt_eval_duration"], 0);
}

#[tokio::test]
async fn generate_stream_true_frames_ndjson_with_eval_counts() {
    let app = test_app();
    let non_resp = post(&app, "/api/generate", gen_body(Some(false))).await;
    assert_eq!(non_resp.status, StatusCode::OK, "body = {}", non_resp.body);
    let non = non_resp.json();
    let want = non["response"].as_str().unwrap().to_string();

    let r = post(&app, "/api/generate", gen_body(Some(true))).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert!(
        r.content_type.starts_with("application/x-ndjson"),
        "content-type = {}",
        r.content_type
    );

    let lines = ndjson_strict(&r.body);
    assert!(lines.len() >= 2, "行が少なすぎる: {lines:?}");

    let mut joined = String::new();
    for (i, l) in lines.iter().take(lines.len() - 1).enumerate() {
        assert_eq!(l["done"], false, "line {i} が done:true: {l}");
        assert!(l.get("message").is_none(), "generate 行に message: {l}");
        joined.push_str(l["response"].as_str().expect("response が文字列でない"));
    }
    assert_eq!(joined, want);

    let last = lines.last().unwrap();
    assert_eq!(last["done"], true);
    assert_eq!(last["response"], "");
    assert_eq!(last["done_reason"], non["done_reason"]);
    assert_eq!(last["prompt_eval_count"], non["prompt_eval_count"]);
    assert_eq!(last["eval_count"], non["eval_count"]);
    assert!(last["eval_count"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn non_stream_returns_single_json_object() {
    let app = test_app();
    for (uri, body) in [
        ("/api/chat", chat_body(Some(false))),
        ("/api/generate", gen_body(Some(false))),
    ] {
        let r = post(&app, uri, body).await;
        assert_eq!(r.status, StatusCode::OK, "{uri} body = {}", r.body);
        assert!(
            r.content_type.starts_with("application/json"),
            "{uri} content-type = {}",
            r.content_type
        );
        // 1 行 = 1 個の JSON オブジェクト (NDJSON ではない)
        assert_eq!(r.body.lines().count(), 1, "{uri} body = {:?}", r.body);
        let v = r.json();
        assert!(v.is_object(), "{uri} が JSON オブジェクトでない: {v}");
        assert_eq!(v["done"], true, "{uri} body = {v}");
        assert!(v.get("eval_count").is_some(), "{uri} body = {v}");
        assert!(v.get("done_reason").is_some(), "{uri} body = {v}");
        assert!(v.get("created_at").is_some(), "{uri} body = {v}");
    }
}

// ---------------------------------------------------------------------------
// 履歴 / options
// ---------------------------------------------------------------------------

#[tokio::test]
async fn multi_turn_history_is_accepted_and_counted() {
    let app = test_app();

    let single = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "options": {"temperature": 0, "num_predict": 8},
    });
    let r = post(&app, "/api/chat", single).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let single_prompt = r.json()["prompt_eval_count"].as_u64().unwrap();

    let multi = json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "again"},
        ],
        "stream": false,
        "options": {"temperature": 0, "num_predict": 8},
    });
    let r = post(&app, "/api/chat", multi).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let v = r.json();
    assert_eq!(v["message"]["role"], "assistant");
    assert!(!v["message"]["content"].as_str().unwrap().is_empty());
    let multi_prompt = v["prompt_eval_count"].as_u64().unwrap();
    assert!(
        multi_prompt > single_prompt,
        "履歴がプロンプトに反映されていない: multi={multi_prompt} single={single_prompt}"
    );
    assert_eq!(v["eval_count"], 8);
}

#[tokio::test]
async fn options_temperature_num_predict_and_stop_variants() {
    let app = test_app();

    // temperature は受理される (上限なしの state では素通し)
    let mut body = chat_body(Some(false));
    body["options"]["temperature"] = json!(0.7);
    body["options"]["top_p"] = json!(0.9);
    let r = post(&app, "/api/chat", body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);

    // num_predict = 1 → eval_count 1 / done_reason length
    let mut body = chat_body(Some(false));
    body["options"]["num_predict"] = json!(1);
    let r = post(&app, "/api/chat", body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let v = r.json();
    assert_eq!(v["eval_count"], 1, "body = {v}");
    assert_eq!(v["done_reason"], "length");

    // num_predict 負値 / 0 は「未指定」= 既定 1024 扱い → 8 トークン生成できる
    for n in [-1i64, -2, 0] {
        let mut body = chat_body(Some(false));
        body["options"]["num_predict"] = json!(n);
        let r = post(&app, "/api/chat", body).await;
        assert_eq!(r.status, StatusCode::OK, "num_predict = {n}");
        assert!(r.json()["eval_count"].as_u64().unwrap() > 1, "num_predict = {n}");
    }

    // stop は文字列 1 本でも配列でもよい
    let (_, full) = {
        let r = post(&app, "/api/chat", chat_body(Some(false))).await;
        assert_eq!(r.status, StatusCode::OK);
        (r.status, r.json())
    };
    let text = full["message"]["content"].as_str().unwrap().to_string();
    let last_word = text.split_whitespace().last().expect("生成本文が空").to_string();

    let mut body = chat_body(Some(false));
    body["options"]["stop"] = json!(last_word);
    let r = post(&app, "/api/chat", body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let v = r.json();
    assert!(!v["message"]["content"].as_str().unwrap().contains(&last_word));
    assert_eq!(v["done_reason"], "stop");

    let mut body = chat_body(Some(false));
    body["options"]["stop"] = json!([last_word, ""]);
    let r = post(&app, "/api/chat", body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let v = r.json();
    assert!(!v["message"]["content"].as_str().unwrap().contains(&last_word));

    // 不正な stop (数値 / 数値配列) は 400 {"error": ...}
    for bad in [json!(1), json!([1, 2])] {
        let mut body = chat_body(Some(false));
        body["options"]["stop"] = bad.clone();
        let r = post(&app, "/api/chat", body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "stop = {bad}");
        assert!(r.json()["error"].as_str().unwrap().contains("stop"), "stop = {bad}");
    }

    // ストリーム経路でも num_predict が最終行に効く
    let mut body = chat_body(Some(true));
    body["options"]["num_predict"] = json!(3);
    let r = post(&app, "/api/chat", body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let lines = ndjson_strict(&r.body);
    let last = lines.last().unwrap();
    assert_eq!(last["done"], true);
    assert_eq!(last["eval_count"], 3);
    assert_eq!(last["done_reason"], "length");
}

// ---------------------------------------------------------------------------
// 入力エラー
// ---------------------------------------------------------------------------

#[tokio::test]
async fn empty_prompt_and_empty_messages_are_400() {
    let app = test_app();

    // prompt 空文字
    let r = post(
        &app,
        "/api/generate",
        json!({"model": "denno-test", "prompt": "", "stream": false}),
    )
    .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    let v = r.json();
    assert!(v["error"].as_str().unwrap().contains("prompt"), "body = {v}");

    // prompt 未指定 (null)
    let r = post(&app, "/api/generate", json!({"model": "denno-test", "stream": false})).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(r.json()["error"].as_str().unwrap().contains("prompt"));

    // ストリーム指定でも同じく 400 (ストリームを開かない)
    let r = post(&app, "/api/generate", json!({"model": "denno-test", "prompt": "", "stream": true})).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(!r.content_type.starts_with("application/x-ndjson"));

    // messages 空
    let mut body = chat_body(Some(false));
    body["messages"] = json!([]);
    let r = post(&app, "/api/chat", body).await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    assert!(r.json()["error"].as_str().unwrap().contains("messages"));

    // system だけ = user/assistant が 1 本も無い
    let r = post(
        &app,
        "/api/chat",
        json!({
            "model": "denno-test",
            "messages": [{"role": "system", "content": "only system"}],
            "stream": false,
        }),
    )
    .await;
    assert_eq!(r.status, StatusCode::BAD_REQUEST);
    let msg = r.json()["error"].as_str().unwrap().to_string();
    assert!(msg.contains("user or assistant"), "body = {msg}");
}

#[tokio::test]
async fn invalid_json_body_is_400_ollama_error() {
    let app = test_app();

    for uri in ["/api/chat", "/api/generate"] {
        for broken in ["{\"messages\": ", "not json", "", "null"] {
            let r = post_str(&app, uri, broken).await;
            assert_eq!(r.status, StatusCode::BAD_REQUEST, "{uri} body = {broken:?}");
            assert!(
                r.content_type.starts_with("application/json"),
                "{uri} content-type = {}",
                r.content_type
            );
            let v = r.json();
            assert!(
                v["error"].as_str().is_some_and(|e| e.contains("invalid request body")),
                "{uri} body = {broken:?} → {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 並行ストリーム / スロット
// ---------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_ndjson_streams_do_not_leak_slots() {
    let state = app_with(Duration::ZERO);
    let app = denno_server::router(state.clone());

    // 期待値: 同じ本文を非ストリームで 1 回生成しておく
    let mut probe = chat_body(Some(false));
    probe["options"]["num_predict"] = json!(16);
    let non = post(&app, "/api/chat", probe).await;
    assert_eq!(non.status, StatusCode::OK, "body = {}", non.body);
    let want = non.json()["message"]["content"].as_str().unwrap().to_string();
    let want_eval = non.json()["eval_count"].as_u64().unwrap();

    let mut set = tokio::task::JoinSet::new();
    for _ in 0..4 {
        let app = app.clone();
        set.spawn(async move {
            let mut body = chat_body(Some(true));
            body["options"]["num_predict"] = json!(16);
            let req = Request::builder()
                .method("POST")
                .uri("/api/chat")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = resp.into_body().collect().await.unwrap().to_bytes();
            (status, String::from_utf8_lossy(&bytes).to_string())
        });
    }

    let mut all_joined: Vec<String> = Vec::new();
    while let Some(res) = set.join_next().await {
        let (status, body) = res.expect("並行ストリームのタスクが panic");
        assert_eq!(status, StatusCode::OK, "body = {body}");
        let lines = ndjson_strict(&body);
        let last = lines.last().expect("行が無い");
        assert_eq!(last["done"], true, "body = {body}");
        assert_eq!(last["eval_count"].as_u64().unwrap(), want_eval, "body = {body}");
        let joined: String = lines
            .iter()
            .filter(|l| l["done"] == false)
            .filter_map(|l| l["message"]["content"].as_str())
            .collect();
        assert!(!joined.is_empty());
        all_joined.push(joined);
    }
    assert_eq!(all_joined.len(), 4);
    for j in &all_joined {
        assert_eq!(*j, want, "並行ストリームの本文が非ストリームと違う");
    }

    // 全ストリーム完了後にスロットが残っていない
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    while state.scheduler.in_flight() != 0 && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(state.scheduler.in_flight(), 0, "スロットが漏れている");

    // 後続の非ストリーム要求も即座に完了する
    let r = post(&app, "/api/chat", chat_body(Some(false))).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert_eq!(state.scheduler.in_flight(), 0);
}
