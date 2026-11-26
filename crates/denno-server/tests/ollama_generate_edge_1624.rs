//! task 1624: Ollama `/api/generate` の NDJSON フレーミング・
//! `done_reason`・`system` / `context`・`keep_alive` / `format` / `think`・
//! 壊れた本文の 400 封筒を固定する契約テスト。
//!
//! `tests/ollama_api.rs` / `tests/ollama_edge_979.rs` /
//! `tests/ollama_chat_edge_1329.rs` と同じダミートークナイザ + 決定的
//! ダミーモデル (直前トークン + 1 が greedy のピーク、id 11 の次は EOS 12) を
//! 使い、HTTP ワイヤレベル (axum Router + tower oneshot) で検査する。
//!
//! 検査する契約:
//!
//! - `stream: true` と `stream: false` の本文・トークン数・`done_reason` が
//!   一致すること (`raw: true` の生プロンプト経路も含む)。
//! - NDJSON の行終端: 最終行まで改行で終わり、空行が無く、各行が単体 JSON。
//!   最終行だけ `done: true`。
//! - `done_reason` が `length` (num_predict 到達) / `stop` (EOS・停止列) に
//!   正しく写ること。
//! - `system` は non-raw 経路でプロンプトに載り、`raw: true` では無視されること。
//!   リクエスト側 `context` は未知キーとして無視され、応答側 `context` は
//!   現行実装が出さないこと (Ollama 本体との差分。実装を変えたらこのテストを
//!   更新する)。
//! - `keep_alive` (文字列 / 整数 / null / object)、`format` (`"json"` /
//!   JSON Schema / 未知値)、`think` (bool / 効力文字列 / 非文字列) が
//!   400 にならず、無視される値は結果を変えないこと。
//! - 構文エラー・非オブジェクト・型違い・空本文が 400 `{"error": ...}` で
//!   返ること (ストリーム既定でも本文検証が先)。
//!
//! 実装 (`crates/denno-server/src/ollama.rs`) は変更しない。

use std::sync::{Arc, Mutex};

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

/// 決定的ダミーモデル (peak = 直前トークン + 1)。
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

async fn post_bytes(app: &Router, body: &str) -> Resp {
    let req = Request::builder()
        .method("POST")
        .uri("/api/generate")
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

async fn post(app: &Router, body: &Value) -> Resp {
    post_bytes(app, &body.to_string()).await
}

/// NDJSON のフレーミングを厳密に検査して分解する。
/// 末尾改行・空行なし・各行が単体 JSON であることをここで固定する。
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

/// 非 raw の baseline 本文。
fn gen_body(stream: Option<bool>) -> Value {
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

/// ストリーム本文から中間行の `response` を連結する。
fn join_deltas(lines: &[Value]) -> String {
    lines
        .iter()
        .take(lines.len() - 1)
        .map(|l| {
            assert_eq!(l["done"], false, "中間行が done:true: {l}");
            l["response"].as_str().unwrap_or_else(|| panic!("response が文字列でない: {l}"))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// 1. stream=true / stream=false の一致
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_stream_and_non_stream_parity() {
    let app = test_app();

    // 非 raw (chat template + system) と raw (生プロンプト) の両経路。
    for (label, body) in [
        ("chat", gen_body(None)),
        (
            "raw",
            json!({
                "model": MODEL,
                "prompt": "t0 t1",
                "raw": true,
                "options": {"temperature": 0, "num_predict": 4},
            }),
        ),
    ] {
        let mut non_body = body.clone();
        non_body["stream"] = json!(false);
        let non = post(&app, &non_body).await;
        assert_eq!(non.status, StatusCode::OK, "{label} body = {}", non.body);
        assert!(
            non.content_type.starts_with("application/json"),
            "{label} content-type = {}",
            non.content_type
        );
        let non = non.json();
        let want = non["response"].as_str().expect("response が文字列でない").to_string();
        assert!(!want.is_empty(), "{label} response が空: {non}");

        let mut stream_body = body;
        stream_body["stream"] = json!(true);
        let r = post(&app, &stream_body).await;
        assert_eq!(r.status, StatusCode::OK, "{label} body = {}", r.body);
        assert!(
            r.content_type.starts_with("application/x-ndjson"),
            "{label} content-type = {}",
            r.content_type
        );

        let lines = ndjson_strict(&r.body);
        assert!(lines.len() >= 2, "{label} 行が少なすぎる: {lines:?}");
        assert_eq!(join_deltas(&lines), want, "{label} delta の連結が非ストリームと違う");

        let last = lines.last().unwrap();
        assert_eq!(last["done"], true, "{label} 最終行 = {last}");
        assert_eq!(last["response"], "", "{label} 最終行の response: {last}");
        assert_eq!(last["eval_count"], non["eval_count"], "{label}: {last} vs {non}");
        assert_eq!(
            last["prompt_eval_count"], non["prompt_eval_count"],
            "{label}: {last} vs {non}"
        );
        assert_eq!(last["done_reason"], non["done_reason"], "{label}: {last} vs {non}");
        assert_eq!(last["model"], MODEL, "{label}: {last}");
    }
}

// ---------------------------------------------------------------------------
// 2. NDJSON の行終端
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_ndjson_terminates_every_line() {
    let app = test_app();
    let mut body = gen_body(Some(true));
    body["options"]["num_predict"] = json!(32);
    let r = post(&app, &body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);

    // ndjson_strict が末尾改行 / 空行なし / 行ごとの JSON を検査する。
    let lines = ndjson_strict(&r.body);
    assert!(lines.len() >= 2, "行が少なすぎる: {lines:?}");

    let created_at = lines[0]["created_at"]
        .as_str()
        .expect("created_at が文字列でない")
        .to_string();
    let done_at: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l["done"] == true)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(done_at, vec![lines.len() - 1], "done:true が最終行だけではない: {lines:?}");

    for (i, l) in lines.iter().enumerate() {
        assert_eq!(l["model"], MODEL, "line {i}");
        assert_eq!(l["created_at"], created_at, "line {i}");
        // /api/generate の行に chat 用の message は出ない。
        assert!(l.get("message").is_none(), "line {i} に message: {l}");
    }

    let last = lines.last().unwrap();
    for key in [
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ] {
        assert!(last.get(key).is_some(), "最終行に {key} が無い: {last}");
    }
    assert_eq!(last["load_duration"], 0, "{last}");
    assert_eq!(last["prompt_eval_duration"], 0, "{last}");
    assert!(last["total_duration"].is_u64(), "{last}");
    assert!(last["eval_duration"].is_u64(), "{last}");

    // 非ストリームは NDJSON ではなく 1 個の JSON オブジェクト。
    let mut non_body = body;
    non_body["stream"] = json!(false);
    let r = post(&app, &non_body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    assert_eq!(r.body.lines().count(), 1, "非ストリーム本文 = {:?}", r.body);
    assert!(!r.body.ends_with("\n\n"), "非ストリームに余分な改行: {:?}", r.body);
}

// ---------------------------------------------------------------------------
// 3. done_reason の写像
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_done_reason_length_on_num_predict() {
    let app = test_app();

    for n in [1u64, 3, 6] {
        for stream in [false, true] {
            let mut body = gen_body(None);
            body["options"]["num_predict"] = json!(n);
            body["stream"] = json!(stream);
            let r = post(&app, &body).await;
            assert_eq!(r.status, StatusCode::OK, "n = {n} stream = {stream}: {}", r.body);

            let v = if stream {
                ndjson_strict(&r.body).last().unwrap().clone()
            } else {
                r.json()
            };
            assert_eq!(v["done"], true, "n = {n} stream = {stream}: {v}");
            assert_eq!(v["eval_count"], n, "n = {n} stream = {stream}: {v}");
            assert_eq!(v["done_reason"], "length", "n = {n} stream = {stream}: {v}");
        }
    }
}

#[tokio::test]
async fn generate_done_reason_stop_on_eos_and_stop_sequence() {
    let app = test_app();

    // 大きな num_predict でもダミーモデルは 16 トークン以内に EOS(12) へ着く。
    for stream in [false, true] {
        let mut body = gen_body(None);
        body["options"]["num_predict"] = json!(64);
        body["stream"] = json!(stream);
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::OK, "stream = {stream}: {}", r.body);

        let (v, text) = if stream {
            let lines = ndjson_strict(&r.body);
            (lines.last().unwrap().clone(), join_deltas(&lines))
        } else {
            let v = r.json();
            let text = v["response"].as_str().unwrap().to_string();
            (v, text)
        };
        assert_eq!(v["done"], true, "stream = {stream}: {v}");
        assert_eq!(v["done_reason"], "stop", "stream = {stream}: {v}");
        assert!(
            v["eval_count"].as_u64().unwrap() <= 16,
            "EOS 前に生成しすぎ: {v}"
        );
        assert!(!text.is_empty(), "stream = {stream} 本文が空: {v}");
    }

    // 停止列 (stop_sequence) も Ollama の done_reason は "stop"。
    let mut full_body = gen_body(Some(false));
    full_body["options"]["num_predict"] = json!(64);
    let full = post(&app, &full_body).await;
    assert_eq!(full.status, StatusCode::OK, "body = {}", full.body);
    let text = full.json()["response"].as_str().unwrap().to_string();
    let last_word = text
        .split_whitespace()
        .last()
        .expect("生成本文が空")
        .to_string();

    let mut body = gen_body(None);
    body["options"]["num_predict"] = json!(64);
    body["options"]["stop"] = json!([last_word.clone()]);
    body["stream"] = json!(true);
    let r = post(&app, &body).await;
    assert_eq!(r.status, StatusCode::OK, "body = {}", r.body);
    let lines = ndjson_strict(&r.body);
    let last = lines.last().unwrap();
    assert_eq!(last["done_reason"], "stop", "stop sequence: {last}");
    assert!(
        !join_deltas(&lines).contains(&last_word),
        "停止列 {last_word:?} が本文に残る: {:?}",
        join_deltas(&lines)
    );
}

// ---------------------------------------------------------------------------
// 4. system / context フィールド
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_system_is_honored_and_context_is_ignored() {
    let app = test_app();

    let without = |stream: bool| {
        json!({
            "model": MODEL,
            "prompt": "hello",
            "stream": stream,
            "options": {"temperature": 0, "num_predict": 4},
        })
    };

    let base = post(&app, &without(false)).await;
    assert_eq!(base.status, StatusCode::OK, "body = {}", base.body);
    let base = base.json();
    let base_prompt = base["prompt_eval_count"].as_u64().unwrap();

    // system を付けるとプロンプトが伸びる (non-raw は chat template に載る)。
    let with_system = post(
        &app,
        &json!({
            "model": MODEL,
            "prompt": "hello",
            "system": "You are a test.",
            "stream": false,
            "options": {"temperature": 0, "num_predict": 4},
        }),
    )
    .await;
    assert_eq!(with_system.status, StatusCode::OK, "body = {}", with_system.body);
    let with_system = with_system.json();
    let sys_prompt = with_system["prompt_eval_count"].as_u64().unwrap();
    assert!(
        sys_prompt > base_prompt,
        "system がプロンプトに載っていない: sys={sys_prompt} base={base_prompt}"
    );
    assert!(with_system["done"].as_bool().unwrap());

    // system が空 / null なら未指定と同じ (brief_system が落とす)。
    for empty in [json!(""), Value::Null] {
        let mut body = without(false);
        body["system"] = empty.clone();
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::OK, "system = {empty}: {}", r.body);
        assert_eq!(
            r.json()["prompt_eval_count"], base["prompt_eval_count"],
            "system = {empty} がプロンプトを変えた"
        );
    }

    // raw: true は system を無視する (chat template を通さない)。
    let raw_base = post(
        &app,
        &json!({
            "model": MODEL,
            "prompt": "t0 t1",
            "raw": true,
            "stream": false,
            "options": {"temperature": 0, "num_predict": 4},
        }),
    )
    .await;
    assert_eq!(raw_base.status, StatusCode::OK, "body = {}", raw_base.body);
    let raw_base = raw_base.json();
    let raw_sys = post(
        &app,
        &json!({
            "model": MODEL,
            "prompt": "t0 t1",
            "system": "You are a test.",
            "raw": true,
            "stream": false,
            "options": {"temperature": 0, "num_predict": 4},
        }),
    )
    .await;
    assert_eq!(raw_sys.status, StatusCode::OK, "body = {}", raw_sys.body);
    assert_eq!(
        raw_sys.json()["prompt_eval_count"], raw_base["prompt_eval_count"],
        "raw 経路が system をプロンプトに載せた"
    );
    assert_eq!(raw_sys.json()["response"], raw_base["response"]);

    // リクエスト側 `context` (Ollama の会話状態) は未知キーとして無視される。
    let mut with_ctx = without(false);
    with_ctx["context"] = json!([1, 2, 3, 4]);
    let r = post(&app, &with_ctx).await;
    assert_eq!(r.status, StatusCode::OK, "context 付き: {}", r.body);
    let r = r.json();
    assert_eq!(r["prompt_eval_count"], base["prompt_eval_count"], "context がプロンプトを変えた");
    assert_eq!(r["response"], base["response"]);
    assert_eq!(r["eval_count"], base["eval_count"]);

    // 応答側 `context` は現行実装が出さない (Ollama 本体との差分)。
    assert!(base.get("context").is_none(), "非ストリームに context: {base}");
    let stream = post(&app, &without(true)).await;
    assert_eq!(stream.status, StatusCode::OK, "stream body = {}", stream.body);
    let lines = ndjson_strict(&stream.body);
    for (i, l) in lines.iter().enumerate() {
        assert!(l.get("context").is_none(), "line {i} に context: {l}");
    }
}

// ---------------------------------------------------------------------------
// 5. keep_alive / format / think の互換
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_keep_alive_forms_are_accepted_and_ignored() {
    let app = test_app();

    let base = post(&app, &gen_body(Some(false))).await;
    assert_eq!(base.status, StatusCode::OK, "body = {}", base.body);
    let base = base.json();

    for (ka, label) in [
        (json!("5m"), "string 5m"),
        (json!("0"), "string 0"),
        (json!(0), "int 0"),
        (json!(300), "int 300"),
        (json!(-1), "int -1"),
        (Value::Null, "null"),
        (json!({"num_gpu": 0}), "object"),
    ] {
        for stream in [false, true] {
            let mut body = gen_body(None);
            body["keep_alive"] = ka.clone();
            body["stream"] = json!(stream);
            let r = post(&app, &body).await;
            assert_eq!(r.status, StatusCode::OK, "keep_alive = {label} stream = {stream}: {}", r.body);

            let v = if stream {
                ndjson_strict(&r.body).last().unwrap().clone()
            } else {
                r.json()
            };
            assert_eq!(v["done"], true, "keep_alive = {label}: {v}");
            assert_eq!(
                v["eval_count"], base["eval_count"],
                "keep_alive = {label} が生成を変えた"
            );
            assert_eq!(v["prompt_eval_count"], base["prompt_eval_count"], "keep_alive = {label}");
            assert_eq!(v["done_reason"], base["done_reason"], "keep_alive = {label}");
        }
    }
}

#[tokio::test]
async fn generate_format_adds_json_hint_and_unknown_values_are_ignored() {
    let app = test_app();

    let base = post(&app, &gen_body(Some(false))).await;
    assert_eq!(base.status, StatusCode::OK, "body = {}", base.body);
    let base = base.json();
    let base_prompt = base["prompt_eval_count"].as_u64().unwrap();

    // "json" と JSON Schema は system へ指示を足す = prompt_eval_count が増える。
    for (fmt, label) in [
        (json!("json"), "json"),
        (json!(" JSON "), "json (大文字/空白)"),
        (
            json!({"type": "object", "properties": {"a": {"type": "integer"}}}),
            "schema object",
        ),
    ] {
        let mut body = gen_body(Some(false));
        body["format"] = fmt.clone();
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::OK, "format = {label}: {}", r.body);
        let v = r.json();
        assert_eq!(v["done"], true, "format = {label}: {v}");
        let prompt = v["prompt_eval_count"].as_u64().unwrap();
        assert!(
            prompt > base_prompt,
            "format = {label} の指示が載っていない: {prompt} <= {base_prompt}"
        );
    }

    // 未知の文字列 / 数値 / null は無視され、結果が baseline と一致する。
    for (fmt, label) in [
        (json!("xml"), "unknown string"),
        (json!(42), "number"),
        (Value::Null, "null"),
    ] {
        let mut body = gen_body(Some(false));
        body["format"] = fmt.clone();
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::OK, "format = {label}: {}", r.body);
        let v = r.json();
        assert_eq!(v["prompt_eval_count"], base["prompt_eval_count"], "format = {label}");
        assert_eq!(v["response"], base["response"], "format = {label} が本文を変えた");
        assert_eq!(v["eval_count"], base["eval_count"], "format = {label}");
    }

    // raw: true では format の指示を足さない (system を作らない経路)。
    let raw = json!({
        "model": MODEL,
        "prompt": "t0 t1",
        "raw": true,
        "stream": false,
        "options": {"temperature": 0, "num_predict": 4},
    });
    let raw_base = post(&app, &raw).await;
    assert_eq!(raw_base.status, StatusCode::OK, "body = {}", raw_base.body);
    let mut raw_fmt = raw;
    raw_fmt["format"] = json!("json");
    let r = post(&app, &raw_fmt).await;
    assert_eq!(r.status, StatusCode::OK, "raw + format: {}", r.body);
    assert_eq!(
        r.json()["prompt_eval_count"], raw_base.json()["prompt_eval_count"],
        "raw 経路が format の指示を載せた"
    );
    assert_eq!(r.json()["response"], raw_base.json()["response"]);
}

#[tokio::test]
async fn generate_think_forms_are_accepted() {
    let app = test_app();

    let base = post(&app, &gen_body(Some(false))).await;
    assert_eq!(base.status, StatusCode::OK, "body = {}", base.body);
    let base = base.json();

    // false 系 / 未知の型は従来既定 (思考オフ) と同じ結果。
    for (think, label) in [
        (json!(false), "false"),
        (json!("false"), "string false"),
        (json!("off"), "string off"),
        (json!("bogus"), "unknown string"),
        (json!(1), "number"),
        (json!({"level": "low"}), "object"),
    ] {
        for stream in [false, true] {
            let mut body = gen_body(None);
            body["think"] = think.clone();
            body["stream"] = json!(stream);
            let r = post(&app, &body).await;
            assert_eq!(r.status, StatusCode::OK, "think = {label} stream = {stream}: {}", r.body);
            let v = if stream {
                ndjson_strict(&r.body).last().unwrap().clone()
            } else {
                r.json()
            };
            assert_eq!(v["done"], true, "think = {label}: {v}");
            assert_eq!(
                v["prompt_eval_count"], base["prompt_eval_count"],
                "think = {label} が従来既定のプロンプトを変えた"
            );
        }
    }

    // true 系 (template 既定に任せる) も 400 にせず受理する。
    for (think, label) in [
        (json!(true), "true"),
        (json!("true"), "string true"),
        (json!("low"), "string low"),
        (json!("high"), "string high"),
    ] {
        let mut body = gen_body(None);
        body["think"] = think.clone();
        body["stream"] = json!(false);
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::OK, "think = {label}: {}", r.body);
        assert_eq!(r.json()["done"], true, "think = {label}: {}", r.body);
    }
}

// ---------------------------------------------------------------------------
// 6. 壊れた本文の 400 封筒
// ---------------------------------------------------------------------------

#[tokio::test]
async fn malformed_bodies_return_400_error_envelope() {
    let app = test_app();

    // 構文エラー / 非オブジェクト / 空本文 / 型違い。stream の既定は true だが
    // 本文検証が先に走り、NDJSON ではなく JSON の 400 を返す。
    for body in [
        "{\"prompt\": ",
        "[]",
        "null",
        "42",
        "\"hello\"",
        "",
        "{\"prompt\": 123}",
        "{\"prompt\": true}",
        "{\"prompt\": [\"hello\"]}",
        "{\"model\": 123, \"prompt\": \"hi\"}",
        "{\"prompt\": \"hi\", \"stream\": \"yes\"}",
        "{\"prompt\": \"hi\", \"stream\": 1}",
        "{\"prompt\": \"hi\", \"options\": []}",
        "{\"prompt\": \"hi\", \"options\": {\"num_predict\": \"many\"}}",
        "{\"prompt\": \"hi\", \"raw\": \"yes\"}",
    ] {
        let r = post_bytes(&app, body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body:?}: {}", r.body);
        assert!(
            r.content_type.starts_with("application/json"),
            "body = {body:?} content-type = {}",
            r.content_type
        );
        let v = r.json();
        assert_eq!(
            v.as_object().map(|o| o.len()),
            Some(1),
            "body = {body:?} 封筒に余計なキー: {v}"
        );
        assert!(!r.error_message().is_empty(), "body = {body:?}");
    }
}

#[tokio::test]
async fn empty_or_missing_prompt_returns_400() {
    let app = test_app();

    for body in [
        json!({"model": MODEL}),
        json!({"model": MODEL, "prompt": ""}),
        json!({"model": MODEL, "prompt": Value::Null}),
        json!({"model": MODEL, "prompt": "", "stream": false}),
        json!({"model": MODEL, "prompt": "", "stream": true}),
        json!({"model": MODEL, "prompt": "", "raw": true}),
    ] {
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "body = {body}: {}", r.body);
        assert!(
            r.content_type.starts_with("application/json"),
            "body = {body} content-type = {}",
            r.content_type
        );
        let v = r.json();
        assert_eq!(
            v.as_object().map(|o| o.len()),
            Some(1),
            "body = {body} 封筒に余計なキー: {v}"
        );
        assert!(
            r.error_message().contains("prompt"),
            "body = {body}: {}",
            r.error_message()
        );
    }

    // 不正な options.stop も 400 で Ollama 封筒。
    for stop in [json!(1), json!([1, 2])] {
        let mut body = gen_body(Some(false));
        body["options"]["stop"] = stop.clone();
        let r = post(&app, &body).await;
        assert_eq!(r.status, StatusCode::BAD_REQUEST, "stop = {stop}: {}", r.body);
        assert!(
            r.error_message().contains("stop"),
            "stop = {stop}: {}",
            r.error_message()
        );
    }
}
