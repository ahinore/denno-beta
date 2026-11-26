//! task 1501: OpenAI `/v1/chat/completions` の `stream_options` と usage の
//! 監査 + パリティテスト。
//!
//! 監査で見つかった食い違い (このコミットで修正):
//!
//! - `stream_options` / `stream_options.include_usage` の型違いが、serde の
//!   既定メッセージのまま 400 になっていた (`error.param` が null)。同じ
//!   リクエストボディ検証をする `seed` (P986) / `logprobs` / `top_logprobs`
//!   (P981) は `param` を埋めており、扱いが揃っていなかった。
//!   修正後は `param: "stream_options"` / `"stream_options.include_usage"`。
//!
//! 監査で確認した (仕様どおりで修正不要だった) 点:
//!
//! - usage chunk は `choices: []` + `usage` で `data: [DONE]` の直前 1 個だけ。
//! - `include_usage = true` の他の chunk は `"usage": null` (キー欠落ではない)。
//! - `false` / 未指定 / `{}` / `null` は `usage` キー自体を出さない。
//! - `stream: false` では `stream_options` を無視する (usage は常に 1 個)。
//! - `prompt_tokens` / `completion_tokens` / `total_tokens` はストリームと
//!   非ストリームで完全一致し、`total = prompt + completion`。
//!
//! 方式は既存の統合テスト (tests/openai_stream_usage_936.rs /
//! tests/stream_usage_edge_1144.rs) と同じ — denno-engine のダミートークナイザと
//! 決定的モデルで HTTP ワイヤを見る。sleep は使わない。

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
    let engine = Engine::from_parts(
        Box::new(SeqModel),
        dummy_tokenizer().unwrap(),
        "denno-test",
        None,
    );
    let info = EngineInfo::from_engine(&engine);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
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

/// 生の応答 (status / content-type / 本文文字列)。
struct Resp {
    status: StatusCode,
    ctype: String,
    text: String,
}

async fn call(app: &Router, uri: &str, body: &Value) -> Resp {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp {
        status,
        ctype,
        text: String::from_utf8(bytes.to_vec()).unwrap(),
    }
}

/// `data: ...` 行の JSON を到着順に返す ([DONE] は除く)。順序も見る。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn chat_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content": "hello"},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

fn completion_body(stream: bool) -> Value {
    json!({
        "model": "denno-test",
        "prompt": "t0 t1 t2",
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    })
}

/// `usage` の 3 フィールドを取り、自己整合 (`total = prompt + completion`) を
/// 確かめる。整数以外 (小数・文字列・欠落) は失敗。
fn usage_of(v: &Value) -> (u64, u64, u64) {
    let usage = &v["usage"];
    assert!(usage.is_object(), "usage が無い: {v}");
    let p = usage["prompt_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("prompt_tokens が整数でない: {usage}"));
    let c = usage["completion_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("completion_tokens が整数でない: {usage}"));
    let t = usage["total_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("total_tokens が整数でない: {usage}"));
    assert_eq!(t, p + c, "total_tokens != prompt + completion: {usage}");
    (p, c, t)
}

/// SSE の usage chunk が「最後」で、その直前が finish_reason chunk、
/// 直後が `[DONE]` であることを確かめて usage を返す。
fn assert_usage_chunk_is_last(chunks: &[Value], text: &str) -> (u64, u64, u64) {
    assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
    let last = chunks.last().unwrap();
    assert_eq!(
        last["choices"],
        json!([]),
        "usage chunk の choices が空配列でない: {last}"
    );
    assert!(last["usage"].is_object(), "最後の chunk が usage でない: {last}");
    let n = chunks.iter().filter(|c| c["usage"].is_object()).count();
    assert_eq!(n, 1, "usage chunk が {n} 個ある: {chunks:?}");
    assert!(
        text.trim_end().ends_with("data: [DONE]"),
        "終端が [DONE] でない:\n{text}"
    );
    // [DONE] の直前の data 行 (= usage chunk であること)。
    let before_done = text
        .trim_end()
        .strip_suffix("data: [DONE]")
        .expect("[DONE] が無い")
        .trim_end();
    let last_data = before_done
        .rfind("data: ")
        .map(|i| &before_done[i + "data: ".len()..])
        .expect("data 行が無い");
    let parsed: Value = serde_json::from_str(last_data.trim()).expect("直前 data が JSON でない");
    assert_eq!(
        parsed, *last,
        "直前の data 行と最後の chunk が一致しない (順序が怪しい)"
    );
    usage_of(last)
}

/// 最後以外の chunk がすべて `"usage": null` を持つ。
fn assert_null_slots_except_last(chunks: &[Value]) {
    assert!(chunks.len() >= 2);
    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(
            c.get("usage"),
            Some(&Value::Null),
            "非最終 chunk の usage が明示 null でない: {c}"
        );
    }
}

/// `usage` という文字列自体が出ていない (キーも null slot も無い)。
fn assert_no_usage_at_all(text: &str) {
    assert!(!text.contains("\"usage\""), "usage が出ている:\n{text}");
}

/// finish_reason 付き (choices が空でない) 最後の chunk。
fn last_choice_chunk(chunks: &[Value]) -> &Value {
    chunks
        .iter()
        .rev()
        .find(|c| !c["choices"].as_array().map(|a| a.is_empty()).unwrap_or(true))
        .expect("choices 付き chunk が無い")
}

// ---------------------------------------------------------------------------
// /v1/chat/completions
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_stream_usage_parity_matrix() {
    let app = test_app();
    // 非ストリーミングを基準 (usage は常に 1 個)。
    let base = call(&app, "/v1/chat/completions", &chat_body(false)).await;
    assert_eq!(base.status, StatusCode::OK, "{}", base.text);
    let baseline = usage_of(&serde_json::from_str(&base.text).unwrap());
    assert!(baseline.0 > 0 && baseline.1 > 0, "usage = {baseline:?}");

    // include_usage = true: 末尾 usage chunk + 他の chunk は null slot。
    let mut body = chat_body(true);
    body["stream_options"] = json!({"include_usage": true});
    let r = call(&app, "/v1/chat/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    assert!(
        r.ctype.starts_with("text/event-stream"),
        "content-type が SSE でない: {}",
        r.ctype
    );
    let chunks = sse_chunks(&r.text);
    assert_null_slots_except_last(&chunks);
    let streamed = assert_usage_chunk_is_last(&chunks, &r.text);
    assert_eq!(streamed, baseline, "stream と non-stream の usage が違う");
    // usage chunk は他の chunk と同じ id / object / created / model。
    let usage = chunks.last().unwrap();
    assert_eq!(usage["object"], "chat.completion.chunk");
    assert_eq!(usage["id"], chunks[0]["id"]);
    assert_eq!(usage["created"], chunks[0]["created"]);
    assert_eq!(usage["model"], chunks[0]["model"]);
    // finish_reason は usage chunk より前の最後の choices 付き chunk にある。
    let lc = last_choice_chunk(&chunks[..chunks.len() - 1]);
    assert!(
        ["stop", "length", "tool_calls"].contains(
            &lc["choices"][0]["finish_reason"]
                .as_str()
                .unwrap_or_default()
        ),
        "finish_reason が想定外: {lc}"
    );

    // usage が「出ない」4 状態は usage キー自体が無い。
    for so in [
        None,
        Some(json!({"include_usage": false})),
        Some(json!({})),
        Some(Value::Null),
    ] {
        let mut b = chat_body(true);
        match so {
            None => {}
            Some(v) => b["stream_options"] = v,
        }
        let r = call(&app, "/v1/chat/completions", &b).await;
        assert_eq!(r.status, StatusCode::OK, "body = {b}");
        assert_no_usage_at_all(&r.text);
        let chunks = sse_chunks(&r.text);
        assert!(chunks.len() >= 2, "chunk が少なすぎる: {chunks:?}");
        assert!(
            chunks.iter().all(|c| c.get("usage").is_none()),
            "usage キーがある: {chunks:?}"
        );
    }

    // 未指定と include_usage: false は同じ本数・同じ形 (null slot も増えない)。
    let mut b = chat_body(true);
    b["stream_options"] = json!({"include_usage": false});
    let r_false = call(&app, "/v1/chat/completions", &b).await;
    let r_omit = call(&app, "/v1/chat/completions", &chat_body(true)).await;
    assert_eq!(
        sse_chunks(&r_false.text).len(),
        sse_chunks(&r_omit.text).len(),
        "include_usage: false で chunk 本数が変わる"
    );
}

#[tokio::test]
async fn chat_non_stream_ignores_stream_options() {
    let app = test_app();
    let plain = call(&app, "/v1/chat/completions", &chat_body(false)).await;
    assert_eq!(plain.status, StatusCode::OK, "{}", plain.text);
    let plain_v: Value = serde_json::from_str(&plain.text).unwrap();
    let plain_usage = usage_of(&plain_v);
    // 非ストリーム: usage は応答に 1 個、chunk ではない。
    assert_eq!(plain_v["object"], "chat.completion");
    assert_eq!(plain_v["choices"].as_array().map(|a| a.len()), Some(1));

    for so in [
        json!({"include_usage": true}),
        json!({"include_usage": false}),
        json!({}),
        Value::Null,
    ] {
        let mut b = chat_body(false);
        b["stream_options"] = so.clone();
        let r = call(&app, "/v1/chat/completions", &b).await;
        assert_eq!(r.status, StatusCode::OK, "so = {so} body = {}", r.text);
        let v: Value = serde_json::from_str(&r.text).unwrap();
        assert_eq!(v["object"], "chat.completion", "so = {so}");
        assert_eq!(usage_of(&v), plain_usage, "so = {so} で usage が変わる");
        assert_eq!(
            v["choices"].as_array().map(|a| a.len()),
            Some(1),
            "so = {so} で choices が変わる"
        );
    }
}

/// 生成長とプロンプト長を変えても、stream / non-stream の usage が一致する。
#[tokio::test]
async fn chat_usage_parity_across_generation_lengths() {
    let app = test_app();
    for prompt in ["hello", "t0 t1 t2 t3 t4 t5", "x"] {
        for max_tokens in [1, 2, 3, 8] {
            let body = |stream: bool| {
                json!({
                    "model": "denno-test",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": stream,
                })
            };
            let ns = call(&app, "/v1/chat/completions", &body(false)).await;
            assert_eq!(ns.status, StatusCode::OK, "{}", ns.text);
            let ns_v: Value = serde_json::from_str(&ns.text).unwrap();
            let (np, nc, _) = usage_of(&ns_v);
            assert!(np > 0, "prompt = {prompt:?} で prompt_tokens = 0");

            let mut sb = body(true);
            sb["stream_options"] = json!({"include_usage": true});
            let r = call(&app, "/v1/chat/completions", &sb).await;
            assert_eq!(r.status, StatusCode::OK, "{}", r.text);
            let chunks = sse_chunks(&r.text);
            assert_null_slots_except_last(&chunks);
            let (sp, sc, _) = assert_usage_chunk_is_last(&chunks, &r.text);
            assert_eq!(
                (sp, sc),
                (np, nc),
                "prompt = {prompt:?} max_tokens = {max_tokens} で usage が違う"
            );
            assert!(sc <= max_tokens as u64, "completion_tokens が上限超: {sc}");
            // finish_reason も stream / non-stream で一致する。
            let lc = last_choice_chunk(&chunks[..chunks.len() - 1]);
            assert_eq!(
                lc["choices"][0]["finish_reason"],
                ns_v["choices"][0]["finish_reason"],
                "finish_reason が stream / non-stream で違う"
            );
        }
    }
}

/// `logprobs` 有効でも usage chunk は 1 個・末尾で、値は非ストリームと一致する。
#[tokio::test]
async fn chat_logprobs_stream_usage_parity() {
    let app = test_app();
    let body = |stream: bool| {
        json!({
            "model": "denno-test",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 6,
            "temperature": 0.0,
            "logprobs": true,
            "top_logprobs": 1,
            "stream": stream,
        })
    };
    let ns = call(&app, "/v1/chat/completions", &body(false)).await;
    assert_eq!(ns.status, StatusCode::OK, "{}", ns.text);
    let ns_v: Value = serde_json::from_str(&ns.text).unwrap();
    let ns_usage = usage_of(&ns_v);
    assert!(ns_v["choices"][0]["logprobs"].is_object(), "非ストリームに logprobs が無い");

    let mut sb = body(true);
    sb["stream_options"] = json!({"include_usage": true});
    let r = call(&app, "/v1/chat/completions", &sb).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    let chunks = sse_chunks(&r.text);
    // logprobs chunk が実際に流れている (この経路の証拠)。
    assert!(
        chunks.iter().any(|c| c["choices"][0]["logprobs"].is_object()),
        "logprobs chunk が無い: {chunks:?}"
    );
    assert_null_slots_except_last(&chunks);
    assert_eq!(
        assert_usage_chunk_is_last(&chunks, &r.text),
        ns_usage,
        "logprobs 有効時に usage が違う"
    );
}

// ---------------------------------------------------------------------------
// 型違い (400 + OpenAI 形式 + param)
// ---------------------------------------------------------------------------

fn assert_typed_400(resp: &Resp, want_param: &str, want_msg_fragments: &[&str]) {
    assert_eq!(
        resp.status,
        StatusCode::BAD_REQUEST,
        "400 でない: {} {}",
        resp.status,
        resp.text
    );
    assert!(
        !resp.ctype.starts_with("text/event-stream"),
        "400 なのに SSE を開始している: {}",
        resp.ctype
    );
    assert!(
        !resp.text.contains("data: "),
        "400 なのに SSE data 行がある: {}",
        resp.text
    );
    let v: Value = serde_json::from_str(&resp.text).expect("エラー本文が JSON でない");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    assert_eq!(v["error"]["code"], Value::Null, "body = {v}");
    assert_eq!(
        v["error"]["param"], want_param,
        "error.param が {want_param:?} でない: {v}"
    );
    let msg = v["error"]["message"].as_str().unwrap_or_default();
    for frag in want_msg_fragments {
        assert!(
            msg.contains(frag),
            "message に {frag:?} が無い: {msg:?}"
        );
    }
}

#[tokio::test]
async fn chat_invalid_stream_options_types_are_openai_shaped_400() {
    let app = test_app();

    // include_usage が bool でない 5 形。
    for bad in [
        json!("true"),
        json!("yes"),
        json!(1),
        json!(0),
        json!([]),
        json!({}),
        Value::Null,
    ] {
        let mut body = chat_body(true);
        body["stream_options"] = json!({"include_usage": bad.clone()});
        let r = call(&app, "/v1/chat/completions", &body).await;
        assert_typed_400(
            &r,
            "stream_options.include_usage",
            &["stream_options.include_usage", "boolean"],
        );
    }

    // stream_options 自体が object でない 6 形 (null は「未指定」なので除外)。
    for bad in [
        json!("on"),
        json!(1),
        json!(0),
        json!(true),
        json!([]),
        json!([{"include_usage": true}]),
    ] {
        let mut body = chat_body(true);
        body["stream_options"] = bad.clone();
        let r = call(&app, "/v1/chat/completions", &body).await;
        assert_typed_400(&r, "stream_options", &["stream_options", "object"]);
    }
}

#[tokio::test]
async fn completions_invalid_stream_options_types_are_openai_shaped_400() {
    let app = test_app();
    let mut body = completion_body(true);
    body["stream_options"] = json!({"include_usage": "yes"});
    let r = call(&app, "/v1/completions", &body).await;
    assert_typed_400(
        &r,
        "stream_options.include_usage",
        &["stream_options.include_usage", "boolean"],
    );

    let mut body = completion_body(true);
    body["stream_options"] = json!(3);
    let r = call(&app, "/v1/completions", &body).await;
    assert_typed_400(&r, "stream_options", &["stream_options", "object"]);
}

// ---------------------------------------------------------------------------
// /v1/completions (legacy text completion)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn completions_stream_usage_parity_matrix() {
    let app = test_app();
    let base = call(&app, "/v1/completions", &completion_body(false)).await;
    assert_eq!(base.status, StatusCode::OK, "{}", base.text);
    let base_v: Value = serde_json::from_str(&base.text).unwrap();
    let baseline = usage_of(&base_v);
    assert_eq!(base_v["object"], "text_completion");

    let mut body = completion_body(true);
    body["stream_options"] = json!({"include_usage": true});
    let r = call(&app, "/v1/completions", &body).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    let chunks = sse_chunks(&r.text);
    assert_null_slots_except_last(&chunks);
    let streamed = assert_usage_chunk_is_last(&chunks, &r.text);
    assert_eq!(streamed, baseline, "stream と non-stream の usage が違う");
    let usage = chunks.last().unwrap();
    assert_eq!(usage["object"], "text_completion");
    assert_eq!(usage["id"], chunks[0]["id"]);
    for c in &chunks {
        assert_eq!(c["object"], "text_completion", "chunk の object: {c}");
    }

    for so in [
        None,
        Some(json!({"include_usage": false})),
        Some(json!({})),
        Some(Value::Null),
    ] {
        let mut b = completion_body(true);
        if let Some(v) = so {
            b["stream_options"] = v;
        }
        let r = call(&app, "/v1/completions", &b).await;
        assert_eq!(r.status, StatusCode::OK, "body = {b}");
        assert_no_usage_at_all(&r.text);
    }

    // 非ストリームは stream_options を無視する。
    for so in [json!({"include_usage": true}), json!({"include_usage": false})] {
        let mut b = completion_body(false);
        b["stream_options"] = so.clone();
        let r = call(&app, "/v1/completions", &b).await;
        assert_eq!(r.status, StatusCode::OK, "{}", r.text);
        let v: Value = serde_json::from_str(&r.text).unwrap();
        assert_eq!(usage_of(&v), baseline, "so = {so} で usage が変わる");
    }
}

/// `echo: true` でも usage は非ストリームと一致し、echo chunk も null slot。
#[tokio::test]
async fn completions_echo_stream_usage_parity() {
    let app = test_app();
    let mk = |stream: bool| {
        json!({
            "model": "denno-test",
            "prompt": "t0 t1 t2",
            "max_tokens": 6,
            "temperature": 0.0,
            "echo": true,
            "stream": stream,
        })
    };
    let ns = call(&app, "/v1/completions", &mk(false)).await;
    assert_eq!(ns.status, StatusCode::OK, "{}", ns.text);
    let ns_v: Value = serde_json::from_str(&ns.text).unwrap();
    let ns_usage = usage_of(&ns_v);
    // echo は非ストリームでも text の先頭にプロンプトを付ける。
    assert!(
        ns_v["choices"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .starts_with("t0 t1 t2"),
        "echo が効いていない: {ns_v}"
    );

    let mut sb = mk(true);
    sb["stream_options"] = json!({"include_usage": true});
    let r = call(&app, "/v1/completions", &sb).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    let chunks = sse_chunks(&r.text);
    assert!(chunks.len() >= 3, "chunk が少なすぎる: {chunks:?}");
    // 最初の chunk が echo (プロンプト全文) で、そこにも usage: null が付く。
    assert_eq!(
        chunks[0]["choices"][0]["text"], "t0 t1 t2",
        "最初の chunk が echo でない: {}",
        chunks[0]
    );
    assert_null_slots_except_last(&chunks);
    assert_eq!(
        assert_usage_chunk_is_last(&chunks, &r.text),
        ns_usage,
        "echo 時に usage が違う (プロンプト二重計上?)"
    );
}
