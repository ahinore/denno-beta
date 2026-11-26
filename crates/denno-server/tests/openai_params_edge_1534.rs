//! task 1534: OpenAI `/v1/chat/completions` / `/v1/completions` の
//! パラメータ境界テスト (stop / max_tokens / temperature / top_p / n / seed)。
//!
//! P966 (`openai_params_966.rs`) と P1164 (`openai_params_edge_1164.rs`) が
//! 「範囲外は param 名付き 400」を固定しているのに対し、ここでは
//! **受理側の意味**と 2 経路 (chat / completions) の一致を固定する:
//!
//! - `stop`: 文字列と 1 要素配列が同一結果、複数指定は最長一致ではなく
//!   「最も早い出現」で打ち切り、順序に依存しない
//! - `stop`: 空文字列 / 空配列 / `null` / 空要素のみは「停止なし」、
//!   配列内の非文字列要素は捨てる、巨大な停止リスト・巨大な停止語も受理
//! - `stop`: 停止語自体は本文に残らない (非ストリーミング / SSE 両方)
//! - `max_tokens` と `max_completion_tokens`: 同値なら同一結果、両方指定時は
//!   `max_tokens` 優先、`0` は空本文 + `finish_reason = "length"`、
//!   語彙の自然終端との境界 (7 / 8) で `length` と `stop` が切り替わる
//! - `temperature` / `top_p`: 境界値 (0.0 / 2.0 / 1.0 / 最小正値) は受理し、
//!   範囲外は 400
//! - `n`: `1` と未指定のみ受理、それ以外は 400 `error.param = "n"`、
//!   非整数も 400
//! - `seed`: 同一 seed + 同一入力は再現する (別 AppState でも同一)、
//!   異なる seed は異なる系列、負値 seed も決定的、型違いは 400
//!
//! ダミーモデルは greedy で `t5 t6 t7 t8 t9 t10 t11` を生成し (語彙 16、
//! EOS = 12)、temperature 0 では常に同じ列になる。seed のテストだけは
//! 一様分布モデル (語彙 0..=11 を logit 0) を使う。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DummyModel, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const CHAT: &str = "/v1/chat/completions";
const CMPL: &str = "/v1/completions";

/// greedy 時の全文 (EOS の直前まで)。
const GREEDY: &str = "t5 t6 t7 t8 t9 t10 t11";
/// `/v1/completions` は chat template を通さないので 1 トークンずれる。
const GREEDY_CMPL: &str = "t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11";

/// 12 候補の一様分布モデル (EOS / 未定義語は -1e30)。seed の再現性検証用。
struct UniformModel;

impl Model for UniformModel {
    fn context(&self) -> usize {
        512
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> anyhow::Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let mut logits = vec![-1e30f32; DUMMY_VOCAB];
        for l in logits.iter_mut().take(12) {
            *l = 0.0;
        }
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, 512)
    }
}

fn app_with(model: Box<dyn Model>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(model, tok, "denno-test", None);
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

fn app() -> Router {
    app_with(Box::new(DummyModel))
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

/// 生 body をそのまま送る (JSON の数値表記を保ちたいとき用)。
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

fn chat_body(extra: Value) -> Value {
    let mut b = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "t0"}],
    });
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn cmpl_body(extra: Value) -> Value {
    let mut b = json!({"model": "denno-test", "prompt": "t0"});
    for (k, v) in extra.as_object().unwrap() {
        b[k] = v.clone();
    }
    b
}

fn chat_text(v: &Value) -> &str {
    v["choices"][0]["message"]["content"].as_str().unwrap_or("")
}

fn cmpl_text(v: &Value) -> &str {
    v["choices"][0]["text"].as_str().unwrap_or("")
}

fn finish(v: &Value) -> &str {
    v["choices"][0]["finish_reason"].as_str().unwrap_or("")
}

/// SSE 本文から `data: ...` 行の JSON を取り出す (`[DONE]` は除外)。
fn sse_chunks(text: &str) -> Vec<Value> {
    text.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).expect("SSE data が JSON ではない"))
        .collect()
}

fn sse_content(chunks: &[Value]) -> String {
    chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

async fn post_sse(app: &Router, body: Value) -> Vec<Value> {
    let req = Request::builder()
        .method("POST")
        .uri(CHAT)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").and_then(|v| v.to_str().ok()),
        Some("text/event-stream")
    );
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    sse_chunks(&String::from_utf8(bytes.to_vec()).unwrap())
}

fn assert_invalid(status: StatusCode, v: &Value, param: Option<&str>) {
    assert_eq!(status, StatusCode::BAD_REQUEST, "body = {v}");
    assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    assert_eq!(v["error"]["param"].as_str(), param, "body = {v}");
}

// ---------------------------------------------------------------------------
// stop: 文字列 vs 配列
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_string_and_single_element_array_are_identical() {
    let app = app();
    let base = json!({"temperature": 0.0, "max_tokens": 32});
    let mut with_str = base.clone();
    with_str["stop"] = json!("t8");
    let mut with_arr = base.clone();
    with_arr["stop"] = json!(["t8"]);

    let (s1, a) = post(&app, CHAT, chat_body(with_str)).await;
    let (s2, b) = post(&app, CHAT, chat_body(with_arr)).await;
    assert_eq!(s1, StatusCode::OK);
    assert_eq!(s2, StatusCode::OK);
    assert_eq!(chat_text(&a), chat_text(&b));
    assert_eq!(finish(&a), finish(&b));
    assert_eq!(a["usage"], b["usage"]);
    // 停止語の直前まで (語間の空白は本文の一部)
    assert_eq!(chat_text(&a), "t5 t6 t7 ");
    assert_eq!(finish(&a), "stop");
}

#[tokio::test]
async fn stop_completions_string_and_array_are_identical() {
    let app = app();
    let (s1, a) = post(
        &app,
        CMPL,
        cmpl_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": "t8"})),
    )
    .await;
    let (s2, b) = post(
        &app,
        CMPL,
        cmpl_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": ["t8"]})),
    )
    .await;
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
    assert_eq!(cmpl_text(&a), cmpl_text(&b));
    assert_eq!(cmpl_text(&a), "t1 t2 t3 t4 t5 t6 t7 ");
    assert_eq!(finish(&a), "stop");
}

#[tokio::test]
async fn stop_streaming_truncates_without_leaking_the_stop_word() {
    let app = app();
    let mut body = chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stream": true}));
    body["stop"] = json!("t8");
    let chunks = post_sse(&app, body).await;
    assert_eq!(sse_content(&chunks), "t5 t6 t7");
    let last = chunks.last().expect("chunk が無い");
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn stop_multiple_sequences_cut_at_the_earliest_match_not_the_longest() {
    let app = app();
    // "t7" は "t10" より前に現れる。指定順も入れ替えて順序非依存を確認する。
    for stop in [json!(["t10", "t7"]), json!(["t7", "t10"])] {
        let (s, v) = post(
            &app,
            CHAT,
            chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": stop})),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        assert_eq!(chat_text(&v), "t5 t6 ", "stop = {stop}");
    }
}

#[tokio::test]
async fn stop_multi_token_phrase_truncates_at_phrase_start() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": ["t6 t7"]})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), "t5 ");
    assert_eq!(finish(&v), "stop");
}

#[tokio::test]
async fn stop_not_present_generates_the_full_sequence() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": ["never", "zz"]})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), GREEDY);
    // EOS による自然終端は OpenAI でも "stop"
    assert_eq!(finish(&v), "stop");
}

#[tokio::test]
async fn stop_whitespace_only_matches_the_first_space() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": " "})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), "t5");
}

// ---------------------------------------------------------------------------
// stop: 空 / 過大
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stop_empty_forms_mean_no_stop() {
    let app = app();
    let forms = [
        json!(""),
        json!([]),
        Value::Null,
        json!(["", ""]),
        json!([null, 7, true, "", {}]),
    ];
    for stop in forms {
        let (s, v) = post(
            &app,
            CHAT,
            chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": stop})),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "stop = {stop}");
        assert_eq!(chat_text(&v), GREEDY, "stop = {stop}");
    }
}

#[tokio::test]
async fn stop_empty_entries_are_dropped_but_real_ones_still_apply() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": ["", "t9"]})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), "t5 t6 t7 t8 ");
}

#[tokio::test]
async fn stop_oversized_list_is_accepted_and_does_not_change_output() {
    let app = app();
    let big: Vec<String> = (0..2000).map(|i| format!("zz{i}")).collect();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": big})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), GREEDY);
}

#[tokio::test]
async fn stop_oversized_single_sequence_is_accepted() {
    let app = app();
    let long = "q".repeat(4096);
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 32, "stop": long})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), GREEDY);
}

#[tokio::test]
async fn stop_non_string_scalars_are_rejected_on_both_endpoints() {
    let app = app();
    for bad in [json!(5), json!(true), json!({"a": 1})] {
        let (s, v) = post(
            &app,
            CHAT,
            chat_body(json!({"temperature": 0.0, "max_tokens": 8, "stop": bad})),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "stop = {bad}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "stop = {bad}");

        let (s, v) = post(
            &app,
            CMPL,
            cmpl_body(json!({"temperature": 0.0, "max_tokens": 8, "stop": bad})),
        )
        .await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "stop = {bad}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "stop = {bad}");
    }
}

// ---------------------------------------------------------------------------
// max_tokens vs max_completion_tokens
// ---------------------------------------------------------------------------

#[tokio::test]
async fn max_tokens_and_max_completion_tokens_are_interchangeable() {
    let app = app();
    let (s1, a) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 3})),
    )
    .await;
    let (s2, b) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_completion_tokens": 3})),
    )
    .await;
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
    assert_eq!(chat_text(&a), "t5 t6 t7");
    assert_eq!(chat_text(&a), chat_text(&b));
    assert_eq!(a["usage"], b["usage"]);
    assert_eq!(finish(&a), "length");
}

#[tokio::test]
async fn max_tokens_wins_when_both_are_given() {
    let app = app();
    // max_tokens が小さい → 3 トークンで止まる
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 3, "max_completion_tokens": 6})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), "t5 t6 t7");

    // max_tokens が大きい → max_completion_tokens の 3 は無視される
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 6, "max_completion_tokens": 3})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), "t5 t6 t7 t8 t9 t10");
}

#[tokio::test]
async fn max_tokens_zero_yields_empty_content_with_length_finish() {
    let app = app();
    for body in [
        chat_body(json!({"temperature": 0.0, "max_tokens": 0})),
        chat_body(json!({"temperature": 0.0, "max_completion_tokens": 0})),
    ] {
        let (s, v) = post(&app, CHAT, body).await;
        assert_eq!(s, StatusCode::OK, "body = {v}");
        assert_eq!(chat_text(&v), "");
        assert_eq!(finish(&v), "length");
        assert_eq!(v["usage"]["completion_tokens"], 0);
    }

    let (s, v) = post(
        &app,
        CMPL,
        cmpl_body(json!({"temperature": 0.0, "max_tokens": 0})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(cmpl_text(&v), "");
    assert_eq!(finish(&v), "length");
}

#[tokio::test]
async fn max_tokens_boundary_between_length_and_natural_stop() {
    let app = app();
    // 全文は 7 トークン。7 で切ると "length"、8 なら EOS を 1 ステップ余裕で
    // 引けるので本文は同じでも "stop" になる。
    let (s, seven) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 7})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&seven), GREEDY);
    assert_eq!(finish(&seven), "length");
    assert_eq!(seven["usage"]["completion_tokens"], 7);

    let (s, eight) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 8})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&eight), GREEDY);
    assert_eq!(finish(&eight), "stop");
    assert_eq!(eight["usage"]["completion_tokens"], 8);
}

#[tokio::test]
async fn max_tokens_giant_value_is_clamped_not_rejected() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": i64::MAX})),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(chat_text(&v), GREEDY);
    assert_eq!(finish(&v), "stop");
}

#[tokio::test]
async fn max_completion_tokens_negative_is_rejected_with_its_param_name() {
    let app = app();
    for uri in [CHAT, CMPL] {
        let (s, v) = post(
            &app,
            uri,
            if uri == CHAT {
                chat_body(json!({"temperature": 0.0, "max_completion_tokens": -1}))
            } else {
                cmpl_body(json!({"temperature": 0.0, "max_completion_tokens": -1}))
            },
        )
        .await;
        assert_invalid(s, &v, Some("max_completion_tokens"));
    }
}

#[tokio::test]
async fn both_token_limits_negative_reports_max_tokens_first() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": -1, "max_completion_tokens": -2})),
    )
    .await;
    assert_invalid(s, &v, Some("max_tokens"));
}

#[tokio::test]
async fn completions_honours_both_token_limit_fields() {
    let app = app();
    let (s1, a) = post(
        &app,
        CMPL,
        cmpl_body(json!({"temperature": 0.0, "max_tokens": 2})),
    )
    .await;
    let (s2, b) = post(
        &app,
        CMPL,
        cmpl_body(json!({"temperature": 0.0, "max_completion_tokens": 2})),
    )
    .await;
    assert_eq!((s1, s2), (StatusCode::OK, StatusCode::OK));
    assert_eq!(cmpl_text(&a), "t1 t2");
    assert_eq!(cmpl_text(&a), cmpl_text(&b));
}

// ---------------------------------------------------------------------------
// temperature / top_p の境界 (受理側)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn temperature_and_top_p_boundary_values_are_accepted() {
    let app = app();
    // temperature 0 の 3 ケースは greedy なので本文まで固定できる。
    for extra in [
        json!({"temperature": 0.0, "top_p": 1.0}),
        json!({"temperature": -0.0, "top_p": 0.5}),
        json!({"top_p": 1e-7}),
    ] {
        let (s, v) = post(&app, CHAT, chat_body(extra.clone())).await;
        assert_eq!(s, StatusCode::OK, "extra = {extra}; body = {v}");
        assert_eq!(chat_text(&v), GREEDY, "extra = {extra}");
    }
    // OpenAI 上限ちょうどの temperature 2.0 は受理される (本文はサンプリング依存)。
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 2.0, "top_p": 1.0, "max_tokens": 8})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "body = {v}");
    assert!(!chat_text(&v).is_empty(), "body = {v}");
}

#[tokio::test]
async fn temperature_and_top_p_out_of_range_are_rejected() {
    let app = app();
    for (extra, param) in [
        (json!({"temperature": 2.5}), "temperature"),
        (json!({"temperature": -0.5}), "temperature"),
        (json!({"top_p": 0.0}), "top_p"),
        (json!({"top_p": -0.0}), "top_p"),
        (json!({"top_p": 1.5}), "top_p"),
    ] {
        let (s, v) = post(&app, CHAT, chat_body(extra.clone())).await;
        assert_invalid(s, &v, Some(param));
        let (s, v) = post(&app, CMPL, cmpl_body(extra.clone())).await;
        assert_invalid(s, &v, Some(param));
    }
}

#[tokio::test]
async fn non_finite_temperature_and_top_p_are_rejected_via_raw_json() {
    let app = app();
    // serde_json の Value は NaN/inf を表現できないため生 body で送る。
    for body in [
        "{\"model\":\"denno-test\",\"messages\":[{\"role\":\"user\",\"content\":\"t0\"}],\"temperature\":1e400}",
        "{\"model\":\"denno-test\",\"messages\":[{\"role\":\"user\",\"content\":\"t0\"}],\"top_p\":1e400}",
    ] {
        let (s, v) = post_raw(&app, CHAT, body).await;
        assert_eq!(s, StatusCode::BAD_REQUEST, "body = {v}");
        assert_eq!(v["error"]["type"], "invalid_request_error", "body = {v}");
    }
}

// ---------------------------------------------------------------------------
// n > 1 の検証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn n_only_accepts_one() {
    let app = app();
    for uri in [CHAT, CMPL] {
        for n in [json!(1), Value::Null] {
            let body = if uri == CHAT {
                chat_body(json!({"temperature": 0.0, "max_tokens": 3, "n": n}))
            } else {
                cmpl_body(json!({"temperature": 0.0, "max_tokens": 3, "n": n}))
            };
            let (s, v) = post(&app, uri, body).await;
            assert_eq!(s, StatusCode::OK, "uri = {uri}; n = {n}; body = {v}");
        }
        for n in [json!(0), json!(2), json!(-1), json!(100)] {
            let body = if uri == CHAT {
                chat_body(json!({"temperature": 0.0, "max_tokens": 3, "n": n}))
            } else {
                cmpl_body(json!({"temperature": 0.0, "max_tokens": 3, "n": n}))
            };
            let (s, v) = post(&app, uri, body).await;
            assert_invalid(s, &v, Some("n"));
        }
    }
}

#[tokio::test]
async fn n_non_integer_is_rejected() {
    let app = app();
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 0.0, "max_tokens": 3, "n": 1.5})),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

// ---------------------------------------------------------------------------
// seed の再現性
// ---------------------------------------------------------------------------

async fn seeded_text(app: &Router, seed: i64) -> String {
    let (s, v) = post(
        app,
        CHAT,
        chat_body(json!({"temperature": 1.0, "max_tokens": 8, "seed": seed})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "seed = {seed}; body = {v}");
    assert_eq!(v["usage"]["completion_tokens"], 8, "seed = {seed}");
    chat_text(&v).to_string()
}

#[tokio::test]
async fn same_seed_is_reproducible_and_different_seeds_diverge() {
    let app = app_with(Box::new(UniformModel));
    let a = seeded_text(&app, 42).await;
    let b = seeded_text(&app, 42).await;
    assert_eq!(a, b, "same seed must reproduce");

    // 8 トークン x 12 候補なので偶然一致する確率は無視できる。
    let mut others = Vec::new();
    for seed in [0, 43, 12345] {
        others.push(seeded_text(&app, seed).await);
    }
    assert!(
        others.iter().any(|t| *t != a),
        "different seeds produced identical text: {others:?}"
    );
}

#[tokio::test]
async fn negative_seed_is_deterministic() {
    let app = app_with(Box::new(UniformModel));
    let a = seeded_text(&app, -1).await;
    let b = seeded_text(&app, -1).await;
    assert_eq!(a, b, "negative seed must reproduce");
    assert_eq!(a.split(' ').count(), 8);
}

#[tokio::test]
async fn seed_reproduces_across_fresh_server_instances() {
    let a = seeded_text(&app_with(Box::new(UniformModel)), 7).await;
    let b = seeded_text(&app_with(Box::new(UniformModel)), 7).await;
    assert_eq!(a, b, "seed must not depend on process-local state");
}

#[tokio::test]
async fn seed_reproducibility_survives_stop_truncation() {
    let app = app_with(Box::new(UniformModel));
    let mut first: Option<String> = None;
    for _ in 0..3 {
        let (s, v) = post(
            &app,
            CHAT,
            chat_body(json!({
                "temperature": 1.0,
                "max_tokens": 8,
                "seed": 99,
                "stop": ["t3"],
            })),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "body = {v}");
        let text = chat_text(&v).to_string();
        // stop 語 "t3" は本文に残らない (単独トークンとして一致した場合)
        assert!(
            !text.contains("t3"),
            "stop 語が本文に残っている: {text:?}"
        );
        match &first {
            None => first = Some(text),
            Some(f) => assert_eq!(f, &text, "seed + stop must reproduce"),
        }
    }
}

#[tokio::test]
async fn seed_null_means_unspecified_and_seed_type_errors_are_400() {
    let app = app_with(Box::new(UniformModel));
    let (s, v) = post(
        &app,
        CHAT,
        chat_body(json!({"temperature": 1.0, "max_tokens": 4, "seed": Value::Null})),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "body = {v}");

    for bad in [json!("42"), json!(1.5), json!(true), json!([42])] {
        let (s, v) = post(
            &app,
            CHAT,
            chat_body(json!({"temperature": 1.0, "max_tokens": 4, "seed": bad})),
        )
        .await;
        assert_invalid(s, &v, Some("seed"));
    }
}
