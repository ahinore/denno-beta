//! task 1164: OpenAI サンプリング境界パラメータの追加エッジテスト。
//!
//! P966 (`openai_params_966.rs`) が「範囲外は param 名付き 400」の骨格を固定して
//! いるのに対し、ここでは境界の**端**を掘る:
//!
//! - `temperature`: f32 の極端値 (denormal / f32::MIN_POSITIVE / f32::MAX / 桁溢れ)、
//!   `-0.0`、f64→f32 丸めで境界がどこに落ちるか、0.0 / 1.0 / 2.0 のちょうど
//! - `top_p`: (0, 1] の端 (denormal 正値は可、`-0.0` と 0.0 は不可、1+ε の丸め)
//! - `n`: 1 以外は i64 の端 (i64::MIN / i64::MAX) でも 400、型不正も 400
//! - `max_tokens` / `max_completion_tokens`: 負値の端 (i64::MIN)、両方負のときの
//!   報告順、正の i64::MAX は従来どおり clamp
//! - `stop`: 数値・真偽値・オブジェクトは 400、配列内の非文字列は捨てられる、
//!   空配列 / 空文字 / null は「無し」
//! - どの 400 も `{"error": {...}}` の 4 フィールドだけを持つ整った JSON で、
//!   `code` は null、内部情報 (panic / ソースパス) を漏らさない
//!
//! chat / completions の両経路は検証関数を共有しているので、基本的に同じ
//! 変形を両方へ流して一致を確認する。

use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::kv_cache::KvCache;
use denno_engine::model::Model;
use denno_engine::testutil::{dummy_tokenizer, DUMMY_EOS, DUMMY_VOCAB};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler, MAX_TOKENS_CLAMP};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

const TEST_CONTEXT: usize = 8192;

/// openai_params_966.rs と同一の決定的ダミーモデル (peak = 直前トークン + 1)。
/// EOS は出さないので `max_tokens` を必ず使い切る。
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
        let mut peak = (last + 1) % DUMMY_VOCAB;
        if peak == DUMMY_EOS as usize {
            peak += 1;
        }
        logits[peak] = 5.0;
        logits[(peak + 1) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, TEST_CONTEXT)
    }
}

fn test_app_with_cap(temperature_cap: Option<f32>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(SeqModel), tok, "denno-test", None);
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
        temperature_cap,
    });
    denno_server::router(state)
}

fn test_app() -> Router {
    test_app_with_cap(None)
}

/// 1 応答分 (状態行・content-type・生 body・パース結果)。
struct Resp {
    status: StatusCode,
    content_type: Option<String>,
    raw: String,
    json: Value,
}

async fn send(app: &Router, uri: &str, body: String) -> Resp {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let raw = String::from_utf8_lossy(&bytes).to_string();
    let json = serde_json::from_str(&raw).unwrap_or(Value::Null);
    Resp { status, content_type, raw, json }
}

/// `extra` は JSON のフィールド断片 (`"temperature":1e-45` 等)。
/// 数値の表記を保ちたいので Value を経由せず文字列で組む。
fn chat_raw(extra: &str) -> String {
    format!(
        "{{\"model\":\"denno-test\",\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}],{extra}}}"
    )
}

fn cmpl_raw(extra: &str) -> String {
    format!("{{\"model\":\"denno-test\",\"prompt\":\"hi\",{extra}}}")
}

const CHAT: &str = "/v1/chat/completions";
const CMPL: &str = "/v1/completions";

/// 同じフィールド断片を両エンドポイントへ流す。
async fn post_both_raw(app: &Router, extra: &str) -> [Resp; 2] {
    [
        send(app, CHAT, chat_raw(extra)).await,
        send(app, CMPL, cmpl_raw(extra)).await,
    ]
}

async fn post_both_json(app: &Router, mutate: impl Fn(&mut Value)) -> [Resp; 2] {
    let mut chat = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "temperature": 0.0,
    });
    mutate(&mut chat);
    let mut cmpl = json!({
        "model": "denno-test",
        "prompt": "hi",
        "max_tokens": 4,
        "temperature": 0.0,
    });
    mutate(&mut cmpl);
    [
        send(app, CHAT, chat.to_string()).await,
        send(app, CMPL, cmpl.to_string()).await,
    ]
}

/// 400 invalid_request_error の形と param を固定する。
/// `param = None` は「param が JSON null」(body パース失敗など) を意味する。
fn assert_invalid(r: &Resp, param: Option<&str>, ctx: &str) {
    assert_eq!(r.status, StatusCode::BAD_REQUEST, "{ctx}: {}", r.raw);
    assert_eq!(r.json["error"]["type"], "invalid_request_error", "{ctx}: {}", r.raw);
    match param {
        Some(p) => assert_eq!(r.json["error"]["param"], p, "{ctx}: {}", r.raw),
        None => assert!(r.json["error"]["param"].is_null(), "{ctx}: {}", r.raw),
    }
    let msg = r.json["error"]["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "{ctx}: message が空: {}", r.raw);
}

/// 400 本文が「整った」OpenAI 形式であることを固定する:
/// top-level は `error` だけ、error は message/type/param/code の 4 つだけ、
/// code は null、content-type は application/json、内部情報を漏らさない。
fn assert_clean_400(r: &Resp, ctx: &str) {
    assert_eq!(r.status, StatusCode::BAD_REQUEST, "{ctx}: {}", r.raw);
    let ct = r.content_type.clone().unwrap_or_default();
    assert!(
        ct.starts_with("application/json"),
        "{ctx}: content-type {ct:?} は JSON でない"
    );
    let top = r.json.as_object().unwrap_or_else(|| panic!("{ctx}: JSON でない: {}", r.raw));
    assert_eq!(top.len(), 1, "{ctx}: top-level に余計なキー: {}", r.raw);
    let err = top
        .get("error")
        .and_then(|e| e.as_object())
        .unwrap_or_else(|| panic!("{ctx}: error が object でない: {}", r.raw));
    let mut keys: Vec<&str> = err.keys().map(|k| k.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        ["code", "message", "param", "type"],
        "{ctx}: error のキー集合: {}",
        r.raw
    );
    assert!(err["code"].is_null(), "{ctx}: code は null のはず: {}", r.raw);
    assert_eq!(err["type"], "invalid_request_error", "{ctx}: {}", r.raw);
    let msg = err["message"].as_str().unwrap_or_default();
    assert!(!msg.is_empty(), "{ctx}: message が空: {}", r.raw);
    for leak in ["panicked", "src/", "openai.rs", "unwrap"] {
        assert!(!msg.contains(leak), "{ctx}: message が内部情報を漏らす ({leak}): {msg}");
    }
}

fn assert_ok(r: &Resp, ctx: &str) {
    assert_eq!(r.status, StatusCode::OK, "{ctx}: {}", r.raw);
}

/// greedy (temperature=0) のダミーモデルは EOS を出さないので、
/// このヘルパは temperature/top_p を固定した決定論的な要求にだけ使う。
fn assert_ok_4_tokens(r: &Resp, ctx: &str) {
    assert_eq!(r.status, StatusCode::OK, "{ctx}: {}", r.raw);
    assert_eq!(r.json["usage"]["completion_tokens"], 4, "{ctx}: {}", r.raw);
}

// ---------- temperature: 極端値 ----------

#[tokio::test]
async fn temperature_extreme_high_values_are_rejected() {
    let app = test_app();
    // f32::MAX ちょうど、桁の大きい有限値、f32 で inf になる値、f64 で inf になる値。
    for raw in ["3.4028235e38", "1e38", "1e30", "3.5e38", "1e400"] {
        let [c, p] = post_both_raw(&app, &format!("\"max_tokens\":4,\"temperature\":{raw}")).await;
        for (r, ctx) in [(&c, format!("chat temperature={raw}")), (&p, format!("completions temperature={raw}"))] {
            assert_clean_400(r, &ctx);
            // 1e400 は serde が body ごと拒否するので param は null。
            let param = if raw == "1e400" { None } else { Some("temperature") };
            assert_invalid(r, param, &ctx);
        }
    }
}

#[tokio::test]
async fn temperature_extreme_low_values_and_negative_zero() {
    let app = test_app();

    // f32 で正のまま残る極小値は受理 (denormal / MIN_POSITIVE)。
    // f64 の極小値は f32 への丸めで 0.0 / -0.0 に落ちるので、これも受理。
    for raw in [
        "1e-45",             // f32 最小 denormal 付近
        "1.1754944e-38",     // f32::MIN_POSITIVE
        "2.2250738e-308",    // f64::MIN_POSITIVE -> f32 では 0.0
        "-0.0",
        "5e-324",            // f64 最小 denormal -> f32 では 0.0
        "-5e-324",           // f32 への丸めで -0.0 (0 として扱われる)
    ] {
        let [c, p] = post_both_raw(&app, &format!("\"max_tokens\":4,\"temperature\":{raw}")).await;
        assert_ok(&c, &format!("chat temperature={raw}"));
        assert_ok(&p, &format!("completions temperature={raw}"));
    }

    // f32 で負のまま残る値は 400。
    for raw in ["-1e-45", "-1.1754944e-38", "-0.1", "-3.4028235e38"] {
        let [c, p] = post_both_raw(&app, &format!("\"max_tokens\":4,\"temperature\":{raw}")).await;
        for (r, ctx) in [(&c, format!("chat temperature={raw}")), (&p, format!("completions temperature={raw}"))] {
            assert_clean_400(r, &ctx);
            assert_invalid(r, Some("temperature"), &ctx);
        }
    }
}

#[tokio::test]
async fn temperature_exact_boundaries_0_1_2() {
    let app = test_app();
    for raw in ["0.0", "0", "1.0", "1", "2.0", "2"] {
        let [c, p] = post_both_raw(&app, &format!("\"max_tokens\":4,\"temperature\":{raw}")).await;
        assert_ok(&c, &format!("chat temperature={raw}"));
        assert_ok(&p, &format!("completions temperature={raw}"));
    }
    // 境界のすぐ外側。
    for raw in ["-0.0000001", "2.000001", "2.5"] {
        let [c, p] = post_both_raw(&app, &format!("\"max_tokens\":4,\"temperature\":{raw}")).await;
        for (r, ctx) in [(&c, format!("chat temperature={raw}")), (&p, format!("completions temperature={raw}"))] {
            assert_invalid(r, Some("temperature"), &ctx);
        }
    }
}

#[tokio::test]
async fn temperature_boundary_is_checked_on_f32_value() {
    // 判定は f32 に丸めた後なので、2.0 の 1/2 ulp 未満の超過は 2.0 として通る。
    let app = test_app();
    let [c, p] = post_both_raw(&app, "\"max_tokens\":4,\"temperature\":2.0000001").await;
    assert_ok(&c, "chat temperature=2.0000001 (丸めで 2.0)");
    assert_ok(&p, "completions temperature=2.0000001 (丸めで 2.0)");

    let [c, p] = post_both_raw(&app, "\"max_tokens\":4,\"temperature\":2.0000005").await;
    assert_invalid(&c, Some("temperature"), "chat temperature=2.0000005");
    assert_invalid(&p, Some("temperature"), "completions temperature=2.0000005");
}

#[tokio::test]
async fn temperature_cap_does_not_rescue_non_finite() {
    // cap は「cap 内へ丸める」ので巨大有限値は救うが、非有限は先に 400。
    let app = test_app_with_cap(Some(0.5));
    let c = send(&app, CHAT, chat_raw("\"max_tokens\":4,\"temperature\":1e30")).await;
    assert_ok(&c, "cap=0.5, temperature=1e30");
    let c = send(&app, CHAT, chat_raw("\"max_tokens\":4,\"temperature\":3.5e38")).await;
    assert_invalid(&c, Some("temperature"), "cap=0.5, temperature=inf");
}

// ---------- top_p: 極端値 ----------

#[tokio::test]
async fn top_p_extreme_and_boundary_values() {
    let app = test_app();

    // (0, 1] の端。denormal 正値は可、1 に丸まる超過も可。
    // temperature=0 と併用して greedy 経路に固定する (生成長を決定的にする)。
    for raw in ["1", "1.0", "0.9999999", "1.00000001", "1.1754944e-38", "1e-45", "0.5"] {
        let [c, p] =
            post_both_raw(&app, &format!("\"max_tokens\":4,\"temperature\":0,\"top_p\":{raw}")).await;
        assert_ok_4_tokens(&c, &format!("chat top_p={raw}"));
        assert_ok_4_tokens(&p, &format!("completions top_p={raw}"));
    }

    // 0 / -0.0 / 負 / 1 超 / f32 で inf。
    for raw in ["0", "0.0", "-0.0", "-1e-45", "-0.5", "1.0000005", "2", "3.5e38"] {
        let [c, p] = post_both_raw(&app, &format!("\"max_tokens\":4,\"top_p\":{raw}")).await;
        for (r, ctx) in [(&c, format!("chat top_p={raw}")), (&p, format!("completions top_p={raw}"))] {
            assert_clean_400(r, &ctx);
            assert_invalid(r, Some("top_p"), &ctx);
        }
    }

    // 非有限はメッセージで区別できる (実装の分岐を固定する)。
    let [c, _] = post_both_raw(&app, "\"max_tokens\":4,\"top_p\":3.5e38").await;
    let msg = c.json["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("finite"), "inf top_p の message: {msg}");
}

#[tokio::test]
async fn top_p_zero_is_rejected_not_treated_as_greedy() {
    // OpenAI 仕様の (0, 1] を守る: top_p=0 は「greedy 扱い」ではなく 400。
    let app = test_app();
    let c = send(&app, CHAT, chat_raw("\"max_tokens\":4,\"top_p\":0")).await;
    assert_invalid(&c, Some("top_p"), "top_p=0");
    let msg = c.json["error"]["message"].as_str().unwrap_or_default();
    assert!(msg.contains("out of range"), "top_p=0 の message: {msg}");
}

// ---------- n: 端と型 ----------

#[tokio::test]
async fn n_extreme_values_are_rejected() {
    let app = test_app();
    let [c, p] = post_both_json(&app, |b| b["n"] = json!(1)).await;
    assert_ok_4_tokens(&c, "chat n=1");
    assert_ok_4_tokens(&p, "completions n=1");

    for n in [0i64, -1, 2, 100, i64::MIN, i64::MAX] {
        let [c, p] = post_both_json(&app, |b| b["n"] = json!(n)).await;
        for (r, ctx) in [(&c, format!("chat n={n}")), (&p, format!("completions n={n}"))] {
            assert_clean_400(r, &ctx);
            assert_invalid(r, Some("n"), &ctx);
            let msg = r.json["error"]["message"].as_str().unwrap_or_default();
            assert!(msg.contains("only n=1"), "{ctx}: message: {msg}");
        }
    }

    // 型不正 (浮動小数 / 文字列) は body パース段階の 400 (param は null)。
    for frag in ["\"n\":1.0", "\"n\":\"1\"", "\"n\":true"] {
        let extra = format!("\"max_tokens\":4,{frag}");
        let [c, p] = post_both_raw(&app, &extra).await;
        for (r, ctx) in [(&c, format!("chat {frag}")), (&p, format!("completions {frag}"))] {
            assert_clean_400(r, &ctx);
            assert_invalid(r, None, &ctx);
        }
    }
}

// ---------- max_tokens / max_completion_tokens ----------

#[tokio::test]
async fn negative_max_tokens_extremes() {
    let app = test_app();
    for v in [-1i64, -4096, i64::MIN] {
        let [c, p] = post_both_json(&app, |b| b["max_tokens"] = json!(v)).await;
        for (r, ctx) in [(&c, format!("chat max_tokens={v}")), (&p, format!("completions max_tokens={v}"))] {
            assert_clean_400(r, &ctx);
            assert_invalid(r, Some("max_tokens"), &ctx);
        }
    }

    // 別名の負値は別名で報告する。
    let b = json!({
        "model": "denno-test",
        "messages": [{"role": "user", "content": "hi"}],
        "max_completion_tokens": -1,
    });
    let c = send(&app, CHAT, b.to_string()).await;
    assert_invalid(&c, Some("max_completion_tokens"), "chat max_completion_tokens=-1");

    // 両方負なら先に検査する max_tokens を報告する (決定的)。
    let c = send(
        &app,
        CHAT,
        chat_raw("\"max_tokens\":-1,\"max_completion_tokens\":-2"),
    )
    .await;
    assert_invalid(&c, Some("max_tokens"), "chat 両方負");
}

#[tokio::test]
async fn max_tokens_extreme_positive_is_clamped_not_rejected() {
    // i64::MAX は負値ではないので 400 にせず、従来どおり上限 clamp する。
    let app = test_app();
    let p = send(&app, CMPL, cmpl_raw("\"max_tokens\":9223372036854775807,\"temperature\":0")).await;
    assert_eq!(p.status, StatusCode::OK, "{}", p.raw);
    assert_eq!(
        p.json["usage"]["completion_tokens"],
        MAX_TOKENS_CLAMP as u64,
        "{}",
        p.raw
    );

    // max_completion_tokens=0 は「prefill のみ」で生成 0。
    let c = send(
        &app,
        CHAT,
        chat_raw("\"max_tokens\":0,\"max_completion_tokens\":0"),
    )
    .await;
    assert_eq!(c.status, StatusCode::OK, "{}", c.raw);
    assert_eq!(c.json["usage"]["completion_tokens"], 0, "{}", c.raw);
}

// ---------- stop ----------

#[tokio::test]
async fn stop_non_string_scalar_is_400_clean() {
    let app = test_app();
    for frag in ["\"stop\":42", "\"stop\":1.5", "\"stop\":true", "\"stop\":{\"a\":1}"] {
        let extra = format!("\"max_tokens\":4,{frag}");
        let [c, p] = post_both_raw(&app, &extra).await;
        for (r, ctx) in [(&c, format!("chat {frag}")), (&p, format!("completions {frag}"))] {
            assert_clean_400(r, &ctx);
            assert_invalid(r, None, &ctx);
            let msg = r.json["error"]["message"].as_str().unwrap_or_default();
            assert!(
                msg.contains("stop must be a string or an array of strings"),
                "{ctx}: message: {msg}"
            );
        }
    }
}

#[tokio::test]
async fn stop_array_ignores_non_string_elements() {
    // 配列内の非文字列は捨てる (文字列だけが stop として効く)。
    let app = test_app();
    for extra in [
        "\"stop\":[1,2]",
        "\"stop\":[null,true,{\"a\":1}]",
        "\"stop\":[\"END\",1]",
    ] {
        let frag = format!("\"max_tokens\":4,\"temperature\":0,{extra}");
        let [c, p] = post_both_raw(&app, &frag).await;
        assert_ok_4_tokens(&c, &format!("chat {extra}"));
        assert_ok_4_tokens(&p, &format!("completions {extra}"));
    }
}

#[tokio::test]
async fn stop_boundary_shapes_accepted() {
    // 空配列 / 空文字 / null は「stop 無し」として通る (既存の既定)。
    let app = test_app();
    for extra in [
        "\"stop\":[]",
        "\"stop\":\"\"",
        "\"stop\":null",
        "\"stop\":\"END\"",
        "\"stop\":[\"\"]",
    ] {
        let frag = format!("\"max_tokens\":4,\"temperature\":0,{extra}");
        let [c, p] = post_both_raw(&app, &frag).await;
        assert_ok_4_tokens(&c, &format!("chat {extra}"));
        assert_ok_4_tokens(&p, &format!("completions {extra}"));
    }
}

// ---------- 400 本文の形 / 状態の健全性 ----------

#[tokio::test]
async fn every_400_is_a_clean_openai_error_object() {
    let app = test_app();
    let cases: Vec<String> = vec![
        chat_raw("\"max_tokens\":4,\"temperature\":3"),
        chat_raw("\"max_tokens\":4,\"top_p\":2"),
        chat_raw("\"max_tokens\":4,\"n\":2"),
        chat_raw("\"max_tokens\":-1"),
        chat_raw("\"max_tokens\":4,\"stop\":5"),
        cmpl_raw("\"max_tokens\":4,\"temperature\":-1"),
        cmpl_raw("\"max_tokens\":4,\"top_p\":0"),
        "{not json".to_string(),
    ];
    for body in cases {
        let r = send(&app, if body.contains("\"messages\"") { CHAT } else { CMPL }, body.clone()).await;
        assert_clean_400(&r, &format!("body={body}"));
    }
}

#[tokio::test]
async fn server_stays_healthy_after_many_rejections() {
    let app = test_app();
    let rejected: Vec<String> = vec![
        "\"max_tokens\":4,\"temperature\":-1e-45".to_string(),
        "\"max_tokens\":4,\"top_p\":1.5".to_string(),
        "\"max_tokens\":4,\"n\":7".to_string(),
        "\"max_tokens\":-2".to_string(),
        "\"max_tokens\":4,\"temperature\":0,\"stop\":[1,{\"a\":2}]".to_string(),
    ];
    for _ in 0..3 {
        for extra in &rejected {
            let r = send(&app, CHAT, chat_raw(extra)).await;
            // 配列内の非文字列は捨てられるので、この 1 件だけは 200。
            if extra.contains("stop") {
                assert_ok_4_tokens(&r, "stop 配列の非文字列は無視");
            } else {
                assert_eq!(r.status, StatusCode::BAD_REQUEST, "{}", r.raw);
            }
        }
    }
    let c = send(&app, CHAT, chat_raw("\"max_tokens\":4,\"temperature\":0.7,\"top_p\":0.9")).await;
    assert_ok(&c, "rejections 後の通常要求");
}
