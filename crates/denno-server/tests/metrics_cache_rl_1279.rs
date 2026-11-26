//! task 1279: `/metrics` の prefix cache 計器とレート制限 429 拒否カウンタの
//! **単調性** 契約テスト。
//!
//! `metrics_extra_1011.rs` は「系列が出る / 1 回の記録で 1 増える」ことを、
//! `metrics_edge_1024.rs` は exposition 全体の文法を見る。ここでは重複を避け、
//! 次の 2 点だけを固定する:
//!
//!   1. prefix cache の `hits` / `misses` / `reused_tokens` は記録のたびに
//!      正確な差分で増え、複数回スクレイプしても決して減らない (counter の
//!      単調性)。`hit_rate` は各スクレイプ時点の累積値と一致する。
//!   2. トークンバケットの 429 拒否は、拒否した回数ぶんだけ
//!      `denno_ratelimit_rejections_total{reason="token_bucket"}` と
//!      `denno_http_requests_total{status="429"}` の両方に載り、スクレイプ間で
//!      減らない。429 は prefix cache の計器を汚さない。
//!
//! 計器はプロセス全体で共有されるため、このテストバイナリ内で `SERIAL` により
//! 直列化し、期待値は絶対値ではなく開始時点からの差分で見る。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::metrics;
use denno_server::ratelimit::{RateLimitConfig, RateLimiter};
use denno_server::{AppState, CorsConfig, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// 共有計器を触るテストを 1 本ずつ実行するためのロック。
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

// ---------------------------------------------------------------------------
// アプリ生成
// ---------------------------------------------------------------------------

fn state() -> Arc<AppState> {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
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

fn app_with(rpm: u32, burst: Option<u32>) -> Router {
    let limiter = RateLimiter::from_config(RateLimitConfig::new(rpm, burst));
    denno_server::router_with_rate_limit(state(), CorsConfig::default(), limiter)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let ct = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    (status, ct, String::from_utf8_lossy(&bytes).to_string())
}

async fn scrape(app: &Router) -> String {
    let (status, ct, body) = get(app, "/metrics").await;
    assert_eq!(status, StatusCode::OK, "/metrics が 200 でない");
    assert!(
        ct.starts_with("text/plain; version=0.0.4"),
        "Content-Type = {ct}"
    );
    body
}

async fn post(app: &Router, uri: &str, body: &Value) -> StatusCode {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(req).await.unwrap().status()
}

// ---------------------------------------------------------------------------
// exposition format 0.0.4 の最小検証器
// ---------------------------------------------------------------------------

/// メトリクス名: `[a-zA-Z_:][a-zA-Z0-9_:]*`
fn valid_metric_name(name: &str) -> bool {
    let mut it = name.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == ':' => {}
        _ => return false,
    }
    it.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

/// `name{labels} value` / `name value` を 1 行読み `(系列キー, 名前, 値)` を返す。
///
/// `metrics.rs` はラベル値のエスケープに `\\`, `\"`, `\n` だけを使う。ここでは
/// 生の改行・CR と、閉じていない引用を弾ければ足りる (本格的な検証は
/// `metrics_edge_1024.rs` の担当)。
fn parse_sample(line: &str) -> Result<(String, String, f64), String> {
    let (series, value_text) = line
        .split_once(' ')
        .ok_or_else(|| "名前と値の区切りが無い".to_string())?;
    if value_text.is_empty() || value_text.contains(' ') {
        return Err(format!("値の書式が不正: {value_text:?}"));
    }
    let value: f64 = value_text
        .parse()
        .map_err(|_| format!("値が数値でない: {value_text:?}"))?;
    if !value.is_finite() {
        return Err(format!("値が有限でない: {value_text:?}"));
    }

    let (name, label_text) = match series.find('{') {
        Some(open) => {
            if !series.ends_with('}') {
                return Err(format!("ラベル集合が閉じていない: {series}"));
            }
            let inner = &series[open + 1..series.len() - 1];
            if inner.contains('\n') || inner.contains('\r') {
                return Err(format!("ラベル値に生の改行: {series}"));
            }
            if inner.ends_with(',') {
                return Err(format!("ラベルの末尾にカンマ: {series}"));
            }
            (&series[..open], inner)
        }
        None => {
            if series.contains('}') {
                return Err(format!("対応しない }}: {series}"));
            }
            (series, "")
        }
    };
    if !valid_metric_name(name) {
        return Err(format!("メトリクス名が不正: {name:?}"));
    }
    Ok((
        format!("{name}{{{label_text}}}"),
        name.to_string(),
        value,
    ))
}

/// 本文全体を format 0.0.4 として検証し、系列キー -> 値を返す。
///
/// - 末尾は改行 1 個、CR を含まない、空行・末尾空白が無い。
/// - `# HELP name text` / `# TYPE name type` の形で、名前は重複しない。
/// - TYPE / HELP は対応するサンプルより前に出る。
/// - 同じ `name{labels}` の系列が 2 回出ない。
fn validate_exposition(body: &str) -> BTreeMap<String, f64> {
    assert!(body.ends_with('\n'), "本文が改行で終わっていない");
    assert!(!body.ends_with("\n\n"), "本文が二重改行で終わっている");
    assert!(!body.contains('\r'), "CR が混ざっている");

    let mut helps: BTreeSet<String> = BTreeSet::new();
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut samples: BTreeMap<String, f64> = BTreeMap::new();

    for (idx, line) in body.lines().enumerate() {
        assert!(!line.is_empty(), "{idx} 行目が空行");
        assert_eq!(line.trim_end(), line, "{idx} 行目に末尾空白: {line:?}");
        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, help) = rest
                .split_once(' ')
                .unwrap_or_else(|| panic!("{idx} 行目: HELP に説明文が無い: {line}"));
            assert!(valid_metric_name(name), "{idx} 行目: HELP の名前が不正");
            assert!(!help.is_empty(), "{idx} 行目: HELP の説明が空");
            assert!(helps.insert(name.to_string()), "{idx} 行目: HELP {name} が重複");
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, ty) = rest
                .split_once(' ')
                .unwrap_or_else(|| panic!("{idx} 行目: TYPE に型が無い: {line}"));
            assert!(valid_metric_name(name), "{idx} 行目: TYPE の名前が不正");
            assert!(
                matches!(ty, "counter" | "gauge" | "histogram" | "summary" | "untyped"),
                "{idx} 行目: 未知の型 {ty}"
            );
            assert!(
                types.insert(name.to_string(), ty.to_string()).is_none(),
                "{idx} 行目: TYPE {name} が重複"
            );
            assert!(
                helps.contains(name),
                "{idx} 行目: TYPE {name} より HELP が後ろ / 欠落"
            );
            continue;
        }
        assert!(!line.starts_with('#'), "{idx} 行目: 未知のコメント行: {line}");

        let (key, name, value) = parse_sample(line)
            .unwrap_or_else(|e| panic!("{idx} 行目: サンプルが不正 ({e}): {line}"));
        // histogram の `_bucket` / `_sum` / `_count` はファミリ名へ寄せる。
        let family = ["_bucket", "_sum", "_count"]
            .iter()
            .find_map(|suffix| {
                name.strip_suffix(suffix)
                    .filter(|base| types.get(*base).map(String::as_str) == Some("histogram"))
            })
            .unwrap_or(name.as_str())
            .to_string();
        assert!(
            types.contains_key(&family),
            "{idx} 行目: {family} の TYPE が無い: {line}"
        );
        assert!(
            helps.contains(&family),
            "{idx} 行目: {family} の HELP が無い: {line}"
        );
        assert!(
            samples.insert(key.clone(), value).is_none(),
            "{idx} 行目: 系列が重複: {key}"
        );
    }
    samples
}

// ---------------------------------------------------------------------------
// 値の取り出し
// ---------------------------------------------------------------------------

fn value_or_zero(samples: &BTreeMap<String, f64>, name: &str, labels: &str) -> f64 {
    let key = if labels.is_empty() {
        format!("{name}{{}}")
    } else {
        format!("{name}{{{labels}}}")
    };
    samples.get(&key).copied().unwrap_or(0.0)
}

fn hit_rate_value(samples: &BTreeMap<String, f64>) -> f64 {
    value_or_zero(samples, "denno_prefix_cache_hit_rate", "")
}

// ---------------------------------------------------------------------------
// 1. exposition の骨格 (prefix cache / ratelimit ファミリ)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_and_ratelimit_families_have_valid_0_0_4_syntax() {
    let _g = SERIAL.lock().await;
    let app = app_with(0, None);
    let body = scrape(&app).await;
    let samples = validate_exposition(&body);

    let expected: &[(&str, &str)] = &[
        ("denno_prefix_cache_hits_total", "counter"),
        ("denno_prefix_cache_misses_total", "counter"),
        ("denno_prefix_cache_reused_tokens_total", "counter"),
        ("denno_prefix_cache_hit_rate", "gauge"),
        ("denno_ratelimit_rejections_total", "counter"),
    ];
    for (name, ty) in expected {
        assert!(
            body.contains(&format!("# TYPE {name} {ty}")),
            "{name} の TYPE {ty} が無い"
        );
        assert!(
            body.lines()
                .any(|l| l.starts_with(&format!("# HELP {name} "))),
            "{name} の HELP が無い"
        );
    }

    // ラベル付き counter は 0 でも系列を出す。
    assert!(
        samples.contains_key("denno_ratelimit_rejections_total{reason=\"token_bucket\"}"),
        "token_bucket の系列が 0 のとき消えている"
    );
    // 3 つの counter はラベル無しの単一系列。
    for name in [
        "denno_prefix_cache_hits_total",
        "denno_prefix_cache_misses_total",
        "denno_prefix_cache_reused_tokens_total",
    ] {
        assert!(
            samples.contains_key(&format!("{name}{{}}")),
            "{name} の系列が無い"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. prefix cache カウンタの単調増加
// ---------------------------------------------------------------------------

#[tokio::test]
async fn prefix_cache_counters_increase_monotonically_by_exact_deltas() {
    let _g = SERIAL.lock().await;
    let app = app_with(0, None);

    let base = validate_exposition(&scrape(&app).await);
    let hits0 = value_or_zero(&base, "denno_prefix_cache_hits_total", "");
    let misses0 = value_or_zero(&base, "denno_prefix_cache_misses_total", "");
    let reused0 = value_or_zero(&base, "denno_prefix_cache_reused_tokens_total", "");

    // 記録 -> スクレイプ を 4 回繰り返し、各時点の差分を固定する。
    // (reused, hits_delta, misses_delta, reused_delta)
    let steps: &[(usize, f64, f64, f64)] = &[
        (5, 1.0, 0.0, 5.0),
        (0, 0.0, 1.0, 0.0),
        (1, 1.0, 0.0, 1.0),
        (64, 1.0, 0.0, 64.0),
    ];

    let mut prev_hits = hits0;
    let mut prev_misses = misses0;
    let mut prev_reused = reused0;
    let mut exp_hits = hits0;
    let mut exp_misses = misses0;
    let mut exp_reused = reused0;

    for (i, (reused, d_hits, d_misses, d_reused)) in steps.iter().enumerate() {
        metrics::record_prefix_cache_reuse(*reused);
        exp_hits += d_hits;
        exp_misses += d_misses;
        exp_reused += d_reused;

        let s = validate_exposition(&scrape(&app).await);
        let hits = value_or_zero(&s, "denno_prefix_cache_hits_total", "");
        let misses = value_or_zero(&s, "denno_prefix_cache_misses_total", "");
        let reused_total = value_or_zero(&s, "denno_prefix_cache_reused_tokens_total", "");

        assert_eq!(hits, exp_hits, "step {i}: hits の絶対値が想定と違う");
        assert_eq!(misses, exp_misses, "step {i}: misses の絶対値が想定と違う");
        assert_eq!(
            reused_total, exp_reused,
            "step {i}: reused_tokens の絶対値が想定と違う"
        );

        // counter は決して減らない (単調性)。
        assert!(hits >= prev_hits, "step {i}: hits が減った");
        assert!(misses >= prev_misses, "step {i}: misses が減った");
        assert!(reused_total >= prev_reused, "step {i}: reused_tokens が減った");

        // hit_rate は同じスクレイプの累積値と一致する。
        let expected_rate = if hits + misses > 0.0 {
            hits / (hits + misses)
        } else {
            0.0
        };
        let got = hit_rate_value(&s);
        assert!(
            (got - expected_rate).abs() < 1e-6,
            "step {i}: hit_rate = {got}, 期待 {expected_rate}"
        );

        prev_hits = hits;
        prev_misses = misses;
        prev_reused = reused_total;
    }

    // 4 回の記録で hits +3 / misses +1 になっている (合計の整合)。
    assert_eq!(prev_hits - hits0, 3.0);
    assert_eq!(prev_misses - misses0, 1.0);
    assert_eq!(prev_reused - reused0, 70.0);
}

// ---------------------------------------------------------------------------
// 3. 429 拒否カウンタの単調増加
// ---------------------------------------------------------------------------

#[tokio::test]
async fn rate_limit_429_rejections_increase_monotonically_and_do_not_touch_cache() {
    let _g = SERIAL.lock().await;
    // rpm = 1 / burst = 1: 数 ms では回復しないので 2 本目以降は必ず 429。
    let app = app_with(1, Some(1));

    let base = validate_exposition(&scrape(&app).await);
    let rl0 = value_or_zero(
        &base,
        "denno_ratelimit_rejections_total",
        "reason=\"token_bucket\"",
    );
    let http0 = value_or_zero(
        &base,
        "denno_http_requests_total",
        "endpoint=\"/v1/messages\",status=\"429\"",
    );

    let body = json!({
        "model": "denno-test",
        "max_tokens": 2,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": "t0"}],
    });

    // 1 本目は通す (バケットのトークンを 1 消費する)。
    let first = post(&app, "/v1/messages", &body).await;
    assert!(
        first != StatusCode::TOO_MANY_REQUESTS,
        "1 本目がレート制限で落ちた: {first}"
    );

    // 通した 1 本は完走して prefix cache を 1 回記録し得る。429 の影響だけを
    // 見たいので、cache の基準値はこの後に取る。
    let settled = validate_exposition(&scrape(&app).await);
    let hits0 = value_or_zero(&settled, "denno_prefix_cache_hits_total", "");
    let misses0 = value_or_zero(&settled, "denno_prefix_cache_misses_total", "");
    let reused0 = value_or_zero(&settled, "denno_prefix_cache_reused_tokens_total", "");

    // 続く 3 本はすべて 429。各リクエストの後にスクレイプして単調性を見る。
    let mut prev_rl = rl0;
    let mut prev_http = http0;
    let mut rejected = 0.0f64;
    for i in 0..3 {
        let status = post(&app, "/v1/messages", &body).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{i} 本目が 429 でない");
        rejected += 1.0;

        let s = validate_exposition(&scrape(&app).await);
        let rl = value_or_zero(
            &s,
            "denno_ratelimit_rejections_total",
            "reason=\"token_bucket\"",
        );
        let http429 = value_or_zero(
            &s,
            "denno_http_requests_total",
            "endpoint=\"/v1/messages\",status=\"429\"",
        );
        assert_eq!(
            rl - rl0,
            rejected,
            "429 拒否カウンタの差分が拒否回数と違う (step {i})"
        );
        assert_eq!(
            http429 - http0,
            rejected,
            "http 429 カウンタの差分が拒否回数と違う (step {i})"
        );
        assert!(rl >= prev_rl, "step {i}: 拒否カウンタが減った");
        assert!(http429 >= prev_http, "step {i}: http 429 カウンタが減った");

        prev_rl = rl;
        prev_http = http429;
    }

    // 429 は prefix cache の計器を汚さない (拒否は完走でもミスでもない)。
    let after = validate_exposition(&scrape(&app).await);
    assert_eq!(
        value_or_zero(&after, "denno_prefix_cache_hits_total", ""),
        hits0,
        "429 で hits が動いた"
    );
    assert_eq!(
        value_or_zero(&after, "denno_prefix_cache_misses_total", ""),
        misses0,
        "429 で misses が動いた"
    );
    assert_eq!(
        value_or_zero(&after, "denno_prefix_cache_reused_tokens_total", ""),
        reused0,
        "429 で reused_tokens が動いた"
    );

    // 枯れていても /metrics は素通しで 200 (上の scrape がそれ)。
    let (status, _, _) = get(&app, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
}
