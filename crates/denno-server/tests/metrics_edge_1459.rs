//! Task 1459: `/metrics` の Prometheus text exposition format とラベル衛生の
//! エッジテスト。
//!
//! 対象 (`crates/denno-server/src/metrics.rs` と `lib.rs` の
//! `metrics_endpoint` を読んで確認した公開面):
//! - `GET /metrics` は `text/plain; version=0.0.4; charset=utf-8` を返す。
//! - 本文は format 0.0.4。ファミリごとに `# HELP` の直後に `# TYPE` が 1 回ずつ。
//! - サンプル値は有限の 10 進数 (NaN / Inf / 指数表記を出さない)。
//! - カウンタはリクエストをまたいで単調増加する。
//! - ラベルは `model` / `reason` / `endpoint` / `status` / `le` だけで、
//!   API キー・メール・私設 IP・`/home` パス・内部ホスト名を含まない。
//!
//! 計器はプロセス全体で共有されるため、このバイナリ内のテストは `SERIAL` で
//! 直列化し、期待値は絶対値ではなく差分で見る。
//!
//! 注: ラベル値を含む本文は「文字列 contains」ではなく自前パーサで読む。
//! エスケープ (生の改行・不正な `\x`) と重複系列を同時に弾くため。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use tower::ServiceExt;

/// 共有計器を触るテストを 1 本ずつ実行するためのロック。
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// `metrics::normalize_endpoint` がそのまま返す既知ルート (`metrics.rs` と 1:1)。
const KNOWN_ENDPOINTS: &[&str] = &[
    "/health",
    "/ready",
    "/metrics",
    "/v1/models",
    "/v1/messages",
    "/v1/messages/count_tokens",
    "/v1/chat/completions",
    "/v1/completions",
    "/api/tags",
    "/api/version",
    "/api/chat",
    "/api/generate",
    "/v1/telemetry/ws",
];

/// 許されるラベル名 (これ以外が出たら PII / 予期しない次元の混入とみなす)。
const ALLOWED_LABELS: &[&str] = &["model", "reason", "endpoint", "status", "le"];

// ---------------------------------------------------------------------------
// アプリ生成 (metrics_endpoint.rs / metrics_edge_1024.rs と同じ組み立て)
// ---------------------------------------------------------------------------

fn app_with(model_name: &str, api_key: Option<&str>) -> Router {
    let tok = dummy_tokenizer().unwrap();
    let engine = Engine::from_parts(Box::new(DummyModel), tok, "denno-test", None);
    let engine = Arc::new(Mutex::new(engine));
    let scheduler = Scheduler::spawn(engine.clone(), 8);
    let mut info = EngineInfo::from_engine(&engine.lock().unwrap());
    info.model_dir = "qwen-test-dir".to_string();
    let state = Arc::new(AppState {
        engine,
        info,
        api_key: api_key.map(str::to_string),
        scheduler,
        model_name: model_name.to_string(),
        model_id: "Qwen/Qwen2.5-0.5B-Instruct".to_string(),
        default_repetition_penalty: None,
        temperature_cap: None,
    });
    denno_server::router(state)
}

struct Resp {
    status: StatusCode,
    content_type: String,
    headers: Vec<(String, String)>,
    body: String,
}

async fn request(
    app: &Router,
    method: &str,
    uri: &str,
    headers_in: &[(&str, &str)],
    body: Option<&str>,
) -> Resp {
    let mut b = Request::builder().method(method).uri(uri);
    for (k, v) in headers_in {
        b = b.header(*k, *v);
    }
    let payload = match body {
        Some(text) => Body::from(text.to_string()),
        None => Body::empty(),
    };
    let req = b.body(payload).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let headers = resp
        .headers()
        .iter()
        .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
        .collect();
    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    Resp {
        status,
        content_type,
        headers,
        body: String::from_utf8_lossy(&bytes).to_string(),
    }
}

async fn get(app: &Router, uri: &str) -> Resp {
    request(app, "GET", uri, &[], None).await
}

async fn scrape(app: &Router) -> Resp {
    let r = get(app, "/metrics").await;
    assert_eq!(r.status, StatusCode::OK, "/metrics が 200 でない");
    r
}

// ---------------------------------------------------------------------------
// exposition format 0.0.4 のパーサ / 検証器
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    /// ラベル部分の生テキスト (ラベル無しは空)。重複判定に使う。
    label_text: String,
    value: f64,
    value_text: String,
}

/// メトリクス名: `[a-zA-Z_:][a-zA-Z0-9_:]*`
fn valid_metric_name(name: &str) -> bool {
    let mut it = name.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == ':' => {}
        _ => return false,
    }
    it.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':')
}

/// サンプル値の書式: 符号なし / `-` + 10 進数のみ。指数表記・`+`・空白を許さない。
fn valid_value_text(s: &str) -> bool {
    let b = s.as_bytes();
    let mut i = 0usize;
    if b.first() == Some(&b'-') {
        i = 1;
    }
    let int_start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == int_start {
        return false;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }
    i == b.len()
}

/// ラベル集合 `k="v",k2="v2"` を読み、値をアンエスケープして返す。
/// 許されるエスケープは `\\` / `\"` / `\n` のみ (format 0.0.4)。
fn parse_labels(s: &str) -> Result<Vec<(String, String)>, String> {
    let mut out: Vec<(String, String)> = Vec::new();
    if s.is_empty() {
        return Ok(out);
    }
    if s.ends_with(',') {
        return Err("末尾にカンマがある".to_string());
    }
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() {
        let start = i;
        while i < chars.len()
            && (chars[i].is_ascii_alphanumeric() || chars[i] == '_' || chars[i] == ':')
        {
            i += 1;
        }
        if i == start {
            return Err("ラベル名が不正".to_string());
        }
        let name: String = chars[start..i].iter().collect();
        if chars.get(i) != Some(&'=') {
            return Err(format!("ラベル {name} に = が無い"));
        }
        i += 1;
        if chars.get(i) != Some(&'"') {
            return Err(format!("ラベル {name} の値が引用されていない"));
        }
        i += 1;
        let mut val = String::new();
        loop {
            match chars.get(i) {
                None => return Err(format!("ラベル {name} の値が閉じていない")),
                Some('"') => {
                    i += 1;
                    break;
                }
                Some('\\') => {
                    i += 1;
                    match chars.get(i) {
                        Some('\\') => val.push('\\'),
                        Some('"') => val.push('"'),
                        Some('n') => val.push('\n'),
                        other => {
                            return Err(format!("ラベル {name} に不正なエスケープ: {other:?}"))
                        }
                    }
                    i += 1;
                }
                Some('\n') | Some('\r') => {
                    return Err(format!("ラベル {name} の値に生の改行がある"))
                }
                Some(c) => {
                    val.push(*c);
                    i += 1;
                }
            }
        }
        if out.iter().any(|(k, _)| *k == name) {
            return Err(format!("ラベル {name} が重複している"));
        }
        out.push((name, val));
        if i == chars.len() {
            break;
        }
        if chars[i] != ',' {
            return Err(format!("ラベル間の区切りが不正: {:?}", chars[i]));
        }
        i += 1;
    }
    Ok(out)
}

/// `name{labels} value` / `name value` を 1 行読む。
fn parse_sample(line: &str) -> Result<Sample, String> {
    if line.contains('\t') {
        return Err("タブが混ざっている".to_string());
    }
    let (series, value_text) = line
        .split_once(' ')
        .ok_or_else(|| "名前と値の区切りが無い".to_string())?;
    if value_text.is_empty() || value_text.contains(' ') {
        return Err(format!("値の書式が不正: {value_text:?}"));
    }
    if !valid_value_text(value_text) {
        return Err(format!("値が 10 進数でない: {value_text:?}"));
    }
    let value: f64 = value_text
        .parse()
        .map_err(|_| format!("値が数値でない: {value_text:?}"))?;
    if !value.is_finite() {
        return Err(format!("値が有限でない: {value_text:?}"));
    }

    let (name, label_text, labels) = match series.find('{') {
        Some(open) => {
            if !series.ends_with('}') {
                return Err(format!("ラベル集合が }} で閉じていない: {series}"));
            }
            let name = &series[..open];
            let inner = &series[open + 1..series.len() - 1];
            (name.to_string(), inner.to_string(), parse_labels(inner)?)
        }
        None => {
            if series.contains('}') {
                return Err(format!("対応しない }}: {series}"));
            }
            (series.to_string(), String::new(), Vec::new())
        }
    };
    if !valid_metric_name(&name) {
        return Err(format!("メトリクス名が不正: {name:?}"));
    }
    Ok(Sample {
        name,
        labels,
        label_text,
        value,
        value_text: value_text.to_string(),
    })
}

/// 本文全体を format 0.0.4 として検証した結果。
struct Doc {
    helps: BTreeMap<String, String>,
    types: BTreeMap<String, String>,
    samples: Vec<Sample>,
}

/// `_bucket` / `_sum` / `_count` を histogram ファミリへ寄せる。
fn family_of<'a>(name: &'a str, types: &'a BTreeMap<String, String>) -> &'a str {
    for suffix in ["_bucket", "_sum", "_count"] {
        if let Some(base) = name.strip_suffix(suffix) {
            if types.get(base).map(String::as_str) == Some("histogram") {
                return base;
            }
        }
    }
    name
}

/// 本文全体を検証する。文法違反は panic。
fn parse_doc(body: &str) -> Doc {
    assert!(body.ends_with('\n'), "本文が改行で終わっていない");
    assert!(!body.contains('\r'), "CR が混ざっている");
    assert!(!body.contains("\n\n"), "空行 (二重改行) が混ざっている");

    let mut helps: BTreeMap<String, String> = BTreeMap::new();
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut samples: Vec<Sample> = Vec::new();
    let mut seen_series: BTreeSet<String> = BTreeSet::new();

    let lines: Vec<&str> = body.lines().collect();
    let mut idx = 0usize;
    while idx < lines.len() {
        let line = lines[idx];
        assert!(!line.is_empty(), "{idx} 行目が空行");
        assert_eq!(line.trim_end(), line, "{idx} 行目に末尾空白: {line:?}");

        if let Some(rest) = line.strip_prefix("# HELP ") {
            let (name, help) = rest
                .split_once(' ')
                .unwrap_or_else(|| panic!("{idx} 行目: HELP に説明文が無い: {line}"));
            assert!(valid_metric_name(name), "{idx} 行目: HELP の名前が不正: {name}");
            assert!(!help.is_empty(), "{idx} 行目: HELP の説明が空");
            assert!(
                helps.insert(name.to_string(), help.to_string()).is_none(),
                "{idx} 行目: HELP {name} が重複"
            );
            // HELP の直後は同じファミリの TYPE (この実装の出力順)。
            let next = lines
                .get(idx + 1)
                .unwrap_or_else(|| panic!("{idx} 行目: HELP {name} の後に TYPE が無い"));
            assert!(
                next.starts_with(&format!("# TYPE {name} ")),
                "{idx} 行目: HELP {name} の直後が TYPE {name} でない: {next}"
            );
            idx += 1;
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            let (name, ty) = rest
                .split_once(' ')
                .unwrap_or_else(|| panic!("{idx} 行目: TYPE に型が無い: {line}"));
            assert!(valid_metric_name(name), "{idx} 行目: TYPE の名前が不正: {name}");
            assert!(
                matches!(ty, "counter" | "gauge" | "histogram" | "summary" | "untyped"),
                "{idx} 行目: 未知の型 {ty}"
            );
            assert!(
                types.insert(name.to_string(), ty.to_string()).is_none(),
                "{idx} 行目: TYPE {name} が重複"
            );
            assert!(
                helps.contains_key(name),
                "{idx} 行目: TYPE {name} に対応する HELP が無い"
            );
            idx += 1;
            continue;
        }
        assert!(!line.starts_with('#'), "{idx} 行目: 未知のコメント行: {line}");

        let sample = parse_sample(line)
            .unwrap_or_else(|e| panic!("{idx} 行目: サンプルが不正 ({e}): {line}"));
        let family = family_of(&sample.name, &types);
        assert!(
            types.contains_key(family),
            "{idx} 行目: {family} の TYPE がサンプルより後ろ / 欠落: {line}"
        );
        assert!(
            helps.contains_key(family),
            "{idx} 行目: {family} の HELP が無い: {line}"
        );
        let key = format!("{}{{{}}}", sample.name, sample.label_text);
        assert!(seen_series.insert(key.clone()), "系列が重複している: {key}");
        samples.push(sample);
        idx += 1;
    }

    Doc {
        helps,
        types,
        samples,
    }
}

/// `name` とラベル完全一致 (`labels` の全項目が存在) の最初の値を返す。
fn find_value(doc: &Doc, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    doc.samples
        .iter()
        .find(|s| {
            s.name == name
                && labels
                    .iter()
                    .all(|(k, v)| s.labels.iter().any(|(lk, lv)| lk == k && lv == v))
        })
        .map(|s| s.value)
}

fn value_or_zero(doc: &Doc, name: &str, labels: &[(&str, &str)]) -> f64 {
    find_value(doc, name, labels).unwrap_or(0.0)
}

/// 本文のカウンタ系列 `(name{labels}) -> value`。型が counter のものだけ。
fn counter_values(doc: &Doc) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for s in &doc.samples {
        let family = family_of(&s.name, &doc.types);
        if doc.types.get(family).map(String::as_str) == Some("counter") {
            out.insert(format!("{}{{{}}}", s.name, s.label_text), s.value);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 1. Content-Type と構文
// ---------------------------------------------------------------------------

#[tokio::test]
async fn content_type_and_exposition_syntax_are_valid() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);

    // /metrics 自身のスクレイプが系列を持つよう、先に 1 本だけ流す。
    let health = get(&app, "/health").await;
    assert_eq!(health.status, StatusCode::OK);

    let metrics = scrape(&app).await;
    assert_eq!(
        metrics.content_type, "text/plain; version=0.0.4; charset=utf-8",
        "Content-Type が Prometheus の text format と違う: {:?}",
        metrics.content_type
    );
    assert!(
        !metrics.content_type.contains("application/json"),
        "JSON の Content-Type が付いている"
    );
    // 応答ヘッダは可視 ASCII のみ (制御文字や非 ASCII を混ぜない)。
    for (name, v) in &metrics.headers {
        assert!(
            v.chars().all(|c| c.is_ascii_graphic() || c == ' '),
            "ヘッダ {name} に制御文字 / 非 ASCII: {v:?}"
        );
    }

    let doc = parse_doc(&metrics.body);
    assert!(!doc.samples.is_empty(), "サンプルが 1 行も無い");
    // 期待するファミリが揃っている。
    let expected: &[(&str, &str)] = &[
        ("denno_model_info", "gauge"),
        ("denno_server_ready", "gauge"),
        ("denno_server_draining", "gauge"),
        ("denno_requests_in_flight", "gauge"),
        ("denno_requests_queued", "gauge"),
        ("denno_requests_completed_total", "counter"),
        ("denno_requests_aborted_total", "counter"),
        ("denno_prompt_tokens_total", "counter"),
        ("denno_completion_tokens_total", "counter"),
        ("denno_kv_cache_read_tokens_total", "counter"),
        ("denno_kv_reuse_ratio", "gauge"),
        ("denno_prefix_cache_hits_total", "counter"),
        ("denno_prefix_cache_misses_total", "counter"),
        ("denno_prefix_cache_reused_tokens_total", "counter"),
        ("denno_prefix_cache_hit_rate", "gauge"),
        ("denno_ratelimit_rejections_total", "counter"),
        ("denno_ttft_seconds", "histogram"),
        ("denno_tpot_seconds", "histogram"),
        ("denno_http_requests_total", "counter"),
    ];
    for (name, ty) in expected {
        assert_eq!(
            doc.types.get(*name).map(String::as_str),
            Some(*ty),
            "{name} の TYPE が {ty} でない"
        );
        assert!(
            doc.helps.get(*name).map(|h| !h.is_empty()) == Some(true),
            "{name} の HELP が無い"
        );
    }
}

#[tokio::test]
async fn help_and_type_declare_every_family_exactly_once() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);
    let _ = get(&app, "/health").await;
    let body = scrape(&app).await.body;
    let doc = parse_doc(&body);

    // HELP と TYPE の集合は一致する (片方だけの宣言を残さない)。
    assert_eq!(
        doc.helps.keys().collect::<BTreeSet<_>>(),
        doc.types.keys().collect::<BTreeSet<_>>(),
        "HELP と TYPE のファミリ集合が不一致"
    );
    // TYPE を宣言したファミリはサンプルを持つ。例外は
    // `denno_http_requests_total` だけで、まだ 1 本も記録していないときは
    // HELP/TYPE のみ出す (このテストでは /health を流済みなので通常は持つ)。
    let with_samples: BTreeSet<String> = doc
        .samples
        .iter()
        .map(|s| family_of(&s.name, &doc.types).to_string())
        .collect();
    for name in doc.types.keys() {
        if name == "denno_http_requests_total" {
            continue;
        }
        assert!(
            with_samples.contains(name),
            "TYPE {name} はあるがサンプルが 1 行も無い"
        );
    }
    // ヒストグラムは _bucket / _sum / _count を必ず 3 種そろえる。
    for (name, ty) in &doc.types {
        if ty != "histogram" {
            continue;
        }
        let mut suffixes: BTreeSet<&str> = BTreeSet::new();
        for s in &doc.samples {
            if let Some(rest) = s.name.strip_prefix(name.as_str()) {
                suffixes.insert(rest);
            }
        }
        assert_eq!(
            suffixes,
            BTreeSet::from(["_bucket", "_sum", "_count"]),
            "{name} のヒストグラム系列が不正"
        );
    }
}

#[tokio::test]
async fn all_samples_are_finite_decimal_numbers() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);
    let _ = get(&app, "/health").await;
    let doc = parse_doc(&scrape(&app).await.body);

    for s in &doc.samples {
        assert!(
            s.value.is_finite(),
            "{} の値が有限でない: {}",
            s.name,
            s.value_text
        );
        assert!(
            s.value >= 0.0 || s.name.ends_with("_ratio") || s.name.ends_with("_rate"),
            "{} の値が負: {}",
            s.name,
            s.value_text
        );
        assert!(
            !s.value_text.contains('e') && !s.value_text.contains('E'),
            "{} の値が指数表記: {}",
            s.name,
            s.value_text
        );
        assert!(
            !s.value_text.contains("NaN")
                && !s.value_text.contains("Inf")
                && !s.value_text.contains("nan"),
            "{} の値が NaN / Inf: {}",
            s.name,
            s.value_text
        );
    }
}

#[tokio::test]
async fn histograms_are_cumulative_and_consistent() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);
    let _ = get(&app, "/health").await;
    let doc = parse_doc(&scrape(&app).await.body);

    for family in ["denno_ttft_seconds", "denno_tpot_seconds"] {
        assert_eq!(
            doc.types.get(family).map(String::as_str),
            Some("histogram")
        );
        let mut buckets: Vec<(String, f64)> = Vec::new();
        for s in &doc.samples {
            if s.name == format!("{family}_bucket") {
                let le = s
                    .labels
                    .iter()
                    .find(|(k, _)| k == "le")
                    .map(|(_, v)| v.clone())
                    .expect("bucket に le が無い");
                buckets.push((le, s.value));
            }
        }
        assert!(!buckets.is_empty(), "{family}_bucket が無い");
        assert_eq!(
            buckets.last().map(|(le, _)| le.as_str()),
            Some("+Inf"),
            "{family} の最後が +Inf でない"
        );
        let mut prev_le = f64::NEG_INFINITY;
        let mut prev = 0.0;
        for (i, (le, v)) in buckets.iter().enumerate() {
            assert!(*v >= prev, "{family} のバケットが累積でない (le={le})");
            prev = *v;
            if le == "+Inf" {
                assert_eq!(i, buckets.len() - 1, "+Inf が途中にある");
            } else {
                let lev: f64 = le
                    .parse()
                    .unwrap_or_else(|_| panic!("{family} の le が数値でない: {le:?}"));
                assert!(lev.is_finite(), "{family} の le が有限でない: {le:?}");
                assert!(lev > prev_le, "{family} の le が昇順でない: {le:?}");
                prev_le = lev;
            }
        }
        let count = find_value(&doc, &format!("{family}_count"), &[]).expect("_count が無い");
        let sum = find_value(&doc, &format!("{family}_sum"), &[]).expect("_sum が無い");
        assert_eq!(prev, count, "{family} の +Inf と _count が不一致");
        assert!(sum.is_finite() && sum >= 0.0, "{family} の _sum が不正");
    }
}

// ---------------------------------------------------------------------------
// 2. カウンタの単調性
// ---------------------------------------------------------------------------

#[tokio::test]
async fn counters_are_monotonic_across_requests() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);

    let d0 = parse_doc(&scrape(&app).await.body);
    let health_before = value_or_zero(
        &d0,
        "denno_http_requests_total",
        &[("endpoint", "/health"), ("status", "200")],
    );

    for _ in 0..5 {
        assert_eq!(get(&app, "/health").await.status, StatusCode::OK);
    }
    let d1 = parse_doc(&scrape(&app).await.body);

    for _ in 0..3 {
        assert_eq!(get(&app, "/health").await.status, StatusCode::OK);
    }
    let d2 = parse_doc(&scrape(&app).await.body);

    assert_eq!(
        value_or_zero(
            &d1,
            "denno_http_requests_total",
            &[("endpoint", "/health"), ("status", "200")]
        ) - health_before,
        5.0,
        "5 リクエストぶん増えていない"
    );
    assert_eq!(
        value_or_zero(
            &d2,
            "denno_http_requests_total",
            &[("endpoint", "/health"), ("status", "200")]
        ) - value_or_zero(
            &d1,
            "denno_http_requests_total",
            &[("endpoint", "/health"), ("status", "200")]
        ),
        3.0,
        "3 リクエストぶん増えていない"
    );

    let c0 = counter_values(&d0);
    let c1 = counter_values(&d1);
    let c2 = counter_values(&d2);
    assert!(!c0.is_empty(), "カウンタ系列が 1 本も無い");
    for (key, v0) in &c0 {
        let v1 = *c1.get(key).unwrap_or_else(|| panic!("{key} が 2 回目のスクレイプで消えた"));
        let v2 = *c2.get(key).unwrap_or_else(|| panic!("{key} が 3 回目のスクレイプで消えた"));
        assert!(*v0 <= v1, "{key} が減少: {v0} -> {v1}");
        assert!(v1 <= v2, "{key} が減少: {v1} -> {v2}");
    }
    // 既知の累積カウンタも単調。
    for name in [
        "denno_requests_completed_total",
        "denno_prompt_tokens_total",
        "denno_completion_tokens_total",
        "denno_kv_cache_read_tokens_total",
        "denno_prefix_cache_hits_total",
        "denno_prefix_cache_misses_total",
        "denno_prefix_cache_reused_tokens_total",
        "denno_ratelimit_rejections_total",
    ] {
        let a = value_or_zero(&d0, name, &[]);
        let b = value_or_zero(&d1, name, &[]);
        let c = value_or_zero(&d2, name, &[]);
        assert!(a <= b && b <= c, "{name} が単調でない: {a} / {b} / {c}");
    }
}

// ---------------------------------------------------------------------------
// 3. ラベル衛生 (秘密 / PII の非漏洩)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn labels_do_not_leak_secrets_or_pii() {
    let _g = SERIAL.lock().await;

    // 秘密 / PII は「実在の値」ではなく実行時に組み立てた合成マーカー。
    // ソース上にメール / IP / ホームパス / ホスト名のリテラルを残さない。
    let api_key = format!("{}-{}", "credential", "9f8e7d6c5b4a");
    let email = format!("{}{}{}", "probe", "@", "example.invalid");
    let private_ip = format!("{}.{}.{}.{}", 198, 51, 100, 7);
    let home_path = format!("/{}/{}/{}", "home", "probe", ".ssh");
    let hostname = format!("{}.{}", "probe-node", "invalid");

    let app = app_with("denno-probe", Some(&api_key));

    // 秘密 / PII をヘッダ・パス・ボディに載せて流す。
    let key_headers: &[(&str, &str)] = &[("x-api-key", api_key.as_str())];
    assert_eq!(get(&app, "/health").await.status, StatusCode::OK);
    assert_eq!(
        request(&app, "GET", "/v1/models", key_headers, None).await.status,
        StatusCode::OK
    );
    assert_eq!(
        request(&app, "GET", "/v1/models", &[], None).await.status,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        request(
            &app,
            "GET",
            "/v1/models",
            &[("authorization", "Bearer wrong-value")],
            None
        )
        .await
        .status,
        StatusCode::UNAUTHORIZED
    );
    // 未知パスに PII を入れる (endpoint ラベルは other に丸められる)。
    let unknown = format!("/no-such-route/{email}");
    assert_eq!(get(&app, &unknown).await.status, StatusCode::NOT_FOUND);
    // 生成リクエストのプロンプトに PII を入れる (トークン数だけが計器に載る)。
    let payload = format!(
        r#"{{"model":"denno-probe","messages":[{{"role":"user","content":"{email} {private_ip} {home_path} {hostname}"}}],"max_tokens":4,"temperature":0.0}}"#
    );
    let gen = request(
        &app,
        "POST",
        "/v1/messages",
        &[
            ("content-type", "application/json"),
            ("x-api-key", api_key.as_str()),
        ],
        Some(&payload),
    )
    .await;
    assert_eq!(gen.status, StatusCode::OK, "生成リクエストが通らない");

    let body = scrape(&app).await.body;
    for needle in [&api_key, &email, &private_ip, &home_path, &hostname] {
        assert!(
            !body.contains(needle.as_str()),
            "秘密 / PII が /metrics 本文に漏れている: {needle:?}"
        );
    }
    // 生のプロンプト断片・ヘッダ名も出ない。
    for needle in ["Bearer", "x-api-key", "authorization"] {
        assert!(
            !body.to_ascii_lowercase().contains(&needle.to_ascii_lowercase()),
            "リクエスト内容が本文に漏れている: {needle:?}"
        );
    }
    // 制御文字 (NUL / 垂直タブ等) を含まない。改行だけは区切りとして許す。
    for c in body.chars() {
        assert!(
            c == '\n' || !c.is_control(),
            "制御文字が混入: {:?}",
            c
        );
    }
}

#[tokio::test]
async fn label_names_and_values_are_bounded() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-probe", None);

    assert_eq!(get(&app, "/health").await.status, StatusCode::OK);
    assert_eq!(get(&app, "/v1/models").await.status, StatusCode::OK);
    // 未知パスは other に丸められる。
    for p in ["/nope", "/nope/2", "/wp-admin", "/health/extra"] {
        assert_eq!(get(&app, p).await.status, StatusCode::NOT_FOUND);
    }
    let body = scrape(&app).await.body;
    let doc = parse_doc(&body);

    let mut endpoints: BTreeSet<String> = BTreeSet::new();
    let mut statuses: BTreeSet<String> = BTreeSet::new();
    let mut reasons: BTreeSet<String> = BTreeSet::new();
    let mut models: BTreeSet<String> = BTreeSet::new();
    let mut le_values: BTreeSet<String> = BTreeSet::new();

    for s in &doc.samples {
        for (k, v) in &s.labels {
            assert!(
                ALLOWED_LABELS.contains(&k.as_str()),
                "未知のラベル名 {k:?} が出ている (PII / 高基数の疑い)"
            );
            assert!(!v.is_empty(), "ラベル {k} の値が空");
            assert!(
                v.chars().all(|c| c.is_ascii_graphic() || c == ' '),
                "ラベル {k} に非 ASCII / 制御文字: {v:?}"
            );
            assert!(!v.contains('@'), "ラベル {k} にメールらしき値: {v:?}");

            match k.as_str() {
                "endpoint" => {
                    assert!(
                        v == "other" || KNOWN_ENDPOINTS.contains(&v.as_str()),
                        "endpoint ラベル {v:?} が許可集合の外 (高基数)"
                    );
                    endpoints.insert(v.clone());
                }
                "status" => {
                    assert!(
                        v.len() == 3 && v.bytes().all(|b| b.is_ascii_digit()),
                        "status ラベルが 3 桁の数字でない: {v:?}"
                    );
                    let n: u16 = v.parse().expect("status が数値でない");
                    assert!((100..=599).contains(&n), "status が範囲外: {n}");
                    statuses.insert(v.clone());
                }
                "reason" => {
                    assert!(
                        matches!(
                            v.as_str(),
                            "timeout" | "disconnected" | "error" | "token_bucket"
                        ),
                        "reason ラベル {v:?} が既知の集合の外"
                    );
                    reasons.insert(v.clone());
                }
                "model" => {
                    assert_eq!(v, "denno-probe", "model ラベルが設定値と違う");
                    models.insert(v.clone());
                }
                "le" => {
                    assert!(
                        v == "+Inf" || v.parse::<f64>().map(|f| f.is_finite()).unwrap_or(false),
                        "le ラベル {v:?} が +Inf でも有限数でもない"
                    );
                    le_values.insert(v.clone());
                }
                other => panic!("許されないラベル {other:?}"),
            }
        }
    }

    assert_eq!(models.len(), 1, "model ラベルは 1 値だけ");
    assert!(
        endpoints.len() <= KNOWN_ENDPOINTS.len() + 1,
        "endpoint の基数が大きすぎる: {endpoints:?}"
    );
    assert!(statuses.len() <= 8, "status の基数が大きすぎる: {statuses:?}");
    // reason は aborted (3 種) と ratelimit (1 種) で計 4 値まで。
    assert!(reasons.len() <= 4, "reason の基数が大きすぎる: {reasons:?}");
    assert!(
        le_values.contains("+Inf") && le_values.len() <= 12 + 10 + 1,
        "le の値が想定と違う: {le_values:?}"
    );
    // 未知パスの生文字列が出ていない。
    for p in ["/nope", "/wp-admin", "/health/extra"] {
        assert!(!body.contains(p), "未知パス {p} の生文字列が漏れている");
    }
    // 系列数の上限 (基数の暴走が無い)。
    assert!(
        doc.samples.len() < 200,
        "系列数が多すぎる: {}",
        doc.samples.len()
    );
}
