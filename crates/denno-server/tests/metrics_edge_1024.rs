//! Task 1024: `/metrics` の Prometheus text exposition format 0.0.4 と
//! 並行アクセスのエッジテスト。
//!
//! 対象 (`crates/denno-server/src/metrics.rs` を全読して確認した公開面):
//! - `metrics::render(model, in_flight, queued, &stats, ready, draining)` — 本文生成
//! - `metrics::record_http(endpoint, status)` / `metrics::normalize_endpoint`
//! - `metrics::record_completion(&CompletionSample)` — ヒストグラム観測
//!
//! 契約として固定するもの:
//!   1. exposition format 0.0.4 の文法 (HELP/TYPE・メトリクス名・ラベル
//!      エスケープ・末尾改行・系列の重複なし)。
//!   2. ヒストグラムの `_bucket` が le 昇順かつ累積、`le="+Inf"` == `_count`、
//!      `_sum` は有限で非負。
//!   3. endpoint ラベルの正規化 (既知ルートはそのまま、未知パスは `other`)。
//!   4. 並行リクエストの下でも計器が panic せず、ゲージが負にならないこと、
//!      および `/metrics` が API キー無しで読めること。
//!
//! 計器はプロセス全体で共有される (このテストバイナリ内で 1 個)。テストは
//! `SERIAL` で直列化し、期待値は絶対値ではなく差分で見る。
//!
//! 注: ラベル値を含む本文を「文字列 contains」でなく自前パーサで読むのは、
//! エスケープの検証 (生の改行・不正な `\x` を許さない) を同時に行うため。

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use denno_engine::testutil::{dummy_tokenizer, DummyModel};
use denno_engine::Engine;
use denno_server::metrics::{self, CompletionSample};
use denno_server::{AppState, EngineInfo, Scheduler};
use http_body_util::BodyExt;
use tower::ServiceExt;

/// 共有計器を触るテストを 1 本ずつ実行するためのロック。
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// `metrics::normalize_endpoint` がそのまま返す既知ルート。
/// (`metrics.rs` の match と 1:1 で維持する。)
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

// ---------------------------------------------------------------------------
// アプリ生成 (metrics_endpoint.rs と同じ組み立て)
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

async fn get_req(app: &Router, uri: &str, headers: &[(&str, &str)]) -> (StatusCode, String, String) {
    let mut b = Request::builder().method("GET").uri(uri);
    for (k, v) in headers {
        b = b.header(*k, *v);
    }
    let req = b.body(Body::empty()).unwrap();
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

async fn get(app: &Router, uri: &str) -> (StatusCode, String, String) {
    get_req(app, uri, &[]).await
}

async fn scrape(app: &Router) -> String {
    let (status, ct, body) = get(app, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        ct.starts_with("text/plain; version=0.0.4"),
        "Content-Type = {ct}"
    );
    body
}

// ---------------------------------------------------------------------------
// exposition format 0.0.4 のパーサ / 検証器
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    /// ラベル部分の生テキスト (ラベル無しは空)。
    label_text: String,
    value: f64,
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
            return Err(format!("ラベル名が不正: {}", chars[start..].iter().collect::<String>()));
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
                            return Err(format!(
                                "ラベル {name} に不正なエスケープ: {:?}",
                                other
                            ))
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
    let (series, value_text) = line
        .split_once(' ')
        .ok_or_else(|| "名前と値の区切りが無い".to_string())?;
    if value_text.is_empty() || value_text.contains(' ') {
        return Err(format!("値の書式が不正: {value_text:?}"));
    }
    let value: f64 = value_text
        .parse()
        .map_err(|_| format!("値が数値でない: {value_text:?}"))?;

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
    })
}

/// 検証結果 (ファミリ -> 型、サンプル列)。
struct Exposition {
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

/// 本文全体を format 0.0.4 として検証する。違反があれば panic。
fn validate_exposition(body: &str) -> Exposition {
    assert!(body.ends_with('\n'), "本文が改行で終わっていない");
    assert!(!body.contains('\r'), "CR が混ざっている");

    let mut helps: BTreeMap<String, String> = BTreeMap::new();
    let mut types: BTreeMap<String, String> = BTreeMap::new();
    let mut samples: Vec<Sample> = Vec::new();
    let mut families_with_samples: BTreeSet<String> = BTreeSet::new();

    for (idx, line) in body.lines().enumerate() {
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
        assert!(
            sample.value.is_finite(),
            "{idx} 行目: 値が有限でない: {line}"
        );
        families_with_samples.insert(family.to_string());
        samples.push(sample);
    }

    // TYPE を宣言したファミリは必ずサンプルを持つ (空宣言を残さない)。
    // 例外は `denno_http_requests_total`: まだ 1 リクエストも記録していない
    // 時点では HELP/TYPE だけが出る (ラベル付きカウンタの初期状態)。
    for name in types.keys() {
        if name == "denno_http_requests_total" {
            continue;
        }
        assert!(
            families_with_samples.contains(name),
            "TYPE {name} はあるがサンプルが 1 行も無い"
        );
    }
    // ヒストグラムは _bucket / _sum / _count の 3 種だけを出す。
    for (name, ty) in &types {
        if ty == "histogram" {
            let mut suffixes: BTreeSet<&str> = BTreeSet::new();
            for s in &samples {
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
    // 同じ名前 + 同じラベル集合の系列は 1 本だけ。
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for s in &samples {
        let key = format!("{}{{{}}}", s.name, s.label_text);
        assert!(seen.insert(key.clone()), "系列が重複している: {key}");
    }

    Exposition {
        helps,
        types,
        samples,
    }
}

// ---------------------------------------------------------------------------
// 値の取り出し
// ---------------------------------------------------------------------------

/// `name` とラベル完全一致 (`labels` の全項目が存在) の最初の値を返す。
fn find_value(text: &str, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
    text.lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| parse_sample(l).ok())
        .find(|s| {
            s.name == name
                && labels
                    .iter()
                    .all(|(k, v)| s.labels.iter().any(|(lk, lv)| lk == k && lv == v))
        })
        .map(|s| s.value)
}

fn value_or_zero(text: &str, name: &str, labels: &[(&str, &str)]) -> f64 {
    find_value(text, name, labels).unwrap_or(0.0)
}

/// `family_bucket` の (le 文字列, 値) を出力順に返す。
fn bucket_values(text: &str, family: &str) -> Vec<(String, f64)> {
    let prefix = format!("{family}_bucket{{");
    let mut out = Vec::new();
    for line in text.lines() {
        if line.starts_with(&prefix) {
            let s = parse_sample(line).expect("bucket 行がパースできない");
            let le = s
                .labels
                .iter()
                .find(|(k, _)| k == "le")
                .map(|(_, v)| v.clone())
                .expect("bucket 行に le が無い");
            out.push((le, s.value));
        }
    }
    out
}

/// ヒストグラム 1 ファミリの不変条件を検証する。
fn assert_histogram_family(text: &str, family: &str, expected_le: usize) {
    assert!(
        text.contains(&format!("# TYPE {family} histogram")),
        "{family} の TYPE histogram が無い"
    );
    let buckets = bucket_values(text, family);
    assert_eq!(
        buckets.len(),
        expected_le,
        "{family} のバケット数が想定と違う: {:?}",
        buckets.iter().map(|(le, _)| le.as_str()).collect::<Vec<_>>()
    );
    assert_eq!(
        buckets.last().map(|(le, _)| le.as_str()),
        Some("+Inf"),
        "{family} の最後のバケットが +Inf でない"
    );

    let mut prev_le = f64::NEG_INFINITY;
    let mut prev_count = 0.0f64;
    for (i, (le, v)) in buckets.iter().enumerate() {
        assert!(v.is_finite() && *v >= 0.0, "{family} のバケット値が不正: {v}");
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
        assert!(
            *v >= prev_count,
            "{family} のバケットが累積でない (le={le}: {v} < {prev_count})"
        );
        prev_count = *v;
    }

    let count = find_value(text, &format!("{family}_count"), &[]).expect("_count が無い");
    let sum = find_value(text, &format!("{family}_sum"), &[]).expect("_sum が無い");
    assert_eq!(prev_count, count, "{family} の +Inf と _count が不一致");
    assert!(count.is_finite() && count >= 0.0, "{family} の _count が不正");
    assert!(sum.is_finite(), "{family} の _sum が有限でない: {sum}");
    assert!(sum >= 0.0, "{family} の _sum が負: {sum}");
}

fn assert_all_values_sane(text: &str) {
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        let s = parse_sample(line).unwrap_or_else(|e| panic!("サンプルが不正 ({e}): {line}"));
        assert!(
            s.value.is_finite() && s.value >= 0.0,
            "{} の値が有限かつ非負でない: {}",
            s.name,
            s.value
        );
    }
}

// ---------------------------------------------------------------------------
// 1. exposition format
// ---------------------------------------------------------------------------

#[tokio::test]
async fn exposition_format_0_0_4_is_valid() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);
    let body = scrape(&app).await;
    let ex = validate_exposition(&body);

    let expected: &[(&str, &str)] = &[
        ("denno_model_info", "gauge"),
        ("denno_requests_in_flight", "gauge"),
        ("denno_requests_queued", "gauge"),
        ("denno_requests_completed_total", "counter"),
        ("denno_requests_aborted_total", "counter"),
        ("denno_prompt_tokens_total", "counter"),
        ("denno_completion_tokens_total", "counter"),
        ("denno_kv_cache_read_tokens_total", "counter"),
        ("denno_kv_reuse_ratio", "gauge"),
        ("denno_http_requests_total", "counter"),
        ("denno_ttft_seconds", "histogram"),
        ("denno_tpot_seconds", "histogram"),
    ];
    for (name, ty) in expected {
        assert_eq!(
            ex.types.get(*name).map(String::as_str),
            Some(*ty),
            "{name} の TYPE が {ty} でない"
        );
        assert!(
            ex.helps.get(*name).map(|h| !h.is_empty()) == Some(true),
            "{name} の HELP が無い"
        );
    }

    // カウンタ / ゲージは負にならない (ゲージもこの実装では非負)。
    assert_all_values_sane(&body);
    // 末尾は改行 1 個で終わる (二重改行を残さない)。
    assert!(!body.ends_with("\n\n"));
    // ラベル無し系列は `name value` の 1 空白区切り。
    assert!(body.lines().any(|l| l == "denno_requests_in_flight 0"));
}

// ---------------------------------------------------------------------------
// 2. ヒストグラム
// ---------------------------------------------------------------------------

#[tokio::test]
async fn histogram_buckets_are_monotonic_and_consistent() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);

    // バケット境界と件数を固定する (metrics.rs の TTFT_LE / TPOT_LE)。
    const TTFT_LE: usize = 12 + 1;
    const TPOT_LE: usize = 10 + 1;

    let before = scrape(&app).await;
    assert_histogram_family(&before, "denno_ttft_seconds", TTFT_LE);
    assert_histogram_family(&before, "denno_tpot_seconds", TPOT_LE);

    let ttft_count_before = value_or_zero(&before, "denno_ttft_seconds_count", &[]);
    let tpot_count_before = value_or_zero(&before, "denno_tpot_seconds_count", &[]);
    let ttft_sum_before = value_or_zero(&before, "denno_ttft_seconds_sum", &[]);
    let tpot_sum_before = value_or_zero(&before, "denno_tpot_seconds_sum", &[]);
    let ttft_buckets_before = bucket_values(&before, "denno_ttft_seconds");
    let tpot_buckets_before = bucket_values(&before, "denno_tpot_seconds");

    // 有限値のみを 3 件観測する。TTFT: 0.001 / 0.2 / 100 (最後は 60 超)。
    for (ttft, tpot) in [(0.001, 0.001), (0.2, 1.5), (100.0, 3.0)] {
        metrics::record_completion(&CompletionSample {
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            ttft_secs: Some(ttft),
            tpot_secs: Some(tpot),
        });
    }

    let after = scrape(&app).await;
    assert_histogram_family(&after, "denno_ttft_seconds", TTFT_LE);
    assert_histogram_family(&after, "denno_tpot_seconds", TPOT_LE);

    assert_eq!(
        value_or_zero(&after, "denno_ttft_seconds_count", &[]) - ttft_count_before,
        3.0
    );
    assert_eq!(
        value_or_zero(&after, "denno_tpot_seconds_count", &[]) - tpot_count_before,
        3.0
    );

    let ttft_buckets_after = bucket_values(&after, "denno_ttft_seconds");
    let delta = |le: &str, before: &[(String, f64)], after: &[(String, f64)]| -> f64 {
        let b = before.iter().find(|(l, _)| l == le).map(|(_, v)| *v).unwrap_or(0.0);
        let a = after.iter().find(|(l, _)| l == le).map(|(_, v)| *v).unwrap_or(0.0);
        a - b
    };
    // 0.001 は le=0.01 に、0.2 は le=0.25 に、100 は +Inf にだけ入る (累積)。
    assert_eq!(delta("0.01", &ttft_buckets_before, &ttft_buckets_after), 1.0);
    assert_eq!(delta("0.25", &ttft_buckets_before, &ttft_buckets_after), 2.0);
    assert_eq!(delta("60", &ttft_buckets_before, &ttft_buckets_after), 2.0);
    assert_eq!(delta("+Inf", &ttft_buckets_before, &ttft_buckets_after), 3.0);

    let tpot_buckets_after = bucket_values(&after, "denno_tpot_seconds");
    assert_eq!(delta("0.002", &tpot_buckets_before, &tpot_buckets_after), 1.0);
    assert_eq!(delta("2", &tpot_buckets_before, &tpot_buckets_after), 2.0);
    assert_eq!(delta("+Inf", &tpot_buckets_before, &tpot_buckets_after), 3.0);

    // _sum は観測値の和 (fmt_f は 6 桁丸めなので許容誤差で見る)。
    let ttft_sum_after = value_or_zero(&after, "denno_ttft_seconds_sum", &[]);
    let tpot_sum_after = value_or_zero(&after, "denno_tpot_seconds_sum", &[]);
    assert!(
        (ttft_sum_after - ttft_sum_before - 100.201).abs() < 1e-3,
        "ttft _sum の差分が想定と違う: {}",
        ttft_sum_after - ttft_sum_before
    );
    assert!(
        (tpot_sum_after - tpot_sum_before - 4.501).abs() < 1e-3,
        "tpot _sum の差分が想定と違う: {}",
        tpot_sum_after - tpot_sum_before
    );

    // 壊れた観測 (NaN / +Inf / 負値) を積んでも不変条件が崩れない。
    for ttft in [f64::NAN, f64::INFINITY, -1.0] {
        metrics::record_completion(&CompletionSample {
            input_tokens: 1,
            output_tokens: 1,
            cached_tokens: 0,
            ttft_secs: Some(ttft),
            tpot_secs: Some(f64::NAN),
        });
    }
    let broken = scrape(&app).await;
    assert_histogram_family(&broken, "denno_ttft_seconds", TTFT_LE);
    assert_histogram_family(&broken, "denno_tpot_seconds", TPOT_LE);
    assert!(
        value_or_zero(&broken, "denno_ttft_seconds_sum", &[]) >= ttft_sum_after,
        "_sum が減少した"
    );
    assert!(
        value_or_zero(&broken, "denno_ttft_seconds_count", &[]) >= ttft_count_before + 6.0,
        "壊れた観測が数えられていない"
    );
    assert_all_values_sane(&broken);
}

// ---------------------------------------------------------------------------
// 3. endpoint ラベルの正規化
// ---------------------------------------------------------------------------

#[test]
fn endpoint_normalization_maps_unknown_paths_to_other() {
    for known in KNOWN_ENDPOINTS {
        assert_eq!(
            metrics::normalize_endpoint(known),
            *known,
            "既知ルート {known} が丸められた"
        );
    }
    for unknown in [
        "",
        "/",
        "/v1",
        "/v1/",
        "/health/",
        "/v1/messages/",
        "/v1/Messages",
        "/v1/messages/count_tokens/x",
        "/metrics/",
        "/api",
        "/no-such-route/abc/def",
        "/v1/models/extra",
        "//health",
        "/health%2Fready",
    ] {
        assert_eq!(
            metrics::normalize_endpoint(unknown),
            "other",
            "未知パス {unknown:?} が other に丸められていない"
        );
    }
    // 返り値は必ず固定の集合に属する (基数の上限)。
    for probe in KNOWN_ENDPOINTS
        .iter()
        .copied()
        .chain(["other", "/whatever", "/v1/messages?x=1"].into_iter())
    {
        let ep = metrics::normalize_endpoint(probe);
        assert!(
            ep == "other" || KNOWN_ENDPOINTS.contains(&ep),
            "正規化結果 {ep:?} が許可集合の外"
        );
    }
}

#[tokio::test]
async fn http_labels_are_normalized_and_bounded() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);
    let before = scrape(&app).await;
    let other_before = value_or_zero(&before, "denno_http_requests_total", &[("endpoint", "other")]);
    let health_before = value_or_zero(
        &before,
        "denno_http_requests_total",
        &[("endpoint", "/health"), ("status", "200")],
    );
    let messages_405_before = value_or_zero(
        &before,
        "denno_http_requests_total",
        &[("endpoint", "/v1/messages"), ("status", "405")],
    );

    // 未知パスは 404 で、ラベルは other に集約される。
    let unknown = [
        "/definitely-not-a-route",
        "/definitely-not-a-route/2",
        "/wp-admin",
        "/health/extra",
        "/v1/models/a/b/c",
    ];
    for path in unknown {
        let (status, _, _) = get(&app, path).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{path} が 404 でない");
    }
    // 既知ルート: クエリ付きでもパスだけで正規化される。
    let (status, _, _) = get(&app, "/health?probe=1").await;
    assert_eq!(status, StatusCode::OK);
    // メソッド不許可でも既知ルート名で数える。
    let (status, _, _) = get(&app, "/v1/messages").await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);

    let after = scrape(&app).await;
    let other_after = value_or_zero(&after, "denno_http_requests_total", &[("endpoint", "other")]);
    assert_eq!(
        other_after - other_before,
        unknown.len() as f64,
        "未知パスの other 集約数が想定と違う"
    );
    let health_after = value_or_zero(
        &after,
        "denno_http_requests_total",
        &[("endpoint", "/health"), ("status", "200")],
    );
    assert_eq!(health_after - health_before, 1.0, "/health のカウンタ");
    let messages_405_after = value_or_zero(
        &after,
        "denno_http_requests_total",
        &[("endpoint", "/v1/messages"), ("status", "405")],
    );
    assert_eq!(
        messages_405_after - messages_405_before,
        1.0,
        "405 が既知ルート名で数えられていない"
    );

    // ラベル値の集合は既知ルート + other に収まる。未知パスの生文字列は出ない。
    let mut endpoints: BTreeSet<String> = BTreeSet::new();
    for line in after.lines().filter(|l| !l.starts_with('#')) {
        if let Ok(s) = parse_sample(line) {
            if s.name == "denno_http_requests_total" {
                let ep = s
                    .labels
                    .iter()
                    .find(|(k, _)| k == "endpoint")
                    .map(|(_, v)| v.clone())
                    .expect("endpoint ラベルが無い");
                endpoints.insert(ep);
            }
        }
    }
    for ep in &endpoints {
        assert!(
            ep == "other" || KNOWN_ENDPOINTS.contains(&ep.as_str()),
            "endpoint ラベル {ep:?} が許可集合の外 (基数が増える)"
        );
    }
    assert!(
        endpoints.len() <= KNOWN_ENDPOINTS.len() + 1,
        "endpoint ラベルの基数が大きすぎる: {endpoints:?}"
    );
    for path in unknown {
        assert!(
            !after.contains(path),
            "未知パスの生文字列 {path} が本文に漏れている"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. 並行アクセス
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_requests_stay_consistent() {
    let _g = SERIAL.lock().await;
    let app = app_with("denno-test", None);
    let before = scrape(&app).await;
    let health_before = value_or_zero(
        &before,
        "denno_http_requests_total",
        &[("endpoint", "/health"), ("status", "200")],
    );
    let metrics_before = value_or_zero(
        &before,
        "denno_http_requests_total",
        &[("endpoint", "/metrics"), ("status", "200")],
    );
    let completed_before = value_or_zero(&before, "denno_requests_completed_total", &[]);
    let aborted_before = value_or_zero(&before, "denno_requests_aborted_total", &[]);

    // 120 リクエストを 24 タスクに分けて並行に投げる。
    let mut workers = Vec::new();
    for _ in 0..24 {
        let app = app.clone();
        workers.push(tokio::spawn(async move {
            let mut ok = 0usize;
            for _ in 0..5 {
                let (status, _, _) = get(&app, "/health").await;
                assert_eq!(status, StatusCode::OK);
                ok += 1;
            }
            ok
        }));
    }
    // 同時に /metrics をスクレイプし、毎回パースできることを確かめる
    // (描画中に他のリクエストがカウンタを更新していても本文が壊れない)。
    let mut scrapers = Vec::new();
    for _ in 0..8 {
        let app = app.clone();
        scrapers.push(tokio::spawn(async move {
            let (status, ct, body) = get(&app, "/metrics").await;
            assert_eq!(status, StatusCode::OK);
            assert!(ct.starts_with("text/plain; version=0.0.4"));
            validate_exposition(&body);
            let in_flight = find_value(&body, "denno_requests_in_flight", &[]).unwrap();
            let queued = find_value(&body, "denno_requests_queued", &[]).unwrap();
            assert!(in_flight >= 0.0 && in_flight.is_finite());
            assert!(queued >= 0.0 && queued.is_finite());
        }));
    }

    let mut total_ok = 0usize;
    for h in workers {
        total_ok += h.await.expect("worker が panic した");
    }
    for h in scrapers {
        h.await.expect("スクレイパが panic した");
    }
    assert_eq!(total_ok, 120);

    let after = scrape(&app).await;
    validate_exposition(&after);
    assert_all_values_sane(&after);

    let health_after = value_or_zero(
        &after,
        "denno_http_requests_total",
        &[("endpoint", "/health"), ("status", "200")],
    );
    assert_eq!(
        health_after - health_before,
        120.0,
        "並行 /health の計数が合わない"
    );
    let metrics_after = value_or_zero(
        &after,
        "denno_http_requests_total",
        &[("endpoint", "/metrics"), ("status", "200")],
    );
    assert!(
        metrics_after - metrics_before >= 9.0,
        "並行スクレイプの計数が合わない: {}",
        metrics_after - metrics_before
    );

    // 生成は走っていないのでゲージは 0 に戻り、負にはならない。
    assert_eq!(find_value(&after, "denno_requests_in_flight", &[]), Some(0.0));
    assert_eq!(find_value(&after, "denno_requests_queued", &[]), Some(0.0));
    assert!(
        value_or_zero(&after, "denno_requests_completed_total", &[]) >= completed_before,
        "完了カウンタが減った"
    );
    assert!(
        value_or_zero(&after, "denno_requests_aborted_total", &[]) >= aborted_before,
        "打ち切りカウンタが減った"
    );
    assert!(
        value_or_zero(&after, "denno_kv_reuse_ratio", &[]) >= 0.0,
        "kv_reuse_ratio が負"
    );
}

// ---------------------------------------------------------------------------
// 5. 認証
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metrics_is_readable_without_api_key() {
    let _g = SERIAL.lock().await;
    const KEY: &str = "test-secret-key";
    let app = app_with("denno-test", Some(KEY));

    // キー無しでもスクレイプできる。
    let (status, ct, body) = get(&app, "/metrics").await;
    assert_eq!(status, StatusCode::OK);
    assert!(ct.starts_with("text/plain; version=0.0.4"));
    validate_exposition(&body);

    // ただし保護されたエンドポイントはキーを要求する (認証が有効なことの対照)。
    let (status, _, _) = get(&app, "/v1/models").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _, _) = get_req(&app, "/v1/models", &[("x-api-key", KEY)]).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = get_req(
        &app,
        "/v1/models",
        &[("authorization", "Bearer test-secret-key")],
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, _) = get_req(&app, "/v1/models", &[("x-api-key", "wrong")]).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

// ---------------------------------------------------------------------------
// 6. ラベルエスケープ
// ---------------------------------------------------------------------------

#[tokio::test]
async fn label_values_are_escaped_in_output() {
    let _g = SERIAL.lock().await;
    const MODEL: &str = "weird\"model\\name\nnewline";
    let app = app_with(MODEL, None);
    let body = scrape(&app).await;
    let ex = validate_exposition(&body);

    let info = ex
        .samples
        .iter()
        .find(|s| s.name == "denno_model_info")
        .expect("denno_model_info が無い");
    assert_eq!(info.labels.len(), 1);
    assert_eq!(info.labels[0].0, "model");
    // パーサで戻した値が元の文字列と一致する (=\", \\, \n が正しく往復する)。
    assert_eq!(info.labels[0].1, MODEL);
    assert_eq!(info.value, 1.0);

    let line = body
        .lines()
        .find(|l| l.starts_with("denno_model_info{"))
        .expect("生の行が無い");
    assert!(line.contains("\\\""), "二重引用符がエスケープされていない");
    assert!(line.contains("\\\\"), "バックスラッシュがエスケープされていない");
    assert!(line.contains("\\n"), "改行が \\n になっていない");
}
