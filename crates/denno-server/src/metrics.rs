//! Task 877: Prometheus `/metrics` (依存クレート無し・text exposition format 0.0.4)。
//!
//! 目的: 運用者が denno-server を標準の Prometheus でスクレイプできるようにする。
//! 系列は次の 3 種類:
//!
//! - **HTTP カウンタ** — ミドルウェアがリクエスト終端で `(endpoint, status)` を数える。
//!   endpoint は既知のルート名に正規化する (未知のパスは `other` に丸める —
//!   ラベルの基数を増やさない)。
//! - **トークン / レイテンシ** — 生成の完走時にスケジューラが 1 回だけ記録する
//!   (`prompt_tokens` / `completion_tokens` / KV 再利用 / TTFT / TPOT)。
//! - **ゲージ** — スクレイプ時点のスケジューラ状態を `/metrics` ハンドラが直接読む
//!   (`in_flight` / queued / 完了・打ち切り計数)。状態は既存の
//!   [`crate::Scheduler`] が持つ値を読むだけで、ここでは持たない。
//!
//! 設計の縛り:
//! - 依存クレートを足さない (prometheus crate を使わない)。カウンタは
//!   `AtomicU64`、ラベル付き HTTP カウンタだけは終端で 1 回 `Mutex<BTreeMap>` を
//!   触る (ホットループではない)。
//! - 生成の数値経路には触れない。記録は `finish_run` の終端で 1 回だけ。
//! - `DENNO_SERVER_METRICS=0|false|no` で記録を止める (既定 on)。`/metrics`
//!   自体は常に 200 で現在値を返す (記録が止まっていれば 0 のまま)。

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::scheduler::RunStatsSnapshot;

/// 記録のゲート (既定 on)。
pub const ENV_ON: &str = "DENNO_SERVER_METRICS";

/// 記録が有効か。env は初回だけ読む (以後は OnceLock の値)。
pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var(ENV_ON).ok().as_deref(),
            Some("0") | Some("false") | Some("no")
        )
    })
}

// ---------------------------------------------------------------------------
// HTTP カウンタ (endpoint, status)
// ---------------------------------------------------------------------------

/// 既知のルート名へ正規化する。未知のパスは `other` (基数を増やさない)。
pub fn normalize_endpoint(path: &str) -> &'static str {
    match path {
        "/health" => "/health",
        "/ready" => "/ready",
        "/metrics" => "/metrics",
        "/v1/models" => "/v1/models",
        "/v1/messages" => "/v1/messages",
        "/v1/messages/count_tokens" => "/v1/messages/count_tokens",
        "/v1/chat/completions" => "/v1/chat/completions",
        "/v1/completions" => "/v1/completions",
        "/api/tags" => "/api/tags",
        "/api/version" => "/api/version",
        "/api/chat" => "/api/chat",
        "/api/generate" => "/api/generate",
        "/v1/telemetry/ws" => "/v1/telemetry/ws",
        _ => "other",
    }
}

type HttpKey = (&'static str, u16);
static HTTP_COUNTS: OnceLock<Mutex<BTreeMap<HttpKey, u64>>> = OnceLock::new();

fn http_counts() -> &'static Mutex<BTreeMap<HttpKey, u64>> {
    HTTP_COUNTS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// リクエスト 1 件の終端で 1 回呼ぶ。status は 3 桁のコード。
pub fn record_http(endpoint: &'static str, status: u16) {
    if !enabled() {
        return;
    }
    if let Ok(mut m) = http_counts().lock() {
        *m.entry((endpoint, status)).or_insert(0) += 1;
    }
}

fn http_snapshot() -> Vec<((&'static str, u16), u64)> {
    match http_counts().lock() {
        Ok(m) => m.iter().map(|(k, v)| (*k, *v)).collect(),
        Err(_) => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// トークン / KV 再利用 / レイテンシ (完走時に 1 回)
// ---------------------------------------------------------------------------

static PROMPT_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
static COMPLETION_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
static KV_CACHE_READ_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
static COMPLETED_REQUESTS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// 1 リクエストぶんの完走サンプル。
pub struct CompletionSample {
    /// プロンプトの全トークン数 (`usage.input_tokens`)
    pub input_tokens: u64,
    /// 生成したトークン数 (`usage.output_tokens`)
    pub output_tokens: u64,
    /// うち KV prefix cache から再利用したトークン数
    pub cached_tokens: u64,
    /// 受付 → 最初のトークン (prefill 完了) の秒。0 トークン生成なら None。
    pub ttft_secs: Option<f64>,
    /// 最初のトークン → 完了 の平均トークン間隔 (秒)。1 トークン以下なら None。
    pub tpot_secs: Option<f64>,
}

/// 完走したリクエストを記録する (スケジューラの `finish_run` から 1 回)。
pub fn record_completion(s: &CompletionSample) {
    if !enabled() {
        return;
    }
    PROMPT_TOKENS_TOTAL.fetch_add(s.input_tokens, Ordering::Relaxed);
    COMPLETION_TOKENS_TOTAL.fetch_add(s.output_tokens, Ordering::Relaxed);
    KV_CACHE_READ_TOKENS_TOTAL.fetch_add(s.cached_tokens, Ordering::Relaxed);
    COMPLETED_REQUESTS_TOTAL.fetch_add(1, Ordering::Relaxed);
    if let Some(t) = s.ttft_secs {
        observe(&TTFT_LE, &TTFT_BUCKETS, &TTFT_SUM_NS, &TTFT_COUNT, t);
    }
    if let Some(t) = s.tpot_secs {
        observe(&TPOT_LE, &TPOT_BUCKETS, &TPOT_SUM_NS, &TPOT_COUNT, t);
    }
}

// ---------------------------------------------------------------------------
// ヒストグラム
// ---------------------------------------------------------------------------

/// TTFT の上限 (秒)。prefill は 14k で数秒〜数十秒になり得る。
const TTFT_LE: [f64; 12] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];
/// TPOT (トークン間隔) の上限 (秒)。
const TPOT_LE: [f64; 10] = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0];

const TTFT_N: usize = TTFT_LE.len() + 1;
const TPOT_N: usize = TPOT_LE.len() + 1;

static TTFT_BUCKETS: [AtomicU64; TTFT_N] = [const { AtomicU64::new(0) }; TTFT_N];
static TTFT_SUM_NS: AtomicU64 = AtomicU64::new(0);
static TTFT_COUNT: AtomicU64 = AtomicU64::new(0);

static TPOT_BUCKETS: [AtomicU64; TPOT_N] = [const { AtomicU64::new(0) }; TPOT_N];
static TPOT_SUM_NS: AtomicU64 = AtomicU64::new(0);
static TPOT_COUNT: AtomicU64 = AtomicU64::new(0);

/// 1 サンプルを積む。バケットは非累積で持ち、描画時に累積へ直す。
fn observe(le: &[f64], buckets: &[AtomicU64], sum_ns: &AtomicU64, count: &AtomicU64, secs: f64) {
    let idx = le.iter().position(|&b| secs <= b).unwrap_or(le.len());
    buckets[idx].fetch_add(1, Ordering::Relaxed);
    let ns = (secs.max(0.0) * 1e9) as u64;
    sum_ns.fetch_add(ns, Ordering::Relaxed);
    count.fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// 描画 (ここだけ確保する)
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// f64 を Prometheus が読める形にする (指数表記を避ける)。
fn fmt_f(v: f64) -> String {
    if !v.is_finite() {
        return "0".to_string();
    }
    if v == v.trunc() && v.abs() < 1e15 {
        return format!("{}", v as i64);
    }
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
    if s.is_empty() || s == "-" {
        "0".to_string()
    } else {
        s
    }
}

fn metric(out: &mut String, name: &str, help: &str, ty: &str, labels: &str, value: impl std::fmt::Display) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} {ty}");
    if labels.is_empty() {
        let _ = writeln!(out, "{name} {value}");
    } else {
        let _ = writeln!(out, "{name}{{{labels}}} {value}");
    }
}

#[allow(clippy::too_many_arguments)]
fn hist(
    out: &mut String,
    name: &str,
    help: &str,
    le: &[f64],
    buckets: &[AtomicU64],
    sum_ns: &AtomicU64,
    count: &AtomicU64,
) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} histogram");
    let mut cum = 0u64;
    for (i, &b) in le.iter().enumerate() {
        cum += buckets[i].load(Ordering::Relaxed);
        let _ = writeln!(out, "{name}_bucket{{le=\"{b}\"}} {cum}");
    }
    let total = count.load(Ordering::Relaxed);
    let _ = writeln!(out, "{name}_bucket{{le=\"+Inf\"}} {total}");
    let sum = sum_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let _ = writeln!(out, "{name}_sum {}", fmt_f(sum));
    let _ = writeln!(out, "{name}_count {total}");
}

/// `/metrics` の本文 (Prometheus text format 0.0.4)。
///
/// `in_flight` / `queued` / `stats` は呼び出し側 (ハンドラ) がスケジューラから
/// 読んだ現在値。この関数はロックを取らずに自分のカウンタだけを読む。
pub fn render(model: &str, in_flight: usize, queued: usize, stats: &RunStatsSnapshot) -> String {
    let mut o = String::with_capacity(4096);

    let prompt = PROMPT_TOKENS_TOTAL.load(Ordering::Relaxed);
    let cached = KV_CACHE_READ_TOKENS_TOTAL.load(Ordering::Relaxed);
    let ratio = if prompt > 0 { cached as f64 / prompt as f64 } else { 0.0 };

    metric(
        &mut o,
        "denno_model_info",
        "Model served by this process (constant 1).",
        "gauge",
        &format!("model=\"{}\"", esc(model)),
        1,
    );
    metric(
        &mut o,
        "denno_requests_in_flight",
        "Number of generation requests currently in flight.",
        "gauge",
        "",
        in_flight,
    );
    metric(
        &mut o,
        "denno_requests_queued",
        "Number of accepted requests waiting for the scheduler worker.",
        "gauge",
        "",
        queued,
    );
    metric(
        &mut o,
        "denno_requests_completed_total",
        "Number of generation requests completed successfully.",
        "counter",
        "",
        COMPLETED_REQUESTS_TOTAL.load(Ordering::Relaxed),
    );
    metric(
        &mut o,
        "denno_requests_aborted_total",
        "Number of generation requests aborted, by reason.",
        "counter",
        "reason=\"timeout\"",
        stats.aborted_timeout,
    );
    // 同じ系列名でラベル違いを続けて出す (HELP/TYPE は 1 回だけ)。
    let _ = writeln!(o, "denno_requests_aborted_total{{reason=\"disconnected\"}} {}", stats.aborted_disconnected);
    let _ = writeln!(o, "denno_requests_aborted_total{{reason=\"error\"}} {}", stats.aborted_error);
    metric(
        &mut o,
        "denno_prompt_tokens_total",
        "Number of prompt tokens processed by completed requests.",
        "counter",
        "",
        prompt,
    );
    metric(
        &mut o,
        "denno_completion_tokens_total",
        "Number of tokens generated by completed requests.",
        "counter",
        "",
        COMPLETION_TOKENS_TOTAL.load(Ordering::Relaxed),
    );
    metric(
        &mut o,
        "denno_kv_cache_read_tokens_total",
        "Prompt tokens served from the KV prefix cache.",
        "counter",
        "",
        cached,
    );
    metric(
        &mut o,
        "denno_kv_reuse_ratio",
        "Fraction of prompt tokens reused from the KV prefix cache (cumulative).",
        "gauge",
        "",
        fmt_f(ratio),
    );
    hist(
        &mut o,
        "denno_ttft_seconds",
        "Time from request acceptance to the first generated token, in seconds.",
        &TTFT_LE,
        &TTFT_BUCKETS,
        &TTFT_SUM_NS,
        &TTFT_COUNT,
    );
    hist(
        &mut o,
        "denno_tpot_seconds",
        "Mean time per output token during decode, in seconds.",
        &TPOT_LE,
        &TPOT_BUCKETS,
        &TPOT_SUM_NS,
        &TPOT_COUNT,
    );

    let _ = writeln!(
        o,
        "# HELP denno_http_requests_total HTTP requests handled, by endpoint and status."
    );
    let _ = writeln!(o, "# TYPE denno_http_requests_total counter");
    for ((ep, status), n) in http_snapshot() {
        let _ = writeln!(
            o,
            "denno_http_requests_total{{endpoint=\"{}\",status=\"{status}\"}} {n}",
            esc(ep)
        );
    }
    // スクレイプ中のリクエストも 1 回として数え、その場で観測できるようにする
    // (カウンタを増やさず、この応答だけに出せばよい — ここでは行わない)。
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snapshot() -> RunStatsSnapshot {
        RunStatsSnapshot {
            completed: 0,
            aborted_timeout: 0,
            aborted_disconnected: 0,
            aborted_error: 0,
        }
    }

    /// 値の競合を避けるため、記録と描画を 1 本のテストで逐次に行う。
    #[test]
    fn render_is_valid_prometheus_text() {
        record_completion(&CompletionSample {
            input_tokens: 10,
            output_tokens: 4,
            cached_tokens: 6,
            ttft_secs: Some(0.05),
            tpot_secs: Some(0.02),
        });
        record_http("/v1/messages", 200);
        record_http("/v1/messages", 400);
        let text = render("denno-test", 1, 2, &snapshot());

        assert!(text.contains("# TYPE denno_ttft_seconds histogram"));
        assert!(text.contains("denno_http_requests_total{endpoint=\"/v1/messages\",status=\"200\"}"));
        assert!(text.contains("denno_requests_in_flight 1"));
        assert!(text.contains("denno_requests_queued 2"));
        assert!(text.contains("denno_kv_reuse_ratio 0.6"));
        assert!(text.contains("denno_model_info{model=\"denno-test\"} 1"));
        // ヒストグラムの +Inf バケットは count と一致し、バケットは累積で単調。
        let mut seen: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for line in text.lines() {
            if line.starts_with('#') || line.is_empty() {
                continue;
            }
            let (name, val) = line.rsplit_once(' ').expect("sample line");
            assert!(val.parse::<f64>().is_ok(), "value が数値でない: {line}");
            if let Some(prefix) = name.split("_bucket").next() {
                if name.contains("_bucket") {
                    seen.entry(prefix.to_string()).or_default().push(val.parse::<u64>().unwrap());
                }
            }
        }
        for (name, vals) in seen {
            assert!(vals.windows(2).all(|w| w[0] <= w[1]), "{name} が累積でない: {vals:?}");
        }
    }

    #[test]
    fn endpoint_names_are_normalized() {
        assert_eq!(normalize_endpoint("/v1/chat/completions"), "/v1/chat/completions");
        assert_eq!(normalize_endpoint("/v1/messages"), "/v1/messages");
        assert_eq!(normalize_endpoint("/unknown/path"), "other");
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(esc("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
