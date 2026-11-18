//! task 347: Prometheus `/metrics` (sparkDash が読む形)。
//!
//! 目的は「vLLM の reader で denno を見られるようにする」ことなので、名前は
//! vLLM の系列名 (`vllm:` 接頭辞) をそのまま使い、vLLM に対応する名前が無い
//! ものだけ `denno_` を付ける。`model_name` ラベルは配信中のモデル名。
//!
//! 設計の縛り (task 347 の Do 2):
//! - **decode / prefill のホットループはロックも確保もしない**。ここは
//!   `AtomicU64` の加算だけで、確保はスクレイプ時 (`render`) にだけ起きる。
//! - スクレイプは読み取りスレッド (`spawn_acceptor`) から走るので、生成中でも
//!   待たされない。そのため `render` は**ロックを取らない**: ゲージ類は
//!   リクエスト終端で 1 回サンプルして atomic に置き、`render` はそれを読む。
//! - ゲートは `DENNO_METRICS` (既定 on)。`0` / `false` / `no` で経路ごと消える。

use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

/// ゲート。既定 on (`0` / `false` / `no` のときだけ off)。
pub const ENV_ON: &str = "DENNO_METRICS";

/// `/v1/models` に `owned_by: "denno"` を足すかのゲート (既定 on)。
pub const ENV_OWNED_BY: &str = "DENNO_MODELS_OWNED_BY";

/// このプロセスのビルド SHA (build.rs が `DENNO_BUILD_SHA` を渡す)。
pub const BUILD_SHA: &str = match option_env!("DENNO_BUILD_SHA") {
    Some(s) => s,
    None => "unknown",
};

// ---------------------------------------------------------------------------
// ゲート
// ---------------------------------------------------------------------------

static ON: AtomicBool = AtomicBool::new(false);

/// `DENNO_METRICS` の値から on/off を決める (純関数。既定 on)。
pub fn enabled_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false") | Some("no"))
}

/// 起動時に 1 回呼ぶ。以後 [`enabled`] は atomic load 1 回。
pub fn init_from_env() {
    ON.store(
        enabled_from_value(std::env::var(ENV_ON).ok().as_deref()),
        Ordering::Relaxed,
    );
}

pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// テスト用: 明示的に on/off する。
#[cfg(test)]
pub fn set_enabled(v: bool) {
    ON.store(v, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// モデルの同一性 (起動時に 1 回)
// ---------------------------------------------------------------------------

static MODEL: OnceLock<String> = OnceLock::new();
static MAX_CTX: AtomicU64 = AtomicU64::new(0);

/// 起動時に 1 回。`model_name` ラベルと `kv_cache_size_tokens` に使う。
pub fn set_model(name: &str, max_ctx: usize) {
    let _ = MODEL.set(name.to_string());
    MAX_CTX.store(max_ctx as u64, Ordering::Relaxed);
}

fn model_name() -> &'static str {
    MODEL.get().map(|s| s.as_str()).unwrap_or("")
}

fn max_ctx() -> u64 {
    MAX_CTX.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// カウンタ (加算のみ)
// ---------------------------------------------------------------------------

static PROMPT_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
static GENERATION_TOKENS_TOTAL: AtomicU64 = AtomicU64::new(0);
static REQUESTS_TOTAL: AtomicU64 = AtomicU64::new(0);
static REQUESTS_FAILED_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREFIX_QUERIES_TOTAL: AtomicU64 = AtomicU64::new(0);
static PREFIX_HITS_TOTAL: AtomicU64 = AtomicU64::new(0);
static PROMPT_TOKENS_CACHED_TOTAL: AtomicU64 = AtomicU64::new(0);
/// denno は 1 スロットで先取り (preemption) をしない。系列は vLLM の reader が
/// 読むので存在だけ合わせておく (常に 0)。
static NUM_PREEMPTIONS_TOTAL: AtomicU64 = AtomicU64::new(0);
/// expert のミスでファイル (mmap) から読んだバイトの積算。
static EXPERT_DISK_READ_BYTES_TOTAL: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// ゲージ (リクエスト終端でサンプル)
// ---------------------------------------------------------------------------

static REQUESTS_RUNNING: AtomicU64 = AtomicU64::new(0);
static CTX_TOKENS: AtomicU64 = AtomicU64::new(0);
static LAST_DECODE_MS_PER_TOKEN: AtomicU64 = AtomicU64::new(0); // f64 のビット
static LAST_PREFILL_TOK_S: AtomicU64 = AtomicU64::new(0); // f64 のビット
static LAST_VRAM_HIT_RATE: AtomicU64 = AtomicU64::new(0); // f64 のビット
static EXPERT_VRAM_USED_BYTES: AtomicU64 = AtomicU64::new(0);
static EXPERT_VRAM_BUDGET_BYTES: AtomicU64 = AtomicU64::new(0);
static EXPERT_RAM_TIER_BYTES: AtomicU64 = AtomicU64::new(0);
static EXPERT_RAM_TIER_EXPERTS: AtomicU64 = AtomicU64::new(0);

fn put_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

fn get_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Relaxed))
}

// ---------------------------------------------------------------------------
// ヒストグラム (vLLM と同じ 4 本)
// ---------------------------------------------------------------------------

/// 上限は「これ以上は 1 つのバケット」で十分 (denno の 14k prefill は 6-25 秒)。
const LE_LAT: [f64; 12] = [
    0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,
];

/// le の数 + 1 (+Inf の枠)。observe() は le.len() 番目に +Inf を積むので、
/// 12 個だと 128 秒を超えた 1 件で index out of bounds の panic になる。
const N_LAT: usize = LE_LAT.len() + 1;

static TTFT_BUCKETS: [AtomicU64; N_LAT] = [const { AtomicU64::new(0) }; N_LAT];
static TTFT_SUM_NS: AtomicU64 = AtomicU64::new(0);
static TTFT_COUNT: AtomicU64 = AtomicU64::new(0);

static E2E_BUCKETS: [AtomicU64; N_LAT] = [const { AtomicU64::new(0) }; N_LAT];
static E2E_SUM_NS: AtomicU64 = AtomicU64::new(0);
static E2E_COUNT: AtomicU64 = AtomicU64::new(0);

static INTER_BUCKETS: [AtomicU64; N_LAT] = [const { AtomicU64::new(0) }; N_LAT];
static INTER_SUM_NS: AtomicU64 = AtomicU64::new(0);
static INTER_COUNT: AtomicU64 = AtomicU64::new(0);

static PREFILL_BUCKETS: [AtomicU64; N_LAT] = [const { AtomicU64::new(0) }; N_LAT];
static PREFILL_SUM_NS: AtomicU64 = AtomicU64::new(0);
static PREFILL_COUNT: AtomicU64 = AtomicU64::new(0);

/// 1 サンプルを積む。バケットは**非累積**で持ち、描画時に累積へ直す。
/// ホットループから呼ばれるので、ロックも確保もしない (atomic add だけ)。
fn observe(le: &[f64], buckets: &[AtomicU64], sum_ns: &AtomicU64, count: &AtomicU64, secs: f64) {
    let mut i = le.len();
    for (k, &b) in le.iter().enumerate() {
        if secs <= b {
            i = k;
            break;
        }
    }
    // +Inf は最終バケットの外 (le の末尾より大きい値) として数える。
    let slot = if i < le.len() { i } else { le.len() };
    buckets[slot].fetch_add(1, Ordering::Relaxed);
    let ns = (secs.max(0.0) * 1e9) as u64;
    sum_ns.fetch_add(ns, Ordering::Relaxed);
    count.fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// リクエストの記録
// ---------------------------------------------------------------------------

/// 最初の 1 トークンが出た瞬間に 1 回だけ呼ぶ (TTFT のヒストグラム)。
/// リクエストあたり 1 回なので、ホットループの負担は無いに等しい。
pub fn note_first_token(secs: f64) {
    if !enabled() {
        return;
    }
    observe(&LE_LAT, &TTFT_BUCKETS, &TTFT_SUM_NS, &TTFT_COUNT, secs);
}

/// 1 リクエストぶんの値。`generate()` の終端で 1 回だけ作る。
pub struct ReqSample {
    /// プロンプトの全トークン数 (`usage.input_tokens`)
    pub prompt_tokens: u64,
    /// うち prefix cache で再利用したトークン数
    pub reused_tokens: u64,
    /// 生成したトークン数 (`usage.output_tokens`)
    pub output_tokens: u64,
    /// prefill の秒 (再利用した前置きは含まない)
    pub prefill_secs: f64,
    /// リクエスト全体の秒
    pub e2e_secs: f64,
    /// トークン間隔の平均 (out >= 2 のときだけ)
    pub inter_token_secs: Option<f64>,
    /// 直近リクエストの decode 速度 (ms/token)
    pub decode_ms_per_token: Option<f64>,
    /// 生成が途中で失敗したか
    pub failed: bool,
    /// いまの文脈長 (トークン)
    pub ctx_tokens: u64,
}

/// リクエスト終端の記録。ロックも確保もしない。
pub fn record_request(s: &ReqSample) {
    if !enabled() {
        return;
    }
    PROMPT_TOKENS_TOTAL.fetch_add(s.prompt_tokens, Ordering::Relaxed);
    GENERATION_TOKENS_TOTAL.fetch_add(s.output_tokens, Ordering::Relaxed);
    REQUESTS_TOTAL.fetch_add(1, Ordering::Relaxed);
    if s.failed {
        REQUESTS_FAILED_TOTAL.fetch_add(1, Ordering::Relaxed);
    }
    PREFIX_QUERIES_TOTAL.fetch_add(s.prompt_tokens, Ordering::Relaxed);
    PREFIX_HITS_TOTAL.fetch_add(s.reused_tokens, Ordering::Relaxed);
    PROMPT_TOKENS_CACHED_TOTAL.fetch_add(s.reused_tokens, Ordering::Relaxed);
    observe(&LE_LAT, &E2E_BUCKETS, &E2E_SUM_NS, &E2E_COUNT, s.e2e_secs);
    if let Some(t) = s.inter_token_secs {
        observe(&LE_LAT, &INTER_BUCKETS, &INTER_SUM_NS, &INTER_COUNT, t);
    }
    observe(
        &LE_LAT,
        &PREFILL_BUCKETS,
        &PREFILL_SUM_NS,
        &PREFILL_COUNT,
        s.prefill_secs,
    );
    if let Some(ms) = s.decode_ms_per_token {
        put_f64(&LAST_DECODE_MS_PER_TOKEN, ms);
    }
    if s.prefill_secs > 0.0 {
        let tok = s.prompt_tokens.saturating_sub(s.reused_tokens);
        put_f64(&LAST_PREFILL_TOK_S, tok as f64 / s.prefill_secs);
    }
    CTX_TOKENS.store(s.ctx_tokens, Ordering::Relaxed);
}

/// expert の VRAM ヒット率 (直前のリクエスト) を書く。
pub fn record_expert_vram(hit_rate: Option<f64>) {
    if let Some(r) = hit_rate {
        put_f64(&LAST_VRAM_HIT_RATE, r);
    }
}

/// `/proc/self/io` の `read_bytes`。プロセスがストレージから読んだ累積バイトで、
/// expert の重みが主。ページキャッシュに載っているぶんは増えない (それが正しい)。
fn process_read_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/io").ok()?;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            return v.trim().parse::<u64>().ok();
        }
    }
    None
}

/// ディスクから読んだバイトをサンプルする (リクエスト終端と起動後)。
/// `/proc/self/io` は単調増加なので、max を取るだけでよい。
pub fn sample_disk_reads() {
    if !enabled() {
        return;
    }
    if let Some(v) = process_read_bytes() {
        EXPERT_DISK_READ_BYTES_TOTAL.fetch_max(v, Ordering::Relaxed);
    }
}

/// VRAM / RAM tier のゲージをサンプルする (リクエスト終端と起動後)。
pub fn sample_tiers() {
    if !enabled() {
        return;
    }
    if let Some((used, _pinned, cap)) = denno_engine::model::expert_vram::budget_usage() {
        EXPERT_VRAM_USED_BYTES.store(used as u64, Ordering::Relaxed);
        EXPERT_VRAM_BUDGET_BYTES.store(cap as u64, Ordering::Relaxed);
    }
    if let Some((experts, _slots, _pinned, used)) = denno_engine::model::expert_ram::status() {
        EXPERT_RAM_TIER_BYTES.store(used, Ordering::Relaxed);
        EXPERT_RAM_TIER_EXPERTS.store(experts as u64, Ordering::Relaxed);
    }
}

/// 生成中の印 (`num_requests_running`)。落ちる経路でも戻るようにガードで持つ。
pub struct RunningGuard;

impl RunningGuard {
    pub fn enter() -> Self {
        REQUESTS_RUNNING.fetch_add(1, Ordering::Relaxed);
        RunningGuard
    }
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        // 0 未満にしない (二重 drop は無いが、飽和させておく)。
        let _ = REQUESTS_RUNNING.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
            Some(v.saturating_sub(1))
        });
    }
}

// ---------------------------------------------------------------------------
// 描画 (ここだけ確保する)
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// ヒストグラム 1 本を描く。`_bucket` は累積で、最後の `le="+Inf"` は `_count` と一致する。
#[allow(clippy::too_many_arguments)]
fn hist(
    out: &mut String,
    name: &str,
    help: &str,
    labels: &str,
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
        let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{b}\"}} {cum}");
    }
    let total = count.load(Ordering::Relaxed);
    // le=+Inf は残り全部 (累積の最後 = count と一致させる)。
    let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {total}");
    let sum = sum_ns.load(Ordering::Relaxed) as f64 / 1e9;
    let _ = writeln!(out, "{name}_sum{{{labels}}} {}", fmt_f(sum));
    let _ = writeln!(out, "{name}_count{{{labels}}} {total}");
}

/// f64 を Prometheus が読める形にする (指数表記を避ける)。
fn fmt_f(v: f64) -> String {
    if !v.is_finite() {
        return "0".into();
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.6}");
        let s = s.trim_end_matches('0').trim_end_matches('.').to_string();
        if s.is_empty() || s == "-" {
            "0".into()
        } else {
            s
        }
    }
}

fn counter(out: &mut String, name: &str, help: &str, labels: &str, v: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} counter");
    let _ = writeln!(out, "{name}{{{labels}}} {v}");
}

fn gauge(out: &mut String, name: &str, help: &str, labels: &str, v: u64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name}{{{labels}}} {v}");
}

fn gauge_f(out: &mut String, name: &str, help: &str, labels: &str, v: f64) {
    let _ = writeln!(out, "# HELP {name} {help}");
    let _ = writeln!(out, "# TYPE {name} gauge");
    let _ = writeln!(out, "{name}{{{labels}}} {}", fmt_f(v));
}

/// `/metrics` の本文 (Prometheus text format 0.0.4)。
///
/// ロックも確保もしない…と言いたいところだが、文字列の組み立てで確保する。
/// スクレイプ経路だけなので許す (ホットループからは呼ばない)。
pub fn render() -> String {
    let model = esc(model_name());
    let ml = format!("model_name=\"{model}\"");
    let mut o = String::with_capacity(4096);

    counter(
        &mut o,
        "vllm:prompt_tokens_total",
        "Number of prompt tokens processed.",
        &ml,
        PROMPT_TOKENS_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut o,
        "vllm:generation_tokens_total",
        "Number of generation tokens processed.",
        &ml,
        GENERATION_TOKENS_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut o,
        "vllm:request_success_total",
        "Number of requests processed (any outcome).",
        &ml,
        REQUESTS_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut o,
        "vllm:request_failure_total",
        "Number of requests that ended with an error.",
        &ml,
        REQUESTS_FAILED_TOTAL.load(Ordering::Relaxed),
    );
    gauge(
        &mut o,
        "vllm:num_requests_running",
        "Number of requests currently running.",
        &ml,
        REQUESTS_RUNNING.load(Ordering::Relaxed),
    );
    gauge(
        &mut o,
        "vllm:num_requests_waiting",
        "Number of requests waiting to be processed (denno serves one at a time).",
        &ml,
        0,
    );
    hist(
        &mut o,
        "vllm:time_to_first_token_seconds",
        "Histogram of time to first token in seconds.",
        &ml,
        &LE_LAT,
        &TTFT_BUCKETS,
        &TTFT_SUM_NS,
        &TTFT_COUNT,
    );
    hist(
        &mut o,
        "vllm:e2e_request_latency_seconds",
        "Histogram of end to end request latency in seconds.",
        &ml,
        &LE_LAT,
        &E2E_BUCKETS,
        &E2E_SUM_NS,
        &E2E_COUNT,
    );
    hist(
        &mut o,
        "vllm:inter_token_latency_seconds",
        "Histogram of per token latency in seconds (one sample per request: the mean gap).",
        &ml,
        &LE_LAT,
        &INTER_BUCKETS,
        &INTER_SUM_NS,
        &INTER_COUNT,
    );
    hist(
        &mut o,
        "vllm:request_prefill_time_seconds",
        "Histogram of prefill time in seconds.",
        &ml,
        &LE_LAT,
        &PREFILL_BUCKETS,
        &PREFILL_SUM_NS,
        &PREFILL_COUNT,
    );
    counter(
        &mut o,
        "vllm:prefix_cache_queries_total",
        "Number of prompt tokens looked up in the prefix cache.",
        &ml,
        PREFIX_QUERIES_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut o,
        "vllm:prefix_cache_hits_total",
        "Number of prompt tokens served from the prefix cache.",
        &ml,
        PREFIX_HITS_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut o,
        "vllm:prompt_tokens_cached_total",
        "Number of prompt tokens that were already cached.",
        &ml,
        PROMPT_TOKENS_CACHED_TOTAL.load(Ordering::Relaxed),
    );
    counter(
        &mut o,
        "vllm:num_preemptions_total",
        "Number of request preemptions (denno has one slot and never preempts).",
        &ml,
        NUM_PREEMPTIONS_TOTAL.load(Ordering::Relaxed),
    );
    {
        let ctx = CTX_TOKENS.load(Ordering::Relaxed);
        let cap = max_ctx();
        let perc = if cap > 0 {
            (ctx as f64 / cap as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        gauge_f(
            &mut o,
            "vllm:kv_cache_usage_perc",
            "Ratio of used context slots to the context limit (0..1).",
            &ml,
            perc,
        );
    }
    {
        let _ = writeln!(
            o,
            "# HELP vllm:cache_config_info Information about the cache configuration."
        );
        let _ = writeln!(o, "# TYPE vllm:cache_config_info gauge");
        let _ = writeln!(
            o,
            "vllm:cache_config_info{{{ml},kv_cache_size_tokens=\"{}\"}} 1",
            max_ctx()
        );
    }

    // ---- denno 固有 (vLLM に対応する名前が無いもの) ----
    gauge_f(
        &mut o,
        "denno_last_decode_ms_per_token",
        "Decode speed of the last request in milliseconds per token.",
        &ml,
        get_f64(&LAST_DECODE_MS_PER_TOKEN),
    );
    gauge_f(
        &mut o,
        "denno_last_prefill_tokens_per_second",
        "Prefill speed of the last request in prompt tokens per second.",
        &ml,
        get_f64(&LAST_PREFILL_TOK_S),
    );
    gauge_f(
        &mut o,
        "denno_expert_vram_hit_rate",
        "Expert VRAM hit rate of the last request (0..1).",
        &ml,
        get_f64(&LAST_VRAM_HIT_RATE),
    );
    counter(
        &mut o,
        "denno_expert_disk_read_bytes_total",
        "Bytes read from storage by the process (/proc/self/io read_bytes; expert weights dominate, page cache hits are not counted).",
        &ml,
        EXPERT_DISK_READ_BYTES_TOTAL.load(Ordering::Relaxed),
    );
    gauge(
        &mut o,
        "denno_expert_vram_used_bytes",
        "Expert bytes resident in VRAM.",
        &ml,
        EXPERT_VRAM_USED_BYTES.load(Ordering::Relaxed),
    );
    gauge(
        &mut o,
        "denno_expert_vram_budget_bytes",
        "Expert VRAM budget.",
        &ml,
        EXPERT_VRAM_BUDGET_BYTES.load(Ordering::Relaxed),
    );
    gauge(
        &mut o,
        "denno_expert_ram_tier_bytes",
        "Resident bytes of the host RAM expert tier.",
        &ml,
        EXPERT_RAM_TIER_BYTES.load(Ordering::Relaxed),
    );
    gauge(
        &mut o,
        "denno_expert_ram_tier_experts",
        "Number of experts resident in the host RAM tier.",
        &ml,
        EXPERT_RAM_TIER_EXPERTS.load(Ordering::Relaxed),
    );
    let _ = writeln!(o, "# HELP denno_build_info Build information.");
    let _ = writeln!(o, "# TYPE denno_build_info gauge");
    let _ = writeln!(
        o,
        "denno_build_info{{build_sha=\"{}\",version=\"{}\"}} 1",
        esc(BUILD_SHA),
        env!("CARGO_PKG_VERSION")
    );
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1 行が「コメント」か「名前{ラベル} 値」のどちらかであること。
    fn parse_line(line: &str) -> Result<(String, Vec<(String, String)>, f64), String> {
        if line.starts_with('#') {
            return Err(String::new()); // コメント扱い (呼び側で捨てる)
        }
        let (head, val) = line
            .rsplit_once(' ')
            .ok_or_else(|| format!("空白が無い: {line}"))?;
        let v: f64 = val.parse().map_err(|_| format!("値が数値でない: {line}"))?;
        let (name, rest) = match head.split_once('{') {
            Some((n, r)) => {
                let r = r
                    .strip_suffix('}')
                    .ok_or_else(|| format!("ラベルが閉じていない: {line}"))?;
                (n.to_string(), Some(r))
            }
            None => (head.to_string(), None),
        };
        let mut labels = Vec::new();
        if let Some(r) = rest {
            for kv in r.split(',') {
                let (k, v) = kv
                    .split_once('=')
                    .ok_or_else(|| format!("ラベルが k=v でない: {line}"))?;
                let v = v
                    .strip_prefix('"')
                    .and_then(|x| x.strip_suffix('"'))
                    .ok_or_else(|| format!("ラベル値が引用されていない: {line}"))?;
                labels.push((k.to_string(), v.to_string()));
            }
        }
        if name.is_empty() {
            return Err(format!("名前が空: {line}"));
        }
        for ch in name.chars() {
            if !(ch.is_ascii_alphanumeric() || ch == '_' || ch == ':') {
                return Err(format!("名前の文字が不正 ({ch}): {line}"));
            }
        }
        Ok((name, labels, v))
    }

    /// すべての行が有効なサンプルかコメント。バケットは累積で、最後の
    /// `le="+Inf"` が `_count` と一致する。カウンタは単調増加。
    #[test]
    fn metrics_text_is_valid_prometheus() {
        set_model("synth", 128);
        set_enabled(true);
        note_first_token(0.25);
        record_request(&ReqSample {
            prompt_tokens: 10,
            reused_tokens: 4,
            output_tokens: 6,
            prefill_secs: 0.5,
            e2e_secs: 1.5,
            inter_token_secs: Some(0.2),
            decode_ms_per_token: Some(16.5),
            failed: false,
            ctx_tokens: 16,
        });
        let a = render();
        // 2 回目の記録のあと、カウンタは減らない。
        record_request(&ReqSample {
            prompt_tokens: 7,
            reused_tokens: 1,
            output_tokens: 3,
            prefill_secs: 0.2,
            e2e_secs: 0.4,
            inter_token_secs: None,
            decode_ms_per_token: Some(17.0),
            failed: false,
            ctx_tokens: 10,
        });
        let b = render();

        let mut n_samples = 0usize;
        for text in [&a, &b] {
            for line in text.lines() {
                if line.starts_with('#') {
                    assert!(
                        line.starts_with("# HELP ") || line.starts_with("# TYPE "),
                        "コメントの形が変: {line}"
                    );
                    continue;
                }
                parse_line(line).unwrap_or_else(|e| panic!("{e}"));
                n_samples += 1;
            }
        }
        assert!(n_samples > 0);

        // カウンタの単調性。
        let counters = |t: &str| -> Vec<(String, f64)> {
            let mut v = Vec::new();
            let mut ty = String::new();
            for line in t.lines() {
                if let Some(rest) = line.strip_prefix("# TYPE ") {
                    let mut it = rest.split_whitespace();
                    let name = it.next().unwrap_or("").to_string();
                    let kind = it.next().unwrap_or("").to_string();
                    if kind == "counter" {
                        ty = name;
                    }
                    continue;
                }
                if ty.is_empty() || line.starts_with('#') {
                    continue;
                }
                if let Ok((name, _, val)) = parse_line(line) {
                    if name == ty {
                        v.push((name, val));
                    }
                }
            }
            v
        };
        for (name, va) in counters(&a) {
            let vb = counters(&b)
                .into_iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| v)
                .unwrap_or(va);
            assert!(vb >= va, "カウンタ {name} が減った: {va} -> {vb}");
        }

        // ヒストグラム: バケットが累積で、+Inf == _count。
        for base in [
            "vllm:time_to_first_token_seconds",
            "vllm:e2e_request_latency_seconds",
            "vllm:inter_token_latency_seconds",
            "vllm:request_prefill_time_seconds",
        ] {
            let mut prev = 0.0f64;
            let mut found_inf = false;
            for line in b.lines() {
                if !line.starts_with(&format!("{base}_bucket")) {
                    continue;
                }
                let (_, labels, v) = parse_line(line).expect("bucket");
                let le = labels
                    .iter()
                    .find(|(k, _)| k == "le")
                    .map(|(_, v)| v.clone())
                    .expect("le ラベル");
                assert!(v >= prev, "{base} のバケットが累積でない: {line}");
                prev = v;
                if le == "+Inf" {
                    found_inf = true;
                }
            }
            assert!(found_inf, "{base} に le=\"+Inf\" が無い");
            let count_line = format!("{base}_count");
            let mut count = None;
            for line in b.lines() {
                if let Ok((name, _, v)) = parse_line(line) {
                    if name == count_line {
                        count = Some(v);
                    }
                }
            }
            let count = count.unwrap_or_else(|| panic!("{base}_count が無い"));
            assert_eq!(prev, count, "{base}: +Inf {prev} != count {count}");
        }
    }

    /// バケットの振り分けと累積 (2 サンプル目が小さい値でも壊れない)。
    #[test]
    fn histogram_buckets_are_cumulative() {
        set_enabled(true);
        let le = [0.1, 1.0];
        let b = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
        let s = AtomicU64::new(0);
        let c = AtomicU64::new(0);
        observe(&le, &b, &s, &c, 0.05); // <=0.1
        observe(&le, &b, &s, &c, 0.5); // <=1.0
        observe(&le, &b, &s, &c, 5.0); // +Inf
        assert_eq!(b[0].load(Ordering::Relaxed), 1);
        assert_eq!(b[1].load(Ordering::Relaxed), 1);
        assert_eq!(b[2].load(Ordering::Relaxed), 1);
        assert_eq!(c.load(Ordering::Relaxed), 3);
        assert_eq!(s.load(Ordering::Relaxed), 5_550_000_000);
    }

    /// 128 秒 (LE_LAT の末尾) を超える値でも panic しない (+Inf の枠がある)。
    #[test]
    fn histogram_over_last_bound_does_not_panic() {
        assert_eq!(TTFT_BUCKETS.len(), LE_LAT.len() + 1);
        assert_eq!(E2E_BUCKETS.len(), LE_LAT.len() + 1);
        assert_eq!(INTER_BUCKETS.len(), LE_LAT.len() + 1);
        assert_eq!(PREFILL_BUCKETS.len(), LE_LAT.len() + 1);
        let before = PREFILL_BUCKETS[LE_LAT.len()].load(Ordering::Relaxed);
        observe(&LE_LAT, &PREFILL_BUCKETS, &PREFILL_SUM_NS, &PREFILL_COUNT, 200.0);
        assert_eq!(PREFILL_BUCKETS[LE_LAT.len()].load(Ordering::Relaxed), before + 1);
    }

    #[test]
    fn gate_default_on_and_off_values() {
        assert!(enabled_from_value(None));
        assert!(enabled_from_value(Some("1")));
        assert!(!enabled_from_value(Some("0")));
        assert!(!enabled_from_value(Some("false")));
        assert!(!enabled_from_value(Some("no")));
    }
}
