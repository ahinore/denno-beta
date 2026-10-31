//! prefill の expert GEMM (`gemm_iq` 系) の 1 行カウンタ
//! (`DENNO_GEMM_IQ_KPROF=1`、既定 off)。
//!
//! decode の expert GEMV は [`crate::model::exp_kprof`] が見るが、prefill の
//! バッチ GEMM ([`crate::model::moe_expert::gemm_rows`] の IQ 系経路) は別経路
//! なのでここで独立に積む。**読み取り専用のカウンタ**で、数値経路には触れない
//! (off のときは `on()` の分岐 1 回だけ)。
//!
//! - `calls` / 重みバイト / MAC / 経過時間は `gemm_rows` の IQ 分岐から
//!   1 呼び出し (行チャンクタスク) ずつ積む。8 スレッドで走ると cpu 時間は
//!   スレッド合計になる (GB/s はその比なので「1 スレッド換算」の目安)。
//! - `m` (expert あたりトークン数) の分布は 8 ビン。mean / max も出す。
//! - dequant / fma の内訳は v2 カーネル ([`crate::model::moe_expert::gemm_iq_v2`])
//!   が [`SAMPLE_EVERY`] ブロックに 1 回 `rdtsc` で区切って積む。rdtsc の分だけ
//!   カーネルが遅くなるので **比率だけ**を読むこと。
//!
//! 出力は serve-flash の prefill / decode 終端 (`prof_flash_q4`) で 1 行
//! (`DENNO_EXP_KPROF` と同じ流儀)。

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// rdtsc を取る間隔 (ブロック数)。区間の**比率**は偏らないが、
/// 絶対値はこの倍数で戻す ([`line`] が掛ける)。
pub const SAMPLE_EVERY: usize = 8;

/// m のビン数 (1 / 2-4 / 5-8 / 9-16 / 17-32 / 33-64 / 65-128 / 129+)。
pub const NBUCKET: usize = 8;

/// m のビンの名前 (レポート用)。
pub const BUCKET_NAME: [&str; NBUCKET] =
    ["1", "2-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+"];

static CALLS: AtomicU64 = AtomicU64::new(0);
static CALLS_V2: AtomicU64 = AtomicU64::new(0);
static CALLS_Z: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
static MACS: AtomicU64 = AtomicU64::new(0);
static NS: AtomicU64 = AtomicU64::new(0);
static M_SUM: AtomicU64 = AtomicU64::new(0);
static M_MAX: AtomicU64 = AtomicU64::new(0);
static M_MIN: AtomicU64 = AtomicU64::new(u64::MAX);
static BUCKET_CALLS: [AtomicU64; NBUCKET] = const { [const { AtomicU64::new(0) }; NBUCKET] };
static DEQ_TICKS: AtomicU64 = AtomicU64::new(0);
static FMA_TICKS: AtomicU64 = AtomicU64::new(0);
static WRITE_TICKS: AtomicU64 = AtomicU64::new(0);
/// m ビンごとの dequant / fma ticks (P85)。`add_deq` / `add_fma` が `m` を
/// 受けて振り分ける。書き込み区間は行単位でサンプルするのでビンに切らない。
static BUCKET_DEQ: [AtomicU64; NBUCKET] = const { [const { AtomicU64::new(0) }; NBUCKET] };
static BUCKET_FMA: [AtomicU64; NBUCKET] = const { [const { AtomicU64::new(0) }; NBUCKET] };
/// expert の数 (flat スケジューラが層ごとに 1 回積む)。
/// `nonempty` = トークンを 1 個以上受け取った expert、total = 層の全 expert。
static EXP_NONEMPTY: AtomicU64 = AtomicU64::new(0);
static EXP_TOTAL: AtomicU64 = AtomicU64::new(0);

/// 計測が有効か (既定 off)。`DENNO_GEMM_IQ_KPROF` が立っているときだけ true。
#[inline]
pub fn on() -> bool {
    #[cfg(test)]
    if FORCE.load(Ordering::Relaxed) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("DENNO_GEMM_IQ_KPROF").is_some())
}

/// テストが env を立てずに計測経路を通すための強制フラグ
/// (env は `OnceLock` なのでプロセス内で 1 回しか読めない)。
#[cfg(test)]
static FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 計測をテストから強制する (`false` で env の解釈に戻る)。
#[cfg(test)]
pub fn set_for_test(v: bool) {
    FORCE.store(v, Ordering::Relaxed);
}

/// `m` (expert あたりトークン数) のビン番号。純粋関数 (テストで固定する)。
pub fn bucket(m: usize) -> usize {
    match m {
        1 => 0,
        2..=4 => 1,
        5..=8 => 2,
        9..=16 => 3,
        17..=32 => 4,
        33..=64 => 5,
        65..=128 => 6,
        _ => 7,
    }
}

/// gemm_iq 系カーネル 1 呼び出しぶんの実測を積む。
///
/// `v2` は AVX2 の新経路 ([`gemm_iq_v2`])、`z` は AVX-512 版
/// ([`gemm_iq_z`]) で回ったか、`bytes` は読んだ重みのバイト数
/// (`(r1-r0) * row_bytes`)、`macs` は `m * n_in * (r1-r0)`、`ns` は呼び出しの
/// 経過時間 (スレッドごとに積む)。
#[inline]
pub fn record(v2: bool, z: bool, bytes: usize, macs: usize, m: usize, ns: u64) {
    if !on() {
        return;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    if v2 {
        CALLS_V2.fetch_add(1, Ordering::Relaxed);
    }
    if z {
        CALLS_Z.fetch_add(1, Ordering::Relaxed);
    }
    BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    MACS.fetch_add(macs as u64, Ordering::Relaxed);
    NS.fetch_add(ns, Ordering::Relaxed);
    M_SUM.fetch_add(m as u64, Ordering::Relaxed);
    M_MAX.fetch_max(m as u64, Ordering::Relaxed);
    M_MIN.fetch_min(m as u64, Ordering::Relaxed);
    BUCKET_CALLS[bucket(m)].fetch_add(1, Ordering::Relaxed);
}

/// flat スケジューラが層ごとに 1 回積む expert 数。
/// `nonempty` = トークン 1 個以上を持った expert、`total` = 層の全 expert。
/// 0 トークンの expert は `gemm_rows` まで来ないので、呼び出し側で数える。
#[inline]
pub fn record_experts(nonempty: usize, total: usize) {
    if on() {
        EXP_NONEMPTY.fetch_add(nonempty as u64, Ordering::Relaxed);
        EXP_TOTAL.fetch_add(total as u64, Ordering::Relaxed);
    }
}

/// v2 系カーネルの dequant 区間 (タイル生成) の ticks を積む。
/// `m` はその呼び出しの expert あたりトークン数 (ビン集計用)。
#[inline]
pub fn add_deq(ticks: u64, m: usize) {
    if on() {
        DEQ_TICKS.fetch_add(ticks, Ordering::Relaxed);
        BUCKET_DEQ[bucket(m)].fetch_add(ticks, Ordering::Relaxed);
    }
}

/// v2 系カーネルの fma 区間 (タイルを m トークンに配る) の ticks を積む。
#[inline]
pub fn add_fma(ticks: u64, m: usize) {
    if on() {
        FMA_TICKS.fetch_add(ticks, Ordering::Relaxed);
        BUCKET_FMA[bucket(m)].fetch_add(ticks, Ordering::Relaxed);
    }
}

/// v2 系カーネルの書き込み区間 (行末の横加算 + out store) の ticks を積む。
/// 行単位でサンプルする (呼び出し側が `SAMPLE_EVERY` 行に 1 回)。
#[inline]
pub fn add_write(ticks: u64, _m: usize) {
    if on() {
        WRITE_TICKS.fetch_add(ticks, Ordering::Relaxed);
    }
}

/// rdtsc の 1 µs あたりの刻み。較正は `line()` から 1 回だけ走る
/// ([`crate::model::exp_kprof`] と同じ方式)。
#[cfg(target_arch = "x86_64")]
fn ticks_per_us() -> f64 {
    static F: OnceLock<f64> = OnceLock::new();
    *F.get_or_init(|| {
        let t0 = std::time::Instant::now();
        let r0 = unsafe { std::arch::x86_64::_rdtsc() };
        while t0.elapsed().as_micros() < 500 {
            std::hint::spin_loop();
        }
        let r1 = unsafe { std::arch::x86_64::_rdtsc() };
        let us = t0.elapsed().as_secs_f64() * 1e6;
        if us <= 0.0 {
            1.0
        } else {
            (r1.wrapping_sub(r0)) as f64 / us
        }
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn ticks_per_us() -> f64 {
    0.0
}

fn ticks_to_ns(ticks: u64) -> f64 {
    let f = ticks_per_us();
    if f <= 0.0 {
        0.0
    } else {
        ticks as f64 / f * 1e3
    }
}

/// 内訳を 1 行の文字列にする。無効時・実測ゼロのときは `None` (出力ゼロ)。
///
/// P85 で足した区間: `m min` / `nonempty` (トークンを持った expert 数) /
/// `write` (行末の横加算 + store、行サンプル) / ビンごとの dequant 比
/// (`dm=` 以下、calls>0 のビンだけ出す。fma 比は 100 - deq - write に当たる)。
pub fn line() -> Option<String> {
    if !on() {
        return None;
    }
    let calls = CALLS.load(Ordering::Relaxed);
    let deq_t = DEQ_TICKS.load(Ordering::Relaxed);
    let fma_t = FMA_TICKS.load(Ordering::Relaxed);
    let wr_t = WRITE_TICKS.load(Ordering::Relaxed);
    if calls == 0 && deq_t + fma_t + wr_t == 0 {
        return None;
    }
    let v2 = CALLS_V2.load(Ordering::Relaxed);
    let zc = CALLS_Z.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let macs = MACS.load(Ordering::Relaxed);
    let ns = NS.load(Ordering::Relaxed);
    let m_sum = M_SUM.load(Ordering::Relaxed);
    let m_max = M_MAX.load(Ordering::Relaxed);
    let m_min = M_MIN.load(Ordering::Relaxed);
    let cpu_ms = ns as f64 / 1e6;
    let gbs = if ns > 0 {
        bytes as f64 / (ns as f64 / 1e9) / 1e9
    } else {
        0.0
    };
    let mut s = format!(
        "[giq-kprof] calls={calls} v2={v2} z={zc} bytes={:.1} MB macs={:.2} G cpu={cpu_ms:.1} ms \
         (cpu_sum 換算 {:.2} GB/s) | m min={} mean={:.1} max={m_max}",
        bytes as f64 / 1e6,
        macs as f64 / 1e9,
        gbs,
        if calls > 0 && m_min != u64::MAX {
            m_min.to_string()
        } else {
            "-".to_string()
        },
        if calls > 0 {
            m_sum as f64 / calls as f64
        } else {
            0.0
        }
    );
    let (ne, tot) = (
        EXP_NONEMPTY.load(Ordering::Relaxed),
        EXP_TOTAL.load(Ordering::Relaxed),
    );
    if tot > 0 {
        s.push_str(&format!(" nonempty={ne}/{tot}"));
    }
    s.push_str(" [");
    for (i, name) in BUCKET_NAME.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{name}:{}", BUCKET_CALLS[i].load(Ordering::Relaxed)));
    }
    s.push(']');
    let wr = wr_t; // 行サンプル: SAMPLE_EVERY 行に 1 回
    if deq_t + fma_t > 0 {
        let d = ticks_to_ns(deq_t) * SAMPLE_EVERY as f64 / 1e6;
        let f = ticks_to_ns(fma_t) * SAMPLE_EVERY as f64 / 1e6;
        let w = ticks_to_ns(wr) * SAMPLE_EVERY as f64 / 1e6;
        let sum = d + f + w;
        let pct = |v: f64| if sum > 0.0 { v * 100.0 / sum } else { 0.0 };
        s.push_str(&format!(
            " | dequant {d:.1} ms ({:.0}%) fma {f:.1} ms ({:.0}%) write {w:.1} ms ({:.0}%)",
            pct(d),
            pct(f),
            pct(w)
        ));
        // m ビンごとの dequant 比 (calls>0 のビンだけ)。
        let mut parts: Vec<String> = Vec::new();
        for i in 0..NBUCKET {
            let c = BUCKET_CALLS[i].load(Ordering::Relaxed);
            if c == 0 {
                continue;
            }
            let bd = BUCKET_DEQ[i].load(Ordering::Relaxed) as f64 * SAMPLE_EVERY as f64;
            let bf = BUCKET_FMA[i].load(Ordering::Relaxed) as f64 * SAMPLE_EVERY as f64;
            if bd + bf > 0.0 {
                parts.push(format!("{}:{:.0}%", BUCKET_NAME[i], bd * 100.0 / (bd + bf)));
            }
        }
        if !parts.is_empty() {
            s.push_str(&format!(" | deq%/m [{}]", parts.join(" ")));
        }
    }
    Some(s)
}

/// カウンタを 0 に戻す (`line()` を出したあとに呼ぶ)。
pub fn reset() {
    CALLS.store(0, Ordering::Relaxed);
    CALLS_V2.store(0, Ordering::Relaxed);
    CALLS_Z.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    MACS.store(0, Ordering::Relaxed);
    NS.store(0, Ordering::Relaxed);
    M_SUM.store(0, Ordering::Relaxed);
    M_MAX.store(0, Ordering::Relaxed);
    M_MIN.store(u64::MAX, Ordering::Relaxed);
    for b in &BUCKET_CALLS {
        b.store(0, Ordering::Relaxed);
    }
    for b in &BUCKET_DEQ {
        b.store(0, Ordering::Relaxed);
    }
    for b in &BUCKET_FMA {
        b.store(0, Ordering::Relaxed);
    }
    DEQ_TICKS.store(0, Ordering::Relaxed);
    FMA_TICKS.store(0, Ordering::Relaxed);
    WRITE_TICKS.store(0, Ordering::Relaxed);
    EXP_NONEMPTY.store(0, Ordering::Relaxed);
    EXP_TOTAL.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_covers_all_m_without_gap() {
        assert_eq!(bucket(1), 0);
        assert_eq!(bucket(2), 1);
        assert_eq!(bucket(4), 1);
        assert_eq!(bucket(5), 2);
        assert_eq!(bucket(8), 2);
        assert_eq!(bucket(9), 3);
        assert_eq!(bucket(16), 3);
        assert_eq!(bucket(17), 4);
        assert_eq!(bucket(32), 4);
        assert_eq!(bucket(33), 5);
        assert_eq!(bucket(64), 5);
        assert_eq!(bucket(65), 6);
        assert_eq!(bucket(128), 6);
        assert_eq!(bucket(129), 7);
        assert_eq!(bucket(100_000), 7);
    }
}
