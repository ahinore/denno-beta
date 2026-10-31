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
static BYTES: AtomicU64 = AtomicU64::new(0);
static MACS: AtomicU64 = AtomicU64::new(0);
static NS: AtomicU64 = AtomicU64::new(0);
static M_SUM: AtomicU64 = AtomicU64::new(0);
static M_MAX: AtomicU64 = AtomicU64::new(0);
static BUCKET_CALLS: [AtomicU64; NBUCKET] = const { [const { AtomicU64::new(0) }; NBUCKET] };
static DEQ_TICKS: AtomicU64 = AtomicU64::new(0);
static FMA_TICKS: AtomicU64 = AtomicU64::new(0);

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
/// `v2` は新経路 ([`gemm_iq_v2`]) で回ったか、`bytes` は読んだ重みのバイト数
/// (`(r1-r0) * row_bytes`)、`macs` は `m * n_in * (r1-r0)`、`ns` は呼び出しの
/// 経過時間 (スレッドごとに積む)。
#[inline]
pub fn record(v2: bool, bytes: usize, macs: usize, m: usize, ns: u64) {
    if !on() {
        return;
    }
    CALLS.fetch_add(1, Ordering::Relaxed);
    if v2 {
        CALLS_V2.fetch_add(1, Ordering::Relaxed);
    }
    BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    MACS.fetch_add(macs as u64, Ordering::Relaxed);
    NS.fetch_add(ns, Ordering::Relaxed);
    M_SUM.fetch_add(m as u64, Ordering::Relaxed);
    M_MAX.fetch_max(m as u64, Ordering::Relaxed);
    BUCKET_CALLS[bucket(m)].fetch_add(1, Ordering::Relaxed);
}

/// v2 カーネルの dequant 区間 (タイル生成) の ticks を積む。
#[inline]
pub fn add_deq(ticks: u64) {
    if on() {
        DEQ_TICKS.fetch_add(ticks, Ordering::Relaxed);
    }
}

/// v2 カーネルの fma 区間 (タイルを m トークンに配る) の ticks を積む。
#[inline]
pub fn add_fma(ticks: u64) {
    if on() {
        FMA_TICKS.fetch_add(ticks, Ordering::Relaxed);
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
pub fn line() -> Option<String> {
    if !on() {
        return None;
    }
    let calls = CALLS.load(Ordering::Relaxed);
    if calls == 0 {
        return None;
    }
    let v2 = CALLS_V2.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let macs = MACS.load(Ordering::Relaxed);
    let ns = NS.load(Ordering::Relaxed);
    let m_sum = M_SUM.load(Ordering::Relaxed);
    let m_max = M_MAX.load(Ordering::Relaxed);
    let cpu_ms = ns as f64 / 1e6;
    let gbs = if ns > 0 {
        bytes as f64 / (ns as f64 / 1e9) / 1e9
    } else {
        0.0
    };
    let mut s = format!(
        "[giq-kprof] calls={calls} v2={v2} bytes={:.1} MB macs={:.2} G cpu={cpu_ms:.1} ms \
         (cpu_sum 換算 {:.2} GB/s) | m mean={:.1} max={m_max} [",
        bytes as f64 / 1e6,
        macs as f64 / 1e9,
        gbs,
        m_sum as f64 / calls as f64
    );
    for (i, name) in BUCKET_NAME.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{name}:{}", BUCKET_CALLS[i].load(Ordering::Relaxed)));
    }
    s.push(']');
    let deq = DEQ_TICKS.load(Ordering::Relaxed);
    let fma = FMA_TICKS.load(Ordering::Relaxed);
    if deq + fma > 0 {
        let d = ticks_to_ns(deq) * SAMPLE_EVERY as f64 / 1e6;
        let f = ticks_to_ns(fma) * SAMPLE_EVERY as f64 / 1e6;
        let pct = |v: f64| if d + f > 0.0 { v * 100.0 / (d + f) } else { 0.0 };
        s.push_str(&format!(
            " | dequant {d:.1} ms ({:.0}%) fma {f:.1} ms ({:.0}%)",
            pct(d),
            pct(f)
        ));
    }
    Some(s)
}

/// カウンタを 0 に戻す (`line()` を出したあとに呼ぶ)。
pub fn reset() {
    CALLS.store(0, Ordering::Relaxed);
    CALLS_V2.store(0, Ordering::Relaxed);
    BYTES.store(0, Ordering::Relaxed);
    MACS.store(0, Ordering::Relaxed);
    NS.store(0, Ordering::Relaxed);
    M_SUM.store(0, Ordering::Relaxed);
    M_MAX.store(0, Ordering::Relaxed);
    for b in &BUCKET_CALLS {
        b.store(0, Ordering::Relaxed);
    }
    DEQ_TICKS.store(0, Ordering::Relaxed);
    FMA_TICKS.store(0, Ordering::Relaxed);
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
