//! task 158: decode で VRAM に載っていない expert (ミス) を **GPU で計算する**。
//!
//! # なぜ要るか
//!
//! task 155 の時点の decode は「VRAM に載っている expert だけ GPU・ミスは CPU」だった
//! (task 150 の重ね打ち)。実測 (lab t155a/t155c、14,139 トークン、ms/tok) では
//! experts の host 11.02 / wait 5.98 / gpu_busy 4.50 / **gpu_idle 6.59**。
//! ヒット率 0.95〜0.99 なので 1 トークンに 0〜3 本/層 = 約 24 本がミスし、1 本あたり
//! CPU で約 350 us (task 153 の実測) かかっていた。GPU はその間 6.6 ms 遊んでいる。
//! 1 本の重みは約 2 MB で、PCIe なら 0.1 ms 程度。GPU で計算した方がずっと安い。
//!
//! # 2 つの変種 (lab で比べて良い方を残す)
//!
//! - [`MODE_SCRATCH`] (=1): **scratch だけ**。pinned の bounce buffer へ写してから
//!   専用のデバイス scratch へ非同期 H2D し、VRAM の LRU には触らない。
//! - [`MODE_CACHE`] (=2): **キャッシュへ挿入**。`expert_vram::acquire` の通常方針
//!   (LRU・予算・追い出し) で 1 本ずつ載せ、載った重みで計算する。
//!
//! どちらも計算は `kernels_exp_iq` の層一括 GEMV (ヒットと同じカーネル) なので、
//! **CPU 経路とは丸め順が違う** (warp 縮約 vs 逐次加算)。md5 が DIFF になり得るが、
//! これは「GPU expert カーネルの丸めが CPU と違う」ことによるもので、値の誤差は
//! task 144 の許容 (絶対 1e-4 + 相対 1e-5) の範囲。
//!
//! # 失敗したとき
//!
//! scratch の確保に失敗する / カーネルが無い / 型が未対応 / ミスが [`MAX_MISS`] を
//! 超える、のいずれでも `None` を返して従来経路 (CPU でミスを計算 + 重ね打ち) へ
//! 落ちる (fail closed)。ゲート off では 1 回の atomic load だけで従来と同一。
//!
//! # ラボ実測 (task 158 / lab t158a、swift15-abl IQ3_S、14T、depth8k、commit edbf04e3a)
//!
//! base = `DENNO_DEC_MISS_UP=0 DENNO_MIX_SYNC_EV=0` (この 2 つ以外は本番 env +
//! `DENNO_EXPERT_IQ_GPU=1 DENNO_EXPERT_VRAM_MB=22000 DENNO_EXPERT_GPU_MIN_HIT=0
//! DENNO_GPU_WEIGHT_MB=12000`)。
//!
//! | cond | decode ms/tok 短 / 3,954 / 14,139 | md5 |
//! | --- | --- | --- |
//! | base | 42 / 40 / 43 | 基準 |
//! | scratch (mode 1) | 50 / 44 / 45 (**+19.0 / +12.9 / +4.8%**) | SAME |
//! | cache (mode 2) | 39 / 40 / 43 (-7.1 / +1.2 / 0.0%) | DIFF (丸め差) |
//!
//! `VERDICT scratch vs base OK decode +19.0% prefill +0.0% md5 SAME errors 0`
//! `VERDICT cache vs base OK decode -7.1% prefill -12.6% md5 DIFF errors 0`
//!
//! ## なぜ負けたか (この機構の限界)
//!
//! 1. **H2D の実測が遅い**: `[dec-miss-probe] mb=2 reps=8 h2d_ms=1.168 gbs=14.36`
//!    (pinned bounce → デバイス scratch、2 MB)。bounce のホスト写しも
//!    `copy_gbs=15〜20`。つまり 1 本 2.05 MB を動かすのに **約 0.14 ms (ホスト写し)
//!    + 0.14 ms (H2D)** かかる。
//! 2. **CPU 経路がその床より速い**: 14,139 の base はミス 27 本/token で
//!    experts の cpu 3.92 ms/tok → **1 本あたり約 145 us** (task 153 の 350 us は
//!    この環境では再現しない)。2 MB の移動 (≒146 us) とほぼ同じで、勝ち目がない。
//! 3. mode 1 は 14,139 で `[dec-miss] misses=3439/128tok bytes=7029MB
//!    copy=372ms` = **2.9 ms/token のホスト写し**を足す。ミスの CPU 計算 (3.9 ms,
//!    ヒットの裏に隠れる) を消しても差し引き +2.4 ms/token になる。
//! 4. mode 2 (キャッシュ挿入) は 1 本 0.27 ms の同期 H2D なので CPU 計算と
//!    ほぼ同じ。短い文脈ではヒット率が 0.85 → 1.000 まで早く上がるため -7.1% に
//!    なるが、14,139 では base が既に 0.972 なので 0.0%。
//!
//! したがって**既定 off**。転送を安くするには `cudaHostRegister` で mmap の範囲を
//! 直接 pin する (bounce のホスト写しを消す) か、GPU に host-mapped の重みを直接
//! 読ませるしかなく、どちらも 2 MB/本 の PCIe 移動という床は残る。次の一手は
//! attention / mix のカーネル (REPORT_P158.md の項目 3)。


/// `DENNO_DEC_MISS_UP`: decode のミス expert を GPU で計算する。**既定 off**。
///
/// - `0` / `off` / `false` — 従来 (ミスは CPU、ヒットの裏で重ね打ち)
/// - `1` / `on` / `true` — scratch 経路 (VRAM の LRU は触らない)
/// - `2` — キャッシュ挿入経路 (`expert_vram::acquire` の通常方針)
pub const ENV: &str = "DENNO_DEC_MISS_UP";

/// `DENNO_DEC_MISS_STATS=1`: リクエスト終端に 1 行 (`[dec-miss]`) を出す。
pub const STATS_ENV: &str = "DENNO_DEC_MISS_STATS";

/// `DENNO_DEC_MISS_PROBE=1`: 起動後に 1 回だけ H2D の実効レートを測って 1 行出す
/// (`[dec-miss-probe]`)。ホットパスには入らない。
pub const PROBE_ENV: &str = "DENNO_DEC_MISS_PROBE";

/// 1 層で GPU に上げるミスの上限 (scratch の本数)。
///
/// 実測のミスは 0〜3 本/層なので 8 本あれば足りる。超えた層は従来の CPU 経路へ
/// 落とす (scratch を無限に大きくしない)。
pub const MAX_MISS: usize = 8;

/// ゲート off。
pub const MODE_OFF: u8 = 0;
/// scratch だけ (LRU は触らない)。
pub const MODE_SCRATCH: u8 = 1;
/// キャッシュへ挿入してから計算する。
pub const MODE_CACHE: u8 = 2;

/// [`ENV`] の値を解釈する (純関数・テスト用)。
///
/// 未知の値は off (既定)。`1` 系は 1、`2` だけ 2。
pub fn mode_from_value(v: Option<&str>) -> u8 {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => match s.as_str() {
            "2" => MODE_CACHE,
            "1" | "on" | "true" | "yes" => MODE_SCRATCH,
            _ => MODE_OFF,
        },
        None => MODE_OFF,
    }
}

/// ゲートの値 (プロセスで 1 回だけ env を読む)。
pub fn mode() -> u8 {
    static M: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *M.get_or_init(|| mode_from_value(std::env::var(ENV).ok().as_deref()))
}

/// ミスを GPU で計算するか (off のときのコストは atomic load 1 回)。
pub fn enabled() -> bool {
    mode() != MODE_OFF
}

/// 診断行 (`[dec-miss]`) を出すか。
pub fn stats_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::model::expert_vram::flag_on(std::env::var(STATS_ENV).ok().as_deref()))
}

/// H2D レートのプローブを走らせるか。
pub fn probe_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::model::expert_vram::flag_on(std::env::var(PROBE_ENV).ok().as_deref()))
}

// ---------------------------------------------------------------------------
// 1 リクエストぶんの積算 (診断)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};

static LAYERS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static BYTES: AtomicU64 = AtomicU64::new(0);
/// ホストの bounce 写し (mmap → pinned) の合計 ns。
static COPY_NS: AtomicU64 = AtomicU64::new(0);
/// ミス分の GPU カーネル待ち (層の同期のうちミスを含む層の wall) の合計 ns。
static GPU_NS: AtomicU64 = AtomicU64::new(0);
/// ミスを含む層の数 (GPU_NS の分母)。
static MISS_LAYERS: AtomicU64 = AtomicU64::new(0);

/// 1 層ぶん積む。`bytes` は bounce した重みの合計バイト数。
pub fn note_layer(misses: u32, bytes: u64) {
    LAYERS.fetch_add(1, Ordering::Relaxed);
    MISSES.fetch_add(misses as u64, Ordering::Relaxed);
    BYTES.fetch_add(bytes, Ordering::Relaxed);
}

/// ホスト写しの時間を積む。
pub fn note_copy_ns(ns: u64) {
    COPY_NS.fetch_add(ns, Ordering::Relaxed);
}

/// ミスを含む層の wall を積む。
pub fn note_gpu_ns(ns: u64) {
    GPU_NS.fetch_add(ns, Ordering::Relaxed);
    MISS_LAYERS.fetch_add(1, Ordering::Relaxed);
}

/// 1 リクエストぶんの 1 行。ゲート off / 実績 0 なら `None`。
pub fn stats_line() -> Option<String> {
    if !enabled() {
        return None;
    }
    let layers = LAYERS.load(Ordering::Relaxed);
    if layers == 0 {
        return None;
    }
    let misses = MISSES.load(Ordering::Relaxed);
    let bytes = BYTES.load(Ordering::Relaxed);
    let copy_ms = COPY_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let miss_layers = MISS_LAYERS.load(Ordering::Relaxed).max(1);
    let gpu_ms = GPU_NS.load(Ordering::Relaxed) as f64 / 1e6;
    let copy_gbs = if copy_ms > 0.0 {
        bytes as f64 / (copy_ms * 1e6)
    } else {
        0.0
    };
    Some(format!(
        "[dec-miss] mode={} layers={layers} miss_layers={} misses={misses} bytes={:.1}MB \
         copy={copy_ms:.2}ms copy_gbs={copy_gbs:.2} miss_layer_wall={:.2}ms us_per_layer={:.1}",
        mode(),
        MISS_LAYERS.load(Ordering::Relaxed),
        bytes as f64 / 1e6,
        gpu_ms,
        gpu_ms * 1e3 / miss_layers as f64
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    for c in [
        &LAYERS,
        &MISSES,
        &BYTES,
        &COPY_NS,
        &GPU_NS,
        &MISS_LAYERS,
    ] {
        c.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_values_are_explicit() {
        assert_eq!(mode_from_value(None), MODE_OFF);
        assert_eq!(mode_from_value(Some("")), MODE_OFF);
        assert_eq!(mode_from_value(Some("0")), MODE_OFF);
        assert_eq!(mode_from_value(Some("off")), MODE_OFF);
        assert_eq!(mode_from_value(Some("false")), MODE_OFF);
        assert_eq!(mode_from_value(Some("1")), MODE_SCRATCH);
        assert_eq!(mode_from_value(Some("on")), MODE_SCRATCH);
        assert_eq!(mode_from_value(Some(" TRUE ")), MODE_SCRATCH);
        // 2 だけが cache 挿入。3 以上は未知 = off (勝手に強くしない)。
        assert_eq!(mode_from_value(Some("2")), MODE_CACHE);
        assert_eq!(mode_from_value(Some("3")), MODE_OFF);
    }

    #[test]
    fn stats_line_is_silent_when_off_or_idle() {
        // 既定 (env 無し) は off。積算が 0 のときも行を出さない。
        stats_reset();
        if !enabled() {
            assert!(stats_line().is_none());
        }
    }

    #[test]
    fn note_and_reset_roundtrip() {
        stats_reset();
        note_layer(2, 4_100_000);
        note_copy_ns(1_000_000);
        note_gpu_ns(2_000_000);
        assert_eq!(LAYERS.load(Ordering::Relaxed), 1);
        assert_eq!(MISSES.load(Ordering::Relaxed), 2);
        assert_eq!(BYTES.load(Ordering::Relaxed), 4_100_000);
        assert_eq!(MISS_LAYERS.load(Ordering::Relaxed), 1);
        stats_reset();
        assert_eq!(LAYERS.load(Ordering::Relaxed), 0);
        assert_eq!(COPY_NS.load(Ordering::Relaxed), 0);
        assert_eq!(GPU_NS.load(Ordering::Relaxed), 0);
    }
}
