//! task 822: prefill の「計算 vs ディスク」の天井を 1 行で出す診断。
//!
//! # なぜ要るか
//!
//! UD-Q4_K_XL の 14,018 トークン prefill は、production env (expert は VRAM に
//! 常駐させない) で **151-158 s** かかり、そのうち
//! `[pf-host] prefill tok=14018 experts_wall=145-150 s` がほぼ全部を占める。
//! ところが同じリクエストの `[exp-ram-probe] io_read_MB` は 31 GB (ディスクから
//! 来た分) で、**page cache が温まっている回は 0.8 GB しか読まないのに壁は
//! 135 s のまま**だった (lab t822c の `pre` COND)。つまりこの壁は SSD でも
//! PCIe でもなく、**CPU の expert GEMV (量子重みの dequant + 内積)** で決まる。
//!
//! その切り分けを毎リクエスト 1 行で読めるようにする。数値経路は触らない
//! (読むだけ)。ゲートは `DENNO_PF_ROOFLINE=1` (既定 off)。
//!
//! # 読み方
//!
//! - `tok`: プロンプトのトークン数 (層 0 でだけ数える)。
//! - `TMAC`: 選択 expert の積和の回数 (トークン × k × 3 行列 × e × ff)。
//! - `GFLOPS` = 2 × TMAC × 10^12 / experts_s。
//! - `disk_MB` / `rchar_MB`: そのリクエストで `/proc/self/io` の
//!   `read_bytes` (実際にストレージから来た量) と `rchar` (read 系システム
//!   コールが返した量。mmap は数えない) が増えた分。
//! - `io_s_ceiling`: `disk_MB` を実測の素の速度 [`ROOFLINE_DISK_MBPS`] で
//!   読んだときに要る秒数。これが `experts_s` の [`IO_BOUND_FRAC`] 倍未満なら
//!   `bound=compute` (ディスクは遊んでいる)、以上なら `bound=io`。

/// `DENNO_PF_ROOFLINE`: 1 行を出すゲート (既定 off)。
pub const ROOFLINE_ENV: &str = "DENNO_PF_ROOFLINE";

/// ディスクの素の速度 (MB/s)。lab t522 の実測 (1 MiB × 8 本で 3.5 GB/s) を採る。
pub const ROOFLINE_DISK_MBPS: f64 = 3500.0;

/// `io_s_ceiling / experts_s` がこれ以上なら `bound=io`。実測 (t822c) は
/// 31 GB 読んだ回でも 0.2 以下だった (compute が 8 割以上)。
pub const IO_BOUND_FRAC: f64 = 0.5;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

static TOK: AtomicU64 = AtomicU64::new(0);
static MAC: AtomicU64 = AtomicU64::new(0);
static NS: AtomicU64 = AtomicU64::new(0);
/// このリクエストの最初の [`note`] で `/proc/self/io` を控えたか。
static SNAP: AtomicBool = AtomicBool::new(false);
static IO_READ0: AtomicU64 = AtomicU64::new(0);
static IO_RCHAR0: AtomicU64 = AtomicU64::new(0);

/// ゲートの解釈 (純関数)。`1` / `true` / `on` / `yes` だけが真 (既定 off)。
pub fn from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("on") | Some("yes")
    )
}

/// テスト用の上書き (-1 = env を見る / 0 = off / 1 = on)。
#[cfg(test)]
static FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// テスト用: ゲートを固定する (`None` で env 読みに戻す)。
#[cfg(test)]
pub fn set_for_test(v: Option<bool>) {
    let raw = match v {
        None => -1i8,
        Some(false) => 0,
        Some(true) => 1,
    };
    FORCE.store(raw, Ordering::Relaxed);
}

/// ゲート (プロセスで 1 回だけ env を読む)。
pub fn on() -> bool {
    #[cfg(test)]
    {
        let f = FORCE.load(Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| from_value(std::env::var(ROOFLINE_ENV).ok().as_deref()))
}

/// `/proc/self/io` の `(read_bytes, rchar)`。読めない環境では `(0, 0)`。
pub fn io_now() -> (u64, u64) {
    let Ok(s) = std::fs::read_to_string("/proc/self/io") else {
        return (0, 0);
    };
    let mut read_bytes = 0u64;
    let mut rchar = 0u64;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            read_bytes = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("rchar:") {
            rchar = v.trim().parse().unwrap_or(0);
        }
    }
    (read_bytes, rchar)
}

/// このリクエストの起点を控える (最初の 1 回だけ)。
fn snapshot_once() {
    if SNAP.swap(true, Ordering::Relaxed) {
        return;
    }
    let (rb, rc) = io_now();
    IO_READ0.store(rb, Ordering::Relaxed);
    IO_RCHAR0.store(rc, Ordering::Relaxed);
}

/// 1 層ぶんの expert 計算を積む (`mac` = 積和の回数、`ns` = その壁時計)。
///
/// トークンは**層 0 のときだけ**数える (それ以外は 1 プロンプトが層数ぶん
/// 二重に数えられる)。`mac` と `ns` は層ごとに足す。
pub fn note(il: usize, nt: usize, mac: u64, ns: u64) {
    if !on() || nt == 0 {
        return;
    }
    snapshot_once();
    if il == 0 {
        TOK.fetch_add(nt as u64, Ordering::Relaxed);
    }
    MAC.fetch_add(mac, Ordering::Relaxed);
    NS.fetch_add(ns, Ordering::Relaxed);
}

/// 積んだ値を 0 に戻す (`take_line` の後に呼ぶ)。
pub fn reset() {
    TOK.store(0, Ordering::Relaxed);
    MAC.store(0, Ordering::Relaxed);
    NS.store(0, Ordering::Relaxed);
    SNAP.store(false, Ordering::Relaxed);
    IO_READ0.store(0, Ordering::Relaxed);
    IO_RCHAR0.store(0, Ordering::Relaxed);
}

/// テスト用: `/proc/self/io` の起点を明示する (`None` で自動へ戻す)。
#[cfg(test)]
pub fn set_io_base_for_test(v: Option<(u64, u64)>) {
    match v {
        Some((rb, rc)) => {
            SNAP.store(true, Ordering::Relaxed);
            IO_READ0.store(rb, Ordering::Relaxed);
            IO_RCHAR0.store(rc, Ordering::Relaxed);
        }
        None => {
            SNAP.store(false, Ordering::Relaxed);
        }
    }
}

/// ディスクの分を天井の速度で読んだときの秒数 (純関数)。
pub fn io_seconds(disk_mb: f64) -> f64 {
    if disk_mb <= 0.0 {
        return 0.0;
    }
    disk_mb / ROOFLINE_DISK_MBPS
}

/// `bound=compute` か `bound=io` か (純関数)。
pub fn bound_of(disk_mb: f64, experts_s: f64) -> &'static str {
    if experts_s <= 0.0 {
        return "compute";
    }
    if io_seconds(disk_mb) / experts_s >= IO_BOUND_FRAC {
        "io"
    } else {
        "compute"
    }
}

/// 1 行を作る (積みが 0 なら `None`)。呼んだ側が `reset` する。
pub fn take_line(tag: &str) -> Option<String> {
    if !on() {
        return None;
    }
    let tok = TOK.load(Ordering::Relaxed);
    if tok == 0 {
        return None;
    }
    let mac = MAC.load(Ordering::Relaxed);
    let expert_s = NS.load(Ordering::Relaxed) as f64 / 1e9;
    let (rb, rc) = io_now();
    let disk_mb = rb.saturating_sub(IO_READ0.load(Ordering::Relaxed)) as f64 / 1e6;
    let rchar_mb = rc.saturating_sub(IO_RCHAR0.load(Ordering::Relaxed)) as f64 / 1e6;
    let gflops = if expert_s > 0.0 {
        2.0 * mac as f64 / 1e9 / expert_s
    } else {
        0.0
    };
    let io_s = io_seconds(disk_mb);
    Some(format!(
        "[pf-roofline] {tag} tok={tok} experts_s={expert_s:.2} TMAC={:.3} GFLOPS={gflops:.0} \
         disk_MB={disk_mb:.1} rchar_MB={rchar_mb:.1} io_s_ceiling={io_s:.2} bound={}",
        mac as f64 / 1e12,
        bound_of(disk_mb, expert_s)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_is_off_by_default() {
        assert!(!from_value(None), "既定が on");
        assert!(!from_value(Some("")));
        for v in ["1", "true", "on", "yes", " ON "] {
            assert!(from_value(Some(v)), "{v} が on にならない");
        }
        for v in ["0", "off", "false", "no", "2"] {
            assert!(!from_value(Some(v)), "{v} が on になった");
        }
    }

    #[test]
    fn bound_switches_at_the_measured_ceiling() {
        // t822c の実測: 31 GB を 145 s で読んだ = 天井 3.5 GB/s なら 8.9 s = 6%。
        assert_eq!(bound_of(31235.0, 145.0), "compute");
        // 同じ 145 s でディスクが 300 GB 来ていれば io (天井 3.5 GB/s で 86 s)。
        assert_eq!(bound_of(300_000.0, 145.0), "io");
        // 0 で割らない。
        assert_eq!(bound_of(100.0, 0.0), "compute");
        assert_eq!(io_seconds(0.0), 0.0);
        assert!((io_seconds(3500.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn note_and_line_add_up_and_reset() {
        // ゲート off では note が何もしない = 行も出ない。
        set_for_test(Some(false));
        note(0, 10, 1_000, 1_000_000);
        assert!(take_line("t").is_none(), "off なのに行が出た");
        assert_eq!(TOK.load(Ordering::Relaxed), 0, "off なのに積んだ");
        // ゲート on の算術 (層 0 のときだけトークンを数える)。
        set_for_test(Some(true));
        reset();
        set_io_base_for_test(Some((1_000, 2_000)));
        note(0, 4, 100 * 1_000_000_000, 500_000_000);
        note(1, 6, 200 * 1_000_000_000, 500_000_000);
        let l = take_line("prefill").expect("行が出ない");
        assert!(l.starts_with("[pf-roofline] prefill tok=4 experts_s=1.00"), "{l}");
        assert!(l.contains("TMAC=0.300"), "{l}");
        assert!(l.contains("GFLOPS=600"), "{l}");
        assert!(l.contains("bound=compute"), "{l}");
        reset();
        assert!(take_line("prefill").is_none(), "reset 後に行が出た");
        set_for_test(None);
        set_io_base_for_test(None);
    }
}
