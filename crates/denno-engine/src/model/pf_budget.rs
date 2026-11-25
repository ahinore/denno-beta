//! task 814: メモリ空きに応じた prefill の資源配分 (`DENNO_PF_MEMADAPT`)。
//!
//! # なぜ要るか
//!
//! UD-Q4_K_XL (111 GB) は 64 GiB ホストの RAM より大きい。長文 prefill の
//! 資源は全部 **静的な既定** で決まっている:
//!
//! - pread の本数 16 本 x 上げスレッド (`DENNO_PF_PREAD_THREADS`)
//! - 先読みの並列度 32 本 (`DENNO_PF_RA_THREADS`)
//! - 先読みの置き場 512 MiB (`DENNO_PF_RC_CACHE_MB`、ヒープ = 匿名メモリ)
//! - 層優先 prefill のトークンタイル 5,120 (`DENNO_PF_LM_TILE`)
//! - prefill の作業域プール 2,688 MiB (`DENNO_PF_POOL_MB`、匿名メモリ)
//! - pread 1 回の長さ 1 MiB (広い塊のとき。`DENNO_PF_PREAD_KB`)
//!
//! 64 GiB のホストで空きが 30 GiB あるときはこれで良いが、他のプロセスが
//! メモリを食っていたり 32 GiB のホストだったりすると、**匿名メモリの確保が
//! swap を押し出す** (プール 2.7 GiB + 置き場 0.5 GiB + pread/バウンスの
//! thread-local が同時に載る)。逆に 128 GiB のホストでは、余っている RAM を
//! 使わずに同じ値で走っている。
//!
//! このモジュールは `/proc/meminfo` の `MemAvailable` / `SwapFree` から
//! **純関数** [`plan_for`] で資源配分を決める。ゲートは
//! [`MEMADAPT_ENV`] (既定 on、明示的な偽値で従来の静的既定へ戻る)。
//!
//! # 不変条件 (数値と決定性)
//!
//! - ここが変えるのは **本数・バイト数・タイル幅だけ**。演算の式も expert の
//!   選択も変えない (タイル幅はチャンク境界を動かすので、`DENNO_PF_STREAM_MIN_TOK`
//!   を跨ぐかどうかで縮約順が変わりうる。丸め順の差はタスクの規則で許容)。
//! - **空きが十分なら [`base_tuning`] をそのまま返す** (恒等)。64 GiB ホストの
//!   平常時は今の既定と 1 バイトも変わらないので、lab の A/B は「圧迫時だけ
//!   効く」ことを確かめる形になる。
//! - 明示の `DENNO_PF_*` は常にこの計算に勝つ (各 getter 側で判定)。
//! - `MemAvailable` は page cache の再利用可能分を含むので、Q4 の mmap が
//!   ページキャッシュを埋めても「空き」は減らない。ここを見るのは正しい。
//!
//! # 圧迫の段
//!
//! | 段 | 条件 (MiB) | 意味 |
//! | --- | --- | --- |
//! | Healthy | `avail >= max(12 GiB, total*18%)` | 平常。恒等 |
//! | Tight | それ未満かつ `avail >= max(6 GiB, total*8%)` | 半分に絞る |
//! | Critical | それ未満、または swap の空きが 25% 未満で `avail < total*25%` | 最小構成 |
//!
//! さらに **匿名メモリの上限** `anon_cap = avail * 60%` を掛け、
//! プールは `anon_cap/4`、置き場は `anon_cap/8` を超えないようにする
//! (残り 40% はカーネルの page cache / スタック / 断片化の余裕)。

use std::sync::Mutex;

/// `DENNO_PF_MEMADAPT`: メモリ空きに応じた配分を使うか。既定 on。
pub const MEMADAPT_ENV: &str = "DENNO_PF_MEMADAPT";

/// [`MEMADAPT_ENV`] の既定 (on)。
pub const DEFAULT_MEMADAPT: bool = true;

/// 1 MiB (KiB 単位の meminfo を MiB へ落とす)。
const KIB_PER_MIB: u64 = 1024;

/// 恒等を返す下限 (total の 18% と 12 GiB の大きい方)。
const HEALTHY_MIN_MIB: u64 = 12 * 1024;
const HEALTHY_FRAC_PCT: u64 = 18;

/// Tight の下限 (total の 8% と 6 GiB の大きい方)。
const TIGHT_MIN_MIB: u64 = 6 * 1024;
const TIGHT_FRAC_PCT: u64 = 8;

/// 128 GiB 級のホストで余った RAM を使う段の閾値 (total >= 96 GiB)。
const LARGE_TOTAL_MIB: u64 = 96 * 1024;

/// 匿名メモリに回してよい `avail` の割合 (%)。
const ANON_PCT: u64 = 60;

/// プール / 置き場が `anon_cap` の何分の 1 までか。
const POOL_DIV: u64 = 4;
const RC_DIV: u64 = 8;

/// 下限 (これ以下へは絞らない。絞りすぎると I/O が直列化して逆効果)。
const MIN_PREAD_THREADS: usize = 4;
const MIN_RA_THREADS: usize = 8;
const MIN_RC_CACHE_MB: usize = 64;
const MIN_POOL_MB: usize = 512;
const MIN_LM_TILE: usize = 1024;

/// メモリの段。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pressure {
    /// 平常。恒等 (base をそのまま返す)。
    Healthy,
    /// 余裕が少ない。半分に絞る。
    Tight,
    /// 危険。swap を押し出しうる。最小構成。
    Critical,
}

impl Pressure {
    /// `[pf-budget]` の 1 語表記。
    pub fn as_str(self) -> &'static str {
        match self {
            Pressure::Healthy => "healthy",
            Pressure::Tight => "tight",
            Pressure::Critical => "critical",
        }
    }
}

/// `/proc/meminfo` の必要な項目 (KiB)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemFacts {
    pub total_kb: u64,
    pub avail_kb: u64,
    pub swap_total_kb: u64,
    pub swap_free_kb: u64,
}

/// 資源配分 (本数・MiB・トークン)。
///
/// `pread_kb_cap` は「1 回の pread の長さの上限 (KiB)」で、0 は上限なし
/// (既存の [`super::exp_pf::pread_kb_default`] をそのまま使う)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub pread_threads: usize,
    pub ra_threads: usize,
    pub rc_cache_mb: usize,
    pub lm_tile: usize,
    pub pool_mb: usize,
    pub pread_kb_cap: usize,
}

/// 判定結果 (ログ用に段と空きを持つ)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub pressure: Pressure,
    pub avail_mib: u64,
    pub total_mib: u64,
    pub tuning: Tuning,
}

/// 今の静的な既定 (恒等の基準)。
pub fn base_tuning() -> Tuning {
    Tuning {
        pread_threads: super::exp_pf::DEFAULT_PREAD_THREADS,
        ra_threads: super::exp_pf::DEFAULT_RC_THREADS,
        rc_cache_mb: super::exp_pf::DEFAULT_RC_CACHE_MB,
        lm_tile: super::exp_pf::DEFAULT_LM_TILE,
        pool_mb: super::pf_pool::DEFAULT_POOL_MB,
        pread_kb_cap: 0,
    }
}

/// `DENNO_PF_MEMADAPT` の解釈 (純関数)。明示的な偽値だけ off。
pub fn memadapt_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim).filter(|s| !s.is_empty()).map(|s| s.to_ascii_lowercase()) {
        None => DEFAULT_MEMADAPT,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no"),
    }
}

/// `/proc/meminfo` の本文を読む (純関数)。`MemTotal` と `MemAvailable` が
/// 無い / 解析できないときは `None`。
pub fn parse_meminfo(s: &str) -> Option<MemFacts> {
    let mut total = None;
    let mut avail = None;
    let mut swap_total = 0u64;
    let mut swap_free = 0u64;
    for line in s.lines() {
        let mut it = line.split_whitespace();
        let Some(key) = it.next() else { continue };
        let Some(val) = it.next().and_then(|v| v.parse::<u64>().ok()) else {
            continue;
        };
        match key {
            "MemTotal:" => total = Some(val),
            "MemAvailable:" => avail = Some(val),
            "SwapTotal:" => swap_total = val,
            "SwapFree:" => swap_free = val,
            _ => {}
        }
    }
    match (total, avail) {
        (Some(t), Some(a)) if t > 0 => Some(MemFacts {
            total_kb: t,
            avail_kb: a.min(t),
            swap_total_kb: swap_total,
            swap_free_kb: swap_free.min(swap_total),
        }),
        _ => None,
    }
}

/// `/proc/meminfo` を読む。無い環境 (macOS 等) では `None`。
pub fn read_meminfo() -> Option<MemFacts> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_meminfo(&s)
}

/// 段の判定 (純関数)。
pub fn pressure_of(m: MemFacts) -> Pressure {
    let total_mib = m.total_kb / KIB_PER_MIB;
    let avail_mib = m.avail_kb / KIB_PER_MIB;
    let healthy = HEALTHY_MIN_MIB.max(total_mib * HEALTHY_FRAC_PCT / 100);
    let tight = TIGHT_MIN_MIB.max(total_mib * TIGHT_FRAC_PCT / 100);
    // swap があり、空きが 25% 未満で、RAM の空きも 25% 未満なら危険とみなす
    // (swap が無いホストは「押し出し先が無い」だけなので段を上げない)。
    let swap_low = m.swap_total_kb > 0 && m.swap_free_kb * 4 < m.swap_total_kb;
    if avail_mib < tight || (swap_low && avail_mib < total_mib * 25 / 100) {
        Pressure::Critical
    } else if avail_mib < healthy {
        Pressure::Tight
    } else {
        Pressure::Healthy
    }
}

/// 段ごとの目標値 (純関数、予算の上限を掛ける前)。
fn target_for(p: Pressure, total_mib: u64, base: Tuning) -> Tuning {
    match p {
        // 128 GiB 級で半分以上空いている: 匿名メモリを増やして置き場と作業域を
        // 太くする。pread の本数は lab t606b で 32 が 16 に負けたので上げない。
        Pressure::Healthy if total_mib >= LARGE_TOTAL_MIB => Tuning {
            pread_threads: base.pread_threads,
            ra_threads: base.ra_threads,
            rc_cache_mb: (base.rc_cache_mb * 2).min(2048),
            lm_tile: (base.lm_tile * 2).min(16384),
            pool_mb: (base.pool_mb * 3 / 2).min(4096),
            pread_kb_cap: 0,
        },
        Pressure::Healthy => base,
        Pressure::Tight => Tuning {
            pread_threads: (base.pread_threads / 2).max(MIN_PREAD_THREADS),
            ra_threads: (base.ra_threads / 2).max(MIN_RA_THREADS),
            rc_cache_mb: (base.rc_cache_mb / 2).max(MIN_RC_CACHE_MB),
            lm_tile: (base.lm_tile * 4 / 5).max(MIN_LM_TILE),
            pool_mb: (base.pool_mb / 2).max(MIN_POOL_MB),
            pread_kb_cap: 512,
        },
        Pressure::Critical => Tuning {
            pread_threads: MIN_PREAD_THREADS,
            ra_threads: MIN_RA_THREADS,
            rc_cache_mb: MIN_RC_CACHE_MB,
            lm_tile: (base.lm_tile / 2).max(MIN_LM_TILE),
            pool_mb: (base.pool_mb / 4).max(MIN_POOL_MB),
            pread_kb_cap: 256,
        },
    }
}

/// 資源配分を決める (純関数)。空きが十分なら `base` をそのまま返す。
pub fn plan_for(m: MemFacts, base: Tuning) -> Plan {
    let total_mib = m.total_kb / KIB_PER_MIB;
    let avail_mib = m.avail_kb / KIB_PER_MIB;
    let pressure = pressure_of(m);
    let mut t = target_for(pressure, total_mib, base);

    // 匿名メモリの上限。残り 40% は page cache / 断片化の余裕として残す。
    let anon_cap = avail_mib * ANON_PCT / 100;
    let pool_cap = (anon_cap / POOL_DIV).max(MIN_POOL_MB as u64);
    let rc_cap = (anon_cap / RC_DIV).max(MIN_RC_CACHE_MB as u64);
    t.pool_mb = t.pool_mb.min(pool_cap as usize);
    t.rc_cache_mb = t.rc_cache_mb.min(rc_cap as usize);

    // 床と total の範囲へ丸める (異常値でも暴れない)。
    t.pread_threads = t.pread_threads.clamp(MIN_PREAD_THREADS, 64);
    t.ra_threads = t.ra_threads.clamp(MIN_RA_THREADS, 32);
    t.rc_cache_mb = t.rc_cache_mb.clamp(MIN_RC_CACHE_MB, 2048);
    t.lm_tile = t.lm_tile.max(MIN_LM_TILE);
    t.pool_mb = t.pool_mb.clamp(MIN_POOL_MB, 8192);

    Plan {
        pressure,
        avail_mib,
        total_mib,
        tuning: t,
    }
}

static STATE: Mutex<Option<(std::time::Instant, Plan)>> = Mutex::new(None);
static CHANGES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static LOGGED: Mutex<Option<Tuning>> = Mutex::new(None);

#[cfg(test)]
static FORCE_PLAN: Mutex<Option<Plan>> = Mutex::new(None);

/// テスト用: [`plan`] を env と `meminfo` に依らず固定する (`None` で解除)。
///
/// `OnceLock` で凍結している getter (`pread_bytes` など) はこの上書きを
/// 取り込まないため、ここで差し替えて確かめるのは **毎回引く getter**
/// (`pread_threads` / `rc_threads` / `rc_cache_bytes` / `lm_tile` /
/// `pf_pool_bytes`) だけ。配線が外れたらこのテストが赤になる。
#[cfg(test)]
pub(crate) fn set_force_plan_for_test(p: Option<Plan>) {
    *FORCE_PLAN.lock().unwrap_or_else(|e| e.into_inner()) = p;
}

/// 計画を作り直す最短間隔。prefill のループから何度も呼ばれるので、
/// `/proc/meminfo` の読みは 1 秒に 1 回までにする。
pub const REFRESH_MS: u128 = 1000;

/// ゲート (`DENNO_PF_MEMADAPT`)。テストは env を触らずに固定できる。
pub fn gate_on() -> bool {
    #[cfg(test)]
    {
        let f = FORCE_GATE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| memadapt_from_value(std::env::var(MEMADAPT_ENV).ok().as_deref()))
}

#[cfg(test)]
static FORCE_GATE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// テスト用: ゲートを env 無しで固定する (`None` で解除)。
#[cfg(test)]
pub(crate) fn set_force_gate_for_test(v: Option<bool>) {
    FORCE_GATE.store(
        match v {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// いまの計画。ゲート off / meminfo が読めないときは恒等 (Healthy + base)。
pub fn plan() -> Plan {
    #[cfg(test)]
    {
        let f = *FORCE_PLAN.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = f {
            return p;
        }
    }
    let base = base_tuning();
    if !gate_on() {
        return Plan {
            pressure: Pressure::Healthy,
            avail_mib: 0,
            total_mib: 0,
            tuning: base,
        };
    }
    let now = std::time::Instant::now();
    {
        let g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, p)) = *g {
            if now.saturating_duration_since(at).as_millis() < REFRESH_MS {
                return p;
            }
        }
    }
    let p = match read_meminfo() {
        Some(m) => plan_for(m, base),
        None => Plan {
            pressure: Pressure::Healthy,
            avail_mib: 0,
            total_mib: 0,
            tuning: base,
        },
    };
    {
        let mut g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        *g = Some((now, p));
    }
    log_if_changed(&p);
    p
}

/// いまの配分 (ゲッタ用の薄い包み)。
pub fn tuning() -> Tuning {
    plan().tuning
}

/// 配分が変わった回数 (テスト / 診断)。
pub fn plan_changes() -> u64 {
    CHANGES.load(std::sync::atomic::Ordering::Relaxed)
}

/// 段か配分が変わったときだけ 1 行出す (`[pf-budget]`)。
fn log_if_changed(p: &Plan) {
    let mut g = LOGGED.lock().unwrap_or_else(|e| e.into_inner());
    if *g == Some(p.tuning) {
        return;
    }
    *g = Some(p.tuning);
    CHANGES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let t = p.tuning;
    eprintln!(
        "[pf-budget] {} total={} MiB avail={} MiB pread={} ra={} rc={} MiB tile={} pool={} MiB pread_kb_cap={}",
        p.pressure.as_str(),
        p.total_mib,
        p.avail_mib,
        t.pread_threads,
        t.ra_threads,
        t.rc_cache_mb,
        t.lm_tile,
        t.pool_mb,
        t.pread_kb_cap
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024;

    fn facts(total_gib: u64, avail_gib: u64) -> MemFacts {
        MemFacts {
            total_kb: total_gib * GIB,
            avail_kb: avail_gib * GIB,
            swap_total_kb: 8 * GIB,
            swap_free_kb: 8 * GIB,
        }
    }

    #[test]
    fn t814_parse_meminfo_reads_the_four_fields() {
        let txt = "MemTotal:       65863304 kB\nMemFree:  1000 kB\nMemAvailable:   31457280 kB\n\
                   SwapTotal:       8388604 kB\nSwapFree:        7340032 kB\n";
        let m = parse_meminfo(txt).expect("読める");
        assert_eq!(m.total_kb, 65_863_304);
        assert_eq!(m.avail_kb, 31_457_280);
        assert_eq!(m.swap_total_kb, 8_388_604);
        assert_eq!(m.swap_free_kb, 7_340_032);
        // MemAvailable が無いものは捨てる。
        assert_eq!(parse_meminfo("MemTotal: 1000 kB\n"), None);
        assert_eq!(parse_meminfo(""), None);
        // avail > total は total へ丸める。
        let m = parse_meminfo("MemTotal: 1000 kB\nMemAvailable: 2000 kB\n").unwrap();
        assert_eq!(m.avail_kb, 1000);
    }

    #[test]
    fn t814_gate_semantics_are_explicit_false_only() {
        assert!(memadapt_from_value(None), "未設定は on");
        assert!(memadapt_from_value(Some("")), "空も on");
        assert!(memadapt_from_value(Some("1")));
        assert!(!memadapt_from_value(Some("0")));
        assert!(!memadapt_from_value(Some("off")));
        assert!(!memadapt_from_value(Some("false")));
        assert!(!memadapt_from_value(Some("NO")));
        // 誤入力は on (安全側 = 動的配分に任せる)。
        assert!(memadapt_from_value(Some("xyz")));
    }

    #[test]
    fn t814_64gib_healthy_is_identity() {
        let b = base_tuning();
        for avail in [20u64, 30, 40, 50, 58] {
            let p = plan_for(facts(64, avail), b);
            assert_eq!(p.pressure, Pressure::Healthy, "avail={avail} GiB");
            assert_eq!(p.tuning, b, "空きが十分なら既定と 1 バイトも変わらない");
        }
    }

    #[test]
    fn t814_64gib_tight_and_critical_shrink() {
        let b = base_tuning();
        // 18% of 64 GiB = 11.5 GiB なので 8 GiB は Tight、6 GiB 未満は Critical。
        let t = plan_for(facts(64, 8), b);
        assert_eq!(t.pressure, Pressure::Tight);
        assert!(t.tuning.pread_threads < b.pread_threads);
        assert!(t.tuning.ra_threads < b.ra_threads);
        assert!(t.tuning.pool_mb < b.pool_mb);
        assert!(t.tuning.rc_cache_mb < b.rc_cache_mb);
        assert!(t.tuning.pread_kb_cap > 0, "pread を細かくする");

        let c = plan_for(facts(64, 5), b);
        assert_eq!(c.pressure, Pressure::Critical);
        assert_eq!(c.tuning.pread_threads, MIN_PREAD_THREADS);
        assert_eq!(c.tuning.ra_threads, MIN_RA_THREADS);
        assert!(c.tuning.pool_mb <= t.tuning.pool_mb);
        assert!(c.tuning.lm_tile <= t.tuning.lm_tile);
    }

    #[test]
    fn t814_32gib_hosts() {
        let b = base_tuning();
        // 20 GiB 空き (18% = 5.8 GiB を上回る) は恒等。
        assert_eq!(plan_for(facts(32, 20), b).tuning, b);
        // 4 GiB 空きは Critical (8% = 2.6 GiB は上回るが 6 GiB の床に届かない)。
        let c = plan_for(facts(32, 4), b);
        assert_eq!(c.pressure, Pressure::Critical);
        // 匿名の上限が効く: avail 4 GiB -> anon 2.4 GiB -> pool <= 614 MiB。
        assert!(c.tuning.pool_mb <= 614, "pool={}", c.tuning.pool_mb);
        assert!(c.tuning.pool_mb >= MIN_POOL_MB);
        assert!(c.tuning.rc_cache_mb >= MIN_RC_CACHE_MB);
    }

    #[test]
    fn t814_128gib_host_uses_the_extra_ram() {
        let b = base_tuning();
        let p = plan_for(facts(128, 100), b);
        assert_eq!(p.pressure, Pressure::Healthy);
        assert!(p.tuning.rc_cache_mb > b.rc_cache_mb);
        assert!(p.tuning.pool_mb > b.pool_mb);
        assert!(p.tuning.lm_tile > b.lm_tile);
        assert_eq!(p.tuning.pread_threads, b.pread_threads, "32 本は負けるので上げない");
        // 128 GiB でも空きが 20 GiB なら Tight (床 12 GiB を超えるが 18% に届かない)。
        let t = plan_for(facts(128, 20), b);
        assert_eq!(t.pressure, Pressure::Tight);
        assert!(t.tuning.pool_mb <= b.pool_mb);
    }

    #[test]
    fn t814_low_swap_free_raises_pressure() {
        let b = base_tuning();
        let mut m = facts(64, 14); // Healthy の床 12 GiB は超える
        assert_eq!(pressure_of(m), Pressure::Healthy);
        // swap の空きが 1/8 で RAM の空きも 25% 未満だと Critical。
        m.swap_free_kb = m.swap_total_kb / 8;
        m.avail_kb = 10 * GIB;
        assert_eq!(pressure_of(m), Pressure::Critical);
        // swap が無いホストは段を上げない。
        let mut n = facts(64, 10);
        n.swap_total_kb = 0;
        n.swap_free_kb = 0;
        assert_eq!(pressure_of(n), Pressure::Tight);
        let _ = b;
    }

    #[test]
    fn t814_never_exceeds_base_and_is_monotone() {
        let b = base_tuning();
        let mut prev = plan_for(facts(64, 60), b).tuning;
        for avail in [40u64, 20, 12, 10, 8, 6, 5, 3, 1] {
            let t = plan_for(facts(64, avail), b).tuning;
            assert!(t.pread_threads <= prev.pread_threads, "avail={avail}");
            assert!(t.ra_threads <= prev.ra_threads, "avail={avail}");
            assert!(t.rc_cache_mb <= prev.rc_cache_mb, "avail={avail}");
            assert!(t.pool_mb <= prev.pool_mb, "avail={avail}");
            assert!(t.pread_threads >= MIN_PREAD_THREADS);
            assert!(t.ra_threads >= MIN_RA_THREADS);
            prev = t;
        }
    }

    #[test]
    fn t814_plan_gate_off_is_base() {
        set_force_gate_for_test(Some(false));
        let p = plan();
        set_force_gate_for_test(None);
        assert_eq!(p.tuning, base_tuning());
        assert_eq!(p.pressure, Pressure::Healthy);
    }

    /// 配線の発火: 配分を差し替えると getter が実際に従うこと。
    ///
    /// ここが赤になったら、`exp_pf` / `pf_pool` のどれかが静的既定へ
    /// 戻っている (計算したのに使われていない)。
    #[test]
    fn t814_getters_follow_the_plan() {
        static LOCK: Mutex<()> = Mutex::new(());
        let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // env 明示があると getter は env を優先する。テストの環境で
        // それらが刺さっていたら配線を確かめられないので、その時は諦める
        // (ゲート本体の CI は env 無しで回る)。
        for k in [
            "DENNO_PF_PREAD_THREADS",
            "DENNO_PF_RA_THREADS",
            "DENNO_PF_RC_CACHE_MB",
            "DENNO_PF_LM_TILE",
            "DENNO_PF_POOL_MB",
        ] {
            assert!(
                std::env::var(k).is_err(),
                "{k} が刺さっていると配線テストが空振りする"
            );
        }
        let t = Tuning {
            pread_threads: 7,
            ra_threads: 9,
            rc_cache_mb: 128,
            lm_tile: 1536,
            pool_mb: 777,
            pread_kb_cap: 0,
        };
        set_force_plan_for_test(Some(Plan {
            pressure: Pressure::Tight,
            avail_mib: 4096,
            total_mib: 65536,
            tuning: t,
        }));
        let got = (
            crate::model::exp_pf::pread_threads(),
            crate::model::exp_pf::rc_threads(),
            crate::model::exp_pf::rc_cache_bytes() >> 20,
            crate::model::exp_pf::lm_tile(8192),
            crate::model::pf_pool::pf_pool_bytes() >> 20,
        );
        set_force_plan_for_test(None);
        assert_eq!(got, (7, 9, 128, 1536, 777));
        // nt より大きいタイルは nt へ丸める (既存の契約は不変)。
        set_force_plan_for_test(Some(Plan {
            pressure: Pressure::Tight,
            avail_mib: 4096,
            total_mib: 65536,
            tuning: t,
        }));
        let small = crate::model::exp_pf::lm_tile(1000);
        set_force_plan_for_test(None);
        assert_eq!(small, 1000);
    }

    /// 実機の `/proc/meminfo` を読む経路 (ゲートで必ず 1 回は通る)。
    /// 値はホスト依存なので、範囲と不変条件だけを見る。
    #[test]
    fn t814_live_plan_is_within_bounds() {
        let Some(m) = read_meminfo() else {
            return; // /proc が無い環境
        };
        let b = base_tuning();
        let p = plan_for(m, b);
        assert!(p.total_mib > 0);
        assert!(p.avail_mib <= p.total_mib);
        assert!((MIN_PREAD_THREADS..=64).contains(&p.tuning.pread_threads));
        assert!((MIN_RA_THREADS..=32).contains(&p.tuning.ra_threads));
        assert!((MIN_RC_CACHE_MB..=2048).contains(&p.tuning.rc_cache_mb));
        assert!(p.tuning.pool_mb >= MIN_POOL_MB);
        assert!(p.tuning.lm_tile >= MIN_LM_TILE);
        // 恒等の段なら base と完全一致 (64 GiB ホストの平常時)。
        if p.pressure == Pressure::Healthy && p.total_mib < LARGE_TOTAL_MIB {
            assert_eq!(p.tuning, b);
        }
        // ゲート本体も 1 回通す (ログは --nocapture で見える)。
        assert!(crate::model::exp_pf::pread_threads() <= 64);
    }
}
