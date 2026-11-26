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
//!
//! # task 937: 監査で見つかった 2 つの穴 (どちらも既定 off のゲートで塞ぐ)
//!
//! ## 1. 段の閾値で配分が 1 Hz で往復する (ヒステリシスが無い)
//!
//! [`plan`] は 1 秒ごとに `/proc/meminfo` を読み直す。[`pressure_of`] の
//! 閾値は 1 本の線なので、他のプロセスの解放/確保で `avail` が線を跨ぐと
//! `Healthy <-> Tight` を毎秒往復し、そのたびに pread 本数・タイル幅・
//! プールが上下する (タイル幅はチャンク境界を動かすので、縮約順まで
//! 変わりうる)。[`HYST_ENV`] (`DENNO_PF_MEMADAPT_HYST`、既定 off) を
//! 明示したときだけ [`pressure_of_hyst`] を使い、**降格は即時・復帰は
//! `HYST_PCT` の余裕を要求** する非対称ヒステリシスを掛ける。swap 由来の
//! Critical は従来どおり即時 (押し出しは待てない)。
//!
//! ## 2. 匿名上限が床に負ける (avail < 3.34 GiB で cap が破れる)
//!
//! [`plan_for`] は cap を掛けたあと `clamp(MIN_POOL_MB, ..)` するので、
//! `anon_cap/4 < MIN_POOL_MB` の領域 (avail < 3414 MiB) ではプールが
//! 512 MiB に張り付き、上限を 3 倍以上超える。置き場も avail < 854 MiB で
//! 同じ。つまり **一番危ない領域でだけ cap が効かない**。`plan_for` の
//! 既定は 1 バイトも変えず (lab の A/B を壊さない)、[`ANON_STRICT_ENV`]
//! (`DENNO_PF_MEMADAPT_ANON_STRICT`、既定 off) のときだけ cap を勝たせ、
//! 床を [`MIN_POOL_MB_STRICT`] / [`MIN_RC_MB_STRICT`] まで下げる。
//! プールが小さいときは `pf_pool::borrow_plan` が従来の確保へ落ちるだけで、
//! レイアウトの契約 (合計 <= cap) は変わらない。
//!
//! ## 3. 監査で残した穴 (今回変えない)
//!
//! `exp_pf` の getter (`pread_threads` / `pread_bytes` / `rc_threads` /
//! `rc_cache_bytes` / `lm_tile`) は **初回呼び出しの配分を `OnceLock` に
//! 凍結**するので、プロセス起動後に段が変わっても本数・置き場・タイルは
//! 追随しない (追随するのは `pf_pool::pf_pool_bytes` だけ)。したがって
//! 上のヒステリシスが効くのは、`plan()` のログと毎回引くプールの経路。
//! 凍結を外すと hot path で `Mutex` を踏むため、今回は触らない (lab で
//! 凍結解除の費用を測ってから決める)。

use std::sync::Mutex;

/// `DENNO_PF_MEMADAPT`: メモリ空きに応じた配分を使うか。既定 on。
pub const MEMADAPT_ENV: &str = "DENNO_PF_MEMADAPT";

/// [`MEMADAPT_ENV`] の既定 (on)。
pub const DEFAULT_MEMADAPT: bool = true;

/// `DENNO_PF_MEMADAPT_HYST`: 段の閾値にヒステリシスを掛けるか (task 937)。既定 off。
pub const HYST_ENV: &str = "DENNO_PF_MEMADAPT_HYST";

/// [`HYST_ENV`] の既定 (off = 従来の 1 本の閾値)。
pub const DEFAULT_HYST: bool = false;

/// 復帰に要求する余裕 (%)。閾値の 10% 外側へ戻るまで段を上げない。
pub const HYST_PCT: u64 = 10;

/// `DENNO_PF_MEMADAPT_ANON_STRICT`: 匿名上限を床より優先するか (task 937)。既定 off。
pub const ANON_STRICT_ENV: &str = "DENNO_PF_MEMADAPT_ANON_STRICT";

/// [`ANON_STRICT_ENV`] の既定 (off = 床が勝つ従来の挙動)。
pub const DEFAULT_ANON_STRICT: bool = false;

/// 厳格モードでの床 (cap を優先するときの「これ以下へは絞らない」)。
pub const MIN_POOL_MB_STRICT: usize = 128;
pub const MIN_RC_MB_STRICT: usize = 32;

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

/// 既定 off のゲートの解釈 (純関数)。**明示的な真値だけ on**。
fn opt_in_from_value(v: Option<&str>, default: bool) -> bool {
    match v.map(str::trim).filter(|s| !s.is_empty()).map(|s| s.to_ascii_lowercase()) {
        None => default,
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
    }
}

/// [`HYST_ENV`] の解釈 (純関数)。明示的な真値だけ on。
pub fn hyst_from_value(v: Option<&str>) -> bool {
    opt_in_from_value(v, DEFAULT_HYST)
}

/// [`ANON_STRICT_ENV`] の解釈 (純関数)。明示的な真値だけ on。
pub fn anon_strict_from_value(v: Option<&str>) -> bool {
    opt_in_from_value(v, DEFAULT_ANON_STRICT)
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

/// 段の 2 本の閾値 (MiB): (Healthy の下限, Tight の下限)。
fn thresholds(total_mib: u64) -> (u64, u64) {
    (
        HEALTHY_MIN_MIB.max(total_mib * HEALTHY_FRAC_PCT / 100),
        TIGHT_MIN_MIB.max(total_mib * TIGHT_FRAC_PCT / 100),
    )
}

/// swap の空きが 25% 未満か (押し出しの危険)。
fn swap_low(m: MemFacts) -> bool {
    m.swap_total_kb > 0 && m.swap_free_kb * 4 < m.swap_total_kb
}

/// `avail` だけで決めた段 (swap を見ない素の判定)。
fn avail_pressure(total_mib: u64, avail_mib: u64) -> Pressure {
    let (healthy, tight) = thresholds(total_mib);
    if avail_mib < tight {
        Pressure::Critical
    } else if avail_mib < healthy {
        Pressure::Tight
    } else {
        Pressure::Healthy
    }
}

/// 段の判定 (純関数)。
pub fn pressure_of(m: MemFacts) -> Pressure {
    let total_mib = m.total_kb / KIB_PER_MIB;
    let avail_mib = m.avail_kb / KIB_PER_MIB;
    let base = avail_pressure(total_mib, avail_mib);
    // swap があり、空きが 25% 未満で、RAM の空きも 25% 未満なら危険とみなす
    // (swap が無いホストは「押し出し先が無い」だけなので段を上げない)。
    if swap_low(m) && avail_mib < total_mib * 25 / 100 {
        Pressure::Critical
    } else {
        base
    }
}

/// 段の判定にヒステリシスを掛ける (純関数、task 937)。
///
/// **降格は即時** (圧迫はすぐ避ける)、**復帰は [`HYST_PCT`] の余裕を要求**
/// する。`prev` が無いときは [`pressure_of`] と同じ。Critical へ入る方向は
/// 余裕を要求しないので、危険側の遅れは生じない。swap 由来の Critical は
/// ヒステリシスの対象外 (即時に効く)。
pub fn pressure_of_hyst(m: MemFacts, prev: Option<Pressure>) -> Pressure {
    let Some(prev) = prev else {
        return pressure_of(m);
    };
    let total_mib = m.total_kb / KIB_PER_MIB;
    let avail_mib = m.avail_kb / KIB_PER_MIB;
    if swap_low(m) && avail_mib < total_mib * 25 / 100 {
        return Pressure::Critical;
    }
    let (healthy, tight) = thresholds(total_mib);
    // 復帰側の線 (閾値の外側)。降格側には余裕を入れない (圧迫は待たない)。
    let healthy_up = healthy + healthy * HYST_PCT / 100;
    let tight_up = tight + tight * HYST_PCT / 100;
    let base = avail_pressure(total_mib, avail_mib);
    match prev {
        // 圧迫は待たない: 下がる方向は素の判定のまま。
        Pressure::Healthy => base,
        Pressure::Tight => match base {
            Pressure::Healthy => {
                if avail_mib >= healthy_up {
                    Pressure::Healthy
                } else {
                    Pressure::Tight
                }
            }
            // Critical へは即時 (安全側)。
            other => other,
        },
        Pressure::Critical => {
            if avail_mib < tight_up {
                Pressure::Critical
            } else if avail_mib >= healthy_up {
                Pressure::Healthy
            } else {
                Pressure::Tight
            }
        }
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
    plan_core(m, base, None, false)
}

/// ヒステリシスと匿名上限の厳格化を明示的に指定して配分を決める (純関数)。
///
/// `prev` が `Some` のとき [`pressure_of_hyst`] を使う。`anon_strict` が
/// `true` のときは匿名上限が床に勝ち、床は [`MIN_POOL_MB_STRICT`] /
/// [`MIN_RC_MB_STRICT`] まで下がる。既定の [`plan_for`] は
/// `(None, false)` と等価 (従来と 1 バイトも変わらない)。
pub fn plan_for_hyst(m: MemFacts, base: Tuning, prev: Option<Pressure>, anon_strict: bool) -> Plan {
    plan_core(m, base, prev, anon_strict)
}

fn plan_core(m: MemFacts, base: Tuning, prev: Option<Pressure>, anon_strict: bool) -> Plan {
    let total_mib = m.total_kb / KIB_PER_MIB;
    let avail_mib = m.avail_kb / KIB_PER_MIB;
    let pressure = pressure_of_hyst(m, prev);
    let mut t = target_for(pressure, total_mib, base);

    // 床。厳格モードでは cap を勝たせるため床を下げる。
    let (floor_pool, floor_rc) = if anon_strict {
        (MIN_POOL_MB_STRICT, MIN_RC_MB_STRICT)
    } else {
        (MIN_POOL_MB, MIN_RC_CACHE_MB)
    };

    // 匿名メモリの上限。残り 40% は page cache / 断片化の余裕として残す。
    let anon_cap = avail_mib * ANON_PCT / 100;
    let pool_cap = (anon_cap / POOL_DIV).max(floor_pool as u64);
    let rc_cap = (anon_cap / RC_DIV).max(floor_rc as u64);
    t.pool_mb = t.pool_mb.min(pool_cap as usize);
    t.rc_cache_mb = t.rc_cache_mb.min(rc_cap as usize);

    // 床と total の範囲へ丸める (異常値でも暴れない)。
    t.pread_threads = t.pread_threads.clamp(MIN_PREAD_THREADS, 64);
    t.ra_threads = t.ra_threads.clamp(MIN_RA_THREADS, 32);
    t.rc_cache_mb = t.rc_cache_mb.clamp(floor_rc, 2048);
    t.lm_tile = t.lm_tile.max(MIN_LM_TILE);
    t.pool_mb = t.pool_mb.clamp(floor_pool, 8192);

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

/// ゲート (`DENNO_PF_MEMADAPT_HYST`)。既定 off。
pub fn hyst_on() -> bool {
    #[cfg(test)]
    {
        let f = FORCE_HYST.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| hyst_from_value(std::env::var(HYST_ENV).ok().as_deref()))
}

/// ゲート (`DENNO_PF_MEMADAPT_ANON_STRICT`)。既定 off。
pub fn anon_strict_on() -> bool {
    #[cfg(test)]
    {
        let f = FORCE_ANON_STRICT.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| anon_strict_from_value(std::env::var(ANON_STRICT_ENV).ok().as_deref()))
}

#[cfg(test)]
static FORCE_GATE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);
#[cfg(test)]
static FORCE_HYST: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);
#[cfg(test)]
static FORCE_ANON_STRICT: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

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

/// テスト用: ヒステリシスのゲートを env 無しで固定する (`None` で解除)。
#[cfg(test)]
pub(crate) fn set_force_hyst_for_test(v: Option<bool>) {
    FORCE_HYST.store(
        match v {
            Some(true) => 1,
            Some(false) => 0,
            None => -1,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// テスト用: 匿名上限の厳格化を env 無しで固定する (`None` で解除)。
#[cfg(test)]
pub(crate) fn set_force_anon_strict_for_test(v: Option<bool>) {
    FORCE_ANON_STRICT.store(
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
    // 直前の段 (ヒステリシスの入力)。キャッシュが新しいうちはそのまま返す。
    let prev = {
        let g = STATE.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, p)) = *g {
            if now.saturating_duration_since(at).as_millis() < REFRESH_MS {
                return p;
            }
        }
        g.as_ref().map(|(_, p)| p.pressure)
    };
    let prev = if hyst_on() { prev } else { None };
    let p = match read_meminfo() {
        Some(m) => plan_for_hyst(m, base, prev, anon_strict_on()),
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
        // 恒等の段なら匿名メモリ上限で丸められない限り base と完全一致 (16 GiB 級 CI ランナーでは pool_mb が上限で丸まる)。
        if p.pressure == Pressure::Healthy && p.total_mib < LARGE_TOTAL_MIB {
            assert_eq!(p.tuning.pread_threads, b.pread_threads);
            assert_eq!(p.tuning.ra_threads, b.ra_threads);
            assert_eq!(p.tuning.lm_tile, b.lm_tile);
            assert_eq!(p.tuning.pread_kb_cap, b.pread_kb_cap);
            assert!(p.tuning.pool_mb <= b.pool_mb);
            assert!(p.tuning.rc_cache_mb <= b.rc_cache_mb);
            let anon_cap = p.avail_mib * ANON_PCT / 100;
            if (anon_cap / POOL_DIV) >= b.pool_mb as u64
                && (anon_cap / RC_DIV) >= b.rc_cache_mb as u64
            {
                assert_eq!(p.tuning, b);
            }
        }
        // ゲート本体も 1 回通す (ログは --nocapture で見える)。
        assert!(crate::model::exp_pf::pread_threads() <= 64);
    }

    // ---- task 937: 監査 (低メモリ段・ヒステリシス・匿名上限) ----

    fn facts_mib(total_mib: u64, avail_mib: u64) -> MemFacts {
        MemFacts {
            total_kb: total_mib * KIB_PER_MIB,
            avail_kb: avail_mib * KIB_PER_MIB,
            swap_total_kb: 8 * GIB,
            swap_free_kb: 8 * GIB,
        }
    }

    #[test]
    fn t937_new_gates_are_explicit_true_only() {
        assert!(!hyst_from_value(None), "ヒステリシスは既定 off");
        assert!(!hyst_from_value(Some("")));
        assert!(!hyst_from_value(Some("0")));
        assert!(hyst_from_value(Some("1")));
        assert!(hyst_from_value(Some(" ON ")));
        // 誤入力は off (既定 off のゲートは安全側 = 従来の挙動)。
        assert!(!hyst_from_value(Some("xyz")));

        assert!(!anon_strict_from_value(None), "匿名 cap の厳格化も既定 off");
        assert!(!anon_strict_from_value(Some("off")));
        assert!(anon_strict_from_value(Some("yes")));
        assert!(anon_strict_from_value(Some("true")));
        assert!(!anon_strict_from_value(Some("2")));

        // ゲートの実体 (force 上書きが効くこと = 分岐が配線されている証拠)。
        set_force_hyst_for_test(Some(true));
        assert!(hyst_on());
        set_force_hyst_for_test(Some(false));
        assert!(!hyst_on());
        set_force_hyst_for_test(None);
        set_force_anon_strict_for_test(Some(true));
        assert!(anon_strict_on());
        set_force_anon_strict_for_test(Some(false));
        assert!(!anon_strict_on());
        set_force_anon_strict_for_test(None);
    }

    /// 低メモリの段: 閾値のちょうど上下で段が切り替わり、床を割らない。
    #[test]
    fn t937_low_memory_tier_boundaries() {
        let b = base_tuning();
        // 64 GiB: healthy = 12 GiB の床、tight = 6 GiB の床。
        assert_eq!(pressure_of(facts_mib(65536, 12288)), Pressure::Healthy);
        assert_eq!(pressure_of(facts_mib(65536, 12287)), Pressure::Tight);
        assert_eq!(pressure_of(facts_mib(65536, 6144)), Pressure::Tight);
        assert_eq!(pressure_of(facts_mib(65536, 6143)), Pressure::Critical);

        // 16 GiB: total の 18% (2.9 GiB) より床 12 GiB が勝つ。
        assert_eq!(pressure_of(facts_mib(16384, 12288)), Pressure::Healthy);
        assert_eq!(pressure_of(facts_mib(16384, 12287)), Pressure::Tight);
        assert_eq!(pressure_of(facts_mib(16384, 6144)), Pressure::Tight);

        // 8 GiB: 床 12 GiB > total なので、全開でも Healthy にはならない。
        assert_eq!(pressure_of(facts_mib(8192, 8192)), Pressure::Tight);
        assert_eq!(pressure_of(facts_mib(8192, 6143)), Pressure::Critical);

        // 段ごとの配分が床を割らず、Critical が Tight 以下であること。
        let mut prev = plan_for(facts_mib(16384, 16000), b).tuning;
        for avail in [12288u64, 9000, 6144, 6143, 4096, 2048, 1024, 256, 32] {
            let p = plan_for(facts_mib(16384, avail), b);
            assert!(p.tuning.pread_threads >= MIN_PREAD_THREADS, "avail={avail}");
            assert!(p.tuning.ra_threads >= MIN_RA_THREADS, "avail={avail}");
            assert!(p.tuning.pool_mb >= MIN_POOL_MB, "avail={avail}");
            assert!(p.tuning.rc_cache_mb >= MIN_RC_CACHE_MB, "avail={avail}");
            assert!(p.tuning.lm_tile >= MIN_LM_TILE, "avail={avail}");
            assert!(p.tuning.pread_threads <= prev.pread_threads, "avail={avail}");
            assert!(p.tuning.pool_mb <= prev.pool_mb, "avail={avail}");
            prev = p.tuning;
        }
    }

    /// 極端に空きが少なくても (32 MiB) 暴れない: 全部の床に張り付く。
    #[test]
    fn t937_starved_host_hits_every_floor() {
        let b = base_tuning();
        let p = plan_for(facts_mib(16384, 32), b);
        assert_eq!(p.pressure, Pressure::Critical);
        assert_eq!(p.tuning.pread_threads, MIN_PREAD_THREADS);
        assert_eq!(p.tuning.ra_threads, MIN_RA_THREADS);
        assert_eq!(p.tuning.rc_cache_mb, MIN_RC_CACHE_MB);
        assert_eq!(p.tuning.pool_mb, MIN_POOL_MB);
        // タイルは base の半分 (5,120 -> 2,560) で、床 1,024 より上。
        assert_eq!(p.tuning.lm_tile, (b.lm_tile / 2).max(MIN_LM_TILE));
        assert!(p.tuning.lm_tile >= MIN_LM_TILE);
        assert_eq!(p.tuning.pread_kb_cap, 256);
        // avail 0 でも同じ (0 除算や underflow が無いこと)。
        let z = plan_for(facts_mib(16384, 0), b);
        assert_eq!(z.tuning, p.tuning);
        assert_eq!(z.avail_mib, 0);
    }

    /// 閾値を毎秒跨ぐ状況で、ヒステリシスが段の往復を止めること。
    #[test]
    fn t937_hysteresis_stops_boundary_flapping() {
        // 64 GiB の healthy = 12288 MiB。閾値の上下 400 MiB を行き来する。
        let seq = [12000u64, 12400, 12000, 12400, 12000, 12400, 12000, 12400];

        // 従来 (ヒステリシス無し): 毎回 Tight <-> Healthy を往復する。
        let mut prev: Option<Pressure> = None;
        let mut raw_changes = 0;
        for &avail in &seq {
            let p = pressure_of(facts_mib(65536, avail));
            if prev.is_some() && prev != Some(p) {
                raw_changes += 1;
            }
            prev = Some(p);
        }
        assert_eq!(raw_changes, seq.len() - 1, "素の判定は毎回反転する");

        // ヒステリシス有り: 一度 Tight に入ったら healthy*1.1 (13516 MiB) まで戻らない。
        let mut prev = Some(Pressure::Tight);
        let mut hyst_changes = 0;
        for &avail in &seq {
            let p = pressure_of_hyst(facts_mib(65536, avail), prev);
            if prev != Some(p) {
                hyst_changes += 1;
            }
            prev = Some(p);
        }
        assert_eq!(hyst_changes, 0, "12.4 GiB では復帰しない");

        // 余裕まで戻れば復帰する (13516 MiB 以上)。
        assert_eq!(
            pressure_of_hyst(facts_mib(65536, 13515), Some(Pressure::Tight)),
            Pressure::Tight
        );
        assert_eq!(
            pressure_of_hyst(facts_mib(65536, 13516), Some(Pressure::Tight)),
            Pressure::Healthy
        );
    }

    /// 非対称: 降格は即時、復帰は余裕が要る。危険側の遅れは無い。
    #[test]
    fn t937_hysteresis_is_asymmetric() {
        let f = |avail: u64| facts_mib(65536, avail);
        // Healthy からは素の判定で即時に落ちる。
        assert_eq!(pressure_of_hyst(f(12000), Some(Pressure::Healthy)), Pressure::Tight);
        assert_eq!(pressure_of_hyst(f(6000), Some(Pressure::Healthy)), Pressure::Critical);
        // Tight から Critical へは即時 (危険側に余裕を入れない)。
        assert_eq!(pressure_of_hyst(f(6143), Some(Pressure::Tight)), Pressure::Critical);
        assert_eq!(pressure_of_hyst(f(4000), Some(Pressure::Tight)), Pressure::Critical);
        // Critical からは tight*1.1 (6,758 MiB) を超えるまで戻らない。
        assert_eq!(pressure_of_hyst(f(6757), Some(Pressure::Critical)), Pressure::Critical);
        assert_eq!(pressure_of_hyst(f(6758), Some(Pressure::Critical)), Pressure::Tight);
        // Critical から直接 Healthy へは healthy*1.1 以上。
        assert_eq!(pressure_of_hyst(f(14000), Some(Pressure::Critical)), Pressure::Healthy);
        // swap 由来の Critical は即時 (ヒステリシスの外)。
        let mut m = facts_mib(65536, 12000);
        m.swap_free_kb = m.swap_total_kb / 8;
        assert_eq!(pressure_of_hyst(m, Some(Pressure::Healthy)), Pressure::Critical);
        // prev 無しは素の判定と一致する。
        for avail in [14000u64, 12288, 12287, 6144, 6143, 100] {
            assert_eq!(
                pressure_of_hyst(f(avail), None),
                pressure_of(f(avail)),
                "avail={avail}"
            );
        }
    }

    /// 既定 (ヒステリシス off) の `plan_for` と `plan_for_hyst(.., None, false)` が同一。
    #[test]
    fn t937_hysteresis_off_is_byte_identical() {
        let b = base_tuning();
        for avail in [48000u64, 13000, 12288, 9000, 6144, 5000, 2048, 64] {
            let m = facts_mib(65536, avail);
            let legacy = plan_for(m, b);
            // `plan()` はヒステリシス off のとき prev を渡さない。その経路が従来と同一。
            let got = plan_for_hyst(m, b, None, false);
            assert_eq!(got.pressure, legacy.pressure, "avail={avail}");
            assert_eq!(got.tuning, legacy.tuning, "avail={avail}");
            // prev を渡したときだけ段が保持される (差が出ること自体が分岐の発火)。
            let held = plan_for_hyst(m, b, Some(Pressure::Tight), true);
            if legacy.pressure == Pressure::Healthy && avail < 13516 {
                assert_eq!(held.pressure, Pressure::Tight, "avail={avail}");
            }
        }
    }

    /// 匿名上限の執行: 床が邪魔しない領域では cap が厳密に効く。
    #[test]
    fn t937_anon_cap_binds_above_the_floor() {
        let b = base_tuning();
        for avail in [3414u64, 4000, 6000, 12000, 20000, 40000, 60000] {
            let p = plan_for(facts_mib(65536, avail), b);
            let anon = p.avail_mib * ANON_PCT / 100;
            assert!(
                p.tuning.pool_mb as u64 <= anon / POOL_DIV,
                "avail={avail} pool={} cap={}",
                p.tuning.pool_mb,
                anon / POOL_DIV
            );
            assert!(
                p.tuning.rc_cache_mb as u64 <= anon / RC_DIV,
                "avail={avail} rc={} cap={}",
                p.tuning.rc_cache_mb,
                anon / RC_DIV
            );
            assert!(p.tuning.pool_mb >= MIN_POOL_MB);
        }
        // 境界: anon/4 = 512 MiB ちょうどは avail 3414 MiB。
        assert_eq!((3414u64 * ANON_PCT / 100) / POOL_DIV, 512);
    }

    /// 監査で見つかった穴: 床が cap に勝つ領域と、厳格モードでの執行。
    #[test]
    fn t937_anon_cap_floor_legacy_vs_strict() {
        let b = base_tuning();
        // avail 2048 MiB -> anon 1228 MiB -> cap 307 MiB だが、従来は床 512 が勝つ。
        let legacy = plan_for(facts_mib(65536, 2048), b);
        let anon = legacy.avail_mib * ANON_PCT / 100;
        assert_eq!(anon / POOL_DIV, 307);
        assert_eq!(legacy.tuning.pool_mb, MIN_POOL_MB, "従来は床に張り付く");
        assert!(
            legacy.tuning.pool_mb as u64 > anon / POOL_DIV,
            "ここが監査で見つかった cap 破れ (既定は変えない)"
        );

        // 厳格モード: cap が勝ち、床は 128 MiB まで下がる。
        let strict = plan_for_hyst(facts_mib(65536, 2048), b, None, true);
        assert_eq!(strict.pressure, Pressure::Critical);
        assert_eq!(strict.tuning.pool_mb, 307);
        assert!(strict.tuning.pool_mb as u64 <= anon / POOL_DIV);

        // 極端 (avail 128 MiB) でも絶対床 128 / 32 は守る。
        let tiny = plan_for_hyst(facts_mib(65536, 128), b, None, true);
        assert_eq!(tiny.tuning.pool_mb, MIN_POOL_MB_STRICT);
        assert_eq!(tiny.tuning.rc_cache_mb, MIN_RC_MB_STRICT);
        assert_eq!(tiny.tuning.pread_threads, MIN_PREAD_THREADS);

        // 厳格モードでも単調 (空きが減って配分が増えない)。
        let mut prev = plan_for_hyst(facts_mib(65536, 60000), b, None, true).tuning;
        for avail in [12000u64, 6000, 3414, 2048, 1024, 128, 0] {
            let t = plan_for_hyst(facts_mib(65536, avail), b, None, true).tuning;
            assert!(t.pool_mb <= prev.pool_mb, "avail={avail}");
            assert!(t.rc_cache_mb <= prev.rc_cache_mb, "avail={avail}");
            assert!(t.pread_threads <= prev.pread_threads, "avail={avail}");
            assert!(t.pool_mb >= MIN_POOL_MB_STRICT, "avail={avail}");
            assert!(t.rc_cache_mb >= MIN_RC_MB_STRICT, "avail={avail}");
            prev = t;
        }
    }

    /// ヒステリシス + 厳格 cap を同時に掛けても、恒等の段は base のまま。
    #[test]
    fn t937_strict_and_hyst_keep_healthy_identity() {
        let b = base_tuning();
        for avail in [20000u64, 40000, 60000] {
            let p = plan_for_hyst(facts_mib(65536, avail), b, Some(Pressure::Tight), true);
            assert_eq!(p.tuning, b, "avail={avail}");
            assert_eq!(p.pressure, Pressure::Healthy);
        }
        // 48 GiB 級 (total < 96 GiB) は Healthy でも base のまま (拡大しない)。
        let p = plan_for_hyst(facts_mib(49152, 48000), b, None, true);
        assert_eq!(p.tuning, b);
    }
}
