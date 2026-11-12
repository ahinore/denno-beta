//! task 260: expert 重みの**ホスト RAM tier** (VRAM tier と mmap の間)。
//!
//! # なぜ要るか
//!
//! production の 2026-11-12 の観測: swift15-abl IQ3_S の expert 52.8 GB は
//! read-only mmap で、`DENNO_START_PREFETCH=1` が起動時に page cache へ読む
//! (52,815 MB / 1.3-1.8 s)。ところが page cache は**誰の物でもない**ので、
//! 別モデルの大きな読みで追い出される。実際、別モデルを回して戻すと起動の
//! prefetch が disk から読み直し (load 19.9 s、平常 2.7 s)、最初のリクエストが
//! 149 ms/tok (平常 26) になった。UD-Q4_K_XL は 111 GB で RAM に収まらず、
//! 初回 797 ms/tok / 以降 57-229 ms/tok が disk 由来だった。
//!
//! # 何をするか
//!
//! `DENNO_EXPERT_RAM_MB` (MiB) の予算で、expert の生バイトを**所有する**
//! ホストバッファへ常駐させる。常駐先は CUDA の `cuMemHostAlloc` (pinned。
//! 取れなければヒープ)。page cache と違って追い出されず、H2D の転送元にも
//! そのまま使える (pinned 52.6 GB/s / pageable 19.6 GB/s)。
//!
//! 引く順番は **VRAM tier → RAM tier → mmap**。RAM tier は
//! [`crate::model::expert_cache::ExpertCacheSet`] の固定長ストア版なので、
//! LRU・鍵 ((層, expert))・返すバイト列は従来の pread キャッシュと同じ。
//!
//! # ゲート
//!
//! - `DENNO_EXPERT_RAM_MB`: **未設定 / `0` / `off` は無効** (従来と 1 バイトも
//!   変わらない)。`auto` は自動 = `MemAvailable` − 余裕
//!   (`DENNO_EXPERT_RAM_MARGIN_MB`, 既定 8192 MiB) を、モデルの expert 総バイト数で
//!   頭打ちにする。数値は MiB。
//!
//!   **既定 off にしてある理由** (lab t260c の実測): 59 GiB の機械で IQ3_S の
//!   expert 47 GB を所有すると MemAvailable が 8-10 GB まで落ち、expert 以外
//!   (dense / PLE / 非常駐 expert) が page cache から追い出されて warm decode が
//!   19.4 → 46.4 ms/tok になった。cold の prefill は 17.21 → 8.89 s (-48%) と
//!   狙いどおりなので、**大きさを絞れば**両立する見込み (未検証)。
//! - `DENNO_EXPERT_RAM_PIN`: `0` で pinned を諦めてヒープ (A/B 用。既定 on)。
//! - `DENNO_EXPERT_RAM_FILL`: `0` で起動時の充填をしない (使用時に埋まる。既定 on)。
//! - `DENNO_EXPERT_RAM_DBG`: `1` で 1 リクエストごとに 1 行 stderr。

use std::sync::{Arc, RwLock};

use crate::model::expert_cache::ExpertCacheSet;

/// 予算 (MiB) の env。`0` で無効、未設定 / `auto` で自動。
pub const RAM_MB_ENV: &str = "DENNO_EXPERT_RAM_MB";
/// pinned を試すか (`0` でヒープ)。既定 on。
pub const PIN_ENV: &str = "DENNO_EXPERT_RAM_PIN";
/// 起動時に予算まで充填するか (`0` で off)。既定 on。
pub const FILL_ENV: &str = "DENNO_EXPERT_RAM_FILL";
/// 自動予算で `MemAvailable` に残す余裕 (MiB)。既定 8192。
pub const MARGIN_MB_ENV: &str = "DENNO_EXPERT_RAM_MARGIN_MB";
/// 充填のスレッド数。既定 12 (`start-warm` と同じ)。
pub const THREADS_ENV: &str = "DENNO_EXPERT_RAM_THREADS";
/// 1 リクエストごとに 1 行 stderr を出す。
pub const DBG_ENV: &str = "DENNO_EXPERT_RAM_DBG";

/// 自動予算で `MemAvailable` に残す余裕の既定 (8 GiB)。
///
/// システム / Hermes の dev コンテナ / dense 重みの page cache のぶん。
/// task 文の指定値。
pub const DEFAULT_MARGIN_MB: u64 = 8192;

/// `DENNO_EXPERT_RAM_MB` の解釈。
///
/// - `None` (未設定) / 空 / `0` / `off` / `false` → `Some(0)` = 無効 (既定)
/// - `auto` → `None` = 自動
/// - 数値 → `Some(MiB)`
/// - その他 → `Some(0)` (無効。誤植で勝手にメモリを確保しない)
pub fn mb_from_value(v: Option<&str>) -> Option<u64> {
    let s = match v {
        Some(s) => s.trim(),
        None => return Some(0),
    };
    if s.eq_ignore_ascii_case("auto") {
        return None;
    }
    if s.is_empty()
        || matches!(
            s.to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        )
    {
        return Some(0);
    }
    s.parse::<u64>().ok().map(Some).unwrap_or(Some(0))
}

/// ゲートが有効か (0 でも自動でもない)。
pub fn enabled_from_value(v: Option<&str>) -> bool {
    mb_from_value(v) != Some(0)
}

/// 予算 (バイト)。`avail` は `MemAvailable`、`expert_bytes` はモデルの
/// expert 総バイト数 (上限)。
///
/// 自動は `avail - margin` を expert 総量で頭打ちにする。数値指定でも
/// `avail - margin` を超えない (指定した予算で機械を押し潰さない)。
pub fn budget_bytes(v: Option<&str>, avail: Option<u64>, expert_bytes: u64, margin: u64) -> u64 {
    let room = match avail {
        Some(a) => a.saturating_sub(margin),
        None => 0,
    };
    match mb_from_value(v) {
        Some(0) => 0,
        Some(mb) => {
            let want = mb << 20;
            match avail {
                Some(_) => want.min(expert_bytes).min(room),
                // MemAvailable が読めないときは明示予算をそのまま使う
                None => want.min(expert_bytes),
            }
        }
        None => room.min(expert_bytes),
    }
}

/// `DENNO_EXPERT_RAM_MARGIN_MB` の解釈 (未設定・非数は既定)。
pub fn margin_mb_from_value(v: Option<&str>) -> u64 {
    match v.map(str::trim).and_then(|s| s.parse::<u64>().ok()) {
        Some(mb) if mb > 0 => mb,
        _ => DEFAULT_MARGIN_MB,
    }
}

/// `DENNO_EXPERT_RAM_PIN` の解釈 (**既定 on**。明示的な偽値だけ off)。
pub fn pin_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(|s| s.to_ascii_lowercase()),
        Some(ref s) if matches!(s.as_str(), "0" | "off" | "false" | "no")
    )
}

/// `DENNO_EXPERT_RAM_FILL` の解釈 (**既定 on**)。
pub fn fill_from_value(v: Option<&str>) -> bool {
    pin_from_value(v)
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok()
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
///
/// **テストでは既定 off** (env 未設定のとき)。合成モデルを組むテストが毎回
/// 数十 GB の予算を計算したり、グローバルな tier がテスト間で混ざったりしない
/// ようにするため。env を明示したときはテストでもその値に従う。
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if cfg!(test) && env(RAM_MB_ENV).is_none() {
            return false;
        }
        enabled_from_value(env(RAM_MB_ENV).as_deref())
    })
}

pub fn want_pinned() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| pin_from_value(env(PIN_ENV).as_deref()))
}

pub fn fill_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| fill_from_value(env(FILL_ENV).as_deref()))
}

pub fn margin_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| margin_mb_from_value(env(MARGIN_MB_ENV).as_deref()).saturating_mul(1 << 20))
}

pub fn threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env(THREADS_ENV)
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(12)
            .min(32)
    })
}

pub fn dbg_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        matches!(
            env(DBG_ENV).as_deref().map(str::trim),
            Some("1") | Some("true") | Some("on")
        )
    })
}

/// 明示予算 (診断用。自動のときは `None`)。
pub fn budget_mb_env() -> Option<u64> {
    mb_from_value(env(RAM_MB_ENV).as_deref())
}

/// env と `MemAvailable` から予算 (バイト) を決める。
pub fn budget_bytes_env(expert_bytes: u64) -> u64 {
    budget_bytes(
        env(RAM_MB_ENV).as_deref(),
        mem_available(),
        expert_bytes,
        margin_bytes(),
    )
}

// ---------------------------------------------------------------------------
// プロセスで 1 本の tier
// ---------------------------------------------------------------------------
//
// VRAM tier ([`crate::model::expert_vram`]) と同じく、モデルではなくプロセスに
// 1 本置く。forward の深い所 (expert のミス計算 / H2D の転送元 / prefill の
// ストリーム) から引くので、引数で配ると経路が増えすぎる。
//
// モデルを差し替えるときは [`install`] が古い Arc を落とす (pinned も解放される)。

static TIER: RwLock<Option<Arc<ExpertCacheSet>>> = RwLock::new(None);

/// tier を差し替える (`None` で外す)。古い tier はここで落ちる。
pub fn install(set: Option<Arc<ExpertCacheSet>>) {
    let mut g = TIER.write().unwrap_or_else(|e| e.into_inner());
    *g = set;
}

/// いまの tier。参照カウントを 1 つ進めて返す (ロックはすぐ離す)。
pub fn get() -> Option<Arc<ExpertCacheSet>> {
    TIER.read()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Arc::clone)
}

/// tier が有効か (ロックを取らない軽い判定は `enabled()` のほう)。
pub fn active() -> bool {
    TIER.read()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

// ---------------------------------------------------------------------------
// 統計 (1 リクエストぶん)
// ---------------------------------------------------------------------------
//
// 数字は tier 自身の LRU 統計から読む (別に数えると、経路ごとに数え漏れが出る)。

/// 1 リクエストぶんの 1 行。tier が無効なら `None`。
pub fn stats_line() -> Option<String> {
    let set = get()?;
    let (mut hits, mut misses, mut read) = (0u64, 0u64, 0u64);
    for m in crate::model::expert_cache::ExpertMatrix::ALL {
        let s = set.stats(m);
        hits += s.hits;
        misses += s.misses;
        read += s.bytes_read;
    }
    let n = hits + misses;
    let used = set.used_experts();
    let cap = set.slots(crate::model::expert_cache::ExpertMatrix::Gate);
    let pinned = set.pinned_bytes() >> 20;
    let reserved = set.reserved_bytes() >> 20;
    Some(format!(
        "[exp-ram] lookups={n} hits={hits} miss={misses} hit_rate={:.3} resident={used}/{cap} pinned={pinned}MiB reserved={reserved}MiB reads_mib={}",
        if n > 0 { hits as f64 / n as f64 } else { 0.0 },
        read >> 20
    ))
}

/// 積算を 0 に戻す (リクエスト終端)。
pub fn stats_reset() {
    if let Some(set) = get() {
        for m in crate::model::expert_cache::ExpertMatrix::ALL {
            set.reset_stats_for(m);
        }
    }
}

/// `MemAvailable` (バイト)。
pub fn mem_available() -> Option<u64> {
    crate::model::expert_cache::mem_available()
}

// ---------------------------------------------------------------------------
// 診断 (task 260 step 1: ホスト側のバイトがどこから来るか)
// ---------------------------------------------------------------------------
//
// 測るもの:
// - `/proc/self/stat` の minflt / majflt の**リクエスト前後の差**。
//   major fault = disk から読んだ回数 (page cache に無かった)。
// - expert テンソルの mmap の常駐量 (mincore)。「page cache に何割載っているか」。
// - MemAvailable。
//
// ゲート: `DENNO_EXPERT_RAM_PROBE=1`、または `DENNO_Q4_PROFILE` が立っている
// (`mincore` は高いので `DENNO_EXPERT_RAM_PROBE=1` のときだけ走る)。
// `DENNO_EXPERT_RAM_DROP=1` は**リクエストの先頭で expert の page cache を落とす**
// (冷えた機械の再現。dense 重みは落とさない)。

/// `DENNO_EXPERT_RAM_PROBE`: 1 で mincore まで測る。
pub const PROBE_ENV: &str = "DENNO_EXPERT_RAM_PROBE";
/// `DENNO_EXPERT_RAM_DROP`: 1 でリクエストごとに expert の page cache を落とす。
pub const DROP_ENV: &str = "DENNO_EXPERT_RAM_DROP";
/// `DENNO_EXPERT_RAM_ADAPT`: 1 で使用時に埋める (LRU)。**既定 off**。
///
/// off のときの tier は「起動時に充填した固定の集合」で、decode のミスでは
/// ファイルを読み直さない。lab t260a の実測: decode のミスで pread して
/// 埋める経路は、ヒット率が上がる前に 29% 遅くなった (page cache と同じ
/// バイト列をもう一度読むだけで、page cache を追い出す)。
pub const ADAPT_ENV: &str = "DENNO_EXPERT_RAM_ADAPT";

/// expert テンソルのファイル範囲 (part, off, len)。落とすときに使う。
static RANGES: RwLock<Vec<(usize, u64, u64)>> = RwLock::new(Vec::new());
/// GGUF のパートのパス。
static PARTS: RwLock<Vec<std::path::PathBuf>> = RwLock::new(Vec::new());
/// expert テンソルの mmap 範囲 (ptr, len)。常駐量を測るのに使う。
static MMAP: RwLock<Vec<(usize, usize)>> = RwLock::new(Vec::new());
/// 前回の (minflt, majflt)。
static LAST_FAULTS: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));

fn flag(k: &str) -> bool {
    matches!(
        env(k).as_deref().map(str::trim),
        Some("1") | Some("true") | Some("on")
    )
}

pub fn probe_enabled() -> bool {
    flag(PROBE_ENV) || env("DENNO_Q4_PROFILE").is_some()
}

pub fn drop_enabled() -> bool {
    flag(DROP_ENV)
}

/// 使用時に埋めるか (`DENNO_EXPERT_RAM_ADAPT=1`。既定 off)。
pub fn adapt_enabled() -> bool {
    flag(ADAPT_ENV)
}

/// モデルの expert 範囲を覚える (リクエストをまたいで使う)。
pub fn install_ranges(
    parts: Vec<std::path::PathBuf>,
    ranges: Vec<(usize, u64, u64)>,
    mmap: Vec<(usize, usize)>,
) {
    *PARTS.write().unwrap_or_else(|e| e.into_inner()) = parts;
    *RANGES.write().unwrap_or_else(|e| e.into_inner()) = ranges;
    *MMAP.write().unwrap_or_else(|e| e.into_inner()) = mmap;
}

/// `/proc/self/stat` の (minflt, majflt)。
fn faults() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // 2 番目 (comm) は括弧でくくられ、空白を含みうる。最後の ')' の後ろから数える。
    let rest = &s[s.rfind(')')? + 1..];
    let f: Vec<&str> = rest.split_whitespace().collect();
    // rest[0] = state (stat の 3 番目)。minflt は 10 番目、majflt は 12 番目。
    Some((f.get(7)?.parse().ok()?, f.get(9)?.parse().ok()?))
}

/// [ptr, ptr+len) のうち常駐しているバイト数 (mincore)。取れなければ 0。
#[cfg(unix)]
fn resident_bytes(ptr: usize, len: usize) -> u64 {
    if len == 0 || ptr == 0 {
        return 0;
    }
    const PAGE: usize = 4096;
    let start = ptr & !(PAGE - 1);
    let pages = (ptr + len - start).div_ceil(PAGE);
    if pages == 0 {
        return 0;
    }
    let mut vec = vec![0u8; pages];
    // glibc の `mincore(2)` (syscall の薄い包み)。
    unsafe {
        #[link(name = "c")]
        extern "C" {
            fn mincore(addr: *mut std::ffi::c_void, length: usize, vec: *mut u8) -> i32;
        }
        let rc = mincore(start as *mut std::ffi::c_void, pages * PAGE, vec.as_mut_ptr());
        if rc != 0 {
            return 0;
        }
    }
    vec.iter().filter(|&&b| b & 1 != 0).count() as u64 * PAGE as u64
}

#[cfg(not(unix))]
fn resident_bytes(_ptr: usize, _len: usize) -> u64 {
    0
}

/// リクエスト終端の 1 行。ゲート off なら `None`。
pub fn probe_line(tag: &str) -> Option<String> {
    if !probe_enabled() {
        return None;
    }
    let (minflt, majflt) = faults()?;
    let (dm, dM) = {
        let mut g = LAST_FAULTS.lock().unwrap_or_else(|e| e.into_inner());
        let d = (minflt.saturating_sub(g.0), majflt.saturating_sub(g.1));
        *g = (minflt, majflt);
        d
    };
    let mut line = format!(
        "[exp-ram-probe] {tag} minflt={dm} majflt={dM} memavail={}MiB",
        mem_available().map(|b| b >> 20).unwrap_or(0)
    );
    if flag(PROBE_ENV) {
        let ranges = MMAP.read().unwrap_or_else(|e| e.into_inner());
        let mut res = 0u64;
        let mut tot = 0u64;
        for &(p, l) in ranges.iter() {
            res += resident_bytes(p, l);
            tot += l as u64;
        }
        line.push_str(&format!(
            " expert_resident={}MiB/{}MiB ({:.1}%)",
            res >> 20,
            tot >> 20,
            if tot > 0 {
                res as f64 * 100.0 / tot as f64
            } else {
                0.0
            }
        ));
    }
    Some(line)
}

/// リクエストの先頭で expert の page cache を落とす (診断。dense は落とさない)。
/// ゲート off なら空 (呼び出し側は何も出力しない)。
pub fn drop_pages_if_enabled() -> Vec<String> {
    if !drop_enabled() {
        return Vec::new();
    }
    let parts = PARTS.read().unwrap_or_else(|e| e.into_inner()).clone();
    let ranges = RANGES.read().unwrap_or_else(|e| e.into_inner()).clone();
    let mut files: Vec<std::fs::File> = Vec::with_capacity(parts.len());
    for p in &parts {
        files.push(match std::fs::File::open(p) {
            Ok(f) => f,
            Err(_) => return vec!["[exp-ram] drop: ファイルを開けない".to_string()],
        });
    }
    let mut ok = 0usize;
    let mut err = 0usize;
    for (part, off, len) in ranges {
        let Some(f) = files.get(part) else {
            err += 1;
            continue;
        };
        if fadvise_dontneed(f, off, len).is_ok() {
            ok += 1;
        } else {
            err += 1;
        }
    }
    vec![format!(
        "[exp-ram] drop expert page cache: ok={ok} err={err} ({DROP_ENV}=1)"
    )]
}

#[cfg(unix)]
fn fadvise_dontneed(f: &std::fs::File, off: u64, len: u64) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    extern "C" {
        fn posix_fadvise(
            fd: std::os::raw::c_int,
            off: i64,
            len: i64,
            advice: std::os::raw::c_int,
        ) -> std::os::raw::c_int;
    }
    const POSIX_FADV_DONTNEED: std::os::raw::c_int = 4;
    let rc = unsafe {
        posix_fadvise(
            f.as_raw_fd(),
            off as i64,
            len as i64,
            POSIX_FADV_DONTNEED,
        )
    };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

#[cfg(not(unix))]
fn fadvise_dontneed(_f: &std::fs::File, _off: u64, _len: u64) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "posix_fadvise は unix のみ",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mb_values_are_explicit() {
        // 既定 (未設定) は無効
        assert_eq!(mb_from_value(None), Some(0));
        assert_eq!(mb_from_value(Some("")), Some(0));
        // 自動は明示したときだけ
        assert_eq!(mb_from_value(Some("auto")), None);
        assert_eq!(mb_from_value(Some("AUTO")), None);
        assert_eq!(mb_from_value(Some("0")), Some(0));
        assert_eq!(mb_from_value(Some(" off ")), Some(0));
        assert_eq!(mb_from_value(Some("false")), Some(0));
        assert_eq!(mb_from_value(Some("4096")), Some(4096));
        // 誤植は無効 (勝手にメモリを確保しない)
        assert_eq!(mb_from_value(Some("4G")), Some(0));
        assert!(!enabled_from_value(Some("0")));
        assert!(!enabled_from_value(None));
        assert!(enabled_from_value(Some("auto")));
        assert!(enabled_from_value(Some("1024")));
    }

    #[test]
    fn budget_is_margin_bounded_and_capped_by_experts() {
        let gb = 1u64 << 30;
        // 未設定は 0 (既定 off)
        assert_eq!(budget_bytes(None, Some(50 * gb), 100 * gb, 8 * gb), 0);
        // 自動 (明示): MemAvailable - margin と expert 総量の小さいほう
        assert_eq!(
            budget_bytes(Some("auto"), Some(50 * gb), 100 * gb, 8 * gb),
            42 * gb
        );
        assert_eq!(
            budget_bytes(Some("auto"), Some(50 * gb), 10 * gb, 8 * gb),
            10 * gb
        );
        // 明示でも MemAvailable - margin を超えない
        assert_eq!(
            budget_bytes(Some("40960"), Some(50 * gb), 100 * gb, 8 * gb),
            40 * gb
        );
        assert_eq!(
            budget_bytes(Some("1024"), Some(50 * gb), 100 * gb, 8 * gb),
            gb
        );
        // 無効
        assert_eq!(budget_bytes(Some("0"), Some(50 * gb), 100 * gb, 8 * gb), 0);
        // MemAvailable が読めないときは明示予算だけが残る (自動は 0)
        assert_eq!(budget_bytes(Some("auto"), None, 100 * gb, 8 * gb), 0);
        assert_eq!(budget_bytes(Some("2048"), None, 100 * gb, 8 * gb), 2 * gb);
        // 余裕より空きが少ない
        assert_eq!(
            budget_bytes(Some("auto"), Some(4 * gb), 100 * gb, 8 * gb),
            0
        );
    }

    #[test]
    fn margin_and_flags_default() {
        assert_eq!(margin_mb_from_value(None), DEFAULT_MARGIN_MB);
        assert_eq!(margin_mb_from_value(Some("0")), DEFAULT_MARGIN_MB);
        assert_eq!(margin_mb_from_value(Some("2048")), 2048);
        assert!(pin_from_value(None));
        assert!(!pin_from_value(Some("0")));
        assert!(!pin_from_value(Some("off")));
        assert!(fill_from_value(None));
        assert!(!fill_from_value(Some("false")));
    }

    #[test]
    fn stats_line_is_none_without_tier() {
        install(None);
        assert!(stats_line().is_none());
        assert!(!active());
    }

    /// `/proc/self/stat` の minflt / majflt が読めること (単調非減少)。
    ///
    /// 「触ったら必ず増える」はアロケータの再利用で落ちるので検査しない
    /// (この binary は 1 プロセスで全テストを回す)。
    #[test]
    fn faults_are_readable_and_monotonic() {
        let a = faults().expect("/proc/self/stat が読めない");
        let mut v = vec![0u8; 8 << 20];
        for i in (0..v.len()).step_by(1024) {
            v[i] = 1;
        }
        let b = faults().expect("/proc/self/stat が読めない");
        assert!(b.0 >= a.0 && b.1 >= a.1, "fault が減った: {a:?} -> {b:?}");
        assert!(a.0 > 0 || b.0 > 0, "minflt が 0 のまま (解析が怪しい)");
    }

    /// mincore の包みが、触ったページを常駐として数えること。
    #[test]
    fn resident_bytes_counts_touched_pages() {
        let n = 8 << 20;
        let mut v = vec![0u8; n];
        for i in (0..n).step_by(4096) {
            v[i] = 1;
        }
        let r = resident_bytes(v.as_ptr() as usize, n);
        assert!(
            r >= (n as u64) / 2,
            "mincore が常駐を数えていない: {r} / {n}"
        );
        assert_eq!(resident_bytes(0, n), 0);
        assert_eq!(resident_bytes(v.as_ptr() as usize, 0), 0);
    }

    /// drop のゲートが off なら何もしない (1 行も出さない)。
    #[test]
    fn drop_is_inert_without_the_gate() {
        if std::env::var(DROP_ENV).is_ok() {
            return; // ゲートを立てて走らせている環境では検査しない
        }
        assert!(drop_pages_if_enabled().is_empty());
    }
}
