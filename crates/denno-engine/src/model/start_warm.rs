//! task 243: 起動時のページキャッシュ温め (expert 行列)。
//!
//! 冷えたサーバの最初のリクエストは、モデル mmap のうち expert 行列
//! (lab の実測で 47.5 GiB) を SSD から読み直す。int-auto 478411fc8 の lab
//! t478 (done/tint_478411fc8) の実測:
//!
//! - 最初の COND の `c2000_cold` (3,826 トークン): `[exp-pf-stream] copy_s` が
//!   10.24 s。2 番目以降の COND は 1.90〜2.77 s。差はほぼ全部 page cache。
//! - 最初の COND の最初の decode (`a1`): `[dec-up-w] dma` が 1,731 ms。
//!   2 番目以降は 563 ms。
//! - 起動直後の 1 本 (lab が捨てる予備リクエスト): 12.8 s 対 6.2 s。
//!
//! ここでは `listening` の前に expert テンソルのバイト範囲を `pread` で読み、
//! page cache に載せる。`madvise(MADV_WILLNEED)` ではなく同期の `pread` に
//! したのは、**戻った時点で載っていることを保証する**ため (WILLNEED は助言で、
//! 戻った時点では I/O はまだ終わっていない)。読むだけで、数値経路は変えない。
//!
//! ゲート:
//! - `DENNO_START_PREFETCH` (`1`/`on`/`true` で有効、`0`/`off`/`false` で無効。
//!   未設定は [`DEFAULT_ON`])。
//! - `DENNO_START_PREFETCH_SECS` (既定 [`DEFAULT_SECS`] 秒)。時間で打ち切る
//!   (起動の延びを 30 s 以内に収めるため)。
//! - `DENNO_START_PREFETCH_THREADS` (既定 [`DEFAULT_THREADS`])。
//! - `DENNO_START_PREFETCH_EXTRA_MB` (既定 4096): expert 以外に温める上限。
//!   0 で expert だけ。

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 有効・無効の env 名。
pub const ENV_ON: &str = "DENNO_START_PREFETCH";
/// 打ち切り時間 (秒) の env 名。
pub const ENV_SECS: &str = "DENNO_START_PREFETCH_SECS";
/// 読みスレッド数の env 名。
pub const ENV_THREADS: &str = "DENNO_START_PREFETCH_THREADS";
/// expert 以外に温める上限 (MiB) の env 名。0 で expert だけ。
pub const ENV_EXTRA_MB: &str = "DENNO_START_PREFETCH_EXTRA_MB";
/// **診断用**: 起動時にモデルの page cache を落とす env 名 (既定 off)。
///
/// 冷えた機械を lab で再現するためのもの。`posix_fadvise(POSIX_FADV_DONTNEED)`
/// は root を要さない (自分のファイルの clean page を落とすだけ)。
pub const ENV_DROP: &str = "DENNO_START_DROP_CACHE";

/// 未設定時の既定 (lab t243b の結果で切り替える)。
pub const DEFAULT_ON: bool = false;
/// 打ち切り時間の既定 (秒)。「起動は最大 30 s 増」の枠に収める。
pub const DEFAULT_SECS: f64 = 27.0;
/// 読みスレッド数の既定。lab t243b は 6 本で 2.2 GB/s しか出なかった (同じ機械で
/// エンジンの prefill は 14 本で 4.1 GB/s)。同時に飛ぶ I/O を増やす。
pub const DEFAULT_THREADS: usize = 12;
/// 1 回の `pread` の長さ。
pub const PIECE: u64 = 4 << 20;
/// expert 以外に温める上限の既定 (バイト)。
pub const DEFAULT_EXTRA_BUDGET: u64 = 4 << 30;

/// gate の文字列解釈 (純関数)。未設定は [`DEFAULT_ON`]。
pub fn enabled_from(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some("1") | Some("true") | Some("on") | Some("ON") => true,
        Some("0") | Some("false") | Some("off") | Some("OFF") => false,
        _ => DEFAULT_ON,
    }
}

/// 打ち切り時間の解釈 (純関数)。未設定・空・非数・0 以下は [`DEFAULT_SECS`]。
pub fn secs_from(v: Option<&str>) -> f64 {
    match v.map(str::trim).and_then(|s| s.parse::<f64>().ok()) {
        Some(x) if x.is_finite() && x > 0.0 => x.min(3600.0),
        _ => DEFAULT_SECS,
    }
}

/// スレッド数の解釈 (純関数)。未設定・空・非数・0 は [`DEFAULT_THREADS`]。
pub fn threads_from(v: Option<&str>) -> usize {
    match v.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(x) if x > 0 => x.min(32),
        _ => DEFAULT_THREADS,
    }
}

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| enabled_from(std::env::var(ENV_ON).ok().as_deref()))
}

/// 打ち切り時間 (env を 1 回だけ読む)。
pub fn budget_secs() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| secs_from(std::env::var(ENV_SECS).ok().as_deref()))
}

/// 読みスレッド数 (env を 1 回だけ読む)。
pub fn read_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| threads_from(std::env::var(ENV_THREADS).ok().as_deref()))
}

/// expert 以外に温める上限バイト (env を 1 回だけ読む)。0 で expert だけ。
pub fn extra_budget() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        match std::env::var(ENV_EXTRA_MB)
            .ok()
            .as_deref()
            .map(str::trim)
            .and_then(|s| s.parse::<u64>().ok())
        {
            Some(mb) => mb.saturating_mul(1 << 20),
            None => DEFAULT_EXTRA_BUDGET,
        }
    })
}

/// `DENNO_START_DROP_CACHE` (診断用)。既定 off。
pub fn drop_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| enabled_from_env(ENV_DROP))
}

/// 指定した env が真か (未設定は偽)。
fn enabled_from_env(k: &str) -> bool {
    matches!(
        std::env::var(k).ok().as_deref().map(str::trim),
        Some("1") | Some("true") | Some("on") | Some("ON")
    )
}

/// モデルの全パートの page cache を落とす (診断用)。1 行のログを返す。
///
/// `mmap` は残したまま、clean な page だけを落とす。まだ触っていないテンソルは
/// PTE が無いので落ちる。落としたあとの最初のリクエストは SSD から読む
/// (= 冷えた機械と同じ)。
pub fn drop_cache(parts: &[PathBuf]) -> String {
    let mut ok = 0usize;
    let mut err = 0usize;
    for p in parts {
        match fadvise_dontneed(p) {
            Ok(()) => ok += 1,
            Err(_) => err += 1,
        }
    }
    format!("[start-prefetch] drop cache: ok={ok} err={err} (診断用 / {ENV_DROP}=1)")
}

#[cfg(unix)]
fn fadvise_dontneed(path: &std::path::Path) -> std::io::Result<()> {
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
    let f = std::fs::File::open(path)?;
    // 戻り値は errno そのもの (0 が成功)。len=0 は「ファイル末尾まで」。
    let rc = unsafe { posix_fadvise(f.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::from_raw_os_error(rc))
    }
}

#[cfg(not(unix))]
fn fadvise_dontneed(_path: &std::path::Path) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "posix_fadvise は unix のみ",
    ))
}

/// 1 つのバイト範囲: (パート番号, ファイル内オフセット, 長さ)。
pub type Range = (usize, u64, u64);

/// expert テンソル (`*_exps.weight`) の範囲を **ファイル順** に集める。
///
/// `lookup` はテンソル名 → 範囲 (`GgufFile::tensor_mapping_parts` をそのまま
/// 包めるようにしてある)。GGUF を書かないテストのために分離した。
pub fn expert_ranges<'a, I, F>(tensor_order: I, mut lookup: F) -> Vec<Range>
where
    I: IntoIterator<Item = &'a str>,
    F: FnMut(&str) -> Option<Range>,
{
    let mut out = Vec::new();
    for name in tensor_order {
        if !name.ends_with("_exps.weight") {
            continue;
        }
        if let Some(r) = lookup(name) {
            if r.2 > 0 {
                out.push(r);
            }
        }
    }
    out
}

/// expert 以外で **リクエスト中に mmap から読まれる** テンソルの範囲。
///
/// lab t243b の実測: expert だけ温めても、最初の short decode の `ple` が
/// 334.7 ms (base は 2.8 ms)、`c2000_cold` の prefill の `other` (残差 = 待ち) が
/// 1296 ms (base 458 ms) 残った。中身は PLE の表と、ホスト側で読む F32/F16/BF16 の
/// 重み (vram budget の `host_weights=1587 MiB`)。
///
/// 選び方: expert でなく、かつ
/// - ホスト型 (F32 / F16 / BF16。`is_requantized_to_q8` でも `is_device_weight`
///   でもない) のテンソル、または
/// - 名前が `ple` を含む / `token_embd.weight` / `output.weight` (語彙表)
///
/// を `budget` バイトまで **小さい順** に取る (大きい未使用テンソルで予算を
/// 食わないため)。`meta` はテンソル名 → (型, バイト数)。
pub fn extra_ranges<'a, I, M, F>(
    tensor_order: I,
    mut meta: M,
    mut lookup: F,
    budget: u64,
) -> Vec<Range>
where
    I: IntoIterator<Item = &'a str>,
    M: FnMut(&str) -> Option<(crate::kquant::GgmlType, u64)>,
    F: FnMut(&str) -> Option<Range>,
{
    let mut cand: Vec<(u64, usize, String, Range)> = Vec::new();
    for (i, name) in tensor_order.into_iter().enumerate() {
        if name.ends_with("_exps.weight") {
            continue;
        }
        let Some((ty, nbytes)) = meta(name) else {
            continue;
        };
        let host = !crate::gpu_budget::is_device_weight(ty)
            && !crate::gpu_budget::is_requantized_to_q8(ty);
        let named = name.contains("ple")
            || name == "per_layer_token_embd.weight"
            || name == "token_embd.weight"
            || name == "output.weight"
            || name == "output_norm.weight";
        if !host && !named {
            continue;
        }
        if let Some(r) = lookup(name) {
            if r.2 > 0 {
                cand.push((nbytes, i, name.to_string(), r));
            }
        }
    }
    cand.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut out = Vec::new();
    let mut used = 0u64;
    for (n, _, _, r) in cand {
        if used + n > budget {
            continue;
        }
        used += n;
        out.push(r);
    }
    out
}

/// 範囲を [`PIECE`] バイトに割る (端数はそのまま)。
pub fn split_pieces(ranges: &[Range]) -> Vec<Range> {
    let mut out = Vec::new();
    for &(part, off, len) in ranges {
        let mut done = 0u64;
        while done < len {
            let n = (len - done).min(PIECE);
            out.push((part, off + done, n));
            done += n;
        }
    }
    out
}

/// 温めの結果 (1 行のログに使う)。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    /// 読んだバイト数
    pub bytes: u64,
    /// 計画したバイト数
    pub planned: u64,
    /// 計画したピース数
    pub pieces: usize,
    /// 打ち切ったか
    pub stopped: bool,
    /// 失敗したピース数 (open できない等)
    pub failed: usize,
    /// 所要秒
    pub secs: f64,
}

impl Report {
    /// 1 行のログ。数値は `start-prefetch` の接頭辞で出す。
    pub fn line(&self) -> String {
        let gbs = if self.secs > 0.0 {
            self.bytes as f64 / self.secs / 1e9
        } else {
            0.0
        };
        format!(
            "[start-prefetch] experts {:.1} MB / {:.1} MB in {:.2}s ({:.1} GB/s) pieces={} failed={} stopped={}",
            self.bytes as f64 / 1e6,
            self.planned as f64 / 1e6,
            self.secs,
            gbs,
            self.pieces,
            self.failed,
            self.stopped,
        )
    }
}

/// 範囲を `pread` で読んで page cache に載せる。
///
/// 失敗しても起動は止めない (読めなかったピースを数えて返す)。`secs` を
/// 超えたら残りを捨てる (起動時間の上限)。
pub fn warm(parts: &[PathBuf], ranges: &[Range], secs: f64, threads: usize) -> Report {
    let pieces = split_pieces(ranges);
    let planned: u64 = ranges.iter().map(|r| r.2).sum();
    let mut rep = Report {
        planned,
        pieces: pieces.len(),
        ..Default::default()
    };
    if pieces.is_empty() || secs <= 0.0 {
        return rep;
    }
    // パートを開く。開けないパートの範囲は落とす (fail-open)。
    let mut files: Vec<Option<Arc<std::fs::File>>> = Vec::with_capacity(parts.len());
    for p in parts {
        match std::fs::File::open(p) {
            Ok(f) => files.push(Some(Arc::new(f))),
            Err(_) => files.push(None),
        }
    }
    let t0 = Instant::now();
    let deadline = t0 + Duration::from_secs_f64(secs);
    let stop = AtomicBool::new(false);
    let bytes = AtomicU64::new(0);
    let failed = AtomicU64::new(0);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .thread_name(|i| format!("start-prefetch-{i}"))
        .build();
    // 1 ピース (4 MiB) を 1 回の `pread` で読む。バッファはスレッドごとに
    // 1 回だけ確保する (ピースごとに `vec![0u8; ..]` を作るとゼロ埋めだけで
    // 読み量と同じ帯域を食う)。
    let one = |piece: &Range, buf: &mut [u8]| -> bool {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        if Instant::now() >= deadline {
            stop.store(true, Ordering::Relaxed);
            return false;
        }
        let (part, off, len) = *piece;
        let Some(Some(f)) = files.get(part) else {
            failed.fetch_add(1, Ordering::Relaxed);
            return true;
        };
        let want = (len as usize).min(buf.len());
        let mut left = want;
        let mut at = off;
        let mut done = 0usize;
        while left > 0 {
            match read_at(f, &mut buf[done..done + left], at) {
                Ok(0) | Err(_) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    break;
                }
                Ok(got) => {
                    bytes.fetch_add(got as u64, Ordering::Relaxed);
                    at += got as u64;
                    done += got;
                    left -= got;
                }
            }
        }
        true
    };
    match &pool {
        Ok(p) => {
            use rayon::prelude::*;
            p.install(|| {
                pieces
                    .par_iter()
                    .map_init(|| vec![0u8; PIECE as usize], |buf, pc| one(pc, buf))
                    .for_each(|_| {});
            });
        }
        Err(_) => {
            // プールが作れないときは 1 本で回す (機能は落とさない)。
            let mut buf = vec![0u8; PIECE as usize];
            for pc in &pieces {
                if !one(pc, &mut buf) {
                    break;
                }
            }
        }
    }
    rep.bytes = bytes.load(Ordering::Relaxed);
    rep.failed = failed.load(Ordering::Relaxed) as usize;
    rep.stopped = stop.load(Ordering::Relaxed);
    rep.secs = t0.elapsed().as_secs_f64();
    rep
}

/// `pread` の薄い包み (unix 以外は seek + read で代用)。
#[cfg(unix)]
fn read_at(f: &std::fs::File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    use std::os::unix::fs::FileExt;
    f.read_at(buf, off)
}

// ---------------------------------------------------------------------------
// task 349: 起動後の**非同期**温め
// ---------------------------------------------------------------------------
//
// 既存の [`warm`] は `listening` の前に同期で走る (task 243) ので、起動が
// そのぶん遅れる。共通ルールで「warming at start」は不採用になっている。
// ここでは**リクエストを 1 バイトも待たせない**形にする: `listening` のあとに
// 1 本のスレッドを起こし、page cache を読むだけ (戻り値は捨てる)。
//
// 冷えた page cache の最初の 14k prefill (lab t343a の x0) の内訳:
// `other` 4,308 ms (base 830 ms) と `experts` +1.2 s。`other` の H2D は
// 4,128.72 MB / 4,110 回で、cold のときだけ 3.5 s かかる = mmap 由来の
// ホスト重み (vram budget の `host_weights=1587 MiB`) と PLE の表を
// 毎リクエスト読み直している。だから **extra_ranges を先に**、そのあと
// expert を `DENNO_PF_BG_WARM_MB` まで読む (後から読むものほど LRU で残る)。

/// 非同期温めの env 名 (既定 off)。
pub const ENV_BG: &str = "DENNO_PF_BG_WARM";
/// 非同期温めで読む expert の上限 (MiB)。
pub const ENV_BG_MB: &str = "DENNO_PF_BG_WARM_MB";
/// 非同期温めの打ち切り時間 (秒)。
pub const ENV_BG_SECS: &str = "DENNO_PF_BG_WARM_SECS";
/// 非同期温めのスレッド数。
pub const ENV_BG_THREADS: &str = "DENNO_PF_BG_WARM_THREADS";

/// 既定 off (lab で測ってから決める)。
pub const DEFAULT_BG: bool = false;
/// expert の上限の既定 (24 GiB)。ホストは 59 GiB で RSS が 15-20 GB あるため、
/// expert 28.38 GB + extra 4 GB が page cache に載る上限に近い。
pub const DEFAULT_BG_MB: usize = 24 * 1024;
/// 打ち切り時間の既定 (秒)。最初の prefill が終わるまでに読み終わる長さ。
pub const DEFAULT_BG_SECS: f64 = 40.0;
/// スレッド数の既定。task 325 の pread は 14 本で 5.8 GB/s、32 本で 6.2 GB/s。
pub const DEFAULT_BG_THREADS: usize = 8;

/// [`ENV_BG`] の解釈 (純関数)。
pub fn bg_from(v: Option<&str>) -> bool {
    enabled_from(v)
}

/// [`ENV_BG_MB`] の解釈 (純関数)。0 = expert を読まない。0..=131072 に丸める。
pub fn bg_mb_from(v: Option<&str>) -> usize {
    match v.map(str::trim).filter(|s| !s.is_empty()) {
        Some(s) => match s.parse::<usize>() {
            Ok(n) => n.min(131072),
            Err(_) => DEFAULT_BG_MB,
        },
        None => DEFAULT_BG_MB,
    }
}

/// [`ENV_BG_SECS`] の解釈 (純関数)。[`secs_from`] と同じ (1..=3600 秒)。
pub fn bg_secs_from(v: Option<&str>) -> f64 {
    secs_from(v)
}

/// [`ENV_BG_THREADS`] の解釈 (純関数)。1..=32 本。
pub fn bg_threads_from(v: Option<&str>) -> usize {
    threads_from(v)
}

/// 非同期温めが有効か (プロセスで 1 回だけ env を読む)。
pub fn bg_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| bg_from(std::env::var(ENV_BG).ok().as_deref()))
}

/// expert の上限バイト (env を 1 回だけ読む)。
pub fn bg_mb() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| bg_mb_from(std::env::var(ENV_BG_MB).ok().as_deref()))
}

/// 打ち切り時間 (env を 1 回だけ読む)。
pub fn bg_secs() -> f64 {
    static V: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *V.get_or_init(|| bg_secs_from(std::env::var(ENV_BG_SECS).ok().as_deref()))
}

/// スレッド数 (env を 1 回だけ読む)。
pub fn bg_threads() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| bg_threads_from(std::env::var(ENV_BG_THREADS).ok().as_deref()))
}

/// 範囲の列を**先頭から** `max` バイトまでに切る (純関数)。
///
/// 1 つのテンソルは丸ごと残す (途中で切ると中途半端なページを温める)。
/// 入らないテンソルに当たったらそこで止める (ファイル順に読むので、飛ばすと
/// シークになるだけ)。
pub fn cap_ranges(ranges: &mut Vec<Range>, max: u64) {
    let mut used = 0u64;
    let mut keep = Vec::with_capacity(ranges.len());
    for &r in ranges.iter() {
        if used + r.2 > max {
            break;
        }
        used += r.2;
        keep.push(r);
    }
    *ranges = keep;
}

/// 背景の温めが読んだ合計バイト (診断とテスト用)。
static BG_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`BG_BYTES`] を読む (背景スレッドが 1 本終わるたびに足される)。
pub fn bg_bytes() -> u64 {
    BG_BYTES.load(std::sync::atomic::Ordering::Relaxed)
}

/// 背景で page cache を温める (戻りを待たない)。
///
/// 失敗しても何も壊さない: 読むだけ、`warm` の中は fail open、スレッドが
/// 作れなければ `false` を返して終わり。数値経路には触らない。
pub fn spawn_background(
    parts: Vec<PathBuf>,
    ranges: Vec<Range>,
    secs: f64,
    threads: usize,
) -> bool {
    if ranges.is_empty() || secs <= 0.0 {
        return false;
    }
    let planned: u64 = ranges.iter().map(|r| r.2).sum();
    std::thread::Builder::new()
        .name("pf-bg-warm".into())
        .spawn(move || {
            let rep = warm(&parts, &ranges, secs, threads);
            BG_BYTES.fetch_add(rep.bytes, std::sync::atomic::Ordering::Relaxed);
            eprintln!("[pf-bg-warm] {}", rep.line());
        })
        .map(|_| {
            eprintln!(
                "[pf-bg-warm] start {:.1} MB ({} secs, {} threads)",
                planned as f64 / 1e6,
                secs,
                threads
            );
            true
        })
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn read_at(_f: &std::fs::File, _buf: &mut [u8], _off: u64) -> std::io::Result<usize> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "pread は unix のみ",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_strings_are_explicit() {
        // 既定 off のあいだは未設定で false。`0` は明示の off。
        assert_eq!(enabled_from(Some("1")), true);
        assert_eq!(enabled_from(Some(" on ")), true);
        assert_eq!(enabled_from(Some("0")), false);
        assert_eq!(enabled_from(Some("off")), false);
        assert_eq!(enabled_from(Some("x")), DEFAULT_ON);
        assert_eq!(enabled_from(None), DEFAULT_ON);
    }

    #[test]
    fn secs_and_threads_are_clamped() {
        assert_eq!(secs_from(None), DEFAULT_SECS);
        assert_eq!(secs_from(Some("")), DEFAULT_SECS);
        assert_eq!(secs_from(Some("0")), DEFAULT_SECS);
        assert_eq!(secs_from(Some("-3")), DEFAULT_SECS);
        assert!((secs_from(Some("3.5")) - 3.5).abs() < 1e-9);
        assert_eq!(threads_from(None), DEFAULT_THREADS);
        assert_eq!(threads_from(Some("0")), DEFAULT_THREADS);
        assert_eq!(threads_from(Some("3")), 3);
        assert_eq!(threads_from(Some("999")), 32);
    }

    #[test]
    fn expert_ranges_take_only_expert_tensors_in_order() {
        let order = [
            "blk.0.ffn_gate_exps.weight",
            "blk.0.attn_norm.weight",
            "blk.1.ffn_down_exps.weight",
            "blk.1.ffn_up_exps.weight",
        ];
        let r = expert_ranges(order.iter().copied(), |n| match n {
            "blk.0.ffn_gate_exps.weight" => Some((0, 100, 10)),
            "blk.1.ffn_down_exps.weight" => Some((1, 200, 20)),
            "blk.1.ffn_up_exps.weight" => Some((1, 300, 0)), // 長さ 0 は落とす
            _ => None,
        });
        assert_eq!(r, vec![(0, 100, 10), (1, 200, 20)]);
    }

    #[test]
    fn pieces_split_at_piece_size() {
        let r = vec![(0, 0u64, PIECE), (0, PIECE, PIECE + 5), (1, 7, 3)];
        let p = split_pieces(&r);
        assert_eq!(p.len(), 4);
        assert_eq!(p[0], (0, 0, PIECE));
        assert_eq!(p[1], (0, PIECE, PIECE));
        assert_eq!(p[2], (0, 2 * PIECE, 5));
        assert_eq!(p[3], (1, 7, 3));
    }

    /// 実ファイルを温める: 計画したバイト数が読め、内容は変わらない。
    #[test]
    fn warm_reads_the_planned_bytes() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "denno_start_warm_{}_{}.bin",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        ));
        let n = 3 * PIECE + 17;
        let data: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let parts = vec![path.clone()];
        let ranges = vec![(0usize, 0u64, data.len() as u64)];
        let rep = warm(&parts, &ranges, 30.0, 2);
        assert_eq!(rep.bytes, data.len() as u64);
        assert_eq!(rep.planned, data.len() as u64);
        assert_eq!(rep.failed, 0);
        assert!(!rep.stopped);
        assert!(rep.line().starts_with("[start-prefetch] experts "));
        // 読んだあとも内容は無傷
        let back = std::fs::read(&path).unwrap();
        assert_eq!(back, data);
        let _ = std::fs::remove_file(&path);
    }

    /// 範囲が無い・時間 0 は何もせずに返る。
    #[test]
    fn warm_is_inert_without_ranges() {
        let rep = warm(&[], &[], 30.0, 2);
        assert_eq!(rep.bytes, 0);
        assert_eq!(rep.pieces, 0);
    }

    /// task 349: 非同期温めの env の解釈 (既定 off、`0` 系だけ偽)。
    #[test]
    fn bg_gate_strings_and_bounds() {
        assert_eq!(bg_from(None), DEFAULT_BG);
        assert!(!bg_from(None), "既定は off");
        assert!(bg_from(Some("1")));
        assert!(bg_from(Some(" on ")));
        assert!(!bg_from(Some("0")));
        assert!(!bg_from(Some("off")));
        assert_eq!(bg_mb_from(None), DEFAULT_BG_MB);
        assert_eq!(bg_mb_from(Some("")), DEFAULT_BG_MB);
        assert_eq!(bg_mb_from(Some("0")), 0, "0 = expert を読まない");
        assert_eq!(bg_mb_from(Some("512")), 512);
        assert_eq!(bg_mb_from(Some("999999")), 131072);
        assert_eq!(bg_mb_from(Some("x")), DEFAULT_BG_MB);
        assert_eq!(bg_secs_from(None), DEFAULT_SECS);
        assert_eq!(bg_threads_from(Some("4")), 4);
    }

    /// task 349: `cap_ranges` は先頭から丸ごと残し、入らない所で止める。
    #[test]
    fn cap_ranges_keeps_whole_tensors_from_the_front() {
        let mut r = vec![(0usize, 0u64, 10u64), (0, 10, 20), (0, 30, 5)];
        cap_ranges(&mut r, 30);
        assert_eq!(r, vec![(0, 0, 10), (0, 10, 20)], "30 バイトちょうどまで");
        let mut r2 = vec![(0usize, 0u64, 10u64), (0, 10, 20)];
        cap_ranges(&mut r2, 29);
        assert_eq!(r2, vec![(0, 0, 10)], "途中では切らない");
        let mut r3 = vec![(0usize, 0u64, 10u64)];
        cap_ranges(&mut r3, 0);
        assert!(r3.is_empty());
    }

    /// task 349: 背景の温めは範囲が空ならスレッドを作らず false を返す。
    #[test]
    fn spawn_background_is_inert_without_ranges() {
        assert!(!spawn_background(Vec::new(), Vec::new(), 5.0, 2));
        assert!(!spawn_background(vec![PathBuf::from("/dev/null")], Vec::new(), 5.0, 2));
    }

    /// task 349: 背景の温めが**実際に読む** (発火の証拠)。戻りは待たず、
    /// 台帳 (`bg_bytes`) が計画したバイト数に達する。
    #[test]
    fn spawn_background_reads_the_planned_bytes() {
        let path = std::env::temp_dir().join(format!(
            "denno_bg_warm_{}_{}.bin",
            std::process::id(),
            Instant::now().elapsed().as_nanos()
        ));
        let n = PIECE + 123;
        let data: Vec<u8> = (0..n).map(|i| (i % 241) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let before = bg_bytes();
        assert!(spawn_background(
            vec![path.clone()],
            vec![(0usize, 0u64, data.len() as u64)],
            30.0,
            2
        ));
        // 背景なので、戻った時点ではまだ読んでいないことがある。
        let t0 = Instant::now();
        while bg_bytes() == before && t0.elapsed() < Duration::from_secs(10) {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            bg_bytes() >= before + data.len() as u64,
            "背景の温めが 1 バイトも読んでいない"
        );
        assert_eq!(std::fs::read(&path).unwrap(), data, "中身が変わった");
        let _ = std::fs::remove_file(&path);
    }

    /// expert 以外はホスト型 (F32/BF16) と名前付きの表だけを、小さい順に予算まで。
    #[test]
    fn extra_ranges_take_host_types_and_tables() {
        use crate::kquant::GgmlType;
        let order = [
            "blk.0.ffn_gate_exps.weight",  // expert: 除外
            "blk.0.attn_q.weight",         // Q8_0: デバイス型で名前も違う → 除外
            "ple_key.weight",              // BF16 + 名前一致
            "blk.0.attn_norm.weight",      // F32
            "token_embd.weight",           // 名前一致
            "per_layer_token_embd.weight", // PLE 表 (IQ4_NL) 名前一致
            "blk.0.other_iq.weight",       // IQ3_S: ホスト型でない・名前も違う → 除外
        ];
        let meta = |n: &str| -> Option<(GgmlType, u64)> {
            Some(match n {
                "blk.0.ffn_gate_exps.weight" => (GgmlType::Q4K, 100),
                "blk.0.attn_q.weight" => (GgmlType::Q8_0, 200),
                "ple_key.weight" => (GgmlType::Bf16, 300),
                "blk.0.attn_norm.weight" => (GgmlType::F32, 20),
                "token_embd.weight" => (GgmlType::Q8_0, 400),
                "per_layer_token_embd.weight" => (GgmlType::IQ4Nl, 250),
                _ => (GgmlType::IQ3S, 500),
            })
        };
        let lookup = |n: &str| -> Option<Range> { Some((0, n.len() as u64, 8)) };
        let all = extra_ranges(order.iter().copied(), meta, lookup, 10_000);
        // 小さい順: attn_norm(20), per_layer_token_embd(250), ple_key(300), token_embd(400)
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].1, "blk.0.attn_norm.weight".len() as u64);
        assert_eq!(all[1].1, "per_layer_token_embd.weight".len() as u64);
        // 予算 350 なら attn_norm と per_layer_token_embd だけ
        let some = extra_ranges(order.iter().copied(), meta, lookup, 350);
        assert_eq!(some.len(), 2);
        // 予算 0 なら空
        assert!(extra_ranges(order.iter().copied(), meta, lookup, 0).is_empty());
    }

    /// 診断用の drop は実ファイルで成功し、無いパスは err に数える。
    #[test]
    fn drop_cache_reports_ok_and_err() {
        let dir = std::env::temp_dir();
        let ok_path = dir.join(format!("denno_start_drop_{}.bin", std::process::id()));
        std::fs::write(&ok_path, b"hello").unwrap();
        let missing = dir.join("denno_start_drop_missing_243.bin");
        let line = drop_cache(&[ok_path.clone(), missing]);
        assert!(line.contains("ok=1 err=1"), "{line}");
        assert!(line.starts_with("[start-prefetch] drop cache: "), "{line}");
        let _ = std::fs::remove_file(&ok_path);
    }
}
