//! task 862: 長いプロンプトの prefill がホストの page cache を食い潰さないための助言。
//!
//! # なぜ要るか
//!
//! 59 GiB のホストで 104 GB の GGUF (UD-Q4_K_XL) を回すと、14k トークンの
//! コールドな prefill が expert の実データ約 52 GB を page cache に載せる。
//! その過程で **直前の decode が温めていた expert** が追い出され、次の decode が
//! ディスクへ落ちて 1 トークン 100-255 ms の山になる (lab t794b の round 2)。
//!
//! # 何をするか
//!
//! `DENNO_EXPERT_PAGE_HINT=1` のとき、prefill の終端で 2 つを投げる。
//!
//! 1. **冷たい範囲を落とす**: prefill が選んだ expert のうち
//!    (a) プロンプト末尾 [`keep_tok`] トークンに選ばれず、かつ
//!    (b) decode の直近 [`dec_tok`] トークンにも選ばれていない
//!    ものを `madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)` で
//!    page cache から落とす。
//! 2. **温かい範囲を戻す**: (b) の decode の作業集合のうち prefill が触った
//!    ものは、落とした後に `mincore(2)` で常駐を確かめ、**載っていない範囲だけ**
//!    `madvise(MADV_WILLNEED)` で読み直す (ディスクから 1-2 GB 程度)。
//!
//! 落とす単位は expert のバイト範囲で、連続する expert id は 1 回の syscall に
//! まとめる。
//!
//! # 落とし方 (ここが要点)
//!
//! このホスト (kernel 7.0) で実測した挙動:
//!
//! | 操作 | プロセスの PTE | page cache |
//! | --- | --- | --- |
//! | `madvise(MADV_DONTNEED)` (共有ファイル写像) | 外れる | **残る** |
//! | `posix_fadvise(POSIX_FADV_DONTNEED)` (写像中) | - | **残る** (写像中は無効化しない) |
//! | 上の 2 つをこの順に | 外れる | **落ちる** |
//! | `madvise(MADV_COLD)` / `MADV_PAGEOUT` | - | その場では落ちない |
//!
//! つまり `madvise(MADV_DONTNEED)` だけでは 1 バイトも解放されない。
//! [`drop_range`] は PTE を外してから `posix_fadvise` で page cache を無効化する。
//! fd はパートごとに 1 本だけ開いて使い回す ([`register_map`] が mmap の指す先の
//! パスを覚える。`memmap2::Mmap` は fd を持たないため)。
//!
//! # 数値への影響
//!
//! **無し**。ページを落としても写像のバイト列は同じで、次に触れば同じ
//! ファイル位置から読み直される。`MADV_WILLNEED` も先に読むだけ (助言)。
//! 既定 off。off では [`note_selection`] / [`note_decode`] が atomic load 1 回で
//! 戻り、確保もしない。

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use memmap2::Mmap;

/// `DENNO_EXPERT_PAGE_HINT`: prefill の冷たい expert 範囲のページを落とす。
pub const ENV: &str = "DENNO_EXPERT_PAGE_HINT";

/// `DENNO_EXPERT_PAGE_HINT_KEEP_TOK`: prefill の末尾で残すトークン数 (既定 256)。
pub const KEEP_TOK_ENV: &str = "DENNO_EXPERT_PAGE_HINT_KEEP_TOK";

/// `DENNO_EXPERT_PAGE_HINT_DEC_TOK`: decode の履歴から残すトークン数 (既定 256)。
pub const DEC_TOK_ENV: &str = "DENNO_EXPERT_PAGE_HINT_DEC_TOK";

/// `DENNO_EXPERT_PAGE_HINT_MIN_TOK`: これ未満のプロンプトでは何もしない (既定 2048)。
///
/// 512 トークンでも 1 層あたり数千本の選択があり、ほぼ全部の expert に触る。
/// 普通のチャットのプロンプトで 40 GB を捨てると、そのあとの decode が
/// ディスクへ落ちるので、長いプロンプト (3,826 / 14k) だけを対象にする。
pub const MIN_TOK_ENV: &str = "DENNO_EXPERT_PAGE_HINT_MIN_TOK";

/// `DENNO_EXPERT_PAGE_HINT_WARM_MB`: 読み直す上限 (MiB、既定 2048。0 = 無制限)。
///
/// 温かい集合はモデル次第で 10 GB を超える。上限を切らないと prefill の後ろに
/// 読み直しの秒数が乗るので、既定で 2 GiB に切る。
pub const WARM_MB_ENV: &str = "DENNO_EXPERT_PAGE_HINT_WARM_MB";

/// [`WARM_MB_ENV`] の既定値 (MiB)。
pub const DEFAULT_WARM_MB: u64 = 2048;

/// `DENNO_EXPERT_PAGE_HINT_WILLNEED`: 温かい範囲を読み直すか (既定 on)。
pub const WILLNEED_ENV: &str = "DENNO_EXPERT_PAGE_HINT_WILLNEED";

/// [`KEEP_TOK_ENV`] の既定値。
pub const DEFAULT_KEEP_TOK: usize = 256;

/// [`KEEP_TOK_ENV`] / [`DEC_TOK_ENV`] の上限 (これを超える指定は丸める)。
pub const TOK_MAX: usize = 4096;

/// [`DEC_TOK_ENV`] の既定値。
///
/// 256 だと decode の作業集合が数 GB になり、`MADV_WILLNEED` の読み直しが
/// prefill の後ろに数秒乗る。直近 64 トークンで十分な代理になる。
pub const DEFAULT_DEC_TOK: usize = 64;

/// [`MIN_TOK_ENV`] の既定値。
pub const DEFAULT_MIN_TOK: usize = 2048;

/// [`ENV`] の解釈 (純関数)。**明示的な真値だけ on** (既定 off)。
pub fn on_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn enabled() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| on_from_value(std::env::var(ENV).ok().as_deref()))
}

/// テスト用: ゲートを固定する (`None` で env に戻す)。
pub fn set_enabled_for_tests(v: Option<bool>) {
    OVERRIDE.store(
        match v {
            Some(true) => 2,
            Some(false) => 1,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// [`WILLNEED_ENV`] の解釈 (純関数)。既定 on、明示的な偽値だけ off。
pub fn willneed_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// 温かい範囲を読み直すか (プロセスで 1 回だけ env を読む)。
pub fn willneed_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| willneed_from_value(std::env::var(WILLNEED_ENV).ok().as_deref()))
}

/// トークン数の env を読む (純関数)。0・解析不能は既定、上限で丸める。
fn tok_from_value(v: Option<&str>, default: usize) -> usize {
    match v.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(n) if n > 0 => n.min(TOK_MAX),
        _ => default,
    }
}

/// [`KEEP_TOK_ENV`] の解釈。
pub fn keep_tok_from_value(v: Option<&str>) -> usize {
    tok_from_value(v, DEFAULT_KEEP_TOK)
}

/// 残す末尾トークン数 (prefill)。
pub fn keep_tok() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| keep_tok_from_value(std::env::var(KEEP_TOK_ENV).ok().as_deref()))
}

/// [`DEC_TOK_ENV`] の解釈。
pub fn dec_tok_from_value(v: Option<&str>) -> usize {
    tok_from_value(v, DEFAULT_DEC_TOK)
}

/// 残す decode 履歴のトークン数。
pub fn dec_tok() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| dec_tok_from_value(std::env::var(DEC_TOK_ENV).ok().as_deref()))
}

/// [`MIN_TOK_ENV`] の解釈 (純関数)。0 は明示的な「常に」、解析不能は既定。
pub fn min_tok_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim).and_then(|s| s.parse::<usize>().ok()) {
        Some(n) => n,
        None => DEFAULT_MIN_TOK,
    }
}

/// [`WARM_MB_ENV`] の解釈 (純関数)。解析不能は既定、0 は無制限。
pub fn warm_mb_from_value(v: Option<&str>) -> u64 {
    match v.map(str::trim).and_then(|s| s.parse::<u64>().ok()) {
        Some(n) => n,
        None => DEFAULT_WARM_MB,
    }
}

/// 読み直す上限 (MiB。0 = 無制限)。
pub fn warm_mb() -> u64 {
    static N: OnceLock<u64> = OnceLock::new();
    *N.get_or_init(|| warm_mb_from_value(std::env::var(WARM_MB_ENV).ok().as_deref()))
}

/// 読み直しの予算を使い切ったか (0 = 無制限)。
pub fn warm_budget_exhausted() -> bool {
    let cap = warm_mb();
    if cap == 0 {
        return false;
    }
    let used = STATS.with(|s| s.borrow().warm_bytes);
    used >= cap.saturating_mul(1024 * 1024)
}

/// 動かす最小プロンプト長。
pub fn min_tok() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| min_tok_from_value(std::env::var(MIN_TOK_ENV).ok().as_deref()))
}

// ---------------------------------------------------------------------------
// mmap のパス登録 (fd を開いて fadvise に使う)
// ---------------------------------------------------------------------------

/// mmap の先頭アドレス → 読み口。1 パート 1 本だけ開く。
fn fds() -> &'static Mutex<std::collections::HashMap<usize, std::fs::File>> {
    static FDS: OnceLock<Mutex<std::collections::HashMap<usize, std::fs::File>>> = OnceLock::new();
    FDS.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

/// `map` が指すファイルのパスを覚える (何度呼んでも 1 回しか開かない)。
///
/// 開けなければ黙って諦める (`drop_range` は madvise だけになり、page cache は
/// 落ちない = 実質 no-op。統計の `failed` に数える)。
pub fn register_map(map: &Arc<Mmap>, path: &Path) {
    let key = map.as_ptr() as usize;
    let mut g = match fds().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if g.contains_key(&key) {
        return;
    }
    if let Ok(f) = std::fs::File::open(path) {
        g.insert(key, f);
    }
}

fn fd_for(map: &Arc<Mmap>) -> Option<i32> {
    use std::os::fd::AsRawFd;
    let key = map.as_ptr() as usize;
    let g = match fds().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    g.get(&key).map(|f| f.as_raw_fd())
}

// ---------------------------------------------------------------------------
// 落とす / 先に読む (syscall)
// ---------------------------------------------------------------------------

/// ページサイズ (この実装は 4 KiB 固定で丸める)。
const PAGE: usize = 4096;

/// `posix_fadvise(2)` の `POSIX_FADV_DONTNEED`。
const POSIX_FADV_DONTNEED: i32 = 4;

/// `madvise(2)` の `MADV_DONTNEED` / `MADV_WILLNEED` (linux/mman-common.h)。
const MADV_DONTNEED: i32 = 4;
const MADV_WILLNEED: i32 = 3;

#[cfg(unix)]
extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
}

/// `map` の `[off, off+len)` を扱った結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropOutcome {
    /// 成功 (戻り値はページに丸めたバイト数)。
    Dropped(usize),
    /// ページに満たない / 範囲外なので何もしなかった。
    Skipped,
    /// syscall が失敗した (動作は変わらない)。
    Failed,
    /// `MADV_WILLNEED` を投げた (戻り値は丸めたバイト数)。
    Warmed(usize),
}

/// ページ境界へ**内側**に丸めた `[start, stop)` を返す (隣のテンソルを巻き込まない)。
fn inner_pages(map_len: usize, off: usize, len: usize) -> Option<(usize, usize)> {
    if len == 0 || off >= map_len {
        return None;
    }
    let end = off.saturating_add(len).min(map_len);
    let start = off.div_ceil(PAGE) * PAGE;
    let stop = (end / PAGE) * PAGE;
    (stop > start).then_some((start, stop))
}

/// `map` の `[off, off+len)` のページを **page cache から** 落とす。
///
/// 手順は「ページ境界へ内側に丸める → `madvise(MADV_DONTNEED)` で PTE を外す →
/// `posix_fadvise(POSIX_FADV_DONTNEED)` で page cache を無効化」。
/// 1 手目だけでは page cache は 1 ページも落ちない (上の表)。数値は変わらない。
pub fn drop_range(map: &Arc<Mmap>, off: usize, len: usize) -> DropOutcome {
    let Some((start, stop)) = inner_pages(map.len(), off, len) else {
        return DropOutcome::Skipped;
    };
    let n = stop - start;
    #[cfg(unix)]
    {
        let addr = unsafe { map.as_ptr().add(start) as *mut std::ffi::c_void };
        // SAFETY: `addr` は `map` の内側のページ境界で `n` は写像の末尾を超えない。
        // fadvise の範囲もファイル内の同じ位置を指す。
        let _ = unsafe { madvise(addr, n, MADV_DONTNEED) };
        let Some(fd) = fd_for(map) else {
            // 読み口が無いと page cache は落とせない (madvise 単独では PTE が
            // 外れるだけ)。失敗として数える。
            return DropOutcome::Failed;
        };
        let rc = unsafe { posix_fadvise(fd, start as i64, n as i64, POSIX_FADV_DONTNEED) };
        if rc != 0 {
            return DropOutcome::Failed;
        }
        DropOutcome::Dropped(n)
    }
    #[cfg(not(unix))]
    {
        let _ = (start, n);
        DropOutcome::Failed
    }
}

/// `map` の `[off, off+len)` に `MADV_WILLNEED` を投げる (非同期の読み直し依頼)。
///
/// 範囲は外側へ丸める (`madvise` はページ単位なので端は巻き込んでよい。
/// WILLNEED は読むだけ)。戻り値は投げたバイト数 (0 = 何もしない)。
pub fn willneed_range(map: &Arc<Mmap>, off: usize, len: usize) -> usize {
    let start = off & !(PAGE - 1);
    let map_len = map.len();
    if off >= map_len || len == 0 {
        return 0;
    }
    let mut stop = off.saturating_add(len).min(map_len);
    stop = (stop + PAGE - 1) & !(PAGE - 1);
    if stop <= start {
        return 0;
    }
    let n = stop - start;
    #[cfg(unix)]
    {
        let addr = unsafe { map.as_ptr().add(start) as *mut std::ffi::c_void };
        // SAFETY: `start` は写像の内側のページ境界で、`n` は写像の末尾を超えない。
        let rc = unsafe { madvise(addr, n, MADV_WILLNEED) };
        if rc != 0 {
            return 0;
        }
        n
    }
    #[cfg(not(unix))]
    {
        let _ = (start, stop);
        0
    }
}

// ---------------------------------------------------------------------------
// 選択の追跡
// ---------------------------------------------------------------------------

/// ビット列に 1 を立てる (語が足りなければ伸ばす)。
fn set_bit(bits: &mut Vec<u64>, i: usize) {
    let w = i / 64;
    if bits.len() <= w {
        bits.resize(w + 1, 0);
    }
    bits[w] |= 1u64 << (i % 64);
}

/// ビット列を `Vec<u16>` (昇順) にする。
fn bits_to_ids(bits: &[u64]) -> Vec<u16> {
    let mut out = Vec::new();
    for (w, &b) in bits.iter().enumerate() {
        let mut b = b;
        while b != 0 {
            let i = b.trailing_zeros() as usize;
            b &= b - 1;
            let e = w * 64 + i;
            if e <= u16::MAX as usize {
                out.push(e as u16);
            }
        }
    }
    out
}

/// 1 回の prefill ぶんの追跡状態。
///
/// `hot[layer]` は「この prefill で 1 回でも選ばれた expert」のビット列、
/// `tail[layer]` は直近 [`keep_tok`] トークンぶんの選択 (トークン順) を平らに持つ。
#[derive(Default)]
struct Track {
    /// 追跡中か ([`begin_prefill`] が立て、[`take_plan`] が下ろす)。
    active: bool,
    /// 残す末尾トークン数。
    keep: usize,
    /// 層ごとの全選択 (ビット列)。
    hot: Vec<Vec<u64>>,
    /// 層ごとの直近の選択 (トークン昇順・j 昇順)。
    tail: Vec<VecDeque<u16>>,
    /// 見たトークン数。
    tok: usize,
}

impl Track {
    fn note(&mut self, layer: usize, nt: usize, k: usize, sels: &[usize]) {
        if nt == 0 || k == 0 || sels.len() != nt * k {
            return;
        }
        if layer == 0 {
            self.tok += nt;
        }
        if self.hot.len() <= layer {
            self.hot.resize_with(layer + 1, Vec::new);
        }
        if self.tail.len() <= layer {
            self.tail.resize_with(layer + 1, VecDeque::new);
        }
        let cap = self.keep.saturating_mul(k).max(1);
        for t in 0..nt {
            for j in 0..k {
                let e = sels[t * k + j];
                set_bit(&mut self.hot[layer], e);
                self.tail[layer].push_back(e as u16);
            }
        }
        while self.tail[layer].len() > cap {
            self.tail[layer].pop_front();
        }
    }

    /// 層 `layer` の「末尾ウィンドウに載っていない選択」を返す。
    fn head_only(&self, layer: usize) -> Vec<u16> {
        let mut keep: Vec<u64> = Vec::new();
        if let Some(row) = self.tail.get(layer) {
            for &e in row.iter() {
                set_bit(&mut keep, e as usize);
            }
        }
        let mut out: Vec<u16> = Vec::new();
        for (w, &bits) in self.hot[layer].iter().enumerate() {
            let mut b = bits & !keep.get(w).copied().unwrap_or(0);
            while b != 0 {
                let i = b.trailing_zeros() as usize;
                b &= b - 1;
                let e = w * 64 + i;
                if e <= u16::MAX as usize {
                    out.push(e as u16);
                }
            }
        }
        out
    }
}

/// decode の履歴 (リクエストをまたいで残す「温かい作業集合」)。
///
/// 層ごとに直近 [`dec_tok`] トークンの選択を平らに持つ。prefill の落とし対象から
/// これを除外し、prefill で追い出されていたら `MADV_WILLNEED` で戻す。
#[derive(Default)]
struct DecHot {
    rings: Vec<VecDeque<u16>>,
    k: usize,
}

thread_local! {
    static TRACK: std::cell::RefCell<Track> = const { std::cell::RefCell::new(Track::new()) };
    static DEC: std::cell::RefCell<DecHot> = const { std::cell::RefCell::new(DecHot::new()) };
    static STATS: std::cell::RefCell<FlushStats> = const { std::cell::RefCell::new(FlushStats::new()) };
}

impl Track {
    const fn new() -> Self {
        Self {
            active: false,
            keep: 0,
            hot: Vec::new(),
            tail: Vec::new(),
            tok: 0,
        }
    }
}

impl DecHot {
    const fn new() -> Self {
        Self {
            rings: Vec::new(),
            k: 0,
        }
    }
}

impl FlushStats {
    const fn new() -> Self {
        Self {
            ranges: 0,
            bytes: 0,
            failed: 0,
            ms: 0.0,
            experts: 0,
            warm_ranges: 0,
            warm_bytes: 0,
        }
    }
}

/// prefill 1 回の始まり。追跡を初期化する (短いプロンプトでは追跡しない)。
pub fn begin_prefill(n_tok: usize) {
    if !enabled() {
        return;
    }
    let active = n_tok >= min_tok();
    TRACK.with(|c| {
        let mut t = c.borrow_mut();
        *t = Track::new();
        t.active = active;
        t.keep = keep_tok();
    });
    STATS.with(|s| *s.borrow_mut() = FlushStats::new());
}

/// 層 `layer` の prefill の選択 (`nt * k` の平らな配列) を数える。
pub fn note_selection(layer: usize, nt: usize, k: usize, sels: &[usize]) {
    if !enabled() {
        return;
    }
    TRACK.with(|c| {
        let mut t = c.borrow_mut();
        if t.active {
            t.note(layer, nt, k, sels);
        }
    });
}

/// 層 `layer` の decode の選択 (`k` 本) を「温かい作業集合」へ足す。
pub fn note_decode(layer: usize, sel: &[usize]) {
    if !enabled() {
        return;
    }
    let k = sel.len();
    if k == 0 {
        return;
    }
    DEC.with(|c| {
        let mut d = c.borrow_mut();
        d.k = k;
        if d.rings.len() <= layer {
            d.rings.resize_with(layer + 1, VecDeque::new);
        }
        for &e in sel {
            d.rings[layer].push_back(e as u16);
        }
        let cap = dec_tok().saturating_mul(k).max(1);
        while d.rings[layer].len() > cap {
            d.rings[layer].pop_front();
        }
    });
}

/// 落とす / 戻す対象。
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Plan {
    /// page cache から落とす `(層, expert id 昇順)`。
    pub cold: Vec<(usize, Vec<u16>)>,
    /// `MADV_WILLNEED` で戻す `(層, expert id 昇順)` = decode の温かい集合 ∧ prefill が触った。
    pub warm: Vec<(usize, Vec<u16>)>,
}

impl Plan {
    /// 何もしない計画か。
    pub fn is_empty(&self) -> bool {
        self.cold.is_empty() && self.warm.is_empty()
    }
}

/// decode の温かい集合 (層ごとのビット列)。
fn decode_bits() -> Vec<Vec<u64>> {
    let mut warm_bits: Vec<Vec<u64>> = Vec::new();
    DEC.with(|c| {
        let d = c.borrow();
        warm_bits.resize_with(d.rings.len(), Vec::new);
        for (layer, ring) in d.rings.iter().enumerate() {
            for &e in ring.iter() {
                set_bit(&mut warm_bits[layer], e as usize);
            }
        }
    });
    warm_bits
}

/// 追跡状態と温かい集合から計画を組む (状態は変えない)。
fn build_plan(t: &Track, warm_bits: &[Vec<u64>]) -> Plan {
    let mut plan = Plan::default();
    let n_layer = t.hot.len().max(warm_bits.len());
    for layer in 0..n_layer {
        let head = if layer < t.hot.len() {
            t.head_only(layer)
        } else {
            Vec::new()
        };
        let warm_row: &[u64] = warm_bits.get(layer).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut cold: Vec<u16> = Vec::new();
        for e in head {
            let w = e as usize / 64;
            let hit = warm_row
                .get(w)
                .map(|v| v & (1u64 << (e as usize % 64)) != 0)
                .unwrap_or(false);
            if !hit {
                cold.push(e);
            }
        }
        if !cold.is_empty() {
            plan.cold.push((layer, cold));
        }
        // 戻す候補: 温かい集合 ∧ この prefill が触った。
        if layer < t.hot.len() && !warm_row.is_empty() {
            let mut both: Vec<u64> = Vec::with_capacity(warm_row.len());
            for (w, &b) in warm_row.iter().enumerate() {
                let hot = t.hot[layer].get(w).copied().unwrap_or(0);
                both.push(b & hot);
            }
            let ids = bits_to_ids(&both);
            if !ids.is_empty() {
                plan.warm.push((layer, ids));
            }
        }
    }
    plan
}

/// 落とす / 戻す対象を組み立てる (**追跡は終了する**)。
///
/// `cold` = prefill が選んだ ∧ 末尾ウィンドウに無い ∧ decode の温かい集合に無い。
/// `warm` = decode の温かい集合 ∧ prefill が選んだ (戻す価値がある = ディスクに
/// 落ちている可能性がある)。
pub fn take_plan() -> Plan {
    let active = TRACK.with(|c| c.borrow().active);
    if !active {
        TRACK.with(|c| *c.borrow_mut() = Track::new());
        return Plan::default();
    }
    let warm_bits = decode_bits();
    let plan = TRACK.with(|c| build_plan(&c.borrow(), &warm_bits));
    TRACK.with(|c| *c.borrow_mut() = Track::new());
    plan
}

/// 計画を**消費せずに**覗く (テスト用。本番は [`take_plan`] だけを使う)。
pub fn peek_plan() -> Plan {
    let warm_bits = decode_bits();
    TRACK.with(|c| {
        let t = c.borrow();
        if t.active {
            build_plan(&t, &warm_bits)
        } else {
            Plan::default()
        }
    })
}

/// 1 回の prefill ぶんの落とし / 戻しの結果。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FlushStats {
    /// `madvise` + `fadvise` を投げた範囲の本数。
    pub ranges: u64,
    /// 落としたバイト数。
    pub bytes: u64,
    /// syscall が失敗した範囲の本数。
    pub failed: u64,
    /// 落とし / 戻しの壁時計 (ms)。
    pub ms: f64,
    /// 落とした expert の本数 (範囲ではない)。
    pub experts: u64,
    /// `MADV_WILLNEED` を投げた範囲の本数。
    pub warm_ranges: u64,
    /// `MADV_WILLNEED` を投げたバイト数。
    pub warm_bytes: u64,
}

/// 落とした結果を足す。
pub fn add_drop_stats(outcome: DropOutcome, experts: u64) {
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        match outcome {
            DropOutcome::Dropped(n) => {
                s.ranges += 1;
                s.bytes += n as u64;
            }
            DropOutcome::Failed => s.failed += 1,
            DropOutcome::Skipped | DropOutcome::Warmed(_) => {}
        }
        s.experts += experts;
    });
}

/// 戻した結果を足す。
pub fn add_warm_stats(outcome: DropOutcome) {
    STATS.with(|s| {
        let mut s = s.borrow_mut();
        match outcome {
            DropOutcome::Warmed(n) => {
                s.warm_ranges += 1;
                s.warm_bytes += n as u64;
            }
            DropOutcome::Failed => s.failed += 1,
            _ => {}
        }
    });
}

/// 処理時間を足す。
pub fn add_flush_ms(ms: f64) {
    STATS.with(|s| s.borrow_mut().ms += ms);
}

/// prefill 1 回の終わり。1 行のログを返す (何もしなかったときは `None`)。
pub fn finish_prefill() -> Option<String> {
    let s = STATS.with(|s| std::mem::take(&mut *s.borrow_mut()));
    if s.ranges == 0 && s.failed == 0 && s.warm_ranges == 0 {
        return None;
    }
    Some(format!(
        "[page-hint] dropped={} experts={} ({:.0} MiB) failed={} warm={} ({:.0} MiB) in {:.1}ms",
        s.ranges,
        s.experts,
        s.bytes as f64 / (1024.0 * 1024.0),
        s.failed,
        s.warm_ranges,
        s.warm_bytes as f64 / (1024.0 * 1024.0),
        s.ms
    ))
}

/// 診断用: 追跡中のトークン数 (テスト用)。
pub fn tracked_tokens() -> usize {
    TRACK.with(|c| c.borrow().tok)
}

/// 診断用: decode の温かい集合の本数 (テスト用)。
pub fn dec_hot_len() -> usize {
    DEC.with(|c| c.borrow().rings.iter().map(|r| r.len()).sum())
}

/// テスト用: 末尾ウィンドウのトークン数を上書きする。
pub fn set_keep_for_tests(n: usize) {
    TRACK.with(|c| c.borrow_mut().keep = n);
}

/// テスト用: 追跡と decode 履歴を消す (テストはプロセスで 1 回ずつ動く前提)。
pub fn reset_for_tests() {
    TRACK.with(|c| *c.borrow_mut() = Track::new());
    DEC.with(|c| *c.borrow_mut() = DecHot::new());
    STATS.with(|s| *s.borrow_mut() = FlushStats::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_parsing_is_explicit_only() {
        assert!(!on_from_value(None));
        assert!(!on_from_value(Some("0")));
        assert!(!on_from_value(Some("off")));
        assert!(on_from_value(Some("1")));
        assert!(on_from_value(Some(" YES ")));
        assert!(willneed_from_value(None));
        assert!(!willneed_from_value(Some("0")));
        assert_eq!(keep_tok_from_value(None), DEFAULT_KEEP_TOK);
        assert_eq!(keep_tok_from_value(Some("0")), DEFAULT_KEEP_TOK);
        assert_eq!(keep_tok_from_value(Some("64")), 64);
        assert_eq!(keep_tok_from_value(Some("999999")), TOK_MAX);
        assert_eq!(dec_tok_from_value(None), DEFAULT_DEC_TOK);
        assert_eq!(dec_tok_from_value(Some("32")), 32);
        assert_eq!(warm_mb_from_value(None), DEFAULT_WARM_MB);
        assert_eq!(warm_mb_from_value(Some("0")), 0);
        assert_eq!(warm_mb_from_value(Some("512")), 512);
        assert_eq!(min_tok_from_value(None), DEFAULT_MIN_TOK);
        assert_eq!(min_tok_from_value(Some("0")), 0);
        assert_eq!(min_tok_from_value(Some("2048")), 2048);
    }

    /// 末尾 keep トークンに載っていない選択だけが cold になる。
    #[test]
    fn cold_set_is_head_only() {
        let mut t = Track::new();
        t.active = true;
        t.keep = 2;
        for tok in 0..4usize {
            t.note(0, 1, 1, &[tok]);
        }
        assert_eq!(t.head_only(0), vec![0u16, 1], "末尾 2 トークン (2,3) は残す");
        assert_eq!(t.head_only(0), vec![0u16, 1], "読むだけ (何度でも同じ)");
    }

    /// 層ごとに独立に数える。
    #[test]
    fn layers_are_tracked_independently() {
        let mut t = Track::new();
        t.active = true;
        t.keep = 1;
        t.note(0, 1, 1, &[7]);
        t.note(1, 1, 1, &[8]);
        t.note(0, 1, 1, &[9]);
        t.note(1, 1, 1, &[10]);
        assert_eq!(t.head_only(0), vec![7u16]);
        assert_eq!(t.head_only(1), vec![8u16]);
    }

    /// 連続した id が昇順で出る (範囲の併合が効く形)。
    #[test]
    fn head_only_is_sorted_and_contiguous() {
        let mut t = Track::new();
        t.active = true;
        t.keep = 1;
        t.note(0, 1, 4, &[5, 6, 7, 9]);
        t.note(0, 1, 4, &[100, 101, 102, 103]);
        assert_eq!(t.head_only(0), vec![5u16, 6, 7, 9]);
    }

    /// ゲート off では数えない。
    #[test]
    fn disabled_tracks_nothing() {
        set_enabled_for_tests(Some(false));
        begin_prefill(4096);
        note_selection(0, 1, 1, &[1]);
        note_decode(0, &[2]);
        TRACK.with(|c| c.borrow_mut().active = true);
        assert_eq!(tracked_tokens(), 0);
        assert_eq!(dec_hot_len(), 0);
        assert!(take_plan().is_empty());
        set_enabled_for_tests(None);
    }

    /// 短いプロンプトでは追跡しない (`min_tok`)。
    #[test]
    fn short_prefill_tracks_nothing() {
        set_enabled_for_tests(Some(true));
        reset_for_tests();
        begin_prefill(min_tok().saturating_sub(1));
        note_selection(0, 1, 1, &[1]);
        assert_eq!(tracked_tokens(), 0);
        assert!(take_plan().is_empty());
        reset_for_tests();
        set_enabled_for_tests(None);
    }

    /// decode の温かい集合は cold から外れ、warm に入る。
    #[test]
    fn decode_hot_set_is_kept_and_refilled() {
        set_enabled_for_tests(Some(true));
        reset_for_tests();
        begin_prefill(4096);
        // 末尾ウィンドウを 1 トークンにする (既定は 256 なので、ここだけ縮める)。
        TRACK.with(|c| c.borrow_mut().keep = 1);
        // 層 0: トークン 0..3 で 1,2,3 を選び、末尾 1 トークン (keep=1) で 3 を選ぶ。
        note_selection(0, 1, 1, &[1]);
        note_selection(0, 1, 1, &[2]);
        note_selection(0, 1, 1, &[3]);
        // decode が 2 を使っていた (温かい作業集合)。
        note_decode(0, &[2]);
        let plan = take_plan();
        assert_eq!(plan.cold, vec![(0, vec![1u16])], "1 は cold、2 は温かい、3 は末尾");
        assert_eq!(plan.warm, vec![(0, vec![2u16])]);
        reset_for_tests();
        set_enabled_for_tests(None);
    }

    /// 実ファイルで `drop_range` が page cache を落とすことを確かめる。
    ///
    /// このホストの `madvise(MADV_DONTNEED)` 単独では落ちない (page cache は残る)
    /// ので、`posix_fadvise` まで投げていることがこのテストの主張。
    #[test]
    fn drop_range_evicts_page_cache() {
        use std::io::{Read, Seek, SeekFrom, Write};
        let dir = std::env::temp_dir().join(format!("denno-page-hint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("blob.bin");
        let total = 8usize << 20;
        {
            let mut f = std::fs::File::create(&path).unwrap();
            let block = vec![7u8; 1 << 20];
            for _ in 0..total / block.len() {
                f.write_all(&block).unwrap();
            }
            f.sync_all().unwrap();
        }
        let file = std::fs::File::open(&path).unwrap();
        let map = std::sync::Arc::new(unsafe { Mmap::map(&file).unwrap() });
        register_map(&map, &path);
        // 全ページを読んで page cache に載せる。
        let mut f = std::fs::File::open(&path).unwrap();
        let mut buf = vec![0u8; total];
        f.seek(SeekFrom::Start(0)).unwrap();
        f.read_exact(&mut buf).unwrap();
        let mid = 2usize << 20;
        let ptr = map.as_ptr() as usize + mid;
        assert_eq!(
            unsafe { crate::model::exp_pf::range_resident(ptr, PAGE) },
            Some(true),
            "触った直後は page cache に載っている"
        );
        // [2 MiB, 6 MiB) を落とす (内側に丸まる)。
        let out = drop_range(&map, 2 << 20, 4 << 20);
        assert!(matches!(out, DropOutcome::Dropped(_)), "{out:?}");
        assert_eq!(
            unsafe { crate::model::exp_pf::range_resident(ptr, PAGE) },
            Some(false),
            "madvise + posix_fadvise で page cache から落ちる"
        );
        drop(map);
        drop(file);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }

    /// ページ未満・範囲外は何もしない。
    #[test]
    fn drop_range_skips_degenerate_ranges() {
        let dir = std::env::temp_dir().join(format!("denno-page-hint2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("small.bin");
        std::fs::write(&path, vec![0u8; 8192]).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let map = std::sync::Arc::new(unsafe { Mmap::map(&file).unwrap() });
        assert_eq!(drop_range(&map, 0, 0), DropOutcome::Skipped);
        assert_eq!(drop_range(&map, 1, 4000), DropOutcome::Skipped);
        assert_eq!(drop_range(&map, 1 << 20, 4096), DropOutcome::Skipped);
        // WILLNEED も範囲外なら 0。
        assert_eq!(willneed_range(&map, 1 << 20, 4096), 0);
        drop(map);
        drop(file);
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);
    }
}
