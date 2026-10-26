//! P-MOE-B2: expert 重みの SSD pread とスロットキャッシュ。
//!
//! Flash-Next の expert 重みは **77.02 GB** (Q4_K 471.9 MB/層 + Q5_1 629.1 MB/層 ×
//! 48 層) で VRAM にも RAM にも載らない。1 トークンが使うのは top-10 expert ×
//! 3 行列だけで **約 30.7 MB** (1 expert あたり gate 921.7 KB + up 921.7 KB +
//! down 1.23 MB) なので、GGUF を mmap したまま必要行だけ読むのではなく、
//! ファイルの絶対オフセットから `pread` して直近の expert をスロットに残す。
//!
//! # 読み出し方式: `pread` (`FileExt::read_at`)
//!
//! mmap を選ばない理由は [`crate::ple`] と同じ (実測はそちらの doc を参照)。
//! mmap のランダム行読みは fault-around で隣接ページまで写像し、RSS が要求
//! バイト数の数十倍に膨らむ。77 GB のテンソルでそれが起きると RAM が埋まる。
//! `pread` の読み先は自前のスロットだけなので、常駐量は
//! `slots * slot_bytes` で一定。
//!
//! # オフセットの求め方 (既存 API との接続)
//!
//! [`crate::gguf::TensorRef::offset`] は **データ部先頭からの相対オフセット**なので、
//! ファイルの絶対オフセットは [`crate::gguf::GgufFile::tensor_mapping_parts`] で引く:
//!
//! ```ignore
//! let (file, tensor_abs, _len) = gguf.tensor_mapping_parts(name).unwrap();
//! let sl = ExpertSlice::new(ty, n_in, n_out, n_expert, e).unwrap();
//! let abs = tensor_abs as u64 + sl.off as u64;   // ← これを get に渡す
//! let mut src = FileReader::open(gguf.path_at(file).unwrap())?;
//! let blocks = cache.get(ExpertKey { layer, expert: e as u16 }, abs, sl.len, &mut src)?;
//! expert_gemv(ty, blocks, n_in, n_out, x, out)?;
//! ```
//!
//! # 契約
//!
//! - 1 スロット = expert 1 個ぶんの生バイト。`len > slot_bytes` は **エラー**
//!   (静かに切り詰めない)。
//! - 未命中で全スロットが埋まっていれば、**最終使用が最も古いスロット**を追い出す。
//!   同じ鍵の再要求はヒット (バイトを読み直さない)。同じ鍵でも `len` が違えば
//!   別内容とみなしてミス扱いにする。
//! - `slots == 0` はエラーにせず「毎回 pread してキャッシュしない」動作にする
//!   (呼び出し側の分岐を減らすため。統計は misses だけ増える)。
//! - 読みが失敗したスロットは **有効化しない** (半端なバイトをヒットで返さない)。
//! - **単一スレッド前提**。`get` は `&mut self` を要求するので、同時に 2 本の
//!   `&mut` は存在し得ない。`ExpertCache` は `Send` (スレッドに移せる) だが、
//!   複数スレッドから 1 個を共有する使い方は想定しない (スロットの使い回しと
//!   LRU の通し番号が競合する)。スレッドごとに 1 個持つこと。

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// 既定のスロット数 (top-k 10 × 層方向の先読みぶん)。
pub const DEFAULT_SLOTS: usize = 64;

/// 既定の 1 スロットのバイト数 (1.25 MiB)。
///
/// 最大の expert は Q5_1 の `ffn_down_exps` で 1,228,800 B (2560 行 × 480 B)。
/// 64 スロットで 83.9 MB の常駐になる。
pub const DEFAULT_SLOT_BYTES: usize = 1_310_720;

/// expert 1 個を指す鍵。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

/// キャッシュの統計。
///
/// `bytes_read` は `pread_at` が返したバイト数の累計で、**失敗した読みは
/// 計上しない** (読めた量を返さないため)。`evictions` は追い出しの回数で、
/// 追い出したあとの読みが失敗しても計上する (スロットは実際に失われている)。
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub evictions: u64,
}

/// ファイルの絶対オフセットから読む口 (テストで差し替えられるように trait にする)。
///
/// 契約: 成功時は `buf.len()` バイトを読み切って `Ok(buf.len())` を返すこと。
/// 読み切れない場合 (`Ok(0)` = EOF、または短読み) は `Err` を返す。
/// [`ExpertCache::get`] は戻り値が `buf.len()` と違えばエラーにするので、
/// 短読みを `Ok` で返す実装でも壊れない (二重の門番)。
pub trait PreadAt {
    fn pread_at(&mut self, off: u64, buf: &mut [u8]) -> Result<usize>;
}

/// 実ファイル版。unix は `FileExt::read_at`、それ以外は `seek` + `read_exact`。
pub struct FileReader {
    file: File,
}

impl FileReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| anyhow::anyhow!("{} を開けない: {}", path.display(), e))?;
        Ok(FileReader { file })
    }
}

#[cfg(unix)]
impl PreadAt for FileReader {
    fn pread_at(&mut self, off: u64, buf: &mut [u8]) -> Result<usize> {
        use std::io::ErrorKind;
        use std::os::unix::fs::FileExt;

        // `read_at` は短読みを返しうる (シグナル / EOF) ので、読み切るまで繰り返す。
        let want = buf.len();
        let mut done = 0usize;
        while done < want {
            let at = off
                .checked_add(done as u64)
                .ok_or_else(|| anyhow::anyhow!("pread のオフセットが u64 を溢れる (off={off})"))?;
            match self.file.read_at(&mut buf[done..], at) {
                Ok(0) => bail!(
                    "pread が EOF に当たった (off={at}, 要求 {want} B, 読めた {done} B)"
                ),
                Ok(n) => done += n,
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => {
                    return Err(anyhow::anyhow!("pread に失敗 (off={at}): {}", e));
                }
            }
        }
        Ok(done)
    }
}

#[cfg(not(unix))]
impl PreadAt for FileReader {
    fn pread_at(&mut self, off: u64, buf: &mut [u8]) -> Result<usize> {
        use std::io::{Read, Seek, SeekFrom};
        self.file
            .seek(SeekFrom::Start(off))
            .map_err(|e| anyhow::anyhow!("seek に失敗 (off={off}): {}", e))?;
        self.file
            .read_exact(buf)
            .map_err(|e| anyhow::anyhow!("read_exact に失敗 (off={off}): {}", e))?;
        Ok(buf.len())
    }
}

/// expert 生バイトの LRU スロットキャッシュ。
///
/// 中身は `slots` 本の `Vec<u8>` (各 `slot_bytes`)。LRU の順序は
/// 「最終使用の通し番号」(`last_use`) で持つ。通し番号は `get` のたびに
/// 単調増加するので、追い出し対象 (最小値) は常に一意に決まる。
pub struct ExpertCache {
    slots: usize,
    slot_bytes: usize,
    /// 各スロットの生バイト (長さ slot_bytes 固定)
    bufs: Vec<Vec<u8>>,
    /// 各スロットの有効バイト数 (`keys[i] == None` のとき未使用)
    lens: Vec<usize>,
    keys: Vec<Option<ExpertKey>>,
    last_use: Vec<u64>,
    clock: u64,
    /// `slots == 0` のときだけ使う読み先 (キャッシュしない経路)
    scratch: Vec<u8>,
    stats: CacheStats,
}

impl std::fmt::Debug for ExpertCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpertCache")
            .field("slots", &self.slots)
            .field("slot_bytes", &self.slot_bytes)
            .field("used", &self.keys.iter().filter(|k| k.is_some()).count())
            .field("stats", &self.stats)
            .finish()
    }
}

impl ExpertCache {
    /// `slots` 本のスロットを確保する (`slot_bytes` は最大の expert バイト数)。
    ///
    /// 確保はここで 1 回だけ。`get` はバイト列を再確保しない
    /// (スロットの使い回しで済ませる)。`slots == 0` なら 1 バイトも確保しない。
    pub fn new(slots: usize, slot_bytes: usize) -> Self {
        ExpertCache {
            slots,
            slot_bytes,
            bufs: (0..slots).map(|_| vec![0u8; slot_bytes]).collect(),
            lens: vec![0usize; slots],
            keys: vec![None; slots],
            last_use: vec![0u64; slots],
            clock: 0,
            scratch: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    /// スロット数。
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// 1 スロットのバイト数。
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// 鍵に対応する生バイトを返す。未命中なら `src` (pread 元) から読む。
    ///
    /// `off` は **ファイル内の絶対オフセット** ([`crate::gguf::GgufFile::tensor_mapping_parts`]
    /// の戻り値 + [`crate::model::moe_expert::ExpertSlice::off`])。
    /// 戻り値の借用は `&mut self` の間だけ有効 (スロットの使い回しを許すため)。
    pub fn get<'a>(
        &'a mut self,
        key: ExpertKey,
        off: u64,
        len: usize,
        src: &mut dyn PreadAt,
    ) -> Result<&'a [u8]> {
        if len == 0 {
            bail!("{:?}: expert のバイト数が 0 (読む意味が無い)", key);
        }
        if len > self.slot_bytes {
            bail!(
                "{:?}: expert のバイト数 {} が 1 スロット {} を超える (切り詰めない)",
                key,
                len,
                self.slot_bytes
            );
        }
        if off.checked_add(len as u64).is_none() {
            bail!("{:?}: off={} + len={} が u64 を溢れる", key, off, len);
        }

        // slots == 0: キャッシュしない。毎回 pread して scratch を返す。
        if self.slots == 0 {
            self.stats.misses += 1;
            if self.scratch.len() < len {
                self.scratch.resize(len, 0);
            }
            let n = read_into(src, off, &mut self.scratch[..len])?;
            self.stats.bytes_read += n as u64;
            return Ok(&self.scratch[..len]);
        }

        // ヒット (同じ鍵かつ同じ長さのときだけ。長さが違えば別内容)
        let hit = (0..self.slots).find(|&i| self.keys[i] == Some(key) && self.lens[i] == len);
        if let Some(i) = hit {
            self.clock = self.clock.saturating_add(1);
            self.last_use[i] = self.clock;
            self.stats.hits += 1;
            return Ok(&self.bufs[i][..len]);
        }

        // ミス: 空きスロットを優先し、無ければ最終使用が最も古いスロットを追い出す。
        self.stats.misses += 1;
        let slot = match (0..self.slots).find(|&i| self.keys[i].is_none()) {
            Some(i) => i,
            None => {
                let mut victim = 0usize;
                for i in 1..self.slots {
                    if self.last_use[i] < self.last_use[victim] {
                        victim = i;
                    }
                }
                self.stats.evictions += 1;
                victim
            }
        };

        // 読む前に無効化しておく (読みが失敗したら半端なスロットを有効にしない)。
        self.keys[slot] = None;
        self.lens[slot] = 0;
        let n = read_into(src, off, &mut self.bufs[slot][..len])?;

        self.stats.bytes_read += n as u64;
        self.clock = self.clock.saturating_add(1);
        self.keys[slot] = Some(key);
        self.lens[slot] = len;
        self.last_use[slot] = self.clock;
        Ok(&self.bufs[slot][..len])
    }

    pub fn stats(&self) -> CacheStats {
        self.stats
    }

    /// 統計だけを 0 に戻す (スロットの中身と LRU の順序はそのまま)。
    pub fn reset_stats(&mut self) {
        self.stats = CacheStats::default();
    }
}

/// `dst` を読み切る。戻り値が要求長と違えばエラー (短読みを成功にしない)。
fn read_into(src: &mut dyn PreadAt, off: u64, dst: &mut [u8]) -> Result<usize> {
    let want = dst.len();
    let n = src.pread_at(off, dst)?;
    if n != want {
        bail!(
            "pread が {want} B 要求に対して {n} B しか返さない (off={off}, EOF か短読み)"
        );
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// P-MOE-G1: 3 行列ぶんの束 (forward 経路への配線)
// ---------------------------------------------------------------------------

/// `DENNO_EXPERT_CACHE_MB`: expert キャッシュに使う MB。**既定 0 = 無効**
/// (このとき forward は今までどおり mmap + `prefetch()` の経路を通る)。
pub const EXPERT_CACHE_MB_ENV: &str = "DENNO_EXPERT_CACHE_MB";

/// `DENNO_EXPERT_CACHE_STATS=1`: デコーダを落とすときに行列ごとの統計を
/// stderr へ 1 行ずつ出す。未設定・"0"・その他の値では何も出さない。
pub const EXPERT_CACHE_STATS_ENV: &str = "DENNO_EXPERT_CACHE_STATS";

/// `DENNO_EXPERT_CACHE_MB` の解釈。未設定・空・"0"・数値でない値は 0 (無効)。
pub fn mb_from_env() -> usize {
    parse_mb(std::env::var(EXPERT_CACHE_MB_ENV).ok().as_deref())
}

/// [`mb_from_env`] の純粋部分 (環境変数を触らないのでテストできる)。
fn parse_mb(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() && s != "0" => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// MB からスロット数を出す (`MB * 1024 * 1024 / slot_bytes` の切り捨て、最低 1)。
///
/// `mb == 0` は「無効」なので呼ばない ([`ExpertCacheSet::from_mb`] の前提)。
/// `slot_bytes == 0` はスロットとして意味を持たないので 1 に寄せる。
pub fn slots_for_mb(mb: usize, slot_bytes: usize) -> usize {
    if slot_bytes == 0 {
        return 1;
    }
    (mb.saturating_mul(1024 * 1024) / slot_bytes).max(1)
}

/// expert 行列の種別 (統計を分けて出す単位)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertMatrix {
    Gate,
    Up,
    Down,
}

impl ExpertMatrix {
    pub const ALL: [ExpertMatrix; 3] = [ExpertMatrix::Gate, ExpertMatrix::Up, ExpertMatrix::Down];

    /// 統計行に出す名前 (`ffn_gate_exps` の短縮)。
    pub fn name(self) -> &'static str {
        match self {
            ExpertMatrix::Gate => "gate",
            ExpertMatrix::Up => "up",
            ExpertMatrix::Down => "down",
        }
    }

    /// 配列の添字。
    pub fn index(self) -> usize {
        match self {
            ExpertMatrix::Gate => 0,
            ExpertMatrix::Up => 1,
            ExpertMatrix::Down => 2,
        }
    }
}

/// 1 本ぶんのキャッシュ指定。`slot_bytes` は**その行列の最大 expert** の
/// バイト数 ([`crate::model::moe_expert::ExpertSlice::new`] の `len`)。
#[derive(Clone, Copy, Debug)]
pub struct CacheSpec {
    pub slot_bytes: usize,
}

/// gate / up / down の 3 本をまとめたキャッシュ (forward 経路から使う口)。
///
/// 3 本を別々に持つのは、行列ごとに expert のバイト数が違う (Q4_K の gate/up
/// 921,600 B、Q5_1 の down 1,228,800 B) ため。`slot_bytes` を共有すると
/// 小さい行列で無駄が出る。
///
/// 鍵は [`ExpertKey`] の `(layer, expert)` なので、層をまたいで 1 本の LRU を
/// 共有する (同じ expert が同じ層で再要求されたときだけヒットする)。
///
/// 読み口は **GGUF のパートごと**に 1 本持つ。split GGUF では層によって
/// expert テンソルの属するファイルが変わるので、1 本では足りない。
///
/// **常駐量は 3 倍**: `DENNO_EXPERT_CACHE_MB` は行列ごとの上限なので、
/// 3 本あわせた常駐は約 `3 * MB` MiB になる (MB = 1024 なら約 3 GiB)。
///
/// [`Drop`] は `DENNO_EXPERT_CACHE_STATS=1` のときだけ 3 行を stderr に出す。
pub struct ExpertCacheSet {
    mb: usize,
    caches: [ExpertCache; 3],
    readers: Vec<FileReader>,
}

impl ExpertCacheSet {
    /// 3 本ぶんのキャッシュを作る。`specs` の順は [`ExpertMatrix::ALL`] と同じ。
    ///
    /// スロット数は行列ごとに `MB * 1024 * 1024 / slot_bytes` (切り捨て・最低 1)。
    /// `slot_bytes` は**その行列の最大 expert** でなければならない
    /// (小さいと [`ExpertCache::get`] が `len > slot_bytes` で落ちる = 静かに
    /// 切り詰めない)。`paths` は GGUF のパート 0..n の順。
    pub fn from_mb(mb: usize, specs: [CacheSpec; 3], paths: &[PathBuf]) -> Result<Self> {
        let slots = [
            slots_for_mb(mb, specs[0].slot_bytes),
            slots_for_mb(mb, specs[1].slot_bytes),
            slots_for_mb(mb, specs[2].slot_bytes),
        ];
        let bytes = [
            specs[0].slot_bytes,
            specs[1].slot_bytes,
            specs[2].slot_bytes,
        ];
        Self::new_with_slots(slots, bytes, paths, mb)
    }

    /// スロット数を直接指定して作る (`slots == 0` も許す = 毎回 pread して
    /// キャッシュしない動作)。[`ExpertCacheSet::from_mb`] の下請け。
    pub fn new_with_slots(
        slots: [usize; 3],
        slot_bytes: [usize; 3],
        paths: &[PathBuf],
        mb: usize,
    ) -> Result<Self> {
        if paths.is_empty() {
            bail!("expert キャッシュ: GGUF のパートが 0 本 (読み口を作れない)");
        }
        let mut readers = Vec::with_capacity(paths.len());
        for p in paths {
            readers.push(FileReader::open(p)?);
        }
        Ok(ExpertCacheSet {
            mb,
            caches: [
                ExpertCache::new(slots[0], slot_bytes[0]),
                ExpertCache::new(slots[1], slot_bytes[1]),
                ExpertCache::new(slots[2], slot_bytes[2]),
            ],
            readers,
        })
    }

    /// 設定された MB (0 は作れない = `from_mb` を呼ばない前提)。
    pub fn mb(&self) -> usize {
        self.mb
    }

    /// 行列ごとのスロット数。
    pub fn slots(&self, m: ExpertMatrix) -> usize {
        self.caches[m.index()].slots()
    }

    /// 行列ごとの 1 スロットのバイト数。
    pub fn slot_bytes(&self, m: ExpertMatrix) -> usize {
        self.caches[m.index()].slot_bytes()
    }

    /// 行列 `m` の expert 生バイトを返す。
    ///
    /// `part` は GGUF のパート番号 ([`crate::gguf::GgufFile::tensor_mapping_parts`]
    /// の 1 番目の戻り値)、`off` は**そのファイル内の絶対オフセット**。
    pub fn get(
        &mut self,
        m: ExpertMatrix,
        part: usize,
        key: ExpertKey,
        off: u64,
        len: usize,
    ) -> Result<&[u8]> {
        let n = self.readers.len();
        if part >= n {
            bail!("expert キャッシュ: パート {part} が範囲外 ({n} 本)");
        }
        let ExpertCacheSet { caches, readers, .. } = self;
        caches[m.index()].get(key, off, len, &mut readers[part])
    }

    /// 行列ごとの統計。
    pub fn stats(&self, m: ExpertMatrix) -> CacheStats {
        self.caches[m.index()].stats()
    }

    /// 行列ごとのヒット率 (要求が 0 件なら 0.0)。
    pub fn hit_rate(&self, m: ExpertMatrix) -> f64 {
        let s = self.stats(m);
        let n = s.hits + s.misses;
        if n == 0 {
            0.0
        } else {
            s.hits as f64 / n as f64
        }
    }

    /// 1 行列ぶんの統計行 (`DENNO_EXPERT_CACHE_STATS=1` の出力そのもの)。
    pub fn report(&self, m: ExpertMatrix) -> String {
        let s = self.stats(m);
        format!(
            "expert_cache {}: slots={} slot_bytes={} hits={} misses={} hit_rate={:.3} \
             bytes_read={} evictions={}",
            m.name(),
            self.slots(m),
            self.slot_bytes(m),
            s.hits,
            s.misses,
            self.hit_rate(m),
            s.bytes_read,
            s.evictions
        )
    }
}

impl Drop for ExpertCacheSet {
    fn drop(&mut self) {
        let on = std::env::var(EXPERT_CACHE_STATS_ENV)
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        if !on {
            return;
        }
        for m in ExpertMatrix::ALL {
            eprintln!("{}", self.report(m));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_mb_accepts_only_positive_integers() {
        // 未設定・空・"0"・数値でない値は 0 (無効)
        assert_eq!(parse_mb(None), 0);
        assert_eq!(parse_mb(Some("")), 0);
        assert_eq!(parse_mb(Some("  ")), 0);
        assert_eq!(parse_mb(Some("0")), 0);
        assert_eq!(parse_mb(Some("off")), 0);
        assert_eq!(parse_mb(Some("-1")), 0);
        // 有効値
        assert_eq!(parse_mb(Some("1")), 1);
        assert_eq!(parse_mb(Some("4096")), 4096);
        assert_eq!(parse_mb(Some(" 512 ")), 512);
    }

    #[test]
    fn slots_for_mb_floors_and_keeps_at_least_one() {
        let mb = 1024 * 1024; // 1 MiB
        assert_eq!(slots_for_mb(1, mb), 1);
        assert_eq!(slots_for_mb(4, mb), 4);
        // 端数は切り捨て
        assert_eq!(slots_for_mb(1, mb / 3), 3);
        assert_eq!(slots_for_mb(1, mb * 4), 1, "1 スロット未満でも 1 に寄せる");
        assert_eq!(slots_for_mb(1, 0), 1, "slot_bytes 0 でも 1 に寄せる");
        // 実モデルの最大 expert (Q5_1 down = 1,228,800 B) で溢れない
        let down = 1_228_800usize;
        assert_eq!(slots_for_mb(4096, down), 4096 * 1024 * 1024 / down);
        // usize を溢れる MB でも panic しない (飽和)
        assert!(slots_for_mb(usize::MAX, 1) > 0);
    }

    #[test]
    fn matrix_names_and_indexes_are_distinct() {
        assert_eq!(ExpertMatrix::Gate.name(), "gate");
        assert_eq!(ExpertMatrix::Up.name(), "up");
        assert_eq!(ExpertMatrix::Down.name(), "down");
        let mut idx: Vec<usize> = ExpertMatrix::ALL.iter().map(|m| m.index()).collect();
        idx.sort_unstable();
        assert_eq!(idx, vec![0, 1, 2]);
    }

    #[test]
    fn hit_rate_of_empty_cache_is_zero() {
        // 要求が 0 件でも 0 除算しない
        let dir = std::env::temp_dir().join("denno_expert_cache_set_empty");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("dummy.bin");
        std::fs::write(&p, [0u8; 32]).expect("temp file");
        let set = ExpertCacheSet::new_with_slots([1, 1, 1], [16, 16, 16], &[p], 1).expect("set");
        assert_eq!(set.hit_rate(ExpertMatrix::Gate), 0.0);
        assert!(set.report(ExpertMatrix::Gate).contains("hit_rate=0.000"));
        assert_eq!(set.slots(ExpertMatrix::Down), 1);
        assert_eq!(set.mb(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
