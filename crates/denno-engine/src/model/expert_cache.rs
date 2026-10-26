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
use std::path::Path;

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
