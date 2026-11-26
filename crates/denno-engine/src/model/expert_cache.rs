//! P-MOE-B2 / P-MOE-G1 / P38: expert 重みの SSD pread と常駐キャッシュ。
//!
//! Flash-Next の expert 重みは **77.02 GB** (Q4_K 471.9 MB/層 + Q5_1 629.1 MB/層 ×
//! 48 層) で VRAM にも RAM にも全部は載らない。1 トークンが使うのは top-10 expert ×
//! 3 行列だけで **約 30.7 MB** (1 expert あたり gate 921.7 KB + up 921.7 KB +
//! down 1.23 MB) なので、GGUF を mmap したまま必要行だけ読むのではなく、
//! ファイルの絶対オフセットから `pread` して直近の expert を常駐させる。
//!
//! # 置き場所: ホスト RAM (P38)
//!
//! 容量 `DENNO_EXPERT_CACHE_MB` の常駐先は **ホスト RAM** にする。VRAM を選ばない
//! 理由は 3 つ:
//!
//! 1. expert の計算は `moe_expert::expert_gemv_any` (CPU) だけで、GPU カーネルは
//!    存在しない。VRAM に置いた瞬間に計算の入口が変わる = **数値経路の変更**になる。
//!    タスク 38 の制約は「expert のデコードは触らない。キャッシュは置き場所の話だけ」。
//! 2. dense 10.8 GB を載せた残り 21 GB では 6.8 層ぶんしか入らない。RAM 60 GB なら
//!    20 GB で 12.7 層ぶん (§REPORT_P38.md)。
//! 3. H2D を省くだけの設計は、expert が既に GPU で計算されていることが前提。
//!    HEAD では expert は CPU で、`moe_gemv_dev` に渡すと **ヒット時もミス時も**
//!    CPU と GPU の丸めが違う可能性を排除できない (アンカー未取得)。
//!
//! VRAM 常駐に移すときに要るのは「ミスで upload した DevBuffer を (layer, expert)
//! で保持し、ヒットで `moe_gemv_dev` に渡す」だけで、この API の `get` が返す
//! バイト列の出所を差し替えれば足りる。ただし **CPU 経路とのビット一致アンカーが
//! 先** (人間判断)。
//!
//! # 並列 (P38)
//!
//! decode は選択 expert 10 体を rayon で並列に計算する。旧実装は `&mut self` の
//! スロットを共有できず、キャッシュを有効にすると **並列化を切っていた**
//! (`qwen4exp.rs` の `parallel = par && cache.is_none()`) = 1 桁遅い。
//! ここでは
//!
//! - 状態 (map / LRU / 統計 / バッファ束) を `Mutex<Inner>` に入れ、
//! - ヒットは `Arc<Vec<u8>>` を clone して即座にロックを離す (O(1))、
//! - ミスの pread は **ロックの外** で行う (I/O を直列化しない)、
//! - `pread` は `read_at(&File)` なので読み口はスレッド間で共有できる
//!
//! とした。同じ鍵を 2 スレッドが同時にミスした場合だけ二重読みになるが、
//! 後から入れた側が勝ち、先に入れた側の `Arc` は読者の参照が切れてから落ちる。
//!
//! # 数値不変
//!
//! キャッシュが返すのは「`ExpertW::slice` と同じ範囲の生バイト」だけで、
//! GEMV の入口 (`expert_gemv_any`) も引数も変えない。ヒット/ミスの違いは
//! バイト列の出所だけ。`ExpertW::slice_abs` と `ExpertSlice` を共有しているので、
//! 範囲がずれる余地は無い。
//!
//! # 契約
//!
//! - 1 スロット = expert 1 個ぶんの生バイト。`len > slot_bytes` は **エラー**
//!   (静かに切り詰めない)。
//! - 未命中で全スロットが埋まっていれば、**最終使用が最も古いスロット**を追い出す。
//!   同じ鍵の再要求はヒット (バイトを読み直さない)。同じ鍵でも `len` が違えば
//!   別内容とみなしてミス扱いにする。
//! - `slots == 0` はエラーにせず「毎回 pread してキャッシュしない」動作にする。
//! - 読みが失敗したスロットは **有効化しない** (半端なバイトをヒットで返さない)。

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};

/// 既定のスロット数 (top-k 10 × 層方向の先読みぶん)。
pub const DEFAULT_SLOTS: usize = 64;

/// 既定の 1 スロットのバイト数 (1.25 MiB)。
///
/// 最大の expert は Q5_1 の `ffn_down_exps` で 1,228,800 B (2560 行 × 480 B)。
/// 64 スロットで 83.9 MB の常駐になる。
pub const DEFAULT_SLOT_BYTES: usize = 1_310_720;

/// LRU リストの「無し」。
const NIL: u32 = u32::MAX;

/// expert 1 個を指す鍵。
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ExpertKey {
    pub layer: u16,
    pub expert: u16,
}

/// キャッシュの統計。
///
/// `bytes_read` は `pread_at` が返したバイト数の累計で、**失敗した読みは
/// 計上しない**。`evictions` は追い出しの回数で、追い出したあとの読みが
/// 失敗しても計上する (スロットは実際に失われている)。
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
    pub evictions: u64,
}

/// ファイルの絶対オフセットから読む口 (テストで差し替えられるように trait にする)。
///
/// `&self` なので **複数スレッドから同時に呼べる** (pread は位置を引数で持つ)。
/// 実装は `Send + Sync` であること。
///
/// 契約: 成功時は `buf.len()` バイトを読み切って `Ok(buf.len())` を返すこと。
/// 読み切れない場合 (`Ok(0)` = EOF、または短読み) は `Err` を返す。
/// [`ExpertCache::get`] は戻り値が `buf.len()` と違えばエラーにするので、短読みを
/// `Ok` で返す実装でも壊れない (二重の門番)。
pub trait PreadAt: Send + Sync {
    fn pread_at(&self, off: u64, buf: &mut [u8]) -> Result<usize>;
}

/// 実ファイル版。unix は `FileExt::read_at`、それ以外は `seek` + `read_exact`。
pub struct FileReader {
    path: PathBuf,
    file: File,
}

impl FileReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| anyhow::anyhow!("{} を開けない: {}", path.display(), e))?;
        Ok(FileReader {
            path: path.to_path_buf(),
            file,
        })
    }
}

#[cfg(unix)]
impl PreadAt for FileReader {
    fn pread_at(&self, off: u64, buf: &mut [u8]) -> Result<usize> {
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
    fn pread_at(&self, off: u64, buf: &mut [u8]) -> Result<usize> {
        use std::io::{Read, Seek, SeekFrom};
        // 非 unix には位置付き読みが無いので 1 回ごとに開き直す (製品対象は Linux)。
        let mut f = File::open(&self.path)
            .map_err(|e| anyhow::anyhow!("{} を開けない: {}", self.path.display(), e))?;
        f.seek(SeekFrom::Start(off))
            .map_err(|e| anyhow::anyhow!("seek に失敗 (off={off}): {e}"))?;
        f.read_exact(buf)
            .map_err(|e| anyhow::anyhow!("read_exact に失敗 (off={off}): {e}"))?;
        Ok(buf.len())
    }
}

/// 1 スロットぶんの常駐。`len` は実際の expert のバイト数で、`buf` の容量は
/// それ以上 (固定長の塊から切り出したときは `slot_bytes` ぶん)。
struct Slot {
    key: ExpertKey,
    len: usize,
    buf: Arc<Region>,
}

// ---------------------------------------------------------------------------
// task 260: 常駐先の塊 (pinned / heap)
// ---------------------------------------------------------------------------

/// `cuda` feature の有無をここで吸収する (GPU 無しのビルドでは常に `None`)。
#[cfg(feature = "cuda")]
type PinnedHost = crate::cuda::PinnedHost;

/// GPU 無しのビルド用の代役 (構築されない)。
#[cfg(not(feature = "cuda"))]
struct PinnedHost;

#[cfg(not(feature = "cuda"))]
impl PinnedHost {
    fn alloc(_bytes: usize) -> Option<Self> {
        None
    }
    fn ptr(&self) -> *mut u8 {
        std::ptr::null_mut()
    }
    fn bytes(&self) -> usize {
        0
    }
}

/// 塊の中身。`Pinned` は CUDA の `cuMemHostAlloc` で確保したページロック済みの
/// ホストメモリ (大きな読みに追い出されない)、`Heap` は従来の `Vec<u8>`。
enum ChunkMem {
    Heap(Vec<u8>),
    Pinned(PinnedHost),
}

/// 同じ塊を指す `Region` が共有する (ページロックは解放まで解かない)。
pub struct Chunk {
    mem: ChunkMem,
}

// 生ポインタ (pinned) と `Vec` を持つだけ。中身の読み書きの排他は
// LRU (ミューテックス) と `Region` の参照カウントが持つ。
unsafe impl Send for Chunk {}
unsafe impl Sync for Chunk {}

impl Chunk {
    fn heap(bytes: usize) -> Arc<Chunk> {
        Arc::new(Chunk {
            mem: ChunkMem::Heap(vec![0u8; bytes.max(1)]),
        })
    }

    /// 読み込み済みの `Vec` をそのまま塊にする (ヒープ経路)。
    fn from_vec(v: Vec<u8>) -> Arc<Chunk> {
        Arc::new(Chunk {
            mem: ChunkMem::Heap(v),
        })
    }

    /// pinned が取れたら pinned、取れなければヒープ (失敗でリクエストを落とさない)。
    fn new(bytes: usize, want_pinned: bool) -> (Arc<Chunk>, bool) {
        if want_pinned {
            if let Some(p) = PinnedHost::alloc(bytes) {
                return (
                    Arc::new(Chunk {
                        mem: ChunkMem::Pinned(p),
                    }),
                    true,
                );
            }
        }
        (Chunk::heap(bytes), false)
    }

    fn as_ptr(&self) -> *mut u8 {
        match &self.mem {
            ChunkMem::Heap(v) => v.as_ptr() as *mut u8,
            ChunkMem::Pinned(p) => p.ptr(),
        }
    }

    fn bytes(&self) -> usize {
        match &self.mem {
            ChunkMem::Heap(v) => v.len(),
            ChunkMem::Pinned(p) => p.bytes(),
        }
    }

    fn is_pinned(&self) -> bool {
        matches!(self.mem, ChunkMem::Pinned(_))
    }
}

/// 塊の中の 1 区間 = expert 1 本ぶんの生バイト。
///
/// `Arc<Region>` を clone すれば、LRU から追い出されてもバイト列は生きる
/// (追い出し側は `Arc::try_unwrap` が成功したときだけ場所を再利用する)。
pub struct Region {
    chunk: Arc<Chunk>,
    off: usize,
    len: usize,
}

impl Region {
    fn new(chunk: Arc<Chunk>, off: usize, len: usize) -> Arc<Region> {
        debug_assert!(off + len <= chunk.bytes(), "Region が塊の外を指す");
        Arc::new(Region { chunk, off, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.chunk.as_ptr().add(self.off), self.len) }
    }

    /// 書き込み用。**この `Region` を排他で持っている** (他の `Arc` が無い) ときだけ
    /// 呼ぶ。同じ塊の他の区間とは重ならないので、区間ごとの排他で足りる。
    ///
    /// # Safety
    /// 呼び出し側が同じ区間への読み書きを同時に行わないこと。
    unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.chunk.as_ptr().add(self.off), self.len)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_pinned(&self) -> bool {
        self.chunk.is_pinned()
    }
}

impl std::ops::Deref for Region {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl std::fmt::Debug for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Region")
            .field("len", &self.len)
            .field("pinned", &self.is_pinned())
            .finish()
    }
}

/// 塊の確保器。**可変長の区間**を塊から切り出す (task 260 の RAM tier)。
///
/// 1 塊 = 1 GiB (必要になったときに 1 枚ずつ確保する)。区間の長さは expert の
/// 実バイト数なので、行列ごとの最大値で丸めない (IQ3_S の mixed 量子化で
/// 1.65 倍の無駄が出た: lab t260a)。確保の前に `/proc/meminfo` の
/// `MemAvailable` を見て、`margin_bytes` を割るならそれ以上伸ばさない。
struct FixedStore {
    /// 1 区間の上限 (行列の最大 expert バイト数)。
    slot_bytes: usize,
    /// これ以上確保しない (この行列の予算)。
    limit_bytes: u64,
    /// 塊を増やす前に残っていなければならない空き (バイト)。
    margin_bytes: u64,
    /// pinned を試すか (`DENNO_EXPERT_RAM_PIN=0` で false)。
    want_pinned: bool,
    inner: Mutex<FixedInner>,
}

struct FixedInner {
    chunks: Vec<Arc<Chunk>>,
    /// いま区間を切り出している塊の番号。
    cur: usize,
    /// その塊の中の次の空き。
    off: usize,
    /// 追い出しで返ってきた区間 (chunk, off, len)。
    free: Vec<(Arc<Chunk>, usize, usize)>,
    /// 確保済みのバイト数 (会計)。
    used: u64,
    /// 予算切れ / 空き不足で伸ばすのをやめた。
    stop: bool,
    /// 読者が居るせいで予算へ戻せなかった区間の本数 (task 907 の診断)。
    lost: u64,
}

/// 1 塊のバイト数。
const RAM_CHUNK: usize = 1 << 30;

impl FixedStore {
    fn new(slot_bytes: usize, limit_bytes: u64, margin_bytes: u64, want_pinned: bool) -> Self {
        FixedStore {
            slot_bytes: slot_bytes.max(1),
            limit_bytes,
            margin_bytes,
            want_pinned,
            inner: Mutex::new(FixedInner {
                chunks: Vec::new(),
                cur: 0,
                off: 0,
                free: Vec::new(),
                used: 0,
                stop: false,
                lost: 0,
            }),
        }
    }

    /// `len` バイトの区間を返す。予算切れ・空き不足なら `None`
    /// (呼び出し側はキャッシュしない = 毎回 mmap 経路)。
    ///
    /// task 907: (a) 大きい空き区間から切り出した**余りを空きへ返す**
    /// (捨てると混合量子化で入れ替えのたびにその差分が予算から消え、
    /// `used` だけが増えて `take` が失敗し続ける = 再調整の thrash)。
    /// (b) 塊の大きさは `RAM_CHUNK` ではなく**その塊の実バイト**で判定する。
    /// (c) `stop` は「直近の確保で伸ばせなかった」の意味にし、成功したら戻す
    /// (一度の空き不足で tier が永久に閉じない)。
    fn take(&self, len: usize) -> Option<Arc<Region>> {
        if len > self.slot_bytes {
            return None;
        }
        let mut g = lock_fixed(&self.inner);
        // 1. 返ってきた区間のうち、収まるものを優先する (小さい順に探す)。
        let pick = g
            .free
            .iter()
            .enumerate()
            .filter(|(_, &(_, _, l))| l >= len)
            .min_by_key(|(_, &(_, _, l))| l)
            .map(|(i, _)| i);
        if let Some(i) = pick {
            let (chunk, off, l) = g.free.swap_remove(i);
            if l > len {
                // 余りは捨てずに空きへ返す (予算の実バイトを失わない)。
                g.free.push((Arc::clone(&chunk), off + len, l - len));
            }
            g.stop = false;
            return Some(Region::new(chunk, off, len));
        }
        // 2. 予算の範囲で次の区間を切り出す。
        if g.used + len as u64 > self.limit_bytes {
            g.stop = true;
            return None;
        }
        // 新しい塊が要るか。1 塊の上限は RAM_CHUNK だが、予算が小さいときは
        // 塊も小さいので、**実バイト**で判定する (越えると Region が塊の外を指す)。
        let need_chunk = match g.chunks.get(g.cur) {
            Some(c) => g.off + len > c.bytes(),
            None => true,
        };
        if need_chunk {
            let new_bytes = RAM_CHUNK.min(self.limit_bytes as usize).max(len);
            // 空きが margin を割るなら伸ばさない。
            if let Some(avail) = mem_available() {
                if avail < self.margin_bytes.saturating_add(new_bytes as u64) {
                    g.stop = true;
                    return None;
                }
            }
            let (chunk, _pinned) = Chunk::new(new_bytes, self.want_pinned);
            g.chunks.push(chunk);
            g.cur = g.chunks.len() - 1;
            g.off = 0;
        }
        let chunk = Arc::clone(&g.chunks[g.cur]);
        let off = g.off;
        g.off += len;
        g.used += len as u64;
        g.stop = false;
        Some(Region::new(chunk, off, len))
    }

    /// 追い出した区間を返す (読者が居なければ再利用する)。
    ///
    /// 読者が居る (`Arc` が残っている) 区間は返せず、そのバイトは予算から
    /// 戻らない。診断用に `lost` へ数える (task 907)。
    fn give_back(&self, region: Arc<Region>) {
        let Ok(r) = Arc::try_unwrap(region) else {
            lock_fixed(&self.inner).lost += 1;
            return; // まだ読者が居る
        };
        let (len, off) = (r.len(), r.off);
        let mut g = lock_fixed(&self.inner);
        g.free.push((r.chunk, off, len));
    }

    /// 読者が居るせいで予算へ戻せなかった区間の本数 (task 907)。
    fn lost(&self) -> u64 {
        lock_fixed(&self.inner).lost
    }

    fn pinned_bytes(&self) -> u64 {
        let g = lock_fixed(&self.inner);
        g.chunks
            .iter()
            .filter(|c| c.is_pinned())
            .map(|c| c.bytes() as u64)
            .sum()
    }

    fn reserved_bytes(&self) -> u64 {
        lock_fixed(&self.inner).used
    }

    fn stopped(&self) -> bool {
        lock_fixed(&self.inner).stop
    }
}

fn lock_fixed(m: &Mutex<FixedInner>) -> std::sync::MutexGuard<'_, FixedInner> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `MemAvailable` (バイト)。読めなければ `None` (このときは伸ばす判断を保留せず、
/// 予算だけで止める)。
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

/// 常駐先。従来のヒープ (プール) と、task 260 の固定長ストア。
enum Store {
    Heap(Mutex<Vec<Vec<u8>>>),
    Fixed(FixedStore),
}

impl Store {
    /// 区間を 1 本確保する (固定長ストアのみ)。ヒープは `None` (Vec を使う)。
    fn take(&self, len: usize) -> Option<Arc<Region>> {
        match self {
            Store::Fixed(f) => f.take(len),
            Store::Heap(_) => None,
        }
    }

    /// task 907: 固定長ストアか (RAM tier か)。
    ///
    /// RAM tier では「予算外は載せない」ので、ヒープへの逃げ道を作らない
    /// ([`ExpertCache::get`] を参照)。
    fn is_fixed(&self) -> bool {
        matches!(self, Store::Fixed(_))
    }

    /// ヒープの作業域を 1 本 (再利用プールから)。
    fn take_vec(&self, len: usize) -> Vec<u8> {
        let mut v = match self {
            Store::Heap(p) => lock_vecs(p).pop().unwrap_or_default(),
            Store::Fixed(_) => Vec::new(),
        };
        if v.len() < len {
            // `resize` は伸びたぶんだけを 0 で埋める (既存の中身は使わない)。
            v.resize(len, 0);
        } else {
            v.truncate(len);
        }
        v
    }

    /// 追い出した区間を返す。読者が居なければ再利用する。
    fn recycle(&self, region: Arc<Region>) {
        match self {
            Store::Fixed(f) => f.give_back(region),
            Store::Heap(p) => {
                let Ok(r) = Arc::try_unwrap(region) else {
                    return;
                };
                let Region { chunk, .. } = r;
                if let Ok(c) = Arc::try_unwrap(chunk) {
                    if let ChunkMem::Heap(v) = c.mem {
                        let mut g = lock_vecs(p);
                        if g.len() < 64 {
                            g.push(v);
                        }
                    }
                }
            }
        }
    }

    fn pinned_bytes(&self) -> u64 {
        match self {
            Store::Fixed(f) => f.pinned_bytes(),
            Store::Heap(_) => 0,
        }
    }

    fn reserved_bytes(&self) -> u64 {
        match self {
            Store::Fixed(f) => f.reserved_bytes(),
            Store::Heap(_) => 0,
        }
    }

    /// task 907: 読者が居て予算へ戻せなかった区間の本数。
    fn lost(&self) -> u64 {
        match self {
            Store::Fixed(f) => f.lost(),
            Store::Heap(_) => 0,
        }
    }

    fn stopped(&self) -> bool {
        match self {
            Store::Fixed(f) => f.stopped(),
            Store::Heap(_) => false,
        }
    }
}

fn lock_vecs(m: &Mutex<Vec<Vec<u8>>>) -> std::sync::MutexGuard<'_, Vec<Vec<u8>>> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// ロックで守る状態。
///
/// LRU は `prev`/`next` の双方向リストで持つ (ヒットも追い出しも O(1))。
/// `head` が最新、`tail` が最古。`free` は未使用スロットの番号。
struct Inner {
    map: HashMap<ExpertKey, u32>,
    slots: Vec<Option<Slot>>,
    prev: Vec<u32>,
    next: Vec<u32>,
    head: u32,
    tail: u32,
    free: Vec<u32>,
    stats: CacheStats,
}

impl Inner {
    fn new(cap: usize) -> Self {
        Inner {
            map: HashMap::with_capacity(cap.min(1 << 16)),
            slots: (0..cap).map(|_| None).collect(),
            prev: vec![NIL; cap],
            next: vec![NIL; cap],
            head: NIL,
            tail: NIL,
            // pop() で 0 から使う
            free: (0..cap as u32).rev().collect(),
            stats: CacheStats::default(),
        }
    }

    fn unlink(&mut self, s: u32) {
        let i = s as usize;
        let (p, n) = (self.prev[i], self.next[i]);
        if p == NIL {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NIL {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
        self.prev[i] = NIL;
        self.next[i] = NIL;
    }

    fn push_front(&mut self, s: u32) {
        let i = s as usize;
        self.prev[i] = NIL;
        self.next[i] = self.head;
        if self.head == NIL {
            self.tail = s;
        } else {
            self.prev[self.head as usize] = s;
        }
        self.head = s;
    }

    fn touch(&mut self, s: u32) {
        self.unlink(s);
        self.push_front(s);
    }

    /// スロットを確保して `key` を入れる。満杯なら tail を追い出す。
    ///
    /// task 907: 同じ鍵が別スロットに居る (len 違い) ときは**先に外して区間を返す**。
    /// 外さずに `map` を上書きすると (a) 古い区間が空きへ戻らず予算から消え、
    /// (b) 古いスロットが tail だった場合に `unlink` が二度走って LRU の
    /// リスト (head/tail) が壊れる。
    fn insert(&mut self, key: ExpertKey, len: usize, buf: Arc<Region>, store: &Store) {
        if let Some(prev) = self.map.remove(&key) {
            self.unlink(prev);
            if let Some(old) = self.slots[prev as usize].take() {
                // 読者が居なければ場所を返す (会計から漏らさない)。
                store.recycle(old.buf);
            }
            self.free.push(prev);
        }
        let sid = match self.free.pop() {
            Some(s) => s,
            None => {
                let v = self.tail;
                debug_assert_ne!(v, NIL, "cap>0 なのに LRU が空");
                self.unlink(v);
                let old = self.slots[v as usize].take().expect("tail は必ず有効");
                if old.key != key {
                    self.map.remove(&old.key);
                }
                self.stats.evictions += 1;
                // 読者が居なければ場所を返す (ヒープはゼロ初期化を避ける)。
                store.recycle(old.buf);
                v
            }
        };
        self.map.insert(key, sid);
        self.slots[sid as usize] = Some(Slot { key, len, buf });
        self.push_front(sid);
    }

    /// task 907: 一番古い (`tail`) を外して区間を予算へ返す。
    ///
    /// 固定長ストアで予算が尽きたとき、`take` の前に LRU を 1 本ゆずるために使う
    /// (以前はヒープへ逃げていた = 予算外の常駐が増えた)。
    /// 使用中の読者が居る区間は返せない (その場合は予算が戻らない)。
    ///
    /// 戻り値は「1 本外したか」。
    fn evict_tail(&mut self, store: &Store) -> bool {
        let v = self.tail;
        if v == NIL {
            return false;
        }
        self.unlink(v);
        let Some(old) = self.slots[v as usize].take() else {
            return false;
        };
        self.map.remove(&old.key);
        self.stats.evictions += 1;
        store.recycle(old.buf);
        self.free.push(v);
        true
    }
}

/// expert 生バイトの LRU キャッシュ (スレッド間で共有できる)。
///
/// 中身は `slots` 本のスロット (各 `slot_bytes` 以下)。確保は初回アクセスまで
/// 遅延する (`slots == 0` なら 1 バイトも確保しない)。
pub struct ExpertCache {
    slots: usize,
    slot_bytes: usize,
    inner: Mutex<Inner>,
    /// task 260: 常駐先 (ヒープのプール / 固定長の塊)。
    store: Store,
}

impl std::fmt::Debug for ExpertCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let used = match self.inner.lock() {
            Ok(g) => g.map.len(),
            Err(_) => usize::MAX,
        };
        f.debug_struct("ExpertCache")
            .field("slots", &self.slots)
            .field("slot_bytes", &self.slot_bytes)
            .field("used", &used)
            .field("stats", &self.stats())
            .finish()
    }
}

impl ExpertCache {
    /// `slots` 本のスロットを持つキャッシュを作る (`slot_bytes` は最大の
    /// expert バイト数)。バッファは初回ミスで確保する。
    pub fn new(slots: usize, slot_bytes: usize) -> Self {
        ExpertCache {
            slots,
            slot_bytes,
            inner: Mutex::new(Inner::new(slots)),
            store: Store::Heap(Mutex::new(Vec::new())),
        }
    }

    /// task 260: **固定長の塊**から切り出すキャッシュを作る (RAM tier)。
    ///
    /// `limit_bytes` はこの行列に使ってよい総バイト数、`margin_bytes` は塊を
    /// 増やす前に `MemAvailable` に残すべきバイト数、`want_pinned` は
    /// `cuMemHostAlloc` を試すか (失敗したらヒープ)。
    ///
    /// `slots` は上限 (バイト予算から決まる) で、塊は必要になったときに
    /// 1 GiB ずつ確保する。
    pub fn new_fixed(
        slots: usize,
        slot_bytes: usize,
        limit_bytes: u64,
        margin_bytes: u64,
        want_pinned: bool,
    ) -> Self {
        // LRU の本数は呼び出し側の見積り (`slots`) をそのまま使う。区間の
        // 長さは実バイトなので、バイト予算は `limit_bytes` で別に持つ。
        let limit_slots = if limit_bytes == 0 { 0 } else { slots };
        ExpertCache {
            slots: limit_slots,
            slot_bytes,
            inner: Mutex::new(Inner::new(limit_slots)),
            store: Store::Fixed(FixedStore::new(
                slot_bytes,
                limit_bytes,
                margin_bytes,
                want_pinned,
            )),
        }
    }

    /// 固定長ストアで pinned に置けているバイト数 (診断用。ヒープなら 0)。
    pub fn pinned_bytes(&self) -> u64 {
        self.store.pinned_bytes()
    }

    /// 固定長ストアが既に切り出したバイト数 (予算の使用量)。
    pub fn reserved_bytes(&self) -> u64 {
        self.store.reserved_bytes()
    }

    /// task 907: 読者が居て予算へ戻せなかった区間の本数 (診断用)。
    ///
    /// 0 でないときは、追い出したのに `used` が減っていない = そのバイトは
    /// 二度と使えない (使用中の `Arc` が drop されるまで)。
    pub fn lost_regions(&self) -> u64 {
        self.store.lost()
    }

    /// 空き不足 / 予算切れで伸ばすのを止めたか。
    pub fn stopped(&self) -> bool {
        self.store.stopped()
    }

    /// スロット数。
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// 1 スロットのバイト数。
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// いま常駐しているキー (= expert) の数 (P54 の `ram_experts`)。
    ///
    /// スロットの中身は追い出しでしか減らないので、`map.len()` がそのまま
    /// 「載っている本数」になる。
    pub fn used(&self) -> usize {
        lock_inner(&self.inner).map.len()
    }

    /// 鍵に対応する生バイトを返す。未命中なら `src` (pread 元) から読む。
    ///
    /// `off` は **ファイル内の絶対オフセット** ([`crate::gguf::GgufFile::tensor_mapping_parts`]
    /// の戻り値 + [`crate::model::moe_expert::ExpertSlice::off`])。
    /// 戻り値は `Arc` なので、呼び出し側が保持している間もスロットを追い出せる
    /// (バイト列は `Arc` の参照が切れるまで生きる)。
    ///
    /// ヒット時はロックを数 ns 持つだけ、ミス時の pread は **ロックの外**で行う。
    pub fn get(
        &self,
        key: ExpertKey,
        off: u64,
        len: usize,
        src: &dyn PreadAt,
    ) -> Result<Arc<Region>> {
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

        // slots == 0: キャッシュしない。毎回 pread して返す。
        if self.slots == 0 {
            let mut buf = vec![0u8; len];
            let n = read_into(src, off, &mut buf)?;
            let mut g = lock_inner(&self.inner);
            g.stats.misses += 1;
            g.stats.bytes_read += n as u64;
            return Ok(Region::new(Chunk::from_vec(buf), 0, len));
        }

        // ヒット判定 (ロック内)。Arc を clone したらすぐ離す。
        {
            let mut g = lock_inner(&self.inner);
            if let Some(&sid) = g.map.get(&key) {
                let slot = g.slots[sid as usize].as_ref().expect("map の指す slot は有効");
                if slot.len == len {
                    let buf = slot.buf.clone();
                    g.touch(sid);
                    g.stats.hits += 1;
                    return Ok(buf);
                }
            }
            g.stats.misses += 1;
        }

        // ミス: 書き込み先を 1 本用意して、ロックの外で読む。
        //
        // task 260: 固定長ストアでは**区間へ直接 pread する** (pinned メモリに
        // そのまま入るので、ヒープへ読んでから写すコピーが無い)。
        let mut region = self.store.take(len);
        // task 907: RAM tier (固定長ストア) では**ヒープへ逃げない**。
        // 逃げると予算外の常駐が増え、pinned の意味 (page cache に追い出されない)
        // も予算の会計も壊れる。代わりに LRU の古い順に 1 本ずつゆずって
        // もう一度取り、それでも入らなければ諦める (呼び出し側は mmap 経路)。
        if region.is_none() && self.store.is_fixed() {
            loop {
                // スロットが埋まっているときだけ LRU をゆずる。空きスロットが
                // あるのに予算が足りない = スロット数の見積りが実バイトより
                // 大きい (混合量子化の丸め) なので、ゆずっても載らない。
                // ここで止めて `stop` を立て、呼び出し側 (充填) に伝える。
                let (evicted, full) = {
                    let mut g = lock_inner(&self.inner);
                    let full = g.map.len() >= self.slots;
                    (if full { g.evict_tail(&self.store) } else { false }, full)
                };
                if !full || !evicted {
                    break;
                }
                region = self.store.take(len);
                if region.is_some() {
                    break;
                }
            }
        }
        if region.is_none() && self.store.is_fixed() {
            bail!(
                "{:?}: RAM tier の予算が尽きた ({} バイトを載せられない)",
                key,
                len
            );
        }
        let mut scratch: Option<Vec<u8>> = None;
        if region.is_none() {
            scratch = Some(self.store.take_vec(len));
        }
        let n = {
            let dst: &mut [u8] = match (&mut region, &mut scratch) {
                (Some(r), _) => {
                    // take 直後で参照は 1 本だけ (書き込みは排他)。
                    let r = Arc::get_mut(r).expect("take 直後の Region は排他");
                    unsafe { r.as_mut_slice() }
                }
                (None, Some(v)) => v.as_mut_slice(),
                (None, None) => unreachable!("書き込み先が無い"),
            };
            match read_into(src, off, dst) {
                Ok(n) => n,
                Err(e) => {
                    // 失敗した作業域は返さない (半端な内容を再利用しない)。
                    return Err(e);
                }
            }
        };
        let arc = match region {
            Some(r) => r,
            None => {
                let v = scratch.expect("書き込み先が無い");
                let n = v.len();
                Region::new(Chunk::from_vec(v), 0, n)
            }
        };
        {
            let mut g = lock_inner(&self.inner);
            g.stats.bytes_read += n as u64;
            g.insert(key, len, arc.clone(), &self.store);
        }
        Ok(arc)
    }

    pub fn stats(&self) -> CacheStats {
        lock_inner(&self.inner).stats
    }

    /// task 260: ヒットだけを引く (ミスで pread しない)。
    pub fn peek(&self, key: ExpertKey, len: usize) -> Option<Arc<Region>> {
        let mut g = lock_inner(&self.inner);
        let sid = *g.map.get(&key)?;
        let hit = match g.slots[sid as usize].as_ref() {
            Some(s) => s.len == len,
            None => false,
        };
        if !hit {
            return None;
        }
        let buf = g.slots[sid as usize].as_ref().expect("有効").buf.clone();
        g.touch(sid);
        g.stats.hits += 1;
        Some(buf)
    }

    /// 統計だけを 0 に戻す (スロットの中身と LRU の順序はそのまま)。
    pub fn reset_stats(&mut self) {
        lock_inner(&self.inner).stats = CacheStats::default();
    }

    /// task 263: いま載っている鍵の一覧 (順序は不定)。
    ///
    /// 再調整が「いまの集合」と「欲しい集合」を比べるのに使う。
    pub fn keys(&self) -> Vec<ExpertKey> {
        lock_inner(&self.inner).map.keys().copied().collect()
    }

    /// task 263: 指定した鍵を落とす (載っていなければ false)。
    ///
    /// 読者が `Arc` を持っている間はバイト列が生き続ける (スロット番号だけ
    /// 先に返す)。固定長ストアでは区間も返却する。
    pub fn evict(&self, key: ExpertKey) -> bool {
        let mut g = lock_inner(&self.inner);
        let Some(sid) = g.map.remove(&key) else {
            return false;
        };
        if let Some(old) = g.slots[sid as usize].take() {
            g.unlink(sid);
            g.store_free(sid);
            self.store.recycle(old.buf);
            return true;
        }
        false
    }

    /// 常駐している実バイト数 (スロット数 x スロット長)。
    pub fn used_bytes(&self) -> u64 {
        let g = lock_inner(&self.inner);
        g.map
            .values()
            .filter_map(|&sid| g.slots[sid as usize].as_ref())
            .map(|s| s.len as u64)
            .sum()
    }
}

impl Inner {
    /// スロット番号を空きに戻す (`evict` 用)。
    fn store_free(&mut self, sid: u32) {
        self.free.push(sid);
    }
}

/// 毒されたロックは中身を取り戻して使う (キャッシュは数値の正しさに無関係)。
fn lock_inner(m: &Mutex<Inner>) -> std::sync::MutexGuard<'_, Inner> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// `dst` を読み切る。戻り値が要求長と違えばエラー (短読みを成功にしない)。
fn read_into(src: &dyn PreadAt, off: u64, dst: &mut [u8]) -> Result<usize> {
    let want = dst.len();
    let n = src.pread_at(off, dst)?;
    if n != want {
        bail!("pread が {want} B 要求に対して {n} B しか返さない (off={off}, EOF か短読み)");
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// P-MOE-G1 / P38: 3 行列ぶんの束 (forward 経路への配線)
// ---------------------------------------------------------------------------

/// `DENNO_EXPERT_CACHE_MB`: expert キャッシュに使う MB (**行列ごと**)。**既定 0 = 無効**
/// (このとき forward は今までどおり mmap + `prefetch()` の経路を通る)。
/// 3 本あわせた常駐は約 `3 * MB` MiB。
pub const EXPERT_CACHE_MB_ENV: &str = "DENNO_EXPERT_CACHE_MB";

/// `DENNO_EXPERT_CACHE_CAP_MB`: 常駐の**総量**上限 (3 行列合計・MiB)。
/// 未設定は無制限。`DENNO_EXPERT_CACHE_MB` で要求した容量がこれを超えるときは
/// 比例縮小する (上限は引数 [`ExpertCacheSet::from_mb_capped`] で受ける)。
pub const EXPERT_CACHE_CAP_MB_ENV: &str = "DENNO_EXPERT_CACHE_CAP_MB";

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

/// `DENNO_EXPERT_CACHE_CAP_MB` の解釈。未設定・空・数値でない値は `None`
/// (上限なし)。"0" は `Some(0)` (常駐させない)。
pub fn cap_bytes_from_env() -> Option<u64> {
    parse_cap_bytes(std::env::var(EXPERT_CACHE_CAP_MB_ENV).ok().as_deref())
}

/// [`cap_bytes_from_env`] の純粋部分。
fn parse_cap_bytes(v: Option<&str>) -> Option<u64> {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() => match s.parse::<u64>() {
            Ok(mb) => Some(mb.saturating_mul(1024 * 1024)),
            Err(_) => None,
        },
        _ => None,
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

/// 総量上限 `cap_bytes` に収まるようにスロット数を比例縮小する (純関数)。
///
/// 返り値は `Σ out[i] * slot_bytes[i] <= cap_bytes` を満たす (切り捨てなので
/// 余ることはあっても超えない)。`cap_bytes == 0` は全部 0。
pub fn slots_within_cap(slots: [usize; 3], slot_bytes: [usize; 3], cap_bytes: u64) -> [usize; 3] {
    let total: u128 = (0..3)
        .map(|i| slots[i] as u128 * slot_bytes[i] as u128)
        .sum();
    if total <= cap_bytes as u128 {
        return slots;
    }
    if cap_bytes == 0 || total == 0 {
        return [0, 0, 0];
    }
    let mut out = [0usize; 3];
    for i in 0..3 {
        out[i] = ((slots[i] as u128 * cap_bytes as u128) / total) as usize;
    }
    out
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

/// task 260: RAM tier の 1 行列ぶんの指定。
///
/// 区間は expert の実バイト数で切るので、予算の配分と LRU の本数は
/// **実バイトの合計**から決める (最大値で丸めると IQ3_S で 1.65 倍の無駄)。
#[derive(Clone, Copy, Debug)]
pub struct RamSpec {
    /// その行列の最大 expert バイト数 (1 区間の上限)。
    pub slot_bytes: usize,
    /// その行列の expert 総バイト数 (モデル全体)。
    pub total_bytes: u64,
    /// **モデル全体**の expert 本数 (層数 × 層あたりの expert 数)。
    ///
    /// ここを層あたりの本数にすると LRU の本数が 1/層数 になり、lab t260b で
    /// 46 GB の予算に対して 493 本しか持てず、decode が 17 GB 読み直して
    /// 3 倍遅くなった。
    pub n_expert: usize,
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
/// [`Drop`] は `DENNO_EXPERT_CACHE_STATS=1` のときだけ 3 行を stderr に出す。
pub struct ExpertCacheSet {
    mb: usize,
    caches: [ExpertCache; 3],
    readers: Vec<Box<dyn PreadAt>>,
    /// task 260: **ヒットだけ**を引く (ミスで pread しない) モード。RAM tier 用。
    ///
    /// ミスしたら呼び出し側が mmap のスライスへ落ちる。page cache にある
    /// バイト列をわざわざ読み直して tier を埋めるのは、lab t260a/b で
    /// decode を 29〜54% 遅くした (読み直し + page cache の追い出し)。
    peek_only: bool,
}

impl ExpertCacheSet {
    /// 3 本ぶんのキャッシュを作る。`specs` の順は [`ExpertMatrix::ALL`] と同じ。
    ///
    /// スロット数は行列ごとに `MB * 1024 * 1024 / slot_bytes` (切り捨て・最低 1)。
    /// `slot_bytes` は**その行列の最大 expert** でなければならない
    /// (小さいと [`ExpertCache::get`] が `len > slot_bytes` で落ちる = 静かに
    /// 切り詰めない)。`paths` は GGUF のパート 0..n の順。
    pub fn from_mb(mb: usize, specs: [CacheSpec; 3], paths: &[PathBuf]) -> Result<Self> {
        Self::from_mb_capped(mb, None, specs, paths)
    }

    /// [`ExpertCacheSet::from_mb`] の総量上限つき。
    ///
    /// `cap_bytes` は 3 行列合計の常駐上限。`None` は無制限。超える要求は
    /// [`slots_within_cap`] で比例縮小する (VRAM / RAM の予算と衝突させないため、
    /// 上限はグローバルに読まず **引数** で受ける)。
    pub fn from_mb_capped(
        mb: usize,
        cap_bytes: Option<u64>,
        specs: [CacheSpec; 3],
        paths: &[PathBuf],
    ) -> Result<Self> {
        let mut slots = [
            slots_for_mb(mb, specs[0].slot_bytes),
            slots_for_mb(mb, specs[1].slot_bytes),
            slots_for_mb(mb, specs[2].slot_bytes),
        ];
        let bytes = [
            specs[0].slot_bytes,
            specs[1].slot_bytes,
            specs[2].slot_bytes,
        ];
        if let Some(cap) = cap_bytes {
            slots = slots_within_cap(slots, bytes, cap);
        }
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
        let mut readers: Vec<Box<dyn PreadAt>> = Vec::with_capacity(paths.len());
        for p in paths {
            readers.push(Box::new(FileReader::open(p)?));
        }
        Ok(ExpertCacheSet {
            mb,
            caches: [
                ExpertCache::new(slots[0], slot_bytes[0]),
                ExpertCache::new(slots[1], slot_bytes[1]),
                ExpertCache::new(slots[2], slot_bytes[2]),
            ],
            readers,
            peek_only: false,
        })
    }

    /// task 260: **RAM tier** を作る。総バイト予算 `budget_bytes` を 3 行列へ
    /// 実バイトの比で配る (1 expert = 3 行列がそろって入るので、行列あたりの
    /// 本数は同じになる)。
    ///
    /// 常駐先は `cuMemHostAlloc` の pinned (取れなければヒープ)。塊を増やす前に
    /// `MemAvailable` が `margin_bytes` を割っていないか確かめる。
    pub fn new_ram(
        budget_bytes: u64,
        specs: [RamSpec; 3],
        paths: &[PathBuf],
        margin_bytes: u64,
        want_pinned: bool,
    ) -> Result<Self> {
        if paths.is_empty() {
            bail!("expert RAM tier: GGUF のパートが 0 本 (読み口を作れない)");
        }
        let total: u64 = specs
            .iter()
            .map(|s| s.total_bytes.max(1))
            .sum::<u64>()
            .max(1);
        // 行列ごとの予算は実バイトの比で配る。LRU の本数は
        // 「予算 × 行列の expert 本数 / 全体バイト数」(1 本あたりの平均で割る)。
        let mut limit = [0u64; 3];
        let mut slots = [0usize; 3];
        for (i, s) in specs.iter().enumerate() {
            limit[i] = (budget_bytes as u128 * s.total_bytes.max(1) as u128 / total as u128) as u64;
            slots[i] = ((limit[i] as u128 * s.n_expert as u128) / s.total_bytes.max(1) as u128)
                as usize;
        }
        let mut readers: Vec<Box<dyn PreadAt>> = Vec::with_capacity(paths.len());
        for p in paths {
            readers.push(Box::new(FileReader::open(p)?));
        }
        Ok(ExpertCacheSet {
            mb: (budget_bytes >> 20) as usize,
            caches: [
                ExpertCache::new_fixed(
                    slots[0],
                    specs[0].slot_bytes.max(1),
                    limit[0],
                    margin_bytes,
                    want_pinned,
                ),
                ExpertCache::new_fixed(
                    slots[1],
                    specs[1].slot_bytes.max(1),
                    limit[1],
                    margin_bytes,
                    want_pinned,
                ),
                ExpertCache::new_fixed(
                    slots[2],
                    specs[2].slot_bytes.max(1),
                    limit[2],
                    margin_bytes,
                    want_pinned,
                ),
            ],
            readers,
            peek_only: true,
        })
    }

    /// task 260: ヒットだけを引くモードか (RAM tier)。
    pub fn peek_only(&self) -> bool {
        self.peek_only
    }

    /// task 263: 常駐している expert の鍵 (gate 行列から。3 行列は同じ集合)。
    pub fn keys(&self) -> Vec<ExpertKey> {
        self.caches[ExpertMatrix::Gate.index()].keys()
    }

    /// task 263: 全行列から鍵を落とす。1 本でも落ちたら true。
    pub fn evict(&self, key: ExpertKey) -> bool {
        let mut any = false;
        for c in &self.caches {
            any |= c.evict(key);
        }
        any
    }

    /// task 263: 常駐している実バイト数 (3 行列の和)。
    pub fn used_bytes(&self) -> u64 {
        self.caches.iter().map(|c| c.used_bytes()).sum()
    }

    /// pinned に置けているバイト数 (3 行列の和)。RAM tier の診断用。
    pub fn pinned_bytes(&self) -> u64 {
        self.caches.iter().map(|c| c.pinned_bytes()).sum()
    }

    /// 塊として既に確保したバイト数 (予算の使用量)。
    pub fn reserved_bytes(&self) -> u64 {
        self.caches.iter().map(|c| c.reserved_bytes()).sum()
    }

    /// 空き不足 / 予算切れで伸ばすのを止めた行列があるか。
    pub fn stopped(&self) -> bool {
        self.caches.iter().any(|c| c.stopped())
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

    /// 行列ごとの常駐上限 (バイト)。`DENNO_EXPERT_CACHE_STATS` の 1 行に出す。
    pub fn resident_bytes(&self, m: ExpertMatrix) -> u64 {
        self.slots(m) as u64 * self.slot_bytes(m) as u64
    }

    /// いま RAM に常駐している expert の数 (P54 の `ram_experts`)。
    ///
    /// expert 1 本は gate / up / down の 3 行列がそろって入るので、gate 行列の
    /// スロット使用数で数える (行列ごとに LRU が独立なので、3 つの和は 3 倍になる)。
    pub fn used_experts(&self) -> usize {
        self.caches[ExpertMatrix::Gate.index()].used()
    }

    /// 3 行列の読みミス (= SSD からの pread) の累計 (P54 の `ssd_reads`)。
    pub fn total_reads(&self) -> u64 {
        ExpertMatrix::ALL.iter().map(|&m| self.stats(m).misses).sum()
    }

    /// 行列 `m` の expert 生バイトを返す。
    ///
    /// `part` は GGUF のパート番号 ([`crate::gguf::GgufFile::tensor_mapping_parts`]
    /// の 1 番目の戻り値)、`off` は**そのファイル内の絶対オフセット**。
    pub fn get(
        &self,
        m: ExpertMatrix,
        part: usize,
        key: ExpertKey,
        off: u64,
        len: usize,
    ) -> Result<Arc<Region>> {
        let n = self.readers.len();
        if part >= n {
            bail!("expert キャッシュ: パート {part} が範囲外 ({n} 本)");
        }
        self.caches[m.index()].get(key, off, len, self.readers[part].as_ref())
    }

    /// 行列ごとの統計。
    pub fn stats(&self, m: ExpertMatrix) -> CacheStats {
        self.caches[m.index()].stats()
    }

    /// task 260: **ヒットだけ**を引く (ミスで pread しない)。prefill の
    /// ストリームが RAM tier に載っている expert だけを使うときに使う。
    ///
    /// ヒットは LRU を触る (使われた事実を残す)。`len` が違うスロットは
    /// ミス扱い (別の層で同じ鍵・別の長さは起きないが、`get` と同じ規則にする)。
    pub fn peek(&self, m: ExpertMatrix, key: ExpertKey, len: usize) -> Option<Arc<Region>> {
        self.caches[m.index()].peek(key, len)
    }

    /// task 260: 統計だけを 0 に戻す (中身と LRU の順序はそのまま)。
    pub fn reset_stats_for(&self, m: ExpertMatrix) {
        let mut g = lock_inner(&self.caches[m.index()].inner);
        g.stats = CacheStats::default();
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
            "expert_cache {}: slots={} slot_bytes={} resident_mib={} hits={} misses={} hit_rate={:.3} \
             bytes_read={} evictions={}",
            m.name(),
            self.slots(m),
            self.slot_bytes(m),
            self.resident_bytes(m) / (1024 * 1024),
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

    /// 決定的なバイト列 (位置ごとに値が違うので、オフセットのずれが即座に分かる)。
    fn pat(n: usize) -> Vec<u8> {
        (0..n)
            .map(|i| (i.wrapping_mul(37).wrapping_add(11) & 0xff) as u8)
            .collect()
    }

    /// `Vec<u8>` をファイルに見立てるテスト用の口 (原子カウンタなので `&self`)。
    struct Mem {
        data: Vec<u8>,
        reads: std::sync::atomic::AtomicUsize,
        bytes: std::sync::atomic::AtomicU64,
    }

    impl Mem {
        fn new(data: Vec<u8>) -> Self {
            Mem {
                data,
                reads: std::sync::atomic::AtomicUsize::new(0),
                bytes: std::sync::atomic::AtomicU64::new(0),
            }
        }
        fn reads(&self) -> usize {
            self.reads.load(std::sync::atomic::Ordering::Relaxed)
        }
        fn bytes(&self) -> u64 {
            self.bytes.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    impl PreadAt for Mem {
        fn pread_at(&self, off: u64, buf: &mut [u8]) -> Result<usize> {
            self.reads.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let off = off as usize;
            if off >= self.data.len() {
                return Ok(0); // EOF
            }
            let n = buf.len().min(self.data.len() - off);
            buf[..n].copy_from_slice(&self.data[off..off + n]);
            self.bytes
                .fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
            Ok(n)
        }
    }

    fn key(layer: u16, expert: u16) -> ExpertKey {
        ExpertKey { layer, expert }
    }

    #[test]
    fn parse_mb_accepts_only_positive_integers() {
        assert_eq!(parse_mb(None), 0);
        assert_eq!(parse_mb(Some("")), 0);
        assert_eq!(parse_mb(Some("  ")), 0);
        assert_eq!(parse_mb(Some("0")), 0);
        assert_eq!(parse_mb(Some("off")), 0);
        assert_eq!(parse_mb(Some("-1")), 0);
        assert_eq!(parse_mb(Some("1")), 1);
        assert_eq!(parse_mb(Some("4096")), 4096);
        assert_eq!(parse_mb(Some(" 512 ")), 512);
    }

    #[test]
    fn cap_env_is_unset_by_default_and_zero_means_zero() {
        assert_eq!(parse_cap_bytes(None), None);
        assert_eq!(parse_cap_bytes(Some("")), None);
        assert_eq!(parse_cap_bytes(Some("abc")), None);
        assert_eq!(parse_cap_bytes(Some("0")), Some(0));
        assert_eq!(parse_cap_bytes(Some("16")), Some(16 * 1024 * 1024));
    }

    #[test]
    fn slots_for_mb_floors_and_keeps_at_least_one() {
        let mb = 1024 * 1024; // 1 MiB
        assert_eq!(slots_for_mb(1, mb), 1);
        assert_eq!(slots_for_mb(4, mb), 4);
        assert_eq!(slots_for_mb(1, mb / 3), 3);
        assert_eq!(slots_for_mb(1, mb * 4), 1, "1 スロット未満でも 1 に寄せる");
        assert_eq!(slots_for_mb(1, 0), 1, "slot_bytes 0 でも 1 に寄せる");
        let down = 1_228_800usize;
        assert_eq!(slots_for_mb(4096, down), 4096 * 1024 * 1024 / down);
        assert!(slots_for_mb(usize::MAX, 1) > 0);
    }

    #[test]
    fn cap_shrinks_slots_and_never_exceeds_the_budget() {
        let slots = [1000usize, 1000, 1000];
        let bytes = [921_600usize, 921_600, 1_228_800];
        let total: u64 = (0..3).map(|i| slots[i] as u64 * bytes[i] as u64).sum();
        // 上限内ならそのまま
        assert_eq!(slots_within_cap(slots, bytes, total), slots);
        // 半分なら約半分 (比例縮小・切り捨て)
        let half = slots_within_cap(slots, bytes, total / 2);
        let used: u64 = (0..3).map(|i| half[i] as u64 * bytes[i] as u64).sum();
        assert!(used <= total / 2, "上限を超えた: {used} > {}", total / 2);
        assert!(used * 100 >= total * 45, "縮めすぎ: {used} vs {}", total / 2);
        // 0 は全部 0
        assert_eq!(slots_within_cap(slots, bytes, 0), [0, 0, 0]);
        // 極端に小さい上限でも 1 スロットに寄せない (0 のまま = 無効化できる)
        assert_eq!(slots_within_cap([1, 1, 1], [1_000_000; 3], 1), [0, 0, 0]);
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
    fn miss_then_hit_returns_same_bytes_and_reads_once() {
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(2, 1024);
        let k = key(0, 1);

        let a = c.get(k, 100, 512, &src).unwrap();
        assert_eq!(&a[..], &data[100..612], "未命中の戻り値がファイル内容と違う");
        assert_eq!(src.reads(), 1);

        let b = c.get(k, 100, 512, &src).unwrap();
        assert_eq!(&b[..], &data[100..612]);
        assert_eq!(src.reads(), 1, "ヒットなのに読み直している");
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.bytes_read, s.evictions), (1, 1, 512, 0));
    }

    #[test]
    fn same_key_with_other_len_is_a_miss() {
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(1, 1024);
        let k = key(3, 7);
        assert_eq!(&c.get(k, 0, 256, &src).unwrap()[..], &data[0..256]);
        assert_eq!(&c.get(k, 0, 512, &src).unwrap()[..], &data[0..512]);
        assert_eq!(src.reads(), 2, "長さ違いをヒット扱いした");
    }

    #[test]
    fn lru_evicts_the_oldest_of_three_slots() {
        let data = pat(8192);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(3, 512);
        for e in 0..3u16 {
            c.get(key(0, e), (e as u64) * 512, 512, &src).unwrap();
        }
        assert_eq!(c.stats().evictions, 0);
        c.get(key(0, 3), 3 * 512, 512, &src).unwrap();
        assert_eq!(c.stats().evictions, 1);
        // 残っている expert 1 は読み直さない
        assert_eq!(&c.get(key(0, 1), 512, 512, &src).unwrap()[..], &data[512..1024]);
        assert_eq!(src.reads(), 4);
        // 追い出された expert 0 は読み直す
        assert_eq!(&c.get(key(0, 0), 0, 512, &src).unwrap()[..], &data[0..512]);
        assert_eq!(src.reads(), 5);
        assert_eq!(c.stats().evictions, 2);
    }

    #[test]
    fn lru_order_follows_last_use_not_insertion() {
        let data = pat(8192);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(3, 512);
        for e in 0..3u16 {
            c.get(key(0, e), (e as u64) * 512, 512, &src).unwrap();
        }
        c.get(key(0, 0), 0, 512, &src).unwrap();
        c.get(key(0, 3), 3 * 512, 512, &src).unwrap();
        assert_eq!(c.stats().evictions, 1);

        let reads = src.reads();
        c.get(key(0, 0), 0, 512, &src).unwrap();
        assert_eq!(src.reads(), reads, "触り直した expert 0 が追い出された");
        c.get(key(0, 1), 512, 512, &src).unwrap();
        assert_eq!(src.reads(), reads + 1, "最終使用が最も古い expert 1 が残っていた");
    }

    #[test]
    fn len_over_slot_bytes_is_an_error() {
        let src = Mem::new(pat(4096));
        let c = ExpertCache::new(1, 1024);
        assert!(c.get(key(0, 0), 0, 1025, &src).is_err(), "スロット超過が通った");
        assert_eq!(src.reads(), 0, "エラーなのに読んでいる");
        assert!(c.get(key(0, 0), 0, 0, &src).is_err());
        assert_eq!(c.get(key(0, 0), 0, 1024, &src).unwrap().len(), 1024);
    }

    #[test]
    fn out_of_range_offset_is_an_error() {
        let src = Mem::new(pat(4096));
        let c = ExpertCache::new(2, 1024);
        assert!(c.get(key(0, 0), 4000, 512, &src).is_err(), "末尾超えが通った");
        assert!(c.get(key(0, 1), 99_999, 512, &src).is_err(), "範囲外が通った");
        assert!(c.get(key(0, 0), u64::MAX - 8, 512, &src).is_err());
    }

    #[test]
    fn failed_read_does_not_leave_a_valid_slot() {
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(1, 1024);
        let k = key(1, 2);
        assert!(c.get(k, 4000, 512, &src).is_err());
        assert_eq!(&c.get(k, 0, 512, &src).unwrap()[..], &data[0..512]);
        let s = c.stats();
        assert_eq!(s.hits, 0, "失敗した読みがヒットになった");
        assert_eq!(s.misses, 2);
        assert_eq!(s.bytes_read, 512, "失敗した読みを bytes_read に計上した");
    }

    #[test]
    fn zero_slots_reads_every_time_but_values_are_right() {
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(0, 1024);
        assert_eq!(c.slots(), 0);
        let k = key(4, 9);
        for _ in 0..3 {
            assert_eq!(&c.get(k, 128, 512, &src).unwrap()[..], &data[128..640]);
        }
        assert_eq!(src.reads(), 3);
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.evictions), (0, 3, 0));
        assert_eq!(s.bytes_read, 3 * 512);
        assert!(c.get(k, 0, 1025, &src).is_err());
    }

    #[test]
    fn stats_are_exact_over_a_mixed_sequence() {
        let data = pat(8192);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new(2, 512);
        for (e, off) in [
            (0u16, 0u64),
            (1, 512),
            (0, 0),
            (2, 1024),
            (0, 0),
            (1, 512),
            (1, 512),
        ] {
            c.get(key(0, e), off, 512, &src).unwrap();
        }
        let s = c.stats();
        assert_eq!(s.hits, 3, "hits");
        assert_eq!(s.misses, 4, "misses");
        assert_eq!(s.evictions, 2, "evictions");
        assert_eq!(s.bytes_read, 4 * 512, "bytes_read");
        assert_eq!(src.reads(), 4, "実際の読み取り回数");
    }

    #[test]
    fn reset_stats_keeps_slots_warm() {
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let mut c = ExpertCache::new(2, 1024);
        let k = key(0, 0);
        c.get(k, 0, 512, &src).unwrap();
        c.get(k, 0, 512, &src).unwrap();
        assert_eq!(c.stats().hits, 1);
        c.reset_stats();
        assert_eq!(c.stats(), CacheStats::default());
        c.get(k, 0, 512, &src).unwrap();
        assert_eq!(c.stats().hits, 1, "reset でスロットまで捨てた");
        assert_eq!(src.reads(), 1, "reset 後に読み直している");
    }

    #[test]
    fn cache_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ExpertCache>();
        assert_send_sync::<FileReader>();
    }

    #[test]
    fn concurrent_hits_of_one_key_return_the_same_bytes() {
        use std::sync::Arc as StdArc;
        let data = pat(64 * 1024);
        let src = StdArc::new(Mem::new(data.clone()));
        let c = StdArc::new(ExpertCache::new(4, 8192));
        // 先に温める
        c.get(key(0, 1), 0, 8192, src.as_ref()).unwrap();
        let mut hs = Vec::new();
        for _ in 0..8 {
            let (c, src, want) = (c.clone(), src.clone(), data[0..8192].to_vec());
            hs.push(std::thread::spawn(move || {
                for _ in 0..50 {
                    let got = c.get(key(0, 1), 0, 8192, src.as_ref()).unwrap();
                    assert_eq!(&got[..], &want[..]);
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        let s = c.stats();
        assert_eq!(s.hits, 400, "8 スレッド × 50 回が全部ヒットしていない");
        assert_eq!(src.reads(), 1, "ヒットなのに読んでいる");
    }

    // ------------------------------------------------------------------
    // ヒット率のシミュレーション (タスク 38)
    //
    // 実 expert の使用履歴は取れていないので、REPORT_P37.md §5 の仮定
    // (**support S=170 から 1 トークン 10 体を一様乱数で重複なしに選ぶ**) を置く。
    // 乱数は SplitMix64 の決定的な列で、テストは毎回同じ結果になる。
    // ------------------------------------------------------------------

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 1 層あたり `cap` 体を常駐させたときのヒット率。
    ///
    /// 実際の [`ExpertCache`] をそのまま LRU として使う (ポリシーを二重実装しない)。
    /// `slot_bytes = 1` / `len = 1` にしてバイト列は 1 バイトだけ確保する。
    fn simulate_hit_rate(
        cap_per_layer: usize,
        layers: usize,
        support: usize,
        top_k: usize,
        tokens: usize,
        seed: u64,
    ) -> f64 {
        let src = Mem::new(vec![0u8; 1]);
        let c = ExpertCache::new(cap_per_layer * layers, 1);
        let mut rng = seed;
        let mut seen = vec![0u64; support.div_ceil(64)];
        for _t in 0..tokens {
            for layer in 0..layers {
                seen.fill(0);
                let mut got = 0usize;
                while got < top_k {
                    let e = (splitmix64(&mut rng) % support as u64) as usize;
                    let (w, b) = (e / 64, 1u64 << (e % 64));
                    if seen[w] & b != 0 {
                        continue;
                    }
                    seen[w] |= b;
                    got += 1;
                    let _ = c
                        .get(
                            ExpertKey {
                                layer: layer as u16,
                                expert: e as u16,
                            },
                            0,
                            1,
                            &src,
                        )
                        .expect("sim get");
                }
            }
        }
        let s = c.stats();
        let n = s.hits + s.misses;
        if n == 0 {
            0.0
        } else {
            s.hits as f64 / n as f64
        }
    }

    #[test]
    fn lru_hit_rate_matches_the_p37_distribution() {
        // REPORT_P37.md §5.2 の表と同じ容量 (体/層) を回す。48 層 × 512 トークン。
        let caps = [0usize, 16, 32, 64, 96, 128, 170];
        let mut prev = -1.0f64;
        println!("  C/層    slots    hit_rate");
        for &cap in &caps {
            let r = simulate_hit_rate(cap, 48, 170, 10, 512, 0x5EED_1234_ABCD_0001);
            println!("{cap:6} {:8} {r:10.4}", cap * 48);
            assert!(r >= prev, "容量を増やしてヒット率が下がった (C={cap})");
            prev = r;
            if cap == 0 {
                assert_eq!(r, 0.0, "容量 0 でヒットした");
            }
        }
        // support 全部が入る容量なら、cold miss (48 層 × 170 体 = 8,160 回) だけが
        // ミスになる。8,160 / 245,760 = 3.3 % なので 96.7 % が期待値。
        let full = simulate_hit_rate(170, 48, 170, 10, 512, 0x5EED_1234_ABCD_0001);
        assert!(
            full > 0.95,
            "S=170 を常駐させてヒット率 {full} (cold miss 3.3% を除けばほぼ 1 のはず)"
        );
        // 半分 (85 体/層) なら一様分布の期待値 ≈ 0.5 の近く。
        let half = simulate_hit_rate(85, 48, 170, 10, 512, 0x5EED_1234_ABCD_0001);
        assert!(
            (0.3..0.75).contains(&half),
            "C=85 (S=170 の半分) のヒット率 {half} が一様分布の見込みから外れている"
        );
    }

    #[test]
    fn hit_rate_of_empty_cache_is_zero() {
        let dir = std::env::temp_dir().join("denno_expert_cache_set_empty");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("dummy.bin");
        std::fs::write(&p, [0u8; 32]).expect("temp file");
        let set = ExpertCacheSet::new_with_slots([1, 1, 1], [16, 16, 16], &[p], 1).expect("set");
        assert_eq!(set.hit_rate(ExpertMatrix::Gate), 0.0);
        assert!(set.report(ExpertMatrix::Gate).contains("hit_rate=0.000"));
        assert!(set.report(ExpertMatrix::Gate).contains("resident_mib=0"));
        assert_eq!(set.slots(ExpertMatrix::Down), 1);
        assert_eq!(set.mb(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // task 260: 固定長ストア (RAM tier)
    // -----------------------------------------------------------------

    #[test]
    fn task260_fixed_store_reuses_slots_and_holds_the_budget() {
        let data = pat(1 << 16);
        let src = Mem::new(data.clone());
        // 2 スロット (slot_bytes 1024)。3 本目で追い出しが起きる。
        let c = ExpertCache::new_fixed(2, 1024, 2 * 1024, 0, false);
        assert_eq!(c.slots(), 2);
        assert_eq!(c.slot_bytes(), 1024);
        let (k0, k1, k2) = (key(0, 0), key(0, 1), key(0, 2));
        let a = c.get(k0, 0, 1024, &src).unwrap();
        assert_eq!(&a[..], &data[0..1024], "区間の中身がファイルと違う");
        // task 907: 読者が `Arc` を持っている間は区間を予算へ戻せないので、
        // 使い終わったら落とす (読者が居る場合の挙動は
        // `task907_ram_store_reports_a_live_reader_blocking_the_budget` が見る)。
        drop(a);
        let _ = c.get(k1, 1024, 1024, &src).unwrap();
        assert_eq!(c.reserved_bytes(), 2048);
        // 3 本目: 追い出して同じ場所を使い回す (確保は増えない)
        let _ = c.get(k2, 2048, 1024, &src).unwrap();
        assert_eq!(c.reserved_bytes(), 2048, "予算を超えて塊を増やした");
        let s = c.stats();
        assert_eq!((s.hits, s.misses, s.evictions), (0, 3, 1));
        // 追い出された k0 はミス (pread が 1 回増える)
        let reads = src.reads();
        let _ = c.get(k0, 0, 1024, &src).unwrap();
        assert_eq!(src.reads(), reads + 1, "追い出した鍵がヒットした");
        assert_eq!(c.used(), 2, "常駐数がスロット数を超えた");
    }

    #[test]
    fn task260_fixed_store_zero_budget_reads_every_time() {
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new_fixed(4, 1024, 0, 0, false);
        assert_eq!(c.slots(), 0, "予算 0 でスロットができた");
        let _ = c.get(key(0, 0), 0, 1024, &src).unwrap();
        let _ = c.get(key(0, 0), 0, 1024, &src).unwrap();
        assert_eq!(src.reads(), 2, "予算 0 は毎回 pread のはず");
        assert_eq!(c.used(), 0);
    }

    #[test]
    fn task260_fixed_store_peek_does_not_read() {
        let data = pat(8192);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new_fixed(1, 512, 512, 0, false);
        let k = key(3, 4);
        assert!(c.peek(k, 512).is_none(), "空のストアで peek が当たった");
        assert_eq!(src.reads(), 0, "peek が pread した");
        let _ = c.get(k, 0, 512, &src).unwrap();
        let r = src.reads();
        let p = c.peek(k, 512).expect("ヒット");
        assert_eq!(&p[..], &data[0..512]);
        assert_eq!(src.reads(), r, "peek が pread した");
        assert!(c.peek(k, 511).is_none(), "長さ違いをヒットにした");
        assert_eq!(c.stats().hits, 1, "peek のヒットが数えられていない");
    }

    #[test]
    fn task260_fixed_store_falls_back_to_heap_without_gpu() {
        // pinned を要求しても GPU / pinned が取れなければヒープで動く
        // (失敗でリクエストを落とさない)。テスト環境には GPU が無い。
        let data = pat(4096);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new_fixed(2, 1024, 2048, 0, true);
        let r = c.get(key(0, 0), 0, 1024, &src).unwrap();
        assert_eq!(&r[..], &data[0..1024]);
        assert_eq!(r.len(), 1024, "len が要求と違う");
        let _ = c.get(key(0, 1), 1024, 1024, &src).unwrap();
        // GPU の無いテストでは pinned は 0
        if crate::model::expert_ram::mem_available().is_none() {
            assert_eq!(c.pinned_bytes(), 0);
        }
    }

    #[test]
    fn task260_new_ram_splits_the_budget_by_slot_bytes() {
        let dir = std::env::temp_dir().join("denno_expert_ram_set");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("dummy.bin");
        std::fs::write(&p, vec![7u8; 1 << 16]).expect("temp file");
        // 1 本 = 1 KiB + 1 KiB + 2 KiB = 4 KiB、8 本ぶんの総バイト
        let specs = [
            RamSpec {
                slot_bytes: 1024,
                total_bytes: 8 * 1024,
                n_expert: 8,
            },
            RamSpec {
                slot_bytes: 1024,
                total_bytes: 8 * 1024,
                n_expert: 8,
            },
            RamSpec {
                slot_bytes: 2048,
                total_bytes: 16 * 1024,
                n_expert: 8,
            },
        ];
        // 32 KiB 予算 / 1 本 4 KiB = 8 本
        let set = ExpertCacheSet::new_ram(32 << 10, specs, &[p.clone()], 0, false).expect("set");
        assert_eq!(set.slots(ExpertMatrix::Gate), 8);
        assert_eq!(set.slots(ExpertMatrix::Up), 8);
        assert_eq!(set.slots(ExpertMatrix::Down), 8);
        assert_eq!(set.reserved_bytes(), 0, "確保は初回アクセスまで遅延する");
        assert!(!set.stopped());
        // 予算が 1 本に満たなければスロット 0
        let tiny = ExpertCacheSet::new_ram(1024, specs, &[p], 0, false).expect("set");
        assert_eq!(tiny.slots(ExpertMatrix::Gate), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // task 907: RAM tier の予算会計と thrash (32-64 GiB の機械を想定した形)
    // -----------------------------------------------------------------

    /// 予算が尽きたら**ヒープへ逃げず**、LRU の古い順に 1 本ゆずる。
    ///
    /// 旧実装は `take` が `None` のときヒープの `Vec` に読んで載せていた
    /// (= 予算外の常駐が増え、pinned の意味も会計も壊れる)。予算 2 KiB /
    /// 1 本 1 KiB で 6 本引いても、常駐は 2 本・確保は 2 KiB のまま。
    #[test]
    fn task907_ram_store_evicts_instead_of_escaping_to_the_heap() {
        let data = pat(1 << 16);
        let src = Mem::new(data.clone());
        let c = ExpertCache::new_fixed(2, 1024, 2048, 0, false);
        for e in 0..6u16 {
            let off = e as u64 * 1024;
            let r = c.get(key(0, e), off, 1024, &src).expect("載るはず");
            assert_eq!(&r[..], &data[e as usize * 1024..][..1024]);
        }
        assert_eq!(c.used(), 2, "予算外の常駐が増えた (ヒープへ逃げた)");
        assert_eq!(c.reserved_bytes(), 2048, "予算を超えて確保した");
        assert_eq!(c.lost_regions(), 0, "読者が居ないのに区間を失った");
        assert!(c.stats().evictions >= 4, "追い出しが起きていない");
    }

    /// 大きい空き区間から小さい区間を切り出したら、**余りを空きへ返す**。
    ///
    /// 余りを捨てると混合量子化 (1 本のバイト数が型ごとに違う) で入れ替えの
    /// たびに予算の実バイトが痩せ、`take` が失敗して再調整が回り続ける。
    #[test]
    fn task907_ram_store_returns_the_split_tail_to_the_free_list() {
        let data = pat(1 << 16);
        let src = Mem::new(data);
        // 予算 4 KiB / スロット 3 本。2048 + 1024 + 1024 で満杯。
        let c = ExpertCache::new_fixed(3, 2048, 4096, 0, false);
        let _ = c.get(key(0, 0), 0, 2048, &src).unwrap();
        let _ = c.get(key(0, 1), 2048, 1024, &src).unwrap();
        let _ = c.get(key(0, 2), 3072, 1024, &src).unwrap();
        assert_eq!(c.reserved_bytes(), 4096);
        // 512 を要求 → 2 KiB の区間をゆずって切り出す。余り 1536 は空きへ。
        let _ = c.get(key(0, 3), 0, 512, &src).unwrap();
        assert_eq!(
            c.reserved_bytes(),
            4096,
            "余りを捨てたので塊を増やした (予算の痩せ = thrash の種)"
        );
        // 1536 は余りから取れる (塊を増やさない)。
        let r = c
            .get(key(0, 4), 4096, 1536, &src)
            .expect("余りから取れるはず");
        assert_eq!(r.len(), 1536);
        assert_eq!(c.reserved_bytes(), 4096, "余りを使わず塊を増やした");
        assert_eq!(c.used(), 3, "スロット数を超えて常駐した");
        assert_eq!(c.lost_regions(), 0);
    }

    /// 使用中の読者が居る区間は予算へ戻せない。載せられないことを隠さず、
    /// pread もしない (会計は正直に `lost` に出る)。
    #[test]
    fn task907_ram_store_reports_a_live_reader_blocking_the_budget() {
        let data = pat(1 << 16);
        let src = Mem::new(data);
        let c = ExpertCache::new_fixed(1, 1024, 1024, 0, false);
        let held = c.get(key(0, 0), 0, 1024, &src).unwrap();
        assert!(
            c.get(key(0, 1), 1024, 1024, &src).is_err(),
            "使用中で場所が無いのに載せた"
        );
        assert_eq!(c.used(), 0, "載らないはずの鍵が載った");
        assert_eq!(c.lost_regions(), 1, "戻せなかった区間を数えていない");
        assert_eq!(src.reads(), 1, "載せられないのに pread した");
        assert_eq!(held.len(), 1024);
    }

    /// 同じ鍵を違う長さで載せ直しても、古い区間を空きへ返し、LRU を壊さない。
    ///
    /// 一番古い (= tail) の鍵を載せ直すのが肝。旧実装は `map` を上書きして
    /// から `unlink` したので、リストの head/tail が NIL に潰れていた。
    #[test]
    fn task907_ram_store_replace_returns_the_old_region() {
        let data = pat(1 << 16);
        let src = Mem::new(data);
        // 予算 4 KiB / スロット 2 本。
        let c = ExpertCache::new_fixed(2, 2048, 4096, 0, false);
        let _ = c.get(key(0, 0), 0, 1024, &src).unwrap();
        let _ = c.get(key(0, 1), 1024, 1024, &src).unwrap();
        // k0 は tail。同じ鍵を 512 バイトで載せ直す。
        let r = c.get(key(0, 0), 0, 512, &src).unwrap();
        assert_eq!(r.len(), 512);
        assert_eq!(c.used(), 2, "同じ鍵を二重に数えた");
        assert_eq!(c.lost_regions(), 0, "置き換えで区間を失った");
        assert_eq!(c.reserved_bytes(), 2560, "置き換えで塊を増やした");
        assert!(c.keys().contains(&key(0, 1)), "k1 が消えた");
        // リストが壊れていなければ、次の追い出しは一番古い k1。
        // (`peek` は LRU を触るので、順序を見る前に呼ばない。)
        let _ = c.get(key(0, 2), 2048, 1024, &src).unwrap();
        assert_eq!(c.used(), 2);
        assert!(
            !c.keys().contains(&key(0, 1)),
            "k1 が追い出されていない keys={:?} stats={:?}",
            c.keys(),
            c.stats()
        );
        assert!(c.peek(key(0, 0), 512).is_some(), "載せ直した k0 が消えた");
    }
}
