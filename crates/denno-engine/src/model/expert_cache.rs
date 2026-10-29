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

/// 1 スロットぶんの常駐。
struct Slot {
    key: ExpertKey,
    len: usize,
    buf: Arc<Vec<u8>>,
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
    /// 追い出した `Vec` の再利用プール (毎ミスで 1 MB を zero 初期化しない)。
    pool: Vec<Vec<u8>>,
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
            pool: Vec::new(),
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
    fn insert(&mut self, key: ExpertKey, len: usize, buf: Arc<Vec<u8>>) {
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
                // 読者が居なければバッファを再利用する (ゼロ初期化を避ける)。
                if let Ok(mut b) = Arc::try_unwrap(old.buf) {
                    b.clear();
                    if self.pool.len() < self.slots.len() {
                        self.pool.push(b);
                    }
                }
                v
            }
        };
        // 同じ鍵が別スロットに居る (len 違い) なら、そちらを捨てる。
        if let Some(prev) = self.map.insert(key, sid) {
            if prev != sid {
                self.unlink(prev);
                self.slots[prev as usize] = None;
                self.free.push(prev);
            }
        }
        self.slots[sid as usize] = Some(Slot { key, len, buf });
        self.push_front(sid);
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
    ) -> Result<Arc<Vec<u8>>> {
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
            return Ok(Arc::new(buf));
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

        // ミス: プールから作業域を 1 本取り、ロックの外で読む。
        let mut buf = {
            let mut g = lock_inner(&self.inner);
            g.pool.pop().unwrap_or_default()
        };
        buf.resize(len, 0);
        let n = match read_into(src, off, &mut buf) {
            Ok(n) => n,
            Err(e) => {
                // 失敗したバッファは返さない (半端な内容をプールに残さない)。
                return Err(e);
            }
        };
        let arc = Arc::new(buf);
        {
            let mut g = lock_inner(&self.inner);
            g.stats.bytes_read += n as u64;
            g.insert(key, len, arc.clone());
        }
        Ok(arc)
    }

    pub fn stats(&self) -> CacheStats {
        lock_inner(&self.inner).stats
    }

    /// 統計だけを 0 に戻す (スロットの中身と LRU の順序はそのまま)。
    pub fn reset_stats(&mut self) {
        lock_inner(&self.inner).stats = CacheStats::default();
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

    /// 行列ごとの常駐上限 (バイト)。`DENNO_EXPERT_CACHE_STATS` の 1 行に出す。
    pub fn resident_bytes(&self, m: ExpertMatrix) -> u64 {
        self.slots(m) as u64 * self.slot_bytes(m) as u64
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
    ) -> Result<Arc<Vec<u8>>> {
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
}
