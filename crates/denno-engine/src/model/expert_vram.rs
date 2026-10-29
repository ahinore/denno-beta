//! P39: expert 重みの VRAM 常駐 LRU と decode の GPU GEMV (**既定 off**)。
//!
//! # 何をするか
//!
//! decode 189 ms/token の主因は expert の CPU GEMV (480 本 × 3.072 MB =
//! 2.36 GMAC/token)。RAM 常駐キャッシュはページキャッシュを追い出して逆効果
//! だった (実測 189 -> 230/293/422/298 ms) ので、置き場所を VRAM に変える。
//!
//! expert 1 本 = gate (Q4_K [2560,640]) + up (Q4_K) + down (Q5_1 [640,2560]) の
//! 3 行列で、合計 3.072 MB。鍵を (layer, expert) にして 3 本まとめて 1 エントリ
//! として載せる (gate/up/down は同じ expert の同じトークンで必ず続けて使う)。
//!
//! # 契約
//!
//! - `DENNO_EXPERT_VRAM_MB` (既定 0) と `DENNO_EXPERT_GPU` (=1) の**両方**が
//!   立ったときだけ有効。片方でも欠ければ従来の CPU 経路で、数値も呼び出し順も
//!   1 バイト変わらない。
//! - 上限に達したら **最終使用が最も古い** expert を落としてから確保する
//!   ([`VramLru::make_room`] が先、`DevBuffer::from_slice` が後)。cuMemAlloc の
//!   失敗を通常経路にしないための順序。
//! - 既存の `cuda.rs::weights_for_gguf` (dense 重みの uid 登録簿) は**通さない**。
//!   あちらは dense 専用なので、同じ expert が 2 か所に常駐することはない。
//! - 1 本が上限より大きい / 型かカーネルが非対応 → `Ok(None)` を返し、
//!   呼び出し側は CPU へ落ちる (fail closed)。
//! - 全 evict の後でも `cuMemAlloc` が失敗したら 1 回だけ警告して以降 CPU
//!   経路に固定する (プロセス寿命のラッチ。リクエストは落とさない)。
//! - `DENNO_EXPERT_GPU_CHECK=1` のときは expert 1 本ごとに CPU で同じ計算を
//!   走らせ、最終寄与 `h` の max abs diff / argmax 一致を**層ごとに 1 行**出す。
//! - 並行 decode では [`acquire`] が返す `Arc` を計算中は保持する。追い出しで
//!   登録簿から外れても、使用中のバッファは最後の利用者が drop するまで残る
//!   (したがって上限は並行時にわずかに超え得る)。
//!
//! # P43: 予算の共有 (prefill のパック済み重みと decode の LRU が同じ上限を見る)
//!
//! P39 の実測では `DENNO_EXPERT_VRAM_MB=18000` でも 10,372 MiB で
//! `cuMemAlloc` が失敗し、CPU 固定のラッチが発火した。原因は prefill の
//! grouped GEMM が層ごとに確保するパック済み重み (1 層 ≈ 1.5 GB) が
//! **上限の外**で VRAM を食っていたこと。
//!
//! そこで LRU に「外部予約 (pinned)」を持たせ、`used + pinned <= cap` を
//! 不変条件にする:
//!
//! - [`reserve_external`] は prefill 側 (LRU 管理外) の常駐を予算から借りる。
//!   借りる前に LRU の古い expert を追い出す ([`VramLru::reserve_pinned`])。
//!   返る [`ExternalReserve`] を drop すると予算へ返る。
//! - decode の [`acquire`] は `pinned` を差し引いた残りにしか載せない
//!   (入らなければ `Ok(None)` = CPU 経路。cuMemAlloc の失敗まで進まない)。
//! - 予算が未初期化 (`DENNO_EXPERT_VRAM_MB` = 0 / 未設定) のときは
//!   [`reserve_external`] が**会計なしの予約**を返すので、既定の挙動は不変。
//! - 予算が埋まっているときに外部予約が入らない場合は `None` を返し、
//!   呼び出し側 (prefill) は CPU 経路へ落ちる (fail closed)。

use std::collections::HashMap;
use std::hash::Hash;

/// `DENNO_EXPERT_VRAM_MB`: expert の VRAM 常駐に使う上限 (MiB)。既定 0 = 無効。
pub const EXPERT_VRAM_MB_ENV: &str = "DENNO_EXPERT_VRAM_MB";
/// `DENNO_EXPERT_GPU=1`: m=1 の expert GEMV を GPU で回す。既定 off。
pub const EXPERT_GPU_ENV: &str = "DENNO_EXPERT_GPU";
/// `DENNO_EXPERT_GPU_CHECK=1`: GPU と CPU の差を層ごとに 1 行で出す。
pub const EXPERT_GPU_CHECK_ENV: &str = "DENNO_EXPERT_GPU_CHECK";

/// `"1"` / `"true"` / `"on"` だけを真として扱う (`cuda_enabled` と同じ集合)。
///
/// 前後の空白は落とす。`"0"` / 空 / 未設定 / その他の値はすべて偽。
pub fn flag_on(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1") | Some("true") | Some("on"))
}

/// [`EXPERT_VRAM_MB_ENV`] の解釈 (純粋関数)。未設定・空・`"0"`・数値でない値は 0。
///
/// 0 は「無効」で、[`resolve_route`] が必ず CPU 経路を返す。
pub fn vram_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) if !s.is_empty() && s != "0" => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// expert の計算経路。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Route {
    /// 従来の CPU (`moe_expert::expert_gemv_any`)。**既定**。
    Cpu,
    /// VRAM 常駐 + `cuda::moe_gemv_dev_buf`。
    Vram,
}

/// 経路の決定 (純粋関数)。env を読む部分から切り離してテストできるようにしてある。
///
/// VRAM 経路は「`DENNO_EXPERT_GPU` が真」かつ「`DENNO_EXPERT_VRAM_MB` > 0」かつ
/// 「CUDA が有効」の 3 つが揃ったときだけ。`VRAM_MB=0` で GPU だけ立てても
/// 毎回 3 MB を H2D するだけなので、そちらは明示的に無効にしてある。
pub fn resolve_route(gpu: Option<&str>, vram_mb: usize, gpu_available: bool) -> Route {
    if !flag_on(gpu) || vram_mb == 0 || !gpu_available {
        return Route::Cpu;
    }
    Route::Vram
}

/// CUDA が使えるか (`cuda` feature が無ければ常に false)。
pub fn gpu_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::cuda_enabled()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

// ---------------------------------------------------------------------------
// LRU (GPU 無しでテストできる中核)
// ---------------------------------------------------------------------------

/// LRU の統計。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VramStats {
    /// ヒットした回数
    pub hits: u64,
    /// 載せた回数 (ミス)
    pub misses: u64,
    /// 追い出した expert の本数
    pub evictions: u64,
    /// 1 本が上限より大きくて載せられなかった回数
    pub oversize: u64,
    /// 外部予約 (prefill のパック済み重み) が予算に収まらなかった回数
    pub pinned_oversize: u64,
    /// H2D したバイト数の累計
    pub upload_bytes: u64,
}

/// [`VramLru::make_room`] の失敗。上限より大きい 1 本は載せない (静かに切り詰めない)。
///
/// P43: `bytes` は**要求量**で、`pinned` (外部予約) と合わせて上限を超えた場合も
/// 要求量をそのまま返す。呼び出し側は「上限を上げるしかない」と読めばよい。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LruError {
    TooBig { bytes: usize, cap: usize },
}

impl std::fmt::Display for LruError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LruError::TooBig { bytes, cap } => {
                write!(f, "1 本 {bytes} B が VRAM 上限 {cap} B を超える")
            }
        }
    }
}

struct Slot<V> {
    value: V,
    bytes: usize,
    last_use: u64,
}

/// `(key, value)` の LRU。追い出しは「最終使用の通し番号が最小」の 1 本から。
///
/// 通し番号は [`VramLru::get_mut`] と [`VramLru::insert`] で単調増加するので、
/// 同じ番号の 2 本は存在せず、追い出し対象は常に一意に決まる。
pub struct VramLru<K, V> {
    cap_bytes: usize,
    used_bytes: usize,
    /// LRU 管理外の常駐 (prefill のパック済み重み) が借りているバイト数
    pinned_bytes: usize,
    clock: u64,
    entries: HashMap<K, Slot<V>>,
    stats: VramStats,
}

impl<K: Eq + Hash + Copy, V> VramLru<K, V> {
    /// 上限 `cap_bytes` で空の LRU を作る。`cap_bytes == 0` は「何も載せない」。
    pub fn new(cap_bytes: usize) -> Self {
        VramLru {
            cap_bytes,
            used_bytes: 0,
            pinned_bytes: 0,
            clock: 0,
            entries: HashMap::new(),
            stats: VramStats::default(),
        }
    }

    pub fn cap_bytes(&self) -> usize {
        self.cap_bytes
    }

    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// 外部予約 (LRU 管理外) が借りているバイト数。
    pub fn pinned_bytes(&self) -> usize {
        self.pinned_bytes
    }

    /// 予算の消費量 (`used + pinned`)。これが `cap_bytes` を超えない。
    pub fn total_bytes(&self) -> usize {
        self.used_bytes + self.pinned_bytes
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn stats(&self) -> VramStats {
        self.stats
    }

    /// ヒットしたら最終使用を更新して値を返す。ミスは数えない
    /// (ミスは [`VramLru::insert`] で数える)。
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let slot = self.entries.get_mut(key)?;
        self.clock = self.clock.saturating_add(1);
        slot.last_use = self.clock;
        self.stats.hits += 1;
        Some(&mut slot.value)
    }

    /// `bytes` を載せる余地を作る。追い出した `(key, value)` を返す
    /// (**呼び出し側がすぐ drop すれば VRAM が空く**)。
    ///
    /// `bytes + pinned > cap` は [`LruError::TooBig`]。このときは 1 本も追い出さない
    /// (外部予約は LRU からは動かせないので、追い出しても入らない)。
    pub fn make_room(&mut self, bytes: usize) -> Result<Vec<(K, V)>, LruError> {
        if bytes + self.pinned_bytes > self.cap_bytes {
            self.stats.oversize += 1;
            return Err(LruError::TooBig {
                bytes,
                cap: self.cap_bytes,
            });
        }
        let mut evicted = Vec::new();
        while self.used_bytes + self.pinned_bytes + bytes > self.cap_bytes {
            let Some(victim) = self.oldest() else {
                // entries が空なら used_bytes == 0 なので、ここには来ない
                // (`bytes + pinned <= cap` が前提)。
                break;
            };
            if let Some(v) = self.remove(&victim) {
                evicted.push((victim, v));
            }
        }
        self.stats.evictions += evicted.len() as u64;
        Ok(evicted)
    }

    /// LRU 管理外の常駐 (prefill のパック済み重み) のために `bytes` を借りる (P43)。
    ///
    /// 予算 (`cap`) に収めるために、必要なら**最終使用が古い expert から追い出す**。
    /// 追い出した `(key, value)` を返す (呼び出し側がすぐ drop すれば VRAM が空く)。
    /// 成功すると `pinned_bytes` が `bytes` 増える ([`VramLru::release_pinned`] で返す)。
    ///
    /// `bytes + pinned > cap` は [`LruError::TooBig`] (このときは 1 本も追い出さない
    /// うえで `pinned_bytes` も増やさない)。
    pub fn reserve_pinned(&mut self, bytes: usize) -> Result<Vec<(K, V)>, LruError> {
        if bytes + self.pinned_bytes > self.cap_bytes {
            self.stats.pinned_oversize += 1;
            return Err(LruError::TooBig {
                bytes,
                cap: self.cap_bytes,
            });
        }
        let mut evicted = Vec::new();
        while self.used_bytes + self.pinned_bytes + bytes > self.cap_bytes {
            let Some(victim) = self.oldest() else {
                break;
            };
            if let Some(v) = self.remove(&victim) {
                evicted.push((victim, v));
            }
        }
        self.stats.evictions += evicted.len() as u64;
        self.pinned_bytes += bytes;
        Ok(evicted)
    }

    /// 外部予約を予算へ返す。借りた量より多く返しても `pinned_bytes` は 0 で止まる
    /// (二重返却で予算が増えない)。
    pub fn release_pinned(&mut self, bytes: usize) {
        self.pinned_bytes = self.pinned_bytes.saturating_sub(bytes);
    }

    /// [`VramLru::make_room`] の後で載せる。同じ key が既にあれば置き換える
    /// (同じ expert を二重に持たない)。
    pub fn insert(&mut self, key: K, bytes: usize, value: V) {
        debug_assert!(
            self.used_bytes + self.pinned_bytes + bytes <= self.cap_bytes,
            "make_room を通さずに上限を超えて insert した"
        );
        self.remove(&key);
        self.clock = self.clock.saturating_add(1);
        self.entries.insert(
            key,
            Slot {
                value,
                bytes,
                last_use: self.clock,
            },
        );
        self.used_bytes += bytes;
        self.stats.misses += 1;
        self.stats.upload_bytes += bytes as u64;
    }

    /// 登録簿から外す (値は呼び出し側へ返す)。
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let slot = self.entries.remove(key)?;
        self.used_bytes = self.used_bytes.saturating_sub(slot.bytes);
        Some(slot.value)
    }

    /// 最終使用が最も古い key。
    fn oldest(&self) -> Option<K> {
        let mut best: Option<(K, u64)> = None;
        for (k, s) in self.entries.iter() {
            match best {
                None => best = Some((*k, s.last_use)),
                Some((_, t)) if s.last_use < t => best = Some((*k, s.last_use)),
                _ => {}
            }
        }
        best.map(|(k, _)| k)
    }

    /// LRU 順 (古い → 新しい) の key 列。テストとログ用。
    pub fn lru_keys(&self) -> Vec<K> {
        let mut v: Vec<(K, u64)> = self.entries.iter().map(|(k, s)| (*k, s.last_use)).collect();
        v.sort_by_key(|(_, t)| *t);
        v.into_iter().map(|(k, _)| k).collect()
    }
}

// ---------------------------------------------------------------------------
// P43: 予算の外部予約 (prefill のパック済み重みなど、LRU 管理外の常駐)
// ---------------------------------------------------------------------------

/// 予算から借りた「LRU 管理外の常駐」1 本。**drop すると予算へ返る**。
///
/// `is_guarded() == false` は「予算が未初期化 (会計なし)」で、返しても何も起きない
/// (既定経路の挙動を変えないための区別)。
pub struct ExternalReserve {
    bytes: usize,
    guarded: bool,
}

impl ExternalReserve {
    /// 借りたバイト数。
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 予算に計上されているか (false = 予算未初期化で素通し)。
    pub fn is_guarded(&self) -> bool {
        self.guarded
    }
}

impl Drop for ExternalReserve {
    fn drop(&mut self) {
        if self.guarded {
            release_external(self.bytes);
        }
    }
}

/// 予算から `bytes` を外部予約する (P43)。
///
/// 戻り値:
/// - `Some(r)` — 予約できた (`r.is_guarded()` が false なら会計なし = 予算未初期化)。
/// - `None` — 予算が埋まっていて入らない。呼び出し側は CPU 経路へ (fail closed)。
///
/// 予算が初期化済みなら、入れるために LRU の古い expert を追い出す。
pub fn reserve_external(bytes: usize) -> Option<ExternalReserve> {
    #[cfg(feature = "cuda")]
    {
        gpu::reserve_external(bytes)
    }
    #[cfg(not(feature = "cuda"))]
    {
        // cuda feature 無しでは VRAM 経路そのものが動かない (予算も未初期化)。
        Some(ExternalReserve {
            bytes,
            guarded: false,
        })
    }
}

/// [`ExternalReserve`] を予算へ返す (`ExternalReserve::drop` から呼ばれる)。
pub fn release_external(bytes: usize) {
    #[cfg(feature = "cuda")]
    {
        gpu::release_pinned(bytes);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = bytes;
    }
}

/// 予算の消費量 `(LRU 使用中, 外部予約, 上限)`。未初期化なら `None`。
pub fn budget_usage() -> Option<(usize, usize, usize)> {
    #[cfg(feature = "cuda")]
    {
        gpu::budget_usage()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

// ---------------------------------------------------------------------------
// GPU 側 (feature = "cuda" のみ)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub use gpu::{
    acquire, cache_usage, check_enabled, fallbacks, init, note_check, take_check, CheckAcc,
    ExpertDev, ExpertShape,
};

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};

    use anyhow::{bail, Result};

    use crate::cuda::DevBuffer;
    use crate::kquant::GgmlType;
    use crate::model::expert_cache::{ExpertKey, ExpertMatrix};

    /// 1 行列ぶんの形。載っているバッファの再利用時に突き合わせる
    /// (同じ鍵に別の形が来たらエラー。黙って別の重みで計算しない)。
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ExpertShape {
        pub ty: GgmlType,
        pub k: usize,
        pub n: usize,
    }

    impl ExpertShape {
        /// 1 expert ぶんのバイト数。
        pub fn bytes(&self) -> Result<usize> {
            let rb = crate::model::moe_expert::row_bytes(self.ty, self.k)?;
            rb.checked_mul(self.n).ok_or_else(|| {
                anyhow::anyhow!(
                    "expert のバイト数が usize を溢れる (k={}, n={}, ty={:?})",
                    self.k,
                    self.n,
                    self.ty
                )
            })
        }
    }

    /// expert 1 本ぶん (gate / up / down の 3 行列) のデバイスバッファ。
    pub struct ExpertDev {
        bufs: [DevBuffer; 3],
        shapes: [ExpertShape; 3],
    }

    impl ExpertDev {
        pub fn bytes(&self) -> usize {
            self.bufs.iter().map(|b| b.bytes()).sum()
        }

        pub fn shapes(&self) -> [ExpertShape; 3] {
            self.shapes
        }

        /// 行列 `m` の重みのデバイスアドレス (P43: 層一括 GEMV のポインタ表に載せる)。
        ///
        /// 1 本の `DevBuffer` の先頭そのもの。呼び出し側は**計算が終わるまで
        /// `Arc<ExpertDev>` を保持する**こと (保持している間は追い出されても実体が残る)。
        pub fn ptr(&self, m: ExpertMatrix) -> u64 {
            self.bufs[m.index()].ptr()
        }

        /// 行列 `m` の GEMV。`x` は行列の入力長、`out` は出力長でなければならない
        /// (検査は `cuda::moe_gemv_dev_buf` が行う)。
        ///
        /// 載せる前に [`crate::cuda::CudaCtx::moe_gemv_dev_ready`] を通しているので、
        /// ここで `false` が返るのは「載せた後にコンテキストが変わった」場合だけ。
        /// 途中で CPU に混ざると 1 expert の計算が 2 経路に割れるので、**エラー**にする。
        pub fn gemv(&self, m: ExpertMatrix, x: &[f32], out: &mut [f32]) -> Result<()> {
            let i = m.index();
            let s = self.shapes[i];
            if !crate::cuda::CudaCtx::moe_gemv_dev_buf(s.ty, &self.bufs[i], x, s.k, s.n, out)? {
                bail!(
                    "expert GEMV: moe_gemv_dev が false を返した (matrix={}, ty={:?}, k={}, n={})",
                    m.name(),
                    s.ty,
                    s.k,
                    s.n
                );
            }
            Ok(())
        }
    }

    static CACHE: OnceLock<Mutex<VramLru<ExpertKey, Arc<ExpertDev>>>> = OnceLock::new();
    /// 全 evict 後も確保できなかった回数 (診断用)
    static FALLBACKS: AtomicU64 = AtomicU64::new(0);
    /// 1 本が上限より大きいことを 1 回だけ言うためのラッチ
    static OVERSIZE_LOGGED: AtomicBool = AtomicBool::new(false);
    /// 確保失敗で VRAM 経路を諦めた (プロセス寿命)
    static ALLOC_DEAD: AtomicBool = AtomicBool::new(false);

    /// 上限 `cap_bytes` で LRU を作る (プロセスで 1 回)。2 回目以降は無視する。
    pub fn init(cap_bytes: usize) {
        let _ = CACHE.set(Mutex::new(VramLru::new(cap_bytes)));
    }

    /// (使用中バイト, 上限バイト)。未初期化なら `None`。
    pub fn cache_usage() -> Option<(usize, usize)> {
        let c = CACHE.get()?;
        let g = c.lock().ok()?;
        Some((g.used_bytes(), g.cap_bytes()))
    }

    /// (LRU 使用中バイト, 外部予約バイト, 上限バイト)。未初期化なら `None`。
    pub fn budget_usage() -> Option<(usize, usize, usize)> {
        let c = CACHE.get()?;
        let g = c.lock().ok()?;
        Some((g.used_bytes(), g.pinned_bytes(), g.cap_bytes()))
    }

    /// P43: LRU 管理外の常駐 (prefill のパック済み重み) を予算から借りる。
    ///
    /// - 予算が未初期化 (`DENNO_EXPERT_VRAM_MB` = 0) → 会計なしの予約
    ///   (`is_guarded() == false`)。既定の挙動を変えない。
    /// - 予算が埋まっていて LRU を全部追い出しても入らない → `None` (CPU へ)。
    /// - `ALLOC_DEAD` (確保失敗で CPU 固定) の間は `None`。
    pub fn reserve_external(bytes: usize) -> Option<ExternalReserve> {
        let Some(cache) = CACHE.get() else {
            return Some(ExternalReserve {
                bytes,
                guarded: false,
            });
        };
        if ALLOC_DEAD.load(Ordering::Relaxed) {
            return None;
        }
        let mut g = match cache.lock() {
            Ok(g) => g,
            // ロックが壊れている = 実装の誤り。会計を諦めて素通しする
            // (ここで落とすと prefill が丸ごと CPU になる)。
            Err(_) => {
                return Some(ExternalReserve {
                    bytes,
                    guarded: false,
                })
            }
        };
        let evicted = g.reserve_pinned(bytes).ok()?;
        drop(evicted); // ここで DevBuffer が落ちて VRAM が空く
        Some(ExternalReserve {
            bytes,
            guarded: true,
        })
    }

    /// P43: 外部予約を予算へ返す (`ExternalReserve::drop` から)。
    pub fn release_pinned(bytes: usize) {
        let Some(cache) = CACHE.get() else {
            return;
        };
        if let Ok(mut g) = cache.lock() {
            g.release_pinned(bytes);
        }
    }

    /// 上限超過・未対応・確保失敗で CPU へ落ちた回数 (診断用)。
    pub fn fallbacks() -> u64 {
        FALLBACKS.load(Ordering::Relaxed)
    }

    /// expert 1 本を VRAM から取る。無ければ `load` でバイト列を取って載せる。
    ///
    /// 戻り値:
    /// - `Ok(Some(dev))` — GPU で回すこと。`dev` は計算中ずっと保持してよい
    ///   (追い出されても drop まで生きている)。
    /// - `Ok(None)` — この expert は GPU で回せない。呼び出し側は CPU へ。
    /// - `Err` — 同じ鍵に別の形が来た / ロックが壊れた (実装の誤り)。
    ///
    /// `load` は**ミスのときだけ**呼ばれる (ヒットで mmap を触らない)。
    pub fn acquire<'a, F>(
        key: ExpertKey,
        shapes: [ExpertShape; 3],
        load: F,
    ) -> Result<Option<Arc<ExpertDev>>>
    where
        F: FnOnce() -> Result<[&'a [u8]; 3]>,
    {
        if ALLOC_DEAD.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let Some(cache) = CACHE.get() else {
            return Ok(None);
        };
        let mut g = cache
            .lock()
            .map_err(|_| anyhow::anyhow!("expert VRAM LRU のロックが poisoned"))?;

        if let Some(dev) = g.get_mut(&key) {
            if dev.shapes != shapes {
                bail!(
                    "expert VRAM: 同じ鍵 {:?} に別の形が来た (登録済み {:?} / 今回 {:?})",
                    key,
                    dev.shapes,
                    shapes
                );
            }
            return Ok(Some(Arc::clone(dev)));
        }

        // 載せる前に「型・カーネルが対応しているか」を確かめる。載せてから
        // `false` で落ちると VRAM が無駄に埋まる。
        let mut want = 0usize;
        for s in shapes {
            if !crate::cuda::CudaCtx::moe_gemv_dev_ready(s.ty, s.k) {
                FALLBACKS.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            want = want.saturating_add(s.bytes()?);
        }

        // 追い出してから確保する (cuMemAlloc の失敗を通常経路にしない)。
        let evicted = match g.make_room(want) {
            Ok(v) => v,
            Err(e) => {
                if !OVERSIZE_LOGGED.swap(true, Ordering::Relaxed) {
                    tracing::warn!(
                        "expert VRAM: {e} → この expert 以降も CPU 経路 (DENNO_EXPERT_VRAM_MB を上げる)"
                    );
                }
                FALLBACKS.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        };
        drop(evicted); // ここで DevBuffer が落ちて VRAM が空く

        let [gb, ub, db] = load()?;
        let alloc = |b: &[u8]| match DevBuffer::from_slice(b) {
            Ok(d) => Some(d),
            Err(e) => {
                if !ALLOC_DEAD.swap(true, Ordering::Relaxed) {
                    tracing::error!(
                        "expert VRAM: 確保に失敗したので以後 CPU 経路に固定する: {e:#}"
                    );
                }
                None
            }
        };
        // 3 本は左から順に評価される。失敗した時点で先に確保したぶんは
        // この関数を抜けるときに drop されて VRAM が戻る。
        let (Some(gate), Some(up), Some(down)) = (alloc(gb), alloc(ub), alloc(db)) else {
            FALLBACKS.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        };

        let dev = Arc::new(ExpertDev {
            bufs: [gate, up, down],
            shapes,
        });
        g.insert(key, want, Arc::clone(&dev));
        Ok(Some(dev))
    }

    // -----------------------------------------------------------------------
    // DENNO_EXPERT_GPU_CHECK: CPU との差を層ごとに 1 行で出す
    // -----------------------------------------------------------------------

    /// 1 層ぶんの比較結果。
    #[derive(Clone, Copy, Debug, Default)]
    pub struct CheckAcc {
        /// 比較した expert 数
        pub n: u64,
        /// argmax が一致した数
        pub argmax: u64,
        /// 最大絶対差
        pub max_abs: f32,
        /// 最大相対差 (`|gpu-cpu| / max(|cpu|, 1e-6)`)
        pub max_rel: f32,
    }

    static CHECK: Mutex<Option<CheckAcc>> = Mutex::new(None);

    /// `DENNO_EXPERT_GPU_CHECK` が真か (プロセスで 1 回だけ読む)。
    pub fn check_enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| flag_on(std::env::var(EXPERT_GPU_CHECK_ENV).ok().as_deref()))
    }

    /// GPU の結果と CPU の結果を 1 本ぶん突き合わせる。
    pub fn note_check(gpu: &[f32], cpu: &[f32]) {
        let mut g = match CHECK.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let acc = g.get_or_insert_with(CheckAcc::default);
        acc.n += 1;
        if argmax(gpu) == argmax(cpu) {
            acc.argmax += 1;
        }
        for (a, b) in gpu.iter().zip(cpu.iter()) {
            let d = (a - b).abs();
            if d > acc.max_abs {
                acc.max_abs = d;
            }
            let r = d / b.abs().max(1e-6);
            if r > acc.max_rel {
                acc.max_rel = r;
            }
        }
    }

    /// 溜まった比較結果を取り出して 0 に戻す (比較が 1 件も無ければ `None`)。
    pub fn take_check() -> Option<CheckAcc> {
        let mut g = CHECK.lock().ok()?;
        g.take()
    }

    fn argmax(v: &[f32]) -> usize {
        let mut best = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 追い出しは「最終使用が最も古い」1 本から。
    #[test]
    fn lru_evicts_least_recently_used() {
        let mut c: VramLru<u8, String> = VramLru::new(300);
        c.make_room(100).unwrap();
        c.insert(1, 100, "a".into());
        c.make_room(100).unwrap();
        c.insert(2, 100, "b".into());
        c.make_room(100).unwrap();
        c.insert(3, 100, "c".into());
        assert_eq!(c.len(), 3);
        assert_eq!(c.used_bytes(), 300);

        // 1 を使う → 次に追い出されるのは 2
        assert_eq!(c.get_mut(&1).map(|s| s.as_str()), Some("a"));
        let ev = c.make_room(100).unwrap();
        assert_eq!(ev.len(), 1, "1 本だけ追い出す");
        assert_eq!(ev[0].0, 2, "最終使用が最も古いのは 2");
        assert_eq!(c.lru_keys(), vec![3, 1]);
        c.insert(4, 100, "d".into());
        assert_eq!(c.used_bytes(), 300, "上限を超えない");
        assert_eq!(c.stats().evictions, 1);
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 4, "insert した回数 (1,2,3,4)");
    }

    /// どの並びで入れても `used_bytes` は上限を超えない。
    #[test]
    fn lru_never_exceeds_cap() {
        let cap = 1000usize;
        let mut c: VramLru<u16, usize> = VramLru::new(cap);
        for i in 0..200u16 {
            let bytes = 17 + (i as usize * 37) % 211;
            let evicted = c.make_room(bytes).unwrap();
            drop(evicted);
            c.insert(i, bytes, i as usize);
            assert!(
                c.used_bytes() <= cap,
                "i={i} で上限超過 ({} > {cap})",
                c.used_bytes()
            );
        }
        assert!(c.stats().evictions > 0, "追い出しが 1 度も起きていない");
        assert_eq!(
            c.used_bytes(),
            (0..200u16)
                .filter(|i| c.entries.contains_key(i))
                .map(|i| c.entries[&i].bytes)
                .sum::<usize>(),
            "used_bytes が実体の合計と合わない"
        );
    }

    /// 上限より大きい 1 本は載せない。このとき既存のエントリを巻き込まない。
    #[test]
    fn lru_rejects_oversize_without_evicting() {
        let mut c: VramLru<u8, &'static str> = VramLru::new(100);
        c.make_room(100).unwrap();
        c.insert(1, 100, "a");
        let err = c.make_room(101).unwrap_err();
        assert_eq!(err, LruError::TooBig { bytes: 101, cap: 100 });
        assert_eq!(c.len(), 1, "追い出してはいけない");
        assert_eq!(c.used_bytes(), 100);
        assert_eq!(c.stats().oversize, 1);
        assert_eq!(c.stats().evictions, 0);
    }

    /// 同じ key を載せ直しても二重に数えない。
    #[test]
    fn lru_replace_does_not_double_count() {
        let mut c: VramLru<u8, &'static str> = VramLru::new(100);
        c.make_room(40).unwrap();
        c.insert(7, 40, "a");
        c.make_room(60).unwrap();
        c.insert(7, 60, "b");
        assert_eq!(c.len(), 1);
        assert_eq!(c.used_bytes(), 60);
        assert_eq!(c.get_mut(&7).map(|s| *s), Some("b"));
        assert_eq!(c.stats().misses, 2);
    }

    /// 上限 0 は「何も載せない」。`insert` は呼ばれない前提だが、呼ばれても
    /// 上限を超えない (make_room が必ず TooBig)。
    #[test]
    fn cap_zero_holds_nothing() {
        let mut c: VramLru<u8, u32> = VramLru::new(0);
        assert!(matches!(
            c.make_room(1),
            Err(LruError::TooBig { bytes: 1, cap: 0 })
        ));
        assert_eq!(c.len(), 0);
        assert_eq!(c.used_bytes(), 0);
    }

    /// 経路の決定表: GPU フラグ・上限・CUDA 可用性の 3 条件がすべて要る。
    #[test]
    fn route_needs_flag_cap_and_cuda() {
        assert_eq!(resolve_route(None, 4096, true), Route::Cpu, "未設定は従来どおり");
        assert_eq!(resolve_route(Some("0"), 4096, true), Route::Cpu);
        assert_eq!(resolve_route(Some(""), 4096, true), Route::Cpu);
        assert_eq!(resolve_route(Some("off"), 4096, true), Route::Cpu);
        assert_eq!(
            resolve_route(Some("1"), 0, true),
            Route::Cpu,
            "上限 0 は完全に従来経路"
        );
        assert_eq!(
            resolve_route(Some("1"), 4096, false),
            Route::Cpu,
            "CUDA が無効なら GPU 経路に入らない"
        );
        assert_eq!(resolve_route(Some("1"), 4096, true), Route::Vram);
        assert_eq!(resolve_route(Some("true"), 1, true), Route::Vram);
        assert_eq!(resolve_route(Some("on"), 1, true), Route::Vram);
        assert_eq!(resolve_route(Some(" 1 "), 1, true), Route::Vram);
        assert_eq!(resolve_route(Some("2"), 1, true), Route::Cpu);
    }

    /// `DENNO_EXPERT_VRAM_MB` の解釈。
    #[test]
    fn vram_mb_parsing() {
        assert_eq!(vram_mb_from_value(None), 0);
        assert_eq!(vram_mb_from_value(Some("")), 0);
        assert_eq!(vram_mb_from_value(Some("  ")), 0);
        assert_eq!(vram_mb_from_value(Some("0")), 0);
        assert_eq!(vram_mb_from_value(Some("4096")), 4096);
        assert_eq!(vram_mb_from_value(Some(" 8192 ")), 8192);
        assert_eq!(vram_mb_from_value(Some("12x")), 0, "数値でない値は無効");
        assert_eq!(vram_mb_from_value(Some("-1")), 0, "負は無効");
    }

    /// フラグの解釈 (`cuda_enabled` と同じ集合)。
    #[test]
    fn flag_parsing() {
        assert!(flag_on(Some("1")));
        assert!(flag_on(Some("true")));
        assert!(flag_on(Some("on")));
        assert!(flag_on(Some(" on ")));
        assert!(!flag_on(None));
        assert!(!flag_on(Some("")));
        assert!(!flag_on(Some("0")));
        assert!(!flag_on(Some("TRUE")));
    }

    // -----------------------------------------------------------------------
    // P43: 予算の共有 (外部予約 = prefill のパック済み重み)
    // -----------------------------------------------------------------------

    /// 外部予約は「LRU の古い expert を追い出して」予算に収まる。
    #[test]
    fn pinned_reservation_evicts_lru_first() {
        let mut c: VramLru<u8, &'static str> = VramLru::new(1000);
        for i in 0..4u8 {
            c.make_room(250).unwrap();
            c.insert(i, 250, "e");
        }
        assert_eq!(c.used_bytes(), 1000);
        assert_eq!(c.pinned_bytes(), 0);

        // 予算は 1000 なので 400 を借りるには 400 ぶん追い出す (2 本)。
        let ev = c.reserve_pinned(400).unwrap();
        assert_eq!(ev.len(), 2, "最終使用が古い 0 と 1 を追い出す");
        assert_eq!(c.pinned_bytes(), 400);
        assert_eq!(c.used_bytes(), 500);
        assert_eq!(c.total_bytes(), 900, "上限は超えない");
        assert_eq!(c.lru_keys(), vec![2, 3]);
        assert_eq!(c.stats().evictions, 2);
    }

    /// 外部予約が載っている間は、残りにしか expert を載せない (入らなければ追い出す)。
    #[test]
    fn pinned_limits_expert_loads() {
        let mut c: VramLru<u8, &'static str> = VramLru::new(1000);
        c.reserve_pinned(800).unwrap();
        assert!(c.make_room(200).is_ok(), "残り 200 は載る");
        c.insert(1, 200, "a");
        // 1000 - 800 = 200 しか空きが無いので、1 バイト載せるには expert 1 を
        // 追い出す (外部予約は LRU から追い出せない)。
        let ev = c.make_room(1).unwrap();
        assert_eq!(ev.len(), 1, "残りに収めるために expert を追い出す");
        assert_eq!(ev[0].0, 1);
        assert_eq!(c.used_bytes(), 0);
        // 残り (200) より大きい要求は、追い出しても入らないので届かない。
        c.make_room(200).unwrap();
        c.insert(2, 200, "b");
        let err = c.make_room(201).unwrap_err();
        assert_eq!(err, LruError::TooBig { bytes: 201, cap: 1000 });
        assert_eq!(c.len(), 1, "追い出しても入らないので何もしない");
        assert_eq!(c.pinned_bytes(), 800);
        assert_eq!(c.stats().oversize, 1);
    }

    /// 予約を返すと、その分だけ expert が載るようになる。
    #[test]
    fn release_pinned_returns_room() {
        let mut c: VramLru<u8, &'static str> = VramLru::new(1000);
        c.reserve_pinned(900).unwrap();
        assert_eq!(c.make_room(200).unwrap_err(), LruError::TooBig { bytes: 200, cap: 1000 });
        c.release_pinned(900);
        assert_eq!(c.pinned_bytes(), 0);
        c.make_room(900).unwrap();
        c.insert(1, 900, "a");
        assert_eq!(c.total_bytes(), 900);
        // 借りた量より多く返しても 0 で止まる。
        c.release_pinned(12345);
        assert_eq!(c.pinned_bytes(), 0);
    }

    /// 上限より大きい外部予約は断る。このとき既存のエントリを巻き込まない。
    #[test]
    fn pinned_rejects_oversize_without_evicting() {
        let mut c: VramLru<u8, &'static str> = VramLru::new(100);
        c.make_room(60).unwrap();
        c.insert(1, 60, "a");
        let err = c.reserve_pinned(101).unwrap_err();
        assert_eq!(err, LruError::TooBig { bytes: 101, cap: 100 });
        assert_eq!(c.len(), 1, "追い出してはいけない");
        assert_eq!(c.pinned_bytes(), 0, "予約も増やさない");
        assert_eq!(c.stats().pinned_oversize, 1);
        assert_eq!(c.stats().evictions, 0);
    }

    /// 予算が未初期化 (cuda feature 無し / `DENNO_EXPERT_VRAM_MB` = 0) のときの
    /// [`reserve_external`] は会計なしで素通しする = 既定の挙動が変わらない。
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn external_reserve_without_budget_is_unguarded() {
        let r = reserve_external(3_000_000).expect("予算が無いときは必ず借りられる");
        assert!(!r.is_guarded(), "会計なし (予算未初期化)");
        assert_eq!(r.bytes(), 3_000_000);
        assert_eq!(budget_usage(), None, "予算そのものが無い");
        drop(r); // 何も起きない (パニックしないこと)
    }
}
