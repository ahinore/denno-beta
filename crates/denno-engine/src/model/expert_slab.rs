//! task 152: expert の VRAM を「大きなチャンク (slab) + 固定スロット」で確保する。
//!
//! # なぜ要るか
//!
//! lab t149c の `[mem-audit]` は、expert を 20〜22 GB 載せたときに
//! **residual (context + JIT + 断片) が約 3,000 MiB** あることを示した。うち
//! 約 2,200 MiB は expert 1 本ごとの 2 MB 級 `cuMemAlloc`/`cuMemFree` の積み上げ
//! (34,000 本) である (推定。REPORT_P149.md)。
//!
//! ここでは expert の 3 行列を**大きな 1 本の確保から切り出す**。一度スロットを
//! 確保したら、以降の入れ替え (eviction) は**デバイスメモリを解放せず**、
//! スロットを空きリストへ返すだけにする。新規の `cuMemAlloc` は
//! 「チャンクが足りないとき」だけ (256 MiB 単位) になる。
//!
//! # 契約
//!
//! - ゲートは `DENNO_EXPERT_SLAB` (既定は lab の採否で決める)。off のとき
//!   呼び出し側は従来の per-expert 確保を使う (確保の順序もバイト列も同じ)。
//! - **数値は変わらない**。チャンクはただの連続したデバイスメモリで、expert の
//!   重みは同じバイト列が同じオフセットに入る。カーネルへ渡すポインタの
//!   相対関係 (gate / up / down) も同じ。
//! - スロットのサイズは「3 行列の合計を 256 B に切り上げたもの」。同じ形の
//!   expert は同じチャンクを共有する (形が違えば別のチャンク列)。
//! - `SlabSlot` を drop するとスロットが空きリストへ戻る。チャンク自体は
//!   プロセスの間は解放しない (断片化を戻さないため)。確保済みバイトは
//!   台帳 (`mem_audit::TAG_EXPERTS`) に載るので、カード予算の計算には入る。
//!
//! # テスト
//!
//! コア ([`Slab`]) はチャンクの「ハンドル」をジェネリックに持つので、GPU 無しで
//! スロットの再利用・チャンクの追加・形の分離を検証できる
//! (`expert_slab::tests`)。

use std::sync::{Arc, Mutex, OnceLock};

/// `DENNO_EXPERT_SLAB`: slab 経路のゲート (`1` / `true` / `on` / `yes` で on)。
pub const SLAB_ENV: &str = "DENNO_EXPERT_SLAB";
/// `DENNO_EXPERT_SLAB_MB`: 1 チャンクの大きさ (MiB)。既定 [`DEFAULT_CHUNK_MB`]。
pub const SLAB_CHUNK_MB_ENV: &str = "DENNO_EXPERT_SLAB_MB";
/// 1 チャンクの既定の大きさ (MiB)。expert 1 本は約 2 MB なので 128 本ぶん。
pub const DEFAULT_CHUNK_MB: usize = 256;
/// スロットの開始オフセットを合わせる単位 (バイト)。
pub const SLOT_ALIGN: usize = 256;

/// [`SLAB_ENV`] の解釈 (純関数)。明示的な真値だけ on。
pub fn slab_from_value(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}

/// task 152: slab 経路を使うか (`DENNO_EXPERT_SLAB`)。
///
/// 既定は lab の採否で決める ([`SLAB_DEFAULT`])。
pub fn slab_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| slab_from_value(std::env::var(SLAB_ENV).ok().as_deref(), SLAB_DEFAULT))
}

/// 既定値。lab t152a/t152b の結果で決める (`REPORT_P152.md`)。
pub const SLAB_DEFAULT: bool = false;

/// [`SLAB_CHUNK_MB_ENV`] を MiB で読む (0・不正値は既定)。
pub fn chunk_mb_from_value(v: Option<&str>) -> usize {
    match v.map(str::trim) {
        Some(s) => s.parse::<usize>().ok().filter(|n| *n > 0).unwrap_or(DEFAULT_CHUNK_MB),
        None => DEFAULT_CHUNK_MB,
    }
}

/// 1 チャンクの大きさ (バイト)。
pub fn chunk_bytes() -> usize {
    static B: OnceLock<usize> = OnceLock::new();
    *B.get_or_init(|| {
        chunk_mb_from_value(std::env::var(SLAB_CHUNK_MB_ENV).ok().as_deref()) << 20
    })
}

/// `n` を [`SLOT_ALIGN`] の倍数へ切り上げる (0 は 0 のまま)。
pub fn align_up(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        n.div_ceil(SLOT_ALIGN) * SLOT_ALIGN
    }
}

/// 3 行列のバイト数からスロットのレイアウトを決める (純関数)。
///
/// 戻り値は `(gate のオフセット, up のオフセット, down のオフセット, スロット全体)`。
/// 各行列の先頭を [`SLOT_ALIGN`] に合わせる (アラインメントを保つ)。
pub fn slot_layout(bytes: [usize; 3]) -> ([usize; 3], usize) {
    let g = 0;
    let u = align_up(bytes[0]);
    let d = u + align_up(bytes[1]);
    let total = d + align_up(bytes[2]);
    ([g, u, d], total)
}

/// 1 チャンク。`handle` はデバイスバッファ (テストでは番号)。
struct Chunk<H> {
    handle: H,
    slot_bytes: usize,
    n_slots: usize,
    /// 空きスロットの番号 (LIFO)。`SlabSlot::drop` が返す。
    free: Mutex<Vec<usize>>,
}

impl<H> Chunk<H> {
    fn used(&self) -> usize {
        self.n_slots - self.free.lock().map(|f| f.len()).unwrap_or(self.n_slots)
    }
}

/// 1 スロット。`chunk` を `Arc` で保持するので、計算中の `Arc<ExpertDev>` が
/// 残っている間はスロットが再利用されない (別の重みで上書きされない)。
pub struct SlabSlot<H> {
    chunk: Arc<Chunk<H>>,
    idx: usize,
}

impl<H> SlabSlot<H> {
    /// チャンクのハンドル (デバイスバッファ)。
    pub fn handle(&self) -> &H {
        &self.chunk.handle
    }

    /// チャンク内のスロット番号。
    pub fn idx(&self) -> usize {
        self.idx
    }

    /// チャンク先頭からのバイトオフセット。
    pub fn offset(&self) -> usize {
        self.idx * self.chunk.slot_bytes
    }

    /// このスロットの大きさ (バイト)。
    pub fn slot_bytes(&self) -> usize {
        self.chunk.slot_bytes
    }

    /// このチャンクのスロット数。
    pub fn n_slots(&self) -> usize {
        self.chunk.n_slots
    }
}

impl<H> Drop for SlabSlot<H> {
    fn drop(&mut self) {
        if let Ok(mut f) = self.chunk.free.lock() {
            f.push(self.idx);
        }
    }
}

/// チャンクの集まり。`slot_bytes` が同じものだけを共有する。
pub struct Slab<H> {
    chunks: Mutex<Vec<Arc<Chunk<H>>>>,
}

impl<H> Default for Slab<H> {
    fn default() -> Self {
        Self::new()
    }
}

impl<H> Slab<H> {
    pub fn new() -> Self {
        Slab {
            chunks: Mutex::new(Vec::new()),
        }
    }

    /// `slot_bytes` の空きスロットを 1 つ返す。無ければ `make` でチャンクを作る。
    ///
    /// `make` には**チャンクのバイト数**を渡す (`n_slots * slot_bytes`)。確保に
    /// 失敗したらそのままエラーを返す (呼び出し側は従来の per-expert 確保へ落ちる)。
    pub fn alloc(
        &self,
        slot_bytes: usize,
        make: impl FnOnce(usize) -> anyhow::Result<H>,
    ) -> anyhow::Result<SlabSlot<H>> {
        if slot_bytes == 0 {
            anyhow::bail!("slab: slot_bytes = 0");
        }
        // 1. 既存チャンクの空きを探す。
        {
            let chunks = self
                .chunks
                .lock()
                .map_err(|_| anyhow::anyhow!("slab: チャンク表のロックが poisoned"))?;
            for c in chunks.iter() {
                if c.slot_bytes != slot_bytes {
                    continue;
                }
                let mut free = c
                    .free
                    .lock()
                    .map_err(|_| anyhow::anyhow!("slab: 空きリストのロックが poisoned"))?;
                if let Some(idx) = free.pop() {
                    return Ok(SlabSlot {
                        chunk: Arc::clone(c),
                        idx,
                    });
                }
            }
        }
        // 2. 新しいチャンクを作る。ロックを持ったまま cuMemAlloc しない
        //    (確保失敗→追い出しの経路をブロックしないため)。
        let n_slots = (chunk_bytes() / slot_bytes).max(1);
        let bytes = n_slots * slot_bytes;
        let handle = make(bytes)?;
        let chunk = Arc::new(Chunk {
            handle,
            slot_bytes,
            n_slots,
            free: Mutex::new((0..n_slots).collect()),
        });
        let idx = chunk
            .free
            .lock()
            .map_err(|_| anyhow::anyhow!("slab: 空きリストのロックが poisoned"))?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("slab: 新しいチャンクに空きが無い"))?;
        {
            let mut chunks = self
                .chunks
                .lock()
                .map_err(|_| anyhow::anyhow!("slab: チャンク表のロックが poisoned"))?;
            chunks.push(Arc::clone(&chunk));
        }
        Ok(SlabSlot { chunk, idx })
    }

    /// チャンク数 (診断用)。
    pub fn chunk_count(&self) -> usize {
        self.chunks.lock().map(|c| c.len()).unwrap_or(0)
    }

    /// 確保済みバイト数 (`slot_bytes * n_slots` の合計)。
    pub fn allocated_bytes(&self) -> usize {
        match self.chunks.lock() {
            Ok(c) => c
                .iter()
                .map(|c| c.slot_bytes.saturating_mul(c.n_slots))
                .sum(),
            Err(_) => 0,
        }
    }

    /// `(使用中スロット, 全スロット)` (診断用)。
    pub fn slots(&self) -> (usize, usize) {
        match self.chunks.lock() {
            Ok(c) => c.iter().fold((0, 0), |(u, t), c| (u + c.used(), t + c.n_slots)),
            Err(_) => (0, 0),
        }
    }
}

// ---------------------------------------------------------------------------
// GPU 側の入口 (cuda feature のときだけ)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
mod gpu {
    use super::*;
    use crate::cuda::DevBuffer;

    fn slab() -> &'static Slab<DevBuffer> {
        static S: OnceLock<Slab<DevBuffer>> = OnceLock::new();
        S.get_or_init(Slab::new)
    }

    /// `slot_bytes` のスロットを 1 つ確保する (チャンクは `TAG_EXPERTS` で計上)。
    pub fn alloc_slot(slot_bytes: usize) -> anyhow::Result<SlabSlot<DevBuffer>> {
        slab().alloc(slot_bytes, DevBuffer::slab_chunk)
    }

    /// `(使用中スロット, 全スロット, 確保済みバイト)` (診断用)。
    pub fn stats() -> (usize, usize, usize) {
        let s = slab();
        let (u, t) = s.slots();
        (u, t, s.allocated_bytes())
    }
}

/// `(使用中スロット, 全スロット, 確保済みバイト)` (診断用。cuda 無しは 0)。
pub fn stats() -> (usize, usize, usize) {
    #[cfg(feature = "cuda")]
    {
        gpu::stats()
    }
    #[cfg(not(feature = "cuda"))]
    {
        (0, 0, 0)
    }
}

/// `slot_bytes` のスロットを確保する (cuda feature のときだけ)。
#[cfg(feature = "cuda")]
pub fn alloc_slot(slot_bytes: usize) -> anyhow::Result<SlabSlot<crate::cuda::DevBuffer>> {
    gpu::alloc_slot(slot_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn align_and_layout_are_monotone() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), SLOT_ALIGN);
        assert_eq!(align_up(SLOT_ALIGN), SLOT_ALIGN);
        assert_eq!(align_up(SLOT_ALIGN + 1), 2 * SLOT_ALIGN);
        let (o, total) = slot_layout([1000, 2000, 3000]);
        assert_eq!(o[0], 0);
        assert_eq!(o[1], align_up(1000));
        assert_eq!(o[2], align_up(1000) + align_up(2000));
        assert_eq!(total, o[2] + align_up(3000));
        assert!(total >= 6000, "3 行列ぶん以上");
        assert_eq!(total % SLOT_ALIGN, 0);
    }

    #[test]
    fn slab_reuses_freed_slots_without_new_chunks() {
        let s: Slab<u32> = Slab::new();
        let mut made = 0;
        let mut mk = |bytes: usize| {
            made += 1;
            assert_eq!(bytes % 1000, 0, "チャンクはスロットの整数倍");
            Ok(made)
        };
        let a = s.alloc(1000, &mut mk).unwrap();
        let b = s.alloc(1000, &mut mk).unwrap();
        assert_eq!(made, 1, "同じ形は 1 チャンクに相乗りする");
        assert_ne!(a.idx(), b.idx(), "別スロット");
        let (u, t) = s.slots();
        assert_eq!(u, 2);
        assert!(t >= 2, "1 チャンクに複数スロット: {t}");
    }

    #[test]
    fn freed_slot_is_reused_before_growing() {
        let s: Slab<u32> = Slab::new();
        let mut made = 0;
        let mut mk = |_: usize| {
            made += 1;
            Ok(made)
        };
        let a = s.alloc(4096, &mut mk).unwrap();
        let idx_a = a.idx();
        drop(a);
        let b = s.alloc(4096, &mut mk).unwrap();
        assert_eq!(made, 1, "空きがあるのでチャンクを増やさない");
        assert_eq!(b.idx(), idx_a, "返したスロットを再利用する");
        assert_eq!(s.chunk_count(), 1);
    }

    #[test]
    fn different_slot_sizes_get_different_chunks() {
        let s: Slab<u32> = Slab::new();
        let mut made = 0;
        let mut mk = |_: usize| {
            made += 1;
            Ok(made)
        };
        let a = s.alloc(1024, &mut mk).unwrap();
        let b = s.alloc(2048, &mut mk).unwrap();
        assert_eq!(made, 2, "形が違えばチャンクを分ける");
        assert_eq!(a.slot_bytes(), 1024);
        assert_eq!(b.slot_bytes(), 2048);
        assert_eq!(b.offset() % 2048, 0, "スロット境界に並ぶ");
        assert!(b.idx() < b.n_slots(), "自分のチャンクの内側");
    }

    #[test]
    fn chunk_grows_when_full() {
        let s: Slab<u32> = Slab::new();
        // 1 チャンクに 1 スロットしか入らない大きさ (チャンク 256 MiB より大)。
        let big = (DEFAULT_CHUNK_MB << 20) + SLOT_ALIGN;
        let mut made = 0;
        let mut mk = |_: usize| {
            made += 1;
            Ok(made)
        };
        let a = s.alloc(big, &mut mk).unwrap();
        let b = s.alloc(big, &mut mk).unwrap();
        assert_eq!(made, 2, "1 スロットで埋まったら次のチャンク");
        assert_eq!(s.chunk_count(), 2);
        assert_ne!(a.handle(), b.handle());
        let (u, t) = s.slots();
        assert_eq!((u, t), (2, 2));
    }

    #[test]
    fn alloc_zero_is_an_error() {
        let s: Slab<u32> = Slab::new();
        assert!(s.alloc(0, |_| Ok(1u32)).is_err());
    }

    #[test]
    fn chunk_make_failure_propagates() {
        let s: Slab<u32> = Slab::new();
        let r: anyhow::Result<SlabSlot<u32>> =
            s.alloc(4096, |_| Err(anyhow::anyhow!("no memory")));
        assert!(r.is_err());
        assert_eq!(s.chunk_count(), 0, "失敗したチャンクは登録しない");
    }

    #[test]
    fn slot_drop_returns_index_to_the_same_chunk() {
        let s: Slab<u32> = Slab::new();
        let holder: RefCell<Vec<u32>> = RefCell::new(Vec::new());
        let a = s.alloc(4096, |_| Ok(7u32)).unwrap();
        holder.borrow_mut().push(*a.handle());
        assert_eq!(*a.handle(), 7);
        let idx = a.idx();
        drop(a);
        let b = s.alloc(4096, |_| Ok(99u32)).unwrap();
        assert_eq!(b.idx(), idx);
        assert_eq!(*b.handle(), 7, "同じチャンクを使い続ける");
        assert_eq!(*holder.borrow().first().unwrap(), 7);
    }

    #[test]
    fn env_parsing() {
        assert!(slab_from_value(Some("1"), false));
        assert!(slab_from_value(Some(" on "), false));
        assert!(!slab_from_value(Some("0"), true));
        assert!(slab_from_value(None, true));
        assert!(!slab_from_value(None, false));
        assert_eq!(chunk_mb_from_value(Some("512")), 512);
        assert_eq!(chunk_mb_from_value(Some("0")), DEFAULT_CHUNK_MB);
        assert_eq!(chunk_mb_from_value(Some("x")), DEFAULT_CHUNK_MB);
        assert_eq!(chunk_mb_from_value(None), DEFAULT_CHUNK_MB);
    }
}
