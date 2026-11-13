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

/// 既定値 (**on**)。lab t152b (e24d69021) の g22 条件で採否 OK:
/// 22,000 MiB で 3,954 / 14,139 token の decode が -5.8% / -5.7%、prefill は
/// 長文で -0.2〜-1.3%、md5 SAME、errors 0 (`REPORT_P152.md`)。
pub const SLAB_DEFAULT: bool = true;

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

    /// task 248: このスロットが属するチャンクの識別子 (ハンドルのアドレス)。
    ///
    /// 同じチャンクのスロットは同じ値を返す。「1 本の確保のために expert を
    /// 丸ごと空けるチャンク」を選ぶのに使う (中身がバラバラに追い出されて
    /// 実メモリが戻らない、を避ける)。
    pub fn chunk_id(&self) -> usize {
        std::sync::Arc::as_ptr(&self.chunk) as usize
    }

    /// このチャンク 1 本のバイト数 (`slot_bytes * n_slots`)。
    pub fn chunk_bytes(&self) -> usize {
        self.chunk.slot_bytes.saturating_mul(self.chunk.n_slots)
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

    /// **空になったチャンク**を落としてデバイスメモリを返す (`need` バイトまで)。
    ///
    /// task 152: slab ではスロットを返しても `cuMemFree` しないので、LRU の
    /// eviction だけでは**実メモリが戻らない**。注意深い消費者 (attention の
    /// ミラー・prefill の一時バッファ) が確保に失敗したとき、ここで空チャンクを
    /// 解放して場所を作る (lab t152a の g24 はこれが無くて 14k で CPU attention に
    /// 落ちた)。
    ///
    /// 使用中のスロットがあるチャンクは落とさない (`SlabSlot` が `Arc` で保持して
    /// いるので、落とすと重みが消える)。`need == 0` は何もしない。
    /// 戻り値は解放したバイト数。
    pub fn release_empty(&self, need: usize) -> usize {
        if need == 0 {
            return 0;
        }
        let mut chunks = match self.chunks.lock() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        let mut freed = 0usize;
        let mut i = 0usize;
        while i < chunks.len() {
            let empty = chunks[i].used() == 0;
            if empty {
                let bytes = chunks[i].slot_bytes.saturating_mul(chunks[i].n_slots);
                // ここで Arc を落とす。スロットが 1 本も無いので参照は自分だけ
                // (チャンク表のロックを持っているので、他のスレッドが空きを
                //  取り出すことはない)。
                chunks.remove(i);
                freed = freed.saturating_add(bytes);
                if freed >= need {
                    break;
                }
            } else {
                i += 1;
            }
        }
        freed
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

    /// 空チャンクを落として `need` バイトまで実メモリを返す。
    pub fn release_empty(need: usize) -> usize {
        slab().release_empty(need)
    }

    /// `(使用中スロット, 全スロット, 確保済みバイト)` (診断用)。
    pub fn stats() -> (usize, usize, usize) {
        let s = slab();
        let (u, t) = s.slots();
        (u, t, s.allocated_bytes())
    }

    /// task 256: チャンク数 (診断と解放数の記録用)。
    pub fn chunk_count() -> usize {
        slab().chunk_count()
    }
}

/// 空チャンクを落として `need` バイトまで実メモリを返す (cuda feature のときだけ)。
#[cfg(feature = "cuda")]
pub fn release_empty(need: usize) -> usize {
    gpu::release_empty(need)
}

/// cuda feature 無しでは何も解放しない (slab 経路が無い)。
#[cfg(not(feature = "cuda"))]
pub fn release_empty(_need: usize) -> usize {
    0
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

/// task 256: いま slab が持っているチャンク数 (cuda 無しは 0)。
pub fn chunk_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        gpu::chunk_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// `slot_bytes` のスロットを確保する (cuda feature のときだけ)。
#[cfg(feature = "cuda")]
pub fn alloc_slot(slot_bytes: usize) -> anyhow::Result<SlabSlot<crate::cuda::DevBuffer>> {
    gpu::alloc_slot(slot_bytes)
}

/// task 256: 1 チャンクの位置と冷たさ。
///
/// `last_use` は**チャンクの中で一番新しい最終使用**。小さいほど「どの expert も
/// 久しく使われていない」= 冷たい。追い出しで失うヒットが少ない順に選ぶため、
/// チャンク内の最古ではなく**最新**を鍵にする (1 本だけ古い玉が混ざったチャンクを
/// 冷たいと誤認しない)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkSpan {
    /// slab のチャンク識別子 ([`SlabSlot::chunk_id`])。
    pub id: usize,
    /// チャンクの先頭アドレス (バイト)。
    pub addr: usize,
    /// チャンク 1 本のバイト数。
    pub bytes: usize,
    /// チャンクの中で一番新しい最終使用。
    pub last_use: u64,
}

/// task 256: 丸ごと空けるチャンクの選択結果。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictPlan {
    /// 空けるチャンクの id (アドレス順)。
    pub ids: Vec<usize>,
    /// 空けるバイト数の合計。
    pub bytes: usize,
    /// 選んだ範囲の冷たさ (範囲内で一番新しい最終使用)。
    pub last_use: u64,
}

/// `free` バイトの空きがある状態で、`need` バイトの**連続した**空きを作るために
/// 丸ごと空けるチャンクを選ぶ (純関数、task 256)。
///
/// - 既に `free >= need` なら何も選ばない (追い出さない)。
/// - アドレスが連続するチャンクの並び (run) の中から、不足分
///   `need - free` 以上になる**窓**を選ぶ。窓のチャンクを全部空けるので、
///   解放後の範囲は必ず連続になる (slab はチャンクが空にならないと実メモリを
///   返さない。バラバラに追い出しても連続した空きはできない)。
/// - 窓は「不足分を超えたところで止める」ので、追い出す量は
///   `不足分 + 1 チャンク未満` に収まる (要求 + 1 チャンクを超えない)。
/// - 窓の選び方: 一番冷たい窓 (窓内で一番新しい最終使用が小さい)、同着は
///   合計バイトが小さい方、次に id 列の辞書順 (決定的)。
/// - どの run でも不足分に届かなければ、全チャンクから冷たい順に必要量まで選ぶ
///   (連続は諦めるが、実メモリは戻る)。
pub fn plan_eviction(spans: &[ChunkSpan], free: usize, need: usize) -> EvictPlan {
    let deficit = need.saturating_sub(free);
    if deficit == 0 || spans.is_empty() {
        return EvictPlan::default();
    }
    // アドレス順に並べ、隙間なく続くチャンクを run に切る。
    let mut v = spans.to_vec();
    v.sort_unstable_by_key(|s| (s.addr, s.id));
    let mut runs: Vec<Vec<ChunkSpan>> = Vec::new();
    for s in v {
        let cont = runs
            .last()
            .and_then(|r| r.last())
            .is_some_and(|p| p.addr.saturating_add(p.bytes) == s.addr);
        if cont {
            if let Some(r) = runs.last_mut() {
                r.push(s);
            }
        } else {
            runs.push(vec![s]);
        }
    }
    let mut best: Option<EvictPlan> = None;
    for r in &runs {
        for i in 0..r.len() {
            let n = count_until(&r[i..], deficit);
            if n == 0 {
                continue;
            }
            let window = &r[i..i + n];
            let sum: usize = window.iter().map(|s| s.bytes).sum();
            if sum < deficit {
                // この run は不足分に届かない (以降の i も届かない)。
                break;
            }
            let cand = EvictPlan {
                ids: window.iter().map(|s| s.id).collect(),
                bytes: sum,
                last_use: window.iter().map(|s| s.last_use).max().unwrap_or(0),
            };
            if best.as_ref().map_or(true, |b| cooler(&cand, b)) {
                best = Some(cand);
            }
        }
    }
    if let Some(b) = best {
        return b;
    }
    // run だけで不足分に届かない: 全体から冷たい順に選ぶ。
    let mut all = spans.to_vec();
    all.sort_unstable_by_key(|s| (s.last_use, s.id));
    let mut out = EvictPlan::default();
    for s in all {
        if out.bytes >= deficit {
            break;
        }
        out.bytes = out.bytes.saturating_add(s.bytes);
        out.last_use = out.last_use.max(s.last_use);
        out.ids.push(s.id);
    }
    out
}

/// task 290: 確保が**既に失敗した**あとの plan。`deficit` が 0 でも必ず 1 本選ぶ。
///
/// `cuMemGetInfo` の空きが要求以上でも、断片化していれば `cuMemAlloc` は失敗する
/// (lab t286c の UD-Q4_K_XL 14k: 空き 3 MiB で 3 MiB の確保が error 2)。
/// そのとき `deficit = need - free = 0` なので [`plan_eviction`] は 1 本も選ばず、
/// 呼び出し側は「空けられない」と判断して諦めてしまう。失敗後の再試行では
/// 空きを見ずに**冷たいチャンクを 1 本**空ける。
pub fn plan_eviction_forced(spans: &[ChunkSpan], free: usize, need: usize) -> EvictPlan {
    let p = plan_eviction(spans, free, need);
    if !p.ids.is_empty() || spans.is_empty() {
        return p;
    }
    plan_eviction(spans, free, free.saturating_add(1))
}

/// 先頭から合計が `deficit` 以上になるまでの本数 (必ず 1 本以上)。
fn count_until(run: &[ChunkSpan], deficit: usize) -> usize {
    let mut sum = 0usize;
    for (i, s) in run.iter().enumerate() {
        sum = sum.saturating_add(s.bytes);
        if sum >= deficit {
            return i + 1;
        }
    }
    run.len()
}

/// `a` のほうが冷たい (同着は小さい・id 列が小さい) か。
fn cooler(a: &EvictPlan, b: &EvictPlan) -> bool {
    (a.last_use, a.bytes, &a.ids) < (b.last_use, b.bytes, &b.ids)
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

    /// task 152: 空チャンクだけが解放される (使用中のチャンクは残る)。
    #[test]
    fn release_empty_frees_only_empty_chunks() {
        let s: Slab<u32> = Slab::new();
        let made = std::cell::Cell::new(0u32);
        let mut mk = |_: usize| {
            made.set(made.get() + 1);
            Ok(made.get())
        };
        let a = s.alloc(4096, &mut mk).unwrap();
        // 2 本目は同じチャンクの別スロット (まだ空きがある)。
        let b = s.alloc(4096, &mut mk).unwrap();
        assert_eq!(made.get(), 1);
        assert_eq!(s.release_empty(1 << 20), 0, "使用中があるので解放しない");
        assert_eq!(s.chunk_count(), 1);
        drop(b);
        // b を返しただけでは a が使っているのでまだ解放しない。
        assert_eq!(s.release_empty(1 << 20), 0);
        drop(a);
        let freed = s.release_empty(1);
        assert!(freed >= 4096, "空になったら解放する: {freed}");
        assert_eq!(s.chunk_count(), 0);
        // 解放後に確保すると新しいチャンクを作る。
        let c = s.alloc(4096, &mut mk).unwrap();
        assert_eq!(made.get(), 2);
        assert_eq!(*c.handle(), 2);
    }

    #[test]
    fn release_empty_zero_is_a_noop() {
        let s: Slab<u32> = Slab::new();
        let a = s.alloc(4096, |_| Ok(1u32)).unwrap();
        drop(a);
        assert_eq!(s.release_empty(0), 0, "need=0 は何もしない");
        assert_eq!(s.chunk_count(), 1);
    }

    #[test]
    fn alloc_zero_is_an_error() {
        let s: Slab<u32> = Slab::new();
        assert!(s.alloc(0, |_| Ok(1u32)).is_err());
    }

    /// task 256: 連続した空きを検査するための偽デバイス (MiB 単位)。
    ///
    /// `cuMemAlloc` は「連続した空き」しか返せないので、チャンクを丸ごと
    /// 空けたあとに本当に連続領域ができるかを、アドレスを持つ偽デバイスで見る。
    struct FakeDev {
        total: usize,
        /// (先頭, 長さ) MiB。確保済みの区間だけを持つ。
        used: Vec<(usize, usize)>,
    }

    impl FakeDev {
        fn new(total: usize) -> Self {
            Self {
                total,
                used: Vec::new(),
            }
        }

        /// 先頭から最初に収まる隙間へ置く (first fit)。置けなければ `None`。
        fn alloc(&mut self, bytes: usize) -> Option<usize> {
            let mut segs = self.used.clone();
            segs.sort_unstable();
            let mut start = 0usize;
            for (s, l) in segs {
                if s.saturating_sub(start) >= bytes {
                    break;
                }
                start = start.max(s.saturating_add(l));
            }
            if start.saturating_add(bytes) > self.total {
                return None;
            }
            self.used.push((start, bytes));
            Some(start)
        }

        fn free(&mut self, start: usize) {
            self.used.retain(|(s, _)| *s != start);
        }

        fn free_bytes(&self) -> usize {
            self.total
                .saturating_sub(self.used.iter().map(|(_, l)| *l).sum::<usize>())
        }

        /// 連続した `bytes` MiB の空きがあるか。
        fn can_alloc_contig(&self, bytes: usize) -> bool {
            if bytes > self.free_bytes() {
                return false;
            }
            let mut segs = self.used.clone();
            segs.sort_unstable();
            let mut start = 0usize;
            for (s, l) in segs {
                if s.saturating_sub(start) >= bytes {
                    return true;
                }
                start = start.max(s.saturating_add(l));
            }
            self.total.saturating_sub(start) >= bytes
        }
    }

    /// task 256: チャンクを丸ごと空けると「連続した N MiB」ができる。
    ///
    /// 追い出すのは run (アドレスが連続するチャンク列) の中の窓だけ。
    /// 追い出した量は不足分 + 1 チャンク未満に収まる。
    #[test]
    fn plan_eviction_creates_a_contiguous_range() {
        let mib = 1usize << 20;
        let chunk = 256usize;
        // 2,816 MiB のカード: A 4 本 (0..1,024)、dense 1,280 (1,024..2,304)、
        // B 2 本 (2,304..2,816)。空き 0。
        let mut dev = FakeDev::new(2816);
        let mut starts = Vec::new();
        for _ in 0..4 {
            starts.push(dev.alloc(chunk).expect("A"));
        }
        assert_eq!(dev.alloc(1280), Some(1024));
        let b_starts: Vec<usize> = (0..2).map(|_| dev.alloc(chunk).expect("B")).collect();
        assert_eq!(dev.free_bytes(), 0);

        // A は冷たい (last_use 10..14)、B は熱い (100, 101)。
        let mut spans: Vec<ChunkSpan> = starts
            .iter()
            .enumerate()
            .map(|(i, s)| ChunkSpan {
                id: 100 + i,
                addr: s * mib,
                bytes: chunk * mib,
                last_use: 10 + i as u64,
            })
            .collect();
        spans.extend(b_starts.iter().enumerate().map(|(i, s)| ChunkSpan {
            id: 200 + i,
            addr: s * mib,
            bytes: chunk * mib,
            last_use: 100 + i as u64,
        }));

        // 600 MiB の確保: 不足分 600 → A の 3 本 (768 MiB)。
        let plan = plan_eviction(&spans, dev.free_bytes() * mib, 600 * mib);
        assert_eq!(plan.bytes, 768 * mib, "3 本で足りる: {plan:?}");
        assert_eq!(plan.ids, vec![100, 101, 102], "冷たい A の先頭から: {plan:?}");
        assert!(
            plan.bytes < 600 * mib + chunk * mib,
            "要求 + 1 チャンクを超えた: {plan:?}"
        );
        for (id, s) in plan.ids.iter().zip(starts.iter()) {
            let _ = id;
            dev.free(*s);
        }
        assert!(dev.can_alloc_contig(600), "連続 600 MiB が取れない");
        assert!(!dev.can_alloc_contig(900), "空けすぎ (900 MiB も取れてしまう)");

        // 空きが足りていれば 1 本も追い出さない。
        assert_eq!(plan_eviction(&spans, 600 * mib, 600 * mib), EvictPlan::default());
        // 0 バイトの要求でも追い出さない。
        assert_eq!(plan_eviction(&spans, 0, 0), EvictPlan::default());
    }

    /// task 256: run が 2 つあるときは、より冷たい窓を選ぶ。
    #[test]
    fn plan_eviction_prefers_the_colder_run() {
        let mib = 1usize << 20;
        let span = |id: usize, addr_mib: usize, last_use: u64| ChunkSpan {
            id,
            addr: addr_mib * mib,
            bytes: 256 * mib,
            last_use,
        };
        // run1: 0, 256 (熱い 90, 91) / run2: 4,096, 4,352 (冷たい 1, 2)。
        let spans = vec![
            span(1, 0, 90),
            span(2, 256, 91),
            span(3, 4096, 1),
            span(4, 4352, 2),
        ];
        let plan = plan_eviction(&spans, 0, 300 * mib);
        assert_eq!(plan.ids, vec![3, 4], "冷たい run2: {plan:?}");
        // run が 1 本ずつしかなくても、全体から冷たい順に選ぶ。
        let apart = vec![span(1, 0, 90), span(2, 512, 5)];
        let plan = plan_eviction(&apart, 0, 300 * mib);
        assert_eq!(plan.ids, vec![2, 1], "全体の冷たい順: {plan:?}");
        assert_eq!(plan.bytes, 512 * mib);
    }

    /// task 290: 確保が失敗したあとは、空きが要求以上に見えても 1 本は空ける。
    #[test]
    fn plan_eviction_forced_always_frees_one_chunk() {
        let mib = 1usize << 20;
        let span = |id: usize, addr_mib: usize, last_use: u64| ChunkSpan {
            id,
            addr: addr_mib * mib,
            bytes: 256 * mib,
            last_use,
        };
        let spans = vec![span(1, 0, 90), span(2, 4, 5), span(3, 8, 40)];
        // 空き 3 MiB / 要求 3 MiB → 通常の plan は 1 本も選ばない (断片化で落ちた)。
        assert_eq!(
            plan_eviction(&spans, 3 * mib, 3 * mib),
            EvictPlan::default()
        );
        let forced = plan_eviction_forced(&spans, 3 * mib, 3 * mib);
        assert_eq!(forced.ids, vec![2], "一番冷たい 1 本: {forced:?}");
        assert_eq!(forced.bytes, 256 * mib);
        // 足りないときは通常の plan と同じ (2 本以上)。
        let forced2 = plan_eviction_forced(&spans, 0, 300 * mib);
        assert_eq!(forced2.ids.len(), 2, "{forced2:?}");
        // チャンクが無ければ何もできない。
        assert_eq!(
            plan_eviction_forced(&[], 3 * mib, 3 * mib),
            EvictPlan::default()
        );
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
