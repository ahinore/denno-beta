//! task 834: expert の H2D を**専用コピーストリーム**で計算と重ねる。
//!
//! これまでの経路 (`DevBuffer::upload_ptr_at`) は、転送元が pageable な mmap の
//! とき pinned のバウンスへ写してから `cuMemcpyHtoD` を **null ストリーム**へ
//! 出して待っていた。同期コピーなので
//!
//! - ホストは転送が終わるまで止まる、
//! - 転送は「先に積んであるカーネルの完了」を待ってから始まる
//!
//! の 2 点で重なりが作れない。cold プロンプトのミスではこれが層ごとに起きる。
//!
//! ここでは
//!
//! 1. pinned の枠 (リング) へホストで 1 回写し、
//! 2. 専用の非ブロッキングストリームへ `cuMemcpyHtoDAsync` を積み、
//! 3. 完了は「その VRAM を読む直前」まで待たない (`cuStreamWaitEvent`)。
//!
//! にする。**デバイスへ渡るバイト列は同期経路と 1 ビットも変わらない**
//! (数値は不変・決定性も不変)。
//!
//! # 安全の前提
//!
//! - 枠の使い回し: 前の DMA がその枠を読んでいる可能性があるので、枠へ写す前に
//!   その枠のイベントをホストで待つ ([`crate::model::h2d_ring::needs_wait`])。
//! - VRAM の解放: 転送中に `cuMemFree` してはいけない。`H2dPending` の `Drop` が
//!   イベントをホストで待つ。`ExpertDev` はこのフィールドを**先頭**に置くこと
//!   (drop は宣言順なので、VRAM を返す前に待ちが入る)。
//! - デバイスの利用: 読む前に `cuStreamWaitEvent(null, ev)` を積む
//!   ([`H2dPending::wait_for_compute`]、冪等)。ドライバに
//!   `cuStreamWaitEvent` が無いときはホストで待つ経路へ落ちる。
//!
//! ゲートは `DENNO_EXPERT_H2D_ASYNC` (既定 off)。何か 1 つでも用意できなければ
//! `None` を返し、呼び出し側は従来の同期経路へ落ちる (fail closed)。

use super::*;
use crate::model::h2d_ring;
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};

/// 枠 1 本の上限。これより大きい行列は同期経路へ (pinned を食いすぎない)。
pub(crate) const MAX_SLOT_BYTES: usize = 8 << 20;

/// スレッドあたりの pinned 合計の上限。
const MAX_TOTAL_BYTES: usize = 32 << 20;

/// 枠のサイズの刻み (1 MiB 単位で育てる)。
const SLOT_GRAIN: usize = 1 << 20;

/// リング 1 枠 (pinned と、その枠を読む DMA のイベント)。
///
/// イベントは枠ごとに 1 本作って**再記録して使い回す**。ExpertDev 側が握る
/// イベントはこのポインタなので、枠が再利用されて後から再記録されても
/// 「同じコピーストリームのもっと後の転送」を待つだけになり、待ちすぎることは
/// あっても**待ち足りないことはない** (安全側)。
struct Slot {
    mem: PinMem,
    ev: CopyEvent,
}

struct Ring {
    slots: Vec<Slot>,
    bytes: usize,
    plan: h2d_ring::RingPlan,
}

thread_local! {
    /// このスレッドの pinned リング (スレッドごとに独立。上げ直しスレッドと
    /// 計算スレッドが混ざらない)。
    static RING: RefCell<Option<Ring>> = const { RefCell::new(None) };
    /// このスレッドの専用コピーストリーム。
    static STREAM: RefCell<Option<CopyStream>> = const { RefCell::new(None) };
}

static UPLOADS: AtomicU64 = AtomicU64::new(0);
static REUSE_WAITS: AtomicU64 = AtomicU64::new(0);
static FALLBACKS: AtomicU64 = AtomicU64::new(0);
static STREAM_WAITS: AtomicU64 = AtomicU64::new(0);
static HOST_SYNCS: AtomicU64 = AtomicU64::new(0);
static LOGGED: std::sync::Once = std::sync::Once::new();

/// 診断用のカウンタ `(積んだ回数, 枠の再利用待ち, 同期経路へ落ちた回数,
/// 計算ストリームへ並べた待ち, ホストで待った回数)`。
pub fn stats() -> (u64, u64, u64, u64, u64) {
    (
        UPLOADS.load(Ordering::Relaxed),
        REUSE_WAITS.load(Ordering::Relaxed),
        FALLBACKS.load(Ordering::Relaxed),
        STREAM_WAITS.load(Ordering::Relaxed),
        HOST_SYNCS.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
thread_local! {
    /// テストで `Drop` / 待ちの順序を数えるための記録。
    pub(crate) static EVENT_LOG: RefCell<Vec<(&'static str, u64)>> = const { RefCell::new(Vec::new()) };
}

/// このスレッドのコピーストリーム (無ければ作る。作れなければ `None`)。
fn stream() -> Option<CUstream> {
    STREAM.with(|c| {
        if let Some(s) = c.borrow().as_ref() {
            return Some(s.0);
        }
        let l = lib().ok()?;
        let create = l.stream_create?;
        const CU_STREAM_NON_BLOCKING: u32 = 0x01;
        let mut st: CUstream = std::ptr::null_mut();
        if unsafe { create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
            return None;
        }
        *c.borrow_mut() = Some(CopyStream(st));
        Some(st)
    })
}

/// `bytes` を 1 MiB 単位に切り上げる (上限で頭打ち)。
fn slot_bytes_for(bytes: usize) -> usize {
    let rounded = bytes.div_ceil(SLOT_GRAIN) * SLOT_GRAIN;
    rounded.min(MAX_SLOT_BYTES).max(SLOT_GRAIN)
}

/// リングを作り直す (`bytes` 以上に育てる)。古い枠の DMA はホストで待ってから捨てる。
fn rebuild(bytes: usize) -> Option<Ring> {
    let n = h2d_ring::slots();
    if bytes > MAX_SLOT_BYTES || n * bytes > MAX_TOTAL_BYTES {
        return None;
    }
    let l = lib().ok()?;
    let ec = l.event_create?;
    const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
    let mut slots = Vec::with_capacity(n);
    for _ in 0..n {
        let mem = PinMem::alloc(bytes).ok()?;
        let mut ev: CUevent = std::ptr::null_mut();
        if unsafe { ec(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
            return None;
        }
        slots.push(Slot {
            mem,
            ev: CopyEvent(ev),
        });
    }
    LOGGED.call_once(|| {
        eprintln!(
            "[exp-h2d] 非同期 H2D: 枠 {} 本 x {} KiB (pinned {} KiB, 専用コピーストリーム)",
            n,
            bytes >> 10,
            (n * bytes) >> 10
        );
    });
    Some(Ring {
        slots,
        bytes,
        plan: h2d_ring::RingPlan::new(n),
    })
}

/// 古いリングの DMA を全部ホストで待つ (枠を捨てる前)。
fn drain(ring: &Ring) {
    let Ok(l) = lib() else { return };
    let Some(es) = l.event_sync else { return };
    for s in &ring.slots {
        unsafe {
            let _ = es(s.ev.0);
        }
    }
}

/// pageable な転送元を非同期で VRAM へ積む。戻り値は完了イベントのハンドル。
///
/// `None` を返したら呼び出し側は従来の同期経路 (`cuMemcpyHtoD`) を使う。
pub(crate) fn upload(dst: CUdeviceptr, data: &[u8]) -> Option<u64> {
    if data.is_empty() {
        return None;
    }
    // リングに載せてよいかの判定は呼び出し側と同じ純関数を通す。
    if !h2d_ring::take_async(true, false, data.len(), MAX_SLOT_BYTES) {
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let l = lib().ok()?;
    let (Some(rec), Some(es)) = (l.event_record, l.event_sync) else {
        return None;
    };
    let st = stream()?;
    let need = slot_bytes_for(data.len());
    RING.with(|cell| {
        let mut g = cell.borrow_mut();
        let too_small = match g.as_ref() {
            Some(r) => r.bytes < data.len(),
            None => true,
        };
        if too_small {
            if let Some(old) = g.take() {
                drain(&old);
            }
            match rebuild(need) {
                Some(r) => *g = Some(r),
                None => {
                    FALLBACKS.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
        }
        let ring = g.as_mut()?;
        let (idx, wait) = ring.plan.pick();
        if wait {
            // この枠の前の DMA が終わるまで写せない (終わっていれば即返る)。
            REUSE_WAITS.fetch_add(1, Ordering::Relaxed);
            unsafe {
                let _ = es(ring.slots[idx].ev.0);
            }
        }
        let slot = &ring.slots[idx];
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), slot.mem.ptr as *mut u8, data.len());
            if (l.memcpy_htod_async)(dst, slot.mem.ptr as *const c_void, data.len(), st)
                != CUDA_SUCCESS
            {
                FALLBACKS.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            if rec(slot.ev.0, st) != CUDA_SUCCESS {
                // 記録できないと完了を知る手段が無い。ここは安全側で待つ。
                let _ = es(slot.ev.0);
                FALLBACKS.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        }
        dbg_htod(data.len());
        UPLOADS.fetch_add(1, Ordering::Relaxed);
        Some(slot.ev.0 as usize as u64)
    })
}

/// ホストでイベントの完了を待つ (VRAM の解放前・ストリーム待ちが使えないとき)。
pub(crate) fn sync(ev: u64) {
    if ev == 0 {
        return;
    }
    HOST_SYNCS.fetch_add(1, Ordering::Relaxed);
    #[cfg(test)]
    EVENT_LOG.with(|c| c.borrow_mut().push(("sync", ev)));
    let Ok(l) = lib() else { return };
    let Some(es) = l.event_sync else { return };
    unsafe {
        let _ = es(ev as usize as CUevent);
    }
}

/// 計算 (null) ストリームに「このイベントの完了待ち」を並べる。
///
/// `cuStreamWaitEvent` が無い / 失敗したときはホストで待つ (待ち足りないまま
/// 進まない)。並べただけなのでホストは止まらない。
pub(crate) fn wait_on_compute_stream(ev: u64) {
    if ev == 0 {
        return;
    }
    let Ok(l) = lib() else {
        sync(ev);
        return;
    };
    match l.stream_wait_event {
        Some(w) if unsafe { w(std::ptr::null_mut(), ev as usize as CUevent, 0) } == CUDA_SUCCESS => {
            STREAM_WAITS.fetch_add(1, Ordering::Relaxed);
        }
        _ => sync(ev),
    }
}

/// 非同期 H2D の完了待ち。expert 1 本ぶんのデバイス領域に付く。
///
/// **`ExpertDev` の先頭のフィールドに置くこと** — 構造体のフィールドは宣言順に
/// drop されるので、`bufs` (VRAM) を解放する前にここでホスト待ちが入る。
pub struct H2dPending {
    ev: u64,
    /// 実際のイベントか (テストで偽の値を入れたときに CUDA を呼ばないため)。
    live: bool,
    /// 計算ストリームへの待ちを積んだか (冪等にする)。
    armed: std::sync::atomic::AtomicBool,
}

impl H2dPending {
    /// 実際に積んだ H2D のイベントから作る。
    pub fn new(ev: u64) -> Self {
        Self {
            ev,
            live: true,
            armed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// テスト用 (CUDA を呼ばない)。
    #[cfg(test)]
    pub(crate) fn new_fake(ev: u64) -> Self {
        Self {
            ev,
            live: false,
            armed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn event(&self) -> u64 {
        self.ev
    }

    /// この expert のデバイス領域を読む前に 1 回だけ呼ぶ (冪等)。
    pub fn wait_for_compute(&self) {
        if self
            .armed
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        #[cfg(test)]
        EVENT_LOG.with(|c| c.borrow_mut().push(("wait", self.ev)));
        if self.live {
            wait_on_compute_stream(self.ev);
        }
    }
}

impl Drop for H2dPending {
    fn drop(&mut self) {
        // VRAM を返す前に、この領域へ書いている DMA を必ず待つ。
        if self.live {
            sync(self.ev);
        } else {
            #[cfg(test)]
            EVENT_LOG.with(|c| c.borrow_mut().push(("drop", self.ev)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn log_take() -> Vec<(&'static str, u64)> {
        EVENT_LOG.with(|c| std::mem::take(&mut *c.borrow_mut()))
    }

    /// 計算前の待ちは 1 回だけ (ptr/gemv が何度呼んでも待ちを積み増さない)。
    #[test]
    fn wait_for_compute_is_idempotent() {
        log_take();
        let p = H2dPending::new_fake(41);
        p.wait_for_compute();
        p.wait_for_compute();
        p.wait_for_compute();
        let log = log_take();
        assert_eq!(log, vec![("wait", 41)], "3 回呼んでも積むのは 1 回");
        drop(p);
        assert_eq!(log_take(), vec![("drop", 41)], "drop では live=false なので待たない");
    }

    /// VRAM を解放する側 (`bufs`) より**先に**ホスト待ちが入る。
    ///
    /// 実物の `ExpertDev` は GPU 無しでは作れないので、同じフィールド順の
    /// 使い捨ての構造体で順序だけを確かめる。
    #[test]
    fn pending_drops_before_buffers() {
        struct Bufs;
        impl Drop for Bufs {
            fn drop(&mut self) {
                EVENT_LOG.with(|c| c.borrow_mut().push(("free", 0)));
            }
        }
        struct Fake {
            pending: Option<H2dPending>,
            bufs: Bufs,
        }
        log_take();
        {
            let _f = Fake {
                pending: Some(H2dPending::new_fake(7)),
                bufs: Bufs,
            };
        }
        let log = log_take();
        assert_eq!(
            log,
            vec![("drop", 7), ("free", 0)],
            "完了待ち (drop) が VRAM の解放より先"
        );
    }

    /// 枠のサイズは 1 MiB 単位で切り上げ、上限で頭打ち。
    #[test]
    fn slot_bytes_rounds_up_and_caps() {
        assert_eq!(slot_bytes_for(1), SLOT_GRAIN);
        assert_eq!(slot_bytes_for(SLOT_GRAIN), SLOT_GRAIN);
        assert_eq!(slot_bytes_for(SLOT_GRAIN + 1), 2 * SLOT_GRAIN);
        assert_eq!(slot_bytes_for(MAX_SLOT_BYTES), MAX_SLOT_BYTES);
        assert_eq!(slot_bytes_for(MAX_SLOT_BYTES + 1), MAX_SLOT_BYTES);
    }
}
