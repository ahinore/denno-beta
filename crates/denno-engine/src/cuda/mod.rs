//! P17: CUDA int8 オフロード (feature = `"cuda"`)。
//!
//! CPU int8 パス ([`crate::tensor`] の `quantize_row_i8` / `matvec_rows_bt_i8`) と
//! **ビット一致**する GPU カーネル群。依存クレートは無し: CUDA Driver API は
//! `dlopen("libcuda.so.1")` で実行時解決するため、feature を外したビルド
//! (CI 等の GPU 無し環境) には一切影響しない。
//!
//! # exactness の設計
//!
//! - アクティベーションの per-token 動的量子化 (maxabs → scale → round → clamp) は
//!   GPU カーネル内で CPU と同一の演算列で実行 (`roundf` = `f32::round`、
//!   max は丸めを伴わないので走査順序に依存しない)。
//! - i8×i8 → i32 累積は整数なので結合則が成立し、warp リダクションの順序に依存
//!   しない = CPU の SIMD カーネル (AVX2/sdot) とビット一致。
//! - スケール乗算は `(float)dot * ascale[i] * scale[j]` を左から順に (CPU と同じ
//!   結合順)。FMA 縮約はカーネルに加算が無いため構造的に発生しない。
//!
//! # 使用方法
//!
//! ```ignore
//! let ctx = CudaCtx::new()?;
//! let w = ctx.upload_i8(&q.w)?;
//! let scale = ctx.upload_f32(&q.scale)?;
//! let out = ctx.linear_bt_i8(&a, &w, &scale, m, k, n)?; // CPU とビット一致
//! ```
//!
//! PTX (`kernels/kernels_int8.ptx`) は compute_75 向けに nvcc で生成したものを
//! コミットしており、ドライバ JIT が以降の全アーキテクチャ (sm_120 を含む) で
//! 実行する。再生成:
//!
//! ```text
//! nvcc -ptx -arch=compute_75 -fmad=false -O3 kernels_int8.cu -o kernels_int8.ptx
//! ```

#![allow(non_snake_case, non_camel_case_types)]

use anyhow::{anyhow, bail, Result};
use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;


use crate::model::exp_pf;

use crate::model::pf_host;


use crate::quant::Quant8Linear;

mod attn;
mod ctx;
mod gdn;
mod gemm;
pub(crate) mod h2d;
mod mix;
mod moe_dec;
pub(crate) mod moe_pf;
mod pf_chain;
// task 1377: PDL (`cuLaunchKernelEx` + PROGRAMMATIC_STREAM_SERIALIZATION)。
pub(crate) mod pdl;
#[cfg(test)] mod tests;

pub use self::attn::*;
pub use self::gdn::*;
pub use self::gemm::*;
pub use self::mix::*;
pub use self::moe_dec::*;
pub use self::moe_pf::*;
pub use self::pf_chain::*;
#[cfg(test)] pub use self::tests::*;


pub(crate) type CUresult = i32;

pub(crate) type CUdevice = c_int;

pub(crate) type CUcontext = *mut c_void;

pub(crate) type CUmodule = *mut c_void;

pub(crate) type CUfunction = *mut c_void;

pub(crate) type CUdeviceptr = u64;


const RTLD_NOW: c_int = 2;

pub(crate) const CUDA_SUCCESS: CUresult = 0;


// dlopen/dlsym (glibc 2.34 以降は libc に統合、Ubuntu 22.04 コンテナで使用)
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}


pub(crate) type FnCuInit = unsafe extern "C" fn(u32) -> CUresult;

pub(crate) type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;

pub(crate) type FnCuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;

pub(crate) type FnCuDeviceComputeCapability =
    unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUresult;

pub(crate) type FnCuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;

pub(crate) type FnCuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;

pub(crate) type FnCuCtxSynchronize = unsafe extern "C" fn() -> CUresult;

pub(crate) type FnCuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;

pub(crate) type FnCuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;

pub(crate) type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;

// task 186: カーネル属性 (レジスタ数・ローカルメモリ) の読み出し。診断専用。
pub(crate) type FnCuFuncGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUfunction) -> CUresult;

pub(crate) type FnCuMemAllocV2 = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;

pub(crate) type FnCuMemFreeV2 = unsafe extern "C" fn(CUdeviceptr) -> CUresult;

pub(crate) type FnCuMemsetD32Async = unsafe extern "C" fn(CUdeviceptr, u32, usize, CUstream) -> CUresult;

pub(crate) type FnCuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;

/// task 818: デバイスの総 VRAM (`cuDeviceTotalMem_v2`)。context を作らない。
pub(crate) type FnCuDeviceTotalMem = unsafe extern "C" fn(*mut usize, CUdevice) -> CUresult;

pub(crate) type FnCuMemcpyHtoDV2 = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;

pub(crate) type FnCuMemcpyDtoHV2 = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;

pub(crate) type FnCuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CUresult;

pub(crate) type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;

// task 227: mmap のページを**写さずに** DMA するための任意シンボル。
// 古い driver や file backed の写像で失敗しても、呼び出し側がバウンス経路へ
// 落ちるだけ (fail closed)。
pub(crate) type FnCuMemHostRegisterV2 = unsafe extern "C" fn(*mut c_void, usize, u32) -> CUresult;

pub(crate) type FnCuMemHostUnregister = unsafe extern "C" fn(*mut c_void) -> CUresult;

pub(crate) type FnCuMemHostGetDevicePointerV2 =
    unsafe extern "C" fn(*mut CUdeviceptr, *mut c_void, u32) -> CUresult;

pub(crate) type FnCuMemcpyHtoDAsyncV2 =
    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;

pub(crate) type FnCuMemcpyDtoHAsyncV2 =
    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;

pub(crate) type FnCuMemcpyDtoDAsyncV2 =
    unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;

pub(crate) type FnCuMemcpyDtoD = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;

pub(crate) type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;

pub(crate) type FnCuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;

// task 174: 診断 (DENNO_LAUNCH_DBG) で起動したカーネル名を出すための任意シンボル。
pub(crate) type FnCuFuncGetName = unsafe extern "C" fn(*mut *const c_char, CUfunction) -> CUresult;

// s167 (P6): h2 の D2H を専用コピーストリームへ載せるための driver API。
// すべて Option で持つ — 古い driver で dlsym に失敗しても全体を落とさず、
// 呼び出し側が従来の同期 D2H へフォールバックする (fail closed)。
pub(crate) type FnCuStreamCreate = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;

pub(crate) type FnCuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;

pub(crate) type FnCuStreamWaitEvent = unsafe extern "C" fn(CUstream, CUevent, u32) -> CUresult;

pub(crate) type FnCuEventCreate = unsafe extern "C" fn(*mut CUevent, u32) -> CUresult;

pub(crate) type FnCuEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;

pub(crate) type FnCuEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;

pub(crate) type FnCuEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;

// s169 (P8): prefill 位相計測 — ストリーム上の 2 点間の経過 (ms) を読む。
pub(crate) type FnCuEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;

pub(crate) type FnCuLaunchKernel = unsafe extern "C" fn(
    CUfunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    CUstream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

pub(crate) type CUstream = *mut c_void;

/// task 1377: PDL 付き起動 (`cuLaunchKernelEx`)。任意シンボル (Option で持つ)。
pub(crate) type FnCuLaunchKernelEx = unsafe extern "C" fn(
    *const pdl::CUlaunchConfig,
    CUfunction,
    *mut *mut c_void,
    *mut *mut c_void,
) -> CUresult;

pub(crate) type CUevent = *mut c_void;


/// Driver API の関数ポインタ群 (dlsym は1度だけ行う)
pub(crate) struct CudaLib {
    cu_init: FnCuInit,
    device_get: FnCuDeviceGet,
    device_name: FnCuDeviceGetName,
    device_cc: FnCuDeviceComputeCapability,
    primary_ctx_retain: FnCuDevicePrimaryCtxRetain,
    ctx_set_current: FnCuCtxSetCurrent,
    ctx_sync: FnCuCtxSynchronize,
    driver_version: FnCuDriverGetVersion,
    module_load_data: FnCuModuleLoadData,
    module_get_function: FnCuModuleGetFunction,
    mem_alloc: FnCuMemAllocV2,
    mem_free: FnCuMemFreeV2,
    mem_get_info: Option<FnCuMemGetInfo>,
    // task 818: context 無しで総 VRAM を引く (`cuMemGetInfo` は current context が要る)。
    device_total_mem: Option<FnCuDeviceTotalMem>,
    memcpy_htod: FnCuMemcpyHtoDV2,
    memcpy_dtoh: FnCuMemcpyDtoHV2,
    memset_d32_async: FnCuMemsetD32Async,
    mem_host_alloc: FnCuMemHostAlloc,
    mem_host_free: FnCuMemFreeHost,
    // task 227: mmap のページを直接 DMA する (写しを消す) ための 3 本。
    // 無い/失敗する driver ではバウンス経路のまま。
    mem_host_register: Option<FnCuMemHostRegisterV2>,
    mem_host_unregister: Option<FnCuMemHostUnregister>,
    mem_host_get_device_ptr: Option<FnCuMemHostGetDevicePointerV2>,
    memcpy_htod_async: FnCuMemcpyHtoDAsyncV2,
    memcpy_dtoh_async: FnCuMemcpyDtoHAsyncV2,
    memcpy_dtod_async: FnCuMemcpyDtoDAsyncV2,
    // s141b (Task B7c): 同期 D2D (GDN 状態の正本 -> ミラー seed 用)。
    memcpy_dtod: FnCuMemcpyDtoD,
    // s48 (C1-a): 動的 smem の opt-in (cuFuncSetAttribute)。CUDA 9.0 以降に存在
    // するが、dlsym 失敗でも全体を落とさないよう Option (無ければ 48KB 既定のまま)。
    device_get_attribute: Option<FnCuDeviceGetAttribute>,
    func_set_attribute: Option<FnCuFuncSetAttribute>,
    // task 174: 診断専用 (DENNO_LAUNCH_DBG)。無ければアドレスで代用する。
    func_get_name: Option<FnCuFuncGetName>,
    // task 186: 診断専用 (DENNO_KERNEL_ATTR)。JIT 後の実レジスタ数とローカル量。
    func_get_attribute: Option<FnCuFuncGetAttribute>,
    // s167 (P6): 専用コピーストリーム (h2 の D2H を qkv/z GEMM と重ねる)。
    stream_create: Option<FnCuStreamCreate>,
    stream_destroy: Option<FnCuStreamDestroy>,
    stream_wait_event: Option<FnCuStreamWaitEvent>,
    event_create: Option<FnCuEventCreate>,
    event_destroy: Option<FnCuEventDestroy>,
    event_record: Option<FnCuEventRecord>,
    event_sync: Option<FnCuEventSynchronize>,
    // s169 (P8): 位相計測 (cuEventElapsedTime)。無ければ計測は静かに OFF。
    event_elapsed_time: Option<FnCuEventElapsedTime>,
    launch: FnCuLaunchKernel,
    // task 1377: PDL 用 (CUDA 11.8 以降)。無ければ従来の launch だけを使う。
    launch_ex: Option<FnCuLaunchKernelEx>,
}


unsafe fn sym(handle: *mut c_void, name: &'static str) -> Result<*mut c_void> {
    let cname = std::ffi::CString::new(name).unwrap();
    let p = dlsym(handle, cname.as_ptr());
    if p.is_null() {
        bail!("dlsym({name}) が失敗 — driver が古い/不完全の可能性");
    }
    Ok(p)
}


/// dlsym だが失敗を許容する (オプションシンボル用)。
unsafe fn sym_opt(handle: *mut c_void, name: &'static str) -> Option<*mut c_void> {
    let cname = std::ffi::CString::new(name).unwrap();
    let p = dlsym(handle, cname.as_ptr());
    if p.is_null() {
        None
    } else {
        Some(p)
    }
}


// task 138: H2D / D2H の全呼び出しを台帳 (`DENNO_Q4_PROFILE`) へ数える trampoline。
//
// コピーは cuda.rs の数十箇所から素の関数ポインタで呼ばれているので、呼び出し側を
// 触らずに**全部**を数えるため、`CudaLib` に入れるポインタ自体を差し替える。
// 実関数はここに 1 度だけ保存し、trampoline が転送バイト数と所要時間を積む。
// 台帳 off では `stamp()` が `None` を返すだけで、転送そのものは従来と同一。
static REAL_HTOD: std::sync::atomic::AtomicPtr<c_void> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

static REAL_DTOH: std::sync::atomic::AtomicPtr<c_void> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

pub(crate) static REAL_HTOD_ASYNC: std::sync::atomic::AtomicPtr<c_void> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

static REAL_DTOH_ASYNC: std::sync::atomic::AtomicPtr<c_void> =
    std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());


unsafe extern "C" fn ledger_htod(d: CUdeviceptr, s: *const c_void, n: usize) -> CUresult {
    let t = crate::model::q4ledger::stamp();
    let f: FnCuMemcpyHtoDV2 = std::mem::transmute(REAL_HTOD.load(Ordering::Relaxed));
    let r = f(d, s, n);
    crate::model::q4ledger::note_h2d(n, t);
    r
}


unsafe extern "C" fn ledger_dtoh(d: *mut c_void, s: CUdeviceptr, n: usize) -> CUresult {
    let t = crate::model::q4ledger::stamp();
    let f: FnCuMemcpyDtoHV2 = std::mem::transmute(REAL_DTOH.load(Ordering::Relaxed));
    let r = f(d, s, n);
    crate::model::q4ledger::note_d2h(n, t);
    r
}


unsafe extern "C" fn ledger_htod_async(
    d: CUdeviceptr,
    s: *const c_void,
    n: usize,
    st: CUstream,
) -> CUresult {
    let t = crate::model::q4ledger::stamp();
    let f: FnCuMemcpyHtoDAsyncV2 = std::mem::transmute(REAL_HTOD_ASYNC.load(Ordering::Relaxed));
    let r = f(d, s, n, st);
    crate::model::q4ledger::note_h2d(n, t);
    r
}


unsafe extern "C" fn ledger_dtoh_async(
    d: *mut c_void,
    s: CUdeviceptr,
    n: usize,
    st: CUstream,
) -> CUresult {
    let t = crate::model::q4ledger::stamp();
    let f: FnCuMemcpyDtoHAsyncV2 = std::mem::transmute(REAL_DTOH_ASYNC.load(Ordering::Relaxed));
    let r = f(d, s, n, st);
    crate::model::q4ledger::note_d2h(n, t);
    r
}


// ---------------- task 153: decode のイベント時系列 (DENNO_Q4_TLINE) ----------------
//
// `DENNO_Q4_PROFILE` の台帳は「ホストがどこで待ったか」しか見ていない。ここでは
// **ストリームに計時イベントを打って**、区間ごとの GPU 実行時間 (busy) と、GPU が
// ホスト待ちで空いていた時間 (idle) を分けて読む。
//
// - 区間 (`q4ledger::enter`) の入口と出口で 1 本ずつイベントを打ち、その 2 本の
//   `cuEventElapsedTime` を「GPU の span」として積む。
// - `DENNO_Q4_TLINE_K=1` のときは**起動ごと**にも前後 1 本ずつ打ち、その差を
//   カーネルの実行時間として積む (span - busy = その区間の GPU idle)。
// - イベントはプールして伸ばすだけ (grow-only)。読むのはリクエスト終端の 1 回だけ。
//   ホットパスに足すのは `cuEventRecord` の呼び出しだけで、同期は 1 本も足さない。
// - 記録は decode の窓 (`q4ledger::tline_begin` .. `tline_end`) だけ。上限
//   (`MAX_EVENTS`) を超えたら静かに止める (台帳の数字はそのまま残る)。
pub(crate) mod tline {
    use super::*;
    use std::cell::RefCell;

    /// プールの上限 (128 トークンの窓 x 約 3,000 本 = 約 38 万本を収める)。
    ///
    /// task 155: 400_000 だと診断行を足したときに窓の途中で尽きて以後の記録が
    /// 止まる (t155b で発生)。60 万本まで許す。
    const MAX_EVENTS: usize = 600_000;

    pub(crate) struct Pool {
        /// 計時イベント (grow-only)。
        evs: Vec<CUevent>,
        /// この窓で使ったイベント数 (= 次の記録位置)。
        used: usize,
        /// 区間: (item, 入口イベント, 出口イベント)。
        spans: Vec<(u8, u32, u32)>,
        /// 起動: (item, 前イベント, 後イベント)。K ゲートのときだけ積む。
        launch: Vec<(u8, u32, u32)>,
        /// 作れなかった (以後は静かに止める)。
        dead: bool,
    }

    impl Pool {
        const fn new() -> Self {
            Self {
                evs: Vec::new(),
                used: 0,
                spans: Vec::new(),
                launch: Vec::new(),
                dead: false,
            }
        }
    }

    thread_local! {
        static P: RefCell<Pool> = const { RefCell::new(Pool::new()) };
    }

    /// 区間ごとの GPU 時間 (ms)。`span` は入口から出口までの GPU 時間
    /// (ホスト待ちで GPU が空いていた分を含む)、`busy` は起動ごとの実行時間の和。
    #[derive(Clone, Copy, Default, Debug)]
    pub struct Item {
        pub span_ms: f64,
        pub busy_ms: f64,
        pub spans: u64,
        pub launches: u64,
    }

    /// 起動ごとのイベントを打つか (`DENNO_Q4_TLINE_K=1`、または prefill の
    /// ホスト台帳が開いているとき = task 254 の GPU 実時間)。
    pub(crate) fn k_on() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *ON.get_or_init(|| std::env::var_os("DENNO_Q4_TLINE_K").is_some()) {
            return true;
        }
        crate::model::q4ledger::pf_span_active()
    }

    pub(crate) fn grow(p: &mut Pool, want: usize) -> bool {
        let Some(create) = lib().ok().and_then(|l| l.event_create) else {
            return false;
        };
        while p.evs.len() < want {
            let mut ev: CUevent = std::ptr::null_mut();
            // CU_EVENT_DEFAULT (0) = 計時有効。作れなければ計測を諦める。
            if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                return false;
            }
            p.evs.push(ev);
        }
        true
    }

    /// 既定ストリームにイベントを 1 本打ち、その通し番号を返す。
    pub(crate) fn mark() -> Option<u32> {
        P.with(|c| {
            let mut p = c.borrow_mut();
            if p.dead {
                return None;
            }
            if p.used >= MAX_EVENTS {
                p.dead = true;
                return None;
            }
            if p.used >= p.evs.len() {
                let want = p.evs.len() + 2048;
                if !grow(&mut p, want) {
                    p.dead = true;
                    return None;
                }
            }
            let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
                p.dead = true;
                return None;
            };
            let ev = p.evs[p.used];
            if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
                p.dead = true;
                return None;
            }
            let i = p.used as u32;
            p.used += 1;
            Some(i)
        })
    }

    /// 区間の入口イベント。時系列が off なら `None` (分岐 1 回)。
    pub fn span_begin() -> Option<u32> {
        if !span_on() {
            return None;
        }
        mark()
    }

    /// イベントを打つ窓が開いているか (decode の時系列 or prefill のホスト台帳)。
    #[inline]
    fn span_on() -> bool {
        crate::model::q4ledger::tline_active() || crate::model::q4ledger::pf_span_active()
    }

    /// task 254: prefill のホスト台帳の窓を開ける (`q4ledger::total` が 1 回)。
    ///
    /// decode の [`prepare`] と同じ用意をするが、窓のフラグは q4ledger 側が持つ。
    pub fn prepare_prefill() {
        if !crate::model::q4ledger::pf_span_active() {
            return;
        }
        if let Some(ctx) = shared_ctx() {
            let _ = ctx.make_current();
        }
        // 1 本だけ打って捨てる (作れない構成なら以後の記録を止める)。
        let _ = mark();
        reset();
    }

    /// 窓の入口で 1 回だけ呼ぶ。イベントは**このスレッドの** current context に
    /// 対して打つので、リクエストのスレッドで context を張っておく
    /// (engine の GPU 入口は毎回 `make_current` するが、最初の `enter` はその前)。
    pub fn prepare() {
        if !crate::model::q4ledger::tline_active() {
            return;
        }
        if let Some(ctx) = shared_ctx() {
            let _ = ctx.make_current();
        }
        // 1 本だけ打って捨てる (作れない構成なら以後の記録を止める)。
        let _ = mark();
        reset();
    }

    /// 区間の出口イベントを打って span を積む。
    pub fn span_end(item: usize, a: u32) {
        if let Some(b) = mark() {
            P.with(|c| c.borrow_mut().spans.push((item as u8, a, b)));
        }
    }

    /// 起動の直前イベント (K ゲートのときだけ)。
    pub fn launch_begin() -> Option<u32> {
        if !k_on() || !span_on() {
            return None;
        }
        mark()
    }

    /// 起動の直後イベントを打って (item, 前, 後) を積む。
    pub fn launch_end(item: usize, a: u32) {
        if let Some(b) = mark() {
            P.with(|c| c.borrow_mut().launch.push((item as u8, a, b)));
        }
    }

    pub(crate) fn elapsed(a: u32, b: u32) -> Option<f32> {
        let f = lib().ok().and_then(|l| l.event_elapsed_time)?;
        P.with(|c| {
            let p = c.borrow();
            let ea = *p.evs.get(a as usize)?;
            let eb = *p.evs.get(b as usize)?;
            let mut ms = 0.0f32;
            if unsafe { f(&mut ms, ea, eb) } != CUDA_SUCCESS || !ms.is_finite() || ms < 0.0 {
                return None;
            }
            Some(ms)
        })
    }

    /// 区間ごとの GPU 時間を読む (区間の数だけ返す)。1 本も打っていなければ `None`。
    pub fn items(n: usize) -> Option<Vec<Item>> {
        P.with(|c| {
            let p = c.borrow();
            if p.spans.is_empty() && p.launch.is_empty() {
                return None;
            }
            let mut out = vec![Item::default(); n];
            for &(item, a, b) in p.spans.iter() {
                let i = (item as usize).min(n.saturating_sub(1));
                if let Some(ms) = elapsed(a, b) {
                    out[i].span_ms += ms as f64;
                    out[i].spans += 1;
                }
            }
            for &(item, a, b) in p.launch.iter() {
                let i = (item as usize).min(n.saturating_sub(1));
                if let Some(ms) = elapsed(a, b) {
                    out[i].busy_ms += ms as f64;
                    out[i].launches += 1;
                }
            }
            Some(out)
        })
    }

    /// task 254: 区間ごとの **GPU 実時間** (起動ペアの差の和、ms)。
    ///
    /// `span` と違ってアイドル (ホスト待ちで GPU が空いていた時間) を含まない。
    /// 1 本も打っていなければ `None`。
    pub fn item_busy_ms(n: usize) -> Option<Vec<f64>> {
        let items = items(n)?;
        Some(items.into_iter().map(|it| it.busy_ms).collect())
    }

    /// 窓の記録を捨てる (イベント自体は次の窓で使い回す)。
    pub fn reset() {
        P.with(|c| {
            let mut p = c.borrow_mut();
            p.used = 0;
            p.spans.clear();
            p.launch.clear();
        });
    }
}


unsafe fn open_libcuda() -> Result<CudaLib> {
    let name = b"libcuda.so.1\0";
    let handle = dlopen(name.as_ptr() as *const c_char, RTLD_NOW);
    if handle.is_null() {
        bail!("libcuda.so.1 を dlopen できません (NVIDIA driver が無い/見えない)");
    }
    Ok(CudaLib {
        cu_init: std::mem::transmute(sym(handle, "cuInit")?),
        device_get: std::mem::transmute(sym(handle, "cuDeviceGet")?),
        device_name: std::mem::transmute(sym(handle, "cuDeviceGetName")?),
        device_cc: std::mem::transmute(sym(handle, "cuDeviceComputeCapability")?),
        primary_ctx_retain: std::mem::transmute(sym(handle, "cuDevicePrimaryCtxRetain")?),
        ctx_set_current: std::mem::transmute(sym(handle, "cuCtxSetCurrent")?),
        ctx_sync: std::mem::transmute(sym(handle, "cuCtxSynchronize")?),
        driver_version: std::mem::transmute(sym(handle, "cuDriverGetVersion")?),
        module_load_data: std::mem::transmute(sym(handle, "cuModuleLoadData")?),
        module_get_function: std::mem::transmute(sym(handle, "cuModuleGetFunction")?),
        mem_alloc: std::mem::transmute(sym(handle, "cuMemAlloc_v2")?),
        mem_free: std::mem::transmute(sym(handle, "cuMemFree_v2")?),
        mem_get_info: sym_opt(handle, "cuMemGetInfo_v2")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuMemGetInfo>(p) }),
        device_total_mem: sym_opt(handle, "cuDeviceTotalMem_v2")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuDeviceTotalMem>(p) }),
        memcpy_htod: {
            let p: FnCuMemcpyHtoDV2 = std::mem::transmute(sym(handle, "cuMemcpyHtoD_v2")?);
            REAL_HTOD.store(p as *mut c_void, Ordering::Relaxed);
            ledger_htod
        },
        memcpy_dtoh: {
            let p: FnCuMemcpyDtoHV2 = std::mem::transmute(sym(handle, "cuMemcpyDtoH_v2")?);
            REAL_DTOH.store(p as *mut c_void, Ordering::Relaxed);
            ledger_dtoh
        },
        memset_d32_async: std::mem::transmute(sym(handle, "cuMemsetD32Async")?),
        mem_host_alloc: std::mem::transmute(sym(handle, "cuMemHostAlloc")?),
        mem_host_free: std::mem::transmute(sym(handle, "cuMemFreeHost")?),
        mem_host_register: sym_opt(handle, "cuMemHostRegister_v2")
            .or_else(|| sym_opt(handle, "cuMemHostRegister"))
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuMemHostRegisterV2>(p) }),
        mem_host_unregister: sym_opt(handle, "cuMemHostUnregister")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuMemHostUnregister>(p) }),
        mem_host_get_device_ptr: sym_opt(handle, "cuMemHostGetDevicePointer_v2")
            .or_else(|| sym_opt(handle, "cuMemHostGetDevicePointer"))
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuMemHostGetDevicePointerV2>(p) }),
        memcpy_htod_async: {
            let p: FnCuMemcpyHtoDAsyncV2 =
                std::mem::transmute(sym(handle, "cuMemcpyHtoDAsync_v2")?);
            REAL_HTOD_ASYNC.store(p as *mut c_void, Ordering::Relaxed);
            ledger_htod_async
        },
        memcpy_dtoh_async: {
            let p: FnCuMemcpyDtoHAsyncV2 =
                std::mem::transmute(sym(handle, "cuMemcpyDtoHAsync_v2")?);
            REAL_DTOH_ASYNC.store(p as *mut c_void, Ordering::Relaxed);
            ledger_dtoh_async
        },
        memcpy_dtod_async: std::mem::transmute(sym(handle, "cuMemcpyDtoDAsync_v2")?),
        memcpy_dtod: std::mem::transmute(sym(handle, "cuMemcpyDtoD_v2")?),
        device_get_attribute: sym_opt(handle, "cuDeviceGetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuDeviceGetAttribute>(p) }),
        func_set_attribute: sym_opt(handle, "cuFuncSetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuFuncSetAttribute>(p) }),
        func_get_name: sym_opt(handle, "cuFuncGetName")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuFuncGetName>(p) }),
        func_get_attribute: sym_opt(handle, "cuFuncGetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuFuncGetAttribute>(p) }),
        stream_create: sym_opt(handle, "cuStreamCreate")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuStreamCreate>(p) }),
        // driver は v2 名で公開する (CUDA 3.2 以降)。無ければ無印を試す。
        stream_destroy: sym_opt(handle, "cuStreamDestroy_v2")
            .or_else(|| sym_opt(handle, "cuStreamDestroy"))
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuStreamDestroy>(p) }),
        stream_wait_event: sym_opt(handle, "cuStreamWaitEvent")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuStreamWaitEvent>(p) }),
        event_create: sym_opt(handle, "cuEventCreate")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuEventCreate>(p) }),
        event_destroy: sym_opt(handle, "cuEventDestroy_v2")
            .or_else(|| sym_opt(handle, "cuEventDestroy"))
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuEventDestroy>(p) }),
        event_record: sym_opt(handle, "cuEventRecord")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuEventRecord>(p) }),
        event_sync: sym_opt(handle, "cuEventSynchronize")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuEventSynchronize>(p) }),
        // s169 (P8): 位相計測用 (無い driver では計測だけが OFF になる)。
        event_elapsed_time: sym_opt(handle, "cuEventElapsedTime")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuEventElapsedTime>(p) }),
        launch: std::mem::transmute(sym(handle, "cuLaunchKernel")?),
        // task 1377: cuLaunchKernelEx は任意 (古い driver では従来経路)。
        launch_ex: sym_opt(handle, "cuLaunchKernelEx")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuLaunchKernelEx>(p) }),
    })
}


static LIB: OnceLock<CudaLib> = OnceLock::new();


pub(crate) fn lib() -> Result<&'static CudaLib> {
    if let Some(l) = LIB.get() {
        return Ok(l);
    }
    let l = unsafe { open_libcuda()? };
    let _ = LIB.set(l);
    Ok(LIB.get().expect("just set"))
}


/// デバイスの (空き, 総) を MiB で返す。`cuMemGetInfo_v2` が無ければ None。
/// s124 の診断用 (VRAM が戻らない件の切り分け)。
pub fn dev_mem_mb() -> Option<(usize, usize)> {
    let f = lib().ok()?.mem_get_info?;
    let (mut free, mut total) = (0usize, 0usize);
    if unsafe { f(&mut free, &mut total) } != CUDA_SUCCESS {
        return None;
    }
    Some((free >> 20, total >> 20))
}


/// task 818: デバイスの**総** VRAM を MiB で返す (context を作らない)。
///
/// [`dev_mem_mb`] の `cuMemGetInfo` は current context を要求するので、
/// モデルを組む前 (起動直後) に呼ぶと `CUDA_ERROR_INVALID_CONTEXT` で
/// 必ず `None` になる (lab t818a で実際に踏んだ)。こちらは
/// `cuInit` + `cuDeviceGet` + `cuDeviceTotalMem_v2` だけで引く。
/// ドライバが `cuDeviceTotalMem_v2` を持たない (CUDA 3.0 以前) なら `None`。
pub fn device_total_mem_mb() -> Option<usize> {
    let l = lib().ok()?;
    let f = l.device_total_mem?;
    if unsafe { (l.cu_init)(0) } != CUDA_SUCCESS {
        return None;
    }
    let idx: i32 = std::env::var("DENNO_CUDA_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut dev: CUdevice = 0;
    if unsafe { (l.device_get)(&mut dev, idx) } != CUDA_SUCCESS {
        return None;
    }
    let mut total: usize = 0;
    if unsafe { f(&mut total, dev) } != CUDA_SUCCESS {
        return None;
    }
    Some(total >> 20)
}


/// task 818: デバイスの compute capability (major, minor)。
///
/// [`device_total_mem_mb`] と同じく context を作らない。auto 計画の
/// ログに出すだけ (計画の式には入れない)。
pub fn device_sm() -> Option<(i32, i32)> {
    let l = lib().ok()?;
    if unsafe { (l.cu_init)(0) } != CUDA_SUCCESS {
        return None;
    }
    let idx: i32 = std::env::var("DENNO_CUDA_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut dev: CUdevice = 0;
    if unsafe { (l.device_get)(&mut dev, idx) } != CUDA_SUCCESS {
        return None;
    }
    let (mut maj, mut min) = (0i32, 0i32);
    if unsafe { (l.device_cc)(&mut maj, &mut min, dev) } != CUDA_SUCCESS {
        return None;
    }
    Some((maj, min))
}


// ---------------------------------------------------------------------------
// task 152: cuMemGetInfo のキャッシュ
// ---------------------------------------------------------------------------
// task 149 のカード予算 (`DENNO_VRAM_BUDGET_CARD`) は expert を 1 本載せるたびに
// cuMemGetInfo を呼んでいた。lab t149c では 28 token の prefill が +3.3% 悪化して
// おり、その主因と推定された (REPORT_P149.md)。ドライバによっては 1 回が
// 数十〜数百 us かかるので、**64 MiB ぶんの増減か 2 秒経過まで**は同じ値を返す。
//
// 予算は「余裕 (既定 1,024 MiB)」を残して切るので、最大 64 MiB の古さは
// 判定を変えない。確保に失敗したときは従来どおり追い出して再試行する
// (br149 の経路) ので、古い値で足りなくてもラッチしない。

/// キャッシュを捨てる「台帳の増減」の閾値 (MiB)。
pub const MEM_CACHE_DRIFT_MB: usize = 64;

/// キャッシュの最大 age (ミリ秒)。
pub const MEM_CACHE_MAX_MS: u128 = 2000;


pub(crate) struct MemCache {
    free_mb: usize,
    total_mb: usize,
    live_at: usize,
    at: std::time::Instant,
}


static MEM_CACHE: Mutex<Option<MemCache>> = Mutex::new(None);


/// キャッシュを捨てて次の [`dev_mem_mb_cached`] で引き直させる。
pub fn dev_mem_refresh() {
    if let Ok(mut g) = MEM_CACHE.lock() {
        *g = None;
    }
}


/// キャッシュを引き直すかの判定 (純関数)。
///
/// `live_delta` は前回の取得から動いた台帳の合計バイト (絶対値)、`age_ms` は
/// 前回の取得からの経過ミリ秒。どちらかが閾値を超えたら引き直す。
pub fn mem_cache_should_refresh(live_delta: usize, age_ms: u128) -> bool {
    live_delta >= MEM_CACHE_DRIFT_MB << 20 || age_ms >= MEM_CACHE_MAX_MS
}


/// [`dev_mem_mb`] のキャッシュ版 (task 152)。
///
/// 引き直すのは次のいずれか:
/// - まだ 1 度も引いていない / [`dev_mem_refresh`] の後
/// - 前回の取得から台帳の合計が [`MEM_CACHE_DRIFT_MB`] 以上動いた
/// - 前回の取得から [`MEM_CACHE_MAX_MS`] 以上経った
///
/// 引けなかったときは古い値ではなく `None` を返す (未取得のままなら `None`)。
pub fn dev_mem_mb_cached() -> Option<(usize, usize)> {
    let live = crate::mem_audit::live_bytes();
    let mut g = match MEM_CACHE.lock() {
        Ok(g) => g,
        Err(_) => return dev_mem_mb(),
    };
    let refresh = match g.as_ref() {
        None => true,
        Some(c) => mem_cache_should_refresh(
            live.abs_diff(c.live_at),
            c.at.elapsed().as_millis(),
        ),
    };
    if refresh {
        match dev_mem_mb() {
            Some((free, total)) => {
                *g = Some(MemCache {
                    free_mb: free,
                    total_mb: total,
                    live_at: live,
                    at: std::time::Instant::now(),
                });
            }
            None => {
                if g.is_none() {
                    return None;
                }
            }
        }
    }
    g.as_ref().map(|c| (c.free_mb, c.total_mb))
}


/// `DENNO_MEM_DBG=1` のときだけ確保/解放をログる (既定は完全に無効)。
pub(crate) fn mem_dbg() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DENNO_MEM_DBG").as_deref(), Ok("1")))
}


/// `DENNO_MEM_BT=1` のときだけ、確保の呼び出し元を 1 行に畳んで返す。
pub(crate) fn mem_bt() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DENNO_MEM_BT").as_deref(), Ok("1")))
}


pub(crate) fn bt_hint() -> String {
    let bt = std::backtrace::Backtrace::force_capture().to_string();
    let mut v: Vec<String> = Vec::new();
    for ln in bt.lines() {
        let ln = ln.trim();
        if ln.is_empty() || ln.starts_with("at ") || ln.starts_with("Backtrace") {
            continue;
        }
        // "3: denno_engine::cuda::foo" 形式のフレーム行だけ拾う
        let name = match ln.split_once(": ") {
            Some((_, r)) => r,
            None => continue,
        };
        if !name.starts_with("denno") && !name.contains("denno_engine") {
            continue;
        }
        v.push(name.to_string());
        if v.len() == 6 {
            break;
        }
    }
    v.join(" < ")
}


pub(crate) fn check(res: CUresult, what: &str) -> Result<()> {
    if res != CUDA_SUCCESS {
        bail!("CUDA {what} が失敗: error {res}");
    }
    Ok(())
}


// ---- task 174: DENNO_LAUNCH_DBG (診断専用・既定 off) ------------------------
//
// decode で `cuCtxSynchronize` が error 700 (illegal address) を返す位置を
// カーネル名で特定するための観測だけの仕掛け。on のときは**デバイス操作の
// たびに**名前を輪へ積んでから同期し、最初に失敗した操作の名前と直前の履歴を
// stderr に出す。数値・経路・起動形状は変えない (off は `on()` の分岐 1 回)。
pub(crate) mod launch_dbg {
    use super::*;
    use std::cell::RefCell;
    use std::sync::atomic::AtomicU64;

    /// 輪に残す操作数 (失敗時に直近 24 件を出す)。
    const RING: usize = 64;

    /// `DENNO_LAUNCH_DBG=1` のときだけ有効。
    pub(crate) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| {
            matches!(
                std::env::var("DENNO_LAUNCH_DBG").ok().as_deref().map(str::trim),
                Some("1")
            )
        })
    }

    thread_local! {
        static RING_BUF: RefCell<Vec<(u64, String)>> = const { RefCell::new(Vec::new()) };
    }

    static SEQ: AtomicU64 = AtomicU64::new(0);

    /// 操作 1 件を輪へ積む (off は None)。ラベルは on のときだけ組み立てる。
    #[inline]
    pub(crate) fn note(label: impl FnOnce() -> String) -> Option<u64> {
        if !on() {
            return None;
        }
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let l = label();
        RING_BUF.with(|b| {
            let mut b = b.borrow_mut();
            if b.len() >= RING {
                b.remove(0);
            }
            b.push((seq, l));
        });
        Some(seq)
    }

    /// 直前の操作を同期し、失敗したら履歴を出して Err を返す (off は何もしない)。
    pub(crate) fn sync(seq: Option<u64>) -> Result<()> {
        let Some(seq) = seq else {
            return Ok(());
        };
        let lib = lib()?;
        let res = unsafe { (lib.ctx_sync)() };
        if res == CUDA_SUCCESS {
            return Ok(());
        }
        let hist = RING_BUF.with(|b| b.borrow().clone());
        eprintln!("[launch-dbg] cuCtxSynchronize が error {res} (最後の操作 seq={seq})");
        eprintln!("[launch-dbg] 直前のデバイス操作 (古い順):");
        let start = hist.len().saturating_sub(24);
        for (s, l) in &hist[start..] {
            eprintln!("[launch-dbg]   seq={s} {l}");
        }
        bail!("CUDA cuCtxSynchronize が失敗: error {res} (launch-dbg: seq={seq})")
    }

    /// カーネル名 (`cuFuncGetName`。取れなければアドレスで代用)。
    pub(crate) fn fn_name(f: CUfunction) -> String {
        if let Ok(lib) = lib() {
            if let Some(get) = lib.func_get_name {
                let mut p: *const c_char = std::ptr::null();
                if unsafe { get(&mut p, f) } == CUDA_SUCCESS && !p.is_null() {
                    return unsafe { std::ffi::CStr::from_ptr(p) }
                        .to_string_lossy()
                        .into_owned();
                }
            }
        }
        format!("fn@{:x}", f as usize)
    }
}


// ---- task 177/179: decode 層の操作別・層別 平均内訳 (DENNO_GDN_DEC_KPROF・既定 off)
//
// 背景 (task 179): lab t177a の 1 層 1 サンプル表は pos=3827 の full で
// wall 557 us / gpu_sum 206 us / gap_sum 341 us を出し、そのうち 276 us が
// 「qkv GEMV の同期のあと、最初の KV ミラー追記の前」の 1 個の隙間だった。
// しかし [q4ledger] の full は 12 層で 3.58 ms/token = 層あたり 298 us なので、
// その 1 サンプルは inflated (初回・cold・イベントの overhead)。1 サンプルで
// 次を選ばないために、task 179 では **同じ帯の中で N 窓 (既定 64) の平均** を
// 層番号ごとに出し、ホスト仕事を名前付きで分離する。
//
// ここでは submission の各操作 (起動 / H2D / D2H / D2D / 明示同期 / 非同期コピー)
// の**直前と直後**に CUDA イベントを打ち、窓を閉じると集計へ畳む。列の意味:
//   wall_us   = 窓のホスト側経過 (層 1 回ぶん)
//   kern_us   = カーネルの GPU 時間の和
//   copy_us   = コピーの GPU 時間 (DMA) の和
//   gap_us    = 操作間の GPU 側の隙間の和 (ホスト仕事・待ちを含む)
//   wait_us   = 明示同期 (sync) のホスト滞在 = 待ち時間
//   launch_us = カーネル発行のホスト滞在 (driver 呼び出し)
//   steps     = 名前付きホスト区間の和 (rotary / norm / split / mirror など)
// gap は steps を**含む** (GPU が遊んでいる時間とホストが働いている時間は
// 同じ区間なので、足し算ではなく読み分けに使う)。
//
// 数値経路: off のとき各サイトが払うのは thread-local の `Cell<bool>` 読み 1 回。
// on でもカーネルの引数・起動形状・発行順・同期の位置は 1 つも変えず、イベントと
// 時刻だけを足す。読み出しは窓の最後 (呼び出し側がすでに同期済みの点) に 1 回。
//
// 出すのは `DENNO_GDN_DEC_KPROF=1` のときだけ。表は 3 つの位置帯
// (短い decode / 3,800..4,100 / 14,000..14,400 トークン) で、(帯, 種類) ごとに
// 1 回だけ出す。N は `DENNO_GDN_DEC_KPROF_N` (既定 64)。
// ---------------------------------------------------------------------------
pub(crate) mod dec_kprof {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::BTreeMap;

    /// 種類: GDN (ssm) 層の submission。
    pub(crate) const GDN: u8 = 1;
    /// 種類: hc mix 1 回 (attn 側 / ffn 側で同じ層番号に 2 回)。
    pub(crate) const MIX: u8 = 2;
    /// 種類: full attention 層。
    pub(crate) const FULL: u8 = 3;
    /// task 192: MoE の router 1 回 (gate_inp の GEMV → logits の D2H → 同期 →
    /// softmax / top-k)。`sync` 行の host が「何を待ったか」の実測になる。
    pub(crate) const ROUTER: u8 = 4;

    /// `DENNO_GDN_DEC_KPROF` が立っているか (既定 off)。
    pub(crate) fn on() -> bool {
        if let Some(v) = force_for_test() {
            return v;
        }
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| match std::env::var("DENNO_GDN_DEC_KPROF").ok() {
            None => false,
            Some(v) => {
                let v = v.trim().to_ascii_lowercase();
                !matches!(v.as_str(), "" | "0" | "off" | "false" | "no")
            }
        })
    }

    /// 平均する窓の数 (既定 64、2..=4096 の外は既定へ)。
    pub(crate) fn avg_n() -> u64 {
        static N: OnceLock<u64> = OnceLock::new();
        *N.get_or_init(|| match std::env::var("DENNO_GDN_DEC_KPROF_N").ok() {
            None => 64,
            Some(v) => match v.trim().parse::<u64>() {
                Ok(n) if (2..=4096).contains(&n) => n,
                _ => 64,
            },
        })
    }

    /// 表を出す位置帯 (0 = 短い decode、1 = 3.8k..4.1k、2 = 14.0k..14.4k)。
    /// 帯の外は `None`。**純関数** (テストで真理値表を固定する)。
    pub(crate) fn band(pos: u64) -> Option<u8> {
        if pos < 400 {
            Some(0)
        } else if (3_800..=4_100).contains(&pos) {
            Some(1)
        } else if (14_000..=14_400).contains(&pos) {
            Some(2)
        } else {
            None
        }
    }

    /// 帯の下端 (表の見出しに出す)。
    fn band_lo(band: u8) -> u64 {
        match band {
            0 => 0,
            1 => 3_800,
            _ => 14_000,
        }
    }

    /// 帯と種類の組を 1 bit にしたもの (出したかどうかの記憶に使う)。
    ///
    /// task 179: u8 だと帯 2 (bit 8..11) が入らず、`(8 + kind) & 7` に化けて
    /// **帯 0 と同じ bit** になっていた (lab t179a で 14k の表が出なかった原因)。
    pub(crate) fn mask(pos: u64, kind: u8) -> u16 {
        match band(pos) {
            Some(b) => 1u16 << ((b as u32) * 4 + kind as u32),
            None => 0,
        }
    }

    /// 計測を始めてよければ true。**純関数** (テストで真理値表を固定する)。
    ///
    /// `active` = すでに窓が開いている (入れ子は外側を優先)、`dead` = イベントを
    /// 作れなかった、`printed` = このプロセスで出し終えた (帯, 種類) のビット、
    /// `decode` = decode 窓の内側か (prefill の逐次 1 トークン経路で古い位置を
    /// 使って表を出すと、その帯の本番の表を食ってしまうので必須)。
    pub(crate) fn should_begin(
        active: bool,
        dead: bool,
        printed: u16,
        pos: u64,
        kind: u8,
        decode: bool,
    ) -> bool {
        if active || dead || !decode || band(pos).is_none() {
            return false;
        }
        (printed & mask(pos, kind)) == 0
    }

    /// 1 操作ぶんの記録。
    struct Entry {
        name: String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        bytes: usize,
        ev_in: u32,
        ev_out: u32,
        /// 前の操作のホスト戻りからこの操作のホスト入口まで (ns)。
        hin_ns: u64,
        /// この操作のホスト滞在 (ns)。
        host_ns: u64,
        t_host: Instant,
    }

    pub(crate) struct St {
        kind: u8,
        li: usize,
        site: u32,
        pos: u64,
        /// 層の重みの実バイト (qkv + out。実効帯域を出すため)。
        wbytes: usize,
        entries: Vec<Entry>,
        evs: Vec<CUevent>,
        used: usize,
        t0: Instant,
        prev_end: Instant,
        /// 名前付きホスト区間 (name, us)。窓の間に積む。
        steps: Vec<(String, f64)>,
        /// (帯, 種類) を出し終えたビット。
        printed: u16,
        dead: bool,
    }

    impl St {
        fn new() -> Self {
            let now = Instant::now();
            Self {
                kind: 0,
                li: 0,
                site: 0,
                pos: 0,
                wbytes: 0,
                entries: Vec::new(),
                evs: Vec::new(),
                used: 0,
                t0: now,
                prev_end: now,
                steps: Vec::new(),
                printed: 0,
                dead: false,
            }
        }
    }

    // ---- 集計 (N 窓の平均) -------------------------------------------------

    /// 操作 1 種類ぶんの累計。
    #[derive(Clone, Copy, Default)]
    struct OpAgg {
        calls: u64,
        gpu_us: f64,
        gap_us: f64,
        hin_us: f64,
        host_us: f64,
        bytes: u64,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
    }

    /// 1 レーン (層, 呼び出し位置) ぶんの累計。
    #[derive(Default)]
    struct LaneAgg {
        calls: u64,
        wall_us: f64,
        kern_us: f64,
        copy_us: f64,
        gap_us: f64,
        wait_us: f64,
        launch_us: f64,
        ops: BTreeMap<String, OpAgg>,
        steps: BTreeMap<String, (u64, f64)>,
    }

    /// (帯, 種類) ぶんの累計。
    #[derive(Default)]
    struct Group {
        n_win: u64,
        lanes: BTreeMap<(usize, u32), LaneAgg>,
    }

    thread_local! {
        static ST: RefCell<Option<St>> = const { RefCell::new(None) };
        /// ホットパス用の鏡 (`St.active` と同じ)。off の分岐はここ 1 回。
        static ACTIVE: Cell<bool> = const { Cell::new(false) };
        /// 直近に居た層 (mix は層番号を知らないので `run_layer` が置く)。
        static IL: Cell<usize> = const { Cell::new(0) };
        /// 直近の decode の絶対位置 (`Qwen4ExpDecode` が置く)。
        static POS: Cell<u64> = const { Cell::new(0) };
        /// テスト用の強制 (None = env の解釈)。
        static FORCE: Cell<Option<bool>> = const { Cell::new(None) };
        /// 集計 (帯, 種類) -> レーン別。
        static AGG: RefCell<Option<BTreeMap<(u8, u8), Group>>> = const { RefCell::new(None) };
        /// 種類ごとの直近の帯 (帯を出たときに未出力の表を流すため)。
        static LAST_BAND: RefCell<BTreeMap<u8, u8>> = const { RefCell::new(BTreeMap::new()) };
    }

    /// テストが env を立てずに on/off を強制する (panic しても戻すのは呼び出し側)。
    #[cfg(test)]
    pub(crate) fn set_force_for_test(v: Option<bool>) {
        FORCE.with(|c| c.set(v));
    }

    pub(crate) fn force_for_test() -> Option<bool> {
        #[cfg(test)]
        {
            return FORCE.with(|c| c.get());
        }
        #[cfg(not(test))]
        {
            None
        }
    }

    /// `run_layer` が層番号を置く (mix の表に li を出すため)。
    #[inline]
    pub(crate) fn set_layer(il: usize) {
        if on() {
            IL.with(|c| c.set(il));
        }
    }

    /// 直近の層番号。
    pub(crate) fn cur_layer() -> usize {
        IL.with(|c| c.get())
    }

    /// decode の絶対位置を置く (`Qwen4ExpDecode` の `step_inner` / `feed_one`)。
    #[inline]
    pub(crate) fn set_pos(pos: u64) {
        if on() {
            POS.with(|c| c.set(pos));
        }
    }

    /// 直近の位置。
    pub(crate) fn cur_pos() -> u64 {
        POS.with(|c| c.get())
    }

    /// 窓を開く (開けなければ何もしない)。戻り値の guard が drop で表を出す。
    ///
    /// `site` は同じ層で複数回呼ばれる経路の区別 (mix の attn 側 / ffn 側など)。
    pub(crate) fn window(kind: u8, li: usize, pos: u64, site: u32) -> Win {
        let decode = crate::model::qwen4exp::gdn_gpu_win::active();
        let want = on()
            && ACTIVE.with(|c| !c.get())
            && ST.with(|c| {
                let slot = c.borrow();
                match slot.as_ref() {
                    None => should_begin(false, false, 0, pos, kind, decode),
                    Some(s) => should_begin(false, s.dead, s.printed, pos, kind, decode),
                }
            });
        if !want {
            return Win { open: false };
        }
        // 帯をまたいだら、前の帯の (未出力の) 表を流す。
        if let Some(b) = band(pos) {
            let prev = LAST_BAND.with(|c| c.borrow_mut().insert(kind, b));
            if let Some(p) = prev {
                if p != b {
                    flush_band(p, kind, 8);
                }
            }
        }
        ST.with(|c| {
            let mut slot = c.borrow_mut();
            let s = slot.get_or_insert_with(St::new);
            s.kind = kind;
            s.li = li;
            s.site = site;
            s.pos = pos;
            s.wbytes = 0;
            s.entries.clear();
            s.steps.clear();
            s.used = 0;
            s.t0 = Instant::now();
            s.prev_end = s.t0;
        });
        ACTIVE.with(|c| c.set(true));
        Win { open: true }
    }

    /// 窓の guard (drop で表を出す)。開いていない guard は drop で何もしない
    /// (入れ子の内側が外側の窓を閉じてしまう取り違えを防ぐ)。
    pub(crate) struct Win {
        open: bool,
    }

    impl Drop for Win {
        fn drop(&mut self) {
            if self.open {
                finish();
            }
        }
    }

    /// 層の重みの実バイトを添える (表の `wbytes` 用)。
    pub(crate) fn note_wbytes(n: usize) {
        if !ACTIVE.with(|c| c.get()) {
            return;
        }
        ST.with(|c| {
            if let Some(s) = c.borrow_mut().as_mut() {
                s.wbytes = s.wbytes.saturating_add(n);
            }
        });
    }

    /// 名前付きホスト区間。窓が開いていなければ closure を呼ぶだけ。
    ///
    /// ここで測るのは**ホストの仕事そのもの** (rotary / norm / split / mirror の
    /// 帳簿 / 詰め替え)。GPU を待つ区間は入れない (それは `sync` の行に出る)。
    #[inline]
    pub(crate) fn step<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
        if !ACTIVE.with(|c| c.get()) {
            return f();
        }
        let t = Instant::now();
        let r = f();
        let us = t.elapsed().as_secs_f64() * 1e6;
        ST.with(|c| {
            if let Some(s) = c.borrow_mut().as_mut() {
                s.steps.push((name.to_string(), us));
            }
        });
        r
    }

    /// [`step`] の「区間の頭を先に取る」版 (途中に `?` や early return がある
    /// 区間を測る)。窓が開いていなければ何もしない。
    #[inline]
    pub(crate) fn note_since(name: &'static str, t: Instant) {
        if !ACTIVE.with(|c| c.get()) {
            return;
        }
        let us = t.elapsed().as_secs_f64() * 1e6;
        ST.with(|c| {
            if let Some(s) = c.borrow_mut().as_mut() {
                s.steps.push((name.to_string(), us));
            }
        });
    }

    /// イベント 1 本を既定ストリームに打ち、pool の添字を返す。
    pub(crate) fn record(s: &mut St) -> Option<u32> {
        if s.dead {
            return None;
        }
        if s.used >= s.evs.len() {
            let want = s.evs.len() + 16;
            let Some(create) = lib().ok().and_then(|l| l.event_create) else {
                s.dead = true;
                return None;
            };
            while s.evs.len() < want {
                let mut ev: CUevent = std::ptr::null_mut();
                // 0 = CU_EVENT_DEFAULT (計時有効)。
                if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                    s.dead = true;
                    return None;
                }
                s.evs.push(ev);
            }
        }
        let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
            s.dead = true;
            return None;
        };
        let ev = s.evs[s.used];
        if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
            s.dead = true;
            return None;
        }
        let i = s.used as u32;
        s.used += 1;
        Some(i)
    }

    /// 操作の入口 (窓が開いていなければ None = 分岐 1 回)。
    #[inline]
    pub(crate) fn op_begin(
        name: impl FnOnce() -> String,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        bytes: usize,
    ) -> Option<usize> {
        if !ACTIVE.with(|c| c.get()) {
            return None;
        }
        ST.with(|c| {
            let mut slot = c.borrow_mut();
            let s = slot.as_mut()?;
            let ev_in = record(s)?;
            let now = Instant::now();
            let hin = now.saturating_duration_since(s.prev_end).as_nanos() as u64;
            s.entries.push(Entry {
                name: name(),
                grid,
                block,
                shmem,
                bytes,
                ev_in,
                ev_out: ev_in,
                hin_ns: hin,
                host_ns: 0,
                t_host: now,
            });
            Some(s.entries.len() - 1)
        })
    }

    /// 操作の出口 (入口が None なら何もしない)。
    #[inline]
    pub(crate) fn op_end(idx: Option<usize>) {
        let Some(idx) = idx else {
            return;
        };
        ST.with(|c| {
            let mut slot = c.borrow_mut();
            let Some(s) = slot.as_mut() else {
                return;
            };
            let ev = record(s);
            let now = Instant::now();
            if let Some(e) = s.entries.get_mut(idx) {
                if let Some(ev) = ev {
                    e.ev_out = ev;
                }
                e.host_ns = now.saturating_duration_since(e.t_host).as_nanos() as u64;
            }
            s.prev_end = now;
        });
    }

    /// 2 イベント間の ms。
    pub(crate) fn elapsed(evs: &[CUevent], a: u32, b: u32) -> Option<f32> {
        let f = lib().ok().and_then(|l| l.event_elapsed_time)?;
        let ea = *evs.get(a as usize)?;
        let eb = *evs.get(b as usize)?;
        let mut ms = 0.0f32;
        if unsafe { f(&mut ms, ea, eb) } != CUDA_SUCCESS || !ms.is_finite() || ms < 0.0 {
            return None;
        }
        Some(ms)
    }

    /// 表 1 行分 (GPU から切り離した値。整形だけをテストできるようにする)。
    #[derive(Clone, Copy)]
    struct Row<'a> {
        name: &'a str,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        bytes: usize,
        gpu_us: f32,
        gap_us: f32,
        hin_us: f64,
        host_us: f64,
    }

    fn kind_name(kind: u8) -> &'static str {
        match kind {
            GDN => "gdn",
            MIX => "mix",
            FULL => "full",
            ROUTER => "router",
            _ => "?",
        }
    }

    /// コピーの行か (名前の先頭で見る)。
    fn is_copy(name: &str) -> bool {
        name.starts_with("H2D") || name.starts_with("D2H")
    }

    /// 窓を閉じる。開いていなければ何もしない。
    pub(crate) fn finish() {
        if !ACTIVE.with(|c| c.replace(false)) {
            return;
        }
        ST.with(|c| {
            let mut slot = c.borrow_mut();
            let Some(s) = slot.as_mut() else {
                return;
            };
            // 最後のイベントまで待つ。窓を閉じる点は呼び出し側がすでに同期して
            // いるので待ちは発生しない (未完了なら下の elapsed が miss になる)。
            if let (Some(done), Some(last)) = (
                lib().ok().and_then(|l| l.event_sync),
                s.evs.get(s.used.saturating_sub(1)),
            ) {
                unsafe { done(*last) };
            }
            let wall_us = s.t0.elapsed().as_secs_f64() * 1e6;
            if s.dead || s.entries.is_empty() {
                eprintln!(
                    "[dec-kprof-avg] {} li={} pos={} wall_us={:.1} ops={} ev=unavailable \
                     (cuEventCreate/Record が失敗)",
                    kind_name(s.kind),
                    s.li,
                    s.pos,
                    wall_us,
                    s.entries.len()
                );
                s.printed |= mask(s.pos, s.kind);
                s.entries.clear();
                s.steps.clear();
                s.used = 0;
                return;
            }
            let mut rows: Vec<Row> = Vec::with_capacity(s.entries.len());
            let mut prev_out: Option<u32> = None;
            for e in &s.entries {
                let gpu = elapsed(&s.evs, e.ev_in, e.ev_out).unwrap_or(f32::NAN);
                let gap = match prev_out {
                    Some(p) => elapsed(&s.evs, p, e.ev_in).unwrap_or(f32::NAN),
                    None => 0.0,
                };
                rows.push(Row {
                    name: &e.name,
                    grid: e.grid,
                    block: e.block,
                    shmem: e.shmem,
                    bytes: e.bytes,
                    gpu_us: gpu * 1e3,
                    gap_us: gap * 1e3,
                    hin_us: e.hin_ns as f64 / 1e3,
                    host_us: e.host_ns as f64 / 1e3,
                });
                prev_out = Some(e.ev_out);
            }
            if fold_window(s, wall_us, &rows) {
                s.printed |= mask(s.pos, s.kind);
            }
            s.entries.clear();
            s.steps.clear();
            s.used = 0;
        });
    }

    /// 1 窓を集計へ畳む。N 窓そろったら表を出して true。
    fn fold_window(s: &St, wall_us: f64, rows: &[Row<'_>]) -> bool {
        let Some(b) = band(s.pos) else {
            return false;
        };
        let target = avg_n();
        let ready = AGG.with(|c| {
            let mut slot = c.borrow_mut();
            let all = slot.get_or_insert_with(BTreeMap::new);
            let grp = all.entry((b, s.kind)).or_default();
            grp.n_win += 1;
            let lane = grp.lanes.entry((s.li, s.site)).or_default();
            lane.calls += 1;
            lane.wall_us += wall_us;
            for (i, r) in rows.iter().enumerate() {
                if i > 0 {
                    lane.gap_us += r.gap_us as f64;
                }
                if r.name == "sync" {
                    lane.wait_us += r.host_us;
                } else if r.name.starts_with("K ") {
                    lane.kern_us += r.gpu_us as f64;
                    lane.launch_us += r.host_us;
                } else if is_copy(r.name) {
                    lane.copy_us += r.gpu_us as f64;
                }
                let e = lane.ops.entry(r.name.to_string()).or_default();
                e.calls += 1;
                e.gpu_us += r.gpu_us as f64;
                e.gap_us += r.gap_us as f64;
                e.hin_us += r.hin_us;
                e.host_us += r.host_us;
                e.bytes += r.bytes as u64;
                e.grid = r.grid;
                e.block = r.block;
                e.shmem = r.shmem;
            }
            for (name, us) in &s.steps {
                let t = lane.steps.entry(name.clone()).or_insert((0, 0.0));
                t.0 += 1;
                t.1 += *us;
            }
            !grp.lanes.is_empty() && grp.lanes.values().all(|l| l.calls >= target)
        });
        if ready {
            AGG.with(|c| {
                let mut slot = c.borrow_mut();
                if let Some(all) = slot.as_mut() {
                    if let Some(grp) = all.remove(&(b, s.kind)) {
                        for line in render_group(b, s.kind, &grp) {
                            eprintln!("{line}");
                        }
                    }
                }
            });
        }
        ready
    }

    /// 帯を出たときに、まだ N 窓に届いていない表を流す (`calls >= min_calls`)。
    fn flush_band(b: u8, kind: u8, min_calls: u64) {
        let mut out: Vec<String> = Vec::new();
        AGG.with(|c| {
            let mut slot = c.borrow_mut();
            let Some(all) = slot.as_mut() else {
                return;
            };
            let Some(grp) = all.get(&(b, kind)) else {
                return;
            };
            let most = grp.lanes.values().map(|l| l.calls).max().unwrap_or(0);
            if most >= min_calls {
                if let Some(grp) = all.remove(&(b, kind)) {
                    out = render_group(b, kind, &grp);
                }
            } else {
                all.remove(&(b, kind));
            }
        });
        for line in out {
            eprintln!("{line}");
        }
    }

    /// 表を組み立てる (整形だけ。GPU に触らない = テストできる)。
    fn render_group(b: u8, kind: u8, grp: &Group) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "[dec-kprof-avg] {} band={} windows={} lanes={} (us/窓。窓 = 1 層 1 回)",
            kind_name(kind),
            band_lo(b),
            grp.n_win,
            grp.lanes.len()
        ));
        out.push(
            "[dec-kprof-avg]   li s calls   wall   kern   copy    gap   wait launch  steps"
                .to_string(),
        );
        for ((li, site), l) in &grp.lanes {
            let steps: f64 = l.steps.values().map(|(_, us)| *us).sum::<f64>();
            out.push(format!(
                "[dec-kprof-avg]   {:>2} {} {:>5} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1} {:>6.1}",
                li,
                site,
                l.calls,
                l.wall_us / l.calls as f64,
                l.kern_us / l.calls as f64,
                l.copy_us / l.calls as f64,
                l.gap_us / l.calls as f64,
                l.wait_us / l.calls as f64,
                l.launch_us / l.calls as f64,
                steps / l.calls as f64
            ));
        }
        // 操作別 (全レーン合計 / 窓数)。
        out.push("[dec-kprof-avg]   op (全レーン。us は 1 窓あたりの平均)".to_string());
        let mut ops: BTreeMap<&str, OpAgg> = BTreeMap::new();
        for l in grp.lanes.values() {
            for (name, e) in &l.ops {
                let t = ops.entry(name.as_str()).or_default();
                t.calls += e.calls;
                t.gpu_us += e.gpu_us;
                t.gap_us += e.gap_us;
                t.hin_us += e.hin_us;
                t.host_us += e.host_us;
                t.bytes += e.bytes;
                t.grid = e.grid;
                t.block = e.block;
                t.shmem = e.shmem;
            }
        }
        for (name, e) in &ops {
            let n = e.calls.max(1) as f64;
            out.push(format!(
                "[dec-kprof-avg]   op {} grid={} block={} shmem={} bytes={} n={} gpu={:.1} gap={:.1} hin={:.1} host={:.1}",
                name,
                grid_s(e.grid),
                block_s(e.block),
                e.shmem,
                e.bytes / (e.calls.max(1) / grp.lanes.len().max(1) as u64).max(1),
                e.calls,
                e.gpu_us / n,
                e.gap_us / n,
                e.hin_us / n,
                e.host_us / n
            ));
        }
        // 名前付きホスト区間。
        let mut steps: BTreeMap<&str, (u64, f64)> = BTreeMap::new();
        for l in grp.lanes.values() {
            for (name, (n, us)) in &l.steps {
                let t = steps.entry(name.as_str()).or_insert((0, 0.0));
                t.0 += n;
                t.1 += us;
            }
        }
        if !steps.is_empty() {
            out.push("[dec-kprof-avg]   host step (全レーン。us は 1 回あたり)".to_string());
            for (name, (n, us)) in &steps {
                out.push(format!(
                    "[dec-kprof-avg]   step {} n={} us={:.1} (総 {:.1} us)",
                    name,
                    n,
                    us / (*n).max(1) as f64,
                    us
                ));
            }
        }
        out
    }

    fn grid_s(g: (u32, u32, u32)) -> String {
        if g == (0, 0, 0) {
            "-".to_string()
        } else {
            format!("{}x{}x{}", g.0, g.1, g.2)
        }
    }

    fn block_s(b: (u32, u32, u32)) -> String {
        if b == (0, 0, 0) {
            "-".to_string()
        } else {
            format!("{}x{}x{}", b.0, b.1, b.2)
        }
    }

    #[cfg(test)]
    pub(crate) mod tests {
        use super::*;

        /// task 810: `DENNO_Q4_DETERMINISTIC=1` は GPU 側の「CPU とビット一致しない」
        /// 経路を止める。固定するのは 3 点:
        ///
        /// - `gemv_kernel_mode()` が V2 (1 行 = 1 スレッド・`i` 昇順) に固定される
        /// - warp 木和の高速 seg カーネル (`DENNO_SEG_K`) が無効
        /// - デバイス和を使う層一括 seg 経路 (`DENNO_SEG_LAYER`) が無効
        ///
        /// 前 2 つは「off にすると値が変わる」ことで発火を数える (seg_k の既定は on)。
        /// GPU は要らない (選択だけのテスト)。
        #[test]
        fn q4_deterministic_disables_non_bit_exact_gpu_paths() {
            use crate::model::moe_expert::{moe_test_lock, set_q4_deterministic_for_test};
            let _g = moe_test_lock();
            set_q4_deterministic_for_test(Some(false));
            let seg_k_off = crate::cuda::moe_dec::seg_k_enabled();
            let seg_layer_off = crate::model::expert_batch::seg_layer_enabled();
            set_q4_deterministic_for_test(Some(true));
            let seg_k_on = crate::cuda::moe_dec::seg_k_enabled();
            let seg_layer_on = crate::model::expert_batch::seg_layer_enabled();
            let mode = gemv_kernel_mode();
            set_q4_deterministic_for_test(None);
            assert!(
                matches!(mode, GemvKernelMode::V2),
                "決定論モードで V2 以外が選ばれた: {mode:?}"
            );
            // 既定 (env 未設定) はどちらも on なので、off になったことで発火が分かる。
            assert!(
                seg_k_off,
                "既定で seg_k が off (このテストの前提が崩れている)"
            );
            assert!(
                seg_layer_off,
                "既定で seg_layer が off (このテストの前提が崩れている)"
            );
            assert!(!seg_k_on, "決定論モードで seg_k が止まっていない");
            assert!(!seg_layer_on, "決定論モードで seg_layer が止まっていない");
        }

        /// env を見ずに on/off を固定する (assert が落ちても戻す)。
        struct Force(Option<bool>);
        impl Force {
            fn set(v: Option<bool>) -> Self {
                let prev = FORCE.with(|c| c.get());
                FORCE.with(|c| c.set(v));
                Force(prev)
            }
        }
        impl Drop for Force {
            fn drop(&mut self) {
                FORCE.with(|c| c.set(self.0));
            }
        }

        /// task 177: GDN decode GEMV の env 解釈。
        ///
        /// task 179 で既定を v4 に、task 184 (lab t184a) で v5 にした。v5 の形が
        /// 合わない単位は v4 へ落ちるので、既定は「v5 を試す」でよい。
        #[test]
        fn gdn_dec_gemv_parse_truth_table() {
            // task 496: 未設定・空・未知の値は既定 (dn8。rows は 8)。
            for raw in [None, Some(""), Some("  "), Some("x")] {
                assert_eq!(parse_gdn_dec_gemv(raw), GdnDecGemv::Dn8, "raw={raw:?}");
            }
            assert_eq!(gdn_dn6_rows_from_value(None), 2, "既定は 2 行");
            assert_eq!(gdn_dn6_rows_from_value(Some("dn6r4")), 4);
            assert_eq!(gdn_dn6_rows_from_value(Some("dn6r6")), 6);
            // dn6 の値は全部 Dn6 へ (rows だけが変わる) = dn8 の切り戻し。
            for raw in ["dn6", "d6", "dn6r1", "dn6r2", "dn6r4", "dn6r6"] {
                assert_eq!(parse_gdn_dec_gemv(Some(raw)), GdnDecGemv::Dn6, "raw={raw:?}");
            }
            // dn6 / dn8 のときも v5 の計画は生かす (形が合わない単位の受け皿)。
            assert!(gdn_dec_gemv_v5_on());
            // 明示 off は v5 以前の既定 = v4。
            for raw in [Some("0"), Some("off"), Some("false"), Some("no")] {
                assert_eq!(parse_gdn_dec_gemv(raw), GdnDecGemv::V4, "raw={raw:?}");
            }
            // 明示指定 (大小文字と前後空白は無視)。
            assert_eq!(parse_gdn_dec_gemv(Some("v2")), GdnDecGemv::V2);
            assert_eq!(parse_gdn_dec_gemv(Some(" V2 ")), GdnDecGemv::V2);
            assert_eq!(parse_gdn_dec_gemv(Some("V3")), GdnDecGemv::V3);
            assert_eq!(parse_gdn_dec_gemv(Some("fp16")), GdnDecGemv::Fp16);
            assert_eq!(parse_gdn_dec_gemv(Some("v4")), GdnDecGemv::V4);
            assert_eq!(parse_gdn_dec_gemv(Some("V4")), GdnDecGemv::V4);
            // task 184: v5 (ベクタロード + 複数アキュムレータ + 先読み)。
            assert_eq!(parse_gdn_dec_gemv(Some("v5")), GdnDecGemv::V5);
            assert_eq!(parse_gdn_dec_gemv(Some(" V5 ")), GdnDecGemv::V5);
            // Default のときも gemv_kernel_mode() 側の選択肢に収まる。
            assert!(matches!(
                gdn_dec_gemv_mode_eff(),
                GemvKernelMode::V2 | GemvKernelMode::V3 | GemvKernelMode::Fp16
            ));
            // task 187: head 用の V5 は明示指定のみ (既定の選択には出ない)。
            assert!(!matches!(gemv_kernel_mode(), GemvKernelMode::V5));
            // task 179: 既定 (env 未設定) は v4 を選ぶ (lab t177c の -1.8% を採る)。
            assert!(gdn_dec_gemv_v4_on());
        }

        /// task 496: dn8 のゲート (env の値 → 選択) と起動形 (純関数)。
        #[test]
        fn gdn_dn8_gate_and_launch_plan_truth_table() {
            // env の値: dn8 系は全部 Dn8 (`dn8` の rows は 4)。
            for raw in [
                "dn8", "d8", "dn8r2", "dn8r4", "dn8r6", "dn8r8", "d8r4", " DN8 ", "Dn8R4",
            ] {
                assert_eq!(parse_gdn_dec_gemv(Some(raw)), GdnDecGemv::Dn8, "raw={raw:?}");
            }
            assert_eq!(gdn_dn8_rows_from_value(None), 8, "既定は 8 行");
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8")), 8);
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8r2")), 2);
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8r6")), 6);
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8r8")), 8);
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8r12")), 12);
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8r16")), 16);
            assert_eq!(gdn_dn8_rows_from_value(Some("dn8ar8")), 8);
            // `dn8a*` だけ「np が 32 の倍数の形にも dn8」。
            assert!(gdn_dn8_all_from_value(None), "既定は out も dn8");
            assert!(!gdn_dn8_all_from_value(Some("dn8")));
            assert!(!gdn_dn8_all_from_value(Some("dn8r8")));
            assert!(gdn_dn8_all_from_value(Some("dn8a")));
            assert!(gdn_dn8_all_from_value(Some("dn8ar8")));
            assert!(gdn_dn8_all_from_value(Some(" D8A ")));
            // dn8 のときも dn6 は受け皿として生きている (np が 32 の倍数の単位)。
            assert!(dn6_alive_for(GdnDecGemv::Dn8));
            assert!(dn6_alive_for(GdnDecGemv::Dn6));
            assert!(!dn6_alive_for(GdnDecGemv::V5));
            assert!(dn8_on_for(GdnDecGemv::Dn8));
            assert!(!dn8_on_for(GdnDecGemv::Dn6));
            // 実形の GDN qkv (n_in 2,560 / n_out 16,384) は rows=4 で
            // items = 160 = 5 warp (埋まり 100%、dn6 は 40/64 = 62.5%)。
            let p = mix_dn8_plan(2560, 16384, 4).expect("実形 qkv の計画");
            assert_eq!((p.np, p.rows, p.block, p.grid), (40, 4, 160, 4096));
            assert_eq!(p.shmem, (40 * 65 + 160) * 4);
            // 同じ形でも rows=1 は dn6 と同じ埋まりなので受けない。
            assert_eq!(mix_dn8_plan(2560, 16384, 1), None);
            // rows=16 まで許す (items = 640 <= 1024、block 640、grid 1,024)。
            let p16 = mix_dn8_plan(2560, 16384, 16).expect("rows=16 の計画");
            assert_eq!((p16.rows, p16.block, p16.grid), (16, 640, 1024));
            assert_eq!(p16.shmem, (40 * 65 + 640) * 4);
            // np が 32 の倍数の形 (out = 6,144 -> np 96) は既定では dn6 のまま。
            assert_eq!(mix_dn8_plan(6144, 2560, 4), None);
            // `dn8a` (allow_full) なら out の形も dn8 (x を rows 本で使い回す)。
            let po = mix_dn8_plan_with(6144, 2560, 4, true).expect("out の dn8a 計画");
            assert_eq!((po.np, po.rows, po.block, po.grid), (96, 4, 384, 640));
            assert_eq!(po.shmem, (96 * 65 + 384) * 4);
            // allow_full でも items > 1024 / block > 640 は受けない
            // (t496b の d8a8 は block 768 で error 701 = LAUNCH_OUT_OF_RESOURCES)。
            assert_eq!(mix_dn8_plan_with(6144, 2560, 16, true), None, "items 1536");
            assert_eq!(mix_dn8_plan_with(6144, 2560, 8, true), None, "block 768");
            // block の上限だけに当たる形 (np 168 = 10,752 / 100、shmem は 46 KB で収まる)。
            assert_eq!(mix_dn8_plan(10752, 100, 4), None, "block 672");
            // 門番: n_in % 64 != 0 / n_out 0 / items > 1024 / shared > 48 KiB。
            assert_eq!(mix_dn8_plan(2592, 16384, 4), None);
            assert_eq!(mix_dn8_plan(2560, 0, 4), None);
            assert_eq!(mix_dn8_plan(12800, 100, 6), None, "items 1200");
            assert_eq!(mix_dn8_plan(12800, 100, 2), None, "shared 53,600 B");
        }

        /// task 504: dn7 を GDN の out / full 層へ広げるゲート (純関数) と起動形。
        #[test]
        fn dn7_dec_gate_and_plan_truth_table() {
            // GDN の out: 既定 (未設定) は dn7、`o7` の値も true。
            assert!(gdn_out_dn7_from_value(None), "既定は out も dn7");
            for raw in ["dn8ao7", "dn8o7", "d8o7", "d8ao7", " DN8AO7 "] {
                assert!(gdn_out_dn7_from_value(Some(raw)), "raw={raw:?}");
            }
            // 明示値はその値が決めるカーネルへ戻す (`dn8ar4` が厳密な切り戻し)。
            for raw in ["dn8ar4", "dn8", "dn8a", "dn8ar8", "dn6", ""] {
                assert!(!gdn_out_dn7_from_value(Some(raw)), "raw={raw:?}");
            }
            // `o7` の値も qkv は dn8 のまま (out の受け皿も dn8 を保つ)。
            for raw in ["dn8ao7", "dn8o7"] {
                assert_eq!(parse_gdn_dec_gemv(Some(raw)), GdnDecGemv::Dn8, "raw={raw:?}");
                assert!(gdn_dn8_all_from_value(Some(raw)), "raw={raw:?}");
            }
            // full 層: `dn7` / `d7` だけ true (既定は dn6 のまま)。
            assert!(!full_dec_dn7_from_value(None));
            assert!(full_dec_dn7_from_value(Some("dn7")));
            assert!(full_dec_dn7_from_value(Some(" D7 ")));
            for raw in ["dn6", "dn6r2", "v5", "1", "0", ""] {
                assert!(!full_dec_dn7_from_value(Some(raw)), "raw={raw:?}");
            }
            // head: 既定 (未設定) は dn7。`dn6` が切り戻し。
            assert!(head_dn7_from_value(None), "既定は head も dn7");
            assert!(head_dn7_from_value(Some("dn7")));
            assert!(head_dn7_from_value(Some(" D7 ")));
            for raw in ["dn6", "dn6r4", "v5", "1", ""] {
                assert!(!head_dn7_from_value(Some(raw)), "raw={raw:?}");
            }
            // 実形の起動形: GDN の out (6,144 -> 2,560) は 1 行 384 スレッド、
            // full の qkv (2,560 -> 6,144) は 1 行 160 スレッド。
            assert_eq!(mix_dn7_plan(6144, 2560), Some((1, 384)));
            assert_eq!(mix_dn7_plan(2560, 6144), Some((1, 160)));
            // 門番: np が 8 の倍数でない形は受けない (mix up の 320 -> 10,240)。
            assert_eq!(mix_dn7_plan(320, 10240), None);
        }

        #[test]
        fn gdn_dec_v4_lanes_parse_and_shape_rule() {
            // 既定 32、4..32 に丸める。
            assert_eq!(parse_gdn_dec_v4_lanes(None), 32);
            assert_eq!(parse_gdn_dec_v4_lanes(Some("")), 32);
            assert_eq!(parse_gdn_dec_v4_lanes(Some("x")), 32);
            assert_eq!(parse_gdn_dec_v4_lanes(Some("2")), 4);
            assert_eq!(parse_gdn_dec_v4_lanes(Some("16")), 16);
            assert_eq!(parse_gdn_dec_v4_lanes(Some("32")), 32);
            assert_eq!(parse_gdn_dec_v4_lanes(Some("64")), 32);
            // 形の規則: n_in % (lanes*32) == 0 を満たす最大の 2 の冪。
            // 実形 k=2560 (qkv / gate): 16 が上限 (2560/16 = 160 = 32*5)。
            assert_eq!(v4_lanes_for(2_560, 32), 16);
            assert_eq!(v4_lanes_for(2_560, 16), 16);
            assert_eq!(v4_lanes_for(2_560, 8), 8);
            assert_eq!(v4_lanes_for(2_560, 4), 4);
            // 実形 k=6144 (attn_out): 32 まで行ける。
            assert_eq!(v4_lanes_for(6_144, 32), 32);
            assert_eq!(v4_lanes_for(6_144, 16), 16);
            // 32 の倍数でない k は v3 に任せる (0)。
            assert_eq!(v4_lanes_for(2_560 + 16, 32), 0);
            // 4 未満しか無い形も v3 に任せる。
            assert_eq!(v4_lanes_for(64, 32), 0);
            // 2 の冪でない上限は切り下げる。
            assert_eq!(v4_lanes_for(6_144, 20), 16);
        }

        /// task 192: 4 つの新しいゲートの既定。q5k と full_dec_fast は「未設定 =
        /// 既定 on、明示的な偽値だけ off」。fold / upw は task 213 の lab t213b で
        /// 目標に届かなかったので **既定 off** (明示的な真値だけ on) に戻した。
        /// A/B と切り戻しの土台。
        ///
        /// task 216 も同じ規則 (`parse_mix_dec_upw` = 明示的な真値だけ on) を採る。
        /// int-auto では PTX 欠落で実挙動が off だったので、既定は動かない。
        #[test]
        fn task192_gates_default_on_and_explicit_off() {
            for raw in [None, Some(""), Some("  "), Some("1"), Some("on"), Some("x")] {
                assert!(parse_gdn_dec_gemv_q5k(raw), "q5k raw={raw:?}");
                assert!(dec_full_dev::parse_full_dec_fast(raw), "fast raw={raw:?}");
            }
            for raw in [Some("0"), Some("off"), Some("false"), Some("no")] {
                assert!(!parse_gdn_dec_gemv_q5k(raw), "q5k raw={raw:?}");
                assert!(!dec_full_dev::parse_full_dec_fast(raw), "fast raw={raw:?}");
            }
            // 大小文字と前後の空白は無視する。
            for raw in [Some("OFF"), Some(" Off "), Some("NO")] {
                assert!(!parse_gdn_dec_gemv_q5k(raw), "q5k raw={raw:?}");
                assert!(!dec_full_dev::parse_full_dec_fast(raw), "fast raw={raw:?}");
            }
            // task 221: fold は既定 on (lab t221c の `all` を採用)。明示的な偽値だけ off。
            for raw in [
                None,
                Some(""),
                Some("  "),
                Some("1"),
                Some("on"),
                Some("true"),
                Some("yes"),
                Some("x"),
            ] {
                assert!(parse_mix_dec_fold(raw), "fold raw={raw:?}");
            }
            for raw in [Some("0"), Some("off"), Some(" FALSE "), Some("no")] {
                assert!(!parse_mix_dec_fold(raw), "fold raw={raw:?}");
            }
            // task 213 / 216: legacy の `DENNO_MIX_DEC_UPW` は既定 off のまま
            // (task 221 が既定 on にしたのは別名の `DENNO_DEC_MIX_UPW`)。未設定・空・
            // 偽値は off、真値だけ on。
            for raw in [
                None,
                Some(""),
                Some("  "),
                Some("0"),
                Some("off"),
                Some("false"),
                Some("no"),
                Some("x"),
            ] {
                assert!(!parse_mix_dec_upw(raw), "upw raw={raw:?}");
            }
            for raw in [Some("1"), Some("on"), Some(" ON "), Some("true"), Some("yes")] {
                assert!(parse_mix_dec_upw(raw), "upw raw={raw:?}");
            }
            // task 192: router early は lab t192a で +0.0% だったので **既定 off**。
            for raw in [None, Some(""), Some("0"), Some("off"), Some("false"), Some("no"), Some("x")] {
                assert!(!parse_dec_router_early(raw), "early raw={raw:?}");
            }
            for raw in [Some("1"), Some("on"), Some(" ON "), Some("true")] {
                assert!(parse_dec_router_early(raw), "early raw={raw:?}");
            }
        }

        /// task 192: Q5_K の v5 はスーパーブロック単位で割る。
        /// 実形 k=2,560 はスーパーブロック 10 本 -> 2 レーン (v4 は 16 レーンで
        /// 1 レーン 160 要素 = スーパーブロックを跨いでいた)。
        #[test]
        fn q5k_v5_lanes_split_superblocks() {
            assert_eq!(v5_lanes_div(10, 8), 2);
            assert_eq!(v5_lanes_div(8, 8), 8);
            assert_eq!(v5_lanes_div(12, 8), 4);
            assert_eq!(v5_lanes_div(16, 8), 8);
            assert_eq!(v5_lanes_div(0, 8), 0);
            // Q8_0 (ブロック対) の実形は今までどおり。
            assert_eq!(v5_lanes_div(40, 8), 8);
            // Q5_K の実形 (k=2,560) で v4 との差が説明できること。
            assert_eq!(v4_lanes_for(2_560, 32), 16);
        }

        #[test]
        fn band_covers_the_three_measured_points() {
            assert_eq!(band(0), Some(0));
            assert_eq!(band(1), Some(0));
            assert_eq!(band(399), Some(0));
            assert_eq!(band(400), None);
            assert_eq!(band(2_000), None);
            assert_eq!(band(3_799), None);
            assert_eq!(band(3_800), Some(1));
            assert_eq!(band(3_965), Some(1));
            assert_eq!(band(4_100), Some(1));
            assert_eq!(band(4_101), None);
            assert_eq!(band(13_999), None);
            assert_eq!(band(14_000), Some(2));
            assert_eq!(band(14_159), Some(2));
            assert_eq!(band(14_400), Some(2));
            assert_eq!(band(14_401), None);
        }

        #[test]
        fn should_begin_truth_table() {
            // 窓が開いていれば入れ子は開かない。
            assert!(!should_begin(true, false, 0, 1, GDN, true));
            // イベントを作れなかったら以後は開かない (静かに no-op にしない)。
            assert!(!should_begin(false, true, 0, 1, GDN, true));
            // 帯の外は開かない。
            assert!(!should_begin(false, false, 0, 2_000, GDN, true));
            // decode 窓の外 (prefill の逐次 1 トークン) は開かない。
            assert!(!should_begin(false, false, 0, 1, GDN, false));
            // 帯の中なら開く。
            assert!(should_begin(false, false, 0, 1, GDN, true));
            assert!(should_begin(false, false, 0, 3_965, GDN, true));
            assert!(should_begin(false, false, 0, 14_159, FULL, true));
            // 同じ (帯, 種類) を出した後は開かない。
            let m = mask(3_965, GDN);
            assert!(!should_begin(false, false, m, 3_970, GDN, true));
            // 違う種類は開く。
            assert!(should_begin(false, false, m, 3_970, MIX, true));
            // 違う帯は開く。
            assert!(should_begin(false, false, m, 14_159, GDN, true));
            // mask は帯と種類で別のビット。
            assert_ne!(mask(1, GDN), mask(3_965, GDN));
            assert_ne!(mask(1, GDN), mask(1, MIX));
            assert_ne!(mask(3_965, GDN), mask(14_159, GDN));
            // task 179 の回帰: 帯 0 を出した後でも帯 2 は開く
            // (u8 の mask では (8 + kind) & 7 に化けて帯 0 と同じ bit になり、
            //  14k の表が 1 度も出なかった)。
            let m0 = mask(1, FULL);
            assert!(should_begin(false, false, m0, 14_159, FULL, true));
            assert!(!should_begin(false, false, m0, 1, FULL, true));
            // 3 帯 x 3 種類のビットが全て異なる。
            let mut bits = std::collections::BTreeSet::new();
            for pos in [1u64, 3_965, 14_159] {
                for kind in [GDN, MIX, FULL] {
                    assert!(bits.insert(mask(pos, kind)), "bit が衝突: pos={pos} kind={kind}");
                }
            }
        }

        /// コピーの判定 (表の列分けの規則)。
        #[test]
        fn copy_rows_are_classified_by_prefix() {
            for n in ["H2D", "H2D@", "H2D@a", "D2H", "D2H@a"] {
                assert!(is_copy(n), "{n}");
            }
            for n in ["K dno_gemv_q8_0_v3", "sync", "D2D", "step"] {
                assert!(!is_copy(n), "{n}");
            }
        }

        /// 平均の組み立て: 2 レーン x 2 窓の合成データで、列の意味
        /// (kern/copy/gap/wait/launch/steps) と平均値が合うこと。
        #[test]
        fn render_averages_the_columns() {
            let mut grp = Group::default();
            grp.n_win = 2;
            for (li, wall, kern) in [(3usize, 300.0f64, 200.0f64), (7, 200.0, 150.0)] {
                let l = grp.lanes.entry((li, 0)).or_default();
                l.calls = 2;
                l.wall_us = wall * 2.0;
                l.kern_us = kern * 2.0;
                l.copy_us = 20.0 * 2.0;
                l.gap_us = 30.0 * 2.0;
                l.wait_us = 50.0 * 2.0;
                l.launch_us = 4.0 * 2.0;
                l.steps
                    .insert("rope_q".to_string(), (2, 12.0 * 2.0));
                let mut e = OpAgg::default();
                e.calls = 2;
                e.gpu_us = kern * 2.0;
                e.host_us = 4.0 * 2.0;
                e.grid = (320, 1, 1);
                e.block = (32, 1, 1);
                l.ops.insert("K dno_gemv_q8_0_v3".to_string(), e);
            }
            let lines = render_group(1, FULL, &grp);
            let all = lines.join("\n");
            assert!(all.contains("full band=3800 windows=2 lanes=2"), "{all}");
            assert!(all.contains("li s calls"), "{all}");
            // li=3: wall 300 / kern 200 / copy 20 / gap 30 / wait 50 / launch 4 / steps 12
            assert!(
                all.contains("3 0     2  300.0  200.0   20.0   30.0   50.0    4.0   12.0"),
                "{all}"
            );
            assert!(all.contains("7 0     2  200.0  150.0"), "{all}");
            // 操作別: 2 レーン x 2 窓 = 4 回。gpu は 1 窓あたりに戻す。
            assert!(all.contains("op K dno_gemv_q8_0_v3"), "{all}");
            assert!(all.contains("n=4"), "{all}");
            assert!(all.contains("gpu=175.0"), "{all}");
            // step は 1 回あたり。
            assert!(all.contains("step rope_q n=4 us=12.0"), "{all}");
        }

        /// off のときは窓が開かず、`op_begin` が 1 件も積まないこと
        /// (既定 off が「静かに計測する」に化けていないことの確認)。
        #[test]
        fn off_opens_no_window_and_records_nothing() {
            let _f = Force::set(Some(false));
            let _w = window(GDN, 7, 3_965, 0);
            assert_eq!(
                op_begin(|| "x".to_string(), (1, 1, 1), (1, 1, 1), 0, 0),
                None
            );
            let n = ST.with(|c| c.borrow().as_ref().map(|s| s.entries.len()));
            assert!(n.unwrap_or(0) == 0, "off で記録が積まれた");
        }

        /// 入れ子の guard が外側の窓を閉じないこと (task 179 で直した取り違え)。
        #[test]
        fn nested_window_guard_does_not_close_the_outer_one() {
            let _f = Force::set(Some(true));
            // decode 窓を開く (window は decode の内側でしか開かない)。
            crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(true);
            // このコンテナには GPU が無いので、窓は開いてもすぐ dead になる。
            // それでも guard の所有権 (open) は外側だけが持つ。
            let outer = window(FULL, 3, 3_965, 0);
            assert!(ACTIVE.with(|c| c.get()), "外側の窓が開かない");
            let inner = window(MIX, 0, 3_965, 0);
            assert!(ACTIVE.with(|c| c.get()));
            drop(inner);
            assert!(ACTIVE.with(|c| c.get()), "内側の drop で外側が閉じた");
            drop(outer);
            assert!(!ACTIVE.with(|c| c.get()), "外側の drop で閉じない");
            crate::model::qwen4exp::gdn_gpu_win::set_window_for_test(false);
        }

        /// GPU が無い環境ではイベントを作れない = dead になり、2 回目以降は
        /// 窓を開かない (無限に試さない)。
        #[test]
        fn gpu_less_host_marks_dead_after_the_first_attempt() {
            let _f = Force::set(Some(true));
            let _w = window(GDN, 0, 3_965, 0);
            // イベントが作れない環境 (このコンテナ) では dead が立つ。
            let dead = ST.with(|c| c.borrow().as_ref().map(|s| s.dead))
                .unwrap_or(false);
            if dead {
                assert!(!should_begin(false, true, 0, 3_970, GDN, true));
            }
        }
    }
}


/// task 210: 確保に失敗したときに expert を追い出す量の階段 (バイト)。
///
/// 1 段目は「要求 + 64 MiB」(driver の内部確保ぶんの余裕)、以降は
/// 512 MiB / 1 GiB / 2 GiB、最後は全部 (`usize::MAX`)。
///
/// 以前は 2 段 (要求 + 64 MiB / 全部) だった。slab はチャンクが空にならないと
/// 実メモリを返さないので小さい段では戻らないことがあり、そのまま全部まで
/// 進んで 22 GB の expert キャッシュを捨てていた。段を刻めば、戻った時点で
/// 止まる (lab t209a の 14k は prefill の頭で 32 MiB が 1 本通らないだけで
/// 全部捨てられ、decode が 21 → 35 ms/tok になった)。
///
/// 単調増加で、最後は必ず `usize::MAX`。
pub fn alloc_retry_wants(bytes: usize) -> Vec<usize> {
    let mut out = vec![bytes.saturating_add(64 << 20)];
    for step in [512usize << 20, 1 << 30, 2 << 30] {
        if step > *out.last().expect("先頭は必ずある") {
            out.push(step);
        }
    }
    out.push(usize::MAX);
    out
}


/// task 248: `DENNO_ALLOC_KEEP_EXPERTS` の解釈 (純関数)。
///
/// 未設定・読めない値は既定 (on)。明示的な偽値だけ off。
pub fn parse_keep_experts(v: Option<&str>, default_on: bool) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) if matches!(s.as_str(), "1" | "true" | "on" | "yes") => true,
        Some(s) if matches!(s.as_str(), "0" | "false" | "off" | "no") => false,
        _ => default_on,
    }
}


/// task 248: 既定 (**on**)。
///
/// 1 本の確保のために expert の LRU を丸ごと捨てる経路を塞ぐ
/// (REPORT_P242 の 14k 層優先: 1,505 MiB の連続確保が通らず 39,755 MiB を全捨て)。
/// `DENNO_ALLOC_KEEP_EXPERTS=0` で従来の階段 (task 210) に戻せる。
///
/// lab t248c (`4cc5129`) の実測: base (on) と `DENNO_ALLOC_KEEP_EXPERTS=0` は
/// 14k cold prefill 25.13 / 25.15 s、3,826 7.08 / 7.08 s、14k decode 18.4 /
/// 18.4 ms/tok、md5 も全リクエストで SAME。**正常経路では 1 回も発火しない**
/// (task248 の警告 0 回)。発火するのは確保に失敗したときだけで、そのときは
/// 追い出しが「要求 + チャンク 1 本」で止まり、足りなければ Err (= 呼び出し側が
/// CPU 経路へ) になる。
///
/// task 256 で追い出しの中身をチャンク単位にした (lab t256a、`c57c92c`)。
/// base は task256 の記録 0 行 (発火せず)、md5 SAME・decode 19.5/19.6/19.4
/// ms/tok・prefill 26.14/26.25/25.21 s (14k cold)。強制 COND
/// (`DENNO_EXPERT_VRAM_MB=24000` + `DENNO_PREFILL_WORK_MB=1500`) は空き 15 MiB
/// まで使い切っても errors 0・テキスト sane で、1 イベントの解放は
/// 要求 255 MiB に対して 255 MiB (上限は要求 + チャンク = 511 MiB)。
/// ゲートの既定は on のまま (採否 yes)。`REPORT_P256.md`。
pub const ALLOC_KEEP_EXPERTS_DEFAULT: bool = true;


/// task 248: 確保失敗時に expert を全捨てしない方針か。
pub fn alloc_keep_experts() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_keep_experts(
            std::env::var("DENNO_ALLOC_KEEP_EXPERTS").ok().as_deref(),
            ALLOC_KEEP_EXPERTS_DEFAULT,
        )
    })
}


/// task 290: `DENNO_ALLOC_FORCE_CHUNK` — 確保が失敗したあと、`cuMemGetInfo` の
/// 空きが要求以上に見えても (断片化) 冷たいチャンクを 1 本空けて再試行する
/// (**既定 on**。`0` / `off` / `false` / `no` で task 256 の従来動作に戻る)。
///
/// lab t286c の UD-Q4_K_XL 14k は「空き 3 MiB で 3 MiB の確保が error 2」で
/// 落ちた。不足分が 0 なので `plan_eviction` が 1 本も選ばず、`emergency_free`
/// も 0 を返して諦めていた。失敗後の再試行だけは空きを見ずに 1 本空ける。
pub const ALLOC_FORCE_CHUNK_ENV: &str = "DENNO_ALLOC_FORCE_CHUNK";


/// `DENNO_ALLOC_FORCE_CHUNK` の解釈 (純関数)。未設定・読めない値は既定 (on)。
pub fn alloc_force_chunk_from_value(v: Option<&str>) -> bool {
    parse_keep_experts(v, true)
}


/// 失敗後に 1 チャンク必ず空けるか (プロセスで 1 回だけ env を読む)。
pub fn alloc_force_chunk() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        alloc_force_chunk_from_value(std::env::var(ALLOC_FORCE_CHUNK_ENV).ok().as_deref())
    })
}


/// task 256: 確保 1 本が失敗したときに expert を追い出してよい上限 (バイト)。
///
/// slab はチャンクが空にならないと実メモリを返さないので、上限はタグに
/// よらず **要求 + チャンク 1 本**。1 MiB の確保が 1 本通らないだけでも、
/// 実メモリを戻すには 1 チャンクは空ける必要がある (`DENNO_EXPERT_SLAB_MB`)。
/// 追い出し側 (`expert_slab::plan_eviction`) は不足分を超えたところで止まるので、
/// 実際の追い出しはこの上限未満に収まる。
///
/// `None` は「上限なし = 従来の階段」(gate off)。
pub fn evict_cap_bytes(need: usize, _tag: &'static str, chunk: usize, keep: bool) -> Option<usize> {
    if !keep {
        return None;
    }
    let chunk = chunk.max(1 << 20);
    Some(need.saturating_add(chunk))
}


// task 279: 直近の確保が expert を追い出したときの鍵 (借りた作業域を返す側で使う)。
//
// `DevBuffer::alloc_tagged` の追い出し経路が積む。**挙動は変えない** (控えるだけ)。
// プール (`pf_pool_acquire_capped`) は自分の確保の前後でこれを読み書きするので、
// 他の確保 (attention のミラーなど) が積んだ鍵は混ざらない。
thread_local! {
    pub(crate) static ALLOC_EVICTED: std::cell::RefCell<Vec<crate::model::expert_cache::ExpertKey>> =
        const { std::cell::RefCell::new(Vec::new()) };
}


/// task 279: 追い出した鍵を控える (`alloc_tagged` の追い出し経路から)。
pub(crate) fn note_alloc_evicted(keys: Vec<crate::model::expert_cache::ExpertKey>) {
    if keys.is_empty() {
        return;
    }
    ALLOC_EVICTED.with(|c| c.borrow_mut().extend(keys));
}


/// task 279: 控えた鍵を取り出す (取り出したら空にする)。
pub fn take_alloc_evicted() -> Vec<crate::model::expert_cache::ExpertKey> {
    ALLOC_EVICTED.with(|c| std::mem::take(&mut *c.borrow_mut()))
}


/// デバイスメモリバッファ (RAII: drop で cuMemFree)
pub struct DevBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    /// s165: false = 他のバッファの一部を指す**非所有**ビュー。drop で解放しない
    /// (所有者は元のバッファ)。重みレジストリの行範囲エイリアス専用。
    owned: bool,
    /// task 149: 台帳のタグ ([`crate::mem_audit`])。解放時に同じタグで戻す。
    tag: &'static str,
}


/// decode GEMV 1 本ぶんの発行計画 (カーネル・起動形状・行バイト)。
///
/// [`CudaCtx::moe_gemv_dev_with`] から切り出した**選択だけ**の部分で、
/// デバイスメモリには触らない。task 124 の重ね打ち経路
/// ([`shared_gemv_begin`]) と同じカーネル・同じ引数を組むために共有する。
/// task 307: 計画を控えておけるように Copy にした (共有 expert の層ごとキャッシュ)。
#[derive(Clone, Copy)]
pub(crate) struct MoeGemvPlan {
    f: CUfunction,
    /// 1 ブロックが受け持つ行数
    rows_per_block: usize,
    block_dim: u32,
    shmem: u32,
    lanes: usize,
    /// 1 行のバイト数
    rb: usize,
    grid: u32,
}


/// CUDA コンテキスト + int8 カーネルモジュール。
/// プロセスで1個作り、複数の線形層で共有する。
pub struct CudaCtx {
    device: CUdevice,
    name: String,
    cc: (c_int, c_int),
    driver_ver: c_int,
    #[allow(dead_code)]
    ctx: CUcontext,
    module: CUmodule,
    f_quant: CUfunction,
    f_mv: CUfunction,
    // P17i: prefill 用 tiled GEMM (dno_gemm_i8)
    f_gemm: CUfunction,
    // s123: Q4_0 ネイティブ (GGUF の 4 bit ブロックを再量子化せず積む)
    f_mv_q4: CUfunction,
    f_gemm_q4: CUfunction,
    /// s132 (Task B1): 複数行を 1 block で受ける Q4 GEMV。
    /// コミット済みの `kernels_int8.ptx` に含まれる場合のみ取得でき、無ければ
    /// None で従来の `dno_gemv_q4_0` (1 行 1 block) を使う。古い PTX でも起動を
    /// 壊さないための Option (s89 の `f_gdn_chunk_scan_hoist` と同じ扱い)。
    /// s133 以降は列ブロック版 (`dno_gemv_q4_0c4` など) があると GEMV 経路は
    /// そちらへ行くので、本番経路からは呼ばれない (ビット一致の参照とテスト用)。
    f_mv_q4_mr: Option<CUfunction>,
    /// s133 (Task B1, attempt 2): A の 1 回読みを C 列で使い回す Q4 GEMV。
    /// C = 2 / 4 / 8 の 3 エントリを PTX に入れてあり、[`Q4_GEMV_C`] で選ぶ。
    /// 同じ理由で Option (未再生成の PTX でも従来経路で起動できる)。
    f_mv_q4_c: Option<[CUfunction; 3]>,
    // P17d: 要素演算モジュール (swiglu / add+rmsnorm / expf probe)
    #[allow(dead_code)]
    module_elem: CUmodule,
    f_swiglu: CUfunction,
    // P17k: interleaved swiglu ([m, 2ff] 行ごと [gate|up] → [m, ff])
    f_swiglu_int: CUfunction,
    f_addnorm: CUfunction,
    #[allow(dead_code)]
    f_expf: CUfunction,
    /// task 640: f16 -> f32 の全 65536 パターン照合 probe。
    /// 未収録の elem PTX を読んだときは None (テストだけが skip する)。
    #[allow(dead_code)]
    f_f16probe: Option<CUfunction>,
    // P17g: decode attention (exactness 緩和・ユーザー承認)
    f_attn: CUfunction,
    // P17j: prefill attention (flash 風 online softmax・exactness 緩和・ユーザー承認)
    f_attn_pf: CUfunction,
    /// task 222: prefill attention v2 (スコアを bf16 mma・PV を 4x16 の
    /// レジスタタイル)。`DENNO_PF_ATTN_TC` が on かつ PTX があるときだけ使う。
    /// 無いときは null (従来の `f_attn_pf` へ fail closed)。
    f_attn_pf2: CUfunction,
    /// task 230: prefill attention v3 (PV も bf16 mma・head 優先のグリッド・
    /// 行最大は最初のタイルで 1 回)。`DENNO_PF_ATTN_V3` が on かつ PTX がある
    /// ときだけ使う。無いときは v2 / 従来へ fail closed。
    f_attn_pf3: CUfunction,
    /// task 135: t を分割する decode attention (`dno_attention_decode`) と、
    /// 部分和を畳む `dno_attention_decode_combine`。PTX に無ければ None で
    /// 従来の `dno_attention` 経路に落ちる (fail closed・古い PTX でも起動可)。
    f_attn_dec: Option<CUfunction>,
    f_attn_dec_cb: Option<CUfunction>,
    // s48 (C1-a): dno_attention_prefill の動的 smem 上限 (bytes)。opt-in 成功時は
    // デバイスの opt-in 上限・失敗時は 48KB 既定 (ATTN_PF_DYN_SMEM にも反映)。
    attn_pf_dyn_smem: usize,
    // P17h: qkv rope + bias (CPU rope_row とビット一致)
    f_rope: CUfunction,
    // P17k ステップ 5: prefill 用行並列 rope (qkv→rope→attention 連続化)
    f_rope_pf: CUfunction,
    // B3: b>1 バッチ decode 用 rope (q/k in-place・行ごと独立位置・CPU rope_row とビット一致)
    f_rope_b: CUfunction,
    // P19-A1: GDN 逐次 AR スキャン (prefill バッチ・GDN は bit 一致不要/決定論のみ)
    #[allow(dead_code)]
    module_gdn: CUmodule,
    f_gdn: CUfunction,
    // A1-r2: GDN chunked scan 3 カーネル (長 prefill 用・CPU 参照 gdn::scan_chunked_core)
    f_gdn_chunk_pre: CUfunction,
    /// s239 (P36): pass 1 の 4 レーン版 (DK=128 専用)。PTX に含まれる場合のみ取得し、
    /// `DENNO_GDN_LANES=4` のときだけ使う (既定は旧カーネル)。
    f_gdn_chunk_pre_l4: Option<CUfunction>,
    /// P55: pass 1 の状態配列 (W/M) を共有メモリへ置く版 (`dno_gdn_chunk_pre_sm`)。
    /// 必要 smem が 48 KiB 既定を超えるため、カーネルが PTX にあり、かつ
    /// cuFuncSetAttribute で opt-in できたときだけ Some。`DENNO_GDN_PRE_SM=1`
    /// のときだけ使う (既定 off)。数値は旧版とビット同一 (格納先だけが違う)。
    f_gdn_chunk_pre_sm: Option<CUfunction>,
    /// P55: 上のカーネルに設定できた動的 smem 上限 (bytes)。opt-in 失敗時は
    /// 48 KiB 既定 = dk=128 の必要量 129 KiB を満たさないのでゲートで落ちる。
    gdn_pre_sm_dyn_smem: usize,
    f_gdn_chunk_state: CUfunction,
    f_gdn_chunk_scan: CUfunction,
    /// s89 (P1-1): 状態グラフ常駐版の chunk scan (pass 2 を折りたたんだ 2 launch 版)。
    /// PTX はコミット済みの `kernels_gdn.ptx` に含まれる (再生成済み) 場合のみ
    /// 取得でき、無ければ None で従来の 3 launch 経路を使う。
    f_gdn_chunk_scan_hoist: Option<CUfunction>,
    /// task 128: qwen4exp (sigmoid 出力ゲート) の decode 用 scan
    /// (`dno_gdn_scan_sig`)。task 137 で gdn.cu に追加し `kernels_gdn.ptx` を
    /// 再生成したので、通常は Some になる。読めないとき (旧 PTX) は
    /// [`gdn_decode_submit`] が計測を積まずに CPU 経路へ落ちる (fail-closed)。
    f_gdn_scan_sig: Option<CUfunction>,
    // P-MOE-E1: Flash-Next の expert GEMV (Q8_0 / Q5_1 / Q4_K / Q5_K) を積んだ
    // モジュール。PTX の読み込みに失敗しても起動は壊さず `f_moe = None` にして、
    // 呼び出し側 (`moe_gemv`) が Ok(false) を返して CPU 参照へ落ちる。
    #[allow(dead_code)]
    module_moe: CUmodule,
    f_moe: Option<[CUfunction; 4]>,
    /// P-gemv2: x 先読み版の GEMV (ビット一致)。読めなければ v1 を使う。
    f_moe2: Option<[CUfunction; 4]>,
    /// P-PF-GB: prefill 用の量子 GEMM (Q8_0 / Q5_1 / Q4_K / Q5_K) を積んだ
    /// モジュール。数値契約は prefill のみ解除済み (`-fmad=true` で JIT)。
    /// 読み込みに失敗しても起動は壊さず `f_gemm_q = None` (CPU バッチへ落ちる)。
    #[allow(dead_code)]
    module_gemm_q: CUmodule,
    f_gemm_q: Option<[CUfunction; 4]>,
}


pub(crate) const PTX: &[u8] = include_bytes!("../../kernels/kernels_int8.ptx");

pub(crate) const PTX_ELEM: &[u8] = include_bytes!("../../kernels/kernels_elem.ptx");

// P-MOE-E1: Flash-Next の expert GEMV (Q8_0 / Q5_1 / Q4_K / Q5_K)。
// CPU 参照 (`model::moe_expert::expert_gemv`) とビット一致する。
pub(crate) const PTX_MOE: &[u8] = include_bytes!("../../kernels/kernels_moe.ptx");

/// P-gemv2: 同じ数値契約のまま x をレジスタへ先読みする版 (`kernels_moe_v2.cu`)。
/// v1 とビット一致する (加算順・丸め順は同一)。`DENNO_GEMV_V2=0` で v1 に戻る。
pub(crate) const PTX_MOE_V2: &[u8] = include_bytes!("../../kernels/kernels_moe_v2.ptx");

/// P-PF-GB: prefill 用の量子 GEMM。契約外 (FMA 融合あり) なので
/// `kernels_moe.ptx` (-fmad=false) とは別ファイルにしてある。
pub(crate) const PTX_GEMM_Q: &[u8] = include_bytes!("../../kernels/kernels_gemm.ptx");


thread_local! {
    pub(crate) static MOE_SCRATCH: std::cell::RefCell<MoeScratch> =
        std::cell::RefCell::new(MoeScratch { x: None, o: None, pin_x: None, pin_o: None });
}


/// テストが rows/block を強制するための上書き (0 で env の値に戻る)。
#[cfg(test)]
pub(crate) static V3_ROWS_FORCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


/// テストから rows/block を固定する (`0` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_v3_rows_for_test(rows: usize) {
    assert!(rows == 0 || matches!(rows, 8 | 16 | 32 | 64));
    V3_ROWS_FORCE.store(rows, std::sync::atomic::Ordering::Relaxed);
}


/// `DENNO_GEMV_V3_ROWS` の生値 → rows/block (純関数。真理値表はテストで固定)。
///
/// 既定は 8 (main の実測: rows 8 / 16 / 32 / 64 で decode 122.6 / 142.3 / 135.0 /
/// 132.3 ms/token。8 は blockDim = 8 x 4 = 32 = 1 warp で、n_out=2560 でも
/// grid が 320 block になり 170 SM を埋められるため)。
pub fn parse_v3_rows(raw: Option<&str>) -> usize {
    match raw.map(str::trim) {
        Some("8") => 8,
        Some("16") => 16,
        Some("32") => 32,
        Some("64") => 64,
        _ => 8,
    }
}


/// v3 カーネルの引数の数 (w, x, out, n_in, n_out, row_bytes, rows_per_block, lanes)。
/// 旧版 (task 66) は 6 本。数を取り違えたまま起動すると **黙って別の行を書く**ので、
/// PTX を読む前に `.param` を数えて弾く (`ptx_entry_params`)。
pub(crate) const V3_NARGS: usize = 8;


/// v3 のカーネル名 (PTX から取る順 = `ki` の順 = 量子型の順)。
pub(crate) const V3_KERNEL_NAMES: [&str; 4] = [
    "dno_gemv_q8_0_v3",
    "dno_gemv_q5_1_v3",
    "dno_gemv_q4k_v3",
    "dno_gemv_q5k_v3",
];


/// PTX テキストの entry `name` の引数の個数 (`.param` 行を数える)。
///
/// nvcc は引数名を `NAME_param_N` にするので、`.visible .entry NAME(` の次から
/// `)` の行までの `.param` 行を数えれば個数が分かる。GPU 無しで検証できる
/// (テスト `v3_ptx_param_count_matches_nvcc_output` がコミット済みの
/// `kernels_moe_v2.ptx` で 6 本を数えて、nvcc の実出力で成立することを固定する)。
pub(crate) fn ptx_entry_params(ptx: &str, name: &str) -> Option<usize> {
    let mut lines = ptx.lines();
    while let Some(l) = lines.next() {
        let head = l.trim_start();
        let after = match head.strip_prefix(".visible .entry ") {
            Some(rest) => rest,
            None => match head.strip_prefix(".entry ") {
                Some(rest) => rest,
                None => continue,
            },
        };
        if !after.starts_with(name) || !after[name.len()..].trim_start().starts_with('(') {
            continue;
        }
        let mut n = 0usize;
        for body in lines.by_ref() {
            let t = body.trim();
            if t.starts_with(')') {
                return Some(n);
            }
            if t.starts_with(".param ") {
                n += 1;
            }
        }
        return None;
    }
    None
}


/// v3 のカーネル一式。4 本そろって初めて `Some`。
pub(crate) struct V3Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: [CUfunction; 4],
}


pub(crate) static V3_KERNELS: OnceLock<Option<V3Kernels>> = OnceLock::new();


// ---------------------------------------------------------------------------
// task 177: v4 (レーン数可変) GEMV (`kernels_moe_v4.cu`)
//
// v3 は 1 行を 4 レーン (または 2) で固定して分けるので、同時に立つスレッド数が
// `n_out * 4` で頭打ちになる。lab t177a の実測では out 投影 (n=2,560 / k=6,144)
// が 320 warp = 1.9 warp/SM で 328 GB/s しか出ていなかった。v4 は lanes を
// カーネル引数で受け、8 / 16 / 32 でも回せる (データの読み方と dequant の
// 演算列は v3 と同一。変わるのはレーン分割と合計の段数だけ)。
//
// 使うのは今のところ GDN decode の submission だけ (`DENNO_GDN_DEC_GEMV=v4`)。
// PTX が無い / 形が合わないときは None を返し、呼び出し側は v3 へ落ちる
// (fail closed。CPU へは落とさない)。
// ---------------------------------------------------------------------------

/// v4 のカーネル名 (`ki` の順 = Q8_0 / Q5_1 / Q4_K / Q5_K)。
pub(crate) const V4_KERNEL_NAMES: [&str; 4] = [
    "dno_gemv_q8_0_v4",
    "dno_gemv_q5_1_v4",
    "dno_gemv_q4k_v4",
    "dno_gemv_q5k_v4",
];


/// v4 の引数の数 (v3 と同じ 8 本: w, x, out, n_in, n_out, row_bytes,
/// rows_per_block, lanes)。
pub(crate) const V4_NARGS: usize = 8;


/// v4 カーネル一式。4 本そろって初めて `Some`。
pub(crate) struct V4Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: [CUfunction; 4],
}


pub(crate) static V4_KERNELS: OnceLock<Option<V4Kernels>> = OnceLock::new();


// ---------------------------------------------------------------------------
// task 184: v5 (`kernels_moe_v5.cu`) — ベクタロード + 複数アキュムレータ +
// 次ブロック先読み。Q8_0 は 2 ブロック (68 B) を 17 ワード、Q4_K は 1 スーパー
// ブロック (144 B) を 9 uint4 で読む。
// ---------------------------------------------------------------------------

/// v5 のカーネル名 (`ki` の順 = Q8_0 / Q4_K)。
pub(crate) const V5_KERNEL_NAMES: [&str; 2] = ["dno_gemv_q8_0_v5", "dno_gemv_q4k_v5"];


/// v5 の引数の数 (v3 / v4 と同じ 8 本)。
pub(crate) const V5_NARGS: usize = 8;


pub(crate) struct V5Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: [CUfunction; 2],
}


pub(crate) static V5_KERNELS: OnceLock<Option<V5Kernels>> = OnceLock::new();


// ---------------------------------------------------------------------------
// task 192: v5 の Q5_K (`kernels_moe_v5_q5k.cu`)。
//
// Q5_K の qkv+gate 融合 GEMV (実形 k=2,560 / n=16,384) は v4 のままで 53.6 us
// だった (lab t184a の `[dec-kprof-avg]`)。v4 は 1 行を 16 レーンに切り、レーン
// 区間 (160 要素) がスーパーブロック (256 要素) より短いので**重なって読む**。
// v5 はスーパーブロック単位で割るのでその重複が無い (Q4_K で -39%)。
// Q8_0 / Q4_K の PTX とは別ファイルなので、モジュールも別に読む。
// ---------------------------------------------------------------------------

/// Q5_K v5 のカーネル名。
pub(crate) const V5_Q5K_KERNEL_NAME: &str = "dno_gemv_q5k_v5";


pub(crate) struct V5Q5KKernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


pub(crate) static V5_Q5K_KERNELS: OnceLock<Option<V5Q5KKernels>> = OnceLock::new();


/// task 281/285: v5 の lane 数上限 (`DENNO_GEMV_V5_LANES`。**既定 32**)。
///
/// v5 の lane 数は「1 レーンが受け持つ量子ブロック対の数が割り切れる」最大の 2 の冪で、
/// 長らく `v5_lanes_div(per_unit, 8)` に固定されていた。Q8_0 の k=6144
/// (`per_unit` = nb/2 = 96) は 8 レーン = block 64 = 640 warp しか立たず、
/// 16.7 MB/層 の読みが 560 GB/s で止まる (同じカーネルでも k=2560 の 31.5 MB は
/// 1,057 GB/s = 1 レーンの依存ロード連鎖が長いだけ)。上限を 16 / 32 にすると
/// 立つ warp が 2 倍 / 4 倍になり、1 レーンの連鎖は 12 -> 6 -> 3 対になる。
///
/// 丸めは変わる (1 レーン分の鎖の長さと lane 木の段数が変わる) が、値の差は
/// 丸めだけで出力は動かない (lab t281a / t285a / t285c で md5 SAME)。task 285 で
/// 既定を 8 -> 32 にした (t285c: 短い decode 18.0 -> 17.9 ms/tok、2 サンプルずつ
/// 揃って同方向、prefill は短い点 +0.8% 以内、14k の prefill は base2 と同値)。
/// 切り戻しは `DENNO_GEMV_V5_LANES=8`。
pub(crate) fn parse_v5_lanes(raw: Option<&str>) -> usize {
    match raw.map(str::trim) {
        Some("32") => 32,
        Some("16") => 16,
        Some("8") => 8,
        Some("4") => 4,
        Some("2") => 2,
        Some("1") => 1,
        _ => 32,
    }
}


// ---------------------------------------------------------------------------
// P-fp16 (task 108): fast mode の fp16 GEMV (`kernels_moe_fp16.cu`)
// ---------------------------------------------------------------------------

/// decode GEMV で使うカーネルの選択 (テストが 1 経路に固定するための口も兼ねる)。
///
/// - [`GemvKernelMode::V2`]: ビット一致契約の既定経路 (CPU と 1 bit も違わない)。
/// - [`GemvKernelMode::V3`]: レーン分割 (`DENNO_GEMV_V3=1`)。加算順が変わるが
///   dequant は f32 のまま (契約の外 = prefill のバッチ GEMM と同じ扱い)。
/// - [`GemvKernelMode::Fp16`]: fast mode (`DENNO_GEMV_FP16=1`)。fp16 dequant +
///   half2 FMA で **丸めも精度も変える** (2026-11-01 オペレーター承認の契約変更)。
///   品質基準と誤差しきい値 (行列全体の相対 2e-2) は REPORT_P108.md。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GemvKernelMode {
    V2,
    V3,
    Fp16,
    /// task 187: v5 (`kernels_moe_v5.cu`)。使えない形なら v2 へ落ちる。
    V5,
}


/// `DENNO_GEMV_FP16` の生値 → fast mode の on/off (純関数。真理値表はテストで固定)。
pub(crate) fn parse_gemv_fp16(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("on") | Some("true"))
}


/// dn6 を撃ってよい選択か (純関数。dn8 のときも true = 受け皿)。
pub(crate) fn dn6_alive_for(c: GdnDecGemv) -> bool {
    matches!(c, GdnDecGemv::Dn6 | GdnDecGemv::Dn8)
}


/// dn8 を撃ってよい選択か (純関数)。
pub(crate) fn dn8_on_for(c: GdnDecGemv) -> bool {
    matches!(c, GdnDecGemv::Dn8)
}


/// task 307: `DENNO_GGUF_GEMV_V5` の解釈 (純関数)。**既定 on** (task 307 で採否。
/// lab t307a/b で full 行の gpu 1.19 → 1.03 ms/token、md5 SAME)。
///
/// 明示的な偽値 (`0` / `off` / `false` / `no`) のときだけ切る。on のときは
/// dense の gguf GEMV ([`maybe_gpu_gguf_gemv_into`]) も v5 の計画を先に試す。
pub(crate) fn parse_gguf_gemv_v5(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 307: dense の gguf GEMV を v5 で回すか (env から 1 回だけ読む)。
///
/// 動機 (lab t299b の `[dec-kprof-avg]` full 表): full 層の q8_0 GEMV が
/// v5 (26 us) と v3 (49 us) に分かれていて、v3 側は**ホスト経路へ落ちた層**の
/// `GgufLinear::gemv` → `maybe_gpu_gguf_gemv_into` → `moe_gemv_dev` が選ぶ
/// v3 だった (`gemv_kernel_mode()` は v5 を知らない)。同じ重み・同じ形でも
/// 入口が違うだけで 1.9 倍の差が出る。
pub(crate) fn gguf_gemv_v5_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_gguf_gemv_v5(std::env::var("DENNO_GGUF_GEMV_V5").ok().as_deref()))
}


/// fp16 カーネルの引数の数 (v3 と同じ 8 本: w, x, out, n_in, n_out, row_bytes,
/// rows_per_block, lanes)。検査は v3 と同じ `ptx_entry_params` を使う。
pub(crate) const FP16_NARGS: usize = 8;


/// fp16 カーネル名 (`ki` の順 = Q8_0 / Q5_1 / Q4_K。Q5_K は対象外)。
pub(crate) const FP16_KERNEL_NAMES: [&str; 3] = [
    "dno_gemv_q8_0_fp16",
    "dno_gemv_q5_1_fp16",
    "dno_gemv_q4k_fp16",
];


/// fp16 カーネル一式。3 本そろって初めて `Some`。
pub(crate) struct Fp16Kernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: [CUfunction; 3],
}


pub(crate) static FP16_KERNELS: OnceLock<Option<Fp16Kernels>> = OnceLock::new();


/// スクラッチの確保サイズを決める純関数 (GPU 無しで検証できる)。
///
/// - `cur >= want` -> `None` (確保しない = 同じサイズの繰り返しで増えない)
/// - `cur <  want` -> `Some(want)` (そのサイズへ grow。0 バイトは確保しない)
///
/// 返す値は `want` に対して単調 (要求が増えれば返る値も増える)。
pub(crate) fn moe_scratch_alloc_bytes(cur: usize, want: usize) -> Option<usize> {
    if cur >= want {
        None
    } else {
        Some(want.max(1))
    }
}


/// task 788: 借用ビュー (プールの区画) を掴んだままにしない版。
///
/// `DENNO_PF_POOL` の区画は prefill の終端で `cuMemFree` される。スレッド
/// ローカルの作業域がそのビューを持ち越すと、次のリクエストは解放済みの
/// 領域へ書く (= CUDA error 700)。借用ビューは容量が足りていても必ず
/// 確保し直す (所有バッファだけを再利用する)。
pub(crate) fn moe_scratch_alloc_bytes_borrowed(
    cur: usize,
    want: usize,
    borrowed: bool,
) -> Option<usize> {
    if borrowed {
        Some(want.max(1))
    } else {
        moe_scratch_alloc_bytes(cur, want)
    }
}


/// task 788: 初回の PTX 読み込みの前に current context を用意する。
///
/// `cuModuleLoadData` は current context が無いと `CUDA_ERROR_INVALID_CONTEXT`
/// で失敗する。prefill のローダは失敗した `None` を `OnceLock` に焼き付ける
/// (二度と試さない) ので、冷えた起動直後の最初のリクエストで context が
/// 未初期化だと **サーバの寿命ぶん GPU prefill が死ぬ**。
/// context が用意できないときは false (= 呼び出し側は焼き付けずに諦める)。
pub(crate) fn ensure_current_for_ptx_load() -> bool {
    match shared_ctx() {
        Some(ctx) => ctx.make_current().is_ok(),
        None => false,
    }
}


/// `DENNO_MOE_GROUPED_GPU` の値の解釈 (純関数。env を読むのは 1 箇所だけ)。
pub fn moe_grouped_gpu_from_value(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("on") | Some("true"))
}


/// grouped expert GEMM を試すか (既定 off)。
///
/// `DENNO_MOE_GROUPED_GPU=1` / `on` / `true` のときだけ true。値はプロセスで
/// 1 回だけ読む (テストは [`moe_grouped_gpu_from_value`] を直接叩く)。
pub fn moe_grouped_gpu_enabled() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        moe_grouped_gpu_from_value(std::env::var("DENNO_MOE_GROUPED_GPU").ok().as_deref())
    })
}


pub(crate) static GROUPED_KERNELS: OnceLock<Option<GroupedKernels>> = OnceLock::new();


thread_local! {
    pub(crate) static GROUPED_SCRATCH: std::cell::RefCell<GroupedScratch> =
        std::cell::RefCell::new(GroupedScratch::new());
}


// ===========================================================================
// P43: 層一括 expert GEMV (decode) — `kernels_moe_batch.cu`
// ===========================================================================
//
// decode の expert FFN は 1 トークン = 480 expert x 3 行列 = 1,440 起動 +
// 1,440 同期だった (実測 382 ms/token の主因。CPU は 189 ms/token)。
// 1 層ぶん (top-k = 10 expert) を segment 表 + デバイスポインタ表でまとめ、
// **gate 1 + up 1 + down 1 = 3 起動 / 2 同期**にする。
//
// 数値: 1 行の累積は `kernels_moe.cu` と同じ「1 スレッドが i の昇順に f32 で
// 加算」。同じ expert・同じ入力なら per-expert 版とビット一致する。gate と up は
// 同じ同期で受け取り、ホストで `silu(g) * u` を作る (これも per-expert 版の
// `silu_inplace` + 積と同じ式・同じ順)。したがって寄与 h は 10 回起動版と
// 1 ビットも変わらない。
//
// PTX は起動時に 1 回だけファイルから読む (`DENNO_MOE_BATCH_PTX` で差し替え可)。
// 読めなければ `false` を返して呼び出し側は per-expert 経路へ落ちる (fail closed)。

/// カーネル (PTX を起動時に 1 回だけ読む)。
pub(crate) struct MoeBatchKernels {
    #[allow(dead_code)]
    module: CUmodule,
    /// Q8_0 / Q5_1 / Q4_K / Q5_K の 4 本
    fns: [CUfunction; 4],
}


pub(crate) static MOE_BATCH_KERNELS: OnceLock<Option<MoeBatchKernels>> = OnceLock::new();


/// カーネル名 (添字は [`moe_batch_slot`] の順)。
pub(crate) const MOE_BATCH_NAMES: [&[u8]; 4] = [
    b"dno_moe_gemv_seg_q8_0\0",
    b"dno_moe_gemv_seg_q5_1\0",
    b"dno_moe_gemv_seg_q4k\0",
    b"dno_moe_gemv_seg_q5k\0",
];


/// 1 層ぶんの行列 1 本 (segment 表と組み合わせて 1 起動になる)。
pub(crate) struct SegJob<'a> {
    ty: crate::kquant::GgmlType,
    /// segment ごとの重み先頭 (デバイスアドレス)。長さ = nseg。
    wptrs: &'a [u64],
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
    out: &'a mut [f32],
}


// ===========================================================================
// task 278: K 型 (Q4_K / Q5_K / Q5_1 / Q8_0) の層を **1 同期**で閉じる
// ===========================================================================
//
// # なぜ要るか
//
// P43 の層一括経路 (`moe_gemv_seg_pair` + `moe_gemv_seg_one`) は 1 層で
// 3 起動 / **2 同期**、その間にホストが
//
//   1. gate/up の結果 (2 x nseg x ff) を D2H
//   2. `silu(g) * u` をホストで計算
//   3. act (nseg x ff) を H2D
//   4. down の結果 (nseg x e) を D2H
//   5. 重み付き和をホストで計算
//
// をする。GPU はこのホスト仕事の間ずっと空く (実測 lab t268b: 1 層 0.84 ms の
// うち sync 待ち 0.48 ms)。さらに prologue の小さな H2D が 1 層 7 回ある。
//
// # 何をするか
//
// 1 層を **5 起動 / 1 待ち / 1 D2H (e 要素だけ)** にする:
//
//   prologue 1 回の H2D (ポインタ表 3 + segment 表 2 + 重み + 入力)
//   → gate → up → silu_mul → down → 重み付き和 → イベントで待つ → D2H(e)
//
// silu と和は `kernels_dec_epi.ptx` の既存カーネル (`dno_dec_silu_mul` /
// `dno_dec_exp_sum`) を使う。和は `acc += w[t] * h[t*e+j]` を t の昇順に
// 1 本のアキュムレータで回すので、**ミスの無い層では従来のホスト足し込みと
// 同じ演算列** (-fmad=false)。silu だけは `expf` の実装が違うので 1 ulp 級の差が
// 出る (速度優先。タスク 278 の規則で許容)。
//
// 待ちは `cuCtxSynchronize` ではなく **null ストリームに打ったイベント**で行う。
// 文脈同期は背景の上げ直し (非ブロッキングストリーム) まで待つため、1 層 2 回の
// 同期が「他人の転送の完了待ち」になっていた。イベントなら自分のカーネルだけを待つ。
//
// 失敗 (PTX 不在 / 型未対応 / 形が合わない) はすべて `Ok(false)` = 呼び出し側は
// 旧経路へ落ちる (fail closed)。

thread_local! {
    /// 層を 1 同期で閉じる経路の作業域 (grow-only)。
    pub(crate) static SEG_LAYER_SCRATCH: std::cell::RefCell<SegLayerScratch> =
        std::cell::RefCell::new(SegLayerScratch::new());
}


/// [`moe_seg_layer_sum`] の作業域。すべて grow-only。
pub(crate) struct SegLayerScratch {
    /// prologue を詰めた 1 本のデバイスバッファ
    stage: Option<DevBuffer>,
    /// 同じ内容のホスト側バッファ (Vec<u8> を使い回す)
    host: Vec<u8>,
    /// gate の出力 (nseg x ff)
    g: Option<DevBuffer>,
    /// up の出力 (nseg x ff)
    u: Option<DevBuffer>,
    /// act = silu(g) * u (nseg x ff)
    act: Option<DevBuffer>,
    /// down の出力 (nseg x e)
    h: Option<DevBuffer>,
    /// 重み付き和の出力 (e)
    out: Option<DevBuffer>,
}


/// task 278: K 系 GEMV の高速版 (`kernels_seg_k.ptx`、1 行 = 1 warp)。
///
/// 無ければ旧 `kernels_moe_batch.ptx` (1 行 = 1 スレッド、CPU とビット一致) で
/// 回す。**あるときは丸め順が変わる** (32 レーンの木) が、決定的ではある。
pub(crate) struct SegKKernels {
    #[allow(dead_code)]
    module: CUmodule,
    /// Q8_0 / Q5_1 / Q4_K / Q5_K ([`moe_batch_slot`] と同じ並び)
    fns: [CUfunction; 4],
    /// task 282: mix 用 (古い PTX には無いので任意)。Q8_0 専用。
    mix_r8: Option<CUfunction>,
    /// task 282: mix の down 用 (1 行 = 1 ブロック 256 スレッド)。Q8_0 専用。
    mix_rowblk: Option<CUfunction>,
    /// task 412 / 526: gate + up + silu を 1 起動で回す版 ([`SEG_K_GU_NAMES`] の順)。
    /// 古い PTX には無いので `None` (そのときは従来の 3 起動へ落ちる)。
    gu: [Option<CUfunction>; 4],
    /// task 534: 上の 8 バイトロード版 ([`SEG_K_GU8_NAMES`] の順)。K 型だけ。
    gu8: [Option<CUfunction>; 4],
    /// task 534: 1 行版の 8 バイトロード版 ([`SEG_K_K8_NAMES`] の順)。K 型だけ。
    k8: [Option<CUfunction>; 4],
    /// task 562: Q4_K の 16 バイトロード版 ([`SEG_K_GU16_NAMES`] の順)。gate/up 用。
    gu16: [Option<CUfunction>; 4],
    /// task 534: Q5_1 の 4 バイトロード版 ([`SEG_K_V4_NAMES`] の順)。down 用。
    v4: [Option<CUfunction>; 4],
}


pub(crate) static SEG_K_KERNELS: OnceLock<Option<SegKKernels>> = OnceLock::new();


/// カーネル名 (添字は [`moe_batch_slot`] の順に合わせる)。
pub(crate) const SEG_K_NAMES: [&[u8]; 4] = [
    b"dno_seg_gemv_k_q8_0\0",
    b"dno_seg_gemv_k_q5_1\0",
    b"dno_seg_gemv_k_q4k\0",
    b"dno_seg_gemv_k_q5k\0",
];


/// task 412 / 526: gate + up + silu を 1 起動で回す版の名前 ([`moe_batch_slot`] の順)。
pub(crate) const SEG_K_GU_NAMES: [&[u8]; 4] = [
    b"dno_seg_gemv_gu_q8_0\0",
    b"dno_seg_gemv_gu_q5_1\0",
    b"dno_seg_gemv_gu_q4k\0",
    b"dno_seg_gemv_gu_q5k\0",
];


/// task 534: 融合版の **8 バイトロード**版 ([`moe_batch_slot`] の順)。
/// **Q8_0 / Q5_1 は 1 要素 1 バイトが最小**なのでこの版は無い = `None`。
pub(crate) const SEG_K_GU8_NAMES: [Option<&[u8]>; 4] = [
    None,
    None,
    Some(b"dno_seg_gemv_gu8_q4k\0"),
    Some(b"dno_seg_gemv_gu8_q5k\0"),
];


/// task 534: 1 行版の **8 バイトロード**版 ([`moe_batch_slot`] の順)。同上。
pub(crate) const SEG_K_K8_NAMES: [Option<&[u8]>; 4] = [
    None,
    None,
    Some(b"dno_seg_gemv_k8_q4k\0"),
    Some(b"dno_seg_gemv_k8_q5k\0"),
];


/// task 562: **Q4_K の 16 バイトロード**版 ([`moe_batch_slot`] の順) のうち
/// gate+up 融合版。K 型のうち Q4_K だけ (UD-Q4_K_XL の gate/up が Q4_K)。
pub(crate) const SEG_K_GU16_NAMES: [Option<&[u8]>; 4] = [
    None,
    None,
    Some(b"dno_seg_gemv_gu16_q4k\0"),
    None,
];


/// task 534: **Q5_1 の 4 バイトロード**版 ([`moe_batch_slot`] の順)。
/// UD-Q4_K_XL の down は Q5_1 (`[pf-ktc] gate/up=Q4K down=Q5_1`) なので、
/// down の速い版はここに入る。1 warp = 8 ブロック = 256 要素。
pub(crate) const SEG_K_V4_NAMES: [Option<&[u8]>; 4] = [
    None,
    Some(b"dno_seg_gemv_v4_q5_1\0"),
    None,
    None,
];


/// task 412 / 526: `DENNO_SEG_GU_FUSE` — seg 経路 (K 型モデルの decode) の
/// gate / up / silu を **1 起動**に畳む (`kernels_seg_k.ptx` の
/// `dno_seg_gemv_gu_*`)。**task 534 から既定 on** (t534c/t534d で
/// down のベクトル版と組にして 2 ラウンド -5.9 / -6.2%。単独では P526 の -0.9%)。
///
/// gate と up は入力 `x` が同じで行数も同じなので、(1) 起動が 3 -> 1、
/// (2) 1 ブロックあたりの `x` の読みが 1 回、(3) `g` / `u` の作業域
/// (`nseg x ff`) を書かない。**行ごとの計算順序は `dno_seg_gemv_k_*` と同じ**
/// (ブロック昇順のアキュムレータ 1 本 + 同じ warp 木) なので、1 行の値は
/// 従来とビット一致する。
pub const SEG_GU_FUSE_ENV: &str = "DENNO_SEG_GU_FUSE";


/// [`SEG_GU_FUSE_ENV`] の解釈 (純関数)。**task 534 から既定 on**:
/// 未設定は有効、`0` / `off` / `false` / `no` だけが偽 (A/B と切り戻し用)。
///
/// P526 は単独で -0.9% (2 ラウンドとも 2% 未満) だったので既定 off のまま
/// 残していた。task 534 では down の Q5_1 ベクトル版と組にして
/// **2 ラウンド -5.9 / -6.2%** が出たので既定 on にした (t534c / t534d)。
pub fn seg_gu_fuse_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// [`SEG_GU_FUSE_ENV`] が有効か (プロセスで 1 回だけ env を読む)。
pub fn seg_gu_fuse_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| seg_gu_fuse_from_value(std::env::var(SEG_GU_FUSE_ENV).ok().as_deref()))
}


/// task 412 / 526: 融合カーネルの起動形。`kernels/kernels_seg_k.cu` の
/// `SK_GU_ROWS` と一致させること (テスト `seg_fuse_shape_matches_cu` が突き合わせる)。
pub(crate) const SEG_WARPS: u32 = 4;

/// 1 warp が受け持つ gate/up の行数 (`SK_GU_ROWS`)。
pub(crate) const SEG_GU_ROWS: u32 = 2;


/// task 534: `DENNO_SEG_GU8` — 融合版 (gate+up+silu) を **8 バイトロード**で回す
/// (`dno_seg_gemv_gu8_*`)。**既定 off**。
///
/// `dno_seg_gemv_gu_*` は `q[qoff + t]` を 8 回読むので 1 命令 1 バイト。
/// Q4_K / Q5_K は lane の 8 要素が連続 8 バイトなので 64 ビットロード 1 回で
/// 同じバイト列を取れる (読む順も足す順も同じ = **値はビット一致**)。
/// 融合 (`DENNO_SEG_GU_FUSE`) の経路でだけ効く (単独では何もしない)。
pub const SEG_GU8_ENV: &str = "DENNO_SEG_GU8";


/// [`SEG_GU8_ENV`] の解釈 (純関数)。`1` / `on` / `true` / `yes` だけが真。
pub fn seg_gu8_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


/// [`SEG_GU8_ENV`] が有効か (プロセスで 1 回だけ env を読む)。
pub fn seg_gu8_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| seg_gu8_from_value(std::env::var(SEG_GU8_ENV).ok().as_deref()))
}


/// task 562: `DENNO_SEG_GU16` — 融合版 (gate+up+silu) を **16 バイトロード**で
/// 回す (`dno_seg_gemv_gu16_q4k`)。**task 562 から既定 on**。
///
/// `dno_seg_gemv_gu8_q4k` は lane が 8 バイト読み、**同じ 8 バイトを隣の lane
/// (上位ニブル側) がもう一度読む** ので、要求バイトが有効バイトの 2 倍になる。
/// この版は 1 warp = 4 ブロック・1 ブロック = 8 lane にして、lane が qs の
/// 連続 16 バイト (= 下位 16 + 上位 16 要素) を読む。要求バイト = 有効バイト。
///
/// 値: 1 lane の和が 32 要素になるので丸め順が変わる (ビット一致しない)。
/// 起動形は `gu` / `gu8` と同一。**Q4_K のときだけ**効く (UD-Q4_K_XL の
/// gate/up が Q4_K)。
///
/// lab t562a (depth) / t562b (depth8k) の 4 サンプル中央値で gate+up+silu の段が
/// **33.3-34.3 -> 23.8-24.2 us/層 (-29%、gbs 228-267 -> 340-342)**。decode の壁は
/// -1% (ノイズ内)、prefill と既定モデルは不変なので、オペレーター規則
/// (段で 2 ラウンド 2% 以上 + 他が 1% 超悪化しない + 本文健全) で既定 on。
pub const SEG_GU16_ENV: &str = "DENNO_SEG_GU16";


/// [`SEG_GU16_ENV`] の解釈 (純関数)。**task 562 から既定 on** (未設定は有効、
/// `0` / `off` / `false` / `no` だけが偽 = A/B と切り戻し用)。
pub fn seg_gu16_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// [`SEG_GU16_ENV`] が有効か (プロセスで 1 回だけ env を読む)。
pub fn seg_gu16_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| seg_gu16_from_value(std::env::var(SEG_GU16_ENV).ok().as_deref()))
}


/// task 534: `DENNO_SEG_DN8` — down をベクトルロード版
/// (`dno_seg_gemv_k8_*` / `dno_seg_gemv_v4_q5_1`) で回す。**task 534 から既定 on**。
/// 引数も起動形も `dno_seg_gemv_k_*` と同一。
pub const SEG_DN8_ENV: &str = "DENNO_SEG_DN8";


/// [`SEG_DN8_ENV`] の解釈 (純関数)。**task 534 から既定 on** (未設定は有効、
/// `0` / `off` / `false` / `no` だけが偽)。
///
/// lab t534c / t534d の `dnv4` は 2 ラウンドとも -5.9 / -6.2% (down 30-33 ->
/// 16-18 us/層、330 -> 555-615 GB/s)。down の型が K 型のときは `k8` 版が
/// 選ばれる (t412c で gate/up の +5-6% は確認済みだが decode では読めなかった)。
pub fn seg_dn8_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// [`SEG_DN8_ENV`] が有効か (プロセスで 1 回だけ env を読む)。
pub fn seg_dn8_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| seg_dn8_from_value(std::env::var(SEG_DN8_ENV).ok().as_deref()))
}


/// task 278: 高速版 GEMV (`kernels_seg_k.ptx`) を使うか。
///
/// **既定 on** (`DENNO_SEG_K=0` / `off` / `false` で旧 `kernels_moe_batch.ptx` =
/// CPU とビット一致の経路へ)。丸め順が変わるので A/B と切り戻しのためのゲート。
pub const SEG_K_ENV: &str = "DENNO_SEG_K";


// ---- task 368: decode の待ちを null ストリームへ狭める -------------------------

/// task 368: `DENNO_DEC_SYNC_STREAM` — decode の待ちを `cuCtxSynchronize` から
/// **null ストリームのイベント 1 本**へ狭めるゲート (既定 off)。
///
/// `Context::sync()` は文脈の**全ストリーム**を待つので、「このスレッドが null
/// ストリームへ積んだ仕事の結果だけ」欲しい場所でも
/// - expert の**上げ直し** (非ブロッキングストリーム。1 本 178 us・実測 0.57 ms/token)
/// - shared expert の side stream
/// の完了まで待ってしまう。狭めても、ホストが読むのは null ストリームの非同期
/// コピーで書いた pinned バッファだけなので**値は変わらない** (同じカーネル・
/// 同じ入力・同じ順序で、待つ相手だけが違う)。
pub const DEC_SYNC_STREAM_ENV: &str = "DENNO_DEC_SYNC_STREAM";


/// 既定 (採否は lab t368x で決める。採用したら true にして既定 on にする)。
pub const DEC_SYNC_STREAM_DEFAULT: bool = false;


/// イベントのスロット。`seg_ev` の 0..=6 / 8 とは分ける (記録と待ちは必ず
/// 隣り合うので、同じスロットを 2 か所で同時に使うことはない)。
pub(crate) const DEC_SYNC_SLOT: usize = 7;


/// [`DEC_SYNC_STREAM_ENV`] の解釈 (純関数)。
pub fn dec_sync_stream_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("1" | "on" | "true" | "yes") => true,
        Some("0" | "off" | "false" | "no") => false,
        _ => DEC_SYNC_STREAM_DEFAULT,
    }
}


/// ゲートが有効か (プロセスで 1 回だけ env を読む)。
pub fn dec_sync_stream_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        dec_sync_stream_from_value(std::env::var(DEC_SYNC_STREAM_ENV).ok().as_deref())
    })
}


/// イベントで待てた回数 (**分岐が発火した証拠**)。
pub(crate) static DEC_SYNC_EVENTS: AtomicU64 = AtomicU64::new(0);

/// イベントが張れず文脈同期へ落ちた回数。
pub(crate) static DEC_SYNC_FALLBACKS: AtomicU64 = AtomicU64::new(0);


/// 1 リクエストぶんの 1 行 (ゲート on で 1 回も通らなければ `None`)。
pub fn dec_sync_stream_line() -> Option<String> {
    dec_sync_stream_line_with(dec_sync_stream_enabled())
}


/// [`dec_sync_stream_line`] の本体 (ゲートの真偽を引数で受ける = テストで両側を踏める)。
pub(crate) fn dec_sync_stream_line_with(on: bool) -> Option<String> {
    if !on {
        return None;
    }
    let ev = DEC_SYNC_EVENTS.load(Ordering::Relaxed);
    let fb = DEC_SYNC_FALLBACKS.load(Ordering::Relaxed);
    if ev == 0 && fb == 0 {
        return None;
    }
    Some(format!("[dec-sync-stream] events={ev} fallback={fb}"))
}


/// 積算を 0 に戻す (リクエスト単位で読む)。
pub fn dec_sync_stream_reset() {
    DEC_SYNC_EVENTS.store(0, Ordering::Relaxed);
    DEC_SYNC_FALLBACKS.store(0, Ordering::Relaxed);
}


/// ホストの `Vec<u8>` の一部へ `u64` の列を書く (prologue の詰め込み)。
pub(crate) fn put_u64s(dst: &mut [u8], src: &[u64]) {
    let bytes = std::mem::size_of_val(src);
    debug_assert_eq!(dst.len(), bytes, "u64 列の長さ");
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst.as_mut_ptr(), bytes);
    }
}


/// ホストの `Vec<u8>` の一部へ `u32` の列を書く。
pub(crate) fn put_u32s(dst: &mut [u8], src: &[u32]) {
    let bytes = std::mem::size_of_val(src);
    debug_assert_eq!(dst.len(), bytes, "u32 列の長さ");
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst.as_mut_ptr(), bytes);
    }
}


/// ホストの `Vec<u8>` の一部へ `f32` の列を書く。
pub(crate) fn put_f32s(dst: &mut [u8], src: &[f32]) {
    let bytes = std::mem::size_of_val(src);
    debug_assert_eq!(dst.len(), bytes, "f32 列の長さ");
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst.as_mut_ptr(), bytes);
    }
}


// ---------------------------------------------------------------------------
// task 498: seg 層の和 (e*4 B) の D2H を小さな pinned バウンス経由にする
// ---------------------------------------------------------------------------
//
// ゲートとカウンタは `crate::model::seg_prof` 側 (CUDA 無しのビルドでも起動ログと
// リクエスト終端の行を出せるようにするため)。ここは転送の実体だけを持つ。
// 動機と数値は `seg_prof::SEG_PIN_ENV` のコメントを見ること
// (lab t498b の `[seg-prof] d2h=13.3-14.8 us/層` = decode の 5%)。

/// 小さな D2H のバウンス 1 本ぶんの大きさ (e*4 = 10 KB に十分な余裕)。
pub(crate) const SEG_PIN_BYTES: usize = 64 << 10;


/// 専用バウンス (プロセスで 1 本、初回だけ確保する)。
///
/// `cuMemHostAlloc` は device を同期させるので、層ごとに取り直さない。
static SEG_PIN_BUF: OnceLock<std::sync::Mutex<Option<PinMem>>> = OnceLock::new();


fn seg_pin_buf() -> &'static std::sync::Mutex<Option<PinMem>> {
    SEG_PIN_BUF.get_or_init(|| std::sync::Mutex::new(None))
}


/// `src` (デバイス) から `dst` (ホストの `u8`) へ pinned バウンス経由で写す。
///
/// 成功で true。pinned が取れない / 転送が失敗したら false を返し、呼び出し側が
/// 従来の pageable 経路でやり直す (値は同じバイト列)。`dst` は [`SEG_PIN_BYTES`]
/// 以下であること。
pub(crate) fn seg_pin_copy_d2h(src: CUdeviceptr, dst: &mut [u8]) -> bool {
    let bytes = dst.len();
    if bytes == 0 || bytes > SEG_PIN_BYTES {
        return false;
    }
    let state = seg_pin_buf();
    let mut g = match state.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if g.is_none() {
        match PinMem::alloc(SEG_PIN_BYTES) {
            Ok(p) => *g = Some(p),
            Err(_) => {
                crate::model::seg_prof::note_seg_pin_fallback();
                return false;
            }
        }
    }
    let pin = g.as_ref().expect("確保済み").ptr;
    let lib = match lib() {
        Ok(l) => l,
        Err(_) => {
            crate::model::seg_prof::note_seg_pin_fallback();
            return false;
        }
    };
    let r = unsafe { check((lib.memcpy_dtoh)(pin, src, bytes), "cuMemcpyDtoH (seg pin)") };
    if r.is_err() {
        crate::model::seg_prof::note_seg_pin_fallback();
        return false;
    }
    unsafe { std::ptr::copy_nonoverlapping(pin as *const u8, dst.as_mut_ptr(), bytes) };
    crate::model::seg_prof::note_seg_pin_call();
    true
}


// ---------------------------------------------------------------------------
// task 502: seg 和の D2H を非同期にしてミスの CPU 計算の裏で流す
// ---------------------------------------------------------------------------
//
// ゲートとカウンタは `crate::model::seg_prof` 側 (`DENNO_SEG_SUM_ASYNC`、既定 off)。
// 動機は `seg_prof::SEG_ASYNC_ENV` のコメント。ここは転送の実体だけを持つ。
//
// - バンスは**スレッドごと**に 1 本 (`cuMemHostAlloc` はデバイスを止めるので初回だけ)。
//   プロセス共有にするとミス計算の間ずっとロックを握ることになるため。
// - コピーは null ストリーム (和カーネルの後ろ) に出し、**コピーの後にイベントを打って**
//   その完了だけを待つ。`cuMemcpyDtoH` (同期) はこの文脈の仕事を待つので、
//   ミス計算の後ろに直列で挟まっていた。

/// 非同期コピーの完了を押さえるイベントのスロット (`seg_ev` の 0..=6 と重ねない)。
const SEG_ASYNC_EV: usize = 7;


/// 和の D2H を非同期で pinned へ出す。`true` = 出した (完了待ちは
/// [`seg_async_d2h_collect`])。`false` = 従来の同期経路へ丸ごと落ちる。
pub(crate) fn seg_async_d2h_issue(src: CUdeviceptr, bytes: usize) -> bool {
    use crate::model::seg_prof as sp;
    if bytes == 0 || bytes > SEG_PIN_BYTES {
        sp::note_seg_async_fallback();
        return false;
    }
    let Ok(l) = lib() else {
        sp::note_seg_async_fallback();
        return false;
    };
    let ptr = SEG_ASYNC_PIN.with(|c| -> Option<*mut c_void> {
        let mut g = c.borrow_mut();
        if g.is_none() {
            match PinMem::alloc(SEG_PIN_BYTES) {
                Ok(p) => *g = Some(p),
                Err(_) => return None,
            }
        }
        g.as_ref().map(|p| p.ptr)
    });
    let Some(ptr) = ptr else {
        sp::note_seg_async_fallback();
        return false;
    };
    let r = unsafe {
        check(
            (l.memcpy_dtoh_async)(ptr, src, bytes, std::ptr::null_mut()),
            "cuMemcpyDtoHAsync (seg sum)",
        )
    };
    if r.is_err() {
        sp::note_seg_async_fallback();
        return false;
    }
    // ここまで来たらコピーは stream に積まれている。完了をイベントで押さえる
    // (イベントが張れない環境では従来経路へ。余分なコピーは無害)。
    if !seg_ev::record_raw(SEG_ASYNC_EV) {
        sp::note_seg_async_fallback();
        return false;
    }
    sp::note_seg_async_call();
    true
}


/// 非同期コピーの完了を待って `dst` へ写す。`false` = 取れなかった。
pub(crate) fn seg_async_d2h_collect(dst: &mut [u8]) -> bool {
    let bytes = dst.len();
    if bytes == 0 || bytes > SEG_PIN_BYTES {
        return false;
    }
    if seg_ev::wait(SEG_ASYNC_EV).is_err() {
        crate::model::seg_prof::note_seg_async_fallback();
        return false;
    }
    SEG_ASYNC_PIN.with(|c| {
        let g = c.borrow();
        match g.as_ref() {
            Some(p) => {
                unsafe {
                    std::ptr::copy_nonoverlapping(p.ptr as *const u8, dst.as_mut_ptr(), bytes)
                };
                true
            }
            None => false,
        }
    })
}


// ===========================================================================
// task 144: IQ 系 expert の層一括 GEMV (`kernels_exp_iq.cu` → `kernels_exp_iq.ptx`)
// ===========================================================================
//
// # なぜ要るか
//
// P39/P43 の VRAM 常駐経路は `gguf_row_bytes` が返す 4 型 (Q8_0 / Q5_1 / Q4_K /
// Q5_K) しか GPU で読めない。本番の expert は IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS /
// IQ4_NL / Q2_0 なので `moe_gemv_dev_ready` が**型の判定で** false になり、
// `expert_vram::acquire` は 1 本も載せず、480 expert 全部が CPU で回っていた
// (実測 99〜108 ms/tok、VRAM 使用 0 MiB)。`f_moe` (PTX モジュール) は正常だった。
//
// # 何をするか
//
// 1 層ぶんの expert FFN を **4 起動 + 1 同期**で回す:
//
//   gate 1 起動 → up 1 起動 → silu_mul 1 起動 → down 1 起動 → sync → D2H(h)
//
// 1 行 = 1 warp (32 レーン) で k 方向を分割して足すので、CPU とは丸め順が違う
// (task 144 はビット一致を要求しない)。起動形が同じなら値は決定的。
//
// ゲート `DENNO_EXPERT_IQ_GPU` が off のときは [`moe_gemv_iq_ready`] が false を
// 返すので、この経路は選ばれない (md5 はベースラインと一致する)。

/// カーネル一式 (PTX を起動時に 1 回だけ読む)。
pub(crate) struct ExpIqKernels {
    #[allow(dead_code)]
    module: CUmodule,
    /// 型ごとの GEMV ([`crate::model::exp_iq::slot`] の順)
    fns: [CUfunction; crate::model::exp_iq::SLOTS],
    /// silu(g) * u
    silu: CUfunction,
}


pub(crate) static EXP_IQ_KERNELS: OnceLock<Option<ExpIqKernels>> = OnceLock::new();

/// task 190: v2 (`kernels_exp_iq_v2.ptx`)。無いときは v1 へ落ちる。
pub(crate) static EXP_IQ_KERNELS_V2: OnceLock<Option<ExpIqKernels>> = OnceLock::new();

/// task 275: v3 (`kernels_exp_iq_v3.ptx`)。IQ2_S / IQ3_S のスロットだけを持つ。
pub(crate) static EXP_IQ_KERNELS_V3: OnceLock<Option<ExpIqKernels>> = OnceLock::new();

/// task 275: decode 用の合成 (v2 か v1 に v3 のスロットを重ねたもの)。
pub(crate) static EXP_IQ_KERNELS_DECODE: OnceLock<Option<ExpIqKernels>> = OnceLock::new();


thread_local! {
    pub(crate) static IQ_SCRATCH: std::cell::RefCell<IqScratch> =
        std::cell::RefCell::new(IqScratch::new());
}


pub(crate) static DEC_EPI_KERNELS: OnceLock<Option<DecEpiKernels>> = OnceLock::new();


// ---------------------------------------------------------------------------
// task 187: decode の head (語彙 GEMV) を v5 カーネルで回す
// ---------------------------------------------------------------------------
//
// 台帳 t187a の実測: head は 675.4 MB を 1.05 ms で読む = 643 GB/s。
// 現行の v2 カーネル (`kernels_moe.cu`) は 1 バイトずつの共有メモリ staging で、
// ピーク (1.79 TB/s) の 36% しか出ていない。v5 (task 184。ベクタロード +
// 4 アキュムレータ) に差し替えると **丸め順が変わる** (v2 とビット不一致)。
// lab t187c で 1.05 → 0.71 ms/token になり、md5 は SAME だったので既定 on。
// 切り戻しは `DENNO_DEC_HEAD_FAST=0`。

/// `DENNO_DEC_HEAD_FAST` の値を解釈する (純関数)。
///
/// **既定 on** (lab t187c: head の wall 1.08 → 0.74 ms/tok、decode -8.0%)。
/// 明示的な偽値 `0` / `off` / `false` / `no` だけ切る。
pub fn head_fast_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}


/// [`head_fast_from_value`] をプロセスで 1 回だけ読む。
pub fn head_fast_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| head_fast_from_value(std::env::var("DENNO_DEC_HEAD_FAST").ok().as_deref()))
}


/// task 427: head の語彙 GEMV を dn6 (1 ブロック rows 行 / 1 レーン 1 対) で回す
/// 行数 (純関数)。`DENNO_DEC_HEAD_FAST` の**既存 env に値を足す**形にした
/// (lab の COND は新しい env 名を落とすことがある)。
///
/// **既定 4 行** (lab t427b: head 段 0.73 -> 0.61 ms/token = -16.4%、gpu 0.70 ->
/// 0.58。短い decode 11.95 -> 11.7 ms/tok)。切り戻しは `DENNO_DEC_HEAD_FAST=v5`。
/// `dn6` / `dn6r4` = 4 行、`dn6r1` = 1、`dn6r2` = 2、`dn6r8` = 8。
/// `v5` / `1` / `on` / 偽値は 0 = これまでどおり v5。
pub fn head_dn6_rows_from_value(raw: Option<&str>) -> usize {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("dn6") | Some("dn6r4") => 4,
        Some("dn6r1") => 1,
        Some("dn6r2") => 2,
        Some("dn6r8") => 8,
        Some("v5") | Some("1") | Some("on") | Some("true") | Some("0") | Some("off")
        | Some("false") | Some("no") => 0,
        _ => 4,
    }
}


/// [`head_dn6_rows_from_value`] をプロセスで 1 回だけ読む。
pub(crate) fn head_dn6_rows() -> usize {
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        head_dn6_rows_from_value(std::env::var("DENNO_DEC_HEAD_FAST").ok().as_deref())
    })
}


/// task 504: head の GEMV を dn7 (1 対 = 4 レーン) で撃つか (純関数)。
///
/// **未設定が既定 on** (lab t504b: head 段 0.60 -> 0.46 ms/tok = -24%、
/// 2 ラウンド同じ、短い decode 10.6 -> 10.4、14k 後 11.6 -> 11.3、md5 全 SAME)。
/// dn6 は 1 レーンが x の 64 要素をレジスタに載せる (64 本) ので occupancy が
/// 下がる。dn7 は 16 要素で済む。切り戻しは `DENNO_DEC_HEAD_FAST=dn6` (旧既定)。
pub fn head_dn7_from_value(raw: Option<&str>) -> bool {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None => true,
        Some("dn7") | Some("d7") => true,
        Some(_) => false,
    }
}


/// [`head_dn7_from_value`] をプロセスで 1 回だけ読む。
pub(crate) fn head_dn7_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        head_dn7_from_value(std::env::var("DENNO_DEC_HEAD_FAST").ok().as_deref())
    })
}


/// task 427: head の GEMV を dn6 で 1 回だけ回す (pinned 経由・同期 1 回)。
///
/// 引数・発行順は [`CudaCtx::moe_gemv_dev_with`] の async 経路と同じで、
/// カーネルだけが `dno_mix_gemv_q8_0_dn6` に変わる (丸め順は v5 と違う)。
/// `Ok(false)` は「撃てなかった」= 呼び出し側は v5 へ落ちる。
pub(crate) fn head_gemv_dn6(
    ctx: &CudaCtx,
    w: &DevBuffer,
    x: &[f32],
    n_in: usize,
    n_out: usize,
    out: &mut [f32],
    row_bytes: usize,
    warps: usize,
    rows: usize,
    // task 504: `Some((rows, threads))` なら dn7 (1 対 = 4 レーン) で撃つ。
    dn7: Option<(usize, usize)>,
) -> Result<bool> {
    let lib = lib()?;
    MOE_SCRATCH.with(|cell| -> Result<bool> {
        let mut s = cell.borrow_mut();
        MoeScratch::ensure_tagged(&mut s.x, n_in * 4, crate::mem_audit::TAG_GEMV)?;
        MoeScratch::ensure_tagged(&mut s.o, n_out * 4, crate::mem_audit::TAG_GEMV)?;
        MoeScratch::ensure_pin(&mut s.pin_x, n_in * 4)?;
        MoeScratch::ensure_pin(&mut s.pin_o, n_out * 4)?;
        {
            let px = s.pin_x.as_mut().expect("pin_x は ensure 済み");
            px.as_mut_slice::<f32>()[..n_in].copy_from_slice(x);
        }
        let (xp, op, ho) = {
            let xb = s.x.as_ref().expect("x は ensure 済み").ptr;
            let ob = s.o.as_ref().expect("o は ensure 済み").ptr;
            let px = s.pin_x.as_ref().expect("pin_x は ensure 済み").ptr;
            (xb, ob, px)
        };
        unsafe {
            check(
                (lib.memcpy_htod_async)(xp, ho as *const c_void, n_in * 4, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (head dn6)",
            )?;
        }
        if let Some((rows7, thr7)) = dn7 {
            launch_mix_dn7(ctx, w.ptr, xp, op, n_in, n_out, row_bytes, rows7, thr7)?;
        } else {
            launch_dn6_gemv(ctx, w.ptr, xp, op, n_in, n_out, row_bytes, warps, rows)?;
        }
        {
            let po = s.pin_o.as_mut().expect("pin_o は ensure 済み");
            unsafe {
                check(
                    (lib.memcpy_dtoh_async)(po.ptr, op, n_out * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (head dn6)",
                )?;
            }
        }
        ctx.sync_decode()?;
        let po = s.pin_o.as_ref().expect("pin_o は ensure 済み");
        out.copy_from_slice(&po.as_slice::<f32>()[..n_out]);
        Ok(true)
    })
}


/// pinned リングの本数 (層 4 つぶんの余裕)。
pub(crate) const DEC_STAGE_SLOTS: usize = 4;


thread_local! {
    pub(crate) static DEC_STAGE: std::cell::RefCell<Option<DecStage>> = const {
        std::cell::RefCell::new(None)
    };
}


// ---------------------------------------------------------------------------
// task 356: expert 段の寄与 D2H を専用ストリームへ (`DENNO_DEC_EXP_D2H_STREAM`)
// ---------------------------------------------------------------------------
//
// IQ expert の層一括経路は「カーネルを積む -> 非同期 D2H -> CPU ミス -> 同期」で
// GPU と CPU を重ねる。ところが D2H を **null ストリーム** に出しているので、
// 小さいコピー (和モードは e = 2,560 要素 = 10 KB) では、driver がコピーの完了を
// 呼び出しの中で待つ経路に入ることがある。その待ちは CPU ミスの**前**に直列に
// 挟まるので、重ね打ちが丸ごと効かなくなる (lab t352a の台帳: experts 段
// 5.39 ms/tok に対し GPU 3.07 + ミス 2.78 で、重なりは 8% しか無い)。
//
// 専用の非ブロッキングストリームへ出すと、コピーは「この層のカーネルの完了」を
// イベントで待つだけになり、呼び出しはホストを止めない。待ちは CPU ミスの後へ
// 回す。**デバイスへ渡るバイト列・カーネル・発行順は 1 つも変わらない**。
//
// ゲートの解釈と積算は `model::exp_iq` 側にある ([`crate::model::exp_iq::DEC_D2H_STREAM_ENV`])。
// ここはストリームとイベントだけを持つ。

/// 専用ストリームとイベント 2 本 (スレッドローカル。作るのは 1 回だけ)。
pub(crate) struct ExpD2h {
    stream: CUstream,
    /// この層のカーネルの完了 (null ストリームに打つ)。
    ready: CUevent,
    /// コピーの完了 (専用ストリームに打つ)。
    done: CUevent,
}


pub(crate) const CU_STREAM_NON_BLOCKING_D2H: u32 = 0x01;

pub(crate) const CU_EVENT_DISABLE_TIMING_D2H: u32 = 0x02;


// ===========================================================================
// task 158: decode のミス expert を GPU で計算する (DENNO_DEC_MISS_UP)
// ===========================================================================
//
// ヒットは VRAM の LRU から、ミスは **pinned の bounce buffer** 経由で専用 scratch へ
// 非同期 H2D して、ヒットと**同じカーネル・同じ segment 表の形**で回す。
// ホストの写し (mmap → pinned) はヒットのカーネルを発行した後に行うので、GPU の
// ヒット計算と重なる。同期は層あたり 1 回のまま (task 150 と同じ)。
//
// lab の実測は REPORT_P158.md。

/// ミス 1 本ぶんの重み (mmap のスライス、ゲート / アップ / ダウン)。
pub struct MissExpertBytes<'a> {
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
}


/// [`moe_experts_iq_layer_miss_up`] の作業域 (grow-only・スレッドローカル)。
pub(crate) struct MissUpScratch {
    /// ミスの重み (ゲート / アップ / ダウン) を詰めたデバイス scratch。
    dg: Option<DevBuffer>,
    du: Option<DevBuffer>,
    dd: Option<DevBuffer>,
    /// bounce buffer (pinned)。mmap からここへ写してから H2D する。
    pin_g: Option<PinMem>,
    pin_u: Option<PinMem>,
    pin_d: Option<PinMem>,
    /// ミス用のポインタ表 (u64) と segment 表 (u32 x 4)。
    pg: Option<DevBuffer>,
    pu: Option<DevBuffer>,
    pd: Option<DevBuffer>,
    sgu: Option<DevBuffer>,
    sgd: Option<DevBuffer>,
    /// 重みの H2D 専用ストリームと完了イベント。
    stream: Option<CopyStream>,
    ev_ready: Option<CopyEvent>,
}


/// 1 バイト列を `reps` 回写して GB/s を返す (ホスト側の写しの実測)。
pub(crate) fn host_copy_gbs(dst: &mut [u8], src: &[u8], reps: usize) -> f64 {
    let t = std::time::Instant::now();
    let dst = &mut dst[..src.len()];
    for _ in 0..reps {
        dst.copy_from_slice(src);
    }
    let ns = t.elapsed().as_nanos() as f64;
    if ns <= 0.0 {
        return 0.0;
    }
    (reps * src.len()) as f64 / ns
}


// ===========================================================================
// task 148: prefill の m 行 expert GEMV (`kernels_exp_iq_pf.cu`)
// ===========================================================================
//
// decode の層一括 GEMV (`kernels_exp_iq`) は 1 起動 = 1 行。prefill は 512 トークンの
// チャンクで 1 expert あたり平均 20 行を受けるので、**同じ重みの読みで 16 行ずつ
// まとめて**積むカーネルを別モジュールに持つ (`exp_pf::ROWS_PER_LAUNCH`)。
//
//   gate 1 起動 → up 1 起動 → silu_mul 1 起動 → down 1 起動 → sync 1 回 → D2H(h)
//
// 1 行ぶんの累積順は decode と同一なので、同じ行を m=1 と m 行で回すとビット一致
// する。行の対応は `seg` (m / rows の先頭) と `rows` (行 → トークン番号) で決まる。
//
// 予算: 作業域は層ごとに grow-only で使い回す。確保に失敗したら **そのプロセスでは
// 以後この経路を試さない** (PF_DEAD)。prefill の途中で毎層 cuMemAlloc を叩いて
// attention 用の領域を削る、という事故を避ける (t147a の 26 GB で起きたのはこれ)。

/// カーネル一式 (PTX を起動時に 1 回だけ読む)。
pub(crate) struct ExpPfKernels {
    #[allow(dead_code)]
    module: CUmodule,
    /// task 309: K 系のモジュール (`kernels_exp_k_pf.ptx`)。読めなければ null で、
    /// K 系のスロットは null のまま = その型は CPU 経路へ落ちる (fail closed)。
    #[allow(dead_code)]
    kmodule: CUmodule,
    /// 型ごとの m 行 GEMV ([`crate::model::exp_pf::kernel_name`] の順)。
    /// 0..IQ_SLOTS は IQ のモジュール、K_SLOT0.. は K のモジュールの関数。
    fns: [CUfunction; crate::model::exp_pf::SLOTS],
    /// silu(g) * u
    silu: CUfunction,
    /// task 162: 行の加重和 (`dno_exp_pf_accum`)。task 151 で未使用に戻した
    /// カーネルで、PTX には定義だけ残っている。ストリーミング経路だけが使う。
    accum: Option<CUfunction>,
}


/// 確保に失敗したら立てる (以後このプロセスでは prefill の GPU 経路を使わない)。
pub(crate) static PF_DEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// PTX の置き場所。`DENNO_EXP_PF_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_exp_iq_pf.ptx`。
pub(crate) fn exp_pf_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_PF_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq_pf.ptx"),
    }
}


/// PTX の置き場所。`DENNO_EXP_K_PF_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_exp_k_pf.ptx` (task 309 の K 系)。
pub(crate) fn exp_k_pf_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_K_PF_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_k_pf.ptx"),
    }
}


/// task 233: 層優先 prefill で、共有 expert の出力を置く「行」を MoE の作業域に足す。
///
/// `total_rows` は routed expert の行数。`sc.h` を `(total_rows + ntok)` 行へ広げ、
/// 末尾 `ntok` 行の先頭を返す。共有 expert の出力をここへ書き、行の加重和
/// (`dno_exp_pf_accum`) が **最後の 1 項**として読む (ホストへ戻さない)。
#[cfg(feature = "cuda")]
pub fn pf_moe_shared_tail(total_rows: usize, ntok: usize, e: usize) -> Option<CUdeviceptr> {
    let rows = total_rows.checked_add(ntok)?;
    let bytes = rows.checked_mul(e)?.checked_mul(4)?;
    let off = total_rows.checked_mul(e)?.checked_mul(4)?;
    PF_SCRATCH.with(|cell| -> Option<CUdeviceptr> {
        let mut sc = cell.borrow_mut();
        if let Some(v) = pool_view(crate::model::pf_pool::R::H, 0, bytes) {
            sc.h = Some(v);
        } else {
            MoeScratch::ensure(&mut sc.h, bytes).ok()?;
        }
        let b = sc.h.as_ref()?;
        if b.bytes() < bytes {
            return None;
        }
        Some(b.ptr() + off as u64)
    })
}


/// task 233: MoE の作業域の `row` 行目 (`[e]`) をホストへ戻す (役割 RMS 用)。
#[cfg(feature = "cuda")]
pub fn pf_moe_h_row(row: usize, e: usize, out: &mut [f32]) -> bool {
    if out.len() != e || e == 0 {
        return false;
    }
    PF_SCRATCH.with(|cell| -> bool {
        let sc = cell.borrow();
        let Some(b) = sc.h.as_ref() else {
            return false;
        };
        let off = row.saturating_mul(e).saturating_mul(4);
        if b.bytes() < off + e * 4 {
            return false;
        }
        match lib() {
            Ok(l) => unsafe {
                check(
                    (l.memcpy_dtoh)(
                        out.as_mut_ptr() as *mut c_void,
                        b.ptr() + off as u64,
                        e * 4,
                    ),
                    "cuMemcpyDtoH (moe h row)",
                )
                .is_ok()
            },
            Err(_) => false,
        }
    })
}


// ---------------------------------------------------------------------------
// task 162: prefill のミス expert を載せるステージング (2 枚 + pinned + 別ストリーム)
// ---------------------------------------------------------------------------
//
// task 156 のステージングを、新型ホスト (PCIe Gen5 x16、H2D 52.6 GB/s pinned /
// 19.6 GB/s pageable) 向けに作り直したもの。task 156 との違いは 3 つ。
//
// 1. 2 枚のステージング (ダブルバッファ) に分け、グループ i+1 の H2D を
//    グループ i の計算と重ねる。上げは専用の**非ブロッキング**ストリームへ積む
//    (blocking にすると legacy の null ストリームと暗黙に同期して 1 バイトも
//    重ならない)。計算側は自分のグループの完了イベントだけを待つ。
// 2. pinned のバウンスバッファ経由 (`DENNO_PF_STREAM_PIN=1`、既定)。ホストの
//    写しは rayon で並列化する (`0` は pageable の同期コピー = task 156 と同じ)。
//    **注意**: pinned 経由は「ページキャッシュ → pinned」の写しがホストの
//    DRAM 帯域を食うので、pageable の driver staging より速いとは限らない。
//    実測は `[exp-pf-stream] probe_GBs=` (最初のグループを同期で測った実速度) で見る。
// 3. ステージングは `cuMemAlloc` を直接叩き、**decode の LRU を 1 本も動かさない**。
//    `DevBuffer::alloc_tagged` の確保失敗フォールバックは expert を追い出すので
//    ここでは使わない。空きが 768 MiB を切らない範囲だけ使う (t156b の事故)。

/// ステージングの行列先頭のアラインメント (カーネルの読み出しに十分な 256 B)。
pub(crate) const PF_STAGE_ALIGN: usize = 256;


/// カードに残す余裕 (MiB)。ステージングはこれを食い潰さない。
const PF_STAGE_RESERVE_MB: usize = 768;


/// ステージング 1 枚の最小サイズ (MiB)。これ未満しか取れないなら諦める。
pub(crate) const PF_STAGE_MIN_MB: usize = 16;


/// ステージングの枚数 (多重バッファ)。
///
/// 2 枚だと「グループ g+1 の写し」が「グループ g-1 の計算」の完了待ちに
/// 引っかかり、ホストの写しと DMA が計算と直列になる (lab t162b: chunk 2048 で
/// 1 グループ 4.4 ms = 計算 2.6 + 写し 0.9 + DMA 0.8 の和)。3 枚にすると
/// 写しは 2 つ前のグループの計算と重なる。
pub(crate) const PF_STAGE_SLOTS: usize = 3;


pub(crate) fn pf_stage_align_up(x: usize) -> usize {
    x.div_ceil(PF_STAGE_ALIGN) * PF_STAGE_ALIGN
}


thread_local! {
    pub(crate) static PF_STREAM: std::cell::RefCell<PfStream> =
        std::cell::RefCell::new(PfStream::new());
    /// task 162: 活性 (`xs`, 層ごとに 20 MB 級) の pinned 転送元。
    pub(crate) static PF_XPIN: std::cell::RefCell<Option<PinMem>> =
        const { std::cell::RefCell::new(None) };
}


/// ステージングの現在の使用量 (確保バイト, 残り空き MiB)。診断用。
pub fn pf_stage_usage() -> (usize, usize) {
    PF_STREAM.with(|cell| {
        let st = cell.borrow();
        let cap: usize = st.slot.iter().map(|s| s.cap).sum();
        (cap, dev_mem_mb().map(|(f, _)| f).unwrap_or(0))
    })
}


/// task 248: prefill の arena の (確保バイト, 実際の使用ピーク)。
///
/// 差が大きいほど「過大な作業域」= expert に回せる VRAM (台帳の
/// `scratch arena` 行は確保バイトしか出さないので、ここで内訳を出す)。
///
/// `try_lock` なのは、`arena_alloc` (arena のロック保持中) から
/// `DevBuffer::alloc_tagged` → `mem_audit::note_alloc` のピーク記録で
/// 呼ばれうるため (通常の Mutex だと自己デッドロックする)。
pub fn arena_usage() -> Option<(usize, usize)> {
    let g = arena_state().try_lock().ok()?;
    Some((g.plan.capacity(), g.plan.high_water))
}


/// 空き VRAM から余裕を引いた範囲で確保する (半分ずつ落とす)。
///
/// `DevBuffer::alloc_tagged` は失敗すると expert を追い出すので**使わない**
/// (decode の在庫を動かさないのがこの経路の前提)。
///
/// task 257: プールがあれば `slot` 枚目の区画をそのまま使う (確保しない)。
pub(crate) fn pf_stage_alloc(want: usize, slot: usize) -> Option<DevBuffer> {
    if want == 0 {
        return None;
    }
    if let Some(v) = pool_view(crate::model::pf_pool::R::Stage, slot, want) {
        return Some(v);
    }
    let free_mb = dev_mem_mb().map(|(f, _)| f).unwrap_or(0);
    let room = free_mb.saturating_sub(PF_STAGE_RESERVE_MB) << 20;
    let mut want = want.min(room);
    while want >= (PF_STAGE_MIN_MB << 20) {
        if let Ok(ptr) = DevBuffer::raw_alloc(want) {
            // task 248: ステージングは expert とは別の行にする (台帳の
            // 「experts」は slab のチャンクだけ、にする)。
            crate::mem_audit::note_alloc(crate::mem_audit::TAG_PF_STAGE, want.max(1));
            return Some(DevBuffer {
                ptr,
                bytes: want,
                owned: true,
                tag: crate::mem_audit::TAG_PF_STAGE,
            });
        }
        want /= 2;
    }
    None
}


/// task 325: pread 専用の rayon プール (スレッド数ごとに 1 個だけ作る)。
///
/// task 554: **int-auto の実装のまま**。上げスレッドを 2 本にしても 1 本の
/// プールを共有する (2 本に分けた版 = `pf_pread_pool_install` は、既定モデルの
/// 3,826 cold が返らなくなる t554c/t554d の原因を切り分けられなかったので捨てた)。
#[cfg(feature = "par")]
pub(crate) fn pf_pread_pool(nt: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    static NT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let first = NT.swap(nt, std::sync::atomic::Ordering::Relaxed) == 0;
    if !first {
        // 2 本目以降の違う本数は無視する (プロセスで 1 個)。
        return POOL.get().and_then(|p| p.as_ref());
    }
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(nt)
            .thread_name(|i| format!("pf-pread-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}


/// task 558: 上げスレッドの番号ごとの pread プール (`DENNO_PF_PREAD_POOL_PER_UP`)。
///
/// プールは**プロセスで 1 組**だけ作り、リクエストをまたいで使い回す
/// (上げスレッドはリクエストごとに作り直されるので、スレッドローカルに持つと
/// リクエストごとに 32 本が解放されず積み上がる。t554 の版はこれで漏れた)。
/// 番号が範囲外 (上げスレッド 3 本を超えた等) は `None` を返し、呼び出し側が
/// 共有プールへ落ちる。
#[cfg(feature = "par")]
pub(crate) fn pf_pread_pool_up(nt: usize, up: usize) -> Option<&'static rayon::ThreadPool> {
    use std::sync::OnceLock;
    // 上げスレッドは最大 3 本 (`up_workers_from_value` の上限)。
    const MAX_UP: usize = 3;
    if up >= MAX_UP {
        return None;
    }
    static POOLS: OnceLock<Vec<Option<rayon::ThreadPool>>> = OnceLock::new();
    let n = crate::model::exp_pf::up_workers().clamp(1, MAX_UP);
    let pools = POOLS.get_or_init(|| {
        (0..n)
            .map(|u| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(nt)
                    .thread_name(move |i| format!("pf-pread-u{u}-{i}"))
                    .build()
                    .ok()
            })
            .collect()
    });
    pools.get(up).and_then(|p| p.as_ref())
}


thread_local! {
    /// task 558: このスレッドが上げ専用スレッドなら、その番号 (0..)。
    /// 計算スレッドは `None` のまま (従来の共有プールを使う)。
    pub(crate) static PF_UP_IDX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}


// ---------------------------------------------------------------------------
// task 325: グループを跨いだ先読み (`DENNO_PF_RA_GROUPS`)
// ---------------------------------------------------------------------------
//
// 写しスレッドは mmap をフォールトしながら memcpy するので、1 スレッドあたり
// 128 KB ずつしか I/O が飛ばない。**写しの外**の専用スレッドから、n グループ
// 先の範囲へ `MADV_WILLNEED` を投げて page cache を先に埋める。投げるだけ
// (戻りを待たない) なので、写しは従来どおりで、フォールトが当たったときには
// ページが載っている、という狙い。**中身は変えない** (助言だけ)。

/// `MADV_WILLNEED` を投げる (範囲をページ境界へ丸める)。投げたら true。
fn pf_madvise_willneed(ptr: usize, len: usize) -> bool {
    if len == 0 {
        return false;
    }
    #[cfg(target_os = "linux")]
    {
        const MADV_WILLNEED: i32 = 3; // linux/mman-common.h
        extern "C" {
            fn madvise(addr: *mut core::ffi::c_void, len: usize, advice: i32) -> i32;
        }
        let a = ptr & !4095usize;
        let b = (ptr + len + 4095) & !4095usize;
        // SAFETY: mmap の範囲 (または RAM tier の領域) を指す。madvise は
        // 中身を変えない。範囲外を指さないよう呼び出し側が同じ区間を渡す。
        unsafe { madvise(a as *mut core::ffi::c_void, b - a, MADV_WILLNEED) == 0 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
        false
    }
}


/// 先読みの依頼口 (深さ 1。埋まっていたら投げない = 先へ行き過ぎない)。
static PF_RA_TX: std::sync::OnceLock<Option<std::sync::mpsc::SyncSender<Vec<(usize, usize)>>>> =
    std::sync::OnceLock::new();


fn pf_ra_sender() -> Option<&'static std::sync::mpsc::SyncSender<Vec<(usize, usize)>>> {
    PF_RA_TX
        .get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<(usize, usize)>>(1);
            std::thread::Builder::new()
                .name("pf-ra".to_string())
                .spawn(move || {
                    while let Ok(v) = rx.recv() {
                        let t = std::time::Instant::now();
                        let mut bytes = 0usize;
                        for (ptr, len) in v {
                            if pf_madvise_willneed(ptr, len) {
                                bytes += len;
                            }
                        }
                        exp_pf::note_ra_job(bytes, t.elapsed().as_nanos() as u64);
                    }
                })
                .ok()
                .map(|_| tx)
        })
        .as_ref()
}


/// 1 グループぶんの先読みを依頼する (計算も写しもブロックしない)。
pub(crate) fn pf_ra_submit(items: &[PfItem<'_>]) {
    let Some(tx) = pf_ra_sender() else { return };
    let mut v: Vec<(usize, usize)> = Vec::with_capacity(items.len() * 3);
    for it in items {
        for b in &it.1 {
            if !b.is_empty() {
                v.push((b.as_ptr() as usize, b.len()));
            }
        }
    }
    if v.is_empty() {
        return;
    }
    // 深さ 1 がいっぱいなら先読みは既に先行している (投げない)。
    let _ = tx.try_send(v);
}


/// ステージングの範囲を「1 回の DMA で流す塊」に切る (純関数)。
///
/// `parts` は (ステージング先頭からのオフセット, 長さ) の昇順。
/// `limit == 0` は「全部で 1 回」。塊は parts の境界で切り、長さが `limit` を
/// 超えないようにする (1 本だけで `limit` を超えるときはその 1 本)。
/// 返すのは塊の [先頭, 終端) で、間の整列の隙間も含む (中身は読まれない)。
pub(crate) fn pf_chunk_spans(parts: &[(usize, usize)], limit: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    if parts.is_empty() {
        return out;
    }
    if limit == 0 {
        let last = parts[parts.len() - 1];
        out.push((parts[0].0, last.0 + last.1));
        return out;
    }
    let mut start = parts[0].0;
    let mut end = parts[0].0 + parts[0].1;
    for &(o, l) in &parts[1..] {
        let e = o + l;
        if e - start > limit {
            out.push((start, end));
            start = o;
        }
        end = e;
    }
    out.push((start, end));
    out
}


// ---------------------------------------------------------------------------
// task 227: mmap のページを写さずに DMA する (cuMemHostRegister)
// ---------------------------------------------------------------------------
//
// 台帳では写し (mmap → WC pinned) が 14k で 5.90 s、DMA のブロックが 8.85 s。
// 写しを消せれば DDR の 2 往復 (158 GB の write + read) が丸ごと消える。
// ただし pin には代償がある (登録の時間と、ページキャッシュを追い出せなくする
// こと) ので、**登録できた範囲だけ**使い、失敗したら pageable へ落とす。

/// 登録済みのホスト範囲 (昇順・非重複)。
pub(crate) static PF_REG: std::sync::Mutex<Vec<(usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// 登録を諦めたか (driver が file backed を受け付けない場合)。
pub(crate) static PF_REG_DEAD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 連続失敗の数 (諦めるまでの猶予)。
pub(crate) static PF_REG_FAILS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);


/// 登録済みの合計バイト (上限 [`exp_pf::reg_bytes`] で頭打ちにする)。
pub(crate) static PF_REG_OK_BYTES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


/// 登録に使ったフラグ (0 = まだ成功していない)。
pub(crate) static PF_REG_FLAGS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);


/// `cuMemHostRegister` の候補フラグ。
/// READ_ONLY (0x08) が file backed の読み取り専用写像向け。0x0a は DEVICEMAP 付き
/// (`cuMemHostGetDevicePointer` を使うため)、0x00 は素の登録。
///
/// # 実測 (lab t227a / t227b)
///
/// この driver は **file backed (MAP_SHARED のモデル mmap) を pin できない**:
/// 0x08 → error 801 (NOT_SUPPORTED)、0x0a → 801、0x00 → error 1 (INVALID_VALUE)。
/// 4 回失敗したら諦めて pageable で積む (`[pf-zc]` の 1 行が出る)。
/// なお pageable 直送は写しを消すが driver が内部でバウンスするので 21.1 GB/s
/// 止まり (WC pinned 単体は 56.8 GB/s) — 3 往復は消えない。
pub(crate) const PF_REG_FLAG_TRIES: [u32; 3] = [0x08, 0x0a, 0x00];


/// グループ `g` の H2D 完了を待つ (ホスト)。待ち時間 (ns) を返す。
pub(crate) fn pf_wait_up_in(st: &mut PfStream, g: u64, pinned: bool) -> Result<u64> {

    if !pinned {
        return Ok(0);
    }
    let s = (g as usize) % PF_STAGE_SLOTS;
    let Some(ev) = st.slot[s].ev_up.as_ref() else {
        return Ok(0);
    };
    let Some(es) = lib()?.event_sync else {
        return Ok(0);
    };
    let t = std::time::Instant::now();
    unsafe {
        check(es(ev.0), "cuEventSynchronize (stage up)")?;
    }
    Ok(t.elapsed().as_nanos() as u64)
}


/// グループ `g` の計算を積んだ後に呼ぶ (スロットの再利用判定用)。
pub(crate) fn pf_mark_used_in(st: &mut PfStream, g: u64, pinned: bool) {
    if !pinned {
        return;
    }
    let s = (g as usize) % PF_STAGE_SLOTS;
    if let Some(ev) = st.slot[s].ev_use.as_ref() {
        if let Ok(l) = lib() {
            if let Some(record) = l.event_record {
                unsafe {
                    let _ = record(ev.0, std::ptr::null_mut());
                }
            }
        }
    }
}


/// グループ番号を通し番号で `n` 本ぶん確保する (task 165)。
///
/// 層ごとに 0 へ戻すと、スロットの持ち主 (`g % 3`) が前の層のグループと
/// 重なり、**前の層の完了イベントを自分のグループの完了と誤認する**。
/// ここはその不変条件を 1 か所に閉じるための純関数。
pub(crate) fn pf_take_group_ids(next: &mut u64, n: usize) -> u64 {
    let base = *next;
    *next = next.saturating_add(n as u64);
    base
}


/// 進捗の置き場を 1 組作る (task 554: ワーカーの本数によらず 1 組を共有する)。
pub(crate) fn new_up_state() -> std::sync::Arc<(std::sync::Mutex<PfUpState>, std::sync::Condvar)> {
    std::sync::Arc::new((
        std::sync::Mutex::new(PfUpState {
            done: Vec::new(),
            times: Vec::new(),
            probe: None,
            err: None,
        }),
        std::sync::Condvar::new(),
    ))
}


thread_local! {
    /// task 165: このスレッドの prefill 用の上げ専用スレッド (初回だけ起動)。
    /// task 554: `DENNO_PF_UP_WORKERS` 本をまとめて持つ。
    pub(crate) static PF_UP: std::cell::RefCell<Option<PfUp>> =
        const { std::cell::RefCell::new(None) };
}


/// グループ `g` の上げを専用スレッドへ依頼する (戻り値はデバイスポインタ表と、
/// そのグループが `pread` を使ったか)。
///
/// task 570: `pread` の真偽を返すのは、依頼の深さを pread のグループだけ深くする
/// ため ([`crate::model::exp_pf::lookahead_for`])。
pub(crate) fn pf_submit_group(
    g: u64,
    items: &[PfItem<'_>],
    probe: bool,
) -> Result<(Option<Vec<[u64; 3]>>, bool)> {
    pf_up_ensure();
    PF_STREAM.with(|cell| -> Result<(Option<Vec<[u64; 3]>>, bool)> {
        let mut st = cell.borrow_mut();
        PF_UP.with(|uc| -> Result<(Option<Vec<[u64; 3]>>, bool)> {
            let slot = uc.borrow();
            let Some(up) = slot.as_ref() else {
                return Ok((None, false));
            };
            let plan = pf_slot_prepare(&mut st, g, items, true)?;
            let s = plan.slot;
            let stream = pf_up_stream(&mut st)?;
            let base = st.slot[s].buf.as_ref().expect("確保済み").ptr();
            let pin = st.pin[s].as_ref().expect("確保済み").ptr as usize;
            // スロットを前回読んだ計算の完了イベント (ワーカーが待つ)。
            let wait_ev = if st.slot[s].used {
                st.slot[s]
                    .ev_use
                    .as_ref()
                    .map(|e| e.0 as usize)
                    .unwrap_or(0)
            } else {
                0
            };
            // task 320: 分離モードでは、ワーカーが**ホストで待つ**のは
            // 「このスロットの pinned を前回読んだ DMA の完了」だけにする
            // (カーネルの完了は上げストリームの待ちで GPU に守らせる)。
            let pin_ev = if exp_pf::up_decouple_on() && st.slot[s].used {
                st.slot[s]
                    .ev_up
                    .as_ref()
                    .map(|e| e.0 as usize)
                    .unwrap_or(0)
            } else {
                0
            };
            let up_ev = st.slot[s]
                .ev_up
                .as_ref()
                .map(|e| e.0 as usize)
                .unwrap_or(0);
            if up_ev == 0 {
                bail!("H2D 完了イベントが無い");
            }
            st.slot[s].used = true;
            up.submit(PfUpJob {
                g,
                need: plan.need,
                pread: plan.pread,
                dst: base,
                pin,
                parts: plan.parts,
                stream,
                wait_ev,
                pin_ev,
                up_ev,
                probe,
            })?;
            Ok((Some(plan.ptrs), plan.pread))
        })
    })
}


/// グループ `g` の依頼がワーカーに**積まれた**ことを待つ (task 324)。
///
/// ワーカーが `ev_up` を記録するまでを待つ。写しと DMA の積み込みは済んでいて、
/// DMA の完了はまだ待たない。戻り値は順番待ちの時間 (ns)。
pub(crate) fn pf_wait_ready(g: u64) -> Result<u64> {
    PF_UP.with(|cell| -> Result<u64> {
        match cell.borrow().as_ref() {
            Some(u) => u.wait_ready(g),
            None => bail!("上げスレッドが居ない"),
        }
    })
}


/// グループ `g` の上げを待つ (ut 経路)。
///
/// 戻り値は (写し ns, 積み込み ns, 順番待ち ns, H2D 待ち ns, スロット待ち ns)。
pub(crate) fn pf_wait_group(g: u64) -> Result<(u64, u64, u64, u64, u64)> {
    let ready_ns = pf_wait_ready(g)?;
    let wait_ns = PF_STREAM.with(|cell| -> Result<u64> {
        let mut st = cell.borrow_mut();
        pf_wait_up_in(&mut st, g, true)
    })?;
    let (copy_ns, enq_ns, slot_ns) = PF_UP.with(|cell| match cell.borrow().as_ref() {
        Some(u) => u.times(g),
        None => (0, 0, 0),
    });
    Ok((copy_ns, enq_ns, ready_ns, wait_ns, slot_ns))
}


// ---------------------------------------------------------------------------
// task 481: 上げスレッドの「写し + DMA」を小さなバウンス 2 枚で重ねる
// ---------------------------------------------------------------------------
//
// 3,826 cold の `[exp-pf-stream]` は up_MB=22,455 に対して copy_s=1.09 s で、
// 壁 2,119 ms の 51%。内訳は 3 往復 (page cache を読む 22.4 GB / バウンスへ
// 書く 22.4 GB / DMA がバウンスを読み直す 22.4 GB) で、DRAM の**読み**が
// 2 本ある。DMA が読むのは直前に書いた行なので、まだ L3 (96 MB) にあれば
// 読み直しは DRAM に降りない。キャッシュありの pinned を 2 枚交互に使い、
// 写し(i+1) を DMA(i) と重ねる形にして、読みを 1 本に落とす。
//
// デバイスへ渡るバイト列は 1 回写しと同一 (値は 1 ビットも変わらない)。
// 外れたとき (DMA が L3 を見ない等) は従来と同じ 3 往復に戻るだけで、
// 悪化は「小さい DMA の起動が増える」ぶんに留まる (**推定**)。

thread_local! {
    /// task 481: 分割 DMA の状態 (上げスレッド専用)。
    pub(crate) static PF_UP_SPLIT: std::cell::RefCell<Option<Box<PfUpSplit>>> =
        const { std::cell::RefCell::new(None) };
}


// ===========================================================================
// task 250: tensor core の grouped expert GEMM (`kernels_exp_iq_tc.cu`)
// ===========================================================================
//
// ストリーミング 1 グループぶんの gate/up と down を、**1 起動ずつ** で回す。
// 現行の m 行 GEMV は 1 block = 8 出力行で活性を読み直すため、行数が多い
// (m = 300 前後) 層優先 prefill では活性律速になる (REPORT_P242)。
// ここは 64 トークン × 64 行のタイルで受け、bf16 の mma (FP32 アキュムレート)
// で積む。数値は変わる (丸めだけ。`DENNO_PF_TC_EXP` で切り戻せる)。

/// カーネル名 (PTX の `.visible .entry`)。slot は `exp_iq::slot` の 0..=3。
pub(crate) const TC_GU_NAMES: [&str; 4] = [
    "dno_exp_tc_gu_iq2s",
    "dno_exp_tc_gu_iq3xxs",
    "dno_exp_tc_gu_iq3s",
    "dno_exp_tc_gu_iq4xs",
];

pub(crate) const TC_DN_NAMES: [&str; 6] = [
    "dno_exp_tc_dn_iq2s",
    "dno_exp_tc_dn_iq3xxs",
    "dno_exp_tc_dn_iq3s",
    "dno_exp_tc_dn_iq4xs",
    "dno_exp_tc_dn_iq4nl",
    "dno_exp_tc_dn_q2_0",
];


/// `.cu` の `EXTC_BM` / `EXTC_BN` / `EXTC_THREADS` と一致させること。
pub(crate) const EXTC_BM: usize = 64;

pub(crate) const EXTC_BN: usize = 64;

pub(crate) const EXTC_THREADS: usize = 256;

/// `.cu` のカーネル引数の数 (ABI ゲート)。gu は 10、down は 9。
pub(crate) const TC_GU_NARGS: usize = 10;

pub(crate) const TC_DN_NARGS: usize = 9;


/// カーネル一式 (PTX を起動時に 1 回だけ読む)。
pub(crate) struct ExpTcKernels {
    #[allow(dead_code)]
    module: CUmodule,
    /// gate/up (silu 込み) の型ごと
    gu: [CUfunction; 4],
    /// down の型ごと (`exp_iq::slot` の 0..=5)
    dn: [CUfunction; 6],
}


pub(crate) static EXP_TC_KERNELS: OnceLock<Option<ExpTcKernels>> = OnceLock::new();


/// PTX の置き場所。`DENNO_EXP_TC_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_exp_iq_tc.ptx`。
pub(crate) fn exp_tc_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_TC_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq_tc.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける
/// (以後このプロセスでは tensor core 経路を使わず、既存の GEMV へ落ちる)。
/// task 257: プールのレイアウト判定からも見られるようにする (g/u を取るか)。
#[cfg(feature = "cuda")]
pub fn exp_tc_ready() -> bool {
    exp_tc_kernels().is_some()
}


/// task 250: この形を tensor core の grouped GEMM で回せるか (純関数)。
///
/// - gate と up が同じ型 (1 回の起動で 2 枚を同時にデコードする)
/// - gate/up は 4 型 (IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS)、down はそれに
///   IQ4_NL / Q2_0 を足した 6 型
/// - k が型のブロックで割り切れる、n が [`EXTC_BN`] の倍数 (端数タイル無し)
pub fn tc_shapes_ok(
    gu_slot: Option<usize>,
    dn_slot: Option<usize>,
    ty_same: bool,
    k_gu: usize,
    k_dn: usize,
    n_gu: usize,
    n_dn: usize,
) -> bool {
    // gate/up は 256 要素ブロックの 4 型だけ (このモデルの gate/up はこの 4 型)。
    let gu_ok = matches!(gu_slot, Some(0..=3)) && k_gu % 256 == 0;
    // down は IQ4_NL (32 要素) / Q2_0 (64 要素) も受ける
    // (実モデルの down はこの 2 型。P202)。
    let dn_ok = match dn_slot {
        Some(0..=3) => k_dn % 256 == 0,
        Some(4) => k_dn % 32 == 0,
        Some(5) => k_dn % 64 == 0,
        _ => false,
    };
    ty_same
        && gu_ok
        && dn_ok
        && k_gu > 0
        && k_dn > 0
        && n_gu % EXTC_BN == 0
        && n_dn % EXTC_BN == 0
}


// ===========================================================================
// task 416: tensor core の grouped expert GEMM — **K 系** (`kernels_exp_k_tc.cu`)
// ===========================================================================
//
// `kernels_exp_iq_tc.cu` (task 250) の K 系版。UD-Q4_K_XL の expert は
// Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0 で、既定 on の IQ 版 TC の型判定に掛からず
// m 行 GEMV (`kernels_exp_k_pf.cu`) のままだった (14k で experts 段の
// gpu_busy 18.4 s = prefill 壁の 76.7%。活性の読み直しが重みの 400 倍)。
// 数値は変わる (重みと活性を bf16 に丸める。prefill の射影は
// 2026-10-28 のオペレーター指示でビット一致契約の外)。
// ゲートは `DENNO_PF_TC_EXP_K` (既定 off)。

/// カーネル名 (PTX の `.visible .entry`)。index = `exp_pf::slot` - `K_SLOT0`。
pub(crate) const KTC_GU_NAMES: [&str; 5] = [
    "dno_exp_ktc_gu_q4k",
    "dno_exp_ktc_gu_q5k",
    "dno_exp_ktc_gu_q6k",
    "dno_exp_ktc_gu_q5_1",
    "dno_exp_ktc_gu_q8_0",
];

pub(crate) const KTC_DN_NAMES: [&str; 5] = [
    "dno_exp_ktc_dn_q4k",
    "dno_exp_ktc_dn_q5k",
    "dno_exp_ktc_dn_q6k",
    "dno_exp_ktc_dn_q5_1",
    "dno_exp_ktc_dn_q8_0",
];


/// K 系の型数 (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0)。
pub(crate) const KTC_SLOTS: usize = 5;


/// カーネル一式 (PTX を起動時に 1 回だけ読む)。
pub(crate) struct ExpKtcKernels {
    #[allow(dead_code)]
    module: CUmodule,
    gu: [CUfunction; KTC_SLOTS],
    dn: [CUfunction; KTC_SLOTS],
}


pub(crate) static EXP_KTC_KERNELS: OnceLock<Option<ExpKtcKernels>> = OnceLock::new();


/// task 416: この形を K 系の tensor core grouped GEMM で回せるか (純関数)。
///
/// - gate と up が同じ型 (1 回の起動で 2 枚を同時にデコードする)
/// - 256 要素ブロックの型 (Q4_K / Q5_K / Q6_K) は k が 256 の倍数、
///   32 要素ブロックの型 (Q5_1 / Q8_0) は 32 の倍数
/// - n が [`EXTC_BN`] の倍数 (端数タイル無し)
pub fn ktc_shapes_ok(
    gu_slot: Option<usize>,
    dn_slot: Option<usize>,
    ty_same: bool,
    k_gu: usize,
    k_dn: usize,
    n_gu: usize,
    n_dn: usize,
) -> bool {
    let k0 = crate::model::exp_pf::K_SLOT0;
    let ok = |slot: Option<usize>, k: usize| -> bool {
        match slot {
            Some(s) if (k0..k0 + 3).contains(&s) => k % 256 == 0,
            Some(s) if (k0 + 3..k0 + KTC_SLOTS).contains(&s) => k % 32 == 0,
            _ => false,
        }
    };
    ty_same
        && ok(gu_slot, k_gu)
        && ok(dn_slot, k_dn)
        && k_gu > 0
        && k_dn > 0
        && n_gu % EXTC_BN == 0
        && n_dn % EXTC_BN == 0
}


/// task 416: グループ GEMM のカーネル一式 (IQ 版 / K 版)。
#[derive(Clone, Copy)]
pub(crate) enum TcSet {
    Iq(&'static ExpTcKernels),
    K(&'static ExpKtcKernels),
}


thread_local! {
    /// task 233: 小さな表 (グループごとのポインタ表) を**非同期で**積むための
    /// pinned bounce と、層内の書き込みオフセット。
    ///
    /// 同期の `cuMemcpyHtoD` (pageable) は null ストリームの前のカーネルと
    /// ステージング DMA を待つので、グループごとに 3 回呼ぶと層あたり
    /// 300 ms 失う (lab t233c: `table=` が 14k で 243-410 ms/層)。
    pub(crate) static PF_TINY: std::cell::RefCell<(Option<PinMem>, usize)> =
        const { std::cell::RefCell::new((None, 0)) };
}


thread_local! {
    /// task 233: 直近のストリーミング層の内訳 (ns)。`[pump, wait, table, launch, accum]`
    /// 診断用 (DENNO_PF_LM_DBG の 1 行に出す)。
    ///
    /// task 296: `[5]` = スロット準備 (`pf_slot_prepare` 全体)、
    /// `[6]` = そのうち pinned バウンスの確保 (`cuMemHostAlloc`)、
    /// `[7]` = pinned を確保し直した回数 (数そのもの)。
    pub(crate) static PF_STREAM_T: std::cell::RefCell<[u64; 8]> = const { std::cell::RefCell::new([0; 8]) };
}


/// task 233: `&[u64]` をバイト列として見る (非同期コピー用)。
pub(crate) fn bytes_of(v: &[u64]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}


// ---------------------------------------------------------------------------
// task 333: ストリーミング経路が `Ok(None)` を返した理由 (診断)
// ---------------------------------------------------------------------------
//
// 層優先 dev MoE が `理由: stream` で落ちたとき、どの門で落ちたのかがログに
// 出ていなかった (task 328 は 3 ラウンドとも `理由: stream` だけで、
// ステージングの確保なのか上げスレッドなのかを切り分けられなかった)。
// 数値経路は変えず、層の処理と同じスレッドで理由だけを控える。

#[cfg(feature = "cuda")]
thread_local! {
    pub(crate) static PF_STREAM_WHY: std::cell::Cell<&'static str> = const { std::cell::Cell::new("") };
}


// ===========================================================================
// P17b: エンジン統合 (重み VRAM 常駐 + ピン留め/非同期パイプライン)
// ===========================================================================
//
// `quant::Linear` の int8 パス (apply_bt / apply_batch_bt / apply) から呼ばれ、
// `DENNO_CUDA=1` + feature `cuda` のとき int8 線形層を GPU にオフロードする。
// - 数値は CPU カーネルとビット一致 (同一カーネル = P17 の exactness 契約そのまま)
// - 重みは初回呼び出し時に VRAM へアップロードして常駐 (Quant8Linear のバッファ
//   ポインタをキーにしたレジストリ。ロード後の重みは不変であることが前提)
// - 固定オーバーヘッド削減: ピン留めホストメモリ + 非同期コピー/起動で
//   **1 線形層 = 1 同期** (P17 マイクロベンチでは起動+同期+コピーで約40µs/呼び出し)
// - 初期化失敗/実行時エラーは CPU パスへフォールバック (両者は同一仕様なので
//   出力は不変。実行時エラー後は二度と試さない)

/// ピン留めホストメモリ (非同期コピーの転送元/転送先)。
pub(crate) struct PinMem {
    ptr: *mut c_void,
    bytes: usize,
}


/// task 382: device から直接読み書きできる pinned の組 (ホスト側の見え方と
/// device 側の見え方は同じ 1 本のメモリ)。
pub(crate) struct MapMem {
    pin: PinMem,
    dev: CUdeviceptr,
}


/// pinned バウンスを使う転送サイズの下限 (既定) / 上限 (1 MiB ..= 256 MiB)。
pub(crate) const PF_PIN_MIN_BYTES: usize = 1 << 20;

pub(crate) const PF_PIN_MAX_BYTES: usize = 256 << 20;


/// task 358: 下限を env で動かす (`DENNO_PF_PIN_MIN_KB`、既定 1024 KiB)。
///
/// 1 MiB の床は task 318 が「小さい転送は写しの固定費が勝つ」として置いたもの。
/// 共有バウンスにした今は、decode の 10 KB D2H (pageable で 11 us) も
/// 同じ 1 本で速くなる可能性があるので、床を下げた条件を 1 つの COND で測れる
/// ようにしておく。
pub const PF_PIN_MIN_KB_ENV: &str = "DENNO_PF_PIN_MIN_KB";


/// task 518: decode の**小さな往復** (層ごとの 4 KiB .. 1 MiB) を pinned バウンスへ。
///
/// # なぜ要るか (2026-11-21)
///
/// UD-Q4_K_XL の warm 短 decode (12.7 ms/tok) は 4 段すべてが GPU の帯域律速
/// (seg の実測 gbs: gate 466 / up 473 / down 346 GB/s) で、ホスト側に残っているのは
/// **層ごとの小さな同期コピー**だけだった。1 トークンあたりの回数は
/// seg experts の prologue H2D (10.8 KB) と結果 D2H (10 KB)、GDN の xba H2D
/// (10.3 KB) と out D2H (10 KB)、full attention の KV ミラー H2D (65 KB) で、
/// 実測 (t518b の `[pf-pin]`) は decode で 131 回/token、3,826 の prefill で
/// 12,572 回・134.8 MB。pageable の 10 KB コピーは 11 us/回 (task 358 の実測)。
///
/// task 318 の 1 MiB の床は「pinned の確保そのものが高い」時代のもの。task 358 で
/// 共有 1 本 (grow-only) になったので、床を外したときの損得を測り直したら
/// **3,826 cold prefill が 2 ラウンドとも -4.6%**、14k cold が -7.0 / -3.9%、
/// cold の深い decode が -17〜-24%、warm 短 decode は -0.7% (2 ラウンド同値)、
/// 既定モデルは md5 全 SAME だった (t518a / t518b / t518c)。既定 on。
/// ここでは **1 MiB 未満だけ**を対象にする (prefill の大きな転送は
/// `DENNO_PF_PIN_STAGE` の判定のまま = 既存の数字を動かさない)。
pub const DEC_PIN_SMALL_ENV: &str = "DENNO_DEC_PIN_SMALL";


/// [`DEC_PIN_SMALL_ENV`] の解釈 (純関数)。**既定 on** (t518a/t518b で採用)。
///
/// 未設定・空文字は on、明示的な偽値 (`0` / `false` / `no` / `off`) だけ切る
/// (切り戻しは `DENNO_DEC_PIN_SMALL=0`)。
pub fn dec_pin_small_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None | Some("") => true,
        Some(s) => !matches!(
            s.to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off"
        ),
    }
}


/// [`DEC_PIN_SMALL_ENV`] が有効か (プロセスで 1 回だけ env を読む)。
pub(crate) fn dec_pin_small_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| dec_pin_small_from_value(std::env::var(DEC_PIN_SMALL_ENV).ok().as_deref()))
}


/// task 518: 小さな往復の窓の下限 (4 KiB 未満は写しの固定費が勝つ)。上限は
/// 既存の 1 MiB の床 (= それ以上は `DENNO_PF_PIN_STAGE` の担当)。
pub(crate) const PF_PIN_SMALL_MIN_BYTES: usize = 4 << 10;


/// 共有バウンスを `bytes` 以上に育てて、ロックを持ったまま `f` を実行する。
///
/// - ゲート off / 窓の外: `Ok(None)` (呼び出し側は pageable 経路へ)。
/// - 確保できない (pinned 不足など): 1 回だけ記録して `Ok(None)` (数値は同じ)。
///
/// ロックは転送が終わるまで持つ。転送は葉 (この中で別の転送を呼ばない) なので
/// 再入は起きない。並行に来た転送は直列になるが、pinned の 1 本は pageable より
/// 桁で速いので損にならない (lab の数字で確認する)。
pub(crate) fn with_pf_pin<T>(
    bytes: usize,
    f: impl FnOnce(*mut c_void) -> Result<T>,
) -> Result<Option<T>> {
    with_pf_pin_gate(bytes, false, f)
}


/// task 565: 全体ゲート (`DENNO_PF_PIN_STAGE`) を無視して窓の判定だけ使う版。
///
/// 残差の往復 (`DENNO_PF_RES_PIN`) のように「この転送だけ pinned にしたい」
/// 呼び出しのためのもの。窓 (1 MiB ..= 256 MiB) の判定は同じく効く。
pub(crate) fn with_pf_pin_forced<T>(
    bytes: usize,
    f: impl FnOnce(*mut c_void) -> Result<T>,
) -> Result<Option<T>> {
    with_pf_pin_gate(bytes, true, f)
}


// ---------------------------------------------------------------------------
// task 260: ホスト RAM tier 用の pinned な塊
// ---------------------------------------------------------------------------
//
// `DENNO_EXPERT_RAM_MB` の常駐先。CUDA の `cuMemHostAlloc` で確保した
// **ページロック済み**のホストメモリで、次の 2 つが狙い:
//
// 1. どの大きな読みにも追い出されない (page cache との違いはここだけ)。production の
//    実測では、別モデルを回したあとに IQ3_S へ戻すと page cache が抜けていて、
//    最初のリクエストが 149 ms/tok (平常 26) になっていた。
// 2. H2D の転送元にできる (pinned 52.6 GB/s / pageable 19.6 GB/s)。expert の
//    上げ直しで bounce コピーを挟まない経路 [`crate::cuda::PinnedHost`] を
//    そのまま `cuMemcpyHtoD` に渡せる。
//
// 確保に失敗したら `None` を返す (呼び出し側はヒープの塊へ落ちる)。失敗で
// リクエストを落とさない。
pub struct PinnedHost {
    ptr: *mut c_void,
    bytes: usize,
}


/// task 565: 戻しの H2D のリング 1 枠 (pinned と、その枠を読む DMA のイベント)。
pub(crate) struct UpPinSlot {
    mem: PinMem,
    ev: CUevent,
}


/// task 565: 戻しの H2D のリング (スレッド局所)。
pub(crate) struct UpPinRing {
    slots: Vec<Option<UpPinSlot>>,
    /// 次に使う枠の通し番号 (枠は `idx % UP_RING_N`)。
    calls: usize,
}


/// task 565: 通し番号 → 枠の添字 (純関数)。
pub(crate) fn up_ring_index(calls: usize) -> usize {
    calls % UP_RING_N
}


/// task 565: 戻しの H2D を「1 本ごとに待たない」モードにする (プロセス全体)。
///
/// [`restore_defer_flush_thread`] を各スレッドで呼ぶまで、積んだ DMA は
/// 走ったままになる。デバイスへ渡るバイト列は 1 本ずつの経路と同一。
static RESTORE_DEFER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// task 565: 遅延モードの入口 (`DENNO_PF_RESTORE_NOSYNC`)。
pub fn restore_defer_begin() {
    RESTORE_DEFER.store(true, std::sync::atomic::Ordering::Relaxed);
}


/// task 565: 遅延モードの終端。呼んだスレッドのリングの DMA を全部待つ。
pub fn restore_defer_end() {
    restore_defer_flush_thread();
    RESTORE_DEFER.store(false, std::sync::atomic::Ordering::Relaxed);
}


/// task 565: このスレッドが積んだリングの DMA を全部待つ (枠は残す)。
pub fn restore_defer_flush_thread() {
    let Ok(l) = lib() else {
        return;
    };
    let Some(es) = l.event_sync else {
        return;
    };
    UP_RING.with(|cell| {
        let ring = cell.borrow();
        for s in ring.slots.iter().flatten() {
            // 失敗しても次の段は同期コピー/カーネルなので、ここでは握る。
            unsafe {
                let _ = es(s.ev);
            }
        }
    });
}


/// task 565: 遅延モードか (戻しの中だけ真)。
pub(crate) fn restore_defer_on() -> bool {
    RESTORE_DEFER.load(std::sync::atomic::Ordering::Relaxed)
}


/// `DENNO_EXPERT_PIN_H2D=1`: expert の H2D を pinned staging 経由にするか。
///
/// 既定 **off**。lab t144d の実測では、この環境の pageable `cuMemcpyHtoD` の方が
/// 速かった (1 本 2.05 MB で pinned 0.34〜0.42 ms / pageable 0.27〜0.30 ms。
/// driver が内部で staging を持つため、自前で pinned へ写すと memcpy が余分)。
/// 転送するバイト列はどちらも同じ (モデルの数値には触らない)。
pub fn expert_pin_h2d() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("DENNO_EXPERT_PIN_H2D").ok().as_deref(),
            Some("1") | Some("true") | Some("on")
        )
    })
}


/// pinned staging 経由でホストのスライスを VRAM へ上げる (task 144)。
///
/// ページャブルなメモリからの `cuMemcpyHtoD` は driver が内部で staging するため、
/// lab t144b の実測では **2.4 GB/s** しか出なかった (expert 1 本 2.05 MB で
/// 0.86 ms)。pinned メモリへ 1 回コピーしてから H2D すると速い。
/// デバイスへ渡るバイト列は [`DevBuffer::from_slice`] と同一 (1 bit も変えない)。
pub fn from_slice_pinned<T: Copy>(data: &[T]) -> Result<DevBuffer> {
    let bytes = std::mem::size_of_val(data);
    let buf = DevBuffer::alloc_tagged(bytes.max(1), crate::mem_audit::TAG_EXPERTS)?;
    if bytes == 0 {
        return Ok(buf);
    }
    EXPERT_PIN.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().map(|p| p.bytes).unwrap_or(0) < bytes {
            *slot = Some(PinMem::alloc(bytes)?);
        }
        let pin = slot.as_ref().expect("確保済み");
        unsafe {
            // ホスト側の 1 回コピー (pinned は DMA 可能な常駐メモリ)。
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, pin.ptr as *mut u8, bytes);
            check(
                (lib()?.memcpy_htod)(buf.ptr, pin.ptr, bytes),
                "cuMemcpyHtoD (pinned)",
            )?;
        }
        dbg_htod(bytes);
        Ok(())
    })?;
    Ok(buf)
}


#[cfg(test)]
pub(crate) static PF_PIN_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


// ---------------------------------------------------------------------------
// task 209: router / shared が mix の出力をデバイスから直接読む
// ---------------------------------------------------------------------------
//
// 層ループは mix の出力 (`mixed_flat`) を D2H したあと、同じバイト列を
// router (`gate_inp` の F32 GEMM) と shared expert の GEMM のために **もう一度
// H2D する** (台帳 14,018: 6.89 GB + 6.89 GB)。デバイスには `pf_mix` の出力が
// そのまま残っているので、そこを読めば転送 2 本が消える。値は同じバイト列。

/// `DENNO_PF_MIXDEVIN` の解釈 (純関数)。**既定 on** (lab t209a で採用)、
/// 明示的な偽値だけ off。
pub(crate) fn pf_dev_in_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


#[cfg(test)]
static PF_DEV_IN_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用: ゲートを強制する (0 = env に戻す)。
#[cfg(test)]
pub(crate) fn pf_dev_in_set_for_test(v: u8) {
    PF_DEV_IN_FORCE.store(v, Ordering::Relaxed);
}


/// task 209: router / shared が mix の出力をデバイスから読むか
/// (`DENNO_PF_MIXDEVIN`、**既定 on**。lab t209a の devin が 3,826 -1.6% /
/// 14,018 -0.9%、both で -4.8% / -4.0%。`=0` で切り戻し)。
pub(crate) fn pf_dev_in_on() -> bool {
    #[cfg(test)]
    {
        match PF_DEV_IN_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_dev_in_from_value(std::env::var("DENNO_PF_MIXDEVIN").ok().as_deref()))
}


/// s167 (P6): 専用コピーストリーム (h2 の D2H を qkv/z GEMM と重ねる)。
/// driver オブジェクトはコンテキストに属するので Drop で明示的に破棄する
/// (スレッド終了時に解放。`stream_destroy` が無い driver ではリークするだけで、
/// コンテキスト破棄時に回収される)。
pub(crate) struct CopyStream(CUstream);


/// s167 (P6): コピーストリームの順序付け/完了待ちに使うイベント (RAII)。
pub(crate) struct CopyEvent(CUevent);


thread_local! {
    /// task 155: 既定 (null) ストリームの完了待ちに使うイベント (スレッドごとに 1 本)。
    pub(crate) static DEF_EV: std::cell::RefCell<Option<CopyEvent>> = const { std::cell::RefCell::new(None) };
}


/// task 155: `cuEventCreate` の `CU_EVENT_DISABLE_TIMING`。
pub(crate) const CU_EVENT_DISABLE_TIMING: u32 = 0x02;


/// task 155: 既定 (null) ストリームの完了だけを**イベント 1 本**で待つ。
///
/// [`CudaCtx::sync`] の `cuCtxSynchronize` はコンテキスト**全体**を待つので、
/// 重ね打ち用のサイドストリーム (shared expert の gate/up) まで待ってしまう。
/// ホストがその結果を要るのは層のずっと後 (shared の join) なので、ここでは
/// null ストリームに打ったイベントだけを待つ。発行済みのカーネル・引数・順序は
/// 1 つも変えない (変わるのは待つ範囲だけ)。
///
/// 実測 (GPU ラボ t155a、swift15-abl IQ3_S、14T、plain = TLINE なし、全ゲート on と
/// off の A/B): decode 46 → 42 ms/tok (短い) / 44 → 38 (3,954) / 47 → 41 (14,139)、
/// prefill +1.1%、md5 は 6 本とも SAME、errors 0。14,139 の台帳は
/// router 4.09 → 1.61 ms/tok (48 層で 52 us/層の同期固定費が消えた)、
/// experts 10.37 → 10.18。**この待ちはモデルの値を 1 ビットも変えない。**
///
/// t155c (TLINE=1 / TLINE_K=1 を両条件に): token 47.06 → 42.29 (3,954) /
/// 49.83 → 44.50 (14,139)。host_wait 25.09 → 22.32 / 29.11 → 26.39 (router の
/// wait が 2.68 → 0.97 / 2.72 → 0.99)、gpu_idle 28.61 → 24.17 / 27.33 → 22.32、
/// gpu_busy は 18.44 → 18.12 / 22.50 → 22.18 で**動かない** (仕事量は同じ)。
///
/// 戻り値 `false` = イベントが使えない。呼び出し側は従来どおり `ctx.sync()` に
/// 落ちること (待たずに進んではいけない)。
pub fn sync_default_stream() -> bool {
    let Ok(l) = lib() else { return false };
    let (Some(create), Some(record), Some(esync)) =
        (l.event_create, l.event_record, l.event_sync)
    else {
        return false;
    };
    DEF_EV.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let mut ev: CUevent = std::ptr::null_mut();
            if unsafe { create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return false;
            }
            *slot = Some(CopyEvent(ev));
        }
        let ev = slot.as_ref().expect("直前に作った").0;
        let t = crate::model::q4ledger::stamp();
        let ok = unsafe { record(ev, std::ptr::null_mut()) } == CUDA_SUCCESS
            && unsafe { esync(ev) } == CUDA_SUCCESS;
        if ok {
            crate::model::q4ledger::note_sync(t);
        }
        ok
    })
}


/// [`sync_default_stream`] が使えない環境用の包み (失敗したら `ctx.sync()`)。
///
/// `scoped == false` (`DENNO_DEC_OVERLAP=0` / `DENNO_ROUTER_FUSED=0`) は従来どおり
/// コンテキスト全体を待つ。
pub(crate) fn sync_default_or_ctx(ctx: &CudaCtx, scoped: bool) -> Result<()> {
    if scoped && sync_default_stream() {
        return Ok(());
    }
    ctx.sync()
}


/// task 155: `DENNO_DEC_OVERLAP` の値解釈 (純関数・テスト用)。
/// **既定 on**、`0` / `off` / `false` で expert の待ちを従来に戻す。
pub fn dec_overlap_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}


/// task 155: expert の完了待ちを null ストリームに絞るか (`DENNO_DEC_OVERLAP`)。
pub fn dec_overlap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dec_overlap_from_value(std::env::var("DENNO_DEC_OVERLAP").ok().as_deref()))
}


/// 1 スレッド分の再利用スクラッチ (grow-only)。scheduler ワーカーは単一スレッド
/// のため thread_local で競合なし。スレッド終了時の解放はベストエフォート
/// (コンテキストは static で生き続けるため、失敗してもリークはワーカー寿命分)。
pub(crate) struct Scratch {
    x: Option<DevBuffer>,   // [m*k] f32 入力
    aq: Option<DevBuffer>,  // [m*k] i8 量子化済み
    asc: Option<DevBuffer>, // [m] f32 ascale
    out: Option<DevBuffer>, // [m*n] f32 出力
    hx: Option<PinMem>,     // x の転送元 (ピン留め)
    ho: Option<PinMem>,     // out の転送先 (ピン留め)
    // P17k: FFN デバイス常駐 (maybe_gpu_ffn)
    act: Option<DevBuffer>,  // [m*ff] f32 swiglu 出力
    aq2: Option<DevBuffer>,  // [m*ff] i8 (act の量子化)
    asc2: Option<DevBuffer>, // [m] f32 (act の ascale)
    out2: Option<DevBuffer>, // [m*e] f32 (down 出力)
}


thread_local! {
    pub(crate) static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        x: None, aq: None, asc: None, out: None, hx: None, ho: None,
        act: None, aq2: None, asc2: None, out2: None,
    });
}


// ---------------------------------------------------------------------------
// P58-3: prefill スクラッチの arena 化
//
// 背景 (与えられた実測): 本番 serve の OOM は prefill の途中で起きる —
//   `CUDA cuMemAlloc が失敗: error 2 (要求 779 MiB / 空き 315 MiB / 総 32110 MiB)`
//   → `chat failed: prefill residual chain: qkv chain failed`
// 779 MiB は `CH_QKV` (57,344 B/行 = (q 12,288 + k 1,024 + v 1,024) 列 x 4 B、
// n = 14,245..14,262 で 779 MiB) と一致する (CH_GA は 1,892.8 MiB・CH_ACT は
// 946.4 MiB なので一致しない = 失敗したのは qkv の出力 1 本)。つまり「直前の
// リクエストで確保した小さめの領域を解放した直後に、より大きい領域を要求する」
// 形で、個別の
// cuMemAlloc / cuMemFree を繰り返すことの (a) 断片化 (b) 失敗経路の解放漏れが
// 効いている。
//
// 方針: prefill の一時バッファを 1 本の arena から切り出す。
//   - arena は**チャンクのリスト**。チャンクは確保後に移動しない (成長時に
//     内容をコピーしない) ので、既に配ったポインタは常に有効。
//   - 予約は (スレッド ID, キー) ごとに保持し、同じキーで同じサイズ以下なら
//     **同じオフセット**を返す。リクエストごとの cuMemAlloc / cuMemFree が消える。
//   - 数値は不変: どのバッファに何を書くか・カーネル引数・発行順・丸め順は
//     1 つも変えない。変わるのは「同じ大きさの領域をどこから取るか」だけ。
//   - リクエスト境界 (`pf_scratch_begin`) では、中身が全て死んでいることを
//     利用して**古いチャンクを解放してから**新しいサイズを確保する
//     (ピークが「旧 + 新」にならない)。前回の 1 行あたり使用量から今回の
//     必要量を推定するので、プロンプト長が伸びた直後の 1 回もここで吸収できる。
//   - `DENNO_SCRATCH_ARENA=0` で従来の個別確保へ戻る (比較・切り戻し用)。
//   - arena の確保に失敗したら `None` を返し、呼び出し側が従来の
//     `DevBuffer::alloc` を使う (fail-open・数値不変)。
// ---------------------------------------------------------------------------

/// arena の成長単位 (1,024 MiB)。
pub(crate) const ARENA_CHUNK: usize = 1 << 30;

/// チャンク内オフセットのアライン。cuMemAlloc の返す境界 (256 B) に合わせる。
pub(crate) const ARENA_ALIGN: usize = 256;

/// 予約キー: デバイス常駐チェーンのスロット (`CH_*`)。
pub(crate) const ARENA_KEY_CHAIN: u32 = 0x0100_0000;

/// 予約キー: `q4_dispatch_tile` の入力 staging。
pub(crate) const ARENA_KEY_Q4_X: u32 = 0x0200_0001;

/// 予約キー: `q4_dispatch_tile` の出力 staging。
pub(crate) const ARENA_KEY_Q4_OUT: u32 = 0x0200_0002;


/// [`arena_enabled`] の純粋な判定 (`DENNO_SCRATCH_ARENA`)。未設定 = 有効。
pub(crate) fn parse_scratch_arena(raw: Option<&str>) -> bool {
    !matches!(raw.map(|v| v.trim()), Some("0") | Some("false") | Some("off"))
}


/// arena を使うか (既定 ON・`DENNO_SCRATCH_ARENA=0` で従来の個別確保)。
pub(crate) fn arena_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_scratch_arena(std::env::var("DENNO_SCRATCH_ARENA").ok().as_deref()))
}


/// `DENNO_SCRATCH_ARENA_DBG=1` で成長・境界の 1 行ログを出す (恒久診断・
/// 未指定時 no-op)。
pub(crate) fn arena_dbg() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_SCRATCH_ARENA_DBG").ok().as_deref() == Some("1"))
}


/// arena の確保計画 (GPU を持たない純粋な部分 = ホストだけで検証できる)。
#[derive(Default)]
pub(crate) struct ArenaPlan {
    /// チャンクごとの (容量, 使用済み)。確保順に埋める。
    chunks: Vec<(usize, usize)>,
    /// (スレッド ID, キー) → (チャンク index, オフセット, 予約バイト数)。
    slots: std::collections::HashMap<(u8, u32), (usize, usize, usize)>,
    /// いまのリクエストで切り出した合計 (推定用)。
    cur_bytes: usize,
    /// 過去のリクエストで切り出した合計の最大 (次回の事前確保量の推定用)。
    high_water: usize,
}


/// arena の実体 (チャンクの `DevBuffer` と計画)。
pub(crate) struct ArenaState {
    plan: ArenaPlan,
    /// `plan.chunks` と同順。チャンクは移動しない (再確保時にコピーしない)。
    bufs: Vec<DevBuffer>,
    /// チャンクを足した回数 (診断)。
    grows: usize,
    /// 切り出した回数 (診断)。
    carves: usize,
    /// 直近のリクエストの行数 (事前確保量の推定用)。
    last_rows: usize,
}


pub(crate) static ARENA: OnceLock<std::sync::Mutex<ArenaState>> = OnceLock::new();

/// スレッド ID の採番 (thread_local のスロットを (tid, key) で区別するため)。
static ARENA_TID_NEXT: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);


/// このスレッドの arena ID (0..255)。スレッドごとに 1 回だけ採番する。
pub(crate) fn arena_tid() -> u8 {
    thread_local! {
        static TID: u8 = (ARENA_TID_NEXT
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed) & 0xff) as u8;
    }
    TID.with(|t| *t)
}


fn arena_state() -> &'static std::sync::Mutex<ArenaState> {
    ARENA.get_or_init(|| {
        std::sync::Mutex::new(ArenaState {
            plan: ArenaPlan::default(),
            bufs: Vec::new(),
            grows: 0,
            carves: 0,
            last_rows: 0,
        })
    })
}


/// arena から `bytes` バイトを切り出し、**非所有ビュー**として返す。
/// `None` = arena 無効・チャンク確保失敗 (呼び出し側は従来の `DevBuffer::alloc`)。
pub(crate) fn arena_alloc(key: u32, bytes: usize, tag: &'static str) -> Option<DevBuffer> {
    if !arena_enabled() {
        return None;
    }
    let tid = arena_tid();
    let mut g = arena_state().lock().ok()?;
    let (ci, off) = match g.plan.reserve(tid, key, bytes) {
        Ok(v) => v,
        Err(want) => {
            let want = want.max(ARENA_CHUNK);
            let buf = match DevBuffer::alloc_tagged(want, crate::mem_audit::TAG_ARENA) {
                Ok(b) => b,
                Err(e) => {
                    if arena_dbg() {
                        eprintln!("scratch arena: チャンク確保に失敗 ({want} B): {e}");
                    }
                    return None;
                }
            };
            if arena_dbg() {
                eprintln!(
                    "scratch arena: チャンク {want} B を追加 (key=0x{key:08x} 要求 {bytes} B)"
                );
            }
            g.bufs.push(buf);
            g.plan.add_chunk(want);
            g.grows += 1;
            g.plan.reserve(tid, key, bytes).ok()?
        }
    };
    g.carves += 1;
    let base = g.bufs.get(ci)?.ptr();
    Some(DevBuffer::view(base + off as u64, bytes))
}


/// arena のビューを保持している thread_local スロットを落とす。
///
/// **呼び出し位置の契約**: [`pf_scratch_begin`] の中だけで呼ぶ。ここで落とす
/// スロットは「使う直前に必ず `chain_ensure_buf` / `ensure_arena` で確保される」
/// ものだけで、リクエスト境界の外では内容を保持しない (CH_PF_RES の残差も、
/// 境界の直後に `chain_pf_res_upload` で入れ直す)。
pub(crate) fn drop_arena_views() {
    CHAIN.with(|cell| {
        let mut s = cell.borrow_mut();
        for b in s.bufs.iter_mut() {
            *b = None;
        }
    });
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        s.x = None;
        s.out = None;
    });
}


/// task 149: `DENNO_SCRATCH_FREE` の解釈。既定 **off** (明示的に 1/true/on で有効)。
///
/// lab t149c の flash では arena が 1 度も確保されず (tag `scratch arena` が
/// 表に出ない)、解放しても 1 MiB も変わらなかった。qwen35 側の経路には効く
/// 可能性があるが未測定なので、既定では従来どおり保持する。
pub(crate) fn parse_scratch_free(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|v| v.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}


/// task 149: `DENNO_SCRATCH_FREE` が有効か (プロセスで 1 回読む)。
pub(crate) fn scratch_free_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_scratch_free(std::env::var("DENNO_SCRATCH_FREE").ok().as_deref()))
}


/// arena の現在値 (診断・テスト用): (チャンク本数, 総容量, 切り出し回数, 成長回数)。
pub fn scratch_arena_stats() -> (usize, usize, usize, usize) {
    match ARENA.get().and_then(|a| a.lock().ok()) {
        Some(g) => (g.bufs.len(), g.plan.capacity(), g.carves, g.grows),
        None => (0, 0, 0, 0),
    }
}


/// s141: 再利用領域 `dst` の先頭 `len` 要素へ D2H し、その部分を**借りて**返す。
/// `DevBuffer::to_vec` と転送は同一 (同じ `cuMemcpyDtoH` を同じ長さで 1 回) で、
/// 違いは「確保しない」ことだけ。容量が足りないときだけ `reserve_exact` で伸ばす
/// (`mem::take` のように領域を奪い返さない = 次の層でも同じ領域を使う)。
pub(crate) fn d2h_raw<'a>(dst: &'a mut Vec<f32>, ptr: CUdeviceptr, len: usize) -> Option<&'a [f32]> {
    unsafe {
        reserve_exact_grow(dst, len);
        let p = dst.as_mut_ptr() as *mut c_void;
        let lib = lib().ok()?;
        if check((lib.memcpy_dtoh)(p, ptr, len * 4), "cuMemcpyDtoH").is_err() {
            return None;
        }
        dst.set_len(len);
    }
    Some(&dst[..])
}


/// s141 (Task B7): `dst` を「長さ `len` を書ける」状態にする。**伸ばすだけ** —
/// 容量が足りていれば確保しない (これが「層ごとに確保しない」の実体)。
/// 実転送 (`d2h_raw`) とテストが同じ関数を通るので、テストは写しではなく
/// 本物の確保規則を検証している。
pub(crate) fn reserve_exact_grow(dst: &mut Vec<f32>, len: usize) {
    dst.clear();
    if len > dst.capacity() {
        dst.reserve_exact(len - dst.len());
    }
}


static GPU_CTX: OnceLock<Option<CudaCtx>> = OnceLock::new();

pub(crate) static GPU_BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// 集計用 (gpu_budget): 重みの失敗などで GPU 経路が恒久停止しているか。
/// 読み取りのみ・状態は変えない。
pub(crate) fn gpu_broken_flag() -> bool {
    GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed)
}


#[cfg(test)]
pub(crate) static GPU_FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// CUDA 経路が有効か (`DENNO_CUDA=1` かつ初期化が壊れていない)。
///
/// テストから「逐次経路が GPU カーネルか CPU 参照か」を判定するのにも使う。
pub fn cuda_enabled() -> bool {
    if GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    #[cfg(test)]
    if GPU_FORCE.load(std::sync::atomic::Ordering::Relaxed) {
        return true;
    }
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(|| {
        matches!(
            std::env::var("DENNO_CUDA").as_deref(),
            Ok("1") | Ok("true") | Ok("on")
        )
    })
}


/// テスト用: GPU 関連テストの直列化ロック (P17j)。
/// `set_gpu_attention_enabled` でグローバルフラグを差し替えるテストと、フラグ
/// 状態に依存するテスト (attention ON/OFF でビット一致/近似一致が変わる) が
/// 並列実行で競合し、片方の失敗が GPU_BROKEN 経由で他テストにも連鎖していた
/// (P17g から潜在・テスト数増加で顕在化)。フラグを扱う/依存するテストは
/// このロックを取得する。
#[cfg(test)]
#[doc(hidden)]
pub fn test_lock() -> &'static std::sync::Mutex<()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
}


/// 共有 CudaCtx (遅延初期化)。無効 or 初期化失敗なら None (CPU フォールバック)。
pub(crate) fn shared_ctx() -> Option<&'static CudaCtx> {
    GPU_CTX
        .get_or_init(|| match CudaCtx::new() {
            Ok(ctx) => {
                tracing::info!("CUDA int8 offload enabled: {}", ctx.info());
                Some(ctx)
            }
            Err(e) => {
                tracing::warn!("CUDA int8 offload unavailable, using CPU int8: {e:#}");
                // テスト/ベンチは tracing を出さないので、1 回だけ stderr にも書く
                // (「緑 = 経路が発動した証拠ではない」の取りこぼしを防ぐ)。
                eprintln!("[cuda] CudaCtx::new 失敗 → CPU へ: {e:#}");
                None
            }
        })
        .as_ref()
}


/// このスレッドの current context を共有 CudaCtx にする (必要なら初期化もする)。
///
/// CUDA の current context は**スレッドローカル**なので、推論スレッド以外
/// (起動時の事前ロードなど) から `DevBuffer` を確保するには、そのスレッドで
/// 1 回これを呼ぶ必要がある。呼ばないと `cuMemAlloc` が error 201
/// (INVALID_CONTEXT) で落ちる (P17c と同じ穴)。
///
/// 戻り値: 設定できたら true。CUDA が無効・初期化失敗なら false。
pub fn make_current_on_this_thread() -> bool {
    match shared_ctx() {
        Some(ctx) => match ctx.make_current() {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("CUDA cuCtxSetCurrent failed (このスレッド): {e:#}");
                false
            }
        },
        None => false,
    }
}


/// int8 重みの VRAM レジストリ (キー = Quant8Linear.uid)。
/// uid は Linear 構築時に採番されるため、Vec アドレスの再利用で別内容の重みが
/// 同一キーに現れることはない。Clone は uid を共有するが内容が同一なので安全。
/// エントリは解放しない (実運用は 1 モデル = 約 2GB、テストは微小)。
pub(crate) struct GpuW {
    w: DevBuffer,
    sc: DevBuffer,
    k: usize,
    n: usize,
    /// s74: `w` の行数 (n を GEMM_PAD_N の倍数に切り上げた値)。dno_gemm_i8 は
    /// j 方向を 256 列単位で処理するため、重みは 256 行に 0 パディングして渡す。
    /// カーネル自身は `n` (実列数) で epilogue を打ち切るので `sc` は n 本のまま。
    npad: usize,
}


/// s74: `gemm_pad_n(n)` を通した**パディング済み**行数 (`w` の行数)。
pub(crate) const GEMM_PAD_N: usize = 256;

/// s74: dno_gemm_i8 のタイル寸法 (kernels_int8.cu の GEMM_TILE / GEMM_TILE_N と
/// 同期すること)。i 方向 128 行 × j 方向 256 列・block 16x16 (1 thread 8x16 出力)。
pub(crate) const GEMM_TILE: usize = 128;

pub(crate) const GEMM_TILE_N: usize = 256;

/// s75: カーネルのブロック形状 (kernels_int8.cu の GEMM_BX / GEMM_BY)。
/// 1 thread 8 行 x 16 列・block 16x16 = 256 thread なので
/// `GEMM_BY * 8 == GEMM_TILE` / `GEMM_BX * 16 == GEMM_TILE_N` が成り立つ。
/// s74 は (16,4) = 64 thread で 32 行しか覆えていなかった (タイル 128 行と
/// 食い違い、残り 96 行が未書き込みのまま返っていた)。
pub(crate) const GEMM_BX: u32 = 16;

pub(crate) const GEMM_BY: u32 = 16;


pub(crate) fn weights_for(uid: u64, qw: &[i8], scale: &[f32], k: usize, n: usize) -> Result<&'static GpuW> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static REG: OnceLock<Mutex<HashMap<u64, Box<GpuW>>>> = OnceLock::new();
    let reg = REG.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().expect("weights registry poisoned");
    if !map.contains_key(&uid) {
        let w = DevBuffer::from_slice(&gemm_pad_apply(qw, k, n))?;
        let sc = DevBuffer::from_slice(scale)?;
        tracing::info!(
            "CUDA weight upload (uid={uid}): k={k} n={n} ({:.1} MB)",
            (k * n + n * 4) as f64 / 1e6
        );
        map.insert(uid, Box::new(GpuW { w, sc, k, n, npad: gemm_pad_n(n) }));
    }
    let e = map.get(&uid).expect("just inserted");
    debug_assert_eq!((e.k, e.n), (k, n), "uid 重複: 同一 uid に異なる形状");
    // &'static 化: REG は static、Box の中身は移動しない
    Ok(unsafe { std::mem::transmute::<&GpuW, &'static GpuW>(e.as_ref()) })
}


/// GPU オフロードする最大の m (行数)。decode / バッチ decode (m≤閾値) のみ
/// オフロードし、prefill (m が大きい GEMM) は CPU のまま — GEMV 型カーネルは
/// 行数に比例して重み再読みが増えるため m が大きいと CPU sgemm に負ける
/// (P17b 実測: m=128 で 0.39x)。閾値は `DENNO_CUDA_MAX_M` で変更可能。
pub(crate) fn max_m() -> usize {
    static MAX_M: OnceLock<usize> = OnceLock::new();
    *MAX_M.get_or_init(|| {
        std::env::var("DENNO_CUDA_MAX_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}


// s57 (T4): PROFILE の実行先ラベル用に「直近の GPU 線形呼び出しが実際に GPU で
// 走ったか」を記録する thread-local。**計測のみ** — `maybe_gpu_linear` が結果を
// 返す直前に書くだけで、数値・分岐・GPU 状態には触れない。値は次の線形呼び出し
// まで古いまま (呼び出し側は同じ区間の計測にしか使わない)。
#[cfg(feature = "cuda")]
thread_local! {
    pub(crate) static LINEAR_RAN_ON_GPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// s61: 区間内で「一度でも GPU 線形が走ったか」の集積フラグ。ネストした
    /// `Probe` を許すため深さも持つ (最深の scope が閉じたときに外側へ引き継ぐ)。
    pub(crate) static LINEAR_SCOPE: (std::cell::Cell<u32>, std::cell::Cell<bool>) =
        const { (std::cell::Cell::new(0), std::cell::Cell::new(false)) };
}


/// 直近の `maybe_gpu_linear` が GPU で走りきったか (PROFILE の実行先ラベル用)。
/// PROFILE 無効時はプロファイラ側が参照しないため、この記録は無害な no-op に
/// 近い (Cell 書き込み 1 回)。
#[cfg(feature = "cuda")]
#[inline]
pub fn last_linear_ran_on_gpu() -> bool {
    LINEAR_RAN_ON_GPU.with(|c| c.get())
}


#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn set_last_linear_ran_on_gpu(v: bool) {
    LINEAR_RAN_ON_GPU.with(|c| c.set(v));
    // s61: 開いている区間 (Probe) があれば OR で集積する (計測のみ)
    if v {
        LINEAR_SCOPE.with(|(depth, any)| {
            if depth.get() > 0 {
                any.set(true);
            }
        });
    }
}


/// s61 (T5a 撤回の知見): 区間の実行先ラベルを「区間内の **いずれか** の線形
/// 呼び出しが GPU で走ったか」で取るためのスコープ。
///
/// `last_linear_ran_on_gpu()` は **直近 1 回** の `maybe_gpu_linear` しか
/// 見ないため、1 区間で複数の線形を呼ぶバケット (qkv = q/k/v の 3 本、
/// gate_up = gate + up) では最後の 1 本の結果に引きずられる (v が GEMV で
/// GPU、k が CPU のような場合に cb が誤る)。`Probe` は区間中に
/// `maybe_gpu_linear` が **OR で** 記録した値を読み、`Drop` で必ず閉じる。
///
/// **計測のみ** — 記録するのは `Cell<bool>` だけで、分岐・数値・GPU 状態には
/// 一切触れない。PROFILE 無効時も呼ばれるが cost は Cell 2 回。
#[cfg(feature = "cuda")]
pub struct Probe {
    _not_send: std::marker::PhantomData<*const ()>,
}


/// s61 (T5a 撤回): GPU 線形層の入口 (`maybe_gpu_linear`) が呼ばれた回数の
/// **診断用カウンタ**。既定経路が入口を迂回していないことを GPU 無し環境でも
/// 検査できるようにする (083218d はゲート未指定で `matmul_bt_i8` を直呼びし、
/// この入口に一度も到達しなかった)。
///
/// 値の読み書きは Relaxed の加算 1 回のみで、分岐・戻り値・数値経路には触れない。
pub(crate) static LINEAR_SEAM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);


/// s61: `maybe_gpu_linear` の入口に到達した累計回数 (回帰テスト用)。
#[cfg(feature = "cuda")]
#[inline]
pub fn linear_seam_calls() -> u64 {
    LINEAR_SEAM_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}


// ---------------------------------------------------------------------------
// s166 (P5): decode 1 ステップの launch / copy 件数 (DENNO_DECODE_DBG=1)
//
// nsys を使わずに「1 ステップに何発カーネルを投げ、何回・何バイト転送したか」を
// 数える。既定は OFF (`DENNO_DECODE_DBG` 未設定) で、各サイトに分岐 1 回が
// 増えるだけ。数値経路・発行順・引数には一切触れない (観測のみ)。
//
// 使い方: `DENNO_DECODE_DBG=1` で serve を起動すると、decode 1 ステップごとに
//   [DECODE_DBG] step=3 launch=337 h2d=2 d2h=33
//     launch=[dno_gemv_q4_0c4=337,dno_add_rmsnorm=129,...]
//     h2d=[20480x1,512x1] d2h=[2048x32,606208x1]
// を stderr に 1 行 (続けて内訳 2 行) 出す。件数は「その step の begin〜end」で
// 数えるので、prefill で積んだ分は次の begin で捨てる (decode の数字に混ぜない)。
// ---------------------------------------------------------------------------
fn decode_dbg_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_DECODE_DBG").ok().as_deref() == Some("1"))
}


/// (キー, 件数) の集計。キーは CUfunction かバイト数。種類は 20 程度なので
/// 線形走査で足りる (1 ステップ 500 件程度)。
pub(crate) type DbgTally = Vec<(usize, u64)>;


fn dbg_bump(t: &mut DbgTally, key: usize) {
    for e in t.iter_mut() {
        if e.0 == key {
            e.1 += 1;
            return;
        }
    }
    t.push((key, 1));
}


#[derive(Default)]
pub(crate) struct DecodeDbg {
    steps: u64,
    launches: DbgTally,
    h2d: DbgTally,
    d2h: DbgTally,
    /// s169 (P10): Q4 GEMV の (n, C) 別の件数。key = (n << 4) | C。
    gemv: DbgTally,
}


thread_local! {
    pub(crate) static DECODE_DBG: std::cell::RefCell<DecodeDbg> =
        std::cell::RefCell::new(DecodeDbg {
            steps: 0,
            launches: Vec::new(),
            h2d: Vec::new(),
            d2h: Vec::new(),
            gemv: Vec::new(),
        });
}


/// カーネル 1 発を数える (`CudaCtx::launch_async` の入口から)。
#[inline]
pub(crate) fn dbg_launch(f: CUfunction) {
    if decode_dbg_on() {
        DECODE_DBG.with(|d| dbg_bump(&mut d.borrow_mut().launches, f as usize));
    }
}


/// s169 (P10): Q4 GEMV 1 発の (出力行数 n, 列ブロック幅 C) を数える。
/// launch 形状の選択 (`DENNO_GEMV_SHAPE`) がどの投影に効いたかを、GPU 無しでも
/// 実機の stderr から読めるようにするための観測のみ (OFF は分岐 1 回)。
#[inline]
pub(crate) fn dbg_gemv_shape(n: usize, c: usize) {
    if decode_dbg_on() {
        DECODE_DBG.with(|d| dbg_bump(&mut d.borrow_mut().gemv, (n << 4) | (c & 0xf)));
    }
}


// ---------------------------------------------------------------------------
// task 214: 上げ直し専用の H2D ストリーム (スレッド単位)
// ---------------------------------------------------------------------------

thread_local! {
    /// このスレッドの H2D を出すストリームと、完了待ち用のイベント。
    /// [`enable_thread_upload_stream`] が成功したスレッドだけ `Some`。
    pub(crate) static UP_STREAM: std::cell::RefCell<Option<(CUstream, CUevent)>> =
        const { std::cell::RefCell::new(None) };
}


/// [`UP_STREAM`] の値を返す (未設定なら `None` = 従来の null ストリーム)。
pub(crate) fn thread_up_stream() -> Option<(CUstream, CUevent)> {
    UP_STREAM.with(|c| *c.borrow())
}


/// task 214: このスレッドの H2D を**専用ストリーム**で出し、イベントで待つ。
///
/// 背景の上げ直しスレッド ([`crate::model::exp_up_worker`]) の入口で 1 回だけ呼ぶ。
/// null ストリーム (メインスレッドのカーネル) と直列にならないので、
/// 上げ直しが GPU の帯を塞がない。**デバイスへ渡るバイト列は同じ**。
///
/// 成功したら true。ドライバのシンボルが無い / 生成に失敗したら false で、
/// そのスレッドは従来の同期 `cuMemcpyHtoD` を使う (fail closed)。
pub fn enable_thread_upload_stream() -> bool {
    UP_STREAM.with(|c| {
        if c.borrow().is_some() {
            return true;
        }
        let Ok(l) = lib() else {
            return false;
        };
        let (Some(create), Some(ec), Some(destroy)) =
            (l.stream_create, l.event_create, l.stream_destroy)
        else {
            return false;
        };
        if l.event_record.is_none() || l.event_sync.is_none() {
            return false;
        }
        const CU_STREAM_NON_BLOCKING: u32 = 0x01;
        const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
        let mut st: CUstream = std::ptr::null_mut();
        if unsafe { create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
            return false;
        }
        let mut ev: CUevent = std::ptr::null_mut();
        if unsafe { ec(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
            unsafe {
                let _ = destroy(st);
            }
            return false;
        }
        *c.borrow_mut() = Some((st, ev));
        true
    })
}


/// H2D 1 回を数える (バイト数で分類)。decode 経路の転送サイトから呼ぶ。
#[inline]
pub fn dbg_htod(bytes: usize) {
    if decode_dbg_on() {
        DECODE_DBG.with(|d| dbg_bump(&mut d.borrow_mut().h2d, bytes));
    }
}


/// D2H 1 回を数える。
#[inline]
pub fn dbg_d2h(bytes: usize) {
    if decode_dbg_on() {
        DECODE_DBG.with(|d| dbg_bump(&mut d.borrow_mut().d2h, bytes));
    }
}


/// CUfunction → カーネル名 (既知のものだけ・未知は "unknown")。
fn dbg_kernel_name(p: CUfunction) -> &'static str {
    let Some(ctx) = chain_ctx() else {
        return "unknown";
    };
    if p == ctx.f_quant {
        return "dno_quant_i8";
    }
    if p == ctx.f_mv {
        return "dno_gemv_i8";
    }
    if p == ctx.f_gemm {
        return "dno_gemm_i8";
    }
    if p == ctx.f_mv_q4 {
        return "dno_gemv_q4_0";
    }
    if p == ctx.f_gemm_q4 {
        return "dno_gemm_q4_0";
    }
    if let Some(f) = ctx.f_mv_q4_mr {
        if p == f {
            return "dno_gemv_q4_0mr";
        }
    }
    if let Some(fs) = ctx.f_mv_q4_c {
        const NAMES: [&str; 3] = ["dno_gemv_q4_0c2", "dno_gemv_q4_0c4", "dno_gemv_q4_0c8"];
        for (i, n) in NAMES.iter().enumerate() {
            if p == fs[i] {
                return n;
            }
        }
    }
    if p == ctx.f_swiglu {
        return "dno_swiglu";
    }
    if p == ctx.f_swiglu_int {
        return "dno_swiglu_int";
    }
    if p == ctx.f_addnorm {
        return "dno_add_rmsnorm";
    }
    if p == ctx.f_expf {
        return "dno_expf";
    }
    if p == ctx.f_attn {
        return "dno_attention";
    }
    if p == ctx.f_attn_pf {
        return "dno_attention_prefill";
    }
    if p == ctx.f_rope {
        return "dno_rope";
    }
    if p == ctx.f_rope_pf {
        return "dno_rope_pf";
    }
    if p == ctx.f_rope_b {
        return "dno_rope_b";
    }
    if p == ctx.f_gdn {
        return "dno_gdn_scan";
    }
    if p == ctx.f_gdn_chunk_pre {
        return "dno_gdn_chunk_pre";
    }
    if let Some(f) = ctx.f_gdn_chunk_pre_sm {
        if p == f {
            return "dno_gdn_chunk_pre_sm";
        }
    }
    if p == ctx.f_gdn_chunk_state {
        return "dno_gdn_chunk_state";
    }
    if p == ctx.f_gdn_chunk_scan {
        return "dno_gdn_chunk_scan";
    }
    if let Some(f) = ctx.f_gdn_chunk_scan_hoist {
        if p == f {
            return "dno_gdn_chunk_scan_hoist";
        }
    }
    if let Some(q) = q35_fns() {
        if p == q.f_matvec_f32 {
            return "dno_matvec_f32";
        }
        if p == q.f_qknorm_rope {
            return "dno_qknorm_rope_pi";
        }
        if p == q.f_gate {
            return "dno_attn_gate_mul";
        }
    }
    "unknown"
}


/// 件数の多い順に `name=count` を並べた文字列 (内訳表示用)。
fn dbg_fmt_launch(t: &DbgTally) -> String {
    let mut v: Vec<(&'static str, u64)> = t
        .iter()
        .map(|&(f, c)| (dbg_kernel_name(f as CUfunction), c))
        .collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    v.iter()
        .map(|(n, c)| format!("{n}={c}"))
        .collect::<Vec<_>>()
        .join(",")
}


/// `bytes×count` をバイト数の大きい順に並べた文字列。
fn dbg_fmt_bytes(t: &DbgTally) -> String {
    let mut v: Vec<(usize, u64)> = t.to_vec();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.iter()
        .map(|(b, c)| format!("{b}x{c}"))
        .collect::<Vec<_>>()
        .join(",")
}


/// s169 (P10): `n/C × count` を n の大きい順に並べた文字列 (Q4 GEMV の形状内訳)。
pub(crate) fn dbg_fmt_gemv(t: &DbgTally) -> String {
    let mut v: Vec<(usize, u64)> = t.to_vec();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.iter()
        .map(|(k, c)| format!("{}/c{}x{}", k >> 4, k & 0xf, c))
        .collect::<Vec<_>>()
        .join(",")
}


/// ステップ開始: 前のステップ (や prefill) で積んだ分を捨てる。
pub(crate) fn decode_dbg_reset() {
    if !decode_dbg_on() {
        return;
    }
    DECODE_DBG.with(|d| {
        let mut d = d.borrow_mut();
        d.launches.clear();
        d.h2d.clear();
        d.d2h.clear();
        d.gemv.clear();
    });
}


/// ステップ末尾: 1 行 (＋内訳 2 行) を stderr に出す。観測のみ。
pub(crate) fn decode_dbg_flush() {
    if !decode_dbg_on() {
        return;
    }
    DECODE_DBG.with(|d| {
        let mut d = d.borrow_mut();
        if d.launches.is_empty() && d.h2d.is_empty() && d.d2h.is_empty() {
            return; // このステップは 1 発も投げていない (CPU 経路) — 出さない
        }
        d.steps += 1;
        let n_l: u64 = d.launches.iter().map(|e| e.1).sum();
        let n_h: u64 = d.h2d.iter().map(|e| e.1).sum();
        let n_d: u64 = d.d2h.iter().map(|e| e.1).sum();
        let lb = d.h2d.iter().map(|e| e.0 as u64 * e.1).sum::<u64>();
        let db = d.d2h.iter().map(|e| e.0 as u64 * e.1).sum::<u64>();
        eprintln!(
            "[DECODE_DBG] step={} launch={} h2d={} ({} B) d2h={} ({} B)",
            d.steps, n_l, n_h, lb, n_d, db
        );
        eprintln!("[DECODE_DBG]   launch=[{}]", dbg_fmt_launch(&d.launches));
        eprintln!(
            "[DECODE_DBG]   h2d=[{}] d2h=[{}]",
            dbg_fmt_bytes(&d.h2d),
            dbg_fmt_bytes(&d.d2h)
        );
        // s169 (P10): Q4 GEMV の (n, C) 内訳。形状選択が効いた投影がここで判る。
        if !d.gemv.is_empty() {
            eprintln!("[DECODE_DBG]   gemv=[{}]", dbg_fmt_gemv(&d.gemv));
        }
        d.launches.clear();
        d.h2d.clear();
        d.d2h.clear();
        d.gemv.clear();
    });
}


/// s123: Q4_0 ネイティブ重みの VRAM レジストリ (キー = `Q4Linear.uid`)。
///
/// SOA で 2 本のバッファに分けて置く: `nib` = 1 ブロック 16B (uint4) を
/// n*nb 個並べたもの、`sc` = 1 ブロック 1 本の f32 スケール。
/// 1 要素あたり 4.5 bit (int8 経路の半分以下) で済む。
pub(crate) struct GpuWq4 {
    nib: DevBuffer,
    sc: DevBuffer,
    k: usize,
    n: usize,
}


/// s178 (P59): `DENNO_Q4_DEDUP` の解釈 (純関数・テスト可能)。
/// `parse_q4_chain` と同じ規約で **`"0"` だけが off** (未設定・空・その他は on)。
pub(crate) fn parse_q4_dedup(raw: Option<&str>) -> bool {
    raw != Some("0")
}


/// s123: [n,k] の Q4_0 ブロック (1 ブロック 18B) を VRAM へ上げる。
/// ホスト側で SOA (nib 16B アライン / sc f32) に変換してから転送する。
/// s165: 連結重みの行範囲エイリアス (登録済み) があれば転送せずビューを返す。
pub(crate) fn weights_for_q4(uid: u64, blocks: &[u8], k: usize, n: usize) -> Result<&'static GpuWq4> {
    let reg = q4_registry();
    let mut map = reg.lock().expect("q4 weights registry poisoned");
    // s165: 連結重みの行範囲エイリアスが登録済みなら転送しない (数値は不変)。
    if let Some(v) = q4_alias_view(uid, k, n, &mut map) {
        return Ok(v);
    }
    if !map.contains_key(&uid) {
        let nb = k / 32;
        assert_eq!(blocks.len(), n * nb * 18, "q4: block size mismatch");
        let mut nib = vec![0u8; n * nb * 16];
        let mut sc = vec![0f32; n * nb];
        for j in 0..n {
            for b in 0..nb {
                let src = &blocks[(j * nb + b) * 18..][..18];
                sc[j * nb + b] = half::f16::from_le_bytes([src[0], src[1]]).to_f32();
                nib[(j * nb + b) * 16..][..16].copy_from_slice(&src[2..18]);
            }
        }
        let nib_dev = DevBuffer::from_slice(&nib)?;
        let sc_dev = DevBuffer::from_slice(&sc)?;
        tracing::info!(
            "CUDA q4 weight upload (uid={uid}): k={k} n={n} ({:.1} MB)",
            (nib.len() + n * nb * 4) as f64 / 1e6
        );
        map.insert(uid, Box::new(GpuWq4 { nib: nib_dev, sc: sc_dev, k, n }));
        crate::gpu_budget::note_weight_upload();
    }
    let e = map.get(&uid).expect("just inserted");
    debug_assert_eq!((e.k, e.n), (k, n), "uid 重複: 同一 uid に異なる形状 (q4)");
    Ok(unsafe { std::mem::transmute::<&GpuWq4, &'static GpuWq4>(e.as_ref()) })
}


/// P-MOE-E5: GGUF の生ブロック (Q8_0 / Q5_1 / Q4_K / Q5_K) を VRAM に常駐させる
/// レジストリ (uid → デバイスバッファ)。重みは 1 度だけ転送し、以後は GEMV だけ。
/// 27B の q4 レジストリと同じ形 (uid は `quant::next_q8_uid` で払い出す)。
pub(crate) struct GpuGgufW {
    buf: DevBuffer,
    ty: crate::kquant::GgmlType,
    k: usize,
    n: usize,
}


/// P46: 診断 (`DENNO_GPU_W_DBG=1`)。登録 1 本ごとに 1 行、既定は完全に無効。
pub(crate) fn w_dbg() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DENNO_GPU_W_DBG").as_deref(), Ok("1")))
}


/// P46: GGUF 重みレジストリ (uid → デバイスバッファ + バイト上限つき LRU)。
///
/// 既定 (`DENNO_GPU_WEIGHT_MB` 未設定 = 0) は退避しない = 従来と同じ挙動。
/// 0 以外のときだけ、登録のたびに「上限」と「空き VRAM」を見て古い uid を
/// **確保の前に**解放する (`cuMemAlloc` の失敗を通常経路にしない)。
pub(crate) struct GgufReg {
    map: std::collections::HashMap<u64, Box<GpuGgufW>>,
    lru: crate::gpu_budget::WeightLru,
}


pub(crate) fn gguf_registry() -> &'static std::sync::Mutex<GgufReg> {
    static REG: OnceLock<std::sync::Mutex<GgufReg>> = OnceLock::new();
    REG.get_or_init(|| {
        std::sync::Mutex::new(GgufReg {
            map: std::collections::HashMap::new(),
            lru: crate::gpu_budget::WeightLru::new(crate::gpu_budget::weight_cap_bytes()),
        })
    })
}


/// uid の重みをレジストリから外し、VRAM を返す。戻り値は解放したバイト数。
///
/// モデルを捨てて組み直す経路 (serve-flash の prefix 不一致) で呼ぶ。これが無いと
/// 古いモデルの重みが VRAM に残り、リクエストを重ねるたびに使用量が増えて
/// 最後は dense の確保に失敗して CPU に落ちる。
///
/// 使用中 (pin) の uid でも外す: 呼ぶ側は「その重みをもう使わない」と分かっている
/// 前提 (`GgufLinear` の `Drop`)。進行中の GEMV がある状態で呼んではいけない。
pub fn gguf_unregister(uid: u64) -> u64 {
    // task 307: 控えている `&'static GpuGgufW` が解放され得るので世代を進める
    // (共有 expert のキャッシュは世代が変わったら丸ごと捨てる)。
    GGUF_REG_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let reg = gguf_registry();
    let Ok(mut g) = reg.lock() else {
        return 0;
    };
    let Some(w) = g.map.remove(&uid) else {
        return 0;
    };
    let bytes = g.lru.note_remove(uid);
    let (k, n, ty) = (w.k, w.n, w.ty);
    drop(w); // DevBuffer の Drop が VRAM を返す
    if w_dbg() {
        eprintln!("[gpu-w] unregister uid={uid} ty={ty:?} k={k} n={n} bytes={bytes}");
    }
    bytes
}


/// 使用中の重み 1 本。Drop で「使用中」を落とす (= 以後は退避対象になる)。
///
/// 中の参照は `'static` だが、このガードが生きている間は LRU がその uid を
/// 退避対象から外すので、指し先が解放されることはない。
pub(crate) struct GgufWRef {
    w: &'static GpuGgufW,
    uid: u64,
}


/// GGUF ブロック 1 行のバイト数。GPU の GEMV / GEMM が扱う 4 型だけ `Some`。
///
/// 型の一覧は [`crate::quant_matrix::DENSE_GPU`] が唯一の定義 (task 858)。
/// decode の GEMV・prefill の GEMM・expert の VRAM 常駐が**共通で** ここを通るので、
/// 表とここがずれると「GPU を試みて必ず落ちる」経路になる
/// (`quant_matrix::tests::dense_gpu_table_matches_cuda_dispatch` が固定する)。
pub(crate) fn gguf_row_bytes(ty: crate::kquant::GgmlType, k: usize) -> Option<usize> {
    use crate::kquant::GgmlType as T;
    if k == 0 || k % ty.blck_size() != 0 || !crate::quant_matrix::dense_gpu_supported(ty) {
        return None;
    }
    Some(match ty {
        T::Q8_0 => (k / 32) * 34,
        T::Q5_1 => (k / 32) * 24,
        T::Q4K => (k / 256) * 144,
        T::Q5K => (k / 256) * 176,
        _ => return None,
    })
}


/// 重みを (初回だけ) VRAM へ上げて常駐エントリを返す。`uid` は呼び出し側が
/// 安定して持つこと (同じ uid に別の形状・型を渡したらエラー)。
///
/// P46: `DENNO_GPU_WEIGHT_MB` が 0 以外なら、確保の前にバイト上限と空き VRAM を
/// 見て最終使用が最も古い uid を解放する (使用中の uid は飛ばす)。解放された
/// 重みは次に使うとき、呼び出し側が渡すホストの生ブロックから**同じバイト列で**
/// 上げ直される (転送はコピーなので GEMV の入力は 1 bit も変わらない)。
pub(crate) fn weights_for_gguf(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
) -> Result<GgufWRef> {
    let reg = gguf_registry();
    let mut g = reg
        .lock()
        .map_err(|_| anyhow::anyhow!("gguf weights registry poisoned"))?;
    // 既に載っている: 形状/型の一致だけ見て「使った」に更新し、使用中にする。
    if g.map.contains_key(&uid) {
        let e = g.map.get(&uid).expect("contains_key");
        let (ek, en, ety) = (e.k, e.n, e.ty);
        if (ek, en, ety) != (k, n, ty) {
            bail!(
                "uid 重複: 同一 uid {uid} に別の形状/型 (登録済み k={ek} n={en} ty={ety:?} / 今回 k={k} n={n} ty={ty:?})"
            );
        }
        g.lru.touch(uid);
        g.lru.pin(uid);
        let e = g.map.get(&uid).expect("pinned");
        let w = unsafe { std::mem::transmute::<&GpuGgufW, &'static GpuGgufW>(e.as_ref()) };
        return Ok(GgufWRef { w, uid });
    }
    let rb = gguf_row_bytes(ty, k)
        .ok_or_else(|| anyhow::anyhow!("gguf: 未対応の型 {ty:?} (k={k})"))?;
    let want = n
        .checked_mul(rb)
        .ok_or_else(|| anyhow::anyhow!("gguf: n * row_bytes が usize を溢れる"))?;
    if blocks.len() != want {
        bail!(
            "gguf: バイト数 {} が n * row_bytes = {} と違う (ty={ty:?}, k={k}, n={n})",
            blocks.len(),
            want
        );
    }
    // P46: 確保の前に退避する。上限 0 (既定) のときは記帳だけで 1 本も
    // 返らない = 従来と同じ経路 (空き VRAM の照会もしない)。
    let cap = crate::gpu_budget::weight_cap_bytes();
    let free_mb = if cap == 0 {
        None
    } else {
        dev_mem_mb().map(|(f, _total)| f as u64)
    };
    let ev0 = (g.lru.evicted_tensors(), g.lru.evicted_bytes());
    let victims =
        crate::gpu_budget::plan_weight_evictions(&mut g.lru, uid, want as u64, cap, free_mb);
    let mut dropped: Vec<u64> = Vec::new();
    for v in &victims {
        if g.map.remove(v).is_some() {
            dropped.push(*v);
        }
    }
    let freed = g.lru.evicted_bytes().saturating_sub(ev0.1);
    let freed_n = g.lru.evicted_tensors().saturating_sub(ev0.0);
    if freed_n > 0 {
        crate::gpu_budget::note_weight_evicted(freed_n, freed);
    }
    let buf = match DevBuffer::from_slice_tagged(blocks, crate::mem_audit::TAG_WEIGHTS) {
        Ok(b) => b,
        Err(e) => {
            // 記帳だけ進んで実体が無い状態を残さない。
            g.lru.note_remove(uid);
            return Err(e);
        }
    };
    tracing::info!(
        "CUDA gguf weight upload (uid={uid}): ty={ty:?} k={k} n={n} ({:.1} MB)",
        want as f64 / 1e6
    );
    g.map.insert(uid, Box::new(GpuGgufW { buf, ty, k, n }));
    g.lru.pin(uid);
    let resident = g.map.len();
    let used = g.lru.used_bytes();
    let evicted_tensors = g.lru.evicted_tensors();
    let evicted_bytes = g.lru.evicted_bytes();
    let w = {
        let e = g.map.get(&uid).expect("just inserted");
        unsafe { std::mem::transmute::<&GpuGgufW, &'static GpuGgufW>(e.as_ref()) }
    };
    drop(g);
    crate::gpu_budget::note_weight_upload();
    crate::gpu_budget::note_weight_upload_bytes(want as u64);
    crate::gpu_budget::note_weight_resident(resident, used);
    if w_dbg() {
        let free = match dev_mem_mb() {
            Some((f, t)) => format!("空き {f} MiB / 全 {t} MiB"),
            None => "空き 不明".to_string(),
        };
        let cap_s = if cap == 0 {
            "無制限".to_string()
        } else {
            format!("{} MiB", cap >> 20)
        };
        let ev = if dropped.is_empty() {
            String::new()
        } else {
            let list: Vec<String> = dropped.iter().take(8).map(|u| u.to_string()).collect();
            format!(" / 今回退避 {} 本 [{}] {:.1} MiB", dropped.len(), list.join(","), freed as f64 / (1 << 20) as f64)
        };
        tracing::info!(
            target: "denno_engine::gpu_w",
            "[gpu-w] uid={uid} ty={ty:?} k={k} n={n} {:.2} MB / 常駐 {} 本 {:.1} MiB / 上限 {cap_s} / 退避 計 {} 本 {:.1} MiB{ev} / {free}",
            want as f64 / 1e6,
            resident,
            used as f64 / (1 << 20) as f64,
            evicted_tensors,
            evicted_bytes as f64 / (1 << 20) as f64,
        );
    }
    Ok(GgufWRef { w, uid })
}


/// テスト用: `maybe_gpu_gguf_gemv_into` を「GPU が取った」ことにするフラグ。
///
/// 実デバイスには一切触らない (即 `true` を返すだけ)。P86 のように
/// 「GPU GEMV が呼び出しを奪ったとき CPU 経路が死ぬ」ことを、GPU の無い
/// コンテナで再現するためのもの。`GemvInterceptGuard` で立てる。
#[cfg(test)]
pub(crate) static GEMV_INTERCEPT_FOR_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// [`GEMV_INTERCEPT_FOR_TEST`] の間、`maybe_gpu_gguf_gemv_into` は
/// 「GPU が計算した」ことにして CPU へ落とさない。Drop で必ず戻す。
#[cfg(test)]
pub struct GemvInterceptGuard;


// ---------------------------------------------------------------------------
// task 124: shared expert の gate/up を router/expert の CPU 仕事と重ねる
// ---------------------------------------------------------------------------
//
// decode の shared expert は 1 層あたり 4 本の**同期** GEMV (gate / up / down /
// gate_inp) を回し、その間 rayon のワーカーは遊んでいる (実測 6.5 ms/tok)。
// このうち gate と up の入力は層の入口の `x` だけで決まるので、router と
// 選択 expert の CPU 仕事を始める**前に**専用ストリームへ積んでおき、
// shared expert の節で 1 回だけ join する。
//
// 規約:
// - カーネル・引数・行の内積順は [`CudaCtx::moe_gemv_dev`] と同一
//   ([`CudaCtx::plan_moe_gemv`] を共有する) ので、値は今の GPU 経路と
//   1 ビットも変わらない (md5 が動いたらバグ)。
// - 積む前に「型・形・重みが VRAM にあること・カーネルが選べること」を全部
//   確かめる (all-or-nothing)。1 本でも駄目なら 1 つも積まずに空のハンドルを
//   返し、呼び出し側は従来の同期経路へ落ちる (fail closed)。
// - 入出力は専用の grow-only バッファ (thread_local)。`MOE_SCRATCH` とは
//   共有しない — 積んだまま expert の GEMV が同じスクラッチを書き換えるため。
// - 重みは in-flight の間 `GgufWRef` で pin する (LRU の退避で解放されない)。
// - join し忘れたまま捨てられたら (エラー経路) Drop が drain する。drain に
//   失敗したら以後この経路は試さない (`broken`)。バッファはモジュール所有で、
//   呼び出し側の `Vec` を指したままにはしない (解放済み領域を触らない)。
// ---------------------------------------------------------------------------

/// 重ね打ち 1 本ぶんの依頼 (重みは `GgufLinear` の生ブロックをそのまま渡す)。
pub struct SharedGemvJob<'a> {
    pub uid: u64,
    pub ty: crate::kquant::GgmlType,
    pub blocks: &'a [u8],
    pub k: usize,
    pub n: usize,
}


pub(crate) static OVERLAP_BEGIN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) static OVERLAP_TAKEN: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) static OVERLAP_JOINED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) static OVERLAP_FAILED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) static OVERLAP_ABORTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


/// 計数を 0 に戻す (テスト用)。
pub fn reset_shared_overlap_stats() {
    use std::sync::atomic::Ordering::Relaxed;
    OVERLAP_BEGIN.store(0, Relaxed);
    OVERLAP_TAKEN.store(0, Relaxed);
    OVERLAP_JOINED.store(0, Relaxed);
    OVERLAP_FAILED.store(0, Relaxed);
    OVERLAP_ABORTED.store(0, Relaxed);
}


pub(crate) struct SharedSlotBuf {
    dout: Option<DevBuffer>,
    pin_o: Option<PinMem>,
    n: usize,
    /// テスト用の偽 GPU の結果 (本物の GPU が無い環境で join 経路を試す)。
    #[cfg(test)]
    fake: Vec<f32>,
}


/// 積んだ本数を持つハンドル。`shared_gemv_finish` を呼ばずに捨てると、
/// Drop が GPU を drain する (エラー経路で in-flight の操作を残さない)。
pub struct SharedPending {
    n: usize,
    /// 別スレッドへ移せない (Drop が同じスレッドの thread_local を drain するため)。
    _not_send: std::marker::PhantomData<*const ()>,
}


// ---------------------------------------------------------------------------
// task 310: 共有 expert の GEMV を v5 で回す (`DENNO_DEC_SH_V5`)
// ---------------------------------------------------------------------------
//
// 動機: task 307 で **同じ q8_0 GEMV が v5 (26 us) と v3 (49 us) に割れて**
// いることが分かった (`DENNO_GGUF_GEMV_V5`、t299b の `[dec-kprof-avg]` full 表)。
// その後 full 層 (`dec_plan` / `full_dec_plan`) と GDN (`dec_plan`) は v5 を
// 先に引くようになったが、**shared expert の 3 本 (gate / up / down、Q8_0) だけは
// `plan_moe_gemv` (v3 / v2) を直接引いていた**。1 層 3 本 x 48 層の GPU 時間は
// そのまま join の待ち (`sh_wait` 0.58 ms/token) になる。
//
// 数値: v5 は 4 本のアキュムレータで加算するので v3 とは丸め順が違う
// (決定性は保つ)。切り戻しはこの env。

/// task 310: `DENNO_DEC_SH_V5` の解釈 (純関数)。
///
/// - `1` = gate / up / down の 3 本とも v5 を先に引く
/// - `2` = gate / up だけ v5 (down は従来。k=640 では lane が 2 に落ちる形)
/// - 明示的な偽値 (`0` / `off` / `false` / `no` / 空) = 従来 (v3 / v2)
/// - 未設定・その他の値 = **既定 2**
///
/// task 319 で既定を 2 にした (lab t310b / t316a / t316c の 3 ラウンドで
/// 短い decode -2.4〜-2.7%、14k 後 -1.3〜-5.6%、prefill 不変、md5 SAME)。
/// 戻すのは `DENNO_DEC_SH_V5=0`。
pub(crate) fn parse_dec_sh_v5(raw: Option<&str>) -> u8 {
    /// 既定 (未設定)。task 319 の採否。
    const DEFAULT: u8 = 2;
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        Some("") | Some("0") | Some("off") | Some("false") | Some("no") => 0,
        _ => DEFAULT,
    }
}


/// [`parse_dec_sh_v5`] をプロセスで 1 回だけ読む。
pub(crate) fn dec_sh_v5_mode() -> u8 {
    static ON: OnceLock<u8> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_sh_v5(std::env::var("DENNO_DEC_SH_V5").ok().as_deref()))
}


/// task 187: shared expert の 4 発をサイドストリームへ**まとめて**積む。
///
///   x H2D → gate → up → silu(gate)*up → down → D2H(down の出力)
///
/// 従来 (task 124/155) は gate/up の出力をホストへ戻して `silu*mul` をホストで
/// 計算し、down を**あとから**積んでいた。そのため down の完了待ち (実測
/// 0.64 ms/token) が expert 段の後に残っていた。ここでは 4 発を最初に積むので、
/// 待ちは expert 段の GPU 仕事の裏に隠れる。
///
/// 数値: `dno_dec_silu_mul` は CUDA の `__expf` を使うので、ホストの
/// `silu_inplace` (f32::exp) とは **ビット一致しない** (丸めの差だけ)。
/// 既定 off のゲート `DENNO_DEC_SHARED_DEV` で切り替える。
///
/// `Ok` 以外の経路 (型・PTX・確保・形) は空のハンドルを返す = 呼び出し側は
/// 従来の 2 段経路へ落ちる (fail closed)。
// ---------------------------------------------------------------------------
// task 307: 共有 expert の plan / 常駐重みのキャッシュ (`DENNO_SH_BEGIN_CACHE`)
// ---------------------------------------------------------------------------
//
// 動機 (lab t299b の `[q4ledger]`): `shared` 行 0.63-0.68 ms/token のうち
// `sh_begin` が 0.45-0.48 で、その中身は毎層 3 本の `weights_for_gguf`
// (登録簿の mutex + LRU の touch/pin と、消費後の unpin 3 本) と plan の作り直し。
// uid の組は層ごとに固定なので、最初の 1 回で作った計画と重み参照を控える。

/// task 307: `DENNO_SH_BEGIN_CACHE` の解釈 (純関数)。**既定 off**。
///
/// 明示的な真値 (`1` / `on` / `true`) のときだけ、共有 expert の計画と常駐重みを
/// 層ごとに控える。
pub(crate) fn parse_sh_begin_cache(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}


/// task 307: 共有 expert の計画と常駐重みを層ごとに控えるか。
pub(crate) fn sh_begin_cache_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_sh_begin_cache(std::env::var("DENNO_SH_BEGIN_CACHE").ok().as_deref()))
}


/// task 307: 重みの登録簿から 1 本でも外されたら増える世代。
///
/// [`GGUF_REG_GEN`] が変わったら、控えている `&'static GpuGgufW` は解放済みに
/// なり得るのでキャッシュを丸ごと捨てる (`gguf_unregister` が増やす)。
static GGUF_REG_GEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);


/// task 307: uid の組でキャッシュを引く。無ければ `None`。
///
/// 戻り値は (計画 3 本, 重みのデバイスポインタ 3 本)。形が登録時と違えば
/// その項目を捨てて `None` を返す (呼び出し側は従来経路で作り直す)。
pub(crate) fn sh_cache_lookup(
    jobs: &[&SharedGemvJob<'_>; 3],
) -> Option<([MoeGemvPlan; 3], [CUdeviceptr; 3])> {
    let gen = GGUF_REG_GEN.load(std::sync::atomic::Ordering::Relaxed);
    SH_CACHE.with(|cell| {
        let mut v = cell.borrow_mut();
        let i = v.iter().position(|e| {
            e.gen == gen && jobs.iter().enumerate().all(|(j, job)| e.shapes[j].0 == job.uid)
        })?;
        let e = &v[i];
        let shapes_ok = jobs.iter().enumerate().all(|(j, job)| {
            let (k, n, ty) = (e.shapes[j].1, e.shapes[j].2, e.shapes[j].3);
            k == job.k && n == job.n && ty == job.ty
        });
        if !shapes_ok {
            v.remove(i);
            return None;
        }
        Some((e.plans, [e.pins[0].buf.ptr, e.pins[1].buf.ptr, e.pins[2].buf.ptr]))
    })
}


/// task 307: キャッシュへ 1 本入れる (既にあれば置き換え)。
pub(crate) fn sh_cache_insert(jobs: &[&SharedGemvJob<'_>; 3], plans: &[MoeGemvPlan], pins: Vec<GgufWRef>) {
    if pins.len() != 3 {
        return;
    }
    let mut it = pins.into_iter();
    let (Some(p0), Some(p1), Some(p2)) = (it.next(), it.next(), it.next()) else {
        return;
    };
    let gen = GGUF_REG_GEN.load(std::sync::atomic::Ordering::Relaxed);
    let entry = ShCacheEntry {
        gen,
        shapes: [
            (jobs[0].uid, jobs[0].k, jobs[0].n, jobs[0].ty),
            (jobs[1].uid, jobs[1].k, jobs[1].n, jobs[1].ty),
            (jobs[2].uid, jobs[2].k, jobs[2].n, jobs[2].ty),
        ],
        plans: [plans[0], plans[1], plans[2]],
        pins: [p0, p1, p2],
    };
    SH_CACHE.with(|cell| {
        let mut v = cell.borrow_mut();
        // 世代が変わっていたら丸ごと捨ててから入れる。
        if v.iter().any(|e| e.gen != gen) {
            v.clear();
        }
        if v.len() >= 256 {
            v.remove(0);
        }
        v.push(entry);
    });
}


// ---------------------------------------------------------------------------
// task 576: shared expert の gate + up + silu*mul を 1 発に畳む (`DENNO_DEC_SH_GU`)
// ---------------------------------------------------------------------------
//
// 動機 (lab t572b の `[q4ledger] shared`、既定モデル warm 短 decode):
//   段 0.59 ms/token のうち launch=0.24 (192 本 = 48 層 x 4) + cpu=0.21 で
//   **sh_wait は 0.05** = 段は GPU ではなくホストの投入で決まっている。
//   1 層 4 発のうち gate と up は同じ x を読み、silu は 640 要素 (0.9 us)。
//   起動 3 本を 1 本にすれば 2 本 x 48 層 = 96 本/token が消える。
//
// 数値: 1 行の積和は `kernels_moe_v5.cu::dno_gemv_q8_0_v5` と同じ演算列
// (アキュムレータ 4 本 / `j & 3` / ブロック 0 -> 1 / shfl 1,2,4...)、silu は
// `kernels_dec_epi.cu::dno_dec_silu_mul` と同じ式。**ビット一致**。

/// 新カーネル名 (`kernels_sh_gu.cu`)。
pub(crate) const SH_GU_KERNEL_NAME: &str = "dno_dec_sh_gateup_silu";

/// 引数の数 (wg, wu, x, act, n_in, n_out, row_bytes, rows_per_block, lanes)。
pub(crate) const SH_GU_NARGS: usize = 9;


pub(crate) struct ShGuKernels {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


pub(crate) static SH_GU_KERNELS: OnceLock<Option<ShGuKernels>> = OnceLock::new();


/// `DENNO_DEC_SH_GU` の解釈 (純関数)。未設定は既定 (on)、明示的な偽値だけ off。
///
/// lab t576b (既定モデル、depth8k、2 ラウンド交互) で採った:
/// **shared 段 0.58 -> 0.45 ms/token (-22%、起動 192 -> 96 本/token)、
/// step 9.97/10.01 -> 9.85/9.81 (-1.2% / -2.0%)、decode 中央値 10.0 -> 9.8
/// (-2.0%)、他の段は不変、md5 全 COND 全 6 文脈 SAME、errors 0**。
/// 切り戻しは `DENNO_DEC_SH_GU=0`。
pub(crate) fn parse_sh_gu(raw: Option<&str>) -> bool {
    match raw {
        None => true,
        Some(s) => {
            let s = s.trim().to_ascii_lowercase();
            !matches!(s.as_str(), "" | "0" | "off" | "false" | "no")
        }
    }
}


/// 発火を数える (診断。`[sh-gu]` の 1 行に出る)。
pub(crate) static SH_GU_FUSED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// 門番で落ちた回数 (v5 経路へ)。
pub(crate) static SH_GU_FALLBACK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// P-MOE-E5: GGUF ブロックのままの線形層 (m=1) を GPU で計算する。
///
/// 対象外の型・GPU 無効・CUDA 不調・登録失敗では `false` を返す。呼び出し側は
/// CPU 参照へ落ちること (fail closed。別経路の数値を黙って返さない)。
pub fn maybe_gpu_gguf_gemv_into(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    set_last_linear_ran_on_gpu(false);
    #[cfg(test)]
    if GEMV_INTERCEPT_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed) {
        // テスト用: GPU が呼び出しを取った体で抜ける (out は書かない)。
        return true;
    }
    if !cuda_enabled() || gguf_row_bytes(ty, k).is_none() {
        return false;
    }
    if x.len() != k || out.len() != n {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU gguf: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    // devprof: 重みのアップロードは「この呼び出しで増えたバイト数」だけ数える
    // (常駐ヒットなら 0)。集計は gpu_budget が持っている。
    let prof = crate::model::devprof::on();
    let up_before = if prof {
        crate::gpu_budget::weight_upload_bytes()
    } else {
        0
    };
    let w = match weights_for_gguf(uid, ty, blocks, k, n) {
        Ok(w) => w,
        Err(e) => {
            // P-pf-arena: 同じ失敗を 1 呼び出しごとに ERROR で出すと 1 リクエストで
            // 数万行になる (実測 40,554 行)。数だけ数え、ログは「1 回目 + 1,000
            // 回ごとの進捗」に畳む。出す内容 (uid とメッセージ) は従来と同じ。
            let msg = format!("{e:#}");
            match crate::gpu_budget::note_weight_failure(uid, &msg) {
                Some(true) => tracing::error!(
                    "gguf weight register failed (uid={uid}): {msg} (同種の失敗は数えるだけ。集計はリクエスト末尾の 1 行)"
                ),
                Some(false) => {
                    let s = crate::gpu_budget::weight_upload_stats();
                    tracing::warn!("gguf weight register failed: 累計 {} 本 (最後の uid={uid})", s.failed);
                }
                None => {}
            }
            return false;
        }
    };
    // task 307: `DENNO_GGUF_GEMV_V5` のときは v5 の計画を先に試す
    // (形が合わなければ従来どおり v3 / v2)。オフでは 1 つも変わらない。
    match ctx.moe_gemv_dev_with_pref(
        ty,
        &w.buf,
        x,
        k,
        n,
        out,
        gemv_kernel_mode(),
        gguf_gemv_v5_on(),
    ) {
        Ok(true) => {
            set_last_linear_ran_on_gpu(true);
            if prof {
                let up_w = crate::gpu_budget::weight_upload_bytes().saturating_sub(up_before);
                crate::model::devprof::note_bytes(k as u64 * 4 + up_w, n as u64 * 4);
            }
            true
        }
        Ok(false) => false,
        Err(e) => {
            tracing::error!("gguf gemv failed (uid={uid}): {e:#}");
            false
        }
    }
}


/// P-PF-GB: prefill 用の量子 GEMM の入口 (nt トークン一括)。
///
/// `maybe_gpu_gguf_gemv_into` のバッチ版。重みは同じレジストリ (`uid`) を使うので、
/// decode で既に載っていれば転送は起きない。`false` = CPU バッチへ落ちること。
/// `DENNO_GPU_GEMM=0` で無効化できる (A/B 用)。
///
/// CPU のバッチ演算とは **ビット一致しない** (FMA 融合 + 加算順が違う)。
/// prefill は 2026-10-28 のオペレーター指示で契約の外にある。CPU 参照と
/// 突き合わせるテストは `set_gpu_gemm_enabled(false)` で切ってから走らせること。
pub fn gpu_gemm_enabled() -> bool {
    if !GPU_GEMM_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    !matches!(
        std::env::var("DENNO_GPU_GEMM").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}


/// テスト用: GPU GEMM を明示的に切る / 入れる。
pub fn set_gpu_gemm_enabled(on: bool) {
    GPU_GEMM_ON.store(on, std::sync::atomic::Ordering::Relaxed);
}


static GPU_GEMM_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);


/// GPU GEMM が実際に計算を肩代わりした回数 / 落ちた回数 (プロセス通算)。
///
/// 「緑 = 経路が発動した証拠ではない」ので、実モデルの prefill で本当に
/// 通ったかをこれで数える (2026-10-28: 共有メモリ超過で全滅していたのを
/// `差 0 要素` から追った)。
pub(crate) static GEMM_USED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) static GEMM_SKIPPED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


/// (`使われた回数`, `CPU へ落ちた回数`)。
pub fn gpu_gemm_stats() -> (usize, usize) {
    (
        GEMM_USED.load(std::sync::atomic::Ordering::Relaxed),
        GEMM_SKIPPED.load(std::sync::atomic::Ordering::Relaxed),
    )
}


/// 落ちた理由の内訳 (添字は [`SKIP_*`] 定数)。
pub const SKIP_NAMES: [&str; 8] = [
    "off(env/flag)",
    "cuda無効",
    "nt範囲外",
    "未対応の型",
    "ctx無し",
    "重み未登録",
    "カーネル拒否",
    "実行エラー",
];

pub const SKIP_OFF: usize = 0;

pub const SKIP_NO_CUDA: usize = 1;

pub const SKIP_NT: usize = 2;

pub const SKIP_TYPE: usize = 3;

pub const SKIP_NO_CTX: usize = 4;

pub const SKIP_NO_WEIGHT: usize = 5;

pub const SKIP_KERNEL: usize = 6;

pub const SKIP_ERR: usize = 7;

static GEMM_SKIP_REASONS: [std::sync::atomic::AtomicUsize; 8] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];


/// 落ちた理由の内訳。
pub fn gpu_gemm_skip_reasons() -> [usize; 8] {
    let mut a = [0usize; 8];
    for (i, c) in GEMM_SKIP_REASONS.iter().enumerate() {
        a[i] = c.load(std::sync::atomic::Ordering::Relaxed);
    }
    a
}


pub(crate) fn skip(r: usize) -> bool {
    GEMM_SKIP_REASONS[r].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    false
}


/// 計数を 0 に戻す (ベンチの区切り用)。
pub fn reset_gpu_gemm_stats() {
    GEMM_USED.store(0, std::sync::atomic::Ordering::Relaxed);
    GEMM_SKIPPED.store(0, std::sync::atomic::Ordering::Relaxed);
    for c in GEMM_SKIP_REASONS.iter() {
        c.store(0, std::sync::atomic::Ordering::Relaxed);
    }
}


// ===========================================================================
// task 169: prefill の射影を BF16 テンソルコア GEMM で回す (`DENNO_PF_TC_GEMM`)
// ===========================================================================
//
// lab t167c (14,018 トークン cold prefill = 93.87 s) の台帳では、ssm の qkv
// 投影が 16.3 s (全体の 17%)、full attention の射影が 9.2 s、shared expert が
// 8.1 s を占めていた。どれも `kernels_gemm.cu` の「16 行 × 128 トークン」版
// (f32 FMA・1 起動 128 トークンまで) で、計算量あたり 0.6〜2.6 TFLOP/s =
// テンソルコアの 1/20 以下。ここでは重みタイルを共有メモリへ 1 回だけ展開し、
// BF16 の mma (FP32 アキュムレート) で 128 トークン × 64 行ずつ畳む。
//
// 数値: prefill の射影はビット一致契約の外 (2026-10-28)。x と重みを BF16 に
// 丸めるので既存カーネルとは値が変わる (丸めだけ。演算の意味は同じ)。
// 既定 off (`DENNO_PF_TC_GEMM=1` で有効)。PTX は実行時読み込みで、無ければ
// 従来カーネルへ fail closed (`[pf-tc]` の 1 行だけ出す)。
//
// PTX: `bash build_ptx.sh gemm_tc` (この target だけ arch = compute_90)。
// このコンテナには nvcc が無いので生成は GPU 側 (lab は PTX=gemm_tc で行う)。

/// 1 ブロックのタイル (`kernels_gemm_tc.cu` の TC_BM / TC_BN / TC_BK と一致)。
pub(crate) const TC_BM: usize = 128;

pub(crate) const TC_BN: usize = 64;

pub(crate) const TC_BK: usize = 32;

/// 1 起動でまとめるトークン数の上限 (重みタイルの再利用を効かせる)。
pub(crate) const TC_MAX_TOKENS: usize = 2048;


#[cfg(test)]
pub(crate) static PF_TC_GEMM_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_tc_gemm_for_test(v: Option<bool>) {
    PF_TC_GEMM_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// テンソルコア GEMM のカーネル (PTX は実行時読み込み)。
pub(crate) struct TcGemm {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    q8_0: CUfunction,
}


/// テンソルコア GEMM のどちらを使うか (br169 / task 176 の v2 / task 181 の小さい N 用)。
pub(crate) enum TcSel {
    V1(&'static TcGemm),
    V2(&'static Tc2Gemm),
    Sn(&'static SnGemm),
}


pub(crate) static TC_GEMM: OnceLock<Option<TcGemm>> = OnceLock::new();


/// PTX の置き場所。`DENNO_PF_TC_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gemm_tc.ptx`。
pub(crate) fn tc_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_TC_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_tc.ptx"),
    }
}


// ===========================================================================
// task 176: prefill の射影を BF16 テンソルコア GEMM v2 で回す (`DENNO_PF_TC_GEMM2`)
// ===========================================================================
//
// br169 (`kernels_gemm_tc.cu`) は 1 ブロック = 128 トークン × 64 行で、k ステップ
// ごとに「global 読み → smem 書き → 同期 → mma → 同期」を直列に回していた
// (プリフェッチ無し)。lab t169b の台帳 (14,018 トークン cold prefill) では
//   ssm qkv  512 x 2560 x 10240  3,371 ms → 8.0 TFLOP/s
//   z        512 x 2560 x  6144  2,896 ms → 5.6 TFLOP/s
//   out      512 x 6144 x  2560    667 ms → 24.3 TFLOP/s
// と同じカーネルでも K 2560 の形だけが極端に遅く、1 ステップの露出レイテンシが
// 固定費として効いている。v2 では
//   1. 出力行 64 → 128 (1 ブロックの仕事が 2 倍 = 固定費が半分)
//   2. 2 段のレジスタプリフェッチ (次の k を先に読み、その mma と重ねる)
//   3. smem の行ストライド 40 (バンク衝突の回避)
//   4. Q4_K / Q5_K も同じカーネルで扱う (実モデルの ssm qkv は 4 層だけ Q5_K、
//      z は Q4_K。br169 はこの 4 層を f32 カーネルへ落としていた = 14k の
//      qkv_pf 6,268 ms のうち約 1.8 s がこの 4 層)
// を行う。数値契約は br169 と同じく無し (x と重みを BF16 に丸め、FP32 で
// アキュムレートする。丸めだけの差で演算の意味は同じ)。
//
// 既定 on (`DENNO_PF_TC_GEMM2=0` で切る)。PTX は実行時読み込みで、無ければ
// br169 のカーネル → 従来カーネルへ fail closed (`[pf-tc2]` の 1 行だけ出す)。
//
// PTX: `bash build_ptx.sh gemm_tc2` (この target だけ arch = compute_90)。
// このコンテナには nvcc が無いので生成は GPU 側 (lab は PTX=gemm_tc2 で行う)。

/// 1 ブロックのタイル (`kernels_gemm_tc2.cu` の TC2_BM / TC2_BN / TC2_BK と一致)。
pub(crate) const TC2_BM: usize = 128;

pub(crate) const TC2_BN: usize = 128;

pub(crate) const TC2_BK: usize = 32;

/// 共有メモリの行ストライド (`TC2_LD`。8 要素の詰め物でバンク衝突を避ける)。
pub(crate) const TC2_LD: usize = 40;

/// 静的共有メモリの合計 (`TC2_SMEM_BYTES` = sW 2 x 128 x 40 x 2 + sX 同)。
pub(crate) const TC2_SMEM_BYTES: usize = 40_960;

/// 1 起動でまとめるトークン数の上限 (br169 と同じ)。
pub(crate) const TC2_MAX_TOKENS: usize = 2048;

/// v2 を使う最小の出力行数。**K 2560 で N が大きい形だけ**を v2 にし、
/// out_proj (N 2560) と shared expert (N 640) は br169 のまま残す
/// (br169 は K 6144 の形で 24.3 TFLOP/s 出ているので、触らない方が安全)。
pub(crate) const TC2_MIN_N: usize = 4096;


/// smem の見積りがカーネル側の `TC2_SMEM_BYTES` と一致することのコンパイル時検査
/// (実行時コスト 0。片側だけ直した事故をここで落とす)。
const _: [(); TC2_SMEM_BYTES] = [(); 2 * TC2_BN * TC2_LD * 2 + 2 * TC2_BM * TC2_LD * 2];


#[cfg(test)]
pub(crate) static PF_TC2_GEMM_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_tc2_gemm_for_test(v: Option<bool>) {
    PF_TC2_GEMM_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// この形を**何らかの**テンソルコア GEMM で回せるか (PTX が読めることも見る)。
///
/// `gemm_try` / GDN 連鎖の「1 起動でまとめる幅」の判定に使う。どちらの
/// カーネルも使えないときは従来カーネル (1 起動 128 トークン) なので、
/// まとめ幅を広げてはいけない (広げると呼び出し側が CPU へ落ちる)。
pub(crate) fn pf_gemm_fast_ok(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
    min_m: usize,
) -> bool {
    if pf_tc2_gemm_on()
        && pf_tc2_eligible(ty, nt, n_in, n_out, min_m)
        && tc2_gemm_kernels().is_some()
    {
        return true;
    }
    // task 181: N が小さい形も 1 起動で 2,048 行まで畳める (BN=64 タイル)。
    if pf_tc_smalln_on()
        && pf_tc_smalln_eligible(ty, nt, n_in, n_out, min_m)
        && tc_smalln_kernels().is_some()
    {
        return true;
    }
    matches!(ty, crate::kquant::GgmlType::Q8_0)
        && pf_tc_gemm_on()
        && pf_tc_gemm_eligible(nt, n_in, n_out, min_m)
        && tc_gemm_kernels().is_some()
}


/// v2 カーネル (PTX は実行時読み込み)。
pub(crate) struct Tc2Gemm {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    q8_0: CUfunction,
    q4k: CUfunction,
    q5k: CUfunction,
}


pub(crate) static TC2_GEMM: OnceLock<Option<Tc2Gemm>> = OnceLock::new();


/// PTX の置き場所。`DENNO_PF_TC2_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gemm_tc2.ptx`。
pub(crate) fn tc2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_TC2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_tc2.ptx"),
    }
}


/// 型ごとのカーネル (純関数)。
pub(crate) fn tc2_func(fns: &Tc2Gemm, ty: crate::kquant::GgmlType) -> Option<CUfunction> {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Q8_0 => Some(fns.q8_0),
        T::Q4K => Some(fns.q4k),
        T::Q5K => Some(fns.q5k),
        _ => None,
    }
}


// ===========================================================================
// task 181: prefill の **N が小さい** 射影を BF16 テンソルコア GEMM で回す
// ===========================================================================
//
// lab t175b の台帳 (14,018 トークン cold prefill、br172 の 3 ゲート on) では
//   shared expert  gpu 3,542 ms (= 1.9 TFLOP/s)
//   ssm out_pf           270 ms (= 24 TFLOP/s、br169)
// だった。shared の行列は N = 640 / 2560 で v2 の下限 TC2_MIN_N = 4096 に
// 入らないため、Q8_0 の up / down は br169 (BN=64・プリフェッチ無し)、
// Q4_K の gate は f32 カーネル (`dno_gemm_q4k_w`) に落ちていた。
// br169 でも N=640 は 640/64 = 10 行タイル x (M=512 で 4) = 40 ブロックで、
// 170 SM の 24% しか使わない。ここでは tc2 と同じ 2 段プリフェッチ +
// smem ストライド 40 のまま **BN=64** にして、N=640 で 40、
// N=2560 で 160 ブロックを出す (N=2560 は 1 波で埋まる)。
//
// 既定 off (`DENNO_PF_TC_SMALLN=1` で on)。PTX は実行時読み込みで、無ければ
// 従来経路へ fail closed (`[pf-sn]` の 1 行だけ出す)。
//
// PTX: `bash build_ptx.sh gemm_sn` (この target だけ arch = compute_90)。

/// 1 ブロックのタイル (`kernels_gemm_sn.cu` の TC_SN_BM / BN / BK と一致)。
pub(crate) const TC_SN_BM: usize = 128;

pub(crate) const TC_SN_BN: usize = 64;

pub(crate) const TC_SN_BK: usize = 32;

/// 共有メモリの行ストライド (`TC_SN_LD`。tc2 と同じ 8 要素の詰め物)。
pub(crate) const TC_SN_LD: usize = 40;

/// 静的共有メモリの合計 (`TC_SN_SMEM_BYTES` = sW 2 x 64 x 40 x 2 + sX 2 x 128 x 40 x 2)。
pub(crate) const TC_SN_SMEM_BYTES: usize = 30_720;

/// 1 起動でまとめるトークン数の上限 (tc2 と同じ)。
pub(crate) const TC_SN_MAX_TOKENS: usize = 2048;


/// smem の見積りがカーネル側の `TC_SN_SMEM_BYTES` と一致することのコンパイル時検査。
const _: [(); TC_SN_SMEM_BYTES] =
    [(); 2 * TC_SN_BN * TC_SN_LD * 2 + 2 * TC_SN_BM * TC_SN_LD * 2];


#[cfg(test)]
pub(crate) static PF_TC_SN_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_tc_smalln_for_test(v: Option<bool>) {
    PF_TC_SN_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// 小さい N 用カーネル (PTX は実行時読み込み)。
pub(crate) struct SnGemm {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    q8_0: CUfunction,
    q4k: CUfunction,
}


pub(crate) static SN_GEMM: OnceLock<Option<SnGemm>> = OnceLock::new();


/// PTX の置き場所。`DENNO_PF_SN_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gemm_sn.ptx`。
pub(crate) fn sn_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SN_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_sn.ptx"),
    }
}


/// 型ごとのカーネル (純関数)。
pub(crate) fn sn_func(fns: &SnGemm, ty: crate::kquant::GgmlType) -> Option<CUfunction> {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Q8_0 => Some(fns.q8_0),
        T::Q4K => Some(fns.q4k),
        _ => None,
    }
}


// ===========================================================================
// task 183: shared expert の射影を 1 ブロック 64 トークンで回す (`DENNO_PF_TC_SHARED2`)
// ===========================================================================
//
// lab t181c の台帳 (14,018 トークン cold prefill、既定) では shared が
// 2,406 ms (gpu 1,220 ms = 5.4 TFLOP/s) で、まだレイテンシ律速だった。
// sn (BM=128) の N=640 は M=512 のチャンクで 10 行タイル x 4 = 40 ブロック
// しか出ず、RTX 5090 の 170 SM の 24% しか埋めない。BM=64 にすると
// 10 x 8 = 80 ブロック (47%) になる。加算順は sn と同じ (k 昇順) なので
// sn とビット一致し、md5 は sn のまま (lab で確認する)。
//
// PTX: `bash build_ptx.sh gemm_sn2` (compute_90)。

/// 1 ブロックのタイル (`kernels_gemm_sn2.cu` の TC_SN2_BM / BN / BK と一致)。
pub(crate) const TC_SN2_BM: usize = 64;

pub(crate) const TC_SN2_BN: usize = 64;

pub(crate) const TC_SN2_BK: usize = 32;

/// 共有メモリの行ストライド (`TC_SN2_LD`。tc2 / sn と同じ 8 要素の詰め物)。
pub(crate) const TC_SN2_LD: usize = 40;

/// 静的共有メモリの合計 (`TC_SN2_SMEM_BYTES` = sW 2 x 64 x 40 x 2 + sX 2 x 64 x 40 x 2)。
pub(crate) const TC_SN2_SMEM_BYTES: usize = 20_480;

/// 1 起動でまとめるトークン数の上限 (sn と同じ)。
pub(crate) const TC_SN2_MAX_TOKENS: usize = 2048;


/// smem の見積りがカーネル側の `TC_SN2_SMEM_BYTES` と一致することのコンパイル時検査。
const _: [(); TC_SN2_SMEM_BYTES] =
    [(); 2 * TC_SN2_BN * TC_SN2_LD * 2 + 2 * TC_SN2_BM * TC_SN2_LD * 2];


#[cfg(test)]
pub(crate) static PF_TC_SN2_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_tc_shared2_for_test(v: Option<bool>) {
    PF_TC_SN2_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// BM=64 のカーネル (PTX は実行時読み込み)。
pub(crate) struct Sn2Gemm {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    q8_0: CUfunction,
    q4k: CUfunction,
}


pub(crate) static SN2_GEMM: OnceLock<Option<Sn2Gemm>> = OnceLock::new();


#[cfg(test)]
pub(crate) static PF_TC_SN4_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_tc_shared4_for_test(v: Option<bool>) {
    PF_TC_SN4_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


// ---------------------------------------------------------------------------
// task 222: BM=64 / BN=64 のタイルで Q8_0 / Q4_K / Q5_K / BF16 を受ける版
// ---------------------------------------------------------------------------
//
// sn2 (task 183) は Q8_0 / Q4_K 専用だった。実モデルの shared expert は
// 層ごとに型が混ざっていて (gate/up が Q8_0 54 本・Q4_K 30 本・Q5_K 12 本)、
// Q5_K の 12 本は従来の量子 GEMM に落ちていた。hc mix の down / up は BF16 で、
// こちらは f32 の FMA カーネル (`dno_gemm_bf16_mt`) が担当していた。
// どちらも同じ sn2 のタイル (lab t183a で検証済み) に載せる。
//
// PTX: `bash build_ptx.sh gemm_sn3` (arch = compute_90。bf16 の mma が要る)。

/// 1 ブロックのタイル (`kernels_gemm_sn3.cu` の SN3_BM / BN / BK と一致)。
pub(crate) const SN3_BM: usize = 64;

/// 出力行のタイル。
pub(crate) const SN3_BN: usize = 64;

/// k の刻み。
pub(crate) const SN3_BK: usize = 32;

/// 静的共有メモリの合計 (`SN3_SMEM_BYTES` = sW 2 x 64 x 40 x 2 + sX 2 x 64 x 40 x 2)。
pub(crate) const SN3_SMEM_BYTES: usize = 20_480;

/// 1 起動でまとめるトークン数の上限 (sn2 と同じ)。
pub(crate) const SN3_MAX_TOKENS: usize = 2048;


/// smem の見積りがカーネル側の `SN3_SMEM_BYTES` と一致することのコンパイル時検査。
const _: [(); SN3_SMEM_BYTES] = [(); 2 * SN3_BN * 40 * 2 + 2 * SN3_BM * 40 * 2];


#[cfg(test)]
pub(crate) static PF_TC_SN3_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_tc_shared3_for_test(v: Option<bool>) {
    PF_TC_SN3_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// BM=64 のカーネル一式 (PTX は実行時読み込み)。
pub(crate) struct Sn3Gemm {
    /// モジュールは生かしておく必要がある (関数ハンドルの持ち主)。
    #[allow(dead_code)]
    module: CUmodule,
    q8_0: CUfunction,
    q4k: CUfunction,
    q5k: CUfunction,
    bf16: CUfunction,
}


pub(crate) static SN3_GEMM: OnceLock<Option<Sn3Gemm>> = OnceLock::new();


// ---------------------------------------------------------------------------
// s138 (Task B6・計測のみ): 線形ラッパ 1 call の段割り (表 B) の小道具。
// `split_stage` で tick を取り、段の切れ目で `split_mark`、最後に `split_flush`。
// 計測 OFF (`split_on()` が偽) は tick が None で、どれも分岐 1 回で戻る —
// 既定経路はクロック読みも確保もしない (挙動・数値は完全に不変)。
// ---------------------------------------------------------------------------

/// s140 (Task B6b): 線形ステージ 1 呼び出しを計測器に通す。
///
/// s138 は線形ラッパ (`linear_pinned` / `q4_dispatch_tile`) の中だけを分割して
/// いたが、serve の prefill は **GEMM 経路 (`linear_gemm_into`) と Q4 GEMV
/// (`q4_dispatch` のマルチタイル) を通る**ため、段の区間が呼び出し全体を覆って
/// いなかった (実測: 608 call / 7846 ms のバケットに対し 802 サンプル / 445.8 ms)。
/// ここでは呼び出しの**手前**で tick を打ち、戻ってきた時点の経過をバケット件数へ
/// 数える。これでタイル分割・GEMM 経路であるかに関係なく 1 call = 1 tick になり、
/// 排他の和が呼び出し全体の壁時計に一致する。
///
/// 計測 OFF (`op_on()` が偽) は分岐 1 回で `f` を素通しする (クロック読み・
/// 表への書き込みなし = 既定経路の挙動不変)。`split_on()` (テレメトリのみ) の
/// ときは表だけ積み、バケット件数は数えない (バケットは prof 専用のため)。
#[inline]
pub(crate) fn lin_stage<T>(op: usize, f: impl FnOnce() -> T) -> T {
    if !crate::prof::op_on() {
        return f();
    }
    let t0 = Instant::now();
    let split = crate::prof::split_on();
    if split {
        crate::prof::split_tick_call(crate::prof::SPLIT_Q4_TABLE);
    }
    let r = f();
    lin_stage_close(op, t0);
    r
}


/// s140 (Task B6b): 実行先ラベル (on_gpu) つきの `lin_stage`。`time_n_ev` が
/// 返すのと同じ `(値, GPU で走ったか)` を返すので、呼び出し側は
/// `let (r, _) = lin_stage_ev(op, || ...);` と書ける (戻り値・数値経路は不変)。
#[inline]
fn lin_stage_ev<T>(op: usize, f: impl FnOnce() -> (T, bool)) -> (T, bool) {
    if !crate::prof::op_on() {
        return f();
    }
    let t0 = Instant::now();
    let split = crate::prof::split_on();
    if split {
        crate::prof::split_tick_call(crate::prof::SPLIT_Q4_TABLE);
    }
    let (r, on_gpu) = f();
    lin_stage_close(op, t0);
    (r, on_gpu)
}


/// s140 (Task B6b): 線形呼び出し 1 回を `lin_stage` で包み、**バケットは呼び出し側が
/// 従来どおり** `prof::time_full` 等で積む形 (chain の段は per-op バケットを
/// 別の `time_full` で積んでいるため、こちらはバケットを二重に積まない)。
/// 戻り値は `f` のまま (数値経路は不変)。
#[inline]
pub fn lin_stage_tagged<T>(op: usize, f: impl FnOnce() -> T) -> T {
    if !crate::prof::op_on() {
        return f();
    }
    let t0 = Instant::now();
    let r = f();
    lin_stage_close(op, t0);
    r
}


/// `lin_stage_tagged` の締め: 分割表 1 サンプル + バケット件数 1。
#[inline]
fn lin_stage_close(op: usize, t0: Instant) {
    if crate::prof::split_on() {
        crate::prof::split_tick_call(crate::prof::SPLIT_Q4_TABLE);
    }
    let ns = t0.elapsed().as_nanos() as u64;
    if crate::prof::enabled() {
        crate::prof::count_linear_stage(op, ns, 1, true);
    }
    crate::prof::split_sample(op, ns, 0);
}


/// 段割りの開始 tick (計測 OFF は None)。
#[inline]
pub(crate) fn split_stage() -> Option<Instant> {
    if crate::prof::split_on() {
        Some(Instant::now())
    } else {
        None
    }
}


/// 直前の段 (`op`) の所要を表 B へ積み、次の段の tick を打ち直す。
/// `op` は「いま終わった段」— 呼び出しは段の切れ目 (確保 / copy / launch / sync
/// の直後) に置く。calls は 1 (その段が起きたら 1 回、起きなかった段は積まない)。
#[inline]
pub(crate) fn split_mark(t: &mut Option<Instant>, op: usize) {
    let Some(t0) = *t else { return };
    let now = Instant::now();
    let ns = now.duration_since(t0).as_nanos() as u64;
    crate::prof::split_sample(op, ns, 1);
    *t = Some(now);
}


/// 1 call 分の段割りを閉じる (残りの段の tick は捨てる)。
/// 段の値は `split_mark` が `split_sample` で直接積んでいるので、ここで表へ
/// 流し込むものは無い (時間の丸めを段ごとに 1 回で済ませるため、8〜9 段の
/// 積算器は持たない — `secs` でまとめて足すと段の数だけ丸め誤差が乗る)。
#[inline]
pub(crate) fn split_flush(t: &mut Option<Instant>) {
    *t = None;
}


/// `DENNO_Q4_M_TILE` の解釈 (純粋関数)。範囲外・非数・刻み外は既定へ落とす
/// (不正な env で panic させない — `q4_gemv_c()` と同じ規約)。
pub(crate) fn parse_q4_m_tile(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(t) if (Q4_M_TILE_STEP..=Q4_M_TILE_MAX).contains(&t) && t % Q4_M_TILE_STEP == 0 => t,
        _ => Q4_M_TILE,
    }
}


/// ピン留め + 非同期 1 同期パイプライン:
///   H2D(x, async) → quantize kernel → gemv kernel → D2H(out, async) → sync ×1
pub(crate) fn linear_pinned(
    ctx: &CudaCtx,
    a: &[f32],
    gw: &GpuW,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    debug_assert_eq!(a.len(), m * k);
    let lib = lib()?;
    // current context はスレッドローカル → 毎回張り直す (冪端・軽量)
    ctx.make_current()?;
    // s138 (Task B6・計測のみ): 段ごとの tick (計測 OFF は None)。
    let mut t = split_stage();
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        // デストラクチャしてフィールド単位の借用にする (RefMut 経由の
        // 全体借用だと dx (不変) と hx/ho (可変) が競合するため)
        let Scratch { x, aq, asc, out, hx, ho, .. } = &mut *s;
        let bytes_x = m * k * 4;
        let bytes_out = m * n * 4;
        Scratch::ensure(x, bytes_x)?;
        Scratch::ensure(aq, m * k)?;
        Scratch::ensure(asc, m * 4)?;
        Scratch::ensure(out, bytes_out)?;
        split_mark(&mut t, crate::prof::Q4P_OUT_ALLOC);
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        split_mark(&mut t, crate::prof::Q4P_PIN_ALLOC);
        let (dx, daq, dasc, dout) = (
            x.as_ref().unwrap(),
            aq.as_ref().unwrap(),
            asc.as_ref().unwrap(),
            out.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..m * k].copy_from_slice(a);
        split_mark(&mut t, crate::prof::Q4P_H2D_STAGE);
        unsafe {
            check(
                (lib.memcpy_htod_async)(dx.ptr(), hx.ptr as *const c_void, bytes_x, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync",
            )?;
            split_mark(&mut t, crate::prof::Q4P_H2D_ISSUE);
            let (mut px, mut paq, mut pasc) = (dx.ptr(), daq.ptr(), dasc.ptr());
            let (mut mc, mut kc) = (m as c_int, k as c_int);
            let mut args = [
                &mut px as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
            ];
            ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args)?;

            let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                (daq.ptr(), dasc.ptr(), gw.w.ptr(), gw.sc.ptr(), dout.ptr());
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paqi as *mut _ as *mut c_void,
                &mut pasci as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let total = (m * n) as u64;
            let blocks = total.div_ceil(8) as u32;
            ctx.launch_async(ctx.f_mv, (blocks, 1, 1), (256, 1, 1), 0, &mut args)?;
            split_mark(&mut t, crate::prof::Q4P_KERNEL_LAUNCH);

            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            split_mark(&mut t, crate::prof::Q4P_D2H_ISSUE);
            // s65: 発行元の件数分類 (表示専用・時間は d2h_state バケット側)
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        split_mark(&mut t, crate::prof::Q4P_SYNC_WAIT);
        // s138: ピン留め → 呼び出し側 Vec のコピー (sync の後・表示専用)
        let v = ho.as_slice::<f32>()[..m * n].to_vec();
        split_mark(&mut t, crate::prof::Q4P_HOST_COPY);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        split_flush(&mut t);
        Ok(v)
    })
}


// ===========================================================================
// s49: fp32 lm_head の GPU 化 (per-op・dno_matvec_f32 流用・.cu/PTX 無変更)
// ===========================================================================
// serve 既定の fp32 モード (--quant fp32) では lm_head が Linear::F32 のまま
// (untied: output.weight / tied: embed) で CPU matvec (tensor::matvec_bt) が
// decode の主律速だった (s45 実機計測: decode 29ms/step の 46.3% = 19.6ms/step)。
// int8 モードの lm_head は P17b (maybe_gpu_linear) / C4 chain
// (chain_linear_grouped) のいずれでも既に GPU・ビット一致のため本節では触れない。
//
// 発動条件: DENNO_CUDA=1 (cuda_enabled) かつ kernels_elem.ptx に dno_matvec_f32
// が収録済み (3931b70 で再生成済み)。カーネルは C4 chain の beta/alpha 制御投影用
// dno_matvec_f32 を**そのまま**流用 (.cu/PTX 変更なし):
//   - 出力 j = blockIdx.x ごとに d を昇順で
//     s = __fadd_rn(s, __fmul_rn(h[d], row[d])) と累積 = tensor::matvec_bt の
//     `s += a[d] * row[d]` (積と加算の丸め分離・d 逐次) と**ビット一致**
//   - カーネルは block 内の全スレッドが同一計算をする実装のため block=(1,1,1)
//     で起動して冗長スレッドを排除する (chain_q35_ctl_f32 の block=(128) は
//     n=数十が前提。vocab n=151936 では 128 倍の冗長演算になる)。grid.x は
//     2^31-1 まで合法なので n>65535 もそのまま扱える (65535 ガードは chain 側
//     呼び出し規約のみでハードウェア限界ではない)。
// - 失敗時は本経路のみのスコープドラッチ (LMV32_BROKEN) で恒久 CPU フォール
//   バック (GPU_BROKEN にはしない = int8 経路への影響ゼロ)。PTX 未収録は
//   q35_fns と同一の None キャッシュ規約 (ラッチ無し・次回プロセス起動まで
//   不発動)。数値はどちらの経路でも同一 (bit 一致) なのでフォールバックは
//   出力を変えない。
// - 重みは呼び出し側が渡すモデルロード毎に一意な key (qwen35 の mid 由来) で
//   グローバル登録し 1 回だけ同期 H2D する (fp32 9B untied で ~2.4GB・初回 1 回
//   のみ。以後は h 16KB の H2D + logits 0.6MB の D2H + sync ×1/ステップ)。

/// s49: fp32 lm_head 経路のスコープド恒久ラッチ (GPU_BROKEN とは独立。
/// 本経路の失敗で int8 経路を落とさない)。
pub(crate) static LMV32_BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// s49: dno_matvec_f32 のハンドル (lazy fetch・未収録 PTX では None をキャッシュ
/// して恒久不発動 = q35_fns と同一規約・GPU_BROKEN にはしない)。
/// 生ハンドルの静的保持は Q35Fns と同一の規約 (所有権移動なし・driver が
/// スレッド安全に管理) で Send/Sync を与える。
#[derive(Clone, Copy)]
pub(crate) struct Lmv32Fn(CUfunction);


/// dno_matvec_f32 のハンドルキャッシュ。s49 (fp32 lm_head) と task 69 (MoE
/// ルータ) が**同じカーネル**を共有するので 1 本にまとめてある。
static MATVEC_F32_FNS: OnceLock<Option<Lmv32Fn>> = OnceLock::new();


/// `dno_matvec_f32` のハンドル (未収録 PTX では None をキャッシュして恒久不発動・
/// ラッチはしない)。カーネルは 1 行 = 1 ブロックで k を昇順に逐次加算し、
/// `mul.rn.f32` + `add.rn.f32` のみ (fma 0 個・ftz 0 個) を PTX で確認済み。
pub(crate) fn matvec_f32_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    MATVEC_F32_FNS
        .get_or_init(|| {
            let lib = lib().ok()?;
            let mut f: CUfunction = std::ptr::null_mut();
            let r = unsafe {
                (lib.module_get_function)(
                    &mut f,
                    ctx.module_elem,
                    b"dno_matvec_f32\0".as_ptr() as *const c_char,
                )
            };
            if r != 0 {
                None
            } else {
                Some(Lmv32Fn(f))
            }
        })
        .as_ref()
        .map(|h| h.0)
}


/// task 121: `dno_matvec_f32_r` (s169 の並列リダクション版) のハンドル。hc mix
/// GPU 経路の長い行 (k >= n) 用。[`matvec_f32_fn`] と同一の lazy fetch 規約で、
/// 未収録 PTX では None をキャッシュする (GPU_BROKEN にはしない = 呼び出し側が
/// 逐次版に落ちるだけ)。生ハンドルの静的保持は Lmv32Fn / Q35Fns と同一の規約
/// (所有権移動なし・driver がスレッド安全に管理) で Send/Sync を与える。
#[derive(Clone, Copy)]
pub(crate) struct Lmv32RFn(CUfunction);


static MATVEC_F32_R_FNS: OnceLock<Option<Lmv32RFn>> = OnceLock::new();


/// `dno_matvec_f32_r` のハンドル (未収録 PTX では None)。1 行 = 1 ブロックで
/// k を blockDim スレッドに分割して strided 加算し、warp shuffle の固定木
/// (o = 16,8,4,2,1) → warp 合計を warp id 昇順に加算して縮約する (atomic 無し =
/// 実行ごとに同一結果)。逐次版 `dno_matvec_f32` とは丸め順が違うため bit 一致は
/// しない。launch は grid=(n,1,1) / block=(256,1,1) (s169 の
/// chain_q35_ctl_f32 と同一幾何)。
pub(crate) fn matvec_f32_r_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    MATVEC_F32_R_FNS
        .get_or_init(|| {
            let lib = lib().ok()?;
            let mut f: CUfunction = std::ptr::null_mut();
            let r = unsafe {
                (lib.module_get_function)(
                    &mut f,
                    ctx.module_elem,
                    b"dno_matvec_f32_r\0".as_ptr() as *const c_char,
                )
            };
            if r != 0 {
                None
            } else {
                Some(Lmv32RFn(f))
            }
        })
        .as_ref()
        .map(|h| h.0)
}


/// task 138: `dno_matvec_f32_r_silu` のハンドル (hc mix の down + lo_apply を
/// 1 カーネルに畳んだ版)。[`matvec_f32_r_fn`] と同じ lazy fetch 規約・未収録
/// PTX では None をキャッシュする (呼び出し側は従来の 2 往復経路に落ちる)。
#[derive(Clone, Copy)]
pub(crate) struct Lmv32RSiluFn(CUfunction);


static MATVEC_F32_R_SILU_FNS: OnceLock<Option<Lmv32RSiluFn>> = OnceLock::new();


pub(crate) fn matvec_f32_r_silu_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    MATVEC_F32_R_SILU_FNS
        .get_or_init(|| {
            let lib = lib().ok()?;
            let mut f: CUfunction = std::ptr::null_mut();
            let r = unsafe {
                (lib.module_get_function)(
                    &mut f,
                    ctx.module_elem,
                    b"dno_matvec_f32_r_silu\0".as_ptr() as *const c_char,
                )
            };
            if r != 0 {
                None
            } else {
                Some(Lmv32RSiluFn(f))
            }
        })
        .as_ref()
        .map(|h| h.0)
}


// ---------------------------------------------------------------------------
// task 152: BF16 のまま読む mix カーネルのハンドル (未収録 PTX は None)
// ---------------------------------------------------------------------------
// 追加カーネルは kernels_elem.cu の末尾 (dno_matvec_bf16 / _r / _r_silu /
// dno_gemm_bf16_mt)。未収録 (古い PTX) のときは None をキャッシュし、呼び出し側は
// **f32 展開経路**に落ちる (ラッチしない = 失敗にしない)。モジュールは f32 版と
// 同じ `module_elem`。

/// task 152: `dno_matvec_bf16` のハンドル。
pub(crate) fn matvec_bf16_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    static FN: OnceLock<Option<Lmv32Fn>> = OnceLock::new();
    FN.get_or_init(|| {
        let lib = lib().ok()?;
        let mut f: CUfunction = std::ptr::null_mut();
        let r = unsafe {
            (lib.module_get_function)(
                &mut f,
                ctx.module_elem,
                b"dno_matvec_bf16\0".as_ptr() as *const c_char,
            )
        };
        if r != 0 {
            None
        } else {
            Some(Lmv32Fn(f))
        }
    })
    .as_ref()
    .map(|h| h.0)
}


/// task 152: `dno_matvec_bf16_r` のハンドル。
pub(crate) fn matvec_bf16_r_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    static FN: OnceLock<Option<Lmv32RFn>> = OnceLock::new();
    FN.get_or_init(|| {
        let lib = lib().ok()?;
        let mut f: CUfunction = std::ptr::null_mut();
        let r = unsafe {
            (lib.module_get_function)(
                &mut f,
                ctx.module_elem,
                b"dno_matvec_bf16_r\0".as_ptr() as *const c_char,
            )
        };
        if r != 0 {
            None
        } else {
            Some(Lmv32RFn(f))
        }
    })
    .as_ref()
    .map(|h| h.0)
}


/// task 152: `dno_matvec_bf16_r_silu` のハンドル。
pub(crate) fn matvec_bf16_r_silu_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    static FN: OnceLock<Option<Lmv32RSiluFn>> = OnceLock::new();
    FN.get_or_init(|| {
        let lib = lib().ok()?;
        let mut f: CUfunction = std::ptr::null_mut();
        let r = unsafe {
            (lib.module_get_function)(
                &mut f,
                ctx.module_elem,
                b"dno_matvec_bf16_r_silu\0".as_ptr() as *const c_char,
            )
        };
        if r != 0 {
            None
        } else {
            Some(Lmv32RSiluFn(f))
        }
    })
    .as_ref()
    .map(|h| h.0)
}


/// s49: fp32 重み (lm_head / tied embed) のデバイスキャッシュ。key はモデル
/// ロード毎に一意 (qwen35 の mid 由来)。無ければ同期 H2D でアップロードし
/// 初回 1 回のみ走る (eviction 無し = P17b 重みレジストリと同一方針)。
pub(crate) static LMV32_W: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, DevBuffer>>> =
    OnceLock::new();


/// task 69: f32 重みを (初回だけ) VRAM へ上げて常駐エントリを返す。
///
/// レジストリは GGUF ブロックと同じ (`gguf_registry`)。予算・退避・使用中ピン・
/// 失敗時の会計は `weights_for_gguf` と同一の規約で、違うのは行の単位
/// (k * 4 バイト固定) だけ。
pub(crate) fn weights_for_f32(uid: u64, bytes: &[u8], k: usize, n: usize) -> Result<GgufWRef> {
    weights_for_gemm_f32(uid, bytes, k, n, crate::kquant::GgmlType::F32)
}


/// task 69: ルータ f32 重みの常駐エントリを解放する (`GgufLinear::drop` から)。
/// 載っていない uid・量子化型の uid は触らない。
pub fn unregister_f32_weights(uid: u64) {
    let Ok(mut g) = gguf_registry().lock() else {
        return;
    };
    let is_f32 = g
        .map
        .get(&uid)
        .map(|e| e.ty == crate::kquant::GgmlType::F32)
        .unwrap_or(false);
    if is_f32 {
        g.map.remove(&uid);
        let freed = g.lru.note_remove(uid);
        if freed > 0 {
            crate::gpu_budget::note_weight_resident(g.map.len(), g.lru.used_bytes());
        }
    }
}


/// task 192: `DENNO_DEC_ROUTER_EARLY` の解釈 (純関数)。
///
/// **既定 off**。lab t192a では decode +0.0% で、ルータ GEMV を先に積んでも
/// 待ちは縮まらなかった (GPU のキューが既に詰まっていて、積む順は効かない)。
/// 明示指定 (`1` / `on` / `true`) のときだけ on。
fn parse_dec_router_early(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}


/// ルータ GEMV を共有 expert の enqueue より先に積むか (`DENNO_DEC_ROUTER_EARLY`)。
///
/// lab t187c の `[q4ledger]` では router が 1 トークン 1.26 ms (gpu 0.64 /
/// sync 0.63 / cpu 0.39、48 回同期) だった。同期の待ちは**ルータ自身の GEMV が
/// 終わるのを待つ**時間なので、GEMV を先に積んでおけば、共有 expert の
/// enqueue (ホスト ~7 us/層) の間だけ GPU が先に進む。
pub fn dec_router_early_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_dec_router_early(std::env::var("DENNO_DEC_ROUTER_EARLY").ok().as_deref())
    })
}


/// task 145: `dno_gemm_f32_mt` (f32 重み × m トークンの GEMM) のハンドル。
///
/// kernels_elem.ptx に未収録なら None をキャッシュする (呼び出し側は CPU の
/// バッチ経路へ落ちるだけで、失敗にはしない = fail closed)。
#[derive(Clone, Copy)]
pub(crate) struct LgemmMtFn(CUfunction);


pub(crate) static GEMM_F32_MT_FNS: OnceLock<Option<LgemmMtFn>> = OnceLock::new();


/// task 145 のカーネル幾何 (kernels_elem.cu の `dno_gemm_f32_mt` と 1:1)。
pub const GEMM_MT_ROWS: usize = 16;

/// 1 ブロックが担当するトークン数。
pub const GEMM_MT_TOKS: usize = 32;


/// task 182: `out[t * n + r] = dot(row r, xs[t])` を GPU の f32 GEMM で回す。
///
/// 重みは [`weights_for_f32`] の常駐 (router の 1 行 GEMV と同じ uid 表) を使う。
/// `nt < 2`・形の不一致・GPU 無し・予算外・PTX 未収録はすべて `false` を返して
/// CPU 経路へ落とす (panic しない)。
pub fn maybe_gpu_f32_gemm_batch_into(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    out: &mut [f32],
    src_ty: crate::kquant::GgmlType,
) -> bool {
    #[cfg(test)]
    F32_GEMM_BATCH_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if xs.len() != nt * k {
        return false;
    }
    f32_gemm_batch_run(uid, bytes, k, n, BatchIn::Host(xs), nt, out, src_ty)
}


/// task 209: `xs` が既にデバイスにある版 (ホストからの H2D をしない)。
///
/// `dev_x` は `[nt][k]` f32 のデバイス領域。router が `pf_mix` の出力
/// (`mixed`) をそのまま読むために使う。ホスト版と**同じカーネル・同じ入力
/// バイト列**なので値は変わらない。
pub fn maybe_gpu_f32_gemm_batch_dev_into(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    dev_x: CUdeviceptr,
    nt: usize,
    out: &mut [f32],
    src_ty: crate::kquant::GgmlType,
) -> bool {
    #[cfg(test)]
    F32_GEMM_BATCH_ATTEMPTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if dev_x == 0 {
        return false;
    }
    f32_gemm_batch_run(uid, bytes, k, n, BatchIn::Dev(dev_x), nt, out, src_ty)
}


/// 入力の置き場所 (ホストのスライス / 既にデバイスにある領域)。
pub(crate) enum BatchIn<'a> {
    Host(&'a [f32]),
    Dev(CUdeviceptr),
}


pub(crate) const PF_NAMES: [&[u8]; 5] = [
    b"dno_pf_rmsgroup\0",
    b"dno_pf_lo\0",
    b"dno_pf_mixed\0",
    b"dno_pf_inj\0",
    b"dno_pf_combine\0",
];


/// task 154: 常駐連鎖の PTX。lab t154e が `PTX=pf` で `kernels_pf.cu` から
/// 再生成したものをそのまま収録した (5 本の entry を含む)。
///
/// `include_bytes!` にする理由: 既定 on の経路なので、実行時にクレート配下の
/// `kernels/` が無い配置でも黙って CPU へ落ちないようにする (task 151 の
/// `kernels_exp_iq_pf.ptx` はファイル読みだが、あちらは既定 on の判断が後追い)。
const PTX_PF: &[u8] = include_bytes!("../../kernels/kernels_pf.ptx");


/// `DENNO_PF_PTX` があればそのイメージ、無ければコミット済みの [`PTX_PF`]。
pub(crate) fn pf_ptx_image() -> Option<Vec<u8>> {
    match std::env::var("DENNO_PF_PTX") {
        Ok(p) if !p.is_empty() => {
            let path = std::path::Path::new(&p);
            match std::fs::read(path) {
                Ok(b) => Some(b),
                Err(e) => {
                    tracing::info!(
                        "[pf] PTX が読めない ({}): {e} → CPU 経路",
                        path.display()
                    );
                    None
                }
            }
        }
        _ => Some(PTX_PF.to_vec()),
    }
}


/// 連鎖 1 回ぶんの失敗を 1 度だけ記録する (以降は CPU 経路に固定)。
pub(crate) static PF_BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


// ---------------------------------------------------------------------------
// task 617: F32 の inject 重みを BF16 に丸めて k 分割 GEMV へ回す
// ---------------------------------------------------------------------------
// `pf_weight` の F32 腕は重みを f32 のまま上げ、汎用の `dno_gemm_f32_mt`
// (1 ブロック 16 行 x 32 トークン) で回していた。既定モデル
// (flash-next-UD-IQ3_XXS) の hc inject (`hc_*_inject`) は **F32** で n = hc = 4
// なので、16 行タイルの 12 行が空回りする。nsys (t613a) では
// `dno_gemm_f32_mt` が 960 呼び出し 平均 315 us = 302 ms (14k と 3,826 の
// prefill 合計)。BF16 だった旧モデルは task 198 で `dno_mix_inj2_bf16`
// (k を 256 スレッドに分割・1 ブロック 1 トークン) へ移っていて、同じ形が
// 40-50 us/呼び出しで済む。ここでは **重みだけ** BF16 に丸めて (RNE) その経路へ
// 載せる。
//
// 数値: 重みの丸めだけが変わる (inject の値が相対 1e-3 程度動き、sigmoid を
// 通って hc combine の倍率になる)。演算の意味・決定性は変わらない。
// 切り戻しは `DENNO_PF_INJ_F32_BF16=0`。

/// task 617: F32 の inject 重みを BF16 に丸めて k 分割 GEMV へ回すか
/// (**既定 on**。lab t617a/t617b で採否 yes。`=0` で切り戻し)。
pub const PF_INJ_F32_BF16_ENV: &str = "DENNO_PF_INJ_F32_BF16";


/// [`PF_INJ_F32_BF16_ENV`] の解釈 (純関数)。**既定 on** (t617a/t617b)。
/// 明示的な偽値 (`0` / `off` / `false` / `no` / 空) だけが off
/// (`DENNO_PF_MIX_INJ2` と同じ表)。
pub(crate) fn pf_inj_f32_bf16_from_value(v: Option<&str>) -> bool {
    match v {
        None => true,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


/// ゲートの実効値 (プロセスで 1 回だけ env を読む)。
pub(crate) fn pf_inj_f32_bf16_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_inj_f32_bf16_from_value(std::env::var(PF_INJ_F32_BF16_ENV).ok().as_deref())
    })
}


/// その形を BF16 化して k 分割の inject へ回してよいか (純関数)。
///
/// inject の実形は k = hc*n_embd = 10,240 / n = hc = 4。トークン数は
/// [`mix_inj2_ok`] 側で見る (重みはトークン数に依らない)。
pub(crate) fn pf_inj_bf16_shape(k: usize, n: usize) -> bool {
    k > 0 && n >= 1 && n <= MIX_INJ2_MAX_N
}


/// f32 のバイト列 (リトルエンディアン、k*n 個) を BF16 (上位 16 bit・RNE) へ畳む。
///
/// 純関数 (テストで固定する)。NaN は上位 16 bit に落とすだけで inf に化けさせない。
pub(crate) fn f32_bytes_to_bf16_bytes(src: &[u8]) -> Option<Vec<u8>> {
    if src.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(src.len() / 2);
    for c in src.chunks_exact(4) {
        let b = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        let hi = if b & 0x7fff_ffff > 0x7f80_0000 {
            // NaN (指数が全部 1 で仮数が非 0)。
            ((b >> 16) | 0x0040) as u16
        } else {
            // round-to-nearest-even。
            let lsb = (b >> 16) & 1;
            (b.wrapping_add(0x7fff + lsb) >> 16) as u16
        };
        out.extend_from_slice(&hi.to_le_bytes());
    }
    Some(out)
}


/// task 617: inject を k 分割カーネルで撃った回数 (発火の証拠。`[pf-mix-kprof]` の
/// `inj2_calls=` に出る)。
pub(crate) static PF_INJ2_CALLS: AtomicU64 = AtomicU64::new(0);


/// [`PF_INJ2_CALLS`] を読む。
pub(crate) fn pf_inj2_calls() -> u64 {
    PF_INJ2_CALLS.load(Ordering::Relaxed)
}


/// 計数を 0 に戻す (リクエスト終端)。
pub(crate) fn pf_inj2_calls_reset() {
    PF_INJ2_CALLS.store(0, Ordering::Relaxed);
}


thread_local! {
    /// task 189: 自ストリーム同期用のイベント (計時不要)。
    pub(crate) static PF_SYNC_EV: std::cell::Cell<CUevent> = const { std::cell::Cell::new(std::ptr::null_mut()) };
}


/// task 334: 段の終端の待ち。`DENNO_PF_STAGE_NOSYNC` が on なら**待たない**。
///
/// 待たなくても、次の段は同じ (レガシー既定) ストリームへの非同期起動か、
/// 同じストリームの同期コピーなので、ストリーム順で前のカーネルを待つ。
/// ホストが結果を読む経路は必ず同期コピーを通る。ステアリングのように
/// ホストが直後に読む経路 (`pf_steer_*`) はこの関数を使わない。
pub(crate) fn pf_stage_sync_maybe(ctx: &CudaCtx) -> Result<Option<std::time::Instant>> {
    if crate::model::exp_pf::stage_nosync_on() {
        return Ok(None);
    }
    pf_stage_sync(ctx)
}


/// task 334: prefill の attention 連鎖 (full / ssm / shared) の段の終端。
/// `DENNO_PF_ATTN_NOSYNC` が on なら**待たない**。
///
/// 待った直後にホストが読むのは同期コピー (`to_slice` / `to_vec`) で落とした
/// 結果だけで、そのコピー自体が同じストリームの前のカーネルを待つ。デバイスに
/// 残す経路 (`kept_dev`) は次の段が同じストリームのカーネル。よって待たなくても
/// 値は変わらない。段ごとの壁時計 (`phase_ns`) は「起動だけ」の値になる。
pub(crate) fn pf_chain_sync(ctx: &CudaCtx) -> Result<()> {
    if crate::model::exp_pf::attn_nosync_on() {
        return Ok(());
    }
    ctx.sync()
}


/// task 189: mix / combine の同期。ゲートが on なら自ストリームだけを待つ。
///
/// 戻り値は「別ストリームを待った時間」を測るための `Instant` (ゲート off / 計測
/// off では `None`)。呼び出し側は `pf_mix_kprof::add_host(SYNC_OTHER, t)` に渡す。
fn pf_stage_sync(ctx: &CudaCtx) -> Result<Option<std::time::Instant>> {
    if !mix_stage_on() {
        let t = if pf_mix_kprof::on() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        ctx.sync()?;
        return Ok(t);
    }
    // 1. 自ストリーム (mix の 6 起動) の完了だけを待つ。
    let t_own = pf_mix_kprof::on().then(std::time::Instant::now);
    pf_sync_own_stream()?;
    pf_mix_kprof::add_host(pf_mix_kprof::SYNC_OWN, t_own);
    // 2. 計測時だけ、そのあと全体を待って「別ストリームの残り」を測る
    //    (診断専用。ゲート on の実経路では 2 回目の同期はしない)。
    if pf_mix_kprof::on() {
        let t = Some(std::time::Instant::now());
        ctx.sync()?;
        return Ok(t);
    }
    Ok(None)
}


pub(crate) fn pf_fail(msg: &str) -> bool {
    if !PF_BROKEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::info!("[pf] {msg} → 以降の prefill は CPU 経路");
    }
    false
}


// ---------------------------------------------------------------------------
// task 257/265: prefill 作業域のプール (expert から借りて、prefill の終端で返す)
// ---------------------------------------------------------------------------
//
// task 250 は層優先 + tensor core で 14k cold prefill を 25.4 s → 9.9-15.0 s に
// 縮めたが、作業域 (1.3-2.2 GB) を expert がカードを埋めた**後**に確保しようと
// して失敗し、追い出し階段が expert を 33,497 MiB 全捨てした (decode 18.4 →
// 32-52 ms/tok)。task 257 は起動時に 1 本確保して解決したが、expert の予算を
// そのぶん削ったので 14k 後の decode が +11.3% になった (lab t257b)。
//
// task 265 は **prefill のあいだだけ借りる**: 長い prefill の入口で 1 本だけ
// `alloc_tagged` に出し (足りない分は task 256 の経路がチャンク単位で expert を
// 追い出して空ける)、終端で `cuMemFree` する。expert の LRU 上限は実空きから
// 引き直される (`expert_vram::effective_expert_cap`) ので、返せば自動で戻る。
//
// レイアウトは [`crate::model::pf_pool::layout`] (純関数) が形から組む。
// 借りられないときは従来の `alloc_tagged` に落ちる (fail open)。

/// 借りているプール本体 (1 prefill のあいだだけ `Some`)。
///
/// 中身は非所有ビュー ([`pool_view`]) が指すので、**返す前にビューを落とす**
/// ([`pf_pool_release`] が 3 つの置き場を先に片付ける)。
#[cfg(feature = "cuda")]
pub(crate) static PF_POOL: std::sync::Mutex<Option<DevBuffer>> = std::sync::Mutex::new(None);


/// task 279: プールの確保が追い出した expert の鍵 (`pf_pool_take_evicted` で取り出す)。
///
/// 借りた作業域を返したあと、呼び出し側がこれを LRU へ戻す (decode の作業集合を
/// prefill 前の状態に戻す)。取り出したら空になる。
#[cfg(feature = "cuda")]
pub(crate) static PF_POOL_EVICTED: std::sync::Mutex<Vec<crate::model::expert_cache::ExpertKey>> =
    std::sync::Mutex::new(Vec::new());


#[cfg(feature = "cuda")]
thread_local! {
    /// このスレッドの prefill で使う区画 (リクエストの先頭で組み直す)。
    pub(crate) static PF_POOL_LAYOUT: std::cell::RefCell<Option<crate::model::pf_pool::Layout>> =
        const { std::cell::RefCell::new(None) };
    /// このスレッドがプールを借りたか (= 返す責任があるか)。
    pub(crate) static PF_POOL_OWNED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}


/// task 265: 区画 `r` の `idx` 枚目を指す非所有ビュー (借りていなければ `None`)。
#[cfg(feature = "cuda")]
pub(crate) fn pool_view(r: crate::model::pf_pool::R, idx: usize, want: usize) -> Option<DevBuffer> {
    let l = PF_POOL_LAYOUT.with(|c| *c.borrow())?;
    let g = l.region(r);
    let u = l.unit(r);
    let slots = match r {
        crate::model::pf_pool::R::LmInj => l.lm_inj_slots,
        crate::model::pf_pool::R::Stage => l.stage_slots,
        crate::model::pf_pool::R::DevPark => l.dev_park_slots,
        _ => 1,
    };
    if idx >= slots || u == 0 || u < want {
        return None;
    }
    let off = g.off.checked_add(idx.checked_mul(u)?)?;
    let pool = PF_POOL.lock().ok()?;
    let pool = pool.as_ref()?;
    if off.checked_add(u)? > pool.bytes() {
        return None;
    }
    Some(pool.view_at(off, u))
}


/// task 257: 区画があればそこを指すビューを `cap` に入れ、無ければ従来どおり確保する。
#[cfg(feature = "cuda")]
pub(crate) fn ensure_region(
    cap: &mut Option<DevBuffer>,
    r: crate::model::pf_pool::R,
    idx: usize,
    want: usize,
) -> anyhow::Result<()> {
    if let Some(v) = pool_view(r, idx, want) {
        *cap = Some(v);
        return Ok(());
    }
    MoeScratch::ensure(cap, want)
}


#[cfg(feature = "cuda")]
thread_local! {
    pub(crate) static PF_LM: std::cell::RefCell<PfLmDev> = std::cell::RefCell::new(PfLmDev::default());
}


/// task 461: 残差のデバイス域 (`pf_upload_res` が使う `PF_DEV.res`) の先頭。
///
/// PLE のデバイス経路 (`pf_ple_dev_run`) が、ホストへ落とさずにその場で
/// 加算するために読む。無ければ 0 (= 従来経路へ)。
pub(crate) fn pf_res_dev_ptr() -> CUdeviceptr {
    PF_DEV.with(|cell| -> CUdeviceptr {
        let d = cell.borrow();
        match d.res.as_ref() {
            Some(b) => b.ptr(),
            None => 0,
        }
    })
}


/// F32 の重み (hc mix の inject など) の常駐表。`mix_gpu_weight` は BF16 専用なので
/// F32 はそのままのバイト列を置く (1 本 40 KB)。
pub(crate) static PF_F32_W: OnceLock<Mutex<std::collections::HashMap<u64, DevBuffer>>> = OnceLock::new();


/// task 617: F32 の inject 重みを BF16 に丸めた写しの常駐表 (uid -> BF16 バイト列)。
///
/// `PF_F32_W` とは別に持つ (f32 のままが要る経路があっても壊さない)。1 本 80 KB。
pub(crate) static PF_INJ_BF_W: OnceLock<Mutex<std::collections::HashMap<u64, DevBuffer>>> = OnceLock::new();


// ---------------------------------------------------------------------------
// task 457: prefill hc mix の要素カーネル第 2 版 (kernels_pf_elem.cu、既定 on / lab t457b)
// ---------------------------------------------------------------------------
//
// 動機は kernels_pf_elem.cu の先頭。既存の `dno_pf_rmsgroup` / `dno_pf_mixed`
// (kernels_pf.ptx、include_bytes で固定) は**触らず**、新しい PTX を実行時に読む。
// 読めない / entry が無い / 引数の数が違う / 形が外れる、のどれでも従来カーネルへ
// fail closed (ゲート on でも既定と同じ経路・同じ値)。

/// norm 版のスレッド数 (カーネルの `PFE_THREADS`)。
pub(crate) const PF_ELEM_NORM_THREADS: usize = 256;

/// 1 スレッドがレジスタに持てる要素数 (カーネルの `PFE_MAX_PER`)。
pub(crate) const PF_ELEM_MAX_PER: usize = 16;


thread_local! {
    pub(crate) static PLE_DEV: std::cell::RefCell<PfPleDev> = const { std::cell::RefCell::new(PfPleDev {
        flat: None,
        key: None,
        value: None,
        gate: None,
        invc: None,
        part: None,
        wgk: None,
        wgq: None,
        wgc: None,
        wwt: None,
        hist: Vec::new(),
        noted: Vec::new(),
    }) };
}


/// task 154: 連鎖の作業域を解放する (VRAM を返す。prefill の終端で呼ぶ)。
pub fn pf_release() {
    PF_DEV.with(|cell| {
        let mut d = cell.borrow_mut();
        *d = PfDev::new();
    });
}


/// task 154: 連鎖が 1 度でも失敗したか (テスト・診断用)。
pub fn pf_broken() -> bool {
    PF_BROKEN.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 154: テスト用にラッチを戻す。
#[cfg(test)]
pub fn pf_reset_for_test() {
    PF_BROKEN.store(false, std::sync::atomic::Ordering::Relaxed);
    pf_release();
}


/// 常駐連鎖の記録用の加算域を確保する (無ければ 0 で作る)。
///
/// 毎リクエストの入口で呼んでよい (既にあれば何もしない = 前のリクエストの
/// 続きから足せる)。使えないときは false。
pub fn pf_steer_acc_ensure(n_layer: usize, hc: usize, e: usize) -> bool {
    if n_layer == 0 || hc == 0 || e == 0 || n_layer * hc * e > (1 << 26) {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return pf_fail("steer: CUDA context が無い");
    };
    if ctx.make_current().is_err() {
        return pf_fail("steer: cuCtxSetCurrent 失敗");
    }
    if steer_kernels().is_none() {
        return false;
    }
    STEER_DEV.with(|cell| -> bool {
        let mut d = cell.borrow_mut();
        if let Err(e) = d.fit(n_layer, hc, e) {
            return pf_fail(&format!("steer: 形の更新に失敗: {e:#}"));
        }
        if d.acc.as_ref().map(|b| b.bytes()).unwrap_or(0) < n_layer * hc * e * 4 {
            match DevBuffer::alloc_tagged(n_layer * hc * e * 4, crate::mem_audit::TAG_OTHER) {
                Ok(b) => {
                    if let Err(e) = b.copy_from(&vec![0.0f32; n_layer * hc * e]) {
                        return pf_fail(&format!("steer: 加算域の初期化に失敗: {e:#}"));
                    }
                    d.acc = Some(b);
                }
                Err(e) => return pf_fail(&format!("steer: 加算域の確保に失敗: {e:#}")),
            }
        }
        // 記録を始めた直後の 1 回は 0 に戻す (前の記録の残りを引き継がない)。
        if crate::model::steer::take_dev_reset() {
            if let Some(b) = d.acc.as_ref() {
                let n = b.bytes() / 4;
                if let Err(e) = b.copy_from(&vec![0.0f32; n]) {
                    return pf_fail(&format!("steer: 加算域の 0 クリアに失敗: {e:#}"));
                }
            }
        }
        true
    })
}


/// 加算域を 0 に戻す (記録を始めるとき)。
pub fn pf_steer_acc_reset() -> bool {
    STEER_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let Some(b) = d.acc.as_ref() else {
            return true;
        };
        let n = b.bytes() / 4;
        match b.copy_from(&vec![0.0f32; n]) {
            Ok(()) => true,
            Err(e) => pf_fail(&format!("steer: 加算域の 0 クリアに失敗: {e:#}")),
        }
    })
}


/// task 218: 1 層ぶん加算する (`res` は `[nt][hc][e]` = 直前に pf_combine が
/// 書いた残差)。
pub fn pf_steer_acc(nt: usize, hc: usize, e: usize, layer: usize) -> bool {
    pf_steer_acc_from(nt, hc, e, layer, 0)
}

/// task 829: `[nt][hc][e]` のうち最後の 1 行 (`nt - 1`) だけを加算する。
///
/// 同じカーネルを `nt = 1` で、行の先頭ポインタをずらして当てる
/// (新しい PTX は要らない。値は `pf_steer_acc(1, ...)` を最後の行に当てたのと同じ)。
pub fn pf_steer_acc_last(nt: usize, hc: usize, e: usize, layer: usize) -> bool {
    if nt == 0 {
        return true;
    }
    pf_steer_acc_from(1, hc, e, layer, nt - 1)
}

fn pf_steer_acc_from(nt: usize, hc: usize, e: usize, layer: usize, row0: usize) -> bool {
    if nt == 0 {
        return true;
    }
    let Some(k) = steer_kernels() else {
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        return pf_fail("steer acc: CUDA context が無い");
    };
    if ctx.make_current().is_err() {
        return pf_fail("steer acc: cuCtxSetCurrent 失敗");
    }
    STEER_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let Some(acc) = d.acc.as_ref() else {
            return pf_fail("steer acc: 加算域が無い");
        };
        PF_DEV.with(|pc| -> bool {
            let pd = pc.borrow();
            let Some(res) = pd.res.as_ref() else {
                return pf_fail("steer acc: 残差の作業域が無い");
            };
            let stride = hc * e;
            let (mut pres, mut pacc) = (res.ptr() + (row0 * stride * 4) as u64, acc.ptr());
            let (mut nt_c, mut hc_c, mut e_c, mut l_c) =
                (nt as i32, hc as i32, e as i32, layer as i32);
            let grid = (stride.div_ceil(256) as u32, 1, 1);
            let mut args = [
                &mut pres as *mut _ as *mut c_void,
                &mut pacc as *mut _ as *mut c_void,
                &mut nt_c as *mut _ as *mut c_void,
                &mut hc_c as *mut _ as *mut c_void,
                &mut e_c as *mut _ as *mut c_void,
                &mut l_c as *mut _ as *mut c_void,
            ];
            if let Err(e) = unsafe { ctx.launch_async(k.acc, grid, (256, 1, 1), 0, &mut args) } {
                return pf_fail(&format!("steer acc launch: {e:#}"));
            }
            let t = match pf_stage_sync(ctx) {
                Ok(t) => t,
                Err(e) => return pf_fail(&format!("steer acc sync: {e:#}")),
            };
            pf_mix_kprof::add_host(pf_mix_kprof::SYNC_OTHER, t);
            true
        })
    })
}


/// 加算域をホストへ降ろして 0 に戻す。`out` は `[n_layer][hc][e]`。
pub fn pf_steer_acc_take(out: &mut [f32]) -> bool {
    STEER_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let Some(b) = d.acc.as_ref() else {
            return false;
        };
        if out.len() * 4 > b.bytes() {
            return false;
        }
        let ok = match b.to_slice(out) {
            Ok(()) => true,
            Err(e) => pf_fail(&format!("steer: 加算域の D2H に失敗: {e:#}")),
        };
        if ok {
            if let Err(e) = b.copy_from(&vec![0.0f32; out.len()]) {
                return pf_fail(&format!("steer: 加算域の 0 クリアに失敗: {e:#}"));
            }
        }
        ok
    })
}


/// デバイスの作業域を捨てる (VRAM を返す)。
pub fn pf_steer_release() {
    STEER_DEV.with(|cell| {
        let mut d = cell.borrow_mut();
        *d = SteerDev::new();
    });
}


/// 適用する合成ベクトル `v` (`[n_layer][hc][e]`) をデバイスへ置く。
///
/// `n_layer * hc * e` は記録と同じ上限。使えないときは false。
pub fn pf_steer_apply_begin(v: &[f32], n_layer: usize, hc: usize, e: usize) -> bool {
    if n_layer == 0 || hc == 0 || e == 0 || v.len() != n_layer * hc * e {
        return pf_fail("steer apply: v の形が違う");
    }
    let Some(ctx) = shared_ctx() else {
        return pf_fail("steer apply: CUDA context が無い");
    };
    if ctx.make_current().is_err() {
        return pf_fail("steer apply: cuCtxSetCurrent 失敗");
    }
    if steer_kernels().is_none() {
        return false;
    }
    STEER_DEV.with(|cell| -> bool {
        let mut d = cell.borrow_mut();
        if let Err(e) = d.fit(n_layer, hc, e) {
            return pf_fail(&format!("steer apply: 形の更新に失敗: {e:#}"));
        }
        if d.v.as_ref().map(|b| b.bytes()).unwrap_or(0) < v.len() * 4 {
            match DevBuffer::alloc_tagged(v.len() * 4, crate::mem_audit::TAG_OTHER) {
                Ok(b) => d.v = Some(b),
                Err(e) => return pf_fail(&format!("steer apply: v の確保に失敗: {e:#}")),
            }
        }
        match d.v.as_ref().map(|b| b.copy_from(v)) {
            Some(Ok(())) => true,
            Some(Err(e)) => pf_fail(&format!("steer apply: v の H2D に失敗: {e:#}")),
            None => pf_fail("steer apply: v が無い"),
        }
    })
}


/// 1 層ぶん適用する。`mode` は 0 = subtract / 1 = project。
pub fn pf_steer_apply(
    nt: usize,
    hc: usize,
    e: usize,
    layer: usize,
    mode: u8,
    alpha: f32,
) -> bool {
    if nt == 0 {
        return true;
    }
    let Some(k) = steer_kernels() else {
        return false;
    };
    let Some(ctx) = shared_ctx() else {
        return pf_fail("steer apply: CUDA context が無い");
    };
    if ctx.make_current().is_err() {
        return pf_fail("steer apply: cuCtxSetCurrent 失敗");
    }
    STEER_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        let Some(v) = d.v.as_ref() else {
            return pf_fail("steer apply: v がデバイスに無い");
        };
        PF_DEV.with(|pc| -> bool {
            let pd = pc.borrow();
            let Some(res) = pd.res.as_ref() else {
                return pf_fail("steer apply: 残差の作業域が無い");
            };
            let stride = hc * e;
            let (mut pres, mut pv) = (res.ptr(), v.ptr());
            let (mut nt_c, mut hc_c, mut e_c, mut l_c) =
                (nt as i32, hc as i32, e as i32, layer as i32);
            let mut a_c = alpha;
            let r = unsafe {
                match mode {
                    0 => {
                        let mut args = [
                            &mut pres as *mut _ as *mut c_void,
                            &mut pv as *mut _ as *mut c_void,
                            &mut nt_c as *mut _ as *mut c_void,
                            &mut hc_c as *mut _ as *mut c_void,
                            &mut e_c as *mut _ as *mut c_void,
                            &mut l_c as *mut _ as *mut c_void,
                            &mut a_c as *mut _ as *mut c_void,
                        ];
                        let grid = (stride.div_ceil(256) as u32, 1, 1);
                        ctx.launch_async(k.sub, grid, (256, 1, 1), 0, &mut args)
                    }
                    _ => {
                        let mut args = [
                            &mut pres as *mut _ as *mut c_void,
                            &mut pv as *mut _ as *mut c_void,
                            &mut nt_c as *mut _ as *mut c_void,
                            &mut hc_c as *mut _ as *mut c_void,
                            &mut e_c as *mut _ as *mut c_void,
                            &mut l_c as *mut _ as *mut c_void,
                            &mut a_c as *mut _ as *mut c_void,
                        ];
                        let grid = ((nt * hc) as u32, 1, 1);
                        ctx.launch_async(k.proj, grid, (256, 1, 1), 0, &mut args)
                    }
                }
            };
            if let Err(e) = r {
                return pf_fail(&format!("steer apply launch: {e:#}"));
            }
            let t = match pf_stage_sync(ctx) {
                Ok(t) => t,
                Err(e) => return pf_fail(&format!("steer apply sync: {e:#}")),
            };
            pf_mix_kprof::add_host(pf_mix_kprof::SYNC_OTHER, t);
            true
        })
    })
}


/// task 218: テスト用にデバイス側の作業域を捨てる。
#[cfg(test)]
pub fn pf_steer_reset_for_test() {
    pf_steer_release();
}


// ===========================================================================
// task 172: prefill の full attention / shared expert / PLE をデバイス連鎖で回す
// (`DENNO_PF_FULL_DEV` / `DENNO_PF_SHARED_DEV` / `DENNO_PF_PLE_GPU`)
// ===========================================================================
//
// lab t169b (int-auto 既定・14,018 トークン cold prefill = 69.57 s) の台帳:
// - full 9,765 ms: sync 1,724 回 (gpu 4,328 ms)・cpu 4,072 ms・h2d 11.8 GB / d2h 14.8 GB。
//   中身は qkv 3 行列 (TC GEMM 各 1 同期) + attention (1 同期) + 出力投影 (1 同期) で、
//   トークンごとの split_q_gate / per-head RMSNorm / rope / 出力ゲートが全部 CPU。
// - shared 6,455 ms: sync 7,476 回 (gpu 3,496 ms)・cpu 1,685 ms・h2d 15.5 GB / d2h 10.3 GB。
//   gate / up / down の 3 GEMM をそれぞれ H2D → 同期 → D2H し、silu と積は CPU。
// - other のうち ple 2,676 ms (SSD の pread) と gemm_cpu 1,287 ms (PLE の key/value 射影)。
//
// ここでは**チャンクぶんの活性を GPU に置いたまま** stage を閉じる:
// - full:   1 H2D (入力行) → qkv 3 GEMM → qknorm+rope → (k/v だけ D2H) → attention →
//           出力ゲート → 出力投影 → 1 D2H。同期は 2 回 (k/v をホストの KV に積むため)。
// - shared: 1 H2D → gate/up GEMM → swiglu (elem の既存カーネル) → down GEMM → 1 D2H。
// - ple:    1 H2D (集めた行) → key/value の 2 GEMM → 1 D2H (残りは CPU のまま)。
//
// 数値: GEMM は既存の TC 経路と同じ (BF16 丸め)、qknorm/rope/出力ゲートは CPU と
// 同じ式・同じ順 (`__*_rn` を明示・-fmad=false)。expf だけ CUDA の libm なので
// CPU の glibc expf と 1 ULP ずれうる = prefill の md5 は変わりうる
// (prefill の射影はビット一致契約の外。decode はこの経路を通らない)。
//
// すべて fail closed: PTX が無い / 形が違う / 重みが VRAM に載らない / 起動に失敗、
// のどれでも `false` を返し、呼び出し側は従来経路をそのまま走らせる。

/// 連鎖を使う最小の行数。これ未満 (19 トークンの next turn prefill など) は
/// 従来経路のほうが速いので切り替えない。
pub(crate) const PF2_MIN_ROWS: usize = 64;


/// 連鎖が動かなかった理由を 1 行だけ出す (プロセスで 1 回)。
pub(crate) static PF2_WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);


/// 発火の集計 (テストと lab の切り分け用)。0=full 成立 / 1=full 断念 /
/// 2=shared 成立 / 3=shared 断念 / 4=ple 成立 / 5=ple 断念。
pub static PF2_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];


thread_local! {
    /// 直前の連鎖呼び出しの行数 (ログ用)。
    pub(crate) static PF2_LAST_NT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}


// ---------------------------------------------------------------------------
// カーネル (実行時読み込み。無ければ fail closed)
// ---------------------------------------------------------------------------

pub(crate) const PF2_NAMES: [&[u8]; 2] = [b"dno_pf2_qknorm_rope\0", b"dno_pf2_gate_mul\0"];


thread_local! {
    pub(crate) static PF2_DEV: std::cell::RefCell<Pf2Dev> = const { std::cell::RefCell::new(Pf2Dev::new()) };
}


// ---------------------------------------------------------------------------
// full attention の前半: qkv 射影 + per-head RMSNorm + partial NeoX rope
// ---------------------------------------------------------------------------

/// `DENNO_PF_D2H_DIRECT` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (lab t181b で採用): 単独で 14k の mix が 5,100 → 4,407 ms
/// (cpu 676 → 4 ms)、cold prefill 59.06 → 58.31 s (-1.3%)、md5 SAME。
pub(crate) fn pf_d2h_direct_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// task 181: デバイス → ホストを**呼び出し側のバッファへ直接**戻すか。
///
/// off のときは従来どおり `to_vec` で中間 Vec を作って `copy_from_slice` する。
/// 毎層 5 MB の Vec 確保 (ページフォルト) と memcpy がもう 1 回入るので、
/// 14,018 トークン cold prefill では shared / full / mix の 3 経路で
/// 合計 20 GB 級の無駄なホストコピーになる (**値は 1 ビットも変わらない**)。
pub(crate) fn pf_d2h_direct_on() -> bool {
    #[cfg(test)]
    {
        match PF_D2H_DIRECT_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_d2h_direct_from_value(std::env::var("DENNO_PF_D2H_DIRECT").ok().as_deref())
    })
}


#[cfg(test)]
static PF_D2H_DIRECT_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_d2h_direct_for_test(v: Option<bool>) {
    PF_D2H_DIRECT_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// `DENNO_PF_MIRROR_DELTA` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (lab t181b で採用)。単独では ±0.1% (mirror の H2D は 14k で
/// 657 MB = 38 ms しかない) だが、`pf_full_qkv` と `pf_full_out` の往復
/// (D2H → ホスト積み → H2D) を D2D 1 回に置き換える。値は変わらない。
pub(crate) fn pf_mirror_delta_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// task 181: KV ミラーを「チャンクの差分だけ」にするか (`DENNO_PF_MIRROR_DELTA`)。
///
/// off のときは従来どおり `pf_full_qkv` が k / v を D2H し、`pf_full_out` が
/// ミラーへ H2D し直す。on にすると `pf_full_qkv` が **デバイスの k / v を
/// そのままミラーへ D2D で書き**、`pf_full_out` は H2D をしない。
pub(crate) fn pf_mirror_delta_on() -> bool {
    #[cfg(test)]
    {
        match PF_MIRROR_DELTA_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_mirror_delta_from_value(std::env::var("DENNO_PF_MIRROR_DELTA").ok().as_deref())
    })
}


#[cfg(test)]
static PF_MIRROR_DELTA_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_mirror_delta_for_test(v: Option<bool>) {
    PF_MIRROR_DELTA_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


#[cfg(test)]
pub(crate) static PF_KV_ASYNC_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf_kv_async_for_test(v: Option<bool>) {
    PF_KV_ASYNC_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// task 553: `DENNO_PF_KV_DEFER`。K/V の書き戻しの**待ち**を 1 呼び出し
/// (= 層チャンク) 後ろへずらす。**既定 on** (lab t553a/b で採った)。
pub const PF_KV_DEFER_ENV: &str = "DENNO_PF_KV_DEFER";


/// 未設定のときの既定 (**lab で採ったので on**)。切り戻しは `=0` / `off`。
///
/// 実測 (GPU ラボ、`MODEL=flash-next-UD-IQ3_XXS`、depth8k、PROFILE=1)。
/// `full` 段とその `kv_wait` / `kv_push` (同じ run の b / b2 と対):
///
/// | run / COND | 3,826 full (kv_wait / kv_push) | 14k full (kv_wait / kv_push) |
/// | --- | --- | --- |
/// | t553a b | 105.1-106.9 (34.0-35.6 / 25.2-26.0) | 777.8 (129.1 / 239.1) |
/// | t553a kvd | 56.8-58.5 (**0.0** / 22.3-23.0) | 857.4 (89.3 / 461.0) |
/// | t553a kvdnp | 49.6-50.5 (**0.0** / 12.5-13.3) | 702.4 (85.2 / 212.4) |
/// | t553b b | 104.3-104.4 (34.1-35.3 / 25.5-25.7) | 783.2 (126.4 / 249.2) |
/// | t553b kvd | 57.6-64.0 (**0.0** / 21.5-25.6) | 715.6 (90.0 / 267.4) |
/// | t553b kvdnp | 53.3-70.3 (**0.0** / 15.0-15.6) | 738.1 (89.4 / 226.5) |
///
/// 14k の壁は 4.96 / 5.09 s (base 5.12 / 5.16)、3,826 の壁は 1.89 s の床で不変。
/// decode 10.4 ms/tok で不変、md5 は全 COND x 全 6 文脈 SAME、errors 0。
pub const DEFAULT_PF_KV_DEFER: bool = true;


#[cfg(test)]
pub(crate) static PF_KV_DEFER_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf_kv_defer_for_test(v: Option<bool>) {
    PF_KV_DEFER_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// 待ちを後ろへずらした回数 (分岐が発火した証拠。lab のログに 1 回だけ出す)。
pub(crate) static PF_KV_DEFER_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// `DENNO_PF_KV_PUSH`: 保留中の k/v を **pinned のまま** `AttnCache::push` に
/// 渡す (`ks`/`vs` への写しをしない)。既定 off。
///
/// pinned から `ks`/`vs` へ写す経路は 14k で 304 ms (689 MB、約 2.3 GB/s)、
/// 3,826 で 89 ms (188 MB) のホスト時間を使っていた (lab t369a の `full` の
/// cpu 列: 239 -> 543 ms / 55 -> 144 ms。うち大半は初回タッチのページフォルト)。
/// 直接 push すれば写しは 1 回 (pinned -> キャッシュ) で済む。
/// デバイスから出るバイト列は同じなので値は不変。
pub const PF_KV_PUSH_ENV: &str = "DENNO_PF_KV_PUSH";


#[cfg(test)]
pub(crate) static PF_KV_PUSH_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) fn set_pf_kv_push_for_test(v: Option<bool>) {
    PF_KV_PUSH_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}


/// shared 連鎖の入力の置き場所。
pub(crate) enum SharedIn<'a> {
    Host(&'a [f32]),
    Dev(CUdeviceptr),
}


/// shared 連鎖の出力の置き場所。
pub(crate) enum SharedOut<'a> {
    Host(&'a mut [f32]),
    Dev(CUdeviceptr),
}


/// task 427: mix 以外の呼び出し点 (full 層の qkv/wo、head) から dn6 を撃つ門番。
///
/// カーネル・値は mix の down と同一 (`dno_mix_gemv_q8_0_dn6`。rows は行の割り当て
/// だけを変えるので dn5 とビット一致)。Q8_0 で、形 (`mix_dn6_plan`) と PTX が
/// 揃うときだけ `Some((warps, rows))`。`want_rows == 0` は「撃たない」。
pub(crate) fn dn6_gemv_plan(
    ty: crate::kquant::GgmlType,
    n_in: usize,
    n_out: usize,
    want_rows: usize,
) -> Option<(usize, usize)> {
    if want_rows == 0 || ty != crate::kquant::GgmlType::Q8_0 {
        return None;
    }
    if mix_dn6_kernels().is_none() {
        return None;
    }
    mix_dn6_plan(n_in, n_out, want_rows)
}


/// task 427: [`dn6_gemv_plan`] の計画を 1 本積む (生ポインタの重み)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_dn6_gemv(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    x: CUdeviceptr,
    o: CUdeviceptr,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
    warps: usize,
    rows: usize,
) -> Result<()> {
    launch_mix_dn6(ctx, w, x, o, n_in, n_out, row_bytes, warps, rows)
}


/// task 427: full 層の qkv/wo を dn6 で撃った回数 (1 プロセス通算)。
static DN6_FULL_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 427: head の GEMV を dn6 で撃った回数 (1 プロセス通算)。
static DN6_HEAD_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 427: 発火の証拠 (`[mix-qchain]` の `f6=` / `h6=`)。
pub fn dn6_dec_hits() -> (u64, u64) {
    (
        DN6_FULL_HITS.load(std::sync::atomic::Ordering::Relaxed),
        DN6_HEAD_HITS.load(std::sync::atomic::Ordering::Relaxed),
    )
}


/// task 427: 集計を 0 に戻す。
pub fn dn6_dec_reset() {
    DN6_FULL_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    DN6_HEAD_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// task 427: full 層の 1 本を撃ったことを数える (full_dec から)。
pub(crate) fn note_dn6_full_hit() {
    DN6_FULL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// task 427: head の 1 本を撃ったことを数える。
pub(crate) fn note_dn6_head_hit() {
    DN6_HEAD_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// レーンの埋まり具合 (`np` 対を `lanes` レーンへ割るときの比)。
pub(crate) fn dn8_fill(np: usize, lanes: usize) -> f64 {
    if lanes == 0 {
        return 0.0;
    }
    np as f64 / lanes as f64
}


/// full 層の qkv/wo を dn7 で撃つか (純関数)。`DENNO_FULL_DEC_FAST=dn7` / `d7`。
pub(crate) fn full_dec_dn7_from_value(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("dn7") | Some("d7")
    )
}


/// [`full_dec_dn7_from_value`] をプロセスで 1 回だけ読む。
pub(crate) fn full_dec_dn7_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        full_dec_dn7_from_value(std::env::var("DENNO_FULL_DEC_FAST").ok().as_deref())
    })
}


/// task 504: dn7 の形で撃ってよいか (型 Q8_0 + 起動形 + PTX)。
pub(crate) fn dn7_dec_plan(ty: crate::kquant::GgmlType, k: usize, n: usize) -> Option<(usize, usize)> {
    if ty != crate::kquant::GgmlType::Q8_0 || mix_dn7_kernels().is_none() {
        return None;
    }
    mix_dn7_plan(k, n)
}


/// task 504: GDN の out を dn7 で撃った回数 (`[mix-qchain]` の `o7=`)。
static DN7_OUT_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// task 504: full 層の qkv/wo を dn7 で撃った回数 (`[mix-qchain]` の `f7=`)。
static DN7_FULL_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 504: 発火の証拠 `(full, out)`。
pub fn dn7_dec_hits() -> (u64, u64) {
    (
        DN7_FULL_HITS.load(std::sync::atomic::Ordering::Relaxed),
        DN7_OUT_HITS.load(std::sync::atomic::Ordering::Relaxed),
    )
}


/// task 504: 集計を 0 に戻す。
pub fn dn7_dec_reset() {
    DN7_FULL_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    DN7_OUT_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
    DN7_HEAD_HITS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// task 504: full 層の 1 本を撃ったことを数える。
pub(crate) fn note_dn7_full_hit() {
    DN7_FULL_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// task 504: GDN の out 1 本を撃ったことを数える。
pub(crate) fn note_dn7_out_hit() {
    DN7_OUT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// task 504: head を dn7 で撃った回数 (`[mix-qchain]` の `o7=` に含めず、
/// `h7=` で出す。head は 1 トークン 1 回)。
static DN7_HEAD_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// task 504: head を dn7 で撃った回数。
pub fn dn7_head_hits() -> u64 {
    DN7_HEAD_HITS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 504: head の 1 本を撃ったことを数える。
pub(crate) fn note_dn7_head_hit() {
    DN7_HEAD_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}


/// task 504: dn7 (`kernels_mix_dn7.ptx`) が既定の経路か (gate_audit の台帳用)。
///
/// mix の down (既定 dn7) / GDN の out (既定 dn7) / head (既定 dn7) のどれかが
/// 既定で dn7 を選ぶなら true。PTX が欠けると静かに dn6 / dn8 へ落ちるので載せる。
pub fn dn7_gate_on() -> bool {
    matches!(mix_qk_enabled().0, MixGemv::Dn7) || gdn_out_dn7_on() || head_dn7_on()
}


thread_local! {
    /// (max|diff|, そのときの |ref|, 閾値超えの個数, 比較した要素数)
    pub(crate) static UPQ8_CHECK: std::cell::RefCell<(f64, f64, u64, u64)> =
        const { std::cell::RefCell::new((0.0, 0.0, 0, 0)) };
}


/// kernels_mix_dev.ptx のカーネル一式 (2 本そろって初めて `Some`)。
/// task 266: mix の量子 chain の要素カーネルのスレッド数 (lo_silu は 256 固定)。
pub(crate) const THREADS_Q: usize = 256;


/// task 204: gate+up 融合の可否 (純関数・テスト用)。
///
/// 畳めるのは「ゲートが on」かつ「gate と up の量子型が同じ」かつ「行長が同じで 0 でない」
/// ときだけ。1 行の計算は同じカーネルなので、畳んでも値は変わらない。
pub(crate) fn dec_exp_gu_shape_ok(on: bool, same_slot: bool, rb_g: usize, rb_u: usize) -> bool {
    on && same_slot && rb_g > 0 && rb_g == rb_u
}


/// task 204: 融合起動が実際に撃たれた回数 (テストと lab の診断用。既定 0)。
#[cfg(test)]
pub(crate) static EXP_GU_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);


/// `DENNO_DEC_EXPGU` の解釈 (純関数)。**既定 on** (task 221 の lab t221c で採用)。
/// gate と up を 1 起動に畳むだけで、同じカーネル・同じ行なので値はビット不変
/// (t221b / t221c とも md5 SAME)。切り戻しは `DENNO_DEC_EXPGU=0`。
pub(crate) fn parse_dec_exp_gu(raw: Option<&str>) -> bool {
    !matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// task 204: expert の gate と up を 1 起動に畳むか (`DENNO_DEC_EXPGU`・既定 off)。
pub(crate) fn exp_gu_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_dec_exp_gu(std::env::var("DENNO_DEC_EXPGU").ok().as_deref()))
}


/// `DENNO_DEC_ONEVNT` の解釈 (純関数)。既定 off。
pub(crate) fn parse_dec_onevnt(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}



// ===========================================================================
// P17d: decode chain (隠れ状態デバイス常駐 + 要素演算融合)
// ===========================================================================
// solo decode (n==1) の 1 ステップを
//   埋め込み H2D → [rmsnorm(+residual) → qkv → (CPU: rope/attention) →
//   o_proj → rmsnorm(+residual) → gate/up → swiglu → down] ×層数
//   → final norm(+residual) → lm_head → logits D2H
// のデバイス常駐パイプラインで実行する。
// - 線形カーネル (f_quant / f_mv) と重みレジストリは P17b/c と同一 → ビット一致
// - rmsnorm / swiglu は CPU と同一の演算順序 (expf は glibc e_expf を f64 演算
//   で完全再現した dno_expf) → ビット一致
// - 同期は層あたり 1 回 (qkv の D2H 待ち)。CPU attention の間 GPU は次層の
//   FFN チェーンを先読み実行する
// - どのステップで失敗しても GPU_BROKEN → 呼び出し側が従来 CPU 経路で forward
//   をやり直す (k/v 書き込みは冪等・advance は最後に 1 回なので安全)

/// デバイス常駐スロット (ChainScratch.bufs のインデックス)
pub const CH_X: usize = 0; // residual stream [m*e]

pub const CH_H: usize = 1; // rmsnorm 出力 [m*e]

pub const CH_QKV: usize = 2; // qkv 連結出力 [m*(qn+2*kvn)] (q|k|v の列ブロック)

pub const CH_ATT: usize = 3; // attention 出力 (o_proj 入力) [m*(heads*dh)]

pub const CH_O: usize = 4; // o_proj 出力 [m*(heads*dh)]

pub const CH_GA: usize = 5; // gate/up 出力 [m*2ff] (2 回 GEMM = gate 前半/up 後半・1 回 GEMM = 行内 [gate|up])

pub const CH_ACT: usize = 6; // swiglu 出力 [m*ff]

pub const CH_DOWN: usize = 7; // down 出力 [m*e]

pub const CH_LOGITS: usize = 8; // lm_head 出力 [m*vocab]

// C4 (qwen35): QK-Norm + partial interleaved rope 済み q [heads*dh]。
// llama chain と異なり qkv 出力が head ごと [q|gate] 融合のため、attention へ渡す
// q は別スロットに分離して書く (chain_q35_qknorm_rope)。
pub const CH_Q35_Q: usize = 9;

/// s70: prefill FFN チェーンの入力 (CH_H は attention/decode が使うため分ける)
pub const CH_PFH: usize = 10;

/// s165 (P2): prefill の残差ストリーム (h) を層の間ずっと保持するスロット。
/// `dno_add_rmsnorm` の x そのもので、残差加算は**次の rmsnorm の proj 引数**に
/// 融合される (PTX に単独の add カーネルは無い)。層をまたいで内容が変わるのは
/// rmsnorm の add と、prefill 開始時の H2D / フォールバック時の D2H だけ。
pub const CH_PF_RES: usize = 11;

pub(crate) const CH_SLOTS: usize = 12;

/// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
pub(crate) const CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

/// CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
pub(crate) const CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;


/// s169 (P10): デバイスの SM 数 (GEMV の launch 形状を n から選ぶため)。
/// CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT。取得できなければ 0 のまま =
/// 形状選択は従来どおりの固定 C に落ちる (数値・発行順は不変)。
pub(crate) const CU_DEV_ATTR_MULTIPROCESSOR_COUNT: c_int = 16;

pub(crate) static GEMV_SM_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);


thread_local! {
    pub(crate) static CHAIN: std::cell::RefCell<ChainScratch> = std::cell::RefCell::new(ChainScratch {
        bufs: (0..CH_SLOTS).map(|_| None).collect(),
        pin_x: None,
        pin_dn: None,
        aq: None,
        attn_q: None,
        attn_sc: None,
        attn_pf_q: None,
        rope_tab: None,
        rope_bias: None,
        pin_rope: None,
        pin_pf: None,
        kv_stage: None,
        qkv_x: None,
        qkv_aq: None,
        qkv_asc: None,
        qkv_pf: None,
        qkv_q: None,
        rope_tab_pf: None,
        pin_qkv: None,
        br_q: None,
        br_k: None,
        br_tab: None,
        br_pin_qk: None,
        br_pin_tab: None,
        br_tab_key: Vec::new(),
        br_tab_dh: 0,
        br_tab_theta: 0.0,
        br_tab_valid: false,
        pins: Vec::new(),
        q35_tab: None,
        q35_pin_tab: None,
        q35_w_cache: std::collections::HashMap::new(),

        rms_w: std::collections::HashMap::new(),
        h2_stream: None,
        h2_ev_ready: None,
        h2_ev_done: None,
        pin_h2: None,
        h2_pending: false,
        pipe_ev: Vec::new(),
        pin_pf_qkv: None,
        qk_stream: None,
        qk_ev_q: None,

    });
}


/// Result を Option に変換しつつ GPU_BROKEN をマークする
pub(crate) fn ok<T>(r: Result<T, anyhow::Error>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            chain_err(e);
            None
        }
    }
}


/// B1: prefill デバイス常駐チェーン (`LlamaQwen2::forward_prefill_gpu`) を発動して
/// よいかの事前ゲート (カーネル制約の mirror)。チェーンは all-or-nothing (途中失敗
/// → forward 全体を従来経路でやり直し) なので、構造的に確実に失敗する形状は
/// 開始前に弾く (GPU_BROKEN ラッチの無駄な発火防止)。
/// - `e` (hidden): dno_add_rmsnorm が動的 shmem = e*4 を使う (≤48KB)
/// - `dh`/`t_total`: dno_attention_prefill の制約 (chain_attention_prefill_dev と同一)
pub fn prefill_chain_gpu_supported(e: usize, dh: usize, t_total: usize) -> bool {
    if e == 0 || e * 4 > 48 * 1024 {
        return false;
    }
    if dh < ATTN_PF_BQ || dh > 1024 || t_total == 0 || t_total > 65535 {
        return false;
    }
    // s48 (C1-a): 48KB 固定 → opt-in 済み上限 (既定は 48KB・llama dh=128 は不変)
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return false;
    }
    true
}


/// s157 (Task B20 段階 1): `DENNO_GPU_Q4_CHAIN` の解釈 (純関数・テスト可能)。
/// **`"0"` だけが off** — 未設定・空・その他の値は既定の on (従来の per-op 経路へ
/// 戻すのは明示的な 0 だけ。タイポで静かに経路が変わるのを避ける)。
pub fn parse_q4_chain(raw: Option<&str>) -> bool {
    raw != Some("0")
}


/// [`chain_mlp_prefill_q4`] の入力次元 (k)。Q4 以外は None。
pub(crate) fn gate_dims_in(w: &crate::quant::Linear) -> Option<usize> {
    match w {
        crate::quant::Linear::Q4(q) => Some(q.k),
        _ => None,
    }
}


/// [`chain_mlp_prefill_q4_fused`] の出力次元 (e = down.n)。Q4 以外は None。
pub(crate) fn down_dims_out(w: &crate::quant::Linear) -> Option<usize> {
    match w {
        crate::quant::Linear::Q4(q) => Some(q.n),
        _ => None,
    }
}


/// s173 (P14): FFN を **1 回の gate|up GEMM** で通せるか (GPU を触らない事前検査)。
///
/// 条件: gate_up / gate / up / down がすべて Q4・形の整合 (`gate_up` が
/// [gate 全行 | up 全行]・k = e・n = 2*ff)・**gate/up が連結側の行範囲として
/// 登録済み** ([`q4_register_row_alias`])。
///
/// 最後の条件は VRAM/H2D を増やさないためのもの: 未登録のまま連結重みを使うと
/// 2*ff*e を別途転送することになる (登録済みなら `weights_for_q4` は連結側の
/// 既存デバイス領域を指すビューを返す)。満たさないときは呼び出し側は従来の
/// 2 回呼び出し経路のまま = fail closed。
pub fn pf_ffn_fused_ready(
    gate_up: &crate::quant::Linear,
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
) -> bool {
    if !pf_ffn_fused_enabled() {
        return false;
    }
    let (Some(c), Some(g), Some(u), Some(d)) =
        (q4_of(gate_up), q4_of(gate), q4_of(up), q4_of(down))
    else {
        return false;
    };
    if !ffn_fused_shape_ok(c, g, u, d) {
        return false;
    }
    q4_fused_ready(c.uid, &[g, u])
}


/// s173 (P14): FFN の gate|up を 1 回の GEMM に畳むか (既定 ON)。
///
/// `DENNO_PF_FFN_FUSED=0` で従来の 2 回呼び出しへ戻す — A/B 用のスイッチで、
/// **数値は経路によらず同一** (カーネル・累積順・タイル幅は不変)。
fn pf_ffn_fused_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_pf_ffn_fused(std::env::var("DENNO_PF_FFN_FUSED").ok().as_deref()))
}


/// s173 (P14): `DENNO_PF_FFN_FUSED` の解釈 (純関数・テスト可能)。
/// `parse_q4_chain` と同じ規約で、**`"0"` だけが off** (未設定・空・その他は on)。
pub(crate) fn parse_pf_ffn_fused(raw: Option<&str>) -> bool {
    raw != Some("0")
}


/// s173 (P14): [`pf_ffn_fused_ready`] の**形状だけ**の契約 (純粋関数・GPU と
/// エイリアス表に触らない)。
///
/// `gate_up` = [gate 全行 | up 全行]・k = e・n = 2*ff・`down` = [ff, e]。
/// ここが崩れると連結重みの行範囲が別の重みを指す (数値が変わる) ので、
/// 満たさない入力はすべて false (= 従来の 2 回呼び出しへフォールバック)。
pub(crate) fn ffn_fused_shape_ok(
    c: &crate::quant::Q4Linear,
    g: &crate::quant::Q4Linear,
    u: &crate::quant::Q4Linear,
    d: &crate::quant::Q4Linear,
) -> bool {
    if c.k != g.k || c.k != u.k || c.n != g.n + u.n {
        return false;
    }
    // ff = g.n = u.n = down.k、e = c.k = down.n、連結幅 = 2*ff
    d.k == g.n && d.n == c.k && c.n == 2 * d.k
}


/// GPU attention が有効か (既定 ON・`DENNO_GPU_ATTN=0` で無効 — P17h から
/// forward_decode_gpu (solo chain) からも参照するため pub)
pub fn gpu_attention_enabled() -> bool {
    GPU_ATTN_INIT.get_or_init(|| {
        if std::env::var("DENNO_GPU_ATTN").ok().as_deref() == Some("0") {
            GPU_ATTN.store(false, std::sync::atomic::Ordering::Relaxed);
        }
    });
    GPU_ATTN.load(std::sync::atomic::Ordering::Relaxed)
}


/// テスト用: GPU attention の ON/OFF を直接差し替える (戻り値は設定前の値)
#[doc(hidden)]
pub fn set_gpu_attention_enabled(on: bool) -> bool {
    GPU_ATTN_INIT.get_or_init(|| {});
    GPU_ATTN.swap(on, std::sync::atomic::Ordering::Relaxed)
}


/// GPU KV ミラーの上限 (MiB) の既定値。P58-1: 8192 → 2048。
///
/// 実測 (27B Q4_0・本番 serve): 既定 8192 MiB のままだとミラーが約 1 GB を常駐し、
/// 18,670 token のリクエスト後に VRAM が 30,932 MiB まで達して `cuMemAlloc` 失敗
/// (要求 779 MiB / 空き 315 MiB) → CPU フォールバック → サーバ固着を 3 回起こした。
/// `DENNO_GPU_KV_MB=1024` に下げるだけで 29,918 MiB に収まった (1,014 MB 減)。
/// ミラーは KV の複製 (ホストが正) なので、上限を下げて evict が増えても出力は
/// 変わらない — evict されたミラーは次の使用時にホストから再シードされる。
pub(crate) const DEFAULT_GPU_KV_LIMIT_MB: usize = 2048;


/// `DENNO_GPU_KV_MB` の解釈 (純関数。env を読まないのでテストから直接叩ける)。
/// 未設定・不正値は [`DEFAULT_GPU_KV_LIMIT_MB`]。
pub(crate) fn gpu_kv_limit_bytes_from(v: Option<usize>) -> usize {
    v.unwrap_or(DEFAULT_GPU_KV_LIMIT_MB) * 1024 * 1024
}


pub(crate) fn gpu_kv_limit_bytes() -> usize {
    gpu_kv_limit_bytes_from(
        std::env::var("DENNO_GPU_KV_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok()),
    )
}


/// P58-1: ミラー 1 本のエビクト判定用メタデータ (ホスト側の値だけ)。
/// 実バッファを持たないので、LRU の判定と上限の効き方は GPU 無しでテストできる。
#[derive(Debug, Clone, Copy)]
pub(crate) struct MirrorMeta {
    key: u64,
    bytes: usize,
    last: std::time::Instant,
    /// GDN 遅延フラッシュ pending (エビクト禁止 — 未フラッシュの状態が唯一のコピー)
    pending: bool,
}


/// 上限超過分を LRU で落とし、落とした key を古い順に返す (純関数)。
///
/// 契約:
/// - 自分 (`keep_key`) は落とさない (これから使うミラー)。
/// - `pending` のミラーは落とさない (ホストより新しい GDN 状態の唯一のコピー)。
/// - 落とせる相手が無ければ止まる (上限超過のまま続くが、データは失わない)。
/// - ミラーは KV の複製なので、ここで落としても出力は変わらない (再シード)。
pub(crate) fn lru_evict_keys(entries: &mut Vec<MirrorMeta>, limit: usize, keep_key: u64) -> Vec<u64> {
    let mut evicted = Vec::new();
    let mut total: usize = entries.iter().map(|e| e.bytes).sum();
    while total > limit && entries.len() > 1 {
        let vic = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.key != keep_key && !e.pending)
            .min_by_key(|(_, e)| e.last)
            .map(|(i, _)| i);
        let Some(vic) = vic else {
            break;
        };
        total -= entries[vic].bytes;
        evicted.push(entries.swap_remove(vic).key);
    }
    evicted
}


/// P58-1: ミラーの実使用量 (エントリ数, 合計バイト数, 上限バイト数)。
/// 合計は [`KvMirror::bytes`] (K/V の実バッファ + GDN 状態) の和。
/// 診断用なのでロックが取れない場合は None (呼び出し側を止めない)。
pub fn gpu_kv_mirror_stats() -> Option<(usize, usize, usize)> {
    let mirrors = KV_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let ms = mirrors.lock().ok()?;
    let used = ms.iter().map(|(_, m, _)| m.bytes()).sum();
    Some((ms.len(), used, gpu_kv_limit_bytes()))
}


/// P58-1: ミラー使用量を 1 行ログる (起動時・各リクエスト完了時)。
/// MB は切り上げ (使用量が 0 でないのに 0 MB と出るのを避ける)。
pub fn log_gpu_kv_mirror_usage() {
    if let Some((entries, used, limit)) = gpu_kv_mirror_stats() {
        const MB: usize = 1024 * 1024;
        tracing::info!(
            "gpu kv mirror: {} entries, {} MB / limit {} MB",
            entries,
            used.div_ceil(MB),
            limit / MB
        );
    }
}


/// P17g/P17h 共通: KV ミラーの取得/生成 + cap 成長 + LRU エビクト。
/// `f` はミラーロック保持中に呼ばれ (行同期 + ポインタ抽出を行う)、
/// `have` はミラーが持っている行数 (rollback で t_total を超えていれば 0)。
/// 呼び出し側が m.len[li] を確定させる (t_total を正として使う — cache.advance
/// は forward の最後なので host len はまだ進んでいない)。
pub(crate) fn with_mirror<T>(
    key: u64,
    li: usize,
    n_layers: usize,
    t_total: usize,
    kv_dim: usize,
    epoch: u64,
    f: impl FnOnce(&mut KvMirror, usize) -> Option<T>,
) -> Option<T> {
    with_mirror_ext(key, li, n_layers, t_total, kv_dim, epoch, true, f)
}


/// C4: with_mirror の拡張版。`alloc_kv = false` で KV cap 成長をスキップする
/// (qwen35 の GDN 層は KV を使わないため、GDN 層の呼び出しで全層分の KV ミラー
/// バッファを確保しない — 27B では数 GB 規模の無駄を避ける)。GDN 層は
/// m.layers[li] / m.len[li] を触らず gdn/gdn_pos のみを使う。
pub(crate) fn with_mirror_ext<T>(
    key: u64,
    li: usize,
    n_layers: usize,
    t_total: usize,
    kv_dim: usize,
    epoch: u64,
    alloc_kv: bool,
    f: impl FnOnce(&mut KvMirror, usize) -> Option<T>,
) -> Option<T> {
    let mirrors = KV_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut ms = ok(mirrors.lock().map_err(|e| anyhow::anyhow!("{e}")))?;
    let now = std::time::Instant::now();
    match ms.iter().position(|(k, _, _)| *k == key) {
        Some(i) => {
            ms[i].2 = now;
        }
        None => {
            ms.push((
                key,
                KvMirror {
                    layers: (0..n_layers).map(|_| (None, None)).collect(),
                    cap: 0,
                    len: vec![0; n_layers],
                    epoch,
                    gdn: (0..n_layers).map(|_| None).collect(),
                    gdn_pos: vec![None; n_layers],
                    gdn_pending: None,
                    gdn_stale: vec![false; n_layers],
                },
                now,
            ));
        }
    };
    // エビクト: 上限超過で LRU (自分は除外)。s50: GDN 遅延フラッシュ pending 中の
    // ミラーは除外 — エビクトすると未フラッシュの GDN 状態 (ホストより新しい唯一の
    // コピー) が失われ回復不能になる。pending は次の prefill / CPU フォールバックの
    // 冒頭でフラッシュして解除されるため、上限超過は高々 1 リクエスト分の遅延。
    // P58-1: 判定は純関数 `lru_evict_keys` に切り出した (ホスト側のメタデータだけで
    // LRU と上限の効き方をテストできる)。ここでは落とす key を ms から除去するだけ。
    let mut meta: Vec<MirrorMeta> = ms
        .iter()
        .map(|(k, m, t)| MirrorMeta {
            key: *k,
            bytes: m.bytes(),
            last: *t,
            pending: m.gdn_pending.is_some(),
        })
        .collect();
    let evicted = lru_evict_keys(&mut meta, gpu_kv_limit_bytes(), key);
    if !evicted.is_empty() {
        ms.retain(|(k, _, _)| !evicted.contains(k));
    }
    // 除去で添字が動くので引き直す (自分はエビクト対象外なので必ず見つかる)。
    let idx = ms.iter().position(|(k, _, _)| *k == key)?;
    let m = &mut ms[idx].1;
    if li >= m.layers.len() {
        return None; // n_layers 不一致 (起こらないはず)
    }
    // s41: コンテンツ世代の検証。同一 kv_id (スロット) でホスト cache が reset /
    // restore され別会話の KV で再 prefill された場合、ミラーの行 [0..len) と
    // GDN 状態は前会話のものなので全層無効化する (次回の層呼び出しで全行再 seed)。
    // 旧実装は「len > t_total なら rollback」という収縮検出のみで、同一スロットに
    // 短い会話の後に長い別会話が来ると 0 < len ≤ t_total の古い行を正規行として
    // 使ってしまい (fresh 初回は len=0 で必ず再 seed・同長以上の次会話は rollback
    // ガードで再 seed)、短→長の切替直後の 1 リクエストだけ出力が変わる 2 値分岐
    // を生んでいた。
    if m.epoch != epoch {
        m.len.iter_mut().for_each(|l| *l = 0);
        m.gdn_pos.iter_mut().for_each(|p| *p = None);
        // s141b: 行/位置を捨てるのと同じく、ホスト遅延フラグも落とす (次の seed は
        // 位置不一致で必ず走り、そこで改めて判定し直す)。
        m.gdn_stale.iter_mut().for_each(|s| *s = false);
        // s50: 世代が変わった = ホストが reset/restore で正に戻っている。遅延
        // pending は前会話のデータを指すため破棄 (chain_q35_gdn_flush_pending が
        // no-op になり、ホスト状態をそのまま使う)。
        m.gdn_pending = None;
        m.epoch = epoch;
    }
    // cap 成長 (全レイヤ realloc + len リセット → 各レイヤ次回使用時に再シード)。
    // alloc_kv = false (GDN 専用呼び出し) ではスキップ — cap/len は使わない。
    if alloc_kv && m.cap < t_total {
        let newcap = kv_mirror_cap(t_total, m.cap);
        // s128 (P4-7): ここで全層分を確保しない。以前は 1 層の prefill で
        // n_layers 本ずつ確保しており、27B では KV を書かない GDN 層 48 本 x K/V
        // まで常駐していた (10,880 token で 8.1 GB)。
        // cap が伸びた時点で既存バッファの行内容は捨てる (Scratch::ensure は
        // 新しいバッファを確保する) ため len は全層 0 に戻し、各層のバッファは
        // その層を使うときに下の ensure で確保する。
        m.cap = newcap;
        m.len.iter_mut().for_each(|l| *l = 0);
    }
    if alloc_kv {
        // 使う層の分だけ確保する (grow-only・容量は cap 行で固定)
        let bytes = m.cap * kv_dim * 4;
        let before = m.layers[li].0.as_ref().map(|b| b.bytes()).unwrap_or(0);
        ok(Scratch::ensure_tagged(
            &mut m.layers[li].0,
            bytes,
            crate::mem_audit::TAG_KV_MIRROR,
        ))?;
        ok(Scratch::ensure_tagged(
            &mut m.layers[li].1,
            bytes,
            crate::mem_audit::TAG_KV_MIRROR,
        ))?;
        if before < bytes {
            // バッファを伸ばした = 行内容は失われた (ensure は新規確保) → 全再シード
            m.len[li] = 0;
        }
    }
    let have = if m.len[li] > t_total { 0 } else { m.len[li] };
    f(m, have)
}


/// 正整数 env の解釈 (未設定・不正値は既定)。
pub(crate) fn parse_usize_default(v: Option<&str>, default: usize) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(default)
}


#[cfg(test)]
pub(crate) static FD_TILE_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);


/// タイル版のカーネル 1 本。読み込み失敗は `None` を焼き付ける (v1 へ落ちる)。
pub(crate) struct FdtKernels {
    /// モジュールはプロセス終了まで保持する (unload しない)。
    _module: CUmodule,
    f_t: CUfunction,
}


#[cfg(test)]
pub(crate) static FD_ILP_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);


#[cfg(test)]
pub(crate) static FD_KHX_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);


/// 第 3 版のカーネル 1 本。読み込み失敗は `None` を焼き付ける (第 2 版へ落ちる)。
pub(crate) struct Fdt2Kernels {
    /// モジュールはプロセス終了まで保持する (unload しない)。
    _module: CUmodule,
    f_t2: CUfunction,
}


// ---- task 460: combine と出力ゲートの融合 (`kernels_attn_dec4.cu`) --------
//
// 動機 (task 456 の `[dec-kprof-avg]`、14k の 1 層あたり):
//   `dno_attn_flashdec_combine` 16.2 us + `dno_dec_gate_mul` 3.8 us が別々に
//   走っていた。間に同期もデータ依存も無い (combine の出力を gate が読むだけ)
//   ので 1 起動にすると、3.8 us の本体と起動 1 本ぶんの隙間が消える。
//   値はビット一致 (縮約順もゲートの演算列も変えない)。
//   lab t460b: full 段 14k 後 1.76 -> 1.63 ms/token (-7.6%)、md5 SAME。

/// 融合カーネルの引数の個数 (`.cu` と一致させる。kernel_invariants が見る)。
pub(crate) const FCBG_NARGS: usize = 6;


#[cfg(test)]
pub(crate) static FD_CBG_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);


/// 融合版のカーネル 1 本。読み込み失敗は `None` を焼き付ける (2 起動へ落ちる)。
pub(crate) struct FcbgKernels {
    /// モジュールはプロセス終了まで保持する (unload しない)。
    _module: CUmodule,
    f_cbg: CUfunction,
}


/// `q4_attn_decode` 内の Result → Option (失敗しても GPU 全体を止めない)。
pub(crate) fn q4ok<T>(r: Result<T, anyhow::Error>) -> Option<T> {
    match r {
        Ok(v) => Some(v),
        Err(e) => {
            q4_attn_fail(e);
            None
        }
    }
}


/// ホスト側だけのミラー台帳 (デバイスに触らない。テストから直接叩ける)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Q4MirrorState {
    /// 層あたりの確保行数 (0 = 未確保)
    pub cap: usize,
    /// 層ごとにミラーが正しく持っている行数
    pub len: Vec<usize>,
    /// その行数がどの内容世代のものか ([`crate::model::qwen4exp_attn::AttnCache::gen`])
    pub gen: Vec<u64>,
}


/// task 210: env の値からゲートを決める (純関数。真偽表テストが規約を固定する)。
///
/// 未設定・空・知らない値は on (既定 on)。
pub(crate) fn mirror_forget_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false") | Some("no"))
}


// ===========================================================================
// C4: qwen35 デバイス常駐 decode チェーン (GDN 状態 GPU ミラー)
// ===========================================================================
//
// 雛形は llama P17b-g/P17h solo chain (forward_decode_gpu)。decode 1 ステップを
// 「埋め込み 1 回 H2D → 全層 GPU 内連結 → 最終 logits 1 回 sync+D2H」にする。
// qwen35 固有の追加要素:
// - full attention 層: qkv 出力が head ごと [q|gate] 融合 → QK-Norm + partial
//   interleaved rope (dno_qknorm_rope_pi) + sigmoid 出力ゲート (dno_attn_gate_mul)
// - GDN 層: beta/alpha 制御投影が Fp32 (int8 では精度崩壊の実測あり) →
//   dno_matvec_f32 (tensor::matvec_bt とビット一致) で CH_QKV テールへ出力
// - GDN 状態 (conv/SSM) の GPU ミラー (KvMirror 拡張・ping-pong 2 面)。
//   ホスト状態はステップ末尾 (logits download の sync 後) に staged D2H から
//   復元して正であり続ける (CPU 再実行 / snapshot 契約の維持)
//
// exactness 契約 (llama chain と同じ枠組み):
// - ビット一致: rmsnorm / int8 GEMV / swiglu / expf / rope / QK-Norm / sigmoid
//   ゲート / fp32 matvec (beta/alpha)
// - 緩和 (ユーザー承認・P17g と同一): attention のスコア/softmax 累積順
// - bit 一致不要契約 (決定論のみ): GDN スキャン (既存 dno_gdn_scan と同一カーネル)
//
// 発動条件を満たせない (kernels_elem.ptx に新カーネルが無い / 非対応形状 / int8
// 混在) 場合は全て None を返し、呼び出し側が従来の per-op 経路で forward を
// やり直す。GPU_BROKEN は立ち**ない** (llama 経路への影響ゼロ)。

/// C4: qwen35 chain 用新カーネルの関数ハンドル。lazy fetch —
/// kernels_elem.ptx に未収録 (PTX 未再生成) の場合は None をキャッシュして
/// chain を恒久不発動にする。GPU_BROKEN にはしない (既存 llama 経路を壊さない。
/// PTX 再生成は人間対応・再生成後の次回プロセス起動から発動)。
/// s169: dno_matvec_f32 の並列リダクション版を使うか (既定 OFF)。
/// 丸め順が逐次版と変わるため、有効化すると出力は bit 一致しなくなる。
pub(crate) fn fast_reduce_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DENNO_FAST_REDUCE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}


pub(crate) fn gguf_gemm_dev_launch(
    ctx: &CudaCtx,
    ty: crate::kquant::GgmlType,
    wptr: CUdeviceptr,
    xptr: CUdeviceptr,
    optr: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
    // task 407: true なら v2 (tc2) を飛ばして br169 (tc) を先に撃つ
    // (`DENNO_PF_GDN_TC1`)。形が合わなければ従来どおり下へ落ちる。
    prefer_tc1: bool,
) -> Result<bool> {
    use crate::kquant::GgmlType as T;
    const ROWS: usize = 16; // kernels_gemm.cu の GEMM_ROWS
    const NTG_MAX: usize = 8; // kernels_gemm.cu の GEMM_NTG_MAX
    // task 176: v2 カーネル (Q8_0 / Q4_K / Q5_K・出力行 128・2 段プリフェッチ)。
    // task 407: `prefer_tc1` のときは飛ばす (br169 の方が速い形がある)。
    if !prefer_tc1 && pf_tc2_gemm_on() && pf_tc2_eligible(ty, nt, n_in, n_out, pf_tc_min_m()) {
        if let Some(rb) = gguf_row_bytes(ty, n_in) {
            if let Some(tc) = tc2_gemm_kernels() {
                tc2_gemm_launch(ctx, tc, ty, wptr, xptr, optr, nt, n_in, n_out, rb)?;
                return Ok(true);
            }
        }
    }
    // task 181: N が小さい形 (ssm out_proj 2560 / shared 640) は BN=64 版。
    if pf_tc_smalln_on() && pf_tc_smalln_eligible(ty, nt, n_in, n_out, pf_tc_min_m()) {
        if let Some(rb) = gguf_row_bytes(ty, n_in) {
            if let Some(sn) = tc_smalln_kernels() {
                sn_gemm_launch(ctx, sn, ty, wptr, xptr, optr, nt, n_in, n_out, rb)?;
                return Ok(true);
            }
        }
    }
    // task 169: Q8_0 は BF16 テンソルコア GEMM (ゲート on・形が合うとき)。
    // 従来カーネルの 128 トークン上限を外し、2048 トークンまで 1 起動で畳む。
    if matches!(ty, T::Q8_0)
        && pf_tc_gemm_on()
        && pf_tc_gemm_eligible(nt, n_in, n_out, pf_tc_min_m())
    {
        if let Some(rb) = gguf_row_bytes(ty, n_in) {
            if let Some(tc) = tc_gemm_kernels() {
                tc_gemm_launch(ctx, tc, wptr, xptr, optr, nt, n_in, n_out, rb)?;
                return Ok(true);
            }
        }
    }
    let Some(fs) = ctx.f_gemm_q else {
        return Ok(false);
    };
    if nt == 0 || nt > ROWS * NTG_MAX || n_in == 0 || n_out == 0 {
        return Ok(false);
    }
    let (f, blk) = match ty {
        T::Q8_0 => (fs[0], 32usize),
        T::Q5_1 => (fs[1], 32usize),
        T::Q4K => (fs[2], 256usize),
        T::Q5K => (fs[3], 256usize),
        _ => return Ok(false),
    };
    if n_in % blk != 0 {
        return Ok(false);
    }
    let Some(rb) = gguf_row_bytes(ty, n_in) else {
        return Ok(false);
    };
    let shmem = ROWS * blk * 4;
    if shmem > 46_000 {
        return Ok(false);
    }
    let grid = ((n_out + ROWS - 1) / ROWS) as u32;
    unsafe {
        let (mut pw, mut px, mut po) = (wptr, xptr, optr);
        let (mut nt_c, mut ni_c, mut no_c, mut rb_c) =
            (nt as c_int, n_in as c_int, n_out as c_int, rb as c_int);
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut nt_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        ctx.launch_async(f, (grid, 1, 1), (ROWS as u32, 16, 1), shmem as u32, &mut args)?;
    }
    Ok(true)
}


// ---------------------------------------------------------------------------
// task 225: prefill の ssm beta/alpha 射影 (F32、n <= 256) を専用 GEMM で回す
// (`DENNO_PF_BA_FAST`、task 231 から既定 on)
// ---------------------------------------------------------------------------
//
// lab t222c (14,018 トークン、PROFILE=1) の `sub ssm beta_pf` は 612.7 ms。
// 中身はホストの `gemv_batch_split` (F32 重み 2,560 x 96 を 16 スレッドで回す)
// で実効 0.4 TFLOP/s。同じ層の qkv GEMM は 93.6 TFLOP/s。
// 既存のデバイス版 (`dno_gemm_f32_mt` を `DENNO_PF_SSM_GEMM_GPU` で使う) は
// lab t183a で 712 ms と**ホストより遅い** (1 ブロック 16 行 x 32 トークン、
// n = 48 では 3 x 16 = 48 ブロック、2 アキュムレータ/スレッドで latency 律速)。
//
// `dno_gemm_f32_ba` (kernels_gemm_ba225.cu) は 48 行 x 64 トークンのタイルで
// 12 アキュムレータ/スレッド。入力も出力もデバイスにあるので転送は 0。
// 数値は F32 の `__fmaf_rn` を k 昇順に積む (ホストの mul+add 逐次和とは丸めだけ
// が違う。決定論)。

/// `dno_gemm_f32_ba` の 1 ブロック行数 (カーネルの BA225_ROWS)。
pub(crate) const BA225_ROWS: usize = 48;

/// `dno_gemm_f32_ba` の 1 ブロックトークン数 (カーネルの BA225_TOKS)。
pub(crate) const BA225_TOKS: usize = 64;


pub(crate) struct BaGemmFns {
    #[allow(dead_code)]
    module: CUmodule,
    /// **フィールド名は他と共有しないこと**: kernel_invariants の起動サイト走査は
    /// `fns.<field>` をカーネル名へ引くので、同名のフィールドが別カーネルを指すと
    /// 表が壊れる (task 225 で `gemm` が mix_pf と衝突した)。
    ba_gemm: CUfunction,
}


/// `DENNO_PF_BA_FAST` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn pf_ba_fast_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        // task 231: lab t228a の ba COND (base + このゲート) が
        // 3826 cold prefill 7.13 s (同じ既定の base/base0/base2 は 7.29〜7.50)、
        // 14018 cold 26.05 s (同じ既定は 26.30〜29.15) で、decode・errors・text は
        // 無影響だった。beta/alpha の射影をデバイスで作るだけ (md5 DIFF は f32 fma の
        // 丸め) なので既定 on。切り戻しは env = 0。
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


#[cfg(test)]
static BA225_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf_ba_fast_for_test(v: Option<bool>) {
    BA225_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// beta/alpha を専用 GEMM で回すか (task 231 から既定 on。t228a の ba COND)。
pub(crate) fn pf_ba_fast_on() -> bool {
    #[cfg(test)]
    {
        match BA225_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_ba_fast_from_value(std::env::var("DENNO_PF_BA_FAST").ok().as_deref()))
}


/// beta/alpha を**デバイスで**作るか。
///
/// task 225: `DENNO_PF_BA_FAST` を足したときは**単独で on にしなかった**。
/// lab t225a では `ba_dev` が作れず (`ssm_beta` が BF16 で F32 前提の入口が
/// 閉じた)、`gdn_pf_chain` が連鎖ごと諦めて ssm が 2.98 → 27.8 s (+79%) に
/// なったため。task 228 が入口を型非依存に直し (`pf_ba_weight_dev` が非 F32 を
/// ホストで 1 回 f32 へ展開)、判定を `pf_ba_w_check` に一本化したので、この
/// 落とし穴は無い (t228a の ba COND: errors 0、text sane、prefill だけ改善)。
pub(crate) fn pf_ba_dev_on() -> bool {
    pf_ssm_gemm_gpu_on() || pf_ba_fast_on()
}


/// 形のゲート (純関数)。n は 1 本あたりの出力行 (実モデルは 48)。
pub(crate) fn pf_ba_fast_ok(nt: usize, k: usize, n: usize) -> bool {
    // n は router の logits (512) まで受ける (同じカーネルを n 方向の
    // ブロック分割で回すだけ。累積順は dno_gemm_f32_mt と同じ k 昇順)。
    nt >= 1 && k >= 1 && n >= 1 && n <= 512
}


/// `DENNO_PF_ROUTER_DEV` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn pf_router_dev_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        // task 228: task 225 の lab t225c で router -458 ms (14k) / decode・text 無影響
        // の実測が出たので既定 on (オペレーター判断: 実利があり 1% 超の悪化が無ければ
        // 部分採用)。切り戻しは `DENNO_PF_ROUTER_DEV=0`。
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


#[cfg(test)]
pub(crate) fn set_pf_router_dev_for_test(v: Option<bool>) {
    ROUTER_DEV_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// router の softmax + top-k をデバイスで回すか (task 228 から既定 on)。
pub(crate) fn pf_router_dev_on() -> bool {
    #[cfg(test)]
    {
        match ROUTER_DEV_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_router_dev_from_value(std::env::var("DENNO_PF_ROUTER_DEV").ok().as_deref()))
}


/// カーネルの前提 (k <= 32 / ne <= 1024)。純関数。
pub(crate) fn pf_router_dev_ok(nt: usize, ne: usize, k: usize) -> bool {
    nt >= 1 && (1..=32).contains(&k) && (1..=1024).contains(&ne)
}


/// task 225: logits をデバイスに残したまま softmax → top-k → 再正規化まで回す。
///
/// `dev_x` が `Some` なら入力は既にデバイスにある (`DENNO_PF_MIXDEVIN` の
/// `mixed`)、`None` ならホストの `xs` を上げる。`sels` / `wts` は `nt * k`、
/// `rowsum` は `nt` 要素。`false` = デバイスでやらなかった (呼び出し側は
/// 従来のホスト経路へ)。
#[allow(clippy::too_many_arguments)]
pub fn pf_router_dev_topk(
    uid: u64,
    bytes: &[u8],
    k_in: usize,
    n: usize,
    dev_x: Option<CUdeviceptr>,
    xs: Option<&[f32]>,
    nt: usize,
    k: usize,
    sels: &mut [u32],
    wts: &mut [f32],
    rowsum: &mut [f32],
    src_ty: crate::kquant::GgmlType,
) -> bool {
    let bpp = match src_ty {
        crate::kquant::GgmlType::F32 => 4usize,
        crate::kquant::GgmlType::Bf16 | crate::kquant::GgmlType::F16 => 2,
        _ => return false,
    };
    if !cuda_enabled()
        || !pf_router_dev_ok(nt, n, k)
        || sels.len() != nt * k
        || wts.len() != nt * k
        || rowsum.len() != nt
        || k_in == 0
        || bytes.len() != k_in.saturating_mul(n).saturating_mul(bpp)
        || match (dev_x, xs) {
            (Some(p), _) => p == 0,
            (None, Some(x)) => x.len() != nt * k_in,
            (None, None) => true,
        }
    {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        return false;
    }
    // logits の GEMM は専用カーネル (`dno_gemm_f32_ba`) が使えればそちら。
    // 使えないときだけ `dno_gemm_f32_mt` を要求する。
    let f_gemm = gemm_f32_mt_fn(ctx);
    let Some(f_rt) = router_dev_fns(ctx) else {
        return false;
    };
    let f_rt = f_rt.kernel;
    let w = match weights_for_gemm_f32(uid, bytes, k_in, n, src_ty) {
        Ok(w) => w,
        Err(_) => return false,
    };
    let Ok(lib) = lib() else {
        return false;
    };
    let r = ROUTER_DEV_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let RouterDevSc {
            x: dx,
            logits: dlogit,
            sels: dsel,
            wts: dwt,
            rowsum: drs,
        } = &mut *s;
        let xp = match (dev_x, xs) {
            (Some(p), _) => p,
            (None, Some(host)) => {
                Scratch::ensure(dx, nt * k_in * 4)?;
                let dx = dx.as_ref().unwrap();
                dx.copy_from(host)?;
                dx.ptr()
            }
            (None, None) => return Ok(()),
        };
        Scratch::ensure(dlogit, nt * n * 4)?;
        Scratch::ensure(dsel, nt * k * 4)?;
        Scratch::ensure(dwt, nt * k * 4)?;
        Scratch::ensure(drs, nt * 4)?;
        let (dlogit, dsel, dwt, drs) = (
            dlogit.as_ref().unwrap(),
            dsel.as_ref().unwrap(),
            dwt.as_ref().unwrap(),
            drs.as_ref().unwrap(),
        );
        let wp = w.buf.ptr();
        unsafe {
            // 1) logits = W @ x^T (デバイスに残す)。task 225: 専用 GEMM
            // (`dno_gemm_f32_ba`) が使えればそちら (累積順は同じ k 昇順なので値は
            // ビット一致)。使えなければ従来の `dno_gemm_f32_mt`。
            let fast = pf_ba_gemm_dev_launch(ctx, wp, xp, dlogit.ptr(), nt, k_in, n)?;
            if !fast {
                let Some(f_gemm) = f_gemm else {
                    return Ok(());
                };
                let (mut pw, mut px, mut po) = (wp, xp, dlogit.ptr());
                let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, k_in as c_int, n as c_int);
                let mut args = [
                    &mut pw as *mut _ as *mut c_void,
                    &mut px as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut nt_c as *mut _ as *mut c_void,
                    &mut k_c as *mut _ as *mut c_void,
                    &mut n_c as *mut _ as *mut c_void,
                ];
                let grid = (
                    n.div_ceil(GEMM_MT_ROWS) as u32,
                    nt.div_ceil(GEMM_MT_TOKS) as u32,
                    1,
                );
                ctx.launch_async(f_gemm, grid, (GEMM_MT_ROWS as u32, 16, 1), 0, &mut args)?;
            }
            // 2) softmax + top-k + 再正規化
            let (mut pl, mut ps, mut pwt, mut prs) =
                (dlogit.ptr(), dsel.ptr(), dwt.ptr(), drs.ptr());
            let (mut nt_c, mut ne_c, mut k_c) = (nt as c_int, n as c_int, k as c_int);
            let mut args = [
                &mut pl as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut pwt as *mut _ as *mut c_void,
                &mut prs as *mut _ as *mut c_void,
                &mut nt_c as *mut _ as *mut c_void,
                &mut ne_c as *mut _ as *mut c_void,
                &mut k_c as *mut _ as *mut c_void,
            ];
            let grid = (nt.div_ceil(4) as u32, 1, 1);
            ctx.launch_async(f_rt, grid, (128, 1, 1), 0, &mut args)?;
            ctx.sync()?;
            // 3) 選んだ k 個と重みだけ降ろす
            check(
                (lib.memcpy_dtoh)(
                    sels.as_mut_ptr() as *mut c_void,
                    dsel.ptr(),
                    nt * k * 4,
                ),
                "cuMemcpyDtoH (router sels)",
            )?;
            check(
                (lib.memcpy_dtoh)(wts.as_mut_ptr() as *mut c_void, dwt.ptr(), nt * k * 4),
                "cuMemcpyDtoH (router wts)",
            )?;
            check(
                (lib.memcpy_dtoh)(
                    rowsum.as_mut_ptr() as *mut c_void,
                    drs.ptr(),
                    nt * 4,
                ),
                "cuMemcpyDtoH (router rowsum)",
            )?;
        }
        Ok(())
    });
    r.is_ok()
}


#[cfg(test)]
pub(crate) static PF_FAST_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


// ---------------------------------------------------------------------------
// task 186: カーネル属性の診断 (DENNO_KERNEL_ATTR=1 のとき 1 行)
// ---------------------------------------------------------------------------
//
// PTX の仮想レジスタ数 (`%f<1161>` など) は NVVM の SSA 形なので実際の割り当ては
// 分からない。JIT 後の実体を `cuFuncGetAttribute` で読むと、ptxas が割り当てた
// レジスタ数とローカルメモリ (スピル) のバイト数が読める。scan が「レジスタに
// 載っていない」疑いを潰すための計測 (既定 off)。
pub(crate) fn log_kernel_attr(lib: &CudaLib, f: CUfunction, name: &str) {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("DENNO_KERNEL_ATTR").is_some()) {
        return;
    }
    let Some(g) = lib.func_get_attribute else {
        return;
    };
    // CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK=0 / SHARED_SIZE_BYTES=1 /
    // LOCAL_SIZE_BYTES=3 / NUM_REGS=4。
    let (mut regs, mut local, mut smem, mut maxthr) = (-1i32, -1i32, -1i32, -1i32);
    unsafe {
        if g(&mut regs, 4, f) != CUDA_SUCCESS {
            regs = -1;
        }
        if g(&mut local, 3, f) != CUDA_SUCCESS {
            local = -1;
        }
        if g(&mut smem, 1, f) != CUDA_SUCCESS {
            smem = -1;
        }
        if g(&mut maxthr, 0, f) != CUDA_SUCCESS {
            maxthr = -1;
        }
    }
    eprintln!(
        "[kattr] {name}: num_regs={regs} local_bytes={local} shared_bytes={smem} max_threads={maxthr}"
    );
}


#[cfg(test)]
pub(crate) static PF2_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// `DENNO_PF_SSM_GEMM_GPU` の解釈 (純関数)。`0`/`off`/`false`/`no`/空だけが off。
pub(crate) fn pf_ssm_gemm_gpu_from_value(v: Option<&str>) -> bool {
    match v {
        None => false,
        Some(s) => !matches!(s.trim(), "0" | "off" | "false" | "no" | ""),
    }
}


#[cfg(test)]
static SSM_GEMM_GPU_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


/// テスト用にゲートを固定する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_pf_ssm_gemm_gpu_for_test(v: Option<bool>) {
    SSM_GEMM_GPU_FORCE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}


/// beta/alpha をデバイスで回すか (既定 off。lab の結果で決める)。
pub(crate) fn pf_ssm_gemm_gpu_on() -> bool {
    #[cfg(test)]
    {
        match SSM_GEMM_GPU_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_ssm_gemm_gpu_from_value(std::env::var("DENNO_PF_SSM_GEMM_GPU").ok().as_deref())
    })
}


/// F32 重みをデバイスへ常駐させてポインタを返す (`PF_F32_W` の表を使う)。
/// **current context を張った後に呼ぶこと**。形が合わなければ `None`。
pub fn pf_f32_weight_dev(uid: u64, blocks: &[u8], k: usize, n: usize) -> Option<CUdeviceptr> {
    let want = k.checked_mul(n)?.checked_mul(4)?;
    if k == 0 || n == 0 || blocks.len() != want {
        return None;
    }
    let reg = PF_F32_W.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut m = match reg.lock() {
        Ok(g) => g,
        Err(_) => return None,
    };
    if let Some(b) = m.get(&uid) {
        return Some(b.ptr());
    }
    let b = DevBuffer::from_slice_tagged(blocks, crate::mem_audit::TAG_OTHER).ok()?;
    let p = b.ptr();
    m.insert(uid, b);
    Some(p)
}


// ---------------------------------------------------------------------------
// task 228: beta/alpha の重みを「型を問わず」デバイスへ常駐させる
// ---------------------------------------------------------------------------
//
// task 225 までの入口 (`pf_f32_weight_dev`) は **元の型が F32** のときだけ受けた。
// 実モデルの `ssm_beta` / `ssm_alpha` は GGUF によって F32 / BF16 / F16 / 量子型が
// 混在する (ローダ側 `f32_tensor` の 2026-10-31 コメント: Swift 1.5 は
// `ssm_alpha` などが BF16) ので、`DENNO_PF_BA_FAST=1` でも `ba_dev` が None に
// なり、`gdn_pf_chain` が **連鎖ごと諦めて**いた (lab t225a: ssm 2.98 -> 27.8 s)。
//
// ここでは非 F32 をホストで 1 回だけ f32 へ展開し (`dequant_row_into` =
// ホスト GEMV と同じ関数)、デバイスには f32 として置く。値が変わるのは
// 「ホストの mul+add 逐次和」と「f32 GEMM の k 昇順 fma」の丸め差だけ。
// 常駐量は 1 本あたり n*k*4 B (実モデル 48x5120 で 983 KB、層あたり beta+alpha で
// 2 MB 未満)。形・型が合わないときは None (呼び出し側はホスト経路へ)。

static PF_BA_W: OnceLock<Mutex<std::collections::HashMap<u64, DevBuffer>>> = OnceLock::new();


/// beta/alpha の重み 1 本を f32 でデバイスへ常駐させる。
/// **current context を張った後に呼ぶこと**。
///
/// 戻り値は (先頭ポインタ, f32 の行バイト数)。`blocks` の長さが
/// `n * row_bytes(ty, k)` と違う / `k` がブロックサイズの倍数でない / 型が
/// 展開できないときは `None`。
pub fn pf_ba_weight_dev(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
) -> Option<(CUdeviceptr, usize)> {
    let host = pf_ba_host_f32(ty, blocks, k, n)?;
    let reg = PF_BA_W.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut m = match reg.lock() {
        Ok(g) => g,
        Err(_) => return None,
    };
    if let Some(b) = m.get(&uid) {
        return Some((b.ptr(), k * 4));
    }
    let b = DevBuffer::from_slice_tagged(&host, crate::mem_audit::TAG_OTHER).ok()?;
    let p = b.ptr();
    m.insert(uid, b);
    Some((p, k * 4))
}


/// beta/alpha の重みを f32 の行優先 (`n` 行 × `k` 列) に展開する。GPU を触らない。
///
/// 1 行ずつ [`crate::kquant::dequant_row_into`] (ホスト GEMV が使うのと同じ関数)
/// で展開するので、出来上がる `f32` はホスト経路が内積に使う値と**ビット一致**する。
/// 形が合わない (`k` がブロックサイズの倍数でない / バイト数が違う) ときは `None`。
pub(crate) fn pf_ba_host_f32(
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if k == 0 || n == 0 || bs == 0 || k % bs != 0 || !crate::kquant::dequant_supported(ty) {
        return None;
    }
    let rb = (k / bs) * ts;
    if blocks.len() != n.checked_mul(rb)? {
        return None;
    }
    let mut host = vec![0f32; k.checked_mul(n)?];
    let mut row = vec![0f32; k];
    for r in 0..n {
        crate::kquant::dequant_row_into(ty, &blocks[r * rb..(r + 1) * rb], &mut row);
        host[r * k..(r + 1) * k].copy_from_slice(&row);
    }
    Some(host)
}


/// beta/alpha の 1 本がデバイス経路に載せられるか。載せられない理由を返す
/// (載せられるときは `None`)。GPU を触らない純関数。
///
/// 呼び出し側 (`forward_batch_chain`) と `gdn_pf_chain` が**同じ判定**を使うので、
/// 入口を開けてから連鎖が諦める (層ごと従来経路へ落ちる) ことが起きない。
pub(crate) fn pf_ba_w_check(
    ty: crate::kquant::GgmlType,
    k: usize,
    n: usize,
    rows: &std::ops::Range<usize>,
    nv: usize,
    e: usize,
) -> Option<String> {
    if !crate::kquant::dequant_supported(ty) {
        return Some(format!("型 {ty:?} は f32 へ展開できない"));
    }
    if k != e {
        return Some(format!("k={k} が n_embd={e} と違う"));
    }
    if rows.len() != nv {
        return Some(format!("行数 {} が n_v_heads={nv} と違う", rows.len()));
    }
    if n == 0 || rows.is_empty() || rows.end > n {
        return Some(format!("行範囲 {rows:?} が n={n} を超える"));
    }
    None
}


/// [`pf_ba_w_check`] の真偽版。
pub(crate) fn pf_ba_w_ok(
    ty: crate::kquant::GgmlType,
    k: usize,
    n: usize,
    rows: &std::ops::Range<usize>,
    nv: usize,
    e: usize,
) -> bool {
    pf_ba_w_check(ty, k, n, rows, nv, e).is_none()
}


/// task 228: 入口 (`DENNO_PF_SSM_GEMM_GPU` / `DENNO_PF_BA_FAST`) を開けたのに
/// beta/alpha のデバイス記述を作れなかった理由を **1 回だけ** warn に出す。
///
/// task 225 ではこれが無く、「開けたつもりが層ごとに従来経路へ落ちる」を
/// lab の ssm 時間 (+79%) から逆算するしかなかった。
pub(crate) fn pf_ba_diag_warn(why: &str) {
    static ONCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!("ssm beta/alpha のデバイス入口を閉じた: {why}");
    }
}


#[cfg(test)]
pub(crate) static PF3_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) static WY5_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) static WY7_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
pub(crate) static WY7Q_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) static WY6_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(test)]
pub(crate) static WY6P_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);




#[cfg(test)]
pub(crate) static WY4_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


#[cfg(test)]
pub(crate) static WY_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


// ---------------------------------------------------------------------------
// s167 (P6): GDN prefill の h2 D2H を専用コピーストリームで qkv/z GEMM と重ねる。
//
// GDN 層は 1 層あたり h2 (n*e*4 = 13k token で約 210MB) をホストへ戻し、beta/alpha
// をホストで計算する。従来はこれを既定 (legacy) ストリームで**同期** D2H して
// いたので、コピー時間がまるごと直列に乗っていた。beta/alpha が要るのは chunk
// scan からで、その手前の qkv/z GEMM は h2 (デバイス) だけに依存する — したがって
//   rmsnorm → { D2H h2 (コピーストリーム) || qkv/z GEMM (既定ストリーム) }
// の形にできる。同期点は「ホストが h2 を読む直前の join」1 つだけで、
//   - 同じバイト列を同じバッファから戻す (pinned 宛先が違うだけ)
//   - ホスト計算 (beta/alpha) の関数・引数・順序は不変
//   - カーネル・grid・引数・chunk 分割・状態契約は不変
// なので数値は 1 つも変わらない。コピーと GEMM は互いに独立 (どちらも CH_PFH を
// 読むだけ) で、書き込み先も別。
//
// 発動条件は `DENNO_PF_OVERLAP` (s172/P13 で**既定 ON**。`=0` で従来の直列経路)。
// 明示 OFF・driver に必要な API が無い・stream/event/pinned の作成に失敗、
// のいずれでも従来と同一の同期経路を通る。
// ---------------------------------------------------------------------------

/// s167 (P6): `DENNO_PF_OVERLAP` の解釈 (純粋関数)。
///
/// s172 (P13): **既定を ON にした** (OFF に戻すのは `DENNO_PF_OVERLAP=0` の
/// 明示指定だけ)。根拠は gdn_proj の実測内訳 (12,850 で 18.0 s) —
/// 「h2 の D2H ≈0.7 s + ホスト beta/alpha 8.1 s + qkv/z GEMM ≈9.2 s」で、
/// ホスト仕事は GEMM の裏に隠れる (隠せるのは min(コピー+ホスト, GEMM))。
/// 数値は経路によらず同一: 同じバイト列を同じバッファから戻すだけで、
/// ホスト計算・カーネル・引数・順序は変わらない。
pub fn parse_pf_overlap(raw: Option<&str>) -> bool {
    !matches!(raw, Some("0") | Some("false") | Some("off"))
}


/// 実行時の env 実効値 (初回だけ読む)。
pub(crate) fn pf_overlap_flag() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_pf_overlap(std::env::var("DENNO_PF_OVERLAP").ok().as_deref()))
}


/// s167 (P6): 経路決定 (純粋関数・GPU 非依存)。
/// `dev_ready` は「コピーストリーム・イベント・pinned 宛先が揃った」ことを表す
/// (呼び出し側が実際に作って渡す)。ここが false なら**必ず**従来経路に落ちる。
/// 桁あふれ・0 バイトも従来経路 (コピーする意味が無い)。
pub fn pf_h2_path(flag_on: bool, n: usize, e: usize, dev_ready: bool) -> PfH2Path {
    if !flag_on || !dev_ready {
        return PfH2Path::SyncD2h;
    }
    match n.checked_mul(e).and_then(|v| v.checked_mul(4)) {
        Some(bytes) if bytes > 0 => PfH2Path::CopyStream,
        _ => PfH2Path::SyncD2h,
    }
}


// ---------------------------------------------------------------------------
// s169 (P8): prefill 残差常駐経路の**位相計測** (表 C の `sl_lay_*` / DENNO_PF_DBG)
//
// なぜホスト span では駄目か: 残差常駐チェーンは層の中のカーネルを**発行する
// だけ**で待たない。`Instant` で区間を挟んでも写るのは発行の往復 (数十 ns) だけで、
// 実際 decode chain の `sl_lay_*` は 0.0 ms と出ていた。ここでは既定ストリームに
// **イベントを打つだけ**で位相を区切り、prefill 末尾に既にある同期 D2H が済んだ
// 後に `cuEventElapsedTime` で区間を読む:
//
//   - イベントは発行順にストリーム上で実行されるので、区間の経過 = 「その区間で
//     発行したカーネルの実行 + ホストが詰まっていた遊び」。待ちを足さない。
//   - **同期は 1 回も足さない** (層ループの中も外も)。読むのは prefill 最後の
//     既存の同期 D2H の後だけ。発行順・引数・カーネルは 1 つも変えない。
//   - `DENNO_PROFILE` も `DENNO_PF_DBG` も未設定なら、各サイトは分岐 1 回で戻る
//     (クロックも読まない・イベントも作らない)。プロファイラ build は要らない
//     (`DENNO_PF_DBG=1` だけで動く)。
//
// 位相の区切りは**発行グループの先頭**に置く (`pf_phase(PF_PH_*)`)。区間
// [イベント k, イベント k+1) はイベント k の時点で指定された位相に積まれる。
// 残差加算は `dno_add_rmsnorm` の `proj` 引数に融合しているので、ホスト経路の
// `sl_lay_residual` に相当する独立区間は**構造的に存在しない** (加算は
// norm_in / norm_post の区間に含まれる)。
// ---------------------------------------------------------------------------

/// 位相 id (表示順)。`PF_PH_NONE` は「位相の外」(prefill 入口 → 最初の位相 /
/// 層ループ末尾 → 最終 norm)。
pub const PF_PH_NONE: u8 = 0xFF;

pub const PF_PH_NORM_IN: usize = 0;

pub const PF_PH_FULL_QK: usize = 4;

pub const PF_PH_FULL_CORE: usize = 5;

pub const PF_PH_FULL_OPROJ: usize = 6;

pub const PF_PH_NORM_POST: usize = 7;

pub const PF_PH_FFN_GATE: usize = 8;

pub const PF_PH_SWIGLU: usize = 9;

pub const PF_PH_FFN_DOWN: usize = 10;

pub const PF_PH_N: usize = 11;


/// 表示名。`sl_lay_*` は表 C の行名と揃え、GDN の o_proj だけは表 C に行が無い
/// (decode chain と同じ扱い = 名前付きバケットを持たない) ので DBG 専用の名前。
pub const PF_PH_NAMES: [&str; PF_PH_N] = [
    "sl_lay_norm_in",
    "sl_lay_gdn_proj",
    "sl_lay_gdn_scan",
    "sl_lay_gdn_oproj",
    "sl_lay_full_qk",
    "sl_lay_full_core",
    "sl_lay_full_oproj",
    "sl_lay_norm_post",
    "sl_lay_ffn_gate",
    "sl_lay_swiglu",
    "sl_lay_ffn_down",
];


/// 位相 → 表 C の行 (None = 表 C に行を持たない = DBG 表示のみ)。
pub(crate) const PF_PH_BUCKETS: [Option<usize>; PF_PH_N] = [
    Some(crate::prof::SL_LAY_NORM_IN),
    Some(crate::prof::SL_LAY_GDN_PROJ),
    Some(crate::prof::SL_LAY_GDN_SCAN),
    None, // GDN o_proj (decode chain も名前付きバケットを持たない)
    Some(crate::prof::SL_LAY_FULL_QK),
    Some(crate::prof::SL_LAY_FULL_CORE),
    Some(crate::prof::SL_LAY_FULL_OPROJ),
    Some(crate::prof::SL_LAY_NORM_POST),
    Some(crate::prof::SL_LAY_FFN_GATE),
    Some(crate::prof::SL_LAY_SWIGLU),
    Some(crate::prof::SL_LAY_FFN_DOWN),
];


/// `DENNO_PF_DBG=1`: prefill 1 回ごとに位相の内訳を stderr へ 1 回出す
/// (プロファイラ build 不要・既定 OFF)。
pub(crate) fn pf_dbg_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_PF_DBG").ok().as_deref() == Some("1"))
}


/// イベントを `need` 個そろえる (grow-only・初回だけ作る)。
pub(crate) fn pf_ensure_events(t: &mut PfTiming, need: usize) -> bool {
    if t.evs.len() >= need {
        return true;
    }
    let Some(create) = lib().ok().and_then(|l| l.event_create) else {
        return false;
    };
    while t.evs.len() < need {
        let mut ev: CUevent = std::ptr::null_mut();
        // CU_EVENT_DEFAULT (0) = 計時有効。作れなければ計測を諦める (fail closed)。
        if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
            return false;
        }
        t.evs.push(ev);
    }
    true
}


/// 既定ストリームに境界イベントを 1 つ打ち、そこから始まる区間の位相を控える。
///
/// プールが尽きたら**その場で足す** (FFN は 1024 行タイルごとに 3 境界を打つので、
/// 層あたりの境界数は n に依存する — `begin` で見積もらずに grow で吸収する)。
pub(crate) fn pf_record(t: &mut PfTiming, phase: u8) -> bool {
    let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
        return false;
    };
    if t.used >= t.evs.len() {
        let want = t.evs.len().saturating_add(64);
        if !pf_ensure_events(t, want) {
            return false;
        }
    }
    let ev = t.evs[t.used];
    if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
        return false;
    }
    t.used += 1;
    t.phase_of.push(phase);
    true
}


/// 位相の先頭で境界を打つ (発行グループの先頭に置く)。計測していなければ
/// 分岐 1 回で戻る — 数値経路・発行順には触れない。
#[inline]
pub fn pf_phase(ph: usize) {
    // s180 (P41): full_core の内訳 (DENNO_FC_KPROF=1) の区間端もここで打つ
    // — 位相境界がそのまま full_core の先頭 / o_proj の先頭だから。
    fc_kprof_phase(ph);
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        if !t.active {
            return;
        }
        if !pf_record(&mut t, ph as u8) {
            t.active = false;
        }
    });
}


/// `DENNO_PF_DBG=1` の内訳 (1 prefill = 1 回)。wall / 位相の和 / 末尾 (最終 norm +
/// logits + 最終行 D2H) を並べるので、「位相の和が wall に届かない分」が必ず見える。
pub(crate) fn pf_dbg_print(t: &PfTiming, acc: &[f64; PF_PH_N]) {
    let wall = t.t0.map(|t0| t0.elapsed().as_secs_f64()).unwrap_or(0.0);
    let sum: f64 = acc.iter().sum();
    let tok = t.tokens.max(1) as f64;
    let per_tok = |secs: f64| secs * 1e3 / tok;
    let pct = |secs: f64| if wall > 0.0 { secs / wall * 100.0 } else { 0.0 };
    eprintln!(
        "[PF_DBG] prefill req={} n={} layers={} wall={:.1} ms | phases {:.1} ms ({:.1}%) | \
         {:.3} ms/token",
        t.req,
        t.tokens,
        t.layers,
        wall * 1e3,
        sum * 1e3,
        pct(sum),
        per_tok(wall)
    );
    for ph in 0..PF_PH_N {
        eprintln!(
            "[PF_DBG]   {:<18} {:>10.1} ms {:>6.1}% {:>9.3} ms/tok",
            PF_PH_NAMES[ph],
            acc[ph] * 1e3,
            pct(acc[ph]),
            per_tok(acc[ph])
        );
    }
    // 末尾 = 最終 norm + 最終行 D2H + lm_head。位相の外なので差で出す。
    let tail = (wall - sum).max(0.0);
    eprintln!(
        "[PF_DBG]   {:<18} {:>10.1} ms {:>6.1}% {:>9.3} ms/tok",
        "tail(final+logits)",
        tail * 1e3,
        pct(tail),
        per_tok(tail)
    );
}


// ---------------------------------------------------------------------------
// s180 (P41): full_core 相の内訳 (DENNO_FC_KPROF=1・既定 OFF)
//
// `sl_lay_full_core` の位相バケットは「区間内に発行したカーネル + ホストの詰まり」
// をまとめて 1 本にしているので、attention カーネル本体 (device) とホスト側の
// prepare / ゲート (host) のどちらが重いのかが分からない。ここでは**既存の
// launch / sync の位置**にイベントを打つだけで、full 層 1 枚の full_core 区間を
// 4 つに割って積算し、prefill の終了時に 1 行で出す:
//
//   [FC_KPROF] prep=<ms> attn=<ms> d2h=<ms> tail=<ms> calls=<n> total=<ms> (<ms>/token)
//
//   prep = 位相 FULL_CORE の境界 → attention カーネルの発行直前
//          (ホスト prepare: QK-Norm + RoPE + KV 行書き込み、KV ミラーの H2D、
//           q の pinned ステージング H2D)
//   attn = カーネル発行直前 → 発行直後 (`dno_attention_prefill` の実行。QK スコア +
//          softmax + PV が 1 発に融合しているので **qk と av には割れない**)
//   d2h  = 発行直後 → att の D2H 完了 (`chain_pf_att_inplace` の `cuMemcpyDtoHAsync`)
//   tail = D2H 完了 → 位相 FULL_OPROJ の境界 (sigmoid ゲート + att の H2D +
//          o_proj の prologue)
//
// 打つのはイベントだけ: launch の順序・引数・同期の位置は 1 つも変えない。
// `DENNO_FC_KPROF` 未設定なら各サイトは分岐 1 回で戻り、イベントも作らない
// (プロファイラ build 不要)。既存の `DENNO_PROFILE` / `DENNO_PF_DBG` の出力は
// この env を立てても変わらない (別プール・別行)。
// ---------------------------------------------------------------------------

/// 区間数 (prep / attn / d2h / tail)。
const FC_IV_N: usize = 4;

/// full 層 1 枚あたりの境界イベント数 (区間数 + 1)。
const FC_MARK_N: usize = FC_IV_N + 1;

/// 区間名 (表示順・REPORT_P41.md の表と同じ)。
const FC_IV_NAMES: [&str; FC_IV_N] = ["prep", "attn", "d2h", "tail"];


/// `DENNO_FC_KPROF=1` のときだけ full_core の内訳を取る (既定 OFF・分岐 1 回)。
fn fc_kprof_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_FC_KPROF").ok().as_deref() == Some("1"))
}


/// 1 prefill 分の full_core 内訳の状態 (スレッドローカル・リクエストごとに使い回す)。
pub(crate) struct FcKprof {
    /// 計測中 (begin 済みで read 前)。
    active: bool,
    /// いま記録中の呼び出し (= full 層 1 枚) が先頭の境界から始まっている。
    open: bool,
    /// 記録済みの境界 (bit k = スロット k)。
    have: u8,
    /// 呼び出し番号 (先頭の境界で 1 増える)。
    call: u32,
    /// イベントのプール (grow-only。初回の prefill で層数ぶん作る)。
    evs: Vec<CUevent>,
    /// このリクエストで打ったイベント数。
    used: usize,
    /// イベント k の (呼び出し番号, スロット)。
    meta: Vec<(u32, u8)>,
    /// 5 点そろった呼び出しの数 (= 内訳を読めた層数)。
    calls: usize,
    /// トークン数 (ms/token の分母)。
    tokens: usize,
}


thread_local! {
    pub(crate) static FC_KPROF: std::cell::RefCell<FcKprof> =
        std::cell::RefCell::new(FcKprof::default());
}


/// イベントを `need` 個そろえる (grow-only・初回だけ作る)。
fn fc_ensure_events(t: &mut FcKprof, need: usize) -> bool {
    if t.evs.len() >= need {
        return true;
    }
    let Some(create) = lib().ok().and_then(|l| l.event_create) else {
        return false;
    };
    while t.evs.len() < need {
        let mut ev: CUevent = std::ptr::null_mut();
        // CU_EVENT_DEFAULT (0) = 計時有効。作れなければ計測を諦める (fail closed)。
        if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
            return false;
        }
        t.evs.push(ev);
    }
    true
}


/// prefill 入口: full_core の内訳計測を開く (`pf_timing_begin` から 1 回)。
pub(crate) fn fc_kprof_begin(tokens: usize, layers: usize) {
    if !fc_kprof_on() {
        return;
    }
    FC_KPROF.with(|c| {
        let mut t = c.borrow_mut();
        t.active = false;
        t.open = false;
        t.have = 0;
        t.call = 0;
        t.used = 0;
        t.meta.clear();
        t.calls = 0;
        t.tokens = tokens;
        // full 層は高々 layers 枚。イベントは grow-only のプールに 1 回だけ作る
        // (2 回目以降の prefill はここで何もしない)。
        let need = layers.saturating_mul(FC_MARK_N).max(FC_MARK_N);
        if !fc_ensure_events(&mut t, need) {
            return;
        }
        t.active = true;
    });
}


/// 位相境界から区間の端を打つ (`pf_phase` から呼ぶ)。FULL_CORE = 先頭、
/// FULL_OPROJ = 末尾 (o_proj の prologue までが tail に入る)。
#[inline]
fn fc_kprof_phase(ph: usize) {
    if !fc_kprof_on() {
        return;
    }
    if ph == PF_PH_FULL_CORE {
        fc_kprof_mark(0);
    } else if ph == PF_PH_FULL_OPROJ {
        fc_kprof_mark(FC_MARK_N as u8 - 1);
    }
}


/// 既定ストリームに境界イベントを 1 つ打つ (イベントのみ・同期も転送もしない)。
///
/// 先頭 (slot 0) を打っていない呼び出しの途中スロットは捨てる — ホスト経路や
/// CPU attend に落ちた層は 5 点そろわないので、区間として読まない。
pub(crate) fn fc_kprof_mark(slot: u8) {
    if !fc_kprof_on() {
        return;
    }
    FC_KPROF.with(|c| {
        let mut t = c.borrow_mut();
        if !t.active {
            return;
        }
        if slot == 0 {
            t.open = true;
            t.have = 1;
            t.call = t.call.wrapping_add(1);
        } else if !t.open {
            return;
        } else {
            t.have |= 1u8 << slot;
        }
        let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
            t.active = false;
            return;
        };
        if t.used >= t.evs.len() {
            let want = t.used.saturating_add(FC_MARK_N * 8);
            if !fc_ensure_events(&mut t, want) {
                t.active = false;
                return;
            }
        }
        let ev = t.evs[t.used];
        if unsafe { rec(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
            t.active = false;
            return;
        }
        t.used += 1;
        let call = t.call;
        t.meta.push((call, slot));
        if slot as usize == FC_MARK_N - 1 {
            if t.have == (1u8 << FC_MARK_N) - 1 {
                t.calls += 1;
            }
            t.open = false;
        }
    });
}


/// prefill 末尾 (**既存の**同期 D2H の後): 区間を読み、1 行だけ stderr へ出す。
/// 積算はリクエストごと (次の `fc_kprof_begin` でリセット)。
pub(crate) fn fc_kprof_read() {
    if !fc_kprof_on() {
        return;
    }
    FC_KPROF.with(|c| {
        let mut t = c.borrow_mut();
        if !t.active {
            return;
        }
        t.active = false;
        t.open = false;
        let Some(elapsed) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            eprintln!("[FC_KPROF] unavailable (no cuEventElapsedTime)");
            return;
        };
        let mut acc = [0.0f64; FC_IV_N];
        let mut bad = false;
        for w in 1..t.meta.len() {
            let (c0, s0) = t.meta[w - 1];
            let (c1, s1) = t.meta[w];
            // 同じ呼び出しの隣り合う境界だけを 1 区間として読む (5 点そろわない
            // 呼び出しは自動的に落ちる)。
            if c0 != c1 || s1 != s0 + 1 {
                continue;
            }
            let mut ms: f32 = 0.0;
            if unsafe { elapsed(&mut ms, t.evs[w - 1], t.evs[w]) } != CUDA_SUCCESS
                || !ms.is_finite()
                || ms < 0.0
            {
                bad = true;
                break;
            }
            acc[s0 as usize] += ms as f64 / 1e3;
        }
        if bad {
            eprintln!(
                "[FC_KPROF] unavailable (event elapsed failed; n={} calls={})",
                t.tokens, t.calls
            );
            return;
        }
        let total: f64 = acc.iter().sum();
        if t.calls == 0 && total <= 0.0 {
            eprintln!(
                "[FC_KPROF] no complete full_core call (n={} marks={})",
                t.tokens, t.used
            );
            return;
        }
        let tok = t.tokens.max(1) as f64;
        let mut line = String::from("[FC_KPROF]");
        for i in 0..FC_IV_N {
            line.push_str(&format!(" {}={:.1}", FC_IV_NAMES[i], acc[i] * 1e3));
        }
        line.push_str(&format!(
            " calls={} total={:.1} ({:.3} ms/token)",
            t.calls,
            total * 1e3,
            total * 1e3 / tok
        ));
        eprintln!("{line}");
    });
}


/// s165 (P2): 残差常駐経路の**グローバル**入口条件 (純粋関数・GPU 非依存)。
/// `all_q4` は「全層の attn/FFN 投影が q4」、`diag` は層別診断 (lchk) の有無。
/// 診断中はホストの `x` が層ごとに更新される前提なので発動しない。
/// `gemm_min` / `max_m` は env 実効値 (`gemm_min_m()` / `max_m()`) を渡す。
pub fn pf_res_gate(
    n: usize,
    e: usize,
    all_q4: bool,
    diag: bool,
    gemm_min: usize,
    max_m: usize,
) -> bool {
    if !all_q4 || diag {
        return false;
    }
    if n == 0 || e == 0 || n < gemm_min || n <= max_m {
        return false;
    }
    // dno_add_rmsnorm の動的 shmem = e*4 (≤48KB)
    if e % 32 != 0 || e.saturating_mul(4) > 48 * 1024 {
        return false;
    }
    // 端数タイルが GEMV 帯域に入る形では per-op 経路と演算列が変わる
    let rem = n % GDN_PREFILL_TILE;
    if rem != 0 && rem <= max_m {
        return false;
    }
    true
}


/// 実行時の env 実効値で [`pf_res_gate`] を評価する。
pub fn pf_res_gate_now(n: usize, e: usize, all_q4: bool, diag: bool) -> bool {
    pf_res_gate(n, e, all_q4, diag, gemm_min_m(), max_m())
}


/// s165 (P2): 層の q4 重みをデバイスへ事前登録する。登録失敗 (確保失敗) を
/// 残差を更新する前に確定させるためのもの (チェーン内の登録はキャッシュヒット)。
pub fn pf_layer_dev_register(
    attn: &PfAttnW<'_>,
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
) -> Option<()> {
    let _ = chain_ctx()?;
    let reg = |w: &crate::quant::Linear| -> Option<()> {
        let q = match w {
            crate::quant::Linear::Q4(q) => q,
            _ => return None,
        };
        weights_for_q4(q.uid, &q.blocks, q.k, q.n).ok()?;
        Some(())
    };
    match attn {
        PfAttnW::Gdn { qkv, z, o } => {
            reg(qkv)?;
            reg(z)?;
            reg(o)?;
        }
        PfAttnW::Full { q, k, v, o } => {
            reg(q)?;
            reg(k)?;
            reg(v)?;
            reg(o)?;
        }
    }
    reg(gate)?;
    reg(up)?;
    reg(down)
}


pub(crate) fn pf_qkv_gen() -> u64 {
    PFQK_GEN.with(|c| c.get())
}


/// s247 (P47): full_qk の pinned ステージング (q + k) の上限バイト数。
/// 既定 1 GiB (`docs/FULL_QK_D2H_PINNED.md` §5/§6 の「プール上限 1.0 GB」)。
///
/// この上限を超える n ではプールを**1 バイトも確保せず**に従来の pageable
/// 経路へ落ちる (fail-closed)。65k (n=14,043) の q+k は 747.8 MB なので既定では
/// 収まる。256 MiB (268,435,456 B) に絞ると n ≤ 5,041 行でしか発動しない
/// (q+k = n×13,312×4 B = 53,248 B/行) — q の 1 発が 690 MB あるので、それ以上の
/// n を 256 MiB に収めるには D2H の分割が必要で、それは設計メモ §3 (3 発・同順を
/// 保つ) と両立しない。
pub(crate) fn pf_qkv_pin_max() -> usize {
    static MAX: OnceLock<usize> = OnceLock::new();
    *MAX.get_or_init(|| {
        std::env::var("DENNO_PF_QKV_PIN_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1 << 30)
    })
}


/// s247 (P47): `DENNO_PF_QKV_PIN_DBG=1` のとき、pinned ステージングが初めて
/// 効いた層で 1 行だけ出す (cold A/B の前に「どちらの経路か」を確定させるため)。
pub(crate) fn pf_qkv_pin_dbg() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_PF_QKV_PIN_DBG").ok().as_deref() == Some("1"))
}


/// s250 (P53): full_qk の q の D2H を k/v の GEMM と重ねるか (純粋関数・既定 ON)。
///
/// `DENNO_PF_QK_OVERLAP=0` で従来の「既定ストリームで 3 発を直列」へ戻す
/// (cold A/B の腕)。`DENNO_PF_OVERLAP=0` は上位の全停止スイッチで、こちらも
/// OFF になる ([`pf_qk_overlap_enabled`])。
pub fn parse_pf_qk_overlap(raw: Option<&str>) -> bool {
    !matches!(raw, Some("0") | Some("false") | Some("off"))
}


/// [`parse_pf_qk_overlap`] の実効値 (env が無ければ ON)。上位の
/// `DENNO_PF_OVERLAP` が OFF ならこちらも OFF (P6 の h2 と同じ停止スイッチ)。
fn pf_qk_overlap_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_pf_qk_overlap(std::env::var("DENNO_PF_QK_OVERLAP").ok().as_deref())
            && pf_overlap_flag()
    })
}


/// s250 (P53): full_qk の q の D2H を k/v の GEMM と重ねるためのコピーストリームを
/// 用意し、既定ストリームの**いまこの時点まで**の完了を待たせる。
///
/// **呼び出し位置が契約**: q の GEMM を起動した直後・k/v の GEMM を起動する前。
/// ここでイベントを打つと q の D2H は q の GEMM の完了だけを待って走り出し、
/// 後続の k/v GEMM (既定ストリーム) と重なる。記録点を k/v の起動より後ろへ
/// ずらすとイベントが 3 発全部の完了を待ち、1 バイトも重ならない。
///
/// 隠れる量 (27B・n=14,043 の 1 層):
/// - q の D2H = rows*qcols*4 = 690.2 MB。実測単価 14.3 GB/s (s68) で ≈48 ms。
/// - k/v の GEMM (各 1,024 列・0.147 TFLOP) ≈ 9.8 ms は必ず隠れる。
/// - k/v の D2H (各 57.5 MB・計 8.0 ms) は q と同じコピーエンジンに並ぶので、
///   隠れるかは実機の取り合い次第 (隠れれば上限 17.8 ms/層)。
/// したがって短くなるのは ≈9.8 ms/層 = 16 層で ≈0.16 s/request (推定・下限)。
/// 上限は ≈0.29 s/request。
///
/// 失敗はすべて `None` (フラグ OFF・driver API 欠落・stream/event 作成失敗) で、
/// **GPU_BROKEN は立てない** — 「既定ストリームで直列に流せばよい」種類の失敗。
/// 数値はどちらの経路でも同一: 同じバイト列を同じ宛先へ戻し、変わるのは
/// 「どのストリームで発行するか」だけ (カーネル・引数・宛先・発行順は不変)。
pub(crate) fn pf_qk_copy_begin() -> Option<CUstream> {
    let _ctx = chain_ctx()?;
    if !pf_qk_overlap_enabled() {
        return None;
    }
    let lib = lib().ok()?;
    let (stream_create, event_create, event_record, stream_wait_event) = (
        lib.stream_create?,
        lib.event_create?,
        lib.event_record?,
        lib.stream_wait_event?,
    );
    CHAIN.with(|cell| -> Option<CUstream> {
        let mut s = cell.borrow_mut();
        if s.qk_stream.is_none() {
            // 非ブロッキング必須: legacy (既定) ストリームは blocking ストリームと
            // 暗黙に同期するため、blocking で作ると GEMM がコピー完了を待つ。
            const CU_STREAM_NON_BLOCKING: u32 = 0x01;
            const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
            let mut st: CUstream = std::ptr::null_mut();
            if unsafe { stream_create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
                return None;
            }
            let st = CopyStream(st);
            let mut ev: CUevent = std::ptr::null_mut();
            if unsafe { event_create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return None;
            }
            s.qk_stream = Some(st);
            s.qk_ev_q = Some(CopyEvent(ev));
        }
        let st = s.qk_stream.as_ref()?.0;
        let ev = s.qk_ev_q.as_ref()?.0;
        // 既定ストリーム (null) に「ここまで」を打ち、コピーストリームを参加させる。
        if unsafe { event_record(ev, std::ptr::null_mut()) } != CUDA_SUCCESS {
            return None;
        }
        if unsafe { stream_wait_event(st, ev, 0) } != CUDA_SUCCESS {
            return None;
        }
        Some(st)
    })
}


/// カーネル別の積算 (ns) と call 数。**表示専用** (数値経路からは読まない)。
pub(crate) static KPROF_PRE_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) static KPROF_STATE_NS: AtomicU64 = AtomicU64::new(0);

pub(crate) static KPROF_SCAN_NS: AtomicU64 = AtomicU64::new(0);

/// 3 区間すべて読めた call 数。
pub(crate) static KPROF_CALLS: AtomicU64 = AtomicU64::new(0);

/// 区間を読めなかった (未完了・負値・イベント欠け) call 数。半端な数字は積まない。
pub(crate) static KPROF_MISS: AtomicU64 = AtomicU64::new(0);


/// 積算の読み出し結果 (整形・テスト用の値型)。
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub(crate) struct KprofAcc {
    pub pre_ns: u64,
    pub state_ns: u64,
    pub scan_ns: u64,
    pub calls: u64,
    /// 区間を読めずに捨てた call 数 (未完了・負値・イベント欠け・イベント生成失敗)。
    pub miss: u64,
}


/// 積算を読む。
pub(crate) fn kprof_acc() -> KprofAcc {
    KprofAcc {
        pre_ns: KPROF_PRE_NS.load(Ordering::Relaxed),
        state_ns: KPROF_STATE_NS.load(Ordering::Relaxed),
        scan_ns: KPROF_SCAN_NS.load(Ordering::Relaxed),
        calls: KPROF_CALLS.load(Ordering::Relaxed),
        miss: KPROF_MISS.load(Ordering::Relaxed),
    }
}


/// 3 区間 (ms) を積算する。`KprofCall::finish` の算術だけを切り出したもの
/// (GPU 無しで試せるようにするため)。
pub(crate) fn kprof_add_ms(pre_ms: f32, state_ms: f32, scan_ms: f32) {
    let ns = |x: f32| (x as f64 * 1e6).round() as u64;
    KPROF_PRE_NS.fetch_add(ns(pre_ms), Ordering::Relaxed);
    KPROF_STATE_NS.fetch_add(ns(state_ms), Ordering::Relaxed);
    KPROF_SCAN_NS.fetch_add(ns(scan_ms), Ordering::Relaxed);
    KPROF_CALLS.fetch_add(1, Ordering::Relaxed);
}


/// prefill 終了時に出す行を決める (整形・テスト用の純関数)。
///
/// - dump 経路が生きている → None (summary 側が同じ行を出す)
/// - それ以外 → **必ず** 1 行 (`calls == 0 && miss == 0` でも出す)。
///   env ON なのに走査サイトへ 1 度も入らなかった場合、`calls=0 miss=0` の行が
///   出るので「静かに何も出ない」が構造的に起こらない (P43 の要件)。
pub(crate) fn kprof_prefill_line(acc: &KprofAcc, dump_path_live: bool) -> Option<String> {
    if dump_path_live {
        return None;
    }
    Some(acc.render())
}


/// 終了フック: 計測を開いたスレッドが終わるときに 1 行だけ出す (dump 経路
/// (`DENNO_PROFILE`) を通らない使い方のための保険)。読むのは static の積算だけ
/// なので CUDA は呼ばない = コンテキスト破棄後でも安全。
pub(crate) struct KprofExitGuard;


thread_local! {
    /// 計測を開いたスレッドに 1 個だけ置く (env OFF なら作られない)。
    pub(crate) static KPROF_EXIT_GUARD: KprofExitGuard = KprofExitGuard;
}


/// 終了フックを置く (初回だけ)。計測していないスレッドには作らない。
fn kprof_arm_exit_guard() {
    KPROF_EXIT_GUARD.with(|_| ());
}


/// 4 本のイベント (e0=開始 / e1=pass1 後 / e2=pass2 後 / e3=pass3 後)。
/// CUevent は `*mut c_void` (Send でない) ので static に置くためのラッパを挟む。
/// ハンドルは不変で、触るのは record / elapsed_time だけ。
pub(crate) struct KprofEvents([CUevent; 4]);


/// イベントの置き場 (初回だけ生成・grow しない)。作れなければ None = 計測しない。
static KPROF_EV: OnceLock<Option<std::sync::Mutex<KprofEvents>>> = OnceLock::new();


fn kprof_events() -> Option<&'static std::sync::Mutex<KprofEvents>> {
    KPROF_EV
        .get_or_init(|| {
            let create = lib().ok()?.event_create?;
            let mut evs: [CUevent; 4] = [std::ptr::null_mut(); 4];
            for slot in evs.iter_mut() {
                let mut ev: CUevent = std::ptr::null_mut();
                // CU_EVENT_DEFAULT (0) = 計時有効。作れなければ計測を諦める
                // (fail closed。途中まで作った分はプロセス終了まで保持する)。
                if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                    return None;
                }
                *slot = ev;
            }
            Some(std::sync::Mutex::new(KprofEvents(evs)))
        })
        .as_ref()
}


/// 1 call 分の計測ハンドル (イベントを排他で握る)。
pub(crate) struct KprofCall {
    lock: std::sync::MutexGuard<'static, KprofEvents>,
    /// 打てたイベントのビット (bit i = e_i)。4 本そろわなければ積算しない。
    marks: u32,
}


// s245 (P45): 3 pass 後の `ctx.sync()` をデバイス出力経路から外したため、区間の
// 読み出しを「次の既存の同期」まで遅らせる。その置き場 (スレッドごと)。
//
// `KprofCall` はイベントプールの排他 (`MutexGuard`) を握っているので保留中も
// そのまま保持する — 途中で手放すと他スレッドが同じ 4 本を上書きして区間が壊れる。
// 読むのは必ず「既存の同期が済んだ後」(次の層の `kprof_begin` / prefill 末尾の
// `pf_timing_read`) だけで、そこでも未完了なら `finish` が miss に数える (P43)。
thread_local! {
    pub(crate) static KPROF_PENDING: std::cell::RefCell<Option<KprofCall>> =
        std::cell::RefCell::new(None);
}


/// 1 call 分の区間を「次の既存の同期」まで保留する (P45)。
pub(crate) fn kprof_defer(k: KprofCall) {
    // 前の call が残っていれば先に読む (取りこぼさない)。通常は `kprof_begin` が
    // 読んでいるのでここは空。
    kprof_read_pending();
    KPROF_PENDING.with(|c| *c.borrow_mut() = Some(k));
}


/// 保留した区間を今の時点で読む (**既存の同期が済んだ後**でのみ呼ぶ)。
/// 同期・launch・イベント生成はしない (読むだけ)。env OFF なら分岐 1 回で戻る。
pub(crate) fn kprof_read_pending() {
    if !gdn_kprof_on() {
        return;
    }
    let k = KPROF_PENDING.with(|c| c.borrow_mut().take());
    if let Some(k) = k {
        k.finish();
    }
}


/// 計測を開く。`DENNO_GDN_KPROF` 未設定なら分岐 1 回で None (イベントも作らない)。
pub(crate) fn kprof_begin() -> Option<KprofCall> {
    if !gdn_kprof_on() {
        return None;
    }
    // s245 (P45): 前の層で保留した区間をここで読む (この時点では次の層の h2 join /
    // 同期 D2H が済んでいる)。読めなければ miss。
    kprof_read_pending();
    let Some(ev) = kprof_events() else {
        // env ON なのにイベントを作れない = 計測不能。黙って 0 行にせず miss に
        // 数える (表示行に miss=<n> が出るので「静かに no-op」にならない)。
        KPROF_MISS.fetch_add(1, Ordering::Relaxed);
        return None;
    };
    kprof_arm_exit_guard();
    // poison しても中身はハンドルだけなので続ける。
    let lock = ev.lock().unwrap_or_else(|e| e.into_inner());
    Some(KprofCall { lock, marks: 0 })
}


/// `kprof_begin()` の結果に 1 本打つ (計測していなければ分岐 1 回で何もしない)。
#[inline]
pub(crate) fn kprof_mark(kp: &mut Option<KprofCall>, idx: usize) {
    if let Some(k) = kp.as_mut() {
        k.mark(idx);
    }
}


/// task 269: F32 / BF16 の beta/alpha 重みを VRAM に常駐させる (uid ごと)。
///
/// 量は 36 層 × 491 KB (BF16) ≈ 18 MB で、`DENNO_GPU_WEIGHT_MB` の追い出し
/// 対象にはしない (毎トークン触るので常駐が前提)。確保に失敗したら `None`
/// = 従来経路 (呼び出し側が事前に型を見て選ぶので、ここでの失敗は OOM だけ)。
pub(crate) fn ba_gpu_weight(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
) -> Option<CUdeviceptr> {
    static REG: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, (usize, usize, usize)>>> =
        OnceLock::new();
    // (k, n, bytes) と DevBuffer の ptr を別表で持つ (DevBuffer は Send でないため
    // static の生ポインタで持つ。プロセス寿命で解放しない)。
    static PTRS: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, usize>>> = OnceLock::new();
    let bpp = match ty {
        crate::kquant::GgmlType::F32 => 4usize,
        crate::kquant::GgmlType::Bf16 => 2usize,
        _ => return None,
    };
    let bytes = k.checked_mul(n)?.checked_mul(bpp)?;
    if blocks.len() != bytes {
        return None;
    }
    let reg = REG.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let ptrs = PTRS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    {
        let g = reg.lock().ok()?;
        if let Some(&(ek, en, eb)) = g.get(&uid) {
            if (ek, en, eb) != (k, n, bytes) {
                return None;
            }
            let p = ptrs.lock().ok()?;
            return p.get(&uid).map(|&v| v as CUdeviceptr);
        }
    }
    let buf = DevBuffer::from_slice_tagged(blocks, crate::mem_audit::TAG_MIX_F32).ok()?;
    let p = buf.ptr();
    // プロセス寿命の常駐 (18-35 MB)。DevBuffer の drop を呼ばない。
    std::mem::forget(buf);
    reg.lock().ok()?.insert(uid, (k, n, bytes));
    ptrs.lock().ok()?.insert(uid, p as usize);
    Some(p)
}


pub(crate) const CU_STREAM_NON_BLOCKING_BAPAR: u32 = 0x01;

pub(crate) const CU_EVENT_DISABLE_TIMING_BAPAR: u32 = 0x02;


/// kernels_gdn_scan2.ptx の `dno_gdn_scan_sig2`。
pub(crate) struct Scan2Kernel {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


pub(crate) fn scan2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GDN_SCAN2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_scan2.ptx"),
    }
}


/// kernels_gdn_scan3.ptx の `dno_gdn_scan_sig3`。
pub(crate) struct Scan3Kernel {
    #[allow(dead_code)]
    module: CUmodule,
    f: CUfunction,
}


pub(crate) fn scan3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GDN_SCAN3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gdn_scan3.ptx"),
    }
}


/// task 664: 分割後の `src/cuda/*.rs` をソート順に連結したソース全文 (テストの
/// テキスト検査用)。旧 `include_str!("cuda.rs")` の置き換え。
#[cfg(test)]
pub(crate) fn cuda_src() -> &'static str {
    static SRC: OnceLock<String> = OnceLock::new();
    SRC.get_or_init(|| {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("src/cuda が読めない: {e}"))
            .filter_map(|e| {
                let p = e.ok()?.path();
                if p.extension().map_or(false, |x| x == "rs") {
                    p.file_name()?.to_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        let mut out = String::new();
        for n in names {
            let t = std::fs::read_to_string(dir.join(&n))
                .unwrap_or_else(|e| panic!("{n} が読めない: {e}"));
            out.push_str(&t);
            out.push('\n');
        }
        out
    })
}
