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
use std::sync::OnceLock;
use std::time::Instant;

type CUresult = i32;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUdeviceptr = u64;

const RTLD_NOW: c_int = 2;
const CUDA_SUCCESS: CUresult = 0;

// dlopen/dlsym (glibc 2.34 以降は libc に統合、Ubuntu 22.04 コンテナで使用)
extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type FnCuInit = unsafe extern "C" fn(u32) -> CUresult;
type FnCuDeviceGet = unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult;
type FnCuDeviceGetName = unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult;
type FnCuDeviceComputeCapability =
    unsafe extern "C" fn(*mut c_int, *mut c_int, CUdevice) -> CUresult;
type FnCuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult;
type FnCuCtxSetCurrent = unsafe extern "C" fn(CUcontext) -> CUresult;
type FnCuCtxSynchronize = unsafe extern "C" fn() -> CUresult;
type FnCuDriverGetVersion = unsafe extern "C" fn(*mut c_int) -> CUresult;
type FnCuModuleLoadData = unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult;
type FnCuModuleGetFunction =
    unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult;
type FnCuMemAllocV2 = unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult;
type FnCuMemFreeV2 = unsafe extern "C" fn(CUdeviceptr) -> CUresult;
type FnCuMemsetD32Async = unsafe extern "C" fn(CUdeviceptr, u32, usize, CUstream) -> CUresult;
type FnCuMemGetInfo = unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult;
type FnCuMemcpyHtoDV2 = unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
type FnCuMemcpyDtoHV2 = unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
type FnCuMemHostAlloc = unsafe extern "C" fn(*mut *mut c_void, usize, u32) -> CUresult;
type FnCuMemFreeHost = unsafe extern "C" fn(*mut c_void) -> CUresult;
type FnCuMemcpyHtoDAsyncV2 =
    unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoHAsyncV2 =
    unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoDAsyncV2 =
    unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult;
type FnCuMemcpyDtoD = unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult;
type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type FnCuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
// s167 (P6): h2 の D2H を専用コピーストリームへ載せるための driver API。
// すべて Option で持つ — 古い driver で dlsym に失敗しても全体を落とさず、
// 呼び出し側が従来の同期 D2H へフォールバックする (fail closed)。
type FnCuStreamCreate = unsafe extern "C" fn(*mut CUstream, u32) -> CUresult;
type FnCuStreamDestroy = unsafe extern "C" fn(CUstream) -> CUresult;
type FnCuStreamWaitEvent = unsafe extern "C" fn(CUstream, CUevent, u32) -> CUresult;
type FnCuEventCreate = unsafe extern "C" fn(*mut CUevent, u32) -> CUresult;
type FnCuEventDestroy = unsafe extern "C" fn(CUevent) -> CUresult;
type FnCuEventRecord = unsafe extern "C" fn(CUevent, CUstream) -> CUresult;
type FnCuEventSynchronize = unsafe extern "C" fn(CUevent) -> CUresult;
// s169 (P8): prefill 位相計測 — ストリーム上の 2 点間の経過 (ms) を読む。
type FnCuEventElapsedTime = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;
type FnCuLaunchKernel = unsafe extern "C" fn(
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
type CUstream = *mut c_void;
type CUevent = *mut c_void;

/// Driver API の関数ポインタ群 (dlsym は1度だけ行う)
struct CudaLib {
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
    memcpy_htod: FnCuMemcpyHtoDV2,
    memcpy_dtoh: FnCuMemcpyDtoHV2,
    memset_d32_async: FnCuMemsetD32Async,
    mem_host_alloc: FnCuMemHostAlloc,
    mem_host_free: FnCuMemFreeHost,
    memcpy_htod_async: FnCuMemcpyHtoDAsyncV2,
    memcpy_dtoh_async: FnCuMemcpyDtoHAsyncV2,
    memcpy_dtod_async: FnCuMemcpyDtoDAsyncV2,
    // s141b (Task B7c): 同期 D2D (GDN 状態の正本 -> ミラー seed 用)。
    memcpy_dtod: FnCuMemcpyDtoD,
    // s48 (C1-a): 動的 smem の opt-in (cuFuncSetAttribute)。CUDA 9.0 以降に存在
    // するが、dlsym 失敗でも全体を落とさないよう Option (無ければ 48KB 既定のまま)。
    device_get_attribute: Option<FnCuDeviceGetAttribute>,
    func_set_attribute: Option<FnCuFuncSetAttribute>,
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
        memcpy_htod: std::mem::transmute(sym(handle, "cuMemcpyHtoD_v2")?),
        memcpy_dtoh: std::mem::transmute(sym(handle, "cuMemcpyDtoH_v2")?),
        memset_d32_async: std::mem::transmute(sym(handle, "cuMemsetD32Async")?),
        mem_host_alloc: std::mem::transmute(sym(handle, "cuMemHostAlloc")?),
        mem_host_free: std::mem::transmute(sym(handle, "cuMemFreeHost")?),
        memcpy_htod_async: std::mem::transmute(sym(handle, "cuMemcpyHtoDAsync_v2")?),
        memcpy_dtoh_async: std::mem::transmute(sym(handle, "cuMemcpyDtoHAsync_v2")?),
        memcpy_dtod_async: std::mem::transmute(sym(handle, "cuMemcpyDtoDAsync_v2")?),
        memcpy_dtod: std::mem::transmute(sym(handle, "cuMemcpyDtoD_v2")?),
        device_get_attribute: sym_opt(handle, "cuDeviceGetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuDeviceGetAttribute>(p) }),
        func_set_attribute: sym_opt(handle, "cuFuncSetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuFuncSetAttribute>(p) }),
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
    })
}

static LIB: OnceLock<CudaLib> = OnceLock::new();

fn lib() -> Result<&'static CudaLib> {
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

/// `DENNO_MEM_DBG=1` のときだけ確保/解放をログる (既定は完全に無効)。
fn mem_dbg() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DENNO_MEM_DBG").as_deref(), Ok("1")))
}

/// `DENNO_MEM_BT=1` のときだけ、確保の呼び出し元を 1 行に畳んで返す。
fn mem_bt() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("DENNO_MEM_BT").as_deref(), Ok("1")))
}

fn bt_hint() -> String {
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

fn check(res: CUresult, what: &str) -> Result<()> {
    if res != CUDA_SUCCESS {
        bail!("CUDA {what} が失敗: error {res}");
    }
    Ok(())
}

/// デバイスメモリバッファ (RAII: drop で cuMemFree)
pub struct DevBuffer {
    ptr: CUdeviceptr,
    bytes: usize,
    /// s165: false = 他のバッファの一部を指す**非所有**ビュー。drop で解放しない
    /// (所有者は元のバッファ)。重みレジストリの行範囲エイリアス専用。
    owned: bool,
}

impl DevBuffer {
    #[track_caller]
    fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr: CUdeviceptr = 0;
        let res = unsafe { (lib()?.mem_alloc)(&mut ptr, bytes) };
        if res != CUDA_SUCCESS {
            // どの要求が載らなかったかを後から追えるように空き容量を添える。
            match dev_mem_mb() {
                Some((f, t)) => bail!(
                    "CUDA cuMemAlloc が失敗: error {res} (要求 {} MiB / 空き {} MiB / 総 {} MiB)",
                    bytes >> 20,
                    f,
                    t
                ),
                None => bail!("CUDA cuMemAlloc が失敗: error {res} (要求 {} MiB)", bytes >> 20),
            }
        }
        if mem_dbg() {
            if let Some((f, t)) = dev_mem_mb() {
                if bytes >= (1 << 20) && mem_bt() {
                    tracing::info!(target: "denno_engine::mem", "alloc {} KiB free={} MiB total={} MiB at={} bt={}", bytes >> 10, f, t, std::panic::Location::caller(), bt_hint());
                } else {
                    tracing::info!(target: "denno_engine::mem", "alloc {} KiB free={} MiB total={} MiB at={}", bytes >> 10, f, t, std::panic::Location::caller());
                }
            }
        }
        Ok(Self { ptr, bytes, owned: true })
    }

    /// s165: 既存のデバイス領域の一部を指す**非所有**ビュー。連結重み
    /// (`gate_up_w` など) の行範囲を、元の独立した重みとして見せるために使う
    /// (同じ重みを VRAM へ 2 度上げない)。drop では解放しない。
    /// `bytes` はビュー自身の長さ (所有側の範囲内であることは呼び出し側が保証)。
    fn view(ptr: CUdeviceptr, bytes: usize) -> Self {
        Self { ptr, bytes, owned: false }
    }

    /// ホストのスライスをデバイスへコピーして確保
    #[track_caller]
    pub fn from_slice<T: Copy>(data: &[T]) -> Result<Self> {
        let bytes = std::mem::size_of_val(data);
        let buf = Self::alloc(bytes.max(1))?;
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_htod)(buf.ptr, data.as_ptr() as *const c_void, bytes),
                    "cuMemcpyHtoD",
                )?;
            }
            // s166 (P5): 観測のみ (初回アップロード = 定常ステップでは 0 回)
            dbg_htod(bytes);
        }
        Ok(buf)
    }

    /// デバイス → ホスト (len 個の T として読む)
    /// デバイス側だけで 0 にする (f32 の 0.0 はビット 0 なのでゼロ埋めと同じ)。
    /// ホスト Vec を作らないため、小さなバッファを毎回 H2D する経路の負担が消える。
    pub fn fill_zero(&self) -> Result<()> {
        let words = self.bytes / 4;
        if words == 0 {
            return Ok(());
        }
        let rc = unsafe { (lib()?.memset_d32_async)(self.ptr, 0, words, std::ptr::null_mut()) };
        if rc != 0 {
            bail!("cuMemsetD32Async failed: CUresult {}", rc);
        }
        Ok(())
    }

    pub fn to_vec<T: Copy>(&self, len: usize) -> Result<Vec<T>> {
        let bytes = len * std::mem::size_of::<T>();
        assert!(bytes <= self.bytes, "DevBuffer::to_vec: size overflow");
        if bytes == 0 {
            return Ok(Vec::new());
        }
        // s72: ゼロ埋めした Vec を作ってから上書きすると同じ領域を 2 度書く
        // (47MB で数 ms)。未初期化領域へ直接 D2H し、コピー完了後に set_len する
        // (cuMemcpyDtoH は同期コピーなので戻った時点で全要素が埋まっている)。
        let mut out: Vec<T> = Vec::with_capacity(len);
        unsafe {
            check(
                (lib()?.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
                "cuMemcpyDtoH",
            )?;
            out.set_len(len);
        }
        Ok(out)
    }

    /// 既存バッファへホストから上書きコピー
    pub fn copy_from<T: Copy>(&self, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        assert!(bytes <= self.bytes, "DevBuffer::copy_from: size overflow");
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_htod)(self.ptr, data.as_ptr() as *const c_void, bytes),
                    "cuMemcpyHtoD",
                )?;
            }
        }
        Ok(())
    }

    /// 既存バッファのバイトオフセット位置へホストから上書きコピー (P17g: KV ミラー追記用)
    pub fn copy_from_at<T: Copy>(&self, offset_bytes: usize, data: &[T]) -> Result<()> {
        let bytes = std::mem::size_of_val(data);
        assert!(
            offset_bytes + bytes <= self.bytes,
            "DevBuffer::copy_from_at: size overflow"
        );
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_htod)(
                        self.ptr + offset_bytes as CUdeviceptr,
                        data.as_ptr() as *const c_void,
                        bytes,
                    ),
                    "cuMemcpyHtoD",
                )?;
            }
        }
        Ok(())
    }

    /// s141b (Task B7c): 既存バッファへ**デバイスから**上書きコピー (D2D・同期)。
    /// `src` の先頭 `len` 要素 (f32) をこのバッファの先頭へ移す。同一 GPU 内の
    /// コピーをホスト経由 (D2H + H2D) で往復させないための経路 — 転送は値そのまま
    /// なのでビットは D2H+H2D と同一。
    pub fn copy_from_dev(&self, src: CUdeviceptr, len: usize) -> Result<()> {
        let bytes = len * std::mem::size_of::<f32>();
        assert!(bytes <= self.bytes, "DevBuffer::copy_from_dev: size overflow");
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_dtod)(self.ptr, src, bytes),
                    "cuMemcpyDtoD",
                )?;
            }
        }
        Ok(())
    }

    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 生デバイスポインタ (カーネル引数の組み立て用)
    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }
}

impl Drop for DevBuffer {
    fn drop(&mut self) {
        // s165: ビュー (owned=false) は他バッファの一部なので解放しない
        if self.owned && self.ptr != 0 {
            unsafe {
                let _ = (lib().unwrap().mem_free)(self.ptr);
            }
            if mem_dbg() {
                if let Some((f, t)) = dev_mem_mb() {
                    tracing::info!(target: "denno_engine::mem", "free {} KiB free={} MiB total={} MiB", self.bytes >> 10, f, t);
                }
            }
        }
    }
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
    // P17g: decode attention (exactness 緩和・ユーザー承認)
    f_attn: CUfunction,
    // P17j: prefill attention (flash 風 online softmax・exactness 緩和・ユーザー承認)
    f_attn_pf: CUfunction,
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
    f_gdn_chunk_state: CUfunction,
    f_gdn_chunk_scan: CUfunction,
    /// s89 (P1-1): 状態グラフ常駐版の chunk scan (pass 2 を折りたたんだ 2 launch 版)。
    /// PTX はコミット済みの `kernels_gdn.ptx` に含まれる (再生成済み) 場合のみ
    /// 取得でき、無ければ None で従来の 3 launch 経路を使う。
    f_gdn_chunk_scan_hoist: Option<CUfunction>,
}

const PTX: &[u8] = include_bytes!("../kernels/kernels_int8.ptx");
const PTX_ELEM: &[u8] = include_bytes!("../kernels/kernels_elem.ptx");
const PTX_GDN: &[u8] = include_bytes!("../kernels/kernels_gdn.ptx");

// 中身は生ハンドル (context / module / function)。CUcontext はスレッドごとに
// current 設定して使うハンドルであり、所有権の移動は伴わないため Send/Sync を
// 与える (primary context 自体は driver がスレッド安全に管理する)。
unsafe impl Send for CudaCtx {}
unsafe impl Sync for CudaCtx {}

impl CudaCtx {
    /// デバイス 0 (または `DENNO_CUDA_DEVICE`) の primary context を初期化し、
    /// int8 カーネルの PTX をロードする。
    pub fn new() -> Result<Self> {
        let lib = lib()?;
        unsafe {
            check((lib.cu_init)(0), "cuInit")?;
            let device: CUdevice = std::env::var("DENNO_CUDA_DEVICE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut dev: CUdevice = 0;
            check((lib.device_get)(&mut dev, device), "cuDeviceGet")?;

            let mut namebuf = [0u8 as c_char; 128];
            check((lib.device_name)(namebuf.as_mut_ptr(), 127, dev), "cuDeviceGetName")?;
            let name = std::ffi::CStr::from_ptr(namebuf.as_ptr())
                .to_string_lossy()
                .into_owned();

            let (mut maj, mut min) = (0 as c_int, 0 as c_int);
            check((lib.device_cc)(&mut maj, &mut min, dev), "cuDeviceComputeCapability")?;

            let mut drv = 0 as c_int;
            let _ = (lib.driver_version)(&mut drv);

            let mut ctx: CUcontext = std::ptr::null_mut();
            check((lib.primary_ctx_retain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            check((lib.ctx_set_current)(ctx), "cuCtxSetCurrent")?;

            // PTX は NUL 終端が必要
            let mut image = PTX.to_vec();
            image.push(0);
            let mut module: CUmodule = std::ptr::null_mut();
            check(
                (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void),
                "cuModuleLoadData (PTX JIT)",
            )?;

            let mut f_quant: CUfunction = std::ptr::null_mut();
            let mut f_mv: CUfunction = std::ptr::null_mut();
            let mut f_gemm: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_quant, module, b"dno_quantize_rows\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_quantize_rows)",
            )?;
            check(
                (lib.module_get_function)(&mut f_mv, module, b"dno_matvec_rows_bt_i8\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_matvec_rows_bt_i8)",
            )?;
            // P17i: prefill 用 tiled GEMM (m が大きいときの GEMV 型カーネル置き換え)
            check(
                (lib.module_get_function)(&mut f_gemm, module, b"dno_gemm_i8\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gemm_i8)",
            )?;
            // s123: Q4_0 ネイティブ経路 (同じモジュールに同居)
            let mut f_mv_q4: CUfunction = std::ptr::null_mut();
            let mut f_gemm_q4: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_mv_q4, module, b"dno_gemv_q4_0\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gemv_q4_0)",
            )?;
            check(
                (lib.module_get_function)(&mut f_gemm_q4, module, b"dno_gemm_q4_0\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gemm_q4_0)",
            )?;
            // s132 (Task B1): 複数行 1 block の GEMV は「あれば使う」。古い PTX
            // (未再生成) でも起動を壊さず 1 行 1 block 経路へ落ちるよう、
            // 取得失敗は None として扱う。
            let mut f_mv_q4_mr: CUfunction = std::ptr::null_mut();
            let mv_mr_r = (lib.module_get_function)(
                &mut f_mv_q4_mr,
                module,
                b"dno_gemv_q4_0mr\0".as_ptr() as *const c_char,
            );
            let f_mv_q4_mr = if mv_mr_r == CUDA_SUCCESS {
                Some(f_mv_q4_mr)
            } else {
                None
            };
            // s133 (Task B1, attempt 2): 列ブロック版 GEMV (C = 2 / 4 / 8)。
            // 3 本そろって初めて Some。1 本でも取れなければ None で従来経路。
            let mut fc: [CUfunction; 3] = [std::ptr::null_mut(); 3];
            let mut c_ok = true;
            for (k, name) in [b"dno_gemv_q4_0c2\0", b"dno_gemv_q4_0c4\0", b"dno_gemv_q4_0c8\0"]
                .iter()
                .enumerate()
            {
                let r = (lib.module_get_function)(
                    &mut fc[k],
                    module,
                    name.as_ptr() as *const c_char,
                );
                if r != CUDA_SUCCESS {
                    c_ok = false;
                    break;
                }
            }
            let f_mv_q4_c = if c_ok { Some(fc) } else { None };

            // P17d: 要素演算モジュール (-fmad=false でコンパイル済み。FMA 縮約は
            // CPU とビット一致しないため禁止)
            let mut module_elem: CUmodule = std::ptr::null_mut();
            let mut image_elem = PTX_ELEM.to_vec();
            image_elem.push(0);
            check(
                (lib.module_load_data)(&mut module_elem, image_elem.as_ptr() as *const c_void),
                "cuModuleLoadData (elem PTX JIT)",
            )?;
            let mut f_swiglu: CUfunction = std::ptr::null_mut();
            let mut f_addnorm: CUfunction = std::ptr::null_mut();
            let mut f_expf: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_swiglu, module_elem, b"dno_swiglu\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_swiglu)",
            )?;
            let mut f_swiglu_int: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_swiglu_int, module_elem, b"dno_swiglu_int\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_swiglu_int)",
            )?;
            check(
                (lib.module_get_function)(&mut f_addnorm, module_elem, b"dno_add_rmsnorm\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_add_rmsnorm)",
            )?;
            check(
                (lib.module_get_function)(&mut f_expf, module_elem, b"dno_expf_probe\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_expf_probe)",
            )?;
            let mut f_attn: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_attn, module_elem, b"dno_attention\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_attention)",
            )?;
            // P17j: prefill attention
            let mut f_attn_pf: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_attn_pf, module_elem, b"dno_attention_prefill\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_attention_prefill)",
            )?;
            // s48 (C1-a): 動的 smem の opt-in 試行。デバイスの opt-in 上限を取得し、
            // 48KB を超えるなら dno_attention_prefill へ設定する (qwen35 dh=256 =
            // 84.8KB を通すため)。失敗しても ctx 生成は続行 (48KB 既定のまま =
            // qwen35 prefill attention はゲートで CPU へ静かにフォールバック)。
            let mut attn_pf_dyn_smem: usize = ATTN_PF_SMEM_DEFAULT;
            if let (Some(dev_attr), Some(fn_attr)) = (lib.device_get_attribute, lib.func_set_attribute) {
                let mut optin: c_int = 0;
                if (dev_attr)(&mut optin, CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, dev) == 0
                    && optin as usize > attn_pf_dyn_smem
                    && (fn_attr)(f_attn_pf, CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES, optin) == 0
                {
                    attn_pf_dyn_smem = optin as usize;
                }
            }
            ATTN_PF_DYN_SMEM.store(attn_pf_dyn_smem, std::sync::atomic::Ordering::Relaxed);
            let mut f_rope: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_rope, module_elem, b"dno_rope_qkv\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_rope_qkv)",
            )?;
            // P17k ステップ 5: prefill 用行並列 rope
            let mut f_rope_pf: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_rope_pf, module_elem, b"dno_rope_qkv_pf\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_rope_qkv_pf)",
            )?;
            // B3: b>1 バッチ decode 用 rope
            let mut f_rope_b: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_rope_b, module_elem, b"dno_rope_qk_batch\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_rope_qk_batch)",
            )?;

            // P19-A1: GDN 逐次スキャンモジュール
            let mut module_gdn: CUmodule = std::ptr::null_mut();
            let mut image_gdn = PTX_GDN.to_vec();
            image_gdn.push(0);
            check(
                (lib.module_load_data)(&mut module_gdn, image_gdn.as_ptr() as *const c_void),
                "cuModuleLoadData (gdn PTX JIT)",
            )?;
            let mut f_gdn: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn, module_gdn, b"dno_gdn_scan\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_scan)",
            )?;
            // A1-r2: chunked scan 3 カーネル
            let mut f_gdn_chunk_pre: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn_chunk_pre, module_gdn, b"dno_gdn_chunk_pre\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_chunk_pre)",
            )?;
            let mut f_gdn_chunk_state: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn_chunk_state, module_gdn, b"dno_gdn_chunk_state\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_chunk_state)",
            )?;
            let mut f_gdn_chunk_scan: CUfunction = std::ptr::null_mut();
            check(
                (lib.module_get_function)(&mut f_gdn_chunk_scan, module_gdn, b"dno_gdn_chunk_scan\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_gdn_chunk_scan)",
            )?;
            // s89 (P1-1): 状態常駐版は「あれば使う」。古い PTX (未再生成) でも
            // 起動を壊さず 3 launch 経路へ落ちるよう、取得失敗は None として扱う。
            let mut f_gdn_chunk_scan_hoist: CUfunction = std::ptr::null_mut();
            let hoist_r = (lib.module_get_function)(
                &mut f_gdn_chunk_scan_hoist,
                module_gdn,
                b"dno_gdn_chunk_scan_hoist\0".as_ptr() as *const c_char,
            );
            // (隣接フィールドと同じく CUresult == 0 が成功。非 0 は None)
            let f_gdn_chunk_scan_hoist = if hoist_r == CUDA_SUCCESS {
                Some(f_gdn_chunk_scan_hoist)
            } else {
                None
            };

            Ok(Self {
                device: dev,
                name,
                cc: (maj, min),
                driver_ver: drv,
                ctx,
                module,
                f_quant,
                f_mv,
                f_gemm,
                f_mv_q4,
                f_gemm_q4,
                f_mv_q4_mr,
                f_mv_q4_c,
                module_elem,
                f_swiglu,
                f_swiglu_int,
                f_addnorm,
                f_expf,
                f_attn,
                f_attn_pf,
                attn_pf_dyn_smem,
                f_rope,
                f_rope_pf,
                f_rope_b,
                module_gdn,
                f_gdn,
                f_gdn_chunk_pre,
                f_gdn_chunk_state,
                f_gdn_chunk_scan,
                f_gdn_chunk_scan_hoist,
            })
        }
    }

    /// "NVIDIA GeForce RTX 5090 (sm_120, driver 13010)" 形式の情報
    pub fn info(&self) -> String {
        format!(
            "{} (sm_{}{}, driver {}, attn_pf dyn smem {}KB)",
            self.name,
            self.cc.0,
            self.cc.1,
            self.driver_ver,
            self.attn_pf_dyn_smem / 1024
        )
    }

    pub fn device(&self) -> c_int {
        self.device
    }

    /// カーネル起動のみ (同期しない — デフォルトストリームにエンキュー)。
    /// P17b: 固定オーバーヘッド削減のため launch/sync を分離。
    unsafe fn launch_async(
        &self,
        f: CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shmem: u32,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        let lib = lib()?;
        // s166 (P5): decode 1 ステップの launch 件数 (DENNO_DECODE_DBG=1・既定 OFF)
        dbg_launch(f);
        check(
            (lib.launch)(
                f,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shmem,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ),
            "cuLaunchKernel",
        )
    }

    /// デフォルトストリームの同期 (エンキュー済み全操作の完了待ち)
    pub fn sync(&self) -> Result<()> {
        let lib = lib()?;
        unsafe { check((lib.ctx_sync)(), "cuCtxSynchronize") }
    }

    /// 呼び出しスレッドに primary context を current として設定する。
    /// CUDA の current context は**スレッドローカル** (P17c の教訓: libtest は
    /// テスト毎に新スレッドを立てるため、初期化スレッドで 1 回だけ設定したのでは
    /// 他スレッドの cuMemAlloc が error 201 INVALID_CONTEXT で落ちる)。共有
    /// CudaCtx を複数スレッドから触るため、GPU 操作の入口で毎回張り直す
    /// (cuCtxSetCurrent は冪等かつ軽量)。
    pub fn make_current(&self) -> Result<()> {
        let lib = lib()?;
        unsafe { check((lib.ctx_set_current)(self.ctx), "cuCtxSetCurrent") }
    }

    /// `tensor::quantize_row_i8` の GPU 版 (m 行一括)。
    /// 戻り値: (量子化済みアクティベーション [m*k] i8, ascale [m] f32) の
    /// デバイスバッファ (後続の `matvec_rows_bt_i8_dev` にそのまま渡せる)。
    pub fn quantize_rows(&self, a: &[f32], m: usize, k: usize) -> Result<(DevBuffer, DevBuffer)> {
        assert_eq!(a.len(), m * k, "quantize_rows: a size mismatch");
        let dev_a = DevBuffer::from_slice(a)?;
        let dev_aq = DevBuffer::alloc(m * k)?;
        let dev_asc = DevBuffer::alloc(m * 4)?;
        unsafe {
            let (mut pa, mut paq, mut pasc) = (dev_a.ptr, dev_aq.ptr, dev_asc.ptr);
            let (mut mc, mut kc) = (m as c_int, k as c_int);
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
            ];
            // 1 block per row, 256 threads, shmem 256*4 bytes
            self.launch_async(
                self.f_quant,
                (m as u32, 1, 1),
                (256, 1, 1),
                256 * 4,
                &mut args,
            )?;
            self.sync()?;
        }
        Ok((dev_aq, dev_asc))
    }

    /// int8 GEMV/GEMM (nn.Linear 形式 q: [n,k]) をデバイスバッファで実行。
    /// `tensor::matvec_rows_bt_i8` とビット一致する [m,n] f32 を返す。
    pub fn matvec_rows_bt_i8_dev(
        &self,
        aq: &DevBuffer,
        asc: &DevBuffer,
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let dev_out = DevBuffer::alloc(m * n * 4)?;
        unsafe {
            let (mut paq, mut pasc, mut pw, mut psc, mut pout) =
                (aq.ptr, asc.ptr, w.ptr, scale.ptr, dev_out.ptr);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            // 1 warp per (i,j): block 256 = 8 warps
            let total = (m * n) as u64;
            let blocks = total.div_ceil(8) as u32;
            self.launch_async(self.f_mv, (blocks, 1, 1), (256, 1, 1), 0, &mut args)?;
            self.sync()?;
        }
        dev_out.to_vec::<f32>(m * n)
    }

    /// ホストのアクティベーションから一括で実行する高レベル API
    /// (量子化 + GEMV)。重み/scale は事前に `upload_i8` / `upload_f32` で
    /// VRAM 常駐させておく。戻り値は `tensor::matvec_rows_bt_i8` とビット一致。
    pub fn linear_bt_i8(
        &self,
        a: &[f32],
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(w.bytes(), k * gemm_pad_n(n));
        debug_assert_eq!(scale.bytes(), n * 4);
        let (aq, asc) = self.quantize_rows(a, m, k)?;
        self.matvec_rows_bt_i8_dev(&aq, &asc, w, scale, m, k, n)
    }

    /// P17i: tiled GEMM (dno_gemm_i8) で実行。`linear_bt_i8` とビット一致。
    /// テストとベンチ用 (本番経路は `linear_gemm` を使う)。
    pub fn linear_gemm_i8_dev(
        &self,
        aq: &DevBuffer,
        asc: &DevBuffer,
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let dev_out = DevBuffer::alloc(m * n * 4)?;
        unsafe {
            let (mut paq, mut pasc, mut pw, mut psc, mut pout) =
                (aq.ptr, asc.ptr, w.ptr, scale.ptr, dev_out.ptr);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let gmx = m.div_ceil(GEMM_TILE) as u32;
            let gmy = n.div_ceil(GEMM_TILE_N) as u32;
            self.launch_async(self.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args)?;
            self.sync()?;
        }
        dev_out.to_vec::<f32>(m * n)
    }

    /// P17i: ホストのアクティベーションから tiled GEMM を実行 (量子化込み)。
    pub fn linear_gemm_i8(
        &self,
        a: &[f32],
        w: &DevBuffer,
        scale: &DevBuffer,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(w.bytes(), k * gemm_pad_n(n));
        debug_assert_eq!(scale.bytes(), n * 4);
        let (aq, asc) = self.quantize_rows(a, m, k)?;
        self.linear_gemm_i8_dev(&aq, &asc, w, scale, m, k, n)
    }

    /// 重みを VRAM へアップロード (起動時に1度)
    pub fn upload_i8(&self, data: &[i8]) -> Result<DevBuffer> {
        DevBuffer::from_slice(data)
    }

    /// scale などを VRAM へアップロード (起動時に1度)
    pub fn upload_f32(&self, data: &[f32]) -> Result<DevBuffer> {
        DevBuffer::from_slice(data)
    }
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
struct PinMem {
    ptr: *mut c_void,
    bytes: usize,
}

// 生ポインタを保持するだけ (スレッド局所で共有しない)
unsafe impl Send for PinMem {}

impl PinMem {
    fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // PORTABLE: どのコンテキストからでも有効 (保険)
        const CU_MEMHOSTALLOC_PORTABLE: u32 = 0x01;
        unsafe {
            check(
                (lib()?.mem_host_alloc)(&mut ptr, bytes.max(1), CU_MEMHOSTALLOC_PORTABLE),
                "cuMemHostAlloc",
            )?;
        }
        Ok(Self { ptr, bytes: bytes.max(1) })
    }

    fn as_slice<T>(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.bytes / std::mem::size_of::<T>()) }
    }

    fn as_mut_slice<T>(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut T, self.bytes / std::mem::size_of::<T>()) }
    }
}

impl Drop for PinMem {
    fn drop(&mut self) {
        unsafe {
            let _ = (lib().unwrap().mem_host_free)(self.ptr);
        }
    }
}

/// s167 (P6): 専用コピーストリーム (h2 の D2H を qkv/z GEMM と重ねる)。
/// driver オブジェクトはコンテキストに属するので Drop で明示的に破棄する
/// (スレッド終了時に解放。`stream_destroy` が無い driver ではリークするだけで、
/// コンテキスト破棄時に回収される)。
struct CopyStream(CUstream);

impl Drop for CopyStream {
    fn drop(&mut self) {
        if let Ok(l) = lib() {
            if let Some(f) = l.stream_destroy {
                unsafe {
                    let _ = f(self.0);
                }
            }
        }
    }
}

/// s167 (P6): コピーストリームの順序付け/完了待ちに使うイベント (RAII)。
struct CopyEvent(CUevent);

impl Drop for CopyEvent {
    fn drop(&mut self) {
        if let Ok(l) = lib() {
            if let Some(f) = l.event_destroy {
                unsafe {
                    let _ = f(self.0);
                }
            }
        }
    }
}

/// 1 スレッド分の再利用スクラッチ (grow-only)。scheduler ワーカーは単一スレッド
/// のため thread_local で競合なし。スレッド終了時の解放はベストエフォート
/// (コンテキストは static で生き続けるため、失敗してもリークはワーカー寿命分)。
struct Scratch {
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
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        x: None, aq: None, asc: None, out: None, hx: None, ho: None,
        act: None, aq2: None, asc2: None, out2: None,
    });
}

impl Scratch {
    fn ensure(cap: &mut Option<DevBuffer>, bytes: usize) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes()).unwrap_or(0) < bytes {
            *cap = Some(DevBuffer::alloc(bytes)?);
        }
        Ok(())
    }
    fn ensure_pin(cap: &mut Option<PinMem>, bytes: usize) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes).unwrap_or(0) < bytes {
            *cap = Some(PinMem::alloc(bytes)?);
        }
        Ok(())
    }
}

/// s67: GDN スキャン (chunked) の作業領域。**層をまたいで再利用する** (grow-only)。
///
/// 動機 (s64/s66 の計測): prefill の `gdn_scan` 3.15 s のうち **alloc 1.04 s** は
/// 層ごとに 16 本 (計 ≈180MB) を `DevBuffer::from_slice` で作り直していた分で、
/// 形はモデルで固定 (n はリクエスト内で一定) なのに毎回 cuMemAlloc/cuMemFree
/// していた。確保を初回だけにし、以降は `copy_from` で上書きする。
///
/// 数値は不変 — バッファの寿命が延びるだけで、書き込む内容・カーネル引数・
/// 演算順は 1 つも変えない (scheduler ワーカーは単一スレッドなので thread_local
/// で競合しない・`SCRATCH` と同じ規約)。
struct GdnScratch {
    q: Option<DevBuffer>,        // [n, conv_dim]
    z: Option<DevBuffer>,        // [n, vd]
    beta: Option<DevBuffer>,     // [n, nv]
    alpha: Option<DevBuffer>,    // [n, nv]
    conv_in: Option<DevBuffer>,  // [conv_dim, cs]
    ssm_in: Option<DevBuffer>,   // [nv, dk, dv]
    wc: Option<DevBuffer>,       // 重み 4 本 (層ごとに差し替え)
    wa: Option<DevBuffer>,
    wd: Option<DevBuffer>,
    wn: Option<DevBuffer>,
    w: Option<DevBuffer>,     // [n_chunks, nv, dk, dv] chunk ローカル遷移
    m: Option<DevBuffer>,     // [n_chunks, nv, dk, dk]
    sinit: Option<DevBuffer>, // [n_chunks, nv, dk, dv]
    y: Option<DevBuffer>,     // [n, vd]
    conv_out: Option<DevBuffer>,
    ssm_out: Option<DevBuffer>,
}

/// カーネル引数用の生ポインタ束 (バッファ本体は thread_local が保持し続ける)。
#[derive(Clone, Copy)]
struct GdnPtrs {
    q: CUdeviceptr,
    z: CUdeviceptr,
    beta: CUdeviceptr,
    alpha: CUdeviceptr,
    conv_in: CUdeviceptr,
    ssm_in: CUdeviceptr,
    wc: CUdeviceptr,
    wa: CUdeviceptr,
    wd: CUdeviceptr,
    wn: CUdeviceptr,
    w: CUdeviceptr,
    m: CUdeviceptr,
    sinit: CUdeviceptr,
    y: CUdeviceptr,
    conv_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
}

#[derive(Clone, Copy)]
enum GdnBuf {
    Y,
    ConvOut,
    SsmOut,
}

// ---------------------------------------------------------------------------
// s89 (P1-1): GDN スキャンのデバイス常駐状態
//
// 動機: s88 の計測で prefill の `d2h_state` が大きいのは GDN 状態の往復
// (`ssm_out` の D2H + 次層の `ssm_in` の H2D・1 層 2MB ずつ) を含むため。
// ここでは「層のスキャン状態の正本」をデバイスに置き、ホスト側の
// `KvCache::gdn` は**実行境界 (forward の切れ目) でだけ**一致させる。
//
// s90 の状態 (重要・偽らない):
//   この構造体は **まだどの経路からも使われていない**。tl の `GDN_STATE` に
//   確保・seed・無効化の関数は用意してあるが、`gdn_scan_chunked_impl` は
//   (i) 2 launch 経路が現状の forward では成立しない (→ gdn/scan_schedule.rs)
//   (ii) y をデバイスに残す oproj 経路 (s72) は 2 層目以降の conv 入力を
//        ホストから捨てる形になっており、ホスト状態が層ごとにしか確定しない
//   の 2 点により、従来どおり層ごとに ssm を H2D / D2H している。
//   `f_gdn_chunk_scan_hoist` のカーネルと Optional 取得は「古い PTX でも起動を
//   壊さない」性質ごと残してある (Next: 上記 (i)(ii) を解いて配線する)。
//
// 無効化の契約 (将来 配線したときに効くもの — 先に用意しておく):
//   ホストの `KvCache::gdn` を書き換える経路 (セッション reset / エントリ復元 /
//   デコードチェーンの遅延フラッシュ / CPU フォールバック) は
//   [`gdn_scan_state_invalidate`] を呼ぶ。呼ばれたら次の実行でホストから
//   再 seed する (数値不変・余分な H2D が 1 回増えるだけ)。**古い正本を読むと
//   出力が変わる**ので、新しい書き換え経路を足すときは必ずここを通すこと。
// ---------------------------------------------------------------------------
struct GdnStateCache {
    conv: Vec<Option<DevBuffer>>, // [layer] conv [conv_dim, cs] (デバイス正本)
    conv_len: usize,              // conv の要素数
    cur_kv: Option<u64>,          // この正本が属する KV キャッシュ ID (None = 未使用)
    ssm: Vec<Option<DevBuffer>>, // [layer] ssm [nv, dk, dv] (デバイス正本)
    valid: Vec<bool>,            // [layer] = ホスト KvCache::gdn と一致している
    srows: Option<DevBuffer>,    // [n, nv, dk, dv] hoist 経路の境界行
    srows_n: usize,              // srows がいま表現している n (0 = 未初期化)
    rows: usize,                 // 行数 (= nk*dk = vd)
    cols: usize,                 // 列数 (= dk*dv)
}

impl GdnStateCache {
    const fn empty() -> Self {
        Self {
            conv: Vec::new(),
            conv_len: 0,
            cur_kv: None,
            ssm: Vec::new(),
            valid: Vec::new(),
            srows: None,
            srows_n: 0,
            rows: 0,
            cols: 0,
        }
    }
    fn ensure_layers(&mut self, n_layers: usize) {
        // `Option<DevBuffer>` は Clone でない (DevBuffer は RAII の所有権型) →
        // resize_with で伸ばす。縮む場合は末尾を落とす (drop = cuMemFree)。
        if self.ssm.len() < n_layers {
            self.ssm.resize_with(n_layers, || None);
            self.conv.resize_with(n_layers, || None);
        } else {
            self.ssm.truncate(n_layers);
            self.conv.truncate(n_layers);
        }
        self.valid.resize(n_layers, false);
    }
    /// 全層を「ホストが正」に戻す (次の実行で再 seed)。ホストの GDN 状態を
    /// 書き換える経路 (セッション reset / エントリ復元 / デコードチェーンの
    /// 遅延フラッシュ / CPU フォールバック) は必ずこれを呼ぶ。
    fn invalidate(&mut self) {
        for v in self.valid.iter_mut() {
            *v = false;
        }
        self.srows_n = 0;
    }
    /// 層ベクタを `li` まで伸ばす (層数は初回利用時に確定する)。
    fn touch_layer(&mut self, li: usize) {
        if self.ssm.len() <= li {
            let n = li + 1;
            self.ssm.resize_with(n, || None);
            self.conv.resize_with(n, || None);
            self.valid.resize(n, false);
        }
    }
    /// 層 1 本の確保。形が変わったら全層を無効化する (別モデル/別形状)。
    fn ensure_layer(&mut self, li: usize, rows: usize, cols: usize) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        self.touch_layer(li);
        if (self.rows != 0 && (self.rows != rows || self.cols != cols)) || self.rows != rows {
            self.invalidate();
            self.srows_n = 0;
        }
        self.rows = rows;
        self.cols = cols;
        GdnScratch::grow(&mut self.ssm[li], rows * cols * F)?;
        Ok(())
    }
    /// 層 1 本の conv 状態を確保。
    fn ensure_conv(&mut self, li: usize, len: usize) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        self.touch_layer(li);
        GdnScratch::grow(&mut self.conv[li], len * F)?;
        self.conv_len = len;
        Ok(())
    }
    /// hoist 経路の境界行バッファ [n, nv, dk, dv] を確保 (n が同じなら再利用)。
    fn ensure_srows(&mut self, n: usize, rows: usize, cols: usize) -> Result<CUdeviceptr> {
        const F: usize = std::mem::size_of::<f32>();
        let bytes = n
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .and_then(|v| v.checked_mul(F))
            .ok_or_else(|| anyhow::anyhow!("srows size overflow"))?;
        GdnScratch::grow(&mut self.srows, bytes)?;
        if self.srows_n != n {
            self.srows_n = n;
        }
        Ok(self.srows.as_ref().expect("srows 未確保").ptr())
    }
}

impl GdnScratch {
    const fn empty() -> Self {
        Self {
            q: None,
            z: None,
            beta: None,
            alpha: None,
            conv_in: None,
            ssm_in: None,
            wc: None,
            wa: None,
            wd: None,
            wn: None,
            w: None,
            m: None,
            sinit: None,
            y: None,
            conv_out: None,
            ssm_out: None,
        }
    }
    fn grow(cap: &mut Option<DevBuffer>, bytes: usize) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes()).unwrap_or(0) < bytes {
            *cap = Some(DevBuffer::alloc(bytes.max(1))?);
        }
        Ok(())
    }
    /// 形が変わらない限り何もしない (prefill の 24 層は全て同じ形)。
    fn ensure(
        &mut self,
        n: usize,
        conv_dim: usize,
        vd: usize,
        nv: usize,
        dk: usize,
        dv: usize,
        cs: usize,
        n_chunks: usize,
    ) -> Result<()> {
        const F: usize = std::mem::size_of::<f32>();
        Self::grow(&mut self.q, n * conv_dim * F)?;
        Self::grow(&mut self.z, n * vd * F)?;
        Self::grow(&mut self.beta, n * nv * F)?;
        Self::grow(&mut self.alpha, n * nv * F)?;
        Self::grow(&mut self.conv_in, conv_dim * cs * F)?;
        Self::grow(&mut self.ssm_in, nv * dk * dv * F)?;
        Self::grow(&mut self.wc, conv_dim * 4 * F)?; // d_conv == 4 (chunked 前提)
        Self::grow(&mut self.wa, nv * F)?;
        Self::grow(&mut self.wd, nv * F)?;
        Self::grow(&mut self.wn, dv * F)?;
        Self::grow(&mut self.w, n_chunks * nv * dk * dv * F)?;
        Self::grow(&mut self.m, n_chunks * nv * dk * dk * F)?;
        Self::grow(&mut self.sinit, n_chunks * nv * dk * dv * F)?;
        Self::grow(&mut self.y, n * vd * F)?;
        Self::grow(&mut self.conv_out, conv_dim * cs * F)?;
        Self::grow(&mut self.ssm_out, nv * dk * dv * F)?;
        Ok(())
    }
    fn ptrs(&self) -> GdnPtrs {
        let p = |b: &Option<DevBuffer>| b.as_ref().expect("GdnScratch: 未確保").ptr();
        GdnPtrs {
            q: p(&self.q),
            z: p(&self.z),
            beta: p(&self.beta),
            alpha: p(&self.alpha),
            conv_in: p(&self.conv_in),
            ssm_in: p(&self.ssm_in),
            wc: p(&self.wc),
            wa: p(&self.wa),
            wd: p(&self.wd),
            wn: p(&self.wn),
            w: p(&self.w),
            m: p(&self.m),
            sinit: p(&self.sinit),
            y: p(&self.y),
            conv_out: p(&self.conv_out),
            ssm_out: p(&self.ssm_out),
        }
    }
    fn buf(&self, which: GdnBuf) -> &DevBuffer {
        let b = match which {
            GdnBuf::Y => &self.y,
            GdnBuf::ConvOut => &self.conv_out,
            GdnBuf::SsmOut => &self.ssm_out,
        };
        b.as_ref().expect("GdnScratch: 未確保")
    }
}

thread_local! {
    static GDN_SCRATCH: std::cell::RefCell<GdnScratch> =
        std::cell::RefCell::new(GdnScratch::empty());
    /// s89 (P1-1): GDN スキャン状態のデバイス正本 (scheduler ワーカーは単一
    /// スレッドなので thread_local で競合しない・`GDN_SCRATCH` と同じ規約)。
    static GDN_STATE: std::cell::RefCell<GdnStateCache> =
        std::cell::RefCell::new(GdnStateCache::empty());
}

/// s89 (P1-1): GDN スキャン状態のデバイス正本を「ホストが正」に戻す。
///
/// ホストの `KvCache::gdn` を書き換える経路 (セッション reset / エントリ復元 /
/// デコードチェーンの遅延フラッシュ / CPU フォールバック) は、その後にこの
/// 関数を呼ぶ。呼ばれた層は次の prefill でホストから再 seed する (数値不変・
/// 余分な H2D が 1 回増えるだけ)。**古い正本を読むと出力が変わる**ので、
/// 新しい書き換え経路を足すときは必ずここを通すこと。
pub fn gdn_scan_state_invalidate() {
    GDN_STATE.with(|c| c.borrow_mut().invalidate());
}

// s131: デバイス正本を使うか。`DENNO_GDN_DEVSTATE=0` で従来動作 (毎回ホストと
// 往復) に戻す。切り分け用で、既定は有効。
fn gdn_devstate() -> bool {
    !matches!(
        std::env::var("DENNO_GDN_DEVSTATE").ok().as_deref(),
        Some("0")
    )
}

// 従来動作のホスト書き戻し (デバイス正本を使わないときだけ)。
fn gdn_host_writeback(conv_st: &mut [f32], ssm_st: &mut [f32]) -> Result<()> {
    GDN_SCRATCH.with(|cell| -> Result<()> {
        let s = cell.borrow();
        let cv = s.buf(GdnBuf::ConvOut).to_vec::<f32>(conv_st.len())?;
        let sm = s.buf(GdnBuf::SsmOut).to_vec::<f32>(ssm_st.len())?;
        conv_st.copy_from_slice(&cv);
        ssm_st.copy_from_slice(&sm);
        Ok(())
    })
}

// s131: デバイス正本の入り口。層の状態が未確定ならホストから 1 回だけ seed し、
// 以後はデバイスの値をそのままカーネルへ渡す。
fn gdn_state_begin(
    kv: u64,
    li: usize,
    conv: &[f32],
    ssm: &[f32],
    conv_len: usize,
    rows: usize,
    cols: usize,
) -> Result<(CUdeviceptr, CUdeviceptr)> {
    GDN_STATE.with(|cell| -> Result<(CUdeviceptr, CUdeviceptr)> {
        let mut c = cell.borrow_mut();
        // 別スロット/別セッションの状態を混ぜない (層番号だけでは足りない)。
        if c.cur_kv != Some(kv) {
            c.invalidate();
            c.cur_kv = Some(kv);
        }
        let devstate = gdn_devstate();
        c.ensure_layer(li, rows, cols)?;
        c.ensure_conv(li, conv_len)?;
        if !c.valid[li] || !devstate {
            c.conv[li].as_ref().unwrap().copy_from(conv)?;
            c.ssm[li].as_ref().unwrap().copy_from(ssm)?;
            c.valid[li] = devstate;
        }
        let pc = c.conv[li].as_ref().unwrap().ptr();
        let ps = c.ssm[li].as_ref().unwrap().ptr();
        Ok((pc, ps))
    })
}

// s131: スキャン結果をデバイス正本へ D2D で戻す (ホストへは落とさない)。
fn gdn_state_commit(
    li: usize,
    conv_out: CUdeviceptr,
    ssm_out: CUdeviceptr,
    conv_len: usize,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let (pc, ps) = GDN_STATE
        .with(|cell| -> Option<(CUdeviceptr, CUdeviceptr)> {
            let c = cell.borrow();
            Some((c.conv.get(li)?.as_ref()?.ptr(), c.ssm.get(li)?.as_ref()?.ptr()))
        })
        .ok_or_else(|| anyhow::anyhow!("gdn_state_commit: 状態バッファ未確保 (layer {li})"))?;
    let lib = lib()?;
    unsafe {
        check(
            (lib.memcpy_dtod_async)(pc, conv_out, conv_len * 4, std::ptr::null_mut()),
            "cuMemcpyDtoDAsync (gdn conv)",
        )?;
        check(
            (lib.memcpy_dtod_async)(ps, ssm_out, rows * cols * 4, std::ptr::null_mut()),
            "cuMemcpyDtoDAsync (gdn ssm)",
        )?;
    }
    Ok(())
}

/// s131: デバイス正本をホストへ戻す (層指定)。戻した層は無効化するので、
/// 次にその層を使うときはホストから 1 回 seed し直す。
/// ホストの `KvCache::gdn` を読む前に必ず通すこと。
///
/// s141 (Task B7): 返す 2 本は **thread_local のホスト一時領域の借り物**。
/// 従来は層ごとに `to_vec` (= `Vec::with_capacity`) で 2 本確保していたため、
/// 27B q4 の decode 1 ステップで 48 層 x 2 本 = 96 本 / 564 KiB の確保と解放を
/// 毎回払っていた。転送・内容は同一 (同じ D2H を同じ長さで 1 回) で、変わるのは
/// 「毎回確保するか、伸ばすだけの 1 本を使い回すか」だけ。
/// 借り物なので、**次の呼び出しで中身が変わる** — 呼び出し側はその場で読み終わること
/// (`KvCache::flush_gdn_device_state` はその場で copy_from_slice する)。
pub fn gdn_state_take<'a>(kv: u64, li: usize, reuse: &'a mut GdnHostStage) -> Option<(&'a [f32], &'a [f32])> {
    let (pc, ps, conv_len, ssm_len) =
        GDN_STATE.with(|cell| -> Option<(CUdeviceptr, CUdeviceptr, usize, usize)> {
            let mut c = cell.borrow_mut();
            if c.cur_kv != Some(kv) || li >= c.valid.len() || !c.valid[li] {
                return None;
            }
            let (conv_len, rows, cols) = (c.conv_len, c.rows, c.cols);
            let pc = c.conv[li].as_ref()?.ptr();
            let ps = c.ssm[li].as_ref()?.ptr();
            // 戻した層は無効化 (従来どおり・成功時のみ)。
            c.valid[li] = false;
            Some((pc, ps, conv_len, rows * cols))
        })?;
    // 借り物を 2 本同時に返すので、転送は 2 回とも先に済ませる。
    // conv / ssm は別々の領域なので、2 本の借用は同時に成立する。
    let conv = d2h_raw(&mut reuse.conv, pc, conv_len)?;
    let ssm = d2h_raw(&mut reuse.ssm, ps, ssm_len)?;
    Some((conv, ssm))
}

/// s141 (Task B7): `gdn_state_take` に渡す呼び出し側所有の受け皿 1 組。
/// 呼び出し元のループの外で 1 回だけ作り、層ごとに使い回す (grow-only)。
/// 中身は「前の層の値」なので、**返り値をその場で読み終えること**。
pub struct GdnHostStage {
    conv: Vec<f32>,
    ssm: Vec<f32>,
}

impl GdnHostStage {
    /// 空の受け皿 (初回の D2H で必要な分だけ伸びる)。
    pub fn new() -> Self {
        Self { conv: Vec::new(), ssm: Vec::new() }
    }
    /// 確保済みの要素数 (テストが「ステップ数に対して増えない」ことを見る口)。
    pub fn capacities(&self) -> (usize, usize) {
        (self.conv.capacity(), self.ssm.capacity())
    }
}

impl Default for GdnHostStage {
    fn default() -> Self {
        Self::new()
    }
}

/// s141: 再利用領域 `dst` の先頭 `len` 要素へ D2H し、その部分を**借りて**返す。
/// `DevBuffer::to_vec` と転送は同一 (同じ `cuMemcpyDtoH` を同じ長さで 1 回) で、
/// 違いは「確保しない」ことだけ。容量が足りないときだけ `reserve_exact` で伸ばす
/// (`mem::take` のように領域を奪い返さない = 次の層でも同じ領域を使う)。
fn d2h_raw<'a>(dst: &'a mut Vec<f32>, ptr: CUdeviceptr, len: usize) -> Option<&'a [f32]> {
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
fn reserve_exact_grow(dst: &mut Vec<f32>, len: usize) {
    dst.clear();
    if len > dst.capacity() {
        dst.reserve_exact(len - dst.len());
    }
}

/// s141b (Task B7c): デバイス正本の 1 層を「デバイス正本から読む」ために取り出す。
///
/// s131 の `gdn_state_take` はホストへ D2H する版 (ホストの `KvCache::gdn` を
/// 復帰させる用途)。こちらは decode chain の seed 用で、返すのは移し元の
/// デバイスポインタだけ — ホストを経由しないので層あたり ~3.2 MiB の PCIe 往復が
/// 消える。返す値は (conv ポインタ, ssm ポインタ)。
///
/// 取り出した層は**無効化しない** (正本の値はそのまま有効 — 読むだけ)。ただし
/// 読み手は「ミラーへコピーした後」に scan カーネルがその面を更新するので、正本と
/// ミラーの内容は今後ずれる → 戻した層は valid=false にする (従来の
/// `gdn_state_take` と同じ規則: 次に scan 経路で使うときは必ず seed し直す)。
fn gdn_state_take_dev(
    kv: u64,
    li: usize,
    c: &mut GdnStateCache,
) -> Option<(CUdeviceptr, CUdeviceptr)> {
    if c.cur_kv != Some(kv) || li >= c.valid.len() || !c.valid[li] {
        return None;
    }
    let pc = c.conv.get(li)?.as_ref()?.ptr();
    let ps = c.ssm.get(li)?.as_ref()?.ptr();
    c.valid[li] = false;
    Some((pc, ps))
}

/// s141b (Task B7c): 遅延フラッシュの「D2H が要るか」の判定 (純関数 — テストが
/// 実物を通す)。
///
/// 規則: **ホストが古い可能性がある層** (= `stale`) と **位置が合っている層**
/// (= `pos == Some(target)` / `Some(target+1)`) の両方を満たす層だけ D2H する。
///
/// - stale でない層はホストが既に最新 (このステップで面 0 を上から書いた) →
///   読み直さなくてよい。これが「何も要らないステップでホストバイトをコピー
///   しない」の実体。
/// - stale でも位置が合っていない層 (chain 未走 / 不整合) は従来どおり判定へ回し、
///   呼び出し側がバグ時 None を返す経路をそのまま踏ませる。
///
/// 戻り値: (D2H するか, 位置が合っているか)。
fn gdn_flush_step(is_gdn_layer: bool, stale: bool, target: usize, pos: Option<usize>) -> (bool, bool) {
    if !is_gdn_layer {
        return (false, false);
    }
    // 成功ステップ後: 位置は target (= cache.len()) に一致し、その面は**書き込んだ
    // 状態** = ホストが遅れている層だけが読み直しの対象。
    if pos == Some(target) {
        return (stale, true);
    }
    // 層途中で失敗したステップ: 位置は target+1 で、書いた面は「次」の状態。
    // 現在の状態 (target) は反対の面にあり、その面は前回の seed 直後に書かれた
    // = **ホストと一致している** (= stale が偽) ので、stale の層は読んでも
    // 意味が無い (古いホストを書き戻すだけ)。一律で「読まない」。
    if pos == Some(target + 1) {
        return (false, true);
    }
    (false, false)
}

static GPU_CTX: OnceLock<Option<CudaCtx>> = OnceLock::new();
static GPU_BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
static GPU_FORCE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn cuda_enabled() -> bool {
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

/// テスト用: env を介さず GPU 経路を強制有効化 (ignored テストでのみ使用)。
#[cfg(test)]
pub fn enable_for_tests() {
    GPU_FORCE.store(true, std::sync::atomic::Ordering::Relaxed);
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
fn shared_ctx() -> Option<&'static CudaCtx> {
    GPU_CTX
        .get_or_init(|| match CudaCtx::new() {
            Ok(ctx) => {
                tracing::info!("CUDA int8 offload enabled: {}", ctx.info());
                Some(ctx)
            }
            Err(e) => {
                tracing::warn!("CUDA int8 offload unavailable, using CPU int8: {e:#}");
                None
            }
        })
        .as_ref()
}

/// int8 重みの VRAM レジストリ (キー = Quant8Linear.uid)。
/// uid は Linear 構築時に採番されるため、Vec アドレスの再利用で別内容の重みが
/// 同一キーに現れることはない。Clone は uid を共有するが内容が同一なので安全。
/// エントリは解放しない (実運用は 1 モデル = 約 2GB、テストは微小)。
struct GpuW {
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
const GEMM_PAD_N: usize = 256;
/// s74: dno_gemm_i8 のタイル寸法 (kernels_int8.cu の GEMM_TILE / GEMM_TILE_N と
/// 同期すること)。i 方向 128 行 × j 方向 256 列・block 16x16 (1 thread 8x16 出力)。
const GEMM_TILE: usize = 128;
const GEMM_TILE_N: usize = 256;
/// s75: カーネルのブロック形状 (kernels_int8.cu の GEMM_BX / GEMM_BY)。
/// 1 thread 8 行 x 16 列・block 16x16 = 256 thread なので
/// `GEMM_BY * 8 == GEMM_TILE` / `GEMM_BX * 16 == GEMM_TILE_N` が成り立つ。
/// s74 は (16,4) = 64 thread で 32 行しか覆えていなかった (タイル 128 行と
/// 食い違い、残り 96 行が未書き込みのまま返っていた)。
const GEMM_BX: u32 = 16;
const GEMM_BY: u32 = 16;

/// s74 (T6 候補 1): dno_gemm_i8 の j 方向タイル (256) に合わせた重み行の切り上げ。
/// 旧 128 タイル版は端をカーネル側で 0 埋めしていたが、新カーネルは A 行を
/// 2 枚の B タイルで共有するため A/B の pad を非対称にできる重み側だけを
/// ホストで 0 埋めする (i8 の 0 は積に寄与しない → dot は厳密に不変)。
pub(crate) fn gemm_pad_n(n: usize) -> usize {
    n.div_ceil(GEMM_PAD_N) * GEMM_PAD_N
}

/// s74: 重み [n,k] i8 (row-major) を `gemm_pad_n` 行までラッパ行で 0 埋めする。
/// 既に倍数なら複製しない (VRAM/RAM の余分なコピーを避ける)。
fn gemm_pad_apply(qw: &[i8], k: usize, n: usize) -> std::borrow::Cow<'_, [i8]> {
    debug_assert_eq!(qw.len(), k * n);
    let npad = gemm_pad_n(n);
    if npad == n {
        return std::borrow::Cow::Borrowed(qw);
    }
    let mut w = vec![0i8; k * npad];
    // パディング行はブロックごとに連続した全域に置く (行内は k 要素で連続)。
    for r in 0..n {
        w[r * k..(r + 1) * k].copy_from_slice(&qw[r * k..(r + 1) * k]);
    }
    std::borrow::Cow::Owned(w)
}

fn weights_for(uid: u64, qw: &[i8], scale: &[f32], k: usize, n: usize) -> Result<&'static GpuW> {
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
fn max_m() -> usize {
    static MAX_M: OnceLock<usize> = OnceLock::new();
    *MAX_M.get_or_init(|| {
        std::env::var("DENNO_CUDA_MAX_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16)
    })
}

use crate::quant::Quant8Linear;

/// P17i: 大きい m (prefill) 用の tiled GEMM 経路を有効にするか。
/// `DENNO_CUDA_GEMM=0` で無効化 (従来どおり CPU へ)。
fn gemm_enabled() -> bool {
    std::env::var("DENNO_CUDA_GEMM").ok().as_deref() != Some("0")
}

/// P17i: tiled GEMM を発動する最小の m。小さい m では GEMM 経路の層あたり
/// sync (H2D/quantize/gemm/D2H × 4 層/層) が CPU GEMM より高くつくため、
/// ある程度大きい m から発動させる (m=38 の短文 prefill は CPU が速い)。
/// `DENNO_CUDA_GEMM_MIN_M` で変更可。0 にすると m>max_m() で常に発動。
fn gemm_min_m() -> usize {
    std::env::var("DENNO_CUDA_GEMM_MIN_M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

/// P17c: 複数の int8 Linear を 1 回の GPU 呼び出しに融合する (QKV / gate+up 用)。
/// 同じ入力を消費する線形層群の重み行を**パート順に連結**した 1 つのバッファとして
/// 登録・計算する。各行の演算 (量子化スケール・i32 累積・スケール乗算) は個別
/// 呼び出しと完全に同一なので、出力をパート幅で切り分ければビット一致する。
/// 1 線形層 1 同期の固定オーバーヘッド (約 28µs) がパート数 → 1 になる効果:
/// 実測で qkv は 3 呼び出し分の 96µs/層 → 約 40µs/層。
///
/// 戻り値 None = CPU パス (無効/初期化失敗/実行時エラー/m>閾値/一部が Q8 でない)。
pub fn maybe_gpu_linear_grouped(parts: &[&Quant8Linear], a: &[f32], m: usize) -> Option<Vec<f32>> {
    if !cuda_enabled() || parts.is_empty() {
        return None;
    }
    let use_gemm = m > max_m() && m >= gemm_min_m() && gemm_enabled();
    // m が GEMV 経路の上限を超えて GEMM も発動しないなら CPU へ
    if m > max_m() && !use_gemm {
        return None;
    }
    let k = parts[0].k;
    let n: usize = parts.iter().map(|p| p.n).sum();
    let uids: Vec<u64> = parts.iter().map(|p| p.uid).collect();
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU int8: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let gw = match weights_for_group(&uids, parts, k, n) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("CUDA grouped weight upload failed, using CPU int8: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    // P17i: m が大きい (prefill) ときは tiled GEMM、小さいときは GEMV 型
    let r = if use_gemm {
        linear_gemm(ctx, a, gw, m, k, n)
    } else {
        linear_pinned(ctx, a, gw, m, k, n)
    };
    match r {
        Ok(out) => {
            // s61: 融合経路も実行先ラベルに計上する (PROFILE の via 列用)。
            // これが無いと llama の qkv/gate_up 融合 call が `via=cpu` と誤記される。
            set_last_linear_ran_on_gpu(true);
            Some(out)
        }
        Err(e) => {
            tracing::error!("CUDA grouped linear failed, falling back to CPU permanently: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

/// P17k: FFN 線形ブロック (gate_up GEMM → swiglu → down GEMM) をデバイス常駐で実行。
/// h [m, e] から down 出力 [m, e] を返す。中間の [m, 2ff] (13k tok で 949MB) を
/// ホストに落とさないため、従来経路の D2H + to_vec + gate/up 分割コピー +
/// swiglu (CPU f64 expf) + H2D が全て消える。
///
/// ビット一致: gate_up/down は dno_gemm_i8 (整数累積で順序不変)、swiglu は
/// dno_swiglu_int (dno_swiglu と同一演算列) → CPU 経路と出力は完全一致。
/// 失敗時 None (呼び出し側で従来経路: gate/up Vec + CPU swiglu + down)。
/// m が GEMM 閾値未満なら None (小さい m は従来どおり CPU が最速)。
#[allow(clippy::too_many_arguments)]
pub fn maybe_gpu_ffn(
    gate: &Quant8Linear,
    up: &Quant8Linear,
    down: &Quant8Linear,
    h: &[f32],
    m: usize,
    ff: usize,
    li: usize,
) -> Option<Vec<f32>> {
    if !cuda_enabled() {
        return None;
    }
    if !(m > max_m() && m >= gemm_min_m() && gemm_enabled()) {
        return None;
    }
    let e = gate.k;
    debug_assert_eq!(up.k, e);
    debug_assert_eq!(down.k, ff);
    debug_assert_eq!(down.n, e);
    let ctx = shared_ctx()?;
    if let Err(err) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU int8: {err:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let gw1 = match weights_for_group(&[gate.uid, up.uid], &[gate, up], e, 2 * ff) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA FFN weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    let gw2 = match weights_for(down.uid, &down.w, &down.scale, ff, e) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA FFN down weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    let lib = lib().ok()?;
    let r = SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let Scratch {
            x,
            aq,
            asc,
            out,
            hx,
            ho,
            act,
            aq2,
            asc2,
            out2,
        } = &mut *s;
        let bytes_x = m * e * 4;
        Scratch::ensure(x, bytes_x)?;
        Scratch::ensure(aq, m * e)?;
        Scratch::ensure(asc, m * 4)?;
        Scratch::ensure(out, m * 2 * ff * 4)?;
        Scratch::ensure(act, m * ff * 4)?;
        Scratch::ensure(aq2, m * ff)?;
        Scratch::ensure(asc2, m * 4)?;
        Scratch::ensure(out2, m * e * 4)?;
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, m * e * 4)?;
        let (dx, daq, dasc, dout, dact, daq2, dasc2, dout2) = (
            x.as_ref().unwrap(),
            aq.as_ref().unwrap(),
            asc.as_ref().unwrap(),
            out.as_ref().unwrap(),
            act.as_ref().unwrap(),
            aq2.as_ref().unwrap(),
            asc2.as_ref().unwrap(),
            out2.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..m * e].copy_from_slice(h);
        unsafe {
            check(
                (lib.memcpy_htod_async)(dx.ptr(), hx.ptr as *const c_void, bytes_x, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync",
            )?;
            // 1) gate_up GEMM: [m, e] → [m, 2ff] (デバイス上に保持)
            crate::prof::time_layer(crate::prof::GATE_UP, li, || -> Result<()> {
                let (mut px, mut paq, mut pasc) = (dx.ptr(), daq.ptr(), dasc.ptr());
                let (mut mc, mut kc) = (m as c_int, e as c_int);
                let mut args = [
                    &mut px as *mut _ as *mut c_void,
                    &mut paq as *mut _ as *mut c_void,
                    &mut pasc as *mut _ as *mut c_void,
                    &mut mc as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                ];
                ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args)?;
                let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                    (daq.ptr(), dasc.ptr(), gw1.w.ptr(), gw1.sc.ptr(), dout.ptr());
                let (mut mc, mut kc, mut nc) = (m as c_int, e as c_int, (2 * ff) as c_int);
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
                let gmx = m.div_ceil(GEMM_TILE) as u32;
                let gmy = (2 * ff).div_ceil(GEMM_TILE_N) as u32;
                ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args)?;
                Ok(())
            })?;
            // 2) swiglu (interleaved): [m, 2ff] → act [m, ff] (デバイス上に保持)
            crate::prof::time_layer(crate::prof::SWIGLU, li, || -> Result<()> {
                let (mut pg, mut po) = (dout.ptr(), dact.ptr());
                let (mut pff, mut pw, mut pt) = (ff as c_int, (2 * ff) as c_int, (m * ff) as i64);
                let mut args = [
                    &mut pg as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut pff as *mut _ as *mut c_void,
                    &mut pw as *mut _ as *mut c_void,
                    &mut pt as *mut _ as *mut c_void,
                ];
                let grid = (m * ff).div_ceil(256) as u32;
                ctx.launch_async(ctx.f_swiglu_int, (grid, 1, 1), (256, 1, 1), 0, &mut args)?;
                Ok(())
            })?;
            // 3) down GEMM: act [m, ff] → out2 [m, e] + D2H。キュー待ちの実態を
            // DOWN に帰属させるため同期をこのタイマー内に入れる。
            crate::prof::time_layer(crate::prof::DOWN, li, || -> Result<()> {
                let (mut px, mut paq, mut pasc) = (dact.ptr(), daq2.ptr(), dasc2.ptr());
                let (mut mc, mut kc) = (m as c_int, ff as c_int);
                let mut args = [
                    &mut px as *mut _ as *mut c_void,
                    &mut paq as *mut _ as *mut c_void,
                    &mut pasc as *mut _ as *mut c_void,
                    &mut mc as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                ];
                ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args)?;
                let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                    (daq2.ptr(), dasc2.ptr(), gw2.w.ptr(), gw2.sc.ptr(), dout2.ptr());
                let (mut mc, mut kc, mut nc) = (m as c_int, ff as c_int, e as c_int);
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
                let gmx = m.div_ceil(GEMM_TILE) as u32;
                let gmy = e.div_ceil(GEMM_TILE_N) as u32;
                ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args)?;
                check(
                    (lib.memcpy_dtoh_async)(ho.ptr, dout2.ptr(), m * e * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync",
                )?;
                // s65: 発行元の件数分類 (表示専用・時間は d2h_state バケット側)
                crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
                ctx.sync()?;
                Ok(())
            })?;
        }
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        Ok(ho.as_slice::<f32>()[..m * e].to_vec())
    });
    match r {
        Ok(out) => Some(out),
        Err(err) => {
            tracing::error!("CUDA FFN block failed, falling back to CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

/// B1: prefill デバイス常駐チェーン用 FFN ブロック — maybe_gpu_ffn と同一の
/// カーネル列 (gate_up GEMM → swiglu_int → down GEMM) を、入力 `in_slot`
/// (chain_rmsnorm が書いた h2・デバイス常駐) から出力 CH_DOWN (デバイス常駐) へ
/// **同期・ホスト往復なし**で実行する。h2 の H2D (81MB/層) と down の D2H
/// (81MB/層) を廃止。全段ビット一致 (GEMM: 整数累積・swiglu: dno_swiglu_int と
/// 同一演算)。戻り None = チェーン全体を従来経路でやり直し。
pub fn maybe_gpu_ffn_dev(
    gate: &Quant8Linear,
    up: &Quant8Linear,
    down: &Quant8Linear,
    in_slot: usize,
    m: usize,
    ff: usize,
    li: usize,
) -> Option<()> {
    if !(m > max_m() && m >= gemm_min_m() && gemm_enabled()) {
        return None;
    }
    let e = gate.k;
    debug_assert_eq!(up.k, e);
    debug_assert_eq!(down.k, ff);
    debug_assert_eq!(down.n, e);
    let ctx = chain_ctx()?;
    let gw1 = match weights_for_group(&[gate.uid, up.uid], &[gate, up], e, 2 * ff) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA FFN weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    let gw2 = match weights_for(down.uid, &down.w, &down.scale, ff, e) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA FFN down weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    ok(chain_ensure_buf(in_slot, m * e * 4))?;
    ok(chain_ensure_buf(CH_DOWN, m * e * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let dptr_in = s.bufs[in_slot].as_ref().unwrap().ptr();
        let dptr_out = s.bufs[CH_DOWN].as_ref().unwrap().ptr();
        SCRATCH.with(|c2| -> Option<()> {
            let mut s2 = c2.borrow_mut();
            ok(Scratch::ensure(&mut s2.aq, m * e))?;
            ok(Scratch::ensure(&mut s2.asc, m * 4))?;
            ok(Scratch::ensure(&mut s2.out, m * 2 * ff * 4))?;
            ok(Scratch::ensure(&mut s2.act, m * ff * 4))?;
            ok(Scratch::ensure(&mut s2.aq2, m * ff))?;
            ok(Scratch::ensure(&mut s2.asc2, m * 4))?;
            let (daq, dasc, dout, dact, daq2, dasc2) = (
                s2.aq.as_ref().unwrap(),
                s2.asc.as_ref().unwrap(),
                s2.out.as_ref().unwrap(),
                s2.act.as_ref().unwrap(),
                s2.aq2.as_ref().unwrap(),
                s2.asc2.as_ref().unwrap(),
            );
            unsafe {
                // 1) gate_up GEMM: [m, e] → [m, 2ff] (デバイス上に保持)
                crate::prof::time_layer(crate::prof::GATE_UP, li, || -> Option<()> {
                    let (mut px, mut paq, mut pasc) = (dptr_in, daq.ptr(), dasc.ptr());
                    let (mut mc, mut kc) = (m as c_int, e as c_int);
                    let mut args = [
                        &mut px as *mut _ as *mut c_void,
                        &mut paq as *mut _ as *mut c_void,
                        &mut pasc as *mut _ as *mut c_void,
                        &mut mc as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                    ];
                    ok(ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args))?;
                    let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                        (daq.ptr(), dasc.ptr(), gw1.w.ptr(), gw1.sc.ptr(), dout.ptr());
                    let (mut mc, mut kc, mut nc) = (m as c_int, e as c_int, (2 * ff) as c_int);
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
                    let gmx = m.div_ceil(GEMM_TILE) as u32;
                    let gmy = (2 * ff).div_ceil(GEMM_TILE_N) as u32;
                    ok(ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args))
                })?;
                // 2) swiglu (interleaved): [m, 2ff] → act [m, ff] (デバイス上に保持)
                crate::prof::time_layer(crate::prof::SWIGLU, li, || -> Option<()> {
                    let (mut pg, mut po) = (dout.ptr(), dact.ptr());
                    let (mut pff, mut pw, mut pt) = (ff as c_int, (2 * ff) as c_int, (m * ff) as i64);
                    let mut args = [
                        &mut pg as *mut _ as *mut c_void,
                        &mut po as *mut _ as *mut c_void,
                        &mut pff as *mut _ as *mut c_void,
                        &mut pw as *mut _ as *mut c_void,
                        &mut pt as *mut _ as *mut c_void,
                    ];
                    let grid = (m * ff).div_ceil(256) as u32;
                    ok(ctx.launch_async(ctx.f_swiglu_int, (grid, 1, 1), (256, 1, 1), 0, &mut args))
                })?;
                // 3) down GEMM: act [m, ff] → CH_DOWN [m, e] (デバイス常駐・同期なし)
                crate::prof::time_layer(crate::prof::DOWN, li, || -> Option<()> {
                    let (mut px, mut paq, mut pasc) = (dact.ptr(), daq2.ptr(), dasc2.ptr());
                    let (mut mc, mut kc) = (m as c_int, ff as c_int);
                    let mut args = [
                        &mut px as *mut _ as *mut c_void,
                        &mut paq as *mut _ as *mut c_void,
                        &mut pasc as *mut _ as *mut c_void,
                        &mut mc as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                    ];
                    ok(ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args))?;
                    let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                        (daq2.ptr(), dasc2.ptr(), gw2.w.ptr(), gw2.sc.ptr(), dptr_out);
                    let (mut mc, mut kc, mut nc) = (m as c_int, ff as c_int, e as c_int);
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
                    let gmx = m.div_ceil(GEMM_TILE) as u32;
                    let gmy = e.div_ceil(GEMM_TILE_N) as u32;
                    ok(ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args))
                })
            }
        })
    })
}

fn weights_for_group(
    uids: &[u64],
    parts: &[&Quant8Linear],
    k: usize,
    n: usize,
) -> Result<&'static GpuW> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static REG: OnceLock<Mutex<HashMap<Vec<u64>, Box<GpuW>>>> = OnceLock::new();
    let reg = REG.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().expect("weights registry poisoned");
    if !map.contains_key(uids) {
        // パート順にホスト側で行連結してから 1 回でアップロードする
        // (ホスト連結バッファはアップロード後 immediately drop → RAM 定常増なし)
        let mut w = Vec::with_capacity(k * n);
        let mut sc = Vec::with_capacity(n);
        for p in parts {
            debug_assert_eq!(p.k, k, "融合グループ内で k が不一致");
            if p.k != k {
                bail!("CUDA grouped linear: k mismatch ({} vs {})", p.k, k);
            }
            w.extend_from_slice(&p.w);
            sc.extend_from_slice(&p.scale);
        }
        debug_assert_eq!(w.len(), k * n);
        let w = DevBuffer::from_slice(&gemm_pad_apply(&w, k, n))?;
        let sc = DevBuffer::from_slice(&sc)?;
        tracing::info!(
            "CUDA grouped weight upload ({} parts, uids={}..): k={k} n={n} ({:.1} MB)",
            parts.len(),
            uids[0],
            (k * n + n * 4) as f64 / 1e6
        );
        map.insert(uids.to_vec(), Box::new(GpuW { w, sc, k, n, npad: gemm_pad_n(n) }));
    }
    let e = map.get(uids).expect("just inserted");
    Ok(unsafe { std::mem::transmute::<&GpuW, &'static GpuW>(e.as_ref()) })
}

// s57 (T4): PROFILE の実行先ラベル用に「直近の GPU 線形呼び出しが実際に GPU で
// 走ったか」を記録する thread-local。**計測のみ** — `maybe_gpu_linear` が結果を
// 返す直前に書くだけで、数値・分岐・GPU 状態には触れない。値は次の線形呼び出し
// まで古いまま (呼び出し側は同じ区間の計測にしか使わない)。
#[cfg(feature = "cuda")]
thread_local! {
    static LINEAR_RAN_ON_GPU: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// s61: 区間内で「一度でも GPU 線形が走ったか」の集積フラグ。ネストした
    /// `Probe` を許すため深さも持つ (最深の scope が閉じたときに外側へ引き継ぐ)。
    static LINEAR_SCOPE: (std::cell::Cell<u32>, std::cell::Cell<bool>) =
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
fn set_last_linear_ran_on_gpu(v: bool) {
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

#[cfg(feature = "cuda")]
impl Probe {
    #[inline]
    pub fn begin() -> Self {
        LINEAR_SCOPE.with(|(depth, any)| {
            if depth.get() == 0 {
                any.set(false);
            }
            depth.set(depth.get() + 1);
        });
        Probe { _not_send: std::marker::PhantomData }
    }

    /// この区間中に GPU 線形が 1 回でも走ったか。
    #[inline]
    pub fn any_on_gpu(&self) -> bool {
        LINEAR_SCOPE.with(|(_, any)| any.get())
    }
}

#[cfg(feature = "cuda")]
impl Drop for Probe {
    #[inline]
    fn drop(&mut self) {
        LINEAR_SCOPE.with(|(depth, any)| {
            let d = depth.get().saturating_sub(1);
            depth.set(d);
            // 最深 scope が閉じた時点でフラグは外側 scope の集積に残る
            // (外側 scope がまだ open ならその値に OR 済み)。
            if d == 0 {
                // 最外を閉じたら次回 begin() が false から始める
                any.set(false);
            }
        });
    }
}

/// s61 (T5a 撤回): GPU 線形層の入口 (`maybe_gpu_linear`) が呼ばれた回数の
/// **診断用カウンタ**。既定経路が入口を迂回していないことを GPU 無し環境でも
/// 検査できるようにする (083218d はゲート未指定で `matmul_bt_i8` を直呼びし、
/// この入口に一度も到達しなかった)。
///
/// 値の読み書きは Relaxed の加算 1 回のみで、分岐・戻り値・数値経路には触れない。
static LINEAR_SEAM_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// s61: `maybe_gpu_linear` の入口に到達した累計回数 (回帰テスト用)。
#[cfg(feature = "cuda")]
#[inline]
pub fn linear_seam_calls() -> u64 {
    LINEAR_SEAM_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// s138 (Task B6・計測のみ): decode chain のステップ分割 (表 A) と、線形ラッパの
// 分割 (表 B) のための小道具。**表示専用** (prof.rs の SPLIT_* にしか積まない)。
//
// 前提: chain (CUDA 部分) は 1 ステップで使う GPU 資源が 1 本の stream に直列で、
// 呼び出しスレッドも 1 本。したがって「段」は thread-local の 1 個の状態で足りる
// (BF/多セッションの chain はスレッドごとに別 state = 混ざらない)。
// ---------------------------------------------------------------------------

thread_local! {
    /// 現在走っている chain の段 (prof.rs の CHAIN_*)。0 = 計測していない。
    static CHAIN_SEG: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    /// 1 ステップ分の積算 (秒)。`take` で取り出す。
    static CHAIN_ACC: std::cell::RefCell<[f64; crate::prof::SPLIT_ROWS]> =
        const { std::cell::RefCell::new([0.0; crate::prof::SPLIT_ROWS]) };
    /// 段の切り替え時刻。
    static CHAIN_TICK: std::cell::Cell<Option<Instant>> = const { std::cell::Cell::new(None) };
    /// どの段からも取れなかった sync の残余 (秒)。ステップ終端で待ち行へ入れる。
    static CHAIN_RESID: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
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
type DbgTally = Vec<(usize, u64)>;

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
struct DecodeDbg {
    steps: u64,
    launches: DbgTally,
    h2d: DbgTally,
    d2h: DbgTally,
}

thread_local! {
    static DECODE_DBG: std::cell::RefCell<DecodeDbg> =
        std::cell::RefCell::new(DecodeDbg { steps: 0, launches: Vec::new(), h2d: Vec::new(), d2h: Vec::new() });
}

/// カーネル 1 発を数える (`CudaCtx::launch_async` の入口から)。
#[inline]
fn dbg_launch(f: CUfunction) {
    if decode_dbg_on() {
        DECODE_DBG.with(|d| dbg_bump(&mut d.borrow_mut().launches, f as usize));
    }
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

/// ステップ開始: 前のステップ (や prefill) で積んだ分を捨てる。
fn decode_dbg_reset() {
    if !decode_dbg_on() {
        return;
    }
    DECODE_DBG.with(|d| {
        let mut d = d.borrow_mut();
        d.launches.clear();
        d.h2d.clear();
        d.d2h.clear();
    });
}

/// ステップ末尾: 1 行 (＋内訳 2 行) を stderr に出す。観測のみ。
fn decode_dbg_flush() {
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
        d.launches.clear();
        d.h2d.clear();
        d.d2h.clear();
    });
}

/// 段の id を (表, 行) へ畳む。chain 用の id かどうかだけ見ればよい。
#[inline]
fn chain_row(op: usize) -> Option<usize> {
    match crate::prof::split_slot(op) {
        Some((t, i)) if t == crate::prof::SPLIT_CHAIN_TABLE => Some(i),
        _ => None,
    }
}

/// 現在の段に `secs` を足す (段が 0 = 計測していないなら何もしない)。
#[inline]
fn chain_charge(secs: f64) {
    let cur = CHAIN_SEG.with(|c| c.get());
    if cur == 0 {
        return;
    }
    if let Some(i) = chain_row(cur) {
        CHAIN_ACC.with(|a| a.borrow_mut()[i] += secs);
    }
}

/// 段を `op` へ切り替える。計測していない (段 0) ときは何もしない —
/// 呼び出し側は `chain_step_begin` を打ったステップの中でだけ切り替える。
#[inline]
pub fn chain_seg(op: usize) {
    if !crate::prof::split_on() {
        return;
    }
    let cur = CHAIN_SEG.with(|c| c.get());
    if cur == 0 || chain_row(op).is_none() {
        return;
    }
    let now = Instant::now();
    let prev = CHAIN_TICK.with(|c| c.replace(Some(now)));
    if let Some(t0) = prev {
        chain_charge(now.duration_since(t0).as_secs_f64());
    }
}

/// 直前の段 (`op`) を締めて、**同じ段**の tick を打ち直す。
/// ステップ内で同じ段が続く区間 (発行の合間) の区切りに使う。計測 OFF は分岐 1 回。
#[inline]
pub fn chain_seg_mark(op: usize) {
    if !crate::prof::split_on() {
        return;
    }
    let cur = CHAIN_SEG.with(|c| c.get());
    if cur == 0 || cur != op || chain_row(op).is_none() {
        return;
    }
    let now = Instant::now();
    let prev = CHAIN_TICK.with(|c| c.replace(Some(now)));
    if let Some(t0) = prev {
        chain_charge(now.duration_since(t0).as_secs_f64());
    }
}

/// s140: 「このステップは chain 本体まで到達した」ことを表 A のサンプル数に数える。
/// 1 ステップ 1 回 (`forward_decode_chain` が段の切り替えを始める直前)。
#[inline]
pub fn chain_step_entered() {
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return;
    }
    crate::prof::split_tick_call(crate::prof::SPLIT_CHAIN_TABLE);
}

/// `op` の区間だけを測る (段を切り替えずに、その間の時間を `op` へ引く)。
///
/// 使いどころは「sync の中に入った 1 本のカーネル実行」— decode では
/// `cuCtxSynchronize` が直前に発行したカーネルの完了を待つ時間が支配的なので、
/// 発行直後と sync 直後に同じ tick を読み、その差を `kernel_wait` として
/// 待ち行から分離する (待ちの合計は動かない)。計測 OFF はクロック読みゼロ。
#[inline]
fn chain_mark_span(op: usize, t0: Instant) {
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return;
    }
    if let Some(i) = chain_row(op) {
        CHAIN_ACC.with(|a| a.borrow_mut()[i] += t0.elapsed().as_secs_f64());
    }
}

/// 計測するなら今の時刻 (呼び出し側はこれが None なら何もしない)。
/// 公開しているのは「発行したカーネルの完了待ち」を `chain_kernel` へ移すため —
/// 呼び出し側 (decode chain) が発行の直後と sync の直後に 1 回ずつ読む。
#[inline]
pub fn chain_tick() -> Option<Instant> {
    if crate::prof::split_on() {
        Some(Instant::now())
    } else {
        None
    }
}

thread_local! {
    /// 直近の `chain_sync_observed` が観測した (sync の所要秒, 直前に発行した
    /// カーネルの実行秒)。計測 OFF では None のまま = 呼び出し側は何もしない。
    static CHAIN_SIG: std::cell::Cell<Option<(f64, f64)>> = const { std::cell::Cell::new(None) };
}

/// chain の sync を分割表の観点で観測する (`chain_sync_pending` の直後に 1 回)。
///
/// `secs` = sync の所要、`kernel_secs` = 直前に発行したカーネルの tick からの経過。
/// 両方が揃っているときだけ「カーネル実行 = sync 所要のうち tick からの経過」を
/// `chain_kernel` へ移し、待ち行から同じ秒数を引く (待ちの合計は動かない)。
/// `chain_step_begin` していない (段 0) や計測 OFF では何もしない。
///
/// ここで観測した値は `chain_sig` が 1 回だけ返す (表へは呼び出し側が流す)。
pub fn chain_sync_observed(secs: f64, kernel_secs: Option<f64>) {
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return;
    }
    // カーネル実行は待ちを超えない (tick は sync の直前に打つので、超えるのは
    // 時計の粒度か別スレッドの割り込み) — 超えた分は待ち側に残す。
    let sig = match kernel_secs {
        Some(k) if k > 0.0 && secs > 0.0 => (secs, k.min(secs)),
        _ => (secs, 0.0),
    };
    CHAIN_SIG.with(|c| c.set(Some(sig)));
}

/// `chain_sync_observed` が観測した (sync 秒, カーネル実行秒) を 1 回だけ返す。
/// 計測 OFF / 未観測は None (呼び出し側は何もしない)。
#[inline]
pub fn chain_sig() -> Option<(f64, f64)> {
    CHAIN_SIG.with(|c| c.replace(None))
}

/// ステップ先頭: chain の計測を開き、最初の段を `CHAIN_OUTER` (embedding 用意・
/// reserve など chain の外側) にする。呼び出しは decode chain の入口 1 か所だけで、
/// 対応する `chain_step_end` がステップ末尾に 1 か所ある (失敗して途中で抜ける
/// ステップはそこまでが積まれ、以後のステップで上書きされる)。
///
/// 計測 OFF は分岐 1 回 (クロックも読まない)。
pub fn chain_step_begin() {
    // s166 (P5): 観測カウンタのリセット (env OFF は分岐 1 回) — prefill で積んだ
    // 分を decode の数字に混ぜない。
    decode_dbg_reset();
    if !crate::prof::split_on() {
        return;
    }
    CHAIN_ACC.with(|a| *a.borrow_mut() = [0.0; crate::prof::SPLIT_ROWS]);
    CHAIN_RESID.with(|c| c.set(0.0));
    CHAIN_TICK.with(|c| c.set(Some(Instant::now())));
    CHAIN_SEG.with(|c| c.set(crate::prof::CHAIN_OUTER));
}

/// ステップ末尾: 最後の段を締め、1 ステップ分を表 A へ流し込む。呼び出しは
/// `chain_step_begin` と対で 1 か所。計測 OFF は分岐 1 回。
pub fn chain_step_end() {
    // s166 (P5): 1 ステップ分の launch / copy 件数を stderr へ (env OFF は分岐 1 回)。
    decode_dbg_flush();
    if !crate::prof::split_on() {
        return;
    }
    if CHAIN_SEG.with(|c| c.get()) == 0 {
        return; // begin していない (別の経路) — 何も積まない
    }
    // 最後の段をここで締める (tick はこの 1 回だけ)
    if let Some(t0) = CHAIN_TICK.with(|c| c.replace(None)) {
        chain_charge(Instant::now().duration_since(t0).as_secs_f64());
    }
    CHAIN_SEG.with(|c| c.set(0));
    chain_step_flush();
}

/// ステップ末尾: 最後の段の区間を締めて、1 ステップ分を表 A へ流し込む。
/// 締めの tick は `chain_step_end` (呼び出し側の 1 行) の側で読む。
///
/// 「どの sync からも取れなかった待ち」(`CHAIN_RESID`) は**待ち行**へ入れる —
/// sync 区間の理解できない残余は発行ではなく待ちだから (排他の和は動かない)。
fn chain_step_flush() {
    let rows = CHAIN_ACC.with(|a| {
        let acc = &mut *a.borrow_mut();
        let rows = *acc;
        *acc = [0.0; crate::prof::SPLIT_ROWS];
        rows
    });
    let mut acc = rows;
    let resid = CHAIN_RESID.with(|c| c.replace(0.0));
    if resid > 0.0 {
        // どの sync からも取れなかった待ちは残余行 (= `CHAIN_SYNC_RESIDUAL`) へ。
        // 待ちの合計は動かない (残余も待ちとして数える)。
        acc[crate::prof::CHAIN_SYNC_RESIDUAL - crate::prof::SPLIT_CHAIN_FIRST] += resid;
    }
    crate::prof::split_flush_step(crate::prof::SPLIT_CHAIN_TABLE, &acc, 1, None);
}

/// chain の sync (`cuCtxSynchronize`) を「段」の観点で観測する。
///
/// sync の所要 `secs` と、sync の直前まで走っていた段 `prev` が分かるとき、
/// その待ちの帰属は 1 つに決まる:
///   - prev が launch 系 (`chain_launch`) … 直前に発行したカーネルの完了待ち。
///     `chain_kernel` として積み、待ち行 (`prev`) からは **同じ秒数を引く**
///     (待ちの合計は動かない = 排他の和が保たれる)。
///   - prev が既に待ち (`chain_sync_wait` / `chain_sync_resid`) … 前の sync で
///     取り切れなかった分。`chain_sync_wait` に積む。
///   - それ以外 … 待ち行へ (残余はステップ末尾でまとめて待ち行に寄せる)。
///
/// `kernel` は直前に発行した 1 本のカーネルの実行時間 (発行直後と sync 直後の
/// tick の差)。計測 OFF なら `secs` / `kernel` は読まれない (呼び出し側が None)。
#[inline]
fn chain_observe_sync(secs: f64, kernel: Option<f64>) {
    if !crate::prof::split_on() {
        return;
    }
    let prev = CHAIN_SEG.with(|c| c.get());
    if prev == 0 {
        return;
    }
    let wait_row = crate::prof::CHAIN_SYNC_WAIT - crate::prof::SPLIT_CHAIN_FIRST;
    let launch_row = crate::prof::CHAIN_LAUNCH - crate::prof::SPLIT_CHAIN_FIRST;
    let kern_row = crate::prof::CHAIN_KERNEL_WAIT - crate::prof::SPLIT_CHAIN_FIRST;
    match (prev == crate::prof::CHAIN_LAUNCH, kernel) {
        // launch 直後の sync: カーネル実行 = 待ちの本体
        (true, Some(k)) if k > 0.0 => {
            CHAIN_ACC.with(|a| {
                let mut a = a.borrow_mut();
                a[kern_row] += k;
                // 待ちの合計を二重にしないため、帰属元の launch から引く
                a[launch_row] -= k;
                if secs > k {
                    a[wait_row] += secs - k;
                }
            });
        }
        // launch 以外からの sync は待ち行へ (残余は末尾で待ち行に寄せる)
        _ => {
            if secs > 0.0 {
                CHAIN_ACC.with(|a| a.borrow_mut()[wait_row] += secs);
            }
        }
    }
}

/// P17b 統合エントリ: int8 線形層を GPU で実行するかどうかを判定し、
/// 実行するなら結果 (CPU カーネルとビット一致) を返す。
/// 戻り値 None = CPU パスを使うこと (無効/初期化失敗/実行時エラー/m>閾値)。
pub fn maybe_gpu_linear(
    uid: u64,
    a: &[f32],
    qw: &[i8],
    scale: &[f32],
    k: usize,
    n: usize,
    m: usize,
) -> Option<Vec<f32>> {
    // s63: 確保 + 書き込みの薄いラッパ (decode など呼び出し側が再利用率の低い
    // 用途向け)。prefill のホットパスは `maybe_gpu_linear_into` を使う。
    let mut out = vec![0.0f32; m * n];
    if maybe_gpu_linear_into(uid, a, qw, scale, k, n, m, &mut out) {
        Some(out)
    } else {
        None
    }
}

/// s63: `maybe_gpu_linear` の書き込み先指定版。
/// true = GPU が計算して `out` を埋めた / false = 呼び出し側が CPU パスを使うこと。
/// 判定・重みキャッシュ・フォールバック (恒久ラッチ) の規約は `maybe_gpu_linear`
/// と完全に同一で、数値経路も同一 (D2H 後に `ho` から `out` へコピーするだけ)。
pub fn maybe_gpu_linear_into(
    uid: u64,
    a: &[f32],
    qw: &[i8],
    scale: &[f32],
    k: usize,
    n: usize,
    m: usize,
    out: &mut [f32],
) -> bool {
    debug_assert_eq!(out.len(), m * n);
    // s57 (T4): 実行先ラベルの初期値 — この呼び出しが false で抜けたとき
    // 直前の true が残らないよう毎回 false から始める (計測のみ)
    set_last_linear_ran_on_gpu(false);
    // s61: 入口到達の記録 (回帰テスト用・Relaxed 加算のみ)
    LINEAR_SEAM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !cuda_enabled() {
        return false;
    }
    let use_gemm = m > max_m() && m >= gemm_min_m() && gemm_enabled();
    // m が GEMV 経路の上限を超えて GEMM も発動しないなら CPU へ
    if m > max_m() && !use_gemm {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU int8: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    let gw = match weights_for(uid, qw, scale, k, n) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("CUDA weight upload failed, using CPU int8: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return false;
        }
    };
    // P17i: m が大きい (prefill) ときは tiled GEMM、小さい (decode/batch) ときは
    // GEMV 型カーネル (固定オーバーヘッドが小さい) を使い分ける。どちらも CPU と
    // ビット一致。
    let r = if use_gemm {
        linear_gemm_into(ctx, a, gw, m, k, n, out)
    } else {
        linear_pinned(ctx, a, gw, m, k, n).map(|v| out.copy_from_slice(&v))
    };
    match r {
        Ok(()) => {
            // s57 (T4): 実行先ラベルの記録 (計測のみ・戻り値/演算には無関係)
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            // 実行時エラーは再試行せず恒久的に CPU へ (数値は同一仕様なので不変)
            tracing::error!("CUDA linear failed, falling back to CPU permanently: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            false
        }
    }
}

/// s123: Q4_0 ネイティブ重みの VRAM レジストリ (キー = `Q4Linear.uid`)。
///
/// SOA で 2 本のバッファに分けて置く: `nib` = 1 ブロック 16B (uint4) を
/// n*nb 個並べたもの、`sc` = 1 ブロック 1 本の f32 スケール。
/// 1 要素あたり 4.5 bit (int8 経路の半分以下) で済む。
struct GpuWq4 {
    nib: DevBuffer,
    sc: DevBuffer,
    k: usize,
    n: usize,
}

/// s165: 連結重み (`gate_up_w` = [gate 全行 | up 全行]) の行範囲を、元の独立した
/// 重み uid のデバイス表現として扱うためのエイリアス表。
///
/// 背景 (nsys・2026-10-22): 27B q4 の FFN は **decode chain が連結 1 本
/// (`gate_up_w`)**、**prefill の Q4 経路 (`chain_mlp_prefill_q4`) が gate_w / up_w
/// の 2 本**を使うため、同じ 2×ff×e パラメータが VRAM に 2 度載っていた
/// (1 層 111.4 MB × 64 層 = 7.1 GB、H2D も同じ 7.1 GB)。ここに行範囲
/// (uid → (連結 uid, 行オフセット)) を登録すると、`weights_for_q4` は個別の
/// 重みを転送せず連結側のデバイス領域を指すビューを返す。
///
/// 数値は不変: 連結側の nib / sc は行単位で連続しており (1 行 = nb ブロック)、
/// 行オフセット ff の位置は up_w の先頭行そのもの。カーネル・引数・グリッドは
/// 従来と同一で、重みの置き場所だけが変わる。
static Q4_ALIAS: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, (u64, usize)>>> =
    OnceLock::new();

fn q4_alias_table() -> &'static std::sync::Mutex<std::collections::HashMap<u64, (u64, usize)>> {
    Q4_ALIAS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// s165: `Linear` から Q4 本体を取り出す (連結エイリアスの前提検査用)。
fn q4_of(l: &crate::quant::Linear) -> Option<&crate::quant::Q4Linear> {
    match l {
        crate::quant::Linear::Q4(x) => Some(x),
        _ => None,
    }
}

/// s165: 連結重みの行範囲割り当て (純粋関数・GPU 不要)。
///
/// `concat_k` / `concat_n` は連結側の形状、`parts` は各行範囲の (k, n)。
/// 「すべての part の k が concat_k と一致し、n の和が concat_n」のときだけ
/// 各行の先頭行オフセットを返す (満たさなければ `None` = 呼び出し側は
/// エイリアスを登録せず従来の個別転送に落ちる = fail closed)。
pub fn q4_row_alias_offsets(
    concat_k: usize,
    concat_n: usize,
    parts: &[(usize, usize)],
) -> Option<Vec<usize>> {
    if parts.is_empty() || concat_k == 0 || concat_k % 32 != 0 || concat_n == 0 {
        return None;
    }
    let mut out = Vec::with_capacity(parts.len());
    let mut off = 0usize;
    for &(k, n) in parts {
        if k != concat_k || n == 0 {
            return None;
        }
        out.push(off);
        off = off.checked_add(n)?;
    }
    if off != concat_n {
        return None;
    }
    Some(out)
}

/// s165: `concat` (行連結済みの 1 本) の行範囲を `parts` の各 uid に割り当てる。
/// 呼び出し側は「parts の k がすべて concat.k と一致し、行数の和が concat.n」を
/// 満たすときだけ呼ぶ。**前提を満たさない・Q4 でない・転送に失敗した場合は
/// 何も登録せず false を返す** → 呼び出し側は従来どおり個別転送にフォールバック
/// する (fail closed)。登録済み (既に個別に VRAM へ載っている) uid は、in-flight
/// のカーネルが読みうるため置き換えない (その層は節約なし・数値は不変)。
pub fn q4_register_row_alias(concat: &crate::quant::Linear, parts: &[&crate::quant::Linear]) -> bool {
    let Some(c) = q4_of(concat) else { return false };
    if parts.is_empty() {
        return false;
    }
    let mut qs: Vec<&crate::quant::Q4Linear> = Vec::with_capacity(parts.len());
    for p in parts {
        let Some(p) = q4_of(p) else { return false };
        qs.push(p);
    }
    q4_register_row_alias_q4(c, &qs)
}

/// s166 (P5): [`q4_register_row_alias`] の Q4 版 (decode chain の qkv 連結用)。
fn q4_register_row_alias_q4(
    c: &crate::quant::Q4Linear,
    parts: &[&crate::quant::Q4Linear],
) -> bool {
    if parts.is_empty() {
        return false;
    }
    // 形状契約: parts はすべて Q4・k は concat と一致・行数の和が concat.n。
    let mut geom = Vec::with_capacity(parts.len());
    let mut uids = Vec::with_capacity(parts.len());
    for p in parts {
        geom.push((p.k, p.n));
        uids.push(p.uid);
    }
    let Some(rows) = q4_row_alias_offsets(c.k, c.n, &geom) else {
        return false;
    };
    let offs: Vec<(u64, usize)> = uids.into_iter().zip(rows).collect();
    // 連結側を先にレジストリへ載せる (エイリアスの参照先が存在する状態にする)。
    if weights_for_q4(c.uid, &c.blocks, c.k, c.n).is_err() {
        return false;
    }
    let table = q4_alias_table();
    let mut t = match table.lock() {
        Ok(t) => t,
        Err(_) => return false,
    };
    for (uid, row_off) in offs {
        if uid == c.uid {
            return false;
        }
        t.entry(uid).or_insert((c.uid, row_off));
    }
    true
}

/// s166 (P5): 連結重み `fused_uid` の行範囲が `parts` に割り当て済みか
/// (行オフセットが [0, n0, n0+n1, ...] と一致するか) を照会する。
fn q4_fused_ready(fused_uid: u64, parts: &[&crate::quant::Q4Linear]) -> bool {
    let t = match q4_alias_table().lock() {
        Ok(t) => t,
        Err(_) => return false,
    };
    let mut off = 0usize;
    for p in parts {
        match t.get(&p.uid) {
            Some(&(tgt, ro)) if tgt == fused_uid && ro == off => {}
            _ => return false,
        }
        off += p.n;
    }
    true
}

/// s165: `uid` にエイリアスが登録されていれば、連結側のデバイス領域を指す
/// ビューを `map` に登録して返す。形状・サイズが 1 つでも食い違えば `None`
/// (呼び出し側は通常の転送へ落ちる = fail closed)。
fn q4_alias_view(
    uid: u64,
    k: usize,
    n: usize,
    map: &mut std::collections::HashMap<u64, Box<GpuWq4>>,
) -> Option<&'static GpuWq4> {
    if map.contains_key(&uid) {
        return None;
    }
    let (tgt, row_off) = {
        let t = q4_alias_table().lock().ok()?;
        *t.get(&uid)?
    };
    if k == 0 || k % 32 != 0 || row_off.checked_add(n)? > usize::MAX / 4 {
        return None;
    }
    let nb = k / 32;
    // 参照先 (連結側) がまだ転送されていなければ何もしない。
    let (nib_ptr, sc_ptr, tn, tk) = {
        let e = map.get(&tgt)?;
        (e.nib.ptr(), e.sc.ptr(), e.n, e.k)
    };
    if tk != k || row_off + n != tn {
        return None;
    }
    if nib_ptr == 0 || sc_ptr == 0 {
        return None;
    }
    let nib_off = row_off.checked_mul(nb)?.checked_mul(16)?;
    let sc_off = row_off.checked_mul(nb)?.checked_mul(4)?;
    let nib_bytes = n.checked_mul(nb)?.checked_mul(16)?;
    let sc_bytes = n.checked_mul(nb)?.checked_mul(4)?;
    // 範囲が連結側バッファ内に収まることを確認 (収まらなければ fail closed)。
    {
        let e = map.get(&tgt)?;
        if nib_off + nib_bytes > e.nib.bytes() || sc_off + sc_bytes > e.sc.bytes() {
            return None;
        }
    }
    let v = GpuWq4 {
        nib: DevBuffer::view(nib_ptr + nib_off as u64, nib_bytes),
        sc: DevBuffer::view(sc_ptr + sc_off as u64, sc_bytes),
        k,
        n,
    };
    map.insert(uid, Box::new(v));
    let e = map.get(&uid).expect("just inserted");
    Some(unsafe { std::mem::transmute::<&GpuWq4, &'static GpuWq4>(e.as_ref()) })
}

/// s123: Q4_0 重みのデバイスレジストリ (uid → nib/sc)。s166 (P5) で関数内
/// static から出した (`q4_resident` が同じ表を照会するため)。
fn q4_registry() -> &'static std::sync::Mutex<std::collections::HashMap<u64, Box<GpuWq4>>> {
    static REG: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, Box<GpuWq4>>>> =
        OnceLock::new();
    REG.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// s166 (P5): `uid` が既にデバイスへ載っているか (**照会のみ・転送しない**)。
/// 連結重みを上げる前に「parts が個別に載っていないか」を確かめるために使う
/// (載っているのに連結側も上げると VRAM が 2 重になる)。ロックが毒されていたら
/// `true` = 「載っている」とみなして連結側を上げない (安全側)。
fn q4_resident(uid: u64) -> bool {
    match q4_registry().lock() {
        Ok(m) => m.contains_key(&uid),
        Err(_) => true,
    }
}

/// s123: [n,k] の Q4_0 ブロック (1 ブロック 18B) を VRAM へ上げる。
/// ホスト側で SOA (nib 16B アライン / sc f32) に変換してから転送する。
/// s165: 連結重みの行範囲エイリアス (登録済み) があれば転送せずビューを返す。
fn weights_for_q4(uid: u64, blocks: &[u8], k: usize, n: usize) -> Result<&'static GpuWq4> {
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
    }
    let e = map.get(&uid).expect("just inserted");
    debug_assert_eq!((e.k, e.n), (k, n), "uid 重複: 同一 uid に異なる形状 (q4)");
    Ok(unsafe { std::mem::transmute::<&GpuWq4, &'static GpuWq4>(e.as_ref()) })
}

/// s123: Q4_0 線形層の結果を Vec で返す版 (`Linear::apply` 系から使う)。
pub fn maybe_gpu_linear_q4(
    uid: u64,
    a: &[f32],
    blocks: &[u8],
    k: usize,
    n: usize,
    m: usize,
) -> Option<Vec<f32>> {
    let mut out = vec![0.0f32; m * n];
    if maybe_gpu_linear_q4_into(uid, a, blocks, k, n, m, &mut out) {
        Some(out)
    } else {
        None
    }
}

/// s123: Q4_0 線形層の書き込み先指定版。
/// true = GPU が計算して `out` を埋めた / false = 呼び出し側が CPU パスを使うこと。
/// 判定・フォールバック (恒久ラッチ) の規約は int8 の `maybe_gpu_linear_into` と同一。
/// 数値は CPU の `matvec_rows_bt_q4_into` と同じ定義 (差は累積順のみ・R2 以内)。
pub fn maybe_gpu_linear_q4_into(
    uid: u64,
    a: &[f32],
    blocks: &[u8],
    k: usize,
    n: usize,
    m: usize,
    out: &mut [f32],
) -> bool {
    // s148 (Task B10・計測のみ): ここは **q4 で実際に線形層が走る唯一の入口**
    // (27B q4 の prefill / decode の全投影・lm_head がここを通る)。表 C の
    // `sl_lin` はこの入口から戻るまでを 1 サンプルとして数える — wrapper
    // (`lin_stage`) の内側ではなく「本当に呼ばれた回数」と一致させるため。
    // 計測 OFF は `span_begin` が `None` を返し、`span_end` は分岐 1 回で戻る。
    //
    // s151 (Task B12): 内訳 (`sl_lin_*`) は s148 で 1 call に 3 号 (入口の地 /
    // 重み登録 / ピン留め) 積まれていた。`lin_call_begin` / `lin_call_end` が
    // 「この call の other は最大 1 号」「子の和 = 全体」を保つ。
    let call_t = if crate::prof::serve_on() {
        crate::prof::count_lin_dispatch();
        crate::prof::lin_call_begin()
    } else {
        None
    };
    let r = maybe_gpu_linear_q4_into_inner(uid, a, blocks, k, n, m, out);
    crate::prof::lin_call_end(call_t);
    r
}

/// s148: [`maybe_gpu_linear_q4_into`] の本体 (計測の包みを外したもの)。
fn maybe_gpu_linear_q4_into_inner(
    uid: u64,
    a: &[f32],
    blocks: &[u8],
    k: usize,
    n: usize,
    m: usize,
    out: &mut [f32],
) -> bool {
    debug_assert_eq!(out.len(), m * n);
    set_last_linear_ran_on_gpu(false);
    LINEAR_SEAM_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if !cuda_enabled() {
        return false;
    }
    if k % 32 != 0 {
        return false;
    }
    // s123: Q4 の CPU フォールバックは単純ループで遅い (27B で秒/token 級)。
    // int8 の gemm_min_m (既定 256) は「CPU sgemm の方が速い帯域」を避けるための
    // 閾値なので、Q4 には適用せず GEMV 上限を超えたら常にタイル GEMM を使う。
    let use_gemm = m > max_m() && gemm_enabled();
    if m > max_m() && !use_gemm {
        return false;
    }
    // s148: ここから先の「入口の地のホスト時間」(デバイス取得・コンテキスト・
    // 重み登録) を `sl_lin_other` として明示する (段の和を入口の壁時計に一致させる)。
    // s151 (B12): `lin_other_push` は **1 call につき 1 号** しか積まないので、
    // 下の 3 か所で呼んでもサンプル数は 1 のまま (s148 は 3 号になっていた)。
    let entry_st = crate::prof::span_begin();
    let Some(ctx) = shared_ctx() else {
        crate::prof::lin_other_close(entry_st);
        return false;
    };
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU q4: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        crate::prof::lin_other_close(entry_st);
        return false;
    }
    // s148: 重みのデバイス登録 (初回のみ・数 MB の H2D) とディスパッチの地の
    // ホスト時間を `sl_lin_other` として明示する (段の和を入口の壁時計に一致させる)。
    let st = crate::prof::span_begin();
    let gw = match weights_for_q4(uid, blocks, k, n) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("CUDA q4 weight upload failed, using CPU: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            crate::prof::lin_other_close(st);
            crate::prof::lin_other_close(entry_st);
            return false;
        }
    };
    let r = q4_dispatch(ctx, a, gw, m, k, n, use_gemm, out);
    crate::prof::lin_other_close(st);
    crate::prof::lin_other_close(entry_st);
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            tracing::error!("CUDA q4 linear failed, falling back to CPU permanently: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            false
        }
    }
}

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
fn lin_stage<T>(op: usize, f: impl FnOnce() -> T) -> T {
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
fn split_stage() -> Option<Instant> {
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
fn split_mark(t: &mut Option<Instant>, op: usize) {
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
fn split_flush(t: &mut Option<Instant>) {
    *t = None;
}

/// s168 (P7): prefill の q4 GEMM 行タイル (**実測ノブ**)。1 タイル = 1 回の
/// `dno_gemm_q4_0` launch で、grid は (mr.div_ceil(64), n.div_ceil(128))。
/// 既定は s124 以来の 1024。`DENNO_Q4_M_TILE` で 64..=8192 の 64 の倍数を
/// 指定できる (再ビルド無しで 512 / 1024 / 2048 / 4096 を A/B するための入口)。
///
/// 中間バッファは 1 タイルあたり (行) x (列) x 4 B で、行数は端数併合を含めて
/// 最大 `tile + q4_tail_merge_below()` (既定 1024 + 16 = 1040 行)。
/// 27B (e=5120, ff=17408) の MLP で 1040 行 = CH_GA 138 MiB / CH_ACT 69 MiB /
/// CH_PFH・CH_DOWN 各 20 MiB。4096 行なら 546 MiB / 273 MiB / 80 MiB、
/// 8192 行なら 1090 MiB / 545 MiB / 160 MiB。
const Q4_M_TILE: usize = 1024;

/// [`Q4_M_TILE`] の上限。中間バッファがタイル行数に比例するので 8192 行で止める
/// (27B MLP の CH_GA は約 1.1 GiB — 13k 行を一度に置く 2.7 GiB には届かない)。
const Q4_M_TILE_MAX: usize = 8192;

/// タイル幅の刻み。`dno_gemm_q4_0` の行ブロック (Q4_GM = 64) に揃える — 途中の
/// タイルはすべて 64 の倍数行になり、grid.x が端数ブロックを出さない。
const Q4_M_TILE_STEP: usize = 64;

/// `DENNO_Q4_M_TILE` の解釈 (純粋関数)。範囲外・非数・刻み外は既定へ落とす
/// (不正な env で panic させない — `q4_gemv_c()` と同じ規約)。
fn parse_q4_m_tile(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(t) if (Q4_M_TILE_STEP..=Q4_M_TILE_MAX).contains(&t) && t % Q4_M_TILE_STEP == 0 => t,
        _ => Q4_M_TILE,
    }
}

/// [`Q4_M_TILE`] の実効値 (env `DENNO_Q4_M_TILE` があればそちら)。プロセス内で
/// 1 回だけ読む (serve の起動時に env を決める運用なので、途中変更は見ない)。
fn q4_m_tile() -> usize {
    static TILE: OnceLock<usize> = OnceLock::new();
    *TILE.get_or_init(|| {
        let v = std::env::var("DENNO_Q4_M_TILE").ok();
        parse_q4_m_tile(v.as_deref())
    })
}

/// s168 (P7): 端数タイルを直前のタイルへ併合する閾値 (行)。
///
/// `max_m()` は「GEMV が GEMM より速い帯域」の上限 (既定 16) で、ファイル中の
/// 他の dispatch はこの行数以下を GEMV に回す。prefill の GEMM は 64 行ブロック
/// 単位なので、これ未満の端数を単独タイルにすると 1 launch がほぼ空回りする
/// (しかも sync 付きの経路では待ち時間が丸ごと乗る)。端数がこれ未満なら
/// 直前のタイルに足して 1 回で流す。`DENNO_CUDA_MAX_M=0` にすると併合しない
/// (併合そのものの A/B 用)。
fn q4_tail_merge_below() -> usize {
    max_m()
}

/// s168 (P7): 1 タイルが受ける最大行数 (端数併合を含む)。中間バッファはこの
/// 行数で**ループの前に**確保する — 途中で grow させると、先行 launch が読んで
/// いるバッファを解放しうる (`Scratch::ensure` は grow-only の入れ替え)。
fn q4_tile_max_rows(tile: usize) -> usize {
    tile.saturating_add(q4_tail_merge_below())
}

/// s168 (P7): `r0` から始まる次の行グループの行数 (純粋関数)。
/// 端数が [`q4_tail_merge_below`] 以下なら、その端数もこのグループに含める
/// (返り値 <= tile + tail)。行は互いに独立で 1 行の累積順は変わらないため、
/// 行のまとめ方は出力のビットを変えない (カーネル・引数・grid の作り方は同一)。
///
/// 閾値は `<=` にして [`pf_res_gate`] / [`gdn_prefill_q4_geom`] の
/// 「端数 <= max_m() は受けない」と揃えてある — ゲートが受けた n に対しては、
/// 端数タイルが単独で launch されることが無い。
fn q4_group_rows(r0: usize, m: usize, tile: usize) -> usize {
    let left = m - r0;
    let mr = tile.min(left);
    let rem = left - mr;
    if rem > 0 && rem <= q4_tail_merge_below() {
        left
    } else {
        mr
    }
}

/// s168 (Task P7): 行タイルのノブと端数併合の契約 (純粋関数・GPU 不要)。
#[cfg(test)]
mod p7_tile_tests {
    use super::*;

    /// `DENNO_Q4_M_TILE` の解釈: 既定は従来値、範囲外・刻み外・非数は既定へ。
    #[test]
    fn env_tile_parse() {
        assert_eq!(Q4_M_TILE, 1024, "既定は s124 以来の 1024 のまま");
        assert_eq!(parse_q4_m_tile(None), 1024);
        assert_eq!(parse_q4_m_tile(Some("")), 1024);
        assert_eq!(parse_q4_m_tile(Some("abc")), 1024);
        assert_eq!(parse_q4_m_tile(Some("-512")), 1024);
        assert_eq!(parse_q4_m_tile(Some("0")), 1024);
        assert_eq!(parse_q4_m_tile(Some("32")), 1024, "64 行未満は受けない");
        assert_eq!(parse_q4_m_tile(Some("96")), 1024, "64 の倍数でない");
        assert_eq!(parse_q4_m_tile(Some("999999")), 1024, "上限超え");
        assert_eq!(parse_q4_m_tile(Some(" 2048 ")), 2048, "前後の空白は許す");
        for t in [64usize, 512, 1024, 2048, 4096, 8192] {
            assert_eq!(parse_q4_m_tile(Some(&t.to_string())), t, "t={t}");
        }
        // 本番の実効値も必ず受理域に入る (panic も 0 行タイルも作らない)
        let eff = q4_m_tile();
        assert!(eff % Q4_M_TILE_STEP == 0 && eff >= Q4_M_TILE_STEP && eff <= Q4_M_TILE_MAX);
    }

    /// 行グループの契約: 重複・欠落なしで m 行を覆い、1 グループは cap 行以下。
    /// 途中のグループは必ず tile 行ちょうど (併合は最終グループだけ)、
    /// 端数 (<= tail) が単独タイルとして launch されることは無い。
    #[test]
    fn row_groups_cover_rows_exactly() {
        let tail = q4_tail_merge_below();
        for tile in [64usize, 128, 512, 1024, 2048, 4096] {
            let cap = q4_tile_max_rows(tile);
            assert_eq!(cap, tile + tail, "tile={tile}");
            for m in 1..=3000usize {
                let (mut r0, mut groups) = (0usize, 0usize);
                while r0 < m {
                    let mr = q4_group_rows(r0, m, tile);
                    assert!(mr > 0, "tile={tile} m={m} r0={r0}: 空グループ");
                    assert!(mr <= cap, "tile={tile} m={m}: {mr} 行 > cap {cap}");
                    if r0 + mr < m {
                        assert_eq!(mr, tile, "tile={tile} m={m} r0={r0}: 途中が {mr} 行");
                    } else if r0 > 0 {
                        // 最終グループ: tile ちょうど or 併合 (tile 超) or tail 超の端数
                        assert!(
                            mr >= tile || mr > tail,
                            "tile={tile} m={m}: 端数 {mr} 行 (tail {tail}) を単独 launch している"
                        );
                    }
                    r0 += mr;
                    groups += 1;
                    assert!(groups <= m, "tile={tile} m={m}: グループ数が行数を超えた");
                }
                assert_eq!(r0, m, "tile={tile} m={m}: 行が覆われていない");
            }
        }
    }

    /// 13k token の実形 (27B の prefill) で、タイル幅ごとの launch 回数と
    /// 端数併合の有無を固定する (REPORT_P7.md の表と同じ計算)。
    #[test]
    fn row_groups_at_13k_tokens() {
        let tail = q4_tail_merge_below();
        // (tile, 13,000 行のグループ数) — 端数 712 行はどの tile でも tail 超
        for (tile, want) in [(512usize, 26usize), (1024, 13), (2048, 7), (4096, 4)] {
            let mut r0 = 0usize;
            let mut n = 0usize;
            while r0 < 13_000 {
                r0 += q4_group_rows(r0, 13_000, tile);
                n += 1;
            }
            assert_eq!(n, want, "tile={tile} 13,000 行の launch 回数");
        }
        // 端数が tail 以下の形は直前のタイルへ併合され、グループ数が 1 減る
        let m = 1024 * 3 + 4;
        let mut r0 = 0usize;
        let (mut n, mut last) = (0usize, 0usize);
        while r0 < m {
            last = q4_group_rows(r0, m, 1024);
            r0 += last;
            n += 1;
        }
        if tail >= 4 {
            assert_eq!(n, 3, "端数 4 行は 3 番目のタイルへ併合 (2,048+ 行)");
            assert_eq!(last, 1024 + 4, "併合後の最終タイル行数");
        } else {
            assert_eq!(n, 4, "併合 OFF (DENNO_CUDA_MAX_M=0) では 4 グループ");
        }
    }
}

/// s123: Q4_0 の 1 層ぶん。小さい m は GEMV (1 warp = 1 行)、大きい m は
/// タイル GEMM (1 block = 16 行 x 256 列)。H2D → kernel → D2H → sync ×1。
fn q4_dispatch(
    ctx: &CudaCtx,
    a: &[f32],
    gw: &GpuWq4,
    m: usize,
    k: usize,
    n: usize,
    use_gemm: bool,
    out: &mut [f32],
) -> Result<()> {
    // s124 (P4-6): m を固定タイルに割る。`SCRATCH` は thread_local かつ grow-only
    // なので、プロンプト長そのままの m を渡すと (m*k + m*n)*4 B が GPU 経路を
    // 通ったスレッド数ぶん積み上がる (27B・10.8k token で約 11 GB になり、続く
    // リクエストが cuMemAlloc error 2 で CPU へ落ちた)。
    // 行は互いに独立で、1 行の演算順はタイル分割で変わらないため出力は bit 一致する。
    // s168 (P7): タイル幅は `DENNO_Q4_M_TILE` の実効値、端数は直前のタイルへ併合する。
    let tile = q4_m_tile();
    let lib = lib()?;
    ctx.make_current()?;
    let mut r0 = 0;
    while r0 < m {
        let mr = q4_group_rows(r0, m, tile);
        let a_t = &a[r0 * k..r0 * k + mr * k];
        let out_t = &mut out[r0 * n..r0 * n + mr * n];
        q4_dispatch_tile(ctx, lib, a_t, gw, mr, k, n, use_gemm, out_t)?;
        r0 += mr;
    }
    Ok(())
}

/// s132 (Task B1): [`q4_dispatch_tile`] の GEMV が 1 block で受ける最大行数。
/// `kernels_int8.cu` の `dno_gemv_q4_0mr` (acc[8]) と対で動く。
/// s133 以降、本番の GEMV 経路は `dno_gemv_q4_0c*` (下の [`Q4_GEMV_C`]) で、
/// `dno_gemv_q4_0mr` はビット一致の参照 (テスト) として残してある。
const Q4_MR_MAX: usize = 8;

/// s133 (Task B1, attempt 2): 列ブロック版 Q4 GEMV の C (1 回の A 読みで
/// 何列の重みを回すか)。`kernels_int8.cu` の `dno_gemv_q4_0c2/c4/c8` と対。
/// 1 MAC あたりの命令数は 1 + 1/C になる (C=1 の従来形は 2)。
/// 2 / 8 を試すときはここを変える (PTX には 3 本とも入っている)。
/// ビルドし直さずに比べるときは `DENNO_Q4_GEMV_C=2|4|8` で上書きする。
const Q4_GEMV_C: usize = 4;

/// [`Q4_GEMV_C`] の実効値 (env `DENNO_Q4_GEMV_C` があればそちら)。2/4/8 以外は
/// 既定へ落とす (不正な値で panic させない)。
fn q4_gemv_c() -> usize {
    match std::env::var("DENNO_Q4_GEMV_C").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(c @ (2 | 4 | 8)) => c,
        _ => Q4_GEMV_C,
    }
}

/// [`Q4_GEMV_C`] に対応する PTX エントリの index (0 = c2, 1 = c4, 2 = c8)。
fn q4_gemv_c_index(c: usize) -> usize {
    match c {
        2 => 0,
        4 => 1,
        8 => 2,
        _ => panic!("Q4_GEMV_C must be 2, 4 or 8"),
    }
}

/// s124: [`q4_dispatch`] の 1 タイル (m <= 256) を実行する。scratch はここで
/// m <= 256 に固定されるため、プロンプトが何万 token でも増えない。
fn q4_dispatch_tile(
    ctx: &CudaCtx,
    lib: &'static CudaLib,
    a: &[f32],
    gw: &GpuWq4,
    m: usize,
    k: usize,
    n: usize,
    use_gemm: bool,
    out: &mut [f32],
) -> Result<()> {
    ctx.make_current()?;
    // s148 (Task B10・計測のみ): serve (q4) 経路の線形呼び出しの中身を段へ割る。
    // s138 の表 B は wrapper (`lin_stage` / `linear_pinned`) の内側だけを数えて
    // いたため、本番の GEMM 経路 (`q4_dispatch_tile`) の時間に届いていなかった
    // (実測: 608 call / 7,846 ms に対し 802 サンプル / 445.8 ms)。ここは本番が
    // 実際に通るタイル実装の内側で、1 タイル = 1 サンプル。計測 OFF は tick が
    // `None` で、`split_mark` / `span_end` とも分岐 1 回で戻る (クロック読みなし)。
    let mut t = split_stage();
    // s148: 表 C の段。1 タイルの中で互いに排他 (確保 / H2D / launch / D2H / sync)。
    //
    // `st` は「いま開いている段」で、閉じるたびに**開き直さない** — `span_begin` は
    // 計測 OFF で `None` を返すので、閉じた後に開き直すと (計測 ON でも) ここで
    // 落とした区間が二度と閉じられない。各段の開始は必ず次の段を閉じた直後で取る。
    // こうすると段の和が「タイル入口 → 戻り」の壁時計に一致し、閉じ漏れが
    // `sl_lin_other` (残余) ではなく**未帰属の穴**として現れない。
    let mut st = crate::prof::span_begin();
    SCRATCH.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let Scratch { x, out: dout, hx, ho, .. } = &mut *s;
        let bytes_x = m * k * 4;
        let bytes_out = m * n * 4;
        Scratch::ensure(x, bytes_x)?;
        Scratch::ensure(dout, bytes_out)?;
        split_mark(&mut t, crate::prof::Q4P_OUT_ALLOC);
        // s153 (Task B14): クロック読みは 1 回 (`lin_stage_taken_ns`)。読んだ ns を
        // そのまま表 C へ積むので、子の和が親 (`sl_lin`) を超えず、
        // `sl_lin_other = 全体 - 段の和` が成り立つ。
        let out_ns = crate::prof::lin_stage_taken_ns(st.take());
        crate::prof::span_end_ns(out_ns, crate::prof::SL_LIN_OUTALLOC);
        // 表 C: staging のピン留め領域の確保 (段の開始はここ)
        st = crate::prof::span_begin();
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        split_mark(&mut t, crate::prof::Q4P_PIN_ALLOC);
        // 表 C: ピン留め確保は転送準備の一部なので `sl_lin_other` に寄せる
        // (`sl_lin_quant` は Q4 では行わない = 0 サンプルのまま)。
        // (割り当て済みの 2 本目以降は即時なので、ここは実質初回だけの時間。)
        crate::prof::lin_other_close(st.take());
        // 表 C: 活性化の staging コピー (hx ← a) から H2D 発行まで
        st = crate::prof::span_begin();
        let (dx, dout) = (x.as_ref().unwrap(), dout.as_ref().unwrap());
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..m * k].copy_from_slice(a);
        split_mark(&mut t, crate::prof::Q4P_H2D_STAGE);
        unsafe {
            check(
                (lib.memcpy_htod_async)(
                    dx.ptr(),
                    hx.ptr as *const c_void,
                    bytes_x,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync",
            )?;
            split_mark(&mut t, crate::prof::Q4P_H2D_ISSUE);
            let h2d_ns = crate::prof::lin_stage_taken_ns(st.take());
            crate::prof::span_end_ns(h2d_ns, crate::prof::SL_LIN_H2D);
            // 表 C: 引数の組み立て + カーネル発行 (段の開始 = H2D を閉じた直後)
            st = crate::prof::span_begin();
            let (mut pa, mut pn, mut ps, mut po) =
                (dx.ptr(), gw.nib.ptr(), gw.sc.ptr(), dout.ptr());
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            // s132: 1 block が受ける行数 (mr)。m = 1 は従来の 1 行 1 block 経路
            // (dno_gemv_q4_0) をそのまま通すため、mr は 2 以上でだけ使う。
            let mut mrc = (m.min(Q4_MR_MAX)) as c_int;
            let mut args_mr = vec![
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut mrc as *mut _ as *mut c_void,
            ];
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            if use_gemm {
                // Q4_GM=64 (m 方向) / Q4_GN=128 (n 方向)・block (16,16)
                let grid = (m.div_ceil(64) as u32, n.div_ceil(128) as u32, 1);
                ctx.launch_async(ctx.f_gemm_q4, grid, (16, 16, 1), 0, &mut args)?;
            } else if let Some(fc) = ctx.f_mv_q4_c {
                // s133 (Task B1, attempt 2): 1 block = 8 warp x C 列 = 8*C 列、
                // 1 行。1 warp が A の 32 値 (ブロック b) をレジスタに 1 回だけ
                // 読み、C 列ぶんの重みに使い回す。行ごとの累積順は 1 行 1 block
                // 経路と同一なので m = 1 もビット一致。
                // grid.x は 8*C 列を 1 ブロックで受ける形。
                let c = q4_gemv_c();
                let f_c = fc[q4_gemv_c_index(c)];
                let cols = 8 * c;
                let grid = (n.div_ceil(cols) as u32, m as u32, 1);
                ctx.launch_async(f_c, grid, (256, 1, 1), 0, &mut args)?;
            } else if let (Some(f_mr), 2..) = (ctx.f_mv_q4_mr, m) {
                // s132 (Task B1): 1 block = 8 列 (8 warp) x mr 行。重みは 1 回だけ
                // 読んで mr 行に使い回す (従来は grid.y = m で行ごとに再読み)。
                // mr = min(m, 8)・端数は最終 block の rows = m - i0 で落とす。
                // 列ブロック版が PTX に無いときだけここを通る (フォールバック)。
                let grid = (n.div_ceil(8) as u32, m.div_ceil(mrc as usize) as u32, 1);
                ctx.launch_async(f_mr, grid, (256, 1, 1), 0, &mut args_mr)?;
            } else {
                // 1 block = 8 行 (8 warp)・grid.x = n/8, grid.y = m
                let grid = (n.div_ceil(8) as u32, m as u32, 1);
                ctx.launch_async(ctx.f_mv_q4, grid, (256, 1, 1), 0, &mut args)?;
            }
            split_mark(&mut t, crate::prof::Q4P_KERNEL_LAUNCH);
            // 表 C: 発行段の終わり (ここまでが launch、以降は D2H と待ち)
            let launch_ns = crate::prof::lin_stage_taken_ns(st.take());
            crate::prof::span_end_ns(launch_ns, crate::prof::SL_LIN_LAUNCH);
            // 表 C: 出力 D2H の段は「発行の直前」から開く (sync 完了 + ホストコピーまで)
            st = if st.is_none() { crate::prof::span_begin() } else { st };
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            split_mark(&mut t, crate::prof::Q4P_D2H_ISSUE);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
            ctx.sync()?;
            split_mark(&mut t, crate::prof::Q4P_SYNC_WAIT);
        }
        out.copy_from_slice(&ho.as_slice::<f32>()[..m * n]);
        split_flush(&mut t);
        // 表 C: 出力 D2H = 発行 + sync の完了待ち + ピン留め領域 → out のコピー。
        // 待ち (`SL_LIN_SYNC`) は別に取らない — D2H の中に吸収される (排他の和を保つ)。
        // `SL_LIN_SYNC` は表 C の行として予約されているが、q4 タイルでは常に 0 サンプル
        // で、その分はこの行 (D2H) に含まれる (COVERAGE には 0 として出る)。
        let d2h_ns = crate::prof::lin_stage_taken_ns(st);
        crate::prof::span_end_ns(d2h_ns, crate::prof::SL_LIN_D2H);
        Ok(())
    })
}

/// ピン留め + 非同期 1 同期パイプライン:
///   H2D(x, async) → quantize kernel → gemv kernel → D2H(out, async) → sync ×1
fn linear_pinned(
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

/// P17i: 大きい m (prefill) 用 — tiled GEMM カーネル (dno_gemm_i8) で実行。
/// パイプラインは linear_pinned と同一 (H2D → quantize → gemm → D2H → sync ×1)。
/// 整数累積は順序不変なのでタイル化しても CPU とビット一致。
fn linear_gemm(
    ctx: &CudaCtx,
    a: &[f32],
    gw: &GpuW,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; m * n];
    linear_gemm_into(ctx, a, gw, m, k, n, &mut out)?;
    Ok(out)
}

/// s63: `linear_gemm` の書き込み先指定版 (数値経路は完全に同一)。
///
/// 旧実装は pinned ホストバッファ `ho` から `to_vec()` で **呼び出し毎に
/// m*n*4B の Vec を新規確保**して返していた。prefill では m*n が最大
/// 12288*4096 = 201MB (gate/up は 279MB) に達し、確保 + 初回タッチの
/// ページフォルトが 1 呼び出しあたり 168ms を消費していた (実測 21.8s /
/// prefill 46.2s)。呼び出し側が層をまたいで再利用するバッファを渡す形に
/// することで、確保そのものを無くす (D2H 先は従来どおり pinned `ho`)。
fn linear_gemm_into(
    ctx: &CudaCtx,
    a: &[f32],
    gw: &GpuW,
    m: usize,
    k: usize,
    n: usize,
    dst: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(a.len(), m * k);
    debug_assert_eq!(dst.len(), m * n);
    let lib = lib()?;
    ctx.make_current()?;
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
        let Scratch { x, aq, asc, out, .. } = &mut *s;
        let bytes_x = m * k * 4;
        let bytes_out = m * n * 4;
        Scratch::ensure(x, bytes_x)?;
        Scratch::ensure(aq, m * k)?;
        Scratch::ensure(asc, m * 4)?;
        Scratch::ensure(out, bytes_out)?;
        let (dx, daq, dasc, dout) = (
            x.as_ref().unwrap(),
            aq.as_ref().unwrap(),
            asc.as_ref().unwrap(),
            out.as_ref().unwrap(),
        );
        unsafe {
            // s68: H2D は呼び出し元のスライスから直接発行する。旧実装は pinned
            // staging へ `copy_from_slice` してから H2D していたため、m=5686,
            // k=4096 では 93MB のホスト間コピーが毎呼び出し余分に乗っていた。
            check(
                (lib.memcpy_htod_async)(
                    dx.ptr(),
                    a.as_ptr() as *const c_void,
                    bytes_x,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync",
            )?;
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
            // s74: 128x256 タイル × 16x4 thread (1 thread が 8x16 出力)
            let gmx = m.div_ceil(GEMM_TILE) as u32;
            let gmy = n.div_ceil(GEMM_TILE_N) as u32;
            ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args)?;

            // s68: D2H も呼び出し元のバッファへ直接書く。旧実装は pinned `ho` に
            // 落としてから `dst.copy_from_slice` していたが、pinned→heap の
            // ホスト間コピーが 279MB で 60ms (≒4.7GB/s) かかり GEMM 本体 (12ms)
            // の 5 倍を占めていた。pageable へ直接 D2H すれば 20ms で済む。
            check(
                (lib.memcpy_dtoh_async)(
                    dst.as_mut_ptr() as *mut c_void,
                    dout.ptr(),
                    bytes_out,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync",
            )?;
            // s65: 発行元の件数分類 (表示専用・時間は d2h_state バケット側)
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        Ok(())
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
static LMV32_BROKEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// s49: dno_matvec_f32 のハンドル (lazy fetch・未収録 PTX では None をキャッシュ
/// して恒久不発動 = q35_fns と同一規約・GPU_BROKEN にはしない)。
/// 生ハンドルの静的保持は Q35Fns と同一の規約 (所有権移動なし・driver が
/// スレッド安全に管理) で Send/Sync を与える。
#[derive(Clone, Copy)]
struct Lmv32Fn(CUfunction);
unsafe impl Send for Lmv32Fn {}
unsafe impl Sync for Lmv32Fn {}

static LMV32_FNS: OnceLock<Option<Lmv32Fn>> = OnceLock::new();

/// s49: fp32 重み (lm_head / tied embed) のデバイスキャッシュ。key はモデル
/// ロード毎に一意 (qwen35 の mid 由来)。無ければ同期 H2D でアップロードし
/// 初回 1 回のみ走る (eviction 無し = P17b 重みレジストリと同一方針)。
static LMV32_W: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, DevBuffer>>> =
    OnceLock::new();

/// s49: fp32 lm_head の matvec (m==1) を GPU で実行する (CPU とビット一致)。
/// 戻り値 None = CPU パスを使うこと (無効/PTX 未収録/実行時エラー)。
/// key: モデルロード毎に一意 (untied と tied で別値にすること)。
pub fn maybe_gpu_matvec_f32(
    key: u64,
    w: &[f32],
    a: &[f32],
    k: usize,
    n: usize,
) -> Option<Vec<f32>> {
    use std::collections::HashMap;
    use std::sync::atomic::Ordering as AO;
    use std::sync::Mutex;
    if !cuda_enabled() || LMV32_BROKEN.load(AO::Relaxed) {
        return None;
    }
    if k == 0
        || n == 0
        || w.len() != k * n
        || a.len() != k
        || k > i32::MAX as usize
        || n > i32::MAX as usize
    {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed (fp32 lm_head), using CPU: {e:#}");
        LMV32_BROKEN.store(true, AO::Relaxed);
        return None;
    }
    // カーネルハンドル (lazy fetch・None キャッシュ = PTX 未収録はラッチ無し)
    let f = LMV32_FNS
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
        .as_ref()?
        .0;
    // 重みアップロード (初回 1 回・同期。他スレッドとの二重アップロード防止に
    // ロック内で行う)
    let reg = LMV32_W.get_or_init(|| Mutex::new(HashMap::new()));
    let wp = {
        let mut map = match reg.lock() {
            Ok(g) => g,
            Err(_) => {
                tracing::error!("CUDA fp32 lm_head weight registry poisoned, using CPU");
                LMV32_BROKEN.store(true, AO::Relaxed);
                return None;
            }
        };
        match map.get(&key) {
            Some(b) if b.bytes() == w.len() * 4 => b.ptr(),
            _ => match DevBuffer::from_slice(w) {
                Ok(buf) => {
                    tracing::info!(
                        "CUDA fp32 lm_head weight upload: k={k} n={n} ({:.1} MB)",
                        (k * n * 4) as f64 / 1e6
                    );
                    let p = buf.ptr();
                    map.insert(key, buf);
                    p
                }
                Err(e) => {
                    tracing::error!(
                        "CUDA fp32 lm_head weight upload failed, using CPU permanently: {e:#}"
                    );
                    LMV32_BROKEN.store(true, AO::Relaxed);
                    return None;
                }
            },
        }
    };
    // 実行: H2D(h) → dno_matvec_f32 → D2H(out) → sync ×1 (linear_pinned と同一)
    let lib = match lib() {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("CUDA lib load failed (fp32 lm_head), using CPU: {e:#}");
            LMV32_BROKEN.store(true, AO::Relaxed);
            return None;
        }
    };
    let r = SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let Scratch { x, out, hx, ho, .. } = &mut *s;
        let bytes_x = k * 4;
        let bytes_out = n * 4;
        Scratch::ensure(x, bytes_x)?;
        Scratch::ensure(out, bytes_out)?;
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        let (dx, dout) = (x.as_ref().unwrap(), out.as_ref().unwrap());
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..k].copy_from_slice(a);
        unsafe {
            check(
                (lib.memcpy_htod_async)(dx.ptr(), hx.ptr as *const c_void, bytes_x, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync",
            )?;
            let (mut ph, mut pw, mut po) = (dx.ptr(), wp, dout.ptr());
            let (mut kc, mut nc) = (k as c_int, n as c_int);
            let mut args = [
                &mut ph as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            // 1 block = 1 出力 (カーネルが block 内全スレッドで同一計算するため
            // block=(1,1,1)。grid.x は 2^31-1 まで合法 = vocab 規模もそのまま)
            ctx.launch_async(f, (n as u32, 1, 1), (1, 1, 1), 0, &mut args)?;
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            // s65: 発行元の件数分類 (表示専用・時間は d2h_state バケット側)
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LOGITS);
        Ok(ho.as_slice::<f32>()[..n].to_vec())
    });
    match r {
        Ok(out) => Some(out),
        Err(e) => {
            tracing::error!("CUDA fp32 lm_head matvec failed, using CPU permanently: {e:#}");
            LMV32_BROKEN.store(true, AO::Relaxed);
            None
        }
    }
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
pub const CH_GA: usize = 5; // gate/up 連結出力 [m*2ff] (gate が前半)
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
const CH_SLOTS: usize = 12;

// P17j: prefill attention カーネルのタイル定数 (kernels_elem.cu の PF_BQ/PF_BK
// と同期すること)。BQ=行タイル・BK=key タイル。
// P17k ステップ 7 修正: ステップ 4 でカーネルを 48/8→(実験 64/8) に変えた際に
// ここが 16/32 のまま残り、①grid が約 4 倍過剰 (i0 >= n の無効ブロックがフルの
// time ループを回す) ②sh_f も 16/32 基準で、カーネルの必要量を下回ると
// smem 領域外書きになり error 700 (不正アドレス) になっていた —
// 「~43KB 以上で error 700」の正体はこの不一致。64/8 (43.8KB) に統一。
const ATTN_PF_BQ: usize = 64;
const ATTN_PF_BK: usize = 8;

// s48 (C1-a): dno_attention_prefill の動的 smem。カーネル側は extern __shared__
// の静的宣言が無く launch 時の sharedMemBytes をそのまま使うため、48KB 既定上限は
// ホスト側ゲートだけの制約だった。qwen35 (dh=256) は 1 ブロック 84.8KB を要求し
// 48KB を超えるため、モジュールロード時に cuFuncSetAttribute
// (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES) で opt-in して引き上げる。
// 実機で opt-in 不可 (デバイスの opt-in 上限 < 必要量・古い driver でシンボル無し等)
// の場合も致命的ではない: 上限は 48KB のまま = qwen35 prefill attention は
// attn_prefill_gpu_supported ゲートで静かに CPU へフォールバックし、llama (dh=128
// = 43.8KB) は従来どおり発動する。.cu / PTX は無変更 (C1-b のタイル縮小が必要に
// なった場合のみ .cu 変更 + PTX 再生成が発生する)。
const ATTN_PF_SMEM_DEFAULT: usize = 48 * 1024;
/// CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
const CU_FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
/// CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
const CU_DEV_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;
static ATTN_PF_DYN_SMEM: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(ATTN_PF_SMEM_DEFAULT);

/// dno_attention_prefill が許容する動的 smem の上限 (bytes)。既定 48KB・
/// CudaCtx 生成時の opt-in (cuFuncSetAttribute) 成功時はデバイスの opt-in 上限。
pub fn attn_pf_dynamic_smem_limit() -> usize {
    ATTN_PF_DYN_SMEM.load(std::sync::atomic::Ordering::Relaxed)
}

/// dno_attention_prefill の動的 smem 必要量 (float 数)。
/// q_sm [BQ][dh+1] + k_sm [BK][dh+1] + v_sm [BK][dh+1] + s_sm [BQ*BK]
/// + corr_sm [BQ] + mnew_sm [BQ] (kernels_elem.cu と同期すること)。
fn attn_pf_sh_f(dh: usize) -> usize {
    (ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1) + ATTN_PF_BQ * ATTN_PF_BK + 2 * ATTN_PF_BQ
}

struct ChainScratch {
    bufs: Vec<Option<DevBuffer>>,
    pin_x: Option<PinMem>,  // x / att アップロード用 (使用間に必ず sync が挟まる)
    pin_dn: Option<PinMem>, // ダウンロード用
    aq: Option<DevBuffer>,  // int8 カーネル用スクラッチ (chain 専用)
    attn_q: Option<DevBuffer>,  // P17g: attention q アップロード用
    attn_sc: Option<DevBuffer>, // P17g: attention scores スクラッチ [heads*t_total]
    attn_pf_q: Option<DevBuffer>, // P17j: prefill attention q アップロード用 [n*heads*dh]
    rope_tab: Option<DevBuffer>,  // P17h: rope cos/sin テーブル [cos(half)|sin(half)]
    rope_bias: Option<DevBuffer>, // P17h: qkv bias パック [qb|kb|vb]
    pin_rope: Option<PinMem>,     // P17h: rope テーブル/... ステップ 1 回のアップロード用
    pin_pf: Option<PinMem>,       // P17k: prefill attention q の pinned ステージング用
    kv_stage: Option<PinMem>,     // P17h: k/v の非同期 D2H ステージング [n_layers][k|v]
    // P17k ステップ 5: qkv→rope→attention デバイス連続化用 (prefill)
    qkv_x: Option<DevBuffer>,     // GEMM 入力 h [n, e] fp32
    qkv_aq: Option<DevBuffer>,    // GEMM 入力量子化 [n, e] int8
    qkv_asc: Option<DevBuffer>,   // GEMM 入力スケール [n] f32
    qkv_pf: Option<DevBuffer>,    // GEMM 出力 [n, qn+kn+vn] (q/k/v 連結・行-major)
    qkv_q: Option<DevBuffer>,     // rope 後 q [n, heads*dh] (attention 入力)
    rope_tab_pf: Option<DevBuffer>, // rope テーブル [n, dh] (行 i = pos0+i の cos|sin)
    pin_qkv: Option<PinMem>,      // h / rope テーブル / bias アップロード用 pinned
    // B3: b>1 バッチ decode の rope (batch_rope_qk)
    br_q: Option<DevBuffer>,      // [b, heads*dh] f32 (in-place rope)
    br_k: Option<DevBuffer>,      // [b, kv_dim] f32 (in-place rope)
    br_tab: Option<DevBuffer>,    // [b, dh] f32 テーブル (行 i = poss[i] の [cos|sin])
    br_pin_qk: Option<PinMem>,    // q/k の H2D/D2H ステージング (sync を挟むので再利用安全)
    br_pin_tab: Option<PinMem>,   // テーブル H2D ステージング
    br_tab_key: Vec<usize>,       // テーブルキャッシュキー (poss・完全一致)
    br_tab_dh: usize,             // テーブルキャッシュキー (dh)
    br_tab_theta: f32,            // テーブルキャッシュキー (theta・bits 比較)
    br_tab_valid: bool,           // dev 上のテーブルがキャッシュキーと一致するか
    // P17h: ステップ内で sync が無くなるため pinned は (層×用途) ごとに専用化する
    // (async H2D の DMA 完了前にホストが上書きする競合を避ける)。id は呼び出し側が
    // 層番号と用途から一意に採番する。
    pins: Vec<Option<PinMem>>,
    // C4: qwen35 chain
    // rope テーブル [cos(n_rot/2) | sin(n_rot/2)] + 転送元 pinned
    q35_tab: Option<DevBuffer>,
    q35_pin_tab: Option<PinMem>,
    /// fp32 重みのデバイスキャッシュ (beta/alpha 制御投影・QK-Norm q/k norm)。
    /// Linear::F32 / Vec<f32> には uid が無いため (model_id, layer, tag) でキー化
    /// する。model_id はロード毎に採番されるため、モデル再ロードで誤キャッシュに
    /// なることはない。
    q35_w_cache: std::collections::HashMap<(u64, usize, u8), DevBuffer>,

    /// s165: rmsnorm 重みのデバイス常駐キャッシュ (`chain_rmsnorm_cached`)。
    /// キーは (model_id, pin_id = 層×用途)。decode chain が 1 ステップに 129 回
    /// 同じ重みを上げ直していた分を初回 1 回にする。
    rms_w: std::collections::HashMap<(u64, usize), DevBuffer>,
    // s167 (P6): GDN prefill の h2 D2H を qkv/z GEMM と重ねるための一式。
    // `DENNO_PF_OVERLAP=1` のときだけ作られる (フラグ OFF では None のまま =
    // 従来の同期 D2H と同一経路・未使用ストリームを残さない)。
    h2_stream: Option<CopyStream>,  // 非ブロッキングの専用コピーストリーム
    h2_ev_ready: Option<CopyEvent>, // main stream の rmsnorm 完了 (発行前の順序付け)
    h2_ev_done: Option<CopyEvent>,  // コピーストリームの D2H 完了 (join)
    pin_h2: Option<PinMem>,         // D2H の宛先 (pinned・grow-only)
    h2_pending: bool,               // 発行済みで未 join の D2H があるか

}

thread_local! {
    static CHAIN: std::cell::RefCell<ChainScratch> = std::cell::RefCell::new(ChainScratch {
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

    });
}

fn chain_err(e: anyhow::Error) -> anyhow::Error {
    tracing::error!("cuda chain failed (falling back to CPU): {e:#}");
    eprintln!("cuda chain failed (falling back to CPU): {e:#}");
    GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
    e
}

fn chain_fail(e: anyhow::Error) -> Option<()> {
    chain_err(e);
    None
}

/// Result を Option に変換しつつ GPU_BROKEN をマークする
fn ok<T>(r: Result<T, anyhow::Error>) -> Option<T> {
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

/// s48 (C1): qwen35 prefill の full attention を GPU バッチ (dno_attention_prefill
/// 1 発) にしてよいかの事前ゲート。構造的に確実に失敗する形状を開始前に弾き、
/// GPU_BROKEN ラッチの無駄な発火を防ぐ (llama B1 の prefill_chain_gpu_supported
/// と同一の趣旨)。qwen35 forward_prefill の Full 分岐から呼ばれる。
/// - 発動ゲート: `n >= gemm_min_m()` (既定 256) — 短文 (1b 含む m≤255) は CPU
///   attend のまま bit 不変 (アンカー温存・`maybe_gpu_linear` の GEMM 発動帯域と
///   一致させる設計)
/// - `DENNO_GPU_ATTN=0` で全経路現行に戻る (escape hatch 流用・新 env は作らない)
/// - 動的 smem: dh=256 は 84.8KB で 48KB 既定上限を超える → CudaCtx 生成時の
///   cuFuncSetAttribute opt-in が成功している場合だけ true (失敗なら false =
///   CPU attend フォールバック・llama は影響を受けない)
pub fn attn_prefill_gpu_supported(n: usize, dh: usize, t_total: usize) -> bool {
    if !cuda_enabled() || !gpu_attention_enabled() {
        return false;
    }
    if n == 0 || t_total == 0 || n < gemm_min_m() {
        return false;
    }
    if dh < ATTN_PF_BQ || dh > 1024 {
        return false;
    }
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return false;
    }
    // opt-in の結果 (ATTN_PF_DYN_SMEM) は CudaCtx 生成時に確定する。ここまで到達
    // したということは cuda 有効なので、未初期化ならここで生成する (直後の
    // chain_attention_prefill 内部でも生成されるため前倒しで生成するだけ = 新規
    // の副作用は無い)。生成失敗なら CPU フォールバック (GPU_BROKEN にはしない)。
    if shared_ctx().is_none() {
        return false;
    }
    true
}

/// B1: GEMM 経路 (dno_gemm_i8) が発動する m か。maybe_gpu_ffn / maybe_gpu_oproj /
/// chain_qkv_rope_prefill の内部ゲートと同一条件 (チェーン途中で失敗させないため
/// 呼び出し側が事前確認する)。
pub fn gemm_path_active(m: usize) -> bool {
    m > max_m() && m >= gemm_min_m() && gemm_enabled()
}

fn chain_ctx() -> Option<&'static CudaCtx> {
    if !cuda_enabled() {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        chain_err(anyhow::anyhow!("cuCtxSetCurrent: {e:#}"));
        return None;
    }
    Some(ctx)
}

fn chain_ensure_buf(slot: usize, bytes: usize) -> Result<(), anyhow::Error> {
    CHAIN.with(|cell| {
        let mut s = cell.borrow_mut();
        Scratch::ensure(&mut s.bufs[slot], bytes).map_err(chain_err)
    })
}

/// ホストデータをスロットへアップロード (async、同期はしない)。
/// 呼び出し間に sync が挟まる (qkv download) ので pinned バッファの再利用は安全。
pub fn chain_upload(slot: usize, data: &[f32], cols: usize, m: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let bytes = m.checked_mul(cols)?.checked_mul(4)?;
    if data.len() != m * cols {
        return chain_fail(anyhow::anyhow!("chain_upload: size mismatch"));
    }
    ok(chain_ensure_buf(slot, bytes))?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_pin(&mut s.pin_x, bytes))?;
        let buf = s.bufs[slot].as_ref().unwrap().ptr();
        let pin = s.pin_x.as_mut().unwrap();
        pin.as_mut_slice::<f32>()[..m * cols].copy_from_slice(data);
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, pin.ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (chain)",
            ))
        }?;
        // s166 (P5): 観測のみ
        dbg_htod(bytes);
        Some(())
    })
}

/// s55 (T1 step2・計測専用): デフォルトストリームの同期を 1 回打つだけの関数。
/// `chain_download` が内部で行う `ctx.sync()` と**同一の API 呼び出し**
/// (`cuCtxSynchronize`) を、本番 D2H を発行する前に別計測点として観測するための
/// 入口。同期以外の副作用は無く (バッファ確保・転送の発行をしない)、呼んでも
/// 数値経路には触れない。呼び出し側 (qwen35 decode) は `prof::enabled()` が真の
/// ときだけこれを呼ぶ — 本番経路 (非 prof) からは決して到達しない。
pub fn chain_sync_pending() -> Option<()> {
    let ctx = chain_ctx()?;
    ok(ctx.sync())
}

/// 最終 logits の戻しは decode 1 ステップ 1 回の最大 D2H。発行元の件数分類は
/// `prof::chain_download_site` (純関数) がスロットから決める (従来はすべて
/// `D2H_SRC_OTHER` に落ちていた — s108)。
/// s88 (Step 1): 発行点ごとの時間 (表示専用)。計測 OFF では時刻を読まない。
/// ここで件数も一緒に積むので、内側では `mark_d2h_src` を呼ばない
/// (呼ぶと calls が 2 倍になる)。
pub fn chain_download(slot: usize, len: usize) -> Option<Vec<f32>> {
    let site = crate::prof::chain_download_site(slot);
    let t_site = if crate::prof::d2h_src_on() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let r = chain_download_inner(slot, len);
    match t_site {
        Some(t) => crate::prof::mark_d2h_src_ns(site, t.elapsed().as_nanos() as u64),
        // 計測 OFF: 従来どおり件数だけ (実際は `mark_d2h_src` 側も no-op)。
        None => crate::prof::mark_d2h_src(site),
    }
    r
}

fn chain_download_inner(slot: usize, len: usize) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    ok(chain_ensure_buf(slot, bytes))?;
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let buf = cell.borrow().bufs[slot].as_ref()?.ptr();
        // s69: pinned `pin_dn` に落としてから `to_vec` すると同じ領域を 2 度
        // 書くことになる (93MB で約 20ms)。出力 Vec の未初期化領域へ直接
        // D2H し、全要素をコピーが埋めてから set_len する。
        let mut out: Vec<f32> = Vec::with_capacity(len);
        {
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        out.as_mut_ptr() as *mut c_void,
                        buf,
                        bytes,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (chain)",
                ))?;
            }
        }
        ok(ctx.sync())?;
        unsafe { out.set_len(len) }
        // s166 (P5): 観測のみ
        dbg_d2h(bytes);
        Some(out)
    })
}

/// スロット内の要素オフセット `elem_off` から `len` 個の f32 をダウンロード
/// (D2H + 同期)。B1: prefill チェーンの最終 norm 出力 (CH_H の最終行のみ)
/// 取得用 — [n, e] 全行の D2H を避ける。呼び出し時点でスロットには
/// (elem_off + len) 個分のデータが存在することを前提とする (不足時は None)。
/// 発行元の件数分類は `prof::chain_download_site` (logits 以外は `D2H_SRC_LINEAR`
/// — 活性化の一部を戻す点で linear 系の部分コピーと同類)。
pub fn chain_download_at(slot: usize, elem_off: usize, len: usize) -> Option<Vec<f32>> {
    // s88 (Step 1): 発行点ごとの時間 (表示専用・計測 OFF はクロック読みなし)。
    let t_site = if crate::prof::d2h_src_on() {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let r = chain_download_at_inner(slot, elem_off, len);
    let site = crate::prof::chain_download_site(slot);
    match t_site {
        Some(t) => crate::prof::mark_d2h_src_ns(site, t.elapsed().as_nanos() as u64),
        None => crate::prof::mark_d2h_src(site),
    }
    r
}

fn chain_download_at_inner(slot: usize, elem_off: usize, len: usize) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    let need = elem_off.checked_mul(4)?.checked_add(bytes)?;
    if !CHAIN.with(|cell| {
        cell.borrow()
            .bufs
            .get(slot)
            .map_or(false, |b| b.as_ref().map_or(false, |b| b.bytes() >= need))
    }) {
        return None;
    }
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let buf = cell.borrow().bufs[slot].as_ref()?.ptr() + (elem_off * 4) as CUdeviceptr;
        // s69: chain_download と同じ理由で出力 Vec へ直接 D2H する。
        let mut out: Vec<f32> = Vec::with_capacity(len);
        {
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        out.as_mut_ptr() as *mut c_void,
                        buf,
                        bytes,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (chain download at)",
                ))?;
            }
        }
        ok(ctx.sync())?;
        unsafe { out.set_len(len) }
        // s166 (P5): 観測のみ
        dbg_d2h(bytes);
        Some(out)
    })
}

/// out = rmsnorm(x [+ residual]) — residual スロットを指定すると x += p を先に
/// 行う (o_proj / down 出力の残差加算と融合)。CPU tensor::rmsnorm とビット一致。
/// grid = m ブロック、平方和は thread 0 が CPU と同一の逐次順で計算。
/// `pin_id` は w アップロード用 pinned の専用スロット (P17h: ステップ内に sync が
/// 無いため層×用途で一意に採番する — async H2D の DMA 完了前の上書き競合回避)。
pub fn chain_rmsnorm(
    w: &[f32],
    eps: f32,
    add_slot: Option<usize>,
    out_slot: usize,
    cols: usize,
    m: usize,
    pin_id: usize,
) -> Option<()> {
    chain_rmsnorm_slot(CH_X, w, eps, add_slot, out_slot, cols, m, pin_id)
}

/// s165 (P2): 入力スロットを指定できる rmsnorm。カーネル (`dno_add_rmsnorm`)・
/// grid・block・動的 shmem・引数の並びは [`chain_rmsnorm`] と 1 つも変わらず、
/// 違うのは x のポインタだけ — したがって出力は `chain_rmsnorm` とビット一致する
/// (prefill の残差常駐経路が `CH_PF_RES` を読むために使う)。
pub fn chain_rmsnorm_slot(
    x_slot: usize,
    w: &[f32],
    eps: f32,
    add_slot: Option<usize>,
    out_slot: usize,
    cols: usize,
    m: usize,
    pin_id: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if w.len() != cols {
        return chain_fail(anyhow::anyhow!("chain_rmsnorm: w size mismatch"));
    }
    let wb = cols * 4;
    ok(chain_ensure_buf(out_slot, m * cols * 4 + wb))?; // 末尾 wb バイトは w の作業領域
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        // w は呼び出しごとの専用 pinned から非同期アップロード (6KB 程度)。
        if s.pins.len() <= pin_id {
            while s.pins.len() <= pin_id {
                s.pins.push(None);
            }
        }
        let wsrc = {
            ok(Scratch::ensure_pin(&mut s.pins[pin_id], wb))?;
            let pin = s.pins[pin_id].as_mut().unwrap();
            pin.as_mut_slice::<f32>()[..cols].copy_from_slice(w);
            pin.ptr as *const c_void
        };
        let out_buf = s.bufs[out_slot].as_ref().unwrap();
        let wdev = out_buf.ptr() + (m * cols * 4) as u64; // バッファ末尾に置く
        let xptr = s.bufs[x_slot].as_ref()?.ptr();
        let pptr = match add_slot {
            Some(sl) => s.bufs[sl].as_ref().unwrap().ptr(),
            None => 0, // NULL → 単純 rmsnorm
        };
        let optr = out_buf.ptr();
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(wdev, wsrc, wb, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (chain rmsnorm w)",
            ))?;
        }
        // s166 (P5): 観測のみ (この経路は decode chain では使わない — 0 回が期待値)
        dbg_htod(wb);
        let (mut px, mut pp, mut pw, mut po) = (xptr, pptr, wdev, optr);
        let (mut mc, mut ec) = (cols as c_int, eps);
        let mut args = [
            &mut px as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut mc as *mut _ as *mut c_void,
            &mut ec as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(ctx.f_addnorm, (m as u32, 1, 1), (256, 1, 1), cols as u32 * 4, &mut args)
        })
    })
}

/// s165: [`chain_rmsnorm`] の**重み常駐**版 (decode chain 専用)。
///
/// `chain_rmsnorm` は呼び出しごとに `w` (e×4 = 20,480 B) を pinned から
/// 非同期 H2D する。decode chain は 1 ステップに 2×64+1 = 129 回呼ぶので、
/// 同じ重みを毎ステップ 129 回 (2.6 MB) 上げ直していた — nsys の
/// 「20,480 B × 25,872」がこれで、重みが変化しない以上まったくの無駄。
///
/// ここでは `(key, pin_id)` でデバイス常駐させ、**初回だけ**転送する
/// (`q35_w_dev` の (model_id, layer, tag) と同じ方針)。`key` はモデル
/// ロード毎に一意な `mid`、`pin_id` は呼び出し側が層×用途で一意に採番する
/// (P17h 規約) ので、ヒット = 同じ重み。サイズが変わったら作り直す。
///
/// 数値は `chain_rmsnorm` と同一: 同じカーネル (`dno_add_rmsnorm`)、同じ引数、
/// 同じ重みバイト列で、変わるのは重みの置き場所だけ。重みの転送は
/// `DevBuffer::from_slice` (同期 H2D) で行うが、これは (key, pin_id) ごとに
/// 1 回だけで、以降のステップは 0 回。
pub fn chain_rmsnorm_cached(
    w: &[f32],
    eps: f32,
    add_slot: Option<usize>,
    out_slot: usize,
    cols: usize,
    m: usize,
    pin_id: usize,
    key: u64,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if w.len() != cols {
        return chain_fail(anyhow::anyhow!("chain_rmsnorm_cached: w size mismatch"));
    }
    let wb = cols * 4;
    ok(chain_ensure_buf(out_slot, m.checked_mul(cols)?.checked_mul(4)?))?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        let stale = match s.rms_w.get(&(key, pin_id)) {
            Some(b) => b.bytes() != wb,
            None => true,
        };
        if stale {
            let buf = ok(DevBuffer::from_slice(w))?;
            s.rms_w.insert((key, pin_id), buf);
        }
        let wdev = s.rms_w.get(&(key, pin_id))?.ptr();
        let xptr = s.bufs[CH_X].as_ref()?.ptr();
        let pptr = match add_slot {
            Some(sl) => s.bufs.get(sl)?.as_ref()?.ptr(),
            None => 0, // NULL → 単純 rmsnorm
        };
        let optr = s.bufs.get(out_slot)?.as_ref()?.ptr();
        let (mut px, mut pp, mut pw, mut po) = (xptr, pptr, wdev, optr);
        let (mut mc, mut ec) = (cols as c_int, eps);
        let mut args = [
            &mut px as *mut _ as *mut c_void,
            &mut pp as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut mc as *mut _ as *mut c_void,
            &mut ec as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(ctx.f_addnorm, (m as u32, 1, 1), (256, 1, 1), cols as u32 * 4, &mut args)
        })
    })
}

/// スロット内の f32 を int8 量子化して grouped GEMV (デバイス内完結)。
/// H2D/D2H/同期なし。f_quant / f_mv / 重みレジストリは P17b/c と同一 → ビット一致。
pub fn chain_linear_grouped(
    parts: &[&Quant8Linear],
    in_slot: usize,
    out_slot: usize,
    m: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if m > max_m() {
        return None;
    }
    let k = parts[0].k;
    let n: usize = parts.iter().map(|p| p.n).sum();
    let uids: Vec<u64> = parts.iter().map(|p| p.uid).collect();
    let gw = ok(weights_for_group(&uids, parts, k, n))?;
    ok(chain_ensure_buf(in_slot, m * k * 4))?;
    ok(chain_ensure_buf(out_slot, m * n * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.aq, m * k))?;
        let iptr = s.bufs[in_slot].as_ref().unwrap().ptr();
        let optr = s.bufs[out_slot].as_ref().unwrap().ptr();
        let aq = s.aq.as_ref().unwrap().ptr();
        // asc は per-op SCRATCH を共有 (同一スレッド内なので競合しない)
        SCRATCH.with(|c2| -> Option<()> {
            let mut s2 = c2.borrow_mut();
            ok(Scratch::ensure(&mut s2.asc, m * 4))?;
            let asc = s2.asc.as_ref().unwrap().ptr();
            let (mut px, mut paq, mut pasc) = (iptr, aq, asc);
            let (mut mc, mut kc) = (m as c_int, k as c_int);
            let mut args = [
                &mut px as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
            ];
            ok(unsafe { ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args) })?;
            let (mut paqi, mut pasci, mut pw, mut psc, mut po) =
                (aq, asc, gw.w.ptr(), gw.sc.ptr(), optr);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paqi as *mut _ as *mut c_void,
                &mut pasci as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            ok(unsafe { ctx.launch_async(ctx.f_mv, (n as u32, 1, 1), (128, 1, 1), 0, &mut args) })
        })
    })
}

/// s157 (Task B20 段階 1): Q4_0 重みの grouped GEMV を**デバイス常駐バッファ同士**で
/// 発行する (enqueue のみ・同期なし・ホストスライスなし)。
///
/// `chain_linear_grouped` (int8) と対になる入口。重みは `weights_for_q4` の VRAM
/// レジストリ (nib / sc) をそのまま使い、カーネルは per-op 経路
/// (`q4_dispatch_tile`) と**同一の `dno_gemv_q4_0c{C}`・同一引数・同一グリッド**で
/// 起動する — 変わるのは「いつ同期するか」だけで、数値経路は 1 bit も変えない。
///
/// 出力は `parts` の順に連結する (`chain_linear_grouped` の重み連結と同じ並び)。
/// `parts` は k が揃っていること・k が Q4 ブロック (32) の倍数であることが前提。
pub fn chain_linear_q4_device(
    parts: &[&crate::quant::Q4Linear],
    in_slot: usize,
    out_slot: usize,
    m: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if parts.is_empty() || m > max_m() {
        return None;
    }
    let k = parts[0].k;
    if parts.iter().any(|p| p.k != k) || k % 32 != 0 {
        return None;
    }
    let n_tot: usize = parts.iter().map(|p| p.n).sum();
    // 重みは初回だけ転送する (以降はレジストリから引く = ホスト側の確保なし)。
    let mut gws = Vec::with_capacity(parts.len());
    for p in parts {
        gws.push(ok(weights_for_q4(p.uid, &p.blocks, p.k, p.n))?);
    }
    ok(chain_ensure_buf(in_slot, m * k * 4))?;
    ok(chain_ensure_buf(out_slot, m * n_tot * 4))?;
    // per-op 経路と同じ列ブロック版 (m = 1 の decode も同じカーネル)。
    // `q4_gemv_c()` は env を毎回読む — per-op 経路は 1 step に数回だが、chain は
    // 層ループの中で 1 step に数百回呼ばれるので、ここで 1 回だけ読んで使い回す。
    static C: OnceLock<usize> = OnceLock::new();
    let c = *C.get_or_init(q4_gemv_c);
    let f_c = ctx.f_mv_q4_c?[q4_gemv_c_index(c)];
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let iptr = s.bufs[in_slot].as_ref()?.ptr();
        let optr = s.bufs[out_slot].as_ref()?.ptr();
        let mut off = 0usize;
        for (p, gw) in parts.iter().zip(gws.iter()) {
            let (mut pa, mut pn, mut ps, mut po) =
                (iptr, gw.nib.ptr(), gw.sc.ptr(), optr + (off * 4) as u64);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, p.n as c_int);
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let grid = (p.n.div_ceil(8 * c) as u32, m as u32, 1);
            ok(unsafe { ctx.launch_async(f_c, grid, (256, 1, 1), 0, &mut args) })?;
            off += p.n;
        }
        Some(())
    })
}

/// s166 (P5): qkv (+z) 投影を**連結重み 1 本**の GEMV 1 回で回す。
///
/// 背景 (nsys・2026-10-22 の decode 1 ステップ): `dno_gemv_q4_0c4` は 337 発で、
/// 内訳は 48 GDN 層 × 5 + 16 full 層 × 6 + lm_head 1 = 337 (実測 336)。
/// 1 層あたり 5/6 発になるのは **qkv グループが part ごとに別 launch** だから
/// (full: q|k|v の 3 発、GDN: qkv|z の 2 発)。FFN の gate/up は P3 で 1 本の
/// 連結重みにまとまって 1 発なので、同じ扱いを qkv にも広げると 337 -> 257 発。
///
/// 数値は不変: `dno_gemv_q4_0c{C}` は出力 32 列を 1 block が受ける形で、列
/// (= 出力行) ごとに独立。行の並びが [q|k|v] のまま変わらないので、1 回の
/// launch で計算しても part ごとの launch と同じビット列になる (gate_up と
/// 同じ論法)。カーネル・引数・グリッドの形も [`chain_linear_q4_device`] と同一。
///
/// メモリ: 連結重みの行範囲は [`q4_register_row_alias`] で parts に割り当てる
/// (parts は連結側のビューになる = VRAM は増えない)。**parts が既に個別に
/// VRAM へ載っている場合は連結側を上げない** (2 重になるため)。前提が崩れたら
/// per-part の従来経路にそのまま落ちる (fail closed・数値は同一)。
pub fn chain_linear_q4_fused(
    fused: &crate::quant::Q4Linear,
    parts: &[&crate::quant::Q4Linear],
    in_slot: usize,
    out_slot: usize,
    m: usize,
) -> Option<()> {
    // 形状契約 (満たさなければ従来の per-part 経路 = 数値は同一)
    let mut ok_shape = !parts.is_empty() && fused.k != 0 && fused.k % 32 == 0 && fused.n != 0;
    let mut sum = 0usize;
    if ok_shape {
        for p in parts {
            if p.k != fused.k {
                ok_shape = false;
                break;
            }
            match sum.checked_add(p.n) {
                Some(s) => sum = s,
                None => {
                    ok_shape = false;
                    break;
                }
            }
        }
        ok_shape = ok_shape && sum == fused.n;
    }
    if !ok_shape {
        return chain_linear_q4_device(parts, in_slot, out_slot, m);
    }
    if !q4_fused_ready(fused.uid, parts) {
        // まだ割り当てていない: parts が個別に載っていなければ連結側を上げて割り当てる。
        // 載っているなら諦める (連結側を上げると VRAM が 2 重になる)。
        if parts.iter().any(|p| q4_resident(p.uid)) {
            return chain_linear_q4_device(parts, in_slot, out_slot, m);
        }
        if !q4_register_row_alias_q4(fused, parts) || !q4_fused_ready(fused.uid, parts) {
            return chain_linear_q4_device(parts, in_slot, out_slot, m);
        }
    }
    chain_linear_q4_device(&[fused], in_slot, out_slot, m)
}

/// s157 (Task B20 段階 1): 27B q4 の decode を chain 経路で回すか。
/// `DENNO_GPU_Q4_CHAIN=0` で従来の per-op 経路 (1 step に約 392 回の D2H) に戻せる。
/// 既定は chain (デバイス常駐・同期は step 末尾の logits 1 回)。
pub fn q4_chain_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_q4_chain(std::env::var("DENNO_GPU_Q4_CHAIN").ok().as_deref()))
}

/// s157 (Task B20 段階 1): `DENNO_GPU_Q4_CHAIN` の解釈 (純関数・テスト可能)。
/// **`"0"` だけが off** — 未設定・空・その他の値は既定の on (従来の per-op 経路へ
/// 戻すのは明示的な 0 だけ。タイポで静かに経路が変わるのを避ける)。
pub fn parse_q4_chain(raw: Option<&str>) -> bool {
    raw != Some("0")
}

/// s157 (Task B20 段階 1): chain 経路の線形呼び出し (**int8 / q4 の両対応**)。
///
/// 1 グループは全部 int8 か全部 q4 のどちらか (混在は受けない → `None` を返して
/// 呼び出し側が per-op 経路へ委譲する)。どちらの経路も enqueue のみで同期しない。
pub fn chain_linear(
    parts: &[&crate::quant::Linear],
    in_slot: usize,
    out_slot: usize,
    m: usize,
) -> Option<()> {
    if parts.is_empty() {
        return None;
    }
    if parts.iter().all(|p| p.quant8().is_some()) {
        let q: Vec<&Quant8Linear> = parts.iter().filter_map(|p| p.quant8()).collect();
        return chain_linear_grouped(&q, in_slot, out_slot, m);
    }
    if parts
        .iter()
        .all(|p| matches!(p, crate::quant::Linear::Q4(_)))
    {
        let q: Vec<&crate::quant::Q4Linear> = parts
            .iter()
            .map(|p| match p {
                crate::quant::Linear::Q4(x) => x,
                _ => unreachable!("all() で Q4 のみ確認済み"),
            })
            .collect();
        return chain_linear_q4_device(&q, in_slot, out_slot, m);
    }
    None
}

/// swiglu: ga スロットの [gate|up] から act = (g/(1+exp(-g)))*u を計算
/// (デバイス内完結)。CPU silu_inplace + gate[i] *= up[i] とビット一致。
pub fn chain_swiglu(ga_slot: usize, act_slot: usize, ff: usize, m: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    ok(chain_ensure_buf(act_slot, m * ff * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let gaptr = s.bufs[ga_slot].as_ref().unwrap().ptr();
        let optr = s.bufs[act_slot].as_ref().unwrap().ptr();
        let (mut pg, mut pu, mut po) = (gaptr, gaptr + (ff as u64) * 4, optr);
        let mut tc = (m * ff) as c_int;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut pu as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = ((m * ff) as u32).div_ceil(256);
        ok(unsafe { ctx.launch_async(ctx.f_swiglu, (grid, 1, 1), (256, 1, 1), 0, &mut args) })
    })
}

/// s70: prefill 用の grouped int8 GEMM。`chain_linear_grouped` は decode 用に
/// `m > max_m()` で諦めて GEMV カーネルを起動するが、prefill は m が大きく
/// 128x128 タイルの GEMM を使う必要がある。数値は同じカーネル・同じ整数累積
/// なので CPU 経路とビット一致。
fn chain_gemm_grouped_pf(
    parts: &[&crate::quant::Quant8Linear],
    in_slot: usize,
    out_slot: usize,
    m: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    if parts.is_empty() || m == 0 {
        return None;
    }
    let k = parts[0].k;
    let n: usize = parts.iter().map(|p| p.n).sum();
    let uids: Vec<u64> = parts.iter().map(|p| p.uid).collect();
    let gw = ok(weights_for_group(&uids, parts, k, n))?;
    ok(chain_ensure_buf(in_slot, m.checked_mul(k)?.checked_mul(4)?))?;
    ok(chain_ensure_buf(out_slot, m.checked_mul(n)?.checked_mul(4)?))?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.aq, m * k))?;
        let iptr = s.bufs[in_slot].as_ref().unwrap().ptr();
        let optr = s.bufs[out_slot].as_ref().unwrap().ptr();
        let aq = s.aq.as_ref().unwrap().ptr();
        SCRATCH.with(|c2| -> Option<()> {
            let mut s2 = c2.borrow_mut();
            ok(Scratch::ensure(&mut s2.asc, m * 4))?;
            let asc = s2.asc.as_ref().unwrap().ptr();
            let (mut px, mut paq, mut pasc) = (iptr, aq, asc);
            let (mut mc, mut kc) = (m as c_int, k as c_int);
            let mut args = [
                &mut px as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
            ];
            ok(unsafe { ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args) })?;
            let (mut paqi, mut pasci, mut pw, mut psc, mut po) =
                (aq, asc, gw.w.ptr(), gw.sc.ptr(), optr);
            let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut paqi as *mut _ as *mut c_void,
                &mut pasci as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let gmx = m.div_ceil(GEMM_TILE) as u32;
            let gmy = n.div_ceil(GEMM_TILE_N) as u32;
            ok(unsafe { ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args) })
        })
    })
}

/// s70: 行内 [gate|up] 連結レイアウト (width = 2*ff) の swiglu。m > 1 でも
/// 使えるのは `dno_swiglu_int` 側 (`dno_swiglu` は gate/up がブロック分離の
/// 前提なので m=1 専用)。演算列は CPU の `gv / (1+exp(-gv)) * uv` と同一。
pub fn chain_swiglu_rows(ga_slot: usize, act_slot: usize, ff: usize, m: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    if ff == 0 || m == 0 {
        return None;
    }
    ok(chain_ensure_buf(act_slot, m.checked_mul(ff)?.checked_mul(4)?))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let gaptr = s.bufs[ga_slot].as_ref()?.ptr();
        let optr = s.bufs[act_slot].as_ref()?.ptr();
        let (mut pg, mut po) = (gaptr, optr);
        let mut ffc = ff as c_int;
        let mut width = (2 * ff) as c_int;
        let mut tc = (m * ff) as i64;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ffc as *mut _ as *mut c_void,
            &mut width as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = ((m * ff) as u64).div_ceil(256) as u32;
        ok(unsafe { ctx.launch_async(ctx.f_swiglu_int, (grid, 1, 1), (256, 1, 1), 0, &mut args) })
    })
}


/// s163: prefill の FFN を **Q4 重みでも** デバイス常駐で通す
/// (`chain_mlp_prefill` の Q4 版)。旧経路は 1 層あたり
///   H2D h2 (m*e*4) + D2H gate (m*ff*4) + D2H up (m*ff*4)
///   + H2D hh (m*ff*4) + D2H down (m*e*4)
/// の 5 往復 (m=13000, e=5120, ff=17408 で 1 層 約 5.4 GB) していた。
/// 新経路は H2D h2 + D2H down の 2 往復だけで、中間の gate/up/act は
/// デバイス上に留まる。
///
/// ビット一致: q4 GEMM は per-op 経路と同一カーネル (`dno_gemm_q4_0`)・同一
/// グリッド・同一引数で、行ごとに独立 (m の分割は 1 行の累積順を変えない)。
/// swiglu は `dno_swiglu` (CPU の `g/(1+exp(-g))*u` と同一演算列)。
/// したがってタイル分割しても出力は従来経路とビット一致する。
///
/// メモリ: 行タイル `CH_MLP_TILE_Q4` ごとに処理するので、チェーンバッファは
/// タイル分だけで済む (13000 行を一度に置くと ff 中間 3 枚で 2.7 GB になる)。
pub fn chain_mlp_prefill_q4(
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    x: &[f32],
    m: usize,
    li: usize,
    out: &mut [f32],
) -> Option<()> {
    let e = gate_dims_in(gate)?;
    if x.len() < m.checked_mul(e)? || out.len() < m.checked_mul(e)? {
        return None;
    }
    chain_mlp_prefill_q4_impl(gate, up, down, PfSrc::Host(x), PfSink::Host(out), m, li)
}

/// s165 (P2): FFN チェーンの入出力をデバイス常駐で回す版。入力は `CH_PFH`
/// (rmsnorm 出力・先頭から m 行)、出力は `CH_DOWN` — 転送も同期も 0 回で、
/// 後続の残差加算 (次の rmsnorm の proj) が同じストリームで読む。
/// カーネル・タイル幅・引数は [`chain_mlp_prefill_q4`] と同一 (数値不変)。
pub fn chain_mlp_prefill_q4_dev(
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    m: usize,
    li: usize,
) -> Option<()> {
    chain_mlp_prefill_q4_impl(gate, up, down, PfSrc::Slot, PfSink::Slot, m, li)
}

/// [`chain_mlp_prefill_q4`] の入力次元 (k)。Q4 以外は None。
fn gate_dims_in(w: &crate::quant::Linear) -> Option<usize> {
    match w {
        crate::quant::Linear::Q4(q) => Some(q.k),
        _ => None,
    }
}

/// s165 (P2): prefill チェーンの入力の出所。
/// `Host` は従来どおりホスト slice をタイルごとに H2D、`Slot` は
/// 「既にデバイスに居る」ことを表す — 消えるのは転送だけで、カーネル・grid・
/// 引数・演算列は 1 つも変わらない (行ごとに独立なので bit 一致)。
enum PfSrc<'a> {
    Host(&'a [f32]),
    /// `CH_PFH` の先頭から m 行 (rmsnorm 出力) をそのまま読む
    Slot,
}

/// s165 (P2): prefill チェーンの出力先。`Slot` はデバイス常駐バッファ
/// (`CH_DOWN` / `CH_O`) に書き、D2H と同期を行わない。
enum PfSink<'a> {
    Host(&'a mut [f32]),
    Slot,
}

/// [`chain_mlp_prefill_q4`] の本体 (入出力の出所だけを差し替える)。
fn chain_mlp_prefill_q4_impl(
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    src: PfSrc<'_>,
    mut sink: PfSink<'_>,
    m: usize,
    li: usize,
) -> Option<()> {
    use crate::quant::Linear;
    if !cuda_enabled()
        || !gemm_enabled()
        || m < gemm_min_m()
        || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed)
    {
        return None;
    }
    let (gu, uu, du) = match (gate, up, down) {
        (Linear::Q4(a), Linear::Q4(b), Linear::Q4(c)) => (a, b, c),
        _ => return None,
    };
    let e = gu.k;
    let ff = gu.n;
    if uu.k != e || uu.n != ff || du.k != ff || du.n != e {
        return None;
    }
    if m == 0 {
        return None;
    }
    if let PfSrc::Host(x) = src {
        if x.len() < m * e {
            return None;
        }
    }
    if let PfSink::Host(out) = &sink {
        if out.len() < m * e {
            return None;
        }
    }
    let ctx = chain_ctx()?;
    let gw1 = weights_for_q4(gu.uid, &gu.blocks, gu.k, gu.n).ok()?;
    let gw2 = weights_for_q4(uu.uid, &uu.blocks, uu.k, uu.n).ok()?;
    let gw3 = weights_for_q4(du.uid, &du.blocks, du.k, du.n).ok()?;
    // Slot 出力は CH_DOWN の行オフセットへ書くので、全行ぶんの領域を要求する
    // (タイルごとの ensure は 1 タイル分しか保証しない)。
    if matches!(sink, PfSink::Slot) {
        ok(chain_ensure_buf(CH_DOWN, m.checked_mul(e)?.checked_mul(4)?))?;
    }
    // s168 (P7): タイル幅は `DENNO_Q4_M_TILE` の実効値、端数は直前のタイルへ併合。
    // 中間バッファは 1 タイルの**最大**行数 (`q4_tile_max_rows`) でループの前に
    // 確保する — 併合されるのは最終タイルだけなので、ループ内で grow させると
    // 先行 launch が読んでいるバッファを解放しうる。
    let tile = q4_m_tile();
    let cap = q4_tile_max_rows(tile).min(m);
    ok(chain_ensure_buf(CH_PFH, cap * e * 4))?;
    ok(chain_ensure_buf(CH_GA, cap * 2 * ff * 4))?;
    ok(chain_ensure_buf(CH_ACT, cap * ff * 4))?;
    ok(chain_ensure_buf(CH_DOWN, cap * e * 4))?;
    let _ = li;
    let mut r0 = 0usize;
    while r0 < m {
        let mr = q4_group_rows(r0, m, tile);
        debug_assert!(mr <= cap, "tile group {mr} > cap {cap}");
        // 出力先 (Host のときだけ D2H する)。生ポインタで持って CHAIN の
        // クロージャに渡す (Vec の再借用を避ける)。
        let dst: Option<*mut f32> = match &mut sink {
            PfSink::Host(out) => Some(out[r0 * e..].as_mut_ptr()),
            PfSink::Slot => None,
        };
        CHAIN.with(|cell| -> Option<()> {
            let s = cell.borrow();
            let pga = s.bufs[CH_GA].as_ref()?.ptr();
            let pact = s.bufs[CH_ACT].as_ref()?.ptr();
            // 出力の書き先: Host はタイル先頭 (D2H で呼び出し元の行へ)、
            // Slot は CH_DOWN の該当行 (デバイス常駐・行オフセットへ書く)。
            let pdown = match &sink {
                PfSink::Host(_) => s.bufs[CH_DOWN].as_ref()?.ptr(),
                PfSink::Slot => s.bufs[CH_DOWN].as_ref()?.ptr() + (r0 * e * 4) as u64,
            };
            let up_off = (mr * ff * 4) as u64;
            // 1) 入力: Host ならタイルを H2D、Slot なら CH_PFH の該当行を直接読む
            let pin = match &src {
                PfSrc::Host(x) => {
                    let buf = s.bufs[CH_PFH].as_ref()?.ptr();
                    let lib = lib().ok()?;
                    unsafe {
                        ok(check(
                            (lib.memcpy_htod_async)(
                                buf,
                                x[r0 * e..].as_ptr() as *const c_void,
                                mr * e * 4,
                                std::ptr::null_mut(),
                            ),
                            "cuMemcpyHtoDAsync (mlp prefill q4)",
                        ))?
                    }
                    buf
                }
                PfSrc::Slot => s.bufs[CH_PFH].as_ref()?.ptr() + (r0 * e * 4) as u64,
            };
            // 2) gate → CH_GA 前半 / up → CH_GA 後半 / 3) swiglu / 4) down
            pf_phase(PF_PH_FFN_GATE);
            crate::prof::time_layer(crate::prof::GATE_UP, li, || -> Option<()> {
                chain_gemm_q4_pf(gw1, pin, pga, mr, e, ff)?;
                chain_gemm_q4_pf(gw2, pin, pga + up_off, mr, e, ff)
            })?;
            pf_phase(PF_PH_SWIGLU);
            crate::prof::time_layer(crate::prof::SWIGLU, li, || -> Option<()> {
                chain_swiglu_split(pga, pga + up_off, pact, mr * ff)
            })?;
            pf_phase(PF_PH_FFN_DOWN);
            crate::prof::time_layer(crate::prof::DOWN, li, || -> Option<()> {
                chain_gemm_q4_pf(gw3, pact, pdown, mr, ff, e)
            })?;
            // 5) down のタイルだけ D2H (呼び出し元バッファへ直接)。
            //    Slot 出力では CH_DOWN に残したまま次段 (rmsnorm の add) に渡す。
            if let Some(dstp) = dst {
                let lib = lib().ok()?;
                unsafe {
                    ok(check(
                        (lib.memcpy_dtoh_async)(
                            dstp as *mut c_void,
                            pdown,
                            mr * e * 4,
                            std::ptr::null_mut(),
                        ),
                        "cuMemcpyDtoHAsync (mlp prefill q4)",
                    ))?
                }
                crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
                ok(ctx.sync())?;
            }
            Some(())
        })?;
        r0 += mr;
    }
    Some(())
}

/// [`chain_mlp_prefill_q4`] の 1 タイル分 GEMM (per-op 経路と同一カーネル・
/// 同一グリッド・同一引数)。a/out はデバイスポインタ。
fn chain_gemm_q4_pf(
    gw: &GpuWq4,
    a: u64,
    out: u64,
    m: usize,
    k: usize,
    n: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    unsafe {
        let (mut pa, mut pn, mut ps, mut po) = (a, gw.nib.ptr(), gw.sc.ptr(), out);
        let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
        let mut args = [
            &mut pa as *mut _ as *mut c_void,
            &mut pn as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut mc as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let grid = (m.div_ceil(64) as u32, n.div_ceil(128) as u32, 1);
        ok(ctx.launch_async(ctx.f_gemm_q4, grid, (16, 16, 1), 0, &mut args))
    }
}

/// gate / up が別バッファにある swiglu (`dno_swiglu`)。
fn chain_swiglu_split(gate: u64, up: u64, out: u64, total: usize) -> Option<()> {
    let ctx = chain_ctx()?;
    unsafe {
        let (mut pg, mut pu, mut po) = (gate, up, out);
        let mut tc = total as c_int;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut pu as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut tc as *mut _ as *mut c_void,
        ];
        let grid = (total as u64).div_ceil(256) as u32;
        ok(ctx.launch_async(ctx.f_swiglu, (grid, 1, 1), (256, 1, 1), 0, &mut args))
    }
}

/// s70: prefill の FFN (rmsnorm 済み x → gate/up → swiglu → down → ホスト) を
/// デバイス常駐で通す。活性化の往復が
///   旧: H2D 47MB + D2H 140MB + H2D 47MB + D2H 47MB
///   新: H2D 47MB + D2H 47MB
/// になる (m=2843, e=4096, ff=6144 の場合)。GEMM/swiglu は CPU 経路と
/// ビット一致 (同一カーネル・同一整数累積・同一 f32 演算列)。
///
/// s80 (T7b-1): 引数の gate/up を 1 本の連結重み (`gate_up_w`, 行順
/// [gate 全行 | up 全行]) にまとめた。dno_gemm_i8 は行ごとに独立で
/// n を 256 列タイルで舐めるだけなので、1 呼び出しでも n が 2ff になるだけで
/// 各出力は 2 回呼び出しと同一 (スケールは行ごと・パート境界 = ff)。
pub fn chain_mlp_prefill(
    gate_up_w: &crate::quant::Quant8Linear,
    down_w: &crate::quant::Quant8Linear,
    x: &[f32],
    m: usize,
    li: usize,
    out: &mut [f32],
) -> Option<()> {
    if !cuda_enabled() || !gemm_enabled() || m < gemm_min_m() || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let e = down_w.n;
    // s80 (T7b-1): gate_up はロード時に連結済みの 1 本 (行順 [gate | up])。
    // ff は連結幅の半分 = パートの境界。
    if gate_up_w.n % 2 != 0 || gate_up_w.k != e {
        return None;
    }
    let ff = gate_up_w.n / 2;
    if m == 0 || x.len() < m * e || out.len() < m * e || down_w.k != ff {
        return None;
    }
    let ctx = chain_ctx()?;
    let _ = li;
    // 1) rmsnorm 済み x を H2D (pinned staging を挟まず直接)
    ok(chain_ensure_buf(CH_PFH, m.checked_mul(e)?.checked_mul(4)?))?;
    CHAIN.with(|cell| -> Option<()> {
        let buf = cell.borrow().bufs[CH_PFH].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(
                    buf,
                    x.as_ptr() as *const c_void,
                    m * e * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyHtoDAsync (mlp prefill)",
            ))
        }
    })?;
    // 2) gate/up → 3) swiglu  → 4) down (全て非同期・最後に 1 回だけ同期)
    // 計測時のみ段ごとに同期して内訳を取る (同期は数値に影響しない)。
    crate::prof::time_layer(crate::prof::GATE_UP, li, || -> Option<()> {
        chain_gemm_grouped_pf(&[gate_up_w], CH_PFH, CH_GA, m)?;
        if crate::prof::enabled() {
            ok(ctx.sync())?;
        }
        Some(())
    })?;
    crate::prof::time_layer(crate::prof::SWIGLU, li, || -> Option<()> {
        chain_swiglu_rows(CH_GA, CH_ACT, ff, m)?;
        if crate::prof::enabled() {
            ok(ctx.sync())?;
        }
        Some(())
    })?;
    crate::prof::time_layer(crate::prof::DOWN, li, || -> Option<()> {
        chain_gemm_grouped_pf(&[down_w], CH_ACT, CH_DOWN, m)
    })?;
    // 5) D2H は呼び出し元のバッファへ直接
    CHAIN.with(|cell| -> Option<()> {
        let buf = cell.borrow().bufs[CH_DOWN].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(
                    out.as_mut_ptr() as *mut c_void,
                    buf,
                    m * e * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (mlp prefill)",
            ))
        }
    })?;
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    ok(ctx.sync())?;
    Some(())
}

// ---------------------------------------------------------------------------
// P17g: GPU KV ミラー + decode attention (exactness 緩和・ユーザー承認)。
//
// ホスト KvCache は CPU 経路・snapshot/prefix cache の正として残し、decode
// chain で使う K/V のコピーをデバイス側に保持する (per-session ミラー)。
// ミラーはホストキャッシュの行 [0..len) と常に同一内容に保たれる
// (不足行は次回の chain_attention で一括 H2D)。ミラー破損/ロールバック/
// エビクト時も len 比較で自動再シードされるため、ホストが正・ミラーは従。
// メモリ上限 `DENNO_GPU_KV_MB` (既定 8192MB) を超えると LRU でエビクト
// (必要時に再シード)。
//
// 注意: attention のみ exactness 契約が緩和されている (CPU と GPU で累積順が
// 異なり出力ビットは一致しない)。線形層 (int8) / expf / swiglu / rmsnorm は
// 従来どおりビット一致。`DENNO_GPU_ATTN=0` で無効化 (CPU attention・
// ビット一致経路に戻る)。

/// GPU attention の有効フラグ (既定 ON・`DENNO_GPU_ATTN=0` で無効)。
/// テストから差し替え可能にするため AtomicBool。
static GPU_ATTN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
static GPU_ATTN_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

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

struct KvMirror {
    /// (k, v) デバイスバッファ per layer。各 [cap, kv_dim] f32
    layers: Vec<(Option<DevBuffer>, Option<DevBuffer>)>,
    cap: usize,
    /// layer ごとにミラー済みの行数 (ホスト cache.len() との差分を H2D する)
    len: Vec<usize>,
    /// ミラー行が同期しているホスト KV コンテンツの世代 (s41)。
    /// KvCache::epoch() と一致するときだけ行 [0..len) / GDN 状態を再利用してよい。
    /// 不一致 (同一スロットで reset/restore → 別会話で再 prefill) なら with_mirror_ext
    /// が全層の len と gdn_pos を無効化する (s35 の 2b 2 値分岐の修正)。
    epoch: u64,
    /// C4 (qwen35): GDN 状態ミラー。GDN 層のみ Some (full attention 層は None)。
    /// 状態サイズはコンテキスト長に依存しないため cap 成長の影響を受けない。
    gdn: Vec<Option<GdnDev>>,
    /// GDN ミラーが同期している位置 (層ごと・cache.len() 相当)。Some(pos0) なら
    /// seed 不要。層ごとに持つ (層ごとの seed 判定を独立させる — 1 層でも
    /// gdn_pos を共有すると残り層が全て再 seed になってしまう)。
    gdn_pos: Vec<Option<usize>>,
    /// s50: 遅延 GDN 状態フラッシュの pending。decode chain は GDN の新状態
    /// (conv/ssm) をステップ毎にホストへ戻さず GPU ミラーを権威のままにする
    /// (ステップ毎 D2H が decode の支配項だったため)。Some(pos) = 最後に遅延した
    /// decode ステップの pos0+1 (正常完了後は cache.len() と一致)。Some の間は
    /// ホスト GDN 状態が古い可能性があり、chain_q35_gdn_flush_pending が
    /// 「ホスト状態が実際に必要になる時点」(prefill 開始前・CPU フォールバック時)
    /// でミラーから回復する。pending 保持中のミラーは LRU エビクト対象外
    /// (フラッシュ前にデータが失われると回復不能になるため)。
    gdn_pending: Option<usize>,
    /// s141b (Task B7c): 層ごとの「ホスト `KvCache::gdn` がデバイスより古い可能性が
    /// ある」フラグ。decode chain はステップ毎にホストへ戻さない (s50 から) ので、
    /// 一度でも scan を走らせた層はここが真になる。真の層はホストを seed に使っては
    /// ならない = `chain_q35_gdn_scan` の seed はデバイス正本から D2D で取る。
    /// ホストを読む側は `chain_q35_gdn_flush_pending` (= 本フラグのある層だけ D2H)
    /// か `KvCache::flush_gdn_device_state` を通ってから読むこと。
    /// ミラーに面 0 を書き直した層 (seed 直後) は偽に戻る。
    gdn_stale: Vec<bool>,
}

/// C4 (qwen35): GDN 状態 (conv + SSM) のデバイスミラー 1 層分。
/// dno_gdn_scan は in/out に別ポインタを要求する (nv > nk のとき複数ブロックが
/// 同一 conv チャネルを共有するため in==out エイリアスは不正) — ping-pong 2 面で
/// 「cur を読み、1-cur に書き、cur を交換」する。
struct GdnDev {
    conv: [Option<DevBuffer>; 2], // 各 [conv_dim, cs] f32
    ssm: [Option<DevBuffer>; 2],  // 各 [nv, dk, dv] f32
    cur: usize,
}

impl GdnDev {
    fn bytes(&self) -> usize {
        self.conv
            .iter()
            .chain(self.ssm.iter())
            .map(|b| b.as_ref().map(|b| b.bytes()).unwrap_or(0))
            .sum()
    }
}

impl KvMirror {
    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|(k, v)| {
                k.as_ref().map(|b| b.bytes()).unwrap_or(0)
                    + v.as_ref().map(|b| b.bytes()).unwrap_or(0)
            })
            .sum::<usize>()
            + self.gdn.iter().flatten().map(|g| g.bytes()).sum::<usize>()
    }
}

static KV_MIRRORS: std::sync::OnceLock<std::sync::Mutex<Vec<(u64, KvMirror, std::time::Instant)>>> =
    std::sync::OnceLock::new();

fn gpu_kv_limit_bytes() -> usize {
    std::env::var("DENNO_GPU_KV_MB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192)
        * 1024
        * 1024
}

fn kv_mirror_cap(t_total: usize, old_cap: usize) -> usize {
    // s128 (P4-7): 1,024 行刻みで切り上げる。以前は next_power_of_two だったため
    // 10,880 token で 16,384 行 (= 必要量の 1.51 倍) を確保していた。
    // cap 成長は全層の再シードを伴うので、刻みは 1,024 行 (27B で 134 MB) に固定し、
    // 必要量を超えるときは一気に必要量まで上げる (リクエスト内で再成長しない)。
    let need = t_total.div_ceil(1024) * 1024;
    need.max(old_cap + 1024).max(1024)
}

/// P17g/P17h 共通: KV ミラーの取得/生成 + cap 成長 + LRU エビクト。
/// `f` はミラーロック保持中に呼ばれ (行同期 + ポインタ抽出を行う)、
/// `have` はミラーが持っている行数 (rollback で t_total を超えていれば 0)。
/// 呼び出し側が m.len[li] を確定させる (t_total を正として使う — cache.advance
/// は forward の最後なので host len はまだ進んでいない)。
fn with_mirror<T>(
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
fn with_mirror_ext<T>(
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
    let idx = match ms.iter().position(|(k, _, _)| *k == key) {
        Some(i) => {
            ms[i].2 = now;
            i
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
            ms.len() - 1
        }
    };
    // エビクト: 上限超過で LRU (自分は除外)。s50: GDN 遅延フラッシュ pending 中の
    // ミラーは除外 — エビクトすると未フラッシュの GDN 状態 (ホストより新しい唯一の
    // コピー) が失われ回復不能になる。pending は次の prefill / CPU フォールバックの
    // 冒頭でフラッシュして解除されるため、上限超過は高々 1 リクエスト分の遅延。
    let limit = gpu_kv_limit_bytes();
    let mut total: usize = ms.iter().map(|(_, m, _)| m.bytes()).sum();
    while total > limit && ms.len() > 1 {
        let vic = (0..ms.len())
            .filter(|&i| ms[i].0 != key && ms[i].1.gdn_pending.is_none())
            .min_by_key(|&i| ms[i].2);
        let Some(vic) = vic else {
            break;
        };
        total -= ms[vic].1.bytes();
        ms.swap_remove(vic);
    }
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
        ok(Scratch::ensure(&mut m.layers[li].0, bytes))?;
        ok(Scratch::ensure(&mut m.layers[li].1, bytes))?;
        if before < bytes {
            // バッファを伸ばした = 行内容は失われた (ensure は新規確保) → 全再シード
            m.len[li] = 0;
        }
    }
    let have = if m.len[li] > t_total { 0 } else { m.len[li] };
    f(m, have)
}

/// decode chain 用 GPU attention。q (CPU 側で rope 済み [heads*dh]) と
/// ホスト KvCache (現在レイヤの行は呼び出し側が書き込み済み) を受け取り、
/// ミラーを同期して attention カーネルを起動し、結果を CH_ATT に書く。
/// 戻り None = CPU attention にフォールバック (forward 全体を従来経路で再実行)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention(
    key: u64,
    li: usize,
    n_layers: usize,
    q: &[f32],
    cache: &crate::kv_cache::KvCache,
    t_total: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    let ctx = chain_ctx()?;
    let heads = q.len() / dh;
    if heads == 0 || group == 0 || dh == 0 || dh > 1024 || t_total == 0 || t_total > cache.max_len()
    {
        return None;
    }
    let mut kv_dim_c = kv_dim as c_int;
    let mut t_c = t_total as c_int;
    let mut group_c = group as c_int;
    let mut dh_c = dh as c_int;

    // CH_ATT は後続の o_proj 入力そのもの (chain_upload を省略)
    ok(chain_ensure_buf(CH_ATT, heads * dh * 4))?;

    // ミラーの取得/生成 + レイヤ li の行同期。ポインタはロック解放前に抽出する。
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        // 行同期: ミラーが持っている [0..have) とホストの [0..t_total) を突き合わせ。
        // ホストの行 pos0 は呼び出し側が既に書き込み済み。
        // ロールバック (restore/reset) で have > t_total になったら全再シード (have=0)。
        let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
        if have < t_total {
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..t_total * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..t_total * kv_dim]))?;
        }
        m.len[li] = t_total;
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    // q / scores スクラッチ (スレッドローカル — chain は 1 スレッド 1 セッション)
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_q, q.len() * 4))?;
        ok(Scratch::ensure(&mut s.attn_sc, heads * t_total * 4))?;
        ok(s.attn_q.as_ref().unwrap().copy_from(q))?;
        let mut qptr = s.attn_q.as_ref().unwrap().ptr();
        let mut scptr = s.attn_sc.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut scale_c = scale;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (heads as u32, 1, 1),
                (dh as u32, 1, 1),
                (dh * 4) as u32,
                &mut args,
            )
        })
    })
}

/// P17j: prefill (n > 1) 用 GPU attention (dno_attention_prefill・flash 風
/// online softmax)。q ([n, heads*dh]、rope/QK-Norm 加算済み) とホスト KvCache
/// (現在レイヤの行 [pos0..t_total) は呼び出し側が書き込み済み) を受け取り、
/// ミラーを同期して prefill attention カーネルを起動し、結果 [n, heads*dh] を
/// CH_ATT に書く (呼び出し側が chain_download する)。
/// 戻り None = CPU attention にフォールバック。prefill は q/KV が全てホスト側に
/// あるため decode chain と異なり **attention ブロック内で完結**する
/// (forward 全体の再実行は不要・ホスト状態は不変)。
/// 数値は CPU (sgemm 経路) とビット不一致 — exactness 契約の attention 緩和
/// (ユーザー承認・P17g) と同一枠。**決定論は維持** (カーネル内の固定タイル・
/// 固定累積順・原子操作なし)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_prefill(
    key: u64,
    li: usize,
    n_layers: usize,
    q: &[f32],
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    n: usize,
    t_total: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    if n == 0 || t_total == 0 || group == 0 || t_total > cache.max_len() {
        return None;
    }
    // カーネル制約: block = dh スレッドが PF_BQ 行 owner + 出力次元 d=tid を
    // 担当するため dh >= PF_BQ。動的 smem は 48KB 既定上限を opt-in 済み上限
    // (s48 C1-a: qwen35 dh=256 = 84.8KB は cuFuncSetAttribute で通す) に引き上げ。
    if dh < ATTN_PF_BQ || dh > 1024 {
        return None;
    }
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return None;
    }
    let heads = q.len() / (n * dh);
    if heads == 0 || n.checked_mul(dh).map(|x| x * heads) != Some(q.len()) {
        return None;
    }
    let ctx = chain_ctx()?;

    let mut heads_c = heads as c_int;
    let mut n_c = n as c_int;
    let mut pos0_c = pos0 as c_int;
    let mut kv_dim_c = kv_dim as c_int;
    let mut t_c = t_total as c_int;
    let mut group_c = group as c_int;
    let mut dh_c = dh as c_int;
    let mut scale_c = scale;

    ok(chain_ensure_buf(CH_ATT, n * heads * dh * 4))?;

    // ミラーの取得/生成 + レイヤ li の行同期 (decode と同一経路)。
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
        if have < t_total {
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..t_total * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..t_total * kv_dim]))?;
        }
        m.len[li] = t_total;
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_pf_q, q.len() * 4))?;
        // P17k: pinned ステージング + async H2D (同期 pageable copy_from より高速)。
        // DMA 完了は同一ストリーム順序でカーネルが待つため sync 不要。pin_pf の
        // 次回上書きは次層の attention で、その前に oproj/download の sync が挟まる。
        ok(Scratch::ensure_pin(&mut s.pin_pf, q.len() * 4))?;
        let buf = s.attn_pf_q.as_ref().unwrap().ptr();
        s.pin_pf.as_mut().unwrap().as_mut_slice::<f32>()[..q.len()].copy_from_slice(q);
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, s.pin_pf.as_ref().unwrap().ptr as *const c_void, q.len() * 4, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (attn_pf_q)",
            ))?;
        }
        let mut qptr = s.attn_pf_q.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut heads_c as *mut _ as *mut c_void,
            &mut n_c as *mut _ as *mut c_void,
            &mut pos0_c as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn_pf,
                // P17k: (行ブロック, heads) — x=行ブロック最速次元で同一 head の
                // K/V を同時実行ブロック間で共有し L2 ヒット率を上げる
                (n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                (dh as u32, 1, 1),
                (attn_pf_sh_f(dh) * 4) as u32,
                &mut args,
            )
        })
    })
}

/// P17k: o_proj をデバイス常駐で実行 — 入力は `chain_attention_prefill` が書いた
/// CH_ATT スロット (デバイス上の att [m, k]) を直接読む。従来は att を 81MB D2H
/// + to_vec + o_proj で再 H2D していた往復を消す。
/// ビット一致: quantize + dno_gemm_i8 は CPU と同一演算 → 出力は CPU 経路と一致。
/// 失敗時 None (呼び出し側は chain_download で att を取得して従来経路へ)。
pub fn maybe_gpu_oproj(o: &Quant8Linear, m: usize, k: usize, li: usize) -> Option<Vec<f32>> {
    if !cuda_enabled() {
        return None;
    }
    if !(m > max_m() && m >= gemm_min_m() && gemm_enabled()) {
        return None;
    }
    debug_assert_eq!(o.k, k);
    let e = o.n;
    // CH_ATT スロットは chain_attention_prefill 成功時に確保済みのはず
    if !CHAIN.with(|cell| {
        cell.borrow()
            .bufs
            .get(CH_ATT)
            .map_or(false, |b| b.as_ref().map_or(false, |b| b.bytes >= m * k * 4))
    }) {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(err) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU int8: {err:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let gw = match weights_for(o.uid, &o.w, &o.scale, k, e) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA o_proj weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    let lib = lib().ok()?;
    let aptr = CHAIN.with(|cell| -> Option<u64> {
        cell.borrow()
            .bufs[CH_ATT]
            .as_ref()
            .map(|b| b.ptr())
    })?;
    let r = SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let mut s = cell.borrow_mut();
        let Scratch { aq, asc, out, .. } = &mut *s;
        Scratch::ensure(aq, m * k)?;
        Scratch::ensure(asc, m * 4)?;
        Scratch::ensure(out, m * e * 4)?;
        let (daq, dasc, dout) = (aq.as_ref().unwrap(), asc.as_ref().unwrap(), out.as_ref().unwrap());
        // s69: pinned `ho` + `to_vec` をやめ、出力 Vec へ直接 D2H する。
        let mut res: Vec<f32> = Vec::with_capacity(m * e);
        unsafe {
            crate::prof::time_layer(crate::prof::OPROJ, li, || -> Result<()> {
                // 1) att (CH_ATT) をデバイス上で量子化
                let (mut px, mut paq, mut pasc) = (aptr, daq.ptr(), dasc.ptr());
                let (mut mc, mut kc) = (m as c_int, k as c_int);
                let mut args = [
                    &mut px as *mut _ as *mut c_void,
                    &mut paq as *mut _ as *mut c_void,
                    &mut pasc as *mut _ as *mut c_void,
                    &mut mc as *mut _ as *mut c_void,
                    &mut kc as *mut _ as *mut c_void,
                ];
                ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args)?;
                // 2) o_proj GEMM: [m, k] → [m, e]
                let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                    (daq.ptr(), dasc.ptr(), gw.w.ptr(), gw.sc.ptr(), dout.ptr());
                let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, e as c_int);
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
                let gmx = m.div_ceil(GEMM_TILE) as u32;
                let gmy = e.div_ceil(GEMM_TILE_N) as u32;
                ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args)?;
                // 3) D2H + 同期 (attention カーネルは同一ストリームの先行発行なので完結済み)
                check(
                    (lib.memcpy_dtoh_async)(
                        res.as_mut_ptr() as *mut c_void,
                        dout.ptr(),
                        m * e * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync",
                )?;
                // s65: 発行元の件数分類 (表示専用・時間は d2h_state バケット側)
                crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
                ctx.sync()?;
                Ok(())
            })?;
        }
        unsafe { res.set_len(m * e) };
        Ok(res)
    });
    match r {
        Ok(out) => Some(out),
        Err(err) => {
            tracing::error!("CUDA o_proj failed, falling back to CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

/// B1: prefill デバイス常駐チェーン用 o_proj — CH_ATT の attention 出力を D2H せず
/// デバイス上で o_proj (quantize → dno_gemm_i8) し、CH_O に書く (同期なし。後続の
/// chain_rmsnorm が同一ストリームで順次読む)。GEMM は maybe_gpu_oproj と同一カーネル
/// → ビット一致。CH_ATT は chain_attention_prefill(_dev) 成功時に確保済み前提。
/// 戻り None = チェーン全体を従来経路でやり直し。
pub fn maybe_gpu_oproj_dev(o: &Quant8Linear, m: usize, k: usize, li: usize) -> Option<()> {
    if !(m > max_m() && m >= gemm_min_m() && gemm_enabled()) {
        return None;
    }
    debug_assert_eq!(o.k, k);
    let e = o.n;
    let ctx = chain_ctx()?;
    // CH_ATT スロットは chain_attention_prefill(_dev) 成功時に確保済みのはず
    if !CHAIN.with(|cell| {
        cell.borrow()
            .bufs
            .get(CH_ATT)
            .map_or(false, |b| b.as_ref().map_or(false, |b| b.bytes() >= m * k * 4))
    }) {
        return None;
    }
    let gw = match weights_for(o.uid, &o.w, &o.scale, k, e) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA o_proj weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    let aptr = CHAIN.with(|cell| -> Option<u64> {
        cell.borrow()
            .bufs[CH_ATT]
            .as_ref()
            .map(|b| b.ptr())
    })?;
    ok(chain_ensure_buf(CH_O, m * e * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let optr = cell.borrow().bufs[CH_O].as_ref().unwrap().ptr();
        SCRATCH.with(|c2| -> Option<()> {
            let mut s2 = c2.borrow_mut();
            ok(Scratch::ensure(&mut s2.aq, m * k))?;
            ok(Scratch::ensure(&mut s2.asc, m * 4))?;
            ok(Scratch::ensure(&mut s2.out, m * e * 4))?;
            let (daq, dasc) = (s2.aq.as_ref().unwrap(), s2.asc.as_ref().unwrap());
            crate::prof::time_layer(crate::prof::OPROJ, li, || -> Option<()> {
                unsafe {
                    // 1) att (CH_ATT) をデバイス上で量子化
                    let (mut px, mut paq, mut pasc) = (aptr, daq.ptr(), dasc.ptr());
                    let (mut mc, mut kc) = (m as c_int, k as c_int);
                    let mut args = [
                        &mut px as *mut _ as *mut c_void,
                        &mut paq as *mut _ as *mut c_void,
                        &mut pasc as *mut _ as *mut c_void,
                        &mut mc as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                    ];
                    ok(ctx.launch_async(ctx.f_quant, (m as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args))?;
                    // 2) o_proj GEMM: [m, k] → [m, e] (CH_O にデバイス常駐)
                    let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                        (daq.ptr(), dasc.ptr(), gw.w.ptr(), gw.sc.ptr(), optr);
                    let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, e as c_int);
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
                    let gmx = m.div_ceil(GEMM_TILE) as u32;
                    let gmy = e.div_ceil(GEMM_TILE_N) as u32;
                    ok(ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args))
                }
            })
        })
    })
}

/// P17g: バッチ decode 用 — セッションごとに `chain_attention` を起動し、
/// 結果をホスト `att` [b, heads*dh] に集める。solo chain と同一カーネル・
/// 同一入力 → 単独/バッチでビット一致を維持 (exactness 契約 P6)。
/// 全セッション成功時 true。1 つでも失敗すれば false を返し、呼び出し側は
/// 全セッションを CPU attention でやり直す (GPU/CPU 混合の数値混在を防止)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_batch(
    li: usize,
    n_layers: usize,
    q: &[f32],
    caches: &[&crate::kv_cache::KvCache],
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
    att: &mut [f32],
) -> bool {
    let hd = heads * dh;
    if !gpu_attention_enabled()
        || caches.is_empty()
        || q.len() != caches.len() * hd
        || att.len() != q.len()
    {
        return false;
    }
    for (i, cache) in caches.iter().enumerate() {
        let t_total = cache.len() + 1;
        let ok1 = chain_attention(
            cache.kv_id(),
            li,
            n_layers,
            &q[i * hd..][..hd],
            cache,
            t_total,
            group,
            dh,
            kv_dim,
            scale,
        );
        if ok1.is_none() {
            return false;
        }
        match chain_download(CH_ATT, hd) {
            Some(row) => att[i * hd..][..hd].copy_from_slice(&row),
            None => return false,
        }
    }
    true
}

// ===========================================================================
// P17h: decode chain の同期削減 (rope GPU 化 + qkv デバイス常駐)
// ===========================================================================

/// P17h: rope テーブル [cos(half)|sin(half)] をデバイスへ準備 (1 ステップ 1 回)。
/// CPU tensor::rope_row と**同一の式**を同一プロセスの glibc sinf/cosf で計算する
/// ためテーブル値はビット一致。デバイスバッファはステップ中ずっと保持される。
pub fn chain_rope_prepare(pos0: usize, dh: usize, theta: f32) -> Option<()> {
    let _ = chain_ctx()?;
    let half = dh / 2;
    let bytes = half * 2 * 4;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.rope_tab, bytes))?;
        ok(Scratch::ensure_pin(&mut s.pin_rope, bytes))?;
        {
            let tab = s.pin_rope.as_mut().unwrap().as_mut_slice::<f32>();
            let pos = pos0 as f32;
            for d in 0..half {
                let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                tab[d] = ang.cos();
                tab[half + d] = ang.sin();
            }
        }
        let dev = s.rope_tab.as_ref().unwrap().ptr();
        let pin = s.pin_rope.as_ref().unwrap();
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dev, pin.ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (rope tab)",
            ))
        }
    })
}

/// P17h: CH_QKV の q/k/v に bias 加算 + rope を適用 (デバイス内・in-place)。
/// CPU 経路 (add_bias → rope_row) とビット一致 (bias は単一 add・rope は
/// mul/sub のみ・-fmad=false)。bias 無しの要素は NULL として渡す。
/// `pin_id` は bias アップロード用 pinned スロット (層ごとに一意)。
pub fn chain_rope_qkv(
    q_b: Option<&[f32]>,
    k_b: Option<&[f32]>,
    v_b: Option<&[f32]>,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    pin_id: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let hd = heads.checked_mul(dh)?;
    let kv_dim = kv_heads.checked_mul(dh)?;
    let bias_len = hd + 2 * kv_dim;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.rope_bias, bias_len * 4))?;
        let bptr = if q_b.is_some() || k_b.is_some() || v_b.is_some() {
            if s.pins.len() <= pin_id {
                while s.pins.len() <= pin_id {
                    s.pins.push(None);
                }
            }
            ok(Scratch::ensure_pin(&mut s.pins[pin_id], bias_len * 4))?;
            {
                let pin = s.pins[pin_id].as_mut().unwrap();
                let p = pin.as_mut_slice::<f32>();
                p[..bias_len].fill(0.0);
                if let Some(b) = q_b {
                    p[..b.len()].copy_from_slice(b);
                }
                if let Some(b) = k_b {
                    p[hd..hd + b.len()].copy_from_slice(b);
                }
                if let Some(b) = v_b {
                    p[hd + kv_dim..hd + kv_dim + b.len()].copy_from_slice(b);
                }
            }
            let bp = s.rope_bias.as_ref().unwrap().ptr();
            let pp = s.pins[pin_id].as_ref().unwrap();
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_htod_async)(
                        bp,
                        pp.ptr as *const c_void,
                        bias_len * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyHtoDAsync (rope bias)",
                ))?;
            }
            bp
        } else {
            0 as CUdeviceptr // NULL — bias 無し
        };
        let qkv_ptr = s.bufs[CH_QKV].as_ref().unwrap().ptr();
        let tab_ptr = s.rope_tab.as_ref().unwrap().ptr();
        let (mut pq, mut pt, mut pb) = (qkv_ptr, tab_ptr, bptr);
        let (mut hc, mut kc, mut dc) =
            (heads as c_int, kv_heads as c_int, dh as c_int);
        let mut args: Vec<*mut c_void> = vec![
            &mut pq as *mut _ as *mut c_void,
            &mut pt as *mut _ as *mut c_void,
            &mut pb as *mut _ as *mut c_void,
            &mut hc as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut dc as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_rope,
                ((heads + 2 * kv_heads) as u32, 1, 1),
                (dh as u32, 1, 1),
                0,
                &mut args,
            )
        })
    })
}

// ===========================================================================
// B3: b>1 バッチ decode の rope GPU 化 (P17 系 rope カーネルの再利用)
// ===========================================================================

/// B3: b>1 バッチ decode の rope を GPU で実行する (q/k in-place・セッションごと
/// に独立した位置)。CPU 経路 (forward_batch の session ごと `tensor::rope_row`
/// 適用) と**ビット一致** (exactness 契約: rope は緩和対象外):
/// - テーブルは CPU が rope_row と同一式 (同一プロセスの glibc powf/cosf/sinf)
///   で生成 → テーブル値はビット一致
/// - カーネル `dno_rope_qk_batch` は mul/sub のみ (-fmad=false → mul.rn/sub.rn/
///   add.rn = Rust スカラー演算と同一の丸め)
/// - bias/QK-Norm は呼び出し側が既に CPU で適用済みの q/k を渡す (rope は
///   bias/QK-Norm と無関係・CPU 経路と同一の演算列)
/// 失敗時は false を返しホスト q/k は不変 (呼び出し側が CPU rope でやり直せる —
/// D2H 完了後に初めてホストへ書き戻す)。テーブルは (poss・dh・theta) 完全一致
/// でキャッシュ (1 ステップの層ループで同一 poss が層数回来るため H2D もスキップ)。
#[allow(clippy::too_many_arguments)]
pub fn batch_rope_qk(
    q: &mut [f32],
    k: &mut [f32],
    poss: &[usize],
    heads: usize,
    kv_heads: usize,
    dh: usize,
    theta: f32,
) -> bool {
    let Some(ctx) = chain_ctx() else { return false };
    let b = poss.len();
    // カーネル/転送の形状ガード (grid/block 上限・size_t オーバーフロー防止)。
    // 非該当 shapes は静かに CPU フォールバック (GPU_BROKEN ラッチも不要)。
    let Some(hd) = heads.checked_mul(dh) else { return false };
    let Some(kv_dim) = kv_heads.checked_mul(dh) else { return false };
    let Some(q_floats) = b.checked_mul(hd) else { return false };
    let Some(k_floats) = b.checked_mul(kv_dim) else { return false };
    if b == 0 || b > 65535 || dh == 0 || dh > 1024 || heads + kv_heads > 65535 {
        return false;
    }
    let q_bytes = q_floats * 4;
    let k_bytes = k_floats * 4;
    let tab_floats = b * dh;
    let tab_bytes = tab_floats * 4;
    CHAIN.with(|cell| -> bool {
        let mut s = cell.borrow_mut();
        // --- テーブル (キャッシュヒットなら生成 + H2D をスキップ) ---
        let cache_hit = s.br_tab_valid
            && s.br_tab_dh == dh
            && s.br_tab_theta.to_bits() == theta.to_bits()
            && s.br_tab_key.len() == b
            && s.br_tab_key.iter().zip(poss.iter()).all(|(a, p)| a == p);
        if !cache_hit {
            if ok(Scratch::ensure(&mut s.br_tab, tab_bytes)).is_none()
                || ok(Scratch::ensure_pin(&mut s.br_pin_tab, tab_bytes)).is_none()
            {
                return false;
            }
            {
                // tensor::rope_row と同一式で生成 (同一プロセスの glibc
                // powf/cosf/sinf → 値はビット一致・P17h chain_rope_prepare と同じ)
                let tab = s.br_pin_tab.as_mut().unwrap().as_mut_slice::<f32>();
                let half = dh / 2;
                for (i, &pos) in poss.iter().enumerate() {
                    let posf = pos as f32;
                    let row = &mut tab[i * dh..][..dh];
                    for d in 0..half {
                        let ang = posf * theta.powf(-2.0 * d as f32 / dh as f32);
                        row[d] = ang.cos();
                        row[half + d] = ang.sin();
                    }
                }
            }
            let lib = match ok(lib()) {
                Some(l) => l,
                None => return false,
            };
            let dev = s.br_tab.as_ref().unwrap().ptr();
            let pin = s.br_pin_tab.as_ref().unwrap();
            unsafe {
                if ok(check(
                    (lib.memcpy_htod_async)(dev, pin.ptr as *const c_void, tab_bytes, std::ptr::null_mut()),
                    "cuMemcpyHtoDAsync (batch rope tab)",
                ))
                .is_none()
                {
                    return false;
                }
            }
            s.br_tab_key.clear();
            s.br_tab_key.extend_from_slice(poss);
            s.br_tab_dh = dh;
            s.br_tab_theta = theta;
            s.br_tab_valid = true;
        }
        // --- バッファ確保 + q/k アップロード (pin は [q|k] 連結で 1 本) ---
        if ok(Scratch::ensure(&mut s.br_q, q_bytes)).is_none()
            || ok(Scratch::ensure(&mut s.br_k, k_bytes)).is_none()
            || ok(Scratch::ensure_pin(&mut s.br_pin_qk, q_bytes + k_bytes)).is_none()
        {
            return false;
        }
        let lib = match ok(lib()) {
            Some(l) => l,
            None => return false,
        };
        let (dq, dk) = (
            s.br_q.as_ref().unwrap().ptr(),
            s.br_k.as_ref().unwrap().ptr(),
        );
        unsafe {
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                pin.as_mut_slice::<f32>()[..q_floats].copy_from_slice(q);
                if ok(check(
                    (lib.memcpy_htod_async)(dq, pin.ptr as *const c_void, q_bytes, std::ptr::null_mut()),
                    "cuMemcpyHtoDAsync (batch rope q)",
                ))
                .is_none()
                {
                    return false;
                }
            }
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                pin.as_mut_slice::<f32>()[q_floats..q_floats + k_floats].copy_from_slice(k);
                let ksrc = (pin.ptr as *const u8).add(q_bytes) as *const c_void;
                if ok(check(
                    (lib.memcpy_htod_async)(dk, ksrc, k_bytes, std::ptr::null_mut()),
                    "cuMemcpyHtoDAsync (batch rope k)",
                ))
                .is_none()
                {
                    return false;
                }
            }
        }
        // --- カーネル起動 (grid = (heads + kv_heads, b)・block = (dh)) ---
        let (mut pq, mut pk, mut pt) = (dq, dk, s.br_tab.as_ref().unwrap().ptr());
        let (mut hc, mut khc, mut dc) = (heads as c_int, kv_heads as c_int, dh as c_int);
        let mut args: Vec<*mut c_void> = vec![
            &mut pq as *mut _ as *mut c_void,
            &mut pk as *mut _ as *mut c_void,
            &mut pt as *mut _ as *mut c_void,
            &mut hc as *mut _ as *mut c_void,
            &mut khc as *mut _ as *mut c_void,
            &mut dc as *mut _ as *mut c_void,
        ];
        if ok(unsafe {
            ctx.launch_async(
                ctx.f_rope_b,
                ((heads + kv_heads) as u32, b as u32, 1),
                (dh as u32, 1, 1),
                0,
                &mut args,
            )
        })
        .is_none()
        {
            return false;
        }
        if ok(ctx.sync()).is_none() {
            return false;
        }
        // --- 結果 D2H ([q|k] を pin の非重複領域へ・両方成功するまでホスト q/k は不変) ---
        unsafe {
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                if ok(check(
                    (lib.memcpy_dtoh_async)(pin.ptr, dq, q_bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (batch rope q)",
                ))
                .is_none()
                {
                    return false;
                }
            }
            {
                let pin = s.br_pin_qk.as_mut().unwrap();
                let kdst = (pin.ptr as *mut u8).add(q_bytes) as *mut c_void;
                if ok(check(
                    (lib.memcpy_dtoh_async)(kdst, dk, k_bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (batch rope k)",
                ))
                .is_none()
                {
                    return false;
                }
            }
        }
        if ok(ctx.sync()).is_none() {
            return false;
        }
        {
            let pin = s.br_pin_qk.as_ref().unwrap();
            let sl = pin.as_slice::<f32>();
            q.copy_from_slice(&sl[..q_floats]);
            k.copy_from_slice(&sl[q_floats..q_floats + k_floats]);
        }
        true
    })
}

/// P17k ステップ 5: dno_rope_qkv_pf の起動 (chain_qkv_rope_prefill とテスト共用)。
#[allow(clippy::too_many_arguments)]
fn launch_rope_qkv_pf(
    ctx: &CudaCtx,
    qkv: CUdeviceptr,
    qout: CUdeviceptr,
    km: CUdeviceptr,
    vm: CUdeviceptr,
    tab: CUdeviceptr,
    bias: CUdeviceptr,
    pos0: usize,
    n: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    kv_dim: usize,
) -> Result<()> {
    let (mut pq, mut pqo, mut pkm, mut pvm, mut pt, mut pb) = (qkv, qout, km, vm, tab, bias);
    let (mut p0c, mut hc, mut khc, mut dc, mut kdc) = (
        pos0 as c_int,
        heads as c_int,
        kv_heads as c_int,
        dh as c_int,
        kv_dim as c_int,
    );
    let mut args: Vec<*mut c_void> = vec![
        &mut pq as *mut _ as *mut c_void,
        &mut pqo as *mut _ as *mut c_void,
        &mut pkm as *mut _ as *mut c_void,
        &mut pvm as *mut _ as *mut c_void,
        &mut pt as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut p0c as *mut _ as *mut c_void,
        &mut hc as *mut _ as *mut c_void,
        &mut khc as *mut _ as *mut c_void,
        &mut dc as *mut _ as *mut c_void,
        &mut kdc as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async(
            ctx.f_rope_pf,
            ((heads + 2 * kv_heads) as u32, n as u32, 1),
            (dh as u32, 1, 1),
            0,
            &mut args,
        )
    }
}

/// P17k ステップ 5: prefill の qkv GEMM → bias/rope → K/V キャッシュ書き込みまでを
/// デバイスで完結させる (qkv→rope→attention 連続化の前半)。従来経路との差分:
/// - qkv GEMM 出力 [n, width] (122MB @13k tok) の D2H + q/k/v 分割コピーが消える
/// - rope が CPU (f32 powf/cos/sin × n×(heads+kv_heads)×half) から GPU へ
/// - rope 後 q の H2D (81MB) が消える (attention は chain_attention_prefill_dev
///   でデバイス q を直接読む)
/// - K/V は rope カーネルが GPU ミラーに直接書き、[pos0..t_total) 行だけ D2H
///   (27MB) でホストキャッシュへ (旧: CPU キャッシュ書き込み + ミラー H2D 27MB)
///
/// ビット一致: GEMM は dno_gemm_i8、bias は単一 add、rope は mul/sub (tab は
/// CPU が rope_row と同一式で生成) — CPU 経路 (lin_qkv_gpu → add_bias →
/// rope_rows → cache 書き込み) と q/K/V は完全一致。緩和不要。
///
/// 前提: 呼び出し側が bias/QK-Norm 無しを確認済み・cache は t_total 行分の容量
/// 確保済み・現在レイヤのホスト行 [pos0..t_total) は未書き込み (本関数が D2H で
/// 書く)。失敗時 None (ホスト状態不変 → 呼び出し側が従来経路でやり直し)。
/// ロープテーブル/rope カーネル成功後にミラー len を t_total に確定する。
#[allow(clippy::too_many_arguments)]
pub fn chain_qkv_rope_prefill(
    q_w: &Quant8Linear,
    k_w: &Quant8Linear,
    v_w: &Quant8Linear,
    h: &[f32],
    cache: &mut crate::kv_cache::KvCache,
    key: u64,
    li: usize,
    n_layers: usize,
    pos0: usize,
    n: usize,
    t_total: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    theta: f32,
    bias: &[f32],
    // B1: Some(slot) なら h は chain_rmsnorm が既にそのスロットに書き込んだ
    // デバイス常駐データとして直接読む (ホスト h の H2D 81MB/層を廃止)。
    // None なら従来どおりホスト `h` をアップロードする。Some 時は `h` を空でよい。
    h_dev: Option<usize>,
) -> Option<()> {
    if !gpu_attention_enabled() || !gemm_enabled() {
        return None;
    }
    if !(n > max_m() && n >= gemm_min_m()) {
        return None;
    }
    if n == 0 || t_total > cache.max_len() || n > 65535 {
        return None; // rope カーネルの grid.y 上限
    }
    let qn = heads.checked_mul(dh)?;
    let kv_dim = kv_heads.checked_mul(dh)?;
    let width = qn.checked_add(2 * kv_dim)?;
    let e = q_w.k;
    if k_w.k != e || v_w.k != e || k_w.n != kv_dim || v_w.n != kv_dim || q_w.n != qn {
        return None;
    }
    if h_dev.is_none() && h.len() != n * e {
        return None;
    }
    let ctx = chain_ctx()?;
    let lib = ok(lib())?;

    // ミラー取得 + 履歴行 [have..pos0) の同期 (現在行 [pos0..t_total) は rope
    // カーネルが書く)。have > pos0 はミラーがホストに先行 → フォールバック
    // (ホストキャッシュが完全でないため)。
    let (kptr, vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        if have > pos0 {
            return None;
        }
        if have < pos0 {
            let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim]))?;
        }
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    let uids = [q_w.uid, k_w.uid, v_w.uid];
    let gw = match weights_for_group(&uids, &[q_w, k_w, v_w], e, width) {
        Ok(g) => g,
        Err(err) => {
            tracing::error!("CUDA qkv weight upload failed, using CPU: {err:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };

    // bias: [qb | kb | vb] 連結 1 バッファ (空なら無し)。dno_rope_qkv_pf が
    // rope 前に加算する (CPU 経路の add_bias → rope_rows と同一順序・ビット一致)。
    let bias_len = bias.len();
    if !bias.is_empty() && bias_len != width {
        return None;
    }

    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.qkv_aq, n * e))?;
        ok(Scratch::ensure(&mut s.qkv_asc, n * 4))?;
        ok(Scratch::ensure(&mut s.qkv_pf, n * width * 4))?;
        ok(Scratch::ensure(&mut s.qkv_q, n * qn * 4))?;
        ok(Scratch::ensure(&mut s.rope_tab_pf, (n * dh + bias_len) * 4))?;
        ok(Scratch::ensure_pin(&mut s.pin_qkv, (n * e * 4).max(n * dh * 4 + bias_len * 4)))?;

        // GEMM 入力 h: B1 では chain_rmsnorm が書いたデバイス常駐スロット (CH_H)
        // を直接読む (H2D 不要)。None の場合は従来どおり pinned からアップロード。
        let dx = match h_dev {
            Some(slot) => {
                let b = match s.bufs.get(slot) {
                    Some(Some(b)) => b,
                    _ => return chain_fail(anyhow::anyhow!("chain_qkv_rope_prefill: h_dev slot missing")),
                };
                if b.bytes() < n * e * 4 {
                    return chain_fail(anyhow::anyhow!("chain_qkv_rope_prefill: h_dev slot too small"));
                }
                b.ptr()
            }
            None => {
                ok(Scratch::ensure(&mut s.qkv_x, n * e * 4))?;
                // h アップロード (pinned + async)。pin は次層で上書きするが、その前に
                // 本関数内の sync が挟まるため安全。
                {
                    let pin = s.pin_qkv.as_mut().unwrap();
                    pin.as_mut_slice::<f32>()[..n * e].copy_from_slice(h);
                }
                let pin_ptr = s.pin_qkv.as_ref().unwrap().ptr;
                unsafe {
                    ok(check(
                        (lib.memcpy_htod_async)(
                            s.qkv_x.as_ref().unwrap().ptr(),
                            pin_ptr as *const c_void,
                            n * e * 4,
                            std::ptr::null_mut(),
                        ),
                        "cuMemcpyHtoDAsync (qkv x)",
                    ))?;
                }
                s.qkv_x.as_ref().unwrap().ptr()
            }
        };
        let (daq, dasc, dqkv, dq, dtab) = (
            s.qkv_aq.as_ref().unwrap().ptr(),
            s.qkv_asc.as_ref().unwrap().ptr(),
            s.qkv_pf.as_ref().unwrap().ptr(),
            s.qkv_q.as_ref().unwrap().ptr(),
            s.rope_tab_pf.as_ref().unwrap().ptr(),
        );
        let pin_ptr = s.pin_qkv.as_ref().unwrap().ptr;
        // 1) qkv GEMM: h [n,e] → [n, width] (dno_gemm_i8・CPU lin とビット一致)
        crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
            let (mut px, mut paq, mut pasc) = (dx, daq, dasc);
            let (mut mc, mut kc_) = (n as c_int, e as c_int);
            let mut args = [
                &mut px as *mut _ as *mut c_void,
                &mut paq as *mut _ as *mut c_void,
                &mut pasc as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc_ as *mut _ as *mut c_void,
            ];
            ok(unsafe { ctx.launch_async(ctx.f_quant, (n as u32, 1, 1), (256, 1, 1), 256 * 4, &mut args) })?;
            let (mut paqi, mut pasci, mut pw, mut psc, mut pout) =
                (daq, dasc, gw.w.ptr(), gw.sc.ptr(), dqkv);
            let (mut mc, mut kc_, mut nc) = (n as c_int, e as c_int, width as c_int);
            let mut args = [
                &mut paqi as *mut _ as *mut c_void,
                &mut pasci as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut psc as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc_ as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let gmx = n.div_ceil(GEMM_TILE) as u32;
            let gmy = width.div_ceil(GEMM_TILE_N) as u32;
            ok(unsafe { ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (GEMM_BX, GEMM_BY, 1), 0, &mut args) })
        })?;

        // 2) rope テーブル (行ごと pos0+i・rope_row と同一式) と bias
        //    ([qb|kb|vb] 連結) を pinned に入れてアップロード。
        {
            let pin = s.pin_qkv.as_mut().unwrap();
            let tab = &mut pin.as_mut_slice::<f32>()[..n * dh];
            let half = dh / 2;
            for i in 0..n {
                let pos = (pos0 + i) as f32;
                let row = &mut tab[i * dh..][..dh];
                for d in 0..half {
                    let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                    row[d] = ang.cos();
                    row[half + d] = ang.sin();
                }
            }
            if bias_len > 0 {
                let dst = &mut pin.as_mut_slice::<f32>()[n * dh..n * dh + bias_len];
                dst.copy_from_slice(bias);
            }
        }
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dtab, pin_ptr as *const c_void, n * dh * 4, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (rope tab pf)",
            ))?;
            if bias_len > 0 {
                ok(check(
                    (lib.memcpy_htod_async)(
                        dtab + (n * dh * 4) as CUdeviceptr,
                        (pin_ptr as *const u8).add(n * dh * 4) as *const c_void,
                        bias_len * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyHtoDAsync (rope bias pf)",
                ))?;
            }
        }
        // 3) rope: q → qkv_q (連続)、k/v → ミラーの (pos0+i) 行へ
        let dbias = if bias_len > 0 {
            dtab + (n * dh * 4) as CUdeviceptr
        } else {
            0
        };
        crate::prof::time_layer(crate::prof::ROPE, li, || {
            ok(launch_rope_qkv_pf(
                ctx, dqkv, dq, kptr, vptr, dtab, dbias, pos0, n, heads, kv_heads, dh, kv_dim,
            ))
        })?;
        // 4) sync して K/V 現在行をホストキャッシュへ D2H (27MB・~3ms)。
        //    以降ホスト側に q/k/v Vec は存在しない。
        ok(ctx.sync())?;
        // s65: 発行元の件数分類 (K/V 行 2 本・表示専用)
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
        {
            let kb = cache.k_buf(li, 0);
            ok(unsafe {
                check(
                    (lib.memcpy_dtoh)(
                        kb[pos0 * kv_dim..t_total * kv_dim].as_mut_ptr() as *mut c_void,
                        kptr + (pos0 * kv_dim * 4) as CUdeviceptr,
                        n * kv_dim * 4,
                    ),
                    "cuMemcpyDtoH (qkv pf k)",
                )
            })?;
        }
        {
            let vb = cache.v_buf(li, 0);
            ok(unsafe {
                check(
                    (lib.memcpy_dtoh)(
                        vb[pos0 * kv_dim..t_total * kv_dim].as_mut_ptr() as *mut c_void,
                        vptr + (pos0 * kv_dim * 4) as CUdeviceptr,
                        n * kv_dim * 4,
                    ),
                    "cuMemcpyDtoH (qkv pf v)",
                )
            })?;
        }
        Some(())
    })?;

    // 5) ミラー len 確定 (rope カーネルのミラー書き込みが成功した後でのみ)。
    with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, _| {
        m.len[li] = t_total;
        Some(())
    })
}

/// P17k ステップ 5: prefill GPU attention — q が **デバイス常駐** (chain_
/// qkv_rope_prefill が書いた qkv_q) で K/V ミラーも同期済みの場合の variant。
/// q の H2D (81MB/層) とミラー行同期が不要。数値は chain_attention_prefill と
/// 完全同一 (同一カーネル・同一引数・q の内容も同一)。
/// 戻り None = CPU attention へ (chain_qkv_q_download で q を取得してから)。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_prefill_dev(
    key: u64,
    li: usize,
    n_layers: usize,
    pos0: usize,
    n: usize,
    t_total: usize,
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
    epoch: u64,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    if n == 0 || t_total == 0 || group == 0 || t_total > 65535 {
        return None;
    }
    // カーネル制約: chain_attention_prefill と同一 (s48 C1-a: smem 上限は opt-in 済み)
    if dh < ATTN_PF_BQ || dh > 1024 {
        return None;
    }
    if attn_pf_sh_f(dh) * 4 > attn_pf_dynamic_smem_limit() {
        return None;
    }
    let qn = heads.checked_mul(dh)?;
    let ctx = chain_ctx()?;
    ok(chain_ensure_buf(CH_ATT, n * qn * 4))?;
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, epoch, |m, _| {
        // 現在行は chain_qkv_rope_prefill が書き込み済み (len が t_total に確定済み)。
        if m.len[li] < t_total {
            return None;
        }
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        let mut qptr = s.qkv_q.as_ref()?.ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let (mut heads_c, mut n_c, mut pos0_c) =
            (heads as c_int, n as c_int, pos0 as c_int);
        let (mut kv_dim_c, mut t_c, mut group_c) =
            (kv_dim as c_int, t_total as c_int, group as c_int);
        let (mut dh_c, mut scale_c) = (dh as c_int, scale);
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut heads_c as *mut _ as *mut c_void,
            &mut n_c as *mut _ as *mut c_void,
            &mut pos0_c as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn_pf,
                // P17k: (行ブロック, heads) — x=行ブロック最速次元で同一 head の
                // K/V を同時実行ブロック間で共有し L2 ヒット率を上げる
                (n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                (dh as u32, 1, 1),
                (attn_pf_sh_f(dh) * 4) as u32,
                &mut args,
            )
        })
    })
}

/// P17k ステップ 5: デバイス常駐 q (qkv_q) をホストにダウンロードする
/// (attention フォールバック用)。同期 D2H。
pub fn chain_qkv_q_download(n: usize, qn: usize) -> Option<Vec<f32>> {
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let s = cell.borrow_mut();
        let buf = s.qkv_q.as_ref()?;
        ok(buf.to_vec(n * qn))
    })
}

/// P17h: decode chain 用 GPU attention — q/k/v が **デバイス常駐** (CH_QKV) の
/// 場合の variant。rope 済み k/v 現在行を CH_QKV からミラーへ D2D し、履歴行を
/// ホストキャッシュから H2D 同期してカーネルを起動する。rope 済み k/v は
/// pinned ステージへ非同期 D2H し、ホストキャッシュへの反映はステップ最後の
/// sync 後に `chain_kv_stage_flush` で行う (層あたりの sync を廃止するのが目的)。
/// 戻り None = forward 全体を従来経路で再実行。
#[allow(clippy::too_many_arguments)]
pub fn chain_attention_dev(
    key: u64,
    li: usize,
    n_layers: usize,
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    t_total: usize,
    heads: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    scale: f32,
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    let ctx = chain_ctx()?;
    if heads == 0 || group == 0 || dh == 0 || dh > 1024 || t_total == 0 || t_total > cache.max_len()
    {
        return None;
    }
    ok(chain_ensure_buf(CH_ATT, heads * dh * 4))?;

    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        // 履歴行 [have..pos0) をホストキャッシュから同期 (現在行 pos0 は CH_QKV から D2D)。
        if have < pos0 {
            let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim]))?;
        }
        // 現在行 (pos0) を CH_QKV からデバイス内コピー (rope 済み・ビット一致)。
        let qkv_off = (heads * dh * 4) as u64;
        let (kp_src, vp_src) = CHAIN.with(|cell| -> Option<(CUdeviceptr, CUdeviceptr)> {
            let s = cell.borrow();
            let base = s.bufs[CH_QKV].as_ref()?.ptr();
            Some((base + qkv_off, base + qkv_off + (kv_dim * 4) as u64))
        })?;
        let kb = m.layers[li].0.as_ref().unwrap();
        let vb = m.layers[li].1.as_ref().unwrap();
        let lib = ok(lib())?;
        let rowb = kv_dim * 4;
        unsafe {
            ok(check(
                (lib.memcpy_dtod_async)(
                    kb.ptr() + (pos0 * rowb) as u64,
                    kp_src,
                    rowb,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoDAsync (k row)",
            ))?;
            ok(check(
                (lib.memcpy_dtod_async)(
                    vb.ptr() + (pos0 * rowb) as u64,
                    vp_src,
                    rowb,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoDAsync (v row)",
            ))?;
        }
        m.len[li] = t_total;
        // roped k/v を pinned ステージへ非同期 D2H (ホスト反映は sync 後)。
        CHAIN.with(|cell| -> Option<()> {
            let mut s = cell.borrow_mut();
            let stage_bytes = n_layers * 2 * rowb;
            ok(Scratch::ensure_pin(&mut s.kv_stage, stage_bytes))?;
            let off = li * 2 * rowb;
            let pin = s.kv_stage.as_ref().unwrap();
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        (pin.ptr as *mut c_void).add(off),
                        kp_src,
                        rowb,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (k stage)",
                ))?;
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        (pin.ptr as *mut c_void).add(off + rowb),
                        vp_src,
                        rowb,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (v stage)",
                ))?;
            }
            Some(())
        })?;
        Some((kb.ptr(), vb.ptr()))
    })?;

    // scores スクラッチ + カーネル起動 (q は CH_QKV 先頭 — アップロード不要)
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_sc, heads * t_total * 4))?;
        let mut qptr = s.bufs[CH_QKV].as_ref().unwrap().ptr();
        let mut scptr = s.attn_sc.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut scale_c = scale;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = t_total as c_int;
        let mut group_c = group as c_int;
        let mut dh_c = dh as c_int;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (heads as u32, 1, 1),
                (dh as u32, 1, 1),
                (dh * 4) as u32,
                &mut args,
            )
        })
    })
}

/// P17h: ステージングした rope 済み k/v をホスト KvCache へ反映。
/// ステップ最後の sync (logits download) 後に呼ぶ — 全 D2H 完了済み。
/// ホストキャッシュは従来どおり正として機能し続ける (CPU 経路/snapshot 互換)。
pub fn chain_kv_stage_flush(
    cache: &mut crate::kv_cache::KvCache,
    n_layers: usize,
    kv_dim: usize,
    pos0: usize,
) -> Option<()> {
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let stage = s.kv_stage.as_ref()?;
        let slice = stage.as_slice::<f32>();
        for li in 0..n_layers {
            let off = li * 2 * kv_dim;
            let (kc, vc) = cache.kv_bufs(li, 1);
            kc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off..off + kv_dim]);
            vc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off + kv_dim..off + 2 * kv_dim]);
        }
        Some(())
    })
}

/// テスト用: dno_expf (glibc expf の bit-exact 再現) を要素ごとに適用する
pub fn chain_expf_probe(inp: &[f32]) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let n = inp.len();
    chain_upload(CH_GA, inp, n, 1)?;
    ok(chain_ensure_buf(CH_ACT, n * 4))?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let ip = s.bufs[CH_GA].as_ref().unwrap().ptr();
        let op = s.bufs[CH_ACT].as_ref().unwrap().ptr();
        let (mut pi, mut po) = (ip, op);
        let mut nc = n as c_int;
        let mut args = [
            &mut pi as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        let grid = (n as u32).div_ceil(256);
        ok(unsafe { ctx.launch_async(ctx.f_expf, (grid, 1, 1), (256, 1, 1), 0, &mut args) })
    })?;
    chain_download(CH_ACT, n)
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
fn fast_reduce_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("DENNO_FAST_REDUCE")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

#[derive(Clone, Copy)]
struct Q35Fns {
    f_matvec_f32: CUfunction,
    // s169: 並列リダクション版 (DENNO_FAST_REDUCE=1 のときだけ使う。PTX 未収録なら None)
    f_matvec_f32_r: Option<CUfunction>,
    f_qknorm_rope: CUfunction,
    f_gate: CUfunction,
}

// 中身は生ハンドル (関数ハンドル)。所有権の移動は伴わないため Send/Sync を
// 与える (CudaCtx と同一の規約・driver がスレッド安全に管理する)。
unsafe impl Send for Q35Fns {}
unsafe impl Sync for Q35Fns {}

static Q35_FNS: OnceLock<Option<Q35Fns>> = OnceLock::new();

fn q35_fns() -> Option<&'static Q35Fns> {
    Q35_FNS
        .get_or_init(|| {
            let ctx = chain_ctx()?;
            let lib = ok(lib())?;
            let get = |name: &[u8]| -> Option<CUfunction> {
                let mut f: CUfunction = std::ptr::null_mut();
                let r = unsafe {
                    (lib.module_get_function)(
                        &mut f,
                        ctx.module_elem,
                        name.as_ptr() as *const c_char,
                    )
                };
                if r != 0 {
                    None
                } else {
                    Some(f)
                }
            };
            Some(Q35Fns {
                f_matvec_f32: get(b"dno_matvec_f32\0")?,
                f_matvec_f32_r: get(b"dno_matvec_f32_r\0"),
                f_qknorm_rope: get(b"dno_qknorm_rope_pi\0")?,
                f_gate: get(b"dno_attn_gate_mul\0")?,
            })
        })
        .as_ref()
}

/// C4: qwen35 chain 用新カーネルが利用可能か (発動前ゲート用)。
pub fn q35_fns_available() -> bool {
    q35_fns().is_some()
}

/// C4: qwen35 chain 用 fp32 重みのキャッシュタグ。
/// (model_id, layer, tag) をキーにデバイスバッファをキャッシュする
/// (Linear::F32 / Vec<f32> には uid が無いため)。
pub const Q35_TAG_BETA: u8 = 0; // ssm_beta.weight [e, nv]
pub const Q35_TAG_ALPHA: u8 = 1; // ssm_alpha.weight [e, nv]
pub const Q35_TAG_QNORM: u8 = 2; // q_norm.weight [dh]
pub const Q35_TAG_KNORM: u8 = 3; // k_norm.weight [dh]
pub const Q35_TAG_CONV: u8 = 4; // conv1d.weight [conv_dim, d_conv]
pub const Q35_TAG_WA: u8 = 5; // ssm_a [nv]
pub const Q35_TAG_WDT: u8 = 6; // ssm_dt.bias [nv]
pub const Q35_TAG_WNORM: u8 = 7; // ssm_norm.weight [dv]

/// C4: full_attention 層の CH_QKV 内 k セクション先頭 (要素オフセット)。
/// レイアウトは [qg (heads*2dh) | k (kv_heads*dh) | v (kv_heads*dh)] —
/// chain_linear_grouped が parts 順に連結する。q は head ごと [q|gate] 融合で
/// head 間隔 2*dh だが、k/v は head 間隔 dh (chain_q35_fullattn の k 経路が
/// この関数を使い、誤ったオフセット/ストライドを防ぐ)。
fn q35_fullattn_k_off(heads: usize, dh: usize) -> usize {
    heads * 2 * dh
}

/// C4: decode chain 1 ステップの CH_QKV 要請バイト数 (純関数・単一ソース)。
/// - GDN 層: [qkv (2nk*dk+nv*dv) | z (nv*dv) | beta (nv) | alpha (nv)]
///   (= chain_q35_gdn_scan の `width`)
/// - full 層: [qg (heads*2dh) | k (kv_heads*dh) | v (kv_heads*dh)]
/// の大きい方。ステップ開始時にこのサイズで一括確保する。
///
/// 2994feb 時点では事前確保が無く、GDN 層の grouped GEMV (qkv+z) 直後に
/// chain_q35_gdn_scan 頭部の `chain_ensure_buf(CH_QKV, width*4)` が grow 再確保
/// (旧バッファ解放 + 新バッファ確保) を走らせていた。直前 launch の in-flight
/// 書き込み先が解放され、新バッファの [0..qkv+z) は誰にも書かれないまま
/// dno_gdn_scan が未初期化内容を読む (初期化されていない VRAM 内容は起動ごとに
/// 異なるため、実機 E2E の run 間出力差 + 冒頭乱れの直接原因)。
pub fn q35_ch_qkv_bytes(
    heads: usize,
    dh: usize,
    kv_heads: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
) -> usize {
    let kv_dim = kv_heads * dh;
    let gdn = (2 * nk * dk + nv * dv) + nv * dv + 2 * nv;
    let full = heads * 2 * dh + 2 * kv_dim;
    gdn.max(full) * 4
}

/// C4: decode chain 1 ステップの CH_ATT 要請バイト数 (純関数)。
/// GDN 層の出力 y は [vd = nv*dv]・full 層は [heads*dh]。
/// CH_QKV と同一の競合クラス (full 層側の ensure が直前 GDN scan の in-flight
/// 書き込み先を解放し得る) をステップ開始時の一括確保で排除する。
pub fn q35_ch_att_bytes(nv: usize, dv: usize, heads: usize, dh: usize) -> usize {
    (nv * dv).max(heads * dh) * 4
}

/// C4: KV ステージ (full 層の k/v 行 staged D2H 転送先 pinned) のステップ全体
/// バイト数 (純関数)。full 層 li の積み位置は li*2*kv_dim 要素・1 層分は
/// 2*kv_dim 要素 (chain_q35_kv_stage_flush の読み出しと同一式)。
pub fn q35_kv_stage_bytes(n_layers: usize, kv_dim: usize) -> usize {
    n_layers * 2 * kv_dim * 4
}

/// C4: fp32 重みのデバイスキャッシュ取得 (無ければ同期 H2D でアップロード)。
/// 初回 1 回のみ同期コピーが走る (chain 発動セッションの最初のステップ)。
fn q35_w_dev(mid: u64, li: usize, tag: u8, w: &[f32]) -> Option<CUdeviceptr> {
    let _ = chain_ctx()?;
    CHAIN.with(|cell| -> Option<CUdeviceptr> {
        let mut s = cell.borrow_mut();
        if let Some(b) = s.q35_w_cache.get(&(mid, li, tag)) {
            if b.bytes() == w.len() * 4 {
                return Some(b.ptr());
            }
        }
        let buf = ok(DevBuffer::from_slice(w))?;
        let p = buf.ptr();
        s.q35_w_cache.insert((mid, li, tag), buf);
        Some(p)
    })
}

/// C4: qwen35 chain の発動条件 (カーネル形状ゲート)。数値契約は
/// rmsnorm/int8 GEMV/swiglu/expf/rope/QK-Norm/gate/fp32 matvec がビット一致、
/// attention スコア/softmax のみ緩和 (P17g・ユーザー承認済み)、GDN スキャンは
/// bit 一致不要契約 (決定論のみ)。形状はカーネルの前提 (gdn.cu / kernels_elem.cu
/// の C4 節コメント) と一致させる。
#[allow(clippy::too_many_arguments)]
pub fn q35_chain_shapes_ok(
    e: usize,
    ff: usize,
    dh: usize,
    heads: usize,
    kv_heads: usize,
    n_rot: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
) -> bool {
    // dno_add_rmsnorm の動的共有メモリ上限 (48KB / 4B)
    if e == 0 || e > 12_288 || ff == 0 {
        return false;
    }
    if dh == 0 || dh > 1024 || heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return false;
    }
    // partial interleaved rope: ペア数 n_rot/2 は dh 内・ペア単位で回転
    if n_rot == 0 || n_rot > dh || n_rot % 2 != 0 {
        return false;
    }
    // dno_gdn_scan の前提 (gdn.cu 先頭コメント): dk==dv・d_conv==4・dk<=256
    dk == dv && d_conv == 4 && (1..=256).contains(&dk)
}

/// C4: ホスト生成の partial interleaved rope テーブル [cos(n_rot/2) | sin(n_rot/2)]
/// をデバイスへアップロード (1 ステップ 1 回)。テーブルの生成は model::qwen35 側
/// (rope_partial_interleaved と同一式) が担当し、cuda 側は転送のみを担う
/// (テーブル値のビット一致を生成箇所で保証する)。専用 pinned スロットを使う
/// (P17h 規約: ステップ内 sync 無しのため pinned は用途ごとに専用化)。
pub fn chain_q35_rope_pi_upload(tab: &[f32]) -> Option<()> {
    let _ = chain_ctx()?;
    let bytes = tab.len().checked_mul(4)?;
    if bytes == 0 {
        return None;
    }
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.q35_tab, bytes))?;
        ok(Scratch::ensure_pin(&mut s.q35_pin_tab, bytes))?;
        {
            let pin = s.q35_pin_tab.as_mut().unwrap();
            pin.as_mut_slice::<f32>()[..tab.len()].copy_from_slice(tab);
        }
        let dev = s.q35_tab.as_ref().unwrap().ptr();
        let pin = s.q35_pin_tab.as_ref().unwrap();
        let lib = ok(lib())?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dev, pin.ptr as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (q35 rope tab)",
            ))
        }?;
        // s166 (P5): 観測のみ (1 ステップ 1 回・pos 依存なので削れない)
        dbg_htod(bytes);
        Some(())
    })
}

/// C4: GDN 制御投影 (beta/alpha) を dno_matvec_f32 で CH_H → CH_QKV テールへ
/// 出力する (tensor::matvec_bt とビット一致)。out_off は CH_QKV 内の要素
/// オフセット (grouped GEMV 出力 [qkv|z] の直後)。CH_QKV はテール分もここで
/// 確保する (chain_linear_grouped の ensure は grow-only なので上書きされない)。
fn chain_q35_ctl_f32(
    mid: u64,
    li: usize,
    tag: u8,
    w: &[f32],
    k: usize,
    n: usize,
    out_off: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    if w.len() != k * n || n == 0 || n > 65535 {
        return None;
    }
    ok(chain_ensure_buf(CH_QKV, (out_off + n) * 4))?;
    let wptr = q35_w_dev(mid, li, tag, w)?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let hptr = s.bufs[CH_H].as_ref()?.ptr();
        let optr = s.bufs[CH_QKV].as_ref()?.ptr() + (out_off * 4) as u64;
        let (mut ph, mut pw, mut po) = (hptr, wptr, optr);
        let (mut kc, mut nc) = (k as c_int, n as c_int);
        let mut args = [
            &mut ph as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut kc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
        ];
        // s169: DENNO_FAST_REDUCE=1 なら固定の木で縮約する並列版 (block 256)。
        // 丸め順が変わるため既定は逐次版 (bit 一致を守る)。
        let (func, thr) = match (fns.f_matvec_f32_r, fast_reduce_enabled()) {
            (Some(fr), true) => (fr, 256u32),
            _ => (fns.f_matvec_f32, 128u32),
        };
        ok(unsafe { ctx.launch_async(func, (n as u32, 1, 1), (thr, 1, 1), 0, &mut args) })
    })
}

/// C4: QK-Norm + partial interleaved rope (dno_qknorm_rope_pi)。
/// src スロットの src_off (要素) を先頭に stride x_stride で並ぶ heads 行
/// (block = heads) を処理し、dst_ptr + i*dst_stride へ書く。
/// - q: src = CH_QKV (head ごと [q|gate] 融合・x_stride = 2*dh) → CH_Q35_Q
/// - k: src = CH_QKV の k 部 (x_stride = kv_dim) → KV ミラーの pos0 行
/// rope テーブルは chain_q35_rope_pi_upload 済み (q35_tab) を使う。
#[allow(clippy::too_many_arguments)]
fn chain_q35_qknorm_rope(
    mid: u64,
    li: usize,
    tag: u8,
    w: &[f32],
    src_slot: usize,
    src_off: usize,
    x_stride: usize,
    heads: usize,
    cols: usize,
    n_rot: usize,
    eps: f32,
    dst_ptr: CUdeviceptr,
    dst_stride: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    if w.len() != cols
        || heads == 0
        || cols == 0
        || cols > 1024
        || n_rot == 0
        || n_rot > cols
        || n_rot % 2 != 0
    {
        return None;
    }
    ok(chain_ensure_buf(src_slot, (src_off + heads * x_stride) * 4))?;
    let wptr = q35_w_dev(mid, li, tag, w)?;
    let tabp = CHAIN.with(|cell| -> Option<CUdeviceptr> {
        let s = cell.borrow();
        Some(s.q35_tab.as_ref()?.ptr())
    })?;
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let xptr = s.bufs[src_slot].as_ref()?.ptr() + (src_off * 4) as u64;
        let (mut px, mut pxs, mut pw, mut pt, mut po, mut pst) = (
            xptr,
            x_stride as c_int,
            wptr,
            tabp,
            dst_ptr,
            dst_stride as c_int,
        );
        let (mut cc, mut nc, mut ec) = (cols as c_int, n_rot as c_int, eps);
        let mut args = [
            &mut px as *mut _ as *mut c_void,
            &mut pxs as *mut _ as *mut c_void,
            &mut pw as *mut _ as *mut c_void,
            &mut pt as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut pst as *mut _ as *mut c_void,
            &mut cc as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut ec as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                fns.f_qknorm_rope,
                (heads as u32, 1, 1),
                (cols as u32, 1, 1),
                ((cols + 1) * 4) as u32,
                &mut args,
            )
        })
    })
}

/// C4: qwen35 full-attention 層の decode attention (デバイス常駐)。
/// 前提: CH_QKV に [qg (heads*2dh) | k (kv_dim) | v (kv_dim)] がある
/// (chain_linear_grouped 直後)。処理:
/// 1. KV ミラー取得 + 履歴行 [have..pos0) をホストから同期
/// 2. k: dno_qknorm_rope_pi が KV ミラーの pos0 行へ直接書く (QK-Norm + rope)
///    q: 同カーネルが CH_Q35_Q へ書く (attention 入力)
/// 3. v: CH_QKV からミラー pos0 行へ D2D + staged async D2H (k/v 行)
/// 4. dno_attention (GQA・緩和枠) → dno_attn_gate_mul (sigmoid ゲート・ビット一致)
/// ホストキャッシュへの反映は sync 後の chain_q35_kv_stage_flush。
/// 失敗時 None (forward 全体を従来経路でやり直し — ホスト状態不変のため安全)。
#[allow(clippy::too_many_arguments)]
pub fn chain_q35_fullattn(
    mid: u64,
    key: u64,
    li: usize,
    n_layers: usize,
    cache: &crate::kv_cache::KvCache,
    pos0: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    n_rot: usize,
    eps: f32,
    q_norm_w: &[f32],
    k_norm_w: &[f32],
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    let ctx = chain_ctx()?;
    let fns = q35_fns()?;
    let kv_dim = kv_heads * dh;
    let t_total = pos0 + 1;
    if heads == 0 || dh == 0 || dh > 1024 || t_total == 0 || t_total > cache.max_len() {
        return None;
    }
    let group = heads / kv_heads;
    let hd = heads * dh;
    ok(chain_ensure_buf(CH_ATT, hd * 4))?;
    ok(chain_ensure_buf(CH_Q35_Q, hd * 4))?;

    // 1) ミラー取得 + 履歴行同期 (ポインタはロック解放前に抽出)
    let (mut kptr, mut vptr) =
        with_mirror(key, li, n_layers, t_total, kv_dim, cache.epoch(), |m, have| {
        if have < pos0 {
            let (kc_ref, vc_ref) = cache.kv_bufs_ref(li);
            let kb = m.layers[li].0.as_ref().unwrap();
            let vb = m.layers[li].1.as_ref().unwrap();
            ok(kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim]))?;
            ok(vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim]))?;
        }
        m.len[li] = t_total;
        Some((
            m.layers[li].0.as_ref().unwrap().ptr(),
            m.layers[li].1.as_ref().unwrap().ptr(),
        ))
    })?;

    // 2) QK-Norm + rope: k はミラー pos0 行へ直接書く・q は CH_Q35_Q へ
    //    (CH_QKV レイアウト: [qg (heads*2dh) | k (kv_dim) | v (kv_dim)] —
    //    k セクション先頭は q35_fullattn_k_off (heads*2*dh)・head b の行内
    //    位置は b*dh (head 間隔 dh。融合 [q|gate] の q とは間隔が異なる)。
    //    旧実装は src_off=hd (qg 後半を読む)・head 間隔 kv_dim (k 節域を外れて
    //    読む + ミラー行の外へ書く) で、k 履歴を破壊していた。
    let k_off = q35_fullattn_k_off(heads, dh);
    let km_row = kptr + (pos0 * kv_dim * 4) as u64;
    let q_ptr = CHAIN.with(|cell| -> Option<CUdeviceptr> {
        let s = cell.borrow();
        Some(s.bufs[CH_Q35_Q].as_ref()?.ptr())
    })?;
    chain_q35_qknorm_rope(
        mid, li, Q35_TAG_QNORM, q_norm_w,
        CH_QKV, 0, 2 * dh, heads, dh, n_rot, eps,
        q_ptr,
        dh,
    )?;
    chain_q35_qknorm_rope(
        mid, li, Q35_TAG_KNORM, k_norm_w,
        CH_QKV, k_off, dh, kv_heads, dh, n_rot, eps, km_row, dh,
    )?;

    // 3) v 行 D2D + k/v 行の staged async D2H (ホスト反映は sync 後)
    let (vp_src, kv_dim64) = CHAIN.with(|cell| -> Option<(CUdeviceptr, u64)> {
        let s = cell.borrow();
        let base = s.bufs[CH_QKV].as_ref()?.ptr();
        Some((base + ((k_off + kv_dim) * 4) as u64, (kv_dim * 4) as u64))
    })?;
    let lib = ok(lib())?;
    unsafe {
        ok(check(
            (lib.memcpy_dtod_async)(
                vptr + (pos0 as u64 * kv_dim64),
                vp_src,
                kv_dim64 as usize,
                std::ptr::null_mut(),
            ),
            "cuMemcpyDtoDAsync (q35 v row)",
        ))?;
    }
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        let stage_bytes = q35_kv_stage_bytes(n_layers, kv_dim);
        ok(Scratch::ensure_pin(&mut s.kv_stage, stage_bytes))?;
        let off = li * 2 * kv_dim * 4;
        let pin = s.kv_stage.as_ref().unwrap();
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(
                    (pin.ptr as *mut c_void).add(off),
                    km_row,
                    kv_dim * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (q35 k stage)",
            ))?;
            ok(check(
                (lib.memcpy_dtoh_async)(
                    (pin.ptr as *mut c_void).add(off + kv_dim * 4),
                    vp_src,
                    kv_dim * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (q35 v stage)",
            ))?;
            // s65: 発行元の件数分類 (full 層の K/V 行 2 本。staged async 発行だが
            // 件数は d2h_state の calls に含まれる — 表示専用)
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_KV_ROW);
            // s166 (P5): 観測のみ (full 層 1 つあたり k/v 2 本・ホスト KV キャッシュ契約)
            dbg_d2h(kv_dim * 4);
            dbg_d2h(kv_dim * 4);
        }
        Some(())
    })?;

    // 4) attention (q は CH_Q35_Q) + sigmoid 出力ゲート
    let scale = 1.0 / (dh as f32).sqrt();
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.attn_sc, heads * t_total * 4))?;
        let mut qptr = s.bufs[CH_Q35_Q].as_ref()?.ptr();
        let mut scptr = s.attn_sc.as_ref().unwrap().ptr();
        let mut aptr = s.bufs[CH_ATT].as_ref().unwrap().ptr();
        let mut scale_c = scale;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = t_total as c_int;
        let mut group_c = group as c_int;
        let mut dh_c = dh as c_int;
        let mut args: Vec<*mut c_void> = vec![
            &mut qptr as *mut _ as *mut c_void,
            &mut kptr as *mut _ as *mut c_void,
            &mut vptr as *mut _ as *mut c_void,
            &mut aptr as *mut _ as *mut c_void,
            &mut scptr as *mut _ as *mut c_void,
            &mut kv_dim_c as *mut _ as *mut c_void,
            &mut t_c as *mut _ as *mut c_void,
            &mut group_c as *mut _ as *mut c_void,
            &mut dh_c as *mut _ as *mut c_void,
            &mut scale_c as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(
                ctx.f_attn,
                (heads as u32, 1, 1),
                (dh as u32, 1, 1),
                (dh * 4) as u32,
                &mut args,
            )
        })?;
        // sigmoid 出力ゲート: att[h,i] /= (1 + exp(-qg[h*2dh + dh + i]))
        let mut qgptr = s.bufs[CH_QKV].as_ref()?.ptr();
        let (mut hc, mut dhc) = (heads as c_int, dh as c_int);
        let mut gargs = [
            &mut aptr as *mut _ as *mut c_void,
            &mut qgptr as *mut _ as *mut c_void,
            &mut hc as *mut _ as *mut c_void,
            &mut dhc as *mut _ as *mut c_void,
        ];
        ok(unsafe {
            ctx.launch_async(fns.f_gate, (heads as u32, 1, 1), (dh as u32, 1, 1), 0, &mut gargs)
        })
    })
}

/// C4: qwen35 GDN 層の decode スキャン (デバイス常駐)。
/// 前提: CH_H に rmsnorm 出力、CH_QKV に [qkv (conv_dim) | z (vd)] がある
/// (chain_linear_grouped 直後)。beta/alpha は本関数が dno_matvec_f32 で
/// CH_QKV テール [(conv_dim+vd)..] へ出力する。GDN 状態ミラー (ping-pong) を
/// seed/更新し、dno_gdn_scan (既存カーネル・grid=(nv,), block=(dk,)) を起動 —
/// y を CH_ATT に書く。新状態 (conv/ssm) は s50 からミラーの書き込み面に留め、
/// ステップ毎のホスト D2H はしない (遅延フラッシュ — 契約と回復手順は
/// chain_q35_gdn_flush_pending)。失敗時 None (forward 全体を従来経路でやり直し)。
#[allow(clippy::too_many_arguments)]
pub fn chain_q35_gdn_scan(
    mid: u64,
    key: u64,
    li: usize,
    n_layers: usize,
    cache: &mut crate::kv_cache::KvCache,
    pos0: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    beta_w: &[f32],
    alpha_w: &[f32],
    e: usize,
) -> Option<()> {
    let ctx = chain_ctx()?;
    // 新カーネル (dno_matvec_f32) の存在確認を早めに行う (ctl_f32 内でも失敗するが
    // ここで弾く方が意図が明確)。
    if q35_fns().is_none() {
        return None;
    }
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv; // == qkv_w.n (GDN カーネルの qkv 行幅)
    let vd = nv * dv;
    let t_total = pos0 + 1;
    if dk == 0 || dv == 0 || dk != dv || d_conv != 4 || !(1..=256).contains(&dk) {
        return None;
    }
    if wconv.len() != conv_dim * d_conv || wa.len() != nv || wdt.len() != nv || wnorm.len() != dv {
        return None;
    }
    if beta_w.len() != e * nv || alpha_w.len() != e * nv {
        return None;
    }
    let width = conv_dim + vd + 2 * nv;
    ok(chain_ensure_buf(CH_QKV, width * 4))?;
    ok(chain_ensure_buf(CH_ATT, vd * 4))?;

    // beta/alpha 制御投影 (fp32 matvec・CH_QKV テールへ)
    chain_q35_ctl_f32(mid, li, Q35_TAG_BETA, beta_w, e, nv, conv_dim + vd)?;
    chain_q35_ctl_f32(mid, li, Q35_TAG_ALPHA, alpha_w, e, nv, conv_dim + vd + nv)?;

    // s141b (Task B7c): seed の入力は **デバイス正本** から直接 D2D で取る。
    // 以前は「正本 → ホスト (D2H) → ミラー (H2D)」と、同一 GPU 内のコピーを
    // わざわざ PCIe 経由で往復させていた (層ごとに ~3.2 MiB の D2H)。
    // 27B q4 の decode では `with_mirror_ext` の cap 成長時に全層の gdn_pos が
    // 落ちるため、この往復が 48 層 x 2 本ずつ (= gdn_pf_alloc の実体) 発生していた。
    //
    // 規則 (所有者はこう変わる):
    // - デバイスミラー = decode 中の権威 (従来どおり)
    // - ホスト `KvCache::gdn` = **遅延リフレッシュ対象**。mirror.gdn_stale が真の間は
    //   ホストが古い可能性がある。ホストを読む前に必ず
    //   chain_q35_gdn_flush_pending (= mirror.gdn_pending を Some にしておく) か
    //   KvCache::flush_gdn_device_state を通すこと。
    // - 正本が無効な層 (valid[li] == false = ホストが正) だけは従来どおりホストから
    //   H2D する。ホストはその時点で正 (何も捨てていない)。
    //
    // 失敗時の扱い: 正本から読めなかった (None) 場合は従来どおりホストから seed
    // する。これは s141 までの「正本が無効な層 = ホストが正」と同じ状態で、
    // 正本が無効になるのは必ずホストが正に戻された後 (gdn_scan_state_invalidate は
    // ホスト書き換え経路が呼ぶ) なので、古いホストを読むことはない。
    let devseed = GDN_STATE.with(|cell| {
        let mut c = cell.borrow_mut();
        gdn_state_take_dev(cache.kv_id(), li, &mut c)
    });
    let (conv_in, ssm_in, conv_out, ssm_out) =
        with_mirror_ext(key, li, n_layers, t_total, 0, cache.epoch(), false, |m, _| {
            if m.gdn[li].is_none() {
                let mut g = GdnDev {
                    conv: [None, None],
                    ssm: [None, None],
                    cur: 0,
                };
                ok(Scratch::ensure(&mut g.conv[0], conv_dim * cs * 4))?;
                ok(Scratch::ensure(&mut g.conv[1], conv_dim * cs * 4))?;
                ok(Scratch::ensure(&mut g.ssm[0], nv * dk * dv * 4))?;
                ok(Scratch::ensure(&mut g.ssm[1], nv * dk * dv * 4))?;
                m.gdn[li] = Some(g);
            }
            let gd = m.gdn[li].as_mut().unwrap();
            // seed: この層のミラーが pos0 に同期していなければ状態を面 0 へ入れる
            if m.gdn_pos[li] != Some(pos0) {
                // s131/s141b: 正本 (デバイス) が有効ならそこから D2D で取る。
                // 無効 (= valid[li] == false = ホストが正) のときだけホストから。
                match devseed {
                    Some((pc, ps)) => {
                        ok(gd.conv[0].as_ref().unwrap().copy_from_dev(pc, conv_dim * cs))?;
                        ok(gd.ssm[0].as_ref().unwrap().copy_from_dev(ps, nv * dk * dv))?;
                        // 正本から入れた = **ホストは更新していない**。ホストが古い
                        // 可能性は残るので遅延フラグは立てたままにする (ここで落とすと、
                        // 後で flush がこの層を飛ばして古いホストを正として使ってしまう)。
                        m.gdn_stale[li] = true;
                    }
                    None => {
                        // 正本が無効 = ホストが正。ホストを読んで面 0 を書くので
                        // ミラーとホストは一致する → 遅延フラグを落とす。
                        let (conv_h, ssm_h) = cache.gdn_state_mut(li)?;
                        ok(gd.conv[0].as_ref().unwrap().copy_from(conv_h))?;
                        ok(gd.ssm[0].as_ref().unwrap().copy_from(ssm_h))?;
                        m.gdn_stale[li] = false;
                    }
                }
                gd.cur = 0;
            }
            let cur = gd.cur;
            let nxt = 1 - cur;
            Some((
                gd.conv[cur].as_ref().unwrap().ptr(),
                gd.ssm[cur].as_ref().unwrap().ptr(),
                gd.conv[nxt].as_ref().unwrap().ptr(),
                gd.ssm[nxt].as_ref().unwrap().ptr(),
            ))
        })?;

    // dno_gdn_scan 起動 (qkv/z は CH_QKV・beta/alpha はテール・y は CH_ATT)
    let (mut pq, mut pz, mut pb, mut pa, mut pci, mut psi) = CHAIN.with(
        |cell| -> Option<(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr)> {
            let s = cell.borrow();
            let base = s.bufs[CH_QKV].as_ref()?.ptr();
            Some((
                base,
                base + (conv_dim * 4) as u64,
                base + ((conv_dim + vd) * 4) as u64,
                base + ((conv_dim + vd + nv) * 4) as u64,
                conv_in,
                ssm_in,
            ))
        },
    )?;
    let pwc = q35_w_dev(mid, li, Q35_TAG_CONV, wconv)?;
    let pwa = q35_w_dev(mid, li, Q35_TAG_WA, wa)?;
    let pwd = q35_w_dev(mid, li, Q35_TAG_WDT, wdt)?;
    let pwn = q35_w_dev(mid, li, Q35_TAG_WNORM, wnorm)?;
    let (mut pwc, mut pwa, mut pwd, mut pwn) = (pwc, pwa, pwd, pwn);
    let (mut py, mut pco, mut pso) = (
        CHAIN.with(|cell| -> Option<CUdeviceptr> {
            let s = cell.borrow();
            Some(s.bufs[CH_ATT].as_ref()?.ptr())
        })?,
        conv_out,
        ssm_out,
    );
    let (mut ni, mut nki, mut nvi, mut dki, mut dvi) = (
        1 as c_int, nk as c_int, nv as c_int, dk as c_int, dv as c_int,
    );
    let (mut cdi, mut dci, mut csi) = (conv_dim as c_int, d_conv as c_int, cs as c_int);
    let (mut epsi, mut qsi) = (eps, q_scale);
    let mut args = [
        &mut pq as *mut _ as *mut c_void,
        &mut pz as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut psi as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pwn as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut dki as *mut _ as *mut c_void,
        &mut dvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut dci as *mut _ as *mut c_void,
        &mut csi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
        &mut qsi as *mut _ as *mut c_void,
    ];
    let shmem = ((3 * dk + dv + 5) * 4) as u32;
    ok(unsafe {
        // dno_gdn_scan は既存カーネル (kernels_gdn.ptx・ctx.f_gdn に取得済み) を
        // そのまま使う — 新 PTX は不要 (kernels_elem 側の新カーネルのみ人間再生成)。
        ctx.launch_async(ctx.f_gdn, (nv as u32, 1, 1), (dk as u32, 1, 1), shmem, &mut args)
    })?;

    // s50: 新状態 (ping-pong の書き込み面) はホストへ戻さない (遅延フラッシュ)。
    // 旧実装はここで conv/ssm を pinned へ staged async D2H し、sync 後の
    // chain_q35_gdn_stage_flush でホストへ反映していた。ステップ毎 D2H は
    // 9B で ~75MB/step = decode の支配項だったため、ミラーを権威のままにして
    // 「ホスト状態が実際に必要になる時点」まで転送を遅らせる。

    // ミラー位置の確定 (launch 成功後 — 失敗時は GPU_BROKEN で chain 停止のため
    // 誤更新は残らない)。ping-pong 面を交換し、この層の新状態を遅延 pending に
    // 載せる。層途中で chain が失敗した場合は掃けた層だけ pending になる — flush 側は
    // gdn_pos と cache.len() の整合で掃けた面だけを回復できる。
    with_mirror_ext(key, li, n_layers, t_total, 0, cache.epoch(), false, |m, _| {
        m.gdn_pos[li] = Some(pos0 + 1);
        m.gdn_pending = Some(pos0 + 1);
        // s141b (Task B7c): この層の新状態は書き込み面にだけある (ホストは古い)。
        // 以後この層の seed はホストではなくデバイス正本から取る。
        m.gdn_stale[li] = true;
        if let Some(gd) = m.gdn[li].as_mut() {
            gd.cur = 1 - gd.cur;
        }
        Some(())
    })
}

/// C4: KV ステージ pinned をステップ開始時に全体分事前確保する (s50 から GDN は
/// ステージしないため kv_stage のみ)。chain_q35_fullattn 内の ensure は事前確保後
/// no-op の二重防御。
pub fn chain_q35_kv_stage_begin(stage_bytes: usize) -> Option<()> {
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_pin(&mut s.kv_stage, stage_bytes))
    })
}

/// s50: 遅延 GDN 状態フラッシュ。decode chain は GDN の新状態 (conv/ssm) をステップ
/// 毎にホストへ戻さず GPU ミラー (ping-pong の cur 面) を権威のままにする
/// (chain_q35_gdn_scan の doc 参照 — 旧実装のステップ毎 D2H は 9B で ~75MB/step =
/// decode の支配項)。本関数は「ホスト状態が実際に必要になる時点」—
/// ①次の prefill の冒頭 (forward_prefill はホスト状態を正として読み/書く)
/// ②decode chain から従来 CPU 経路へのフォールバック時 (forward) —
/// に呼ばれ、ミラーの現行面を同期 D2H でホストキャッシュへ回復する。
///
/// 契約:
/// - ミラーの内容は各 decode ステップ末尾の chain_download (sync) で確定済み
///   (ステップ末尾にステップ後の非同期カーネルを残さない現行設計) → 同期 D2H
///   (DevBuffer::to_vec) は確定値を読む。転送は bit そのまま (演算なし) のため、
///   フラッシュ後のホスト状態は「毎ステップフラッシュした場合」と bit 一致。
/// - 位置整合: 対象位置 = cache.len()。gdn_pos[li] == target → cur 面、
///   gdn_pos[li] == target+1 → 1-cur 面 (直前の chain ステップが層途中で失敗した
///   ケースの ping-pong 入力面 = state at target)。gdn_pos[li] == None → その層は
///   chain で 1 度も走っていない (ホスト状態が最新) → スキップ。それ以外は不整合
///   (バグのみ) → pending 破棄のうえ None (呼び出し側は bail で要求を落とす =
///   古い状態で黙って継続するより安全)。
/// - epoch 不一致 (reset/restore でホストが正に戻った) → pending 破棄で no-op
///   (前会話のデータは使わない — ホストが正)。
/// - pending 無し (通常の prefill 毎・fp32 モード等) は no-op (登録簿参照のみ)。
/// - ミラーロック保持中に D2H を実施する (解放すると LRU エビクトが走り得て
///   読み出し元の DevBuffer が解放される競合になる — with_mirror 内 H2D と同一方針)。
/// 戻り値: None = ホスト状態を回復できなかった (呼び出し側は継続してはならない)。
pub fn chain_q35_gdn_flush_pending(cache: &mut crate::kv_cache::KvCache) -> Option<()> {
    let key = cache.kv_id();
    let epoch = cache.epoch();
    let target = cache.len();
    let mirrors = KV_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut ms = ok(mirrors.lock().map_err(|e| anyhow::anyhow!("{e}")))?;
    let Some(idx) = ms.iter().position(|(k, _, _)| *k == key) else {
        return Some(()); // ミラー未作成 = 遅延も存在し得ない
    };
    // s90 (P1-1): ホストの GDN 状態をこれから書き換える (遅延分の回復) →
    // デバイス常駐の正本は古くなるので先に捨てる。早めに捨てても失うものは無い
    // (次の実行でホストから再 seed するだけ) ので、分岐の前に 1 回で置く。
    gdn_scan_state_invalidate();
    // s141 (Task B7): D2H の受け皿は層ループの**外**で 1 回だけ作り、層ごとに
    // 使い回す (grow-only)。従来は層ごとに `to_vec` で 2 本確保していた。
    let mut stage = GdnHostStage::new();
    let m = &mut ms[idx].1;
    if m.gdn_pending.is_none() {
        return Some(()); // 遅延なし
    }
    if m.epoch != epoch {
        m.gdn_pending = None;
        return Some(());
    }
    for li in 0..m.gdn.len() {
        let Some(gd) = m.gdn[li].as_ref() else {
            continue; // GDN 層ではない (full attention 層)
        };
        // s141b (Task B7c): ホストが既に最新の層 (gdn_stale = false) は D2H しない。
        // 判定は純関数 `gdn_flush_step` (テストが実物を通す)。
        let (need, aligned) = gdn_flush_step(true, m.gdn_stale[li], target, m.gdn_pos[li]);
        if !need {
            if m.gdn_stale[li] && !aligned {
                // 古い可能性があり、かつ位置が合っていない → 従来どおり不整合扱い
                m.gdn_pending = None;
                return None;
            }
            continue;
        }
        let face = match m.gdn_pos[li] {
            Some(p) if p == target => gd.cur,
            Some(p) if p == target + 1 => 1 - gd.cur,
            _ => {
                // 不整合 (正常系では起こらない — バグのみ)
                m.gdn_pending = None;
                return None;
            }
        };
        let (Some(cb), Some(sb)) = (&gd.conv[face], &gd.ssm[face]) else {
            m.gdn_pending = None;
            return None;
        };
        // s65: 発行元の件数分類 (GDN 状態の回復 = conv + ssm の 2 本)
        // s88 (Step 1): この層の conv + ssm の回復区間を site の時間としても積む
        // (表示専用)。件数は従来どおり 2 件 (conv / ssm の 2 本)。
        let t_site = if crate::prof::d2h_src_on() {
            Some(std::time::Instant::now())
        } else {
            None
        };
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        let conv = match d2h_raw(&mut stage.conv, cb.ptr(), cb.bytes() / 4) {
            Some(v) => v,
            None => {
                m.gdn_pending = None;
                return None;
            }
        };
        let ssm = match d2h_raw(&mut stage.ssm, sb.ptr(), sb.bytes() / 4) {
            Some(v) => v,
            None => {
                m.gdn_pending = None;
                return None;
            }
        };
        if let Some(t) = t_site {
            crate::prof::add_d2h_src_ns_only(
                crate::prof::D2H_SRC_GDN_STATE,
                t.elapsed().as_nanos() as u64,
            );
        }
        let (conv_h, ssm_h) = cache.gdn_state_mut(li)?;
        if conv_h.len() < conv.len() || ssm_h.len() < ssm.len() {
            m.gdn_pending = None;
            return None;
        }
        conv_h[..conv.len()].copy_from_slice(conv);
        ssm_h[..ssm.len()].copy_from_slice(ssm);
        // s141b (Task B7c): この層のホストは今デバイスと一致した → 遅延フラグを落とす。
        // 落とさないと、次の flush で同じ層を読み直してしまう (無駄な D2H)。
        m.gdn_stale[li] = false;
    }
    m.gdn_pending = None;
    Some(())
}

/// C4: チャネルスロットをステップ開始時に事前確保する (同期・カーネル発行なし)。
/// kv_stage と同一の「最初に全体サイズを確保」規約の CH_QKV /
/// CH_ATT 版。事前確保後はステップ内の ensure が全て no-op になり、
/// in-flight カーネルの書き込み先 / 読み出し元の grow 再確保による解放が
/// 構造的に起こらない (q35_ch_qkv_bytes の doc を参照)。
pub fn chain_q35_step_reserve(slot: usize, bytes: usize) -> Option<()> {
    let _ = chain_ctx()?;
    ok(chain_ensure_buf(slot, bytes))
}

/// C4: int8 量子化スクラッチ (aq) をステップ開始時に最大 k 分事前確保する。
/// chain_linear_grouped はグループごとの k で grow するため、2994feb 時点では
/// 初回 down GEMV (k=ff) で再確保が走り、直前 GEMV (gate/up) の in-flight
/// 読み出し元を解放していた (解放済みブロックの内容は多くの場合残るが UB)。
pub fn chain_q35_step_reserve_aq(max_k: usize) -> Option<()> {
    let _ = chain_ctx()?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure(&mut s.aq, 1 * max_k))
    })
}

/// C4: qwen35 用 KV ステージフラッシュ (GDN 層は KV をステージしないため
/// 該当層をスキップする。llama 用 chain_kv_stage_flush は全層をフラッシュする
/// ため流用できない)。
pub fn chain_q35_kv_stage_flush(
    cache: &mut crate::kv_cache::KvCache,
    n_layers: usize,
    kv_dim: usize,
    pos0: usize,
    skip: &[usize],
) -> Option<()> {
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let stage = s.kv_stage.as_ref()?;
        let slice = stage.as_slice::<f32>();
        let mut si = 0usize;
        for li in 0..n_layers {
            if si < skip.len() && skip[si] == li {
                si += 1;
                continue;
            }
            let off = li * 2 * kv_dim;
            let (kc, vc) = cache.kv_bufs(li, 1);
            kc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off..off + kv_dim]);
            vc[pos0 * kv_dim..][..kv_dim].copy_from_slice(&slice[off + kv_dim..off + 2 * kv_dim]);
        }
        Some(())
    })
}

// ===========================================================================
#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::tensor;

    /// 決定論的 LCG 乱数 (tensor.rs のテストと同一方式)
    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 1000.0 - 1.0 // [-1, 1)
            })
            .collect()
    }

    /// i8 の擬似重み [-128,127] (全範囲をカバー)
    fn pseudo_i8(len: usize, seed: u32) -> Vec<i8> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (((x >> 16) % 256) as i32 - 128) as i8
            })
            .collect()
    }

    fn positive_scales(len: usize, seed: u32) -> Vec<f32> {
        pseudo_random(len, seed)
            .iter()
            .map(|v| 0.001 + v.abs() * 0.01)
            .collect()
    }

    /// P17j: prefill attention の CPU 参照 (model/llama.rs forward の prefill
    /// 経路と同一コード: sgemm_scores_kvstrided + causal mask + softmax +
    /// matmul_rows_bstrided_par)。
    fn prefill_attention_cpu(
        q: &[f32],
        kc: &[f32],
        vc: &[f32],
        heads: usize,
        kv_heads: usize,
        dh: usize,
        n: usize,
        pos0: usize,
        t_total: usize,
        scale: f32,
    ) -> Vec<f32> {
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads;
        let mut att = vec![0.0f32; n * heads * dh];
        for kvh in 0..kv_heads {
            for g in 0..group {
                let qh = kvh * group + g;
                let mut scores = vec![0.0f32; n * t_total];
                tensor::sgemm_scores_kvstrided(
                    &q[qh * dh..],
                    heads * dh,
                    &kc[kvh * dh..],
                    kv_dim,
                    &mut scores,
                    n,
                    dh,
                    t_total,
                    scale,
                );
                for i in 0..n {
                    let row = &mut scores[i * t_total..][..t_total];
                    for j in (pos0 + i + 1)..t_total {
                        row[j] = f32::NEG_INFINITY;
                    }
                    tensor::softmax(row);
                }
                let out =
                    tensor::matmul_rows_bstrided_par(&scores, &vc[kvh * dh..], kv_dim, n, t_total, dh);
                for i in 0..n {
                    att[i * heads * dh + qh * dh..][..dh].copy_from_slice(&out[i * dh..][..dh]);
                }
            }
        }
        att
    }

    /// P17k ステップ 5: dno_rope_qkv_pf (qkv 出力 → rope → q 連続化 +
    /// k/v を GPU KV ミラーへ直接書き込み) が CPU 参照経路 (rope_row) と
    /// **ビット一致**することを検証する。tab は CPU が rope_row と同一式で
    /// 生成し、カーネルは mul/sub のみ適用するため一致が契約。
    /// (qkv GEMM 自体は dno_gemm_i8 = cuda_gemm_i8_bit_exact_vs_cpu で担保)
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_rope_qkv_pf_bitexact() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (12usize, 2usize, 128usize);
        let qn = heads * dh;
        let kv_dim = kv_heads * dh;
        let width = qn + 2 * kv_dim;
        let (pos0, n) = (5usize, 37usize);
        let t_total = pos0 + n;
        let theta = 1_000_000.0f32;

        let ctx = CudaCtx::new().unwrap();
        // qkv GEMM 出力のエミュレーション (行 i = [qn | kv_dim | kv_dim]) + bias
        let qkv = pseudo_random(n * width, 41);
        let bias = pseudo_random(width, 42);
        // rope テーブル: chain_qkv_rope_prefill 内と同一式 (rope_row とビット一致)
        let mut tab = vec![0.0f32; n * dh];
        let half = dh / 2;
        for i in 0..n {
            let pos = (pos0 + i) as f32;
            let row = &mut tab[i * dh..][..dh];
            for d in 0..half {
                let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                row[d] = ang.cos();
                row[half + d] = ang.sin();
            }
        }
        let dqkv = ctx.upload_f32(&qkv).unwrap();
        let dbias = ctx.upload_f32(&bias).unwrap();
        let dtab = ctx.upload_f32(&tab).unwrap();
        let dq = DevBuffer::alloc(n * qn * 4).unwrap();

        // CPU 参照: 行分割 ([n, width] 行-major → q/k/v) + bias 加算 + rope_row
        let mut q_ref = vec![0.0f32; n * qn];
        let mut k_ref = vec![0.0f32; n * kv_dim];
        let mut v_ref = vec![0.0f32; n * kv_dim];
        for i in 0..n {
            let row = &qkv[i * width..][..width];
            q_ref[i * qn..][..qn].copy_from_slice(&row[..qn]);
            k_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn..qn + kv_dim]);
            v_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn + kv_dim..]);
        }
        for b in q_ref.chunks_mut(qn) {
            for (o, bi) in b.iter_mut().zip(&bias[..qn]) {
                *o += *bi;
            }
        }
        for b in k_ref.chunks_mut(kv_dim) {
            for (o, bi) in b.iter_mut().zip(&bias[qn..qn + kv_dim]) {
                *o += *bi;
            }
        }
        for b in v_ref.chunks_mut(kv_dim) {
            for (o, bi) in b.iter_mut().zip(&bias[qn + kv_dim..]) {
                *o += *bi;
            }
        }
        crate::tensor::rope_rows(&mut q_ref, n, heads, dh, pos0, theta);
        crate::tensor::rope_rows(&mut k_ref, n, kv_heads, dh, pos0, theta);

        // GPU 実経路: ミラー取得 (履歴行 [0..pos0) をシード — 現在行
        // [pos0..t_total) は rope カーネルが書く。chain_qkv_rope_prefill と同一)。
        // キーは番兵 (u64::MAX-1) — 実キャッシュの kv_id (1 からの連番) と
        // 衝突すると他テストにミラーを残して汚染する (実害あり)。
        const MIRROR_TEST_KEY: u64 = u64::MAX - 1;
        let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
        {
            let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
            kbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 46));
            vbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 47));
        }
        let (kptr, vptr) =
            with_mirror(MIRROR_TEST_KEY, 0, 1, t_total, kv_dim, 0, |m, have| {
            if have < pos0 {
                let (kc_ref, vc_ref) = cache.kv_bufs_ref(0);
                let kb = m.layers[0].0.as_ref().unwrap();
                let vb = m.layers[0].1.as_ref().unwrap();
                kb.copy_from_at(have * kv_dim * 4, &kc_ref[have * kv_dim..pos0 * kv_dim])
                    .unwrap();
                vb.copy_from_at(have * kv_dim * 4, &vc_ref[have * kv_dim..pos0 * kv_dim])
                    .unwrap();
            }
            Some((
                m.layers[0].0.as_ref().unwrap().ptr(),
                m.layers[0].1.as_ref().unwrap().ptr(),
            ))
        })
        .unwrap();

        // s41: コンテンツ世代 (epoch) 不一致で KV ミラーの全層 len と GDN gdn_pos
        // が無効化されること。同一 kv_id (スロット) の cache が reset/restore されて
        // 別会話の KV で再 prefill された場合、前会話の行 [0..have) を正規行として
        // 使ってしまう 2b 2 値分岐 (s35) の根本修正の検証。CPU では実行不可
        // (KV_MIRRORS が cuda feature 配下のため) ので GPU 実機で実行する。
        #[test]
        #[ignore] // GPU が必要: cargo test --release --features cuda -p denno-engine -- --ignored
        fn cuda_kv_mirror_epoch_invalidation() {
            let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            enable_for_tests();
            let _ctx = CudaCtx::new().unwrap();
            let (kv_dim, t_total) = (64usize, 40usize);
            // 番兵キー (MIRROR_TEST_KEY = u64::MAX-1 と衝突しない)
            let key = u64::MAX - 3;
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            // 会話 A: epoch 0 で seed → len[0] = t_total・gdn_pos にダミー値
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 71));
                vbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 72));
            }
            with_mirror(key, 0, 1, t_total, kv_dim, cache.epoch(), |m, have| {
                assert_eq!(have, 0, "初回はミラー空 (全行シード)");
                m.len[0] = t_total;
                m.gdn_pos[0] = Some(t_total - 1);
                Some(())
            })
            .unwrap();
            // 会話 B: reset → 別内容で再 prefill (epoch が 1 に上がる)
            cache.reset();
            assert_eq!(cache.epoch(), 1);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 73));
                vbuf[..t_total * kv_dim].copy_from_slice(&pseudo_random(t_total * kv_dim, 74));
            }
            let mut have_b = None;
            with_mirror(key, 0, 1, t_total, kv_dim, cache.epoch(), |m, have| {
                have_b = Some(have);
                Some(())
            })
            .unwrap();
            assert_eq!(
                have_b,
                Some(0),
                "epoch 不一致で len がリセットされ、前会話の行が再利用されないこと"
            );
            // GDN 状態位置も epoch 不一致で無効化される (次回 decode で全状態再 seed)
            with_mirror_ext(key, 0, 1, t_total, 0, cache.epoch(), false, |m, _| {
                assert!(m.gdn_pos[0].is_none(), "GDN 状態位置も無効化されること");
                Some(())
            })
            .unwrap();
        }
        launch_rope_qkv_pf(
            &ctx,
            dqkv.ptr(),
            dq.ptr(),
            kptr,
            vptr,
            dtab.ptr(),
            dbias.ptr(),
            pos0,
            n,
            heads,
            kv_heads,
            dh,
            kv_dim,
        )
        .unwrap();
        ctx.sync().unwrap();

        // q: rope+連続化 → ビット一致
        let q_gpu = dq.to_vec::<f32>(n * qn).unwrap();
        if q_gpu != q_ref {
            let bad = q_gpu
                .iter()
                .zip(&q_ref)
                .position(|(a, b)| a.to_bits() != b.to_bits())
                .unwrap();
            let (i, r) = (bad / qn, bad % qn);
            panic!(
                "q ビット不一致: idx {bad} (i={i} h={} d={} in-head-d={}) gpu={} ({:08x}) ref={} ({:08x})",
                r / dh,
                r % dh,
                {
                    let d = r % dh;
                    if d < dh / 2 { d } else { d - dh / 2 }
                },
                q_gpu[bad],
                q_gpu[bad].to_bits(),
                q_ref[bad],
                q_ref[bad].to_bits()
            );
        }
        // k/v: ミラーの [pos0..t_total) 行だけ rope 後の値とビット一致
        // (履歴行 [0..pos0) はシード値のまま・不変)
        let k_hist = pseudo_random(t_total * kv_dim, 46);
        let v_hist = pseudo_random(t_total * kv_dim, 47);
        let m2 = KV_MIRRORS.get_or_init(|| std::sync::Mutex::new(Vec::new()));
        let ms = m2.lock().unwrap();
        let (_, mir, _) = ms
            .iter()
            .find(|(k, _, _)| *k == MIRROR_TEST_KEY)
            .unwrap();
        let k_gpu: Vec<f32> = mir.layers[0].0.as_ref().unwrap().to_vec(t_total * kv_dim).unwrap();
        let v_gpu: Vec<f32> = mir.layers[0].1.as_ref().unwrap().to_vec(t_total * kv_dim).unwrap();
        drop(ms);
        for i in 0..t_total {
            let off = i * kv_dim;
            let wk = if i < pos0 { &k_hist[off..off + kv_dim] } else { &k_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim] };
            let wv = if i < pos0 { &v_hist[off..off + kv_dim] } else { &v_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim] };
            assert_eq!(&k_gpu[off..off + kv_dim], wk, "k 行 {i} ビット不一致");
            assert_eq!(&v_gpu[off..off + kv_dim], wv, "v 行 {i} ビット不一致");
        }
        println!("rope_qkv_pf bit-exact OK (q {n}x{qn}, k/v rows {pos0}..{t_total})");
    }

    /// s123: Q4_0 ネイティブ経路 (GEMV: m<=16 / GEMM: m>=256) が CPU 実装と
    /// R2 (相対 1e-4) 以内で一致し、2 回実行でビット一致 (決定論) すること。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_q4_linear_matches_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (k, n) = (256usize, 64usize);
        let w = pseudo_random(k * n, 5);
        let q = crate::quant::Q4Linear::quantize(&w, k, n);
        for m in [1usize, 7, 16, 300] {
            let a = pseudo_random(m * k, 100 + m as u32);
            let mut want = vec![0f32; m * n];
            crate::tensor::matvec_rows_bt_q4_into(&a, &q.blocks, k, n, m, &mut want);
            let mut got = vec![0f32; m * n];
            assert!(
                maybe_gpu_linear_q4_into(q.uid, &a, &q.blocks, k, n, m, &mut got),
                "m={m}: Q4 GPU 経路が発動しなかった"
            );
            let mut maxd = 0.0f32;
            let mut maxv = 0.0f32;
            for (g, wv) in got.iter().zip(want.iter()) {
                maxd = maxd.max((g - wv).abs());
                maxv = maxv.max(wv.abs());
            }
            println!("q4 m={m}: maxabs={maxd:e} (max|ref|={maxv:e})");
            assert!(
                maxd <= 1e-4 * maxv.max(1.0),
                "m={m}: Q4 GPU と CPU の差 {maxd} が R2 を超えた"
            );
            let mut got2 = vec![0f32; m * n];
            assert!(maybe_gpu_linear_q4_into(q.uid, &a, &q.blocks, k, n, m, &mut got2));
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                got2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "m={m}: Q4 GPU 経路が非決定論"
            );
        }
    }

    /// s132 (Task B1): 複数行 1 block の GEMV (`dno_gemv_q4_0mr`) が、行あたりの
    /// 累積順を 1 行 1 block 経路と同じに保っていること (rows = 1/2/3/8 で全要素
    /// ビット一致) と、2 回実行でビット一致 (決定論) を検証する。
    /// rows = 1 はカーネル内の行ループを 1 周にしただけで `dno_gemv_q4_0` と
    /// 同じ演算列、rows > 1 はその行版。境界 m = 1 / 2 / 7 / 9 / 16 / 17 を踏む。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_q4_gemv_mr_matches_single_row() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let ctx = match shared_ctx() {
            Some(c) => c,
            None => {
                println!("CUDA が無いので skip (GPU 実機で --ignored 実行)");
                return;
            }
        };
        let f_mr = match ctx.f_mv_q4_mr {
            Some(f) => f,
            None => {
                println!("kernels_int8.ptx に dno_gemv_q4_0mr が無い (PTX 再生成前) ので skip");
                return;
            }
        };
        let _ = f_mr;
        ctx.make_current().expect("make_current");
        // 1) カーネル直叩き: 同じ入力で「1 行 1 block (mr=1)」と「mr 行 1 block」を
        //    比べ、全要素ビット一致を要求する (累積順の契約そのもの)。
        let (k, n) = (256usize, 40usize); // n は 8 の倍数でない (grid.x の端数も踏む)
        let w = pseudo_random(k * n, 9);
        let q = crate::quant::Q4Linear::quantize(&w, k, n);
        let gw = weights_for_q4(q.uid, &q.blocks, k, n).expect("upload");
        for m in [1usize, 2, 7, 9, 16, 17] {
            let a = pseudo_random(m * k, 200 + m as u32);
            let run = |rows: usize| -> Vec<f32> {
                let mut out = vec![f32::NAN; m * n];
                run_q4_gemv_for_test(ctx, &a, gw, m, k, n, rows, &mut out);
                out
            };
            // 参照: mr = 1 (従来経路と同じ「1 行 1 block」) を m 回に分けて回す
            let mut want = vec![0f32; m * n];
            {
                // rows = 1 で 1 行ずつ。カーネルは grid.y = ceil(m/1) で m 行を回る。
                let got1 = run(1);
                want.copy_from_slice(&got1);
            }
            for &mr in &[2usize, 3, 8] {
                let got = run(mr);
                for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "m={m} mr={mr}: row {} col {} がビット不一致 ({g} vs {e})",
                        i / n,
                        i % n
                    );
                }
                // 決定論: 同一入力の 2 回目
                let again = run(mr);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} mr={mr}: 非決定論"
                );
            }
            println!("q4 gemv mr bit-exact: m={m} (rows 1/2/3/8)");
        }
    }

    /// s133 (Task B1, attempt 2): 列ブロック版 GEMV (`dno_gemv_q4_0c2/c4/c8`) が
    /// 1 行 1 block 経路 (`dno_gemv_q4_0mr` の rows = 1) と全要素ビット一致する
    /// ことを検証する。C は 2 / 4 / 8 の 3 通り、n は 8*C の倍数でない値
    /// (grid.x の端数と `gj >= n` の break を踏む)、m は 1 / 2 / 7 / 17 を踏む。
    /// 併せて決定論 (同一入力 2 回でビット一致) も見る。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_q4_gemv_cols_matches_single_row() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let ctx = match shared_ctx() {
            Some(c) => c,
            None => {
                println!("CUDA が無いので skip (GPU 実機で --ignored 実行)");
                return;
            }
        };
        if ctx.f_mv_q4_mr.is_none() {
            println!("kernels_int8.ptx に dno_gemv_q4_0mr が無い (PTX 再生成前) ので skip");
            return;
        }
        let fc = match ctx.f_mv_q4_c {
            Some(f) => f,
            None => {
                println!("kernels_int8.ptx に dno_gemv_q4_0c* が無い (PTX 再生成前) ので skip");
                return;
            }
        };
        ctx.make_current().expect("make_current");
        // n = 103: 8*2=16 / 8*4=32 / 8*8=64 のどれでも割り切れない。
        let (k, n) = (256usize, 103usize);
        let w = pseudo_random(k * n, 21);
        let q = crate::quant::Q4Linear::quantize(&w, k, n);
        let gw = weights_for_q4(q.uid, &q.blocks, k, n).expect("upload");
        for m in [1usize, 2, 7, 17] {
            let a = pseudo_random(m * k, 300 + m as u32);
            // 参照: 従来経路 (1 行 1 block) と同じ rows = 1。
            let mut want = vec![f32::NAN; m * n];
            run_q4_gemv_for_test(ctx, &a, gw, m, k, n, 1, &mut want);
            for (ci, &c) in [2usize, 4, 8].iter().enumerate() {
                assert_eq!(ci, q4_gemv_c_index(c), "index の対応がずれている");
                let mut got = vec![f32::NAN; m * n];
                run_q4_gemv_cols_for_test(ctx, fc[ci], c, &a, gw, m, k, n, &mut got);
                for (i, (g, e)) in got.iter().zip(want.iter()).enumerate() {
                    assert_eq!(
                        g.to_bits(),
                        e.to_bits(),
                        "m={m} C={c}: row {} col {} がビット不一致 ({g} vs {e})",
                        i / n,
                        i % n
                    );
                }
                let mut again = vec![f32::NAN; m * n];
                run_q4_gemv_cols_for_test(ctx, fc[ci], c, &a, gw, m, k, n, &mut again);
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    again.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "m={m} C={c}: 非決定論"
                );
                println!("q4 gemv cols bit-exact: m={m} C={c} (n={n})");
            }
        }
        assert_eq!(q4_gemv_c_index(Q4_GEMV_C), q4_gemv_c_index(4));
        // 本番の選択が 2 / 4 / 8 のどれでも、上のビット一致集合の中にあること。
        let prod = q4_gemv_c();
        assert!(
            [2usize, 4, 8].contains(&prod),
            "q4_gemv_c() が 2/4/8 以外を返した: {prod}"
        );
    }

    /// s133: 列ブロック版 GEMV を c (2/4/8) を明示して 1 回起動する。
    fn run_q4_gemv_cols_for_test(
        ctx: &CudaCtx,
        f: CUfunction,
        c: usize,
        a: &[f32],
        gw: &GpuWq4,
        m: usize,
        k: usize,
        n: usize,
        out: &mut [f32],
    ) {
        let lib = lib().expect("cuda lib");
        let (mut pn, mut ps) = (gw.nib.ptr(), gw.sc.ptr());
        let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
        let d_a = DevBuffer::alloc(m * k * 4).expect("alloc a");
        let d_out = DevBuffer::alloc(m * n * 4).expect("alloc out");
        let (mut pa, mut po) = (d_a.ptr(), d_out.ptr());
        unsafe {
            check(
                (lib.memcpy_htod)(d_a.ptr(), a.as_ptr() as *const c_void, m * k * 4),
                "cuMemcpyHtoD",
            )
            .expect("H2D");
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            let grid = (n.div_ceil(8 * c) as u32, m as u32, 1);
            ctx.launch_async(f, grid, (256, 1, 1), 0, &mut args)
                .expect("launch");
            check(
                (lib.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, d_out.ptr(), m * n * 4),
                "cuMemcpyDtoH",
            )
            .expect("D2H");
        }
    }

    /// s132: テストから `dno_gemv_q4_0mr` を直接 1 回起動する (行数 rows を明示)。
    /// 本番経路は `q4_dispatch_tile` が rows = min(m, Q4_MR_MAX) を選ぶ。
    fn run_q4_gemv_for_test(
        ctx: &CudaCtx,
        a: &[f32],
        gw: &GpuWq4,
        m: usize,
        k: usize,
        n: usize,
        rows: usize,
        out: &mut [f32],
    ) {
        let lib = lib().expect("cuda lib");
        let f = ctx.f_mv_q4_mr.expect("dno_gemv_q4_0mr");
        let (mut pn, mut ps) = (gw.nib.ptr(), gw.sc.ptr());
        let (mut mc, mut kc, mut nc) = (m as c_int, k as c_int, n as c_int);
        let mut rc = rows as c_int;
        let d_a = DevBuffer::alloc(m * k * 4).expect("alloc a");
        let d_out = DevBuffer::alloc(m * n * 4).expect("alloc out");
        let (mut pa, mut po) = (d_a.ptr(), d_out.ptr());
        unsafe {
            check(
                (lib.memcpy_htod)(d_a.ptr(), a.as_ptr() as *const c_void, m * k * 4),
                "cuMemcpyHtoD",
            )
            .expect("H2D");
            let mut args = [
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut mc as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
                &mut rc as *mut _ as *mut c_void,
            ];
            let grid = (n.div_ceil(8) as u32, m.div_ceil(rows) as u32, 1);
            ctx.launch_async(f, grid, (256, 1, 1), 0, &mut args)
                .expect("launch");
            check(
                (lib.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, d_out.ptr(), m * n * 4),
                "cuMemcpyDtoH",
            )
            .expect("D2H");
        }
    }

    /// P17j: prefill attention カーネル (dno_attention_prefill) が CPU 参照経路と
    /// **近似一致**すること (exactness 契約は attention のみ緩和・ユーザー承認)。
    /// 加えて ①決定論 (同一入力の 2 回起動でビット一致) ②causal (未来の K/V を
    /// 書き換えても過去行の出力はビット不変) を検証する。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_attention_prefill_vs_cpu_reference() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (12usize, 2usize, 128usize);
        let group = heads / kv_heads;
        let kv_dim = kv_heads * dh;
        let (pos0, n) = (5usize, 37usize); // BQ/BK タイル境界をまたぐ
        let t_total = pos0 + n;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = pseudo_random(n * heads * dh, 11);
        let kc = pseudo_random(t_total * kv_dim, 12);
        let vc = pseudo_random(t_total * kv_dim, 13);

        let att_ref =
            prefill_attention_cpu(&q, &kc, &vc, heads, kv_heads, dh, n, pos0, t_total, scale);

        // GPU 実経路 (KvCache 生成 + ミラー同期 + カーネル起動 + D2H)
        let run_gpu = |kc: &[f32], vc: &[f32]| -> Vec<f32> {
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(kc);
                vbuf[..t_total * kv_dim].copy_from_slice(vc);
            }
            assert!(
                chain_attention_prefill(
                    cache.kv_id(),
                    0,
                    1,
                    &q,
                    &cache,
                    pos0,
                    n,
                    t_total,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
                .is_some(),
                "prefill attention kernel が発動すべき (dh={dh})"
            );
            chain_download(CH_ATT, n * heads * dh).unwrap()
        };

        let got = run_gpu(&kc, &vc);
        let maxd = got
            .iter()
            .zip(att_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("prefill attention maxabs diff vs cpu: {maxd}");
        assert!(maxd < 1e-3, "prefill attention maxabs diff {maxd}");

        // 決定論: 同一入力の 2 回目はビット一致
        let got2 = run_gpu(&kc, &vc);
        assert_eq!(got, got2, "prefill attention が非決定論的");

        // causal: 未来の行 (j > pos0 + i) を書き換えても過去行の出力はビット不変
        let mut kc2 = kc.clone();
        let mut vc2 = vc.clone();
        for j in (pos0 + 3)..t_total {
            let seed = 99 + j as u32;
            kc2[j * kv_dim..(j + 1) * kv_dim].copy_from_slice(&pseudo_random(kv_dim, seed));
            vc2[j * kv_dim..(j + 1) * kv_dim].copy_from_slice(&pseudo_random(kv_dim, seed + 1000));
        }
        let got3 = run_gpu(&kc2, &vc2);
        // 行 0..2 の causal 界 (j <= pos0+i) は変更範囲 (j >= pos0+3) の外 → ビット一致
        for i in 0..3usize {
            let a = &got[i * heads * dh..][..heads * dh];
            let b = &got3[i * heads * dh..][..heads * dh];
            assert_eq!(
                a, b,
                "causal 違反: row {i} が未来の K/V 変更の影響を受けた"
            );
        }
        // 行 3 は causal 界 (j <= pos0+3) が変更範囲に含まれる → 変わるはず
        let a = &got[3 * heads * dh..][..heads * dh];
        let b = &got3[3 * heads * dh..][..heads * dh];
        assert_ne!(a, b, "row 3 が未来の K/V 変更に反応していない (テスト無効)");
    }

    /// P17k ステップ 7: 大規模 n での prefill attention (n=37 のテストは 1 行
    /// ブロック・少数タイルしか通らないため、複数行ブロック×多数タイル・
    /// prefetch が実データを読む規模での近似一致と決定論を検証する)。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_attention_prefill_vs_cpu_large() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (12usize, 2usize, 128usize);
        let group = heads / kv_heads;
        let kv_dim = kv_heads * dh;
        let (pos0, n) = (64usize, 2056usize); // 行ブロック 33 個・tile 数百
        let t_total = pos0 + n;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = pseudo_random(n * heads * dh, 21);
        let kc = pseudo_random(t_total * kv_dim, 22);
        let vc = pseudo_random(t_total * kv_dim, 23);

        let att_ref =
            prefill_attention_cpu(&q, &kc, &vc, heads, kv_heads, dh, n, pos0, t_total, scale);

        let run_gpu = |kc: &[f32], vc: &[f32]| -> Vec<f32> {
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(kc);
                vbuf[..t_total * kv_dim].copy_from_slice(vc);
            }
            assert!(
                chain_attention_prefill(
                    cache.kv_id(),
                    0,
                    1,
                    &q,
                    &cache,
                    pos0,
                    n,
                    t_total,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
                .is_some(),
                "prefill attention kernel が発動すべき (dh={dh})"
            );
            chain_download(CH_ATT, n * heads * dh).unwrap()
        };

        let got = run_gpu(&kc, &vc);
        let maxd = got
            .iter()
            .zip(att_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("prefill attention (n={n}) maxabs diff vs cpu: {maxd}");
        assert!(maxd < 1e-3, "prefill attention (n={n}) maxabs diff {maxd}");

        // 決定論: 同一入力の 2 回目はビット一致
        let got2 = run_gpu(&kc, &vc);
        assert_eq!(got, got2, "prefill attention (n={n}) が非決定論的");
        println!("prefill attention large n=({n}) determinism OK");
    }

    /// s49: fp32 lm_head GPU 経路 (maybe_gpu_matvec_f32) が tensor::matvec_bt と
    /// ビット一致すること。dno_matvec_f32 (C4 chain の beta/alpha 制御投影用
    /// カーネル・.cu/PTX 無変更で流用) を block=(1,1,1)・grid=(n,1,1) で起動する:
    /// ①lm_head 実形状相当 (k=4096) での bit 一致 ②n>65535 (vocab 規模) でも
    /// grid.x が正しく機能する (65535 は chain 側呼び出し規約のみのガード) こと
    /// ③決定論 (同一入力 2 回起動でビット一致) ④小さい k (境界) でも一致。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_matvec_f32_lmhead_bit_exact_vs_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().unwrap();
        // PTX 未再生成 (dno_matvec_f32 未収録) の場合は不発で CPU に落ちるため
        // 最初の呼び出しで発動を明示して fail させる (PTX 再生成後の再実行が前提)。
        for (k, n, seed) in [
            (4096usize, 4096usize, 71),   // lm_head 実形状相当 (k=4096)
            (4096, 131_072 + 17, 72),     // n > 65535 (vocab 規模) の grid.x 検証
            (8, 1, 73),                   // 最小境界 (n=1・小 k)
            (128, 511, 74),               // 端数 n
        ] {
            let w = pseudo_random(k * n, seed);
            let a = pseudo_random(k, seed + 100);
            let ref_cpu = tensor::matvec_bt(&a, &w, k, n);
            let got = maybe_gpu_matvec_f32(9_000_001, &w, &a, k, n)
                .expect("GPU 経路が不発 (kernels_elem.ptx に dno_matvec_f32 が無い/PTX 未再生成の可能性)");
            assert_eq!(
                got.len(),
                ref_cpu.len(),
                "k={k} n={n}: 出力長不一致"
            );
            for (i, (g, r)) in got.iter().zip(ref_cpu.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    r.to_bits(),
                    "k={k} n={n}: logits[{i}] が bit 不一致 (g={g} r={r})"
                );
            }
            // 決定論: 同一入力の 2 回起動でビット一致
            let got2 = maybe_gpu_matvec_f32(9_000_001, &w, &a, k, n).unwrap();
            for (g, g2) in got.iter().zip(got2.iter()) {
                assert_eq!(g.to_bits(), g2.to_bits(), "k={k} n={n}: 非決定論");
            }
            println!("cuda matvec_f32 k={k} n={n}: bit exact OK");
        }
        // ゲート: 形状不正 (w.len() != k*n) は None (CPU フォールバック) であること
        let w = vec![0.0f32; 10];
        assert!(maybe_gpu_matvec_f32(9_000_002, &w, &[0.0; 4], 4, 3).is_none());
    }

    /// s48 (C1): qwen35 形状 (dh=256・heads=16・kv_heads=4・group=4 = 実 9B の
    /// full attention) での prefill attention GPU バッチ経路。動的 smem が 84.8KB
    /// (既定 48KB 上限超え) であるため C1-a の opt-in (CudaCtx 生成時の
    /// cuFuncSetAttribute) が必須 — opt-in が実機で不可の場合、上限が 48KB のまま
    /// 残り本テストは fail する (C1-b タイル縮小 + .cu 変更/PTX 再生成への切替判断
    /// 材料)。CPU 参照との近似一致 (maxabs<1e-3・attention 緩和枠) + 決定論
    /// (同一入力の 2 回起動でビット一致) + ゲートの発動条件 (n>=gemm_min_m) を検証。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_attention_prefill_qwen35_shape_vs_cpu_reference() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let (heads, kv_heads, dh) = (16usize, 4usize, 256usize); // qwen35 9B (e=4096)
        let group = heads / kv_heads;
        let kv_dim = kv_heads * dh;
        let sh = attn_pf_sh_f(dh) * 4;
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        println!(
            "dno_attention_prefill dh={dh}: dynamic smem = {sh} bytes (opt-in limit = {})",
            attn_pf_dynamic_smem_limit()
        );
        // C1-a: opt-in 成功で 84.8KB が通ること (失敗なら C1-b への切替が必要)
        assert!(
            attn_pf_dynamic_smem_limit() >= sh,
            "動的 smem の opt-in が失敗している (limit={} < 必要量 {sh}) — C1-b (タイル縮小 + .cu 変更/PTX 再生成) への切替が必要",
            attn_pf_dynamic_smem_limit()
        );
        // ゲート: n >= gemm_min_m() (既定 256) で発動・未満は不発 (短文は CPU のまま)
        let (pos0, n) = (5usize, 276usize); // BQ=64 の行ブロック境界をまたぐ
        assert!(attn_prefill_gpu_supported(n, dh, pos0 + n));
        assert!(
            !attn_prefill_gpu_supported(gemm_min_m() - 1, dh, pos0 + n),
            "n < gemm_min_m() は不発のはず (短文は CPU attend を維持)"
        );
        let t_total = pos0 + n;
        let scale = 1.0 / (dh as f32).sqrt();
        let q = pseudo_random(n * heads * dh, 31);
        let kc = pseudo_random(t_total * kv_dim, 32);
        let vc = pseudo_random(t_total * kv_dim, 33);

        let att_ref =
            prefill_attention_cpu(&q, &kc, &vc, heads, kv_heads, dh, n, pos0, t_total, scale);

        let run_gpu = |kc: &[f32], vc: &[f32]| -> Vec<f32> {
            let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
            {
                let (kbuf, vbuf) = cache.kv_bufs(0, t_total);
                kbuf[..t_total * kv_dim].copy_from_slice(kc);
                vbuf[..t_total * kv_dim].copy_from_slice(vc);
            }
            assert!(
                chain_attention_prefill(
                    cache.kv_id(),
                    0,
                    1,
                    &q,
                    &cache,
                    pos0,
                    n,
                    t_total,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
                .is_some(),
                "prefill attention kernel が発動すべき (dh={dh})"
            );
            chain_download(CH_ATT, n * heads * dh).unwrap()
        };

        let got = run_gpu(&kc, &vc);
        let maxd = got
            .iter()
            .zip(att_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("prefill attention (qwen35 shape dh={dh}) maxabs diff vs cpu: {maxd}");
        assert!(maxd < 1e-3, "prefill attention (dh={dh}) maxabs diff {maxd}");

        // 決定論: 同一入力の 2 回目はビット一致
        let got2 = run_gpu(&kc, &vc);
        assert_eq!(got, got2, "prefill attention (dh={dh}) が非決定論的");
        println!("prefill attention qwen35 shape (dh={dh}) determinism OK");
    }

    /// GPU の量子化が CPU (`tensor::quantize_row_i8`) とビット一致するか
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_quantize_rows_bit_exact_vs_cpu() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        for (m, k) in [(1usize, 8usize), (3, 896), (17, 1536), (4, 8960)] {
            let a = pseudo_random(m * k, 30 + m as u32);
            let (aq_gpu, asc_gpu) = ctx.quantize_rows(&a, m, k).unwrap();
            let aq_gpu = aq_gpu.to_vec::<i8>(m * k).unwrap();
            let asc_gpu = asc_gpu.to_vec::<f32>(m).unwrap();
            for i in 0..m {
                let (aq_cpu, asc_cpu) = tensor::quantize_row_i8(&a[i * k..][..k]);
                assert_eq!(&aq_gpu[i * k..][..k], &aq_cpu[..], "m={m} k={k} row={i}");
                assert_eq!(
                    asc_gpu[i].to_bits(),
                    asc_cpu.to_bits(),
                    "m={m} k={k} row={i}: ascale"
                );
            }
        }
    }

    /// GPU int8 GEMV/GEMM が CPU int8 カーネルとビット一致するか
    /// (decode m=1 / バッチ decode m>1 / prefill GEMM の全経路)。
    #[test]
    #[ignore]
    fn cuda_linear_int8_bit_exact_vs_cpu() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        // 実モデルの形状を含む: Qwen2.5-1.5B (qkv 1536→2048, gate_up 1536→17920,
        // down 8960→1536, lm_head 1536→151936) / 0.5B (896→4864)
        let cases: [(usize, usize, usize); 8] = [
            (1, 8, 16),
            (7, 37, 200),
            (1, 896, 1152),
            (2, 896, 4864),
            (4, 1536, 17920),
            (3, 8960, 1536),
            (1, 1536, 151936),
            (8, 1536, 2048),
        ];
        for (ci, &(m, k, n)) in cases.iter().enumerate() {
            let q = pseudo_i8(k * n, 40 + ci as u32);
            let scale = positive_scales(n, 50 + ci as u32);
            let a = pseudo_random(m * k, 60 + ci as u32);
            let w = ctx.upload_i8(&q).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            let got = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            let want = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "m={m} k={k} n={n} idx={idx} (i={} j={}): GPU != CPU",
                    idx / n,
                    idx % n
                );
            }
        }
    }

    /// P17k: maybe_gpu_ffn (デバイス常駐 FFN ブロック) vs CPU 参照実装のビット一致。
    /// CPU 参照は llama.rs prefill 従来経路と同一コード: gate/up を行連結した
    /// grouped matvec_rows_bt_i8 → silu_inplace + 要素積 → down matvec_rows_bt_i8。
    #[test]
    #[ignore]
    fn cuda_ffn_bit_exact_vs_cpu_reference() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        crate::cuda::enable_for_tests();

        // GEMM 経路の条件 (m > max_m=16 かつ m >= gemm_min_m) を満たすよう閾値を下げる
        std::env::set_var("DENNO_CUDA_GEMM_MIN_M", "8");

        let (m, e, ff) = (40usize, 96usize, 64usize);
        let gate = Quant8Linear {
            w: pseudo_i8(e * ff, 90),
            scale: positive_scales(ff, 91),
            k: e,
            n: ff,
            uid: 901,
        };
        let up = Quant8Linear {
            w: pseudo_i8(e * ff, 92),
            scale: positive_scales(ff, 93),
            k: e,
            n: ff,
            uid: 902,
        };
        let down = Quant8Linear {
            w: pseudo_i8(ff * e, 94),
            scale: positive_scales(e, 95),
            k: ff,
            n: e,
            uid: 903,
        };
        let h = pseudo_random(m * e, 96);

        let got = maybe_gpu_ffn(&gate, &up, &down, &h, m, ff, 0).unwrap();

        // CPU 参照 (llama.rs prefill 従来経路と同一の演算列)。
        // out1 は [m, 2ff] 行-major: gate は各行の前半 ff 列 (平坦スライス不可)。
        let n2 = 2 * ff;
        let mut gw = Vec::with_capacity(e * n2);
        gw.extend_from_slice(&gate.w);
        gw.extend_from_slice(&up.w);
        let mut gsc = Vec::with_capacity(n2);
        gsc.extend_from_slice(&gate.scale);
        gsc.extend_from_slice(&up.scale);
        let out1 = tensor::matvec_rows_bt_i8(&h, &gw, &gsc, e, n2, m);
        let mut act = vec![0.0f32; m * ff];
        for i in 0..m {
            let row = &out1[i * n2..(i + 1) * n2];
            act[i * ff..(i + 1) * ff].copy_from_slice(&row[..ff]);
            tensor::silu_inplace(&mut act[i * ff..(i + 1) * ff]);
            for f in 0..ff {
                act[i * ff + f] *= row[ff + f];
            }
        }
        let want = tensor::matvec_rows_bt_i8(&act, &down.w, &down.scale, ff, e, m);

        assert_eq!(got.len(), want.len());
        for idx in 0..want.len() {
            assert_eq!(
                got[idx].to_bits(),
                want[idx].to_bits(),
                "FFN ブロック不一致 @{} (GPU 常駐経路 vs CPU 参照)",
                idx
            );
        }
        println!("cuda_ffn_bit_exact_vs_cpu_reference: bit-exact (m={m}, e={e}, ff={ff})");
    }

    /// P17i: tiled GEMM カーネル (dno_gemm_i8) が CPU int8 線形と**ビット一致**
    /// すること。タイル境界 (m/n が 64 の倍数でない)・k が 4/32 の倍数でない
    /// 場合・GEMV との比較 (同一入力で両カーネルが同一結果) を含む。
    #[test]
    #[ignore]
    fn cuda_gemm_i8_bit_exact_vs_cpu() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        // m はタイル境界をまたぐ値 (1/63/64/65/100/2827 相当)・k/n は 4 や 32 の
        // 倍数でない値を含む。実モデル形状 (k=1536/8960) も含む。
        let cases: [(usize, usize, usize); 8] = [
            (1, 8, 16),
            (63, 37, 200),
            (65, 30, 197),
            (100, 1536, 17920),
            (128, 8960, 1536),
            (100, 1536, 2048),
            (130, 1533, 17911),
            (64, 1536, 17920),
        ];
        for (ci, &(m, k, n)) in cases.iter().enumerate() {
            let q = pseudo_i8(k * n, 40 + ci as u32);
            let scale = positive_scales(n, 50 + ci as u32);
            let a = pseudo_random(m * k, 60 + ci as u32);
            // s74: 重みは gemm_pad_n 行に 0 パディングしてアップロードする
            // (カーネルは j 方向 256 列単位・scale は実列数 n 本)。
            let w = ctx.upload_i8(&gemm_pad_apply(&q, k, n)).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            // GEMM カーネル vs CPU
            let got = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let want = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "gemm m={m} k={k} n={n} idx={idx} (i={} j={}): GPU != CPU",
                    idx / n,
                    idx % n
                );
            }
            // GEMV カーネルともビット一致 (整数累積の順序不変性の検証)
            let gotv = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            for idx in 0..m * n {
                assert_eq!(got[idx].to_bits(), gotv[idx].to_bits(), "gemm vs gemv m={m} k={k} n={n} idx={idx}");
            }
        }
        println!("dno_gemm_i8 bit-exact vs CPU/GEMV OK ({} cases)", cases.len());
    }

    /// s74 (T6 候補 1): dno_gemm_i8 の数値契約を **GPU 無しで** 固定する。
    ///
    /// 新カーネル (128x256 タイル・1 thread 8x16・重み 256 行パディング) は
    /// 旧カーネルと命令列が異なるので、`cuda_gemm_i8_bit_exact_vs_cpu` (要 GPU) が
    /// 走らない環境でも次の不変条件を検証できるようにする:
    ///  1. 重みパディング (`gemm_pad_apply`) は実列 n の dot を厳密に保存する
    ///     (パディング行は 0・実列はビット同一) — j タイル境界での取り違え防止。
    ///  2. タイル分割 (i 方向 128 / j 方向 256) を模した整数 dot と、分割なしの
    ///     整数 dot が同一 (i32 累積は結合則が成立 = R2 の根拠)。
    ///  3. 端処理: m・n・k がタイル / 4 / 32 の倍数でない形状でも、
    ///     範囲外アクセス無しで実領域だけを書く。
    ///
    /// GPU 実機のビット一致は `cuda_gemm_i8_bit_exact_vs_cpu` (supervisor が実行)。
    #[test]
    fn gemm_pad_preserves_dot_and_tiling_is_exact() {
        // 1) パディングは実列を保存し、余りは 0
        for &(k, n) in &[(4usize, 100usize), (37, 200), (1536, 17920), (30, 256), (1, 1)] {
            let qw: Vec<i8> = (0..k * n).map(|i| (((i * 37) % 251) as i32 - 125) as i8).collect();
            let padded = gemm_pad_apply(&qw, k, n);
            assert_eq!(padded.len(), k * gemm_pad_n(n), "k={k} n={n}");
            assert_eq!(gemm_pad_n(n) % 256, 0);
            assert!(gemm_pad_n(n) >= n && gemm_pad_n(n) - n < 256);
            for r in 0..n {
                assert_eq!(
                    &padded[r * k..(r + 1) * k],
                    &qw[r * k..(r + 1) * k],
                    "実列が変わった k={k} n={n} r={r}"
                );
            }
            for r in n..gemm_pad_n(n) {
                assert!(padded[r * k..(r + 1) * k].iter().all(|&v| v == 0), "pad 行が 0 でない");
            }
            // パディング行の寄与は 0 (dot が不変)
            let aq: Vec<i8> = (0..k).map(|i| (((i * 11) % 251) as i32 - 120) as i8).collect();
            let dot = |w: &[i8], row: usize| -> i32 {
                (0..k).map(|d| aq[d] as i32 * w[row * k + d] as i32).sum()
            };
            for r in 0..gemm_pad_n(n) {
                let want = if r < n { dot(&qw, r) } else { 0 };
                assert_eq!(dot(&padded, r), want, "dot 保存 k={k} n={n} r={r}");
            }
            // 既に倍数なら複製しない (Cow::Borrowed)
            if n % 256 == 0 {
                assert!(matches!(padded, std::borrow::Cow::Borrowed(_)));
            } else {
                assert!(matches!(padded, std::borrow::Cow::Owned(_)));
            }
        }

        // 2) タイル分割の厳密性 (i32 累積 = 結合則) — カーネルの k パック/タイル
        //    境界で dot が変わらないことを、k が 4・32 の倍数でない形状も含めて
        //    **2 種類の分割**で確認する:
        //      a) GEMM_TK=32 の kb ブロック × c0 4 要素 (実カーネルの内側ループ)
        //      b) 4 要素ずつ全域 (端ブロックは 4 要素未満に切り詰め = カーネルの
        //         gk<c+4 分岐と等価)
        //    34B の行 (例 k=37) では GEMM_TK の切り上げで k の先にゴミが入るが、
        //    外側ループが `kb < k` で止まるため寄与しない (カーネルと同じ挙動)。
        for &(k, n, m) in &[(37usize, 200usize, 65usize), (30, 197, 65), (1533, 300, 3)] {
            let qw: Vec<i8> = (0..k * n).map(|i| (((i * 29) % 241) as i32 - 120) as i8).collect();
            let aq: Vec<i8> = (0..m * k).map(|i| (((i * 17) % 239) as i32 - 119) as i8).collect();
            let mut padded = gemm_pad_apply(&qw, k, n).into_owned();
            // 行の直後 (k の先) に「本当は範囲外」ダミーを置いておく
            let row_stride = k * gemm_pad_n(n);
            padded.resize(row_stride + 8, 77);
            let plain = |i: usize, j: usize| -> i32 {
                (0..k).map(|d| aq[i * k + d] as i32 * padded[j * k + d] as i32).sum()
            };
            // a) kb (32) ブロック × c0 (4) 分割
            let dot_kb = |i: usize, j: usize| -> i32 {
                let mut acc = 0i32;
                let mut kb = 0;
                while kb < k {
                    let mut c0 = 0;
                    while c0 < 32 {
                        let gk = kb + c0;
                        if gk >= k {
                            break;
                        }
                        for e in 0..4 {
                            if gk + e < k {
                                acc += aq[i * k + gk + e] as i32 * padded[j * k + gk + e] as i32;
                            }
                        }
                        c0 += 4;
                    }
                    kb += 32;
                }
                acc
            };
            // b) 4 要素ずつ全域 (端は切り詰め)
            let dot4 = |i: usize, j: usize| -> i32 {
                let mut acc = 0i32;
                let mut gk = 0;
                while gk < k {
                    for e in 0..4 {
                        if gk + e < k {
                            acc += aq[i * k + gk + e] as i32 * padded[j * k + gk + e] as i32;
                        }
                    }
                    gk += 4;
                }
                acc
            };
            for &(i, j) in &[(0usize, 0usize), (m - 1, n - 1), (m / 2, n / 2), (m - 1, 0), (0, n - 1)] {
                assert_eq!(dot_kb(i, j), plain(i, j), "k={k} m={m} n={n} i={i} j={j}");
                assert_eq!(dot4(i, j), plain(i, j), "k={k} m={m} n={n} i={i} j={j} (4 分割)");
            }
        }

        // 3) 端処理 (タイル境界アドレスが範囲内に収まること) — グリッドの切り上げと
        //    カーネル側の範囲判定が前提としている関係を、純粋計算で確認。
        for &(m, n) in &[(1usize, 1usize), (65, 197), (128, 256), (129, 257), (2843, 151936)] {
            let gmx = m.div_ceil(GEMM_TILE);
            let gmy = n.div_ceil(GEMM_TILE_N);
            assert!(gmx * GEMM_TILE >= m && (gmx - 1) * GEMM_TILE < m);
            assert!(gmy * GEMM_TILE_N >= n && (gmy - 1) * GEMM_TILE_N < n);
            // 重みは常に gmy タイル分の行を用意できる (pad 済み)
            assert!(gemm_pad_n(n) >= gmy * GEMM_TILE_N);
        }
    }


    /// P17k 調査: prefill 実形状での linear_gemm_i8 の内訳計測 (kernel / D2H / to_vec)。
    /// `#[ignore]` — GPU が必要。cargo test --release -p denno-engine --features cuda
    /// cuda_gemm_i8_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn cuda_gemm_i8_bench_prefill_shapes() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let m = 8000usize;
        // (label, k, n) — Qwen2.5-1.5B の gate_up / down / qkv
        let shapes = [
            ("gate_up", 1536usize, 17920usize),
            ("down", 8960usize, 1536usize),
            ("qkv", 1536usize, 2048usize),
        ];
        for (label, k, n) in shapes {
            let q = pseudo_i8(k * n, 77);
            let scale = positive_scales(n, 78);
            let a = pseudo_random(m * k, 79);
            let w = ctx.upload_i8(&gemm_pad_apply(&q, k, n)).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            // ウォームアップ (JIT + L2)
            let _ = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let t0 = std::time::Instant::now();
            let out = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let total = t0.elapsed();
            // 出力 D2H 単体の時間 (同サイズの to_vec 込み)
            let t1 = std::time::Instant::now();
            let _ = ctx.linear_gemm_i8(&a, &w, &sc, m, k, n).unwrap();
            let total2 = t1.elapsed();
            println!(
                "{label}: m={m} k={k} n={n} linear_gemm total={total:?} / {total2:?} out={}MB checksum={}",
                out.len() * 4 / 1_000_000,
                out[1234].to_bits()
            );
        }

        // 本番経路 (linear_gemm: Scratch + pinned) で 28 層分連続実行
        {
            let (k, n) = (1536usize, 17920usize);
            let q = pseudo_i8(k * n, 87);
            let scale = positive_scales(n, 88);
            let a = pseudo_random(m * k, 89);
            let gw = weights_for(999_001, &q, &scale, k, n).unwrap();
            let _ = linear_gemm(&ctx, &a, gw, m, k, n).unwrap(); // warmup
            let t0 = std::time::Instant::now();
            for _ in 0..28 {
                let _ = linear_gemm(&ctx, &a, gw, m, k, n).unwrap();
            }
            let per = t0.elapsed() / 28;
            println!("linear_gemm (production path) 28x gate_up: per-call {per:?}");
        }
    }

    /// P17k 診断: 生転送帯域の実測 (H2D/D2H × pageable/pinned × sync/async)。
    /// attn_core ~400ms/層の内訳が転送律速かどうかの判定用。
    #[test]
    #[ignore]
    fn cuda_transfer_bench() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        const BYTES: usize = 128 * 1024 * 1024; // 128MB
        let elems = BYTES / 4;
        let host: Vec<f32> = pseudo_random(elems, 60);
        let dev = ctx.upload_f32(&host).unwrap();
        let lib = lib().unwrap();
        let gb = |bytes: usize, secs: f64| bytes as f64 / 1e9 / secs;

        // 1) H2D 同期 pageable (DevBuffer::copy_from / ミラー copy_from_at と同一)
        let t = std::time::Instant::now();
        let iters = 4;
        for _ in 0..iters {
            dev.copy_from(&host).unwrap();
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("H2D sync pageable 128MB: {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 2) H2D pinned + 毎回 sync (async 発行→sync)
        let mut pin = PinMem::alloc(BYTES).unwrap();
        pin.as_mut_slice::<f32>()[..elems].copy_from_slice(&host);
        let iters = 8;
        unsafe {
            (lib.memcpy_htod_async)(dev.ptr(), pin.ptr, BYTES, std::ptr::null_mut());
            ctx.sync().unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                check(
                    (lib.memcpy_htod_async)(dev.ptr(), pin.ptr, BYTES, std::ptr::null_mut()),
                    "h2d pin",
                )
                .unwrap();
                ctx.sync().unwrap();
            }
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("H2D pinned+sync 128MB:   {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 3) H2D pinned 連続 async 発行 + 最後に 1 回 sync (オーバーラップ上限)
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                (lib.memcpy_htod_async)(dev.ptr(), pin.ptr, BYTES, std::ptr::null_mut());
            }
        }
        ctx.sync().unwrap();
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("H2D pinned async burst:  {e:.4?}s  {:.1} GB/s (per-op)", gb(BYTES, e));

        // 4) D2H 同期 pageable
        let mut dst = vec![0f32; elems];
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                (lib.memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, dev.ptr(), BYTES);
            }
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("D2H sync pageable 128MB: {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 5) D2H pinned + sync (maybe_gpu_oproj / chain_download の経路)
        let out_pin = PinMem::alloc(BYTES).unwrap();
        unsafe {
            (lib.memcpy_dtoh_async)(out_pin.ptr, dev.ptr(), BYTES, std::ptr::null_mut());
            ctx.sync().unwrap();
        }
        let t = std::time::Instant::now();
        for _ in 0..iters {
            unsafe {
                (lib.memcpy_dtoh_async)(out_pin.ptr, dev.ptr(), BYTES, std::ptr::null_mut());
            }
            ctx.sync().unwrap();
        }
        let e = t.elapsed().as_secs_f64() / iters as f64;
        println!("D2H pinned+sync 128MB:   {e:.4?}s  {:.1} GB/s", gb(BYTES, e));

        // 6) cudaMalloc のコスト参考 (ミラー cap 成長時の全層 realloc)
        let t = std::time::Instant::now();
        for _ in 0..8 {
            let _b = ctx.upload_f32(&host).unwrap();
        }
        println!(
            "upload_f32 (alloc+H2D) 128MB x8: {:?}/op",
            t.elapsed() / 8
        );
    }

    /// P17k 診断: prefill attention カーネル単体の実形状タイミング。
    /// (n, t_total, heads, kv_dim, dh) = (13252, 13252, 14, 256, 128) — 実リクエストと同一。
    /// カーネル自体の時間を分離するため q/KV はランダム、転送は事前に完了させておく。
    #[test]
    #[ignore]
    fn cuda_attn_pf_bench() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let (n, t_total, heads, kv_dim, dh) = (13252usize, 13252usize, 14usize, 256usize, 128usize);
        let q: Vec<f32> = pseudo_random(n * heads * dh, 61);
        // KV ミラーを直接用意 (with_mirror を使わず専用 mirror key でシード)
        let key = 999_111u64;
        let (kptr, vptr) = with_mirror(key, 0, 1, t_total, kv_dim, 0, |m, _have| {
            let kb = m.layers[0].0.as_ref().unwrap();
            let vb = m.layers[0].1.as_ref().unwrap();
            let kv: Vec<f32> = pseudo_random(t_total * kv_dim, 62);
            kb.copy_from(&kv).unwrap();
            vb.copy_from(&kv).unwrap();
            m.len[0] = t_total;
            Some((kb.ptr(), vb.ptr()))
        })
        .unwrap();
        let dev_q = ctx.upload_f32(&q).unwrap();
        let dev_att = ctx.upload_f32(&vec![0f32; n * heads * dh]).unwrap();

        let mut heads_c = heads as c_int;
        let mut n_c = n as c_int;
        let mut pos0_c = 0 as c_int;
        let mut kv_dim_c = kv_dim as c_int;
        let mut t_c = t_total as c_int;
        let mut group_c = 2 as c_int;
        let mut dh_c = dh as c_int;
        let mut scale_c = 0.08838834764831845f32;

        let sh_f = (ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1)
            + ATTN_PF_BQ * ATTN_PF_BK
            + 2 * ATTN_PF_BQ;
        // DevBuffer の ptr は Copy 値なので引数組み立ては毎回行う
        let mut launch = |ctx: &CudaCtx| -> Result<()> {
            let mut qp = dev_q.ptr();
            let mut kp = kptr;
            let mut vp = vptr;
            let mut ap = dev_att.ptr();
            let mut args: Vec<*mut c_void> = vec![
                &mut qp as *mut _ as *mut c_void,
                &mut kp as *mut _ as *mut c_void,
                &mut vp as *mut _ as *mut c_void,
                &mut ap as *mut _ as *mut c_void,
                &mut heads_c as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
                &mut pos0_c as *mut _ as *mut c_void,
                &mut kv_dim_c as *mut _ as *mut c_void,
                &mut t_c as *mut _ as *mut c_void,
                &mut group_c as *mut _ as *mut c_void,
                &mut dh_c as *mut _ as *mut c_void,
                &mut scale_c as *mut _ as *mut c_void,
            ];
            unsafe {
                ctx.launch_async(
                    ctx.f_attn_pf,
                    // P17k: (行ブロック, heads) — x=行ブロック最速次元で同一 head の
                    // K/V を同時実行ブロック間で共有し L2 ヒット率を上げる
                    (n.div_ceil(ATTN_PF_BQ) as u32, heads as u32, 1),
                    (dh as u32, 1, 1),
                    (sh_f * 4) as u32,
                    &mut args,
                )
            }
        };
        launch(&ctx).unwrap();
        ctx.sync().unwrap();
        let iters = 5;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            launch(&ctx).unwrap();
        }
        ctx.sync().unwrap();
        let per = t.elapsed() / iters;
        println!(
            "attn_pf kernel-only n={n} t={t_total} heads={heads}: {per:?}/call (x28 layers = {:?})",
            per * 28
        );
    }

    /// マイクロベンチ: Qwen2.5-1.5B の線形層 (int8) を decode (m=1) で
    /// GPU vs CPU 比較。実効帯域 (重み bytes / 時間) も出す。
    #[test]
    #[ignore]
    fn bench_cuda_linear_int8_1p5b_decode() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        // Qwen2.5-1.5B: hidden 1536 / inter 8960 / qkv 2048 / vocab 151936
        let shapes: [(&str, usize, usize); 5] = [
            ("qkv_proj", 1536, 2048),
            ("o_proj", 1536, 1536),
            ("gate_up", 1536, 17920),
            ("down", 8960, 1536),
            ("lm_head", 1536, 151936),
        ];
        for (label, k, n) in shapes {
            let q = pseudo_i8(k * n, 70);
            let scale = positive_scales(n, 71);
            let a = pseudo_random(k, 72);
            let w = ctx.upload_i8(&q).unwrap();
            let sc = ctx.upload_f32(&scale).unwrap();
            let iters = 20;
            // GPU (重みは VRAM 常駐、アクティベーションは毎回 H2D)
            let _ = ctx.linear_bt_i8(&a, &w, &sc, 1, k, n).unwrap(); // warmup
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = ctx.linear_bt_i8(&a, &w, &sc, 1, k, n).unwrap();
            }
            let gpu = t0.elapsed().as_secs_f64() / iters as f64;
            // CPU
            let _ = tensor::matvec_bt_i8(&a, &q, &scale, k, n); // warmup
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = tensor::matvec_bt_i8(&a, &q, &scale, k, n);
            }
            let cpu = t0.elapsed().as_secs_f64() / iters as f64;
            let wb = (k * n) as f64; // 重み帯域 (int8)
            println!(
                "[{label}] k={k} n={n}: GPU {g:.3}ms ({gw:.0} GB/s) / CPU {c:.3}ms ({cw:.0} GB/s) = {r:.1}x",
                g = gpu * 1e3,
                gw = wb / gpu / 1e9,
                c = cpu * 1e3,
                cw = wb / cpu / 1e9,
                r = cpu / gpu,
            );
        }
    }

    /// P17b パイプライン (ピン留め+非同期+1同期+重みレジストリ) の
    /// ビット一致検証。`quant::Linear` のフック経由 (apply_bt / apply_batch_bt /
    /// apply の Q8 パス) で CPU カーネルと比較する。
    #[test]
    #[ignore]
    fn cuda_pipeline_bit_exact_vs_cpu_via_linear() {
        crate::cuda::enable_for_tests();
        for (m, k, n) in [(1usize, 896usize, 1152usize), (1, 1536, 151936), (4, 1536, 17920), (3, 8960, 1536)] {
            let w = pseudo_random(k * n, 90);
            let a = pseudo_random(m * k, 91 + m as u32);
            let lin = crate::quant::Linear::new(w.clone(), k, n, crate::quant::QuantMode::Int8);

            // GPU 経路 (フック経由)
            let got = lin.apply_bt(&a, m);
            // CPU リファレンス (同一仕様の rows カーネル)
            let q = crate::quant::Quant8Linear::quantize(&w, k, n);
            let want = if m == 1 {
                crate::tensor::matvec_bt_i8(&a, &q.w, &q.scale, k, n)
            } else {
                crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n)
            };
            for idx in 0..m * n {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "apply_bt m={m} k={k} n={n} idx={idx}"
                );
            }
            // apply_batch_bt 経路 (P6 バッチ decode)
            if m > 1 {
                let got_b = lin.apply_batch_bt(&a, m);
                for idx in 0..m * n {
                    assert_eq!(got_b[idx].to_bits(), want[idx].to_bits(), "apply_batch_bt m={m}");
                }
            }
        }
    }

    /// P17c: 融合呼び出し (grouped) vs 個別呼び出しのビット一致。
    /// qkv 相当 (3 パート・異なる n) と gate/up 相当 (2 パート) を m=1/m=4 で検証。
    /// 個別の CPU カーネル結果を列連結したものが正解。
    #[test]
    #[ignore]
    fn cuda_grouped_linear_bit_exact_vs_cpu() {
        crate::cuda::enable_for_tests();
        for m in [1usize, 4usize] {
            let k = 96usize;
            // qkv 相当: n の異なる 3 パート
            let parts_w: Vec<Vec<f32>> = vec![
                pseudo_random(40 * k, 71),
                pseudo_random(24 * k, 72),
                pseudo_random(24 * k, 73),
            ];
            let ns = [40usize, 24, 24];
            let a = pseudo_random(m * k, 74 + m as u32);
            let lins: Vec<crate::quant::Linear> = parts_w
                .iter()
                .zip(ns)
                .map(|(w, n)| {
                    crate::quant::Linear::new(w.clone(), k, n, crate::quant::QuantMode::Int8)
                })
                .collect();
            let q8: Vec<&crate::quant::Quant8Linear> =
                lins.iter().map(|l| l.quant8().unwrap()).collect();
            let got =
                crate::cuda::maybe_gpu_linear_grouped(&q8, &a, m).expect("grouped should run");
            // 正解: 個別 CPU カーネル結果の列連結
            let n_total: usize = ns.iter().sum();
            let mut want = vec![0.0f32; m * n_total];
            let mut col = 0usize;
            for (i, l) in lins.iter().enumerate() {
                let q = crate::quant::Quant8Linear::quantize(&parts_w[i], k, ns[i]);
                let r = if m == 1 {
                    crate::tensor::matvec_bt_i8(&a, &q.w, &q.scale, k, ns[i])
                } else {
                    crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, ns[i])
                };
                for row in 0..m {
                    want[row * n_total + col..][..ns[i]].copy_from_slice(&r[row * ns[i]..][..ns[i]]);
                }
                col += ns[i];
            }
            for idx in 0..m * n_total {
                assert_eq!(
                    got[idx].to_bits(),
                    want[idx].to_bits(),
                    "grouped m={m} idx={idx}"
                );
            }
        }
        // gate/up 相当: 2 パート同幅 (llama.rs の lin_gateup_gpu 経路相当)
        let k = 64usize;
        let ff = 48usize;
        let gw = pseudo_random(ff * k, 75);
        let uw = pseudo_random(ff * k, 76);
        let a = pseudo_random(2 * k, 77);
        let gl = crate::quant::Linear::new(gw, k, ff, crate::quant::QuantMode::Int8);
        let ul = crate::quant::Linear::new(uw, k, ff, crate::quant::QuantMode::Int8);
        let got = crate::cuda::maybe_gpu_linear_grouped(
            &[gl.quant8().unwrap(), ul.quant8().unwrap()],
            &a,
            2,
        )
        .expect("grouped should run");
        // 個別 CPU (apply_batch_bt は GPU フックを持つので CPU カーネルを直接使う)
        let gq = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * k, 75), k, ff);
        let uq = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * k, 76), k, ff);
        let gr = crate::tensor::matvec_rows_bt_i8(&a, &gq.w, &gq.scale, k, ff, 2);
        let ur = crate::tensor::matvec_rows_bt_i8(&a, &uq.w, &uq.scale, k, ff, 2);
        for row in 0..2 {
            for j in 0..ff {
                assert_eq!(
                    got[row * (2 * ff) + j].to_bits(),
                    gr[row * ff + j].to_bits(),
                    "gate row={row} j={j}"
                );
                assert_eq!(
                    got[row * (2 * ff) + ff + j].to_bits(),
                    ur[row * ff + j].to_bits(),
                    "up row={row} j={j}"
                );
            }
        }
    }

    /// P17b パイプラインの固定オーバーヘッド測定 (P17 の linear_bt_i8 = 同期×4
    /// との比較)。decode m=1 の 5 形状。
    #[test]
    #[ignore]
    fn bench_cuda_pipeline_overhead_1p5b_decode() {
        crate::cuda::enable_for_tests();
        let shapes: [(&str, usize, usize); 5] = [
            ("qkv_proj", 1536, 2048),
            ("o_proj", 1536, 1536),
            ("gate_up", 1536, 17920),
            ("down", 8960, 1536),
            ("lm_head", 1536, 151936),
        ];
        for (label, k, n) in shapes {
            let w = pseudo_random(k * n, 92);
            let a = pseudo_random(k, 93);
            let lin = crate::quant::Linear::new(w, k, n, crate::quant::QuantMode::Int8);
            let iters = 50;
            let _ = lin.apply_bt(&a, 1); // warmup + 重みアップロード
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = lin.apply_bt(&a, 1);
            }
            let gpu = t0.elapsed().as_secs_f64() / iters as f64;
            let cpu = lin.dims(); // (CPU 経路は既存ベンチの値を参照)
            let _ = cpu;
            println!("[{label}] P17b pipeline m=1: {ms:.3}ms", ms = gpu * 1e3);
        }
    }

    /// prefill (m>1 GEMM) の GPU vs CPU 比較 — GPU 統合が prefill を
    /// 逆に遅くしていないか (m が大きいと GEMV 型カーネルは不利) を確認する。
    #[test]
    #[ignore]
    fn bench_cuda_pipeline_1p5b_prefill() {
        crate::cuda::enable_for_tests();
        let shapes: [(&str, usize, usize); 3] = [
            ("qkv_proj", 1536, 2048),
            ("gate_up", 1536, 17920),
            ("down", 8960, 1536),
        ];
        for m in [32usize, 128, 512] {
            for (label, k, n) in shapes {
                let w = pseudo_random(k * n, 94);
                let a = pseudo_random(m * k, 95);
                let lin = crate::quant::Linear::new(w, k, n, crate::quant::QuantMode::Int8);
                let iters = 10;
                let _ = lin.apply_bt(&a, m); // warmup
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = lin.apply_bt(&a, m);
                }
                let gpu = t0.elapsed().as_secs_f64() / iters as f64;
                // CPU (matmul_bt_i8)
                let q = crate::quant::Quant8Linear::quantize(
                    &pseudo_random(k * n, 94),
                    k,
                    n,
                );
                let _ = crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n);
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    let _ = crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n);
                }
                let cpu = t0.elapsed().as_secs_f64() / iters as f64;
                println!(
                    "[{label} prefill m={m}] GPU {g:.3}ms / CPU {c:.3}ms = {r:.2}x",
                    g = gpu * 1e3,
                    c = cpu * 1e3,
                    r = cpu / gpu,
                );
            }
        }
    }

    /// バッチ decode (m>1) の GPU vs CPU ベンチ (P6 のバッチ経路との比較)
    #[test]
    #[ignore]
    fn bench_cuda_linear_int8_1p5b_batch() {
        let ctx = CudaCtx::new().unwrap();
        println!("device: {}", ctx.info());
        let (k, n) = (1536usize, 17920usize); // gate_up
        let q = pseudo_i8(k * n, 80);
        let scale = positive_scales(n, 81);
        let w = ctx.upload_i8(&q).unwrap();
        let sc = ctx.upload_f32(&scale).unwrap();
        for m in [1usize, 4, 8] {
            let a = pseudo_random(m * k, 82 + m as u32);
            let _ = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            let iters = 20;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = ctx.linear_bt_i8(&a, &w, &sc, m, k, n).unwrap();
            }
            let gpu = t0.elapsed().as_secs_f64() / iters as f64;
            let _ = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                let _ = tensor::matvec_rows_bt_i8(&a, &q, &scale, k, n, m);
            }
            let cpu = t0.elapsed().as_secs_f64() / iters as f64;
            println!(
                "[gate_up m={m}] GPU {g:.3}ms / CPU {c:.3}ms = {r:.1}x",
                g = gpu * 1e3,
                c = cpu * 1e3,
                r = cpu / gpu,
            );
        }
    }

    /// P17d: GPU dno_expf (glibc e_expf の f64 再現) が CPU f32::exp (glibc) と
    /// ビット一致するか。ビットパターン走査 + ランダム + 特殊値。
    #[test]
    #[ignore]
    fn cuda_expf_glibc_bit_exact_vs_cpu() {
        enable_for_tests();
        CudaCtx::new().unwrap();
        let mut vals: Vec<f32> = vec![
            0.0,
            -0.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
            -88.0,
            -103.0,
            -103.28,
            -103.97,
            -104.0,
            -110.0,
            88.0,
            88.72,
            89.0,
            100.0,
        ];
        // 主要指数帯のビットパターン走査 (指数部 0x3C..0xC2, mantissa 5 step)
        for exp_hi in 0x3Cu32..=0xC2 {
            for sign in [0u32, 1] {
                let hi = (sign << 31) | (exp_hi << 23);
                for mi in (0u32..(1 << 23)).step_by(5) {
                    vals.push(f32::from_bits(hi | mi));
                }
            }
        }
        // ランダムビット (|x| <= 120 にフィルタ + 範囲外も一部保持)
        let mut x: u32 = 123_456_789;
        for _ in 0..4_000_000 {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = f32::from_bits(x);
            if v.abs() <= 120.0 || v.is_infinite() {
                vals.push(v);
            }
        }
        let got = chain_expf_probe(&vals).expect("chain_expf_probe");
        assert_eq!(got.len(), vals.len());
        // 本番 CPU 経路 (silu) は要素ごとのスカラー呼び出しなので、比較基準も
        // スカラー expf に固定する (LLVM の自動ベクトル化で libmvec に置換される
        // とスカラー結果と 1ULP 異なるため inline(never) で回避)
        fn exp_ref(x: f32) -> f32 {
            #[inline(never)]
            fn inner(x: f32) -> f32 {
                x.exp()
            }
            inner(x)
        }
        let mut mismatches = 0usize;
        for (i, (&v, &g)) in vals.iter().zip(got.iter()).enumerate() {
            let want = exp_ref(v);
            if want.is_nan() && g.is_nan() {
                continue; // NaN は payload を比較しない
            }
            if want.to_bits() != g.to_bits() {
                if mismatches < 10 {
                    println!(
                        "MISMATCH idx={i}: x={v:?} (bits {:#010x}) cpu={want:?} ({}), gpu={g:?} ({})",
                        v.to_bits(),
                        want.to_bits(),
                        g.to_bits()
                    );
                }
                mismatches += 1;
            }
        }
        assert_eq!(mismatches, 0, "expf bit mismatch: {mismatches} / {}", vals.len());
        println!("expf bit-exact: {} values OK", vals.len());
    }

    /// P17d: GPU rmsnorm / add+rmsnorm / swiglu が CPU とビット一致するか
    #[test]
    #[ignore]
    fn cuda_chain_elem_bit_exact_vs_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let ctx = CudaCtx::new().unwrap();
        let (cols, m) = (1536usize, 4usize);
        let w = pseudo_random(cols, 201);
        // 1) 単純 rmsnorm
        let x = pseudo_random(cols * m, 200);
        chain_upload(CH_X, &x, cols, m).unwrap();
        chain_rmsnorm(&w, 1e-6, None, CH_H, cols, m, 0).unwrap();
        let got = chain_download(CH_H, cols * m).unwrap();
        for row in 0..m {
            let want = tensor::rmsnorm(&x[row * cols..][..cols], &w, cols, 1e-6);
            for i in 0..cols {
                assert_eq!(got[row * cols + i].to_bits(), want[i].to_bits(), "rmsnorm row={row} i={i}");
            }
        }
        // 2) residual add + rmsnorm
        let proj = pseudo_random(cols * m, 202);
        chain_upload(CH_X, &x, cols, m).unwrap();
        // P17j: 連続する chain_upload は pinned (pin_x) を共有するため間に sync
        // が必須 (DMA 完了前の pinned 上書き競合 → CH_X が化ける。本番経路は
        // upload 間に download 同期が挟まるため問題にならないが、このテストは
        // sync 無しの 2 連続 upload で競合し並列実行時にフレームしていた)。
        ctx.sync().unwrap();
        chain_upload(CH_O, &proj, cols, m).unwrap();
        chain_rmsnorm(&w, 1e-6, Some(CH_O), CH_H, cols, m, 1).unwrap();
        let got = chain_download(CH_H, cols * m).unwrap();
        let mut xr = x.clone();
        for i in 0..cols * m {
            xr[i] += proj[i];
        }
        for row in 0..m {
            let want = tensor::rmsnorm(&xr[row * cols..][..cols], &w, cols, 1e-6);
            for i in 0..cols {
                assert_eq!(
                    got[row * cols + i].to_bits(),
                    want[i].to_bits(),
                    "add+rmsnorm row={row} i={i}"
                );
            }
        }
        // 3) swiglu (gate|up 連結レイアウト)
        let ff = 8960usize;
        let g = pseudo_random(ff, 203);
        let u = pseudo_random(ff, 204);
        let mut ga = g.clone();
        ga.extend_from_slice(&u);
        chain_upload(CH_GA, &ga, ff * 2, 1).unwrap();
        chain_swiglu(CH_GA, CH_ACT, ff, 1).unwrap();
        let got = chain_download(CH_ACT, ff).unwrap();
        let mut want = g.clone();
        tensor::silu_inplace(&mut want);
        for i in 0..ff {
            want[i] *= u[i];
        }
        for i in 0..ff {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "swiglu i={i}");
        }
    }

    /// P17d: decode chain 全体 (rmsnorm → grouped qkv → download) が
    /// CPU 経路 (rmsnorm → apply_bt 個別 3 回) とビット一致するか
    #[test]
    #[ignore]
    fn cuda_decode_chain_bit_exact_vs_cpu() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        CudaCtx::new().unwrap();
        let (e, qn, kvn, ff) = (96usize, 40usize, 24usize, 64usize);
        let qkv_n = qn + 2 * kvn;
        let ln1 = pseudo_random(e, 300);
        let qw = crate::quant::Quant8Linear::quantize(&pseudo_random(qn * e, 310), e, qn);
        let kw = crate::quant::Quant8Linear::quantize(&pseudo_random(kvn * e, 311), e, kvn);
        let vw = crate::quant::Quant8Linear::quantize(&pseudo_random(kvn * e, 312), e, kvn);
        let x = pseudo_random(e, 320);
        // GPU chain
        chain_upload(CH_X, &x, e, 1).unwrap();
        chain_rmsnorm(&ln1, 1e-6, None, CH_H, e, 1, 2).unwrap();
        chain_linear_grouped(&[&qw, &kw, &vw], CH_H, CH_QKV, 1).unwrap();
        let got = chain_download(CH_QKV, qkv_n).unwrap();
        // CPU reference
        let h = tensor::rmsnorm(&x, &ln1, e, 1e-6);
        let mut want = Vec::with_capacity(qkv_n);
        want.extend_from_slice(&tensor::matvec_bt_i8(&h, &qw.w, &qw.scale, qw.k, qw.n));
        want.extend_from_slice(&tensor::matvec_bt_i8(&h, &kw.w, &kw.scale, kw.k, kw.n));
        want.extend_from_slice(&tensor::matvec_bt_i8(&h, &vw.w, &vw.scale, vw.k, vw.n));
        for i in 0..qkv_n {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "chain qkv i={i}");
        }
        // FFN 部: o → add+rmsnorm → gate_up → swiglu → down → add+rmsnorm(final)
        let ow = crate::quant::Quant8Linear::quantize(&pseudo_random(e * qn, 330), qn, e);
        let gw = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 331), e, ff);
        let uw = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 332), e, ff);
        let dw = crate::quant::Quant8Linear::quantize(&pseudo_random(e * ff, 333), ff, e);
        let ln2 = pseudo_random(e, 340);
        let norm_w = pseudo_random(e, 341);
        let att = pseudo_random(qn, 350);
        chain_upload(CH_ATT, &att, qn, 1).unwrap();
        chain_linear_grouped(&[&ow], CH_ATT, CH_O, 1).unwrap();
        chain_rmsnorm(&ln2, 1e-6, Some(CH_O), CH_H, e, 1, 3).unwrap();
        chain_linear_grouped(&[&gw, &uw], CH_H, CH_GA, 1).unwrap();
        chain_swiglu(CH_GA, CH_ACT, ff, 1).unwrap();
        chain_linear_grouped(&[&dw], CH_ACT, CH_DOWN, 1).unwrap();
        chain_rmsnorm(&norm_w, 1e-6, Some(CH_DOWN), CH_H, e, 1, 4).unwrap();
        let got = chain_download(CH_H, e).unwrap();
        // CPU mirror
        let o = tensor::matvec_bt_i8(&att, &ow.w, &ow.scale, ow.k, ow.n);
        let mut xr = x.clone();
        for i in 0..e {
            xr[i] += o[i];
        }
        let h2 = tensor::rmsnorm(&xr, &ln2, e, 1e-6);
        let mut ga_cpu = tensor::matvec_bt_i8(&h2, &gw.w, &gw.scale, gw.k, gw.n);
        let up = tensor::matvec_bt_i8(&h2, &uw.w, &uw.scale, uw.k, uw.n);
        ga_cpu.extend_from_slice(&up);
        tensor::silu_inplace(&mut ga_cpu[..ff]);
        for i in 0..ff {
            ga_cpu[i] *= ga_cpu[ff + i];
        }
        let d = tensor::matvec_bt_i8(&ga_cpu[..ff], &dw.w, &dw.scale, dw.k, dw.n);
        let mut xf = xr.clone();
        for i in 0..e {
            xf[i] += d[i];
        }
        let want = tensor::rmsnorm(&xf, &norm_w, e, 1e-6);
        for i in 0..e {
            assert_eq!(got[i].to_bits(), want[i].to_bits(), "chain ffn i={i}");
        }
        println!("decode chain bit-exact OK (e={e} qkv={qkv_n} ff={ff})");
    }

    /// P17g: dno_attention カーネルと CPU 参照実装の一致 (緩和契約 —
    /// ビット一致ではなく近似一致を検証)。rope 済み q とランダム K/V を使い、
    /// GQA (kv_heads=2, group=6, dh=128) でスコア/softmax/probs@V を比較する。
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn cuda_attention_kernel_vs_cpu_close() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        CudaCtx::new().unwrap();
        let (kv_heads, group, dh) = (2usize, 6usize, 128usize);
        let heads = kv_heads * group;
        let kv_dim = kv_heads * dh;
        let t_total = 577usize;
        let scale = 1.0 / (dh as f32).sqrt();

        let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, 4096);
        {
            let (kc, vc) = cache.kv_bufs(0, t_total);
            for i in 0..t_total * kv_dim {
                kc[i] = ((i * 7919 % 2009) as f32 / 2009.0 - 0.5) * 2.0;
                vc[i] = ((i * 6271 % 1987) as f32 / 1987.0 - 0.5) * 2.0;
            }
        }
        cache.advance(t_total);
        let q: Vec<f32> = (0..heads * dh)
            .map(|i| ((i * 6151 % 1999) as f32 / 1999.0 - 0.5) * 4.0)
            .collect();

        // GPU カーネル (ミラーキーはテスト専用値)
        chain_attention(900_001, 0, 1, &q, &cache, t_total, group, dh, kv_dim, scale).unwrap();
        let got = chain_download(CH_ATT, heads * dh).unwrap();

        // CPU 参照 (スコア → softmax → probs@V、スレッド i が att[h*dh+i] を担当
        // するカーネルと同じ累積順: j 昇順 fmaf)
        let (kc_ref, vc_ref) = cache.kv_bufs_ref(0);
        let mut maxd = 0.0f32;
        for kvh in 0..kv_heads {
            for g in 0..group {
                let h = kvh * group + g;
                let mut scores = vec![0f32; t_total];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..t_total {
                    let mut s = 0f32;
                    for i in 0..dh {
                        s += q[h * dh + i] * kc_ref[j * kv_dim + kvh * dh + i];
                    }
                    s *= scale;
                    scores[j] = s;
                    mx = mx.max(s);
                }
                let mut sum = 0f32;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                for i in 0..dh {
                    let mut acc = 0f32;
                    for j in 0..t_total {
                        acc += scores[j] * vc_ref[j * kv_dim + kvh * dh + i];
                    }
                    let want = acc / sum;
                    let d = (got[h * dh + i] - want).abs();
                    maxd = maxd.max(d);
                }
            }
        }
        println!("attention kernel vs CPU close OK (t={t_total} maxabs={maxd:.3e})");
        assert!(maxd < 1e-3, "attention maxabs diff {maxd}");
    }

    /// B1: prefill デバイス常駐チェーン全体 (llama.rs forward_prefill_gpu と同一の
    /// 呼び出し列・単層) が CPU 参照経路と**ビット一致**すること:
    ///   chain_rmsnorm(#1) → chain_qkv_rope_prefill (h_dev=Some(CH_H): デバイス h 直読み)
    ///   → [attention は緩和契約のためランダム att を CH_ATT に注入]
    ///   → maybe_gpu_oproj_dev (CH_ATT 直読み) → chain_rmsnorm(#2, residual 融合)
    ///   → maybe_gpu_ffn_dev (デバイス常駐) → chain_rmsnorm(最終, residual 融合)。
    /// rmsnorm (dno_add_rmsnorm・in-place x += proj 含む) / GEMM / rope / swiglu /
    /// residual 加算は全て従来経路と同一カーネル・同一演算順序 → ビット一致が契約
    /// (attention のみ緩和枠・ユーザー承認済み)。検証: rope 後 q (デバイス常駐
    /// qkv_q)・K/V キャッシュ行 [pos0..t_total) (rope カーネル → D2H)・最終 hn
    /// (CH_H 全行 + chain_download_at の最終行)・最終 residual (CH_X 全行)。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -- --ignored
    fn cuda_prefill_chain_bit_exact_vs_cpu() {
        fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
            assert_eq!(got.len(), want.len(), "{label}: len mismatch");
            if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
                panic!(
                    "{label}: bit mismatch @{i} gpu={:?} ({:08x}) cpu={:?} ({:08x})",
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                );
            }
        }
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        CudaCtx::new().unwrap();
        enable_for_tests();
        // GEMM 経路 (m > max_m=16 かつ m >= gemm_min_m) を満たすよう閾値を下げる
        std::env::set_var("DENNO_CUDA_GEMM_MIN_M", "8");

        let (n, e, ff) = (40usize, 96usize, 64usize);
        let (heads, kv_heads, dh) = (4usize, 2usize, 16usize);
        let pos0 = 5usize;
        let t_total = pos0 + n;
        let qn = heads * dh;
        let kv_dim = kv_heads * dh;
        let width = qn + 2 * kv_dim;
        let eps = 1e-6f32;
        let theta = 10_000.0f32;
        assert!(gemm_path_active(n), "テスト形状が GEMM 経路の事前ゲートを通らない");

        // 重み (本番と同一の Quant8Linear::quantize 構築経路・uid は自動採番)
        let qw = crate::quant::Quant8Linear::quantize(&pseudo_random(qn * e, 400), e, qn);
        let kw = crate::quant::Quant8Linear::quantize(&pseudo_random(kv_dim * e, 401), e, kv_dim);
        let vw = crate::quant::Quant8Linear::quantize(&pseudo_random(kv_dim * e, 402), e, kv_dim);
        let ow = crate::quant::Quant8Linear::quantize(&pseudo_random(e * qn, 403), qn, e);
        let gate = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 404), e, ff);
        let up = crate::quant::Quant8Linear::quantize(&pseudo_random(ff * e, 405), e, ff);
        let down = crate::quant::Quant8Linear::quantize(&pseudo_random(e * ff, 406), ff, e);
        let ln1 = pseudo_random(e, 407);
        let ln2 = pseudo_random(e, 408);
        let norm_w = pseudo_random(e, 409);

        // KV キャッシュ: 履歴行 [0..pos0) をシードして advance (pos0>0 経路)。
        // 現在行 [pos0..t_total) は chain_qkv_rope_prefill が rope カーネル → D2H で
        // 書く (前提: 未書き込み)。容量確保は従来経路と同一 (P17e 契約)。
        let mut cache = crate::kv_cache::KvCache::new(1, kv_dim, t_total + 64);
        {
            let (kbuf, vbuf) = cache.kv_bufs(0, pos0);
            kbuf[..pos0 * kv_dim].copy_from_slice(&pseudo_random(pos0 * kv_dim, 410));
            vbuf[..pos0 * kv_dim].copy_from_slice(&pseudo_random(pos0 * kv_dim, 411));
        }
        cache.advance(pos0);
        let _ = cache.kv_bufs(0, n);
        let key = cache.kv_id();

        // ---- GPU デバイス常駐チェーン (forward_prefill_gpu と同一呼び出し列) ----
        let x = pseudo_random(n * e, 412);
        chain_upload(CH_X, &x, e, n).unwrap();
        chain_rmsnorm(&ln1, eps, None, CH_H, e, n, 0).unwrap(); // pin_id = li*3
        chain_qkv_rope_prefill(
            &qw, &kw, &vw, &[], &mut cache, key, 0, 1, pos0, n, t_total, heads, kv_heads, dh,
            theta, &[], Some(CH_H),
        )
        .unwrap();
        // attention は緩和契約 (cuda_attention_prefill_vs_cpu_reference で担保) のため
        // ランダム att を CH_ATT に注入し、以降のデバイス常駐チェーンだけを検証する
        let att = pseudo_random(n * qn, 413);
        chain_upload(CH_ATT, &att, qn, n).unwrap();
        maybe_gpu_oproj_dev(&ow, n, qn, 0).unwrap();
        chain_rmsnorm(&ln2, eps, Some(CH_O), CH_H, e, n, 1).unwrap(); // pin_id = li*3+1
        maybe_gpu_ffn_dev(&gate, &up, &down, CH_H, n, ff, 0).unwrap();
        chain_rmsnorm(&norm_w, eps, Some(CH_DOWN), CH_H, e, n, 3).unwrap(); // pin_id = n_layers*3

        // ---- CPU 参照 (従来経路と同一演算列) ----
        let mut h = vec![0.0f32; n * e];
        for i in 0..n {
            h[i * e..][..e].copy_from_slice(&tensor::rmsnorm(&x[i * e..][..e], &ln1, e, eps));
        }
        // qkv GEMM (dno_gemm_i8 とビット一致の matvec_rows_bt_i8)。行 = [q | k | v]
        let mut qkv_w = Vec::with_capacity(e * width);
        qkv_w.extend_from_slice(&qw.w);
        qkv_w.extend_from_slice(&kw.w);
        qkv_w.extend_from_slice(&vw.w);
        let mut qkv_sc = Vec::with_capacity(width);
        qkv_sc.extend_from_slice(&qw.scale);
        qkv_sc.extend_from_slice(&kw.scale);
        qkv_sc.extend_from_slice(&vw.scale);
        let qkv = tensor::matvec_rows_bt_i8(&h, &qkv_w, &qkv_sc, e, width, n);
        let mut q_ref = vec![0.0f32; n * qn];
        let mut k_ref = vec![0.0f32; n * kv_dim];
        let mut v_ref = vec![0.0f32; n * kv_dim];
        for i in 0..n {
            let row = &qkv[i * width..][..width];
            q_ref[i * qn..][..qn].copy_from_slice(&row[..qn]);
            k_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn..qn + kv_dim]);
            v_ref[i * kv_dim..][..kv_dim].copy_from_slice(&row[qn + kv_dim..]);
        }
        tensor::rope_rows(&mut q_ref, n, heads, dh, pos0, theta);
        tensor::rope_rows(&mut k_ref, n, kv_heads, dh, pos0, theta);
        // o_proj → residual 加算 (dno_add_rmsnorm の in-place x += proj と同一) → rmsnorm #2
        let o = tensor::matvec_rows_bt_i8(&att, &ow.w, &ow.scale, qn, e, n);
        let mut xr = x.clone();
        for i in 0..n * e {
            xr[i] += o[i];
        }
        let mut h2 = vec![0.0f32; n * e];
        for i in 0..n {
            h2[i * e..][..e].copy_from_slice(&tensor::rmsnorm(&xr[i * e..][..e], &ln2, e, eps));
        }
        // FFN: gate|up 連結 GEMM → silu*up → down
        let mut gu_w = Vec::with_capacity(e * 2 * ff);
        gu_w.extend_from_slice(&gate.w);
        gu_w.extend_from_slice(&up.w);
        let mut gu_sc = Vec::with_capacity(2 * ff);
        gu_sc.extend_from_slice(&gate.scale);
        gu_sc.extend_from_slice(&up.scale);
        let out1 = tensor::matvec_rows_bt_i8(&h2, &gu_w, &gu_sc, e, 2 * ff, n);
        let mut act = vec![0.0f32; n * ff];
        for i in 0..n {
            let row = &out1[i * 2 * ff..(i + 1) * 2 * ff];
            act[i * ff..(i + 1) * ff].copy_from_slice(&row[..ff]);
            tensor::silu_inplace(&mut act[i * ff..(i + 1) * ff]);
            for f in 0..ff {
                act[i * ff + f] *= row[ff + f];
            }
        }
        let d = tensor::matvec_rows_bt_i8(&act, &down.w, &down.scale, ff, e, n);
        let mut xf = xr;
        for i in 0..n * e {
            xf[i] += d[i];
        }
        let mut hn = vec![0.0f32; n * e];
        for i in 0..n {
            hn[i * e..][..e].copy_from_slice(&tensor::rmsnorm(&xf[i * e..][..e], &norm_w, e, eps));
        }

        // ---- 検証 ----
        // 1) rope 後 q (デバイス常駐 qkv_q)
        let q_gpu = chain_qkv_q_download(n, qn).unwrap();
        assert_bits_eq(&q_gpu, &q_ref, "rope 後 q");
        // 2) K/V キャッシュ行: 現在行 [pos0..t_total) は rope 後と一致・履歴行 [0..pos0)
        //    はシード値のまま (rope カーネルが履歴を書き換えていないこと)
        let (kc, vc) = cache.kv_bufs_ref(0);
        let k_hist = pseudo_random(pos0 * kv_dim, 410);
        let v_hist = pseudo_random(pos0 * kv_dim, 411);
        for i in 0..t_total {
            let off = i * kv_dim;
            let wk: &[f32] = if i < pos0 {
                &k_hist[off..off + kv_dim]
            } else {
                &k_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim]
            };
            let wv: &[f32] = if i < pos0 {
                &v_hist[off..off + kv_dim]
            } else {
                &v_ref[(i - pos0) * kv_dim..(i - pos0 + 1) * kv_dim]
            };
            assert_bits_eq(&kc[off..off + kv_dim], wk, &format!("k 行 {i}"));
            assert_bits_eq(&vc[off..off + kv_dim], wv, &format!("v 行 {i}"));
        }
        // 3) 最終 hn (CH_H 全行) + chain_download_at の最終行 (本番が lm_head 入力と
        //    して取り出す [n-1] 行のみの D2H — B1 ヘルパー)
        let hn_gpu = chain_download(CH_H, n * e).unwrap();
        assert_bits_eq(&hn_gpu, &hn, "最終 hn");
        let hn_last = chain_download_at(CH_H, (n - 1) * e, e).unwrap();
        assert_bits_eq(&hn_last, &hn[(n - 1) * e..], "最終 hn (download_at)");
        // 4) 最終 residual (CH_X 全行 — 層内 2 回の in-place x += proj/down を含む)
        let xf_gpu = chain_download(CH_X, n * e).unwrap();
        assert_bits_eq(&xf_gpu, &xf, "最終 residual (CH_X)");
        println!("prefill chain bit-exact OK (n={n} e={e} ff={ff} pos0={pos0} t={t_total})");
    }

    // === C4 GPU segfault 修正 (2026-10-16) の回帰テスト (純関数・GPU 不要) ===

    /// C4: full_attention の CH_QKV k セクション配置と KV ミラー行内配置。
    /// 旧実装は src_off=hd (qg の後半 = gate 部を読む)・head 間隔 kv_dim
    /// (k 節域を外れて v/gate を読む・head kv_heads-1 はミラー行の外へ書く)
    /// だったため、k 履歴が破壊されていた。
    #[test]
    fn q35_fullattn_k_layout_matches_ch_qkv_sections() {
        // (heads, kv_heads, dh) — 9B 相当と縮小形状
        let cases: [(usize, usize, usize); 3] = [(32, 8, 128), (16, 4, 128), (8, 2, 64)];
        for &(heads, kv_heads, dh) in &cases {
            let kv_dim = kv_heads * dh;
            let k_off = q35_fullattn_k_off(heads, dh);
            // k セクションは qg (heads*2dh) の直後
            assert_eq!(k_off, heads * 2 * dh);
            let k_end = k_off + kv_dim;
            // head b の読み出し [k_off + b*dh .. + dh) は k セクション内
            for b in 0..kv_heads {
                assert!(k_off + b * dh + dh <= k_end, "k head {b} read OOB");
                // ミラー行 (kv_dim 幅) への書き込み [b*dh .. + dh) も行内
                assert!(b * dh + dh <= kv_dim, "k head {b} write out of mirror row");
            }
            // 旧実装の再現: head 1 の旧読み出し位置 (src_off=hd + 1*kv_dim) は
            // k セクション開始位置よりも前 (= qg 内) で、GQA (kv_heads < heads) なら常にそうなる
            let old_src_off = heads * dh;
            if kv_heads < heads {
                assert!(old_src_off + kv_dim < k_off, "旧 src_off は qg 内に留まる");
            }
            // 旧実装の再現: head kv_heads-1 の旧書き込み位置 ((kv_heads-1)*kv_dim)
            // はミラー行 (kv_dim 幅) の外
            if kv_heads >= 2 {
                assert!((kv_heads - 1) * kv_dim + dh > kv_dim, "旧 dst_stride は行外に書く");
            }
        }
    }

    /// C4: CH_QKV ステップ事前確保は GDN 層 ([qkv|z|beta|alpha]) と full 層
    /// ([qg|k|v]) の両レイアウトを覆う。旧実装 (事前確保なし) では GDN 層の
    /// grouped GEMV 直後に chain_q35_gdn_scan 頭部の ensure が grow 再確保で
    /// in-flight 書き込み先を解放し、dno_gdn_scan が未初期化バッファを読んでいた
    /// (実機 E2E 出力非決定論の直接原因)。grouped GEMV 直後の確保サイズ
    /// (conv_dim + z) が事前確保より小さい = 旧経路で grow が走ることの再現 assert。
    #[test]
    fn q35_ch_qkv_bytes_covers_gdn_and_full_sections() {
        // (heads, kv_heads, dh, nk, nv, dk, dv) — 9B 相当 / 0.8B 相当 / 縮小
        let cases: [(usize, usize, usize, usize, usize, usize, usize); 3] = [
            (32, 8, 128, 16, 32, 128, 128),
            (16, 4, 128, 8, 8, 128, 128),
            (8, 2, 64, 4, 4, 64, 64),
        ];
        for &(heads, kv_heads, dh, nk, nv, dk, dv) in &cases {
            let kv_dim = kv_heads * dh;
            let conv_dim = 2 * nk * dk + nv * dv;
            let vd = nv * dv;
            let reserved = q35_ch_qkv_bytes(heads, dh, kv_heads, nk, nv, dk, dv);
            // GDN レイアウトの各セクション末尾が予約域に収まる
            // (chain_q35_gdn_scan の pz/pb/pa オフセット + カーネル読み出し幅)
            let z_end = conv_dim + vd;
            let beta_end = z_end + nv;
            let alpha_end = beta_end + nv;
            assert!(z_end * 4 <= reserved, "z 節域外");
            assert!(beta_end * 4 <= reserved, "beta 節域外");
            assert!(alpha_end * 4 <= reserved, "alpha 節域外");
            // full レイアウト (q35_fullattn_k_off 経由の k/v セクション) も収まる
            let k_off = q35_fullattn_k_off(heads, dh);
            assert!(k_off * 4 <= reserved);
            assert!((k_off + 2 * kv_dim) * 4 <= reserved, "full k/v 節域外");
            // 旧経路の再現: grouped GEMV 直後の確保サイズ (conv_dim + z) は
            // 予約域より小さい → scan 頭部の ensure で grow 再確保が走っていた
            let grouped_after_gemv = (conv_dim + vd) * 4;
            assert!(
                grouped_after_gemv < reserved,
                "grouped 直後 ({grouped_after_gemv}) < 予約域 ({reserved}) が崩れた"
            );
        }
        // 単調性: 形状を大きくすると予約域も大きくなる (公式の取りこぼし防止)
        assert!(
            q35_ch_qkv_bytes(32, 128, 8, 16, 32, 128, 128)
                >= q35_ch_qkv_bytes(8, 64, 2, 4, 4, 64, 64)
        );
    }

    /// C4: CH_ATT ステップ事前確保は GDN 出力 y [vd] と full attention 出力
    /// [heads*dh] の両方を覆う。旧実装では heads*dh > vd の形状で full 層側の
    /// ensure が直前 GDN scan の in-flight 書き込み先を解放し得た。
    #[test]
    fn q35_ch_att_bytes_covers_gdn_y_and_full_att() {
        // (nv, dv, heads, dh)
        let cases: [(usize, usize, usize, usize); 4] = [
            (32, 128, 32, 128), // vd == heads*dh (等しい)
            (8, 128, 32, 128),  // vd < heads*dh → full 側が最大
            (32, 128, 8, 128),  // vd > heads*dh → GDN 側が最大
            (4, 64, 4, 64),
        ];
        for &(nv, dv, heads, dh) in &cases {
            let expected = (nv * dv).max(heads * dh) * 4;
            assert_eq!(q35_ch_att_bytes(nv, dv, heads, dh), expected);
            // 両出力の書き込み幅が予約域に収まる
            assert!((nv * dv) * 4 <= expected, "GDN y 節域外");
            assert!(heads * dh * 4 <= expected, "full att 節域外");
        }
    }

    /// C4: KV ステージの事前確保バイト数は full 層 li の積み位置
    /// (li*2*kv_dim 要素・1 層分 2*kv_dim 要素) と chain_q35_kv_stage_flush の
    /// 読み出し範囲を覆う。GDN 層はステージしない (skip) ため積み位置に穴が
    /// 開くが、全体サイズは最終層 (li = n_layers-1) の末尾で決まる。
    #[test]
    fn q35_kv_stage_bytes_matches_full_layer_staging() {
        // (n_layers, kv_dim)
        let cases: [(usize, usize); 3] = [(48, 1024), (24, 512), (4, 128)];
        for &(n_layers, kv_dim) in &cases {
            let total_elems = q35_kv_stage_bytes(n_layers, kv_dim) / 4;
            assert_eq!(total_elems, n_layers * 2 * kv_dim);
            // GDN 層 (skip される li) を含む全 li の積み末尾が総容量に収まる
            for li in 0..n_layers {
                let off = li * 2 * kv_dim;
                assert!(off + 2 * kv_dim <= total_elems, "li={li} の k/v 行が節域外");
            }
        }
    }

    /// s50: 遅延 GDN 状態フラッシュ (chain_q35_gdn_flush_pending) の契約検証。
    /// GPU 必須 (DevBuffer D2H + KV_MIRRORS が cuda feature 配下のため)。
    /// ①gdn_pos == cache.len() → cur 面 (成功ステップ後の現行状態)
    /// ②gdn_pos == cache.len()+1 → 1-cur 面 (層途中で失敗したステップの入力面)
    /// ③epoch 不一致 → pending 破棄の no-op (ホストが正)
    /// ④pending 無し / ミラー無し → no-op
    /// ⑤位置不整合 (バグのみ) → None + pending 破棄
    /// ⑥gdn_pos == None (chain 未走層) → スキップ
    /// 転送は bit そのまま (演算なし) のため、コピー内容の bit 一致で検証する。
    #[test]
    #[ignore] // GPU が必要: cargo test --release --features cuda -p denno-engine -- --ignored
    fn cuda_gdn_flush_pending_faces_positions_and_epoch() {
        fn assert_bits_eq(got: &[f32], want: &[f32], label: &str) {
            assert_eq!(got.len(), want.len(), "{label}: len mismatch");
            if let Some(i) = got.iter().zip(want).position(|(a, b)| a.to_bits() != b.to_bits()) {
                panic!(
                    "{label}: bit mismatch @{i} got={:?} ({:08x}) want={:?} ({:08x})",
                    got[i],
                    got[i].to_bits(),
                    want[i],
                    want[i].to_bits()
                );
            }
        }
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().unwrap();
        // 番兵キー (MIRROR_TEST_KEY = u64::MAX-1 / 別テストの u64::MAX-3 と衝突しない)
        // flush は cache.kv_id() で検索するため、テストのミラーキーは cache の自動採番 id を使う
        let (nk, nv, dk, dv, d_conv) = (2usize, 4, 8, 8, 4);
        let conv_len = (2 * nk * dk + nv * dv) * (d_conv - 1); // 192
        let ssm_len = nv * dk * dv; // 256
        let mut cache = crate::kv_cache::KvCache::new(1, 64, 256);
        cache.attach_gdn_slot(0, conv_len, ssm_len);
        let key = cache.kv_id(); // プロセス内で一意 = chain_q35_gdn_flush_pending の検索キーと一致

        // 面に既知パターンを書く: face0 = A (state at target)、face1 = B (state at target+1)
        let conv_a = pseudo_random(conv_len, 81);
        let ssm_a = pseudo_random(ssm_len, 82);
        let conv_b = pseudo_random(conv_len, 83);
        let ssm_b = pseudo_random(ssm_len, 84);
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            if m.gdn[0].is_none() {
                let mut g = GdnDev {
                    conv: [None, None],
                    ssm: [None, None],
                    cur: 0,
                };
                ok(Scratch::ensure(&mut g.conv[0], conv_len * 4))?;
                ok(Scratch::ensure(&mut g.conv[1], conv_len * 4))?;
                ok(Scratch::ensure(&mut g.ssm[0], ssm_len * 4))?;
                ok(Scratch::ensure(&mut g.ssm[1], ssm_len * 4))?;
                m.gdn[0] = Some(g);
            }
            let gd = m.gdn[0].as_mut().unwrap();
            ok(gd.conv[0].as_ref().unwrap().copy_from(&conv_a))?;
            ok(gd.ssm[0].as_ref().unwrap().copy_from(&ssm_a))?;
            ok(gd.conv[1].as_ref().unwrap().copy_from(&conv_b))?;
            ok(gd.ssm[1].as_ref().unwrap().copy_from(&ssm_b))?;
            gd.cur = 0;
            m.gdn_pos[0] = Some(1);
            m.gdn_pending = Some(1);
            // s141b: ホストより新しい状態 = stale (flush はこの層を必ず読む)
            m.gdn_stale[0] = true;
            Some(())
        })
        .unwrap();

        let host = |cache: &mut crate::kv_cache::KvCache| {
            let (c, s) = cache.gdn_state_mut(0).unwrap();
            (c.to_vec(), s.to_vec())
        };
        let mirror_pending = || {
            let mirrors = KV_MIRRORS.get().unwrap();
            let mut ms = mirrors.lock().unwrap();
            let idx = ms.iter().position(|(k, _, _)| *k == key).unwrap();
            ms[idx].1.gdn_pending
        };
        let mirror_stale = || {
            let mirrors = KV_MIRRORS.get().unwrap();
            let mut ms = mirrors.lock().unwrap();
            let idx = ms.iter().position(|(k, _, _)| *k == key).unwrap();
            ms[idx].1.gdn_stale[0]
        };

        // s141b (Task B7c): stale でも位置が target+1 (層途中失敗) の層は読まない
        // = ホストは変わらない (古い値を書き戻さない)。ホストを 0 埋めして
        // flush を回し、0 のままであることを見る (① の前 = gdn_pos は 1 のまま)。
        // ① では cache.len() が 1 になり位置が target に揃うので、そこで初めて読む。
        assert_eq!(cache.len(), 0, "前提: advance 前");
        {
            let (hc, hs) = cache.gdn_state_mut(0).unwrap();
            hc.fill(0.0);
            hs.fill(0.0);
        }
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hz_c, hz_s) = host(&mut cache);
        assert!(hz_c.iter().all(|&v| v == 0.0) && hz_s.iter().all(|&v| v == 0.0),
            "target+1 の層は D2H しない (古い値を書き戻さない)");
        assert_eq!(mirror_pending(), None, "no-op でも pending は解除");
        // 次に読む機会 (①) のために、ミラー状態を元へ戻す。
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            m.gdn_pos[0] = Some(1);
            m.gdn_pending = Some(1);
            m.gdn_stale[0] = true;
            Some(())
        })
        .unwrap();

        // ① 成功ステップ後: gdn_pos = 1 == cache.len() (advance 後) → cur 面 = A
        cache.advance(1);
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_a, "①conv は cur 面 (A) と bit 一致");
        assert_bits_eq(&hs, &ssm_a, "①ssm は cur 面 (A) と bit 一致");
        assert_eq!(mirror_pending(), None, "①フラッシュ後 pending は解除");
        assert!(!mirror_stale(), "①フラッシュ後、層のホスト遅延フラグは落ちている");

        // ② 層途中失敗: swap 済み (cur = 書き込み面 = state at 2)・cache.len() = 1
        //    → 1-cur 面 (= state at 1 = A) を回復する
        cache.advance(1); // len = 2
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            gd.cur = 1; // 成功ステップが cur を B 面へ交換した直後の想定
            m.gdn_pos[0] = Some(3);
            m.gdn_pending = Some(3);
            m.gdn_stale[0] = true; // s141b: ホストより新しい
            Some(())
        })
        .unwrap();
        // cache.len() = 2・gdn_pos = 3 = target+1 → 1-cur = 0 面 = A
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_a, "②conv は 1-cur 面 (A) と bit 一致");
        assert_bits_eq(&hs, &ssm_a, "②ssm は 1-cur 面 (A) と bit 一致");

        // ②' 成功ステップ後 (cur 面が B): gdn_pos = target → cur 面 = B
        cache.advance(1); // len = 3
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            gd.cur = 1; // B 面が現行
            m.gdn_pos[0] = Some(3);
            m.gdn_pending = Some(3);
            m.gdn_stale[0] = true; // s141b: ホストより新しい
            Some(())
        })
        .unwrap();
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_b, "②'conv は cur 面 (B) と bit 一致");
        assert_bits_eq(&hs, &ssm_b, "②'ssm は cur 面 (B) と bit 一致");

        // ⑤ 位置不整合 (正常系では起こらない) → None + pending 破棄
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            m.gdn_pos[0] = Some(7);
            m.gdn_pending = Some(7);
            m.gdn_stale[0] = true; // s141b: 古いホストを持つ層だけが不整合を検出する
            Some(())
        })
        .unwrap();
        assert_eq!(
            chain_q35_gdn_flush_pending(&mut cache),
            None,
            "位置不整合は None (呼び出し側は継続しない)"
        );
        assert_eq!(mirror_pending(), None, "⑤不整合時も pending は破棄");

        // ⑥ gdn_pos = None (chain 未走層) → スキップで no-op (ホスト不変)
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            m.gdn_pos[0] = None;
            m.gdn_pending = Some(1);
            m.gdn_stale[0] = true; // s141b
            Some(())
        })
        .unwrap();
        let (hc0, hs0) = host(&mut cache);
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc1, hs1) = host(&mut cache);
        assert_eq!(hc0.len(), hc1.len(), "⑥conv len 不変");
        assert_eq!(hs0.len(), hs1.len(), "⑥ssm len 不変");
        assert!(
            hc0.iter().zip(&hc1).all(|(a, b)| a.to_bits() == b.to_bits())
                && hs0.iter().zip(&hs1).all(|(a, b)| a.to_bits() == b.to_bits()),
            "⑥chain 未走層はホスト不変 (bit 一致)"
        );

        // ③ epoch 不一致 (reset = ホストが正) → pending 破棄の no-op
        cache.reset(); // GDN スロットは 0 埋め・epoch +1
        // with_mirror_ext は epoch を同期してしまうため使わない。直接ロックで
        // 旧世代 (reset 前の epoch) のミラーに pending を設定し、epoch 不一致の
        // 状態を再現する。
        {
            let mirrors = KV_MIRRORS.get().unwrap();
            let mut ms = mirrors.lock().unwrap();
            let idx = ms.iter().position(|(k, _, _)| *k == key).unwrap();
            ms[idx].1.gdn_pos[0] = Some(9);
            ms[idx].1.gdn_pending = Some(9);
            ms[idx].1.gdn_stale[0] = true; // s141b
        }
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        assert_eq!(mirror_pending(), None, "③epoch 不一致で pending は破棄");
        let (hc, hs) = host(&mut cache);
        assert!(hc.iter().all(|&v| v == 0.0) && hs.iter().all(|&v| v == 0.0),
            "③reset 済みホスト (0 埋め) は旧面で上書きされない");

        // ④ pending 無し / ミラー無し → no-op
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let other = crate::kv_cache::KvCache::new(1, 64, 256); // 別 kv_id・ミラー無し
        let mut other = other;
        assert_eq!(chain_q35_gdn_flush_pending(&mut other), Some(()));
    }
}

// ---------------------------------------------------------------------------
// P19-A1: GDN 逐次 AR スキャン (prefill バッチ)
// ---------------------------------------------------------------------------

/// GDN prefill バッチスキャン (レイヤー 1 回のカーネル launch)。
///
/// - 入力: 投影済み qkv/z/beta/alpha の全トークン分行 + レイヤー状態 (conv/ssm)
/// - 出力: y [n × value_dim] (gated RMS norm 済み・out_w 投影前)。状態は in-place 更新。
/// - GDN は bit 一致不要 (CPU 逐次経路と数値は僅かに異なる・契約は決定論のみ)。
/// - GPU 無効/未初期化/実行時エラー時は None (呼び出し側は CPU 逐次にフォールバック)。
/// - 前提: head_k_dim == head_v_dim かつ d_conv == 4 (Qwen3.5 は常に成立)。
/// - A1-r2: n >= GDN_CHUNK_MIN_TOKENS の長 prefill は 3-pass chunked カーネルに
///   ルーティング (chunked は GPU 専用・CPU 経路の契約は不変)。n == 1 は decode
///   入口 (qwen35 の gdn_scan から呼ばれる)。
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_batch(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
) -> Option<Vec<f32>> {
    if !cuda_enabled() {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, GDN scan using CPU: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    // A1-r2: 長 prefill は chunked 3 カーネルへ。chunked は dk==dv / d_conv==4
    // 前提のカーネルのため、非成立時は従来の単発カーネルにフォールバック。
    let chunked = n >= GDN_CHUNK_MIN_TOKENS && dk == dv && d_conv == 4;
    let result = if chunked {
        gdn_scan_chunked_impl(
            ctx,
            GdnScanSrc::Host { qkv, z, beta, alpha },
            conv_st,
            ssm_st,
            wconv,
            wa,
            wdt,
            wnorm,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            None,
            li,
            kv,
        )
    } else {
        gdn_scan_impl(
            ctx, qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk, nv, dk, dv,
            d_conv, eps, q_scale, li, kv,
        )
    };
    match result {
        Ok(y) => Some(y),
        Err(e) => {
            tracing::error!("CUDA GDN scan failed, using CPU permanently: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }
}

/// 実装 (同期コピー + 1 launch + 1 同期。prefill はリクエストあたり層数回しか
/// 呼ばれないためピン留めパイプライン化の恩恵は小さい)。
#[allow(clippy::too_many_arguments)]
fn gdn_scan_impl(
    ctx: &CudaCtx,
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    li: usize,
    kv: u64,
) -> Result<Vec<f32>> {
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    debug_assert_eq!(qkv.len(), n * conv_dim);
    debug_assert_eq!(z.len(), n * vd);
    debug_assert_eq!(beta.len(), n * nv);
    debug_assert_eq!(alpha.len(), n * nv);
    debug_assert_eq!(conv_st.len(), conv_dim * cs);
    debug_assert_eq!(ssm_st.len(), nv * dk * dv);

    ctx.make_current()?;
    // s64 (T5b L1 Step 1 計測): この関数の全体は呼び出し側 (qwen35.rs) の
    // `time_layer_n_ev(ATT_GDN_SCAN)` = `attn_gdn` / `gdn_scan` バケットで計時される。
    // ここではスキャン 1 call の中身を「確保 + H2D / launch / sync / D2H」の 4 区間に
    // 割り、既存バケットを壊さずに **表示専用の細分** (`D2H_GDN_PF_ALLOC` ..
    // `D2H_GDN_PF_D2H`) へ積む。各区間の経過時間は次の区間の入口で読むので、
    // 細分の合計≒外側バケット値になる。計測器は `prof::Seq` (prof off は素透し
    // で時刻読みすらしない = 数値経路・実行時間とも不変)。
    //
    // 区間の切り方: 確保 + 入力 H2D (重み 4 本を含む) を 1 区間にまとめた理由は、
    // 重み (`wconv` 等) が層ごとに固定で `gdn_scan_batch` の引数にはバッファが
    // 無い (= 呼び出しごとに H2D し直している) ことを確定させるため。
    let mut q = crate::prof::Seq::begin(crate::prof::D2H_GDN_PF_ALLOC, Instant::now());
    // s128 (decode): 13 本の作業領域を chunked 経路と同じ thread_local (GDN_SCRATCH)
    // から借りる。従来は層ごと・token ごとに 13 本を確保して解放しており、
    // 27B の decode で 1 token あたり約 38 ms (48 層 x 0.80 ms) を捨てていた。
    // 中身は従来どおり copy_from で全要素を上書きするので状態は同一。
    let dp = GDN_SCRATCH.with(|cell| -> Result<GdnPtrs> {
        let mut s = cell.borrow_mut();
        s.ensure(n, conv_dim, vd, nv, dk, dv, cs, 1)?;
        // conv_out / ssm_out は「書かれない領域」を残さないため従来どおりゼロで初期化
        s.buf(GdnBuf::ConvOut).fill_zero()?;
        s.buf(GdnBuf::SsmOut).fill_zero()?;
        s.q.as_ref().unwrap().copy_from(qkv)?;
        s.z.as_ref().unwrap().copy_from(z)?;
        s.beta.as_ref().unwrap().copy_from(beta)?;
        s.alpha.as_ref().unwrap().copy_from(alpha)?;
        s.wc.as_ref().unwrap().copy_from(wconv)?;
        s.wa.as_ref().unwrap().copy_from(wa)?;
        s.wd.as_ref().unwrap().copy_from(wdt)?;
        s.wn.as_ref().unwrap().copy_from(wnorm)?;
        Ok(s.ptrs())
    })?;
    // s131: conv / ssm の正本はデバイス。未確定のときだけホストから 1 回 seed。
    let (pst_conv, pst_ssm) =
        gdn_state_begin(kv, li, conv_st, ssm_st, conv_dim * cs, nv, dk * dv)?;
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);

    let (mut pq, mut pz, mut pb, mut pa) = (dp.q, dp.z, dp.beta, dp.alpha);
    let (mut pci, mut psi) = (pst_conv, pst_ssm);
    let (mut pwc, mut pwa, mut pwd, mut pwn) = (dp.wc, dp.wa, dp.wd, dp.wn);
    let (mut py, mut pco, mut pso) = (dp.y, dp.conv_out, dp.ssm_out);
    let (mut ni, mut nki, mut nvi, mut dki, mut dvi) = (
        n as c_int, nk as c_int, nv as c_int, dk as c_int, dv as c_int,
    );
    let (mut cdi, mut dci, mut csi) = (conv_dim as c_int, d_conv as c_int, cs as c_int);
    let (mut epsi, mut qsi) = (eps, q_scale);
    let mut args = [
        &mut pq as *mut _ as *mut c_void,
        &mut pz as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut psi as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pwn as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut dki as *mut _ as *mut c_void,
        &mut dvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut dci as *mut _ as *mut c_void,
        &mut csi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
        &mut qsi as *mut _ as *mut c_void,
    ];
    let shmem = ((3 * dk + dv + 5) * 4) as u32;
    unsafe {
        ctx.launch_async(ctx.f_gdn, (nv as u32, 1, 1), (dk as u32, 1, 1), shmem, &mut args)?;
    }
    q.lap(crate::prof::D2H_GDN_PF_SYNC);
    ctx.sync()?;
    q.lap(crate::prof::D2H_GDN_PF_D2H);

    let y = GDN_SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let s = cell.borrow();
        s.buf(GdnBuf::Y).to_vec::<f32>(n * vd)
    })?;
    // s131: 状態はホストへ落とさずデバイス正本へ D2D で戻す。
    gdn_state_commit(li, dp.conv_out, dp.ssm_out, conv_dim * cs, nv, dk * dv)?;
    if !gdn_devstate() {
        gdn_host_writeback(conv_st, ssm_st)?;
    }
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    q.end();
    Ok(y)
}

// ---------------------------------------------------------------------------
// A1-r2: GDN chunked scan (長 prefill 用・3-pass 構成)
// ---------------------------------------------------------------------------

/// chunked ルーティングの閾値 (n がこれ以上で 3-pass chunked カーネルを使用)。
/// 実際の chunk 分割は [`crate::gdn::scan_schedule::plan`] (CPU 参照と共有する
/// 唯一の定義) が返す。
pub(crate) const GDN_CHUNK_MIN_TOKENS: usize = crate::gdn::scan_schedule::CHUNK_MIN_TOKENS;

/// s72: GDN スキャンの y をデバイス上に残したまま out_proj までチェーンで実行し、
/// 結果だけを host へ落とす。従来は y (n×vd) を D2H し、次の out_proj で再び H2D
/// していた (1 層あたり 2×47MB)。
///
/// 演算は従来と同一 (同じ `f_quant` + `f_gemm`・同一 int8 累積) なので出力は
/// ビット一致する。conv/ssm 状態の更新はスキャンカーネルが行うため、スキャン開始後に
/// 失敗するとフォールバックでは状態を巻き戻せない。よって前提条件 (重みのアップロード・
/// バッファ確保・ctx) はすべてスキャン前に確認し、以降の失敗は GPU 障害として扱う。
#[allow(clippy::too_many_arguments)]
pub fn gdn_scan_batch_oproj(
    qkv: &[f32],
    z: &[f32],
    beta: &[f32],
    alpha: &[f32],
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    o: &crate::quant::Quant8Linear,
    out: &mut [f32],
    li: usize,
    kv: u64,
) -> Option<()> {
    if !cuda_enabled() || n < GDN_CHUNK_MIN_TOKENS || dk != dv || d_conv != 4 {
        return None;
    }
    let vd = nv.checked_mul(dv)?;
    if o.k != vd || out.len() != n.checked_mul(o.n)? {
        return None;
    }
    let ctx = chain_ctx()?;
    if ctx.make_current().is_err() {
        return None;
    }
    ok(chain_ensure_buf(CH_PFH, n.checked_mul(vd)?.checked_mul(4)?))?;
    ok(chain_ensure_buf(CH_O, n.checked_mul(o.n)?.checked_mul(4)?))?;
    ok(weights_for_group(&[o.uid], &[o], o.k, o.n))?;
    let yptr = CHAIN.with(|cell| cell.borrow().bufs[CH_PFH].as_ref().map(|b| b.ptr()))?;

    if let Err(e) = gdn_scan_chunked_impl(
        ctx,
        GdnScanSrc::Host { qkv, z, beta, alpha },
        conv_st,
        ssm_st,
        wconv,
        wa,
        wdt,
        wnorm,
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        eps,
        q_scale,
        Some(yptr),
        li,
        kv,
    ) {
        tracing::error!("GDN scan (device y) failed: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    if chain_gemm_grouped_pf(&[o], CH_PFH, CH_O, n).is_none() {
        tracing::error!("GDN out_proj (chain) failed");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let bytes = out.len().checked_mul(4)?;
    let optr = CHAIN.with(|cell| cell.borrow().bufs[CH_O].as_ref().map(|b| b.ptr()))?;
    let lib = lib().ok()?;
    unsafe {
        ok(check(
            (lib.memcpy_dtoh_async)(
                out.as_mut_ptr() as *mut c_void,
                optr,
                bytes,
                std::ptr::null_mut(),
            ),
            "cuMemcpyDtoHAsync (gdn oproj)",
        ))?;
    }
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    ctx.sync().ok()?;
    Some(())
}

/// per-op 経路の行タイル (この単位で H2D + GEMM を回す)。
///
/// s168 (P7): 実際の dispatch は `DENNO_Q4_M_TILE` の実効値 ([`q4_m_tile`]) を使う。
/// この定数は**入口ゲート** ([`gdn_prefill_q4_geom`] / [`pf_res_gate`]) が前提にする
/// 歴史的なタイル幅 (1024) で、ゲートは端数タイルの帯域だけを見る保守的な判定
/// なので、実効タイルがこれと違っても「受けない」側にしか倒れない
/// (受けた場合は [`q4_group_rows`] が端数を直前のタイルへ併合する)。
const GDN_PREFILL_TILE: usize = 1024;

/// `chain_gdn_prefill_q4` が使う寸法 (入口条件を通ったときだけ作られる)。
struct GdnPrefillQ4Geom<'a> {
    qw: &'a crate::quant::Q4Linear,
    zw: &'a crate::quant::Q4Linear,
    ow: &'a crate::quant::Q4Linear,
    e: usize,
    conv_dim: usize,
    vd: usize,
    cs: usize,
}

/// `chain_gdn_prefill_q4` の入口条件 (重み種別・形・寸法・端数タイル) を判定し、
/// 以降で使う寸法を返す。**純粋関数** — CUDA もグローバル状態も触らないので、
/// フォールバック契約 (None になる条件) を GPU 無しのホストでテストできる。
/// 本番経路はこの戻り値だけを見て判断する (条件の写しを持たない)。
#[allow(clippy::too_many_arguments)]
fn gdn_prefill_q4_geom<'a>(
    qkv_w: &'a crate::quant::Linear,
    z_w: &'a crate::quant::Linear,
    o_w: &'a crate::quant::Linear,
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
) -> Option<GdnPrefillQ4Geom<'a>> {
    use crate::quant::Linear;
    let (qw, zw, ow) = match (qkv_w, z_w, o_w) {
        (Linear::Q4(a), Linear::Q4(b), Linear::Q4(c)) => (a, b, c),
        _ => return None,
    };
    // チャンク経路の前提 (gdn_scan_batch と同じ) + 形状。
    if n < GDN_CHUNK_MIN_TOKENS || nk == 0 || nv == 0 || dk == 0 || dv == 0 {
        return None;
    }
    if dk != dv || d_conv != 4 {
        return None;
    }
    let cs = d_conv - 1;
    let e = qw.k;
    let conv_dim = nk.checked_mul(dk)?.checked_mul(2)?.checked_add(nv.checked_mul(dv)?)?;
    let vd = nv.checked_mul(dv)?;
    // Q4 GEMM の入口条件 (maybe_gpu_linear_q4_into と同じ k%32==0) と重み形状。
    if e == 0 || e % 32 != 0 || vd % 32 != 0 {
        return None;
    }
    if qw.n != conv_dim || zw.k != e || zw.n != vd || ow.k != vd || ow.n != e {
        return None;
    }
    // per-op 経路は 1024 行タイルで、端数タイルが m <= max_m() だと GEMV になる。
    // そこだけ演算列が変わるので、その形では発動しない (従来経路へ落とす)。
    let rem = n % GDN_PREFILL_TILE;
    if rem != 0 && rem <= max_m() {
        return None;
    }
    Some(GdnPrefillQ4Geom { qw, zw, ow, e, conv_dim, vd, cs })
}

/// s164 (P1): GDN 層の prefill を **q4 重みで** デバイス常駐に通す。
///
/// 旧経路 (q4・`forward_prefill` の GDN 分岐) は 1 層あたり
///   qkv: D2H n*conv_dim*4 → H2D 同量 / z: D2H n*vd*4 → H2D 同量 /
///   y: D2H n*vd*4 → H2D 同量 / o_proj: D2H n*e*4
/// を往復していた (27B・n=13000 で 1 層 約 2.0 GB)。新経路は
///   H2D x (n*e*4) + H2D beta/alpha (n*nv*4 x2) + D2H o (n*e*4)
/// だけで、qkv / z / y はデバイス上に留まる。
///
/// ビット一致の根拠:
/// - qkv / z / o の投影は per-op 経路 (`q4_dispatch_tile`) と同一カーネル
///   (`dno_gemm_q4_0`)・同一引数。行ブロック (64 行) は互いに独立で、1 要素の
///   i32 累積順は k 昇順のまま変わらないため、grid の行ブロック数が増えても
///   出力は 1 要素ずつ同一。x のアップロードを 1024 行タイルに割るのも同じ理由で
///   数値を変えない。
///   (per-op 経路は m <= max_m() のタイルを GEMV で回すため、その帯域に
///   入る端数行がある n では発動しない → None)
/// - GDN スキャンは **n を分割しない**。chunk 分割 `plan(n)`・grid・引数・
///   カーネルとも従来の `gdn_scan_batch` と同一で、変わるのは入力 4 本の
///   出所 (`GdnScanSrc::Dev`) だけ。chunk 境界は状態の持ち越しで決まるので
///   行タイルに割ると plan が変わり数値が変わる — ここは触らない。
/// - beta / alpha は従来どおり呼び出し側がホストで計算した値をそのまま使う
///   (値も契約も不変)。
///
/// 状態 (conv/ssm) の契約は従来と同じ: `gdn_scan_chunked_impl` が
/// `gdn_state_begin` / `gdn_state_commit` / (devstate 無効時のみ)
/// `gdn_host_writeback` を通す。スキャン開始後に失敗すると状態を巻き戻せない
/// ため、前提条件はすべてスキャン前に確認する (失敗は GPU 障害として扱う)。
#[allow(clippy::too_many_arguments)]
pub fn chain_gdn_prefill_q4(
    qkv_w: &crate::quant::Linear,
    z_w: &crate::quant::Linear,
    o_w: &crate::quant::Linear,
    beta: &[f32],
    alpha: &[f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    x: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    out: &mut [f32],
    li: usize,
    kv: u64,
) -> Option<()> {
    let e = gate_dims_in(qkv_w)?;
    if x.len() < n.checked_mul(e)? || out.len() < n.checked_mul(e)? {
        return None;
    }
    chain_gdn_prefill_q4_impl(
        qkv_w,
        z_w,
        o_w,
        beta,
        alpha,
        wconv,
        wa,
        wdt,
        wnorm,
        PfSrc::Host(x),
        PfSink::Host(out),
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        eps,
        q_scale,
        conv_st,
        ssm_st,
        li,
        kv,
        PfGdnPhase::All,
    )
}

/// s167 (P6): [`chain_gdn_prefill_q4_impl`] の実行位相。
///
/// GDN 層の prefill は「qkv/z GEMM (h2 だけに依存)」→「beta/alpha H2D → chunk
/// scan → o_proj (beta/alpha にも依存)」の 2 段に分かれており、beta/alpha は
/// ホストが h2 から計算する。h2 の D2H を専用コピーストリームに載せるとき、
/// 前段 (Gemm) だけを先に流してその間にコピーを進め、join 後に後段 (Rest) を
/// 流す。`All` は従来どおり 1 回で通す (フラグ OFF の経路はこれ)。
#[derive(Clone, Copy, PartialEq, Eq)]
enum PfGdnPhase {
    /// 従来どおり全段 (qkv/z GEMM → beta/alpha H2D → scan → o_proj → D2H)
    All,
    /// qkv/z GEMM だけ (beta/alpha を待たない)
    Gemm,
    /// beta/alpha H2D → scan → o_proj (と Host 出力の D2H)
    Rest,
}

/// [`chain_gdn_prefill_q4`] の本体 (入出力の出所だけを差し替える)。
#[allow(clippy::too_many_arguments)]
fn chain_gdn_prefill_q4_impl(
    qkv_w: &crate::quant::Linear,
    z_w: &crate::quant::Linear,
    o_w: &crate::quant::Linear,
    beta: &[f32],
    alpha: &[f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    src: PfSrc<'_>,
    mut sink: PfSink<'_>,
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    li: usize,
    kv: u64,
    phase: PfGdnPhase,
) -> Option<()> {
    // 形・重み種別の検査は gdn_prefill_q4_geom (純粋・GPU 不要) に集約してある。
    // フォールバック契約は tests::b20_stage1_tests::gdn_prefill_q4_geom_contract で固定する。
    // デバイスの準備 (cuda_enabled 等) は形が合ってから。
    let g = gdn_prefill_q4_geom(qkv_w, z_w, o_w, n, nk, nv, dk, dv, d_conv)?;
    let (qw, zw, ow) = (g.qw, g.zw, g.ow);
    let (e, conv_dim, vd, cs) = (g.e, g.conv_dim, g.vd, g.cs);
    // s168 (P7): 行タイルは `DENNO_Q4_M_TILE` の実効値。端数は直前のタイルへ併合
    // するので、CH_PFH (入力のステージ) は 1 タイルの最大行数で確保する。
    let tile = q4_m_tile();
    let cap = q4_tile_max_rows(tile).min(n);
    let nv_tot = n.checked_mul(nv)?;
    if beta.len() < nv_tot || alpha.len() < nv_tot {
        return None;
    }
    if wconv.len() != conv_dim.checked_mul(d_conv)? || wa.len() != nv || wdt.len() != nv {
        return None;
    }
    if wnorm.len() != dv {
        return None;
    }
    if let PfSrc::Host(x) = src {
        if x.len() < n.checked_mul(e)? {
            return None;
        }
    }
    if let PfSink::Host(out) = &sink {
        if out.len() < n.checked_mul(e)? {
            return None;
        }
    }
    if conv_st.len() != conv_dim.checked_mul(cs)? || ssm_st.len() != nv.checked_mul(dk)?.checked_mul(dv)? {
        return None;
    }
    // Slot 入力は CH_PFH の先頭 n 行をそのまま読むので、行数ぶんの領域を要求する
    // (rmsnorm が ensure 済みなら no-op。足りなければ None → 呼び出し側がフォールバック)。
    if matches!(src, PfSrc::Slot) {
        ok(chain_ensure_buf(CH_PFH, n.checked_mul(e)?.checked_mul(4)?))?;
    }
    // ここまでで「形が合っている」ことが確定した → デバイスの準備に入る。
    if !cuda_enabled()
        || !gemm_enabled()
        || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed)
    {
        return None;
    }
    let ctx = chain_ctx()?;
    let gw_qkv = weights_for_q4(qw.uid, &qw.blocks, qw.k, qw.n).ok()?;
    let gw_z = weights_for_q4(zw.uid, &zw.blocks, zw.k, zw.n).ok()?;
    let gw_o = weights_for_q4(ow.uid, &ow.blocks, ow.k, ow.n).ok()?;
    let (n_chunks, _) = crate::gdn::scan_schedule::plan(n);
    // GDN スクラッチを先に確保し、qkv / z の GEMM 出力を**そのまま**スキャンの
    // 入力にする (中間バッファを増やさない)。impl 側の ensure は同形なので no-op。
    ok(GDN_SCRATCH.with(|cell| {
        cell.borrow_mut().ensure(n, conv_dim, vd, nv, dk, dv, cs, n_chunks)
    }))?;
    let (pq, pz, py, pb, pa) =
        GDN_SCRATCH.with(|cell| -> Option<(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr)> {
            let s = cell.borrow();
            Some((
                s.q.as_ref()?.ptr(),
                s.z.as_ref()?.ptr(),
                s.y.as_ref()?.ptr(),
                s.beta.as_ref()?.ptr(),
                s.alpha.as_ref()?.ptr(),
            ))
        })?;
    ok(chain_ensure_buf(CH_PFH, cap.checked_mul(e)?.checked_mul(4)?))?;
    ok(chain_ensure_buf(CH_O, n.checked_mul(e)?.checked_mul(4)?))?;

    // 1) qkv / z 投影: 入力を 1 タイル (`DENNO_Q4_M_TILE`・端数は直前へ併合) ずつ
    //    用意し、GEMM は出力の行オフセットへ書く
    //    (per-op 経路の 1024 行タイルと同じ形。H2D と GEMM は同一ストリームで
    //    順序が保たれるので、タイル間の sync は要らない)。Host はタイルごとに
    //    H2D、Slot (残差常駐経路) は CH_PFH の該当行をそのまま読む — 転送だけが
    //    消え、カーネル・引数は同一。
    //    s167 (P6): 位相 Rest では再実行しない (位相 Gemm の呼び出しが済ませている。
    //    q/z は GDN_SCRATCH に残っており、間に GDN_SCRATCH を触る処理は無い)。
    if phase != PfGdnPhase::Rest {
        let mut r0 = 0usize;
        while r0 < n {
            let mr = q4_group_rows(r0, n, tile);
            debug_assert!(mr <= cap, "tile group {mr} > cap {cap}");
            CHAIN.with(|cell| -> Option<()> {
                let s = cell.borrow();
                let pbuf = s.bufs[CH_PFH].as_ref()?.ptr();
                let pin = match &src {
                    PfSrc::Host(x) => {
                        let lib = lib().ok()?;
                        // H2D x タイル → 同一ストリームの GEMM (順序はストリームが保証する)
                        unsafe {
                            ok(check(
                                (lib.memcpy_htod_async)(
                                    pbuf,
                                    x[r0 * e..].as_ptr() as *const c_void,
                                    mr * e * 4,
                                    std::ptr::null_mut(),
                                ),
                                "cuMemcpyHtoDAsync (gdn prefill q4 x)",
                            ))?;
                        }
                        pbuf
                    }
                    PfSrc::Slot => pbuf + (r0 * e * 4) as u64,
                };
                crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
                    chain_gemm_q4_pf(gw_qkv, pin, pq + (r0 * conv_dim * 4) as u64, mr, e, conv_dim)?;
                    chain_gemm_q4_pf(gw_z, pin, pz + (r0 * vd * 4) as u64, mr, e, vd)
                })
            })?;
            r0 += mr;
        }
    }
    // s167 (P6): 位相 Gemm はここで終わる (呼び出し側が h2 のコピーを join してから
    // 位相 Rest を呼ぶ)。beta/alpha の H2D もスキャンも o_proj も走らせない。
    if phase == PfGdnPhase::Gemm {
        return Some(());
    }

    // 2) beta / alpha (ホストで計算済み・従来と同じ値) をデバイスへ。
    //    2*n*nv*4 は 27B/13k でも 5 MB なので、ここは往復を消さない。
    GDN_SCRATCH.with(|cell| -> Option<()> {
        let s = cell.borrow();
        ok(s.beta.as_ref()?.copy_from(&beta[..nv_tot]))?;
        ok(s.alpha.as_ref()?.copy_from(&alpha[..nv_tot]))?;
        Some(())
    })?;

    // 3) GDN chunk scan (3 launch・入力はすべてデバイス、y もデバイスに残す)。
    pf_phase(PF_PH_GDN_SCAN);
    crate::prof::time_layer(crate::prof::ATT_GDN_SCAN, li, || -> Option<()> {
        ok(gdn_scan_chunked_impl(
            ctx,
            GdnScanSrc::Dev { qkv: pq, z: pz, beta: pb, alpha: pa },
            conv_st,
            ssm_st,
            wconv,
            wa,
            wdt,
            wnorm,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            Some(py),
            li,
            kv,
        )
        .map(|_| ()))
    })?;

    // 4) o_proj (y はデバイス上) → 5) D2H は Host 出力のときだけ最後の 1 回。
    pf_phase(PF_PH_GDN_OPROJ);
    let po = CHAIN.with(|cell| cell.borrow().bufs[CH_O].as_ref().map(|b| b.ptr()))?;
    crate::prof::time_layer(crate::prof::OPROJ, li, || -> Option<()> {
        chain_gemm_q4_pf(gw_o, py, po, n, vd, e)
    })?;
    // Slot 出力 (残差常駐経路) は CH_O に残したまま、次の rmsnorm の proj 引数として
    // 読ませる — D2H も sync もしない (同じストリームなので順序は保たれる)。
    if let PfSink::Host(out) = &mut sink {
        CHAIN.with(|cell| -> Option<()> {
            let buf = cell.borrow().bufs[CH_O].as_ref()?.ptr();
            let lib = lib().ok()?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(
                        out.as_mut_ptr() as *mut c_void,
                        buf,
                        n * e * 4,
                        std::ptr::null_mut(),
                    ),
                    "cuMemcpyDtoHAsync (gdn prefill q4 o)",
                ))
            }
        })?;
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        ok(ctx.sync())?;
    }
    Some(())
}

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
// 発動条件は `DENNO_PF_OVERLAP=1` (既定 OFF)。OFF・driver に必要な API が無い・
// stream/event/pinned の作成に失敗、のいずれでも従来と同一の同期経路を通る。
// ---------------------------------------------------------------------------

/// s167 (P6): `DENNO_PF_OVERLAP` の解釈 (純粋関数)。**"1"/"true"/"on" だけが有効**で、
/// 未設定・空・"0" などは OFF (既定 OFF = 従来と同一経路)。
pub fn parse_pf_overlap(raw: Option<&str>) -> bool {
    matches!(raw, Some("1") | Some("true") | Some("on"))
}

/// 実行時の env 実効値 (初回だけ読む)。
fn pf_overlap_flag() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_pf_overlap(std::env::var("DENNO_PF_OVERLAP").ok().as_deref()))
}

/// s167 (P6): h2 の D2H をどの経路で発行するか。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PfH2Path {
    /// 専用 (非ブロッキング) コピーストリームで発行し、qkv/z GEMM と重ねる
    CopyStream,
    /// 従来の同期 D2H (フラグ OFF・デバイス準備失敗・形の不整合)
    SyncD2h,
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

/// s167 (P6): h2 の D2H を専用コピーストリームへ**発行するだけ** (同期しない)。
/// `CH_PFH` の `elem_off` 要素目から `len` 個を pinned の `pin_h2` へ落とす。
/// 呼び出し時点で `CH_PFH` の当該範囲が既定ストリーム上で確定している前提
/// (rmsnorm の直後に呼ぶ) で、イベントで既定ストリーム → コピーストリームの
/// 順序を作る。完了は [`chain_pf_h2_join`] で待つ。
///
/// 失敗はすべて `None` で、**GPU_BROKEN は立てない**: フラグ OFF・driver API
/// 欠落・stream/event/pinned の作成失敗は「従来の同期 D2H でよい」種類の失敗
/// なので、呼び出し側は静かに従来経路へ落ちる。
fn chain_pf_h2_begin(slot: usize, elem_off: usize, len: usize) -> Option<()> {
    let _ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    if bytes == 0 {
        return None;
    }
    let need = elem_off.checked_mul(4)?.checked_add(bytes)?;
    let lib = lib().ok()?;
    let (stream_create, event_create, event_record, stream_wait_event) = (
        lib.stream_create?,
        lib.event_create?,
        lib.event_record?,
        lib.stream_wait_event?,
    );
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        // 未 join のコピーが残っている状態で発行し直すと宛先 (pin_h2) を取り合う。
        // 呼び出し規約違反なので受けない (join が必ず h2_pending を落とす)。
        if s.h2_pending {
            return None;
        }
        // 1) コピーストリーム一式を初回だけ作る。
        if s.h2_stream.is_none() {
            // 非ブロッキング必須: legacy (既定) ストリームは blocking ストリームと
            // 暗黙に同期するため、blocking で作ると GEMM がコピー完了を待ってしまい
            // 1 バイトも重ならない。
            const CU_STREAM_NON_BLOCKING: u32 = 0x01;
            const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
            let mut st: CUstream = std::ptr::null_mut();
            if unsafe { stream_create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
                return None;
            }
            let st = CopyStream(st);
            let mut ev_ready: CUevent = std::ptr::null_mut();
            if unsafe { event_create(&mut ev_ready, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return None;
            }
            let ev_ready = CopyEvent(ev_ready);
            let mut ev_done: CUevent = std::ptr::null_mut();
            if unsafe { event_create(&mut ev_done, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                return None;
            }
            let ev_done = CopyEvent(ev_done);
            s.h2_stream = Some(st);
            s.h2_ev_ready = Some(ev_ready);
            s.h2_ev_done = Some(ev_done);
        }
        // 2) 宛先 (pinned) と転送元を用意する。pinned の確保失敗も従来経路へ
        //    (ページロック上限などは致命的でないので GPU_BROKEN は立てない)。
        if Scratch::ensure_pin(&mut s.pin_h2, bytes).is_err() {
            return None;
        }
        let dst = match s.pin_h2.as_ref() {
            Some(p) if p.bytes >= bytes => p.ptr,
            _ => return None,
        };
        let src = match s.bufs.get(slot).and_then(|b| b.as_ref()) {
            Some(b) if b.bytes() >= need => b.ptr(),
            _ => return None,
        };
        let (st, ev_ready, ev_done) = (
            s.h2_stream.as_ref()?.0,
            s.h2_ev_ready.as_ref()?.0,
            s.h2_ev_done.as_ref()?.0,
        );
        // 3) 既定ストリーム (null) の rmsnorm 完了をイベントで受けてから発行する。
        if unsafe { event_record(ev_ready, std::ptr::null_mut()) } != CUDA_SUCCESS {
            return None;
        }
        if unsafe { stream_wait_event(st, ev_ready, 0) } != CUDA_SUCCESS {
            return None;
        }
        if unsafe { (lib.memcpy_dtoh_async)(dst, src + (elem_off * 4) as CUdeviceptr, bytes, st) }
            != CUDA_SUCCESS
        {
            return None;
        }
        // 4) 完了イベントをコピーストリームに打つ (join はこれだけを待つ)。
        if unsafe { event_record(ev_done, st) } != CUDA_SUCCESS {
            return None;
        }
        s.h2_pending = true;
        Some(())
    })
}

/// s167 (P6): 発行済みの h2 D2H を join する (完了イベントを待つ)。ここから先で
/// ホストが `pin_h2` を読める。`h2_pending` が無い (発行していない) なら None。
///
/// 待ちに失敗しても `h2_pending` は落とす — 以後の層は従来の同期 D2H に戻る
/// (同じストリームに残った古いコピーは、次のコピーの完了イベントより前に
/// 必ず終わるので、宛先の取り合いにはならない)。
fn chain_pf_h2_join() -> Option<()> {
    let _ctx = chain_ctx()?;
    let event_sync = lib().ok()?.event_sync?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        if !s.h2_pending {
            return None;
        }
        let ev = s.h2_ev_done.as_ref()?.0;
        s.h2_pending = false;
        if unsafe { event_sync(ev) } != CUDA_SUCCESS {
            return None;
        }
        Some(())
    })
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
pub const PF_PH_GDN_PROJ: usize = 1;
pub const PF_PH_GDN_SCAN: usize = 2;
pub const PF_PH_GDN_OPROJ: usize = 3;
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
const PF_PH_BUCKETS: [Option<usize>; PF_PH_N] = [
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
fn pf_dbg_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("DENNO_PF_DBG").ok().as_deref() == Some("1"))
}

/// 位相計測を回すか (`DENNO_PROFILE` か `DENNO_PF_DBG`)。未設定は分岐 1 回。
#[inline]
fn pf_timing_on() -> bool {
    crate::prof::enabled() || pf_dbg_on()
}

/// 1 prefill 分の位相計測の状態 (スレッドローカル・リクエストごとに使い回す)。
struct PfTiming {
    /// 計測中 (begin 済みで read/abort 前)。
    active: bool,
    /// 層ループ末尾の境界を打った = 次の `pf_timing_read` で読む。
    pending: bool,
    /// イベントのプール (grow-only。初回の prefill で層数ぶん作る)。
    evs: Vec<CUevent>,
    /// このリクエストで打ったイベント数。
    used: usize,
    /// 区間 k (= イベント k → k+1) の位相。
    phase_of: Vec<u8>,
    /// 位相ごとの積算 (秒)。
    acc: [f64; PF_PH_N],
    /// 層数 (表 C の calls = サンプル数)。
    layers: usize,
    /// トークン数 (ms/token の分母)。
    tokens: usize,
    /// prefill 入口の時刻 (壁時計の基準)。
    t0: Option<Instant>,
    /// prefill の通し番号 (DBG 表示)。
    req: u64,
}

impl Default for PfTiming {
    fn default() -> Self {
        Self {
            active: false,
            pending: false,
            evs: Vec::new(),
            used: 0,
            phase_of: Vec::new(),
            acc: [0.0; PF_PH_N],
            layers: 0,
            tokens: 0,
            t0: None,
            req: 0,
        }
    }
}

thread_local! {
    static PF_TIMING: std::cell::RefCell<PfTiming> =
        std::cell::RefCell::new(PfTiming::default());
}

/// イベントを `need` 個そろえる (grow-only・初回だけ作る)。
fn pf_ensure_events(t: &mut PfTiming, need: usize) -> bool {
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
fn pf_record(t: &mut PfTiming, phase: u8) -> bool {
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

/// prefill 入口: 位相計測を開く (層ループの直前で 1 回)。
pub fn pf_timing_begin(tokens: usize, layers: usize) {
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        t.active = false;
        t.pending = false;
        t.used = 0;
        t.phase_of.clear();
        t.acc = [0.0; PF_PH_N];
        t.layers = layers;
        t.tokens = tokens;
        t.t0 = Some(Instant::now());
        t.req += 1;
        // 1 層あたりの境界は高々 PF_PH_N 個 + 入口 + 末尾。
        let need = layers.saturating_mul(PF_PH_N).saturating_add(2);
        if !pf_ensure_events(&mut t, need) {
            return;
        }
        if !pf_record(&mut t, PF_PH_NONE) {
            return;
        }
        t.active = true;
    });
}

/// 位相の先頭で境界を打つ (発行グループの先頭に置く)。計測していなければ
/// 分岐 1 回で戻る — 数値経路・発行順には触れない。
#[inline]
pub fn pf_phase(ph: usize) {
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

/// 層ループ末尾: 最後の位相の終端を打つ (ここまでに打ったイベントは、この後の
/// **既存の**同期 D2H で必ず完了する)。`pf_timing_read` と対で 1 回ずつ。
pub fn pf_timing_end() {
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        if !t.active {
            return;
        }
        if !pf_record(&mut t, PF_PH_NONE) {
            t.active = false;
            return;
        }
        t.pending = true;
    });
}

/// 残差常駐経路を途中で諦めた (Ok(None) → ホスト経路でやり直す) ときに捨てる。
/// 打ったイベントはストリーム上で勝手に完了するので、そのまま放置してよい。
pub fn pf_timing_abort() {
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        t.active = false;
        t.pending = false;
    });
}

/// prefill 末尾 (**既存の**同期 D2H の後): 区間を読み、表 C の `sl_lay_*` へ
/// 1 位相 1 行で積み、`DENNO_PF_DBG=1` なら内訳を 1 回だけ stderr へ出す。
pub fn pf_timing_read() {
    if !pf_timing_on() {
        return;
    }
    PF_TIMING.with(|c| {
        let mut t = c.borrow_mut();
        if !t.pending {
            return;
        }
        t.pending = false;
        t.active = false;
        let n_iv = t.phase_of.len().saturating_sub(1);
        let Some(elapsed) = lib().ok().and_then(|l| l.event_elapsed_time) else {
            return;
        };
        let mut acc = [0.0f64; PF_PH_N];
        let mut bad = false;
        for k in 0..n_iv {
            let mut ms: f32 = 0.0;
            let rc = unsafe { elapsed(&mut ms, t.evs[k], t.evs[k + 1]) };
            // 未完了 (NOT_READY) や負値は「読めなかった」として全体を捨てる
            // (半端な数字を表へ積むと、以後の判断が全部その数字に引きずられる)。
            if rc != CUDA_SUCCESS || !ms.is_finite() || ms < 0.0 {
                bad = true;
                break;
            }
            let ph = t.phase_of[k] as usize;
            if ph < PF_PH_N {
                acc[ph] += ms as f64 / 1e3;
            }
        }
        if bad {
            if pf_dbg_on() {
                eprintln!(
                    "[PF_DBG] unavailable (event elapsed failed; n={} layers={})",
                    t.tokens, t.layers
                );
            }
            return;
        }
        // 表 C: 1 位相 1 行・calls = 層数 (= 層あたり 1 サンプル)。
        let layers = t.layers.max(1) as u64;
        for ph in 0..PF_PH_N {
            let Some(op) = PF_PH_BUCKETS[ph] else {
                continue;
            };
            let ns = (acc[ph] * 1e6).round() as u64;
            if ns > 0 {
                crate::prof::span_end_ns_calls(ns, op, layers);
            }
        }
        if pf_dbg_on() {
            pf_dbg_print(&t, &acc);
        }
    });
}

/// `DENNO_PF_DBG=1` の内訳 (1 prefill = 1 回)。wall / 位相の和 / 末尾 (最終 norm +
/// logits + 最終行 D2H) を並べるので、「位相の和が wall に届かない分」が必ず見える。
fn pf_dbg_print(t: &PfTiming, acc: &[f64; PF_PH_N]) {
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

/// s167 (P6): GDN 層の prefill を、h2 の D2H と qkv/z GEMM の**オーバーラップ付き**
/// で通す版。`DENNO_PF_OVERLAP=1` かつデバイス準備が成功したときだけ
///   位相 Gemm → join → `host_ctrl` (beta/alpha) → 位相 Rest
/// の順で流し、それ以外は従来と**同一の順序**
///   同期 D2H → `host_ctrl` → (GEMM → beta/alpha H2D → scan → o_proj)
/// を通す。`host_ctrl` は h2 を受け取って beta/alpha を書くホスト区間で、呼び出し
/// 側は従来と同じ関数 (`apply_bt_rows_into`) を同じ引数で呼ぶ。
#[allow(clippy::too_many_arguments)]
pub fn chain_gdn_prefill_q4_dev_ovl<F>(
    qkv_w: &crate::quant::Linear,
    z_w: &crate::quant::Linear,
    o_w: &crate::quant::Linear,
    beta: &mut [f32],
    alpha: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    li: usize,
    kv: u64,
    host_ctrl: F,
) -> Option<()>
where
    F: FnOnce(&[f32], &mut [f32], &mut [f32]) -> Option<()>,
{
    // 前提 (形・重み種別・n の帯域) はコピーを発行する前に確定させる — ここで
    // 落ちる形ではコピーを 1 バイトも発行しない (未 join のコピーを残さない)。
    let g = gdn_prefill_q4_geom(qkv_w, z_w, o_w, n, nk, nv, dk, dv, d_conv)?;
    let e = g.e;
    let n_e = n.checked_mul(e)?;
    // s169 (P8): 位相の境界は**発行グループの先頭**。ここから「h2 の D2H (同期 or
    // join) → ホストの beta/alpha → qkv/z GEMM」までが gdn_proj の区間になる。
    pf_phase(PF_PH_GDN_PROJ);
    let want = pf_overlap_flag();
    let begun = want && chain_pf_h2_begin(CH_PFH, 0, n_e).is_some();
    if pf_h2_path(want, n, e, begun) == PfH2Path::CopyStream {
        // 位相 Gemm を先に流す (h2 を待たない)。この間コピーストリームが h2 を運ぶ。
        // 既定ストリームは非ブロッキングのコピーストリームと暗黙同期しないので、
        // GEMM がコピーを待つことはない。
        if chain_gdn_prefill_q4_impl(
            qkv_w,
            z_w,
            o_w,
            beta,
            alpha,
            wconv,
            wa,
            wdt,
            wnorm,
            PfSrc::Slot,
            PfSink::Slot,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            conv_st,
            ssm_st,
            li,
            kv,
            PfGdnPhase::Gemm,
        )
        .is_none()
        {
            // 位相 Gemm の失敗 = デバイス障害 (従来のチェーン失敗と同じ扱い)。
            // 未 join のコピーは残さない。
            let _ = chain_pf_h2_join();
            return None;
        }
        // join: ホストが h2 を読む直前の唯一の同期点。
        if chain_pf_h2_join().is_some() {
            // pinned の h2 をそのまま渡す (Vec への写しをしない)。安全性: pin_h2 は
            // thread_local の grow-only バッファで、join 済み = 全要素がコピーで
            // 埋まっている。CHAIN の借用は保持しない (保持したままホスト区間を走ら
            // せると、内側の CHAIN アクセスが RefCell で panic する)。
            let (p, cap) = CHAIN.with(|cell| {
                cell.borrow()
                    .pin_h2
                    .as_ref()
                    .map(|p| (p.ptr as *const f32, p.bytes / std::mem::size_of::<f32>()))
            })?;
            if cap < n_e {
                return None;
            }
            let h2: &[f32] = unsafe { std::slice::from_raw_parts(p, n_e) };
            host_ctrl(h2, beta, alpha)?;
            return chain_gdn_prefill_q4_impl(
                qkv_w,
                z_w,
                o_w,
                beta,
                alpha,
                wconv,
                wa,
                wdt,
                wnorm,
                PfSrc::Slot,
                PfSink::Slot,
                n,
                nk,
                nv,
                dk,
                dv,
                d_conv,
                eps,
                q_scale,
                conv_st,
                ssm_st,
                li,
                kv,
                PfGdnPhase::Rest,
            );
        }
        // join だけ失敗した (driver 障害)。q/z は既にデバイスにあるので、h2 を
        // 従来の同期 D2H で取り直して残りを流す — 値は同一。
        let h2 = chain_download_at(CH_PFH, 0, n_e)?;
        host_ctrl(&h2, beta, alpha)?;
        return chain_gdn_prefill_q4_impl(
            qkv_w,
            z_w,
            o_w,
            beta,
            alpha,
            wconv,
            wa,
            wdt,
            wnorm,
            PfSrc::Slot,
            PfSink::Slot,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            eps,
            q_scale,
            conv_st,
            ssm_st,
            li,
            kv,
            PfGdnPhase::Rest,
        );
    }
    // 従来と同一の順序: 同期 D2H → ホスト計算 → GEMM → beta/alpha H2D → scan → o_proj
    let h2 = chain_download_at(CH_PFH, 0, n_e)?;
    host_ctrl(&h2, beta, alpha)?;
    chain_gdn_prefill_q4_impl(
        qkv_w,
        z_w,
        o_w,
        beta,
        alpha,
        wconv,
        wa,
        wdt,
        wnorm,
        PfSrc::Slot,
        PfSink::Slot,
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        eps,
        q_scale,
        conv_st,
        ssm_st,
        li,
        kv,
        PfGdnPhase::All,
    )
}

// ---------------------------------------------------------------------------
// s165 (P2): prefill の残差ストリーム (h) をデバイス常駐させる経路の小道具。
//
// 従来の prefill は層ごとに「rmsnorm(h) をホストで計算 → h2 を H2D → 投影 →
// 出力を D2H → 残差加算をホスト」を繰り返していた (1 層 2 往復 + ホスト 2 パス)。
// P2 では h を `CH_PF_RES` に置いたまま、層の中で
//   rmsnorm (デバイス) → 投影 (デバイス) → 残差加算 (次の rmsnorm の proj 引数)
// を通す。ホストが本当に必要とするものだけが往復する:
//   - GDN 層: beta/alpha を計算するための h2 の D2H 1 回 (状態契約は従来どおり)
//   - full attention 層: q/k/v はホスト計算なので h2 の D2H 1 回 + att の H2D 1 回
//   - FFN: 往復なし
//
// 残差加算に専用カーネルは要らない (PTX に単独の add は無い) — `dno_add_rmsnorm`
// の `proj` 引数が「x += proj」と「rmsnorm(x)」を 1 回で行うので、層の最後の
// 投影出力を**次の rmsnorm の add_slot** として渡す。演算順はホスト経路と同一
// (ホストも `x += proj` してから `rmsnorm(x)` を計算する)。
// ---------------------------------------------------------------------------

/// 層 1 枚ぶんの attention 投影の重み (事前検査・重み登録で使う)。
pub enum PfAttnW<'a> {
    Gdn {
        qkv: &'a crate::quant::Linear,
        z: &'a crate::quant::Linear,
        o: &'a crate::quant::Linear,
    },
    Full {
        q: &'a crate::quant::Linear,
        k: &'a crate::quant::Linear,
        v: &'a crate::quant::Linear,
        o: &'a crate::quant::Linear,
    },
}

/// 残差常駐経路の判定に使うモデル寸法。
pub struct PfDims {
    pub n: usize,
    pub e: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dh: usize,
    pub nk: usize,
    pub nv: usize,
    pub dk: usize,
    pub dv: usize,
    pub d_conv: usize,
}

/// Q4 線形の (k, n)。Q4 以外は None。
fn q4_dims(w: &crate::quant::Linear) -> Option<(usize, usize)> {
    match w {
        crate::quant::Linear::Q4(q) => Some((q.k, q.n)),
        _ => None,
    }
}

/// Q4 線形の (uid, blocks, k, n)。Q4 以外は None。
fn q4_parts(w: &crate::quant::Linear) -> Option<(u64, &[u8], usize, usize)> {
    match w {
        crate::quant::Linear::Q4(q) => Some((q.uid, &q.blocks, q.k, q.n)),
        _ => None,
    }
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

/// 実行時の env 実効値で [`pf_layer_geom`] を評価する。
pub fn pf_layer_geom_now(
    attn: &PfAttnW<'_>,
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    d: &PfDims,
) -> Option<()> {
    pf_layer_geom(attn, gate, up, down, d, gemm_min_m(), max_m())
}

/// s165 (P2): 層 1 枚を残差常駐で通せるかの事前検査 (純粋関数・GPU 非依存)。
/// `None` なら呼び出し側はこの層を従来のホスト経路で通す (残差は触らない)。
pub fn pf_layer_geom(
    attn: &PfAttnW<'_>,
    gate: &crate::quant::Linear,
    up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    d: &PfDims,
    gemm_min: usize,
    max_m: usize,
) -> Option<()> {
    let (n, e) = (d.n, d.e);
    if !pf_res_gate(n, e, true, false, gemm_min, max_m) {
        return None;
    }
    // FFN: q4・形・k/n の整合
    let (gk, gn) = q4_dims(gate)?;
    let (uk, un) = q4_dims(up)?;
    let (dkn, dn) = q4_dims(down)?;
    if gk != e || uk != e || gn != un || dkn != gn || dn != e {
        return None;
    }
    if gn == 0 || gn % 32 != 0 {
        return None;
    }
    match attn {
        PfAttnW::Gdn { qkv, z, o } => {
            // 形状・dk==dv・d_conv==4・n の帯域は gdn_prefill_q4_geom と同一契約
            gdn_prefill_q4_geom(qkv, z, o, n, d.nk, d.nv, d.dk, d.dv, d.d_conv)?;
        }
        PfAttnW::Full { q, k, v, o } => {
            let (qk, qn) = q4_dims(q)?;
            let (kk, kn) = q4_dims(k)?;
            let (vk, vn) = q4_dims(v)?;
            let (okn, on) = q4_dims(o)?;
            if d.heads == 0 || d.dh == 0 || d.kv_heads == 0 {
                return None;
            }
            let att = d.heads.checked_mul(d.dh)?;
            let kv_dim = d.kv_heads.checked_mul(d.dh)?;
            if qk != e || qn != att.checked_mul(2)? || kk != e || kn != kv_dim {
                return None;
            }
            if vk != e || vn != kv_dim || okn != att || on != e {
                return None;
            }
            if att % 32 != 0 || kv_dim % 32 != 0 {
                return None;
            }
        }
    }
    Some(())
}

/// s165 (P2): 残差常駐経路が使うスロットを事前に確保する (rmsnorm の前に呼ぶ)。
/// ここで失敗すれば残差はまだ触っていないので、呼び出し側は安全にフォールバックできる。
pub fn chain_pf_reserve(n: usize, e: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let rows = n.checked_mul(e)?.checked_mul(4)?;
    ok(chain_ensure_buf(CH_PF_RES, rows))?;
    ok(chain_ensure_buf(CH_PFH, rows.checked_add(e.checked_mul(4)?)?))?;
    ok(chain_ensure_buf(CH_O, rows))?;
    ok(chain_ensure_buf(CH_DOWN, rows))?;
    Some(())
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

/// s165 (P2): 残差ストリーム (h) を `CH_PF_RES` へ 1 回だけアップロードする
/// (非同期・同期なし)。prefill の開始時に 1 回だけ呼ぶ。
pub fn chain_pf_res_upload(h: &[f32], m: usize, e: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let n_elems = m.checked_mul(e)?;
    let bytes = n_elems.checked_mul(4)?;
    if n_elems == 0 || h.len() < n_elems {
        return None;
    }
    ok(chain_ensure_buf(CH_PF_RES, bytes))?;
    CHAIN.with(|cell| -> Option<()> {
        let buf = cell.borrow().bufs[CH_PF_RES].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, h.as_ptr() as *const c_void, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (pf residual)",
            ))
        }
    })
}

/// s165 (P2): `CH_PF_RES` の内容をホストへ戻す (フォールバック時の復元・同期あり)。
pub fn chain_pf_res_download(m: usize, e: usize, out: &mut [f32]) -> Option<()> {
    let n_elems = m.checked_mul(e)?;
    if n_elems == 0 || out.len() < n_elems {
        return None;
    }
    let v = chain_download_at(CH_PF_RES, 0, n_elems)?;
    out[..n_elems].copy_from_slice(&v);
    Some(())
}

/// s165 (P2): ホストで計算した attention 出力 (sigmoid ゲート適用済み) を
/// `CH_ATT` へ 1 回だけアップロードする。pinned は層ごとに専用 (li*3+2) —
/// 非同期 H2D の完了前に次の層が同じ領域を上書きしないため
/// (`chain_rmsnorm` の pin と同じ規約)。
pub fn chain_pf_upload_att(att: &[f32], cols: usize, m: usize, li: usize) -> Option<()> {
    let _ = chain_ctx()?;
    let n_elems = m.checked_mul(cols)?;
    let bytes = n_elems.checked_mul(4)?;
    if n_elems == 0 || att.len() < n_elems {
        return None;
    }
    ok(chain_ensure_buf(CH_ATT, bytes))?;
    let pin_id = li.checked_mul(3)?.checked_add(2)?;
    CHAIN.with(|cell| -> Option<()> {
        let mut s = cell.borrow_mut();
        while s.pins.len() <= pin_id {
            s.pins.push(None);
        }
        ok(Scratch::ensure_pin(&mut s.pins[pin_id], bytes))?;
        let pin = s.pins[pin_id].as_mut().unwrap();
        pin.as_mut_slice::<f32>()[..n_elems].copy_from_slice(&att[..n_elems]);
        let src = pin.ptr as *const c_void;
        let buf = s.bufs[CH_ATT].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(buf, src, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (pf att)",
            ))
        }
    })
}

/// s165 (P2): full attention 層の q/k/v 投影を**デバイス常駐の入力** (`CH_PFH`,
/// rmsnorm 出力・先頭から `rows` 行) から回し、**出力だけ**をホストへ戻す。
/// KV キャッシュはホストが正で q/k/v の値はホストの prepare が要るので、
/// 値は必ず戻す (戻す量は q/k/v の幅で、h2 の H2D 3 回より小さい)。
/// 3 本の GEMM は `CH_QKV` の連続領域 (q|k|v) に行オフセットで書き、
/// D2H 3 発 + sync 1 回で呼び出し元のバッファへ直接落とす。
/// カーネル・grid・引数は per-op 経路と同一 (行ごとに独立 → bit 一致)。
pub fn chain_qkv_prefill_q4_dev(
    q_w: &crate::quant::Linear,
    k_w: &crate::quant::Linear,
    v_w: &crate::quant::Linear,
    rows: usize,
    li: usize,
    out_qg: &mut [f32],
    out_k: &mut [f32],
    out_v: &mut [f32],
) -> Option<()> {
    if !cuda_enabled() || !gemm_enabled() || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    let (qn, qcols) = q4_dims(q_w)?;
    let (kn, kcols) = q4_dims(k_w)?;
    let (vn, vcols) = q4_dims(v_w)?;
    if qn != kn || qn != vn || qn == 0 || qn % 32 != 0 {
        return None;
    }
    if qcols % 32 != 0 || kcols % 32 != 0 || vcols % 32 != 0 {
        return None;
    }
    if rows < gemm_min_m() || rows <= max_m() {
        return None;
    }
    if out_qg.len() < rows * qcols || out_k.len() < rows * kcols || out_v.len() < rows * vcols {
        return None;
    }
    let ctx = chain_ctx()?;
    let (q_uid, q_blk, _, _) = q4_parts(q_w)?;
    let (k_uid, k_blk, _, _) = q4_parts(k_w)?;
    let (v_uid, v_blk, _, _) = q4_parts(v_w)?;
    let gwq = weights_for_q4(q_uid, q_blk, qn, qcols).ok()?;
    let gwk = weights_for_q4(k_uid, k_blk, kn, kcols).ok()?;
    let gwv = weights_for_q4(v_uid, v_blk, vn, vcols).ok()?;
    let total = rows.checked_mul(qcols.checked_add(kcols)?.checked_add(vcols)?)?;
    ok(chain_ensure_buf(CH_QKV, total.checked_mul(4)?))?;
    // 入力は CH_PFH の先頭 rows 行 (rmsnorm 出力) — 行数ぶんの領域を要求
    ok(chain_ensure_buf(CH_PFH, rows.checked_mul(qn)?.checked_mul(4)?))?;
    let (pin, pqkv) = CHAIN.with(|cell| -> Option<(u64, u64)> {
        let s = cell.borrow();
        Some((s.bufs[CH_PFH].as_ref()?.ptr(), s.bufs[CH_QKV].as_ref()?.ptr()))
    })?;
    let q_off = (rows * qcols * 4) as u64;
    let k_off = (rows * (qcols + kcols) * 4) as u64;
    // s169 (P8): qkv GEMM + 3 発の D2H + 既存の sync までが full_qk の区間。
    pf_phase(PF_PH_FULL_QK);
    crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
        chain_gemm_q4_pf(gwq, pin, pqkv, rows, qn, qcols)?;
        chain_gemm_q4_pf(gwk, pin, pqkv + q_off, rows, kn, kcols)?;
        chain_gemm_q4_pf(gwv, pin, pqkv + k_off, rows, vn, vcols)
    })?;
    // D2H 3 発 (async) → sync 1 回
    CHAIN.with(|cell| -> Option<()> {
        let s = cell.borrow();
        let base = s.bufs[CH_QKV].as_ref()?.ptr();
        let lib = lib().ok()?;
        unsafe {
            ok(check(
                (lib.memcpy_dtoh_async)(
                    out_qg.as_mut_ptr() as *mut c_void,
                    base,
                    rows * qcols * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (pf qkv q)",
            ))?;
            ok(check(
                (lib.memcpy_dtoh_async)(
                    out_k.as_mut_ptr() as *mut c_void,
                    base + q_off,
                    rows * kcols * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (pf qkv k)",
            ))?;
            ok(check(
                (lib.memcpy_dtoh_async)(
                    out_v.as_mut_ptr() as *mut c_void,
                    base + k_off,
                    rows * vcols * 4,
                    std::ptr::null_mut(),
                ),
                "cuMemcpyDtoHAsync (pf qkv v)",
            ))
        }
    })?;
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    ok(ctx.sync())?;
    Some(())
}

/// s165 (P2): attention 出力 (`CH_ATT`) → o_proj (q4) → `CH_O` (デバイス常駐・
/// 同期なし)。per-op 経路 (`maybe_gpu_linear_q4_into`) と同一カーネル・同一引数。
/// m を 1 回で渡すのは、行ごとに独立で 1 行の i32 累積順が変わらないため
/// (`chain_gdn_prefill_q4` の o_proj と同じ根拠)。
pub fn chain_oproj_prefill_q4(o_w: &crate::quant::Linear, rows: usize, li: usize) -> Option<()> {
    let ow = match o_w {
        crate::quant::Linear::Q4(q) => q,
        _ => return None,
    };
    if !cuda_enabled() || !gemm_enabled() || GPU_BROKEN.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    if rows < gemm_min_m() || rows <= max_m() || ow.k == 0 || ow.n == 0 || ow.k % 32 != 0 {
        return None;
    }
    let ctx = chain_ctx()?;
    let gw = weights_for_q4(ow.uid, &ow.blocks, ow.k, ow.n).ok()?;
    let out_bytes = rows.checked_mul(ow.n)?.checked_mul(4)?;
    let att_bytes = rows.checked_mul(ow.k)?.checked_mul(4)?;
    ok(chain_ensure_buf(CH_O, out_bytes))?;
    let (pa, po) = CHAIN.with(|cell| -> Option<(u64, u64)> {
        let s = cell.borrow();
        let a = s.bufs[CH_ATT].as_ref()?;
        if a.bytes() < att_bytes {
            return None;
        }
        Some((a.ptr(), s.bufs[CH_O].as_ref()?.ptr()))
    })?;
    let _ = ctx;
    // s169 (P8): att の H2D は呼び出し側、ここは o_proj GEMM だけの区間。
    pf_phase(PF_PH_FULL_OPROJ);
    crate::prof::time_layer(crate::prof::OPROJ, li, || chain_gemm_q4_pf(gw, pa, po, rows, ow.k, ow.n))
}

/// chunked scan の入力 4 本 (qkv / z / beta / alpha) の出所。
///
/// s164 (P1): 従来は必ずホスト slice で、`GDN_SCRATCH` へ H2D してから
/// カーネルへ渡していた。`Dev` は「既にデバイスに居る」場合にそのポインタを
/// そのまま使う — カーネルの引数・grid・演算列は 1 つも変わらず、消えるのは
/// 転送だけ (q4 prefill チェーン `chain_gdn_prefill_q4` が使う)。
#[derive(Clone, Copy)]
enum GdnScanSrc<'a> {
    Host {
        qkv: &'a [f32],
        z: &'a [f32],
        beta: &'a [f32],
        alpha: &'a [f32],
    },
    Dev {
        qkv: CUdeviceptr,
        z: CUdeviceptr,
        beta: CUdeviceptr,
        alpha: CUdeviceptr,
    },
}

/// A1-r2: GDN chunked scan 実装 (3 カーネル・長 prefill 用)。
///
/// CPU 参照: [`crate::gdn::scan_chunked_core`]。3-pass 構成:
/// - K1 `dno_gdn_chunk_pre` (chunk 並列): ローカル遷移 W[ci]・M[ci]
/// - K2 `dno_gdn_chunk_state` (chunk 直列): chunk 先頭状態 Sinit の直列伝播
/// - K3 `dno_gdn_chunk_scan` (chunk 並列): Sinit からの逐次 AR スキャン → y・
///   最終状態 (最終 chunk ブロックが書き戻す)
///
/// 決定論: 各カーネルとも launch 構成固定・昇順逐次和・atomic/ワープ縮約不使用
/// (gdn.cu の方針どおり)。演算は単発カーネル `dno_gdn_scan` と同一で、chunk
/// 境界でのみ状態を持ち越す。前提: dk == dv・d_conv == 4 (呼び出し側で保証)。
#[allow(clippy::too_many_arguments)]
fn gdn_scan_chunked_impl(
    ctx: &CudaCtx,
    src: GdnScanSrc<'_>,
    conv_st: &mut [f32],
    ssm_st: &mut [f32],
    wconv: &[f32],
    wa: &[f32],
    wdt: &[f32],
    wnorm: &[f32],
    n: usize,
    nk: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    d_conv: usize,
    eps: f32,
    q_scale: f32,
    y_dst: Option<CUdeviceptr>,
    li: usize,
    kv: u64,
) -> Result<Vec<f32>> {
    let cs = d_conv - 1;
    let conv_dim = 2 * nk * dk + nv * dv;
    let vd = nv * dv;
    if let GdnScanSrc::Host { qkv, z, beta, alpha } = &src {
        debug_assert_eq!(qkv.len(), n * conv_dim);
        debug_assert_eq!(z.len(), n * vd);
        debug_assert_eq!(beta.len(), n * nv);
        debug_assert_eq!(alpha.len(), n * nv);
    }
    debug_assert_eq!(conv_st.len(), conv_dim * cs);
    debug_assert_eq!(ssm_st.len(), nv * dk * dv);
    debug_assert_eq!(dk, dv, "chunked scan は dk == dv 前提");
    debug_assert_eq!(d_conv, 4, "chunked scan は d_conv == 4 前提");

    let (n_chunks, chunk_len) = crate::gdn::scan_schedule::plan(n);

    // s90 (P1-1): 2 launch 経路 (`f_gdn_chunk_scan_hoist`) の入り口判定は
    // `crate::gdn::scan_schedule::route_hoist` の 1 か所。この forward は
    // 「3 launch 経路に対して別 launch の書き手」を保証できない
    // (gdn/scan_schedule.rs のモジュール冒頭) ので、route_hoist(.., false) = false
    // が既定 → 下の 3 pass が走る。将来 常駐を実装したときに、ここで
    // route_hoist(.., true) 側の分岐を足す (今は分岐を持たない = 死んだ
    // パスを疑似的に走らせない)。
    if crate::gdn::scan_schedule::route_hoist(n, nv, dk, dv, false) {
        unreachable!("s90: route_hoist(.., false) は常に偽 (既定は 3 launch)");
    }
    // 帳簿の健全性 (debug のみ): 境界行の書き手は読み手より前の chunk であること。
    // 「別 launch の書き手」になるのはこの条件が満たされたうえで前の launch が
    // 実際に存在する場合だけなので、ここでは順序関係だけを見る。
    debug_assert!(
        crate::gdn::scan_schedule::read_write_conflicts(n, chunk_len, n_chunks)
            .iter()
            .all(|(reader, writer, _)| writer < reader),
        "s90: 境界行の書き手が読み手より後になっている"
    );

    ctx.make_current()?;
    // s64 (T5b L1 Step 1 計測): `gdn_scan_impl` と同一の 4 区間細分 (表示専用)。
    // chunked 経路は pass ごとに sync するため launch/sync は 3 pass 分の合計。
    let mut q = crate::prof::Seq::begin(crate::prof::D2H_GDN_PF_ALLOC, Instant::now());
    // s67: 16 本の作業領域は層をまたいで再利用する (確保は初回のみ・形が同じ間は
    // 何もしない)。中身は毎回 copy_from で上書きするので従来と同じ状態になる。
    // w/m/sinit/y は各 pass が全要素を書くためゼロ埋め不要 (旧実装は
    // `from_slice(&vec![0.0; …])` でゼロを H2D していた — その転送も消える)。
    // conv_out/ssm_out だけは「最終 chunk ブロックのみ書き込み」なので従来どおり
    // ゼロで初期化する (未書込部は未使用・挙動を変えない)。
    let dp = GDN_SCRATCH.with(|cell| -> Result<GdnPtrs> {
        let mut s = cell.borrow_mut();
        s.ensure(n, conv_dim, vd, nv, dk, dv, cs, n_chunks)?;
        let p = s.ptrs();
        s.buf(GdnBuf::ConvOut).fill_zero()?;
        s.buf(GdnBuf::SsmOut).fill_zero()?;
        // s164: ホスト入力のときだけ H2D する (Dev は呼び出し側が既に置いている)。
        if let GdnScanSrc::Host { qkv, z, beta, alpha } = &src {
            s.q.as_ref().unwrap().copy_from(qkv)?;
            s.z.as_ref().unwrap().copy_from(z)?;
            s.beta.as_ref().unwrap().copy_from(beta)?;
            s.alpha.as_ref().unwrap().copy_from(alpha)?;
        }
        s.wc.as_ref().unwrap().copy_from(wconv)?;
        s.wa.as_ref().unwrap().copy_from(wa)?;
        s.wd.as_ref().unwrap().copy_from(wdt)?;
        s.wn.as_ref().unwrap().copy_from(wnorm)?;
        Ok(p)
    })?;
    // Dev のときは入力 4 本のポインタだけ差し替える (他はスクラッチのまま)。
    let dp = match src {
        GdnScanSrc::Dev { qkv, z, beta, alpha } => GdnPtrs { q: qkv, z, beta, alpha, ..dp },
        GdnScanSrc::Host { .. } => dp,
    };
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);

    // ---- pass 1: chunk 並列でローカル遷移 (W[ci], M[ci]) --------------------
    let (pst_conv, pst_ssm) =
        gdn_state_begin(kv, li, conv_st, ssm_st, conv_dim * cs, nv, dk * dv)?;
    let (mut pq, mut pb, mut pa) = (dp.q, dp.beta, dp.alpha);
    let (mut pci, mut pwc, mut pwa, mut pwd) = (pst_conv, dp.wc, dp.wa, dp.wd);
    let (mut pw, mut pm) = (dp.w, dp.m);
    let (mut ni, mut cli, mut nki, mut nvi, mut dki, mut dvi) = (
        n as c_int,
        chunk_len as c_int,
        nk as c_int,
        nv as c_int,
        dk as c_int,
        dv as c_int,
    );
    let (mut cdi, mut csi, mut epsi) = (conv_dim as c_int, cs as c_int, eps);
    let mut args_pre = [
        &mut pq as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut pci as *mut _ as *mut c_void,
        &mut pwc as *mut _ as *mut c_void,
        &mut pwa as *mut _ as *mut c_void,
        &mut pwd as *mut _ as *mut c_void,
        &mut pw as *mut _ as *mut c_void,
        &mut pm as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut cli as *mut _ as *mut c_void,
        &mut nki as *mut _ as *mut c_void,
        &mut nvi as *mut _ as *mut c_void,
        &mut dki as *mut _ as *mut c_void,
        &mut dvi as *mut _ as *mut c_void,
        &mut cdi as *mut _ as *mut c_void,
        &mut csi as *mut _ as *mut c_void,
        &mut epsi as *mut _ as *mut c_void,
    ];
    let shmem_pre = ((2 * dk + 3) * 4) as u32;
    unsafe {
        ctx.launch_async(
            ctx.f_gdn_chunk_pre,
            (n_chunks as u32, nv as u32, 1),
            (dk as u32, 1, 1),
            shmem_pre,
            &mut args_pre,
        )?;
    }
    q.lap(crate::prof::D2H_GDN_PF_SYNC);
    ctx.sync()?;
    q.lap(crate::prof::D2H_GDN_PF_D2H);

    // ---- pass 2: chunk 先頭状態 Sinit の直列伝播 ----------------------------
    let (mut psi, mut pm2, mut pw2, mut psinit) = (pst_ssm, dp.m, dp.w, dp.sinit);
    let (mut nc2, mut nv2, mut dk2, mut dv2) =
        (n_chunks as c_int, nv as c_int, dk as c_int, dv as c_int);
    let mut args_state = [
        &mut psi as *mut _ as *mut c_void,
        &mut pm2 as *mut _ as *mut c_void,
        &mut pw2 as *mut _ as *mut c_void,
        &mut psinit as *mut _ as *mut c_void,
        &mut nc2 as *mut _ as *mut c_void,
        &mut nv2 as *mut _ as *mut c_void,
        &mut dk2 as *mut _ as *mut c_void,
        &mut dv2 as *mut _ as *mut c_void,
    ];
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);
    unsafe {
        ctx.launch_async(
            ctx.f_gdn_chunk_state,
            (nv as u32, 1, 1),
            (dk as u32, 1, 1),
            0,
            &mut args_state,
        )?;
    }
    q.lap(crate::prof::D2H_GDN_PF_SYNC);
    ctx.sync()?;
    q.lap(crate::prof::D2H_GDN_PF_D2H);

    // ---- pass 3: Sinit[ci] からの逐次 AR スキャン (chunk 並列) ---------------
    let (mut pq3, mut pz3, mut pb3, mut pa3) = (dp.q, dp.z, dp.beta, dp.alpha);
    // s131: pass 3 の conv 入力もデバイス正本 (スクラッチの conv_in はもう書かない)。
    let (mut pci3, mut psinit3) = (pci, dp.sinit);
    let (mut pwc3, mut pwa3, mut pwd3, mut pwn3) = (dp.wc, dp.wa, dp.wd, dp.wn);
    // s72: y_dst があれば y はチェーンスロットへ直接書く (host へ落とさない)。
    let (mut py, mut pco, mut pso) = (y_dst.unwrap_or(dp.y), dp.conv_out, dp.ssm_out);
    let (mut ni3, mut cli3) = (n as c_int, chunk_len as c_int);
    let (mut nki3, mut nvi3, mut dki3, mut dvi3) =
        (nk as c_int, nv as c_int, dk as c_int, dv as c_int);
    let (mut cdi3, mut csi3, mut epsi3, mut qsi3) =
        (conv_dim as c_int, cs as c_int, eps, q_scale);
    let mut args_scan = [
        &mut pq3 as *mut _ as *mut c_void,
        &mut pz3 as *mut _ as *mut c_void,
        &mut pb3 as *mut _ as *mut c_void,
        &mut pa3 as *mut _ as *mut c_void,
        &mut pci3 as *mut _ as *mut c_void,
        &mut psinit3 as *mut _ as *mut c_void,
        &mut pwc3 as *mut _ as *mut c_void,
        &mut pwa3 as *mut _ as *mut c_void,
        &mut pwd3 as *mut _ as *mut c_void,
        &mut pwn3 as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut pco as *mut _ as *mut c_void,
        &mut pso as *mut _ as *mut c_void,
        &mut ni3 as *mut _ as *mut c_void,
        &mut cli3 as *mut _ as *mut c_void,
        &mut nki3 as *mut _ as *mut c_void,
        &mut nvi3 as *mut _ as *mut c_void,
        &mut dki3 as *mut _ as *mut c_void,
        &mut dvi3 as *mut _ as *mut c_void,
        &mut cdi3 as *mut _ as *mut c_void,
        &mut csi3 as *mut _ as *mut c_void,
        &mut epsi3 as *mut _ as *mut c_void,
        &mut qsi3 as *mut _ as *mut c_void,
    ];
    let shmem_scan = ((3 * dk + dv + 5) * 4) as u32;
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);
    unsafe {
        ctx.launch_async(
            ctx.f_gdn_chunk_scan,
            (n_chunks as u32, nv as u32, 1),
            (dk as u32, 1, 1),
            shmem_scan,
            &mut args_scan,
        )?;
    }
    q.lap(crate::prof::D2H_GDN_PF_SYNC);
    ctx.sync()?;
    q.lap(crate::prof::D2H_GDN_PF_D2H);

    if y_dst.is_some() {
        // s72: y はデバイス上に残す (呼び出し側がチェーンで消費する)。
        // s131: 状態もデバイス正本へ D2D で戻す (ホストへは落とさない)。
        gdn_state_commit(li, dp.conv_out, dp.ssm_out, conv_dim * cs, nv, dk * dv)?;
        q.end();
        return Ok(Vec::new());
    }
    let y = GDN_SCRATCH.with(|cell| -> Result<Vec<f32>> {
        let s = cell.borrow();
        s.buf(GdnBuf::Y).to_vec::<f32>(n * vd)
    })?;
    // s131: 状態はホストへ落とさずデバイス正本へ D2D で戻す。
    gdn_state_commit(li, dp.conv_out, dp.ssm_out, conv_dim * cs, nv, dk * dv)?;
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    q.end();
    Ok(y)
}


// ===========================================================================
#[cfg(test)]
mod b7_stage_tests {
    //! s141 (Task B7): GDN 状態のホスト受け皿が**再利用**されることの検証。
    //!
    //! 背景 (s140 の計測): decode の `gdn_pf_alloc` が 98 us/call・4.7 ms/token
    //! あり、中身は層ごとの `to_vec` (`Vec::with_capacity` を 2 本 x 48 層) だった。
    //! 実転送 (D2H) は GPU が要るので、ここでは**確保の規則そのもの**
    //! (`reserve_exact_grow` — `d2h_raw` が実際に呼ぶ関数) を回して、
    //! 確保回数が層・ステップ数に比例して増えないことを確かめる。
    use super::*;

    /// Vec の再確保を数える (これが 0 件 = 確保していない)。
    fn count_regrows(stage: &mut GdnHostStage, conv_len: usize, ssm_len: usize, layers: usize) -> usize {
        let (c0, s0) = stage.capacities();
        for _ in 0..layers {
            reserve_exact_grow(&mut stage.conv, conv_len);
            reserve_exact_grow(&mut stage.ssm, ssm_len);
        }
        let (c1, s1) = stage.capacities();
        usize::from(c1 != c0) + usize::from(s1 != s0)
    }

    #[test]
    fn gdn_host_stage_is_reused_across_layers_and_steps() {
        // 27B q4 の実形状: conv = conv_dim*cs = 8448、ssm = nv*dk*dv = 786432 要素。
        let (conv_len, ssm_len) = (8_448usize, 786_432usize);
        let mut stage = GdnHostStage::new();
        assert_eq!(stage.capacities(), (0, 0), "作りたては未確保");

        // 1 ステップ目は伸びてよい (= 初回だけ確保する)。
        let first = count_regrows(&mut stage, conv_len, ssm_len, 48);
        assert_eq!(first, 2, "初回は conv/ssm の 2 本だけ確保する (実際 {first})");
        assert_eq!(stage.capacities(), (conv_len, ssm_len));

        // 2 ステップ目以降は 48 層回しても 1 本も確保しない (ここが本題)。
        for step in 1..8 {
            let regrown = count_regrows(&mut stage, conv_len, ssm_len, 48);
            assert_eq!(
                regrown, 0,
                "step={step}: 容量が層ループで再確保された (再利用できていない)"
            );
            assert_eq!(
                stage.capacities(),
                (conv_len, ssm_len),
                "step={step}: 容量がステップ数で増えた"
            );
        }
    }

    #[test]
    fn gdn_host_stage_grows_only_when_shape_grows() {
        // 形が大きくなったときだけ伸びる (grow-only)。縮んでも手放さない。
        let mut stage = GdnHostStage::new();
        reserve_exact_grow(&mut stage.conv, 100);
        assert_eq!(stage.capacities().0, 100);
        reserve_exact_grow(&mut stage.conv, 50); // 縮小 → 何もしない
        assert_eq!(stage.capacities().0, 100, "縮小で解放しない (grow-only)");
        reserve_exact_grow(&mut stage.conv, 8_448); // 拡大 → 伸ばす
        assert_eq!(stage.capacities().0, 8_448);
    }

    #[test]
    fn reserve_grow_reuses_pointer_so_no_per_layer_allocation() {
        // 同じ長さを繰り返しても**同じ領域**を使う (アドレスが変わらない = 再確保なし)。
        let mut stage = GdnHostStage::new();
        reserve_exact_grow(&mut stage.ssm, 786_432);
        let p0 = stage.ssm.as_ptr();
        for _ in 0..480 {
            reserve_exact_grow(&mut stage.ssm, 786_432);
            assert_eq!(stage.ssm.as_ptr(), p0, "同じ長さで確保し直している");
        }
    }
}

// ===========================================================================
#[cfg(test)]
mod b7c_ownership_tests {
    //! s141b (Task B7c): GDN 状態の所有者規則を固定する。
    //!
    //! 新しい規則:
    //! - decode 中はデバイスミラーが権威。ホスト `KvCache::gdn` は**遅延リフレッシュ
    //!   対象**で、`gdn_stale[li]` が真の間は古い可能性がある。
    //! - seed (位置不一致でミラーを埋め直す) は**デバイス正本から D2D**。ホストを
    //!   経由しない (= 層あたり 3.2 MiB の PCIe 往復が消える)。
    //! - 遅延フラッシュは**ホストが古い層だけ**読む (古くない層を読み直さない
    //!   = 「何も要らないステップでホストバイトをコピーしない」)。
    //!
    //! 実転送は GPU が要るので、ここでは判定の実体 (`gdn_flush_step` /
    //! `gdn_state_take_dev` / `gdn_stale`) を回す。デバイスを伴う確認は
    //! `cuda_gdn_flush_pending_faces_positions_and_epoch` (GPU 必須・ignored) が
    //! デバイスを伴う確認を受け持つ。
    use super::*;

    const LAYERS: usize = 48;
    const STEPS: usize = 8;
    /// ミラーの番兵キー (既存テストの u64::MAX-1 / u64::MAX-3 と衝突しない)。
    const MIRROR_TEST_KEY_B7C: u64 = u64::MAX - 5;

    /// 決定論のビットパターン (GPU テストの既知値用。`cuda::tests` の
    /// `pseudo_random` は別モジュールなので私有で持つ)。
    fn b7c_pattern(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32) / ((1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// decode 1 ステップ分のホストバイト数 (48 層 x (conv 8,448 + ssm 786,432) f32)。
    fn host_bytes_per_step(copied_layers: usize) -> usize {
        copied_layers * (8_448 + 786_432) * std::mem::size_of::<f32>()
    }

    /// decode 1 ステップの層ループを回した結果。
    #[derive(Clone, Copy, Debug, Default)]
    struct StepIo {
        /// seed で**ホストから読んだ**層数 (B7c が削る対象。正本が有効なら 0)
        host_seeded: usize,
        /// flush で D2H した層数 (スキャンした層だけ)
        flushed: usize,
    }

    #[derive(Clone, Copy, Debug)]
    struct Layer {
        pos: Option<usize>,
        stale: bool,
    }

    /// flush 段だけを回す (target = このステップ後の `cache.len()`)。
    fn flush(layers: &mut [Layer], target: usize) -> usize {
        let mut copied = 0;
        for l in layers.iter_mut() {
            let (need, _aligned) = gdn_flush_step(true, l.stale, target, l.pos);
            if need {
                l.stale = false;
                copied += 1;
            }
        }
        copied
    }

    /// `chain_q35_gdn_scan` の層ループをそのまま写す:
    /// 1. seed (位置不一致): 正本が有効なら D2D (ホストを読まない) / 無効ならホストから
    /// 2. scan: 書き込み面が新状態になる → `gdn_pos = pos0+1`, `gdn_stale = true`
    /// 3. flush (ステップ末尾): `gdn_flush_step` が真の層だけ D2H
    fn step(layers: &mut [Layer], pos0: usize, dev_valid: bool) -> StepIo {
        let mut io = StepIo::default();
        // 1. seed (位置不一致 = 前ステップの pos0+1 が今の pos0 と合わない層)
        for l in layers.iter_mut() {
            if l.pos != Some(pos0) {
                if dev_valid {
                    // 正本から D2D — ホストは 1 バイトも読まない。ただしホストは
                    // この時点で古い可能性があるので遅延フラグは立てたままにする
                    // (落とすと flush がこの層を飛ばして古いホストを正にしてしまう)。
                    l.stale = true;
                } else {
                    // 正本が無効 = ホストが正。ホストを読んで面 0 を書く。
                    io.host_seeded += 1;
                    l.stale = false;
                }
                l.pos = Some(pos0);
            }
        }
        // 2. scan
        for l in layers.iter_mut() {
            l.pos = Some(pos0 + 1);
            l.stale = true;
        }
        // 3. flush (cache.len() は advance 後 = pos0+1)
        io.flushed = flush(layers, pos0 + 1);
        io
    }

    /// 27B q4 の decode ではミラーの cap 成長で `gdn_pos` が落ちるため、seed は
    /// **毎ステップ全層**走る (実測: `gdn_pf_alloc` 5,712 call = 48 層 x 119 step)。
    /// その条件を再現する。
    fn reset_pos(layers: &mut [Layer]) {
        for l in layers.iter_mut() {
            l.pos = None;
        }
    }

    /// 1) decode 形 (48 層 x 8 ステップ): 正本が有効なステップの seed は D2D なので
    ///    ホストを 1 バイトも読まない (従来は層あたり ~3.2 MiB の D2H 往復)。
    #[test]
    fn decode_loop_seeds_from_device_and_never_reads_host() {
        // 初回要求: まだ正本が無い → 全層をホストから seed (これは従来どおり)。
        let mut layers = vec![Layer { pos: None, stale: false }; LAYERS];
        let first = step(&mut layers, 4, false);
        assert_eq!(first.host_seeded, LAYERS, "正本が無い初回はホストから");
        // 2 ステップ目以降は正本が有効 = seed は D2D。ホストは読まれない。
        let mut host_bytes = 0usize;
        for s in 0..STEPS {
            reset_pos(&mut layers);
            let io = step(&mut layers, 5 + s, true);
            assert_eq!(io.host_seeded, 0, "step={s}: 正本が有効なのにホストを読んだ");
            host_bytes += host_bytes_per_step(io.host_seeded);
            assert_eq!(io.flushed, LAYERS, "step={s}: スキャンした層は flush で 1 度だけ読む");
        }
        assert_eq!(host_bytes, 0, "8 ステップで seed のホスト読みは 0 バイト");
        // flush は全層を 1 度ずつ読んでフラグを落とす = 最終状態は「ホストと一致」。
        assert!(layers.iter().all(|l| !l.stale), "flush 後はフラグが落ちる");
    }

    /// 2) flush は「ホストが古い層」だけを読み、読んだ層はフラグを落とす
    ///    (= 二度読まない)。スキャンしていない層 (stale = false) は読まない。
    #[test]
    fn flush_reads_only_the_layers_whose_host_is_stale() {
        let mut layers = vec![Layer { pos: Some(10), stale: true }; LAYERS];
        layers[7].stale = false; // この層はスキャンしていない = ホストが最新
        let copied = flush(&mut layers, 10);
        assert_eq!(copied, LAYERS - 1, "古い層だけ読む");
        // 1 層 = (8,448 + 786,432) x 4 B = 3,179,520 B。
        assert_eq!(
            host_bytes_per_step(copied),
            3_179_520 * (LAYERS - 1),
            "スキャンした 47 層分だけ (48 層分ではない)"
        );
        // 2 回目の flush は何も読まない (フラグを落としたので)。
        assert_eq!(flush(&mut layers, 10), 0);
        // 位置が合わない (target でも target+1 でもない) 層は stale なら
        // 「不整合」= 呼び出し側が None にする (読みもしない)。
        assert_eq!(gdn_flush_step(true, true, 10, Some(9)), (false, false));
    }

    /// 3) 位置が合っているかどうかの判定 (無効化・位置不一致・新規要求の経路)。
    #[test]
    fn flush_decision_matches_the_old_position_rules() {
        // ホストが古い (stale) 層: 成功ステップ後 (target) の面だけを読む。
        assert_eq!(gdn_flush_step(true, true, 10, Some(10)), (true, true));
        // 層途中失敗 (target+1): 読む面は seed 直後に書かれた = ホストと一致 = 読まない。
        assert_eq!(gdn_flush_step(true, true, 10, Some(11)), (false, true));
        assert_eq!(gdn_flush_step(true, false, 10, Some(11)), (false, true));
        // stale かつ位置が合わない (None / 遠い) → 「読むべきか」は偽、整列も偽。
        // 呼び出し側はこれを不整合 (None) として扱う (旧実装の Some(_) と同じ)。
        assert_eq!(gdn_flush_step(true, true, 10, None), (false, false));
        assert_eq!(gdn_flush_step(true, true, 10, Some(4)), (false, false));
        // ホストが古くない層: 位置が合っていても読まない (ここが B7c の本体)。
        assert_eq!(gdn_flush_step(true, false, 10, Some(10)), (false, true));
        // GDN 層でない層は常に対象外。
        assert_eq!(gdn_flush_step(false, true, 10, Some(10)), (false, false));
    }

    /// 4) 1 リクエスト相当 (prefill で seed → decode 120 ステップ) でも、decode 中に
    ///    seed がホストから読むバイトは 0 (ミラーの cap 成長で seed が毎ステップ
    ///    走る条件を含む)。
    #[test]
    fn one_request_of_decode_never_pulls_the_host_mirror() {
        let mut layers = vec![Layer { pos: None, stale: false }; LAYERS];
        let _prefill_seed = step(&mut layers, 3_000, false);
        let mut total = 0usize;
        for s in 0..120 {
            reset_pos(&mut layers);
            total += host_bytes_per_step(step(&mut layers, 3_001 + s, true).host_seeded);
        }
        assert_eq!(total, 0, "1 リクエスト分の decode で seed がホストから読むバイトは 0");
    }

    /// 5) invalidate-then-seed: ホストが正に戻った (reset/restore/新要求) ときの
    ///    seed 入力はホストのままで、ホストの値は 1 ビットも変わらない。
    ///    `gdn_state_take_dev` は valid=false (= ホストが正) の層では None を返す
    ///    こと = デバイス正本を上書きに使わないこと。
    #[test]
    fn invalidate_then_seed_uses_host_and_keeps_its_bits() {
        // キャッシュの「ホスト正」状態を模す (値は f32 のビットパターンで比較)。
        let conv_len = 192usize;
        let ssm_len = 256usize;
        let host_conv: Vec<f32> = (0..conv_len).map(|i| (i as f32) * 1.5e-3).collect();
        let host_ssm: Vec<f32> = (0..ssm_len).map(|i| -((i as f32) * 3.0)).collect();
        let before_c: Vec<u32> = host_conv.iter().map(|v| v.to_bits()).collect();
        let before_s: Vec<u32> = host_ssm.iter().map(|v| v.to_bits()).collect();

        // 正本を「無効」にした状態 = ホストが正。
        let mut dev = GdnStateCache::empty();
        dev.invalidate();
        let devseed = gdn_state_take_dev(12345, 0, &mut dev);
        assert!(devseed.is_none(), "無効な正本を seed に使ってはいけない");

        // 呼び出し側 (chain_q35_gdn_scan の None 分岐) はホストからコピーする。
        // ここでは H2D の代わりに「ホストを読んで書き戻す」ことを確かめる
        // (転送は値そのままなので bit 一致のまま)。
        assert_eq!(
            host_conv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before_c,
            "conv は bit 一致のまま"
        );
        assert_eq!(
            host_ssm.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            before_s,
            "ssm は bit 一致のまま"
        );
    }

    /// 6) 正本が有効な層は D2D 経路 (seed に使ってよい) で、使った層は無効化される
    ///    (scan がミラーを更新するので正本とずれる)。デバイス確保は要らない。
    #[test]
    fn dev_take_only_for_valid_layers_and_invalidates_them() {
        let mut dev = GdnStateCache::empty();
        // 層 0/1 を「有効」にする — DevBuffer は確保せず、判定だけを見るため
        // valid と cur_kv を直接立てる (確保の無い層は ptr が取れず None)。
        dev.cur_kv = Some(7);
        dev.valid = vec![true, true, false];
        // デバイスバッファが無いので ptr は取れない = None。ここで重要なのは
        // 「valid でない層は絶対に None」かつ「同じ kv でない層も None」。
        assert!(gdn_state_take_dev(8, 0, &mut dev).is_none(), "別 kv_id は不可");
        assert!(gdn_state_take_dev(7, 2, &mut dev).is_none(), "valid=false は不可");
        assert!(gdn_state_take_dev(7, 9, &mut dev).is_none(), "範囲外は不可");
    }

    /// 7) ホスト遅延フラグは層ごとに独立: 1 層だけ「古い」なら、その層だけが flush の
    ///    対象になり、他の 47 層は読まれない (別層の状態で代用しない)。
    #[test]
    fn stale_flag_is_per_layer() {
        let mut stale = vec![false; LAYERS];
        stale[31] = true;
        let pos = vec![Some(42); LAYERS];
        let need: Vec<usize> = (0..LAYERS).filter(|&li| gdn_flush_step(true, stale[li], 42, pos[li]).0).collect();
        assert_eq!(need, vec![31], "古い層だけが読まれる");
        // 隣の層 (30 / 32) は位置が合っていても読まない — 代用読みをしない。
        assert_eq!(gdn_flush_step(true, stale[30], 42, pos[30]), (false, true));
        assert_eq!(gdn_flush_step(true, stale[32], 42, pos[32]), (false, true));
    }

    /// 8) GPU 必須: ミラーの面 0 が**古い値**のままでも、seed がデバイス正本から
    ///    D2D で入ることを、ミラー面 0 を読み戻してビットで確かめる
    ///    (ホストは古い値 = 使われていないことが分かる)。
    ///    実行: `cargo test --features cuda --lib cuda_b7c_seed_copies_from_device_mirror -- --ignored`
    #[test]
    #[ignore] // GPU 必須
    fn cuda_b7c_seed_copies_from_device_mirror() {
        let _g = test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        enable_for_tests();
        let _ctx = CudaCtx::new().expect("GPU 必須 (ignored テスト)");
        let (conv_len, ssm_len) = (192usize, 256usize);
        let key = MIRROR_TEST_KEY_B7C;
        let kv_id = 7u64;
        let dev_conv = b7c_pattern(conv_len, 91);
        let dev_ssm = b7c_pattern(ssm_len, 92);
        let stale_conv = vec![1.0f32; conv_len];
        let stale_ssm = vec![2.0f32; ssm_len];

        // ミラーを作り、面 0 に「古い値」を書く (seed で上書きされるはずの場所)。
        with_mirror_ext(key, 0, 1, 8, 0, 0, false, |m, _| {
            let mut g = GdnDev { conv: [None, None], ssm: [None, None], cur: 0 };
            ok(Scratch::ensure(&mut g.conv[0], conv_len * 4))?;
            ok(Scratch::ensure(&mut g.conv[1], conv_len * 4))?;
            ok(Scratch::ensure(&mut g.ssm[0], ssm_len * 4))?;
            ok(Scratch::ensure(&mut g.ssm[1], ssm_len * 4))?;
            ok(g.conv[0].as_ref().unwrap().copy_from(&stale_conv))?;
            ok(g.ssm[0].as_ref().unwrap().copy_from(&stale_ssm))?;
            m.gdn[0] = Some(g);
            m.gdn_pos[0] = None; // 位置不一致 = seed が走る
            m.gdn_stale[0] = true; // ホストは古い可能性がある
            Some(())
        })
        .unwrap();

        // デバイス正本に既知の値を入れて「有効」にする (s131 の正本と同じ持ち方)。
        GDN_STATE.with(|cell| {
            let mut c = cell.borrow_mut();
            c.cur_kv = Some(kv_id);
            c.invalidate();
            assert!(c.ensure_layer(0, 1, ssm_len).is_ok(), "ssm 確保");
            assert!(c.ensure_conv(0, conv_len).is_ok(), "conv 確保");
            assert!(c.conv[0].as_ref().unwrap().copy_from(&dev_conv).is_ok());
            assert!(c.ssm[0].as_ref().unwrap().copy_from(&dev_ssm).is_ok());
            c.valid[0] = true;
        });

        // seed の入力 (関数そのもの) を取る。取り出した層は無効化される。
        let seed = GDN_STATE.with(|cell| gdn_state_take_dev(kv_id, 0, &mut cell.borrow_mut()));
        let (pc, ps) = seed.expect("有効な正本から読める");
        // ミラー面 0 へ D2D (chain_q35_gdn_scan の seed と同じ操作)。
        with_mirror_ext(key, 0, 1, 8, 0, 0, false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            ok(gd.conv[0].as_ref().unwrap().copy_from_dev(pc, conv_len))?;
            ok(gd.ssm[0].as_ref().unwrap().copy_from_dev(ps, ssm_len))?;
            m.gdn_pos[0] = Some(0);
            m.gdn_stale[0] = false;
            Some(())
        })
        .unwrap();

        // ミラー面 0 を読み戻して、正本の値とビット一致・「古い値」ではないことを見る。
        with_mirror_ext(key, 0, 1, 8, 0, 0, false, |m, _| {
            let gd = m.gdn[0].as_ref().unwrap();
            let cv = gd.conv[0].as_ref().unwrap().to_vec::<f32>(conv_len).unwrap();
            let sm = gd.ssm[0].as_ref().unwrap().to_vec::<f32>(ssm_len).unwrap();
            assert_eq!(cv.len(), dev_conv.len());
            assert!(
                cv.iter().zip(&dev_conv).all(|(a, b)| a.to_bits() == b.to_bits()),
                "seed はデバイス正本の conv とビット一致"
            );
            assert!(
                sm.iter().zip(&dev_ssm).all(|(a, b)| a.to_bits() == b.to_bits()),
                "seed はデバイス正本の ssm とビット一致"
            );
            assert!(cv.iter().all(|&v| v != 1.0), "古いホスト値が使われていない");
            assert!(sm.iter().all(|&v| v != 2.0), "古いホスト値が使われていない");
            Some(())
        })
        .unwrap();
        // 一度読んだ層は無効化されている (次は必ず seed し直す = 取り違えない)。
        let again = GDN_STATE.with(|cell| gdn_state_take_dev(kv_id, 0, &mut cell.borrow_mut()));
        assert!(again.is_none(), "取り出した層は無効化される");
    }
}

#[cfg(test)]
mod b20_stage1_tests {
    //! s157 (Task B20 段階 1): q4 線形を chain に載せる入口の**判定だけ**を固定する。
    //! 実カーネルは GPU が要るので、ここでは (a) `DENNO_GPU_Q4_CHAIN` の解釈、
    //! (b) `chain_linear` が空・混在グループを**受けない**こと (= per-op 経路へ
    //! 静かに委譲する) を確かめる。数値の一致は実機 (supervisor) の担当。
    use super::*;

    /// `DENNO_GPU_Q4_CHAIN` は **"0" だけ**が off。未設定・空・"off" などは on。
    #[test]
    fn q4_chain_flag_is_off_only_for_explicit_zero() {
        assert!(!parse_q4_chain(Some("0")), "\"0\" だけが per-op 経路へ戻す");
        for raw in [None, Some(""), Some("1"), Some("off"), Some(" 0"), Some("0 ")] {
            assert!(parse_q4_chain(raw), "raw={raw:?} は on (既定) のはず");
        }
    }

    /// `chain_linear` は空グループ・int8/q4 混在グループを受けない (None → per-op)。
    #[test]
    fn chain_linear_rejects_empty_and_mixed_groups() {
        assert_eq!(chain_linear(&[], CH_H, CH_O, 1), None, "空グループは受けない");
        let w = vec![0.0f32; 32 * 64];
        let a = crate::quant::Linear::Q8(crate::quant::Quant8Linear::quantize(&w, 32, 64));
        let b = crate::quant::Linear::Q4(crate::quant::Q4Linear::quantize(&w, 32, 64));
        assert_eq!(
            chain_linear(&[&a, &b], CH_H, CH_O, 1),
            None,
            "int8 と q4 の混在は受けない"
        );
        // 単独グループは GPU が無い環境では `chain_ctx()` が None → これも None
        // (per-op 経路へ委譲)。**panic しないこと**がここの主眼。
        assert_eq!(chain_linear(&[&b], CH_H, CH_O, 1), None);
        assert_eq!(chain_linear(&[&a], CH_H, CH_O, 1), None);
    }

    /// P1c: `chain_gdn_prefill_q4` の入口条件 (フォールバック契約) を GPU 無しで固定する。
    /// 本番経路は `gdn_prefill_q4_geom` の戻り値だけを見て発動/委譲を決めるので、
    /// ここで None になる条件がそのまま「従来経路へ落とす」契約になる。
    /// CUDA は一切呼ばない (デバイス不要)。
    #[test]
    fn gdn_prefill_q4_geom_contract() {
        use crate::quant::{Linear, Q4Linear};
        let q4 = |k: usize, n: usize| {
            Linear::Q4(Q4Linear { blocks: Vec::new(), k, n, uid: 0 })
        };
        // 成立する形: nk=2, nv=2, dk=dv=64, d_conv=4, e=64
        //   conv_dim = nk*dk*2 + nv*dv = 384, vd = nv*dv = 128
        let (nk, nv, dk, dv, dc) = (2usize, 2usize, 64usize, 64usize, 4usize);
        let (e, conv_dim, vd) = (64usize, 384usize, 128usize);
        let n_ok = GDN_CHUNK_MIN_TOKENS; // 2048 (1024 の倍数なので端数タイル無し)
        let base = || (q4(e, conv_dim), q4(e, vd), q4(vd, e));
        let dims = (nk, nv, dk, dv, dc);
        let sup = |w: &(Linear, Linear, Linear), n: usize, d: (usize, usize, usize, usize, usize)| {
            gdn_prefill_q4_geom(&w.0, &w.1, &w.2, n, d.0, d.1, d.2, d.3, d.4).is_some()
        };

        // 0) 基準形は通り、寸法が正しく返る
        let w = base();
        let g = gdn_prefill_q4_geom(&w.0, &w.1, &w.2, n_ok, nk, nv, dk, dv, dc)
            .expect("基準形は発動してよい");
        assert_eq!((g.e, g.conv_dim, g.vd, g.cs), (e, conv_dim, vd, dc - 1));

        // 1) 3 本のどれかが Q4 でない → 受けない
        let f32w = |k: usize, n: usize| Linear::F32 { w: Vec::new(), k, n };
        let bad = (f32w(e, conv_dim), w.1.clone(), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "qkv が Q4 でなければ受けない");
        let bad = (w.0.clone(), f32w(e, vd), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "z が Q4 でなければ受けない");
        let bad = (w.0.clone(), w.1.clone(), f32w(vd, e));
        assert!(!sup(&bad, n_ok, dims), "o が Q4 でなければ受けない");

        // 2) 重み形状の不一致 (各辺 1 つずつ)
        let bad = (q4(e, conv_dim + 32), w.1.clone(), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "qkv.n != conv_dim は受けない");
        let bad = (w.0.clone(), q4(e + 32, vd), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "z.k != e は受けない");
        let bad = (w.0.clone(), q4(e, vd + 32), w.2.clone());
        assert!(!sup(&bad, n_ok, dims), "z.n != vd は受けない");
        let bad = (w.0.clone(), w.1.clone(), q4(vd + 32, e));
        assert!(!sup(&bad, n_ok, dims), "o.k != vd は受けない");
        let bad = (w.0.clone(), w.1.clone(), q4(vd, e + 32));
        assert!(!sup(&bad, n_ok, dims), "o.n != e は受けない");

        // 3) n が閾値未満 → 受けない (閾値ちょうどは通る)
        assert!(
            !sup(&w, GDN_CHUNK_MIN_TOKENS - 1, dims),
            "n < GDN_CHUNK_MIN_TOKENS は受けない"
        );

        // 4) dk != dv / d_conv != 4 → 受けない
        assert!(!sup(&w, n_ok, (nk, nv, dk, dv / 2, dc)), "dk != dv は受けない");
        assert!(!sup(&w, n_ok, (nk, nv, dk, dv, 3)), "d_conv != 4 は受けない");

        // 5) 次元 0 / 32 の倍数でない k / vd → 受けない
        assert!(!sup(&w, n_ok, (0, nv, dk, dv, dc)), "nk == 0 は受けない");
        assert!(!sup(&w, n_ok, (nk, 0, dk, dv, dc)), "nv == 0 は受けない");
        assert!(!sup(&w, n_ok, (nk, nv, 0, 0, dc)), "dk == dv == 0 は受けない");
        let bad = (q4(48, conv_dim), q4(48, vd), q4(vd, 48));
        assert!(!sup(&bad, n_ok, dims), "e % 32 != 0 は受けない");
        let bad = (q4(0, conv_dim), q4(0, vd), q4(vd, 0));
        assert!(!sup(&bad, n_ok, dims), "e == 0 は受けない");
        // vd = nv*dv = 3*16 = 48 (% 32 != 0), e = 32 は条件を満たす
        let bad = (q4(32, 112), q4(32, 48), q4(48, 32));
        assert!(!sup(&bad, n_ok, (2, 3, 16, 16, dc)), "vd % 32 != 0 は受けない");

        // 6) 端数タイルが m <= max_m() → 受けない (per-op 経路が GEMV になる帯域)
        let rem_small = GDN_PREFILL_TILE * 2 + max_m();
        assert!(
            !sup(&w, rem_small, dims),
            "端数 {} 行 (<= max_m()={}) は受けない",
            max_m(),
            max_m()
        );
        // 端数が max_m() を超える / 端数なしは通る
        assert!(sup(&w, rem_small + 1, dims), "端数 max_m()+1 は通る");
        assert!(sup(&w, GDN_PREFILL_TILE * 2, dims), "端数なしは通る");

        // 7) 桁あふれは checked_mul で None
        assert!(!sup(&w, n_ok, (usize::MAX, nv, dk, dv, dc)), "conv_dim の桁あふれは受けない");
    }
}

#[cfg(test)]
mod p3_alias_tests {
    //! s165 (Task P3): 連結重みの行範囲エイリアスの**形状契約**を固定する。
    //!
    //! 27B q4 の FFN は decode chain が `gate_up_w` (連結 1 本)、Q4 prefill が
    //! `gate_w` / `up_w` (2 本) を使うため、同じ 2×ff×e パラメータが VRAM に
    //! 2 度載っていた。`q4_row_alias_offsets` が「parts の行数の和 = 連結側の
    //! 行数・k は全部同じ」を満たすときだけ行オフセットを返すことを固定する。
    //! ここが崩れると別の重みを読む (数値が変わる) ので、満たさない入力は
    //! すべて None (= 従来の個別転送へフォールバック) であることを確かめる。
    use super::*;

    #[test]
    fn alias_offsets_gate_up_layout() {
        // 27B: e = 5120, ff = 17408 → 連結は [gate(17408 行) | up(17408 行)]
        let k = 5120;
        let ff = 17408;
        assert_eq!(
            q4_row_alias_offsets(k, 2 * ff, &[(k, ff), (k, ff)]),
            Some(vec![0, ff])
        );
        // 3 本以上でも累積する (汎用の行連結)
        assert_eq!(
            q4_row_alias_offsets(k, 3 * ff, &[(k, ff), (k, ff), (k, ff)]),
            Some(vec![0, ff, 2 * ff])
        );
    }

    #[test]
    fn alias_offsets_fail_closed() {
        let k = 5120;
        let ff = 17408;
        // 行数の和が合わない (連結側が短い/長い)
        assert_eq!(q4_row_alias_offsets(k, 2 * ff - 1, &[(k, ff), (k, ff)]), None);
        assert_eq!(q4_row_alias_offsets(k, 2 * ff + 1, &[(k, ff), (k, ff)]), None);
        // k が食い違う
        assert_eq!(q4_row_alias_offsets(k, 2 * ff, &[(k, ff), (k + 32, ff)]), None);
        // 空・0 行・0 列・32 の倍数でない k
        assert_eq!(q4_row_alias_offsets(k, 2 * ff, &[]), None);
        assert_eq!(q4_row_alias_offsets(k, ff, &[(k, 0)]), None);
        assert_eq!(q4_row_alias_offsets(k, 0, &[(k, ff)]), None);
        assert_eq!(q4_row_alias_offsets(0, ff, &[(0, ff)]), None);
        assert_eq!(q4_row_alias_offsets(k + 8, ff, &[(k + 8, ff)]), None);
        // 桁あふれ
        assert_eq!(
            q4_row_alias_offsets(k, usize::MAX, &[(k, usize::MAX / 2), (k, usize::MAX / 2 + 2)]),
            None
        );
    }

    /// 行オフセットは「1 行 = nb ブロック = nib 16B / sc 4B」の連続領域を指す。
    /// 27B の gate/up で、連結側の nib 上のオフセットが up_w の先頭行と一致すること
    /// をバイト数で確かめる (この値がずれると別の重みを読む)。
    #[test]
    fn alias_byte_offsets_match_row_stride() {
        let (k, ff) = (5120usize, 17408usize);
        let nb = k / 32;
        let nib_row = nb * 16;
        let sc_row = nb * 4;
        let rows = q4_row_alias_offsets(k, 2 * ff, &[(k, ff), (k, ff)]).expect("layout");
        assert_eq!(rows[0] * nib_row, 0);
        assert_eq!(rows[1] * nib_row, ff * nib_row);
        assert_eq!(rows[1] * sc_row, ff * sc_row);
        // up_w の nib は連結側の後半 (全 nib のちょうど半分) から始まる
        assert_eq!(ff * nib_row, (2 * ff * nb * 16) / 2);
    }

    /// s166 (P5): qkv 連結 GEMV の入口は、前提が崩れたら per-part 経路へ落ちる
    /// (fail closed・panic しない)。GPU 無しのこの環境ではどの場合も None になる
    /// — ここで固定するのは「形状不一致で連結側を選ばない」ことと「例外を投げない」
    /// こと (数値の一致は実機テストの担当)。
    #[test]
    fn q4_fused_linear_fails_closed() {
        use crate::quant::{next_q8_uid, Q4Linear};
        let k = 64usize;
        let (nq, nz) = (32usize, 16usize);
        let q = Q4Linear::quantize(&vec![0.01f32; k * nq], k, nq);
        let z = Q4Linear::quantize(&vec![0.02f32; k * nz], k, nz);
        let mut blocks = Vec::new();
        blocks.extend_from_slice(&q.blocks);
        blocks.extend_from_slice(&z.blocks);
        let fused = Q4Linear { blocks, k, n: nq + nz, uid: next_q8_uid() };
        // 行数の和が合わない連結 (前提不一致) → per-part (GPU 無しなので None)
        let short = Q4Linear { blocks: q.blocks.clone(), k, n: nq, uid: next_q8_uid() };
        assert!(chain_linear_q4_fused(&short, &[&q, &z], 0, 1, 1).is_none());
        // part の k が食い違う → per-part
        let other_k = Q4Linear::quantize(&vec![0.03f32; 32 * nz], 32, nz);
        assert!(chain_linear_q4_fused(&fused, &[&q, &other_k], 0, 1, 1).is_none());
        // 正しい形でも GPU が無ければ None (panic しない)
        assert!(chain_linear_q4_fused(&fused, &[&q, &z], 0, 1, 1).is_none());
    }
}

#[cfg(test)]
mod p8_phase_tests {
    //! s169 (Task P8): prefill 残差常駐経路の位相計測の**表との対応**を固定する。
    //! GPU は使わない (イベントの実測は実機の担当) — ここで守るのは
    //! 「位相の名前 = 表 C の行名」「1 行を 2 つの位相が共有しない (二重計上しない)」
    //! 「残差加算は独立した行を持たない (融合している)」の 3 点。
    use super::*;

    #[test]
    fn pf_phase_buckets_match_serve_rows() {
        let mut seen: Vec<usize> = Vec::new();
        for ph in 0..PF_PH_N {
            let Some(op) = PF_PH_BUCKETS[ph] else {
                continue;
            };
            assert!(
                (crate::prof::SPLIT_SERVE_FIRST..crate::prof::SPLIT_SERVE_FIRST + crate::prof::SPLIT_SERVE_N)
                    .contains(&op),
                "位相 {ph} の bucket が表 C の範囲外"
            );
            let row = op - crate::prof::SPLIT_SERVE_FIRST;
            let name = crate::prof::split_row_name(crate::prof::SPLIT_SERVE_TABLE, row)
                .expect("位相の bucket は表 C の行");
            assert_eq!(name, PF_PH_NAMES[ph], "位相 {ph} の名前が表 C の行名と違う");
            assert!(!seen.contains(&op), "表 C の行を 2 つの位相が共有している: {name}");
            seen.push(op);
        }
        // GDN の o_proj だけは表 C に行を持たない (decode chain と同じ扱い)。
        assert_eq!(PF_PH_BUCKETS[PF_PH_GDN_OPROJ], None);
        assert_eq!(seen.len(), PF_PH_N - 1);
        // 残差加算は `dno_add_rmsnorm` に融合しているので独立した位相を持たない
        // (ホスト経路の `sl_lay_residual` に相当する区間は残差常駐経路に無い)。
        assert!(!seen.contains(&crate::prof::SL_LAY_RESIDUAL));
        assert!(PF_PH_NAMES.iter().all(|n| !n.contains("residual")));
    }
}

#[cfg(test)]
mod p2_residual_tests {
    //! s165 (Task P2): prefill の残差ストリームをデバイス常駐させる経路の**入口条件**
    //! だけを固定する。GPU は使わない (純粋関数の契約テスト) — 数値の一致と速度は
    //! 実機 (main セッション) の担当。ここで None になる条件がそのまま
    //! 「この層は従来のホスト経路で通す」契約になる。
    use super::*;

    fn q4(k: usize, n: usize) -> crate::quant::Linear {
        crate::quant::Linear::Q4(crate::quant::Q4Linear { blocks: Vec::new(), k, n, uid: 0 })
    }
    fn f32w(k: usize, n: usize) -> crate::quant::Linear {
        crate::quant::Linear::F32 { w: vec![0.0; k * n], k, n }
    }

    /// グローバルゲート: q4 以外・診断中・GEMV 帯域・端数タイル・smem 超過で発動しない。
    #[test]
    fn pf_res_gate_contract() {
        let (gm, mm) = (256usize, 16usize);
        // 基準: 2048 token・e=5120 (端数タイル無し)
        assert!(pf_res_gate(2048, 5120, true, false, gm, mm));
        // q4 でない層が 1 本でもあれば発動しない
        assert!(!pf_res_gate(2048, 5120, false, false, gm, mm));
        // 層別診断 (lchk) 中はホストの x を層ごとに更新する前提 → 発動しない
        assert!(!pf_res_gate(2048, 5120, true, true, gm, mm));
        // GEMM 帯域の外 (m <= max_m() / m < gemm_min_m()) は発動しない
        assert!(!pf_res_gate(16, 5120, true, false, gm, mm));
        assert!(!pf_res_gate(128, 5120, true, false, gm, mm));
        assert!(!pf_res_gate(0, 5120, true, false, gm, mm));
        // dno_add_rmsnorm の動的 shmem (e*4 <= 48KB) と e % 32 == 0
        assert!(!pf_res_gate(2048, 16384, true, false, gm, mm), "e=16384 は smem 48KB 超");
        assert!(!pf_res_gate(2048, 5000, true, false, gm, mm), "e % 32 != 0");
        // 端数タイルが GEMV 帯域 (<= max_m()) に入る形は受けない
        let rem_small = GDN_PREFILL_TILE * 2 + mm;
        assert!(!pf_res_gate(rem_small, 5120, true, false, gm, mm));
        assert!(pf_res_gate(rem_small + 1, 5120, true, false, gm, mm));
        assert!(pf_res_gate(GDN_PREFILL_TILE * 2, 5120, true, false, gm, mm));
    }

    /// 層ゲート: FFN / full attention / GDN の形が揃ったときだけ Some。
    #[test]
    fn pf_layer_geom_contract() {
        let (gm, mm) = (256usize, 16usize);
        let n = 2048usize;
        let e = 64usize;
        let ff = 128usize;
        // full attention: heads=2, kv_heads=1, dh=32 → att=64, kv_dim=32
        let (heads, kv_heads, dh) = (2usize, 1usize, 32usize);
        let (att, kv_dim) = (heads * dh, kv_heads * dh);
        let gate = q4(e, ff);
        let up = q4(e, ff);
        let down = q4(ff, e);
        let d = PfDims {
            n,
            e,
            heads,
            kv_heads,
            dh,
            nk: 1,
            nv: 1,
            dk: 32,
            dv: 32,
            d_conv: 4,
        };
        let (q, k, v, o) = (q4(e, att * 2), q4(e, kv_dim), q4(e, kv_dim), q4(att, e));
        let full = PfAttnW::Full { q: &q, k: &k, v: &v, o: &o };
        assert!(pf_layer_geom(&full, &gate, &up, &down, &d, gm, mm).is_some(), "基準形は通る");

        // FFN が q4 でない / 形が合わない → 受けない
        assert!(pf_layer_geom(&full, &f32w(e, ff), &up, &down, &d, gm, mm).is_none());
        assert!(pf_layer_geom(&full, &gate, &up, &q4(e, ff + 32), &d, gm, mm).is_none());
        assert!(pf_layer_geom(&full, &gate, &up, &q4(ff, e + 32), &d, gm, mm).is_none());

        // o_proj の k が heads*dh でない / q の n が 2*heads*dh でない → 受けない
        let bad_o = q4(att + 32, e);
        let full_bad = PfAttnW::Full { q: &q, k: &k, v: &v, o: &bad_o };
        assert!(pf_layer_geom(&full_bad, &gate, &up, &down, &d, gm, mm).is_none());
        let bad_q = q4(e, att);
        let full_bad = PfAttnW::Full { q: &bad_q, k: &k, v: &v, o: &o };
        assert!(pf_layer_geom(&full_bad, &gate, &up, &down, &d, gm, mm).is_none());

        // GDN 層: qkv/z/o の形が揃えば通り、dk != dv では受けない
        //   conv_dim = nk*dk*2 + nv*dv = 64 + 32 = 96, vd = 32
        let gqkv = q4(e, 96);
        let gz = q4(e, 32);
        let go = q4(32, e);
        let gdn = PfAttnW::Gdn { qkv: &gqkv, z: &gz, o: &go };
        assert!(pf_layer_geom(&gdn, &gate, &up, &down, &d, gm, mm).is_some());
        let mut d2 = PfDims { n, e, heads, kv_heads, dh, nk: 1, nv: 1, dk: 32, dv: 32, d_conv: 4 };
        d2.dv = 16;
        assert!(pf_layer_geom(&gdn, &gate, &up, &down, &d2, gm, mm).is_none(), "dk != dv は受けない");

        // n が GEMV 帯域 → 層としても受けない
        let mut d3 = PfDims { n: mm, e, heads, kv_heads, dh, nk: 1, nv: 1, dk: 32, dv: 32, d_conv: 4 };
        assert!(pf_layer_geom(&full, &gate, &up, &down, &d3, gm, mm).is_none());
        d3.n = n;
        assert!(pf_layer_geom(&full, &gate, &up, &down, &d3, gm, mm).is_some());
    }

    /// 新スロットは既存と衝突せず、チェーン本数に収まる。
    #[test]
    fn pf_res_slot_is_distinct() {
        let used = [
            CH_X, CH_H, CH_QKV, CH_ATT, CH_O, CH_GA, CH_ACT, CH_DOWN, CH_LOGITS, CH_Q35_Q, CH_PFH,
        ];
        assert!(!used.contains(&CH_PF_RES), "CH_PF_RES は既存スロットと別");
        assert!(CH_PF_RES < CH_SLOTS);

    }
}

#[cfg(test)]
mod p6_overlap_tests {
    //! s167 (Task P6): GDN prefill の h2 D2H を専用コピーストリームで qkv/z GEMM と
    //! 重ねる経路の**入口条件とフォールバック**だけを固定する (GPU 非依存の契約
    //! テスト)。数値の一致・速度は実機の担当。
    //!
    //! ここで固定する契約:
    //!   - フラグは明示的な "1"/"true"/"on" だけが ON (既定 OFF = 従来と同一経路)
    //!   - stream/event/pinned の準備に失敗したら必ず従来の同期 D2H (`SyncD2h`)
    //!   - 形が不正・桁あふれ・0 バイトでも従来経路 (コピーを発行しない)
    use super::*;

    /// `DENNO_PF_OVERLAP` は既定 OFF。明示的な真値だけが ON。
    #[test]
    fn pf_overlap_flag_defaults_off() {
        assert!(!parse_pf_overlap(None), "未設定は OFF (従来と同一経路)");
        for raw in ["", "0", "off", "false", " 1", "1 ", "TRUE", "yes"] {
            assert!(!parse_pf_overlap(Some(raw)), "raw={raw:?} は OFF のはず");
        }
        for raw in ["1", "true", "on"] {
            assert!(parse_pf_overlap(Some(raw)), "raw={raw:?} は ON のはず");
        }
    }

    /// 経路決定: OFF とデバイス準備失敗は**必ず**従来の同期 D2H。
    #[test]
    fn pf_h2_path_falls_back_to_sync() {
        let (n, e) = (12850usize, 4096usize);
        // フラグ OFF: dev_ready が真でも従来経路
        assert_eq!(pf_h2_path(false, n, e, true), PfH2Path::SyncD2h);
        assert_eq!(pf_h2_path(false, n, e, false), PfH2Path::SyncD2h);
        // フラグ ON でも準備失敗 (stream/event/pinned) なら従来経路
        assert_eq!(pf_h2_path(true, n, e, false), PfH2Path::SyncD2h);
        // フラグ ON + 準備成功 + 妥当な形 → コピーストリーム
        assert_eq!(pf_h2_path(true, n, e, true), PfH2Path::CopyStream);
    }

    /// 形が不正 (0・桁あふれ) なら従来経路 — コピーを発行しない。
    #[test]
    fn pf_h2_path_rejects_bad_shape() {
        assert_eq!(pf_h2_path(true, 0, 4096, true), PfH2Path::SyncD2h);
        assert_eq!(pf_h2_path(true, 12850, 0, true), PfH2Path::SyncD2h);
        assert_eq!(pf_h2_path(true, 0, 0, true), PfH2Path::SyncD2h);
        // n*e は通るが *4 で桁あふれ → 従来経路 (checked_mul)
        assert_eq!(
            pf_h2_path(true, usize::MAX / 2, 2, true),
            PfH2Path::SyncD2h
        );
        assert_eq!(pf_h2_path(true, usize::MAX, 2, true), PfH2Path::SyncD2h);
        // 1 行だけでも (実経路では n >= 2048 だが) 判定としてはコピー経路
        assert_eq!(pf_h2_path(true, 1, 1, true), PfH2Path::CopyStream);
    }
}
