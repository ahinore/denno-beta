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
type FnCuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, CUdevice) -> CUresult;
type FnCuFuncSetAttribute = unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult;
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
    memcpy_htod: FnCuMemcpyHtoDV2,
    memcpy_dtoh: FnCuMemcpyDtoHV2,
    mem_host_alloc: FnCuMemHostAlloc,
    mem_host_free: FnCuMemFreeHost,
    memcpy_htod_async: FnCuMemcpyHtoDAsyncV2,
    memcpy_dtoh_async: FnCuMemcpyDtoHAsyncV2,
    memcpy_dtod_async: FnCuMemcpyDtoDAsyncV2,
    // s48 (C1-a): 動的 smem の opt-in (cuFuncSetAttribute)。CUDA 9.0 以降に存在
    // するが、dlsym 失敗でも全体を落とさないよう Option (無ければ 48KB 既定のまま)。
    device_get_attribute: Option<FnCuDeviceGetAttribute>,
    func_set_attribute: Option<FnCuFuncSetAttribute>,
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
        memcpy_htod: std::mem::transmute(sym(handle, "cuMemcpyHtoD_v2")?),
        memcpy_dtoh: std::mem::transmute(sym(handle, "cuMemcpyDtoH_v2")?),
        mem_host_alloc: std::mem::transmute(sym(handle, "cuMemHostAlloc")?),
        mem_host_free: std::mem::transmute(sym(handle, "cuMemFreeHost")?),
        memcpy_htod_async: std::mem::transmute(sym(handle, "cuMemcpyHtoDAsync_v2")?),
        memcpy_dtoh_async: std::mem::transmute(sym(handle, "cuMemcpyDtoHAsync_v2")?),
        memcpy_dtod_async: std::mem::transmute(sym(handle, "cuMemcpyDtoDAsync_v2")?),
        device_get_attribute: sym_opt(handle, "cuDeviceGetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuDeviceGetAttribute>(p) }),
        func_set_attribute: sym_opt(handle, "cuFuncSetAttribute")
            .map(|p| unsafe { std::mem::transmute::<*mut c_void, FnCuFuncSetAttribute>(p) }),
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
}

impl DevBuffer {
    fn alloc(bytes: usize) -> Result<Self> {
        let mut ptr: CUdeviceptr = 0;
        unsafe { check((lib()?.mem_alloc)(&mut ptr, bytes), "cuMemAlloc")? };
        Ok(Self { ptr, bytes })
    }

    /// ホストのスライスをデバイスへコピーして確保
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
        }
        Ok(buf)
    }

    /// デバイス → ホスト (len 個の T として読む)
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
        if self.ptr != 0 {
            unsafe {
                let _ = (lib().unwrap().mem_free)(self.ptr);
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
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        let (dx, daq, dasc, dout) = (
            x.as_ref().unwrap(),
            aq.as_ref().unwrap(),
            asc.as_ref().unwrap(),
            out.as_ref().unwrap(),
        );
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..m * k].copy_from_slice(a);
        unsafe {
            check(
                (lib.memcpy_htod_async)(dx.ptr(), hx.ptr as *const c_void, bytes_x, std::ptr::null_mut()),
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
            let total = (m * n) as u64;
            let blocks = total.div_ceil(8) as u32;
            ctx.launch_async(ctx.f_mv, (blocks, 1, 1), (256, 1, 1), 0, &mut args)?;

            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            // s65: 発行元の件数分類 (表示専用・時間は d2h_state バケット側)
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        ctx.sync()?;
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        Ok(ho.as_slice::<f32>()[..m * n].to_vec())
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
const CH_SLOTS: usize = 11;

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
        }
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

/// スロットからホストへダウンロード (D2H + 同期)。
pub fn chain_download(slot: usize, len: usize) -> Option<Vec<f32>> {
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
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_OTHER);
        ok(ctx.sync())?;
        unsafe { out.set_len(len) };
        Some(out)
    })
}

/// スロット内の要素オフセット `elem_off` から `len` 個の f32 をダウンロード
/// (D2H + 同期)。B1: prefill チェーンの最終 norm 出力 (CH_H の最終行のみ)
/// 取得用 — [n, e] 全行の D2H を避ける。呼び出し時点でスロットには
/// (elem_off + len) 個分のデータが存在することを前提とする (不足時は None)。
pub fn chain_download_at(slot: usize, elem_off: usize, len: usize) -> Option<Vec<f32>> {
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
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_OTHER);
        ok(ctx.sync())?;
        unsafe { out.set_len(len) };
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
        let xptr = s.bufs[CH_X].as_ref().unwrap().ptr();
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

/// s70: prefill の FFN (rmsnorm 済み x → gate/up → swiglu → down → ホスト) を
/// デバイス常駐で通す。活性化の往復が
///   旧: H2D 47MB + D2H 140MB + H2D 47MB + D2H 47MB
///   新: H2D 47MB + D2H 47MB
/// になる (m=2843, e=4096, ff=6144 の場合)。GEMM/swiglu は CPU 経路と
/// ビット一致 (同一カーネル・同一整数累積・同一 f32 演算列)。
pub fn chain_mlp_prefill(
    gate_w: &crate::quant::Quant8Linear,
    up_w: &crate::quant::Quant8Linear,
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
    let ff = gate_w.n;
    if m == 0 || x.len() < m * e || out.len() < m * e {
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
        chain_gemm_grouped_pf(&[gate_w, up_w], CH_PFH, CH_GA, m)?;
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
    // 倍々成長 (ホスト KvCache と同じ方針)。下限 1024 行
    (old_cap.max(1024) * 2).max(t_total.next_power_of_two())
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
        let bytes = newcap * kv_dim * 4;
        for l in m.layers.iter_mut() {
            l.0 = None;
            l.1 = None;
            ok(Scratch::ensure(&mut l.0, bytes))?;
            ok(Scratch::ensure(&mut l.1, bytes))?;
        }
        m.cap = newcap;
        m.len.iter_mut().for_each(|l| *l = 0);
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
#[derive(Clone, Copy)]
struct Q35Fns {
    f_matvec_f32: CUfunction,
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
        }
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
        ok(unsafe {
            ctx.launch_async(fns.f_matvec_f32, (n as u32, 1, 1), (128, 1, 1), 0, &mut args)
        })
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

    // GDN 状態ミラー: 取得/seed (ポインタはロック解放前に抽出)
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
            // seed: この層のミラーが pos0 に同期していなければホスト状態を H2D
            if m.gdn_pos[li] != Some(pos0) {
                let (conv_h, ssm_h) = cache.gdn_state_mut(li)?;
                ok(gd.conv[0].as_ref().unwrap().copy_from(conv_h))?;
                ok(gd.ssm[0].as_ref().unwrap().copy_from(ssm_h))?;
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
        let face = match m.gdn_pos[li] {
            Some(p) if p == target => gd.cur,
            Some(p) if p == target + 1 => 1 - gd.cur,
            Some(_) => {
                // 不整合 (正常系では起こらない — バグのみ)
                m.gdn_pending = None;
                return None;
            }
            None => continue, // chain 未走層 = ホストが最新
        };
        let (Some(cb), Some(sb)) = (&gd.conv[face], &gd.ssm[face]) else {
            m.gdn_pending = None;
            return None;
        };
        // s65: 発行元の件数分類 (GDN 状態の回復 = conv + ssm の 2 本)
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        let conv = ok(cb.to_vec::<f32>(cb.bytes() / 4))?;
        let ssm = ok(sb.to_vec::<f32>(sb.bytes() / 4))?;
        let (conv_h, ssm_h) = cache.gdn_state_mut(li)?;
        if conv_h.len() < conv.len() || ssm_h.len() < ssm.len() {
            m.gdn_pending = None;
            return None;
        }
        conv_h[..conv.len()].copy_from_slice(&conv);
        ssm_h[..ssm.len()].copy_from_slice(&ssm);
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
            let qw: Vec<i8> = (0..k * n).map(|i| ((i * 37) % 251) as i32 as i8 - 125).collect();
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
            let aq: Vec<i8> = (0..k).map(|i| ((i * 11) % 251) as i32 as i8 - 120).collect();
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
            let qw: Vec<i8> = (0..k * n).map(|i| ((i * 29) % 241) as i32 as i8 - 120).collect();
            let aq: Vec<i8> = (0..m * k).map(|i| ((i * 17) % 239) as i32 as i8 - 119).collect();
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

    /// s75 (不具合修正): `kernels_int8.cu` の共有メモリ / ブロック形状の不変条件を
    /// **GPU 無しで** 検査する。s74 の 2 つの欠陥はどちらも CPU テストを素通りして
    /// GPU 実機で初めて露見した (CUDA error 716 / 未書き込み出力) ので、原因を
    /// ソースの静的解析で固定する。
    ///
    ///  1. 共有ステージの行ストライド `GEMM_SP` が 4 の倍数であること
    ///     (s74 は 33 = 奇数で、`*(int*)&sA[r][c]` が行ごとに未アラインになり
    ///      `Invalid __shared__ write` → sticky な error 716 を起こした)。
    ///  2. ブロック形状とタイル寸法が整合すること (s74 は block 16x4 = 64 thread で
    ///     32 行しか覆えず、タイル 128 行のうち 96 行が未書き込みだった)。
    ///  3. 共有配列への i32 アクセスのオフセットがすべて 4 の倍数であること。
    ///
    /// .cu は `include_str!` で取り込むので、値を変えたらこのテストが落ちる。
    #[test]
    fn kernel_source_shared_stride_and_block_shape_are_consistent() {
        const SRC: &str = include_str!("../kernels/kernels_int8.cu");

        // `#define NAME VALUE` を取り出す (VALUE は他 define を参照し得るので
        // 依存順に評価する簡易評価器)。
        fn defines(src: &str) -> std::collections::HashMap<String, String> {
            let mut map = std::collections::HashMap::new();
            for line in src.lines() {
                let line = line.trim_start();
                let Some(rest) = line.strip_prefix("#define ") else { continue };
                let rest = rest.trim();
                let (name, val) = match rest.split_once(char::is_whitespace) {
                    Some((n, v)) => (n, v.trim()),
                    None => continue,
                };
                map.insert(name.to_string(), val.to_string());
            }
            map
        }
        let d = defines(SRC);
        let get_raw = |name: &str| -> &str {
            d.get(name)
                .unwrap_or_else(|| panic!("kernels_int8.cu に #define {name} が無い"))
        };
        // 数値評価: トークン単位で name → 値 に展開してから式を評価する。
        fn eval(expr: &str, d: &std::collections::HashMap<String, String>, depth: u32) -> i64 {
            assert!(depth < 16, "define の循環参照: {expr}");
            let mut expanded = String::new();
            let bytes = expr.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let c = bytes[i] as char;
                if c.is_ascii_alphabetic() || c == '_' {
                    let start = i;
                    while i < bytes.len() {
                        let ch = bytes[i] as char;
                        if ch.is_ascii_alphanumeric() || ch == '_' {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    let ident = &expr[start..i];
                    let raw = d
                        .get(ident)
                        .unwrap_or_else(|| panic!("未定義の識別子 {ident} ({expr})"));
                    expanded.push_str(&format!("({})", eval(raw, d, depth + 1)));
                } else {
                    expanded.push(c);
                    i += 1;
                }
            }
            eval_int_expr(&expanded)
        }
        // 四則演算 + 括弧 + 剰余のみの整数式評価 (テスト用の最小実装)。
        fn eval_int_expr(s: &str) -> i64 {
            let cs: Vec<char> = s.chars().filter(|c| !c.is_whitespace()).collect();
            let mut pos = 0usize;
            fn parse_sum(cs: &[char], pos: &mut usize) -> i64 {
                let mut v = parse_prod(cs, pos);
                while *pos < cs.len() && (cs[*pos] == '+' || cs[*pos] == '-') {
                    let op = cs[*pos];
                    *pos += 1;
                    let r = parse_prod(cs, pos);
                    v = if op == '+' { v + r } else { v - r };
                }
                v
            }
            fn parse_prod(cs: &[char], pos: &mut usize) -> i64 {
                let mut v = parse_atom(cs, pos);
                while *pos < cs.len() && (cs[*pos] == '*' || cs[*pos] == '/' || cs[*pos] == '%') {
                    let op = cs[*pos];
                    *pos += 1;
                    let r = parse_atom(cs, pos);
                    v = match op {
                        '*' => v * r,
                        '/' => v / r,
                        _ => v % r,
                    };
                }
                v
            }
            fn parse_atom(cs: &[char], pos: &mut usize) -> i64 {
                if *pos < cs.len() && cs[*pos] == '(' {
                    *pos += 1;
                    let v = parse_sum(cs, pos);
                    assert_eq!(cs.get(*pos), Some(&')'), "括弧が閉じていない");
                    *pos += 1;
                    return v;
                }
                let start = *pos;
                while *pos < cs.len() && cs[*pos].is_ascii_digit() {
                    *pos += 1;
                }
                assert!(start != *pos, "数値が期待される位置: {cs:?}");
                cs[start..*pos].iter().collect::<String>().parse().unwrap()
            }
            let v = parse_sum(&cs, &mut pos);
            assert_eq!(pos, cs.len(), "式を評価しきれない: {s}");
            v
        }

        // 列添字の評価。`x * 4` や `c` (ローカルエイリアス) のように **変数** を
        // 含む式があるので、カーネル内の宣言を解決して取り得る値ごとに評価する。
        // 取り得る全ての値が 4 の倍数であることを要求する = 任意の反復でアライン。
        //
        // 解決規則 (このカーネルで実際に使われている形だけを扱う):
        //   - `#define` 定数        -> 数値
        //   - `const int c = <式>;` -> <式> を再帰評価
        //   - `for (int v = A; v < B; v++)` / `v += S` -> A..B を S 刻みで評価
        fn eval_col(
            expr: &str,
            d: &std::collections::HashMap<String, String>,
            depth: u32,
        ) -> i64 {
            assert!(depth < 8, "列添字の解決が深すぎる (循環?): {expr}");
            let idents = idents_of(expr);
            let mut resolved = expr.to_string();
            let mut const_val = i64::MAX;
            for ident in &idents {
                if d.contains_key(ident) {
                    continue; // #define は eval 側で展開する
                }
                let (values, is_loop) = var_values(ident, d, depth);
                if !is_loop {
                    // 単一定数: 式へ直接代入する
                    for v in &values {
                        resolved = subst_ident(&resolved, ident, *v);
                    }
                    continue;
                }
                // ループ変数: 全反復で 4 の倍数であることを要求する
                for v in &values {
                    let sub = subst_ident(&resolved, ident, *v);
                    let off = eval(&sub, d, 0);
                    assert_eq!(
                        off % 4,
                        0,
                        "列添字 {expr} が {ident}={v} のとき {off} で 4 の倍数でない"
                    );
                    const_val = off;
                }
                resolved = resolved.replace(ident.as_str(), "0"); // 以後は定数扱い
            }
            let v = eval(&resolved, d, 0);
            if const_val != i64::MAX {
                return const_val;
            }
            v
        }

        // 式に現れる識別子 (出現順・重複なし)
        fn idents_of(expr: &str) -> Vec<String> {
            let b = expr.as_bytes();
            let mut out: Vec<String> = Vec::new();
            let mut i = 0;
            while i < b.len() {
                let c = b[i] as char;
                if c.is_ascii_alphabetic() || c == '_' {
                    let s = i;
                    while i < b.len() {
                        let ch = b[i] as char;
                        if ch.is_ascii_alphanumeric() || ch == '_' {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    let id = expr[s..i].to_string();
                    if !out.contains(&id) {
                        out.push(id);
                    }
                } else {
                    i += 1;
                }
            }
            out
        }
        // 識別子を数値に置換 (識別子境界を保つ)
        fn subst_ident(expr: &str, ident: &str, val: i64) -> String {
            let mut out = String::new();
            let b = expr.as_bytes();
            let mut i = 0;
            while i < b.len() {
                let c = b[i] as char;
                if c.is_ascii_alphabetic() || c == '_' {
                    let s = i;
                    while i < b.len() {
                        let ch = b[i] as char;
                        if ch.is_ascii_alphanumeric() || ch == '_' {
                            i += 1;
                        } else {
                            break;
                        }
                    }
                    if &expr[s..i] == ident {
                        out.push_str(&val.to_string());
                    } else {
                        out.push_str(&expr[s..i]);
                    }
                } else {
                    out.push(c);
                    i += 1;
                }
            }
            out
        }
        // 変数の取り得る値。(values, is_loop)
        //   `const int c = <式>;`          -> (vec![評価値], false)
        //   `for (int v = A; v < B; v++)`  -> ((A..B の各値), true)
        fn var_values(
            var: &str,
            d: &std::collections::HashMap<String, String>,
            depth: u32,
        ) -> (Vec<i64>, bool) {
            // ループ宣言を優先 (同じ名前が両方あることは無い)
            let loop_decl = format!("for (int {var} = ");
            if let Some(pos) = SRC.find(&loop_decl) {
                let tail = &SRC[pos + loop_decl.len()..];
                let semi = tail.find(';').expect("for の ; が無い");
                let from_expr = &tail[..semi];
                let rest = &tail[semi + 1..];
                let semi2 = rest.find(';').expect("for 条件の ; が無い");
                let cond = rest[..semi2].trim();
                let rest2 = &rest[semi2 + 1..];
                let inc = &rest2[..rest2.find(')').expect("for 増分の ) が無い")];
                let from = eval_col(from_expr, d, depth + 1);
                let lt = format!("{var} <");
                let to_expr = cond
                    .strip_prefix(&lt)
                    .unwrap_or_else(|| panic!("for 条件が `{var} <` でない: {cond:?}"));
                let to = eval_col(to_expr.trim(), d, depth + 1);
                let (step, tight) = if inc.trim() == format!("{var}++") {
                    (1, true)
                } else if let Some(r) = inc.trim().strip_prefix(&format!("{var} +=")) {
                    (eval_col(r.trim(), d, depth + 1), false)
                } else {
                    panic!("for 増分が未対応: {inc:?}");
                };
                let _ = tight;
                assert!(step > 0, "for 増分が正でない");
                let mut vals = Vec::new();
                let mut v = from;
                while v < to {
                    vals.push(v);
                    v += step;
                }
                assert!(!vals.is_empty(), "ループ {var} が 1 回も回らない");
                return (vals, true);
            }
            // ローカル定数 `const int c = <式>;` / `int c = <式>;`
            for prefix in ["const int ", "int "] {
                let decl = format!("{prefix}{var} = ");
                if let Some(pos) = SRC.find(&decl) {
                    let tail = &SRC[pos + decl.len()..];
                    let semi = tail.find(';').expect("ローカル宣言の ; が無い");
                    let v = eval_col(&tail[..semi], d, depth + 1);
                    return (vec![v], false);
                }
            }
            panic!("列添字の変数 {var} の宣言が見つからない");
        }

        let tile = eval(get_raw("GEMM_TILE"), &d, 0);
        let tile_n = eval(get_raw("GEMM_TILE_N"), &d, 0);
        let tk = eval(get_raw("GEMM_TK"), &d, 0);
        let sp = eval(get_raw("GEMM_SP"), &d, 0);
        let bx = eval(get_raw("GEMM_BX"), &d, 0);
        let by = eval(get_raw("GEMM_BY"), &d, 0);

        // (1) 共有ステージの行ストライドは 4 の倍数 (i32 キャストの前提)
        assert_eq!(
            sp % 4,
            0,
            "GEMM_SP={sp} が 4 の倍数でない: 共有配列への (int*) アクセスが未アラインになり \
             'Invalid __shared__ write' -> CUDA error 716 を起こす (s74 の 33 がこれ)"
        );
        assert_ne!(
            sp % 32,
            0,
            "GEMM_SP={sp} が 32 の倍数だと隣接 thread の 8 行差が同一 bank に並び conflict する"
        );
        assert_eq!(tk % 4, 0, "GEMM_TK={tk} は dp4a パックのため 4 の倍数であること");

        // (2) ブロック形状はタイルを過不足なく覆う (s74 の 16x4 がこれに違反した)
        assert_eq!(
            bx * 16,
            tile_n,
            "GEMM_BX * (1 thread の列数 16) = {} != GEMM_TILE_N {tile_n}",
            bx * 16
        );
        assert_eq!(
            by * 8,
            tile,
            "GEMM_BY * (1 thread の行数 8) = {} != GEMM_TILE {tile}: \
             ブロックがタイル全域を覆わず、未書き込みの出力行が残る (s74 の不具合)",
            by * 8
        );
        assert!(bx * by <= 1024, "ブロック {} スレッドは上限 1024 を超える", bx * by);

        // Rust 側のミラー定数とカーネルの値が一致すること (片方だけ直す事故の防止)
        assert_eq!(tile as usize, GEMM_TILE, "GEMM_TILE がカーネルと Rust で不一致");
        assert_eq!(tile_n as usize, GEMM_TILE_N, "GEMM_TILE_N がカーネルと Rust で不一致");
        assert_eq!(bx as u32, GEMM_BX, "GEMM_BX がカーネルと Rust で不一致");
        assert_eq!(by as u32, GEMM_BY, "GEMM_BY がカーネルと Rust で不一致");

        // 共有メモリ量が既定の 48KiB に収まること
        let smem = (tile + tile_n) * sp;
        assert!(
            smem <= 48 * 1024,
            "共有メモリ {smem}B が既定 48KiB を超える (動的確保か opt-in が必要)"
        );

        // (3) 共有配列への i32 アクセス: すべて `[行][<4 の倍数>]` であること。
        //     カーネル本体 (dno_gemm_i8) の範囲だけを見る。
        let body_start = SRC
            .find("void dno_gemm_i8")
            .expect("dno_gemm_i8 が無い");
        let body = &SRC[body_start..];
        let mut checked = 0usize;
        for (lineno, line) in body.lines().enumerate() {
            // コメント行は除外
            if line.trim_start().starts_with("//") {
                continue;
            }
            for prefix in ["sA[", "sB["] {
                let mut search = 0usize;
                while let Some(off) = line[search..].find(prefix) {
                    let at = search + off;
                    // 直前に `(int*)` / `(const int*)` のキャストがあるものだけ対象
                    // (`*(int*)&sA[..]` のように `&` が挟まるので取り除いて判定する)
                    let lead = line[..at].trim_end().trim_end_matches('&').trim_end();
                    let cast = lead.ends_with("(int*)") || lead.ends_with("(const int*)");
                    if !cast {
                        search = at + prefix.len();
                        continue;
                    }
                    // `[行][列]` の列部分を取り出す
                    let rest = &line[at..];
                    let close = rest.find(']').expect("配列の [ が閉じていない");
                    let after = &rest[close + 1..];
                    let col_open = after.find('[').expect("列添字が無い");
                    let col_close = after.find(']').expect("列添字が閉じていない");
                    let col_expr = &after[col_open + 1..col_close];
                    let col = eval_col(col_expr, &d, 0);
                    assert_eq!(
                        col % 4,
                        0,
                        "line {}: {prefix}..][{col_expr}] = バイトオフセット {col} が 4 の倍数でない \
                         (行 {lineno}): {}",
                        lineno + 1,
                        line.trim()
                    );
                    assert!(col + 4 <= tk, "列 {col} が GEMM_TK {tk} を超えて 4 バイト読む");
                    checked += 1;
                    search = at + prefix.len();
                }
            }
        }
        assert!(checked >= 5, "共有配列の i32 アクセスを {checked} 件しか検査していない");
        println!(
            "kernels_int8.cu: tile={tile}x{tile_n} tk={tk} sp={sp} block={bx}x{by} smem={smem}B \
             checked {checked} shared int accesses"
        );
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

        // ① 成功ステップ後: gdn_pos = 1 == cache.len() (advance 後) → cur 面 = A
        cache.advance(1);
        assert_eq!(chain_q35_gdn_flush_pending(&mut cache), Some(()));
        let (hc, hs) = host(&mut cache);
        assert_bits_eq(&hc, &conv_a, "①conv は cur 面 (A) と bit 一致");
        assert_bits_eq(&hs, &ssm_a, "①ssm は cur 面 (A) と bit 一致");
        assert_eq!(mirror_pending(), None, "①フラッシュ後 pending は解除");

        // ② 層途中失敗: swap 済み (cur = 書き込み面 = state at 2)・cache.len() = 1
        //    → 1-cur 面 (= state at 1 = A) を回復する
        cache.advance(1); // len = 2
        with_mirror_ext(key, 0, 1, 4, 0, cache.epoch(), false, |m, _| {
            let gd = m.gdn[0].as_mut().unwrap();
            gd.cur = 1; // 成功ステップが cur を B 面へ交換した直後の想定
            m.gdn_pos[0] = Some(3);
            m.gdn_pending = Some(3);
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
            ctx, qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk, nv, dk, dv,
            d_conv, eps, q_scale, None,
        )
    } else {
        gdn_scan_impl(
            ctx, qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk, nv, dk, dv,
            d_conv, eps, q_scale,
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
    let dq = DevBuffer::from_slice(qkv)?;
    let dz = DevBuffer::from_slice(z)?;
    let dbeta = DevBuffer::from_slice(beta)?;
    let dalpha = DevBuffer::from_slice(alpha)?;
    let dconv_in = DevBuffer::from_slice(conv_st)?;
    let dssm_in = DevBuffer::from_slice(ssm_st)?;
    let dwc = DevBuffer::from_slice(wconv)?;
    let dwa = DevBuffer::from_slice(wa)?;
    let dwd = DevBuffer::from_slice(wdt)?;
    let dwn = DevBuffer::from_slice(wnorm)?;
    let dy = DevBuffer::from_slice(&vec![0.0f32; n * vd])?;
    let dconv_out = DevBuffer::from_slice(&vec![0.0f32; conv_dim * cs])?;
    let dssm_out = DevBuffer::from_slice(&vec![0.0f32; nv * dk * dv])?;
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);

    let (mut pq, mut pz, mut pb, mut pa) = (
        dq.ptr(),
        dz.ptr(),
        dbeta.ptr(),
        dalpha.ptr(),
    );
    let (mut pci, mut psi) = (dconv_in.ptr(), dssm_in.ptr());
    let (mut pwc, mut pwa, mut pwd, mut pwn) = (dwc.ptr(), dwa.ptr(), dwd.ptr(), dwn.ptr());
    let (mut py, mut pco, mut pso) = (dy.ptr(), dconv_out.ptr(), dssm_out.ptr());
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

    let y = dy.to_vec::<f32>(n * vd)?;
    let conv_new = dconv_out.to_vec::<f32>(conv_dim * cs)?;
    let ssm_new = dssm_out.to_vec::<f32>(nv * dk * dv)?;
    // s65: 発行元の件数分類 (y は linear / conv+ssm は GDN 状態)
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
    conv_st.copy_from_slice(&conv_new);
    ssm_st.copy_from_slice(&ssm_new);
    q.end();
    Ok(y)
}

// ---------------------------------------------------------------------------
// A1-r2: GDN chunked scan (長 prefill 用・3-pass 構成)
// ---------------------------------------------------------------------------

/// chunked ルーティングの閾値 (n がこれ以上で 3-pass chunked カーネルを使用)。
/// chunk 分割は [`crate::gdn::chunk_plan`](n, この閾値, 1024, 16) を CPU 参照と共有。
pub(crate) const GDN_CHUNK_MIN_TOKENS: usize = 2048;

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
        ctx, qkv, z, beta, alpha, conv_st, ssm_st, wconv, wa, wdt, wnorm, n, nk, nv, dk, dv,
        d_conv, eps, q_scale, Some(yptr),
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
    y_dst: Option<u64>,
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
    debug_assert_eq!(dk, dv, "chunked scan は dk == dv 前提");
    debug_assert_eq!(d_conv, 4, "chunked scan は d_conv == 4 前提");

    let (n_chunks, chunk_len) = crate::gdn::chunk_plan(n, GDN_CHUNK_MIN_TOKENS, 1024, 16);

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
        s.buf(GdnBuf::ConvOut)
            .copy_from(&vec![0.0f32; conv_dim * cs])?;
        s.buf(GdnBuf::SsmOut)
            .copy_from(&vec![0.0f32; nv * dk * dv])?;
        s.q.as_ref().unwrap().copy_from(qkv)?;
        s.z.as_ref().unwrap().copy_from(z)?;
        s.beta.as_ref().unwrap().copy_from(beta)?;
        s.alpha.as_ref().unwrap().copy_from(alpha)?;
        s.conv_in.as_ref().unwrap().copy_from(conv_st)?;
        s.ssm_in.as_ref().unwrap().copy_from(ssm_st)?;
        s.wc.as_ref().unwrap().copy_from(wconv)?;
        s.wa.as_ref().unwrap().copy_from(wa)?;
        s.wd.as_ref().unwrap().copy_from(wdt)?;
        s.wn.as_ref().unwrap().copy_from(wnorm)?;
        Ok(p)
    })?;
    q.lap(crate::prof::D2H_GDN_PF_LAUNCH);

    // ---- pass 1: chunk 並列でローカル遷移 (W[ci], M[ci]) --------------------
    let (mut pq, mut pb, mut pa) = (dp.q, dp.beta, dp.alpha);
    let (mut pci, mut pwc, mut pwa, mut pwd) = (dp.conv_in, dp.wc, dp.wa, dp.wd);
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
    let (mut psi, mut pm2, mut pw2, mut psinit) = (dp.ssm_in, dp.m, dp.w, dp.sinit);
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
    let (mut pci3, mut psinit3) = (dp.conv_in, dp.sinit);
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
        // s72: y はデバイス上に残す (呼び出し側がチェーンで消費する)。状態だけ落とす。
        let (conv_new, ssm_new) =
            GDN_SCRATCH.with(|cell| -> Result<(Vec<f32>, Vec<f32>)> {
                let s = cell.borrow();
                Ok((
                    s.buf(GdnBuf::ConvOut).to_vec::<f32>(conv_dim * cs)?,
                    s.buf(GdnBuf::SsmOut).to_vec::<f32>(nv * dk * dv)?,
                ))
            })?;
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
        conv_st.copy_from_slice(&conv_new);
        ssm_st.copy_from_slice(&ssm_new);
        q.end();
        return Ok(Vec::new());
    }
    let (y, conv_new, ssm_new) =
        GDN_SCRATCH.with(|cell| -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
            let s = cell.borrow();
            Ok((
                s.buf(GdnBuf::Y).to_vec::<f32>(n * vd)?,
                s.buf(GdnBuf::ConvOut).to_vec::<f32>(conv_dim * cs)?,
                s.buf(GdnBuf::SsmOut).to_vec::<f32>(nv * dk * dv)?,
            ))
        })?;
    // s65: 発行元の件数分類 (y は linear / conv+ssm は GDN 状態)
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
    crate::prof::mark_d2h_src(crate::prof::D2H_SRC_GDN_STATE);
    conv_st.copy_from_slice(&conv_new);
    ssm_st.copy_from_slice(&ssm_new);
    q.end();
    Ok(y)
}
