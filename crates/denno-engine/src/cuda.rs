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
        let mut out = vec![unsafe { std::mem::zeroed::<T>() }; len.max(1)];
        if bytes > 0 {
            unsafe {
                check(
                    (lib()?.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, bytes),
                    "cuMemcpyDtoH",
                )?;
            }
        }
        out.truncate(len);
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
    // P17h: qkv rope + bias (CPU rope_row とビット一致)
    f_rope: CUfunction,
    // P17k ステップ 5: prefill 用行並列 rope (qkv→rope→attention 連続化)
    f_rope_pf: CUfunction,
}

const PTX: &[u8] = include_bytes!("../kernels/kernels_int8.ptx");
const PTX_ELEM: &[u8] = include_bytes!("../kernels/kernels_elem.ptx");

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
                f_rope,
                f_rope_pf,
            })
        }
    }

    /// "NVIDIA GeForce RTX 5090 (sm_120, driver 13010)" 形式の情報
    pub fn info(&self) -> String {
        format!("{} (sm_{}{}, driver {})", self.name, self.cc.0, self.cc.1, self.driver_ver)
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
        debug_assert_eq!(w.bytes(), k * n);
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
            let gmx = m.div_ceil(128) as u32;
            let gmy = n.div_ceil(128) as u32;
            self.launch_async(self.f_gemm, (gmx, gmy, 1), (16, 16, 1), 0, &mut args)?;
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
        debug_assert_eq!(w.bytes(), k * n);
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
}

fn weights_for(uid: u64, qw: &[i8], scale: &[f32], k: usize, n: usize) -> Result<&'static GpuW> {
    use std::collections::HashMap;
    use std::sync::Mutex;
    static REG: OnceLock<Mutex<HashMap<u64, Box<GpuW>>>> = OnceLock::new();
    let reg = REG.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = reg.lock().expect("weights registry poisoned");
    if !map.contains_key(&uid) {
        let w = DevBuffer::from_slice(qw)?;
        let sc = DevBuffer::from_slice(scale)?;
        tracing::info!(
            "CUDA weight upload (uid={uid}): k={k} n={n} ({:.1} MB)",
            (k * n + n * 4) as f64 / 1e6
        );
        map.insert(uid, Box::new(GpuW { w, sc, k, n }));
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
        Ok(out) => Some(out),
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
                let gmx = m.div_ceil(128) as u32;
                let gmy = (2 * ff).div_ceil(128) as u32;
                ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (16, 16, 1), 0, &mut args)?;
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
                let gmx = m.div_ceil(128) as u32;
                let gmy = e.div_ceil(128) as u32;
                ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (16, 16, 1), 0, &mut args)?;
                check(
                    (lib.memcpy_dtoh_async)(ho.ptr, dout2.ptr(), m * e * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync",
                )?;
                ctx.sync()?;
                Ok(())
            })?;
        }
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
        let w = DevBuffer::from_slice(&w)?;
        let sc = DevBuffer::from_slice(&sc)?;
        tracing::info!(
            "CUDA grouped weight upload ({} parts, uids={}..): k={k} n={n} ({:.1} MB)",
            parts.len(),
            uids[0],
            (k * n + n * 4) as f64 / 1e6
        );
        map.insert(uids.to_vec(), Box::new(GpuW { w, sc, k, n }));
    }
    let e = map.get(uids).expect("just inserted");
    Ok(unsafe { std::mem::transmute::<&GpuW, &'static GpuW>(e.as_ref()) })
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
    if !cuda_enabled() {
        return None;
    }
    let use_gemm = m > max_m() && m >= gemm_min_m() && gemm_enabled();
    // m が GEMV 経路の上限を超えて GEMM も発動しないなら CPU へ
    if m > max_m() && !use_gemm {
        return None;
    }
    let ctx = shared_ctx()?;
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU int8: {e:#}");
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    }
    let gw = match weights_for(uid, qw, scale, k, n) {
        Ok(g) => g,
        Err(e) => {
            tracing::error!("CUDA weight upload failed, using CPU int8: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    };
    // P17i: m が大きい (prefill) ときは tiled GEMM、小さい (decode/batch) ときは
    // GEMV 型カーネル (固定オーバーヘッドが小さい) を使い分ける。どちらも CPU と
    // ビット一致。
    let r = if use_gemm {
        linear_gemm(ctx, a, gw, m, k, n)
    } else {
        linear_pinned(ctx, a, gw, m, k, n)
    };
    match r {
        Ok(out) => Some(out),
        Err(e) => {
            // 実行時エラーは再試行せず恒久的に CPU へ (数値は同一仕様なので不変)
            tracing::error!("CUDA linear failed, falling back to CPU permanently: {e:#}");
            GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
            None
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
        }
        ctx.sync()?;
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
    debug_assert_eq!(a.len(), m * k);
    let lib = lib()?;
    ctx.make_current()?;
    SCRATCH.with(|cell| {
        let mut s = cell.borrow_mut();
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
            // 128x128 タイル × 16x16 thread (1 thread が 8x8 出力)
            let gmx = m.div_ceil(128) as u32;
            let gmy = n.div_ceil(128) as u32;
            ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (16, 16, 1), 0, &mut args)?;

            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
        }
        ctx.sync()?;
        Ok(ho.as_slice::<f32>()[..m * n].to_vec())
    })
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
const CH_SLOTS: usize = 9;

// P17j: prefill attention カーネルのタイル定数 (kernels_elem.cu の PF_BQ/PF_BK
// と同期すること)。BQ=行タイル・BK=key タイル。
const ATTN_PF_BQ: usize = 16;
const ATTN_PF_BK: usize = 32;

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
    // P17h: ステップ内で sync が無くなるため pinned は (層×用途) ごとに専用化する
    // (async H2D の DMA 完了前にホストが上書きする競合を避ける)。id は呼び出し側が
    // 層番号と用途から一意に採番する。
    pins: Vec<Option<PinMem>>,
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
        pins: Vec::new(),
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

/// スロットからホストへダウンロード (D2H + 同期)。
pub fn chain_download(slot: usize, len: usize) -> Option<Vec<f32>> {
    let ctx = chain_ctx()?;
    let bytes = len.checked_mul(4)?;
    ok(chain_ensure_buf(slot, bytes))?;
    CHAIN.with(|cell| -> Option<Vec<f32>> {
        let mut s = cell.borrow_mut();
        ok(Scratch::ensure_pin(&mut s.pin_dn, bytes))?;
        let buf = s.bufs[slot].as_ref().unwrap().ptr();
        let pin = s.pin_dn.as_mut().unwrap();
        {
            let lib = ok(lib())?;
            unsafe {
                ok(check(
                    (lib.memcpy_dtoh_async)(pin.ptr as *mut c_void, buf, bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (chain)",
                ))?;
            }
        }
        ok(ctx.sync())?;
        Some(pin.as_slice::<f32>()[..len].to_vec())
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
}

impl KvMirror {
    fn bytes(&self) -> usize {
        self.layers
            .iter()
            .map(|(k, v)| {
                k.as_ref().map(|b| b.bytes()).unwrap_or(0)
                    + v.as_ref().map(|b| b.bytes()).unwrap_or(0)
            })
            .sum()
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
                },
                now,
            ));
            ms.len() - 1
        }
    };
    // エビクト: 上限超過で LRU (自分は除外)
    let limit = gpu_kv_limit_bytes();
    let mut total: usize = ms.iter().map(|(_, m, _)| m.bytes()).sum();
    while total > limit && ms.len() > 1 {
        let vic = (0..ms.len())
            .filter(|&i| ms[i].0 != key)
            .min_by_key(|&i| ms[i].2)
            .unwrap();
        total -= ms[vic].1.bytes();
        ms.swap_remove(vic);
    }
    let idx = ms.iter().position(|(k, _, _)| *k == key)?;
    let m = &mut ms[idx].1;
    if li >= m.layers.len() {
        return None; // n_layers 不一致 (起こらないはず)
    }
    // cap 成長 (全レイヤ realloc + len リセット → 各レイヤ次回使用時に再シード)
    if m.cap < t_total {
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
    let (mut kptr, mut vptr) = with_mirror(key, li, n_layers, t_total, kv_dim, |m, have| {
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
    // 担当するため dh >= PF_BQ、smem <= 48KB。
    if dh < ATTN_PF_BQ || dh > 1024 {
        return None;
    }
    let sh_f = (ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1)
        + ATTN_PF_BQ * ATTN_PF_BK
        + 2 * ATTN_PF_BQ;
    if sh_f * 4 > 48 * 1024 {
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
    let (mut kptr, mut vptr) = with_mirror(key, li, n_layers, t_total, kv_dim, |m, have| {
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
                (sh_f * 4) as u32,
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
        let Scratch { aq, asc, out, ho, .. } = &mut *s;
        Scratch::ensure(aq, m * k)?;
        Scratch::ensure(asc, m * 4)?;
        Scratch::ensure(out, m * e * 4)?;
        Scratch::ensure_pin(ho, m * e * 4)?;
        let (daq, dasc, dout) = (aq.as_ref().unwrap(), asc.as_ref().unwrap(), out.as_ref().unwrap());
        let ho = ho.as_mut().unwrap();
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
                let gmx = m.div_ceil(128) as u32;
                let gmy = e.div_ceil(128) as u32;
                ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (16, 16, 1), 0, &mut args)?;
                // 3) D2H + 同期 (attention カーネルは同一ストリームの先行発行なので完結済み)
                check(
                    (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), m * e * 4, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync",
                )?;
                ctx.sync()?;
                Ok(())
            })?;
        }
        Ok(ho.as_slice::<f32>()[..m * e].to_vec())
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
    if h.len() != n * e {
        return None;
    }
    let ctx = chain_ctx()?;
    let lib = ok(lib())?;

    // ミラー取得 + 履歴行 [have..pos0) の同期 (現在行 [pos0..t_total) は rope
    // カーネルが書く)。have > pos0 はミラーがホストに先行 → フォールバック
    // (ホストキャッシュが完全でないため)。
    let (kptr, vptr) = with_mirror(key, li, n_layers, t_total, kv_dim, |m, have| {
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
        ok(Scratch::ensure(&mut s.qkv_x, n * e * 4))?;
        ok(Scratch::ensure(&mut s.qkv_aq, n * e))?;
        ok(Scratch::ensure(&mut s.qkv_asc, n * 4))?;
        ok(Scratch::ensure(&mut s.qkv_pf, n * width * 4))?;
        ok(Scratch::ensure(&mut s.qkv_q, n * qn * 4))?;
        ok(Scratch::ensure(&mut s.rope_tab_pf, (n * dh + bias_len) * 4))?;
        ok(Scratch::ensure_pin(&mut s.pin_qkv, (n * e * 4).max(n * dh * 4 + bias_len * 4)))?;

        // h アップロード (pinned + async)。pin は次層で上書きするが、その前に
        // 本関数内の sync が挟まるため安全。
        {
            let pin = s.pin_qkv.as_mut().unwrap();
            pin.as_mut_slice::<f32>()[..n * e].copy_from_slice(h);
        }
        let (dx, daq, dasc, dqkv, dq, dtab) = (
            s.qkv_x.as_ref().unwrap().ptr(),
            s.qkv_aq.as_ref().unwrap().ptr(),
            s.qkv_asc.as_ref().unwrap().ptr(),
            s.qkv_pf.as_ref().unwrap().ptr(),
            s.qkv_q.as_ref().unwrap().ptr(),
            s.rope_tab_pf.as_ref().unwrap().ptr(),
        );
        let pin_ptr = s.pin_qkv.as_ref().unwrap().ptr;
        unsafe {
            ok(check(
                (lib.memcpy_htod_async)(dx, pin_ptr as *const c_void, n * e * 4, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (qkv x)",
            ))?;
        }
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
            let gmx = n.div_ceil(128) as u32;
            let gmy = width.div_ceil(128) as u32;
            ok(unsafe { ctx.launch_async(ctx.f_gemm, (gmx, gmy, 1), (16, 16, 1), 0, &mut args) })
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
    with_mirror(key, li, n_layers, t_total, kv_dim, |m, _| {
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
) -> Option<()> {
    if !gpu_attention_enabled() {
        return None;
    }
    if n == 0 || t_total == 0 || group == 0 || t_total > 65535 {
        return None;
    }
    // カーネル制約: chain_attention_prefill と同一
    if dh < ATTN_PF_BQ || dh > 1024 {
        return None;
    }
    let sh_f = (ATTN_PF_BQ + 2 * ATTN_PF_BK) * (dh + 1)
        + ATTN_PF_BQ * ATTN_PF_BK
        + 2 * ATTN_PF_BQ;
    if sh_f * 4 > 48 * 1024 {
        return None;
    }
    let qn = heads.checked_mul(dh)?;
    let ctx = chain_ctx()?;
    ok(chain_ensure_buf(CH_ATT, n * qn * 4))?;
    let (mut kptr, mut vptr) = with_mirror(key, li, n_layers, t_total, kv_dim, |m, _| {
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
                (sh_f * 4) as u32,
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

    let (mut kptr, mut vptr) = with_mirror(key, li, n_layers, t_total, kv_dim, |m, have| {
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
        let (kptr, vptr) = with_mirror(MIRROR_TEST_KEY, 0, 1, t_total, kv_dim, |m, have| {
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
            let w = ctx.upload_i8(&q).unwrap();
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
            let w = ctx.upload_i8(&q).unwrap();
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
        let (kptr, vptr) = with_mirror(key, 0, 1, t_total, kv_dim, |m, _have| {
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
}
