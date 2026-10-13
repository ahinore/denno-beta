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
}

const PTX: &[u8] = include_bytes!("../kernels/kernels_int8.ptx");

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
            check(
                (lib.module_get_function)(&mut f_quant, module, b"dno_quantize_rows\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_quantize_rows)",
            )?;
            check(
                (lib.module_get_function)(&mut f_mv, module, b"dno_matvec_rows_bt_i8\0".as_ptr() as *const c_char),
                "cuModuleGetFunction(dno_matvec_rows_bt_i8)",
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
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch {
        x: None, aq: None, asc: None, out: None, hx: None, ho: None,
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

/// P17c: 複数の int8 Linear を 1 回の GPU 呼び出しに融合する (QKV / gate+up 用)。
/// 同じ入力を消費する線形層群の重み行を**パート順に連結**した 1 つのバッファとして
/// 登録・計算する。各行の演算 (量子化スケール・i32 累積・スケール乗算) は個別
/// 呼び出しと完全に同一なので、出力をパート幅で切り分ければビット一致する。
/// 1 線形層 1 同期の固定オーバーヘッド (約 28µs) がパート数 → 1 になる効果:
/// 実測で qkv は 3 呼び出し分の 96µs/層 → 約 40µs/層。
///
/// 戻り値 None = CPU パス (無効/初期化失敗/実行時エラー/m>閾値/一部が Q8 でない)。
pub fn maybe_gpu_linear_grouped(parts: &[&Quant8Linear], a: &[f32], m: usize) -> Option<Vec<f32>> {
    if !cuda_enabled() || m > max_m() || parts.is_empty() {
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
    match linear_pinned(ctx, a, gw, m, k, n) {
        Ok(out) => Some(out),
        Err(e) => {
            tracing::error!("CUDA grouped linear failed, falling back to CPU permanently: {e:#}");
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
    if !cuda_enabled() || m > max_m() {
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
    match linear_pinned(ctx, a, gw, m, k, n) {
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
        let Scratch { x, aq, asc, out, hx, ho } = &mut *s;
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
}
