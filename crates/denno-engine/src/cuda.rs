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

    /// カーネル起動 + 同期 (デフォルトストリーム)
    unsafe fn launch(
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
        )?;
        check((lib.ctx_sync)(), "cuCtxSynchronize")
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
            self.launch(
                self.f_quant,
                (m as u32, 1, 1),
                (256, 1, 1),
                256 * 4,
                &mut args,
            )?;
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
            self.launch(self.f_mv, (blocks, 1, 1), (256, 1, 1), 0, &mut args)?;
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
