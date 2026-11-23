//! dense / tensor-core prefill GEMM と router / lm_head
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


/// P-async: decode の GEMV を「pinned staging + async コピー + 同期 1 回」で回すか。
///
/// 既定 **on** (`DENNO_GEMV_ASYNC=0` で従来の同期経路に戻る)。従来経路は
/// `copy_from` (同期 H2D) → カーネル → `sync` → `to_vec` (同期 D2H) で、
/// **カーネル・引数・発行順・丸め順は同じ**、コピーの API だけが変わる
/// (バイト列は同一)。実測 2026-10-30: 26.15 -> 25.66 s (118 tok)、
/// 出力 md5 は v1 と一致 (`3b77a3b996fd`、12 ラン全部同じ)。
pub fn gemv_async_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("DENNO_GEMV_ASYNC").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}


/// v3 の PTX の置き場所。`DENNO_GEMV_V3_PTX` があればそれ、無ければビルド時の
/// クレートディレクトリ配下 `kernels/kernels_moe_v3.ptx`。
fn gemv_v3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GEMV_V3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_v3.ptx"),
    }
}


/// v3 のカーネルを取得する (初回だけ PTX を読む)。
///
/// 失敗は `None` をキャッシュする (= 二度と試さない)。読めなかったときは
/// **1 行だけ**警告を出し、呼び出し側は v2 へ落ちる。呼ぶ前に必ず context を
/// current にしておくこと (cuModuleLoadData は current context に入る)。
pub(crate) fn gemv_v3_kernels() -> Option<&'static V3Kernels> {
    V3_KERNELS
        .get_or_init(|| {
            let path = gemv_v3_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[gemv-v3] PTX が読めない ({}) : {e} → v2 へ",
                        path.display()
                    );
                    return None;
                }
            };
            // 引数の数が合わない PTX (task 66 の旧版 = 6 引数) を読むと、
            // cuLaunchKernel が rows_per_block / lanes を渡さないまま起動して
            // 黙って別の行を書く。**モジュールを読む前に** ABI を確かめて弾く。
            let text = String::from_utf8_lossy(&image);
            let mut short: Vec<&str> = Vec::new();
            for name in V3_KERNEL_NAMES {
                if ptx_entry_params(&text, name) != Some(V3_NARGS) {
                    short.push(name);
                }
            }
            if !short.is_empty() {
                eprintln!(
                    "[gemv-v3] PTX の引数が {} 本でない ({}) → v2 へ。\
                     `bash build_ptx.sh moe_v3` で再生成する",
                    V3_NARGS,
                    short.join(", ")
                );
                return None;
            }
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[gemv-v3] cuModuleLoadData 失敗 → v2 へ");
                    return None;
                }
                let mut f: [CUfunction; 4] = [std::ptr::null_mut(); 4];
                for (k, name) in V3_KERNEL_NAMES.iter().enumerate() {
                    let mut cname = name.as_bytes().to_vec();
                    cname.push(0); // cuModuleGetFunction は NUL 終端の名前を期待する
                    if (lib.module_get_function)(&mut f[k], module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[gemv-v3] PTX に {name} が無い → v2 へ");
                        return None;
                    }
                }
                Some(V3Kernels { module, f })
            }
        })
        .as_ref()
}


/// PTX の置き場所。`DENNO_GEMV_V4_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_moe_v4.ptx` (v3 と同じ規約)。
fn gemv_v4_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GEMV_V4_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_v4.ptx"),
    }
}


/// v4 のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
/// 呼ぶ前に必ず context を current にしておくこと (`cuModuleLoadData` は
/// current context に入る)。
pub(crate) fn gemv_v4_kernels() -> Option<&'static V4Kernels> {
    V4_KERNELS
        .get_or_init(|| {
            let path = gemv_v4_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[gemv-v4] PTX が読めない ({}) : {e} → v3 へ",
                        path.display()
                    );
                    return None;
                }
            };
            // ABI (引数の数) を **モジュールを読む前**に確かめる。合わない PTX を
            // 読むと rows/lanes を渡さないまま起動して黙って別の行を書く。
            let text = String::from_utf8_lossy(&image);
            let mut short: Vec<&str> = Vec::new();
            for name in V4_KERNEL_NAMES {
                if ptx_entry_params(&text, name) != Some(V4_NARGS) {
                    short.push(name);
                }
            }
            if !short.is_empty() {
                eprintln!(
                    "[gemv-v4] PTX の引数が {} 本でない ({}) → v3 へ。\
                     `bash build_ptx.sh moe_v4` で再生成する",
                    V4_NARGS,
                    short.join(", ")
                );
                return None;
            }
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[gemv-v4] cuModuleLoadData 失敗 → v3 へ");
                    return None;
                }
                let mut f: [CUfunction; 4] = [std::ptr::null_mut(); 4];
                for (k, name) in V4_KERNEL_NAMES.iter().enumerate() {
                    let mut cname = name.as_bytes().to_vec();
                    cname.push(0); // cuModuleGetFunction は NUL 終端の名前を期待する
                    if (lib.module_get_function)(&mut f[k], module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[gemv-v4] PTX に {name} が無い → v3 へ");
                        return None;
                    }
                }
                Some(V4Kernels { module, f })
            }
        })
        .as_ref()
}


fn gemv_v5_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GEMV_V5_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_v5.ptx"),
    }
}


/// v5 のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
pub(crate) fn gemv_v5_kernels() -> Option<&'static V5Kernels> {
    V5_KERNELS
        .get_or_init(|| {
            let path = gemv_v5_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            for name in V5_KERNEL_NAMES {
                if ptx_entry_params(&text, name) != Some(V5_NARGS) {
                    eprintln!(
                        "[gemv-v5] PTX の引数が {V5_NARGS} 本でない ({name}) → v4 へ。\
                         `bash build_ptx.sh moe_v5` で再生成する"
                    );
                    return None;
                }
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut f: [CUfunction; 2] = [std::ptr::null_mut(); 2];
                for (k, name) in V5_KERNEL_NAMES.iter().enumerate() {
                    let mut cname = name.as_bytes().to_vec();
                    cname.push(0);
                    if (lib.module_get_function)(&mut f[k], module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[gemv-v5] PTX に {name} が無い → v4 へ");
                        return None;
                    }
                }
                Some(V5Kernels { module, f })
            }
        })
        .as_ref()
}


/// Q5_K v5 のカーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
pub(crate) fn gemv_v5_q5k_kernels() -> Option<&'static V5Q5KKernels> {
    V5_Q5K_KERNELS
        .get_or_init(|| {
            let path = v5_q5k_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, V5_Q5K_KERNEL_NAME) != Some(V5_NARGS) {
                eprintln!(
                    "[gemv-v5-q5k] PTX の引数が {V5_NARGS} 本でない ({V5_Q5K_KERNEL_NAME}) → v4 へ。\
                     `bash build_ptx.sh moe_v5_q5k` で再生成する"
                );
                return None;
            }
            image.push(0);
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
                let mut cname = V5_Q5K_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[gemv-v5-q5k] PTX に {V5_Q5K_KERNEL_NAME} が無い → v4 へ");
                    return None;
                }
                Some(V5Q5KKernels { module, f })
            }
        })
        .as_ref()
}


/// decode GEMV が使うカーネル一式を env から決める (`moe_gemv_dev` と
/// task 124 の重ね打ち経路で共有する)。優先順は fp16 → v3 → v2。
pub(crate) fn gemv_kernel_mode() -> GemvKernelMode {
    if moe_gemv_fp16_on() {
        GemvKernelMode::Fp16
    } else if moe_gemv_v3_on() {
        GemvKernelMode::V3
    } else {
        GemvKernelMode::V2
    }
}


/// fp16 カーネルが対応する型か (純関数。Q5_K とそれ以外は v3 / v2 へ落ちる)。
pub(crate) fn gemv_fp16_ty_ok(ty: crate::kquant::GgmlType) -> bool {
    use crate::kquant::GgmlType as T;
    matches!(ty, T::Q8_0 | T::Q5_1 | T::Q4K)
}


/// fp16 の PTX の置き場所。`DENNO_GEMV_FP16_PTX` があればそれ、無ければビルド時の
/// クレートディレクトリ配下 `kernels/kernels_moe_fp16.ptx`。
fn gemv_fp16_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GEMV_FP16_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_fp16.ptx"),
    }
}


/// fp16 カーネルを取得する (初回だけ PTX を読む)。
///
/// 失敗は `None` をキャッシュする (= 二度と試さない)。読めなかったときは
/// **1 行だけ**警告を出し、呼び出し側は v3 / v2 へ落ちる。呼ぶ前に必ず context を
/// current にしておくこと (cuModuleLoadData は current context に入る)。
pub(crate) fn gemv_fp16_kernels() -> Option<&'static Fp16Kernels> {
    FP16_KERNELS
        .get_or_init(|| {
            let path = gemv_fp16_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[gemv-fp16] PTX が読めない ({}) : {e} → v3/v2 へ。\
                         `bash build_ptx.sh moe_fp16` で再生成する",
                        path.display()
                    );
                    return None;
                }
            };
            // 引数の数が合わない PTX を読むと、cuLaunchKernel が余分な引数を無視 /
            // 足りない引数を不定のまま起動して**黙って別の行を書く**。モジュールを
            // 読む前に ABI を確かめて弾く (v3 と同じ規約)。
            let text = String::from_utf8_lossy(&image);
            let mut short: Vec<&str> = Vec::new();
            for name in FP16_KERNEL_NAMES {
                if ptx_entry_params(&text, name) != Some(FP16_NARGS) {
                    short.push(name);
                }
            }
            if !short.is_empty() {
                eprintln!(
                    "[gemv-fp16] PTX の引数が {} 本でない ({}) → v3/v2 へ。\
                     `bash build_ptx.sh moe_fp16` で再生成する",
                    FP16_NARGS,
                    short.join(", ")
                );
                return None;
            }
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            unsafe {
                let mut module: CUmodule = std::ptr::null_mut();
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[gemv-fp16] cuModuleLoadData 失敗 → v3/v2 へ");
                    return None;
                }
                let mut f: [CUfunction; 3] = [std::ptr::null_mut(); 3];
                for (k, name) in FP16_KERNEL_NAMES.iter().enumerate() {
                    let mut cname = name.as_bytes().to_vec();
                    cname.push(0); // cuModuleGetFunction は NUL 終端の名前を期待する
                    if (lib.module_get_function)(&mut f[k], module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[gemv-fp16] PTX に {name} が無い → v3/v2 へ");
                        return None;
                    }
                }
                Some(Fp16Kernels { module, f })
            }
        })
        .as_ref()
}


/// 一時診断: 最初の N 回だけ GEMV の内訳 (H2D / 起動 / 同期 / D2H) を stderr に出す。
/// `DENNO_GEMV_TRACE=N` (未設定 = 無効)。
pub(crate) fn gemv_trace_allow() -> bool {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static LEFT: std::sync::OnceLock<AtomicUsize> = std::sync::OnceLock::new();
    let left = LEFT.get_or_init(|| {
        let n = std::env::var("DENNO_GEMV_TRACE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        AtomicUsize::new(n)
    });
    left.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| {
        if v == 0 {
            None
        } else {
            Some(v - 1)
        }
    })
    .is_ok()
}


// ===========================================================================
// P40: grouped expert GEMM (可変 m・1 起動) — `kernels_gemm_grouped.cu`
// ===========================================================================
//
// 目的: prefill で expert ごとに回している GEMM (expert × 3 行列) を、
// 「層全体の segment 表 + パック済み重み」の 2 起動 (gate+up+silu / down) に
// まとめる。重みは `model::moe_grouped::pack_expert_weights` の
// `[gate(e0..eN)][up(e0..eN)][down(e0..eN)]` をそのまま渡す。
//
// 既定 off (`DENNO_MOE_GROUPED_GPU=1` で有効)。off のとき呼び出し側はこの節に
// 入らないので、既存の per-expert 経路は 1 バイトも変わらない。
//
// PTX を `include_bytes!` しない理由: `kernels_gemm_grouped.ptx` の生成は
// メインセッションの担当で、このコンテナには nvcc が無い。埋め込むと PTX が
// 置かれるまで `cargo check --lib` が通らない。起動時に 1 回だけファイルから
// 読む (`DENNO_MOE_GROUPED_PTX` で差し替え可)。読めなければ None を返して
// CPU へ落ちる (fail closed)。
//
// 数値: prefill はビット一致契約の外 (2026-10-28)。カーネルは FMA 融合と
// 共有メモリ経由のデコードを使うので、CPU 参照とは丸め順が違う。差は呼び出し側
// (`model::moe_grouped_gpu`) が argmax / max abs diff で出す。

/// パック済み重みの 1 ブロック (gate / up / down のどれか) の形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedMatrix {
    pub ty: crate::kquant::GgmlType,
    /// 行長 (量子型のブロック長で割り切れること)
    pub n_in: usize,
    /// 行数
    pub n_out: usize,
    /// 1 行のバイト数 ([`crate::model::moe_expert::row_bytes`] と同じ値)
    pub row_bytes: usize,
    /// パック済みバッファ内のこのブロックの先頭バイト
    pub w_off: usize,
}


impl GroupedMatrix {
    /// カーネルに渡す (量子型 id, ブロック要素数, ブロックバイト数)。
    /// 未対応の型は `None` (呼び出し側は CPU へ落ちる)。
    pub fn kernel_layout(&self) -> Option<(i32, usize, usize)> {
        use crate::kquant::GgmlType as T;
        // kernels_gemm_grouped.cu の dno_gr_deq の分岐と 1:1 で対応する。
        match self.ty {
            T::Q8_0 => Some((0, 32, 34)),
            T::Q5_1 => Some((1, 32, 24)),
            T::Q4K => Some((2, 256, 144)),
            T::Q5K => Some((3, 256, 176)),
            _ => None,
        }
    }

    /// 1 expert ぶんのバイト数 (パック済みブロック内のストライド)。
    pub fn expert_bytes(&self) -> Option<usize> {
        self.n_out.checked_mul(self.row_bytes)
    }

    /// 形の検査 (起動前)。`Ok(())` 以外は起動しない。
    pub fn check(&self) -> Result<()> {
        let Some((_, blkj, _)) = self.kernel_layout() else {
            bail!("grouped GEMM の対象外の量子型 {:?}", self.ty);
        };
        if self.n_in == 0 || self.n_out == 0 || self.row_bytes == 0 {
            bail!(
                "grouped GEMM: 形に 0 がある (n_in={}, n_out={}, row_bytes={})",
                self.n_in,
                self.n_out,
                self.row_bytes
            );
        }
        if self.n_in % blkj != 0 {
            bail!(
                "grouped GEMM: n_in={} がブロック長 {blkj} で割り切れない",
                self.n_in
            );
        }
        let want_rb = self.n_in / blkj * match self.ty {
            crate::kquant::GgmlType::Q8_0 => 34,
            crate::kquant::GgmlType::Q5_1 => 24,
            crate::kquant::GgmlType::Q4K => 144,
            crate::kquant::GgmlType::Q5K => 176,
            _ => 0,
        };
        if want_rb != self.row_bytes {
            bail!(
                "grouped GEMM: row_bytes={} が型から決まる {want_rb} と違う",
                self.row_bytes
            );
        }
        Ok(())
    }
}


/// 1 起動ぶんの grid / block / 動的共有メモリ (GPU 無しで検証できる純関数)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GroupedLaunch {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
}


/// groupd GEMM の起動形を決める。`None` = この形では起動しない (CPU へ)。
///
/// - grid.x = segment 数、grid.y = 行タイル (ceil(n_out / 16))、
///   grid.z = トークンタイル (ceil(m_max / 16))。
/// - 1 ブロックは 16 行 × 16 トークン。m < 16 の expert では z の余分なブロックが
///   即 return する (実形の平均 m は 10 なので 1 タイルで足りる)。
/// - 共有メモリは「デコード済み f32 の 1 ブロックぶん」だけ。
///   `nmat` 本の行列を同時にデコードするので `16 * blkj * 4 * nmat` バイト。
pub fn grouped_launch_plan(
    nseg: usize,
    m_max: usize,
    n_out: usize,
    blkj: usize,
    nmat: usize,
) -> Result<Option<GroupedLaunch>> {
    const ROWS: usize = 16; // kernels_gemm_grouped.cu の GR_ROWS
    const TG: usize = 16; // 同 GR_TG
    if nseg == 0 || m_max == 0 || n_out == 0 || blkj == 0 || nmat == 0 {
        return Ok(None);
    }
    let tiles_y = n_out.div_ceil(ROWS);
    let tiles_z = m_max.div_ceil(TG);
    // grid の上限 (y/z は 65535)。実形 (n_out 2560 / m_max 128) では届かない。
    if nseg > u32::MAX as usize || tiles_y > 65535 || tiles_z > 65535 {
        return Ok(None);
    }
    let smem = ROWS
        .checked_mul(blkj)
        .and_then(|v| v.checked_mul(4))
        .and_then(|v| v.checked_mul(nmat))
        .ok_or_else(|| anyhow!("grouped GEMM: 共有メモリ量が usize を溢れる"))?;
    // 48 KiB の既定を超えると起動できない (実績: 62 KiB で全滅)。
    if smem > 46_000 {
        return Ok(None);
    }
    Ok(Some(GroupedLaunch {
        grid: (nseg as u32, tiles_y as u32, tiles_z as u32),
        block: (ROWS as u32, TG as u32, 1),
        smem: smem as u32,
    }))
}


/// grouped GEMM のカーネル (PTX を起動時に 1 回だけ読む)。
pub(crate) struct GroupedKernels {
    #[allow(dead_code)]
    module: CUmodule,
    pub(crate) gate_up: CUfunction,
    pub(crate) down: CUfunction,
}


// 生ハンドル (context / module / function)。CUcontext はスレッドごとに current を
// 設定して使うハンドルで、所有権の移動を伴わない (CudaCtx と同じ規約)。
unsafe impl Send for GroupedKernels {}

unsafe impl Sync for GroupedKernels {}


/// PTX の置き場所。`DENNO_MOE_GROUPED_PTX` があればそれ、無ければビルド時の
/// クレートディレクトリ配下 `kernels/kernels_gemm_grouped.ptx`。
fn grouped_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MOE_GROUPED_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_grouped.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None をキャッシュする
/// (`moe_gemm_dev` の `f_gemm_q = None` と同じ扱い = 二度と試さない)。
pub(crate) fn grouped_kernels() -> Option<&'static GroupedKernels> {
    GROUPED_KERNELS
        .get_or_init(|| {
            let path = grouped_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[moe-grouped] PTX が読めない ({}) : {e} → CPU へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[moe-grouped] cuModuleLoadData 失敗 → CPU へ");
                    return None;
                }
                let mut gate_up: CUfunction = std::ptr::null_mut();
                let mut down: CUfunction = std::ptr::null_mut();
                let rc1 = (lib.module_get_function)(
                    &mut gate_up,
                    module,
                    b"dno_moe_grouped_gate_up\0".as_ptr() as *const c_char,
                );
                let rc2 = (lib.module_get_function)(
                    &mut down,
                    module,
                    b"dno_moe_grouped_down\0".as_ptr() as *const c_char,
                );
                if rc1 != CUDA_SUCCESS || rc2 != CUDA_SUCCESS {
                    eprintln!(
                        "[moe-grouped] PTX に dno_moe_grouped_gate_up / dno_moe_grouped_down が無い → CPU へ"
                    );
                    return None;
                }
                Some(GroupedKernels {
                    module,
                    gate_up,
                    down,
                })
            }
        })
        .as_ref()
}


/// grouped GEMM 用のスレッド局所スクラッチ (grow-only)。
///
/// `packed` だけは「層の重み」なので key (層 id) が変わったら作り直す。
/// それ以外 (seg / x / act / h) は要求サイズが増えたときだけ確保し直す。
pub(crate) struct GroupedScratch {
    /// `packed` の持ち主 (層) の識別子
    pub(crate) key: u64,
    pub(crate) packed: Option<DevBuffer>,
    pub(crate) packed_len: usize,
    /// P43: パック済み重みが予算から借りている分 (drop で返る)。
    pub(crate) reserved: Option<crate::model::expert_vram::ExternalReserve>,
    /// P43: この呼び出しで予算が足りず載せられなかった (呼び出し側は CPU へ)
    pub(crate) denied: bool,
    pub(crate) seg: Option<DevBuffer>,
    /// down 用の segment 表 (w_off のストライドが gate と違う)
    pub(crate) seg_dn: Option<DevBuffer>,
    pub(crate) x: Option<DevBuffer>,
    pub(crate) act: Option<DevBuffer>,
    pub(crate) h: Option<DevBuffer>,
}


impl GroupedScratch {
    pub(crate) const fn new() -> Self {
        Self {
            key: u64::MAX,
            packed: None,
            packed_len: 0,
            reserved: None,
            denied: false,
            seg: None,
            seg_dn: None,
            x: None,
            act: None,
            h: None,
        }
    }
}


/// 1 グループぶんの m 行カーネルを起動する (gate, up, silu, down)。
///
/// `seg` は**グループの先頭**を指し、`nseg` はそのグループの segment 数。
/// `rows` は全体表のまま (row_off が絶対添字なので offsets を足さない)。
/// 出力 (g / u / act / h) も絶対の行添字で書かれる。
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pf_launch_group(
    ctx: &CudaCtx,
    kernels: &ExpPfKernels,
    slot_g: usize,
    slot_u: usize,
    slot_d: usize,
    pg: CUdeviceptr,
    pu: CUdeviceptr,
    pd: CUdeviceptr,
    xb: CUdeviceptr,
    gb: CUdeviceptr,
    ub: CUdeviceptr,
    act: CUdeviceptr,
    hb: CUdeviceptr,
    seg: CUdeviceptr,
    rows_gu: CUdeviceptr,
    rows_id: CUdeviceptr,
    nseg: usize,
    ro_start: usize,
    ro_end: usize,
    n_in: usize,
    ff: usize,
    e: usize,
    rb_g: usize,
    rb_u: usize,
    rb_d: usize,
) -> Result<()> {
    let plan_g = exp_pf::launch_plan(slot_g, nseg, ff)
        .ok_or_else(|| anyhow!("gate の起動形が作れない (nseg={nseg}, ff={ff})"))?;
    let plan_d = exp_pf::launch_plan(slot_d, nseg, e)
        .ok_or_else(|| anyhow!("down の起動形が作れない (nseg={nseg}, e={e})"))?;
    // gate と up (同じ行表・同じ入力)。
    for (f, pw0, pb, rb) in [
        (kernels.fns[slot_g], pg, gb, rb_g),
        (kernels.fns[slot_u], pu, ub, rb_u),
    ] {
        let mut pw = pw0;
        let mut px = xb;
        let mut po = pb;
        let mut ps = seg;
        let mut pr = rows_gu;
        let (mut ns, mut ni, mut no, mut rbb) = (
            nseg as c_int,
            n_in as c_int,
            ff as c_int,
            rb as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut pr as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut no as *mut _ as *mut c_void,
            &mut rbb as *mut _ as *mut c_void,
        ];
        ctx.launch_async(f, plan_g.grid, plan_g.block, 0, &mut args)?;
    }
    // act = silu(g) * u (このグループの行の範囲だけ)。
    {
        let rows_here = ro_end.saturating_sub(ro_start);
        if rows_here > 0 {
            let n = rows_here * ff;
            let threads = 256usize;
            let blocks = n.div_ceil(threads) as u32;
            let mut pgg = gb + (ro_start * ff * 4) as u64;
            let mut puu = ub + (ro_start * ff * 4) as u64;
            let mut paa = act + (ro_start * ff * 4) as u64;
            let mut pn = n as c_int;
            let mut args = [
                &mut pgg as *mut _ as *mut c_void,
                &mut puu as *mut _ as *mut c_void,
                &mut paa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
            ];
            ctx.launch_async(
                kernels.silu,
                (blocks.max(1), 1, 1),
                (threads as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    // down (入力は act の行が詰まった形 = rows_id)。
    {
        let mut pw = pd;
        let mut px = act;
        let mut po = hb;
        let mut ps = seg;
        let mut pr = rows_id;
        let (mut ns, mut ni, mut no, mut rb) = (
            nseg as c_int,
            ff as c_int,
            e as c_int,
            rb_d as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut pr as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut no as *mut _ as *mut c_void,
            &mut rb as *mut _ as *mut c_void,
        ];
        ctx.launch_async(
            kernels.fns[slot_d],
            plan_d.grid,
            plan_d.block,
            0,
            &mut args,
        )?;
    }
    Ok(())
}


/// グループ 1 個ぶんの gate/up と down を tensor core で積む。
///
/// `seg` は**相対化していない**表 (`gl` は使わない)、`rows_gu` は行 → トークン、
/// `rows_id` は down 用の恒等行表。`max_m` はグループ内の最大行数。
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn pf_launch_group_tc(
    ctx: &CudaCtx,
    gu_fn: CUfunction,
    dn_fn: CUfunction,
    pg: CUdeviceptr,
    pu: CUdeviceptr,
    pd: CUdeviceptr,
    xb: CUdeviceptr,
    act: CUdeviceptr,
    hb: CUdeviceptr,
    seg: CUdeviceptr,
    rows_gu: CUdeviceptr,
    rows_id: CUdeviceptr,
    nseg: usize,
    max_m: usize,
    n_in: usize,
    ff: usize,
    e: usize,
    rb_g: usize,
    rb_d: usize,
) -> Result<()> {
    if nseg == 0 || max_m == 0 {
        return Ok(());
    }
    let tiles = max_m.div_ceil(EXTC_BM) as u32;
    let threads = EXTC_THREADS as u32;
    // gate/up: 入力 e、出力 ff。
    {
        let mut pw_g = pg;
        let mut pw_u = pu;
        let mut px = xb;
        let mut po = act;
        let mut ps = seg;
        let mut pr = rows_gu;
        let (mut ns, mut ni, mut no, mut rbg) = (
            nseg as c_int,
            n_in as c_int,
            ff as c_int,
            rb_g as c_int,
        );
        let mut args = [
            &mut pw_g as *mut _ as *mut c_void,
            &mut pw_u as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut pr as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut no as *mut _ as *mut c_void,
            &mut rbg as *mut _ as *mut c_void,
        ];
        ctx.launch_async(
            gu_fn,
            ((ff / EXTC_BN) as u32, tiles, nseg as u32),
            (threads, 1, 1),
            0,
            &mut args,
        )?;
    }
    // down: 入力 ff、出力 e。入力の行は act が詰まっている (恒等行表)。
    {
        let mut pw = pd;
        let mut px = act;
        let mut po = hb;
        let mut ps = seg;
        let mut pr = rows_id;
        let (mut ns, mut ni, mut no, mut rbd) = (
            nseg as c_int,
            ff as c_int,
            e as c_int,
            rb_d as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut ps as *mut _ as *mut c_void,
            &mut pr as *mut _ as *mut c_void,
            &mut ns as *mut _ as *mut c_void,
            &mut ni as *mut _ as *mut c_void,
            &mut no as *mut _ as *mut c_void,
            &mut rbd as *mut _ as *mut c_void,
        ];
        ctx.launch_async(
            dn_fn,
            ((e / EXTC_BN) as u32, tiles, nseg as u32),
            (threads, 1, 1),
            0,
            &mut args,
        )?;
    }
    Ok(())
}


#[cfg(test)]
mod grouped_tests {
    //! P40: 起動形の純関数 (GPU 無しでも検証できる部分)。
    use super::*;

    #[test]
    fn launch_plan_sizes_smem_for_two_matrices() {
        // Q4_K (blkj=256) の gate+up: 16 行 × 256 × 4 B × 2 本 = 32 KiB
        let p = grouped_launch_plan(86, 60, 640, 256, 2)
            .expect("plan")
            .expect("some");
        assert_eq!(p.grid, (86, 40, 4)); // 640/16 = 40 行タイル、ceil(60/16) = 4
        assert_eq!(p.block, (16, 16, 1));
        assert_eq!(p.smem, 16 * 256 * 4 * 2);
        assert!(p.smem < 46_000, "48 KiB 既定を超えないこと");
        // Q5_1 (blkj=32) の down: 16 × 32 × 4 = 2 KiB
        let q = grouped_launch_plan(86, 60, 2560, 32, 1)
            .expect("plan")
            .expect("some");
        assert_eq!(q.grid, (86, 160, 4));
        assert_eq!(q.smem, 16 * 32 * 4);
    }

    #[test]
    fn launch_plan_rejects_oversized_smem_and_empty() {
        assert!(grouped_launch_plan(0, 10, 640, 256, 2)
            .expect("plan")
            .is_none());
        assert!(grouped_launch_plan(8, 0, 640, 256, 2)
            .expect("plan")
            .is_none());
        // 3 本同時 (Q4_K) は 48 KiB で既定を超える = 過去に全滅した形
        assert!(grouped_launch_plan(8, 10, 640, 256, 3)
            .expect("plan")
            .is_none());
        // 32 行タイル x 256 要素 x 4 B x 2 = 64 KiB も落ちる
        assert!(grouped_launch_plan(8, 10, 640, 512, 2)
            .expect("plan")
            .is_none());
    }

    #[test]
    fn env_value_parsing() {
        assert!(moe_grouped_gpu_from_value(Some("1")));
        assert!(moe_grouped_gpu_from_value(Some("on")));
        assert!(moe_grouped_gpu_from_value(Some("true")));
        assert!(!moe_grouped_gpu_from_value(None));
        assert!(!moe_grouped_gpu_from_value(Some("0")));
        assert!(!moe_grouped_gpu_from_value(Some("off")));
        assert!(!moe_grouped_gpu_from_value(Some("")));
    }
}


/// task 155: `DENNO_ROUTER_FUSED` の値解釈 (純関数・テスト用)。
/// **既定 on**、`0` / `off` / `false` でルータの待ちを従来 (cuCtxSynchronize) に戻す。
pub fn router_fused_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}


/// task 155: ルータの待ちを null ストリームのイベントに絞るか (`DENNO_ROUTER_FUSED`)。
pub fn router_fused_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| router_fused_from_value(std::env::var("DENNO_ROUTER_FUSED").ok().as_deref()))
}


/// s74 (T6 候補 1): dno_gemm_i8 の j 方向タイル (256) に合わせた重み行の切り上げ。
/// 旧 128 タイル版は端をカーネル側で 0 埋めしていたが、新カーネルは A 行を
/// 2 枚の B タイルで共有するため A/B の pad を非対称にできる重み側だけを
/// ホストで 0 埋めする (i8 の 0 は積に寄与しない → dot は厳密に不変)。
pub(crate) fn gemm_pad_n(n: usize) -> usize {
    n.div_ceil(GEMM_PAD_N) * GEMM_PAD_N
}


/// s74: 重み [n,k] i8 (row-major) を `gemm_pad_n` 行までラッパ行で 0 埋めする。
/// 既に倍数なら複製しない (VRAM/RAM の余分なコピーを避ける)。
pub(crate) fn gemm_pad_apply(qw: &[i8], k: usize, n: usize) -> std::borrow::Cow<'_, [i8]> {
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


/// P17i: 大きい m (prefill) 用の tiled GEMM 経路を有効にするか。
/// `DENNO_CUDA_GEMM=0` で無効化 (従来どおり CPU へ)。
pub(crate) fn gemm_enabled() -> bool {
    std::env::var("DENNO_CUDA_GEMM").ok().as_deref() != Some("0")
}


/// P17i: tiled GEMM を発動する最小の m。小さい m では GEMM 経路の層あたり
/// sync (H2D/quantize/gemm/D2H × 4 層/層) が CPU GEMM より高くつくため、
/// ある程度大きい m から発動させる (m=38 の短文 prefill は CPU が速い)。
/// `DENNO_CUDA_GEMM_MIN_M` で変更可。0 にすると m>max_m() で常に発動。
pub(crate) fn gemm_min_m() -> usize {
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


pub(crate) fn weights_for_group(
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
pub(crate) static Q4_ALIAS: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, (u64, usize)>>> =
    OnceLock::new();


/// s178 (P59): 連結重みと分割重みの**同居**を避ける (重複排除) か。既定 ON。
///
/// `DENNO_Q4_DEDUP=0` で「エイリアスを使わない」= 各 uid を個別に転送する
/// 従来動作へ戻せる。数値は経路によらず同一 (同じ重み・同じカーネル・同じ
/// 累積順) で、変わるのは VRAM と H2D の量だけ。
pub(crate) fn q4_dedup_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_q4_dedup(std::env::var("DENNO_Q4_DEDUP").ok().as_deref()))
}


/// s165: `Linear` から Q4 本体を取り出す (連結エイリアスの前提検査用)。
pub(crate) fn q4_of(l: &crate::quant::Linear) -> Option<&crate::quant::Q4Linear> {
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


/// s166 (P5): 連結重み `fused_uid` の行範囲が `parts` に割り当て済みか
/// (行オフセットが [0, n0, n0+n1, ...] と一致するか) を照会する。
pub(crate) fn q4_fused_ready(fused_uid: u64, parts: &[&crate::quant::Q4Linear]) -> bool {
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


/// s178 (P59): 連結重み `[tn 行 × tk 列]` の行範囲 `[row_off, row_off+n)` を
/// `[n 行 × k 列]` のビューとして切り出すための**バイト幾何** (純粋関数・GPU 不要)。
///
/// 連結側の nib は 1 行 nb (= k/32) ブロック × 16 B、sc は 1 行 nb 本 × 4 B (f32)。
/// 行は連結側で連続しているので、行オフセットはそのままバイトオフセットになる。
///
/// 契約: `k` が 32 の倍数・`tk == k`・`row_off + n <= tn` (範囲が連結側に収まる)。
/// **`<=` が要点**: 先頭の part は `row_off = 0` なので `row_off + n != tn` に
/// なり、旧実装 (`!=`) では末尾の part しかビューにならず先頭の part が VRAM へ
/// 二重に載っていた (P59 の実測で 465 本 / 22,818.6 MB)。満たさない入力は `None`
/// = 呼び出し側は通常の転送へ落ちる (fail closed)。
pub(crate) fn q4_view_geo(
    k: usize,
    n: usize,
    row_off: usize,
    tk: usize,
    tn: usize,
) -> Option<(usize, usize, usize, usize)> {
    if k == 0 || k % 32 != 0 || n == 0 || tk != k {
        return None;
    }
    if row_off.checked_add(n)? > tn {
        return None;
    }
    let nb = k / 32;
    let nib_off = row_off.checked_mul(nb)?.checked_mul(16)?;
    let sc_off = row_off.checked_mul(nb)?.checked_mul(4)?;
    let nib_bytes = n.checked_mul(nb)?.checked_mul(16)?;
    let sc_bytes = n.checked_mul(nb)?.checked_mul(4)?;
    Some((nib_off, nib_bytes, sc_off, sc_bytes))
}


/// s123: Q4_0 重みのデバイスレジストリ (uid → nib/sc)。s166 (P5) で関数内
/// static から出した (`q4_resident` が同じ表を照会するため)。
pub(crate) fn q4_registry() -> &'static std::sync::Mutex<std::collections::HashMap<u64, Box<GpuWq4>>> {
    static REG: OnceLock<std::sync::Mutex<std::collections::HashMap<u64, Box<GpuWq4>>>> =
        OnceLock::new();
    REG.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}


/// s166 (P5): `uid` が既にデバイスへ載っているか (**照会のみ・転送しない**)。
/// 連結重みを上げる前に「parts が個別に載っていないか」を確かめるために使う
/// (載っているのに連結側も上げると VRAM が 2 重になる)。ロックが毒されていたら
/// `true` = 「載っている」とみなして連結側を上げない (安全側)。
pub(crate) fn q4_resident(uid: u64) -> bool {
    match q4_registry().lock() {
        Ok(m) => m.contains_key(&uid),
        Err(_) => true,
    }
}


/// s169 (P10): `parts` のどれかが既に VRAM へ載っているか (**照会のみ・転送しない**)。
/// ホスト prefill 経路の q|k|v 連結エイリアス登録で「載る前に登録する」順序を
/// 守るために使う (載っているなら登録しない = VRAM を 2 重にしない)。
pub fn q4_any_resident(parts: &[&crate::quant::Linear]) -> bool {
    parts.iter().filter_map(|p| q4_of(p)).any(|p| q4_resident(p.uid))
}


/// 入口。計数を付けて [`gemm_try`] に渡す (再帰の内側は数えない)。
pub fn maybe_gpu_gguf_gemm_into(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
    nt: usize,
    out: &mut [f32],
) -> bool {
    // devprof: 1 呼び出しぶんの時間・転送量・同期待ちを op "gemm" に積む。
    // `nt > 128` は `gemm_try` の内側で分割されるが、ここで数えるのは
    // 呼び出し側から見た 1 回ぶん (H2D = nt * k * 4, D2H = nt * n * 4)。
    let prof = crate::model::devprof::on();
    let t0 = prof.then(std::time::Instant::now);
    let up_before = if prof {
        crate::gpu_budget::weight_upload_bytes()
    } else {
        0
    };
    if prof {
        crate::model::devprof::clear();
    }
    let ok = gemm_try(uid, ty, blocks, k, n, x, nt, out);
    let c = if ok { &GEMM_USED } else { &GEMM_SKIPPED };
    c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let (true, Some(t0)) = (ok, t0) {
        let up_w = crate::gpu_budget::weight_upload_bytes().saturating_sub(up_before);
        let (_, _, sync_ns) = crate::model::devprof::take_pending();
        crate::model::devprof::add(
            "gemm",
            t0.elapsed().as_nanos() as u64,
            (nt * k * 4) as u64 + up_w,
            (nt * n * 4) as u64,
            sync_ns,
        );
    }
    ok
}


pub(crate) fn gemm_try(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
    nt: usize,
    out: &mut [f32],
) -> bool {
    if !gpu_gemm_enabled() {
        return skip(SKIP_OFF);
    }
    if !cuda_enabled() {
        return skip(SKIP_NO_CUDA);
    }
    if nt <= 1 {
        return skip(SKIP_NT);
    }
    if gguf_row_bytes(ty, k).is_none() {
        return skip(SKIP_TYPE);
    }
    // カーネルは 1 起動で 128 トークンまで (GEMM_ROWS × GEMM_NTG_MAX = 16 × 8)。
    // それより大きい chunk はここで割る (重みは VRAM 常駐なので読み直しは
    // VRAM 内で済む)。task 169: テンソルコア GEMM が使えるときは 1 起動
    // 2,048 トークンまでまとめる (起動・同期・H2D/D2H の回数が 1/16 になる。
    // プリフィルのチャンクは 512 なので実際は 1 回)。
    let tile = gemm_try_tile_for(
        pf_tc_gemm_on(),
        pf_tc2_gemm_on(),
        ty,
        nt,
        k,
        n,
        pf_tc_min_m(),
    );
    if nt > tile {
        let mut t0 = 0usize;
        while t0 < nt {
            let t1 = (t0 + tile).min(nt);
            let m = t1 - t0;
            let ok = gemm_try(
                uid,
                ty,
                blocks,
                k,
                n,
                &x[t0 * k..t1 * k],
                m,
                &mut out[t0 * n..t1 * n],
            );
            if !ok {
                return false; // 途中で落ちたら呼び出し側が CPU で全部やり直す
            }
            t0 = t1;
        }
        return true;
    }
    if nt <= 1 {
        return skip(SKIP_NT);
    }
    let Some(ctx) = shared_ctx() else {
        return skip(SKIP_NO_CTX);
    };
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed, using CPU gguf gemm: {e:#}");
        return skip(SKIP_ERR);
    }
    let w = match weights_for_gguf(uid, ty, blocks, k, n) {
        Ok(w) => w,
        Err(e) => {
            let msg = format!("{e:#}");
            if let Some(true) = crate::gpu_budget::note_weight_failure(uid, &msg) {
                tracing::error!("gguf weight register failed (gemm, uid={uid}): {msg}");
            }
            return skip(SKIP_NO_WEIGHT);
        }
    };
    match ctx.moe_gemm_dev(ty, &w.buf, x, nt, k, n, out) {
        Ok(true) => true,
        Ok(false) => skip(SKIP_KERNEL),
        Err(e) => {
            tracing::error!("gguf gemm failed (uid={uid}): {e:#}");
            skip(SKIP_ERR)
        }
    }
}

// 共有メモリはカーネル側の静的配列 (`__shared__`) なので、ホストからは
// サイズを渡さない (取り違えようがない)。

/// `DENNO_PF_TC_GEMM` の値の解釈 (純関数)。`0`・`off`・`false`・`no`・空文字だけが off。
///
/// **既定 on** (lab t169a で採用): 14,018 トークンの cold prefill が
/// 96.55 → 70.68 s (-26.8%)、3,826 で 27.69 → 20.88 s (-24.6%)、decode と
/// 19 トークン next turn は不変、errors 0・text sane。md5 は BF16 丸めで変わる。
pub(crate) fn pf_tc_gemm_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// `DENNO_PF_TC_MIN_M` の値の解釈 (純関数)。0 や壊れた値は既定 64。
pub(crate) fn pf_tc_min_m_from_value(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}


/// テンソルコア GEMM を使うか (**既定 off**)。
pub(crate) fn pf_tc_gemm_on() -> bool {
    #[cfg(test)]
    {
        match PF_TC_GEMM_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_tc_gemm_from_value(std::env::var("DENNO_PF_TC_GEMM").ok().as_deref()))
}


/// テンソルコア GEMM を使う最小の行数 (`DENNO_PF_TC_MIN_M`、既定 64)。
///
/// 19 トークンの next turn prefill のような小さい M は、従来カーネルの方が
/// 起動もタイル効率も良いので切り替えない (decode と next turn を動かさない)。
pub(crate) fn pf_tc_min_m() -> usize {
    static MIN_M: OnceLock<usize> = OnceLock::new();
    *MIN_M.get_or_init(|| pf_tc_min_m_from_value(std::env::var("DENNO_PF_TC_MIN_M").ok().as_deref()))
}


/// この形をテンソルコア GEMM で回せるか (純関数。カーネルの前提と同じ)。
///
/// `kernels_gemm_tc.cu` の前提: Q8_0 / `n_in % 32 == 0` / `n_out % 64 == 0`。
/// 型は呼び出し側が Q8_0 に限る (このモデルの射影は全部 Q8_0)。
pub(crate) fn pf_tc_gemm_eligible(nt: usize, n_in: usize, n_out: usize, min_m: usize) -> bool {
    // 呼び出し側 (`gemm_try` は nt>1、GDN 連鎖は nt>=2) はどれも 2 以上でしか
    // 来ない。nt=1 はカーネルとしても正しく動く (行が 1 本残るだけ) ので、
    // 下限は `DENNO_PF_TC_MIN_M` だけで決める。
    nt >= min_m.max(1)
        && n_in > 0
        && n_in % TC_BK == 0
        && n_out >= TC_BN
        && n_out % TC_BN == 0
}


/// `gemm_try` が 1 起動でまとめるトークン数 (純関数)。
///
/// 従来カーネルは 128 トークンまで (GEMM_ROWS × GEMM_NTG_MAX) なので、それより
/// 大きいチャンクはここで割る。テンソルコア GEMM が使えるときだけ
/// [`TC_MAX_TOKENS`] までまとめる (起動・同期・H2D/D2H の回数が減る)。
fn gemm_try_tile_for(
    tc_on: bool,
    tc2_on: bool,
    ty: crate::kquant::GgmlType,
    nt: usize,
    k: usize,
    n: usize,
    min_m: usize,
) -> usize {
    let tc = tc_on
        && matches!(ty, crate::kquant::GgmlType::Q8_0)
        && pf_tc_gemm_eligible(nt, k, n, min_m);
    // task 176: v2 カーネルも 2,048 トークンまで 1 起動で畳める (Q4_K / Q5_K も含む)。
    let tc2 = tc2_on && pf_tc2_eligible(ty, nt, k, n, min_m);
    // task 181: N が小さい形 (shared 640 / out 2560) も BN=64 のカーネルで
    // 2,048 トークンまで畳める (ゲートは env。既定 off)。
    let sn = pf_tc_smalln_on() && pf_tc_smalln_eligible(ty, nt, k, n, min_m);
    if tc || tc2 || sn {
        TC_MAX_TOKENS
    } else {
        128
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける
/// (= 二度と試さない)。**current context を張った後に呼ぶこと**。
pub(crate) fn tc_gemm_kernels() -> Option<&'static TcGemm> {
    TC_GEMM
        .get_or_init(|| {
            let path = tc_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-tc] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-tc] cuModuleLoadData 失敗 → 従来の GEMM へ");
                    return None;
                }
                // 変数名 `q8_0` は kernel_invariants の起動サイト表
                // (`module_get_function(&mut X, ..)` と `fns.X` の対応) が読む。
                let mut q8_0: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(
                    &mut q8_0,
                    module,
                    b"dno_gemm_tc_q8_0_w\0".as_ptr() as *const c_char,
                ) != CUDA_SUCCESS
                {
                    eprintln!("[pf-tc] PTX に dno_gemm_tc_q8_0_w が無い → 従来の GEMM へ");
                    return None;
                }
                Some(TcGemm { module, q8_0 })
            }
        })
        .as_ref()
}


/// テンソルコア GEMM を `TC_MAX_TOKENS` 行ずつ起動する (w / x / out はデバイス)。
pub(crate) fn tc_gemm_launch(
    ctx: &CudaCtx,
    fns: &TcGemm,
    wptr: CUdeviceptr,
    xptr: CUdeviceptr,
    optr: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
) -> Result<()> {
    if nt == 0 {
        return Ok(());
    }
    let grid_x = (n_out / TC_BN) as u32;
    let mut t0 = 0usize;
    while t0 < nt {
        let m = (nt - t0).min(TC_MAX_TOKENS);
        let grid_y = m.div_ceil(TC_BM) as u32;
        let (mut pw, mut px, mut po) = (
            wptr,
            xptr + (t0 * n_in * 4) as CUdeviceptr,
            optr + (t0 * n_out * 4) as CUdeviceptr,
        );
        let (mut m_c, mut ni_c, mut no_c, mut rb_c) = (
            m as c_int,
            n_in as c_int,
            n_out as c_int,
            row_bytes as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut m_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(fns.q8_0, (grid_x, grid_y, 1), (256, 1, 1), 0, &mut args)?;
        }
        t0 += m;
    }
    Ok(())
}


/// `DENNO_PF_TC_GEMM2` の値の解釈 (純関数)。`0`・`off`・`false`・`no`・空文字だけが off。
///
/// **既定 on** (lab t176a で採用): 14,018 トークンの cold prefill が
/// 72.28 → 63.05 s (-12.8%)、3,826 で 20.57 → 18.57 s (-9.7%)、ssm の
/// qkv_pf が 6,268 → 579 ms (-90.8%)、out_pf が 667 → 271 ms、decode と
/// 19 トークン next turn は不変、errors 0・text sane。md5 は BF16 丸め
/// (と 4 層の Q5_K/Q4_K が f32 経路から TC 経路へ移ること) で変わる。
/// br178 (int-auto `5d2238843` に br176 をマージした木) の lab t178a でも
/// 同じ: 3,826 で 19.76 → 17.83 s (-9.8%)、14,018 で 66.65 → 59.02 s
/// (-11.4%)、decode 27 → 27 ms/tok、errors 0、text sane。
pub(crate) fn pf_tc2_gemm_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// v2 カーネルを使うか (**既定 on**。`DENNO_PF_TC_GEMM2=0` で切る)。
pub(crate) fn pf_tc2_gemm_on() -> bool {
    #[cfg(test)]
    {
        match PF_TC2_GEMM_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| pf_tc2_gemm_from_value(std::env::var("DENNO_PF_TC_GEMM2").ok().as_deref()))
}


/// v2 カーネルが扱える型か (純関数)。
pub(crate) fn pf_tc2_type_ok(ty: crate::kquant::GgmlType) -> bool {
    use crate::kquant::GgmlType as T;
    matches!(ty, T::Q8_0 | T::Q4K | T::Q5K)
}


/// v2 カーネルの k の前提 (純関数)。Q8_0 は 32、K-quant は 256 の倍数。
pub(crate) fn pf_tc2_k_ok(ty: crate::kquant::GgmlType, n_in: usize) -> bool {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Q8_0 => n_in > 0 && n_in % TC2_BK == 0,
        T::Q4K | T::Q5K => n_in > 0 && n_in % 256 == 0,
        _ => false,
    }
}


/// 立つブロック数 (純関数。診断・テスト用)。
#[cfg(test)]
pub(crate) fn pf_tc2_blocks(nt: usize, n_out: usize) -> usize {
    if n_out % TC2_BN != 0 {
        return 0;
    }
    (n_out / TC2_BN) * nt.div_ceil(TC2_BM)
}


/// この形を v2 カーネルで回せるか (純関数。カーネルの前提と同じ)。
///
/// 型 (Q8_0 / Q4_K / Q5_K)・k の倍数・出力行 128 の倍数・`N >= TC2_MIN_N`。
/// 小さい N (out_proj 2560 / shared expert 640) はブロック数が足りず v2 の
/// 利点が無いので br169 に任せる。
pub(crate) fn pf_tc2_eligible(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
    min_m: usize,
) -> bool {
    pf_tc2_type_ok(ty)
        && pf_tc2_k_ok(ty, n_in)
        && nt >= min_m.max(1)
        && n_out >= TC2_MIN_N
        && n_out % TC2_BN == 0
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける
/// (= 二度と試さない)。**current context を張った後に呼ぶこと**。
pub(crate) fn tc2_gemm_kernels() -> Option<&'static Tc2Gemm> {
    TC2_GEMM
        .get_or_init(|| {
            let path = tc2_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-tc2] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-tc2] cuModuleLoadData 失敗 → 従来の GEMM へ");
                    return None;
                }
                // 変数名は kernel_invariants の起動サイト表 (`module_get_function` と
                // `fns.X` の対応) が読む。
                let mut q8_0: CUfunction = std::ptr::null_mut();
                let mut q4k: CUfunction = std::ptr::null_mut();
                let mut q5k: CUfunction = std::ptr::null_mut();
                for (dst, name, label) in [
                    (
                        &mut q8_0,
                        b"dno_gemm_tc2_q8_0_w\0".as_ptr(),
                        "dno_gemm_tc2_q8_0_w",
                    ),
                    (
                        &mut q4k,
                        b"dno_gemm_tc2_q4k_w\0".as_ptr(),
                        "dno_gemm_tc2_q4k_w",
                    ),
                    (
                        &mut q5k,
                        b"dno_gemm_tc2_q5k_w\0".as_ptr(),
                        "dno_gemm_tc2_q5k_w",
                    ),
                ] {
                    if (lib.module_get_function)(dst, module, name as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[pf-tc2] PTX に {label} が無い → 従来の GEMM へ");
                        return None;
                    }
                }
                Some(Tc2Gemm {
                    module,
                    q8_0,
                    q4k,
                    q5k,
                })
            }
        })
        .as_ref()
}


/// v2 カーネルを `TC2_MAX_TOKENS` 行ずつ起動する (w / x / out はデバイス)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn tc2_gemm_launch(
    ctx: &CudaCtx,
    fns: &Tc2Gemm,
    ty: crate::kquant::GgmlType,
    wptr: CUdeviceptr,
    xptr: CUdeviceptr,
    optr: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
) -> Result<()> {
    if nt == 0 {
        return Ok(());
    }
    let Some(f) = tc2_func(fns, ty) else {
        bail!("tc2_gemm_launch: 未対応の型 {ty:?}");
    };
    let grid_x = (n_out / TC2_BN) as u32;
    let mut t0 = 0usize;
    while t0 < nt {
        let m = (nt - t0).min(TC2_MAX_TOKENS);
        let grid_y = m.div_ceil(TC2_BM) as u32;
        let (mut pw, mut px, mut po) = (
            wptr,
            xptr + (t0 * n_in * 4) as CUdeviceptr,
            optr + (t0 * n_out * 4) as CUdeviceptr,
        );
        let (mut m_c, mut ni_c, mut no_c, mut rb_c) = (
            m as c_int,
            n_in as c_int,
            n_out as c_int,
            row_bytes as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut m_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(f, (grid_x, grid_y, 1), (256, 1, 1), 0, &mut args)?;
        }
        t0 += m;
    }
    Ok(())
}


/// `DENNO_PF_TC_SMALLN` の値の解釈 (純関数)。明示的な偽値だけが off。
///
/// **既定 on** (lab t181b で採用): 単独で cold prefill 3,826 tok 17.80 → 17.20 s
/// (-3.4%)、14,018 tok 59.08 → 57.36 s (-2.9%)。shared expert の gpu が
/// 3,508 → 1,276 ms。decode と 19 トークン next turn は不変、errors 0。
/// md5 は BF16 丸めで変わる (prefill の射影は契約外)。
pub(crate) fn pf_tc_smalln_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        None => true,
        Some(s) => !matches!(s.as_str(), "0" | "off" | "false" | "no" | ""),
    }
}


/// 小さい N 用のカーネルを使うか (**既定 off**。`DENNO_PF_TC_SMALLN=1` で on)。
pub(crate) fn pf_tc_smalln_on() -> bool {
    #[cfg(test)]
    {
        match PF_TC_SN_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_tc_smalln_from_value(std::env::var("DENNO_PF_TC_SMALLN").ok().as_deref())
    })
}


/// 小さい N 用カーネルが扱える型か (純関数)。Q4_K を含むのが要点
/// (shared expert の gate は Q4_K。br169 は Q8_0 専用)。
pub(crate) fn pf_tc_smalln_type_ok(ty: crate::kquant::GgmlType) -> bool {
    use crate::kquant::GgmlType as T;
    matches!(ty, T::Q8_0 | T::Q4K)
}


/// この形を小さい N 用カーネルで回せるか (純関数。カーネルの前提と同じ)。
///
/// 型 (Q8_0 / Q4_K)・k の倍数・`n_out % 64 == 0`・`n_out < TC2_MIN_N`。
/// `N >= 4096` は tc2 の担当 (128 行タイルの方が仕事が 2 倍 = 固定費が半分)。
pub(crate) fn pf_tc_smalln_eligible(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
    min_m: usize,
) -> bool {
    pf_tc_smalln_type_ok(ty)
        && pf_tc2_k_ok(ty, n_in)
        && nt >= min_m.max(1)
        && n_out > 0
        && n_out % TC_SN_BN == 0
        && n_out < TC2_MIN_N
}


/// 立つブロック数 (純関数。診断・テスト用)。
#[cfg(test)]
pub(crate) fn pf_tc_smalln_blocks(nt: usize, n_out: usize) -> usize {
    if n_out % TC_SN_BN != 0 {
        return 0;
    }
    (n_out / TC_SN_BN) * nt.div_ceil(TC_SN_BM)
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
pub(crate) fn tc_smalln_kernels() -> Option<&'static SnGemm> {
    SN_GEMM
        .get_or_init(|| {
            let path = sn_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-sn] PTX が読めない ({}) : {e} → 従来の GEMM へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0); // cuModuleLoadData は NUL 終端の PTX を期待する
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-sn] cuModuleLoadData 失敗 → 従来の GEMM へ");
                    return None;
                }
                // 変数名は kernel_invariants の起動サイト表 (`module_get_function` と
                // `fns.X` の対応) が読む。
                let mut q8_0: CUfunction = std::ptr::null_mut();
                let mut q4k: CUfunction = std::ptr::null_mut();
                for (dst, name, label) in [
                    (
                        &mut q8_0,
                        b"dno_gemm_sn_q8_0_w\0".as_ptr(),
                        "dno_gemm_sn_q8_0_w",
                    ),
                    (
                        &mut q4k,
                        b"dno_gemm_sn_q4k_w\0".as_ptr(),
                        "dno_gemm_sn_q4k_w",
                    ),
                ] {
                    if (lib.module_get_function)(dst, module, name as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[pf-sn] PTX に {label} が無い → 従来の GEMM へ");
                        return None;
                    }
                }
                Some(SnGemm {
                    module,
                    q8_0,
                    q4k,
                })
            }
        })
        .as_ref()
}


/// 小さい N 用カーネルを `TC_SN_MAX_TOKENS` 行ずつ起動する (w / x / out はデバイス)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn sn_gemm_launch(
    ctx: &CudaCtx,
    fns: &SnGemm,
    ty: crate::kquant::GgmlType,
    wptr: CUdeviceptr,
    xptr: CUdeviceptr,
    optr: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
) -> Result<()> {
    if nt == 0 {
        return Ok(());
    }
    let Some(f) = sn_func(fns, ty) else {
        bail!("sn_gemm_launch: 未対応の型 {ty:?}");
    };
    let grid_x = (n_out / TC_SN_BN) as u32;
    let mut t0 = 0usize;
    while t0 < nt {
        let m = (nt - t0).min(TC_SN_MAX_TOKENS);
        let grid_y = m.div_ceil(TC_SN_BM) as u32;
        let (mut pw, mut px, mut po) = (
            wptr,
            xptr + (t0 * n_in * 4) as CUdeviceptr,
            optr + (t0 * n_out * 4) as CUdeviceptr,
        );
        let (mut m_c, mut ni_c, mut no_c, mut rb_c) = (
            m as c_int,
            n_in as c_int,
            n_out as c_int,
            row_bytes as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut m_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(f, (grid_x, grid_y, 1), (256, 1, 1), 0, &mut args)?;
        }
        t0 += m;
    }
    Ok(())
}


/// `DENNO_PF_TC_SHARED2` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (lab t183a で採用)。off のときは sn (BM=128) のまま。
/// lab t183a (depth8k、14,018 トークン) の実測: sn2 単独で shared の gpu が
/// 1,201.0 → 1,143.5 ms (-4.8%)、cold prefill 51.58 → 50.11 s、**md5 は base と
/// 完全一致** (sn と同じ k 昇順のアキュムレートなので値が変わらない)。
pub(crate) fn pf_tc_shared2_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


/// shared expert の射影を BM=64 のカーネルで回すか (**既定 on**。`0` で切る)。
pub(crate) fn pf_tc_shared2_on() -> bool {
    #[cfg(test)]
    {
        match PF_TC_SN2_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_tc_shared2_from_value(std::env::var("DENNO_PF_TC_SHARED2").ok().as_deref())
    })
}


/// この形を BM=64 のカーネルで回せるか (純関数。sn と同じ条件 + 行数)。
///
/// `nt >= TC_SN2_BM` を要求する: それ未満 (19 トークンの next turn prefill など)
/// ではタイルが埋まらず、sn (BM=128) と仕事量が変わらないため。
pub(crate) fn pf_tc_smalln2_eligible(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
    min_m: usize,
) -> bool {
    pf_tc_smalln_type_ok(ty)
        && pf_tc2_k_ok(ty, n_in)
        && nt >= min_m.max(TC_SN2_BM)
        && n_out > 0
        && n_out % TC_SN2_BN == 0
        && n_out < TC2_MIN_N
}


/// 立つブロック数 (純関数。診断・テスト用)。
#[cfg(test)]
pub(crate) fn pf_tc_smalln2_blocks(nt: usize, n_out: usize) -> usize {
    if n_out % TC_SN2_BN != 0 {
        return 0;
    }
    (n_out / TC_SN2_BN) * nt.div_ceil(TC_SN2_BM)
}


/// PTX の置き場所。`DENNO_PF_SN2_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gemm_sn2.ptx`。
fn sn2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SN2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_sn2.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
fn tc_smalln2_kernels() -> Option<&'static Sn2Gemm> {
    SN2_GEMM
        .get_or_init(|| {
            let path = sn2_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-sn2] PTX が読めない ({}) : {e} → sn / 従来の GEMM へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0);
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-sn2] cuModuleLoadData 失敗 → sn / 従来の GEMM へ");
                    return None;
                }
                let mut q8_0: CUfunction = std::ptr::null_mut();
                let mut q4k: CUfunction = std::ptr::null_mut();
                for (dst, name, label) in [
                    (
                        &mut q8_0,
                        b"dno_gemm_sn2_q8_0_w\0".as_ptr(),
                        "dno_gemm_sn2_q8_0_w",
                    ),
                    (
                        &mut q4k,
                        b"dno_gemm_sn2_q4k_w\0".as_ptr(),
                        "dno_gemm_sn2_q4k_w",
                    ),
                ] {
                    if (lib.module_get_function)(dst, module, name as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[pf-sn2] PTX に {label} が無い → sn / 従来の GEMM へ");
                        return None;
                    }
                }
                Some(Sn2Gemm {
                    module,
                    q8_0,
                    q4k,
                })
            }
        })
        .as_ref()
}


/// 型ごとのカーネル (純関数)。
fn sn2_func(fns: &Sn2Gemm, ty: crate::kquant::GgmlType) -> Option<CUfunction> {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Q8_0 => Some(fns.q8_0),
        T::Q4K => Some(fns.q4k),
        _ => None,
    }
}


/// BM=64 のカーネルを `TC_SN2_MAX_TOKENS` 行ずつ起動する (w / x / out はデバイス)。
#[allow(clippy::too_many_arguments)]
fn sn2_gemm_launch(
    ctx: &CudaCtx,
    fns: &Sn2Gemm,
    ty: crate::kquant::GgmlType,
    wptr: CUdeviceptr,
    xptr: CUdeviceptr,
    optr: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
) -> Result<()> {
    if nt == 0 {
        return Ok(());
    }
    let Some(f) = sn2_func(fns, ty) else {
        bail!("sn2_gemm_launch: 未対応の型 {ty:?}");
    };
    let grid_x = (n_out / TC_SN2_BN) as u32;
    let mut t0 = 0usize;
    while t0 < nt {
        let m = (nt - t0).min(TC_SN2_MAX_TOKENS);
        let grid_y = m.div_ceil(TC_SN2_BM) as u32;
        let (mut pw, mut px, mut po) = (
            wptr,
            xptr + (t0 * n_in * 4) as CUdeviceptr,
            optr + (t0 * n_out * 4) as CUdeviceptr,
        );
        let (mut m_c, mut ni_c, mut no_c, mut rb_c) = (
            m as c_int,
            n_in as c_int,
            n_out as c_int,
            row_bytes as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut m_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(f, (grid_x, grid_y, 1), (256, 1, 1), 0, &mut args)?;
        }
        t0 += m;
    }
    Ok(())
}


/// shared expert 用の GEMM 入口 (task 183)。
///
/// `DENNO_PF_TC_SHARED3` (task 222) が on なら Q5_K / BF16 も扱える sn3、
/// 無ければ `DENNO_PF_TC_SHARED2` の sn2、それも無ければ従来の [`pf2_gemm`]。
/// 同期はしない。
#[allow(clippy::too_many_arguments)]
pub(crate) fn pf2_gemm_shared(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    kind: PfWKind,
    x: CUdeviceptr,
    y: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    if let PfWKind::Quant(ty, _, rb, _) = kind {
        // task 222: N が大きい側 (shared の down 2560) は tc2 の BM=128 タイルへ。
        // gate/up の 640 は 5 行タイルしか出ないので sn2 / sn3 のまま。
        if pf_tc_shared4_on() && pf_tc_shared4_eligible(ty, nt, n_in, n_out, 1) {
            if let Some(tc2) = tc2_gemm_kernels() {
                return tc2_gemm_launch(ctx, tc2, ty, w, x, y, nt, n_in, n_out, rb);
            }
        }
        if pf_tc_shared3_on() && pf_tc_sn3_eligible(ty, nt, n_in, n_out, 1) {
            if let Some(sn3) = tc_sn3_kernels() {
                return sn3_gemm_launch(ctx, sn3, ty, w, x, y, nt, n_in, n_out, rb);
            }
        }
        if pf_tc_shared2_on() && pf_tc_smalln2_eligible(ty, nt, n_in, n_out, 1) {
            if let Some(sn2) = tc_smalln2_kernels() {
                return sn2_gemm_launch(ctx, sn2, ty, w, x, y, nt, n_in, n_out, rb);
            }
        }
    }
    pf2_gemm(ctx, w, kind, x, y, nt, n_in, n_out)
}


/// `DENNO_PF_TC_SHARED4` の値の解釈 (純関数)。明示的な真値だけ on。
///
/// **既定 off** (lab t222b で採否を決める)。
pub(crate) fn pf_tc_shared4_from_value(v: Option<&str>) -> bool {
    matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}


/// shared expert の down (N=2560) を tc2 (BM=128 / BN=128) で回すか
/// (**既定 off**。`1` で on)。
pub(crate) fn pf_tc_shared4_on() -> bool {
    #[cfg(test)]
    {
        match PF_TC_SN4_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_tc_shared4_from_value(std::env::var("DENNO_PF_TC_SHARED4").ok().as_deref())
    })
}


/// shared の down を tc2 に任せられる形か (純関数)。
///
/// tc2 の前提 (`pf_tc2_eligible`) から **N の下限 4096 だけを外す**。
/// gate/up の N=640 は BN=128 で 5 行タイルしか出ず並列度が落ちるので対象外
/// (下限 2048 = down の 2560 だけが入る)。
pub(crate) fn pf_tc_shared4_eligible(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
    min_m: usize,
) -> bool {
    pf_tc2_type_ok(ty)
        && pf_tc2_k_ok(ty, n_in)
        && nt >= min_m
        && n_out >= 2048
        && n_out % TC2_BN == 0
}


/// `DENNO_PF_TC_SHARED3` の値の解釈 (純関数)。明示的な偽値だけ off。
///
/// **既定 on** (lab t222b で採用): shared の gpu が 1,101 → 111 ms (14,018)、
/// 297 → 28 ms (3,826)。本番の既定経路は BM=64 の TC カーネルに入っておらず、
/// v1 (Q8_0) と従来の量子 GEMM (Q4_K / Q5_K) で回っていた。
/// md5 は BF16 丸め (と K-quant の TC 化) で変わる。
pub(crate) fn pf_tc_shared3_from_value(v: Option<&str>) -> bool {
    match v.map(|s| s.trim().to_ascii_lowercase()) {
        Some(s) => matches!(s.as_str(), "1" | "on" | "true" | "yes"),
        None => true,
    }
}


/// shared expert の射影を sn3 で回すか (**既定 on**。`0` で切る)。
pub(crate) fn pf_tc_shared3_on() -> bool {
    #[cfg(test)]
    {
        match PF_TC_SN3_FORCE.load(Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        pf_tc_shared3_from_value(std::env::var("DENNO_PF_TC_SHARED3").ok().as_deref())
    })
}


/// sn3 が扱える型か (純関数)。
pub(crate) fn pf_tc_sn3_type_ok(ty: crate::kquant::GgmlType) -> bool {
    use crate::kquant::GgmlType as T;
    matches!(ty, T::Q8_0 | T::Q4K | T::Q5K | T::Bf16)
}


/// sn3 の k の前提 (純関数)。BF16 と Q8_0 は 32、K-quant は 256 の倍数。
pub(crate) fn pf_tc_sn3_k_ok(ty: crate::kquant::GgmlType, n_in: usize) -> bool {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Bf16 | T::Q8_0 => n_in > 0 && n_in % SN3_BK == 0,
        T::Q4K | T::Q5K => n_in > 0 && n_in % 256 == 0,
        _ => false,
    }
}


/// この形を sn3 で回せるか (純関数。カーネルの前提と同じ)。
pub(crate) fn pf_tc_sn3_eligible(
    ty: crate::kquant::GgmlType,
    nt: usize,
    n_in: usize,
    n_out: usize,
    min_m: usize,
) -> bool {
    pf_tc_sn3_type_ok(ty)
        && pf_tc_sn3_k_ok(ty, n_in)
        && nt >= min_m.max(SN3_BM)
        && n_out > 0
        && n_out % SN3_BN == 0
}


/// PTX の置き場所。`DENNO_PF_SN3_PTX` があればそれ、無ければクレートの
/// `kernels/kernels_gemm_sn3.ptx`。
fn sn3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_SN3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_sn3.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
/// **current context を張った後に呼ぶこと**。
pub(crate) fn tc_sn3_kernels() -> Option<&'static Sn3Gemm> {
    SN3_GEMM
        .get_or_init(|| {
            let path = sn3_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-sn3] PTX が読めない ({}) : {e} → sn2 / 従来の GEMM へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0);
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-sn3] cuModuleLoadData 失敗 → sn2 / 従来の GEMM へ");
                    return None;
                }
                let mut q8_0: CUfunction = std::ptr::null_mut();
                let mut q4k: CUfunction = std::ptr::null_mut();
                let mut q5k: CUfunction = std::ptr::null_mut();
                let mut bf16: CUfunction = std::ptr::null_mut();
                for (dst, name, label) in [
                    (
                        &mut q8_0,
                        b"dno_gemm_sn3_q8_0_w\0".as_ptr(),
                        "dno_gemm_sn3_q8_0_w",
                    ),
                    (
                        &mut q4k,
                        b"dno_gemm_sn3_q4k_w\0".as_ptr(),
                        "dno_gemm_sn3_q4k_w",
                    ),
                    (
                        &mut q5k,
                        b"dno_gemm_sn3_q5k_w\0".as_ptr(),
                        "dno_gemm_sn3_q5k_w",
                    ),
                    (
                        &mut bf16,
                        b"dno_gemm_sn3_bf16_w\0".as_ptr(),
                        "dno_gemm_sn3_bf16_w",
                    ),
                ] {
                    if (lib.module_get_function)(dst, module, name as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[pf-sn3] PTX に {label} が無い → sn2 / 従来の GEMM へ");
                        return None;
                    }
                }
                Some(Sn3Gemm {
                    module,
                    q8_0,
                    q4k,
                    q5k,
                    bf16,
                })
            }
        })
        .as_ref()
}


/// 型ごとのカーネル (純関数)。
fn sn3_func(fns: &Sn3Gemm, ty: crate::kquant::GgmlType) -> Option<CUfunction> {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Q8_0 => Some(fns.q8_0),
        T::Q4K => Some(fns.q4k),
        T::Q5K => Some(fns.q5k),
        T::Bf16 => Some(fns.bf16),
        _ => None,
    }
}


/// BM=64 のカーネルを `SN3_MAX_TOKENS` 行ずつ起動する (w / x / out はデバイス)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn sn3_gemm_launch(
    ctx: &CudaCtx,
    fns: &Sn3Gemm,
    ty: crate::kquant::GgmlType,
    wptr: CUdeviceptr,
    xptr: CUdeviceptr,
    optr: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
) -> Result<()> {
    if nt == 0 {
        return Ok(());
    }
    let Some(f) = sn3_func(fns, ty) else {
        bail!("sn3_gemm_launch: 未対応の型 {ty:?}");
    };
    let grid_x = (n_out / SN3_BN) as u32;
    let mut t0 = 0usize;
    while t0 < nt {
        let m = (nt - t0).min(SN3_MAX_TOKENS);
        let grid_y = m.div_ceil(SN3_BM) as u32;
        let (mut pw, mut px, mut po) = (
            wptr,
            xptr + (t0 * n_in * 4) as CUdeviceptr,
            optr + (t0 * n_out * 4) as CUdeviceptr,
        );
        let (mut m_c, mut ni_c, mut no_c, mut rb_c) = (
            m as c_int,
            n_in as c_int,
            n_out as c_int,
            row_bytes as c_int,
        );
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut po as *mut _ as *mut c_void,
            &mut m_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(f, (grid_x, grid_y, 1), (256, 1, 1), 0, &mut args)?;
        }
        t0 += m;
    }
    Ok(())
}



#[cfg(test)]
mod pf_tc_tests {
    use super::*;

    /// env の真理値表 (ゲートの解釈を固定する。既定は on)。
    #[test]
    fn pf_tc_gate_truth_table() {
        assert!(pf_tc_gemm_from_value(None), "未設定は既定 on");
        for v in ["0", "off", "false", "no", "", " OFF "] {
            assert!(!pf_tc_gemm_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_tc_gemm_from_value(Some(v)), "{v}");
        }
        assert_eq!(pf_tc_min_m_from_value(None), 64);
        assert_eq!(pf_tc_min_m_from_value(Some("0")), 64);
        assert_eq!(pf_tc_min_m_from_value(Some("abc")), 64);
        assert_eq!(pf_tc_min_m_from_value(Some("1")), 1);
        assert_eq!(pf_tc_min_m_from_value(Some(" 512 ")), 512);
    }

    /// ゲートが env/強制上書きで切り替わること (テストは 1 プロセス並列なので
    /// 既定値だけを見る: 強制は set/戻しを対で行う)。
    #[test]
    fn pf_tc_gate_force_switch() {
        let before = pf_tc_gemm_on();
        set_pf_tc_gemm_for_test(Some(true));
        assert!(pf_tc_gemm_on());
        set_pf_tc_gemm_for_test(Some(false));
        assert!(!pf_tc_gemm_on());
        set_pf_tc_gemm_for_test(None);
        assert_eq!(pf_tc_gemm_on(), before);
    }

    /// 実モデルの形 (ssm qkv 16384×2560 / full qkv 13312×2560 / out 2560×6144 /
    /// shared 640×2560, 640×2560) がすべて前提を満たすこと。
    #[test]
    fn pf_tc_eligibility_real_shapes() {
        for (n_in, n_out) in [
            (2560usize, 16384usize),
            (2560, 10240),
            (2560, 6144),
            (2560, 13312),
            (6144, 2560),
            (2560, 640),
            (640, 2560),
        ] {
            assert!(
                pf_tc_gemm_eligible(512, n_in, n_out, 64),
                "k={n_in} n={n_out}"
            );
        }
        // 19 トークンの next turn prefill は従来カーネル (M の下限)。
        assert!(!pf_tc_gemm_eligible(19, 2560, 16384, 64));
        assert!(pf_tc_gemm_eligible(19, 2560, 16384, 1));
        // ブロック形に合わない形は fail closed。
        for (n_in, n_out) in [(2559usize, 1024usize), (2560, 1024 + 1), (2560, 32)] {
            assert!(!pf_tc_gemm_eligible(512, n_in, n_out, 64), "k={n_in} n={n_out}");
        }
    }

    /// task 222: 3 本は lab t222b で採用して**既定 on**、sh4 は未測定で既定 off。
    #[test]
    fn pf_sn3_gate_truth_table() {
        for f in [
            pf_tc_shared3_from_value,
            pf_mix_tc_from_value,
            pf_attn_tc_from_value,
        ] {
            assert!(f(None), "未設定は既定 on (lab t222b で採用)");
            for v in ["0", "off", "false", "no", "", " OFF ", "2"] {
                assert!(!f(Some(v)), "{v}");
            }
            for v in ["1", "on", "true", "yes", " ON ", " 1 "] {
                assert!(f(Some(v)), "{v}");
            }
        }
        // sh4 (shared の down を tc2 へ) は未測定なので既定 off のまま。
        assert!(!pf_tc_shared4_from_value(None), "sh4 は既定 off");
        for v in ["0", "off", "false", "no", "", "2"] {
            assert!(!pf_tc_shared4_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes"] {
            assert!(pf_tc_shared4_from_value(Some(v)), "{v}");
        }
        // 強制上書きが効き、戻すと env の解釈に戻ること。
        let before = (pf_tc_shared3_on(), pf_mix_tc_on(), pf_tc_shared4_on());
        assert!(before.0 && before.1 && !before.2, "既定は sn3/mix_tc on・sh4 off");
        set_pf_tc_shared3_for_test(Some(true));
        set_pf_mix_tc_for_test(Some(true));
        set_pf_tc_shared4_for_test(Some(true));
        assert!(pf_tc_shared3_on() && pf_mix_tc_on() && pf_tc_shared4_on());
        set_pf_tc_shared3_for_test(Some(false));
        set_pf_mix_tc_for_test(Some(false));
        set_pf_tc_shared4_for_test(Some(false));
        assert!(!pf_tc_shared3_on() && !pf_mix_tc_on() && !pf_tc_shared4_on());
        set_pf_tc_shared3_for_test(None);
        set_pf_mix_tc_for_test(None);
        set_pf_tc_shared4_for_test(None);
        assert_eq!(
            (pf_tc_shared3_on(), pf_mix_tc_on(), pf_tc_shared4_on()),
            before
        );
        // attention v2 のゲートも強制上書きが往復すること。
        let before_a = pf_attn_tc_on();
        set_pf_attn_tc_for_test(Some(true));
        assert!(pf_attn_tc_on());
        set_pf_attn_tc_for_test(Some(false));
        assert!(!pf_attn_tc_on());
        set_pf_attn_tc_for_test(None);
        assert_eq!(pf_attn_tc_on(), before_a);
    }

    /// task 301: UD-Q4_K_XL の mix (Q8_0 の down / up) を sn3 に載せるゲート。
    /// **既定 on** (lab t301a で採用)。明示的な偽値だけ off。強制上書きは往復する。
    #[test]
    fn pf_mix_q8_tc_gate_truth_table() {
        assert!(pf_mix_q8_tc_from_value(None), "既定 on");
        for v in ["0", "off", "false", "no", "", " OFF ", "2"] {
            assert!(!pf_mix_q8_tc_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON ", " 1 "] {
            assert!(pf_mix_q8_tc_from_value(Some(v)), "{v}");
        }
        let before = pf_mix_q8_tc_on();
        assert!(before, "既定 on");
        set_pf_mix_q8_tc_for_test(Some(false));
        assert!(!pf_mix_q8_tc_on());
        set_pf_mix_q8_tc_for_test(Some(true));
        assert!(pf_mix_q8_tc_on());
        set_pf_mix_q8_tc_for_test(None);
        assert_eq!(pf_mix_q8_tc_on(), before);
    }

    /// task 301: 実形が sn3 の前提 (n_out % 64 == 0 / n_in % 32 == 0 /
    /// nt >= 64) を満たすこと。満たさない形は従来経路へ落ちる (fail closed)。
    #[test]
    fn pf_mix_q8_tc_ud_shapes() {
        use crate::kquant::GgmlType as T;
        // 3,826 のチャンク (1,913) と 14k のチャンク (2,047)。
        for nt in [1913usize, 2047] {
            for (n_in, n_out) in [(10240usize, 320usize), (320, 10240)] {
                assert!(
                    pf_tc_sn3_eligible(T::Q8_0, nt, n_in, n_out, 1),
                    "nt={nt} k={n_in} n={n_out}"
                );
            }
        }
        // next turn (19 トークン) は従来経路 (BM 未満)。
        assert!(!pf_tc_sn3_eligible(T::Q8_0, 19, 10240, 320, 1));
        assert!(!pf_tc_sn3_eligible(T::Q8_0, 19, 320, 10240, 1));
        // inject (n=4) は sn3 の対象外。
        assert!(!pf_tc_sn3_eligible(T::Q8_0, 1913, 10240, 4, 1));
        // Q8_0 以外の量子型は sn3 の Q8_0 入口を使わない (型ごとの入口が要る)。
        assert!(!pf_tc_sn3_k_ok(T::Q5_1, 10240));
    }

    /// task 222: sn3 は実モデルの 4 つの型と形 (shared の Q5_K と mix の BF16) を
    /// 受け、sn2 が落としていた型を取りこぼさないこと。
    #[test]
    fn pf_sn3_eligibility_real_shapes() {
        use crate::kquant::GgmlType as T;
        // shared expert の実形 (gate/up = 2560→640 / down = 640→2560)。
        // gate/up は層ごとに Q8_0 / Q4_K / Q5_K。down は実モデルでは Q8_0 だけで、
        // k=640 は 256 の倍数でないので K-quant は対象外 (正しい fail closed)。
        for ty in [T::Q8_0, T::Q4K, T::Q5K] {
            assert!(pf_tc_sn3_eligible(ty, 512, 2560, 640, 1), "{ty:?} gate");
        }
        assert!(pf_tc_sn3_eligible(T::Q8_0, 512, 640, 2560, 1), "Q8_0 down");
        assert!(pf_tc_sn3_eligible(T::Bf16, 512, 640, 2560, 1), "BF16 down");
        assert!(
            !pf_tc_sn3_eligible(T::Q4K, 512, 640, 2560, 1),
            "k=640 は 256 の倍数でないので Q4_K は不可"
        );
        // hc mix の実形 (down = 10240→320 / up = 320→10240、重みは BF16)。
        assert!(pf_tc_sn3_eligible(T::Bf16, 512, 10240, 320, 1));
        assert!(pf_tc_sn3_eligible(T::Bf16, 512, 320, 10240, 1));
        // sn2 と同じ型 (Q8_0 / Q4_K) は sn3 でも通る (置き換え可能)。
        assert!(pf_tc_sn3_eligible(T::Q4K, 512, 2560, 640, 1));
        // 19 トークンの next turn prefill は従来経路 (64 行の下限)。
        assert!(!pf_tc_sn3_eligible(T::Bf16, 19, 10240, 320, 1));
        assert!(pf_tc_sn3_eligible(T::Bf16, 64, 10240, 320, 64));
        // 64 の倍数でない出力行・未対応の型・k の倍数ずれは fail closed。
        assert!(!pf_tc_sn3_eligible(T::Bf16, 512, 10240, 4, 1), "inject (n=4)");
        assert!(!pf_tc_sn3_eligible(T::Q8_0, 512, 2560, 100, 1), "n_out=100");
        assert!(!pf_tc_sn3_eligible(T::Q6K, 512, 2560, 640, 1), "Q6_K は対象外");
        assert!(!pf_tc_sn3_eligible(T::Q4K, 512, 2559, 640, 1), "k が 256 の倍数でない");
        assert!(
            !pf_tc_sn3_eligible(T::Bf16, 512, 10241, 320, 1),
            "BF16 の k が 32 の倍数でない"
        );
        // task 222: shared の down (N=2560) は tc2 (BM=128) の担当。
        // gate/up (N=640) は対象外 (並列度が落ちる)。
        assert!(pf_tc_shared4_eligible(T::Q8_0, 512, 640, 2560, 1), "down 2560");
        assert!(
            !pf_tc_shared4_eligible(T::Q8_0, 512, 2560, 640, 1),
            "gate/up の 640 は tc2 にしない"
        );
        assert!(
            !pf_tc_shared4_eligible(T::Q8_0, 512, 10240, 320, 1),
            "mix の 320 は tc2 にしない"
        );
        assert!(
            !pf_tc_shared4_eligible(T::Bf16, 512, 640, 2560, 1),
            "BF16 は tc2 の型に無い"
        );
    }

    /// task 175: テンソルコア経路の選択は **型** も見る。
    ///
    /// v1 カーネル (`dno_gemm_tc_q8_0_w`) は Q8_0 専用。Q4_K / Q5_K / Q5_1 の
    /// 重みを渡すとバイト列を Q8_0 として読むので出力が壊れる。lab t175a では
    /// shared expert の gate が Q4_K だったため、br172 の `pf2_gemm` が形だけで
    /// v1 を選び、prefill の出力が全部 `!` になった。
    #[test]
    fn pf2_tc_kind_respects_weight_type() {
        use crate::kquant::GgmlType as T;
        // shared expert の実形 (gate/up = 2560→640 / down = 640→2560)。
        // task 181 で N < 4096 の形は kind 3 (BN=64 の小さい N 用) が受ける。
        // どれも **型を見て** 選ぶ (Q4_K を Q8_0 専用の v1 に流してはいけない)。
        assert_eq!(
            pf2_tc_kind(T::Q4K, 512, 2560, 640),
            3,
            "Q4_K を Q8_0 専用の v1 に流した"
        );
        assert_eq!(pf2_tc_kind(T::Q5K, 512, 2560, 640), 0);
        assert_eq!(pf2_tc_kind(T::Q5_1, 512, 2560, 640), 0);
        assert_eq!(pf2_tc_kind(T::Q8_0, 512, 2560, 640), 3, "N=640 は kind 3");
        assert_eq!(pf2_tc_kind(T::Q8_0, 512, 640, 2560), 3);
        // N >= TC2_MIN_N (4096) は v2 (Q8_0 / Q4_K / Q5_K を型で切り替える)。
        assert_eq!(pf2_tc_kind(T::Q8_0, 512, 2560, 10240), 2);
        assert_eq!(pf2_tc_kind(T::Q4K, 512, 2560, 10240), 2);
        assert_eq!(pf2_tc_kind(T::Q5K, 512, 2560, 10240), 2);
        // v2 が扱えない型は v1 にも流さない (従来カーネルへ)。
        assert_eq!(pf2_tc_kind(T::Q5_1, 512, 2560, 10240), 0);
        // 形が前提に合わないものは従来。
        assert_eq!(pf2_tc_kind(T::Q8_0, 512, 2559, 640), 0);
        assert_eq!(pf2_tc_kind(T::Q8_0, 512, 2560, 641), 0);
    }

    /// タイルの索引算術を CPU ミラーで確かめる (`kernels_gemm_tc.cu` の
    /// デコード配置・warp の担当範囲・書き出し条件を 1 対 1 で写す)。
    ///
    /// GPU 実測の代わりではない。**どの要素がどのスレッド/warp/スロットに
    /// 入るか**の取り違えだけが落ちる。
    #[test]
    fn pf_tc_tile_index_mirror() {
        // 1. 重みタイル: 256 スレッド × 8 要素が [64][32] をちょうど 1 回覆う。
        let mut seen = vec![false; TC_BN * TC_BK];
        for tid in 0..256usize {
            let e0 = tid * 8;
            let row = e0 >> 5;
            let j = e0 & 31;
            assert!(row < TC_BN && j + 8 <= TC_BK);
            for u in 0..8 {
                let idx = row * TC_BK + j + u;
                assert!(!seen[idx], "重みスロットの二重書き row={row} j={j}");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "重みスロットの取りこぼし");

        // 2. 入力タイル: 256 スレッド × 16 要素 (8 × 2) が [128][32] を覆う。
        let mut seen = vec![false; TC_BM * TC_BK];
        for tid in 0..256usize {
            for half in 0..2usize {
                let e0 = (tid + half * 256) * 8;
                let tok = e0 >> 5;
                let j = e0 & 31;
                assert!(tok < TC_BM && j + 8 <= TC_BK);
                for u in 0..8 {
                    let idx = tok * TC_BK + j + u;
                    assert!(!seen[idx], "入力スロットの二重書き tok={tok} j={j}");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "入力スロットの取りこぼし");

        // 3. warp のアキュムレータ (4(M) × 2(N) の warp × 2 × 2 タイル) が
        //    [128][64] をちょうど 1 回覆い、行/列の先頭が 16 の倍数であること。
        let mut seen = vec![false; TC_BM * TC_BN];
        for warp in 0..8usize {
            let (wm, wn) = (warp & 3, warp >> 2);
            for i in 0..2usize {
                let mrow = wm * 32 + i * 16;
                assert_eq!(mrow % 16, 0);
                for j in 0..2usize {
                    let ncol = wn * 32 + j * 16;
                    assert_eq!(ncol % 16, 0);
                    for r in 0..16usize {
                        for c in 0..16usize {
                            let idx = (mrow + r) * TC_BN + ncol + c;
                            assert!(!seen[idx], "アキュムレータの二重担当");
                            seen[idx] = true;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "アキュムレータの取りこぼし");

        // 4. 端数タイル: M が 128 の倍数でないとき、書き出す行が nt 未満だけに
        //    なること (mrow + 16 > nt のタイルはステージ経由)。
        for nt in [1usize, 17, 63, 65, 127, 128, 129, 194, 512, 1024, 2048] {
            for warp in 0..8usize {
                let mwarp = warp & 3;
                for i in 0..2usize {
                    let mrow = mwarp * 32 + i * 16;
                    let direct = mrow + 16 <= nt;
                    let staged = !direct && mrow < nt;
                    if staged {
                        assert!(nt - mrow < 16);
                    }
                    if direct {
                        assert!(mrow + 16 <= nt);
                    }
                }
            }
        }

        // 5. 静的共有メモリの見積り (カーネル側の宣言と対) が上限に収まること。
        let smem = TC_BN * TC_BK * 2 + TC_BM * TC_BK * 2 + 8 * 16 * 16 * 4;
        assert_eq!(smem, 20_480, "共有メモリの合計が 20 KiB でない");
        assert!(smem <= 48 * 1024, "静的 smem の上限 (48 KiB) を超えない");
    }

    /// `gemm_try` のまとめ幅: TC が使えるときだけ 2048、それ以外は 128。
    ///
    /// task 181 で小さい N 用 (N < 4096 の Q8_0 / Q4_K) が既定 on になったので、
    /// 「TC を使わない」側は小さい N のゲートも明示的に切って測る。
    #[test]
    fn pf_tc_gemm_try_tile() {
        use crate::kquant::GgmlType as T;
        for (k, n) in [
            (2560usize, 16384usize),
            (2560, 13312),
            (6144, 2560),
            (640, 2560),
        ] {
            assert_eq!(
                gemm_try_tile_for(true, false, T::Q8_0, 512, k, n, 64),
                TC_MAX_TOKENS
            );
            // 小さい N のゲートも切れば 128 (従来のまとめ幅)。
            set_pf_tc_smalln_for_test(Some(false));
            let off = gemm_try_tile_for(false, false, T::Q8_0, 512, k, n, 64);
            set_pf_tc_smalln_for_test(None);
            assert_eq!(off, 128, "k={k} n={n}: 小さい N のゲートが切れていない");
        }
        // 小さい N 用が既定 on: N < 4096 の形は Q8_0 / Q4_K とも 2,048 まで。
        for (k, n) in [(6144usize, 2560usize), (640, 2560)] {
            assert_eq!(
                gemm_try_tile_for(false, false, T::Q8_0, 512, k, n, 64),
                TC_MAX_TOKENS,
                "k={k} n={n}: smallN がまとめ幅を広げていない"
            );
        }
        assert_eq!(
            gemm_try_tile_for(false, false, T::Q4K, 512, 2560, 640, 64),
            TC_MAX_TOKENS
        );
        // 19 トークンの next turn は従来の 128 のまま (1 回で収まる)。
        assert_eq!(
            gemm_try_tile_for(true, false, T::Q8_0, 19, 2560, 16384, 64),
            128
        );
        // Q8_0 以外と、ブロック形が合わない形も従来のまま。
        assert_eq!(
            gemm_try_tile_for(true, false, T::Q5K, 512, 2560, 16384, 64),
            128
        );
        assert_eq!(
            gemm_try_tile_for(true, false, T::Q8_0, 512, 2559, 16384, 64),
            128
        );
        assert_eq!(
            gemm_try_tile_for(true, false, T::Q8_0, 512, 2560, 1000, 64),
            128
        );
        // task 176: v2 だけが使える形 (Q5K / Q4K) でも 2,048 までまとめる。
        assert_eq!(
            gemm_try_tile_for(false, true, T::Q5K, 512, 2560, 10240, 64),
            TC_MAX_TOKENS
        );
        assert_eq!(
            gemm_try_tile_for(false, true, T::Q4K, 512, 2560, 6144, 64),
            TC_MAX_TOKENS
        );
        // v2 の条件 (出力行 128 の倍数・ブロック数) を満たさない形は、task 181 の
        // 小さい N 用 (N < 4096) が受けるので 2,048 までまとめる。
        assert_eq!(
            gemm_try_tile_for(false, true, T::Q8_0, 512, 6144, 2560, 64),
            TC_MAX_TOKENS
        );
        // 64 の倍数でない出力行はどの TC 経路にも入らない (128 のまま)。
        assert_eq!(
            gemm_try_tile_for(false, true, T::Q8_0, 512, 6144, 2600, 64),
            128
        );
        assert_eq!(TC_MAX_TOKENS, 2048);
    }

    /// 端数 M のときにステージ経由で写す行数が 0..15 に収まること (境界の固定)。
    #[test]
    fn pf_tc_partial_rows() {
        for nt in [1usize, 15, 16, 17, 128, 129, 2047, 2048] {
            for warp in 0..8usize {
                let mwarp = warp & 3;
                for i in 0..2usize {
                    let mrow = mwarp * 32 + i * 16;
                    if mrow < nt && mrow + 16 > nt {
                        let rows = nt - mrow;
                        assert!(rows >= 1 && rows <= 15, "nt={nt} mrow={mrow}");
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // task 176: v2 (kernels_gemm_tc2.cu) のゲート・適格性・索引ミラー
    // ------------------------------------------------------------------

    /// `DENNO_PF_TC_GEMM2` の真理値表 (既定 on。lab t176a で採用)。
    #[test]
    fn pf_tc2_gate_truth_table() {
        assert!(pf_tc2_gemm_from_value(None), "未設定は既定 on");
        for v in ["0", "off", "false", "no", "", " OFF "] {
            assert!(!pf_tc2_gemm_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_tc2_gemm_from_value(Some(v)), "{v}");
        }
    }

    /// 強制上書きが効くこと (テストは 1 プロセス並列なので set/戻しを対で行う)。
    #[test]
    fn pf_tc2_gate_force_switch() {
        let before = pf_tc2_gemm_on();
        set_pf_tc2_gemm_for_test(Some(true));
        assert!(pf_tc2_gemm_on());
        set_pf_tc2_gemm_for_test(Some(false));
        assert!(!pf_tc2_gemm_on());
        set_pf_tc2_gemm_for_test(None);
        assert_eq!(pf_tc2_gemm_on(), before);
    }

    /// 実モデルの形で v2 の適格性が期待どおりになること。
    ///
    /// 14k cold prefill の 1 チャンク (nt=512) の 3 本:
    ///   ssm qkv  512 x 2560 x 10240 (Q8_0 / 4 層だけ Q5_K) → v2
    ///   z        512 x 2560 x  6144 (Q8_0 / 4 層だけ Q4_K)  → v2
    ///   out      512 x 6144 x  2560                          → br169 (N が小さい)
    #[test]
    fn pf_tc2_eligibility_real_shapes() {
        use crate::kquant::GgmlType as T;
        // v2 が担当する形 (N >= 4096)。M は 1 でも形の条件は満たす
        // (GPU 検算テストが M=1/17 を v2 に通せるようにするため)。
        for (ty, k, n) in [
            (T::Q8_0, 2560usize, 10240usize),
            (T::Q5K, 2560, 10240),
            (T::Q8_0, 2560, 6144),
            (T::Q4K, 2560, 6144),
            (T::Q8_0, 2560, 13312),
        ] {
            assert!(pf_tc2_eligible(ty, 512, k, n, 64), "{ty:?} k={k} n={n}");
            assert!(pf_tc2_eligible(ty, 1, k, n, 1), "{ty:?} M=1 k={k} n={n}");
            assert!(pf_tc2_blocks(512, n) >= 48);
        }
        // 出力行が小さい形は br169 に任せる (out_proj 2560 / shared 640)。
        for (k, n) in [(6144usize, 2560usize), (2560, 640), (2560, 1024)] {
            assert!(
                !pf_tc2_eligible(T::Q8_0, 512, k, n, 64),
                "k={k} n={n} が v2 適格になっている"
            );
        }
        // 型・k の倍数・M の下限。
        assert!(!pf_tc2_eligible(T::Q5_1, 512, 2560, 10240, 64));
        assert!(!pf_tc2_eligible(T::Q8_0, 512, 2559, 10240, 64));
        assert!(!pf_tc2_eligible(T::Q5K, 512, 2560, 10240, 513));
        assert!(!pf_tc2_eligible(T::Q4K, 512, 2400, 10240, 64), "256 の倍数でない");
        assert!(pf_tc2_eligible(T::Q4K, 512, 2560, 10240, 64));
        assert!(pf_tc2_eligible(T::Q5K, 512, 5120, 10240, 64));
        // 2048 トークン (まとめ幅の上限) でも適格。
        assert!(pf_tc2_eligible(T::Q8_0, 2048, 2560, 10240, 64));
        // 出力行が 128 の倍数でない形は落ちる。
        assert!(!pf_tc2_eligible(T::Q8_0, 512, 2560, 10240 + 64, 64));
    }

    /// `pf_tc2_blocks` の数え方 (純関数)。
    #[test]
    fn pf_tc2_blocks_count() {
        assert_eq!(pf_tc2_blocks(512, 10240), 80 * 4);
        assert_eq!(pf_tc2_blocks(512, 6144), 48 * 4);
        assert_eq!(pf_tc2_blocks(512, 2560), 20 * 4);
        assert_eq!(pf_tc2_blocks(19, 10240), 80 * 1);
        assert_eq!(pf_tc2_blocks(2048, 10240), 80 * 16);
        // 出力行が 128 の倍数でない形は 0 (適格判定で落ちる)。
        assert_eq!(pf_tc2_blocks(512, 10240 + 64), 0);
    }

    /// task 181: 小さい N 用のゲートは **既定 on** (`DENNO_PF_TC_SMALLN=0` で off)。
    #[test]
    fn pf_tc_smalln_gate_truth_table() {
        assert!(pf_tc_smalln_from_value(None), "未設定は既定 on (lab t181b で採用)");
        for v in ["0", "off", "false", "no", "", " OFF "] {
            assert!(!pf_tc_smalln_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_tc_smalln_from_value(Some(v)), "{v}");
        }
    }

    /// 強制上書きが効くこと (set/戻しを対で行う)。
    #[test]
    fn pf_tc_smalln_gate_force_switch() {
        let before = pf_tc_smalln_on();
        set_pf_tc_smalln_for_test(Some(true));
        assert!(pf_tc_smalln_on());
        set_pf_tc_smalln_for_test(Some(false));
        assert!(!pf_tc_smalln_on());
        set_pf_tc_smalln_for_test(None);
        assert_eq!(pf_tc_smalln_on(), before);
    }

    /// task 181: 実モデルの小さい N の形 (shared 640 / out 2560) が適格で、
    /// N >= 4096 (v2 の担当) と Q5_K (v2 / v1 の担当) は落ちること。
    #[test]
    fn pf_tc_smalln_eligibility_real_shapes() {
        use crate::kquant::GgmlType as T;
        // shared expert: gate (Q4_K) 2560→640 / up (Q8_0) 2560→640 / down 640→2560。
        for (ty, k, n) in [
            (T::Q4K, 2560usize, 640usize),
            (T::Q8_0, 2560, 640),
            (T::Q8_0, 640, 2560),
            // ssm out_proj (N 2560, K 6144)。
            (T::Q8_0, 6144, 2560),
        ] {
            assert!(pf_tc_smalln_eligible(ty, 512, k, n, 64), "{ty:?} k={k} n={n}");
            assert!(pf_tc_smalln_eligible(ty, 1, k, n, 1), "{ty:?} M=1 k={k} n={n}");
        }
        // N >= TC2_MIN_N は v2 の担当 (ここでは適格にしない)。
        assert!(!pf_tc_smalln_eligible(T::Q8_0, 512, 2560, 10240, 64));
        assert!(!pf_tc_smalln_eligible(T::Q4K, 512, 2560, 4096, 64));
        // 型: Q5_K / Q5_1 は扱わない (v2 / 従来へ)。
        assert!(!pf_tc_smalln_eligible(T::Q5K, 512, 2560, 640, 64));
        assert!(!pf_tc_smalln_eligible(T::Q5_1, 512, 2560, 640, 64));
        // k の倍数 (Q8_0 = 32 / Q4_K = 256) と出力行 64 の倍数。
        assert!(!pf_tc_smalln_eligible(T::Q8_0, 512, 2559, 640, 64));
        assert!(!pf_tc_smalln_eligible(T::Q4K, 512, 2400, 640, 64));
        assert!(!pf_tc_smalln_eligible(T::Q8_0, 512, 2560, 641, 64));
        assert!(!pf_tc_smalln_eligible(T::Q8_0, 512, 2560, 32, 64));
        // M の下限 (19 トークンの next turn は従来経路)。
        assert!(!pf_tc_smalln_eligible(T::Q8_0, 19, 2560, 640, 64));
        assert!(pf_tc_smalln_eligible(T::Q8_0, 19, 2560, 640, 1));
        // 2048 トークン (まとめ幅の上限) でも適格。
        assert!(pf_tc_smalln_eligible(T::Q8_0, 2048, 2560, 640, 64));
    }

    /// task 181: ブロック数が並列度の根拠になっていること (BN=64 の意味)。
    #[test]
    fn pf_tc_smalln_blocks_count() {
        // M=512 の 1 チャンク: shared の gate/up は 40、down は 160 ブロック。
        assert_eq!(pf_tc_smalln_blocks(512, 640), 10 * 4);
        assert_eq!(pf_tc_smalln_blocks(512, 2560), 40 * 4);
        assert_eq!(pf_tc_smalln_blocks(19, 640), 10);
        assert_eq!(pf_tc_smalln_blocks(2048, 640), 10 * 16);
        // 同じ形を v2 (BN=128) で撃つと 5 x 4 = 20 ブロックしか出ない (比較の固定)。
        assert_eq!(pf_tc2_blocks(512, 640), 5 * 4);
        // 64 の倍数でない形は 0。
        assert_eq!(pf_tc_smalln_blocks(512, 641), 0);
    }

    /// task 181: 小さい N 用カーネルの索引算術を CPU ミラーで確かめる
    /// (`kernels_gemm_sn.cu` のステージ担当・warp の担当範囲・端数の条件を
    /// 1 対 1 で写す)。GPU 実測の代わりではない。
    #[test]
    fn pf_tc_smalln_tile_index_mirror() {
        // 1. 重みタイル: 256 スレッド × 8 要素 × **1 組** が [64][32] を
        //    ちょうど 1 回覆う (tc2 は 2 組 = 128 行)。行 = tid>>2、列 = (tid&3)*8。
        let mut seen = vec![false; TC_SN_BN * TC_SN_BK];
        for tid in 0..256usize {
            let row = tid >> 2;
            let j = (tid & 3) * 8;
            assert!(row < TC_SN_BN && j + 8 <= TC_SN_BK, "row={row} j={j}");
            for u in 0..8usize {
                let idx = row * TC_SN_BK + j + u;
                assert!(!seen[idx], "重みスロットの二重書き row={row} j={j}");
                seen[idx] = true;
            }
        }
        assert!(seen.iter().all(|&b| b), "重みスロットの取りこぼし");

        // 2. 入力タイル: 256 スレッド × 8 要素 × 2 組が [128][32] を覆う。
        let mut seen = vec![false; TC_SN_BM * TC_SN_BK];
        for tid in 0..256usize {
            for hf in 0..2usize {
                let row = (tid >> 2) + hf * 64;
                let j = (tid & 3) * 8;
                assert!(row < TC_SN_BM && j + 8 <= TC_SN_BK, "row={row} j={j}");
                for u in 0..8usize {
                    let idx = row * TC_SN_BK + j + u;
                    assert!(!seen[idx], "入力スロットの二重書き row={row} j={j}");
                    seen[idx] = true;
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "入力スロットの取りこぼし");

        // 3. warp のアキュムレータ (4 トークン群 × 2 行群 × 2 × 2 タイル) が
        //    [128][64] をちょうど 1 回覆い、行/列の先頭が 16 の倍数であること。
        let mut seen = vec![false; TC_SN_BM * TC_SN_BN];
        for warp in 0..8usize {
            let (wm, wn) = (warp & 3, warp >> 2);
            assert_eq!(wn, warp / 4);
            for i in 0..2usize {
                let mrow = wm * 32 + i * 16;
                assert_eq!(mrow % 16, 0);
                for j in 0..2usize {
                    let ncol = wn * 32 + j * 16;
                    assert_eq!(ncol % 16, 0);
                    for r in 0..16usize {
                        for c in 0..16usize {
                            let idx = (mrow + r) * TC_SN_BN + ncol + c;
                            assert!(!seen[idx], "アキュムレータの二重担当");
                            seen[idx] = true;
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "アキュムレータの取りこぼし");

        // 4. 行ストライド (ldm は bf16 で 16 バイトの倍数が要る) と smem の見積り。
        assert_eq!(TC_SN_LD % 8, 0);
        assert_eq!(TC_SN_LD * 2 % 16, 0);
        assert!(TC_SN_LD > TC_SN_BK, "詰め物が無い (バンク衝突)");
        let smem = 2 * TC_SN_BN * TC_SN_LD * 2 + 2 * TC_SN_BM * TC_SN_LD * 2;
        assert_eq!(smem, TC_SN_SMEM_BYTES, "共有メモリの合計が .cu と違う");
        assert_eq!(smem, 30_720);
        assert!(smem <= 48 * 1024);
        let stage_floats = 8 * 256;
        assert!(
            stage_floats * 4 <= 2 * TC_SN_BM * TC_SN_LD * 2,
            "端数ステージが sX に収まらない"
        );

        // 5. 端数タイル: 128 の倍数でない M で、ステージ経由の行数が 0..15。
        for nt in [1usize, 15, 16, 17, 127, 128, 129, 2047, 2048] {
            for warp in 0..8usize {
                let wm = warp & 3;
                for i in 0..2usize {
                    let mrow = wm * 32 + i * 16;
                    if mrow < nt && mrow + 16 > nt {
                        let rows = nt - mrow;
                        assert!((1..=15).contains(&rows), "nt={nt} mrow={mrow}");
                    }
                }
            }
        }
    }

    /// task 183: `DENNO_PF_TC_SHARED2` の真理値表 (既定 off。明示的な真値だけ on)。
    #[test]
    fn pf_tc_shared2_gate_truth_table() {
        assert!(pf_tc_shared2_from_value(None), "未設定は既定 on (lab t183a で採用)");
        for v in ["0", "off", "false", "no", "", " OFF "] {
            assert!(!pf_tc_shared2_from_value(Some(v)), "{v}");
        }
        for v in ["1", "on", "true", "yes", " ON "] {
            assert!(pf_tc_shared2_from_value(Some(v)), "{v}");
        }
    }

    /// task 183: テスト用の強制上書きが効き、戻すと env に戻ること。
    #[test]
    fn pf_tc_shared2_gate_force_switch() {
        let before = pf_tc_shared2_on();
        set_pf_tc_shared2_for_test(Some(true));
        assert!(pf_tc_shared2_on());
        set_pf_tc_shared2_for_test(Some(false));
        assert!(!pf_tc_shared2_on());
        set_pf_tc_shared2_for_test(None);
        assert_eq!(pf_tc_shared2_on(), before);
    }

    /// task 183: 実形 (shared の gate/up = 2560->640、down = 640->2560) で
    /// BM=64 のカーネルが選ばれること。19 トークンの next turn は sn に残す。
    #[test]
    fn pf_tc_smalln2_eligibility_real_shapes() {
        use crate::kquant::GgmlType as T;
        // 実形: gate/up は k=2560 -> n=640、down は k=640 -> n=2560。
        // Q4_K は k が 256 の倍数のときだけ扱える (down の k=640 は Q8_0 のみ)。
        for (ty, k, n) in [
            (T::Q8_0, 2560usize, 640usize),
            (T::Q8_0, 640, 2560),
            (T::Q4K, 2560, 640),
        ] {
            assert!(
                pf_tc_smalln2_eligible(ty, 512, k, n, 1),
                "{ty:?} k={k} n={n}"
            );
            // 19 トークンはタイルが埋まらないので sn / 従来へ。
            assert!(!pf_tc_smalln2_eligible(ty, 19, k, n, 1), "{ty:?} nt=19");
            assert!(pf_tc_smalln2_eligible(ty, 64, k, n, 1), "{ty:?} nt=64");
        }
        // N >= 4096 は tc2 の担当、割り切れない N と未対応の型は受けない。
        assert!(!pf_tc_smalln2_eligible(T::Q8_0, 512, 2560, 10240, 1));
        assert!(!pf_tc_smalln2_eligible(T::Q8_0, 512, 2560, 641, 1));
        assert!(!pf_tc_smalln2_eligible(T::Q5K, 512, 2560, 640, 1));
        // k が 256 の倍数でない Q4_K は受けない (down の k=640 がこれ)。
        assert!(!pf_tc_smalln2_eligible(T::Q4K, 512, 640, 2560, 1));
    }

    /// task 183: 立つブロック数。N=640 の M=512 で sn は 40、sn2 は 80。
    #[test]
    fn pf_tc_smalln2_blocks_count() {
        assert_eq!(pf_tc_smalln2_blocks(512, 640), 80);
        assert_eq!(pf_tc_smalln2_blocks(512, 2560), 320);
        assert_eq!(pf_tc_smalln2_blocks(194, 640), 40);
        assert_eq!(pf_tc_smalln2_blocks(64, 640), 10);
        assert_eq!(pf_tc_smalln2_blocks(63, 640), 10);
        assert_eq!(pf_tc_smalln2_blocks(512, 641), 0);
        // sn (BM=128) との比較: 40 -> 80 になること (タスクの根拠)。
        assert_eq!(pf_tc_smalln_blocks(512, 640), 40);
        assert_eq!(pf_tc_smalln2_blocks(512, 640), 2 * pf_tc_smalln_blocks(512, 640));
    }

    /// task 183: BM=64 カーネルの索引算術を CPU ミラーで確かめる
    /// (`kernels_gemm_sn2.cu` のステージ担当・warp の担当範囲・端数の条件)。
    #[test]
    fn pf_tc_smalln2_tile_index_mirror() {
        // 1. 重みタイルと入力タイル: 256 スレッド × 8 要素 × 1 組が
        //    [64][32] をちょうど 1 回覆う (sn は入力が 2 組)。
        for what in ["重み", "入力"] {
            let mut seen = vec![false; TC_SN2_BM * TC_SN2_BK];
            for tid in 0..256usize {
                let row = tid >> 2;
                let j = (tid & 3) * 8;
                assert!(row < TC_SN2_BM && j + 8 <= TC_SN2_BK, "{what}: row={row} j={j}");
                for u in 0..8usize {
                    let idx = row * TC_SN2_BK + j + u;
                    assert!(!seen[idx], "{what}スロットの二重書き row={row} j={j}");
                    seen[idx] = true;
                }
            }
            assert!(seen.iter().all(|&b| b), "{what}スロットの取りこぼし");
        }

        // 2. warp のアキュムレータ (2 トークン群 × 4 行群 × 2 タイル) が
        //    [64][64] をちょうど 1 回覆う。
        let mut seen = vec![false; TC_SN2_BM * TC_SN2_BN];
        for warp in 0..8usize {
            let (wm, wn) = (warp & 1, warp >> 1);
            assert_eq!(wn, warp / 2);
            for i in 0..2usize {
                let mrow = wm * 32 + i * 16;
                assert_eq!(mrow % 16, 0);
                let ncol = wn * 16;
                assert_eq!(ncol % 16, 0);
                for r in 0..16usize {
                    for c in 0..16usize {
                        let idx = (mrow + r) * TC_SN2_BN + ncol + c;
                        assert!(!seen[idx], "アキュムレータの二重担当");
                        seen[idx] = true;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&b| b), "アキュムレータの取りこぼし");

        // 3. 行ストライド (bf16 の ldm は 16 バイトの倍数) と smem の見積り。
        assert_eq!(TC_SN2_LD % 8, 0);
        assert_eq!(TC_SN2_LD * 2 % 16, 0);
        assert!(TC_SN2_LD > TC_SN2_BK, "詰め物が無い (バンク衝突)");
        let smem = 2 * TC_SN2_BN * TC_SN2_LD * 2 + 2 * TC_SN2_BM * TC_SN2_LD * 2;
        assert_eq!(smem, TC_SN2_SMEM_BYTES, "共有メモリの合計が .cu と違う");
        assert_eq!(smem, 20_480);
        assert!(smem <= 48 * 1024);
        assert!(
            8 * 256 * 4 <= 2 * TC_SN2_BM * TC_SN2_LD * 2,
            "端数ステージが sX に収まらない"
        );

        // 4. 端数タイル: 64 の倍数でない M で、ステージ経由の行数が 0..15。
        for nt in [17usize, 63, 64, 65, 127, 193, 194, 511, 512, 513] {
            for warp in 0..8usize {
                let wm = warp & 1;
                for i in 0..2usize {
                    let mrow = wm * 32 + i * 16;
                    if mrow < nt && mrow + 16 > nt {
                        let rows = nt - mrow;
                        assert!((1..=15).contains(&rows), "nt={nt} mrow={mrow}");
                    }
                }
            }
        }
    }

    /// task 183: `dno_pf3_gather` の索引 (dst[t][i] = src[t][ld + off + i]) を
    /// CPU で写して、融合バッファから v 行だけが抜けることを確かめる。
    #[test]
    fn pf3_gather_index_mirror() {
        let (rows, ld, off, n) = (5usize, 13usize, 10usize, 3usize);
        let src: Vec<f32> = (0..rows * ld).map(|i| i as f32).collect();
        let mut dst = vec![0f32; rows * n];
        for t in 0..rows {
            for i in 0..n {
                dst[t * n + i] = src[t * ld + off + i];
            }
        }
        for t in 0..rows {
            for i in 0..n {
                assert_eq!(dst[t * n + i], (t * ld + off + i) as f32, "t={t} i={i}");
            }
        }
        // gate の列位置: [q|gate] の gate は h*2*dh + dh + i。
        let (heads, dh) = (2usize, 4usize);
        let gate_ld = 2 * heads * dh + 2 * 2;
        let mut qg = vec![0f32; 3 * gate_ld];
        for t in 0..3 {
            for c in 0..gate_ld {
                qg[t * gate_ld + c] = (t * 1000 + c) as f32;
            }
        }
        for t in 0..3 {
            for h in 0..heads {
                for i in 0..dh {
                    let got = qg[t * gate_ld + (h * 2 + 1) * dh + i];
                    assert_eq!(got, (t * 1000 + (h * 2 + 1) * dh + i) as f32);
                }
            }
        }
    }

    /// v2 カーネルの索引算術を CPU ミラーで確かめる
    /// (`kernels_gemm_tc2.cu` のステージ担当・warp の担当範囲・端数の条件を
    /// 1 対 1 で写す)。GPU 実測の代わりではない。
    #[test]
    fn pf_tc2_tile_index_mirror() {
        // 1. 重み / 入力タイル: 256 スレッド × 8 要素 × 2 組が [128][32] を
        //    ちょうど 1 回覆う。行 = (tid>>2) + 64*hf、列 j = (tid&3)*8。
        for (bm, what) in [(TC2_BN, "重み"), (TC2_BM, "入力")] {
            let mut seen = vec![false; bm * TC2_BK];
            for tid in 0..256usize {
                for hf in 0..2usize {
                    let row = (tid >> 2) + hf * 64;
                    let j = (tid & 3) * 8;
                    assert!(row < bm && j + 8 <= TC2_BK, "{what}: row={row} j={j}");
                    for u in 0..8usize {
                        let idx = row * TC2_BK + j + u;
                        assert!(!seen[idx], "{what}スロットの二重書き row={row} j={j}");
                        seen[idx] = true;
                    }
                }
            }
            assert!(seen.iter().all(|&b| b), "{what}スロットの取りこぼし");
        }

        // 2. warp のアキュムレータ (4 トークン群 × 2 行群 × 2 × 4 タイル) が
        //    [128][128] をちょうど 1 回覆い、行/列の先頭が 16 の倍数であること。
        let mut seen = vec![false; TC2_BM * TC2_BN];
        let mut warps = 0usize;
        for warp in 0..8usize {
            warps += 1;
            let (wm, wn) = (warp & 3, warp >> 2);
            for i in 0..2usize {
                let mrow = wm * 32 + i * 16;
                assert_eq!(mrow % 16, 0);
                for j in 0..4usize {
                    let ncol = wn * 64 + j * 16;
                    assert_eq!(ncol % 16, 0);
                    for r in 0..16usize {
                        for c in 0..16usize {
                            let idx = (mrow + r) * TC2_BN + ncol + c;
                            assert!(!seen[idx], "アキュムレータの二重担当");
                            seen[idx] = true;
                        }
                    }
                }
            }
        }
        assert_eq!(warps, 8, "warp 数が 8 でない");
        assert!(seen.iter().all(|&b| b), "アキュムレータの取りこぼし");

        // 3. 行ストライドの詰め物: ldm は bf16 で 16 バイトの倍数が要る
        //    (wmma::load_matrix_sync の前提)。40 * 2 = 80 B は 16 の倍数。
        assert_eq!(TC2_LD % 8, 0, "TC2_LD が 8 の倍数でない (ldm の 16B 整列)");
        assert_eq!(TC2_LD * 2 % 16, 0);
        assert!(TC2_LD > TC2_BK, "詰め物が無い (バンク衝突)");

        // 4. 静的共有メモリの見積りがカーネル側の宣言と一致し、48 KiB に収まる。
        let smem = 2 * TC2_BN * TC2_LD * 2 + 2 * TC2_BM * TC2_LD * 2;
        assert_eq!(smem, TC2_SMEM_BYTES, "共有メモリの合計が .cu と違う");
        assert_eq!(smem, 40_960);
        assert!(smem <= 48 * 1024, "静的 smem の上限 (48 KiB) を超える");
        // 端数ステージ (warp ごとに 16x16 f32) が sX に収まること。
        let stage_floats = 8 * 256;
        assert!(
            stage_floats * 4 <= 2 * TC2_BM * TC2_LD * 2,
            "端数ステージが sX に収まらない"
        );

        // 5. 端数タイル: 128 の倍数でない M で、ステージ経由の行数が 0..15。
        for nt in [1usize, 15, 16, 17, 127, 128, 129, 2047, 2048] {
            for warp in 0..8usize {
                let wm = warp & 3;
                for i in 0..2usize {
                    let mrow = wm * 32 + i * 16;
                    if mrow < nt && mrow + 16 > nt {
                        let rows = nt - mrow;
                        assert!(rows >= 1 && rows <= 15, "nt={nt} mrow={mrow}");
                    }
                }
            }
        }
    }
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
            let msg = format!("{e:#}");
            // この経路は 1 本失敗した時点で GPU 全体を止めるので、抑制せず必ず 1 行出す
            // (プロセスあたり 1 回しか来ない)。数えるほうは共通の集計に任せる。
            crate::gpu_budget::note_weight_failure(uid, &msg);
            tracing::error!(
                "CUDA q4 weight upload failed, using CPU permanently (この 1 本で GPU 経路全体を停止): {msg}"
            );
            crate::gpu_budget::note_gpu_broken("q4 weight upload failed");
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
pub(crate) const Q4_M_TILE: usize = 1024;


/// [`Q4_M_TILE`] の上限。中間バッファがタイル行数に比例するので 8192 行で止める
/// (27B MLP の CH_GA は約 1.1 GiB — 13k 行を一度に置く 2.7 GiB には届かない)。
pub(crate) const Q4_M_TILE_MAX: usize = 8192;


/// タイル幅の刻み。`dno_gemm_q4_0` の行ブロック (Q4_GM = 64) に揃える — 途中の
/// タイルはすべて 64 の倍数行になり、grid.x が端数ブロックを出さない。
pub(crate) const Q4_M_TILE_STEP: usize = 64;


/// [`Q4_M_TILE`] の実効値 (env `DENNO_Q4_M_TILE` があればそちら)。プロセス内で
/// 1 回だけ読む (serve の起動時に env を決める運用なので、途中変更は見ない)。
pub(crate) fn q4_m_tile() -> usize {
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
pub(crate) fn q4_tail_merge_below() -> usize {
    max_m()
}


/// s168 (P7): 1 タイルが受ける最大行数 (端数併合を含む)。中間バッファはこの
/// 行数で**ループの前に**確保する — 途中で grow させると、先行 launch が読んで
/// いるバッファを解放しうる (`Scratch::ensure` は grow-only の入れ替え)。
pub(crate) fn q4_tile_max_rows(tile: usize) -> usize {
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
pub(crate) fn q4_group_rows(r0: usize, m: usize, tile: usize) -> usize {
    let left = m - r0;
    let mr = tile.min(left);
    let rem = left - mr;
    if rem > 0 && rem <= q4_tail_merge_below() {
        left
    } else {
        mr
    }
}


/// s132 (Task B1): [`q4_dispatch_tile`] の GEMV が 1 block で受ける最大行数。
/// `kernels_int8.cu` の `dno_gemv_q4_0mr` (acc[8]) と対で動く。
/// s133 以降、本番の GEMV 経路は `dno_gemv_q4_0c*` (下の [`Q4_GEMV_C`]) で、
/// `dno_gemv_q4_0mr` はビット一致の参照 (テスト) として残してある。
pub(crate) const Q4_MR_MAX: usize = 8;


/// s133 (Task B1, attempt 2): 列ブロック版 Q4 GEMV の C (1 回の A 読みで
/// 何列の重みを回すか)。`kernels_int8.cu` の `dno_gemv_q4_0c2/c4/c8` と対。
/// 1 MAC あたりの命令数は 1 + 1/C になる (C=1 の従来形は 2)。
/// 2 / 8 を試すときはここを変える (PTX には 3 本とも入っている)。
/// ビルドし直さずに比べるときは `DENNO_Q4_GEMV_C=2|4|8` で上書きする。
pub(crate) const Q4_GEMV_C: usize = 4;


/// [`Q4_GEMV_C`] の実効値 (env `DENNO_Q4_GEMV_C` があればそちら)。2/4/8 以外は
/// 既定へ落とす (不正な値で panic させない)。
pub(crate) fn q4_gemv_c() -> usize {
    match std::env::var("DENNO_Q4_GEMV_C").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(c @ (2 | 4 | 8)) => c,
        _ => Q4_GEMV_C,
    }
}


/// [`Q4_GEMV_C`] に対応する PTX エントリの index (0 = c2, 1 = c4, 2 = c8)。
pub(crate) fn q4_gemv_c_index(c: usize) -> usize {
    match c {
        2 => 0,
        4 => 1,
        8 => 2,
        _ => panic!("Q4_GEMV_C must be 2, 4 or 8"),
    }
}


/// P17i: 大きい m (prefill) 用 — tiled GEMM カーネル (dno_gemm_i8) で実行。
/// パイプラインは linear_pinned と同一 (H2D → quantize → gemm → D2H → sync ×1)。
/// 整数累積は順序不変なのでタイル化しても CPU とビット一致。
pub(crate) fn linear_gemm(
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
pub(crate) fn linear_gemm_into(
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


/// task 152: `dno_gemm_bf16_mt` のハンドル (prefill の m トークン一括)。
pub(crate) fn gemm_bf16_mt_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    static FN: OnceLock<Option<LgemmMtFn>> = OnceLock::new();
    FN.get_or_init(|| {
        let lib = lib().ok()?;
        let mut f: CUfunction = std::ptr::null_mut();
        let r = unsafe {
            (lib.module_get_function)(
                &mut f,
                ctx.module_elem,
                b"dno_gemm_bf16_mt\0".as_ptr() as *const c_char,
            )
        };
        if r != 0 {
            None
        } else {
            Some(LgemmMtFn(f))
        }
    })
    .as_ref()
    .map(|h| h.0)
}


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
    let f = matvec_f32_fn(ctx)?;
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
            _ => match DevBuffer::from_slice_tagged(w, crate::mem_audit::TAG_LMHEAD) {
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
// task 69: Flash-Next の MoE ルータ (ffn_gate_inp, f32 512x2560) を GPU へ
// ===========================================================================
// 前提 (main の実測・118 tok / DENNO_Q4_PROFILE=1): router = 1,962.7 ms =
// 16.6 ms/token。48 層 x (512 行 x k=2560) の f32 GEMV = 63 MMAC/token で、
// CPU の実効 3.8 GMAC/s は f32 の帯域 (重み 250 MB/token・50 GB/s なら 5 ms)
// からも大きく外れている。
//
// カーネルは新規に作らず、s49 の fp32 lm_head と同じ `dno_matvec_f32` を流用する
// (.cu / PTX は 1 バイトも変えない = このコンテナでは PTX を再生成できない):
//   - 1 行 = 1 ブロック (grid=(n,1,1))・k を d=0..k-1 の昇順に
//     `s = __fadd_rn(s, __fmul_rn(h[d], row[d]))`
//   - PTX を確認: `mul.rn.f32` + `add.rn.f32` のみ。`fma.rn` 0 個・`.ftz` 0 個
//     (非正規化も保存)。加算の連鎖は d の昇順そのもの (4 倍展開でも鎖は直列)。
//   - CPU 側 (GgufLinear の F32 経路) は同じ 4 バイトを `f32::from_le_bytes` で
//     開き、`acc += row[i] * x[i]` を i 昇順で 1 要素ずつ丸める (FMA 無し・行並列
//     でも行内の順序は不変)。同じ演算列なので**ビット一致**する。
//
// 重みは GGUF ブロックのレジストリと同じ LRU を共有する (`DENNO_GPU_WEIGHT_MB`
// の枠は 1 本。48 層 x 5.2 MB = 250 MB)。枠が足りない・GPU が無い・確保に
// 失敗した場合は全部 CPU 参照へ落ちる (fail-closed・panic しない)。

/// task 69: ルータ GPU 経路のスコープド恒久ラッチ (他の GPU 経路へ波及させない)。
static ROUTER_GPU_BROKEN: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// task 69: ルータ GPU の入口に入った回数 (テスト専用)。
///
/// 「ゲートを足したが分岐が通っていない」を GPU 無しで検出するために、
/// `cuda_enabled()` の判定より**前**で数える (`moe_ffn_quant` → `GgufLinear::gemv`
/// → ここ、の配線が生きていることの証拠)。
#[cfg(test)]
pub(crate) static ROUTER_GPU_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);


#[cfg(test)]
pub(crate) fn router_gpu_attempts() -> usize {
    ROUTER_GPU_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed)
}


#[cfg(test)]
pub(crate) fn router_gpu_attempts_reset() {
    ROUTER_GPU_ATTEMPTS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// task 127: ルータ GPU のカーネル選択 (純関数・GPU 無しで検証可)。
///
/// ルータ (`ffn_gate_inp`) の実形は 2560 -> 512 の f32 GEMV で、**k >= n の長い行**。
/// 逐次版 `dno_matvec_f32` は 1 ブロック = 1 行・`block=(1,1,1)` なので並列度が
/// 行数 n=512 しかなく、1 スレッドが k=2560 を昇順に舐める。task 69 の却下
/// (2026-10-31) はこの形のまま測っていた。`_r` は 1 ブロック = 1 行のまま
/// block 256 で k をスレッド分割し、warp shuffle の固定木で縮約する
/// (atomic 無し = 実行ごとに同一結果)。丸め順が変わるので CPU の逐次累積とは
/// ビット一致しない ([`mix_use_reduce_kernel`] と同じ扱い・速度優先)。
/// `has_r` は PTX に `_r` が無いとき false となり逐次版に落ちる (fail closed)。
pub(crate) fn router_use_reduce_kernel(k: usize, n: usize, has_r: bool) -> bool {
    has_r && n > 0 && k >= n
}


/// task 127: `DENNO_ROUTER_GPU_REDUCE` の解釈 (純関数)。**既定 on**、明示的な
/// 偽値だけ off。
///
/// ルータ GPU 経路の中だけで効く (`DENNO_ROUTER_GPU` 自体が既定 off なので、
/// 何も設定しないプロセスのモデル挙動は変わらない)。`DENNO_ROUTER_GPU=1` +
/// `DENNO_ROUTER_GPU_REDUCE=0` で逐次版 (task 69 と同じ丸め) に戻せる。
pub(crate) fn router_gpu_reduce_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}


/// ルータ GPU 経路で `_r` を使ってよいか (`DENNO_ROUTER_GPU_REDUCE`、既定 on)。
pub(crate) fn router_gpu_reduce_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        router_gpu_reduce_from_value(std::env::var("DENNO_ROUTER_GPU_REDUCE").ok().as_deref())
    })
}


/// task 127: `_r` を選んだことをプロセスで 1 回だけログる (lab のログで分岐の
/// 発火を確認するため。`router=` の数値だけでは逐次版との区別が付かない)。
static ROUTER_GPU_REDUCE_LOGGED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// task 69: ルータ用スクラッチ。lm_head の `Scratch` と分ける (形が違うので
/// 同じ枠を使うと毎回作り直しになる)。x = [k] f32・out = [n] f32。
pub(crate) struct RouterScratch {
    x: Option<DevBuffer>,
    pub(crate) out: Option<DevBuffer>,
    hx: Option<PinMem>,
    pub(crate) ho: Option<PinMem>,
}


thread_local! {
    pub(crate) static ROUTER_SC: std::cell::RefCell<RouterScratch> = std::cell::RefCell::new(RouterScratch {
        x: None,
        out: None,
        hx: None,
        ho: None,
    });
}


/// task 182: F32 / BF16 / F16 の重みを **行優先 f32 に開いて** VRAM へ上げる。
///
/// PLE の `ple_key` / `ple_value` は実ファイルで F32 か BF16 (目録は Q8_0) で、
/// 量子型専用の `pf_ple_proj` は形の段階で断る。BF16 -> f32 は上位 16 bit の
/// 複製 (誤差なし)、F16 -> f32 は `dequant_row_into`。VRAM 上は常に f32 なので、
/// レジストリの型は F32 で登録する (ルータの 1 行 GEMV と同じ表を共有)。
pub(crate) fn weights_for_gemm_f32(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    src: crate::kquant::GgmlType,
) -> Result<GgufWRef> {
    use crate::kquant::GgmlType as T;
    let reg = gguf_registry();
    let mut g = reg
        .lock()
        .map_err(|_| anyhow!("gguf weights registry poisoned"))?;
    if g.map.contains_key(&uid) {
        let e = g.map.get(&uid).expect("contains_key");
        let (ek, en, ety) = (e.k, e.n, e.ty);
        if (ek, en, ety) != (k, n, T::F32) {
            bail!(
                "uid 重複: 同一 uid {uid} に別の形状/型 (登録済み k={ek} n={en} ty={ety:?} / 今回 k={k} n={n} ty=F32)"
            );
        }
        g.lru.touch(uid);
        g.lru.pin(uid);
        let e = g.map.get(&uid).expect("pinned");
        let w = unsafe { std::mem::transmute::<&GpuGgufW, &'static GpuGgufW>(e.as_ref()) };
        return Ok(GgufWRef { w, uid });
    }
    let want = n
        .checked_mul(k)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow!("f32: n * k * 4 が usize を溢れる"))?;
    let bpp = match src {
        T::F32 => 4usize,
        T::Bf16 | T::F16 => 2,
        other => bail!("f32 常駐: 未対応の元型 {other:?} (F32 / BF16 / F16 のみ)"),
    };
    let src_want = n
        .checked_mul(k)
        .and_then(|v| v.checked_mul(bpp))
        .ok_or_else(|| anyhow!("f32: n * k * {bpp} が usize を溢れる"))?;
    if bytes.len() != src_want {
        bail!(
            "f32: 元型 {src:?} のバイト数 {} が n * k * {bpp} = {src_want} と違う (k={k}, n={n})",
            bytes.len()
        );
    }
    // P46 と同じ: 確保の前に上限と空き VRAM を見て古い uid を解放する。
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
    // 生バイトのスライスは 4 バイト整列を保証しない (`Vec<u8>` / mmap の途中) ので、
    // CPU と同じ `from_le_bytes` で f32 に開いてから上げる (値は 1 bit も変わらない)。
    // task 182: BF16 は上位 16 bit の複製 (誤差なし)、F16 は dequant_row_into。
    let mut host: Vec<f32> = Vec::with_capacity(k * n);
    match src {
        T::F32 => {
            for c in bytes.chunks_exact(4) {
                host.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        T::Bf16 => {
            for c in bytes.chunks_exact(2) {
                let b = u16::from_le_bytes([c[0], c[1]]);
                host.push(f32::from_bits((b as u32) << 16));
            }
        }
        T::F16 => {
            let mut row = vec![0f32; k];
            for r in 0..n {
                let s = r * k * 2;
                crate::kquant::dequant_row_into(T::F16, &bytes[s..s + k * 2], &mut row);
                host.extend_from_slice(&row);
            }
        }
        other => bail!("f32 常駐: 未対応の元型 {other:?}"),
    }
    if host.len() != k * n {
        bail!("f32 常駐: 展開後の要素数 {} が k*n = {} と違う", host.len(), k * n);
    }
    let buf = match DevBuffer::from_slice_tagged(&host, crate::mem_audit::TAG_ROUTER) {
        Ok(b) => b,
        Err(e) => {
            g.lru.note_remove(uid);
            return Err(e);
        }
    };
    tracing::info!(
        "CUDA f32 weight upload (uid={uid}): router k={k} n={n} ({:.1} MB)",
        want as f64 / 1e6
    );
    g.map.insert(uid, Box::new(GpuGgufW { buf, ty: T::F32, k, n }));
    g.lru.pin(uid);
    let resident = g.map.len();
    let used = g.lru.used_bytes();
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
        let ev = if dropped.is_empty() {
            String::new()
        } else {
            format!(
                " / 今回退避 {} 本 {:.1} MiB",
                dropped.len(),
                freed as f64 / (1 << 20) as f64
            )
        };
        tracing::info!(
            target: "denno_engine::gpu_w",
            "[gpu-w] uid={uid} f32 router k={k} n={n} {:.2} MB / 常駐 {} 本 {:.1} MiB{ev} / {free}",
            want as f64 / 1e6,
            resident,
            used as f64 / (1 << 20) as f64,
        );
    }
    Ok(GgufWRef { w, uid })
}


thread_local! {
    /// task 192: `maybe_gpu_router_gemv_begin` が積んだ出力行数 (0 = 無し)。
    pub(crate) static ROUTER_PENDING_N: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}


/// task 364: qchain がルータの控え (**積む前**の判定) を見つけた回数。
///
/// task 307 の `DENNO_ROUTER_IN_MIX` は BF16 の chain3 にしか入っていなかった。
/// 新しい既定モデルは mix が Q8_0 なので qchain を通る = ルータ GEMV が
/// 鎖の外に残っていた。ここは「載せにいった回数」を GPU 無しでも数える。
#[cfg(test)]
pub(crate) static ROUTER_IN_QCHAIN_ARMS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);


/// task 364: [`ROUTER_IN_QCHAIN_ARMS`] を読む (テスト専用)。
#[cfg(test)]
pub(crate) fn router_in_qchain_arms() -> usize {
    ROUTER_IN_QCHAIN_ARMS.load(std::sync::atomic::Ordering::Relaxed)
}


/// task 364: カウンタを 0 に戻す (テスト専用)。
#[cfg(test)]
pub(crate) fn router_in_qchain_arms_reset() {
    ROUTER_IN_QCHAIN_ARMS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// task 364: ルータ GEMV を qchain の末尾に実際に積んだ回数 (1 プロセス通算)。
/// 診断行 `[mix-qchain]` の `rtr=` に出る (0 なら載っていない)。
pub(crate) static ROUTER_IN_QCHAIN_HITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);


/// task 307: 次の mix の鎖にルータ GEMV を載せるための控え。
pub(crate) struct RouterArm {
    pub(crate) uid: u64,
    /// ルータ重みのホスト生バイト (F32)。**同じ層・同じトークンの mix の中で
    /// 消費する** 前提で借りる (モデルの重みは mix の間ずっと生きている)。
    pub(crate) ptr: *const u8,
    pub(crate) len: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
    pub(crate) layer: usize,
    pub(crate) pos: u64,
}


thread_local! {
    pub(crate) static ROUTER_ARM: std::cell::RefCell<Option<RouterArm>> =
        const { std::cell::RefCell::new(None) };
    /// task 307: mix の鎖が作った logits の (uid, k, n)。0 = 無し。
    pub(crate) static ROUTER_FILLED: std::cell::Cell<(u64, usize, usize)> =
        const { std::cell::Cell::new((0, 0, 0)) };
    /// task 307: `ROUTER_FILLED` の中身はもう同期済みか (finish で待たない)。
    pub(crate) static ROUTER_SYNCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}


/// task 69 / 192: ルータ (f32・m=1) の GEMV を **同期せずに** 積む。
///
/// `maybe_gpu_router_gemv_into` の前半。`true` を返したら、呼び出し側は別の
/// ホスト仕事 (共有 expert の enqueue など) を挟んでから
/// [`maybe_gpu_router_gemv_finish`] で待つ。`false` = 従来経路で計算すること
/// (GPU 無効・PTX 未収録・予算外・実行時エラー)。
pub fn maybe_gpu_router_gemv_begin(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
) -> bool {
    use std::sync::atomic::Ordering as AO;
    #[cfg(test)]
    ROUTER_GPU_ATTEMPTS.fetch_add(1, AO::Relaxed);
    if !cuda_enabled() || ROUTER_GPU_BROKEN.load(AO::Relaxed) {
        return false;
    }
    if k == 0
        || n == 0
        || x.len() != k
        || bytes.len() != k.saturating_mul(n).saturating_mul(4)
        || k > i32::MAX as usize
        || n > i32::MAX as usize
    {
        return false;
    }
    // task 307: mix の鎖 (`DENNO_ROUTER_IN_MIX`) がこの層の logits を既に作って
    // いて、鎖の同期でホストへ届いていれば、ここでは何も積まない。uid / k / n が
    // 違えば (別の層の残り) 捨てて従来経路へ。
    let filled = ROUTER_FILLED.with(|c| c.get());
    if filled.0 != 0 {
        if filled == (uid, k, n) {
            // 使い切り (次の層・次のトークンで古い logits を使わない)。
            ROUTER_FILLED.with(|c| c.set((0, 0, 0)));
            ROUTER_SYNCED.with(|c| c.set(true));
            ROUTER_PENDING_N.with(|c| c.set(n));
            return true;
        }
        ROUTER_FILLED.with(|c| c.set((0, 0, 0)));
        ROUTER_SYNCED.with(|c| c.set(false));
    } else {
        // 印が無い回は「鎖の値」を使わない (前回の SYNCED が残っていても無効)。
        ROUTER_SYNCED.with(|c| c.set(false));
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if let Err(e) = ctx.make_current() {
        tracing::error!("CUDA cuCtxSetCurrent failed (router f32), using CPU: {e:#}");
        ROUTER_GPU_BROKEN.store(true, AO::Relaxed);
        return false;
    }
    // カーネルハンドル (未収録 PTX は None をキャッシュ = ラッチしない)。
    // task 127: ルータは k=2560 >= n=512 の長い行なので、`_r` (1 ブロック = 1 行・
    // block 256・固定木縮約) が選べるならそちらを使う。逐次版は 1 ブロック =
    // 1 スレッドで並列度が行数 n しかない (task 69 の却下はこの形だった)。
    // `_r` が未収録 / `DENNO_ROUTER_GPU_REDUCE=0` なら逐次版に落ちる (fail closed)。
    let f_seq = matvec_f32_fn(ctx);
    let f_red = matvec_f32_r_fn(ctx);
    let use_reduce = router_gpu_reduce_enabled() && router_use_reduce_kernel(k, n, f_red.is_some());
    if !use_reduce && f_seq.is_none() {
        return false;
    }
    let prof = crate::model::devprof::on();
    let up_before = if prof {
        crate::gpu_budget::weight_upload_bytes()
    } else {
        0
    };
    let w = match weights_for_f32(uid, bytes, k, n) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(
                "router f32 weight register failed (uid={uid}), using CPU permanently: {e:#}"
            );
            ROUTER_GPU_BROKEN.store(true, AO::Relaxed);
            return false;
        }
    };
    let lib = match lib() {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("CUDA lib load failed (router f32), using CPU: {e:#}");
            ROUTER_GPU_BROKEN.store(true, AO::Relaxed);
            return false;
        }
    };
    let r = ROUTER_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let RouterScratch {
            x: dx,
            out: dout,
            hx,
            ho,
        } = &mut *s;
        let bytes_x = k * 4;
        let bytes_out = n * 4;
        Scratch::ensure(dx, bytes_x)?;
        Scratch::ensure(dout, bytes_out)?;
        Scratch::ensure_pin(hx, bytes_x)?;
        Scratch::ensure_pin(ho, bytes_out)?;
        let (dx, dout) = (dx.as_ref().unwrap(), dout.as_ref().unwrap());
        let (hx, ho) = (hx.as_mut().unwrap(), ho.as_mut().unwrap());
        hx.as_mut_slice::<f32>()[..k].copy_from_slice(x);
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
            let (mut ph, mut pw, mut po) = (dx.ptr(), w.buf.ptr(), dout.ptr());
            let (mut kc, mut nc) = (k as c_int, n as c_int);
            let mut args = [
                &mut ph as *mut _ as *mut c_void,
                &mut pw as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut nc as *mut _ as *mut c_void,
            ];
            // task 127: 長い行 (k >= n) は `_r` (1 ブロック = 1 行・block 256 で
            // k を分割し warp shuffle の固定木で縮約・決定論)、それ以外と
            // ゲート off は従来どおり逐次版 (カーネルが block 内全スレッドで
            // 同一計算するため block=(1,1,1))。grid.x は 2^31-1 まで合法。
            let (mv_fn, mv_thr): (CUfunction, u32) = if use_reduce {
                (f_red.expect("use_reduce は has_r=true を要求する"), 256)
            } else {
                (f_seq.expect("use_reduce=false は逐次版のハンドルを要求する"), 1)
            };
            if use_reduce && !ROUTER_GPU_REDUCE_LOGGED.swap(true, AO::Relaxed) {
                tracing::info!(
                    "[router-gpu] dno_matvec_f32_r を使用 (block 256・k={k} n={n})"
                );
            }
            ctx.launch_async(mv_fn, (n as u32, 1, 1), (mv_thr, 1, 1), 0, &mut args)?;
            check(
                (lib.memcpy_dtoh_async)(ho.ptr, dout.ptr(), bytes_out, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync",
            )?;
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        // task 192: 待たずに返す (待ちは `maybe_gpu_router_gemv_finish`)。
        ROUTER_PENDING_N.with(|c| c.set(n));
        Ok(())
    });
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            if prof {
                let up_w = crate::gpu_budget::weight_upload_bytes().saturating_sub(up_before);
                crate::model::devprof::note_bytes(k as u64 * 4 + up_w, n as u64 * 4);
            }
            true
        }
        Err(e) => {
            tracing::error!("CUDA router f32 gemv failed, using CPU permanently: {e:#}");
            ROUTER_GPU_BROKEN.store(true, AO::Relaxed);
            false
        }
    }
}


/// task 192: [`maybe_gpu_router_gemv_begin`] の後半 (同期 + ホストへの写し)。
///
/// `false` = 待ちに失敗した (呼び出し側は CPU 経路で計算し直す)。
pub fn maybe_gpu_router_gemv_finish(out: &mut [f32]) -> bool {
    use std::sync::atomic::Ordering as AO;
    let n = ROUTER_PENDING_N.with(|c| c.replace(0));
    if n == 0 || out.len() != n {
        return false;
    }
    // task 307: mix の鎖が積んで、その同期で既に届いている場合 (待たない)。
    if ROUTER_SYNCED.with(|c| c.replace(false)) {
        ROUTER_FILLED.with(|c| c.set((0, 0, 0)));
        let r = ROUTER_SC.with(|cell| -> Result<()> {
            let s = cell.borrow();
            let ho = s
                .ho
                .as_ref()
                .ok_or_else(|| anyhow!("router scratch が無い (mix の鎖を飛ばした)"))?;
            out.copy_from_slice(&ho.as_slice::<f32>()[..n]);
            Ok(())
        });
        if r.is_ok() {
            set_last_linear_ran_on_gpu(true);
            return true;
        }
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    let r = (|| -> Result<()> {
        // task 155: 待つのは null ストリームのぶんだけ (重ね打ち中の shared
        // expert のサイドストリームは待たない)。イベントが無ければ従来どおり。
        sync_default_or_ctx(ctx, router_fused_enabled())?;
        ROUTER_SC.with(|cell| -> Result<()> {
            let s = cell.borrow();
            let ho = s
                .ho
                .as_ref()
                .ok_or_else(|| anyhow!("router scratch が無い (begin を飛ばした)"))?;
            out.copy_from_slice(&ho.as_slice::<f32>()[..n]);
            Ok(())
        })
    })();
    match r {
        Ok(()) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        Err(e) => {
            tracing::error!("CUDA router f32 gemv (finish) failed, using CPU permanently: {e:#}");
            ROUTER_GPU_BROKEN.store(true, AO::Relaxed);
            false
        }
    }
}


/// task 69: ルータ (f32・m=1) を GPU で計算して `out` に書く (begin + finish)。
///
/// `false` = 呼び出し側は CPU 参照で計算すること (GPU 無効・PTX 未収録・予算外・
/// 実行時エラー)。逐次版 `dno_matvec_f32` のときは CPU 経路とビット一致するので
/// どちらに落ちても出力は変わらない。task 127 で既定は並列リダクション版
/// `dno_matvec_f32_r` (丸め順が違う = ビット一致しない・決定論は保つ)。
pub fn maybe_gpu_router_gemv_into(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    maybe_gpu_router_gemv_begin(uid, bytes, k, n, x) && maybe_gpu_router_gemv_finish(out)
}


pub(crate) fn gemm_f32_mt_fn(ctx: &CudaCtx) -> Option<CUfunction> {
    GEMM_F32_MT_FNS
        .get_or_init(|| {
            let lib = lib().ok()?;
            let mut f: CUfunction = std::ptr::null_mut();
            let r = unsafe {
                (lib.module_get_function)(
                    &mut f,
                    ctx.module_elem,
                    b"dno_gemm_f32_mt\0".as_ptr() as *const c_char,
                )
            };
            if r != 0 {
                None
            } else {
                Some(LgemmMtFn(f))
            }
        })
        .as_ref()
        .map(|h| h.0)
}


// ---------------------------------------------------------------------------
// task 182: prefill の router / PLE 射影 (F32 重み × m トークン) を GPU で回す
// ---------------------------------------------------------------------------
//
// task 170 (br170) が層優先 prefill 用に書いたものと同じ経路。ここでは通常の
// チャンク順 prefill から、層ごと・チャンクごとに 1 回ずつ呼ぶ:
//   - router (`ffn_gate_inp`、実形 2,560 x 512 の F32) は 14k の prefill で 2.8 s
//   - PLE の key / value 射影 (F32、`blk.1.ple_key/value`) は合わせて約 1.3 s
// どちらも CPU の行バッチで回っていた。重みは `weights_for_f32` の常駐
// (decode のルータ GEMV と同じ uid 表) に載せ、既存の `dno_gemm_f32_mt` で
// m トークンをまとめて回す (.cu / PTX の変更なし)。
//
// 数値は FMA + タイル順なので CPU の `mul + add` 逐次とはビット一致しない
// (prefill の行列積は速度優先の契約)。`false` = CPU のバッチ経路へ。

pub(crate) struct F32GemmBatchSc {
    x: Option<DevBuffer>,
    out: Option<DevBuffer>,
    /// task 296: 大きい D2H 用の pinned 置き場 (grow-only)。
    pin: Option<PinMem>,
}


thread_local! {
    pub(crate) static F32_GEMM_BATCH_SC: std::cell::RefCell<F32GemmBatchSc> = const {
        std::cell::RefCell::new(F32GemmBatchSc { x: None, out: None, pin: None })
    };
}


/// `DENNO_F32_GEMM_PIN_D2H`: 1 MiB 以上の D2H を pinned 経由の非同期コピーに
/// する (task 296、既定 **off**)。
///
/// 層優先 prefill ではこの関数の D2H が毎層・毎チャンク走る:
/// router (3,826 x 512 = 7.8 MB/層) と PLE の key/value
/// (3,826 x 320 + 3,826 x 2560 = 22 MB/チャンク x 96)。pageable の
/// `cuMemcpyDtoH` は 1 枚 6 GB/s 前後しか出ないので、pinned へ非同期で落として
/// からホストへ写す (デバイスから出るバイト列は同じ = 値は不変)。
///
/// lab t296c: 3,826 cold は base 3.41 / 既定 (両方 on) 3.24 で、内訳では
/// `other` が -170 ms。ただしこの差は同時に on にした [`PIN_GROW_ENV`] の
/// ぶんで、このゲート単独の差は見えていない (`ple` は 154 → 160 ms で不変)。
/// 効いた証拠が無いので既定 off。
pub const F32_GEMM_PIN_D2H_ENV: &str = "DENNO_F32_GEMM_PIN_D2H";


/// この関数で pinned D2H を使う下限 (これ未満は従来の pageable)。
pub(crate) const F32_GEMM_PIN_D2H_MIN: usize = 1 << 20;


/// [`F32_GEMM_PIN_D2H_ENV`] の解釈 (明示的な真値だけ on)。
pub(crate) fn f32_gemm_pin_d2h_from_value(v: Option<&str>) -> bool {
    let low = v.map(|s| s.trim().to_ascii_lowercase());
    matches!(
        low.as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}


/// 大きい D2H を pinned 経由にするか (既定 off)。
fn f32_gemm_pin_d2h_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        f32_gemm_pin_d2h_from_value(std::env::var(F32_GEMM_PIN_D2H_ENV).ok().as_deref())
    })
}


/// task 182: `maybe_gpu_f32_gemm_batch_into` の入口に入った回数 (テスト専用)。
///
/// このコンテナに GPU は無いので、計算は `cuda_enabled()` の所で CPU へ落ちる。
/// ゲートと配線が生きていることを「入口の回数」で数える (decode の
/// `router_gpu_attempts` と同じ規約)。
#[cfg(test)]
pub(crate) static F32_GEMM_BATCH_ATTEMPTS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);


#[cfg(test)]
pub(crate) fn f32_gemm_batch_attempts() -> usize {
    F32_GEMM_BATCH_ATTEMPTS.load(std::sync::atomic::Ordering::Relaxed)
}


#[cfg(test)]
pub(crate) fn f32_gemm_batch_attempts_reset() {
    F32_GEMM_BATCH_ATTEMPTS.store(0, std::sync::atomic::Ordering::Relaxed);
}


/// [`maybe_gpu_f32_gemm_batch_into`] と [`maybe_gpu_f32_gemm_batch_dev_into`] の本体。
pub(crate) fn f32_gemm_batch_run(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    src: BatchIn<'_>,
    nt: usize,
    out: &mut [f32],
    src_ty: crate::kquant::GgmlType,
) -> bool {
    if !cuda_enabled() {
        return false;
    }
    let bpp = match src_ty {
        crate::kquant::GgmlType::F32 => 4usize,
        crate::kquant::GgmlType::Bf16 | crate::kquant::GgmlType::F16 => 2,
        _ => return false,
    };
    if k == 0
        || n == 0
        || nt < 2
        || out.len() != nt * n
        || bytes.len() != k.saturating_mul(n).saturating_mul(bpp)
        || k > i32::MAX as usize
        || n > i32::MAX as usize
        || nt > i32::MAX as usize
    {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        return false;
    }
    let Some(f) = gemm_f32_mt_fn(ctx) else {
        return false;
    };
    let w = match weights_for_gemm_f32(uid, bytes, k, n, src_ty) {
        Ok(w) => w,
        Err(_) => return false,
    };
    let lib = match lib() {
        Ok(l) => l,
        Err(_) => return false,
    };
    let r = F32_GEMM_BATCH_SC.with(|cell| -> Result<()> {
        let mut s = cell.borrow_mut();
        let F32GemmBatchSc { x: dx, out: dout, pin } = &mut *s;
        // task 209: デバイス入力のときは `x` の作業域 (と H2D) を丸ごと省く。
        let xp = match src {
            BatchIn::Host(xs) => {
                Scratch::ensure(dx, nt * k * 4)?;
                let dx = dx.as_ref().unwrap();
                dx.copy_from(xs)?;
                dx.ptr()
            }
            BatchIn::Dev(p) => p,
        };
        Scratch::ensure(dout, nt * n * 4)?;
        let dout = dout.as_ref().unwrap();
        let wp = w.buf.ptr();
        unsafe {
            let (mut pw, mut px, mut po) = (wp, xp, dout.ptr());
            let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, k as c_int, n as c_int);
            let mut args = [
                &mut pw as *mut _ as *mut c_void,
                &mut px as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut nt_c as *mut _ as *mut c_void,
                &mut k_c as *mut _ as *mut c_void,
                &mut n_c as *mut _ as *mut c_void,
            ];
            let grid = (
                ((n + GEMM_MT_ROWS - 1) / GEMM_MT_ROWS) as u32,
                ((nt + GEMM_MT_TOKS - 1) / GEMM_MT_TOKS) as u32,
                1,
            );
            ctx.launch_async(f, grid, (GEMM_MT_ROWS as u32, 16, 1), 0, &mut args)?;
            let bytes = nt * n * 4;
            // task 296: 大きい D2H は pinned へ非同期で落として 1 回の同期で待つ
            // (pageable の同期コピーは 6 GB/s 前後しか出ない)。
            if bytes >= F32_GEMM_PIN_D2H_MIN && f32_gemm_pin_d2h_on() {
                Scratch::ensure_pin(pin, bytes)?;
                let p = pin.as_ref().expect("ensure 済み");
                check(
                    (lib.memcpy_dtoh_async)(p.ptr, dout.ptr(), bytes, std::ptr::null_mut()),
                    "cuMemcpyDtoHAsync (f32 batch pin)",
                )?;
                ctx.sync()?;
                // pinned → `out` (デバイスから出たバイト列は従来と同じ)。
                std::ptr::copy_nonoverlapping(
                    p.ptr as *const u8,
                    out.as_mut_ptr() as *mut u8,
                    bytes,
                );
            } else {
                ctx.sync()?;
                // D2H は同期コピーで `out` へ直接 (中間 Vec / pinned を作らない)。
                check(
                    (lib.memcpy_dtoh)(
                        out.as_mut_ptr() as *mut c_void,
                        dout.ptr(),
                        bytes,
                    ),
                    "cuMemcpyDtoH (f32 batch)",
                )?;
            }
            crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        }
        Ok(())
    });
    r.is_ok()
}


/// `out = x @ W^T` (f32 重み) を**デバイスの出力域へ**書く (同期しない)。
///
/// [`maybe_gpu_f32_gemm_batch_into`] と同じカーネル・同じタイル
/// (`GEMM_MT_ROWS` x `GEMM_MT_TOKS`) で、D2H だけをしない版。
pub(crate) fn f32_gemm_dev_out(
    uid: u64,
    bytes: &[u8],
    k: usize,
    n: usize,
    dev_x: CUdeviceptr,
    nt: usize,
    dev_out: CUdeviceptr,
    src_ty: crate::kquant::GgmlType,
) -> bool {
    if dev_x == 0 || dev_out == 0 || k == 0 || n == 0 || nt < 2 {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        return false;
    }
    let Some(f) = gemm_f32_mt_fn(ctx) else {
        return false;
    };
    let w = match weights_for_gemm_f32(uid, bytes, k, n, src_ty) {
        Ok(w) => w,
        Err(_) => return false,
    };
    let wp = w.buf.ptr();
    unsafe {
        let (mut pw, mut px, mut po) = (wp, dev_x, dev_out);
        let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, k as c_int, n as c_int);
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
        ctx.launch_async(f, grid, (GEMM_MT_ROWS as u32, 16, 1), 0, &mut args)
            .is_ok()
    }
}


/// 1 行列ぶんのデバイス GEMM (`x` / `y` はデバイス。**同期しない**)。
///
/// 型と形が tensor core の前提に合えば `kernels_gemm_tc` / `kernels_gemm_tc2` を
/// 1 起動 (2,048 行まで)、合わなければ従来の量子 GEMM (`f_gemm_q`) を 128 行ずつ
/// 並べる。どちらも起動を並べるだけで、待つのは呼び出し側の最後の 1 回。
#[allow(clippy::too_many_arguments)]
pub(crate) fn pf2_gemm(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    kind: PfWKind,
    x: CUdeviceptr,
    y: CUdeviceptr,
    nt: usize,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    let PfWKind::Quant(ty, slot, rb, blk) = kind else {
        bail!("pf2_gemm: 量子型の重みだけを受ける")
    };
    if nt == 0 || n_in == 0 || n_out == 0 || n_in % blk != 0 {
        bail!("pf2_gemm: 形が不正 (nt={nt}, n_in={n_in}, n_out={n_out}, blk={blk})");
    }
    // 1. v2 (Q8_0 / Q4_K / Q5_K。型ごとにカーネルを選ぶ)。
    let tc_kind = pf2_tc_kind(ty, nt, n_in, n_out);
    if tc_kind == 2 && pf_tc2_gemm_on() {
        if let Some(tc2) = tc2_gemm_kernels() {
            return tc2_gemm_launch(ctx, tc2, ty, w, x, y, nt, n_in, n_out, rb);
        }
    }
    // 1b. task 181: N が小さい形 (shared expert 640 / out_proj 2560) の BN=64 版。
    //     ゲートが off・PTX が無いときは下の v1 / 従来カーネルへ落ちる。
    if tc_kind == 3 && pf_tc_smalln_on() {
        if let Some(sn) = tc_smalln_kernels() {
            return sn_gemm_launch(ctx, sn, ty, w, x, y, nt, n_in, n_out, rb);
        }
    }
    // 2. v1 (Q8_0 専用)。
    if matches!(ty, crate::kquant::GgmlType::Q8_0)
        && pf_tc_gemm_on()
        && pf_tc_gemm_eligible(nt, n_in, n_out, 1)
    {
        if let Some(tc) = tc_gemm_kernels() {
            return tc_gemm_launch(ctx, tc, w, x, y, nt, n_in, n_out, rb);
        }
    }
    // 3. 従来の量子 GEMM (型ごとのスロット)。
    let fs = ctx
        .f_gemm_q
        .ok_or_else(|| anyhow!("量子 GEMM カーネルが無い"))?;
    let f = fs[slot.min(fs.len() - 1)];
    const ROWS: usize = 16;
    const NTG_MAX: usize = 8;
    let shmem = (ROWS * blk * 4) as u32;
    let grid = (((n_out + ROWS - 1) / ROWS) as u32, 1, 1);
    let step = ROWS * NTG_MAX;
    let mut t0 = 0usize;
    while t0 < nt {
        let m = (nt - t0).min(step);
        let (mut pw, mut px, mut py) = (
            w,
            x + (t0 * n_in * 4) as CUdeviceptr,
            y + (t0 * n_out * 4) as CUdeviceptr,
        );
        let (mut m_c, mut ni_c, mut no_c, mut rb_c) =
            (m as c_int, n_in as c_int, n_out as c_int, rb as c_int);
        let mut args = [
            &mut pw as *mut _ as *mut c_void,
            &mut px as *mut _ as *mut c_void,
            &mut py as *mut _ as *mut c_void,
            &mut m_c as *mut _ as *mut c_void,
            &mut ni_c as *mut _ as *mut c_void,
            &mut no_c as *mut _ as *mut c_void,
            &mut rb_c as *mut _ as *mut c_void,
        ];
        unsafe {
            ctx.launch_async(f, grid, (ROWS as u32, 16, 1), shmem, &mut args)?;
        }
        t0 += m;
    }
    Ok(())
}


/// s169 (P10): SM 数 (0 = 不明)。`DENNO_GEMV_SHAPE` の形状選択だけが読む。
pub fn gemv_sm_count() -> usize {
    GEMV_SM_COUNT.load(std::sync::atomic::Ordering::Relaxed)
}


/// s169 (P10): GEMV の列ブロック幅 C の選び方 (`DENNO_GEMV_SHAPE`)。
///   0 (既定) = 従来どおり `q4_gemv_c()` の固定 C
///   1 = n が小さい投影 (block 数が SM 数に対して足りない) だけ C=2
///   2 = 常に C=2
///   3 = 小さい投影は C=2 / 大きい投影は C=8 / 中間は固定 C
/// どの値でも**列 (出力行) ごとの k 方向の縮約順は不変** — C は 1 block が
/// 何列をまとめて受けるかだけを変える (既存テスト
/// `cuda_q4_gemv_cols_matches_single_row` が c2/c4/c8 の bit 一致を検証している)。
pub(crate) fn gemv_shape_mode() -> u32 {
    static MODE: OnceLock<u32> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("DENNO_GEMV_SHAPE").ok().and_then(|v| v.parse::<u32>().ok()) {
            Some(m @ (1 | 2 | 3)) => m,
            _ => 0,
        }
    })
}


/// s169 (P10): 純関数 — n 行の Q4 GEMV を blocks = ceil(n / (8*C)) で割るときの
/// C を選ぶ。`sm` は SM 数 (0 = 不明 → 常に `base`)。
///
/// 根拠: `dno_gemv_q4_0c{C}` は blockDim = 256 固定 (8 warp が 8*C 列を受ける)
/// なので、並列度は block 数 = ceil(n / (8*C)) だけで決まる。RTX 5090
/// (170 SM) では n = 5120 の投影が 160 block = 0.94 block/SM になり、1 SM
/// あたり 8 warp (= 64 スロットの 12.5%) しか埋まらない。C=2 にすると同じ
/// 重みバイト数を 2 倍の block で読む (総 DRAM バイト数は不変・列ごとの縮約順も
/// 不変) ので、この帯だけ並列度を上げる。
pub(crate) fn gemv_c_for_n(n: usize, sm: usize, mode: u32, base: usize) -> usize {
    if sm == 0 || n == 0 {
        return base;
    }
    // C=4 で 2 block/SM を割れない大きさ = 並列度が足りない帯
    let small = 64 * sm;
    // C=4 で 8 block/SM を超える大きさ = 十分に詰まっている帯
    let large = 128 * sm;
    match mode {
        1 => {
            if n < small {
                2
            } else {
                base
            }
        }
        2 => 2,
        3 => {
            if n < small {
                2
            } else if n >= large {
                8
            } else {
                base
            }
        }
        _ => base,
    }
}


/// B1: GEMM 経路 (dno_gemm_i8) が発動する m か。maybe_gpu_ffn / maybe_gpu_oproj /
/// chain_qkv_rope_prefill の内部ゲートと同一条件 (チェーン途中で失敗させないため
/// 呼び出し側が事前確認する)。
pub fn gemm_path_active(m: usize) -> bool {
    m > max_m() && m >= gemm_min_m() && gemm_enabled()
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
    let base_c = *C.get_or_init(q4_gemv_c);
    // s169 (P10): 列ブロック幅 C は**投影ごと**に選ぶ (`DENNO_GEMV_SHAPE`)。
    // 既定 0 は従来どおり全投影で同じ C = 挙動不変。C は 1 block が受ける列数
    // だけを変えるので、列ごとの k 方向の縮約順 (lane が受けるブロック昇順 ->
    // shfl 木) は C に依存しない = どの C でもビット一致 (c2/c4/c8 の一致は
    // 既存テスト `cuda_q4_gemv_cols_matches_single_row` が見ている)。
    let shape = gemv_shape_mode();
    let sm = gemv_sm_count();
    let fc_all = ctx.f_mv_q4_c?;
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
            // s169 (P10): 投影ごとに C を選ぶ (既定 0 は base_c = 従来と同一)。
            let c = gemv_c_for_n(p.n, sm, shape, base_c);
            let f_c = fc_all[q4_gemv_c_index(c)];
            let grid = (p.n.div_ceil(8 * c) as u32, m as u32, 1);
            ok(unsafe { ctx.launch_async(f_c, grid, (256, 1, 1), 0, &mut args) })?;
            dbg_gemv_shape(p.n, c);
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


/// s70: prefill 用の grouped int8 GEMM。`chain_linear_grouped` は decode 用に
/// `m > max_m()` で諦めて GEMV カーネルを起動するが、prefill は m が大きく
/// 128x128 タイルの GEMM を使う必要がある。数値は同じカーネル・同じ整数累積
/// なので CPU 経路とビット一致。
pub(crate) fn chain_gemm_grouped_pf(
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
    chain_mlp_prefill_q4_impl(
        PfGu::Split(gate, up),
        down,
        PfSrc::Host(x),
        PfSink::Host(out),
        m,
        li,
    )
}


/// s173 (P14): [`chain_mlp_prefill_q4`] の**連結重み**版 (ホスト入出力)。
///
/// `gate_up` は `Layer::gate_up_w` = [gate 全行 | up 全行] の行連結 1 本。
/// 1 タイルあたりの launch は「gate|up の 1 回の GEMM + swiglu + down」の
/// 3 発になる (従来は gate / up が別で 4 発)。
///
/// 前提 (すべて Q4・形の整合・`gate_up` が gate/up の行範囲として登録済み) は
/// [`pf_ffn_fused_ready`] で確認する。満たさないときは `None` を返す =
/// 呼び出し側は 2 回呼び出しの [`chain_mlp_prefill_q4`] へ落ちる (fail closed)。
pub fn chain_mlp_prefill_q4_fused(
    gate_up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    x: &[f32],
    m: usize,
    li: usize,
    out: &mut [f32],
) -> Option<()> {
    let e = down_dims_out(down)?;
    if x.len() < m.checked_mul(e)? || out.len() < m.checked_mul(e)? {
        return None;
    }
    chain_mlp_prefill_q4_impl(
        PfGu::Fused(gate_up),
        down,
        PfSrc::Host(x),
        PfSink::Host(out),
        m,
        li,
    )
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
    chain_mlp_prefill_q4_impl(
        PfGu::Split(gate, up),
        down,
        PfSrc::Slot,
        PfSink::Slot,
        m,
        li,
    )
}


/// s173 (P14): [`chain_mlp_prefill_q4_dev`] の**連結重み**版。
///
/// 入力は `CH_PFH`・出力は `CH_DOWN` のまま (転送 0 回) で、違うのは
/// 「gate と up を 1 回の GEMM で `CH_GA` の行内連結 [gate|up] へ書く」点だけ。
/// 続く swiglu は行内レイアウト用の `dno_swiglu_int` になる (演算列は
/// `dno_swiglu` と同一 = CPU の `silu(g)*u` とビット一致)。
///
/// ビット一致の根拠: `dno_gemm_q4_0` の出力 `out[i*n + j]` は k 方向の
/// 累積だけで決まり、n は行ストライドと境界判定にしか使われない
/// (`j0 = blockIdx.y * 128`)。したがって n=2ff の 1 回呼び出しは、
/// n=ff の 2 回呼び出しと**同じ値**を、行内 [gate|up] の並びで書く。
pub fn chain_mlp_prefill_q4_dev_fused(
    gate_up: &crate::quant::Linear,
    down: &crate::quant::Linear,
    m: usize,
    li: usize,
) -> Option<()> {
    chain_mlp_prefill_q4_impl(
        PfGu::Fused(gate_up),
        down,
        PfSrc::Slot,
        PfSink::Slot,
        m,
        li,
    )
}


/// [`chain_mlp_prefill_q4`] の本体 (gate/up の与え方と入出力の出所を差し替える)。
fn chain_mlp_prefill_q4_impl(
    gu_src: PfGu<'_>,
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
    let ctx = chain_ctx()?;
    let du = match down {
        Linear::Q4(c) => c,
        _ => return None,
    };
    // s173 (P14): gate/up の重みとレイアウト。`Fused` では 1 本の連結重みで
    // gate|up を 1 回の GEMM にまとめる (ff は down.k、e は down.n から取る)。
    let (e, ff, gw1, gw2) = match gu_src {
        PfGu::Split(gate, up) => {
            let (gu, uu) = match (gate, up) {
                (Linear::Q4(a), Linear::Q4(b)) => (a, b),
                _ => return None,
            };
            let e = gu.k;
            let ff = gu.n;
            if uu.k != e || uu.n != ff || du.k != ff || du.n != e {
                return None;
            }
            let gw1 = weights_for_q4(gu.uid, &gu.blocks, gu.k, gu.n).ok()?;
            let gw2 = weights_for_q4(uu.uid, &uu.blocks, uu.k, uu.n).ok()?;
            (e, ff, gw1, Some(gw2))
        }
        PfGu::Fused(gate_up) => {
            let c = match gate_up {
                Linear::Q4(x) => x,
                _ => return None,
            };
            let e = du.n;
            let ff = du.k;
            if c.k != e || ff == 0 || c.n != 2 * ff {
                return None;
            }
            let gw1 = weights_for_q4(c.uid, &c.blocks, c.k, c.n).ok()?;
            (e, ff, gw1, None)
        }
    };
    let gw3 = weights_for_q4(du.uid, &du.blocks, du.k, du.n).ok()?;
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
            // s173 (P14): 連結重みがあるときは gate|up を**1 回の GEMM**で
            // CH_GA へ行内 [gate|up] で書く (launch 4 -> 3)。続く swiglu は
            // 行内レイアウト用の `dno_swiglu_int` (演算列は `dno_swiglu` と同一)。
            pf_phase(PF_PH_FFN_GATE);
            crate::prof::time_layer(crate::prof::GATE_UP, li, || -> Option<()> {
                match gw2 {
                    Some(gw_up) => {
                        chain_gemm_q4_pf(gw1, pin, pga, mr, e, ff)?;
                        chain_gemm_q4_pf(gw_up, pin, pga + up_off, mr, e, ff)
                    }
                    None => chain_gemm_q4_pf(gw1, pin, pga, mr, e, 2 * ff),
                }
            })?;
            pf_phase(PF_PH_SWIGLU);
            crate::prof::time_layer(crate::prof::SWIGLU, li, || -> Option<()> {
                match gw2 {
                    Some(_) => chain_swiglu_split(pga, pga + up_off, pact, mr * ff),
                    None => chain_swiglu_int_at(pga, pact, ff, mr * ff),
                }
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
// task 133: qwen4exp (Flash-Next) の full attention (decode) を GPU で回す
//
// 設計:
// - ホストの `AttnCache` が正。GPU 側は層ごとの K/V 行の複製 (ミラー) を持つ。
// - CPU prefill の間はミラーを触らない。**最初の decode トークン**で層ごとに
//   履歴 `[0..len)` を 1 回だけ H2D する (遅延シード)。以降は 1 トークン 1 行。
// - ミラーは `AttnCache::id()` で引き、層ごとの内容世代 `gen` が変わった層だけ
//   全行 H2D し直す。`AttnCache` は `clear` / `clear_all` / `truncate_layer` で
//   gen を進めるので、prefix reuse / model reuse (`reset`) / spec rollback の
//   どれでも古い行を attention に使わない。
// - cap は 1,024 行単位の倍々。成長時は旧バッファから D2D で行を温存する
//   (再シードしない)。65,536 行 x 512 x 4 B x 2 (K/V) x 12 層 = 3,072 MiB。
// - カーネルは既存の `dno_attention` (kernels_elem.cu)。.cu / .ptx は触らない。
// - GPU が無い・カーネルの制約に合わない・確保/転送/起動に失敗したときは
//   `None` を返し、呼び出し側は CPU 経路へ落ちる (fail closed)。ここで
//   `GPU_BROKEN` は立てない — attention だけの問題で他の GPU 経路
//   (GEMV / mix / router / expert) を巻き込まない。
//
// 実測 (lab t133b, swift15-abl IQ3_S, PROFILE=1, off -> on):
//   decode ms/tok: a/b 81 -> 80、c2000_cold 122 -> 96、c2000_turn2 124 -> 95、
//   c8000_cold 210 -> 121、c8000_turn2 208 -> 121 (prefill は ±1.8% 以内)。
//   6 プロンプトの出力 md5 は off と同一。1 層 1 呼び出しの内訳は
//   28 tok で 84 us、3,954 で 637 us、14,018 で 2.09 ms
//   (14k はカーネル待ち 2.03 ms が支配的。1 head = 1 block で占有率が低い)。
// ---------------------------------------------------------------------------

/// `DENNO_Q4_GPU_ATTN` の既定。lab t133b (最終コード) で a/b 81 -> 80、
/// c2000 122/124 -> 96/95、c8000 210/208 -> 121/121、md5 は 6 本とも同一。
/// 短いプロンプトは MIN で CPU のままなので既定 on。
pub(crate) const Q4_GPU_ATTN_DEFAULT: bool = true;

/// `DENNO_Q4_GPU_ATTN_MIN` の既定 (この KV 本数未満は CPU 経路のまま)。
/// 実測の交叉は約 23 token (推定: 1 呼び出しの固定費 79 us x 12 → 0.95 ms/token と
/// 短 context の CPU scan 1.04 ms/token が 28 token 付近で並ぶ)。
/// 短いプロンプト (lab の a/b = 28/33) を確実に CPU のままにするため 64 にする。
pub(crate) const Q4_GPU_ATTN_MIN_DEFAULT: usize = 64;

/// ミラー上限の既定 (MiB)。65,536 行をフルに使って 3,072 MiB。
pub(crate) const Q4_GPU_ATTN_MIRROR_MB_DEFAULT: usize = 3584;


/// task 134: `DENNO_Q4_GPU_ATTN_PF` の既定 (prefill の full attention を GPU へ)。
///
/// lab t134a (SCENARIO=depth) で c2000 cold prefill 134.83 -> 110.59 s (-18.0%)、
/// c2000 turn2 +0.7%、短いプロンプト 0.0%、decode -1.2%、md5 は 4 本とも同一。
/// lab t134b (SCENARIO=depth8k) で c2000 cold -18.1%、c8000 cold
/// 758.94 -> 379.20 s (-50.0%)、c8000 turn2 2.53 -> 2.53 s (+0.0%)、decode 不変、
/// errors 0、text sane。md5 は c8000 の 2 本だけ DIFF (GPU attention の丸め順で
/// greedy が動いた。c2000 と短文は同一)。目標 (c2000 15% / c8000 35%) を
/// 満たしたので既定 on。
pub(crate) const Q4_GPU_ATTN_PF_DEFAULT: bool = true;


/// `DENNO_Q4_GPU_ATTN_PF_MIN` の既定 = 1 チャンクの行数の下限。
///
/// task 135: カーネルの構造上の下限を `gemm_min_m()` (256) から attention 専用の
/// [`ATTN_Q4_PF_MIN_ROWS`] へ移した。qwen35 は従来の
/// `attn_prefill_gpu_supported` (n >= gemm_min_m()) のまま。
/// 実測は [`ATTN_Q4_PF_MIN_ROWS`] の doc と REPORT_P135.md。
pub(crate) const Q4_GPU_ATTN_PF_MIN_DEFAULT: usize = ATTN_Q4_PF_MIN_ROWS;

/// `DENNO_Q4_GPU_ATTN_PF_MIN_KV` の既定 = 文脈長 (t_total) の下限。
/// 短いプロンプト (数百 token) は CPU のままにする。
pub(crate) const Q4_GPU_ATTN_PF_MIN_KV_DEFAULT: usize = 512;


#[cfg(test)]
pub(crate) static Q4_ATTN_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
pub(crate) static Q4_ATTN_MIN_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);


/// q4 full attention の GPU 経路が有効か (`DENNO_Q4_GPU_ATTN`)。
///
/// `DENNO_GPU_ATTN=0` は [`gpu_attention_enabled`] 側で見る (両方 ON のときだけ
/// 発動する)。
pub(crate) fn q4_gpu_attn_enabled() -> bool {
    #[cfg(test)]
    {
        let f = Q4_ATTN_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_q4_gpu_attn(std::env::var("DENNO_Q4_GPU_ATTN").ok().as_deref()))
}


/// この KV 本数未満では CPU 経路のまま (`DENNO_Q4_GPU_ATTN_MIN`)。
pub(crate) fn q4_gpu_attn_min() -> usize {
    #[cfg(test)]
    {
        let f = Q4_ATTN_MIN_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f;
        }
    }
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        parse_q4_gpu_attn_min(std::env::var("DENNO_Q4_GPU_ATTN_MIN").ok().as_deref())
    })
}


#[cfg(test)]
pub(crate) static Q4_ATTN_PF_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
pub(crate) static Q4_ATTN_PF_MIN_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

#[cfg(test)]
pub(crate) static Q4_ATTN_PF_MIN_KV_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);


/// prefill の full attention GPU 経路が有効か (`DENNO_Q4_GPU_ATTN_PF`)。
pub(crate) fn q4_gpu_attn_pf_enabled() -> bool {
    #[cfg(test)]
    {
        let f = Q4_ATTN_PF_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return f == 1;
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_q4_gpu_attn_pf(std::env::var("DENNO_Q4_GPU_ATTN_PF").ok().as_deref())
    })
}


/// 1 チャンクの行数の下限 (`DENNO_Q4_GPU_ATTN_PF_MIN`)。
pub(crate) fn q4_gpu_attn_pf_min_rows() -> usize {
    #[cfg(test)]
    {
        let f = Q4_ATTN_PF_MIN_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f;
        }
    }
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        parse_usize_default(
            std::env::var("DENNO_Q4_GPU_ATTN_PF_MIN").ok().as_deref(),
            Q4_GPU_ATTN_PF_MIN_DEFAULT,
        )
    })
}


/// 文脈長 (チャンク終端の位置数) の下限 (`DENNO_Q4_GPU_ATTN_PF_MIN_KV`)。
pub(crate) fn q4_gpu_attn_pf_min_kv() -> usize {
    #[cfg(test)]
    {
        let f = Q4_ATTN_PF_MIN_KV_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f;
        }
    }
    static MIN: OnceLock<usize> = OnceLock::new();
    *MIN.get_or_init(|| {
        parse_usize_default(
            std::env::var("DENNO_Q4_GPU_ATTN_PF_MIN_KV").ok().as_deref(),
            Q4_GPU_ATTN_PF_MIN_KV_DEFAULT,
        )
    })
}


/// `DENNO_Q4_GPU_ATTN_KERNEL` の既定。
///
/// lab t139a (depth, ptx=elem) の実測で `split` (`dno_attention_decode`) が
/// 3,954 文脈の 1 呼び出しを 571 us -> 7.6 us にし、c2000 turn2 decode を
/// 93 -> 86 ms/tok にした (`[q4attn] split calls=1116 fallback=0`)。
/// lab t139b (depth8k, PTX は commit 済みのもの) でも c8000 turn2 decode が
/// 119 -> 93 ms/tok (-21.8%)、c2000 turn2 が 96 -> 86、短いプロンプトは
/// 81 -> 81 で md5 は 6 本とも SAME。既定にする。
/// PTX にカーネルが無いビルドでは fail closed で `dno_attention` に落ちる。
/// `pf` (prefill カーネルを n=1 で使う) は lab t135a で 3.9 倍遅かったので
/// 既定にはしない (`DENNO_Q4_GPU_ATTN_KERNEL=pf` で選べる)。
pub(crate) const Q4_GPU_ATTN_KERNEL_DEFAULT: Q4AttnKernel = Q4AttnKernel::Split;


/// t の分割数 (`DENNO_Q4_GPU_ATTN_SPLIT`, 1..=64)。
///
/// 1 呼び出しの block 数は heads x 分割数。qwen4exp は heads=24 なので 8 分割で
/// 192 block (SM 170 前後) になり、t=14k でも全 SM が働く。
pub(crate) const Q4_GPU_ATTN_SPLIT_DEFAULT: usize = 8;


/// `dno_attention_decode` の動的 smem (float 数)。q + 生スコア + 確率 + corr。
pub(crate) fn q4_dec_sh_f(dh: usize) -> usize {
    dh + 2 * ATTN_DEC_BK + 1
}


/// `t_total` と分割数から (実際の分割数, 1 分割あたりの key 数) を返す (純関数)。
///
/// 空の分割を作らない (空の分割は m = -inf になるだけで combine は 0 寄与だが、
/// 無駄な block を増やさない)。カーネル側の `chunk = ceil(t_total / n_split)` と
/// 同じ式で、ホストとカーネルが同じ分割を見る。
pub(crate) fn q4_dec_split(t_total: usize, n_split: usize) -> (usize, usize) {
    let n = n_split.max(1);
    if t_total == 0 {
        return (1, 1);
    }
    let chunk = t_total.div_ceil(n).max(1);
    (t_total.div_ceil(chunk).max(1), chunk)
}


/// 文脈長から 1 分割の key 数を決める (`auto` のとき。純関数)。
pub(crate) fn q4_fdec_chunk_auto(t_total: usize) -> usize {
    t_total
        .div_ceil(ATTN_FLASHDEC_AUTO_SPLITS)
        .clamp(ATTN_FLASHDEC_CHUNK_DEFAULT, ATTN_FLASHDEC_AUTO_CHUNK_MAX)
}


/// 文脈長から分割数を決める (純関数)。カーネル側の
/// `chunk = ceil(t_total / n_split)` と同じ式で、ホストとカーネルが同じ分割を見る。
///
/// block 数は `n_split x n_kv`。n_kv = 4 のモデルなら 256 key 分割で
/// 14,139 -> 56 分割 = 224 block (SM 170) になり、全 SM が働く。
pub(crate) fn q4_fdec_split(t_total: usize, chunk_keys: usize) -> (usize, usize) {
    if t_total == 0 {
        return (1, 1);
    }
    let keys = chunk_keys.max(ATTN_FLASHDEC_CHUNK_MIN);
    let n = t_total.div_ceil(keys).clamp(1, ATTN_FDEC_SPLIT_MAX);
    let chunk = t_total.div_ceil(n).max(1);
    (t_total.div_ceil(chunk).max(1), chunk)
}


/// 分割数を 1 回だけログる (task 160 の "log it")。len は毎トークン変わるので
/// 分割数が変わったときだけ出す (14k でも数十トークンに 1 行)。
pub(crate) fn fdec_log(len: usize, n_split: usize, chunk: usize, kv_dim: usize, dh: usize, group: usize) {
    static LAST: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(usize::MAX);
    if LAST.load(std::sync::atomic::Ordering::Relaxed) == n_split {
        return;
    }
    if LAST.swap(n_split, std::sync::atomic::Ordering::Relaxed) != n_split {
        eprintln!(
            "[attn-fdec] len={len} n_split={n_split} chunk={chunk} kv_dim={kv_dim} dh={dh} \
             group={group} kv_bytes={:.1}MB",
            (len * kv_dim * 4 * 2) as f64 / (1024.0 * 1024.0)
        );
    }
}


/// flash decoding の PTX (`kernels_attn_dec.ptx`)。`DENNO_ATTN_FLASHDEC_PTX` で
/// 差し替え可 (lab は再生成した PTX をツリーに置いてから測る)。
fn fdec_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_ATTN_FLASHDEC_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_attn_dec.ptx"),
    }
}


/// flash decoding のカーネル 2 本。
///
/// `include_bytes!` せず実行時に 1 回だけ読む (このコンテナには nvcc が無く、
/// PTX を先に置かないとビルドが通らないため。exp_iq / gemm_grouped と同じ方針)。
/// 読めない・entry が無い場合は None を焼き付け、呼び出し側は従来カーネルへ落ちる。
pub(crate) struct FdecKernels {
    /// モジュールはプロセス終了まで保持する (unload しない)。
    _module: CUmodule,
    pub(crate) f_dec: CUfunction,
    pub(crate) f_cb: CUfunction,
}


// 生ハンドル (module / function)。CudaCtx と同じ規約。
unsafe impl Send for FdecKernels {}

unsafe impl Sync for FdecKernels {}


pub(crate) fn fdec_kernels() -> Option<&'static FdecKernels> {
    static K: OnceLock<Option<FdecKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = fdec_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[attn-fdec] PTX が読めない ({}) : {e} → 従来カーネルへ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[attn-fdec] cuModuleLoadData 失敗 → 従来カーネルへ");
                return None;
            }
        }
        let mut f_dec: CUfunction = std::ptr::null_mut();
        let mut f_cb: CUfunction = std::ptr::null_mut();
        unsafe {
            if (lib.module_get_function)(
                &mut f_dec,
                module,
                b"dno_attn_flashdec\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
                || (lib.module_get_function)(
                    &mut f_cb,
                    module,
                    b"dno_attn_flashdec_combine\0".as_ptr() as *const c_char,
                ) != CUDA_SUCCESS
            {
                eprintln!("[attn-fdec] PTX にカーネルが無い (再生成が必要) → 従来カーネルへ");
                return None;
            }
        }
        Some(FdecKernels {
            _module: module,
            f_dec,
            f_cb,
        })
    })
    .as_ref()
}


// ---- task 448: flash decoding 第 2 版 (K/V を shared のタイルへ協調ロード) ----

/// タイル版の PTX (`kernels_attn_dec2.ptx`)。
/// `DENNO_ATTN_FLASHDEC2_PTX` で差し替え可 (lab は再生成した PTX をツリーに置く)。
fn fdt_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_ATTN_FLASHDEC2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_attn_dec2.ptx"),
    }
}


/// タイル版カーネルの引数の個数 (`.cu` と一致させる。kernel_invariants が見る)。
pub(crate) const FDT_NARGS: usize = 11;


/// タイル版の動的 shared の上限 (これを超える形では v1 へ落とす)。
/// 48 KiB は cuLaunchKernel が opt-in 無しで受け付ける上限。
pub(crate) const FDT_SMEM_MAX: usize = 48 * 1024;


/// タイル版の 1 block あたりの動的 shared (バイト)。`.cu` の
/// `2 * tile * dh * sizeof(float)` と一致させること。
pub(crate) fn fdt_smem_bytes(tile: usize, dh: usize) -> usize {
    2 * tile * dh * 4
}


/// タイル版の形の門番 (純関数)。
///
/// `.cu` の前提は「dh が 4 の倍数」「dh <= DNO_FDT_MAXB * 128」「smem <= 48 KiB」。
/// 1 つでも外れたら v1 へ落とす (v1 はこの形をすべて満たしている)。
pub(crate) fn fdt_shape_ok(dh: usize, tile: usize) -> bool {
    dh >= 4
        && dh % 4 == 0
        && dh <= ATTN_FDEC_MAXB * 128
        && (1..=64).contains(&tile)
        && fdt_smem_bytes(tile, dh) <= FDT_SMEM_MAX
}


/// カーネル選択からタイル版の on/off を決める (純関数)。
///
/// `tile` / `fdec2` で on、`fdec1` で off、それ以外 (`split` / `decode` / `pf` /
/// 未設定) は [`ATTN_FLASHDEC_TILE_DEFAULT`] に従う。
pub(crate) fn fdec_tile_from_kernel(k: Q4AttnKernel) -> bool {
    match k {
        Q4AttnKernel::Tile => true,
        Q4AttnKernel::Fdec1 => false,
        _ => ATTN_FLASHDEC_TILE_DEFAULT,
    }
}


/// task 448: タイル版の on/off の **新しい env** (`DENNO_ATTN_FLASHDEC2`)。
///
/// 純関数。未設定・空は `None` (カーネル選択に委ねる)。lab の COND は新しい env 名
/// でも届くが、既存名 (`DENNO_Q4_GPU_ATTN_KERNEL`) の上書きは届かない回があった
/// (t448a では `tile` が効かず v1 が走った) ので、A/B はこちらで取る。
pub(crate) fn fdec_tile_from_env(v: Option<&str>) -> Option<bool> {
    let t = v?.trim().to_ascii_lowercase();
    if t.is_empty() {
        return None;
    }
    match t.as_str() {
        "1" | "on" | "tile" | "true" | "yes" => Some(true),
        _ => Some(false),
    }
}


/// 門の判定を 1 回だけログる (発火しなかった回の原因を lab の log で見るため)。
pub(crate) fn fdt_gate_log(tile_on: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[attn-fdec2] gate tile_on={tile_on} tile={} min_ctx={} \
             (DENNO_ATTN_FLASHDEC_TILE / DENNO_Q4_GPU_ATTN_KERNEL)",
            attn_fdec_tile(),
            attn_flashdec_min_ctx()
        );
    });
}


pub(crate) fn fdt_kernels() -> Option<&'static FdtKernels> {
    static K: OnceLock<Option<FdtKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = fdt_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[attn-fdec2] PTX が読めない ({}) : {e} → v1 の flashdec へ",
                    path.display()
                );
                return None;
            }
        };
        // ABI: 引数の個数が違う PTX を起動すると、足りない引数は不定のまま読まれる。
        // モジュールを載せる前に数えられないと分からないので、ここで弾く。
        let text = String::from_utf8_lossy(&image).into_owned();
        if ptx_entry_params(&text, "dno_attn_flashdec_t") != Some(FDT_NARGS) {
            eprintln!(
                "[attn-fdec2] PTX の dno_attn_flashdec_t が {FDT_NARGS} 引数でない \
                 (再生成が必要) → v1 の flashdec へ"
            );
            return None;
        }
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[attn-fdec2] cuModuleLoadData 失敗 → v1 の flashdec へ");
                return None;
            }
        }
        let mut f_t: CUfunction = std::ptr::null_mut();
        unsafe {
            if (lib.module_get_function)(
                &mut f_t,
                module,
                b"dno_attn_flashdec_t\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[attn-fdec2] PTX に dno_attn_flashdec_t が無い → v1 の flashdec へ");
                return None;
            }
        }
        let k = FdtKernels { _module: module, f_t };
        fdt_loaded_log();
        Some(k)
    })
    .as_ref()
}


/// モジュールが載ったことを 1 回だけログる (発火の証拠。tile 行数と env も出す)。
fn fdt_loaded_log() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[attn-fdec2] PTX ok: dno_attn_flashdec_t nargs={} tile={} (DENNO_ATTN_FLASHDEC2)",
            FDT_NARGS,
            attn_fdec_tile()
        );
    });
}


/// タイル版の分割・tile を 1 回だけログる (発火の証拠。tile か n_split が
/// 変わったときだけ出す)。
pub(crate) fn fdt_log(len: usize, n_split: usize, chunk: usize, dh: usize, tile: usize) {
    static LAST: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(usize::MAX);
    let key = n_split.wrapping_mul(1024).wrapping_add(tile);
    if LAST.load(std::sync::atomic::Ordering::Relaxed) == key {
        return;
    }
    if LAST.swap(key, std::sync::atomic::Ordering::Relaxed) != key {
        eprintln!(
            "[attn-fdec2] len={len} n_split={n_split} chunk={chunk} dh={dh} tile={tile} \
             smem={}KiB",
            fdt_smem_bytes(tile, dh) / 1024
        );
    }
}


// ---- task 520: flash decoding 第 3 版 (2 key 同時、`kernels_attn_dec5.cu`) -----
//
// 動機 (t520a の `[dec-kprof-avg]`): `dno_attn_flashdec_t` は 14k で 66.4 us/層、
// 2k で 29.3 us/層。key 数が 7 倍でも時間は 2.3 倍 = **1 key の単価がほぼ一定**
// (458 -> 519 ns) で、帯域ではなく命令発行と依存鎖で決まっている。第 3 版は
// 「2 key の内積と shfl 縮約を交互に出す」+「max が動かない key では acc の
// 乗算を省く」。どちらも算術の列を変えないので **値は第 2 版とビット一致**。

/// 第 3 版の引数の個数 (`.cu` と一致。第 2 版の 11 本 + grid 入れ替えの 1 本)。
pub(crate) const FDT2_NARGS: usize = 12;


/// 第 3 版の PTX (`kernels_attn_dec5.ptx`)。`DENNO_ATTN_FLASHDEC5_PTX` で差し替え可。
fn fdt2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_ATTN_FLASHDEC5_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_attn_dec5.ptx"),
    }
}


pub(crate) fn fdt2_kernels() -> Option<&'static Fdt2Kernels> {
    static K: OnceLock<Option<Fdt2Kernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = fdt2_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[attn-fdec5] PTX が読めない ({}) : {e} → 第 2 版 (タイル) へ",
                    path.display()
                );
                return None;
            }
        };
        let text = String::from_utf8_lossy(&image).into_owned();
        if ptx_entry_params(&text, "dno_attn_flashdec_t2") != Some(FDT2_NARGS) {
            eprintln!(
                "[attn-fdec5] PTX の dno_attn_flashdec_t2 が {FDT2_NARGS} 引数でない \
                 (再生成が必要) → 第 2 版 (タイル) へ"
            );
            return None;
        }
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[attn-fdec5] cuModuleLoadData 失敗 → 第 2 版 (タイル) へ");
                return None;
            }
        }
        let mut f_t2: CUfunction = std::ptr::null_mut();
        unsafe {
            if (lib.module_get_function)(
                &mut f_t2,
                module,
                b"dno_attn_flashdec_t2\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[attn-fdec5] PTX に dno_attn_flashdec_t2 が無い → 第 2 版 (タイル) へ");
                return None;
            }
        }
        let k = Fdt2Kernels { _module: module, f_t2 };
        eprintln!("[attn-fdec5] PTX ok: dno_attn_flashdec_t2 nargs={FDT2_NARGS}");
        Some(k)
    })
    .as_ref()
}


/// 第 3 版の分割・tile を 1 回だけログる (発火の証拠)。
pub(crate) fn fdt2_log(len: usize, n_split: usize, tile: usize) {
    static LAST: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(usize::MAX);
    let key = n_split.wrapping_mul(1024).wrapping_add(tile);
    if LAST.load(std::sync::atomic::Ordering::Relaxed) == key {
        return;
    }
    if LAST.swap(key, std::sync::atomic::Ordering::Relaxed) != key {
        eprintln!("[attn-fdec5] len={len} n_split={n_split} tile={tile} (2 key 同時)");
    }
}


/// 第 3 版の門を 1 回だけログる (届いたかの証拠)。
pub(crate) fn fdt2_gate_log() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[attn-fdec5] gate ilp_on={} tile={} (DENNO_ATTN_FLASHDEC_ILP)",
            attn_fdec_ilp_on(),
            attn_fdec_tile()
        );
    });
}


/// 融合版の PTX (`kernels_attn_dec4.ptx`)。`DENNO_ATTN_FLASHDEC4_PTX` で差し替え可。
fn fcbg_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_ATTN_FLASHDEC4_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_attn_dec4.ptx"),
    }
}


pub(crate) fn fcbg_kernels() -> Option<&'static FcbgKernels> {
    static K: OnceLock<Option<FcbgKernels>> = OnceLock::new();
    K.get_or_init(|| {
        let path = fcbg_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[attn-fdec4] PTX が読めない ({}) : {e} → combine + gate の 2 起動へ",
                    path.display()
                );
                return None;
            }
        };
        let text = String::from_utf8_lossy(&image).into_owned();
        if ptx_entry_params(&text, "dno_attn_flashdec_cbg") != Some(FCBG_NARGS) {
            eprintln!(
                "[attn-fdec4] PTX の dno_attn_flashdec_cbg が {FCBG_NARGS} 引数でない \
                 (再生成が必要) → 2 起動へ"
            );
            return None;
        }
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[attn-fdec4] cuModuleLoadData 失敗 → 2 起動へ");
                return None;
            }
        }
        let mut f_cbg: CUfunction = std::ptr::null_mut();
        unsafe {
            if (lib.module_get_function)(
                &mut f_cbg,
                module,
                b"dno_attn_flashdec_cbg\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[attn-fdec4] PTX に dno_attn_flashdec_cbg が無い → 2 起動へ");
                return None;
            }
        }
        let k = FcbgKernels { _module: module, f_cbg };
        eprintln!("[attn-fdec4] PTX ok: dno_attn_flashdec_cbg nargs={FCBG_NARGS}");
        Some(k)
    })
    .as_ref()
}


/// 融合の門を 1 回だけログる (既定 off の確認用。t456 と同じ流儀)。
pub(crate) fn fcbg_gate_log(gate_ok: bool) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[attn-fdec4] gate cbg_on={} gate_ok={gate_ok} (DENNO_ATTN_FLASHDEC_CBG)",
            attn_fdec_cbg_on()
        );
    });
}


#[cfg(test)]
pub(crate) static Q4_ATTN_KERNEL_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

#[cfg(test)]
pub(crate) static Q4_ATTN_SPLIT_FORCE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);


/// decode 経路のカーネル (`DENNO_Q4_GPU_ATTN_KERNEL`)。未設定・不正値は既定。
pub(crate) fn q4_gpu_attn_kernel() -> Q4AttnKernel {
    #[cfg(test)]
    {
        let f = Q4_ATTN_KERNEL_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f >= 0 {
            return match f {
                1 => Q4AttnKernel::Prefill,
                2 => Q4AttnKernel::Split,
                3 => Q4AttnKernel::Tile,
                4 => Q4AttnKernel::Fdec1,
                _ => Q4AttnKernel::Decode,
            };
        }
    }
    static K: OnceLock<Q4AttnKernel> = OnceLock::new();
    *K.get_or_init(|| {
        parse_q4_gpu_attn_kernel(std::env::var("DENNO_Q4_GPU_ATTN_KERNEL").ok().as_deref())
    })
}


/// t の分割数 (`DENNO_Q4_GPU_ATTN_SPLIT`)。
pub(crate) fn q4_gpu_attn_split() -> usize {
    #[cfg(test)]
    {
        let f = Q4_ATTN_SPLIT_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != usize::MAX {
            return f.clamp(1, 64);
        }
    }
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        parse_q4_gpu_attn_split(std::env::var("DENNO_Q4_GPU_ATTN_SPLIT").ok().as_deref())
    })
}


/// 必要な行数 `need` に対する新しい cap。倍々 + 1,024 行の下限。
pub(crate) fn q4_mirror_cap(need: usize, old: usize) -> usize {
    if need <= old {
        return old;
    }
    let target = need.max(1024).next_power_of_two();
    target.max(old.saturating_mul(2))
}


/// ミラー 1 本の確保行数 (台帳の cap + 投機読みの余白)。
pub(crate) fn q4_mirror_alloc_rows(cap: usize) -> usize {
    cap.saturating_add(ATTN_MIRROR_GUARD_ROWS)
}


/// ミラーが既に正しく持っている行数。世代が違う / 縮んだら 0 (全行 H2D)。
pub(crate) fn q4_mirror_have(st: &Q4MirrorState, li: usize, len: usize, gen: u64) -> usize {
    if li >= st.len.len() {
        return 0;
    }
    if st.gen[li] == gen && st.len[li] <= len {
        st.len[li]
    } else {
        0
    }
}


/// 同期後にミラーの台帳を更新する。
pub(crate) fn q4_mirror_note(st: &mut Q4MirrorState, li: usize, len: usize, gen: u64) {
    if li < st.len.len() {
        st.len[li] = len;
        st.gen[li] = gen;
    }
}


/// ミラーの使用バイト数 `cap 行 x kv_dim x 4 B x K/V x 使った層数`。
/// 使っていない層 (ssm 層) は確保しないので、その分は数えない。
/// task 174: 実確保は投機読みの余白 (`ATTN_MIRROR_GUARD_ROWS`) を含む。
pub(crate) fn q4_mirror_bytes(cap: usize, kv_dim: usize, used_layers: usize) -> usize {
    q4_mirror_alloc_rows(cap)
        .saturating_mul(kv_dim)
        .saturating_mul(4)
        .saturating_mul(2)
        .saturating_mul(used_layers)
}


pub(crate) static Q4_ATTN_MIRRORS: OnceLock<std::sync::Mutex<Vec<(u64, Q4AttnMirror, std::time::Instant)>>> =
    OnceLock::new();


/// task 210: [`q4_attn_mirror_forget`] が呼ばれた回数 (発火テスト用)。
///
/// ミラーは VRAM なので CPU では効果を測れないが、「cache を捨てたときに
/// 解放の口が呼ばれたか」はこれで見える。
pub(crate) static Q4_ATTN_MIRROR_FORGETS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);


/// flash decoding の実装 (v1 / task 448 のタイル版)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum FdecImpl {
    V1,
    Tile,
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


/// PTX の置き場所 (`DENNO_PF_BA_PTX` があればそれ)。
fn ba_gemm_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_BA_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_gemm_ba225.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
fn ba_gemm_fns(ctx: &CudaCtx) -> Option<&'static BaGemmFns> {
    static K: OnceLock<Option<BaGemmFns>> = OnceLock::new();
    K.get_or_init(|| {
        let path = ba_gemm_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[pf225-ba] PTX が読めない ({}) : {e} → ホストの射影へ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[pf225-ba] cuModuleLoadData 失敗 → ホストの射影へ");
                return None;
            }
            let mut ba_gemm: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut ba_gemm,
                module,
                b"dno_gemm_f32_ba\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[pf225-ba] PTX に dno_gemm_f32_ba が無い → ホストの射影へ");
                return None;
            }
            let _ = ctx;
            log_kernel_attr(lib, ba_gemm, "dno_gemm_f32_ba");
            Some(BaGemmFns { module, ba_gemm })
        }
    })
    .as_ref()
}


/// 専用 GEMM を 1 本撃つ (beta と alpha で 2 回)。`false` = 撃てなかった。
#[allow(clippy::too_many_arguments)]
pub(crate) fn pf_ba_gemm_dev_launch(
    ctx: &CudaCtx,
    w: CUdeviceptr,
    x: CUdeviceptr,
    y: CUdeviceptr,
    nt: usize,
    k: usize,
    n: usize,
) -> Result<bool> {
    if !pf_ba_fast_ok(nt, k, n) {
        return Ok(false);
    }
    let Some(fns) = ba_gemm_fns(ctx) else {
        return Ok(false);
    };
    let (mut pw, mut px, mut py) = (w, x, y);
    let (mut nt_c, mut k_c, mut n_c) = (nt as c_int, k as c_int, n as c_int);
    let mut args = [
        &mut pw as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut py as *mut _ as *mut c_void,
        &mut nt_c as *mut _ as *mut c_void,
        &mut k_c as *mut _ as *mut c_void,
        &mut n_c as *mut _ as *mut c_void,
    ];
    let grid = (
        nt.div_ceil(BA225_TOKS) as u32,
        n.div_ceil(BA225_ROWS) as u32,
        1,
    );
    unsafe { ctx.launch_async(fns.ba_gemm, grid, (256, 1, 1), 0, &mut args)? };
    Ok(true)
}


// ---------------------------------------------------------------------------
// task 225: prefill の router softmax + top-k をデバイスで回す
// (`DENNO_PF_ROUTER_DEV`、task 228 から既定 on)
// ---------------------------------------------------------------------------
//
// lab t222c: `sub router softmax` 239.9 ms (ホスト)。logits は既に GPU の f32
// GEMM で作っているが、全行 (14k で 1,378 MB) をホストへ降ろして 512 列の
// softmax と top-k を回していた。ここでは logits をデバイスに残し、softmax →
// top-k → 再正規化までを `dno_router_softmax_topk` (kernels_router225.cu) で
// 行って、**選んだ k 個の index と重みだけ**をホストへ返す。
//
// 数値: 和の結合順が違うので softmax の分母が 1 ulp 級に動く (丸めだけの差)。
// 選択規則 (値の降順・同値は小さい index・出力は index 昇順) と再正規化の加算順
// (index 昇順) はホストの `router_topk_rows` / `router_renorm` と同一。

pub(crate) struct RouterDevFns {
    #[allow(dead_code)]
    module: CUmodule,
    pub(crate) kernel: CUfunction,
}


unsafe impl Send for RouterDevFns {}

unsafe impl Sync for RouterDevFns {}


fn router_dev_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_PF_ROUTER_DEV_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_router225.ptx"),
    }
}


pub(crate) fn router_dev_fns(ctx: &CudaCtx) -> Option<&'static RouterDevFns> {
    static K: OnceLock<Option<RouterDevFns>> = OnceLock::new();
    K.get_or_init(|| {
        let path = router_dev_ptx_path();
        let mut image = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "[pf225-rt] PTX が読めない ({}) : {e} → ホストの softmax へ",
                    path.display()
                );
                return None;
            }
        };
        image.push(0);
        let lib = lib().ok()?;
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void) != CUDA_SUCCESS {
                eprintln!("[pf225-rt] cuModuleLoadData 失敗 → ホストの softmax へ");
                return None;
            }
            let mut kernel: CUfunction = std::ptr::null_mut();
            if (lib.module_get_function)(
                &mut kernel,
                module,
                b"dno_router_softmax_topk\0".as_ptr() as *const c_char,
            ) != CUDA_SUCCESS
            {
                eprintln!("[pf225-rt] PTX に dno_router_softmax_topk が無い → ホストへ");
                return None;
            }
            let _ = ctx;
            log_kernel_attr(lib, kernel, "dno_router_softmax_topk");
            Some(RouterDevFns { module, kernel })
        }
    })
    .as_ref()
}


#[cfg(test)]
pub(crate) static ROUTER_DEV_FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);


pub(crate) struct RouterDevSc {
    pub(crate) x: Option<DevBuffer>,
    pub(crate) logits: Option<DevBuffer>,
    pub(crate) sels: Option<DevBuffer>,
    pub(crate) wts: Option<DevBuffer>,
    pub(crate) rowsum: Option<DevBuffer>,
}


thread_local! {
    pub(crate) static ROUTER_DEV_SC: std::cell::RefCell<RouterDevSc> = const {
        std::cell::RefCell::new(RouterDevSc {
            x: None,
            logits: None,
            sels: None,
            wts: None,
            rowsum: None,
        })
    };
}


/// Q4 線形の (k, n)。Q4 以外は None。
pub(crate) fn q4_dims(w: &crate::quant::Linear) -> Option<(usize, usize)> {
    match w {
        crate::quant::Linear::Q4(q) => Some((q.k, q.n)),
        _ => None,
    }
}


/// Q4 線形の (uid, blocks, k, n)。Q4 以外は None。
pub(crate) fn q4_parts(w: &crate::quant::Linear) -> Option<(u64, &[u8], usize, usize)> {
    match w {
        crate::quant::Linear::Q4(q) => Some((q.uid, &q.blocks, q.k, q.n)),
        _ => None,
    }
}
