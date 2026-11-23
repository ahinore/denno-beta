//! decode MoE GEMV (v1..v5 / seg / exp_iq / dec_epi / shared_overlap)
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


// ---------------------------------------------------------------------------
// P-pf-moe-scratch: `moe_gemv_dev` の入出力スクラッチ (thread_local, grow-only)
// ---------------------------------------------------------------------------
//
// 背景 (pf_arena の調査 = 85cd04d8): `--max-ctx` を上げると初回リクエストの
// 重み登録が失敗する (`要求 644 MiB / 空き 7 MiB`)。max_ctx は GPU 側の確保に
// 入っていないので、残る原因は「1 GEMV = cuMemAlloc 2 本 + cuMemFree 2 本」を
// decode 中に数万回繰り返すことによる断片化。行列の形はモデルで固定なので、
// スレッドごとに grow-only で持ち回す (既存 `Scratch` と同じ規約)。
//
// - 確保するのは「今の容量 < 今回の要求」のときだけ。より大きい 1 本に
//   置き換え、古い方はその場で drop = cuMemFree する (毎回は free しない)。
// - 数値は不変: カーネル・引数・発行順・丸め順は 1 つも変えない。x は毎回
//   `n_in` 要素を先頭から書き、out は `n_out` 要素だけ読む (前回の残りは読まない)。
// - スレッド終了時の解放はベストエフォート (`Scratch` と同じ規約)。
pub(crate) struct MoeScratch {
    pub(crate) x: Option<DevBuffer>, // >= n_in * 4 B
    pub(crate) o: Option<DevBuffer>, // >= n_out * 4 B
    // P-async: pinned の staging (H2D / D2H を async にするため)。
    // [`gemv_async_on`] が真のときだけ確保する (既定 on、`DENNO_GEMV_ASYNC=0` で off)。
    pub(crate) pin_x: Option<PinMem>,
    pub(crate) pin_o: Option<PinMem>,
}


/// P-gemv2: x 先読み版の GEMV カーネルを使うか (既定 **on**)。
///
/// `DENNO_GEMV_V2=0` で v1 に戻る。v1 と v2 は同じ数値契約なので、切り替えても
/// 出力ビットは一致する (速度だけが変わる)。
pub fn moe_gemv_v2_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("DENNO_GEMV_V2").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}


/// decode GEMV のレーン分割版 (`kernels_moe_v3.cu`、4 レーンまたは 2 レーン) を
/// 使うか (既定 **off**)。
///
/// `DENNO_GEMV_V3=1` のときだけ v3。未設定 / `0` / `off` / `false` では v2 (既定
/// 経路) のままで、1 バイトも挙動が変わらない。
///
/// v3 は 1 行の累積を lanes レーンで分担するので **v2 とビット一致しない** (加算順が
/// 変わる。prefill の GPU バッチ GEMM と同じ扱い = 契約の外)。既定で使わない理由。
pub fn moe_gemv_v3_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("DENNO_GEMV_V3").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}


/// v3 のレーン数 (4 / 2 / 0 = v2)。純関数なので GPU 無しで検証できる。
///
/// v3 のレーン l は k を lanes 等分した連続区間 `[l*k/lanes, (l+1)*k/lanes)` を
/// 持ち、区間の中で量子ブロック (最小 32 要素) を丸ごと読む。したがって区間長
/// k/lanes が 32 の倍数であることが要る:
///
/// - k が 128 の倍数 → 4 レーン (区間長 k/4 が 32 の倍数)
/// - k が 64 の倍数で 128 の倍数でない → 2 レーン (区間長 k/2 が 32 の倍数)
/// - それ以外 → 0 (= v2 へ落ちる。panic しない)
///
/// 実形: 2560 / 5120 / 6144 は 4 レーン、320 (`output_hc_up` の k) は 2 レーン、
/// 1000 は v2。2 レーン経路はカーネル引数 `lanes` で切り替える (カーネルは 1 組のまま)。
pub fn moe_gemv_v3_lanes(k: usize) -> usize {
    if k >= 128 && k % 128 == 0 {
        4
    } else if k >= 64 && k % 64 == 0 {
        2
    } else {
        0
    }
}


/// v3 (4 レーン版または 2 レーン版) が使える k か。
pub fn moe_gemv_v3_k_ok(k: usize) -> bool {
    moe_gemv_v3_lanes(k) > 0
}


/// v3 の 1 ブロックあたりの行数 (`DENNO_GEMV_V3_ROWS`、既定 8)。
///
/// 8 / 16 / 32 / 64 のどれか。それ以外 (未設定・空・数値でない・対象外の値) は 64。
/// blockDim = rows * lanes、grid = ceil(n_out / rows) になる。n_out=2560 の形状では
/// rows=64 で 40 block、rows=8 で 320 block (5090 の 170 SM を埋める)。
/// 既定 8 は main の実測 (rows 8 が最良) による。kernels_moe_v3.cu の MOE_V3_ROWS_DEFAULT は引数の既定値なので、実際の値はホスト側が渡す。
pub fn moe_gemv_v3_rows() -> usize {
    // テストが rows を振って 1 プロセスで全組み合わせを回すための上書き
    // (0 = env を見る)。env は OnceLock で 1 回しか読めないため。
    #[cfg(test)]
    {
        let f = V3_ROWS_FORCE.load(std::sync::atomic::Ordering::Relaxed);
        if f != 0 {
            return f;
        }
    }
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        let raw = std::env::var("DENNO_GEMV_V3_ROWS").ok();
        let rows = parse_v3_rows(raw.as_deref());
        if let Some(v) = raw.as_deref() {
            if !matches!(v.trim(), "8" | "16" | "32" | "64") {
                eprintln!("[gemv-v3] DENNO_GEMV_V3_ROWS={v:?} は 8/16/32/64 以外 → {rows} を使う");
            }
        }
        rows
    })
}


/// v3 の PTX が読めているか (テスト専用。**context を current にしてから**呼ぶこと。
/// 失敗を `OnceLock` に焼き付けるので、本番経路からは呼ばない)。
#[cfg(test)]
pub(crate) fn moe_gemv_v3_loaded() -> bool {
    gemv_v3_kernels().is_some()
}


/// 形からレーン数を決める (純関数)。`cap` は上限 (4..32 に丸める)。
///
/// レーン l は k を l 等分した連続区間を持つので、区間長が量子ブロック (32 要素)
/// の倍数でなければならない = `n_in % (l * 32) == 0`。満たす最大の 2 の冪を
/// 返し、4 以上が無ければ 0 (= v3 に任せる)。
pub(crate) fn v4_lanes_for(n_in: usize, cap: usize) -> usize {
    let mut l = cap.clamp(4, 32);
    while l & (l - 1) != 0 {
        l -= 1; // 2 の冪へ切り下げる
    }
    while l >= 4 {
        if n_in % (l * 32) == 0 {
            return l;
        }
        l >>= 1;
    }
    0
}


/// v4 が載っているか (テスト専用。context を current にしてから呼ぶこと)。
#[cfg(test)]
pub(crate) fn moe_gemv_v4_loaded() -> bool {
    gemv_v4_kernels().is_some()
}


/// v5 が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn moe_gemv_v5_loaded() -> bool {
    gemv_v5_kernels().is_some()
}


pub(crate) fn v5_q5k_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_GEMV_V5_Q5K_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_v5_q5k.ptx"),
    }
}


/// Q5_K v5 が載っているか (テスト専用)。
#[cfg(test)]
pub(crate) fn moe_gemv_v5_q5k_loaded() -> bool {
    gemv_v5_q5k_kernels().is_some()
}


/// [`parse_v5_lanes`] をプロセスで 1 回だけ読む。
pub(crate) fn v5_lanes_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    *CAP.get_or_init(|| parse_v5_lanes(std::env::var("DENNO_GEMV_V5_LANES").ok().as_deref()))
}


/// 割り切れる最大の 2 の冪 (1 以上、`cap` 以下)。無ければ 0。
pub(crate) fn v5_lanes_div(n: usize, cap: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut l = cap.clamp(1, 32);
    while l & (l - 1) != 0 {
        l -= 1;
    }
    while l >= 1 {
        if l > 0 && n % l == 0 {
            return l;
        }
        l >>= 1;
    }
    0
}


/// fast mode (`kernels_moe_fp16.cu`、fp16 dequant + half2 FMA) を使うか (既定 **off**)。
///
/// `DENNO_GEMV_FP16=1` のときだけ有効。v3 と同じ降下経路に乗る: 対応型は
/// Q8_0 / Q5_1 / Q4_K の 3 つ (Q5_K は対象外)、k が 64 の倍数でない、PTX が
/// 読めない / 旧 ABI のときは v3 (on なら) → v2 へ落ちる。env 未指定では
/// 1 バイトも挙動が変わらない。
pub fn moe_gemv_fp16_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_gemv_fp16(std::env::var("DENNO_GEMV_FP16").ok().as_deref()))
}


/// fp16 の PTX が読めているか (テスト専用。**context を current にしてから**呼ぶこと。
/// 失敗を `OnceLock` に焼き付けるので、本番経路からは呼ばない)。
#[cfg(test)]
pub(crate) fn moe_gemv_fp16_loaded() -> bool {
    gemv_fp16_kernels().is_some()
}


impl MoeScratch {
    pub(crate) fn ensure(cap: &mut Option<DevBuffer>, want: usize) -> Result<()> {
        Self::ensure_tagged(cap, want, crate::mem_audit::TAG_OTHER)
    }
    /// task 149: 台帳のタグ付き [`Self::ensure`]。
    pub(crate) fn ensure_tagged(cap: &mut Option<DevBuffer>, want: usize, tag: &'static str) -> Result<()> {
        let cur = cap.as_ref().map(|b| b.bytes()).unwrap_or(0);
        if let Some(bytes) = moe_scratch_alloc_bytes(cur, want) {
            *cap = Some(DevBuffer::alloc_tagged(bytes, tag)?);
        }
        Ok(())
    }

    /// P-async: pinned staging の grow-only 確保。
    pub(crate) fn ensure_pin(cap: &mut Option<PinMem>, bytes: usize) -> Result<()> {
        if cap.as_ref().map(|b| b.bytes).unwrap_or(0) < bytes {
            *cap = Some(PinMem::alloc(bytes)?);
        }
        Ok(())
    }
}


/// 層一括 GEMV が使えるか (`false` なら呼び出し側は per-expert 経路へ)。
///
/// `DENNO_CUDA` 無効・未対応の型・PTX 不在のいずれでも false。**載せる前**の
/// 判定に使う (載せてから落ちると VRAM が無駄に埋まる)。
pub fn moe_gemv_seg_ready(ty: crate::kquant::GgmlType, k: usize) -> bool {
    if !cuda_enabled() || gguf_row_bytes(ty, k).is_none() {
        return false;
    }
    moe_batch_slot(ty).is_some() && moe_batch_kernels().is_some()
}


/// スレッド局所のスクラッチ (grow-only)。`MoeScratch` と同じ規約。
pub(crate) struct SegScratch {
    /// segment ごとの重みポインタ表 (デバイス)
    ptr: [Option<DevBuffer>; 2],
    /// segment 表 (4 語 x nseg)
    seg: Option<DevBuffer>,
    /// 入力 (segment の x_off を足して読む)
    x: Option<DevBuffer>,
    /// 出力 (segment の out_off を足して書く)
    out: [Option<DevBuffer>; 2],
}


impl SegScratch {
    pub(crate) fn new() -> Self {
        SegScratch {
            ptr: [None, None],
            seg: None,
            x: None,
            out: [None, None],
        }
    }
}


thread_local! {
    pub(crate) static SEG_SCRATCH: std::cell::RefCell<SegScratch> =
        std::cell::RefCell::new(SegScratch::new());
}


/// 1 回の同期で終わる層一括 GEMV。`jobs` は 1 本か 2 本 (gate, up)。
///
/// 前提 (呼び出し側が検査済み): 全 job が同じ `n_out` / `row_bytes` / segment 表。
/// 発行順は job の並び順で、**同期は最後に 1 回だけ**。
fn moe_gemv_seg_run(jobs: &mut [SegJob<'_>], seg: &[u32], m_max: usize, x: &[f32]) -> Result<()> {
    if jobs.is_empty() || jobs.len() > 2 {
        bail!("moe_gemv_seg_run: job 数 {} が 1..=2 でない", jobs.len());
    }
    if !seg.len().is_multiple_of(4) {
        bail!("moe_gemv_seg_run: segment 表の長さ {} が 4 の倍数でない", seg.len());
    }
    let nseg = seg.len() / 4;
    let (n_out, row_bytes) = (jobs[0].n_out, jobs[0].row_bytes);
    for (i, j) in jobs.iter().enumerate() {
        if j.n_out != n_out || j.row_bytes != row_bytes {
            bail!(
                "moe_gemv_seg_run: job {i} の形が揃っていない (n_out={}/{n_out}, row_bytes={}/{row_bytes})",
                j.n_out,
                j.row_bytes
            );
        }
        if j.wptrs.len() != nseg {
            bail!(
                "moe_gemv_seg_run: job {i} のポインタ表 {} が segment 数 {nseg} と違う",
                j.wptrs.len()
            );
        }
        let (mx, mo) = crate::model::expert_batch::seg_extents(seg, j.n_in, j.n_out)?;
        if x.len() < mx {
            bail!(
                "moe_gemv_seg_run: job {i} の入力長 {} が segment の最大 x 範囲 {mx} より短い",
                x.len()
            );
        }
        if j.out.len() < mo {
            bail!(
                "moe_gemv_seg_run: job {i} の出力長 {} が segment の最大 out 範囲 {mo} より短い",
                j.out.len()
            );
        }
    }
    let Some(plan) = crate::model::expert_batch::seg_launch_plan(nseg, m_max, n_out, row_bytes) else {
        bail!("moe_gemv_seg_run: 起動形が作れない (nseg={nseg}, m_max={m_max}, n_out={n_out})");
    };
    let Some(ctx) = shared_ctx() else {
        bail!("moe_gemv_seg_run: CUDA コンテキストが無い");
    };
    ctx.make_current()?;
    let Some(kernels) = moe_batch_kernels() else {
        bail!("moe_gemv_seg_run: カーネルが無い");
    };

    SEG_SCRATCH.with(|cell| -> Result<()> {
        let mut sc = cell.borrow_mut();
        // task 278: 旧経路の区間計測 (`DENNO_SEG_PROF`)。off は stamp が None。
        let prof = crate::model::seg_prof::enabled();
        let mut gpu_bytes = [0usize; 2];
        // 1. ポインタ表 / segment 表 / 入力 (すべて起動より前に H2D)
        let _t_h2d = crate::model::seg_prof::stamp();
        for (i, job) in jobs.iter().enumerate() {
            MoeScratch::ensure_tagged(&mut sc.ptr[i], job.wptrs.len() * 8, crate::mem_audit::TAG_SEG)?;
            let pb = sc.ptr[i].as_ref().expect("確保済み");
            pb.copy_from(job.wptrs)?;
            dbg_htod(job.wptrs.len() * 8);
            gpu_bytes[i] = job.row_bytes * job.n_out * nseg;
        }
        MoeScratch::ensure_tagged(&mut sc.seg, seg.len() * 4, crate::mem_audit::TAG_SEG)?;
        MoeScratch::ensure_tagged(&mut sc.x, x.len() * 4, crate::mem_audit::TAG_SEG)?;
        {
            let sb = sc.seg.as_ref().expect("確保済み");
            sb.copy_from(seg)?;
            dbg_htod(seg.len() * 4);
            let xb = sc.x.as_ref().expect("確保済み");
            xb.copy_from(x)?;
            dbg_htod(x.len() * 4);
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_H2D, _t_h2d);
        for (i, job) in jobs.iter().enumerate() {
            MoeScratch::ensure_tagged(&mut sc.out[i], job.out.len() * 4, crate::mem_audit::TAG_SEG)?;
        }
        // 2. 起動 (job ごと。まだ同期しない)
        let seg_buf = sc.seg.as_ref().expect("確保済み");
        let x_buf = sc.x.as_ref().expect("確保済み");
        let _ = seg_ev::record(0);
        unsafe {
            for (i, job) in jobs.iter().enumerate() {
                let slot = moe_batch_slot(job.ty)
                    .ok_or_else(|| anyhow!("未対応の量子型 {:?}", job.ty))?;
                let mut pw = sc.ptr[i].as_ref().expect("確保済み").ptr;
                let mut px = x_buf.ptr;
                let mut po = sc.out[i].as_ref().expect("確保済み").ptr;
                let mut ps = seg_buf.ptr;
                let (mut ns, mut ni, mut no, mut rb) = (
                    nseg as c_int,
                    job.n_in as c_int,
                    job.n_out as c_int,
                    job.row_bytes as c_int,
                );
                let mut args = [
                    &mut pw as *mut _ as *mut c_void,
                    &mut px as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut ps as *mut _ as *mut c_void,
                    &mut ns as *mut _ as *mut c_void,
                    &mut ni as *mut _ as *mut c_void,
                    &mut no as *mut _ as *mut c_void,
                    &mut rb as *mut _ as *mut c_void,
                ];
                let _t_launch = crate::model::seg_prof::stamp();
                ctx.launch_async(
                    kernels.fns[slot],
                    plan.grid,
                    plan.block,
                    plan.smem,
                    &mut args,
                )?;
                crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
                let _ = seg_ev::record(i + 1);
            }
            // 3. 同期は 1 回だけ (gate と up が同じ同期で終わる)
            let _t_sync = crate::model::seg_prof::stamp();
            ctx.sync()?;
            crate::model::seg_prof::add(crate::model::seg_prof::PH_SYNC, _t_sync);
        }
        if prof {
            // 起動ごとの GPU 時間 (イベント)。1 本目 = gate、2 本目 = up。
            for i in 0..jobs.len().min(2) {
                if let Some(ns) = seg_ev::elapsed_ns(i, i + 1) {
                    let g = if jobs.len() == 1 {
                        crate::model::seg_prof::GP_DOWN
                    } else if i == 0 {
                        crate::model::seg_prof::GP_GATE
                    } else {
                        crate::model::seg_prof::GP_UP
                    };
                    crate::model::seg_prof::add_gpu(g, ns, gpu_bytes[i]);
                }
            }
        }
        // 4. D2H
        let _t_d2h = crate::model::seg_prof::stamp();
        for (i, job) in jobs.iter_mut().enumerate() {
            let got: Vec<f32> = sc.out[i].as_ref().expect("確保済み").to_vec(job.out.len())?;
            dbg_d2h(job.out.len() * 4);
            job.out.copy_from_slice(&got);
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_D2H, _t_d2h);
        Ok(())
    })
}


/// gate と up を **2 起動 + 1 同期**で回し、両方の出力を返す (P43)。
///
/// `seg` は gate/up 用 (全 segment が同じ入力 x を読むので `x_off = 0`)。
/// `Ok(false)` は「GPU でやらなかった」= 呼び出し側は per-expert 経路へ。
#[allow(clippy::too_many_arguments)]
pub fn moe_gemv_seg_pair(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    wptrs_g: &[u64],
    wptrs_u: &[u64],
    seg: &[u32],
    m_max: usize,
    x: &[f32],
    n_in: usize,
    n_out: usize,
    out_g: &mut [f32],
    out_u: &mut [f32],
) -> Result<bool> {
    let (Some(rb_g), Some(rb_u)) = (gguf_row_bytes(ty_g, n_in), gguf_row_bytes(ty_u, n_in)) else {
        return Ok(false);
    };
    if rb_g != rb_u || moe_batch_slot(ty_g).is_none() || moe_batch_slot(ty_u).is_none() {
        return Ok(false);
    }
    if !moe_gemv_seg_ready(ty_g, n_in) || !moe_gemv_seg_ready(ty_u, n_in) {
        return Ok(false);
    }
    let mut jobs = [
        SegJob {
            ty: ty_g,
            wptrs: wptrs_g,
            n_in,
            n_out,
            row_bytes: rb_g,
            out: out_g,
        },
        SegJob {
            ty: ty_u,
            wptrs: wptrs_u,
            n_in,
            n_out,
            row_bytes: rb_u,
            out: out_u,
        },
    ];
    // task 278: 層の数を 1 回だけ数える (down の呼び出しと二重に数えない)。
    crate::model::seg_prof::note_layer();
    moe_gemv_seg_run(&mut jobs, seg, m_max, x)?;
    Ok(true)
}


/// 1 本だけ (down) を **1 起動 + 1 同期**で回す (P43)。
///
/// `seg` は down 用 (`x_off = s * n_in`, `out_off = s * n_out`)。
/// `Ok(false)` は「GPU でやらなかった」= 呼び出し側は per-expert 経路へ。
#[allow(clippy::too_many_arguments)]
pub fn moe_gemv_seg_one(
    ty: crate::kquant::GgmlType,
    wptrs: &[u64],
    seg: &[u32],
    m_max: usize,
    x: &[f32],
    n_in: usize,
    n_out: usize,
    out: &mut [f32],
) -> Result<bool> {
    let Some(rb) = gguf_row_bytes(ty, n_in) else {
        return Ok(false);
    };
    if moe_batch_slot(ty).is_none() || !moe_gemv_seg_ready(ty, n_in) {
        return Ok(false);
    }
    let mut jobs = [SegJob {
        ty,
        wptrs,
        n_in,
        n_out,
        row_bytes: rb,
        out,
    }];
    moe_gemv_seg_run(&mut jobs, seg, m_max, x)?;
    Ok(true)
}


/// [`SEG_K_ENV`] の解釈 (純関数)。未設定は有効。
pub fn seg_k_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(str::trim).map(str::to_ascii_lowercase).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}


/// 高速版 GEMV が有効か (プロセスで 1 回だけ env を読む)。
pub(crate) fn seg_k_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| seg_k_from_value(std::env::var(SEG_K_ENV).ok().as_deref()))
}


/// PTX の置き場所。`DENNO_SEG_K_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_seg_k.ptx`。
fn seg_k_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_SEG_K_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_seg_k.ptx"),
    }
}


/// 高速版のカーネル一式 (初回だけ PTX を読む。失敗は None をキャッシュ)。
pub(crate) fn seg_k_kernels() -> Option<&'static SegKKernels> {
    SEG_K_KERNELS
        .get_or_init(|| {
            let path = seg_k_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    // 既定 on のゲートなので、欠けたら**静かに**遅い方へ落ちる。
                    // 1 行だけ大きく出す (gate_audit の監査と同じ趣旨)。
                    eprintln!(
                        "[seg-k] PTX が読めない ({}) : {e} → 旧カーネル (遅い) で回す",
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
                    return None;
                }
            }
            let get = |name: &[u8]| -> Option<CUfunction> {
                let mut f: CUfunction = std::ptr::null_mut();
                let r = unsafe {
                    (lib.module_get_function)(&mut f, module, name.as_ptr() as *const c_char)
                };
                if r == CUDA_SUCCESS {
                    Some(f)
                } else {
                    None
                }
            };
            let mut fns = [std::ptr::null_mut(); 4];
            for (i, name) in SEG_K_NAMES.iter().enumerate() {
                fns[i] = get(name)?;
            }
            // task 282: mix 用の 2 本は任意 (無ければ旧 GEMV へ落ちる)。
            let mix_r8 = get(b"dno_mix_gemv_q8_0_r8\0");
            let mix_rowblk = get(b"dno_mix_gemv_q8_0_rowblk\0");
            // task 412 / 526: 融合 (gu) の 4 型は任意 (古い PTX でも動く)。
            let mut gu = [None; 4];
            for (i, name) in SEG_K_GU_NAMES.iter().enumerate() {
                gu[i] = get(name);
            }
            // task 534: 8 バイトロード版 (K 型だけ)。古い PTX では None。
            let mut gu8 = [None; 4];
            for (i, name) in SEG_K_GU8_NAMES.iter().enumerate() {
                if let Some(name) = name {
                    gu8[i] = get(name);
                }
            }
            let mut k8 = [None; 4];
            for (i, name) in SEG_K_K8_NAMES.iter().enumerate() {
                if let Some(name) = name {
                    k8[i] = get(name);
                }
            }
            let mut v4 = [None; 4];
            for (i, name) in SEG_K_V4_NAMES.iter().enumerate() {
                if let Some(name) = name {
                    v4[i] = get(name);
                }
            }
            // task 562: 16 バイトロード版 (Q4_K の gate/up)。古い PTX では None。
            let mut gu16 = [None; 4];
            for (i, name) in SEG_K_GU16_NAMES.iter().enumerate() {
                if let Some(name) = name {
                    gu16[i] = get(name);
                }
            }
            Some(SegKKernels {
                module,
                fns,
                mix_r8,
                mix_rowblk,
                gu,
                gu8,
                k8,
                v4,
                gu16,
            })
        })
        .as_ref()
}


/// prof 用のイベント (null ストリーム)。`DENNO_SEG_PROF` off では何もしない。
pub(crate) mod seg_ev {
    use super::*;

    thread_local! {
        static EVS: std::cell::RefCell<Vec<CUevent>> = const { std::cell::RefCell::new(Vec::new()) };
        static DEAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }

    /// スロット `i` のイベントを用意する (初回だけ作る)。
    pub(crate) fn slot(i: usize) -> Option<CUevent> {
        if DEAD.with(|d| d.get()) {
            return None;
        }
        let l = lib().ok()?;
        let Some(create) = l.event_create else {
            DEAD.with(|d| d.set(true));
            return None;
        };
        EVS.with(|c| {
            let mut v = c.borrow_mut();
            while v.len() <= i {
                let mut ev: CUevent = std::ptr::null_mut();
                // 0 = CU_EVENT_DEFAULT (計時あり)
                if unsafe { create(&mut ev, 0) } != CUDA_SUCCESS || ev.is_null() {
                    DEAD.with(|d| d.set(true));
                    return None;
                }
                v.push(ev);
            }
            Some(v[i])
        })
    }

    /// null ストリームの「いま」をイベントに打つ (**prof の on/off を見ない**)。
    pub(crate) fn record_raw(i: usize) -> bool {
        let Some(ev) = slot(i) else { return false };
        let Some(rec) = lib().ok().and_then(|l| l.event_record) else {
            return false;
        };
        unsafe { rec(ev, std::ptr::null_mut()) == CUDA_SUCCESS }
    }

    /// prof が on のときだけイベントを打つ。
    pub(crate) fn record(i: usize) -> bool {
        if !crate::model::seg_prof::enabled() {
            return false;
        }
        record_raw(i)
    }

    /// イベント 1 本を待つ (文脈全体ではなく null ストリームの仕事だけ)。
    pub(crate) fn wait(i: usize) -> Result<()> {
        let Some(ev) = slot(i) else {
            return Ok(());
        };
        let Some(es) = lib().ok().and_then(|l| l.event_sync) else {
            return Ok(());
        };
        check(unsafe { es(ev) }, "cuEventSynchronize (seg lay)")?;
        Ok(())
    }

    /// 2 イベント間の ns (取れなければ `None`)。
    pub(crate) fn elapsed_ns(a: usize, b: usize) -> Option<u64> {
        let (ea, eb) = (slot(a)?, slot(b)?);
        let f = lib().ok().and_then(|l| l.event_elapsed_time)?;
        let mut ms = 0.0f32;
        if unsafe { f(&mut ms, ea, eb) } != CUDA_SUCCESS {
            return None;
        }
        Some((ms.max(0.0) as f64 * 1e6) as u64)
    }
}


thread_local! {
    /// 非同期 D2H の行き先 (このスレッド専用)。
    pub(crate) static SEG_ASYNC_PIN: std::cell::RefCell<Option<PinMem>> =
        const { std::cell::RefCell::new(None) };
}


/// PTX の置き場所。`DENNO_EXP_IQ_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_exp_iq.ptx`。
fn exp_iq_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_IQ_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq.ptx"),
    }
}


/// task 190: v2 の PTX の置き場所 (`DENNO_EXP_IQ_V2_PTX` で差し替え可)。
fn exp_iq_v2_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_IQ_V2_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq_v2.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける
/// (= 二度と試さない。呼び出し側は CPU へ落ちる)。
fn exp_iq_kernels() -> Option<&'static ExpIqKernels> {
    EXP_IQ_KERNELS
        .get_or_init(|| load_exp_iq_module(&exp_iq_ptx_path(), "v1"))
        .as_ref()
}


/// task 190: v2 カーネル (無ければ `None`)。
fn exp_iq_kernels_v2() -> Option<&'static ExpIqKernels> {
    EXP_IQ_KERNELS_V2
        .get_or_init(|| load_exp_iq_module(&exp_iq_v2_ptx_path(), "v2"))
        .as_ref()
}


/// task 275: v3 の PTX の置き場所 (`DENNO_EXP_IQ_V3_PTX` で差し替え可)。
fn exp_iq_v3_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_IQ_V3_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq_v3.ptx"),
    }
}


/// task 275: v3 カーネル (無ければ `None` = v2 のまま)。
fn exp_iq_kernels_v3() -> Option<&'static ExpIqKernels> {
    EXP_IQ_KERNELS_V3
        .get_or_init(|| {
            load_exp_iq_slots(
                &exp_iq_v3_ptx_path(),
                "v3",
                &crate::model::exp_iq::IQ_V3_SLOTS,
            )
        })
        .as_ref()
}


/// decode の層一括 GEMV に使うカーネル一式。
///
/// `DENNO_EXP_IQ_V2=1` かつ v2 の PTX が読めたときだけ v2。それ以外は v1
/// (v2 の PTX が無いときに CPU へ落とさない = 性能だけのゲート)。
/// task 275: `DENNO_EXP_IQ_V3=1` かつ v3 の PTX が読めたときは、v3 が持つスロット
/// (`IQ_V3_SLOTS`) だけ v3 の関数で置き換える。PTX のモジュールはそれぞれの
/// `OnceLock` がプロセス終了まで保持するので、ハンドルは有効なまま。
pub(crate) fn exp_iq_kernels_decode() -> Option<&'static ExpIqKernels> {
    EXP_IQ_KERNELS_DECODE
        .get_or_init(|| {
            let base = if crate::model::exp_iq::iq_v2_enabled() {
                exp_iq_kernels_v2().or_else(exp_iq_kernels)?
            } else {
                exp_iq_kernels()?
            };
            let mut k = ExpIqKernels {
                module: base.module,
                fns: base.fns,
                silu: base.silu,
            };
            if crate::model::exp_iq::iq_v3_enabled() {
                if let Some(v3) = exp_iq_kernels_v3() {
                    for &i in crate::model::exp_iq::IQ_V3_SLOTS.iter() {
                        if !v3.fns[i].is_null() {
                            k.fns[i] = v3.fns[i];
                        }
                    }
                }
            }
            Some(k)
        })
        .as_ref()
}


/// この型・この k を GPU で回せるか。
///
/// `DENNO_EXPERT_IQ_GPU` が off・未対応の型・PTX 不在のいずれでも false。
/// **載せる前**の判定 (`expert_vram::acquire`) に使う (載せてから落ちると VRAM が
/// 無駄に埋まる)。
pub fn moe_gemv_iq_ready(ty: crate::kquant::GgmlType, k: usize) -> bool {
    if !cuda_enabled() || !crate::model::exp_iq::iq_gpu_enabled() {
        return false;
    }
    if !crate::model::exp_iq::supported(ty)
        || crate::model::exp_iq::row_bytes(ty, k).is_none()
    {
        return false;
    }
    exp_iq_kernels().is_some()
}


/// 1 層ぶんの作業域 (grow-only。スレッドローカル)。
pub(crate) struct IqScratch {
    pub(crate) x: Option<DevBuffer>,
    pub(crate) g: Option<DevBuffer>,
    pub(crate) u: Option<DevBuffer>,
    pub(crate) act: Option<DevBuffer>,
    pub(crate) h: Option<DevBuffer>,
    pub(crate) pg: Option<DevBuffer>,
    pub(crate) pu: Option<DevBuffer>,
    pub(crate) pd: Option<DevBuffer>,
    pub(crate) sgu: Option<DevBuffer>,
    pub(crate) sgd: Option<DevBuffer>,
    /// task 150: `h` の D2H を非同期で受ける pinned staging (overlap 経路だけ)。
    pub(crate) hpin: Option<PinMem>,
    /// task 187: 和モードの重み (nseg 本) と和の出力 (e 要素)。
    pub(crate) wt: Option<DevBuffer>,
    pub(crate) psum: Option<DevBuffer>,
    /// task 356: 和モードの ZC (`DENNO_DEC_EXP_SUM_ZC`) のときの和の出力先。
    /// pinned のホスト (UVA でデバイスからも書ける) = D2H を消すため。
    pub(crate) psum_zc: Option<PinMem>,
    /// task 204: gate+up 融合 (`DENNO_DEC_EXPGU`) の連結出力 `[g[ng] | u[ng]]`。
    pub(crate) gu: Option<DevBuffer>,
}


impl IqScratch {
    pub(crate) fn new() -> Self {
        IqScratch {
            x: None,
            g: None,
            u: None,
            act: None,
            h: None,
            pg: None,
            pu: None,
            pd: None,
            sgu: None,
            sgd: None,
            hpin: None,
            wt: None,
            psum: None,
            psum_zc: None,
            gu: None,
        }
    }
}


/// 1 層ぶんの expert FFN を GPU で回す (gate/up は IQ, down は IQ4_NL/Q2_0)。
///
/// `pg` / `pu` / `pd` は segment ごとの重み先頭 (デバイスアドレス)、
/// `seg_gu` は gate/up 用 (`x_off = 0`)、`seg_d` は down 用 (`x_off = s*ff`)。
/// `out_h` は `nseg * e` 要素。呼び出し側は結果を expert index 昇順で足し込む。
///
/// `Ok(false)` は「GPU でやらなかった」= 呼び出し側は CPU 経路へ (fail closed)。
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_layer(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    pg: &[u64],
    pu: &[u64],
    pd: &[u64],
    seg_gu: &[u32],
    seg_d: &[u32],
    x: &[f32],
    n_in: usize,
    ff: usize,
    e: usize,
    out_h: &mut [f32],
) -> Result<bool> {
    iq_layer_run(
        ty_g, ty_u, ty_d, pg, pu, pd, seg_gu, seg_d, x, n_in, ff, e, out_h, &[], 0, false,
        || Ok(()),
    )
}


/// task 150: [`moe_experts_iq_layer`] の**重ね打ち版**。
///
/// 発行するカーネル・引数・順序・値は逐次版と 1 つも変わらない。違うのは待ち方だけで、
/// **D2H を非同期で出してから `cpu_miss` を実行し、最後に 1 回だけ同期する**。
/// decode では 1 層あたり GPU 5〜6 ms 待つのに CPU ミスが 57 ms あるので (lab t148b)、
/// 待ちを CPU の仕事の裏へ隠せる。`cpu_miss` は GPU の作業域に触れてはいけない
/// (呼び出し側はミス expert の CPU 寄与だけを書く)。
///
/// `Ok(false)` = GPU でやらなかった (どの起動よりも前に落ちた)。このとき
/// `cpu_miss` は**呼ばれない**ので、呼び出し側は従来どおり全 expert を CPU で
/// 計算し直せる。`cpu_miss` が `Err` を返したときは**そのまま Err を返す**
/// (起動済みで `out_h` の一部が確定しているため CPU 経路へは戻さない)。
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_layer_overlap<F>(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    pg: &[u64],
    pu: &[u64],
    pd: &[u64],
    seg_gu: &[u32],
    seg_d: &[u32],
    x: &[f32],
    n_in: usize,
    ff: usize,
    e: usize,
    out_h: &mut [f32],
    cpu_miss: F,
) -> Result<bool>
where
    // task 388: `DENNO_DEC_MISS_SPAWN` で `cpu_miss` を rayon のプールへ渡すので
    // `Send` を要求する (借用は scoped のまま)。
    F: FnOnce() -> Result<()> + Send,
{
    iq_layer_run(
        ty_g, ty_u, ty_d, pg, pu, pd, seg_gu, seg_d, x, n_in, ff, e, out_h, &[], 0, true,
        cpu_miss,
    )
}


/// task 187: [`moe_experts_iq_layer_overlap`] の**デバイス和**版。
///
/// gate/up/silu/down を積んだあとに `dno_dec_exp_sum` を 1 発積み、
/// `out` (`e` 要素) へ `sum_t w[t] * h[t*e+j]` を書いて **e 要素だけ** D2H する。
/// 寄与 (`nseg * e` 要素 = 約 100 KB/層) はホストへ戻さない。
///
/// `w` は `pg` と同じ並び (segment = 選択順) の重み。`Ok(false)` は
/// 「GPU でやらなかった」= 呼び出し側は従来経路へ (このとき `out` は 1 要素も
/// 書かれない)。
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_layer_sum_overlap<F>(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    pg: &[u64],
    pu: &[u64],
    pd: &[u64],
    seg_gu: &[u32],
    seg_d: &[u32],
    x: &[f32],
    n_in: usize,
    ff: usize,
    e: usize,
    w: &[f32],
    out: &mut [f32],
    cpu_miss: F,
) -> Result<bool>
where
    F: FnOnce() -> Result<()> + Send,
{
    iq_layer_run(
        ty_g, ty_u, ty_d, pg, pu, pd, seg_gu, seg_d, x, n_in, ff, e, out, w, e, true, cpu_miss,
    )
}


// ===========================================================================
// task 187: decode の epilogue カーネル (`kernels_dec_epi.cu` → `.ptx`)
// ===========================================================================
//
// - `dno_dec_exp_sum`: ヒットした expert の寄与を **デバイスで**重み付き和にする。
//   従来は 1 層あたり `nseg * e` 要素 (約 100 KB) を D2H してホストで足していた。
//   t 昇順の 1 アキュムレータなので、ミスの無い層ではホストの足し込みとビット一致。
// - `dno_dec_silu_mul`: shared expert の `silu(gate) * up` をデバイスで作る
//   (将来 shared 段をデバイスで閉じるためのもの。今回は未使用でも PTX は読む)。
//
// PTX が無い / 読めないときは `dec_epi_kernels()` が `None` を返し、呼び出し側は
// 従来経路へ落ちる (fail closed。数値は変わらない)。

pub(crate) struct DecEpiKernels {
    #[allow(dead_code)]
    module: CUmodule,
    /// ヒットの寄与 + shared の足し込み
    pub(crate) sum: CUfunction,
    /// silu(g) * u
    #[allow(dead_code)]
    pub(crate) silu: CUfunction,
}


unsafe impl Send for DecEpiKernels {}

unsafe impl Sync for DecEpiKernels {}


/// PTX の置き場所。`DENNO_DEC_EPI_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_dec_epi.ptx`。
fn dec_epi_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_DEC_EPI_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_dec_epi.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む。失敗は None を焼き付ける)。
pub(crate) fn dec_epi_kernels() -> Option<&'static DecEpiKernels> {
    DEC_EPI_KERNELS
        .get_or_init(|| {
            let path = dec_epi_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            image.push(0);
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    return None;
                }
            }
            let get = |module: CUmodule, name: &str| -> Option<CUfunction> {
                let mut cname = name.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                unsafe {
                    if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        return None;
                    }
                }
                Some(f)
            };
            let sum = get(module, "dno_dec_exp_sum")?;
            let silu = get(module, "dno_dec_silu_mul")?;
            Some(DecEpiKernels { module, sum, silu })
        })
        .as_ref()
}


/// 和のカーネルが使えるか (PTX があるか)。
pub fn dec_epi_ready() -> bool {
    cuda_enabled() && dec_epi_kernels().is_some()
}


/// 語彙 GEMV (head) を v5 カーネルで回す。`false` = 従来経路へ。
pub fn maybe_gpu_head_gemv_v5(
    uid: u64,
    ty: crate::kquant::GgmlType,
    blocks: &[u8],
    k: usize,
    n: usize,
    x: &[f32],
    out: &mut [f32],
) -> bool {
    use crate::kquant::GgmlType as T;
    set_last_linear_ran_on_gpu(false);
    if !head_fast_enabled() || !cuda_enabled() {
        return false;
    }
    // v5 が読める型だけ (Q8_0 / Q4_K)。
    if !matches!(ty, T::Q8_0 | T::Q4K) {
        return false;
    }
    if x.len() != k || out.len() != n {
        return false;
    }
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        GPU_BROKEN.store(true, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    let w = match weights_for_gguf(uid, ty, blocks, k, n) {
        Ok(w) => w,
        Err(_) => return false,
    };
    // task 427: dn6 経路 (`DENNO_DEC_HEAD_FAST=dn6*`)。形 (Q8_0 / n_in % 64 == 0 /
    // warps <= 8) と PTX が揃ったときだけ。撃てなければ下の v5 へ落ちる。
    // task 504: `DENNO_DEC_HEAD_FAST=dn7` なら dn7 (1 対 = 4 レーン) で撃つ
    // (dn7 は 1 レーン 16 要素 = レジスタが軽い)。形が合わなければ dn6 / v5 へ。
    let dn7_head = if head_dn7_on() {
        dn7_dec_plan(ty, k, n)
    } else {
        None
    };
    if let (Some(dn7p), Some(rb)) = (
        dn7_head,
        gguf_row_bytes(ty, k),
    ) {
        if w.buf.bytes() == n * rb {
            if let Ok(true) = head_gemv_dn6(ctx, &w.buf, x, k, n, out, rb, 1, 1, Some(dn7p)) {
                note_dn7_head_hit();
                set_last_linear_ran_on_gpu(true);
                return true;
            }
        }
    }
    if let (Some((warps, rows)), Some(rb)) = (
        dn6_gemv_plan(ty, k, n, head_dn6_rows()),
        gguf_row_bytes(ty, k),
    ) {
        if w.buf.bytes() == n * rb {
            if let Ok(true) = head_gemv_dn6(ctx, &w.buf, x, k, n, out, rb, warps, rows, None) {
                note_dn6_head_hit();
                set_last_linear_ran_on_gpu(true);
                return true;
            }
        }
    }
    match ctx.moe_gemv_dev_with(ty, &w.buf, x, k, n, out, GemvKernelMode::V5) {
        Ok(true) => {
            set_last_linear_ran_on_gpu(true);
            true
        }
        _ => false,
    }
}


// ===========================================================================
// task 164: decode の expert 層 prologue を 1 回の非同期 H2D にまとめる
// ===========================================================================
//
// 層ごとの prologue は「x (n_in f32) + ポインタ表 (nseg u64 x 3) + segment 表
// (4 u32 x nseg x 2)」の 6 本を `cuMemcpyHtoD` (同期) で送っていた。同期コピーは
// null ストリーム順序なので、**その時点で先行しているカーネルの完了を待つ**。
// decode は 48 層 x 6 本 = 288 回/token になり、実測 (lab t164b の台帳) では
// experts 区間の転送に 1.35 ms/token かかっていた (11.84 MB。28.5 GB/s なら
// 0.42 ms なので、差分はドライバ往復と待ち)。
//
// ここでは 6 本を pinned の 1 スロットに詰め、**非同期 H2D 1 回**にする。
// ホストは待たない (転送は直前の層の GPU 仕事と重なる)。pinned のスロットは
// リング (4 本) で使い回し、書き戻す前に前回のコピーの完了だけを待つ
// (4 層ぶんの余裕があるので通常は即返る)。

/// `DENNO_DEC_PRO_ASYNC`: 層 prologue を 1 回の非同期 H2D にまとめる。
///
/// **既定 on** (lab t164c で目標を満たした)。明示的な偽値 `0` / `off` / `false` /
/// `no` だけ切る。切り戻しはこの env で行う。
pub fn dec_pro_async_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}


/// [`dec_pro_async_from_value`] をプロセスで 1 回だけ読む。
pub fn dec_pro_async_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| dec_pro_async_from_value(std::env::var("DENNO_DEC_PRO_ASYNC").ok().as_deref()))
}


/// prologue の 1 ブロックのレイアウト (バイトオフセット)。全部 8 バイト整列。
pub struct DecStageLayout {
    pub x: usize,
    pub pg: usize,
    pub pu: usize,
    pub pd: usize,
    pub sgu: usize,
    pub sgd: usize,
    /// task 187: 和モードの重み (`nseg` 本の f32)。
    pub w: usize,
    /// task 204: gate+up 融合の**倍の**ポインタ表 (`DENNO_DEC_EXPGU`)。
    /// `gu_nseg == 0` のときは `pg2 == sgu2 == total` (0 バイト)。
    pub pg2: usize,
    /// task 204: gate+up 融合の倍の segment 表。
    pub sgu2: usize,
    pub total: usize,
}


/// [`DecStageLayout`] を作る (純関数・テスト用)。
///
/// task 187: 末尾に和モードの重み (`nseg` 本の f32) のブロックを足した。
/// 使わないときは 0 バイト書くだけなので、従来の経路の転送量は増えない。
/// task 204: `gu_nseg` は gate+up 融合を使うときの segment 数 (`2 * nseg`)。
/// 0 なら倍の表のブロックを作らない (従来と同じレイアウト)。
pub fn dec_stage_layout(
    n_in: usize,
    nseg: usize,
    sgu_len: usize,
    sgd_len: usize,
    w_len: usize,
    gu_nseg: usize,
) -> DecStageLayout {
    let x = 0usize;
    let pg = x + n_in * 4;
    let pu = pg + nseg * 8;
    let pd = pu + nseg * 8;
    let sgu = pd + nseg * 8;
    let sgd = sgu + sgu_len * 4;
    let w = sgd + sgd_len * 4;
    // task 204: ポインタ表 (u64) は 8 バイト整列が要る (和モードの w は 4*nseg)。
    let pg2 = (w + w_len * 4 + 7) & !7usize;
    let sgu2 = pg2 + gu_nseg * 8;
    let total = sgu2 + gu_nseg * 16;
    DecStageLayout {
        x,
        pg,
        pu,
        pd,
        sgu,
        sgd,
        w,
        pg2,
        sgu2,
        total,
    }
}


/// ステージングへ 1 ブロック写す (ホスト側の memcpy。pinned なので速い)。
///
/// # Safety
/// `src` は `n` バイト読めること。`off + n <= s.len()` を呼び側が保証する。
pub unsafe fn dec_stage_put(s: &mut [u8], off: usize, src: *const u8, n: usize) {
    s[off..off + n].copy_from_slice(std::slice::from_raw_parts(src, n));
}


pub(crate) struct DecStage {
    pin: Vec<PinMem>,
    ev: Vec<CopyEvent>,
    /// 各スロットに「対応するコピーをまだ記録していない」場合は false。
    used: Vec<bool>,
    next: usize,
    dev: Option<DevBuffer>,
    bytes: usize,
}


/// 次のスロットを用意して `(デバイスのベース, ホストのスロット先頭)` を返す。
///
/// デバイス側は grow-only の 1 本、ホスト側は pinned のリング。書き込む前に
/// **前回そのスロットから出したコピーの完了**を待つ (待ちは通常ゼロ)。
pub fn dec_stage_slot(total: usize) -> Result<(u64, *mut u8)> {
    let l = lib()?;
    let (Some(create), Some(esync)) = (l.event_create, l.event_sync) else {
        bail!("event_create / event_sync が無い (dec stage)");
    };
    let total = total.max(1);
    DEC_STAGE.with(|c| -> Result<(u64, *mut u8)> {
        let mut c = c.borrow_mut();
        if c.is_none() {
            *c = Some(DecStage {
                pin: Vec::new(),
                ev: Vec::new(),
                used: Vec::new(),
                next: 0,
                dev: None,
                bytes: 0,
            });
        }
        let st = c.as_mut().expect("直前に作った");
        // デバイス側 (grow-only)。
        if st.bytes < total || st.dev.is_none() {
            let want = moe_scratch_alloc_bytes(st.bytes, total).unwrap_or(total);
            st.dev = Some(DevBuffer::alloc_tagged(want, crate::mem_audit::TAG_GDN_SC)?);
            st.bytes = want;
        }
        // ホスト側 (pinned リング)。ブロックは 10〜20 KB なので 1 MiB を下限に
        // して作り、形が増えても作り直さない (作り直しは in-flight のコピーを
        // 壊す危険があるため、下限を大きく取って経路を踏ませない)。
        let cap = total.max(1 << 20);
        while st.pin.len() < DEC_STAGE_SLOTS {
            let mut ev: CUevent = std::ptr::null_mut();
            if unsafe { create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                bail!("cuEventCreate (dec stage) が失敗");
            }
            st.pin.push(PinMem::alloc(cap)?);
            st.ev.push(CopyEvent(ev));
            st.used.push(false);
        }
        for i in 0..st.pin.len() {
            if st.pin[i].bytes < total {
                // 形が大きくなった層 (nseg 増) に備えて作り直す。未コピーの
                // スロットだけなので安全。
                st.pin[i] = PinMem::alloc(total)?;
                st.used[i] = false;
            }
        }
        let i = st.next % DEC_STAGE_SLOTS;
        st.next = st.next.wrapping_add(1);
        if st.used[i] && unsafe { esync(st.ev[i].0) } != CUDA_SUCCESS {
            bail!("cuEventSynchronize (dec stage) が失敗");
        }
        Ok((st.dev.as_ref().expect("確保済み").ptr, st.pin[i].ptr as *mut u8))
    })
}


/// 直前に [`dec_stage_slot`] が返したスロットから、非同期 H2D を 1 回出す。
pub fn dec_stage_issue(total: usize) -> Result<()> {
    let total = total.max(1);
    let l = lib()?;
    let (Some(rec), Some(htod)) = (l.event_record, Some(l.memcpy_htod_async)) else {
        bail!("event_record / memcpy_htod_async が無い (dec stage)");
    };
    DEC_STAGE.with(|c| -> Result<()> {
        let mut c = c.borrow_mut();
        let st = c.as_mut().ok_or_else(|| anyhow!("dec stage 未初期化"))?;
        let i = (st.next + DEC_STAGE_SLOTS - 1) % DEC_STAGE_SLOTS;
        let dev = st.dev.as_ref().expect("確保済み").ptr;
        let src = st.pin[i].ptr;
        check(
            unsafe { htod(dev, src, total, std::ptr::null_mut()) },
            "cuMemcpyHtoDAsync (dec stage)",
        )?;
        if unsafe { rec(st.ev[i].0, std::ptr::null_mut()) } != CUDA_SUCCESS {
            bail!("cuEventRecord (dec stage) が失敗");
        }
        st.used[i] = true;
        Ok(())
    })
}


thread_local! {
    pub(crate) static EXP_D2H: std::cell::RefCell<Option<ExpD2h>> =
        const { std::cell::RefCell::new(None) };
    /// 直前に [`exp_d2h_issue`] がコピーを出したか (±[`exp_d2h_wait`] で落とす)。
    pub(crate) static EXP_D2H_INFLIGHT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}


thread_local! {
    pub(crate) static MISS_UP: std::cell::RefCell<MissUpScratch> = const {
        std::cell::RefCell::new(MissUpScratch {
            dg: None,
            du: None,
            dd: None,
            pin_g: None,
            pin_u: None,
            pin_d: None,
            pg: None,
            pu: None,
            pd: None,
            sgu: None,
            sgd: None,
            stream: None,
            ev_ready: None,
        })
    };
}


/// `act = silu(g) * u` を `n` 要素だけ積む。
fn iq_launch_silu(ctx: &CudaCtx, f: CUfunction, pg: u64, pu: u64, pa: u64, n: usize) -> Result<()> {
    const THREADS: usize = 256;
    if n == 0 {
        return Ok(());
    }
    let blocks = n.div_ceil(THREADS) as u32;
    unsafe {
        let (mut pg, mut pu, mut pa) = (pg, pu, pa);
        let mut pn = n as c_int;
        let mut args = [
            &mut pg as *mut _ as *mut c_void,
            &mut pu as *mut _ as *mut c_void,
            &mut pa as *mut _ as *mut c_void,
            &mut pn as *mut _ as *mut c_void,
        ];
        ctx.launch_async(
            f,
            (blocks.max(1), 1, 1),
            (THREADS as u32, 1, 1),
            0,
            &mut args,
        )
    }
}


/// ミス expert の重みを GPU へ上げて、ヒットと同じカーネルで計算する (task 158)。
///
/// 戻り値 `Ok(true)`: `out_h` に `n_hit + n_miss` 本ぶんの寄与を書いた
/// (並びは [ヒット..., ミス...])。`Ok(false)`: GPU でやらなかった =
/// 呼び出し側は従来経路 (CPU でミス + 重ね打ち) へ落ちること (fail closed)。
///
/// 発行順 (すべて null ストリーム。同期は最後の 1 回だけ):
///   1. x / ポインタ表 / segment 表の H2D (同期コピー。数百バイト)
///   2. ヒットの gate+up → silu → down (nseg = `n_hit`)
///   3. ホスト: ミスの重みを mmap から pinned へ写す (ヒットの GPU 計算と重なる)
///   4. コピーストリーム: pinned → デバイス scratch (3 コピー) → 完了イベント
///   5. null ストリームがイベントを待つ → ミスの gate+up → silu → down
///   6. h 全体の D2H → 1 回だけ同期
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_layer_miss_up(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    pg: &[u64],
    pu: &[u64],
    pd: &[u64],
    ids_hit: &[u16],
    miss: &[MissExpertBytes<'_>],
    ids_miss: &[u16],
    x: &[f32],
    n_in: usize,
    ff: usize,
    e: usize,
    out_h: &mut Vec<f32>,
) -> Result<bool> {
    let n_hit = pg.len();
    let n_miss = miss.len();
    if n_miss == 0
        || n_miss > crate::model::exp_miss_up::MAX_MISS
        || n_hit == 0
        || pu.len() != n_hit
        || pd.len() != n_hit
        || ids_hit.len() != n_hit
        || ids_miss.len() != n_miss
        || n_in == 0
        || ff == 0
        || e == 0
        || x.len() < n_in
    {
        return Ok(false);
    }
    if !moe_gemv_iq_ready(ty_g, n_in)
        || !moe_gemv_iq_ready(ty_u, n_in)
        || !moe_gemv_iq_ready(ty_d, ff)
    {
        return Ok(false);
    }
    let (Some(slot_g), Some(slot_u), Some(slot_d)) = (
        crate::model::exp_iq::slot(ty_g),
        crate::model::exp_iq::slot(ty_u),
        crate::model::exp_iq::slot(ty_d),
    ) else {
        return Ok(false);
    };
    let (Some(plan_gu_hit), Some(plan_d_hit), Some(plan_gu_miss), Some(plan_d_miss)) = (
        crate::model::exp_iq::launch_plan(n_hit, ff),
        crate::model::exp_iq::launch_plan(n_hit, e),
        crate::model::exp_iq::launch_plan(n_miss, ff),
        crate::model::exp_iq::launch_plan(n_miss, e),
    ) else {
        return Ok(false);
    };
    let Some(ctx) = shared_ctx() else {
        return Ok(false);
    };
    let Some(kernels) = exp_iq_kernels_decode() else {
        return Ok(false);
    };
    let Ok(l) = lib() else {
        return Ok(false);
    };
    let (
        Some(stream_create),
        Some(event_create),
        Some(event_record),
        Some(stream_wait_event),
        Some(memcpy_htod_async),
        Some(memcpy_dtoh_async),
    ) = (
        l.stream_create,
        l.event_create,
        l.event_record,
        l.stream_wait_event,
        Some(l.memcpy_htod_async),
        Some(l.memcpy_dtoh_async),
    )
    else {
        return Ok(false);
    };
    ctx.make_current()?;

    let (Some(rb_g), Some(rb_u), Some(rb_d)) = (
        crate::model::exp_iq::row_bytes(ty_g, n_in),
        crate::model::exp_iq::row_bytes(ty_u, n_in),
        crate::model::exp_iq::row_bytes(ty_d, ff),
    ) else {
        return Ok(false);
    };
    let (bytes_g, bytes_u, bytes_d) = (rb_g * ff, rb_u * ff, rb_d * e);
    // 重みの長さが型・形と合わないミスが 1 本でもあれば CPU へ (fail closed)。
    if miss
        .iter()
        .any(|m| m.gate.len() != bytes_g || m.up.len() != bytes_u || m.down.len() != bytes_d)
    {
        return Ok(false);
    }

    let n_all = n_hit + n_miss;
    let ng_hit = n_hit * ff;
    let ng_miss = n_miss * ff;
    let ng_all = n_all * ff;
    let nh_all = n_all * e;
    let seg_gu_hit = crate::model::expert_batch::seg_table_shared_x(ids_hit, ff);
    let seg_d_hit = crate::model::expert_batch::seg_table_per_act(ids_hit, ff, e);
    let seg_gu_miss = crate::model::expert_batch::seg_table_shared_x(ids_miss, ff);
    let seg_d_miss = crate::model::expert_batch::seg_table_per_act(ids_miss, ff, e);

    let mut copied = 0u64;
    let mut copy_ns = 0u64;
    let t_layer = std::time::Instant::now();
    IQ_SCRATCH.with(|cell| -> Result<()> {
        let mut sc = cell.borrow_mut();
        MoeScratch::ensure_tagged(&mut sc.x, n_in * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.g, ng_all * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.u, ng_all * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.act, ng_all * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.h, nh_all * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.pg, n_hit * 8, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.pu, n_hit * 8, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.pd, n_hit * 8, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.sgu, seg_gu_hit.len() * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.sgd, seg_d_hit.len() * 4, crate::mem_audit::TAG_GDN_SC)?;
        Scratch::ensure_pin(&mut sc.hpin, nh_all * 4)?;
        {
            let xb = sc.x.as_ref().expect("確保済み");
            xb.copy_from(&x[..n_in])?;
            dbg_htod(n_in * 4);
            sc.pg.as_ref().expect("確保済み").copy_from(pg)?;
            sc.pu.as_ref().expect("確保済み").copy_from(pu)?;
            sc.pd.as_ref().expect("確保済み").copy_from(pd)?;
            sc.sgu.as_ref().expect("確保済み").copy_from(&seg_gu_hit)?;
            sc.sgd.as_ref().expect("確保済み").copy_from(&seg_d_hit)?;
            dbg_htod(n_hit * 8 * 3 + (seg_gu_hit.len() + seg_d_hit.len()) * 4);
        }

        // 1.5 ミスの受け皿と表を**起動前**に用意する。`copy_from` は同期コピー
        //     (`cuMemcpyHtoD`) なので、ヒットのカーネルを積んだ後に置くと null
        //     ストリームの先行カーネルを待ってしまい、重ね打ちが消える。
        MISS_UP.with(|mcell| -> Result<()> {
            let mut ms = mcell.borrow_mut();
            MoeScratch::ensure_tagged(
                &mut ms.dg,
                n_miss * bytes_g,
                crate::mem_audit::TAG_GDN_SC,
            )?;
            MoeScratch::ensure_tagged(
                &mut ms.du,
                n_miss * bytes_u,
                crate::mem_audit::TAG_GDN_SC,
            )?;
            MoeScratch::ensure_tagged(
                &mut ms.dd,
                n_miss * bytes_d,
                crate::mem_audit::TAG_GDN_SC,
            )?;
            Scratch::ensure_pin(&mut ms.pin_g, n_miss * bytes_g)?;
            Scratch::ensure_pin(&mut ms.pin_u, n_miss * bytes_u)?;
            Scratch::ensure_pin(&mut ms.pin_d, n_miss * bytes_d)?;
            MoeScratch::ensure_tagged(&mut ms.pg, n_miss * 8, crate::mem_audit::TAG_GDN_SC)?;
            MoeScratch::ensure_tagged(&mut ms.pu, n_miss * 8, crate::mem_audit::TAG_GDN_SC)?;
            MoeScratch::ensure_tagged(&mut ms.pd, n_miss * 8, crate::mem_audit::TAG_GDN_SC)?;
            MoeScratch::ensure_tagged(
                &mut ms.sgu,
                seg_gu_miss.len() * 4,
                crate::mem_audit::TAG_GDN_SC,
            )?;
            MoeScratch::ensure_tagged(
                &mut ms.sgd,
                seg_d_miss.len() * 4,
                crate::mem_audit::TAG_GDN_SC,
            )?;
            if ms.stream.is_none() {
                const CU_STREAM_NON_BLOCKING: u32 = 0x01;
                let mut st: CUstream = std::ptr::null_mut();
                if unsafe { stream_create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
                    bail!("cuStreamCreate (dec miss) が失敗");
                }
                let mut ev: CUevent = std::ptr::null_mut();
                if unsafe { event_create(&mut ev, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                    unsafe {
                        if let Some(f) = l.stream_destroy {
                            let _ = f(st);
                        }
                    }
                    bail!("cuEventCreate (dec miss) が失敗");
                }
                ms.stream = Some(CopyStream(st));
                ms.ev_ready = Some(CopyEvent(ev));
            }
            // ポインタ表 (scratch の先頭アドレス)。segment 表は scratch の並び。
            let (dg_p, du_p, dd_p) = (
                ms.dg.as_ref().expect("確保済み").ptr,
                ms.du.as_ref().expect("確保済み").ptr,
                ms.dd.as_ref().expect("確保済み").ptr,
            );
            let tpg: Vec<u64> = (0..n_miss)
                .map(|j| dg_p + (j * bytes_g) as u64)
                .collect();
            let tpu: Vec<u64> = (0..n_miss)
                .map(|j| du_p + (j * bytes_u) as u64)
                .collect();
            let tpd: Vec<u64> = (0..n_miss)
                .map(|j| dd_p + (j * bytes_d) as u64)
                .collect();
            ms.pg.as_ref().expect("確保済み").copy_from(&tpg)?;
            ms.pu.as_ref().expect("確保済み").copy_from(&tpu)?;
            ms.pd.as_ref().expect("確保済み").copy_from(&tpd)?;
            ms.sgu.as_ref().expect("確保済み").copy_from(&seg_gu_miss)?;
            ms.sgd.as_ref().expect("確保済み").copy_from(&seg_d_miss)?;
            dbg_htod(n_miss * 8 * 3 + (seg_gu_miss.len() + seg_d_miss.len()) * 4);
            Ok(())
        })?;

        let xb = sc.x.as_ref().expect("確保済み");
        let g = sc.g.as_ref().expect("確保済み");
        let u = sc.u.as_ref().expect("確保済み");
        let act = sc.act.as_ref().expect("確保済み");
        let h = sc.h.as_ref().expect("確保済み");
        // 1. ヒット (nseg = n_hit)。出力の先頭 n_hit*e。
        iq_launch_gemv(
            ctx,
            kernels.fns[slot_g],
            plan_gu_hit,
            sc.pg.as_ref().expect("確保済み").ptr,
            xb.ptr,
            g.ptr,
            sc.sgu.as_ref().expect("確保済み").ptr,
            n_hit,
            n_in,
            ff,
            rb_g,
        )?;
        iq_launch_gemv(
            ctx,
            kernels.fns[slot_u],
            plan_gu_hit,
            sc.pu.as_ref().expect("確保済み").ptr,
            xb.ptr,
            u.ptr,
            sc.sgu.as_ref().expect("確保済み").ptr,
            n_hit,
            n_in,
            ff,
            rb_u,
        )?;
        iq_launch_silu(ctx, kernels.silu, g.ptr, u.ptr, act.ptr, ng_hit)?;
        iq_launch_gemv(
            ctx,
            kernels.fns[slot_d],
            plan_d_hit,
            sc.pd.as_ref().expect("確保済み").ptr,
            act.ptr,
            h.ptr,
            sc.sgd.as_ref().expect("確保済み").ptr,
            n_hit,
            ff,
            e,
            rb_d,
        )?;

        // 2. ミス: pinned へ写してから scratch へ非同期 H2D (ヒットの裏で走る)。
        //    受け皿と表は 1.5 で用意済み (ここでは触らない)。
        MISS_UP.with(|mcell| -> Result<()> {
            let mut ms = mcell.borrow_mut();
            let (dg_p, du_p, dd_p) = (
                ms.dg.as_ref().expect("確保済み").ptr,
                ms.du.as_ref().expect("確保済み").ptr,
                ms.dd.as_ref().expect("確保済み").ptr,
            );

            // ホスト写し (mmap → pinned)。ここがヒットの GPU 計算と重なる区間。
            // `ms` は RefMut なので 3 本を同時に借りない (1 本ずつ写す)。
            let t0 = std::time::Instant::now();
            for (j, m) in miss.iter().enumerate() {
                ms.pin_g.as_mut().expect("確保済み").as_mut_slice::<u8>()
                    [j * bytes_g..(j + 1) * bytes_g]
                    .copy_from_slice(m.gate);
            }
            for (j, m) in miss.iter().enumerate() {
                ms.pin_u.as_mut().expect("確保済み").as_mut_slice::<u8>()
                    [j * bytes_u..(j + 1) * bytes_u]
                    .copy_from_slice(m.up);
            }
            for (j, m) in miss.iter().enumerate() {
                ms.pin_d.as_mut().expect("確保済み").as_mut_slice::<u8>()
                    [j * bytes_d..(j + 1) * bytes_d]
                    .copy_from_slice(m.down);
            }
            copy_ns = t0.elapsed().as_nanos() as u64;
            copied = (n_miss * (bytes_g + bytes_u + bytes_d)) as u64;

            let stream = ms.stream.as_ref().expect("作成済み").0;
            let ev = ms.ev_ready.as_ref().expect("作成済み").0;
            for (dst, src, bytes) in [
                (
                    dg_p,
                    ms.pin_g.as_ref().expect("確保済み").ptr,
                    n_miss * bytes_g,
                ),
                (
                    du_p,
                    ms.pin_u.as_ref().expect("確保済み").ptr,
                    n_miss * bytes_u,
                ),
                (
                    dd_p,
                    ms.pin_d.as_ref().expect("確保済み").ptr,
                    n_miss * bytes_d,
                ),
            ] {
                if unsafe { memcpy_htod_async(dst, src as *const c_void, bytes, stream) }
                    != CUDA_SUCCESS
                {
                    bail!("cuMemcpyHtoDAsync (dec miss) が失敗");
                }
            }
            if unsafe { event_record(ev, stream) } != CUDA_SUCCESS {
                bail!("cuEventRecord (dec miss) が失敗");
            }
            if unsafe { stream_wait_event(std::ptr::null_mut(), ev, 0) } != CUDA_SUCCESS {
                bail!("cuStreamWaitEvent (dec miss) が失敗");
            }
            // 3. ミス (nseg = n_miss)。基底はヒットの後ろ。
            iq_launch_gemv(
                ctx,
                kernels.fns[slot_g],
                plan_gu_miss,
                ms.pg.as_ref().expect("確保済み").ptr,
                xb.ptr,
                g.ptr + (ng_hit * 4) as u64,
                ms.sgu.as_ref().expect("確保済み").ptr,
                n_miss,
                n_in,
                ff,
                rb_g,
            )?;
            iq_launch_gemv(
                ctx,
                kernels.fns[slot_u],
                plan_gu_miss,
                ms.pu.as_ref().expect("確保済み").ptr,
                xb.ptr,
                u.ptr + (ng_hit * 4) as u64,
                ms.sgu.as_ref().expect("確保済み").ptr,
                n_miss,
                n_in,
                ff,
                rb_u,
            )?;
            iq_launch_silu(
                ctx,
                kernels.silu,
                g.ptr + (ng_hit * 4) as u64,
                u.ptr + (ng_hit * 4) as u64,
                act.ptr + (ng_hit * 4) as u64,
                ng_miss,
            )?;
            iq_launch_gemv(
                ctx,
                kernels.fns[slot_d],
                plan_d_miss,
                ms.pd.as_ref().expect("確保済み").ptr,
                act.ptr + (ng_hit * 4) as u64,
                h.ptr + (n_hit * e * 4) as u64,
                ms.sgd.as_ref().expect("確保済み").ptr,
                n_miss,
                ff,
                e,
                rb_d,
            )?;
            Ok(())
        })?;

        // 4. h 全体 (ヒット + ミス) の D2H → 1 回だけ同期。
        let h_ptr = h.ptr;
        let pin = sc.hpin.as_mut().expect("確保済み");
        unsafe {
            check(
                (memcpy_dtoh_async)(pin.ptr, h_ptr, nh_all * 4, std::ptr::null_mut()),
                "cuMemcpyDtoHAsync (iq miss up)",
            )?;
        }
        crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
        sync_default_or_ctx(ctx, dec_overlap_enabled())?;
        let pin = sc.hpin.as_ref().expect("確保済み");
        out_h.resize(nh_all, 0.0);
        out_h.copy_from_slice(&pin.as_slice::<f32>()[..nh_all]);
        dbg_d2h(nh_all * 4);
        Ok(())
    })?;
    crate::model::exp_miss_up::note_layer(n_miss as u32, copied);
    if copy_ns > 0 {
        crate::model::exp_miss_up::note_copy_ns(copy_ns);
    }
    crate::model::exp_miss_up::note_gpu_ns(t_layer.elapsed().as_nanos() as u64);
    Ok(true)
}


// ===========================================================================
/// task 158: ミス重みの H2D 実効レートを測る (`DENNO_DEC_MISS_PROBE=1`)。
///
/// decode のミスは「pinned の bounce buffer → デバイス scratch」の 1 本道なので、
/// その実効レート (GB/s) を 1 回だけ測って 1 行出す。ホットパスには入らない。
/// 2 MB を `REPS` 回、タイミング付きイベントで挟んで測る (warmup 2 回は除く)。
///
/// task 164: 新しい機材 (Gen5 x16) では「どの経路がいくらか」で損益が変わるので、
/// 4 本まとめて測る:
///
/// - `gbs` — pinned → device の非同期 H2D (イベント計測)
/// - `pg_gbs` — pageable (Vec) → device の H2D (現行 `iq_warm_misses` の経路)
/// - `copy_gbs` — `src` (mmap の実スライス) → pinned のホスト写し (bounce の代価)
/// - `direct_gbs` — `src` → device を直接 (bounce 無しの pageable H2D)
pub fn dec_miss_probe_once(mb: usize, src: Option<&[u8]>) {
    use std::sync::OnceLock;
    static DONE: OnceLock<()> = OnceLock::new();
    DONE.get_or_init(|| {
        if let Some(line) = dec_miss_probe_line(mb, src) {
            eprintln!("{line}");
        }
    });
}


/// [`dec_miss_probe_once`] の本体 (失敗したら `None`)。
pub fn dec_miss_probe_line(mb: usize, src: Option<&[u8]>) -> Option<String> {
    const REPS: usize = 8;
    let bytes = mb.max(1) << 20;
    let l = lib().ok()?;
    let ctx = shared_ctx()?;
    ctx.make_current().ok()?;
    let mut pin = PinMem::alloc(bytes).ok()?;
    let d = DevBuffer::alloc_tagged(bytes, crate::mem_audit::TAG_GDN_SC).ok()?;
    let (create, record, esync, elapsed) = (
        l.event_create?,
        l.event_record?,
        l.event_sync?,
        l.event_elapsed_time?,
    );
    let mut ev0: CUevent = std::ptr::null_mut();
    let mut ev1: CUevent = std::ptr::null_mut();
    unsafe {
        if create(&mut ev0, 0) != CUDA_SUCCESS || create(&mut ev1, 0) != CUDA_SUCCESS {
            return None;
        }
    }
    // warmup (pinned の実体化 + ドライバの初回準備) は測らない。
    unsafe {
        for _ in 0..2 {
            let _ = (l.memcpy_htod_async)(d.ptr(), pin.ptr, bytes, std::ptr::null_mut());
        }
        let _ = ctx.sync();
    }
    let t = std::time::Instant::now();
    unsafe {
        if record(ev0, std::ptr::null_mut()) != CUDA_SUCCESS {
            return None;
        }
        for _ in 0..REPS {
            if (l.memcpy_htod_async)(d.ptr(), pin.ptr, bytes, std::ptr::null_mut()) != CUDA_SUCCESS
            {
                return None;
            }
        }
        if record(ev1, std::ptr::null_mut()) != CUDA_SUCCESS || esync(ev1) != CUDA_SUCCESS {
            return None;
        }
    }
    let wall_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut ms: f32 = 0.0;
    unsafe {
        if elapsed(&mut ms, ev0, ev1) != CUDA_SUCCESS {
            return None;
        }
        if let Some(destroy) = l.event_destroy {
            let _ = destroy(ev0);
            let _ = destroy(ev1);
        }
    }
    let gbs = (REPS * bytes) as f64 / (ms as f64 * 1e6);

    // pageable (Vec) → device。現行の上げ直しと同じ経路 (ドライバが内部で
    // staging する)。呼び出しスレッドは終わるまで戻らない。
    let pageable = vec![0u8; bytes];
    let mut pg_gbs = 0.0f64;
    {
        let t = std::time::Instant::now();
        let mut ok = true;
        unsafe {
            for _ in 0..REPS {
                if (l.memcpy_htod_async)(d.ptr(), pageable.as_ptr() as *const c_void, bytes, std::ptr::null_mut())
                    != CUDA_SUCCESS
                {
                    ok = false;
                    break;
                }
            }
            let _ = ctx.sync();
        }
        let s = t.elapsed().as_secs_f64();
        if ok && s > 0.0 {
            pg_gbs = (REPS * bytes) as f64 / (s * 1e9);
        }
    }

    // mmap の実スライス → pinned (bounce) と → device 直接。
    // 実スライスは 2 MB より小さいことがある (gate 1 枚) ので、長さはスライスに合わせる。
    let (mut copy_gbs, mut direct_gbs) = (0.0f64, 0.0f64);
    if let Some(src) = src.filter(|s| !s.is_empty()) {
        let n = src.len().min(bytes);
        let src = &src[..n];
        let bytes = n;
        copy_gbs = host_copy_gbs(pin.as_mut_slice::<u8>(), src, REPS);
        let t = std::time::Instant::now();
        let mut ok = true;
        unsafe {
            for _ in 0..REPS {
                if (l.memcpy_htod_async)(d.ptr(), src.as_ptr() as *const c_void, bytes, std::ptr::null_mut())
                    != CUDA_SUCCESS
                {
                    ok = false;
                    break;
                }
            }
            let _ = ctx.sync();
        }
        let s = t.elapsed().as_secs_f64();
        if ok && s > 0.0 {
            direct_gbs = (REPS * bytes) as f64 / (s * 1e9);
        }
    }
    Some(format!(
        "[dec-miss-probe] mb={mb} reps={REPS} h2d_ms={ms:.3} gbs={gbs:.2} pg_gbs={pg_gbs:.2} \
         copy_gbs={copy_gbs:.2} direct_gbs={direct_gbs:.2} wall_ms={wall_ms:.3}"
    ))
}


/// この型・この k を m 行カーネルで回せるか (`DENNO_EXPERT_IQ_GPU_PF` と
/// `DENNO_EXPERT_IQ_GPU` の両方が on のときだけ真)。
pub fn moe_gemv_iq_pf_ready(ty: crate::kquant::GgmlType, k: usize) -> bool {
    if !cuda_enabled() || !crate::model::exp_pf::pf_enabled() {
        return false;
    }
    if !crate::model::exp_pf::supported(ty) || crate::model::exp_pf::row_bytes(ty, k).is_none() {
        return false;
    }
    let Some(slot) = crate::model::exp_pf::slot(ty) else {
        return false;
    };
    // task 309: K 系は専用のゲート (`DENNO_PF_EXP_K`) が要る。既定 off。
    if crate::model::exp_pf::is_k_slot(slot) && !crate::model::exp_pf::k_enabled() {
        return false;
    }
    exp_pf_kernels().is_some_and(|kernels| !kernels.fns[slot].is_null())
}


/// 同時に積める本数の上限 (shared expert は 2 本。余裕を見て 4)。
const SHARED_OVERLAP_MAX_JOBS: usize = 4;


/// 重ね打ち経路の計数 (プロセス通算)。テストと `DENNO_SHARED_OVERLAP` の
/// 発火確認に使う。緑 = 経路が発動した証拠ではない、の裏取り。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SharedOverlapStats {
    /// `shared_gemv_begin` が呼ばれた回数
    pub begin: usize,
    /// 実際に GPU へ積めた回数 (fake 経路も含む)
    pub taken: usize,
    /// join できた回数
    pub joined: usize,
    /// join に失敗した回数 (呼び出し側は CPU へ落ちる)
    pub failed: usize,
    /// join せずに捨てられた回数 (Drop が drain した)
    pub aborted: usize,
}


/// 計数を読む。
pub fn shared_overlap_stats() -> SharedOverlapStats {
    use std::sync::atomic::Ordering::Relaxed;
    SharedOverlapStats {
        begin: OVERLAP_BEGIN.load(Relaxed),
        taken: OVERLAP_TAKEN.load(Relaxed),
        joined: OVERLAP_JOINED.load(Relaxed),
        failed: OVERLAP_FAILED.load(Relaxed),
        aborted: OVERLAP_ABORTED.load(Relaxed),
    }
}


/// 重ね打ち用の thread_local 状態。decode は 1 スレッドなので競合しない。
pub(crate) struct SharedOverlapState {
    stream: Option<CopyStream>,
    /// 重みの H2D (null ストリーム) と専用ストリームの順序を取るため
    ev_ready: Option<CopyEvent>,
    /// 積んだ 1 群の完了
    ev_done: Option<CopyEvent>,
    dx: Option<DevBuffer>,
    pin_x: Option<PinMem>,
    /// task 187: gate / up の出力と silu*up をデバイスに残すための作業域。
    dg: Option<DevBuffer>,
    du: Option<DevBuffer>,
    dact: Option<DevBuffer>,
    slots: Vec<SharedSlotBuf>,
    /// in-flight の間だけ重みを pin する
    pins: Vec<GgufWRef>,
    in_flight: usize,
    /// 失敗して以後この経路を試さない (壊れた状態でバッファを再利用しない)
    pub(crate) broken: bool,
    #[cfg(test)]
    fake: bool,
}


impl SharedOverlapState {
    pub(crate) fn new() -> Self {
        Self {
            stream: None,
            ev_ready: None,
            ev_done: None,
            dx: None,
            pin_x: None,
            dg: None,
            du: None,
            dact: None,
            slots: Vec::new(),
            pins: Vec::new(),
            in_flight: 0,
            broken: false,
            #[cfg(test)]
            fake: false,
        }
    }
}


thread_local! {
    pub(crate) static SHARED_OVERLAP: std::cell::RefCell<SharedOverlapState> =
        std::cell::RefCell::new(SharedOverlapState::new());
}


#[cfg(test)]
static SHARED_OVERLAP_FAKE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// テスト用: 本物の GPU を使わずに「積めた」ことにする (`Drop` で戻す)。
///
/// このコンテナには GPU が無いので、join 経路 (バッファの写し先・順序) を
/// 検証するにはこれが要る。値は CPU の `expert_gemv` で作るので、ゲート off の
/// 経路と**ビット一致**するはず (同じ関数・同じ順)。
#[cfg(test)]
pub struct SharedOverlapFakeGuard;


#[cfg(test)]
impl SharedOverlapFakeGuard {
    pub fn new() -> Self {
        SHARED_OVERLAP_FAKE.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}


#[cfg(test)]
impl Drop for SharedOverlapFakeGuard {
    fn drop(&mut self) {
        SHARED_OVERLAP_FAKE.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}


#[cfg(test)]
static SHARED_OVERLAP_FAIL_FINISH: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);


/// テスト用: `shared_gemv_finish` を必ず失敗させる (`Drop` で戻す)。
///
/// join に失敗したときに呼び出し側が従来の同期経路へ落ちること (値が変わらない
/// こと・abort が二重に走らないこと) を GPU 無しで確かめるためのもの。
#[cfg(test)]
pub struct SharedOverlapFailGuard;


#[cfg(test)]
impl SharedOverlapFailGuard {
    pub fn new() -> Self {
        SHARED_OVERLAP_FAIL_FINISH.store(true, std::sync::atomic::Ordering::Relaxed);
        Self
    }
}


#[cfg(test)]
impl Drop for SharedOverlapFailGuard {
    fn drop(&mut self) {
        SHARED_OVERLAP_FAIL_FINISH.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}


/// in-flight の操作を drain する (join し忘れ・join 失敗の後始末)。
///
/// 戻り値 `false` は「完了を確認できなかった」。呼び出し側は `broken` を立てる。
pub(crate) fn shared_overlap_drain(s: &mut SharedOverlapState) -> bool {
    #[cfg(test)]
    if s.fake {
        s.in_flight = 0;
        s.fake = false;
        s.pins.clear();
        return true;
    }
    let pending = s.in_flight != 0 || !s.pins.is_empty();
    s.in_flight = 0;
    s.pins.clear();
    if !pending {
        return true;
    }
    let Some(l) = lib().ok() else {
        return false;
    };
    let Some(event_sync) = l.event_sync else {
        return false;
    };
    let ctx_sync = l.ctx_sync;
    let Some(ctx) = shared_ctx() else {
        return false;
    };
    if ctx.make_current().is_err() {
        return false;
    }
    let ev = s.ev_done.as_ref().map(|e| e.0);
    let mut ok = match ev {
        Some(ev) => (unsafe { event_sync(ev) }) == CUDA_SUCCESS,
        None => true,
    };
    if !ok {
        // ストリームの完了が取れない: デバイス全体で drain する (取りこぼし防止)。
        ok = unsafe { ctx_sync() } == CUDA_SUCCESS;
    }
    ok
}


/// 重ね打ち用のストリームとイベントを用意する (初回だけ)。blocking ストリームだと
/// legacy (null) ストリームと暗黙に同期し、expert の GEMV と 1 バイトも重ならない。
fn shared_overlap_ensure_stream(s: &mut SharedOverlapState, l: &CudaLib) -> Result<()> {
    if s.stream.is_some() {
        return Ok(());
    }
    const CU_STREAM_NON_BLOCKING: u32 = 0x01;
    const CU_EVENT_DISABLE_TIMING: u32 = 0x02;
    let (Some(stream_create), Some(event_create)) = (l.stream_create, l.event_create) else {
        bail!("cuStreamCreate / cuEventCreate が無い");
    };
    let mut st: CUstream = std::ptr::null_mut();
    if unsafe { stream_create(&mut st, CU_STREAM_NON_BLOCKING) } != CUDA_SUCCESS {
        bail!("cuStreamCreate が失敗");
    }
    let mut ev_ready: CUevent = std::ptr::null_mut();
    if unsafe { event_create(&mut ev_ready, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
        unsafe {
            if let Some(f) = l.stream_destroy {
                let _ = f(st);
            }
        }
        bail!("cuEventCreate (ready) が失敗");
    }
    let mut ev_done: CUevent = std::ptr::null_mut();
    if unsafe { event_create(&mut ev_done, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
        unsafe {
            if let Some(f) = l.stream_destroy {
                let _ = f(st);
            }
            if let Some(f) = l.event_destroy {
                let _ = f(ev_ready);
            }
        }
        bail!("cuEventCreate (done) が失敗");
    }
    s.stream = Some(CopyStream(st));
    s.ev_ready = Some(CopyEvent(ev_ready));
    s.ev_done = Some(CopyEvent(ev_done));
    Ok(())
}


/// 依頼の検査と発行 (GPU 操作の前に全部そろえる = all-or-nothing)。
#[allow(clippy::too_many_arguments)]
fn shared_overlap_enqueue(
    s: &mut SharedOverlapState,
    ctx: &CudaCtx,
    l: &CudaLib,
    x: &[f32],
    jobs: &[SharedGemvJob<'_>],
    plans: &[MoeGemvPlan],
    pins: &[GgufWRef],
    uploaded: bool,
) -> Result<()> {
    if s.broken || s.in_flight != 0 {
        bail!(
            "重ね打ちは使わない (broken={}, in_flight={})",
            s.broken,
            s.in_flight
        );
    }
    // 1) 専用の非ブロッキングストリーム + イベント (初回だけ)。
    shared_overlap_ensure_stream(s, l)?;
    // 2) grow-only の入出力 (MOE_SCRATCH とは別)。
    let bytes_x = x.len() * 4;
    Scratch::ensure(&mut s.dx, bytes_x)?;
    Scratch::ensure_pin(&mut s.pin_x, bytes_x)?;
    while s.slots.len() < jobs.len() {
        s.slots.push(SharedSlotBuf::new());
    }
    for (i, j) in jobs.iter().enumerate() {
        Scratch::ensure(&mut s.slots[i].dout, j.n * 4)?;
        Scratch::ensure_pin(&mut s.slots[i].pin_o, j.n * 4)?;
        s.slots[i].n = j.n;
    }

    let stream = s.stream.as_ref().expect("作成済み").0;
    let ev_ready = s.ev_ready.as_ref().expect("作成済み").0;
    let ev_done = s.ev_done.as_ref().expect("作成済み").0;
    // 入力は pinned へ写してから積む (呼び出し側の Vec を in-flight 中に指さない)。
    {
        let px = s.pin_x.as_mut().expect("ensure 済み");
        px.as_mut_slice::<f32>()[..x.len()].copy_from_slice(x);
    }
    let px_ptr = s.pin_x.as_ref().expect("ensure 済み").ptr;
    let dx = s.dx.as_ref().expect("ensure 済み").ptr();
    let (Some(event_record), Some(stream_wait_event)) = (l.event_record, l.stream_wait_event) else {
        bail!("cuEventRecord / cuStreamWaitEvent が無い");
    };
    // 3) 重みの H2D が起きた層だけ null ストリームとの順序を取る。
    if uploaded {
        if unsafe { event_record(ev_ready, std::ptr::null_mut()) } != CUDA_SUCCESS {
            bail!("cuEventRecord (ready) が失敗");
        }
        if unsafe { stream_wait_event(stream, ev_ready, 0) } != CUDA_SUCCESS {
            bail!("cuStreamWaitEvent が失敗");
        }
    }
    // 4) x の H2D → 各 GEMV の起動 → 各 D2H を 1 本のストリームに積む。
    if unsafe { (l.memcpy_htod_async)(dx, px_ptr as *const c_void, bytes_x, stream) } != CUDA_SUCCESS
    {
        bail!("cuMemcpyHtoDAsync (shared x) が失敗");
    }
    for (i, plan) in plans.iter().enumerate() {
        let dout = s.slots[i].dout.as_ref().expect("ensure 済み").ptr();
        let po = s.slots[i].pin_o.as_ref().expect("ensure 済み").ptr;
        ctx.launch_moe_gemv_on(plan, pins[i].buf.ptr, dx, dout, jobs[i].k, jobs[i].n, stream)?;
        if unsafe { (l.memcpy_dtoh_async)(po, dout, jobs[i].n * 4, stream) } != CUDA_SUCCESS {
            bail!("cuMemcpyDtoHAsync (shared {i}) が失敗");
        }
    }
    // 5) 完了イベント (join はこれだけを待つ)。
    if unsafe { event_record(ev_done, stream) } != CUDA_SUCCESS {
        bail!("cuEventRecord (done) が失敗");
    }
    Ok(())
}


/// shared expert の GPU GEMV を先に積む。
///
/// 空のハンドルが返ったら呼び出し側は従来の同期経路で計算すること。
pub fn shared_gemv_begin(x: &[f32], jobs: &[SharedGemvJob<'_>]) -> SharedPending {
    OVERLAP_BEGIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if jobs.is_empty() || jobs.len() > SHARED_OVERLAP_MAX_JOBS {
        return SharedPending::none();
    }
    #[cfg(test)]
    if SHARED_OVERLAP_FAKE.load(std::sync::atomic::Ordering::Relaxed) {
        return shared_gemv_begin_fake(x, jobs);
    }
    if !cuda_enabled() {
        return SharedPending::none();
    }
    let Some(ctx) = shared_ctx() else {
        return SharedPending::none();
    };
    if ctx.make_current().is_err() {
        return SharedPending::none();
    }
    let Ok(l) = lib() else {
        return SharedPending::none();
    };
    if l.stream_create.is_none()
        || l.stream_wait_event.is_none()
        || l.event_create.is_none()
        || l.event_record.is_none()
        || l.event_sync.is_none()
    {
        return SharedPending::none();
    }
    let mode = gemv_kernel_mode();
    // 計画と重み (GPU へはまだ 1 つも積まない)。
    let mut plans = Vec::with_capacity(jobs.len());
    let mut pins = Vec::with_capacity(jobs.len());
    let mut uploaded = false;
    for j in jobs {
        if j.k == 0 || j.n == 0 || j.k != x.len() || j.k % j.ty.blck_size() != 0 {
            return SharedPending::none();
        }
        // task 310: gate / up (この経路は 2 本だけ) を v5 で先に引く。
        let plan = match ctx.plan_shared_gemv(j.ty, j.k, j.n, mode, dec_sh_v5_mode() >= 1) {
            Some(p) => p,
            None => return SharedPending::none(),
        };
        let existed = gguf_registry()
            .lock()
            .map(|g| g.map.contains_key(&j.uid))
            .unwrap_or(false);
        let w = match weights_for_gguf(j.uid, j.ty, j.blocks, j.k, j.n) {
            Ok(w) => w,
            Err(_) => return SharedPending::none(),
        };
        if w.buf.bytes() != j.n.saturating_mul(plan.rb) {
            return SharedPending::none();
        }
        uploaded |= !existed;
        plans.push(plan);
        pins.push(w);
    }
    SHARED_OVERLAP.with(|cell| {
        let mut s = cell.borrow_mut();
        match shared_overlap_enqueue(&mut s, ctx, l, x, jobs, &plans, &pins, uploaded) {
            Ok(()) => {
                s.pins = pins;
                s.in_flight = jobs.len();
                OVERLAP_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SharedPending {
                    n: jobs.len(),
                    _not_send: std::marker::PhantomData,
                }
            }
            Err(_) => {
                let _ = shared_overlap_drain(&mut s);
                s.broken = true;
                SharedPending::none()
            }
        }
    })
}


/// キャッシュ 1 本ぶん (層 = uid の組ごと)。
pub(crate) struct ShCacheEntry {
    /// 控えたときの登録簿の世代
    pub(crate) gen: u64,
    /// (uid, k, n, ty) x 3 (gate / up / down)。形が変わったら作り直す。
    pub(crate) shapes: [(u64, usize, usize, crate::kquant::GgmlType); 3],
    pub(crate) plans: [MoeGemvPlan; 3],
    /// **握り続ける**重み (pin が立ったまま = 退避されない)
    pub(crate) pins: [GgufWRef; 3],
}


thread_local! {
    /// task 307: 層ごとの計画キャッシュ (`DENNO_SH_BEGIN_CACHE`)。
    pub(crate) static SH_CACHE: std::cell::RefCell<Vec<ShCacheEntry>> =
        const { std::cell::RefCell::new(Vec::new()) };
}


fn sh_gu_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_SH_GU_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_sh_gu.ptx"),
    }
}


/// 新カーネルを取得する (初回だけ PTX を読む。失敗は `None` を焼き付ける)。
fn sh_gu_kernels() -> Option<&'static ShGuKernels> {
    SH_GU_KERNELS
        .get_or_init(|| {
            let path = sh_gu_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(_) => return None,
            };
            let text = String::from_utf8_lossy(&image).to_string();
            if ptx_entry_params(&text, SH_GU_KERNEL_NAME) != Some(SH_GU_NARGS) {
                eprintln!(
                    "[sh-gu] PTX の引数が {SH_GU_NARGS} 本でない → v5 (gate/up 2 発) へ。\
                     `bash build_ptx.sh sh_gu` で再生成する"
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
                let mut cname = SH_GU_KERNEL_NAME.as_bytes().to_vec();
                cname.push(0);
                let mut f: CUfunction = std::ptr::null_mut();
                if (lib.module_get_function)(&mut f, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[sh-gu] PTX に {SH_GU_KERNEL_NAME} が無い → v5 (gate/up 2 発) へ");
                    return None;
                }
                Some(ShGuKernels { module, f })
            }
        })
        .as_ref()
}


/// 新カーネルが載っているか (テスト / 診断用)。
pub fn sh_gu_loaded() -> bool {
    sh_gu_kernels().is_some()
}


/// 融合経路を使うか (`DENNO_DEC_SH_GU`、既定 **on**。切り戻しは `=0`)。
fn sh_gu_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| parse_sh_gu(std::env::var("DENNO_DEC_SH_GU").ok().as_deref()))
}


/// 融合経路の門番 (純関数)。`.cu` の前提をここで全部見る。
///
/// - Q8_0 (ホスト側で別途) / `n_in % 64 == 0` / 32 要素ブロック数が偶数 /
///   `np = n_in/64` が lanes で割り切れる / `rows * lanes <= 1024`。
pub fn sh_gu_shape_ok(k: usize, n: usize, rows: usize, lanes: usize) -> bool {
    if k == 0 || n == 0 || rows == 0 || lanes == 0 {
        return false;
    }
    if k % 64 != 0 || !lanes.is_power_of_two() || lanes > 32 {
        return false;
    }
    if (k / 64) % lanes != 0 {
        return false;
    }
    rows.saturating_mul(lanes) <= 1024
}


/// 融合カーネルを 1 発積む (grid = ceil(n_out / rows)、block = rows * lanes)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_sh_gu(
    ctx: &CudaCtx,
    stream: CUstream,
    wg: CUdeviceptr,
    wu: CUdeviceptr,
    x: CUdeviceptr,
    act: CUdeviceptr,
    n_in: usize,
    n_out: usize,
    row_bytes: usize,
    rows: usize,
    lanes: usize,
) -> Result<()> {
    let k = sh_gu_kernels().ok_or_else(|| anyhow!("sh_gu のカーネルが無い"))?;
    let (mut pwg, mut pwu, mut px, mut pa) = (wg, wu, x, act);
    let (mut ni, mut no, mut rb) = (n_in as c_int, n_out as c_int, row_bytes as c_int);
    let (mut rw, mut ln) = (rows as c_int, lanes as c_int);
    let mut args = [
        &mut pwg as *mut _ as *mut c_void,
        &mut pwu as *mut _ as *mut c_void,
        &mut px as *mut _ as *mut c_void,
        &mut pa as *mut _ as *mut c_void,
        &mut ni as *mut _ as *mut c_void,
        &mut no as *mut _ as *mut c_void,
        &mut rb as *mut _ as *mut c_void,
        &mut rw as *mut _ as *mut c_void,
        &mut ln as *mut _ as *mut c_void,
    ];
    unsafe {
        ctx.launch_async_on(
            stream,
            k.f,
            (n_out.div_ceil(rows) as u32, 1, 1),
            ((rows * lanes) as u32, 1, 1),
            0,
            &mut args,
        )
    }
}


/// `[sh-gu]` の 1 行 (ゲート off なら `None`)。テスト用にも公開する。
pub fn sh_gu_line() -> Option<String> {
    if !sh_gu_on() {
        return None;
    }
    let f = SH_GU_FUSED.load(std::sync::atomic::Ordering::Relaxed);
    let b = SH_GU_FALLBACK.load(std::sync::atomic::Ordering::Relaxed);
    Some(format!(
        "[sh-gu] fused={f} fallback={b} ptx={}",
        if sh_gu_loaded() { "ok" } else { "none" }
    ))
}


/// task 187: shared expert の 4 発 (gate/up/silu/down) をサイドストリームへ積む。
pub fn shared_expert_begin(
    x: &[f32],
    gate: &SharedGemvJob<'_>,
    up: &SharedGemvJob<'_>,
    down: &SharedGemvJob<'_>,
) -> SharedPending {
    OVERLAP_BEGIN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    // 形の検査 (全部そろってから GPU に触る)。
    if gate.n == 0
        || up.n == 0
        || down.n == 0
        || up.n != gate.n
        || gate.k != up.k
        || down.k != gate.n
        || x.len() != gate.k
        || gate.k == 0
    {
        return SharedPending::none();
    }
    #[cfg(test)]
    if SHARED_OVERLAP_FAKE.load(std::sync::atomic::Ordering::Relaxed) {
        return shared_expert_begin_fake(x, gate, up, down);
    }
    if !cuda_enabled() {
        return SharedPending::none();
    }
    let Some(ctx) = shared_ctx() else {
        return SharedPending::none();
    };
    if ctx.make_current().is_err() {
        return SharedPending::none();
    }
    let Ok(l) = lib() else {
        return SharedPending::none();
    };
    if l.stream_create.is_none()
        || l.stream_wait_event.is_none()
        || l.event_create.is_none()
        || l.event_record.is_none()
        || l.event_sync.is_none()
    {
        return SharedPending::none();
    }
    // silu カーネル (kernels_dec_epi.ptx) が無ければ従来経路へ。
    let Some(epi) = dec_epi_kernels() else {
        return SharedPending::none();
    };
    let mode = gemv_kernel_mode();
    let jobs: [&SharedGemvJob<'_>; 3] = [gate, up, down];
    // task 307: 形の門番 (キャッシュ経路でも同じ条件を通す)。
    for j in jobs {
        if j.k % j.ty.blck_size() != 0 {
            return SharedPending::none();
        }
    }
    let mut plans: Vec<MoeGemvPlan> = Vec::with_capacity(3);
    let mut pins: Vec<GgufWRef> = Vec::with_capacity(3);
    let mut wptrs: [CUdeviceptr; 3] = [0; 3];
    let mut uploaded = false;
    let mut cached = false;
    if sh_begin_cache_on() {
        if let Some((p, w)) = sh_cache_lookup(&jobs) {
            plans.extend_from_slice(&p);
            wptrs = w;
            cached = true;
        }
    }
    if !cached {
        // task 310: `DENNO_DEC_SH_V5` のとき v5 の計画を先に引く (形が合わなければ
        // 従来の v3 / v2 へ落ちる)。down は `2` のときだけ従来のまま。
        let v5 = dec_sh_v5_mode();
        for (i, j) in jobs.iter().enumerate() {
            let allow_v5 = v5 == 1 || (v5 == 2 && i < 2);
            let plan = match ctx.plan_shared_gemv(j.ty, j.k, j.n, mode, allow_v5) {
                Some(p) => p,
                None => return SharedPending::none(),
            };
            let existed = gguf_registry()
                .lock()
                .map(|g| g.map.contains_key(&j.uid))
                .unwrap_or(false);
            let w = match weights_for_gguf(j.uid, j.ty, j.blocks, j.k, j.n) {
                Ok(w) => w,
                Err(_) => return SharedPending::none(),
            };
            if w.buf.bytes() != j.n.saturating_mul(plan.rb) {
                return SharedPending::none();
            }
            uploaded |= !existed;
            wptrs[plans.len()] = w.buf.ptr;
            plans.push(plan);
            pins.push(w);
        }
        // task 307: 控えておく (握り続ける = 以後この経路は plan も登録簿も触らない)。
        // 所有権はキャッシュへ移す (この回の in-flight 中もキャッシュが生かす)。
        if sh_begin_cache_on() {
            sh_cache_insert(&jobs, &plans, pins);
            pins = Vec::new();
        }
    }
    SHARED_OVERLAP.with(|cell| {
        let mut s = cell.borrow_mut();
        match shared_expert_enqueue(&mut s, ctx, l, x, &jobs, &plans, &wptrs, uploaded, epi) {
            Ok(()) => {
                s.pins = pins;
                s.in_flight = 1;
                OVERLAP_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                SharedPending {
                    n: 1,
                    _not_send: std::marker::PhantomData,
                }
            }
            Err(_) => {
                let _ = shared_overlap_drain(&mut s);
                s.broken = true;
                SharedPending::none()
            }
        }
    })
}


/// [`shared_expert_begin`] の発行本体 (検査の後にだけ呼ぶ)。
#[allow(clippy::too_many_arguments)]
fn shared_expert_enqueue(
    s: &mut SharedOverlapState,
    ctx: &CudaCtx,
    l: &CudaLib,
    x: &[f32],
    jobs: &[&SharedGemvJob<'_>; 3],
    plans: &[MoeGemvPlan],
    wptrs: &[CUdeviceptr; 3],
    uploaded: bool,
    epi: &DecEpiKernels,
) -> Result<()> {
    if s.broken || s.in_flight != 0 {
        bail!("重ね打ちは使わない (in_flight={})", s.in_flight);
    }
    shared_overlap_ensure_stream(s, l)?;
    let (gate, up, down) = (jobs[0], jobs[1], jobs[2]);
    let bytes_x = x.len() * 4;
    Scratch::ensure(&mut s.dx, bytes_x)?;
    Scratch::ensure_pin(&mut s.pin_x, bytes_x)?;
    Scratch::ensure(&mut s.dg, gate.n * 4)?;
    Scratch::ensure(&mut s.du, up.n * 4)?;
    Scratch::ensure(&mut s.dact, gate.n * 4)?;
    while s.slots.is_empty() {
        s.slots.push(SharedSlotBuf::new());
    }
    Scratch::ensure(&mut s.slots[0].dout, down.n * 4)?;
    Scratch::ensure_pin(&mut s.slots[0].pin_o, down.n * 4)?;
    s.slots[0].n = down.n;

    let stream = s.stream.as_ref().expect("作成済み").0;
    let ev_ready = s.ev_ready.as_ref().expect("作成済み").0;
    let ev_done = s.ev_done.as_ref().expect("作成済み").0;
    {
        let px = s.pin_x.as_mut().expect("ensure 済み");
        px.as_mut_slice::<f32>()[..x.len()].copy_from_slice(x);
    }
    let px_ptr = s.pin_x.as_ref().expect("ensure 済み").ptr;
    let dx = s.dx.as_ref().expect("ensure 済み").ptr();
    let dg = s.dg.as_ref().expect("ensure 済み").ptr();
    let du = s.du.as_ref().expect("ensure 済み").ptr();
    let dact = s.dact.as_ref().expect("ensure 済み").ptr();
    let dout = s.slots[0].dout.as_ref().expect("ensure 済み").ptr();
    let po = s.slots[0].pin_o.as_ref().expect("ensure 済み").ptr;
    let (Some(event_record), Some(stream_wait_event)) = (l.event_record, l.stream_wait_event) else {
        bail!("cuEventRecord / cuStreamWaitEvent が無い");
    };
    if uploaded {
        if unsafe { event_record(ev_ready, std::ptr::null_mut()) } != CUDA_SUCCESS {
            bail!("cuEventRecord (ready) が失敗");
        }
        if unsafe { stream_wait_event(stream, ev_ready, 0) } != CUDA_SUCCESS {
            bail!("cuStreamWaitEvent が失敗");
        }
    }
    if unsafe { (l.memcpy_htod_async)(dx, px_ptr as *const c_void, bytes_x, stream) }
        != CUDA_SUCCESS
    {
        bail!("cuMemcpyHtoDAsync (shared x) が失敗");
    }
    // gate → up → silu*up → down (すべて同じサイドストリーム)。
    // task 576: ゲート + 形 + PTX が揃っていれば gate/up/silu を 1 発に畳む。
    let mut gu_fused = false;
    if sh_gu_on() {
        let both_q8 = gate.ty == crate::kquant::GgmlType::Q8_0
            && up.ty == crate::kquant::GgmlType::Q8_0;
        let same = gate.k == up.k
            && gate.n == up.n
            && plans[0].rows_per_block == plans[1].rows_per_block
            && plans[0].lanes == plans[1].lanes
            && plans[0].rb == plans[1].rb;
        let v5q8 = gemv_v5_kernels().map(|k| k.f[0]);
        if both_q8
            && same
            && Some(plans[0].f) == v5q8
            && Some(plans[1].f) == v5q8
            && sh_gu_shape_ok(gate.k, gate.n, plans[0].rows_per_block, plans[0].lanes)
            && sh_gu_kernels().is_some()
        {
            match launch_sh_gu(
                ctx,
                stream,
                wptrs[0],
                wptrs[1],
                dx,
                dact,
                gate.k,
                gate.n,
                plans[0].rb,
                plans[0].rows_per_block,
                plans[0].lanes,
            ) {
                Ok(()) => {
                    gu_fused = true;
                    SH_GU_FUSED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                Err(err) => {
                    let _ = shared_overlap_drain(s);
                    s.broken = true;
                    bail!("sh_gu の起動に失敗: {err:#}");
                }
            }
        } else {
            SH_GU_FALLBACK.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if !gu_fused {
        ctx.launch_moe_gemv_on(&plans[0], wptrs[0], dx, dg, gate.k, gate.n, stream)?;
        ctx.launch_moe_gemv_on(&plans[1], wptrs[1], dx, du, up.k, up.n, stream)?;
        unsafe {
            let (mut pg, mut pu, mut pa, mut pn) = (
                dg as *mut c_void,
                du as *mut c_void,
                dact as *mut c_void,
                gate.n as c_int,
            );
            let mut args = [
                &mut pg as *mut _ as *mut c_void,
                &mut pu as *mut _ as *mut c_void,
                &mut pa as *mut _ as *mut c_void,
                &mut pn as *mut _ as *mut c_void,
            ];
            let threads = 256usize;
            let blocks = gate.n.div_ceil(threads) as u32;
            ctx.launch_async_on(
                stream,
                epi.silu,
                (blocks.max(1), 1, 1),
                (threads as u32, 1, 1),
                0,
                &mut args,
            )?;
        }
    }
    ctx.launch_moe_gemv_on(&plans[2], wptrs[2], dact, dout, down.k, down.n, stream)?;
    if unsafe { (l.memcpy_dtoh_async)(po, dout, down.n * 4, stream) } != CUDA_SUCCESS {
        bail!("cuMemcpyDtoHAsync (shared down) が失敗");
    }
    if unsafe { event_record(ev_done, stream) } != CUDA_SUCCESS {
        bail!("cuEventRecord (done) が失敗");
    }
    Ok(())
}


/// テスト用の偽 GPU 経路 (デバイスを 1 つも触らない)。ホストの `expert_gemv` と
/// `silu_inplace` で同じ計算をする (丸めはホストのまま = 参照値)。
#[cfg(test)]
fn shared_expert_begin_fake(
    x: &[f32],
    gate: &SharedGemvJob<'_>,
    up: &SharedGemvJob<'_>,
    down: &SharedGemvJob<'_>,
) -> SharedPending {
    let mut g = vec![0f32; gate.n];
    let mut u = vec![0f32; up.n];
    if crate::model::moe_expert::expert_gemv(gate.ty, gate.blocks, gate.k, gate.n, x, &mut g)
        .is_err()
        || crate::model::moe_expert::expert_gemv(up.ty, up.blocks, up.k, up.n, x, &mut u).is_err()
    {
        return SharedPending::none();
    }
    crate::tensor::silu_inplace(&mut g);
    for r in 0..g.len() {
        g[r] *= u[r];
    }
    let mut h = vec![0f32; down.n];
    if crate::model::moe_expert::expert_gemv(down.ty, down.blocks, down.k, down.n, &g, &mut h)
        .is_err()
    {
        return SharedPending::none();
    }
    SHARED_OVERLAP.with(|cell| {
        let mut s = cell.borrow_mut();
        while s.slots.is_empty() {
            s.slots.push(SharedSlotBuf::new());
        }
        s.slots[0].n = h.len();
        s.slots[0].fake = h;
        s.fake = true;
        s.in_flight = 1;
        OVERLAP_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SharedPending {
            n: 1,
            _not_send: std::marker::PhantomData,
        }
    })
}


/// 積んだ GEMV を join して `outs[i]` へ写す。
///
/// 戻り値 `false` = 「GPU の結果が取れなかった」。呼び出し側は従来の同期経路で
/// 計算し直すこと (このとき `outs` は 1 要素も書かれていない)。
pub fn shared_gemv_finish(p: &mut SharedPending, outs: &mut [&mut [f32]]) -> bool {
    let n = std::mem::replace(&mut p.n, 0);
    if n == 0 {
        return false;
    }
    // テスト用: join を必ず失敗させる (呼び出し側の切り戻し経路を試す)。
    #[cfg(test)]
    if SHARED_OVERLAP_FAIL_FINISH.load(std::sync::atomic::Ordering::Relaxed) {
        let _ = SHARED_OVERLAP.try_with(|cell| {
            let mut s = cell.borrow_mut();
            let _ = shared_overlap_drain(&mut s);
        });
        OVERLAP_FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return false;
    }
    SHARED_OVERLAP.with(|cell| -> bool {
        let mut s = cell.borrow_mut();
        if s.in_flight != n || outs.len() != n {
            let _ = shared_overlap_drain(&mut s);
            s.broken = true;
            return false;
        }
        // テスト用の偽 GPU: 計算済みの値を書き戻すだけ (GPU には触らない)。
        #[cfg(test)]
        if s.fake {
            for (i, out) in outs.iter_mut().enumerate().take(n) {
                let v = std::mem::take(&mut s.slots[i].fake);
                let len = v.len().min(out.len());
                out[..len].copy_from_slice(&v[..len]);
            }
            s.in_flight = 0;
            s.fake = false;
            OVERLAP_JOINED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return true;
        }
        for (i, out) in outs.iter().enumerate().take(n) {
            if out.len() < s.slots[i].n {
                let _ = shared_overlap_drain(&mut s);
                s.broken = true;
                return false;
            }
        }
        let Some(ctx) = shared_ctx() else {
            let _ = shared_overlap_drain(&mut s);
            s.broken = true;
            return false;
        };
        if ctx.make_current().is_err() {
            let _ = shared_overlap_drain(&mut s);
            s.broken = true;
            return false;
        }
        let Some(event_sync) = lib().ok().and_then(|l| l.event_sync) else {
            let _ = shared_overlap_drain(&mut s);
            s.broken = true;
            return false;
        };
        let Some(ev) = s.ev_done.as_ref().map(|e| e.0) else {
            let _ = shared_overlap_drain(&mut s);
            s.broken = true;
            return false;
        };
        // task 187: 台帳の副区間 (join の待ち / 写し)。
        {
            let _w = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_SH_WAIT);
            if unsafe { event_sync(ev) } != CUDA_SUCCESS {
                let _ = shared_overlap_drain(&mut s);
                s.broken = true;
                OVERLAP_FAILED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return false;
            }
        }
        {
            let _c = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_SH_COPY);
            for (i, out) in outs.iter_mut().enumerate().take(n) {
                let len = s.slots[i].n;
                let pin = s.slots[i].pin_o.as_ref().expect("ensure 済み");
                out[..len].copy_from_slice(&pin.as_slice::<f32>()[..len]);
            }
        }
        s.in_flight = 0;
        s.pins.clear();
        OVERLAP_JOINED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        true
    })
}


/// テスト用の偽 GPU 経路 (GPU を 1 つも触らない)。
#[cfg(test)]
fn shared_gemv_begin_fake(x: &[f32], jobs: &[SharedGemvJob<'_>]) -> SharedPending {
    let mut vals = Vec::with_capacity(jobs.len());
    for j in jobs {
        let mut v = vec![0f32; j.n];
        if crate::model::moe_expert::expert_gemv(j.ty, j.blocks, j.k, j.n, x, &mut v).is_err() {
            return SharedPending::none();
        }
        vals.push(v);
    }
    SHARED_OVERLAP.with(|cell| {
        let mut s = cell.borrow_mut();
        while s.slots.len() < vals.len() {
            s.slots.push(SharedSlotBuf::new());
        }
        for (i, v) in vals.into_iter().enumerate() {
            s.slots[i].n = v.len();
            s.slots[i].fake = v;
        }
        s.fake = true;
        s.in_flight = jobs.len();
        OVERLAP_TAKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        SharedPending {
            n: jobs.len(),
            _not_send: std::marker::PhantomData,
        }
    })
}
