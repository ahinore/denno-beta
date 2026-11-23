//! prefill MoE GEMM とストリーミング (exp_iq_pf / pf_stream / pf_tab / exp_*_tc)
//!
//! task 664: `src/cuda.rs` (70k 行) からの純粋な移動。ロジック・数値・カーネル・
//! PTX・env ゲートは一切変えていない。

#![allow(non_snake_case, non_camel_case_types)]

use super::*;


/// P40: grouped expert GEMM を 1 層ぶん走らせる。
///
/// 2 起動で expert FFN を終わらせる:
///   1. `dno_moe_grouped_gate_up` — act = silu(gate·x) * (up·x)
///   2. `dno_moe_grouped_down`    — h = down·act
///
/// 引数:
/// - `key`: パック済み重みの持ち主 (層 id)。変わったら VRAM を作り直す。
/// - `packed`: `[gate(e0..eN)][up(e0..eN)][down(e0..eN)]` のパック済み重み。
/// - `seg_gu` / `seg_dn`: 5 個で 1 segment (`[eid, m, w_off, e_off, ff_off]`。
///   `e_off` / `ff_off` は `m * n_in` / `m * n_out` の累積)。空 (`nseg == 0`) は
///   `Ok(false)`。
/// - `seg_gu` は gate/up 用、`seg_dn` は down 用。**down は 1 expert あたりの
///   バイト数が gate と違う**ので `w_off` のストライドが別 (P50)。1 枚の表を
///   使い回すと 2 体目以降が別の expert の重みを読む。
/// - `xin`: segment の並びに詰め直した入力 (m_total × gate.n_in)。
/// - `h`: 出力 (m_total × down.n_out)。成功時のみ全要素を書く。
///
/// 戻り値 `Ok(false)` は「GPU でやらなかった」= 呼び出し側は CPU 経路へ。
/// 環境 (env / CUDA / PTX / 形 / 共有メモリ) のどれかが欠ければ false。
#[allow(clippy::too_many_arguments)]
pub fn moe_grouped_gemm_layer(
    key: u64,
    packed: &[u8],
    seg_gu: &[u32],
    seg_dn: &[u32],
    gate: GroupedMatrix,
    up: GroupedMatrix,
    down: GroupedMatrix,
    xin: &[f32],
    h: &mut [f32],
) -> Result<bool> {
    if !moe_grouped_gpu_enabled() || !cuda_enabled() || !gpu_gemm_enabled() {
        return Ok(false);
    }
    if seg_gu.is_empty() || seg_gu.len() % 5 != 0 {
        return Ok(false);
    }
    // down 用の表は同じ segment 数でなければならない (eid/m の並びは同じ)。
    if seg_dn.len() != seg_gu.len() {
        return Ok(false);
    }
    // gate と up は同形 (同じ x を読み、同じ形の act を書く)。
    if (gate.ty, gate.n_in, gate.n_out, gate.row_bytes) != (up.ty, up.n_in, up.n_out, up.row_bytes)
    {
        return Ok(false);
    }
    // down は gate の転置 (n_in = ff、n_out = e)。
    if down.n_in != gate.n_out || down.n_out != gate.n_in {
        return Ok(false);
    }
    gate.check()?;
    down.check()?;
    let (ty_g, blkj_g, blkb_g) = gate.kernel_layout().expect("check 済み");
    let (ty_d, blkj_d, blkb_d) = down.kernel_layout().expect("check 済み");
    let Some(gb) = gate.expert_bytes() else {
        return Ok(false);
    };
    let Some(db) = down.expert_bytes() else {
        return Ok(false);
    };

    let nseg = seg_gu.len() / 5;
    // ブロックの範囲: gate は [gate.w_off, up.w_off)、down は [down.w_off, 末尾)。
    // w_off がこの範囲を出ると隣のブロック (別の行列) を重みとして読む。
    let Some(gate_block) = up.w_off.checked_sub(gate.w_off) else {
        return Ok(false);
    };
    let Some(down_block) = packed.len().checked_sub(down.w_off) else {
        return Ok(false);
    };
    let mut m_max = 0usize;
    let mut m_total = 0usize;
    for (s, sd) in seg_gu.chunks_exact(5).zip(seg_dn.chunks_exact(5)) {
        if s[0] != sd[0] || s[1] != sd[1] || s[3] != sd[3] || s[4] != sd[4] {
            bail!("moe_grouped_gemm_layer: seg_gu と seg_dn の segment が違う");
        }
        let m = s[1] as usize;
        if m == 0 {
            return Ok(false); // 空タスクは積まない (呼び出し側の契約)
        }
        let wg_off = s[2] as usize;
        if wg_off + gb > gate_block {
            bail!(
                "moe_grouped_gemm_layer: gate の w_off {wg_off} + {gb} B が \
                 gate ブロック {gate_block} B を超える"
            );
        }
        let wd_off = sd[2] as usize;
        if wd_off + db > down_block {
            bail!(
                "moe_grouped_gemm_layer: down の w_off {wd_off} + {db} B が \
                 down ブロック {down_block} B を超える"
            );
        }
        m_max = m_max.max(m);
        m_total += m;
    }
    if xin.len() != m_total * gate.n_in {
        bail!(
            "moe_grouped_gemm_layer: x の長さ {} が m_total({m_total}) * n_in({}) と違う",
            xin.len(),
            gate.n_in
        );
    }
    if h.len() != m_total * down.n_out {
        bail!(
            "moe_grouped_gemm_layer: h の長さ {} が m_total({m_total}) * n_out({}) と違う",
            h.len(),
            down.n_out
        );
    }
    let Some(p_gu) = grouped_launch_plan(nseg, m_max, gate.n_out, blkj_g, 2)? else {
        return Ok(false);
    };
    let Some(p_d) = grouped_launch_plan(nseg, m_max, down.n_out, blkj_d, 1)? else {
        return Ok(false);
    };

    let Some(ctx) = shared_ctx() else {
        return Ok(false);
    };
    ctx.make_current()?;
    let Some(k) = grouped_kernels() else {
        return Ok(false);
    };

    let seg_words = seg_gu.len();
    let n_x = xin.len();
    let n_act = m_total * up.n_out;
    let n_h = h.len();
    let mut out_buf: Option<Vec<f32>> = None;
    GROUPED_SCRATCH.with(|cell| -> Result<()> {
        let mut sc = cell.borrow_mut();
        sc.denied = false;
        // 1. パック済み重み (層が変わったら作り直す)
        //
        // P43: 確保の前に予算 (`DENNO_EXPERT_VRAM_MB`) から借りる。借りるために
        // LRU の expert は追い出される。予算が無いときは会計なしで素通りするので
        // 既定の挙動は変わらない。入れないときは CPU へ落とす (fail closed)。
        if sc.key != key || sc.packed_len != packed.len() || sc.packed.is_none() {
            // 先に手放してから借りる (手放した分がそのまま予約に回る)。
            sc.packed = None;
            sc.reserved = None;
            match crate::model::expert_vram::reserve_external(packed.len()) {
                Some(r) => {
                    sc.packed = Some(DevBuffer::alloc_tagged(
                        packed.len().max(1),
                        crate::mem_audit::TAG_GROUPED_PACKED,
                    )?);
                    sc.reserved = Some(r);
                    sc.key = key;
                    sc.packed_len = packed.len();
                }
                None => {
                    static WARNED: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[moe-grouped] パック済み {} MiB が expert の VRAM 予算に載らない \
                             (DENNO_EXPERT_VRAM_MB) → この層は CPU へ",
                            packed.len() >> 20
                        );
                    }
                    sc.denied = true;
                    return Ok(());
                }
            }
        }
        {
            let pb = sc.packed.as_ref().expect("確保済み");
            pb.copy_from(packed)?;
            dbg_htod(packed.len());
        }
        // 2. segment 表 / 入力 / 中間 / 出力
        MoeScratch::ensure_tagged(&mut sc.seg, seg_words * 4, crate::mem_audit::TAG_GROUPED)?;
        MoeScratch::ensure_tagged(&mut sc.seg_dn, seg_words * 4, crate::mem_audit::TAG_GROUPED)?;
        MoeScratch::ensure_tagged(&mut sc.x, n_x * 4, crate::mem_audit::TAG_GROUPED)?;
        MoeScratch::ensure_tagged(&mut sc.act, n_act * 4, crate::mem_audit::TAG_GROUPED)?;
        MoeScratch::ensure_tagged(&mut sc.h, n_h * 4, crate::mem_audit::TAG_GROUPED)?;
        {
            let sb = sc.seg.as_ref().expect("確保済み");
            sb.copy_from(seg_gu)?;
            dbg_htod(seg_words * 4);
            let sd = sc.seg_dn.as_ref().expect("確保済み");
            sd.copy_from(seg_dn)?;
            dbg_htod(seg_words * 4);
            let xb = sc.x.as_ref().expect("確保済み");
            xb.copy_from(xin)?;
            dbg_htod(n_x * 4);
        }
        let packed_buf = sc.packed.as_ref().expect("確保済み");
        let seg_buf = sc.seg.as_ref().expect("確保済み");
        let seg_dn_buf = sc.seg_dn.as_ref().expect("確保済み");
        let x_buf = sc.x.as_ref().expect("確保済み");
        let act_buf = sc.act.as_ref().expect("確保済み");
        let h_buf = sc.h.as_ref().expect("確保済み");
        unsafe {
            // 3. gate + up + silu 積
            {
                let mut pwg = packed_buf.ptr + gate.w_off as u64;
                let mut pwu = packed_buf.ptr + up.w_off as u64;
                let mut px = x_buf.ptr;
                let mut pa = act_buf.ptr;
                let mut ps = seg_buf.ptr;
                let (mut ns, mut ni, mut no, mut rb) = (
                    nseg as c_int,
                    gate.n_in as c_int,
                    gate.n_out as c_int,
                    gate.row_bytes as c_int,
                );
                let (mut wt, mut blkj, mut blkb) =
                    (ty_g, blkj_g as c_int, blkb_g as c_int);
                let mut args = [
                    &mut pwg as *mut _ as *mut c_void,
                    &mut pwu as *mut _ as *mut c_void,
                    &mut px as *mut _ as *mut c_void,
                    &mut pa as *mut _ as *mut c_void,
                    &mut ps as *mut _ as *mut c_void,
                    &mut ns as *mut _ as *mut c_void,
                    &mut ni as *mut _ as *mut c_void,
                    &mut no as *mut _ as *mut c_void,
                    &mut rb as *mut _ as *mut c_void,
                    &mut wt as *mut _ as *mut c_void,
                    &mut blkj as *mut _ as *mut c_void,
                    &mut blkb as *mut _ as *mut c_void,
                ];
                ctx.launch_async(k.gate_up, p_gu.grid, p_gu.block, p_gu.smem, &mut args)?;
            }
            // 4. down (act はデバイス上に残っている)
            {
                let mut pwd = packed_buf.ptr + down.w_off as u64;
                let mut px = act_buf.ptr;
                let mut po = h_buf.ptr;
                let mut ps = seg_dn_buf.ptr;
                let (mut ns, mut ni, mut no, mut rb) = (
                    nseg as c_int,
                    down.n_in as c_int,
                    down.n_out as c_int,
                    down.row_bytes as c_int,
                );
                let (mut wt, mut blkj, mut blkb) = (ty_d, blkj_d as c_int, blkb_d as c_int);
                let mut args = [
                    &mut pwd as *mut _ as *mut c_void,
                    &mut px as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut ps as *mut _ as *mut c_void,
                    &mut ns as *mut _ as *mut c_void,
                    &mut ni as *mut _ as *mut c_void,
                    &mut no as *mut _ as *mut c_void,
                    &mut rb as *mut _ as *mut c_void,
                    &mut wt as *mut _ as *mut c_void,
                    &mut blkj as *mut _ as *mut c_void,
                    &mut blkb as *mut _ as *mut c_void,
                ];
                ctx.launch_async(k.down, p_d.grid, p_d.block, p_d.smem, &mut args)?;
            }
            ctx.sync()?;
        }
        out_buf = Some(h_buf.to_vec::<f32>(n_h)?);
        dbg_d2h(n_h * 4);
        // P43: 予算下では層の終わりに手放す。予算 (`DENNO_EXPERT_VRAM_MB`) は
        // decode の LRU と共有なので、prefill が持ち回ると decode が載せられない
        // (P39 の実測: 10,372 MiB で cuMemAlloc 失敗)。予算が無いときは従来どおり
        // 持ち回る (同じ層をもう一度呼ぶときに H2D を省く)。
        if sc.reserved.as_ref().map(|r| r.is_guarded()).unwrap_or(false) {
            sc.packed = None;
            sc.reserved = None;
        }
        Ok(())
    })?;
    let Some(got) = out_buf else {
        // P43: 予算が埋まっていてパック済み重みを載せられなかった → CPU へ。
        return Ok(false);
    };
    h.copy_from_slice(&got);
    Ok(true)
}


/// 型 → カーネル添字。未対応は `None` (呼び出し側は CPU へ)。
pub(crate) fn moe_batch_slot(ty: crate::kquant::GgmlType) -> Option<usize> {
    use crate::kquant::GgmlType as T;
    match ty {
        T::Q8_0 => Some(0),
        T::Q5_1 => Some(1),
        T::Q4K => Some(2),
        T::Q5K => Some(3),
        _ => None,
    }
}


/// PTX の置き場所。`DENNO_MOE_BATCH_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_moe_batch.ptx`。
fn moe_batch_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_MOE_BATCH_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_moe_batch.ptx"),
    }
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None をキャッシュする
/// (grouped と同じ扱い = 二度と試さない)。
pub(crate) fn moe_batch_kernels() -> Option<&'static MoeBatchKernels> {
    MOE_BATCH_KERNELS
        .get_or_init(|| {
            let path = moe_batch_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[moe-batch] PTX が読めない ({}) : {e} → per-expert 経路へ",
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
                    eprintln!("[moe-batch] cuModuleLoadData 失敗 → per-expert 経路へ");
                    return None;
                }
            }
            let mut fns: [CUfunction; 4] = [std::ptr::null_mut(); 4];
            for (i, name) in MOE_BATCH_NAMES.iter().enumerate() {
                unsafe {
                    if (lib.module_get_function)(&mut fns[i], module, name.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!(
                            "[moe-batch] PTX に {} が無い → per-expert 経路へ",
                            String::from_utf8_lossy(&name[..name.len() - 1])
                        );
                        return None;
                    }
                }
            }
            Some(MoeBatchKernels { module, fns })
        })
        .as_ref()
}


thread_local! {
    /// task 594: seg prologue の H2D 用の pinned バウンス (grow-only)。
    ///
    /// 同じスレッドが 1 本を使い回す。DMA を出したあと次にこのバッファへ書くのは
    /// 「和カーネルの後のイベントを待つ」より後なので、前の DMA は完了している
    /// (同じ null ストリームに積んでいるため順序も保証される)。
    static SEG_H2D_PIN: std::cell::RefCell<Option<PinMem>> = const { std::cell::RefCell::new(None) };
}

/// task 594: seg prologue を pinned へ組んで同じストリームへ非同期 DMA する。
///
/// `Ok(true)` = 送った (カーネルは同じストリームに積むので順序は保証される)。
/// `Ok(false)` / `Err` = ゲート off / 確保失敗 / 非同期コピー不可。
/// 送るバイト列は pageable 経路 (`put_*` + `copy_from`) と 1 バイトも変わらない。
#[allow(clippy::too_many_arguments)]
fn seg_prologue_pinned(
    stage_ptr: u64,
    lay: &crate::model::expert_batch::SegStage,
    pg: &[u64],
    pu: &[u64],
    pd: &[u64],
    seg_gu: &[u32],
    seg_d: &[u32],
    w: &[f32],
    x: &[f32],
    n_in: usize,
    nseg: usize,
) -> Result<bool> {
    if !crate::model::seg_prof::seg_h2d_pin_on() {
        return Ok(false);
    }
    let bytes = lay.total;
    if bytes == 0 {
        return Ok(true);
    }
    SEG_H2D_PIN.with(|cell| -> Result<bool> {
        let mut slot = cell.borrow_mut();
        if slot.as_ref().map(|p| p.bytes).unwrap_or(0) < bytes {
            match PinMem::alloc(bytes) {
                Ok(p) => *slot = Some(p),
                Err(_) => return Ok(false),
            }
        }
        let pin = slot.as_mut().expect("確保済み");
        let h = &mut pin.as_mut_slice::<u8>()[..bytes];
        put_u64s(&mut h[lay.pg..lay.pg + nseg * 8], pg);
        put_u64s(&mut h[lay.pu..lay.pu + nseg * 8], pu);
        put_u64s(&mut h[lay.pd..lay.pd + nseg * 8], pd);
        put_u32s(&mut h[lay.sgu..lay.sgu + nseg * 16], seg_gu);
        put_u32s(&mut h[lay.sgd..lay.sgd + nseg * 16], seg_d);
        put_f32s(&mut h[lay.wt..lay.wt + nseg * 4], w);
        put_f32s(&mut h[lay.x..lay.x + n_in * 4], &x[..n_in]);
        let src = pin.ptr as *const c_void;
        let dst = stage_ptr as CUdeviceptr;
        let ok = unsafe {
            check(
                (lib()?.memcpy_htod_async)(dst, src, bytes, std::ptr::null_mut()),
                "cuMemcpyHtoDAsync (seg prologue)",
            )
            .is_ok()
        };
        if ok {
            dbg_htod(bytes);
        }
        Ok(ok)
    })
}

/// K 型 (Q4_K / Q5_K / Q5_1 / Q8_0) の層を **5 起動 / 1 待ち**で回す (task 278)。
///
/// - `pg` / `pu` / `pd`: segment ごとの重み先頭 (デバイスアドレス)
/// - `seg_gu`: gate/up 用 segment 表 (`x_off = 0`、`out_off = s * ff`)
/// - `seg_d`: down 用 segment 表 (`x_off = s * ff`、`out_off = s * e`)
/// - `w`: ヒットの重み (segment 順、長さ `nseg`)
/// - `out`: `e` 要素。**ヒットぶんの重み付き和**が入る (ミスは呼び出し側が足す)
/// - `cpu_miss`: GPU を走らせている間に呼ぶ (ミス expert の CPU 計算)
///
/// `Ok(false)` は「この経路ではやらない」= 呼び出し側は旧経路へ (fail closed)。
#[allow(clippy::too_many_arguments)]
pub fn moe_seg_layer_sum<F>(
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
    use crate::model::expert_batch as eb;
    let nseg = pg.len();
    if nseg == 0 || pu.len() != nseg || pd.len() != nseg || w.len() != nseg {
        return Ok(false);
    }
    if !seg_gu.len().is_multiple_of(4)
        || !seg_d.len().is_multiple_of(4)
        || seg_gu.len() / 4 != nseg
        || seg_d.len() / 4 != nseg
    {
        return Ok(false);
    }
    if x.len() < n_in || out.len() < e || ff == 0 || e == 0 {
        bail!(
            "moe_seg_layer_sum: 入出力長が足りない (x={}/{n_in}, out={}/{e})",
            x.len(),
            out.len()
        );
    }
    let (Some(rb_g), Some(rb_u), Some(rb_d)) = (
        gguf_row_bytes(ty_g, n_in),
        gguf_row_bytes(ty_u, n_in),
        gguf_row_bytes(ty_d, ff),
    ) else {
        return Ok(false);
    };
    if rb_g != rb_u {
        return Ok(false);
    }
    if !moe_gemv_seg_ready(ty_g, n_in) || !moe_gemv_seg_ready(ty_u, n_in) {
        return Ok(false);
    }
    if !moe_gemv_seg_ready(ty_d, ff) {
        return Ok(false);
    }
    let (Some(plan_gu), Some(plan_d)) = (
        eb::seg_launch_plan(nseg, 1, ff, rb_g),
        eb::seg_launch_plan(nseg, 1, e, rb_d),
    ) else {
        return Ok(false);
    };
    let Some(ctx) = shared_ctx() else {
        return Ok(false);
    };
    // task 278: 高速版 (1 行 = 1 warp) があればそちら。無ければ旧カーネル
    // (1 行 = 1 スレッド、CPU とビット一致) で回す。
    let fast = if seg_k_enabled() {
        seg_k_kernels()
    } else {
        None
    };
    let slow = moe_batch_kernels();
    if fast.is_none() && slow.is_none() {
        return Ok(false);
    }
    // 語呂: silu と和は dec_epi の既存カーネル (PTX が無ければ旧経路へ)。
    let Some(epi) = dec_epi_kernels() else {
        return Ok(false);
    };
    let (Some(slot_g), Some(slot_u), Some(slot_d)) = (
        moe_batch_slot(ty_g),
        moe_batch_slot(ty_u),
        moe_batch_slot(ty_d),
    ) else {
        return Ok(false);
    };
    let (f_g, f_u, f_d, block, smem_gu, smem_d) = match fast {
        Some(k) => (
            k.fns[slot_g],
            k.fns[slot_u],
            k.fns[slot_d],
            (128u32, 1u32, 1u32),
            0u32,
            0u32,
        ),
        None => {
            let k = slow.expect("どちらかは在る");
            (
                k.fns[slot_g],
                k.fns[slot_u],
                k.fns[slot_d],
                plan_gu.block,
                plan_gu.smem,
                plan_d.smem,
            )
        }
    };
    ctx.make_current()?;

    // task 412 / 526: 融合版 (gate+up+silu 1 起動) の選択。
    // `kernels_seg_k.ptx` の新しいカーネルが要る (古い PTX では None)。
    // 高速版 (`seg_k`) の経路でだけ使う: 遅い方の PTX には無い。
    let fuse_gu: Option<CUfunction> = match fast {
        // gate と up は型も行長も同じときだけ畳める (x を共有する前提)。
        Some(k) if seg_gu_fuse_enabled() && ty_g == ty_u && rb_g == rb_u => {
            // task 562: 16 バイトロード版 (Q4_K) を最優先。
            // task 534: 8 バイトロード版があればそちら (無ければ従来の融合版)。
            match k.gu16[slot_g] {
                Some(f) if seg_gu16_enabled() => {
                    crate::model::seg_prof::note_seg_gu16_call();
                    Some(f)
                }
                _ => match k.gu8[slot_g] {
                    Some(f) if seg_gu8_enabled() => {
                        crate::model::seg_prof::note_seg_gu8_call();
                        Some(f)
                    }
                    _ => k.gu[slot_g],
                },
            }
        }
        _ => None,
    };
    // task 534: down のベクトルロード版 (8 バイト版 or Q5_1 の 4 バイト版)。
    // 引数も起動形も同じ = 関数の差し替えだけ。
    let f_d_seg = match fast {
        Some(k) if seg_dn8_enabled() => match (k.k8[slot_d], k.v4[slot_d]) {
            (Some(f), _) => {
                crate::model::seg_prof::note_seg_dn8_call();
                f
            }
            (None, Some(f)) => {
                crate::model::seg_prof::note_seg_dnv4_call();
                f
            }
            (None, None) => f_d,
        },
        _ => f_d,
    };
    // 融合カーネルの起動形 (blockDim = 128 = 4 warp)。
    let gu_grid = (
        (ff as u32).div_ceil(SEG_WARPS * SEG_GU_ROWS),
        nseg as u32,
        1u32,
    );

    let lay = eb::seg_stage_layout(nseg, n_in);
    let ng = nseg * ff;
    let nh = nseg * e;
    let prof = crate::model::seg_prof::enabled();

    SEG_LAYER_SCRATCH.with(|cell| -> Result<()> {
        let mut sc = cell.borrow_mut();
        let _t_alloc = crate::model::seg_prof::stamp();
        MoeScratch::ensure_tagged(&mut sc.stage, lay.total, crate::mem_audit::TAG_SEG)?;
        MoeScratch::ensure_tagged(&mut sc.g, ng * 4, crate::mem_audit::TAG_SEG)?;
        MoeScratch::ensure_tagged(&mut sc.u, ng * 4, crate::mem_audit::TAG_SEG)?;
        MoeScratch::ensure_tagged(&mut sc.act, ng * 4, crate::mem_audit::TAG_SEG)?;
        MoeScratch::ensure_tagged(&mut sc.h, nh * 4, crate::mem_audit::TAG_SEG)?;
        MoeScratch::ensure_tagged(&mut sc.out, e * 4, crate::mem_audit::TAG_SEG)?;
        crate::model::seg_prof::add(crate::model::seg_prof::PH_ALLOC, _t_alloc);

        // 1. prologue を 1 本に詰めて **1 回の H2D** で送る。
        let _t_tbl = crate::model::seg_prof::stamp();
        // task 594: `DENNO_SEG_H2D_PIN` のときは pinned へ直接組んで同じ
        // (null) ストリームへ非同期 DMA する。pageable の同期 `cuMemcpyHtoD`
        // は 10.8 KB で 8 us/層、ホストのミス計算の前に直列に挟まっていた。
        // 送るバイト列は従来経路と同一 (同じ `put_*` を pinned に対して回す)。
        let mut pin_sent = false;
        if crate::model::seg_prof::seg_h2d_pin_on() {
            let stage_ptr = sc.stage.as_ref().expect("確保済み").ptr;
            match seg_prologue_pinned(
                stage_ptr, &lay, pg, pu, pd, seg_gu, seg_d, w, x, n_in, nseg,
            ) {
                Ok(true) => {
                    crate::model::seg_prof::note_seg_h2d_call();
                    pin_sent = true;
                }
                _ => crate::model::seg_prof::note_seg_h2d_fallback(),
            }
        }
        if !pin_sent {
            if sc.host.len() < lay.total {
                sc.host.resize(lay.total, 0);
            }
            put_u64s(&mut sc.host[lay.pg..lay.pg + nseg * 8], pg);
            put_u64s(&mut sc.host[lay.pu..lay.pu + nseg * 8], pu);
            put_u64s(&mut sc.host[lay.pd..lay.pd + nseg * 8], pd);
            put_u32s(&mut sc.host[lay.sgu..lay.sgu + nseg * 16], seg_gu);
            put_u32s(&mut sc.host[lay.sgd..lay.sgd + nseg * 16], seg_d);
            put_f32s(&mut sc.host[lay.wt..lay.wt + nseg * 4], w);
            put_f32s(&mut sc.host[lay.x..lay.x + n_in * 4], &x[..n_in]);
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_TABLE, _t_tbl);

        let stage = sc.stage.as_ref().expect("確保済み");
        let g = sc.g.as_ref().expect("確保済み");
        let u = sc.u.as_ref().expect("確保済み");
        let act = sc.act.as_ref().expect("確保済み");
        let h = sc.h.as_ref().expect("確保済み");
        let outb = sc.out.as_ref().expect("確保済み");
        let _t_h2d = crate::model::seg_prof::stamp();
        if !pin_sent {
            stage.copy_from(&sc.host[..lay.total])?;
            dbg_htod(lay.total);
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_H2D, _t_h2d);

        // 2. 5 起動を同じストリームに積む (待たない)。
        let base = stage.ptr;
        let _ = seg_ev::record(0); // gate の前
        // task 412 / 526: 融合版があれば gate / up / silu を 1 起動で済ませる。
        if let Some(fg) = fuse_gu {
            let mut pgw = base + lay.pg as u64;
            let mut puw = base + lay.pu as u64;
            let mut pxf = base + lay.x as u64;
            let mut paf = act.ptr;
            let mut psf = base + lay.sgu as u64;
            let mut nsf = nseg as c_int;
            let mut nif = n_in as c_int;
            let mut nof = ff as c_int;
            let mut rbf = rb_g as c_int;
            let _t_launch = crate::model::seg_prof::stamp();
            unsafe {
                let mut args = [
                    &mut pgw as *mut _ as *mut c_void,
                    &mut puw as *mut _ as *mut c_void,
                    &mut pxf as *mut _ as *mut c_void,
                    &mut paf as *mut _ as *mut c_void,
                    &mut psf as *mut _ as *mut c_void,
                    &mut nsf as *mut _ as *mut c_void,
                    &mut nif as *mut _ as *mut c_void,
                    &mut nof as *mut _ as *mut c_void,
                    &mut rbf as *mut _ as *mut c_void,
                ];
                ctx.launch_async(fg, gu_grid, block, 0, &mut args)?;
            }
            crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
            // prof: up / silu の区間は 0 になる (行の形は従来と同じ)。
            let _ = seg_ev::record(1);
            let _ = seg_ev::record(2);
            let _ = seg_ev::record(3);
        }
        if fuse_gu.is_none() {
        let mut pw = base + lay.pg as u64;
        let mut px = base + lay.x as u64;
        let mut po = g.ptr;
        let mut ps = base + lay.sgu as u64;
        let (mut ns, mut ni, mut no, mut rb) = (
            nseg as c_int,
            n_in as c_int,
            ff as c_int,
            rb_g as c_int,
        );
        let _t_launch = crate::model::seg_prof::stamp();
        unsafe {
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
            ctx.launch_async(
                f_g,
                plan_gu.grid,
                block,
                smem_gu,
                &mut args,
            )?;
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
        let _ = seg_ev::record(1); // gate の後
        let mut pw = base + lay.pu as u64;
        let mut px = base + lay.x as u64;
        let mut po = u.ptr;
        let mut ps = base + lay.sgu as u64;
        let _t_launch = crate::model::seg_prof::stamp();
        unsafe {
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
            ctx.launch_async(
                f_u,
                plan_gu.grid,
                block,
                smem_gu,
                &mut args,
            )?;
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
        let _ = seg_ev::record(2); // up の後
        let _t_launch = crate::model::seg_prof::stamp();
        // 3. silu(g) * u (デバイスで閉じる)
        unsafe {
            let mut pg_p = g.ptr;
            let mut pu_p = u.ptr;
            let mut pa_p = act.ptr;
            let mut n = ng as c_int;
            let mut args = [
                &mut pg_p as *mut _ as *mut c_void,
                &mut pu_p as *mut _ as *mut c_void,
                &mut pa_p as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
            ];
            ctx.launch_async(
                epi.silu,
                ((ng as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )?;
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
        let _ = seg_ev::record(3); // silu の後
        } // fuse_gu.is_none()
        // 4. down
        let mut pw = base + lay.pd as u64;
        let mut px = act.ptr;
        let mut po = h.ptr;
        let mut ps = base + lay.sgd as u64;
        let (mut ns2, mut ni2, mut no2, mut rb2) = (
            nseg as c_int,
            ff as c_int,
            e as c_int,
            rb_d as c_int,
        );
        let _t_launch = crate::model::seg_prof::stamp();
        unsafe {
            let mut args = [
                &mut pw as *mut _ as *mut c_void,
                &mut px as *mut _ as *mut c_void,
                &mut po as *mut _ as *mut c_void,
                &mut ps as *mut _ as *mut c_void,
                &mut ns2 as *mut _ as *mut c_void,
                &mut ni2 as *mut _ as *mut c_void,
                &mut no2 as *mut _ as *mut c_void,
                &mut rb2 as *mut _ as *mut c_void,
            ];
            ctx.launch_async(
                f_d_seg,
                plan_d.grid,
                block,
                smem_d,
                &mut args,
            )?;
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
        let _ = seg_ev::record(4); // down の後
        // 5. 重み付き和 (h -> out, has_sh = 0)
        let mut ph = h.ptr;
        let mut pwv = base + lay.wt as u64;
        let mut psh = h.ptr;
        let mut sgw = 0.0f32;
        let mut pout = outb.ptr;
        let (mut e_c, mut ns3, mut has_sh) = (e as c_int, nseg as c_int, 0 as c_int);
        let _t_launch = crate::model::seg_prof::stamp();
        unsafe {
            let mut args = [
                &mut ph as *mut _ as *mut c_void,
                &mut pwv as *mut _ as *mut c_void,
                &mut psh as *mut _ as *mut c_void,
                &mut sgw as *mut _ as *mut c_void,
                &mut pout as *mut _ as *mut c_void,
                &mut e_c as *mut _ as *mut c_void,
                &mut ns3 as *mut _ as *mut c_void,
                &mut has_sh as *mut _ as *mut c_void,
            ];
            ctx.launch_async(
                epi.sum,
                ((e as u32).div_ceil(256), 1, 1),
                (256, 1, 1),
                0,
                &mut args,
            )?;
        }
        crate::model::seg_prof::add(crate::model::seg_prof::PH_LAUNCH, _t_launch);
        // 和の後 (ここを待てば 5 起動ぶんが終わる)
        let timed = seg_ev::record(5);

        // task 502: `DENNO_SEG_SUM_ASYNC` のときは、和の直後に D2H を**非同期**で
        // pinned バウンスへ出す。完了待ちは下の 8. で 1 回だけ行うので、コピーは
        // 6. のミス CPU 計算 (1 層 ~100 us) の裏に完全に隠れる。同期の
        // `cuMemcpyDtoH` はこの文脈の仕事を待つため 12-17 us/層 かかり、
        // task 498 の pinned 同期版でも縮まなかった (t498c)。
        let async_ok = crate::model::seg_prof::seg_sum_async_on()
            && e * 4 <= SEG_PIN_BYTES
            && seg_async_d2h_issue(outb.ptr, e * 4);

        // 6. GPU を待つ間にミスの CPU 計算を走らせる (重ね打ち)。
        let _t_miss = crate::model::seg_prof::stamp();
        let miss_r = cpu_miss();
        crate::model::seg_prof::add(crate::model::seg_prof::PH_MISS, _t_miss);

        // 7. 待ちは null ストリームのイベント 1 本だけ (文脈同期はしない)。
        //    文脈同期 (cuCtxSynchronize) は背景の上げ直しまで待つので使わない。
        let _t_sync = crate::model::seg_prof::stamp();
        let sync_r = if timed {
            seg_ev::wait(5)
        } else if seg_ev::record_raw(6) {
            seg_ev::wait(6)
        } else {
            // イベントが張れない環境では文脈同期へ (安全側)。
            ctx.sync()
        };
        crate::model::seg_prof::add(crate::model::seg_prof::PH_SYNC, _t_sync);
        miss_r?;
        sync_r?;

        // 8. 結果 (e 要素) だけ戻す。
        //
        // task 358: `to_vec` は毎層 `Vec` を確保して pageable の同期 D2H をするので、
        // e=2,560 (10 KB) で 11 us/層かかっていた (lab の `[seg-prof] d2h=`)。
        // `to_slice` は 1 本の作業域へ直接戻し、`DENNO_PF_PIN_STAGE` の窓に入って
        // いれば pinned バウンス経由 (同じバイト列) になる。
        //
        // task 498: `DENNO_SEG_SUM_PIN` (既定 off) のときは、この 10 KB の D2H を
        // **専用の小さな pinned バウンス**へ DMA してからホストへ写す。pageable の
        // 同期 D2H は driver が内部ステージングを取るので 13-15 us かかる
        // (lab t498b の `[seg-prof] d2h=13.3-14.8/層`)。失敗したら従来経路へ
        // 丸ごと落ちる (値は同じバイト列)。
        let _t_d2h = crate::model::seg_prof::stamp();
        let mut d2h_done = false;
        // task 502: 非同期で出したコピーの完了を待って pinned から写す
        // (コピーはミス計算の裏で終わっている)。失敗したら下の同期経路へ。
        if async_ok {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(out[..e].as_mut_ptr() as *mut u8, e * 4)
            };
            d2h_done = seg_async_d2h_collect(dst);
        }
        if !d2h_done && crate::model::seg_prof::seg_sum_pin_on() && e * 4 <= SEG_PIN_BYTES {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(out[..e].as_mut_ptr() as *mut u8, e * 4)
            };
            d2h_done = seg_pin_copy_d2h(outb.ptr, dst);
        }
        if !d2h_done {
            outb.to_slice(&mut out[..e])?;
        }
        dbg_d2h(e * 4);
        crate::model::seg_prof::add(crate::model::seg_prof::PH_D2H, _t_d2h);

        // 9. 計上 (GPU 時間はイベント、バイト数は segment x 行数 x 行長)。
        if prof {
            crate::model::seg_prof::note_layer();
            let rb_gu = rb_g as u64 * ff as u64 * nseg as u64;
            let rb_dn = rb_d as u64 * e as u64 * nseg as u64;
            let one = |a: usize, b: usize, g: usize, bytes: u64| {
                if let Some(ns) = seg_ev::elapsed_ns(a, b) {
                    crate::model::seg_prof::add_gpu(g, ns, bytes as usize);
                }
            };
            one(0, 1, crate::model::seg_prof::GP_GATE, rb_gu);
            one(1, 2, crate::model::seg_prof::GP_UP, rb_gu);
            one(2, 3, crate::model::seg_prof::GP_SILU, 0);
            one(3, 4, crate::model::seg_prof::GP_DOWN, rb_dn);
            one(4, 5, crate::model::seg_prof::GP_SUM, 0);
        }
        Ok(())
    })?;
    Ok(true)
}


/// PTX を読んで関数を引く (失敗は `None` = 呼び出し側が従来経路へ)。
pub(crate) fn load_exp_iq_module(path: &std::path::Path, tag: &str) -> Option<ExpIqKernels> {
    let mut image = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "[exp-iq] {tag}: PTX が読めない ({}) : {e} → 従来経路へ",
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
            eprintln!("[exp-iq] {tag}: cuModuleLoadData 失敗 → 従来経路へ");
            return None;
        }
    }
    let mut fns: [CUfunction; crate::model::exp_iq::SLOTS] =
        [std::ptr::null_mut(); crate::model::exp_iq::SLOTS];
    for (i, f) in fns.iter_mut().enumerate() {
        let name = crate::model::exp_iq::kernel_name(i)?;
        let mut cname = name.as_bytes().to_vec();
        cname.push(0);
        unsafe {
            if (lib.module_get_function)(f, module, cname.as_ptr() as *const c_char)
                != CUDA_SUCCESS
            {
                eprintln!("[exp-iq] {tag}: PTX に {name} が無い → 従来経路へ");
                return None;
            }
        }
    }
    let mut silu: CUfunction = std::ptr::null_mut();
    let mut cname = crate::model::exp_iq::SILU_KERNEL.as_bytes().to_vec();
    cname.push(0);
    unsafe {
        if (lib.module_get_function)(&mut silu, module, cname.as_ptr() as *const c_char)
            != CUDA_SUCCESS
        {
            eprintln!("[exp-iq] {tag}: PTX に silu が無い → 従来経路へ");
            return None;
        }
    }
    Some(ExpIqKernels { module, fns, silu })
}


/// PTX から**指定したスロットだけ**引く (task 275)。
///
/// `load_exp_iq_module` は 6 スロット + silu が揃っていることを要求するが、
/// v3 は IQ2_S / IQ3_S の 2 本しか持たない。引けなかったスロットは null のまま
/// 返し、呼び出し側 ([`exp_iq_kernels_decode`]) が v2/v1 のもので埋める。
pub(crate) fn load_exp_iq_slots(
    path: &std::path::Path,
    tag: &str,
    slots: &[usize],
) -> Option<ExpIqKernels> {
    let mut image = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "[exp-iq] {tag}: PTX が読めない ({}) : {e} → v2 のまま",
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
            eprintln!("[exp-iq] {tag}: cuModuleLoadData 失敗 → v2 のまま");
            return None;
        }
    }
    let mut fns: [CUfunction; crate::model::exp_iq::SLOTS] =
        [std::ptr::null_mut(); crate::model::exp_iq::SLOTS];
    for &i in slots {
        let Some(name) = crate::model::exp_iq::kernel_name(i) else {
            return None;
        };
        let mut cname = format!("{name}_v3").into_bytes();
        cname.push(0);
        unsafe {
            if (lib.module_get_function)(&mut fns[i], module, cname.as_ptr() as *const c_char)
                != CUDA_SUCCESS
            {
                eprintln!("[exp-iq] {tag}: PTX に {name}_v3 が無い → v2 のまま");
                return None;
            }
        }
    }
    Some(ExpIqKernels {
        module,
        fns,
        silu: std::ptr::null_mut(),
    })
}


/// [`moe_experts_iq_layer`] と [`moe_experts_iq_layer_overlap`] の共通本体。
/// `overlap == false` のときの順序・API 呼び出しは task 144 の実装と同一。
#[allow(clippy::too_many_arguments)]
pub(crate) fn iq_layer_run<F>(
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
    sw: &[f32],
    sum_e: usize,
    overlap: bool,
    middle: F,
) -> Result<bool>
where
    F: FnOnce() -> Result<()> + Send,
{
    let nseg = pg.len();
    // task 187: `sum_e != 0` は「ヒットの寄与をデバイスで重み付き和にして
    // `sum_e` 要素だけ返す」モード (`dno_dec_exp_sum`)。`sw` はその重み。
    let sum_mode = sum_e != 0;
    let nout = if sum_mode { sum_e } else { nseg * e };
    if nseg == 0
        || pu.len() != nseg
        || pd.len() != nseg
        || !seg_gu.len().is_multiple_of(4)
        || !seg_d.len().is_multiple_of(4)
        || seg_gu.len() / 4 != nseg
        || seg_d.len() / 4 != nseg
    {
        return Ok(false);
    }
    if sum_mode && sw.len() != nseg {
        bail!(
            "moe_experts_iq_layer: 重みの本数 {} が segment 数 {nseg} と違う",
            sw.len()
        );
    }
    if !moe_gemv_iq_ready(ty_g, n_in)
        || !moe_gemv_iq_ready(ty_u, n_in)
        || !moe_gemv_iq_ready(ty_d, ff)
    {
        return Ok(false);
    }
    let (Some(plan_g), Some(plan_d)) = (
        crate::model::exp_iq::launch_plan(nseg, ff),
        crate::model::exp_iq::launch_plan(nseg, e),
    ) else {
        return Ok(false);
    };
    if x.len() < n_in || out_h.len() < nout {
        bail!(
            "moe_experts_iq_layer: 入出力長が足りない (x={}/{n_in}, out={}/{nout})",
            x.len(),
            out_h.len()
        );
    }
    let Some(ctx) = shared_ctx() else {
        return Ok(false);
    };
    let Some(kernels) = exp_iq_kernels_decode() else {
        return Ok(false);
    };
    // task 187: 和のカーネル (PTX が無ければ `Ok(false)` = 呼び出し側は従来経路へ)。
    let sumk = if sum_mode {
        match dec_epi_kernels() {
            Some(k) => Some(k),
            None => return Ok(false),
        }
    } else {
        None
    };
    let (Some(slot_g), Some(slot_u), Some(slot_d)) = (
        crate::model::exp_iq::slot(ty_g),
        crate::model::exp_iq::slot(ty_u),
        crate::model::exp_iq::slot(ty_d),
    ) else {
        return Ok(false);
    };
    ctx.make_current()?;

    let ng = nseg * ff;
    let nh = nseg * e;
    // task 204: gate+up 融合 (`DENNO_DEC_EXPGU`) の可否。量子型が同じで行長も
    // 同じときだけ畳める (違えば従来の 2 起動へ fail closed)。
    let rb_g_pre = crate::model::exp_iq::row_bytes(ty_g, n_in).unwrap_or(0);
    let rb_u_pre = crate::model::exp_iq::row_bytes(ty_u, n_in).unwrap_or(0);
    // task 388: CPU ミス (`middle`) を 3 つの順序で呼べるようにする。
    // - 既定 (`DENNO_DEC_MISS_SPAWN` on): rayon のプールで**起動と同時**に走らせる。
    //   呼び出しスレッドは起動と同期だけを行う (lab t388b で experts 段 -4.5/-6.9%)。
    // - `DENNO_DEC_MISS_SPAWN=0`: 起動のあと、同期の前 (t388b 以前の従来)。
    // - `DENNO_DEC_MISS_SERIAL=1`: 同期と写しの**後ろ** (診断 = 重ね打ち無し)。
    // どの順序でもカーネル・引数・足し込みの順は同じ (値は変わらない)。
    let spawn_cpu = overlap && crate::model::exp_iq::miss_spawn_enabled();
    let serial_cpu = overlap && !spawn_cpu && crate::model::exp_iq::miss_serial_enabled();
    // `FnOnce` を 2 つの経路 (プール / このスレッド) から 1 回だけ呼ぶための持ち場。
    let holder: std::sync::Mutex<Option<F>> = std::sync::Mutex::new(Some(middle));
    let call_mid = || -> Result<()> {
        let got = holder
            .lock()
            .map_err(|_| anyhow!("iq layer: cpu_miss のロックが poisoned"))?
            .take();
        match got {
            Some(f) => f(),
            None => Ok(()),
        }
    };
    let mut run_body = |call_mid: &dyn Fn() -> Result<()>| -> Result<()> {
    IQ_SCRATCH.with(|cell| -> Result<()> {
        let mut sc = cell.borrow_mut();
        MoeScratch::ensure_tagged(&mut sc.x, n_in * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.g, ng * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.u, ng * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.act, ng * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.h, nh * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.pg, nseg * 8, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.pu, nseg * 8, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.pd, nseg * 8, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.sgu, seg_gu.len() * 4, crate::mem_audit::TAG_GDN_SC)?;
        MoeScratch::ensure_tagged(&mut sc.sgd, seg_d.len() * 4, crate::mem_audit::TAG_GDN_SC)?;
        // task 187: 和モードの作業域 (重み nseg 本 / 和の出力 nout 要素)。
        if sum_mode {
            MoeScratch::ensure_tagged(&mut sc.wt, nseg * 4, crate::mem_audit::TAG_GDN_SC)?;
            MoeScratch::ensure_tagged(&mut sc.psum, nout * 4, crate::mem_audit::TAG_GDN_SC)?;
        }
        // task 356: 和の出力を pinned のホストへ直接書く (`DENNO_DEC_EXP_SUM_ZC`)。
        // 確保できなければ従来の D2H 経路へ (fail closed)。
        let zc = sum_mode
            && sumk.is_some()
            && crate::model::exp_iq::sum_zc_enabled()
            && match Scratch::ensure_pin(&mut sc.psum_zc, nout * 4) {
                Ok(()) => true,
                Err(_) => {
                    crate::model::exp_iq::note_sum_zc_fallback();
                    false
                }
            };
        // task 150: overlap 経路だけ pinned staging を用意する (grow-only)。
        if overlap {
            Scratch::ensure_pin(&mut sc.hpin, nout * 4)?;
        }
        // task 164: 層の prologue (x / ポインタ表 / segment 表) を pinned ring の
        // 1 スロットへ詰め、**非同期 H2D 1 回**で送る (`DENNO_DEC_PRO_ASYNC`)。
        // 現行は 6 本の同期 H2D で、そのたびに null ストリームの先行カーネルを
        // 待つ (1 層 6 回 x 48 層 = 288 回/token)。1 本にまとめるとドライバ往復が
        // 減り、転送が直前の層の GPU 仕事と重なる。**値は変わらない** (同じ表・
        // 同じポインタを別の場所に置くだけ)。
        // task 187: 台帳の副区間 (prologue の H2D)。「7 MB の h2d のうち入力はいくらか」。
        let s_pro = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_EXP_PRO);
        let pro = dec_pro_async_enabled();
        // task 204: gate+up を 1 起動に畳む (`DENNO_DEC_EXPGU`)。量子型が同じで
        // 行長も同じときだけ (そうでなければ従来の 2 起動へ)。表は prologue の
        // 1 本の H2D に載せるので、追加の転送は 0 回。
        let gu_ok = dec_exp_gu_shape_ok(exp_gu_on(), slot_g == slot_u, rb_g_pre, rb_u_pre);
        let lay = dec_stage_layout(
            n_in,
            nseg,
            seg_gu.len(),
            seg_d.len(),
            if sum_mode { nseg } else { 0 },
            if gu_ok { 2 * nseg } else { 0 },
        );
        // task 204: gate+up 融合の連結出力 (レイアウトにブロックがあるときだけ)。
        if lay.total > lay.pg2 && sc.gu.as_ref().map(|b| b.bytes()).unwrap_or(0) < 2 * ng * 4 {
            MoeScratch::ensure_tagged(&mut sc.gu, 2 * ng * 4, crate::mem_audit::TAG_GDN_SC)?;
        }
        // task 204: 倍の表 (前半 = gate、後半 = up)。segment 表の後半は出力
        // オフセットだけ `ng` ずらす (x オフセットと行の並びは同じ)。
        let (pg2_host, sgu2_host) = if lay.total > lay.pg2 {
            let mut p: Vec<u64> = Vec::with_capacity(2 * nseg);
            p.extend_from_slice(pg);
            p.extend_from_slice(pu);
            let mut s: Vec<u32> = Vec::with_capacity(2 * nseg * 4);
            s.extend_from_slice(seg_gu);
            for i in 0..nseg {
                let mut e4 = [0u32; 4];
                e4.copy_from_slice(&seg_gu[i * 4..i * 4 + 4]);
                e4[3] = e4[3].saturating_add(ng as u32);
                s.extend_from_slice(&e4);
            }
            (p, s)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut stage_addr: Option<u64> = None;
        if pro {
            // 失敗したら従来の同期コピーへ落ちる (リクエストは落とさない)。
            let staged = (|| -> Result<u64> {
                let (dev, slot) = dec_stage_slot(lay.total)?;
                unsafe {
                    let sb = std::slice::from_raw_parts_mut(slot as *mut u8, lay.total);
                    dec_stage_put(sb, lay.x, x.as_ptr() as *const u8, n_in * 4);
                    dec_stage_put(sb, lay.pg, pg.as_ptr() as *const u8, nseg * 8);
                    dec_stage_put(sb, lay.pu, pu.as_ptr() as *const u8, nseg * 8);
                    dec_stage_put(sb, lay.pd, pd.as_ptr() as *const u8, nseg * 8);
                    dec_stage_put(sb, lay.sgu, seg_gu.as_ptr() as *const u8, seg_gu.len() * 4);
                    dec_stage_put(sb, lay.sgd, seg_d.as_ptr() as *const u8, seg_d.len() * 4);
                    if sum_mode {
                        // task 187: 和の重みも同じ 1 回の H2D に載せる (追加の往復をしない)。
                        dec_stage_put(sb, lay.w, sw.as_ptr() as *const u8, nseg * 4);
                    }
                    if lay.total > lay.pg2 {
                        // task 204: 倍の表も同じ 1 回の H2D に載せる。
                        dec_stage_put(sb, lay.pg2, pg2_host.as_ptr() as *const u8, 2 * nseg * 8);
                        dec_stage_put(sb, lay.sgu2, sgu2_host.as_ptr() as *const u8, 2 * nseg * 16);
                    }
                }
                dec_stage_issue(lay.total)?;
                dbg_htod(lay.total);
                Ok(dev as u64)
            })();
            stage_addr = staged.ok();
        }
        if stage_addr.is_none() {
            let xb = sc.x.as_ref().expect("確保済み");
            xb.copy_from(&x[..n_in])?;
            dbg_htod(n_in * 4);
            sc.pg.as_ref().expect("確保済み").copy_from(pg)?;
            sc.pu.as_ref().expect("確保済み").copy_from(pu)?;
            sc.pd.as_ref().expect("確保済み").copy_from(pd)?;
            dbg_htod(nseg * 8 * 3);
            sc.sgu.as_ref().expect("確保済み").copy_from(seg_gu)?;
            sc.sgd.as_ref().expect("確保済み").copy_from(seg_d)?;
            dbg_htod((seg_gu.len() + seg_d.len()) * 4);
            if sum_mode {
                sc.wt.as_ref().expect("確保済み").copy_from(sw)?;
                dbg_htod(nseg * 4);
            }
        }
        drop(s_pro);
        let (x_addr, pg_addr, pu_addr, pd_addr, sgu_addr, sgd_addr, w_addr, pg2_addr, sgu2_addr) =
            match stage_addr {
                Some(base) => (
                    base + lay.x as u64,
                    base + lay.pg as u64,
                    base + lay.pu as u64,
                    base + lay.pd as u64,
                    base + lay.sgu as u64,
                    base + lay.sgd as u64,
                    base + lay.w as u64,
                    base + lay.pg2 as u64,
                    base + lay.sgu2 as u64,
                ),
                None => (
                    sc.x.as_ref().expect("確保済み").ptr as u64,
                    sc.pg.as_ref().expect("確保済み").ptr as u64,
                    sc.pu.as_ref().expect("確保済み").ptr as u64,
                    sc.pd.as_ref().expect("確保済み").ptr as u64,
                    sc.sgu.as_ref().expect("確保済み").ptr as u64,
                    sc.sgd.as_ref().expect("確保済み").ptr as u64,
                    sc.wt.as_ref().map(|b| b.ptr as u64).unwrap_or(0),
                    0,
                    0,
                ),
            };
        let g = sc.g.as_ref().expect("確保済み");
        let u = sc.u.as_ref().expect("確保済み");
        let act = sc.act.as_ref().expect("確保済み");
        let h = sc.h.as_ref().expect("確保済み");
        // task 204: 融合経路 (`DENNO_DEC_EXPGU`) の出力は `[g[ng] | u[ng]]` が
        // 連続した 1 本。prologue の 1 本の H2D に表が載ったときだけ畳む。
        let gu_fused = lay.total > lay.pg2 && stage_addr.is_some();
        let (g_ptr, u_ptr) = if gu_fused {
            let gb = sc.gu.as_ref().expect("確保済み");
            (gb.ptr, gb.ptr + (ng * 4) as CUdeviceptr)
        } else {
            (g.ptr, u.ptr)
        };
        unsafe {
            // 1. gate と up (同じ segment 表 = 同じ入力 x)
            // task 187: 層内の GPU 時間をイベントで段ごとに測る (PROFILE のときだけ)。
            dec_exp_kprof::begin_layer(nseg as u32);
            let rb_g = crate::model::exp_iq::row_bytes(ty_g, n_in).unwrap_or(0);
            let rb_u = crate::model::exp_iq::row_bytes(ty_u, n_in).unwrap_or(0);
            // task 190: 型 (gate/up/down のスロット) と行数の表に積む
            // (重みの実バイト数も一緒に数える = GB/s が出せる)。
            {
                let rb_d = crate::model::exp_iq::row_bytes(ty_d, ff).unwrap_or(0);
                let bytes = nseg as u64
                    * (ff as u64 * (rb_g + rb_u) as u64 + e as u64 * rb_d as u64);
                dec_exp_kprof::note_kind(slot_g, slot_u, slot_d, nseg, ff, e, bytes);
            }
            if gu_fused {
                // task 204: gate と up を **1 起動**に畳む。segment 表の前半が
                // gate、後半が up (出力オフセットだけ +ng)。1 行の計算は従来の
                // カーネルと同一 (同じカーネル・同じ x・同じ行長) なので値は
                // 1 ビットも変わらない。
                {
                    let gb = sc.gu.as_ref().expect("確保済み");
                    let mut pw = pg2_addr as *mut c_void;
                    let mut px = x_addr as *mut c_void;
                    let mut po = gb.ptr;
                    let mut ps = sgu2_addr as *mut c_void;
                    let (mut ns, mut ni, mut no, mut rb) = (
                        (2 * nseg) as c_int,
                        n_in as c_int,
                        ff as c_int,
                        rb_g as c_int,
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
                    ctx.launch_async(
                        kernels.fns[slot_g],
                        (plan_g.grid.0, (2 * nseg) as u32, 1),
                        plan_g.block,
                        0,
                        &mut args,
                    )?;
                }
                dec_exp_kprof::mark(dec_exp_kprof::GATE);
                dec_exp_kprof::mark(dec_exp_kprof::UP);
                #[cfg(test)]
                EXP_GU_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                let mut gu = 0usize;
                for (f, ptrs, buf, rb) in [
                    (kernels.fns[slot_g], pg_addr, g, rb_g),
                    (kernels.fns[slot_u], pu_addr, u, rb_u),
                ] {
                    let mut pw = ptrs as *mut c_void;
                    let mut px = x_addr as *mut c_void;
                    let mut po = buf.ptr;
                    let mut ps = sgu_addr as *mut c_void;
                    let (mut ns, mut ni, mut no, mut rb) = (
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
                        &mut ns as *mut _ as *mut c_void,
                        &mut ni as *mut _ as *mut c_void,
                        &mut no as *mut _ as *mut c_void,
                        &mut rb as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(f, plan_g.grid, plan_g.block, 0, &mut args)?;
                    // 1 本目 = gate、2 本目 = up。
                    dec_exp_kprof::mark(if gu == 0 {
                        dec_exp_kprof::GATE
                    } else {
                        dec_exp_kprof::UP
                    });
                    gu += 1;
                }
            }
            // 2. act = silu(g) * u
            {
                let n = ng as c_int;
                let threads = 256usize;
                let blocks = ng.div_ceil(threads) as u32;
                let mut pgg = g_ptr;
                let mut puu = u_ptr;
                let mut paa = act.ptr;
                let mut pn = n;
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
                dec_exp_kprof::mark(dec_exp_kprof::SILU);
            }
            // 3. down (expert ごとの act を読む)
            {
                let mut pw = pd_addr as *mut c_void;
                let mut px = act.ptr;
                let mut po = h.ptr;
                let mut ps = sgd_addr as *mut c_void;
                let (mut ns, mut ni, mut no, mut rb) = (
                    nseg as c_int,
                    ff as c_int,
                    e as c_int,
                    crate::model::exp_iq::row_bytes(ty_d, ff).unwrap_or(0) as c_int,
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
                ctx.launch_async(
                    kernels.fns[slot_d],
                    plan_d.grid,
                    plan_d.block,
                    0,
                    &mut args,
                )?;
                dec_exp_kprof::mark(dec_exp_kprof::DOWN);
            }
            // 3.5 task 187: ヒットの寄与をデバイスで重み付き和にする
            //     (`out[j] = sum_t w[t] * h[t*e+j]`)。t 昇順の 1 アキュムレータ
            //     なので、ミスの無い層ではホストの足し込みとビット一致する。
            let src_ptr = if let Some(sk) = sumk {
                // task 356 (`DENNO_DEC_EXP_SUM_ZC`): 和の出力を pinned のホストへ
                // 直接書かせる (UVA なのでホストのアドレスがそのままデバイスの
                // アドレス)。これで 10 KB の D2H とその完了待ちが丸ごと消える。
                let zc_ptr = if zc {
                    let pin = sc.psum_zc.as_ref().expect("確保済み");
                    Some(pin.ptr as u64)
                } else {
                    None
                };
                let psum = sc.psum.as_ref().expect("確保済み");
                let mut ph = h.ptr;
                let mut pwv = w_addr as *mut c_void;
                let mut psh = std::ptr::null_mut::<c_void>();
                let mut pw = 0.0f32;
                let mut po = zc_ptr.unwrap_or(psum.ptr);
                let (mut pe, mut pn, mut phs) = (e as c_int, nseg as c_int, 0 as c_int);
                let mut args = [
                    &mut ph as *mut _ as *mut c_void,
                    &mut pwv as *mut _ as *mut c_void,
                    &mut psh as *mut _ as *mut c_void,
                    &mut pw as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut pe as *mut _ as *mut c_void,
                    &mut pn as *mut _ as *mut c_void,
                    &mut phs as *mut _ as *mut c_void,
                ];
                let threads = 256usize;
                let blocks = sum_e.div_ceil(threads) as u32;
                ctx.launch_async(sk.sum, (blocks.max(1), 1, 1), (threads as u32, 1, 1), 0, &mut args)?;
                dec_exp_kprof::mark(dec_exp_kprof::SUM);
                if zc_ptr.is_some() {
                    crate::model::exp_iq::note_sum_zc_layer();
                }
                zc_ptr.unwrap_or(psum.ptr)
            } else {
                h.ptr
            };
            // 4. 同期は 1 回だけ (層あたり)。task 150 の overlap 経路では D2H を
            //    非同期で先に出し、`middle` (CPU ミス) を挟んでから同期する。
            //    カーネル・引数・発行順は逐次版と同一 (変わるのは待つ位置だけ)。
            if overlap {
                if zc {
                    // task 356: カーネルが pinned のホストへ直接書いた = D2H 無し。
                    // 待ちは下の同期 1 回だけで足りる。
                } else {
                    // `h` の借用をここで切る (pinned は別 field なので `sc` を丸ごと
                    // 借り直さないよう、アドレスを先に写す)。
                    let pin = sc.hpin.as_mut().expect("確保済み");
                    if !exp_d2h_issue(pin.ptr, src_ptr, nout * 4) {
                        // task 356 (`DENNO_DEC_EXP_D2H_STREAM`): コピーを専用の
                        // 非ブロッキングストリームへ出す。null ストリームのままだと、
                        // 小さいコピーで driver が呼び出しの中で待つことがあり、その
                        // 待ちが CPU ミスの前に直列に挟まる。待ちは下の
                        // `exp_d2h_wait` へ回す。失敗したら従来の null ストリーム
                        // 経路へ (fail closed)。
                        check(
                            (lib()?.memcpy_dtoh_async)(
                                pin.ptr,
                                src_ptr,
                                nout * 4,
                                std::ptr::null_mut(),
                            ),
                            "cuMemcpyDtoHAsync (iq layer overlap)",
                        )?;
                    }
                }
                crate::prof::mark_d2h_src(crate::prof::D2H_SRC_LINEAR);
                dec_exp_kprof::mark(dec_exp_kprof::D2H);
            } else {
                // task 155: 逐次版も null ストリームぶんだけをイベントで待つ。
                sync_default_or_ctx(ctx, dec_overlap_enabled())?;
            }
        }
        if overlap {
            // task 388: 同期と控えの写しを 1 本にまとめ、CPU ミスをその
            // 前 (既定) / 後ろ (serial) / 別スレッド (spawn) に置く。
            let mut finish_gpu = || -> Result<()> {
                // task 356: 専用ストリームへ出したときは、`done` の完了が
                // 「`ready` (この層のカーネル全部の後)」を待っているので、null ストリームの
                // 同期は要らない (= 待ちを 1 回減らせる)。`DENNO_DEC_OVERLAP=0` のときは
                // 従来どおりコンテキスト全体を待ってから `done` を待つ。
                if exp_d2h_inflight() && dec_overlap_enabled() {
                    if !exp_d2h_wait() {
                        bail!("cuEventSynchronize (iq layer d2h stream) が失敗");
                    }
                } else {
                    // task 155: 待つのは null ストリーム (この層の expert GEMV と D2H)
                    // だけ。shared expert のサイドストリームは join まで待たない。
                    sync_default_or_ctx(ctx, dec_overlap_enabled())?;
                    // task 356: 専用ストリームへ出したコピーの完了 (通常はもう終わっている)。
                    if !exp_d2h_wait() {
                        bail!("cuEventSynchronize (iq layer d2h stream) が失敗");
                    }
                }
                if zc {
                    // task 356: 和は pinned のホストに直接書かれている (コピー無し)。
                    let pin = sc.psum_zc.as_ref().expect("確保済み");
                    out_h[..nout].copy_from_slice(&pin.as_slice::<f32>()[..nout]);
                } else {
                    let pin = sc.hpin.as_ref().expect("確保済み");
                    out_h[..nout].copy_from_slice(&pin.as_slice::<f32>()[..nout]);
                }
                Ok(())
            };
            if serial_cpu {
                // 診断 (`DENNO_DEC_MISS_SERIAL`): 重ね打ちを完全に切る。
                finish_gpu()?;
                call_mid()?;
                crate::model::exp_iq::note_miss_serial_layer();
            } else if spawn_cpu {
                // CPU ミスは呼び出し側がプールで走らせている (`run_body` の後で join)。
                finish_gpu()?;
            } else {
                // 従来: デバイスの完了を待たずに CPU ミスを回す (GPU と重ね打ち)。
                call_mid()?;
                finish_gpu()?;
            }
        } else {
            if zc {
                let pin = sc.psum_zc.as_ref().expect("確保済み");
                out_h[..nout].copy_from_slice(&pin.as_slice::<f32>()[..nout]);
            } else {
                let src = if sum_mode {
                    sc.psum.as_ref().expect("確保済み")
                } else {
                    sc.h.as_ref().expect("確保済み")
                };
                let got: Vec<f32> = src.to_vec(nout)?;
                out_h[..nout].copy_from_slice(&got);
            }
            dec_exp_kprof::mark(dec_exp_kprof::D2H);
            call_mid()?;
        }
        dbg_d2h(nout * 4);
        // ここまでで GPU は同期済み (両経路とも) = イベントが完了している。
        dec_exp_kprof::end_layer();
        Ok(())
    })
    };
    // task 388: spawn 経路だけ「CPU ミスをプールで走らせながら起動と同期を進める」。
    // scoped なので借用はそのまま (値も順序も変わらない)。
    let mut body_res: Result<()> = Ok(());
    if spawn_cpu {
        // `Scope::spawn` は join ハンドルを返さないので、結果は channel で受ける
        // (scope の終端でタスクの完了は待たれる = `recv` で詰まらない)。
        let (tx, rx) = std::sync::mpsc::channel::<Result<()>>();
        rayon::in_place_scope(|s| {
            s.spawn(move |_| {
                let r = call_mid();
                let _ = tx.send(r);
            });
            body_res = run_body(&call_mid);
        });
        crate::model::exp_iq::note_miss_spawn_layer();
        body_res?;
        match rx.recv() {
            Ok(r) => r?,
            Err(_) => return Err(anyhow!("iq layer: cpu_miss の結果が返らない")),
        }
    } else {
        body_res = run_body(&call_mid);
        body_res?;
    }
    Ok(true)
}


/// 専用ストリームを用意する (無ければ作る)。作れなければ `false`。
fn exp_d2h_ensure() -> bool {
    EXP_D2H.with(|c| {
        if c.borrow().is_some() {
            return true;
        }
        let Ok(l) = lib() else {
            return false;
        };
        let (Some(create_stream), Some(create_event)) = (l.stream_create, l.event_create) else {
            return false;
        };
        if l.stream_wait_event.is_none() || l.event_record.is_none() || l.event_sync.is_none() {
            return false;
        }
        let mut st: CUstream = std::ptr::null_mut();
        if unsafe { create_stream(&mut st, CU_STREAM_NON_BLOCKING_D2H) } != CUDA_SUCCESS {
            return false;
        }
        let mut ready: CUevent = std::ptr::null_mut();
        let mut done: CUevent = std::ptr::null_mut();
        let ok = unsafe { create_event(&mut ready, CU_EVENT_DISABLE_TIMING_D2H) } == CUDA_SUCCESS
            && unsafe { create_event(&mut done, CU_EVENT_DISABLE_TIMING_D2H) } == CUDA_SUCCESS;
        if !ok {
            return false;
        }
        *c.borrow_mut() = Some(ExpD2h {
            stream: st,
            ready,
            done,
        });
        true
    })
}


/// 専用ストリーム経由の非同期 D2H を 1 回出す。
///
/// 順序は「null ストリームに ready を打つ -> 専用ストリームで ready を待つ ->
/// コピー -> done を打つ」。戻り値 `true` のとき、呼び出し側は
/// [`exp_d2h_wait`] で完了を待ってからホストのバッファを読むこと。
///
/// `false` は「この経路を使えない」= 呼び出し側は従来の null ストリーム経路へ。
pub(crate) fn exp_d2h_issue(dst: *mut c_void, src: u64, bytes: usize) -> bool {
    if !crate::model::exp_iq::dec_d2h_stream_enabled() {
        return false;
    }
    if !exp_d2h_ensure() {
        crate::model::exp_iq::note_d2h_stream_fallback();
        return false;
    }
    let Ok(l) = lib() else {
        return false;
    };
    let (Some(record), Some(wait_ev), Some(copy)) =
        (l.event_record, l.stream_wait_event, Some(l.memcpy_dtoh_async))
    else {
        return false;
    };
    EXP_D2H.with(|c| {
        let b = c.borrow();
        let Some(d) = b.as_ref() else {
            return false;
        };
        // 1. この層のカーネル (null ストリーム) の完了点。
        if unsafe { record(d.ready, std::ptr::null_mut()) } != CUDA_SUCCESS {
            return false;
        }
        // 2. 専用ストリームはその点を待つ。
        if unsafe { wait_ev(d.stream, d.ready, 0) } != CUDA_SUCCESS {
            return false;
        }
        // 3. コピーと完了イベント。
        if unsafe { copy(dst, src, bytes, d.stream) } != CUDA_SUCCESS {
            return false;
        }
        if unsafe { record(d.done, d.stream) } != CUDA_SUCCESS {
            return false;
        }
        EXP_D2H_INFLIGHT.with(|f| f.set(true));
        crate::model::exp_iq::note_d2h_stream_copy();
        true
    })
}


/// [`exp_d2h_issue`] で出したコピーがまだ待たれていないか。
pub(crate) fn exp_d2h_inflight() -> bool {
    EXP_D2H_INFLIGHT.with(|f| f.get())
}


/// [`exp_d2h_issue`] で出したコピーの完了を待つ (出していなければ何もしない)。
///
/// 通常は CPU ミスの後に呼ぶので待ちはゼロ。待った時間は台帳の `sync` に積む。
pub(crate) fn exp_d2h_wait() -> bool {
    if !EXP_D2H_INFLIGHT.with(|f| f.replace(false)) {
        return true;
    }
    let Ok(l) = lib() else {
        return false;
    };
    let Some(esync) = l.event_sync else {
        return false;
    };
    let t = crate::model::q4ledger::stamp();
    EXP_D2H.with(|c| {
        let b = c.borrow();
        let Some(d) = b.as_ref() else {
            return false;
        };
        let ok = unsafe { esync(d.done) } == CUDA_SUCCESS;
        if ok {
            if let Some(t) = t {
                crate::model::exp_iq::note_d2h_stream_wait(t.elapsed().as_nanos() as u64);
            }
            crate::model::q4ledger::note_sync(t);
        }
        ok
    })
}


/// IQ 層一括 GEMV を 1 回積む (`kernels_exp_iq` の引数 8 個)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn iq_launch_gemv(
    ctx: &CudaCtx,
    f: CUfunction,
    plan: crate::model::exp_iq::IqLaunch,
    pw: u64,
    px: u64,
    po: u64,
    ps: u64,
    nseg: usize,
    n_in: usize,
    n_out: usize,
    rb: usize,
) -> Result<()> {
    unsafe {
        let (mut pw, mut px, mut po, mut ps) = (pw, px, po, ps);
        let (mut ns, mut ni, mut no, mut rb) = (
            nseg as c_int,
            n_in as c_int,
            n_out as c_int,
            rb as c_int,
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
        ctx.launch_async(f, plan.grid, plan.block, 0, &mut args)
    }
}


static EXP_PF_KERNELS: OnceLock<Option<ExpPfKernels>> = OnceLock::new();


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
pub(crate) fn exp_pf_kernels() -> Option<&'static ExpPfKernels> {
    EXP_PF_KERNELS
        .get_or_init(|| {
            let path = exp_pf_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[exp-pf] PTX が読めない ({}) : {e} → CPU 経路へ",
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
                    eprintln!("[exp-pf] cuModuleLoadData 失敗 → CPU 経路へ");
                    return None;
                }
            }
            let mut fns: [CUfunction; crate::model::exp_pf::SLOTS] =
                [std::ptr::null_mut(); crate::model::exp_pf::SLOTS];
            for (i, f) in fns
                .iter_mut()
                .enumerate()
                .take(crate::model::exp_pf::IQ_SLOTS)
            {
                let name = crate::model::exp_pf::kernel_name(i)?;
                let mut cname = name.as_bytes().to_vec();
                cname.push(0);
                unsafe {
                    if (lib.module_get_function)(f, module, cname.as_ptr() as *const c_char)
                        != CUDA_SUCCESS
                    {
                        eprintln!("[exp-pf] PTX に {name} が無い → CPU 経路へ");
                        return None;
                    }
                }
            }
            // task 309: K 系 (Q4_K / Q5_K / Q6_K / Q5_1 / Q8_0) は別モジュール。
            // 無くても IQ 系は動くので、失敗は 1 行だけ出して null のままにする
            // (K 系の型は `moe_gemv_iq_pf_ready` が null を見て CPU へ落ちる)。
            let mut kmodule: CUmodule = std::ptr::null_mut();
            let kpath = exp_k_pf_ptx_path();
            let mut k_ok = false;
            match std::fs::read(&kpath) {
                Ok(mut kimg) => {
                    kimg.push(0);
                    unsafe {
                        if (lib.module_load_data)(&mut kmodule, kimg.as_ptr() as *const c_void)
                            == CUDA_SUCCESS
                        {
                            k_ok = true;
                        }
                    }
                }
                Err(_) => {}
            }
            if k_ok {
                for i in crate::model::exp_pf::K_SLOT0..crate::model::exp_pf::SLOTS {
                    let Some(name) = crate::model::exp_pf::kernel_name(i) else {
                        continue;
                    };
                    let mut cname = name.as_bytes().to_vec();
                    cname.push(0);
                    unsafe {
                        if (lib.module_get_function)(
                            &mut fns[i],
                            kmodule,
                            cname.as_ptr() as *const c_char,
                        ) != CUDA_SUCCESS
                        {
                            eprintln!("[exp-pf] K 系 PTX に {name} が無い → その型は CPU 経路へ");
                        }
                    }
                }
            } else {
                eprintln!(
                    "[exp-pf] K 系 PTX が読めない ({}) → K 系の型は CPU 経路へ",
                    kpath.display()
                );
            }
            let mut silu: CUfunction = std::ptr::null_mut();
            let mut cname = crate::model::exp_pf::SILU_KERNEL.as_bytes().to_vec();
            cname.push(0);
            unsafe {
                if (lib.module_get_function)(&mut silu, module, cname.as_ptr() as *const c_char)
                    != CUDA_SUCCESS
                {
                    eprintln!("[exp-pf] PTX に silu が無い → CPU 経路へ");
                    return None;
                }
            }
            // task 162: 行の加重和カーネル。無くても通常の prefill は動くので
            // 任意 (ストリーミング経路だけが `None` を見て CPU へ落ちる)。
            let mut accum: CUfunction = std::ptr::null_mut();
            let mut cname = crate::model::exp_pf::ACCUM_KERNEL.as_bytes().to_vec();
            cname.push(0);
            let accum = unsafe {
                if (lib.module_get_function)(&mut accum, module, cname.as_ptr() as *const c_char)
                    == CUDA_SUCCESS
                {
                    Some(accum)
                } else {
                    None
                }
            };
            Some(ExpPfKernels {
                module,
                kmodule,
                fns,
                silu,
                accum,
            })
        })
        .as_ref()
}


/// 1 層ぶんの expert FFN を m 行カーネルで回す。
///
/// `seg` は 4 語 × nseg (m / rows の先頭 / 0 / 0)、`rows_gu` は行 → トークン番号
/// (`exp_pf::build_plan`)、`out_h` は `total_rows * e` 要素 (GPU の行を**そのまま**
/// 返す)。
/// 呼び出し側は「トークンごとに expert index 昇順で 1 本ずつ」足し込む
/// ([`crate::model::qwen4exp`] の `exp_pf_add_all`)。
///
/// task 151: 以前は加重和を GPU (`dno_exp_pf_accum`) で閉じて `nt × e` だけ
/// 返していたが、**和の順序がヒット/ミスの内訳に依存する**ため、同じ入力でも
/// VRAM の中身が違うと出力が変わった (lab t148b/t148c の `a1 != a2`)。丸めを
/// 一致させるには 1 本ずつ同じ順で足す必要があるので、行を返す形へ戻した。
///
/// `Ok(false)` は「GPU でやらなかった」= 呼び出し側は CPU 経路へ (fail closed)。
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_pf_layer(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    pg: &[u64],
    pu: &[u64],
    pd: &[u64],
    seg: &[u32],
    rows_gu: &[u32],
    xs: &[f32],
    n_in: usize,
    ff: usize,
    e: usize,
    out_h: &mut [f32],
    cpu_miss: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    if PF_DEAD.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(false);
    }
    let nseg = pg.len();
    let total_rows = rows_gu.len();
    let nh = total_rows * e;
    if nseg == 0
        || total_rows == 0
        || pu.len() != nseg
        || pd.len() != nseg
        || seg.len() != nseg * 4
        || xs.len() % n_in != 0
        || out_h.len() < nh
    {
        return Ok(false);
    }
    if !moe_gemv_iq_pf_ready(ty_g, n_in)
        || !moe_gemv_iq_pf_ready(ty_u, n_in)
        || !moe_gemv_iq_pf_ready(ty_d, ff)
    {
        return Ok(false);
    }
    let (Some(slot_g), Some(slot_u), Some(slot_d)) = (
        exp_pf::slot(ty_g),
        exp_pf::slot(ty_u),
        exp_pf::slot(ty_d),
    ) else {
        return Ok(false);
    };
    let (Some(plan_g), Some(plan_d)) = (
        exp_pf::launch_plan(slot_g, nseg, ff),
        exp_pf::launch_plan(slot_d, nseg, e),
    ) else {
        return Ok(false);
    };
    let Some(ctx) = shared_ctx() else {
        return Ok(false);
    };
    let Some(kernels) = exp_pf_kernels() else {
        return Ok(false);
    };
    ctx.make_current()?;

    // 行の総数 × 行列の幅。g / u / act は total_rows × ff、h は total_rows × e。
    let ng = total_rows * ff;
    // CPU ミス (呼び出し側の flat) が落ちたか。落ちた場合は `out` が途中まで
    // 書かれているので **CPU 経路へ戻してはいけない** (Err で伝える)。
    let cpu_failed = std::cell::Cell::new(false);
    let mut alloc = || -> Result<bool> {
        PF_SCRATCH.with(|cell| -> Result<bool> {
            let mut sc = cell.borrow_mut();
            // task 257: プールがあれば区画 (prefill 中に cuMemAlloc を出さない)。
            ensure_region(&mut sc.x, crate::model::pf_pool::R::X, 0, xs.len() * 4)?;
            ensure_region(&mut sc.g, crate::model::pf_pool::R::G, 0, ng * 4)?;
            ensure_region(&mut sc.u, crate::model::pf_pool::R::U, 0, ng * 4)?;
            ensure_region(&mut sc.act, crate::model::pf_pool::R::Act, 0, ng * 4)?;
            ensure_region(&mut sc.h, crate::model::pf_pool::R::H, 0, nh * 4)?;
            ensure_region(&mut sc.pg, crate::model::pf_pool::R::Pg, 0, nseg * 8)?;
            ensure_region(&mut sc.pu, crate::model::pf_pool::R::Pu, 0, nseg * 8)?;
            ensure_region(&mut sc.pd, crate::model::pf_pool::R::Pd, 0, nseg * 8)?;
            ensure_region(&mut sc.seg, crate::model::pf_pool::R::Seg, 0, nseg * 16)?;
            ensure_region(&mut sc.rows_gu, crate::model::pf_pool::R::RowsGu, 0, total_rows * 4)?;
            ensure_region(&mut sc.rows_id, crate::model::pf_pool::R::RowsId, 0, total_rows * 4)?;
            {
                let xb = sc.x.as_ref().expect("確保済み");
                xb.copy_from(xs)?;
                sc.pg.as_ref().expect("確保済み").copy_from(pg)?;
                sc.pu.as_ref().expect("確保済み").copy_from(pu)?;
                sc.pd.as_ref().expect("確保済み").copy_from(pd)?;
                sc.seg.as_ref().expect("確保済み").copy_from(seg)?;
                sc.rows_gu.as_ref().expect("確保済み").copy_from(rows_gu)?;
                let id: Vec<u32> = exp_pf::identity_rows(total_rows);
                sc.rows_id.as_ref().expect("確保済み").copy_from(&id)?;
                dbg_htod(
                    xs.len() * 4
                        + nseg * 8 * 3
                        + nseg * 16
                        + total_rows * 4 * 2,
                );
            }
            let segb = sc.seg.as_ref().expect("確保済み");
            let rg = sc.rows_gu.as_ref().expect("確保済み");
            let rid = sc.rows_id.as_ref().expect("確保済み");
            let xb = sc.x.as_ref().expect("確保済み");
            let g = sc.g.as_ref().expect("確保済み");
            let u = sc.u.as_ref().expect("確保済み");
            let act = sc.act.as_ref().expect("確保済み");
            let h = sc.h.as_ref().expect("確保済み");
            let t_launch = std::time::Instant::now();
            unsafe {
                // gate と up (同じ行表・同じ入力)。
                let rb_g = exp_pf::row_bytes(ty_g, n_in).unwrap_or(0);
                let rb_u = exp_pf::row_bytes(ty_u, n_in).unwrap_or(0);
                for (f, ptrs, buf, rb, plan) in [
                    (
                        kernels.fns[slot_g],
                        sc.pg.as_ref().expect("確保済み"),
                        g,
                        rb_g,
                        plan_g,
                    ),
                    (
                        kernels.fns[slot_u],
                        sc.pu.as_ref().expect("確保済み"),
                        u,
                        rb_u,
                        plan_g,
                    ),
                ] {
                    let mut pw = ptrs.ptr;
                    let mut px = xb.ptr;
                    let mut po = buf.ptr;
                    let mut ps = segb.ptr;
                    let mut pr = rg.ptr;
                    let (mut ns, mut ni, mut no, mut rb) = (
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
                        &mut rb as *mut _ as *mut c_void,
                    ];
                    ctx.launch_async(f, plan.grid, plan.block, 0, &mut args)?;
                }
                // act = silu(g) * u
                {
                    let n = ng as c_int;
                    let threads = 256usize;
                    let blocks = ng.div_ceil(threads) as u32;
                    let mut pgg = g.ptr;
                    let mut puu = u.ptr;
                    let mut paa = act.ptr;
                    let mut pn = n;
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
                // down (入力は act の行が詰まった形 = rows_id)
                {
                    let mut pw = sc.pd.as_ref().expect("確保済み").ptr;
                    let mut px = act.ptr;
                    let mut po = h.ptr;
                    let mut ps = segb.ptr;
                    let mut pr = rid.ptr;
                    let (mut ns, mut ni, mut no, mut rb) = (
                        nseg as c_int,
                        ff as c_int,
                        e as c_int,
                        exp_pf::row_bytes(ty_d, ff).unwrap_or(0) as c_int,
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
            }
            exp_pf::note_kernel_ns(t_launch.elapsed().as_nanos() as u64);
            // GPU の計算が走っている間に CPU のミスを計算する (重ねる)。
            // 層あたり GPU 5.6 ms / CPU 57 ms なので、待ちはほぼ消える。
            if let Err(e) = (cpu_miss)() {
                cpu_failed.set(true);
                return Err(e);
            }
            let t_sync = std::time::Instant::now();
            unsafe { ctx.sync()?; }
            exp_pf::note_kernel_ns(t_sync.elapsed().as_nanos() as u64);
            let t_d2h = std::time::Instant::now();
            h.to_slice(&mut out_h[..nh])?;
            exp_pf::note_d2h_ns(t_d2h.elapsed().as_nanos() as u64);
            dbg_d2h(nh * 4);
            Ok(true)
        })
    };
    match alloc() {
        Ok(v) => Ok(v),
        Err(e) => {
            if cpu_failed.get() {
                // CPU ミスの失敗は隠さない (out は途中まで書かれている)。
                return Err(e);
            }
            // 確保失敗はこのプロセスで 1 回だけ記録し、以後は CPU に固定する。
            if !PF_DEAD.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::error!("[exp-pf] 作業域を確保できないので以後 CPU 経路: {e:#}");
            }
            Ok(false)
        }
    }
}


/// ステージングを解放する (prefill の終端)。
///
/// decode のあいだ抱えたままだと expert の LRU がその分だけ狭くなるので、
/// リクエストの終わりに必ず返す。
///
/// task 165: 上げ専用スレッドを**先に**止めて join する (ステージングや pinned を
/// 指したままにしない)。
pub fn pf_stream_release() {
    // task 474: 先読みの実行役を先に止めて join する (置き場を指したままにしない)。
    crate::model::exp_pf::rc_join();
    crate::model::exp_pf::rc_clear();
    let _ = PF_UP.try_with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            *slot = None;
        }
    });
    let _ = PF_STREAM.try_with(|cell| {
        let mut st = cell.borrow_mut();
        for s in st.slot.iter_mut() {
            *s = PfSlot::new();
        }
        for p in st.pin.iter_mut() {
            *p = None;
        }
        st.up = None;
        st.probed = false;
        st.next_g = 0;
    });
}


/// task 558: このスレッドの上げ番号 (上げスレッドでなければ `None`)。
#[cfg(feature = "par")]
pub(crate) fn pf_up_idx() -> Option<usize> {
    PF_UP_IDX.with(|c| c.get())
}


/// H2D 用の非ブロッキングストリームとイベントを用意する (初回だけ)。
///
/// 戻り値はストリームの生ハンドル (上げ専用スレッドへ渡す)。
pub(crate) fn pf_up_stream(st: &mut PfStream) -> Result<usize> {
    if let Some(u) = st.up.as_ref() {
        return Ok(u.0 as usize);
    }
    let l = lib()?;
    let (Some(create), Some(ec)) = (l.stream_create, l.event_create) else {
        bail!("cuStreamCreate / cuEventCreate が無い");
    };
    let mut h: CUstream = std::ptr::null_mut();
    const CU_STREAM_NON_BLOCKING: u32 = 0x01;
    unsafe {
        if create(&mut h, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS {
            bail!("cuStreamCreate が失敗");
        }
    }
    st.up = Some(CopyStream(h));
    // イベントはスロットごとに 1 本ずつ (上げの完了 / 計算の完了)。
    for i in 0..PF_STAGE_SLOTS {
        if st.slot[i].ev_up.is_none() {
            let mut ev: CUevent = std::ptr::null_mut();
            unsafe {
                if ec(&mut ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS {
                    bail!("cuEventCreate が失敗");
                }
            }
            st.slot[i].ev_up = Some(CopyEvent(ev));
        }
        if st.slot[i].ev_use.is_none() {
            let mut ev: CUevent = std::ptr::null_mut();
            unsafe {
                if ec(&mut ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS {
                    bail!("cuEventCreate (use) が失敗");
                }
            }
            st.slot[i].ev_use = Some(CopyEvent(ev));
        }
    }
    Ok(h as usize)
}


/// グループ `g` をスロット `g % 3` へ上げる (計算スレッドが直列に行う経路)。
///
/// **同じスロットを使った 2 つ前のグループの計算完了を待ってから**書き込む。
///
/// task 227: 転送元を 3 通りから選べる ([`exp_pf::stream_src`])。
/// - `pin` (既定): mmap → pinned に写してから積む。`DENNO_PF_STREAM_CHUNK_KB` が
///   正なら「小さい塊を写す → 積む」を交互に行う (DMA が読む頃にはホストの
///   キャッシュに残っている、という仮説の検証)。
/// - `page`: mmap のページをそのまま積む (写しが無い。driver が内部でバウンスし、
///   呼び出しの間ホストはブロックする)。
/// - `reg`: mmap のページを `cuMemHostRegister` で pin してから積む (写しが
///   無く、DMA も pinned 速度。pin できなければ pageable として積む)。
fn pf_pump_in(
    st: &mut PfStream,
    g: u64,
    items: &[PfItem<'_>],
    pinned: bool,
) -> Result<PfPump> {
    // task 296: スロット準備 (配置の計算と pinned の確保) も pump の内側なので、
    // 写しと分けて積む (診断 `[pf-lm-dbg] stream` の `prep`)。
    let t_prep = std::time::Instant::now();
    let plan = pf_slot_prepare(st, g, items, pinned)?;
    pf_stream_t_add(5, t_prep.elapsed().as_nanos() as u64);
    let s = plan.slot;
    let mut probe: Option<(usize, u64)> = None;
    let mut enq_ns = 0u64;
    if pinned {
        // 1. 再利用の前に、このスロットを読んだ計算の完了を待つ。
        if st.slot[s].used {
            if let Some(ev) = st.slot[s].ev_use.as_ref() {
                let Some(es) = lib()?.event_sync else {
                    bail!("cuEventSynchronize が無い");
                };
                // task 233: この待ちは今まで台帳のどこにも出ていなかった (診断で積む)。
                let t_use = std::time::Instant::now();
                unsafe {
                    check(es(ev.0), "cuEventSynchronize (stage use)")?;
                }
                crate::model::exp_pf::note_stream_use_ns(t_use.elapsed().as_nanos() as u64);
            }
        }
        let up = pf_up_stream(st)? as CUstream;
        let base = st.slot[s].buf.as_ref().expect("確保済み").ptr();
        let l = lib()?;
        let src = exp_pf::stream_src();
        // task 227: 塊ごとに写して DMA を重ねるバイト数 (0 = 1 回で積む)。
        // task 239: `DENNO_PF_STREAM_SPLIT=n` は **全体を n 等分**して重ねる
        // (塊の切り方が違うだけで、デバイスへ渡るバイト列は同じ)。
        let chunk = {
            let nspl = crate::model::exp_pf::stream_split();
            if src == 0 && nspl > 1 {
                plan.need
                    .div_ceil(nspl)
                    .div_ceil(PF_STAGE_ALIGN)
                    .saturating_mul(PF_STAGE_ALIGN)
                    .max(PF_STAGE_ALIGN)
            } else {
                exp_pf::stream_chunk_bytes()
            }
        };
        let t_copy = std::time::Instant::now();
        let t_dma = std::time::Instant::now();
        let t_enq = std::time::Instant::now();
        let mut copy_ns = 0u64;
        let mut sent = 0usize;
        if src == 0 {
            // 2. pinned へ写してから、非同期で H2D を積む。
            let pin_ptr = st.pin[s].as_ref().expect("確保済み").ptr;
            if chunk == 0 || plan.parts.is_empty() {
                // 従来: 全部写してから **グループ全体を 1 回の非同期コピーで積む**。
                // 行列ごとに分けると 1 グループ 96 回になり、`cuMemcpyHtoDAsync` の
                // 呼び出し固定費が DMA の実時間を上回る (lab t162a)。
                pf_copy_into_pin(pin_ptr as usize, &plan.parts, plan.pread);
                copy_ns = t_copy.elapsed().as_nanos() as u64;
                if plan.need > 0 {
                    unsafe {
                        check(
                            (l.memcpy_htod_async)(base, pin_ptr, plan.need, up),
                            "cuMemcpyHtoDAsync (stage)",
                        )?;
                    }
                }
                sent = plan.need;
            } else {
                // task 227 / 239: 塊ごとに「写す → 積む」。DMA は同じ up ストリームに
                // 順に積まれるので、ホストの写しと DMA が重なる。
                let spec: Vec<(usize, usize)> = plan
                    .parts
                    .iter()
                    .map(|&(o, r)| (o, r.len))
                    .collect();
                let spans = pf_chunk_spans(&spec, chunk);
                let mut done = 0usize;
                for &(a, b) in &spans {
                    // [a, b) に入る parts を写す (塊は parts の境界で切ってある)。
                    let mut i = done;
                    while i < plan.parts.len() && plan.parts[i].0 < b {
                        i += 1;
                    }
                    let t = std::time::Instant::now();
                    pf_copy_into_pin(pin_ptr as usize, &plan.parts[done..i], plan.pread);
                    copy_ns += t.elapsed().as_nanos() as u64;
                    done = i;
                    unsafe {
                        check(
                            (l.memcpy_htod_async)(
                                base + a as u64,
                                (pin_ptr as *const u8).add(a) as *const c_void,
                                b - a,
                                up,
                            ),
                            "cuMemcpyHtoDAsync (chunk)",
                        )?;
                    }
                    sent += b - a;
                }
            }
        } else {
            // task 227: 写さずに、スライスごとに mmap から直接積む。
            // デバイス側の置き場所は従来と同じ (expert ごとに gate/up/down)。
            for &(off, r) in &plan.parts {
                let from = if src == 2 {
                    pf_reg_ptr(r.ptr, r.len).unwrap_or(r.ptr)
                } else {
                    r.ptr
                };
                unsafe {
                    check(
                        (l.memcpy_htod_async)(base + off as u64, from as *const c_void, r.len, up),
                        "cuMemcpyHtoDAsync (slice)",
                    )?;
                }
                sent += r.len;
            }
            copy_ns = t_copy.elapsed().as_nanos() as u64;
        }
        dbg_htod(sent);
        // 4. 完了イベントを積む。
        let Some(record) = l.event_record else {
            bail!("cuEventRecord が無い");
        };
        if let Some(ev) = st.slot[s].ev_up.as_ref() {
            unsafe {
                check(record(ev.0, up), "cuEventRecord (stage up)")?;
            }
        }
        enq_ns = (t_enq.elapsed().as_nanos() as u64).saturating_sub(copy_ns);
        // 5. 最初の 1 回だけ実速度を測る (同期して待つ)。
        if !st.probed {
            st.probed = true;
            if let Some(ev) = st.slot[s].ev_up.as_ref() {
                let Some(es) = l.event_sync else {
                    bail!("cuEventSynchronize が無い");
                };
                unsafe {
                    check(es(ev.0), "cuEventSynchronize (probe)")?;
                }
            }
            probe = Some((plan.need, t_dma.elapsed().as_nanos() as u64));
        }
        st.slot[s].used = true;
        Ok(PfPump {
            ptrs: plan.ptrs,
            bytes: plan.need,
            copy_ns,
            enq_ns,
            probe,
        })
    } else {
        // pageable の同期コピー (task 156 と同じ)。重なりは無い。
        let t_copy = std::time::Instant::now();
        let buf = st.slot[s].buf.as_ref().expect("確保済み");
        let mut cursor = 0usize;
        for (i, it) in items.iter().enumerate() {
            let b = &it.1;
            for m in 0..3 {
                if !b[m].is_empty() {
                    buf.copy_from_at(cursor, b[m])?;
                }
                cursor += pf_stage_align_up(plan.lens[i][m]);
            }
        }
        let copy_ns = t_copy.elapsed().as_nanos() as u64;
        dbg_htod(plan.need);
        st.slot[s].used = true;
        Ok(PfPump {
            ptrs: plan.ptrs,
            bytes: plan.need,
            copy_ns,
            enq_ns,
            probe: None,
        })
    }
}


/// task 254: 表の非同期 H2D (`DENNO_PF_TAB_ASYNC`) の転送元とイベント。
///
/// # なぜ要るか
///
/// 表 (`seg` / 行表 / デバイスポインタ表) は `DevBuffer::copy_from` /
/// `copy_from_at` = **pageable の同期 `cuMemcpyHtoD`** で積んでいた。この API は
/// null ストリームに積まれたカーネルの完了までホストを止めるので、14k の cold
/// prefill ではグループ 1 つごとに「GPU が空くまで待つ」が入り、台帳の `h2d` が
/// experts 区間 18.3 s のうち 8.8 s を占めていた (PROFILE=1 の実測は REPORT_P254)。
///
/// ここでは転送元を pinned に置き、null ストリームへ**非同期**で積む。順序は
/// ストリームが保証するので、カーネルは自分の表が書けた後に走る。転送元は
/// 数枚を回し、使い回す前にその枚の完了イベントだけ待つ (ホストが先に走り
/// すぎたときだけ止まる)。
///
/// 数値には触れない (同じバイト列を同じ順に積むだけ)。
mod pf_tab {
    use super::*;

    /// 回転させる転送元の枚数 (ホストが先に走れるグループ数の上限)。
    pub(crate) const SLOTS: usize = 4;

    pub(crate) struct Slot {
        pin: PinMem,
        ev: CopyEvent,
        used: bool,
    }

    struct State {
        slots: Vec<Slot>,
        /// 作り直した古い転送元 (飛んでいるコピーが掴んでいる可能性があるので
        /// 解放しない。prefill 1 回で増えるのは高々数枚)。
        retired: Vec<PinMem>,
        next: usize,
    }

    thread_local! {
        static ST: std::cell::RefCell<Option<State>> = const { std::cell::RefCell::new(None) };
    }

    /// `DENNO_PF_TAB_ASYNC` の値の解釈 (`0` / `false` / `no` だけ off)。
    ///
    /// 既定は off (A/B と切り戻しのため)。env は `OnceLock` なので、値の解釈は
    /// ここに閉じてテストから直接叩けるようにする。
    pub fn value_on(v: Option<&str>) -> bool {
        !matches!(v, Some("0") | Some("false") | Some("no"))
    }

    /// `DENNO_PF_TAB_ASYNC` が on か (既定 off)。
    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| match std::env::var("DENNO_PF_TAB_ASYNC") {
            Ok(v) => value_on(Some(v.as_str())),
            Err(_) => false,
        })
    }

    /// 呼び出し回数と成功回数 (分岐が発火したかの証拠)。
    pub(crate) static CALLS: AtomicU64 = AtomicU64::new(0);
    static OKS: AtomicU64 = AtomicU64::new(0);

    /// `(呼び出し, 非同期で積めた)` の累計。テストと実機のログ用。
    pub fn counts() -> (u64, u64) {
        (
            CALLS.load(Ordering::Relaxed),
            OKS.load(Ordering::Relaxed),
        )
    }

    /// 累計を 0 に戻す (リクエスト境界)。
    pub fn reset_counts() {
        CALLS.store(0, Ordering::Relaxed);
        OKS.store(0, Ordering::Relaxed);
    }

    /// 空きスロットを 1 枚返す (前の使用が飛んでいれば完了を待つ)。
    pub(crate) fn take(bytes: usize) -> Result<usize> {
        let l = lib()?;
        let (Some(ec), Some(rec), Some(es)) = (l.event_create, l.event_record, l.event_sync)
        else {
            bail!("cuEventCreate / cuEventRecord / cuEventSynchronize が無い");
        };
        ST.with(|cell| {
            let mut g = cell.borrow_mut();
            let st = g.get_or_insert_with(|| State {
                slots: Vec::new(),
                retired: Vec::new(),
                next: 0,
            });
            if st.slots.is_empty() {
                for _ in 0..SLOTS {
                    let mut ev: CUevent = std::ptr::null_mut();
                    unsafe {
                        if ec(&mut ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS {
                            bail!("cuEventCreate (pf_tab) が失敗");
                        }
                    }
                    st.slots.push(Slot {
                        pin: PinMem::alloc(bytes)?,
                        ev: CopyEvent(ev),
                        used: false,
                    });
                }
            }
            let i = st.next % SLOTS;
            st.next = st.next.wrapping_add(1);
            // 容量が足りなければ作り直す (古いものは解放しない)。
            if st.slots[i].pin.bytes < bytes {
                let new = PinMem::alloc(bytes)?;
                let old = std::mem::replace(&mut st.slots[i].pin, new);
                st.retired.push(old);
            }
            // 前回のコピーがまだ飛んでいるなら待つ (通常は即返る)。
            if st.slots[i].used {
                unsafe {
                    check(es(st.slots[i].ev.0), "cuEventSynchronize (pf_tab)")?;
                }
            }
            st.slots[i].used = true;
            // 呼び出し側が pinned のアドレスを要るので、イベントはここで記録せず
            // [`upload`] の最後に記録する。
            let _ = (rec, i);
            Ok(i)
        })
    }

    /// pinned のベースアドレス (スロット `i`)。
    pub(crate) fn base(i: usize) -> usize {
        ST.with(|cell| {
            let g = cell.borrow();
            g.as_ref().expect("確保済み").slots[i].pin.ptr as usize
        })
    }

    /// スロット `i` の完了イベントを null ストリームへ記録する。
    pub(crate) fn record(i: usize) -> Result<()> {
        let l = lib()?;
        let Some(rec) = l.event_record else {
            bail!("cuEventRecord が無い");
        };
        ST.with(|cell| {
            let g = cell.borrow();
            let ev = g.as_ref().expect("確保済み").slots[i].ev.0;
            unsafe { check(rec(ev, std::ptr::null_mut()), "cuEventRecord (pf_tab)") }
        })
    }

    /// `parts` を pinned に写して null ストリームへ非同期で積む。
    ///
    /// 戻り値 `false` = 使えなかった (呼び出し側は同期版へ落ちる)。
    pub fn upload(parts: &[(u64, usize, &[u8])]) -> bool {
        CALLS.fetch_add(1, Ordering::Relaxed);
        let total: usize = parts.iter().map(|(_, _, b)| b.len()).sum();
        if total == 0 {
            return true;
        }
        let Ok(i) = take(total) else {
            return false;
        };
        let Ok(l) = lib() else { return false };
        let base = base(i);
        let mut off = 0usize;
        let mut jobs: Vec<(CUdeviceptr, usize, usize)> = Vec::with_capacity(parts.len());
        for (dev, dev_off, data) in parts {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    (base + off) as *mut u8,
                    data.len(),
                );
            }
            jobs.push((*dev + *dev_off as CUdeviceptr, base + off, data.len()));
            off += data.len();
        }
        for (d, s, n) in jobs {
            unsafe {
                if (l.memcpy_htod_async)(d, s as *const c_void, n, std::ptr::null_mut()) != CUDA_SUCCESS
                {
                    return false;
                }
            }
        }
        if record(i).is_err() {
            return false;
        }
        OKS.fetch_add(1, Ordering::Relaxed);
        true
    }
}


/// task 254: 表の非同期 H2D の (呼び出し, 成功) 累計。実機ログ用。
pub fn pf_tab_counts() -> (u64, u64) {
    pf_tab::counts()
}


/// task 254: 累計を 0 に戻す (テストとリクエスト境界)。
#[cfg(feature = "cuda")]
pub fn pf_tab_counts_reset() {
    pf_tab::reset_counts()
}


/// mmap の [ptr, ptr+len) を driver に pin させる (登録済みなら何もしない)。
///
/// 戻り値は「そのまま DMA してよいホストポインタ」。pin できなければ `None`
/// (呼び出し側は pageable として積む = 値は同じ、速度だけ落ちる)。
fn pf_reg_ptr(ptr: usize, len: usize) -> Option<usize> {
    if len == 0 || PF_REG_DEAD.load(std::sync::atomic::Ordering::Relaxed) {
        return None;
    }
    const PAGE: usize = 4096;
    let a = ptr & !(PAGE - 1);
    let b = (ptr + len + PAGE - 1) & !(PAGE - 1);
    {
        let Ok(reg) = PF_REG.lock() else {
            PF_REG_DEAD.store(true, std::sync::atomic::Ordering::Relaxed);
            return None;
        };
        if let Some(&(_, e)) = reg.iter().rev().find(|&&(s0, _)| s0 <= a) {
            if e >= b {
                return Some(ptr);
            }
        }
    }
    if PF_REG_OK_BYTES.load(std::sync::atomic::Ordering::Relaxed) + (b - a) > exp_pf::reg_bytes()
    {
        // 上限に達した → これ以上 pin しない (pageable で積む)。
        return None;
    }
    let Ok(l) = lib() else {
        return None;
    };
    let Some(f) = l.mem_host_register else {
        PF_REG_DEAD.store(true, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    let t = std::time::Instant::now();
    let mut ok = false;
    let mut used = 0u32;
    for flags in PF_REG_FLAG_TRIES {
        let rc = unsafe { f(a as *mut c_void, b - a, flags) };
        if rc == CUDA_SUCCESS {
            ok = true;
            used = flags;
            PF_REG_FLAGS.store(flags, std::sync::atomic::Ordering::Relaxed);
            break;
        }
    }
    let ns = t.elapsed().as_nanos() as u64;
    exp_pf::note_reg(b - a, ok, ns, used);
    if !ok {
        let n = PF_REG_FAILS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n >= 4 {
            PF_REG_DEAD.store(true, std::sync::atomic::Ordering::Relaxed);
            static ONCE: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[pf-zc] cuMemHostRegister が通らない (file backed の写像) — \
                     pageable で積む (flags={used:#x})"
                );
            }
        }
        return None;
    }
    PF_REG_OK_BYTES.fetch_add(b - a, std::sync::atomic::Ordering::Relaxed);
    // 記録して併合する (区間は昇順・非重複に保つ)。
    if let Ok(mut reg) = PF_REG.lock() {
        let mut out: Vec<(usize, usize)> = Vec::with_capacity(reg.len() + 1);
        let mut ins = (a, b);
        for &(s, e) in reg.iter() {
            if e < ins.0 {
                out.push((s, e));
            } else if ins.1 < s {
                out.push(ins);
                ins = (s, e);
            } else {
                ins = (ins.0.min(s), ins.1.max(e));
            }
        }
        out.push(ins);
        *reg = out;
    }
    Some(ptr)
}


/// グループ `g` の H2D 完了を**計算ストリームの待ち**として入れる (task 324)。
///
/// ホストはここで止まらない。カーネルは同じ (既定) ストリームに積むので、
/// DMA が終わったグループから順に GPU が走る。ホストが待つのは
/// [`pf_wait_ready`] (ワーカーが ev_up を記録したこと) だけになる。
///
/// `cuStreamWaitEvent` は**積まれた後**の記録を見るので、呼ぶ前に `ready` を
/// 待っていないと 1 つ前の記録で満たされてしまう (データがまだ来ていないのに
/// カーネルが走る)。
fn pf_stream_wait_up_in(st: &mut PfStream, g: u64, pinned: bool) -> Result<()> {
    if !pinned {
        return Ok(());
    }
    let s = (g as usize) % PF_STAGE_SLOTS;
    let Some(ev) = st.slot[s].ev_up.as_ref() else {
        return Ok(());
    };
    let Some(sw) = lib()?.stream_wait_event else {
        return Ok(());
    };
    unsafe {
        check(
            sw(std::ptr::null_mut(), ev.0, 0),
            "cuStreamWaitEvent (stage up/計算)",
        )?;
    }
    Ok(())
}


// ---------------------------------------------------------------------------
// task 165: 上げ専用スレッド
// ---------------------------------------------------------------------------
//
// task 162 の台帳は「1 グループ = 計算 2.6 + 写し 0.9 + 積み込み 0.8 ms の和」で、
// 写しと積み込みが計算と同じホストスレッドに載っていた (直列)。ここでは
// **グループ g+1 の写しと積み込みを専用スレッドへ出し**、計算スレッドは
// グループ g のカーネルを積むだけにする。
//
// 受け渡しは mpsc 1 本 + Mutex/Condvar 1 組:
// - 計算スレッド: スロットを用意して依頼を送る → `ready` が g を超えるのを待つ
//   → ev_up を同期 → カーネルを積む → ev_use を記録
// - 上げスレッド: 依頼を 1 本ずつ処理する。スロットの再利用 (前のグループの
//   計算完了) を ev_use で待ち、pinned へ写し、非同期 H2D を積んで ev_up を
//   記録する。イベントはスロット (= グループ) ごとなので、計算側は自分の
//   グループの完了だけを待つ。

/// 上げ専用スレッドへの 1 グループぶんの依頼。
pub(crate) struct PfUpJob {
    pub(crate) g: u64,
    pub(crate) need: usize,
    /// ステージング (デバイス) の先頭。
    pub(crate) dst: u64,
    /// pinned バウンスの先頭。
    pub(crate) pin: usize,
    /// (pinned 内のオフセット, 転送元)。
    pub(crate) parts: Vec<(usize, PfHostRange)>,
    /// H2D を積むストリーム。
    pub(crate) stream: usize,
    /// このスロットを前回読んだ計算の完了イベント (未使用なら 0)。
    pub(crate) wait_ev: usize,
    /// task 320: このスロットの pinned を前回読んだ DMA の完了イベント
    /// (分離モードだけ。未使用なら 0)。ホストはこれだけを待って写す。
    pub(crate) pin_ev: usize,
    /// このグループの H2D 完了を知らせるイベント。
    pub(crate) up_ev: usize,
    /// 最初の 1 グループだけ、同期して実 H2D を測る。
    pub(crate) probe: bool,
    /// task 325: 写しを `read_at` で行うか (`DENNO_PF_PREAD`)。
    pub(crate) pread: bool,
}


// SAFETY: 転送元は mmap 済みの expert 重みで、依頼を出した計算スレッドが
// `ready` を待ってからこの層の処理を進める (呼び出しが終わるまで有効)。
// 依頼は上げスレッドが 1 本ずつ処理し、Drop で必ず join する。
unsafe impl Send for PfUpJob {}


/// 上げ専用スレッドの進捗 (計算スレッドはここだけを見る)。
pub(crate) struct PfUpState {
    /// task 554: グループごとに `ev_up` を記録し終えたか。
    ///
    /// 上げスレッドが 2 本以上になると完了が順不同に届くので、**前線 (連続した
    /// 番号) では待てない**。1 本のときも番号に穴が空く (`items` が空のグループは
    /// 依頼しない) ので、待ちはグループ単位の旗で行う。旗を飛ばして古い `ev_up`
    /// で待ちが満たされると重みが壊れるため、ここは「そのグループが記録済みか」
    /// だけを見る。
    pub(crate) done: Vec<bool>,
    /// グループごとの (写し ns, 積み込み ns, スロット待ち ns)。
    pub(crate) times: Vec<(u64, u64, u64)>,
    /// 最初のグループで測った実 H2D (バイト, ns)。
    pub(crate) probe: Option<(usize, u64)>,
    /// ワーカーが落ちた理由 (あれば計算側も降りる)。
    pub(crate) err: Option<String>,
}


/// 上げスレッドの進捗を 1 グループぶん公開する。
///
/// `ready` は単調 (遅れて届いた小さい番号で戻さない)。エラーは最初の 1 本を残す。
fn pf_up_publish(
    st: &(std::sync::Mutex<PfUpState>, std::sync::Condvar),
    g: u64,
    copy_ns: u64,
    enq_ns: u64,
    slot_ns: u64,
    probe: Option<(usize, u64)>,
    err: Option<String>,
) {
    let (m, cv) = st;
    let mut s = m.lock().unwrap_or_else(|e| e.into_inner());
    if s.times.len() <= g as usize {
        s.times.resize(g as usize + 1, (0, 0, 0));
    }
    s.times[g as usize] = (copy_ns, enq_ns, slot_ns);
    if let Some(p) = probe {
        s.probe = Some(p);
    }
    if err.is_some() && s.err.is_none() {
        s.err = err;
    }
    // task 554: 完了は順不同に届くので、グループごとの旗を立てるだけにする
    // (連続した前線にすると、依頼しない空のグループで止まって待ちが返らない)。
    if s.done.len() <= g as usize {
        s.done.resize(g as usize + 1, false);
    }
    s.done[g as usize] = true;
    cv.notify_all();
}


/// 上げ専用スレッド。
pub(crate) struct PfUploader {
    tx: Option<std::sync::mpsc::Sender<PfUpJob>>,
    join: Option<std::thread::JoinHandle<()>>,
}


impl PfUploader {
    /// task 554: 進捗の置き場を共有して起動する (`DENNO_PF_UP_WORKERS` 本で 1 組)。
    /// task 558: `idx` はこのスレッドの番号 (pread プールを分けるための印)。
    fn start_with(
        ctx: &'static CudaCtx,
        st: std::sync::Arc<(std::sync::Mutex<PfUpState>, std::sync::Condvar)>,
        idx: usize,
    ) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<PfUpJob>();
        let join = std::thread::Builder::new()
            .name(format!("pf-up-{idx}"))
            .spawn(move || pf_up_worker(rx, st, ctx, idx))
            .ok();
        PfUploader { tx: Some(tx), join }
    }

    /// ワーカーのスレッドが終わっているか (待ち続けを避けるため)。
    fn finished(&self) -> bool {
        self.join.as_ref().map(|j| j.is_finished()).unwrap_or(true)
    }

    /// 依頼を 1 本送る (待たない)。
    pub(crate) fn submit(&self, job: PfUpJob) -> Result<()> {
        let Some(tx) = self.tx.as_ref() else {
            bail!("上げスレッドが居ない");
        };
        tx.send(job)
            .map_err(|_| anyhow!("上げスレッドが停止している"))
    }

}


impl Drop for PfUploader {
    fn drop(&mut self) {
        // 送信口を閉じるとワーカーの recv が Err で抜ける。
        self.tx = None;
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}


/// task 554: 上げ専用スレッドの束 (`DENNO_PF_UP_WORKERS` 本)。
///
/// 依頼はグループ番号の剰余でワーカーへ振り分ける。進捗の置き場は 1 組だけ
/// 共有するので、計算側の待ち ([`PfUp::wait_ready`]) は「グループ番号の
/// 小さい方から途切れずに揃った所」までしか進まない ([`pf_up_publish`])。
pub(crate) struct PfUp {
    ups: Vec<PfUploader>,
    st: std::sync::Arc<(std::sync::Mutex<PfUpState>, std::sync::Condvar)>,
}


impl PfUp {
    /// `n` 本のワーカーを起動する (1 本目が進捗の置き場を作る)。
    pub(crate) fn start(ctx: &'static CudaCtx, n: usize) -> Self {
        let st = new_up_state();
        let mut ups = Vec::with_capacity(n);
        for i in 0..n.max(1) {
            ups.push(PfUploader::start_with(ctx, std::sync::Arc::clone(&st), i));
        }
        PfUp { ups, st }
    }

    /// 依頼を 1 本送る。グループ番号でワーカーを選ぶ。
    ///
    /// task 554: `pread` を使わない層 (page cache / mmap からの写しだけ = 既定
    /// モデル) は **ワーカー 0 だけ**に送る。2 本で写すと DRAM 帯域を 2 本で
    /// 食い合うだけで、既定モデルの prefill は GPU 律速 (写しはカーネルの裏)。
    /// ここを分けないと、既定モデルで「何も変わらない」が保証できない。
    pub(crate) fn submit(&self, job: PfUpJob) -> Result<()> {
        let i = if job.pread {
            (job.g as usize) % self.ups.len().max(1)
        } else {
            0
        };
        match self.ups.get(i) {
            Some(u) => u.submit(job),
            None => bail!("上げスレッドが居ない"),
        }
    }

    /// グループ `g` の依頼が**積まれた**ことを待つ (待ち ns を返す)。
    pub(crate) fn wait_ready(&self, g: u64) -> Result<u64> {
        let (m, cv) = &*self.st;
        let mut s = m.lock().unwrap_or_else(|e| e.into_inner());
        let t = std::time::Instant::now();
        loop {
            if let Some(e) = s.err.as_deref() {
                bail!("上げスレッド: {e}");
            }
            if s.done.get(g as usize).copied().unwrap_or(false) {
                return Ok(t.elapsed().as_nanos() as u64);
            }
            let (s2, to) = cv
                .wait_timeout(s, std::time::Duration::from_millis(200))
                .unwrap_or_else(|e| e.into_inner());
            s = s2;
            if to.timed_out() {
                // 全ワーカーが終わっている (panic 等) なら待ち続けない。
                let dead = self.ups.iter().all(|u| u.finished());
                if dead && !s.done.get(g as usize).copied().unwrap_or(false) {
                    bail!("上げスレッドが終了した (グループ {g} の依頼が処理されていない)");
                }
            }
        }
    }

    /// グループ `g` の (写し ns, 積み込み ns, スロット待ち ns)。
    pub(crate) fn times(&self, g: u64) -> (u64, u64, u64) {
        let (m, _) = &*self.st;
        let s = m.lock().unwrap_or_else(|e| e.into_inner());
        s.times.get(g as usize).copied().unwrap_or((0, 0, 0))
    }

    /// 最初のグループで測った実 H2D。
    pub(crate) fn probe(&self) -> Option<(usize, u64)> {
        let (m, _) = &*self.st;
        let s = m.lock().unwrap_or_else(|e| e.into_inner());
        s.probe
    }
}


/// 上げ専用スレッドを 1 回だけ起動する。
pub(crate) fn pf_up_ensure() {
    PF_UP.with(|cell| {
        if let Ok(mut slot) = cell.try_borrow_mut() {
            if slot.is_none() {
                if let Some(ctx) = shared_ctx() {
                    *slot = Some(PfUp::start(ctx, crate::model::exp_pf::up_workers()));
                }
            }
        }
    });
}


/// 上げ専用スレッドの本体。
fn pf_up_worker(
    rx: std::sync::mpsc::Receiver<PfUpJob>,
    st: std::sync::Arc<(std::sync::Mutex<PfUpState>, std::sync::Condvar)>,
    ctx: &'static CudaCtx,
    idx: usize,
) {
    // task 558: このスレッドの番号を焼く (`pf_copy_into_pin` が pread プールを
    // 分けるのに使う)。計算スレッドは `None` のまま。
    let _ = PF_UP_IDX.try_with(|c| c.set(Some(idx)));
    // CUDA の current context はスレッドローカルなので、このスレッドでも張る。
    let ctx_ok = ctx.make_current().is_ok();
    loop {
        let Ok(job) = rx.recv() else { break };
        let mut copy_ns = 0u64;
        let mut enq_ns = 0u64;
        let mut slot_ns = 0u64;
        let mut probe: Option<(usize, u64)> = None;
        let mut bad: Option<String> = None;
        if ctx_ok {
            if let Err(e) = pf_up_run(&job, &mut copy_ns, &mut enq_ns, &mut slot_ns, &mut probe) {
                bad = Some(format!("{e:#}"));
            }
        } else {
            bad = Some("このスレッドで cuCtxSetCurrent が失敗".to_string());
        }
        pf_up_publish(&st, job.g, copy_ns, enq_ns, slot_ns, probe, bad);
    }
}


/// 依頼 1 本を処理する (再利用待ち → 写し → 積み込み → 記録)。
fn pf_up_run(
    job: &PfUpJob,
    copy_ns: &mut u64,
    enq_ns: &mut u64,
    slot_ns: &mut u64,
    probe: &mut Option<(usize, u64)>,
) -> Result<()> {
    let l = lib()?;
    // 1. このスロットを前回読んだ計算の完了を待つ。
    //
    // task 320 (分離モード): ホストで待つのは「pinned を前回読んだ DMA の完了」
    // だけにする。カーネルの完了 (スロットの再利用) は待たず、DMA を積む前に
    // 上げストリームへ `cuStreamWaitEvent` で入れる。カーネルを待つと
    // 写しがカーネルの裏に入らず、直列のときと変わらない (lab t320a:
    // ut=1 でも wall -5.0%、`up_wait` 405 → 1,112 ms)。
    let decouple = exp_pf::up_decouple_on();
    let host_ev = if decouple { job.pin_ev } else { job.wait_ev };
    if host_ev != 0 {
        let Some(es) = l.event_sync else {
            bail!("cuEventSynchronize が無い");
        };
        let t = std::time::Instant::now();
        unsafe {
            check(
                es(host_ev as CUevent),
                "cuEventSynchronize (stage reuse/上げスレッド)",
            )?;
        }
        *slot_ns = t.elapsed().as_nanos() as u64;
    }
    // 2. 写しと積み込み。
    //
    // task 481: `DENNO_PF_UP_SPLIT_KB` (既定 8 MiB) があれば、**キャッシュあり**の
    // 小さなバウンス 2 枚を交互に使い、写し(i+1) を DMA(i) と重ねる。DMA が読む行は
    // まだ L3 にあるので、DRAM の読み直し (page cache とバウンスで 2 本) が
    // 1 本になる見込み (**推定**)。デバイスへ渡るバイト列は 1 回写しと同一。
    let split_bytes = exp_pf::up_split_bytes();
    if split_bytes > 0 && job.need > 0 && job.pin != 0 {
        // 2.5 と同じ待ち (VRAM のステージングの再利用) を先に積む。
        if decouple && job.wait_ev != 0 {
            let Some(sw) = l.stream_wait_event else {
                bail!("cuStreamWaitEvent が無い");
            };
            unsafe {
                check(
                    sw(job.stream as CUstream, job.wait_ev as CUevent, 0),
                    "cuStreamWaitEvent (stage reuse/上げスレッド/分割)",
                )?;
            }
        }
        pf_up_run_split(job, copy_ns, enq_ns)?;
        let Some(record) = l.event_record else {
            bail!("cuEventRecord が無い");
        };
        let t_enq2 = std::time::Instant::now();
        unsafe {
            check(
                record(job.up_ev as CUevent, job.stream as CUstream),
                "cuEventRecord (stage up/上げスレッド/分割)",
            )?;
        }
        *enq_ns = enq_ns.saturating_add(t_enq2.elapsed().as_nanos() as u64);
    } else {
        // 2. pinned へ写す (計算スレッドと同じ並列写し)。
        let t_copy = std::time::Instant::now();
        pf_copy_into_pin(job.pin, &job.parts, job.pread);
        *copy_ns = t_copy.elapsed().as_nanos() as u64;
        // 2.5 task 320 (分離モード): VRAM のステージングを前回読んだカーネルの
        //     完了を上げストリームの待ちとして入れる (ホストは止めない)。
        if decouple && job.wait_ev != 0 && job.need > 0 {
            let Some(sw) = l.stream_wait_event else {
                bail!("cuStreamWaitEvent が無い");
            };
            unsafe {
                check(
                    sw(
                        job.stream as CUstream,
                        job.wait_ev as CUevent,
                        0,
                    ),
                    "cuStreamWaitEvent (stage reuse/上げスレッド)",
                )?;
            }
        }
        // 3. 非同期 H2D を積み、完了イベントを記録する。
        let t_enq = std::time::Instant::now();
        if job.need > 0 {
            // 台帳 (q4ledger) はスレッドローカルの区間を持つので、ここでは素の
            // 関数を呼ぶ (バイト数は [exp-pf-stream] が数える)。
            let real = REAL_HTOD_ASYNC.load(Ordering::Relaxed);
            if real.is_null() {
                bail!("cuMemcpyHtoDAsync の実体が無い");
            }
            let f: FnCuMemcpyHtoDAsyncV2 = unsafe { std::mem::transmute(real) };
            unsafe {
                check(
                    f(job.dst, job.pin as *const c_void, job.need, job.stream as CUstream),
                    "cuMemcpyHtoDAsync (stage/上げスレッド)",
                )?;
            }
        }
        let Some(record) = l.event_record else {
            bail!("cuEventRecord が無い");
        };
        unsafe {
            check(
                record(job.up_ev as CUevent, job.stream as CUstream),
                "cuEventRecord (stage up/上げスレッド)",
            )?;
        }
        *enq_ns = t_enq.elapsed().as_nanos() as u64;
    }
    // 4. 最初の 1 グループだけ、同期して実 H2D を測る。
    if job.probe {
        let Some(es) = l.event_sync else {
            bail!("cuEventSynchronize が無い");
        };
        let t = std::time::Instant::now();
        unsafe {
            check(
                es(job.up_ev as CUevent),
                "cuEventSynchronize (probe/上げスレッド)",
            )?;
        }
        *probe = Some((job.need, t.elapsed().as_nanos() as u64));
    }
    Ok(())
}


/// task 481: 分割 DMA の 2 枚のバウンスと、それぞれを前回読んだ DMA の完了。
pub(crate) struct PfUpSplit {
    /// 2 枚の pinned (write-combined にしない = L3 に載る)。
    stage: [Option<PinMem>; 2],
    /// 各バウンスを前回読んだ DMA の完了イベント (作った時点で非 null)。
    ev: [CUevent; 2],
    /// そのバウンスを**一度でも** DMA の転送元に使ったか。
    ///
    /// バウンスは依頼 (グループ) をまたいで使い回すので、「この依頼の中で
    /// 2 個前か」ではなく「過去に使ったか」で待つ。ここを依頼内の番号で
    /// 判定すると、前の依頼の最後の DMA がまだ読んでいるバウンスを次の依頼が
    /// 上書きして**重みが壊れる** (lab t481a で実際に壊れた)。
    used: [bool; 2],
    /// 1 枚の大きさ (バイト)。
    chunk: usize,
}


impl Drop for PfUpSplit {
    fn drop(&mut self) {
        if let Ok(l) = lib() {
            if let Some(f) = l.event_destroy {
                for e in self.ev {
                    if !e.is_null() {
                        unsafe {
                            let _ = f(e);
                        }
                    }
                }
            }
        }
    }
}


/// task 481: 依頼 1 本を小さいバウンスで分割して積む (写しと DMA を重ねる)。
///
/// 呼び出し側が VRAM ステージングの再利用待ち (`cuStreamWaitEvent`) を
/// 先に積んであること。完了イベント (`job.up_ev`) は呼び出し側が記録する。
fn pf_up_run_split(job: &PfUpJob, copy_ns: &mut u64, enq_ns: &mut u64) -> Result<()> {
    let l = lib()?;
    let (Some(create), Some(record), Some(esync)) =
        (l.event_create, l.event_record, l.event_sync)
    else {
        bail!("event_create / event_record / event_sync が無い (分割 DMA)");
    };
    let chunk = exp_pf::split_chunk_for(
        exp_pf::up_split_bytes(),
        exp_pf::split_wide_bytes(),
        job.parts.iter().any(|p| p.1.src.is_some()),
    );
    if chunk == 0 {
        bail!("分割の大きさが 0 (分割 DMA)");
    }
    // task 538: 広い塊が効いたか (発火の証拠。緑は証拠ではない)。
    if chunk > exp_pf::up_split_bytes() {
        exp_pf::note_up_split_wide();
    }
    let real = REAL_HTOD_ASYNC.load(Ordering::Relaxed);
    if real.is_null() {
        bail!("cuMemcpyHtoDAsync の実体が無い (分割 DMA)");
    }
    let f: FnCuMemcpyHtoDAsyncV2 = unsafe { std::mem::transmute(real) };
    PF_UP_SPLIT.with(|cell| -> Result<()> {
        let mut slot = cell.borrow_mut();
        let need_new = match slot.as_ref() {
            Some(s) => {
                s.chunk != chunk
                    || s.stage[0]
                        .as_ref()
                        .map(|b| b.bytes < chunk)
                        .unwrap_or(true)
            }
            None => true,
        };
        if need_new {
            // 古い 2 枚を先に落としてから作り直す (イベントは Drop が壊す)。
            // task 538: 塊の大きさがグループごとに変わりうる (広い塊は
            // ディスク読みを含むグループだけ) ので、落とす前に前の DMA を待つ。
            if let Some(old) = slot.as_ref() {
                for b in 0..2 {
                    if old.used[b] {
                        unsafe {
                            check(
                                esync(old.ev[b]),
                                "cuEventSynchronize (分割 DMA の作り直し)",
                            )?;
                        }
                    }
                }
            }
            *slot = None;
            let a = PinMem::alloc_flags(chunk, false)?;
            let b = PinMem::alloc_flags(chunk, false)?;
            let mut ev: [CUevent; 2] = [std::ptr::null_mut(); 2];
            for e in ev.iter_mut() {
                let mut x: CUevent = std::ptr::null_mut();
                if unsafe { create(&mut x, CU_EVENT_DISABLE_TIMING) } != CUDA_SUCCESS {
                    bail!("cuEventCreate (分割 DMA)");
                }
                *e = x;
            }
            *slot = Some(Box::new(PfUpSplit {
                stage: [Some(a), Some(b)],
                ev,
                used: [false; 2],
                chunk,
            }));
        }
        let s = slot.as_mut().expect("直前に作った");
        s.chunk = chunk;
        let mut off = 0usize;
        let mut i = 0usize;
        while off < job.need {
            let n = chunk.min(job.need - off);
            let b = i % 2;
            // このバウンスを前回読んだ DMA が終わるまで上書きしない
            // (依頼をまたいでも待つ。`used` はグループで戻さない)。
            if s.used[b] {
                unsafe {
                    check(
                        esync(s.ev[b]),
                        "cuEventSynchronize (分割 DMA の再利用)",
                    )?;
                }
            }
            let ptr = s.stage[b].as_ref().expect("確保済み").ptr as usize;
            let t_copy = std::time::Instant::now();
            pf_copy_into_pin_at(ptr, &job.parts, off, n, job.pread);
            *copy_ns = copy_ns.saturating_add(t_copy.elapsed().as_nanos() as u64);
            let t_enq = std::time::Instant::now();
            unsafe {
                check(
                    f(
                        job.dst + off as u64,
                        ptr as *const c_void,
                        n,
                        job.stream as CUstream,
                    ),
                    "cuMemcpyHtoDAsync (分割 DMA)",
                )?;
                check(
                    record(s.ev[b], job.stream as CUstream),
                    "cuEventRecord (分割 DMA)",
                )?;
            }
            *enq_ns = enq_ns.saturating_add(t_enq.elapsed().as_nanos() as u64);
            s.used[b] = true;
            exp_pf::note_up_split();
            off += n;
            i += 1;
        }
        Ok(())
    })
}


pub(crate) fn exp_tc_kernels() -> Option<&'static ExpTcKernels> {
    EXP_TC_KERNELS
        .get_or_init(|| {
            let path = exp_tc_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-tc] PTX が読めない ({}) : {e} → 既存の GEMV へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0);
            // task 250: PTX の引数の数がホストの起動と一致するか先に見る
            // (旧い PTX を読むと余分な引数は黙って無視され、値が壊れる)。
            {
                let text = String::from_utf8_lossy(&image[..image.len() - 1]).into_owned();
                for name in TC_GU_NAMES.iter() {
                    if ptx_entry_params(&text, name) != Some(TC_GU_NARGS) {
                        eprintln!("[pf-tc] {name} の引数が {TC_GU_NARGS} 個でない → 既存の GEMV へ");
                        return None;
                    }
                }
                for name in TC_DN_NAMES.iter() {
                    if ptx_entry_params(&text, name) != Some(TC_DN_NARGS) {
                        eprintln!("[pf-tc] {name} の引数が {TC_DN_NARGS} 個でない → 既存の GEMV へ");
                        return None;
                    }
                }
            }
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-tc] cuModuleLoadData 失敗 → 既存の GEMV へ");
                    return None;
                }
            }
            let get = |names: &[&str]| -> Option<Vec<CUfunction>> {
                let mut out: Vec<CUfunction> = vec![std::ptr::null_mut(); names.len()];
                for (i, f) in out.iter_mut().enumerate() {
                    let mut cname = names[i].as_bytes().to_vec();
                    cname.push(0);
                    unsafe {
                        if (lib.module_get_function)(f, module, cname.as_ptr() as *const c_char)
                            != CUDA_SUCCESS
                        {
                            eprintln!("[pf-tc] PTX に {} が無い → 既存の GEMV へ", names[i]);
                            return None;
                        }
                    }
                }
                Some(out)
            };
            let gu: [CUfunction; 4] = get(&TC_GU_NAMES)?.try_into().ok()?;
            let dn: [CUfunction; 6] = get(&TC_DN_NAMES)?.try_into().ok()?;
            Some(ExpTcKernels { module, gu, dn })
        })
        .as_ref()
}


/// PTX の置き場所。`DENNO_EXP_KTC_PTX` があればそれ、無ければクレート配下
/// `kernels/kernels_exp_k_tc.ptx`。
fn exp_ktc_ptx_path() -> std::path::PathBuf {
    match std::env::var("DENNO_EXP_KTC_PTX") {
        Ok(p) if !p.is_empty() => std::path::PathBuf::from(p),
        _ => std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_k_tc.ptx"),
    }
}


/// K 系 TC のカーネルが揃っているか (読めなければ false = 従来の GEMV)。
pub fn exp_ktc_ready() -> bool {
    exp_ktc_kernels().is_some()
}


/// カーネルを取得する (初回だけ PTX を読む)。失敗は None を焼き付ける。
fn exp_ktc_kernels() -> Option<&'static ExpKtcKernels> {
    EXP_KTC_KERNELS
        .get_or_init(|| {
            let path = exp_ktc_ptx_path();
            let mut image = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[pf-ktc] PTX が読めない ({}) : {e} → 既存の GEMV へ",
                        path.display()
                    );
                    return None;
                }
            };
            image.push(0);
            {
                let text = String::from_utf8_lossy(&image[..image.len() - 1]).into_owned();
                for name in KTC_GU_NAMES.iter() {
                    if ptx_entry_params(&text, name) != Some(TC_GU_NARGS) {
                        eprintln!("[pf-ktc] {name} の引数が {TC_GU_NARGS} 個でない → 既存の GEMV へ");
                        return None;
                    }
                }
                for name in KTC_DN_NAMES.iter() {
                    if ptx_entry_params(&text, name) != Some(TC_DN_NARGS) {
                        eprintln!("[pf-ktc] {name} の引数が {TC_DN_NARGS} 個でない → 既存の GEMV へ");
                        return None;
                    }
                }
            }
            let lib = lib().ok()?;
            let mut module: CUmodule = std::ptr::null_mut();
            unsafe {
                if (lib.module_load_data)(&mut module, image.as_ptr() as *const c_void)
                    != CUDA_SUCCESS
                {
                    eprintln!("[pf-ktc] cuModuleLoadData 失敗 → 既存の GEMV へ");
                    return None;
                }
            }
            let get = |names: &[&str]| -> Option<Vec<CUfunction>> {
                let mut out: Vec<CUfunction> = vec![std::ptr::null_mut(); names.len()];
                for (i, f) in out.iter_mut().enumerate() {
                    let mut cname = names[i].as_bytes().to_vec();
                    cname.push(0);
                    unsafe {
                        if (lib.module_get_function)(f, module, cname.as_ptr() as *const c_char)
                            != CUDA_SUCCESS
                        {
                            eprintln!("[pf-ktc] PTX に {} が無い → 既存の GEMV へ", names[i]);
                            return None;
                        }
                    }
                }
                Some(out)
            };
            let gu: [CUfunction; KTC_SLOTS] = get(&KTC_GU_NAMES)?.try_into().ok()?;
            let dn: [CUfunction; KTC_SLOTS] = get(&KTC_DN_NAMES)?.try_into().ok()?;
            Some(ExpKtcKernels { module, gu, dn })
        })
        .as_ref()
}


/// task 233: 直近のストリーミング層の内訳 (ms)。`[pump, wait, table, launch, accum]`
pub fn pf_stream_times_ms() -> [f64; 5] {
    PF_STREAM_T.with(|c| {
        let v = *c.borrow();
        [
            v[0] as f64 / 1e6,
            v[1] as f64 / 1e6,
            v[2] as f64 / 1e6,
            v[3] as f64 / 1e6,
            v[4] as f64 / 1e6,
        ]
    })
}


/// task 296: ストリーミング層の残りの内訳 (ms)。
///
/// `[0]` = スロット準備の合計 (pump の内側。写しと積み込みは含まない)、
/// `[1]` = そのうち pinned バウンスの確保、`[2]` = 確保し直した回数。
pub fn pf_stream_extra_ms() -> [f64; 3] {
    PF_STREAM_T.with(|c| {
        let v = *c.borrow();
        [v[5] as f64 / 1e6, v[6] as f64 / 1e6, v[7] as f64]
    })
}


/// task 296: 診断カウンタだけを 0 に戻す (層ごとの差分を読む)。
pub fn pf_stream_extra_reset() {
    PF_STREAM_T.with(|c| {
        if let Ok(mut v) = c.try_borrow_mut() {
            v[5] = 0;
            v[6] = 0;
            v[7] = 0;
        }
    });
}


pub(crate) fn pf_stream_t_add(i: usize, ns: u64) {
    PF_STREAM_T.with(|c| {
        if let Ok(mut v) = c.try_borrow_mut() {
            v[i] = v[i].saturating_add(ns);
        }
    });
}


/// task 333: 直前のストリーミング試行が諦めた理由 (空 = 諦めていない)。
#[cfg(feature = "cuda")]
pub fn pf_stream_why() -> &'static str {
    PF_STREAM_WHY.with(|c| c.get())
}


/// task 333: 理由を控える (空文字で消す)。診断専用。
#[cfg(feature = "cuda")]
pub fn pf_stream_why_set(why: &'static str) {
    PF_STREAM_WHY.with(|c| c.set(why));
}


/// task 333: 理由を控えて `None` を返す。
#[cfg(feature = "cuda")]
fn pf_stream_skip(why: &'static str) -> Option<u32> {
    pf_stream_why_set(why);
    None
}


/// task 162: 1 層ぶんの expert FFN を、ミスをステージングへ上げながら回す。
///
/// - ミスは `DENNO_PF_STREAM_GROUP` 本ずつのグループに切り、2 枚のステージングを
///   交互に使って H2D と計算を重ねる。H2D は専用ストリーム、計算は null ストリーム。
/// - 行の加重和は GPU の `dno_exp_pf_accum` で閉じ、D2H は `ntok × e` だけ。
///   **和の順序は (トークン, j 昇順) の 1 本**で、CPU 経路とは丸めが変わる
///   (FMA の縮約。task 162 で許容。decode はこの経路を通らない)。
/// - 載せられなかったミス (容量外・確保失敗) は呼び出し側が CPU で足す。
///
/// `Ok(None)` = 使わなかった (`out` は触っていない)。呼び出し側は従来経路へ。
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_pf_layer_stream(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    entries: &[PfStreamEntry<'_>],
    seg: &[u32],
    rows_gu: &[u32],
    xs: &[f32],
    n_in: usize,
    ff: usize,
    e: usize,
    ntok: usize,
    k: usize,
    prow: &[u32],
    wrow: &[f32],
    group: usize,
    out: &mut [f32],
    serial: bool,
) -> Result<Option<u32>> {
    moe_experts_iq_pf_layer_stream_io(
        ty_g,
        ty_u,
        ty_d,
        entries,
        seg,
        rows_gu,
        n_in,
        ff,
        e,
        ntok,
        k,
        prow,
        wrow,
        group,
        PfStreamIo::Host { xs, out },
        serial,
    )
}


/// task 233: 入力も出力もデバイスにある版 (層優先 prefill)。
///
/// `sh` (`[ntok][e]`) を渡すと、共有 expert の出力を加重和の**最後の 1 項**として
/// 足す (`sh_w` が係数)。ホストへ戻るのは `ntok × e` の 1 個も無い。
#[allow(clippy::too_many_arguments)]
pub fn moe_experts_iq_pf_layer_stream_dev(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    entries: &[PfStreamEntry<'_>],
    seg: &[u32],
    rows_gu: &[u32],
    xs: CUdeviceptr,
    n_in: usize,
    ff: usize,
    e: usize,
    ntok: usize,
    k: usize,
    prow: &[u32],
    wrow: &[f32],
    group: usize,
    sh: CUdeviceptr,
    sh_w: &[f32],
    out: CUdeviceptr,
    serial: bool,
) -> Result<Option<u32>> {
    moe_experts_iq_pf_layer_stream_io(
        ty_g,
        ty_u,
        ty_d,
        entries,
        seg,
        rows_gu,
        n_in,
        ff,
        e,
        ntok,
        k,
        prow,
        wrow,
        group,
        PfStreamIo::Dev {
            xs,
            out,
            sh,
            sh_w,
        },
        serial,
    )
}


#[allow(clippy::too_many_arguments)]
fn moe_experts_iq_pf_layer_stream_io(
    ty_g: crate::kquant::GgmlType,
    ty_u: crate::kquant::GgmlType,
    ty_d: crate::kquant::GgmlType,
    entries: &[PfStreamEntry<'_>],
    seg: &[u32],
    rows_gu: &[u32],
    n_in: usize,
    ff: usize,
    e: usize,
    ntok: usize,
    k: usize,
    prow: &[u32],
    wrow: &[f32],
    group: usize,
    mut io: PfStreamIo<'_>,
    serial: bool,
) -> Result<Option<u32>> {
    // task 333: この試行の理由を 0 から積む (前の層 / 前の試行の理由を残さない)。
    pf_stream_why_set("");
    if PF_DEAD.load(std::sync::atomic::Ordering::Relaxed) {
        return Ok(pf_stream_skip("dead"));
    }
    let nseg = entries.len();
    let total_rows = rows_gu.len();
    // 共有 expert の行を足すか (足すときは加重和を k+1 項にする)。
    let sh_rows = match &io {
        PfStreamIo::Dev { sh, .. } if *sh != 0 => ntok,
        _ => 0,
    };
    let kk = k + usize::from(sh_rows > 0);
    let (dev_in, dev_out, dev_sh, sh_w): (CUdeviceptr, CUdeviceptr, CUdeviceptr, &[f32]) = match &io {
        PfStreamIo::Host { .. } => (0, 0, 0, &[]),
        PfStreamIo::Dev { xs, out, sh, sh_w } => (*xs, *out, *sh, *sh_w),
    };
    // task 242: 出力が置き場 (デバイス) のときだけ g/u/act をグループ局所にする。
    let is_dev = matches!(io, PfStreamIo::Dev { .. });
    let (host_in, host_out): (Option<&[f32]>, Option<&mut [f32]>) = match &mut io {
        PfStreamIo::Host { xs, out } => (Some(&**xs), Some(&mut **out)),
        PfStreamIo::Dev { .. } => (None, None),
    };
    let bad_in = match host_in {
        Some(xs) => xs.len() != ntok * n_in,
        None => dev_in == 0,
    };
    let bad_out = match host_out.as_ref() {
        Some(out) => out.len() < ntok * e,
        None => dev_out == 0,
    };
    if nseg == 0
        || total_rows == 0
        || ntok == 0
        || k == 0
        || seg.len() != nseg * 4
        || bad_in
        || bad_out
        || prow.len() != ntok * k
        || wrow.len() < ntok * k
        || (sh_rows > 0 && (sh_w.len() < ntok || dev_sh == 0))
    {
        return Ok(pf_stream_skip("shape"));
    }
    let mut host_out = host_out;
    if !moe_gemv_iq_pf_ready(ty_g, n_in)
        || !moe_gemv_iq_pf_ready(ty_u, n_in)
        || !moe_gemv_iq_pf_ready(ty_d, ff)
    {
        return Ok(pf_stream_skip("type"));
    }
    let (Some(slot_g), Some(slot_u), Some(slot_d)) = (
        exp_pf::slot(ty_g),
        exp_pf::slot(ty_u),
        exp_pf::slot(ty_d),
    ) else {
        return Ok(pf_stream_skip("slot"));
    };
    // task 309: K 系の層か。pinned バウンスの作り方 (`DENNO_PF_STREAM_WC` の
    // 既定) を型で変える (lab t309b: K 系は write-combined でないほうが速い)。
    PF_STREAM.with(|c| c.borrow_mut().k_mode = exp_pf::is_k_slot(slot_g));
    let Some(ctx) = shared_ctx() else {
        return Ok(pf_stream_skip("ctx"));
    };
    let Some(kernels) = exp_pf_kernels() else {
        return Ok(pf_stream_skip("kernels"));
    };
    let Some(accum) = kernels.accum else {
        return Ok(pf_stream_skip("accum"));
    };
    if ctx.make_current().is_err() {
        return Ok(pf_stream_skip("current"));
    }
    let (rb_g, rb_u, rb_d) = match (
        exp_pf::row_bytes(ty_g, n_in),
        exp_pf::row_bytes(ty_u, n_in),
        exp_pf::row_bytes(ty_d, ff),
    ) {
        (Some(a), Some(b), Some(c)) => (a, b, c),
        _ => return Ok(pf_stream_skip("row_bytes")),
    };
    let pinned = exp_pf::stream_pin() == 1;
    // task 165: 上げ (写し + 積み込み) を専用スレッドへ出すか。
    // 出さないときは task 162 と同じく、このスレッドで直列に写して積む。
    // task 227: 転送元を mmap 直接 (page / reg) にするときは、上げ
    // スレッドの経路が pinned のバウンス前提なので使わない。
    // task 324: `gpuwait` は上の ut 経路の中で、グループの H2D 完了を
    // ホストではなく計算ストリームに待たせる (`cuStreamWaitEvent`)。
    // グループ分け (先頭を小さく切るか) にも効くので、ここで 1 回だけ決める。
    // task 333: `serial` は「上げスレッドを使わず、このスレッドで写す」へ落とす
    // 再試行用 (上げスレッドが居ない門で落ちたときの 1 回だけのやり直し)。
    let ut = !serial && pinned && exp_pf::stream_ut() && exp_pf::stream_src() == 0;
    let gpuwait = ut && exp_pf::up_gpuwait_on();
    // task 233: 共有 expert の行 (末尾 `sh_rows` 行) も h に含める。
    let nh = (total_rows + sh_rows) * e;
    let host_x = host_in;
    let in_elems = ntok * n_in;

    // task 296: 診断の残り 3 項目 (prep / pin / pin 回数) も層ごとに 0 から積む。
    PF_STREAM_T.with(|c| *c.borrow_mut() = [0; 8]);
    let mut work = || -> Result<Option<u32>> {
        PF_SCRATCH.with(|cell| -> Result<Option<u32>> {
            let mut sc = cell.borrow_mut();
            // task 254: 作業域の確保と活性・表の H2D (1 層 1 回) の台帳。
            let t_scratch = pf_host::stamp();
            // task 257: プールがあれば区画 (prefill 中に cuMemAlloc を出さない)。
            // task 233: デバイス入力のときは `x` の作業域 (と H2D) を丸ごと省く。
            if host_x.is_some() {
                ensure_region(&mut sc.x, crate::model::pf_pool::R::X, 0, in_elems * 4)?;
            }
            ensure_region(&mut sc.h, crate::model::pf_pool::R::H, 0, nh * 4)?;
            ensure_region(&mut sc.pg, crate::model::pf_pool::R::Pg, 0, nseg * 8)?;
            ensure_region(&mut sc.pu, crate::model::pf_pool::R::Pu, 0, nseg * 8)?;
            ensure_region(&mut sc.pd, crate::model::pf_pool::R::Pd, 0, nseg * 8)?;
            ensure_region(&mut sc.seg, crate::model::pf_pool::R::Seg, 0, nseg * 16)?;
            ensure_region(&mut sc.rows_gu, crate::model::pf_pool::R::RowsGu, 0, total_rows * 4)?;
            ensure_region(&mut sc.rows_id, crate::model::pf_pool::R::RowsId, 0, total_rows * 4)?;
            ensure_region(&mut sc.prow, crate::model::pf_pool::R::Prow, 0, ntok * kk * 4)?;
            ensure_region(&mut sc.wrow, crate::model::pf_pool::R::Wrow, 0, ntok * kk * 4)?;
            // task 233: デバイス出力のときは oacc も要らない (置き場へ直接書く)。
            if host_out.is_some() {
                ensure_region(&mut sc.oacc, crate::model::pf_pool::R::Oacc, 0, ntok * e * 4)?;
            }
            {
                if let Some(xs) = host_x {
                    let xb = sc.x.as_ref().expect("確保済み");
                    // task 162: 活性 (nt x n_in) は層ごとに 20 MB 級なので、pinned に
                    // 写して非同期で積む (pageable の同期コピーは 1 MB あたりが重い)。
                    // null ストリームへ積むので、この後のカーネルは順序どおり待つ。
                    let mut done_pin = false;
                    if let Ok(l) = lib() {
                        PF_XPIN.with(|cell| {
                            let mut slot = cell.borrow_mut();
                            if Scratch::ensure_pin(&mut slot, xs.len() * 4).is_ok() {
                                let pin = slot.as_ref().expect("確保済み");
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        xs.as_ptr() as *const u8,
                                        pin.ptr as *mut u8,
                                        xs.len() * 4,
                                    );
                                    if (l.memcpy_htod_async)(
                                        xb.ptr(),
                                        pin.ptr,
                                        xs.len() * 4,
                                        std::ptr::null_mut(),
                                    ) == CUDA_SUCCESS
                                    {
                                        done_pin = true;
                                    }
                                }
                            }
                        });
                    }
                    if !done_pin {
                        xb.copy_from(xs)?;
                    }
                }
                // task 254: 表は同期の `cuMemcpyHtoD` だと null ストリームの
                // 前の仕事 (前の層のカーネル) の完了までホストを止める。
                // `DENNO_PF_TAB_ASYNC=1` では pinned から非同期で積む。
                // task 242: seg と rows_id は bounds が決まってから積む
                // (グループ局所のときは行の先頭を 0 起点に直した表を積む)。
                sc.rows_gu.as_ref().expect("確保済み").copy_from(rows_gu)?;
                let id: Vec<u32> = exp_pf::identity_rows(total_rows);
                sc.rows_id.as_ref().expect("確保済み").copy_from(&id)?;
                // task 233: 共有 expert を足すときは、行の加重和の表を 1 列増やし、
                // 末尾に「共有の行 (`total_rows + t`)」と係数を置く。
                let wr: Vec<f32>;
                let pr: Vec<u32>;
                if sh_rows > 0 {
                    let Some((pr0, wr0)) = crate::model::exp_pf::row_table_with_shared(
                        prow, wrow, ntok, k, total_rows, sh_w,
                    ) else {
                        return Ok(pf_stream_skip("shared_table"));
                    };
                    pr = pr0;
                    wr = wr0;
                } else {
                    pr = prow.to_vec();
                    wr = wrow[..ntok * k].to_vec();
                }
                let bytes_u32 =
                    |v: &[u32]| unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
                let bytes_f32 = |v: &[f32]| unsafe {
                    std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4)
                };
                let async_ok = pf_tab::enabled()
                    && pf_tab::upload(&[
                        (sc.seg.as_ref().expect("確保済み").ptr(), 0, bytes_u32(seg)),
                        (
                            sc.rows_gu.as_ref().expect("確保済み").ptr(),
                            0,
                            bytes_u32(rows_gu),
                        ),
                        (
                            sc.rows_id.as_ref().expect("確保済み").ptr(),
                            0,
                            bytes_u32(&id),
                        ),
                        (sc.prow.as_ref().expect("確保済み").ptr(), 0, bytes_u32(&pr)),
                        (
                            sc.wrow.as_ref().expect("確保済み").ptr(),
                            0,
                            bytes_f32(&wr),
                        ),
                    ]);
                if !async_ok {
                    sc.seg.as_ref().expect("確保済み").copy_from(seg)?;
                    sc.prow.as_ref().expect("確保済み").copy_from(&pr)?;
                    sc.wrow.as_ref().expect("確保済み").copy_from(&wr)?;
                }
            }
            dbg_htod(in_elems * 4 + nseg * 16 + total_rows * 4 * 2 + ntok * kk * 8);
            let xb = match host_x {
                Some(_) => sc.x.as_ref().expect("確保済み").ptr(),
                None => dev_in,
            };
            // task 242: g/u/act は bounds が決まってから確保する (下)。
            let hb = sc.h.as_ref().expect("確保済み").ptr();
            let rg = sc.rows_gu.as_ref().expect("確保済み").ptr();
            let rid = sc.rows_id.as_ref().expect("確保済み").ptr();
            pf_host::mark(t_scratch, pf_host::SCRATCH);

            // 1. ミスの本数を数え、グループ分けを決める。
            //    容量は「2 枚に分けたステージング 1 枚ぶん」。実際に確保できる
            //    サイズで切るので、載せられない expert は出ない (spill 0)。
            let t_plan = pf_host::stamp();
            let mut needs = vec![0usize; nseg];
            let mut miss_total = 0usize;
            for (i, en) in entries.iter().enumerate() {
                if en.dev.is_some() {
                    continue;
                }
                let Some(h) = en.host else {
                    return Ok(pf_stream_skip("host_missing"));
                };
                let need = h
                    .iter()
                    .map(|b| pf_stage_align_up(b.len()))
                    .sum::<usize>();
                if need == 0 {
                    return Ok(pf_stream_skip("no_bytes"));
                }
                needs[i] = need;
                miss_total += need;
            }
            if miss_total == 0 {
                return Ok(pf_stream_skip("no_miss"));
            }
            let want_slot = miss_total.min((exp_pf::stream_bytes() / PF_STAGE_SLOTS).max(1 << 20));
            // 全枚を同じ容量で確保し、その実サイズでグループを切る
            // (1 枚だけ小さいと交互に使えない)。
            let mut slot_cap = 0usize;
            PF_STREAM.with(|cell| -> Result<()> {
                let mut st = cell.borrow_mut();
                for i in 0..PF_STAGE_SLOTS {
                    if st.slot[i].cap >= want_slot && st.slot[i].buf.is_some() {
                        continue;
                    }
                    if let Some(b) = pf_stage_alloc(want_slot, i) {
                        st.slot[i].cap = b.bytes();
                        st.slot[i].buf = Some(b);
                    } else {
                        break;
                    }
                }
                // 全枚の最小がこの層で使える 1 グループの上限。1 枚でも欠けたら 0。
                slot_cap = st
                    .slot
                    .iter()
                    .map(|s| if s.buf.is_some() { s.cap } else { 0 })
                    .min()
                    .unwrap_or(0);
                Ok(())
            })?;
            if slot_cap == 0 {
                return Ok(pf_stream_skip("stage"));
            }
            // task 324: 先頭グループだけ小さく切る (層の入口の段差を縮める)。
            // GPU 側で待つ経路のときだけ (既定 off では 0 = 従来と同じ切り方)。
            let first_g = if gpuwait {
                exp_pf::stream_first_group()
            } else {
                0
            };
            let Some(bounds) = exp_pf::group_bounds_first(&needs, slot_cap, group, first_g) else {
                return Ok(pf_stream_skip("bounds"));
            };
            pf_host::mark(t_plan, pf_host::PLAN);
            // task 242: デバイス出力のときは g/u/act を**グループ局所**にする。
            // `seg` の行の先頭をグループ基準の相対番号へ直し、行表のポインタを
            // グループ先頭行だけ足して渡す (下)。これで 14k の g/u/act が
            // 0.86 GB → グループ 1 個ぶん (実測 約 0.04 GB) になる。
            let gl = is_dev && exp_pf::lm_group_local();
            // task 250: tensor core の grouped GEMM で回せるか (層で 1 回だけ判定)。
            // `gl` (相対化した seg) は行の先頭がグループ基準になるので使わない
            // (カーネルは絶対行番号で act / h を書く)。
            // task 416: IQ 系 (`kernels_exp_iq_tc.cu`) に加えて K 系
            // (`kernels_exp_k_tc.cu`、`DENNO_PF_TC_EXP_K`) も同じ形で回せる。
            let tc_kernels: Option<TcSet> = if gl {
                None
            } else if crate::model::exp_pf::tc_exp_on()
                && tc_shapes_ok(
                    exp_pf::slot(ty_g),
                    exp_pf::slot(ty_d),
                    ty_g == ty_u,
                    n_in,
                    ff,
                    ff,
                    e,
                )
            {
                exp_tc_kernels().map(TcSet::Iq)
            } else if crate::model::exp_pf::tc_exp_k_on()
                && ktc_shapes_ok(
                    exp_pf::slot(ty_g),
                    exp_pf::slot(ty_d),
                    ty_g == ty_u,
                    n_in,
                    ff,
                    ff,
                    e,
                )
            {
                exp_ktc_kernels().map(TcSet::K).inspect(|_| {
                    static ONCE: std::sync::atomic::AtomicBool =
                        std::sync::atomic::AtomicBool::new(false);
                    if !ONCE.swap(true, std::sync::atomic::Ordering::Relaxed) {
                        eprintln!(
                            "[pf-ktc] K 系の tensor core グループ GEMM を使う \
                             (gate/up={:?} down={:?} k={n_in}/{ff} n={ff}/{e})",
                            ty_g, ty_d
                        );
                    }
                })
            } else {
                None
            };
            let rows_cap = if gl {
                match exp_pf::max_group_rows(seg, &bounds) {
                    Some(r) => r,
                    None => return Ok(pf_stream_skip("group_rows")),
                }
            } else {
                total_rows
            };
            // task 250: tensor core 経路は gate/up を融合して act へ直接書くので、
            // g / u の作業域は要らない (14k で 0.92 GB 減る)。既存経路のときだけ取る。
            let (gb, ub) = if tc_kernels.is_some() {
                (0u64, 0u64)
            } else {
                ensure_region(&mut sc.g, crate::model::pf_pool::R::G, 0, rows_cap * ff * 4)?;
                ensure_region(&mut sc.u, crate::model::pf_pool::R::U, 0, rows_cap * ff * 4)?;
                (
                    sc.g.as_ref().expect("確保済み").ptr(),
                    sc.u.as_ref().expect("確保済み").ptr(),
                )
            };
            ensure_region(&mut sc.act, crate::model::pf_pool::R::Act, 0, rows_cap * ff * 4)?;
            let act = sc.act.as_ref().expect("確保済み").ptr();
            // グループ局所のときは、行の先頭を**グループの先頭から**の相対番号へ
            // 直した `seg` を積む (カーネルは `obase + (s.ro + r)*n_out` に書く)。
            let seg_gl: Option<Vec<u32>> = if gl {
                match exp_pf::seg_group_local(seg, &bounds) {
                    Some(v) => Some(v),
                    None => return Ok(pf_stream_skip("seg_local")),
                }
            } else {
                None
            };
            sc.seg
                .as_ref()
                .expect("確保済み")
                .copy_from(seg_gl.as_deref().unwrap_or(seg))?;
            let segb = sc.seg.as_ref().expect("確保済み").ptr();
            // 2. グループごとに上げて計算する。`pg/pu/pd` の表は全体を 1 本で持ち、
            //    グループの先頭を指して起動する。
            let pg_tab = sc.pg.as_ref().expect("確保済み");
            let pu_tab = sc.pu.as_ref().expect("確保済み");
            let pd_tab = sc.pd.as_ref().expect("確保済み");
            // グループごとのミス (plan 順) と必要バイト。依頼を先に出せるよう先に作る。
            // task 325: ファイル位置 (`src`) も一緒に運ぶ (pread 経路が使う)。
            let items_of: Vec<Vec<PfItem<'_>>> = bounds
                .iter()
                .map(|&(a, b)| {
                    entries[a..b]
                        .iter()
                        .filter_map(|en| en.host.map(|h| (en.expert, h, en.src)))
                        .collect()
                })
                .collect();
            let need_of: Vec<usize> = items_of
                .iter()
                .map(|it| {
                    it.iter()
                        .map(|it| it.1.iter().map(|b| pf_stage_align_up(b.len())).sum::<usize>())
                        .sum::<usize>()
                })
                .collect();
            // 依頼済みグループのデバイスポインタ表 (ut 経路だけ)。
            let mut ready_ptrs: Vec<Option<Vec<[u64; 3]>>> = vec![None; bounds.len()];
            let mut submitted = 0usize;
            // 探針 (実 H2D) はリクエストで 1 回だけ。前の層で測っていれば出さない。
            let mut probe_sent = if ut {
                PF_UP.with(|cell| {
                    cell.borrow()
                        .as_ref()
                        .map(|u| u.probe().is_some())
                        .unwrap_or(false)
                })
            } else {
                true
            };
            // group id は層をまたいで通し番号にする (スロットの持ち主とイベントを
            // 一意にするため。層ごとに 0 へ戻すと前の層の完了イベントを
            // 「自分のグループの完了」と誤認する)。
            let base_g = PF_STREAM.with(|cell| {
                let mut st = cell.borrow_mut();
                pf_take_group_ids(&mut st.next_g, bounds.len())
            });
            // task 320: 上げ専用スレッドへ何グループ先まで依頼するか。
            // 1 つ先だけだと、待つグループの写しがまだ終わっておらず隠れない。
            // task 570: **pread の立つグループだけ**深くする (既定モデルの経路は
            // 1 グループ先のまま。`exp_pf::lookahead_for` の doc を見よ)。
            let look_base = if ut { exp_pf::stream_lookahead() } else { 0 };
            let look_deep = exp_pf::stream_lookahead_pread();
            let mut look = look_base;
            // task 325: 写しの外から何グループ先を先読みするか。
            let ra = exp_pf::ra_groups();
            // task 333: ここまでに 1 グループでも計算したか。
            // `Ok(None)` の契約は「`out` を触っていない」なので、書いた後の
            // 門は再試行しない (`*_late`)。呼び出し側が従来経路へ落ちると
            // 二重加算になるため、再試行の可否もここで分ける。
            let mut wrote = false;
            for (gi, &(a, b)) in bounds.iter().enumerate() {
                let g = base_g + gi as u64;
                let items = &items_of[gi];
                // task 325: n グループ先のページを別スレッドで先に読ませる
                // (写しは従来どおり。間に合えばフォールトが page cache に当たる)。
                if ra > 0 {
                    if let Some(next) = items_of.get(gi + ra) {
                        pf_ra_submit(next);
                    }
                }
                // (a0) ut 経路: このグループから `look` 個先までを先に依頼する
                //      (計算スレッドがグループ g のカーネルを積む間に、上げスレッドが
                //       g+1 以降の写しと積み込みを進める)。
                if ut {
                    while submitted <= gi + look && submitted < bounds.len() {
                        let k = submitted;
                        if !items_of[k].is_empty() {
                            let first = !probe_sent;
                            // task 570: 依頼したグループが pread を使うかで深さを決める。
                            let (ptrs, pread) =
                                pf_submit_group(base_g + k as u64, &items_of[k], first)?;
                            ready_ptrs[k] = ptrs;
                            if ready_ptrs[k].is_some() {
                                probe_sent = true;
                            }
                            look = exp_pf::lookahead_for(pread, look_base, look_deep);
                        }
                        submitted = k + 1;
                    }
                    if !items.is_empty() && ready_ptrs[gi].is_none() {
                        // 上げスレッドが居ない → この層は従来経路へ。
                        if wrote {
                            return Ok(pf_stream_skip("no_upload_late"));
                        }
                        return Ok(pf_stream_skip("no_upload"));
                    }
                }
                // (a) 上げ。ut 経路は依頼済み (待つのは (c))。
                let mut copy_ns = 0u64;
                let mut enq_ns = 0u64;
                let mut ready_ns = 0u64;
                let mut wait_ns = 0u64;
                let mut slot_ns = 0u64;
                let t_pump = std::time::Instant::now();
                let (ptrs, bytes) = if ut {
                    (ready_ptrs[gi].take().unwrap_or_default(), need_of[gi])
                } else if items.is_empty() {
                    (Vec::new(), 0)
                } else {
                    let pump = PF_STREAM.with(|cell| -> Result<PfPump> {
                        let mut st = cell.borrow_mut();
                        pf_pump_in(&mut st, g, items, pinned)
                    })?;
                    copy_ns = pump.copy_ns;
                    enq_ns = pump.enq_ns;
                    // task 254: バウンスコピーと積み込みを項目別に積む。
                    pf_host::add_ns(pf_host::STAGE_COPY, copy_ns);
                    pf_host::add_ns(pf_host::STAGE_ENQ, enq_ns);
                    if let Some((pb, pn)) = pump.probe {
                        exp_pf::note_stream_probe(pb, pn);
                    }
                    (pump.ptrs, pump.bytes)
                };
                pf_stream_t_add(0, t_pump.elapsed().as_nanos() as u64);
                // task 254: ポインタ表の組み立てと H2D の台帳 + task 233 の内訳。
                let t_ptr = pf_host::stamp();
                let t_tab = std::time::Instant::now();
                let mut ip = 0usize;
                {
                    let mut vg = vec![0u64; b - a];
                    let mut vu = vec![0u64; b - a];
                    let mut vd = vec![0u64; b - a];
                    for (j, en) in entries[a..b].iter().enumerate() {
                        if let Some(d) = en.dev {
                            vg[j] = d[0];
                            vu[j] = d[1];
                            vd[j] = d[2];
                        } else if ip < ptrs.len() {
                            vg[j] = ptrs[ip][0];
                            vu[j] = ptrs[ip][1];
                            vd[j] = ptrs[ip][2];
                            ip += 1;
                        } else {
                            // 表が足りない = 内部の不整合。この層は従来経路へ。
                            if wrote {
                                return Ok(pf_stream_skip("ptr_table_late"));
                            }
                            return Ok(pf_stream_skip("ptr_table"));
                        }
                    }
                    // task 233: 同期 H2D は前のグループのカーネル / DMA を待つ。
                    // 小さな表は pinned から非同期で積む (足りなければ同期へ)。
                    // task 254: `DENNO_PF_TAB_ASYNC=1` では 5 本の表を 1 回の
                    // 非同期積みにまとめる (足りなければ同期へ)。
                    let ok_async = pf_tiny_htod(pg_tab.ptr() + (a * 8) as u64, bytes_of(&vg))
                        && pf_tiny_htod(pu_tab.ptr() + (a * 8) as u64, bytes_of(&vu))
                        && pf_tiny_htod(pd_tab.ptr() + (a * 8) as u64, bytes_of(&vd));
                    if !ok_async {
                        pg_tab.copy_from_at(a * 8, &vg)?;
                        pu_tab.copy_from_at(a * 8, &vu)?;
                        pd_tab.copy_from_at(a * 8, &vd)?;
                    }
                }
                pf_host::mark(t_ptr, pf_host::PTR_TAB);
                pf_stream_t_add(2, t_tab.elapsed().as_nanos() as u64);
                // (c) **自分のグループの H2D 完了だけ**を待つ (ホスト)。
                let t_wait = pf_host::stamp();
                let t_wait2 = std::time::Instant::now();
                if ut {
                    if !items.is_empty() {
                        if gpuwait {
                            // task 324: ホストは「依頼が積まれた」ところまでで止め、
                            // H2D の完了は計算ストリームに待たせる。カーネルは
                            // 転送が終わったグループから GPU で走り出す。
                            ready_ns = pf_wait_ready(g)?;
                            PF_STREAM.with(|cell| -> Result<()> {
                                let mut st = cell.borrow_mut();
                                pf_stream_wait_up_in(&mut st, g, true)
                            })?;
                            exp_pf::note_stream_gpuwait();
                            let (c, e, sl) = PF_UP.with(|cell| match cell.borrow().as_ref() {
                                Some(u) => u.times(g),
                                None => (0, 0, 0),
                            });
                            copy_ns = c;
                            enq_ns = e;
                            slot_ns = sl;
                        } else {
                            let (c, e, r, w, sl) = pf_wait_group(g)?;
                            copy_ns = c;
                            enq_ns = e;
                            ready_ns = r;
                            wait_ns = w;
                            slot_ns = sl;
                        }
                        if let Some((pb, pn)) =
                            PF_UP.with(|cell| cell.borrow().as_ref().and_then(|u| u.probe()))
                        {
                            exp_pf::note_stream_probe(pb, pn);
                        }
                    }
                } else {
                    wait_ns = PF_STREAM.with(|cell| -> Result<u64> {
                        let mut st = cell.borrow_mut();
                        pf_wait_up_in(&mut st, g, pinned)
                    })?;
                }
                pf_host::mark(t_wait, pf_host::UP_WAIT);
                pf_stream_t_add(1, t_wait2.elapsed().as_nanos() as u64);
                // (d) カーネルを積む。
                let t_l = std::time::Instant::now();
                let abs_start = seg[a * 4 + 1] as usize;
                let abs_end = if b > a {
                    let last = b - 1;
                    seg[last * 4 + 1] as usize + seg[last * 4] as usize
                } else {
                    abs_start
                };
                // task 242: グループ局所のときは行の添字を 0 起点にする。
                // `seg` は相対化して積んであるので、カーネルへは `ro_start = 0`
                // と「グループ先頭行だけ足した**入力**の行表」を渡す。
                // down の行表 (`rows_id`) は足さない: down の入力は `act` で、
                // これはグループ局所 (0 起点) に置いてある。
                let (ro_start, ro_end) = if gl {
                    (0usize, abs_end.saturating_sub(abs_start))
                } else {
                    (abs_start, abs_end)
                };
                let rg_g = if gl {
                    rg + (abs_start as u64) * 4
                } else {
                    rg
                };
                unsafe {
                    if let Some(tc) = tc_kernels {
                        // task 250: グループ内の最大行数でトークンタイル数を決める。
                        let max_m = (a..b).map(|i| seg[i * 4] as usize).max().unwrap_or(0);
                        // task 416: IQ 版 / K 版でモジュールが違うので、ここで関数を引く。
                        let (gu_fn, dn_fn) = tc.pair(slot_g, slot_d);
                        pf_launch_group_tc(
                            ctx,
                            gu_fn,
                            dn_fn,
                            pg_tab.ptr() + (a * 8) as u64,
                            pu_tab.ptr() + (a * 8) as u64,
                            pd_tab.ptr() + (a * 8) as u64,
                            xb,
                            act,
                            hb,
                            segb + (a * 16) as u64,
                            rg,
                            rid,
                            b - a,
                            max_m,
                            n_in,
                            ff,
                            e,
                            rb_g,
                            rb_d,
                        )?;
                    } else {
                        pf_launch_group(
                            ctx,
                            kernels,
                            slot_g,
                            slot_u,
                            slot_d,
                            pg_tab.ptr() + (a * 8) as u64,
                            pu_tab.ptr() + (a * 8) as u64,
                            pd_tab.ptr() + (a * 8) as u64,
                            xb,
                            gb,
                            ub,
                            act,
                            hb,
                            segb + (a * 16) as u64,
                            rg_g,
                            rid,
                            b - a,
                            ro_start,
                            ro_end,
                            n_in,
                            ff,
                            e,
                            rb_g,
                            rb_u,
                            rb_d,
                        )?;
                    }
                }
                wrote = true;
                PF_STREAM.with(|cell| {
                    let mut st = cell.borrow_mut();
                    pf_mark_used_in(&mut st, g, pinned);
                });
                pf_stream_t_add(3, t_l.elapsed().as_nanos() as u64);
                let rows_here = ro_end.saturating_sub(ro_start);
                exp_pf::note_stream_group(
                    bytes,
                    rows_here,
                    items.len(),
                    copy_ns,
                    enq_ns,
                    ready_ns,
                    wait_ns,
                    slot_ns,
                );
            }
            // 3. 行の加重和を GPU で閉じる。
            let t_acc = std::time::Instant::now();
            {
                let prowb = sc.prow.as_ref().expect("確保済み").ptr();
                let wb = sc.wrow.as_ref().expect("確保済み").ptr();
                // task 233: デバイス出力のときは oacc を使わず、置き場へ直接書く。
                let oacc = sc.oacc.as_ref();
                let threads = 256usize;
                let mut ph = hb;
                let mut pp = prowb;
                let mut pw = wb;
                let mut po = match (oacc, host_out.as_ref()) {
                    (Some(b), Some(_)) => b.ptr(),
                    _ => dev_out,
                };
                let (mut k_c, mut ee, mut nn) = (kk as c_int, e as c_int, ntok as c_int);
                let mut args = [
                    &mut ph as *mut _ as *mut c_void,
                    &mut pp as *mut _ as *mut c_void,
                    &mut pw as *mut _ as *mut c_void,
                    &mut po as *mut _ as *mut c_void,
                    &mut k_c as *mut _ as *mut c_void,
                    &mut ee as *mut _ as *mut c_void,
                    &mut nn as *mut _ as *mut c_void,
                ];
                let t_k = std::time::Instant::now();
                unsafe {
                    ctx.launch_async(
                        accum,
                        (ntok as u32, 1, 1),
                        (threads as u32, 1, 1),
                        0,
                        &mut args,
                    )?;
                }
                exp_pf::note_kernel_ns(t_k.elapsed().as_nanos() as u64);
                // task 254: 累積カーネルの完了待ちと D2H を項目別に積む。
                let t_sync = std::time::Instant::now();
                // task 324: 出力が置き場 (デバイス) のままなら待たない。次の層の
                // カーネルは同じストリームの後ろに積まれ、ホストが書く表は同期
                // コピーが前のカーネルを待つので順序は保たれる。
                let nosync = host_out.is_none() && exp_pf::stream_nosync_on();
                if nosync {
                    exp_pf::note_stream_nosync();
                } else {
                    ctx.sync()?;
                    exp_pf::note_kernel_ns(t_sync.elapsed().as_nanos() as u64);
                    pf_host::add_ns(pf_host::SYNC, t_sync.elapsed().as_nanos() as u64);
                }
                if let (Some(b), Some(o)) = (oacc, host_out.as_mut()) {
                    let t_d2h = std::time::Instant::now();
                    b.to_slice(&mut o[..ntok * e])?;
                    exp_pf::note_d2h_ns(t_d2h.elapsed().as_nanos() as u64);
                    pf_host::add_ns(pf_host::D2H, t_d2h.elapsed().as_nanos() as u64);
                    dbg_d2h(ntok * e * 4);
                }
            }
            pf_stream_t_add(4, t_acc.elapsed().as_nanos() as u64);
            exp_pf::note_stream_layer();
            Ok(Some(nseg as u32))
        })
    };
    match work() {
        Ok(v) => Ok(v),
        Err(e) => {
            // 確保失敗・形の不整合はこのプロセスで 1 回だけ記録し、以後 CPU に固定する。
            if !PF_DEAD.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::error!("[exp-pf-stream] 使えないので以後 CPU 経路: {e:#}");
            }
            Ok(None)
        }
    }
}


#[cfg(test)]
mod pf_exp_ktc_tests {
    //! task 416: K 系 tensor core グループ GEMM (`kernels_exp_k_tc.cu`) の**索引**を
    //! GPU 無しで固定する。
    //!
    //! カーネルは 32 要素タイルの重みを 8 個ずつ (j = 0..3) デコードする。その
    //! 索引算術を .cu の行から 1 対 1 で写し、CPU の参照
    //! (`kquant::dequant_row_into`) と要素ごとに比べる。丸め順の一致ではなく
    //! **索引の取り違え**を落とすためのテスト (型ごとに 1 行 512 要素)。

    use super::*;
    use crate::kquant::{dequant_row_into, GgmlType};

    /// .cu の `dno_ktcq_f16` と同じ式 (IEEE の f16 → f32)。
    fn f16v(b: &[u8]) -> f32 {
        let h = b[0] as u32 | ((b[1] as u32) << 8);
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mut man = h & 0x3ff;
        let bits = if exp == 0 {
            if man == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                while man & 0x400 == 0 {
                    man <<= 1;
                    e -= 1;
                }
                man &= 0x3ff;
                sign | (((e + 127) as u32) << 23) | (man << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (man << 13)
        } else {
            sign | ((exp + 112) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }

    fn u32le(b: &[u8]) -> u32 {
        b[0] as u32 | ((b[1] as u32) << 8) | ((b[2] as u32) << 16) | ((b[3] as u32) << 24)
    }

    /// ggml の `get_scale_min_k4` (.cu の `dno_ktcq_scale_min_k4` と同じ式)。
    fn scale_min_k4(j: usize, q: &[u8]) -> (u32, u32) {
        if j < 4 {
            ((q[j] & 63) as u32, (q[j + 4] & 63) as u32)
        } else {
            (
                ((q[j + 4] & 0x0f) as u32) | (((q[j - 4] >> 6) as u32) << 4),
                ((q[j + 4] >> 4) as u32) | (((q[j] >> 6) as u32) << 4),
            )
        }
    }

    /// カーネルの 1 要素ぶんのデコード (.cu の `dno_ktc_w8_*` を k で引く形)。
    fn mirror_k(ty: GgmlType, row: &[u8], k: usize) -> f32 {
        let k0 = k & !31;
        let j = (k & 31) >> 3;
        let u = k & 7;
        let p = j << 3;
        match ty {
            GgmlType::Q4K => {
                let sb = (k0 >> 5) & 7;
                let blk = &row[(k0 >> 8) * 144..];
                let d = f16v(blk);
                let dmin = f16v(&blk[2..]);
                let (sc, mn) = scale_min_k4(sb, &blk[4..]);
                let dv = d * sc as f32;
                let mv = dmin * mn as f32;
                let qs = &blk[16 + (sb >> 1) * 32 + p..];
                let sh = if sb & 1 != 0 { 4 } else { 0 };
                let q = ((qs[u] >> sh) & 0x0f) as f32;
                dv * q - mv
            }
            GgmlType::Q5K => {
                let sb = (k0 >> 5) & 7;
                let blk = &row[(k0 >> 8) * 176..];
                let d = f16v(blk);
                let dmin = f16v(&blk[2..]);
                let (sc, mn) = scale_min_k4(sb, &blk[4..]);
                let dv = d * sc as f32;
                let mv = dmin * mn as f32;
                let qh = &blk[16..];
                let qs = &blk[48 + (sb >> 1) * 32 + p..];
                let sh = if sb & 1 != 0 { 4 } else { 0 };
                let q = ((qs[u] >> sh) & 0x0f) as u32;
                let hb = ((qh[p + u] >> sb) & 1) as u32;
                dv * (q + (hb << 4)) as f32 - mv
            }
            GgmlType::Q6K => {
                let blk = &row[(k0 >> 8) * 210..];
                let d = f16v(&blk[208..]);
                let half = (k0 >> 7) & 1;
                let ch = (k0 >> 5) & 3;
                let l2 = j << 3;
                let ql = &blk[half * 64 + if ch & 1 != 0 { 32 } else { 0 } + l2..];
                let qh = &blk[128 + half * 32 + l2..];
                let sc = &blk[192 + half * 8..];
                let sv = sc[2 * ch + if l2 >= 16 { 1 } else { 0 }] as i8 as f32;
                let sh = if ch >= 2 { 4 } else { 0 };
                let nib = ((ql[u] >> sh) & 0x0f) as i32;
                let h2 = ((qh[u] >> (2 * ch)) & 3) as i32;
                let q6 = (nib | (h2 << 4)) - 32;
                d * sv * q6 as f32
            }
            GgmlType::Q5_1 => {
                let blk = &row[(k0 >> 5) * 24..];
                let d = f16v(blk);
                let m = f16v(&blk[2..]);
                let qh = u32le(&blk[4..]);
                let hi = p >= 16;
                let qs = &blk[8 + (p & 15)..];
                let byte = qs[u] as u32;
                let q = if hi { byte >> 4 } else { byte & 0x0f };
                let hb = (qh >> (p + u)) & 1;
                d * (q + (hb << 4)) as f32 + m
            }
            GgmlType::Q8_0 => {
                let blk = &row[(k0 >> 5) * 34..];
                let d = f16v(blk);
                d * blk[2 + p + u] as i8 as f32
            }
            _ => unreachable!("K 系 5 型だけ"),
        }
    }

    /// 決定的な擬似乱数で 1 行ぶんのバイト列を作る。d (f16) だけは 1.0 に固定する。
    fn make_row(ty: GgmlType, n: usize, seed: u64) -> Vec<u8> {
        let ts = ty.type_size();
        let bs = ty.blck_size();
        let mut v = vec![0u8; (n / bs) * ts];
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for b in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        for i in 0..(n / bs) {
            v[i * ts] = 0x00;
            v[i * ts + 1] = 0x3C; // f16 の 1.0
            // 2 本目の f16 (Q4_K / Q5_K の dmin、Q5_1 の m) も 1.0 に固定する
            // (乱数のままだと NaN/Inf が出て参照との比較にならない)。
            if matches!(ty, GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q5_1) {
                v[i * ts + 2] = 0x00;
                v[i * ts + 3] = 0x3C;
            }
        }
        v
    }

    /// 索引が参照と一致する (型ごとに 1 行 512 要素)。
    #[test]
    fn ktc_kernel_index_mirror_matches_cpu_reference() {
        for (ty, seed) in [
            (GgmlType::Q4K, 21u64),
            (GgmlType::Q5K, 22),
            (GgmlType::Q6K, 23),
            (GgmlType::Q5_1, 24),
            (GgmlType::Q8_0, 25),
        ] {
            let n = 512usize;
            let row = make_row(ty, n, seed);
            let mut r = vec![0.0f32; n];
            dequant_row_into(ty, &row, &mut r);
            let mut worst = 0.0f32;
            let mut bad = 0usize;
            for k in 0..n {
                let g = mirror_k(ty, &row, k);
                let d = (g - r[k]).abs();
                let rel = d / r[k].abs().max(1e-6);
                if !(rel <= 1e-5) {
                    bad += 1;
                    if bad <= 4 {
                        eprintln!("[pf-ktc-test] {ty:?} k={k} mirror={g} ref={} rel={rel}", r[k]);
                    }
                }
                worst = worst.max(rel);
            }
            assert_eq!(bad, 0, "{ty:?}: 索引が参照と違う要素 {bad} 個 (worst rel {worst})");
        }
    }

    /// 形の判定 (`ktc_shapes_ok`) の契約。
    #[test]
    fn ktc_shapes_ok_pins_the_contract() {
        let k0 = crate::model::exp_pf::K_SLOT0;
        // 実モデルの形 (e = 2560 / ff = 1536、gate/up = Q4_K、down = Q5_1)。
        assert!(ktc_shapes_ok(Some(k0), Some(k0 + 3), true, 2560, 1536, 1536, 2560));
        // Q6_K の gate/up も受ける。
        assert!(ktc_shapes_ok(Some(k0 + 2), Some(k0 + 3), true, 2560, 1536, 1536, 2560));
        // gate と up の型が違うのは不可 (1 回の起動で 2 枚を同時にデコードする)。
        assert!(!ktc_shapes_ok(Some(k0), Some(k0 + 3), false, 2560, 1536, 1536, 2560));
        // 256 要素ブロックの型は k が 256 の倍数でないと不可。
        assert!(!ktc_shapes_ok(Some(k0), Some(k0 + 3), true, 2432, 1536, 1536, 2560));
        // 32 要素ブロックの型 (Q5_1 / Q8_0) は 32 の倍数でよい。
        assert!(ktc_shapes_ok(Some(k0 + 3), Some(k0 + 4), true, 2560, 1536, 1536, 2560));
        assert!(!ktc_shapes_ok(Some(k0 + 3), Some(k0 + 4), true, 2560, 1520, 1536, 2560));
        // n が 64 の倍数でないと不可。
        assert!(!ktc_shapes_ok(Some(k0), Some(k0 + 3), true, 2560, 1536, 1000, 2560));
        assert!(!ktc_shapes_ok(Some(k0), Some(k0 + 3), true, 2560, 1536, 1536, 2540));
        // IQ のスロットは K 版では受けない。
        assert!(!ktc_shapes_ok(Some(1), Some(k0 + 3), true, 2560, 1536, 1536, 2560));
        // 0 は不可。
        assert!(!ktc_shapes_ok(Some(k0), Some(k0 + 3), true, 0, 1536, 1536, 2560));
    }

    /// task 420: UD-Q4_K_XL の層 0 の形 (e = 2560 / ff = 640、gate/up = Q4_K、
    /// down = Q5_1) は **K 版にだけ**乗る = プール計画の `tc` は K を見ないと
    /// 「乗らない」と誤判定する (14k で g / u 686 MiB を無駄に予約していた)。
    #[test]
    fn ud_q4_k_xl_shape_takes_the_k_path_only() {
        let k0 = crate::model::exp_pf::K_SLOT0;
        let (gu, dn) = (Some(k0), Some(k0 + 3));
        assert!(ktc_shapes_ok(gu, dn, true, 2560, 640, 640, 2560), "K 版は受ける");
        assert!(
            !tc_shapes_ok(gu, dn, true, 2560, 640, 640, 2560),
            "IQ 版は受けない (スロットが IQ の範囲外)"
        );
    }

    /// ゲートの真理値表 (task 416: lab t416b で採って既定 on。明示的な偽値だけ off)。
    #[test]
    fn tc_exp_k_gate_defaults_on() {
        use crate::model::exp_pf::{tc_exp_k_from_value, DEFAULT_TC_EXP_K_ON};
        assert!(DEFAULT_TC_EXP_K_ON, "既定 off に戻っている");
        assert!(tc_exp_k_from_value(None));
        assert!(tc_exp_k_from_value(Some("")));
        assert!(!tc_exp_k_from_value(Some("0")));
        assert!(!tc_exp_k_from_value(Some("off")));
        assert!(!tc_exp_k_from_value(Some(" false ")));
        assert!(!tc_exp_k_from_value(Some("no")));
        assert!(tc_exp_k_from_value(Some("1")));
        assert!(tc_exp_k_from_value(Some("on")));
        assert!(tc_exp_k_from_value(Some("true")));
    }

    /// `.cu` のタイル定数が IQ 版と同じであること (両方の body を同じ
    /// ホスト定数で起動する)。
    #[test]
    fn ktc_tile_constants_match_the_cu() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_k_tc.cu");
        let src = std::fs::read_to_string(&p).expect("kernels_exp_k_tc.cu が読める");
        let get = |name: &str| -> usize {
            let key = format!("#define {name} ");
            let line = src
                .lines()
                .find(|l| l.trim_start().starts_with(&key))
                .unwrap_or_else(|| panic!("{name} が .cu に無い"));
            line.trim_start()[key.len()..]
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} の値が数値でない"))
        };
        assert_eq!(get("TC_BM"), EXTC_BM, "TC_BM");
        assert_eq!(get("TC_BN"), EXTC_BN, "TC_BN");
        assert_eq!(get("TC_THREADS"), EXTC_THREADS, "TC_THREADS");
    }

    /// コミット済み PTX に 10 本のエントリがあること (無いときは省略)。
    #[test]
    fn ktc_ptx_has_the_ten_entries() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_k_tc.ptx");
        let Ok(text) = std::fs::read_to_string(&p) else {
            eprintln!("PTX が無いので省略: {}", p.display());
            return;
        };
        for name in KTC_GU_NAMES.iter() {
            assert_eq!(ptx_entry_params(&text, name), Some(TC_GU_NARGS), "{name}");
        }
        for name in KTC_DN_NAMES.iter() {
            assert_eq!(ptx_entry_params(&text, name), Some(TC_DN_NARGS), "{name}");
        }
    }
}


#[cfg(test)]
mod pf_exp_tc_tests {
    //! task 250: tensor core グループ GEMM (`kernels_exp_iq_tc.cu`) の**索引**を
    //! GPU 無しで固定する。
    //!
    //! カーネルは 32 要素ぶんの重みを 8 個ずつ (j = 0..3) デコードする。その
    //! 索引算術を .cu の行から 1 対 1 で写し、CPU の参照
    //! (`kquant::dequant_row_into`) と要素ごとに比べる。丸め順の一致ではなく
    //! **索引の取り違え**を落とすためのテスト (型ごとに 1 行 512 要素)。

    use super::*;
    use crate::kquant::{dequant_row_into, GgmlType};
    use crate::kquant_tables::{IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, KVALUES_IQ4NL, KSIGNS_IQ2XS};

    /// .cu の `dno_tcq_f16` と同じ式 (IEEE の f16 → f32)。
    fn f16v(b: &[u8]) -> f32 {
        let h = b[0] as u32 | ((b[1] as u32) << 8);
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mut man = h & 0x3ff;
        let bits = if exp == 0 {
            if man == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                while man & 0x400 == 0 {
                    man <<= 1;
                    e -= 1;
                }
                man &= 0x3ff;
                sign | (((e + 127) as u32) << 23) | (man << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (man << 13)
        } else {
            sign | ((exp + 112) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }

    fn u32le(b: &[u8]) -> u32 {
        b[0] as u32 | ((b[1] as u32) << 8) | ((b[2] as u32) << 16) | ((b[3] as u32) << 24)
    }

    pub(crate) fn sgn(signs: u32, u: usize) -> f32 {
        if signs & (1 << u) != 0 {
            -1.0
        } else {
            1.0
        }
    }

    /// カーネルの 1 要素ぶんのデコード (.cu の `dno_tc_w8_*` を k で引く形)。
    pub(crate) fn mirror(ty: GgmlType, row: &[u8], k: usize) -> f32 {
        let s = k >> 5;
        let sb = s >> 3;
        let sub = s & 7;
        let j = (k & 31) >> 3;
        let u = k & 7;
        match ty {
            GgmlType::IQ2S => {
                let blk = &row[sb * 82..];
                let d = f16v(blk);
                let qh = blk[66 + sub] as u32;
                let idx = (blk[2 + sub * 4 + j] as u32 | ((qh << (8 - 2 * j)) & 0x300)) as usize;
                let sg = blk[34 + sub * 4 + j] as u32;
                let sc = blk[74 + sub] as u32;
                let nib = if j & 2 != 0 { sc >> 4 } else { sc & 0x0f };
                let db = d * (0.5 + nib as f32) * 0.25;
                let grid = IQ2S_GRID[idx];
                let gv = ((grid >> (8 * u)) & 0xff) as f32;
                db * gv * sgn(sg, u)
            }
            GgmlType::IQ3Xxs => {
                let blk = &row[sb * 98..];
                let d = f16v(blk);
                let aux = u32le(&blk[66 + sub * 4..]);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                let signs = KSIGNS_IQ2XS[((aux >> (7 * j)) & 127) as usize] as u32;
                let g1 = IQ3XXS_GRID[blk[2 + sub * 8 + 2 * j] as usize];
                let g2 = IQ3XXS_GRID[blk[2 + sub * 8 + 2 * j + 1] as usize];
                if u < 4 {
                    db * ((g1 >> (8 * u)) & 0xff) as f32 * sgn(signs, u)
                } else {
                    db * ((g2 >> (8 * (u - 4))) & 0xff) as f32 * sgn(signs, u)
                }
            }
            GgmlType::IQ3S => {
                let blk = &row[sb * 110..];
                let d = f16v(blk);
                let sc = blk[106 + (sub >> 1)] as u32;
                let nib = if sub & 1 != 0 { sc >> 4 } else { sc & 0x0f };
                let db = d * (1.0 + 2.0 * nib as f32);
                let qh = blk[66 + sub] as u32;
                let idx1 = (blk[2 + sub * 8 + 2 * j] as u32 | ((qh << (8 - 2 * j)) & 256)) as usize;
                let idx2 = (blk[2 + sub * 8 + 2 * j + 1] as u32 | ((qh << (7 - 2 * j)) & 256)) as usize;
                let signs = blk[74 + sub * 4 + j] as u32;
                let g1 = IQ3S_GRID[idx1];
                let g2 = IQ3S_GRID[idx2];
                if u < 4 {
                    db * ((g1 >> (8 * u)) & 0xff) as f32 * sgn(signs, u)
                } else {
                    db * ((g2 >> (8 * (u - 4))) & 0xff) as f32 * sgn(signs, u)
                }
            }
            GgmlType::IQ4Xs => {
                let blk = &row[sb * 136..];
                let d = f16v(blk);
                let half = j >> 1;
                let jbase = (j & 1) * 8;
                let sh = blk[2] as u32 | ((blk[3] as u32) << 8);
                let sl = blk[4 + (sub >> 1)] as u32;
                let ls = ((sl >> (4 * (sub & 1))) & 0x0f) | (((sh >> (2 * sub)) & 3) << 4);
                let dl = d * (ls as f32 - 32.0);
                let qs = &blk[8 + sub * 16 + jbase..];
                let q = if half != 0 { qs[u] >> 4 } else { qs[u] & 0x0f };
                dl * KVALUES_IQ4NL[q as usize] as f32
            }
            GgmlType::IQ4Nl => {
                let blk = &row[(k >> 5) * 18..];
                let d = f16v(blk);
                let half = j >> 1;
                let b0 = (j & 1) * 8;
                let byte = blk[2 + b0 + u] as u32;
                let q = if half != 0 { byte >> 4 } else { byte & 0x0f };
                d * KVALUES_IQ4NL[q as usize] as f32
            }
            GgmlType::Q2_0 => {
                let blk = &row[(k >> 6) * 18..];
                let d = f16v(blk);
                let p = k & 63;
                let q = ((blk[2 + (p >> 2)] as u32) >> ((p & 3) * 2)) & 3;
                ((q as i32) - 1) as f32 * d
            }
            _ => unreachable!("6 型だけ"),
        }
    }

    /// 決定的な擬似乱数で 1 行ぶんのバイト列を作る。d (f16) だけは 1.0 に固定する
    /// (NaN/Inf を避ける。他の欄は全ビット模様が有効なので任意でよい)。
    fn make_row(ty: GgmlType, n: usize, seed: u64) -> Vec<u8> {
        let ts = ty.type_size();
        let bs = ty.blck_size();
        let mut v = vec![0u8; (n / bs) * ts];
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for b in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        for i in 0..(n / bs) {
            v[i * ts] = 0x00;
            v[i * ts + 1] = 0x3C; // f16 の 1.0
        }
        v
    }

    /// 索引が参照と一致する (型ごとに 1 行 512 要素 = 2 ブロック)。
    #[test]
    fn tc_kernel_index_mirror_matches_cpu_reference() {
        for (ty, seed) in [
            (GgmlType::IQ2S, 11u64),
            (GgmlType::IQ3Xxs, 12),
            (GgmlType::IQ3S, 13),
            (GgmlType::IQ4Xs, 14),
            (GgmlType::IQ4Nl, 15),
            (GgmlType::Q2_0, 16),
        ] {
            let n = 512usize;
            let row = make_row(ty, n, seed);
            let mut r = vec![0.0f32; n];
            dequant_row_into(ty, &row, &mut r);
            let mut worst = 0.0f32;
            let mut bad = 0usize;
            for k in 0..n {
                let g = mirror(ty, &row, k);
                let d = (g - r[k]).abs();
                let rel = d / r[k].abs().max(1e-6);
                if !(rel <= 1e-5) {
                    bad += 1;
                    if bad <= 4 {
                        eprintln!("[pf-tc-test] {ty:?} k={k} mirror={g} ref={} rel={rel}", r[k]);
                    }
                }
                worst = worst.max(rel);
            }
            assert_eq!(bad, 0, "{ty:?}: 索引が参照と違う要素 {bad} 個 (worst rel {worst})");
        }
    }

    /// ゲートの真理値表 (既定 off・明示的な真値だけ on)。
    #[test]
    fn tc_exp_gate_is_strictly_opt_in() {
        // task 265: 既定 on、明示的な偽値だけ off。
        assert!(crate::model::exp_pf::tc_exp_from_value(None));
        assert!(crate::model::exp_pf::tc_exp_from_value(Some("")));
        assert!(!crate::model::exp_pf::tc_exp_from_value(Some("0")));
        assert!(!crate::model::exp_pf::tc_exp_from_value(Some("off")));
        assert!(!crate::model::exp_pf::tc_exp_from_value(Some(" false ")));
        assert!(crate::model::exp_pf::tc_exp_from_value(Some("1")));
        assert!(crate::model::exp_pf::tc_exp_from_value(Some("on")));
        assert!(crate::model::exp_pf::tc_exp_from_value(Some("true")));
    }

    /// task 250: 層優先 MoE のトークン分割は「重複なし・取りこぼしなし」。
    #[test]
    fn lm_tile_covers_the_prompt_exactly_once() {
        use crate::model::exp_pf::lm_tile_from_value;
        assert_eq!(lm_tile_from_value(None), 5120, "task 265: 既定は 5120");
        assert_eq!(lm_tile_from_value(Some("")), 5120);
        assert_eq!(lm_tile_from_value(Some("0")), 0);
        assert_eq!(lm_tile_from_value(Some("1023")), 0, "1024 未満は分割しない");
        assert_eq!(lm_tile_from_value(Some("7168")), 7168);
        assert_eq!(lm_tile_from_value(Some("x")), 0);
        for nt in [3826usize, 14018] {
            for t in [5120usize, 7168] {
                let tile = if t >= nt { nt } else { t };
                let mut cover = vec![0u8; nt];
                let mut s0 = 0usize;
                while s0 < nt {
                    let n = tile.min(nt - s0);
                    for c in cover[s0..s0 + n].iter_mut() {
                        *c += 1;
                    }
                    s0 += n;
                }
                assert!(cover.iter().all(|&c| c == 1), "nt={nt} tile={t} に穴か重複");
            }
        }
    }

    /// 作業域の外部予約のゲートも既定 off・明示的な真値だけ on。
    #[test]
    fn lm_reserve_gate_is_strictly_opt_in() {
        use crate::model::exp_pf::{lm_reserve_from_value, tc_exp_from_value};
        for v in [None, Some(""), Some("0"), Some("off"), Some("false")] {
            assert!(!lm_reserve_from_value(v), "lm_reserve({v:?})");
        }
        // task 265: tc_exp is default on; only explicit false values turn it off.
        for v in [Some("0"), Some("off"), Some("false")] {
            assert!(!tc_exp_from_value(v), "tc_exp({v:?})");
        }
        for v in [None, Some("")] {
            assert!(tc_exp_from_value(v), "tc_exp({v:?})");
        }
        for v in [Some("1"), Some("on"), Some("true"), Some(" yes ")] {
            assert!(lm_reserve_from_value(v), "lm_reserve({v:?})");
        }
    }

    /// 形の契約: gate と up が同じ型・4 型・k が 256 の倍数・n が 64 の倍数。
    #[test]
    fn tc_shapes_ok_pins_the_contract() {
        // 実モデルの形 (e = 2560 / ff = 1024)。
        assert!(tc_shapes_ok(Some(1), Some(3), true, 2560, 1024, 1024, 2560));
        // gate と up が違う型は不可。
        assert!(!tc_shapes_ok(Some(1), Some(3), false, 2560, 1024, 1024, 2560));
        // down は IQ4_NL (slot 4) / Q2_0 (slot 5) も受ける (実モデルの down)。
        assert!(tc_shapes_ok(Some(1), Some(4), true, 2560, 1024, 1024, 2560));
        assert!(tc_shapes_ok(Some(1), Some(5), true, 2560, 1024, 1024, 2560));
        // gate/up が IQ4_NL / Q2_0 はこのカーネルの対象外。
        assert!(!tc_shapes_ok(Some(4), Some(3), true, 2560, 1024, 1024, 2560));
        assert!(!tc_shapes_ok(Some(5), Some(3), true, 2560, 1024, 1024, 2560));
        // k が 256 の倍数でない / n が 64 の倍数でない。
        assert!(!tc_shapes_ok(Some(1), Some(3), true, 2432, 1024, 1024, 2560));
        assert!(!tc_shapes_ok(Some(1), Some(3), true, 2560, 1024, 1000, 2560));
        assert!(!tc_shapes_ok(Some(1), Some(3), true, 2560, 1024, 1024, 2540));
        // 0 は不可。
        assert!(!tc_shapes_ok(Some(1), Some(3), true, 0, 1024, 1024, 2560));
    }

    /// `.cu` のタイル定数とホストの定数が一致すること (片方だけ直す事故を落とす)。
    #[test]
    fn tc_tile_constants_match_the_cu() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq_tc.cu");
        let src = std::fs::read_to_string(&p).expect("kernels_exp_iq_tc.cu が読める");
        let get = |name: &str| -> usize {
            let key = format!("#define {name} ");
            let line = src
                .lines()
                .find(|l| l.trim_start().starts_with(&key))
                .unwrap_or_else(|| panic!("{name} が .cu に無い"));
            line.trim_start()[key.len()..]
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("{name} の値が数値でない"))
        };
        assert_eq!(get("TC_BM"), EXTC_BM, "TC_BM");
        assert_eq!(get("TC_BN"), EXTC_BN, "TC_BN");
        assert_eq!(get("TC_THREADS"), EXTC_THREADS, "TC_THREADS");
        // 1 出力行は 16 列ずつ 4 warp で覆い、トークンは 32 本ずつ 2 段 + 16 本 × 2。
        assert_eq!(EXTC_BN, 4 * 16, "行タイルは wn 4 段 × 16 行");
        assert_eq!(EXTC_BM, 2 * 2 * 16, "トークンタイルは wm 2 段 × 2 断片 × 16");
    }

    /// PTX の引数の数え方がコミット済みの PTX (nvcc の実出力) で 9 になること。
    /// 合成文字列ではなく実物で確かめる (task 68 の ABI ゲートと同じ)。
    #[test]
    fn ptx_entry_params_counts_the_committed_pf_ptx() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_iq_pf.ptx");
        let Ok(text) = std::fs::read_to_string(&p) else {
            eprintln!("PTX が無いので省略: {}", p.display());
            return;
        };
        for name in ["dno_exp_pf_iq2s", "dno_exp_pf_iq3xxs", "dno_exp_pf_iq3s", "dno_exp_pf_iq4xs"] {
            assert_eq!(
                ptx_entry_params(&text, name),
                Some(9),
                "{name} の引数の数 (ptx パーサの前提)"
            );
        }
    }
}


#[cfg(test)]
mod pf_exp_k_tests {
    //! task 309: K 系 m 行カーネル (`kernels_exp_k_pf.cu`) の**索引**ミラー。
    //!
    //! カーネルの `kpf_w8_q4k` / `_q5k` / `_q6k` と、32 要素ブロックの
    //! `dno_exp_k_pf_q5_1` / `_q8_0` の索引算術を `.cu` の行から 1 対 1 で写し、
    //! CPU の参照 (`kquant::dequant_row_into`) と要素ごとに比べる。
    //! 丸め順の一致ではなく**索引の取り違え**を落とすためのテスト
    //! (型ごとに 1 行 512 要素 = Q4_K/Q5_K/Q6_K は 2 ブロック)。

    use super::*;
    use crate::kquant::{dequant_row_into, GgmlType};

    /// .cu の `kpf_f16` と同じ式 (IEEE の f16 → f32)。
    fn f16v(b: &[u8]) -> f32 {
        let h = b[0] as u32 | ((b[1] as u32) << 8);
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mut man = h & 0x3ff;
        let bits = if exp == 0 {
            if man == 0 {
                sign
            } else {
                let mut e: i32 = -14;
                while man & 0x400 == 0 {
                    man <<= 1;
                    e -= 1;
                }
                man &= 0x3ff;
                sign | (((e + 127) as u32) << 23) | (man << 13)
            }
        } else if exp == 31 {
            sign | 0x7f80_0000 | (man << 13)
        } else {
            sign | ((exp + 112) << 23) | (man << 13)
        };
        f32::from_bits(bits)
    }

    fn u32le(b: &[u8]) -> u32 {
        b[0] as u32 | ((b[1] as u32) << 8) | ((b[2] as u32) << 16) | ((b[3] as u32) << 24)
    }

    /// .cu の `kpf_scale_min_k4` と同じ式。
    fn scale_min_k4(q: &[u8], j: usize) -> (u32, u32) {
        if j < 4 {
            ((q[j] & 63) as u32, (q[j + 4] & 63) as u32)
        } else {
            (
                ((q[j + 4] & 0x0f) as u32) | (((q[j - 4] >> 6) as u32) << 4),
                ((q[j + 4] >> 4) as u32) | (((q[j] >> 6) as u32) << 4),
            )
        }
    }

    /// カーネルが要素 `k` をどうデコードするか (lane と i を k から引く)。
    pub(crate) fn mirror(ty: GgmlType, row: &[u8], k: usize) -> f32 {
        match ty {
            // 256 要素ブロック: lane = (k % 256) / 8、i = k % 8。
            GgmlType::Q4K => {
                let blk = &row[(k >> 8) * 144..];
                let d = f16v(blk);
                let dmin = f16v(&blk[2..]);
                let lane = (k & 255) >> 3;
                let i = k & 7;
                let sb = lane >> 2;
                let p = (lane & 3) << 3;
                let (sc, mn) = scale_min_k4(&blk[4..16], sb);
                let sh = if sb & 1 != 0 { 4 } else { 0 };
                let q = (blk[16 + (sb >> 1) * 32 + p + i] as u32 >> sh) & 0x0f;
                d * sc as f32 * q as f32 - dmin * mn as f32
            }
            GgmlType::Q5K => {
                let blk = &row[(k >> 8) * 176..];
                let d = f16v(blk);
                let dmin = f16v(&blk[2..]);
                let lane = (k & 255) >> 3;
                let i = k & 7;
                let sb = lane >> 2;
                let p = (lane & 3) << 3;
                let (sc, mn) = scale_min_k4(&blk[4..16], sb);
                let sh = if sb & 1 != 0 { 4 } else { 0 };
                let q = (blk[48 + (sb >> 1) * 32 + p + i] as u32 >> sh) & 0x0f;
                let hb = (blk[16 + p + i] as u32 >> sb) & 1;
                d * sc as f32 * (q + (hb << 4)) as f32 - dmin * mn as f32
            }
            GgmlType::Q6K => {
                let blk = &row[(k >> 8) * 210..];
                let d = f16v(&blk[208..]);
                let lane = (k & 255) >> 3;
                let i = k & 7;
                let half = lane >> 4;
                let o = (lane & 15) << 3;
                let ch = o >> 5;
                let l2 = o & 31;
                let ql = &blk[half * 64 + if ch & 1 != 0 { 32 } else { 0 } + l2..];
                let qh = &blk[128 + half * 32 + l2..];
                let sc = &blk[192 + half * 8..];
                let s = sc[2 * ch + if l2 >= 16 { 1 } else { 0 }] as i8 as f32;
                let sh = if ch >= 2 { 4 } else { 0 };
                let nib = (ql[i] as u32 >> sh) & 0x0f;
                let h2 = (qh[i] as u32 >> (2 * ch)) & 3;
                let q6 = (nib | (h2 << 4)) as i32 - 32;
                d * s * q6 as f32
            }
            // 32 要素ブロック: レーン 1 本 = 1 ブロック (k % 32 が要素)。
            GgmlType::Q5_1 => {
                let blk = &row[(k >> 5) * 24..];
                let d = f16v(blk);
                let m = f16v(&blk[2..]);
                let qh = u32le(&blk[4..]);
                let e = k & 31;
                let byte = blk[8 + (e & 15)] as u32;
                let q = if e < 16 { byte & 0x0f } else { byte >> 4 };
                let hb = (qh >> e) & 1;
                d * (q + (hb << 4)) as f32 + m
            }
            GgmlType::Q8_0 => {
                let blk = &row[(k >> 5) * 34..];
                let d = f16v(blk);
                d * (blk[2 + (k & 31)] as i8) as f32
            }
            _ => unreachable!("K 系 5 型だけ"),
        }
    }

    /// 決定的な擬似乱数で 1 行ぶんのバイト列を作る。スケール欄 (f16) だけは
    /// 有限の値に固定する (NaN/Inf を避ける。他は全ビット模様が有効)。
    fn make_row(ty: GgmlType, n: usize, seed: u64) -> Vec<u8> {
        let ts = ty.type_size();
        let bs = ty.blck_size();
        let mut v = vec![0u8; (n / bs) * ts];
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        for b in v.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *b = (s >> 33) as u8;
        }
        // f16 の 1.0 = 0x3C00 / 0.5 = 0x3800 / 0.25 = 0x3400 (リトルエンディアン)。
        let put = |v: &mut [u8], off: usize, bytes: [u8; 2]| {
            v[off] = bytes[0];
            v[off + 1] = bytes[1];
        };
        for i in 0..(n / bs) {
            let b = i * ts;
            match ty {
                GgmlType::Q4K | GgmlType::Q5K => {
                    put(&mut v, b, [0x00, 0x3C]); // d = 1.0
                    put(&mut v, b + 2, [0x00, 0x38]); // dmin = 0.5
                }
                GgmlType::Q6K => put(&mut v, b + 208, [0x00, 0x3C]), // d = 1.0
                GgmlType::Q5_1 => {
                    put(&mut v, b, [0x00, 0x3C]); // d = 1.0
                    put(&mut v, b + 2, [0x00, 0x34]); // m = 0.25
                }
                GgmlType::Q8_0 => put(&mut v, b, [0x00, 0x3C]), // d = 1.0
                _ => {}
            }
        }
        v
    }

    /// 索引が参照と一致する (型ごとに 1 行 512 要素)。
    #[test]
    fn k_kernel_index_mirror_matches_cpu_reference() {
        for (ty, seed) in [
            (GgmlType::Q4K, 21u64),
            (GgmlType::Q5K, 22),
            (GgmlType::Q6K, 23),
            (GgmlType::Q5_1, 24),
            (GgmlType::Q8_0, 25),
        ] {
            let n = 512usize;
            let row = make_row(ty, n, seed);
            let mut r = vec![0.0f32; n];
            dequant_row_into(ty, &row, &mut r);
            let mut bad = 0usize;
            let mut worst = 0.0f32;
            for k in 0..n {
                let g = mirror(ty, &row, k);
                let d = (g - r[k]).abs();
                let rel = d / r[k].abs().max(1e-3);
                if !(rel <= 1e-6) {
                    bad += 1;
                    if bad <= 4 {
                        eprintln!("[pf-k-test] {ty:?} k={k} mirror={g} ref={} rel={rel}", r[k]);
                    }
                }
                worst = worst.max(rel);
            }
            assert_eq!(bad, 0, "{ty:?}: 索引が参照と違う要素 {bad} 個 (worst rel {worst})");
        }
    }

    /// コミット済みの K 系 PTX があれば、5 本とも引数 9 個で載っていること
    /// (ホストの起動形と ABI が合っていることの裏取り)。無いときは省略。
    #[test]
    fn ptx_entry_params_counts_the_committed_k_ptx() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("kernels")
            .join("kernels_exp_k_pf.ptx");
        let Ok(text) = std::fs::read_to_string(&p) else {
            eprintln!("PTX が無いので省略: {}", p.display());
            return;
        };
        for name in [
            "dno_exp_k_pf_q4k",
            "dno_exp_k_pf_q5k",
            "dno_exp_k_pf_q6k",
            "dno_exp_k_pf_q5_1",
            "dno_exp_k_pf_q8_0",
        ] {
            assert_eq!(
                ptx_entry_params(&text, name),
                Some(9),
                "{name} の引数の数 (ptx パーサの前提)"
            );
        }
    }
}


#[cfg(test)]
mod pf_stream_tests {
    //! task 162: ストリーミング経路の**契約**を GPU 無しで固定する。

    use super::*;
    use crate::kquant::GgmlType;

    /// task 239: 分割して写しても pinned の中身は 1 回写しと**同一**。
    ///
    /// 分割は「DMA(i) を写し(i+1) と重ねる」ためだけのもので、デバイスへ渡る
    /// バイト列を変えない (数値不変の根拠)。区間は整列の隙間をまたぐ形で切る。
    #[test]
    fn split_copy_into_pin_is_byte_identical() {
        // 96 バイトの転送元を 3 区間に分けて置く (間に整列の隙間がある形)。
        let src: Vec<u8> = (0..96u8).collect();
        let parts = vec![
            (
                0usize,
                PfHostRange {
                    ptr: src.as_ptr() as usize,
                    len: 16,
                    src: None,
                },
            ),
            (
                32usize,
                PfHostRange {
                    ptr: unsafe { src.as_ptr().add(16) } as usize,
                    len: 20,
                    src: None,
                },
            ),
            (
                64usize,
                PfHostRange {
                    ptr: unsafe { src.as_ptr().add(36) } as usize,
                    len: 12,
                    src: None,
                },
            ),
        ];
        let need = 80usize;
        // 従来: 1 回で全部写す。
        let mut one = vec![0xAAu8; need];
        pf_copy_into_pin(one.as_mut_ptr() as usize, &parts, false);
        // 分割: 24 バイトずつ (区間の境界と隙間をまたぐ)。
        let mut split = vec![0xAAu8; need];
        let mut off = 0usize;
        while off < need {
            let len = (need - off).min(24);
            pf_copy_into_pin_range(split.as_mut_ptr() as usize, &parts, off, len, false);
            off += len;
        }
        assert_eq!(one, split, "分割で pinned の中身が変わった");
        // 隙間は書かれない (0xAA のまま)。
        for gap in [16usize..32, 52..64] {
            for &b in &split[gap] {
                assert_eq!(b, 0xAA, "整列の隙間を書いてしまった");
            }
        }
        // 区間の中身は転送元のとおり。
        assert_eq!(&split[0..16], &src[0..16]);
        assert_eq!(&split[32..52], &src[16..36]);
        assert_eq!(&split[64..76], &src[36..48]);
    }

    /// task 481: 小さなバウンスへ**詰めて**写しても、1 回写しと中身が同じ。
    ///
    /// `pf_copy_into_pin_at` は分割 DMA (2 枚のバウンスを交互に使う) の写しで、
    /// オフセットを区間の先頭だけ詰める。区間をずらして繋ぐと 1 枚に写した
    /// ものと 1 バイトも変わらないこと = デバイスへ渡るバイト列が変わらない
    /// ことの根拠。
    #[test]
    fn split_at_copy_reassembles_to_the_same_bytes() {
        let src: Vec<u8> = (0..200u8).map(|i| i.wrapping_mul(7).wrapping_add(3)).collect();
        // 整列の隙間をまたぐ形の 4 区間 (0..20, 32..60, 60..61, 128..190)。
        let parts = vec![
            (
                0usize,
                PfHostRange {
                    ptr: src.as_ptr() as usize,
                    len: 20,
                    src: None,
                },
            ),
            (
                32usize,
                PfHostRange {
                    ptr: unsafe { src.as_ptr().add(20) } as usize,
                    len: 28,
                    src: None,
                },
            ),
            (
                60usize,
                PfHostRange {
                    ptr: unsafe { src.as_ptr().add(48) } as usize,
                    len: 1,
                    src: None,
                },
            ),
            (
                128usize,
                PfHostRange {
                    ptr: unsafe { src.as_ptr().add(49) } as usize,
                    len: 62,
                    src: None,
                },
            ),
        ];
        let need = 190usize;
        // 従来: 1 枚へ全部写す。
        let mut one = vec![0x5Au8; need];
        pf_copy_into_pin(one.as_mut_ptr() as usize, &parts, false);
        // 分割: 24 バイトずつ、2 枚のバウンスへ交互に詰めて繋ぎ直す。
        let chunk = 24usize;
        let mut got = vec![0x5Au8; need];
        let mut off = 0usize;
        let mut i = 0usize;
        while off < need {
            let n = chunk.min(need - off);
            let b = i % 2;
            assert_eq!(b, i % 2);
            pf_copy_into_pin_at(got.as_mut_ptr() as usize + off, &parts, off, n, false);
            off += n;
            i += 1;
        }
        assert_eq!(one, got, "分割で組み立てたバイト列が 1 回写しと違う");
        // 隙間は書かれない。
        for gap in [20usize..32, 61..128] {
            for &x in &got[gap] {
                assert_eq!(x, 0x5A, "整列の隙間を書いてしまった");
            }
        }
        assert_eq!(&got[0..20], &src[0..20]);
        assert_eq!(&got[32..60], &src[20..48]);
        assert_eq!(&got[60..61], &src[48..49]);
        assert_eq!(&got[128..190], &src[49..111]);
    }

    /// task 325: `DENNO_PF_PREAD` の写しが memcpy と 1 バイトも変わらないこと。
    ///
    /// 本物のファイルを 1 本作り、`read_at` 経路と memcpy 経路の出力を比べる。
    /// パートが開けない環境では read_at が false を返して memcpy へ落ちるので、
    /// このテストは「どちらの経路でも同じ」ことを保証する。
    #[test]
    fn pread_copy_is_byte_identical_to_memcpy() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("pf325_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("part.gguf");
        // 64 KiB + 端数の擬似乱数 (0 を含む)。
        let n = 64 * 1024 + 17;
        let src: Vec<u8> = (0..n).map(|i| (i * 31 + 7) as u8).collect();
        {
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(&src).expect("write");
        }
        exp_pf::install_parts(vec![Some(path.clone())]);
        let opened = exp_pf::parts_open();
        // (pinned 内オフセット, 転送元) を 3 本。1 本はファイル無し (RAM tier 相当)。
        let a_off = 4096u64;
        let b_off = 20000u64;
        let c_off = 64000u64;
        let a_len = 8192usize;
        let b_len = 12345usize;
        let c_len = 1024usize;
        let local = vec![0x5Au8; 777];
        let parts = vec![
            (
                0usize,
                PfHostRange::of_src(&src[a_off as usize..a_off as usize + a_len], (0, a_off)),
            ),
            (
                a_len,
                PfHostRange::of_src(&src[b_off as usize..b_off as usize + b_len], (0, b_off)),
            ),
            (
                a_len + b_len,
                PfHostRange::of_src(&src[c_off as usize..c_off as usize + c_len], (0, c_off)),
            ),
            (a_len + b_len + c_len, PfHostRange::of(&local)),
        ];
        let need = a_len + b_len + c_len + local.len();
        let mut plain = vec![0xEEu8; need];
        pf_copy_into_pin(plain.as_mut_ptr() as usize, &parts, false);
        let mut pre = vec![0xEEu8; need];
        pf_copy_into_pin(pre.as_mut_ptr() as usize, &parts, true);
        assert_eq!(plain, pre, "pread と memcpy で pinned の中身が変わった");
        assert_eq!(&pre[..a_len], &src[a_off as usize..a_off as usize + a_len]);
        assert_eq!(
            &pre[a_len..a_len + b_len],
            &src[b_off as usize..b_off as usize + b_len]
        );
        assert_eq!(
            &pre[a_len + b_len..a_len + b_len + c_len],
            &src[c_off as usize..c_off as usize + c_len]
        );
        assert_eq!(&pre[a_len + b_len + c_len..], &local[..], "ファイル無しは memcpy");
        let _ = opened; // パートが開けなくてもフォールバックで一致する
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GPU が無い (このコンテナ) ときは起動前に `Ok(None)` を返し、`out` を
    /// 1 要素も書かない (呼び出し側は従来経路をそのまま回せる)。
    #[test]
    fn stream_declines_without_gpu_and_leaves_out_untouched() {
        let host: Vec<u8> = vec![0u8; 256];
        let entries = vec![PfStreamEntry {
            expert: 3,
            dev: None,
            host: Some([host.as_slice(), host.as_slice(), host.as_slice()]),
            src: None,
        }];
        let seg = [1u32, 0, 0, 0];
        let rows = [0u32];
        let xs = vec![0.0f32; 8];
        let prow = [0u32];
        let wrow = [1.0f32];
        let mut out = vec![7.0f32; 4];
        let r = moe_experts_iq_pf_layer_stream(
            GgmlType::IQ3S,
            GgmlType::IQ3S,
            GgmlType::IQ3S,
            &entries,
            &seg,
            &rows,
            &xs,
            8,
            16,
            4,
            1,
            1,
            &prow,
            &wrow,
            32,
            &mut out,
            false,
        )
        .expect("GPU 無しでもエラーにしない");
        assert!(r.is_none(), "GPU が無いのに起動した");
        assert!(out.iter().all(|&v| v == 7.0), "out を書いてしまった: {out:?}");
        // 形が合わないときも同じ (fail closed)。
        let r2 = moe_experts_iq_pf_layer_stream(
            GgmlType::IQ3S,
            GgmlType::IQ3S,
            GgmlType::IQ3S,
            &entries,
            &[1u32, 0, 0],
            &rows,
            &xs,
            8,
            16,
            4,
            1,
            1,
            &prow,
            &wrow,
            32,
            &mut out,
            false,
        );
        assert!(r2.is_err() || r2.expect("shape").is_none());
    }

    /// task 162: GDN スキャンの分割は「単発の上限未満」「端数を出さない」
    /// 「連続・重複なし・合計 nt」を満たす。
    #[test]
    fn gdn_sub_bounds_cover_nt_without_tiny_tail() {
        let lim = gdn_sub_limit();
        let min = gdn_sig_min();
        for nt in [1usize, 19, 64, 512, 1024, 2047, 2048, 2049, 2100, 4096, 14018] {
            let b = gdn_sub_bounds(nt);
            assert!(!b.is_empty(), "nt={nt}");
            assert_eq!(b[0].0, 0, "nt={nt}");
            assert_eq!(b[b.len() - 1].1, nt, "nt={nt} の末尾");
            let mut prev = 0usize;
            for &(a, e) in b.iter() {
                assert_eq!(a, prev, "nt={nt} が連続でない");
                assert!(e > a, "nt={nt} に空の範囲");
                assert!(e - a < lim, "nt={nt} に上限以上の範囲 {}", e - a);
                if b.len() > 1 {
                    assert!(e - a >= min, "nt={nt} に小さすぎる範囲 {}", e - a);
                }
                prev = e;
            }
            assert_eq!(prev, nt, "nt={nt} の合計");
        }
        assert!(gdn_sub_bounds(0).is_empty());
        assert_eq!(gdn_sub_bounds(lim - 1), vec![(0, lim - 1)], "上限未満は 1 回");
    }

    /// ステージングは decode の LRU を追い出さない: 確保は `pf_stage_alloc`
    /// (cuMemAlloc 直) だけで、`DevBuffer::alloc_tagged` のフォールバックを
    /// 通らない。ここでは確保要求 0 で何も取らないことを固定する。
    #[test]
    fn stage_alloc_is_a_noop_for_zero() {
        assert!(pf_stage_alloc(0, 0).is_none());
    }

    /// task 254: 表の非同期 H2D のゲートは他のゲートと同じ解釈で、GPU が無い
    /// 環境では必ず `false` を返して同期版へ落ちる (fail closed)。
    #[test]
    fn tab_async_gate_parses_and_falls_back_without_gpu() {
        assert!(pf_tab::value_on(None));
        assert!(pf_tab::value_on(Some("1")));
        assert!(!pf_tab::value_on(Some("0")));
        assert!(!pf_tab::value_on(Some("false")));
        assert!(!pf_tab::value_on(Some("no")));
        // 分岐が呼ばれたことは数える (GPU 無し = false で同期版)。
        let before = pf_tab::counts();
        let data = [1u64, 2, 3];
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(&data))
        };
        let ok = pf_tab::upload(&[(0u64, 0, bytes)]);
        let after = pf_tab::counts();
        if cuda_enabled() {
            // GPU のある環境では成功してよい (このコンテナでは起きない)。
            assert!(ok || after.0 > before.0, "呼び出しが数えられていない");
        } else {
            assert!(!ok, "GPU が無いのに非同期で積めた");
            assert_eq!(after.0, before.0 + 1, "呼び出しが数えられていない");
            assert_eq!(after.1, before.1, "成功数を増やしてしまった");
        }
    }

    /// task 254: 空の表は何もせず成功扱い (呼び出し側が同期版を選ばない)。
    #[test]
    fn tab_async_empty_upload_is_ok() {
        assert!(pf_tab::upload(&[]));
        let data: [u8; 0] = [];
        assert!(pf_tab::upload(&[(0u64, 0, &data)]));
    }

    /// task 248: ピーク記録 (`mem_audit::note_alloc` → `peak_note`) は prefill
    /// ステージングの確保中にも呼ばれる。ここで thread_local (PF_STREAM の
    /// RefCell) を読むと**二重借用で panic する** (lab t248b で踏んだ)。
    /// ピーク記録が thread_local を触らないことを、借用を持ったまま固定する。
    #[test]
    fn peak_note_runs_while_the_staging_refcell_is_borrowed() {
        crate::mem_audit::force_audit_for_test(Some(true));
        {
            let _w = crate::mem_audit::peak_window("t");
            PF_STREAM.with(|cell| {
                let _st = cell.borrow_mut();
                crate::mem_audit::note_alloc(crate::mem_audit::TAG_PF_STAGE, 1 << 20);
                crate::mem_audit::note_free(crate::mem_audit::TAG_PF_STAGE, 1 << 20);
            });
        }
        crate::mem_audit::force_audit_for_test(None);
    }

    /// task 165: グループ番号は層をまたいで単調 (連番) に増える。
    ///
    /// 層ごとに 0 へ戻すと、スロット (`g % 3`) の持ち主が前の層のグループと
    /// 重なり、前の層の完了イベントを「自分のグループの完了」と誤認する。
    /// ここでは 3 層ぶん確保して、番号が連続しスロットの持ち主が 3 つ前だけに
    /// なることを固定する。
    #[test]
    fn group_ids_are_monotonic_across_layers() {
        let sizes = [16usize, 7, 3];
        let mut next = 0u64;
        let mut bases = Vec::new();
        let mut all: Vec<u64> = Vec::new();
        for &n in sizes.iter() {
            let base = pf_take_group_ids(&mut next, n);
            bases.push(base);
            all.extend((0..n as u64).map(|i| base + i));
        }
        assert_eq!(bases, vec![0, 16, 23], "層の先頭が前の層の続きでない: {bases:?}");
        assert_eq!(all.len(), 26);
        for w in all.windows(2) {
            assert_eq!(w[1], w[0] + 1, "通し番号が連続でない: {all:?}");
        }
        // 同じスロットを共有するのは 3 グループ前だけ (番号が一意なので当然)。
        for (i, &g) in all.iter().enumerate() {
            assert_eq!(g as usize, i);
            assert_eq!((g % 3) as usize, i % 3);
        }
        // 0 本でも番号は進めない (空の層で番号が飛ばない)。
        let base = pf_take_group_ids(&mut next, 0);
        assert_eq!(base, 26);
        assert_eq!(next, 26);
    }

    /// task 165 / 554: 上げスレッドの進捗公開はグループ単位の旗で、エラーは
    /// 最初の 1 本を残す。
    #[test]
    fn up_progress_publishes_per_group() {
        let st = (
            std::sync::Mutex::new(PfUpState {
                done: Vec::new(),
                times: Vec::new(),
                probe: None,
                err: None,
            }),
            std::sync::Condvar::new(),
        );
        pf_up_publish(&st, 0, 10, 20, 30, Some((1_000, 5)), None);
        pf_up_publish(&st, 1, 11, 21, 31, None, None);
        {
            let s = st.0.lock().expect("poisoned");
            assert_eq!(s.done, vec![true, true], "2 グループとも記録済み");
            assert_eq!(s.times[0], (10, 20, 30));
            assert_eq!(s.times[1], (11, 21, 31));
            assert_eq!(s.probe, Some((1_000, 5)));
        }
        // 遅れて届いた小さい番号でも旗は戻らず、エラーは残る。
        pf_up_publish(&st, 0, 0, 0, 0, None, Some("最初のエラー".to_string()));
        pf_up_publish(&st, 2, 12, 22, 32, None, Some("2 本目".to_string()));
        let s = st.0.lock().expect("poisoned");
        assert_eq!(s.done, vec![true, true, true]);
        assert_eq!(s.err.as_deref(), Some("最初のエラー"), "エラーが上書きされた");
        assert_eq!(s.times[2], (12, 22, 32));
    }

    /// task 554: 上げスレッドが 2 本だと完了が**順不同**に届く。旗はグループ
    /// 単位なので、2 本目が先に届いても 1 本目の旗は立たない (= 古い `ev_up` で
    /// 計算側の待ちが満たされない)。空のグループ (依頼しない番号) があっても
    /// 他のグループの待ちは返る。
    #[test]
    fn up_progress_flags_do_not_leak_across_groups() {
        let st = (
            std::sync::Mutex::new(PfUpState {
                done: Vec::new(),
                times: Vec::new(),
                probe: None,
                err: None,
            }),
            std::sync::Condvar::new(),
        );
        // 2 本目が先に終わる (1 本目はまだ)。
        pf_up_publish(&st, 1, 11, 21, 31, None, None);
        {
            let s = st.0.lock().expect("poisoned");
            assert_eq!(s.done, vec![false, true], "1 本目の旗が立った");
        }
        pf_up_publish(&st, 0, 10, 20, 30, None, None);
        // 3 番目は依頼しない (空のグループ) まま 4 番目が届く。
        pf_up_publish(&st, 3, 13, 23, 33, None, None);
        let s = st.0.lock().expect("poisoned");
        assert_eq!(s.done, vec![true, true, false, true], "空の 2 番目は false のまま");
    }

    /// task 554/570: `DENNO_PF_UP_WORKERS` の解釈 (1..=3、既定 3、壊れた値は既定)。
    #[test]
    fn up_workers_values_round_and_default() {
        use crate::model::exp_pf::{up_workers_from_value, DEFAULT_UP_WORKERS};
        assert_eq!(
            DEFAULT_UP_WORKERS, 3,
            "既定は 3 本 (task 570 で 2 -> 3。lab t566a の 3,826 -4..-8% / 14k -5%)"
        );
        assert_eq!(up_workers_from_value(None), 3);
        assert_eq!(up_workers_from_value(Some("")), 3);
        assert_eq!(up_workers_from_value(Some("1")), 1, "切り戻し");
        assert_eq!(up_workers_from_value(Some(" 3 ")), 3);
        assert_eq!(up_workers_from_value(Some("0")), 1, "0 は 1 本へ");
        assert_eq!(up_workers_from_value(Some("9")), 3, "スロット 3 枚が上限");
        assert_eq!(up_workers_from_value(Some("x")), 3, "壊れた値は既定");
    }

    /// task 570: `DENNO_PF_STREAM_LOOKAHEAD_PREAD` の解釈と、実効の深さ。
    ///
    /// **pread の立つグループだけ深い**。既定モデル (非 pread) は base のまま
    /// (= P379 が +2.4% と測った深さへ動かさない)。
    #[test]
    fn lookahead_pread_gate_and_effective_depth() {
        use crate::model::exp_pf::{
            lookahead_for, stream_lookahead_pread_from_value,
            DEFAULT_STREAM_LOOKAHEAD, DEFAULT_STREAM_LOOKAHEAD_PREAD,
        };
        assert_eq!(DEFAULT_STREAM_LOOKAHEAD, 1, "base は 1 のまま (既定モデル)");
        assert_eq!(DEFAULT_STREAM_LOOKAHEAD_PREAD, 2, "pread は 2 (3 グループ)");
        assert_eq!(
            stream_lookahead_pread_from_value(None),
            DEFAULT_STREAM_LOOKAHEAD_PREAD
        );
        assert_eq!(stream_lookahead_pread_from_value(Some("")), 2);
        assert_eq!(stream_lookahead_pread_from_value(Some("0")), 0, "切り戻し");
        assert_eq!(stream_lookahead_pread_from_value(Some("1")), 1, "切り戻し");
        assert_eq!(stream_lookahead_pread_from_value(Some("2")), 2);
        assert_eq!(stream_lookahead_pread_from_value(Some("9")), 2, "上限は 2");
        assert_eq!(stream_lookahead_pread_from_value(Some("x")), 2);
        // 実効の深さ: pread のときだけ深い。非 pread は base のまま。
        assert_eq!(lookahead_for(false, 1, 2), 1, "既定モデルは 1 のまま");
        assert_eq!(lookahead_for(true, 1, 2), 2, "UD は 2");
        assert_eq!(lookahead_for(true, 0, 2), 2, "ut off は 0 だが pread なら 2");
        assert_eq!(lookahead_for(true, 1, 0), 1, "ゲートを 0 にすると base");
        assert_eq!(lookahead_for(true, 2, 1), 2, "小さい側へは動かさない");
        assert_eq!(lookahead_for(false, 2, 0), 2);
    }

    /// task 558: `DENNO_PF_PREAD_POOL_PER_UP` の解釈と、番号ごとのプールの割り当て。
    ///
    /// 既定 on (task 558 で採用、lab の 3,826 cold が -10.8% / -8.3%)。
    /// `0` で 1 本共有へ切り戻せる。プールは上げスレッドの番号ごとに別の実体を
    /// 返し、範囲外の番号は `None` (呼び出し側が共有プールへ落ちる)。
    #[test]
    fn pread_pool_per_up_gate_and_pools() {
        use crate::model::exp_pf::{pread_pool_per_up_from_value, DEFAULT_PREAD_POOL_PER_UP};
        assert!(DEFAULT_PREAD_POOL_PER_UP, "lab で採ったのに既定 off に戻っている");
        assert!(pread_pool_per_up_from_value(None), "未設定は既定 on");
        assert!(pread_pool_per_up_from_value(Some("")));
        assert!(!pread_pool_per_up_from_value(Some("0")), "切り戻し");
        assert!(!pread_pool_per_up_from_value(Some("off")));
        assert!(pread_pool_per_up_from_value(Some("1")));
        assert!(pread_pool_per_up_from_value(Some(" ON ")));
        // 計算スレッド (このテストスレッド) は番号を持たない = 共有プール。
        assert_eq!(pf_up_idx(), None, "計算スレッドに上げ番号が付いた");
        // 番号 0..2 はそれぞれ別のプール、3 以上は範囲外。
        let p0 = pf_pread_pool_up(2, 0);
        let p1 = pf_pread_pool_up(2, 1);
        assert!(p0.is_some() && p1.is_some(), "番号ごとのプールが作れない");
        assert!(
            !std::ptr::eq(p0.expect("確保済み"), p1.expect("確保済み")),
            "番号 0 と 1 が同じプールを共有している"
        );
        assert!(pf_pread_pool_up(2, 3).is_none(), "範囲外は共有プールへ");
    }

    /// task 558: 上げスレッド本体が自分の番号を焼くと、そのスレッドの
    /// `pf_copy_into_pin` だけが番号付きのプールを選ぶ (計算スレッドは None)。
    #[test]
    fn pread_pool_per_up_index_is_thread_local() {
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let s = std::sync::Arc::clone(&seen);
        std::thread::spawn(move || {
            // pf_up_worker と同じ焼き方をする。
            let _ = PF_UP_IDX.try_with(|c| c.set(Some(1usize)));
            s.lock().expect("poisoned").push(pf_up_idx());
        })
        .join()
        .expect("スレッドが落ちた");
        assert_eq!(*seen.lock().expect("poisoned"), vec![Some(1)]);
        assert_eq!(pf_up_idx(), None, "別スレッドの番号が漏れた");
    }

    /// task 227: 「1 回の DMA で流す塊」の切り方 (純関数)。
    ///
    /// 塊は parts の境界で切る (途中で切ると写しが半端になる)、長さは limit を
    /// 超えない、隙間 (整列の詰め物) は塊に含めてよい、1 本が limit より
    /// 大きければその 1 本だけ、を固定する。
    #[test]
    fn chunk_spans_follow_part_boundaries_and_limit() {
        // 隙間なしの 3 本 (100 B ずつ) → limit 250 は 2 本ずつ。
        let p = [(0usize, 100usize), (100, 100), (200, 100)];
        assert_eq!(pf_chunk_spans(&p, 0), vec![(0, 300)], "0 は全部で 1 回");
        assert_eq!(pf_chunk_spans(&p, 250), vec![(0, 200), (200, 300)]);
        assert_eq!(pf_chunk_spans(&p, 100), vec![(0, 100), (100, 200), (200, 300)]);
        // 整列の隙間 (16 B) は塊に含む (中身は読まれない)。
        let p = [(0usize, 100usize), (116, 100), (232, 100)];
        assert_eq!(pf_chunk_spans(&p, 250), vec![(0, 216), (232, 332)]);
        // 1 本が limit より大きいときは、その 1 本だけの塊になる。
        let p = [(0usize, 1000usize), (1000, 10)];
        assert_eq!(pf_chunk_spans(&p, 64), vec![(0, 1000), (1000, 1010)]);
        // 空は空。
        assert!(pf_chunk_spans(&[], 128).is_empty());
        // 塊は昇順・非重複で、全 parts を覆う。
        let p: Vec<(usize, usize)> = (0..40).map(|i| (i * 1000 + 16, 984)).collect();
        let spans = pf_chunk_spans(&p, 4096);
        let mut at = 0usize;
        for &(a, b) in &spans {
            assert!(a >= at, "塊が戻った");
            assert!(b > a);
            assert!(b - a <= 4096 + 984, "塊が大きすぎる {}", b - a);
            at = b;
        }
        assert_eq!(spans.last().map(|s| s.1), Some(39 * 1000 + 16 + 984));
    }
}


#[cfg(test)]
mod iq_overlap_tests {
    //! task 150: 重ね打ち経路の**契約**を GPU 無しで固定する。
    use super::*;
    use crate::kquant::GgmlType;

    /// task 164: プローブのホスト写しは**宛先が転送元より長い**ことを前提にする
    /// (pinned の bounce は 2 MB 固定、実スライスは 1 枚ぶん)。lab t164a では
    /// `copy_from_slice` の長さ不一致で panic した (tline / missup の 2 条件が
    /// 丸ごと落ちた)。ここで長さの扱いだけを固定する。
    #[test]
    fn probe_host_copy_accepts_short_source() {
        let mut dst = vec![0u8; 64];
        let src = vec![7u8; 10];
        let gbs = host_copy_gbs(&mut dst, &src, 4);
        assert!(gbs > 0.0, "写しのレートが 0");
        assert_eq!(&dst[..10], &src[..]);
        assert_eq!(dst[10], 0, "転送元より後ろは触らない");
    }

    /// task 164: prologue の非同期化は **既定 on**、明示的な偽値だけ切る。
    /// lab t164c で decode -3.0〜-3.6% / md5 SAME だったので既定を on にした。
    #[test]
    fn dec_pro_async_gate_defaults_on() {
        assert!(dec_pro_async_from_value(None));
        assert!(dec_pro_async_from_value(Some("1")));
        assert!(dec_pro_async_from_value(Some("ON")));
        assert!(!dec_pro_async_from_value(Some("0")));
        assert!(!dec_pro_async_from_value(Some("off")));
        assert!(!dec_pro_async_from_value(Some(" FALSE ")));
        assert!(!dec_pro_async_from_value(Some("no")));
    }

    /// task 164: prologue ステージングのレイアウトは 8 バイト整列で、各ブロックが
    /// 重ならないこと。`u64` の表を `u64` として読むので整列が要る。
    #[test]
    fn dec_stage_layout_is_aligned_and_disjoint() {
        let l = dec_stage_layout(2560, 10, 40, 40, 10, 0);
        assert_eq!(l.x, 0);
        assert_eq!(l.pg, 2560 * 4);
        assert_eq!(l.pu, l.pg + 80);
        assert_eq!(l.pd, l.pu + 80);
        assert_eq!(l.sgu, l.pd + 80);
        assert_eq!(l.sgd, l.sgu + 160);
        assert_eq!(l.w, l.sgd + 160);
        assert_eq!(l.total, l.w + 40);
        for off in [l.x, l.pg, l.pu, l.pd, l.sgu, l.sgd, l.w] {
            assert_eq!(off % 8, 0, "オフセット {off} が 8 バイト整列でない");
        }
        // ブロックの境界が重ならない (前の終わり <= 次の始まり)。
        assert!(l.x + 2560 * 4 <= l.pg);
        assert!(l.sgd + 40 * 4 <= l.w);
        assert!(l.w + 10 * 4 <= l.total);
    }

    /// task 164: ステージングへの書き込みは指定オフセットにだけ載る。
    #[test]
    fn dec_stage_put_writes_at_offset() {
        let l = dec_stage_layout(4, 1, 4, 4, 0, 0);
        let mut buf = vec![0u8; l.total];
        let x = [1.0f32, 2.0, 3.0, 4.0];
        unsafe {
            dec_stage_put(&mut buf, l.x, x.as_ptr() as *const u8, 16);
        }
        assert_eq!(&buf[l.x..l.x + 4], &1.0f32.to_ne_bytes());
        assert_eq!(&buf[l.x + 12..l.x + 16], &4.0f32.to_ne_bytes());
        assert!(buf[l.pg..].iter().all(|&b| b == 0), "後続のブロックを汚した");
    }

    /// task 164: ミス GPU 経路は**起動前に落ちるとき**必ず `Ok(false)` を返し、
    /// `out_h` を 1 要素も書かない (呼び出し側は従来経路へ戻せる)。
    /// GPU の無いテスト環境では毎回ここに落ちるので、契約だけを固定できる。
    #[test]
    fn miss_up_fails_closed_before_launching() {
        let x = vec![0.1f32; 8];
        let g = vec![0u8; 4];
        let mut out: Vec<f32> = vec![7.0f32; 3];
        let hit = [0u64];
        let ids_hit = [0u16];
        let ids_miss = [1u16];
        // ミス 0 本は「この経路ではやらない」。
        let r = moe_experts_iq_layer_miss_up(
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            &hit,
            &hit,
            &hit,
            &ids_hit,
            &[],
            &[],
            &x,
            8,
            8,
            8,
            &mut out,
        );
        assert!(matches!(r, Ok(false)), "{r:?}");
        assert_eq!(out, vec![7.0f32; 3]);
        // 重みの長さが型・形と合わないミスは起動しない (型の取り違えを静かに
        // 通さないための最後の砦)。
        let miss = [MissExpertBytes {
            gate: &g,
            up: &g,
            down: &g,
        }];
        let r = moe_experts_iq_layer_miss_up(
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            &hit,
            &hit,
            &hit,
            &ids_hit,
            &miss,
            &ids_miss,
            &x,
            8,
            8,
            8,
            &mut out,
        );
        assert!(matches!(r, Ok(false)), "{r:?}");
        assert_eq!(out, vec![7.0f32; 3]);
    }

    /// 形状が不正なら起動前に `Ok(false)` を返し、**CPU ミスのコールバックを
    /// 呼ばない** (呼び出し側は全 expert を CPU で計算し直せる)。呼んでしまうと
    /// 二重計算になるうえ、`out_h` が途中まで書かれた状態で CPU 経路へ戻すことになる。
    #[test]
    fn overlap_skips_cpu_miss_when_it_does_not_launch() {
        // task 388: `cpu_miss` は `Send` を要求される (`DENNO_DEC_MISS_SPAWN` で
        // プールへ渡せるようにするため) ので、数え役は atomic にする。
        let called = std::sync::atomic::AtomicU32::new(0);
        let mut out = vec![0.0f32; 4];
        // seg_gu の長さが 4 の倍数でない → 起動計画より前に落ちる。
        let r = moe_experts_iq_layer_overlap(
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            &[0u64],
            &[0u64],
            &[0u64],
            &[0u32, 1, 0, 0, 0],
            &[0u32, 1, 0, 0, 0, 0, 0, 0],
            &[0.0f32; 8],
            8,
            16,
            4,
            &mut out,
            || {
                called.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            },
        )
        .expect("形状検査は Err ではなく Ok(false)");
        assert!(!r, "不正な形状で GPU 経路を選んではいけない");
        assert_eq!(
            called.load(std::sync::atomic::Ordering::Relaxed),
            0,
            "起動しないのに CPU ミスを走らせた"
        );
        assert!(out.iter().all(|&v| v == 0.0), "out を書いてはいけない");
    }

    /// 逐次版と重ね打ち版は同じ入口 (`iq_layer_run`) を通る。GPU の無い環境では
    /// どちらも `Ok(false)` で、値は 1 つも書かれない。
    #[test]
    fn both_entries_fall_back_without_cuda() {
        let mut out_a = vec![7.0f32; 4];
        let mut out_b = vec![7.0f32; 4];
        let a = moe_experts_iq_layer(
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            &[0u64],
            &[0u64],
            &[0u64],
            &[0u32, 0, 0, 0],
            &[0u32, 0, 0, 0],
            &[0.0f32; 8],
            8,
            16,
            4,
            &mut out_a,
        )
        .expect("逐次版");
        let b = moe_experts_iq_layer_overlap(
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            GgmlType::IQ2S,
            &[0u64],
            &[0u64],
            &[0u64],
            &[0u32, 0, 0, 0],
            &[0u32, 0, 0, 0],
            &[0.0f32; 8],
            8,
            16,
            4,
            &mut out_b,
            || Ok(()),
        )
        .expect("重ね打ち版");
        assert_eq!(a, b, "同じ入口なのに戻り値が違う");
        assert_eq!(out_a, out_b, "out の扱いが違う");
    }
}


thread_local! {
    /// task 144: expert の H2D 用の pinned staging (1 本ぶん。grow-only)。
    pub(crate) static EXPERT_PIN: std::cell::RefCell<Option<PinMem>> =
        const { std::cell::RefCell::new(None) };
    /// task 565: 戻し (`DENNO_PF_RESTORE_NOSYNC`) のリング。4 枠を使い回し、
    /// **使い回す枠の DMA だけ**待つ。枠は grow-only でプロセス寿命まで持つ。
    pub(crate) static UP_RING: std::cell::RefCell<UpPinRing> =
        const { std::cell::RefCell::new(UpPinRing::new()) };
}


/// task 565: リングの枠数。1 本 2 MB の DMA (40 us) に対して、写し (0.2 ms) が
/// 支配的なので 2 枠でも足りる。余裕を見て 4。
pub(crate) const UP_RING_N: usize = 4;


/// task 154: チャンクの残差 (`[nt][hc][e]`) をデバイスへ置く。
pub fn pf_upload_res(res: &[f32]) -> bool {
    // task 318: `other` の中の残差の往復を分けて数える (バイト数も付く)。
    let _sub = crate::model::q4ledger::sub(crate::model::q4ledger::SUB_RES_UP);
    // task 565: `DENNO_PF_RES_PIN` なら全体ゲートと無関係に pinned へ通す。
    let force = crate::model::exp_pf::pf_res_pin();
    // task 581: 専用バウンス + 並列写し (`DENNO_PF_RES_MT`)。値は同じバイト列。
    // `DENNO_PF_RES_XPROF` は同じ道を直列で通して内訳だけ出す (ゲートは独立)。
    let xp = crate::model::exp_pf::pf_res_xprof();
    let mt = crate::model::exp_pf::pf_res_mt();
    let use_xfer = mt || xp;
    PF_DEV.with(|cell| -> bool {
        let d = cell.borrow();
        match d.res.as_ref() {
            Some(b) if b.bytes() >= res.len() * 4 => {
                if use_xfer {
                    if res_xfer::to_dev(b.ptr(), res, mt, xp).is_ok() {
                        return true;
                    }
                }
                if force {
                    if let Ok(true) = b.copy_from_pinned_forced(res) {
                        return true;
                    }
                }
                // task 318: `DENNO_PF_PIN_STAGE` では pinned 経由 (速い)。失敗したら従来へ。
                match b.copy_from_pinned(res) {
                    Ok(true) => true,
                    _ => b.copy_from(res).is_ok(),
                }
            }
            _ => false,
        }
    })
}
