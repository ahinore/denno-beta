//! P40: prefill の expert FFN を GPU の可変 m grouped GEMM にまとめる (ホスト側)。
//!
//! # 何をするか
//!
//! 既存の prefill 経路 (`qwen4exp::moe_experts_flat`) は expert ごと・行列ごとに
//! GEMM を回す。ここでは層全体の segment 表 ([`Segment`]) と、P36 のパック済み
//! 重み (`[gate(e0..eN)][up(e0..eN)][down(e0..eN)]`) を作り、
//! `cuda::moe_grouped_gemm_layer` に渡して **1 層 2 起動**で expert FFN を終わらせる。
//!
//! ```text
//! 1. build_grouped_batch   … segment 表 (m 降順) と詰め直した入力 xin
//! 2. plan_grouped          … 使う expert の一覧と packed バッファ内の位置
//! 3. pack_from_sources     … mmap のスライスを 1 本のバッファへコピー (デコード無し)
//! 4. moe_grouped_gemm_layer… gate+up+silu 積 → down の 2 起動
//! 5. scatter_add           … 重み付きでトークン行へ足し込む (expert 昇順)
//! ```
//!
//! # ゲート
//!
//! `DENNO_MOE_GROUPED_GPU=1` のときだけ動く (既定 off)。off のとき
//! [`try_experts`] は先頭で `false` を返し、既存経路は 1 バイトも変わらない。
//!
//! # 数値
//!
//! prefill はビット一致契約の外 (2026-10-28)。足し込みの順序だけは既存の
//! `moe_experts_flat` と同じ (expert 昇順 → トークン昇順) にそろえてある。
//! GPU 結果と CPU 参照の差は `DENNO_MOE_GROUPED_VERIFY=1` で argmax と
//! max abs diff として stderr に出る ([`reference_h`] が CPU 側)。
//!
//! # P36 の pack を使わない理由
//!
//! `moe_grouped::pack_expert_weights` は `GgufFile` を受けるが、推論経路
//! (`LayerW`) は mmap のスライス (`ExpertW::blocks`) しか持たない。レイアウトは
//! `GroupedPlan::block_off` / `strides` をそのまま使うので、書き込み先は P36 と
//! 同じ (テスト `packed_layout_matches_plan` が範囲を固定する)。

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::{bail, Result};

use crate::kquant::GgmlType;
use crate::model::moe_expert::{
    build_grouped_batch, grouped_expert_gemm_cpu, row_bytes, ExpertSlice, Segment,
};
use crate::model::moe_grouped::{plan_grouped, ExpertLayout, ExpertTensorShape};

/// expert 3 テンソルの 1 本 (mmap のバイト列 + 形)。
///
/// `blocks` はテンソル全体 (`[n_out][row_bytes][n_expert]`、expert が最外)。
#[derive(Debug, Clone, Copy)]
pub(crate) struct ExpertSource<'a> {
    pub blocks: &'a [u8],
    pub ty: GgmlType,
    pub n_in: usize,
    pub n_out: usize,
    pub n_expert: usize,
}

impl<'a> ExpertSource<'a> {
    /// 形を [`ExpertTensorShape`] に変換する (P36 の配車と同じ検査を通す)。
    fn shape(&self) -> Result<ExpertTensorShape> {
        ExpertTensorShape::new(self.ty, self.n_in, self.n_out, self.n_expert)
    }

    /// 1 expert ぶんのバイト範囲。
    fn slice(&self, expert: u16) -> Result<ExpertSlice> {
        let s = ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, expert as usize)?;
        if s.end() > self.blocks.len() {
            bail!(
                "expert {expert} の範囲 {}..{} がテンソル長 {} を超える",
                s.off,
                s.end(),
                self.blocks.len()
            );
        }
        Ok(s)
    }
}

/// grouped GEMM を試すか (env `DENNO_MOE_GROUPED_GPU=1`・既定 off)。
#[cfg(feature = "cuda")]
pub(crate) fn enabled() -> bool {
    crate::cuda::moe_grouped_gpu_enabled()
}

/// cuda feature 無しでは常に false (カーネルを積む先が無い)。
#[cfg(not(feature = "cuda"))]
pub(crate) fn enabled() -> bool {
    false
}

/// `DENNO_MOE_GROUPED_VERIFY=1` で CPU 参照との差を stderr に出す。
fn verify_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("DENNO_MOE_GROUPED_VERIFY").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    })
}

/// パック済みバッファの上限 (MiB)。実形 (512 expert × 3.07 MB) で約 1.5 GB。
fn max_pack_bytes() -> usize {
    static BYTES: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let mib = std::env::var("DENNO_MOE_GROUPED_MAX_PACK_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2560);
        mib.saturating_mul(1024 * 1024)
    })
}

/// 1 segment を 5 語へ。[eid, m, w_off, e_off, ff_off]。
///
/// `e_off` / `ff_off` は segment の並びでの「f32 要素単位の先頭」で、
/// gate/up の入力と down の入力 (gate/up の出力) を 1 枚の表で表すために両方持つ。
/// `w_off` は [`Segment`] には無いので配車 (`GroupedPlan`) から引く。
pub(crate) fn seg_table(
    segs: &[Segment],
    w_off: &[u32],
    e_off: &[u32],
    ff_off: &[u32],
) -> Result<Vec<u32>> {
    if segs.len() != w_off.len() || segs.len() != e_off.len() || segs.len() != ff_off.len() {
        bail!(
            "seg_table: 長さが揃っていない (segs={}, w_off={}, e_off={}, ff_off={})",
            segs.len(),
            w_off.len(),
            e_off.len(),
            ff_off.len()
        );
    }
    let mut t = Vec::with_capacity(segs.len() * 5);
    for (i, s) in segs.iter().enumerate() {
        if s.m == 0 {
            bail!("seg_table: segment {i} の m が 0");
        }
        t.push(s.expert as u32);
        t.push(s.m);
        t.push(w_off[i]);
        t.push(e_off[i]);
        t.push(ff_off[i]);
    }
    Ok(t)
}

/// mmap のスライスからパック済みバッファを埋める。
///
/// レイアウトは `plan` が決める (`block_off[k] + i * strides[k]`)。量子ブロックは
/// デコードせずバイト列を移すだけなので、値は元テンソルと同一。
fn pack_from_sources(
    plan: &crate::model::moe_grouped::GroupedPlan,
    sources: [&ExpertSource<'_>; 3],
    dst: &mut [u8],
) -> Result<()> {
    if dst.len() != plan.packed_bytes {
        bail!(
            "pack: 宛先の長さ {} が packed_bytes {} と違う",
            dst.len(),
            plan.packed_bytes
        );
    }
    let strides = plan.strides()?;
    for (i, &expert) in plan.expert_ids.iter().enumerate() {
        for (b, src) in sources.iter().enumerate() {
            let sl = src.slice(expert)?;
            let d0 = plan.block_off[b] + i * strides[b];
            dst[d0..d0 + sl.len].copy_from_slice(&src.blocks[sl.off..sl.end()]);
        }
    }
    Ok(())
}

/// パック済みブロックから「expert id で引ける」スライス列を作る
/// ([`grouped_expert_gemm_cpu`] の `weights` 引数の形)。
fn packed_by_expert<'a>(
    packed: &'a [u8],
    block_off: usize,
    stride: usize,
    expert_ids: &[u16],
    n_expert: usize,
) -> Result<Vec<&'a [u8]>> {
    let mut w: Vec<&'a [u8]> = vec![&[]; n_expert];
    for (i, &eid) in expert_ids.iter().enumerate() {
        let d0 = block_off + i * stride;
        if d0 + stride > packed.len() {
            bail!("packed_by_expert: expert {eid} の範囲が {} を超える", packed.len());
        }
        w[eid as usize] = &packed[d0..d0 + stride];
    }
    Ok(w)
}

/// gate/up 用と down 用の segment 表を作り直す (offset の意味が入れ替わる)。
fn offsets_for_pass(segs: &[Segment], n_in: usize, n_out: usize) -> (Vec<u32>, Vec<u32>) {
    let mut xo = 0u32;
    let mut oo = 0u32;
    let mut xs = Vec::with_capacity(segs.len());
    let mut os = Vec::with_capacity(segs.len());
    for s in segs {
        xs.push(xo);
        os.push(oo);
        xo += s.m * n_in as u32;
        oo += s.m * n_out as u32;
    }
    (xs, os)
}

/// GPU カーネルと同じ式を CPU で書いた参照実装 (act = silu(g)*u → h = down·act)。
///
/// `DENNO_MOE_GROUPED_VERIFY=1` の比較と、テストの基準に使う。丸めは
/// [`grouped_expert_gemm_cpu`] (ビット一致の基準) と同じ。
pub(crate) fn reference_h(
    gate: &ExpertSource<'_>,
    up: &ExpertSource<'_>,
    down: &ExpertSource<'_>,
    plan: &crate::model::moe_grouped::GroupedPlan,
    packed: &[u8],
    segs: &[Segment],
    xin: &[f32],
) -> Result<Vec<f32>> {
    let (e, ff) = (gate.n_in, gate.n_out);
    let m_total: usize = segs.iter().map(|s| s.m as usize).sum();
    let strides = plan.strides()?;
    let wg = packed_by_expert(
        packed,
        plan.block_off[0],
        strides[0],
        &plan.expert_ids,
        gate.n_expert,
    )?;
    let wu = packed_by_expert(
        packed,
        plan.block_off[1],
        strides[1],
        &plan.expert_ids,
        up.n_expert,
    )?;
    let wd = packed_by_expert(
        packed,
        plan.block_off[2],
        strides[2],
        &plan.expert_ids,
        down.n_expert,
    )?;
    let (x_gu, o_gu) = offsets_for_pass(segs, e, ff);
    let gu_segs: Vec<Segment> = segs
        .iter()
        .enumerate()
        .map(|(i, s)| Segment {
            expert: s.expert,
            m: s.m,
            x_off: x_gu[i],
            out_off: o_gu[i],
        })
        .collect();
    let (x_d, o_d) = offsets_for_pass(segs, ff, e);
    let d_segs: Vec<Segment> = segs
        .iter()
        .enumerate()
        .map(|(i, s)| Segment {
            expert: s.expert,
            m: s.m,
            x_off: x_d[i],
            out_off: o_d[i],
        })
        .collect();

    let mut g = vec![0.0f32; m_total * ff];
    let mut u = vec![0.0f32; m_total * ff];
    grouped_expert_gemm_cpu(gate.ty, &wg, e, ff, &gu_segs, xin, &mut g)?;
    grouped_expert_gemm_cpu(up.ty, &wu, e, ff, &gu_segs, xin, &mut u)?;
    let mut act = vec![0.0f32; m_total * ff];
    for i in 0..m_total * ff {
        act[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
    }
    let mut h = vec![0.0f32; m_total * e];
    grouped_expert_gemm_cpu(down.ty, &wd, ff, e, &d_segs, &act, &mut h)?;
    Ok(h)
}

/// 重み付きの足し込み (expert 昇順 → トークン昇順)。
///
/// 並びを `moe_experts_flat` の scatter と同じにしてあるので、寄与の和の順序が
/// 変わらない (各トークンの `out` は expert 昇順に足される)。
pub(crate) fn scatter_add(
    out: &mut [f32],
    h: &[f32],
    segs: &[Segment],
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    e: usize,
) -> Result<()> {
    let ne = by_expert.len();
    if e == 0 || out.len() % e != 0 {
        bail!("scatter_add: out の長さ {} が e={e} で割り切れない", out.len());
    }
    // expert id → (h の行先頭, m)。h の行は segment の並び (m 降順) に
    // 詰まっているので、行先頭は `Σ m` の累積で求める
    // (`Segment::out_off` は f32 要素単位で n_out 倍されており、行番号ではない)。
    let mut at: Vec<Option<(usize, usize)>> = vec![None; ne];
    let mut row = 0usize;
    for s in segs {
        if at[s.expert as usize].is_some() {
            bail!("scatter_add: expert {} の segment が 2 本ある", s.expert);
        }
        at[s.expert as usize] = Some((row, s.m as usize));
        row += s.m as usize;
    }
    if out.len() < h.len() {
        bail!(
            "scatter_add: h の長さ {} が out の長さ {} を超える",
            h.len(),
            out.len()
        );
    }
    if h.len() != row * e {
        bail!(
            "scatter_add: h の長さ {} が Σ m({row}) * e({e}) と違う",
            h.len()
        );
    }
    for (ei, slot) in at.iter().enumerate() {
        let Some((r0, m)) = *slot else { continue };
        let picks = &by_expert[ei];
        if picks.len() != m {
            bail!(
                "scatter_add: expert {ei} の picks {} が segment の m {m} と違う",
                picks.len()
            );
        }
        for (i, &(t, j)) in picks.iter().enumerate() {
            let w = weights[t * k + j];
            let src = &h[(r0 + i) * e..(r0 + i + 1) * e];
            let dst = &mut out[t * e..(t + 1) * e];
            for d in 0..e {
                dst[d] += src[d] * w;
            }
        }
    }
    Ok(())
}

/// 前回のパック済みバッファ (1 本だけ持ち回る)。
struct PackCache {
    ids: Vec<u16>,
    bytes: Vec<u8>,
}

static PACK_CACHE: std::sync::Mutex<Option<PackCache>> = std::sync::Mutex::new(None);

/// パック済みバッファを借りて `f` を呼び、返す。
fn with_packed<R>(
    plan: &crate::model::moe_grouped::GroupedPlan,
    sources: [&ExpertSource<'_>; 3],
    f: impl FnOnce(&[u8]) -> R,
) -> Result<R> {
    if plan.packed_bytes > max_pack_bytes() {
        bail!(
            "packed {} MiB が上限 {} MiB を超える (DENNO_MOE_GROUPED_MAX_PACK_MB)",
            plan.packed_bytes >> 20,
            max_pack_bytes() >> 20
        );
    }
    let mut g = PACK_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    let mut cache = g.take().unwrap_or(PackCache {
        ids: Vec::new(),
        bytes: Vec::new(),
    });
    if cache.bytes.len() != plan.packed_bytes {
        cache.bytes.resize(plan.packed_bytes, 0);
        cache.ids.clear();
    }
    if cache.ids != plan.expert_ids {
        pack_from_sources(plan, sources, &mut cache.bytes)?;
        cache.ids = plan.expert_ids.clone();
    }
    let r = f(&cache.bytes);
    *g = Some(cache);
    Ok(r)
}

/// パック内容の識別子 (同じ層・同じ expert 集合なら同じ値)。
fn pack_stamp(layer_key: u64, ids: &[u16]) -> u64 {
    let mut h = DefaultHasher::new();
    layer_key.hash(&mut h);
    ids.hash(&mut h);
    h.finish()
}

/// grouped expert FFN を GPU で走らせ、`out` (ゼロ初期化済み・`nt * e`) へ足し込む。
///
/// 戻り値 `false` は「GPU でやらなかった」。呼び出し側は従来の CPU 経路
/// (`moe_experts_flat` / legacy) をそのまま実行すること。
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_experts(
    layer_key: u64,
    gate: ExpertSource<'_>,
    up: ExpertSource<'_>,
    down: ExpertSource<'_>,
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    xs: &[f32],
    nt: usize,
    e: usize,
    ff: usize,
    out: &mut [f32],
) -> Result<bool> {
    if !enabled() {
        return Ok(false);
    }
    if xs.len() != nt * e || out.len() != nt * e {
        bail!("moe_grouped_gpu: 入出力の長さが nt({nt}) * e({e}) と違う");
    }
    // 形の検査は配車の前に行う (壊れた形で pack へ進まない)。
    for (name, s) in [("gate", &gate), ("up", &up), ("down", &down)] {
        if s.blocks.len() != s.n_out * row_bytes(s.ty, s.n_in)? * s.n_expert {
            bail!("moe_grouped_gpu: {name} のバイト数が形と違う");
        }
    }
    // 1. 詰め直し + segment 表 (m 降順)
    let batch = build_grouped_batch(by_expert, xs, e, ff)?;
    if batch.segs.is_empty() {
        return Ok(false);
    }
    // 2. 配車 (P36)
    let layout = ExpertLayout::new(gate.shape()?, up.shape()?, down.shape()?)?;
    let plan = plan_grouped(&batch.segs, &layout)?;
    let m_total = plan.m_total();
    // 3. パック + 起動 + ばらまき
    let (e_off, ff_off) = {
        let mut a = Vec::with_capacity(batch.segs.len());
        let mut b = Vec::with_capacity(batch.segs.len());
        for s in &batch.segs {
            a.push(s.x_off);
            b.push(s.out_off);
        }
        (a, b)
    };
    let table = seg_table(&batch.segs, &plan.w_off, &e_off, &ff_off)?;
    let stamp = pack_stamp(layer_key, &plan.expert_ids);

    let ran = with_packed(&plan, [&gate, &up, &down], |packed| -> Result<bool> {
        let mut h = vec![0.0f32; m_total * e];
        let ok = launch(
            stamp,
            packed,
            &table,
            &gate,
            &up,
            &down,
            &plan,
            &batch.xs,
            &mut h,
        )?;
        if !ok {
            return Ok(false);
        }
        if verify_on() {
            let cpu = reference_h(&gate, &up, &down, &plan, packed, &batch.segs, &batch.xs)?;
            let mut max_abs = 0.0f32;
            let mut am_g = 0usize;
            let mut am_c = 0usize;
            for i in 0..h.len() {
                let d = (h[i] - cpu[i]).abs();
                if d > max_abs {
                    max_abs = d;
                }
                if h[i] > h[am_g] {
                    am_g = i;
                }
                if cpu[i] > cpu[am_c] {
                    am_c = i;
                }
            }
            eprintln!(
                "[moe-grouped] nseg={} m_total={m_total} argmax_gpu={am_g}({}) argmax_cpu={am_c}({}) max_abs_diff={max_abs:.6e} argmax_eq={}",
                batch.segs.len(),
                h[am_g],
                cpu[am_c],
                am_g == am_c
            );
        }
        scatter_add(&mut *out, &h, &batch.segs, by_expert, weights, k, e)?;
        Ok(true)
    })??;
    Ok(ran)
}

/// GPU 起動 (cuda feature 無しは常に false)。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn launch(
    stamp: u64,
    packed: &[u8],
    table: &[u32],
    gate: &ExpertSource<'_>,
    up: &ExpertSource<'_>,
    down: &ExpertSource<'_>,
    plan: &crate::model::moe_grouped::GroupedPlan,
    xin: &[f32],
    h: &mut [f32],
) -> Result<bool> {
    let g = crate::cuda::GroupedMatrix {
        ty: gate.ty,
        n_in: gate.n_in,
        n_out: gate.n_out,
        row_bytes: row_bytes(gate.ty, gate.n_in)?,
        w_off: plan.block_off[0],
    };
    let u = crate::cuda::GroupedMatrix {
        ty: up.ty,
        n_in: up.n_in,
        n_out: up.n_out,
        row_bytes: row_bytes(up.ty, up.n_in)?,
        w_off: plan.block_off[1],
    };
    let d = crate::cuda::GroupedMatrix {
        ty: down.ty,
        n_in: down.n_in,
        n_out: down.n_out,
        row_bytes: row_bytes(down.ty, down.n_in)?,
        w_off: plan.block_off[2],
    };
    crate::cuda::moe_grouped_gemm_layer(stamp, packed, table, g, u, d, xin, h)
}

/// cuda feature 無しのスタブ。
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn launch(
    _stamp: u64,
    _packed: &[u8],
    _table: &[u32],
    _gate: &ExpertSource<'_>,
    _up: &ExpertSource<'_>,
    _down: &ExpertSource<'_>,
    _plan: &crate::model::moe_grouped::GroupedPlan,
    _xin: &[f32],
    _h: &mut [f32],
) -> Result<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // 合成フィクスチャの量子値を「全ブロック同じ」にしないためのロック
    // (乱数は使わず、テンソルごとの seed で決定的にずらす)。
    static FIXTURE_LOCK: Mutex<()> = Mutex::new(());

    const N_IN: usize = 512; // Q4_K のブロック長 256 で割り切れる
    const N_OUT: usize = 64; // Q5_1 (down の n_in) のブロック長 32 で割り切れる
    const N_EXPERT: usize = 4;

    /// Q4_K / Q5_1 のバイト列を「それらしい」値で埋める。
    ///
    /// `crate::testutil` の実量子化器を使うと依存が増えるので、ここでは
    /// ブロックの d / dmin / scale / qs を seed から決定的に書き込む
    /// (値の正しさはこのテストの目的ではない。**同じ入力に対する 2 経路の
    /// 一致**だけを見る)。
    fn make_blocks(ty: GgmlType, n_in: usize, n_out: usize, n_expert: usize, seed: u64) -> Vec<u8> {
        let rb = row_bytes(ty, n_in).expect("row_bytes");
        let mut out = vec![0u8; rb * n_out * n_expert];
        let mut st = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut next = move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (st >> 33) as u32
        };
        for e in 0..n_expert {
            for r in 0..n_out {
                let base = (e * n_out + r) * rb;
                let blkb = match ty {
                    GgmlType::Q4K => 144,
                    GgmlType::Q5K => 176,
                    GgmlType::Q5_1 => 24,
                    GgmlType::Q8_0 => 34,
                    _ => panic!("未対応"),
                };
                for b in 0..(rb / blkb) {
                    for k in 0..blkb {
                        out[base + b * blkb + k] = (next() & 0xff) as u8;
                    }
                    // d / dmin は f16 のまま乱数にすると inf / NaN になり、
                    // 「2 経路が同じ値になる」ことの検証にならない。有限の
                    // 正規化数 (0x2400 = 2^-6) に固定する。
                    out[base + b * blkb] = 0x00;
                    out[base + b * blkb + 1] = 0x24;
                    out[base + b * blkb + 2] = 0x00;
                    out[base + b * blkb + 3] = 0x24;
                }
            }
        }
        out
    }

    fn source(ty: GgmlType, n_in: usize, n_out: usize) -> ExpertSource<'static> {
        let blocks = make_blocks(ty, n_in, n_out, N_EXPERT, 7);
        ExpertSource {
            blocks: Box::leak(blocks.into_boxed_slice()),
            ty,
            n_in,
            n_out,
            n_expert: N_EXPERT,
        }
    }

    /// 詰め直した入力と picks (3 expert に m を変えて割り当てる)。
    fn fixture() -> (
        Vec<Vec<(usize, usize)>>,
        Vec<f32>,
        Vec<f32>,
        usize,
        usize,
    ) {
        let (e, ff, k, nt) = (N_IN, N_OUT, 2, 9);
        // expert 1 に 5、expert 3 に 3、expert 0 に 1 トークン。
        let mut picks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); N_EXPERT];
        let assign = [
            (1usize, 0usize),
            (1, 1),
            (3, 0),
            (1, 0),
            (0, 1),
            (3, 1),
            (1, 1),
            (3, 0),
            (1, 0),
        ];
        for (t, &(ei, j)) in assign.iter().enumerate() {
            picks[ei].push((t, j));
        }
        let mut xs = vec![0.0f32; nt * e];
        for i in 0..xs.len() {
            xs[i] = ((i % 17) as f32 - 8.0) * 0.0625;
        }
        let mut w = vec![0.0f32; nt * k];
        for i in 0..w.len() {
            w[i] = 0.25 + (i % 5) as f32 * 0.125;
        }
        (picks, xs, w, e, ff)
    }

    fn pack(
        gate: &ExpertSource<'_>,
        up: &ExpertSource<'_>,
        down: &ExpertSource<'_>,
    ) -> Result<(
        Vec<u8>,
        crate::model::moe_grouped::GroupedPlan,
        Vec<Segment>,
        Vec<f32>,
    )> {
        let (picks, xs, _w, e, ff) = fixture();
        let batch = build_grouped_batch(&picks, &xs, e, ff)?;
        let layout = ExpertLayout::new(gate.shape()?, up.shape()?, down.shape()?)?;
        let plan = plan_grouped(&batch.segs, &layout)?;
        let mut packed = vec![0u8; plan.packed_bytes];
        pack_from_sources(&plan, [gate, up, down], &mut packed)?;
        Ok((packed, plan, batch.segs, batch.xs))
    }

    /// per-expert (従来の並べ方) の参照: expert ごとに 3 行列を回して足し込む。
    fn per_expert_reference(
        gate: &ExpertSource<'_>,
        up: &ExpertSource<'_>,
        down: &ExpertSource<'_>,
        picks: &[Vec<(usize, usize)>],
        weights: &[f32],
        xs: &[f32],
        nt: usize,
        e: usize,
        ff: usize,
    ) -> Vec<f32> {
        let k = if nt == 0 { 0 } else { weights.len() / nt };
        let mut out = vec![0.0f32; nt * e];
        for (ei, p) in picks.iter().enumerate() {
            if p.is_empty() {
                continue;
            }
            let m = p.len();
            let mut xin = vec![0.0f32; m * e];
            for (i, &(t, _)) in p.iter().enumerate() {
                xin[i * e..(i + 1) * e].copy_from_slice(&xs[t * e..(t + 1) * e]);
            }
            let seg = [Segment {
                expert: ei as u16,
                m: m as u32,
                x_off: 0,
                out_off: 0,
            }];
            let s = gate.slice(ei as u16).expect("slice");
            let g = &gate.blocks[s.off..s.end()];
            let s = up.slice(ei as u16).expect("slice");
            let u = &up.blocks[s.off..s.end()];
            let s = down.slice(ei as u16).expect("slice");
            let d = &down.blocks[s.off..s.end()];
            // `weights` は expert id で引かれるので、使う 1 本だけを置いた表を作る。
            let mut wv: Vec<&[u8]> = vec![&[]; N_EXPERT];
            wv[ei] = g;
            let mut act = vec![0.0f32; m * ff];
            let mut upo = vec![0.0f32; m * ff];
            grouped_expert_gemm_cpu(gate.ty, &wv, e, ff, &seg, &xin, &mut act).expect("gate");
            let mut wv: Vec<&[u8]> = vec![&[]; N_EXPERT];
            wv[ei] = u;
            grouped_expert_gemm_cpu(up.ty, &wv, e, ff, &seg, &xin, &mut upo).expect("up");
            for i in 0..m * ff {
                act[i] = act[i] / (1.0 + (-act[i]).exp()) * upo[i];
            }
            let mut h = vec![0.0f32; m * e];
            let mut wv: Vec<&[u8]> = vec![&[]; N_EXPERT];
            wv[ei] = d;
            grouped_expert_gemm_cpu(down.ty, &wv, ff, e, &seg, &act, &mut h).expect("down");
            for (i, &(t, j)) in p.iter().enumerate() {
                let w = weights[t * k + j];
                for d in 0..e {
                    out[t * e + d] += h[i * e + d] * w;
                }
            }
        }
        out
    }

    #[test]
    fn packed_layout_matches_plan() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gate = source(GgmlType::Q4K, N_IN, N_OUT);
        let up = source(GgmlType::Q4K, N_IN, N_OUT);
        let down = source(GgmlType::Q5_1, N_OUT, N_IN);
        let (packed, plan, _segs, _xin) = pack(&gate, &up, &down).expect("pack");
        // 3 ブロックが隙間なく並び、各 expert のバイト列が元テンソルと同一。
        let strides = plan.strides().expect("strides");
        assert_eq!(plan.block_off[0], 0);
        assert_eq!(plan.block_off[1], strides[0] * plan.n_expert());
        assert_eq!(
            plan.block_off[2],
            strides[0] * plan.n_expert() + strides[1] * plan.n_expert()
        );
        assert_eq!(packed.len(), plan.packed_bytes);
        for (i, &eid) in plan.expert_ids.iter().enumerate() {
            let s = gate.slice(eid).expect("slice");
            let d0 = plan.block_off[0] + i * strides[0];
            assert_eq!(
                &packed[d0..d0 + strides[0]],
                &gate.blocks[s.off..s.end()],
                "gate expert {eid} のバイト列が違う"
            );
        }
    }

    #[test]
    fn seg_table_offsets_are_cumulative() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (picks, xs, _w, e, ff) = fixture();
        let batch = build_grouped_batch(&picks, &xs, e, ff).expect("batch");
        let (xo, ff_o) = offsets_for_pass(&batch.segs, e, ff);
        let w_off: Vec<u32> = (0..batch.segs.len() as u32).map(|i| i * 7).collect();
        let t = seg_table(&batch.segs, &w_off, &xo, &ff_o).expect("seg_table");
        assert_eq!(t.len(), batch.segs.len() * 5);
        for (i, s) in batch.segs.iter().enumerate() {
            assert_eq!(t[i * 5], s.expert as u32);
            assert_eq!(t[i * 5 + 1], s.m);
            assert_eq!(t[i * 5 + 3], xo[i]);
            assert_eq!(t[i * 5 + 4], ff_o[i]);
            // e_off / ff_off は m の累積 (segment の並び順)
            let want_e: u32 = batch.segs[..i].iter().map(|s| s.m * e as u32).sum();
            let want_ff: u32 = batch.segs[..i].iter().map(|s| s.m * ff as u32).sum();
            assert_eq!(xo[i], want_e);
            assert_eq!(ff_o[i], want_ff);
        }
        // m 降順に並んでいる (配車の契約)
        assert!(batch.segs.windows(2).all(|w| w[0].m >= w[1].m));
    }

    #[test]
    fn reference_matches_per_expert_bitwise() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gate = source(GgmlType::Q4K, N_IN, N_OUT);
        let up = source(GgmlType::Q4K, N_IN, N_OUT);
        let down = source(GgmlType::Q5_1, N_OUT, N_IN);
        let (picks, xs, weights, e, ff) = fixture();
        let nt = xs.len() / e;
        let (packed, plan, segs, xin) = pack(&gate, &up, &down).expect("pack");
        // reference_h は「詰め直した入力」を受ける (segment の並び)。
        let h = reference_h(&gate, &up, &down, &plan, &packed, &segs, &xin).expect("reference_h");
        let mut got = vec![0.0f32; nt * e];
        scatter_add(&mut got, &h, &segs, &picks, &weights, 2, e).expect("scatter");
        let want =
            per_expert_reference(&gate, &up, &down, &picks, &weights, &xs, nt, e, ff);
        assert_eq!(got.len(), want.len());
        for i in 0..got.len() {
            assert_eq!(
                got[i].to_bits(),
                want[i].to_bits(),
                "要素 {i}: grouped {} vs per-expert {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn try_experts_is_off_by_default() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // env を立てていないので経路に入らない = out は 1 バイトも変わらない。
        assert!(!enabled(), "DENNO_MOE_GROUPED_GPU 未設定では off のはず");
        let gate = source(GgmlType::Q4K, N_IN, N_OUT);
        let up = source(GgmlType::Q4K, N_IN, N_OUT);
        let down = source(GgmlType::Q5_1, N_OUT, N_IN);
        let (picks, xs, weights, e, ff) = fixture();
        let nt = xs.len() / e;
        let mut out = vec![7.0f32; nt * e];
        let before = out.clone();
        let did = try_experts(1, gate, up, down, &picks, &weights, 2, &xs, nt, e, ff, &mut out)
            .expect("try_experts");
        assert!(!did);
        assert_eq!(out, before);
    }
}
