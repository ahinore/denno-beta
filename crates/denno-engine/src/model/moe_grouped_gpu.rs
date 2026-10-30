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
//! 5. ScatterPlan           … 「h の行 → (トークン, スロット)」の対応表
//! 6. scatter               … その表の順でトークン行へ足し込む (expert 昇順)
//! ```
//!
//! # 行の対応 (P42)
//!
//! `h` の行は **segment の並び** (m 降順、行数 `Σ m = nt × k`)、`out` の行は
//! **トークンの並び** (行数 `nt`)。両者は行数が違うので、長さの大小で検査しては
//! いけない (v1 は `out.len() < h.len()` を要求して実機の全 48 層で落ちた)。
//! 対応は [`ScatterPlan`] が明示的な配列として持ち、scatter はそれだけを根拠にする。
//!
//! # 重みの番地 (P50)
//!
//! 実機で出力が `[PAD…]` の連続になった原因は 3 つとも「別の expert の重みを
//! 読んでいた」こと:
//!
//! 1. segment 表の `w_off` を **segment の添字**で `plan.w_off` から引いていた。
//!    segment は m 降順、packed は expert 昇順なので、並びが一致しない層では
//!    全 expert が別人の重みになる。→ expert id で引く
//!    ([`crate::model::moe_grouped::GroupedPlan::w_off_for`])。
//! 2. down の `w_off` に gate のストライドを使っていた。実形は gate 921,600 B /
//!    down 1,228,800 B で違う。→ down 専用の表を作り、別のデバイスバッファで渡す。
//! 3. パック済みバッファの使い回しが expert 集合だけで、**層 id を見ていなかった**。
//!    実形は全層でほぼ全 512 expert を使うので、2 層目以降が 1 層目の重みで
//!    計算されていた。→ [`PackCache`] の鍵に層 id を入れる。
//!
//! 1 と 2 は `check_seg_addresses` が起動のたびに配車と突き合わせる。3 は
//! `DENNO_MOE_GROUPED_CHECK=1` (最初の 1 層だけ、元テンソルからの CPU 参照と
//! 比較) で値の差として出る。
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

/// grouped 経路の失敗回数 (層ごとに 1 行出さないための集計)。
static FAILURES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// 失敗を記録する。**詳細は最初の 1 回だけ**出し、以降は数えるだけ。
///
/// 実機では 48 層すべてが同じ理由で落ちるので、層ごとに 1 行出すと 48 行になる。
/// 集計行は [`flush_failures`] が prefill の最後に 1 回だけ出す。
pub(crate) fn note_failure(layer: usize, err: &anyhow::Error) {
    let n = FAILURES.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n == 1 {
        eprintln!(
            "[moe-grouped] 層 {layer}: 失敗 → CPU 経路へ: {err:#} (以降の層は同じ理由なので集計に畳む)"
        );
    }
}

/// 集計行を 1 回だけ出す (0 件なら何も出さない)。返り値は失敗回数。
pub(crate) fn flush_failures() -> usize {
    let n = FAILURES.swap(0, std::sync::atomic::Ordering::Relaxed);
    if n > 1 {
        eprintln!(
            "[moe-grouped] grouped GPU 経路の失敗 {n} 回 → 該当層は CPU 経路で計算 (理由は最初の 1 行)"
        );
    }
    n
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
/// `w_off` は [`GroupedPlan`] から **expert id で**引く
/// ([`crate::model::moe_grouped::GroupedPlan::w_off_for`])。`pass` は
/// 0 = gate / 1 = up / 2 = down で、**down はストライドが gate と違う**ので
/// パスごとに表を作り直す (P50)。
pub(crate) fn seg_table(
    segs: &[Segment],
    plan: &crate::model::moe_grouped::GroupedPlan,
    pass: usize,
    e_off: &[u32],
    ff_off: &[u32],
) -> Result<Vec<u32>> {
    if segs.len() != e_off.len() || segs.len() != ff_off.len() {
        bail!(
            "seg_table: 長さが揃っていない (segs={}, e_off={}, ff_off={})",
            segs.len(),
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
        t.push(plan.w_off_for(s.expert, pass)?);
        t.push(e_off[i]);
        t.push(ff_off[i]);
    }
    Ok(t)
}

/// segment 表の `w_off` が配車 (`GroupedPlan`) の expert スライスと一致するか。
///
/// GPU 無しで回る純関数。P50 の実機バグ (segment の並び順で `w_off` を引き、
/// 別 expert の重みを読んでいた) と同じ種類の取り違えを、起動のたびに
/// 1 層あたり O(nseg · log n_expert) で落とす。
pub(crate) fn check_seg_addresses(
    plan: &crate::model::moe_grouped::GroupedPlan,
    segs: &[Segment],
    table_gu: &[u32],
    table_dn: &[u32],
) -> Result<()> {
    if table_gu.len() != segs.len() * 5 || table_dn.len() != segs.len() * 5 {
        bail!(
            "seg 表の長さが segment 数と違う (segs={}, gu={}, dn={})",
            segs.len(),
            table_gu.len(),
            table_dn.len()
        );
    }
    for (i, s) in segs.iter().enumerate() {
        let Some(sl) = plan.expert_slices(s.expert) else {
            bail!("segment {i}: expert {} のスライスが配車に無い", s.expert);
        };
        let want_g = sl[0].start - plan.block_off[0];
        let want_d = sl[2].start - plan.block_off[2];
        let gu = table_gu[i * 5 + 2] as usize;
        let dn = table_dn[i * 5 + 2] as usize;
        if gu != want_g || dn != want_d {
            bail!(
                "segment {i} (expert {}): w_off が gate {gu}/down {dn} で、配車の \
                 gate {want_g}/down {want_d} と違う (別 expert の重みを読む)",
                s.expert
            );
        }
        // 1 expert ぶんがブロック内に収まること (範囲外は隣のブロックを読む)。
        if sl[0].end > plan.block_off[1] || sl[2].end > plan.packed_bytes {
            bail!(
                "segment {i} (expert {}): スライス {:?} がブロックの外へ出る",
                s.expert,
                sl
            );
        }
    }
    Ok(())
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
    reference_h_with_weights(
        gate.ty, up.ty, down.ty, &wg, &wu, &wd, gate.n_in, gate.n_out, segs, xin,
    )
}

/// mmap のスライスから直接作る CPU 参照 (**pack を経由しない**)。
///
/// `DENNO_MOE_GROUPED_VERIFY` / `DENNO_MOE_GROUPED_CHECK` の比較相手。
/// [`reference_h`] は pack 済みバッファを読むので、pack が古い (層の取り違え)
/// ときは GPU と同じ誤った重みで計算してしまい、差が出ない。こちらは元の
/// テンソルを読むので、pack の取り違えも値の差として出る。
pub(crate) fn reference_h_from_sources(
    gate: &ExpertSource<'_>,
    up: &ExpertSource<'_>,
    down: &ExpertSource<'_>,
    segs: &[Segment],
    xin: &[f32],
) -> Result<Vec<f32>> {
    let wg = slices_by_expert(gate, segs)?;
    let wu = slices_by_expert(up, segs)?;
    let wd = slices_by_expert(down, segs)?;
    reference_h_with_weights(
        gate.ty, up.ty, down.ty, &wg, &wu, &wd, gate.n_in, gate.n_out, segs, xin,
    )
}

/// `segs` に出てくる expert の 1 本ぶんのバイト列を expert id で引ける表にする。
fn slices_by_expert<'a>(src: &ExpertSource<'a>, segs: &[Segment]) -> Result<Vec<&'a [u8]>> {
    let mut w: Vec<&'a [u8]> = vec![&[]; src.n_expert];
    for s in segs {
        let eid = s.expert as usize;
        if eid >= src.n_expert {
            bail!("expert {eid} が範囲外 (n_expert={})", src.n_expert);
        }
        if !w[eid].is_empty() {
            continue;
        }
        let sl = src.slice(s.expert)?;
        w[eid] = &src.blocks[sl.off..sl.end()];
    }
    Ok(w)
}

/// 3 行列の「expert id で引けるスライス列」から参照 h を作る共通部。
#[allow(clippy::too_many_arguments)]
fn reference_h_with_weights(
    gate_ty: GgmlType,
    up_ty: GgmlType,
    down_ty: GgmlType,
    wg: &[&[u8]],
    wu: &[&[u8]],
    wd: &[&[u8]],
    e: usize,
    ff: usize,
    segs: &[Segment],
    xin: &[f32],
) -> Result<Vec<f32>> {
    let m_total: usize = segs.iter().map(|s| s.m as usize).sum();
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
    grouped_expert_gemm_cpu(gate_ty, wg, e, ff, &gu_segs, xin, &mut g)?;
    grouped_expert_gemm_cpu(up_ty, wu, e, ff, &gu_segs, xin, &mut u)?;
    let mut act = vec![0.0f32; m_total * ff];
    for i in 0..m_total * ff {
        act[i] = g[i] / (1.0 + (-g[i]).exp()) * u[i];
    }
    let mut h = vec![0.0f32; m_total * e];
    grouped_expert_gemm_cpu(down_ty, wd, ff, e, &d_segs, &act, &mut h)?;
    Ok(h)
}

/// 「h の行 → 元のトークン」の対応表。scatter はこの表だけを根拠にする。
///
/// # なぜ表を持つか
///
/// `h` の行は segment の並び (m 降順) に詰まっており、`out` の行はトークンの
/// 並びである。行数は `Σ m = nt × k` と `nt` で違うので、長さの大小で対応を
/// 検査することはできない。build_grouped_batch が gather した順
/// (`by_expert[expert]` の順) をここで 1 度だけ表に落とし、scatter 側は
/// 並び順を仮定しない。
#[derive(Debug, Clone)]
pub(crate) struct ScatterPlan {
    /// h の行 → (トークン, ルータのスロット)。長さ = Σ m。
    rows: Vec<(usize, usize)>,
    /// 足し込みの順 (h の行番号)。expert 昇順 → segment 内の行順で、
    /// `moe_experts_flat` の scatter と同じ順序。
    order: Vec<usize>,
}

impl ScatterPlan {
    /// segment 表と `by_expert` から対応表を作る。
    ///
    /// 契約: `by_expert[expert]` は segment の行順と同じ (build_grouped_batch が
    /// この順で gather する)。件数と index をここで突き合わせる。
    pub(crate) fn new(
        segs: &[Segment],
        by_expert: &[Vec<(usize, usize)>],
        nt: usize,
    ) -> Result<Self> {
        let m_total: usize = segs.iter().map(|s| s.m as usize).sum();
        let mut rows: Vec<(usize, usize)> = Vec::with_capacity(m_total);
        let mut heads: Vec<(usize, usize, u16)> = Vec::with_capacity(segs.len());
        let mut used = vec![false; by_expert.len()];
        let mut r0 = 0usize;
        for (i, s) in segs.iter().enumerate() {
            let ei = s.expert as usize;
            if ei >= by_expert.len() {
                bail!(
                    "scatter_plan: segment {i} の expert {ei} が by_expert({}) の範囲外",
                    by_expert.len()
                );
            }
            if used[ei] {
                bail!("scatter_plan: expert {ei} の segment が 2 本ある");
            }
            used[ei] = true;
            let picks = &by_expert[ei];
            if picks.len() != s.m as usize {
                bail!(
                    "scatter_plan: expert {ei} の picks {} が segment の m {} と違う",
                    picks.len(),
                    s.m
                );
            }
            for &(t, j) in picks {
                if t >= nt {
                    bail!("scatter_plan: expert {ei} のトークン {t} が範囲外 (nt={nt})");
                }
                rows.push((t, j));
            }
            heads.push((r0, s.m as usize, s.expert));
            r0 += s.m as usize;
        }
        if r0 != m_total || rows.len() != m_total {
            bail!("scatter_plan: 行数 {} が Σ m({m_total}) と違う", rows.len());
        }
        // expert 昇順 (同一 expert の segment は 1 本だけなので同順位は無い)。
        heads.sort_by_key(|&(_, _, expert)| expert);
        let mut order = Vec::with_capacity(m_total);
        for (start, m, _) in heads {
            order.extend(start..start + m);
        }
        if order.len() != m_total {
            bail!(
                "scatter_plan: 足し込み順の長さ {} が Σ m({m_total}) と違う",
                order.len()
            );
        }
        Ok(Self { rows, order })
    }

    /// 行数 (= Σ m)。
    pub(crate) fn n_rows(&self) -> usize {
        self.rows.len()
    }

    /// h の行 → (トークン, スロット)。
    pub(crate) fn rows(&self) -> &[(usize, usize)] {
        &self.rows
    }

    /// 足し込み順 (h の行番号)。
    pub(crate) fn order(&self) -> &[usize] {
        &self.order
    }

    /// `h` (segment の並び) を `out` (トークンの並び) へ重み付きで足し込む。
    pub(crate) fn scatter(
        &self,
        out: &mut [f32],
        h: &[f32],
        weights: &[f32],
        k: usize,
        e: usize,
    ) -> Result<()> {
        let nt = if e == 0 { 0 } else { out.len() / e };
        check_scatter_shapes(out.len(), h.len(), self.rows.len(), weights.len(), nt, k, e)?;
        for (i, &(_, j)) in self.rows.iter().enumerate() {
            if j >= k {
                bail!("scatter: 行 {i} のスロット {j} が k({k}) を超える");
            }
        }
        for &row in &self.order {
            let (t, j) = self.rows[row];
            let w = weights[t * k + j];
            let src = &h[row * e..(row + 1) * e];
            let dst = &mut out[t * e..(t + 1) * e];
            for d in 0..e {
                dst[d] += src[d] * w;
            }
        }
        Ok(())
    }
}

/// scatter の形の検算 (GPU 無しで回る純関数)。
///
/// `h` は segment の並び (`Σ m × n_embd`)、`out` はトークンの並び
/// (`nt × n_embd`)。`Σ m = nt × k` なので `h` は `out` の k 倍になる
/// (706 トークン / top-10 / n_embd 2560 なら h=18,073,600 / out=1,807,360)。
/// 見るのは長さの大小ではなく、定義どおりの要素数かどうかだけ。
pub(crate) fn check_scatter_shapes(
    out_len: usize,
    h_len: usize,
    n_rows: usize,
    weights_len: usize,
    nt: usize,
    k: usize,
    e: usize,
) -> Result<()> {
    if e == 0 {
        bail!("scatter: e が 0");
    }
    if nt == 0 {
        bail!("scatter: nt が 0");
    }
    if nt.checked_mul(e) != Some(out_len) {
        bail!("scatter: out の長さ {out_len} が nt({nt}) * e({e}) と違う");
    }
    if n_rows.checked_mul(e) != Some(h_len) {
        bail!("scatter: h の長さ {h_len} が Σ m({n_rows}) * e({e}) と違う");
    }
    if nt.checked_mul(k) != Some(weights_len) {
        bail!("scatter: weights の長さ {weights_len} が nt({nt}) * k({k}) と違う");
    }
    Ok(())
}

/// 前回のパック済みバッファ (1 本だけ持ち回る)。
///
/// **層 id も鍵に入れる**。P50 以前は expert 集合 (`ids`) だけで比較していたため、
/// expert 集合が同じ層 (実形ではほぼ全層) で前の層の重みを持ち回り、2 層目以降が
/// 1 層目の expert 重みで計算されていた (実機で `[PAD…]` の連続になった原因の 1 つ)。
struct PackCache {
    layer: u64,
    ids: Vec<u16>,
    bytes: Vec<u8>,
}

static PACK_CACHE: std::sync::Mutex<Option<PackCache>> = std::sync::Mutex::new(None);

/// パック済みバッファを借りて `f` を呼び、返す。
fn with_packed<R>(
    layer: u64,
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
        layer: u64::MAX,
        ids: Vec::new(),
        bytes: Vec::new(),
    });
    if cache.bytes.len() != plan.packed_bytes {
        cache.bytes.resize(plan.packed_bytes, 0);
        cache.ids.clear();
        cache.layer = u64::MAX;
    }
    if cache.layer != layer || cache.ids != plan.expert_ids {
        pack_from_sources(plan, sources, &mut cache.bytes)?;
        cache.ids = plan.expert_ids.clone();
        cache.layer = layer;
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
    let table_gu = seg_table(&batch.segs, &plan, 0, &e_off, &ff_off)?;
    // down は 1 expert あたりのバイト数が gate と違う (Q4_K 640×2560 と
    // Q5_1 2560×640)。w_off を gate のストライドで作ると 2 体目以降が別の
    // expert の重みを読むので、down 用の表は strides[2] で作り直す (P50)。
    let table_dn = seg_table(&batch.segs, &plan, 2, &e_off, &ff_off)?;
    // 表と配車の突き合わせ (GPU 無しで回る・毎層)。取り違えはここで落とす。
    check_seg_addresses(&plan, &batch.segs, &table_gu, &table_dn)?;
    let stamp = pack_stamp(layer_key, &plan.expert_ids);
    // 行 → (トークン, スロット) の対応はここで 1 度だけ作る。
    let scatter = ScatterPlan::new(&batch.segs, by_expert, nt)?;

    let ran = with_packed(layer_key, &plan, [&gate, &up, &down], |packed| -> Result<bool> {
        let mut h = vec![0.0f32; m_total * e];
        let ok = launch(
            stamp,
            packed,
            &table_gu,
            &table_dn,
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
        // P50: 最初の 1 層だけ、**元テンソル**から作った CPU 参照と比べる
        // (pack を経由しないので、層の取り違え・w_off の取り違えも値に出る)。
        if check_first_layer() {
            let cpu = reference_h_from_sources(&gate, &up, &down, &batch.segs, &batch.xs)?;
            report_check(layer_key, &h, &cpu, e)?;
        }
        scatter.scatter(&mut *out, &h, weights, k, e)?;
        Ok(true)
    })??;
    // P60: 実際に VRAM に残っているパックを在庫として登録する。
    //
    // `with_packed` が `true` を返した = パック済みバッファがデバイスに載って
    // カーネルが走った、という意味なので、そのときだけ登録する (走らなかった
    // 回に登録すると在庫を過大に報告する)。バッファは次の層まで残るので、
    // 登録はそのまま「いま VRAM にある」を表す。
    //
    // `guarded` は「この VRAM がすでに expert の予算 (`pinned`) に計上済みか」。
    // 計上済みなら `budget_usage` 側で二重に足さない。
    if ran {
        if let Ok(layer) = u16::try_from(layer_key) {
            crate::model::expert_vram::note_staging(
                stamp,
                layer,
                &plan.expert_ids,
                plan.packed_bytes,
                crate::model::expert_vram::accounting_active(),
            );
        }
    }
    Ok(ran)
}

/// `DENNO_MOE_GROUPED_CHECK=1` で、プロセス内の最初の 1 層だけ true。
fn check_first_layer() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static USED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let on = *ON.get_or_init(|| {
        matches!(
            std::env::var("DENNO_MOE_GROUPED_CHECK").ok().as_deref(),
            Some("1") | Some("on") | Some("true")
        )
    });
    on && !USED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

/// CHECK の比較結果を 1 行で出す。
///
/// GPU は `fmaf` + `expf` で計算する (ビット一致契約の外) ので、ビット比較は
/// 「最初にビットが違った位置」の報告まで。判定は相対 1e-4 を超えた最初の
/// (行, 列) で行う。行は segment の並び (詰め直した入力の行)、列は n_embd。
fn report_check(layer: u64, h: &[f32], cpu: &[f32], e: usize) -> Result<()> {
    if h.len() != cpu.len() {
        bail!(
            "CHECK: GPU の h が {} 要素、CPU 参照が {} 要素で違う",
            h.len(),
            cpu.len()
        );
    }
    if e == 0 {
        bail!("CHECK: e が 0");
    }
    let mut first_bit: Option<usize> = None;
    let mut first_over: Option<usize> = None;
    let mut max_abs = 0.0f32;
    let mut am_g = 0usize;
    let mut am_c = 0usize;
    for i in 0..h.len() {
        if first_bit.is_none() && h[i].to_bits() != cpu[i].to_bits() {
            first_bit = Some(i);
        }
        let d = (h[i] - cpu[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
        let tol = 1e-4 * cpu[i].abs().max(1.0);
        if first_over.is_none() && d > tol {
            first_over = Some(i);
        }
        if h[i] > h[am_g] {
            am_g = i;
        }
        if cpu[i] > cpu[am_c] {
            am_c = i;
        }
    }
    let at = |i: Option<usize>| match i {
        Some(i) => format!("row={} col={} gpu={} cpu={}", i / e, i % e, h[i], cpu[i]),
        None => "なし".to_string(),
    };
    eprintln!(
        "[moe-grouped] CHECK 層 {layer}: rows={} max_abs={max_abs:.6e} 許容超え={} (最初のビット差 {}) argmax_eq={}",
        h.len() / e,
        at(first_over),
        at(first_bit),
        am_g == am_c
    );
    Ok(())
}

/// GPU 起動 (cuda feature 無しは常に false)。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn launch(
    stamp: u64,
    packed: &[u8],
    table_gu: &[u32],
    table_dn: &[u32],
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
    crate::cuda::moe_grouped_gemm_layer(stamp, packed, table_gu, table_dn, g, u, d, xin, h)
}

/// cuda feature 無しのスタブ。
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn launch(
    _stamp: u64,
    _packed: &[u8],
    _table_gu: &[u32],
    _table_dn: &[u32],
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
        source_seed(ty, n_in, n_out, 7)
    }

    /// 量子値を seed でずらしたソース (層ごとに重みが違うことの検証用)。
    fn source_seed(ty: GgmlType, n_in: usize, n_out: usize, seed: u64) -> ExpertSource<'static> {
        let blocks = make_blocks(ty, n_in, n_out, N_EXPERT, seed);
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
            let mut wv: Vec<&[u8]> = vec![&[]; gate.n_expert];
            wv[ei] = g;
            let mut act = vec![0.0f32; m * ff];
            let mut upo = vec![0.0f32; m * ff];
            grouped_expert_gemm_cpu(gate.ty, &wv, e, ff, &seg, &xin, &mut act).expect("gate");
            let mut wv: Vec<&[u8]> = vec![&[]; gate.n_expert];
            wv[ei] = u;
            grouped_expert_gemm_cpu(up.ty, &wv, e, ff, &seg, &xin, &mut upo).expect("up");
            for i in 0..m * ff {
                act[i] = act[i] / (1.0 + (-act[i]).exp()) * upo[i];
            }
            let mut h = vec![0.0f32; m * e];
            let mut wv: Vec<&[u8]> = vec![&[]; gate.n_expert];
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
    fn seg_table_w_off_follows_expert_id_and_pass_stride() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (picks, xs, _w, e, ff) = fixture();
        let gate = source(GgmlType::Q4K, N_IN, N_OUT);
        let up = source(GgmlType::Q4K, N_IN, N_OUT);
        let down = source(GgmlType::Q5_1, N_OUT, N_IN);
        let batch = build_grouped_batch(&picks, &xs, e, ff).expect("batch");
        let layout = ExpertLayout::new(
            gate.shape().expect("g"),
            up.shape().expect("u"),
            down.shape().expect("d"),
        )
        .expect("layout");
        let plan = plan_grouped(&batch.segs, &layout).expect("plan");
        // 前提: segment は m 降順 (expert 1, 3, 0)、expert_ids は昇順 (0, 1, 3)。
        // つまり segment の添字と expert_ids の添字はずれる。
        assert_eq!(
            batch.segs.iter().map(|s| s.expert).collect::<Vec<_>>(),
            vec![1, 3, 0]
        );
        assert_eq!(plan.expert_ids, vec![0, 1, 3]);
        let strides = plan.strides().expect("strides");
        assert_ne!(strides[0], strides[2], "gate と down の stride は違う形");
        let (xo, ff_o) = offsets_for_pass(&batch.segs, e, ff);
        let t_gu = seg_table(&batch.segs, &plan, 0, &xo, &ff_o).expect("gu");
        let t_dn = seg_table(&batch.segs, &plan, 2, &xo, &ff_o).expect("dn");
        for (i, s) in batch.segs.iter().enumerate() {
            let k = plan.expert_ids.binary_search(&s.expert).expect("配車にある");
            assert_eq!(
                t_gu[i * 5 + 2] as usize,
                k * strides[0],
                "segment {i} (expert {}) の gate w_off",
                s.expert
            );
            assert_eq!(
                t_dn[i * 5 + 2] as usize,
                k * strides[2],
                "segment {i} (expert {}) の down w_off",
                s.expert
            );
        }
        check_seg_addresses(&plan, &batch.segs, &t_gu, &t_dn).expect("配車と一致するはず");
        // P50 以前の引き方 (segment index で plan.w_off を引く) で作った表は落ちる。
        let bad_gu: Vec<u32> = batch
            .segs
            .iter()
            .enumerate()
            .flat_map(|(i, s)| [s.expert as u32, s.m, plan.w_off[i], xo[i], ff_o[i]])
            .collect();
        let err = check_seg_addresses(&plan, &batch.segs, &bad_gu, &t_dn)
            .expect_err("segment index で引いた表は別 expert を指す");
        assert!(
            format!("{err:#}").contains("segment 0"),
            "最初のずれを報告する: {err:#}"
        );
        // down 用の表を gate のストライドで作っても落ちる。
        let bad_dn = seg_table(&batch.segs, &plan, 0, &xo, &ff_o).expect("gu で作る");
        assert!(check_seg_addresses(&plan, &batch.segs, &t_gu, &bad_dn).is_err());
    }

    /// 実形の寸法 (n_embd 2560 / n_ff 640 / Q4_K gate・up / Q5_1 down) で
    /// grouped と per-expert が **ビット一致**すること。
    ///
    /// 1 expert あたり gate 921,600 B / down 1,228,800 B でストライドが違う。
    /// P50 の実機バグ (w_off を segment の並びで引く / down に gate の stride を
    /// 使う) は、どちらもこの寸法で別 expert の重みを読む。
    #[test]
    fn real_dims_grouped_matches_per_expert_bitwise() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const NE: usize = 2;
        let (e, ff) = (E_REAL, 640); // 実形: n_embd 2560 / n_ff_exp 640
        let gate = ExpertSource {
            blocks: Box::leak(make_blocks(GgmlType::Q4K, e, ff, NE, 21).into_boxed_slice()),
            ty: GgmlType::Q4K,
            n_in: e,
            n_out: ff,
            n_expert: NE,
        };
        let up = ExpertSource {
            blocks: Box::leak(make_blocks(GgmlType::Q4K, e, ff, NE, 22).into_boxed_slice()),
            ty: GgmlType::Q4K,
            n_in: e,
            n_out: ff,
            n_expert: NE,
        };
        let down = ExpertSource {
            blocks: Box::leak(make_blocks(GgmlType::Q5_1, ff, e, NE, 23).into_boxed_slice()),
            ty: GgmlType::Q5_1,
            n_in: ff,
            n_out: e,
            n_expert: NE,
        };
        // nt=3 / k=2。expert 1 が 4 本、expert 0 が 2 本 → segment は expert 1 が先頭。
        let picks: Vec<Vec<(usize, usize)>> = vec![
            vec![(0, 0), (1, 0)],
            vec![(0, 1), (1, 1), (2, 0), (2, 1)],
        ];
        let nt = 3usize;
        let k = 2usize;
        let xs: Vec<f32> = (0..nt * e)
            .map(|i| ((i % 23) as f32 - 11.0) * 0.03125)
            .collect();
        let weights: Vec<f32> = vec![0.5, 1.0, 0.25, 0.75, 0.125, 0.625];
        let batch = build_grouped_batch(&picks, &xs, e, ff).expect("batch");
        assert_eq!(
            batch.segs.iter().map(|s| s.expert).collect::<Vec<_>>(),
            vec![1, 0],
            "m 降順で expert 1 が先頭"
        );
        let layout = ExpertLayout::new(
            gate.shape().expect("g"),
            up.shape().expect("u"),
            down.shape().expect("d"),
        )
        .expect("layout");
        let plan = plan_grouped(&batch.segs, &layout).expect("plan");
        let strides = plan.strides().expect("strides");
        assert_eq!(strides[0], 640 * 1440, "gate 1 expert = 921,600 B");
        assert_eq!(strides[2], 2560 * 480, "down 1 expert = 1,228,800 B");
        let mut packed = vec![0u8; plan.packed_bytes];
        pack_from_sources(&plan, [&gate, &up, &down], &mut packed).expect("pack");
        let h = reference_h(&gate, &up, &down, &plan, &packed, &batch.segs, &batch.xs)
            .expect("reference_h");
        let sp = ScatterPlan::new(&batch.segs, &picks, nt).expect("scatter_plan");
        let mut got = vec![0.0f32; nt * e];
        sp.scatter(&mut got, &h, &weights, k, e).expect("scatter");
        let want = per_expert_reference(&gate, &up, &down, &picks, &weights, &xs, nt, e, ff);
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
        // segment 表: segment 0 は expert 1 (expert_ids の位置 1)。
        let (xo, ff_o) = offsets_for_pass(&batch.segs, e, ff);
        let t_gu = seg_table(&batch.segs, &plan, 0, &xo, &ff_o).expect("gu");
        let t_dn = seg_table(&batch.segs, &plan, 2, &xo, &ff_o).expect("dn");
        assert_eq!(t_gu[2] as usize, strides[0], "segment 0 (expert 1) の gate");
        assert_eq!(t_dn[2] as usize, strides[2], "segment 0 (expert 1) の down");
        check_seg_addresses(&plan, &batch.segs, &t_gu, &t_dn).expect("配車と一致");
    }

    #[test]
    fn reference_from_sources_matches_packed_reference() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gate = source(GgmlType::Q4K, N_IN, N_OUT);
        let up = source(GgmlType::Q4K, N_IN, N_OUT);
        let down = source(GgmlType::Q5_1, N_OUT, N_IN);
        let (packed, plan, segs, xin) = pack(&gate, &up, &down).expect("pack");
        let from_pack = reference_h(&gate, &up, &down, &plan, &packed, &segs, &xin)
            .expect("packed 参照");
        let from_src = reference_h_from_sources(&gate, &up, &down, &segs, &xin)
            .expect("元テンソル参照");
        assert_eq!(from_pack.len(), from_src.len());
        for i in 0..from_pack.len() {
            assert_eq!(
                from_pack[i].to_bits(),
                from_src[i].to_bits(),
                "要素 {i}: pack 経由 {} vs 元テンソル {}",
                from_pack[i],
                from_src[i]
            );
        }
    }

    #[test]
    fn pack_cache_is_keyed_by_layer() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let gate_a = source_seed(GgmlType::Q4K, N_IN, N_OUT, 7);
        let up_a = source_seed(GgmlType::Q4K, N_IN, N_OUT, 8);
        let down_a = source_seed(GgmlType::Q5_1, N_OUT, N_IN, 9);
        let gate_b = source_seed(GgmlType::Q4K, N_IN, N_OUT, 10);
        let up_b = source_seed(GgmlType::Q4K, N_IN, N_OUT, 11);
        let down_b = source_seed(GgmlType::Q5_1, N_OUT, N_IN, 12);
        let (picks, xs, _w, e, ff) = fixture();
        let batch = build_grouped_batch(&picks, &xs, e, ff).expect("batch");
        let layout = ExpertLayout::new(
            gate_a.shape().expect("g"),
            up_a.shape().expect("u"),
            down_a.shape().expect("d"),
        )
        .expect("layout");
        let plan = plan_grouped(&batch.segs, &layout).expect("plan");
        let mut a = Vec::new();
        with_packed(1, &plan, [&gate_a, &up_a, &down_a], |p| -> Result<()> {
            a = p.to_vec();
            Ok(())
        })
        .expect("層 1");
        let mut b = Vec::new();
        with_packed(2, &plan, [&gate_b, &up_b, &down_b], |p| -> Result<()> {
            b = p.to_vec();
            Ok(())
        })
        .expect("層 2");
        // expert 集合は同じでも、層が変われば重みは別人。パックし直すこと。
        assert_eq!(plan.expert_ids, vec![0, 1, 3]);
        assert_ne!(a, b, "層が変わったらパックし直す (前の層の重みを使い回さない)");
        assert_eq!(a.len(), b.len());
        let mut c = Vec::new();
        with_packed(2, &plan, [&gate_b, &up_b, &down_b], |p| -> Result<()> {
            c = p.to_vec();
            Ok(())
        })
        .expect("層 2 の 2 回目");
        assert_eq!(b, c, "同じ層の 2 回目は同じパック");
    }

    #[test]
    fn seg_table_offsets_are_cumulative() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (picks, xs, _w, e, ff) = fixture();
        let batch = build_grouped_batch(&picks, &xs, e, ff).expect("batch");
        let gate = source(GgmlType::Q4K, N_IN, N_OUT);
        let up = source(GgmlType::Q4K, N_IN, N_OUT);
        let down = source(GgmlType::Q5_1, N_OUT, N_IN);
        let layout = ExpertLayout::new(
            gate.shape().expect("g"),
            up.shape().expect("u"),
            down.shape().expect("d"),
        )
        .expect("layout");
        let plan = plan_grouped(&batch.segs, &layout).expect("plan");
        let (xo, ff_o) = offsets_for_pass(&batch.segs, e, ff);
        let t = seg_table(&batch.segs, &plan, 0, &xo, &ff_o).expect("seg_table");
        assert_eq!(t.len(), batch.segs.len() * 5);
        let strides = plan.strides().expect("strides");
        for (i, s) in batch.segs.iter().enumerate() {
            assert_eq!(t[i * 5], s.expert as u32);
            assert_eq!(t[i * 5 + 1], s.m);
            let k = plan.expert_ids.binary_search(&s.expert).expect("配車にある");
            assert_eq!(t[i * 5 + 2] as usize, k * strides[0]);
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
        let sp = ScatterPlan::new(&segs, &picks, nt).expect("scatter_plan");
        let mut got = vec![0.0f32; nt * e];
        sp.scatter(&mut got, &h, &weights, 2, e).expect("scatter");
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

    // ------------------------------------------------------------------
    // 実形状の固定 (P42: 実機で 48 層すべてが形状エラーになった)
    //
    // 実機は RTX 5090 / 706 トークン / chunk 512 / n_embd 2560 / top-10 /
    // support 170。末尾チャンク (706 - 512 = 194 トークン) のとき
    //   h   = Σ m × n_embd = 1,940 × 2,560 = 4,966,400
    //   out = nt  × n_embd =   194 × 2,560 =   496,640
    // が出ていた 2 つの数 (比はちょうど k = 10)。
    // ------------------------------------------------------------------
    const E_REAL: usize = 2560;
    const FF_REAL: usize = 1536;
    const K_REAL: usize = 10;
    const SUPPORT_REAL: usize = 170;

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 1 トークンあたり `k` 体を `support` から重複なしで選ぶ (P37 の仮定と同じ)。
    fn uniform_picks(
        tokens: usize,
        k: usize,
        support: usize,
        seed: u64,
    ) -> Vec<Vec<(usize, usize)>> {
        let mut rng = seed;
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); support];
        let mut seen = vec![0u64; support.div_ceil(64)];
        for t in 0..tokens {
            seen.fill(0);
            let mut got = 0usize;
            while got < k {
                let e = (splitmix64(&mut rng) % support as u64) as usize;
                let (w, b) = (e / 64, 1u64 << (e % 64));
                if seen[w] & b != 0 {
                    continue;
                }
                seen[w] |= b;
                by_expert[e].push((t, got));
                got += 1;
            }
        }
        by_expert
    }

    #[test]
    fn real_shape_rows_and_lengths_are_consistent() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for &nt in &[706usize, 194] {
            let picks = uniform_picks(nt, K_REAL, SUPPORT_REAL, 0x5EED_1234_ABCD_0001);
            let xs = vec![0.0f32; nt * E_REAL];
            let batch = build_grouped_batch(&picks, &xs, E_REAL, FF_REAL).expect("batch");
            let m_total: usize = batch.segs.iter().map(|s| s.m as usize).sum();
            // 全トークンが k 体ずつ選ぶので Σ m = nt × k。
            assert_eq!(m_total, nt * K_REAL, "Σ m が nt × k と違う (nt={nt})");
            assert_eq!(batch.m_total, m_total);
            // h は segment の並び、out はトークンの並び。
            let h_len = m_total * E_REAL;
            let out_len = nt * E_REAL;
            assert_eq!(h_len, m_total * E_REAL);
            assert_eq!(out_len, nt * E_REAL);
            if nt == 194 {
                // 実機ログの 2 つの数 (4966400 / 496640 = 10)。
                assert_eq!(h_len, 4_966_400, "実機ログの h の長さと違う");
                assert_eq!(out_len, 496_640, "実機ログの out の長さと違う");
                assert_eq!(h_len / out_len, K_REAL);
            }
            let sp = ScatterPlan::new(&batch.segs, &picks, nt).expect("scatter_plan");
            assert_eq!(sp.n_rows(), m_total);
            assert_eq!(sp.rows().len(), m_total);
            check_scatter_shapes(out_len, h_len, sp.n_rows(), nt * K_REAL, nt, K_REAL, E_REAL)
                .expect("長さは定義どおりのはず");
            // 行 → トークンの対応: 全行が使われ、各トークンはちょうど k 行。
            let mut mask = vec![0u32; nt];
            for &(t, j) in sp.rows() {
                assert!(t < nt, "トークン {t} が範囲外");
                let b = 1u32 << j;
                assert_eq!(mask[t] & b, 0, "トークン {t} のスロット {j} が重複");
                mask[t] |= b;
            }
            assert!(
                mask.iter().all(|&m| m.count_ones() as usize == K_REAL),
                "各トークンが k 行ずつになっていない"
            );
            // 足し込み順は h の行の置換 (重複も抜けも無い)。
            let mut hit = vec![false; m_total];
            for &row in sp.order() {
                assert!(!hit[row], "行 {row} が 2 回足される");
                hit[row] = true;
            }
            assert!(hit.iter().all(|&b| b), "足されない行がある");
            // 旧コードは `out.len() < h.len()` を要求していた (これが実機の失敗)。
            // h は out のちょうど k 倍なので、この検査は必ず落ちる。
            assert!(out_len < h_len);
            assert_eq!(h_len, out_len * K_REAL);
        }
    }

    #[test]
    fn check_scatter_shapes_pins_the_definitions() {
        let (nt, k, e) = (194usize, 10usize, 2560usize);
        let n_rows = nt * k;
        // 定義どおりなら通る。
        check_scatter_shapes(nt * e, n_rows * e, n_rows, nt * k, nt, k, e).expect("正常系");
        // out を h と同じ長さにする (旧コードの前提) と落ちる。
        assert!(check_scatter_shapes(n_rows * e, n_rows * e, n_rows, nt * k, nt, k, e).is_err());
        // h が 1 行足りない / weights が 1 本足りない / nt が 0。
        assert!(check_scatter_shapes(nt * e, (n_rows - 1) * e, n_rows, nt * k, nt, k, e).is_err());
        assert!(check_scatter_shapes(nt * e, n_rows * e, n_rows, nt * k - 1, nt, k, e).is_err());
        assert!(check_scatter_shapes(0, 0, n_rows, nt * k, 0, k, e).is_err());
    }

    #[test]
    fn failures_are_counted_once_per_prefill() {
        // 実機では 48 層が同じ理由で落ちる。詳細は 1 行目だけ、集計は 1 行。
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _ = flush_failures(); // 他のテストの取りこぼしを消す
        let err = anyhow::anyhow!("テスト用の失敗");
        for layer in 0..48 {
            note_failure(layer, &err);
        }
        assert_eq!(flush_failures(), 48, "集計は 48 回ぶんを 1 行に畳むはず");
        assert_eq!(flush_failures(), 0, "2 回目の集計行は出さない");
    }

    #[test]
    fn scatter_plan_rejects_broken_mapping() {
        let _g = FIXTURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let segs = [
            Segment {
                expert: 1,
                m: 1,
                x_off: 0,
                out_off: 0,
            },
            Segment {
                expert: 0,
                m: 1,
                x_off: 0,
                out_off: 0,
            },
        ];
        let mut picks: Vec<Vec<(usize, usize)>> = vec![Vec::new(); 2];
        picks[0].push((0, 0));
        picks[1].push((0, 1));
        assert!(ScatterPlan::new(&segs, &picks, 1).is_ok());
        // picks の本数が segment の m と違う。
        let mut bad = picks.clone();
        bad[0].push((0, 0));
        assert!(ScatterPlan::new(&segs, &bad, 1).is_err());
        // 同じ expert の segment が 2 本。
        assert!(ScatterPlan::new(&[segs[0], segs[0]], &picks, 1).is_err());
        // トークン index が nt を超える。
        assert!(ScatterPlan::new(&segs, &picks, 0).is_err());
        // expert id が by_expert の範囲外。
        assert!(ScatterPlan::new(&segs, &picks[..1], 1).is_err());
    }
}
