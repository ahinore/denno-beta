//! grouped expert GEMM の「配車」(plan) と expert 重みのパッキング (タスク 36)。
//!
//! GPU 側 (`.cu` カーネル) はメインセッションが担当する。ここに置くのは
//! **ホスト側だけ**:
//!
//! 1. [`ExpertLayout`] — 1 層の expert 3 テンソル (gate / up / down) の形
//!    (`ne = [n_in, n_out, n_expert]` と量子型)。実ファイルの次元・型を読む。
//! 2. [`plan_grouped`] — segment 表から「使う expert の一覧」と、packed バッファ内の
//!    重み位置を決める。segment の並び (m 降順) は**変えない**。
//! 3. [`pack_expert_weights`] / [`pack_expert_weights_into`] — 使う expert の 3 テンソルを
//!    **1 本の連続バッファ**へ mmap からコピーする。レイアウトは
//!    `[gate(e0..eN)][up(e0..eN)][down(e0..eN)]` の 3 ブロック。
//!    量子ブロックは**デコードしない** (GPU 側がデコードする)。`_into` は宛先を
//!    呼び出し側が持ち、層をまたいで使い回せる (ページフォルトの再払いを避ける)。
//! 4. [`grouped_expert_gemm_gpu_ready`] — 起動前の前提条件の検査。壊れていたら false。
//!
//! # 数値
//!
//! ここはバイト列を動かすだけで f32 の演算をしない。したがって packed バッファの
//! スライスを [`crate::model::moe_expert::grouped_expert_gemm_cpu`] に渡した結果は、
//! 元テンソルの該当スライスを渡した結果と **バイト同一 = ビット一致**する
//! (テスト `packed_slices_feed_cpu_gemm_bitwise` が固定する)。
//!
//! # レイアウトの根拠
//!
//! expert テンソルは GGUF 上で expert が最外 (`off = e * n_out * row_bytes`) なので、
//! expert 1 体ぶんのバイト列は連続している。コピーは [`ExpertSlice`] の範囲を
//! そのまま `copy_from_slice` するだけで、ブロックの並べ替えも端数の扱いも無い。

use std::ops::Range;
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::moe_expert::{row_bytes, ExpertSlice, Segment};

/// expert テンソルの種類 (1 層に 3 本)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTensorKind {
    /// `blk.N.ffn_gate_exps.weight` — `[n_embd, n_ff_exp, n_expert]`
    Gate,
    /// `blk.N.ffn_up_exps.weight` — `[n_embd, n_ff_exp, n_expert]`
    Up,
    /// `blk.N.ffn_down_exps.weight` — `[n_ff_exp, n_embd, n_expert]`
    Down,
}

impl ExpertTensorKind {
    /// `blk.{layer}.` を除いたテンソル名。
    pub fn suffix(self) -> &'static str {
        match self {
            ExpertTensorKind::Gate => "ffn_gate_exps.weight",
            ExpertTensorKind::Up => "ffn_up_exps.weight",
            ExpertTensorKind::Down => "ffn_down_exps.weight",
        }
    }

    /// GGUF 上のフルネーム。
    pub fn name(self, layer: usize) -> String {
        format!("blk.{layer}.{}", self.suffix())
    }

    /// packed バッファ内のブロック番号 (gate=0 / up=1 / down=2)。
    pub fn block(self) -> usize {
        match self {
            ExpertTensorKind::Gate => 0,
            ExpertTensorKind::Up => 1,
            ExpertTensorKind::Down => 2,
        }
    }
}

/// 1 本の expert テンソルの形 (`ne = [n_in, n_out, n_expert]`)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertTensorShape {
    pub ty: GgmlType,
    /// 行長 (最内次元)。量子型のブロック長で割り切れること。
    pub n_in: usize,
    /// 行数。
    pub n_out: usize,
    /// expert 数。
    pub n_expert: usize,
}

impl ExpertTensorShape {
    /// 形を検証して作る。`n_in` が量子型のブロック長で割り切れない形は拒否する
    /// (`row_bytes` が門番)。
    pub fn new(ty: GgmlType, n_in: usize, n_out: usize, n_expert: usize) -> Result<Self> {
        if n_in == 0 || n_out == 0 || n_expert == 0 {
            bail!(
                "expert テンソルの形に 0 がある (n_in={n_in}, n_out={n_out}, n_expert={n_expert})"
            );
        }
        row_bytes(ty, n_in)?;
        Ok(Self {
            ty,
            n_in,
            n_out,
            n_expert,
        })
    }

    /// GGUF の実テンソルから形を読む。次元は 3 次元 (`ne[0], ne[1], ne[2]`) を要求する。
    pub fn from_gguf(gguf: &GgufFile, layer: usize, kind: ExpertTensorKind) -> Result<Self> {
        let name = kind.name(layer);
        let t = gguf
            .tensor(&name)
            .ok_or_else(|| anyhow::anyhow!("テンソル {name} が無い"))?;
        // 末尾の 1 は落とす (qwen4exp の目録検証と同じ規則)。
        let mut d: Vec<u64> = t.dims.to_vec();
        while d.len() > 1 && d.last() == Some(&1) {
            d.pop();
        }
        if d.len() != 3 {
            bail!(
                "{name}: 次元 {:?} が 3 次元でない (ne = [n_in, n_out, n_expert] を期待)",
                d
            );
        }
        let to_usize = |v: u64, what: &str| -> Result<usize> {
            usize::try_from(v).map_err(|_| anyhow::anyhow!("{name}: {what}={v} が usize を超える"))
        };
        Self::new(
            t.ty,
            to_usize(d[0], "n_in")?,
            to_usize(d[1], "n_out")?,
            to_usize(d[2], "n_expert")?,
        )
    }

    /// 1 行あたりのバイト数 (量子ブロックをデコードしない生の長さ)。
    pub fn row_bytes(&self) -> Result<usize> {
        row_bytes(self.ty, self.n_in)
    }

    /// 1 expert あたりのバイト数 (`n_out * row_bytes`)。
    pub fn expert_bytes(&self) -> Result<usize> {
        self.n_out.checked_mul(self.row_bytes()?).ok_or_else(|| {
            anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={})", self.n_out)
        })
    }
}

/// 1 層ぶんの expert 3 テンソルの形。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertLayout {
    pub gate: ExpertTensorShape,
    pub up: ExpertTensorShape,
    pub down: ExpertTensorShape,
}

impl ExpertLayout {
    /// 3 本の整合 (expert 数が同じ・gate と up が同形・down が gate の転置) を検証する。
    pub fn new(
        gate: ExpertTensorShape,
        up: ExpertTensorShape,
        down: ExpertTensorShape,
    ) -> Result<Self> {
        if gate.n_expert != up.n_expert || gate.n_expert != down.n_expert {
            bail!(
                "expert 数が 3 本で違う (gate={}, up={}, down={})",
                gate.n_expert,
                up.n_expert,
                down.n_expert
            );
        }
        if gate.n_in != up.n_in || gate.n_out != up.n_out {
            bail!(
                "gate と up の形が違う (gate=[{},{}], up=[{},{}])",
                gate.n_in,
                gate.n_out,
                up.n_in,
                up.n_out
            );
        }
        if down.n_in != gate.n_out || down.n_out != gate.n_in {
            bail!(
                "down が gate の転置になっていない (gate=[{},{}], down=[{},{}])",
                gate.n_in,
                gate.n_out,
                down.n_in,
                down.n_out
            );
        }
        Ok(Self { gate, up, down })
    }

    /// 1 層ぶんの実テンソルから読む。
    pub fn from_gguf(gguf: &GgufFile, layer: usize) -> Result<Self> {
        Self::new(
            ExpertTensorShape::from_gguf(gguf, layer, ExpertTensorKind::Gate)?,
            ExpertTensorShape::from_gguf(gguf, layer, ExpertTensorKind::Up)?,
            ExpertTensorShape::from_gguf(gguf, layer, ExpertTensorKind::Down)?,
        )
    }

    /// 種類ごとの形。
    pub fn shape(&self, kind: ExpertTensorKind) -> &ExpertTensorShape {
        match kind {
            ExpertTensorKind::Gate => &self.gate,
            ExpertTensorKind::Up => &self.up,
            ExpertTensorKind::Down => &self.down,
        }
    }

    /// expert 数 (3 本で共通)。
    pub fn n_expert(&self) -> usize {
        self.gate.n_expert
    }

    /// 種類ごとの「1 expert あたりのバイト数」。
    pub fn stride(&self, kind: ExpertTensorKind) -> Result<usize> {
        self.shape(kind).expert_bytes()
    }

    /// `[gate, up, down]` の 1 expert あたりのバイト数。
    pub fn strides(&self) -> Result<[usize; 3]> {
        Ok([
            self.gate.expert_bytes()?,
            self.up.expert_bytes()?,
            self.down.expert_bytes()?,
        ])
    }
}

/// 1 層ぶんの配車。segment 表 (m 降順のまま) と、packed バッファの重み位置を持つ。
///
/// packed バッファのレイアウト:
///
/// ```text
/// [ gate: expert_ids[0] | expert_ids[1] | ... | expert_ids[N-1] ]
/// [ up  : expert_ids[0] | expert_ids[1] | ... | expert_ids[N-1] ]
/// [ down: expert_ids[0] | expert_ids[1] | ... | expert_ids[N-1] ]
/// ```
///
/// `w_off[i]` は **gate ブロック**内の `expert_ids[i]` の先頭バイト。
/// up / down の先頭は `block_off[1] + i * stride[1]` / `block_off[2] + i * stride[2]`
/// ([`GroupedPlan::expert_slices`] が 3 本まとめて返す)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupedPlan {
    /// `build_grouped_batch` が返した segment 表 (m 降順のまま)。
    pub segs: Vec<Segment>,
    /// 重複なし・昇順の expert id。
    pub expert_ids: Vec<u16>,
    /// `expert_ids[i]` の gate スライスの先頭 (packed バッファ内のバイト位置)。
    pub w_off: Vec<u32>,
    /// packed バッファの総バイト数 (3 ブロック合計)。
    pub packed_bytes: usize,
    /// 層の形 (pack と GPU のブロック位置に使う)。
    pub layout: ExpertLayout,
    /// packed バッファ内の 3 ブロックの先頭 `[gate, up, down]`。
    pub block_off: [usize; 3],
}

impl GroupedPlan {
    /// `Σ m` (= 行数)。
    pub fn m_total(&self) -> usize {
        self.segs.iter().map(|s| s.m as usize).sum()
    }

    /// packed する expert 数。
    pub fn n_expert(&self) -> usize {
        self.expert_ids.len()
    }

    /// `[gate, up, down]` の 1 expert あたりのバイト数。
    pub fn strides(&self) -> Result<[usize; 3]> {
        self.layout.strides()
    }

    /// `expert` の 3 スライス (gate / up / down) のバイト範囲。未使用なら `None`。
    pub fn expert_slices(&self, expert: u16) -> Option<[Range<usize>; 3]> {
        let i = self.expert_ids.binary_search(&expert).ok()?;
        let strides = self.strides().ok()?;
        Some([
            self.block_off[0] + i * strides[0]..self.block_off[0] + (i + 1) * strides[0],
            self.block_off[1] + i * strides[1]..self.block_off[1] + (i + 1) * strides[1],
            self.block_off[2] + i * strides[2]..self.block_off[2] + (i + 1) * strides[2],
        ])
    }

    /// `expert` の `pass` ブロック内の先頭バイト (pass = 0 gate / 1 up / 2 down)。
    ///
    /// `w_off` は **expert id で引く**。segment 表は m 降順で並ぶが packed は
    /// expert 昇順に詰まっているので、segment の添字をそのまま `w_off` の添字に
    /// してはならない (P50 の実機バグ)。
    pub fn w_off_for(&self, expert: u16, pass: usize) -> Result<u32> {
        let strides = self.strides()?;
        let stride = *strides
            .get(pass)
            .ok_or_else(|| anyhow::anyhow!("w_off_for: pass {pass} が 3 ブロックの範囲外"))?;
        let i = self.expert_ids.binary_search(&expert).map_err(|_| {
            anyhow::anyhow!("w_off_for: expert {expert} が配車 (expert_ids) に無い")
        })?;
        let off = i
            .checked_mul(stride)
            .ok_or_else(|| anyhow::anyhow!("w_off_for: {i} x {stride} が usize を溢れる"))?;
        u32::try_from(off)
            .map_err(|_| anyhow::anyhow!("w_off_for: オフセット {off} が u32 を超える"))
    }

    /// 内部整合 (expert_ids が昇順・重複なし、w_off と block_off がストライドの累積)。
    pub fn is_consistent(&self) -> bool {
        let Ok(strides) = self.strides() else {
            return false;
        };
        let n = self.expert_ids.len();
        if n != self.w_off.len() {
            return false;
        }
        if self.expert_ids.windows(2).any(|w| w[0] >= w[1]) {
            return false;
        }
        if self
            .w_off
            .iter()
            .enumerate()
            .any(|(i, &o)| o as usize != i * strides[0])
        {
            return false;
        }
        let Some(up) = n.checked_mul(strides[0]) else {
            return false;
        };
        let Some(down) = up.checked_add(n.checked_mul(strides[1]).unwrap_or(usize::MAX)) else {
            return false;
        };
        let Some(total) = down.checked_add(n.checked_mul(strides[2]).unwrap_or(usize::MAX)) else {
            return false;
        };
        self.block_off == [0, up, down] && self.packed_bytes == total
    }
}

/// segment 表から配車を作る。**segs の順序は変えない** (m 降順のまま)。
///
/// `expert_ids` は重複を除いた昇順、`w_off` は gate ブロック内のバイト位置。
/// `m == 0` の segment と範囲外の expert は `Err` (空タスクは GPU に積まない)。
///
/// # 引数が 2 つある理由
///
/// segment 表は expert id・m・要素オフセットしか持たない (1 expert あたりのバイト数を
/// 持たない) ので、`packed_bytes` / `w_off` をバイトで返すには層の形
/// ([`ExpertLayout`]) が要る。形は GGUF から読む ([`ExpertLayout::from_gguf`])。
pub fn plan_grouped(segs: &[Segment], layout: &ExpertLayout) -> Result<GroupedPlan> {
    let n_expert = layout.n_expert();
    let mut expert_ids: Vec<u16> = Vec::with_capacity(segs.len());
    for (i, s) in segs.iter().enumerate() {
        if s.m == 0 {
            bail!("segment {i}: m が 0 (空タスクは積まない)");
        }
        if s.expert as usize >= n_expert {
            bail!(
                "segment {i}: expert {} が範囲外 (n_expert={n_expert})",
                s.expert
            );
        }
        expert_ids.push(s.expert);
    }
    expert_ids.sort_unstable();
    expert_ids.dedup();

    let strides = layout.strides()?;
    let n = expert_ids.len();
    let mul = |a: usize, b: usize, what: &str| -> Result<usize> {
        a.checked_mul(b)
            .ok_or_else(|| anyhow::anyhow!("{what} が usize を溢れる ({a} x {b})"))
    };
    let add = |a: usize, b: usize, what: &str| -> Result<usize> {
        a.checked_add(b)
            .ok_or_else(|| anyhow::anyhow!("{what} が usize を溢れる ({a} + {b})"))
    };
    let gate_block = mul(n, strides[0], "gate ブロック")?;
    let up_block = mul(n, strides[1], "up ブロック")?;
    let down_block = mul(n, strides[2], "down ブロック")?;
    let packed_bytes = add(add(gate_block, up_block, "packed")?, down_block, "packed")?;

    let mut w_off = Vec::with_capacity(n);
    for i in 0..n {
        let off = mul(i, strides[0], "w_off")?;
        w_off.push(u32::try_from(off).map_err(|_| {
            anyhow::anyhow!("w_off {off} が u32 を超える (gate ブロック {gate_block} バイト)")
        })?);
    }

    Ok(GroupedPlan {
        segs: segs.to_vec(),
        expert_ids,
        w_off,
        packed_bytes,
        layout: *layout,
        block_off: [0, gate_block, add(gate_block, up_block, "block_off")?],
    })
}

/// [`pack_expert_weights`] の実測値。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackStats {
    /// コピーした総バイト数 (= `plan.packed_bytes`)。
    pub bytes: usize,
    /// コピーした expert 数。
    pub experts: usize,
    /// 3 ブロックのバイト数 `[gate, up, down]`。
    pub block_bytes: [usize; 3],
    /// 確保 + コピーの実時間。
    pub elapsed: Duration,
}

/// 使う expert の 3 テンソルを 1 本の連続バッファへコピーする。
///
/// - レイアウトは `[gate(e0..eN)][up(e0..eN)][down(e0..eN)]`。
/// - mmap のスライスから `copy_from_slice` するだけで、量子ブロックはデコードしない。
/// - 確保は 1 回 (`vec![0u8; packed_bytes]`)。確保したバッファを層をまたいで
///   使い回したいなら [`pack_expert_weights_into`] を使う (初回タッチの
///   ページフォルトが pack 時間の 8 割を占める。REPORT_P36.md)。
/// - `plan.layout` と `gguf` の `blk.{layer}` の形が違えば `Err` (別の層・別のモデルで
///   黙って詰めない)。
pub fn pack_expert_weights(
    gguf: &GgufFile,
    layer: usize,
    plan: &GroupedPlan,
) -> Result<(Vec<u8>, PackStats)> {
    let mut out: Vec<u8> = vec![0u8; plan.packed_bytes];
    let stats = pack_expert_weights_into(gguf, layer, plan, &mut out)?;
    Ok((out, stats))
}

/// [`pack_expert_weights`] の宛先を呼び出し側が持つ版。
///
/// `out.len()` は `plan.packed_bytes` と一致していること。宛先を使い回すと
/// 初回タッチのページフォルトを層ごとに払わずに済む (実測: 783 MB で
/// 298 ms → 62 ms)。書き込みは全バイトなのでゼロ初期化には依存しない。
pub fn pack_expert_weights_into(
    gguf: &GgufFile,
    layer: usize,
    plan: &GroupedPlan,
    out: &mut [u8],
) -> Result<PackStats> {
    let t0 = Instant::now();
    if out.len() != plan.packed_bytes {
        bail!(
            "宛先の長さ {} が packed_bytes {} と違う",
            out.len(),
            plan.packed_bytes
        );
    }
    let got = ExpertLayout::from_gguf(gguf, layer)?;
    if got != plan.layout {
        bail!(
            "plan の形 {:?} が blk.{layer} の実テンソル {:?} と違う",
            plan.layout,
            got
        );
    }
    if !plan.is_consistent() {
        bail!("plan が内部整合でない (expert_ids / w_off / block_off / packed_bytes)");
    }

    let kinds = [
        ExpertTensorKind::Gate,
        ExpertTensorKind::Up,
        ExpertTensorKind::Down,
    ];
    let mut src_bytes: Vec<&[u8]> = Vec::with_capacity(3);
    for kind in kinds {
        let name = kind.name(layer);
        src_bytes.push(
            gguf.tensor_bytes(&name)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        );
    }
    let strides = plan.strides()?;

    for (i, &expert) in plan.expert_ids.iter().enumerate() {
        for kind in kinds {
            let b = kind.block();
            let shape = plan.layout.shape(kind);
            let src = ExpertSlice::new(
                shape.ty,
                shape.n_in,
                shape.n_out,
                shape.n_expert,
                expert as usize,
            )?;
            if src.end() > src_bytes[b].len() {
                bail!(
                    "{}: expert {expert} の範囲 {}..{} がテンソルの {} バイトを超える",
                    kind.name(layer),
                    src.off,
                    src.end(),
                    src_bytes[b].len()
                );
            }
            let dst = plan.block_off[b] + i * strides[b];
            out[dst..dst + strides[b]].copy_from_slice(&src_bytes[b][src.off..src.end()]);
        }
    }
    debug_assert_eq!(out.len(), plan.packed_bytes);

    Ok(PackStats {
        bytes: plan.packed_bytes,
        experts: plan.expert_ids.len(),
        block_bytes: [
            plan.block_off[1] - plan.block_off[0],
            plan.block_off[2] - plan.block_off[1],
            plan.packed_bytes - plan.block_off[2],
        ],
        elapsed: t0.elapsed(),
    })
}

/// GPU 経路に渡してよいかの検査。壊れていたら `false` (呼び出し側は fail closed)。
///
/// 見るもの:
///
/// - `segs` が空でない。
/// - `segs` の `(expert, m)` の並びが `plan.segs` と**同一** (別の配車に古い plan を
///   合わせていない)。これが「全 m >= 1」と「Σm == m_total」を兼ねる。
///   `x_off` / `out_off` は幅で変わるので比較しない — down は `n_in` / `n_out` が
///   入れ替わるため `segs` を作り直す必要があり、その表は `(expert, m)` が同じで
///   offsets だけ違う。
/// - 全 segment の expert が `expert_ids` に含まれる。
/// - `x_off` / `out_off` が 0 から隙間なく連続し、区間の幅が全 segment で同じ。
///   幅は segment に書かれていないので「offset の差分 / m」で復元し、それが層の
///   実テンソルのどちらか (gate/up = `[n_in, n_out]`、down = `[n_out, n_in]`) と
///   一致することを要求する (取り違えた表を弾く)。
/// - plan の内部整合 (`expert_ids` 昇順・`w_off` と `block_off` がストライドの累積)。
pub fn grouped_expert_gemm_gpu_ready(plan: &GroupedPlan, segs: &[Segment]) -> bool {
    if segs.is_empty() || segs.len() != plan.segs.len() {
        return false;
    }
    if segs
        .iter()
        .zip(&plan.segs)
        .any(|(a, b)| a.expert != b.expert || a.m != b.m)
    {
        return false;
    }
    if !plan.is_consistent() {
        return false;
    }
    if segs.iter().any(|s| {
        s.expert as usize >= plan.layout.n_expert()
            || plan.expert_ids.binary_search(&s.expert).is_err()
    }) {
        return false;
    }

    // 1. offsets は 0 から始まる。
    if segs[0].x_off != 0 || segs[0].out_off != 0 {
        return false;
    }
    // 2. 隣り合う segment の offset 差が「m × 同じ幅」になっていること。
    let (mut wx, mut wo): (Option<usize>, Option<usize>) = (None, None);
    for w in segs.windows(2) {
        let m = w[0].m as usize;
        if m == 0 {
            return false;
        }
        let Some(dx) = (w[1].x_off as usize).checked_sub(w[0].x_off as usize) else {
            return false;
        };
        let Some(doff) = (w[1].out_off as usize).checked_sub(w[0].out_off as usize) else {
            return false;
        };
        if dx % m != 0 || doff % m != 0 {
            return false;
        }
        let (cx, co) = (dx / m, doff / m);
        if cx == 0 || co == 0 {
            return false;
        }
        if wx.map_or(false, |v| v != cx) || wo.map_or(false, |v| v != co) {
            return false;
        }
        wx = Some(cx);
        wo = Some(co);
    }
    // 3. 復元した幅がこの層の実テンソルのどちらかであること。
    //    segment が 1 本だけのときは幅を復元できない (offset は 0 で矛盾しない) ので通す。
    let g = plan.layout.gate;
    let d = plan.layout.down;
    match (wx, wo) {
        (Some(wx), Some(wo)) => (wx == g.n_in && wo == g.n_out) || (wx == d.n_in && wo == d.n_out),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::moe_expert::{build_grouped_batch, check_segments, grouped_expert_gemm_cpu};
    use std::path::{Path, PathBuf};

    const ALIGN: usize = 32;
    const GT_U32: u32 = 4;
    const TY_Q4_K: u32 = 12;
    const TY_Q5_1: u32 = 7;

    fn blck_size(ty: u32) -> u64 {
        match ty {
            TY_Q4_K => 256,
            TY_Q5_1 => 32,
            other => panic!("テストヘルパ未対応の型 {other}"),
        }
    }

    fn type_size(ty: u32) -> u64 {
        match ty {
            TY_Q4_K => 144,
            TY_Q5_1 => 24,
            other => panic!("テストヘルパ未対応の型 {other}"),
        }
    }

    fn nbytes(dims: &[u64], ty: u32) -> u64 {
        let ne: u64 = dims.iter().product();
        ne / blck_size(ty) * type_size(ty)
    }

    fn align_up(v: usize, a: usize) -> usize {
        (v + a - 1) & !(a - 1)
    }

    fn push_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    struct TensorSpec {
        name: String,
        dims: Vec<u64>,
        ty: u32,
        seed: u32,
    }

    fn f16b(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// 量子ブロックを「まともなスケール」で埋める (乱数だと f16 の指数が跳んで
    /// NaN / 1e38 になり、値の比較が意味を持たない)。expert ごとに値が変わる。
    fn fill_quant(data: &mut [u8], ty: u32, blocks_per_expert: usize, seed: u32) {
        let ts = type_size(ty) as usize;
        let bpe = blocks_per_expert.max(1) as u32;
        for (bi, blk) in data.chunks_mut(ts).enumerate() {
            let e = (bi as u32) / bpe;
            let r = (bi as u32) % bpe;
            match ty {
                TY_Q4_K => {
                    blk[0..2].copy_from_slice(&f16b(0.02 + 0.001 * ((r + e + seed) % 7) as f32));
                    blk[2..4].copy_from_slice(&f16b(0.0005));
                    for i in 0..12u32 {
                        blk[4 + i as usize] = (0x11 + ((r * 3 + i + e) % 6)) as u8;
                    }
                    for i in 0..128u32 {
                        blk[16 + i as usize] =
                            ((i * 7 + r * 13 + e * 5 + seed) % 251) as u8;
                    }
                }
                TY_Q5_1 => {
                    blk[0..2].copy_from_slice(&f16b(0.02 + 0.001 * ((r + e + seed) % 5) as f32));
                    blk[2..4].copy_from_slice(&f16b(0.0004));
                    let qh = ((r * 37 + e * 7 + seed) & 0xffff) | ((r & 0xffff) << 16);
                    blk[4..8].copy_from_slice(&qh.to_le_bytes());
                    for i in 0..16u32 {
                        blk[8 + i as usize] = ((i * 13 + r * 5 + e * 11 + seed) % 251) as u8;
                    }
                }
                other => panic!("テストヘルパ未対応の型 {other}"),
            }
        }
    }

    /// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列。
    fn write_gguf(path: &Path, tensors: &[TensorSpec]) {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&1u64.to_le_bytes()); // kv 1 本 (alignment)
        push_str(&mut head, "general.alignment");
        head.extend_from_slice(&GT_U32.to_le_bytes());
        head.extend_from_slice(&(ALIGN as u32).to_le_bytes());

        let mut offsets = Vec::with_capacity(tensors.len());
        let mut cur = 0u64;
        for t in tensors {
            offsets.push(cur);
            cur = align_up((cur + nbytes(&t.dims, t.ty)) as usize, ALIGN) as u64;
        }
        for (t, off) in tensors.iter().zip(&offsets) {
            push_str(&mut head, &t.name);
            head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&t.ty.to_le_bytes());
            head.extend_from_slice(&off.to_le_bytes());
        }

        let mut out = head;
        out.resize(align_up(out.len(), ALIGN), 0);
        for t in tensors {
            let n = nbytes(&t.dims, t.ty) as usize;
            let at = out.len();
            out.resize(at + n, 0);
            // 1 expert あたりのブロック数 (nbytes / ブロック長 / expert 数)。
            let bpe = n / type_size(t.ty) as usize / t.dims[2] as usize;
            fill_quant(&mut out[at..], t.ty, bpe, t.seed);
            out.resize(align_up(out.len(), ALIGN), 0);
        }
        std::fs::write(path, out).expect("テスト GGUF の書き出し");
    }

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "denno_moe_grouped_{tag}_{}_{n}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).expect("temp dir");
            TempDir(p)
        }

        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// 層 `layer` の expert 3 テンソルを書いた GGUF。
    fn write_layer_gguf(path: &Path, layer: usize, n_embd: u64, n_ff: u64, n_expert: u64) {
        let t = |kind: ExpertTensorKind, dims: Vec<u64>, ty: u32, seed: u32| TensorSpec {
            name: kind.name(layer),
            dims,
            ty,
            seed,
        };
        write_gguf(
            path,
            &[
                t(
                    ExpertTensorKind::Gate,
                    vec![n_embd, n_ff, n_expert],
                    TY_Q4_K,
                    11,
                ),
                t(
                    ExpertTensorKind::Up,
                    vec![n_embd, n_ff, n_expert],
                    TY_Q4_K,
                    22,
                ),
                t(
                    ExpertTensorKind::Down,
                    vec![n_ff, n_embd, n_expert],
                    TY_Q5_1,
                    33,
                ),
            ],
        );
    }

    /// 実形 (n_embd=2560 / n_ff_exp=640 / Q4_K gate+up / Q5_1 down) の形だけを作る。
    fn real_layout() -> ExpertLayout {
        ExpertLayout::new(
            ExpertTensorShape::new(GgmlType::Q4K, 2560, 640, 512).unwrap(),
            ExpertTensorShape::new(GgmlType::Q4K, 2560, 640, 512).unwrap(),
            ExpertTensorShape::new(GgmlType::Q5_1, 640, 2560, 512).unwrap(),
        )
        .unwrap()
    }

    /// 与えられた m 分布の segment 表と入力を組み立てる (m 降順は build_grouped_batch が作る)。
    fn m_dist_batch(m_dist: &[usize], n_in: usize, n_out: usize) -> (Vec<Segment>, Vec<f32>) {
        let nt: usize = m_dist.iter().sum();
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); m_dist.len()];
        let mut base = 0usize;
        for (ei, &m) in m_dist.iter().enumerate() {
            for i in 0..m {
                by_expert[ei].push((base + i, 0));
            }
            base += m;
        }
        assert_eq!(base, nt);
        let xs: Vec<f32> = (0..nt * n_in)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.013)
            .collect();
        let batch = build_grouped_batch(&by_expert, &xs, n_in, n_out).unwrap();
        (batch.segs, batch.xs)
    }

    /// テスト 1: 実形の配車。packed_bytes が「expert 数 × 1 体ぶん」に一致する。
    ///
    /// 1 体ぶんは gate 921,600 + up 921,600 + down 1,228,800 = 3,072,000 B。
    /// タスク文の「約 5 MB/体」は次元 [2560,640,512] と Q4_K/Q5_1 の計算
    /// (3.07 MB/体) と合わない (REPORT_P36.md に書いた)。
    #[test]
    fn plan_real_shape_packed_bytes_match_layout() {
        let layout = real_layout();
        assert_eq!(layout.gate.expert_bytes().unwrap(), 921_600);
        assert_eq!(layout.up.expert_bytes().unwrap(), 921_600);
        assert_eq!(layout.down.expert_bytes().unwrap(), 1_228_800);
        let per_expert = 3_072_000usize;

        let dist = [492usize, 300, 100, 40, 20, 5, 3, 1];
        let (segs, _) = m_dist_batch(&dist, 2560, 640);
        let plan = plan_grouped(&segs, &layout).unwrap();

        assert_eq!(plan.n_expert(), dist.len());
        assert_eq!(plan.expert_ids, (0..dist.len() as u16).collect::<Vec<_>>());
        assert_eq!(plan.m_total(), 961);
        // segs の並びは m 降順のまま (触らない)。
        assert_eq!(plan.segs, segs);
        assert_eq!(
            plan.segs.iter().map(|s| s.m).collect::<Vec<_>>(),
            vec![492, 300, 100, 40, 20, 5, 3, 1]
        );
        assert_eq!(plan.packed_bytes, dist.len() * per_expert);
        assert_eq!(plan.block_off, [0, 8 * 921_600, 8 * 921_600 * 2]);
        assert_eq!(
            plan.w_off,
            (0..8).map(|i| (i * 921_600) as u32).collect::<Vec<_>>()
        );
        assert!(plan.is_consistent());

        // 層あたりの実数 (非空 expert 86〜255、上限 512)。
        for n in [86usize, 130, 255, 512] {
            let segs: Vec<Segment> = (0..n)
                .map(|e| Segment {
                    expert: e as u16,
                    m: 20,
                    x_off: (e * 20 * 2560) as u32,
                    out_off: (e * 20 * 640) as u32,
                })
                .collect();
            let plan = plan_grouped(&segs, &layout).unwrap();
            assert_eq!(plan.packed_bytes, n * per_expert);
            assert!(plan.is_consistent());
            // w_off は u32 に収まる (255 体で 234 MB、512 体で 472 MB)。
            assert_eq!(plan.w_off[n - 1] as usize, (n - 1) * 921_600);
        }
    }

    /// テスト 2: 512 expert を宣言したテンソルから 8 体を pack し、元スライスと
    /// バイト一致することを先頭 64 / 末尾 64 / 全体で確かめる。
    ///
    /// 次元は小さくしてある (n_embd=256 / n_ff=32) ので 1 体 = 15,360 B。
    /// 「512 expert」はテンソルの宣言数 (実ファイルと同じ) で、pack するのは
    /// plan が選んだ 8 体だけ。ファイルは 512 × 15,360 = 7.9 MB。
    #[test]
    fn pack_bytes_match_source_slices() {
        let dir = TempDir::new("pack");
        let path = dir.path("m512.gguf");
        write_layer_gguf(&path, 1, 256, 32, 512);
        let gguf = GgufFile::open(&path).unwrap();

        let layout = ExpertLayout::from_gguf(&gguf, 1).unwrap();
        assert_eq!(layout.n_expert(), 512);
        let dist = [492usize, 300, 100, 40, 20, 5, 3, 1];
        let (segs, _) = m_dist_batch(&dist, 256, 32);
        let plan = plan_grouped(&segs, &layout).unwrap();
        assert_eq!(plan.n_expert(), 8);
        assert_eq!(plan.packed_bytes, 8 * (4_608 + 4_608 + 6_144));

        let (packed, stats) = pack_expert_weights(&gguf, 1, &plan).unwrap();
        assert_eq!(stats.bytes, plan.packed_bytes);
        assert_eq!(stats.experts, 8);
        assert_eq!(stats.block_bytes, [8 * 4_608, 8 * 4_608, 8 * 6_144]);
        assert_eq!(packed.len(), plan.packed_bytes);

        let kinds = [
            ExpertTensorKind::Gate,
            ExpertTensorKind::Up,
            ExpertTensorKind::Down,
        ];
        for kind in kinds {
            let name = kind.name(1);
            let src = gguf.tensor_bytes(&name).unwrap();
            let shape = plan.layout.shape(kind);
            for (i, &expert) in plan.expert_ids.iter().enumerate() {
                let sl = ExpertSlice::new(
                    shape.ty,
                    shape.n_in,
                    shape.n_out,
                    shape.n_expert,
                    expert as usize,
                )
                .unwrap();
                let want = &src[sl.off..sl.end()];
                let r = plan.expert_slices(expert).unwrap()[kind.block()].clone();
                let got = &packed[r];
                assert_eq!(got, want, "{kind:?} expert {expert}: 全体が違う");
                let n = want.len();
                assert_eq!(got[..64], want[..64], "{kind:?} expert {expert}: 先頭 64 B");
                assert_eq!(
                    got[n - 64..],
                    want[n - 64..],
                    "{kind:?} expert {expert}: 末尾 64 B"
                );
            }
        }

        // 未使用の expert は None。
        assert!(plan.expert_slices(8).is_none());

        // 宛先を呼び出し側が持つ版でも同じバイト列 (層をまたいだ使い回し用)。
        let mut dst = vec![0u8; plan.packed_bytes];
        pack_expert_weights_into(&gguf, 1, &plan, &mut dst).unwrap();
        assert_eq!(dst, packed, "pack_expert_weights_into の出力が違う");
        // 長さの違う宛先は拒否。
        let mut short = vec![0u8; plan.packed_bytes - 1];
        assert!(pack_expert_weights_into(&gguf, 1, &plan, &mut short).is_err());
    }

    /// テスト 3: packed のスライスを CPU 参照実装に渡すと、元テンソルのスライスと
    /// ビット一致する (gate = Q4_K、down = Q5_1 の両方)。
    #[test]
    fn packed_slices_feed_cpu_gemm_bitwise() {
        let dir = TempDir::new("cpu");
        let path = dir.path("m512.gguf");
        write_layer_gguf(&path, 1, 256, 32, 512);
        let gguf = GgufFile::open(&path).unwrap();
        let layout = ExpertLayout::from_gguf(&gguf, 1).unwrap();

        let dist = [492usize, 300, 100, 40, 20, 5, 3, 1];
        let (segs, xs) = m_dist_batch(&dist, 256, 32);
        let plan = plan_grouped(&segs, &layout).unwrap();
        let (packed, _) = pack_expert_weights(&gguf, 1, &plan).unwrap();
        let m_total = plan.m_total();
        check_segments(&segs, 256, 32, xs.len(), 32 * m_total).unwrap();

        // 元テンソルのスライス (expert id で引く表)。
        let build_src = |kind: ExpertTensorKind| -> Vec<&[u8]> {
            let t = gguf.tensor_bytes(&kind.name(1)).unwrap();
            let shape = plan.layout.shape(kind);
            (0..8)
                .map(|e| {
                    let sl = ExpertSlice::new(
                        shape.ty,
                        shape.n_in,
                        shape.n_out,
                        shape.n_expert,
                        e,
                    )
                    .unwrap();
                    &t[sl.off..sl.end()]
                })
                .collect()
        };
        let build_packed = |kind: ExpertTensorKind| -> Vec<&[u8]> {
            (0..8u16)
                .map(|e| {
                    let r = plan.expert_slices(e).unwrap()[kind.block()].clone();
                    &packed[r]
                })
                .collect()
        };

        // gate: xs (m_total x 256) -> outs (m_total x 8)
        let src_gate = build_src(ExpertTensorKind::Gate);
        let packed_gate = build_packed(ExpertTensorKind::Gate);
        let mut a = vec![0f32; m_total * 32];
        let mut b = vec![0f32; m_total * 32];
        grouped_expert_gemm_cpu(GgmlType::Q4K, &src_gate, 256, 32, &segs, &xs, &mut a)
            .unwrap();
        grouped_expert_gemm_cpu(
            GgmlType::Q4K,
            &packed_gate,
            256,
            32,
            &segs,
            &xs,
            &mut b,
        )
        .unwrap();
        assert!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "gate (Q4_K) がビット一致しない"
        );
        assert!(
            a.iter().any(|v| v.is_finite() && *v != 0.0),
            "gate の出力が全部 0 (テストが空回りしている)"
        );

        // down: xs' (m_total x 32) -> outs' (m_total x 256)。
        // Segment の x_off / out_off は **要素単位** なので、幅の違う down は
        // 同じ表を使い回せない (行番号ではない)。down 用に組み直す。
        let (segs_down, xs2) = m_dist_batch(&dist, 32, 256);
        let src_down = build_src(ExpertTensorKind::Down);
        let packed_down = build_packed(ExpertTensorKind::Down);
        let mut c = vec![0f32; m_total * 256];
        let mut d = vec![0f32; m_total * 256];
        grouped_expert_gemm_cpu(GgmlType::Q5_1, &src_down, 32, 256, &segs_down, &xs2, &mut c)
            .unwrap();
        grouped_expert_gemm_cpu(
            GgmlType::Q5_1,
            &packed_down,
            32,
            256,
            &segs_down,
            &xs2,
            &mut d,
        )
        .unwrap();
        assert!(
            c.iter().zip(&d).all(|(x, y)| x.to_bits() == y.to_bits()),
            "down (Q5_1) がビット一致しない"
        );
        assert!(
            c.iter().any(|v| v.is_finite() && *v != 0.0),
            "down の出力が全部 0 (テストが空回りしている)"
        );
    }

    /// テスト 4: 前提条件の検査。壊れた表は false、正しい表は true。
    #[test]
    fn gpu_ready_checks_segments() {
        let dir = TempDir::new("ready");
        let path = dir.path("m512.gguf");
        write_layer_gguf(&path, 1, 256, 32, 512);
        let gguf = GgufFile::open(&path).unwrap();
        let layout = ExpertLayout::from_gguf(&gguf, 1).unwrap();

        let dist = [64usize, 32, 8, 4];
        let (segs, _) = m_dist_batch(&dist, 256, 32);
        let plan = plan_grouped(&segs, &layout).unwrap();
        assert!(grouped_expert_gemm_gpu_ready(&plan, &segs));
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &[]));

        // m が 0。
        let mut bad = segs.clone();
        bad[1].m = 0;
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &bad));

        // plan に無い expert。
        let mut bad = segs.clone();
        bad[0].expert = 300;
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &bad));

        // x_off の隙間。
        let mut bad = segs.clone();
        bad[2].x_off += 1;
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &bad));

        // out_off の重複。
        let mut bad = segs.clone();
        bad[3].out_off = bad[2].out_off;
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &bad));

        // 幅の違う down 用に作り直した表 ((expert, m) は同じ・offsets だけ違う) は通す。
        // gate/up と down は n_in / n_out が入れ替わるので、呼び出し側はこの形で渡す。
        let (segs_down, _) = m_dist_batch(&dist, 32, 256);
        assert_ne!(segs_down[1].x_off, segs[1].x_off);
        assert_ne!(segs_down[1].out_off, segs[1].out_off);
        assert!(grouped_expert_gemm_gpu_ready(&plan, &segs_down));

        // 同じ expert でも m が違う表は拒否 (配車そのものが変わっている)。
        let mut bad = segs_down.clone();
        bad[0].m += 1;
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &bad));

        // expert の並びが違う表は拒否。
        let mut bad = segs.clone();
        bad.swap(0, 1);
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &bad));

        // 本数の違う表は拒否。
        let mut other = segs.clone();
        other.pop();
        assert!(!grouped_expert_gemm_gpu_ready(&plan, &other));
    }

    /// テスト 5: 層・形の取り違えを黙って通さない。
    #[test]
    fn pack_rejects_wrong_layer_and_shape() {
        let dir = TempDir::new("wrong");
        let path = dir.path("m512.gguf");
        write_layer_gguf(&path, 1, 256, 32, 512);
        let gguf = GgufFile::open(&path).unwrap();
        let layout = ExpertLayout::from_gguf(&gguf, 1).unwrap();
        let (segs, _) = m_dist_batch(&[64usize, 32], 256, 32);
        let plan = plan_grouped(&segs, &layout).unwrap();

        // 層 2 はテンソルが無い。
        assert!(pack_expert_weights(&gguf, 2, &plan).is_err());

        // 形の違う層 (n_ff=16) には詰めない。
        let path2 = dir.path("other.gguf");
        write_layer_gguf(&path2, 1, 256, 64, 512);
        let gguf2 = GgufFile::open(&path2).unwrap();
        assert!(pack_expert_weights(&gguf2, 1, &plan).is_err());

        // 範囲外 expert の segment は plan にできない。
        let mut bad = segs.clone();
        bad[0].expert = 512;
        assert!(plan_grouped(&bad, &layout).is_err());
        // m = 0 の segment も拒否。
        let mut bad = segs.clone();
        bad[0].m = 0;
        assert!(plan_grouped(&bad, &layout).is_err());
    }

    /// 実形 (86 / 130 / 255 体) のパック時間。`--ignored --nocapture` で回す。
    ///
    /// 783 MB の一時ファイルを書くので既定のテストでは回さない。
    /// このコンテナは共有ホストで負荷が動くので、条件ごとに 9 回測って min と
    /// 中央値を出す (1 回の値はばらつく)。
    #[test]
    #[ignore]
    fn bench_pack_real_layer() {
        const REPS: usize = 9;
        let dir = TempDir::new("bench");
        let path = dir.path("real255.gguf");
        write_layer_gguf(&path, 0, 2560, 640, 255);
        let gguf = GgufFile::open(&path).unwrap();
        let layout = ExpertLayout::from_gguf(&gguf, 0).unwrap();

        let segs_for = |n: usize| -> Vec<Segment> {
            (0..n)
                .map(|e| Segment {
                    expert: e as u16,
                    m: 20,
                    x_off: (e * 20 * 2560) as u32,
                    out_off: (e * 20 * 640) as u32,
                })
                .collect()
        };

        for n in [86usize, 130, 255] {
            let segs = segs_for(n);
            let plan = plan_grouped(&segs, &layout).unwrap();
            let mut ms: Vec<f64> = Vec::with_capacity(REPS);
            let mut bytes = 0usize;
            for _ in 0..REPS {
                let (packed, stats) = pack_expert_weights(&gguf, 0, &plan).unwrap();
                bytes = packed.len();
                ms.push(stats.elapsed.as_secs_f64() * 1e3);
                std::hint::black_box(&packed);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "pack experts={n} bytes={:.2} MB min={:.2} ms median={:.2} ms rate(min)={:.2} GB/s",
                bytes as f64 / 1e6,
                ms[0],
                ms[REPS / 2],
                bytes as f64 / (ms[0] / 1e3) / 1e9
            );
        }

        // 宛先を再利用したとき (pack_expert_weights_into)。確保のコストを除いた分。
        let segs = segs_for(255);
        let plan = plan_grouped(&segs, &layout).unwrap();
        let mut reuse = vec![0u8; plan.packed_bytes];
        let mut ms: Vec<f64> = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let stats = pack_expert_weights_into(&gguf, 0, &plan, &mut reuse).unwrap();
            ms.push(stats.elapsed.as_secs_f64() * 1e3);
        }
        ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
        std::hint::black_box(&reuse);
        eprintln!(
            "pack_into(reuse, experts=255) bytes={:.2} MB min={:.2} ms median={:.2} ms rate(min)={:.2} GB/s",
            reuse.len() as f64 / 1e6,
            ms[0],
            ms[REPS / 2],
            reuse.len() as f64 / (ms[0] / 1e3) / 1e9
        );

        // 較正: このコンテナの素の memcpy 速度 (256 MB、確保済みバッファ間)。
        let a = vec![7u8; 256 << 20];
        let mut b = vec![0u8; 256 << 20];
        let mut cm: Vec<f64> = Vec::with_capacity(3);
        for _ in 0..3 {
            let t0 = Instant::now();
            b.copy_from_slice(&a);
            cm.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        cm.sort_by(|x, y| x.partial_cmp(y).unwrap());
        std::hint::black_box(&b);
        eprintln!(
            "memcpy 256 MB min={:.2} ms median={:.2} ms rate(min)={:.2} GB/s",
            cm[0],
            cm[1],
            (256u64 << 20) as f64 / (cm[0] / 1e3) / 1e9
        );
    }
}
