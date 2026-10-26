//! P-MOE-F2: Flash-Next (`qwen4exp`) の ssm 層を既存の GDN 実装で回す薄いアダプタ。
//!
//! Flash-Next の 48 層のうち 36 層 (`il % 4 != 3`) は GDN (ゲート付きデルタ則 +
//! 深さ方向 conv) と同じ演算で、`gdn::forward_token_core` がそのまま使える。
//! このモジュールがやるのは「GGUF のテンソル名から投影を読み、投影済みの値を
//! `forward_token_core` に渡す」ことだけ。新しい数式は書かない。
//!
//! 出典: llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`) の
//! `build_layer_attn_linear` / `build_qkvz` / `build_conv_state_at`。
//!
//! 諸元 (GGUF メタデータで確定):
//! - `ssm.state_size = 128` → head_k_dim = head_v_dim = 128
//! - `ssm.group_count = 16` → n_k_heads、`ssm.time_step_rank = 48` → n_v_heads
//! - `ssm.inner_size = 6144`、`ssm.conv_kernel = 4`
//! - `attn_qkv` の出力順は **[q 2048][k 2048][v 6144]** (q が 6144 ではない)。
//!   参照 `build_layer_attn_linear` の `q_conv` / `k_conv` / `v_conv` の view
//!   オフセット (0 / head_k_dim*num_k_heads / 2*head_k_dim*num_k_heads) が根拠。
//!
//! 重みの持ち方:
//! - 大きい 3 本 (`attn_qkv` / `attn_gate` / `ssm_out`) は量子化ブロックのまま
//!   [`GgufLinear`] で持つ (f32 へ展開しない。1 層 61 MB・36 層 2.2 GB)。
//!   mmap が使えるならブロックを写さずに共有写像を参照する。
//! - `ssm_beta` / `ssm_alpha` も [`GgufLinear`] の F32 経路で持つ (491 KB/層)。
//! - 小さい 4 本 (`ssm_conv1d` / `ssm_a` / `ssm_dt` / `ssm_norm`) は f32 常駐。
//!
//! **既知の差分 (F4 で判定する)**: 参照実装の出力ゲートは `sigmoid(z)` で、
//! `gdn.rs` の `silu(z)` とは違う (`qwen4exp.cpp` の `build_norm_gated`:
//! "the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not
//! silu")。本タスクの指示により `gdn.rs` の数式は変えていないので、この
//! アダプタは `silu(z)` で計算する。詳細は REPORT_P_MOE_F2.md。

use anyhow::{bail, Context, Result};

use crate::gdn::{self, GdnCore, GdnParams, GdnState};
use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::linear_gguf::GgufLinear;
use crate::model::qwen4exp::{
    one, IdentityAttn, LayerAttn, LayerKind, Qwen4ExpError, Qwen4ExpParams,
};
use crate::quant::WeightBlocks;

/// ssm 層 1 本ぶんの重みと状態。
pub struct SsmLayer {
    /// 層番号 (`blk.{il}.*`)。診断ダンプのゲートに使う。
    pub il: usize,
    /// GDN のハイパーパラメータ (`GdnParams::qwen4exp()`)。
    pub p: GdnParams,
    /// [2560 -> 10240] q/k/v の融合投影 (出力順は [q 2048][k 2048][v 6144])
    w_qkv: GgufLinear,
    /// [2560 -> 6144] z (出力ゲート)
    w_gate: GgufLinear,
    /// [2560 -> 48] beta の生値
    w_beta: GgufLinear,
    /// [2560 -> 48] alpha の生値
    w_alpha: GgufLinear,
    /// [6144 -> 2560] 出力投影
    w_out: GgufLinear,
    /// [conv_dim][d_conv] — チャネルが外・tap が内
    conv1d: Vec<f32>,
    /// [n_v_heads] — 負値のまま (-exp(A_log))。符号を反転しない。
    ssm_a: Vec<f32>,
    /// [n_v_heads] — dt bias
    ssm_dt: Vec<f32>,
    /// [head_v_dim] — head ごとの gated RMS norm の重み
    ssm_norm: Vec<f32>,
    /// conv 履歴 + SSM 行列
    pub state: GdnState,
}

impl SsmLayer {
    /// `blk.{il}.*` から読む。ハイパーパラメータは [`GdnParams::qwen4exp`] (実モデル) 固定。
    pub fn from_gguf(f: &GgufFile, il: usize) -> Result<Self> {
        Self::from_gguf_with(f, il, GdnParams::qwen4exp())
    }

    /// ハイパーパラメータを外から与えて `blk.{il}.*` から読む。
    ///
    /// 形状・型・本数の検査を全部ここで行い、欠けたら Err。
    /// 実モデルでは [`gdn_params`] の導出値が [`GdnParams::qwen4exp`] と一致する。
    pub fn from_gguf_with(f: &GgufFile, il: usize, p: GdnParams) -> Result<Self> {
        let name = |suffix: &str| format!("blk.{il}.{suffix}");

        let n_qkv = name("attn_qkv.weight");
        let n_gate = name("attn_gate.weight");
        let n_beta = name("ssm_beta.weight");
        let n_alpha = name("ssm_alpha.weight");
        let n_conv = name("ssm_conv1d.weight");
        let n_a = name("ssm_a");
        let n_dt = name("ssm_dt.bias");
        let n_norm = name("ssm_norm.weight");
        let n_out = name("ssm_out.weight");

        // ---- 形状 (GGUF の ne0 が入力・ne1 が出力) ----------------------
        expect_2d(f, &n_qkv, p.n_embd, p.qkv_dim())?;
        expect_2d(f, &n_gate, p.n_embd, p.value_dim())?;
        expect_2d(f, &n_beta, p.n_embd, p.n_v_heads)?;
        expect_2d(f, &n_alpha, p.n_embd, p.n_v_heads)?;
        expect_2d(f, &n_out, p.value_dim(), p.n_embd)?;
        // conv1d は ne0 = tap、ne1 = チャネル。素直に f32 へ落とすと
        // [channel][tap] になる (gdn.rs が欲しい並びそのもの)。
        expect_2d(f, &n_conv, p.d_conv, p.conv_dim())?;
        expect_1d(f, &n_a, p.n_v_heads)?;
        expect_1d(f, &n_dt, p.n_v_heads)?;
        expect_1d(f, &n_norm, p.head_v_dim)?;

        // ---- 大きい 3 本 + beta/alpha はブロックのまま -------------------
        let w_qkv = linear(f, &n_qkv, p.n_embd, p.qkv_dim())?;
        let w_gate = linear(f, &n_gate, p.n_embd, p.value_dim())?;
        let w_beta = linear(f, &n_beta, p.n_embd, p.n_v_heads)?;
        let w_alpha = linear(f, &n_alpha, p.n_embd, p.n_v_heads)?;
        let w_out = linear(f, &n_out, p.value_dim(), p.n_embd)?;

        // ---- 小さい 4 本は f32 常駐 -------------------------------------
        let conv1d = dequant_exact(f, &n_conv, p.conv_dim() * p.d_conv)?;
        let ssm_a = dequant_exact(f, &n_a, p.n_v_heads)?;
        let ssm_dt = dequant_exact(f, &n_dt, p.n_v_heads)?;
        let ssm_norm = dequant_exact(f, &n_norm, p.head_v_dim)?;

        let state = GdnState::new(&p);
        Ok(Self {
            il,
            p,
            w_qkv,
            w_gate,
            w_beta,
            w_alpha,
            w_out,
            conv1d,
            ssm_a,
            ssm_dt,
            ssm_norm,
            state,
        })
    }

    /// `cur: [n_embd]` → 出力 `[n_embd]`。state を 1 トークンぶん進める。
    pub fn forward_token(&mut self, cur: &[f32]) -> Result<Vec<f32>> {
        if cur.len() != self.p.n_embd {
            bail!(
                "SsmLayer::forward_token: 入力長 {} が n_embd={} と違う",
                cur.len(),
                self.p.n_embd
            );
        }
        let prof = crate::model::qwen4exp::q4prof::enabled();
        let t = std::time::Instant::now();
        let qkv = self.project_qkv(cur)?;
        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_QKV, t);
        }
        let t = std::time::Instant::now();

        let mut z = vec![0f32; self.p.value_dim()];
        self.w_gate.gemv(cur, &mut z)?;
        let mut beta_raw = vec![0f32; self.p.n_v_heads];
        self.w_beta.gemv(cur, &mut beta_raw)?;
        let mut alpha = vec![0f32; self.p.n_v_heads];
        self.w_alpha.gemv(cur, &mut alpha)?;

        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_GATE, t);
        }
        let t = std::time::Instant::now();

        // 診断ダンプ (DENNO_DEBUG_GDN_DUMP_LAYER) のゲートに層番号を渡す。
        gdn::gdn_dbg_set_layer(self.il as i64);
        let cw = GdnCore {
            conv1d: &self.conv1d,
            ssm_a: &self.ssm_a,
            ssm_dt: &self.ssm_dt,
            ssm_norm: &self.ssm_norm,
        };
        // 数式・演算順序は gdn.rs のものをそのまま使う (この層では書き直さない)。
        // qwen4exp の出力ゲートは **sigmoid** (qwen35 の silu ではない)。
        let y = gdn::forward_token_core_gate(
            &self.p,
            &cw,
            &qkv,
            &z,
            &beta_raw,
            &alpha,
            &mut self.state,
            gdn::GateKind::Sigmoid,
        );

        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_GDN, t);
        }
        let t = std::time::Instant::now();

        let mut out = vec![0f32; self.p.n_embd];
        self.w_out.gemv(&y, &mut out)?;
        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_OUT, t);
        }
        Ok(out)
    }

    /// `attn_qkv` の投影だけを行う (state は進めない)。
    ///
    /// 戻り値の並びは **[q key_dim][k key_dim][v value_dim]**。F4 の切り分けと
    /// レイアウト検証で使う。
    pub fn project_qkv(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut qkv = vec![0f32; self.p.qkv_dim()];
        self.w_qkv.gemv(cur, &mut qkv)?;
        Ok(qkv)
    }

    /// qkv 出力の切り分け `(q_end, k_end, v_end)`。qwen4exp では `(2048, 4096, 10240)`。
    pub fn qkv_split(&self) -> (usize, usize, usize) {
        let kd = self.p.key_dim();
        let vd = self.p.value_dim();
        (kd, 2 * kd, 2 * kd + vd)
    }

    /// 1 トークンあたりの重みバイト数 (大きい 5 本の合計)。
    pub fn weight_bytes(&self) -> usize {
        self.w_qkv.weight_bytes()
            + self.w_gate.weight_bytes()
            + self.w_beta.weight_bytes()
            + self.w_alpha.weight_bytes()
            + self.w_out.weight_bytes()
            + self.conv1d.len() * 4
            + self.ssm_a.len() * 4
            + self.ssm_dt.len() * 4
            + self.ssm_norm.len() * 4
    }
}

/// `(ne0, ne1)` が期待どおりで、余次元が 1 であることを確かめる。
fn expect_2d(f: &GgufFile, name: &str, k: usize, n: usize) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if t.dims[0] as usize != k || t.dims[1] as usize != n {
        bail!(
            "{name}: ne={:?} が期待 [{k}, {n}] と違う",
            &t.dims[..2]
        );
    }
    if t.dims[2] != 1 || t.dims[3] != 1 {
        bail!("{name}: 2 次元を超えるテンソル (ne={:?})", t.dims);
    }
    Ok(())
}

/// `ne0 == n` の 1 次元テンソルであることを確かめる。
fn expect_1d(f: &GgufFile, name: &str, n: usize) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if t.dims[0] as usize != n || t.dims[1..].iter().any(|&d| d != 1) {
        bail!("{name}: ne={:?} が期待 [{n}] と違う", t.dims);
    }
    Ok(())
}

/// [`GgufLinear`] の `gemv` が扱える型か。
fn gemv_supported(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q8_0
            | GgmlType::Q5_1
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
    )
}

/// 量子化ブロックを保持したまま [`GgufLinear`] を作る。
///
/// mmap のスライスが取れるときはバイト列を写さない (`WeightBlocks::mapped`)。
fn linear(f: &GgufFile, name: &str, k: usize, n: usize) -> Result<GgufLinear> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if !gemv_supported(t.ty) {
        bail!("{name}: 未対応の量子型 {:?}", t.ty);
    }
    let ty = t.ty;
    if let Some((file, off, len)) = f.tensor_mapping_parts(name) {
        if let Some(map) = f.mmap_handle_at(file) {
            return GgufLinear::new(ty, k, n, WeightBlocks::mapped(map, off, len))
                .map(|l| l.with_fast_q8(true))
                .with_context(|| format!("{name}: mmap ブロックから GgufLinear を作れない"));
        }
    }
    let bytes = f
        .tensor_bytes(name)
        .with_context(|| format!("tensor {name}"))?;
    GgufLinear::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
        .with_context(|| format!("{name}: GgufLinear を作れない"))
}

/// 小さい f32 テンソルを厳密な要素数で読み出す。
fn dequant_exact(f: &GgufFile, name: &str, want: usize) -> Result<Vec<f32>> {
    let v = f
        .dequant_tensor(name)
        .with_context(|| format!("tensor {name}"))?;
    if v.len() != want {
        bail!("{name}: 要素数 {} が期待 {want} と違う", v.len());
    }
    Ok(v)
}

// ---------------------------------------------------------------------------
// P-MOE-F3: ssm 層を `LayerAttn` として層ループに接続する
// ---------------------------------------------------------------------------

/// [`Qwen4ExpParams`] (GGUF メタデータ) から GDN のハイパーパラメータを導出する。
///
/// 対応 (参照実装 `qwen4exp.cpp` の `load_arch_hparams` と同じ読み替え):
/// - `embedding_length` → `n_embd`
/// - `ssm.group_count` → `n_k_heads`
/// - `ssm.time_step_rank` → `n_v_heads`
/// - `ssm.state_size` → `head_k_dim`
/// - `ssm.inner_size / ssm.time_step_rank` → `head_v_dim`
/// - `ssm.conv_kernel` → `d_conv`
///
/// 実モデル (2560 / 16 / 48 / 128 / 6144 / 4 / 1e-6) では
/// [`GdnParams::qwen4exp`] と同じ値になる (`tests/qwen4exp_layers.rs` で固定)。
pub fn gdn_params(p: &Qwen4ExpParams) -> Result<GdnParams> {
    if p.ssm_dt_rank == 0 {
        bail!("qwen4exp.ssm.time_step_rank が 0 (head_v_dim を決められない)");
    }
    if p.ssm_d_inner % p.ssm_dt_rank != 0 {
        bail!(
            "qwen4exp.ssm.inner_size {} が time_step_rank {} で割り切れない",
            p.ssm_d_inner,
            p.ssm_dt_rank
        );
    }
    let g = GdnParams {
        n_embd: p.n_embd,
        n_k_heads: p.ssm_n_group,
        n_v_heads: p.ssm_dt_rank,
        head_k_dim: p.ssm_d_state,
        head_v_dim: p.ssm_d_inner / p.ssm_dt_rank,
        d_conv: p.ssm_d_conv,
        eps: p.rms_eps,
    };
    // 派生値どうしの突き合わせ (メタデータの 3 つの数が同じ次元を指していること)。
    if g.key_dim() != p.ssm_key_dim() {
        bail!(
            "ssm key dim {} が state_size * group_count = {} と違う",
            g.key_dim(),
            p.ssm_key_dim()
        );
    }
    if g.value_dim() != p.ssm_value_dim() {
        bail!(
            "ssm value dim {} が inner_size = {} と違う",
            g.value_dim(),
            p.ssm_value_dim()
        );
    }
    if g.conv_dim() != p.ssm_conv_dim() {
        bail!(
            "conv dim {} が 2*key + value = {} と違う",
            g.conv_dim(),
            p.ssm_conv_dim()
        );
    }
    Ok(g)
}

/// ssm 層を [`LayerAttn`] として回す。full attention 層は `full` に委譲する
/// (F3 の既定は [`IdentityAttn`]。F4 が実装を差し替える)。
///
/// `layers[il]` が `Some` なら ssm 層 (`il % interval != interval-1`)、
/// `None` なら full attention 層。状態 (`conv` / `ssm`) は層ごとに独立で、
/// `forward` を呼ぶたびに 1 トークンぶん進む。
pub struct SsmAttn {
    layers: Vec<Option<SsmLayer>>,
    full: Box<dyn LayerAttn>,
}

impl SsmAttn {
    /// `p.layer_kind(il)` を見て ssm 層だけ [`SsmLayer`] を組む。
    /// full attention 層は [`IdentityAttn`] に委譲する。
    pub fn from_gguf(f: &GgufFile, p: &Qwen4ExpParams) -> Result<Self> {
        Self::with_full(f, p, Box::new(IdentityAttn))
    }

    /// full attention 層の実装を差し替えて組む (F4 用)。
    pub fn with_full(
        f: &GgufFile,
        p: &Qwen4ExpParams,
        full: Box<dyn LayerAttn>,
    ) -> Result<Self> {
        let g = gdn_params(p)?;
        let mut layers = Vec::with_capacity(p.n_layer);
        for il in 0..p.n_layer {
            let kind = p
                .layer_kind(il)
                .map_err(|e| anyhow::anyhow!("層 {il}: {e}"))?;
            layers.push(match kind {
                LayerKind::Ssm => Some(
                    SsmLayer::from_gguf_with(f, il, g.clone())
                        .with_context(|| format!("層 {il}: ssm 層を組めない"))?,
                ),
                LayerKind::FullAttn => None,
            });
        }
        Ok(Self { layers, full })
    }

    /// 層ごとの種別 (組み上げた結果。テスト・診断用)。
    pub fn kinds(&self) -> Vec<LayerKind> {
        self.layers
            .iter()
            .map(|l| {
                if l.is_some() {
                    LayerKind::Ssm
                } else {
                    LayerKind::FullAttn
                }
            })
            .collect()
    }

    /// ssm 層の本数 (= [`Qwen4ExpParams::n_ssm_layers`])
    pub fn n_ssm(&self) -> usize {
        self.layers.iter().filter(|l| l.is_some()).count()
    }

    /// full attention 層の本数 (= [`Qwen4ExpParams::n_full_attn_layers`])
    pub fn n_full(&self) -> usize {
        self.layers.iter().filter(|l| l.is_none()).count()
    }

    /// ssm 層の状態 (テスト・診断用)。full attention 層は `None`。
    pub fn ssm_layer(&self, il: usize) -> Option<&SsmLayer> {
        self.layers.get(il).and_then(|l| l.as_ref())
    }
}

impl LayerAttn for SsmAttn {
    fn forward(
        &mut self,
        layer: usize,
        cur: &[f32],
    ) -> std::result::Result<Vec<f32>, Qwen4ExpError> {
        let n = self.layers.len();
        match self.layers.get_mut(layer) {
            None => Err(one(format!(
                "SsmAttn::forward: 層 {layer} は範囲外 (層数 {n})"
            ))),
            Some(Some(s)) => s
                .forward_token(cur)
                // anyhow の文脈を 1 行に畳む (`{:#}` は "外側: 内側" を連結する)
                .map_err(|e| Qwen4ExpError(vec![format!("層 {layer}: {e:#}")])),
            Some(None) => self.full.forward(layer, cur),
        }
    }
}
