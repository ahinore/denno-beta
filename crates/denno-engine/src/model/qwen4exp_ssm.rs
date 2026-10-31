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
use crate::model::linear_gguf::{FusedLinear, GgufLinear};
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
    /// [2560 -> 10240] q/k/v の融合投影 (出力順は [q 2048][k 2048][v 6144]) と
    /// [2560 -> 6144] z (出力ゲート)。
    ///
    /// task 72: どちらも同じ入力 `cur` を取るので、**行方向に連結して 1 本の
    /// GEMV** にする (量子型が同じとき)。行は独立なので値はビット一致する。
    qkv_gate: FusedLinear,
    /// [2560 -> 48] beta と alpha の生値 (どちらも F32・同じ入力なので連結)。
    beta_alpha: FusedLinear,
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
        // task 72: 同じ入力を取る投影を 1 本の GEMV にまとめる。qkv+gate は
        // 実モデルではどちらも Q8_0、beta/alpha はどちらも F32 なので 2 組に
        // 分かれて連結される (型が揃わなければ自動で個別のまま)。
        let qkv_gate = FusedLinear::concat(vec![w_qkv, w_gate]);
        let beta_alpha = FusedLinear::concat(vec![w_beta, w_alpha]);

        // ---- 小さい 4 本は f32 常駐 -------------------------------------
        let conv1d = dequant_exact(f, &n_conv, p.conv_dim() * p.d_conv)?;
        let ssm_a = dequant_exact(f, &n_a, p.n_v_heads)?;
        let ssm_dt = dequant_exact(f, &n_dt, p.n_v_heads)?;
        let ssm_norm = dequant_exact(f, &n_norm, p.head_v_dim)?;

        let state = GdnState::new(&p);
        Ok(Self {
            il,
            p,
            qkv_gate,
            beta_alpha,
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
        // task 72: qkv と z (出力ゲート) は同じ入力 cur を取るので 1 本の GEMV に
        // まとめる。行は独立・1 行の中の加算順は連結しても変わらないので、
        // 個別に呼んだ場合とビット一致する。ゲートの時間もここ (A_QKV) に入る。
        let mut qkvg = vec![0f32; self.qkv_gate.n()];
        self.qkv_gate.gemv_concat_into("qkv", cur, &mut qkvg)?;
        let qkv = &qkvg[self.qkv_gate.range(0)];
        let z = &qkvg[self.qkv_gate.range(1)];
        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_QKV, t);
        }
        let t = std::time::Instant::now();

        // task 72: beta と alpha も同じ入力なので 1 本にまとめる (どちらも F32)。
        let mut ba = vec![0f32; self.beta_alpha.n()];
        self.beta_alpha.gemv_concat_into("ssm_beta", cur, &mut ba)?;
        let beta_raw = &ba[self.beta_alpha.range(0)];
        let alpha = &ba[self.beta_alpha.range(1)];

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
            qkv,
            z,
            beta_raw,
            alpha,
            &mut self.state,
            gdn::GateKind::Sigmoid,
        );

        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_GDN, t);
        }
        let t = std::time::Instant::now();

        let mut out = vec![0f32; self.p.n_embd];
        self.w_out.gemv_label("attn_out", &y, &mut out)?;
        if prof {
            crate::model::qwen4exp::q4prof::add(&crate::model::qwen4exp::q4prof::A_OUT, t);
        }
        Ok(out)
    }

    /// チャンクぶんの入力をまとめて回す (バッチ prefill 用)。
    ///
    /// 変えるのは行列積と GDN 本体の呼び方だけ:
    /// - 5 本の投影をチャンクで 1 回ずつ [`gemv_batch`] に流す (GPU 常駐時は
    ///   1 トークンあたり 5 往復していたのがチャンクあたり 5 往復になる)。
    /// - GDN 本体は [`gdn::forward_batch_core_gate`] にチャンク一括で渡す
    ///   (conv1d をトークン方向にまとめ、状態スキャンをヘッド範囲で並列化)。
    ///   ヘッドは互いに独立なので、トークン順を保てば要素ごとの丸め順は不変。
    ///
    /// 数値: 行列積は逐次と同じカーネルをトークンごとに当てるだけ、GDN 本体は
    /// 要素ごとに独立な演算を同じ順序で並べ替えただけなので、`forward_token` と
    /// ビット一致する (テスト `gdn_batch_matches_token_by_token_bitexact`、
    /// `ssm_batch_matches_token_by_token_bitexact`)。
    ///
    /// 診断ダンプ (`DENNO_DEBUG_GDN_DUMP_LAYER`) がこの層を指しているときだけ
    /// 従来のトークン逐次に落ちる (サブステージの追記順を保つため)。
    pub fn forward_batch(&mut self, curs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let nt = curs.len();
        if nt == 0 {
            return Ok(Vec::new());
        }
        if nt == 1 {
            return Ok(vec![self.forward_token(&curs[0])?]);
        }
        let e = self.p.n_embd;
        let mut flat = Vec::with_capacity(nt * e);
        for cur in curs {
            if cur.len() != e {
                bail!(
                    "SsmLayer::forward_batch: 入力長 {} が n_embd={e} と違う",
                    cur.len()
                );
            }
            flat.extend_from_slice(cur);
        }

        let qd = self.p.qkv_dim();
        let vd = self.p.value_dim();
        let nh = self.p.n_v_heads;

        let t = crate::model::qwen4exp::q4prof::stamp();
        // task 72: qkv と z、beta と alpha をそれぞれ 1 回のバッチ GEMV にまとめる。
        // トークンごとの値は個別に呼んだ場合と同じ (行の独立)。
        let mut qkv = vec![0f32; nt * qd];
        let mut z = vec![0f32; nt * vd];
        self.qkv_gate
            .gemv_batch_split(&flat, nt, &mut [&mut qkv[..], &mut z[..]])?;
        crate::model::qwen4exp::q4prof::mark(
            t,
            &crate::model::qwen4exp::q4prof::A_QKV,
            self.il,
            crate::model::qwen4exp::q4prof::ST_QKV,
        );
        crate::model::qwen4exp::q4prof::bump_nt(self.il, nt);

        let t = crate::model::qwen4exp::q4prof::stamp();
        let mut beta_raw = vec![0f32; nt * nh];
        let mut alpha = vec![0f32; nt * nh];
        self.beta_alpha
            .gemv_batch_split(&flat, nt, &mut [&mut beta_raw[..], &mut alpha[..]])?;
        crate::model::qwen4exp::q4prof::mark(
            t,
            &crate::model::qwen4exp::q4prof::A_GATE,
            self.il,
            crate::model::qwen4exp::q4prof::ST_GATE,
        );

        let t = crate::model::qwen4exp::q4prof::stamp();
        gdn::gdn_dbg_set_layer(self.il as i64);
        let cw = GdnCore {
            conv1d: &self.conv1d,
            ssm_a: &self.ssm_a,
            ssm_dt: &self.ssm_dt,
            ssm_norm: &self.ssm_norm,
        };
        // サブステージのダンプがこの層を指しているときだけトークン逐次に落ちる
        // (追記順 = llama.cpp の lchk と同一に保つ)。通常はチャンク一括。
        let ys = if gdn::gdn_dbg_enabled_for(self.il as i64) {
            let mut ys = Vec::with_capacity(nt * vd);
            for i in 0..nt {
                let y = gdn::forward_token_core_gate(
                    &self.p,
                    &cw,
                    &qkv[i * qd..(i + 1) * qd],
                    &z[i * vd..(i + 1) * vd],
                    &beta_raw[i * nh..(i + 1) * nh],
                    &alpha[i * nh..(i + 1) * nh],
                    &mut self.state,
                    gdn::GateKind::Sigmoid,
                );
                if y.len() != vd {
                    bail!(
                        "SsmLayer::forward_batch: GDN の出力長 {} が value_dim={vd} と違う",
                        y.len()
                    );
                }
                ys.extend_from_slice(&y);
            }
            ys
        } else {
            gdn::forward_batch_core_gate(
                &self.p,
                &cw,
                &qkv,
                &z,
                &beta_raw,
                &alpha,
                &mut self.state,
                gdn::GateKind::Sigmoid,
            )
        };
        if ys.len() != nt * vd {
            bail!(
                "SsmLayer::forward_batch: GDN の出力長 {} が nt={nt} * value_dim={vd} と違う",
                ys.len()
            );
        }
        crate::model::qwen4exp::q4prof::mark_l(
            t,
            self.il,
            crate::model::qwen4exp::q4prof::ST_GDN,
        );

        let t = crate::model::qwen4exp::q4prof::stamp();
        let mut outs = vec![0f32; nt * e];
        crate::model::qwen4exp::gemv_batch(&self.w_out, &ys, nt, &mut outs)?;
        crate::model::qwen4exp::q4prof::mark(
            t,
            &crate::model::qwen4exp::q4prof::A_OUT,
            self.il,
            crate::model::qwen4exp::q4prof::ST_OUT,
        );
        Ok(outs.chunks_exact(e).map(|c| c.to_vec()).collect())
    }

    /// `attn_qkv` の投影だけを行う (state は進めない)。
    ///
    /// 戻り値の並びは **[q key_dim][k key_dim][v value_dim]**。F4 の切り分けと
    /// レイアウト検証で使う。
    ///
    /// task 72: qkv は gate と連結されているので、この入口でも連結ぶん
    /// (gate) を計算する。診断・テスト用の入口で、decode の本線は
    /// [`Self::forward_token`] (1 回の GEMV で qkv と z を同時に得る)。
    pub fn project_qkv(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut buf = vec![0f32; self.qkv_gate.n()];
        self.qkv_gate.gemv_concat_into("qkv", cur, &mut buf)?;
        Ok(buf[self.qkv_gate.range(0)].to_vec())
    }

    /// task 72: qkv+gate が 1 本の GEMV に連結されているか (発動確認用)。
    pub fn qkv_gate_is_fused(&self) -> bool {
        self.qkv_gate.is_fused()
    }

    /// task 72: beta+alpha が 1 本の GEMV に連結されているか (発動確認用)。
    pub fn beta_alpha_is_fused(&self) -> bool {
        self.beta_alpha.is_fused()
    }

    /// qkv 出力の切り分け `(q_end, k_end, v_end)`。qwen4exp では `(2048, 4096, 10240)`。
    pub fn qkv_split(&self) -> (usize, usize, usize) {
        let kd = self.p.key_dim();
        let vd = self.p.value_dim();
        (kd, 2 * kd, 2 * kd + vd)
    }

    /// 1 トークンあたりの重みバイト数 (大きい 5 本の合計)。
    pub fn weight_bytes(&self) -> usize {
        self.qkv_gate.weight_bytes()
            + self.beta_alpha.weight_bytes()
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
    // 融合カーネルの有無ではなく「デコードできるか」で通す (2026-10-31)。
    crate::kquant::dequant_supported(ty)
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
            let l = GgufLinear::new(ty, k, n, WeightBlocks::mapped(map, off, len))
                .with_context(|| format!("{name}: mmap ブロックから GgufLinear を作れない"))?;
            return crate::model::linear_gguf::maybe_requant(l)
                .map(|l| l.with_fast_q8(true))
                .with_context(|| format!("{name}: Q8_0 へ作り直せない"));
        }
    }
    let bytes = f
        .tensor_bytes(name)
        .with_context(|| format!("tensor {name}"))?;
    let l = GgufLinear::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
        .with_context(|| format!("{name}: GgufLinear を作れない"))?;
    crate::model::linear_gguf::maybe_requant(l)
        .map(|l| l.with_fast_q8(true))
        .with_context(|| format!("{name}: Q8_0 へ作り直せない"))
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

    /// ssm 層は投影をチャンクでまとめる。full 層は `self.full` の
    /// [`LayerAttn::forward_batch`] に委譲する (バッチ prefill で `FullAttn` が
    /// 入っているときは KV を 1 回走査するバッチ版が走る。既定の実装 =
    /// 1 トークンずつの `forward` と同じ値)。
    fn forward_batch(
        &mut self,
        layer: usize,
        curs: &[Vec<f32>],
    ) -> std::result::Result<Vec<Vec<f32>>, Qwen4ExpError> {
        let n = self.layers.len();
        match self.layers.get_mut(layer) {
            None => Err(one(format!(
                "SsmAttn::forward_batch: 層 {layer} は範囲外 (層数 {n})"
            ))),
            Some(Some(s)) => s
                .forward_batch(curs)
                .map_err(|e| Qwen4ExpError(vec![format!("層 {layer}: {e:#}")])),
            Some(None) => self.full.forward_batch(layer, curs),
        }
    }

    /// 層優先の順序でも安全。`forward` は「1 層の状態を 1 トークン進める」だけで、
    /// 層どうしは独立している。層優先にしても各層の中でトークン順は保たれるので、
    /// GDN の conv / ssm 状態の進み方は逐次版と同じになる。
    fn layer_major_safe(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// P-PF-GDN: ssm 層のバッチ経路 (`forward_batch` → `gdn::forward_batch_core_gate`)
// が 1 トークンずつの `forward_token` とビット一致することの固定。
//
// 実モデル (111 GB) は使わない。合成した F32 の投影行列で「構造は実モデルと同じ・
// 寸法だけ小さい」層を組み、同じ入力列を 2 つの層に「逐次」と「チャンク」で
// 食わせて比べる。出力だけでなく conv / ssm 状態も比べる (状態がずれると次の
// チャンクで静かに壊れる)。
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn lcg_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut rng = seed;
        move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / (u32::MAX as f32) - 0.5
        }
    }

    /// F32 の重みで [`GgufLinear`] を作る (`WeightBlocks` は行優先 [n][k])。
    fn f32_linear(k: usize, n: usize, rnd: &mut impl FnMut() -> f32) -> GgufLinear {
        let data: Vec<f32> = (0..k * n).map(|_| rnd()).collect();
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in &data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        GgufLinear::new(GgmlType::F32, k, n, WeightBlocks::from_vec(bytes))
            .expect("GgufLinear (F32)")
    }

    /// 構造は実モデルと同じ・寸法だけ小さい ssm 層 (nk=2 / nv=4 で GQA の
    /// タイル対応 h % nk も通る)。
    fn small_layer() -> (SsmLayer, GdnParams) {
        let p = GdnParams {
            n_embd: 16,
            n_k_heads: 2,
            n_v_heads: 4,
            head_k_dim: 8,
            head_v_dim: 8,
            d_conv: 4,
            eps: 1e-6,
        };
        let mut rnd = lcg_rng(0x0f0f_1234_5678_9abc);
        let vd = p.value_dim();
        let layer = SsmLayer {
            il: 0,
            p: p.clone(),
            qkv_gate: FusedLinear::concat(vec![
                f32_linear(p.n_embd, p.qkv_dim(), &mut rnd),
                f32_linear(p.n_embd, vd, &mut rnd),
            ]),
            beta_alpha: FusedLinear::concat(vec![
                f32_linear(p.n_embd, p.n_v_heads, &mut rnd),
                f32_linear(p.n_embd, p.n_v_heads, &mut rnd),
            ]),
            w_out: f32_linear(vd, p.n_embd, &mut rnd),
            conv1d: (0..p.conv_dim() * p.d_conv).map(|_| 0.4 * rnd()).collect(),
            ssm_a: (0..p.n_v_heads).map(|_| -0.05 - 0.05 * rnd().abs()).collect(),
            ssm_dt: (0..p.n_v_heads).map(|_| 0.1 * rnd()).collect(),
            ssm_norm: (0..p.head_v_dim).map(|_| 0.5 + rnd().abs()).collect(),
            state: GdnState::new(&p),
        };
        (layer, p)
    }

    fn inputs(p: &GdnParams, nt: usize) -> Vec<Vec<f32>> {
        (0..nt)
            .map(|t| {
                (0..p.n_embd)
                    .map(|j| 0.3 * (((t * 7 + j * 3) % 5) as f32) - 0.6)
                    .collect()
            })
            .collect()
    }

    /// 2 トークン以上で `forward_batch` のチャンク経路が発動し、`forward_token` を
    /// nt 回呼ぶのと出力・状態ともビット一致する。
    #[test]
    fn ssm_batch_matches_token_by_token_bitexact() {
        for nt in [2usize, 3, 5, 9, 17] {
            let (mut seq, p) = small_layer();
            let (mut bat, _) = small_layer();
            let curs = inputs(&p, nt);

            let want: Vec<Vec<f32>> = curs
                .iter()
                .map(|c| seq.forward_token(c).expect("forward_token"))
                .collect();
            let got = bat.forward_batch(&curs).expect("forward_batch");

            assert_eq!(got.len(), nt, "nt={nt}: 出力トークン数");
            for t in 0..nt {
                assert_eq!(
                    bits(&got[t]),
                    bits(&want[t]),
                    "nt={nt}: トークン {t} の出力がビット不一致"
                );
            }
            assert_eq!(bits(&seq.state.ssm), bits(&bat.state.ssm), "nt={nt}: ssm 状態");
            assert_eq!(bits(&seq.state.conv), bits(&bat.state.conv), "nt={nt}: conv 状態");
        }
    }

    /// チャンクの切り方を変えても同じ (状態の持ち越しが正しい)。
    #[test]
    fn ssm_batch_chunk_split_matches_single_chunk() {
        let (mut one, p) = small_layer();
        let (mut split, _) = small_layer();
        let curs = inputs(&p, 11);

        let whole = one.forward_batch(&curs).expect("forward_batch (1 回)");
        let mut pieces: Vec<Vec<f32>> = Vec::new();
        for (a, b) in [(0usize, 3usize), (3, 4), (4, 11)] {
            pieces.extend(split.forward_batch(&curs[a..b]).expect("forward_batch (分割)"));
        }
        assert_eq!(pieces.len(), whole.len());
        for t in 0..whole.len() {
            assert_eq!(bits(&pieces[t]), bits(&whole[t]), "トークン {t}");
        }
        assert_eq!(bits(&one.state.ssm), bits(&split.state.ssm), "ssm 状態");
        assert_eq!(bits(&one.state.conv), bits(&split.state.conv), "conv 状態");
    }

    /// nt=1 は `forward_token` へ委譲する (バッチ経路に入らない)。
    #[test]
    fn ssm_batch_single_token_delegates() {
        let (mut a, p) = small_layer();
        let (mut b, _) = small_layer();
        let curs = inputs(&p, 1);
        let want = a.forward_token(&curs[0]).expect("forward_token");
        let got = b.forward_batch(&curs).expect("forward_batch");
        assert_eq!(got.len(), 1);
        assert_eq!(bits(&got[0]), bits(&want));
    }

    // ---- task 72: qkv+gate / beta+alpha の融合 ------------------------------

    /// 上書きを必ず env 読みに戻す (panic しても戻る)。
    struct FusedGuard;
    impl FusedGuard {
        fn new() -> Self {
            crate::model::linear_gguf::set_fused_gemv_for_test(None);
            Self
        }
    }
    impl Drop for FusedGuard {
        fn drop(&mut self) {
            crate::model::linear_gguf::set_fused_gemv_for_test(None);
        }
    }

    /// 融合した層と分けた層が、出力も状態も**ビット一致**する。
    ///
    /// 数値不変の根拠: 行は独立で、1 行の内積は `i` 昇順の逐次加算のまま
    /// (連結しても 1 行の中の順序は変わらない)。
    #[test]
    fn ssm_fused_matches_separate_bitexact() {
        let _g = crate::model::linear_gguf::par_test_lock();
        let _p = FusedGuard::new();

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(true));
        let (mut fused, p) = small_layer();
        assert!(fused.qkv_gate_is_fused(), "qkv+gate が連結されていない");
        assert!(fused.beta_alpha_is_fused(), "beta+alpha が連結されていない");

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(false));
        let (mut sep, _) = small_layer();
        assert!(!sep.qkv_gate_is_fused(), "off なのに連結された");
        assert!(!sep.beta_alpha_is_fused(), "off なのに連結された");

        let curs = inputs(&p, 7);
        for (t, c) in curs.iter().enumerate() {
            let a = fused.forward_token(c).expect("fused forward_token");
            let b = sep.forward_token(c).expect("separate forward_token");
            assert_eq!(bits(&a), bits(&b), "トークン {t}");
        }
        assert_eq!(bits(&fused.state.ssm), bits(&sep.state.ssm), "ssm 状態");
        assert_eq!(bits(&fused.state.conv), bits(&sep.state.conv), "conv 状態");
    }

    /// バッチ経路 (`gemv_batch_split`) も融合の有無でビット一致する。
    #[test]
    fn ssm_fused_batch_matches_separate_bitexact() {
        let _g = crate::model::linear_gguf::par_test_lock();
        let _p = FusedGuard::new();

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(true));
        let (mut fused, p) = small_layer();
        assert!(fused.qkv_gate_is_fused());

        crate::model::linear_gguf::set_fused_gemv_for_test(Some(false));
        let (mut sep, _) = small_layer();

        let curs = inputs(&p, 5);
        let a = fused.forward_batch(&curs).expect("fused forward_batch");
        let b = sep.forward_batch(&curs).expect("separate forward_batch");
        assert_eq!(a.len(), b.len());
        for t in 0..a.len() {
            assert_eq!(bits(&a[t]), bits(&b[t]), "トークン {t}");
        }
        assert_eq!(bits(&fused.state.ssm), bits(&sep.state.ssm), "ssm 状態");
        assert_eq!(bits(&fused.state.conv), bits(&sep.state.conv), "conv 状態");
    }
}
