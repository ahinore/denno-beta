//! P-MOE-F4: Flash-Next (`qwen4exp`) の full attention 層 (12 層) を dense 経路で実装する。
//!
//! 出典: llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、MIT License,
//! Copyright (c) 2023-2026 The ggml authors) の `build_layer_attn` (1014-1100 行) と
//! `src/ggml/src/ggml-cpu/ggml-cpu.c` の `rotate_pairs` / `ggml_mrope_cache_init`。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//!
//! # 担当範囲
//!
//! `(il + 1) % full_attention_interval == 0` の 12 層 (3, 7, ..., 47) を
//! dense GQA + 出力ゲートで回す。
//!
//! 参照実装は `compress_ratios[il] > 0` のときだけ `build_qsa_sel` を通す。
//! ここでは既定で dense 側 (`build_attn`) を回し、`DENNO_QSA=1` のときだけ
//! [`crate::model::qwen4exp_qsa::QsaState`] の選択結果を
//! [`AttnLayer::attend_visible`] に食わせる (G2)。選択の継ぎ目は
//! [`AttnLayer::norm_rope_q`] の後・[`AttnLayer::attend`] の前。
//! `DENNO_QSA` が未設定 / 0 なら indexer のテンソルは読まない (dense 経路そのまま)。
//!
//! ssm 層 (`il % 4 != 3`) は [`FullAttn::forward`] が「担当外」を返す
//! (F2 の [`crate::model::qwen4exp_ssm::SsmLayer`] と組み合わせるのは F5)。
//!
//! # 数値の契約 (参照実装との対応)
//!
//! 1. `attn_q.weight` は q と gate の融合で、head h の `2 * head_dim` 要素のうち
//!    先頭 `head_dim` が q、後半 `head_dim` が gate。`build_layer_attn` の
//!    `Qcur` / `gate` の `ggml_view_3d` (1034 行 / 1052 行) は
//!    `nb1 = elem * n_embd_head * 2` で offset が 0 と `elem * n_embd_head` なので、
//!    head ストライドは `2 * head_dim`。
//! 2. q は head_dim ごとに RMS 正規化してから `attn_q_norm` を掛ける (1039 行)。
//!    k も同じ (`attn_k_norm`、1049 行)。**gate は正規化しない** (生のまま sigmoid)。
//! 3. rope は q と k の先頭 `n_rot` 次元だけ (1062 行 / 1068 行の `ggml_rope_multi`)。
//!    **テキストのみの入力では M-RoPE の 3 軸が同じ位置を指す**ので `sections` は
//!    実質効かず、通常の rope と等価になる。等価性は [`rope_neox_partial`] と
//!    [`rope_mrope_partial`] のビット一致でテストに固定してある。
//!    回転は NeoX 方式 = 対 `(j, j + n_rot/2)`。ggml の `rotate_pairs` は
//!    NEOX / MROPE / IMROPE のいずれでも `n_offset = n_dims/2` で呼ばれる
//!    (`ggml_compute_forward_rope_flt` の 6523-6529 行)。`GGML_ROPE_TYPE_NORMAL`
//!    だけが `n_offset = 1` だが、`ggml_rope_multi` は sections を伴う
//!    MROPE / IMROPE 経路なのでここでは選ばれない。
//! 4. attention は GQA (q head h → kv head `h * n_head_kv / n_head`)、
//!    score = `dot * (1/sqrt(head_dim))` (1078 行の `kq_scale`)、softmax は
//!    max 引きの f32 逐次、出力は j 昇順の逐次加算。
//! 5. 出力ゲートは **`attn_output` の手前** で掛ける。参照実装の順は
//!    `cur = ggml_mul(attn, ggml_sigmoid(gate))` (1089 / 1092 行) →
//!    `build_lora_mm(wo, cur)` (1095 行) なので、掛ける相手は
//!    `head_dim * n_head` の連結ベクトル (2560 次元ではない)。
//!
//! # 重みの持ち方
//!
//! `attn_q` / `attn_k` / `attn_v` / `attn_output` は量子化ブロックのまま
//! [`GgufLinear`] で持つ (f32 へ展開しない)。`attn_q_norm` / `attn_k_norm`
//! (各 `[head_dim]` = 1 KB) だけ f32 常駐。

use anyhow::{bail, Context, Result};

use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::linear_gguf::GgufLinear;
use crate::model::qwen4exp::{LayerAttn, LayerKind, Qwen4ExpError, Qwen4ExpParams};
use crate::model::qwen4exp_qsa::QsaState;
use crate::quant::WeightBlocks;
use crate::tensor::{qknorm_rows_inplace, softmax};

/// 出力ゲートの sigmoid (`expf` = f32 の指数関数)。
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

/// 部分 RoPE (NeoX 方式)。各 head の先頭 `n_rot` 次元の対 `(j, j + n_rot/2)` を
/// 角度 `theta_j = pos * freq_base^(-2j/n_rot)` で回す。`n_rot..head_dim` は触らない。
///
/// 演算列は `x0 * c - x1 * s` / `x0 * s + x1 * c` の順 (左から)。
/// `cos` / `sin` は f32 の [`f32::cos`] / [`f32::sin`] (= `cosf` / `sinf`)。
///
/// 角度は対ごとに `pos * theta^(-2j/n_rot)` を `powf` で出す (リポジトリの
/// `tensor::rope_row` / `qwen35::rope_partial_interleaved` と同じ式・同じ順)。
/// ggml は `theta_scale = powf(freq_base, -2/n_dims)` を 1 回だけ出して
/// 対ごとに掛けていくので、角度の下位ビットが 1 ULP ずれうる。実機の
/// アンカー比較で差が出たら、ここを掛け算の連鎖に置き換える (テストの
/// 期待値も同じ列で作る)。
pub fn rope_neox_partial(
    x: &mut [f32],
    heads: usize,
    dh: usize,
    n_rot: usize,
    pos: usize,
    theta: f32,
) {
    let half = n_rot / 2;
    let posf = pos as f32;
    for h in 0..heads {
        let base = h * dh;
        for j in 0..half {
            let ang = posf * theta.powf(-2.0 * j as f32 / n_rot as f32);
            let (c, s) = (ang.cos(), ang.sin());
            let x0 = x[base + j];
            let x1 = x[base + half + j];
            x[base + j] = x0 * c - x1 * s;
            x[base + half + j] = x0 * s + x1 * c;
        }
    }
}

/// M-RoPE (IMRoPE) 版。[`rope_neox_partial`] と同じ回転で、**対ごとに使う位置**
/// だけを `sections` で切り替える (ggml `ggml_mrope_cache_init` の `is_imrope`
/// 分岐と同じ規則)。
///
/// `sections = [t, h, w, e]`、`sector = j % (t + h + w + e)` として
/// - `sector % 3 == 1 && sector < 3t'` → h 軸
/// - `sector % 3 == 2 && sector < 3w'` → w 軸
/// - `sector % 3 == 0 && sector < 3t'` → t 軸
/// - それ以外 → 追加軸 (e)
///
/// テキストのみの入力では `pos_t == pos_h == pos_w` なので、どの分岐でも同じ
/// 角度になり [`rope_neox_partial`] とビット一致する (テストで固定)。
/// この関数はその等価性の証明用で、本番経路 ([`AttnLayer::norm_rope_q`]) は
/// [`rope_neox_partial`] を呼ぶ。
pub fn rope_mrope_partial(
    x: &mut [f32],
    heads: usize,
    dh: usize,
    n_rot: usize,
    sections: [u32; 4],
    pos_t: usize,
    pos_h: usize,
    pos_w: usize,
    pos_e: usize,
    theta: f32,
) {
    let half = n_rot / 2;
    let s: [usize; 4] = [
        sections[0] as usize,
        sections[1] as usize,
        sections[2] as usize,
        sections[3] as usize,
    ];
    let sect_dims = s[0] + s[1] + s[2] + s[3];
    for h in 0..heads {
        let base = h * dh;
        for j in 0..half {
            let sector = if sect_dims == 0 { 0 } else { j % sect_dims };
            let pos = if sector % 3 == 1 && sector < 3 * s[1] {
                pos_h
            } else if sector % 3 == 2 && sector < 3 * s[2] {
                pos_w
            } else if sector % 3 == 0 && sector < 3 * s[0] {
                pos_t
            } else {
                pos_e
            };
            let ang = pos as f32 * theta.powf(-2.0 * j as f32 / n_rot as f32);
            let (c, sn) = (ang.cos(), ang.sin());
            let x0 = x[base + j];
            let x1 = x[base + half + j];
            x[base + j] = x0 * c - x1 * sn;
            x[base + half + j] = x0 * sn + x1 * c;
        }
    }
}

// ---------------------------------------------------------------------------
// KV キャッシュ
// ---------------------------------------------------------------------------

/// full attention 層の KV キャッシュ。層ごとに `[pos][n_head_kv][head_dim]` を
/// 平坦に持つ (行 = `n_head_kv * head_dim`)。
///
/// 容量は `max_ctx` で固定し、超過は**エラー**にする (黙って捨てない)。
/// 領域は push のたびに必要長まで伸ばすので、ロード時に
/// `max_ctx * 層数 * 2 * row * 4 B` を確保しない。
pub struct AttnCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    len: Vec<usize>,
    n_head_kv: usize,
    head_dim: usize,
    max_ctx: usize,
}

impl AttnCache {
    /// `n_layer` 層ぶんの空きを作る。`max_ctx == 0` は不正。
    pub fn new(n_layer: usize, n_head_kv: usize, head_dim: usize, max_ctx: usize) -> Result<Self> {
        if n_head_kv == 0 || head_dim == 0 {
            bail!("AttnCache: n_head_kv={n_head_kv}, head_dim={head_dim} は不正");
        }
        if max_ctx == 0 {
            bail!("AttnCache: max_ctx=0 は不正 (容量が無い)");
        }
        Ok(Self {
            k: vec![Vec::new(); n_layer],
            v: vec![Vec::new(); n_layer],
            len: vec![0; n_layer],
            n_head_kv,
            head_dim,
            max_ctx,
        })
    }

    /// 1 位置ぶんの行長 = `n_head_kv * head_dim`
    pub fn row(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }
    pub fn n_layer(&self) -> usize {
        self.len.len()
    }
    /// 積まれている位置数
    pub fn len(&self, layer: usize) -> Result<usize> {
        self.len
            .get(layer)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("AttnCache: 層 {layer} は範囲外 (n_layer = {})", self.len.len()))
    }
    pub fn is_empty(&self, layer: usize) -> Result<bool> {
        Ok(self.len(layer)? == 0)
    }

    /// 層 `layer` の k (`len * row` 要素)。
    pub fn k(&self, layer: usize) -> Result<&[f32]> {
        let n = self.len(layer)?;
        Ok(&self.k[layer][..n * self.row()])
    }
    /// 層 `layer` の v (`len * row` 要素)。
    pub fn v(&self, layer: usize) -> Result<&[f32]> {
        let n = self.len(layer)?;
        Ok(&self.v[layer][..n * self.row()])
    }

    /// 末尾に 1 位置ぶん積む。`k` / `v` は `row()` 要素ちょうど。
    pub fn push(&mut self, layer: usize, k: &[f32], v: &[f32]) -> Result<()> {
        let row = self.row();
        let n = self.len(layer)?;
        if k.len() != row || v.len() != row {
            bail!(
                "AttnCache::push: 層 {layer} の k/v 長 {} / {} が row = {row} と違う",
                k.len(),
                v.len()
            );
        }
        if n >= self.max_ctx {
            bail!(
                "AttnCache::push: 層 {layer} が容量 max_ctx = {} に達した (明示的に clear すること)",
                self.max_ctx
            );
        }
        let need = (n + 1) * row;
        if self.k[layer].len() < need {
            self.k[layer].resize(need, 0.0);
            self.v[layer].resize(need, 0.0);
        }
        self.k[layer][n * row..need].copy_from_slice(k);
        self.v[layer][n * row..need].copy_from_slice(v);
        self.len[layer] = n + 1;
        Ok(())
    }

    /// 層 `layer` を空にする (確保済みの領域は残す)。
    pub fn clear(&mut self, layer: usize) -> Result<()> {
        self.len(layer)?;
        self.len[layer] = 0;
        Ok(())
    }

    /// 全層を空にする。
    pub fn clear_all(&mut self) {
        for n in self.len.iter_mut() {
            *n = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// 1 層ぶんの重み
// ---------------------------------------------------------------------------

/// full attention 層 1 つぶんの重み。
pub struct AttnLayer {
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// `1 / sqrt(head_dim)`。参照実装の `kq_scale` で、この arch には
    /// `attention.scale` (`f_attention_scale`) が無いので常にこの値。
    pub kq_scale: f32,
    /// `attn_q.weight` — in `n_embd` → out `2 * n_head * head_dim`
    wq: GgufLinear,
    /// `attn_k.weight` — in `n_embd` → out `n_head_kv * head_dim`
    wk: GgufLinear,
    /// `attn_v.weight` — in `n_embd` → out `n_head_kv * head_dim`
    wv: GgufLinear,
    /// `attn_output.weight` — in `n_head * head_dim` → out `n_embd`
    wo: GgufLinear,
    /// `attn_q_norm.weight` — `[head_dim]` (head 間で共有)
    pub q_norm: Vec<f32>,
    /// `attn_k_norm.weight` — `[head_dim]`
    pub k_norm: Vec<f32>,
}

impl AttnLayer {
    /// `blk.{il}.*` から読む。形状・型の取り違えはここで落とす。
    pub fn from_gguf(f: &GgufFile, p: &Qwen4ExpParams, il: usize) -> Result<Self> {
        let dh = p.n_embd_head_k;
        if p.n_embd_head_v != dh {
            bail!(
                "層 {il}: value_length {} が key_length {dh} と違う (参照実装は等しい前提)",
                p.n_embd_head_v
            );
        }
        if dh == 0 || p.n_head == 0 || p.n_head_kv == 0 {
            bail!(
                "層 {il}: head_dim / head 数が 0 (dh={dh}, n_head={}, n_head_kv={})",
                p.n_head,
                p.n_head_kv
            );
        }
        if p.n_head % p.n_head_kv != 0 {
            bail!(
                "層 {il}: head_count {} が head_count_kv {} で割り切れない",
                p.n_head,
                p.n_head_kv
            );
        }
        if p.n_rot > dh || p.n_rot % 2 != 0 {
            bail!("層 {il}: n_rot {} が head_dim {dh} に対して不正 (偶数かつ <= head_dim)", p.n_rot);
        }

        let name = |s: &str| format!("blk.{il}.{s}");
        let q_out = 2 * p.n_head * dh;
        let kv_out = p.n_head_kv * dh;
        expect_2d(f, &name("attn_q.weight"), p.n_embd, q_out)?;
        expect_2d(f, &name("attn_k.weight"), p.n_embd, kv_out)?;
        expect_2d(f, &name("attn_v.weight"), p.n_embd, kv_out)?;
        expect_2d(f, &name("attn_output.weight"), p.n_head * dh, p.n_embd)?;
        expect_1d(f, &name("attn_q_norm.weight"), dh)?;
        expect_1d(f, &name("attn_k_norm.weight"), dh)?;

        let q_norm = dequant_exact(f, &name("attn_q_norm.weight"), dh)?;
        let k_norm = dequant_exact(f, &name("attn_k_norm.weight"), dh)?;

        Ok(Self {
            n_embd: p.n_embd,
            n_head: p.n_head,
            n_head_kv: p.n_head_kv,
            head_dim: dh,
            n_rot: p.n_rot,
            rope_theta: p.rope_freq_base,
            rms_eps: p.rms_eps,
            kq_scale: 1.0 / (dh as f32).sqrt(),
            wq: linear(f, &name("attn_q.weight"), p.n_embd, q_out)?,
            wk: linear(f, &name("attn_k.weight"), p.n_embd, kv_out)?,
            wv: linear(f, &name("attn_v.weight"), p.n_embd, kv_out)?,
            wo: linear(f, &name("attn_output.weight"), p.n_head * dh, p.n_embd)?,
            q_norm,
            k_norm,
        })
    }

    /// q の長さ = `n_head * head_dim` (= `attn_output` の入力長)
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    /// k / v の長さ = `n_head_kv * head_dim`
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    /// `attn_q` の出力長 = `2 * n_head * head_dim` (q と gate の融合)
    pub fn qg_dim(&self) -> usize {
        2 * self.q_dim()
    }
    /// q head `h` が読む kv head (`h * n_head_kv / n_head`)
    pub fn kv_head(&self, h: usize) -> usize {
        h * self.n_head_kv / self.n_head
    }

    /// `attn_q` の投影 (q と gate が交互に並ぶ)。
    pub fn project_q_gate(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.qg_dim()];
        self.wq.gemv(cur, &mut out)?;
        Ok(out)
    }
    /// `attn_k` の投影
    pub fn project_k(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.kv_dim()];
        self.wk.gemv(cur, &mut out)?;
        Ok(out)
    }
    /// `attn_v` の投影
    pub fn project_v(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.kv_dim()];
        self.wv.gemv(cur, &mut out)?;
        Ok(out)
    }

    /// `attn_q` の出力を q と gate に割る。head h の 2*dh 要素のうち
    /// 先頭 dh が q、後半 dh が gate。
    pub fn split_q_gate(&self, qg: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if qg.len() != self.qg_dim() {
            bail!(
                "split_q_gate: 入力長 {} が 2 * n_head * head_dim = {} と違う",
                qg.len(),
                self.qg_dim()
            );
        }
        let dh = self.head_dim;
        let mut q = vec![0f32; self.q_dim()];
        let mut gate = vec![0f32; self.q_dim()];
        for h in 0..self.n_head {
            let base = h * 2 * dh;
            q[h * dh..(h + 1) * dh].copy_from_slice(&qg[base..base + dh]);
            gate[h * dh..(h + 1) * dh].copy_from_slice(&qg[base + dh..base + 2 * dh]);
        }
        Ok((q, gate))
    }

    /// q に head_dim ごとの RMS 正規化 (`attn_q_norm`) と rope を掛ける (in-place)。
    pub fn norm_rope_q(&self, q: &mut [f32], pos: usize) -> Result<()> {
        if q.len() != self.q_dim() {
            bail!("norm_rope_q: 長さ {} が n_head * head_dim = {} と違う", q.len(), self.q_dim());
        }
        qknorm_rows_inplace(q, 1, self.n_head, self.head_dim, &self.q_norm, self.rms_eps);
        rope_neox_partial(q, self.n_head, self.head_dim, self.n_rot, pos, self.rope_theta);
        Ok(())
    }

    /// k に head_dim ごとの RMS 正規化 (`attn_k_norm`) と rope を掛ける (in-place)。
    /// v には掛けない。
    pub fn norm_rope_k(&self, k: &mut [f32], pos: usize) -> Result<()> {
        if k.len() != self.kv_dim() {
            bail!("norm_rope_k: 長さ {} が n_head_kv * head_dim = {} と違う", k.len(), self.kv_dim());
        }
        qknorm_rows_inplace(k, 1, self.n_head_kv, self.head_dim, &self.k_norm, self.rms_eps);
        rope_neox_partial(k, self.n_head_kv, self.head_dim, self.n_rot, pos, self.rope_theta);
        Ok(())
    }

    /// 正規化・rope 済みの q と、層 `layer` のキャッシュ `0..len` で GQA attention。
    /// **見るのは積まれた本数 (`len`) だけ** — 未来の位置はキャッシュに無いので
    /// スコアに入らない (因果性はここで閉じる)。
    ///
    /// 実体は [`AttnLayer::attend_impl`] (`cells = None` = 全範囲)。dense 経路の
    /// 丸め順序 (dot は k 昇順、softmax は max 引きの逐次、出力は j 昇順の逐次加算)
    /// は G2 以前と同じで、値はビット一致する。
    pub fn attend(&self, q: &[f32], cache: &AttnCache, layer: usize) -> Result<Vec<f32>> {
        let len = cache.len(layer)?;
        self.attend_impl(q, cache, layer, None, len)
    }

    /// 見る cell を `cells` に限定した attention (QSA の選択結果を食わせる口)。
    ///
    /// `cells` は **昇順・重複なし・すべて `cache.len(layer)` 未満** でなければ
    /// エラー (並べ替えて黙って通さない)。softmax の和も出力の積算も `cells` の
    /// 昇順で回すので、`cells == 0..len` のとき [`AttnLayer::attend`] と
    /// ビット一致する (テスト A)。
    pub fn attend_visible(
        &self,
        q: &[f32],
        cache: &AttnCache,
        layer: usize,
        cells: &[usize],
    ) -> Result<Vec<f32>> {
        let len = cache.len(layer)?;
        if cells.is_empty() {
            bail!("attend_visible: 見る cell が空 (層 {layer})");
        }
        let mut prev: Option<usize> = None;
        for &c in cells {
            if c >= len {
                bail!("attend_visible: cell {c} が積まれた本数 {len} の範囲外 (層 {layer})");
            }
            if let Some(p) = prev {
                if c <= p {
                    bail!("attend_visible: cells が昇順でない ({p} の次が {c}、層 {layer})");
                }
            }
            prev = Some(c);
        }
        self.attend_impl(q, cache, layer, Some(cells), len)
    }

    /// `attend` / `attend_visible` の共通本体。
    ///
    /// `cells = None` のとき `i -> i` (全範囲)、`Some(c)` のとき `i -> c[i]`。
    /// どちらも `i` 昇順に同じ式を回すので、`c == 0..len` なら dense と
    /// ビット一致する。
    fn attend_impl(
        &self,
        q: &[f32],
        cache: &AttnCache,
        layer: usize,
        cells: Option<&[usize]>,
        len: usize,
    ) -> Result<Vec<f32>> {
        if q.len() != self.q_dim() {
            bail!("attend: q の長さ {} が n_head * head_dim = {} と違う", q.len(), self.q_dim());
        }
        let row = cache.row();
        if row != self.kv_dim() {
            bail!("attend: キャッシュの row = {row} が kv_dim = {} と違う", self.kv_dim());
        }
        let kb = cache.k(layer)?;
        let vb = cache.v(layer)?;
        let dh = self.head_dim;
        let n = match cells {
            Some(c) => c.len(),
            None => len,
        };
        let at = |i: usize| match cells {
            Some(c) => c[i],
            None => i,
        };
        let mut out = vec![0f32; self.q_dim()];
        let mut scores = vec![0f32; n];
        for h in 0..self.n_head {
            let hk = self.kv_head(h);
            for i in 0..n {
                let j = at(i);
                let krow = &kb[j * row + hk * dh..][..dh];
                let mut acc = 0f32;
                for d in 0..dh {
                    acc += q[h * dh + d] * krow[d];
                }
                scores[i] = acc * self.kq_scale;
            }
            // max 引きの f32 逐次 softmax (expf)
            softmax(&mut scores);
            for i in 0..n {
                let j = at(i);
                let vrow = &vb[j * row + hk * dh..][..dh];
                let p = scores[i];
                for d in 0..dh {
                    out[h * dh + d] += p * vrow[d];
                }
            }
        }
        Ok(out)
    }

    /// 出力ゲート: `attn * sigmoid(gate)` (要素ごと)。
    pub fn gate_mul(&self, attn: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        if attn.len() != self.q_dim() || gate.len() != self.q_dim() {
            bail!(
                "gate_mul: 長さ {} / {} が n_head * head_dim = {} と違う",
                attn.len(),
                gate.len(),
                self.q_dim()
            );
        }
        Ok(attn
            .iter()
            .zip(gate.iter())
            .map(|(a, g)| a * sigmoid(*g))
            .collect())
    }

    /// `attn_output` の投影 (`n_head * head_dim` → `n_embd`)。
    pub fn project_out(&self, x: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.n_embd];
        self.wo.gemv(x, &mut out)?;
        Ok(out)
    }

    /// 1 トークンぶん通す: 投影 → 切り分け → 正規化+rope → キャッシュに積む →
    /// attention → 出力ゲート → `attn_output`。
    pub fn forward_token(
        &mut self,
        cur: &[f32],
        cache: &mut AttnCache,
        layer: usize,
        pos: usize,
    ) -> Result<Vec<f32>> {
        self.forward_token_impl(cur, cache, layer, pos, None)
    }

    /// [`AttnLayer::forward_token`] の「見る cell を限定する」版。
    /// attention だけ [`AttnLayer::attend_visible`] に差し替える (投影・rope・
    /// ゲート・`attn_output` は同じ)。
    pub fn forward_token_visible(
        &mut self,
        cur: &[f32],
        cache: &mut AttnCache,
        layer: usize,
        pos: usize,
        cells: &[usize],
    ) -> Result<Vec<f32>> {
        self.forward_token_impl(cur, cache, layer, pos, Some(cells))
    }

    fn forward_token_impl(
        &mut self,
        cur: &[f32],
        cache: &mut AttnCache,
        layer: usize,
        pos: usize,
        cells: Option<&[usize]>,
    ) -> Result<Vec<f32>> {
        if cur.len() != self.n_embd {
            bail!(
                "AttnLayer::forward_token: 入力長 {} が n_embd = {} と違う",
                cur.len(),
                self.n_embd
            );
        }
        let qg = self.project_q_gate(cur)?;
        let (mut q, gate) = self.split_q_gate(&qg)?;
        self.norm_rope_q(&mut q, pos)?;

        let mut k = self.project_k(cur)?;
        self.norm_rope_k(&mut k, pos)?;
        let v = self.project_v(cur)?;

        // 積んでから全 len に対して attention (参照実装の KV 書き込みと同じ順)。
        cache.push(layer, &k, &v)?;
        let attn = match cells {
            None => self.attend(&q, cache, layer)?,
            Some(c) => self.attend_visible(&q, cache, layer, c)?,
        };
        let gated = self.gate_mul(&attn, &gate)?;
        self.project_out(&gated)
    }

    /// 1 トークンあたりの重みバイト数 (量子化ブロック + norm)。
    pub fn weight_bytes(&self) -> usize {
        self.wq.weight_bytes()
            + self.wk.weight_bytes()
            + self.wv.weight_bytes()
            + self.wo.weight_bytes()
            + self.q_norm.len() * 4
            + self.k_norm.len() * 4
    }
}

// ---------------------------------------------------------------------------
// 全層をまとめる差し替え口
// ---------------------------------------------------------------------------

/// full attention 層 12 本をまとめ、[`LayerAttn`] として回す。
///
/// ssm 層 (`layers[il] == None`) に来たら「担当外」を返す。F2 の
/// [`crate::model::qwen4exp_ssm::SsmLayer`] との合成は F5 で行う。
pub struct FullAttn {
    n_embd: usize,
    layers: Vec<Option<AttnLayer>>,
    cache: AttnCache,
    pos: usize,
    /// full attention 層ごとに「このトークンで既に積んだか」。同じ層を 1 トークンで
    /// 二度通したときに黙って二重 push しないための番人。全層が揃ったら false に戻す。
    pushed: Vec<bool>,
    /// QSA (indexer による key 選択)。`DENNO_QSA=1` のときだけ `Some`。
    /// 既定 (`DENNO_QSA` 未設定 / 0) は `None` で、indexer のテンソルは読まない
    /// (dense 経路そのまま・追加確保なし)。
    qsa: Option<QsaState>,
}

/// `DENNO_QSA=1` のときだけ QSA 選択を有効にする (既定 0 = dense 経路そのまま)。
/// `"1"` 以外 (`"true"` / `"yes"` など) は無効として扱う (取り違えを増やさない)。
fn qsa_from_env() -> bool {
    matches!(std::env::var("DENNO_QSA").as_deref(), Ok("1"))
}

impl FullAttn {
    /// 実 GGUF から組む。full attention 層のテンソルだけを読む
    /// (ssm 層のテンソルは F2 の担当)。
    ///
    /// QSA は環境変数 `DENNO_QSA` で切り替える (`1` のときだけ indexer を組む)。
    pub fn from_gguf(f: &GgufFile, p: &Qwen4ExpParams, max_ctx: usize) -> Result<Self> {
        Self::from_gguf_with(f, p, max_ctx, qsa_from_env())
    }

    /// QSA の有無を明示する版 (テストが環境変数に触らずに両経路を比べるため)。
    pub fn from_gguf_with(
        f: &GgufFile,
        p: &Qwen4ExpParams,
        max_ctx: usize,
        use_qsa: bool,
    ) -> Result<Self> {
        if p.full_attn_interval == 0 {
            bail!("FullAttn::from_gguf: full_attention_interval = 0 では層を選べない");
        }
        let mut layers = Vec::with_capacity(p.n_layer);
        let mut n_full = 0usize;
        for il in 0..p.n_layer {
            match p.layer_kind(il)? {
                LayerKind::FullAttn => {
                    layers.push(Some(
                        AttnLayer::from_gguf(f, p, il)
                            .with_context(|| format!("層 {il} (full attention)"))?,
                    ));
                    n_full += 1;
                }
                LayerKind::Ssm => layers.push(None),
            }
        }
        let want = (p.n_layer + p.full_attn_interval - 1) / p.full_attn_interval;
        if n_full != want {
            bail!(
                "FullAttn::from_gguf: full attention 層が {n_full} 本 (期待 {want} 本 = block_count {} / interval {})",
                p.n_layer,
                p.full_attn_interval
            );
        }
        let cache = AttnCache::new(p.n_layer, p.n_head_kv, p.n_embd_head_k, max_ctx)?;
        let layers_len = layers.len();
        // QSA を切っているときは indexer のテンソルに触らない (dense 経路の前提を変えない)。
        let qsa = if use_qsa {
            Some(QsaState::from_gguf(f, p, max_ctx).context("FullAttn: QSA を組めない")?)
        } else {
            None
        };
        Ok(Self {
            n_embd: p.n_embd,
            layers,
            cache,
            pos: 0,
            pushed: vec![false; layers_len],
            qsa,
        })
    }

    /// 次に積む位置 (= これまでに通したトークン数)
    pub fn pos(&self) -> usize {
        self.pos
    }
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn cache(&self) -> &AttnCache {
        &self.cache
    }
    pub fn cache_mut(&mut self) -> &mut AttnCache {
        &mut self.cache
    }
    /// 層 `il` が full attention なら重みを返す (ssm 層・範囲外は `None`)。
    pub fn layer(&self, il: usize) -> Option<&AttnLayer> {
        self.layers.get(il).and_then(|l| l.as_ref())
    }
    /// 層 `il` が full attention の担当か。
    pub fn is_full(&self, il: usize) -> bool {
        matches!(self.layers.get(il), Some(Some(_)))
    }
    /// キャッシュを空にし、位置も 0 に戻す。
    pub fn reset(&mut self) {
        self.cache.clear_all();
        self.pos = 0;
        self.pushed.iter_mut().for_each(|b| *b = false);
        if let Some(q) = self.qsa.as_mut() {
            q.reset();
        }
    }

    /// QSA の状態 (無効なら `None`)。
    pub fn qsa(&self) -> Option<&QsaState> {
        self.qsa.as_ref()
    }

    /// QSA の重みバイト数 (無効なら 0)。呼び出し側が容量を見積もるための値で、
    /// 生キー / pool の作業域は含まない (それは文脈長に比例する)。
    pub fn qsa_weight_bytes(&self) -> usize {
        self.qsa.as_ref().map_or(0, |q| q.weight_bytes())
    }

    /// [`LayerAttn::forward`] の anyhow 版 (テストと F5 の合成から使う)。
    pub fn forward_token(&mut self, layer: usize, cur: &[f32]) -> Result<Vec<f32>> {
        if cur.len() != self.n_embd {
            bail!(
                "FullAttn::forward: 入力長 {} が n_embd = {} と違う",
                cur.len(),
                self.n_embd
            );
        }
        let Some(slot) = self.layers.get(layer) else {
            bail!("FullAttn::forward: 層 {layer} は範囲外 (n_layer = {})", self.layers.len());
        };
        if slot.is_none() {
            bail!("FullAttn::forward: 層 {layer} は ssm 層で full attention の担当外 (F5 で合成する)");
        }
        let pos = self.pos;
        // 番人を先に見る。二重 push のときはキャッシュ長が pos+1 になるので、
        // 長さの検査を先に置くと「既に積んである」が影に隠れてしまう。
        if self.pushed[layer] {
            bail!("FullAttn::forward: 層 {layer} はこのトークン (pos {pos}) で既に積んである");
        }
        let len = self.cache.len(layer)?;
        if len != pos {
            bail!("FullAttn::forward: 層 {layer} のキャッシュ長 {len} が pos {pos} と違う");
        }
        let l = self.layers[layer].as_mut().expect("上の is_none で除外済み");
        // QSA 層なら、このトークンの生キーを indexer に積んで見える cell を貰う。
        // 積むのは attention の投影より前 (どちらも `cur` だけに依存し、順序は
        // 数値に影響しない)。QSA 無効時は何も確保しない。
        let cells = match self.qsa.as_mut() {
            Some(q) if q.is_qsa(layer) => Some(q.push_token(layer, cur, pos)?),
            _ => None,
        };
        let out = match cells.as_deref() {
            Some(c) => l.forward_token_visible(cur, &mut self.cache, layer, pos, c)?,
            None => l.forward_token(cur, &mut self.cache, layer, pos)?,
        };
        self.pushed[layer] = true;
        // 位置は 1 トークンにつき 1 回だけ進める。full attention 層は 1 トークンで
        // 12 本ぶん呼ばれるので、全層が揃ったときだけ進める。
        let all = (0..self.layers.len()).all(|il| self.layers[il].is_none() || self.pushed[il]);
        if all {
            self.pos = pos + 1;
            self.pushed.iter_mut().for_each(|b| *b = false);
        }
        Ok(out)
    }
}

impl LayerAttn for FullAttn {
    fn forward(
        &mut self,
        layer: usize,
        cur: &[f32],
    ) -> std::result::Result<Vec<f32>, Qwen4ExpError> {
        self.forward_token(layer, cur)
            .map_err(|e| Qwen4ExpError(vec![format!("{e:#}")]))
    }
}

// ---------------------------------------------------------------------------
// GGUF からの読み出し
// ---------------------------------------------------------------------------

/// `(ne0, ne1)` が期待どおりで、余次元が 1 であることを確かめる。
fn expect_2d(f: &GgufFile, name: &str, k: usize, n: usize) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF に無い"))?;
    if t.dims[0] as usize != k || t.dims[1] as usize != n {
        bail!("{name}: ne={:?} が期待 [{k}, {n}] と違う", &t.dims[..2]);
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

/// 量子化ブロックを保持したまま [`GgufLinear`] を作る (mmap のスライスを共有)。
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
