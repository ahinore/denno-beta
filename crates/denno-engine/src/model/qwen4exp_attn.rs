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

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AtomicOrdering};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};

use crate::gguf::GgufFile;
use crate::kquant::GgmlType;
use crate::model::linear_gguf::GgufLinear;
use crate::model::qwen4exp::{
    gemv_batch, q4prof, LayerAttn, LayerKind, Qwen4ExpError, Qwen4ExpParams,
};
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
    // half が表の上限を超える形だけ従来版に落とす (実形状の n_rot は 64 = half 32)。
    if rope_tab_on() && n_rot / 2 <= ROPE_TAB_MAX {
        rope_neox_partial_tab(x, heads, dh, n_rot, pos, theta);
    } else {
        rope_neox_partial_serial(x, heads, dh, n_rot, pos, theta);
    }
}

/// 表の上限 (対の本数)。`n_rot <= 2 * ROPE_TAB_MAX` まで表引き版を使う。
const ROPE_TAB_MAX: usize = 128;

/// 表引き (`DENNO_PF_ATT_ROPE_TAB=0` で従来版・既定 ON)。A/B 用。
static ROPE_TAB: OnceLock<AtomicBool> = OnceLock::new();

fn rope_tab_on() -> bool {
    ROPE_TAB
        .get_or_init(|| AtomicBool::new(std::env::var("DENNO_PF_ATT_ROPE_TAB").ok().as_deref() != Some("0")))
        .load(AtomicOrdering::Relaxed)
}

/// 従来版 (基準実装): 対ごとの `powf` / `cos` / `sin` を **head ごとに** 払い直す。
///
/// 実形状 (heads 24 / half 32) では 1 トークンあたり 768 回の超越関数になる。
/// 値の基準なので消さず、`DENNO_PF_ATT_ROPE_TAB=0` とテストの比較対象に残す。
fn rope_neox_partial_serial(
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

/// 表引き版: 角度 `pos * theta^(-2j/n_rot)` と `cos` / `sin` は **head に
/// 依存しない**ので、対ごとに 1 回だけ出して全 head で使い回す。
///
/// 1 要素の演算列 (`x0*c - x1*s` / `x0*s + x1*c`、左から) と角度の式は
/// 従来版と同一の式・同一の f32 演算なので、出力はビット一致する
/// (テスト `rope_tab_matches_serial_bitexact`)。
fn rope_neox_partial_tab(
    x: &mut [f32],
    heads: usize,
    dh: usize,
    n_rot: usize,
    pos: usize,
    theta: f32,
) {
    let half = n_rot / 2;
    if half == 0 {
        return;
    }
    let posf = pos as f32;
    // (c, s) の表。half <= ROPE_TAB_MAX は呼び出し側で保証済み。
    let mut cs = [0f32; ROPE_TAB_MAX * 2];
    for j in 0..half {
        let ang = posf * theta.powf(-2.0 * j as f32 / n_rot as f32);
        cs[j * 2] = ang.cos();
        cs[j * 2 + 1] = ang.sin();
    }
    for h in 0..heads {
        let base = h * dh;
        for j in 0..half {
            let (c, s) = (cs[j * 2], cs[j * 2 + 1]);
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

/// [`AttnLayer::attend_batch`] が走った回数。
///
/// 緑 = 経路が発動した証拠ではないので、テストが「バッチ版が実際に呼ばれた」
/// ことをこれで見る。層×チャンクで 1 回だけ増える (relaxed の fetch_add) ので
/// 実測への影響は無視できる。
static ATTN_BATCH_CALLS: AtomicUsize = AtomicUsize::new(0);

/// [`AttnLayer::attend_batch`] が走った回数 (プロセス通算)。
pub fn attend_batch_calls() -> usize {
    ATTN_BATCH_CALLS.load(AtomicOrdering::Relaxed)
}

/// [`AttnLayer::attend_batch`] がトークン方向の rayon 経路を走らせた回数。
///
/// 緑 = 経路が発動した証拠ではないので、テストが「並列の枝が実際に通った」
/// ことをこれで見る (層×チャンクで 1 回)。
static ATTN_BATCH_PAR_CALLS: AtomicUsize = AtomicUsize::new(0);

/// [`AttnLayer::attend_batch`] が並列経路を走らせた回数 (プロセス通算)。
pub fn attend_batch_par_calls() -> usize {
    ATTN_BATCH_PAR_CALLS.load(AtomicOrdering::Relaxed)
}

/// トークン方向の並列化を始める最小 `nt` (既定 8)。
///
/// 1 トークンぶんの計算は head_dim 256 × 24 head × 位置数で、nt が 1 桁だと
/// rayon の分割コスト (µs 台) が仕事量に比べて無視できない。実測 (CPU) で
/// nt = 8 前後から並列が勝ち始めたのでここを既定にした
/// (`crates/denno-engine/tests/qwen4exp_attn_batch.rs` のベンチ)。
const ATTN_PAR_MIN_DEFAULT: usize = 8;

static ATTN_PAR_MIN: AtomicUsize = AtomicUsize::new(ATTN_PAR_MIN_DEFAULT);

/// 現在の並列化しきい値 (`nt` がこれ未満なら逐次)。
pub fn attend_par_min() -> usize {
    ATTN_PAR_MIN.load(AtomicOrdering::Relaxed)
}

/// しきい値を差し替え、以前の値を返す。
///
/// ベンチが「バッチ逐次」を測るためだけに使う (`usize::MAX` で強制逐次)。
/// 本番経路は既定値のまま。
pub fn set_attend_par_min(nt: usize) -> usize {
    ATTN_PAR_MIN.swap(nt, AtomicOrdering::Relaxed)
}

/// decode の GQA まとめ経路 (`DENNO_ATTN_GQA_FAST=0` で従来の head 個別版)。
///
/// 既定 ON。head ごとの加算順 (内積は d 昇順、出力は j 昇順) を変えないので
/// ビット一致が保てる。env は最初の 1 回だけ読む (プロセス内で固定)。
static ATTN_GQA_FAST: OnceLock<AtomicBool> = OnceLock::new();

fn attn_gqa_fast() -> bool {
    ATTN_GQA_FAST
        .get_or_init(|| {
            AtomicBool::new(std::env::var("DENNO_ATTN_GQA_FAST").ok().as_deref() != Some("0"))
        })
        .load(AtomicOrdering::Relaxed)
}

/// decode の GQA まとめ経路が走った回数 (プロセス通算)。
///
/// 緑 = 経路が発動した証拠ではないので、テストが「まとめ経路が実際に
/// 呼ばれた」ことをこれで見る (層 × トークンで 1 回)。
static ATTN_GQA_CALLS: AtomicUsize = AtomicUsize::new(0);

/// [`ATTN_GQA_CALLS`] の読み出し。
pub fn attend_gqa_calls() -> usize {
    ATTN_GQA_CALLS.load(AtomicOrdering::Relaxed)
}

/// まとめ経路のうち rayon の並列枝を走らせた回数 (プロセス通算)。
static ATTN_GQA_PAR_CALLS: AtomicUsize = AtomicUsize::new(0);

/// [`ATTN_GQA_PAR_CALLS`] の読み出し。
pub fn attend_gqa_par_calls() -> usize {
    ATTN_GQA_PAR_CALLS.load(AtomicOrdering::Relaxed)
}

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
        self.wq.gemv_label("qkv", cur, &mut out)?;
        Ok(out)
    }
    /// `attn_k` の投影
    pub fn project_k(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.kv_dim()];
        self.wk.gemv_label("qkv", cur, &mut out)?;
        Ok(out)
    }
    /// `attn_v` の投影
    pub fn project_v(&self, cur: &[f32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; self.kv_dim()];
        self.wv.gemv_label("qkv", cur, &mut out)?;
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

    /// [`AttnLayer::attend`] を GQA まとめ経路の on/off 明示で回す (テスト・A/B 用)。
    ///
    /// `gqa_fast = false` は `DENNO_ATTN_GQA_FAST=0` と同じ従来の head 個別経路。
    /// どちらも同じ値になる (テスト `attend_group_matches_per_head_bits`)。
    #[doc(hidden)]
    pub fn attend_gqa_mode(
        &self,
        q: &[f32],
        cache: &AttnCache,
        layer: usize,
        gqa_fast: bool,
    ) -> Result<Vec<f32>> {
        let len = cache.len(layer)?;
        self.attend_impl_mode(q, cache, layer, None, len, gqa_fast)
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
        self.attend_impl_mode(q, cache, layer, cells, len, attn_gqa_fast())
    }

    /// [`AttnLayer::attend_impl`] の本体。`gqa_fast` は decode のまとめ経路の
    /// on/off で、本番は env `DENNO_ATTN_GQA_FAST`、テストは明示指定で両経路を
    /// 比べる (テストが env のプロセス内キャッシュに触らないようにするため)。
    #[allow(clippy::too_many_arguments)]
    fn attend_impl_mode(
        &self,
        q: &[f32],
        cache: &AttnCache,
        layer: usize,
        cells: Option<&[usize]>,
        len: usize,
        gqa_fast: bool,
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
        // decode (cells = None) は GQA まとめ経路: kv head ごとに q head を
        // まとめ、KV の行を 1 回だけ読む。`cells` 付き (QSA) は位置の対応が
        // 飛び番になるので従来の head 個別版のまま (どちらもビット一致)。
        if cells.is_none() && gqa_fast {
            ATTN_GQA_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
            return Ok(self.attend_dense_grouped(q, kb, vb, row, len));
        }
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

    /// 1 つの kv head を共有する q head 数 (`n_head / n_head_kv`)。
    /// [`AttnLayer::from_gguf`] が割り切れることを検査済み。
    fn heads_per_kv(&self) -> usize {
        self.n_head / self.n_head_kv
    }

    /// 1 トークンの attention を kv head グループ単位で回す ([`AttnLayer::attend_impl`]
    /// の decode 用まとめ経路、`DENNO_ATTN_GQA_FAST=0` で従来版)。
    ///
    /// q head を「同じ kv head を共有する範囲」に区切り、その中を
    /// [`gqa_chunk_heads`] ごとのタスクに割って rayon に渡す。各 (head, d) の
    /// 出力は j 昇順の逐次加算のまま、各 (head, j) の内積は d 昇順のままなので、
    /// **スレッド数にも分割の仕方にも依存せず**従来の head 個別版とビット一致する
    /// (変わるのは「どの head をどのスレッドが計算するか」だけ)。
    ///
    /// タスクは同じ kv head の中だけで切るので、1 タスクが 2 つの kv head を
    /// またぐことはない (`chunk` は `heads_per_kv()` の約数)。
    fn attend_dense_grouped(
        &self,
        q: &[f32],
        kb: &[f32],
        vb: &[f32],
        row: usize,
        len: usize,
    ) -> Vec<f32> {
        let dh = self.head_dim;
        let ng = self.heads_per_kv();
        let mut out = vec![0f32; self.q_dim()];
        if len == 0 {
            return out;
        }
        let chunk = gqa_chunk_heads(ng);
        let ntasks = self.n_head / chunk;
        debug_assert_eq!(ntasks * chunk, self.n_head);

        #[cfg(feature = "par")]
        if ntasks > 1 && rayon::current_num_threads() > 1 {
            use rayon::prelude::*;

            ATTN_GQA_PAR_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
            out.par_chunks_mut(chunk * dh)
                .enumerate()
                .for_each_init(|| vec![0f32; chunk * len], |scores, (t, obuf)| {
                    let h0 = t * chunk;
                    attend_group_into(
                        q,
                        kb,
                        vb,
                        row,
                        len,
                        h0 / ng,
                        h0,
                        chunk,
                        dh,
                        self.kq_scale,
                        h0,
                        scores,
                        obuf,
                    );
                });
            return out;
        }

        // 逐次 (rayon 無効 / 1 スレッド / タスク 1 本)。並列と同じ関数を回す。
        let mut scores = vec![0f32; chunk * len];
        for t in 0..ntasks {
            let h0 = t * chunk;
            attend_group_into(
                q,
                kb,
                vb,
                row,
                len,
                h0 / ng,
                h0,
                chunk,
                dh,
                self.kq_scale,
                h0,
                &mut scores,
                &mut out[h0 * dh..(h0 + chunk) * dh],
            );
        }
        out
    }

    /// チャンクぶんの attention を回す (バッチ prefill 用)。
    ///
    /// `qs` は `nt * q_dim` (トークン順、各 head は正規化 + rope 済み)。
    /// 呼ぶ前に層 `layer` のキャッシュへ今回の `nt` 行を push してあること
    /// (末尾 `nt` 行が今回のぶん)。トークン `t` が見るのは `0..=(base + t)`
    /// (`base = cache.len(layer) - nt`) で、[`AttnLayer::attend`] を
    /// `len = base + t + 1` で呼んだ場合と同じ範囲 (因果性はここで閉じる)。
    ///
    /// 丸め順序の契約 (1 トークン版と同一):
    /// - スコアの内積は d 昇順 (`acc += q[d] * k[d]`)、`* kq_scale` は最後に 1 回
    /// - softmax は [`softmax`] (max 引き → exp → 前からの和)
    /// - 出力の重み付き和は **j 昇順**の逐次加算 (`out[d] += p * v[d]`)
    ///
    /// # 並列化 (トークン方向)
    ///
    /// トークン `t` が見る範囲は `0..=(base + t)` で閉じていて、トークン同士は
    /// 状態を共有しない (KV キャッシュは読むだけ)。1 トークンぶんの計算を
    /// [`AttnLayer::attend_one_into`] に切り出し、`0..nt` を `par_chunks_mut` で
    /// 割る。各 (t, head, d) の加算列は j 昇順、各 (t, 位置, head) の内積は
    /// d 昇順のまま変わらないので、**スレッド数にも分割の仕方にも依存せず
    /// ビット一致する**。変わるのは「どのトークンをどのスレッドが計算するか」
    /// だけ。KV は (トークン, kv head) ごとに 1 回だけ読む (GQA の q head は
    /// 同じ行を共有する)。
    ///
    /// `nt` が [`attend_par_min`] 未満のときは**同じ関数**をトークン順に回す
    /// (rayon の分割コストが見合わない)。経路が違っても計算は同一なので値は
    /// 動かない。呼び出し側が既に rayon の中でも work-stealing でそのまま走る。
    pub fn attend_batch(
        &self,
        qs: &[f32],
        nt: usize,
        cache: &AttnCache,
        layer: usize,
    ) -> Result<Vec<f32>> {
        let qd = self.q_dim();
        if qs.len() != nt * qd {
            bail!(
                "attend_batch: q の長さ {} が nt({nt}) * q_dim({qd}) と違う",
                qs.len()
            );
        }
        let row = cache.row();
        if row != self.kv_dim() {
            bail!(
                "attend_batch: キャッシュの row = {row} が kv_dim = {} と違う",
                self.kv_dim()
            );
        }
        let len = cache.len(layer)?;
        if nt > len {
            bail!("attend_batch: nt({nt}) が積まれている本数 {len} より多い (層 {layer})");
        }
        let base = len - nt;
        let mut out = vec![0f32; nt * qd];
        if nt == 0 {
            return Ok(out);
        }
        ATTN_BATCH_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
        let kb = cache.k(layer)?;
        let vb = cache.v(layer)?;

        // 作業バッファは 1 トークンぶん (1 kv head を共有する q head 数 × 可視位置の最大)。
        let scratch_len = self.heads_per_kv() * len;

        // トークン方向の並列。`nt` がしきい値以上なら rayon に割る。
        #[cfg(feature = "par")]
        if nt >= attend_par_min() {
            use rayon::prelude::*;

            ATTN_BATCH_PAR_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
            // `for_each_init` でスレッドごとに 1 本の作業バッファを使い回す。
            out.par_chunks_mut(qd)
                .enumerate()
                .for_each_init(|| vec![0f32; scratch_len], |scores, (t, o)| {
                    self.attend_one_into(
                        &qs[t * qd..(t + 1) * qd],
                        kb,
                        vb,
                        row,
                        base + t + 1,
                        scores,
                        o,
                    );
                });
            return Ok(out);
        }

        // 逐次 (nt が小さいとき / `par` 無効時)。並列と同じ関数をトークン順に回す。
        let mut scores = vec![0f32; scratch_len];
        for t in 0..nt {
            self.attend_one_into(
                &qs[t * qd..(t + 1) * qd],
                kb,
                vb,
                row,
                base + t + 1,
                &mut scores,
                &mut out[t * qd..(t + 1) * qd],
            );
        }
        Ok(out)
    }

    /// 1 トークンぶんの GQA attention を `out` に書く
    /// ([`AttnLayer::attend_batch`] の本体)。
    ///
    /// `nvis` = 見る位置の本数 (トークン `t` なら `base + t + 1`)。キャッシュの
    /// `0..nvis` 行だけを見る (因果性は呼び出し側が閉じる)。加算順は 1 トークン版
    /// と同一: 内積は d 昇順 → `* kq_scale` 1 回、[`softmax`] は前からの和、
    /// 出力は j 昇順の逐次加算。
    ///
    /// `scores` は `ng * nvis` 以上 (ng = 1 kv head を共有する q head 数)、`out` は
    /// `q_dim` で、呼ぶ前に 0 で埋まっていること。トークン内では kv head ごとに
    /// k / v の行を 1 回だけ読み、その kv head を共有する q head をまとめて回す。
    #[allow(clippy::too_many_arguments)]
    fn attend_one_into(
        &self,
        qrow: &[f32],
        kb: &[f32],
        vb: &[f32],
        row: usize,
        nvis: usize,
        scores: &mut [f32],
        out: &mut [f32],
    ) {
        debug_assert_eq!(qrow.len(), self.q_dim());
        debug_assert_eq!(out.len(), self.q_dim());
        debug_assert!(nvis >= 1);
        let ng = self.heads_per_kv();
        for hk in 0..self.n_head_kv {
            debug_assert!(scores.len() >= ng * nvis);
            attend_group_into(
                qrow,
                kb,
                vb,
                row,
                nvis,
                hk,
                hk * ng,
                ng,
                self.head_dim,
                self.kq_scale,
                0,
                scores,
                out,
            );
            // 次のグループは自分の `ng * nvis` だけを使う (先頭から書き直す)。
        }
    }
}

/// 1 タスクぶんの q head 数 (GQA まとめ経路の分割単位)。
///
/// 出力は head ごとに独立に足すので、足し算の順序を保つには **head 境界で
/// しか割れない**。細かく割るほど並列度は上がるが、そのぶん同じ kv 行を
/// タスク数だけ読み直す。実形状 (12 head 共有) では半分の 6 head = 4 タスクに
/// すると、KV の再読みが 2 倍・タスクが 4 本になり、読み出し増 (全体の数 %)
/// より並列の利得が上回る。共有数が 12 未満なら割らない (内積の ILP 幅 8 に
/// 対してレーンが余り、割るほど 1 本あたりの効率が落ちる)。
fn gqa_chunk_heads(ng: usize) -> usize {
    let mut want = ATTN_GQA_CHUNK.load(AtomicOrdering::Relaxed);
    if want == usize::MAX {
        // 最初の 1 回だけ env を読む (負けた側は同じ値なので上書きしない)。
        let v = std::env::var("DENNO_ATTN_GQA_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let _ = ATTN_GQA_CHUNK.compare_exchange(
            usize::MAX,
            v,
            AtomicOrdering::Relaxed,
            AtomicOrdering::Relaxed,
        );
        want = ATTN_GQA_CHUNK.load(AtomicOrdering::Relaxed);
    }
    let mut c = if want == 0 {
        if ng >= 12 && ng % 2 == 0 {
            ng / 2
        } else {
            ng
        }
    } else {
        want.clamp(1, ng)
    };
    // `out` は head 境界でしか切れないので、ng の約数に丸める。
    while c > 1 && ng % c != 0 {
        c -= 1;
    }
    c
}

/// 1 タスクの q head 数 (`DENNO_ATTN_GQA_CHUNK`、0 = 自動)。`usize::MAX` = 未初期化。
static ATTN_GQA_CHUNK: AtomicUsize = AtomicUsize::new(usize::MAX);

/// テスト・A/B 用: 1 タスクの q head 数を上書きし、以前の値を返す (0 = 自動)。
#[doc(hidden)]
pub fn set_attn_gqa_chunk(c: usize) -> usize {
    ATTN_GQA_CHUNK.swap(c, AtomicOrdering::Relaxed)
}

/// 1 トークン・1 kv head グループぶんの attention ([`AttnLayer::attend_one_into`]
/// と [`AttnLayer::attend_dense_grouped`] の本体)。
///
/// `head0..head0 + ng` は同じ kv head を共有する q head (昇順)。`scores` は
/// `ng * nvis` 以上、`out` は `out_base * dh` ぶん後ろから
/// `(head0 + ng - out_base) * dh` 要素以上 (`out_base` はタスクの先頭 head。
/// グループ全体を 1 タスクで回すときは 0)。
///
/// 計算順は [`AttnLayer::attend_batch`] の doc と同じ:
/// 内積は d 昇順 → `* kq_scale` 1 回、softmax は [`softmax`]、出力は j 昇順。
/// 内積を head 方向にまとめる版 ([`attend_group_scores_ilp`]) も **head ごとの
/// 累積列は d 昇順のまま**なので、値は逐次版とビット一致する。
#[allow(clippy::too_many_arguments)]
fn attend_group_into(
    qrow: &[f32],
    kb: &[f32],
    vb: &[f32],
    row: usize,
    nvis: usize,
    hk: usize,
    head0: usize,
    ng: usize,
    dh: usize,
    kq_scale: f32,
    out_base: usize,
    scores: &mut [f32],
    out: &mut [f32],
) {
    debug_assert!(ng >= 1);
    debug_assert!(scores.len() >= ng * nvis);
    let sc = &mut scores[..ng * nvis];
    // 1) スコア: j (位置) を外・head を内にして k 行を 1 回だけ読む。
    if attend_ilp_on() && ng <= ATTEND_ILP_LANES {
        attend_group_scores_ilp(qrow, kb, row, nvis, hk, head0, ng, dh, kq_scale, sc);
    } else {
        attend_group_scores_serial(qrow, kb, row, nvis, hk, head0, ng, dh, kq_scale, sc);
    }
    // 2) softmax は可視範囲だけ ([`softmax`] そのもの)。
    for li in 0..ng {
        softmax(&mut sc[li * nvis..(li + 1) * nvis]);
    }
    // 3) 出力: j 昇順の逐次加算 (v 行も 1 回だけ読む)。
    for j in 0..nvis {
        let vrow = &vb[j * row + hk * dh..][..dh];
        for li in 0..ng {
            let p = sc[li * nvis + j];
            let orow = &mut out[(head0 + li - out_base) * dh..][..dh];
            for d in 0..dh {
                orow[d] += p * vrow[d];
            }
        }
    }
}

/// head を内側で同時に進める本数 (実形状の GQA 比 24/2 = 12 を 8 + 4 で覆う)。
const ATTEND_ILP_LANES: usize = 8;

/// 内積の head 同時化 (`DENNO_PF_ATT_ILP=0` で従来版・既定 ON)。A/B 用。
static ATTEND_ILP: OnceLock<AtomicBool> = OnceLock::new();

fn attend_ilp_on() -> bool {
    ATTEND_ILP
        .get_or_init(|| {
            AtomicBool::new(std::env::var("DENNO_PF_ATT_ILP").ok().as_deref() != Some("0"))
        })
        .load(AtomicOrdering::Relaxed)
}

/// 内積を head 方向に同時に進める版 (`DENNO_PF_ATT_ILP=0` で逐次版)。
///
/// 1 head ずつ回すと 256 要素の FMA 依存鎖 (レイテンシ 4 クロック/要素) が
/// そのまま律速になり、FMA ユニットが埋まらない。独立な鎖を 8 本立てる
/// (レーンが足りないときは末尾 head で埋めて捨てる) と、同じ FMA 本数でも
/// スループット側に寄る。
///
/// 数値: `acc[k]` の加算は d 昇順・head ごとに独立なので、逐次版とビット一致
/// (テスト `attend_ilp_matches_serial_bitexact`)。
#[allow(clippy::too_many_arguments)]
fn attend_group_scores_ilp(
    qrow: &[f32],
    kb: &[f32],
    row: usize,
    nvis: usize,
    hk: usize,
    head0: usize,
    ng: usize,
    dh: usize,
    kq_scale: f32,
    sc: &mut [f32],
) {
    const L: usize = ATTEND_ILP_LANES;
    let mut i = 0;
    while i < ng {
        let w = (ng - i).min(L);
        // 8 レーン固定で回すため、余りは末尾 head で埋める (値は捨てる)。
        let mut off = [0usize; L];
        for k in 0..L {
            let h = if k < w { head0 + i + k } else { head0 + ng - 1 };
            off[k] = h * dh;
        }
        for j in 0..nvis {
            let krow = &kb[j * row + hk * dh..][..dh];
            let mut acc = [0f32; L];
            for d in 0..dh {
                let kv = krow[d];
                acc[0] += qrow[off[0] + d] * kv;
                acc[1] += qrow[off[1] + d] * kv;
                acc[2] += qrow[off[2] + d] * kv;
                acc[3] += qrow[off[3] + d] * kv;
                acc[4] += qrow[off[4] + d] * kv;
                acc[5] += qrow[off[5] + d] * kv;
                acc[6] += qrow[off[6] + d] * kv;
                acc[7] += qrow[off[7] + d] * kv;
            }
            for k in 0..w {
                sc[(i + k) * nvis + j] = acc[k] * kq_scale;
            }
        }
        i += w;
    }
}

/// 従来版 (基準実装): head ごとに 1 本の累積で d 昇順に足す。
#[allow(clippy::too_many_arguments)]
fn attend_group_scores_serial(
    qrow: &[f32],
    kb: &[f32],
    row: usize,
    nvis: usize,
    hk: usize,
    head0: usize,
    ng: usize,
    dh: usize,
    kq_scale: f32,
    sc: &mut [f32],
) {
    for j in 0..nvis {
        let krow = &kb[j * row + hk * dh..][..dh];
        for li in 0..ng {
            let qh = &qrow[(head0 + li) * dh..][..dh];
            let mut acc = 0f32;
            for d in 0..dh {
                acc += qh[d] * krow[d];
            }
            sc[li * nvis + j] = acc * kq_scale;
        }
    }
}

impl AttnLayer {
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
        self.wo.gemv_label("attn_out", x, &mut out)?;
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
        // 位置は**層ごとのキャッシュ長**から取る。層優先の prefill では、層 L の
        // t 番目を積む時点で他層は既に t+1 番目まで進んでいるため、全層で共有する
        // `pos` は使えない (逐次経路では全層が同じ長さなので値は一致する)。
        // 二重 push の番人は置かない。層ごとの長さが位置そのものなので、同じ層を
        // 続けて 2 回呼ばれた場合と「次のトークン」が区別できない (token-major の
        // 共有 pos があって初めて区別できた)。守りは 16 トークンの参照一致
        // (real_model_greedy_matches_llamacpp) と全テストで取る。
        let pos = self.cache.len(layer)?;
        if pos >= self.cache.max_ctx() {
            bail!("FullAttn::forward: 層 {layer} が容量 max_ctx = {} に達した", self.cache.max_ctx());
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
        // 対外的な「何トークン進んだか」は最大値で持つ (診断とリセット用)。
        if self.pos < pos + 1 {
            self.pos = pos + 1;
        }
        Ok(out)
    }

    /// チャンクぶんの full attention 層をまとめて回す (バッチ prefill 用)。
    ///
    /// [`FullAttn::forward_token`] をトークン順に呼んだ場合とビット一致する。
    /// 変えるのは (1) 4 本の投影をチャンクで 1 回にまとめる ([`gemv_batch`] は
    /// 1 トークン版と同じカーネル・同じ丸め順)、(2) `attend` を
    /// [`AttnLayer::attend_batch`] (KV を 1 回走査) にする、の 2 点だけ。
    /// KV キャッシュへは `forward_token` と同じくトークン順に push するので、
    /// 積まれる内容も順序も同じ。位置は層ごとのキャッシュ長から取る
    /// (`forward_token` と同じ規則)。
    ///
    /// 1 トークンのチャンクと QSA 層 (`DENNO_QSA=1` で選択が有効な層) は
    /// 逐次経路 ([`FullAttn::forward_token`]) に任せる。QSA の選択結果は
    /// トークンごとに依存するので dense のバッチ経路と混ぜない。
    pub fn forward_batch(&mut self, layer: usize, curs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let nt = curs.len();
        if nt == 0 {
            return Ok(Vec::new());
        }
        for (i, cur) in curs.iter().enumerate() {
            if cur.len() != self.n_embd {
                bail!(
                    "FullAttn::forward_batch: トークン {i} の入力長 {} が n_embd = {} と違う",
                    cur.len(),
                    self.n_embd
                );
            }
        }
        let Some(slot) = self.layers.get(layer) else {
            bail!(
                "FullAttn::forward_batch: 層 {layer} は範囲外 (n_layer = {})",
                self.layers.len()
            );
        };
        if slot.is_none() {
            bail!("FullAttn::forward_batch: 層 {layer} は ssm 層で full attention の担当外");
        }
        if nt == 1 || matches!(&self.qsa, Some(q) if q.is_qsa(layer)) {
            let mut outs = Vec::with_capacity(nt);
            for cur in curs {
                outs.push(self.forward_token(layer, cur)?);
            }
            return Ok(outs);
        }
        let base = self.cache.len(layer)?;
        if base + nt > self.cache.max_ctx() {
            bail!(
                "FullAttn::forward_batch: 層 {layer} が容量 max_ctx = {} を超える ({base} + {nt})",
                self.cache.max_ctx()
            );
        }
        let n_embd = self.n_embd;
        let mut flat = Vec::with_capacity(nt * n_embd);
        for cur in curs {
            flat.extend_from_slice(cur);
        }

        // P49: 段別の時間 (`DENNO_Q4_PROFILE` か `DENNO_PF_ATT_DBG` のときだけ取る)。
        let t_all = q4prof::stamp();
        q4prof::bump_nt(layer, nt);

        let l = self.layers[layer]
            .as_mut()
            .expect("上の is_none で除外済み");
        let qd = l.q_dim();
        let kv = l.kv_dim();
        let qg = l.qg_dim();

        // 投影はチャンクで 1 回ずつ (トークンごとの `gemv` と同じカーネル)。
        let t_qkv = q4prof::stamp();
        let mut qgs = vec![0f32; nt * qg];
        gemv_batch(&l.wq, &flat, nt, &mut qgs)?;
        let mut ks = vec![0f32; nt * kv];
        gemv_batch(&l.wk, &flat, nt, &mut ks)?;
        let mut vs = vec![0f32; nt * kv];
        gemv_batch(&l.wv, &flat, nt, &mut vs)?;
        q4prof::mark_l(t_qkv, layer, q4prof::ST_QKV);

        // 正規化 + rope はトークンごと (pos が違う)。k は rope 後にキャッシュへ積む。
        let t_rope = q4prof::stamp();
        let mut qs = vec![0f32; nt * qd];
        let mut gates = vec![0f32; nt * qd];
        for t in 0..nt {
            let (mut q, gate) = l.split_q_gate(&qgs[t * qg..(t + 1) * qg])?;
            let pos = base + t;
            l.norm_rope_q(&mut q, pos)?;
            qs[t * qd..(t + 1) * qd].copy_from_slice(&q);
            gates[t * qd..(t + 1) * qd].copy_from_slice(&gate);
            l.norm_rope_k(&mut ks[t * kv..(t + 1) * kv], pos)?;
            self.cache.push(
                layer,
                &ks[t * kv..(t + 1) * kv],
                &vs[t * kv..(t + 1) * kv],
            )?;
        }
        q4prof::mark_l(t_rope, layer, q4prof::ST_ROPE);

        let t_att = q4prof::stamp();
        let attn = l.attend_batch(&qs, nt, &self.cache, layer)?;
        q4prof::mark_l(t_att, layer, q4prof::ST_ATTEND);
        let t_gmul = q4prof::stamp();
        let mut gated = vec![0f32; nt * qd];
        for t in 0..nt {
            let g = l.gate_mul(
                &attn[t * qd..(t + 1) * qd],
                &gates[t * qd..(t + 1) * qd],
            )?;
            gated[t * qd..(t + 1) * qd].copy_from_slice(&g);
        }
        q4prof::mark_l(t_gmul, layer, q4prof::ST_GMUL);
        let t_out = q4prof::stamp();
        let mut outs = vec![0f32; nt * n_embd];
        gemv_batch(&l.wo, &gated, nt, &mut outs)?;
        q4prof::mark_l(t_out, layer, q4prof::ST_OUT);
        // full 層の合計 (`dump_attn` の A_FULL。従来は誰も加算していなかった)。
        if let Some(t) = t_all {
            q4prof::add(&q4prof::A_FULL, t);
        }
        // 対外的な「何トークン進んだか」は最大値で持つ (`forward_token` と同じ)。
        if self.pos < base + nt {
            self.pos = base + nt;
        }
        Ok(outs.chunks_exact(n_embd).map(|c| c.to_vec()).collect())
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

    /// チャンクぶんの full attention を KV 1 回走査で回す
    /// ([`FullAttn::forward_batch`])。1 トークン版とビット一致する。
    fn forward_batch(
        &mut self,
        layer: usize,
        curs: &[Vec<f32>],
    ) -> std::result::Result<Vec<Vec<f32>>, Qwen4ExpError> {
        FullAttn::forward_batch(self, layer, curs)
            .map_err(|e| Qwen4ExpError(vec![format!("{e:#}")]))
    }

    /// 層優先の順序でも安全。KV は層ごとに独立して持ち、forward は 1 層に
    /// 1 トークンを積むだけなので、層優先でも各層の中ではトークン順が保たれる。
    fn layer_major_safe(&self) -> bool {
        true
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

// ---------------------------------------------------------------------------
// P49: attn の段別テストと実形状ベンチ (このコンテナの CPU)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod p49 {
    //! prefill の full attention を段別に見るためのテスト群。
    //!
    //! - 数値: 新しい経路 (rope の表引き・内積の head 同時化) が従来版と
    //!   **ビット一致**することを固定する。
    //! - 実測: 実形状 (Flash-Next = 24 head / 2 kv head / dh 256) の段別コスト。
    //!   `cargo test --release -p denno-engine --lib p49 -- --ignored --nocapture -j 8`
    //!   で回す (既定のテストでは走らない)。
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        }
    }

    fn fill(seed: u64, v: &mut [f32]) {
        let mut r = lcg(seed);
        for x in v.iter_mut() {
            *x = r();
        }
    }

    /// 表引き版 rope が従来版とビット一致する (実形状 + 端の形)。
    #[test]
    fn rope_tab_matches_serial_bitexact() {
        for &(heads, dh, n_rot, pos) in &[
            (24usize, 256usize, 64usize, 0usize),
            (24, 256, 64, 1),
            (24, 256, 64, 511),
            (24, 256, 64, 65535),
            (2, 256, 64, 7),
            (3, 128, 32, 13),
            (1, 64, 64, 5),
        ] {
            let mut a = vec![0f32; heads * dh];
            fill(0x51 + pos as u64, &mut a);
            let mut b = a.clone();
            rope_neox_partial_serial(&mut a, heads, dh, n_rot, pos, 10_000_000.0);
            rope_neox_partial_tab(&mut b, heads, dh, n_rot, pos, 10_000_000.0);
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "rope 表引きが従来版と違う (heads {heads} dh {dh} n_rot {n_rot} pos {pos} i {i})"
                );
            }
            // 回転していること (全要素が入力と同一なら経路が死んでいる)
            assert!(
                a.iter().zip(b.iter()).any(|(x, _)| *x != 0.0),
                "rope の出力が全部 0"
            );
        }
    }

    /// 内積の head 同時化が従来版とビット一致する (グループ長 1 / 5 / 12 / 16)。
    #[test]
    fn attend_ilp_matches_serial_bitexact() {
        let dh = 64usize;
        let row = 2 * dh; // kv head 2 本ぶんの行
        for &ng in &[1usize, 5, 8, 12, 16] {
            for &nvis in &[1usize, 7, 33] {
                let mut q = vec![0f32; 24 * dh];
                fill(0x91 + ng as u64, &mut q);
                let mut kb = vec![0f32; 40 * row];
                fill(0x37 + nvis as u64, &mut kb);
                let mut serial = vec![0f32; ng * nvis];
                let mut ilp = vec![0f32; ng * nvis];
                attend_group_scores_serial(
                    &q,
                    &kb,
                    row,
                    nvis,
                    1,
                    0,
                    ng,
                    dh,
                    1.0 / (dh as f32).sqrt(),
                    &mut serial,
                );
                attend_group_scores_ilp(
                    &q,
                    &kb,
                    row,
                    nvis,
                    1,
                    0,
                    ng,
                    dh,
                    1.0 / (dh as f32).sqrt(),
                    &mut ilp,
                );
                for (i, (a, b)) in serial.iter().zip(ilp.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "head 同時化が逐次版と違う (ng {ng} nvis {nvis} i {i})"
                    );
                }
                assert!(
                    serial.iter().any(|v| *v != 0.0),
                    "スコアが全部 0 (経路が死んでいる)"
                );
            }
        }
    }

    // ---- 実形状ベンチ -----------------------------------------------------

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    /// Flash-Next full 層の実形状: n_head 24 / n_head_kv 2 / dh 256 / n_rot 64。
    const P49_HEADS: usize = 24;
    const P49_KV: usize = 2;
    const P49_DH: usize = 256;
    const P49_NROT: usize = 64;
    const P49_ROW: usize = P49_KV * P49_DH;

    /// rope 1 トークンぶん (q 24 head + k 2 head) の従来版 / 表引き版。
    #[test]
    #[ignore]
    fn bench_rope_p49() {
        let mut q_serial = vec![0f32; P49_HEADS * P49_DH];
        let mut q_tab = vec![0f32; P49_HEADS * P49_DH];
        let mut k_serial = vec![0f32; P49_KV * P49_DH];
        let mut k_tab = vec![0f32; P49_KV * P49_DH];
        fill(1, &mut q_serial);
        fill(2, &mut k_serial);
        q_tab.copy_from_slice(&q_serial);
        k_tab.copy_from_slice(&k_serial);
        let reps = 2000usize;
        let mut ts = Vec::new();
        let mut tt = Vec::new();
        for _ in 0..7 {
            let s = std::time::Instant::now();
            for i in 0..reps {
                rope_neox_partial_serial(&mut q_serial, P49_HEADS, P49_DH, P49_NROT, i, 1e7);
                rope_neox_partial_serial(&mut k_serial, P49_KV, P49_DH, P49_NROT, i, 1e7);
            }
            ts.push(s.elapsed().as_secs_f64() * 1e6 / reps as f64);
            let s = std::time::Instant::now();
            for i in 0..reps {
                rope_neox_partial_tab(&mut q_tab, P49_HEADS, P49_DH, P49_NROT, i, 1e7);
                rope_neox_partial_tab(&mut k_tab, P49_KV, P49_DH, P49_NROT, i, 1e7);
            }
            tt.push(s.elapsed().as_secs_f64() * 1e6 / reps as f64);
        }
        let (a, b) = (median(ts), median(tt));
        println!(
            "[p49] rope 1 token (24 head + 2 head, dh 256, n_rot 64): serial {a:.2} us / tab {b:.2} us = {:.2}x",
            a / b
        );
        assert!(b < a, "表引き版が遅い (serial {a:.2} us / tab {b:.2} us)");
    }

    /// attend 1 トークン (nvis 512 = チャンク末尾) の スコア / softmax+PV / 合計。
    #[test]
    #[ignore]
    fn bench_attend_p49() {
        let nvis = 512usize;
        let mut q = vec![0f32; P49_HEADS * P49_DH];
        fill(11, &mut q);
        let mut kb = vec![0f32; nvis * P49_ROW];
        let mut vb = vec![0f32; nvis * P49_ROW];
        fill(12, &mut kb);
        fill(13, &mut vb);
        let ng = P49_HEADS / P49_KV;
        let mut sc_serial = vec![0f32; ng * nvis];
        let mut sc_ilp = vec![0f32; ng * nvis];
        let mut out = vec![0f32; P49_HEADS * P49_DH];
        // `attend_group_into` は softmax 後でスコアを上書きするので、比較用の
        // 生スコアは別バッファに残す。
        let mut sc_used = vec![0f32; ng * nvis];
        let kq = 1.0f32 / (P49_DH as f32).sqrt();
        let reps = 200usize;
        let mut t_serial = Vec::new();
        let mut t_ilp = Vec::new();
        let mut t_full = Vec::new();
        for _ in 0..5 {
            let s = std::time::Instant::now();
            for _ in 0..reps {
                attend_group_scores_serial(
                    &q, &kb, P49_ROW, nvis, 0, 0, ng, P49_DH, kq, &mut sc_serial,
                );
            }
            t_serial.push(s.elapsed().as_secs_f64() * 1e6 / reps as f64);
            let s = std::time::Instant::now();
            for _ in 0..reps {
                attend_group_scores_ilp(
                    &q, &kb, P49_ROW, nvis, 0, 0, ng, P49_DH, kq, &mut sc_ilp,
                );
            }
            t_ilp.push(s.elapsed().as_secs_f64() * 1e6 / reps as f64);
            let s = std::time::Instant::now();
            for _ in 0..reps {
                // 1 kv head グループ × 2 (2 kv head ぶんが 1 トークンの全 q head)
                attend_group_into(
                    &q, &kb, &vb, P49_ROW, nvis, 0, 0, ng, P49_DH, kq, 0, &mut sc_used,
                    &mut out,
                );
                attend_group_into(
                    &q, &kb, &vb, P49_ROW, nvis, 1, ng, ng, P49_DH, kq, 0, &mut sc_used,
                    &mut out,
                );
            }
            t_full.push(s.elapsed().as_secs_f64() * 1e6 / reps as f64);
        }
        let (a, b, c) = (median(t_serial), median(t_ilp), median(t_full));
        println!(
            "[p49] attend 1 token nvis {nvis} (2 kv head x {ng} head, dh {P49_DH}): \
             scores/1kvhead serial {a:.1} us / ilp {b:.1} us ({:.2}x) / 2kvhead total {c:.1} us / \
             softmax+PV {:.1} us",
            a / b,
            c - 2.0 * b,
        );
        for (i, (x, y)) in sc_serial.iter().zip(sc_ilp.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "スコアが一致しない (i {i})");
        }
    }

    /// `DENNO_PF_ATT_DBG` の層別集計と 1 行出力が実際に組み上がる
    /// (緑 = 経路が発動した証拠ではないので、行の中身まで見る)。
    #[test]
    fn att_dbg_layers_build_one_line_each() {
        use crate::model::qwen4exp::q4prof;
        q4prof::set_att_dbg_for_test(true);
        q4prof::reset();

        // ssm 層 (0): gdn に 1 ms、qkv に 0.5 ms 相当を積む
        q4prof::bump_nt(0, 512);
        let t = std::time::Instant::now();
        std::thread::sleep(std::time::Duration::from_millis(1));
        q4prof::mark(Some(t), &q4prof::A_QKV, 0, q4prof::ST_QKV);
        q4prof::mark_l(Some(t), 0, q4prof::ST_GDN);

        // full 層 (47): attend に 1 ms 相当
        q4prof::bump_nt(47, 512);
        q4prof::mark_l(Some(t), 47, q4prof::ST_ATTEND);

        let ssm = q4prof::layer_line(0, false).expect("ssm 層の行が出ない");
        assert!(ssm.starts_with("[pf-att] layer 0 "), "行の頭が違う: {ssm}");
        assert!(ssm.contains("qkv ") && ssm.contains("gate ") && ssm.contains("gdn "),
            "ssm の段名が揃っていない: {ssm}");
        assert!(ssm.contains("nt 512"), "分母 (nt) が出ていない: {ssm}");
        let gdn_ms: f64 = ssm
            .split("gdn ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|s| s.parse().ok())
            .expect("gdn の ms が読めない");
        // 並行して走る他のテストも同じ束に積むので「以上」で見る (値は増える方向だけ)。
        assert!(gdn_ms >= 0.5, "gdn の ms が積まれていない: {ssm}");

        let full = q4prof::layer_line(47, true).expect("full 層の行が出ない");
        assert!(full.starts_with("[pf-att] layer 47 (full)"), "行の頭が違う: {full}");
        assert!(full.contains("rope ") && full.contains("attend ") && full.contains("gmul "),
            "full の段名が揃っていない: {full}");

        // 内訳の対象外の層は行を出さない (空行で埋めない)。DBG_LAYERS の 23 は
        // 並行テストが積みうるので、対象外の層番号で見る。
        assert!(q4prof::layer_line(5, true).is_none(), "対象外の層に行が出た");

        // 内訳は Q4_PROFILE の束 (A_FULL) を汚さない
        q4prof::reset();
        q4prof::set_att_dbg_for_test(false);
    }
}
