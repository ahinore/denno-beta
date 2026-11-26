//! Qwen3.8-Flash-Next (`general.architecture = "qwen4exp"`) のパラメータ構造と
//! 層ごとのテンソル束 (P-MOE-E3)。
//!
//! 出典: メタデータのキー名・テンソル名・形状・量子型の対応は llama.cpp
//! `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) の
//! `load_arch_hparams` / `load_arch_tensors` を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! # 目的
//!
//! 実 GGUF の**名前と形状をそのまま受ける**。テンソルを 1 本でも欠いたり、
//! 形状・量子型を 1 つでも取り違えたりしたら、**推論を始める前に**落とす。
//! `validate` は最初の 1 件で止めず、全件を 1 行 1 件の文字列で集めて返す
//! (実ファイルは 1224 本あるので、1 件ずつ潰すと起動検証が回らない)。
//!
//! # 数値の出所
//!
//! 期待値はすべてメタデータから導出する (目録の実値をコードに埋めない)。
//! 例: `hc_dim = embedding_length * hyper_connection.count = 10240`、
//! `conv_dim = 2 * (ssm.state_size * ssm.group_count) + ssm.inner_size = 10240`。
//! 目録の実値 (2560 / 512 / 640 / 10240 / 12288 / 320001536 など) は
//! `tests/qwen4exp_params.rs` に固定してある。
//!
//! # 層の種別
//!
//! `full_attention_interval = 4` なので `(il + 1) % 4 == 0` の層
//! (3, 7, ..., 47 の 12 層) が full attention、残り 36 層が ssm
//! (`is_recr` = linear attention)。ssm 層は `attn_qkv` / `attn_gate` を持ち、
//! full attention 層は `attn_q` / `attn_k` / `attn_v` / `indexer.*` を持つ。
//! **同じ層に両方は置けない** — `validate` は期待表に無い `blk.N.*` を
//! 「余分なテンソル」として報告するので、種別の取り違えはここで捕まる。

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::gguf::{GgufFile, GgufTensorInfo, GgufValue, TensorRef};
use crate::kquant::GgmlType;
use crate::model::expert_cache::{self, CacheSpec, ExpertCacheSet, ExpertKey, ExpertMatrix, RamSpec};
use crate::model::expert_vram::{self, Route};
use crate::model::hyper::{self, HyperParams, HyperStreams};
use crate::model::intervene::{self, Block as IvBlock, View as IvView};
use crate::model::steer;
use crate::model::linear_gguf::GgufLinear;
use crate::model::moe::{
    select_topk, select_topk_into, select_topk_into_fast, select_topk_into_mask,
    select_topk_mask,
};
use crate::model::moe_expert::{self, ExpertSlice};
use crate::model::prekprof;
use crate::model::q4ledger;
use crate::ple::{PleNgram, PleTable, PLE_TABLE_NAME};
use crate::quant::WeightBlocks;
use crate::tensor::{silu_inplace, softmax};

// task 668: 31k 行の単体ファイルを領域ごとの子モジュールへ移した (純粋な移動。
// ロジック・数値・ゲートの既定は変更なし)。外部から見た `crate::model::qwen4exp::*` は
// 下の `pub use ...::*` で従来どおり解決する。
mod mix;
mod moe;
/// task 874: cold prefill の選択 expert の f32 展開を使い回すキャッシュ (既定 off)。
mod pf_dq_cache;
mod prefill;
mod prof;
/// task 822: prefill の天井 (計算 vs ディスク) の 1 行診断。
pub mod roofline;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests2;

pub use mix::*;
pub use moe::*;
pub use prefill::*;
pub use prof::*;

/// PLE の n-gram 索引に渡す窓を作る。
///
/// `PleNgram::indices` は **新しい順** (`ctx[0]` = 現在のトークン) を要求する。
/// 呼び出し側の `ctx` は古い順なので、末尾から `n_gram` 個を取って反転する。
/// 長さが足りなければ短いまま返す (足りない古い側は `indices` が EOS で埋める)。
///
/// ここを取り違えると n-gram が 2 つずれた窓で引かれる (2026-10-26 に実際に踏んだ)。
pub fn ple_window(ctx: &[u32], n_gram: usize) -> Vec<u32> {
    ctx.iter().rev().take(n_gram).copied().collect()
}

/// 検証エラー。**1 行 1 件**で、最初の 1 件で止めずに全部を集める。
///
/// 表示は行を `\n` で連結する (呼び出し側が `for line in e.lines()` で
/// 1 件ずつ扱える)。中身は [`Qwen4ExpError::0`] で直接見られる。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen4ExpError(pub Vec<String>);

impl fmt::Display for Qwen4ExpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, line) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{line}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Qwen4ExpError {}

type Result<T> = std::result::Result<T, Qwen4ExpError>;

/// 1 件だけのエラー。`model::qwen4exp_ssm` の [`crate::model::qwen4exp::LayerAttn`]
/// 実装からも使う (`pub(crate)`)。
pub(crate) fn one(msg: impl Into<String>) -> Qwen4ExpError {
    Qwen4ExpError(vec![msg.into()])
}

/// ggml の型名 (エラーメッセージ用)。目録の表記に合わせる。
pub fn ty_name(t: GgmlType) -> &'static str {
    match t {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q2K => "Q2_K",
        GgmlType::Q3K => "Q3_K",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q5K => "Q5_K",
        GgmlType::Q6K => "Q6_K",
        GgmlType::IQ2Xxs => "IQ2_XXS",
        GgmlType::IQ2Xs => "IQ2_XS",
        GgmlType::IQ3Xxs => "IQ3_XXS",
        GgmlType::IQ1S => "IQ1_S",
        GgmlType::IQ4Nl => "IQ4_NL",
        GgmlType::IQ3S => "IQ3_S",
        GgmlType::IQ2S => "IQ2_S",
        GgmlType::IQ4Xs => "IQ4_XS",
        GgmlType::Iq1M => "IQ1_M",
        GgmlType::Bf16 => "BF16",
        GgmlType::Tq1_0 => "TQ1_0",
        GgmlType::Mxfp4 => "MXFP4",
        GgmlType::Q2_0 => "Q2_0",
    }
}

fn fmt_dims(d: &[u64]) -> String {
    let mut s = String::from("[");
    for (i, x) in d.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

// ---------------------------------------------------------------------------
// パラメータ
// ---------------------------------------------------------------------------

/// 層の種別。ssm = linear attention (`is_recr`)、FullAttn = QSA full attention。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Ssm,
    FullAttn,
}

impl LayerKind {
    pub fn name(self) -> &'static str {
        match self {
            LayerKind::Ssm => "ssm",
            LayerKind::FullAttn => "full attention",
        }
    }
}

/// `qwen4exp` のメタデータ全部 (目録の 67 キーのうち、構造に効くもの)。
///
/// 数値は GGUF の生値のまま保持する。派生値 (`hc_dim` など) はメソッドで出す。
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen4ExpParams {
    /// `block_count` = 48
    pub n_layer: usize,
    /// `context_length` = 262144
    pub n_ctx: usize,
    /// `embedding_length` = 2560
    pub n_embd: usize,
    /// `attention.head_count` = 24
    pub n_head: usize,
    /// `attention.head_count_kv` = 2
    pub n_head_kv: usize,
    /// `attention.key_length` = 256
    pub n_embd_head_k: usize,
    /// `attention.value_length` = 256
    pub n_embd_head_v: usize,
    /// `attention.layer_norm_rms_epsilon` = 1e-6
    pub rms_eps: f32,
    /// `rope.dimension_count` = 64
    pub n_rot: usize,
    /// `rope.dimension_sections` = [11, 11, 10, 0] (IMRoPE の 4 分割)
    pub rope_sections: [u32; 4],
    /// `rope.freq_base` = 1e7
    pub rope_freq_base: f32,
    /// `expert_count` = 512
    pub n_expert: usize,
    /// `expert_used_count` = 10
    pub n_expert_used: usize,
    /// `expert_feed_forward_length` = 640
    pub n_ff_exp: usize,
    /// `expert_shared_feed_forward_length` = 640
    pub n_ff_shexp: usize,
    /// `ssm.conv_kernel` = 4
    pub ssm_d_conv: usize,
    /// `ssm.state_size` = 128 (head あたりの k/v 次元)
    pub ssm_d_state: usize,
    /// `ssm.group_count` = 16 (k head 数)
    pub ssm_n_group: usize,
    /// `ssm.time_step_rank` = 48 (v head 数)
    pub ssm_dt_rank: usize,
    /// `ssm.inner_size` = 6144 (= ssm_d_state * ssm_dt_rank)
    pub ssm_d_inner: usize,
    /// `full_attention_interval` = 4
    pub full_attn_interval: usize,
    /// `hyper_connection.count` = 4
    pub n_hc: usize,
    /// `hyper_connection.low_rank` = 320
    pub hc_low_rank: usize,
    /// `attention.indexer.head_count` = 4
    pub indexer_n_head: usize,
    /// `attention.indexer.key_length` = 128
    pub indexer_head_size: usize,
    /// `attention.indexer.top_k` = 2048
    pub indexer_top_k: usize,
    /// `attention.compress_ratios` (48 要素。0 = QSA なし、4 = 4 cell で 1 pool)
    pub compress_ratios: Vec<u32>,
    /// `embedding_length_per_layer_input` = 160 (PLE の 1 行の次元)
    pub n_embd_per_layer: usize,
    /// `ple.layers` = [1] (PLE を持つ層。目録では 1 層だけ)
    pub ple_layers: Vec<usize>,
    /// `ple.ngram_size` = 3
    pub ple_ngram_size: usize,
    /// `ple.heads_per_ngram` = 8
    pub ple_heads_per_ngram: usize,
    /// `ple.conv_kernel` = 4
    pub ple_conv_kernel: usize,
    /// `ple.eos_token_id` = 248044
    pub ple_eos_token_id: u32,
    /// `ple.image_token_id` = 248056 (無ければ eos に落ちる)
    pub ple_image_token_id: u32,
    /// `ple.layer_multipliers` (ngram_size 個。u64 のまま)
    pub ple_layer_multipliers: Vec<u64>,
    /// `ple.head_offsets` (n_ple_heads 個。表の行番号)
    pub ple_head_offsets: Vec<u32>,
    /// `ple.head_vocab_sizes` (n_ple_heads 個)
    pub ple_head_vocab_sizes: Vec<u32>,
    /// `tokenizer.ggml.tokens` の要素数 = 248320
    pub n_vocab: usize,
}

impl Qwen4ExpParams {
    /// メタデータから読む。**最初の 1 件で止めず**、欠落・型違い・値の不整合を
    /// 全部集めて返す (1 行 1 件)。
    pub fn from_meta(f: &GgufFile) -> Result<Self> {
        let mut acc = Acc::new();

        // 接頭辞は qwen4exp 固定。アーキテクチャ違いはそれ自体を 1 件のエラーに
        // するが、読むキーは qwen4exp.* のまま (他アーキの値を qwen4exp と
        // 誤読しないため)。
        let arch = f
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        if arch != "qwen4exp" {
            acc.push(format!(
                "general.architecture: '{arch}' が期待 'qwen4exp' と違う"
            ));
        }

        let n_layer = acc.u(f, "block_count");
        let n_ctx = acc.u(f, "context_length");
        let n_embd = acc.u(f, "embedding_length");
        let n_head = acc.u(f, "attention.head_count");
        let n_head_kv = acc.u(f, "attention.head_count_kv");
        let n_embd_head_k = acc.u(f, "attention.key_length");
        let n_embd_head_v = acc.u(f, "attention.value_length");
        let rms_eps = acc.f32(f, "attention.layer_norm_rms_epsilon");
        let n_rot = acc.u(f, "rope.dimension_count");
        let sections = acc.arr_u32(f, "rope.dimension_sections", 4, Some(4));
        let rope_freq_base = acc.f32(f, "rope.freq_base");
        let n_expert = acc.u(f, "expert_count");
        let n_expert_used = acc.u(f, "expert_used_count");
        let n_ff_exp = acc.u(f, "expert_feed_forward_length");
        let n_ff_shexp = acc.u(f, "expert_shared_feed_forward_length");
        let ssm_d_conv = acc.u(f, "ssm.conv_kernel");
        let ssm_d_state = acc.u(f, "ssm.state_size");
        let ssm_n_group = acc.u(f, "ssm.group_count");
        let ssm_dt_rank = acc.u(f, "ssm.time_step_rank");
        let ssm_d_inner = acc.u(f, "ssm.inner_size");
        let full_attn_interval = acc.u(f, "full_attention_interval");
        let n_hc = acc.u(f, "hyper_connection.count");
        let hc_low_rank = acc.u(f, "hyper_connection.low_rank");
        let indexer_n_head = acc.u(f, "attention.indexer.head_count");
        let indexer_head_size = acc.u(f, "attention.indexer.key_length");
        let indexer_top_k = acc.u(f, "attention.indexer.top_k");
        let compress_ratios = acc.arr_u32(f, "attention.compress_ratios", 1, None);
        let n_vocab = acc.arr_len(f, "tokenizer.ggml.tokens");

        // PLE は「層の一覧」があるときだけ他のキーを要求する
        // (llama.cpp も n_ple == 0 なら PLE のキーを読まない)。キー自体が無い
        // ファイルは PLE 無しとして扱い、エラーにしない。
        let ple_layers: Vec<usize> = if f.get("qwen4exp.ple.layers").is_none() {
            Vec::new()
        } else {
            match acc.arr_u32(f, "ple.layers", 0, None) {
                None => Vec::new(),
                Some(v) => v.into_iter().map(|x| x as usize).collect(),
            }
        };
        let (n_embd_per_layer, ple_ngram_size, ple_heads_per_ngram, ple_conv_kernel) =
            if ple_layers.is_empty() {
                (0, 0, 0, 0)
            } else {
                (
                    acc.u(f, "embedding_length_per_layer_input").unwrap_or(0),
                    acc.u(f, "ple.ngram_size").unwrap_or(0),
                    acc.u(f, "ple.heads_per_ngram").unwrap_or(0),
                    acc.u(f, "ple.conv_kernel").unwrap_or(0),
                )
            };
        let (ple_eos_token_id, ple_image_token_id, ple_layer_multipliers, ple_head_offsets,
            ple_head_vocab_sizes) = if ple_layers.is_empty() {
            (0u32, 0u32, Vec::new(), Vec::new(), Vec::new())
        } else {
            let n_heads = ple_ngram_size.saturating_sub(1) * ple_heads_per_ngram;
            (
                acc.u32(f, "ple.eos_token_id").unwrap_or(0),
                acc.u32_opt(f, "ple.image_token_id").unwrap_or(0),
                acc.arr_u64(f, "ple.layer_multipliers", ple_ngram_size)
                    .unwrap_or_default(),
                acc.arr_u32(f, "ple.head_offsets", n_heads, None)
                    .unwrap_or_default(),
                acc.arr_u32(f, "ple.head_vocab_sizes", n_heads, None)
                    .unwrap_or_default(),
            )
        };

        // ここまでの読み取りで 1 件でも失敗していれば、値の整合性は見ない
        // (0 埋めの値で二次的なエラーを量産しない)。
        if !acc.errs.is_empty() {
            return Err(Qwen4ExpError(acc.errs));
        }
        // 以降は「errs が空 = 全キーが読めた」ので expect は到達しない。
        let n_layer = n_layer.expect("上の収集で None は検出済み");
        let n_ctx = n_ctx.expect("同上");
        let n_embd = n_embd.expect("同上");
        let n_head = n_head.expect("同上");
        let n_head_kv = n_head_kv.expect("同上");
        let n_embd_head_k = n_embd_head_k.expect("同上");
        let n_embd_head_v = n_embd_head_v.expect("同上");
        let rms_eps = rms_eps.expect("同上");
        let n_rot = n_rot.expect("同上");
        let sections = sections.expect("同上");
        let rope_freq_base = rope_freq_base.expect("同上");
        let n_expert = n_expert.expect("同上");
        let n_expert_used = n_expert_used.expect("同上");
        let n_ff_exp = n_ff_exp.expect("同上");
        let n_ff_shexp = n_ff_shexp.expect("同上");
        let ssm_d_conv = ssm_d_conv.expect("同上");
        let ssm_d_state = ssm_d_state.expect("同上");
        let ssm_n_group = ssm_n_group.expect("同上");
        let ssm_dt_rank = ssm_dt_rank.expect("同上");
        let ssm_d_inner = ssm_d_inner.expect("同上");
        let full_attn_interval = full_attn_interval.expect("同上");
        let n_hc = n_hc.expect("同上");
        let hc_low_rank = hc_low_rank.expect("同上");
        let indexer_n_head = indexer_n_head.expect("同上");
        let indexer_head_size = indexer_head_size.expect("同上");
        let indexer_top_k = indexer_top_k.expect("同上");
        let compress_ratios = compress_ratios.expect("同上");
        let n_vocab = n_vocab.expect("同上");

        let mut errs = Vec::new();
        // ゼロは形状が潰れる = 静かに誤読するので拒否する
        let nonzero: [(&str, usize); 13] = [
            ("block_count", n_layer),
            ("embedding_length", n_embd),
            ("attention.head_count", n_head),
            ("attention.head_count_kv", n_head_kv),
            ("attention.key_length", n_embd_head_k),
            ("attention.value_length", n_embd_head_v),
            ("expert_count", n_expert),
            ("expert_feed_forward_length", n_ff_exp),
            ("expert_shared_feed_forward_length", n_ff_shexp),
            ("ssm.conv_kernel", ssm_d_conv),
            ("ssm.state_size", ssm_d_state),
            ("ssm.group_count", ssm_n_group),
            ("ssm.time_step_rank", ssm_dt_rank),
        ];
        for (key, v) in nonzero {
            if v == 0 {
                errs.push(format!("qwen4exp.{key}: 0 は不正"));
            }
        }
        for (key, v) in [
            ("ssm.inner_size", ssm_d_inner),
            ("hyper_connection.low_rank", hc_low_rank),
            ("attention.indexer.head_count", indexer_n_head),
            ("attention.indexer.key_length", indexer_head_size),
            ("attention.indexer.top_k", indexer_top_k),
        ] {
            if v == 0 {
                errs.push(format!("qwen4exp.{key}: 0 は不正"));
            }
        }
        if full_attn_interval == 0 {
            errs.push("qwen4exp.full_attention_interval: 0 は不正".to_string());
        }
        // HC は 1 だと混ぜる対象が無い (transformers / vLLM / SGLang も <=1 を拒否)
        if n_hc <= 1 {
            errs.push(format!(
                "qwen4exp.hyper_connection.count: 1 以下は不正 (実際 {n_hc})"
            ));
        }
        // ssm.inner_size は v head 次元 × v head 数と一致する (グラフ側の前提)
        if ssm_d_state != 0 && ssm_dt_rank != 0 && ssm_d_inner != ssm_d_state * ssm_dt_rank {
            errs.push(format!(
                "qwen4exp.ssm.inner_size: {ssm_d_inner} が state_size * time_step_rank = {} と違う",
                ssm_d_state * ssm_dt_rank
            ));
        }
        if compress_ratios.len() != n_layer {
            errs.push(format!(
                "qwen4exp.attention.compress_ratios: {} 要素が block_count = {n_layer} と違う",
                compress_ratios.len()
            ));
        }
        // QSA は全層で同じ pool 比を使い、その比が top_k を割り切る
        let kpool = compress_ratios.iter().copied().find(|&r| r != 0).unwrap_or(0);
        if kpool != 0 {
            for (il, &r) in compress_ratios.iter().enumerate() {
                if r != 0 && r != kpool {
                    errs.push(format!(
                        "qwen4exp.attention.compress_ratios: 層 {il} の比 {r} が層 {} の {kpool} と違う (QSA は 1 モデル 1 比)",
                        compress_ratios.iter().position(|&x| x == kpool).unwrap_or(0)
                    ));
                }
            }
            if kpool == 1 {
                errs.push(
                    "qwen4exp.attention.compress_ratios: 1 は pool にならない (2 以上)".to_string(),
                );
            }
            if indexer_top_k != 0 && indexer_top_k % kpool as usize != 0 {
                errs.push(format!(
                    "qwen4exp.attention.indexer.top_k: {indexer_top_k} が compress_ratio {kpool} で割り切れない"
                ));
            }
        }

        // PLE: 定数 (ハッシュ) は 1 組しか持てないので層は 1 つまで
        let mut ple_head_offsets = ple_head_offsets;
        let mut ple_head_vocab_sizes = ple_head_vocab_sizes;
        let mut ple_layer_multipliers = ple_layer_multipliers;
        if !ple_layers.is_empty() {
            if ple_layers.len() != 1 {
                errs.push(format!(
                    "qwen4exp.ple.layers: 1 層でなければならない (実際 {} 層: {ple_layers:?})",
                    ple_layers.len()
                ));
            }
            for &il in &ple_layers {
                if il >= n_layer {
                    errs.push(format!(
                        "qwen4exp.ple.layers: 層 {il} は範囲外 (block_count = {n_layer})"
                    ));
                } else if full_attn_interval != 0 && (il + 1) % full_attn_interval == 0 {
                    // PLE の conv 履歴は recurrent cache の 1 行なので linear 層だけ
                    errs.push(format!(
                        "qwen4exp.ple.layers: 層 {il} は full attention 層 (PLE は ssm 層のみ)"
                    ));
                }
            }
            if ple_ngram_size < 2 {
                errs.push(format!(
                    "qwen4exp.ple.ngram_size: {ple_ngram_size} は 2 未満"
                ));
            }
            if ple_heads_per_ngram == 0 {
                errs.push("qwen4exp.ple.heads_per_ngram: 0 は不正".to_string());
            }
            if ple_conv_kernel == 0 {
                errs.push("qwen4exp.ple.conv_kernel: 0 は不正".to_string());
            }
            if n_embd_per_layer == 0 {
                errs.push("qwen4exp.embedding_length_per_layer_input: 0 は不正".to_string());
            }
            // 配列長は必要な数を下回らないこと (llama.cpp の require_arr_len と同じ
            // 「以上」。余りは捨てる)
            let need = ple_ngram_size.saturating_sub(1) * ple_heads_per_ngram;
            if ple_layer_multipliers.len() < ple_ngram_size {
                errs.push(format!(
                    "qwen4exp.ple.layer_multipliers: {} 要素が ngram_size = {ple_ngram_size} に足りない",
                    ple_layer_multipliers.len()
                ));
            }
            if ple_head_offsets.len() < need {
                errs.push(format!(
                    "qwen4exp.ple.head_offsets: {} 要素が head 数 {need} に足りない",
                    ple_head_offsets.len()
                ));
            }
            if ple_head_vocab_sizes.len() < need {
                errs.push(format!(
                    "qwen4exp.ple.head_vocab_sizes: {} 要素が head 数 {need} に足りない",
                    ple_head_vocab_sizes.len()
                ));
            }
            ple_layer_multipliers.truncate(ple_ngram_size);
            ple_head_offsets.truncate(need);
            ple_head_vocab_sizes.truncate(need);
            // 行番号は int32 に収まること (gather の index が int32)
            for h in 0..need.min(ple_head_offsets.len()).min(ple_head_vocab_sizes.len()) {
                let off = ple_head_offsets[h] as u64;
                let size = ple_head_vocab_sizes[h] as u64;
                if size == 0 {
                    errs.push(format!("qwen4exp.ple.head_vocab_sizes: head {h} が 0"));
                }
                if off + size > i32::MAX as u64 {
                    errs.push(format!(
                        "qwen4exp.ple.head_offsets: head {h} の範囲 {off}+{size} が int32 に収まらない"
                    ));
                }
            }
        }

        if !errs.is_empty() {
            return Err(Qwen4ExpError(errs));
        }

        Ok(Qwen4ExpParams {
            n_layer,
            n_ctx,
            n_embd,
            n_head,
            n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            rms_eps,
            n_rot,
            rope_sections: [sections[0], sections[1], sections[2], sections[3]],
            rope_freq_base,
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_ff_shexp,
            ssm_d_conv,
            ssm_d_state,
            ssm_n_group,
            ssm_dt_rank,
            ssm_d_inner,
            full_attn_interval,
            n_hc,
            hc_low_rank,
            indexer_n_head,
            indexer_head_size,
            indexer_top_k,
            compress_ratios,
            n_embd_per_layer,
            ple_layers,
            ple_ngram_size,
            ple_heads_per_ngram,
            ple_conv_kernel,
            ple_eos_token_id,
            ple_image_token_id,
            ple_layer_multipliers,
            ple_head_offsets,
            ple_head_vocab_sizes,
            n_vocab,
        })
    }

    /// `embedding_length * hyper_connection.count` = 10240
    pub fn hc_dim(&self) -> usize {
        self.n_embd * self.n_hc
    }

    /// PLE の head 数 = `(ngram_size - 1) * heads_per_ngram` = 16
    pub fn n_ple_heads(&self) -> usize {
        if self.ple_layers.is_empty() {
            0
        } else {
            self.ple_ngram_size.saturating_sub(1) * self.ple_heads_per_ngram
        }
    }

    /// PLE 表に必要な最小行数 = `max(head_offsets[h] + head_vocab_sizes[h])`
    /// (目録では 320001446。実テンソルは 320001536 行 = 90 行パディング)。
    pub fn ple_min_rows(&self) -> u64 {
        let n = self
            .n_ple_heads()
            .min(self.ple_head_offsets.len())
            .min(self.ple_head_vocab_sizes.len());
        let mut max = 0u64;
        for h in 0..n {
            max = max.max(self.ple_head_offsets[h] as u64 + self.ple_head_vocab_sizes[h] as u64);
        }
        max
    }

    /// ssm の k/v head 次元 (`ssm.state_size * ssm.group_count` = 2048)
    pub fn ssm_key_dim(&self) -> usize {
        self.ssm_d_state * self.ssm_n_group
    }

    /// ssm の v 次元 (`ssm.inner_size` = 6144)
    pub fn ssm_value_dim(&self) -> usize {
        self.ssm_d_inner
    }

    /// `ssm_conv1d` / `attn_qkv` の第 2 次元 = `2 * key_dim + value_dim` = 10240
    pub fn ssm_conv_dim(&self) -> usize {
        2 * self.ssm_key_dim() + self.ssm_value_dim()
    }

    /// QSA の pool 比 (compress_ratios の非ゼロ値。0 = QSA 無し)
    pub fn indexer_kpool(&self) -> usize {
        self.compress_ratios
            .iter()
            .copied()
            .find(|&r| r != 0)
            .unwrap_or(0) as usize
    }

    /// PLE を持つ層か
    pub fn is_ple(&self, il: usize) -> bool {
        self.ple_layers.contains(&il)
    }

    /// 層の種別。範囲外 (il >= block_count) はエラー。
    pub fn layer_kind(&self, il: usize) -> Result<LayerKind> {
        if il >= self.n_layer {
            return Err(one(format!(
                "層 {il} は範囲外 (block_count = {})",
                self.n_layer
            )));
        }
        if self.full_attn_interval == 0 {
            return Err(one(
                "qwen4exp.full_attention_interval: 0 では層の種別を決められない".to_string(),
            ));
        }
        if (il + 1) % self.full_attn_interval == 0 {
            Ok(LayerKind::FullAttn)
        } else {
            Ok(LayerKind::Ssm)
        }
    }

    /// full attention 層の数 (48 層・interval 4 なら 12)
    pub fn n_full_attn_layers(&self) -> usize {
        (0..self.n_layer)
            .filter(|&il| self.layer_kind(il) == Ok(LayerKind::FullAttn))
            .count()
    }

    /// ssm 層の数 (48 層・interval 4 なら 36)
    pub fn n_ssm_layers(&self) -> usize {
        self.n_layer - self.n_full_attn_layers()
    }

    /// task 854: 文脈が 1 トークン伸びるごとに KV キャッシュが増えるバイト数。
    ///
    /// full attention 層だけが文脈長に比例する (ssm 層の状態は層あたり固定長)。
    /// ホスト側の `AttnCache` は f32 の K + V を持つので、その形で見積る
    /// (`DENNO_EXPERT_VRAM_DYNAMIC` の上限計算にだけ使う。数値経路は触らない)。
    pub fn kv_bytes_per_token(&self) -> usize {
        let row = self
            .n_head_kv
            .saturating_mul(self.n_embd_head_k.saturating_add(self.n_embd_head_v));
        self.n_full_attn_layers()
            .saturating_mul(row)
            .saturating_mul(std::mem::size_of::<f32>())
    }
}

// ---------------------------------------------------------------------------
// メタデータ読み取りのエラー収集
// ---------------------------------------------------------------------------

struct Acc {
    errs: Vec<String>,
}

impl Acc {
    fn new() -> Self {
        Acc { errs: Vec::new() }
    }

    fn push(&mut self, msg: impl Into<String>) {
        self.errs.push(msg.into());
    }

    fn u(&mut self, f: &GgufFile, key: &str) -> Option<usize> {
        let full = format!("qwen4exp.{key}");
        match f.get(&full) {
            None => {
                self.push(format!("{full}: キーが無い"));
                None
            }
            Some(v) => match v.as_u64() {
                Some(x) => Some(x as usize),
                None => {
                    self.push(format!("{full}: 整数でない"));
                    None
                }
            },
        }
    }

    fn u32(&mut self, f: &GgufFile, key: &str) -> Option<u32> {
        let full = format!("qwen4exp.{key}");
        match f.get(&full) {
            None => {
                self.push(format!("{full}: キーが無い"));
                None
            }
            Some(v) => match v.as_u64() {
                Some(x) if x <= u32::MAX as u64 => Some(x as u32),
                Some(x) => {
                    self.push(format!("{full}: {x} が u32 に収まらない"));
                    None
                }
                None => {
                    self.push(format!("{full}: 整数でない"));
                    None
                }
            },
        }
    }

    /// 任意キー (無ければ None、エラーにしない)。
    fn u32_opt(&mut self, f: &GgufFile, key: &str) -> Option<u32> {
        let full = format!("qwen4exp.{key}");
        match f.get(&full) {
            None => None,
            Some(v) => match v.as_u64() {
                Some(x) if x <= u32::MAX as u64 => Some(x as u32),
                Some(x) => {
                    self.push(format!("{full}: {x} が u32 に収まらない"));
                    None
                }
                None => {
                    self.push(format!("{full}: 整数でない"));
                    None
                }
            },
        }
    }

    fn f32(&mut self, f: &GgufFile, key: &str) -> Option<f32> {
        let full = format!("qwen4exp.{key}");
        match f.get(&full) {
            None => {
                self.push(format!("{full}: キーが無い"));
                None
            }
            Some(v) => match v.as_f32() {
                Some(x) => Some(x),
                None => {
                    self.push(format!("{full}: 実数でない"));
                    None
                }
            },
        }
    }

    /// 配列の要素数だけ読む (`tokenizer.ggml.tokens` のように中身が要らないとき)。
    fn arr_len(&mut self, f: &GgufFile, key: &str) -> Option<usize> {
        match f.get(key) {
            None => {
                self.push(format!("{key}: キーが無い (n_vocab を決められない)"));
                None
            }
            Some(v) => match v.as_array_len() {
                Some(n) => Some(n),
                None => {
                    self.push(format!("{key}: 配列でない"));
                    None
                }
            },
        }
    }

    /// u32 配列。`exact` があれば長さの完全一致、無ければ `min_len` 以上を要求する
    /// (llama.cpp の `get_arr_n(..., required)` は「以上」)。
    fn arr_u32(
        &mut self,
        f: &GgufFile,
        key: &str,
        min_len: usize,
        exact: Option<usize>,
    ) -> Option<Vec<u32>> {
        let full = format!("qwen4exp.{key}");
        let Some(v) = f.get(&full) else {
            self.push(format!("{full}: キーが無い"));
            return None;
        };
        let GgufValue::Arr(items) = v else {
            self.push(format!("{full}: 配列でない"));
            return None;
        };
        if let Some(n) = exact {
            if items.len() != n {
                self.push(format!("{full}: {n} 要素でない ({} 要素)", items.len()));
                return None;
            }
        } else if items.len() < min_len {
            self.push(format!(
                "{full}: {} 要素が {min_len} 要素に足りない",
                items.len()
            ));
            return None;
        }
        let mut out = Vec::with_capacity(items.len());
        for (i, x) in items.iter().enumerate() {
            match x.as_u64() {
                Some(n) if n <= u32::MAX as u64 => out.push(n as u32),
                Some(n) => {
                    self.push(format!("{full}: 要素 {i} の {n} が u32 に収まらない"));
                    return None;
                }
                None => {
                    self.push(format!("{full}: 要素 {i} が整数でない"));
                    return None;
                }
            }
        }
        Some(out)
    }

    /// u64 配列 (ple.layer_multipliers は u32 に収まらない値を持つ)。
    fn arr_u64(&mut self, f: &GgufFile, key: &str, min_len: usize) -> Option<Vec<u64>> {
        let full = format!("qwen4exp.{key}");
        let Some(v) = f.get(&full) else {
            self.push(format!("{full}: キーが無い"));
            return None;
        };
        let GgufValue::Arr(items) = v else {
            self.push(format!("{full}: 配列でない"));
            return None;
        };
        if items.len() < min_len {
            self.push(format!(
                "{full}: {} 要素が {min_len} 要素に足りない",
                items.len()
            ));
            return None;
        }
        let mut out = Vec::with_capacity(items.len());
        for (i, x) in items.iter().enumerate() {
            match x.as_u64() {
                Some(n) => out.push(n),
                None => {
                    self.push(format!("{full}: 要素 {i} が整数でない"));
                    return None;
                }
            }
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// 期待テンソル表
// ---------------------------------------------------------------------------

/// 期待する 1 本。名前は `blk.{il}.` まで含んだ完全名。
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedTensor {
    pub name: String,
    /// GGUF の ne 順 (ne[0] が最内)
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// `ty` 以外に許す量子型 (空なら `ty` だけ)。
    ///
    /// UD 量子化 (`UD-Q4_K_XL`) は層ごとに型を変える。実モデルでは
    /// `blk.2` の `ffn_gate_exps` / `ffn_up_exps` が Q5_K、`blk.2 / 4 / 30 / 46 / 47` の
    /// `ffn_down_exps` が Q8_0 だった。expert の 3 本はエンジンが扱える量子型の
    /// どれでもよい (`Q4_K` / `Q5_K` / `Q5_1` / `Q8_0`)。形状は厳密に一致させる。
    pub ty_alt: &'static [GgmlType],
    /// true なら `dims` の末尾は「以上」を要求する
    /// (`per_layer_token_embd` の行数は converter がパディングする)。
    pub min_last_dim: bool,
}

/// expert の 3 本 (`ffn_{gate,up,down}_exps.weight`) に許す量子型。
/// E1 / E4 / E5 の GPU 経路と `expert_gemv` が対応済みの 4 型。
pub const EXPERT_TYS: &[GgmlType] = &[
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q5_1,
    GgmlType::Q8_0,
];

/// UD 量子化で型が揺れるテンソルの代替型。名前で決める (表の各行に書かない)。
fn ty_alternates(name: &str) -> &'static [GgmlType] {
    if name.ends_with("_exps.weight") {
        EXPERT_TYS
    } else {
        &[]
    }
}

impl ExpectedTensor {
    fn fmt_dims(&self) -> String {
        if !self.min_last_dim {
            return fmt_dims(&self.dims);
        }
        let mut s = String::from("[");
        for (i, x) in self.dims.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            if i + 1 == self.dims.len() {
                s.push_str(&format!("{x} 以上"));
            } else {
                s.push_str(&x.to_string());
            }
        }
        s.push(']');
        s
    }
}

/// 形状をメタデータから組み立てる規則。表 (const 配列) は名前とこの規則だけを持ち、
/// 数値はここで params から引く (目録の実値をコードに埋めない)。
#[derive(Debug, Clone, Copy)]
enum Shape {
    /// [n_embd]
    Emb,
    /// [n_embd, n_expert]
    EmbExpert,
    /// [n_embd, n_ff_exp, n_expert]
    EmbFfExpExpert,
    /// [n_ff_exp, n_embd, n_expert]
    FfExpEmbExpert,
    /// [n_embd, n_ff_shexp]
    EmbFfShexp,
    /// [n_ff_shexp, n_embd]
    FfShexpEmb,
    /// [hc_dim]
    HcDim,
    /// [hc_lr, hc_dim]
    HcLrHcDim,
    /// [hc_dim, hc_lr]
    HcDimHcLr,
    /// [hc_dim, n_hc]
    HcDimHc,
    /// [ssm_dt_rank]
    SsmDtRank,
    /// [n_embd, ssm_dt_rank]
    EmbSsmDtRank,
    /// [ssm_d_conv, conv_dim]
    SsmConv1d,
    /// [ssm_d_state]
    SsmState,
    /// [ssm.inner_size, n_embd]
    SsmInnerEmb,
    /// [n_embd, conv_dim] (= attn_qkv: q(value) + k(key) + v(key))
    SsmEmbConvDim,
    /// [n_embd, ssm.inner_size] (= attn_gate)
    SsmEmbInner,
    /// [n_embd, 2 * n_head * head_k] (= attn_q: q + gate 融合)
    AttnQ,
    /// [n_embd, n_head_kv * head_k]
    AttnK,
    /// [n_embd, n_head_kv * head_v]
    AttnV,
    /// [head_k]
    HeadDimK,
    /// [n_head * head_v, n_embd]
    AttnOut,
    /// [n_embd, indexer_n_head * indexer_head_size]
    IdxQ,
    /// [n_embd, indexer_head_size]
    IdxK,
    /// [indexer_head_size]
    IdxDim,
    /// [n_embd, hc_dim]
    EmbHcDim,
    /// [n_embd, n_embd]
    EmbEmb,
    /// [ple_conv_kernel, hc_dim]
    PleConv1d,
    /// [n_embd, n_vocab]
    EmbVocab,
    /// [n_embd_per_layer, ple_min_rows 以上]
    PleTable,
}

impl Shape {
    fn dims(self, p: &Qwen4ExpParams) -> (Vec<u64>, bool) {
        use Shape::*;
        let u = |x: usize| x as u64;
        match self {
            Emb => (vec![u(p.n_embd)], false),
            EmbExpert => (vec![u(p.n_embd), u(p.n_expert)], false),
            EmbFfExpExpert => (vec![u(p.n_embd), u(p.n_ff_exp), u(p.n_expert)], false),
            FfExpEmbExpert => (vec![u(p.n_ff_exp), u(p.n_embd), u(p.n_expert)], false),
            EmbFfShexp => (vec![u(p.n_embd), u(p.n_ff_shexp)], false),
            FfShexpEmb => (vec![u(p.n_ff_shexp), u(p.n_embd)], false),
            HcDim => (vec![u(p.hc_dim())], false),
            HcLrHcDim => (vec![u(p.hc_low_rank), u(p.hc_dim())], false),
            HcDimHcLr => (vec![u(p.hc_dim()), u(p.hc_low_rank)], false),
            HcDimHc => (vec![u(p.hc_dim()), u(p.n_hc)], false),
            SsmDtRank => (vec![u(p.ssm_dt_rank)], false),
            EmbSsmDtRank => (vec![u(p.n_embd), u(p.ssm_dt_rank)], false),
            SsmConv1d => (vec![u(p.ssm_d_conv), u(p.ssm_conv_dim())], false),
            SsmState => (vec![u(p.ssm_d_state)], false),
            SsmInnerEmb => (vec![u(p.ssm_value_dim()), u(p.n_embd)], false),
            SsmEmbConvDim => (vec![u(p.n_embd), u(p.ssm_conv_dim())], false),
            SsmEmbInner => (vec![u(p.n_embd), u(p.ssm_value_dim())], false),
            AttnQ => (vec![u(p.n_embd), u(2 * p.n_head * p.n_embd_head_k)], false),
            AttnK => (vec![u(p.n_embd), u(p.n_head_kv * p.n_embd_head_k)], false),
            AttnV => (vec![u(p.n_embd), u(p.n_head_kv * p.n_embd_head_v)], false),
            HeadDimK => (vec![u(p.n_embd_head_k)], false),
            AttnOut => (vec![u(p.n_head * p.n_embd_head_v), u(p.n_embd)], false),
            IdxQ => (
                vec![
                    u(p.n_embd),
                    u(p.indexer_n_head * p.indexer_head_size),
                ],
                false,
            ),
            IdxK => (vec![u(p.n_embd), u(p.indexer_head_size)], false),
            IdxDim => (vec![u(p.indexer_head_size)], false),
            EmbHcDim => (vec![u(p.n_embd), u(p.hc_dim())], false),
            EmbEmb => (vec![u(p.n_embd), u(p.n_embd)], false),
            PleConv1d => (vec![u(p.ple_conv_kernel), u(p.hc_dim())], false),
            EmbVocab => (vec![u(p.n_embd), u(p.n_vocab)], false),
            PleTable => (vec![u(p.n_embd_per_layer), p.ple_min_rows()], true),
        }
    }
}

/// 全 48 層に共通の 16 本 (`blk.N.` の後ろ)。
const COMMON: &[(&str, Shape, GgmlType)] = &[
    ("ffn_gate_inp.weight", Shape::EmbExpert, GgmlType::F32),
    ("ffn_gate_inp_shexp.weight", Shape::Emb, GgmlType::F32),
    ("ffn_gate_exps.weight", Shape::EmbFfExpExpert, GgmlType::Q4K),
    ("ffn_up_exps.weight", Shape::EmbFfExpExpert, GgmlType::Q4K),
    ("ffn_down_exps.weight", Shape::FfExpEmbExpert, GgmlType::Q5_1),
    ("ffn_gate_shexp.weight", Shape::EmbFfShexp, GgmlType::Q8_0),
    ("ffn_up_shexp.weight", Shape::EmbFfShexp, GgmlType::Q8_0),
    ("ffn_down_shexp.weight", Shape::FfShexpEmb, GgmlType::Q8_0),
    ("hc_attn_norm.weight", Shape::HcDim, GgmlType::F32),
    ("hc_attn_up.weight", Shape::HcLrHcDim, GgmlType::Q8_0),
    ("hc_attn_down.weight", Shape::HcDimHcLr, GgmlType::Q8_0),
    ("hc_attn_inject.weight", Shape::HcDimHc, GgmlType::F32),
    ("hc_ffn_norm.weight", Shape::HcDim, GgmlType::F32),
    ("hc_ffn_up.weight", Shape::HcLrHcDim, GgmlType::Q8_0),
    ("hc_ffn_down.weight", Shape::HcDimHcLr, GgmlType::Q8_0),
    ("hc_ffn_inject.weight", Shape::HcDimHc, GgmlType::F32),
];

/// ssm 層だけが持つ 9 本。`ssm_a` と `ssm_dt.bias` は `.weight` が付かない
/// (目録でも `blk.0.ssm_a` / `blk.0.ssm_dt.bias` そのまま)。
const SSM: &[(&str, Shape, GgmlType)] = &[
    ("ssm_a", Shape::SsmDtRank, GgmlType::F32),
    ("ssm_dt.bias", Shape::SsmDtRank, GgmlType::F32),
    ("ssm_alpha.weight", Shape::EmbSsmDtRank, GgmlType::F32),
    ("ssm_beta.weight", Shape::EmbSsmDtRank, GgmlType::F32),
    ("ssm_conv1d.weight", Shape::SsmConv1d, GgmlType::F32),
    ("ssm_norm.weight", Shape::SsmState, GgmlType::F32),
    ("ssm_out.weight", Shape::SsmInnerEmb, GgmlType::Q8_0),
    ("attn_qkv.weight", Shape::SsmEmbConvDim, GgmlType::Q8_0),
    ("attn_gate.weight", Shape::SsmEmbInner, GgmlType::Q8_0),
];

/// full attention 層だけが持つ 10 本 (`indexer.*_proj` は BF16)。
const FULL: &[(&str, Shape, GgmlType)] = &[
    ("attn_q.weight", Shape::AttnQ, GgmlType::Q8_0),
    ("attn_k.weight", Shape::AttnK, GgmlType::Q8_0),
    ("attn_v.weight", Shape::AttnV, GgmlType::Q8_0),
    ("attn_q_norm.weight", Shape::HeadDimK, GgmlType::F32),
    ("attn_k_norm.weight", Shape::HeadDimK, GgmlType::F32),
    ("attn_output.weight", Shape::AttnOut, GgmlType::Q8_0),
    ("indexer.q_proj.weight", Shape::IdxQ, GgmlType::Bf16),
    ("indexer.k_proj.weight", Shape::IdxK, GgmlType::Bf16),
    ("indexer.q_norm.weight", Shape::IdxDim, GgmlType::F32),
    ("indexer.k_norm.weight", Shape::IdxDim, GgmlType::F32),
];

/// PLE 層 (`ple.layers` の層) だけが持つ 6 本。
const PLE: &[(&str, Shape, GgmlType)] = &[
    ("ple_key.weight", Shape::EmbHcDim, GgmlType::Q8_0),
    ("ple_value.weight", Shape::EmbEmb, GgmlType::Q8_0),
    ("ple_norm_key.weight", Shape::HcDim, GgmlType::F32),
    ("ple_norm_query.weight", Shape::HcDim, GgmlType::F32),
    ("ple_norm_conv.weight", Shape::HcDim, GgmlType::F32),
    ("ple_conv1d.weight", Shape::PleConv1d, GgmlType::F32),
];

/// 層に属さないテンソル (完全名)。
const GLOBAL: &[(&str, Shape, GgmlType)] = &[
    ("token_embd.weight", Shape::EmbVocab, GgmlType::Q8_0),
    ("output.weight", Shape::EmbVocab, GgmlType::Q8_0),
    ("output_hc_norm.weight", Shape::HcDim, GgmlType::F32),
    ("output_hc_up.weight", Shape::HcLrHcDim, GgmlType::Q8_0),
    ("output_hc_down.weight", Shape::HcDimHcLr, GgmlType::Q8_0),
    ("per_layer_token_embd.weight", Shape::PleTable, GgmlType::IQ4Nl),
];

fn make(name: String, shape: Shape, ty: GgmlType, p: &Qwen4ExpParams) -> ExpectedTensor {
    let (dims, min_last_dim) = shape.dims(p);
    let ty_alt = ty_alternates(&name);
    ExpectedTensor {
        name,
        dims,
        ty,
        ty_alt,
        min_last_dim,
    }
}

/// 層に属さない 6 本 (PLE が無いモデルでは `per_layer_token_embd` を除く 5 本)。
pub fn expected_global_tensors(p: &Qwen4ExpParams) -> Vec<ExpectedTensor> {
    GLOBAL
        .iter()
        .filter(|(_, shape, _)| !matches!(shape, Shape::PleTable) || p.n_ple_heads() > 0)
        .map(|(name, shape, ty)| make((*name).to_string(), *shape, *ty, p))
        .collect()
}

/// 層 1 つぶんの期待テンソル (共通 16 + 種別固有 + PLE)。
pub fn expected_layer_tensors(p: &Qwen4ExpParams, il: usize) -> Result<Vec<ExpectedTensor>> {
    let kind = p.layer_kind(il)?;
    let mut out = Vec::with_capacity(COMMON.len() + FULL.len() + PLE.len());
    for (suffix, shape, ty) in COMMON {
        out.push(make(format!("blk.{il}.{suffix}"), *shape, *ty, p));
    }
    let specific = match kind {
        LayerKind::Ssm => SSM,
        LayerKind::FullAttn => FULL,
    };
    for (suffix, shape, ty) in specific {
        out.push(make(format!("blk.{il}.{suffix}"), *shape, *ty, p));
    }
    if p.is_ple(il) {
        for (suffix, shape, ty) in PLE {
            out.push(make(format!("blk.{il}.{suffix}"), *shape, *ty, p));
        }
    }
    Ok(out)
}

/// モデル全体の期待テンソル。目録の実ファイル (48 層・512 expert) では 1224 本。
pub fn expected_tensors(p: &Qwen4ExpParams) -> Result<Vec<ExpectedTensor>> {
    let mut out = expected_global_tensors(p);
    out.reserve(p.n_layer * (COMMON.len() + FULL.len()));
    for il in 0..p.n_layer {
        out.extend(expected_layer_tensors(p, il)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 検証
// ---------------------------------------------------------------------------

/// GGUF の ne から末尾の 1 を落とす (`TensorRef::ne` と同じ規則)。
fn trimmed_dims(t: &GgufTensorInfo) -> Vec<u64> {
    let mut v = t.dims.to_vec();
    while v.len() > 1 && v.last() == Some(&1) {
        v.pop();
    }
    v
}

fn check_one(e: &ExpectedTensor, t: &GgufTensorInfo, errs: &mut Vec<String>) {
    let got = trimmed_dims(t);
    let dims_ok = if e.min_last_dim {
        got.len() == e.dims.len()
            && got[..got.len() - 1] == e.dims[..e.dims.len() - 1]
            && got[got.len() - 1] >= e.dims[e.dims.len() - 1]
    } else {
        got == e.dims
    };
    if !dims_ok {
        errs.push(format!(
            "{}: 形状 {} が期待 {} と違う",
            e.name,
            fmt_dims(&got),
            e.fmt_dims()
        ));
    }
    // 2026-10-31: 量子型の一致は「構造の取り違え検出」ではなく「量子化の選択」の
    // 話なので、**デコードできる型なら通す** (Swift 1.5 の重みは層ごとに型が混在し、
    // 期待表は UD-Q4_K_XL 時代の固定値)。形状と名前の照合は従来どおり厳密。
    let ty_decodable = crate::kquant::dequant_supported(t.ty);
    if t.ty != e.ty && !e.ty_alt.contains(&t.ty) && !ty_decodable {
        let allowed = if e.ty_alt.is_empty() {
            ty_name(e.ty).to_string()
        } else {
            let mut v: Vec<&str> = vec![ty_name(e.ty)];
            for x in e.ty_alt {
                let n = ty_name(*x);
                if !v.contains(&n) {
                    v.push(n);
                }
            }
            v.join(" / ")
        };
        errs.push(format!(
            "{}: 型 {} が期待 {} と違う",
            e.name,
            ty_name(t.ty),
            allowed
        ));
    }
}

/// 全体を 1 回検証する。最初の 1 件で止めず、**全部集めて**返す
/// (1 行 1 件。名前・期待・実際が入る)。
///
/// - メタデータの欠落・型違い・値の不整合
/// - 期待テンソルの欠落 / 形状不一致 / 量子型不一致
/// - `blk.N.*` で期待表に無いテンソル (層種別の取り違えを含む)
pub fn validate(f: &GgufFile) -> Result<Qwen4ExpParams> {
    let p = Qwen4ExpParams::from_meta(f)?;
    let expected = expected_tensors(&p)?;

    let mut errs = Vec::new();
    let mut names: HashSet<&str> = HashSet::with_capacity(expected.len());
    for e in &expected {
        names.insert(e.name.as_str());
        match f.tensor(&e.name) {
            None => errs.push(format!(
                "{}: 見つからない (期待 {} {})",
                e.name,
                e.fmt_dims(),
                ty_name(e.ty)
            )),
            Some(t) => check_one(e, t, &mut errs),
        }
    }

    // 期待表に無い blk.N.* = 余分。ssm 層に attn_q を置いたような
    // 層種別の取り違えはここで捕まる。
    for name in &f.tensor_order {
        if names.contains(name.as_str()) || !name.starts_with("blk.") {
            continue;
        }
        let il = name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|n| n.parse::<usize>().ok());
        let at = match il {
            Some(il) if il < p.n_layer => match p.layer_kind(il) {
                Ok(k) => format!(" (層 {il} は {} 層)", k.name()),
                Err(_) => String::new(),
            },
            Some(il) => format!(" (層 {il} は範囲外: block_count = {})", p.n_layer),
            None => String::new(),
        };
        errs.push(format!("{name}: 余分なテンソル{at}"));
    }

    if errs.is_empty() {
        // task 261: `DENNO_TENSOR_TYPES=1` のときだけ、量子型の分布を役割ごとに
        // 1 行ずつ出す (診断用。既定 off なので数値・ログは従来と同一)。
        if std::env::var_os(TENSOR_TYPES_ENV).is_some() {
            for line in tensor_type_table(f, &p) {
                eprintln!("{line}");
            }
        }
        Ok(p)
    } else {
        Err(Qwen4ExpError(errs))
    }
}

/// `DENNO_TENSOR_TYPES`: 読み込んだ GGUF の量子型を役割ごとに stderr へ出す。既定 off。
pub const TENSOR_TYPES_ENV: &str = "DENNO_TENSOR_TYPES";

/// テンソル名 → 役割 (診断の集計キー)。`blk.N.` を落として後ろで見る。
fn tensor_role(name: &str) -> &'static str {
    let suffix = name
        .strip_prefix("blk.")
        .and_then(|r| r.split_once('.'))
        .map(|(_, s)| s)
        .unwrap_or(name);
    if suffix.starts_with("ffn_gate_exps") {
        "expert_gate"
    } else if suffix.starts_with("ffn_up_exps") {
        "expert_up"
    } else if suffix.starts_with("ffn_down_exps") {
        "expert_down"
    } else if suffix.starts_with("ffn_gate_inp") {
        "router"
    } else if suffix.starts_with("ffn_") && suffix.contains("shexp") {
        "shared"
    } else if suffix.starts_with("attn_") || suffix.starts_with("ssm_") {
        "attn_gdn"
    } else if suffix.starts_with("indexer.") {
        "indexer"
    } else if suffix.starts_with("hc_") {
        "mix"
    } else if suffix.starts_with("ple_") {
        "ple_layer"
    } else if suffix == "per_layer_token_embd.weight" {
        "ple_table"
    } else if suffix == "token_embd.weight" {
        "embed"
    } else if suffix.starts_with("output") {
        "head"
    } else {
        "other"
    }
}

/// 役割ごとの量子型ヒストグラム + expert 1 本のバイト数 (診断用の文字列)。
///
/// `DENNO_TENSOR_TYPES=1` のとき [`validate`] が stderr へ出す。実 GGUF を
/// 読まないと分からない「どの型が GPU 経路に載るか」を 1 画面で見るためのもの。
pub fn tensor_type_table(f: &GgufFile, p: &Qwen4ExpParams) -> Vec<String> {
    use std::collections::BTreeMap;
    // 役割 -> (型名 -> (本数, バイト数))
    let mut m: BTreeMap<&'static str, BTreeMap<&'static str, (usize, u64)>> = BTreeMap::new();
    let mut expert_bytes = 0u64;
    let mut expert_n = 0u64;
    let mut layer0 = 0u64;
    for name in &f.tensor_order {
        let Some(t) = f.tensor(name) else { continue };
        let role = tensor_role(name);
        let e = m
            .entry(role)
            .or_default()
            .entry(ty_name(t.ty))
            .or_insert((0, 0));
        e.0 += 1;
        e.1 += t.nbytes;
        if role.starts_with("expert_") {
            expert_bytes += t.nbytes;
            expert_n += p.n_expert as u64;
            if name.starts_with("blk.0.") {
                layer0 += t.nbytes;
            }
        }
    }
    let mut out = Vec::new();
    for (role, tys) in &m {
        let total: u64 = tys.values().map(|(_, b)| *b).sum();
        let parts: Vec<String> = tys
            .iter()
            .map(|(t, (n, b))| format!("{t}x{n} {:.1}MB", *b as f64 / 1e6))
            .collect();
        out.push(format!(
            "[ttypes] {role}: {:.2}GB ({})",
            total as f64 / 1e9,
            parts.join(" / ")
        ));
    }
    if layer0 > 0 {
        let cap_mb = crate::model::expert_vram::vram_mb_from_value(
            std::env::var(crate::model::expert_vram::EXPERT_VRAM_MB_ENV).ok().as_deref(),
        );
        let cap = cap_mb as u64 * 1024 * 1024;
        // 層 0 の 3 行列の合計 = 1 expert のバイト数 × n_expert。
        let one = layer0 / p.n_expert.max(1) as u64;
        let fit = if one > 0 { cap / one } else { 0 };
        let total_n = p.n_layer as u64 * p.n_expert as u64;
        out.push(format!(
            "[ttypes] expert: 1 本 {:.3}MB (層 0) / 平均 {:.3}MB / 全 {} 本 {:.1}GB / \
             予算 {}MiB で {} 本 ({:.1}%)",
            one as f64 / 1e6,
            expert_bytes as f64 / expert_n as f64 / 1e6,
            total_n,
            expert_bytes as f64 / 1e9,
            cap_mb,
            fit,
            if total_n > 0 {
                fit as f64 / total_n as f64 * 100.0
            } else {
                0.0
            }
        ));
    }
    out
}

// ---------------------------------------------------------------------------
// 層 1 つぶんのテンソル参照
// ---------------------------------------------------------------------------

/// ssm (linear attention) 層の固有テンソル。
#[derive(Debug, Clone)]
pub struct SsmTensors {
    /// `ssm_a` (サフィックス無し) — [ssm_dt_rank]
    pub ssm_a: TensorRef,
    /// `ssm_dt.bias` — [ssm_dt_rank]
    pub ssm_dt: TensorRef,
    /// `ssm_alpha.weight` — [n_embd, ssm_dt_rank]
    pub ssm_alpha: TensorRef,
    /// `ssm_beta.weight` — [n_embd, ssm_dt_rank]
    pub ssm_beta: TensorRef,
    /// `ssm_conv1d.weight` — [ssm_d_conv, conv_dim]
    pub ssm_conv1d: TensorRef,
    /// `ssm_norm.weight` — [ssm_d_state]
    pub ssm_norm: TensorRef,
    /// `ssm_out.weight` — [ssm.inner_size, n_embd]
    pub ssm_out: TensorRef,
    /// `attn_qkv.weight` — [n_embd, conv_dim] (q(value) + k(key) + v(key))
    pub attn_qkv: TensorRef,
    /// `attn_gate.weight` — [n_embd, ssm.inner_size]
    pub attn_gate: TensorRef,
}

/// full attention 層の固有テンソル。
#[derive(Debug, Clone)]
pub struct FullAttnTensors {
    /// `attn_q.weight` — [n_embd, 2 * n_head * head_k] (q と gate の融合)
    pub attn_q: TensorRef,
    /// `attn_k.weight` — [n_embd, n_head_kv * head_k]
    pub attn_k: TensorRef,
    /// `attn_v.weight` — [n_embd, n_head_kv * head_v]
    pub attn_v: TensorRef,
    /// `attn_q_norm.weight` — [head_k]
    pub attn_q_norm: TensorRef,
    /// `attn_k_norm.weight` — [head_k]
    pub attn_k_norm: TensorRef,
    /// `attn_output.weight` — [n_head * head_v, n_embd]
    pub attn_output: TensorRef,
    /// `indexer.q_proj.weight` — [n_embd, indexer_n_head * indexer_head_size] (BF16)
    pub indexer_q_proj: TensorRef,
    /// `indexer.k_proj.weight` — [n_embd, indexer_head_size] (BF16)
    pub indexer_k_proj: TensorRef,
    /// `indexer.q_norm.weight` — [indexer_head_size]
    pub indexer_q_norm: TensorRef,
    /// `indexer.k_norm.weight` — [indexer_head_size]
    pub indexer_k_norm: TensorRef,
}

/// PLE 層の固有テンソル (`ple.layers` の層のみ)。
#[derive(Debug, Clone)]
pub struct PleTensors {
    /// `ple_key.weight` — [n_embd, hc_dim]
    pub ple_key: TensorRef,
    /// `ple_value.weight` — [n_embd, n_embd]
    pub ple_value: TensorRef,
    /// `ple_norm_key.weight` — [hc_dim]
    pub ple_norm_key: TensorRef,
    /// `ple_norm_query.weight` — [hc_dim]
    pub ple_norm_query: TensorRef,
    /// `ple_norm_conv.weight` — [hc_dim]
    pub ple_norm_conv: TensorRef,
    /// `ple_conv1d.weight` — [ple_conv_kernel, hc_dim]
    pub ple_conv1d: TensorRef,
}

/// 層の種別ごとの固有テンソル。
#[derive(Debug, Clone)]
pub enum MixerTensors {
    Ssm(SsmTensors),
    FullAttn(FullAttnTensors),
}

/// 層 1 つぶんのテンソル参照。形状・量子型は params と突き合わせて検証済み。
#[derive(Debug, Clone)]
pub struct LayerTensors {
    pub il: usize,
    pub kind: LayerKind,
    /// `ffn_gate_inp.weight` — [n_embd, n_expert] (ルータ)
    pub ffn_gate_inp: TensorRef,
    /// `ffn_gate_inp_shexp.weight` — [n_embd] (shared expert のルータ重み)
    pub ffn_gate_inp_shexp: TensorRef,
    /// `ffn_gate_exps.weight` — [n_embd, n_ff_exp, n_expert]
    pub ffn_gate_exps: TensorRef,
    /// `ffn_up_exps.weight` — [n_embd, n_ff_exp, n_expert]
    pub ffn_up_exps: TensorRef,
    /// `ffn_down_exps.weight` — [n_ff_exp, n_embd, n_expert]
    pub ffn_down_exps: TensorRef,
    /// `ffn_gate_shexp.weight` — [n_embd, n_ff_shexp]
    pub ffn_gate_shexp: TensorRef,
    /// `ffn_up_shexp.weight` — [n_embd, n_ff_shexp]
    pub ffn_up_shexp: TensorRef,
    /// `ffn_down_shexp.weight` — [n_ff_shexp, n_embd]
    pub ffn_down_shexp: TensorRef,
    /// `hc_attn_norm.weight` — [hc_dim]
    pub hc_attn_norm: TensorRef,
    /// `hc_attn_up.weight` — [hc_lr, hc_dim]
    pub hc_attn_up: TensorRef,
    /// `hc_attn_down.weight` — [hc_dim, hc_lr]
    pub hc_attn_down: TensorRef,
    /// `hc_attn_inject.weight` — [hc_dim, n_hc]
    pub hc_attn_inject: TensorRef,
    /// `hc_ffn_norm.weight` — [hc_dim]
    pub hc_ffn_norm: TensorRef,
    /// `hc_ffn_up.weight` — [hc_lr, hc_dim]
    pub hc_ffn_up: TensorRef,
    /// `hc_ffn_down.weight` — [hc_dim, hc_lr]
    pub hc_ffn_down: TensorRef,
    /// `hc_ffn_inject.weight` — [hc_dim, n_hc]
    pub hc_ffn_inject: TensorRef,
    /// トークンミキサー (ssm か full attention)
    pub mixer: MixerTensors,
    /// PLE 層のみ `Some`
    pub ple: Option<PleTensors>,
}

impl LayerTensors {
    /// 層 1 つぶんのテンソルを引き、形状・量子型を検証する。
    /// エラーは 1 行 1 件 (`Vec<String>`)。
    pub fn load(f: &GgufFile, p: &Qwen4ExpParams, il: usize) -> Result<Self> {
        let kind = p.layer_kind(il)?;
        let expected = expected_layer_tensors(p, il)?;

        let mut errs = Vec::new();
        let mut map: HashMap<String, TensorRef> = HashMap::with_capacity(expected.len());
        for e in &expected {
            match f.tensor_ref(&e.name) {
                None => errs.push(format!(
                    "{}: 見つからない (期待 {} {})",
                    e.name,
                    e.fmt_dims(),
                    ty_name(e.ty)
                )),
                Some(t) => {
                    check_one(e, &t.info, &mut errs);
                    map.insert(e.name.clone(), t);
                }
            }
        }
        if !errs.is_empty() {
            return Err(Qwen4ExpError(errs));
        }

        // 以降は全本が検証済みで map にある (無ければ内部不整合として返す)。
        macro_rules! tf {
            ($suffix:expr) => {
                match map.remove(&format!("blk.{il}.{}", $suffix)) {
                    Some(t) => t,
                    None => {
                        return Err(one(format!(
                            "blk.{il}.{}: 検証済みなのに引けない (内部不整合)",
                            $suffix
                        )))
                    }
                }
            };
        }

        let mixer = match kind {
            LayerKind::Ssm => MixerTensors::Ssm(SsmTensors {
                ssm_a: tf!("ssm_a"),
                ssm_dt: tf!("ssm_dt.bias"),
                ssm_alpha: tf!("ssm_alpha.weight"),
                ssm_beta: tf!("ssm_beta.weight"),
                ssm_conv1d: tf!("ssm_conv1d.weight"),
                ssm_norm: tf!("ssm_norm.weight"),
                ssm_out: tf!("ssm_out.weight"),
                attn_qkv: tf!("attn_qkv.weight"),
                attn_gate: tf!("attn_gate.weight"),
            }),
            LayerKind::FullAttn => MixerTensors::FullAttn(FullAttnTensors {
                attn_q: tf!("attn_q.weight"),
                attn_k: tf!("attn_k.weight"),
                attn_v: tf!("attn_v.weight"),
                attn_q_norm: tf!("attn_q_norm.weight"),
                attn_k_norm: tf!("attn_k_norm.weight"),
                attn_output: tf!("attn_output.weight"),
                indexer_q_proj: tf!("indexer.q_proj.weight"),
                indexer_k_proj: tf!("indexer.k_proj.weight"),
                indexer_q_norm: tf!("indexer.q_norm.weight"),
                indexer_k_norm: tf!("indexer.k_norm.weight"),
            }),
        };
        let ple = if p.is_ple(il) {
            Some(PleTensors {
                ple_key: tf!("ple_key.weight"),
                ple_value: tf!("ple_value.weight"),
                ple_norm_key: tf!("ple_norm_key.weight"),
                ple_norm_query: tf!("ple_norm_query.weight"),
                ple_norm_conv: tf!("ple_norm_conv.weight"),
                ple_conv1d: tf!("ple_conv1d.weight"),
            })
        } else {
            None
        };

        Ok(LayerTensors {
            il,
            kind,
            ffn_gate_inp: tf!("ffn_gate_inp.weight"),
            ffn_gate_inp_shexp: tf!("ffn_gate_inp_shexp.weight"),
            ffn_gate_exps: tf!("ffn_gate_exps.weight"),
            ffn_up_exps: tf!("ffn_up_exps.weight"),
            ffn_down_exps: tf!("ffn_down_exps.weight"),
            ffn_gate_shexp: tf!("ffn_gate_shexp.weight"),
            ffn_up_shexp: tf!("ffn_up_shexp.weight"),
            ffn_down_shexp: tf!("ffn_down_shexp.weight"),
            hc_attn_norm: tf!("hc_attn_norm.weight"),
            hc_attn_up: tf!("hc_attn_up.weight"),
            hc_attn_down: tf!("hc_attn_down.weight"),
            hc_attn_inject: tf!("hc_attn_inject.weight"),
            hc_ffn_norm: tf!("hc_ffn_norm.weight"),
            hc_ffn_up: tf!("hc_ffn_up.weight"),
            hc_ffn_down: tf!("hc_ffn_down.weight"),
            hc_ffn_inject: tf!("hc_ffn_inject.weight"),
            mixer,
            ple,
        })
    }
}

// ---------------------------------------------------------------------------
// P-MOE-F1: 1 トークン decode の骨格 (attention / ssm 以外)
// ---------------------------------------------------------------------------
//
// 参照実装の層ループは `qwen4exp.cpp` 486-556 行、`build_hc_mix` 344-420 行、
// `build_hc_combine` 422-460 行、`build_layer_ffn` 1240-1285 行、
// `build_ple` 1467-1555 行、最終 mixer は 564-575 行。
//
// この節が持つのは **形と順序** だけ。attention / ssm の中身は
// [`LayerAttn`] の差し替え口で、F1 では [`IdentityAttn`] (恒等) が入る。
// 数値の正しさは F4 で実モデルを回して判定する。
//
// 丸め順序の契約:
// - 行列積はすべて [`GgufLinear::gemv`] (行昇順・内積は列昇順の f32 逐次加算)。
//   `hyper::hc_mix` の f32 経路と同じ `dot_row` 系の順序なので、重みを
//   f32 に展開して `hc_mix` に渡した場合とビット一致する。
// - `hc_mix` の各段 (RMS → lo → gate → mixed) は `hyper` の関数をそのまま呼ぶ
//   (式をこのファイルに写さない)。

/// 層のトークンミキサー (attention / ssm) の差し替え口。
///
/// `hc_mix` の出力 `cur` (`[n_embd]`) を受けて、その層の出力 (`[n_embd]`) を返す。
pub trait LayerAttn: Send {
    fn forward(&mut self, layer: usize, cur: &[f32]) -> Result<Vec<f32>>;

    /// task 468: 層 `layer` の入力を**デバイスから**渡してよいか。
    ///
    /// `true` を返した層では、呼び出し側が attn の mix の鎖を defer して
    /// `cur` を空で渡す (入力は鎖がデバイスに残した `mixed`)。既定は `false`
    /// = 従来どおりホストの `cur` を渡す。
    fn takes_dev_input(&self, _layer: usize) -> bool {
        false
    }

    /// チャンクぶんの入力をまとめて処理する (バッチ prefill 用)。
    ///
    /// 既定は [`Self::forward`] をトークン順に呼ぶだけなので、数値は逐次と
    /// まったく同じ。行列積をまとめて GPU へ流せる実装だけが上書きする。
    /// 上書きする実装は **状態を `forward` と同じ順で進めること**
    /// (層の中で 1 トークンずつ。KV / GDN の積み方が変わってはいけない)。
    fn forward_batch(&mut self, layer: usize, curs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let mut outs = Vec::with_capacity(curs.len());
        for cur in curs {
            outs.push(self.forward(layer, cur)?);
        }
        Ok(outs)
    }

    /// 平坦な入力 (`nt` 行 × 幅を連結したもの) をそのまま受け取り、平坦な出力を返す。
    ///
    /// task 200: 層優先バッチ prefill の層ループは活性を**もともと平坦な配列**で
    /// 持っている (`mixed_flat`)。従来はここでトークンごとの `Vec<f32>` を作り、
    /// 実装側がまた平坦に戻していたため、1 層チャンクあたり 4 回ぶんの
    /// コピーと `nt` 個のアロケーションが乗っていた。
    ///
    /// 既定は [`Self::forward_batch`] に `Vec<Vec<f32>>` を作って渡し、結果を
    /// 平坦に戻すだけ (数値は同じ)。`Vec<Vec<f32>>` を作らずに回せる実装
    /// ([`crate::model::qwen4exp_attn::FullAttn`] /
    /// [`crate::model::qwen4exp_ssm::SsmAttn`]) が上書きする。
    /// 呼び出し側は `DENNO_PF_ATTN_FLAT` で切り替える。
    fn forward_batch_flat(&mut self, layer: usize, flat: &[f32], nt: usize) -> Result<Vec<f32>> {
        if nt == 0 {
            return Ok(Vec::new());
        }
        if flat.len() % nt != 0 {
            return Err(one(format!(
                "forward_batch_flat: 入力長 {} が nt={nt} で割り切れない",
                flat.len()
            )));
        }
        let w = flat.len() / nt;
        let curs: Vec<Vec<f32>> = flat.chunks_exact(w).map(|c| c.to_vec()).collect();
        let outs = self.forward_batch(layer, &curs)?;
        let mut out = Vec::with_capacity(nt * w);
        for (i, r) in outs.iter().enumerate() {
            if r.len() != w {
                return Err(one(format!(
                    "forward_batch_flat: 出力 {i} の長さ {} が {w} と違う",
                    r.len()
                )));
            }
            out.extend_from_slice(r);
        }
        Ok(out)
    }

    /// 層をまたいで呼び出し順を入れ替えても状態が壊れないか。
    ///
    /// 層優先 (トークンをまとめて層ごとに回す) のバッチ prefill は、1 層ぶんの
    /// 全トークンを先に処理してから次の層へ進む。状態を持つ実装 (full attention の
    /// KV / pos、ssm の conv・GDN) が「1 トークンで全層を 1 周する」前提で位置を
    /// 進めている場合、この順序では壊れる。
    ///
    /// 既定は `false` (安全側 = 逐次経路に落ちる)。層優先で正しく動く実装だけが
    /// `true` を返す。full attention の `pos` は「全 full attention 層が揃うまで
    /// 進めない」実装なので、現状は [`IdentityAttn`] だけが `true`。
    fn layer_major_safe(&self) -> bool {
        false
    }

    /// P109: n-gram 投機的デコードの巻き戻し ([`Self::spec_restore`]) に対応するか。
    ///
    /// 既定は `false` (安全側 = spec の batch 経路は逐次に落ちる)。状態の持ち方が
    /// 「長さを戻せば復元できる」実装 (full attention の KV など) だけが `true`。
    fn spec_restorable(&self) -> bool {
        false
    }

    /// P109: 現在の状態を保存する ([`Self::spec_restore`] と対で使う)。
    ///
    /// 戻り値は実装固有の型を `Box<dyn Any + Send>` に入れたもの
    /// (FullAttn は [`SpecFullSnap`]、SsmAttn は [`SpecGdnSnap`])。restore は
    /// スナップショットを作った同じオブジェクトにしか呼ばれないので
    /// downcast で受ける (違う型を渡したら panic = 呼び出し側のバグ)。
    fn spec_snapshot(&self) -> Box<dyn std::any::Any + Send> {
        Box::new(())
    }

    /// P118: 系列状態を全て消す (新しい系列の先頭に戻す)。
    ///
    /// **必須メソッド** (既定実装を設けない)。状態を持つ実装がうっかり
    /// 上書き忘れても黙って通らないようにするため。実装は次のどれか:
    /// - 状態を持たない ([`IdentityAttn`] など): 何もしない (明示的に書く)
    /// - [`FullAttn`]: KV・pos・QSA の indexer 状態を消す
    /// - [`SsmAttn`]: ssm 層の GDN conv/ssm を 0 に戻し、full 側へも委譲する
    ///
    /// 重み (uid 登録込み) は触らない — GPU 常駐の重みが再 upload されないのが
    /// このメソッドの存在理由 (P118: serve の no-reuse 経路でモデルを使い回す)。
    /// reset 直後の forward は新規構築直後とビット一致しなければならない
    /// (テスト `reset_matches_fresh_decoder_bitexact`)。
    fn reset(&mut self);

    /// task 553: 後ろへ回した状態の書き出しを今すぐ済ませる。
    ///
    /// 層優先 prefill の終端で 1 回だけ呼ぶ。`DENNO_PF_KV_DEFER` は full 層の
    /// KV 書き戻しを次の呼び出しへ回すので、リクエストの切れ目で持ち越さない
    /// ようにする。既定は何もしない (状態を後ろへ回す実装だけが上書きする)。
    fn flush_deferred(&mut self) -> std::result::Result<(), Qwen4ExpError> {
        Ok(())
    }

    /// P109: [`Self::spec_snapshot`] の状態へ戻す (検証列を棄却したとき)。
    fn spec_restore(
        &mut self,
        _snap: &dyn std::any::Any,
    ) -> std::result::Result<(), Qwen4ExpError> {
        Ok(())
    }

    /// task 206: 系列状態を **move で** 差し替えられる実装か。
    ///
    /// serve は 1 つのモデル (重み) を共有したまま、文脈ごとの系列状態
    /// (KV / GDN conv+ssm / PLE conv / pos) だけを入れ替えて持ち回る。
    /// 状態を持たない実装と、QSA 有効の [`FullAttn`] (indexer の生キーが
    /// 文脈長ぶんあり、move の口が無い) は `false` を返す。
    fn seq_state_supported(&self) -> bool {
        false
    }

    /// task 206: 系列状態を丸ごと取り出し、この実装を「新しい系列の先頭」に戻す。
    ///
    /// 取り出した状態は [`Self::seq_state_put`] で戻すまで保持してよい
    /// (別の状態を載せて別の文脈を回しても、取り出した状態は壊れない)。
    /// 取り出しは **コピーしない** (KV は Vec ごと移す)。対応しない実装は
    /// `None` (状態は変わらない)。
    fn seq_state_take(&mut self) -> Option<SeqState> {
        None
    }

    /// task 206: [`Self::seq_state_take`] で取り出した状態へ戻す。
    /// 取り出したあとの空の状態は捨てる。
    fn seq_state_put(&mut self, _st: SeqState) -> std::result::Result<(), Qwen4ExpError> {
        Err(one(
            "seq_state_put: この実装は系列状態の差し替えに対応していない",
        ))
    }
}

/// task 206: 1 つの文脈 (会話) ぶんの系列状態を運ぶ入れ物。
///
/// 中身は実装固有 ([`crate::model::qwen4exp_attn::FullSeqState`] /
/// [`crate::model::qwen4exp_ssm::SsmSeqState`]) で、作ったモデルにしか戻せない。
/// `bytes` は取り出した時点の RAM 使用量 (KV の容量ベース。確保済みの分を含む)。
pub struct SeqState {
    inner: Box<dyn std::any::Any + Send>,
    bytes: usize,
}

impl SeqState {
    pub(crate) fn new(inner: Box<dyn std::any::Any + Send>, bytes: usize) -> Self {
        Self { inner, bytes }
    }

    /// この状態が持っている RAM バイト数 (KV の Vec 容量 + 付随データ)。
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// 中身を取り出す (`seq_state_put` の実装が downcast する)。
    pub(crate) fn into_inner(self) -> Box<dyn std::any::Any + Send> {
        self.inner
    }
}

/// task 206: 1 つの文脈ぶんの **モデル全体** の系列状態。
///
/// PLE の conv 履歴 (`Qwen4Exp::ple_conv_state`) と attention 側の状態
/// ([`SeqState`]) をまとめたもの。重み・expert キャッシュ・PLE の表は含まない。
pub struct ModelSeqState {
    ple: Vec<Option<Vec<f32>>>,
    attn: SeqState,
}

impl ModelSeqState {
    /// この状態が持っている RAM バイト数。
    pub fn bytes(&self) -> usize {
        let ple: usize = self
            .ple
            .iter()
            .flatten()
            .map(|b| b.capacity() * std::mem::size_of::<f32>())
            .sum();
        self.attn.bytes() + ple
    }
}

/// P109: [`crate::model::qwen4exp_attn::FullAttn`] の spec スナップショット。
///
/// KV は push でしか伸びないので、復元は「層ごとの長さを戻す」だけで足りる
/// ([`crate::model::qwen4exp_attn::AttnCache::truncate_layer`]。領域は残る)。
/// QSA 有効時は巻き戻しに非対応 (`spec_restorable == false`) ので使われない。
pub struct SpecFullSnap {
    /// 全体位置 ([`crate::model::qwen4exp_attn::FullAttn::pos`])。
    pub pos: usize,
    /// 層ごとの KV 長 (full attention 層以外の層も含む全層分)。
    pub lens: Vec<usize>,
}

/// P109: [`crate::model::qwen4exp_ssm::SsmAttn`] の spec スナップショット。
///
/// GDN 状態 (conv / ssm) を層ごと clone して持ち、full attention 側は
/// 委譲先の実装固有スナップショットに任せる。`prefix_cache` の GDN
/// スナップショットと同じ (conv, ssm) 形だが依存を切るため独立の型。
pub struct SpecGdnSnap {
    /// 層ごとの GDN 状態 (full attention 層は `None`)。
    pub gdn: Vec<Option<(Vec<f32>, Vec<f32>)>>,
    /// 委譲先 (full attention 実装) のスナップショット。
    pub full: Box<dyn std::any::Any + Send>,
}

/// P109: spec decode ([`Qwen4ExpDecoder::spec_verify`]) の巻き戻し用スナップショット。
///
/// モデルの状態は (1) PLE の conv 履歴と (2) attention 実装の状態
/// (full attention の KV / ssm の GDN) の 2 箇所で進む。検証列を棄却したときは
/// この 2 つを揃えて戻さないと次のステップが静かに壊れる。
pub struct SpecSnapshot {
    ple: Vec<Option<Vec<f32>>>,
    attn: Box<dyn std::any::Any + Send>,
}

/// F1 の恒等実装。`cur` をそのまま返す (attention / ssm は F2 / F3)。
pub struct IdentityAttn;

impl LayerAttn for IdentityAttn {
    fn forward(&mut self, _layer: usize, cur: &[f32]) -> Result<Vec<f32>> {
        Ok(cur.to_vec())
    }

    /// 状態を持たない (恒等) ので、層優先の順序でも壊れない。
    fn layer_major_safe(&self) -> bool {
        true
    }

    /// 状態を持たない (恒等) ので、消すものも無い (明示的に書く)。
    fn reset(&mut self) {}
}

/// GGUF テンソルから [`GgufLinear`] を作る。**mmap のスライスを共有**し、
/// バイト列をコピーしない (`WeightBlocks::mapped`)。
///
/// `ne[0]` が入力長 (行長)、`ne[1..]` の積が行数。1 次元テンソル
/// (`ffn_gate_inp_shexp`) は行数 1 の線形層として扱う。
fn linear(f: &GgufFile, name: &str) -> Result<GgufLinear> {
    let t = f
        .tensor(name)
        .ok_or_else(|| one(format!("{name}: テンソルが無い")))?;
    let dims = trimmed_dims(t);
    let k = usize::try_from(dims[0])
        .map_err(|_| one(format!("{name}: ne[0] {} が usize に収まらない", dims[0])))?;
    let mut n = 1usize;
    for d in &dims[1..] {
        n = n
            .checked_mul(usize::try_from(*d).unwrap_or(usize::MAX))
            .ok_or_else(|| one(format!("{name}: 行数が usize を溢れる")))?;
    }
    let (file, off, len) = f
        .tensor_mapping_parts(name)
        .ok_or_else(|| one(format!("{name}: データ範囲がファイル終端を超過している")))?;
    let map = f
        .mmap_handle_at(file)
        .ok_or_else(|| one(format!("{name}: パート {file} の mmap が無い")))?;
    let blocks = WeightBlocks::mapped(map, off, len);
    let l = GgufLinear::new(t.ty, k, n, blocks).map_err(|e| one(format!("{name}: {e}")))?;
    crate::model::linear_gguf::maybe_requant(l)
        .map(|l| l.with_fast_q8(true))
        .map_err(|e| one(format!("{name}: {e}")))
}

/// f32 の Vec として読む小さなテンソル (norm の gamma・conv の重み用)。
///
/// 2026-10-31: F32 固定をやめた。Swift 1.5 は `ple_conv1d` が F16、`ssm_alpha` などが
/// BF16 で入っており、どれも f32 へ展開して同じに扱える。F32 以外は `dequant_tensor`
/// の経路 (要素数一致を確認したうえで 1 要素ずつ変換) を通す。
fn f32_tensor(f: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| one(format!("{name}: テンソルが無い")))?;
    if !crate::kquant::dequant_supported(t.ty) {
        return Err(one(format!(
            "{name}: 型 {} はデコードできない",
            ty_name(t.ty)
        )));
    }
    f.dequant_tensor(name)
        .map_err(|e| one(format!("{name}: {e}")))
}

/// expert テンソル (`ne = [n_in, n_out, n_expert]`) の生ブロック。
///
/// 1 トークンで使うのは top-k 個だけなので、行列全体を f32 に展開しない
/// (512 expert × 640 × 2560 を f32 にすると 3.3 GB/本になる)。
struct ExpertW {
    name: String,
    blocks: WeightBlocks,
    ty: GgmlType,
    n_in: usize,
    n_out: usize,
    n_expert: usize,
    /// テンソルが属する GGUF のパート番号 (split GGUF の読み口の選択に使う)
    file: usize,
    /// テンソルのデータ先頭の**ファイル内絶対オフセット**
    /// ([`crate::gguf::GgufFile::tensor_mapping_parts`] の 2 番目)。
    /// `blocks` の先頭と同じ位置 (mmap はこのオフセットから写像している)。
    abs: u64,
}

impl ExpertW {
    fn load(f: &GgufFile, name: &str, p: &Qwen4ExpParams) -> Result<Self> {
        let t = f
            .tensor(name)
            .ok_or_else(|| one(format!("{name}: テンソルが無い")))?;
        let dims = trimmed_dims(t);
        if dims.len() != 3 {
            return Err(one(format!("{name}: 次元が 3 でない (ne={dims:?})")));
        }
        let (file, off, len) = f
            .tensor_mapping_parts(name)
            .ok_or_else(|| one(format!("{name}: データ範囲がファイル終端を超過している")))?;
        let map = f
            .mmap_handle_at(file)
            .ok_or_else(|| one(format!("{name}: パート {file} の mmap が無い")))?;
        let w = ExpertW {
            name: name.to_string(),
            blocks: WeightBlocks::mapped(map, off, len),
            ty: t.ty,
            n_in: dims[0] as usize,
            n_out: dims[1] as usize,
            n_expert: dims[2] as usize,
            file,
            abs: off as u64,
        };
        // 期待値との突き合わせ (validate 済みだが、ここでも次元の取り違えを落とす)。
        if w.n_expert != p.n_expert {
            return Err(one(format!(
                "{name}: expert 数 {} が期待 {} と違う",
                w.n_expert, p.n_expert
            )));
        }
        // task 67: expert テンソルの mmap に `MADV_HUGEPAGE` を付ける (既定 off)。
        // 失敗しても読むバイト列は変わらないので、警告 1 行を出して続行する。
        if expert_hugepage_enabled() {
            #[cfg(test)]
            EXPERT_HUGEPAGE_ADVISES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let ok = crate::model::moe_expert::advise_hugepage_range(w.blocks.as_slice());
            if !ok {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    eprintln!(
                        "[flash] DENNO_EXPERT_HUGEPAGE=1: madvise(MADV_HUGEPAGE) が失敗 \
                         (2 MB ページは使いません。動作は変わりません)"
                    )
                });
            }
        }
        Ok(w)
    }

    /// task 227: テンソル全体の mmap (診断のプローブが連続した転送元に使う)。
    fn full(&self) -> &[u8] {
        self.blocks.as_slice()
    }

    /// expert `e` の 1 本ぶんのバイト範囲。
    fn slice(&self, e: usize) -> Result<&[u8]> {
        let s = ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, e)
            .map_err(|err| one(format!("{}: {err}", self.name)))?;
        let b = self.blocks.as_slice();
        if s.end() > b.len() {
            return Err(one(format!(
                "{}: expert {e} の範囲 {}..{} がテンソル長 {} を超える",
                self.name,
                s.off,
                s.end(),
                b.len()
            )));
        }
        Ok(&b[s.off..s.end()])
    }

    /// expert `e` のバイト範囲に先読みを投げる (mmap のページフォルトを先に起こす)。
    ///
    /// task 67: `DENNO_EXPERT_PREFETCH=1` のときだけ投げる (**既定 off**。
    /// 実測で on が 6-8 % 遅かった。`REPORT_task67.md`)。
    fn prefetch(&self, e: usize) {
        if !expert_prefetch_enabled() {
            return;
        }
        #[cfg(test)]
        EXPERT_PREFETCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(s) = ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, e) {
            let b = self.blocks.as_slice();
            if s.end() <= b.len() {
                crate::model::moe_expert::prefetch_range(&b[s.off..s.end()]);
            }
        }
    }

    /// task 214: expert `e` のバイト範囲を **mmap のハンドル付き**で返す。
    ///
    /// 背景スレッド ([`crate::model::exp_up_worker`]) へ渡すための形。
    /// `&[u8]` はスレッドを跨げないので、`Arc<Mmap>` + 絶対オフセット + 長さにする。
    /// `WeightBlocks` がヒープ実体 (テスト経路) のときは `None` を返し、
    /// 呼び出し側は従来の同期経路を使う。
    ///
    /// 範囲は [`ExpertW::slice`] と同じ [`ExpertSlice`] から作る (ずれる余地を作らない)。
    fn slice_mapped(
        &self,
        e: usize,
    ) -> Result<Option<crate::model::exp_up_worker::Part>> {
        let Some((map, base, _)) = self.blocks.mapping() else {
            return Ok(None);
        };
        let s = ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, e)
            .map_err(|err| one(format!("{}: {err}", self.name)))?;
        if s.end() > self.blocks.as_slice().len() {
            return Err(one(format!(
                "{}: expert {e} の範囲 {}..{} がテンソル長 {} を超える",
                self.name,
                s.off,
                s.end(),
                self.blocks.as_slice().len()
            )));
        }
        Ok(Some(crate::model::exp_up_worker::Part {
            map: std::sync::Arc::clone(map),
            off: base + s.off,
            len: s.end() - s.off,
        }))
    }

    /// task 862: テンソル先頭の写像と、写像内でのテンソル先頭オフセット。
    ///
    /// ヒープ実体 (テスト経路) では `None`。page hint はこの 2 つと
    /// [`ExpertW::expert_stride`] から expert の絶対範囲を出す。
    fn map_base(&self) -> Option<(std::sync::Arc<memmap2::Mmap>, usize)> {
        self.blocks
            .mapping()
            .map(|(map, base, _)| (std::sync::Arc::clone(map), base))
    }

    /// task 862: 1 expert あたりのバイト数 (テンソル内の stride)。
    fn expert_stride(&self) -> Result<usize> {
        ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, 0)
            .map(|s| s.len)
            .map_err(|err| one(format!("{}: {err}", self.name)))
    }

    /// expert `e` の**ファイル内絶対オフセット**とバイト数 (キャッシュ経路用)。
    ///
    /// mmap 経路の [`ExpertW::slice`] と同じ [`ExpertSlice`] を使うので、
    /// 2 経路が指す範囲は必ず一致する (ずれる余地を作らない)。
    fn slice_abs(&self, e: usize) -> Result<(u64, usize)> {
        let s = ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, e)
            .map_err(|err| one(format!("{}: {err}", self.name)))?;
        let b = self.blocks.as_slice();
        if s.end() > b.len() {
            return Err(one(format!(
                "{}: expert {e} の範囲 {}..{} がテンソル長 {} を超える",
                self.name,
                s.off,
                s.end(),
                b.len()
            )));
        }
        Ok((self.abs + s.off as u64, s.len))
    }

    /// `out[r] = dot(row_r, x)` を 1 expert ぶん計算する。
    ///
    /// Flash-Next の expert は型が混在している (`ffn_gate_exps`/`ffn_up_exps` は
    /// ほぼ Q4_K、`ffn_down_exps` はほぼ Q5_1、一部 Q8_0)。振り分けは
    /// [`moe_expert::expert_gemv_any`] に集約してある (キャッシュ経路と同じ関数)。
    fn gemv(&self, e: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        let s = self.slice(e)?;
        let (n_in, n_out) = (self.n_in, self.n_out);
        // decode 用の VNNI 経路 (Q4_K / Q5_1 / Q8_0、`DENNO_GEMV_VNNI=1` のときだけ)。
        // 使えないときは false が返るので従来の入口へ落とす (既定は必ずこちら)。
        if moe_expert::expert_gemv_any_vnni(self.ty, s, n_in, n_out, x, out)
            .map_err(|err| one(format!("{}: expert {e}: {err}", self.name)))?
        {
            return Ok(());
        }
        moe_expert::expert_gemv_any(self.ty, s, n_in, n_out, x, out)
            .map_err(|err| one(format!("{}: expert {e}: {err}", self.name)))
    }

    /// [`ExpertW::gemv`] の**行並列を使わない**版 (P-MOE-G5)。
    ///
    /// 呼ぶ側 (MoE の expert ループ) が rayon で並列化されているときはこちらを使う。
    /// 内側でさらに行を分けるとスレッドが二重に立つため。行は独立なので、
    /// 分割の有無で各要素の計算順序は変わらず、[`ExpertW::gemv`] とビット一致する。
    fn gemv_seq(&self, e: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        let s = self.slice(e)?;
        let (n_in, n_out) = (self.n_in, self.n_out);
        // [`ExpertW::gemv`] と同じゲート (行並列なしの版)。
        if moe_expert::expert_gemv_any_vnni_seq(self.ty, s, n_in, n_out, x, out)
            .map_err(|err| one(format!("{}: expert {e}: {err}", self.name)))?
        {
            return Ok(());
        }
        moe_expert::expert_gemv_any_seq(self.ty, s, n_in, n_out, x, out)
            .map_err(|err| one(format!("{}: expert {e}: {err}", self.name)))
    }

    /// [`ExpertW::gemv_seq`] の**行範囲**版 (P70)。
    ///
    /// 1 expert のうち行 `[r0, r1)` だけを計算する。行は独立なので、チャンクに
    /// 切っても各要素の積和順は変わらず [`ExpertW::gemv_seq`] と**ビット一致**する。
    /// decode の expert 寄与を expert 横断で平坦化するときに使う
    /// (`DENNO_EXPERT_FLAT_DECODE`)。VNNI のゲートは [`ExpertW::gemv_seq`] と同じ。
    #[allow(clippy::too_many_arguments)]
    fn gemv_rows_seq(
        &self,
        e: usize,
        r0: usize,
        r1: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let s = self.slice(e)?;
        let (n_in, n_out) = (self.n_in, self.n_out);
        if moe_expert::expert_gemv_any_vnni_rows(self.ty, s, n_in, n_out, r0, r1, x, out)
            .map_err(|err| one(format!("{}: expert {e}: {err}", self.name)))?
        {
            return Ok(());
        }
        moe_expert::expert_gemv_any_rows(self.ty, s, n_in, n_out, r0, r1, x, out)
            .map_err(|err| one(format!("{}: expert {e}: {err}", self.name)))
    }
}

/// PLE 層 1 つぶんの重み。
struct PleW {
    /// `ple_key` — in `n_embd` → out `hc_dim`
    key: GgufLinear,
    /// `ple_value` — in `n_embd` → out `n_embd`
    value: GgufLinear,
    /// `ple_norm_key` — `[hc_dim]`
    norm_key: Vec<f32>,
    /// `ple_norm_query` — `[hc_dim]`
    norm_query: Vec<f32>,
    /// `ple_norm_conv` — `[hc_dim]`
    norm_conv: Vec<f32>,
    /// `ple_conv1d` — GGUF の ne は `[kern, hc_dim]` なので `w[c * kern + k]`
    conv1d: Vec<f32>,
    /// conv のカーネル長 (GGUF の `ple.conv_kernel`)
    kern: usize,
    /// `conv1d` を **kern 優先**に並べ直したもの (`conv1d_t[k*hc_dim + c]`)。
    ///
    /// task 195: `ple_conv_silu` の内側のループは `w[c*kern + k]` を読むので
    /// c 方向に stride kern の間隔が空き、ベクトル化できない (実測 27 us/トークン、
    /// 14k で 384 ms = PLE 注入の 42%)。kern 優先にすると k ごとの内側ループが
    /// 連続になり、同じ足し算の順序 (k 昇順・c 昇順) のまま速くなる。
    conv1d_t: Vec<f32>,
}

impl PleW {
    fn load(f: &GgufFile, il: usize, p: &Qwen4ExpParams) -> Result<Self> {
        let name = |s: &str| format!("blk.{il}.{s}");
        // task 195: conv1d は [hc_dim][kern] なので kern 優先にも並べておく
        // (`ple_conv_silu_t` が使う。値は同じで、読む順だけが連続になる)。
        let conv1d = f32_tensor(f, &name("ple_conv1d.weight"))?;
        let hc_dim = p.hc_dim();
        let kern = p.ple_conv_kernel;
        let mut conv1d_t = vec![0.0f32; conv1d.len()];
        if kern > 0 && conv1d.len() == kern * hc_dim {
            for c in 0..hc_dim {
                for k in 0..kern {
                    conv1d_t[k * hc_dim + c] = conv1d[c * kern + k];
                }
            }
        }
        Ok(PleW {
            key: linear(f, &name("ple_key.weight"))?,
            value: linear(f, &name("ple_value.weight"))?,
            norm_key: f32_tensor(f, &name("ple_norm_key.weight"))?,
            norm_query: f32_tensor(f, &name("ple_norm_query.weight"))?,
            norm_conv: f32_tensor(f, &name("ple_norm_conv.weight"))?,
            conv1d: conv1d.clone(),
            conv1d_t,
            kern,
        })
    }
}

/// 層 1 つぶんの重みハンドル。
struct LayerW {
    il: usize,
    /// `hc_attn_norm` — `[hc_dim]` (変換器が `1 + w` に畳んだ gamma)
    hc_attn_norm: Vec<f32>,
    /// `hc_attn_down` — in `hc_dim` → out `hc_lr`
    hc_attn_down: GgufLinear,
    /// `hc_attn_up` — in `hc_lr` → out `hc_dim`
    hc_attn_up: GgufLinear,
    /// `hc_attn_inject` — in `hc_dim` → out `n_hc`
    hc_attn_inject: GgufLinear,
    hc_ffn_norm: Vec<f32>,
    hc_ffn_down: GgufLinear,
    hc_ffn_up: GgufLinear,
    hc_ffn_inject: GgufLinear,
    /// `ffn_gate_inp` — in `n_embd` → out `n_expert`
    gate_inp: GgufLinear,
    /// `ffn_gate_inp_shexp` — in `n_embd` → out 1
    gate_inp_shexp: GgufLinear,
    gate_exps: ExpertW,
    up_exps: ExpertW,
    down_exps: ExpertW,
    /// `ffn_gate_shexp` — in `n_embd` → out `n_ff_shexp`
    gate_shexp: GgufLinear,
    /// `ffn_up_shexp` — in `n_embd` → out `n_ff_shexp`
    up_shexp: GgufLinear,
    /// `ffn_down_shexp` — in `n_ff_shexp` → out `n_embd`
    down_shexp: GgufLinear,
    /// PLE 層のみ `Some`
    ple: Option<PleW>,
}

impl LayerW {
    fn load(f: &GgufFile, p: &Qwen4ExpParams, il: usize) -> Result<Self> {
        let name = |s: &str| format!("blk.{il}.{s}");
        let ple = if p.is_ple(il) {
            Some(PleW::load(f, il, p)?)
        } else {
            None
        };
        Ok(LayerW {
            il,
            hc_attn_norm: f32_tensor(f, &name("hc_attn_norm.weight"))?,
            hc_attn_down: linear(f, &name("hc_attn_down.weight"))?,
            hc_attn_up: linear(f, &name("hc_attn_up.weight"))?,
            hc_attn_inject: linear(f, &name("hc_attn_inject.weight"))?,
            hc_ffn_norm: f32_tensor(f, &name("hc_ffn_norm.weight"))?,
            hc_ffn_down: linear(f, &name("hc_ffn_down.weight"))?,
            hc_ffn_up: linear(f, &name("hc_ffn_up.weight"))?,
            hc_ffn_inject: linear(f, &name("hc_ffn_inject.weight"))?,
            gate_inp: linear(f, &name("ffn_gate_inp.weight"))?.with_router_gpu(true),
            gate_inp_shexp: linear(f, &name("ffn_gate_inp_shexp.weight"))?,
            gate_exps: ExpertW::load(f, &name("ffn_gate_exps.weight"), p)?,
            up_exps: ExpertW::load(f, &name("ffn_up_exps.weight"), p)?,
            down_exps: ExpertW::load(f, &name("ffn_down_exps.weight"), p)?,
            gate_shexp: linear(f, &name("ffn_gate_shexp.weight"))?,
            up_shexp: linear(f, &name("ffn_up_shexp.weight"))?,
            down_shexp: linear(f, &name("ffn_down_shexp.weight"))?,
            ple,
        })
    }
}

/// 層 `l` の行列 `m` の [`ExpertW`]。
fn matrix_w(l: &LayerW, m: ExpertMatrix) -> &ExpertW {
    match m {
        ExpertMatrix::Gate => &l.gate_exps,
        ExpertMatrix::Up => &l.up_exps,
        ExpertMatrix::Down => &l.down_exps,
    }
}

/// task 862: expert テンソルの写像 → 読み口を page hint に登録する。
///
/// `DENNO_EXPERT_PAGE_HINT` の経路は `posix_fadvise(2)` に fd が要るが、
/// `memmap2::Mmap` は fd を持たない。パートごとに 1 本だけ開いて覚える
/// (既定 off でも開くだけで、syscall は投げない)。
fn register_expert_maps(f: &GgufFile, layers: &[LayerW]) {
    for l in layers {
        for w in [&l.gate_exps, &l.up_exps, &l.down_exps] {
            let Some((map, _)) = w.map_base() else {
                continue;
            };
            let Some(path) = f.path_at(w.file) else {
                continue;
            };
            crate::model::exp_page_hint::register_map(&map, path);
        }
    }
}

/// task 862: [`crate::model::exp_page_hint::take_plan`] の計画を実行する。
///
/// 1. cold: `madvise(MADV_DONTNEED)` + `posix_fadvise(POSIX_FADV_DONTNEED)` で
///    page cache から落とす。
/// 2. warm: 常駐していない範囲だけ `madvise(MADV_WILLNEED)` で読み直す。
///
/// 連続する expert id は 1 回の syscall にまとめる (stride 一定なので範囲が連続する)。
/// 失敗しても数値は変わらない (統計の `failed` に数えるだけ)。
fn page_hint_apply(layers: &[LayerW]) {
    let plan = crate::model::exp_page_hint::take_plan();
    if plan.is_empty() {
        return;
    }
    let t0 = std::time::Instant::now();
    for (layer, ids) in &plan.cold {
        page_hint_runs(layers, *layer, ids, true);
    }
    if crate::model::exp_page_hint::willneed_on() {
        for (layer, ids) in &plan.warm {
            page_hint_runs(layers, *layer, ids, false);
        }
    }
    crate::model::exp_page_hint::add_flush_ms(t0.elapsed().as_secs_f64() * 1e3);
}

/// task 862: 層 `layer` の `ids` (昇順) を連続する塊ごとに 1 回の syscall にする。
///
/// `drop` が真ならページを落とし、偽なら `MADV_WILLNEED` を投げる。
/// gate / up / down を別々に回す (テンソル境界をまたがない)。
fn page_hint_runs(layers: &[LayerW], layer: usize, ids: &[u16], drop: bool) {
    use crate::model::exp_page_hint::{add_drop_stats, add_warm_stats, drop_range};
    let Some(lw) = layers.get(layer) else {
        return;
    };
    for w in [&lw.gate_exps, &lw.up_exps, &lw.down_exps] {
        let Some((map, base)) = w.map_base() else {
            continue;
        };
        let Ok(stride) = w.expert_stride() else {
            continue;
        };
        let mut i = 0;
        while i < ids.len() {
            let first = ids[i] as usize;
            let mut j = i + 1;
            while j < ids.len() && ids[j] as usize == ids[j - 1] as usize + 1 {
                j += 1;
            }
            let n = j - i;
            let off = base + first * stride;
            let len = n.saturating_mul(stride);
            if !drop && crate::model::exp_page_hint::warm_budget_exhausted() {
                // 読み直しの上限 (`DENNO_EXPERT_PAGE_HINT_WARM_MB`) に達した。
                return;
            }
            if drop {
                let out = drop_range(&map, off, len);
                add_drop_stats(out, n as u64);
            } else {
                // 常駐していれば投げない (`mincore(2)`)。落ちている範囲だけ読む。
                let ptr = unsafe { map.as_ptr().add(off) } as usize;
                let out = match unsafe { crate::model::exp_pf::range_resident(ptr, len) } {
                    Some(true) => crate::model::exp_page_hint::DropOutcome::Skipped,
                    _ => {
                        let n = crate::model::exp_page_hint::willneed_range(&map, off, len);
                        if n == 0 {
                            crate::model::exp_page_hint::DropOutcome::Failed
                        } else {
                            crate::model::exp_page_hint::DropOutcome::Warmed(n)
                        }
                    }
                };
                add_warm_stats(out);
            }
            i = j;
        }
    }
}

/// `DENNO_EXPERT_CACHE_MB` (> 0) のとき expert キャッシュを組む (既定 0 = 無効)。
///
/// - スロット数 = `MB * 1024 * 1024 / slot_bytes` (切り捨て・最低 1) を行列ごとに。
/// - `slot_bytes` は**層をまたいだ最大**の expert バイト数を採る。UD 量子化では
///   層ごとに expert の型が変わる (`ffn_down_exps` が Q5_1 と Q8_0) ので、最小を
///   採ると `ExpertCache::get` が `len > slot_bytes` で落ちる。
/// - 読み口は GGUF のパート数だけ開く (split GGUF では層ごとに属するファイルが
///   変わる)。
fn build_expert_caches(f: &GgufFile, layers: &[LayerW]) -> Result<Option<ExpertCacheSet>> {
    build_expert_caches_with_mb(expert_cache::mb_from_env(), f, layers)
}

/// [`build_expert_caches`] の本体 (MB を引数で受けるのでテストできる)。
fn build_expert_caches_with_mb(
    mb: usize,
    f: &GgufFile,
    layers: &[LayerW],
) -> Result<Option<ExpertCacheSet>> {
    if mb == 0 {
        return Ok(None);
    }
    let mut specs = Vec::with_capacity(3);
    for m in ExpertMatrix::ALL {
        let mut slot_bytes = 0usize;
        for l in layers {
            let w = matrix_w(l, m);
            let s = ExpertSlice::new(w.ty, w.n_in, w.n_out, w.n_expert, 0)
                .map_err(|err| one(format!("{}: {err}", w.name)))?;
            slot_bytes = slot_bytes.max(s.len);
        }
        specs.push(CacheSpec { slot_bytes });
    }
    let [gate, up, down]: [CacheSpec; 3] = specs
        .try_into()
        .map_err(|_| one("expert キャッシュ: 行列の指定が 3 本でない"))?;
    let mut paths = Vec::with_capacity(f.part_count());
    for i in 0..f.part_count() {
        let p = f
            .path_at(i)
            .ok_or_else(|| one(format!("expert キャッシュ: パート {i} のパスが無い")))?;
        paths.push(p.to_path_buf());
    }
    let set = ExpertCacheSet::from_mb_capped(
        mb,
        expert_cache::cap_bytes_from_env(),
        [gate, up, down],
        &paths,
    )
    .map_err(|err| one(format!("expert キャッシュの構築に失敗: {err}")))?;
    // 総量上限 (`DENNO_EXPERT_CACHE_CAP_MB`) で全行列が 0 スロットになったら
    // 「無効」と同じ扱いにする (mmap 経路のまま。毎回 pread する経路へは落とさない)。
    if ExpertMatrix::ALL.iter().all(|&m| set.slots(m) == 0) {
        return Ok(None);
    }
    Ok(Some(set))
}

// ---------------------------------------------------------------------------
// task 260: ホスト RAM tier
// ---------------------------------------------------------------------------
//
// VRAM tier (P39) と mmap の間に「所有するホスト RAM」を置く。page cache は
// 誰の物でもなく大きな読みで追い出されるが、ここは追い出されない。
// 引く順番は VRAM tier → RAM tier → mmap。

/// 行列ごとの `RamSpec` と、モデルの expert 総バイト数 (RAM tier の上限)。
fn expert_ram_shape(layers: &[(ExpertW, ExpertW, ExpertW)]) -> Result<([RamSpec; 3], u64)> {
    let mut specs = Vec::with_capacity(3);
    let mut expert_bytes = 0u64;
    for (mi, _) in ExpertMatrix::ALL.iter().enumerate() {
        let mut slot_bytes = 0usize;
        let mut total = 0u64;
        let mut n_expert = 0usize;
        for (g, u, d) in layers {
            let w = match mi {
                0 => g,
                1 => u,
                _ => d,
            };
            let s = ExpertSlice::new(w.ty, w.n_in, w.n_out, w.n_expert, 0)
                .map_err(|err| one(format!("{}: {err}", w.name)))?;
            slot_bytes = slot_bytes.max(s.len);
            total = total.saturating_add(s.len as u64 * w.n_expert as u64);
            // **モデル全体**の本数 (層数 × 層あたり)。
            n_expert = n_expert.saturating_add(w.n_expert);
        }
        specs.push(RamSpec {
            slot_bytes,
            total_bytes: total,
            n_expert,
        });
        expert_bytes = expert_bytes.saturating_add(total);
    }
    let [gate, up, down]: [RamSpec; 3] = specs
        .try_into()
        .map_err(|_| one("expert RAM tier: 行列の指定が 3 本でない"))?;
    Ok(([gate, up, down], expert_bytes))
}

/// task 260/263: 起動時に RAM tier の下ごしらえだけする (**確保しない**)。
///
/// tier の作成と充填は背景の起動プラン ([`plan_experts_from_stats`]) が行う。
/// ここで `install(None)` してはいけない: プランは `build_model` より先に
/// 走るので、先に出来た tier を消してしまう (プランは起動の数秒を
/// 事前ロードに使うため、構築より前に起こしてある)。
///
/// 覚えるのは診断用の範囲だけ (`DENNO_EXPERT_RAM_PROBE` / `DROP`)。
fn expert_ram_prepare(f: &GgufFile, layers: &[LayerW]) -> Result<()> {
    use crate::model::expert_ram;

    if layers.is_empty() {
        return Ok(());
    }
    // task 325: `DENNO_PF_PREAD` の読み口。常に開いておく (4 本の open だけ)。
    // 有効化は prefill の写しの側 (`DENNO_PF_PREAD=1` + 開けたパート) が決める。
    let parts: Vec<Option<std::path::PathBuf>> = (0..f.part_count())
        .map(|i| f.path_at(i).map(|p| p.to_path_buf()))
        .collect();
    crate::model::exp_pf::install_parts(parts);
    // 診断用の範囲 (probe の mincore / page cache の drop)。tier の有無に依らず覚える。
    // task 514: `DENNO_PF_FADV_END` でも範囲が要る (終端の drop が使う)。
    if expert_ram::probe_enabled()
        || expert_ram::drop_enabled()
        || expert_ram::fadv_end_enabled()
    {
        let mut ranges: Vec<(usize, u64, u64)> = Vec::new();
        let mut mmap: Vec<(usize, usize)> = Vec::new();
        for l in layers {
            for m in ExpertMatrix::ALL {
                let w = matrix_w(l, m);
                if let Some((part, off, len)) = f.tensor_mapping_parts(&w.name) {
                    ranges.push((part, off as u64, len as u64));
                }
                let b = w.blocks.as_slice();
                mmap.push((b.as_ptr() as usize, b.len()));
            }
        }
        let parts: Vec<std::path::PathBuf> = (0..f.part_count())
            .filter_map(|i| f.path_at(i).map(|p| p.to_path_buf()))
            .collect();
        expert_ram::install_ranges(parts, ranges, mmap);
        // task 514: モデル全体 (dense / PLE を含む) の常駐量を測るための mmap。
        let mut pmap: Vec<(usize, usize)> = Vec::with_capacity(f.part_count());
        for i in 0..f.part_count() {
            if let Some(m) = f.mmap_handle_at(i) {
                pmap.push((m.as_ptr() as usize, m.len()));
            }
        }
        expert_ram::install_part_map(pmap);
    }
    Ok(())
}

/// task 260/263: RAM tier を作って install する (充填はしない)。
///
/// 予算は `DENNO_EXPERT_RAM_MB` (未設定 = 無効。`auto` = MemAvailable − 余裕を
/// expert 総量で頭打ち)。`DENNO_EXPERT_RAM_MB` が 0 / 未設定なら `None`。
fn expert_ram_build(
    f: &GgufFile,
    layers: &[(ExpertW, ExpertW, ExpertW)],
) -> Result<Option<std::sync::Arc<ExpertCacheSet>>> {
    use crate::model::expert_ram;

    if layers.is_empty() {
        return Ok(None);
    }
    if !expert_ram::enabled() {
        return Ok(None);
    }
    let (specs, expert_bytes) = expert_ram_shape(layers)?;
    let n_experts = layers
        .len()
        .saturating_mul(layers[0].0.n_expert)
        .max(1);
    let per_expert = expert_bytes / n_experts as u64;
    // task 268: モデル総バイト (全パートの長さ) から非 expert を出し、その分を
    // auto 予算で `MemAvailable` から引く (dense / PLE の page cache を残す)。
    let model_bytes: u64 = (0..f.part_count())
        .filter_map(|i| f.path_at(i))
        .filter_map(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .sum();
    let non_expert_bytes = model_bytes.saturating_sub(expert_bytes);
    let budget = expert_ram::budget_bytes_env_model(expert_bytes, non_expert_bytes);
    if budget < per_expert.max(1) {
        eprintln!(
            "[exp-ram] off: 予算 {} MiB が expert 1 本 ({:.2} MiB) に満たない (MemAvailable={:?} MiB / expert 総量 {} MiB)",
            budget >> 20,
            per_expert as f64 / (1024.0 * 1024.0),
            expert_ram::mem_available().map(|b| b >> 20),
            expert_bytes >> 20
        );
        return Ok(None);
    }
    let mut paths = Vec::with_capacity(f.part_count());
    for i in 0..f.part_count() {
        let p = f
            .path_at(i)
            .ok_or_else(|| one(format!("expert RAM tier: パート {i} のパスが無い")))?;
        paths.push(p.to_path_buf());
    }
    let set = ExpertCacheSet::new_ram(
        budget,
        specs,
        &paths,
        expert_ram::margin_bytes(),
        expert_ram::want_pinned(),
    )
    .map_err(|err| one(format!("expert RAM tier の構築に失敗: {err}")))?;
    let slots = set.slots(ExpertMatrix::Gate);
    if slots == 0 {
        eprintln!("[exp-ram] off: スロット 0");
        return Ok(None);
    }
    let set = std::sync::Arc::new(set);
    expert_ram::install(Some(std::sync::Arc::clone(&set)));
    eprintln!(
        "[exp-ram] tier: budget={}MiB slots={} pin={} margin={}MiB model={}MiB non_expert={}MiB",
        budget >> 20,
        slots,
        u8::from(expert_ram::want_pinned()),
        expert_ram::margin_bytes() >> 20,
        model_bytes >> 20,
        non_expert_bytes >> 20
    );
    Ok(Some(set))
}

/// 予算ぶんを充填する。戻り値は (バイト, 本数, 秒, 上限で止めたか)。
fn expert_ram_fill(
    set: &ExpertCacheSet,
    layers: &[(ExpertW, ExpertW, ExpertW)],
    order: &[(u16, u16)],
) -> (u64, usize, f64, bool) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    let cap = set.slots(ExpertMatrix::Gate);
    let t0 = Instant::now();
    let bytes = AtomicU64::new(0);
    let done = AtomicU64::new(0);
    let mut stopped = false;
    // 1 チャンクずつ「予算が残っているか」と「decode 中か」を見る。
    //
    // task 263: 512 から 64 へ下げた。lab t263c (UD-Q4_K_XL) で 512 本
    // (= 約 512 MB の pread) の塊がリクエスト中に 1 個走り、33 トークンの
    // prefill が 0.88 -> 1.13 s (+29%) になった。64 本 = 約 64 MB なら
    // 1 リクエストに重なっても ~30 ms で済む (推定)。
    //
    // task 268: 32 本に下げ、**decode 中でも止まらない**ようにした。
    // t268a の `pre` は 1 リクエスト 10 秒の間ずっと `busy` で待っていたので、
    // 5,428 スロットのうち 384 本しか埋まらず、tier が 1 度も効かなかった
    // (miss_ram=14 / miss_mmap=5,925)。待つ代わりに 1 チャンク読んで 2 ms
    // 譲る (干渉は 1 チャンク = 約 96 MB の pread に限られる)。
    const CHUNK: usize = 32;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(crate::model::expert_ram::threads().max(1))
        .thread_name(|i| format!("exp-ram-fill-{i}"))
        .build();
    for chunk in order.chunks(CHUNK) {
        if set.used_experts() >= cap {
            stopped = true;
            break;
        }
        if set.stopped() {
            stopped = true;
            break;
        }
        if crate::model::exp_stats::busy() {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let one = |&(layer, expert): &(u16, u16)| {
            let Some((g, u, d)) = layers.get(layer as usize) else {
                return;
            };
            let key = ExpertKey { layer, expert };
            let mut n = 0u64;
            let mut ok = true;
            for (mi, w) in [g, u, d].into_iter().enumerate() {
                let Ok((off, len)) = w.slice_abs(expert as usize) else {
                    return;
                };
                let m = ExpertMatrix::ALL[mi];
                match set.get(m, w.file, key, off, len) {
                    Ok(_) => n += len as u64,
                    // task 907: 予算が尽きたら 3 行列そろわない (tier には載らない)。
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
            bytes.fetch_add(n, Ordering::Relaxed);
            if ok {
                done.fetch_add(1, Ordering::Relaxed);
            }
        };
        match &pool {
            Ok(p) => {
                use rayon::prelude::*;
                // pinned の確保は context の張ったスレッドで行う (1 回だけ)。
                p.install(|| {
                    chunk.par_iter().for_each_init(
                        || {
                            #[cfg(feature = "cuda")]
                            {
                                if crate::model::expert_ram::want_pinned() {
                                    let _ = crate::cuda::make_current_on_this_thread();
                                }
                            }
                        },
                        |_, it| one(it),
                    )
                });
            }
            Err(_) => {
                for it in chunk {
                    one(it);
                }
            }
        }
    }
    (
        bytes.load(Ordering::Relaxed),
        done.load(Ordering::Relaxed) as usize,
        t0.elapsed().as_secs_f64(),
        stopped,
    )
}

/// task 260: RAM tier から expert 1 本 (3 行列) のバイト列を取る。
///
/// ミスしたらファイルから読んで**埋める** (`DENNO_EXPERT_RAM_MB` の予算内)。
/// tier が無効 / 予算外 / 失敗なら `None` を返し、呼び出し側は mmap へ落ちる。
fn ram_regions(
    lw: &LayerW,
    ei: usize,
    fill: bool,
) -> Option<[std::sync::Arc<crate::model::expert_cache::Region>; 3]> {
    let set = crate::model::expert_ram::get()?;
    let key = ExpertKey {
        layer: lw.il as u16,
        expert: ei as u16,
    };
    let mut out: Vec<std::sync::Arc<crate::model::expert_cache::Region>> = Vec::with_capacity(3);
    for m in ExpertMatrix::ALL {
        let w = matrix_w(lw, m);
        let Ok((off, len)) = w.slice_abs(ei) else {
            return None;
        };
        if fill {
            match set.get(m, w.file, key, off, len) {
                Ok(r) => out.push(r),
                Err(_) => return None,
            }
        } else {
            // `fill == false`: ヒットだけ。ミスで pread しない (decode の
            // クリティカルパスで 3 MB を読み直さないため。lab t260a)。
            match set.peek(m, key, len) {
                Some(r) => out.push(r),
                None => return None,
            }
        }
    }
    let [a, b, c]: [std::sync::Arc<crate::model::expert_cache::Region>; 3] = out.try_into().ok()?;
    Some([a, b, c])
}

/// task 260: RAM tier の**ヒットだけ**を引く (ミスで pread しない。prefill 用)。
fn ram_peek(lw: &LayerW, ei: usize) -> Option<[std::sync::Arc<crate::model::expert_cache::Region>; 3]> {
    ram_regions(lw, ei, false)
}

/// task 260: VRAM へ上げる expert 1 本のホストバイト列の出所。
///
/// RAM tier に載っていればその区間 (pinned なら bounce コピー無しで DMA)、
/// 無ければ mmap。tier の区間は `Arc` で保持するので、`acquire` の間ずっと
/// 有効 (LRU が追い出しても場所は再利用されない)。
struct HostExpert {
    regions: Option<[std::sync::Arc<crate::model::expert_cache::Region>; 3]>,
}

impl HostExpert {
    fn new(lw: &LayerW, ei: usize) -> Self {
        HostExpert {
            // `DENNO_EXPERT_RAM_ADAPT=1` のときだけ使用時に埋める (既定 off:
            // decode のミスで 3 MB を読み直すコストが勝つ。lab t260a)。
            regions: ram_regions(lw, ei, crate::model::expert_ram::adapt_enabled()),
        }
    }

    fn slices<'a>(&'a self, lw: &'a LayerW, ei: usize) -> Result<[&'a [u8]; 3]> {
        match &self.regions {
            Some(r) => Ok([r[0].as_slice(), r[1].as_slice(), r[2].as_slice()]),
            None => Ok([
                lw.gate_exps.slice(ei)?,
                lw.up_exps.slice(ei)?,
                lw.down_exps.slice(ei)?,
            ]),
        }
    }

    fn pinned(&self) -> bool {
        match &self.regions {
            Some(r) => r.iter().all(|x| x.is_pinned()),
            None => false,
        }
    }
}

/// task 260: VRAM へ 1 本上げる。転送元は **RAM tier → mmap** の順。
///
/// すでに VRAM に載っている expert では RAM tier を**触らない** (触ると
/// ミスのたびに 3 MB の pread が走る)。ミスのときだけ tier を引き、
/// tier にも無ければ pread で埋めてからその区間を DMA する (fill from use)。
#[cfg(feature = "cuda")]
fn acquire_expert(
    lw: &LayerW,
    ei: usize,
    key: ExpertKey,
    shapes: [crate::model::expert_vram::ExpertShape; 3],
) -> Result<Option<crate::model::expert_vram::Acquired>> {
    let hit = crate::model::expert_vram::lookup(key, shapes)
        .map_err(|e| one(format!("層 {}: expert {ei}: VRAM 参照: {e:#}", lw.il)))?;
    let ctx = |e: anyhow::Error| one(format!("層 {}: expert {ei}: VRAM 経路: {e:#}", lw.il));
    if hit.is_some() {
        return crate::model::expert_vram::acquire(key, shapes, || {
            Ok([
                lw.gate_exps.slice(ei)?,
                lw.up_exps.slice(ei)?,
                lw.down_exps.slice(ei)?,
            ])
        })
        .map_err(ctx);
    }
    let host = HostExpert::new(lw, ei);
    // task 268: 転送元を数える (RAM tier から / mmap から)。「tier が効いている」
    // ことを緑ではなく数字で示すための計器。
    crate::model::expert_ram::note_miss_source(host.regions.is_some());
    if host.regions.is_some() {
        let pinned = host.pinned();
        crate::model::expert_vram::acquire_host(key, shapes, || Ok(host.slices(lw, ei)?), pinned)
            .map_err(ctx)
    } else {
        crate::model::expert_vram::acquire(key, shapes, || {
            Ok([
                lw.gate_exps.slice(ei)?,
                lw.up_exps.slice(ei)?,
                lw.down_exps.slice(ei)?,
            ])
        })
        .map_err(ctx)
    }
}

/// 最終段の重み (参照実装では `output_norm` が無く、head の mixer がそれを担う)。
///
/// テンソル名は実ファイルに合わせて `output_hc_*` (タスク文の `hc_head_*` と
/// 同じもの。`load_arch_tensors` の `LLM_TENSOR_HC_HEAD_*` が `output_hc_*` に
/// 対応する)。
struct HeadW {
    /// `output_hc_norm` — `[hc_dim]`
    norm: Vec<f32>,
    /// `output_hc_down` — in `hc_dim` → out `hc_lr`
    down: GgufLinear,
    /// `output_hc_up` — in `hc_lr` → out `hc_dim`
    up: GgufLinear,
    /// `output` — in `n_embd` → out `n_vocab`
    output: GgufLinear,
}

impl HeadW {
    fn load(f: &GgufFile) -> Result<Self> {
        Ok(HeadW {
            norm: f32_tensor(f, "output_hc_norm.weight")?,
            down: linear(f, "output_hc_down.weight")?,
            up: linear(f, "output_hc_up.weight")?,
            output: linear(f, "output.weight")?,
        })
    }
}

/// PLE の表と索引 (層に依らない部分)。
pub struct PleRuntime {
    /// `per_layer_token_embd` の SSD 常駐 gather
    pub table: PleTable,
    /// n-gram 索引 (`set_input` と同じ規則)
    pub ngram: PleNgram,
}

/// task 254: チャンクぶんの PLE 行 gather の本体。
///
/// [`Qwen4Exp::gather_ple_many`] と先読みスレッドの両方がこれを呼ぶので、
/// 先読みの有無で値は変わらない (変わるのは pread を出す順序とスレッドだけ)。
fn gather_ple_many_rt(
    rt: Option<&PleRuntime>,
    ctx: &[u32],
    start: usize,
) -> Result<Vec<Option<Vec<f32>>>> {
    let nt = ctx.len().saturating_sub(start);
    let rt = match rt {
        Some(rt) => rt,
        None => return Ok(vec![None; nt]),
    };
    #[cfg(test)]
    PLE_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let n_gram = rt.ngram.n_gram();
    let mut wins: Vec<Vec<u32>> = Vec::with_capacity(nt);
    for i in start..ctx.len() {
        wins.push(ple_window(&ctx[..=i], n_gram));
    }
    let refs: Vec<&[u32]> = wins.iter().map(|w| w.as_slice()).collect();
    let flat = rt
        .ngram
        .gather_many(&rt.table, &refs)
        .map_err(|err| one(format!("PLE 索引: {err}")))?;
    let stride = rt
        .ngram
        .n_heads()
        .checked_mul(rt.table.n_dim())
        .ok_or_else(|| one("PLE 索引: n_heads * n_dim がオーバーフロー"))?;
    if flat.len() != nt * stride {
        return Err(one(format!(
            "PLE 索引: まとめ引きの長さ {} が nt {nt} * {stride} と違う",
            flat.len()
        )));
    }
    Ok(flat
        .chunks_exact(stride)
        .map(|c| Some(c.to_vec()))
        .collect())
}

/// task 254: 次チャンクの PLE 行 gather を 1 本のスレッドで先に走らせる
/// (`DENNO_PF_PLE_PRE`、既定 off)。
///
/// 14k cold prefill では PLE の行読みが層ループの前に丸ごと露出する
/// (1.36〜1.59 s、page cache が冷えているとき)。172k 行の 90 B ランダム pread は
/// デバイスの IOPS 律速なので短くはできないが、チャンクの層ループ (14k で
/// 約 2.5 s) の裏には隠せる。値は [`gather_ple_many_rt`] をそのまま呼ぶので
/// ビット一致し、失敗も join でそのまま返る。
#[cfg(any(feature = "cuda", test))]
struct PlePrefetch {
    h: Option<std::thread::JoinHandle<Result<Vec<Option<Vec<f32>>>>>>,
}

/// 先読み専用プールのスレッド数 (8 コア 16 スレッドの実機で、層ループ側の
/// 写しと発行を邪魔しない本数)。
#[cfg(any(feature = "cuda", test))]
const PLE_PRE_THREADS: usize = 6;

#[cfg(any(feature = "cuda", test))]
impl PlePrefetch {
    fn none() -> Self {
        PlePrefetch { h: None }
    }

    /// `start..end` のトークンの窓を先読みする。PLE を持たないモデル・
    /// 空の範囲・範囲外では何もしない ([`PlePrefetch::none`])。
    fn spawn(
        rt: Option<std::sync::Arc<PleRuntime>>,
        ctx: &[u32],
        start: usize,
        end: usize,
    ) -> Self {
        let rt = match rt {
            Some(rt) if end > start && end <= ctx.len() => rt,
            _ => return Self::none(),
        };
        let c: Vec<u32> = ctx[..end].to_vec();
        prekprof::note_ple_pre_spawn();
        Self {
            h: Some(std::thread::spawn(move || {
                // グローバルの rayon プールは層ループ側 (PLE 注入の par_iter) も
                // 使うので、先読みは専用のプールで読む。8 コア 16 スレッドのうち
                // 6 本だけ使い、写し (stage_copy) とカーネル発行の邪魔をしない。
                // プールを作れないときは逐次 (遅いが値は同じ)。
                match rayon::ThreadPoolBuilder::new()
                    .num_threads(PLE_PRE_THREADS)
                    .build()
                {
                    Ok(p) => p.install(|| gather_ple_many_rt(Some(&rt), &c, start)),
                    Err(_) => gather_ple_many_rt(Some(&rt), &c, start),
                }
            })),
        }
    }

    fn is_some(&self) -> bool {
        self.h.is_some()
    }

    /// 先読みの結果を受け取る。join でホストが止まった時間を `[pre-kprof]` の
    /// `pre wait=` に積む (0 に近いほど層ループの裏に隠れている)。
    fn join(mut self) -> Result<Vec<Option<Vec<f32>>>> {
        match self.h.take() {
            Some(h) => {
                let t = std::time::Instant::now();
                let r = h
                    .join()
                    .map_err(|_| one("PLE の先読みスレッドが panic した"))?;
                if let Ok(v) = &r {
                    prekprof::note_ple_pre_join(t.elapsed().as_nanos() as u64, v.len());
                }
                r
            }
            None => Ok(Vec::new()),
        }
    }
}

#[cfg(any(feature = "cuda", test))]
impl Drop for PlePrefetch {
    /// 途中で降りた (エラーで `?` を通した) ときにスレッドを残さない。
    /// 残すと**次のリクエストのカウンタ**に読みが混ざる。
    fn drop(&mut self) {
        if let Some(h) = self.h.take() {
            let _ = h.join();
        }
    }
}

/// 1 トークン decode の骨格。
pub struct Qwen4Exp {
    pub p: Qwen4ExpParams,
    layers: Vec<LayerW>,
    head: HeadW,
    tok_embd: GgufLinear,
    /// task 254: 先読みスレッドへ渡せるように `Arc` で持つ (gather は read-only)。
    ple: Option<std::sync::Arc<PleRuntime>>,
    /// PLE の conv 履歴 (層ごと)。`(kern-1) * ngram_size` トークン分の
    /// `grouped_rms_norm(gated)` を古い順に保持する。層に PLE が無ければ `None`。
    ple_conv_state: Vec<Option<Vec<f32>>>,
    pub attn: Box<dyn LayerAttn>,
    /// P-MOE-G1: `DENNO_EXPERT_CACHE_MB` > 0 のときだけ `Some`。
    /// `None` は従来どおり mmap + `prefetch()` の経路 (バイト列も挙動も変えない)。
    expert_cache: Option<ExpertCacheSet>,
    /// P58: 構造テレメトリ (ExpertResidency / TierStats) の間引き状態。
    arch_emit: ArchEmit,
}

/// 層の出口で観測フックに渡す情報 (P53)。
///
/// 数値経路には触れない (**読むだけ**)。フックを付けないとき
/// ([`Qwen4Exp::forward_token`] / [`Qwen4Exp::forward_prefill_with`]) は
/// 実装側で `None` を渡すので、層ループには `if let Some(..)` が 1 個増えるだけで
/// 走査も確保も起きない。
pub struct LayerOut<'a> {
    /// 層番号
    pub il: usize,
    /// この forward 呼び出し内のトークン位置 (バッチ prefill はチャンク内、
    /// 逐次経路は `ctx` の先頭から)
    pub t: usize,
    /// この forward 呼び出しで回したトークン本数
    pub nt: usize,
    /// 埋め込み次元 (残差ストリーム 1 本の長さ)
    pub n_embd: usize,
    /// 層の出力 (残差を足した後)。`[hc][n_embd]` の連結。
    pub res: &'a [f32],
    /// 役割別 RMS (P54)。観測していないときは `None`。
    ///
    /// `Some` のときだけ engine が層の中で RMS を取る (取らないときは
    /// 数値経路に触れないのと同じで、走査もしない)。
    pub roles: Option<&'a RoleRms>,
}

/// 1 層ぶんの役割別 RMS (P54)。
///
/// テレメトリ (3D ビューア) 用の**読み取り専用**の値。層の出力を変えない。
/// 単位はバラバラ (ベクトルの RMS と、router の確率和) なので、送る側で
/// 層内最大により 0..1 に正規化する ([`RoleRms::normalized`])。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RoleRms {
    /// full attention 層の attention 出力の RMS (ssm 層では 0)
    pub attn: f32,
    /// ssm (GDN) 層の scan 出力の RMS (full attention 層では 0)
    pub gdn: f32,
    /// router の top-k 確率和 (0..1)。スカラーなので RMS は値そのもの
    pub router: f32,
    /// shared expert の出力 (`ffn_down_shexp` の出力) の RMS
    pub shex: f32,
    /// PLE 注入の寄与 (`out - res` = gated + conv) の RMS
    pub ple: f32,
}

impl RoleRms {
    /// `[attn, gdn, router, shex, ple]` の生値。
    pub fn as_array(&self) -> [f32; 5] {
        [self.attn, self.gdn, self.router, self.shex, self.ple]
    }

    /// 層内最大で割った 0..1 の 5 値 (最大が 0 なら全部 0)。
    ///
    /// 正規化の規則は `denno_telemetry::normalize_roles` と同じ。engine から
    /// テレメトリへの依存を増やさないため、ここでは生値だけを持ち、正規化は
    /// 送る側 (CLI) で行う。
    pub fn normalized(&self) -> [f32; 5] {
        let a = self.as_array();
        let mut max = 0.0f32;
        for v in a {
            if v > max {
                max = v;
            }
        }
        if !(max > 0.0) {
            return [0.0; 5];
        }
        let mut out = [0.0f32; 5];
        for (o, v) in out.iter_mut().zip(a) {
            *o = (v / max).clamp(0.0, 1.0);
        }
        out
    }
}

impl<'a> LayerOut<'a> {
    /// 代表ストリーム (stream 0) の `n_embd` 個。P53 の層活性はこれを使う。
    ///
    /// hyper-connection は hc 本の残差ストリームを持つが、活性の走査コストを
    /// `n_layer * n_embd` に固定するため 1 本だけ見る (残差を足した後の値)。
    pub fn stream0(&self) -> &'a [f32] {
        let n = self.n_embd.min(self.res.len());
        &self.res[..n]
    }
}

/// どのトークンの層出口を観測するか (P53)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TapTokens {
    /// 全部 (decode は 1 トークン呼び出しなので実質ごとトークン)
    All,
    /// forward 呼び出しの**最後のトークンだけ** (prefill のチャンク末尾で 1 回)
    Last,
}

/// task 209: prefill の resident 経路を出るときに「mixed のホスト/デバイス対応」
/// の記録を消す (`DENNO_PF_MIXDEVIN` が古い記録を拾わないため)。
#[cfg(feature = "cuda")]
struct PfMixedMarkGuard;

#[cfg(feature = "cuda")]
impl Drop for PfMixedMarkGuard {
    fn drop(&mut self) {
        crate::cuda::pf_mixed_mark_clear();
    }
}

/// task 209: prefill の活性ホストバッファ。`cuda` feature では pinned を試す
/// ([`crate::cuda::PfHostBuf`])、無いときは普通の `Vec`。
#[cfg(feature = "cuda")]
use crate::cuda::PfHostBuf as PfHostBuf;

/// task 209: `cuda` feature 無しのホストバッファ (常に `Vec`)。
#[cfg(not(feature = "cuda"))]
struct PfHostBuf {
    v: Vec<f32>,
}

#[cfg(not(feature = "cuda"))]
impl PfHostBuf {
    fn new(len: usize) -> Self {
        Self {
            v: vec![0.0f32; len],
        }
    }
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.v
    }
}

impl Drop for PrefillSeqGuard {
    fn drop(&mut self) {
        IN_PREFILL.with(|c| c.set(self.0));
    }
}

/// task 484: inject の GEMV 1 発。4 行同時の F32 FMA 経路が使えればそこへ書き、
/// `true` を返す。使えなければ**何も書かずに** `false` (呼び出し側が従来経路)。
///
/// 対象は `hc_inject` の実形 (F32・hc 行 x hc_dim、hc <= 4・hc_dim は 8 の倍数) だけ。
/// 発火は [`MIX_INJ_R4_HITS`] で数える (`[mix-qchain]` の `inj4=`)。
fn inject_gemv_r4(w: &GgufLinear, x: &[f32], out: &mut [f32]) -> bool {
    if !mix_inj_r4_on() || w.ty() != crate::kquant::GgmlType::F32 {
        return false;
    }
    let ok = crate::model::moe_expert::expert_gemv_f32_rows_small(
        w.blocks().as_slice(),
        w.k(),
        w.n(),
        x,
        out,
    );
    if ok {
        MIX_INJ_R4_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 台帳 (`[mix-kprof]` の gpu=) が直前の GEMV のフラグを引き継がないように。
        #[cfg(feature = "cuda")]
        crate::cuda::set_last_linear_ran_on_gpu(false);
    }
    ok
}

/// GPU の `moe_gemv_dev` / `moe_gemm_dev` が読める 4 型か (dense 重み用)。
fn quant_gpu_ty_ok(ty: crate::kquant::GgmlType) -> bool {
    use crate::kquant::GgmlType as T;
    matches!(ty, T::Q8_0 | T::Q5_1 | T::Q4K | T::Q5K)
}

/// task 150: `DENNO_DECODE_CHAIN2` の値を解釈する (純関数・テスト用)。
/// **既定 on** (`DENNO_MIX_CHAIN` と同じ規則。明示的な偽値 `0` / `off` / `false` /
/// `no` だけ切る)。lab t150b で目標を満たしたため t150 の最終コミットで既定を
/// off から on へ反転した (採用条件は REPORT_P150.md)。
fn decode_chain2_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// task 150: decode の GPU 往復を減らす束ねゲート (`DENNO_DECODE_CHAIN2`、
/// lab t150b で採用・**既定 on**)。
///
/// 中の変更は部分ゲートで個別に切れる:
/// - [`decode_chain2_exp_enabled`]: 選択 expert の IQ 層一括 GEMV を CPU ミスと重ねる。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn decode_chain2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        decode_chain2_from_value(std::env::var("DENNO_DECODE_CHAIN2").ok().as_deref())
    })
}

/// task 187: `DENNO_DEC_SHARED_DEV` の値を解釈する (純関数・テスト用)。
///
/// **既定 on** (lab t187c: shared の wall 1.46 → 0.63 ms/tok、decode -8.0%)。
/// 明示的な偽値だけ切る。on のときは shared expert の gate → up → silu*up → down を
/// router の時点で 1 本のサイドストリームに積み、down の出力だけを join する。
/// silu を CUDA の `ex2.approx` で計算するので、ホストの `f32::exp` とは丸めが違う。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn dec_shared_dev_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static DEC_SHARED_DEV_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// [`dec_shared_dev_from_value`] をプロセスで 1 回だけ読む。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn dec_shared_dev_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match DEC_SHARED_DEV_OVERRIDE.load(Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        dec_shared_dev_from_value(std::env::var("DENNO_DEC_SHARED_DEV").ok().as_deref())
    })
}

/// テスト専用: 4 発経路の on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
fn set_dec_shared_dev_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    DEC_SHARED_DEV_OVERRIDE.store(
        match v {
            Some(false) => 1,
            Some(true) => 2,
            None => 0,
        },
        Ordering::Relaxed,
    );
}

/// task 187: `kernels_dec_epi.cu` の `dno_dec_exp_sum` と**同じ順序・同じ丸め**の
/// ホスト実装 (テストと仕様の固定用)。
///
/// `out[j] = sum_{t<nseg} w[t] * h[t*e+j]` を t 昇順の 1 アキュムレータで積み、
/// 最後に `out[j] += sgw * sh[j]` を足す。カーネルは `-fmad=false` なので積と和は
/// 別の丸め (ホストの `pe * hh[j]` + `+=` と同じ 2 演算)。
fn dec_exp_sum_mirror(
    h: &[f32],
    w: &[f32],
    e: usize,
    sh: Option<(&[f32], f32)>,
    out: &mut [f32],
) {
    debug_assert!(h.len() >= w.len() * e);
    for (j, o) in out.iter_mut().enumerate().take(e) {
        let mut acc = 0.0f32;
        for (t, &pe) in w.iter().enumerate() {
            acc += pe * h[t * e + j];
        }
        if let Some((shv, sgw)) = sh {
            acc += sgw * shv[j];
        }
        *o = acc;
    }
}

/// task 187: `DENNO_DEC_EXP_SUMDEV` の値を解釈する (純関数・テスト用)。
///
/// **既定 on** (lab t187c で採否 yes: decode -8.0%、prefill +0.4%、md5 SAME)。
/// 明示的な偽値 `0` / `off` / `false` / `no` だけ切る。on のときは、ヒットした
/// expert の寄与をデバイスで重み付き和にして `e` 要素だけ D2H する
/// (`kernels_dec_epi.cu` の `dno_dec_exp_sum`)。PTX が無い / 形が合わないときは
/// 従来経路へ落ちる (数値は変わらない)。
fn dec_exp_sumdev_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// [`dec_exp_sumdev_from_value`] をプロセスで 1 回だけ読む。
pub(crate) fn dec_exp_sumdev_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        dec_exp_sumdev_from_value(std::env::var("DENNO_DEC_EXP_SUMDEV").ok().as_deref())
    })
}

/// task 150: `DENNO_DECODE_CHAIN2_EXP` の値を解釈する (純関数・テスト用)。
/// **既定 on** で、明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
fn decode_chain2_exp_from_value(v: Option<&str>) -> bool {
    match v {
        Some(s) => !matches!(
            s.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        None => true,
    }
}

/// task 150: 束ねゲートと部分ゲートの合成 (純関数・テスト用)。
/// 部分ゲートは束ねゲートが on のときだけ効く。
fn chain2_exp_active(chain2: bool, exp_flag: bool) -> bool {
    chain2 && exp_flag
}

/// task 150: `DENNO_DECODE_CHAIN2_MIX` の値を解釈する (純関数・テスト用)。
/// 部分ゲート `DENNO_DECODE_CHAIN2_EXP` と同じ規則 (既定 on・明示的な偽値だけ切る)。
fn decode_chain2_mix_from_value(v: Option<&str>) -> bool {
    decode_chain2_exp_from_value(v)
}

/// task 150: hc mix の hc_mixed まで GPU で閉じる (`DENNO_DECODE_CHAIN2_MIX`、
/// 既定 on)。`DENNO_DECODE_CHAIN2` が on のときだけ意味を持つ。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn decode_chain2_mix_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    chain2_exp_active(
        decode_chain2_enabled(),
        *ON.get_or_init(|| {
            decode_chain2_mix_from_value(
                std::env::var("DENNO_DECODE_CHAIN2_MIX").ok().as_deref(),
            )
        }),
    )
}

/// task 150: 選択 expert の IQ 層一括 GEMV を CPU ミスと重ねる
/// (`DENNO_DECODE_CHAIN2_EXP`、既定 on)。`DENNO_DECODE_CHAIN2` が on のときだけ
/// 意味を持つ (束ねゲート off では従来どおり GEMV の完了を待ってから CPU を回す)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn decode_chain2_exp_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    chain2_exp_active(
        decode_chain2_enabled(),
        *ON.get_or_init(|| {
            decode_chain2_exp_from_value(
                std::env::var("DENNO_DECODE_CHAIN2_EXP").ok().as_deref(),
            )
        }),
    )
}

/// task 145: テスト専用: バッチ mix の GPU 経路を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_mix_batch_gpu_for_test(v: Option<bool>) {
    MIX_BATCH_GPU_OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// task 110: テスト専用: hc mix GPU 経路の on/off を強制する (`None` で env に戻す)。
/// 上書きはプロセス全体で 1 個なので、触るテストは直列化する (下の MIX_GPU_GUARD)。
#[cfg(test)]
pub(crate) fn set_mix_gpu_for_test(v: Option<bool>) {
    MIX_GPU_OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

impl MixScratch {
    /// `buf` を `n` 要素ぶん確保して `&mut [..n]` を返す (長さは呼び出し側が明示する)。
    fn take(buf: &mut Vec<f32>, n: usize) -> &mut [f32] {
        if buf.len() < n {
            buf.resize(n, 0.0);
        }
        &mut buf[..n]
    }
}

/// [`PLE_REUSE_ENV`] の既定値。
///
/// lab t300a (c18c505b7、depth8k、x0/base/pg/pr/pr0/base2) の実測:
///
/// | COND | c2000_cold r1 / r2 | c8000_cold (r1) |
/// | --- | --- | --- |
/// | base (両ゲート off) | 4.14 / 3.41 s | 10.09 s |
/// | pg (PIN_GROW のみ) | 3.22 / 3.23 s | 8.30 s |
/// | **pr (PIN_GROW + これ)** | **3.18 / 3.16 s** | **8.09 s** |
/// | pr0 (これのみ) | 3.35 / 3.36 s | 8.28 s |
/// | base2 | 3.40 / 3.41 s | 8.45 s |
///
/// 単独で -1.5%、PIN_GROW と合わせて -7.0% (3,826 cold)。14k も base2 比
/// -4.3%、decode は全点 ±2% 以内、md5 は全 COND SAME。
pub const DEFAULT_PLE_REUSE: bool = true;

/// [`PLE_FLAT_REUSE_ENV`] の既定 (**off**)。切り戻しは `=0`。
pub const DEFAULT_PLE_FLAT_REUSE: bool = false;

/// [`PLE_PIN_ENV`] の既定 (**lab で採ったので on**)。`=0` で切り戻す。
///
/// 実測 (lab t393a / t393b、depth8k、2 ラウンドずつ): 3,826 の `other` 段が
/// wall 409.3 / 238.9 (base) -> 226.2 / 226.1 ms (`other` の内側は
/// res_up 44.9-46.2 -> 32.1-33.0、ple 111.3 -> 101.3-101.9)。
/// 14k は `other` 764.6 -> 743.2 ms、`res_up` 206.0 -> 163.1 ms。
/// 壁は 3,826 で -0.6%、14k で -0.1%。decode と md5 は全 COND 不変。
pub const DEFAULT_PLE_PIN: bool = true;

#[cfg(not(feature = "cuda"))]
impl PlePinBuf {
    fn new(len: usize) -> Self {
        Self {
            v: vec![0.0f32; len],
        }
    }
    fn len(&self) -> usize {
        self.v.len()
    }
    fn as_slice(&self) -> &[f32] {
        &self.v
    }
    fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.v
    }
}

impl PleBack {
    fn as_slice(&self) -> &[f32] {
        match &self.pin {
            Some(b) => &b.as_slice()[..self.len],
            None => &self.v[..self.len],
        }
    }

    fn as_mut_slice(&mut self) -> &mut [f32] {
        match &mut self.pin {
            Some(b) => &mut b.as_mut_slice()[..self.len],
            None => &mut self.v[..self.len],
        }
    }
}

impl std::ops::Deref for PleBack {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.as_slice()
    }
}

impl std::ops::DerefMut for PleBack {
    fn deref_mut(&mut self) -> &mut [f32] {
        self.as_mut_slice()
    }
}

impl PleInjScratch {
    fn zeros(hc_dim: usize) -> Self {
        Self {
            key_n: vec![0.0; hc_dim],
            query_n: vec![0.0; hc_dim],
            gated: vec![0.0; hc_dim],
            nx: vec![0.0; hc_dim],
            conv: vec![0.0; hc_dim],
        }
    }

    /// スレッドローカルから取り出す (足りなければ伸ばす)。中身は毎回全部書く。
    fn take(hc_dim: usize) -> Self {
        let mut s = PLE_INJ_SCRATCH.with(|c| std::mem::take(&mut *c.borrow_mut()));
        for v in [
            &mut s.key_n,
            &mut s.query_n,
            &mut s.gated,
            &mut s.nx,
            &mut s.conv,
        ] {
            if v.len() < hc_dim {
                v.resize(hc_dim, 0.0);
            }
        }
        s
    }

    fn put(self) {
        PLE_INJ_SCRATCH.with(|c| *c.borrow_mut() = self);
    }
}

/// ベクトルの RMS (P54)。空は 0。**読むだけ** (数値経路に触れない)。
fn rms_of(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    let mut s = 0.0f32;
    for &v in x {
        s += v * v;
    }
    (s / x.len() as f32).sqrt()
}

/// P60: 初回送出用に**全層**ぶんの在庫を並べる (`n_layer` 層。在庫の無い層は空)。
///
/// 差分だけではビューワが初期状態を描けないので、最初の 1 回はこれを送る。
/// `now` に `n_layer` 以上の層が混ざっていた場合 (モデルの層数より多い在庫) は
/// 落とさずに後ろへ足す。
fn full_residency(now: &[(u16, Vec<u16>)], n_layer: usize) -> Vec<(u16, Vec<u16>)> {
    let mut out: Vec<(u16, Vec<u16>)> = Vec::with_capacity(n_layer.max(now.len()));
    for il in 0..n_layer {
        let layer = il as u16;
        let idx = now
            .iter()
            .find(|(l, _)| *l == layer)
            .map(|(_, v)| v.clone())
            .unwrap_or_default();
        out.push((layer, idx));
    }
    for (layer, idx) in now {
        if (*layer as usize) >= n_layer {
            out.push((*layer, idx.clone()));
        }
    }
    out
}

/// P58: [`Frame::ExpertResidency`] の間引き規則 (前回の送出との差分)。
///
/// - `now` に在って `prev` と集合が違う層 → `now` の集合を返す
/// - `prev` に在って `now` に無い層 (全部追い出された) → 空で返す
///   (送らないとビューアが古い在庫を描き続ける)
/// - 集合が同じ層は返さない (変化が無い層は沈黙する)
///
/// 並びは `now` の順、そのあと消えた層の昇順。`n` の上限は
/// [`denno_telemetry::RESIDENCY_MAX`] を [`expert_residency_frame`] が切る。
///
/// [`Frame::ExpertResidency`]: denno_telemetry::Frame::ExpertResidency
/// [`expert_residency_frame`]: denno_telemetry::expert_residency_frame
fn residency_delta(prev: &[(u16, Vec<u16>)], now: &[(u16, Vec<u16>)]) -> Vec<(u16, Vec<u16>)> {
    let mut out: Vec<(u16, Vec<u16>)> = Vec::new();
    for (layer, idx) in now {
        let same = prev.iter().any(|(l, p)| l == layer && p == idx);
        if !same {
            out.push((*layer, idx.clone()));
        }
    }
    for (layer, idx) in prev {
        if !idx.is_empty() && !now.iter().any(|(l, _)| l == layer) {
            out.push((*layer, Vec::new()));
        }
    }
    out
}

/// P58: 構造テレメトリの間引き状態 (モデル 1 個につき 1 個)。
#[derive(Default)]
struct ArchEmit {
    /// 前回送った層別の VRAM 常駐 expert ([`expert_vram::resident_by_layer`] の写し)。
    prev_res: Vec<(u16, Vec<u16>)>,
    /// P60: 初回の「全層」送出を済ませたか。
    ///
    /// `prev_res` が空かどうかで判定すると、在庫が 0 本のときに毎回「初回」に
    /// なってしまう (全層の空フレームを毎トークン送る)。フラグで 1 回だけにする。
    sent_initial: bool,
    /// TierStats の間引き用カウンタ (1 トークン呼び出しごとに 1 増える)。
    tokens: u32,
}

/// P58: decode で TierStats を送る間隔 (トークン数)。
///
/// 仕様は「1 秒に 1 回」だが、engine は時刻を持たないのでトークン数で粗く間引く
/// (27B の decode は 1 トークン数十 ms なので、100 トークンで数秒に 1 回)。
const TIER_STATS_TOKENS: u32 = 100;

impl Qwen4Exp {
    /// P58: decode 1 トークンぶんの構造テレメトリ。
    ///
    /// ExpertResidency は**変化した層だけ**、TierStats は
    /// [`TIER_STATS_TOKENS`] トークンごと。arch off なら即座に戻る (確保もしない)。
    pub(crate) fn arch_emit_token(&mut self) {
        if !arch_frames_enabled() {
            return;
        }
        self.arch_emit.tokens = self.arch_emit.tokens.wrapping_add(1);
        if self.arch_emit.tokens % TIER_STATS_TOKENS == 0 {
            self.emit_tier_stats();
        }
        self.emit_residency();
    }

    /// P58: prefill 終了時の構造テレメトリ (1 リクエスト 1 回)。
    pub(crate) fn arch_emit_prefill_done(&mut self) {
        if !arch_frames_enabled() {
            return;
        }
        self.emit_tier_stats();
        self.emit_residency();
    }

    /// P58/P60: いま VRAM に載っている expert を送る。
    ///
    /// - **初回は全層**を送る (在庫の無い層は空)。差分だけではビューワが初期状態を
    ///   描けない。初回の判定は [`ArchEmit::sent_initial`] で 1 回だけ。
    /// - 2 回目以降は「変化した層だけ」。消えた層は空フレームで送る
    ///   ([`residency_delta`])。
    ///
    /// 走査は [`expert_vram::resident_keys`] 1 回 (LRU の写し + staging の和集合)。
    /// 全 512 本 × 48 層を毎層並べることはしない。
    fn emit_residency(&mut self) {
        let now = expert_vram::resident_by_layer();
        if !self.arch_emit.sent_initial {
            let full = full_residency(&now, self.layers.len());
            for (layer, idx) in &full {
                denno_telemetry::emit(denno_telemetry::expert_residency_frame(*layer, idx));
            }
            self.arch_emit.prev_res = full;
            self.arch_emit.sent_initial = true;
            return;
        }
        if now == self.arch_emit.prev_res {
            return;
        }
        for (layer, idx) in residency_delta(&self.arch_emit.prev_res, &now) {
            denno_telemetry::emit(denno_telemetry::expert_residency_frame(layer, &idx));
        }
        self.arch_emit.prev_res = now;
    }

    /// P60: 置き場所の統計 (kind 15)。**実際の計測値だけ**を送る。
    ///
    /// - `vram_used_mb` / `vram_total_mb`: expert の VRAM 会計
    ///   ([`expert_vram::budget_usage`] = LRU 使用 + 外部予約 / 予算)。予算は
    ///   `DENNO_EXPERT_VRAM_MB` + `DENNO_GPU_WEIGHT_MB` の合計で、カードの全容量では
    ///   ない。どちらも未設定で在庫も無ければ 0 / 0。
    /// - `ram_experts`: RAM の pread キャッシュに載っている expert 本数
    ///   (`DENNO_EXPERT_CACHE_MB` 無効なら 0)。
    /// - `ssd_reads`: キャッシュのミス (= SSD からの pread) の累計。
    fn emit_tier_stats(&self) {
        let (used_mb, total_mb) = expert_vram::tier_mb(expert_vram::budget_usage());
        let ram_experts = self
            .expert_cache
            .as_ref()
            .map_or(0, |c| c.used_experts() as u32);
        let ssd_reads = self
            .expert_cache
            .as_ref()
            .map_or(0, |c| c.total_reads().min(u32::MAX as u64) as u32);
        denno_telemetry::emit(denno_telemetry::Frame::TierStats {
            vram_used_mb: used_mb,
            vram_total_mb: total_mb,
            ram_experts,
            ssd_reads,
        });
    }
}

/// task 830: リクエスト終端の P45 ゲートの 1 行 (ゲートが無ければ `None`)。
/// 迂回が効いている証拠は `bypass_gpu` が増えて `cpu` が 0 のままであること。
pub fn expert_gate_line() -> Option<String> {
    moe::expert_gate_line()
}

/// [`expert_gate_decide`] の本体。ゲートを引数で受けるので、env を触らずに
/// 「冷えたゲート → CPU」「温まったゲート → GPU」「見積り不能 → CPU」を通せる。
fn gate_decide(
    g: Option<&std::sync::Mutex<expert_vram::ExpertGate>>,
    layer: usize,
    n_sel: usize,
    per_expert: Option<u64>,
) -> expert_vram::GateDecision {
    let Some(g) = g else {
        return expert_vram::GateDecision::Gpu;
    };
    let h2d = per_expert.and_then(|b| b.checked_mul(n_sel as u64));
    let Ok(mut s) = g.lock() else {
        return expert_vram::GateDecision::Gpu;
    };
    let (rate, samples) = (s.hit_rate(), s.samples());
    let d = s.decide(layer, h2d);
    // 切り替えは 1 回ずつだけ言う (decode のたびに 48 層ぶん出さない)。
    static CPU_LOGGED: std::sync::Once = std::sync::Once::new();
    static PROBE_LOGGED: std::sync::Once = std::sync::Once::new();
    match d {
        expert_vram::GateDecision::Cpu => CPU_LOGGED.call_once(|| {
            let rate = match rate {
                Some(r) => format!("{r:.2}"),
                None => "n/a (未測定)".to_string(),
            };
            eprintln!(
                "[flash] expert GPU gate: 層 {layer} で CPU 経路 (直近 {samples} トークンのヒット率 {rate} \
                 < {:.2} / CPU に倒してから {} トークンごとに 1 回 GPU で測り直す)",
                s.cfg().min_hit,
                s.cfg().probe_every
            );
        }),
        expert_vram::GateDecision::Probe => PROBE_LOGGED.call_once(|| {
            eprintln!(
                "[flash] expert GPU gate: ヒット率が低いので CPU 経路。計測のため {} トークンに 1 回だけ \
                 GPU で試す (DENNO_EXPERT_GPU_MIN_HIT=0 で常時 GPU に戻る)",
                s.cfg().probe_every
            );
        }),
        expert_vram::GateDecision::Gpu => {}
    }
    d
}

/// テスト専用: 平坦化経路の on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_expert_flat_decode_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    EXPERT_FLAT_DECODE_OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// task 285: `DENNO_DEC_ROUTER_TOPK_FAST` の解釈 (純関数)。
///
/// decode の top-k を挿入法 ([`select_topk_into_fast`]) で選ぶか。**既定 on**。
/// 選択規則は `select_nth_unstable_by` 版と同一 (値の降順・同値は小さい index・
/// 出力は index 昇順) なので、値も選ばれる集合も 1 ビット変わらない。
/// 明示的な偽値 (`0` / `off` / `false` / `no`) だけ切る。
fn dec_router_topk_fast_from_value(v: Option<&str>) -> bool {
    !matches!(
        v.map(|s| s.trim().to_ascii_lowercase()).as_deref(),
        Some("0") | Some("off") | Some("false") | Some("no")
    )
}

/// テスト専用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static DEC_ROUTER_TOPK_FAST_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// テスト専用: 挿入法の分岐が実際に通った回数 (「緑 = 発火の証拠」にしないため)。
#[cfg(test)]
static DEC_ROUTER_TOPK_FAST_CALLS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// task 285: decode の top-k に挿入法を使うか (`DENNO_DEC_ROUTER_TOPK_FAST`)。
fn dec_router_topk_fast_on() -> bool {
    match DEC_ROUTER_TOPK_FAST_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        dec_router_topk_fast_from_value(std::env::var("DENNO_DEC_ROUTER_TOPK_FAST").ok().as_deref())
    })
}

/// テスト専用: 上書きを設定する (`None` で env に戻す)。
#[cfg(test)]
fn set_dec_router_topk_fast_for_test(v: Option<bool>) {
    DEC_ROUTER_TOPK_FAST_OVERRIDE.store(
        match v {
            Some(false) => 1,
            Some(true) => 2,
            None => 0,
        },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// テスト専用: ミス平坦化の on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_expert_miss_flat_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    EXPERT_MISS_FLAT_OVERRIDE.store(
        match v {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        },
        Ordering::Relaxed,
    );
}

/// 測定専用: 剪定ゲートと W 統計の収集をまとめて上書きする。
///
/// `Some(cfg)` は **env を無視して** cfg を使う (両ゲートの on/off も cfg で決まる)。
/// `None` で env に戻す。プロセス全体で 1 個なので、触るテストは
/// `moe_test_guard()` を取ること。
pub fn set_expert_prune_measurement(o: Option<ExpertPruneMeasurement>) {
    use std::sync::atomic::Ordering;
    match o {
        Some(cfg) => {
            EXPERT_PRUNE_TOPK.store(cfg.topk.unwrap_or(0) as u32, Ordering::Relaxed);
            EXPERT_PRUNE_MIN_W.store(
                cfg.min_w.map(f32::to_bits).unwrap_or(0),
                Ordering::Relaxed,
            );
            EXPERT_PRUNE_W_STATS.store(cfg.w_stats, Ordering::Relaxed);
            EXPERT_PRUNE_OVERRIDE.store(true, Ordering::Relaxed);
        }
        None => {
            EXPERT_PRUNE_OVERRIDE.store(false, Ordering::Relaxed);
        }
    }
}

impl Default for ExpertWAccum {
    fn default() -> Self {
        Self {
            calls: 0,
            experts_total: 0,
            rank_sum: [0.0; EXPERT_W_RANKS],
            rank_n: [0; EXPERT_W_RANKS],
        }
    }
}

impl ExpertWAccum {
    fn record(&mut self, weights_desc: &[f32], n_run: usize) {
        self.calls += 1;
        self.experts_total += n_run as u64;
        for (r, &w) in weights_desc.iter().take(EXPERT_W_RANKS).enumerate() {
            self.rank_sum[r] += w as f64;
            self.rank_n[r] += 1;
        }
    }
}

impl ExpertWStatsSnapshot {
    /// 層呼び出し数。
    pub fn calls(&self) -> u64 {
        self.calls
    }

    /// 層呼び出しあたりの平均実行 expert 数 (剪定後)。
    pub fn mean_experts_per_call(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.experts_total as f64 / self.calls as f64
        }
    }

    /// rank `rank1` (1 始まり) の平均再正規化重み。その rank を持つ呼び出しが
    /// 無ければ `None`。
    pub fn mean_rank_weight(&self, rank1: usize) -> Option<f64> {
        if rank1 == 0 || rank1 > EXPERT_W_RANKS {
            return None;
        }
        let r = rank1 - 1;
        if self.rank_n[r] == 0 {
            None
        } else {
            Some(self.rank_sum[r] / self.rank_n[r] as f64)
        }
    }

    /// `[WSTATS]` 行の本体 (1 リクエスト 1 行)。rank の無い桁は `-`。
    pub fn line(&self) -> String {
        let mut s = format!(
            "calls={} mean_experts={:.2}",
            self.calls,
            self.mean_experts_per_call()
        );
        for r in 0..EXPERT_W_RANKS {
            if self.rank_n[r] == 0 {
                s.push_str(&format!(" w{}=-", r + 1));
            } else {
                s.push_str(&format!(" w{}={:.4}", r + 1, self.rank_sum[r] / self.rank_n[r] as f64));
            }
        }
        s
    }
}

/// 蓄積を取り出す (取り出すと 0 に戻る)。何も溜まっていなければ `None`。
pub fn take_expert_w_stats() -> Option<ExpertWStatsSnapshot> {
    let mut a = EXPERT_W_ACCUM
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if a.calls == 0 {
        return None;
    }
    let snap = ExpertWStatsSnapshot {
        calls: a.calls,
        experts_total: a.experts_total,
        rank_sum: a.rank_sum,
        rank_n: a.rank_n,
    };
    *a = ExpertWAccum::default();
    Some(snap)
}

/// テスト専用: task 67 の 3 ゲートを env を触らずに切り替える (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_expert_fault_gates_for_test(
    prefetch: Option<bool>,
    hugepage: Option<bool>,
    scratch: Option<bool>,
) {
    use std::sync::atomic::Ordering;
    let enc = |v: Option<bool>| match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    };
    EXPERT_PREFETCH_OVERRIDE.store(enc(prefetch), Ordering::Relaxed);
    EXPERT_HUGEPAGE_OVERRIDE.store(enc(hugepage), Ordering::Relaxed);
    EXPERT_SCRATCH_OVERRIDE.store(enc(scratch), Ordering::Relaxed);
}

/// 選択 expert (`sel`) の 3 行列ぶんの先読みをまとめて投げる (task 67 で切り出し)。
///
/// `DENNO_EXPERT_PREFETCH=0` のときは [`ExpertW::prefetch`] が即座に戻る。
fn prefetch_selected(lw: &LayerW, sel: &[usize]) {
    for &ei in sel {
        lw.gate_exps.prefetch(ei);
        lw.up_exps.prefetch(ei);
        lw.down_exps.prefetch(ei);
    }
}

// ---------------------------------------------------------------------------
// task 850: 層間の expert 共起 (Markov bigram) で層 L+1 を先読みする
// ---------------------------------------------------------------------------

thread_local! {
    /// `run_layer` が層 L の MoE を呼ぶ間だけ張る「次の層」の参照。
    ///
    /// ルータの選択は `moe_ffn_quant_impl` の中で決まるので、層 L+1 の重みを
    /// 引数で渡すには呼び出し鎖を全部変える必要がある。ここでは
    /// [`with_markov_next`] のクロージャの寿命に閉じた生ポインタを 1 本だけ
    /// TLS に置き、その中でだけ読む (外へは出さない。Drop で必ず消す)。
    static MARKOV_NEXT: std::cell::Cell<*const LayerW> = const { std::cell::Cell::new(std::ptr::null()) };

    /// 直近に作った「VRAM 常駐の写し」と、そのときのトークン番号。
    static MARKOV_RESIDENT: std::cell::RefCell<
        Option<(u64, std::rc::Rc<std::collections::HashSet<(u16, u16)>>)>,
    > = std::cell::RefCell::new(None);
}

/// TLS の後始末 (パニック時も消えるようにガードで持つ)。
struct MarkovNextGuard;

impl Drop for MarkovNextGuard {
    fn drop(&mut self) {
        MARKOV_NEXT.with(|c| c.set(std::ptr::null()));
    }
}

/// 層 L の MoE 呼び出しを「次の層 L+1 の重み付き」で包む。
///
/// ゲート off では TLS に触らず `f()` をそのまま呼ぶ (分岐 1 個)。
pub(crate) fn with_markov_next<R>(next: Option<&LayerW>, f: impl FnOnce() -> R) -> R {
    if !crate::model::exp_markov::enabled() {
        return f();
    }
    let p = next.map_or(std::ptr::null(), |l| l as *const LayerW);
    MARKOV_NEXT.with(|c| c.set(p));
    let _g = MarkovNextGuard;
    f()
}

/// ルータ直後に呼ぶ。層 `layer` の選択を記録し、層 `layer+1` の予測 expert を
/// 背景スレッドへ渡す (層 `layer` の計算と読みを重ねる)。
///
/// 予測が外れても捨てるだけで、値・経路は変わらない。VRAM 常駐は積まない。
pub(crate) fn markov_note_and_prefetch(layer: usize, sel: &[usize]) {
    if !crate::model::exp_markov::enabled() {
        return;
    }
    crate::model::exp_markov::observe(layer, sel);
    let next_ptr = MARKOV_NEXT.with(|c| c.get());
    if next_ptr.is_null() {
        return;
    }
    // SAFETY: ポインタは `with_markov_next` のクロージャの間だけ有効で、
    // この関数はそのクロージャの中でしか呼ばれない (`run_layer` の MoE 呼び出し)。
    let next = unsafe { &*next_ptr };
    if next.il != layer + 1 {
        return;
    }
    let max = crate::model::exp_markov::max_per_layer();
    let preds = crate::model::exp_markov::predict(layer, sel, max);
    if preds.is_empty() {
        return;
    }
    crate::model::exp_markov::note_prediction(layer + 1, &preds);
    // VRAM 常駐の写しは 1 トークン 1 回だけ作る (層ごとに LRU のロックを取ると
    // H2D の経路と取り合いになる。lab t850a の decode 退行の一因)。
    let tok = crate::model::exp_markov::token_index();
    let resident = MARKOV_RESIDENT.with(|c| {
        let mut b = c.borrow_mut();
        match b.as_ref() {
            Some((t, set)) if *t == tok => std::rc::Rc::clone(set),
            _ => {
                let set: std::rc::Rc<std::collections::HashSet<(u16, u16)>> =
                    std::rc::Rc::new(
                        expert_vram::resident_keys()
                            .into_iter()
                            .map(|k| (k.layer, k.expert))
                            .collect(),
                    );
                *b = Some((tok, std::rc::Rc::clone(&set)));
                set
            }
        }
    });
    let mut jobs: Vec<Vec<crate::model::exp_up_worker::Part>> = Vec::new();
    let (mut n_res, mut n_nopart) = (0u64, 0u64);
    for eid in preds {
        let key = (next.il as u16, eid);
        if resident.contains(&key) {
            n_res += 1;
            continue;
        }
        let part = |w: &ExpertW| w.slice_mapped(eid as usize).ok().flatten();
        let (Some(g), Some(u), Some(d)) = (
            part(&next.gate_exps),
            part(&next.up_exps),
            part(&next.down_exps),
        ) else {
            n_nopart += 1;
            continue;
        };
        jobs.push(vec![g, u, d]);
    }
    crate::model::exp_markov::note_skips(n_res, n_nopart);
    let n = jobs.len();
    crate::model::exp_markov::note_submitted(n);
    crate::model::exp_prefetch::submit_raw(jobs);
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static SHARED_OVERLAP_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// テスト専用: 重ね打ちの on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
fn set_shared_overlap_for_test(v: Option<bool>) {
    let x = match v {
        Some(false) => 1,
        Some(true) => 2,
        None => 0,
    };
    SHARED_OVERLAP_OVERRIDE.store(x, std::sync::atomic::Ordering::Relaxed);
}

impl SharedExpertOverlap {
    fn none() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            pending: crate::cuda::SharedPending::none(),
            lens: [0, 0],
            full: false,
            down_n: 0,
        }
    }

    /// task 187: down までデバイスで積んだか (呼び出し側は `finish_shared` を使う)。
    fn has_down(&self) -> bool {
        self.full
    }

    /// task 187: デバイスで積んだ down の出力を `sh` へ写す。
    ///
    /// `false` = 取れなかった (呼び出し側は従来の 2 段経路へ落ちる)。
    fn finish_shared(&mut self, sh: &mut [f32]) -> bool {
        #[cfg(feature = "cuda")]
        {
            if !self.full || self.pending.is_empty() || sh.len() < self.down_n {
                return false;
            }
            let mut outs: [&mut [f32]; 1] = [sh];
            return crate::cuda::shared_gemv_finish(&mut self.pending, &mut outs);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = sh;
            false
        }
    }

    /// 層の入口で呼ぶ。`x` は gate/up の入力そのもの。
    fn begin(lw: &LayerW, x: &[f32]) -> Self {
        #[cfg(feature = "cuda")]
        {
            if !shared_overlap_enabled() {
                return Self::none();
            }
            let jobs = [
                crate::cuda::SharedGemvJob {
                    uid: lw.gate_shexp.uid(),
                    ty: lw.gate_shexp.ty(),
                    blocks: lw.gate_shexp.blocks().as_slice(),
                    k: lw.gate_shexp.k(),
                    n: lw.gate_shexp.n(),
                },
                crate::cuda::SharedGemvJob {
                    uid: lw.up_shexp.uid(),
                    ty: lw.up_shexp.ty(),
                    blocks: lw.up_shexp.blocks().as_slice(),
                    k: lw.up_shexp.k(),
                    n: lw.up_shexp.n(),
                },
            ];
            // task 187: 4 発 (gate/up/silu/down) を 1 回で積む経路を先に試す。
            if dec_shared_dev_enabled() {
                let down = crate::cuda::SharedGemvJob {
                    uid: lw.down_shexp.uid(),
                    ty: lw.down_shexp.ty(),
                    blocks: lw.down_shexp.blocks().as_slice(),
                    k: lw.down_shexp.k(),
                    n: lw.down_shexp.n(),
                };
                let pending = crate::cuda::shared_expert_begin(x, &jobs[0], &jobs[1], &down);
                if !pending.is_empty() {
                    return Self {
                        pending,
                        lens: [jobs[0].n, jobs[1].n],
                        full: true,
                        down_n: down.n,
                    };
                }
            }
            let pending = crate::cuda::shared_gemv_begin(x, &jobs);
            return Self {
                pending,
                lens: [jobs[0].n, jobs[1].n],
                full: false,
                down_n: 0,
            };
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (lw, x);
            Self::none()
        }
    }

    /// 積んだ gate/up を join して `sact` / `sup` へ書く。
    ///
    /// `false` = 「GPU の結果が取れなかった」 = 呼び出し側は従来の同期経路で
    /// 計算し直すこと。このとき `sact` / `sup` は 1 要素も書かれていない。
    fn finish(&mut self, sact: &mut [f32], sup: &mut [f32]) -> bool {
        #[cfg(feature = "cuda")]
        {
            if self.pending.is_empty() {
                return false;
            }
            if sact.len() != self.lens[0] || sup.len() != self.lens[1] {
                return false;
            }
            let mut outs: [&mut [f32]; 2] = [sact, sup];
            return crate::cuda::shared_gemv_finish(&mut self.pending, &mut outs);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&self.lens, sact, sup);
            false
        }
    }
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
static SHARED_TAIL_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// テスト専用: 上書きを設定する (`None` で env に戻す)。
#[cfg(test)]
fn set_shared_tail_for_test(v: Option<bool>) {
    let x = match v {
        Some(false) => 1,
        Some(true) => 2,
        None => 0,
    };
    SHARED_TAIL_OVERRIDE.store(x, std::sync::atomic::Ordering::Relaxed);
}

impl SharedDownOverlap {
    fn none() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            pending: crate::cuda::SharedPending::none(),
            n: 0,
        }
    }

    /// `act` は down の入力 (`silu(gate) * up`)、`n` は down の出力長。
    fn begin(lw: &LayerW, act: &[f32], n: usize) -> Self {
        #[cfg(feature = "cuda")]
        {
            // 同じサイドストリームの状態を使うので、`DENNO_SHARED_OVERLAP=0`
            // (重ね打ちそのものが off) のときは積まない。
            if !shared_tail_enabled() || !shared_overlap_enabled() {
                return Self::none();
            }
            let jobs = [crate::cuda::SharedGemvJob {
                uid: lw.down_shexp.uid(),
                ty: lw.down_shexp.ty(),
                blocks: lw.down_shexp.blocks().as_slice(),
                k: lw.down_shexp.k(),
                n: lw.down_shexp.n(),
            }];
            let pending = crate::cuda::shared_gemv_begin(act, &jobs);
            return Self {
                pending,
                n: jobs[0].n,
            };
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (lw, act, n);
            Self::none()
        }
    }

    /// 積んだ down の結果を `out` へ書く。`false` = 取れなかった (呼び出し側が
    /// 従来の同期経路で計算し直す)。
    fn finish(&mut self, out: &mut [f32]) -> bool {
        #[cfg(feature = "cuda")]
        {
            if self.pending.is_empty() || out.len() != self.n {
                return false;
            }
            let mut outs: [&mut [f32]; 1] = [out];
            return crate::cuda::shared_gemv_finish(&mut self.pending, &mut outs);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = (&self.n, out);
            false
        }
    }
}

/// task 144: ミスした expert を **トークンあたりの上限つき**で VRAM へ上げ直す。
///
/// ミスは CPU で計算済みなので、ここでやるのは次のトークンのための H2D だけ。
/// 上限 (`DENNO_EXPERT_IQ_UPLOAD`、既定 8 本 = 約 16 MB) を超えたら何もしない。
/// 予算は層 0 (トークン先頭) で戻す。載っている expert は予算を食わない。
///
/// 失敗 (確保失敗・型未対応) は静かに諦める (リクエストは落とさない)。
#[cfg(feature = "cuda")]
fn iq_warm_misses(lw: &LayerW, sel: &[usize], x_len: usize, e: usize, ff: usize, n_layer: usize) {
    use crate::model::expert_vram::ExpertShape;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    static LEFT: AtomicUsize = AtomicUsize::new(0);
    static DEAD: AtomicBool = AtomicBool::new(false);
    /// このトークンで上げ直す担当層の開始位置 (層に均すための回転)。
    static OFFSET: AtomicUsize = AtomicUsize::new(0);
    if DEAD.load(Ordering::Relaxed) {
        return;
    }
    // LRU に空きがあるうちは詰め込み、埋まったら控えめにする (lab t144d:
    // 埋まってからも 48 本/token 入れると H2D が 7 ms/token かかり、節約を食う)。
    //
    // task 214: `DENNO_DEC_UP_WORKER_MAX` が設定されていればその値を優先する
    // (背景スレッド経路では上げ直しがメインスレッドの外なので、本数の上限だけ
    // 別に測れるようにしてある)。未設定は従来の規則そのもの。
    let worker = crate::model::exp_up_worker::enabled();
    // task 406: 空きがある間に詰め込むか (`DENNO_EXP_UP_FILL`) と、その倍率。
    // 倍率を当てるのは**背景スレッド経路だけ**: 同期 H2D の経路 (ワーカー無し /
    // 起動失敗) はメインスレッドが 1 本 0.16 ms を払うので従来のままにする。
    let fill_on = worker && crate::model::exp_up_worker::fill_enabled();
    let fill_mult = if fill_on {
        crate::model::exp_up_worker::fill_mult()
    } else {
        1
    };
    // `DENNO_DEC_UP_WORKER_MAX` が無いときのワーカー経路は、予算を**トークン先頭で
    // 1 回**だけ決める。`cache_usage()` は LRU のロックを取るので、層ごとに呼ぶと
    // ワーカーの転送中 (ロック保持 0.16 ms) に当たる (48 回/token)。
    static TOKEN_BUDGET: AtomicUsize = AtomicUsize::new(usize::MAX);
    // task 406: 空きがある間の予算 (`DENNO_EXP_UP_FILL`)。埋まっているとき
    // (追い出しが起きる) は従来の控えめな値のまま = 充填の速さだけを変える。
    let free_budget = || {
        let base = crate::model::exp_iq::upload_per_token();
        crate::model::exp_up_worker::upload_budget(base, fill_mult)
    };
    let budget = match crate::model::exp_up_worker::max_per_token() {
        Some(n) => n,
        None => match (worker, lw.il) {
            (true, 0) => {
                let b = match expert_vram::cache_usage() {
                    Some((used, cap)) if cap > 0 && used.saturating_add(32 << 20) >= cap => {
                        crate::model::exp_iq::upload_when_full()
                    }
                    _ => free_budget(),
                };
                TOKEN_BUDGET.store(b, Ordering::Relaxed);
                b
            }
            (true, _) => {
                let b = TOKEN_BUDGET.load(Ordering::Relaxed);
                if b == usize::MAX {
                    free_budget()
                } else {
                    b
                }
            }
            (false, _) => match expert_vram::cache_usage() {
                Some((used, cap)) if cap > 0 && used.saturating_add(32 << 20) >= cap => {
                    crate::model::exp_iq::upload_when_full()
                }
                _ => free_budget(),
            },
        },
    };
    let nl = n_layer.max(1);
    if lw.il == 0 {
        // 予算 `budget` 本/token を層に均す: 1 トークンに budget 層ぶんを担当し、
        // 次のトークンはその続きから (前は層 0 に全部積んで層 1 以降が永久に
        // 当たらなかった)。budget >= n_layer なら全層が毎トークン担当。
        OFFSET.store(
            (OFFSET.load(Ordering::Relaxed) + budget.max(1)) % nl,
            Ordering::Relaxed,
        );
        LEFT.store(budget, Ordering::Relaxed);
    }
    let shapes = [
        ExpertShape {
            ty: lw.gate_exps.ty,
            k: x_len,
            n: ff,
        },
        ExpertShape {
            ty: lw.up_exps.ty,
            k: x_len,
            n: ff,
        },
        ExpertShape {
            ty: lw.down_exps.ty,
            k: ff,
            n: e,
        },
    ];
    let key_of = |ei: usize| ExpertKey {
        layer: lw.il as u16,
        expert: ei as u16,
    };

    // 1. 参照とヒットは**全層**で数える (担当外の層も数えないとヒット率が偏る)。
    let mut hits = 0u32;
    let mut miss: Vec<usize> = Vec::new();
    if worker {
        // task 214: ワーカーが転送中は LRU のロックが 1 本 0.16 ms 埋まる。
        // 1 層 10 回のロック取得を 1 回に寄せる (`lookup_many` は `lookup` と
        // 同じ `get_mut` を順に呼ぶだけ = 触り方も順序も同じ)。
        let keys: Vec<ExpertKey> = sel.iter().map(|&ei| key_of(ei)).collect();
        match expert_vram::lookup_many(&keys, shapes) {
            Ok(got) => {
                for (i, g) in got.iter().enumerate() {
                    match g {
                        Some(_) => hits += 1,
                        None => miss.push(sel[i]),
                    }
                }
            }
            Err(_) => {
                // 形の不一致 = 実装の誤り。参照を数えるだけで上げ直しはしない。
                crate::model::exp_iq::note_lookup(sel.len() as u32, hits);
                return;
            }
        }
    } else {
        for &ei in sel {
            match expert_vram::lookup(key_of(ei), shapes) {
                Ok(Some(_)) => hits += 1,
                Ok(None) => miss.push(ei),
                Err(_) => break,
            }
        }
    }
    crate::model::exp_iq::note_lookup(sel.len() as u32, hits);

    // 2. 上げ直しは担当層だけ、予算の範囲で。
    let off = OFFSET.load(Ordering::Relaxed) % nl;
    let served = budget >= nl || (lw.il + nl - off) % nl < budget;
    if !served || budget == 0 {
        return;
    }
    // task 190: ヒット本数ごとの表 (PROFILE のときだけ Instant 2 回)。
    let t_up = crate::model::q4ledger::stamp();
    let mut n_up = 0u32;
    // 1 層あたりの上限: budget >= n_layer のときは均等割 (1 層に集中させない)。
    let quota = if budget >= nl { budget / nl } else { 1 };
    let mut left = LEFT.load(Ordering::Relaxed);
    let mut can = left.min(quota);
    for ei in miss {
        if can == 0 {
            break;
        }
        if worker {
            // task 214: 背景スレッドへ渡す (メインスレッドは待たない)。
            // 1 本でも積めなければ従来の同期経路へ落ちる。
            // task 406: `DENNO_EXP_UP_FILL` のときは `Already` / `Refused` で
            // **この層の残りを諦める** (予算を 2 倍にした目的はメインスレッドを
            // 止めないことなので、行列が詰まったときに 0.16 ms の同期 H2D を
            // 撃つと逆行する)。写像できない重み (`NoMap`) は従来どおり同期経路へ。
            let outcome = submit_deferred_upload(lw, ei, shapes);
            if crate::model::exp_up_worker::skip_on_refusal(fill_on, outcome) {
                break;
            }
            if outcome == crate::model::exp_up_worker::SubmitOutcome::Queued {
                left -= 1;
                can -= 1;
                LEFT.store(left, Ordering::Relaxed);
                // 本数は「上げ直し」として数える (時間はワーカーの [dec-up-w] が持つ)。
                crate::model::exp_iq::note_upload();
                n_up += 1;
                continue;
            }
        }
        let t = std::time::Instant::now();
        // task 260: 転送元は RAM tier → mmap の順 (`acquire_expert`)。
        let got = acquire_expert(lw, ei, key_of(ei), shapes);
        crate::model::exp_iq::note_upload_ns(t.elapsed().as_nanos() as u64);
        match got {
            Ok(Some(_)) => {
                left -= 1;
                can -= 1;
                LEFT.store(left, Ordering::Relaxed);
                crate::model::exp_iq::note_upload();
                n_up += 1;
            }
            // 確保失敗・型未対応は以後やらない (リクエストは落とさない)。
            _ => {
                DEAD.store(true, Ordering::Relaxed);
                break;
            }
        }
    }
    if let Some(t) = t_up {
        crate::model::exp_iq::note_layer_upload(hits, n_up, t.elapsed().as_nanos() as u64);
    }
}

/// task 214: 1 本を背景スレッドの待ち行列へ積む。
///
/// 積めたら [`SubmitOutcome::Queued`]。積めない理由は呼び出し側が使う
/// (task 406 の `DENNO_EXP_UP_FILL` は `Already` / `Refused` でこのトークンを諦め、
/// メインスレッドの同期 H2D へ落とさない)。
#[cfg(feature = "cuda")]
fn submit_deferred_upload(
    lw: &LayerW,
    ei: usize,
    shapes: [crate::model::expert_vram::ExpertShape; 3],
) -> crate::model::exp_up_worker::SubmitOutcome {
    use crate::model::exp_up_worker::SubmitOutcome;
    let part = |w: &ExpertW| w.slice_mapped(ei).ok().flatten();
    let (Some(g), Some(u), Some(d)) = (
        part(&lw.gate_exps),
        part(&lw.up_exps),
        part(&lw.down_exps),
    ) else {
        return SubmitOutcome::NoMap;
    };
    crate::model::exp_up_worker::submit_detailed(crate::model::exp_up_worker::Job {
        layer: lw.il as u16,
        expert: ei as u16,
        shapes,
        parts: [g, u, d],
    })
}

/// task 146/263: 起動プラン。記録した使用回数の多い順に **VRAM tier → RAM tier**
/// へ 1 回のパスで載せる (起動時の背景スレッド)。
///
/// 呼び出し側 ([`crate::model::exp_stats`] と CLI) の契約:
///
/// - ゲートは [`crate::model::exp_stats::enabled`]。呼ばれても、ファイルが無い・
///   壊れている・版違い・別モデルなら **何も載せずに 1 行だけ返す**。起動も
///   リクエストも落とさない。
/// - **回数の多い順**に載せる。まず VRAM (`DENNO_EXPERT_VRAM_MB` - 予備、
///   実カードの空きでも丸める)、次に残りを RAM tier (`DENNO_EXPERT_RAM_MB`)。
///   どちらも decode 中 (`exp_stats::busy()`) は 2 ms 寝て譲る。
/// - VRAM は載せ終わったら「少ない順」に touch し直す (H2D なし)。LRU の末尾 =
///   最初に追い出される、が一番使わない expert になる。
/// - 戻り値は 1 行のログ (tier ごとの本数・MB・記録された使用回数の覆盖率・時間)。
#[cfg(feature = "cuda")]
pub fn plan_experts_from_stats(
    parts: &[std::path::PathBuf],
    dir_name: &str,
) -> Option<String> {
    use crate::model::exp_stats;
    use crate::model::expert_vram::ExpertShape;
    use std::time::{Duration, Instant};

    let path = exp_stats::stats_path()?;
    let t0 = Instant::now();
    let f = match GgufFile::open_split(parts) {
        Ok(f) => f,
        Err(e) => return Some(format!("[exp-plan] off: GGUF を開けない: {e}")),
    };
    let p = match Qwen4ExpParams::from_meta(&f) {
        Ok(p) => p,
        Err(e) => return Some(format!("[exp-plan] off: メタデータを読めない: {e}")),
    };
    let total_bytes = expert_identity_bytes(&f, &p);
    // task 798: テンソル目録の指紋 (同じ名前 + 同じバイト数でも別リビジョンを弾く)。
    let fingerprint = exp_stats::fingerprint_tensors(&f);
    // task 268: 本番の「モデルを切り替えた直後の cold start」を lab で作る。
    //
    // ゲート (`DENNO_EXPERT_DROP_CACHE_ON_START=1`) のときだけ、**事前ロードが
    // 1 バイトも読む前**に expert のページを落とす。dense / PLE は落とさない
    // (本番で冷えるのは expert の範囲だけで、dense は起動時に GPU へ上げる)。
    // 記録の有無に依らず効くよう、記録を読む**前**に置く (lab の base 側も
    // 同じ条件で冷やせる)。
    let mut return_line = String::new();
    if crate::model::expert_ram::drop_on_start_enabled() {
        let mut ranges: Vec<(usize, u64, u64)> = Vec::with_capacity(p.n_layer * 3);
        let mut probes: Vec<(usize, usize)> = Vec::with_capacity(p.n_layer * 3);
        let mut resident_before = 0u64;
        for il in 0..p.n_layer {
            for m in [
                "ffn_gate_exps.weight",
                "ffn_up_exps.weight",
                "ffn_down_exps.weight",
            ] {
                let name = format!("blk.{il}.{m}");
                let Some((part, off, len)) = f.tensor_mapping_parts(&name) else {
                    continue;
                };
                if let Some(map) = f.mmap_handle_at(part) {
                    let ptr = map.as_ptr() as usize + off;
                    resident_before += crate::model::expert_ram::resident_bytes_of(ptr, len);
                    probes.push((ptr, len));
                }
                ranges.push((part, off as u64, len as u64));
            }
        }
        let t_drop = Instant::now();
        let line = crate::model::expert_ram::drop_ranges(parts, &ranges);
        let mut resident_after = 0u64;
        for (ptr, len) in probes {
            resident_after += crate::model::expert_ram::resident_bytes_of(ptr, len);
        }
        return_line.push_str(&format!(
            "{line} resident={}MiB->{}MiB in {:.2}s; ",
            resident_before >> 20,
            resident_after >> 20,
            t_drop.elapsed().as_secs_f64()
        ));
        eprintln!("{return_line}");
    }
    let zeros = |dir: &str| {
        exp_stats::Counts::zeros_with_fingerprint(
            dir,
            total_bytes,
            fingerprint,
            p.n_layer as u32,
            p.n_expert as u32,
        )
    };
    let counts = match exp_stats::load_file(&path, dir_name, total_bytes, fingerprint) {
        Ok(c) => c,
        Err(e) => {
            // 記録は使わないが、このセッションの回数は保存したいので履歴を据える。
            exp_stats::install_prior(&zeros(dir_name));
            return Some(format!("[exp-plan] off: {e}{return_line}"));
        }
    };
    if counts.n_layer as usize != p.n_layer || counts.n_expert as usize != p.n_expert {
        exp_stats::install_prior(&zeros(dir_name));
        return Some(format!(
            "[exp-plan] off: 記録の形が違う ({}x{} / 今回 {}x{}){return_line}",
            counts.n_layer, counts.n_expert, p.n_layer, p.n_expert
        ));
    }
    exp_stats::install_prior(&counts);
    // task 798: `DENNO_START_WARM_EXPERTS` は起動時の **page cache warm** の本数上限。
    // RAM tier / VRAM には積まない (UD 級で 12 GB を RAM に写すと page cache を
    // 追い出して decode が +63% になった — lab t798b)。プラン自体は切らない。
    let warm_cap = exp_stats::warm_experts_cap();
    let plan = exp_stats::preload_order(&counts);
    if plan.is_empty() {
        return Some(format!(
            "[exp-plan] off: 記録が空 (回数 0) (warm_experts={warm_cap:?}){return_line}"
        ));
    }
    let n_plan = plan.len();

    // 層ごとの読み口 (mmap のスライスだけ。ページは触らない)。
    let mut layers: Vec<(ExpertW, ExpertW, ExpertW)> = Vec::with_capacity(p.n_layer);
    for il in 0..p.n_layer {
        let load = |m: &str| ExpertW::load(&f, &format!("blk.{il}.{m}"), &p);
        match (
            load("ffn_gate_exps.weight"),
            load("ffn_up_exps.weight"),
            load("ffn_down_exps.weight"),
        ) {
            (Ok(g), Ok(u), Ok(d)) => layers.push((g, u, d)),
            (e, _, _) => {
                let msg = match e {
                    Err(err) => format!("{err}"),
                    Ok(_) => "読み口を開けない".to_string(),
                };
                return Some(format!("[exp-plan] off: 層 {il}: {msg}"));
            }
        }
    }

    // task 798: 起動時の page cache warm。熱い側から `DENNO_START_WARM_EXPERTS` 本の
    // expert 範囲を pread して page cache に載せる (mmap はそのまま、RSS は増えない)。
    //
    // RAM tier や VRAM に写す方式は UD 級で負ける: 4,000 本 = 11.98 GB を RAM tier に
    // 写すと page cache を同じだけ追い出し、しかも VRAM のぶんを差し引いた「冷たい側」
    // が残るので、decode が 12.0 -> 19.5 ms/tok (+63%) になった (lab t798b、UD-Q4_K_XL)。
    if let Some(n) = warm_cap {
        if n > 0 {
            let budget = exp_stats::warm_mb_bytes().unwrap_or(u64::MAX);
            let mut ranges: Vec<crate::model::start_warm::Range> = Vec::new();
            let mut used = 0u64;
            let mut picked = 0usize;
            for &(layer, expert) in plan.iter().rev().take(n) {
                let Some((g, u, d)) = layers.get(layer as usize) else {
                    continue;
                };
                let mut got = [(0usize, 0u64, 0u64); 3];
                let mut ok = true;
                for (i, w) in [g, u, d].into_iter().enumerate() {
                    match w.slice_abs(expert as usize) {
                        Ok((off, len)) => {
                            if exp_stats::budget_full(used as usize, len, budget as usize) {
                                ok = false;
                                break;
                            }
                            used = used.saturating_add(len as u64);
                            got[i] = (w.file, off, len as u64);
                        }
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if !ok {
                    break;
                }
                picked += 1;
                ranges.extend_from_slice(&got);
            }
            if !ranges.is_empty() {
                let rep = crate::model::start_warm::warm(
                    parts,
                    &ranges,
                    crate::model::start_warm::budget_secs(),
                    crate::model::start_warm::read_threads(),
                );
                eprintln!(
                    "[exp-warm] hot_experts={picked} budget={}MiB {}",
                    budget >> 20,
                    rep.line()
                );
            }
        }
    }

    // VRAM 予算をここで初期化する (初回の MoE 呼び出しと同じ 1 回きりの決定)。
    let _ = expert_route();
    let vram_cap = expert_vram::cache_usage().map(|(_, cap)| cap);
    let ram_on = crate::model::expert_ram::enabled();
    if vram_cap.is_none() && !ram_on {
        return Some(format!(
            "[exp-plan] off: VRAM も RAM tier も無効 (DENNO_EXPERT_VRAM_MB=0 / DENNO_EXPERT_RAM_MB=0){return_line}"
        ));
    }
    // CUDA の current context はスレッドローカル。これをしないと最初の
    // cuMemAlloc が error 201 (INVALID_CONTEXT) で落ちる (lab t146a)。
    if !crate::cuda::make_current_on_this_thread() {
        return Some(format!(
            "[exp-plan] off: CUDA の context をこのスレッドに張れない{return_line}"
        ));
    }
    let reserve = exp_stats::preload_reserve_bytes();
    let mut target = vram_cap.unwrap_or(0).saturating_sub(reserve);
    // 実カードの容量でも丸める: dense 重みの予算 (`DENNO_GPU_WEIGHT_MB`) と
    // 予備を引いたぶんまで (t147a: 24,000 + 重み 12,000 は 32 GB に収まらず OOM)。
    let weight_cap = crate::gpu_budget::weight_cap_bytes() as usize;
    if let Some((_, total_mb)) = crate::cuda::dev_mem_mb() {
        let physical = (total_mb << 20)
            .saturating_sub(weight_cap)
            .saturating_sub(reserve);
        target = target.min(physical);
    }
    let vram_on = vram_cap.is_some_and(|c| c > reserve) && target > 0;
    // task 268: **UD 級** (expert 総量 > 空きメモリ) は VRAM の事前ロードをしない。
    //
    // 根拠 (lab t268a、UD-Q4_K_XL): 事前ロードは (a) 最初のリクエストの prefill が
    // VRAM を自分の expert で埋めるので数秒で追い出され、(b) 18 GB の読みが
    // 最初のリクエストと disk を取り合う。t268a の `pre` は VRAM 段だけで 50 秒
    // かかり (その間ずっと busy 待ち)、RAM tier の構築が最初のリクエストの後に
    // なって tier が 1 度も効かなかった。UD は **RAM tier に全部を寄せる**。
    let ud_class = ram_on && total_bytes > crate::model::expert_ram::mem_available().unwrap_or(u64::MAX);
    let vram_on = vram_on && !ud_class;

    let mut loaded = 0usize;
    let mut paused_ms = 0f64;
    // 前の失敗理由が残っていれば捨てる (停止理由の取り違えを防ぐ)。
    let _ = expert_vram::take_alloc_error();
    // 止まった理由 (None = 最後まで載せた)。
    let mut stop: Option<String> = if ud_class {
        Some(format!(
            "UD 級 (expert {} MiB > MemAvailable): VRAM へは載せず RAM tier のみ",
            total_bytes >> 20
        ))
    } else {
        None
    };
    // 載せた順 (最後に touch を掛け直して「一番よく使う = 最後に使った」にする)。
    let mut loaded_keys: Vec<ExpertKey> = Vec::new();
    // まだ載せていない plan の本数 (RAM tier はここから続ける)。
    let mut remain = plan.len();
    if ud_class {
        // task 268: VRAM の事前ロードをしない代わりに、**decode の LRU が埋める
        // ぶん (target / 1 本) を RAM tier から除く**。除かないと tier が一番熱い
        // 層を抱えてしまい、それは VRAM に載っているので 1 度も引かれない
        // (t268a の miss_ram=14 がまさにそれ)。tier は「VRAM の次に熱い層」を受ける。
        let n_all = (p.n_layer * p.n_expert).max(1) as u64;
        let per = (total_bytes / n_all).max(1);
        let vram_slots = (target as u64 / per) as usize;
        remain = plan.len().saturating_sub(vram_slots);
    }
    if vram_on {
        for i in (0..plan.len()).rev() {
            let (layer, expert) = plan[i];
            remain = i + 1;
            let (li, ei) = (layer as usize, expert as usize);
            let Some((g, u, d)) = layers.get(li) else {
                continue;
            };
            let shapes = [
                ExpertShape {
                    ty: g.ty,
                    k: g.n_in,
                    n: g.n_out,
                },
                ExpertShape {
                    ty: u.ty,
                    k: u.n_in,
                    n: u.n_out,
                },
                ExpertShape {
                    ty: d.ty,
                    k: d.n_in,
                    n: d.n_out,
                },
            ];
            let Ok(want) = shapes.iter().try_fold(0usize, |a, s| s.bytes().map(|b| a + b)) else {
                stop = Some("expert のバイト数が求まらない".to_string());
                break;
            };
            let used = expert_vram::cache_usage().map_or(0, |(u, _)| u);
            if !exp_stats::fits(used as u64, want as u64, target as u64) {
                stop = Some(format!("VRAM 予算 (target {} MiB) に到達", target >> 20));
                break;
            }
            // decode 中は譲る (リクエストを待たせない)。
            if exp_stats::busy() {
                let t = Instant::now();
                while exp_stats::busy() {
                    std::thread::sleep(Duration::from_millis(2));
                }
                paused_ms += t.elapsed().as_secs_f64() * 1e3;
            }
            let key = ExpertKey { layer, expert };
            let got = expert_vram::acquire_soft(key, shapes, || {
                Ok([g.slice(ei)?, u.slice(ei)?, d.slice(ei)?])
            });
            match got {
                Ok(Some(_)) => {
                    loaded += 1;
                    loaded_keys.push(key);
                    remain = i;
                }
                // 型が未対応・確保失敗はそこで止める。理由は 1 行に残す。
                Ok(None) => {
                    stop = Some(match expert_vram::take_alloc_error() {
                        Some(e) => format!("確保に失敗: {e}"),
                        None => "この expert は GPU に載せられない (型・予算)".to_string(),
                    });
                    break;
                }
                Err(e) => {
                    stop = Some(format!("{e:#}"));
                    break;
                }
            }
        }
    }
    // 載せたものを「少ない順」に触り直す (H2D なし)。LRU の末尾 (= 最初に
    // 追い出される) が一番使わない expert になり、task の「most used are
    // evicted last」になる。
    loaded_keys.reverse();
    if !loaded_keys.is_empty() {
        expert_vram::touch_many(&loaded_keys);
    }
    let vram_bytes = expert_vram::cache_usage().map_or(0, |(u, _)| u) as u64;
    let vkeys: Vec<(u16, u16)> = loaded_keys.iter().map(|k| (k.layer, k.expert)).collect();
    let (cov_vram, total_uses) = exp_stats::coverage(&counts, &vkeys);
    // task 268: VRAM 段が終わった時点で 1 行出す。RAM tier の充填が長引いて
    // サーバが落とされても、VRAM に何本載ったかがログに残るようにする
    // (t263c は [exp-plan] が最後まで出ず、VRAM の結果が分からなかった)。
    if vram_on {
        eprintln!(
            "[exp-plan] vram phase: loaded={loaded} experts {}MiB (cover {:.3}) in {:.2}s stop={stop:?}",
            vram_bytes >> 20,
            if total_uses > 0 {
                cov_vram as f64 / total_uses as f64
            } else {
                0.0
            },
            t0.elapsed().as_secs_f64()
        );
    }

    // ---- RAM tier: 残りを回数の多い順に ----
    let mut ram_experts = 0usize;
    let mut ram_bytes = 0u64;
    let mut cov_ram = 0u64;
    let mut ram_msg = String::new();
    if ram_on && remain > 0 && crate::model::expert_ram::fill_enabled() {
        match expert_ram_build(&f, &layers) {
            Ok(Some(set)) => {
                let cap = set.slots(ExpertMatrix::Gate);
                let todo: Vec<(u16, u16)> = (0..remain).rev().map(|i| plan[i]).collect();
                let (bytes, n, secs, rstop) = expert_ram_fill(&set, &layers, &todo);
                ram_experts = n;
                ram_bytes = bytes;
                let rkeys: Vec<(u16, u16)> = crate::model::expert_ram::keys()
                    .iter()
                    .map(|k| (k.layer, k.expert))
                    .collect();
                cov_ram = exp_stats::coverage(&counts, &rkeys).0;
                ram_msg = format!(
                    " ram={n}/{cap} experts {}MiB (cover {:.3}) in {secs:.2}s{}",
                    bytes >> 20,
                    if total_uses > 0 { cov_ram as f64 / total_uses as f64 } else { 0.0 },
                    if rstop { " stopped" } else { "" }
                );
            }
            Ok(None) => ram_msg = " ram=off".to_string(),
            Err(e) => ram_msg = format!(" ram=off ({e:#})"),
        }
    } else if ram_on {
        ram_msg = " ram=full (残り 0)".to_string();
    }

    let secs = t0.elapsed().as_secs_f64();
    let frac = |a: u64| {
        if total_uses > 0 {
            a.min(total_uses) as f64 / total_uses as f64
        } else {
            0.0
        }
    };
    let cover_total = frac(cov_vram.saturating_add(cov_ram));
    exp_stats::set_plan(exp_stats::PlanSnapshot {
        at: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        planned_experts: n_plan,
        vram_experts: loaded,
        vram_bytes,
        ram_experts,
        ram_bytes,
        cover_vram: frac(cov_vram),
        cover_ram: frac(cov_ram),
        cover_total,
        secs,
        promoted: 0,
        dropped: 0,
    });
    Some(format!(
        "[exp-plan] planned={n_plan} uses={total_uses} vram={loaded} experts {}MiB (cover {:.3}){ram_msg} \
         total cover {cover_total:.3} rest=mmap in {secs:.2}s paused={paused_ms:.0}ms \
         warm_cap={warm_cap:?} warm_mb={:?} stop={stop:?}{return_line}",
        vram_bytes >> 20,
        frac(cov_vram),
        exp_stats::warm_mb_bytes().map(|b| b >> 20)
    ))
}

/// `cuda` feature 無しでは tier に載せられない (常に `None` = 何もしない)。
#[cfg(not(feature = "cuda"))]
pub fn plan_experts_from_stats(
    _parts: &[std::path::PathBuf],
    _dir_name: &str,
) -> Option<String> {
    None
}

/// task 263: RAM tier を「いまの回数の上位」に寄せる 1 パス。
///
/// - 欲しい集合 = 回数の多い順に `slots` 本 (スロット数 = RAM 予算ぶん)。
/// - それ以外の常駐を落とし、足りないぶんを昇順 (少ない順) に埋める
///   (一番熱いのが最後に触られて、次に落とされにくい)。
/// - 64 本ずつ・チャンクの間で 20 ms 寝る。decode 中は譲る。**低優先度**。
///
/// 変更が無ければ `None`。
#[cfg(feature = "cuda")]
fn rebalance_once(parts: &[std::path::PathBuf], dir_name: &str) -> Option<String> {
    use crate::model::{exp_stats, expert_ram};
    use std::time::Duration;

    let set = expert_ram::get()?;
    let counts = exp_stats::current_counts()?;
    if counts.dir != dir_name {
        return None;
    }
    let plan = exp_stats::preload_order(&counts);
    let cap = set.slots(ExpertMatrix::Gate);
    if cap == 0 || plan.is_empty() {
        return None;
    }
    // VRAM に載っている expert は RAM tier に重複して置かない (起動プランと
    // 同じ「VRAM → RAM」の順)。VRAM の LRU は走行中に入れ替わるので毎回引く。
    let vram: std::collections::HashSet<(u16, u16)> = crate::model::expert_vram::keys()
        .iter()
        .map(|k| (k.layer, k.expert))
        .collect();
    let have: Vec<(u16, u16)> = set.keys().iter().map(|k| (k.layer, k.expert)).collect();
    let (drop_keys, fill_keys) = expert_ram::rebalance_plan(&plan, cap, &vram, &have);
    if drop_keys.is_empty() && fill_keys.is_empty() {
        return None;
    }
    let mut dropped = 0usize;
    for chunk in drop_keys.chunks(64) {
        while exp_stats::busy() {
            std::thread::sleep(Duration::from_millis(2));
        }
        for &(layer, expert) in chunk {
            if set.evict(ExpertKey { layer, expert }) {
                dropped += 1;
            }
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut promoted = 0usize;
    if !fill_keys.is_empty() {
        let f = GgufFile::open_split(parts).ok()?;
        let p = Qwen4ExpParams::from_meta(&f).ok()?;
        let mut layers: Vec<(ExpertW, ExpertW, ExpertW)> = Vec::with_capacity(p.n_layer);
        for il in 0..p.n_layer {
            let load = |m: &str| ExpertW::load(&f, &format!("blk.{il}.{m}"), &p);
            match (
                load("ffn_gate_exps.weight"),
                load("ffn_up_exps.weight"),
                load("ffn_down_exps.weight"),
            ) {
                (Ok(g), Ok(u), Ok(d)) => layers.push((g, u, d)),
                _ => return None,
            }
        }
        for chunk in fill_keys.chunks(64) {
            while exp_stats::busy() {
                std::thread::sleep(Duration::from_millis(2));
            }
            for &(layer, expert) in chunk {
                let Some((g, u, d)) = layers.get(layer as usize) else {
                    continue;
                };
                let key = ExpertKey { layer, expert };
                let mut ok = true;
                for (mi, w) in [g, u, d].into_iter().enumerate() {
                    let Ok((off, len)) = w.slice_abs(expert as usize) else {
                        ok = false;
                        break;
                    };
                    if set.get(ExpertMatrix::ALL[mi], w.file, key, off, len).is_err() {
                        ok = false;
                        break;
                    }
                }
                if ok {
                    promoted += 1;
                }
            }
            // task 907: 予算 / 空きが尽きたら、このパスはここで止める
            // (残りは次のパスで、状態が変わったときだけ試す)。
            if set.stopped() {
                break;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }
    exp_stats::add_rebalance(promoted, dropped);
    let (cov, total) = exp_stats::coverage(&counts, &expert_ram::keys().iter().map(|k| (k.layer, k.expert)).collect::<Vec<_>>());
    Some(format!(
        "[exp-plan] rebalance promoted={promoted} dropped={dropped} resident={} cover={:.3}",
        expert_ram::keys().len(),
        if total > 0 { cov as f64 / total as f64 } else { 0.0 }
    ))
}

/// task 263: 起動プラン + 走行中の再調整を 1 本の背景スレッドで回す。
///
/// 起動プランは `build_model` より前に起こす (構築の数秒を充填に使う)。
/// RAM tier が無い・記録が無いときはプランが 1 行返すだけで、再調整も回さない。
#[cfg(feature = "cuda")]
pub fn spawn_expert_plan(parts: Vec<std::path::PathBuf>, dir_name: String) {
    std::thread::Builder::new()
        .name("exp-plan".into())
        .spawn(move || {
            if let Some(line) = plan_experts_from_stats(&parts, &dir_name) {
                eprintln!("{line}");
            }
            let secs = crate::model::expert_ram::rebalance_secs();
            // `DENNO_EXPERT_RAM_FILL=0` は「起動時に埋めない」なので、
            // 追従 (再調整) も回さない (使用時に埋まる経路だけ残す)。
            if secs == 0 || !crate::model::expert_ram::fill_enabled() {
                return;
            }
            loop {
                std::thread::sleep(std::time::Duration::from_secs(secs));
                if !crate::model::expert_ram::active() || !crate::model::exp_stats::enabled() {
                    continue;
                }
                if let Some(line) = rebalance_once(&parts, &dir_name) {
                    eprintln!("{line}");
                }
            }
        })
        .ok();
}

/// `cuda` feature 無しでは何もしない。
#[cfg(not(feature = "cuda"))]
pub fn spawn_expert_plan(_parts: Vec<std::path::PathBuf>, _dir_name: String) {}

/// テストが env を立てずに token 分割を強制するための上書き。
/// `usize::MAX` = 未設定 (env と既定に従う)。
#[cfg(test)]
static TOK_SPLIT_MAX_OVR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

#[cfg(test)]
static TOK_SPLIT_MIN_OVR: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(usize::MAX);

/// テスト専用: token 分割の (最大チャンク数, 最小 m) を強制する (`None` で env に戻す)。
#[cfg(test)]
fn set_moe_tok_split_for_test(max_c: Option<usize>, min_m: Option<usize>) {
    use std::sync::atomic::Ordering::Relaxed;
    TOK_SPLIT_MAX_OVR.store(max_c.unwrap_or(usize::MAX), Relaxed);
    TOK_SPLIT_MIN_OVR.store(min_m.unwrap_or(usize::MAX), Relaxed);
}

impl MoeScratch {
    /// 必要な長さに合わせる (伸ばすときだけゼロ埋めが走る = 初回だけ)。
    fn resize(&mut self, tot: usize, e: usize, ff: usize, need_x: bool) {
        // task 291: 索引付き経路 (`DENNO_MOE_FLAT_IDX=1`) は gather をしないので
        // `xin` を確保しない (確保すると毎層 300 MB 超のゼロ埋めが残る)。
        self.xin.resize(if need_x { tot * e } else { 0 }, 0.0);
        self.act.resize(tot * ff, 0.0);
        self.up.resize(tot * ff, 0.0);
        self.h.resize(tot * e, 0.0);
    }

    fn bytes(&self) -> usize {
        (self.xin.capacity() + self.act.capacity() + self.up.capacity() + self.h.capacity()) * 4
            + self.q8.capacity()
            + (self.qsc.capacity() + self.qgs.capacity()) * 4
    }
}

#[cfg(feature = "cuda")]
impl Drop for PfStreamGuard {
    fn drop(&mut self) {
        crate::cuda::pf_stream_release();
    }
}

impl ExpertScratch {
    /// P64 の新経路用 (`act` / `up` だけを用意する)。
    #[cfg(feature = "par")]
    fn new(ff: usize) -> Self {
        let mut s = Self::default();
        s.prepare(ff, 0);
        s
    }

    /// 使う長さまで伸ばし (grow-only)、**使う範囲をゼロ埋めする**。
    fn prepare(&mut self, ff: usize, e: usize) {
        if self.act.len() < ff {
            self.act.resize(ff, 0.0);
        }
        if self.up.len() < ff {
            self.up.resize(ff, 0.0);
        }
        if self.h.len() < e {
            self.h.resize(e, 0.0);
        }
        self.act[..ff].fill(0.0);
        self.up[..ff].fill(0.0);
        self.h[..e].fill(0.0);
    }
}

/// 作業域を借りて `f` を実行する。`DENNO_EXPERT_SCRATCH=0` のときは従来どおり
/// 呼び出しごとに確保する (A/B 用。数値はどちらも同じ)。
fn with_expert_scratch<R>(ff: usize, e: usize, f: impl FnOnce(&mut ExpertScratch) -> R) -> R {
    if expert_scratch_enabled() {
        #[cfg(test)]
        EXPERT_SCRATCH_REUSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        EXPERT_SCRATCH.with(|cell| {
            let mut s = cell.borrow_mut();
            s.prepare(ff, e);
            f(&mut s)
        })
    } else {
        let mut s = ExpertScratch::default();
        s.prepare(ff, e);
        f(&mut s)
    }
}
// impl Qwen4Exp のうち decode 本線 (run_layer / forward_token_at / spec / arch 出力)。
impl Qwen4Exp {
    /// 実 GGUF から組む。**全テンソルの存在・形状・量子型を先に検査**して、
    /// 1 本でも欠けたら `Err` にする (黙って 0 で埋めない)。
    pub fn new(f: &GgufFile) -> Result<Self> {
        Self::with_attn(f, Box::new(IdentityAttn))
    }

    /// attention / ssm の実装を差し替えて組む (F2 / F3 用)。
    pub fn with_attn(f: &GgufFile, attn: Box<dyn LayerAttn>) -> Result<Self> {
        let p = validate(f)?;
        // task 217: `DENNO_INTERVENE` の env プリロード (lab 用)。プロセスで 1 回だけ。
        intervene::init_from_env(&intervene::Shape {
            n_layer: p.n_layer,
            n_expert: p.n_expert,
            n_expert_used: p.n_expert_used,
            n_hc: p.n_hc,
        });
        // task 218: 残差の平均の記録 / ステアリングの初期化は **serve_flash が
        // 起動時に 1 回だけ**行う (`Flash` を組むところ。`cfg.dir` からモデル id を
        // 決める)。エンジン側では初期化しない — 2 箇所で初期化するとモデル id が
        // 食い違ってスナップショットを取り違えるため。

        let tok_embd = linear(f, "token_embd.weight")?;
        let head = HeadW::load(f)?;
        let mut layers = Vec::with_capacity(p.n_layer);
        for il in 0..p.n_layer {
            layers.push(LayerW::load(f, &p, il)?);
        }

        let ple = if p.n_ple_heads() > 0 {
            let tr = f
                .tensor_ref(PLE_TABLE_NAME)
                .ok_or_else(|| one(format!("{PLE_TABLE_NAME}: テンソルが無い")))?;
            let table = PleTable::open(&tr, f).map_err(|e| one(format!("{PLE_TABLE_NAME}: {e}")))?;
            let ngram = PleNgram::from_meta(f).map_err(|e| one(format!("PLE 索引: {e}")))?;
            Some(std::sync::Arc::new(PleRuntime { table, ngram }))
        } else {
            None
        };

        // PLE の conv 履歴スロット (層ごと)。参照実装の `hist = (kern-1)*dil`。
        let hist_len = p.ple_conv_kernel.saturating_sub(1) * p.ple_ngram_size;
        let hc_dim = p.n_hc * p.n_embd;
        let ple_conv_state: Vec<Option<Vec<f32>>> = (0..p.n_layer)
            .map(|il| {
                if p.ple_layers.contains(&il) {
                    Some(vec![0.0f32; hist_len * hc_dim])
                } else {
                    None
                }
            })
            .collect();

        // P-MOE-G1: expert キャッシュ (既定 OFF。`DENNO_EXPERT_CACHE_MB` > 0 で構築)。
        let expert_cache = build_expert_caches(f, &layers)?;
        // task 862: page hint (`DENNO_EXPERT_PAGE_HINT`) が fadvise を投げられる
        // ように、expert テンソルの写像 → 読み口を登録する (既定 off でも登録だけ
        // する。開くのはパート数ぶんだけで、モデルの寿命いっぱい使い回す)。
        register_expert_maps(f, &layers);
        // task 260/263: ホスト RAM tier (`DENNO_EXPERT_RAM_MB`)。**作成と充填は
        // 背景の起動プラン**が行う (ここで確保すると `/v1/models` が遅れる)。
        // ここでは診断用の mmap 範囲を覚えるだけ。
        expert_ram_prepare(f, &layers)?;

        Ok(Qwen4Exp {
            p,
            layers,
            head,
            tok_embd,
            ple,
            ple_conv_state,
            attn,
            expert_cache,
            arch_emit: ArchEmit::default(),
        })
    }

    /// task 862: prefill の終端で、冷たい expert 範囲のページを落とし、
    /// decode の温かい集合のうち prefill に追い出されたものを読み直す。
    ///
    /// 中身は [`page_hint_apply`] (テストが同じ経路を単層で回せるように分けてある)。
    pub(crate) fn pf_page_hint(&self) {
        page_hint_apply(&self.layers);
    }

    /// PLE の conv 履歴を消す (系列の切れ目で呼ぶ)。
    pub fn reset_ple_conv(&mut self) {
        for b in self.ple_conv_state.iter_mut().flatten() {
            b.fill(0.0);
        }
    }

    /// P118: 系列状態を全て消して新しい系列の先頭に戻す。
    ///
    /// リセットするのは per-sequence 状態だけ ([`Self::reset_ple_conv`] の
    /// PLE conv 履歴、attention 実装の状態 = full attention の KV / QSA / ssm の
    /// GDN conv+ssm)。重み・expert キャッシュ・PLE の表と索引 (索引は状態を持たない)
    /// は触らない。テレメトリの間引き状態 ([`ArchEmit`]) も既定に戻す
    /// (数値経路には触れないが、「新規構築直後と同じ」を観測側まで揃える)。
    ///
    /// 数値不変の根拠: forward が読む状態はすべて「積んだ位置」に限定されており
    /// (KV は `push` が同じ位置に上書きし、読みは層ごとの長さまで。GDN conv/ssm と
    /// PLE conv は 0 初期化と同じ値へ戻る)、reset 後の長さは新規構築直後と同じ 0。
    pub fn reset(&mut self) {
        self.reset_ple_conv();
        self.attn.reset();
        self.arch_emit = ArchEmit::default();
    }

    /// task 206: 系列状態の差し替えに対応しているか ([`LayerAttn::seq_state_supported`])。
    pub fn seq_state_supported(&self) -> bool {
        self.attn.seq_state_supported()
    }

    /// task 206: 系列状態 (PLE conv + attention) を move で取り出し、
    /// このモデルを新しい系列の先頭に戻す。
    ///
    /// 取り出しはコピーしない (KV は `Vec` ごと移す)。attention 側が非対応
    /// (QSA 有効など) なら `None` を返し、状態は変えない。
    pub fn seq_state_take(&mut self) -> Option<ModelSeqState> {
        let attn = self.attn.seq_state_take()?;
        // 空の PLE conv 履歴を置いて、今の中身を移す。
        let fresh: Vec<Option<Vec<f32>>> = self
            .ple_conv_state
            .iter()
            .map(|b| b.as_ref().map(|v| vec![0.0; v.len()]))
            .collect();
        let ple = std::mem::replace(&mut self.ple_conv_state, fresh);
        Some(ModelSeqState { ple, attn })
    }

    /// task 206: [`Self::seq_state_take`] で取り出した状態へ戻す。
    /// 今の (空の) 状態は捨てる。形状が違えば `Err` (状態は変えない)。
    pub fn seq_state_put(&mut self, st: ModelSeqState) -> Result<()> {
        if st.ple.len() != self.ple_conv_state.len() {
            return Err(one(format!(
                "seq_state_put: PLE conv の層数 {} が現在の層数 {} と違う",
                st.ple.len(),
                self.ple_conv_state.len()
            )));
        }
        for (i, (dst, src)) in self.ple_conv_state.iter().zip(st.ple.iter()).enumerate() {
            match (dst, src) {
                (Some(d), Some(s)) if d.len() == s.len() => {}
                (None, None) => {}
                (Some(d), Some(s)) => {
                    return Err(one(format!(
                        "seq_state_put: 層 {i} の PLE conv 長 {} が {} と違う",
                        s.len(),
                        d.len()
                    )))
                }
                _ => {
                    return Err(one(format!(
                        "seq_state_put: 層 {i} の PLE conv の有無が現在のモデルと違う"
                    )))
                }
            }
        }
        // 先に attention 側を戻す (失敗したら PLE を触らない)。
        self.attn.seq_state_put(st.attn)?;
        self.ple_conv_state = st.ple;
        Ok(())
    }

    /// 諸元（読み取り専用）。
    pub fn params(&self) -> &Qwen4ExpParams {
        &self.p
    }

    /// 語彙数。
    pub fn n_vocab(&self) -> usize {
        self.p.n_vocab
    }

    /// 層数。
    pub fn n_layer(&self) -> usize {
        self.p.n_layer
    }

    /// hyper-connection の形状パラメータ。
    pub fn hp(&self) -> HyperParams {
        HyperParams {
            hc: self.p.n_hc,
            n_embd: self.p.n_embd,
            low_rank: self.p.hc_low_rank,
            eps: self.p.rms_eps,
        }
    }

    /// 1 トークンぶんの logits (`[n_vocab]`)。
    ///
    /// `ctx` は **新しい順** ではなく **古い順**で、末尾が現在のトークン
    /// (`ple.ngram` の窓は末尾 3 個 = 現在 + 直前 2)。PLE が無いモデルでは
    /// 末尾しか使わない。
    pub fn forward_token(&mut self, ctx: &[u32]) -> Result<Vec<f32>> {
        self.forward_token_at(ctx, 0, 1, false, None, None)
    }

    /// [`Qwen4Exp::forward_token`] の層ごとフック付き版。
    ///
    /// `on_layer(il, &res_hc)` は各層の **出口** (ffn の `hc_combine` の後) で
    /// 呼ばれる。実モデルのスモークで `||res_hc||` を出すために使う。
    pub fn forward_token_with<F>(&mut self, ctx: &[u32], mut on_layer: F) -> Result<Vec<f32>>
    where
        F: FnMut(usize, &[f32]) -> Result<()>,
    {
        let mut hook = |o: &LayerOut<'_>| -> Result<()> { on_layer(o.il, o.res) };
        self.forward_token_at(ctx, 0, 1, false, Some(&mut hook), None)
    }

    /// P53: 層の出口観測フック付きの 1 トークン forward。
    ///
    /// `on_layer` は層ごとに 1 回、**層の出口 (ffn の `hc_combine` の後)** で
    /// 呼ばれる。戻り値を持たないので、観測側は数値経路に影響を与えられない
    /// (読むだけ)。`t` / `nt` は (0, 1)。
    pub fn forward_token_tapped<F>(&mut self, ctx: &[u32], on_layer: F) -> Result<Vec<f32>>
    where
        F: FnMut(&LayerOut<'_>),
    {
        self.forward_token_tapped_roles(ctx, on_layer, None)
    }

    /// P54: 役割別 RMS も取る [`Qwen4Exp::forward_token_tapped`]。
    ///
    /// `roles` が `Some` のときだけ層の中で RMS を取り、[`LayerOut::roles`] に載せる。
    /// 読むだけなので logits は [`Qwen4Exp::forward_token`] とビット一致する。
    pub fn forward_token_tapped_roles<F>(
        &mut self,
        ctx: &[u32],
        mut on_layer: F,
        roles: Option<&mut RoleRms>,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(&LayerOut<'_>),
    {
        let mut hook = |o: &LayerOut<'_>| -> Result<()> {
            on_layer(o);
            Ok(())
        };
        self.forward_token_at(ctx, 0, 1, false, Some(&mut hook), roles)
    }

    /// head (`output`) の GEMV を 1 回だけ回し、`DENNO_HEAD_PROF` の区間を積む。
    ///
    /// decode は 1 トークンで 1 回だけ、prefill はチャンク末尾で 1 回だけ通る。
    /// `DENNO_HEAD_PAR=0` のときだけ従来の [`GgufLinear::gemv_label`] (汎用の門番
    /// = 512 行)。既定は [`head_row_par_min`] の行数で**語彙方向を rayon に割る**。
    /// どの行をどのスレッドが担当しても 1 行の計算順序は変わらない (同じ Q8_0 の
    /// 融合カーネル・行内は `b` 昇順) ので、分割の有無で値はビット一致する。
    pub(crate) fn head_logits_into(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let t = headprof::stamp();
        let bytes = self.head.output.weight_bytes() as u64;
        // task 187: `DENNO_DEC_HEAD_FAST=1` のときだけ v5 カーネル (ベクタロード +
        // 4 アキュムレータ) で回す。v2 とは丸め順が違う (ビット不一致)。
        // 使えない形 (型・PTX) のときは下の従来経路へ落ちる。
        #[cfg(feature = "cuda")]
        {
            let o = &self.head.output;
            if crate::cuda::head_fast_enabled()
                && crate::cuda::maybe_gpu_head_gemv_v5(
                    o.uid(),
                    o.ty(),
                    o.blocks().as_slice(),
                    o.k(),
                    o.n(),
                    x,
                    out,
                )
            {
                headprof::mark(headprof::GEMV, t, bytes);
                return Ok(());
            }
        }
        let row_min = head_row_par_min();
        let r = if row_min == 0 {
            self.head.output.gemv_label("output", x, out)
        } else {
            self.head.output.gemv_label_par("output", x, out, row_min)
        };
        headprof::mark(headprof::GEMV, t, bytes);
        r.map_err(|err| one(format!("output: {err}")))
    }

    /// [`Qwen4Exp::forward_token_tapped`] の本体。`on_layer` が `None` のときは
    /// 層ループに `if let` が 1 個増えるだけで、走査も確保もしない
    /// (`qwen4exp_decode` の `step_inner` からも使う)。
    pub(crate) fn forward_token_at(
        &mut self,
        ctx: &[u32],
        t: usize,
        nt: usize,
        is_prefill: bool,
        mut on_layer: Option<&mut dyn FnMut(&LayerOut<'_>) -> Result<()>>,
        mut roles: Option<&mut RoleRms>,
    ) -> Result<Vec<f32>> {
        let hp = self.hp();
        // task 217: 介入設定は **forward step の先頭で 1 回だけ**読む (ロックはここだけ)。
        // 以降の層ループは `&iv` を参照するだけで、ホットパスにロックは無い。
        let iv_cfg = intervene::snapshot();
        let iv = IvView::of(&iv_cfg);
        // task 218: 残差の平均の記録 / ステアリング。適用の設定も step の先頭で
        // 1 回だけ読む (層ループの中では Arc を参照するだけ)。
        // task 957: 未設定ならロックを取らない (`apply_on` の原子的読み 1 回だけ)。
        let steer_state = steer::active();
        let steer_apply = steer_state.as_ref().and_then(|s| s.apply.as_ref());
        let steer_rec = steer::rec_on();
        // P78: 逐次 prefill の中だけ層の外を区間計測する (decode では false)。
        let prof = prekprof::measuring();
        // task 132: decode の層の外の内訳 (`DENNO_Q4_PROFILE`)。無効なら
        // `dstamp` は None を返すだけで、区間は積まれない。
        let t_emb = prekprof::stamp_when(prof);
        let d_emb = q4prof::dstamp();
        let mut res = self.forward_embed(ctx, &hp)?;
        prekprof::mark(t_emb, prekprof::EMBED);
        q4prof::dmark(q4prof::D_EMBED, d_emb);
        let t_ple = prekprof::stamp_when(prof);
        let d_ple = q4prof::dstamp();
        let f0 = if prof { prekprof::faults() } else { None };
        let ple_emb = self.gather_ple(ctx)?;
        prekprof::mark(t_ple, prekprof::PLE);
        q4prof::dmark(q4prof::D_PLE, d_ple);
        if prof {
            prekprof::note_fault_delta(f0);
        }

        // P58: arch テレメトリが有効なら、呼び出し側が roles を渡していなくても
        // 役割 RMS を取る (RoleAct を送るため)。取るのは読むだけなので logits は
        // 変わらない。off のときは従来どおり `roles = None` のまま (走査もしない)。
        let mut auto_roles = RoleRms::default();
        let auto = roles.is_none() && arch_frames_enabled();

        let t_lay = prekprof::stamp_when(prof);
        let d_lay = q4prof::dstamp();
        // task 128: GPU scan 用の decode 窓 (env に依存しない。SsmLayer::forward_token
        // がこれを見て decode だけ GPU 経路に入る)。`nt == 1` = decode 1 トークン
        // (逐次 prefill の末尾 1 トークンもここに入るが、値は CPU 経路と一致する
        // 契約なので問題にしない)。off ではガードの drop が 1 回走るだけ。
        let _gw = gdn_gpu_win::window(nt == 1);
        // task 266: decode (nt == 1) のトークン先頭で、前のトークンの選択に基づく
        // 先読みを背景スレッドへ積む (`DENNO_DEC_PREF=1` のときだけ)。
        if nt == 1 {
            self.decode_prefetch_token_start();
        }
        for il in 0..self.p.n_layer {
            // P54: 役割別 RMS は層ごとに 0 から取り直す (観測しないときは None)。
            // task 155: 層の入口の下ごしらえを台帳の `loop` 行に分ける。
            let mut rr: Option<&mut RoleRms>;
            {
                let _l = q4ledger::enter(q4ledger::LOOP);
                rr = if auto {
                    Some(&mut auto_roles)
                } else {
                    roles.as_deref_mut()
                };
                if let Some(d) = rr.as_deref_mut() {
                    *d = RoleRms::default();
                }
            }
            // P56: RouterTopK はこの呼び出しの最終トークンだけ (`t + 1 == nt`)。
            // decode は (0, 1) なので常に真、逐次 prefill は末尾の 1 トークンだけ。
            // P58 の RoleAct も同じ規則。
            // task 218: ステアリングは層の入口 (層 `il-1` の出口と同じ点) に掛け、
            // 記録は層の出口で取る。task 957: `covers` を先に見て、範囲外の層では
            // `apply_one` を呼ばない (規則の走査もしない)。
            if let Some(a) = steer_apply {
                if a.covers(il) {
                    a.apply_one(il, &mut res.data);
                }
            }
            self.run_layer(
                il,
                &mut res,
                ple_emb.as_ref(),
                &hp,
                // task 155: 中間値を見るフックは decode では誰も登録しない
                // (層の観測は `on_layer` 側)。呼び出しごと飛ばす。
                None,
                rr,
                t + 1 == nt,
                iv,
            )?;
            if steer_rec {
                // task 829: prefill は位置付きで渡す (LastPromptToken は最終トークンだけ)。
                // decode (`is_prefill = false`) は acc_host 側で LastPromptToken のとき捨てる。
                if is_prefill {
                    steer::acc_host_prefill(il, &res.data, t, nt);
                } else {
                    steer::acc_host(il, &res.data);
                }
            }
            if let Some(h) = on_layer.as_deref_mut() {
                // task 155: 層の出口 (`LayerOut` の組み立てと `on_layer`) を `loop` 行に。
                let _l = q4ledger::enter(q4ledger::LOOP);
                h(&LayerOut {
                    il,
                    t,
                    nt,
                    n_embd: self.p.n_embd,
                    res: &res.data,
                    roles: roles.as_deref(),
                })?;
            }
        }

        // 4. 最終 mixer (output_norm の代わり) + 5. logits
        prekprof::mark(t_lay, prekprof::LAYER);
        q4prof::dmark(q4prof::D_LAYER, d_lay);
        let t_mix = headprof::stamp();
        let t_fin = prekprof::stamp_when(prof);
        let d_fin = q4prof::dstamp();
        let cur = {
            // task 138: 台帳の区間 (最終 mixer の mix)。
            let _l = q4ledger::enter(q4ledger::MIX);
            hc_mix_quant(
                &res,
                &self.head.norm,
                &self.head.down,
                &self.head.up,
                None,
                &hp,
            )?
            .0
        };
        headprof::mark(headprof::MIX, t_mix, 0);
        prekprof::mark(t_fin, prekprof::FINAL_NORM);
        q4prof::dmark(q4prof::D_FINAL, d_fin);
        let mut logits = vec![0.0f32; self.head.output.n()];
        let t_head = prekprof::stamp_when(prof);
        let d_head = q4prof::dstamp();
        {
            // task 153: 台帳の区間 (head)。最終 head GEMV と logits の D2H。
            let _l = q4ledger::enter(q4ledger::HEAD);
            self.head_logits_into(&cur, &mut logits)?;
        }
        prekprof::mark(t_head, prekprof::HEAD);
        q4prof::dmark(q4prof::D_HEAD, d_head);
        Ok(logits)
    }

    /// 1. `emb = tok_embd[token]` と 2. `res_hc = [emb; ...; emb]`。
    pub(crate) fn forward_embed(&self, ctx: &[u32], hp: &HyperParams) -> Result<HyperStreams> {
        let token = match ctx.last() {
            Some(t) => *t,
            None => return Err(one("ctx が空 (現在のトークンが無い)")),
        };
        let e = self.p.n_embd;
        let _sub = q4ledger::sub(q4ledger::SUB_EMBED);
        let emb = self
            .tok_embd
            .dequant_row(token as usize)
            .map_err(|err| one(format!("token_embd: 行 {token}: {err}")))?;
        if emb.len() != e {
            return Err(one(format!(
                "token_embd: 行の長さ {} が n_embd {e} と違う",
                emb.len()
            )));
        }
        Ok(HyperStreams::from_embed(&emb, hp))
    }

    /// PLE の行 gather (1 トークンにつき 1 回、全層で共有)。
    ///
    /// `PleNgram::indices` は **新しい順** (`ctx[0]` = 現在のトークン) を要求する。
    /// 呼び出し側の `ctx` は古い順なので、末尾から n_gram 個を反転して渡す。
    /// ここを取り違えると n-gram が 2 つずれた窓で引かれる (参照実装の
    /// `ctx[0] = 現在, ctx[s] = s 個前` に合わせる)。
    pub(crate) fn gather_ple(&self, ctx: &[u32]) -> Result<Option<Vec<f32>>> {
        match &self.ple {
            Some(rt) => {
                let win = ple_window(ctx, rt.ngram.n_gram());
                Ok(Some(
                    rt.ngram
                        .gather(&rt.table, &win)
                        .map_err(|err| one(format!("PLE 索引: {err}")))?,
                ))
            }
            None => Ok(None),
        }
    }

    /// チャンクぶんの PLE の行 gather をまとめて行う (P78)。
    ///
    /// `ctx[..]` は古い順の履歴で、`start..ctx.len()` の各トークンの窓を作る。
    /// 戻り値は長さ `ctx.len() - start` で、PLE を持たないモデルでは全部 `None`。
    ///
    /// 窓ごとに [`Qwen4Exp::gather_ple`] を呼ぶのと値はビット一致する
    /// ([`PleNgram::gather_many`] は I/O の順序だけを変える)。1 チャンクで
    /// `トークン数 × 16` 本のランダムな 90 B pread をまとめて出すので、
    /// 1 トークンずつ待つよりキューを深く使える。
    pub(crate) fn gather_ple_many(&self, ctx: &[u32], start: usize) -> Result<Vec<Option<Vec<f32>>>> {
        gather_ple_many_rt(self.ple.as_deref(), ctx, start)
    }

    /// task 266: decode のトークン先頭で、**1 トークン前の選択**のうち VRAM に
    /// 載っていない expert のページを背景スレッドへ先読みさせる。
    ///
    /// `DENNO_DEC_PREF=1` のときだけ動く (既定 off)。予測が外れても捨てるだけで、
    /// 値・経路は 1 バイトも変わらない。VRAM に載っている expert は mmap を
    /// 触らないので積まない (ヒット率 0.9 のぶんを読まない)。
    #[cfg(feature = "cuda")]
    pub(crate) fn decode_prefetch_token_start(&self) {
        if !crate::model::exp_prefetch::enabled() {
            return;
        }
        let snap = crate::model::exp_prefetch::begin_token(self.p.n_layer);
        if snap.is_empty() {
            return;
        }
        // いま VRAM に載っている (層, expert) を 1 回だけ写す。
        let resident: std::collections::HashSet<(u16, u16)> =
            expert_vram::resident_keys()
                .into_iter()
                .map(|k| (k.layer, k.expert))
                .collect();
        let max = crate::model::exp_prefetch::max_per_token();
        let mut jobs: Vec<Vec<crate::model::exp_up_worker::Part>> = Vec::new();
        let mut keys: Vec<(u16, u16)> = Vec::new();
        let (mut n_res, mut n_nopart) = (0u64, 0u64);
        'layers: for (il, ids) in snap.iter().enumerate() {
            if il >= self.layers.len() {
                break;
            }
            let lw = &self.layers[il];
            for &eid in ids {
                if jobs.len() >= max {
                    break 'layers;
                }
                let key = (il as u16, eid);
                if resident.contains(&key) {
                    n_res += 1;
                    continue;
                }
                let part = |w: &ExpertW| w.slice_mapped(eid as usize).ok().flatten();
                let (Some(g), Some(u), Some(d)) = (
                    part(&lw.gate_exps),
                    part(&lw.up_exps),
                    part(&lw.down_exps),
                ) else {
                    n_nopart += 1;
                    continue;
                };
                jobs.push(vec![g, u, d]);
                keys.push(key);
            }
        }
        crate::model::exp_prefetch::note_skips(n_res, n_nopart);
        crate::model::exp_prefetch::note_submitted_keys(&keys);
        crate::model::exp_prefetch::submit(jobs);
    }

    /// cuda feature 無しでは先読みしない (mmap の Part を作る口が無い)。
    #[cfg(not(feature = "cuda"))]
    pub(crate) fn decode_prefetch_token_start(&self) {}

    /// 1 層ぶんの本体 (PLE 注入 → attention → MoE FFN)。
    ///
    /// `hook(name, il, vals)` は層の中間値の観測用。通常の forward は空のフックを
    /// 渡すので、この関数を経由しても数値は変わらない (演算順序は同じ)。
    ///
    /// `roles` が `Some` のときだけ役割別 RMS (P54) を書く。**読むだけ**なので
    /// 数値経路は変わらない。`None` のときは分岐 1 個ぶんのコストも無い。
    ///
    /// `emit_router` は RouterTopK (P56) を送るか。呼び出し側が「このトークンが
    /// prefill のチャンク末尾か」を知っているので、そこで決める (decode は常に
    /// true、逐次 prefill は末尾の 1 トークンだけ true)。役割 RMS と RouterTopK は
    /// どちらも読むだけの観測で、同じ本体 ([`moe_ffn_quant_full`]) を通る。
    pub(crate) fn run_layer(
        &mut self,
        il: usize,
        res: &mut HyperStreams,
        ple_emb: Option<&Vec<f32>>,
        hp: &HyperParams,
        hook: Option<&mut dyn FnMut(&str, usize, &[f32]) -> Result<()>>,
        mut roles: Option<&mut RoleRms>,
        emit_router: bool,
        iv: IvView<'_>,
    ) -> Result<()> {
        let e = self.p.n_embd;
        // task 177: 診断 (`DENNO_GDN_DEC_KPROF`) が mix の表に層番号を出すため。
        #[cfg(feature = "cuda")]
        crate::cuda::dec_kprof::set_layer(il);
        // task 307: 層 0 の入口 = トークンの境目。前のトークンのルータの控えを落とす
        // (`DENNO_ROUTER_IN_MIX` の印が次のトークンへ漏れないように)。
        #[cfg(feature = "cuda")]
        if il == 0 {
            crate::cuda::router_in_mix_token_start();
        }
        let lw = &self.layers[il];
        // task 155: 購読者が居ないフックは**呼び出しごと**飛ばす。`None` は
        // 「このランでは誰も層の中間値を見ない」= decode の通常経路。
        let mut hook = hook;
        macro_rules! hook_at {
            ($name:expr) => {
                if let Some(h) = hook.as_mut() {
                    let _l = q4ledger::enter(q4ledger::HOOKS);
                    h($name, il, &res.data)?;
                }
            };
            ($name:expr, $vals:expr) => {
                if let Some(h) = hook.as_mut() {
                    let _l = q4ledger::enter(q4ledger::HOOKS);
                    h($name, il, $vals)?;
                }
            };
        }

        if q4prof::enabled() {
            q4prof::N_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // task 217: `ple` block を切った層では注入そのものを計算しない
        // (残差の置き換えもしない)。
        if let Some(pw) = &lw.ple {
            if !iv.block_off(il, IvBlock::Ple) {
                let pe = ple_emb.ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?;
                let hist_buf = match self.ple_conv_state[il].as_mut() {
                    Some(b) => b,
                    None => return Err(one(format!("層 {il}: PLE の conv 履歴が無い"))),
                };
                // P54: PLE の寄与 (gated + conv) の RMS を ple_inject の中で 1 パスで取る。
                let mut contrib = 0.0f32;
                *res = ple_inject(
                    &self.p,
                    pw,
                    pe,
                    res,
                    hp,
                    hist_buf,
                    roles.is_some().then_some(&mut contrib),
                )?;
                if let Some(r) = roles.as_deref_mut() {
                    r.ple = contrib;
                }
                hook_at!("ple_after");
            }
        }

        // task 217: `attn` block を切った層は mix / attn / combine を丸ごと飛ばす。
        if !iv.block_off(il, IvBlock::Attn) {
            let t_mix = std::time::Instant::now();
            // task 464: ssm 層の attn mix は出力をホストで読まない (GDN が
            // デバイスの `dm` をそのまま読む) ので、鎖の D2H と待ちを省いて
            // 直後の GDN submission の同期へまとめる (`DENNO_MIX_NOWAIT`)。
            // 前提が崩れたら ssm 側が空入力を検出して要求を落とす。
            #[cfg(feature = "cuda")]
            let mix_defer = crate::cuda::mix_nowait_on()
                && crate::cuda::gdn_x_mix_on()
                && gdn_gpu_win::active()
                && hook.is_none()
                && match self.p.layer_kind(il) {
                    // ssm 層: ba を GPU で作るモデルだけ defer できる
                    // (ホスト ba は cur が要る)。
                    Ok(LayerKind::Ssm) => {
                        crate::cuda::gdn_gpu_scan_on() && crate::cuda::gdn_ba_gpu_on()
                    }
                    // task 468: full 層は attention の device 経路が鎖の `mixed` を
                    // 直接読める回だけ (`DENNO_MIX_NOWAIT=full`)。
                    Ok(LayerKind::FullAttn) => self.attn.takes_dev_input(il),
                    _ => false,
                };
            #[cfg(not(feature = "cuda"))]
            let mix_defer = false;
            let (cur, inject) = {
                // task 138: 台帳の区間 (mix)。off では stamp を取らないだけ。
                let _l = q4ledger::enter(q4ledger::MIX);
                if mix_defer {
                    hc_mix_quant_defer(
                        res,
                        &lw.hc_attn_norm,
                        &lw.hc_attn_down,
                        &lw.hc_attn_up,
                        Some(&lw.hc_attn_inject),
                        hp,
                    )?
                } else {
                    hc_mix_quant(
                        res,
                        &lw.hc_attn_norm,
                        &lw.hc_attn_down,
                        &lw.hc_attn_up,
                        Some(&lw.hc_attn_inject),
                        hp,
                    )?
                }
            };
            {
                let _l = q4ledger::enter(q4ledger::QPROF);
                q4prof::add(&q4prof::MIX, t_mix);
            }
            hook_at!("hc_mixed_attn", &cur);
            hook_at!("hc_inject_attn", &inject);
            let t_attn = std::time::Instant::now();
            let cur = {
                // task 138: 層種別で full / ssm に分ける (台帳 on のときだけ引く)。
                let item = if q4ledger::enabled() {
                    match self.p.layer_kind(il) {
                        Ok(LayerKind::FullAttn) => q4ledger::FULL,
                        _ => q4ledger::SSM,
                    }
                } else {
                    q4ledger::OTHER
                };
                let _l = q4ledger::enter(item);
                self.attn.forward(il, &cur)?
            };
            {
                let _l = q4ledger::enter(q4ledger::QPROF);
                q4prof::add(&q4prof::ATTN, t_attn);
            }
            if cur.len() != e {
                return Err(one(format!(
                    "層 {il}: attention の出力長 {} が n_embd {e} と違う",
                    cur.len()
                )));
            }
            hook_at!("attn_out", &cur);
            // P54: 層の種別で attn (full attention) と gdn (ssm) のどちらに入れるかが決まる。
            // task 153: 台帳の区間 (roles)。2560 要素の RMS と役割の詰め直し。
            if roles.is_some() {
                let _l = q4ledger::enter(q4ledger::ROLES);
                if let Some(r) = roles.as_deref_mut() {
                    let v = rms_of(&cur);
                    if matches!(self.p.layer_kind(il), Ok(LayerKind::FullAttn)) {
                        r.attn = v;
                    } else {
                        r.gdn = v;
                    }
                }
            }
            {
                // task 153: 台帳の区間 (combine)。hyper connection の足し込み。
                // task 155: 作業域を使い回す (`hc_combine` と同じ式・同じ順)。
                // task 217: 無効にしたストリームにはこの層の更新を足さない。
                let _l = q4ledger::enter(q4ledger::COMBINE);
                hyper::hc_combine_inplace_masked(
                    res,
                    &cur,
                    &inject,
                    hp,
                    iv.stream_keep(il, hp.hc),
                );
            }
            hook_at!("res_after_attn");
        }

        // task 217: routed expert と shared expert の両方を切った層は mix / combine も
        // 飛ばす。片方だけ切るときは mix が要る (切っていない側の入力になる)。
        let skip_routed = iv.block_off(il, IvBlock::Moe);
        let skip_shared = iv.block_off(il, IvBlock::Shared);
        if !(skip_routed && skip_shared) {
            let t_mix = std::time::Instant::now();
            // task 307: FFN mix の鎖 (down → up → hc_mixed) の末尾にルータ GEMV を
            // 載せる (`DENNO_ROUTER_IN_MIX`・既定 on。lab t307c で採否)。載れば
            // router の H2D と
            // 同期 1 回が消える。載らなければ従来どおり `gate_inp` の GEMV が
            // H2D + 起動 + D2H + 同期を行う (数値は同じ)。
            #[cfg(feature = "cuda")]
            if gdn_gpu_win::active() && lw.gate_inp.router_gpu_ok() {
                let _ = crate::cuda::router_in_mix_arm(
                    lw.gate_inp.uid(),
                    lw.gate_inp.blocks().as_slice(),
                    lw.gate_inp.k(),
                    lw.gate_inp.n(),
                );
            }
            let (cur, inject) = {
                // task 138: 台帳の区間 (mix)。
                let _l = q4ledger::enter(q4ledger::MIX);
                hc_mix_quant(
                    res,
                    &lw.hc_ffn_norm,
                    &lw.hc_ffn_down,
                    &lw.hc_ffn_up,
                    Some(&lw.hc_ffn_inject),
                    hp,
                )?
            };
            {
                let _l = q4ledger::enter(q4ledger::QPROF);
                q4prof::add(&q4prof::MIX, t_mix);
            }
            hook_at!("hc_mixed_ffn", &cur);
            let t_moe = std::time::Instant::now();
            // task 850: ルータ直後に層 L+1 の予測 expert を先読みさせるため、
            // 次の層の重みを TLS に張ってから MoE を呼ぶ (ゲート off では素通し)。
            let cur = with_markov_next(self.layers.get(il + 1), || {
                moe_ffn_quant_tapped(
                    &self.p,
                    lw,
                    &cur,
                    self.expert_cache.as_ref(),
                    roles.as_deref_mut(),
                    emit_router,
                    iv,
                )
            })?;
            {
                let _l = q4ledger::enter(q4ledger::QPROF);
                q4prof::add(&q4prof::MOE, t_moe);
            }
            hook_at!("ffn_out", &cur);
            {
                // task 153: 台帳の区間 (combine)。
                // task 155: 作業域を使い回す (`hc_combine` と同じ式・同じ順)。
                let _l = q4ledger::enter(q4ledger::COMBINE);
                hyper::hc_combine_inplace_masked(
                    res,
                    &cur,
                    &inject,
                    hp,
                    iv.stream_keep(il, hp.hc),
                );
            }
        }

        // P58: 層の役割別活性 (kind 13)。`emit_router` は「このトークンが forward
        // 呼び出しの末尾か」の意味も持つ (decode は常に真、逐次 prefill は末尾の
        // 1 トークンだけ)。読むだけなので数値経路には触れない。
        if emit_router {
            if roles.as_deref().is_some() {
                // task 153: 台帳の区間 (roles)。RoleAct フレームの組み立てと送出。
                let _l = q4ledger::enter(q4ledger::ROLES);
                if let Some(r) = roles.as_deref() {
                    emit_role_act(il, r);
                }
            }
        }
        Ok(())
    }

    /// デバッグ用: `ctx` を 1 個ずつ食わせ、最後のトークンの `layer` 層の中間値を
    /// `sink(name, vals)` に渡す。**数値は通常の forward と同じ** (同じ関数を通る)。
    pub fn debug_layer(
        &mut self,
        ctx: &[u32],
        layer: usize,
        sink: &mut dyn FnMut(&str, &[f32]),
    ) -> Result<()> {
        let hp = self.hp();
        // task 217: デバッグ経路も同じ介入設定を通す (数値は本番と同じ、の契約)。
        let iv_cfg = intervene::snapshot();
        let iv = IvView::of(&iv_cfg);
        let last = ctx.len().saturating_sub(1);
        for i in 0..ctx.len() {
            let is_last = i == last;
            let mut res = self.forward_embed(&ctx[..=i], &hp)?;
            let ple_emb = self.gather_ple(&ctx[..=i])?;
            for il in 0..self.p.n_layer {
                let mut hook = |name: &str, hil: usize, vals: &[f32]| {
                    if is_last && hil == layer {
                        sink(name, vals);
                    }
                    Ok(())
                };
                self.run_layer(
                    il,
                    &mut res,
                    ple_emb.as_ref(),
                    &hp,
                    Some(&mut hook),
                    None,
                    false,
                    iv,
                )?;
            }
        }
        Ok(())
    }

    /// P109: spec decode の巻き戻しに対応しているか (attention 実装に委譲)。
    pub fn spec_restorable(&self) -> bool {
        self.attn.spec_restorable()
    }

    /// P109: PLE conv 履歴 + attention 状態のスナップショットを取る。
    pub fn spec_snapshot(&self) -> SpecSnapshot {
        SpecSnapshot {
            ple: self.ple_conv_state.clone(),
            attn: self.attn.spec_snapshot(),
        }
    }

    /// P109: [`Qwen4Exp::spec_snapshot`] の状態へ戻す (検証列の棄却時)。
    pub fn spec_restore(&mut self, snap: &SpecSnapshot) -> Result<()> {
        self.ple_conv_state = snap.ple.clone();
        self.attn
            .spec_restore(&*snap.attn)
            .map_err(|e| one(format!("spec_restore: {e}")))
    }
}

/// task 668: 分割前は 1 ファイルだった。ソース文字列を見るテストのために、
/// 分割後の全ファイルを固定順で連結して返す (assert は弱めない)。
#[cfg(test)]
pub(crate) fn src_text() -> &'static str {
    concat!(
        include_str!("mod.rs"),
        include_str!("prefill.rs"),
        include_str!("mix.rs"),
        include_str!("moe.rs"),
        include_str!("prof.rs"),
    )
}
