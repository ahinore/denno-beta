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
use crate::model::expert_cache::{self, CacheSpec, ExpertCacheSet, ExpertKey, ExpertMatrix};
use crate::model::hyper::{self, HyperParams, HyperStreams};
use crate::model::linear_gguf::GgufLinear;
use crate::model::moe::{select_topk, select_topk_into};
use crate::model::moe_expert::{self, ExpertSlice};
use crate::ple::{PleNgram, PleTable, PLE_TABLE_NAME};
use crate::quant::WeightBlocks;
use crate::tensor::{silu_inplace, softmax};

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
    if t.ty != e.ty && !e.ty_alt.contains(&t.ty) {
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
        Ok(p)
    } else {
        Err(Qwen4ExpError(errs))
    }
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
    GgufLinear::new(t.ty, k, n, blocks)
        .map(|l| l.with_fast_q8(true))
        .map_err(|e| one(format!("{name}: {e}")))
}

/// F32 テンソルを f32 の Vec として読む (norm の gamma・conv の重み用)。
fn f32_tensor(f: &GgufFile, name: &str) -> Result<Vec<f32>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| one(format!("{name}: テンソルが無い")))?;
    if t.ty != GgmlType::F32 {
        return Err(one(format!(
            "{name}: 型 {} が期待 F32 と違う",
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
        Ok(w)
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
    fn prefetch(&self, e: usize) {
        if let Ok(s) = ExpertSlice::new(self.ty, self.n_in, self.n_out, self.n_expert, e) {
            let b = self.blocks.as_slice();
            if s.end() <= b.len() {
                crate::model::moe_expert::prefetch_range(&b[s.off..s.end()]);
            }
        }
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
        moe_expert::expert_gemv_any_seq(self.ty, s, n_in, n_out, x, out)
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
}

impl PleW {
    fn load(f: &GgufFile, il: usize, p: &Qwen4ExpParams) -> Result<Self> {
        let name = |s: &str| format!("blk.{il}.{s}");
        Ok(PleW {
            key: linear(f, &name("ple_key.weight"))?,
            value: linear(f, &name("ple_value.weight"))?,
            norm_key: f32_tensor(f, &name("ple_norm_key.weight"))?,
            norm_query: f32_tensor(f, &name("ple_norm_query.weight"))?,
            norm_conv: f32_tensor(f, &name("ple_norm_conv.weight"))?,
            conv1d: f32_tensor(f, &name("ple_conv1d.weight"))?,
            kern: p.ple_conv_kernel,
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
            gate_inp: linear(f, &name("ffn_gate_inp.weight"))?,
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
    let set = ExpertCacheSet::from_mb(mb, [gate, up, down], &paths)
        .map_err(|err| one(format!("expert キャッシュの構築に失敗: {err}")))?;
    Ok(Some(set))
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

/// 1 トークン decode の骨格。
pub struct Qwen4Exp {
    pub p: Qwen4ExpParams,
    layers: Vec<LayerW>,
    head: HeadW,
    tok_embd: GgufLinear,
    ple: Option<PleRuntime>,
    /// PLE の conv 履歴 (層ごと)。`(kern-1) * ngram_size` トークン分の
    /// `grouped_rms_norm(gated)` を古い順に保持する。層に PLE が無ければ `None`。
    ple_conv_state: Vec<Option<Vec<f32>>>,
    pub attn: Box<dyn LayerAttn>,
    /// P-MOE-G1: `DENNO_EXPERT_CACHE_MB` > 0 のときだけ `Some`。
    /// `None` は従来どおり mmap + `prefetch()` の経路 (バイト列も挙動も変えない)。
    expert_cache: Option<ExpertCacheSet>,
}

impl Qwen4Exp {
    /// 実 GGUF から組む。**全テンソルの存在・形状・量子型を先に検査**して、
    /// 1 本でも欠けたら `Err` にする (黙って 0 で埋めない)。
    pub fn new(f: &GgufFile) -> Result<Self> {
        Self::with_attn(f, Box::new(IdentityAttn))
    }

    /// attention / ssm の実装を差し替えて組む (F2 / F3 用)。
    pub fn with_attn(f: &GgufFile, attn: Box<dyn LayerAttn>) -> Result<Self> {
        let p = validate(f)?;

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
            Some(PleRuntime { table, ngram })
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

        Ok(Qwen4Exp {
            p,
            layers,
            head,
            tok_embd,
            ple,
            ple_conv_state,
            attn,
            expert_cache,
        })
    }

    /// PLE の conv 履歴を消す (系列の切れ目で呼ぶ)。
    pub fn reset_ple_conv(&mut self) {
        for b in self.ple_conv_state.iter_mut().flatten() {
            b.fill(0.0);
        }
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
        self.forward_token_with(ctx, |_, _| Ok(()))
    }

    /// [`Qwen4Exp::forward_token`] の層ごとフック付き版。
    ///
    /// `on_layer(il, &res_hc)` は各層の **出口** (ffn の `hc_combine` の後) で
    /// 呼ばれる。実モデルのスモークで `||res_hc||` を出すために使う。
    pub fn forward_token_with<F>(&mut self, ctx: &[u32], mut on_layer: F) -> Result<Vec<f32>>
    where
        F: FnMut(usize, &[f32]) -> Result<()>,
    {
        let hp = self.hp();
        let mut res = self.forward_embed(ctx, &hp)?;
        let ple_emb = self.gather_ple(ctx)?;

        for il in 0..self.p.n_layer {
            self.run_layer(il, &mut res, ple_emb.as_ref(), &hp, &mut |_, _, _| Ok(()))?;
            on_layer(il, &res.data)?;
        }

        // 4. 最終 mixer (output_norm の代わり) + 5. logits
        let cur = hc_mix_quant(
            &res,
            &self.head.norm,
            &self.head.down,
            &self.head.up,
            None,
            &hp,
        )?
        .0;
        let mut logits = vec![0.0f32; self.head.output.n()];
        self.head
            .output
            .gemv(&cur, &mut logits)
            .map_err(|err| one(format!("output: {err}")))?;
        Ok(logits)
    }

    /// 1. `emb = tok_embd[token]` と 2. `res_hc = [emb; ...; emb]`。
    fn forward_embed(&self, ctx: &[u32], hp: &HyperParams) -> Result<HyperStreams> {
        let token = match ctx.last() {
            Some(t) => *t,
            None => return Err(one("ctx が空 (現在のトークンが無い)")),
        };
        let e = self.p.n_embd;
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
    fn gather_ple(&self, ctx: &[u32]) -> Result<Option<Vec<f32>>> {
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

    /// 1 層ぶんの本体 (PLE 注入 → attention → MoE FFN)。
    ///
    /// `hook(name, il, vals)` は層の中間値の観測用。通常の forward は空のフックを
    /// 渡すので、この関数を経由しても数値は変わらない (演算順序は同じ)。
    fn run_layer(
        &mut self,
        il: usize,
        res: &mut HyperStreams,
        ple_emb: Option<&Vec<f32>>,
        hp: &HyperParams,
        hook: &mut dyn FnMut(&str, usize, &[f32]) -> Result<()>,
    ) -> Result<()> {
        let e = self.p.n_embd;
        let lw = &self.layers[il];

        if q4prof::enabled() {
            q4prof::N_LAYER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // 3.1 PLE 注入 (層 1 のみ)。res_hc を置き換える。
        if let Some(pw) = &lw.ple {
            let pe = ple_emb.ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?;
            let hist_buf = match self.ple_conv_state[il].as_mut() {
                Some(b) => b,
                None => return Err(one(format!("層 {il}: PLE の conv 履歴が無い"))),
            };
            *res = ple_inject(&self.p, pw, pe, res, hp, hist_buf)?;
            hook("ple_after", il, &res.data)?;
        }

        // 3.2-3.4 attention 入口 → attn → 出口
        let t_mix = std::time::Instant::now();
        let (cur, inject) = hc_mix_quant(
            res,
            &lw.hc_attn_norm,
            &lw.hc_attn_down,
            &lw.hc_attn_up,
            Some(&lw.hc_attn_inject),
            hp,
        )?;
        q4prof::add(&q4prof::MIX, t_mix);
        hook("hc_mixed_attn", il, &cur)?;
        hook("hc_inject_attn", il, &inject)?;
        let t_attn = std::time::Instant::now();
        let cur = self.attn.forward(il, &cur)?;
        q4prof::add(&q4prof::ATTN, t_attn);
        if cur.len() != e {
            return Err(one(format!(
                "層 {il}: attention の出力長 {} が n_embd {e} と違う",
                cur.len()
            )));
        }
        hook("attn_out", il, &cur)?;
        *res = hyper::hc_combine(res, &cur, &inject, hp);
        hook("res_after_attn", il, &res.data)?;

        // 3.5-3.7 ffn 入口 → MoE FFN → 出口
        let t_mix = std::time::Instant::now();
        let (cur, inject) = hc_mix_quant(
            res,
            &lw.hc_ffn_norm,
            &lw.hc_ffn_down,
            &lw.hc_ffn_up,
            Some(&lw.hc_ffn_inject),
            hp,
        )?;
        q4prof::add(&q4prof::MIX, t_mix);
        hook("hc_mixed_ffn", il, &cur)?;
        let t_moe = std::time::Instant::now();
        let cur = moe_ffn_quant(&self.p, lw, &cur, self.expert_cache.as_mut())?;
        q4prof::add(&q4prof::MOE, t_moe);
        hook("ffn_out", il, &cur)?;
        *res = hyper::hc_combine(res, &cur, &inject, hp);
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
                self.run_layer(il, &mut res, ple_emb.as_ref(), &hp, &mut hook)?;
            }
        }
        Ok(())
    }

    /// プロンプト全体を 1 回で処理して**最後のトークンの logits** を返す。
    ///
    /// `DENNO_PREFILL_BATCH=1` のときだけ層優先のバッチ経路を使う (既定 off)。
    /// off のとき、または attention が層優先の順序に耐えない実装のときは、
    /// decoder と同じ「1 トークンずつ [`Qwen4Exp::forward_token`]」に落ちる。
    /// どちらの経路も出力はビット一致する (バッチ経路は行列積を
    /// [`gemv_batch`] = 行優先・同一カーネルに置き換えるだけ)。
    pub fn forward_prefill(&mut self, ctx: &[u32]) -> Result<Vec<f32>> {
        self.forward_prefill_with(ctx, prefill_batch_enabled())
    }

    /// [`Qwen4Exp::forward_prefill`] の切替を引数で受ける版 (テスト用)。
    ///
    /// `batch == false` は既存の逐次経路そのもの (1 ビットも変えない)。
    pub fn forward_prefill_with(&mut self, ctx: &[u32], batch: bool) -> Result<Vec<f32>> {
        if batch && self.attn.layer_major_safe() {
            self.forward_prefill_batch(ctx, prefill_batch_chunk())
        } else {
            self.prefill_seq(ctx)
        }
    }

    /// 逐次 prefill (= decoder と同じ経路)。既存の [`Qwen4Exp::forward_token`] を
    /// トークン数だけ呼ぶだけで、数値経路には触らない。
    fn prefill_seq(&mut self, ctx: &[u32]) -> Result<Vec<f32>> {
        if ctx.is_empty() {
            return Err(one("ctx が空 (現在のトークンが無い)"));
        }
        let mut logits = Vec::new();
        for i in 0..ctx.len() {
            logits = self.forward_token(&ctx[..=i])?;
        }
        Ok(logits)
    }

    /// 層優先のバッチ prefill。`ctx` を `chunk` トークンずつに切り、チャンク内では
    /// 「層 0 の全トークン → 層 1 の全トークン → ...」の順に回す。
    ///
    /// チャンクを跨いで持ち越す残差は無い (各トークンの層 0 の入力は埋め込みだけで、
    /// 残差は層の中で閉じる)。attention の KV / GDN 状態は `attn` の中に
    /// **トークン順**で積まれるので、層優先でも順序は保たれる
    /// (`LayerAttn::layer_major_safe` が true の実装だけがここへ来る)。
    ///
    /// 数値: 1 トークンぶんの演算は逐次版と同じ関数・同じ順序で、行列積だけを
    /// [`gemv_batch`] に置き換える。したがって f32 でビット一致する。
    fn forward_prefill_batch(&mut self, ctx: &[u32], chunk: usize) -> Result<Vec<f32>> {
        if ctx.is_empty() {
            return Err(one("ctx が空 (現在のトークンが無い)"));
        }
        #[cfg(test)]
        PREFILL_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let hp = self.hp();
        let e = self.p.n_embd;
        let n_layer = self.p.n_layer;
        let chunk = chunk.max(1);
        let mut last: Option<HyperStreams> = None;
        let mut start = 0usize;
        // mix の要素演算の作業域 (実形で xn/gate 20 MB ずつ)。層ループの外で 1 回だけ持つ。
        let mut mx = MixScratch::default();
        while start < ctx.len() {
            let end = (start + chunk).min(ctx.len());
            let nt = end - start;
            let mut ress: Vec<HyperStreams> = Vec::with_capacity(nt);
            let mut ples: Vec<Option<Vec<f32>>> = Vec::with_capacity(nt);
            for i in start..end {
                ress.push(self.forward_embed(&ctx[..=i], &hp)?);
                ples.push(self.gather_ple(&ctx[..=i])?);
            }

            for il in 0..n_layer {
                let lw = &self.layers[il];
                if q4prof::enabled() {
                    q4prof::N_LAYER.fetch_add(nt as u64, std::sync::atomic::Ordering::Relaxed);
                }

                // 3.1 PLE 注入 (層 1 のみ)。conv 履歴はトークン順に進める。
                if let Some(pw) = &lw.ple {
                    for t in 0..nt {
                        let pe = ples[t]
                            .as_ref()
                            .ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?;
                        let hist = self.ple_conv_state[il]
                            .as_mut()
                            .ok_or_else(|| one(format!("層 {il}: PLE の conv 履歴が無い")))?;
                        ress[t] = ple_inject(&self.p, pw, pe, &ress[t], &hp, hist)?;
                    }
                }

                // 3.2-3.4 attention 入口 (バッチ) → attn (トークン順) → 出口
                let t_mix = std::time::Instant::now();
                let (curs, injs) = hc_mix_quant_batch(
                    &ress,
                    &lw.hc_attn_norm,
                    &lw.hc_attn_down,
                    &lw.hc_attn_up,
                    Some(&lw.hc_attn_inject),
                    &hp,
                    &mut mx,
                )?;
                q4prof::add(&q4prof::MIX, t_mix);
                let t_attn = std::time::Instant::now();
                // 投影をチャンクでまとめる実装 (ssm 層) はここで GPU 往復を減らす。
                let outs = self.attn.forward_batch(il, &curs)?;
                for o in &outs {
                    if o.len() != e {
                        return Err(one(format!(
                            "層 {il}: attention の出力長 {} が n_embd {e} と違う",
                            o.len()
                        )));
                    }
                }
                q4prof::add(&q4prof::ATTN, t_attn);
                if outs.len() != nt {
                    return Err(one(format!(
                        "層 {il}: attention の出力本数 {} が nt {nt} と違う",
                        outs.len()
                    )));
                }
                let flat_outs = flatten_rows(&outs, e)?;
                hyper::hc_combine_batch(&mut ress, &flat_outs, &injs, &hp);

                // 3.5-3.7 ffn 入口 (バッチ) → MoE FFN (バッチ) → 出口
                let t_mix = std::time::Instant::now();
                let (curs, injs) = hc_mix_quant_batch(
                    &ress,
                    &lw.hc_ffn_norm,
                    &lw.hc_ffn_down,
                    &lw.hc_ffn_up,
                    Some(&lw.hc_ffn_inject),
                    &hp,
                    &mut mx,
                )?;
                q4prof::add(&q4prof::MIX, t_mix);
                let flat = flatten_rows(&curs, e)?;
                let t_moe = std::time::Instant::now();
                let ffn = moe_ffn_quant_batch(&self.p, lw, &flat, nt, self.expert_cache.as_mut())?;
                q4prof::add(&q4prof::MOE, t_moe);
                if ffn.len() != nt * e {
                    return Err(one(format!(
                        "層 {il}: MoE の出力長 {} が nt {nt} * n_embd {e} と違う",
                        ffn.len()
                    )));
                }
                hyper::hc_combine_batch(&mut ress, &ffn, &injs, &hp);
            }

            // 次のチャンクは自分の埋め込みから始まる (残差の持ち越しは無い)。
            last = ress.pop();
            start = end;
        }

        // 4. 最終 mixer (output_norm の代わり) + 5. logits (最後のトークンだけ)
        let res = last.ok_or_else(|| one("ctx が空 (現在のトークンが無い)"))?;
        let cur = hc_mix_quant(
            &res,
            &self.head.norm,
            &self.head.down,
            &self.head.up,
            None,
            &hp,
        )?
        .0;
        let mut logits = vec![0.0f32; self.head.output.n()];
        self.head
            .output
            .gemv(&cur, &mut logits)
            .map_err(|err| one(format!("output: {err}")))?;
        Ok(logits)
    }
}

/// `hc_mix` を**量子化重み**で計算する (`hyper::hc_mix` と同じ丸め順序)。
///
/// `inject` が `None` なら inject は 0 で返す (head の mixer は inject を
/// 持たない: 参照実装は `w_inject` / `w_down_inject` とも `nullptr`)。
fn hc_mix_quant(
    res: &HyperStreams,
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let xn = hyper::grouped_rms_norm(res, norm, hp);

    let mut lo = vec![0.0f32; down.n()];
    down.gemv(&xn, &mut lo)
        .map_err(|err| one(format!("hc_mix down: {err}")))?;
    hyper::hc_lo_apply(&mut lo, hp);

    let mut gate = vec![0.0f32; up.n()];
    up.gemv(&lo, &mut gate)
        .map_err(|err| one(format!("hc_mix up: {err}")))?;
    let mixed = hyper::hc_mixed(&xn, &gate, hp);

    let inj = match inject {
        Some(w) => {
            let mut v = vec![0.0f32; w.n()];
            w.gemv(&xn, &mut v)
                .map_err(|err| one(format!("hc_mix inject: {err}")))?;
            hyper::hc_inject_apply(&mut v, hp);
            v
        }
        None => vec![0.0f32; hp.hc],
    };
    Ok((mixed, inj))
}

/// [`hc_mix_quant_batch`] が使う作業域 (grow-only)。
///
/// 実形 (nt=512 / hc_dim=10240) では `xn` と `gate` が 20 MB ずつ、`mixed` が
/// 5 MB ある。呼び出しごとに `vec![0.0; ..]` で確保すると、書き込み時の
/// first-touch のページフォルトだけで 2 ms 前後乗る (実測:
/// `crate::model::hyper::tests::bench_mix_call_real_shape` で
/// grouped_rms_norm_batch 3.02 → 1.05 ms)。層ループの外で 1 回だけ持ち、
/// チャンクを跨いで使い回す。
#[derive(Default)]
struct MixScratch {
    xn: Vec<f32>,
    lo: Vec<f32>,
    gate: Vec<f32>,
    mixed: Vec<f32>,
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

/// [`hc_mix_quant`] のバッチ版 (`nt` トークンぶんまとめて行列積)。
///
/// 行列積 (`down` / `up` / `inject`) を [`gemv_batch`] に、要素演算
/// (grouped RMS 正規化・`hc_lo_apply` / `hc_mixed` / `hc_inject_apply`) を
/// `hyper::*_batch` に置き換える。要素演算は要素ごとに独立で、トークン内の
/// 加算順序も逐次版と同じなので、1 トークンぶんの出力は [`hc_mix_quant`] と
/// ビット一致する (`inject` は `[nt][hc]` の連結で返す)。
///
/// 要素演算の出力は `scratch` に書く (戻り値で返す `mixed` / `inject` だけ確保する)。
fn hc_mix_quant_batch(
    ress: &[HyperStreams],
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
    scratch: &mut MixScratch,
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    let nt = ress.len();
    let e = hp.n_embd;
    let hc_dim = hp.hc * hp.n_embd;
    for (t, r) in ress.iter().enumerate() {
        if r.data.len() != hc_dim {
            return Err(one(format!(
                "hc_mix (batch): トークン {t} の res 長 {} が hc_dim {hc_dim} と違う",
                r.data.len()
            )));
        }
    }
    let xn_flat = MixScratch::take(&mut scratch.xn, nt * hc_dim);
    hyper::grouped_rms_norm_batch_into(ress, norm, xn_flat, hp);

    let dn = down.n();
    if dn != hp.low_rank {
        return Err(one(format!(
            "hc_mix down (batch): 行数 {dn} が low_rank {} と違う",
            hp.low_rank
        )));
    }
    let lo = MixScratch::take(&mut scratch.lo, nt * dn);
    gemv_batch(down, xn_flat, nt, lo)
        .map_err(|err| one(format!("hc_mix down (batch): {err}")))?;
    hyper::hc_lo_apply_batch(lo, nt, hp);

    let un = up.n();
    if un != hc_dim {
        return Err(one(format!(
            "hc_mix up (batch): 行数 {un} が hc_dim {hc_dim} と違う"
        )));
    }
    let gate = MixScratch::take(&mut scratch.gate, nt * un);
    gemv_batch(up, lo, nt, gate).map_err(|err| one(format!("hc_mix up (batch): {err}")))?;
    let mixed_flat = MixScratch::take(&mut scratch.mixed, nt * e);
    hyper::hc_mixed_batch_into(xn_flat, gate, nt, mixed_flat, hp);
    let mut mixed = Vec::with_capacity(nt);
    for t in 0..nt {
        mixed.push(mixed_flat[t * e..(t + 1) * e].to_vec());
    }

    let injs = match inject {
        Some(w) => {
            let wn = w.n();
            if wn != hp.hc {
                return Err(one(format!(
                    "hc_mix inject (batch): 行数 {wn} が hc {} と違う",
                    hp.hc
                )));
            }
            let mut v = vec![0.0f32; nt * wn];
            gemv_batch(w, xn_flat, nt, &mut v)
                .map_err(|err| one(format!("hc_mix inject (batch): {err}")))?;
            hyper::hc_inject_apply_batch(&mut v, nt, hp);
            v
        }
        None => vec![0.0f32; nt * hp.hc],
    };
    Ok((mixed, injs))
}

/// PLE 注入 (層 `ple.layers` のみ)。`res_hc` を置き換えて返す。
///
/// 参照実装 `build_ple` (1467-1555 行) の順序:
/// 1. `ple_emb` (16 head × 160 = 2560。head の順に連結) を `ple_key` / `ple_value` に通す
/// 2. `key` を `ple_norm_key` で、`res_hc` を `ple_norm_query` で grouped RMS
/// 3. ストリームごとの内積 / sqrt(n_embd) → `sigmoid(sign(s) * sqrt(max(|s|, 1e-6)))`
/// 4. `gated[s] = value * gate[s]` (value は hc 本へ broadcast)
/// 5. `gated` を `ple_norm_conv` で grouped RMS → depthwise 因果 conv → silu
/// 6. `res_hc + (gated + conv_out)` (この足し方の順序も参照実装どおり)
///
/// **F1 の制限**: conv の履歴は 0 で埋める。`t = 0` で効くのは遅延 0 の項
/// (`k = kern - 1`) だけ。履歴バッファは F2 で入れる。
fn ple_inject(
    p: &Qwen4ExpParams,
    pw: &PleW,
    ple_emb: &[f32],
    res: &HyperStreams,
    hp: &HyperParams,
    hist_buf: &mut Vec<f32>,
) -> Result<HyperStreams> {
    let e = p.n_embd;
    let hc = p.n_hc;
    let hc_dim = p.hc_dim();

    // 1. key / value
    let mut key = vec![0.0f32; pw.key.n()];
    pw.key
        .gemv(ple_emb, &mut key)
        .map_err(|err| one(format!("ple_key: {err}")))?;
    if key.len() != hc_dim {
        return Err(one(format!(
            "ple_key: 出力長 {} が hc_dim {hc_dim} と違う",
            key.len()
        )));
    }
    let mut value = vec![0.0f32; pw.value.n()];
    pw.value
        .gemv(ple_emb, &mut value)
        .map_err(|err| one(format!("ple_value: {err}")))?;
    if value.len() != e {
        return Err(one(format!(
            "ple_value: 出力長 {} が n_embd {e} と違う",
            value.len()
        )));
    }

    // 2. grouped RMS (key は [hc][n_embd] と見る)
    let key_streams = HyperStreams {
        data: key,
        hc,
        n_embd: e,
    };
    let key_n = hyper::grouped_rms_norm(&key_streams, &pw.norm_key, hp);
    let query_n = hyper::grouped_rms_norm(res, &pw.norm_query, hp);

    // 3. ストリームごとの内積 → 符号付き平方根 → sigmoid
    let inv_sqrt_e = 1.0f32 / (e as f32).sqrt();
    let mut gate = Vec::with_capacity(hc);
    for s in 0..hc {
        let mut acc = 0.0f32;
        for i in 0..e {
            let j = s * e + i;
            acc += key_n[j] * query_n[j];
        }
        let sv = acc * inv_sqrt_e;
        let mag = sv.abs().max(1e-6).sqrt();
        let sign = if sv > 0.0 {
            1.0f32
        } else if sv < 0.0 {
            -1.0f32
        } else {
            0.0f32
        };
        gate.push(hyper::sigmoid(sign * mag));
    }

    // 4. gated[s][d] = value[d] * gate[s]
    let mut gated = vec![0.0f32; hc_dim];
    for (s, &g) in gate.iter().enumerate() {
        for d in 0..e {
            gated[s * e + d] = value[d] * g;
        }
    }

    // 5. grouped RMS → depthwise 因果 conv → silu
    let gated_streams = HyperStreams {
        data: gated.clone(),
        hc,
        n_embd: e,
    };
    let x = hyper::grouped_rms_norm(&gated_streams, &pw.norm_conv, hp);
    let kern = pw.kern;
    if pw.conv1d.len() != kern * hc_dim {
        return Err(one(format!(
            "ple_conv1d: 長さ {} が kern({kern}) * hc_dim({hc_dim}) と違う",
            pw.conv1d.len()
        )));
    }
    let dil = p.ple_ngram_size;
    let hist = (kern - 1) * dil;
    if hist_buf.len() != hist * hc_dim {
        return Err(one(format!(
            "ple_conv_state: 長さ {} が hist({hist}) * hc_dim({hc_dim}) と違う",
            hist_buf.len()
        )));
    }
    let mut conv = vec![0.0f32; hc_dim];
    for c in 0..hc_dim {
        // out[c] = Σ_k w[k][c] * x[c, t - (kern-1-k)*dil]
        // GGUF の ne は [kern, hc_dim] なので、列 c の k 番目は conv1d[c*kern + k]。
        // 参照実装は履歴を `padded = [hist + n_tokens]` として前置きするので、
        // `back` トークン前の値は hist_buf の (hist - back) 番目のスロット。
        let mut acc = 0.0f32;
        for k in 0..kern {
            let back = (kern - 1 - k) * dil;
            let v = if back == 0 {
                x[c]
            } else {
                hist_buf[(hist - back) * hc_dim + c]
            };
            acc += pw.conv1d[c * kern + k] * v;
        }
        conv[c] = hyper::silu(acc);
    }
    // 履歴を 1 トークン進める (古い側を捨て、末尾に今の x を入れる)。
    if hist > 0 {
        hist_buf.copy_within(hc_dim.., 0);
        hist_buf[(hist - 1) * hc_dim..].copy_from_slice(&x);
    }

    // 6. res + (gated + conv)
    let mut out = Vec::with_capacity(hc_dim);
    for i in 0..hc_dim {
        out.push(res.data[i] + (gated[i] + conv[i]));
    }
    Ok(HyperStreams {
        data: out,
        hc,
        n_embd: e,
    })
}

/// 層内フェーズ別の累積時間 (ns)。`DENNO_Q4_PROFILE=1` のときだけ加算する。
///
/// 27B の `DENNO_PROFILE` はこのモデルには配線されていないので、Flash-Next 用に
/// 独立した軽量なカウンタを置く。GEMV の中ではなく層の区切りで測るため、
/// 有効化しても各フェーズの内訳は見えるが内側の粒度は無い。
pub mod q4prof {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::sync::OnceLock;

    pub static PLE: AtomicU64 = AtomicU64::new(0);
    pub static MIX: AtomicU64 = AtomicU64::new(0);
    pub static ATTN: AtomicU64 = AtomicU64::new(0);
    pub static MOE: AtomicU64 = AtomicU64::new(0);
    pub static N_LAYER: AtomicU64 = AtomicU64::new(0);
    /// attention の内訳 (SSM 層: qkv 投影 / gate 投影 / GDN 本体 / 出力投影、
    /// full attention 層: まとめて FULL)。
    pub static A_QKV: AtomicU64 = AtomicU64::new(0);
    pub static A_GATE: AtomicU64 = AtomicU64::new(0);
    pub static A_GDN: AtomicU64 = AtomicU64::new(0);
    pub static A_OUT: AtomicU64 = AtomicU64::new(0);
    pub static A_FULL: AtomicU64 = AtomicU64::new(0);

    fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("DENNO_Q4_PROFILE").is_some())
    }

    #[inline]
    pub fn enabled() -> bool {
        static E: OnceLock<bool> = OnceLock::new();
        *E.get_or_init(on)
    }

    #[inline]
    pub fn add(c: &AtomicU64, t: std::time::Instant) {
        if enabled() {
            c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    /// (ple, mix, attn, moe, 層数) をミリ秒で返す。
    pub fn dump() -> (f64, f64, f64, f64, u64) {
        let f = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        (
            f(&PLE),
            f(&MIX),
            f(&ATTN),
            f(&MOE),
            N_LAYER.load(Ordering::Relaxed),
        )
    }

    /// attention の内訳をミリ秒で返す (qkv, gate, gdn, out, full)。
    pub fn dump_attn() -> (f64, f64, f64, f64, f64) {
        let f = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e6;
        (f(&A_QKV), f(&A_GATE), f(&A_GDN), f(&A_OUT), f(&A_FULL))
    }

    pub fn reset() {
        for c in [&PLE, &MIX, &ATTN, &MOE, &A_QKV, &A_GATE, &A_GDN, &A_OUT, &A_FULL] {
            c.store(0, Ordering::Relaxed);
        }
        N_LAYER.store(0, Ordering::Relaxed);
    }
}

/// MoE FFN (参照実装 `build_layer_ffn` 1240-1285 行)。
///
/// - ルータ: `ffn_gate_inp @ x` → softmax → top-k (`n_expert_used`) →
///   選択分の和で再正規化 (`norm_w = true`)。
/// - expert: `down( silu(gate(x)) * up(x) )` を選択 index 昇順に
///   `w_e` を掛けて積む。行列は expert 単位で切って GEMV する。
/// - shared expert: `sigmoid(ffn_gate_inp_shexp @ x)` を掛けて最後に 1 回加算。
///
/// `silu` は dense 経路 (`crate::tensor::silu_inplace`) と同じ式を使う。
///
/// `cache` が `Some` のときだけ expert の生バイトを [`ExpertCacheSet`] から取る
/// (P-MOE-G1)。`None` は従来どおり mmap のスライス + `prefetch()` で、バイト列も
/// 呼び出し順も変えない。**どちらの経路も同じ [`moe_expert::expert_gemv_any`] を
/// 通る**ので、同じバイト列に対しては出力がビット一致する。
///
/// P-MOE-G5: 選択 expert の寄与 `h_e` は互いに独立なので、`DENNO_MOE_PAR` (既定 1) が
/// 有効で選択 expert が 2 個以上のときだけ rayon で並列に計算する。**重み付きの
/// 足し込みは index 昇順の逐次**のままなので、丸めの順序は従来と変わらない。
fn moe_ffn_quant(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&mut ExpertCacheSet>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_with_par(p, lw, x, cache, moe_par_enabled())
}

/// `DENNO_MOE_PAR` の値の解釈 (純粋関数)。既定 (未設定) は有効、`"0"` のときだけ無効。
///
/// env を読む部分から切り離してあるのは、テストが環境変数を触らずに
/// 両方の分岐を確かめられるようにするため。
fn moe_par_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0"))
}

/// expert ループの並列化が有効か (`DENNO_MOE_PAR`, 既定 1)。プロセスで 1 回だけ読む。
fn moe_par_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_par_from_value(std::env::var("DENNO_MOE_PAR").ok().as_deref()))
}

/// [`moe_ffn_quant`] の本体。`par` は expert ループの並列化 on/off
/// (env を読まずに両経路を比べられるよう引数で受ける)。
fn moe_ffn_quant_with_par(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    x: &[f32],
    cache: Option<&mut ExpertCacheSet>,
    par: bool,
) -> Result<Vec<f32>> {
    let t_router = std::time::Instant::now();
    let e = p.n_embd;
    let ne = p.n_expert;
    let k = p.n_expert_used;
    let ff = p.n_ff_exp;

    // 1. ルータ
    let mut probs = vec![0.0f32; ne];
    lw.gate_inp
        .gemv(x, &mut probs)
        .map_err(|err| one(format!("層 {}: ffn_gate_inp: {err}", lw.il)))?;
    softmax(&mut probs);

    // 2. top-k (index 昇順で返る)
    let sel = select_topk(&probs, k);

    // 3. 再正規化 (選択 index 昇順に逐次加算した和で割る)
    let mut weights = vec![0.0f32; k];
    let mut sum = 0.0f32;
    for &i in &sel {
        sum += probs[i];
    }
    for (t, &i) in sel.iter().enumerate() {
        weights[t] = probs[i] / sum;
    }

    // 3.5 選択 expert の重みを先に読み始める (I/O と計算を重ねる)。
    //     1 トークンで 48 層 × 11 行列 × 921 KB を触るので、フォルト待ちが支配的になる。
    //     キャッシュ経路は pread で必要ぶんだけ読むので、mmap のページフォルトを
    //     先に起こす意味が無い (`prefetch()` 自体は無効化も変更もしない)。
    let mut cache = cache;
    if cache.is_none() {
        for &ei in &sel {
            lw.gate_exps.prefetch(ei);
            lw.up_exps.prefetch(ei);
            lw.down_exps.prefetch(ei);
        }
    }

    // 4. 選択 expert を index 昇順に積む
    //
    // P-MOE-G5: 各 expert の寄与 `h_e` は x だけで決まる独立な計算なので並列化できる。
    // ただし **重み付きの足し込みは index 昇順の逐次** のままにする (並列に足すと
    // 丸めの順序が変わる)。並列化するのは `h_e` の計算だけ。
    //
    // キャッシュ経路 (`cache: Some`) は並列化しない: `ExpertCacheSet` が `&mut` で、
    // スロットの取り合いを直列化する仕組みが無いため (従来どおりの逐次ループ)。
    let mut out = vec![0.0f32; e];
    #[cfg(feature = "par")]
    let parallel = par && cache.is_none() && sel.len() > 1;
    #[cfg(not(feature = "par"))]
    let parallel = {
        // par 機能なしでは行並列も expert 並列も無い (従来と同じ逐次経路)。
        let _ = par;
        false
    };

    moe_sub_add(0, t_router.elapsed());
    let t_experts = std::time::Instant::now();

    if parallel {
        #[cfg(feature = "par")]
        {
            let hs = expert_contribs_par(lw, &sel, x, e, ff)?;
            for (t, &_ei) in sel.iter().enumerate() {
                let pe = weights[t];
                let hh = &hs[t];
                for j in 0..e {
                    out[j] += pe * hh[j];
                }
            }
        }
    } else {
        let mut act = vec![0.0f32; ff];
        let mut up = vec![0.0f32; ff];
        let mut h = vec![0.0f32; e];
        for (t, &ei) in sel.iter().enumerate() {
            match cache.as_deref_mut() {
                // 従来経路: mmap のスライスを直接 GEMV する。
                None => {
                    lw.gate_exps.gemv(ei, x, &mut act)?;
                    silu_inplace(&mut act);
                    lw.up_exps.gemv(ei, x, &mut up)?;
                    for r in 0..ff {
                        act[r] *= up[r];
                    }
                    lw.down_exps.gemv(ei, &act, &mut h)?;
                }
                // P-MOE-G1: SSD pread + LRU スロット経路。同じ expert は読み直さない。
                Some(c) => {
                    let key = ExpertKey {
                        layer: lw.il as u16,
                        expert: ei as u16,
                    };
                    expert_gemv_cached(c, ExpertMatrix::Gate, &lw.gate_exps, key, x, &mut act)?;
                    silu_inplace(&mut act);
                    expert_gemv_cached(c, ExpertMatrix::Up, &lw.up_exps, key, x, &mut up)?;
                    for r in 0..ff {
                        act[r] *= up[r];
                    }
                    expert_gemv_cached(c, ExpertMatrix::Down, &lw.down_exps, key, &act, &mut h)?;
                }
            }
            let pe = weights[t];
            for j in 0..e {
                out[j] += pe * h[j];
            }
        }
    }

    // 5. shared expert (係数は sigmoid(gate_inp_shexp @ x))
    moe_sub_add(1, t_experts.elapsed());
    let t_shared = std::time::Instant::now();
    let sff = lw.gate_shexp.n();
    let mut sact = vec![0.0f32; sff];
    lw.gate_shexp
        .gemv(x, &mut sact)
        .map_err(|err| one(format!("層 {}: ffn_gate_shexp: {err}", lw.il)))?;
    silu_inplace(&mut sact);
    let mut sup = vec![0.0f32; sff];
    lw.up_shexp
        .gemv(x, &mut sup)
        .map_err(|err| one(format!("層 {}: ffn_up_shexp: {err}", lw.il)))?;
    for r in 0..sff {
        sact[r] *= sup[r];
    }
    let mut sh = vec![0.0f32; e];
    lw.down_shexp
        .gemv(&sact, &mut sh)
        .map_err(|err| one(format!("層 {}: ffn_down_shexp: {err}", lw.il)))?;
    let mut sg = vec![0.0f32; lw.gate_inp_shexp.n()];
    lw.gate_inp_shexp
        .gemv(x, &mut sg)
        .map_err(|err| one(format!("層 {}: ffn_gate_inp_shexp: {err}", lw.il)))?;
    let sgw = hyper::sigmoid(sg[0]);
    for j in 0..e {
        out[j] += sh[j] * sgw;
    }
    moe_sub_add(2, t_shared.elapsed());

    Ok(out)
}

/// MoE フェーズの内訳を測るための累積カウンタ (ns)。0 = router、1 = 選択 expert、2 = shared。
///
/// `moe` のフェーズ時間が 110〜124 ms と大きく、どこに落ちているか分からなかったので
/// 入れた。常時計測する (1 層あたり `Instant::now()` が 4 回で、無視できる)。
static MOE_SUB_NS: [std::sync::atomic::AtomicU64; 3] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn moe_sub_add(idx: usize, d: std::time::Duration) {
    MOE_SUB_NS[idx].fetch_add(d.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// `moe` の内訳 (router / 選択 expert / shared expert) を ms で返し、カウンタを 0 に戻す。
pub fn moe_sub_take() -> (f64, f64, f64) {
    use std::sync::atomic::Ordering;
    let mut v = [0f64; 3];
    for (i, slot) in MOE_SUB_NS.iter().enumerate() {
        v[i] = slot.swap(0, Ordering::Relaxed) as f64 / 1e6;
    }
    (v[0], v[1], v[2])
}

/// バッチ GEMV が 1 行ぶんを計算するときの入口。
///
/// **同じ量子型でも入口によってカーネルが違う**ので、呼び出し側に合わせる:
/// - [`GgufLinear::gemv`] は `moe_expert::expert_gemv` を通る。Q4_K / Q5_K だけが
///   融合カーネル、Q5_1 / Q8_0 はスカラー。ただしこのモデルのテンソルは
///   [`linear`] が `with_fast_q8(true)` で組むので、**Q8_0 だけ**は
///   `expert_gemv_q8_fast` (AVX2 融合) になる。
/// - [`ExpertW::gemv`] は `moe_expert::expert_gemv_any` を通る。Q5_1 / Q8_0 も
///   融合カーネル (AVX2+FMA) になる。
///
/// ここを取り違えると、同じバイト列・同じ `x` でも Q5_1 / Q8_0 で 1 ulp ずれる
/// (2026-10-27 に実際に踏んだ)。
#[derive(Clone, Copy)]
enum GemvKernel {
    /// [`GgufLinear::gemv`] と同じ入口。`fast_q8` は層の設定
    /// ([`GgufLinear::fast_q8`]) をそのまま渡す (Q8_0 で融合カーネルに載るかが
    /// 変わる)。`from_slice` で組んだ層を比べるときは、その層も同じ設定に
    /// しておくこと。
    Linear { fast_q8: bool },
    /// [`ExpertW::gemv`] と同じ入口 (`expert_gemv_any`)
    Expert,
}

/// バッチ GEMV が 1 回の呼び出しで触る `x` の上限 (バイト)。
///
/// ブロックを 1 回だけデコードして全トークンへ配る形は、1 行あたり
/// `nt * k * 4` バイトの `x` を読む。これが L2 (≈1 MB) を超えると行ごとの
/// 読み直しでキャッシュを落とし、逐次版より遅くなる (k=10240 / nt=64 で
/// 31 ms/chunk、逐次版 1.7 ms)。256 KB に切ると 1 呼び出しの `x` が L2 に
/// 収まり、重みのデコード回数も nt 回から nt/T 回に減る。
const GEMV_BATCH_TILE_BYTES: usize = 256 * 1024;

/// ブロック 1 回デコード版に載せる最小の行数。
///
/// 行数が少ない行列は、重みを nt 回デコードし直す代わりに行を 16 分割できる
/// 旧経路のほうが速い (実測 nt=64: n=640 で x0.88、n=1024 で x0.96、
/// n=2560 で x1.23〜1.47)。しきい値はその境目。
const GEMV_BATCH_MIN_ROWS: usize = 2048;

/// 量子化重みのバッチ GEMV: `out[t * n + r] = dot(row_r, x_t)`。
///
/// 行 `r` について `acc[t]` (t = 0..nt) を 0 から始め、ブロックを先頭から順に
/// **1 回だけ** f32 へ開いて、その場で全トークンへ配る。1 行ぶんの計算は
/// `kern` が指す入口 (= 逐次版と同じカーネル) なので、Q4_K の融合カーネル
/// (AVX2+FMA) を含めてビット一致する。行は互いに独立で、`x` に依存する前計算も
/// 同じ `x` から作られる。
///
/// 速くなる根拠は演算量ではなく**重みのデコードと読み込み**: 逐次版はトークン
/// ごとに同じ行を最初から読み直すが、ここでは重みのバイト列を 1 チャンクあたり
/// 1 回しか読まない。27B の expert は 1 トークンで 487 MB を触るので、ここが支配的。
fn gemv_rows_batch(
    kern: GemvKernel,
    ty: GgmlType,
    bytes: &[u8],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    if nt == 0 {
        return Ok(());
    }
    if xs.len() != nt * k {
        return Err(one(format!(
            "gemv_batch: x の長さ {} が nt({nt}) * k({k}) と違う",
            xs.len()
        )));
    }
    if outs.len() != nt * n {
        return Err(one(format!(
            "gemv_batch: out の長さ {} が nt({nt}) * n({n}) と違う",
            outs.len()
        )));
    }
    if n == 0 || bytes.len() % n != 0 {
        return Err(one(format!(
            "gemv_batch: 行数 {n} でバイト数 {} を割り切れない",
            bytes.len()
        )));
    }
    match kern {
        // 1 行版 (`GgufLinear::gemv`) と同じカーネルに振る。同じ型でも入口で
        // カーネルが違うので、ここを取り違えると 1 ulp ずれる ([`GemvKernel`]):
        //   Q8_0 + `fast_q8` → `expert_gemv_q8_fast` (AVX2+FMA 融合)
        //   Q8_0 (`fast_q8` 無し) / Q5_1 → `expert_gemv` のスカラー経路
        //   Q4_K → `expert_gemv` (AVX2+FMA があれば融合、無ければスカラー)
        //   Q5_K → `expert_gemv` (常にスカラー)
        // どちらのバッチ経路も「重みのブロックを 1 回だけデコードし、その場で
        // 全トークンへ配る」ので、重みのバイト列は 1 チャンクあたり 1 回しか読まない。
        // トークンごとの累積順は 1 行版と同一 = ビット一致。
        //
        // 2026-10-28: 以前は行ごとに `expert_gemv*` を nt 回呼んでいた (行優先)。
        // トークンを外側にしていたので、同じ重み行を nt 回デコードし直していた。
        GemvKernel::Linear { fast_q8 } => {
            let scalar_only = ty == GgmlType::Q5_1 || (ty == GgmlType::Q8_0 && !fast_q8);
            // 行数が少ない形はバッチ版が行並列を使えない (gemm 側の門番は
            // `PAR_MIN_ROWS`) うえ、重みのデコードし直しより行分割の効きが大きい。
            // その場合は行を自分で分ける旧経路の方が速い (実測: k=10240 / n=320 で
            // 旧 1.7 ms/chunk、バッチ版 8.6 ms。n=640 / 1024 も旧がわずかに速い)。
            if !gemv_batch_blockdecode(n, nt) {
                gemv_rows_pertoken(ty, fast_q8, bytes, k, n, xs, nt, outs)?;
            } else {
                #[cfg(test)]
                GEMV_BATCH_BLOCKDECODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // トークンをまとめて渡すと、1 行の計算で触る x が nt * k * 4 バイトに
                // なる。k=10240 / nt=64 だと 2.6 MB で L2 に載らず、行ごとに
                // 読み直すたびにキャッシュを落とす (実測: 31 ms/chunk、逐次版 1.7 ms)。
                // そこで 1 呼び出しの x を L2 に収まる枚数へ切り、重みのデコード回数を
                // nt 回から nt/T 回に減らす。トークンは独立なので値は変わらない。
                let t_max = (GEMV_BATCH_TILE_BYTES / (4 * k.max(1))).max(1);
                let mut done = 0usize;
                while done < nt {
                    let t = t_max.min(nt - done);
                    let res = if scalar_only {
                        moe_expert::expert_gemm_scalar(
                            ty,
                            bytes,
                            k,
                            n,
                            &xs[done * k..(done + t) * k],
                            t,
                            &mut outs[done * n..(done + t) * n],
                        )
                    } else {
                        moe_expert::expert_gemm_any(
                            ty,
                            bytes,
                            k,
                            n,
                            &xs[done * k..(done + t) * k],
                            t,
                            &mut outs[done * n..(done + t) * n],
                        )
                    };
                    res.map_err(|err| one(format!("gemv_batch: {err}")))?;
                    done += t;
                }
            }
        }
        // expert の重みは型ごとの融合カーネルを使う。1 行版 `expert_gemv_any` と
        // ビット一致することを H3 側のテストで確認済み。
        GemvKernel::Expert => {
            moe_expert::expert_gemm_any(ty, bytes, k, n, xs, nt, outs)
                .map_err(|err| one(format!("gemv_batch: {err}")))?;
        }
    }
    #[cfg(test)]
    GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// 量子型のバッチ GEMV を「ブロック 1 回デコード」で回すか (純粋関数)。
///
/// 行数が [`GEMV_BATCH_MIN_ROWS`] 未満の行列は、重みを nt 回デコードし直す
/// 代わりに行を 16 分割できる旧経路 ([`gemv_rows_pertoken`]) の方が速いので
/// そちらへ振る (実測値は定数の doc)。境目はテストで固定する。
fn gemv_batch_blockdecode(n: usize, nt: usize) -> bool {
    nt > 0 && n >= GEMV_BATCH_MIN_ROWS
}

/// 旧経路: 行ごとに `expert_gemv*` を `nt` 回呼び、行方向だけ並列にする。
///
/// バッチ版 ([`gemv_rows_batch`] の `Linear`) は行並列を gemm 側
/// ([`moe_expert::PAR_MIN_ROWS`]) に任せるので、行数がその門番より少ない形では
/// 1 スレッドに落ちる。行数が少ない行列はむしろこちらが速い (実測: k=10240 /
/// n=320 / nt=64 で 1.7 ms/chunk、バッチ版 8.6 ms)。重み行は nt 回デコードするが、
/// x は 1 トークンずつ L1 に載る。
///
/// 数値: 1 行 1 トークンの計算は [`GgufLinear::gemv`] と同じカーネル・同じ引数
/// (`n_out = 1`) なので、`fast_q8` の扱いさえ合わせればビット一致する。
fn gemv_rows_pertoken(
    ty: GgmlType,
    fast_q8: bool,
    bytes: &[u8],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    let par = n >= 64 && k >= 256;
    let rb = bytes.len() / n;
    // 行ごとに書くと出力がトークン優先の並びなので、いったん行優先で作って
    // 最後に詰め替える (n * nt * 4 バイト。640x8 で 20 KB)。
    let mut tmp = vec![0f32; n * nt];
    let run_row = |r: usize, o: &mut [f32]| -> Result<()> {
        let row = &bytes[r * rb..(r + 1) * rb];
        for t in 0..nt {
            let x = &xs[t * k..(t + 1) * k];
            let slot = &mut o[t..t + 1];
            let res = if ty == GgmlType::Q8_0 && fast_q8 {
                moe_expert::expert_gemv_q8_fast(row, k, 1, x, slot)
            } else {
                moe_expert::expert_gemv(ty, row, k, 1, x, slot)
            };
            res.map_err(|err| one(format!("gemv_batch: 行 {r} トークン {t}: {err}")))?;
        }
        Ok(())
    };
    #[cfg(feature = "par")]
    if par {
        use rayon::prelude::*;
        tmp.par_chunks_mut(nt)
            .enumerate()
            .try_for_each(|(r, o)| run_row(r, o))?;
    } else {
        for (r, o) in tmp.chunks_mut(nt).enumerate() {
            run_row(r, o)?;
        }
    }
    #[cfg(not(feature = "par"))]
    {
        let _ = par;
        for (r, o) in tmp.chunks_mut(nt).enumerate() {
            run_row(r, o)?;
        }
    }
    for t in 0..nt {
        for r in 0..n {
            outs[t * n + r] = tmp[r * nt + t];
        }
    }
    Ok(())
}

/// F32 行列積のトークンブロック幅 (1 ブロックのトークン数)。
///
/// 32 にすると 1 行の内側ループが 32 レーン (AVX2 で 4 本、AVX-512 で 2 本の
/// ベクタ) になり、重み 1 要素の読み出しで 32 MAC を回せる。ルータ
/// (`ffn_gate_inp` 2560→512, F32) の重みの再読み出しが「トークンごと」= nt 回から
/// nt/32 回へ減る (実測は REPORT_P33.md)。
const ROUTER_TB: usize = 32;

/// [`ROUTER_TB`] 本のトークンと同時に回す行数。
///
/// 2 × 32 = 64 f32 = 512 bit ベクタ 4 本で、アキュムレータがレジスタに載る。
/// 実測 (`moe_router_microbench`, 実形 nt=512 / n=512 / k=2560, 同一実行内 A/B):
/// RB=1 で 3.9〜4.6 ms、RB=2 で 2.5〜3.0 ms、RB=4 で 2.6 ms (8 スレッドでは 4.2)、
/// RB=8 で 4.1 ms。RB=4 以上はレジスタが溢れて遅くなる。
const ROUTER_RB: usize = 2;

/// [`ROUTER_TB`] 本のトークンを列優先へ詰め替える (`src` はトークン優先)。
///
/// `dst[(i * tb + t)] = src[t * k + i]`。8×8 タイルで回すと、読み書きとも
/// 連続 f32 8 本 (= 1 ベクタ) になる。
fn transpose_token_block(src: &[f32], dst: &mut [f32], k: usize, tb: usize) {
    const T: usize = 8;
    let kt = k / T * T;
    let tt = tb / T * T;
    for t0 in (0..tt).step_by(T) {
        for i0 in (0..kt).step_by(T) {
            // tile[i][t] (i = 列内の行方向、t = トークン)
            let mut tile = [[0f32; T]; T];
            for r in 0..T {
                let s = &src[(t0 + r) * k + i0..(t0 + r) * k + i0 + T];
                for (c, &v) in s.iter().enumerate() {
                    tile[c][r] = v;
                }
            }
            for c in 0..T {
                dst[(i0 + c) * tb + t0..(i0 + c) * tb + t0 + T].copy_from_slice(&tile[c]);
            }
        }
        // k の端数
        for t in t0..t0 + T {
            for i in kt..k {
                dst[i * tb + t] = src[t * k + i];
            }
        }
    }
    // トークンの端数
    for t in tt..tb {
        for i in 0..k {
            dst[i * tb + t] = src[t * k + i];
        }
    }
}

/// [`ROUTER_TB`] 本のトークン × 全行の内積を、行 [`ROUTER_RB`] 本ずつの
/// アキュムレータで回す (`xt` は列優先、`out` は `[ROUTER_TB][n]` のトークン優先)。
///
/// 1 要素の演算列は [`gemv_f32_blocked`] の doc を参照 (i 昇順の f32 逐次加算)。
fn router_block_kernel(wf: &[f32], k: usize, n: usize, xt: &[f32], out: &mut [f32]) {
    let mut r0 = 0usize;
    while r0 < n {
        let rb = (n - r0).min(ROUTER_RB);
        let mut acc = [[0f32; ROUTER_TB]; ROUTER_RB];
        for i in 0..k {
            let xv = &xt[i * ROUTER_TB..i * ROUTER_TB + ROUTER_TB];
            for r in 0..rb {
                let w = wf[(r0 + r) * k + i];
                let a = &mut acc[r];
                for t in 0..ROUTER_TB {
                    a[t] += w * xv[t];
                }
            }
        }
        for (t, o) in out.chunks_exact_mut(n).enumerate() {
            for r in 0..rb {
                o[r0 + r] = acc[r][t];
            }
        }
        r0 += rb;
    }
}

/// F32 行列積のトークンブロック版: `out[t*n + r] = dot(row_r, x_t)`。
///
/// # 速くなる理由
///
/// 従来は「トークンごとに重み全行を読む」ので、重み 5.2 MB (2560×512 F32) を
/// `nt` 回読んでいた (512 トークンで 2.7 GB/層)。ここでは 32 トークンをまとめて
/// 1 行の内積を 32 レーンで回すので、重みの読み出しが `nt/32` 回 = 16 回
/// (84 MB/層) になる。
///
/// # 数値 (ビット一致の根拠)
///
/// 1 要素は `acc = 0.0; for i in 0..k { acc += w[r*k + i] * x_t[i] }` の i 昇順
/// 逐次で、行を f32 に開いてから同じ順で内積する従来経路 ([`gemv_f32_pertoken`])
/// と**同じ演算列**。トークン 32 本と行 [`ROUTER_RB`] 本は互いに独立なので、同時に
/// 回しても丸めの順序は変わらない (mul と add は融合しない: Rust は fast-math を
/// 付けない)。端数 (nt % [`ROUTER_TB`]、n % [`ROUTER_RB`]) は同じ関数の中で
/// 1 本ずつ処理する。
fn gemv_f32_blocked(
    wf: &[f32],
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) {
    let nb = nt / ROUTER_TB;
    // 1. トークンを 32 本ずつ列優先へ詰め替える (ブロックごとに連続配置)。
    let mut xt = vec![0f32; nb * ROUTER_TB * k];
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        xt.par_chunks_mut(ROUTER_TB * k)
            .enumerate()
            .for_each(|(b, dst)| {
                let src = &xs[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k];
                transpose_token_block(src, dst, k, ROUTER_TB);
            });
    }
    #[cfg(not(feature = "par"))]
    for b in 0..nb {
        let src = &xs[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k];
        transpose_token_block(src, &mut xt[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k], k, ROUTER_TB);
    }
    // 2. ブロック × 全行 (ブロックは独立なので並列化しても値は変わらない)。
    let blocks: Vec<&mut [f32]> = outs[..nb * ROUTER_TB * n]
        .chunks_mut(ROUTER_TB * n)
        .collect();
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        blocks.into_par_iter().enumerate().for_each(|(b, o)| {
            router_block_kernel(wf, k, n, &xt[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k], o);
        });
    }
    #[cfg(not(feature = "par"))]
    for (b, o) in blocks.into_iter().enumerate() {
        router_block_kernel(wf, k, n, &xt[b * ROUTER_TB * k..(b + 1) * ROUTER_TB * k], o);
    }
    // 3. 端数のトークン (nt % 32) は従来どおり 1 トークンずつ。
    for t in nb * ROUTER_TB..nt {
        let x = &xs[t * k..(t + 1) * k];
        let o = &mut outs[t * n..(t + 1) * n];
        for r in 0..n {
            let row = &wf[r * k..(r + 1) * k];
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += row[i] * x[i];
            }
            o[r] = acc;
        }
    }
}

/// [`GgufLinear`] のバッチ GEMV (`nt` トークン × 1 行列)。
///
/// 量子型は [`gemv_rows_batch`] (逐次版と同じカーネル)、
/// 無量子型 (F32/F16/BF16) は [`GgufLinear::gemv`] と同じ「行を f32 に開いて
/// 列昇順に逐次加算」を、行ごとに 1 回だけ展開してトークンぶん使い回す。
///
/// 注意: `--features cuda` で重みが VRAM に常駐している場合、逐次版
/// ([`GgufLinear::gemv`]) は GPU カーネルで計算するため、ここ (CPU 参照) とは
/// 一致しない。バッチ経路は現状 CPU のみ (GPU 常駐時は `DENNO_PREFILL_BATCH` を
/// 立てないこと)。
///
/// 2026-10-27: GPU 常駐の重みをバッチでも使う版 (`grid.y` = トークンのカーネル) を
/// 実装して A/B したが、802 トークンで **GPU 版のほうが遅かった** (mix 0.30 → 0.72,
/// attn 1.04 → 1.20, moe 1.70 → 1.75 ms/層)。呼び出しごとの H2D + 起動 + 同期 +
/// D2H が、この行列サイズ (k=320〜10240) では計算量を上回る。CPU 参照のままにする。
pub(crate) fn gemv_batch(w: &GgufLinear, xs: &[f32], nt: usize, outs: &mut [f32]) -> Result<()> {
    let (k, n) = (w.k(), w.n());
    match w.ty() {
        GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => {
            // P-PF-GB: prefill のバッチ投影は GPU GEMM を先に試す。
            // 数値契約は prefill のみ解除済み (FMA 融合 + 加算順が違う)。
            // `DENNO_GPU_GEMM=0` で切れる。false なら CPU バッチへ落ちる。
            if crate::cuda::maybe_gpu_gguf_gemm_into(
                w.uid(),
                w.ty(),
                w.blocks().as_slice(),
                k,
                n,
                xs,
                nt,
                outs,
            ) {
                #[cfg(test)]
                GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            gemv_rows_batch(
                GemvKernel::Linear {
                    fast_q8: w.fast_q8(),
                },
                w.ty(),
                w.blocks().as_slice(),
                k,
                n,
                xs,
                nt,
                outs,
            )
        }
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
            if nt == 0 {
                return Ok(());
            }
            if xs.len() != nt * k {
                return Err(one(format!(
                    "gemv_batch: x の長さ {} が nt({nt}) * k({k}) と違う",
                    xs.len()
                )));
            }
            if outs.len() != nt * n {
                return Err(one(format!(
                    "gemv_batch: out の長さ {} が nt({nt}) * n({n}) と違う",
                    outs.len()
                )));
            }
            let bytes = w.blocks().as_slice();
            if n == 0 || bytes.len() % n != 0 {
                return Err(one(format!(
                    "gemv_batch: 行数 {n} でバイト数 {} を割り切れない",
                    bytes.len()
                )));
            }
            let rb = bytes.len() / n;
            let ty = w.ty();
            // P33: ルータ級の形 (32 トークン × 2 ブロック以上) はトークンブロック版。
            // 重みの再読み出しが nt 回 → nt/32 回になる。値は同じ (doc 参照)。
            if nt / ROUTER_TB >= 2 && n >= ROUTER_RB {
                let wf = dequant_rows_f32(ty, bytes, rb, k, n);
                gemv_f32_blocked(&wf, k, n, xs, nt, outs);
                #[cfg(test)]
                GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                #[cfg(test)]
                GEMV_BATCH_BLOCKED_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(());
            }
            gemv_f32_pertoken(ty, bytes, rb, k, n, xs, nt, outs);
            #[cfg(test)]
            GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        other => Err(one(format!("gemv_batch: 未対応の量子型 {other:?}"))),
    }
}

/// 重みを 1 行ずつ f32 へ開く (行は独立なので rayon で分割してよい)。
fn dequant_rows_f32(ty: GgmlType, bytes: &[u8], rb: usize, k: usize, n: usize) -> Vec<f32> {
    let mut wf = vec![0.0f32; n * k];
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        wf.par_chunks_mut(k).enumerate().for_each(|(r, row)| {
            crate::kquant::dequant_row_into(ty, &bytes[r * rb..(r + 1) * rb], row);
        });
    }
    #[cfg(not(feature = "par"))]
    for r in 0..n {
        crate::kquant::dequant_row_into(ty, &bytes[r * rb..(r + 1) * rb], &mut wf[r * k..(r + 1) * k]);
    }
    wf
}

/// 従来経路: 行を 1 回だけ f32 に開き、1 トークンずつ全行の内積を回す
/// (並列化するのはトークン方向だけ)。
///
/// 1 要素の累積は `acc += row[i] * x[i]` の i 昇順逐次なので、逐次版とビット一致する。
/// P33 でルータ級の形は [`gemv_f32_blocked`] に置き換えたが、形が小さいとき
/// (nt < 2 ブロック / n < [`ROUTER_RB`]) はこちらが速いので残してある。
#[allow(clippy::too_many_arguments)]
fn gemv_f32_pertoken(
    ty: GgmlType,
    bytes: &[u8],
    rb: usize,
    k: usize,
    n: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) {
    debug_assert_eq!(outs.len(), nt * n, "gemv_f32_pertoken: out の長さ");
    let wf = dequant_rows_f32(ty, bytes, rb, k, n);
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        outs.par_chunks_mut(n).enumerate().for_each(|(t, o)| {
            let x = &xs[t * k..(t + 1) * k];
            for r in 0..n {
                let row = &wf[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                o[r] = acc;
            }
        });
    }
    #[cfg(not(feature = "par"))]
    {
        for t in 0..nt {
            let x = &xs[t * k..(t + 1) * k];
            for r in 0..n {
                let row = &wf[r * k..(r + 1) * k];
                let mut acc = 0.0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                outs[t * n + r] = acc;
            }
        }
    }
}

/// [`ExpertW::gemv`] のバッチ版 (1 expert × `nt` トークン)。
///
/// [`ExpertW::slice`] が返す 1 expert ぶんのバイト列を [`gemv_rows_batch`] に渡す。
/// 逐次版と同じ [`moe_expert::expert_gemv_any`] を 1 行ずつ通るのでビット一致する
/// (`GgufLinear::gemv` とは Q5_1 / Q8_0 で入口が違う点に注意 → [`GemvKernel`])。
fn expert_gemv_batch(
    w: &ExpertW,
    e: usize,
    xs: &[f32],
    nt: usize,
    outs: &mut [f32],
) -> Result<()> {
    let s = w.slice(e)?;
    gemv_rows_batch(
        GemvKernel::Expert,
        w.ty,
        s,
        w.n_in,
        w.n_out,
        xs,
        nt,
        outs,
    )
    .map_err(|err| one(format!("{}: expert {e} (batch): {err}", w.name)))
}

/// 行のベクタを 1 本の平坦な配列に連結する ([`gemv_batch`] の入力形式)。
fn flatten_rows(rows: &[Vec<f32>], width: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(rows.len() * width);
    for (t, r) in rows.iter().enumerate() {
        if r.len() != width {
            return Err(one(format!(
                "flatten_rows: 行 {t} の長さ {} が {width} と違う",
                r.len()
            )));
        }
        out.extend_from_slice(r);
    }
    Ok(out)
}

/// テスト用: 層優先バッチ prefill / バッチ GEMV が実際に走った回数。
/// 「緑 = 経路が発動した証拠」ではないので、分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static PREFILL_BATCH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static GEMV_BATCH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// テスト用: 量子型のバッチ GEMV が**ブロック 1 回デコード経路**に入った回数。
/// 行数が少ない形は旧経路 (行ごとに nt 回デコード) に落ちるので、どちらを通ったかを
/// 値の一致だけからは判定できない。分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static GEMV_BATCH_BLOCKDECODE_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// テスト用: 無量子型 (F32/F16/BF16) のバッチ GEMV が**トークンブロック経路**
/// ([`gemv_f32_blocked`]) に入った回数。形が小さいと従来の 1 トークン経路へ
/// 落ちるので、分岐の通過をここで数える (緑 = 経路が発動した証拠ではない)。
#[cfg(test)]
pub(crate) static GEMV_BATCH_BLOCKED_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `DENNO_PREFILL_BATCH` の値の解釈 (純粋関数)。既定 (未設定) は無効。
///
/// `"1"` のときだけ有効。`"0"`・空文字・その他は無効 (既定 off を崩さない)。
fn prefill_batch_from_value(v: Option<&str>) -> bool {
    matches!(v, Some("1"))
}

/// 層優先バッチ prefill が有効か (`DENNO_PREFILL_BATCH`, 既定 off)。プロセスで 1 回だけ読む。
fn prefill_batch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| prefill_batch_from_value(std::env::var("DENNO_PREFILL_BATCH").ok().as_deref()))
}

/// 1 チャンクのトークン数 (`DENNO_PREFILL_BATCH_CHUNK`, 既定 8, 1..=512)。
///
/// チャンクを大きくするほど重みの再読み込みは減るが、残差 `[nt][hc_dim]` と
/// `hc_mix` の中間がそのぶん増える (27B の hc_dim は 20480 f32 = 80 KiB/トークン)。
///
/// 2026-10-28: 上限を 64 から 512 へ。Flash-Next は 512 expert / top-10 なので、
/// chunk 64 では 1 チャンクで 365 体ぶんの expert 重みを読む (1 体あたり 1.4
/// トークンしか使わない)。512 まで上げると 1 体あたり 10 トークンになり、
/// トークンあたりの expert 読み出しが 5.7 分の 1 になる。
fn prefill_batch_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DENNO_PREFILL_BATCH_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.clamp(1, 512))
            .unwrap_or(8)
    })
}

/// flat スケジューラの 1 行チャンク幅 (行数)。
///
/// 1 タスクの仕事は `MOE_FLAT_ROWS × n_in × m` MAC。実形 (n_in=2560, m=60) なら
/// 64 行で約 9.8 MMAC。1 層のタスク数は expert 数 × (n_ff/64 + n_embd/64) で、
/// 86 expert / n_ff 640 / n_embd 2560 なら 860 + 3,440 = 4,300 本。16 スレッドで
/// 269 本ずつなので、最後の 1 本の偏りは 1 タスク = 0.4% 程度。
///
/// 実測 (`moe_flat_schedule_microbench`, m=5..120 / 16 スレッド): 32 行 220 ms・
/// 64 行 210 ms・128 行 212 ms・256 行 213 ms。64〜256 は測定誤差 (±2%) の範囲で
/// 同等なので、偏りの小さい 64 を採る。
const MOE_FLAT_ROWS: usize = 64;

/// flat 経路へ切り替える閾値 (1 expert あたりの最大トークン数 `m_max`)。
///
/// 実測 (`moe_flat_schedule_microbench`, REPORT_P30.md) では **m = 1 でも flat が
/// 勝つ** (legacy の 10.5 ms/層 → flat 7.2 ms。1 層あたりの fork-join が 258 回 →
/// 2 回 + gather/scatter の 2 回になるため)。よって既定は閾値 0 = 常に flat。
/// legacy は `DENNO_MOE_FLAT=0` と、ビット一致の参照用テストに残してある。
const MOE_FLAT_MIN_M: usize = 0;

/// 選択 expert のループの並べ方。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MoeSchedule {
    /// `m_max` の閾値 ([`MOE_FLAT_MIN_M`]) で選ぶ (既定)
    Auto,
    /// 層全体の (expert, 行列, 行チャンク) を 1 本のタスク列にして 1 回だけ並列化する
    Flat,
    /// expert ごと × 行列ごとに 1 回ずつ rayon を立てる (従来の形)
    Legacy,
}

/// `DENNO_MOE_FLAT` の解釈 (純粋関数)。未設定・その他の値は [`MoeSchedule::Auto`]。
fn moe_schedule_from_value(v: Option<&str>) -> MoeSchedule {
    match v {
        Some("1") => MoeSchedule::Flat,
        Some("0") => MoeSchedule::Legacy,
        _ => MoeSchedule::Auto,
    }
}

/// expert ループの並べ方 (既定 Auto)。プロセスで 1 回だけ読む。
fn moe_schedule() -> MoeSchedule {
    static ON: std::sync::OnceLock<MoeSchedule> = std::sync::OnceLock::new();
    *ON.get_or_init(|| moe_schedule_from_value(std::env::var("DENNO_MOE_FLAT").ok().as_deref()))
}

/// テスト用: flat 経路 / legacy 経路が実際に走った回数。
/// 「緑 = その経路が発動した証拠」ではないので、分岐の通過をここで数える。
#[cfg(test)]
pub(crate) static MOE_FLAT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
pub(crate) static MOE_LEGACY_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// flat スケジューラの 1 タスク (1 expert × 行チャンク `[a, b)`)。
///
/// 行列の別 (gate / up / down) はフィールドに持たず、タスク列を 2 本に分ける。
/// gate と up は同じ入力 `xin` を読むので 1 タスクにまとめ、silu の積までその場で
/// 書く (3 回目のバリアを作らないため)。`pos` は active な expert の並び (`eid`) の添字。
#[derive(Clone, Copy)]
struct MoeTask {
    pos: u16,
    a: u16,
    b: u16,
}

/// 1 行チャンクのタスクを作る (u16 に収まらない形は黙って切らずにエラーで落とす)。
fn moe_task(pos: usize, a: usize, b: usize) -> Result<MoeTask> {
    let conv = |v: usize, what: &str| {
        u16::try_from(v).map_err(|_| one(format!("flat スケジューラ: {what} {v} が u16 を超える")))
    };
    Ok(MoeTask {
        pos: conv(pos, "expert 添字")?,
        a: conv(a, "行")?,
        b: conv(b, "行")?,
    })
}

/// `n_out` 行を `rows` 行ずつのタスクへ切って `out` に積む。
fn moe_row_chunks(pos: usize, n_out: usize, rows: usize, out: &mut Vec<MoeTask>) -> Result<()> {
    let mut a = 0usize;
    while a < n_out {
        let b = (a + rows).min(n_out);
        out.push(moe_task(pos, a, b)?);
        a = b;
    }
    Ok(())
}

/// `silu(v) * u`。`silu_inplace` と同じ式 (`v / (1 + exp(-v))`) の後に積を取る。
#[inline]
fn silu_mul(v: f32, u: f32) -> f32 {
    v / (1.0 + (-v).exp()) * u
}

/// flat スケジューラの作業バッファ。**層をまたいで使い回す**。
///
/// 層ごとに `vec![0.0; n]` で確保すると、初回タッチのページフォルトを毎層払う
/// (実形 m=64 で 4 本合計 140 MB ≈ 35,000 ページ。実測は REPORT_P30.md)。
/// 中身は「読む前に必ず全部書く」のでゼロ初期化には依存しない
/// (`gemm_rows` は行チャンクの全要素を書き、gather は `xin` を全部埋める)。
#[derive(Default)]
struct MoeScratch {
    xin: Vec<f32>,
    act: Vec<f32>,
    up: Vec<f32>,
    h: Vec<f32>,
}

impl MoeScratch {
    /// 必要な長さに合わせる (伸ばすときだけゼロ埋めが走る = 初回だけ)。
    fn resize(&mut self, tot: usize, e: usize, ff: usize) {
        self.xin.resize(tot * e, 0.0);
        self.act.resize(tot * ff, 0.0);
        self.up.resize(tot * ff, 0.0);
        self.h.resize(tot * e, 0.0);
    }

    fn bytes(&self) -> usize {
        (self.xin.capacity() + self.act.capacity() + self.up.capacity() + self.h.capacity()) * 4
    }
}

/// 使い回し用の置き場 (プロセスで 1 つ)。使用中は `None` にして貸し出す。
static MOE_SCRATCH: std::sync::Mutex<Option<MoeScratch>> = std::sync::Mutex::new(None);

/// 借りる。使用中 (同時に 2 本走っている) なら新しく確保する = 壊れない。
fn moe_scratch_take() -> MoeScratch {
    MOE_SCRATCH
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .unwrap_or_default()
}

/// 返す。大きすぎるものは捨てる (巨大な層のあとメモリを抱え込まない)。
fn moe_scratch_put(s: MoeScratch) {
    const KEEP_BYTES: usize = 512 * 1024 * 1024;
    let mut g = MOE_SCRATCH.lock().unwrap_or_else(|e| e.into_inner());
    *g = if s.bytes() <= KEEP_BYTES { Some(s) } else { None };
}

/// 選択 expert の FFN を「層全体の (expert, 行列, 行チャンク) を 1 本のタスク列に
/// して 1 回だけ並列化する」形で計算し、`out` に重み付きで足し込む (呼び出し側で
/// ゼロ初期化済み)。
///
/// 従来の [`moe_experts_legacy`] は expert 1 体につき gate / up / down で 3 回
/// rayon の fork-join をしていた (86 expert × 3 × 48 層 = 12,384 回/chunk)。
/// ここでは 1 層あたり「gate+up+silu 積」の 1 パスと「down」の 1 パスの計 2 回に減らす。
/// down の入力は gate/up の結果 (silu の積) なので、この 2 パスは分けざるをえない。
///
/// # 数値 (ビット一致の根拠)
///
/// - 1 要素の計算は `moe_expert::expert_gemm_any` と**同じ関数** (`moe_expert::gemm_rows`)
///   を同じ引数 (同じ `n_in` / `m` / 行) で呼ぶ。行チャンクは `[0, n_out)` の分割で、
///   行の中の累積順は行内で閉じるので、どのスレッドがどの行を計算しても値は変わらない。
/// - silu の積は `v / (1 + (-v).exp()) * u` で、legacy の `silu_inplace` + `*=` と同順。
/// - 足し込みは expert index 昇順 → トークン昇順の逐次 (legacy と同順)。
/// - タスク表は **m の降順** に並べるが、タスクは (expert, 行チャンク) ごとに
///   独立な領域を書くだけなので順序は値に影響しない (gather / scatter は eid 昇順)。
#[allow(clippy::too_many_arguments)]
fn moe_experts_flat(
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    xs: &[f32],
    e: usize,
    ff: usize,
    rows: usize,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(test)]
    MOE_FLAT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // 1. expert の並びと、expert ごとの領域の先頭 (f32 要素) を確定する。
    let mut eid: Vec<usize> = Vec::new();
    let mut ms: Vec<usize> = Vec::new();
    let mut off_x: Vec<usize> = Vec::new();
    let mut off_a: Vec<usize> = Vec::new();
    let mut off_h: Vec<usize> = Vec::new();
    let mut tot = 0usize;
    for (ei, picks) in by_expert.iter().enumerate() {
        if picks.is_empty() {
            continue;
        }
        eid.push(ei);
        ms.push(picks.len());
        off_x.push(tot * e);
        off_a.push(tot * ff);
        off_h.push(tot * e);
        tot += picks.len();
    }
    let n = eid.len();
    if n == 0 {
        return Ok(());
    }

    // フェーズ内訳 (`DENNO_MOE_FLAT_PHASE=1` のときだけ stderr へ出す)。
    let ph = std::env::var_os("DENNO_MOE_FLAT_PHASE").is_some();
    let mut ph_t = std::time::Instant::now();
    macro_rules! lap {
        ($name:expr) => {
            if ph {
                eprintln!("[moe-flat] {:.1} ms {}", ph_t.elapsed().as_secs_f64() * 1e3, $name);
                ph_t = std::time::Instant::now();
            }
        };
    }

    // 2. 作業バッファ (層をまたいで使い回す) を必要な長さに合わせる。
    let mut sc = moe_scratch_take();
    sc.resize(tot, e, ff);
    let MoeScratch {
        mut xin,
        mut act,
        mut up,
        mut h,
    } = sc;

    // 3. gather: expert ごとにトークンを詰め直す (legacy と同じ順序 = トークン昇順)。
    //    expert ごとに独立なコピーなので、expert を単位に並列化しても値は変わらない。
    {
        let mut parts: Vec<&mut [f32]> = Vec::with_capacity(n);
        let mut rest: &mut [f32] = &mut xin[..];
        for pos in 0..n {
            let (head, tail) = rest.split_at_mut(ms[pos] * e);
            parts.push(head);
            rest = tail;
        }
        let gather_one = |(pos, dst): (usize, &mut [f32])| {
            let ei = eid[pos];
            for (idx, &(t, _)) in by_expert[ei].iter().enumerate() {
                dst[idx * e..(idx + 1) * e].copy_from_slice(&xs[t * e..(t + 1) * e]);
            }
        };
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            parts.into_par_iter().enumerate().for_each(gather_one);
        }
        #[cfg(not(feature = "par"))]
        parts.into_iter().enumerate().for_each(gather_one);
    }
    lap!("gather");

    // 4. 行列・入力・出力長の検証 (1 expert 1 行列につき 1 度だけ) とバイト列の切り出し
    let (g, u, d) = (&lw.gate_exps, &lw.up_exps, &lw.down_exps);
    let (gty, uty, dty) = (g.ty, u.ty, d.ty);
    let rb_g = moe_expert::row_bytes(gty, e).map_err(|err| one(format!("{}: {err}", g.name)))?;
    let rb_u = moe_expert::row_bytes(uty, e).map_err(|err| one(format!("{}: {err}", u.name)))?;
    let rb_d = moe_expert::row_bytes(dty, ff).map_err(|err| one(format!("{}: {err}", d.name)))?;
    let mut gs: Vec<&[u8]> = Vec::with_capacity(n);
    let mut us: Vec<&[u8]> = Vec::with_capacity(n);
    let mut ds: Vec<&[u8]> = Vec::with_capacity(n);
    for pos in 0..n {
        let m = ms[pos];
        let ei = eid[pos];
        let xa = &xin[off_x[pos]..off_x[pos] + m * e];
        let ad = &act[off_a[pos]..off_a[pos] + m * ff];
        let bg = g.slice(ei)?;
        moe_expert::check_gemm_matrix(gty, bg, e, ff, xa, m)
            .map_err(|err| one(format!("{}: expert {ei}: {err}", g.name)))?;
        let bu = u.slice(ei)?;
        moe_expert::check_gemm_matrix(uty, bu, e, ff, xa, m)
            .map_err(|err| one(format!("{}: expert {ei}: {err}", u.name)))?;
        let bd = d.slice(ei)?;
        moe_expert::check_gemm_matrix(dty, bd, ff, e, ad, m)
            .map_err(|err| one(format!("{}: expert {ei}: {err}", d.name)))?;
        gs.push(bg);
        us.push(bu);
        ds.push(bd);
    }
    lap!("validate+slices+task_table");

    // 5. タスク表 (gate+up+silu 積 / down)。**m の降順** に並べる
    //    (m の大きい expert から先にタスク列へ入るので、rayon の分割で
    //    重い仕事が先に始まる)。タスクは独立な領域を書くので順序は値に影響しない。
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| ms[b].cmp(&ms[a]).then_with(|| a.cmp(&b)));
    let mut ta: Vec<MoeTask> = Vec::new();
    let mut tb: Vec<MoeTask> = Vec::new();
    let rows = rows.max(1);
    for &pos in &order {
        moe_row_chunks(pos, ff, rows, &mut ta)?;
        moe_row_chunks(pos, e, rows, &mut tb)?;
    }

    // SAFETY (act / up / h をタスク間で共有してよい根拠):
    // - タスク (pos, a, b) が書くのは expert `pos` の領域の `t * n_out + [a, b)`
    //   (t < ms[pos]) だけで、`[a, b)` はタスク間で重ならない。
    // - 領域の大きさは `tot * ff` / `tot * e` で、添字は必ずその内側。
    // - `moe_expert::gemm_rows` は `m * stride` 要素以上を書き、行チャンクの外へは書かない。
    // よって異なるタスクが同じ要素へ書くことはない (data race なし)。
    let act_p = moe_expert::OutPtr::new(act.as_mut_ptr());
    let up_p = moe_expert::OutPtr::new(up.as_mut_ptr());
    let h_p = moe_expert::OutPtr::new(h.as_mut_ptr());

    // 6. gate+up+silu 積 (行チャンクごとに 1 タスク)
    let run_a = |t: &MoeTask| {
        let pos = t.pos as usize;
        let (a, b) = (t.a as usize, t.b as usize);
        let m = ms[pos];
        let x = &xin[off_x[pos]..off_x[pos] + m * e];
        let dst_a = act_p.add(off_a[pos] + a);
        let dst_u = up_p.add(off_a[pos] + a);
        // 前提 (安全に呼べる条件): `[a, b)` は [0, n_out) の分割の 1 つで、領域は
        // `m * stride` 要素以上。`gemm_rows` 自身は unsafe fn ではないが、生ポインタの
        // 指す先の条件はこのモジュールのタスク分割が保証する (上の SAFETY を参照)。
        moe_expert::gemm_rows(gty, rb_g, gs[pos], e, a, b, x, m, dst_a.get(), ff, true);
        moe_expert::gemm_rows(uty, rb_u, us[pos], e, a, b, x, m, dst_u.get(), ff, true);
        // act の行 [a, b) を silu(act) * up で置き換える (このタスクが書いた行だけ)。
        let pa = act_p.get();
        let pu = up_p.get();
        for idx in 0..m {
            let base = off_a[pos] + idx * ff;
            for r in a..b {
                // SAFETY: base + r はこのタスクが上で書いた範囲 (他タスクと重ならない)。
                unsafe {
                    let v = *pa.add(base + r);
                    *pa.add(base + r) = silu_mul(v, *pu.add(base + r));
                }
            }
        }
    };

    // 7. down (act は前のパスで確定しているので読むだけ)
    let act_r: &[f32] = &act;
    let run_b = |t: &MoeTask| {
        let pos = t.pos as usize;
        let (a, b) = (t.a as usize, t.b as usize);
        let m = ms[pos];
        let x = &act_r[off_a[pos]..off_a[pos] + m * ff];
        let dst = h_p.add(off_h[pos] + a);
        // 前提は run_a と同じ (`[a, b)` は [0, n_out) の分割の 1 つ、領域は `m * stride` 以上)。
        moe_expert::gemm_rows(dty, rb_d, ds[pos], ff, a, b, x, m, dst.get(), e, true);
    };

    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        ta.par_iter().for_each(run_a);
        tb.par_iter().for_each(run_b);
    }
    #[cfg(not(feature = "par"))]
    {
        ta.iter().for_each(run_a);
        tb.iter().for_each(run_b);
    }
    lap!("phaseA+phaseB");

    // 8. 重み付きの足し込み。トークンごとに書く先が違う (`out[t*e..]`) ので
    //    トークン単位で並列化する。1 トークンの中は expert 昇順の逐次 = legacy と同順。
    {
        let nt = out.len() / e.max(1);
        // (pos, expert 内の添字, 重みの添字)。pos 昇順に積むので index 昇順が保たれる。
        let mut by_token: Vec<Vec<(u16, usize, u16)>> = vec![Vec::new(); nt];
        for (pos, &ei) in eid.iter().enumerate() {
            let pos_u16 = moe_task(pos, 0, 0)?.pos;
            for (idx, &(t, j)) in by_expert[ei].iter().enumerate() {
                by_token[t].push((pos_u16, idx, j as u16));
            }
        }
        let mut parts: Vec<&mut [f32]> = out.chunks_mut(e.max(1)).collect();
        let scatter_one = |(t, o): (usize, &mut [f32])| {
            for &(pos, idx, j) in &by_token[t] {
                let pos = pos as usize;
                let pe = weights[t * k + j as usize];
                let hh = &h[off_h[pos] + idx * e..off_h[pos] + (idx + 1) * e];
                for d in 0..e {
                    o[d] += pe * hh[d];
                }
            }
        };
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            parts.into_par_iter().enumerate().for_each(scatter_one);
        }
        #[cfg(not(feature = "par"))]
        parts.into_iter().enumerate().for_each(scatter_one);
    }
    lap!("scatter");
    moe_scratch_put(MoeScratch { xin, act, up, h });
    Ok(())
}

/// 選択 expert の FFN を expert ごとに回す (従来の形)。
///
/// expert 1 体につき gate / up / down で 3 回 rayon の fork-join をする。
/// `m == 1` の expert は 1 トークン用の入口 ([`ExpertW::gemv`]) を通る。
/// `m_max` が小さい層 ([`MOE_FLAT_MIN_M`] 以下) ではこちらが速い
/// (`moe_flat_schedule_microbench` の実測)。
///
/// 数値: 各 expert の `h_e` は (行, トークン) ごとに逐次版と同じカーネルで計算し、
/// 足し込みは expert index 昇順 → トークン昇順の逐次。`moe_ffn_quant` と一致する。
fn moe_experts_legacy(
    lw: &LayerW,
    by_expert: &[Vec<(usize, usize)>],
    weights: &[f32],
    k: usize,
    xs: &[f32],
    e: usize,
    ff: usize,
    out: &mut [f32],
) -> Result<()> {
    #[cfg(test)]
    MOE_LEGACY_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    for ei in 0..by_expert.len() {
        let picks = &by_expert[ei];
        if picks.is_empty() {
            continue;
        }
        let m = picks.len();
        let mut xin = Vec::with_capacity(m * e);
        for &(t, _) in picks {
            xin.extend_from_slice(&xs[t * e..(t + 1) * e]);
        }
        let mut act = vec![0.0f32; m * ff];
        let mut up = vec![0.0f32; m * ff];
        let mut h = vec![0.0f32; m * e];
        if m == 1 {
            // 1 トークンなら逐次版とまったく同じ入口を使う (行並列も従来どおり)。
            lw.gate_exps.gemv(ei, &xin, &mut act)?;
            silu_inplace(&mut act);
            lw.up_exps.gemv(ei, &xin, &mut up)?;
            for r in 0..m * ff {
                act[r] *= up[r];
            }
            lw.down_exps.gemv(ei, &act, &mut h)?;
        } else {
            expert_gemv_batch(&lw.gate_exps, ei, &xin, m, &mut act)?;
            silu_inplace(&mut act);
            expert_gemv_batch(&lw.up_exps, ei, &xin, m, &mut up)?;
            for r in 0..m * ff {
                act[r] *= up[r];
            }
            expert_gemv_batch(&lw.down_exps, ei, &act, m, &mut h)?;
        }
        // 重み付きの足し込みはトークンごとに逐次 (index 昇順のまま)
        for (idx, &(t, j)) in picks.iter().enumerate() {
            let pe = weights[t * k + j];
            let hh = &h[idx * e..(idx + 1) * e];
            for d in 0..e {
                out[t * e + d] += pe * hh[d];
            }
        }
    }
    Ok(())
}

/// [`moe_ffn_quant`] のバッチ版 (`nt` トークン)。
///
/// ルータ / shared expert / 選択 expert の行列積を [`gemv_batch`] に置き換える。
/// 同じ expert を選んだトークンはまとめて GEMV する。
///
/// 数値: トークンごとの softmax → top-k → 再正規化 → **index 昇順**の重み付き加算、
/// という順序をそのまま保つ。1 トークンの `h_e` は (行, トークン) ごとに逐次版と
/// 同じカーネルで計算され、足し込みは index 昇順の逐次なので f32 でビット一致する。
///
/// キャッシュ経路 (`cache: Some`) は [`ExpertCacheSet`] が `&mut` でスロットを
/// 共有するためバッチ化しない (トークンごとに逐次版へ落とす)。
fn moe_ffn_quant_batch(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&mut ExpertCacheSet>,
) -> Result<Vec<f32>> {
    moe_ffn_quant_batch_with(p, lw, xs, nt, cache, moe_schedule())
}

/// [`moe_ffn_quant_batch`] の本体。expert ループの並べ方 (`sched`) を外から渡せる
/// 形にしてある (テストと A/B 計測で両方の経路を同じ入力に通すため)。
fn moe_ffn_quant_batch_with(
    p: &Qwen4ExpParams,
    lw: &LayerW,
    xs: &[f32],
    nt: usize,
    cache: Option<&mut ExpertCacheSet>,
    sched: MoeSchedule,
) -> Result<Vec<f32>> {
    let e = p.n_embd;
    let ne = p.n_expert;
    let k = p.n_expert_used;
    let ff = p.n_ff_exp;
    if nt == 0 {
        return Ok(Vec::new());
    }
    if xs.len() != nt * e {
        return Err(one(format!(
            "moe_ffn_quant (batch): x の長さ {} が nt({nt}) * n_embd({e}) と違う",
            xs.len()
        )));
    }
    let mut cache = cache;
    if let Some(c) = cache.as_deref_mut() {
        let mut out = Vec::with_capacity(nt * e);
        for t in 0..nt {
            out.extend_from_slice(&moe_ffn_quant(p, lw, &xs[t * e..(t + 1) * e], Some(c))?);
        }
        return Ok(out);
    }

    // 1. ルータ (バッチ)
    let t_router = std::time::Instant::now();
    let mut probs = vec![0.0f32; nt * ne];
    gemv_batch(&lw.gate_inp, xs, nt, &mut probs)
        .map_err(|err| one(format!("層 {}: ffn_gate_inp (batch): {err}", lw.il)))?;

    // 2. トークンごとの softmax (行は独立なので並列化しても値は変わらない)
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        probs.par_chunks_mut(ne).for_each(softmax);
    }
    #[cfg(not(feature = "par"))]
    for t in 0..nt {
        softmax(&mut probs[t * ne..(t + 1) * ne]);
    }

    // 3. top-k (index 昇順) + 再正規化 (選択 index 昇順に逐次加算した和で割る)。
    //    P33: トークンごとの `Vec` を作らず、平らな配列へ直接書く (1 層あたり
    //    1,024 回の確保が 2 回になる)。選択規則は [`select_topk_into`] が
    //    [`select_topk`] と同じ (同値は小さい index) なので値は変わらない。
    let mut sels: Vec<usize> = vec![0; nt * k];
    let mut weights: Vec<f32> = vec![0.0; nt * k];
    let renorm = |pr: &[f32], sel: &[usize], w: &mut [f32]| {
        let mut sum = 0.0f32;
        for &i in sel {
            sum += pr[i];
        }
        for (j, &i) in sel.iter().enumerate() {
            w[j] = pr[i] / sum;
        }
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        sels.par_chunks_mut(k)
            .zip(weights.par_chunks_mut(k))
            .enumerate()
            .for_each_init(Vec::<u32>::new, |buf, (t, (s, w))| {
                let pr = &probs[t * ne..(t + 1) * ne];
                select_topk_into(pr, k, buf, s);
                renorm(pr, s, w);
            });
    }
    #[cfg(not(feature = "par"))]
    {
        let mut buf: Vec<u32> = Vec::new();
        for t in 0..nt {
            let pr = &probs[t * ne..(t + 1) * ne];
            select_topk_into(pr, k, &mut buf, &mut sels[t * k..(t + 1) * k]);
            renorm(pr, &sels[t * k..(t + 1) * k], &mut weights[t * k..(t + 1) * k]);
        }
    }
    moe_sub_add(0, t_router.elapsed());

    // 4. 選択 expert の重みを先に読み始める (逐次版と同じ expert を同じ順で)
    //    同じ expert は 1 層で何度も選ばれる。madvise は 1 expert 1 回で足りるので、
    //    トークン×k 回 (512 × 10 = 5,120 回/層) ではなく distinct な expert の数 (86 回/層)
    //    だけ呼ぶ。選択順は変わらないので数値には影響しない。
    {
        let mut seen = vec![false; ne];
        for &ei in &sels {
            if !seen[ei] {
                seen[ei] = true;
                lw.gate_exps.prefetch(ei);
                lw.up_exps.prefetch(ei);
                lw.down_exps.prefetch(ei);
            }
        }
    }

    // 5. expert ごとのトークン一覧 (`picks`) と件数 `m` を 1 パスで確定する。
    //    sels は index 昇順の distinct な expert なので、legacy が
    //    `position()` で作っていた順 (トークン昇順, 同トークン内は j 昇順) と一致する。
    //    件数を先に数えて容量を確保するので、push のたびの再確保は無い。
    let t_experts = std::time::Instant::now();
    let mut out = vec![0.0f32; nt * e];
    let mut counts = vec![0usize; ne];
    for &ei in &sels {
        counts[ei] += 1;
    }
    let mut by_expert: Vec<Vec<(usize, usize)>> = counts
        .iter()
        .map(|&c| Vec::with_capacity(c))
        .collect();
    for t in 0..nt {
        for j in 0..k {
            by_expert[sels[t * k + j]].push((t, j));
        }
    }
    let m_max = by_expert.iter().map(|p| p.len()).max().unwrap_or(0);
    let flat = match sched {
        MoeSchedule::Flat => true,
        MoeSchedule::Legacy => false,
        MoeSchedule::Auto => m_max > MOE_FLAT_MIN_M,
    };
    if std::env::var_os("DENNO_MOE_M_DEBUG").is_some() {
        let nonempty = by_expert.iter().filter(|p| !p.is_empty()).count();
        let sum_m: usize = by_expert.iter().map(|p| p.len()).sum();
        eprintln!(
            "[moe-m] nt={nt} ne={ne} k={k} nonempty={nonempty} sum_m={sum_m} mean={:.2} max={m_max} flat={flat}",
            sum_m as f32 / nonempty.max(1) as f32
        );
    }
    if flat {
        moe_experts_flat(
            lw,
            &by_expert,
            &weights,
            k,
            xs,
            e,
            ff,
            MOE_FLAT_ROWS,
            &mut out,
        )?;
    } else {
        moe_experts_legacy(lw, &by_expert, &weights, k, xs, e, ff, &mut out)?;
    }

    // 5. shared expert (バッチ)。係数は sigmoid(gate_inp_shexp @ x)
    moe_sub_add(1, t_experts.elapsed());
    let t_shared = std::time::Instant::now();
    let sff = lw.gate_shexp.n();
    let mut sact = vec![0.0f32; nt * sff];
    gemv_batch(&lw.gate_shexp, xs, nt, &mut sact)
        .map_err(|err| one(format!("層 {}: ffn_gate_shexp (batch): {err}", lw.il)))?;
    silu_inplace(&mut sact);
    let mut sup = vec![0.0f32; nt * sff];
    gemv_batch(&lw.up_shexp, xs, nt, &mut sup)
        .map_err(|err| one(format!("層 {}: ffn_up_shexp (batch): {err}", lw.il)))?;
    for r in 0..nt * sff {
        sact[r] *= sup[r];
    }
    let mut sh = vec![0.0f32; nt * e];
    gemv_batch(&lw.down_shexp, &sact, nt, &mut sh)
        .map_err(|err| one(format!("層 {}: ffn_down_shexp (batch): {err}", lw.il)))?;
    let gn = lw.gate_inp_shexp.n();
    let mut sg = vec![0.0f32; nt * gn];
    gemv_batch(&lw.gate_inp_shexp, xs, nt, &mut sg)
        .map_err(|err| one(format!("層 {}: ffn_gate_inp_shexp (batch): {err}", lw.il)))?;
    for t in 0..nt {
        let sgw = hyper::sigmoid(sg[t * gn]);
        for d in 0..e {
            out[t * e + d] += sh[t * e + d] * sgw;
        }
    }
    moe_sub_add(2, t_shared.elapsed());

    Ok(out)
}

/// テスト用: 並列の寄与計算を何回走らせたか (P-MOE-G5 の分岐が実際に通ったことの証拠)。
#[cfg(test)]
pub(crate) static PAR_CONTRIB_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 選択 expert の寄与 `h_e` (長さ `e = n_embd`) を rayon で並列に計算する (P-MOE-G5)。
///
/// 返り値の `hs[t]` は `sel[t]` の寄与で、**足し込みは呼び出し側が index 昇順に
/// 逐次行う** (丸めの順序を変えないため。ここで足してはいけない)。
///
/// 内側は [`ExpertW::gemv_seq`] (行並列なし) を使う。外側が expert 並列なので、
/// 内側でも行を分けるとスレッドが二重に立つ。行分割の有無は各要素の計算順序を
/// 変えないので、逐次版とビット一致する。
#[cfg(feature = "par")]
fn expert_contribs_par(
    lw: &LayerW,
    sel: &[usize],
    x: &[f32],
    e: usize,
    ff: usize,
) -> Result<Vec<Vec<f32>>> {
    use rayon::prelude::*;
    sel.par_iter()
        .map(|&ei| {
            let mut act = vec![0.0f32; ff];
            let mut up = vec![0.0f32; ff];
            let mut h = vec![0.0f32; e];
            lw.gate_exps.gemv_seq(ei, x, &mut act)?;
            silu_inplace(&mut act);
            lw.up_exps.gemv_seq(ei, x, &mut up)?;
            for r in 0..ff {
                act[r] *= up[r];
            }
            lw.down_exps.gemv_seq(ei, &act, &mut h)?;
            #[cfg(test)]
            PAR_CONTRIB_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(h)
        })
        .collect()
}

/// キャッシュ経路の 1 expert GEMV (P-MOE-G1)。
///
/// `w` はオフセット・量子型・形状の出所で、バイト列は `c` から取る。
/// 使う範囲は [`ExpertW::slice_abs`] = mmap 経路の [`ExpertW::slice`] と同じ
/// [`ExpertSlice`] なので、2 経路が読むバイト列は同一になる。
fn expert_gemv_cached(
    c: &mut ExpertCacheSet,
    m: ExpertMatrix,
    w: &ExpertW,
    key: ExpertKey,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let (off, len) = w.slice_abs(key.expert as usize)?;
    let bytes = c
        .get(m, w.file, key, off, len)
        .map_err(|err| one(format!("{}: expert {} (cache): {err}", w.name, key.expert)))?;
    moe_expert::expert_gemv_any(w.ty, bytes, w.n_in, w.n_out, x, out)
        .map_err(|err| one(format!("{}: expert {} (cache): {err}", w.name, key.expert)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ty_name_covers_manifest_types() {
        // 目録に出る型の表記がそのまま出ること (エラーメッセージの読み手は目録)
        assert_eq!(ty_name(GgmlType::Q4K), "Q4_K");
        assert_eq!(ty_name(GgmlType::Q5_1), "Q5_1");
        assert_eq!(ty_name(GgmlType::Q8_0), "Q8_0");
        assert_eq!(ty_name(GgmlType::IQ4Nl), "IQ4_NL");
        assert_eq!(ty_name(GgmlType::Bf16), "BF16");
        assert_eq!(ty_name(GgmlType::F32), "F32");
    }

    #[test]
    fn error_display_is_one_line_per_item() {
        let e = Qwen4ExpError(vec!["a: 1".to_string(), "b: 2".to_string()]);
        assert_eq!(e.to_string(), "a: 1\nb: 2");
        assert_eq!(e.to_string().lines().count(), 2);
    }

    #[test]
    fn expected_dims_renders_min_last() {
        let e = ExpectedTensor {
            name: "t".to_string(),
            dims: vec![160, 40],
            ty: GgmlType::IQ4Nl,
            ty_alt: &[],
            min_last_dim: true,
        };
        assert_eq!(e.fmt_dims(), "[160, 40 以上]");
    }

    // -----------------------------------------------------------------
    // P-MOE-G1: expert キャッシュ経路 (合成 GGUF)
    // -----------------------------------------------------------------

    use std::path::{Path, PathBuf};

    use crate::model::expert_cache::CacheStats;

    /// `moe_ffn_quant` (mmap 経路 = 並列分岐に入りうる) を回すテストを直列化する。
    /// P-MOE-G5 のカウンタ (PAR_CONTRIB_CALLS / SEQ_CALLS) はプロセス全体の
    /// 静的値なので、並行に走る別テストの寄与を混ぜないため。
    static MOE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn moe_test_guard() -> std::sync::MutexGuard<'static, ()> {
        MOE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 合成 GGUF のデータ部アラインメント (`gguf.rs` の既定と同じ)。
    const ALIGN: usize = 32;
    const GT_U32: u32 = 4;
    const GT_STRING: u32 = 8;
    /// ggml_type id
    const TY_Q4_K: u32 = 12;
    const TY_Q5_1: u32 = 7;

    /// 合成モデルの次元 (Q4_K は ne[0] が 256 の倍数である必要がある)。
    const N_EMBD: usize = 256;
    const N_FF: usize = 640;
    const N_EXPERT: usize = 4;

    enum Kv {
        U32(u32),
        Str(&'static str),
    }

    impl Kv {
        fn write(&self, b: &mut Vec<u8>) {
            match self {
                Kv::U32(v) => {
                    b.extend_from_slice(&GT_U32.to_le_bytes());
                    b.extend_from_slice(&v.to_le_bytes());
                }
                Kv::Str(s) => {
                    b.extend_from_slice(&GT_STRING.to_le_bytes());
                    push_str(b, s);
                }
            }
        }
    }

    fn push_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    fn align_up(v: usize, a: usize) -> usize {
        (v + a - 1) & !(a - 1)
    }

    /// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列。
    fn write_gguf(
        path: &Path,
        kvs: &[(&str, Kv)],
        tensors: &[(String, Vec<u64>, u32, Vec<u8>)],
    ) {
        let mut head = Vec::new();
        head.extend_from_slice(b"GGUF");
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            push_str(&mut head, k);
            v.write(&mut head);
        }

        let mut offs = Vec::with_capacity(tensors.len());
        let mut cur = 0usize;
        for (_, _, _, data) in tensors {
            offs.push(cur as u64);
            cur = align_up(cur + data.len(), ALIGN);
        }
        for ((name, dims, ty, _), off) in tensors.iter().zip(&offs) {
            push_str(&mut head, name);
            head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&ty.to_le_bytes());
            head.extend_from_slice(&off.to_le_bytes());
        }

        let mut out = head;
        out.resize(align_up(out.len(), ALIGN), 0);
        for (_, _, _, data) in tensors {
            out.extend_from_slice(data);
            out.resize(align_up(out.len(), ALIGN), 0);
        }
        std::fs::write(path, out).expect("合成 GGUF の書き出し");
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// Q4_K (144 B = 256 要素) の 1 expert ぶん。expert ごとに値が違うこと。
    fn q4k_expert_bytes(n_in: usize, n_out: usize, e: usize) -> Vec<u8> {
        let nb = n_in / 256;
        let mut v = Vec::with_capacity(n_out * nb * 144);
        for r in 0..n_out {
            for b in 0..nb {
                v.extend_from_slice(&f16_bytes(0.02 + 0.001 * ((r + b + e) % 7) as f32));
                v.extend_from_slice(&f16_bytes(0.0005));
                for i in 0..12 {
                    v.push((0x11 + ((r * 3 + i + e) % 6)) as u8);
                }
                for i in 0..128 {
                    v.push(((i * 7 + r * 13 + b * 3 + e * 5) % 251) as u8);
                }
            }
        }
        v
    }

    /// Q5_1 (24 B = 32 要素) の 1 expert ぶん。
    fn q5_1_expert_bytes(n_in: usize, n_out: usize, e: usize) -> Vec<u8> {
        let nb = n_in / 32;
        let mut v = Vec::with_capacity(n_out * nb * 24);
        for r in 0..n_out {
            for b in 0..nb {
                v.extend_from_slice(&f16_bytes(0.02 + 0.001 * ((r + b + e) % 5) as f32));
                v.extend_from_slice(&f16_bytes(0.0004));
                let qh = ((r * 37 + b * 11 + e * 7) as u32) & 0xffff
                    | (((r as u32) & 0xffff) << 16);
                v.extend_from_slice(&qh.to_le_bytes());
                for i in 0..16 {
                    v.push(((i * 13 + r * 5 + b * 3 + e * 11) % 251) as u8);
                }
            }
        }
        v
    }

    /// テスト用の一時ディレクトリ (drop で消す)。
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let n = N.fetch_add(1, Ordering::Relaxed);
            let p = std::env::temp_dir().join(format!(
                "denno_q4exp_cache_{tag}_{}_{n}",
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

    /// 実モデルと同じ型の expert 3 本 (Q4_K gate/up・Q5_1 down・4 expert) を持つ
    /// 合成 GGUF を書く。
    fn write_expert_gguf(dir: &TempDir) -> PathBuf {
        let path = dir.path("experts.gguf");
        let (n_in, n_ff, ne) = (N_EMBD, N_FF, N_EXPERT);
        let mut gate = Vec::new();
        let mut up = Vec::new();
        let mut down = Vec::new();
        for e in 0..ne {
            gate.extend_from_slice(&q4k_expert_bytes(n_in, n_ff, e));
            up.extend_from_slice(&q4k_expert_bytes(n_in, n_ff, e + 7));
            down.extend_from_slice(&q5_1_expert_bytes(n_ff, n_in, e + 3));
        }
        let tensors = vec![
            (
                "blk.0.ffn_gate_exps.weight".to_string(),
                vec![n_in as u64, n_ff as u64, ne as u64],
                TY_Q4_K,
                gate,
            ),
            (
                "blk.0.ffn_up_exps.weight".to_string(),
                vec![n_in as u64, n_ff as u64, ne as u64],
                TY_Q4_K,
                up,
            ),
            (
                "blk.0.ffn_down_exps.weight".to_string(),
                vec![n_ff as u64, n_in as u64, ne as u64],
                TY_Q5_1,
                down,
            ),
        ];
        let kvs = [
            ("general.architecture", Kv::Str("qwen4exp")),
            ("general.alignment", Kv::U32(ALIGN as u32)),
        ];
        write_gguf(&path, &kvs, &tensors);
        path
    }

    /// `moe_ffn_quant` が読むぶんだけ埋めた諸元 (validate は通さない)。
    fn cache_test_params() -> Qwen4ExpParams {
        Qwen4ExpParams {
            n_layer: 1,
            n_ctx: 8,
            n_embd: N_EMBD,
            n_head: 1,
            n_head_kv: 1,
            n_embd_head_k: 32,
            n_embd_head_v: 32,
            rms_eps: 1e-6,
            n_rot: 32,
            rope_sections: [0, 0, 0, 0],
            rope_freq_base: 1e7,
            n_expert: N_EXPERT,
            n_expert_used: 2,
            n_ff_exp: N_FF,
            n_ff_shexp: 64,
            ssm_d_conv: 4,
            ssm_d_state: 32,
            ssm_n_group: 1,
            ssm_dt_rank: 1,
            ssm_d_inner: 32,
            full_attn_interval: 4,
            n_hc: 1,
            hc_low_rank: 32,
            indexer_n_head: 1,
            indexer_head_size: 32,
            indexer_top_k: 1,
            compress_ratios: vec![0],
            n_embd_per_layer: 0,
            ple_layers: Vec::new(),
            ple_ngram_size: 0,
            ple_heads_per_ngram: 0,
            ple_conv_kernel: 0,
            ple_eos_token_id: 0,
            ple_image_token_id: 0,
            ple_layer_multipliers: Vec::new(),
            ple_head_offsets: Vec::new(),
            ple_head_vocab_sizes: Vec::new(),
            n_vocab: 8,
        }
    }

    /// 決定的な f32 の重み (F32 の GgufLinear)。
    fn f32_linear(k: usize, n: usize, seed: f32) -> GgufLinear {
        let mut v = Vec::with_capacity(k * n * 4);
        for i in 0..k * n {
            let x = seed + (i % 17) as f32 * 0.01;
            v.extend_from_slice(&x.to_le_bytes());
        }
        GgufLinear::new(GgmlType::F32, k, n, WeightBlocks::from_vec(v))
            .expect("GgufLinear::new (F32)")
    }

    /// `moe_ffn_quant` が触るフィールドだけ実体を入れた層。
    /// hc_* は MoE では使わないので 1x1 のダミー。
    fn cache_test_layer(f: &GgufFile, p: &Qwen4ExpParams) -> LayerW {
        let name = |s: &str| format!("blk.0.{s}");
        LayerW {
            il: 0,
            hc_attn_norm: vec![1.0; p.n_embd],
            hc_attn_down: f32_linear(1, 1, 0.5),
            hc_attn_up: f32_linear(1, 1, 0.5),
            hc_attn_inject: f32_linear(1, 1, 0.5),
            hc_ffn_norm: vec![1.0; p.n_embd],
            hc_ffn_down: f32_linear(1, 1, 0.5),
            hc_ffn_up: f32_linear(1, 1, 0.5),
            hc_ffn_inject: f32_linear(1, 1, 0.5),
            gate_inp: f32_linear(p.n_embd, p.n_expert, 0.1),
            gate_inp_shexp: f32_linear(p.n_embd, 1, 0.2),
            gate_exps: ExpertW::load(f, &name("ffn_gate_exps.weight"), p).expect("gate"),
            up_exps: ExpertW::load(f, &name("ffn_up_exps.weight"), p).expect("up"),
            down_exps: ExpertW::load(f, &name("ffn_down_exps.weight"), p).expect("down"),
            gate_shexp: f32_linear(p.n_embd, p.n_ff_shexp, 0.3),
            up_shexp: f32_linear(p.n_embd, p.n_ff_shexp, 0.4),
            down_shexp: f32_linear(p.n_ff_shexp, p.n_embd, 0.5),
            ple: None,
        }
    }

    /// 行列ごとの 1 expert のバイト数 (合成 GGUF の実値)。
    fn expert_lens(lw: &LayerW) -> [usize; 3] {
        [
            lw.gate_exps.slice_abs(0).expect("gate len").1,
            lw.up_exps.slice_abs(0).expect("up len").1,
            lw.down_exps.slice_abs(0).expect("down len").1,
        ]
    }

    fn cache_test_x(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
            .collect()
    }

    fn assert_bitwise_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: 長さが違う");
        for i in 0..a.len() {
            assert_eq!(
                a[i].to_bits(),
                b[i].to_bits(),
                "{what}: 要素 {i} がビット一致しない (mmap {} / cache {})",
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn moe_par_gate_value_parsing() {
        // DENNO_MOE_PAR の解釈 (既定 1)。env を触らずに純粋関数だけを見る。
        assert!(moe_par_from_value(None), "未設定は既定で有効");
        assert!(moe_par_from_value(Some("1")));
        assert!(moe_par_from_value(Some("8")));
        assert!(!moe_par_from_value(Some("0")), "\"0\" で無効");
        assert!(
            moe_par_from_value(Some("")),
            "空文字は「未設定」と同じ扱い (有効)"
        );
    }

    /// P-MOE-G5: expert ループの並列化は数値を 1 ビットも変えない。
    ///
    /// 同じ合成 expert で `par=0` (従来の逐次) と `par=1` (expert 並列) を回し、
    /// `to_bits()` が全要素一致することを見る。併せて「並列分岐が実際に通った」
    /// 「内側が行並列なしの入口を通った」をカウンタで確かめる (緑 = 発動の証拠ではない)。
    #[test]
    fn moe_expert_loop_par_is_bit_identical_to_seq() {
        let _g = moe_test_guard();
        let dir = TempDir::new("par");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        assert!(
            p.n_expert_used > 1,
            "並列経路を通すには選択 expert が 2 個以上必要 (n_expert_used = {})",
            p.n_expert_used
        );

        // 1. 従来経路 (DENNO_MOE_PAR=0 相当)
        let seq = moe_ffn_quant_with_par(&p, &lw, &x, None, false).expect("逐次 (par=0)");
        assert!(seq.iter().all(|v| v.is_finite()), "逐次経路に NaN/inf");
        assert!(seq.iter().any(|v| *v != 0.0), "逐次経路が全 0");

        // 2. 並列経路 (DENNO_MOE_PAR=1 相当)
        let par_before = PAR_CONTRIB_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let seq_before = moe_expert::SEQ_CALLS.load(std::sync::atomic::Ordering::Relaxed);
        let par = moe_ffn_quant_with_par(&p, &lw, &x, None, true).expect("並列 (par=1)");
        let par_calls = PAR_CONTRIB_CALLS.load(std::sync::atomic::Ordering::Relaxed) - par_before;
        let seq_calls = moe_expert::SEQ_CALLS.load(std::sync::atomic::Ordering::Relaxed) - seq_before;

        // 3. 並列分岐が発動したこと (top-k の数だけ寄与計算が走る)
        #[cfg(feature = "par")]
        {
            assert_eq!(
                par_calls, p.n_expert_used,
                "並列の寄与計算が選択 expert 数ぶん走っていない (par=1 で並列分岐に入っていない)"
            );
            // 内側は gemv_seq = 1 expert あたり gate/up/down の 3 本
            assert_eq!(
                seq_calls,
                3 * p.n_expert_used,
                "並列時は内側が行並列なしの入口 (gemv_seq) を通るはず"
            );
        }
        #[cfg(not(feature = "par"))]
        {
            assert_eq!(par_calls, 0, "par 機能なしで並列分岐に入った");
            assert_eq!(seq_calls, 0, "par 機能なしで逐次入口を通った");
        }

        // 4. ビット一致
        assert_bitwise_eq(&seq, &par, "par=0 vs par=1");
    }

    #[test]
    fn expert_cache_path_is_bit_identical_to_mmap() {
        let _g = moe_test_guard();
        let dir = TempDir::new("bit");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);

        // 1. 従来経路 (mmap + prefetch)
        let want = moe_ffn_quant(&p, &lw, &x, None).expect("mmap 経路");
        assert!(want.iter().all(|v| v.is_finite()), "mmap 経路に NaN/inf");
        assert!(want.iter().any(|v| *v != 0.0), "mmap 経路が全 0");

        // 2. キャッシュ経路 (2 スロット)
        let lens = expert_lens(&lw);
        let mut set =
            ExpertCacheSet::new_with_slots([2, 2, 2], lens, &[path.clone()], 2).expect("cache set");
        assert_eq!(set.slots(ExpertMatrix::Gate), 2, "スロット数が 2 でない");
        let got = moe_ffn_quant(&p, &lw, &x, Some(&mut set)).expect("cache 経路");
        assert_bitwise_eq(&want, &got, "mmap vs cache");

        // 3. 初回は全ミス (top-2 なので行列ごとに 2 回読む)
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!(s.hits, 0, "{m:?}: 初回でヒットした");
            assert_eq!(s.misses, 2, "{m:?}: 初回のミス数");
            assert_eq!(s.bytes_read, 2 * lens[k] as u64, "{m:?}: 読んだバイト数");
            assert_eq!(s.evictions, 0, "{m:?}: 2 スロットに top-2 で追い出しは起きない");
        }

        // 4. 同じ x をもう一度 → 同じ expert が選ばれるのでヒット (ミスは増えない)
        let before: Vec<u64> = ExpertMatrix::ALL.iter().map(|&m| set.stats(m).misses).collect();
        let again = moe_ffn_quant(&p, &lw, &x, Some(&mut set)).expect("cache 経路 2 回目");
        assert_bitwise_eq(&want, &again, "mmap vs cache (2 回目)");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!(s.misses, before[k], "{m:?}: 同じ expert なのにミスが増えた");
            assert_eq!(s.hits, 2, "{m:?}: 2 回目でヒットしていない");
        }
        assert_eq!(set.hit_rate(ExpertMatrix::Gate), 0.5, "ヒット率");
    }

    #[test]
    fn one_slot_cache_keeps_evicting() {
        let dir = TempDir::new("one");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let lens = expert_lens(&lw);

        let mut set =
            ExpertCacheSet::new_with_slots([1, 1, 1], lens, &[path.clone()], 1).expect("cache set");
        for m in ExpertMatrix::ALL {
            assert_eq!(set.slots(m), 1, "{m:?}: スロット数が 1 でない");
        }

        // 1 スロットで top-2 → 1 回の forward で必ず追い出しが起きる
        moe_ffn_quant(&p, &lw, &x, Some(&mut set)).expect("1 回目");
        let first: Vec<(u64, u64)> = ExpertMatrix::ALL
            .iter()
            .map(|&m| (set.stats(m).misses, set.stats(m).evictions))
            .collect();
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let (misses, evictions) = first[k];
            assert_eq!(misses, 2, "{m:?}: 1 回目のミス数");
            assert_eq!(evictions, 1, "{m:?}: 1 回目の追い出し数");
        }

        // 2 回目も同じ expert を引くが、1 スロットなので前回の 2 個目しか残っていない
        // → 少なくとも 1 回は必ずミスし、追い出しが増える
        moe_ffn_quant(&p, &lw, &x, Some(&mut set)).expect("2 回目");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert!(s.misses > first[k].0, "{m:?}: 2 回目でミスが増えていない");
            assert!(s.evictions > first[k].1, "{m:?}: 2 回目で追い出しが増えていない");
        }
    }

    #[test]
    fn zero_slots_still_returns_mmap_bytes() {
        let _g = moe_test_guard();
        let dir = TempDir::new("zero");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let lens = expert_lens(&lw);

        let want = moe_ffn_quant(&p, &lw, &x, None).expect("mmap 経路");
        let mut set =
            ExpertCacheSet::new_with_slots([0, 0, 0], lens, &[path.clone()], 0).expect("cache set");
        for m in ExpertMatrix::ALL {
            assert_eq!(set.slots(m), 0, "{m:?}: スロット数が 0 でない");
        }
        let got = moe_ffn_quant(&p, &lw, &x, Some(&mut set)).expect("slots=0 経路");
        assert_bitwise_eq(&want, &got, "mmap vs slots=0");
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            let s = set.stats(m);
            assert_eq!((s.hits, s.misses), (0, 2), "{m:?}: slots=0 の統計");
            assert_eq!(s.bytes_read, 2 * lens[k] as u64, "{m:?}: 読んだバイト数");
        }
    }

    #[test]
    fn build_expert_caches_takes_mb_and_per_matrix_max() {
        let dir = TempDir::new("build");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let layers = vec![cache_test_layer(&f, &p)];
        let lens = expert_lens(&layers[0]);

        // MB = 0 は「無効」= 構築しない (mmap 経路のまま)
        assert!(
            build_expert_caches_with_mb(0, &f, &layers)
                .expect("mb=0")
                .is_none(),
            "mb=0 でキャッシュを組んだ"
        );

        // MB = 4: 行列ごとに 4 MiB / slot_bytes (切り捨て)
        let set = build_expert_caches_with_mb(4, &f, &layers)
            .expect("mb=4")
            .expect("キャッシュが組まれていない");
        assert_eq!(set.mb(), 4);
        for (k, &m) in ExpertMatrix::ALL.iter().enumerate() {
            assert_eq!(set.slot_bytes(m), lens[k], "{m:?}: slot_bytes");
            assert_eq!(
                set.slots(m),
                4 * 1024 * 1024 / lens[k],
                "{m:?}: スロット数が MB の式と違う"
            );
            assert_eq!(set.stats(m), CacheStats::default(), "{m:?}: 初期統計");
        }
    }

    #[test]
    fn slot_bytes_below_the_largest_expert_is_an_error() {
        // 静かに切り詰めないこと (Q5_1 down の 1 expert より小さいスロット)
        let dir = TempDir::new("small");
        let path = write_expert_gguf(&dir);
        let f = GgufFile::open(&path).expect("open");
        let p = cache_test_params();
        let lw = cache_test_layer(&f, &p);
        let x = cache_test_x(p.n_embd);
        let mut lens = expert_lens(&lw);
        lens[2] -= 1; // down だけ 1 バイト足りない

        let mut set =
            ExpertCacheSet::new_with_slots([2, 2, 2], lens, &[path.clone()], 2).expect("cache set");
        let err = moe_ffn_quant(&p, &lw, &x, Some(&mut set)).expect_err("切り詰めて通った");
        assert!(
            err.to_string().contains("1 スロット"),
            "エラーがスロット超過でない: {err}"
        );
    }

    // -----------------------------------------------------------------
    // H2: prefill のバッチ化 (層優先) — 逐次経路とのビット一致
    //
    // 合成モデルは**目録 (`expected_tensors`) から組む**: テンソル名・形状・
    // 量子型をここに写さないので、目録が変わればテストも自動で追随する。
    // -----------------------------------------------------------------

    mod h2 {
        use super::*;
        use std::sync::atomic::Ordering;

        // 次元 (tests/qwen4exp_fwd.rs の合成モデルと同じ値。目録の下限を満たす最小)
        const E: usize = 256;
        const VOCAB: usize = 8;
        const NHEAD: usize = 2;
        const HEAD: usize = 256;
        const NKV: usize = 1;
        const NEXP: usize = 2;
        const FF: usize = 256;
        const HC: usize = 2;
        const HC_LR: usize = 32;
        const SSM_CONV: usize = 4;
        const SSM_STATE: usize = 32;
        const SSM_GROUP: usize = 2;
        const SSM_DT: usize = 2;
        const SSM_INNER: usize = 64;
        const INTERVAL: usize = 4;
        const IDX_HEAD: usize = 2;
        const IDX_DIM: usize = 32;
        const IDX_TOP_K: usize = 8;
        const RATIO: u32 = 2;
        const PLE_HEADS_PER_NGRAM: usize = 2;
        const EPL: usize = 64;
        const PLE_LAYER: usize = 1;
        const PLE_NGRAM: usize = 3;
        const PLE_CONV: usize = 4;
        const PLE_ROWS: u64 = 40;
        const LAYERS: usize = 4;

        const GT_F32: u32 = 6;
        const GT_ARRAY: u32 = 9;
        const GT_U64: u32 = 10;

        enum V {
            U32(u32),
            F32(f32),
            S(&'static str),
            A32(Vec<u32>),
            A64(Vec<u64>),
            AStr(Vec<String>),
        }

        fn push_str(b: &mut Vec<u8>, s: &str) {
            b.extend_from_slice(&(s.len() as u64).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
        }

        /// 決定的な擬似乱数バイト (0..=255)。
        fn pat(i: usize, seed: u32) -> u8 {
            let mut x = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            x ^= x >> 13;
            x = x.wrapping_mul(0x9E37_79B1);
            x ^= x >> 16;
            x as u8
        }

        fn write_meta(b: &mut Vec<u8>, k: &str, v: &V) {
            push_str(b, k);
            match v {
                V::U32(x) => {
                    b.extend_from_slice(&GT_U32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::F32(x) => {
                    b.extend_from_slice(&GT_F32.to_le_bytes());
                    b.extend_from_slice(&x.to_le_bytes());
                }
                V::S(s) => {
                    b.extend_from_slice(&GT_STRING.to_le_bytes());
                    push_str(b, s);
                }
                V::A32(v) => {
                    b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                    b.extend_from_slice(&GT_U32.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in v {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::A64(v) => {
                    b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                    b.extend_from_slice(&GT_U64.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for x in v {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                }
                V::AStr(v) => {
                    b.extend_from_slice(&GT_ARRAY.to_le_bytes());
                    b.extend_from_slice(&GT_STRING.to_le_bytes());
                    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
                    for s in v {
                        push_str(b, s);
                    }
                }
            }
        }

        fn kvs() -> Vec<(&'static str, V)> {
            let mut ratios = vec![0u32; LAYERS];
            for (il, r) in ratios.iter_mut().enumerate() {
                if (il + 1) % INTERVAL == 0 {
                    *r = RATIO;
                }
            }
            vec![
                ("general.architecture", V::S("qwen4exp")),
                ("general.alignment", V::U32(ALIGN as u32)),
                ("qwen4exp.block_count", V::U32(LAYERS as u32)),
                ("qwen4exp.context_length", V::U32(1024)),
                ("qwen4exp.embedding_length", V::U32(E as u32)),
                ("qwen4exp.attention.head_count", V::U32(NHEAD as u32)),
                ("qwen4exp.attention.head_count_kv", V::U32(NKV as u32)),
                ("qwen4exp.attention.key_length", V::U32(HEAD as u32)),
                ("qwen4exp.attention.value_length", V::U32(HEAD as u32)),
                ("qwen4exp.attention.layer_norm_rms_epsilon", V::F32(1e-6)),
                ("qwen4exp.rope.dimension_count", V::U32(4)),
                ("qwen4exp.rope.dimension_sections", V::A32(vec![2, 2, 2, 0])),
                ("qwen4exp.rope.freq_base", V::F32(1e7)),
                ("qwen4exp.expert_count", V::U32(NEXP as u32)),
                // 2 expert を両方選ばせる: MoE のバッチ経路 (同じ expert を選んだ
                // トークンをまとめる) を実際に通すため。1 だと 1 トークン 1 expert で
                // 逐次版と同じ入口に落ち、まとめる経路が一度も走らない。
                ("qwen4exp.expert_used_count", V::U32(NEXP as u32)),
                ("qwen4exp.expert_feed_forward_length", V::U32(FF as u32)),
                (
                    "qwen4exp.expert_shared_feed_forward_length",
                    V::U32(FF as u32),
                ),
                ("qwen4exp.ssm.conv_kernel", V::U32(SSM_CONV as u32)),
                ("qwen4exp.ssm.state_size", V::U32(SSM_STATE as u32)),
                ("qwen4exp.ssm.group_count", V::U32(SSM_GROUP as u32)),
                ("qwen4exp.ssm.time_step_rank", V::U32(SSM_DT as u32)),
                ("qwen4exp.ssm.inner_size", V::U32(SSM_INNER as u32)),
                ("qwen4exp.full_attention_interval", V::U32(INTERVAL as u32)),
                ("qwen4exp.hyper_connection.count", V::U32(HC as u32)),
                ("qwen4exp.hyper_connection.low_rank", V::U32(HC_LR as u32)),
                ("qwen4exp.attention.indexer.head_count", V::U32(IDX_HEAD as u32)),
                ("qwen4exp.attention.indexer.key_length", V::U32(IDX_DIM as u32)),
                ("qwen4exp.attention.indexer.top_k", V::U32(IDX_TOP_K as u32)),
                ("qwen4exp.attention.compress_ratios", V::A32(ratios)),
                ("qwen4exp.embedding_length_per_layer_input", V::U32(EPL as u32)),
                ("qwen4exp.ple.layers", V::A32(vec![PLE_LAYER as u32])),
                ("qwen4exp.ple.ngram_size", V::U32(PLE_NGRAM as u32)),
                (
                    "qwen4exp.ple.heads_per_ngram",
                    V::U32(PLE_HEADS_PER_NGRAM as u32),
                ),
                ("qwen4exp.ple.conv_kernel", V::U32(PLE_CONV as u32)),
                ("qwen4exp.ple.eos_token_id", V::U32(7)),
                ("qwen4exp.ple.image_token_id", V::U32(9)),
                ("qwen4exp.ple.layer_multipliers", V::A64(vec![1, 2, 3])),
                ("qwen4exp.ple.head_offsets", V::A64(vec![0, 10, 20, 30])),
                ("qwen4exp.ple.head_vocab_sizes", V::A64(vec![10, 10, 10, 10])),
                ("tokenizer.ggml.tokens", V::AStr(vec![String::new(); VOCAB])),
            ]
        }

        /// 最小 GGUF (v3) を書く。データ部は 32 B 整列。
        fn write_model(path: &Path, tensors: &[(String, Vec<u64>, u32, Vec<u8>)]) {
            let ks = kvs();
            let mut head = Vec::new();
            head.extend_from_slice(b"GGUF");
            head.extend_from_slice(&3u32.to_le_bytes());
            head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            head.extend_from_slice(&(ks.len() as u64).to_le_bytes());
            for (k, v) in &ks {
                write_meta(&mut head, k, v);
            }

            let mut offs = Vec::with_capacity(tensors.len());
            let mut cur = 0usize;
            for (_, _, _, data) in tensors {
                offs.push(cur as u64);
                cur = align_up(cur + data.len(), ALIGN);
            }
            for ((name, dims, ty, _), off) in tensors.iter().zip(&offs) {
                push_str(&mut head, name);
                head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
                for d in dims {
                    head.extend_from_slice(&d.to_le_bytes());
                }
                head.extend_from_slice(&ty.to_le_bytes());
                head.extend_from_slice(&off.to_le_bytes());
            }

            let mut out = head;
            out.resize(align_up(out.len(), ALIGN), 0);
            for (_, _, _, data) in tensors {
                out.extend_from_slice(data);
                out.resize(align_up(out.len(), ALIGN), 0);
            }
            std::fs::write(path, out).expect("合成 GGUF の書き出し");
        }

        /// 目録の量子型 → ggml_type id。
        fn ggml_id(ty: GgmlType) -> u32 {
            match ty {
                GgmlType::F32 => 0,
                GgmlType::F16 => 1,
                GgmlType::Q5_1 => 7,
                GgmlType::Q8_0 => 8,
                GgmlType::Q4K => 12,
                GgmlType::Q5K => 13,
                GgmlType::IQ4Nl => 20,
                GgmlType::Bf16 => 30,
                other => panic!("テストヘルパ未対応の型 {other:?}"),
            }
        }

        /// テンソルの中身。**有限値だけ**を作る (スケール欄を乱数にすると NaN になる)。
        fn tensor_bytes(name: &str, dims: &[u64], ty: GgmlType, seed: u32) -> Vec<u8> {
            let ne: u64 = dims.iter().product();
            let bs = ty.blck_size() as u64;
            let ts = ty.type_size() as u64;
            let nb = (ne / bs * ts + if ne % bs == 0 { 0 } else { ts }) as usize;
            let mut b: Vec<u8> = (0..nb).map(|i| pat(i, 7 + seed)).collect();
            match ty {
                GgmlType::F32 => {
                    // norm の gamma は 1.0 (変換器が 1 + w に畳んだ前提の値)
                    let gamma = name.ends_with("norm.weight");
                    for i in 0..nb / 4 {
                        let v = if gamma {
                            1.0f32
                        } else {
                            (((i as u32).wrapping_add(seed * 5) % 17) as f32 - 8.0) * 0.01
                        };
                        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                }
                GgmlType::Bf16 => {
                    for i in 0..nb / 2 {
                        let v = (((i as u32).wrapping_add(seed * 3) % 13) as f32 - 6.0) * 0.01;
                        let bits = (v.to_bits() >> 16) as u16;
                        b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
                GgmlType::Q8_0 => {
                    for blk in 0..nb / 34 {
                        let o = blk * 34;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        for j in 0..32u32 {
                            // ブロックごとに量子値をずらす (行ごとに違う値になる)
                            let q = ((j as i32 + blk as i32 * 3 + seed as i32 * 7) % 11) - 5;
                            b[o + 2 + j as usize] = q as i8 as u8;
                        }
                    }
                }
                GgmlType::Q5_1 => {
                    for blk in 0..nb / 24 {
                        let o = blk * 24;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
                    }
                }
                GgmlType::Q4K => {
                    for blk in 0..nb / 144 {
                        let o = blk * 144;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
                    }
                }
                GgmlType::Q5K => {
                    // 176 B = d(2) + dmin(2) + scales(12) + qh(32) + qs(128)
                    for blk in 0..nb / 176 {
                        let o = blk * 176;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                        b[o + 2..o + 4].copy_from_slice(&f16_bytes(0.001));
                    }
                }
                GgmlType::IQ4Nl => {
                    for blk in 0..nb / 18 {
                        let o = blk * 18;
                        b[o..o + 2].copy_from_slice(&f16_bytes(0.02));
                    }
                }
                other => panic!("テストヘルパ未対応の型 {other:?}"),
            }
            b
        }

        /// メタデータだけの GGUF から params を取って、目録の全テンソルを
        /// 決定的な中身で埋めた GGUF を書く。
        fn build(dir: &TempDir) -> PathBuf {
            let meta = dir.path("h2_meta.gguf");
            write_model(&meta, &[]);
            let f = GgufFile::open(&meta).expect("メタデータ GGUF を開けない");
            let p = Qwen4ExpParams::from_meta(&f).expect("from_meta が通らない");
            let specs = expected_tensors(&p).expect("目録を組めない");
            let mut tensors = Vec::with_capacity(specs.len());
            for (i, e) in specs.iter().enumerate() {
                let data = tensor_bytes(&e.name, &e.dims, e.ty, i as u32 + 1);
                tensors.push((e.name.clone(), e.dims.clone(), ggml_id(e.ty), data));
            }
            let path = dir.path("h2_model.gguf");
            write_model(&path, &tensors);
            path
        }

        /// 1 回ぶんの prefill を新しいモデルで回し、logits と
        /// 「バッチ分岐 / バッチ GEMV が何回走ったか」を返す。
        fn run(f: &GgufFile, ctx: &[u32], batch: bool, chunk: usize) -> (Vec<f32>, usize, usize) {
            let mut m = Qwen4Exp::new(f).expect("Qwen4Exp::new");
            let b0 = PREFILL_BATCH_CALLS.load(Ordering::Relaxed);
            let g0 = GEMV_BATCH_CALLS.load(Ordering::Relaxed);
            let out = if batch {
                m.forward_prefill_batch(ctx, chunk).expect("バッチ prefill")
            } else {
                m.forward_prefill_with(ctx, false).expect("逐次 prefill")
            };
            (
                out,
                PREFILL_BATCH_CALLS.load(Ordering::Relaxed) - b0,
                GEMV_BATCH_CALLS.load(Ordering::Relaxed) - g0,
            )
        }

        #[test]
        fn prefill_batch_gate_defaults_off() {
            // DENNO_PREFILL_BATCH の解釈 (既定 off)。env を触らずに純粋関数だけを見る。
            assert!(!prefill_batch_from_value(None), "未設定は既定で無効");
            assert!(prefill_batch_from_value(Some("1")));
            assert!(!prefill_batch_from_value(Some("0")));
            assert!(!prefill_batch_from_value(Some("")));
            assert!(
                !prefill_batch_from_value(Some("true")),
                "\"1\" 以外は無効 (既定 off を崩さない)"
            );
        }

        /// hc_mix のバッチ版が逐次版とビット一致する (要素ごとの演算は同じ関数)。
        #[test]
        fn hc_mix_batch_matches_seq() {
            let hp = HyperParams {
                hc: 2,
                n_embd: 8,
                low_rank: 4,
                eps: 1e-6,
            };
            let hc_dim = hp.hc * hp.n_embd;
            let norm = vec![1.0f32; hc_dim];
            let down = f32_linear(hc_dim, hp.low_rank, 0.1);
            let up = f32_linear(hp.low_rank, hc_dim, 0.2);
            let inject = f32_linear(hc_dim, hp.hc, 0.3);
            let ress: Vec<HyperStreams> = (0..3)
                .map(|t| HyperStreams {
                    data: (0..hc_dim)
                        .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                        .collect(),
                    hc: hp.hc,
                    n_embd: hp.n_embd,
                })
                .collect();

            let (mixed, injs) = hc_mix_quant_batch(
                &ress,
                &norm,
                &down,
                &up,
                Some(&inject),
                &hp,
                &mut MixScratch::default(),
            )
            .expect("hc_mix_quant_batch");
            for (t, r) in ress.iter().enumerate() {
                let (m, i) = hc_mix_quant(r, &norm, &down, &up, Some(&inject), &hp)
                    .expect("hc_mix_quant");
                assert_bitwise_eq(&m, &mixed[t], &format!("hc_mix mixed (token {t})"));
                assert_bitwise_eq(
                    &i,
                    &injs[t * hp.hc..(t + 1) * hp.hc],
                    &format!("hc_mix inject (token {t})"),
                );
            }
            assert!(
                mixed[0].iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// hyper のバッチ版が逐次版とビット一致する。
        ///
        /// nt = 1 / 3 は逐次ループ、nt = 64 は rayon 経路
        /// ([`hyper::HC_BATCH_PAR_MIN_TOKENS`]) に入る。どちらもトークン内の
        /// 演算順序は変えていないので `to_bits()` で一致する。
        #[test]
        fn hyper_batch_matches_seq_bitwise() {
            let hp = HyperParams {
                hc: 4,
                n_embd: 16,
                low_rank: 5,
                eps: 1e-6,
            };
            let e = hp.n_embd;
            let hc_dim = hp.hc * hp.n_embd;
            assert_eq!(hc_dim, 64, "テストの前提 (hc_dim = 64)");
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.5 + ((i * 7) % 11) as f32 * 0.1).collect();

            let make = |nt: usize| -> Vec<HyperStreams> {
                (0..nt)
                    .map(|t| HyperStreams {
                        data: (0..hc_dim)
                            .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                            .collect(),
                        hc: hp.hc,
                        n_embd: e,
                    })
                    .collect()
            };

            let par_before = par_calls();
            for &nt in &[1usize, 3, 64] {
                let ress = make(nt);

                // 1. grouped RMS 正規化
                let seq_xn: Vec<f32> = ress
                    .iter()
                    .flat_map(|r| hyper::grouped_rms_norm(r, &norm, &hp))
                    .collect();
                let bat_xn = hyper::grouped_rms_norm_batch(&ress, &norm, &hp);
                assert_bitwise_eq(&seq_xn, &bat_xn, &format!("grouped_rms_norm nt={nt}"));

                // 2. lo の後処理 (silu(v/hc))
                let lo_in: Vec<f32> = (0..nt * hp.low_rank)
                    .map(|i| (((i * 11) % 19) as f32 - 9.0) * 0.02)
                    .collect();
                let mut lo_seq = lo_in.clone();
                for t in 0..nt {
                    hyper::hc_lo_apply(&mut lo_seq[t * hp.low_rank..(t + 1) * hp.low_rank], &hp);
                }
                let mut lo_bat = lo_in.clone();
                hyper::hc_lo_apply_batch(&mut lo_bat, nt, &hp);
                assert_bitwise_eq(&lo_seq, &lo_bat, &format!("hc_lo_apply nt={nt}"));

                // 3. mixed (s の昇順に加算してから 1/hc)
                let gate: Vec<f32> = (0..nt * hc_dim)
                    .map(|i| (((i * 17) % 29) as f32 - 14.0) * 0.05)
                    .collect();
                let seq_mixed: Vec<f32> = (0..nt)
                    .flat_map(|t| {
                        hyper::hc_mixed(
                            &seq_xn[t * hc_dim..(t + 1) * hc_dim],
                            &gate[t * hc_dim..(t + 1) * hc_dim],
                            &hp,
                        )
                    })
                    .collect();
                let bat_mixed = hyper::hc_mixed_batch(&seq_xn, &gate, nt, &hp);
                assert_bitwise_eq(&seq_mixed, &bat_mixed, &format!("hc_mixed nt={nt}"));

                // 4. inject の後処理 (silu 無し)
                let inj_in: Vec<f32> = (0..nt * hp.hc)
                    .map(|i| (((i * 5) % 7) as f32 - 3.0) * 0.1)
                    .collect();
                let mut inj_seq = inj_in.clone();
                for t in 0..nt {
                    hyper::hc_inject_apply(&mut inj_seq[t * hp.hc..(t + 1) * hp.hc], &hp);
                }
                let mut inj_bat = inj_in.clone();
                hyper::hc_inject_apply_batch(&mut inj_bat, nt, &hp);
                assert_bitwise_eq(&inj_seq, &inj_bat, &format!("hc_inject_apply nt={nt}"));

                // 5. combine (逐次版は新しい HyperStreams を作る / バッチ版は in-place)
                let outs: Vec<f32> = (0..nt * e)
                    .map(|i| (((i * 23) % 31) as f32 - 15.0) * 0.01)
                    .collect();
                let mut seq_res = make(nt);
                for t in 0..nt {
                    let out = hyper::hc_combine(
                        &seq_res[t],
                        &outs[t * e..(t + 1) * e],
                        &inj_seq[t * hp.hc..(t + 1) * hp.hc],
                        &hp,
                    );
                    seq_res[t] = out;
                }
                let mut bat_res = make(nt);
                hyper::hc_combine_batch(&mut bat_res, &outs, &inj_bat, &hp);
                for t in 0..nt {
                    assert_bitwise_eq(
                        &seq_res[t].data,
                        &bat_res[t].data,
                        &format!("hc_combine nt={nt} token={t}"),
                    );
                }
                assert!(
                    bat_res.iter().flat_map(|r| r.data.iter()).any(|v| *v != 0.0),
                    "combine の出力が全 0 (経路の生死が分からない)"
                );
            }

            // nt >= 64 で rayon 経路に実際に入ったことを数える
            // (feature par のときだけ。off なら逐次で回るのが正しい)。
            #[cfg(feature = "par")]
            assert!(
                par_calls() > par_before,
                "nt=64 でバッチ版の rayon 経路に入っていない (HC_BATCH_PAR_CALLS が増えない)"
            );
            #[cfg(not(feature = "par"))]
            assert_eq!(par_calls(), par_before, "par off なのに並列カウンタが増えた");
        }

        /// バッチ版が rayon 経路に入った回数 (テスト用カウンタ)。
        #[cfg(feature = "par")]
        fn par_calls() -> usize {
            hyper::HC_BATCH_PAR_CALLS.load(std::sync::atomic::Ordering::Relaxed)
        }

        #[cfg(not(feature = "par"))]
        fn par_calls() -> usize {
            0
        }

        /// マイクロベンチ (手動)。mix の**要素演算だけ**を旧ループ / バッチで比べる。
        ///
        /// 行列積 (gemv) は GPU 側の仕事なので含めない。実形 (hc=4, n_embd=2560,
        /// hc_dim=10240) で nt 別に 200 回まわし、ms/トークンを出す。
        ///
        /// `cargo test --release -p denno-engine --lib hc_mix_elementwise_microbench -- --ignored --nocapture --test-threads=1`
        #[test]
        #[ignore]
        fn hc_mix_elementwise_microbench() {
            let hp = HyperParams {
                hc: 4,
                n_embd: 2560,
                low_rank: 320,
                eps: 1e-6,
            };
            let e = hp.n_embd;
            let hc_dim = hp.hc * hp.n_embd; // 10240 (実形)
            let reps = 200usize;
            let norm: Vec<f32> = (0..hc_dim).map(|i| 0.9 + ((i % 13) as f32) * 0.01).collect();

            println!("reps={reps} hc_dim={hc_dim} low_rank={}", hp.low_rank);
            println!("nt\told_ms_per_tok\tnew_ms_per_tok\tratio(old/new)");
            for &nt in &[1usize, 8, 64, 512] {
                let base: Vec<Vec<f32>> = (0..nt)
                    .map(|t| {
                        (0..hc_dim)
                            .map(|i| (((i * 7 + t * 13) % 23) as f32 - 11.0) * 0.03)
                            .collect()
                    })
                    .collect();
                let gate: Vec<f32> = (0..nt * hc_dim)
                    .map(|i| (((i * 17) % 29) as f32 - 14.0) * 0.05)
                    .collect();
                let lo_in: Vec<f32> = (0..nt * hp.low_rank)
                    .map(|i| (((i * 11) % 19) as f32 - 9.0) * 0.02)
                    .collect();
                let inj_in: Vec<f32> = (0..nt * hp.hc)
                    .map(|i| (((i * 5) % 7) as f32 - 3.0) * 0.1)
                    .collect();
                let outs: Vec<f32> = (0..nt * e)
                    .map(|i| (((i * 23) % 31) as f32 - 15.0) * 0.01)
                    .collect();

                // 旧: トークンごとに関数を呼ぶ (hc_mix_quant_batch の元の形)
                let mut ress: Vec<HyperStreams> = base
                    .iter()
                    .map(|d| HyperStreams {
                        data: d.clone(),
                        hc: hp.hc,
                        n_embd: e,
                    })
                    .collect();
                let mut lo_buf = lo_in.clone();
                let t0 = std::time::Instant::now();
                for _ in 0..reps {
                    for t in 0..nt {
                        let xn = hyper::grouped_rms_norm(&ress[t], &norm, &hp);
                        let lo = &mut lo_buf[t * hp.low_rank..(t + 1) * hp.low_rank];
                        lo.copy_from_slice(&lo_in[t * hp.low_rank..(t + 1) * hp.low_rank]);
                        hyper::hc_lo_apply(lo, &hp);
                        let mixed =
                            hyper::hc_mixed(&xn, &gate[t * hc_dim..(t + 1) * hc_dim], &hp);
                        let mut inj = inj_in[t * hp.hc..(t + 1) * hp.hc].to_vec();
                        hyper::hc_inject_apply(&mut inj, &hp);
                        ress[t] = hyper::hc_combine(
                            &ress[t],
                            &outs[t * e..(t + 1) * e],
                            &inj,
                            &hp,
                        );
                        std::hint::black_box(&mixed);
                        std::hint::black_box(&lo);
                    }
                }
                let old_ms = t0.elapsed().as_secs_f64() * 1e3 / (reps * nt) as f64;

                // 新: バッチ版 (トークン方向に 1 回で回す。nt >= 64 は rayon)
                let mut ress2: Vec<HyperStreams> = base
                    .iter()
                    .map(|d| HyperStreams {
                        data: d.clone(),
                        hc: hp.hc,
                        n_embd: e,
                    })
                    .collect();
                let mut lo_buf2 = lo_in.clone();
                let mut inj_buf2 = inj_in.clone();
                let t1 = std::time::Instant::now();
                for _ in 0..reps {
                    let xn_flat = hyper::grouped_rms_norm_batch(&ress2, &norm, &hp);
                    lo_buf2.copy_from_slice(&lo_in);
                    hyper::hc_lo_apply_batch(&mut lo_buf2, nt, &hp);
                    let mixed = hyper::hc_mixed_batch(&xn_flat, &gate, nt, &hp);
                    inj_buf2.copy_from_slice(&inj_in);
                    hyper::hc_inject_apply_batch(&mut inj_buf2, nt, &hp);
                    hyper::hc_combine_batch(&mut ress2, &outs, &inj_buf2, &hp);
                    std::hint::black_box(&mixed);
                    std::hint::black_box(&lo_buf2);
                }
                let new_ms = t1.elapsed().as_secs_f64() * 1e3 / (reps * nt) as f64;
                println!(
                    "{nt}\t{old_ms:.4}\t{new_ms:.4}\t{:.2}",
                    old_ms / new_ms
                );
            }
        }

        /// MoE FFN のバッチ版が逐次版とビット一致する (同じ expert を選んだ
        /// トークンをまとめる経路 = `m > 1` を実際に通す)。
        #[test]
        fn moe_ffn_batch_matches_seq() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る。
            crate::cuda::set_gpu_gemm_enabled(false);
            let _g = moe_test_guard();
            let dir = TempDir::new("h2moe");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            assert!(
                p.n_expert_used > 1,
                "まとめる経路を通すには n_expert_used > 1 が必要"
            );
            let lw = cache_test_layer(&f, &p);
            let nt = 3;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            let bat = moe_ffn_quant_batch(&p, &lw, &xs, nt, None).expect("batch");
            for t in 0..nt {
                let seq = moe_ffn_quant(&p, &lw, &xs[t * p.n_embd..(t + 1) * p.n_embd], None)
                    .expect("seq");
                assert_bitwise_eq(&seq, &bat[t * p.n_embd..(t + 1) * p.n_embd], &format!("moe token {t}"));
            }
            assert!(
                bat.iter().any(|v| *v != 0.0),
                "MoE の出力が全 0 (経路の生死が分からない)"
            );
        }

        /// P33: ルータ F32 行列積のトークンブロック経路が、1 トークンずつの
        /// `gemv` (従来経路) とビット一致する。
        ///
        /// 実形 (k=2560 / n=512 / nt=512) と、端数が出る形 (k=2570 / n=6 / nt=71:
        /// トークン 7 本・行 2 行・列 2 列があまる) の両方で確かめる。
        #[test]
        fn gemv_batch_f32_blocked_matches_gemv() {
            crate::cuda::set_gpu_gemm_enabled(false);
            let _g = moe_test_guard();
            for (k, n, nt) in [(2560usize, 512usize, 512usize), (2570, 6, 71)] {
                let mut bytes = vec![0u8; n * k * 4];
                for i in 0..n * k {
                    let r = (i as u64)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let v = ((r >> 33) % 2001) as f32;
                    bytes[i * 4..i * 4 + 4].copy_from_slice(&((v - 1000.0) * 1e-3).to_le_bytes());
                }
                let w = GgufLinear::from_slice(GgmlType::F32, k, n, &bytes).expect("F32 層");
                let xs: Vec<f32> = (0..nt * k)
                    .map(|i| (((i * 37 + 11) % 401) as f32 - 200.0) * 1e-3)
                    .collect();
                // 参照: 1 トークンずつ `gemv` (従来経路) で計算する。
                let mut want = vec![0.0f32; nt * n];
                for t in 0..nt {
                    w.gemv(&xs[t * k..(t + 1) * k], &mut want[t * n..(t + 1) * n])
                        .expect("gemv");
                }
                let before = GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
                let mut got = vec![0.0f32; nt * n];
                gemv_batch(&w, &xs, nt, &mut got).expect("gemv_batch");
                let after = GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
                if nt / ROUTER_TB >= 2 && n >= ROUTER_RB {
                    assert!(
                        after > before,
                        "k={k} n={n} nt={nt}: トークンブロック経路が発動していない"
                    );
                }
                assert_bitwise_eq(&want, &got, &format!("k={k} n={n} nt={nt}"));
            }
        }

        /// P33: 長いバッチ (nt=64) でも MoE のバッチ経路が逐次版とビット一致する。
        /// ルータ (`ffn_gate_inp`) がトークンブロック経路に入ることをカウンタで見る。
        #[test]
        fn moe_ffn_batch_long_matches_seq() {
            crate::cuda::set_gpu_gemm_enabled(false);
            let _g = moe_test_guard();
            let dir = TempDir::new("h2moe64");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 64;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            let before = GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let bat = moe_ffn_quant_batch(&p, &lw, &xs, nt, None).expect("batch");
            assert!(
                GEMV_BATCH_BLOCKED_CALLS.load(std::sync::atomic::Ordering::Relaxed) > before,
                "ルータのトークンブロック経路が発動していない"
            );
            for t in 0..nt {
                let seq = moe_ffn_quant(&p, &lw, &xs[t * p.n_embd..(t + 1) * p.n_embd], None)
                    .expect("seq");
                assert_bitwise_eq(
                    &seq,
                    &bat[t * p.n_embd..(t + 1) * p.n_embd],
                    &format!("moe long token {t}"),
                );
            }
            assert!(
                bat.iter().any(|v| *v != 0.0),
                "MoE の出力が全 0 (経路の生死が分からない)"
            );
        }

        /// flat スケジューラ (層全体の (expert, 行列, 行チャンク) を 1 本のタスク列で回す) が
        /// legacy (expert ごと × 行列ごとに 1 回 rayon) とビット一致する。
        ///
        /// 同じ入力に対して両方の経路を**明示的に**通し、出力を突き合わせる。小さい形なので
        /// `m` は 1〜4 に散らばり、`m == 1` の expert と `m > 1` の expert の両方ができる。
        #[test]
        fn moe_flat_matches_legacy_bitwise() {
            crate::cuda::set_gpu_gemm_enabled(false);
            let _g = moe_test_guard();
            let dir = TempDir::new("h2flat");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 8;
            let xs: Vec<f32> = (0..nt * p.n_embd)
                .map(|i| (((i * 37 % 101) as f32) - 50.0) * 0.013)
                .collect();

            let flat0 = MOE_FLAT_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let leg0 = MOE_LEGACY_CALLS.load(std::sync::atomic::Ordering::Relaxed);
            let flat =
                moe_ffn_quant_batch_with(&p, &lw, &xs, nt, None, MoeSchedule::Flat).expect("flat");
            let legacy = moe_ffn_quant_batch_with(&p, &lw, &xs, nt, None, MoeSchedule::Legacy)
                .expect("legacy");
            assert!(
                MOE_FLAT_CALLS.load(std::sync::atomic::Ordering::Relaxed) > flat0,
                "flat 経路が発動していない"
            );
            assert!(
                MOE_LEGACY_CALLS.load(std::sync::atomic::Ordering::Relaxed) > leg0,
                "legacy 経路が発動していない"
            );
            assert!(
                flat.iter().any(|v| *v != 0.0),
                "flat の出力が全 0 (経路の生死が分からない)"
            );
            assert_bitwise_eq(&legacy, &flat, "flat vs legacy");

            // 逐次版 (トークンごとの `moe_ffn_quant`) とも一致する
            for t in 0..nt {
                let seq = moe_ffn_quant(&p, &lw, &xs[t * p.n_embd..(t + 1) * p.n_embd], None)
                    .expect("seq");
                assert_bitwise_eq(
                    &seq,
                    &flat[t * p.n_embd..(t + 1) * p.n_embd],
                    &format!("flat token {t}"),
                );
            }
        }

        /// `DENNO_MOE_FLAT` の解釈 (未設定・不明な値は既定の Auto に落ちる)。
        #[test]
        fn moe_schedule_env_values() {
            assert_eq!(moe_schedule_from_value(None), MoeSchedule::Auto);
            assert_eq!(moe_schedule_from_value(Some("1")), MoeSchedule::Flat);
            assert_eq!(moe_schedule_from_value(Some("0")), MoeSchedule::Legacy);
            assert_eq!(moe_schedule_from_value(Some("")), MoeSchedule::Auto);
            assert_eq!(moe_schedule_from_value(Some("true")), MoeSchedule::Auto);
        }

        /// タスク分割が `[0, n_out)` を 1 度ずつ覆う (端数・0 行も落とさない)。
        #[test]
        fn moe_row_chunks_cover_all_rows_once() {
            for n_out in [0usize, 1, 31, 32, 33, 640, 2560] {
                let mut v = Vec::new();
                moe_row_chunks(0, n_out, MOE_FLAT_ROWS, &mut v).expect("chunks");
                let mut covers = vec![0u8; n_out];
                for t in &v {
                    assert!(t.a < t.b, "空のチャンク (n_out={n_out})");
                    assert!(t.b as usize <= n_out, "行が範囲外 (n_out={n_out})");
                    for r in t.a as usize..t.b as usize {
                        covers[r] += 1;
                    }
                }
                assert!(
                    covers.iter().all(|&c| c == 1),
                    "n_out={n_out}: 行の重複か欠落がある"
                );
                assert_eq!(v.len(), n_out.div_ceil(MOE_FLAT_ROWS), "n_out={n_out}: 本数");
            }
        }

        /// 実形状 (n_in=2560, n_ff=640) の expert 3 本だけを持つ合成 GGUF。
        ///
        /// 実形は 1 expert あたり gate/up が 921 KB・down が 1.23 MB なので、
        /// 86 expert で約 264 MB になる (テスト用の一時ファイル)。
        fn write_expert_gguf_big(dir: &TempDir, n_in: usize, n_ff: usize, ne: usize) -> PathBuf {
            let path = dir.path("experts_big.gguf");
            let mut gate = Vec::new();
            let mut up = Vec::new();
            let mut down = Vec::new();
            for e in 0..ne {
                gate.extend_from_slice(&q4k_expert_bytes(n_in, n_ff, e));
                up.extend_from_slice(&q4k_expert_bytes(n_in, n_ff, e + 7));
                down.extend_from_slice(&q5_1_expert_bytes(n_ff, n_in, e + 3));
            }
            let tensors = vec![
                (
                    "blk.0.ffn_gate_exps.weight".to_string(),
                    vec![n_in as u64, n_ff as u64, ne as u64],
                    TY_Q4_K,
                    gate,
                ),
                (
                    "blk.0.ffn_up_exps.weight".to_string(),
                    vec![n_in as u64, n_ff as u64, ne as u64],
                    TY_Q4_K,
                    up,
                ),
                (
                    "blk.0.ffn_down_exps.weight".to_string(),
                    vec![n_ff as u64, n_in as u64, ne as u64],
                    TY_Q5_1,
                    down,
                ),
            ];
            let kvs = [
                ("general.architecture", Kv::Str("qwen4exp")),
                ("general.alignment", Kv::U32(ALIGN as u32)),
            ];
            write_gguf(&path, &kvs, &tensors);
            path
        }

        /// マイクロベンチ用の諸元 (実形)。
        fn bench_params(n_in: usize, n_ff: usize, ne: usize, k: usize) -> Qwen4ExpParams {
            let mut p = cache_test_params();
            p.n_embd = n_in;
            p.n_ff_exp = n_ff;
            p.n_ff_shexp = n_ff;
            p.n_expert = ne;
            p.n_expert_used = k;
            p
        }

        /// マイクロベンチ (手動)。選択 expert の FFN **1 層ぶん**を
        /// flat (層全体のタスク表) と legacy (expert ごと) で A/B する。
        ///
        /// 実形: 86 expert / n_embd 2560 / n_ff_exp 640 / gate・up は Q4_K、down は Q5_1
        /// (Flash-Next の実型)。m は 5〜120 に散らし (平均 ≈ 60 = 512 トークン × top-10 / 86)、
        /// 一様な m で閾値の当たりも取る。
        ///
        /// このコンテナはホスト共有で load average が動くので、**同一バイナリ・同一ループ内**で
        /// legacy / flat (行幅 4 種) を交互に測り、rep の min を取る (別バイナリ比較はしない)。
        ///
        /// 16 スレッドで走らせる:
        /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --features cuda --lib \
        ///   moe_flat_schedule_microbench -- --ignored --nocapture --test-threads=1`
        #[test]
        #[ignore = "CPU のスケジュール A/B (実機の ms/token は main が測る)"]
        fn moe_flat_schedule_microbench() {
            crate::cuda::set_gpu_gemm_enabled(false);
            let _g = moe_test_guard();
            let (n_in, n_ff, ne, k) = (2560usize, 640usize, 86usize, 10usize);
            let dir = TempDir::new("h2bench");
            let path = write_expert_gguf_big(&dir, n_in, n_ff, ne);
            let f = GgufFile::open(&path).expect("open");
            let p = bench_params(n_in, n_ff, ne, k);
            let lw = cache_test_layer(&f, &p);
            let widths = [64usize, 128, 256, 512];

            // expert ごとの件数 `ms` から、(t, j) の割り当て・重み・入力を組み立てる。
            // スロット (t, j) を決定的にシャッフルして各 expert に配るので、
            // 1 expert のトークンは層全体に散らばる (実モデルと同じ形)。
            let build = |ms: &[usize]| {
                let sum: usize = ms.iter().sum();
                let nt = sum.div_ceil(k);
                let total = nt * k;
                let mut slots: Vec<usize> = (0..total).collect();
                let mut seed = 12345u64;
                for i in (1..total).rev() {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let j = ((seed >> 33) as usize) % (i + 1);
                    slots.swap(i, j);
                }
                let mut it = slots.into_iter();
                let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ms.len()];
                for (ei, &m) in ms.iter().enumerate() {
                    let mut v: Vec<(usize, usize)> = (0..m)
                        .map(|_| {
                            let s = it.next().expect("slot");
                            (s / k, s % k)
                        })
                        .collect();
                    // legacy の `position()` 順 (トークン昇順, 同トークン内は j 昇順)
                    v.sort_unstable();
                    by_expert[ei] = v;
                }
                let weights = vec![0.1f32; nt * k];
                let xs: Vec<f32> = (0..nt * n_in)
                    .map(|i| (((i * 37) % 101) as f32 - 50.0) * 0.003)
                    .collect();
                (by_expert, weights, xs, nt)
            };

            println!(
                "threads={} ne={ne} n_in={n_in} n_ff={n_ff} k={k}",
                rayon::current_num_threads()
            );

            // 1 つの分布について legacy + flat (行幅 4 種) を交互に測り、min を出す。
            let run = |ms: &[usize], label: &str| {
                let (by_expert, weights, xs, nt) = build(ms);
                let mut outs: Vec<Vec<f32>> = (0..1 + widths.len())
                    .map(|_| vec![0.0f32; nt * n_in])
                    .collect();
                // 温め (ページフォルト・分岐・レイアウト) + 値の突き合わせ
                for (vi, out) in outs.iter_mut().enumerate() {
                    if vi == 0 {
                        moe_experts_legacy(&lw, &by_expert, &weights, k, &xs, n_in, n_ff, out)
                            .expect("legacy");
                    } else {
                        moe_experts_flat(
                            &lw, &by_expert, &weights, k, &xs, n_in, n_ff, widths[vi - 1], out,
                        )
                        .expect("flat");
                    }
                }
                for vi in 1..outs.len() {
                    assert_bitwise_eq(
                        &outs[0],
                        &outs[vi],
                        &format!("{label}: flat(rows={}) vs legacy", widths[vi - 1]),
                    );
                }

                let reps = 3usize;
                let mut best = vec![f64::INFINITY; outs.len()];
                for _ in 0..reps {
                    for (vi, out) in outs.iter_mut().enumerate() {
                        for v in out.iter_mut() {
                            *v = 0.0;
                        }
                        let t0 = std::time::Instant::now();
                        if vi == 0 {
                            moe_experts_legacy(&lw, &by_expert, &weights, k, &xs, n_in, n_ff, out)
                                .expect("legacy");
                        } else {
                            moe_experts_flat(
                                &lw, &by_expert, &weights, k, &xs, n_in, n_ff, widths[vi - 1], out,
                            )
                            .expect("flat");
                        }
                        let us = t0.elapsed().as_secs_f64() * 1e6;
                        best[vi] = best[vi].min(us);
                    }
                }
                let tot: usize = ms.iter().sum();
                println!(
                    "{label:>10} (tot_m={tot:5}) legacy={:8.0}us | flat 32={:8.0} ({:.2}x) \
                     64={:8.0} ({:.2}x) 128={:8.0} ({:.2}x) 256={:8.0} ({:.2}x)",
                    best[0],
                    best[1],
                    best[0] / best[1],
                    best[2],
                    best[0] / best[2],
                    best[3],
                    best[0] / best[3],
                    best[4],
                    best[0] / best[4],
                );
            };

            // 参考: 作業バッファを毎層 `vec![0.0; n]` で作るときの「確保 + 初回タッチ」。
            // flat は層をまたいでプールを使い回すので、この分は毎層は払わない。
            let alloc_probe = |tot: usize| -> f64 {
                let t0 = std::time::Instant::now();
                {
                    let mut v = vec![0.0f32; tot * n_in];
                    let mut a = vec![0.0f32; tot * n_ff];
                    let mut b = vec![0.0f32; tot * n_ff];
                    let mut hh = vec![0.0f32; tot * n_in];
                    for x in v.iter_mut() {
                        *x = 1.0;
                    }
                    for x in a.iter_mut() {
                        *x = 1.0;
                    }
                    for x in b.iter_mut() {
                        *x = 1.0;
                    }
                    for x in hh.iter_mut() {
                        *x = 1.0;
                    }
                    std::hint::black_box((&v, &a, &b, &hh));
                }
                t0.elapsed().as_secs_f64() * 1e6
            };

            // 実測されている分布: m は 5〜120 (平均 ≈ 60)
            let spread: Vec<usize> = (0..ne).map(|ei| 5 + (ei * 37) % 116).collect();
            let tot: usize = spread.iter().sum();
            println!(
                "m=5..120: mean={:.1}  alloc+touch(毎層)={:.0}us",
                tot as f32 / ne as f32,
                alloc_probe(tot)
            );
            run(&spread, "m=5..120");

            // 閾値 (`MOE_FLAT_MIN_M`) の当たり: 一様な m
            for &mu in &[1usize, 2, 4, 8, 16, 32, 64, 120] {
                run(&vec![mu; ne], &format!("m={mu}"));
            }
        }

        /// バッチ GEMV は逐次版と同じ**入口**を通るので、融合カーネルを
        /// 含めてビット一致する。2 つの入口 (`GgufLinear::gemv` = expert_gemv、
        /// `ExpertW::gemv` = expert_gemv_any) を別々に確かめる
        /// (Q5_1 / Q8_0 は入口でカーネルが変わるため)。
        #[test]
        fn gemv_batch_matches_gemv_bitwise() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る (契約は prefill のみ解除)。
            crate::cuda::set_gpu_gemm_enabled(false);
            let dir = TempDir::new("h2gemv");
            let path = write_expert_gguf(&dir);
            let f = GgufFile::open(&path).expect("open");
            let p = cache_test_params();
            let lw = cache_test_layer(&f, &p);
            let nt = 3;
            for (w, name) in [
                (&lw.gate_exps, "gate (Q4_K = 融合カーネル)"),
                (&lw.down_exps, "down (Q5_1 = 入口でカーネルが変わる)"),
            ] {
                let xs: Vec<f32> = (0..nt * w.n_in)
                    .map(|i| (((i * 29) % 101) as f32 - 50.0) * 0.007)
                    .collect();

                // 1. GgufLinear 経路 (`linear()` と同じく fast_q8 = true で組む)
                let lin = GgufLinear::from_slice(
                    w.ty,
                    w.n_in,
                    w.n_out,
                    w.slice(0).expect("expert 0"),
                )
                .expect("GgufLinear::from_slice")
                .with_fast_q8(true);
                let mut bat = vec![0.0f32; nt * w.n_out];
                gemv_batch(&lin, &xs, nt, &mut bat).expect("gemv_batch");
                let mut seq = vec![0.0f32; nt * w.n_out];
                for t in 0..nt {
                    let mut one = vec![0.0f32; w.n_out];
                    lin.gemv(&xs[t * w.n_in..(t + 1) * w.n_in], &mut one)
                        .expect("gemv");
                    seq[t * w.n_out..(t + 1) * w.n_out].copy_from_slice(&one);
                }
                assert!(
                    bat.iter().any(|v| *v != 0.0),
                    "{name}: GgufLinear 経路の出力が全 0 (経路の生死が分からない)"
                );
                assert_bitwise_eq(&seq, &bat, &format!("{name} / GgufLinear"));

                // 2. ExpertW 経路 (expert_gemv_any = Q5_1/Q8_0 も融合)
                let mut bat2 = vec![0.0f32; nt * w.n_out];
                expert_gemv_batch(w, 0, &xs, nt, &mut bat2).expect("expert_gemv_batch");
                let mut seq2 = vec![0.0f32; nt * w.n_out];
                for t in 0..nt {
                    let mut one = vec![0.0f32; w.n_out];
                    w.gemv(0, &xs[t * w.n_in..(t + 1) * w.n_in], &mut one)
                        .expect("ExpertW::gemv");
                    seq2[t * w.n_out..(t + 1) * w.n_out].copy_from_slice(&one);
                }
                assert_bitwise_eq(&seq2, &bat2, &format!("{name} / ExpertW"));
            }

            // Q8_0 は fast_q8 = true のとき融合カーネルになる (モデルの linear() と
            // 同じ設定)。ここも行優先バッチでビット一致すること。
            let (k8, n8) = (64usize, 5usize);
            let bytes8 = tensor_bytes("w", &[k8 as u64, n8 as u64], GgmlType::Q8_0, 11);
            let lin8 = GgufLinear::from_slice(GgmlType::Q8_0, k8, n8, &bytes8)
                .expect("Q8_0 の GgufLinear")
                .with_fast_q8(true);
            let xs8: Vec<f32> = (0..nt * k8)
                .map(|i| (((i * 17) % 97) as f32 - 48.0) * 0.011)
                .collect();
            let mut b8 = vec![0.0f32; nt * n8];
            gemv_batch(&lin8, &xs8, nt, &mut b8).expect("gemv_batch (Q8_0)");
            let mut s8 = vec![0.0f32; nt * n8];
            for t in 0..nt {
                let mut one = vec![0.0f32; n8];
                lin8.gemv(&xs8[t * k8..(t + 1) * k8], &mut one)
                    .expect("gemv (Q8_0)");
                s8[t * n8..(t + 1) * n8].copy_from_slice(&one);
            }
            assert!(
                b8.iter().any(|v| *v != 0.0),
                "Q8_0: 出力が全 0 (経路の生死が分からない)"
            );
            assert_bitwise_eq(&s8, &b8, "Q8_0 (fast_q8 = 融合)");
        }

        /// 合成 GGUF に書いた 4 種の量子型 (Q8_0 / Q5_1 / Q4_K / Q5_K) で、
        ///
        /// (a) `gemv_batch(nt=1)` が逐次 `gemv` とビット一致する、
        /// (b) `gemv_batch(nt=64)` の各トークンが `gemv` (1 トークン) とビット一致する、
        /// (c) 行数の多い形が「ブロック 1 回デコード」経路を通る (カウンタで確認)、
        ///
        /// を確かめる。Q8_0 は `fast_q8` の両方 (融合カーネル / スカラー) を通す。
        /// 行数の少ない形 (n=5) は旧経路 (行ごとに nt 回デコード) に落ちるが、
        /// 同じ `gemv_batch` から同じ値が出ることをここで見る。
        #[test]
        fn gemv_batch_all_quant_types_bitwise() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る (契約は prefill のみ解除)。
            crate::cuda::set_gpu_gemm_enabled(false);
            let dir = TempDir::new("h2q4gt");
            let types = [
                GgmlType::Q8_0,
                GgmlType::Q5_1,
                GgmlType::Q4K,
                GgmlType::Q5K,
            ];
            // 大きい形: ブロック 1 回デコード経路 + トークンのタイル分割
            // (T = GEMV_BATCH_TILE_BYTES / (4k) = 32 < nt=64 なので 2 枚に割れる)。
            let (bk, bn, bnt) = (2048usize, GEMV_BATCH_MIN_ROWS, 64usize);
            // 小さい形: 旧経路。
            let (sk, sn, snt) = (512usize, 5usize, 3usize);

            // 境目は純粋関数で固定する (行数 / トークン数)
            assert!(!gemv_batch_blockdecode(GEMV_BATCH_MIN_ROWS - 1, 64));
            assert!(gemv_batch_blockdecode(GEMV_BATCH_MIN_ROWS, 64));
            assert!(gemv_batch_blockdecode(bn, bnt), "大きい形がバッチ経路に載らない");
            assert!(!gemv_batch_blockdecode(sn, snt), "小さい形がバッチ経路に載った");
            assert!(!gemv_batch_blockdecode(usize::MAX, 0), "nt=0 では回さない");

            let mut tensors = Vec::new();
            for (i, ty) in types.iter().enumerate() {
                for (tag, k, n) in [("b", bk, bn), ("s", sk, sn)] {
                    let name = format!("{tag}{i}");
                    let data = tensor_bytes(&name, &[k as u64, n as u64], *ty, i as u32 + 1);
                    tensors.push((name, vec![k as u64, n as u64], ggml_id(*ty), data));
                }
            }
            let path = dir.path("gemv_types.gguf");
            write_model(&path, &tensors);
            let f = GgufFile::open(&path).expect("合成 GGUF を開けない");

            let c0 = GEMV_BATCH_BLOCKDECODE_CALLS.load(Ordering::Relaxed);
            for (i, ty) in types.iter().enumerate() {
                for (tag, k, n, nt) in [("b", bk, bn, bnt), ("s", sk, sn, snt)] {
                    let name = format!("{tag}{i}");
                    let (file, off, len) = f.tensor_mapping_parts(&name).expect("データ範囲");
                    let map = f.mmap_handle_at(file).expect("part の mmap");
                    let bytes = &map[off..off + len];
                    for fast_q8 in [false, true] {
                        let lin = GgufLinear::from_slice(*ty, k, n, bytes)
                            .expect("GgufLinear")
                            .with_fast_q8(fast_q8);
                        let want = |i: usize, w: usize| -> Vec<f32> {
                            (0..w).map(|j| (((i + j * 37) % 101) as f32 - 50.0) * 0.011).collect()
                        };

                        // (a) nt=1 が逐次版とビット一致
                        let x1 = want(0, k);
                        let mut b1 = vec![0.0f32; n];
                        gemv_batch(&lin, &x1, 1, &mut b1).expect("gemv_batch (nt=1)");
                        let mut s1 = vec![0.0f32; n];
                        lin.gemv(&x1, &mut s1).expect("gemv");
                        assert!(
                            s1.iter().any(|v| *v != 0.0),
                            "{name}: 出力が全 0 (経路の生死が分からない)"
                        );
                        assert_bitwise_eq(
                            &s1,
                            &b1,
                            &format!("{name} fast_q8={fast_q8} nt=1 / gemv"),
                        );

                        // (b) nt トークンの各行が 1 トークンの計算とビット一致
                        let mut xs = Vec::with_capacity(nt * k);
                        for t in 0..nt {
                            xs.extend_from_slice(&want(t + 1, k));
                        }
                        let mut bat = vec![0.0f32; nt * n];
                        gemv_batch(&lin, &xs, nt, &mut bat).expect("gemv_batch (nt)");
                        for t in 0..nt {
                            let mut seq = vec![0.0f32; n];
                            lin.gemv(&xs[t * k..(t + 1) * k], &mut seq).expect("gemv");
                            assert_bitwise_eq(
                                &seq,
                                &bat[t * n..(t + 1) * n],
                                &format!("{name} fast_q8={fast_q8} token {t} / gemv"),
                            );
                        }
                    }
                }
            }
            assert!(
                GEMV_BATCH_BLOCKDECODE_CALLS.load(Ordering::Relaxed) > c0,
                "ブロック 1 回デコード経路が 1 回も走っていない"
            );
        }

        /// 実形状 (Flash-Next / n_embd 2560) の Q8_0 行列で、バッチ GEMV の
        /// 1 呼び出しを CPU で測る。**GPU 実測の代わりではない**。
        /// カーネル構造の A/B (per-token 呼び出し vs ブロック 1 回デコード) 用。
        #[test]
        #[ignore = "CPU のカーネル A/B (実機の ms/token は main が測る)"]
        fn gemv_batch_shape_ab() {
            // P-PF-GB: CPU 参照とのビット比較なので GPU GEMM を切る (契約は prefill のみ解除)。
            crate::cuda::set_gpu_gemm_enabled(false);
            use std::time::Instant;
            let nt = 64usize;
            // (名前, k, n) — qwen4exp の実形状 (n_embd 2560 / conv_dim 10240 /
            // inner 6144 / kv 1024 / q 12288 / shexp 640 / hc_lr 320)
            let shapes = [
                ("ssm_qkv", 2560usize, 10240usize),
                ("ssm_gate", 2560, 6144),
                ("ssm_out", 6144, 2560),
                ("hc_up", 320, 10240),
                ("hc_down", 10240, 320),
                ("shexp_gate", 2560, 640),
                ("shexp_down", 640, 2560),
                ("full_q", 2560, 12288),
                ("full_kv", 2560, 1024),
                ("full_o", 6144, 2560),
            ];
            let mut tot_new = 0.0f64;
            let mut tot_old = 0.0f64;
            for (name, k, n) in shapes {
                let bytes = tensor_bytes("w", &[k as u64, n as u64], GgmlType::Q8_0, 3);
                let lin = GgufLinear::from_slice(GgmlType::Q8_0, k, n, &bytes)
                    .expect("GgufLinear")
                    .with_fast_q8(true);
                let xs: Vec<f32> = (0..nt * k)
                    .map(|i| (((i * 31) % 97) as f32 - 48.0) * 0.01)
                    .collect();
                let mut outs = vec![0.0f32; nt * n];
                let mut old = vec![0.0f32; nt * n];
                gemv_batch(&lin, &xs, nt, &mut outs).expect("warm-up");
                assert!(outs.iter().any(|v| *v != 0.0), "{name}: 出力が全 0");
                // 旧経路 (行ごとに 1 行 GEMV を nt 回) を同じ入力で測る
                gemv_rows_pertoken(
                    GgmlType::Q8_0,
                    true,
                    &bytes,
                    k,
                    n,
                    &xs,
                    nt,
                    &mut old,
                )
                .expect("pertoken");
                assert_bitwise_eq(&old, &outs, &format!("{name}: 旧経路と一致"));
                let reps = 7;
                let mut best_new = f64::MAX;
                let mut best_old = f64::MAX;
                for _ in 0..reps {
                    let t0 = Instant::now();
                    gemv_batch(&lin, &xs, nt, &mut outs).expect("bench");
                    best_new = best_new.min(t0.elapsed().as_secs_f64());
                    let t0 = Instant::now();
                    gemv_rows_pertoken(
                        GgmlType::Q8_0,
                        true,
                        &bytes,
                        k,
                        n,
                        &xs,
                        nt,
                        &mut old,
                    )
                    .expect("bench old");
                    best_old = best_old.min(t0.elapsed().as_secs_f64());
                }
                tot_new += best_new;
                tot_old += best_old;
                println!(
                    "{name:<11} k={k:<6} n={n:<6} 新 {:7.3} / 旧 {:7.3} ms/chunk  x{:.2}",
                    best_new * 1e3,
                    best_old * 1e3,
                    best_old / best_new
                );
            }
            println!(
                "合計 新 {:.3} / 旧 {:.3} ms/chunk (= {:.3} / {:.3} ms/token)  x{:.2}",
                tot_new * 1e3,
                tot_old * 1e3,
                tot_new * 1e3 / nt as f64,
                tot_old * 1e3 / nt as f64,
                tot_old / tot_new
            );
        }

        /// 層優先バッチ prefill の logits が逐次経路とビット一致する。
        ///
        /// 併せて「バッチ分岐とバッチ GEMV が実際に走った」ことをカウンタで見る
        /// (緑 = 経路が発動した証拠ではない)。チャンク 1 / 3 / 8 で切れ目を変えても
        /// 一致することを確かめる (チャンク跨ぎの状態持ち越しが無いことの確認)。
        #[test]
        fn prefill_batch_matches_sequential() {
            let _g = moe_test_guard();
            let dir = TempDir::new("h2model");
            let path = build(&dir);
            let f = GgufFile::open(&path).expect("合成モデルを開けない");
            // 語彙内のトークン列 (PLE の n-gram 窓は末尾 3 個)
            let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];

            let (seq, bseq, _) = run(&f, &ctx, false, 0);
            assert_eq!(bseq, 0, "逐次経路でバッチ分岐が走った");
            assert_eq!(seq.len(), VOCAB, "logits の長さが n_vocab と違う");
            assert!(seq.iter().all(|v| v.is_finite()), "逐次 prefill に NaN/inf");
            assert!(
                seq.iter().any(|&v| v != seq[0]),
                "logits が全要素同値 (経路の生死が分からない)"
            );

            // 決定性 (同じモデルを組み直して同じ値)
            let (seq2, _, _) = run(&f, &ctx, false, 0);
            assert_bitwise_eq(&seq, &seq2, "逐次 prefill 2 回目");

            for chunk in [1usize, 3, 8] {
                let (bat, nb, ng) = run(&f, &ctx, true, chunk);
                assert!(nb > 0, "chunk {chunk}: 層優先バッチ経路が発動していない");
                assert!(ng > 0, "chunk {chunk}: バッチ GEMV が 1 回も走っていない");
                assert_bitwise_eq(&seq, &bat, &format!("chunk {chunk}"));
            }
        }

        // -----------------------------------------------------------------
        // P33: MoE ルータ (gate 行列積 + softmax + top-k + m 集計) の実形状ベンチ
        // -----------------------------------------------------------------

        /// 実形状のルータ 1 層ぶんの入力を組み立てる (モデル不要)。
        ///
        /// `blk.N.ffn_gate_inp.weight` は実行 GGUF で `[2560, 512] ty=F32`
        /// (5,242,880 B)。値を [-1, 1] の決定的な擬似乱数にするので、
        /// logits は N(0, 3.4) 程度に散り、top-k はほぼ一様に選ばれる。
        fn router_bench_inputs(nt: usize, ne: usize, n_in: usize) -> (GgufLinear, Vec<f32>) {
            let mut bytes = vec![0u8; ne * n_in * 4];
            for i in 0..ne * n_in {
                let r = (i as u64)
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((r >> 33) % 2001) as f32;
                bytes[i * 4..i * 4 + 4].copy_from_slice(&((v - 1000.0) * 1e-3).to_le_bytes());
            }
            let w = GgufLinear::from_slice(GgmlType::F32, n_in, ne, &bytes).expect("gate_inp F32");
            let xs: Vec<f32> = (0..nt * n_in)
                .map(|i| (((i * 37 + 11) % 401) as f32 - 200.0) * 1e-3)
                .collect();
            (w, xs)
        }

        /// `reps` 回まわして最小値 (ms) を返す。ホストは共有なので**同一実行内**で比べる。
        fn best_of_ms(reps: usize, mut f: impl FnMut()) -> f64 {
            let mut best = f64::INFINITY;
            for _ in 0..reps {
                let t = std::time::Instant::now();
                f();
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            best
        }

        /// ルータ 1 層ぶん (nt=512) の (a) gate 行列積 / (b) softmax /
        /// (c) top-k + m 集計 を、**旧実装と新実装の両方**で同一ループ内で測る。
        /// 1 層合計と ms/token/層 も出す。
        ///
        /// `cargo test --release -p denno-engine --lib --features cuda \
        ///   moe_router_microbench -- --ignored --nocapture --test-threads=1`
        /// を `RAYON_NUM_THREADS=8` と `16` で回す。
        #[test]
        #[ignore = "CPU ルータのマイクロベンチ (実機の ms/token は main が測る)"]
        fn moe_router_microbench() {
            crate::cuda::set_gpu_gemm_enabled(false);
            let _g = moe_test_guard();
            let (nt, ne, n_in, k) = (512usize, 512usize, 2560usize, 10usize);
            let (gate, xs) = router_bench_inputs(nt, ne, n_in);
            let reps = 20usize;
            println!(
                "threads={} nt={nt} ne={ne} n_in={n_in} k={k} reps={reps}",
                rayon::current_num_threads()
            );

            // ---- (a) gate 行列積: out[t*ne + r] = dot(row_r, xs[t]) ----
            let bytes = gate.blocks().as_slice();
            let rb = bytes.len() / ne;
            let ty = gate.ty();
            let mut old = vec![0.0f32; nt * ne];
            let a_old = best_of_ms(reps, || {
                gemv_f32_pertoken(ty, bytes, rb, n_in, ne, &xs, nt, &mut old)
            });
            let mut new = vec![0.0f32; nt * ne];
            let a_new = best_of_ms(reps, || {
                let wf = dequant_rows_f32(ty, bytes, rb, n_in, ne);
                gemv_f32_blocked(&wf, n_in, ne, &xs, nt, &mut new);
            });
            assert_bitwise_eq(&old, &new, "(a) 旧 vs 新 (gate 行列積)");
            let a_new_dequant = best_of_ms(reps, || {
                let wf = dequant_rows_f32(ty, bytes, rb, n_in, ne);
                std::hint::black_box(wf.as_ptr());
            });

            // ---- (b) softmax (トークンごと) ----
            let mut probs = old.clone();
            let probs0 = probs.clone();
            let b_old = best_of_ms(reps, || {
                for t in 0..nt {
                    softmax(&mut probs[t * ne..(t + 1) * ne]);
                }
            });
            probs.copy_from_slice(&probs0);
            let b_new = best_of_ms(reps, || {
                #[cfg(feature = "par")]
                {
                    use rayon::prelude::*;
                    probs.par_chunks_mut(ne).for_each(softmax);
                }
                #[cfg(not(feature = "par"))]
                for t in 0..nt {
                    softmax(&mut probs[t * ne..(t + 1) * ne]);
                }
            });
            // (c) 用の入力は 1 回だけ作る (どちらの実装もこの probs を読む)
            probs.copy_from_slice(&probs0);
            for t in 0..nt {
                softmax(&mut probs[t * ne..(t + 1) * ne]);
            }

            // ---- (c) top-k 選択 + 再正規化 + expert ごとのトークン集計 ----
            // 旧: トークンごとの Vec + 「最大値を k 回走査」の選択
            let mut m_old = (0usize, 0usize, 0usize);
            let c_old = best_of_ms(reps, || {
                let mut sels: Vec<Vec<usize>> = Vec::with_capacity(nt);
                let mut weights: Vec<Vec<f32>> = Vec::with_capacity(nt);
                for t in 0..nt {
                    let pr = &probs[t * ne..(t + 1) * ne];
                    let mut sel = vec![0usize; k];
                    crate::model::moe::select_topk_scan(pr, k, &mut sel);
                    let mut sum = 0.0f32;
                    for &i in &sel {
                        sum += pr[i];
                    }
                    let mut w = vec![0.0f32; k];
                    for (j, &i) in sel.iter().enumerate() {
                        w[j] = pr[i] / sum;
                    }
                    sels.push(sel);
                    weights.push(w);
                }
                let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ne];
                for (t, sel) in sels.iter().enumerate() {
                    for (j, &ei) in sel.iter().enumerate() {
                        by_expert[ei].push((t, j));
                    }
                }
                let nonempty = by_expert.iter().filter(|p| !p.is_empty()).count();
                let m_max = by_expert.iter().map(|p| p.len()).max().unwrap_or(0);
                m_old = (nonempty, nt * k / nonempty.max(1), m_max);
                std::hint::black_box((sels, weights, by_expert));
            });

            // 新: 平らな配列 + select_nth_unstable_by の選択 + 件数を先に数える集計
            let mut m_new = (0usize, 0usize, 0usize);
            let c_new = best_of_ms(reps, || {
                let mut sels = vec![0usize; nt * k];
                let mut weights = vec![0.0f32; nt * k];
                let renorm = |pr: &[f32], sel: &[usize], w: &mut [f32]| {
                    let mut sum = 0.0f32;
                    for &i in sel {
                        sum += pr[i];
                    }
                    for (j, &i) in sel.iter().enumerate() {
                        w[j] = pr[i] / sum;
                    }
                };
                #[cfg(feature = "par")]
                {
                    use rayon::prelude::*;
                    sels.par_chunks_mut(k)
                        .zip(weights.par_chunks_mut(k))
                        .enumerate()
                        .for_each_init(Vec::<u32>::new, |buf, (t, (s, w))| {
                            let pr = &probs[t * ne..(t + 1) * ne];
                            select_topk_into(pr, k, buf, s);
                            renorm(pr, s, w);
                        });
                }
                #[cfg(not(feature = "par"))]
                {
                    let mut buf: Vec<u32> = Vec::new();
                    for t in 0..nt {
                        let pr = &probs[t * ne..(t + 1) * ne];
                        select_topk_into(pr, k, &mut buf, &mut sels[t * k..(t + 1) * k]);
                        renorm(pr, &sels[t * k..(t + 1) * k], &mut weights[t * k..(t + 1) * k]);
                    }
                }
                let mut counts = vec![0usize; ne];
                for &ei in &sels {
                    counts[ei] += 1;
                }
                let mut by_expert: Vec<Vec<(usize, usize)>> = counts
                    .iter()
                    .map(|&c| Vec::with_capacity(c))
                    .collect();
                for t in 0..nt {
                    for j in 0..k {
                        by_expert[sels[t * k + j]].push((t, j));
                    }
                }
                let nonempty = by_expert.iter().filter(|p| !p.is_empty()).count();
                let m_max = by_expert.iter().map(|p| p.len()).max().unwrap_or(0);
                m_new = (nonempty, nt * k / nonempty.max(1), m_max);
                std::hint::black_box((sels, weights, by_expert));
            });
            assert_eq!(m_old, m_new, "(c) 旧 vs 新で m の分布が違う");

            let tot_old = a_old + b_old + c_old;
            let tot_new = a_new + b_new + c_new;
            println!("                       旧 (ms/層)   新 (ms/層)   倍率");
            println!("(a) gate 行列積       {a_old:9.3}   {a_new:9.3}   x{:.2}", a_old / a_new);
            println!("(b) softmax           {b_old:9.3}   {b_new:9.3}   x{:.2}", b_old / b_new);
            println!("(c) top-k + m 集計    {c_old:9.3}   {c_new:9.3}   x{:.2}", c_old / c_new);
            println!("合計                  {tot_old:9.3}   {tot_new:9.3}   x{:.2}", tot_old / tot_new);
            println!(
                "ms/token/層            {:.4}      {:.4}",
                tot_old / nt as f64,
                tot_new / nt as f64
            );
            println!(
                "うち (a) の重み展開    {a_new_dequant:9.3} ms/層 (新経路の内訳)"
            );
            println!(
                "m 分布: nonempty={} mean={} max={}",
                m_new.0, m_new.1, m_new.2
            );
        }
    }
}
