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
use crate::model::moe::select_topk;
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
                )?;
                q4prof::add(&q4prof::MIX, t_mix);
                let t_attn = std::time::Instant::now();
                let mut outs = Vec::with_capacity(nt);
                for cur in &curs {
                    let o = self.attn.forward(il, cur)?;
                    if o.len() != e {
                        return Err(one(format!(
                            "層 {il}: attention の出力長 {} が n_embd {e} と違う",
                            o.len()
                        )));
                    }
                    outs.push(o);
                }
                q4prof::add(&q4prof::ATTN, t_attn);
                for t in 0..nt {
                    ress[t] = hyper::hc_combine(&ress[t], &outs[t], &injs[t], &hp);
                }

                // 3.5-3.7 ffn 入口 (バッチ) → MoE FFN (バッチ) → 出口
                let t_mix = std::time::Instant::now();
                let (curs, injs) = hc_mix_quant_batch(
                    &ress,
                    &lw.hc_ffn_norm,
                    &lw.hc_ffn_down,
                    &lw.hc_ffn_up,
                    Some(&lw.hc_ffn_inject),
                    &hp,
                )?;
                q4prof::add(&q4prof::MIX, t_mix);
                let flat = flatten_rows(&curs, e)?;
                let t_moe = std::time::Instant::now();
                let ffn = moe_ffn_quant_batch(&self.p, lw, &flat, nt, self.expert_cache.as_mut())?;
                q4prof::add(&q4prof::MOE, t_moe);
                for t in 0..nt {
                    ress[t] = hyper::hc_combine(&ress[t], &ffn[t * e..(t + 1) * e], &injs[t], &hp);
                }
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

/// [`hc_mix_quant`] のバッチ版 (`nt` トークンぶんまとめて行列積)。
///
/// 行列積 (`down` / `up` / `inject`) だけを [`gemv_batch`] に置き換え、
/// ストリームごとの RMS 正規化・`hc_lo_apply` / `hc_mixed` / `hc_inject_apply` は
/// 逐次版と同じ関数をトークンごとに呼ぶ。要素ごとの演算順序は変わらないので、
/// 1 トークンぶんの出力は [`hc_mix_quant`] とビット一致する。
fn hc_mix_quant_batch(
    ress: &[HyperStreams],
    norm: &[f32],
    down: &GgufLinear,
    up: &GgufLinear,
    inject: Option<&GgufLinear>,
    hp: &HyperParams,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
    let nt = ress.len();
    let hc_dim = hp.hc * hp.n_embd;
    let mut xn: Vec<Vec<f32>> = Vec::with_capacity(nt);
    for (t, r) in ress.iter().enumerate() {
        if r.data.len() != hc_dim {
            return Err(one(format!(
                "hc_mix (batch): トークン {t} の res 長 {} が hc_dim {hc_dim} と違う",
                r.data.len()
            )));
        }
        xn.push(hyper::grouped_rms_norm(r, norm, hp));
    }
    let xn_flat = flatten_rows(&xn, hc_dim)?;

    let dn = down.n();
    let mut lo = vec![0.0f32; nt * dn];
    gemv_batch(down, &xn_flat, nt, &mut lo)
        .map_err(|err| one(format!("hc_mix down (batch): {err}")))?;
    for t in 0..nt {
        hyper::hc_lo_apply(&mut lo[t * dn..(t + 1) * dn], hp);
    }

    let un = up.n();
    let mut gate = vec![0.0f32; nt * un];
    gemv_batch(up, &lo, nt, &mut gate).map_err(|err| one(format!("hc_mix up (batch): {err}")))?;
    let mut mixed = Vec::with_capacity(nt);
    for t in 0..nt {
        mixed.push(hyper::hc_mixed(&xn[t], &gate[t * un..(t + 1) * un], hp));
    }

    let injs = match inject {
        Some(w) => {
            let wn = w.n();
            let mut v = vec![0.0f32; nt * wn];
            gemv_batch(w, &xn_flat, nt, &mut v)
                .map_err(|err| one(format!("hc_mix inject (batch): {err}")))?;
            for t in 0..nt {
                hyper::hc_inject_apply(&mut v[t * wn..(t + 1) * wn], hp);
            }
            let mut out = Vec::with_capacity(nt);
            for t in 0..nt {
                out.push(v[t * wn..(t + 1) * wn].to_vec());
            }
            out
        }
        None => vec![vec![0.0f32; hp.hc]; nt],
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

    Ok(out)
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
    /// [`GgufLinear::gemv`] と同じ入口。**`linear()` 経由のテンソル前提**
    /// (`fast_q8 = true`)。`from_slice` で組んだ層を比べるときは、その層も
    /// `with_fast_q8(true)` にしておくこと。
    Linear,
    /// [`ExpertW::gemv`] と同じ入口 (`expert_gemv_any`)
    Expert,
}

/// 量子化重みの行優先バッチ GEMV: `out[t * n + r] = dot(row_r, x_t)`。
///
/// 行 `r` を外側で 1 回だけ切り出し、内側でトークンを回す。1 行ぶんの計算は
/// `kern` が指す入口を `n_out = 1` で呼ぶ = 逐次版と**同じカーネル**なので、
/// Q4_K の融合カーネル (AVX2+FMA) を含めてビット一致する。行は互いに独立で、
/// `x` に依存する前計算も同じ `x` から作られる。
///
/// 速くなる根拠は演算量ではなく**重みの再読み込み**: 逐次版はトークンごとに同じ
/// 行を最初から読み直すが、ここでは 1 行がキャッシュに載ったまま `nt` トークンぶん
/// 使われる (27B の expert は 1 トークンで 487 MB を触るので、ここが支配的)。
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
    let rb = bytes.len() / n;
    for r in 0..n {
        let row = &bytes[r * rb..(r + 1) * rb];
        for t in 0..nt {
            let x = &xs[t * k..(t + 1) * k];
            let o = &mut outs[t * n + r..t * n + r + 1];
            let res = match kern {
                // モデルの Q8_0 は fast_q8 = true (linear() がそう組む) なので、
                // GgufLinear::gemv と同じく融合カーネルの入口を使う。
                GemvKernel::Linear if ty == GgmlType::Q8_0 => {
                    moe_expert::expert_gemv_q8_fast(row, k, 1, x, o)
                }
                GemvKernel::Linear => moe_expert::expert_gemv(ty, row, k, 1, x, o),
                GemvKernel::Expert => moe_expert::expert_gemv_any(ty, row, k, 1, x, o),
            };
            res.map_err(|err| one(format!("gemv_batch: 行 {r} トークン {t}: {err}")))?;
        }
    }
    #[cfg(test)]
    GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// [`GgufLinear`] のバッチ GEMV (`nt` トークン × 1 行列)。
///
/// 量子型は [`gemv_rows_batch`] (行優先・`GgufLinear::gemv` と同じ入口)、
/// 無量子型 (F32/F16/BF16) は [`GgufLinear::gemv`] と同じ「行を f32 に開いて
/// 列昇順に逐次加算」を、行ごとに 1 回だけ展開してトークンぶん使い回す。
///
/// 注意: `--features cuda` で重みが VRAM に常駐している場合、逐次版
/// ([`GgufLinear::gemv`]) は GPU カーネルで計算するため、ここ (CPU 参照) とは
/// 一致しない。バッチ経路は現状 CPU のみ (GPU 常駐時は `DENNO_PREFILL_BATCH` を
/// 立てないこと)。
fn gemv_batch(w: &GgufLinear, xs: &[f32], nt: usize, outs: &mut [f32]) -> Result<()> {
    let (k, n) = (w.k(), w.n());
    match w.ty() {
        GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => gemv_rows_batch(
            GemvKernel::Linear,
            w.ty(),
            w.blocks().as_slice(),
            k,
            n,
            xs,
            nt,
            outs,
        ),
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
            let mut row = vec![0.0f32; k];
            for r in 0..n {
                crate::kquant::dequant_row_into(w.ty(), &bytes[r * rb..(r + 1) * rb], &mut row);
                for t in 0..nt {
                    let x = &xs[t * k..(t + 1) * k];
                    let mut acc = 0.0f32;
                    for i in 0..k {
                        acc += row[i] * x[i];
                    }
                    outs[t * n + r] = acc;
                }
            }
            #[cfg(test)]
            GEMV_BATCH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
        other => Err(one(format!("gemv_batch: 未対応の量子型 {other:?}"))),
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

/// 1 チャンクのトークン数 (`DENNO_PREFILL_BATCH_CHUNK`, 既定 8, 1..=64)。
///
/// チャンクを大きくするほど重みの再読み込みは減るが、残差 `[nt][hc_dim]` と
/// `hc_mix` の中間がそのぶん増える (27B の hc_dim は 20480 f32 = 80 KiB/トークン)。
fn prefill_batch_chunk() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("DENNO_PREFILL_BATCH_CHUNK")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.clamp(1, 64))
            .unwrap_or(8)
    })
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
    let mut probs = vec![0.0f32; nt * ne];
    gemv_batch(&lw.gate_inp, xs, nt, &mut probs)
        .map_err(|err| one(format!("層 {}: ffn_gate_inp (batch): {err}", lw.il)))?;

    // 2. トークンごとの softmax / top-k / 再正規化 (逐次版と同じ順序)
    let mut sels: Vec<Vec<usize>> = Vec::with_capacity(nt);
    let mut weights: Vec<Vec<f32>> = Vec::with_capacity(nt);
    for t in 0..nt {
        let pr = &mut probs[t * ne..(t + 1) * ne];
        softmax(pr);
        let sel = select_topk(pr, k);
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

    // 3. 選択 expert の重みを先に読み始める (逐次版と同じ expert を同じ順で)
    for sel in &sels {
        for &ei in sel {
            lw.gate_exps.prefetch(ei);
            lw.up_exps.prefetch(ei);
            lw.down_exps.prefetch(ei);
        }
    }

    // 4. expert を index 昇順に、同じ expert を選んだトークンをまとめて回す
    let mut out = vec![0.0f32; nt * e];
    for ei in 0..ne {
        let picks: Vec<(usize, usize)> = sels
            .iter()
            .enumerate()
            .filter_map(|(t, s)| s.iter().position(|&x| x == ei).map(|j| (t, j)))
            .collect();
        if picks.is_empty() {
            continue;
        }
        let m = picks.len();
        let mut xin = Vec::with_capacity(m * e);
        for &(t, _) in &picks {
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
            let pe = weights[t][j];
            let hh = &h[idx * e..(idx + 1) * e];
            for d in 0..e {
                out[t * e + d] += pe * hh[d];
            }
        }
    }

    // 5. shared expert (バッチ)。係数は sigmoid(gate_inp_shexp @ x)
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
            )
            .expect("hc_mix_quant_batch");
            for (t, r) in ress.iter().enumerate() {
                let (m, i) = hc_mix_quant(r, &norm, &down, &up, Some(&inject), &hp)
                    .expect("hc_mix_quant");
                assert_bitwise_eq(&m, &mixed[t], &format!("hc_mix mixed (token {t})"));
                assert_bitwise_eq(&i, &injs[t], &format!("hc_mix inject (token {t})"));
            }
            assert!(
                mixed[0].iter().any(|v| *v != 0.0),
                "mixed が全 0 (経路の生死が分からない)"
            );
        }

        /// MoE FFN のバッチ版が逐次版とビット一致する (同じ expert を選んだ
        /// トークンをまとめる経路 = `m > 1` を実際に通す)。
        #[test]
        fn moe_ffn_batch_matches_seq() {
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

        /// バッチ GEMV は逐次版と同じ**入口**を通るので、融合カーネルを
        /// 含めてビット一致する。2 つの入口 (`GgufLinear::gemv` = expert_gemv、
        /// `ExpertW::gemv` = expert_gemv_any) を別々に確かめる
        /// (Q5_1 / Q8_0 は入口でカーネルが変わるため)。
        #[test]
        fn gemv_batch_matches_gemv_bitwise() {
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
    }
}
