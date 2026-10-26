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
use crate::model::hyper::{self, HyperParams, HyperStreams};
use crate::model::linear_gguf::GgufLinear;
use crate::model::moe::select_topk;
use crate::model::moe_expert::{expert_gemv, ExpertSlice};
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
}

/// F1 の恒等実装。`cur` をそのまま返す (attention / ssm は F2 / F3)。
pub struct IdentityAttn;

impl LayerAttn for IdentityAttn {
    fn forward(&mut self, _layer: usize, cur: &[f32]) -> Result<Vec<f32>> {
        Ok(cur.to_vec())
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
    GgufLinear::new(t.ty, k, n, blocks).map_err(|e| one(format!("{name}: {e}")))
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

    /// `out[r] = dot(row_r, x)` を 1 expert ぶん計算する。
    fn gemv(&self, e: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        expert_gemv(self.ty, self.slice(e)?, self.n_in, self.n_out, x, out)
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
    pub attn: Box<dyn LayerAttn>,
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

        Ok(Qwen4Exp {
            p,
            layers,
            head,
            tok_embd,
            ple,
            attn,
        })
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
            Some(&t) => *t,
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

        // 3.1 PLE 注入 (層 1 のみ)。res_hc を置き換える。
        if let Some(pw) = &lw.ple {
            let pe = ple_emb.ok_or_else(|| one(format!("層 {il}: PLE の行が無い")))?;
            *res = ple_inject(&self.p, pw, pe, res, hp)?;
            hook("ple_after", il, &res.data)?;
        }

        // 3.2-3.4 attention 入口 → attn → 出口
        let (cur, inject) = hc_mix_quant(
            res,
            &lw.hc_attn_norm,
            &lw.hc_attn_down,
            &lw.hc_attn_up,
            Some(&lw.hc_attn_inject),
            hp,
        )?;
        hook("hc_mixed_attn", il, &cur)?;
        hook("hc_inject_attn", il, &inject)?;
        let cur = self.attn.forward(il, &cur)?;
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
        let (cur, inject) = hc_mix_quant(
            res,
            &lw.hc_ffn_norm,
            &lw.hc_ffn_down,
            &lw.hc_ffn_up,
            Some(&lw.hc_ffn_inject),
            hp,
        )?;
        hook("hc_mixed_ffn", il, &cur)?;
        let cur = moe_ffn_quant(&self.p, lw, &cur)?;
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
    let mut conv = vec![0.0f32; hc_dim];
    for c in 0..hc_dim {
        // out[c] = Σ_k w[k][c] * x[c, t - (kern-1-k)*dil]
        // GGUF の ne は [kern, hc_dim] なので、列 c の k 番目は conv1d[c*kern + k]。
        let mut acc = 0.0f32;
        for k in 0..kern {
            let delay = (kern - 1 - k) * dil;
            // F1: 履歴は 0 埋め。t = 0 では delay == 0 の項だけが残る。
            let v = if delay == 0 { x[c] } else { 0.0f32 };
            acc += pw.conv1d[c * kern + k] * v;
        }
        conv[c] = hyper::silu(acc);
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

/// MoE FFN (参照実装 `build_layer_ffn` 1240-1285 行)。
///
/// - ルータ: `ffn_gate_inp @ x` → softmax → top-k (`n_expert_used`) →
///   選択分の和で再正規化 (`norm_w = true`)。
/// - expert: `down( silu(gate(x)) * up(x) )` を選択 index 昇順に
///   `w_e` を掛けて積む。行列は expert 単位で切って GEMV する。
/// - shared expert: `sigmoid(ffn_gate_inp_shexp @ x)` を掛けて最後に 1 回加算。
///
/// `silu` は dense 経路 (`crate::tensor::silu_inplace`) と同じ式を使う。
fn moe_ffn_quant(p: &Qwen4ExpParams, lw: &LayerW, x: &[f32]) -> Result<Vec<f32>> {
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

    // 4. 選択 expert を index 昇順に積む
    let mut out = vec![0.0f32; e];
    let mut act = vec![0.0f32; ff];
    let mut up = vec![0.0f32; ff];
    let mut h = vec![0.0f32; e];
    for (t, &ei) in sel.iter().enumerate() {
        lw.gate_exps.gemv(ei, x, &mut act)?;
        silu_inplace(&mut act);
        lw.up_exps.gemv(ei, x, &mut up)?;
        for r in 0..ff {
            act[r] *= up[r];
        }
        lw.down_exps.gemv(ei, &act, &mut h)?;
        let pe = weights[t];
        for j in 0..e {
            out[j] += pe * h[j];
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
}
