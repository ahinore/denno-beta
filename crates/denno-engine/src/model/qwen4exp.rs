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

fn one(msg: impl Into<String>) -> Qwen4ExpError {
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
    /// true なら `dims` の末尾は「以上」を要求する
    /// (`per_layer_token_embd` の行数は converter がパディングする)。
    pub min_last_dim: bool,
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
    ExpectedTensor {
        name,
        dims,
        ty,
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
    if t.ty != e.ty {
        errs.push(format!(
            "{}: 型 {} が期待 {} と違う",
            e.name,
            ty_name(t.ty),
            ty_name(e.ty)
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
            min_last_dim: true,
        };
        assert_eq!(e.fmt_dims(), "[160, 40 以上]");
    }
}
