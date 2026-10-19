//! Qwen3.5 ハイブリッドアーキテクチャ (model_type = "qwen3_5")。
//! P19-B/C: GGUF (Unsloth Dynamic 混合量子含む) → dequant → 既存 Int8 パイプライン
//! へ再量子化 (承認済み設計)。CPU 実装 (GDN は gdn.rs の AR 形式・決定論)。
//!
//! 構成 (0.8B): 24 層 = 18 linear_attention (Gated DeltaNet) + 6 full_attention
//! (layer_types = 3:1 インターバル)。full_attention 層は Q+gate 融合
//! (attn_output_gate)、partial MRoPE (n_rot = head_dim*0.25 = 64、interleaved、
//! sections [11,11,10] — テキストでは t=h=w なので通常 RoPE と等価)、
//! QK-Norm、sigmoid 出力ゲートを持つ。embedding は tied (lm_head 無し)。
//!
//! 27B (Qwen3.8) も同一構造 (2026-10-16 実 GGUF ヘッダ + transformers
//! modeling_qwen3_5 + llama.cpp src/models/qwen35.cpp の照合で確認):
//! attn_q = [e, 2*heads*dh] 融合 (head ごと [q(dh)|gate(dh)])・out-gate は
//! attention 後 o_proj 前 (sigmoid)・attn_output = [heads*dh, e]。9B は
//! heads*dh == e (4096) で偶然正方形だが、27B は非正方形 (6144→5120)・
//! 0.8B も非正方形 (2048→1024) でエンジンの既定経路がそのまま正解。ローダは
//! テンソル形状をヘッダ段階で config 由来の期待値と照合し (expect_ne)、
//! 想定外の構造を静かに誤読せずロード時に拒否する。GGUF block_count が
//! config num_hidden_layers を超過する分は MTP (nextn) 層
//! (blk.{n}.nextn.*) として読み飛ばす (27B: block_count=65 = trunk 64 + 1)。
//!
//! GDN 層の状態 (conv バッファ + SSM 行列) は KvCache の GDN スロットに保持
//! (セッションごと)。forward はトークン逐次 (prefill/decode 共通・bit 一致)。
//!
//! exactness: GDN/attention は新規経路のため数値緩和 (maxabs < 1e-3) 枠内。
//! 決定論は同一入力→同一演算順序で保証 (テストで bit 一致 assert)。
//!
//! 参照: モデルアーキテクチャは Qwen3.5 (Apache-2.0) の設定値・GGUF テンソル名は
//! llama.cpp (MIT License — Copyright (c) 2023-2026 The ggml authors) の正規名。
//! 実装は独自の Rust コード (GGUF ローダは gguf.rs・GDN 数式は gdn.rs 参照)。

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::Path;

use crate::gguf::GgufFile;
use crate::gdn::{forward_token_core, GdnCore, GdnParams};
use crate::kv_cache::KvCache;
use crate::model::{cfg_u64, Model};
use crate::quant::{Linear, QuantMode};
use crate::tensor::{rmsnorm, softmax};

/// GatedAttention (full_attention) 層
struct FullLayer {
    ln1_w: Vec<f32>,
    q_w: Linear, // [e, heads*(2*dh)] — head ごとに [q(dh) | gate(dh)]
    k_w: Linear, // [e, kv_heads*dh]
    v_w: Linear, // [e, kv_heads*dh]
    q_norm_w: Vec<f32>,
    k_norm_w: Vec<f32>,
    o_w: Linear, // [heads*dh, e]
}

/// Gated DeltaNet (linear_attention) 層
struct GdnLayer {
    ln1_w: Vec<f32>,
    qkv_w: Linear,   // [e, 2*key_dim+value_dim] = [q|k|v]
    gate_w: Linear,  // [e, value_dim] (attn_gate → z)
    beta_w: Linear,  // [e, n_v_heads]
    alpha_w: Linear, // [e, n_v_heads]
    conv1d: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt: Vec<f32>,
    ssm_norm: Vec<f32>,
    out_w: Linear, // [value_dim, e]
}

struct Layer {
    attn: AttnKind,
    ln2_w: Vec<f32>,
    gate_w: Linear, // mlp.gate_proj
    up_w: Linear,   // mlp.up_proj
    down_w: Linear, // mlp.down_proj
}

impl Layer {
    fn attn_ln1(&self) -> &[f32] {
        match &self.attn {
            AttnKind::Gdn(g) => &g.ln1_w,
            AttnKind::Full(f) => &f.ln1_w,
        }
    }
}

enum AttnKind {
    Gdn(GdnLayer),
    Full(FullLayer),
}

pub struct Qwen35 {
    e: usize,
    heads: usize,
    kv_heads: usize,
    dh: usize,
    n_rot: usize, // partial rotary dims (=64)
    ff: usize,
    ctx: usize,
    vocab: usize,
    eps: f32,
    theta: f32,
    eos_ids: Vec<u32>,
    embed: Vec<f32>,         // [vocab, e] (gather 用 fp32)
    norm_w: Vec<f32>,
    lm_head: Option<Linear>, // tied: Int8 モードのみ量子化コピー / Fp32 は None
    gdn_p: GdnParams,
    layers: Vec<Layer>,
    quant_bytes: usize,
    /// C4: プロセス内で一意なモデル ID (ロード毎に採番)。cuda chain が fp32 重み
    /// (beta/alpha/QK-Norm/conv/ssm 系) を (model_id, layer, tag) でデバイス
    /// キャッシュするためのキー。Linear には uid が無い (F32 系) ための代替。
    mid: u64,
}

/// C4: Qwen35 ロード毎のモデル ID 採番 (1 から・0 は未使用マーカに相当)。
static Q35_MODEL_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// モデルディレクトリから GGUF ファイルを選ぶ。優先順: Q8_0 (精度検証用) >
/// UD-IQ2_XXS > 辞書順先頭。環境変数 DENNO_GGUF_FILE で明示指定も可。
fn pick_gguf(dir: &Path) -> Result<std::path::PathBuf> {
    if let Ok(name) = std::env::var("DENNO_GGUF_FILE") {
        let p = dir.join(&name);
        if p.exists() {
            return Ok(p);
        }
    }
    let mut candidates: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| {
            let p = e.ok()?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("gguf") {
                Some(p)
            } else {
                None
            }
        })
        .collect();
    if candidates.is_empty() {
        bail!("GGUF ファイルが見つかりません ({:?})", dir);
    }
    candidates.sort();
    for pat in ["Q8_0", "UD-IQ2_XXS"] {
        if let Some(p) = candidates
            .iter()
            .find(|p| p.to_string_lossy().contains(pat))
        {
            return Ok(p.clone());
        }
    }
    Ok(candidates[0].clone())
}

/// ロード時の構造検証: テンソルを dequant する前に、GGUF ヘッダ上の形状
/// (ne[0], ne[1], …) を config 由来の期待値と照合する (末尾の 1 は無視)。
/// 想定外の構造 (full-attn の q|gate 融合の有無・out-gate の位置の違い・
/// MTP 層の混入など) を静かに誤読すると logits が壊れるため、不一致は
/// ロード時に明確なエラーで拒否する (27B out-gate 構造調査の反映)。
fn expect_ne(f: &GgufFile, name: &str, ne: &[u64]) -> Result<()> {
    let t = f
        .tensor(name)
        .ok_or_else(|| anyhow::anyhow!("テンソル {name} が GGUF にありません"))?;
    for (i, &want) in ne.iter().enumerate() {
        if t.dims[i] != want {
            bail!(
                "{name}: GGUF 形状 ne={:?} が期待 ne={ne:?} (末尾 1 を除く) と不一致 — config と GGUF の構造ミスマッチ (full-attn の q|gate 融合 / out-gate 構造 / MTP 層の有無を確認)",
                t.dims
            );
        }
    }
    if t.dims.iter().skip(ne.len()).any(|&d| d != 1) {
        bail!("{name}: 期待 ne={ne:?} より高い次元のテンソル (ne={:?})", t.dims);
    }
    Ok(())
}

/// interleaved partial RoPE: 各 head の先頭 n_rot 次元のみペア (2d, 2d+1) で回転。
/// GGML/HF (mrope_interleaved) と同一規約。テキストでは t=h=w で
/// mrope sections 分割は通常 RoPE に縮退する。
fn rope_partial_interleaved(
    x: &mut [f32],
    heads: usize,
    dh: usize,
    n_rot: usize,
    pos: usize,
    theta: f32,
) {
    let pos = pos as f32;
    for h in 0..heads {
        let base = h * dh;
        for d in 0..n_rot / 2 {
            let ang = pos * theta.powf(-2.0 * d as f32 / n_rot as f32);
            let (c, s) = (ang.cos(), ang.sin());
            let x1 = x[base + 2 * d];
            let x2 = x[base + 2 * d + 1];
            x[base + 2 * d] = x1 * c - x2 * s;
            x[base + 2 * d + 1] = x1 * s + x2 * c;
        }
    }
}

impl Qwen35 {
    pub fn load(dir: &Path, cfg: &serde_json::Value, quant: QuantMode) -> Result<Self> {
        // text_config を優先 (0.8B は VLM ラッパ形式)
        let tc = cfg.get("text_config").unwrap_or(cfg);
        let e = cfg_u64(tc, "hidden_size")? as usize;
        let heads = cfg_u64(tc, "num_attention_heads")? as usize;
        let kv_heads = cfg_u64(tc, "num_key_value_heads")? as usize;
        let dh = cfg_u64(tc, "head_dim")? as usize;
        let ff = cfg_u64(tc, "intermediate_size")? as usize;
        let ctx = tc["max_position_embeddings"].as_u64().unwrap_or(262_144) as usize;
        let vocab = cfg_u64(tc, "vocab_size")? as usize;
        let eps = tc["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;

        // rope_parameters (partial_rotary_factor / theta)
        let rp = tc.get("rope_parameters").unwrap_or(&serde_json::Value::Null);
        let theta = rp["rope_theta"].as_f64().unwrap_or(10_000_000.0) as f32;
        let partial = rp["partial_rotary_factor"].as_f64().unwrap_or(0.25) as f64;
        let n_rot = (dh as f64 * partial).round() as usize;

        // GDN ハイパーパラメータ
        let gdn_p = GdnParams {
            n_embd: e,
            n_k_heads: cfg_u64(tc, "linear_num_key_heads")? as usize,
            n_v_heads: cfg_u64(tc, "linear_num_value_heads")? as usize,
            head_k_dim: cfg_u64(tc, "linear_key_head_dim")? as usize,
            head_v_dim: cfg_u64(tc, "linear_value_head_dim")? as usize,
            d_conv: cfg_u64(tc, "linear_conv_kernel_dim")? as usize,
            eps,
        };
        // A3: タイル repeat 規約 (gdn.rs: v-head h → k/q head h % nk・llama.cpp
        // ggml_repeat 相当) は n_v_heads が n_k_heads の整数倍であることを要求する
        // (0.8B 1x・9B 2x・27B 3x。llama.cpp delta-net-base も H_v % H_k == 0 を
        // assert)。割り切れない構成はロード時に拒否する。
        if gdn_p.n_v_heads % gdn_p.n_k_heads != 0 {
            bail!(
                "linear_num_value_heads ({}) が linear_num_key_heads ({}) の整数倍ではありません (GDN k/q head タイル repeat 規約)",
                gdn_p.n_v_heads,
                gdn_p.n_k_heads
            );
        }

        // Qwen3.8-27B config に新設されたキー (値 "swish")。参照実装
        // (transformers / vLLM / llama.cpp) はどこも消費しておらず、full
        // attention 出力ゲートは sigmoid・GDN z ゲートは silu (=swish) が正。
        // 本実装も同一のため動作は不変だが、参照実装が将来これを消費した場合に
        // 備えて存在をログに残す (実機 E2E で出力品質を確認する)。
        match tc.get("output_gate_type").and_then(|v| v.as_str()) {
            Some("swish") => tracing::info!(
                "output_gate_type=swish (full-attn 出力ゲート sigmoid / GDN z ゲート silu の参照規約を採用・動作不変)"
            ),
            Some(other) => tracing::warn!(
                "output_gate_type={other} は未対応 (swish のみ想定)。参照実装が消費する場合は出力が乖離する可能性"
            ),
            None => {}
        }

        let mut eos_ids = Vec::new();
        match &tc["eos_token_id"] {
            serde_json::Value::Number(n) => eos_ids.push(n.as_u64().unwrap() as u32),
            serde_json::Value::Array(a) => {
                for v in a {
                    if let Some(n) = v.as_u64() {
                        eos_ids.push(n as u32);
                    }
                }
            }
            _ => eos_ids.push(151643),
        }

        // GGUF を開く
        let gguf_path = pick_gguf(dir)?;
        let f =
            GgufFile::open(&gguf_path).with_context(|| format!("GGUF を開けません: {:?}", gguf_path))?;
        let arch = f
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        tracing::info!(
            "loading {:?} (arch={}, {} tensors, quant={:?})",
            gguf_path,
            arch,
            f.tensors.len(),
            quant
        );

        let dt = |name: &str| -> Result<Vec<f32>> {
            f.dequant_tensor(name)
                .with_context(|| format!("tensor {name}"))
        };
        // 構造検証付き dequant: ヘッダ上の形状を config 由来の期待値と照合して
        // から読む (想定外の構造は静かに誤読せずロード時に拒否)
        let dt_shaped = |name: &str, ne: &[u64]| -> Result<Vec<f32>> {
            expect_ne(&f, name, ne).with_context(|| format!("tensor {name}"))?;
            dt(name)
        };

        let n_layer = cfg_u64(tc, "num_hidden_layers")? as usize;

        // 層数の整合確認 (MTP/nextn 除外): GGUF の block_count は MTP (nextn)
        // 層を含み得る (27B: block_count=65 = trunk 64 + nextn 1・
        // nextn_predict_layers=1)。推論トランクの層数決定元は config の
        // num_hidden_layers を正とし、超過分の blk.{n_layer}.. は nextn 層
        // (nextn.eh_proj を持ち ssm_a を持たない) のときだけ読み飛ばす。
        // nextn を通常層として読むと意味論が壊れるため、不整合はロード時に拒否。
        // (9B: block_count=32 == num_hidden_layers、nextn KV 無し → 何もしない)
        let block_count = f
            .get(&format!("{arch}.block_count"))
            .and_then(|v| v.as_u64());
        let nextn_predict = f
            .get(&format!("{arch}.nextn_predict_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if let Some(bc) = block_count {
            if (bc as usize) < n_layer {
                bail!(
                    "config num_hidden_layers ({n_layer}) が GGUF block_count ({bc}) を超過 — config と GGUF の不一致"
                );
            }
            let extra = bc as usize - n_layer;
            if extra > 0 {
                if nextn_predict as usize != extra {
                    bail!(
                        "GGUF block_count ({bc}) - config num_hidden_layers ({n_layer}) = {extra} 層の超過が nextn_predict_layers ({nextn_predict}) と不一致 — config と GGUF の構造ミスマッチ"
                    );
                }
                for i in n_layer..bc as usize {
                    let p = format!("blk.{i}.");
                    if f.tensor(&format!("{p}nextn.eh_proj.weight")).is_none()
                        || f.tensor(&format!("{p}ssm_a")).is_some()
                    {
                        bail!(
                            "blk.{i} が nextn (MTP) 層の構成を持たない (nextn.eh_proj.weight 無し or ssm_a 有り) — config num_hidden_layers={n_layer} と GGUF block_count={bc} の不整合"
                        );
                    }
                }
                tracing::info!(
                    "MTP (nextn) 層 {extra} 個 (blk.{n_layer}..blk.{}) を読み飛ばし (推論トランク = config の {n_layer} 層)",
                    bc - 1
                );
            }
        }

        let mut layers: Vec<Layer> = Vec::with_capacity(n_layer);
        let mut quant_bytes = std::cell::Cell::new(0usize);
        let lin = |w: Vec<f32>, k: usize, n: usize| -> Linear {
            let l = Linear::new(w, k, n, quant);
            quant_bytes.set(quant_bytes.get() + l.weight_bytes());
            l
        };
        // 制御信号系の極小テンソル (sigmoid/softplus を通る) は Int8 モードでも
        // Fp32 を維持する。ssm_beta/alpha は delta-rule のゲート (beta ∈ [0,1])・
        // 減衰 exp(g) に直接効き、相対誤差 1% でも出力が大きく崩れる
        // (int8_vs_fp32 logits 差 0.87 → 北京/東京 反転を実測)。
        // サイズは [e, n_v_heads] × 24 層 ≒ 0.8MB f32 で無視できる。
        let lin_ctl = |w: Vec<f32>, k: usize, n: usize| -> Linear {
            let l = Linear::new(w, k, n, QuantMode::Fp32);
            quant_bytes.set(quant_bytes.get() + l.weight_bytes());
            l
        };

        let vd = gdn_p.value_dim();
        let qkv_dim = gdn_p.qkv_dim();

        let eu = e as u64;
        for i in 0..n_layer {
            let p = format!("blk.{i}.");
            let ln1_w = dt_shaped(&format!("{p}attn_norm.weight"), &[eu])?;
            let ln2_w = dt_shaped(&format!("{p}post_attention_norm.weight"), &[eu])?;
            let gate_w = lin(dt_shaped(&format!("{p}ffn_gate.weight"), &[eu, ff as u64])?, e, ff);
            let up_w = lin(dt_shaped(&format!("{p}ffn_up.weight"), &[eu, ff as u64])?, e, ff);
            let down_w = lin(dt_shaped(&format!("{p}ffn_down.weight"), &[ff as u64, eu])?, ff, e);

            let attn = if f.tensor(&format!("{p}ssm_a")).is_some() {
                // GDN (linear_attention) 層
                AttnKind::Gdn(GdnLayer {
                    ln1_w,
                    qkv_w: lin(
                        dt_shaped(&format!("{p}attn_qkv.weight"), &[eu, qkv_dim as u64])?,
                        e,
                        qkv_dim,
                    ),
                    gate_w: lin(
                        dt_shaped(&format!("{p}attn_gate.weight"), &[eu, vd as u64])?,
                        e,
                        vd,
                    ),
                    beta_w: lin_ctl(
                        dt_shaped(&format!("{p}ssm_beta.weight"), &[eu, gdn_p.n_v_heads as u64])?,
                        e,
                        gdn_p.n_v_heads,
                    ),
                    alpha_w: lin_ctl(
                        dt_shaped(&format!("{p}ssm_alpha.weight"), &[eu, gdn_p.n_v_heads as u64])?,
                        e,
                        gdn_p.n_v_heads,
                    ),
                    conv1d: dt_shaped(
                        &format!("{p}ssm_conv1d.weight"),
                        &[gdn_p.d_conv as u64, qkv_dim as u64],
                    )?,
                    ssm_a: dt_shaped(&format!("{p}ssm_a"), &[gdn_p.n_v_heads as u64])?,
                    ssm_dt: dt_shaped(&format!("{p}ssm_dt.bias"), &[gdn_p.n_v_heads as u64])?,
                    ssm_norm: dt_shaped(&format!("{p}ssm_norm.weight"), &[gdn_p.head_v_dim as u64])?,
                    out_w: lin(
                        dt_shaped(&format!("{p}ssm_out.weight"), &[vd as u64, eu])?,
                        vd,
                        e,
                    ),
                })
            } else {
                // full_attention 層 (Q+gate 融合: out = heads*[q(dh)|gate(dh)])。
                // 構造は 0.8B/9B/27B 実 GGUF + transformers/llama.cpp 両参照で共通
                // を確認 (27B: attn_q=[e, 2*heads*dh] 融合・attn_output=
                // [heads*dh, e] 非正方形・out-gate は attention 後 o_proj 前)。
                let q_total = heads * dh * 2;
                AttnKind::Full(FullLayer {
                    ln1_w,
                    q_w: lin(
                        dt_shaped(&format!("{p}attn_q.weight"), &[eu, q_total as u64])?,
                        e,
                        q_total,
                    ),
                    k_w: lin(
                        dt_shaped(&format!("{p}attn_k.weight"), &[eu, (kv_heads * dh) as u64])?,
                        e,
                        kv_heads * dh,
                    ),
                    v_w: lin(
                        dt_shaped(&format!("{p}attn_v.weight"), &[eu, (kv_heads * dh) as u64])?,
                        e,
                        kv_heads * dh,
                    ),
                    q_norm_w: dt_shaped(&format!("{p}attn_q_norm.weight"), &[dh as u64])?,
                    k_norm_w: dt_shaped(&format!("{p}attn_k_norm.weight"), &[dh as u64])?,
                    o_w: lin(
                        dt_shaped(
                            &format!("{p}attn_output.weight"),
                            &[(heads * dh) as u64, eu],
                        )?,
                        heads * dh,
                        e,
                    ),
                })
            };
            layers.push(Layer {
                attn,
                ln2_w,
                gate_w,
                up_w,
                down_w,
            });
        }

        // tied lm_head: Int8 モードでは logits 用の量子化コピーを持つ
        // (tie_word_embeddings は text_config 内 or トップレベル (VLM ラッパ) の
        //  どちらかにあり得る — 9B はトップレベルのみ)
        let tied = tc["tie_word_embeddings"]
            .as_bool()
            .or_else(|| cfg["tie_word_embeddings"].as_bool())
            .unwrap_or(true);
        let embed = dt_shaped("token_embd.weight", &[eu, vocab as u64])?;
        let lm_head = if tied {
            match quant {
                QuantMode::Fp32 => None,
                QuantMode::Int8 => {
                    let l = Linear::new(embed.clone(), e, vocab, quant);
                    quant_bytes.set(quant_bytes.get() + l.weight_bytes());
                    Some(l)
                }
            }
        } else {
            Some(lin(dt_shaped("output.weight", &[eu, vocab as u64])?, e, vocab))
        };
        let norm_w = match f.tensor("output_norm.weight") {
            Some(_) => dt_shaped("output_norm.weight", &[eu])?,
            None => dt_shaped("model.norm.weight", &[eu])?,
        };

        // A3: 層構成の可観測性 (27B = 48 GDN + 16 full を起動ログで確認できる)
        let n_gdn = layers
            .iter()
            .filter(|l| matches!(l.attn, AttnKind::Gdn(_)))
            .count();
        tracing::info!(
            "qwen35 layers: {} ({} GDN + {} full_attention)",
            layers.len(),
            n_gdn,
            layers.len() - n_gdn
        );

        if quant == QuantMode::Int8 {
            let qb = quant_bytes.get();
            tracing::info!(
                "qwen35 int8: linear weights {:.0} MB (fp32 換算 {:.0} MB)",
                qb as f64 / 1e6,
                qb as f64 * 4.0 / 1e6
            );
        }

        Ok(Self {
            e,
            heads,
            kv_heads,
            dh,
            n_rot,
            ff,
            ctx,
            vocab,
            eps,
            theta,
            eos_ids,
            embed,
            norm_w,
            lm_head,
            gdn_p,
            layers,
            quant_bytes: quant_bytes.get(),
            mid: Q35_MODEL_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        })
    }

    /// full_attention 層の 1 トークン attention (pos = 絶対位置) → attn 出力 [e]。
    fn full_attn_token(
        &self,
        lf: &FullLayer,
        h: &[f32],
        cache: &mut KvCache,
        li: usize,
        pos: usize,
    ) -> Result<Vec<f32>> {
        let qg = lf.q_w.apply_bt(h, 1); // [heads*2*dh]
        let k = lf.k_w.apply_bt(h, 1); // [kv_dim]
        let v = lf.v_w.apply_bt(h, 1);
        let att = self.full_attn_core(lf, &qg, &k, &v, cache, li, pos);
        Ok(lf.o_w.apply_bt(&att, 1))
    }

    /// full attention コア: 投影済み qg [heads*2*dh] / k, v [kv_dim] の 1 行から
    /// QK-Norm → partial RoPE → KV 書き込み → GQA attention → sigmoid 出力ゲート
    /// まで (decode n==1 経路)。内部は full_attn_prepare (KV 書き込み + QK-Norm +
    /// RoPE) と full_attn_attend_head (1 head 分の attend) に分割してあり、prefill
    /// バッチ経路 (forward_prefill) は同一の関数列を prepare 逐次 → attend を
    /// (i, head) 並列で実行する — トークン 1 行 / 1 head 分の演算列はこの逐次実行
    /// と同一のため bit-exact (テスト: qwen38_27b_shape_prefill_batch_bit_exact +
    /// qwen38_27b_shape_prefill_thread_count_bit_exact)。
    #[allow(clippy::too_many_arguments)]
    fn full_attn_core(
        &self,
        lf: &FullLayer,
        qg: &[f32],
        k_row: &[f32],
        v_row: &[f32],
        cache: &mut KvCache,
        li: usize,
        pos: usize,
    ) -> Vec<f32> {
        let heads = self.heads;
        let dh = self.dh;
        let mut q = vec![0.0f32; heads * dh];
        self.full_attn_prepare(lf, qg, k_row, v_row, cache, li, pos, &mut q);
        let (kcr, vcr) = cache.kv_bufs_ref(li);
        let mut att = vec![0.0f32; heads * dh];
        for hh in 0..heads {
            self.full_attn_attend_head(
                qg,
                &q,
                kcr,
                vcr,
                pos + 1,
                hh,
                &mut att[hh * dh..(hh + 1) * dh],
            );
        }
        att
    }

    /// full attention の prepare: QK-Norm (head 単位・RoPE 前) + partial interleaved
    /// RoPE + K/V をキャッシュの pos 行へ書き込み。norm+rope 済の q [heads*dh] を
    /// qout に出力する。decode (full_attn_core) と prefill バッチ (forward_prefill)
    /// で共用 — トークン 1 行分の演算列はトークン逐次経路と同一のため bit-exact。
    /// prefill では全トークンの prepare をトークン順に逐次実行してから attend に
    /// 入る (attend は読み取り専用のため全行書き込み後でも結果は同一)。
    /// (s50: prefill の prof を llama 規約の QKNORM / ROPE バケットに分割計上する
    /// ため full_attn_qknorm → full_attn_rope → KV 書き込みの 3 段構成にした —
    /// 呼び出し順と演算列は不変で bit-exact・decode 経路 (full_attn_core) は本関数
    /// 経由で現行どおり。)
    #[allow(clippy::too_many_arguments)]
    fn full_attn_prepare(
        &self,
        lf: &FullLayer,
        qg: &[f32],
        k_row: &[f32],
        v_row: &[f32],
        cache: &mut KvCache,
        li: usize,
        pos: usize,
        qout: &mut [f32],
    ) {
        let mut k = self.full_attn_qknorm(lf, qg, k_row, qout);
        self.full_attn_rope(qout, &mut k, pos);
        // K/V をキャッシュへ
        let kv_dim = self.kv_heads * self.dh;
        let (kc, vc) = cache.kv_bufs(li, 1);
        kc[pos * kv_dim..][..kv_dim].copy_from_slice(&k);
        vc[pos * kv_dim..][..kv_dim].copy_from_slice(v_row);
    }

    /// prepare 前段: QK-Norm (head 単位・RoPE 前) + q の分離 (qg の head ごと前半
    /// dh)。norm 済 k [kv_dim] を返す (RoPE は未適用)。full_attn_prepare と同一の
    /// 演算列 (分離のみで数値不変)。
    fn full_attn_qknorm(
        &self,
        lf: &FullLayer,
        qg: &[f32],
        k_row: &[f32],
        qout: &mut [f32],
    ) -> Vec<f32> {
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let mut k = k_row.to_vec();

        // QK-Norm (head 単位・RoPE 前) + q の分離
        for hh in 0..heads {
            let src = &qg[hh * 2 * dh..hh * 2 * dh + dh];
            let wq = rmsnorm(src, &lf.q_norm_w, dh, self.eps);
            qout[hh * dh..(hh + 1) * dh].copy_from_slice(&wq);
        }
        for kvh in 0..kv_heads {
            let seg = &mut k[kvh * dh..(kvh + 1) * dh];
            let wn = rmsnorm(seg, &lf.k_norm_w, dh, self.eps);
            seg.copy_from_slice(&wn);
        }
        k
    }

    /// prepare 中段: partial interleaved RoPE (q/k・in place)。full_attn_prepare と
    /// 同一の演算列 (分離のみで数値不変)。
    fn full_attn_rope(&self, qout: &mut [f32], k: &mut [f32], pos: usize) {
        let dh = self.dh;
        rope_partial_interleaved(qout, self.heads, dh, self.n_rot, pos, self.theta);
        rope_partial_interleaved(k, self.kv_heads, dh, self.n_rot, pos, self.theta);
    }

    /// full attention の 1 head 分: scores (QK ドット・k 昇順の逐次累積) → softmax
    /// → probs@V (d 昇順の逐次累積) → sigmoid 出力ゲート。読み取り専用 (kcr/vcr は
    /// キャッシュ KV 行・q は prepare 済・qg はゲート部参照) で結果を att_head [dh]
    /// に書く。出力要素ごとの累積順はトークン逐次経路と同一のため bit-exact・
    /// (i, head) ごとに独立 → prefill バッチ経路はこの関数を rayon で並列呼び出し
    /// する (スレッド数非依存・scores はタスクローカルスクラッチ = llama.rs の
    /// SCORES_SCRATCH と同一パターン)。
    #[allow(clippy::too_many_arguments)]
    fn full_attn_attend_head(
        &self,
        qg: &[f32],
        q: &[f32],
        kcr: &[f32],
        vcr: &[f32],
        t_total: usize,
        hh: usize,
        att_head: &mut [f32],
    ) {
        let dh = self.dh;
        let kv_heads = self.kv_heads;
        let kv_dim = kv_heads * dh;
        let group = self.heads / kv_heads;
        let kvh = hh / group;
        let qh = &q[hh * dh..(hh + 1) * dh];
        let scale = 1.0 / (dh as f32).sqrt();
        let mut scores = vec![0.0f32; t_total];
        for j in 0..t_total {
            let kj = &kcr[j * kv_dim + kvh * dh..][..dh];
            let mut acc = 0.0f32;
            for (a, b) in qh.iter().zip(kj.iter()) {
                acc += a * b;
            }
            scores[j] = acc * scale;
        }
        softmax(&mut scores);
        for j in 0..t_total {
            let vj = &vcr[j * kv_dim + kvh * dh..][..dh];
            let pv = scores[j];
            for (a, b) in att_head.iter_mut().zip(vj.iter()) {
                *a += pv * b;
            }
        }

        // sigmoid 出力ゲート (head ごと [dh]) — gate は q 投影出力の後半
        let gate = &qg[hh * 2 * dh + dh..(hh + 1) * 2 * dh];
        for (d, av) in att_head.iter_mut().enumerate() {
            *av /= 1.0 + (-gate[d]).exp(); // sigmoid(gate)
        }
    }

    /// s48 (C1): full attention prefill の attend を GPU カーネル 1 発
    /// (dno_attention_prefill) で実行する。prepare (KV 書き込み + QK-Norm + RoPE)
    /// は呼び出し側がトークン逐次で完了済みの前提 (KV はホスト cache に書き込み
    /// 済み → KV ミラー同期は chain 内で行われ、失敗時もホスト状態は不変で冪等
    /// なため forward 全体の再実行は不要 — llama P17j と同一の設計)。
    /// 成功時は att に sigmoid 出力ゲートまで適用した結果を書いて true を返す。
    /// 失敗時は att を変更せず false を返す (呼び出し側が既存 CPU attend に落ちる
    /// = 数値は現行どおり bit 一致)。GPU 実行時エラーで GPU_BROKEN が立った場合
    /// も以後の経路は CPU に静かに戻る (llama と同一挙動)。
    /// 数値: カーネルは flash 風 online softmax (タイル昇順・固定結合順・原子操作
    /// なし = 決定論・スレッド数/起動順に依存しない) で CPU attend とは厳密には
    /// 異なるが attention 緩和枠内 (maxabs<1e-3・llama P17j と同一クラス・承認済み)。
    /// sigmoid 出力ゲートはカーネル外: 既存の dno_attn_gate_mul は 1 トークン専用
    /// (grid=(heads)) なため、D2H 後にホストで要素ごとに適用する (行独立 = 決定論
    /// 維持・C3 でデバイス連結する際に n トークン版カーネルを追加する)。
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn full_attn_attend_gpu(
        &self,
        qg: &[f32],
        q: &[f32],
        cache: &mut KvCache,
        li: usize,
        pos0: usize,
        n: usize,
        att: &mut [f32],
    ) -> bool {
        let heads = self.heads;
        let dh = self.dh;
        let kv_dim = self.kv_heads * dh;
        let group = heads / self.kv_heads;
        let t_total = pos0 + n;
        if !crate::cuda::attn_prefill_gpu_supported(n, dh, t_total) {
            return false;
        }
        if crate::cuda::chain_attention_prefill(
            cache.kv_id(),
            li,
            self.layers.len(),
            q,
            cache,
            pos0,
            n,
            t_total,
            group,
            dh,
            kv_dim,
            1.0 / (dh as f32).sqrt(),
        )
        .is_none()
        {
            return false;
        }
        let mut ga = match crate::prof::time(crate::prof::D2H_ATTN_PF, || {
            crate::cuda::chain_download(crate::cuda::CH_ATT, n * heads * dh)
        }) {
            Some(v) => v,
            None => return false,
        };
        // sigmoid 出力ゲート: att[t] /= 1+exp(-gate[t]) — CPU attend
        // (full_attn_attend_head) と同一の要素ごと演算 (行独立 → 行並列・
        // スレッド数非依存で決定論)。
        ga.par_chunks_exact_mut(dh)
            .enumerate()
            .for_each(|(t, att_head)| {
                let i = t / heads;
                let hh = t % heads;
                let gate = &qg[i * heads * 2 * dh + hh * 2 * dh + dh..][..dh];
                for (d, av) in att_head.iter_mut().enumerate() {
                    *av /= 1.0 + (-gate[d]).exp();
                }
            });
        att.copy_from_slice(&ga);
        true
    }

    /// GDN 層の 1 トークン forward → attn 出力 [e]。
    fn gdn_attn_token(&self, lg: &GdnLayer, h: &[f32], cache: &mut KvCache, li: usize) -> Vec<f32> {
        let qkv = lg.qkv_w.apply_bt(h, 1);
        let z = lg.gate_w.apply_bt(h, 1);
        let beta_raw = lg.beta_w.apply_bt(h, 1);
        let alpha = lg.alpha_w.apply_bt(h, 1);
        let y = self.gdn_scan(lg, &qkv, &z, &beta_raw, &alpha, cache, li);
        lg.out_w.apply_bt(&y, 1)
    }

    /// GDN スキャン 1 トークン分 (投影済みの行を受け取る)。prefill バッチ経路と
    /// 共用 — 状態読み出し → forward_token_core → 状態書き戻しのみを担当する
    /// (演算順序はトークン逐次経路と同一のため bit-exact)。
    ///
    /// A1-r2: cuda 有効時は decode (n==1) も GPU カーネル (dno_gdn_scan) で処理
    /// する。prefill バッチ経路と同一カーネル・同一 launch 構成のため、トークン
    /// 逐次 GPU 経路とバッチ GPU 経路の GDN 出力は bit 一致する。GDN は bit 一致
    /// 不要契約 (CPU 逐次とは数値が僅かに異なる・決定論のみ保証)。GPU 無効/
    /// エラー時は CPU 逐次にフォールバック (GPU_BROKEN 後は再試行しない)。
    #[allow(clippy::too_many_arguments)]
    fn gdn_scan(
        &self,
        lg: &GdnLayer,
        qkv: &[f32],
        z: &[f32],
        beta_raw: &[f32],
        alpha: &[f32],
        cache: &mut KvCache,
        li: usize,
    ) -> Vec<f32> {
        #[cfg(feature = "cuda")]
        if self.gdn_p.head_k_dim == self.gdn_p.head_v_dim {
            let (conv, ssm) = cache
                .gdn_state_mut(li)
                .expect("GDN state slot が未初期化");
            if let Some(y) = crate::cuda::gdn_scan_batch(
                qkv,
                z,
                beta_raw,
                alpha,
                conv,
                ssm,
                &lg.conv1d,
                &lg.ssm_a,
                &lg.ssm_dt,
                &lg.ssm_norm,
                1,
                self.gdn_p.n_k_heads,
                self.gdn_p.n_v_heads,
                self.gdn_p.head_k_dim,
                self.gdn_p.head_v_dim,
                self.gdn_p.d_conv,
                self.gdn_p.eps,
                1.0 / (self.gdn_p.head_k_dim as f32).sqrt(),
            ) {
                return y;
            }
        }
        let (conv, ssm) = cache
            .gdn_state_mut(li)
            .expect("GDN state slot が未初期化");
        let mut st = crate::gdn::GdnState {
            conv: conv.to_vec(),
            ssm: ssm.to_vec(),
        };
        crate::gdn::gdn_dbg_set_layer(li as i64); // 恒久診断 (env ゲート・未指定時 no-op)
        let cw = GdnCore {
            conv1d: &lg.conv1d,
            ssm_a: &lg.ssm_a,
            ssm_dt: &lg.ssm_dt,
            ssm_norm: &lg.ssm_norm,
        };
        let y = forward_token_core(&self.gdn_p, &cw, qkv, z, beta_raw, alpha, &mut st);
        let (conv, ssm) = cache
            .gdn_state_mut(li)
            .expect("GDN state slot が未初期化");
        conv.copy_from_slice(&st.conv);
        ssm.copy_from_slice(&st.ssm);
        y
    }

    /// GDN スキャンのトークン逐次フォールバック (GPU 無効時・n==1)。
    /// 投影済みの全トークン行を受け取り、行ごとに gdn_scan を回す。
    #[allow(clippy::too_many_arguments)]
    fn gdn_scan_rows(
        &self,
        g: &GdnLayer,
        qkv: &[f32],
        z: &[f32],
        beta_raw: &[f32],
        alpha: &[f32],
        cache: &mut KvCache,
        li: usize,
        n: usize,
        dy: usize,
    ) -> Vec<f32> {
        let nqkv = g.qkv_w.dims().1;
        let nz = g.gate_w.dims().1;
        let nb = g.beta_w.dims().1;
        let na = g.alpha_w.dims().1;
        let mut ys = vec![0.0f32; n * dy];
        for i in 0..n {
            let y = self.gdn_scan(
                g,
                &qkv[i * nqkv..(i + 1) * nqkv],
                &z[i * nz..(i + 1) * nz],
                &beta_raw[i * nb..(i + 1) * nb],
                &alpha[i * na..(i + 1) * na],
                cache,
                li,
            );
            ys[i * dy..(i + 1) * dy].copy_from_slice(&y);
        }
        ys
    }

    /// FFN (SwiGLU)。
    fn ffn(&self, layer: &Layer, x: &[f32]) -> Vec<f32> {
        let gate = layer.gate_w.apply_bt(x, 1);
        let up = layer.up_w.apply_bt(x, 1);
        let mut h = vec![0.0f32; self.ff];
        for i in 0..self.ff {
            h[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
        }
        layer.down_w.apply_bt(&h, 1)
    }

    /// logits (tied fp32 は embed 直接 matvec・rayon 並列)。
    /// s49: fp32 の lm_head (untied: output.weight / tied: embed) は cuda 有効時
    /// dno_matvec_f32 (tensor::matvec_bt とビット一致・積と加算の丸め分離・d 逐次)
    /// で GPU 試行し、不発/失敗時は従来 CPU 経路 (bit 不変) に落ちる。
    /// int8 モードの lm_head (Q8) は apply_bt 内の P17b hook が既存で対応。
    /// 計上は prof::LM_HEAD (prefill 最終 logits も含む = llama per-op と同一)。
    fn logits(&self, h: &[f32]) -> Vec<f32> {
        crate::prof::time(crate::prof::LM_HEAD, || {
            #[cfg(feature = "cuda")]
            {
                // untied と tied で重みキャッシュの key を分ける (mid はロード毎に
                // 一意なので mid*2 / mid*2+1 で衝突しない)。
                let (w, k, n, key) = if let Some(lm) = &self.lm_head {
                    match lm {
                        crate::quant::Linear::F32 { w, k, n } => (w, *k, *n, self.mid * 2),
                        // Q8 は GPU hook 済み (maybe_gpu_linear) → 従来経路へ
                        _ => return self.lm_logits(h),
                    }
                } else {
                    let n_rows = self.embed.len() / self.e;
                    (&self.embed, self.e, n_rows, self.mid * 2 + 1)
                };
                if let Some(out) = crate::cuda::maybe_gpu_matvec_f32(key, w, h, k, n) {
                    return out;
                }
            }
            self.lm_logits(h)
        })
    }

    /// logits の従来 CPU 経路 (bit 不変・untied: apply_bt / tied: embed 直 matvec)。
    fn lm_logits(&self, h: &[f32]) -> Vec<f32> {
        if let Some(lm) = &self.lm_head {
            return lm.apply_bt(h, 1);
        }
        let e = self.e;
        let n_rows = self.embed.len() / e;
        let mut out = vec![0.0f32; n_rows];
        out.par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, dst)| {
                let row0 = ci * 4096;
                for (r, dst_r) in dst.iter_mut().enumerate() {
                    let row = &self.embed[(row0 + r) * e..][..e];
                    let mut acc = 0.0f32;
                    for (a, b) in row.iter().zip(h.iter()) {
                        acc += a * b;
                    }
                    *dst_r = acc;
                }
            });
        out
    }
}

impl Model for Qwen35 {
    fn context(&self) -> usize {
        self.ctx
    }

    fn is_recurrent(&self) -> bool {
        true // GDN の conv/SSM 状態は位置スライス再利用不可
    }

    fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    fn new_cache(&self) -> KvCache {
        let kv_dim = self.kv_heads * self.dh;
        let mut c = KvCache::new(self.layers.len(), kv_dim, self.ctx);
        for (li, layer) in self.layers.iter().enumerate() {
            if matches!(layer.attn, AttnKind::Gdn(_)) {
                c.attach_gdn_slot(li, self.gdn_p.conv_state_len(), self.gdn_p.ssm_state_len());
            }
        }
        c
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let e = self.e;
        let pos0 = cache.len();
        // 先に全トークン分の K/V 容量を確保 (cache.len() は最後に advance する
        // ため、トークンごとの kv_bufs(li,1) では不十分)
        let _ = cache.kv_bufs(0, tokens.len());
        // A1: prefill (n>1) は投影 (Linear) をバッチ一括で回す (GPU では tiled
        // GEMM dno_gemm_i8 が発動)。GDN スキャン / full attention コアはトークン
        // 逐次のまま (再帰状態は逐次更新が必須)。投影の m>1 経路は i32 累積が
        // 順序不変のため m==1 (matvec) と行単位でビット一致 → 出力はトークン
        // 逐次経路と bit-exact。
        if tokens.len() > 1 {
            // s57 (T4): 計測フェーズを prefill に切り替える (PROFILE 無効時は
            // set/get すら行わない = 完全 no-op・数値経路不変)
            return crate::prof::with_phase(crate::prof::Phase::Prefill, || {
                self.forward_prefill(tokens, cache)
            });
        }
        // C4: solo decode (n==1) はデバイス常駐チェーンで実行 (GDN 状態 GPU ミラー)。
        // GPU 無効/非対応構造/形状非対応/kernels_elem.ptx に新カーネルが無い/
        // 実行時エラー → None → 従来の per-op 経路で forward 全体をやり直し
        // (ホスト状態は冪等・未汚染のため安全。DESIGN_C4.md 参照)。
        #[cfg(feature = "cuda")]
        if tokens.len() == 1 {
            if let Some(logits) = self.forward_decode_chain(tokens[0], cache, pos0) {
                return Ok(logits);
            }
            // s50: chain 不発/失敗 → 従来経路。直前の decode chain が GDN の新状態を
            // GPU ミラーに遅延させている可能性があるため、まずホストへ回復してから
            // CPU 経路に入る (回復不能 = ミラー失同步 → 古い状態で黙って継続するより
            // 要求を落とす方が安全)。
            crate::cuda::chain_q35_gdn_flush_pending(cache).ok_or_else(|| {
                anyhow::anyhow!("deferred GDN state flush failed; cannot fall back to CPU path")
            })?;
        }
        // s57 (T4): ここから先は decode (n==1 の per-op 経路 or token 逐次ループ)。
        // 計測フェーズを明示する。PROFILE 無効時は set すら行わない (完全 no-op)。
        if crate::prof::enabled() {
            crate::prof::set_phase(crate::prof::Phase::Decode);
        }
        let mut logits = Vec::new();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = pos0 + i;
            if pos >= self.ctx {
                bail!("context overflow: pos={pos} >= {}", self.ctx);
            }
            if tok as usize >= self.vocab {
                bail!("token id {tok} が vocab ({}) を超過", self.vocab);
            }
            let mut x: Vec<f32> = self.embed[tok as usize * e..][..e].to_vec();

            for (li, layer) in self.layers.iter().enumerate() {
                // --- attention ---
                let h = rmsnorm(&x, layer.attn_ln1(), e, self.eps);
                let attn_out = match &layer.attn {
                    AttnKind::Gdn(g) => self.gdn_attn_token(g, &h, cache, li),
                    AttnKind::Full(fl) => self.full_attn_token(fl, &h, cache, li, pos)?,
                };
                for (xi, ai) in x.iter_mut().zip(attn_out.iter()) {
                    *xi += ai;
                }
                // --- FFN ---
                let h2 = rmsnorm(&x, &layer.ln2_w, e, self.eps);
                let ffn_out = self.ffn(layer, &h2);
                for (xi, fi) in x.iter_mut().zip(ffn_out.iter()) {
                    *xi += fi;
                }
            }

            let hf = rmsnorm(&x, &self.norm_w, e, self.eps);
            logits = self.logits(&hf);
        }
        cache.advance(tokens.len());
        Ok(logits)
    }
}

impl Qwen35 {
    /// prefill (n>1): 投影をバッチ一括 GEMM、GDN スキャン / full attention は
    /// トークン逐次。トークン逐次経路と bit-exact (テスト: qwen35_prefill_batch
    /// _matches_tokenwise)。
    fn forward_prefill(&self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        // s50: decode chain で遅延中の GDN 状態 (GPU ミラーが権威) をホストへ回復
        // する。prefill はホスト状態を正として読み/書くため、ここで同期 D2H で
        // 一致させておく (回復不能 = ミラー失同步 → 要求を落とす)。
        #[cfg(feature = "cuda")]
        crate::cuda::chain_q35_gdn_flush_pending(cache)
            .ok_or_else(|| anyhow::anyhow!("deferred GDN state flush failed before prefill"))?;
        let e = self.e;
        let n = tokens.len();
        let pos0 = cache.len();
        for (i, &tok) in tokens.iter().enumerate() {
            let pos = pos0 + i;
            if pos >= self.ctx {
                bail!("context overflow: pos={pos} >= {}", self.ctx);
            }
            if tok as usize >= self.vocab {
                bail!("token id {tok} が vocab ({}) を超過", self.vocab);
            }
        }
        // 埋め込み X [n, e]
        let mut x = vec![0.0f32; n * e];
        // s50: prefill prof 内訳 (llama.rs 規約準拠 — LM_HEAD は s49 の logits()
        // ラッパで計上済み・prof 無効時は分岐 1 回で演算不変 = bit-exact)
        crate::prof::time(crate::prof::EMBED, || {
            for (i, &tok) in tokens.iter().enumerate() {
                x[i * e..(i + 1) * e].copy_from_slice(&self.embed[tok as usize * e..][..e]);
            }
        });
        lchk("prefill emb", &x[(n - 1) * e..]); // 恒久診断 (env ゲート・未指定時 no-op)
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let dh = self.dh;
        let kv_dim = kv_heads * dh;
        let ff = self.ff;

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention: rmsnorm を行単位で一括計算 (s42: 行独立 → 行並列。
            // 各要素の演算列は不変のため bit-exact・スレッド数非依存) ---
            let mut h = vec![0.0f32; n * e];
            crate::prof::time_n(crate::prof::RMSNORM, n, || {
                h.par_chunks_exact_mut(e)
                    .zip(x.par_chunks_exact(e))
                    .for_each(|(h_row, x_row)| {
                        let r = rmsnorm(x_row, layer.attn_ln1(), e, self.eps);
                        h_row.copy_from_slice(&r);
                    });
            });
            match &layer.attn {
                AttnKind::Gdn(g) => {
                    // 投影はバッチ一括 (m=n → GPU tiled GEMM)。qkv+z は decode chain
                    // と同一の QKV バケット・beta/alpha は chain の dno_gdn_scan 頭部
                    // matvec と同一の ATTN バケットに計上 (prefill と decode chain の
                    // プロファイル比較可能性維持)。
                    // s57 (T4): attn_core は層種別で別バケット (ATT_GDN) に分離し、
                    // バッチ幅 n を計測に載せる (`time_full` は集計のみで演算列に
                    // 触れない)。実行先ラベルは区間内の GPU 線形呼び出しの実測で
                    // 決まる (cuda 統計) ため、ここでは主張しない。
                    let (kqkv, _) = g.qkv_w.dims();
                    debug_assert_eq!(kqkv, e);
                    let qkv = crate::prof::time_n_ev(crate::prof::QKV, n, || {
                        g.qkv_w.apply_bt_rows(&h, n)
                    });
                    let z = crate::prof::time_n_ev(crate::prof::QKV, n, || g.gate_w.apply_bt_rows(&h, n));
                    let beta_raw = crate::prof::time_n_ev(crate::prof::ATT_GDN, n, || {
                        g.beta_w.apply_bt_rows(&h, n)
                    });
                    let alpha = crate::prof::time_n_ev(crate::prof::ATT_GDN, n, || {
                        g.alpha_w.apply_bt_rows(&h, n)
                    });
                    // スキャン: GPU カーネル 1 発 (P19-A1・cuda 有効時)。GDN は
                    // bit 一致不要のため GPU/CPU で数値は僅かに異なる (決定論は維持)。
                    // GPU 無効時はトークン逐次 (再帰状態) にフォールバック。
                    let dy = g.out_w.dims().0; // value_dim
                    // s57 (T4): GDN スキャンは attn_core の最大寄与。GPU カーネルが
                    // 実際に走ったか (Some/None) を実行先ラベルとして計測に載せる —
                    // ラベルは戻り値から得るので数値経路・分岐は不変。
                    #[cfg(feature = "cuda")]
                    let (ys, scan_on_gpu): (Vec<f32>, bool) =
                        crate::prof::time_layer_n_ev(crate::prof::ATT_GDN, li, n, || {
                            let p = &self.gdn_p;
                            if n > 1 && p.head_k_dim == p.head_v_dim {
                                let (conv, ssm) = cache
                                    .gdn_state_mut(li)
                                    .expect("GDN state slot が未初期化");
                                match crate::cuda::gdn_scan_batch(
                                    &qkv,
                                    &z,
                                    &beta_raw,
                                    &alpha,
                                    conv,
                                    ssm,
                                    &g.conv1d,
                                    &g.ssm_a,
                                    &g.ssm_dt,
                                    &g.ssm_norm,
                                    n,
                                    p.n_k_heads,
                                    p.n_v_heads,
                                    p.head_k_dim,
                                    p.head_v_dim,
                                    p.d_conv,
                                    p.eps,
                                    1.0 / (p.head_k_dim as f32).sqrt(),
                                ) {
                                    Some(y) => (y, true),
                                    None => (
                                        self.gdn_scan_rows(g, &qkv, &z, &beta_raw, &alpha, cache, li, n, dy),
                                        false,
                                    ),
                                }
                            } else {
                                (
                                    self.gdn_scan_rows(g, &qkv, &z, &beta_raw, &alpha, cache, li, n, dy),
                                    false,
                                )
                            }
                        });
                    #[cfg(not(feature = "cuda"))]
                    let (ys, scan_on_gpu): (Vec<f32>, bool) =
                        crate::prof::time_layer_n_ev(crate::prof::ATT_GDN, li, n, || {
                            (
                                self.gdn_scan_rows(g, &qkv, &z, &beta_raw, &alpha, cache, li, n, dy),
                                false,
                            )
                        });
                    let _ = scan_on_gpu;
                    let out = crate::prof::time_n_ev(crate::prof::OPROJ, n, || {
                        g.out_w.apply_bt_rows(&ys, n)
                    });
                    if crate::gdn::gdn_dbg_enabled_for(li as i64) {
                        // 恒久診断 (env ゲート・未指定時 no-op): 出力投影後 (llama.cpp linear_attn_out 相当)
                        for i in 0..n {
                            crate::gdn::gdn_dbg_pub("attnout", &out[i * e..(i + 1) * e]);
                        }
                    }
                    // s42: 残差加算は行独立 → 行並列 (bit-exact)
                    x.par_chunks_exact_mut(e)
                        .zip(out.par_chunks_exact(e))
                        .for_each(|(x_row, out_row)| {
                            for (xi, ai) in x_row.iter_mut().zip(out_row.iter()) {
                                *xi += ai;
                            }
                        });
                }
                AttnKind::Full(fl) => {
                    let (qg, k, v) = crate::prof::time_n_ev(crate::prof::QKV, n, || {
                        (
                            fl.q_w.apply_bt_rows(&h, n), // [n, heads*2*dh]
                            fl.k_w.apply_bt_rows(&h, n), // [n, kv_dim]
                            fl.v_w.apply_bt_rows(&h, n), // [n, kv_dim]
                        )
                    });
                    let mut att = vec![0.0f32; n * heads * dh];
                    // s42 (Plan B): prepare (KV 書き込み + QK-Norm + RoPE) をトークン順
                    // に逐次実行 (トークン逐次経路と同一の演算列・decode 経路の
                    // full_attn_core と共用)。attend は読み取り専用のため全トークンの
                    // prepare 完了後に並列実行しても結果は同一 (bit-exact)。
                    // (s50: prof は QKNORM / ROPE バケットに分割計上 — 演算列は
                    // full_attn_prepare 1 回呼び出しと同一 (順序不変) = bit-exact。
                    // KV 書き込みは llama prefill と同様に個別計上しない)
                    let mut q = vec![0.0f32; n * heads * dh];
                    for i in 0..n {
                        let q_r = &mut q[i * heads * dh..(i + 1) * heads * dh];
                        let mut kn = crate::prof::time_layer(crate::prof::QKNORM, li, || {
                            self.full_attn_qknorm(
                                fl,
                                &qg[i * heads * 2 * dh..(i + 1) * heads * 2 * dh],
                                &k[i * kv_dim..(i + 1) * kv_dim],
                                q_r,
                            )
                        });
                        crate::prof::time_layer(crate::prof::ROPE, li, || {
                            self.full_attn_rope(q_r, &mut kn, pos0 + i)
                        });
                        let (kc, vc) = cache.kv_bufs(li, 1);
                        kc[(pos0 + i) * kv_dim..][..kv_dim].copy_from_slice(&kn);
                        vc[(pos0 + i) * kv_dim..][..kv_dim]
                            .copy_from_slice(&v[i * kv_dim..(i + 1) * kv_dim]);
                    }
                    // s48 (C1): cuda ビルドでは attend を GPU バッチ (dno_attention_prefill
                    // 1 発 + D2H 後のホスト sigmoid ゲート) に置き換えて試行する。
                    // prepare は上記のとおりトークン逐次のまま (KV はホスト cache に
                    // 書き込み済み = 冪等) で、chain 失敗時は下の既存 rayon attend に
                    // 落ちる (数値は現行どおり bit 一致・発動ゲート n>=gemm_min_m()
                    // で短文は CPU のまま・`DENNO_GPU_ATTN=0` で全経路現行に戻る)。
                    // s57 (T4): full attention は attn_core の別バケット (ATT_FULL) に
                    // 分離。GPU バッチ (dno_attention_prefill) が実際に走ったかを
                    // 実行先ラベルとして計測に載せる (`att_done` をそのまま返すだけ =
                    // 分岐・演算列は不変)。
                    crate::prof::time_layer_n_ev(crate::prof::ATT_FULL, li, n, || {
                        #[cfg(feature = "cuda")]
                        let att_done =
                            self.full_attn_attend_gpu(&qg, &q, cache, li, pos0, n, &mut att);
                        #[cfg(not(feature = "cuda"))]
                        let att_done = false;
                        if !att_done {
                            // s42: attend を (i, head) 並列で実行 (旧実装はトークン逐次 —
                            // O(n²) の単スレッド部分が長 prefill の律速)。1 head 分の
                            // scores → softmax → probs@V → gate はトークン逐次経路と同一の
                            // 演算列のため bit-exact・rayon スレッド数非依存 (決定論検証:
                            // qwen38_27b_shape_prefill_thread_count_bit_exact)。scores は
                            // タスクローカル (共有スクラッチなし・書き込み先は各タスク専属の
                            // att チャンクのみ)。
                            let (kcr, vcr) = cache.kv_bufs_ref(li);
                            att.par_chunks_exact_mut(dh)
                                .enumerate()
                                .for_each(|(t, att_head)| {
                                    let i = t / heads;
                                    let hh = t % heads;
                                    self.full_attn_attend_head(
                                        &qg[i * heads * 2 * dh..(i + 1) * heads * 2 * dh],
                                        &q[i * heads * dh..(i + 1) * heads * dh],
                                        kcr,
                                        vcr,
                                        pos0 + i + 1,
                                        hh,
                                        att_head,
                                    );
                                });
                        }
                        (att_done, att_done)
                    });
                    let out = crate::prof::time_n_ev(crate::prof::OPROJ, n, || {
                        fl.o_w.apply_bt_rows(&att, n)
                    });
                    // s42: 残差加算は行独立 → 行並列 (bit-exact)
                    x.par_chunks_exact_mut(e)
                        .zip(out.par_chunks_exact(e))
                        .for_each(|(x_row, out_row)| {
                            for (xi, ai) in x_row.iter_mut().zip(out_row.iter()) {
                                *xi += ai;
                            }
                        });
                }
            }
            lchk(&format!("L{li:02} attn"), &x[(n - 1) * e..]); // 恒久診断 (env ゲート・未指定時 no-op)
            // --- FFN: SwiGLU をバッチ一括 ---
            // s57 (T4): prefill 内の全 op に 1 call あたりのバッチ幅 (n) を載せる
            // (decode 側は chain の 1 行 = 1）。集計のみで演算列は不変。
            let mut h2 = vec![0.0f32; n * e];
            // s42: rmsnorm は行独立 → 行並列 (bit-exact)
            crate::prof::time_n(crate::prof::RMSNORM, n, || {
                h2.par_chunks_exact_mut(e)
                    .zip(x.par_chunks_exact(e))
                    .for_each(|(h2_row, x_row)| {
                        let r = rmsnorm(x_row, &layer.ln2_w, e, self.eps);
                        h2_row.copy_from_slice(&r);
                    });
            });
            let (gate, up) = crate::prof::time_n_ev(crate::prof::GATE_UP, n, || {
                (layer.gate_w.apply_bt_rows(&h2, n), layer.up_w.apply_bt_rows(&h2, n))
            });
            let mut hh = vec![0.0f32; n * ff];
            // s42: SwiGLU は行独立 → 行並列 (bit-exact)
            crate::prof::time_n(crate::prof::SWIGLU, n, || {
                hh.par_chunks_exact_mut(ff)
                    .zip(gate.par_chunks_exact(ff).zip(up.par_chunks_exact(ff)))
                    .for_each(|(hh_row, (gate_row, up_row))| {
                        for ((gv, uv), hhv) in gate_row
                            .iter()
                            .zip(up_row.iter())
                            .zip(hh_row.iter_mut())
                        {
                            *hhv = gv / (1.0 + (-gv).exp()) * uv;
                        }
                    });
            });
            let down = crate::prof::time_n_ev(crate::prof::DOWN, n, || {
                layer.down_w.apply_bt_rows(&hh, n)
            });
            // s42: 残差加算は行独立 → 行並列 (bit-exact)
            x.par_chunks_exact_mut(e)
                .zip(down.par_chunks_exact(e))
                .for_each(|(x_row, down_row)| {
                    for (xi, di) in x_row.iter_mut().zip(down_row.iter()) {
                        *xi += di;
                    }
                });
            lchk(&format!("L{li:02} ffn"), &x[(n - 1) * e..]); // 恒久診断 (env ゲート・未指定時 no-op)
        }

        // 最終 norm + logits は最終トークンのみ (トークン逐次経路の最終値と同一)。
        // FINAL_NORM を個別計上 (LM_HEAD は s49 の logits() ラッパで計上済み)
        let hf = crate::prof::time(crate::prof::FINAL_NORM, || {
            rmsnorm(&x[(n - 1) * e..], &self.norm_w, e, self.eps)
        });
        let logits = self.logits(&hf);
        lchk("final hf", &hf); // 恒久診断 (env ゲート・未指定時 no-op)
        lchk("final logits", &logits); // 恒久診断 (env ゲート・未指定時 no-op)
        lchk_argmax("final logits", &logits); // 恒久診断 (env ゲート・未指定時 no-op)
        cache.advance(n);
        Ok(logits)
    }
}

// ---------------------------------------------------------------------------
// 恒久診断 (env ゲート・未指定時 no-op): 層別 hidden-state checksum ダンプ
// (DENNO_DEBUG_LAYER_CHKSUM=1)
// 27B 出力破綻の初発散層特定用 (HANDOFF_NEXT 第14セッション)。llama.cpp 側の
// リファレンスツール (/work/llama.cpp/tools/lchk) と同一書式の [LCHK] 行を
// stderr に出力し、同一プロンプト・同一 GGUF での層別比較に使う。
// DENNO_DEBUG_LAYER_CHKSUM_FILE が設定されていれば、さらにステージ順に最終
// トークン行の raw f32 (LE) を追記する (dim 単位比較用)。
// env ゲート済み・未指定時は no-op (恒久保持)。
// ---------------------------------------------------------------------------
fn lchk_enabled() -> bool {
    static G: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *G.get_or_init(|| std::env::var("DENNO_DEBUG_LAYER_CHKSUM").ok().as_deref() == Some("1"))
}

fn lchk(stage: &str, v: &[f32]) {
    if !lchk_enabled() {
        return;
    }
    let mut bitsum: u64 = 0;
    let mut maxabs: f32 = 0.0;
    let mut ssq: f64 = 0.0;
    for &f in v {
        bitsum = bitsum.wrapping_add(f.to_bits() as u64);
        let a = f.abs();
        if a > maxabs {
            maxabs = a;
        }
        ssq += (f as f64) * (f as f64);
    }
    eprintln!(
        "[LCHK] {stage} len={} bitsum={bitsum:016x} maxabs={maxabs:.6e} ssq={ssq:.10e}",
        v.len()
    );
    if let Ok(path) = std::env::var("DENNO_DEBUG_LAYER_CHKSUM_FILE") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = f.write_all(&bytes);
        }
    }
}

fn lchk_argmax(stage: &str, logits: &[f32]) {
    if !lchk_enabled() {
        return;
    }
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    eprintln!("[LCHK] {stage} argmax={bi} logit={bv:.6e}");
}

// ===========================================================================
// C4: デバイス常駐 decode chain (GDN 状態 GPU ミラー)。DESIGN_C4.md 参照。
// 雛形は llama P17h solo chain (forward_decode_gpu + cuda.rs chain_*)。
// decode 1 ステップを「埋め込み 1 回 H2D → 全層 GPU 内連結 → 最終 logits
// 1 回 sync+D2H」にする。GDN 状態 (conv/ssm) は GPU ミラーに置き、ホスト状態は
// ステップ末尾 (logits download の sync 後) に staged D2H → フラッシュして
// 正であり続ける (snapshot / CPU 再実行契約維持)。

/// partial interleaved rope テーブル 1 行分 [cos(n_rot/2) | sin(n_rot/2)]。
/// pos 依存のため decode ステップ毎に生成 + chain_q35_rope_pi_upload する。
/// 値は rope_partial_interleaved と**同一式** (ang = pos * theta^(-2d/n_rot) を
/// f32 で cos/sin) でビット一致が契約 — dno_qknorm_rope_pi の回転結果を
/// CPU 経路と bit-exact にするため (テーブル値の一致 = 回転演算の一致)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn q35_rope_pi_table(pos: usize, n_rot: usize, theta: f32) -> Vec<f32> {
    let half = n_rot / 2;
    let mut tab = vec![0.0f32; 2 * half];
    let pos = pos as f32;
    for d in 0..half {
        let ang = pos * theta.powf(-2.0 * d as f32 / n_rot as f32);
        tab[d] = ang.cos();
        tab[half + d] = ang.sin();
    }
    tab
}

/// C4: decode chain が使える構造か (全 int8 線形 + lm_head 量子化 +
/// beta/alpha は Fp32)。beta/alpha は dno_matvec_f32 が fp32 重みの生スライスを
/// 受ける (Linear::F32 に uid は無いため q35_w_dev は (model_id, layer, tag) キー
/// でキャッシュ) — Q8 化されているモデルは chain 不発動 (現ローダは lin_ctl で
/// 常に Fp32 を維持するため実モデルは常にこの条件を満たす)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn q35_chain_compatible(m: &Qwen35) -> bool {
    if m.lm_head.as_ref().map_or(true, |l| l.quant8().is_none()) {
        return false;
    }
    for layer in &m.layers {
        match &layer.attn {
            AttnKind::Gdn(g) => {
                for w in [&g.qkv_w, &g.gate_w, &g.out_w] {
                    if w.quant8().is_none() {
                        return false;
                    }
                }
                if !matches!(&g.beta_w, Linear::F32 { .. })
                    || !matches!(&g.alpha_w, Linear::F32 { .. })
                {
                    return false;
                }
            }
            AttnKind::Full(f) => {
                for w in [&f.q_w, &f.k_w, &f.v_w, &f.o_w] {
                    if w.quant8().is_none() {
                        return false;
                    }
                }
            }
        }
        for w in [&layer.gate_w, &layer.up_w, &layer.down_w] {
            if w.quant8().is_none() {
                return false;
            }
        }
    }
    true
}

#[cfg(feature = "cuda")]
impl Qwen35 {
    /// C4: decode 1 ステップのデバイス常駐 chain。llama forward_decode_gpu と同一
    /// 骨格 + GDN 層 (chain_q35_gdn_scan) / full 層 (chain_q35_fullattn) の分岐。
    /// 成功: logits download の sync で全 staged async D2H が完了 → Full 層の
    /// rope 済み k/v 行と GDN 状態 (conv/ssm) をホストキャッシュへ反映 → advance(1)。
    /// 失敗: advance 前に None を返す → forward 全体を従来経路でやり直し
    /// (GDN 状態・KV 行は冪等・ホスト未汚染。ミラー側は pos 位置合わせで自動再 seed)。
    fn forward_decode_chain(
        &mut self,
        token: u32,
        cache: &mut KvCache,
        pos0: usize,
    ) -> Option<Vec<f32>> {
        use crate::cuda::{
            chain_download, chain_linear_grouped, chain_q35_fullattn, chain_q35_gdn_scan,
            chain_q35_kv_stage_begin, chain_q35_kv_stage_flush, chain_q35_rope_pi_upload,
            chain_q35_step_reserve, chain_q35_step_reserve_aq, chain_rmsnorm, chain_swiglu,
            chain_upload, gpu_attention_enabled, q35_chain_shapes_ok, q35_ch_att_bytes,
            q35_ch_qkv_bytes, q35_fns_available, q35_kv_stage_bytes, CH_ACT, CH_ATT, CH_DOWN,
            CH_GA, CH_H, CH_LOGITS, CH_O, CH_QKV, CH_X,
        };
        let e = self.e;
        let ff = self.ff;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let dh = self.dh;
        let kv_dim = kv_heads * dh;
        let n_layers = self.layers.len();
        let (nk, nv, dk, dv, d_conv) = (
            self.gdn_p.n_k_heads,
            self.gdn_p.n_v_heads,
            self.gdn_p.head_k_dim,
            self.gdn_p.head_v_dim,
            self.gdn_p.d_conv,
        );
        let eps = self.eps;

        // --- 事前ゲート (チェーン途中失敗の回避)。非対応は静かに従来経路へ委譲 ---
        // DENNO_GPU_ATTN=0 (escape hatch) / 新カーネル未生成 (kernels_elem.ptx に
        // dno_matvec_f32 等が無い → lazy fetch 失敗) / 非対応構造 / 形状制約違反。
        if !gpu_attention_enabled() || !q35_fns_available() || !q35_chain_compatible(self) {
            return None;
        }
        if !q35_chain_shapes_ok(e, ff, dh, heads, kv_heads, self.n_rot, dk, dv, d_conv) {
            return None;
        }
        // 検証 (pos>=ctx / tok>=vocab) は従来経路の契約 (bail!) に委譲する
        if pos0 >= self.ctx || token as usize >= self.vocab {
            return None;
        }
        if t_total_overflow(pos0, cache.max_len()) {
            return None;
        }

        let key = cache.kv_id();
        let mid = self.mid;
        // KV 容量 (forward 側で kv_bufs(0, 1) 済みだが単独呼び出しでも安全なように)
        let _ = cache.kv_bufs(0, 1);

        // rope テーブル (pos 依存・1 ステップ 1 回。CPU rope_partial_interleaved と
        // 同一式 → dno_qknorm_rope_pi の QK-Norm+rope を bit-exact にする)
        let tab = q35_rope_pi_table(pos0, self.n_rot, self.theta);
        chain_q35_rope_pi_upload(&tab)?;

        // CH_QKV / CH_ATT / aq / kv_stage をステップ開始時に一括確保
        // (「最初に全体サイズ」規約)。2994feb 時点では CH_QKV の事前確保が
        // 無く、GDN 層の grouped GEMV (qkv+z) 直後に chain_q35_gdn_scan 頭部の
        // chain_ensure_buf(CH_QKV, width*4) が grow 再確保で in-flight 書き込み先を
        // 解放し、dno_gdn_scan が未初期化の新バッファを読んでいた (実機 E2E の
        // 出力非決定論 + 冒頭乱れの直接原因)。aq は初回 down GEMV (k=ff) の
        // 再確保が直前 GEMV の in-flight 読み出し元を解放するのを避ける。
        // (s50: GDN 状態のステージは廃止 — 新状態は GPU ミラーに留める。pinned
        // ステージの事前確保は kv_stage のみ残る。)
        chain_q35_step_reserve(
            CH_QKV,
            q35_ch_qkv_bytes(heads, dh, kv_heads, nk, nv, dk, dv),
        )?;
        chain_q35_step_reserve(CH_ATT, q35_ch_att_bytes(nv, dv, heads, dh))?;
        chain_q35_step_reserve_aq(e.max(ff).max(nv * dv).max(heads * dh))?;
        chain_q35_kv_stage_begin(q35_kv_stage_bytes(n_layers, kv_dim))?;

        // 埋め込み 1 行 → residual stream デバイス常駐開始
        chain_upload(CH_X, &self.embed[token as usize * e..][..e], e, 1)?;

        let mut pending: Option<usize> = None;
        // GDN 層の層番号 (昇順) — KV ステージフラッシュの skip リスト
        let mut gdn_layers: Vec<usize> = Vec::new();

        for (li, layer) in self.layers.iter().enumerate() {
            crate::telemetry_layer_begin(li as u16);
            crate::prof::time_full(crate::prof::RMSNORM, 1, true, || {
                chain_rmsnorm(layer.attn_ln1(), eps, pending, CH_H, e, 1, li * 3)
            })?;
            match &layer.attn {
                AttnKind::Gdn(g) => {
                    gdn_layers.push(li);
                    // qkv + z 投影: CH_QKV = [qkv (2nk*dk+nv*dv) | z (nv)]
                    crate::prof::time_full(crate::prof::QKV, 1, true, || -> Option<()> {
                        let qkv = g.qkv_w.quant8()?;
                        let zw = g.gate_w.quant8()?;
                        chain_linear_grouped(&[qkv, zw], CH_H, CH_QKV, 1)
                    })?;
                    // beta/alpha (fp32 matvec・CH_QKV テール) → dno_gdn_scan
                    // (y → CH_ATT・新状態は staged async D2H・ping-pong 面 交換)
                    // s57 (T4): decode chain は全 op がデバイス常駐で走るため
                    // 実行先ラベルは gpu 固定。attn_core は GDN バケットに分離。
                    crate::prof::time_full(crate::prof::ATT_GDN, 1, true, || -> Option<()> {
                        let (bw, aw) = match (&g.beta_w, &g.alpha_w) {
                            (Linear::F32 { w: bw, .. }, Linear::F32 { w: aw, .. }) => (bw, aw),
                            _ => return None, // q35_chain_compatible で除外済み
                        };
                        chain_q35_gdn_scan(
                            mid,
                            key,
                            li,
                            n_layers,
                            cache,
                            pos0,
                            nk,
                            nv,
                            dk,
                            dv,
                            d_conv,
                            self.gdn_p.eps,
                            1.0 / (dk as f32).sqrt(),
                            &g.conv1d,
                            &g.ssm_a,
                            &g.ssm_dt,
                            &g.ssm_norm,
                            bw,
                            aw,
                            e,
                        )
                    })?;
                    // out_proj: y (CH_ATT) → e (CH_O)
                    crate::prof::time_full(crate::prof::OPROJ, 1, true, || -> Option<()> {
                        let ow = g.out_w.quant8()?;
                        chain_linear_grouped(&[ow], CH_ATT, CH_O, 1)
                    })?;
                }
                AttnKind::Full(f) => {
                    // qkv 投影: CH_QKV = [qg (heads*2dh) | k (kv_dim) | v (kv_dim)]
                    crate::prof::time_full(crate::prof::QKV, 1, true, || -> Option<()> {
                        let qw = f.q_w.quant8()?;
                        let kw = f.k_w.quant8()?;
                        let vw = f.v_w.quant8()?;
                        chain_linear_grouped(&[qw, kw, vw], CH_H, CH_QKV, 1)
                    })?;
                    // QK-Norm + partial rope (q → CH_Q35_Q・k → KV ミラー pos0 行直書き)
                    // → GQA attention → sigmoid 出力ゲート (k/v 行は staged async D2H)
                    // s57 (T4): full attention バケット (ATT_FULL)・devices chain 上で走る。
                    crate::prof::time_full(crate::prof::ATT_FULL, 1, true, || {
                        chain_q35_fullattn(
                            mid,
                            key,
                            li,
                            n_layers,
                            cache,
                            pos0,
                            heads,
                            kv_heads,
                            dh,
                            self.n_rot,
                            eps,
                            &f.q_norm_w,
                            &f.k_norm_w,
                        )
                    })?;
                    // o_proj: att (CH_ATT) → e (CH_O)
                    crate::prof::time_full(crate::prof::OPROJ, 1, true, || -> Option<()> {
                        let ow = f.o_w.quant8()?;
                        chain_linear_grouped(&[ow], CH_ATT, CH_O, 1)
                    })?;
                }
            }
            pending = Some(CH_O);
            // --- FFN (llama chain と同一): residual+rmsnorm → gate/up → swiglu → down
            crate::prof::time_full(crate::prof::RMSNORM, 1, true, || {
                chain_rmsnorm(&layer.ln2_w, eps, pending, CH_H, e, 1, li * 3 + 1)
            })?;
            crate::prof::time_full(crate::prof::GATE_UP, 1, true, || -> Option<()> {
                let gw = layer.gate_w.quant8()?;
                let uw = layer.up_w.quant8()?;
                chain_linear_grouped(&[gw, uw], CH_H, CH_GA, 1)
            })?;
            crate::prof::time_full(crate::prof::SWIGLU, 1, true, || {
                chain_swiglu(CH_GA, CH_ACT, self.ff, 1)
            })?;
            crate::prof::time_full(crate::prof::DOWN, 1, true, || -> Option<()> {
                let dw = layer.down_w.quant8()?;
                chain_linear_grouped(&[dw], CH_ACT, CH_DOWN, 1)
            })?;
            pending = Some(CH_DOWN);
            crate::telemetry_layer_flush();
        }

        // 最終 norm + lm_head (s49: logits download を D2H バケットに分離 — この
        // download の sync がステップ全体の GPU 実行の完了待ちを吸収するため、
        // lm_head 計算だけのバケットに分けないと実機プロファイルの
        // 「lm_head」が sync 待ち込みで過大帰属になる。観測のみで数値経路不変)
        crate::prof::time_full(crate::prof::LM_HEAD, 1, true, || -> Option<()> {
            chain_rmsnorm(&self.norm_w, eps, pending, CH_H, e, 1, n_layers * 3)?;
            let lmq = self.lm_head.as_ref().unwrap().quant8()?;
            chain_linear_grouped(&[lmq], CH_H, CH_LOGITS, 1)
        })?;
        // s55 (T1 step2・観測のみ): 従来 `D2H_LOGITS` 1 本だった区間を
        //   WAIT = logits D2H を発行する直前に打つ追加 sync の所要
        //        (= GPU 側の未完了処理待ち。転送は含まない)
        //   COPY = 従来区間そのもの (chain_download の発行 + 既存 sync + pin コピー)
        // に分離する。**追加 sync は `prof::enabled()` が真のときだけ**打つ —
        // 非 prof 経路は if の else 側に落ち、追加の GPU 同期は 1 回も入らない
        // (数値経路・既定挙動は不変)。WAIT + COPY は従来の D2H_LOGITS と一致する。
        // 最初のステップ (prefill 直後の 1 点目) は logits D2H の手前に既存 sync が無く、
        // WAIT が「prefill 全体の完了待ち」を含むため平均を汚す — 1 点目のみ
        // D2H_LOGITS_PF_HELD に退避する (2 点目以降が decode の定常値)。
        // 1 点目の判定 = プロンプト長 p = pos0 (この chain は decode 1 トークンのみを
        // 受けるので pos0 は常にプロンプト末尾 = 直前の prefill 後に sync 済みの長さ)。
        // 各ステップ成功時に cache.advance(1) するため pos0 + 1 == cache.len() が常に
        // 成り立ち、従来の `cache.len() > pos0` は decode では常に false だった
        // (= 待ち時間が全部 pfhold = 集計外に入り wait 行が 1 度も出ない)。
        if crate::prof::enabled() {
            let op = crate::prof::d2h_logits_split_op(pos0, cache.len());
            crate::prof::time(op, || crate::cuda::chain_sync_pending())?;
        }
        let logits = crate::prof::time_if_on(crate::prof::D2H_LOGITS_COPY, || {
            chain_download(CH_LOGITS, self.vocab)
        })?;
        // ホスト反映: Full 層 k/v 行のみ (s50: GDN 状態はステップ毎に戻さず
        // GPU ミラーを権威のままにする — 次の prefill 冒頭 / CPU フォールバック時に
        // chain_q35_gdn_flush_pending が回復する。ステップ毎 D2H ~75MB/step が
        // decode の支配項だったため)
        crate::prof::time_full(crate::prof::D2H_KV_FLUSH, 1, true, || {
            chain_q35_kv_stage_flush(cache, n_layers, kv_dim, pos0, &gdn_layers)
        })?;
        cache.advance(1);
        Some(logits)
    }
}

/// pos0+1 がキャッシュ上限を超えるか (chain は従来経路へ委譲)。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn t_total_overflow(pos0: usize, max_len: usize) -> bool {
    pos0 + 1 > max_len
}

#[cfg(test)]
mod tests {
    //! A3 (Qwen3.8-27B スケール): 実 GGUF が配置されていない環境でも 27B と
    //! 同一構造 (64 層 = 48 GDN + 16 full・nv=3*nk・untied lm_head・VLM ラッパ
    //! config) のモデルをテスト内で合成し、ローダ → forward → KV 予算までを
    //! 検証する。実 GGUF の dequant (UD-IQ2_XXS 混合量子等) は既存の
    //! dequant_ref.rs / qwen35_model.rs (ignored・実モデル) が担保する。

    use super::*;
    use std::path::PathBuf;

    // ---- 合成 27B シェイプ (実 27B の構造比を保持した小型 dims) ----
    const E: usize = 96; // hidden (実 5120)
    const HEADS: usize = 6; // full attention q heads (実 24)
    const KVH: usize = 2; // full attention kv heads (実 4)
    const DH: usize = 16; // head_dim (実 256)
    const FF: usize = 128; // intermediate (実 17408)
    const VOCAB: usize = 128; // (実 248320)
    const NL: usize = 64; // 64 層 = 48 GDN + 16 full (interval 4)
    const NK: usize = 2; // linear_num_key_heads (実 16)
    const NV: usize = 6; // linear_num_value_heads (実 48 = 3x)
    const DK: usize = 8; // linear_key_head_dim (実 128)
    const DV: usize = 8; // linear_value_head_dim (実 128)
    const DCONV: usize = 4; // linear_conv_kernel_dim

    const KD: usize = NK * DK; // 16
    const VD: usize = NV * DV; // 48
    const QKVD: usize = 2 * KD + VD; // 80
    const KVDIM: usize = KVH * DH; // 32
    const QTOT: usize = HEADS * DH * 2; // 192 (q|gate 融合)

    fn lcg_next(s: &mut u32) -> f32 {
        *s = s.wrapping_mul(1103515245).wrapping_add(12345);
        (((*s >> 16) & 0xff) as f32 / 255.0 - 0.5) * 0.16
    }

    fn wr_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn wr_u64(v: &mut Vec<u8>, x: u64) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn wr_str(v: &mut Vec<u8>, s: &str) {
        wr_u64(v, s.len() as u64);
        v.extend_from_slice(s.as_bytes());
    }

    /// F32 テンソルのみの最小 GGUF v3 ライタ (テスト専用)。
    /// ne は GGUF 順 (ne[0] = 最速次元 = in_features)。
    /// `extra_kv` は general.architecture に追加する u64 KV (例: block_count)。
    fn write_gguf_f32(
        path: &PathBuf,
        tensors: &[(String, Vec<u64>, Vec<f32>)],
        extra_kv: &[(&str, u64)],
    ) {
        const ALIGN: usize = 32;
        let mut b = Vec::new();
        wr_u32(&mut b, 0x4655_4747); // magic "GGUF"
        wr_u32(&mut b, 3); // version
        wr_u64(&mut b, tensors.len() as u64);
        wr_u64(&mut b, 1 + extra_kv.len() as u64); // metadata kv count
        wr_str(&mut b, "general.architecture");
        wr_u32(&mut b, 8); // GT_STRING
        wr_str(&mut b, "qwen3_5");
        for (key, val) in extra_kv {
            wr_str(&mut b, key);
            wr_u32(&mut b, 10); // GT_U64
            wr_u64(&mut b, *val);
        }
        let mut off = 0u64;
        let mut data = Vec::new();
        for (name, ne, vals) in tensors {
            wr_str(&mut b, name);
            wr_u32(&mut b, ne.len() as u32);
            for d in ne {
                wr_u64(&mut b, *d);
            }
            wr_u32(&mut b, 0); // GGML_F32
            let pad = (ALIGN - (off as usize % ALIGN)) % ALIGN;
            if pad != 0 {
                off += pad as u64;
                data.extend(std::iter::repeat(0u8).take(pad));
            }
            wr_u64(&mut b, off);
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
            off += (vals.len() * 4) as u64;
        }
        while b.len() % ALIGN != 0 {
            b.push(0);
        }
        b.extend_from_slice(&data);
        std::fs::write(path, b).unwrap();
    }

    /// 27B シェイプの合成モデルディレクトリ (config.json + model.gguf) を作る。
    /// `wrapper` = VLM ラッパ形式 (text_config、9B/27B と同形) / false = 平文 config。
    fn build_dir(name: &str, wrapper: bool, nk: usize, nv: usize) -> PathBuf {
        build_dir_x(name, wrapper, nk, nv, false, |_| {})
    }

    /// 拡張版。`mtp_nextn` = true で blk.NL を nextn (MTP) 層として追加し、
    /// GGUF KV に block_count=NL+1 / nextn_predict_layers=1 を足す (実 27B の
    /// blk.64 と同一構成のテンソル名)。`mutate` は書き出し前のテンソルリスト
    /// 調整 (形状不一致テスト用)。
    fn build_dir_x(
        name: &str,
        wrapper: bool,
        nk: usize,
        nv: usize,
        mtp_nextn: bool,
        mutate: impl FnOnce(&mut Vec<(String, Vec<u64>, Vec<f32>)>),
    ) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("denno_a3_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let kd = nk * DK;
        let vd = nv * DV;
        let qkvd = 2 * kd + vd;
        let mut st = 0x12_34_56_78u32;
        // 全ての乱数生成をこのクロージャに集約 (st への直接アクセスは競合するため)
        let mut w = |n: usize| -> Vec<f32> { (0..n).map(|_| lcg_next(&mut st)).collect() };
        let mut ts: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
        ts.push(("token_embd.weight".into(), vec![E as u64, VOCAB as u64], w(VOCAB * E)));
        ts.push(("output.weight".into(), vec![E as u64, VOCAB as u64], w(VOCAB * E)));
        ts.push(("output_norm.weight".into(), vec![E as u64], vec![1.0; E]));
        for i in 0..NL {
            let p = format!("blk.{i}.");
            let is_gdn = i % 4 != 3;
            ts.push((format!("{p}attn_norm.weight"), vec![E as u64], vec![1.0; E]));
            ts.push((format!("{p}post_attention_norm.weight"), vec![E as u64], vec![1.0; E]));
            ts.push((format!("{p}ffn_gate.weight"), vec![E as u64, FF as u64], w(FF * E)));
            ts.push((format!("{p}ffn_up.weight"), vec![E as u64, FF as u64], w(FF * E)));
            ts.push((format!("{p}ffn_down.weight"), vec![FF as u64, E as u64], w(E * FF)));
            if is_gdn {
                ts.push((format!("{p}attn_qkv.weight"), vec![E as u64, qkvd as u64], w(qkvd * E)));
                ts.push((format!("{p}attn_gate.weight"), vec![E as u64, vd as u64], w(vd * E)));
                ts.push((format!("{p}ssm_beta.weight"), vec![E as u64, nv as u64], w(nv * E)));
                ts.push((format!("{p}ssm_alpha.weight"), vec![E as u64, nv as u64], w(nv * E)));
                ts.push((
                    format!("{p}ssm_conv1d.weight"),
                    vec![DCONV as u64, qkvd as u64],
                    w(qkvd * DCONV),
                ));
                // ssm_a は負値 (-exp(A_log))・ssm_norm はゲート重み (~1)
                let a: Vec<f32> = w(nv).iter().map(|v| -(0.5 + v.abs())).collect();
                ts.push((format!("{p}ssm_a"), vec![nv as u64], a));
                ts.push((format!("{p}ssm_dt.bias"), vec![nv as u64], w(nv)));
                ts.push((format!("{p}ssm_norm.weight"), vec![DV as u64], vec![1.0; DV]));
                ts.push((format!("{p}ssm_out.weight"), vec![vd as u64, E as u64], w(E * vd)));
            } else {
                ts.push((format!("{p}attn_q.weight"), vec![E as u64, QTOT as u64], w(QTOT * E)));
                ts.push((format!("{p}attn_k.weight"), vec![E as u64, KVDIM as u64], w(KVDIM * E)));
                ts.push((format!("{p}attn_v.weight"), vec![E as u64, KVDIM as u64], w(KVDIM * E)));
                ts.push((format!("{p}attn_q_norm.weight"), vec![DH as u64], vec![1.0; DH]));
                ts.push((format!("{p}attn_k_norm.weight"), vec![DH as u64], vec![1.0; DH]));
                ts.push((
                    format!("{p}attn_output.weight"),
                    vec![(HEADS * DH) as u64, E as u64],
                    w(E * HEADS * DH),
                ));
            }
        }
        if mtp_nextn {
            // blk.NL: 実 27B の blk.64 と同一構成のテンソル名 (中身はダミー —
            // トランクは config の NL 層のみで nextn は読まれない)
            let p = format!("blk.{NL}.");
            ts.push((format!("{p}attn_norm.weight"), vec![E as u64], vec![1.0; E]));
            ts.push((format!("{p}post_attention_norm.weight"), vec![E as u64], vec![1.0; E]));
            ts.push((format!("{p}ffn_gate.weight"), vec![E as u64, FF as u64], w(FF * E)));
            ts.push((format!("{p}ffn_up.weight"), vec![E as u64, FF as u64], w(FF * E)));
            ts.push((format!("{p}ffn_down.weight"), vec![FF as u64, E as u64], w(E * FF)));
            ts.push((format!("{p}attn_q.weight"), vec![E as u64, QTOT as u64], w(QTOT * E)));
            ts.push((format!("{p}attn_k.weight"), vec![E as u64, KVDIM as u64], w(KVDIM * E)));
            ts.push((format!("{p}attn_v.weight"), vec![E as u64, KVDIM as u64], w(KVDIM * E)));
            ts.push((format!("{p}attn_q_norm.weight"), vec![DH as u64], vec![1.0; DH]));
            ts.push((format!("{p}attn_k_norm.weight"), vec![DH as u64], vec![1.0; DH]));
            ts.push((
                format!("{p}attn_output.weight"),
                vec![(HEADS * DH) as u64, E as u64],
                w(E * HEADS * DH),
            ));
            ts.push((
                format!("{p}nextn.eh_proj.weight"),
                vec![(2 * E) as u64, E as u64],
                w(2 * E * E),
            ));
            ts.push((format!("{p}nextn.enorm.weight"), vec![E as u64], vec![1.0; E]));
            ts.push((format!("{p}nextn.hnorm.weight"), vec![E as u64], vec![1.0; E]));
        }
        mutate(&mut ts);
        let kvs: Vec<(&str, u64)> = if mtp_nextn {
            vec![
                ("qwen3_5.block_count", (NL + 1) as u64),
                ("qwen3_5.nextn_predict_layers", 1),
            ]
        } else {
            vec![]
        };
        write_gguf_f32(&dir.join("model.gguf"), &ts, &kvs);

        let layer_types: Vec<&str> = (0..NL)
            .map(|i| if i % 4 == 3 { "full_attention" } else { "linear_attention" })
            .collect();
        let tc = serde_json::json!({
            "attention_bias": false,
            "attn_output_gate": true,
            "dtype": "bfloat16",
            "eos_token_id": 151643,
            "full_attention_interval": 4,
            "head_dim": DH,
            "hidden_act": "silu",
            "hidden_size": E,
            "intermediate_size": FF,
            "layer_types": layer_types,
            "linear_conv_kernel_dim": DCONV,
            "linear_key_head_dim": DK,
            "linear_num_key_heads": nk,
            "linear_num_value_heads": nv,
            "linear_value_head_dim": DV,
            "max_position_embeddings": 512,
            "mtp_num_hidden_layers": 1,
            "num_attention_heads": HEADS,
            "num_hidden_layers": NL,
            "num_key_value_heads": KVH,
            "output_gate_type": "swish",
            "rms_norm_eps": 1e-6,
            "rope_parameters": {
                "mrope_interleaved": true,
                "mrope_section": [2, 1, 1],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.25
            },
            "vocab_size": VOCAB
        });
        let cfg = if wrapper {
            serde_json::json!({
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": tc,
                "tie_word_embeddings": false,
                "vision_config": {"model_type": "qwen3_5", "hidden_size": 64}
            })
        } else {
            let mut c = tc.clone();
            c["model_type"] = serde_json::json!("qwen3_5");
            c["tie_word_embeddings"] = serde_json::json!(false);
            c["architectures"] = serde_json::json!(["Qwen3_5ForCausalLM"]);
            c
        };
        std::fs::write(
            dir.join("config.json"),
            serde_json::to_string_pretty(&cfg).unwrap(),
        )
        .unwrap();
        dir
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn load_int8(dir: &PathBuf) -> Qwen35 {
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap())
                .unwrap();
        Qwen35::load(dir, &cfg, QuantMode::Int8).expect("合成 27B モデルのロードに失敗")
    }

    /// 構造検証: 64 層 = 48 GDN + 16 full・untied lm_head・int8 再量子化・
    /// KV 予算 (GDN 層を除外した bytes_per_token)。
    #[test]
    fn qwen38_27b_shape_structure_and_kv_budget() {
        let dir = build_dir("struct", true, NK, NV);
        let m = load_int8(&dir);
        assert_eq!(m.layers.len(), NL);
        let n_gdn = m.layers.iter().filter(|l| matches!(l.attn, AttnKind::Gdn(_))).count();
        assert_eq!(n_gdn, 48, "GDN 層数");
        assert_eq!(m.layers.len() - n_gdn, 16, "full attention 層数");
        assert_eq!((m.gdn_p.n_k_heads, m.gdn_p.n_v_heads, m.gdn_p.head_k_dim), (NK, NV, DK));
        assert_eq!((m.e, m.heads, m.kv_heads, m.dh, m.ff, m.vocab), (E, HEADS, KVH, DH, FF, VOCAB));
        assert!(m.lm_head.is_some(), "untied lm_head (27B は output.weight を持つ)");
        assert!(m.quant_bytes > 0, "int8 再量子化経路");
        // KV 予算: KV を使うのは full attention 16 層のみ
        let c = m.new_cache();
        assert_eq!(c.n_layers(), NL);
        assert_eq!(c.bytes_per_token(), 16 * KVDIM * 2 * 4, "KV 層のみで計量 (A3)");
        assert_eq!(c.max_len(), 512);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// forward: 有限 logits・決定論 (連続 2 回 bit 一致)・キャッシュ連続性
    /// (3+1 == 4 bit 一致)・decode 継続。
    #[test]
    fn qwen38_27b_shape_forward_determinism_and_continuity() {
        let dir = build_dir("fwd", true, NK, NV);
        let mut m = load_int8(&dir);
        let toks: Vec<u32> = [7u32, 33, 100, 64].to_vec();

        let mut c1 = m.new_cache();
        let l1 = m.forward(&toks, &mut c1).unwrap();
        assert_eq!(l1.len(), VOCAB);
        assert!(l1.iter().all(|v| v.is_finite()));

        let mut c2 = m.new_cache();
        let l2 = m.forward(&toks, &mut c2).unwrap();
        assert_eq!(bits(&l1), bits(&l2), "決定論違反");

        // キャッシュ連続性: 3+1 == 4 (bit 一致)
        let mut c3 = m.new_cache();
        let _ = m.forward(&toks[..3], &mut c3).unwrap();
        let l3 = m.forward(&[toks[3]], &mut c3).unwrap();
        let mut c4 = m.new_cache();
        let l4 = m.forward(&toks[..4], &mut c4).unwrap();
        assert_eq!(bits(&l3), bits(&l4), "キャッシュ連続性違反 (3+1 != 4)");

        // decode 継続
        let mut c5 = m.new_cache();
        let _ = m.forward(&toks[..3], &mut c5).unwrap();
        for t in &toks[3..] {
            let l = m.forward(&[*t], &mut c5).unwrap();
            assert!(l.iter().all(|v| v.is_finite()));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A1 契約の 27B シェイプ再確認: バッチ prefill (n>1) がトークン逐次と
    /// bit-exact (i32 累積の順序不変 + F32 は apply_bt_rows の順序保存)。
    #[test]
    fn qwen38_27b_shape_prefill_batch_bit_exact() {
        let dir = build_dir("prefill", true, NK, NV);
        let mut m = load_int8(&dir);
        let toks: Vec<u32> = [5u32, 41, 97, 12, 78].to_vec();

        // バッチ prefill → 続き 1 トークン
        let mut cb = m.new_cache();
        let lb1 = m.forward(&toks, &mut cb).unwrap();
        let lb2 = m.forward(&[9u32], &mut cb).unwrap();

        // トークン逐次
        let mut cs = m.new_cache();
        let mut ls = Vec::new();
        for t in &toks {
            ls.push(m.forward(&[*t], &mut cs).unwrap());
        }
        let ls2 = m.forward(&[9u32], &mut cs).unwrap();

        assert_eq!(bits(&lb1), bits(ls.last().unwrap()), "prefill batch != tokenwise");
        assert_eq!(bits(&lb2), bits(&ls2), "prefill 後の decode が不一致");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// s46 (A5): prefill を 2 回の forward に分割しても単一 forward と bit 一致
    /// (engine が generation prompt 境界で prefill を分割して base 境界の
    /// KV+GDN スナップショットを退避するための契約)。logits・base 境界までの
    /// KV 行 (単一 forward の先頭 base.len() 行と一致 = engine が prefix store に
    /// 退避する内容の正当性)・分割完了後の GDN 状態 (単一 forward と同一位置)
    /// を比較し、分割後の decode 継続も一致させる。
    #[test]
    fn qwen38_27b_shape_split_prefill_bit_exact() {
        let dir = build_dir("split", true, NK, NV);
        let mut m = load_int8(&dir);
        let base = [5u32, 41, 97, 12, 78, 33, 100];
        let gen = [64u32, 7]; // generation prompt 相当 (base の直後に来る 2 トークン)

        // 単一 forward (base + gen)
        let mut c1 = m.new_cache();
        let l1 = m.forward(&base.iter().chain(gen.iter()).copied().collect::<Vec<_>>(), &mut c1).unwrap();

        // 分割 forward: base → (base 境界の KV スナップショット) → gen
        let mut c2 = m.new_cache();
        let _ = m.forward(&base, &mut c2).unwrap();
        let (snap_k, snap_v) = {
            let kv_len = c2.len();
            c2.snapshot(kv_len)
        };
        let l2 = m.forward(&gen, &mut c2).unwrap();
        // GDN 状態は分割完了後 (pos = base+gen) で単一 forward と同一位置
        let snap_gdn = c2.gdn_snapshot();

        assert_eq!(bits(&l1), bits(&l2), "分割 prefill の logits が単一 forward と不一致");
        // base 境界の KV スナップショット = 単一 forward の先頭 base.len() 行と一致
        // (GDN 層は両側空ベクタなので自明に一致)
        let kv_len = c1.len();
        let (k1, v1) = c1.snapshot(kv_len);
        for (li, (a, b)) in snap_k.iter().zip(k1.iter()).enumerate() {
            assert_eq!(a[..], b[..a.len()], "層 {li} の K 行 (base 境界まで) が不一致");
        }
        for (li, (a, b)) in snap_v.iter().zip(v1.iter()).enumerate() {
            assert_eq!(a[..], b[..a.len()], "層 {li} の V 行 (base 境界まで) が不一致");
        }
        // GDN 状態 (分割完了後 pos = base+gen vs 単一 forward 同一位置):
        // トークン逐次なので分割で状態遷移列は不変
        let g1 = c1.gdn_snapshot();
        for (li, (a, b)) in snap_gdn.iter().zip(g1.iter()).enumerate() {
            match (a, b) {
                (None, None) => {}
                (Some((ac, as_)), Some((bc, bs))) => {
                    assert_eq!(ac, bc, "層 {li} の conv 状態が不一致");
                    assert_eq!(as_, bs, "層 {li} の ssm 状態が不一致");
                }
                _ => panic!("層 {li} の GDN スロット有無が不一致"),
            }
        }
        // 分割経路の直後 decode も単一経路と bit 一致
        let d1 = m.forward(&[9u32], &mut c1).unwrap();
        let d2 = m.forward(&[9u32], &mut c2).unwrap();
        assert_eq!(bits(&d1), bits(&d2), "分割後の decode 継続が不一致");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// s50: decode chain の GDN 状態遅延フラッシュ (chain_q35_gdn_flush_pending) の
    /// chain レベル bit 一致検証 (合成 int8 27B モデル)。旧挙動 (s49 以前: 各ステップ
    /// 末尾に staged async D2H → sync 後ホスト反映) は「各ステップ後に明示 flush」で
    /// 再現し、s50 の遅延 (ステップ毎のホスト反映なし・GPU ミラー権威) と
    /// ①ステップ毎 logits ②prefill 境界継続 (forward_prefill 冒頭の flush でホスト
    /// 回復してから計算) ③attn 無効化による CPU フォールバック継続 (forward 冒頭の
    /// flush で回復) を bits 比較する。遅延フラッシュの転送は bit そのまま (演算なし)
    /// のため全比較が bit 一致するはず — 不一致なら遅延経路が数値経路に触れている
    /// (=バグ)。
    /// 実行: `cargo test --features cuda --lib qwen38_27b_shape_chain_gdn_deferred -- --ignored`
    /// (実機 GPU + kernels_elem.ptx 再生成後・コンテナ内は GPU 無しのため非実行)
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn qwen38_27b_shape_chain_gdn_deferred_flush_bit_exact() {
        fn argmax(v: &[f32]) -> u32 {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32
        }
        let _g =
            crate::cuda::test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        let _ctx = crate::cuda::CudaCtx::new().expect("GPU 必須 (ignored テスト)");
        let dir = build_dir("c4defer", true, NK, NV);
        let prompt = vec![5u32, 11, 23];
        let steps = 4usize;

        // --- 経路 A: 旧挙動相当 (毎ステップ明示 flush → ホスト GDN 状態は常に最新) ---
        let mut ma = load_int8(&dir);
        let mut ca = ma.new_cache();
        let mut la = ma.forward(&prompt, &mut ca).unwrap();
        let pre_a = bits(&la); // 初回 prefill logits (経路間比較用)
        let mut seq: Vec<u32> = Vec::new();
        let mut hist_a: Vec<Vec<u32>> = Vec::new();
        for step in 0..steps {
            let t = argmax(&la);
            seq.push(t);
            let pos0 = prompt.len() + step;
            la = ma
                .forward_decode_chain(t, &mut ca, pos0)
                .expect("decode chain 発動 (int8 + lm_head Q8 + beta/alpha F32)");
            // 旧挙動相当: ステップ毎にホストへ反映 (s49 以前の staged async D2H +
            // sync 後 flush と同一効果・遅延分は直前ステップの 1 ステップ分のみ)
            crate::cuda::chain_q35_gdn_flush_pending(&mut ca)
                .expect("明示フラッシュ (旧挙動相当) は成功する");
            hist_a.push(bits(&la));
        }
        // ② prefill 境界継続 (n>1 → forward_prefill・A は冒頭 flush が no-op)
        let pa = ma.forward(&[64u32, 7], &mut ca).unwrap();
        // ③ attn 無効化 → chain 不発 → forward 冒頭 flush (no-op) → CPU 経路
        let prev = crate::cuda::set_gpu_attention_enabled(false);
        let fa = ma.forward(&[seq[0]], &mut ca).unwrap();
        crate::cuda::set_gpu_attention_enabled(prev);

        // --- 経路 B: s50 遅延 (flush 無し・GPU ミラーが権威・ホストは古いまま) ---
        let mut mb = load_int8(&dir);
        let mut cb = mb.new_cache();
        let mut lb = mb.forward(&prompt, &mut cb).unwrap();
        assert_eq!(bits(&lb), pre_a, "初回 prefill logits が不一致");
        for (step, &t) in seq.iter().enumerate() {
            let pos0 = prompt.len() + step;
            lb = mb
                .forward_decode_chain(t, &mut cb, pos0)
                .expect("decode chain 発動");
            // flush しない — chain は GPU ミラーの ping-pong 面を読むため出力は不変
            assert_eq!(
                bits(&lb),
                hist_a[step],
                "①ステップ毎 logits が旧挙動 (毎ステップ flush) と不一致 step={step}"
            );
        }
        // ② 遅延中の prefill: forward_prefill 冒頭の flush でホスト回復してから計算
        let pb = mb.forward(&[64u32, 7], &mut cb).unwrap();
        assert_eq!(bits(&pb), bits(&pa), "②prefill 境界継続が不一致");
        // ③ 遅延中の CPU フォールバック: chain 不発 → forward 冒頭 flush で回復 →
        //    CPU 経路 (GDN はホスト状態の GPU カーネル・full attention は CPU attend)
        let prev = crate::cuda::set_gpu_attention_enabled(false);
        let fb = mb.forward(&[seq[0]], &mut cb).unwrap();
        crate::cuda::set_gpu_attention_enabled(prev);
        assert_eq!(bits(&fb), bits(&fa), "③CPU フォールバック継続が不一致");
        // ④ 最終 GDN ホスト状態も旧挙動と bit 一致 (転送が bit そのままの裏取り)
        let gdn_bits = |c: &KvCache| -> Vec<Option<(Vec<u32>, Vec<u32>)>> {
            c.gdn_snapshot()
                .iter()
                .map(|o| o.as_ref().map(|(a, b)| (bits(a), bits(b))))
                .collect()
        };
        assert_eq!(gdn_bits(&cb), gdn_bits(&ca), "④最終 GDN ホスト状態が不一致");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// s42 (Plan B): CPU prefill の rayon 並列化 (full attention attend の (i, head)
    /// 並列・rmsnorm/swiglu/残差の行並列・matmul_bt_i8 の j ブロック並列) が
    /// rayon スレッド数に依存しないこと (bits 完全一致)。par_* は pool.install()
    /// 中ではそのローカルプールで実行されるため、同一 forward を 1/2/7/16 スレッド
    /// で回して prefill logits と継続 decode を比較する。1 スレッド = 逐次相当で、
    /// 既存の qwen38_27b_shape_prefill_batch_bit_exact (トークン逐次との bit-exact)
    /// と合わせて「並列化しても逐次と bit 一致」を両面から担保する。
    /// (実 GPU が無い環境では GDN/projection は全て CPU 経路で走る)
    #[test]
    fn qwen38_27b_shape_prefill_thread_count_bit_exact() {
        let dir = build_dir("rayon", true, NK, NV);
        let toks: Vec<u32> = (0..24u32).map(|i| (i * 5 + 3) % VOCAB as u32).collect();

        // 指定スレッド数のローカルプールで prefill (n=24) → 続き 1 トークン decode
        let run = |threads: usize| -> (Vec<f32>, Vec<f32>) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut m = load_int8(&dir);
            let mut c = m.new_cache();
            pool.install(|| {
                let a = m.forward(&toks, &mut c).unwrap();
                let b = m.forward(&[toks[0]], &mut c).unwrap();
                (a, b)
            })
        };

        let (ref1, dec1) = run(1);
        for th in [2usize, 7, 16] {
            let (lp, decp) = run(th);
            assert_eq!(
                bits(&ref1),
                bits(&lp),
                "prefill logits がスレッド数 {th} で不一致"
            );
            assert_eq!(
                bits(&dec1),
                bits(&decp),
                "継続 decode がスレッド数 {th} で不一致"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// int8 再量子化が fp32 と相関のある出力になること ( cosine > 0.8・両者有限)。
    /// 完全ランダム (未学習) の重みでは int8 のテンソルごとの丸め誤差が 64 層で
    /// 累積するため、閾値は 0.8 に留める (実モデルでの品質は qwen35_int8_probe.rs
    /// の実機 E2E が担保)。
    #[test]
    fn qwen38_27b_shape_int8_vs_fp32_close() {
        let dir = build_dir("quant", true, NK, NV);
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap()).unwrap();
        let mut m32 = Qwen35::load(&dir, &cfg, QuantMode::Fp32).unwrap();
        let mut mi8 = load_int8(&dir);
        let toks: Vec<u32> = [7u32, 33, 100, 64].to_vec();
        let l32 = m32.forward(&toks, &mut m32.new_cache()).unwrap();
        let li8 = mi8.forward(&toks, &mut mi8.new_cache()).unwrap();
        assert!(l32.iter().all(|v| v.is_finite()) && li8.iter().all(|v| v.is_finite()));
        let dot: f32 = l32.iter().zip(li8.iter()).map(|(a, b)| a * b).sum();
        let n32: f32 = l32.iter().map(|v| v * v).sum::<f32>().sqrt();
        let n8: f32 = li8.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (n32 * n8);
        assert!(cos > 0.8, "int8 と fp32 の logits cosine が低い: {cos}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 平文 config (text_config ラッパ無し) もロードできる (27B がラッパ無しで
    /// 提供された場合の保険。実 27B はラッパ形式を想定)。
    #[test]
    fn qwen38_27b_shape_plain_config_loads() {
        let dir = build_dir("plain", false, NK, NV);
        let mut m = load_int8(&dir);
        let mut c = m.new_cache();
        let l = m.forward(&[3u32, 50, 11], &mut c).unwrap();
        assert!(l.iter().all(|v| v.is_finite()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 実 27B と同形の MTP (nextn) 層 (blk.NL + block_count=NL+1 KV) があって
    /// も、推論トランクは config の NL 層のみでロードに成功する (nextn 層は
    /// 読み飛ばされる — blk.NL を通常層としては読まない)。
    #[test]
    fn qwen38_27b_shape_mtp_nextn_layer_skipped() {
        let dir = build_dir_x("mtp", true, NK, NV, true, |_| {});
        let mut m = load_int8(&dir);
        assert_eq!(
            m.layers.len(),
            NL,
            "トランクは config の NL 層のみ (nextn 層を含まない)"
        );
        let mut c = m.new_cache();
        let l1 = m.forward(&[3u32, 50, 11], &mut c).unwrap();
        let l2 = m.forward(&[3u32, 50, 11], &mut m.new_cache()).unwrap();
        assert!(l1.iter().all(|v| v.is_finite()));
        assert_eq!(bits(&l1), bits(&l2), "MTP 層の存在で決定論が変わってはいけない");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// blk.NL が存在するのに nextn 構成を持たない (nextn.eh_proj 無し) 場合は
    /// ロードを拒否する (config/GGUF 不整合を静かに無視しない)。
    #[test]
    fn qwen38_27b_shape_non_nextn_extra_block_rejected() {
        let dir = build_dir_x("mtpbad", true, NK, NV, true, |ts| {
            ts.retain(|(n, _, _)| !n.contains(&format!("blk.{NL}.nextn.")));
        });
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap()).unwrap();
        let err = match Qwen35::load(&dir, &cfg, QuantMode::Int8) {
            Err(e) => e,
            Ok(_) => panic!("nextn 構成を持たない超過ブロックのロードが通ってしまった"),
        };
        assert!(
            format!("{err:#}").contains("nextn"),
            "unexpected error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// full_attention 層の attn_output が想定外の形状 ([2*heads*dh, e] —
    /// 「o_proj の手前で [att|gate] を連結し o_proj 後に gate を適用する」
    /// 別構造の想定) なら、静かに誤読せずロードを拒否する (27B out-gate
    /// 調査の反映: 27B の正は [heads*dh, e]・不一致はロード時に明確化)。
    #[test]
    fn qwen38_27b_shape_attn_output_shape_mismatch_rejected() {
        let dir = build_dir_x("oshape", true, NK, NV, false, |ts| {
            for (n, ne, vals) in ts.iter_mut() {
                if n == "blk.3.attn_output.weight" {
                    *ne = vec![(2 * HEADS * DH) as u64, E as u64];
                    *vals = vec![0.01; 2 * HEADS * DH * E];
                }
            }
        });
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap()).unwrap();
        let err = match Qwen35::load(&dir, &cfg, QuantMode::Int8) {
            Err(e) => e,
            Ok(_) => panic!("attn_output 形状不一致のロードが通ってしまった"),
        };
        assert!(
            format!("{err:#}").contains("attn_output"),
            "unexpected error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// attn_q が q|gate 融合でない想定外構造 (out = heads*dh のみ・ゲート無し)
    /// なら、head ごとの [q|gate] split を静かに誤読しないようロードを拒否する。
    #[test]
    fn qwen38_27b_shape_fused_q_gate_shape_mismatch_rejected() {
        let dir = build_dir_x("qshape", true, NK, NV, false, |ts| {
            for (n, ne, vals) in ts.iter_mut() {
                if n == "blk.3.attn_q.weight" {
                    *ne = vec![E as u64, (HEADS * DH) as u64];
                    *vals = vec![0.01; E * HEADS * DH];
                }
            }
        });
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap()).unwrap();
        let err = match Qwen35::load(&dir, &cfg, QuantMode::Int8) {
            Err(e) => e,
            Ok(_) => panic!("attn_q 形状不一致のロードが通ってしまった"),
        };
        assert!(format!("{err:#}").contains("attn_q"), "unexpected error: {err:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// GDN ヘッド比が整数倍でない構成はロード時に明確なエラーで拒否される
    /// (静かに誤マップしないための A3 ガード)。
    #[test]
    fn gdn_head_ratio_must_divide() {
        let dir = build_dir("ratio", true, 4, 6); // 6 % 4 != 0
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap()).unwrap();
        let err = match Qwen35::load(&dir, &cfg, QuantMode::Int8) {
            Err(e) => e,
            Ok(_) => panic!("GDN ヘッド比違反がロードを通過してしまった"),
        };
        assert!(
            format!("{err:#}").contains("整数倍ではありません"),
            "unexpected error: {err:#}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A4: GDN 状態スナップショット (prompt 境界) + KV 復元からの継続 forward が
    /// フル prefill と **bit 一致**する。GDN 状態はシーケンス全体の関数なので、
    /// 「prompt 全体を処理した直後の conv + SSM 状態」を KV スナップショットと
    /// ペアで保存・復元すれば、続きの prefill/decode はフル prefill と同じ計算に
    /// なる (キャッシュ連続性 3+1==4 と同じ性質を、キャッシュをまたいで成立させる)。
    #[test]
    fn qwen38_27b_shape_gdn_snapshot_restore_bit_exact() {
        let dir = build_dir("snap", true, NK, NV);
        let mut m = load_int8(&dir);
        let p: Vec<u32> = [7u32, 33, 100, 64].to_vec();
        let q: Vec<u32> = [9u32, 41].to_vec();

        fn gdn_bits(g: &[Option<(Vec<f32>, Vec<f32>)>]) -> Vec<Option<(Vec<u32>, Vec<u32>)>> {
            g.iter()
                .map(|o| {
                    o.as_ref().map(|(a, b)| {
                        (
                            a.iter().map(|x| x.to_bits()).collect(),
                            b.iter().map(|x| x.to_bits()).collect(),
                        )
                    })
                })
                .collect()
        }

        // 参照: フル prefill (P++Q)
        let mut cref = m.new_cache();
        let lref = m.forward(&[p.clone(), q.clone()].concat(), &mut cref).unwrap();
        let sref = gdn_bits(&cref.gdn_snapshot());
        assert!(sref.iter().any(|o| o.is_some()), "GDN 状態が存在する");

        // スナップショット経路: P を forward → prompt 境界で KV + GDN 状態を退避 →
        // 新しいキャッシュへ復元 → Q を継続 forward
        let mut c1 = m.new_cache();
        let _ = m.forward(&p, &mut c1).unwrap();
        let (k, v) = c1.snapshot(p.len());
        let gdn = c1.gdn_snapshot();

        let mut c2 = m.new_cache();
        c2.restore(&k, &v, p.len());
        c2.gdn_restore(&gdn);
        let l2 = m.forward(&q, &mut c2).unwrap();
        assert_eq!(bits(&l2), bits(&lref), "スナップショット復元後の継続がフル prefill と不一致");
        assert_eq!(
            gdn_bits(&c2.gdn_snapshot()),
            sref,
            "復元経路の最終 GDN 状態がフル prefill と不一致"
        );

        // 対照実験: KV のみ復元 (GDN 状態を復元しない) では出力が変わる =
        // テストが状態の復元を実際に検証している
        let mut c3 = m.new_cache();
        c3.restore(&k, &v, p.len());
        let l3 = m.forward(&q, &mut c3).unwrap();
        assert_ne!(
            bits(&l3),
            bits(&lref),
            "GDN 状態を復元しなくても一致してしまった (テストが無 tooth)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A4: 合成 27B シェイプモデルを実際の Engine に載せたマルチターン会話で、
    /// 2 ターン目以降が prefix cache エントリ (KV + GDN 状態) の復元で処理され、
    /// 出力が「別エンジンでフル prefill した参照」と完全一致する。
    /// engine.rs の RecurStateModel テスト (engine 経路の形状) に対し、
    /// こちらは実 Qwen35 実装 (GDN 層 + full attention 層の混在) での検証。
    #[test]
    fn qwen38_27b_shape_engine_multi_turn_entry_restore() {
        use crate::engine::{ChatMessage, Engine, GenParams, PrefixHit};
        use crate::testutil::dummy_tokenizer;

        fn params(max_tokens: usize) -> GenParams {
            GenParams { max_tokens, temperature: Some(0.0), ..Default::default() }
        }
        fn msgs(turns: &[(&str, &str)]) -> Vec<ChatMessage> {
            turns
                .iter()
                .map(|(role, content)| ChatMessage {
                    role: (*role).into(),
                    content: (*content).into(),
                })
                .collect()
        }

        let dir = build_dir("a4engine", true, NK, NV);
        let model = load_int8(&dir);
        let mut e = Engine::from_parts(Box::new(model), dummy_tokenizer().unwrap(), "a4", None);
        let ref_m = load_int8(&dir);
        let mut r = Engine::from_parts(
            Box::new(ref_m),
            dummy_tokenizer().unwrap(),
            "a4-ref",
            None,
        );

        // turn1: フル prefill
        let out1 = e.chat(None, &msgs(&[("user", "t0")]), &params(2), |_| {}).unwrap();
        assert_eq!(e.prefix_cache_entries(), 1, "prefill 直後にエントリが作られる");
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);

        // turn2: turn1 の prompt を完全に含むマルチターン prompt
        let m2 = msgs(&[("user", "t0"), ("assistant", &out1.text), ("user", "t15")]);
        let expect2 = r.chat(None, &m2, &params(2), |_| {}).unwrap();
        let out2 = e.chat(None, &m2, &params(2), |_| {}).unwrap();
        assert_eq!(
            e.last_prefix_hit(),
            PrefixHit::Entry,
            "GDN 状態スナップショット経由で復元されるべき"
        );
        assert_eq!(out2, expect2, "エントリ復元後の出力がフル prefill の参照と不一致");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ===========================================================================
    // C4: デバイス常駐 decode chain の CPU 検証 (rope テーブル bit 一致 / mid
    // 一意性 / chain ゲート)。GPU 経路自体は qwen35_solo_chain_gpu_matches_cpu
    // (ignored・実機 + kernels_elem.ptx 再生成後) が担保する。
    // ===========================================================================

    /// C4: rope テーブル (q35_rope_pi_table) が rope_partial_interleaved と
    /// **ビット一致**する回転を与えること。dno_qknorm_rope_pi はテーブル値で
    /// `x1*c - x2*s` / `x1*s + x2*c` を計算するため、テーブル値の bit 一致 =
    /// 回転結果の bit 一致 (CPU 経路との exactness 契約の根拠)。
    /// partial 契約 (d >= n_rot は無回転) も検証する。
    #[test]
    fn q35_rope_pi_table_bit_matches_rope_partial_interleaved() {
        let mut st = 0xC4_00_01u32;
        let mut next = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1103515245).wrapping_add(12345);
            (((*s >> 16) & 0xff) as f32 / 255.0 - 0.5) * 2.0
        };
        let theta = 10000.0f32;
        // 合成モデル実次元 (n_rot=4/dh=16) を含む・n_rot <= dh・偶数のみ
        for &(n_rot, dh) in &[(2usize, 8usize), (4, 16), (16, 64), (128, 128)] {
            for &pos in &[0usize, 1, 7, 63, 1000, 5000] {
                let tab = q35_rope_pi_table(pos, n_rot, theta);
                assert_eq!(tab.len(), n_rot, "テーブル長 = [cos(n_rot/2)|sin(n_rot/2)]");
                let heads = 2usize;
                let x0: Vec<f32> = (0..heads * dh).map(|_| next(&mut st)).collect();
                // (a) リファレンス: rope_partial_interleaved
                let mut a = x0.clone();
                rope_partial_interleaved(&mut a, heads, dh, n_rot, pos, theta);
                // (b) テーブル値で同一回転 (dno_qknorm_rope_pi と同一演算列)
                let half = n_rot / 2;
                let mut b = x0.clone();
                for h in 0..heads {
                    let base = h * dh;
                    for d in 0..half {
                        let (c, s) = (tab[d], tab[half + d]);
                        let (x1, x2) = (b[base + 2 * d], b[base + 2 * d + 1]);
                        b[base + 2 * d] = x1 * c - x2 * s;
                        b[base + 2 * d + 1] = x1 * s + x2 * c;
                    }
                    // partial 契約: n_rot 以降は無回転
                    assert_eq!(
                        bits(&b[base + n_rot..base + dh]),
                        bits(&x0[base + n_rot..base + dh]),
                        "partial rope: d >= n_rot は無回転 (n_rot={n_rot} dh={dh} pos={pos})"
                    );
                }
                assert_eq!(
                    bits(&a),
                    bits(&b),
                    "rope テーブル経路の回転が rope_partial_interleaved と不一致 (n_rot={n_rot} pos={pos})"
                );
            }
        }
    }

    /// C4: model_id (mid) がロード毎に一意に採番されること。cuda chain は fp32
    /// 重み (beta/alpha/QK-Norm/conv 系) を (mid, layer, tag) でデバイスキャッシュ
    /// する — Linear::F32 に uid が無いための代替キーで、同 dir 再ロードでも
    /// 別 mid が採番されないとモデル切替時に誤キャッシュを参照する。
    #[test]
    fn q35_model_id_unique_per_load() {
        let dir = build_dir("c4mid", true, NK, NV);
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap())
                .unwrap();
        let m1 = Qwen35::load(&dir, &cfg, QuantMode::Int8).unwrap();
        let m2 = Qwen35::load(&dir, &cfg, QuantMode::Int8).unwrap();
        let m3 = Qwen35::load(&dir, &cfg, QuantMode::Fp32).unwrap();
        assert_ne!(m1.mid, 0, "mid は 1 起点採番 (0 は未使用マーカ)");
        assert_ne!(m1.mid, m2.mid, "同 dir の再ロードでも別 mid");
        assert_ne!(m1.mid, m3.mid);
        assert_ne!(m2.mid, m3.mid);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// C4: chain 適合ゲート (q35_chain_compatible)。合成 27B シェイプ
    /// (全 int8 線形 + untied lm_head Q8 + beta/alpha F32) で適合、Fp32 ロード
    /// (quant8 無し) では不適合。形状ゲート (cuda.rs q35_chain_shapes_ok) は
    /// cuda cfg テスト (q35_chain_shapes_ok_gates) が担保する。
    #[test]
    fn q35_chain_gate_on_synthetic_model() {
        let dir = build_dir("c4gate", true, NK, NV);
        let cfg: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("config.json")).unwrap())
                .unwrap();
        let mi = Qwen35::load(&dir, &cfg, QuantMode::Int8).unwrap();
        assert_eq!(mi.n_rot, DH / 4, "partial_rotary_factor 0.25");
        assert!(q35_chain_compatible(&mi), "int8 ロードは chain 適合");
        let mf = Qwen35::load(&dir, &cfg, QuantMode::Fp32).unwrap();
        assert!(
            !q35_chain_compatible(&mf),
            "Fp32 ロードは lm_head/線形に quant8 が無く chain 不適合"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// C4: カーネル形状ゲート (cuda.rs q35_chain_shapes_ok・純関数)。合成モデルの
    /// 実次元で発動し、dno_gdn_scan 前提違反 (dk!=dv / d_conv!=4 / dk>256 /
    /// dk=0) や QK-Norm 側前提違反 (dh>1024 / n_rot>dh / n_rot 奇数 / GQA 非整除)
    /// で拒否されること。
    #[cfg(feature = "cuda")]
    #[test]
    fn q35_chain_shapes_ok_gates() {
        use crate::cuda::q35_chain_shapes_ok;
        let n_rot = DH / 4; // partial_rotary_factor 0.25
        assert!(q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, n_rot, DK, DV, DCONV));
        // dno_gdn_scan 前提 (kernels/gdn.cu)
        assert!(!q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, n_rot, DK + 1, DV, DCONV), "dk != dv");
        assert!(!q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, n_rot, DK, DV, 3), "d_conv != 4");
        assert!(!q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, n_rot, 512, 512, DCONV), "dk > 256");
        assert!(!q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, n_rot, 0, 0, DCONV), "dk == 0");
        // rmsnorm / QK-Norm / attention 側の前提
        assert!(!q35_chain_shapes_ok(0, FF, DH, HEADS, KVH, n_rot, DK, DV, DCONV), "e == 0");
        assert!(!q35_chain_shapes_ok(E, 0, DH, HEADS, KVH, n_rot, DK, DV, DCONV), "ff == 0");
        assert!(!q35_chain_shapes_ok(E, FF, 0, HEADS, KVH, n_rot, DK, DV, DCONV), "dh == 0");
        assert!(!q35_chain_shapes_ok(E, FF, 2048, HEADS, KVH, 1024, DK, DV, DCONV), "dh > 1024");
        assert!(!q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, DH + 2, DK, DV, DCONV), "n_rot > dh");
        assert!(!q35_chain_shapes_ok(E, FF, DH, HEADS, KVH, 3, DK, DV, DCONV), "n_rot 奇数");
        assert!(!q35_chain_shapes_ok(E, FF, DH, 6, 4, n_rot, DK, DV, DCONV), "GQA 非整除");
        assert!(!q35_chain_shapes_ok(E, FF, DH, 0, KVH, n_rot, DK, DV, DCONV), "heads == 0");
    }

    /// C4: solo decode chain (GDN 状態 GPU ミラー) が forward_decode_chain で
    /// **発動**し (Some 返却を明示検証 — 教訓⑭: 「テスト緑」は経路発動の証拠に
    /// ならない)、CPU 経路と greedy 一致 + logits 近似一致すること。
    /// 実機 GPU + kernels_elem.ptx 再生成後のみ意味を持つ (新カーネルが PTX に
    /// 無い場合は lazy fetch が None を返し expect で失敗する = 発動検証)。
    /// 実行: `cargo test --features cuda --lib qwen35_solo_chain -- --ignored`
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn qwen35_solo_chain_gpu_matches_cpu() {
        fn argmax(v: &[f32]) -> u32 {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0 as u32
        }
        let _g = crate::cuda::test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        let dir = build_dir("c4gpu", true, NK, NV);
        let prompt = vec![5u32, 11, 23];
        let steps = 4;

        // CPU 参照 (GPU attention 無効 → chain は不発動 → 従来 per-op 経路)
        let prev = crate::cuda::set_gpu_attention_enabled(false);
        let mut cpu_m = load_int8(&dir);
        let mut cpu_cache = cpu_m.new_cache();
        let mut ref_logits = cpu_m.forward(&prompt, &mut cpu_cache).unwrap();
        let mut ref_seq = Vec::new();
        let mut ref_hist: Vec<Vec<f32>> = Vec::new();
        for _ in 0..steps {
            let t = argmax(&ref_logits);
            ref_seq.push(t);
            ref_logits = cpu_m.forward(&[t], &mut cpu_cache).unwrap();
            ref_hist.push(ref_logits.clone());
        }
        crate::cuda::set_gpu_attention_enabled(true);

        // GPU solo chain (C4: 埋め込み 1 回 H2D → 全層 GPU 内連結 → logits 1 回 D2H)
        let mut gpu_m = load_int8(&dir);
        let mut gpu_cache = gpu_m.new_cache();
        let mut logits = gpu_m.forward(&prompt, &mut gpu_cache).unwrap();
        for (step, want_tok) in ref_seq.iter().enumerate() {
            let t = argmax(&logits);
            assert_eq!(t, *want_tok, "greedy token step={step}");
            let pos0 = prompt.len() + step;
            let lg = gpu_m
                .forward_decode_chain(t, &mut gpu_cache, pos0)
                .expect("solo chain must fire (C4: int8 + lm_head Q8 + beta/alpha F32 + 新カーネル)");
            logits = lg;
            let maxd = logits
                .iter()
                .zip(ref_hist[step].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            // attention スコア/softmax のみ緩和契約 (P17g と同一)・GDN は bit 一致不要
            assert!(maxd < 0.05, "logits maxabs diff step={step}: {maxd}");
        }
        crate::cuda::set_gpu_attention_enabled(prev);
        println!("C4 solo chain (GDN GPU mirror) matches CPU OK ({steps} steps)");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
