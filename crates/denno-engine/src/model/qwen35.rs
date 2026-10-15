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
}

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
        // A3: repeat_interleave 規約 (gdn.rs: v-head h → k-head h*nk/nv) は
        // n_v_heads が n_k_heads の整数倍であることを要求する (0.8B 1x・9B 2x・
        // 27B 3x)。割り切れない構成は静かに誤マップするためロード時に拒否する。
        if gdn_p.n_v_heads % gdn_p.n_k_heads != 0 {
            bail!(
                "linear_num_value_heads ({}) が linear_num_key_heads ({}) の整数倍ではありません (GDN k-head repeat_interleave 規約)",
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

        let n_layer = cfg_u64(tc, "num_hidden_layers")? as usize;
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

        for i in 0..n_layer {
            let p = format!("blk.{i}.");
            let ln1_w = dt(&format!("{p}attn_norm.weight"))?;
            let ln2_w = dt(&format!("{p}post_attention_norm.weight"))?;
            let gate_w = lin(dt(&format!("{p}ffn_gate.weight"))?, e, ff);
            let up_w = lin(dt(&format!("{p}ffn_up.weight"))?, e, ff);
            let down_w = lin(dt(&format!("{p}ffn_down.weight"))?, ff, e);

            let attn = if f.tensor(&format!("{p}ssm_a")).is_some() {
                // GDN (linear_attention) 層
                AttnKind::Gdn(GdnLayer {
                    ln1_w,
                    qkv_w: lin(dt(&format!("{p}attn_qkv.weight"))?, e, qkv_dim),
                    gate_w: lin(dt(&format!("{p}attn_gate.weight"))?, e, vd),
                    beta_w: lin_ctl(dt(&format!("{p}ssm_beta.weight"))?, e, gdn_p.n_v_heads),
                    alpha_w: lin_ctl(dt(&format!("{p}ssm_alpha.weight"))?, e, gdn_p.n_v_heads),
                    conv1d: dt(&format!("{p}ssm_conv1d.weight"))?,
                    ssm_a: dt(&format!("{p}ssm_a"))?,
                    ssm_dt: dt(&format!("{p}ssm_dt.bias"))?,
                    ssm_norm: dt(&format!("{p}ssm_norm.weight"))?,
                    out_w: lin(dt(&format!("{p}ssm_out.weight"))?, vd, e),
                })
            } else {
                // full_attention 層 (Q+gate 融合: out = heads*[q(dh)|gate(dh)])
                let q_total = heads * dh * 2;
                AttnKind::Full(FullLayer {
                    ln1_w,
                    q_w: lin(dt(&format!("{p}attn_q.weight"))?, e, q_total),
                    k_w: lin(dt(&format!("{p}attn_k.weight"))?, e, kv_heads * dh),
                    v_w: lin(dt(&format!("{p}attn_v.weight"))?, e, kv_heads * dh),
                    q_norm_w: dt(&format!("{p}attn_q_norm.weight"))?,
                    k_norm_w: dt(&format!("{p}attn_k_norm.weight"))?,
                    o_w: lin(dt(&format!("{p}attn_output.weight"))?, heads * dh, e),
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
        let embed = dt("token_embd.weight")?;
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
            Some(lin(dt("output.weight")?, e, vocab))
        };
        let norm_w = match f.tensor("output_norm.weight") {
            Some(_) => dt("output_norm.weight")?,
            None => dt("model.norm.weight")?,
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
    /// まで (prefill バッチ経路と共用・演算順序はトークン逐次経路と同一のため
    /// bit-exact)。
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
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads;

        let mut k = k_row.to_vec();
        let v = v_row;

        // QK-Norm (head 単位・RoPE 前) + q の分離
        let mut q = vec![0.0f32; heads * dh];
        for hh in 0..heads {
            let src = &qg[hh * 2 * dh..hh * 2 * dh + dh];
            let wq = rmsnorm(src, &lf.q_norm_w, dh, self.eps);
            q[hh * dh..(hh + 1) * dh].copy_from_slice(&wq);
        }
        for kvh in 0..kv_heads {
            let seg = &mut k[kvh * dh..(kvh + 1) * dh];
            let wn = rmsnorm(seg, &lf.k_norm_w, dh, self.eps);
            seg.copy_from_slice(&wn);
        }

        // partial interleaved RoPE
        rope_partial_interleaved(&mut q, heads, dh, self.n_rot, pos, self.theta);
        rope_partial_interleaved(&mut k, kv_heads, dh, self.n_rot, pos, self.theta);

        // K/V をキャッシュへ
        let (kc, vc) = cache.kv_bufs(li, 1);
        kc[pos * kv_dim..][..kv_dim].copy_from_slice(&k);
        vc[pos * kv_dim..][..kv_dim].copy_from_slice(&v);

        // attention (1 query トークン・GQA)
        let (kcr, vcr) = cache.kv_bufs_ref(li);
        let t_total = pos + 1;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut att = vec![0.0f32; heads * dh];
        for hh in 0..heads {
            let kvh = hh / group;
            let qh = &q[hh * dh..(hh + 1) * dh];
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
            let avh = &mut att[hh * dh..(hh + 1) * dh];
            for j in 0..t_total {
                let vj = &vcr[j * kv_dim + kvh * dh..][..dh];
                let pv = scores[j];
                for (a, b) in avh.iter_mut().zip(vj.iter()) {
                    *a += pv * b;
                }
            }
        }

        // sigmoid 出力ゲート (head ごと [dh]) — gate は q 投影出力の後半
        for hh in 0..heads {
            let gate = &qg[hh * 2 * dh + dh..(hh + 1) * 2 * dh];
            for (d, av) in att[hh * dh..(hh + 1) * dh].iter_mut().enumerate() {
                *av /= 1.0 + (-gate[d]).exp(); // sigmoid(gate)
            }
        }

        att
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
    fn logits(&self, h: &[f32]) -> Vec<f32> {
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
            return self.forward_prefill(tokens, cache);
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
        for (i, &tok) in tokens.iter().enumerate() {
            x[i * e..(i + 1) * e].copy_from_slice(&self.embed[tok as usize * e..][..e]);
        }
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let dh = self.dh;
        let kv_dim = kv_heads * dh;
        let ff = self.ff;

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention: rmsnorm を行単位で一括計算 ---
            let mut h = vec![0.0f32; n * e];
            for i in 0..n {
                let r = rmsnorm(&x[i * e..(i + 1) * e], layer.attn_ln1(), e, self.eps);
                h[i * e..(i + 1) * e].copy_from_slice(&r);
            }
            match &layer.attn {
                AttnKind::Gdn(g) => {
                    // 投影はバッチ一括 (m=n → GPU tiled GEMM)
                    let (kqkv, _) = g.qkv_w.dims();
                    debug_assert_eq!(kqkv, e);
                    let qkv = g.qkv_w.apply_bt_rows(&h, n);
                    let z = g.gate_w.apply_bt_rows(&h, n);
                    let beta_raw = g.beta_w.apply_bt_rows(&h, n);
                    let alpha = g.alpha_w.apply_bt_rows(&h, n);
                    // スキャン: GPU カーネル 1 発 (P19-A1・cuda 有効時)。GDN は
                    // bit 一致不要のため GPU/CPU で数値は僅かに異なる (決定論は維持)。
                    // GPU 無効時はトークン逐次 (再帰状態) にフォールバック。
                    let dy = g.out_w.dims().0; // value_dim
                    #[cfg(feature = "cuda")]
                    let p = &self.gdn_p;
                    #[cfg(feature = "cuda")]
                    let ys: Vec<f32> = if n > 1 && p.head_k_dim == p.head_v_dim {
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
                            Some(y) => y,
                            None => self.gdn_scan_rows(g, &qkv, &z, &beta_raw, &alpha, cache, li, n, dy),
                        }
                    } else {
                        self.gdn_scan_rows(g, &qkv, &z, &beta_raw, &alpha, cache, li, n, dy)
                    };
                    #[cfg(not(feature = "cuda"))]
                    let ys: Vec<f32> =
                        self.gdn_scan_rows(g, &qkv, &z, &beta_raw, &alpha, cache, li, n, dy);
                    let out = g.out_w.apply_bt_rows(&ys, n);
                    for i in 0..n {
                        for (xi, ai) in x[i * e..(i + 1) * e]
                            .iter_mut()
                            .zip(&out[i * e..(i + 1) * e])
                        {
                            *xi += ai;
                        }
                    }
                }
                AttnKind::Full(fl) => {
                    let qg = fl.q_w.apply_bt_rows(&h, n); // [n, heads*2*dh]
                    let k = fl.k_w.apply_bt_rows(&h, n); // [n, kv_dim]
                    let v = fl.v_w.apply_bt_rows(&h, n); // [n, kv_dim]
                    let mut att = vec![0.0f32; n * heads * dh];
                    for i in 0..n {
                        let a = self.full_attn_core(
                            fl,
                            &qg[i * heads * 2 * dh..(i + 1) * heads * 2 * dh],
                            &k[i * kv_dim..(i + 1) * kv_dim],
                            &v[i * kv_dim..(i + 1) * kv_dim],
                            cache,
                            li,
                            pos0 + i,
                        );
                        att[i * heads * dh..(i + 1) * heads * dh].copy_from_slice(&a);
                    }
                    let out = fl.o_w.apply_bt_rows(&att, n);
                    for i in 0..n {
                        for (xi, ai) in x[i * e..(i + 1) * e]
                            .iter_mut()
                            .zip(&out[i * e..(i + 1) * e])
                        {
                            *xi += ai;
                        }
                    }
                }
            }
            // --- FFN: SwiGLU をバッチ一括 ---
            let mut h2 = vec![0.0f32; n * e];
            for i in 0..n {
                let r = rmsnorm(&x[i * e..(i + 1) * e], &layer.ln2_w, e, self.eps);
                h2[i * e..(i + 1) * e].copy_from_slice(&r);
            }
            let gate = layer.gate_w.apply_bt_rows(&h2, n);
            let up = layer.up_w.apply_bt_rows(&h2, n);
            let mut hh = vec![0.0f32; n * ff];
            for i in 0..n {
                for j in 0..ff {
                    let gv = gate[i * ff + j];
                    let uv = up[i * ff + j];
                    hh[i * ff + j] = gv / (1.0 + (-gv).exp()) * uv;
                }
            }
            let down = layer.down_w.apply_bt_rows(&hh, n);
            for i in 0..n {
                for (xi, di) in x[i * e..(i + 1) * e]
                    .iter_mut()
                    .zip(&down[i * e..(i + 1) * e])
                {
                    *xi += di;
                }
            }
        }

        // 最終 norm + logits は最終トークンのみ (トークン逐次経路の最終値と同一)
        let hf = rmsnorm(&x[(n - 1) * e..], &self.norm_w, e, self.eps);
        let logits = self.logits(&hf);
        cache.advance(n);
        Ok(logits)
    }
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
    fn write_gguf_f32(path: &PathBuf, tensors: &[(String, Vec<u64>, Vec<f32>)]) {
        const ALIGN: usize = 32;
        let mut b = Vec::new();
        wr_u32(&mut b, 0x4655_4747); // magic "GGUF"
        wr_u32(&mut b, 3); // version
        wr_u64(&mut b, tensors.len() as u64);
        wr_u64(&mut b, 1); // metadata kv count
        wr_str(&mut b, "general.architecture");
        wr_u32(&mut b, 8); // GT_STRING
        wr_str(&mut b, "qwen3_5");
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
        write_gguf_f32(&dir.join("model.gguf"), &ts);

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
}
