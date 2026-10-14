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
        let tied = tc["tie_word_embeddings"].as_bool().unwrap_or(true);
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
