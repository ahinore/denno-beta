//! GPT-2 (124M 等)。最も単純な構成: LayerNorm + GELU + causal attention。
//! 重みは HF Conv1D 形式で [in, out] 保存 → forward は x @ W + b。
//!
//! 線形層の重みは `quant::Linear` で保持され、QuantMode::Int8 時はロード時に
//! per-channel scale の int8 へオフライン量子化される (P3)。int8 は転置した
//! [out,in] 形で保持するため bt カーネルで計算する。lm_head (tied: wte を転置
//! 使用) も int8 モードでは量子化コピーを持つ。

use anyhow::Result;
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::load::WeightStore;
use crate::model::{cfg_u64, Model};
use crate::quant::{Linear, QuantMode};
use crate::tensor::{add_bias, gelu_inplace, layernorm, softmax};

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_w: Linear,   // [e, 3e] (Conv1D)
    attn_b: Vec<f32>, // [3e]
    attn_proj_w: Linear, // [e, e]
    attn_proj_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc_w: Linear,     // [e, 4e]
    fc_b: Vec<f32>,
    mlp_proj_w: Linear, // [4e, e]
    mlp_proj_b: Vec<f32>,
}

pub struct Gpt2 {
    e: usize,          // n_embd
    heads: usize,      // n_head
    ctx: usize,        // n_ctx
    vocab: usize,
    eos: u32,
    wte: Vec<f32>,     // [vocab, e] (embedding lookup 用、常に fp32)
    wpe: Vec<f32>,     // [ctx, e]
    lnf_w: Vec<f32>,
    lnf_b: Vec<f32>,
    /// lm_head。None なら tied → wte を [vocab,e] のまま転置使用 (fp32)。
    /// int8 モードでは常に Some (wte の量子化コピー)。
    lm_head: Option<Linear>,
    layers: Vec<Layer>,
}

impl Gpt2 {
    pub fn load(dir: &Path, cfg: &serde_json::Value, quant: QuantMode) -> Result<Self> {
        let e = cfg_u64(cfg, "n_embd")? as usize;
        let heads = cfg_u64(cfg, "n_head")? as usize;
        let ctx = cfg_u64(cfg, "n_ctx")? as usize;
        let vocab = cfg_u64(cfg, "vocab_size")? as usize;
        let n_layer = cfg_u64(cfg, "n_layer")? as usize;
        let eos = cfg["eos_token_id"].as_u64().unwrap_or(50256) as u32;

        // 単一ファイル / 分割 checkpoint を透過的に開く
        let ws = WeightStore::open(dir)?;
        let g = |name: &str| ws.get_f32(name);
        let mut quant_bytes = 0usize;
        let mut lin_col = |w: &[f32], k: usize, n: usize| -> Linear {
            let l = Linear::from_colmajor(w, k, n, quant);
            quant_bytes += l.weight_bytes();
            l
        };

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("h.{i}.");
            layers.push(Layer {
                ln1_w: g(&format!("{p}ln_1.weight"))?,
                ln1_b: g(&format!("{p}ln_1.bias"))?,
                attn_w: lin_col(&g(&format!("{p}attn.c_attn.weight"))?, e, 3 * e),
                attn_b: g(&format!("{p}attn.c_attn.bias"))?,
                attn_proj_w: lin_col(&g(&format!("{p}attn.c_proj.weight"))?, e, e),
                attn_proj_b: g(&format!("{p}attn.c_proj.bias"))?,
                ln2_w: g(&format!("{p}ln_2.weight"))?,
                ln2_b: g(&format!("{p}ln_2.bias"))?,
                fc_w: lin_col(&g(&format!("{p}mlp.c_fc.weight"))?, e, 4 * e),
                fc_b: g(&format!("{p}mlp.c_fc.bias"))?,
                mlp_proj_w: lin_col(&g(&format!("{p}mlp.c_proj.weight"))?, 4 * e, e),
                mlp_proj_b: g(&format!("{p}mlp.c_proj.bias"))?,
            });
        }

        // lm_head: tied は wte を [vocab,e] (= nn.Linear 形) のまま転置使用。
        // int8 モードでは lm_head 用の量子化コピーを持つ (wte 本体は lookup 用に
        // fp32 のまま残す)。
        let lm_head = match quant {
            QuantMode::Fp32 => None,
            QuantMode::Int8 | QuantMode::Q4_0 => {
                let l = Linear::new(g("wte.weight")?, e, vocab, quant);
                quant_bytes += l.weight_bytes();
                Some(l)
            }
        };
        if quant == QuantMode::Int8 {
            tracing::info!(
                "int8 quantization: GPT-2 linear weights quantized ({:.0} MB, fp32 なら {:.0} MB)",
                quant_bytes as f64 / 1e6,
                (quant_bytes as f64 * 4.0) / 1e6
            );
        }

        Ok(Self {
            e,
            heads,
            ctx,
            vocab,
            eos,
            wte: g("wte.weight")?,
            wpe: g("wpe.weight")?,
            lnf_w: g("ln_f.weight")?,
            lnf_b: g("ln_f.bias")?,
            lm_head,
            layers,
        })
    }
}

impl Model for Gpt2 {
    fn context(&self) -> usize {
        self.ctx
    }

    fn eos_ids(&self) -> &[u32] {
        std::slice::from_ref(&self.eos)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(self.layers.len(), self.e, self.ctx)
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let e = self.e;
        let heads = self.heads;
        let dh = e / heads;
        let pos0 = cache.len();
        let n = tokens.len();

        // embedding: wte[t] + wpe[pos]
        let mut x = vec![0.0f32; n * e];
        for (i, &t) in tokens.iter().enumerate() {
            let emb = &self.wte[t as usize * e..][..e];
            let pe = &self.wpe[(pos0 + i) * e..][..e];
            for j in 0..e {
                x[i * e + j] = emb[j] + pe[j];
            }
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            let h = layernorm(&x, &layer.ln1_w, &layer.ln1_b, e, 1e-5);
            let mut qkv = layer.attn_w.apply(&h, n); // [n, 3e]
            add_bias(&mut qkv, &layer.attn_b, 3 * e);

            // K, V をキャッシュに書き込み
            {
                let (kc, vc) = cache.kv_bufs(li, n);
                for i in 0..n {
                    let src_k = &qkv[i * 3 * e + e..][..e];
                    let src_v = &qkv[i * 3 * e + 2 * e..][..e];
                    kc[(pos0 + i) * e..][..e].copy_from_slice(src_k);
                    vc[(pos0 + i) * e..][..e].copy_from_slice(src_v);
                }
            }

            // causal attention (query位置 i は cache の 0..pos0+i+1 を参照)
            let mut att = vec![0.0f32; n * e];
            {
                let (kc, vc) = cache.kv_bufs_ref(li);
                let mut scores = vec![0.0f32; pos0 + n];
                let scale = 1.0 / (dh as f32).sqrt();
                for i in 0..n {
                    let t_ctx = pos0 + i + 1;
                    for hd in 0..heads {
                        let off = hd * dh;
                        let qrow = &qkv[i * 3 * e + off..][..dh];
                        for j in 0..t_ctx {
                            let krow = &kc[j * e + off..][..dh];
                            let mut dot = 0.0f32;
                            for d in 0..dh {
                                dot += qrow[d] * krow[d];
                            }
                            scores[j] = dot * scale;
                        }
                        softmax(&mut scores[..t_ctx]);
                        let out = &mut att[i * e + off..][..dh];
                        for v_val in out.iter_mut() {
                            *v_val = 0.0;
                        }
                        for j in 0..t_ctx {
                            let w = scores[j];
                            let vrow = &vc[j * e + off..][..dh];
                            for d in 0..dh {
                                out[d] += w * vrow[d];
                            }
                        }
                    }
                }
            }

            let mut proj = layer.attn_proj_w.apply(&att, n);
            add_bias(&mut proj, &layer.attn_proj_b, e);
            for i in 0..n * e {
                x[i] += proj[i];
            }

            // --- mlp block ---
            let h2 = layernorm(&x, &layer.ln2_w, &layer.ln2_b, e, 1e-5);
            let mut f1 = layer.fc_w.apply(&h2, n);
            add_bias(&mut f1, &layer.fc_b, 4 * e);
            gelu_inplace(&mut f1);
            let mut f2 = layer.mlp_proj_w.apply(&f1, n);
            add_bias(&mut f2, &layer.mlp_proj_b, e);
            for i in 0..n * e {
                x[i] += f2[i];
            }
        }

        // 最終位置だけ LayerNorm → lm head
        // fp32: wte を [vocab,e] のまま転置使用。int8: 量子化コピーの bt カーネル。
        let last = &x[(n - 1) * e..];
        let hn = layernorm(last, &self.lnf_w, &self.lnf_b, e, 1e-5);
        let logits = match &self.lm_head {
            Some(w) => w.apply_bt(&hn, 1),
            None => crate::tensor::matvec_bt(&hn, &self.wte, e, self.vocab), // tied
        };

        cache.advance(n);
        Ok(logits)
    }
}
