//! Llama系アーキテクチャ(RMSNorm + RoPE + SwiGLU + GQA)。
//! 最初のターゲットは Qwen2.5-0.5B-Instruct (model_type = "qwen2")。
//! 重みは nn.Linear 形式で [out, in] 保存 → forward は x @ W^T。

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::load::get_f32;
use crate::model::{cfg_u64, Model};
use crate::tensor::{matmul_bt, rmsnorm, rope_rows, silu_inplace};

struct Layer {
    ln1_w: Vec<f32>,         // input_layernorm
    q_w: Vec<f32>,           // [H, e]
    k_w: Vec<f32>,           // [K, e]
    v_w: Vec<f32>,           // [K, e]
    o_w: Vec<f32>,           // [e, H]
    ln2_w: Vec<f32>,         // post_attention_layernorm
    gate_w: Vec<f32>,        // [ff, e]
    up_w: Vec<f32>,          // [ff, e]
    down_w: Vec<f32>,        // [e, ff]
}

pub struct LlamaQwen2 {
    e: usize,                // hidden_size
    heads: usize,            // num_attention_heads
    kv_heads: usize,         // num_key_value_heads
    dh: usize,               // head_dim
    ff: usize,               // intermediate_size
    ctx: usize,              // max_position_embeddings
    vocab: usize,
    eps: f32,
    theta: f32,              // rope_theta
    eos_ids: Vec<u32>,
    embed: Vec<f32>,         // [vocab, e]
    norm_w: Vec<f32>,        // model.norm
    lm_head: Option<Vec<f32>>, // None なら tied → embed を転置使用
    layers: Vec<Layer>,
}

impl LlamaQwen2 {
    pub fn load(dir: &Path, cfg: &serde_json::Value) -> Result<Self> {
        let e = cfg_u64(cfg, "hidden_size")? as usize;
        let heads = cfg_u64(cfg, "num_attention_heads")? as usize;
        let kv_heads = cfg["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
        let dh = cfg["head_dim"].as_u64().map(|v| v as usize).unwrap_or(e / heads);
        let ff = cfg_u64(cfg, "intermediate_size")? as usize;
        let ctx = cfg["max_position_embeddings"].as_u64().unwrap_or(32768) as usize;
        let vocab = cfg_u64(cfg, "vocab_size")? as usize;
        let eps = cfg["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32;
        let theta = cfg["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32;
        let tied = cfg["tie_word_embeddings"].as_bool().unwrap_or(false);

        // eos_token_id は単一 or 配列の両方があり得る
        let mut eos_ids = Vec::new();
        match &cfg["eos_token_id"] {
            serde_json::Value::Number(n) => eos_ids.push(n.as_u64().unwrap() as u32),
            serde_json::Value::Array(a) => {
                for v in a {
                    if let Some(n) = v.as_u64() {
                        eos_ids.push(n as u32);
                    }
                }
            }
            _ => eos_ids.push(151643), // Qwen系のデフォルト
        }

        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;
        let g = |name: &str| get_f32(&st, name);

        let mut layers = Vec::with_capacity(cfg_u64(cfg, "num_hidden_layers")? as usize);
        let n_layer = layers.capacity();
        for i in 0..n_layer {
            let p = format!("model.layers.{i}.");
            layers.push(Layer {
                ln1_w: g(&format!("{p}input_layernorm.weight"))?,
                q_w: g(&format!("{p}self_attn.q_proj.weight"))?,
                k_w: g(&format!("{p}self_attn.k_proj.weight"))?,
                v_w: g(&format!("{p}self_attn.v_proj.weight"))?,
                o_w: g(&format!("{p}self_attn.o_proj.weight"))?,
                ln2_w: g(&format!("{p}post_attention_layernorm.weight"))?,
                gate_w: g(&format!("{p}mlp.gate_proj.weight"))?,
                up_w: g(&format!("{p}mlp.up_proj.weight"))?,
                down_w: g(&format!("{p}mlp.down_proj.weight"))?,
            });
        }

        let lm_head = if tied {
            None
        } else {
            Some(g("lm_head.weight")?)
        };

        Ok(Self {
            e,
            heads,
            kv_heads,
            dh,
            ff,
            ctx,
            vocab,
            eps,
            theta,
            eos_ids,
            embed: g("model.embed_tokens.weight")?,
            norm_w: g("model.norm.weight")?,
            lm_head,
            layers,
        })
    }
}

impl Model for LlamaQwen2 {
    fn context(&self) -> usize {
        self.ctx
    }

    fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    fn new_cache(&self) -> KvCache {
        let kv_dim = self.kv_heads * self.dh;
        KvCache::new(self.layers.len(), kv_dim, self.ctx)
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let e = self.e;
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads; // GQA group size
        let pos0 = cache.len();
        let n = tokens.len();

        let mut x = vec![0.0f32; n * e];
        for (i, &t) in tokens.iter().enumerate() {
            let emb = &self.embed[t as usize * e..][..e];
            x[i * e..][..e].copy_from_slice(emb);
        }

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention ---
            let h = rmsnorm(&x, &layer.ln1_w, e, self.eps);
            let mut q = matmul_bt(&h, &layer.q_w, n, e, heads * dh); // [n, H*dh]
            let mut k = matmul_bt(&h, &layer.k_w, n, e, kv_dim); // [n, K*dh]
            let v = matmul_bt(&h, &layer.v_w, n, e, kv_dim);

            rope_rows(&mut q, n, heads, dh, pos0, self.theta);
            rope_rows(&mut k, n, kv_heads, dh, pos0, self.theta);

            {
                let (kc, vc) = cache.kv_bufs(li);
                for i in 0..n {
                    kc[(pos0 + i) * kv_dim..][..kv_dim].copy_from_slice(&k[i * kv_dim..][..kv_dim]);
                    vc[(pos0 + i) * kv_dim..][..kv_dim].copy_from_slice(&v[i * kv_dim..][..kv_dim]);
                }
            }

            let mut att = vec![0.0f32; n * heads * dh];
            {
                let (kc, vc) = cache.kv_bufs(li);
                let mut scores = vec![0.0f32; pos0 + n];
                let scale = 1.0 / (dh as f32).sqrt();
                for i in 0..n {
                    let t_ctx = pos0 + i + 1;
                    for qh in 0..heads {
                        let kvh = qh / group;
                        let off = qh * dh;
                        let kv_off = kvh * dh;
                        let qrow = &q[i * heads * dh + off..][..dh];
                        for j in 0..t_ctx {
                            let krow = &kc[j * kv_dim + kv_off..][..dh];
                            let mut dot = 0.0f32;
                            for d in 0..dh {
                                dot += qrow[d] * krow[d];
                            }
                            scores[j] = dot * scale;
                        }
                        crate::tensor::softmax(&mut scores[..t_ctx]);
                        let out = &mut att[i * heads * dh + off..][..dh];
                        for v_val in out.iter_mut() {
                            *v_val = 0.0;
                        }
                        for j in 0..t_ctx {
                            let w = scores[j];
                            let vrow = &vc[j * kv_dim + kv_off..][..dh];
                            for d in 0..dh {
                                out[d] += w * vrow[d];
                            }
                        }
                    }
                }
            }

            let proj = matmul_bt(&att, &layer.o_w, n, heads * dh, e);
            for i in 0..n * e {
                x[i] += proj[i];
            }

            // --- mlp (SwiGLU) ---
            let h2 = rmsnorm(&x, &layer.ln2_w, e, self.eps);
            let mut gate = matmul_bt(&h2, &layer.gate_w, n, e, self.ff);
            let up = matmul_bt(&h2, &layer.up_w, n, e, self.ff);
            silu_inplace(&mut gate);
            for i in 0..n * self.ff {
                gate[i] *= up[i];
            }
            let down = matmul_bt(&gate, &layer.down_w, n, self.ff, e);
            for i in 0..n * e {
                x[i] += down[i];
            }
        }

        let last = &x[(n - 1) * e..];
        let hn = rmsnorm(last, &self.norm_w, e, self.eps);
        let logits = match &self.lm_head {
            Some(w) => matmul_bt(&hn, w, 1, e, self.vocab),
            None => matmul_bt(&hn, &self.embed, 1, e, self.vocab), // tied
        };

        cache.advance(n);
        Ok(logits)
    }
}
