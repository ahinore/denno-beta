//! Llama系アーキテクチャ(RMSNorm + RoPE + SwiGLU + GQA)。
//! 最初のターゲットは Qwen2.5-0.5B-Instruct (model_type = "qwen2")。
//! 重みは nn.Linear 形式で [out, in] 保存 → forward は x @ W^T。

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::load::{get_f32, get_f32_opt};
use crate::model::{cfg_u64, Model};
use crate::tensor::{add_bias, matmul_bt, rmsnorm, rope_rows, silu_inplace};

struct Layer {
    ln1_w: Vec<f32>,         // input_layernorm
    q_w: Vec<f32>,           // [H, e]
    k_w: Vec<f32>,           // [K, e]
    v_w: Vec<f32>,           // [K, e]
    q_b: Option<Vec<f32>>,   // Qwen2.5-0.5B 等は QKV bias あり
    k_b: Option<Vec<f32>>,
    v_b: Option<Vec<f32>>,
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
        let go = |name: &str| get_f32_opt(&st, name);

        let mut layers = Vec::with_capacity(cfg_u64(cfg, "num_hidden_layers")? as usize);
        let n_layer = layers.capacity();
        for i in 0..n_layer {
            let p = format!("model.layers.{i}.");
            layers.push(Layer {
                ln1_w: g(&format!("{p}input_layernorm.weight"))?,
                q_w: g(&format!("{p}self_attn.q_proj.weight"))?,
                k_w: g(&format!("{p}self_attn.k_proj.weight"))?,
                v_w: g(&format!("{p}self_attn.v_proj.weight"))?,
                q_b: go(&format!("{p}self_attn.q_proj.bias"))?,
                k_b: go(&format!("{p}self_attn.k_proj.bias"))?,
                v_b: go(&format!("{p}self_attn.v_proj.bias"))?,
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
            let mut v = matmul_bt(&h, &layer.v_w, n, e, kv_dim);
            if let Some(b) = &layer.q_b {
                add_bias(&mut q, b, heads * dh);
            }
            if let Some(b) = &layer.k_b {
                add_bias(&mut k, b, kv_dim);
            }
            if let Some(b) = &layer.v_b {
                add_bias(&mut v, b, kv_dim);
            }

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
                let t_total = pos0 + n;
                let scale = 1.0 / (dh as f32).sqrt();
                // GQA: 同じ kv ヘッドを共有する group 個の q ヘッドをまとめて処理する。
                // scores[i,j] = q[i,gh] · k[j,kh] は sgemm で一括計算し(行列積)、
                // causal mask → softmax → att = probs @ v も sgemm で行う。
                let mut scores = vec![0.0f32; n * t_total];
                let mut kh = vec![0.0f32; t_total * dh];
                let mut vh = vec![0.0f32; t_total * dh];
                for kvh in 0..kv_heads {
                    // K/V ヘッド列の抽出 (contiguous 化)
                    for j in 0..t_total {
                        kh[j * dh..][..dh].copy_from_slice(&kc[j * kv_dim + kvh * dh..][..dh]);
                        vh[j * dh..][..dh].copy_from_slice(&vc[j * kv_dim + kvh * dh..][..dh]);
                    }
                    for g in 0..group {
                        let qh = kvh * group + g;
                        // Q ヘッド列の抽出 [n, dh]
                        let mut qg = vec![0.0f32; n * dh];
                        for i in 0..n {
                            qg[i * dh..][..dh].copy_from_slice(
                                &q[i * heads * dh + qh * dh..][..dh],
                            );
                        }
                        // scores[i,j] = qg[i] · kh[j] * scale  (kh は [t_total, dh] row-major → matmul_bt で a@b^T)
                        for s in scores.iter_mut() {
                            *s = 0.0;
                        }
                        crate::tensor::sgemm_acc(
                            &qg, &kh, &mut scores, n, dh, t_total, scale,
                        );
                        // causal mask (prefill 時、未来の位置を -inf に)
                        for i in 0..n {
                            let row = &mut scores[i * t_total..][..t_total];
                            for j in (pos0 + i + 1)..t_total {
                                row[j] = f32::NEG_INFINITY;
                            }
                            crate::tensor::softmax(row);
                        }
                        // att_head[i] = Σ_j probs[i,j] * vh[j]  (通常の行列積)
                        let out = crate::tensor::matmul(&scores, &vh, n, t_total, dh);
                        for i in 0..n {
                            att[i * heads * dh + qh * dh..][..dh]
                                .copy_from_slice(&out[i * dh..][..dh]);
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
