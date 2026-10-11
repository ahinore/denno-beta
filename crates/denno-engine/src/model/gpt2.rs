//! GPT-2 (124M 等)。最も単純な構成: LayerNorm + GELU + causal attention。
//! 重みは HF Conv1D 形式で [in, out] 保存 → forward は x @ W + b。

use anyhow::Result;
use memmap2::Mmap;
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::load::get_f32;
use crate::model::{cfg_u64, Model};
use crate::tensor::{add_bias, gelu_inplace, layernorm, matmul, matmul_bt, softmax};

struct Layer {
    ln1_w: Vec<f32>,
    ln1_b: Vec<f32>,
    attn_w: Vec<f32>,   // [e, 3e]
    attn_b: Vec<f32>,   // [3e]
    attn_proj_w: Vec<f32>, // [e, e]
    attn_proj_b: Vec<f32>,
    ln2_w: Vec<f32>,
    ln2_b: Vec<f32>,
    fc_w: Vec<f32>,     // [e, 4e]
    fc_b: Vec<f32>,
    mlp_proj_w: Vec<f32>, // [4e, e]
    mlp_proj_b: Vec<f32>,
}

pub struct Gpt2 {
    e: usize,          // n_embd
    heads: usize,      // n_head
    ctx: usize,        // n_ctx
    vocab: usize,
    eos: u32,
    wte: Vec<f32>,     // [vocab, e]
    wpe: Vec<f32>,     // [ctx, e]
    lnf_w: Vec<f32>,
    lnf_b: Vec<f32>,
    layers: Vec<Layer>,
}

impl Gpt2 {
    pub fn load(dir: &Path, cfg: &serde_json::Value) -> Result<Self> {
        let e = cfg_u64(cfg, "n_embd")? as usize;
        let heads = cfg_u64(cfg, "n_head")? as usize;
        let ctx = cfg_u64(cfg, "n_ctx")? as usize;
        let vocab = cfg_u64(cfg, "vocab_size")? as usize;
        let n_layer = cfg_u64(cfg, "n_layer")? as usize;
        let eos = cfg["eos_token_id"].as_u64().unwrap_or(50256) as u32;

        let file = std::fs::File::open(dir.join("model.safetensors"))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;

        let g = |name: &str| get_f32(&st, name);

        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = format!("h.{i}.");
            layers.push(Layer {
                ln1_w: g(&format!("{p}ln_1.weight"))?,
                ln1_b: g(&format!("{p}ln_1.bias"))?,
                attn_w: g(&format!("{p}attn.c_attn.weight"))?,
                attn_b: g(&format!("{p}attn.c_attn.bias"))?,
                attn_proj_w: g(&format!("{p}attn.c_proj.weight"))?,
                attn_proj_b: g(&format!("{p}attn.c_proj.bias"))?,
                ln2_w: g(&format!("{p}ln_2.weight"))?,
                ln2_b: g(&format!("{p}ln_2.bias"))?,
                fc_w: g(&format!("{p}mlp.c_fc.weight"))?,
                fc_b: g(&format!("{p}mlp.c_fc.bias"))?,
                mlp_proj_w: g(&format!("{p}mlp.c_proj.weight"))?,
                mlp_proj_b: g(&format!("{p}mlp.c_proj.bias"))?,
            });
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
            let mut qkv = matmul(&h, &layer.attn_w, n, e, 3 * e);
            add_bias(&mut qkv, &layer.attn_b, 3 * e);

            // K, V をキャッシュに書き込み
            {
                let (kc, vc) = cache.kv_bufs(li);
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
                let (kc, vc) = cache.kv_bufs(li);
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

            let mut proj = matmul(&att, &layer.attn_proj_w, n, e, e);
            add_bias(&mut proj, &layer.attn_proj_b, e);
            for i in 0..n * e {
                x[i] += proj[i];
            }

            // --- mlp block ---
            let h2 = layernorm(&x, &layer.ln2_w, &layer.ln2_b, e, 1e-5);
            let mut f1 = matmul(&h2, &layer.fc_w, n, e, 4 * e);
            add_bias(&mut f1, &layer.fc_b, 4 * e);
            gelu_inplace(&mut f1);
            let mut f2 = matmul(&f1, &layer.mlp_proj_w, n, 4 * e, e);
            add_bias(&mut f2, &layer.mlp_proj_b, e);
            for i in 0..n * e {
                x[i] += f2[i];
            }
        }

        // 最終位置だけ LayerNorm → lm head (wte を転置使用: tied embeddings)
        let last = &x[(n - 1) * e..];
        let hn = layernorm(last, &self.lnf_w, &self.lnf_b, e, 1e-5);
        let logits = matmul_bt(&hn, &self.wte, 1, e, self.vocab);

        cache.advance(n);
        Ok(logits)
    }
}
