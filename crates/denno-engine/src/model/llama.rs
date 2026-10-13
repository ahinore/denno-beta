//! Llama系アーキテクチャ(RMSNorm + RoPE + SwiGLU + GQA)。
//! 最初のターゲットは Qwen2.5-0.5B-Instruct (model_type = "qwen2")。
//! 重みは nn.Linear 形式で [out, in] 保存 → forward は x @ W^T。
//!
//! 線形層の重みは `quant::Linear` で保持され、QuantMode::Int8 時はロード時に
//! per-channel scale の int8 へオフライン量子化される (P3)。embedding は
//! lookup (gather) に必要なため常に fp32。tied lm_head (Qwen2.5-0.5B) の場合は
//! int8 モードで embedding とは別に lm_head 用の量子化コピーを作る
//! (decode 時の logits matvec も int8 帯域の対象にするため)。

use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::load::WeightStore;
use crate::model::{cfg_u64, Model};
use crate::quant::{Linear, QuantMode};
use crate::tensor::{add_bias, rmsnorm, rope_rows, silu_inplace};

struct Layer {
    ln1_w: Vec<f32>,         // input_layernorm
    q_w: Linear,             // [H, e]
    k_w: Linear,             // [K, e]
    v_w: Linear,             // [K, e]
    q_b: Option<Vec<f32>>,   // Qwen2.5-0.5B 等は QKV bias あり
    k_b: Option<Vec<f32>>,
    v_b: Option<Vec<f32>>,
    o_w: Linear,             // [e, H]
    ln2_w: Vec<f32>,         // post_attention_layernorm
    gate_w: Linear,          // [ff, e]
    up_w: Linear,            // [ff, e]
    down_w: Linear,          // [e, ff]
    q_norm_w: Option<Vec<f32>>, // Qwen3 QK-Norm: [dh] (q/k の投影直後・RoPE 前に適用)
    k_norm_w: Option<Vec<f32>>, // None (Qwen2.5 等) なら QK-Norm 無効
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
    embed: Vec<f32>,         // [vocab, e] (gather 用なので常に fp32)
    norm_w: Vec<f32>,        // model.norm
    lm_head: Option<Linear>, // None なら tied → embed を転置使用
    layers: Vec<Layer>,
    /// int8 モードで量子化された重みの合計バイト数 (ログ用)
    quant_bytes: usize,
}

impl LlamaQwen2 {
    pub fn load(dir: &Path, cfg: &serde_json::Value, quant: QuantMode) -> Result<Self> {
        let e = cfg_u64(cfg, "hidden_size")? as usize;
        let heads = cfg_u64(cfg, "num_attention_heads")? as usize;
        let kv_heads = cfg["num_key_value_heads"].as_u64().unwrap_or(heads as u64) as usize;
        let dh = cfg["head_dim"].as_u64().map(|v| v as usize).unwrap_or(e / heads);
        let ff = cfg_u64(cfg, "intermediate_size")? as usize;
        let ctx = cfg["max_position_embeddings"].as_u64().unwrap_or(32768) as usize;
        let vocab = cfg_u64(cfg, "vocab_size")? as usize;
        let kv_dim = kv_heads * dh;
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

        // 単一ファイル / 分割 checkpoint(index.json + shard 群)を透過的に開く。
        // untied lm_head (例: Qwen2.5-7B の lm_head.weight) は WeightStore が
        // どの shard にあっても解決する。
        let ws = WeightStore::open(dir)?;
        let g = |name: &str| ws.get_f32(name);
        let go = |name: &str| ws.get_f32_opt(name);

        let mut layers = Vec::with_capacity(cfg_u64(cfg, "num_hidden_layers")? as usize);
        let n_layer = layers.capacity();
        let mut quant_bytes = 0usize;
        let mut lin = |w: Vec<f32>, k: usize, n: usize| -> Linear {
            let l = Linear::new(w, k, n, quant);
            quant_bytes += l.weight_bytes();
            l
        };
        for i in 0..n_layer {
            let p = format!("model.layers.{i}.");
            layers.push(Layer {
                ln1_w: g(&format!("{p}input_layernorm.weight"))?,
                q_w: lin(g(&format!("{p}self_attn.q_proj.weight"))?, e, heads * dh),
                k_w: lin(g(&format!("{p}self_attn.k_proj.weight"))?, e, kv_dim),
                v_w: lin(g(&format!("{p}self_attn.v_proj.weight"))?, e, kv_dim),
                q_b: go(&format!("{p}self_attn.q_proj.bias"))?,
                k_b: go(&format!("{p}self_attn.k_proj.bias"))?,
                v_b: go(&format!("{p}self_attn.v_proj.bias"))?,
                o_w: lin(g(&format!("{p}self_attn.o_proj.weight"))?, heads * dh, e),
                ln2_w: g(&format!("{p}post_attention_layernorm.weight"))?,
                gate_w: lin(g(&format!("{p}mlp.gate_proj.weight"))?, e, ff),
                up_w: lin(g(&format!("{p}mlp.up_proj.weight"))?, e, ff),
                down_w: lin(g(&format!("{p}mlp.down_proj.weight"))?, ff, e),
                // Qwen3 のみ存在する QK-Norm 重み (Qwen2.5 では None)
                q_norm_w: go(&format!("{p}self_attn.q_norm.weight"))?,
                k_norm_w: go(&format!("{p}self_attn.k_norm.weight"))?,
            });
        }

        // lm_head: untied なら独立重み、tied なら embed を転置使用。
        // int8 モードでは tied の場合も lm_head 用の量子化コピーを持つ
        // (decode 毎の logits matvec を int8 帯域の対象にするため。embed 本体は
        // lookup 用に fp32 のまま残す)。
        let lm_head = if tied {
            match quant {
                QuantMode::Fp32 => None,
                QuantMode::Int8 => {
                    let l = Linear::new(g("model.embed_tokens.weight")?, e, vocab, quant);
                    quant_bytes += l.weight_bytes();
                    Some(l)
                }
            }
        } else {
            Some(lin(g("lm_head.weight")?, e, vocab))
        };

        if quant == QuantMode::Int8 {
            tracing::info!(
                "int8 quantization: {} linear weights quantized ({:.0} MB, fp32 なら {:.0} MB)",
                n_layer * 7 + if lm_head.is_some() { 1 } else { 0 },
                quant_bytes as f64 / 1e6,
                (quant_bytes as f64 * 4.0) / 1e6
            );
        }

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
            quant_bytes,
        })
    }

    /// int8 モードで量子化された重みの合計バイト数
    pub fn quantized_weight_bytes(&self) -> usize {
        self.quant_bytes
    }
}

/// nn.Linear 相当 (w: [out,in])。decode (m==1) は出力分割並列の
/// matvec に切り替える (重み約1GB/トークンを1スレッドで読むと律速になるため)。
/// int8 モードではデ量子化込みの fused matvec が使われる。
fn lin(a: &[f32], w: &Linear, m: usize) -> Vec<f32> {
    w.apply_bt(a, m)
}

// P12: decode attention のスクラッチバッファ (rayon ワーカースレッドごとに再利用)。
// t×dh のバッファは malloc の mmap 閾値 (~128KB) を超えるため、毎ステップ
// 毎レイヤーの確保/解放が mmap/munmap システムコールになる — スレッドローカル
// で再利用して排除する。内容は毎回全上書きするため数値・決定論には影響しない。
thread_local! {
    static KH_SCRATCH: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
    static SCORES_SCRATCH: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
}

/// decode 1 ステップの attention 本体 (1 セッション × 1 kv ヘッド) → [group*dh]。
/// `forward` (n==1) と `forward_batch` の双方からこの関数を通るため、
/// 単独 decode とバッチ decode はビット一致する。
///
/// P12 の変更点 (数値は従来経路とビット一致):
/// - probs@V: 従来の per-g `matmul(m=1)` × group 回 + vh 連続化コピー →
///   1 回の `matmul_rows_bstrided(m=group, v ストライド直接参照)`
///   (`decode_attention_probs_v_bitexact` テストでビット一致を担保)
/// - scores: 従来通り per-g `sgemm_acc(m=1, kh 連続化コピー)`
///   (行統合・ストライド参照は matrixmultiply がカーネルを選び替えて
///   ビット不一致になるため使わない — tensor.rs の検証メモ参照)
/// - qg のコピーを廃止 (q 行の当該区間は連続なのでそのまま渡す)
/// - kh/scores をスレッドローカルスクラッチで再利用 (mmap チャーン除去)
///
/// 数値的な注意: scores/probs/v の各要素の演算列は従来と完全に同一
/// (sgemm_acc(m=1) → softmax → matmul の同一形状呼び出し)。セッション/
/// kv ヘッド間に数値的な相互作用はない → 単独/バッチ・変更前/変更後でビット一致。
#[allow(clippy::too_many_arguments)]
fn attention_decode_kvhead(
    q_row: &[f32],  // セッション 1 行分 [heads*dh]
    kc: &[f32],     // レイヤ全体 K バッファ (行ストライド kv_dim)
    vc: &[f32],     // レイヤ全体 V バッファ (行ストライド kv_dim)
    kvh: usize,
    group: usize,
    dh: usize,
    kv_dim: usize,
    t_total: usize,
    scale: f32,
) -> Vec<f32> {
    let kh_off = kvh * dh;
    KH_SCRATCH.with(|kh_cell| {
        SCORES_SCRATCH.with(|sc_cell| {
            let mut kh = kh_cell.borrow_mut();
            kh.clear();
            kh.resize(t_total * dh, 0.0);
            for j in 0..t_total {
                kh[j * dh..][..dh].copy_from_slice(&kc[j * kv_dim + kh_off..][..dh]);
            }
            let mut scores = sc_cell.borrow_mut();
            scores.clear();
            scores.resize(group * t_total, 0.0);
            for g in 0..group {
                let qg = &q_row[(kvh * group + g) * dh..][..dh];
                let row = &mut scores[g * t_total..][..t_total];
                // scores[j] = qg · kh[j] * scale (従来の n==1 sgemm 経路と同一)。
                // causal mask: decode では未来位置なし (マスク範囲が空) → softmax のみ
                crate::tensor::sgemm_acc(qg, &kh, row, 1, dh, t_total, scale);
                crate::tensor::softmax(row);
            }
            // att_head[g] = Σ_j probs[g,j] * vh[j] — 行統合 + v ストライド直接参照
            crate::tensor::matmul_rows_bstrided(
                &scores, &vc[kh_off..], kv_dim, group, t_total, dh,
            )
        })
    })
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
        crate::prof::time(crate::prof::EMBED, || {
            for (i, &t) in tokens.iter().enumerate() {
                let emb = &self.embed[t as usize * e..][..e];
                x[i * e..][..e].copy_from_slice(emb);
            }
        });

        for (li, layer) in self.layers.iter().enumerate() {
            crate::telemetry_layer_begin(li as u16);
            // --- attention ---
            let h = crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                rmsnorm(&x, &layer.ln1_w, e, self.eps)
            });
            let mut q = crate::prof::time_layer(crate::prof::QKV, li, || lin(&h, &layer.q_w, n)); // [n, H*dh]
            let mut k = crate::prof::time_layer(crate::prof::QKV, li, || lin(&h, &layer.k_w, n)); // [n, K*dh]
            let mut v = crate::prof::time_layer(crate::prof::QKV, li, || lin(&h, &layer.v_w, n));
            if let Some(b) = &layer.q_b {
                add_bias(&mut q, b, heads * dh);
            }
            if let Some(b) = &layer.k_b {
                add_bias(&mut k, b, kv_dim);
            }
            if let Some(b) = &layer.v_b {
                add_bias(&mut v, b, kv_dim);
            }
            // Qwen3 QK-Norm: q/k を head_dim 単位で RMSNorm する (RoPE の前)。
            // 行/ヘッド単位で独立 → バッチ/単独でビット一致
            if layer.q_norm_w.is_some() || layer.k_norm_w.is_some() {
                crate::prof::time_layer(crate::prof::QKNORM, li, || {
                    if let Some(w) = &layer.q_norm_w {
                        crate::tensor::qknorm_rows_inplace(&mut q, n, heads, dh, w, self.eps);
                    }
                    if let Some(w) = &layer.k_norm_w {
                        crate::tensor::qknorm_rows_inplace(&mut k, n, kv_heads, dh, w, self.eps);
                    }
                });
            }

            crate::prof::time_layer(crate::prof::ROPE, li, || {
                rope_rows(&mut q, n, heads, dh, pos0, self.theta);
                rope_rows(&mut k, n, kv_heads, dh, pos0, self.theta);
            });

            {
                let (kc, vc) = cache.kv_bufs(li);
                for i in 0..n {
                    kc[(pos0 + i) * kv_dim..][..kv_dim].copy_from_slice(&k[i * kv_dim..][..kv_dim]);
                    vc[(pos0 + i) * kv_dim..][..kv_dim].copy_from_slice(&v[i * kv_dim..][..kv_dim]);
                }
            }

            let mut att = vec![0.0f32; n * heads * dh];
            crate::prof::time_layer(crate::prof::ATTN, li, || {
                let (kc, vc) = cache.kv_bufs(li);
                let t_total = pos0 + n;
                let scale = 1.0 / (dh as f32).sqrt();
                if n == 1 {
                    // decode: P12 統一経路 (forward_batch と同一関数
                    // `attention_decode_kvhead` → 単独/バッチ decode でビット一致)
                    for kvh in 0..kv_heads {
                        let a =
                            attention_decode_kvhead(&q, kc, vc, kvh, group, dh, kv_dim, t_total, scale);
                        att[kvh * group * dh..][..group * dh].copy_from_slice(&a);
                    }
                    return;
                }
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
            });

            let proj = crate::prof::time_layer(crate::prof::OPROJ, li, || lin(&att, &layer.o_w, n));
            for i in 0..n * e {
                x[i] += proj[i];
            }

            // --- mlp (SwiGLU) ---
            let h2 = crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                rmsnorm(&x, &layer.ln2_w, e, self.eps)
            });
            let mut gate =
                crate::prof::time_layer(crate::prof::GATE_UP, li, || lin(&h2, &layer.gate_w, n));
            let up = crate::prof::time_layer(crate::prof::GATE_UP, li, || lin(&h2, &layer.up_w, n));
            crate::prof::time_layer(crate::prof::SWIGLU, li, || {
                silu_inplace(&mut gate);
                for i in 0..n * self.ff {
                    gate[i] *= up[i];
                }
            });
            let down = crate::prof::time_layer(crate::prof::DOWN, li, || lin(&gate, &layer.down_w, n));
            for i in 0..n * e {
                x[i] += down[i];
            }
            crate::telemetry_layer_flush();
        }

        let last = &x[(n - 1) * e..];
        let hn = crate::prof::time(crate::prof::FINAL_NORM, || {
            rmsnorm(last, &self.norm_w, e, self.eps)
        });
        let logits = crate::prof::time(crate::prof::LM_HEAD, || match &self.lm_head {
            Some(w) => lin(&hn, w, 1),
            // tied (fp32 モード): embed を [vocab, e] のまま転置使用
            None => crate::tensor::matvec_bt(&hn, &self.embed, e, self.vocab),
        });

        cache.advance(n);
        Ok(logits)
    }

    /// P6: 複数セッションの decode 1 ステップをまとめて実行する (各セッション
    /// 1 トークン)。線形層 (qkv/o_proj/gate/up/down/lm_head) は全セッション分の
    /// 行を 1 回の重みストリーミングで計算するビット一致バッチカーネル
    /// (`apply_batch_bt`) を使い、attention は各セッションの KV キャッシュに
    /// 対して `forward` の decode (n==1) 経路と同一の演算列を実行する。
    /// → 各セッションの logits は単独 decode とビット一致する。
    fn forward_batch(
        &mut self,
        toks: &[u32],
        caches: &mut [&mut KvCache],
    ) -> Result<Vec<Vec<f32>>> {
        debug_assert_eq!(toks.len(), caches.len());
        let b = toks.len();
        let e = self.e;
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads; // GQA group size

        let lin_b = |a: &[f32], w: &Linear, m: usize| w.apply_batch_bt(a, m);

        // --- embed (セッションごとに lookup、行間の相互作用なし) ---
        let mut x = vec![0.0f32; b * e];
        crate::prof::time(crate::prof::EMBED, || {
            for (i, &t) in toks.iter().enumerate() {
                let emb = &self.embed[t as usize * e..][..e];
                x[i * e..][..e].copy_from_slice(emb);
            }
        });

        for (li, layer) in self.layers.iter().enumerate() {
            crate::telemetry_layer_begin(li as u16);
            // --- attention ---
            // rmsnorm は行独立なので [b, e] にまとめて適用できる (行ごとに同一演算)
            let h = crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                rmsnorm(&x, &layer.ln1_w, e, self.eps)
            });
            let mut q =
                crate::prof::time_layer(crate::prof::QKV, li, || lin_b(&h, &layer.q_w, b)); // [b, H*dh]
            let mut k =
                crate::prof::time_layer(crate::prof::QKV, li, || lin_b(&h, &layer.k_w, b)); // [b, K*dh]
            let mut v = crate::prof::time_layer(crate::prof::QKV, li, || lin_b(&h, &layer.v_w, b));
            if let Some(bias) = &layer.q_b {
                add_bias(&mut q, bias, heads * dh);
            }
            if let Some(bias) = &layer.k_b {
                add_bias(&mut k, bias, kv_dim);
            }
            if let Some(bias) = &layer.v_b {
                add_bias(&mut v, bias, kv_dim);
            }
            // Qwen3 QK-Norm: q/k を head_dim 単位で RMSNorm する (RoPE の前)。
            // forward() と同一の行/ヘッド単位演算 → 単独 decode とビット一致
            if layer.q_norm_w.is_some() || layer.k_norm_w.is_some() {
                crate::prof::time_layer(crate::prof::QKNORM, li, || {
                    if let Some(w) = &layer.q_norm_w {
                        crate::tensor::qknorm_rows_inplace(&mut q, b, heads, dh, w, self.eps);
                    }
                    if let Some(w) = &layer.k_norm_w {
                        crate::tensor::qknorm_rows_inplace(&mut k, b, kv_heads, dh, w, self.eps);
                    }
                });
            }

            // RoPE: セッションごとに位置が異なる (cache.len()) ので 1 行ずつ適用。
            // rope_row は rope_rows の 1 行処理と同一演算列 → ビット一致。
            crate::prof::time_layer(crate::prof::ROPE, li, || {
                for i in 0..b {
                    let pos = caches[i].len();
                    crate::tensor::rope_row(
                        &mut q[i * heads * dh..][..heads * dh],
                        heads,
                        dh,
                        pos,
                        self.theta,
                    );
                    crate::tensor::rope_row(
                        &mut k[i * kv_dim..][..kv_dim],
                        kv_heads,
                        dh,
                        pos,
                        self.theta,
                    );
                }
            });

            // KV 書き込み: 各セッションの自分のキャッシュへ (decode 1 トークン分)
            for i in 0..b {
                let pos = caches[i].len();
                let (kc, vc) = caches[i].kv_bufs(li);
                kc[pos * kv_dim..][..kv_dim].copy_from_slice(&k[i * kv_dim..][..kv_dim]);
                vc[pos * kv_dim..][..kv_dim].copy_from_slice(&v[i * kv_dim..][..kv_dim]);
            }

            // attention: セッション×KVヘッドを rayon で並列実行 (P10)。
            // 各タスクは (セッション i, kv ヘッド kvh) を担当し、att への書き込み
            // 先は非重複スライス。タスク内は `attention_decode_kvhead` に統一され
            // forward() の decode (n==1) 経路と同一関数・同一演算列 → ビット一致。
            // (P12: probs@V は行統合+ストライド直接参照、スクラッチ再利用 —
            //  数値は従来の per-(kvh,g) m=1 経路とビット一致、テスト担保)
            let mut att = vec![0.0f32; b * heads * dh];
            crate::prof::time_layer(crate::prof::ATTN, li, || {
                let scale = 1.0 / (dh as f32).sqrt();
                // 読み取り専用借用に切り替えて各セッションのキャッシュを共有
                let cache_refs: Vec<&KvCache> = caches.iter().map(|c| &**c).collect();
                let parts: Vec<(usize, usize, Vec<f32>)> = (0..b)
                    .into_par_iter()
                    .flat_map_iter(|i| (0..kv_heads).map(move |kvh| (i, kvh)))
                    .map(|(i, kvh)| {
                        let t_total = cache_refs[i].len() + 1; // pos + 1
                        let (kc, vc) = cache_refs[i].kv_bufs_ref(li);
                        let a = attention_decode_kvhead(
                            &q[i * heads * dh..][..heads * dh],
                            kc,
                            vc,
                            kvh,
                            group,
                            dh,
                            kv_dim,
                            t_total,
                            scale,
                        );
                        (i, kvh, a)
                    })
                    .collect();
                for (i, kvh, a) in parts {
                    let o = i * heads * dh + kvh * group * dh;
                    att[o..][..group * dh].copy_from_slice(&a);
                }
            });

            let proj = crate::prof::time_layer(crate::prof::OPROJ, li, || lin_b(&att, &layer.o_w, b));
            for i in 0..b * e {
                x[i] += proj[i];
            }

            // --- mlp (SwiGLU) ---
            let h2 = crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                rmsnorm(&x, &layer.ln2_w, e, self.eps)
            });
            let mut gate = crate::prof::time_layer(crate::prof::GATE_UP, li, || {
                lin_b(&h2, &layer.gate_w, b)
            });
            let up =
                crate::prof::time_layer(crate::prof::GATE_UP, li, || lin_b(&h2, &layer.up_w, b));
            crate::prof::time_layer(crate::prof::SWIGLU, li, || {
                silu_inplace(&mut gate);
                for i in 0..b * self.ff {
                    gate[i] *= up[i];
                }
            });
            let down =
                crate::prof::time_layer(crate::prof::DOWN, li, || lin_b(&gate, &layer.down_w, b));
            for i in 0..b * e {
                x[i] += down[i];
            }
            crate::telemetry_layer_flush();
        }

        // --- final norm (行独立) + lm_head (バッチ) ---
        let mut hn = vec![0.0f32; b * e];
        crate::prof::time(crate::prof::FINAL_NORM, || {
            for i in 0..b {
                let r = rmsnorm(&x[i * e..][..e], &self.norm_w, e, self.eps);
                hn[i * e..][..e].copy_from_slice(&r);
            }
        });
        let logits = crate::prof::time(crate::prof::LM_HEAD, || match &self.lm_head {
            Some(w) => lin_b(&hn, w, b),
            // tied (fp32 モード): embed を [vocab, e] のまま転置使用
            None => crate::tensor::matvec_rows_bt(&hn, &self.embed, e, self.vocab, b),
        });

        for c in caches.iter_mut() {
            c.advance(1);
        }
        Ok((0..b).map(|i| logits[i * self.vocab..][..self.vocab].to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::QuantMode;

    /// 決定論的な擬似乱数ベクトル (tensor.rs の pseudo_random と同一系列)
    fn prng(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 977.0 - 1.0
            })
            .collect()
    }

    /// テスト用の極小 Qwen2 (2層, GQA, QKV bias あり) を直接構築する。
    /// Fp32 は tied (lm_head なし) / Int8 は untied 相当の独立 lm_head。
    /// `with_qk` = true で Qwen3 相当の QK-Norm 重みを持たせる。
    fn tiny(mode: QuantMode) -> LlamaQwen2 {
        tiny_opt(mode, false)
    }

    fn tiny_opt(mode: QuantMode, with_qk: bool) -> LlamaQwen2 {
        let e = 16usize;
        let heads = 4;
        let kv_heads = 2;
        let dh = 4;
        let ff = 32;
        let vocab = 64;
        let ctx = 64;
        let mk = |w: Vec<f32>, k: usize, n: usize| Linear::new(w, k, n, mode);
        let mut layers = Vec::new();
        for i in 0..2u32 {
            layers.push(Layer {
                ln1_w: prng(e, 100 + i),
                q_w: mk(prng(heads * dh * e, 200 + i), e, heads * dh),
                k_w: mk(prng(kv_heads * dh * e, 300 + i), e, kv_heads * dh),
                v_w: mk(prng(kv_heads * dh * e, 400 + i), e, kv_heads * dh),
                q_b: Some(prng(heads * dh, 500 + i)),
                k_b: Some(prng(kv_heads * dh, 600 + i)),
                v_b: Some(prng(kv_heads * dh, 700 + i)),
                o_w: mk(prng(e * heads * dh, 800 + i), heads * dh, e),
                ln2_w: prng(e, 900 + i),
                gate_w: mk(prng(ff * e, 1000 + i), e, ff),
                up_w: mk(prng(ff * e, 1100 + i), e, ff),
                down_w: mk(prng(e * ff, 1200 + i), ff, e),
                q_norm_w: if with_qk { Some(prng(dh, 1300 + i)) } else { None },
                k_norm_w: if with_qk { Some(prng(dh, 1400 + i)) } else { None },
            });
        }
        let lm_head = match mode {
            QuantMode::Fp32 => None,
            QuantMode::Int8 => Some(mk(prng(vocab * e, 9999), e, vocab)),
        };
        LlamaQwen2 {
            e,
            heads,
            kv_heads,
            dh,
            ff,
            ctx,
            vocab,
            eps: 1e-6,
            theta: 10_000.0,
            eos_ids: vec![63],
            embed: prng(vocab * e, 7),
            norm_w: prng(e, 8),
            lm_head,
            layers,
            quant_bytes: 0,
        }
    }

    /// greedy argmax (テスト用。tie は両経路で同一 logits なので同一インデックス)
    fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        best as u32
    }

    #[test]
    fn forward_batch_matches_solo_decode_bit_exactly() {
        // P6 の中核検証: 複数セッションをバッチ decode しても、各セッションの
        // logits は単独 decode とビット一致する (fp32 / int8 両モード)。
        for mode in [QuantMode::Fp32, QuantMode::Int8] {
            let prompts: Vec<Vec<u32>> = vec![
                vec![5, 11, 23],
                vec![1, 2, 3, 4, 5],
                vec![7, 9],
            ];
            let steps = 4;

            // --- 単独リファレンス: 各セッションを別々に decode ---
            let mut solo = tiny(mode);
            let mut solo_caches: Vec<KvCache> =
                (0..prompts.len()).map(|_| solo.new_cache()).collect();
            let mut solo_seq: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
            // prefill (各セッション単独)
            let mut solo_logits: Vec<Vec<f32>> = Vec::new();
            for (s, p) in prompts.iter().enumerate() {
                let lg = solo.forward(p, &mut solo_caches[s]).unwrap();
                solo_logits.push(lg);
            }
            // step ごとの decode 後 logits を保存 (バッチ側と段階比較するため)
            let mut solo_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); steps];
            for step in 0..steps {
                for s in 0..prompts.len() {
                    let t = argmax(&solo_logits[s]);
                    solo_seq[s].push(t);
                    solo_logits[s] = solo.forward(&[t], &mut solo_caches[s]).unwrap();
                    solo_hist[step].push(solo_logits[s].clone());
                }
            }

            // --- バッチ: 全セッションを 1 forward にまとめて decode ---
            // (エンジンと同じ流れ: prefill logits から 1 トークン目をサンプルし、
            //  そのトークンを全セッション分まとめて forward する)
            let mut bat = tiny(mode);
            let mut bat_caches: Vec<KvCache> =
                (0..prompts.len()).map(|_| bat.new_cache()).collect();
            let mut bat_seq: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
            let mut bat_logits: Vec<Vec<f32>> = Vec::new();
            for (s, p) in prompts.iter().enumerate() {
                let lg = bat.forward(p, &mut bat_caches[s]).unwrap();
                bat_logits.push(lg);
            }
            for step in 0..steps {
                let toks: Vec<u32> = (0..prompts.len())
                    .map(|s| argmax(&bat_logits[s]))
                    .collect();
                let mut caches: Vec<&mut KvCache> = bat_caches.iter_mut().collect();
                let outs = bat.forward_batch(&toks, &mut caches).unwrap();
                for s in 0..prompts.len() {
                    assert_eq!(
                        outs[s], solo_hist[step][s],
                        "{mode:?} step={step} session={s}: バッチ logits が単独と不一致"
                    );
                    bat_seq[s].push(toks[s]);
                    bat_logits[s] = outs[s].clone();
                }
            }
            for s in 0..prompts.len() {
                assert_eq!(bat_seq[s], solo_seq[s], "{mode:?} session={s}: 生成列が不一致");
            }
        }
    }

    #[test]
    fn forward_batch_single_session_matches_solo() {
        // バッチサイズ 1 でも経路が単独 decode とビット一致すること
        let mut solo = tiny(QuantMode::Fp32);
        let mut c1 = solo.new_cache();
        let lg1 = solo.forward(&[3, 14, 15], &mut c1).unwrap();

        let mut bat = tiny(QuantMode::Fp32);
        let mut c2 = bat.new_cache();
        let lg2 = bat.forward(&[3, 14, 15], &mut c2).unwrap();
        assert_eq!(lg1, lg2);
        let t = argmax(&lg1);
        let out = bat.forward_batch(&[t], &mut [&mut c2]).unwrap();
        let ref_lg = solo.forward(&[t], &mut c1).unwrap();
        assert_eq!(out[0], ref_lg);
    }

    #[test]
    fn qk_norm_batch_matches_solo_decode_bit_exactly() {
        // Qwen3 (QK-Norm あり) でもバッチ decode が単独 decode とビット一致する。
        // また QK-Norm が適用されていること (無効化した場合と出力が変わる) を確認。
        for mode in [QuantMode::Fp32, QuantMode::Int8] {
            let prompts: Vec<Vec<u32>> = vec![vec![5, 11, 23], vec![1, 2, 3, 4, 5], vec![7, 9]];

            // --- 単独リファレンス ---
            let mut solo = tiny_opt(mode, true);
            let mut solo_caches: Vec<KvCache> =
                (0..prompts.len()).map(|_| solo.new_cache()).collect();
            let mut solo_prefill: Vec<Vec<f32>> = Vec::new();
            for (s, p) in prompts.iter().enumerate() {
                let lg = solo.forward(p, &mut solo_caches[s]).unwrap();
                solo_prefill.push(lg);
            }
            let mut solo_decode: Vec<Vec<f32>> = Vec::new();
            for s in 0..prompts.len() {
                let t = argmax(&solo_prefill[s]);
                solo_decode.push(solo.forward(&[t], &mut solo_caches[s]).unwrap());
            }

            // --- バッチ decode ---
            let mut bat = tiny_opt(mode, true);
            let mut bat_caches: Vec<KvCache> =
                (0..prompts.len()).map(|_| bat.new_cache()).collect();
            let mut bat_logits: Vec<Vec<f32>> = Vec::new();
            for (s, p) in prompts.iter().enumerate() {
                let lg = bat.forward(p, &mut bat_caches[s]).unwrap();
                assert_eq!(lg, solo_prefill[s], "{mode:?} prefill s={s}");
                bat_logits.push(lg);
            }
            let toks: Vec<u32> = (0..prompts.len()).map(|s| argmax(&bat_logits[s])).collect();
            let mut caches: Vec<&mut KvCache> = bat_caches.iter_mut().collect();
            let outs = bat.forward_batch(&toks, &mut caches).unwrap();
            for s in 0..prompts.len() {
                assert_eq!(outs[s], solo_decode[s], "{mode:?} decode s={s}");
            }

            // --- QK-Norm が効いていることの確認 (無し版と出力が異なる) ---
            let mut noqk = tiny_opt(mode, false);
            let mut c = noqk.new_cache();
            let lg_noqk = noqk.forward(&[5, 11, 23], &mut c).unwrap();
            assert_ne!(lg_noqk, solo_prefill[0], "QK-Norm が適用されていない");
        }
    }

    #[test]
    fn debug_batch_layer0() {
        // デバッグ用: レイヤー0の各段階でバッチと単独を比較する
        let mode = QuantMode::Fp32;
        let prompts: Vec<Vec<u32>> = vec![vec![5, 11, 23], vec![1, 2, 3, 4, 5], vec![7, 9]];
        let mut solo = tiny(mode);
        let mut solo_caches: Vec<KvCache> =
            (0..prompts.len()).map(|_| solo.new_cache()).collect();
        let mut solo_logits: Vec<Vec<f32>> = Vec::new();
        for (s, p) in prompts.iter().enumerate() {
            let lg = solo.forward(p, &mut solo_caches[s]).unwrap();
            solo_logits.push(lg);
        }

        let mut bat = tiny(mode);
        let mut bat_caches: Vec<KvCache> =
            (0..prompts.len()).map(|_| bat.new_cache()).collect();
        let mut bat_logits: Vec<Vec<f32>> = Vec::new();
        for (s, p) in prompts.iter().enumerate() {
            let lg = bat.forward(p, &mut bat_caches[s]).unwrap();
            assert_eq!(lg, solo_logits[s], "prefill s{s}");
            bat_logits.push(lg);
        }

        for step in 0..3 {
            let toks: Vec<u32> = (0..prompts.len()).map(|s| argmax(&bat_logits[s])).collect();
            let mut caches: Vec<&mut KvCache> = bat_caches.iter_mut().collect();
            let outs = bat.forward_batch(&toks, &mut caches).unwrap();
            for s in 0..prompts.len() {
                let r = solo.forward(&[toks[s]], &mut solo_caches[s]).unwrap();
                let diff: Vec<usize> = outs[s]
                    .iter()
                    .zip(&r)
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, _)| i)
                    .collect();
                println!("step={step} s={s} toks[s]={} diff={}", toks[s], diff.len());
                assert!(diff.is_empty(), "step={step} s={s}: {diff:?}");
                solo_logits[s] = r;
                bat_logits[s] = outs[s].clone();
            }
        }
    }
}
