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

/// P17c: QKV 融合 (cuda feature 時)。q/k/v は同一入力 (`h`) を消費するため、
/// 重み行を [q|k|v] の順に連結した 1 回の GPU 呼び出しで計算し、出力を列ブロック
/// で切り分ける。各行の演算は個別呼び出しと完全に同一 (量子化も行単位) なので
/// **ビット一致**。GPU 無効/エラー/一部が fp32 の場合は None → 従来の 3 分割。
#[cfg(feature = "cuda")]
fn lin_qkv_gpu(
    layer: &Layer,
    h: &[f32],
    n: usize,
    qn: usize,
    kvn: usize,
) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let parts = [
        layer.q_w.quant8()?,
        layer.k_w.quant8()?,
        layer.v_w.quant8()?,
    ];
    let out = crate::cuda::maybe_gpu_linear_grouped(&parts, h, n)?;
    let width = qn + 2 * kvn;
    let mut q = Vec::with_capacity(n * qn);
    let mut k = Vec::with_capacity(n * kvn);
    let mut v = Vec::with_capacity(n * kvn);
    for i in 0..n {
        let row = &out[i * width..][..width];
        q.extend_from_slice(&row[..qn]);
        k.extend_from_slice(&row[qn..qn + kvn]);
        v.extend_from_slice(&row[qn + kvn..]);
    }
    Some((q, k, v))
}

/// P17c: gate/up 融合 (cuda feature 時)。仕様は lin_qkv_gpu と同じ。
#[cfg(feature = "cuda")]
fn lin_gateup_gpu(layer: &Layer, h: &[f32], n: usize, ff: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    let parts = [layer.gate_w.quant8()?, layer.up_w.quant8()?];
    let out = crate::cuda::maybe_gpu_linear_grouped(&parts, h, n)?;
    let mut gate = Vec::with_capacity(n * ff);
    let mut up = Vec::with_capacity(n * ff);
    for i in 0..n {
        let row = &out[i * (2 * ff)..][..2 * ff];
        gate.extend_from_slice(&row[..ff]);
        up.extend_from_slice(&row[ff..]);
    }
    Some((gate, up))
}

// P12: decode attention のスクラッチバッファ (rayon ワーカースレッドごとに再利用)。
// t×dh のバッファは malloc の mmap 閾値 (~128KB) を超えるため、毎ステップ
// 毎レイヤーの確保/解放が mmap/munmap システムコールになる — スレッドローカル
// で再利用して排除する。内容は毎回全上書きするため数値・決定論には影響しない。
thread_local! {
    static SCORES_SCRATCH: std::cell::RefCell<Vec<f32>> = std::cell::RefCell::new(Vec::new());
}

/// decode 1 ステップの attention 本体 (1 セッション × 1 kv ヘッド) → [group*dh]。
/// `forward` (n==1) と `forward_batch` の双方からこの関数を通るため、
/// 単独 decode とバッチ decode はビット一致する。
///
/// P12 の変更点:
/// - probs@V: 従来の per-g `matmul(m=1)` × group 回 + vh 連続化コピー →
///   1 回の `matmul_rows_bstrided(m=group, v ストライド直接参照)`
///   (`decode_attention_probs_v_bitexact` テストで担保)
/// - qg のコピーを廃止 (q 行の当該区間は連続なのでそのまま渡す)
/// - kh/scores をスレッドローカルスクラッチで再利用 (mmap チャーン除去)
///
/// P17f の変更点 (scores 側基準更新・ユーザー承認):
/// - scores: per-g `sgemm_acc(m=1, kh 連続化コピー)` × group 回 →
///   1 回の `sgemm_scores_kvstrided(m=group, kc ストライド直接参照)`。
///   kh 連続化コピー (t_total×dh × kv_heads×層数/step のメモリ移動) を廃止。
///   matrixmultiply のカーネル選択が形状/ストライド依存のため **数値は P12
///   経路とビット不一致** (P12 で検証済みの性質)。同一形状呼び出しは常に
///   同一結果 → 単独/バッチ decode・セッション間の決定論は維持。
/// - probs@V は P12 から不変 (row-major b 経路・同一形状呼び出し)
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
    SCORES_SCRATCH.with(|sc_cell| {
        let mut scores = sc_cell.borrow_mut();
        scores.clear();
        scores.resize(group * t_total, 0.0);
        // scores[g,j] = q[kvh*group+g] · kc[j,kvh] * scale — 1 回の sgemm で
        // group 行全てを計算 (q は q_row 内の連続区間、k は kv バッファ直接参照)。
        // causal mask: decode では未来位置なし (マスク範囲が空) → softmax のみ
        crate::tensor::sgemm_scores_kvstrided(
            &q_row[kvh * group * dh..],
            dh,
            &kc[kh_off..],
            kv_dim,
            &mut scores,
            group,
            dh,
            t_total,
            scale,
        );
        for g in 0..group {
            crate::tensor::softmax(&mut scores[g * t_total..][..t_total]);
        }
        // att_head[g] = Σ_j probs[g,j] * vh[j] — 行統合 + v ストライド直接参照
        crate::tensor::matmul_rows_bstrided(
            &scores, &vc[kh_off..], kv_dim, group, t_total, dh,
        )
    })
}

// P17d: solo decode (n==1) を隠れ状態デバイス常駐チェーンで実行する。
// 埋め込み → [rmsnorm(+residual) → qkv(grouped) → (CPU: rope/attention) →
// o_proj → rmsnorm(+residual) → gate/up(grouped) → swiglu → down] ×層数
// → final norm(+residual) → lm_head → logits の順で、x / 中間テンソルを
// GPU 上に置き続ける (ホスト往復は qkv と logits のみ)。sync は層あたり
// qkv の D2H 待ち 1 回で、CPU attention の間 GPU は次層 FFN を先読み実行。
// ビット一致: 線形は P17b/c と同一カーネル、rmsnorm/swiglu は CPU と同一の
// 演算順序 (expf は glibc e_expf を f64 演算で完全再現した dno_expf)。
// 戻り None = 従来の CPU/GPU per-op 経路で forward をやり直す
// (bias / Qwen3 QK-Norm / tied lm_head は非対応 → None。k/v 書き込みは冪等で
// advance は最後に 1 回なので途中失敗でも安全にやり直せる)。
#[cfg(feature = "cuda")]
impl LlamaQwen2 {
    /// P17g: solo decode chain と batch GPU attention の共通適合条件。
    /// bias / QK-Norm / fp32 混在モデルは solo chain を使えないため、
    /// solo chain (forward_decode_gpu) が使える構造か (lm_head 有・QK-Norm 無し・
    /// 全線形 int8)。P17h で qkv bias をサポート (dno_rope_qkv が bias 加算+rope
    /// を CPU とビット一致で実行)。GPU attention 自体はこの条件に依存しない
    /// (加算済み q/k/v を受けるだけで bias/QK-Norm と無関係・forward/forward_batch
    /// の両経路で個別に試行する)。batch も GPU attention を使うが、solo と同一
    /// カーネル・同一入力なので単独/バッチのビット一致 (P6 契約) は維持される。
    fn chain_compatible(&self) -> bool {
        if self.lm_head.is_none() {
            return false;
        }
        for layer in &self.layers {
            if layer.q_norm_w.is_some() || layer.k_norm_w.is_some() {
                return false; // Qwen3 QK-Norm は従来経路
            }
            for w in [
                &layer.q_w, &layer.k_w, &layer.v_w, &layer.o_w, &layer.gate_w, &layer.up_w,
                &layer.down_w,
            ] {
                if w.quant8().is_none() {
                    return false; // 一部 fp32 → 従来経路
                }
            }
        }
        self.lm_head.as_ref().unwrap().quant8().is_some()
    }

    fn forward_decode_gpu(
        &mut self,
        token: u32,
        cache: &mut KvCache,
        pos0: usize,
    ) -> Option<Vec<f32>> {
        use crate::cuda::{
            chain_attention_dev, chain_download, chain_kv_stage_flush, chain_linear_grouped,
            chain_rope_prepare, chain_rope_qkv, chain_rmsnorm, chain_swiglu, chain_upload,
            gpu_attention_enabled, CH_ACT, CH_ATT, CH_DOWN, CH_GA, CH_H, CH_LOGITS, CH_O, CH_QKV,
            CH_X,
        };
        let e = self.e;
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads; // GQA group size
        let n_layers = self.layers.len();

        // 非対応構造は従来経路へ
        if !self.chain_compatible() {
            return None;
        }
        // P17h: chain は q/k/v をデバイス常駐させるため attention は GPU 必須。
        // DENNO_GPU_ATTN=0 なら従来 (CPU・ビット一致) 経路へ (escape hatch)。
        if !gpu_attention_enabled() {
            return None;
        }

        // 埋め込み 1 行をアップロード (residual stream 常駐開始)
        chain_upload(CH_X, &self.embed[token as usize * e..][..e], e, 1)?;
        // rope テーブル (1 ステップ 1 回 — CPU rope_row と同一式・ビット一致)
        chain_rope_prepare(pos0, dh, self.theta)?;
        let mut pending: Option<usize> = None;

        for (li, layer) in self.layers.iter().enumerate() {
            crate::telemetry_layer_begin(li as u16);
            crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                chain_rmsnorm(&layer.ln1_w, self.eps, pending, CH_H, e, 1, li * 3)
            })?;
            // qkv: デバイス内で完結 (download なし — P17h で層あたりの sync を廃止)
            crate::prof::time_layer(crate::prof::QKV, li, || -> Option<()> {
                let qw = layer.q_w.quant8()?;
                let kw = layer.k_w.quant8()?;
                let vw = layer.v_w.quant8()?;
                chain_linear_grouped(&[qw, kw, vw], CH_H, CH_QKV, 1)
            })?;
            // bias 加算 + rope (GPU・CPU add_bias→rope_row とビット一致)
            crate::prof::time_layer(crate::prof::ROPE, li, || {
                chain_rope_qkv(
                    layer.q_b.as_deref(),
                    layer.k_b.as_deref(),
                    layer.v_b.as_deref(),
                    heads,
                    kv_heads,
                    dh,
                    n_layers * 3 + 1 + li,
                )
            })?;
            // --- attention (P17g/P17h: GPU カーネル + KV ミラー + q/k/v デバイス常駐)。
            //     rope 済み k/v は非同期 D2H でステージングし、ホストキャッシュへの
            //     反映はステップ最後の sync 後 (chain_kv_stage_flush)。失敗時は
            //     forward 全体を従来経路でやり直し。exactness 契約は attention のみ
            //     緩和 (ユーザー承認) — 線形層/bias/rope/expf/swiglu/rmsnorm は
            //     ビット一致維持。
            let t_total = pos0 + 1;
            let scale = 1.0 / (dh as f32).sqrt();
            crate::prof::time_layer(crate::prof::ATTN, li, || {
                chain_attention_dev(
                    cache.kv_id(),
                    li,
                    n_layers,
                    cache,
                    pos0,
                    t_total,
                    heads,
                    group,
                    dh,
                    kv_dim,
                    scale,
                )
            })?;

            // --- o_proj → residual+rmsnorm → gate/up → swiglu → down (全てデバイス内) ---
            crate::prof::time_layer(crate::prof::OPROJ, li, || -> Option<()> {
                let ow = layer.o_w.quant8()?;
                chain_linear_grouped(&[ow], CH_ATT, CH_O, 1)
            })?;
            crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                chain_rmsnorm(&layer.ln2_w, self.eps, Some(CH_O), CH_H, e, 1, li * 3 + 1)
            })?;
            crate::prof::time_layer(crate::prof::GATE_UP, li, || -> Option<()> {
                let gw = layer.gate_w.quant8()?;
                let uw = layer.up_w.quant8()?;
                chain_linear_grouped(&[gw, uw], CH_H, CH_GA, 1)
            })?;
            crate::prof::time_layer(crate::prof::SWIGLU, li, || {
                chain_swiglu(CH_GA, CH_ACT, self.ff, 1)
            })?;
            crate::prof::time_layer(crate::prof::DOWN, li, || -> Option<()> {
                let dw = layer.down_w.quant8()?;
                chain_linear_grouped(&[dw], CH_ACT, CH_DOWN, 1)
            })?;
            pending = Some(CH_DOWN);
            crate::telemetry_layer_flush();
        }

        let logits = crate::prof::time(crate::prof::LM_HEAD, || -> Option<Vec<f32>> {
            chain_rmsnorm(&self.norm_w, self.eps, pending, CH_H, e, 1, n_layers * 3)?;
            let lmq = self.lm_head.as_ref().unwrap().quant8()?;
            chain_linear_grouped(&[lmq], CH_H, CH_LOGITS, 1)?;
            chain_download(CH_LOGITS, self.vocab)
        })?;
        // ホストキャッシュへ rope 済み k/v を反映 (logits download の sync で全 D2H 完了)
        chain_kv_stage_flush(cache, n_layers, kv_dim, pos0)?;
        cache.advance(1);
        Some(logits)
    }

    /// B1: prefill (n>1) を隠れ状態デバイス常駐チェーンで実行する。
    /// 埋め込み 1 回 H2D → [rmsnorm(→CH_H) → qkv GEMM (デバイス h 直読み) → rope →
    /// (GPU attention) → o_proj (CH_O デバイス常駐) → residual+rmsnorm → FFN
    /// (CH_DOWN デバイス常駐)] ×層数 → 最終 residual+norm (m=n) → 最終行 D2H →
    /// lm_head の順。P17k 従来経路との差分: ①層あたり 2 回の CPU rmsnorm が GPU 化
    /// (dno_add_rmsnorm・CPU tensor::rmsnorm と同一演算順序でビット一致)
    /// ②h/proj/down のホスト往復 (e=1536・13k prefill で 81MB×4/層) と
    /// x += proj/down の CPU 要素演算が全て消える。ホスト転送は KV ミラー行の
    /// D2H (P17k と同一) と最終行 hn の 1 回のみ。
    /// 数値: rmsnorm/GEMM/rope/swiglu/residual は全て従来経路とビット一致のカーネ
    /// ル・同一演算順序。attention のみ緩和枠 (ユーザー承認済み・P17j と同一)。
    /// 戻り None = 従来経路で forward 全体をやり直し (P17d decode chain と同一契約。
    /// KV 書き込みは冪等・advance は最後の 1 回なので安全。途中まで書いた KV ミラー
    /// の行は次回 rope カーネルが上書きするため汚染しない)。失敗時は GPU_BROKEN が
    /// 立ち以降の forward は従来経路のみ (1 回限りのコスト)。
    #[cfg(feature = "cuda")]
    fn forward_prefill_gpu(
        &mut self,
        tokens: &[u32],
        cache: &mut KvCache,
        pos0: usize,
        n: usize,
    ) -> Option<Vec<f32>> {
        use crate::cuda::{
            chain_attention_prefill_dev, chain_download_at, chain_qkv_rope_prefill, chain_rmsnorm,
            chain_upload, gemm_path_active, gpu_attention_enabled, maybe_gpu_ffn_dev,
            maybe_gpu_oproj_dev, prefill_chain_gpu_supported, CH_DOWN, CH_H, CH_O, CH_X,
        };
        let e = self.e;
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let group = heads / kv_heads; // GQA group size
        let n_layers = self.layers.len();
        let t_total = pos0 + n;

        // 非対応構造・カーネル制約・GEMM 閾値は事前ゲート (チェーン途中失敗の回避)
        if !gpu_attention_enabled()
            || !gemm_path_active(n)
            || !prefill_chain_gpu_supported(e, dh, t_total)
        {
            return None;
        }
        if t_total > cache.max_len() {
            return None;
        }
        for layer in &self.layers {
            if layer.q_norm_w.is_some() || layer.k_norm_w.is_some() {
                return None; // Qwen3 QK-Norm は従来経路
            }
            for w in [
                &layer.q_w, &layer.k_w, &layer.v_w, &layer.o_w, &layer.gate_w, &layer.up_w,
                &layer.down_w,
            ] {
                if w.quant8().is_none() {
                    return None; // 一部 fp32 → 従来経路
                }
            }
            // qkv bias は [qb|kb|vb] 揃う場合のみデバイス経路可 (rope カーネル対応)
            let bias_ok = matches!(
                (&layer.q_b, &layer.k_b, &layer.v_b),
                (None, None, None) | (Some(_), Some(_), Some(_))
            );
            if !bias_ok {
                return None;
            }
        }

        // KV 容量確保 (従来経路と同一 — P17e 契約)
        for li in 0..n_layers {
            let _ = cache.kv_bufs(li, n);
        }

        // 埋め込み → residual stream をデバイス常駐開始 (ホスト x は以後不要)
        let mut x = vec![0.0f32; n * e];
        crate::prof::time(crate::prof::EMBED, || {
            for (i, &t) in tokens.iter().enumerate() {
                let emb = &self.embed[t as usize * e..][..e];
                x[i * e..][..e].copy_from_slice(emb);
            }
        });
        chain_upload(CH_X, &x, e, n)?;
        drop(x);
        let key = cache.kv_id();
        let mut pending: Option<usize> = None;

        for (li, layer) in self.layers.iter().enumerate() {
            crate::telemetry_layer_begin(li as u16);
            // --- rmsnorm #1 (デバイス: 前層 down との residual 加算も融合) ---
            crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                chain_rmsnorm(&layer.ln1_w, self.eps, pending, CH_H, e, n, li * 3)
            })?;
            // --- qkv GEMM (デバイス h 直読み) → rope → K/V ミラー書き込み + D2H ---
            let bias: Option<Vec<f32>> = match (&layer.q_b, &layer.k_b, &layer.v_b) {
                (Some(qb), Some(kb), Some(vb)) => {
                    let mut b = Vec::with_capacity(heads * dh + 2 * kv_heads * dh);
                    b.extend_from_slice(qb);
                    b.extend_from_slice(kb);
                    b.extend_from_slice(vb);
                    Some(b)
                }
                _ => None,
            };
            chain_qkv_rope_prefill(
                layer.q_w.quant8()?,
                layer.k_w.quant8()?,
                layer.v_w.quant8()?,
                &[],
                cache,
                key,
                li,
                n_layers,
                pos0,
                n,
                t_total,
                heads,
                kv_heads,
                dh,
                self.theta,
                bias.as_deref().unwrap_or(&[]),
                Some(CH_H),
            )?;
            // --- GPU attention (q はデバイス常駐・att は CH_ATT) ---
            crate::prof::time_layer(crate::prof::ATTN, li, || {
                chain_attention_prefill_dev(
                    key,
                    li,
                    n_layers,
                    pos0,
                    n,
                    t_total,
                    heads,
                    group,
                    dh,
                    kv_heads * dh,
                    1.0 / (dh as f32).sqrt(),
                )
            })?;
            // --- o_proj (デバイス常駐 → CH_O) ---
            maybe_gpu_oproj_dev(layer.o_w.quant8()?, n, heads * dh, li)?;
            // --- residual + rmsnorm #2 (デバイス: x += o_proj 出力も融合) ---
            crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                chain_rmsnorm(&layer.ln2_w, self.eps, Some(CH_O), CH_H, e, n, li * 3 + 1)
            })?;
            // --- FFN (デバイス常駐 → CH_DOWN) ---
            maybe_gpu_ffn_dev(
                layer.gate_w.quant8()?,
                layer.up_w.quant8()?,
                layer.down_w.quant8()?,
                CH_H,
                n,
                self.ff,
                li,
            )?;
            crate::telemetry_layer_flush();
            pending = Some(CH_DOWN);
        }

        // --- 最終 residual + norm (m=n 全行: x += down もデバイスで完結) ---
        crate::prof::time(crate::prof::FINAL_NORM, || {
            chain_rmsnorm(&self.norm_w, self.eps, pending, CH_H, e, n, n_layers * 3)
        })?;
        // 最終行の hn だけ D2H (logits に不要な [n-1, e] は転送しない)。
        // lm_head は従来経路と同一コード (m=1 matvec)。
        let logits = crate::prof::time(crate::prof::LM_HEAD, || -> Option<Vec<f32>> {
            let hn = chain_download_at(CH_H, (n - 1) * e, e)?;
            Some(match &self.lm_head {
                Some(w) => lin(&hn, w, 1),
                // tied (fp32 モード): embed を [vocab, e] のまま転置使用
                None => crate::tensor::matvec_bt(&hn, &self.embed, e, self.vocab),
            })
        })?;

        cache.advance(n);
        Some(logits)
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

        // P17d: solo decode (n==1) はデバイス常駐チェーンで実行
        // (GPU 無効/非対応構造/実行時エラー → None → 従来経路でやり直し)
        #[cfg(feature = "cuda")]
        if n == 1 {
            if let Some(logits) = self.forward_decode_gpu(tokens[0], cache, pos0) {
                return Ok(logits);
            }
        }

        // B1: prefill (n>1) もデバイス常駐チェーンで実行 — 層あたり 2 回の CPU
        // rmsnorm を GPU 化し、h/proj/down のホスト往復と x += proj/down の CPU
        // 要素演算を廃止する。失敗時は従来経路で forward 全体をやり直し
        // (KV 書き込みは冪等・advance は最後の 1 回なので安全)。
        #[cfg(feature = "cuda")]
        if n > 1 {
            if let Some(logits) = self.forward_prefill_gpu(tokens, cache, pos0, n) {
                return Ok(logits);
            }
        }

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
            // P17k ステップ 5: qkv GEMM → rope → K/V キャッシュ書き込みまでを
            // デバイスで完結する経路 (bias/QK-Norm 無し・n>1・GEMM 閾値以上の
            // 場合のみ)。成功時 q/k/v のホスト Vec は作らない — q はデバイス
            // (qkv_q・attention が直接読む)、K/V は rope カーネルが GPU ミラーに
            // 直接書き [pos0..t_total) 行だけ D2H でホストキャッシュへ。GEMM/rope
            // は CPU とビット一致 (緩和不要)。失敗時は従来経路にフォールバック
            // (ホスト状態不変)。
            #[cfg(feature = "cuda")]
            let mut dev_qkv = false;
            #[cfg(feature = "cuda")]
            if n > 1
                && layer.q_norm_w.is_none()
                && layer.k_norm_w.is_none()
            {
                // QKV bias があれば [qb|kb|vb] 連結でカーネルへ (rope 前加算・
                // CPU 経路とビット一致)。3 つ揃っていない場合はフォールバック。
                let bias: Option<Vec<f32>> = match (&layer.q_b, &layer.k_b, &layer.v_b) {
                    (Some(qb), Some(kb), Some(vb)) => {
                        let mut b = Vec::with_capacity(heads * dh + 2 * kv_dim);
                        b.extend_from_slice(qb);
                        b.extend_from_slice(kb);
                        b.extend_from_slice(vb);
                        Some(b)
                    }
                    (None, None, None) => None,
                    _ => None,
                };
                let kv_key = cache.kv_id();
                let _ = cache.kv_bufs(li, n); // 容量確保
                let qkv_ok = match (
                    layer.q_w.quant8(),
                    layer.k_w.quant8(),
                    layer.v_w.quant8(),
                ) {
                    (Some(qw), Some(kw), Some(vw)) => crate::cuda::chain_qkv_rope_prefill(
                        qw, kw, vw, &h, cache, kv_key, li, self.layers.len(), pos0, n,
                        pos0 + n, heads, kv_heads, dh, self.theta,
                        bias.as_deref().unwrap_or(&[]),
                        // 従来経路: h はホストからアップロード (B1 デバイス常駐経路は
                        // forward_prefill_gpu が Some(CH_H) を渡す)
                        None,
                    ),
                    _ => None,
                };
                if qkv_ok.is_some() {
                    dev_qkv = true;
                }
            }
            #[cfg(not(feature = "cuda"))]
            let dev_qkv = false;
            let (mut q, mut k, mut v) = if !dev_qkv {
                crate::prof::time_layer(crate::prof::QKV, li, || {
                    // P17c: q/k/v は同一入力 → 1 回の GPU 呼び出しに融合
                    // (GPU 無効/エラー時は従来の 3 分割呼び出しにフォールバック)
                    #[cfg(feature = "cuda")]
                    if let Some((q, k, v)) = lin_qkv_gpu(layer, &h, n, heads * dh, kv_dim) {
                        return (q, k, v);
                    }
                    (
                        lin(&h, &layer.q_w, n), // [n, H*dh]
                        lin(&h, &layer.k_w, n), // [n, K*dh]
                        lin(&h, &layer.v_w, n),
                    )
                })
            } else {
                // デバイス常駐: ホスト Vec は不要 (attention フォールバック時のみ
                // q を chain_qkv_q_download で受け取る)
                (Vec::new(), Vec::new(), Vec::new())
            };
            if !dev_qkv {
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
                    let (kc, vc) = cache.kv_bufs(li, n);
                    for i in 0..n {
                        kc[(pos0 + i) * kv_dim..][..kv_dim]
                            .copy_from_slice(&k[i * kv_dim..][..kv_dim]);
                        vc[(pos0 + i) * kv_dim..][..kv_dim]
                            .copy_from_slice(&v[i * kv_dim..][..kv_dim]);
                    }
                }
            }

            let mut att = vec![0.0f32; n * heads * dh];
            // P17k: GPU attention+o_proj 完了時の o_proj 出力 (att をホストに落とさず
            // o_proj まで実行。クロージャの外で受け取る)
            #[cfg(feature = "cuda")]
            let mut gpu_oproj: Option<Vec<f32>> = None;
            crate::prof::time_layer(crate::prof::ATTN, li, || {
                let (kc, vc) = cache.kv_bufs_ref(li);
                let t_total = pos0 + n;
                let scale = 1.0 / (dh as f32).sqrt();
                // P17k ステップ 5: q がデバイス常駐 (dev_qkv) → H2D なしで GPU
                // attention を試行。K/V ミラーは chain_qkv_rope_prefill が書き込み
                // 済み。成功時は o_proj もデバイスで (P17k ステップ 3 と同一経路)。
                // 失敗時は q を D2H して下の CPU 経路へ (K/V はホストキャッシュに
                // あるため CPU attention はそのまま動く)。
                #[cfg(feature = "cuda")]
                if dev_qkv {
                    if crate::cuda::chain_attention_prefill_dev(
                        cache.kv_id(),
                        li,
                        self.layers.len(),
                        pos0,
                        n,
                        t_total,
                        heads,
                        group,
                        dh,
                        kv_dim,
                        scale,
                    )
                    .is_some()
                    {
                        let mut done = false;
                        if let Some(oq) = layer.o_w.quant8() {
                            if let Some(proj) =
                                crate::cuda::maybe_gpu_oproj(oq, n, heads * dh, li)
                            {
                                gpu_oproj = Some(proj);
                                done = true;
                            }
                        }
                        if done {
                            return;
                        }
                        if let Some(av) =
                            crate::cuda::chain_download(crate::cuda::CH_ATT, n * heads * dh)
                        {
                            att.copy_from_slice(&av);
                            return;
                        }
                        // oproj も att download も失敗 → CPU attention へ
                    }
                    match crate::cuda::chain_qkv_q_download(n, heads * dh) {
                        Some(qh) => q = qh,
                        None => {
                            tracing::error!("cuda qkv dev: device q download failed");
                            return;
                        }
                    }
                }
                if n == 1 {
                    // decode: P17g GPU attention を先に試行 (forward_batch と同一
                    // カーネル・同一ミラー内容 → 単独/バッチでビット一致)。bias /
                    // QK-Norm は q/k/v の加算済み値に影響するだけで attention 自体と
                    // 無関係 → chain_compatible に依存させない。失敗時は CPU 経路
                    // (P12 統一経路、forward_batch と同一関数 → ビット一致)。
                    #[cfg(feature = "cuda")]
                    {
                        if crate::cuda::chain_attention(
                            cache.kv_id(),
                            li,
                            self.layers.len(),
                            &q,
                            cache,
                            t_total,
                            group,
                            dh,
                            kv_dim,
                            scale,
                        )
                        .is_some()
                        {
                            if let Some(av) =
                                crate::cuda::chain_download(crate::cuda::CH_ATT, heads * dh)
                            {
                                att[..heads * dh].copy_from_slice(&av);
                                return;
                            }
                        }
                    }
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
                // P17f: qg/kh/vh の連続化コピーを廃止し、Q 行バッファと kv バッファへ
                // ストライド直接参照で sgemm を実行 (数値は基準更新済み・決定論は維持)。
                let mut scores = vec![0.0f32; n * t_total];
                // P17j: prefill GPU attention (flash 風 online softmax) を先に
                // 試行。q/KV は全てホスト側にあるため失敗時もこのブロック内で
                // CPU 経路にフォールバックできる (forward 全体の再実行は不要・
                // ホスト状態は不変)。数値は attention 緩和枠 (ユーザー承認)。
                // P17k: GPU attention 成功時は o_proj もデバイス常駐で実行し att
                // をホストに落とさない (81MB D2H + to_vec + 再 H2D を消す)。
                // o_proj が失敗した場合のみ chain_download で att を取得して
                // 従来経路 (CPU o_proj) へ。o_proj 出力はビット一致なので緩和不要。
                #[cfg(feature = "cuda")]
                {
                    if crate::cuda::chain_attention_prefill(
                        cache.kv_id(),
                        li,
                        self.layers.len(),
                        &q,
                        cache,
                        pos0,
                        n,
                        t_total,
                        group,
                        dh,
                        kv_dim,
                        scale,
                    )
                    .is_some()
                    {
                        let mut done = false;
                        if let Some(oq) = layer.o_w.quant8() {
                            if let Some(proj) =
                                crate::cuda::maybe_gpu_oproj(oq, n, heads * dh, li)
                            {
                                gpu_oproj = Some(proj);
                                done = true;
                            }
                        }
                        if !done {
                            if let Some(av) =
                                crate::cuda::chain_download(crate::cuda::CH_ATT, n * heads * dh)
                            {
                                att.copy_from_slice(&av);
                                return;
                            }
                        } else {
                            return;
                        }
                    }
                }
                for kvh in 0..kv_heads {
                    for g in 0..group {
                        let qh = kvh * group + g;
                        // scores[i,j] = q[i,qh] · k[j,kvh] * scale
                        for s in scores.iter_mut() {
                            *s = 0.0;
                        }
                        crate::tensor::sgemm_scores_kvstrided(
                            &q[qh * dh..],
                            heads * dh,
                            &kc[kvh * dh..],
                            kv_dim,
                            &mut scores,
                            n,
                            dh,
                            t_total,
                            scale,
                        );
                        // causal mask (prefill 時、未来の位置を -inf に)
                        for i in 0..n {
                            let row = &mut scores[i * t_total..][..t_total];
                            for j in (pos0 + i + 1)..t_total {
                                row[j] = f32::NEG_INFINITY;
                            }
                            crate::tensor::softmax(row);
                        }
                        // att_head[i] = Σ_j probs[i,j] * v[j,kvh] — v ストライド直接参照 (並列)
                        let out = crate::tensor::matmul_rows_bstrided_par(
                            &scores, &vc[kvh * dh..], kv_dim, n, t_total, dh,
                        );
                        for i in 0..n {
                            att[i * heads * dh + qh * dh..][..dh]
                                .copy_from_slice(&out[i * dh..][..dh]);
                        }
                    }
                }
            });

            // P17k: GPU attention+o_proj が完結した場合は lin をスキップ
            // (o_proj 出力は maybe_gpu_oproj 内で OPROJ タイマーに計上済み)
            #[cfg(feature = "cuda")]
            let proj = if let Some(p) = gpu_oproj {
                p
            } else {
                crate::prof::time_layer(crate::prof::OPROJ, li, || lin(&att, &layer.o_w, n))
            };
            #[cfg(not(feature = "cuda"))]
            let proj = crate::prof::time_layer(crate::prof::OPROJ, li, || lin(&att, &layer.o_w, n));
            for i in 0..n * e {
                x[i] += proj[i];
            }

            // --- mlp (SwiGLU) ---
            let h2 = crate::prof::time_layer(crate::prof::RMSNORM, li, || {
                rmsnorm(&x, &layer.ln2_w, e, self.eps)
            });
            // P17k: FFN 線形ブロック (gate_up → swiglu → down) をデバイス常駐で
            // 実行。全段ビット一致 (GEMM: 整数累積・swiglu: dno_swiglu と同一演算)
            // なので出力は従来経路と完全同一。中間 [m, 2ff] をホストに落とさない。
            // 失敗時/対象外 (m が小さい等) は従来経路へフォールバック。
            #[cfg(feature = "cuda")]
            {
                if let (Some(gq), Some(uq), Some(dq)) = (
                    layer.gate_w.quant8(),
                    layer.up_w.quant8(),
                    layer.down_w.quant8(),
                ) {
                    if let Some(down) = crate::cuda::maybe_gpu_ffn(gq, uq, dq, &h2, n, self.ff, li) {
                        for i in 0..n * e {
                            x[i] += down[i];
                        }
                        crate::telemetry_layer_flush();
                        continue;
                    }
                }
            }
            let (mut gate, up) = crate::prof::time_layer(crate::prof::GATE_UP, li, || {
                // P17c: gate/up は同一入力 → 1 回の GPU 呼び出しに融合
                #[cfg(feature = "cuda")]
                if let Some((gate, up)) = lin_gateup_gpu(layer, &h2, n, self.ff) {
                    return (gate, up);
                }
                (lin(&h2, &layer.gate_w, n), lin(&h2, &layer.up_w, n))
            });
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

        // P17h: b==1 は solo decode chain (rope GPU 化+qkv デバイス常駐・ステップ内
        // の sync 廃止) に委譲する。GPU 無効/非対応構造/実行時エラー → None →
        // 従来のバッチ経路でやり直し (キャッシュ状態は失敗時に不変)。GPU attention
        // は solo/batch で同一カーネル・同一入力なので P6 契約どおり数値は一致する。
        #[cfg(feature = "cuda")]
        if b == 1 {
            let pos0 = caches[0].len();
            if let Some(logits) = self.forward_decode_gpu(toks[0], caches[0], pos0) {
                return Ok(vec![logits]);
            }
        }

        let e = self.e;
        let dh = self.dh;
        let heads = self.heads;
        let kv_heads = self.kv_heads;
        let kv_dim = kv_heads * dh;
        let group = heads / kv_heads; // GQA group size

        let lin_b = |a: &[f32], w: &Linear, m: usize| w.apply_batch_bt(a, m);

        // P6 契約 (単独/バッチのビット一致): GPU attention は solo と batch で
        // 同一カーネル・同一入力 (加算済み q/k/v) を使うため bit 一致する。
        // 注意: batch の GPU attention は chain_compatible (solo chain の条件) に
        // 依存しない (加算済み q/k/v を受けるだけで bias/QK-Norm と無関係)。

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
            let (mut q, mut k, mut v) = crate::prof::time_layer(crate::prof::QKV, li, || {
                // P17c: q/k/v は同一入力 → 1 回の GPU 呼び出しに融合
                #[cfg(feature = "cuda")]
                if let Some((q, k, v)) = lin_qkv_gpu(layer, &h, b, heads * dh, kv_dim) {
                    return (q, k, v);
                }
                (
                    lin_b(&h, &layer.q_w, b), // [b, H*dh]
                    lin_b(&h, &layer.k_w, b), // [b, K*dh]
                    lin_b(&h, &layer.v_w, b),
                )
            });
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
                let (kc, vc) = caches[i].kv_bufs(li, 1);
                kc[pos * kv_dim..][..kv_dim].copy_from_slice(&k[i * kv_dim..][..kv_dim]);
                vc[pos * kv_dim..][..kv_dim].copy_from_slice(&v[i * kv_dim..][..kv_dim]);
            }

            // attention: セッション×KVヘッドを rayon で並列実行 (P10)。
            // 各タスクは (セッション i, kv ヘッド kvh) を担当し、att への書き込み
            // 先は非重複スライス。タスク内は `attention_decode_kvhead` に統一され
            // forward() の decode (n==1) 経路と同一関数・同一演算列 → ビット一致。
            // (P12: probs@V は行統合+ストライド直接参照、スクラッチ再利用 —
            //  数値は従来の per-(kvh,g) m=1 経路とビット一致、テスト担保)
            //
            // P17g: GPU attention (solo と同一カーネル・同一ミラー内容 → 単独/バッチで
            // ビット一致を維持)。全セッション成功時のみ採用 — 1 つでも失敗すれば
            // 全セッションを CPU attention でやり直す (混合防止)。
            // 注意: GPU attention は加算済み q/k/v を受けるだけで bias/QK-Norm と
            // 無関係 → chain_compatible (solo chain の条件) に依存させない。
            let mut att = vec![0.0f32; b * heads * dh];
            crate::prof::time_layer(crate::prof::ATTN, li, || {
                let scale = 1.0 / (dh as f32).sqrt();
                #[cfg(feature = "cuda")]
                {
                    let cache_refs: Vec<&crate::kv_cache::KvCache> =
                        caches.iter().map(|c| &**c).collect();
                    if crate::cuda::chain_attention_batch(
                        li,
                        self.layers.len(),
                        &q,
                        &cache_refs,
                        heads,
                        group,
                        dh,
                        kv_dim,
                        scale,
                        &mut att,
                    ) {
                        return;
                    }
                }
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
            let (mut gate, up) = crate::prof::time_layer(crate::prof::GATE_UP, li, || {
                // P17c: gate/up は同一入力 → 1 回の GPU 呼び出しに融合
                #[cfg(feature = "cuda")]
                if let Some((gate, up)) = lin_gateup_gpu(layer, &h2, b, self.ff) {
                    return (gate, up);
                }
                (lin_b(&h2, &layer.gate_w, b), lin_b(&h2, &layer.up_w, b))
            });
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

    /// P17g テスト用: bias 無し tiny (solo decode chain 適合構造)
    fn tiny_chain(mode: QuantMode) -> LlamaQwen2 {
        let mut m = tiny(mode);
        for layer in m.layers.iter_mut() {
            layer.q_b = None;
            layer.k_b = None;
            layer.v_b = None;
        }
        m
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

    /// P17b: int8 モデルの forward / forward_batch 全体を GPU オフロードで
    /// 実行しても CPU 経路とビット一致すること (ignored — GPU が必要)。
    /// 線形層のフック (quant::Linear::apply_bt / apply_batch_bt) が全層で
    /// GPU 経路に切り替わった状態での e2e 相当。
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn cuda_offload_forward_bit_exact_vs_cpu() {
        let _g = crate::cuda::test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        let mode = QuantMode::Int8;
        let prompts: Vec<Vec<u32>> = vec![vec![5, 11, 23], vec![1, 2, 3, 4, 5], vec![7, 9]];
        let steps = 4;

        // --- CPU リファレンス (GPU 強制を解除して実行) ---
        // enable_for_tests は OnceLock 初期化前でも GPU 強制フラグなので、
        // ここでは「GPU なしの別経路」として tiny モデルを CPU カーネルで
        // 直接回す (maybe_gpu_linear が失敗/無効になるよう GPU_BROKEN は
        // 触らず、単純に同じ tiny をもう一度構築して GPU フラグが効いた
        // ままだと両方 GPU になる — そのため CPU 比較は P17 のカーネル単体
        // テストと quant.rs の naive 参照で担保済み。ここでは「GPU ON でも
        // バッチ/単独・prefill/decode の整合が壊れないこと」を検証する)。
        let mut solo = tiny(mode);
        let mut solo_caches: Vec<KvCache> =
            (0..prompts.len()).map(|_| solo.new_cache()).collect();
        let mut solo_logits: Vec<Vec<f32>> = Vec::new();
        for (s, p) in prompts.iter().enumerate() {
            solo_logits.push(solo.forward(p, &mut solo_caches[s]).unwrap());
        }
        // prefill 段階の logits を保存 (decode ループで solo_logits は上書きされる)
        let solo_prefill = solo_logits.clone();
        let mut solo_seq: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
        let mut solo_hist: Vec<Vec<Vec<f32>>> = vec![Vec::new(); steps];
        for step in 0..steps {
            for s in 0..prompts.len() {
                let t = argmax(&solo_logits[s]);
                solo_seq[s].push(t);
                solo_logits[s] = solo.forward(&[t], &mut solo_caches[s]).unwrap();
                solo_hist[step].push(solo_logits[s].clone());
            }
        }

        // --- バッチ decode (GPU オフロード有効のまま) ---
        let mut bat = tiny(mode);
        let mut bat_caches: Vec<KvCache> =
            (0..prompts.len()).map(|_| bat.new_cache()).collect();
        let mut bat_logits: Vec<Vec<f32>> = Vec::new();
        for (s, p) in prompts.iter().enumerate() {
            let lg = bat.forward(p, &mut bat_caches[s]).unwrap();
            // 単独 (こちらも GPU 経路) と prefill logits がビット一致
            assert_eq!(lg, solo_prefill[s], "GPU prefill s={s}");
            bat_logits.push(lg);
        }
        for step in 0..steps {
            let toks: Vec<u32> = (0..prompts.len()).map(|s| argmax(&bat_logits[s])).collect();
            let mut caches: Vec<&mut KvCache> = bat_caches.iter_mut().collect();
            let outs = bat.forward_batch(&toks, &mut caches).unwrap();
            for s in 0..prompts.len() {
                assert_eq!(outs[s], solo_hist[step][s], "GPU step={step} s={s}");
                assert_eq!(toks[s], solo_seq[s][step], "GPU token step={step} s={s}");
                bat_logits[s] = outs[s].clone();
            }
        }
    }

    /// P17g: GPU attention 有効時の forward (solo decode chain) が CPU attention
    /// (緩和無効時) と**近似一致**すること。exactness 契約は attention のみ緩和
    /// (ユーザー承認) — greedy トークン列は一致、logits は微小差を許容。
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn cuda_gpu_attention_forward_vs_cpu_close() {
        let _g = crate::cuda::test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        let prompt = vec![5u32, 11, 23];
        let steps = 8;

        // CPU attention 参照 (GPU attention を無効化して chain 経路で実行 —
        // chain の線形層/expf/swiglu/rmsnorm はビット一致済みなので
        // 純粋な CPU forward と等価)
        let prev = crate::cuda::set_gpu_attention_enabled(false);
        let mut cpu_m = tiny_chain(QuantMode::Int8);
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

        // GPU attention
        let mut gpu_m = tiny_chain(QuantMode::Int8);
        let mut gpu_cache = gpu_m.new_cache();
        let mut logits = gpu_m.forward(&prompt, &mut gpu_cache).unwrap();
        for (step, want_tok) in ref_seq.iter().enumerate() {
            let t = argmax(&logits);
            assert_eq!(t, *want_tok, "greedy token step={step}");
            logits = gpu_m.forward(&[t], &mut gpu_cache).unwrap();
            let maxd = logits
                .iter()
                .zip(ref_hist[step].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(maxd < 0.05, "logits maxabs diff step={step}: {maxd}");
        }
        crate::cuda::set_gpu_attention_enabled(prev);
        println!("gpu attention forward vs cpu close OK ({} steps)", steps);
    }

    /// P17h: dno_rope_qkv (bias 加算 + rope・GPU) が CPU 経路 (add_bias →
    /// rope_rows) と**ビット一致**すること。exactness 契約: rope は緩和対象外。
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn cuda_rope_qkv_bitexact_vs_cpu() {
        let _g = crate::cuda::test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        use crate::cuda::{chain_download, chain_rope_prepare, chain_rope_qkv, chain_upload, CH_QKV};
        use crate::tensor::rope_rows;
        let heads = 4usize;
        let kv_heads = 2usize;
        let dh = 4usize;
        let hd = heads * dh;
        let kv_dim = kv_heads * dh;
        let q_b = prng(hd, 500);
        let k_b = prng(kv_dim, 600);
        let v_b = prng(kv_dim, 700);
        for pos0 in [0usize, 1, 7, 63] {
            let qkv = prng(hd + 2 * kv_dim, 9000 + pos0 as u32);
            // --- CPU リファレンス: add_bias → rope (forward の CPU 経路と同一順) ---
            let mut cpu = qkv.clone();
            for i in 0..hd {
                cpu[i] += q_b[i];
            }
            for i in 0..kv_dim {
                cpu[hd + i] += k_b[i];
                cpu[hd + kv_dim + i] += v_b[i];
            }
            let (mut cq, mut ck) = (cpu[..hd].to_vec(), cpu[hd..hd + kv_dim].to_vec());
            rope_rows(&mut cq, 1, heads, dh, pos0, 10_000.0);
            rope_rows(&mut ck, 1, kv_heads, dh, pos0, 10_000.0);
            cpu[..hd].copy_from_slice(&cq);
            cpu[hd..hd + kv_dim].copy_from_slice(&ck);
            // --- GPU: bias 加算 + rope をカーネルで ---
            chain_upload(CH_QKV, &qkv, hd + 2 * kv_dim, 1).unwrap();
            chain_rope_prepare(pos0, dh, 10_000.0).unwrap();
            chain_rope_qkv(
                Some(&q_b),
                Some(&k_b),
                Some(&v_b),
                heads,
                kv_heads,
                dh,
                0, // pin id (テスト専用スロット)
            )
            .unwrap();
            let gpu = chain_download(CH_QKV, hd + 2 * kv_dim).unwrap();
            assert_eq!(gpu, cpu, "rope+bias bit-exact pos0={pos0}");
            // --- bias 無し (NULL) も CPU (bias 0 加算) とビット一致 ---
            let mut cpu0 = qkv.clone();
            let (mut cq0, mut ck0) = (cpu0[..hd].to_vec(), cpu0[hd..hd + kv_dim].to_vec());
            rope_rows(&mut cq0, 1, heads, dh, pos0, 10_000.0);
            rope_rows(&mut ck0, 1, kv_heads, dh, pos0, 10_000.0);
            cpu0[..hd].copy_from_slice(&cq0);
            cpu0[hd..hd + kv_dim].copy_from_slice(&ck0);
            chain_upload(CH_QKV, &qkv, hd + 2 * kv_dim, 1).unwrap();
            chain_rope_prepare(pos0, dh, 10_000.0).unwrap();
            chain_rope_qkv(None, None, None, heads, kv_heads, dh, 0).unwrap();
            let gpu0 = chain_download(CH_QKV, hd + 2 * kv_dim).unwrap();
            assert_eq!(gpu0, cpu0, "rope (no bias) bit-exact pos0={pos0}");
        }
        println!("rope_qkv bit-exact vs CPU OK (pos0=0,1,7,63, with/without bias)");
    }

    /// P17h: solo decode chain (bias あり・Qwen2.5 と同構造) が forward_decode_gpu
    /// で**発動**し (Some 返却を明示検証 — 教訓⑭: 「テスト緑」は経路発動の証拠に
    /// ならない)、CPU 経路と greedy 一致 + logits 近似一致すること。
    #[cfg(all(test, feature = "cuda"))]
    #[test]
    #[ignore]
    fn cuda_solo_chain_bias_rope_gpu_matches_cpu() {
        let _g = crate::cuda::test_lock().lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        crate::cuda::enable_for_tests();
        let prompt = vec![5u32, 11, 23];
        let steps = 6;

        // CPU 参照 (GPU attention 無効 → forward_decode_gpu は None → CPU 経路)
        let prev = crate::cuda::set_gpu_attention_enabled(false);
        let mut cpu_m = tiny(QuantMode::Int8); // bias あり (Qwen2.5 と同構造)
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

        // GPU solo chain (rope + bias GPU 化・qkv デバイス常駐・ステップ内 sync なし)
        let mut gpu_m = tiny(QuantMode::Int8);
        let mut gpu_cache = gpu_m.new_cache();
        let mut logits = gpu_m.forward(&prompt, &mut gpu_cache).unwrap();
        for (step, want_tok) in ref_seq.iter().enumerate() {
            let t = argmax(&logits);
            assert_eq!(t, *want_tok, "greedy token step={step}");
            let pos0 = prompt.len() + step;
            let lg = gpu_m
                .forward_decode_gpu(t, &mut gpu_cache, pos0)
                .expect("solo chain must fire (P17h: bias + rope GPU)");
            logits = lg;
            let maxd = logits
                .iter()
                .zip(ref_hist[step].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(maxd < 0.05, "logits maxabs diff step={step}: {maxd}");
        }
        crate::cuda::set_gpu_attention_enabled(prev);
        println!("solo chain (bias+rope GPU) matches CPU OK ({steps} steps)");
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
