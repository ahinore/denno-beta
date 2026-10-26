//! モデルファミリー抽象とディスパッチ

pub mod expert_cache;
pub mod gpt2;
pub mod hyper;
pub mod indexer;
pub mod llama;
pub mod linear_gguf;
pub mod moe;
pub mod moe_expert;
pub mod qwen35;

use anyhow::{bail, Result};
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::quant::QuantMode;

pub trait Model: Send {
    /// コンテキスト長(prompt+生成 の上限)
    fn context(&self) -> usize;
    /// EOS トークンID(複数あり得る: Qwen2.5 は generation_config で2つ指定される)
    fn eos_ids(&self) -> &[u32];
    /// 再帰状態 (Qwen3.5 GDN の conv/SSM 状態) を持つモデルか。
    /// true の場合、KV の位置スライス再利用 (truncate / prefix entry 復元) は
    /// 状態を無効化するため、エンジンは拡張 (lcp == 現在長) の場合のみ再利用する。
    fn is_recurrent(&self) -> bool {
        false
    }
    /// tokens を forward し、最後の位置の logits を返す
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>>;
    /// P6: 複数セッションの decode 1 ステップ (各 1 トークン) を 1 forward に
    /// まとめて進める。toks[i] はセッション i の次トークン、caches[i] はその
    /// KV キャッシュ。戻り値は各セッションの logits。
    /// **exactness 契約**: 各セッションの logits は単独 forward とビット一致
    /// すること (デフォルト実装は単独 forward のループで満たす。バッチ化する
    /// 実装は演算順序を保つカーネルで満たす必要がある)。
    fn forward_batch(&mut self, toks: &[u32], caches: &mut [&mut KvCache]) -> Result<Vec<Vec<f32>>> {
        debug_assert_eq!(toks.len(), caches.len());
        let mut outs = Vec::with_capacity(toks.len());
        for (t, c) in toks.iter().zip(caches.iter_mut()) {
            outs.push(self.forward(&[*t], c)?);
        }
        Ok(outs)
    }
    /// 新しい KvCache を作る
    fn new_cache(&self) -> KvCache;
}

/// config.json から必須の整数値を取り出す
pub fn cfg_u64(cfg: &serde_json::Value, key: &str) -> Result<u64> {
    cfg.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow::anyhow!("config.json に {} がありません", key))
}

/// config.json の model_type からファミリーを決めてロードする。
/// `mode` が Int8 の場合は線形層の重みをロード時に int8 へオフライン量子化する
/// (P3。デフォルトの Fp32 は従来動作を完全維持)。
pub fn load_model(dir: &Path, mode: QuantMode) -> Result<Box<dyn Model>> {
    let cfg_path = dir.join("config.json");
    let cfg: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&cfg_path)
            .map_err(|e| anyhow::anyhow!("config.json を開けません ({:?}): {}", cfg_path, e))?,
    )?;
    match cfg.get("model_type").and_then(|v| v.as_str()) {
        Some("gpt2") => Ok(Box::new(gpt2::Gpt2::load(dir, &cfg, mode)?)),
        Some("qwen2") => Ok(Box::new(llama::LlamaQwen2::load(dir, &cfg, mode)?)),
        // Qwen3: Qwen2 との差分は QK-Norm (q/k 投影直後の head_dim 単位 RMSNorm) のみ。
        // QKV bias なし / tied・untied は config.json に従い LlamaQwen2 が両対応。
        Some("qwen3") => Ok(Box::new(llama::LlamaQwen2::load(dir, &cfg, mode)?)),
        // Qwen3.5: ハイブリッド GDN + GatedAttention・GGUF ローダ (P19-B/C)
        Some("qwen3_5") => Ok(Box::new(qwen35::Qwen35::load(dir, &cfg, mode)?)),
        other => bail!("未対応の model_type: {:?}(gpt2 / qwen2 / qwen3 のみ対応)", other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cfg_u64_reads_present_value() {
        let cfg: serde_json::Value = serde_json::json!({"num_layers": 24, "zero": 0});
        assert_eq!(cfg_u64(&cfg, "num_layers").unwrap(), 24);
        assert_eq!(cfg_u64(&cfg, "zero").unwrap(), 0);
    }

    #[test]
    fn cfg_u64_errors_on_missing_or_non_integer() {
        let cfg: serde_json::Value = serde_json::json!({"name": "gpt2", "f": 1.5});
        assert!(cfg_u64(&cfg, "missing").is_err());
        assert!(cfg_u64(&cfg, "name").is_err()); // 文字列
        assert!(cfg_u64(&cfg, "f").is_err()); // 小数 (as_u64 は None)
    }
}
