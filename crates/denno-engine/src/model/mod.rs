//! モデルファミリー抽象とディスパッチ

pub mod gpt2;
pub mod llama;

use anyhow::{bail, Result};
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::quant::QuantMode;

pub trait Model: Send {
    /// コンテキスト長(prompt+生成 の上限)
    fn context(&self) -> usize;
    /// EOS トークンID(複数あり得る: Qwen2.5 は generation_config で2つ指定される)
    fn eos_ids(&self) -> &[u32];
    /// tokens を forward し、最後の位置の logits を返す
    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>>;
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
        other => bail!("未対応の model_type: {:?}(gpt2 / qwen2 のみ対応)", other),
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
