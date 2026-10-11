//! モデルファミリー抽象とディスパッチ

pub mod gpt2;
pub mod llama;

use anyhow::{bail, Result};
use std::path::Path;

use crate::kv_cache::KvCache;

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

/// config.json の model_type からファミリーを決めてロードする
pub fn load_model(dir: &Path) -> Result<Box<dyn Model>> {
    let cfg_path = dir.join("config.json");
    let cfg: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(&cfg_path)
            .map_err(|e| anyhow::anyhow!("config.json を開けません ({:?}): {}", cfg_path, e))?,
    )?;
    match cfg.get("model_type").and_then(|v| v.as_str()) {
        Some("gpt2") => Ok(Box::new(gpt2::Gpt2::load(dir, &cfg)?)),
        Some("qwen2") => Ok(Box::new(llama::LlamaQwen2::load(dir, &cfg)?)),
        other => bail!("未対応の model_type: {:?}(gpt2 / qwen2 のみ対応)", other),
    }
}
