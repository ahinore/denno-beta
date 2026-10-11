//! safetensors からの重み読込ヘルパー(F16/BF16/F32 → f32)。
//! 単一 model.safetensors と分割 checkpoint(model.safetensors.index.json +
//! model-xxxxx-of-xxxxx.safetensors 群)の両対応。

use anyhow::{bail, Context, Result};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::Path;

/// SafeTensors から named tensor を f32 で取り出すヘルパー
pub fn get_f32(st: &SafeTensors, name: &str) -> Result<Vec<f32>> {
    let view = st
        .tensor(name)
        .map_err(|e| anyhow::anyhow!("tensor {:?} not found: {}", name, e))?;
    tensor_to_f32(&view)
}

/// 存在しない場合は None (例: bias の有無がモデルにより異なる)
pub fn get_f32_opt(st: &SafeTensors, name: &str) -> Result<Option<Vec<f32>>> {
    match st.tensor(name) {
        Ok(view) => Ok(Some(tensor_to_f32(&view)?)),
        Err(_) => Ok(None),
    }
}

/// TensorView のデータを f32 に変換
pub fn tensor_to_f32(view: &safetensors::tensor::TensorView) -> Result<Vec<f32>> {
    let n: usize = view.shape().iter().product();
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => {
            if data.len() != n * 4 {
                bail!("F32 tensor size mismatch");
            }
            Ok(data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
        }
        Dtype::F16 => {
            if data.len() != n * 2 {
                bail!("F16 tensor size mismatch");
            }
            Ok(data
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            if data.len() != n * 2 {
                bail!("BF16 tensor size mismatch");
            }
            // bf16 は f32 の上位16bit。bitを左に16シフトすれば f32 になる。
            Ok(data
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect())
        }
        other => bail!("unsupported dtype: {:?}", other),
    }
}

/// safetensors の重みストア。単一 `model.safetensors` と、HF 形式の分割
/// checkpoint (`model.safetensors.index.json` の weight_map が指す shard 群)の
/// 両方を透過的に扱う。
///
/// 全 shard を mmap し、tensor 名 → shard 番号の対応表を index.json(または
/// 単一ファイルの header)から構築する。呼び出し側はテンソル名だけで
/// shard の存在を意識せずに読める(untied lm_head の `lm_head.weight` が
/// どの shard にあっても同じ API で読める)。
pub struct WeightStore {
    /// `shards` が借用するメモリ領域。ライフタイムを 'static にエスケープして
    /// いるため、ここに保持し続けることで実質的な生存期間を担保する
    /// (mmap はストアが生きている間マップされ続ける)。
    _mmaps: Vec<memmap2::Mmap>,
    shards: Vec<SafeTensors<'static>>,
    map: HashMap<String, usize>,
}

impl WeightStore {
    /// モデルディレクトリ(HF形式)から開く。`model.safetensors.index.json` が
    /// あれば weight_map に現れる全 shard を開き、無ければ単一
    /// `model.safetensors` にフォールバックする。
    pub fn open(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let files: Vec<String> = if index_path.exists() {
            let index: serde_json::Value = serde_json::from_reader(
                std::fs::File::open(&index_path)?,
            )
            .with_context(|| format!("{} の解析に失敗", index_path.display()))?;
            let weight_map = index
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| anyhow::anyhow!("{} に weight_map がありません", index_path.display()))?;
            let mut files: Vec<String> =
                weight_map.values().filter_map(|v| v.as_str().map(String::from)).collect();
            files.sort();
            files.dedup();
            if files.is_empty() {
                bail!("{} の weight_map が空です", index_path.display());
            }
            files
        } else {
            vec!["model.safetensors".to_string()]
        };

        let mut mmaps = Vec::with_capacity(files.len());
        let mut shards = Vec::with_capacity(files.len());
        let mut map = HashMap::new();
        for (si, fname) in files.iter().enumerate() {
            let path = dir.join(fname);
            let file = std::fs::File::open(&path)
                .with_context(|| format!("重みファイルを開けません: {}", path.display()))?;
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            let st = unsafe {
                // mmap は self._mmaps に保持され、ストア生存中はマップされ続ける
                std::mem::transmute::<SafeTensors<'_>, SafeTensors<'static>>(
                    SafeTensors::deserialize(&mmap)?,
                )
            };
            for name in st.names() {
                map.insert(name.to_string(), si);
            }
            mmaps.push(mmap);
            shards.push(st);
        }
        Ok(Self { _mmaps: mmaps, shards, map })
    }

    /// tensor 名から shard を解決して TensorView を返す
    pub fn tensor(&self, name: &str) -> Result<safetensors::tensor::TensorView<'_>> {
        let si = *self
            .map
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {:?} が checkpoint に存在しません", name))?;
        Ok(self.shards[si].tensor(name)?)
    }

    /// named tensor を f32 で取り出す(単一ファイル時代の get_f32 相当)
    pub fn get_f32(&self, name: &str) -> Result<Vec<f32>> {
        tensor_to_f32(&self.tensor(name)?)
    }

    /// 存在しない場合は None(単一ファイル時代の get_f32_opt 相当)
    pub fn get_f32_opt(&self, name: &str) -> Result<Option<Vec<f32>>> {
        match self.map.get(name) {
            Some(_) => Ok(Some(self.get_f32(name)?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト用に safetensors ファイルを書き出す
    /// entries: (tensor名, shape, f32データ)
    fn write_st_file(path: &Path, entries: &[(&str, Vec<usize>, Vec<f32>)]) {
        // 先に全バイト列を確保してから view を作る (TensorView はデータを借用する)
        let mut all_bytes = Vec::with_capacity(entries.len());
        for (_, _, data) in entries {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            all_bytes.push(bytes);
        }
        let mut map = HashMap::new();
        for ((name, shape, _), bytes) in entries.iter().zip(all_bytes.iter()) {
            let view =
                safetensors::tensor::TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
            map.insert(name.to_string(), view);
        }
        safetensors::tensor::serialize_to_file(map, &None, path).unwrap();
    }

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "denno-load-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn weight_store_reads_across_shards() {
        let dir = temp_dir("sharded");
        // shard 0: embed と layer.0、shard 1: lm_head と layer.1 (HF分割の典型配置)
        write_st_file(
            &dir.join("model-00001-of-00002.safetensors"),
            &[
                ("model.embed_tokens.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                ("model.layers.0.weight", vec![2], vec![10.0, 20.0]),
            ],
        );
        write_st_file(
            &dir.join("model-00002-of-00002.safetensors"),
            &[
                ("lm_head.weight", vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]),
                ("model.layers.1.weight", vec![2], vec![30.0, 40.0]),
            ],
        );
        let index = serde_json::json!({
            "metadata": {"total_size": 48},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_string(&index).unwrap(),
        )
        .unwrap();

        let ws = WeightStore::open(&dir).unwrap();
        assert_eq!(ws.get_f32("model.embed_tokens.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ws.get_f32("model.layers.0.weight").unwrap(), vec![10.0, 20.0]);
        // 別 shard の tensor (untied lm_head) も透過的に読める
        assert_eq!(ws.get_f32("lm_head.weight").unwrap(), vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(ws.get_f32("model.layers.1.weight").unwrap(), vec![30.0, 40.0]);
        // 存在しない tensor: required はエラー、opt は None
        assert!(ws.get_f32("model.norm.weight").is_err());
        assert_eq!(ws.get_f32_opt("model.norm.weight").unwrap(), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn weight_store_falls_back_to_single_file() {
        let dir = temp_dir("single");
        write_st_file(
            &dir.join("model.safetensors"),
            &[("model.embed_tokens.weight", vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])],
        );

        let ws = WeightStore::open(&dir).unwrap();
        assert_eq!(ws.get_f32("model.embed_tokens.weight").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ws.get_f32_opt("lm_head.weight").unwrap(), None); // tied 前提
        assert!(ws.get_f32("lm_head.weight").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
