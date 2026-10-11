//! safetensors からの重み読込ヘルパー(F16/BF16/F32 → f32)

use anyhow::{bail, Result};
use safetensors::{Dtype, SafeTensors};

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
