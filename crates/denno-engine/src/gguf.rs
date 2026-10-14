//! GGUF コンテナリーダ (P19)。
//!
//! v2/v3 形式に対応 (Qwen3.5 の unsloth GGUF は v3)。仕様:
//! https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/gguf.cpp
//!
//! - 参照: GGUF 形式仕様 (ggml/llama.cpp — MIT License,
//!   Copyright (c) 2023-2026 The ggml authors)。形式仕様の参照のみで、
//!   パーサ本体は独自の Rust 実装 (コードの逐語コピーは含まない)。
//! - メタデータ KV とテンソル情報のみパースし、データ部は mmap のまま参照する。
//! - テンソル名は llama.cpp が付ける正規名 (例: blk.0.attn_q.weight)。

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::kquant::GgmlType;

#[derive(Debug)]
pub struct GgufError(pub String);

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gguf: {}", self.0)
    }
}

impl std::error::Error for GgufError {}

fn err<T>(msg: impl Into<String>) -> Result<T, GgufError> {
    Err(GgufError(msg.into()))
}

// GGUF メタデータ値型 (gguf.h enum gguf_type と同一)
const GT_U8: u32 = 0;
const GT_I8: u32 = 1;
const GT_U16: u32 = 2;
const GT_I16: u32 = 3;
const GT_U32: u32 = 4;
const GT_I32: u32 = 5;
const GT_F32: u32 = 6;
const GT_BOOL: u32 = 7;
const GT_STRING: u32 = 8;
const GT_ARRAY: u32 = 9;
const GT_U64: u32 = 10;
const GT_I64: u32 = 11;
const GT_F64: u32 = 12;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    /// 配列はフラットに保持 (ネスト配列は未対応 = 仕様上も非推奨)
    Arr(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) => Some(*v as u64),
            GgufValue::I16(v) => Some(*v as u64),
            GgufValue::I32(v) => Some(*v as u64),
            GgufValue::I64(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            GgufValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    /// 次元 (ne[0] が最速・行)
    pub dims: [u64; 4],
    pub ty: GgmlType,
    /// データ部先頭からのオフセット
    pub offset: u64,
    pub nbytes: u64,
}

pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub alignment: u64,
    /// テンソルデータ部のファイル内開始位置
    pub data_start: u64,
    pub metadata: HashMap<String, GgufValue>,
    /// (テンソル名 → 情報)
    pub tensors: HashMap<String, GgufTensorInfo>,
    /// ファイル内順 (dequant の列挙・検証用)
    pub tensor_order: Vec<String>,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let file = File::open(path.as_ref())
            .map_err(|e| GgufError(format!("open {:?}: {e}", path.as_ref())))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| GgufError(format!("mmap: {e}")))? };
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self, GgufError> {
        let b = &mmap[..];
        let mut p = 0usize;

        let magic = rd_u32(b, &mut p)?;
        if magic != 0x46554747 {
            return err(format!("bad magic {magic:#x} (not GGUF)"));
        }
        let version = rd_u32(b, &mut p)?;
        if version < 2 {
            return err(format!("GGUF v{version} は未対応 (v2/v3 のみ)"));
        }
        let tensor_count = rd_u64(b, &mut p)?;
        let kv_count = rd_u64(b, &mut p)?;

        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = rd_string(b, &mut p)?;
            let value = rd_value(b, &mut p)?;
            metadata.insert(key, value);
        }

        let mut tensor_order = Vec::with_capacity(tensor_count as usize);
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let name = rd_string(b, &mut p)?;
            let n_dims = rd_u32(b, &mut p)? as usize;
            if n_dims > 4 {
                return err(format!("{name}: n_dims={n_dims} > 4"));
            }
            let mut dims = [1u64; 4];
            for d in dims.iter_mut().take(n_dims) {
                *d = rd_u64(b, &mut p)?;
            }
            let ty_id = rd_u32(b, &mut p)?;
            let ty = GgmlType::from_id(ty_id)
                .ok_or_else(|| GgufError(format!("{name}: 未対応の型 id={ty_id}")))?;
            let offset = rd_u64(b, &mut p)?;

            let ne: u64 = dims.iter().take(n_dims).product::<u64>().max(1);
            let blck = ty.blck_size() as u64;
            let nbytes = ne / blck * ty.type_size() as u64
                + if ne % blck != 0 { ty.type_size() as u64 } else { 0 };
            tensor_order.push(name.clone());
            tensors.insert(name.clone(), GgufTensorInfo { name, dims, ty, offset, nbytes });
        }

        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32);
        if !alignment.is_power_of_two() {
            return err(format!("general.alignment={alignment} (2の冪が必要)"));
        }
        // データ部は alignment に整列
        let data_start = (p + (alignment as usize - 1)) & !(alignment as usize - 1);
        if data_start > b.len() {
            return err("データ部がファイル終端を超えている");
        }

        Ok(GgufFile { mmap, version, alignment, data_start: data_start as u64, metadata, tensors, tensor_order })
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// テンソルの生バイト列を返す。
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError(format!("テンソル {name} が無い")))?;
        let start = self.data_start + t.offset;
        let end = start + t.nbytes;
        if end as usize > self.mmap.len() {
            return err(format!("{name}: データがファイル終端を超過 ({end} > {})", self.mmap.len()));
        }
        Ok(&self.mmap[start as usize..end as usize])
    }

    /// テンソル全体を f32 へデ量子化して返す (総要素数 = Π dims)。
    pub fn dequant_tensor(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError(format!("テンソル {name} が無い")))?;
        let bytes = self.tensor_bytes(name)?;
        let n = t.dims.iter().product::<u64>() as usize;
        let bs = t.ty.blck_size();
        if bs > 1 && n % bs != 0 {
            return err(format!("{name}: 要素数 {n} がブロックサイズ {bs} の倍数でない"));
        }
        Ok(crate::kquant::dequant_row(t.ty, bytes, n))
    }
}

#[inline]
fn rd_u32(b: &[u8], p: &mut usize) -> Result<u32, GgufError> {
    if *p + 4 > b.len() {
        return err("EOF (u32)");
    }
    let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4;
    Ok(v)
}

#[inline]
fn rd_u64(b: &[u8], p: &mut usize) -> Result<u64, GgufError> {
    if *p + 8 > b.len() {
        return err("EOF (u64)");
    }
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[*p..*p + 8]);
    *p += 8;
    Ok(u64::from_le_bytes(a))
}

fn rd_string(b: &[u8], p: &mut usize) -> Result<String, GgufError> {
    let len = rd_u64(b, p)? as usize;
    if *p + len > b.len() {
        return err("EOF (string)");
    }
    let s = String::from_utf8_lossy(&b[*p..*p + len]).into_owned();
    *p += len;
    Ok(s)
}

fn rd_value(b: &[u8], p: &mut usize) -> Result<GgufValue, GgufError> {
    let ty = rd_u32(b, p)?;
    Ok(match ty {
        GT_U8 => GgufValue::U8(rd_u8(b, p)?),
        GT_I8 => GgufValue::I8(rd_u8(b, p)? as i8),
        GT_U16 => GgufValue::U16(rd_u16(b, p)?),
        GT_I16 => GgufValue::I16(rd_u16(b, p)? as i16),
        GT_U32 => GgufValue::U32(rd_u32(b, p)?),
        GT_I32 => GgufValue::I32(rd_u32(b, p)? as i32),
        GT_F32 => GgufValue::F32(f32::from_bits(rd_u32(b, p)?)),
        GT_BOOL => GgufValue::Bool(rd_u8(b, p)? != 0),
        GT_STRING => GgufValue::Str(rd_string(b, p)?),
        GT_ARRAY => {
            let elem_ty = rd_u32(b, p)?;
            let count = rd_u64(b, p)? as usize;
            let mut v = Vec::with_capacity(count.min(1 << 20));
            for _ in 0..count {
                v.push(rd_value_of(b, p, elem_ty)?);
            }
            GgufValue::Arr(v)
        }
        GT_U64 => GgufValue::U64(rd_u64(b, p)?),
        GT_I64 => GgufValue::I64(rd_u64(b, p)? as i64),
        GT_F64 => GgufValue::F64(f64::from_bits(rd_u64(b, p)?)),
        other => return err(format!("未知のメタデータ型 {other}")),
    })
}

/// 配列要素用 (型を外から与える)
fn rd_value_of(b: &[u8], p: &mut usize, ty: u32) -> Result<GgufValue, GgufError> {
    Ok(match ty {
        GT_U8 => GgufValue::U8(rd_u8(b, p)?),
        GT_I8 => GgufValue::I8(rd_u8(b, p)? as i8),
        GT_U16 => GgufValue::U16(rd_u16(b, p)?),
        GT_I16 => GgufValue::I16(rd_u16(b, p)? as i16),
        GT_U32 => GgufValue::U32(rd_u32(b, p)?),
        GT_I32 => GgufValue::I32(rd_u32(b, p)? as i32),
        GT_F32 => GgufValue::F32(f32::from_bits(rd_u32(b, p)?)),
        GT_BOOL => GgufValue::Bool(rd_u8(b, p)? != 0),
        GT_STRING => GgufValue::Str(rd_string(b, p)?),
        GT_U64 => GgufValue::U64(rd_u64(b, p)?),
        GT_I64 => GgufValue::I64(rd_u64(b, p)? as i64),
        GT_F64 => GgufValue::F64(f64::from_bits(rd_u64(b, p)?)),
        other => return err(format!("未知の配列要素型 {other}")),
    })
}

#[inline]
fn rd_u8(b: &[u8], p: &mut usize) -> Result<u8, GgufError> {
    if *p >= b.len() {
        return err("EOF (u8)");
    }
    let v = b[*p];
    *p += 1;
    Ok(v)
}

#[inline]
fn rd_u16(b: &[u8], p: &mut usize) -> Result<u16, GgufError> {
    if *p + 2 > b.len() {
        return err("EOF (u16)");
    }
    let v = u16::from_le_bytes([b[*p], b[*p + 1]]);
    *p += 2;
    Ok(v)
}
