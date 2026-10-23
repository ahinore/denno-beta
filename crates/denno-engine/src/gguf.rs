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
use crate::quant::{ChunkedQuantStats, Linear};

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
    /// s164 (K5): 配列の要素数。配列以外は None。
    /// (既存の as_u64 / as_f32 / as_str / as_bool の意味は変えない = additive)
    pub fn as_array_len(&self) -> Option<usize> {
        match self {
            GgufValue::Arr(v) => Some(v.len()),
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
        let p = parse_fields(&mmap)?;
        Ok(GgufFile {
            mmap,
            version: p.version,
            alignment: p.alignment,
            data_start: p.data_start,
            metadata: p.metadata,
            tensors: p.tensors,
            tensor_order: p.tensor_order,
        })
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

    /// P37: 行チャンク量子化の前処理。テンソルが `(k, n)` の 2-D として読め、
    /// 1 行がブロック境界に載っているときだけ
    /// `(生バイト列, 1 行のバイト数, 型)` を返す。
    ///
    /// どれか 1 つでも成り立たなければ `Ok(None)` を返し、呼び出し側は従来の
    /// `dequant_tensor` → 量子化へ落ちる (**fail-closed**)。
    fn chunked_rows(
        &self,
        name: &str,
        k: usize,
        n: usize,
    ) -> Result<Option<(&[u8], usize, GgmlType)>, GgufError> {
        let Some(t) = self.tensor(name) else {
            return Ok(None);
        };
        if t.dims.is_empty() || t.dims[0] as usize != k || n == 0 {
            return Ok(None);
        }
        let rows: u64 = t.dims[1..].iter().product();
        if rows as usize != n {
            return Ok(None);
        }
        let bs = t.ty.blck_size();
        if bs == 0 || k % bs != 0 {
            return Ok(None);
        }
        let row_bytes = (k / bs) * t.ty.type_size();
        let bytes = self.tensor_bytes(name)?;
        if bytes.len() != n * row_bytes {
            return Ok(None);
        }
        Ok(Some((bytes, row_bytes, t.ty)))
    }

    /// P37: 2-D 線形重みを**行チャンクで**量子化する (テンソル全体の f32 を
    /// 作らない)。生成される量子化バイト列は従来経路
    /// (`dequant_tensor` → `Linear::new`) と同一。
    ///
    /// 形・型・バイト数のどれかが想定外なら `Ok(None)` (= 呼び出し側は従来経路)。
    /// ピーク一時領域は 1 チャンク分だけ (`ChunkedQuantStats::peak_bytes`)。
    pub fn linear_chunked(
        &self,
        name: &str,
        k: usize,
        n: usize,
        mode: crate::quant::QuantMode,
        chunk_rows: usize,
    ) -> Result<Option<(Linear, ChunkedQuantStats)>, GgufError> {
        let Some((bytes, row_bytes, ty)) = self.chunked_rows(name, k, n)? else {
            return Ok(None);
        };
        let res = crate::quant::gguf_rows_quantized_chunked(
            ty,
            mode,
            k,
            n,
            chunk_rows,
            |r0, rows, buf| {
                let off = r0 * row_bytes;
                let len = rows * row_bytes;
                buf.clear();
                buf.extend_from_slice(&bytes[off..off + len]);
                Ok(())
            },
        );
        match res {
            Ok(v) => Ok(Some(v)),
            Err(e) => {
                tracing::warn!("{name}: 行チャンク量子化を中止して従来経路へ ({e})");
                Ok(None)
            }
        }
    }

    /// P37: `token_embd.weight` 級の巨大 2-D 重みを、**常駐 f32 表** (embedding
    /// gather 用) と**量子化コピー** (tied lm_head 用) へ同じ行チャンクで埋める。
    ///
    /// 従来経路は (1) 全体を f32 へ展開 → (2) `embed.clone()` でもう 1 部 →
    /// (3) 量子化、で 2 部の f32 が同時に生きていた (27B Q4_0 で約 10.9 GB)。
    /// ここでは 1 チャンク分の f32 だけを経由する。
    pub fn embed_and_linear_chunked(
        &self,
        name: &str,
        k: usize,
        n: usize,
        mode: crate::quant::QuantMode,
        chunk_rows: usize,
    ) -> Result<Option<(Vec<f32>, Linear, ChunkedQuantStats)>, GgufError> {
        let Some((bytes, row_bytes, ty)) = self.chunked_rows(name, k, n)? else {
            return Ok(None);
        };
        let Some(total) = k.checked_mul(n) else {
            return Ok(None);
        };
        let mut table = vec![0f32; total];
        let res = crate::quant::gguf_rows_quantized_chunked_with_sink(
            ty,
            mode,
            k,
            n,
            chunk_rows,
            |r0, rows, buf| {
                let off = r0 * row_bytes;
                let len = rows * row_bytes;
                buf.clear();
                buf.extend_from_slice(&bytes[off..off + len]);
                Ok(())
            },
            |r0, w| {
                table[r0 * k..r0 * k + w.len()].copy_from_slice(w);
            },
        );
        match res {
            Ok((l, st)) => Ok(Some((table, l, st))),
            Err(e) => {
                tracing::warn!("{name}: 行チャンク量子化を中止して従来経路へ ({e})");
                Ok(None)
            }
        }
    }

    /// P42: `token_embd.weight` を **ファイル自身の生ブロックのまま**常駐させ、
    /// tied lm_head 用の量子化コピーを同じ行チャンクから作る。
    ///
    /// 従来の [`GgufFile::embed_and_linear_chunked`] は gather 用に `k*n` 個の
    /// f32 (27B Q4_0 で 5,085.6 MB) を常駐させる。ここではその代わりに
    /// `n*row_bytes` (同じ 27B で 715.2 MB) の生ブロックを確保するだけで、
    /// 行は呼び出し側が `kquant::dequant_row(_into)` で必要分だけ復元する
    /// (**再量子化はしない** — 復元値は f32 表の同じ行とビット一致。P38 が実
    /// 27B の全 248,320 行で `to_bits()` 差分 0 を実測)。
    ///
    /// `fill` は従来と同じチャンク読み出しなので、`blocks` は
    /// `tensor_bytes(name)` とバイト一致し、`Linear` は
    /// `embed_and_linear_chunked` が返すものとバイト一致する (同じ
    /// `gguf_rows_quantized_chunked_with_sink` を同じ順で通る)。
    ///
    /// 形・型・バイト数が想定外なら `Ok(None)` (呼び出し側は従来経路 = fail-closed)。
    pub fn embed_blocks_and_linear_chunked(
        &self,
        name: &str,
        k: usize,
        n: usize,
        mode: crate::quant::QuantMode,
        chunk_rows: usize,
    ) -> Result<Option<(EmbedBlocks, Linear, ChunkedQuantStats)>, GgufError> {
        let Some((bytes, row_bytes, ty)) = self.chunked_rows(name, k, n)? else {
            return Ok(None);
        };
        // P42 v1 は Q4_0 のみ (行境界 = ブロック境界が保証される型)。
        if ty != GgmlType::Q4_0 {
            return Ok(None);
        }
        let Some(total) = n.checked_mul(row_bytes) else {
            return Ok(None);
        };
        let mut blocks = vec![0u8; total];
        let res = crate::quant::gguf_rows_quantized_chunked_with_sink(
            ty,
            mode,
            k,
            n,
            chunk_rows,
            |r0, rows, buf| {
                let off = r0 * row_bytes;
                let len = rows * row_bytes;
                buf.clear();
                buf.extend_from_slice(&bytes[off..off + len]);
                // 同じチャンクを生ブロックの常駐先へも書く (ピークを増やさない)。
                blocks[off..off + len].copy_from_slice(&buf[..]);
                Ok(())
            },
            |_, _| {},
        );
        match res {
            Ok((l, st)) => Ok(Some((
                EmbedBlocks {
                    blocks,
                    k,
                    n,
                    row_bytes,
                    ty,
                },
                l,
                st,
            ))),
            Err(e) => {
                tracing::warn!("{name}: 生ブロック常駐を中止して従来経路へ ({e})");
                Ok(None)
            }
        }
    }

    /// P42: [`GgufFile::embed_blocks_and_linear_chunked`] の前提判定 (ヘッダのみ)。
    ///
    /// `tied && chunked` は呼び出し側 (モデルローダ) の条件なのでここには無い。
    /// 1 つでも崩れたら `false` = 従来の f32 表経路。
    pub fn embed_blocks_eligible(&self, name: &str, k: usize, n: usize) -> bool {
        let Some(t) = self.tensor(name) else {
            return false;
        };
        embed_blocks_eligible(Some(t.ty), k, n, t.nbytes)
    }
}

/// P42: `token_embd` を生ブロック常駐 + 行デ量子化で扱えるかの **純関数**。
///
/// 前提 (どれか 1 つでも崩れたら従来の f32 表):
/// - ファイル型が Q4_0 (行境界 = ブロック境界)、
/// - `k` がブロックサイズ (32) の倍数、
/// - `nbytes == n * (k/32)*18` (= 行が隙間なく並ぶ)。
///
/// `vocab * e == elems` は `chunked_rows` が dims から検証済み。
pub fn embed_blocks_eligible(ty: Option<GgmlType>, k: usize, n: usize, nbytes: u64) -> bool {
    let bs = GgmlType::Q4_0.blck_size();
    if ty != Some(GgmlType::Q4_0) || k == 0 || n == 0 || bs == 0 || k % bs != 0 {
        return false;
    }
    let row_bytes = (k / bs) * GgmlType::Q4_0.type_size();
    nbytes == (n as u64) * (row_bytes as u64)
}

/// P42: `token_embd.weight` の生ブロック常駐 (Option B) の戻り値。
///
/// `blocks` はファイルの `n*row_bytes` バイトそのもの (再量子化なし)。
/// 行 `r` は `blocks[r*row_bytes .. (r+1)*row_bytes]` を `ty` でデ量子化する。
#[derive(Debug, Clone)]
pub struct EmbedBlocks {
    pub blocks: Vec<u8>,
    /// 1 行の要素数
    pub k: usize,
    /// 行数
    pub n: usize,
    /// 1 行のバイト数
    pub row_bytes: usize,
    pub ty: GgmlType,
}

/// P35: ヘッダ (メタデータ + テンソル情報) だけを保持する mmap 無しのビュー。
///
/// `GgufFile` は 16 GB 級のモデルを丸ごと mmap するが、ロード経路の検証
/// (P35 の passthrough 証明テスト) では「ヘッダだけ読んで、必要なテンソルの
/// バイト列を pread で必要分だけ読む」形にしたい。`GgufHeader::parse` は
/// ヘッダを含む先頭バッファ (`&[u8]`) から同じパーサで読み、データ部の
/// 絶対オフセット (`data_start + offset`) を呼び出し側へ渡す。
#[derive(Debug, Clone)]
pub struct GgufHeader {
    pub version: u32,
    pub alignment: u64,
    /// テンソルデータ部のファイル内開始位置
    pub data_start: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    pub tensor_order: Vec<String>,
}

impl GgufHeader {
    /// 先頭バッファからヘッダを解析する (データ部は含まなくてよい)。
    /// バッファがヘッダ末尾に届いていない場合は "EOF" 系のエラーを返すので、
    /// 呼び出し側はバッファを伸ばして再試行できる。
    pub fn parse(b: &[u8]) -> Result<Self, GgufError> {
        let p = parse_fields(b)?;
        Ok(GgufHeader {
            version: p.version,
            alignment: p.alignment,
            data_start: p.data_start,
            metadata: p.metadata,
            tensors: p.tensors,
            tensor_order: p.tensor_order,
        })
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.get(key)
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.get(name)
    }

    /// テンソルデータのファイル内 [start, end) オフセット。
    pub fn tensor_range(&self, name: &str) -> Result<(u64, u64), GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError(format!("テンソル {name} が無い")))?;
        let start = self.data_start + t.offset;
        Ok((start, start + t.nbytes))
    }
}

/// GGUF ヘッダの解析結果 (GgufFile / GgufHeader の共通部)。
struct ParsedFields {
    version: u32,
    alignment: u64,
    data_start: u64,
    metadata: HashMap<String, GgufValue>,
    tensors: HashMap<String, GgufTensorInfo>,
    tensor_order: Vec<String>,
}

/// ヘッダ部の解析本体 (mmap でも先頭バッファでも同じ関数を通す)。
fn parse_fields(b: &[u8]) -> Result<ParsedFields, GgufError> {
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

    Ok(ParsedFields { version, alignment, data_start: data_start as u64, metadata, tensors, tensor_order })
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

#[cfg(test)]
mod tests {
    //! P35: GGUF の Q4_0 生ブロックを再量子化せずに積んでよいかの実データ証明。
    //!
    //! ホストのメモリ規律: 16 GB のモデルを mmap しない・全体を読まない・
    //! 2 部コピーを持たない。ヘッダだけ先頭バッファから解析し (`GgufHeader`)、
    //! テンソルのバイト列は pread (`seek` + `read_exact`) で 1 チャンク
    //! (行境界に整列した ~4 MB) ずつ読んで、f32 展開 → 再量子化 → 生バイト比較を
    //! その場で捨てる。ピークは 1 チャンク分 (数十 MB)。

    use super::*;
    use crate::kquant::{dequant_row, GgmlType};
    use crate::quant::{
        gguf_rows_quantized_chunked, ChunkedQuantStats, Linear, Quant8Linear, QuantMode, Q4Linear,
        Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS,
    };
    use std::collections::BTreeMap;
    use std::io::{Read, Seek, SeekFrom};

    /// サンプルするテンソル数の上限 (既定スイートを重くしない・時間を有界にする)
    const MAX_TENSORS: usize = 100;
    /// 1 回に読むバイト数の目安 (行境界に切り上げ)
    const CHUNK_BYTES: usize = 4 << 20;
    /// ヘッダ読みの上限 (27B のヘッダは数 MB。ここを超えるのは異常)
    const HEADER_CAP: usize = 512 << 20;

    fn mem_available_gib() -> Option<f64> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("MemAvailable:") {
                let kb: f64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb / (1024.0 * 1024.0));
            }
        }
        None
    }

    /// ヘッダだけを先頭バッファから解析する (mmap しない)。足りなければ伸ばす。
    fn read_header(file: &mut std::fs::File, file_len: u64) -> GgufHeader {
        let mut want = 1usize << 20;
        loop {
            let n = want.min(file_len as usize);
            let mut buf = vec![0u8; n];
            file.seek(SeekFrom::Start(0)).expect("seek");
            file.read_exact(&mut buf).expect("read header");
            match GgufHeader::parse(&buf) {
                Ok(h) => return h,
                Err(e) => {
                    let short = e.0.contains("EOF") || e.0.contains("データ部がファイル終端");
                    if !short || n >= HEADER_CAP {
                        panic!("GGUF ヘッダ解析失敗 ({n} B 読み): {e}");
                    }
                    want *= 2;
                }
            }
        }
    }

    /// このプロセスのピーク RSS (VmHWM) を bytes で返す。
    fn peak_rss_bytes() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    /// VmHWM を現在の RSS まで戻す (Linux: /proc/self/clear_refs へ "5")。
    /// これで「直前の区間だけのピーク」を測れる。
    fn reset_peak_rss() {
        let _ = std::fs::write("/proc/self/clear_refs", "5");
    }

    /// 現在の RSS (VmRSS) を bytes で返す (区間の基準値)。
    fn rss_now_bytes() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in s.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
                return Some(kb * 1024);
            }
        }
        None
    }

    /// 実測 RSS の増分 (MB)。基準が取れないときは NaN。
    fn delta_mb(peak: Option<u64>, base: Option<u64>) -> f64 {
        match (peak, base) {
            (Some(p), Some(b)) => p.saturating_sub(b) as f64 / 1e6,
            _ => f64::NAN,
        }
    }

    #[derive(Default)]
    struct Stat {
        /// 型に属するテンソル総数 (ファイル全体)
        total: usize,
        total_bytes: u64,
        /// サンプルした本数
        sampled: usize,
        /// passthrough 対象 (型 Q4_0・2D・k が 32 の倍数・バイト数一致)
        eligible: usize,
        /// 生ブロック == 再量子化バイト列 だった本数
        identical: usize,
        compared_bytes: u64,
        /// 最初の不一致 (テンソル名, テンソル内バイト位置, ファイル内絶対位置,
        /// ファイル側のブロックスケール, 再量子化側のブロックスケール,
        /// そのブロック内で一致したバイト数 / 18)
        first_diff: Option<FirstDiff>,
    }

    #[derive(Debug, Clone)]
    struct FirstDiff {
        name: String,
        byte_in_tensor: u64,
        file_offset: u64,
        stored_scale: f32,
        requant_scale: f32,
        /// スケール (先頭 2B) を除く 16B のニブルが一致したか
        nibbles_equal: bool,
    }

    /// P35: 実 GGUF の Q4_0 生ブロックが、engine の再量子化経路
    /// (GGUF → fp32 → Q4_0) と **ビット一致**するかを型ごとに判定する。
    ///
    /// 実行:
    ///   GGUF_PATH=/work/models/Qwen3.8-27B-Q4_0.gguf \
    ///   cargo test --release -p denno-engine --features cuda --lib -- --ignored \
    ///     --nocapture q4_verbatim_matches_requant_real_gguf
    #[test]
    #[ignore]
    fn q4_verbatim_matches_requant_real_gguf() {
        let path = match std::env::var("GGUF_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                let d = std::path::PathBuf::from("/work/models/Qwen3.8-27B-Q4_0.gguf");
                if !d.exists() {
                    eprintln!("skip: GGUF_PATH 未設定かつ /work/models に既定モデルが無い");
                    return;
                }
                d
            }
        };

        // メモリ規律: 8 GiB 未満なら走らせない (ホストが OOM 寸前のことがある)
        match mem_available_gib() {
            Some(g) if g < 8.0 => {
                panic!("MemAvailable {g:.1} GiB < 8 GiB — 走らせない (`free -g` を確認してから再実行)")
            }
            Some(g) => println!("[p35] MemAvailable {g:.1} GiB"),
            None => println!("[p35] MemAvailable 不明 (/proc/meminfo 無し) — 続行"),
        }

        let mut file = std::fs::File::open(&path).expect("GGUF を開けない");
        let file_len = file.metadata().expect("metadata").len();
        println!("[p35] file {} ({:.2} GiB)", path.display(), file_len as f64 / (1u64 << 30) as f64);

        // ヘッダだけを先頭バッファから解析する (mmap しない)。足りなければ伸ばす。
        let hdr = read_header(&mut file, file_len);
        println!(
            "[p35] version {} / alignment {} / data_start {} / {} tensors",
            hdr.version,
            hdr.alignment,
            hdr.data_start,
            hdr.tensor_order.len()
        );

        // 型ごとの全テンソル (ファイル順) と総バイト数
        let mut by_ty: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut stats: BTreeMap<String, Stat> = BTreeMap::new();
        for name in &hdr.tensor_order {
            let t = hdr.tensor(name).expect("tensor_order の名前は必ず引ける");
            let key = format!("{:?}", t.ty);
            by_ty.entry(key.clone()).or_default().push(name.clone());
            let st = stats.entry(key).or_default();
            st.total += 1;
            st.total_bytes += t.nbytes;
        }

        // 型ごとに均等間隔でサンプルする (合計 ~MAX_TENSORS 本)
        let per_ty = (MAX_TENSORS / by_ty.len().max(1)).max(1);
        let mut sampled: Vec<String> = Vec::new();
        for names in by_ty.values() {
            if names.len() <= per_ty {
                sampled.extend(names.iter().cloned());
            } else {
                for i in 0..per_ty {
                    let idx = i * (names.len() - 1) / (per_ty.max(2) - 1);
                    sampled.push(names[idx].clone());
                }
            }
        }
        println!("[p35] sampling {} tensors ({} per type cap)", sampled.len(), per_ty);

        let t0 = std::time::Instant::now();
        for name in &sampled {
            let t = hdr.tensor(name).expect("sampled name").clone();
            let key = format!("{:?}", t.ty);
            let k = t.dims[0] as usize;
            let n: usize = t.dims[1..].iter().product::<u64>() as usize;
            let row_bytes = if k > 0 && k % Q4_BLOCK_ELEMS == 0 {
                Q4Linear::row_bytes(k)
            } else {
                0
            };
            let eligible = t.ty == GgmlType::Q4_0
                && row_bytes > 0
                && t.dims[2..].iter().all(|&d| d == 1)
                && n > 0
                && t.nbytes as usize == n * row_bytes;

            let st = stats.get_mut(&key).expect("型は登録済み");
            st.sampled += 1;
            if !eligible {
                println!("[p35]   {name:<40} {key:<6} k={k:<6} n={n:<6} n/a (layout != Q4_0)");
                continue;
            }
            st.eligible += 1;

            let (start, end) = hdr.tensor_range(name).expect("range");
            assert_eq!(end - start, t.nbytes, "{name}: range と nbytes の不一致");
            let rows_per_chunk = (CHUNK_BYTES / row_bytes).max(1);
            let mut row = 0usize;
            let mut identical = true;
            while row < n {
                let rows = (n - row).min(rows_per_chunk);
                let off = start + (row * row_bytes) as u64;
                file.seek(SeekFrom::Start(off)).expect("seek");
                let mut buf = vec![0u8; rows * row_bytes];
                file.read_exact(&mut buf).expect("read tensor chunk");
                // 現行の再量子化経路と同一の関数 (dequant_row → Q4Linear::quantize)
                let w = dequant_row(GgmlType::Q4_0, &buf, rows * k);
                let q = Q4Linear::quantize(&w, k, rows);
                st.compared_bytes += buf.len() as u64;
                if q.blocks != buf {
                    let i = q
                        .blocks
                        .iter()
                        .zip(buf.iter())
                        .position(|(a, b)| a != b)
                        .expect("!= なら必ず差がある");
                    let b0 = (i / Q4_BLOCK_BYTES) * Q4_BLOCK_BYTES;
                    let stored_scale = half::f16::from_le_bytes([buf[b0], buf[b0 + 1]]).to_f32();
                    let requant_scale =
                        half::f16::from_le_bytes([q.blocks[b0], q.blocks[b0 + 1]]).to_f32();
                    let nibbles_equal = buf[b0 + 2..b0 + Q4_BLOCK_BYTES]
                        == q.blocks[b0 + 2..b0 + Q4_BLOCK_BYTES];
                    st.first_diff = Some(FirstDiff {
                        name: name.clone(),
                        byte_in_tensor: i as u64,
                        file_offset: off + i as u64,
                        stored_scale,
                        requant_scale,
                        nibbles_equal,
                    });
                    identical = false;
                    break;
                }
                row += rows;
            }
            if identical {
                st.identical += 1;
                println!("[p35]   {name:<40} {key:<6} k={k:<6} n={n:<6} BIT-IDENTICAL");
            } else {
                let d = st.first_diff.as_ref().expect("DIFF なら first_diff がある");
                println!(
                    "[p35]   {name:<40} {key:<6} k={k:<6} n={n:<6} DIFF @byte {} (file {}) \
                     scale {:.6e} -> {:.6e} nibbles {}",
                    d.byte_in_tensor,
                    d.file_offset,
                    d.stored_scale,
                    d.requant_scale,
                    if d.nibbles_equal { "equal" } else { "differ" }
                );
            }
        }
        let secs = t0.elapsed().as_secs_f64();

        println!("\n=== P35 per-type verdict (verbatim GGUF blocks vs re-quantised) ===");
        println!(
            "{:<8} {:>7} {:>7} {:>7} {:>8} {:>9} {:>10}  verdict",
            "type", "total", "GiB", "sampled", "eligible", "identical", "GiB(comp)"
        );
        let mut any_eligible = false;
        let mut all_identical = true;
        for (ty, st) in &stats {
            if st.eligible > 0 {
                any_eligible = true;
            }
            let verdict = if st.eligible == 0 {
                "n/a: not Q4_0 layout (stays on the fp32 path)"
            } else if st.identical == st.eligible {
                "BIT-IDENTICAL"
            } else {
                all_identical = false;
                "NOT identical (keep fp32 path)"
            };
            println!(
                "{:<8} {:>7} {:>7.2} {:>7} {:>8} {:>9} {:>10.2}  {verdict}",
                ty,
                st.total,
                st.total_bytes as f64 / (1u64 << 30) as f64,
                st.sampled,
                st.eligible,
                st.identical,
                st.compared_bytes as f64 / (1u64 << 30) as f64
            );
            if let Some(d) = &st.first_diff {
                println!(
                    "    first diff: {} byte {} within tensor (file offset {}); \
                     block scale {:.6e} (file) vs {:.6e} (re-quant), nibbles {}",
                    d.name,
                    d.byte_in_tensor,
                    d.file_offset,
                    d.stored_scale,
                    d.requant_scale,
                    if d.nibbles_equal { "equal" } else { "differ" }
                );
            }
        }
        println!("\n[p35] {:.1} s, {} tensors compared", secs, sampled.len());

        assert!(
            any_eligible,
            "Q4_0 テンソルが 1 本も比較されていない (対象ファイル/型を確認)"
        );
        // 安全不変条件: passthrough を許可している型は、このサンプルで
        // ビット一致していなければならない。逆に「一致しない型」は許可リストに
        // 入れてはいけない (P35 の実測では Q4_0 が一致せず、許可リストは空)。
        for (ty, st) in &stats {
            let allow = match ty.as_str() {
                "Q4_0" => crate::quant::gguf_passthrough_allows(GgmlType::Q4_0),
                _ => false,
            };
            if allow {
                assert!(
                    st.identical == st.eligible && st.eligible > 0,
                    "{ty} は passthrough 許可リストにあるがビット一致しない \
                     → 許可リストから外すこと (fp32 経路を維持)"
                );
            }
        }
        let _ = all_identical; // 判定は上の表とレポートに出す (テストの合否ではない)
    }

    // -----------------------------------------------------------------------
    // P37: 行チャンク量子化のバイト一致証明 (実 GGUF)
    // -----------------------------------------------------------------------

    /// サンプルするテンソル数の上限 (メモリと時間を有界にする)
    const MAX_SAMPLE: usize = 10;
    /// int8 の副次比較を行う要素数の上限 (token_embd の 1.27 G 要素は除外)
    const INT8_ELEM_LIMIT: usize = 64 << 20;

    /// 1 テンソル分の判定結果
    struct ChunkVerdict {
        name: String,
        ty: GgmlType,
        k: usize,
        n: usize,
        identical: bool,
        /// 最初の不一致バイト位置 (テンソル内)
        first_diff: Option<usize>,
        /// チャンク経路の統計 (1 チャンクの一時領域 = ピーク)
        chunk: ChunkedQuantStats,
        /// チャンク経路の実測ピーク RSS (区間内) と区間開始時の RSS
        chunk_rss: Option<u64>,
        chunk_base_rss: Option<u64>,
        /// 従来経路 (全体 f32) の計算上のピーク f32 一時領域
        whole_f32_bytes: usize,
        /// 従来経路の実測ピーク RSS (区間内) と区間開始時の RSS
        whole_rss: Option<u64>,
        whole_base_rss: Option<u64>,
        /// int8 の副次比較 (None = 対象外)
        int8: Option<bool>,
    }

    /// P37 の受け入れ条件: 行チャンク量子化が「テンソル全体を f32 へ展開 →
    /// 量子化」と**バイト一致**することを実 GGUF で示す。
    ///
    /// - 比較するのは `Q4Linear` の blocks (engine の `--quant q4` の重みそのもの)。
    ///   チャンク経路 = `gguf_rows_quantized_chunked` (pread で行チャンクだけ読む)、
    ///   従来経路 = `dequant_row(全体)` → `Q4Linear::quantize` (= `Linear::new`)。
    /// - メモリ規律: mmap しない (`GgufHeader` だけ先頭バッファから解析)、
    ///   テンソルは seek + read_exact、2 部コピーを持たない。
    ///   `MemAvailable` が 8 GiB 未満なら走らせない。
    /// - サンプルは 10 本まで (token_embd + output + ffn 3 + attn 4 + ssm_out)。
    ///
    /// 実行:
    ///   GGUF_PATH=/work/models/Qwen3.8-27B-Q4_0.gguf \
    ///   CARGO_TARGET_DIR=/tmp/p37-target \
    ///   cargo test --release -p denno-engine --features cuda --lib -- --ignored \
    ///     --nocapture q4_chunked_matches_whole_tensor_real_gguf
    #[test]
    #[ignore]
    fn q4_chunked_matches_whole_tensor_real_gguf() {
        let path = match std::env::var("GGUF_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                let d = std::path::PathBuf::from("/work/models/Qwen3.8-27B-Q4_0.gguf");
                if !d.exists() {
                    eprintln!("skip: GGUF_PATH 未設定かつ /work/models に既定モデルが無い");
                    return;
                }
                d
            }
        };

        // メモリ規律: 8 GiB 未満なら走らせない (ホストが OOM 寸前のことがある)
        let avail_gib = mem_available_gib();
        match avail_gib {
            Some(g) if g < 8.0 => {
                panic!("MemAvailable {g:.1} GiB < 8 GiB — 走らせない (`free -g` を確認してから再実行)")
            }
            Some(g) => println!("[p37] MemAvailable {g:.1} GiB"),
            None => println!("[p37] MemAvailable 不明 (/proc/meminfo 無し) — 続行"),
        }

        let mut file = std::fs::File::open(&path).expect("GGUF を開けない");
        let file_len = file.metadata().expect("metadata").len();
        println!(
            "[p37] file {} ({:.2} GiB)",
            path.display(),
            file_len as f64 / (1u64 << 30) as f64
        );
        let hdr = read_header(&mut file, file_len);
        println!("[p37] {} tensors / data_start {}", hdr.tensor_order.len(), hdr.data_start);

        // 代表テンソル: token_embd (最大) → output (K 系) → ffn 3 種 → attn 4 種
        let mut sampled: Vec<String> = Vec::new();
        for name in ["token_embd.weight", "output.weight"] {
            if hdr.tensor(name).is_some() {
                sampled.push(name.to_string());
            }
        }
        for suf in [
            ".ffn_up.weight",
            ".ffn_gate.weight",
            ".ffn_down.weight",
            ".attn_q.weight",
            ".attn_qkv.weight",
            ".attn_v.weight",
            ".attn_output.weight",
            ".ssm_out.weight",
        ] {
            if let Some(n) = hdr.tensor_order.iter().find(|n| n.ends_with(suf)) {
                if !sampled.iter().any(|s| s == n) {
                    sampled.push(n.clone());
                }
            }
        }
        sampled.truncate(MAX_SAMPLE);
        println!("[p37] sampling {} tensors:", sampled.len());
        for s in &sampled {
            println!("[p37]   {s}");
        }

        let mut verdicts: Vec<ChunkVerdict> = Vec::new();
        for name in &sampled {
            let t = hdr.tensor(name).expect("sampled name").clone();
            let k = t.dims[0] as usize;
            let n: usize = t.dims[1..].iter().product::<u64>() as usize;
            let bs = t.ty.blck_size();
            let row_bytes =
                if bs > 0 && k % bs == 0 { (k / bs) * t.ty.type_size() } else { 0 };
            if t.dims.len() < 2 || row_bytes == 0 || n == 0 || t.nbytes as usize != n * row_bytes {
                println!(
                    "[p37]   {name:<32} n/a (2-D として読めない: dims={:?} k={k} n={n})",
                    t.dims
                );
                continue;
            }
            let elems = k * n;
            let (start, end) = hdr.tensor_range(name).expect("range");
            assert_eq!(end - start, t.nbytes, "{name}: range と nbytes の不一致");

            // メモリ見積り: 従来経路は生バイトと f32 が同時に生きる。
            // 足りないのに確保すると abort するので、先に弾く。
            let need = t.nbytes + 4 * elems as u64;
            if let Some(g) = avail_gib {
                let avail = (g * (1u64 << 30) as f64) as u64;
                assert!(
                    need + (2u64 << 30) < avail,
                    "{name}: 従来経路の見積り {:.2} GiB + 余裕 2 GiB が MemAvailable {:.2} GiB を超過 — \
                     メモリを空けてから再実行",
                    need as f64 / (1u64 << 30) as f64,
                    g
                );
            }

            // (a) 行チャンク経路を**先に**測る (直前の巨大確保が無い clean な
            //     RSS 基準で測るため)。必要な行だけ pread して 1 チャンクずつ量子化。
            reset_peak_rss();
            let chunk_base_rss = rss_now_bytes();
            let rows = crate::quant::chunk_rows_for(k);
            let mut reader = |r0: usize, rows: usize, buf: &mut Vec<u8>| -> Result<(), String> {
                let off = start + (r0 * row_bytes) as u64;
                let len = rows * row_bytes;
                buf.clear();
                buf.resize(len, 0);
                file.seek(SeekFrom::Start(off)).map_err(|e| format!("seek {off}: {e}"))?;
                file.read_exact(buf).map_err(|e| format!("read {len} @{off}: {e}"))
            };
            let (a_q4, st) =
                gguf_rows_quantized_chunked(t.ty, QuantMode::Q4_0, k, n, rows, &mut reader)
                    .expect("行チャンク量子化 (Q4_0)");
            let chunk_rss = peak_rss_bytes();

            // (b) 従来経路: テンソル全体を f32 へ展開 → 量子化 (= Linear::new)
            reset_peak_rss();
            let whole_base_rss = rss_now_bytes();
            let mut raw = vec![0u8; t.nbytes as usize];
            reader(0, n, &mut raw).expect("read whole tensor");
            let whole = dequant_row(t.ty, &raw, elems);
            drop(raw);
            let b_q4 = Q4Linear::quantize(&whole, k, n);
            let b_q8 = if elems <= INT8_ELEM_LIMIT {
                Some(Quant8Linear::quantize(&whole, k, n))
            } else {
                None
            };
            let whole_rss = peak_rss_bytes();
            drop(whole);

            // int8 の副次比較 (チャンク経路も従来経路も同じ `Quant8Linear`)
            let a_q8 = if b_q8.is_some() {
                let (l, _) =
                    gguf_rows_quantized_chunked(t.ty, QuantMode::Int8, k, n, rows, &mut reader)
                        .expect("行チャンク量子化 (int8)");
                Some(l)
            } else {
                None
            };

            // 参照のまま比較する (大きな blocks を clone しない)
            let (a_blocks, b_blocks) = match &a_q4 {
                Linear::Q4(a) => (&a.blocks, &b_q4.blocks),
                _ => panic!("Q4 の線形重みではない"),
            };
            let identical = a_blocks == b_blocks;
            let first_diff = a_blocks.iter().zip(b_blocks.iter()).position(|(x, y)| x != y);
            let int8 = match (&a_q8, &b_q8) {
                (Some(Linear::Q8(a)), Some(b)) => {
                    let bits = |v: &Vec<f32>| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    Some(a.w == b.w && bits(&a.scale) == bits(&b.scale))
                }
                (None, None) => None,
                _ => Some(false),
            };
            println!(
                "[p37]   {name:<32} {:?} k={k:<6} n={n:<6} {}  chunks={} x {} rows  \
                 chunk-peak={:.1} MB (rss {} MB over base {} MB)  whole-f32={:.1} MB (rss {} MB over base {} MB)  int8={}",
                t.ty,
                if identical { "IDENTICAL" } else { "DIFF" },
                st.chunks,
                st.chunk_rows,
                st.peak_bytes() as f64 / 1e6,
                delta_mb(chunk_rss, chunk_base_rss),
                chunk_base_rss.map(|b| b as f64 / 1e6).unwrap_or(f64::NAN),
                elems as f64 * 4.0 / 1e6,
                delta_mb(whole_rss, whole_base_rss),
                whole_base_rss.map(|b| b as f64 / 1e6).unwrap_or(f64::NAN),
                match int8 {
                    Some(true) => "IDENTICAL".to_string(),
                    Some(false) => "DIFF".to_string(),
                    None => "n/a (>64Mi elems)".to_string(),
                }
            );
            if let Some(i) = first_diff {
                let b0 = (i / Q4_BLOCK_BYTES) * Q4_BLOCK_BYTES;
                println!(
                    "[p37]     !!! 最初の不一致 byte {i} (ブロック先頭 {b0}, ブロック内 {} B)",
                    i - b0
                );
            }
            verdicts.push(ChunkVerdict {
                name: name.clone(),
                ty: t.ty,
                k,
                n,
                identical,
                first_diff,
                chunk: st,
                chunk_rss,
                chunk_base_rss,
                whole_f32_bytes: elems * 4,
                whole_rss,
                whole_base_rss,
                int8,
            });
        }

        println!("\n=== P37 per-tensor verdict (行チャンク vs 全体 f32 → 量子化) ===");
        println!(
            "{:<32} {:<6} {:>6} {:>7} {:>10} {:>13} {:>13} {:>10} {:>10}  q4/int8",
            "tensor",
            "type",
            "k",
            "n",
            "verdict",
            "chunk peak(MB)",
            "whole f32(MB)",
            "rss ch(MB)",
            "rss wh(MB)"
        );
        let mut all_identical = true;
        let mut max_chunk_peak = 0usize;
        let mut max_whole_peak = 0usize;
        for v in &verdicts {
            all_identical &= v.identical;
            max_chunk_peak = max_chunk_peak.max(v.chunk.peak_bytes());
            max_whole_peak = max_whole_peak.max(v.whole_f32_bytes);
            println!(
                "{:<32} {:<6} {:>6} {:>7} {:>10} {:>13.2} {:>13.2} {:>10.1} {:>10.1}  {}/{}",
                v.name,
                format!("{:?}", v.ty),
                v.k,
                v.n,
                if v.identical { "IDENTICAL" } else { "DIFF" },
                v.chunk.peak_bytes() as f64 / 1e6,
                v.whole_f32_bytes as f64 / 1e6,
                delta_mb(v.chunk_rss, v.chunk_base_rss),
                delta_mb(v.whole_rss, v.whole_base_rss),
                if v.identical { "ok" } else { "DIFF" },
                match v.int8 {
                    Some(true) => "ok",
                    Some(false) => "DIFF",
                    None => "n/a",
                }
            );
        }
        println!(
            "[p37] max peak transient (COMPUTED): chunked {:.2} MB / whole-tensor {:.2} MB ({:.0}x)",
            max_chunk_peak as f64 / 1e6,
            max_whole_peak as f64 / 1e6,
            max_whole_peak as f64 / max_chunk_peak.max(1) as f64
        );
        for v in &verdicts {
            if !v.identical {
                println!(
                    "[p37] NOT IDENTICAL: {} {:?} k={} n={} first_diff={:?}",
                    v.name, v.ty, v.k, v.n, v.first_diff
                );
            }
        }

        assert!(!verdicts.is_empty(), "比較したテンソルが 0 本 (対象ファイル/型を確認)");
        assert!(
            verdicts.iter().any(|v| v.name == "token_embd.weight"),
            "token_embd.weight が比較されていない (受け入れ条件)"
        );
        assert!(
            all_identical,
            "行チャンク量子化が従来経路とバイト一致しないテンソルがある → そのテンソルは \
             チャンク経路を有効にしてはいけない (上の表と first_diff を参照)"
        );
    }

    // =====================================================================
    // P38: token_embd.weight の f32 gather 表 (27B Q4_0 で 5.086 GB 常駐) を
    // 「行単位のオンデマンド・デ量子化」で置き換えられるかの証明テスト。
    //
    // 表はローダがファイルの Q4_0 ブロックを **デ量子化するだけ** で作られる
    // (再量子化ではない) ので、同じブロックから 1 行だけデ量子化すれば
    // ビット一致するはず — それを実測で確かめる。数値経路は一切変更しない。
    // =====================================================================

    /// P38: 1 行分の判定。
    #[derive(Debug, Clone)]
    struct P38Row {
        row: usize,
        /// 行が属するローダのチャンク先頭行
        chunk_r0: usize,
        /// (a) 1 行だけ pread → dequant と (b) 表の同じ行がビット一致したか
        identical: bool,
        /// 最初の不一致要素 (None = 全一致)
        first_diff: Option<usize>,
        a_bits: u32,
        b_bits: u32,
    }

    fn p38_push(v: &mut Vec<usize>, n: usize, r: usize) {
        if r < n && !v.contains(&r) {
            v.push(r);
        }
    }

    /// P38 のサンプル行: 行 0・最終行・チャンク境界の前後・等間隔の散らし。
    fn p38_sample_rows(n: usize, chunk_rows: usize, want: usize) -> Vec<usize> {
        let mut v: Vec<usize> = Vec::new();
        p38_push(&mut v, n, 0);
        p38_push(&mut v, n, 1);
        p38_push(&mut v, n, 2);
        p38_push(&mut v, n, n - 1);
        p38_push(&mut v, n, n - 2);
        for c in 0..3usize {
            let b = c * chunk_rows;
            p38_push(&mut v, n, b.saturating_sub(1));
            p38_push(&mut v, n, b);
            p38_push(&mut v, n, b + 1);
        }
        let step = (n / want.max(1)).max(1);
        let mut r = 3usize;
        while v.len() < want && r < n - 1 {
            p38_push(&mut v, n, r);
            r += step;
        }
        let mut r = n / 2;
        while v.len() < want && r < n - 1 {
            p38_push(&mut v, n, r);
            r += 1;
        }
        v.sort_unstable();
        v
    }

    /// P38: ファイルの [off, off + rows*row_bytes) を pread する (mmap しない)。
    fn p38_read_rows(
        file: &mut std::fs::File,
        off: u64,
        row_bytes: usize,
        rows: usize,
    ) -> Vec<u8> {
        let len = rows * row_bytes;
        let mut buf = vec![0u8; len];
        file.seek(SeekFrom::Start(off)).expect("seek rows");
        file.read_exact(&mut buf).expect("read rows");
        buf
    }

    /// P38 の受け入れ条件テスト。
    ///
    /// 実行:
    ///   GGUF_PATH=/work/models/Qwen3.8-27B-Q4_0.gguf \
    ///   CARGO_TARGET_DIR=/tmp/p38-target \
    ///   cargo test --release -p denno-engine --features cuda --lib -- --ignored \
    ///     --nocapture q4_row_dequant_matches_embed_table_real_gguf
    ///
    /// 環境変数:
    ///   DENNO_P38_SWEEP=0 で全行スイープ (248,320 行) を省略する (既定は実行)。
    #[test]
    #[ignore]
    fn q4_row_dequant_matches_embed_table_real_gguf() {
        let path = match std::env::var("GGUF_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                let d = std::path::PathBuf::from("/work/models/Qwen3.8-27B-Q4_0.gguf");
                if !d.exists() {
                    eprintln!("skip: GGUF_PATH 未設定かつ /work/models に既定モデルが無い");
                    return;
                }
                d
            }
        };

        // メモリ規律: 8 GiB 未満なら走らせない (ホストが OOM 寸前のことがある)
        let avail_gib = mem_available_gib();
        match avail_gib {
            Some(g) if g < 8.0 => {
                panic!("MemAvailable {g:.1} GiB < 8 GiB — 走らせない (`free -g` を確認してから再実行)")
            }
            Some(g) => println!("[p38] MemAvailable {g:.2} GiB"),
            None => println!("[p38] MemAvailable 不明 (/proc/meminfo 無し) — 続行"),
        }

        let mut file = std::fs::File::open(&path).expect("GGUF を開けない");
        let file_len = file.metadata().expect("metadata").len();
        println!(
            "[p38] file {} ({:.2} GiB)",
            path.display(),
            file_len as f64 / (1u64 << 30) as f64
        );
        let hdr = read_header(&mut file, file_len);
        let t = hdr
            .tensor("token_embd.weight")
            .expect("token_embd.weight が無い")
            .clone();
        assert!(t.dims.len() >= 2, "token_embd.weight が 2-D でない: {:?}", t.dims);
        let k = t.dims[0] as usize; // 5120 (= e, 行の最内)
        // 行数は dim1 以降の積 (GGUF は末尾に 1 を詰めるので dims.len() は 4 になり得る)
        let n: usize = t.dims[1..].iter().product::<u64>() as usize; // 248320 (= vocab)
        let elems = k * n;
        let bs = t.ty.blck_size();
        assert!(bs > 0 && k % bs == 0, "1 行 {k} 要素がブロック {bs} の倍数でない");
        let row_bytes = (k / bs) * t.ty.type_size();
        assert_eq!(
            t.nbytes as usize,
            n * row_bytes,
            "nbytes と 行数×行長 の不一致 (行が連続でない)"
        );
        let (start, end) = hdr.tensor_range("token_embd.weight").expect("range");
        assert_eq!(end - start, t.nbytes, "tensor_range と nbytes の不一致");
        let table_bytes = elems * 4;
        let raw_bytes = t.nbytes as usize;
        println!(
            "[p38] tensor {:?} k={k} n={n} blocks/row={} row_bytes={row_bytes} \
             raw={:.1} MB  f32 table={:.1} MB ({:.3} GiB)  file+{start}",
            t.ty,
            k / bs,
            raw_bytes as f64 / 1e6,
            table_bytes as f64 / 1e6,
            table_bytes as f64 / (1u64 << 30) as f64
        );

        let chunk_rows = crate::quant::chunk_rows_for(k).clamp(1, n);
        println!(
            "[p38] loader chunk = {chunk_rows} rows ({} elems, f32 {:.1} MB, raw {:.1} MB) — \
             P37 の embed_and_linear_chunked と同じ粒度",
            chunk_rows * k,
            (chunk_rows * k * 4) as f64 / 1e6,
            (chunk_rows * row_bytes) as f64 / 1e6
        );

        // ---- サンプル行 (受け入れ条件: 32 行以上・行 0・最終行・散らし) ----
        let want = 40usize;
        let rows = p38_sample_rows(n, chunk_rows, want);
        assert!(
            rows.len() >= 32,
            "サンプル行が 32 未満 ({} 行) — 受け入れ条件を満たさない",
            rows.len()
        );
        println!("[p38] sampling {} rows", rows.len());

        // ---- (a) 1 行だけ pread → dequant_row  /  (b) ローダのチャンクの同じ行 ----
        // (b) は `gguf_rows_quantized_chunked_with_sink` が表へ書く値そのもの
        //     (quant.rs の `dequant_row(ty, &buf, rows * k)` → sink → table コピー)。
        let mut verdicts: Vec<P38Row> = Vec::new();
        let mut chunk_cache: std::collections::BTreeMap<usize, Vec<f32>> =
            std::collections::BTreeMap::new();
        let mut per_row_peak = 0usize;
        reset_peak_rss();
        let base_rss = rss_now_bytes();
        for &r in &rows {
            let r0 = (r / chunk_rows) * chunk_rows;
            if !chunk_cache.contains_key(&r0) {
                // 直前のチャンクだけ保持する (全チャンクを溜めない)
                chunk_cache.clear();
                let crows = (n - r0).min(chunk_rows);
                let buf = p38_read_rows(
                    &mut file,
                    start + (r0 * row_bytes) as u64,
                    row_bytes,
                    crows,
                );
                let w = dequant_row(t.ty, &buf, crows * k);
                chunk_cache.insert(r0, w);
            }
            let table = chunk_cache.get(&r0).expect("chunk");

            // (a) この行のブロックだけを pread してデ量子化する
            let a = dequant_row(
                t.ty,
                &p38_read_rows(&mut file, start + (r * row_bytes) as u64, row_bytes, 1),
                k,
            );
            per_row_peak = per_row_peak.max(row_bytes + k * 4);

            // (b) 表の同じ行
            let b = &table[(r - r0) * k..(r - r0) * k + k];
            let first_diff = a
                .iter()
                .zip(b.iter())
                .position(|(x, y)| x.to_bits() != y.to_bits());
            let (a_bits, b_bits) = match first_diff {
                Some(i) => (a[i].to_bits(), b[i].to_bits()),
                None => (0, 0),
            };
            verdicts.push(P38Row {
                row: r,
                chunk_r0: r0,
                identical: first_diff.is_none(),
                first_diff,
                a_bits,
                b_bits,
            });
        }
        let per_row_rss = delta_mb(peak_rss_bytes(), base_rss);

        println!("\n=== P38 per-row verdict (1 行 dequant vs ローダの表の同じ行) ===");
        println!(
            "{:>8} {:>10} {:>10} {:>10}  first_diff",
            "row", "chunk_r0", "verdict", "elems"
        );
        for v in &verdicts {
            println!(
                "{:>8} {:>10} {:>10} {:>10}  {}",
                v.row,
                v.chunk_r0,
                if v.identical { "IDENTICAL" } else { "DIFF" },
                k,
                match v.first_diff {
                    None => "-".to_string(),
                    Some(i) => format!("elem {i}: a={:#010x} b={:#010x}", v.a_bits, v.b_bits),
                }
            );
        }
        let all_identical = verdicts.iter().all(|v| v.identical);
        let distinct_chunks = {
            let mut c: Vec<usize> = verdicts.iter().map(|v| v.chunk_r0).collect();
            c.sort_unstable();
            c.dedup();
            c.len()
        };
        println!(
            "[p38] sampled {} rows over {} loader chunks: {} identical / {} diff",
            verdicts.len(),
            distinct_chunks,
            verdicts.iter().filter(|v| v.identical).count(),
            verdicts.iter().filter(|v| !v.identical).count()
        );
        println!(
            "[p38] per-row path transient (COMPUTED) = row_bytes {} B + f32 row {} B = {} B ({:.1} KB); \
             measured rss delta {:.2} MB (chunk cache {:.1} MB を含む)",
            row_bytes,
            k * 4,
            per_row_peak,
            per_row_peak as f64 / 1e3,
            per_row_rss,
            (chunk_rows * k * 4) as f64 / 1e6
        );

        // ---- 追加の参照: テンソル全体を 1 回の dequant_row で f32 表にする ----
        // (メモリ規律: 5.086 GB + 生 715 MB。MemAvailable が 20 GiB 以上のときだけ)
        let whole_ok = match avail_gib {
            Some(g) if g >= 20.0 => {
                let need = table_bytes + raw_bytes + (2usize << 30);
                let avail = (g * (1u64 << 30) as f64) as usize;
                if need < avail {
                    println!(
                        "[p38] whole-tensor table 参照を実行 (必要 {:.2} GiB < MemAvailable {g:.2} GiB)",
                        need as f64 / (1u64 << 30) as f64
                    );
                    reset_peak_rss();
                    let wbase = rss_now_bytes();
                    let raw = p38_read_rows(&mut file, start, row_bytes, n);
                    let whole = dequant_row(t.ty, &raw, elems);
                    drop(raw);
                    let wpeak = peak_rss_bytes();
                    let mut bad = 0usize;
                    for &r in &rows {
                        let a = dequant_row(
                            t.ty,
                            &p38_read_rows(
                                &mut file,
                                start + (r * row_bytes) as u64,
                                row_bytes,
                                1,
                            ),
                            k,
                        );
                        let b = &whole[r * k..r * k + k];
                        if a.iter().zip(b.iter()).any(|(x, y)| x.to_bits() != y.to_bits()) {
                            bad += 1;
                            println!("[p38] WHOLE-TENSOR DIFF at row {r}");
                        }
                    }
                    println!(
                        "[p38] whole-tensor table 参照: {} rows compared, {bad} diff, \
                         measured rss +{:.0} MB (計算値 {:.1} MB)",
                        rows.len(),
                        delta_mb(wpeak, wbase),
                        (table_bytes + raw_bytes) as f64 / 1e6
                    );
                    drop(whole);
                    Some(bad == 0)
                } else {
                    println!(
                        "[p38] whole-tensor table 参照をスキップ (必要 {:.2} GiB + 余裕 2 GiB > \
                         MemAvailable {g:.2} GiB)",
                        need as f64 / (1u64 << 30) as f64
                    );
                    None
                }
            }
            _ => {
                println!(
                    "[p38] whole-tensor table 参照をスキップ (MemAvailable {} GiB < 20 GiB の規律) \
                     — 代わりに全行スイープでローダのチャンク表の全行と突き合わせる",
                    avail_gib.map(|g| format!("{g:.2}")).unwrap_or_else(|| "?".to_string())
                );
                None
            }
        };

        // ---- 全行スイープ (メモリ有界): 全 248,320 行を 1 行 dequant と比較 ----
        let sweep_on = std::env::var("DENNO_P38_SWEEP")
            .ok()
            .map(|v| {
                !matches!(
                    v.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            })
            .unwrap_or(true);
        let (mut swept_rows, mut swept_bad, mut sweep_first_bad) = (0usize, 0usize, None::<usize>);
        let mut sweep_rss = f64::NAN;
        if sweep_on {
            reset_peak_rss();
            let sbase = rss_now_bytes();
            let t0 = std::time::Instant::now();
            let mut r0 = 0usize;
            while r0 < n {
                let crows = (n - r0).min(chunk_rows);
                let buf = p38_read_rows(
                    &mut file,
                    start + (r0 * row_bytes) as u64,
                    row_bytes,
                    crows,
                );
                let w = dequant_row(t.ty, &buf, crows * k);
                for j in 0..crows {
                    let per = dequant_row(
                        t.ty,
                        &buf[j * row_bytes..(j + 1) * row_bytes],
                        k,
                    );
                    let tab = &w[j * k..(j + 1) * k];
                    if per
                        .iter()
                        .zip(tab.iter())
                        .any(|(x, y)| x.to_bits() != y.to_bits())
                    {
                        swept_bad += 1;
                        if sweep_first_bad.is_none() {
                            sweep_first_bad = Some(r0 + j);
                        }
                    }
                }
                swept_rows += crows;
                r0 += crows;
            }
            sweep_rss = delta_mb(peak_rss_bytes(), sbase);
            println!(
                "[p38] FULL SWEEP: {swept_rows} rows / {swept_bad} diff (first {:?}) in {:.2} s, \
                 peak rss +{:.1} MB (1 チャンクだけ保持)",
                sweep_first_bad,
                t0.elapsed().as_secs_f64(),
                sweep_rss
            );
        } else {
            println!("[p38] FULL SWEEP をスキップ (DENNO_P38_SWEEP=0)");
        }

        // ---- マイクロベンチ: B 行のオンデマンド・デ量子化 ----
        // 実経路 = 1 行ずつ seek + read_exact + dequant_row。
        // 予算 17.3 ms/token (docs/KPI.md: 17.28〜17.32 ms/token) に対する算術。
        const DECODE_MS_PER_TOKEN: f64 = 17.30;
        println!("\n=== P38 microbench (B 行/token のオンデマンド・デ量子化) ===");
        println!(
            "{:>3} {:>10} {:>12} {:>12} {:>12} {:>12} {:>10}",
            "B", "reps", "median us", "us/row", "mean us", "pure-dq us/row", "% of 17.30ms"
        );
        for &b in &[1usize, 8usize, 32usize] {
            // 決定的に散らした行 id (重複なし)
            let ids: Vec<usize> = (0..b).map(|i| (i * 7919 + 13) % n).collect();
            // ページキャッシュを温める
            for _ in 0..3 {
                for &r in &ids {
                    let _ = dequant_row(
                        t.ty,
                        &p38_read_rows(&mut file, start + (r * row_bytes) as u64, row_bytes, 1),
                        k,
                    );
                }
            }
            let reps = match b {
                1 => 300usize,
                8 => 150,
                _ => 80,
            };
            let mut samples: Vec<f64> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t0 = std::time::Instant::now();
                let mut acc = 0f32;
                for &r in &ids {
                    let w = dequant_row(
                        t.ty,
                        &p38_read_rows(&mut file, start + (r * row_bytes) as u64, row_bytes, 1),
                        k,
                    );
                    acc += w[0] + w[k - 1];
                }
                std::hint::black_box(acc);
                samples.push(t0.elapsed().as_secs_f64());
            }
            samples.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
            let med = samples[samples.len() / 2];
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;

            // 純デ量子化 (I/O 無し) の単価: 1 行分のバイト列を 1 回だけ読んで使い回す
            let rowbuf = p38_read_rows(&mut file, start, row_bytes, 1);
            let pure_reps = 500usize;
            let t1 = std::time::Instant::now();
            for _ in 0..pure_reps {
                std::hint::black_box(dequant_row(t.ty, &rowbuf, k));
            }
            let pure_per_row = t1.elapsed().as_secs_f64() / pure_reps as f64;

            println!(
                "{:>3} {:>10} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>10.4}",
                b,
                reps,
                med * 1e6,
                med * 1e6 / b as f64,
                mean * 1e6,
                pure_per_row * 1e6,
                med * 1e3 / DECODE_MS_PER_TOKEN * 100.0
            );
            println!(
                "[p38]   B={b}: {:.1} us/token → decode 予算 {DECODE_MS_PER_TOKEN:.2} ms の {:.3} % \
                 (arithmetic against the KPI, not a GPU measurement)",
                med * 1e6,
                med * 1e3 / DECODE_MS_PER_TOKEN * 100.0
            );
        }

        // ---- 判定 ----
        assert!(verdicts.iter().any(|v| v.row == 0), "行 0 がサンプルに無い");
        assert!(
            verdicts.iter().any(|v| v.row == n - 1),
            "最終行 (行 {}) がサンプルに無い",
            n - 1
        );
        assert!(
            distinct_chunks >= 8,
            "サンプルが {} チャンクにしか跨っていない (散らし不足)",
            distinct_chunks
        );
        assert!(
            all_identical,
            "1 行 dequant がローダの f32 表の同じ行とビット一致しない → 置き換え不可"
        );
        if sweep_on {
            assert_eq!(
                swept_bad, 0,
                "全行スイープで {swept_bad} 行が不一致 (最初の行 {:?})",
                sweep_first_bad
            );
        }
        if let Some(false) = whole_ok {
            panic!("whole-tensor 表の参照とビット一致しない行がある");
        }
        println!("\n[p38] VERDICT: per-row dequant of token_embd.weight is BIT-IDENTICAL to the \
                  loader's f32 table row ({} sampled rows + {} swept rows)", verdicts.len(), swept_rows);
    }

    // =====================================================================
    // P42: 常駐 f32 表 (5,086 MB) を「ファイル自身の Q4_0 生ブロック +
    // 行オンデマンド復元」(Option B) へ置き換える経路の証明。
    //
    //  (1) 前提判定 (純関数) が想定外の形を必ず弾く
    //  (2) `embed_blocks_and_linear_chunked` の生ブロックがテンソルのバイト列と
    //      一致し、lm_head コピーが従来経路とバイト一致する (合成 GGUF)
    //  (3) 実 27B で `EmbedTable::Q4Rows` の行がローダの f32 表の同じ行と
    //      全行ビット一致する (#[ignore]・mmap しない・MemAvailable ガード)
    // =====================================================================

    /// 合成 Q4_0 ブロック列 (`n` 行 × `k` 要素・k は 32 の倍数)。
    /// スケールは f16 の実在値 (有限・負値 = 実ファイルと同じ符号)、
    /// ニブルは決定的な擬似乱数。
    fn p42_synth_blocks(k: usize, n: usize) -> (Vec<u8>, usize) {
        assert!(k % 32 == 0, "k={k} が 32 の倍数でない");
        let row_bytes = (k / 32) * 18;
        let mut blocks = vec![0u8; n * row_bytes];
        const SCALES: [u16; 6] = [0x9c00, 0xa400, 0x8e66, 0xb1c3, 0x9400, 0x9d55];
        let mut s = 0x1234_5678u32;
        let mut next = |s: &mut u32| -> u32 {
            *s ^= *s << 13;
            *s ^= *s >> 17;
            *s ^= *s << 5;
            *s
        };
        for r in 0..n {
            for b in 0..(k / 32) {
                let off = r * row_bytes + b * 18;
                let sc = SCALES[(next(&mut s) as usize) % SCALES.len()];
                blocks[off..off + 2].copy_from_slice(&sc.to_le_bytes());
                for j in 0..16 {
                    blocks[off + 2 + j] = (next(&mut s) & 0xff) as u8;
                }
            }
        }
        (blocks, row_bytes)
    }

    /// 合成 Q4_0 GGUF (1 テンソル・v3・alignment 32) を書く。
    /// ヘッダ配置は実ファイルと同じ規則 (データ部は 32 B 整列)。
    fn p42_write_gguf_q4(path: &std::path::Path, name: &str, k: usize, n: usize, blocks: &[u8]) {
        const ALIGN: usize = 32;
        fn u32b(v: &mut Vec<u8>, x: u32) {
            v.extend_from_slice(&x.to_le_bytes());
        }
        fn u64b(v: &mut Vec<u8>, x: u64) {
            v.extend_from_slice(&x.to_le_bytes());
        }
        fn strb(v: &mut Vec<u8>, s: &str) {
            u64b(v, s.len() as u64);
            v.extend_from_slice(s.as_bytes());
        }
        let mut b = Vec::new();
        u32b(&mut b, 0x4655_4747); // magic
        u32b(&mut b, 3); // version
        u64b(&mut b, 1); // tensor_count
        u64b(&mut b, 1); // kv_count
        strb(&mut b, "general.architecture");
        u32b(&mut b, 8); // GT_STRING
        strb(&mut b, "qwen3_5");
        strb(&mut b, name);
        u32b(&mut b, 2); // n_dims
        u64b(&mut b, k as u64);
        u64b(&mut b, n as u64);
        u32b(&mut b, 2); // GGML_TYPE Q4_0
        u64b(&mut b, 0); // offset
        while b.len() % ALIGN != 0 {
            b.push(0);
        }
        b.extend_from_slice(blocks);
        std::fs::write(path, b).unwrap();
    }

    /// P42: 前提判定の表 (陽性 1 + 陰性 5)。
    #[test]
    fn embed_blocks_eligible_rejects_unexpected_geometry() {
        let k = 128usize; // 4 ブロック/行
        let n = 7usize;
        let row_bytes = (k / 32) * 18;
        let nbytes = (n * row_bytes) as u64;
        // 陽性: Q4_0・k は 32 の倍数・nbytes == n*row_bytes
        assert!(embed_blocks_eligible(Some(GgmlType::Q4_0), k, n, nbytes));
        // 陰性 1: 型が Q4_0 でない
        assert!(!embed_blocks_eligible(Some(GgmlType::F16), k, n, nbytes));
        assert!(!embed_blocks_eligible(Some(GgmlType::Q8_0), k, n, nbytes));
        // 陰性 2: テンソルが無い
        assert!(!embed_blocks_eligible(None, k, n, nbytes));
        // 陰性 3: 1 行がブロック境界に載らない
        assert!(!embed_blocks_eligible(Some(GgmlType::Q4_0), k + 1, n, nbytes));
        // 陰性 4: nbytes が行数×行長でない (行が隙間なく並ばない)
        assert!(!embed_blocks_eligible(Some(GgmlType::Q4_0), k, n, nbytes - 1));
        assert!(!embed_blocks_eligible(
            Some(GgmlType::Q4_0),
            k,
            n,
            nbytes + row_bytes as u64
        ));
        // 陰性 5: 行数 0 / 行長 0
        assert!(!embed_blocks_eligible(Some(GgmlType::Q4_0), k, 0, 0));
        assert!(!embed_blocks_eligible(Some(GgmlType::Q4_0), 0, n, 0));
    }

    /// P42: 生ブロック経路が (a) テンソルのバイト列そのもの・(b) lm_head コピーが
    /// 従来経路とバイト一致・(c) 行復元が f32 表の同じ行とビット一致、を合成
    /// GGUF で確かめる。
    #[test]
    fn embed_blocks_and_linear_chunked_matches_f32_table_path() {
        use crate::model::qwen35::EmbedTable;
        let k = 128usize;
        let n = 7usize;
        let chunk_rows = 3usize; // 端数チャンクを作る (7 = 3+3+1)
        let (blocks, row_bytes) = p42_synth_blocks(k, n);
        assert_eq!(blocks.len(), n * row_bytes);

        let dir = std::env::temp_dir().join(format!("denno_p42_gguf_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.gguf");
        p42_write_gguf_q4(&path, "token_embd.weight", k, n, &blocks);

        let f = GgufFile::open(&path).expect("合成 GGUF を開けない");
        let t = f.tensor("token_embd.weight").expect("tensor").clone();
        assert_eq!(t.ty, GgmlType::Q4_0);
        assert_eq!(t.nbytes as usize, n * row_bytes);
        assert!(f.embed_blocks_eligible("token_embd.weight", k, n));

        // (a) 生ブロック == テンソルのバイト列
        let (bl, l_blocks, st) = f
            .embed_blocks_and_linear_chunked(
                "token_embd.weight",
                k,
                n,
                QuantMode::Q4_0,
                chunk_rows,
            )
            .expect("生ブロック経路がエラー")
            .expect("生ブロック経路が None");
        assert_eq!(bl.blocks.as_slice(), f.tensor_bytes("token_embd.weight").unwrap());
        assert_eq!(bl.blocks.as_slice(), &blocks[..]);
        assert_eq!((bl.k, bl.n, bl.row_bytes), (k, n, row_bytes));
        assert_eq!(bl.ty, GgmlType::Q4_0);
        assert_eq!(st.chunks, 3, "端数チャンクを含む 3 チャンク");

        // (b) lm_head コピーは従来 (f32 表) 経路とバイト一致
        let (table, l_table, _) = f
            .embed_and_linear_chunked("token_embd.weight", k, n, QuantMode::Q4_0, chunk_rows)
            .expect("f32 表経路がエラー")
            .expect("f32 表経路が None");
        match (&l_blocks, &l_table) {
            (Linear::Q4(a), Linear::Q4(b)) => {
                assert_eq!(a.blocks, b.blocks, "lm_head の Q4 コピーが従来経路と不一致");
            }
            _ => panic!("Q4_0 モードで Q4Linear が返らない"),
        }
        assert_eq!(l_blocks.weight_bytes(), l_table.weight_bytes());

        // (c) 行復元 (EmbedTable::Q4Rows) == f32 表の同じ行 (ビット一致)
        let emb = EmbedTable::Q4Rows {
            blocks: bl.blocks,
            k,
            n,
            row_bytes,
            ty: GgmlType::Q4_0 as u16,
        };
        let mut buf = vec![0.0f32; k];
        for r in 0..n {
            emb.row_into(r, &mut buf);
            let want = &table[r * k..(r + 1) * k];
            for (i, (a, b)) in buf.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {r} elem {i}: 生ブロック復元 {a} != f32 表 {b}"
                );
            }
            assert_eq!(emb.row(r, k).as_ref(), want, "row {r} (row()) が不一致");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// P42: 実 27B Q4_0 で、生ブロック常駐 + 行復元 (`EmbedTable::Q4Rows`) が
    /// ローダのチャンク dequant (= f32 表の作り方) と **全行ビット一致**する
    /// ことを確かめる。
    ///
    /// 実行:
    ///   GGUF_PATH=/work/models/Qwen3.8-27B-Q4_0.gguf \
    ///   CARGO_TARGET_DIR=/tmp/p42-target \
    ///   cargo test --release -p denno-engine --features cuda --lib -- --ignored \
    ///     --nocapture q4_embed_rows_ondemand_matches_loader_real_gguf
    ///
    /// 環境変数: `DENNO_P42_SWEEP=0` で全行スイープを省略 (既定は実行)。
    /// mmap しない (ヘッダ + pread のみ)。常駐は Option B と同じ 715.2 MB +
    /// 1 チャンク (16.8 MB)。
    #[test]
    #[ignore]
    fn q4_embed_rows_ondemand_matches_loader_real_gguf() {
        use crate::model::qwen35::EmbedTable;
        let path = match std::env::var("GGUF_PATH") {
            Ok(p) => std::path::PathBuf::from(p),
            Err(_) => {
                let d = std::path::PathBuf::from("/work/models/Qwen3.8-27B-Q4_0.gguf");
                if !d.exists() {
                    eprintln!("skip: GGUF_PATH 未設定かつ /work/models に既定モデルが無い");
                    return;
                }
                d
            }
        };
        // メモリ規律: 8 GiB 未満なら走らせない
        match mem_available_gib() {
            Some(g) if g < 8.0 => {
                panic!("MemAvailable {g:.1} GiB < 8 GiB — 走らせない (`free -g` を確認)")
            }
            Some(g) => println!("[p42] MemAvailable {g:.2} GiB"),
            None => println!("[p42] MemAvailable 不明 (/proc/meminfo 無し) — 続行"),
        }

        let mut file = std::fs::File::open(&path).expect("GGUF を開けない");
        let file_len = file.metadata().expect("metadata").len();
        let hdr = read_header(&mut file, file_len);
        let t = hdr
            .tensor("token_embd.weight")
            .expect("token_embd.weight が無い")
            .clone();
        let k = t.dims[0] as usize;
        let n: usize = t.dims[1..].iter().product::<u64>() as usize;
        assert_eq!(t.ty, GgmlType::Q4_0, "P42 v1 は Q4_0 のみ");
        assert!(
            embed_blocks_eligible(Some(t.ty), k, n, t.nbytes),
            "前提判定が false (k={k} n={n} nbytes={})",
            t.nbytes
        );
        let row_bytes = (k / 32) * 18;
        let (start, end) = hdr.tensor_range("token_embd.weight").expect("range");
        assert_eq!(end - start, t.nbytes);
        println!(
            "[p42] k={k} n={n} row_bytes={row_bytes} raw={:.1} MB f32 table={:.1} MB",
            t.nbytes as f64 / 1e6,
            (k * n * 4) as f64 / 1e6
        );

        // 生ブロックを pread で読む (全体 map しない)。読みは行境界に整列。
        reset_peak_rss();
        let base_rss = rss_now_bytes();
        let t_read = std::time::Instant::now();
        let mut blocks = vec![0u8; n * row_bytes];
        let rows_per_read = ((64usize << 20) / row_bytes).max(1);
        let mut r0 = 0usize;
        while r0 < n {
            let rows = (n - r0).min(rows_per_read);
            let buf = p38_read_rows(
                &mut file,
                start + (r0 * row_bytes) as u64,
                row_bytes,
                rows,
            );
            blocks[r0 * row_bytes..(r0 + rows) * row_bytes].copy_from_slice(&buf);
            r0 += rows;
        }
        println!(
            "[p42] raw blocks resident {:.1} MB (pread {:.2} s, peak RSS +{:.1} MB)",
            blocks.len() as f64 / 1e6,
            t_read.elapsed().as_secs_f64(),
            delta_mb(peak_rss_bytes(), base_rss)
        );
        let emb = EmbedTable::Q4Rows {
            blocks,
            k,
            n,
            row_bytes,
            ty: t.ty as u16,
        };
        // 参照側でも同じ生ブロックを読む (コピーを作らない)。
        let raw: &[u8] = match &emb {
            EmbedTable::Q4Rows { blocks, .. } => blocks,
            EmbedTable::F32(_) => unreachable!(),
        };

        // 参照: ローダ (gguf_rows_quantized_chunked_with_sink) と同じチャンクで
        // dequant した f32 表の同じ行。
        let chunk_rows = crate::quant::chunk_rows_for(k).clamp(1, n);
        let sweep_on = std::env::var("DENNO_P42_SWEEP").map(|v| v.trim() != "0").unwrap_or(true);
        let sample = p38_sample_rows(n, chunk_rows, 40);
        let t_cmp = std::time::Instant::now();
        let mut buf = vec![0.0f32; k];
        let mut checked = 0usize;
        let mut bad = 0usize;
        let mut first_bad: Option<(usize, usize, u32, u32)> = None;
        let mut c0 = 0usize;
        while c0 < n {
            let rows = (n - c0).min(chunk_rows);
            let w = dequant_row(
                t.ty,
                &raw[c0 * row_bytes..(c0 + rows) * row_bytes],
                rows * k,
            );
            let row_list: Vec<usize> = if sweep_on {
                (0..rows).collect()
            } else {
                sample
                    .iter()
                    .filter(|&&r| r >= c0 && r < c0 + rows)
                    .map(|&r| r - c0)
                    .collect()
            };
            for rr in row_list {
                emb.row_into(c0 + rr, &mut buf);
                let want = &w[rr * k..(rr + 1) * k];
                checked += 1;
                for (i, (a, b)) in buf.iter().zip(want.iter()).enumerate() {
                    if a.to_bits() != b.to_bits() {
                        bad += 1;
                        if first_bad.is_none() {
                            first_bad = Some((c0 + rr, i, a.to_bits(), b.to_bits()));
                        }
                        break;
                    }
                }
            }
            c0 += rows;
        }
        println!(
            "[p42] compared {checked} rows (sweep={sweep_on}, chunk_rows={chunk_rows}) in {:.2} s",
            t_cmp.elapsed().as_secs_f64()
        );
        assert_eq!(
            bad, 0,
            "行復元がローダの f32 表とビット不一致 ({bad} 行・最初 {first_bad:?})"
        );
        assert_eq!(checked, if sweep_on { n } else { sample.len() });

        // P42: 製品経路 (常駐ブロック + row_into・pread 無し) の 1 行コストを
        // CPU で実測する。GPU の KPI (ms/token) はここでは測らない。
        const BENCH_ROWS: usize = 20_000;
        let t_bench = std::time::Instant::now();
        let mut acc = 0.0f64;
        for i in 0..BENCH_ROWS {
            let r = (i * 7919) % n; // 行を散らす (キャッシュに載り切らない)
            emb.row_into(r, &mut buf);
            acc += buf[0] as f64;
        }
        let per_row_us = t_bench.elapsed().as_secs_f64() * 1e6 / BENCH_ROWS as f64;
        println!(
            "[p42] product-path row_into: {per_row_us:.3} us/row over {BENCH_ROWS} rows \
             (resident blocks, no pread; checksum {acc:.3})"
        );
        println!("[p42] VERDICT: EmbedTable::Q4Rows is BIT-IDENTICAL to the loader's f32 table ({checked} rows)");
    }
}
