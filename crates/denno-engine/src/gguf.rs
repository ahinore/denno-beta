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
    use crate::quant::{Q4Linear, Q4_BLOCK_BYTES, Q4_BLOCK_ELEMS};
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
        let hdr = {
            let mut want = 1usize << 20;
            loop {
                let n = want.min(file_len as usize);
                let mut buf = vec![0u8; n];
                file.seek(SeekFrom::Start(0)).expect("seek");
                file.read_exact(&mut buf).expect("read header");
                match GgufHeader::parse(&buf) {
                    Ok(h) => break h,
                    Err(e) => {
                        let short = e.0.contains("EOF") || e.0.contains("データ部がファイル終端");
                        if !short || n >= HEADER_CAP {
                            panic!("GGUF ヘッダ解析失敗 ({n} B 読み): {e}");
                        }
                        want *= 2;
                    }
                }
            }
        };
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
}
