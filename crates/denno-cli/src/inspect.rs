//! task 921: `denno inspect <gguf> [--json]` — GGUF の**ヘッダだけ**を読む。
//!
//! 読むのはメタデータ KV とテンソル情報 (名前・次元・型・オフセット) だけで、
//! テンソルのデータ部 (重み) は mmap も pread もしない。40 GB のモデルでも
//! 数 MiB の読み取りで終わる。
//!
//! 出すもの:
//! - アーキ情報 (architecture / 層数 / hidden / expert 数 / ctx / vocab)
//! - 量子型ヒストグラム (型ごとの本数と合計 MiB)
//! - VRAM 見積り (全段 GPU / dense + expert キャッシュ)
//!
//! 数値経路には触れない。推論の既定挙動は不変。

use anyhow::{bail, Context, Result};
use denno_engine::gguf::GgufHeader;
use denno_engine::gpu_budget::{classify_weights, n_full_layers};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// ヘッダ読みの初期プローブ長。大半の GGUF はこれで収まる。
const HEADER_PROBE_START: u64 = 1 << 20;
/// ヘッダ読みの上限。これを超えるヘッダは異常 (分割 GGUF を 1 本ずつ見る)。
const HEADER_PROBE_MAX: u64 = 64 << 20;
/// expert キャッシュの既定予算 (MiB)。`--expert-cache-mib` で変えられる。
pub const DEFAULT_EXPERT_CACHE_MIB: u64 = 4096;

const MIB: u64 = 1 << 20;

fn mib(bytes: u64) -> u64 {
    bytes.saturating_add(MIB / 2) / MIB
}

/// 量子型 1 種の集計。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuantEntry {
    /// ggml の型名 (`GgmlType::name`。ggml.h の enum 名)
    pub ty: String,
    pub tensors: usize,
    pub bytes: u64,
    pub mib: u64,
}

/// 量子型 1 種の内訳 (task 1096)。`quant` と同じ集計値に、全テンソル
/// バイトに対する割合を足したもの。並びは型名の昇順 (決定的)。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuantBreakdownEntry {
    /// ggml の型名 (`GgmlType::name`。`quant` と同じ綴り)
    pub ty: String,
    pub tensors: usize,
    pub bytes: u64,
    pub mib: u64,
    /// `total_tensor_bytes` に対する割合。ベーシスポイント (1/100 %、
    /// 10000 = 100%) の整数で持つ。浮動小数を使わないので JSON の値は
    /// 環境によらず同じになり、`Eq` も保てる。
    ///
    /// 型ごとに独立して四捨五入するため、合計は 10000 に一致しないことが
    /// ある (最大で型数の半分ぶんずれる)。
    pub pct_bp: u32,
}

/// VRAM / RAM の見積り。すべてバイト (mib は丸め)。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VramEstimate {
    /// dense (expert 以外) の重み。Q8_0 へ作り直される型は作り直し後の
    /// 行長で数える (`gpu_budget::classify_weights` と同じ式)。
    pub dense_bytes: u64,
    pub expert_bytes: u64,
    /// KV/状態キャッシュ。式は `gpu_budget::flash_budget` と同じ
    /// (full 層数 × ctx × kv_dim × 2 × f32)。
    pub kv_cache_bytes: u64,
    /// expert キャッシュに載せる予算 (MiB 指定、実際は expert_bytes で頭打ち)。
    pub expert_cache_mib: u64,
    pub expert_cache_bytes: u64,
    /// 全段を GPU に置いた場合 (dense + expert + kv)。
    pub full_gpu_bytes: u64,
    /// dense を GPU に置き、expert は予算ぶんだけ GPU に載せる場合。
    pub dense_plus_expert_cache_bytes: u64,
}

/// `denno inspect` の結果。`--json` はこれをそのまま出す。
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct InspectReport {
    pub path: String,
    pub gguf_version: u32,
    pub architecture: Option<String>,
    pub layer_count: Option<u64>,
    pub hidden_size: Option<u64>,
    pub expert_count: Option<u64>,
    pub context_length: Option<u64>,
    pub vocab_size: Option<u64>,
    pub tensor_count: usize,
    pub total_tensor_bytes: u64,
    pub quant: Vec<QuantEntry>,
    /// 型ごとの本数・バイト数・割合 (型名の昇順)。`quant` の後方互換な
    /// 追加分で、内容は `quant` と一致する (割合だけ足したもの)。
    pub quant_breakdown: Vec<QuantBreakdownEntry>,
    pub vram: VramEstimate,
}

/// ヘッダに必要な先頭バイトだけを読む。
///
/// `GgufHeader::parse` はバッファがヘッダ末尾に届いていないと EOF 系の
/// エラーを返すので、そのときだけプローブ長を伸ばして再試行する
/// (ファイル長と上限で頭打ち)。壊れたファイルはそのまま `Err` にする。
pub fn read_header(path: &Path) -> Result<GgufHeader> {
    let meta = std::fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    if meta.is_dir() {
        bail!("{} はディレクトリ (GGUF ファイルを指定する)", path.display());
    }
    let file_len = meta.len();
    let mut size = HEADER_PROBE_START.min(file_len.max(1));
    loop {
        let buf = read_prefix(path, size)?;
        match GgufHeader::parse(&buf) {
            Ok(h) => return Ok(h),
            Err(e) => {
                let truncated =
                    e.0.starts_with("EOF") || e.0.contains("ファイル終端を超えている");
                if truncated && size < file_len && size < HEADER_PROBE_MAX {
                    size = (size.saturating_mul(4)).min(HEADER_PROBE_MAX).min(file_len);
                    continue;
                }
                bail!("{}: {e}", path.display());
            }
        }
    }
}

fn read_prefix(path: &Path, n: u64) -> Result<Vec<u8>> {
    let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut buf = vec![0u8; n as usize];
    let mut got = 0usize;
    while got < buf.len() {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                return Err(e).with_context(|| format!("read {}", path.display()));
            }
        }
    }
    buf.truncate(got);
    Ok(buf)
}

fn meta_u(h: &GgufHeader, key: &str) -> Option<u64> {
    h.get(key).and_then(|v| v.as_u64())
}

/// ヘッダ + パスからレポートを組む純関数 (ファイル I/O 無し)。
pub fn analyze(header: &GgufHeader, path: &Path, expert_cache_mib: u64) -> InspectReport {
    let architecture = header
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let arch_u = |suffix: &str| -> Option<u64> {
        let arch = architecture.as_deref()?;
        meta_u(header, &format!("{arch}.{suffix}"))
    };

    let layer_count = arch_u("block_count");
    let hidden_size = arch_u("embedding_length");
    let expert_count = arch_u("expert_count");
    let context_length = arch_u("context_length");
    let vocab_size = header
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array_len())
        .map(|n| n as u64)
        .or_else(|| arch_u("vocab_size"));

    // 量子型ヒストグラム (tensor_order 順に足すので決定的)。
    let mut hist: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
    let mut total_tensor_bytes = 0u64;
    for name in &header.tensor_order {
        let Some(t) = header.tensors.get(name) else {
            continue;
        };
        total_tensor_bytes = total_tensor_bytes.saturating_add(t.nbytes);
        let e = hist.entry(t.ty.name()).or_insert((0, 0));
        e.0 += 1;
        e.1 = e.1.saturating_add(t.nbytes);
    }
    let quant: Vec<QuantEntry> = hist
        .into_iter()
        .map(|(ty, (tensors, bytes))| QuantEntry {
            ty: ty.to_string(),
            tensors,
            bytes,
            mib: mib(bytes),
        })
        .collect();

    // task 1096: 型ごとの内訳。`quant` と同じ値に割合を足し、型名の昇順に
    // 並べ替える (BTreeMap の反復順に暗黙に頼らない = 決定的であることを
    // この関数単体で保証する)。
    let mut quant_breakdown: Vec<QuantBreakdownEntry> = quant
        .iter()
        .map(|q| QuantBreakdownEntry {
            ty: q.ty.clone(),
            tensors: q.tensors,
            bytes: q.bytes,
            mib: q.mib,
            pct_bp: pct_bp(q.bytes, total_tensor_bytes),
        })
        .collect();
    quant_breakdown.sort_by(|a, b| a.ty.cmp(&b.ty));

    // dense / expert の分離は engine と同じ式を使う。
    let totals = classify_weights(header.tensors.values().map(|t| {
        (t.name.as_str(), t.ty, t.dims, t.nbytes)
    }));
    let dense_bytes = totals.dev_bytes.saturating_add(totals.host_bytes);
    let expert_bytes = totals.expert_bytes;

    // KV は flash_budget と同じ式 (full 層数 × ctx × kv_dim × 2 × 4 B)。
    let kv_cache_bytes = match (
        layer_count,
        context_length,
        arch_u("full_attention_interval"),
        arch_u("attention.head_count_kv"),
        arch_u("attention.key_length"),
    ) {
        (Some(layers), Some(ctx), Some(interval), Some(n_kv), Some(head_k)) if interval > 0 => {
            let kv_dim = n_kv.saturating_mul(head_k);
            (n_full_layers(layers as usize, interval as usize) as u64)
                .saturating_mul(ctx)
                .saturating_mul(kv_dim)
                .saturating_mul(2)
                .saturating_mul(std::mem::size_of::<f32>() as u64)
        }
        _ => 0,
    };

    let expert_cache_bytes = expert_bytes.min(expert_cache_mib.saturating_mul(MIB));
    let base = dense_bytes.saturating_add(kv_cache_bytes);
    let vram = VramEstimate {
        dense_bytes,
        expert_bytes,
        kv_cache_bytes,
        expert_cache_mib,
        expert_cache_bytes,
        full_gpu_bytes: base.saturating_add(expert_bytes),
        dense_plus_expert_cache_bytes: base.saturating_add(expert_cache_bytes),
    };

    InspectReport {
        path: path.display().to_string(),
        gguf_version: header.version,
        architecture,
        layer_count,
        hidden_size,
        expert_count,
        context_length,
        vocab_size,
        tensor_count: header.tensors.len(),
        total_tensor_bytes,
        quant,
        quant_breakdown,
        vram,
    }
}

/// `bytes` が `total` に占める割合をベーシスポイント (1/100 % = 1 bp、
/// 10000 = 100%) で返す。`total == 0` なら 0。四捨五入は整数演算だけで行う
/// (浮動小数を持ち込まない)。
fn pct_bp(bytes: u64, total: u64) -> u32 {
    if total == 0 {
        return 0;
    }
    // bytes * 10000 は最大でも 4e14 程度 (40 GB でも 4e14) で u64 に収まるが、
    // saturating で念のため頭打ちにする。
    let scaled = bytes.saturating_mul(10_000).saturating_add(total / 2);
    (scaled / total).min(10_000) as u32
}

/// ベーシスポイントを `12.34` 形式の文字列にする (テキスト出力用)。
fn pct_string(bp: u32) -> String {
    format!("{}.{:02}", bp / 100, bp % 100)
}

fn show_u(v: Option<u64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "?".to_string())
}

fn show_s(v: Option<&str>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "?".to_string())
}

/// 人が読む形 (1 行 1 項目、`?` はヘッダに無い)。
pub fn render_text(r: &InspectReport) -> String {
    let mut s = String::new();
    s.push_str(&format!("path: {}\n", r.path));
    s.push_str(&format!("gguf version: {}\n", r.gguf_version));
    s.push_str(&format!("architecture: {}\n", show_s(r.architecture.as_deref())));
    s.push_str(&format!("layers: {}\n", show_u(r.layer_count)));
    s.push_str(&format!("hidden size: {}\n", show_u(r.hidden_size)));
    s.push_str(&format!("experts: {}\n", show_u(r.expert_count)));
    s.push_str(&format!("context length: {}\n", show_u(r.context_length)));
    s.push_str(&format!("vocab size: {}\n", show_u(r.vocab_size)));
    s.push_str(&format!(
        "tensors: {} ({} MiB)\n",
        r.tensor_count,
        mib(r.total_tensor_bytes)
    ));

    s.push_str("quantization:\n");
    for q in &r.quant {
        s.push_str(&format!(
            "  {:<10} tensors={:<8} {:>12} MiB\n",
            q.ty, q.tensors, q.mib
        ));
    }

    // task 1096: 型ごとの内訳 (割合つき、型名の昇順)。
    s.push_str("quant breakdown:\n");
    for b in &r.quant_breakdown {
        s.push_str(&format!(
            "  {:<10} tensors={:<8} {:>12} MiB {:>7}%\n",
            b.ty,
            b.tensors,
            b.mib,
            pct_string(b.pct_bp)
        ));
    }
    let total_pct = if r.total_tensor_bytes == 0 { 0 } else { 10_000 };
    s.push_str(&format!(
        "  {:<10} tensors={:<8} {:>12} MiB {:>7}%\n",
        "total",
        r.tensor_count,
        mib(r.total_tensor_bytes),
        pct_string(total_pct)
    ));

    let v = &r.vram;
    let row = |label: String, bytes: u64| format!("  {label:<26}{:>10} MiB\n", mib(bytes));
    s.push_str("vram estimate:\n");
    s.push_str(&row("kv cache".to_string(), v.kv_cache_bytes));
    s.push_str(&row("dense".to_string(), v.dense_bytes));
    s.push_str(&row("experts (all)".to_string(), v.expert_bytes));
    s.push_str(&row("full gpu".to_string(), v.full_gpu_bytes));
    s.push_str(&row(
        format!("expert cache ({} MiB)", v.expert_cache_mib),
        v.expert_cache_bytes,
    ));
    s.push_str(&row(
        "dense + expert cache".to_string(),
        v.dense_plus_expert_cache_bytes,
    ));
    s
}

/// `denno inspect` の本体。`json` なら 1 個の JSON オブジェクトだけを stdout に出す。
pub fn run(path: &Path, json: bool, expert_cache_mib: u64) -> Result<()> {
    let header = read_header(path)?;
    let report = analyze(&header, path, expert_cache_mib);
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print!("{}", render_text(&report));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mib_rounds_to_nearest() {
        assert_eq!(mib(0), 0);
        assert_eq!(mib(MIB - 1), 1);
        assert_eq!(mib(MIB / 2), 1);
        assert_eq!(mib(MIB * 3 + MIB / 2 + 1), 4);
    }

    #[test]
    fn show_helpers_mark_unknown_as_question() {
        assert_eq!(show_u(None), "?");
        assert_eq!(show_u(Some(0)), "0");
        assert_eq!(show_s(None), "?");
    }
}
