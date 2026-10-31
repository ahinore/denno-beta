//! P-MOE-C1: PLE (n-gram 埋め込み表) の SSD 常駐行 gather。
//!
//! 出典: n-gram 索引の数式 (トークン列からのハッシュと行番号の決め方) は
//! llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! 行のデコードは本リポジトリの `kquant.rs` (同じく MIT 由来) を再利用している。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! Flash-Next の `per_layer_token_embd.weight` は IQ4_NL で 28.80 GB
//! (`[160, 320001536]` = 3.2 億行 × 160 次元)。VRAM にも RAM にも常駐させず、
//! 必要な行だけを要求時に読んで f32 へ展開する。
//!
//! # 読み出し方式: `File` + `pread` (`read_exact_at`)
//!
//! mmap ではなく pread を選んだ理由 (実測に基づく):
//!
//! - mmap の行読みは **ページ単位の fault-around** で隣接ページまで写像する。
//!   実測 (6.17 カーネル, 275 MB のファイル, 90 B の行をランダムに 8,000 回):
//!   `MADV_RANDOM` を掛けても RSS が 1 行あたり 7.4 ページ分 (約 30 KB) 増え、
//!   8,000 行で RSS が 235 MB まで膨らんだ (表全体 274 MB の 86%)。
//!   28.8 GB の表で同じことが起きると RSS が表サイズまで積み上がる。
//! - pread は読み先が自前のバッファ (最大 64 KiB) だけなので、RSS はバッファ分で
//!   一定。データはページキャッシュに載るが、これはカーネルが回収できる。
//! - `GgufFile` はパスを保持していなかったため、A1 の `TensorRef::file`
//!   (split のパート番号) から fd を開けるように [`GgufFile::path_at`] を足した。
//!
//! # ビット一致
//!
//! 行長 160 は IQ4_NL のブロック長 32 の倍数なので、行境界 = ブロック境界。
//! 1 行のバイト列を [`crate::kquant::dequant_row_into`] (中身は
//! `dequant_block` をブロック順に呼ぶだけ) に渡すと、行ごとの丸め順序は
//! 表全体を 1 回でデコードした場合と同一になる。式は書き起こしていない。
//!
//! # P-MOE-E2: n-gram 索引
//!
//! 表引きに渡す行索引は [`PleNgram`] が計算する (参照実装
//! `qwen4exp.cpp` の `llm_graph_input_qwen4exp_ple::set_input` と同じ規則)。
//! 索引が 1 つでもずれると 28.80 GB の表から別の行を読むので、
//! 剰余は u64 で取り、加算の後で i32 に落とす。

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::OnceLock;

use crate::gguf::{GgufFile, GgufValue, TensorRef};
use crate::kquant::{self, GgmlType};

/// 1 回のまとめ読みの上限 (要件: 64 KiB を超えない)。
pub const DEFAULT_MAX_READ: usize = 64 * 1024;

/// PLE 表の読み出しエラー。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PleError(pub String);

impl std::fmt::Display for PleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PleError {}

type Result<T> = std::result::Result<T, PleError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(PleError(msg.into()))
}

/// SSD 常駐の PLE 表 (行 gather 専用)。
///
/// 表そのものは保持せず、パートの fd とテンソルの絶対オフセットだけを持つ。
/// 1 行のデコードに使う作業バッファは `gather_rows` の呼び出しごとに確保する
/// (最大 [`DEFAULT_MAX_READ`] バイト) ので、表のサイズに比例する常駐は無い。
pub struct PleTable {
    /// テンソルが属するパートの読み取り専用ハンドル
    file: File,
    /// このパートのファイル内のテンソル先頭オフセット
    base: u64,
    n_rows: usize,
    n_dim: usize,
    /// 1 行のバイト数 ((n_dim / blck_size) * type_size)
    row_bytes: usize,
    ty: GgmlType,
    /// 1 回のまとめ読みの上限
    max_read: usize,
    /// 開いたパートのパス (診断用)
    path: PathBuf,
}

impl std::fmt::Debug for PleTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PleTable")
            .field("path", &self.path)
            .field("n_rows", &self.n_rows)
            .field("n_dim", &self.n_dim)
            .field("row_bytes", &self.row_bytes)
            .field("ty", &self.ty)
            .field("base", &self.base)
            .field("max_read", &self.max_read)
            .finish()
    }
}

impl PleTable {
    /// GGUF のテンソル参照から開く。
    ///
    /// 検証する内容:
    /// - 量子型が IQ4_NL (この表の型) であること
    /// - 次元が `[n_dim, n_rows]` の 2 次元で、`n_dim` がブロック長の倍数であること
    /// - `n_rows * row_bytes` がテンソルの実バイト数と一致すること
    /// - 渡された [`TensorRef`] のメタデータ (dims / 型 / offset / nbytes / パート) が
    ///   `gguf` 側のテンソル情報と一致すること
    ///   (古い参照や別ファイルの参照を掴んだまま開こうとしたらエラー)
    ///
    /// ここでは 1 バイトも読まない (fd を開いてオフセットを決めるだけ)。
    pub fn open(tensor: &TensorRef, gguf: &GgufFile) -> Result<Self> {
        let name = tensor.name();
        if tensor.ty() != GgmlType::IQ4Nl {
            return err(format!(
                "{name}: 未対応の量子型 {:?} (PLE 表は IQ4_NL のみ)",
                tensor.ty()
            ));
        }

        // 1. 渡された参照そのものの形状検査。
        let ne = tensor.ne();
        if ne.len() != 2 {
            return err(format!(
                "{name}: 次元が 2 でない (ne={ne:?})。[n_dim, n_rows] を期待"
            ));
        }
        let n_dim = usize::try_from(ne[0])
            .map_err(|_| PleError(format!("{name}: n_dim {} が usize に収まらない", ne[0])))?;
        let n_rows = usize::try_from(ne[1])
            .map_err(|_| PleError(format!("{name}: n_rows {} が usize に収まらない", ne[1])))?;
        if n_dim == 0 || n_rows == 0 {
            return err(format!("{name}: 次元に 0 がある (n_dim={n_dim}, n_rows={n_rows})"));
        }
        let bs = GgmlType::IQ4Nl.blck_size();
        if n_dim % bs != 0 {
            return err(format!(
                "{name}: n_dim {n_dim} がブロック長 {bs} の倍数でない (行境界がブロック境界に乗らない)"
            ));
        }
        let row_bytes = (n_dim / bs) * GgmlType::IQ4Nl.type_size();
        let total = (n_rows as u64)
            .checked_mul(row_bytes as u64)
            .ok_or_else(|| PleError(format!("{name}: n_rows * row_bytes がオーバーフロー")))?;
        if total != tensor.nbytes() {
            return err(format!(
                "{name}: n_rows({n_rows}) * row_bytes({row_bytes}) = {total} が \
                 テンソルの実サイズ {} と一致しない",
                tensor.nbytes()
            ));
        }

        // 2. gguf 側のテンソル情報との突き合わせ。
        let info = match gguf.tensor(name) {
            Some(i) => i,
            None => return err(format!("{name}: gguf にテンソルが無い")),
        };
        if info.dims != tensor.info.dims
            || info.ty != tensor.info.ty
            || info.offset != tensor.info.offset
            || info.nbytes != tensor.info.nbytes
            || info.file != tensor.info.file
            || tensor.file != info.file
        {
            return err(format!(
                "{name}: テンソル参照が gguf のメタデータと一致しない \
                 (参照: file={} dims={:?} ty={:?} off={} nbytes={} / \
                 gguf: file={} dims={:?} ty={:?} off={} nbytes={})",
                tensor.file,
                tensor.info.dims,
                tensor.info.ty,
                tensor.info.offset,
                tensor.info.nbytes,
                info.file,
                info.dims,
                info.ty,
                info.offset,
                info.nbytes
            ));
        }

        // 3. ファイル内の絶対範囲。テンソルがパート終端を超えていれば None。
        let (file, off, len) = match gguf.tensor_mapping_parts(name) {
            Some(m) => m,
            None => return err(format!("{name}: データ範囲がファイル終端を超過している")),
        };
        if file != tensor.file {
            return err(format!(
                "{name}: 参照のパート {} と gguf のパート {file} が一致しない",
                tensor.file
            ));
        }
        if len as u64 != total {
            return err(format!(
                "{name}: ファイル上の長さ {len} が計算値 {total} と一致しない"
            ));
        }
        let path = match gguf.path_at(file) {
            Some(p) => p.to_path_buf(),
            None => return err(format!("{name}: パート {file} のパスが無い")),
        };
        let fh = match File::open(&path) {
            Ok(f) => f,
            Err(e) => return err(format!("{name}: {} を開けない: {e}", path.display())),
        };
        let file_len = match fh.metadata() {
            Ok(m) => m.len(),
            Err(e) => return err(format!("{name}: {} の stat に失敗: {e}", path.display())),
        };
        if off as u64 + total > file_len {
            return err(format!(
                "{name}: 範囲 {}..{} がファイル長 {file_len} を超える",
                off,
                off as u64 + total
            ));
        }

        Ok(PleTable {
            file: fh,
            base: off as u64,
            n_rows,
            n_dim,
            row_bytes,
            ty: GgmlType::IQ4Nl,
            max_read: DEFAULT_MAX_READ,
            path,
        })
    }

    /// 行数 (320001536)
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// 1 行の次元 (160)
    pub fn n_dim(&self) -> usize {
        self.n_dim
    }

    /// 1 行のバイト数 (90)
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// 量子型 (常に IQ4_NL)
    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// 表の総バイト数
    pub fn nbytes(&self) -> u64 {
        self.n_rows as u64 * self.row_bytes as u64
    }

    /// 開いたパートのパス (診断用)
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// 1 回のまとめ読みの上限 (既定 64 KiB)。
    pub fn max_read(&self) -> usize {
        self.max_read
    }

    /// まとめ読みの上限を変える。64 KiB 超は要件違反なので受け付けない。
    pub fn set_max_read(&mut self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return err("max_read に 0 は指定できない");
        }
        if bytes > DEFAULT_MAX_READ {
            return err(format!("max_read {bytes} が上限 {DEFAULT_MAX_READ} を超える"));
        }
        self.max_read = bytes;
        Ok(())
    }

    /// 行 gather。`rows` の各 index について 160 次元を f32 に展開して `out` に書く。
    ///
    /// - `out.len() == rows.len() * n_dim` でなければエラー
    /// - 範囲外 index はエラー (黙って 0 にしない)
    /// - 出力順は `rows` の順 (重複 index はそのまま重複して書く)
    /// - 連続する index は 1 回の pread にまとめ、1 回の読みは `max_read` 以下
    ///
    /// 読み出しは `pread` (`read_exact_at`) なので fd の位置を共有しない。
    /// 表の常駐バッファは持たない (作業バッファは呼び出しごとに最大 `max_read`)。
    pub fn gather_rows(&self, rows: &[u32], out: &mut [f32]) -> Result<()> {
        let want = rows
            .len()
            .checked_mul(self.n_dim)
            .ok_or_else(|| PleError("rows.len() * n_dim がオーバーフロー".into()))?;
        if out.len() != want {
            return err(format!(
                "out の長さ {} が rows.len()({}) * n_dim({}) = {want} と一致しない",
                out.len(),
                rows.len(),
                self.n_dim
            ));
        }
        for &r in rows {
            if r as usize >= self.n_rows {
                return err(format!("行 index {r} が範囲外 (n_rows={})", self.n_rows));
            }
        }

        // 連続 index のまとめ読み。1 回の pread は max_read バイト以下に切る。
        let max_rows_per_read = (self.max_read / self.row_bytes).max(1);
        // P78: `DENNO_PRE_KPROF` のときだけ pread の回数・行数・バイト数を積む
        // (off のときはここで atomic load 1 回、ループの中は何も増えない)。
        let prof = crate::model::prekprof::on();
        let mut buf: Vec<u8> = Vec::new();
        let mut i = 0usize;
        while i < rows.len() {
            let start_row = rows[i] as usize;
            let mut j = i + 1;
            while j < rows.len() {
                let cur = rows[j] as usize;
                if cur != rows[j - 1] as usize + 1 {
                    break;
                }
                if cur - start_row + 1 > max_rows_per_read {
                    break;
                }
                j += 1;
            }
            let run_rows = j - i;
            let span = run_rows * self.row_bytes;
            buf.clear();
            buf.resize(span, 0);
            let off = self.base + (start_row * self.row_bytes) as u64;
            if let Err(e) = self.file.read_exact_at(&mut buf, off) {
                return err(format!(
                    "pread 失敗 (行 {start_row} から {run_rows} 行, off={off}, len={span}): {e}"
                ));
            }
            if prof {
                crate::model::prekprof::note_ple_pread(run_rows, span);
            }
            for (k, &r) in rows[i..j].iter().enumerate() {
                let rb = (r as usize - start_row) * self.row_bytes;
                let o = &mut out[(i + k) * self.n_dim..(i + k + 1) * self.n_dim];
                // 行境界 = ブロック境界なので、行単位のデコードは表全体の
                // デコードと丸め順序が変わらない (`dequant_block` を再利用)。
                kquant::dequant_row_into(self.ty, &buf[rb..rb + self.row_bytes], o);
            }
            i = j;
        }
        Ok(())
    }

    /// [`PleTable::gather_rows`] を `parts` 本に割って並列に読む (P78)。
    ///
    /// 行は独立 (1 行のデコードはその行のバイト列だけで閉じる) なので、分割して
    /// も値は [`PleTable::gather_rows`] とビット一致する。90 B のランダム読みが
    /// デバイスのキューに複数同時に乗るぶん、レイテンシ律速のときに効く。
    /// エラーの**発生順**は並列度に依存する (どの行で落ちたかは同じ)。
    ///
    /// `parts <= 1` または行が 1 本以下なら [`PleTable::gather_rows`] に落ちる。
    pub fn gather_rows_par(&self, rows: &[u32], out: &mut [f32], parts: usize) -> Result<()> {
        let parts = parts.min(rows.len());
        if parts <= 1 {
            return self.gather_rows(rows, out);
        }
        let per = rows.len().div_ceil(parts);
        use rayon::prelude::*;
        let results: Vec<Result<()>> = out
            .par_chunks_mut(per * self.n_dim)
            .zip(rows.par_chunks(per))
            .map(|(o, r)| self.gather_rows(r, o))
            .collect();
        for r in results {
            r?;
        }
        Ok(())
    }
}

/// PLE 表のテンソル名 (行索引が指す先)。
pub const PLE_TABLE_NAME: &str = "per_layer_token_embd.weight";

/// まとめ引き (`PleNgram::gather_many`) の並列度 (`DENNO_PLE_PAR`)。
///
/// 既定 8。`0` / `1` で逐次 (1 行 1 pread を順に出す)。行数が
/// [`PLE_PAR_MIN_ROWS`] 未満のときは並列にしない (起動のほうが高い)。
fn ple_par_parts(rows: usize) -> usize {
    /// これ未満は並列にしない (1 回のまとめ引きの行数)。
    const PLE_PAR_MIN_ROWS: usize = 64;
    if rows < PLE_PAR_MIN_ROWS {
        return 1;
    }
    static N: OnceLock<usize> = OnceLock::new();
    let n = *N.get_or_init(|| {
        std::env::var("DENNO_PLE_PAR")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .map(|n| n.min(16))
            .unwrap_or(8)
    });
    n.max(1)
}

/// n-gram 索引の異常値を弾くための上限。
///
/// 実ファイル (Qwen3.5-27B Flash-Next 系) は `n_gram = 3`, `n_heads = 16`。
/// llama.cpp 側の `LLAMA_MAX_PLE_NGRAM` / `LLAMA_MAX_PLE_HEADS` は参照ツリーに
/// 無いため、ここでは「明らかに壊れた値で巨大な Vec を確保しない」ための
/// 独立した上限として置いている (参照実装の規則そのものではない)。
const MAX_N_GRAM: usize = 32;
const MAX_N_HEADS: usize = 4096;

/// PLE の n-gram 索引計算 (参照実装 `qwen4exp.cpp` の `build_ple_indices` 相当)。
///
/// メタデータだけを保持する。表そのものは [`PleTable`] が持つ。
#[derive(Debug, Clone)]
pub struct PleNgram {
    /// n-gram 長 (実ファイルは 3)
    n_gram: usize,
    /// 1 つの n-gram 長あたりのヘッド数 (実ファイルは 8)
    per_gram: usize,
    /// 文脈のカットに使う EOS トークン (実ファイルは 248044)
    eos: u32,
    /// 層ごとの乗数 (n_gram 個)
    mult: Vec<u64>,
    /// ヘッドごとの行オフセット (n_heads 個)
    head_offsets: Vec<u32>,
    /// ヘッドごとの語彙サイズ (n_heads 個)
    head_vocab_sizes: Vec<u32>,
}

/// メタデータの整数 1 個。
fn meta_u32(f: &GgufFile, key: &str) -> Result<u32> {
    let v = match f.get(key) {
        Some(v) => v,
        None => return err(format!("{key}: メタデータに無い")),
    };
    let x = match v.as_u64() {
        Some(x) => x,
        None => return err(format!("{key}: 整数でない ({v:?})")),
    };
    u32::try_from(x).map_err(|_| PleError(format!("{key}: {x} が u32 に収まらない")))
}

/// メタデータの整数配列 (u64 として読む)。
fn meta_u64_arr(f: &GgufFile, key: &str) -> Result<Vec<u64>> {
    let v = match f.get(key) {
        Some(v) => v,
        None => return err(format!("{key}: メタデータに無い")),
    };
    let items = match v {
        GgufValue::Arr(a) => a,
        _ => return err(format!("{key}: 配列でない ({v:?})")),
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, e) in items.iter().enumerate() {
        match e.as_u64() {
            Some(x) => out.push(x),
            None => return err(format!("{key}[{i}]: 整数でない ({e:?})")),
        }
    }
    Ok(out)
}

impl PleNgram {
    /// GGUF のメタデータから n-gram 索引のパラメータを読む。
    ///
    /// キーの前置きは `general.architecture` の値 (実ファイルは `qwen4exp`)。
    /// 検証する内容:
    /// - 必須キーが存在し、型が合っていること
    /// - `layer_multipliers` が `n_gram` 個、`head_offsets` / `head_vocab_sizes` が
    ///   `n_heads = (n_gram - 1) * heads_per_ngram` 個であること
    /// - 各ヘッドで `vocab != 0`、`offset + vocab` が i32 に収まること
    ///   (行 index は i32 で gather に渡る)
    /// - 表テンソルが同じ GGUF にあれば、`offset + vocab` がその行数以下で、
    ///   次元が `embedding_length_per_layer_input` と一致すること
    ///   (参照実装も同じ突き合わせをしている)
    pub fn from_meta(f: &GgufFile) -> Result<Self> {
        let arch = match f.get("general.architecture").and_then(|v| v.as_str()) {
            Some(a) => a.to_string(),
            None => return err("general.architecture: メタデータに無い"),
        };
        let key = |s: &str| format!("{arch}.{s}");

        // PLE 層は 1 つだけ (参照実装も複数層を弾く)。
        let layers = meta_u64_arr(f, &key("ple.layers"))?;
        if layers.len() != 1 {
            return err(format!(
                "{}.ple.layers: {} 個あるが、対応するのは 1 個だけ",
                arch,
                layers.len()
            ));
        }

        let n_gram = meta_u32(f, &key("ple.ngram_size"))? as usize;
        let per_gram = meta_u32(f, &key("ple.heads_per_ngram"))? as usize;
        let eos = meta_u32(f, &key("ple.eos_token_id"))?;
        let n_dim = meta_u32(f, &key("embedding_length_per_layer_input"))? as usize;

        if n_gram < 2 {
            return err(format!("{arch}.ple.ngram_size: {n_gram} は 2 未満"));
        }
        if n_gram > MAX_N_GRAM {
            return err(format!(
                "{arch}.ple.ngram_size: {n_gram} は上限 {MAX_N_GRAM} を超える"
            ));
        }
        if per_gram == 0 {
            return err(format!("{arch}.ple.heads_per_ngram: 0 は不可"));
        }
        let n_heads = (n_gram - 1)
            .checked_mul(per_gram)
            .ok_or_else(|| PleError("n_heads がオーバーフロー".into()))?;
        if n_heads == 0 || n_heads > MAX_N_HEADS {
            return err(format!(
                "{arch}.ple: n_heads = {n_heads} が範囲 [1, {MAX_N_HEADS}] の外"
            ));
        }
        if n_dim == 0 {
            return err(format!("{arch}.embedding_length_per_layer_input: 0 は不可"));
        }

        let mult = meta_u64_arr(f, &key("ple.layer_multipliers"))?;
        let offsets = meta_u64_arr(f, &key("ple.head_offsets"))?;
        let vocabs = meta_u64_arr(f, &key("ple.head_vocab_sizes"))?;
        if mult.len() != n_gram {
            return err(format!(
                "{arch}.ple.layer_multipliers: {} 個 (n_gram = {n_gram} 個であるべき)",
                mult.len()
            ));
        }
        if offsets.len() != n_heads {
            return err(format!(
                "{arch}.ple.head_offsets: {} 個 (n_heads = {n_heads} 個であるべき)",
                offsets.len()
            ));
        }
        if vocabs.len() != n_heads {
            return err(format!(
                "{arch}.ple.head_vocab_sizes: {} 個 (n_heads = {n_heads} 個であるべき)",
                vocabs.len()
            ));
        }

        // ヘッド範囲 → u32 (i32 の行 index に収まること)。
        let mut head_offsets = Vec::with_capacity(n_heads);
        let mut head_vocab_sizes = Vec::with_capacity(n_heads);
        let mut max_row = 0u64;
        for h in 0..n_heads {
            let (o, v) = (offsets[h], vocabs[h]);
            if v == 0 {
                return err(format!("{arch}.ple.head_vocab_sizes[{h}]: 0 は不可"));
            }
            let end = o
                .checked_add(v)
                .ok_or_else(|| PleError(format!("{arch}.ple: ヘッド {h} の offset + vocab がオーバーフロー")))?;
            if end > i32::MAX as u64 {
                return err(format!(
                    "{arch}.ple: ヘッド {h} の範囲 {o}..{end} が i32 の行 index に収まらない"
                ));
            }
            max_row = max_row.max(end);
            head_offsets.push(o as u32);
            head_vocab_sizes.push(v as u32);
        }

        // 表テンソルがあれば行数・次元を突き合わせる (合成 GGUF には無いことがある)。
        if let Some(t) = f.tensor(PLE_TABLE_NAME) {
            let rows = t.dims[1];
            if rows < max_row {
                return err(format!(
                    "{PLE_TABLE_NAME}: 行数 {rows} がヘッド範囲の最大 {max_row} より少ない"
                ));
            }
            if t.dims[0] != n_dim as u64 {
                return err(format!(
                    "{PLE_TABLE_NAME}: 次元 {} が embedding_length_per_layer_input {n_dim} と一致しない",
                    t.dims[0]
                ));
            }
        }

        Ok(PleNgram {
            n_gram,
            per_gram,
            eos,
            mult,
            head_offsets,
            head_vocab_sizes,
        })
    }

    /// ヘッド数 (n_gram - 1) * per_gram (実ファイルは 16)。
    pub fn n_heads(&self) -> usize {
        (self.n_gram - 1) * self.per_gram
    }

    /// n-gram 長 (実ファイルは 3)。
    pub fn n_gram(&self) -> usize {
        self.n_gram
    }

    /// 文脈カットに使う EOS トークン (実ファイルは 248044)。
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// `ctx` から行索引を計算する。
    ///
    /// `ctx` は **新しい順** (`ctx[0]` = 現在のトークン)。長さが `n_gram` 未満なら
    /// 足りない古い側は EOS として扱う (参照実装では前トークンが
    /// `LLAMA_TOKEN_NULL` になる場合に相当)。
    ///
    /// - `ctx[0]` は自分自身なのでカットしない (EOS でもカットしない)。
    /// - `ctx[s]` (s >= 1) が EOS、または履歴に無い場合は s 以降をすべて EOS にする。
    /// - `n = 2 ..= n_gram` のそれぞれで `mixed` を計算し直す (累積しない)。
    /// - 戻り値の並びは `n = 2` の `per_gram` 個 → `n = 3` の `per_gram` 個 …
    pub fn indices(&self, ctx: &[u32]) -> Result<Vec<i32>> {
        if ctx.is_empty() {
            return err("ctx が空 (現在のトークンが無い)");
        }
        // 実効文脈 (古い側を EOS で埋め、カットを反映したもの)。
        let mut c = vec![self.eos; self.n_gram];
        c[0] = ctx[0];
        let mut cut = false;
        for s in 1..self.n_gram {
            let t = if cut {
                self.eos
            } else {
                ctx.get(s).copied().unwrap_or(self.eos)
            };
            cut = cut || t == self.eos;
            c[s] = if cut { self.eos } else { t };
        }

        let n_heads = self.n_heads();
        let mut out = vec![0i32; n_heads];
        for n in 2..=self.n_gram {
            let mut mixed = (c[0] as u64).wrapping_mul(self.mult[0]);
            for j in 1..n {
                mixed ^= (c[j] as u64).wrapping_mul(self.mult[j]);
            }
            let base = (n - 2) * self.per_gram;
            for g in 0..self.per_gram {
                let h = base + g;
                // 剰余は u64 で取る。加算の後に i32 へ落とす
                // (from_meta で offset + vocab <= i32::MAX を検査済み)。
                let row = mixed % self.head_vocab_sizes[h] as u64 + self.head_offsets[h] as u64;
                debug_assert!(row <= i32::MAX as u64);
                out[h] = row as i32;
            }
        }
        Ok(out)
    }

    /// 索引 → [`PleTable::gather_rows`] → `n_heads * n_dim` の f32。
    ///
    /// 表の範囲外を指す索引は `gather_rows` がエラーにする (黙って 0 にしない)。
    pub fn gather(&self, table: &PleTable, ctx: &[u32]) -> Result<Vec<f32>> {
        self.gather_many(table, &[ctx])
    }

    /// 複数の窓をまとめて引く (P78)。
    ///
    /// `ctxs` の各要素は 1 つの窓 ([`PleNgram::gather`] に渡す `ctx` と同じもの)。
    /// 戻り値は `[ctxs.len()][n_heads * n_dim]` の平坦な f32 で、窓の順に並ぶ。
    ///
    /// 窓ごとに `gather` を呼ぶのとの違いは **I/O の順序だけ**:
    ///
    /// 1. 全窓の行 index を集める (i32 の負値はここで弾く)
    /// 2. 昇順に並べ、重複を落とす (同じ行を 2 回読まない)
    /// 3. 1 行 1 回だけ読む (`parts > 1` ならキューに複数同時に載せる)
    /// 4. 窓の順に配り直す
    ///
    /// 1 行のデコードは [`PleTable::gather_rows`] の同じ関数・同じバイト列なので、
    /// 行の読み順・重複の有無に関わらず値は `gather` とビット一致する
    /// (行長はブロック長の倍数 = 行境界がブロック境界に乗る)。
    pub fn gather_many(&self, table: &PleTable, ctxs: &[&[u32]]) -> Result<Vec<f32>> {
        let n_heads = self.n_heads();
        let n_dim = table.n_dim();
        let per_ctx = n_heads
            .checked_mul(n_dim)
            .ok_or_else(|| PleError("n_heads * n_dim がオーバーフロー".into()))?;
        let total = ctxs
            .len()
            .checked_mul(per_ctx)
            .ok_or_else(|| PleError("ctxs.len() * n_heads * n_dim がオーバーフロー".into()))?;
        let mut out = vec![0f32; total];
        if ctxs.is_empty() {
            return Ok(out);
        }

        // 1. 窓ごとの行 index (出力と同じ順で並べる)。
        let mut rows: Vec<u32> = Vec::with_capacity(ctxs.len() * n_heads);
        for c in ctxs {
            let idx = self.indices(c)?;
            for (h, &r) in idx.iter().enumerate() {
                if r < 0 {
                    return err(format!("ヘッド {h} の行 index {r} が負"));
                }
                rows.push(r as u32);
            }
        }

        // 2. 昇順 + 重複排除。同じ行は 1 回だけ読む。
        let mut uniq = rows.clone();
        uniq.sort_unstable();
        uniq.dedup();

        // 3. 表を 1 行 1 回だけ読む。
        let parts = ple_par_parts(uniq.len());
        let mut buf = vec![0f32; uniq.len() * n_dim];
        table.gather_rows_par(&uniq, &mut buf, parts)?;

        // 4. 窓の順に配り直す (binary_search は uniq が昇順なので必ず当たる)。
        for (i, &r) in rows.iter().enumerate() {
            let pos = uniq.binary_search(&r).map_err(|_| {
                PleError(format!("行 {r} が重複排除の結果に無い (内部不整合)"))
            })?;
            out[i * n_dim..(i + 1) * n_dim].copy_from_slice(&buf[pos * n_dim..(pos + 1) * n_dim]);
        }
        Ok(out)
    }
}
