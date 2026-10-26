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

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use crate::gguf::{GgufFile, TensorRef};
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
}
