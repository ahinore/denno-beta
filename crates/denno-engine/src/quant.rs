//! int8 weight-only 量子化 (HANDOFF §7 P3)。
//!
//! # 仕様 (重要)
//!
//! - **デフォルトは fp32**。量子化は CLI オプション (`denno serve --quant int8` /
//!   `denno generate --quant int8`) で明示的に有効化した場合のみ使われる。
//! - 量子化は safetensors ロード時の**オフライン変換** (per-output-channel 対称
//!   int8: 各出力チャネル行ごとに scale = max|w| / 127、w_q = round(w / scale))。
//! - **exact decoding は量子化パス内でも維持される**: カーネルは決定論的
//!   (各出力要素を固定順序で畳み込み、チャンク分割もスレッド数に依存しない) なので、
//!   同一 seed + 温度 0 → 同一出力。
//! - ただし **int8 モードの出力は fp32 モードと同一ではない** (丸め誤差により
//!   logits が僅かにずれる)。fp32 の exactness 基準 (HANDOFF §5) はあくまで
//!   fp32 モードのもの。

use anyhow::{bail, Result};

use crate::kquant::GgmlType;

/// 量子化モード。デフォルトは Fp32 (従来動作を完全維持)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantMode {
    /// 重みを f32 のまま使用 (デフォルト)。HANDOFF §5 の exactness 基準はこちら。
    #[default]
    Fp32,
    /// 線形層の重みを per-channel scale の int8 に量子化 (weight-only)。
    /// 決定論は維持されるが fp32 と同一出力にはならない。
    Int8,
    /// 線形層の重みを **GGUF Q4_0 の 4 bit ブロックのまま**保持する (s123)。
    ///
    /// int8 経路は GGUF を fp32 へ展開してから 1 要素 1 バイトへ詰め直すため、
    /// 元が 4.5 bit でも常に 8 bit 分の VRAM を積む (27B で 25.7 GB)。
    /// Q4_0 は元のブロックをそのまま保持するので 4.5 bit (27B で 13.7 GB) で済む。
    /// 数値は fp32 とも int8 とも一致しない (別の量子化)。決定論は維持される。
    Q4_0,
}

impl QuantMode {
    /// CLI 文字列 ("fp32" / "int8") から解析する
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "fp32" | "f32" | "none" => Ok(QuantMode::Fp32),
            "int8" | "i8" | "q8" => Ok(QuantMode::Int8),
            "q4" | "q4_0" => Ok(QuantMode::Q4_0),
            other => bail!("--quant の値は fp32 / int8 / q4 (got: {other})"),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            QuantMode::Fp32 => "fp32",
            QuantMode::Int8 => "int8",
            QuantMode::Q4_0 => "q4",
        }
    }
}

/// per-output-channel 対称 int8 量子化された重み。
/// w は nn.Linear 形式 [n, k] row-major、scale は出力チャネル (行) ごと。
#[derive(Debug, Clone)]
pub struct Quant8Linear {
    /// 量子化重み [n, k] row-major
    pub w: Vec<i8>,
    /// デ量子化スケール [n] (行ごと)。w_f32[j][d] ≒ w[j*k+d] * scale[j]
    pub scale: Vec<f32>,
    pub k: usize,
    pub n: usize,
    /// GPU オフロード用の重み識別子 (P17b)。構築時に採番され、Clone でも共有する
    /// (Clone は同一内容のディープコピーなので VRAM バッファの共有は数値的に安全)。
    /// ポインタではなく uid で識別するのは、Vec の解放→再確保でアドレスが再利用
    /// され、別内容の重みが同一アドレスに現れても誤参照しないため。
    pub uid: u64,
}

static NEXT_UID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_uid() -> u64 {
    std::sync::atomic::AtomicU64::fetch_add(&NEXT_UID, 1, std::sync::atomic::Ordering::Relaxed)
}

/// s80 (T7b): 既存の重み断片から**連結重み** (`Quant8Linear { .. }` を直接組む)
/// を作るときの uid 採番。uid は「内容が一意な重み」の識別子として GPU 側の
/// キャッシュキーに使われるため、連結で新しく作った重みは必ず新しい uid を持つ
/// 必要がある (元の uid を流用すると別形状の重みが同じキーで衝突する)。
pub fn next_q8_uid() -> u64 {
    next_uid()
}

impl Quant8Linear {
    /// f32 重み (w: [n,k] row-major) を int8 へ量子化する。
    ///
    /// 対称量子化: scale[j] = max(|w[j]|) / 127.0、q = round(w / scale)。
    /// 丸め誤差は各要素 |w - deq| ≤ scale[j] / 2 に収まる。
    pub fn quantize(w: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(w.len(), k * n, "quantize: weight size mismatch");
        let mut q = Self::zeros(k, n);
        q.quantize_rows_into(0, w);
        q
    }

    /// P37: 空 (全ゼロ) の [n,k] を作る (行チャンク書き込みの受け皿)。
    pub fn zeros(k: usize, n: usize) -> Self {
        Self { w: vec![0i8; k * n], scale: vec![0f32; n], k, n, uid: next_uid() }
    }

    /// P37: 行 `[r0, r0+rows)` を f32 から量子化して書き込む
    /// (`w.len()` は `k` の倍数)。量子化は行単位で閉じている (scale も行ごと)
    /// ので、何行ずつ書いても `quantize` の結果と**バイト一致**する。
    pub fn quantize_rows_into(&mut self, r0: usize, w: &[f32]) {
        assert_eq!(self.k * self.n, self.w.len(), "q8 quantize_rows: 次元不整合");
        assert_eq!(w.len() % self.k, 0, "q8 quantize_rows: w が k の倍数でない");
        let rows = w.len() / self.k;
        assert!(r0 + rows <= self.n, "q8 quantize_rows: 行範囲が n を超過");
        for (i, row) in w.chunks_exact(self.k).enumerate() {
            let j = r0 + i;
            let maxabs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // 全行ゼロの場合の 0 除算回避 (scale は任意の値でよい)
            let s = if maxabs > 0.0 { maxabs / 127.0 } else { 1.0 };
            self.scale[j] = s;
            let inv = 1.0 / s;
            for (q, v) in self.w[j * self.k..][..self.k].iter_mut().zip(row) {
                // round-half-away-from-zero + 安全側のクランプ (±127 に収める)
                let qi = (*v * inv).round();
                *q = qi.clamp(-127.0, 127.0) as i8;
            }
        }
    }

    /// デ量子化して f32 重み [n, k] を返す (テストと prefill 用)。
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.k * self.n];
        for j in 0..self.n {
            let s = self.scale[j];
            for d in 0..self.k {
                out[j * self.k + d] = self.w[j * self.k + d] as f32 * s;
            }
        }
        out
    }

    /// 重みの実メモリ量 (bytes)
    pub fn weight_bytes(&self) -> usize {
        self.w.len() + self.scale.len() * 4
    }
}


/// Q4_0 (ggml 互換) の重みを**元の 4 bit ブロックのまま**保持する (s123)。
///
/// 1 ブロック = 32 要素 = f16 スケール (2B) + ニブル 16B = 18B。
/// ggml の `quantize_row_q4_0_ref` と同じ規則で量子化するので、Q4_0 の GGUF を
/// fp32 経由で読み直しても同じ値に戻る。デ量子化は `y = d * (q - 8)`。
///
/// `Quant8Linear` との違いは scale の粒度 (行ごと vs 32 要素ごと) と保持形式だけ。
/// 経路 (GPU 試行 → CPU フォールバック) は同じ規約に従う。
#[derive(Debug, Clone)]
pub struct Q4Linear {
    /// [n, k/32] ブロックの生バイト列 (行ごとに連続・1 ブロック 18B)
    pub blocks: Vec<u8>,
    pub k: usize,
    pub n: usize,
    /// GPU 重みキャッシュのキー (`Quant8Linear` と同一の採番系)
    pub uid: u64,
}

/// Q4_0 の 1 ブロックあたり要素数 (ggml QK4_0)
pub const Q4_BLOCK_ELEMS: usize = 32;
/// Q4_0 の 1 ブロックあたりバイト数 (f16 scale 2B + ニブル 16B)
pub const Q4_BLOCK_BYTES: usize = 18;

impl Q4Linear {
    /// 1 行あたりのバイト数 (k は 32 の倍数であること)
    pub fn row_bytes(k: usize) -> usize {
        (k / Q4_BLOCK_ELEMS) * Q4_BLOCK_BYTES
    }

    /// f32 重み [n,k] を Q4_0 へ量子化する (ggml quantize_row_q4_0_ref と同規則)。
    pub fn quantize(w: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(w.len(), k * n, "q4 quantize: weight size mismatch");
        let mut q = Self::zeros(k, n);
        q.quantize_rows_into(0, w);
        q
    }

    /// P37: 空 (全ゼロ) の [n,k] ブロック列を作る (行チャンク書き込みの受け皿)。
    pub fn zeros(k: usize, n: usize) -> Self {
        assert_eq!(k % Q4_BLOCK_ELEMS, 0, "q4: k must be a multiple of 32");
        let nb = k / Q4_BLOCK_ELEMS;
        Self { blocks: vec![0u8; n * nb * Q4_BLOCK_BYTES], k, n, uid: next_uid() }
    }

    /// P37: 行 `[r0, r0+rows)` を f32 から量子化して blocks の該当位置へ書く
    /// (`w.len()` は `k` の倍数)。ブロックは 32 要素 (k 方向) で行をまたがず、
    /// 量子化はブロック内で閉じているので、何行ずつ書いても `quantize` の
    /// 結果と**バイト一致**する。
    pub fn quantize_rows_into(&mut self, r0: usize, w: &[f32]) {
        assert_eq!(self.k % Q4_BLOCK_ELEMS, 0, "q4: k must be a multiple of 32");
        assert_eq!(
            self.blocks.len(),
            self.n * Self::row_bytes(self.k),
            "q4 quantize_rows: 次元不整合"
        );
        assert_eq!(w.len() % self.k, 0, "q4 quantize_rows: w が k の倍数でない");
        let rows = w.len() / self.k;
        assert!(r0 + rows <= self.n, "q4 quantize_rows: 行範囲が n を超過");
        let nb = self.k / Q4_BLOCK_ELEMS;
        for (ri, row) in w.chunks_exact(self.k).enumerate() {
            let j = r0 + ri;
            for b in 0..nb {
                let src = &row[b * Q4_BLOCK_ELEMS..][..Q4_BLOCK_ELEMS];
                // ggml: |x| 最大の要素を符号付きで覚え、d = max / -8 とする。
                // こうすると max がちょうど ±8 に写り、ニブルは 0..15 に収まる。
                let mut amax = 0.0f32;
                let mut max = 0.0f32;
                for &x in src {
                    if x.abs() > amax {
                        amax = x.abs();
                        max = x;
                    }
                }
                let d = max / -8.0;
                let id = if d != 0.0 { 1.0 / d } else { 0.0 };
                let out = &mut self.blocks[(j * nb + b) * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
                out[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for (i, &x) in src.iter().enumerate() {
                    // ggml: (int8_t)(x*id + 8.5f) を 15 で頭切り。
                    // 0.5..16.5 の正値なので floor == 切り捨て。
                    let q = (x * id + 8.5).floor().clamp(0.0, 15.0) as u8;
                    if i < 16 {
                        out[2 + i] |= q;
                    } else {
                        out[2 + (i - 16)] |= q << 4;
                    }
                }
            }
        }
    }

    /// j 行目を f32 へ展開する (out は k 要素)。
    pub fn dequant_row_into(&self, j: usize, out: &mut [f32]) {
        let nb = self.k / Q4_BLOCK_ELEMS;
        debug_assert_eq!(out.len(), self.k);
        for b in 0..nb {
            let blk = &self.blocks[(j * nb + b) * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            for i in 0..16 {
                let byte = blk[2 + i];
                out[b * Q4_BLOCK_ELEMS + i] = d * (((byte & 0x0f) as i32 - 8) as f32);
                out[b * Q4_BLOCK_ELEMS + i + 16] = d * (((byte >> 4) as i32 - 8) as f32);
            }
        }
    }

    /// 全行を f32 へ展開する (テスト・参照用)。
    pub fn dequantize(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.k * self.n];
        for j in 0..self.n {
            self.dequant_row_into(j, &mut out[j * self.k..][..self.k]);
        }
        out
    }

    /// 重みの実メモリ量 (bytes)
    pub fn weight_bytes(&self) -> usize {
        self.blocks.len()
    }

    /// P35: GGUF の Q4_0 生ブロックを**再量子化せずに**保持する。
    ///
    /// engine の `Q4Linear` ブロックは ggml の `block_q4_0` と同一レイアウト
    /// (f16 scale 2B + 16B ニブル、低位ニブル = 先頭 16 要素・高位 = 後半 16 要素)
    /// なので、GGUF 側の型が Q4_0 のテンソルは fp32 へ展開せずそのまま積める。
    /// 1 行 k 要素 × n 行のバイト数が `n * row_bytes(k)` と一致することを検証し、
    /// 合わなければ `None` を返す (**fail-closed**: 呼び出し側は従来の fp32
    /// 経路へ落ちる)。型が Q4_0 であることの判定は呼び出し側の責務。
    pub fn from_gguf_blocks(blocks: &[u8], k: usize, n: usize) -> Option<Self> {
        if k == 0 || k % Q4_BLOCK_ELEMS != 0 {
            return None;
        }
        let want = n.checked_mul(Self::row_bytes(k))?;
        if blocks.len() != want {
            return None;
        }
        Some(Self { blocks: blocks.to_vec(), k, n, uid: next_uid() })
    }
}

/// P35: GGUF の生ブロックを再量子化せずに積む経路の env ゲート名。
///
/// 未設定・空・"0" (および "1"/"true"/"yes"/"on" 以外の値) は**無効** =
/// 従来経路 (GGUF → fp32 → 再量子化) と 1 バイトも変わらない。
pub const GGUF_PASSTHROUGH_ENV: &str = "DENNO_GGUF_PASSTHROUGH";

/// P35: env 値の解釈 (純関数・テスト可能)。有効化は明示的な真値のみ。
pub fn gguf_passthrough_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        Some(s) => matches!(s.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        None => false,
    }
}

/// P35: 現在の env 設定で passthrough が有効か (ロード開始時に 1 回だけ呼ぶ)。
pub fn gguf_passthrough_enabled() -> bool {
    let v = std::env::var(GGUF_PASSTHROUGH_ENV).ok();
    gguf_passthrough_from_value(v.as_deref())
}

/// P35: passthrough を**許可する** GGUF 型。proof test でビット一致を確認した型
/// だけを入れる (現在は空 = ゲートを ON にしても fp32 経路のまま)。
///
/// 2026-10-23 の実測 (`gguf::tests::q4_verbatim_matches_requant_real_gguf`,
/// /work/models/Qwen3.8-27B-Q4_0.gguf): **Q4_0 は一致しない**。このファイルの
/// Q4_0 ブロックは ggml の `quantize_row_q4_0_ref` (d = max/-8) ではなく
/// imatrix 重み付きのスケールで量子化されているため、デ量子化 → 再量子化すると
/// ブロックの f16 スケールが変わる (最初の不一致 = `blk.60.ffn_up.weight` の
/// byte 0、ニブルは一致)。よって Q4_0 は許可しない = 従来の fp32 経路を維持。
/// d = max/-8 で量子化し直したファイルで proof test が一致を報告したときに
/// だけ、ここへ `GgmlType::Q4_0` を足す (数値が変わる変更なので実機アンカー
/// の再取得が必要)。
pub fn gguf_passthrough_allows(_ty: GgmlType) -> bool {
    false
}

// ---------------------------------------------------------------------------
// P37: 行チャンク量子化 (大きな 2-D 重みの fp32 一時領域を上限つきにする)
// ---------------------------------------------------------------------------

/// P37: 1 チャンクの f32 一時領域の目標要素数 (4 M 要素 = 16 MiB)。
///
/// 行数は `k` から決める (`chunk_rows_for`)。行を丸ごと単位にするので、
/// 32 要素ブロック (Q4_0) も K 系スーパーブロック (256) も行内で閉じる。
pub const CHUNK_TARGET_ELEMS: usize = 4 << 20;

/// P37: `k` 要素の行に対して 1 チャンク何行にするか (最低 1 行)。
pub fn chunk_rows_for(k: usize) -> usize {
    if k == 0 {
        return 1;
    }
    (CHUNK_TARGET_ELEMS / k).max(1)
}

/// P37: 行チャンク量子化の env ゲート名。
///
/// **既定は有効** (新経路)。`0` / `false` / `no` / `off` を明示したときだけ
/// 従来の「テンソル全体を f32 へ展開 → 量子化」経路に戻る。どちらの経路も
/// 生成される量子化バイト列は同一 (テストで固定)。
pub const GGUF_CHUNKED_QUANT_ENV: &str = "DENNO_GGUF_CHUNKED_QUANT";

/// P37: env 値の解釈 (純関数・テスト可能)。未設定・空は既定 = 有効。
pub fn gguf_chunked_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None => true,
        Some(s) if s.is_empty() => true,
        Some(s) => !matches!(s.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
    }
}

/// P37: 現在の env 設定で行チャンク経路が有効か (ロード開始時に 1 回だけ呼ぶ)。
pub fn gguf_chunked_enabled() -> bool {
    let v = std::env::var(GGUF_CHUNKED_QUANT_ENV).ok();
    gguf_chunked_from_value(v.as_deref())
}

/// P42: `token_embd.weight` を「生ブロック常駐 + 行オンデマンド復元」で持つか。
///
/// **既定は有効**。`0` / `false` / `no` / `off` を明示したときだけ従来の常駐
/// f32 表 (27B Q4_0 で 5,085.6 MB) に戻る。復元値は f32 表の同じ行と
/// ビット一致するので、どちらの経路でも数値は変わらない (P38 の実ファイル証明)。
pub const EMBED_ONDEMAND_ENV: &str = "DENNO_EMBED_ONDEMAND";

/// P42: env 値の解釈 (純関数・テスト可能)。未設定・空は既定 = 有効。
pub fn embed_ondemand_from_value(v: Option<&str>) -> bool {
    match v.map(str::trim) {
        None => true,
        Some(s) if s.is_empty() => true,
        Some(s) => !matches!(s.to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
    }
}

/// P42: 現在の env 設定で行オンデマンド復元が有効か (ロード開始時に 1 回だけ呼ぶ)。
pub fn embed_ondemand_enabled() -> bool {
    let v = std::env::var(EMBED_ONDEMAND_ENV).ok();
    embed_ondemand_from_value(v.as_deref())
}

/// P37: 行チャンク量子化の実測/計算値 (レポート・テスト用)。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkedQuantStats {
    /// 1 チャンクの行数
    pub chunk_rows: usize,
    /// 実際に処理したチャンク数
    pub chunks: usize,
    /// 1 チャンクの最大 f32 要素数 (= 一時領域の要素数)
    pub chunk_elems: usize,
    /// 1 チャンクの最大ブロックバイト数 (= 読み取りバッファ)
    pub chunk_bytes: usize,
}

impl ChunkedQuantStats {
    /// 1 チャンクの f32 一時領域 (bytes)。これが新経路のピーク一時領域。
    pub fn peak_f32_bytes(&self) -> usize {
        self.chunk_elems * 4
    }

    /// 1 チャンクの f32 一時領域 + ブロックの読み取りバッファ (bytes)。
    pub fn peak_bytes(&self) -> usize {
        self.peak_f32_bytes() + self.chunk_bytes
    }
}

/// P37: GGUF の生ブロックを**行チャンク**で量子化済み線形重みへ変換する。
///
/// `fill(r0, rows, buf)` は行 `[r0, r0+rows)` のブロックバイト列を `buf` へ
/// 書く責務を持つ (長さは `rows * row_bytes`)。呼び出し側はファイルから
/// `seek` + `read_exact` で読んでも、既に mmap したスライスを写してもよい。
///
/// # ビット一致の根拠 (証明ではなく実装上の不変条件)
///
/// - 行は GGUF 上で連続 (dim0 = k が最内) なので、行境界はブロック境界でもある
///   (`k % blck_size == 0` を検証。違えば `Err` = 呼び出し側は従来経路へ)。
/// - `kquant::dequant_row` はブロック単位で閉じている (F32/F16/Bf16 は要素単位)
///   ので、行チャンクへ切っても各要素の値は全体を一度に展開した場合と同一。
/// - `Q4Linear` / `Quant8Linear` の量子化は**行単位**で閉じている
///   (`quantize_rows_into` は `quantize` の本体そのもの) ので、行の集合と順序が
///   同じなら結果のバイト列も同一。
///
/// したがって `n` 行を何行ずつ処理しても出力は `Linear::new(全体 f32)` と
/// バイト一致する。ピーク一時領域は 1 チャンク分 (`ChunkedQuantStats`)。
pub fn gguf_rows_quantized_chunked<F>(
    ty: GgmlType,
    mode: QuantMode,
    k: usize,
    n: usize,
    chunk_rows: usize,
    fill: F,
) -> std::result::Result<(Linear, ChunkedQuantStats), String>
where
    F: FnMut(usize, usize, &mut Vec<u8>) -> std::result::Result<(), String>,
{
    gguf_rows_quantized_chunked_with_sink(ty, mode, k, n, chunk_rows, fill, |_, _| {})
}

/// P37: [`gguf_rows_quantized_chunked`] の「チャンクの f32 を覗ける」版。
///
/// `sink(r0, w)` は量子化の直前に、そのチャンクの f32 (`rows * k` 要素) を
/// 受け取る。token_embd.weight のように **f32 の常駐表も必要な**テンソルで、
/// 展開を 1 回で済ませるために使う (表へのコピーと量子化が同じチャンクで起きる)。
pub fn gguf_rows_quantized_chunked_with_sink<F, S>(
    ty: GgmlType,
    mode: QuantMode,
    k: usize,
    n: usize,
    chunk_rows: usize,
    mut fill: F,
    mut sink: S,
) -> std::result::Result<(Linear, ChunkedQuantStats), String>
where
    F: FnMut(usize, usize, &mut Vec<u8>) -> std::result::Result<(), String>,
    S: FnMut(usize, &[f32]),
{
    if k == 0 || n == 0 {
        return Err(format!("行チャンク量子化: 形状が不正 (k={k}, n={n})"));
    }
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if k % bs != 0 {
        return Err(format!("行チャンク量子化: 1 行 {k} 要素がブロックサイズ {bs} の倍数でない"));
    }
    let row_bytes = (k / bs) * ts;
    let mut dst = match mode {
        QuantMode::Q4_0 => Linear::Q4(Q4Linear::zeros(k, n)),
        QuantMode::Int8 => Linear::Q8(Quant8Linear::zeros(k, n)),
        QuantMode::Fp32 => {
            return Err("行チャンク量子化: fp32 モードは量子化しない (対象外)".to_string())
        }
    };
    let chunk_rows = chunk_rows.clamp(1, n);
    let mut buf: Vec<u8> = Vec::new();
    let mut st = ChunkedQuantStats { chunk_rows, ..Default::default() };
    let mut r0 = 0usize;
    while r0 < n {
        let rows = (n - r0).min(chunk_rows);
        let want = rows * row_bytes;
        buf.clear();
        buf.resize(want, 0);
        fill(r0, rows, &mut buf)?;
        if buf.len() != want {
            return Err(format!(
                "行チャンク量子化: 行 {r0}..{} のバイト数 {} が期待 {want} と不一致",
                r0 + rows,
                buf.len()
            ));
        }
        // このチャンクだけを f32 へ展開する (ループ末尾で捨てる)。
        let w = crate::kquant::dequant_row(ty, &buf, rows * k);
        sink(r0, &w);
        match &mut dst {
            Linear::Q4(q) => q.quantize_rows_into(r0, &w),
            Linear::Q8(q) => q.quantize_rows_into(r0, &w),
            Linear::F32 { .. } => unreachable!("fp32 は上で弾いている"),
        }
        st.chunks += 1;
        st.chunk_elems = st.chunk_elems.max(w.len());
        st.chunk_bytes = st.chunk_bytes.max(buf.len());
        r0 += rows;
    }
    Ok((dst, st))
}

/// s80 (T7b): int8 重み 1 本に対する `Linear::apply_bt_rows_into` の Q8 分岐と
/// **同一の経路** (GPU `maybe_gpu_linear_into` 試行 → CPU matvec / matmul)。
/// ロード時に連結した重み (`Linear` を一度包まずに持つ層ゲート用) を同じ数値で
/// 計算するための入口で、演算列・丸め・分岐条件は `apply_bt_rows_into` と
/// 完全に同一 (どちらも同じカーネルを k,n 違いで呼ぶだけ)。
pub fn q8_apply_bt_rows_into(q: &Quant8Linear, a: &[f32], m: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), m * q.n);
    #[cfg(feature = "cuda")]
    if crate::cuda::maybe_gpu_linear_into(q.uid, a, &q.w, &q.scale, q.k, q.n, m, out) {
        return;
    }
    if m == 1 {
        crate::tensor::matvec_bt_i8_into(a, &q.w, &q.scale, q.k, q.n, out);
    } else {
        crate::tensor::matmul_bt_i8_into(a, &q.w, &q.scale, m, q.k, q.n, out);
    }
}

/// 線形層の重み保持: fp32 または int8 (weight-only)。
///
/// - `F32` は従来通り f32 `Vec` をそのまま保持する (カーネルも既存のまま)。
/// - `Q8` は常に **[n,k] (nn.Linear / matmul_bt 形)** で保持する。
///   GPT-2 の Conv1D 形式 ([k,n]) の重みはロード時に転置してから量子化する
///   (`Linear::from_colmajor`)。
#[derive(Debug, Clone)]
pub enum Linear {
    F32 { w: Vec<f32>, k: usize, n: usize },
    Q8(Quant8Linear),
    /// Q4_0 ネイティブ (s123): GGUF の 4 bit ブロックをそのまま保持する。
    Q4(Q4Linear),
}

/// s61 (T5a 撤回): GPU 線形層の唯一の入口は `cuda::maybe_gpu_linear`。
/// cuda 無効ビルドにはそのモジュール自体が無いため、回帰テスト用の
/// 「入口に到達したか」の計数だけをここに集約する。
pub(crate) mod seam {
    #[cfg(feature = "cuda")]
    pub use crate::cuda::linear_seam_calls;

    /// cuda 無効ビルド: GPU 入口が存在しないので常に 0
    /// (このとき CPU カーネル直呼びが正しい既定)。
    #[cfg(not(feature = "cuda"))]
    pub fn linear_seam_calls() -> u64 {
        0
    }
}

impl Linear {
    /// int8 (Q8) なら内部の Quant8Linear を返す (P17c 融合呼び出し用)。
    /// fp32 モードでは None (融合は int8 のみ — ビット一致のため)。
    pub fn quant8(&self) -> Option<&Quant8Linear> {
        match self {
            Linear::Q8(q) => Some(q),
            Linear::F32 { .. } | Linear::Q4(_) => None,
        }
    }

    /// w: [n,k] row-major (nn.Linear 形式) から構築。mode が Int8 ならここで
    /// オフライン量子化する (safetensors ロード直後に呼ばれる)。
    pub fn new(w: Vec<f32>, k: usize, n: usize, mode: QuantMode) -> Self {
        match mode {
            QuantMode::Fp32 => Linear::F32 { w, k, n },
            QuantMode::Int8 => Linear::Q8(Quant8Linear::quantize(&w, k, n)),
            QuantMode::Q4_0 => Linear::Q4(Q4Linear::quantize(&w, k, n)),
        }
    }

    /// m>1 のバッチ投影 (prefill 用)。各行が m==1 (apply_bt) と**ビット一致**する
    /// 経路を選ぶ:
    /// - Q8: apply_bt (matmul_bt_i8 / GPU dno_gemm_i8) — i32 累積は順序不変なので
    ///   行単位で m==1 と一致
    /// - F32: matvec_rows_bt (P9 順序保存レーン並列・k 方向は逐次) — matmul_bt
    ///   (sgemm) は累積順が異なり一致しないため使わない
    pub fn apply_bt_rows(&self, a: &[f32], m: usize) -> Vec<f32> {
        // s63: 実装を `apply_bt_rows_into` に一本化 (確保だけを行う薄いラッパ)。
        let mut out = vec![0.0f32; m * self.dims().1];
        self.apply_bt_rows_into(a, m, &mut out);
        out
    }

    /// s172 (P13): **2 本の小 n 投影 (GDN の beta/alpha) を 1 回の走査で**まとめて
    /// 計算する。`self.apply_bt_rows_into` と `other.apply_bt_rows_into` を順に
    /// 呼ぶのと**ビット一致**する (各 (i,j) の d 昇順逐次累積は不変で、変わるのは
    /// 処理順だけ)。
    ///
    /// 速い経路は「両方が F32・同じ k」のときだけ。Q8/Q4 が混ざる形・形の不一致・
    /// 出力長の不足は**従来の 2 回呼び出しへ落ちる** (fail-closed・数値は同一)。
    pub fn apply_bt_rows_pair_into(
        &self,
        other: &Self,
        a: &[f32],
        m: usize,
        out1: &mut [f32],
        out2: &mut [f32],
    ) {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        if let (
            Linear::F32 { w: w1, k: k1, n: n1 },
            Linear::F32 { w: w2, k: k2, n: n2 },
        ) = (self, other)
        {
            if k1 == k2
                && *k1 > 0
                && a.len() >= m * *k1
                && out1.len() >= m * *n1
                && out2.len() >= m * *n2
            {
                crate::tensor::matvec_rows_bt_pair_into(
                    &a[..m * *k1],
                    &w1[..*n1 * *k1],
                    &w2[..*n2 * *k2],
                    *k1,
                    *n1,
                    *n2,
                    m,
                    &mut out1[..m * *n1],
                    &mut out2[..m * *n2],
                );
                return;
            }
        }
        self.apply_bt_rows_into(a, m, out1);
        other.apply_bt_rows_into(a, m, out2);
    }

    /// s63: `apply_bt_rows` の書き込み先指定版 (prefill ホットパス用)。
    ///
    /// 呼び出し側が層をまたいで再利用するバッファを渡す前提で、ここでは
    /// 出力の確保を行わない (GPU 経路は D2H 後に直接コピー・CPU 経路は
    /// 一時 Vec からコピー)。演算列・丸め・分岐条件は `apply_bt_rows` と
    /// 完全に同一なので数値は不変。
    pub fn apply_bt_rows_into(&self, a: &[f32], m: usize, out: &mut [f32]) {
        debug_assert_eq!(out.len(), m * self.dims().1);
        match self {
            Linear::F32 { w, k, n } => {
                crate::tensor::matvec_rows_bt_into(a, w, *k, *n, m, out);
            }
            Linear::Q8(q) => {
                // GPU が計算できたら out は埋まっている (確保なし)
                #[cfg(feature = "cuda")]
                if crate::cuda::maybe_gpu_linear_into(q.uid, a, &q.w, &q.scale, q.k, q.n, m, out) {
                    return;
                }
                // CPU 経路も呼び出し側の領域へ直接書く (T5c の *_into を利用)
                if m == 1 {
                    crate::tensor::matvec_bt_i8_into(a, &q.w, &q.scale, q.k, q.n, out);
                } else {
                    crate::tensor::matmul_bt_i8_into(a, &q.w, &q.scale, m, q.k, q.n, out);
                }
            }
            Linear::Q4(q) => {
                #[cfg(feature = "cuda")]
                if crate::cuda::maybe_gpu_linear_q4_into(q.uid, a, &q.blocks, q.k, q.n, m, out) {
                    return;
                }
                crate::tensor::matvec_rows_bt_q4_into(a, &q.blocks, q.k, q.n, m, out);
            }
        }
    }

    /// w: [k,n] row-major (GPT-2 Conv1D 形式) から構築。
    /// Int8 の場合は転置して [n,k] 形で量子化する (fp32 は既存カーネルを
    /// 使えるよう元の形状のまま保持)。
    pub fn from_colmajor(w: &[f32], k: usize, n: usize, mode: QuantMode) -> Self {
        match mode {
            QuantMode::Fp32 => Linear::F32 { w: w.to_vec(), k, n },
            QuantMode::Int8 => {
                let mut wt = vec![0f32; k * n];
                for d in 0..k {
                    for j in 0..n {
                        wt[j * k + d] = w[d * n + j];
                    }
                }
                Linear::Q8(Quant8Linear::quantize(&wt, k, n))
            }
            QuantMode::Q4_0 => {
                let mut wt = vec![0f32; k * n];
                for d in 0..k {
                    for j in 0..n {
                        wt[j * k + d] = w[d * n + j];
                    }
                }
                Linear::Q4(Q4Linear::quantize(&wt, k, n))
            }
        }
    }

    /// c = a @ W^T を計算する (a: [m,k], W: [n,k]、戻り値 [m,n])。
    /// m == 1 (decode) では出力分割並列の matvec に切り替わる
    /// (Q8 は int8→fp32 デ量子化込みの fused matvec)。
    pub fn apply_bt(&self, a: &[f32], m: usize) -> Vec<f32> {
        match self {
            Linear::F32 { w, k, n } => {
                if m == 1 {
                    crate::tensor::matvec_bt(a, w, *k, *n)
                } else {
                    crate::tensor::matmul_bt(a, w, m, *k, *n)
                }
            }
            Linear::Q8(q) => {
                // P17b: DENNO_CUDA=1 なら GPU オフロード (CPU カーネルとビット一致)。
                // 無効/初期化失敗/実行時エラー時は None で既存 CPU 経路にフォールバック。
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    crate::cuda::maybe_gpu_linear(q.uid, a, &q.w, &q.scale, q.k, q.n, m)
                {
                    return out;
                }
                if m == 1 {
                    crate::tensor::matvec_bt_i8(a, &q.w, &q.scale, q.k, q.n)
                } else {
                    crate::tensor::matmul_bt_i8(a, &q.w, &q.scale, m, q.k, q.n)
                }
            }
            Linear::Q4(q) => {
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    crate::cuda::maybe_gpu_linear_q4(q.uid, a, &q.blocks, q.k, q.n, m)
                {
                    return out;
                }
                let mut out = vec![0.0f32; m * q.n];
                crate::tensor::matvec_rows_bt_q4_into(a, &q.blocks, q.k, q.n, m, &mut out);
                out
            }
        }
    }

    /// P6 バッチ decode 用: 複数セッション分の行 (a: [m,k]) を 1 回の重み
    /// ストリーミングで計算する。各 (i,j) の演算順序は `apply_bt` の m==1
    /// 経路 (matvec_bt / matvec_bt_i8) と完全一致するため、**バッチでまとめて
    /// decode しても各セッションは単独実行とビット一致する**。
    /// (prefill が使う m>1 の sgemm 経路は fp32 で累積順が異なるため使わない)
    pub fn apply_batch_bt(&self, a: &[f32], m: usize) -> Vec<f32> {
        match self {
            Linear::F32 { w, k, n } => crate::tensor::matvec_rows_bt(a, w, *k, *n, m),
            Linear::Q8(q) => {
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    crate::cuda::maybe_gpu_linear(q.uid, a, &q.w, &q.scale, q.k, q.n, m)
                {
                    return out;
                }
                crate::tensor::matvec_rows_bt_i8(a, &q.w, &q.scale, q.k, q.n, m)
            }
            Linear::Q4(q) => {
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    crate::cuda::maybe_gpu_linear_q4(q.uid, a, &q.blocks, q.k, q.n, m)
                {
                    return out;
                }
                let mut out = vec![0.0f32; m * q.n];
                crate::tensor::matvec_rows_bt_q4_into(a, &q.blocks, q.k, q.n, m, &mut out);
                out
            }
        }
    }

    /// c = a @ W を計算する (a: [m,k], W: [k,n] fp32 / Q8 は内部 [n,k]、戻り値 [m,n])。
    /// GPT-2 の Conv1D 形式用。Q8 は転置格納なので bt カーネルで計算する。
    pub fn apply(&self, a: &[f32], m: usize) -> Vec<f32> {
        match self {
            Linear::F32 { w, k, n } => crate::tensor::matmul(a, w, m, *k, *n),
            Linear::Q8(q) => {
                // Q8 は転置格納 [n,k] → bt カーネル (GPU 経路も同じ rows 仕様)
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    crate::cuda::maybe_gpu_linear(q.uid, a, &q.w, &q.scale, q.k, q.n, m)
                {
                    return out;
                }
                crate::tensor::matmul_bt_i8(a, &q.w, &q.scale, m, q.k, q.n)
            }
            Linear::Q4(q) => {
                #[cfg(feature = "cuda")]
                if let Some(out) =
                    crate::cuda::maybe_gpu_linear_q4(q.uid, a, &q.blocks, q.k, q.n, m)
                {
                    return out;
                }
                let mut out = vec![0.0f32; m * q.n];
                crate::tensor::matvec_rows_bt_q4_into(a, &q.blocks, q.k, q.n, m, &mut out);
                out
            }
        }
    }

    /// (k, n) 対
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Linear::F32 { k, n, .. } => (*k, *n),
            Linear::Q8(q) => (q.k, q.n),
            Linear::Q4(q) => (q.k, q.n),
        }
    }

    /// 重みの実メモリ量 (bytes)。量子化効果のログ/レポート用。
    pub fn weight_bytes(&self) -> usize {
        match self {
            Linear::F32 { w, .. } => w.len() * 4,
            Linear::Q8(q) => q.weight_bytes(),
            Linear::Q4(q) => q.weight_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// s123: 飽和しない値域では、Q4_0 の量子化は
    /// (1) 誤差がブロックスケールの 1/2 以下、(2) デ量子化 → 再量子化で同一バイト列。
    #[test]
    fn q4_quantize_is_idempotent() {
        let (k, n) = (64usize, 2usize);
        // 絶対値最大が負側 (-1.0) で、正側は +0.75 までに抑える。
        // d = +0.125 になり、どの要素も 15 に飽和しない値域。
        let mut w = Vec::with_capacity(k * n);
        for _ in 0..n {
            for i in 0..k {
                w.push(-1.0 + (i % 15) as f32 * 0.125);
            }
        }
        let q = Q4Linear::quantize(&w, k, n);
        assert_eq!(q.blocks.len(), n * (k / Q4_BLOCK_ELEMS) * Q4_BLOCK_BYTES);
        let dq = q.dequantize();
        let q2 = Q4Linear::quantize(&dq, k, n);
        assert_eq!(q.blocks, q2.blocks, "dequant -> quant でバイト列が変わった");
        for j in 0..n {
            for b in 0..k / Q4_BLOCK_ELEMS {
                let blk =
                    &q.blocks[(j * (k / Q4_BLOCK_ELEMS) + b) * Q4_BLOCK_BYTES..][..Q4_BLOCK_BYTES];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32().abs();
                for i in 0..Q4_BLOCK_ELEMS {
                    let idx = j * k + b * Q4_BLOCK_ELEMS + i;
                    assert!(
                        (w[idx] - dq[idx]).abs() <= d * 0.5 + 1e-6,
                        "誤差超過 idx={idx} err={} d={d}",
                        (w[idx] - dq[idx]).abs()
                    );
                }
            }
        }
    }

    /// s123: 既知の 1 ブロックで、レイアウト (low/high ニブルの位置) と
    /// 量子化規則をバイト列で固定する。
    ///
    /// w[i] = -1.0 + 0.0625*i (i=0..31) なので |max| = 1.0 (負) → d = 0.125。
    /// q_i = min(15, floor(-8 + 0.5*i + 8.5)) = min(15, floor(0.5 + 0.5*i))。
    /// i = 0,1,2,3 → 0,1,1,2 / i = 30,31 → 15 (飽和), 15 (飽和)。
    #[test]
    fn q4_known_block_bytes() {
        let w: Vec<f32> = (0..32).map(|i| -1.0 + 0.0625 * i as f32).collect();
        let q = Q4Linear::quantize(&w, 32, 1);
        assert_eq!(q.blocks.len(), 18);
        // f16(0.125) = 0x3000 (LE: 00 30)
        assert_eq!(&q.blocks[..2], &[0x00, 0x30], "scale が f16 0.125 でない");
        // byte = q_i | q_{i+16} << 4
        assert_eq!(q.blocks[2], 0x80); // i=0 q=0, i=16 q=8
        assert_eq!(q.blocks[3], 0x91); // i=1 q=1, i=17 q=9
        assert_eq!(q.blocks[4], 0x91); // i=2 q=1, i=18 q=9
        assert_eq!(q.blocks[5], 0xA2); // i=3 q=2, i=19 q=10
        assert_eq!(q.blocks[17], 0xF8); // i=15 q=8, i=31 q=15 (飽和)
        // 飽和した要素の誤差は最大でブロックスケール 1 個分 (ggml と同じ挙動)
        let dq = q.dequantize();
        let d = 0.125f32;
        for i in 0..32 {
            let e = (w[i] - dq[i]).abs();
            assert!(e <= d + 1e-6, "i={i} err={e}");
        }
        // 非飽和の要素は 1/2 スケール以内
        for i in 0..30 {
            assert!((w[i] - dq[i]).abs() <= d * 0.5 + 1e-6, "i={i}");
        }
    }

    /// s123: Q4 の CPU 経路は「デ量子化した重みの fp32 行列積」と一致する
    /// (丸めのみの差なので 1e-6 級)。
    #[test]
    fn q4_matvec_matches_fp32_on_dequantized_weights() {
        let (k, n, m) = (128usize, 5usize, 2usize);
        let w = pseudo_random(k * n, 999);
        let a = pseudo_random(k * m, 777);
        let q = Q4Linear::quantize(&w, k, n);
        let dq = q.dequantize();
        let mut got = vec![0f32; m * n];
        crate::tensor::matvec_rows_bt_q4_into(&a, &q.blocks, k, n, m, &mut got);
        let want = crate::tensor::matmul_bt(&a, &dq, m, k, n);
        for i in 0..m * n {
            assert!((got[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0), "idx={i} {} vs {}", got[i], want[i]);
        }
    }

    /// s123: Q4 経路は 2 回実行で同一ビット (決定論)。
    #[test]
    fn q4_matvec_is_deterministic() {
        let (k, n, m) = (64usize, 7usize, 3usize);
        let w = pseudo_random(k * n, 4242);
        let a = pseudo_random(k * m, 2424);
        let q = Q4Linear::quantize(&w, k, n);
        let mut c1 = vec![0f32; m * n];
        let mut c2 = vec![0f32; m * n];
        crate::tensor::matvec_rows_bt_q4_into(&a, &q.blocks, k, n, m, &mut c1);
        crate::tensor::matvec_rows_bt_q4_into(&a, &q.blocks, k, n, m, &mut c2);
        assert_eq!(
            c1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            c2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "Q4 matvec が非決定論"
        );
    }

    /// s123: Q4_0 モードの Linear は dims / weight_bytes が 4.5 bit 相当になる。
    #[test]
    fn q4_linear_dims_and_bytes() {
        let (k, n) = (256usize, 8usize);
        let w = pseudo_random(k * n, 1);
        let l = Linear::new(w, k, n, QuantMode::Q4_0);
        assert_eq!(l.dims(), (k, n));
        assert_eq!(l.weight_bytes(), n * (k / Q4_BLOCK_ELEMS) * Q4_BLOCK_BYTES);
        assert!(l.weight_bytes() < k * n); // 1 バイト/要素より小さい
        assert!(l.quant8().is_none(), "Q4 は融合経路 (quant8) に載らない");
    }


    /// 決定論的な疑似乱数 (LCG) でテスト用行列を作る (tensor.rs と同じ方式)
    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 1000.0 - 1.0 // [-1, 1)
            })
            .collect()
    }

    /// 素朴なリファレンス (整数ドット仕様): a をトークンごとに動的量子化し、
    /// c[j] = (Σ_d aq[d]*q[j*k+d]) as f32 * ascale * scale[j] (i32 逐次累積)。
    /// 整数累積は結合則が成立するため SIMD カーネルとビット一致する。
    fn naive_matvec_bt_i8(a: &[f32], q: &[i8], scale: &[f32], k: usize, n: usize) -> Vec<f32> {
        let maxabs = a.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let ascale = if maxabs > 0.0 { maxabs / 127.0 } else { 1.0 };
        let inv = 1.0 / ascale;
        let aq: Vec<i8> = a
            .iter()
            .map(|&v| ((v * inv).round().clamp(-127.0, 127.0)) as i8)
            .collect();
        (0..n)
            .map(|j| {
                let mut s = 0i32;
                for d in 0..k {
                    s += aq[d] as i32 * q[j * k + d] as i32;
                }
                s as f32 * ascale * scale[j]
            })
            .collect()
    }

    /// 素朴なリファレンス: c[i][j] = Σ_d a[i][d] * deq_w[j][d]
    fn naive_matmul_bt(
        a: &[f32],
        w_deq: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Vec<f32> {
        let mut c = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..k {
                    s += a[i * k + d] * w_deq[j * k + d];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    #[test]
    fn quantize_roundtrip_error_within_half_scale() {
        // 量子化/デ量子化の丸め誤差は per-channel scale の半分以下に収まる
        for (k, n) in [(16usize, 8usize), (64, 200), (3, 5)] {
            let w = pseudo_random(k * n, 11);
            let q = Quant8Linear::quantize(&w, k, n);
            let deq = q.dequantize();
            for j in 0..n {
                let s = q.scale[j];
                // scale は max|row| / 127
                let maxabs = w[j * k..][..k].iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!((s - maxabs / 127.0).abs() < 1e-6 * s.max(1e-30));
                for d in 0..k {
                    let err = (w[j * k + d] - deq[j * k + d]).abs();
                    // round による誤差は scale/2 (丸めの境界で ±1ulp 緩和)
                    assert!(
                        err <= s / 2.0 * 1.0001 + 1e-9,
                        "k={k} n={n} j={j} d={d}: err={err} > scale/2={}",
                        s / 2.0
                    );
                }
            }
        }
    }

    #[test]
    fn quantize_extreme_values_clamp_to_int8_range() {
        // 巨大/非正規の値でも i8 の範囲を逸脱しない
        let w = vec![-1e30f32, 1e30, 0.0, 1e-30, -1e-30, 42.0];
        let q = Quant8Linear::quantize(&w, 3, 2);
        assert!(q.w.iter().all(|&v| (-127..=127).contains(&v)));
        // ゼロ行: scale は 1.0 にフォールバックし、量子化値はすべて 0
        let qz = Quant8Linear::quantize(&[0.0, 0.0, 0.0, 0.0], 2, 2);
        assert_eq!(qz.scale[0], 1.0);
        assert!(qz.w.iter().all(|&v| v == 0));
        assert!(qz.dequantize().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn quantized_output_is_not_identical_to_fp32() {
        // 仕様の明記: 量子化は丸め誤差を持ち、デ量子化重みは元の f32 と一致しない
        let w = pseudo_random(8 * 32, 12);
        let q = Quant8Linear::quantize(&w, 8, 32);
        let deq = q.dequantize();
        let diff: usize = w.iter().zip(&deq).filter(|(a, b)| a != b).count();
        assert!(diff > 0, "量子化パスが fp32 と完全一致してしまう (仕様違反)");
    }

    #[test]
    fn matvec_bt_i8_matches_naive_reference() {
        // int8 matvec カーネル (並列) vs 素朴なスカラー実装: 完全一致
        for (k, n) in [(8usize, 16usize), (64, 200), (128, 70), (5, 3)] {
            let w = pseudo_random(k * n, 13);
            let q = Quant8Linear::quantize(&w, k, n);
            let a = pseudo_random(k, 14);
            let got = crate::tensor::matvec_bt_i8(&a, &q.w, &q.scale, k, n);
            let want = naive_matvec_bt_i8(&a, &q.w, &q.scale, k, n);
            for (g, wv) in got.iter().zip(&want) {
                assert_eq!(g.to_bits(), wv.to_bits(), "k={k} n={n}");
            }
        }
    }

    #[test]
    fn matvec_bt_i8_known_values() {
        // a = [-127, 127] は max|a|=127 → ascale=1 で厳密に量子化される (aq = a)。
        // 行 [1,0]/s=1 → dot=-127 → -127*1*1
        // 行 [0,1]/s=2 → dot= 127 → 127*1*2 = 254
        // 行 [1,1]/s=0.5 → dot= 0 → 0*0.5 = 0
        let a = [-127.0f32, 127.0];
        let qw = [1i8, 0, 0, 1, 1, 1];
        let scale = [1.0f32, 2.0, 0.5];
        let got = crate::tensor::matvec_bt_i8(&a, &qw, &scale, 2, 3);
        let want = [-127.0f32, 254.0, 0.0];
        for (g, wv) in got.iter().zip(&want) {
            assert_eq!(g.to_bits(), wv.to_bits());
        }
    }

    #[test]
    fn matmul_bt_i8_matches_matvec_rowwise_and_naive() {
        // prefill 用 i8 GEMM: 各行が decode 用 matvec と**ビット完全一致**
        // (両方とも同一規則の per-token 量子化 + 厳密な整数ドットのため。
        //  これが prefix cache ヒット/非ヒットでの決定論の根拠)。
        // デ量子化重みでの素朴な fp32 実装とは、アクティベーション量子化誤差の
        // 理論上界 |差| ≤ 127*k*ascale_i*scale_j 内で一致する。
        for (m, k, n) in [(3usize, 8usize, 16usize), (1, 64, 200), (17, 5, 3), (40, 896, 1152)] {
            let w = pseudo_random(k * n, 15);
            let q = Quant8Linear::quantize(&w, k, n);
            let a = pseudo_random(m * k, 16);
            let got = crate::tensor::matmul_bt_i8(&a, &q.w, &q.scale, m, k, n);

            // 行ごとに matvec とビット一致
            for i in 0..m {
                let row = crate::tensor::matvec_bt_i8(&a[i * k..][..k], &q.w, &q.scale, k, n);
                for j in 0..n {
                    assert_eq!(
                        got[i * n + j].to_bits(),
                        row[j].to_bits(),
                        "m={m} k={k} n={n} i={i} j={j}: prefill row != decode matvec"
                    );
                }
            }

            // デ量子化重みでの fp32 素朴実装と一致 (アクティベーション量子化の
            // 誤差上界内: |Δa| ≤ ascale_i/2、|deq_w| ≤ 127*scale_j、
            // |a| ≤ 127*ascale_i、|Δw| ≤ scale_j/2 → |差| ≤ 127*k*ascale_i*scale_j)
            let deq = q.dequantize();
            let want = naive_matmul_bt(&a, &deq, m, k, n);
            for i in 0..m {
                let amax = a[i * k..][..k].iter().fold(0f32, |mv, v| mv.max(v.abs()));
                let ascale_i = if amax > 0.0 { amax / 127.0 } else { 1.0 };
                for j in 0..n {
                    let bound = 127.0 * k as f32 * ascale_i * q.scale[j] + 1e-6;
                    assert!(
                        (got[i * n + j] - want[i * n + j]).abs() <= bound,
                        "m={m} k={k} n={n} i={i} j={j}: {got} != {want} (bound={bound})",
                        got = got[i * n + j],
                        want = want[i * n + j],
                        bound = bound
                    );
                }
            }
        }
    }

    /// s172 (P13): `apply_bt_rows_pair_into` は「両方 F32」以外では従来の 2 回
    /// 呼び出しへ落ちる (fail-closed)。落ちた場合も値は 2 回呼び出しと同一で
    /// あることを Q8 混在・形の不一致で固定する。
    #[test]
    fn apply_bt_rows_pair_falls_back_outside_f32_pair() {
        let (m, k, n) = (5usize, 64usize, 3usize);
        let a = pseudo_random(m * k, 31);
        let w1 = pseudo_random(k * n, 32);
        let w2 = pseudo_random(k * n, 33);
        let f32_a = Linear::new(w1.clone(), k, n, QuantMode::Fp32);
        let f32_b = Linear::new(w2.clone(), k, n, QuantMode::Fp32);
        let q8_a = Linear::new(w1.clone(), k, n, QuantMode::Int8);
        // (1) F32 + F32: 速い経路 (値は 2 回呼び出しと同一)
        // (2) Q8 + F32: フォールバック (Q8 は GPU 入口 / CPU matmul_bt_i8 を通る)
        let cases: [(Linear, Linear); 2] = [
            (f32_a.clone(), f32_b.clone()),
            (q8_a, f32_b.clone()),
        ];
        for (i, (l1, l2)) in cases.iter().enumerate() {
            let n1 = l1.dims().1;
            let n2 = l2.dims().1;
            let mut got1 = vec![0.0f32; m * n1];
            let mut got2 = vec![0.0f32; m * n2];
            l1.apply_bt_rows_pair_into(l2, &a, m, &mut got1, &mut got2);
            let mut want1 = vec![0.0f32; m * n1];
            let mut want2 = vec![0.0f32; m * n2];
            l1.apply_bt_rows_into(&a, m, &mut want1);
            l2.apply_bt_rows_into(&a, m, &mut want2);
            for idx in 0..m * n1 {
                assert_eq!(
                    got1[idx].to_bits(),
                    want1[idx].to_bits(),
                    "case={i} idx={idx}: 1 本目が 2 回呼び出しと不一致"
                );
            }
            for idx in 0..m * n2 {
                assert_eq!(
                    got2[idx].to_bits(),
                    want2[idx].to_bits(),
                    "case={i} idx={idx}: 2 本目が 2 回呼び出しと不一致"
                );
            }
        }
    }

    #[test]
    fn linear_apply_bt_q8_matches_fp32_within_tolerance() {
        // Linear::apply_bt: Q8 パスと fp32 パスの相対誤差は小さい
        // (ただし完全一致ではない = 仕様)
        for (k, n) in [(64usize, 128usize), (5, 3)] {
            let w = pseudo_random(k * n, 17);
            let a = pseudo_random(3 * k, 18);
            let f32_lin = Linear::new(w.clone(), k, n, QuantMode::Fp32);
            let q8_lin = Linear::new(w.clone(), k, n, QuantMode::Int8);
            let got = q8_lin.apply_bt(&a, 3);
            let want = f32_lin.apply_bt(&a, 3);
            // 出力のスケールに対する相対誤差 (要素がゼロ近傍でも評価できるよう
            // 行の最大絶対値で正規化する)
            let mut max_rel = 0f32;
            for i in 0..3 {
                let row_max = want[i * n..][..n]
                    .iter()
                    .fold(0f32, |m, v| m.max(v.abs()))
                    .max(1e-6);
                for (g, wv) in got[i * n..][..n].iter().zip(&want[i * n..][..n]) {
                    max_rel = max_rel.max((g - wv).abs() / row_max);
                }
            }
            // per-channel int8 (7bit 精度) の相対誤差のオーダー
            assert!(max_rel < 0.05, "k={k} n={n}: max_rel={max_rel}");
        }
    }

    #[test]
    fn linear_from_colmajor_q8_equals_explicit_transpose() {
        // GPT-2 Conv1D 形式: 転置してから量子化するのと同じ結果になる
        let (k, n) = (4usize, 6usize);
        let w = pseudo_random(k * n, 19); // [k,n]
        let q_col = Linear::from_colmajor(&w, k, n, QuantMode::Int8);
        let mut wt = vec![0f32; k * n];
        for d in 0..k {
            for j in 0..n {
                wt[j * k + d] = w[d * n + j];
            }
        }
        let q_row = Linear::new(wt, k, n, QuantMode::Int8);
        let a = pseudo_random(2 * k, 20);
        assert_eq!(
            q_col.apply(&a, 2),
            // c = a @ W = a @ (W^T)^T → bt カーネル
            q_row.apply_bt(&a, 2)
        );
        // fp32 モードでは元の [k,n] 形状が保持される
        let f32_col = Linear::from_colmajor(&w, k, n, QuantMode::Fp32);
        assert_eq!(f32_col.dims(), (k, n));
        // apply (a @ W) は素朴な実装と一致
        let got = f32_col.apply(&a, 2);
        for i in 0..2 {
            for j in 0..n {
                let mut s = 0f32;
                for d in 0..k {
                    s += a[i * k + d] * w[d * n + j];
                }
                assert!((got[i * n + j] - s).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn quant_mode_parse_and_default() {
        assert_eq!(QuantMode::default(), QuantMode::Fp32);
        assert_eq!(QuantMode::parse("int8").unwrap(), QuantMode::Int8);
        assert_eq!(QuantMode::parse("fp32").unwrap(), QuantMode::Fp32);
        assert!(QuantMode::parse("int4").is_err());
        assert_eq!(QuantMode::Int8.label(), "int8");
    }

    /// P35: `DENNO_GGUF_PASSTHROUGH` の解釈。未設定・空・"0"・未知値は無効
    /// (= 従来経路)、明示的な真値のみ有効。
    #[test]
    fn gguf_passthrough_gate_parsing() {
        assert!(!gguf_passthrough_from_value(None));
        assert!(!gguf_passthrough_from_value(Some("")));
        assert!(!gguf_passthrough_from_value(Some("0")));
        assert!(!gguf_passthrough_from_value(Some(" 0 ")));
        assert!(!gguf_passthrough_from_value(Some("2")));
        assert!(!gguf_passthrough_from_value(Some("off")));
        assert!(gguf_passthrough_from_value(Some("1")));
        assert!(gguf_passthrough_from_value(Some("true")));
        assert!(gguf_passthrough_from_value(Some("ON")));
    }

    /// P35: GGUF の Q4_0 生ブロックは engine の `Q4Linear` ブロックと同一
    /// レイアウトなので、再量子化したバイト列と一致する。形が合わない入力は
    /// None (fail-closed = 呼び出し側は従来の fp32 経路へ落ちる)。
    #[test]
    fn q4_from_gguf_blocks_matches_requantize() {
        let (k, n) = (64usize, 3usize);
        let w = pseudo_random(k * n, 31);
        let q = Q4Linear::quantize(&w, k, n);
        let verbatim = Q4Linear::from_gguf_blocks(&q.blocks, k, n).expect("形は一致するはず");
        assert_eq!(verbatim.blocks, q.blocks, "生ブロックは再量子化バイト列と同一");
        assert_eq!((verbatim.k, verbatim.n), (k, n));
        assert_eq!(verbatim.weight_bytes(), q.weight_bytes());
        // 形の不一致は None
        assert!(Q4Linear::from_gguf_blocks(&q.blocks[..q.blocks.len() - 1], k, n).is_none());
        assert!(Q4Linear::from_gguf_blocks(&q.blocks, k + 1, n).is_none()); // k が 32 の倍数でない
        assert!(Q4Linear::from_gguf_blocks(&q.blocks, k, n + 1).is_none());
        assert!(Q4Linear::from_gguf_blocks(&q.blocks, 0, 0).is_none());
    }

    /// P37: env ゲートの解釈。**未設定・空は既定 = 有効** (新経路)、
    /// 明示的な偽値だけが従来経路に戻す。
    #[test]
    fn gguf_chunked_gate_parsing() {
        assert!(gguf_chunked_from_value(None), "未設定は既定 (有効)");
        assert!(gguf_chunked_from_value(Some("")), "空は既定 (有効)");
        assert!(gguf_chunked_from_value(Some("1")));
        assert!(gguf_chunked_from_value(Some("yes")));
        assert!(gguf_chunked_from_value(Some("TRUE")));
        assert!(!gguf_chunked_from_value(Some("0")));
        assert!(!gguf_chunked_from_value(Some(" 0 ")));
        assert!(!gguf_chunked_from_value(Some("off")));
        assert!(!gguf_chunked_from_value(Some("false")));
        assert!(!gguf_chunked_from_value(Some("NO")));
    }

    /// P42: token_embd の行オンデマンド復元の env ゲート。**未設定・空は既定 =
    /// 有効** (新経路)、明示的な偽値だけが従来の常駐 f32 表に戻す。
    #[test]
    fn embed_ondemand_gate_parsing() {
        assert!(embed_ondemand_from_value(None), "未設定は既定 (有効)");
        assert!(embed_ondemand_from_value(Some("")), "空は既定 (有効)");
        assert!(embed_ondemand_from_value(Some("1")));
        assert!(embed_ondemand_from_value(Some("on")));
        assert!(!embed_ondemand_from_value(Some("0")));
        assert!(!embed_ondemand_from_value(Some(" 0 ")));
        assert!(!embed_ondemand_from_value(Some("off")));
        assert!(!embed_ondemand_from_value(Some("false")));
        assert!(!embed_ondemand_from_value(Some("NO")));
    }

    /// P37: 行チャンク数は `k` から決まり、f32 一時領域が
    /// `CHUNK_TARGET_ELEMS` を超えない (k が大きいほど行数が減る)。
    #[test]
    fn chunk_rows_keeps_transient_bounded() {
        for k in [32usize, 64, 5120, 6144, 17408, 1 << 20] {
            let rows = chunk_rows_for(k);
            assert!(rows >= 1, "k={k} で行数 0");
            assert!(
                rows * k <= CHUNK_TARGET_ELEMS.max(k),
                "k={k} rows={rows} で一時領域が上限を超過"
            );
        }
        assert_eq!(chunk_rows_for(0), 1);
        // 27B の実形状: 5120 → 819 行 (16.0 MiB) / 17408 → 240 行 (16.0 MiB)
        assert_eq!(chunk_rows_for(5120), 819);
        assert_eq!(chunk_rows_for(17408), 240);
    }

    /// P37: 行チャンク量子化は「テンソル全体を f32 へ展開 → 量子化」と
    /// **バイト一致**する。チャンク幅を 1 行から n 行まで変えても不変。
    ///
    /// 入力は GGUF の生ブロック列そのもの (Q4_0 = 18B ブロック, F32 = 4B 要素) で、
    /// どちらの型でも `dequant_row` がブロック単位で閉じていることを固定する。
    #[test]
    fn gguf_chunked_quant_matches_whole_tensor() {
        let (k, n) = (128usize, 7usize);
        let w = pseudo_random(k * n, 4242);
        // Q4_0 の生ブロック (GGUF のバイト列に相当) と F32 のバイト列
        let q4_src = Q4Linear::quantize(&w, k, n).blocks;
        let f32_src: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();

        for (ty, src) in [(GgmlType::Q4_0, &q4_src), (GgmlType::F32, &f32_src)] {
            let bs = ty.blck_size();
            let ts = ty.type_size();
            let row_bytes = (k / bs) * ts;
            assert_eq!(src.len(), n * row_bytes);
            // 従来経路: 全体を f32 へ展開してから量子化 (dequant_tensor → Linear::new)
            let whole = crate::kquant::dequant_row(ty, src, k * n);
            for mode in [QuantMode::Q4_0, QuantMode::Int8] {
                let want = Linear::new(whole.clone(), k, n, mode);
                for chunk in [1usize, 2, 3, n, n + 5] {
                    let (got, st) = gguf_rows_quantized_chunked(
                        ty,
                        mode,
                        k,
                        n,
                        chunk,
                        |r0, rows, buf| {
                            buf.clear();
                            buf.extend_from_slice(&src[r0 * row_bytes..(r0 + rows) * row_bytes]);
                            Ok(())
                        },
                    )
                    .unwrap_or_else(|e| panic!("{ty:?}/{mode:?}/chunk={chunk}: {e}"));
                    assert_eq!(st.chunk_rows, chunk.min(n));
                    assert_eq!(st.chunks, n.div_ceil(chunk.min(n)));
                    assert!(st.peak_f32_bytes() <= CHUNK_TARGET_ELEMS.max(k) * 4);
                    match (&got, &want) {
                        (Linear::Q4(a), Linear::Q4(b)) => assert_eq!(
                            a.blocks, b.blocks,
                            "{ty:?}/Q4_0/chunk={chunk}: ブロック列が一致しない"
                        ),
                        (Linear::Q8(a), Linear::Q8(b)) => {
                            assert_eq!(
                                a.w, b.w,
                                "{ty:?}/int8/chunk={chunk}: 量子化バイト列が一致しない"
                            );
                            assert_eq!(
                                a.scale.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                b.scale.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                                "{ty:?}/int8/chunk={chunk}: scale が一致しない"
                            );
                        }
                        _ => panic!("量子化モードが違う"),
                    }
                }
            }
        }
    }

    /// P37: 行がブロック境界に載らない (k がブロックサイズの倍数でない) 型は
    /// `Err` = 呼び出し側は従来経路へ落ちる。fp32 モードも対象外。
    #[test]
    fn gguf_chunked_quant_rejects_unaligned_rows() {
        let src = vec![0u8; 100];
        let e = gguf_rows_quantized_chunked(GgmlType::Q4_0, QuantMode::Q4_0, 33, 1, 1, |_, _, b| {
            b.clear();
            b.extend_from_slice(&src);
            Ok(())
        })
        .expect_err("k=33 は 32 の倍数でないので拒否するはず");
        assert!(e.contains("倍数"), "{e}");
        let e = gguf_rows_quantized_chunked(GgmlType::F32, QuantMode::Fp32, 32, 1, 1, |_, _, _| Ok(()))
            .expect_err("fp32 モードは対象外");
        assert!(e.contains("fp32"), "{e}");
        // 供給バイト数が足りない場合も Err (静かに短い重みを作らない)
        let e = gguf_rows_quantized_chunked(GgmlType::F32, QuantMode::Int8, 32, 2, 1, |_, _, b| {
            b.clear();
            b.extend_from_slice(&[0u8; 4]);
            Ok(())
        })
        .expect_err("バイト数不足は拒否するはず");
        assert!(e.contains("不一致"), "{e}");
    }

    #[test]
    fn int8_weights_are_quarter_size() {
        // 帯域半減の前提確認: int8 重みは fp32 の約 1/4 (+scale 分)
        let (k, n) = (1024usize, 1024usize);
        let w = pseudo_random(k * n, 21);
        let f32_lin = Linear::new(w, k, n, QuantMode::Fp32);
        let q8_lin = Linear::new(pseudo_random(k * n, 21), k, n, QuantMode::Int8);
        assert_eq!(f32_lin.weight_bytes(), k * n * 4);
        assert!(q8_lin.weight_bytes() < k * n * 4 / 4 + n * 4 + 16);
    }

    // ------------------------------------------------------------------
    // s61 (T5a 撤回): prefill 経路が `maybe_gpu_linear` を迂回しないこと。
    //
    // s60 (T5a) は prefill の投影を `apply_bt_rows_maybe_gpu` に差し替えたが、
    // ゲート未指定時は `tensor::matmul_bt_i8` を直接呼んでいたため、GPU 線形層の
    // 唯一の入口 (`cuda::maybe_gpu_linear`) を **静かに迂回** し、既定構成で
    // prefill が 2.85x 遅くなった。同時に proj のプロファイルが `via=cpu` と
    // 出ていた (迂回の目撃証拠)。
    //
    // この退行は「数値が合っているのに遅い」形なので、正確性テストでは捕まらない。
    // 捕まえるには **入口に到達したか** を見るしかない。以下のテストは GPU 実機
    // 不要 (入口到達の計数 + CPU カーネルとのビット一致で判定する)。
    // ------------------------------------------------------------------

    /// 既定の prefill 経路 (`apply_bt_rows`) は結果が CPU カーネルとビット一致し、
    /// かつ cuda 有効ビルドでは GPU 線形の入口 (`maybe_gpu_linear`) に **必ず**
    /// 到達する。
    ///
    /// 083218d の形 (ゲート未指定で `matmul_bt_i8` 直呼び) に戻すと入口呼び出しが
    /// 増えないので、cuda 有効ビルドでこのテストが落ちる。
    #[test]
    fn prefill_apply_bt_rows_goes_through_gpu_seam() {
        // s163 (K4): このテストは `prof::time_n_ev` を直接呼ぶ (1 サンプル) ので、
        // prof の厳密比較の窓と排他する (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        // m=64 は既定の GEMV 上限 (~16) 超・GEMM 閾値 (256) 未満 = prefill の典型
        let (k, n, m) = (64usize, 32usize, 64usize);
        let w = pseudo_random(k * n, 23);
        let a = pseudo_random(m * k, 24);
        let lin = Linear::new(w, k, n, QuantMode::Int8);

        let calls_before = seam::linear_seam_calls();
        let want = linear_cpu_rows_i8(&a, lin.quant8().unwrap(), m);
        let got = crate::prof::time_n_ev(crate::prof::QKV, m, || lin.apply_bt_rows(&a, m));
        let calls_after = seam::linear_seam_calls();

        assert_eq!(
            got, want,
            "prefill の既定経路は CPU カーネルとビット一致 (GPU 無効時 = 純 CPU)"
        );
        // 入口到達の計数は cuda 有効ビルドでしか意味を持たない (無効ビルドは
        // 入口そのものが存在せず、CPU 直呼びが正しい既定)。
        #[cfg(feature = "cuda")]
        assert!(
            calls_after > calls_before,
            "apply_bt_rows (m={m}) が cuda::maybe_gpu_linear に到達していない: \
             prefill の投影は GPU 入口を迂回してはならない (T5a の退行)"
        );
        #[cfg(not(feature = "cuda"))]
        assert_eq!(calls_after, calls_before, "cuda 無効ビルドに GPU 入口は無い");
    }

    /// `apply_bt` / `apply_bt_rows` / `apply` / `apply_batch_bt` の **全 Q8 経路** が
    /// 同じ入口を通ること (入口を 1 つに保つ = 迂回を作らない)。
    #[test]
    fn all_q8_linear_paths_share_the_gpu_seam() {
        let (k, n, m) = (32usize, 16usize, 64usize);
        let w = pseudo_random(k * n, 25);
        let a = pseudo_random(m * k, 26);
        let lin = Linear::new(w, k, n, QuantMode::Int8);
        let paths: [(&str, fn(&Linear, &[f32], usize) -> Vec<f32>); 4] = [
            ("apply_bt", |l, a, m| l.apply_bt(a, m)),
            ("apply_bt_rows", |l, a, m| l.apply_bt_rows(a, m)),
            ("apply", |l, a, m| l.apply(a, m)),
            ("apply_batch_bt", |l, a, m| l.apply_batch_bt(a, m)),
        ];
        for (name, f) in paths {
            let before = seam::linear_seam_calls();
            let _ = f(&lin, &a, m);
            let after = seam::linear_seam_calls();
            #[cfg(feature = "cuda")]
            assert!(
                after > before,
                "{name} が cuda::maybe_gpu_linear を迂回している (GPU 入口は 1 つに保つ)"
            );
            #[cfg(not(feature = "cuda"))]
            let _ = (name, after, before);
        }
    }

    /// Q8 の CPU 行並列カーネル (参考実装: `apply_bt` の CPU 分岐と同一)。
    fn linear_cpu_rows_i8(a: &[f32], q: &Quant8Linear, m: usize) -> Vec<f32> {
        crate::tensor::matmul_bt_i8(a, &q.w, &q.scale, m, q.k, q.n)
    }
}
