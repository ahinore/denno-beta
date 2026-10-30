//! P-MOE-E4: GGUF の量子化ブロックを**そのまま保持する**線形層。
//!
//! 既存の `Quant8Linear` / `Q4Linear` はロード時に int8 / Q4_0 へ量子化し直す
//! (27B の重みはその経路で作られている)。Flash-Next の重みは GGUF の中で既に
//! Q8_0 / Q4_K / Q5_K / Q5_1 / BF16 / F32 なので、**作り直さず生バイトのまま
//! 計算する**。量子化し直すと数値が変わり、アンカー比較の意味が無くなる。
//!
//! 数値契約:
//! - `y[r] = dot(row_r, x)` を r の昇順、内積は i の昇順に f32 で加算する
//!   (本体は `model::moe_expert::expert_gemv` を共有する。CPU と GPU の
//!   ビット一致はこの順序で決まる)。
//! - 行のデコードは `kquant::dequant_row_into` と同一 (GGML のブロック仕様)。
//!
//! GPU 経路はまだ繋いでいない (重みの常駐キャッシュと一緒に入れる)。現状は
//! 常に CPU 参照で計算する。Flash-Next の 1 トークンあたりの実測はこの経路で
//! 取れるが、120B を CPU で回すので秒/token 級になる。

use anyhow::{bail, Result};

use crate::kquant::{dequant_row_into, GgmlType};
use crate::model::moe_expert::{expert_gemv, expert_gemv_q8_fast};
use crate::quant::WeightBlocks;

/// 1 行のバイト数。`moe_expert::row_bytes` は量子型だけを受けるので、
/// F32 / F16 / BF16 も含めて受ける版をここに持つ (ブロック仕様は同じ)。
fn row_bytes_any(ty: GgmlType, k: usize) -> Result<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if k == 0 || k % bs != 0 {
        bail!("GgufLinear: k={k} が {ty:?} のブロックサイズ {bs} の倍数でない");
    }
    Ok((k / bs) * ts)
}

/// `DENNO_CPU_GEMV_PAR` の値の解釈 (純粋関数)。**既定 on**、`0` のときだけ無効。
///
/// 2026-10-30 の実測 (118 出力 tok、3 ラン/構成) で 25.66 -> 25.14 s (-2.2 %)、
/// 出力 md5 は全ラン一致 (`3b77a3b996fd`)。行は互いに独立なので数値は変わらない。
/// env を読む部分から切り離してあるのは、テストが環境変数を触らずに両方の分岐を
/// 確かめられるようにするため (`moe_par_from_value` と同じ流儀)。
fn cpu_gemv_par_from_value(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("off") | Some("false"))
}

/// テスト用の上書き (0 = env を見る / 1 = 強制 off / 2 = 強制 on)。
///
/// env はプロセスで 1 回しか読めず、テストは並列に走るので、`devprof` の
/// `set_on_for_test` と同じく明示的に切り替えられる入口を用意する。
static CPU_GEMV_PAR_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(0);

/// 行方向の並列化が有効か (`DENNO_CPU_GEMV_PAR=1`、既定 off)。プロセスで 1 回だけ読む。
fn cpu_gemv_par_enabled() -> bool {
    match CPU_GEMV_PAR_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        1 => return false,
        2 => return true,
        _ => {}
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| cpu_gemv_par_from_value(std::env::var("DENNO_CPU_GEMV_PAR").ok().as_deref()))
}

/// テスト専用: 行並列の on/off を強制する (`None` で env に戻す)。
#[cfg(test)]
pub(crate) fn set_cpu_gemv_par_for_test(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    CPU_GEMV_PAR_OVERRIDE.store(match v {
        None => 0,
        Some(false) => 1,
        Some(true) => 2,
    }, Ordering::Relaxed);
}

/// 行並列を立てる条件 (実測で決めた)。
///
/// - **行数が 8 以上**、または
/// - **1 呼び出しの積和 (行数 × k) が 32768 以上** (行数 4 でも k が広ければ得)。
///
/// 根拠は `bench_cpu_gemv_par_threshold_sweep` (8 / 16 スレッド、reps=50):
/// 行数 4・k=2560 (10240 MAC) は 0.78〜0.92x で損、行数 2〜3 は k=10240 でも
/// 0.87〜1.08x で損益なし。行数 4・k=10240 (40960 MAC) は 1.22〜1.45x、
/// 行数 8 は k=2560 で 1.46〜1.99x、k=10240 で 1.89〜2.31x。
/// 実モデルの 4 形は router (512×2560)・ssm_beta/alpha (48×2560)・
/// hc_inject (4×10240) がすべて条件を満たす (`REPORT_task65.md`)。
const CPU_GEMV_PAR_MIN_ROWS: usize = 8;
/// 行数がこれ未満でも、積和がこの値以上なら並列化する。
const CPU_GEMV_PAR_MIN_MACS: usize = 32_768;

/// テスト専用: しきい値を下げて小さい形の並列損得を測るための上書き (0 = 既定)。
#[cfg(test)]
static PAR_MIN_ROWS_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// 実際に使うしきい値 (テストでは上書きできる)。
fn cpu_gemv_par_min_rows() -> usize {
    #[cfg(test)]
    {
        let v = PAR_MIN_ROWS_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
        if v > 0 {
            return v;
        }
    }
    CPU_GEMV_PAR_MIN_ROWS
}

#[cfg(test)]
pub(crate) fn set_par_min_rows_for_test(v: Option<usize>) {
    PAR_MIN_ROWS_OVERRIDE.store(v.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
}

/// この形を並列化するか (行数と積和のしきい値)。
fn cpu_gemv_par_applies(n: usize, k: usize) -> bool {
    n >= cpu_gemv_par_min_rows() || n.saturating_mul(k) >= CPU_GEMV_PAR_MIN_MACS
}

/// 行並列の起動回数 (テストが「1 呼び出し = 1 領域」を数えるため)。
#[cfg(test)]
static PAR_LAUNCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// 1 呼び出しで割った行チャンクの本数 (行ごとに並列領域を立てていないことの確認)。
#[cfg(test)]
static PAR_TASKS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// テスト専用: 行並列の状態 (env 上書き・カウンタ) を触るテストを直列化する。
///
/// 上書きはプロセス全体で 1 個なので、crate 内の他のテスト (`qwen4exp` の
/// 入口テストなど) も同じ mutex を取って、カウンタの取り合いを避ける。
#[cfg(test)]
pub(crate) fn par_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
    L.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
pub(crate) fn par_reset() {
    PAR_LAUNCHES.store(0, std::sync::atomic::Ordering::Relaxed);
    PAR_TASKS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
pub(crate) fn par_launches() -> usize {
    PAR_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(test)]
pub(crate) fn par_tasks() -> usize {
    PAR_TASKS.load(std::sync::atomic::Ordering::Relaxed)
}

#[inline]
fn note_par_launch() {
    #[cfg(test)]
    PAR_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn note_par_task() {
    #[cfg(test)]
    PAR_TASKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// 非量子型 (F32 / F16 / BF16) の `out[r] = dot(row_r, x)` を**行方向に**分ける。
///
/// - 行は互いに独立で、1 行の内積は `i` の昇順に f32 で逐次加算する (従来と同じ
///   `dequant_row_into` + 同じ順の `acc +=`)。どの行をどのスレッドが担当しても
///   各要素の丸め順序は変わらないので、**分割の有無でビット一致する**。
/// - 並列領域は 1 呼び出しにつき 1 回だけ立てる (行ループの内側では立てない)。
fn gemv_rows_par(ty: GgmlType, blocks: &[u8], rb: usize, k: usize, x: &[f32], out: &mut [f32]) {
    use rayon::prelude::*;
    let threads = rayon::current_num_threads().max(1);
    // 1 スレッドあたり最低 1 行。行数がスレッド数で割り切れない端数は最後の
    // チャンクに入る (`par_chunks` の末尾は短くなる)。
    let rows = out.len().div_ceil(threads).max(1).min(out.len());
    let bytes = rows * rb;
    note_par_launch();
    blocks
        .par_chunks(bytes)
        .zip(out.par_chunks_mut(rows))
        .for_each(|(bb, oo)| {
            note_par_task();
            let mut row = vec![0f32; k];
            for (r, slot) in oo.iter_mut().enumerate() {
                dequant_row_into(ty, &bb[r * rb..(r + 1) * rb], &mut row);
                let mut acc = 0f32;
                for i in 0..k {
                    acc += row[i] * x[i];
                }
                *slot = acc;
            }
        });
}

/// GGUF のブロックを保持したまま GEMV する線形層 (n 行 k 列、y = W x)。
pub struct GgufLinear {
    uid: u64,
    ty: GgmlType,
    k: usize,
    n: usize,
    blocks: WeightBlocks,
    /// Q8_0 のときだけ、AVX2 融合カーネルを使うか。
    ///
    /// 既定は false (27B の Q8_0 テンソル 1 本がビット一致契約の対象なので、
    /// 明示的に選んだモデルだけ速い経路に載せる)。
    fast_q8: bool,
}

impl GgufLinear {
    /// 生バイトから作る。`blocks.len()` は `n * row_bytes(ty, k)` と一致すること。
    pub fn new(ty: GgmlType, k: usize, n: usize, blocks: WeightBlocks) -> Result<Self> {
        let rb = row_bytes_any(ty, k)?;
        if n == 0 || k == 0 {
            bail!("GgufLinear: 形状が 0 (k={k}, n={n})");
        }
        let want = n
            .checked_mul(rb)
            .ok_or_else(|| anyhow::anyhow!("GgufLinear: n * row_bytes が usize を溢れる"))?;
        if blocks.as_slice().len() != want {
            bail!(
                "GgufLinear: バイト数 {} が n * row_bytes = {} と違う (ty={ty:?}, k={k}, n={n})",
                blocks.as_slice().len(),
                want
            );
        }
        Ok(Self {
            uid: crate::quant::next_q8_uid(),
            ty,
            k,
            n,
            blocks,
            fast_q8: false,
        })
    }

    /// Q8_0 の AVX2 融合カーネルを有効にする (ビット一致は保証しない)。
    pub fn with_fast_q8(mut self, on: bool) -> Self {
        self.fast_q8 = on;
        self
    }

    /// Q8_0 の融合カーネルが有効か (既定 false)。
    pub fn fast_q8(&self) -> bool {
        self.fast_q8
    }

    /// スライスから作る (テスト・小さいテンソル用)。
    pub fn from_slice(ty: GgmlType, k: usize, n: usize, bytes: &[u8]) -> Result<Self> {
        Self::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
    }

    pub fn uid(&self) -> u64 {
        self.uid
    }
    pub fn ty(&self) -> GgmlType {
        self.ty
    }
    /// 入力長 (列数)
    pub fn k(&self) -> usize {
        self.k
    }
    /// 出力長 (行数)
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn blocks(&self) -> &WeightBlocks {
        &self.blocks
    }
    pub fn weight_bytes(&self) -> usize {
        self.blocks.as_slice().len()
    }

    /// `out[r] = dot(row_r, x)`。`x.len() == k`、`out.len() == n` が必要。
    ///
    /// プロファイラ上の行名は既定で `"gemv"` (op 名を付けたいときは
    /// [`Self::gemv_label`])。
    pub fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        self.gemv_label("gemv", x, out)
    }

    /// [`Self::gemv`] の op 名付き入口。
    ///
    /// `label` は device プロファイラ (`DENNO_DEV_PROFILE=1`) の行名にだけ使う。
    /// 数値経路には影響しない (文字列を渡すだけで、カーネル・引数・丸め順は
    /// 同一)。off のときは `devprof::on()` の atomic load 1 回で従来と同じ経路。
    pub fn gemv_label(&self, label: &'static str, x: &[f32], out: &mut [f32]) -> Result<()> {
        use crate::model::devprof;
        if !devprof::on() {
            return self.gemv_impl(x, out);
        }
        let t = std::time::Instant::now();
        devprof::clear();
        let r = self.gemv_impl(x, out);
        let ns = t.elapsed().as_nanos() as u64;
        let (up, down, sync_ns) = devprof::take_pending();
        devprof::add(label, ns, up, down, sync_ns);
        r
    }

    /// 本体。[`Self::gemv_label`] が時間と転送量を測るために外側から包む。
    fn gemv_impl(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.k {
            bail!("GgufLinear::gemv: x の長さ {} が k={} と違う", x.len(), self.k);
        }
        if out.len() != self.n {
            bail!("GgufLinear::gemv: out の長さ {} が n={} と違う", out.len(), self.n);
        }
        match self.ty {
            // 量子型は expert GEMV と本体を共有する (1 行ずつデコードして逐次内積)。
            GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => {
                // P-MOE-E5: 重みが VRAM に常駐していれば GPU で計算する。
                // 対象外・GPU 無効・不調では false が返り、下の CPU 参照へ落ちる。
                #[cfg(feature = "cuda")]
                {
                    if crate::cuda::maybe_gpu_gguf_gemv_into(
                        self.uid,
                        self.ty,
                        self.blocks.as_slice(),
                        self.k,
                        self.n,
                        x,
                        out,
                    ) {
                        return Ok(());
                    }
                }
                if self.fast_q8 && self.ty == GgmlType::Q8_0 {
                    return expert_gemv_q8_fast(self.blocks.as_slice(), self.k, self.n, x, out);
                }
                expert_gemv(self.ty, self.blocks.as_slice(), self.k, self.n, x, out)
            }
            // 無量子型 (F32 / F16 / BF16) は expert GEMV の対象外なので同じ規則で回す。
            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
                let rb = row_bytes_any(self.ty, self.k)?;
                let b = self.blocks.as_slice();
                // task 65: `DENNO_CPU_GEMV_PAR=1` のときだけ、行方向を rayon で分ける。
                // 既定 off は従来経路 (下の feature="par" の分割も含めて 1 行も変えない)。
                // 行は独立・行内の加算順は不変なので、分割の有無で値はビット一致する。
                if cpu_gemv_par_enabled() && self.k >= 256 && cpu_gemv_par_applies(self.n, self.k) {
                    gemv_rows_par(self.ty, b, rb, self.k, x, out);
                    return Ok(());
                }
                // P-MOE-G11: 行を rayon で分割する。行は独立で、行内の逐次加算順は
                // 変えないので、分割の有無で結果はビット一致する (従来の逐次版と
                // 同じ `dequant_row_into` + 同じ順の内積を通る)。
                // MoE のルータ (`ffn_gate_inp`, F32, 2560x512 = 5.2 MB) がここを通り、
                // 1 層あたり 1.07 ms かかっていた (moe の 42%)。
                #[cfg(feature = "par")]
                if self.n >= 64 && self.k >= 256 {
                    use rayon::prelude::*;
                    let threads = rayon::current_num_threads().max(1);
                    let rows = self.n.div_ceil(threads).max(32).min(self.n);
                    let bytes = rows * rb;
                    b.par_chunks(bytes).zip(out.par_chunks_mut(rows)).for_each(
                        |(bb, oo)| {
                            let mut row = vec![0f32; self.k];
                            for (r, slot) in oo.iter_mut().enumerate() {
                                dequant_row_into(self.ty, &bb[r * rb..(r + 1) * rb], &mut row);
                                let mut acc = 0f32;
                                for i in 0..self.k {
                                    acc += row[i] * x[i];
                                }
                                *slot = acc;
                            }
                        },
                    );
                    return Ok(());
                }
                let mut row = vec![0f32; self.k];
                for r in 0..self.n {
                    dequant_row_into(self.ty, &b[r * rb..(r + 1) * rb], &mut row);
                    let mut acc = 0f32;
                    for i in 0..self.k {
                        acc += row[i] * x[i];
                    }
                    out[r] = acc;
                }
                Ok(())
            }
            other => bail!("GgufLinear::gemv: 未対応の量子型 {other:?}"),
        }
    }

    /// 行 j を f32 に展開する (検証・デバッグ用)。
    pub fn dequant_row(&self, j: usize) -> Result<Vec<f32>> {
        if j >= self.n {
            bail!("GgufLinear::dequant_row: 行 {j} が n={} の範囲外", self.n);
        }
        let rb = row_bytes_any(self.ty, self.k)?;
        let b = self.blocks.as_slice();
        let mut out = vec![0f32; self.k];
        dequant_row_into(self.ty, &b[j * rb..(j + 1) * rb], &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! 実 GGUF はこのコンテナから見えないのでブロックはテスト内で組み立てる。
    //! 期待値は「全行列を f32 へ展開してから素直な逐次内積」を独立に書き下した
    //! 参照ループと `to_bits()` で比較する (融合実装のレイアウト検算)。

    use super::*;
    use crate::kquant::dequant_row;

    const K: usize = 256;
    const N: usize = 5;

    /// 決定的な擬似乱数バイト。
    fn pat(i: usize, seed: u32) -> u8 {
        let mut x = (i as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(seed.wrapping_mul(40_503));
        x ^= x >> 13;
        x = x.wrapping_mul(0x9E37_79B1);
        x ^= x >> 16;
        x as u8
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// 1 ブロックを組み立てる。スケール欄は有限値に固定する。
    fn make_block(ty: GgmlType, seed: u32) -> Vec<u8> {
        let ts = ty.type_size();
        let mut b: Vec<u8> = (0..ts).map(|i| pat(i, seed)).collect();
        match ty {
            GgmlType::Q8_0 => b[0..2].copy_from_slice(&f16_bytes(0.125)),
            GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => {
                b[0..2].copy_from_slice(&f16_bytes(0.125));
                b[2..4].copy_from_slice(&f16_bytes(0.0625));
            }
            GgmlType::F32 => {
                for i in 0..(ts / 4) {
                    let v = ((pat(i, seed) as f32) / 128.0) - 1.0;
                    b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            GgmlType::F16 | GgmlType::Bf16 => {
                for i in 0..(ts / 2) {
                    let v = ((pat(i, seed) as f32) / 128.0) - 1.0;
                    if ty == GgmlType::F16 {
                        b[i * 2..i * 2 + 2].copy_from_slice(&f16_bytes(v));
                    } else {
                        let bits = (v.to_bits() >> 16) as u16;
                        b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
            }
            _ => {}
        }
        b
    }

    fn make_matrix(ty: GgmlType, k: usize, n: usize, seed: u32) -> Vec<u8> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert_eq!(k % bs, 0);
        let nb = k / bs;
        let mut out = Vec::with_capacity(n * nb * ts);
        for r in 0..n {
            for blk in 0..nb {
                out.extend_from_slice(&make_block(ty, seed.wrapping_add((r * 31 + blk) as u32)));
            }
        }
        out
    }

    fn make_x(k: usize, seed: u32) -> Vec<f32> {
        (0..k)
            .map(|i| ((pat(i, seed) as f32) / 128.0) - 1.0)
            .collect()
    }

    /// 独立参照: 全行列を f32 に展開してから素直な逐次内積。
    fn naive_gemv(ty: GgmlType, w: &[u8], k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        let rb = (k / bs) * ts;
        let mut out = vec![0f32; n];
        for r in 0..n {
            let row = dequant_row(ty, &w[r * rb..(r + 1) * rb], k);
            let mut acc = 0f32;
            for i in 0..k {
                acc += row[i] * x[i];
            }
            out[r] = acc;
        }
        out
    }

    fn check(ty: GgmlType) {
        let w = make_matrix(ty, K, N, 7);
        let x = make_x(K, 11);
        let lin = GgufLinear::from_slice(ty, K, N, &w).expect("GgufLinear");
        assert_eq!(lin.k(), K);
        assert_eq!(lin.n(), N);
        assert_eq!(lin.ty(), ty);
        assert_eq!(lin.weight_bytes(), w.len());

        let mut got = vec![0f32; N];
        lin.gemv(&x, &mut got).expect("gemv");
        let want = naive_gemv(ty, &w, K, N, &x);
        for r in 0..N {
            assert_eq!(got[r].to_bits(), want[r].to_bits(), "{ty:?}: 行 {r}");
        }
    }

    #[test]
    fn q8_0_matches_naive() {
        check(GgmlType::Q8_0);
    }

    #[test]
    fn q5_1_matches_naive() {
        check(GgmlType::Q5_1);
    }

    #[test]
    fn q4_k_matches_naive() {
        check(GgmlType::Q4K);
    }

    #[test]
    fn q5_k_matches_naive() {
        check(GgmlType::Q5K);
    }

    #[test]
    fn f32_matches_naive() {
        check(GgmlType::F32);
    }

    #[test]
    fn f16_matches_naive() {
        check(GgmlType::F16);
    }

    #[test]
    fn bf16_matches_naive() {
        check(GgmlType::Bf16);
    }

    #[test]
    fn dequant_row_is_consistent_with_matrix_layout() {
        let w = make_matrix(GgmlType::Q8_0, K, N, 3);
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w).unwrap();
        for r in 0..N {
            let row = lin.dequant_row(r).unwrap();
            let bs = GgmlType::Q8_0.blck_size();
            let ts = GgmlType::Q8_0.type_size();
            let rb = (K / bs) * ts;
            let direct = dequant_row(GgmlType::Q8_0, &w[r * rb..(r + 1) * rb], K);
            assert_eq!(row, direct);
        }
        assert!(lin.dequant_row(N).is_err());
    }

    #[test]
    fn rejects_bad_shapes() {
        let w = make_matrix(GgmlType::Q8_0, K, N, 5);
        // バイト数不足
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w[..w.len() - 1]).is_err());
        // 行数とバイト数の不一致
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K, N + 1, &w).is_err());
        // ブロック境界でない k
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K + 1, N, &w).is_err());

        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w).unwrap();
        let x = make_x(K, 1);
        let mut out = vec![0f32; N];
        assert!(lin.gemv(&x[..K - 1], &mut out).is_err());
        assert!(lin.gemv(&x, &mut out[..N - 1]).is_err());
    }

    #[test]
    fn zero_input_gives_zero() {
        let w = make_matrix(GgmlType::Q4K, K, N, 9);
        let lin = GgufLinear::from_slice(GgmlType::Q4K, K, N, &w).unwrap();
        let x = vec![0f32; K];
        let mut out = vec![1.0f32; N];
        lin.gemv(&x, &mut out).unwrap();
        for r in 0..N {
            assert_eq!(out[r], 0.0, "行 {r}");
        }
    }

    // ---- task 65: CPU GEMV の行並列 (`DENNO_CPU_GEMV_PAR=1`) ---------------
    //
    // 新経路は「行を rayon で分けるだけで、1 行の内積は i 昇順の逐次加算のまま」。
    // そのため旧経路 (逐次 / 既存の feature="par" の分割) と **f32 のビットで
    // 一致**する。ここでは実形状 4 本 (router / ssm_beta / ssm_alpha / hc_inject)
    // としきい値の境界を固定する。上書き・カウンタ・`par_test_lock` は
    // `super` (linear_gguf) 側にあり、crate 内の他のテストと共有する。

    /// 上書きを必ず env 読みに戻す (panic しても戻る)。
    struct ParGuard;
    impl ParGuard {
        fn new() -> Self {
            set_cpu_gemv_par_for_test(None);
            set_par_min_rows_for_test(None);
            Self
        }
    }
    impl Drop for ParGuard {
        fn drop(&mut self) {
            set_cpu_gemv_par_for_test(None);
            set_par_min_rows_for_test(None);
        }
    }

    /// 実形状 (k, n) を作る。F32 は 1 要素 4 バイトなので k*4*n バイト。
    fn f32_lin(k: usize, n: usize, seed: u32) -> (GgufLinear, Vec<f32>) {
        let w = make_matrix(GgmlType::F32, k, n, seed);
        let lin = GgufLinear::from_slice(GgmlType::F32, k, n, &w).expect("GgufLinear");
        (lin, make_x(k, seed.wrapping_add(1)))
    }

    #[test]
    fn cpu_gemv_par_env_values() {
        // 既定 on。`0` / `off` / `false` のときだけ無効。
        for v in [None, Some("1"), Some("true"), Some("")] {
            assert!(cpu_gemv_par_from_value(v), "value={v:?}");
        }
        for v in [Some("0"), Some("off"), Some("false")] {
            assert!(!cpu_gemv_par_from_value(v), "value={v:?}");
        }
    }

    /// 実形状 4 本で旧経路と新経路がビット一致し、期待どおりの側に落ちること。
    #[test]
    fn cpu_gemv_par_bitwise_on_real_shapes() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        // (k, n, seed, 名前, 並列化されるか)
        let cases = [
            (2560usize, 512usize, 11u32, "router (ffn_gate_inp)", true),
            (2560, 48, 13, "ssm_beta / ssm_alpha", true),
            (10240, 4, 17, "hc_inject", true),
            (10240, 32, 19, "hc_up/inject (k=10240, 32 行)", true),
            (2560, 4, 47, "しきい値未満 (2560->4)", false),
        ];
        for (k, n, seed, name, want_par) in cases {
            let (lin, x) = f32_lin(k, n, seed);

            // 旧: env off (既定)。既存の feature="par" の分割はそのまま通る。
            set_cpu_gemv_par_for_test(Some(false));
            par_reset();
            let mut old = vec![-1.0e30f32; n];
            lin.gemv(&x, &mut old).expect(name);
            assert_eq!(par_launches(), 0, "{name}: off で新経路が立った");

            // 新: env on。
            set_cpu_gemv_par_for_test(Some(true));
            par_reset();
            let mut new = vec![1.0e30f32; n];
            lin.gemv(&x, &mut new).expect(name);
            let launches = par_launches();
            if want_par {
                assert_eq!(launches, 1, "{name}: 並列領域が 1 回でない");
                // 行ごとに領域を立てていないこと (チャンク数 <= 行数)。
                let tasks = par_tasks();
                assert!(tasks >= 1 && tasks <= n, "{name}: tasks={tasks} n={n}");
                if n >= 32 {
                    assert!(tasks < n, "{name}: 行ごとにタスクを立てている tasks={tasks}");
                }
            } else {
                assert_eq!(launches, 0, "{name}: しきい値未満で並列化された");
            }

            // ビット一致 (f32)。
            for r in 0..n {
                assert_eq!(
                    old[r].to_bits(),
                    new[r].to_bits(),
                    "{name}: 行 {r} が違う (old={} new={})",
                    old[r],
                    new[r]
                );
            }
            // どちらも独立参照 (全行 f32 展開 + 素直な逐次内積) と一致する。
            let w = lin.blocks().as_slice();
            let want = naive_gemv(GgmlType::F32, w, k, n, &x);
            for r in 0..n {
                assert_eq!(new[r].to_bits(), want[r].to_bits(), "{name}: 参照と違う 行 {r}");
            }
        }
    }

    /// 1 行だけの形は並列化しない側に落ち、値は逐次と一致する。
    #[test]
    fn cpu_gemv_par_skips_single_row() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (k, n) = (2560usize, 1usize);
        let (lin, x) = f32_lin(k, n, 23);
        set_cpu_gemv_par_for_test(Some(true));
        par_reset();
        let mut got = vec![0f32; n];
        lin.gemv(&x, &mut got).expect("gemv");
        assert_eq!(par_launches(), 0, "1 行で並列領域を立てた");
        let want = naive_gemv(GgmlType::F32, lin.blocks().as_slice(), k, n, &x);
        assert_eq!(got[0].to_bits(), want[0].to_bits());
    }

    /// しきい値の境界。行数の下限 (8) と積和の下限 (32768) の両方を見る。
    #[test]
    fn cpu_gemv_par_threshold_boundary() {
        // 純粋関数の側 (起動カウンタを立てずに境界を固定する)。
        set_par_min_rows_for_test(None);
        // 行数 >= 8 は無条件。
        assert!(cpu_gemv_par_applies(8, 2560));
        assert!(!cpu_gemv_par_applies(7, 2560));
        // 行数 < 8 でも積和 >= 32768 なら並列 (hc_inject 4×10240 = 40960)。
        assert!(cpu_gemv_par_applies(4, 10240));
        assert!(!cpu_gemv_par_applies(3, 10240)); // 30720 < 32768
        assert!(!cpu_gemv_par_applies(4, 2560)); // 10240 < 32768
        assert!(!cpu_gemv_par_applies(2, 10240)); // 20480 < 32768 (実測でも損)

        let _g = par_test_lock();
        let _p = ParGuard::new();
        let k = 2560usize;
        set_cpu_gemv_par_for_test(Some(true));

        let n_below = CPU_GEMV_PAR_MIN_ROWS - 1;
        let (lin, x) = f32_lin(k, n_below, 29);
        par_reset();
        let mut out = vec![0f32; n_below];
        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 0, "{n_below} 行で並列化した");

        let (lin, x) = f32_lin(k, CPU_GEMV_PAR_MIN_ROWS, 31);
        par_reset();
        let mut out = vec![0f32; CPU_GEMV_PAR_MIN_ROWS];
        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 1, "しきい値ちょうどで並列化しない");
    }

    /// 1 呼び出し = 1 並列領域。行数ぶん起動しないこと (連続 2 回で 2)。
    #[test]
    fn cpu_gemv_par_one_region_per_call() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (k, n) = (2560usize, 512usize);
        let (lin, x) = f32_lin(k, n, 37);
        set_cpu_gemv_par_for_test(Some(true));
        let mut out = vec![0f32; n];

        par_reset();
        lin.gemv(&x, &mut out).expect("gemv");
        let tasks_one = par_tasks();
        assert_eq!(par_launches(), 1, "1 呼び出しで {tasks_one} タスク / 領域 1 のはず");
        assert!(tasks_one < n, "行ごとにタスクを立てている (tasks={tasks_one})");
        let threads = rayon::current_num_threads().max(1);
        let rows = n.div_ceil(threads).max(1);
        assert_eq!(tasks_one, n.div_ceil(rows), "分割数が想定と違う");

        lin.gemv(&x, &mut out).expect("gemv");
        assert_eq!(par_launches(), 2, "2 回目で領域が 2 にならない");
    }

    /// off (既定) では新経路が発動しないこと (従来経路そのまま)。
    #[test]
    fn cpu_gemv_par_off_keeps_old_path() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let (k, n) = (2560usize, 512usize);
        let (lin, x) = f32_lin(k, n, 41);
        set_cpu_gemv_par_for_test(Some(false));
        par_reset();
        let mut got = vec![0f32; n];
        lin.gemv(&x, &mut got).expect("gemv");
        assert_eq!(par_launches(), 0);
        let want = naive_gemv(GgmlType::F32, lin.blocks().as_slice(), k, n, &x);
        for r in 0..n {
            assert_eq!(got[r].to_bits(), want[r].to_bits(), "行 {r}");
        }
    }

    /// 旧/新の最小時間を交互に取る (片方だけ先に回すとキャッシュ状態が揃わない)。
    /// 出力バッファはここで持つ (両方の closure が同じ `out` を借りるため)。
    fn best_pair(
        reps: usize,
        n: usize,
        mut old: impl FnMut(&mut [f32]),
        mut new: impl FnMut(&mut [f32]),
    ) -> (f64, f64) {
        let mut out = vec![0f32; n];
        let (mut bo, mut bn) = (f64::INFINITY, f64::INFINITY);
        for _ in 0..reps {
            let t = std::time::Instant::now();
            old(&mut out);
            bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
            let t = std::time::Instant::now();
            new(&mut out);
            bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
        }
        (bo, bn)
    }

    /// 実形状 4 本の 1 呼び出しを 100 回ずつ、旧 (off) / 新 (on) で測る。
    ///
    /// `RAYON_NUM_THREADS=16 cargo test --release -p denno-engine --lib -- \
    ///  --ignored --nocapture bench_cpu_gemv_par_real_shapes`
    #[test]
    #[ignore = "実形状の CPU マイクロベンチ (手動)"]
    fn bench_cpu_gemv_par_real_shapes() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let reps = 100usize;
        println!(
            "\n=== CPU GEMV 行並列: 実形状 1 呼び出し (reps={reps}, rayon threads={}) ===",
            rayon::current_num_threads()
        );
        println!(
            "{:<34} {:>9} {:>9} {:>8} {:>10} {:>6}",
            "shape", "old_ms", "new_ms", "gain", "MAC", "GMAC/s"
        );
        // (k, n, seed, 名前) — 実モデルの 4 形 (router / ssm_beta / ssm_alpha / hc_inject)
        for (k, n, seed, name) in [
            (2560usize, 512usize, 11u32, "router ffn_gate_inp 2560->512"),
            (2560, 48, 13, "ssm_beta/alpha 2560->48"),
            (10240, 4, 17, "hc_inject 10240->4"),
            (2560, 2048, 43, "参考: 2560->2048"),
        ] {
            let (lin, x) = f32_lin(k, n, seed);
            set_cpu_gemv_par_for_test(Some(false));
            let (old_ms, new_ms) = best_pair(
                reps,
                n,
                |out| {
                    lin.gemv(&x, out).expect("old");
                    std::hint::black_box(out);
                },
                |out| {
                    set_cpu_gemv_par_for_test(Some(true));
                    lin.gemv(&x, out).expect("new");
                    std::hint::black_box(out);
                    set_cpu_gemv_par_for_test(Some(false));
                },
            );
            let us = old_ms * 1e3;
            let mac = (k * n) as f64;
            println!(
                "{:<34} {:>9.3} {:>9.3} {:>7.2}x {:>10.0} {:>6.1}",
                name,
                us,
                new_ms * 1e3,
                if new_ms > 0.0 { old_ms / new_ms } else { 0.0 },
                mac,
                mac / (new_ms * 1e6)
            );
        }
    }

    /// しきい値の根拠: 行数を 1..64 で振り、旧 (逐次寄り) と新 (行並列) を測る。
    /// 新は**しきい値を 1 に強制**して、小さい形でも実際に並列を立てた値を取る
    /// (しきい値の損得はこの表で決める)。
    ///
    /// `RAYON_NUM_THREADS=8 cargo test --release -p denno-engine --lib -- \
    ///  --ignored --nocapture bench_cpu_gemv_par_threshold_sweep`
    #[test]
    #[ignore = "しきい値の根拠を取る CPU マイクロベンチ (手動)"]
    fn bench_cpu_gemv_par_threshold_sweep() {
        let _g = par_test_lock();
        let _p = ParGuard::new();
        let reps = 50usize;
        println!(
            "\n=== 行並列のしきい値 (reps={reps}, rayon threads={}) ===",
            rayon::current_num_threads()
        );
        println!(
            "{:<6} {:>7} {:>10} {:>10} {:>8} {:>9} {:>9}",
            "k", "rows", "old_us", "new_us", "gain", "old_MAC/us", "new_MAC/us"
        );
        // 新はしきい値 1 (= 何行でも並列領域を立てる) で測る。
        set_par_min_rows_for_test(Some(1));
        for k in [2560usize, 10240] {
            for n in [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64] {
                let (lin, x) = f32_lin(k, n, 101 + n as u32);
                set_cpu_gemv_par_for_test(Some(false));
                let (old_ms, new_ms) = best_pair(
                    reps,
                    n,
                    |out| {
                        lin.gemv(&x, out).expect("old");
                        std::hint::black_box(out);
                    },
                    |out| {
                        set_cpu_gemv_par_for_test(Some(true));
                        lin.gemv(&x, out).expect("new");
                        std::hint::black_box(out);
                        set_cpu_gemv_par_for_test(Some(false));
                    },
                );
                let mac = (k * n) as f64;
                println!(
                    "{:<6} {:>7} {:>10.2} {:>10.2} {:>7.2}x {:>9.2} {:>9.2}",
                    k,
                    n,
                    old_ms * 1e3,
                    new_ms * 1e3,
                    if new_ms > 0.0 { old_ms / new_ms } else { 0.0 },
                    mac / (old_ms * 1e6),
                    mac / (new_ms * 1e6)
                );
            }
        }
    }
}
