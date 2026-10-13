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

/// 量子化モード。デフォルトは Fp32 (従来動作を完全維持)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantMode {
    /// 重みを f32 のまま使用 (デフォルト)。HANDOFF §5 の exactness 基準はこちら。
    #[default]
    Fp32,
    /// 線形層の重みを per-channel scale の int8 に量子化 (weight-only)。
    /// 決定論は維持されるが fp32 と同一出力にはならない。
    Int8,
}

impl QuantMode {
    /// CLI 文字列 ("fp32" / "int8") から解析する
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "fp32" | "f32" | "none" => Ok(QuantMode::Fp32),
            "int8" | "i8" | "q8" => Ok(QuantMode::Int8),
            other => bail!("--quant の値は fp32 または int8 (got: {other})"),
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            QuantMode::Fp32 => "fp32",
            QuantMode::Int8 => "int8",
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

impl Quant8Linear {
    /// f32 重み (w: [n,k] row-major) を int8 へ量子化する。
    ///
    /// 対称量子化: scale[j] = max(|w[j]|) / 127.0、q = round(w / scale)。
    /// 丸め誤差は各要素 |w - deq| ≤ scale[j] / 2 に収まる。
    pub fn quantize(w: &[f32], k: usize, n: usize) -> Self {
        assert_eq!(w.len(), k * n, "quantize: weight size mismatch");
        let mut qw = vec![0i8; k * n];
        let mut scale = vec![0f32; n];
        for (j, row) in w.chunks_exact(k).enumerate() {
            let maxabs = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // 全行ゼロの場合の 0 除算回避 (scale は任意の値でよい)
            let s = if maxabs > 0.0 { maxabs / 127.0 } else { 1.0 };
            scale[j] = s;
            let inv = 1.0 / s;
            for (q, v) in qw[j * k..][..k].iter_mut().zip(row) {
                // round-half-away-from-zero + 安全側のクランプ (±127 に収める)
                let qi = (*v * inv).round();
                *q = qi.clamp(-127.0, 127.0) as i8;
            }
        }
        Self { w: qw, scale, k: k, n: n, uid: next_uid() }
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
}

impl Linear {
    /// w: [n,k] row-major (nn.Linear 形式) から構築。mode が Int8 ならここで
    /// オフライン量子化する (safetensors ロード直後に呼ばれる)。
    pub fn new(w: Vec<f32>, k: usize, n: usize, mode: QuantMode) -> Self {
        match mode {
            QuantMode::Fp32 => Linear::F32 { w, k, n },
            QuantMode::Int8 => Linear::Q8(Quant8Linear::quantize(&w, k, n)),
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
        }
    }

    /// (k, n) 対
    pub fn dims(&self) -> (usize, usize) {
        match self {
            Linear::F32 { k, n, .. } => (*k, *n),
            Linear::Q8(q) => (q.k, q.n),
        }
    }

    /// 重みの実メモリ量 (bytes)。量子化効果のログ/レポート用。
    pub fn weight_bytes(&self) -> usize {
        match self {
            Linear::F32 { w, .. } => w.len() * 4,
            Linear::Q8(q) => q.weight_bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
