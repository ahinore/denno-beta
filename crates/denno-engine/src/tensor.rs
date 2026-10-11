//! 最小限のテンソル演算。重い GEMM は matrixmultiply(sgemm) + rayon で並列化する。

use rayon::prelude::*;

/// c = a @ b  (a: row-major [m,k], b: row-major [k,n], 戻り値: row-major [m,n])
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    par_sgemm(a, b, n as isize, &mut c, m, k, n);
    c
}

/// c = a @ b^T (a: [m,k], b: row-major [n,k] = nn.Linear の重み格納形状)
pub fn matmul_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    par_sgemm(a, b, 1isize, &mut c, m, k, n);
    c
}

/// a は常に row-major [m,k] (rs=k, cs=1)。
/// rs_b は b の行ストライド: row-major [k,n] なら n、[n,k] (b^T使用) なら 1。
fn par_sgemm(a: &[f32], b: &[f32], rs_b: isize, c: &mut [f32], m: usize, k: usize, n: usize) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let rs_a = k as isize;
    let cs_a = 1isize;
    let cs_b = if rs_b == 1 { k as isize } else { 1isize };
    // 行方向にチャンク分割して並列実行
    let rows_per_task = ((m + 7) / 8).clamp(1, m);
    c.par_chunks_mut(rows_per_task * n)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let row0 = ti * rows_per_task;
            let rows = (m - row0).min(rows_per_task);
            let a_ptr = unsafe { a.as_ptr().add(row0 * k) };
            let c_ptr = chunk.as_mut_ptr();
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, n, 1.0,
                    a_ptr, rs_a, cs_a,
                    b.as_ptr(), rs_b, cs_b,
                    0.0,
                    c_ptr, n as isize, 1,
                );
            }
        });
}

/// c += alpha * a @ b^T (a: [m,k] row-major, b: [n,k] row-major, c: [m,n] row-major)
/// attention の scores 計算用 (既存値に加算)
pub fn sgemm_acc(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize, alpha: f32) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    let rows_per_task = ((m + 7) / 8).clamp(1, m);
    c.par_chunks_mut(rows_per_task * n)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let row0 = ti * rows_per_task;
            let rows = (m - row0).min(rows_per_task);
            let a_ptr = unsafe { a.as_ptr().add(row0 * k) };
            let c_ptr = chunk.as_mut_ptr();
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, n, alpha,
                    a_ptr, k as isize, 1isize,
                    b.as_ptr(), 1isize, k as isize,
                    1.0,
                    c_ptr, n as isize, 1,
                );
            }
        });
}

/// c[j] = dot(a, b[j*k..][..k]) を出力分割で並列に計算 (decode 用 matvec)。
/// m=1 の行列積は A 側で分割できないため、出力列 (= B の行) で並列化する。
/// b: [n,k] row-major (nn.Linear 形状)。
pub fn matvec_bt(a: &[f32], b: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n];
    c.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(j, cj)| {
            let row = &b[j * k..][..k];
            let mut s = 0.0f32;
            for d in 0..k {
                s += a[d] * row[d];
            }
            *cj = s;
        });
    c
}

/// c[j] = scale[j] * Σ_d a[d] * q[j*k+d] — int8 weight-only の fused matvec
/// (decode 用、matvec_bt の int8 版)。
/// q: [n,k] row-major (nn.Linear 形式)、scale: [n] は出力チャネルごとのデ量子化係数。
/// int8 重みを直接読むため帯域が fp32 の約 1/4 で済む (P3)。
/// 各出力要素を固定順序で畳み込むため決定論的 (量子化パスの exact decoding)。
pub fn matvec_bt_i8(a: &[f32], q: &[i8], scale: &[f32], k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(q.len(), k * n);
    debug_assert_eq!(scale.len(), n);
    let mut c = vec![0.0f32; n];
    c.par_iter_mut()
        .enumerate()
        .with_min_len(64)
        .for_each(|(j, cj)| {
            let row = &q[j * k..][..k];
            let mut s = 0.0f32;
            for d in 0..k {
                s += a[d] * row[d] as f32;
            }
            *cj = s * scale[j];
        });
    c
}

/// c = a @ q^T (int8 weight-only の prefill 用 GEMM、q: [n,k] row-major)。
/// int8 → f32 へデ量子化してから既存の並列 sgemm に渡す。
/// decode 経路 (matvec_bt_i8) と異なりデ量子化バッファを一度作るため
/// 追加のメモリ書き込みが発生するが、prefill は attention sgemm が支配的なため
/// 実用上の影響は小さい。
pub fn matmul_bt_i8(a: &[f32], q: &[i8], scale: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(q.len(), k * n);
    debug_assert_eq!(scale.len(), n);
    let mut w = vec![0.0f32; k * n];
    for j in 0..n {
        let s = scale[j];
        for d in 0..k {
            w[j * k + d] = q[j * k + d] as f32 * s;
        }
    }
    matmul_bt(a, &w, m, k, n)
}

/// 各行にバイアスを加算 (x: row-major [rows, cols])
pub fn add_bias(x: &mut [f32], bias: &[f32], cols: usize) {
    for row in x.chunks_mut(cols) {
        for (v, b) in row.iter_mut().zip(bias) {
            *v += b;
        }
    }
}

/// LayerNorm (GPT-2系)。input: [rows, cols] → 新しいバッファ
pub fn layernorm(x: &[f32], w: &[f32], b: &[f32], cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks(cols).zip(out.chunks_mut(cols)) {
        let mean = row_in.iter().sum::<f32>() / cols as f32;
        let var = row_in.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..cols {
            row_out[i] = (row_in[i] - mean) * inv * w[i] + b[i];
        }
    }
    out
}

/// RMSNorm (Llama/Qwen系)
pub fn rmsnorm(x: &[f32], w: &[f32], cols: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (row_in, row_out) in x.chunks(cols).zip(out.chunks_mut(cols)) {
        let ss = row_in.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv = 1.0 / (ss + eps).sqrt();
        for i in 0..cols {
            row_out[i] = row_in[i] * inv * w[i];
        }
    }
    out
}

/// gelu_new (GPT-2 の tanh 近似)
pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        let t = *v;
        *v = 0.5 * t * (1.0 + (0.7978845608028654 * (t + 0.044715 * t * t * t)).tanh());
    }
}

/// SiLU (Llama/Qwen系)
pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// 1行分のsoftmax
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// RoPE (rotate_half 方式: Llama/Qwen2)
/// x: [rows, heads*dh] を in-place で回転。theta は rope_theta。
pub fn rope_rows(x: &mut [f32], rows: usize, heads: usize, dh: usize, pos0: usize, theta: f32) {
    let half = dh / 2;
    for i in 0..rows {
        let pos = (pos0 + i) as f32;
        for h in 0..heads {
            let base = i * heads * dh + h * dh;
            for d in 0..half {
                let ang = pos * theta.powf(-2.0 * d as f32 / dh as f32);
                let (c, s) = (ang.cos(), ang.sin());
                let x1 = x[base + d];
                let x2 = x[base + half + d];
                x[base + d] = x1 * c - x2 * s;
                x[base + half + d] = x1 * s + x2 * c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 素朴な (遅い) リファレンス実装: c = a @ b
    fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..k {
                    s += a[i * k + d] * b[d * n + j];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    /// 素朴なリファレンス実装: c = a @ b^T (b: [n,k] row-major)
    fn naive_matmul_bt(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for d in 0..k {
                    s += a[i * k + d] * b[j * k + d];
                }
                c[i * n + j] = s;
            }
        }
        c
    }

    /// 決定論的な疑似乱数 (LCG) でテスト用行列を作る
    fn pseudo_random(len: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..len)
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 16) % 2000) as f32 / 1000.0 - 1.0 // [-1, 1)
            })
            .collect()
    }

    #[test]
    fn matmul_known_values() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let i = [1.0, 0.0, 0.0, 1.0];
        assert_eq!(matmul(&a, &i, 2, 2, 2), vec![1.0, 2.0, 3.0, 4.0]);

        let b = [1.0, 1.0, 2.0, 2.0];
        // [1 2; 3 4] @ [1 1; 2 2] = [5 5; 11 11]
        assert_eq!(matmul(&a, &b, 2, 2, 2), vec![5.0, 5.0, 11.0, 11.0]);
    }

    #[test]
    fn matmul_matches_naive() {
        // 並列チャンク境界 (rows_per_task) をまたぐサイズを含める
        for (m, k, n) in [(3usize, 4usize, 5usize), (1, 1, 1), (9, 7, 2), (16, 3, 13)] {
            let a = pseudo_random(m * k, 1);
            let b = pseudo_random(k * n, 2);
            let got = matmul(&a, &b, m, k, n);
            let want = naive_matmul(&a, &b, m, k, n);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-3, "m={m} k={k} n={n}: {g} != {w}");
            }
        }
    }

    #[test]
    fn matmul_bt_matches_naive() {
        for (m, k, n) in [(3usize, 4usize, 5usize), (1, 6, 1), (9, 7, 2), (17, 3, 13)] {
            let a = pseudo_random(m * k, 3);
            let b = pseudo_random(n * k, 4); // nn.Linear 形状 [n,k]
            let got = matmul_bt(&a, &b, m, k, n);
            let want = naive_matmul_bt(&a, &b, m, k, n);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-3, "m={m} k={k} n={n}: {g} != {w}");
            }
        }
    }

    #[test]
    fn matvec_bt_matches_matmul_bt_single_row() {
        // decode カーネル (m=1) は GEMM の 1 行目と完全一致しなければならない
        for (k, n) in [(8usize, 16usize), (64usize, 200usize)] {
            let a = pseudo_random(k, 5);
            let b = pseudo_random(n * k, 6);
            let by_matvec = matvec_bt(&a, &b, k, n);
            let by_gemm = matmul_bt(&a, &b, 1, k, n);
            for (g, w) in by_matvec.iter().zip(&by_gemm) {
                assert!((g - w).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn matvec_bt_known_dots() {
        // a = [1, 2], b の行: [1,0], [0,1], [1,1], [2,2]
        let a = [1.0, 2.0];
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
        // dot([1,2], [1,0])=1, dot([1,2],[0,1])=2, dot([1,2],[1,1])=3, dot([1,2],[2,2])=6
        assert_eq!(matvec_bt(&a, &b, 2, 4), vec![1.0, 2.0, 3.0, 6.0]);
    }

    #[test]
    fn sgemm_acc_accumulates() {
        // c += alpha * a @ b^T の accumulator 形式 (attention の scores 計算用)
        let a = [1.0, 2.0, 3.0, 4.0]; // [2,2]
        let b = [1.0, 0.0, 1.0, 1.0]; // [2,2] → a@b^T = [1,3; 3,7]
        let mut c = vec![10.0, 20.0, 30.0, 40.0];
        sgemm_acc(&a, &b, &mut c, 2, 2, 2, 2.0);
        // 2*[1 3; 3 7] + [10 20; 30 40] = [12 26; 36 54]
        assert_eq!(c, vec![12.0, 26.0, 36.0, 54.0]);
    }

    #[test]
    fn sgemm_acc_zero_dims_are_noop() {
        let mut c = vec![1.0f32; 6];
        sgemm_acc(&[], &[], &mut c, 0, 3, 2, 1.0);
        sgemm_acc(&[1.0], &[1.0], &mut c, 2, 0, 3, 1.0);
        sgemm_acc(&[1.0], &[1.0], &mut c, 2, 3, 0, 1.0);
        assert_eq!(c, vec![1.0; 6]);
    }

    #[test]
    fn add_bias_applies_per_row() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        add_bias(&mut x, &[0.5, -0.5], 2);
        assert_eq!(x, vec![1.5, 1.5, 3.5, 3.5]);
    }

    #[test]
    fn layernorm_known_values() {
        // x=[0,2]: mean=1, var=1 → out = (x-1)/sqrt(1+eps) * w + b
        let x = [0.0, 2.0];
        let out = layernorm(&x, &[1.0, 1.0], &[0.0, 0.0], 2, 0.0);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);

        // アフィン: w=2, b=1
        let out = layernorm(&x, &[2.0, 2.0], &[1.0, 1.0], 2, 0.0);
        assert!((out[0] - (-1.0)).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn layernorm_normalizes_each_row() {
        // 複数行: 各行の平均≈0、分散≈1
        let x = pseudo_random(6 * 8, 7);
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let out = layernorm(&x, &w, &b, 8, 1e-5);
        for row in out.chunks(8) {
            let mean = row.iter().sum::<f32>() / 8.0;
            assert!(mean.abs() < 1e-4);
            let var = row.iter().map(|v| v * v).sum::<f32>() / 8.0;
            assert!((var - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn rmsnorm_known_values() {
        // x=[3,4]: ss=(9+16)/2=12.5 → inv=1/sqrt(12.5)
        let x = [3.0, 4.0];
        let out = rmsnorm(&x, &[1.0, 1.0], 2, 0.0);
        let inv = 1.0 / 12.5f32.sqrt();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_is_stable() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        // 単調: 最大値が最大確率
        assert!(x[2] > x[1] && x[1] > x[0]);
        // softmax は正規化される: x[0] = e^-2 / (e^-2 + e^-1 + 1)
        let denom = (-2.0f32).exp() + (-1.0f32).exp() + 1.0;
        assert!((x[0] - (-2.0f32).exp() / denom).abs() < 1e-6);

        // 巨大な値でも NaN にならない (max-shift の確認)
        let mut big = vec![1e9, 1e9 + 1.0, 1e9 + 2.0];
        softmax(&mut big);
        assert!(big.iter().all(|v| v.is_finite()));
        assert!((big.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn activations_known_values() {
        // gelu_new (tanh近似): gelu(0)=0, gelu(1)≈0.8412
        let mut x = vec![0.0f32, 1.0];
        gelu_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 0.8412).abs() < 1e-3);

        // silu: silu(0)=0, silu(1)=1/(1+e^-1)≈0.7311, silu(-1)≈-0.2689
        let mut x = vec![0.0f32, 1.0, -1.0];
        silu_inplace(&mut x);
        assert_eq!(x[0], 0.0);
        assert!((x[1] - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
        // silu(-1) = -1/(1+e^1) = -e^-1/(1+e^-1)
        assert!((x[2] + (-1.0f32).exp() / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    }

    #[test]
    fn rope_identity_at_position_zero() {
        // pos=0 では角度が 0 → 回転は恒等写像
        let mut x = vec![0.5f32, -1.5];
        rope_rows(&mut x, 1, 1, 2, 0, 10000.0);
        assert!((x[0] - 0.5).abs() < 1e-6);
        assert!((x[1] - (-1.5)).abs() < 1e-6);
    }

    #[test]
    fn rope_rotates_by_expected_angle() {
        // dh=2, pos=1, theta=10000: d=0 の角度 = 1 * theta^0 = 1 (rad)
        // x=[1,0] → [cos1, sin1]
        let mut x = vec![1.0f32, 0.0];
        rope_rows(&mut x, 1, 1, 2, 1, 10000.0);
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);

        // 回転はノルムを保存する
        let src = pseudo_random(2 * 4, 8); // rows=2, heads=1, dh=4
        let mut x = src.clone();
        rope_rows(&mut x, 2, 1, 4, 3, 10000.0);
        for r in 0..2 {
            let n0: f32 = src[r * 4..(r + 1) * 4].iter().map(|v| v * v).sum();
            let n1: f32 = x[r * 4..(r + 1) * 4].iter().map(|v| v * v).sum();
            assert!((n0 - n1).abs() < 1e-4);
        }
    }

    #[test]
    fn rope_applies_per_head() {
        // heads=2 のとき各ヘッドが独立に回転する (dh=2, pos=0 は恒等なので pos=1 で確認)
        let mut x = vec![1.0f32, 0.0, 1.0, 0.0]; // rows=1, heads=2, dh=2
        rope_rows(&mut x, 1, 2, 2, 1, 10000.0);
        // ヘッドごとに同じ角度 (dh が同じなので) → 両ヘッドとも [cos1, sin1]
        assert!((x[0] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0f32.sin()).abs() < 1e-6);
        assert!((x[2] - 1.0f32.cos()).abs() < 1e-6);
        assert!((x[3] - 1.0f32.sin()).abs() < 1e-6);
    }
}
