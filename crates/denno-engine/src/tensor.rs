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
