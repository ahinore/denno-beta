use std::time::Instant;
use rayon::prelude::*;

fn bench(name: &str, m: usize, k: usize, n: usize, threads_rows: usize) {
    let a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.01).collect();
    let b: Vec<f32> = (0..n * k).map(|i| (i % 5) as f32 * 0.01).collect(); // [n,k] row-major (bt case)
    let mut c = vec![0.0f32; m * n];
    let t = Instant::now();
    let rows_per_task = ((m + threads_rows - 1) / threads_rows).max(1);
    c.par_chunks_mut(rows_per_task * n)
        .enumerate()
        .for_each(|(ti, chunk)| {
            let row0 = ti * rows_per_task;
            let rows = (m - row0).min(rows_per_task);
            let a_ptr = unsafe { a.as_ptr().add(row0 * k) };
            unsafe {
                matrixmultiply::sgemm(
                    rows, k, n, 1.0,
                    a_ptr, k as isize, 1isize,
                    b.as_ptr(), 1isize, k as isize,
                    0.0,
                    chunk.as_mut_ptr(), n as isize, 1,
                );
            }
        });
    let el = t.elapsed().as_secs_f64();
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    println!("{name}: {el:.3}s  {:.1} GFLOPS  (m={m} k={k} n={n} tasks={threads_rows})", gflop / el);
    // 検算
    let mut sum = 0.0f32;
    for &v in &c { sum += v; }
    println!("  checksum {sum:.2}");
}

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());
    // Qwen2.5-0.5B の代表的な GEMM (prefill 1524 トークン想定)
    bench("mlp gate [1524,896]x[4864,896]^T", 1524, 896, 4864, 8);
    bench("mlp gate 同上 task=24", 1524, 896, 4864, 24);
    bench("mlp gate 同上 task=192(16行ずつ)", 1524, 896, 4864, 96);
    bench("qkv  [1524,896]x[1152,896]^T", 1524, 896, 1152, 8);
    bench("down [1524,4864]x[896,4864]^T", 1524, 4864, 896, 8);
    // matvec (decode)
    bench("decode gate [1,896]x[4864,896]^T", 1, 896, 4864, 1);
}
