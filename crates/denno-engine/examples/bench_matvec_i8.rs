//! decode 用 matvec カーネルの単体ベンチ (P3 SIMD int8 ドットの効果測定)。
//! Qwen2.5-0.5B の線形層形状を使用: qkv [1152,896]、o [896,896]、
//! gate/up [4864,896]、down [896,4864]、lm_head [151936,896]。
//!
//! 使い方: cargo run --release -p denno-engine --example bench_matvec_i8

use std::time::Instant;

use denno_engine::tensor::{matvec_bt, matvec_bt_i8};
use rayon::prelude::*;

fn bench_matvec(name: &str, k: usize, n: usize, iters: usize) {
    // 重み生成 (決定論的)。fp32 と int8 は同じ値を量子化したもの
    let w_f32: Vec<f32> = (0..n * k).map(|i| ((i % 251) as f32 / 125.0) - 1.0).collect();
    let scale: Vec<f32> = (0..n)
        .map(|j| {
            w_f32[j * k..][..k]
                .iter()
                .fold(0f32, |m, v| m.max(v.abs()))
                / 127.0
        })
        .collect();
    let w_i8: Vec<i8> = (0..n)
        .flat_map(|j| {
            let inv = 1.0 / scale[j];
            w_f32[j * k..][..k]
                .iter()
                .map(|&v| ((v * inv).round().clamp(-127.0, 127.0)) as i8)
                .collect::<Vec<i8>>()
        })
        .collect();
    let a: Vec<f32> = (0..k).map(|i| ((i % 97) as f32 / 48.0) - 1.0).collect();

    // ウォームアップ
    let _ = matvec_bt(&a, &w_f32, k, n);
    let _ = matvec_bt_i8(&a, &w_i8, &scale, k, n);

    let t = Instant::now();
    let mut sink_f32 = 0.0f32;
    for _ in 0..iters {
        let c = matvec_bt(&a, &w_f32, k, n);
        sink_f32 += c[0];
    }
    let el_f32 = t.elapsed().as_secs_f64() / iters as f64;

    let t = Instant::now();
    let mut sink_i8 = 0.0f32;
    for _ in 0..iters {
        let c = matvec_bt_i8(&a, &w_i8, &scale, k, n);
        sink_i8 += c[0];
    }
    let el_i8 = t.elapsed().as_secs_f64() / iters as f64;

    let ms_f32 = el_f32 * 1e3;
    let ms_i8 = el_i8 * 1e3;
    println!(
        "{name} [1,{k}]x[{n},{k}]^T: fp32 {ms_f32:.3}ms  int8 {ms_i8:.3}ms  ({:.2}x)",
        el_f32 / el_i8
    );
    // bytes/token: fp32 は n*k*4、int8 は n*k+4n
    let bw_f32 = (n * k * 4) as f64 / el_f32 / 1e9;
    let bw_i8 = (n * k + 4 * n) as f64 / el_i8 / 1e9;
    println!("   実効帯域: fp32 {bw_f32:.1} GB/s  int8 {bw_i8:.1} GB/s  (checksum {sink_f32:.3}/{sink_i8:.3})");
}

/// 整数ドット単体のスループットも確認 (sdot の実効性能)
fn bench_i8_dot(k: usize) {
    let a: Vec<i8> = (0..k).map(|i| (i % 127) as i8).collect();
    let b: Vec<i8> = (0..k).map(|i| ((i * 7) % 127) as i8).collect();
    let rows = 4096usize;
    let t = Instant::now();
    let total: i64 = (0..rows)
        .into_par_iter()
        .map(|r| {
            let off = r % 7;
            denno_engine::tensor::i8_dot(&a[off..], &b[off..]) as i64
        })
        .sum();
    let el = t.elapsed().as_secs_f64();
    let macs = rows as f64 * (k - 7) as f64;
    let ms = el * 1e3;
    let gmacs = macs / el / 1e9;
    println!(
        "i8_dot k={k} x{rows} rows: {ms:.2}ms  {gmacs:.1} GMAC/s ({} threads) checksum={total}",
        rayon::current_num_threads()
    );
}

fn main() {
    println!("rayon threads: {}", rayon::current_num_threads());
    bench_matvec("qkv   ", 896, 1152, 2000);
    bench_matvec("o_proj", 896, 896, 2000);
    bench_matvec("gate  ", 896, 4864, 2000);
    bench_matvec("up    ", 896, 4864, 2000);
    bench_matvec("down  ", 4864, 896, 2000);
    bench_matvec("lm_head", 896, 151936, 100);
    bench_i8_dot(896);
}
