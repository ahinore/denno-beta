//! Q4_0 線形層の単体ベンチ。
//!
//! 27B の decode / prefill で一番重い層を、モデルを起動せずに (数秒で) 測る。
//! 形状は実モデルと同じ (既定: 27B の gate_up = k 5120, n 34816)。重みは合成値
//! だが、レイアウト (Q4_0 ブロック列) とサイズは本物と同じなので、カーネルの
//! 帯域・レイテンシはそのまま再現できる。
//!
//! 使い方:
//!   cargo run --release -p denno-engine --features cuda --example bench_q4_line
//!   cargo run --release -p denno-engine --features cuda --example bench_q4_line -- --k 5120 --n 34816 --layers 64 --iters 50
//!
//! 出力の見方: m=1 が decode の GEMV、m=300 が prefill のタイル GEMM。
//! `layers` を掛けた値が 1 token あたりの見積り (プロファイルの bucket と比較する)。

use std::time::Instant;

use denno_engine::quant::{next_q8_uid, Q4Linear};

fn arg_u(args: &[String], name: &str, dflt: usize) -> usize {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(v) = v.parse() {
                    return v;
                }
            }
        }
    }
    dflt
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let k = arg_u(&args, "--k", 5120); // hidden
    let n = arg_u(&args, "--n", 34816); // 2 * ff  (27B: 2 * 17408)
    let layers = arg_u(&args, "--layers", 64);
    let iters = arg_u(&args, "--iters", 30);
    assert_eq!(k % 32, 0, "k must be a multiple of 32");
    std::env::set_var("DENNO_CUDA", "1");

    let row_bytes = Q4Linear::row_bytes(k);
    let bytes = row_bytes * n;
    println!(
        "shape k={k} n={n}  Q4 weights {:.1} MB ({} B/row)  layers={layers} iters={iters}",
        bytes as f64 / 1e6,
        row_bytes
    );

    // 合成重み (決定論的)。Q4_0 へ量子化してからベンチに使う。
    let mut w = vec![0f32; k * n];
    for (i, v) in w.iter_mut().enumerate() {
        *v = (((i * 7) % 251) as f32 / 125.0) - 1.0;
    }
    let t = Instant::now();
    let q4 = Q4Linear::quantize(&w, k, n);
    drop(w);
    println!("quantize: {:.2} s", t.elapsed().as_secs_f64());

    let uid = next_q8_uid();
    let mut worst = 0.0f64;
    for &m in &[1usize, 2, 8, 64, 300] {
        let a: Vec<f32> = (0..m * k).map(|i| (((i % 17) as f32) / 9.0) - 0.5).collect();
        let mut out = vec![0f32; m * n];
        let t = Instant::now();
        let ok = denno_engine::cuda::maybe_gpu_linear_q4_into(
            uid,
            &a,
            &q4.blocks,
            k,
            n,
            m,
            &mut out,
        );
        let first = t.elapsed().as_secs_f64();
        if !ok {
            println!("m={m:>4}: GPU path not taken (CPU fallback)");
            continue;
        }
        let t = Instant::now();
        for _ in 0..iters {
            let ok = denno_engine::cuda::maybe_gpu_linear_q4_into(
                uid,
                &a,
                &q4.blocks,
                k,
                n,
                m,
                &mut out,
            );
            assert!(ok, "GPU path dropped out on iteration");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        let gbs = if m == 1 {
            bytes as f64 / per / 1e9
        } else {
            // prefill は重みを 1 回読むだけなので参考値
            bytes as f64 / per / 1e9
        };
        println!(
            "m={m:>4}  {:.3} ms/call  {:.1} GB/s  first {:.3} s  x{layers} layers = {:.1} ms/token",
            per * 1e3,
            gbs,
            first,
            per * 1e3 * layers as f64
        );
        // 先頭 4 行だけ CPU 参照と突き合わせる (回帰の検出用)
        let mut row = vec![0f32; k];
        let mut mx = 0.0f64;
        for j in 0..4 {
            q4.dequant_row_into(j, &mut row);
            let d: f32 = row.iter().zip(&a[..k]).map(|(x, y)| x * y).sum();
            let e = (out[j] - d).abs() as f64;
            if e > mx {
                mx = e;
            }
        }
        worst = worst.max(mx);
        println!("      maxabs vs CPU (rows 0-3) = {mx:.2e}");
    }
    if worst > 1e-3 {
        println!("WARN: deviation {worst:.2e} exceeds the R3 budget");
    }
}