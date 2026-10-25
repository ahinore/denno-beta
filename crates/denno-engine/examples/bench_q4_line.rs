//! Q4_0 線形層の単体ベンチ。
//!
//! 27B の decode / prefill で一番重い層を、モデルを起動せずに (数秒で) 測る。
//! 形状は実モデルと同じ (既定: 27B の gate_up = k 5120, n 34816)。重みは合成値
//! だが、レイアウト (Q4_0 ブロック列) とサイズは本物と同じなので、カーネルの
//! 帯域・レイテンシはそのまま再現できる。
//!
//! 使い方:
//!   cargo run --release -p denno-engine --features cuda --example bench_q4_line
//!   ... --example bench_q4_line -- --k 17408 --n 5120          # down_proj
//!   ... --example bench_q4_line -- --n 248320 --layers 1 --raw  # lm_head
//!
//! 出力の見方: m=1 が decode の GEMV、m=300 が prefill のタイル GEMM。
//! `layers` を掛けた値が 1 token あたりの見積り (プロファイルの bucket と比較する)。
//! `--raw` は f32 の重みを作らずブロックを直接埋める (巨大な層用。CPU 比較は省く)。
//!
//! 注意: 同じ重みバッファを反復で読むので、L2 (この機材は 96 MiB) に収まる層では
//! キャッシュに当たった値が出る。実モデルの decode は 1 token で全層 16.7 GB を
//! 冷えた状態で読むため、絶対値ではなく同じ条件どうしの相対比較に使うこと。
//!
//! GEMV の列ブロック幅 C (s133) を比べるときは、本体の定数を書き換えずに
//! `DENNO_Q4_GEMV_C=2|4|8` で切り替える (省略時はビルド既定の C=4)。
//! 例: `DENNO_Q4_GEMV_C=8 cargo run --release -p denno-engine --features cuda
//!      --example bench_q4_line -- --iters 30`

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
    let raw = args.iter().any(|a| a == "--raw");
    assert_eq!(k % 32, 0, "k must be a multiple of 32");
    std::env::set_var("DENNO_CUDA", "1");

    let row_bytes = Q4Linear::row_bytes(k);
    let bytes = row_bytes * n;
    println!(
        "shape k={k} n={n}  Q4 weights {:.1} MB ({} B/row)  layers={layers} iters={iters}{}",
        bytes as f64 / 1e6,
        row_bytes,
        if raw { "  [raw]" } else { "" }
    );

    // 合成重み。既定は f32 を作ってから量子化するので CPU 参照と比較できる。
    let q4 = if raw {
        let nb = k / 32;
        let mut blocks = vec![0u8; n * nb * 18];
        for (bi, b) in blocks.chunks_exact_mut(18).enumerate() {
            // f16 の 0.0625 を little-endian で入れる (scale 0 だと縮退する)
            b[0] = 0x00;
            b[1] = 0x2C;
            for (i, x) in b[2..].iter_mut().enumerate() {
                *x = ((bi * 31 + i * 17) % 251) as u8;
            }
        }
        Q4Linear {
            blocks: blocks.into(),
            k,
            n,
            uid: next_q8_uid(),
        }
    } else {
        let mut w = vec![0f32; k * n];
        for (i, v) in w.iter_mut().enumerate() {
            *v = (((i * 7) % 251) as f32 / 125.0) - 1.0;
        }
        let t = Instant::now();
        let q = Q4Linear::quantize(&w, k, n);
        drop(w);
        println!("quantize: {:.2} s", t.elapsed().as_secs_f64());
        q
    };

    let mut worst = 0.0f64;
    for &m in &[1usize, 2, 8, 64, 300] {
        let a: Vec<f32> = (0..m * k).map(|i| (((i % 17) as f32) / 9.0) - 0.5).collect();
        let mut out = vec![0f32; m * n];
        let t = Instant::now();
        let ok =
            denno_engine::cuda::maybe_gpu_linear_q4_into(q4.uid, &a, &q4.blocks, k, n, m, &mut out);
        let first = t.elapsed().as_secs_f64();
        if !ok {
            println!("m={m:>4}: GPU path not taken (CPU fallback)");
            continue;
        }
        let t = Instant::now();
        for _ in 0..iters {
            let ok =
                denno_engine::cuda::maybe_gpu_linear_q4_into(q4.uid, &a, &q4.blocks, k, n, m, &mut out);
            assert!(ok, "GPU path dropped out on iteration");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;
        // m=1 は重み 1 回読みなので GB/s、m>1 は同じ重みを使い回すので GFLOP/s で見る。
        let gbs = bytes as f64 / per / 1e9;
        let gflops = 2.0 * (m * n * k) as f64 / per / 1e9;
        println!(
            "m={m:>4}  {:.3} ms/call  {:.1} GB/s  {:.1} GFLOP/s  first {:.3} s  x{layers} layers = {:.1} ms/token",
            per * 1e3,
            gbs,
            gflops,
            first,
            per * 1e3 * layers as f64
        );
        // 先頭 4 行だけ CPU 参照と突き合わせる (回帰の検出用)
        if !raw {
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
    }
    if worst > 1e-3 {
        println!("WARN: deviation {worst:.2e} exceeds the R3 budget");
    }
}