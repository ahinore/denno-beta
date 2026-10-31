//! hc mix (`hc_down` / `hc_up`) の decode 実形を CPU だけで測るマイクロベンチ (task 84)。
//!
//! `#[ignore]` なので通常の `cargo test` では走らない。実行:
//!
//! ```text
//! cargo test --release -p denno-engine --test hc_mix_bench -- --ignored --nocapture
//! ```
//!
//! 実形 (27B Flash-Next、`hc_dim = 10240` / `hc_lr = 320`、Q8_0 + `fast_q8`):
//!
//! - `hc_down`: k=10240, n=320 (x = 40 KB、重み 3.4 MB、3.28 M MAC)
//! - `hc_up`  : k=320,  n=10240 (x = 1.3 KB、重み 3.4 MB、3.28 M MAC)
//!
//! 実機の mix-kprof (task 84 の前提) ではどちらも 1 呼び出し 0.3〜0.7 ms 程度で
//! 実効 4.5〜9 GMAC/s と AVX2 の上限よりずっと遅い。ここで出すのは:
//!
//! 1. 1 / 2 / 4 / 8 / 16 スレッドの曲線 (本番入口 `gemv_label_par` / `gemv_label`
//!    を通す。行は独立なのでスレッド数で値はビット一致する)
//! 2. dequant 単体 (全行を `dequant_row_into` で開くだけ) の時間 — 「dequant の
//!    配分」と「積和の配分」を分ける
//! 3. down (x を行ごとに読み直す) と up (x は L1 に載る) の 1 スレッド比較 —
//!    x 再読みが律速なら down が遅いはず、同速なら重み側 (dequant/帯域) が律速
//!
//! 注: 重みは無名メモリ (常に RAM) なので mmap のページフォルトは含まない。

use std::time::Instant;

use denno_engine::kquant::{dequant_row_into, GgmlType};
use denno_engine::model::linear_gguf::GgufLinear;
use denno_engine::model::qwen4exp::HC_DOWN_ROW_PAR_MIN;

/// 1 形あたりの繰り返し (最良値を採る)。
const REP: usize = 5;

/// Q8_0 の合成重み (34 B ブロック = f16 スケール + i8 x 32)。
/// ブロックごとにスケールと i8 を変える (同一値の特殊ケースを避ける)。
fn synth_q8(n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q8_0 のブロックは 32 要素");
    let nb = k / 32;
    let rb = nb * 34;
    let mut b = vec![0u8; n * rb];
    for (r, row) in b.chunks_exact_mut(rb).enumerate() {
        for (blk, o) in row.chunks_exact_mut(34).enumerate() {
            let d = half::f16::from_f32(0.005 + (r % 7) as f32 * 0.001 + (blk % 5) as f32 * 0.0002);
            o[0..2].copy_from_slice(&d.to_le_bytes());
            for j in 0..32usize {
                let q = ((j as i32 * 7 + blk as i32 * 3 + r as i32) % 19) - 9;
                o[2 + j] = q as i8 as u8;
            }
        }
    }
    b
}

/// 最良 (最小) の ms を返す。
fn best_ms(mut f: impl FnMut()) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..REP {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

/// dequant 単体 (1 スレッド): 全行を f32 に開くだけの時間。
fn dequant_only_ms(ty: GgmlType, blocks: &[u8], k: usize, n: usize) -> f64 {
    let rb = k / 32 * 34;
    let mut row = vec![0f32; k];
    // 1 回ウォームアップしてから測る。
    dequant_row_into(ty, &blocks[0..rb], &mut row);
    let t = Instant::now();
    for r in 0..n {
        dequant_row_into(ty, &blocks[r * rb..(r + 1) * rb], &mut row);
    }
    t.elapsed().as_secs_f64() * 1e3
}

#[test]
#[ignore = "6.8 MB の合成重みとスレッドプールを使う (明示実行のみ)"]
fn hc_mix_gemv_shape_scaling() {
    let shapes: [(&str, usize, usize); 2] = [("hc_down", 10240, 320), ("hc_up", 320, 10240)];
    for (name, k, n) in shapes {
        let blocks = synth_q8(n, k);
        let mb = blocks.len() as f64 / 1e6;
        let gmac = (k * n) as f64 / 1e9;
        let x: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.01).collect();
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, k, n, &blocks)
            .expect("GgufLinear::from_slice (Q8_0)")
            .with_fast_q8(true);
        println!(
            "\n[hc-bench] {name}: k={k} n={n} Q8_0 = {mb:.1} MB, {gmac:.2} GMAC"
        );

        // 0. dequant 単体 (1 スレッド)。
        let ms_deq = dequant_only_ms(GgmlType::Q8_0, &blocks, k, n);
        println!(
            "[hc-bench]   dequant 1thr            : {ms_deq:8.3} ms ({:.1} GB/s)",
            mb / 1e3 / (ms_deq * 1e-3)
        );

        // 1. 本番入口をスレッド数別に (down は行並列しきい値 64 を明示、up は既定門番)。
        //    classic カーネル (P84 まで) と 2 行同時カーネル (P84) の両方を測る。
        let mut rows: Vec<(usize, f64, f64, Vec<f32>, Vec<f32>)> = Vec::new();
        for threads in [1usize, 2, 4, 8, 16] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut out = vec![0f32; n];
            let ms = best_ms(|| {
                pool.install(|| {
                    lin.gemv_label_par("hc_mix_bench", &x, &mut out, HC_DOWN_ROW_PAR_MIN)
                        .expect("gemv_label_par");
                });
            });
            let mut out_f = vec![0f32; n];
            let ms_f = best_ms(|| {
                pool.install(|| {
                    lin
                        .gemv_label_par_2row(
                            "hc_mix_bench_2row",
                            &x,
                            &mut out_f,
                            HC_DOWN_ROW_PAR_MIN,
                        )
                        .expect("gemv_label_par_2row");
                });
            });
            rows.push((threads, ms, ms_f, out, out_f));
        }
        let base = rows[0].1;
        for (threads, ms, ms_f, out, out_f) in rows.iter() {
            println!(
                "[hc-bench]   classic {threads:2}thr        : {ms:8.3} ms ({:.1} GMAC/s, \
                 {:.1} GB/s, {:.2}x vs 1thr)",
                gmac / (ms * 1e-3),
                mb / 1e3 / (ms * 1e-3),
                base / ms
            );
            println!(
                "[hc-bench]   2row    {threads:2}thr        : {ms_f:8.3} ms ({:.1} GMAC/s, \
                 {:.1} GB/s, {:.2}x vs 1thr, {:.2}x vs classic)",
                gmac / (ms_f * 1e-3),
                mb / 1e3 / (ms_f * 1e-3),
                base / ms_f,
                ms / ms_f
            );
            // 2 行カーネルと classic カーネルのビット一致 (同じスレッド数で)。
            for (r, (a, b)) in out_f.iter().zip(out.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} 2row {threads} スレッドで行 {r} が classic とビット一致しない \
                     ({a} vs {b})"
                );
            }
            // スレッド数が変わっても classic / 2row とも値は動かない (行は独立)。
            let (_, _, _, out1, out_f1) = &rows[0];
            for (r, (a, b)) in out.iter().zip(out1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} classic {threads} スレッドで行 {r} が 1 スレッドとビット一致しない"
                );
            }
            for (r, (a, b)) in out_f.iter().zip(out_f1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} 2row {threads} スレッドで行 {r} が 1 スレッドとビット一致しない"
                );
            }
        }
        assert!(
            rows[0].3.iter().any(|v| *v != 0.0),
            "{name}: 全 0 (経路の生死が分からない)"
        );
    }
}
