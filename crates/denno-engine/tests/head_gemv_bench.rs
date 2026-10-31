//! head (`output` / lm_head) の GEMV を CPU だけで測るマイクロベンチ。
//!
//! `#[ignore]` なので通常の `cargo test` では走らない。実行:
//!
//! ```text
//! cargo test --release -p denno-engine --test head_gemv_bench -- --ignored --nocapture
//! ```
//!
//! 実モデル (Qwen3.5-27B Flash-Next) の head は `output.weight` が Q8_0 の
//! [n_vocab = 151936, n_embd = 2560] で、1 トークンあたり 413,265,920 B
//! (= 151936 行 x 80 ブロック x 34 B) を読む。ここでは同じ形の**合成重み**を作り、
//!
//! 1. スカラー経路 (`expert_gemv_scalar` = `dequant_row_into` + `dot_seq`)
//! 2. 融合 AVX2 カーネル (実経路) を 1 / 2 / 4 / 8 / 16 スレッドで
//!
//! の 1 呼び出し時間を出す。スレッド数を変えても**出力はビット一致**することを
//! 同時に確かめる (行は独立で、行内の `b` 昇順は誰が回しても同じ)。
//!
//! 注: この重みは無名メモリ (常に RAM) なので、ここで測れるのは**計算 + RAM 帯域**
//! だけ。実機で mmap 越しにページフォルトが乗る場合の I/O は含まない。

use std::time::Instant;

use denno_engine::kquant::GgmlType;
use denno_engine::model::linear_gguf::GgufLinear;
use denno_engine::model::moe_expert::expert_gemv_scalar;
use denno_engine::model::qwen4exp::HEAD_ROW_PAR_MIN;

/// n_vocab (27B Flash-Next の実形)
const N: usize = 151_936;
/// n_embd (同上)
const K: usize = 2_560;
/// 1 形あたりの繰り返し (最良値を採る)
const REP: usize = 3;

/// Q8_0 の合成重み (34 B ブロック = f16 スケール + i8 x 32)。
fn synth_q8(n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q8_0 のブロックは 32 要素");
    let rb = k / 32 * 34;
    let mut tpl = [0u8; 34];
    tpl[0..2].copy_from_slice(&half::f16::from_f32(0.007_812_5).to_le_bytes());
    for j in 0..32usize {
        let q = ((j as i32 * 7) % 19) - 9;
        tpl[2 + j] = q as i8 as u8;
    }
    let mut b = vec![0u8; n * rb];
    for blk in b.chunks_exact_mut(34) {
        blk.copy_from_slice(&tpl);
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

#[test]
#[ignore = "413 MB の合成重みを使う (明示実行のみ)"]
fn head_gemv_shape_scaling() {
    let blocks = synth_q8(N, K);
    let gb = blocks.len() as f64 / 1e9;
    println!(
        "[head-bench] shape n={N} k={K} Q8_0 = {} B ({gb:.3} GB)",
        blocks.len()
    );
    let x: Vec<f32> = (0..K).map(|i| ((i % 29) as f32 - 14.0) * 0.01).collect();
    let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &blocks)
        .expect("GgufLinear::from_slice (Q8_0)")
        .with_fast_q8(true);

    // 1. スカラー経路 (参照実装。dequant_row_into + dot_seq)
    let mut out_seq = vec![0f32; N];
    let ms_seq = best_ms(|| {
        expert_gemv_scalar(GgmlType::Q8_0, &blocks, K, N, &x, &mut out_seq).expect("scalar");
    });
    println!(
        "[head-bench] seq  1thr          : {ms_seq:8.2} ms ({:.1} GB/s)",
        gb / (ms_seq * 1e-3)
    );

    // 2. 融合 AVX2 カーネル (GgufLinear の実経路) をスレッド数別に
    let mut rows: Vec<(usize, f64, Vec<f32>)> = Vec::new();
    for threads in [1usize, 2, 4, 8, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        let mut out = vec![0f32; N];
        let ms = best_ms(|| {
            pool.install(|| {
                lin.gemv_label_par("output", &x, &mut out, HEAD_ROW_PAR_MIN)
                    .expect("gemv_label_par");
            });
        });
        rows.push((threads, ms, out));
    }
    let base = rows[0].1;
    for (threads, ms, out) in rows.iter() {
        println!(
            "[head-bench] fused {threads:2}thr        : {ms:8.2} ms ({:.1} GB/s, {:.2}x vs 1thr)",
            gb / (ms * 1e-3),
            base / ms
        );
        // 行並列の有無・スレッド数で値が動かないこと (ビット一致)。
        for (r, (a, b)) in out.iter().zip(rows[0].2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{threads} スレッドで行 {r} がビット一致しない ({a} vs {b})"
            );
        }
    }
    assert!(
        rows[0].2.iter().any(|v| *v != 0.0),
        "全 0 (経路の生死が分からない)"
    );
    println!(
        "[head-bench] 8thr / seq = {:.2}x (seq は参照実装なので実経路との差ではない)",
        ms_seq / rows[3].1
    );
}
