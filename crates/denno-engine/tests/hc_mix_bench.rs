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
use denno_engine::model::moe_expert::expert_gemv_scalar;
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

/// IQ4_XS の 1 ブロック (136 B = d(2) + scales_h(2) + scales_l(4) + qs(128)) を作る。
///
/// 6bit スケール `ls` は `(seed + ib * 9) % 64` で 0..=63 を巡回 (9 と 64 は互いに素)、
/// qs は決定的擬似乱数で nibble 全域 (0..=15) を通す。ブロックごと / 行ごとに
/// seed を変えて同一値の特殊ケースを避ける (task 89b)。
fn synth_iq4xs_block(seed: u32) -> [u8; 136] {
    fn pat(i: usize, seed: u32) -> u8 {
        let mut x = (i as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(seed.wrapping_mul(40_503));
        x ^= x >> 13;
        x = x.wrapping_mul(0x9E37_79B1);
        x ^= x >> 16;
        x as u8
    }
    let mut b = [0u8; 136];
    let d = half::f16::from_f32(0.005 + (seed % 13) as f32 * 0.0007);
    b[0..2].copy_from_slice(&d.to_le_bytes());
    let mut scales_l = [0u8; 4];
    let mut scales_h: u16 = 0;
    for ib in 0..8usize {
        let ls = ((seed as usize + ib * 9) % 64) as u16;
        let nib = (ls & 0xf) as u8;
        let hi2 = ((ls >> 4) & 0x3) as u16;
        // kquant 参照と同じ詰め方: 偶数 ib が scales_l の下位 nibble、奇数が上位。
        scales_l[ib / 2] = if ib % 2 == 0 {
            (scales_l[ib / 2] & 0xf0) | nib
        } else {
            (scales_l[ib / 2] & 0x0f) | (nib << 4)
        };
        scales_h |= hi2 << (2 * ib);
    }
    b[2..4].copy_from_slice(&scales_h.to_le_bytes());
    b[4..8].copy_from_slice(&scales_l);
    for i in 8..136 {
        b[i] = pat(i, seed);
    }
    b
}

/// IQ4_XS の合成重み (`n` 行 x `k` 列、`k` は 256 の倍数 = 136 B ブロック/256 要素)。
fn synth_iq4xs(n: usize, k: usize) -> Vec<u8> {
    assert_eq!(k % 256, 0, "IQ4_XS の行長は 256 の倍数");
    let nb = k / 256;
    let rb = nb * 136;
    let mut v = vec![0u8; n * rb];
    for (r, row) in v.chunks_exact_mut(rb).enumerate() {
        for (blk, b) in row.chunks_exact_mut(136).enumerate() {
            b.copy_from_slice(&synth_iq4xs_block((r * 131 + blk * 17) as u32 + 5));
        }
    }
    v
}

/// P89: IQ4_XS 融合 GEMV の実形マイクロベンチ (task 89b)。
///
/// IQ4_XS は 256 要素 / 136 B ブロックなので、実形として表現できるのは
/// `hc_down` (k=10240 = 40 ブロック/行, n=320) の 1 形。`hc_up` の実形 k=320 は
/// 256 ブロックで割り切れず `row_bytes` が拒否する (IQ4_XS では表現不可能) —
/// 2 形目は k=512 の**代用形**で「細い行 x 多数の行」の傾向だけを見る
/// (`hc_up` の実型確認はタスク側の宿題、REPORT_P89.md 参照)。
///
/// - classic = [`expert_gemv_scalar`] (dequant + 逐次内積 = P88 までの本番経路。
///   逐次なのでスレッド数で変わらない)
/// - fused 1row / 2row = 本番入口 ([`GgufLinear::gemv_label_par`] /
///   [`GgufLinear::gemv_label_par_2row`]) を通す (P89 で IQ4_XS が融合へ振れる)
/// - 値は classic とビット一致**しない** (FMA 累積)。スレッド数と 1 行 / 2 行の
///   差ではビット一致する (行は独立) のでここで確認する
#[test]
#[ignore = "IQ4_XS 合成重み (最大 1.7 MB) とスレッドプールを使う (明示実行のみ)"]
fn hc_mix_gemv_iq4xs_shape_scaling() {
    let shapes: [(&str, usize, usize, usize); 2] = [
        ("hc_down", 10240, 320, HC_DOWN_ROW_PAR_MIN),
        ("hc_up*", 512, 10240, 0),
    ];
    for (name, k, n, row_par_min) in shapes {
        let blocks = synth_iq4xs(n, k);
        let mb = blocks.len() as f64 / 1e6;
        let gmac = (k * n) as f64 / 1e9;
        let x: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.01).collect();
        let lin = GgufLinear::from_slice(GgmlType::IQ4Xs, k, n, &blocks)
            .expect("GgufLinear::from_slice (IQ4_XS)");
        println!("\n[hc-bench-iq4xs] {name}: k={k} n={n} IQ4_XS = {mb:.2} MB, {gmac:.2} GMAC");

        // dequant 単体 (1 スレッド): classic のうち「開くだけ」の配分。
        let rb = k / 256 * 136;
        let mut row = vec![0f32; k];
        dequant_row_into(GgmlType::IQ4Xs, &blocks[0..rb], &mut row);
        let t = Instant::now();
        for r in 0..n {
            dequant_row_into(GgmlType::IQ4Xs, &blocks[r * rb..(r + 1) * rb], &mut row);
        }
        let ms_deq = t.elapsed().as_secs_f64() * 1e3;
        println!(
            "[hc-bench-iq4xs]   dequant 1thr            : {ms_deq:8.3} ms ({:.1} GB/s)",
            mb / 1e3 / (ms_deq * 1e-3)
        );

        // classic (P88 までの本番経路: 逐次 dequant + 逐次内積)。
        let mut out_c = vec![0f32; n];
        let ms_c = best_ms(|| {
            expert_gemv_scalar(GgmlType::IQ4Xs, &blocks, k, n, &x, &mut out_c)
                .expect("expert_gemv_scalar");
        });
        println!(
            "[hc-bench-iq4xs]   classic (逐次)          : {ms_c:8.3} ms ({:.1} GMAC/s, {:.1} GB/s)",
            gmac / (ms_c * 1e-3),
            mb / 1e3 / (ms_c * 1e-3)
        );

        // fused: 本番入口をスレッド数別に (1 行版と 2 行版)。
        let mut rows: Vec<(usize, f64, f64, Vec<f32>, Vec<f32>)> = Vec::new();
        for threads in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut out = vec![0f32; n];
            let ms = best_ms(|| {
                pool.install(|| {
                    lin.gemv_label_par("hc_mix_bench_iq4xs", &x, &mut out, row_par_min)
                        .expect("gemv_label_par");
                });
            });
            let mut out_f = vec![0f32; n];
            let ms_f = best_ms(|| {
                pool.install(|| {
                    lin.gemv_label_par_2row("hc_mix_bench_iq4xs_2row", &x, &mut out_f, row_par_min)
                        .expect("gemv_label_par_2row");
                });
            });
            rows.push((threads, ms, ms_f, out, out_f));
        }
        let base = rows[0].1;
        for (threads, ms, ms_f, out, out_f) in rows.iter() {
            println!(
                "[hc-bench-iq4xs]   fused 1row {threads:2}thr     : {ms:8.3} ms ({:.1} GMAC/s, \
                 {:.1} GB/s, {:.2}x vs 1thr, {:.1}x vs classic)",
                gmac / (ms * 1e-3),
                mb / 1e3 / (ms * 1e-3),
                base / ms,
                ms_c / ms
            );
            println!(
                "[hc-bench-iq4xs]   fused 2row {threads:2}thr     : {ms_f:8.3} ms ({:.1} GMAC/s, \
                 {:.1} GB/s, {:.2}x vs 1thr, {:.1}x vs classic)",
                gmac / (ms_f * 1e-3),
                mb / 1e3 / (ms_f * 1e-3),
                base / ms_f,
                ms_c / ms_f
            );
            // 2 行版は 1 行版とビット一致 (累積順不変)。
            for (r, (a, b)) in out_f.iter().zip(out.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused 2row {threads} スレッドで行 {r} が 1 行版とビット不一致 \
                     ({a} vs {b})"
                );
            }
            // スレッド数が変わっても値は動かない (行は独立)。
            let (_, _, _, out1, out_f1) = &rows[0];
            for (r, (a, b)) in out.iter().zip(out1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused 1row {threads} スレッドで行 {r} が 1 スレッドとビット不一致"
                );
            }
            for (r, (a, b)) in out_f.iter().zip(out_f1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused 2row {threads} スレッドで行 {r} が 1 スレッドとビット不一致"
                );
            }
        }
        assert!(
            rows[0].3.iter().any(|v| *v != 0.0),
            "{name}: 全 0 (経路の生死が分からない)"
        );
    }
}

/// task 92: BF16 融合 GEMV の実形マイクロベンチ (hc mix の down/up 実型)。
///
/// 本番モデルの hc テンソルは BF16 (実機 mix-kprof: down 0.34 ms/呼び出し、
/// down+up で 0.68 ms) なので、2 形とも**実形そのまま**で測れる:
///
/// - `hc_down`: k=10240, n=320 (x = 40 KB、重み 6.55 MB、3.28 M MAC)
/// - `hc_up`  : k=320,  n=10240 (x = 1.3 KB、重み 6.55 MB、3.28 M MAC)
///
/// - classic = [`expert_gemv_scalar`] (1 行 dequant + [`dot_seq`] = P91 までの本番経路)
/// - fused   = 本番入口 ([`GgufLinear::gemv_label_par`] /
///   [`GgufLinear::gemv_label_par_2row`]) を通す (task 92 で BF16 が融合へ振れる)。
///   1 / 2 / 4 / 8 スレッド。
/// - 値は classic と**ビット一致**する (BF16 -> f32 変換は正確で、カーネルは
///   積を i 昇順で逐次加算する。FMA もレーン並列の部分和も使わない) ので、
///   スレッド数と 2 行指定の差と合わせてここで `to_bits()` を固定する。
///
/// 実行:
/// ```text
/// cargo test --release -p denno-engine --test hc_mix_bench -- --ignored --nocapture hc_mix_gemv_bf16
/// ```
#[test]
#[ignore = "BF16 合成重み (6.55 MB x 2) とスレッドプールを使う (明示実行のみ)"]
fn hc_mix_gemv_bf16_shape_scaling() {
    use denno_engine::model::moe_expert::{bf16_fast_enabled, expert_gemv_scalar};

    assert!(
        bf16_fast_enabled(),
        "DENNO_BF16_FAST=0 で起動されている (fused 経路を測れない)"
    );
    let shapes: [(&str, usize, usize, usize); 2] = [
        ("hc_down", 10240, 320, HC_DOWN_ROW_PAR_MIN),
        ("hc_up", 320, 10240, 0),
    ];
    for (name, k, n, row_par_min) in shapes {
        // BF16 合成重み: 要素ごとに決定的な値 (f32 を bf16 に切り捨て)。
        let rb = k * 2;
        let mut blocks = vec![0u8; n * rb];
        for r in 0..n {
            for i in 0..k {
                let h = ((i.wrapping_mul(2_654_435_761)) ^ (r.wrapping_mul(40_503))) & 0xFF_FFFF;
                let f = (h as f32 / 2_048.0 - 4_096.0) * 2.0;
                let bits = (f.to_bits() >> 16) as u16;
                blocks[(r * k + i) * 2..(r * k + i) * 2 + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        let mb = blocks.len() as f64 / 1e6;
        let gmac = (k * n) as f64 / 1e9;
        let x: Vec<f32> = (0..k).map(|i| ((i % 29) as f32 - 14.0) * 0.01).collect();
        let lin = GgufLinear::from_slice(GgmlType::Bf16, k, n, &blocks)
            .expect("GgufLinear::from_slice (BF16)");
        println!("\n[hc-bench-bf16] {name}: k={k} n={n} BF16 = {mb:.2} MB, {gmac:.2} GMAC");

        // classic (P91 までの本番経路: 逐次 dequant + 逐次内積)。
        let mut out_c = vec![0f32; n];
        let ms_c = best_ms(|| {
            expert_gemv_scalar(GgmlType::Bf16, &blocks, k, n, &x, &mut out_c)
                .expect("expert_gemv_scalar");
        });
        println!(
            "[hc-bench-bf16]   classic (逐次)          : {ms_c:8.3} ms ({:.1} GMAC/s, {:.1} GB/s)",
            gmac / (ms_c * 1e-3),
            mb / 1e3 / (ms_c * 1e-3)
        );

        // fused: 本番入口をスレッド数別に (1 行版と 2 行指定版。値は同じ)。
        let mut rows: Vec<(usize, f64, f64, Vec<f32>, Vec<f32>)> = Vec::new();
        for threads in [1usize, 2, 4, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let mut out = vec![0f32; n];
            let ms = best_ms(|| {
                pool.install(|| {
                    lin.gemv_label_par("hc_mix_bench_bf16", &x, &mut out, row_par_min)
                        .expect("gemv_label_par");
                });
            });
            let mut out_f = vec![0f32; n];
            let ms_f = best_ms(|| {
                pool.install(|| {
                    lin.gemv_label_par_2row(
                        "hc_mix_bench_bf16_2row",
                        &x,
                        &mut out_f,
                        row_par_min,
                    )
                    .expect("gemv_label_par_2row");
                });
            });
            rows.push((threads, ms, ms_f, out, out_f));
        }
        let base = rows[0].1;
        for (threads, ms, ms_f, out, out_f) in rows.iter() {
            println!(
                "[hc-bench-bf16]   fused 1row {threads:2}thr     : {ms:8.3} ms ({:.1} GMAC/s, \
                 {:.1} GB/s, {:.2}x vs 1thr, {:.1}x vs classic)",
                gmac / (ms * 1e-3),
                mb / 1e3 / (ms * 1e-3),
                base / ms,
                ms_c / ms
            );
            println!(
                "[hc-bench-bf16]   fused 2row {threads:2}thr     : {ms_f:8.3} ms ({:.1} GMAC/s, \
                 {:.1} GB/s, {:.2}x vs 1thr, {:.1}x vs classic)",
                gmac / (ms_f * 1e-3),
                mb / 1e3 / (ms_f * 1e-3),
                base / ms_f,
                ms_c / ms_f
            );
            // 融合は classic とビット一致 (BF16 の契約)。
            for (r, (a, b)) in out.iter().zip(out_c.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused {threads} スレッドで行 {r} が classic とビット不一致 \
                     ({a} vs {b})"
                );
            }
            for (r, (a, b)) in out_f.iter().zip(out_c.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused 2row {threads} スレッドで行 {r} が classic とビット不一致"
                );
            }
            // スレッド数が変わっても値は動かない (行は独立)。
            let (_, _, _, out1, out_f1) = &rows[0];
            for (r, (a, b)) in out.iter().zip(out1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused {threads} スレッドで行 {r} が 1 スレッドとビット不一致"
                );
            }
            for (r, (a, b)) in out_f.iter().zip(out_f1.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{name} fused 2row {threads} スレッドで行 {r} が 1 スレッドとビット不一致"
                );
            }
        }
        assert!(
            out_c.iter().any(|v| *v != 0.0),
            "{name}: 全 0 (経路の生死が分からない)"
        );
    }
}
