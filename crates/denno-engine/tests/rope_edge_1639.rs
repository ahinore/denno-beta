//! RoPE (rotary embedding) の境界・精度テスト (task 1639)。
//!
//! 対象は `src/tensor.rs` の公開 API `rope_row` / `rope_rows`。CPU-only・決定的。
//!
//! 固定する性質:
//!
//! 1. pos=0 は恒等写像 (角度 0 → cos=1, sin=0 なのでビット一致で戻る)。
//! 2. 大きな位置 (8192 / 32768) でも NaN/Inf を出さず、ノルムを保存する
//!    (回転行列は直交。f32 の丸め誤差のみ許容 <= 1e-5)。
//! 3. 同じ入力への反復呼び出しはビット一致 (プロセス内状態を持たない)。
//! 4. `rope_rows` は行ごとの `rope_row(pos0 + i)` とビット一致する。
//! 5. 大きな位置でも直交性 (2 点間距離の保存) が成り立つ。
//! 6. ヘッドは独立に回転し、角度は `pos * theta^(-2d/dh)`。
//!
//! 乱数はテスト内の LCG のみ (プロセス外状態に依存しない)。

use denno_engine::tensor::{rope_row, rope_rows};

/// 決定的な入力生成器 (線形合同法)。生成規則をここに固定する。
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // [-1, 1)
        ((self.0 >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn norm(v: &[f32]) -> f64 {
    v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
}

/// 2 点間距離 (f64 で評価して f32 の丸めを混ぜない)。
fn dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            let d = *p as f64 - *q as f64;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

const THETA_QWEN: f32 = 1.0e7;

/// pos=0 では角度が 0。cos(0)=1, sin(0)=0 なので x*c - y*s, x*s + y*c は
/// ビット一致で元に戻る (浮動小数の丸めも入らない)。
/// `rope_rows` は行 i を `pos0 + i` で回すため、恒等になるのは 1 行だけ。
#[test]
fn pos_zero_is_bit_exact_identity() {
    let mut rng = Lcg::new(1639);
    for &dh in &[2usize, 4, 8, 64, 128] {
        for &heads in &[1usize, 2, 3, 8] {
            for &theta in &[10_000.0f32, THETA_QWEN] {
                let src = rng.vec(heads * dh);
                let mut x = src.clone();
                rope_rows(&mut x, 1, heads, dh, 0, theta);
                assert_eq!(bits(&x), bits(&src), "pos=0 は恒等 (dh={dh}, heads={heads})");
            }
        }
    }
}

/// `rope_rows(x, rows, heads, dh, pos0, theta)` は
/// 各行 `rope_row(&mut x[i*width..], heads, dh, pos0+i, theta)` と同一。
#[test]
fn rope_rows_equals_per_row_rope_row() {
    let (rows, heads, dh) = (4usize, 3usize, 8usize);
    let width = heads * dh;
    let mut rng = Lcg::new(1639_1);
    let src = rng.vec(rows * width);

    for &pos0 in &[0usize, 1, 8192, 32_768] {
        let mut whole = src.clone();
        rope_rows(&mut whole, rows, heads, dh, pos0, THETA_QWEN);

        let mut per_row = src.clone();
        for i in 0..rows {
            rope_row(
                &mut per_row[i * width..(i + 1) * width],
                heads,
                dh,
                pos0 + i,
                THETA_QWEN,
            );
        }
        assert_eq!(
            bits(&whole),
            bits(&per_row),
            "行ごとの rope_row とビット一致しない (pos0={pos0})"
        );
    }
}

/// 大きな位置でもノルムが保存される (許容 1e-5、相対)。
/// 回転行列 [[c,-s],[s,c]] は直交で、ペアごとにノルムを保つ。
#[test]
fn norm_preserved_at_large_positions() {
    let (rows, heads, dh) = (3usize, 4usize, 64usize);
    let width = heads * dh;
    let mut rng = Lcg::new(1639_2);
    let src = rng.vec(rows * width);

    for &pos0 in &[0usize, 1, 1024, 8192, 32_768] {
        let mut x = src.clone();
        rope_rows(&mut x, rows, heads, dh, pos0, THETA_QWEN);
        for (i, (a, b)) in src
            .chunks(width)
            .zip(x.chunks(width))
            .enumerate()
        {
            let (n0, n1) = (norm(a), norm(b));
            assert!(n1.is_finite(), "pos={} 行 {i}: 非有限値", pos0);
            let rel = (n0 - n1).abs() / n0.max(1e-30);
            assert!(
                rel <= 1e-5,
                "pos={pos0} 行 {i}: ノルム保存が崩れた (src={n0:.9}, rot={n1:.9}, rel={rel:.3e})"
            );
        }
    }
}

/// 直交変換なので 2 点間距離も保存される。位置が大きいと角度の評価誤差は
/// 増えるが、ペア内で同じ cos/sin を使うため距離の誤差は丸め誤差のまま。
#[test]
fn pairwise_distance_preserved_at_large_positions() {
    let (heads, dh) = (2usize, 32usize);
    let width = heads * dh;
    let mut rng = Lcg::new(1639_3);
    let a = rng.vec(width);
    let b = rng.vec(width);

    for &pos in &[8192usize, 32_768] {
        let (mut ra, mut rb) = (a.clone(), b.clone());
        rope_row(&mut ra, heads, dh, pos, THETA_QWEN);
        rope_row(&mut rb, heads, dh, pos, THETA_QWEN);
        let (d0, d1) = (dist(&a, &b), dist(&ra, &rb));
        let rel = (d0 - d1).abs() / d0.max(1e-30);
        assert!(
            rel <= 1e-5,
            "pos={pos}: 距離保存が崩れた (src={d0:.9}, rot={d1:.9}, rel={rel:.3e})"
        );
    }
}

/// 反復呼び出しはビット一致 (入力以外の状態を持たない)。
#[test]
fn repeated_invocation_is_bit_identical() {
    let (heads, dh) = (4usize, 64usize);
    let mut rng = Lcg::new(1639_4);
    let src = rng.vec(2 * heads * dh);

    for &pos in &[0usize, 1, 8192, 32_768] {
        let mut first = src.clone();
        rope_rows(&mut first, 2, heads, dh, pos, THETA_QWEN);
        for round in 1..5 {
            let mut again = src.clone();
            rope_rows(&mut again, 2, heads, dh, pos, THETA_QWEN);
            assert_eq!(
                bits(&first),
                bits(&again),
                "pos={pos} の {round} 回目がビット一致しない"
            );
        }
    }
}

/// dh=2, heads=1 では回転角が `pos` (d=0 の指数は theta^0 = 1)。
#[test]
fn dh_two_rotates_by_exact_position_angle() {
    for &pos in &[1usize, 7, 8192] {
        let mut x = [0.75f32, -1.25];
        rope_row(&mut x, 1, 2, pos, THETA_QWEN);
        let ang = pos as f32;
        let (c, s) = (ang.cos(), ang.sin());
        let (x1, x2) = (0.75f32, -1.25f32);
        let expect = [x1 * c - x2 * s, x1 * s + x2 * c];
        assert_eq!(bits(&x), bits(&expect), "pos={pos}: 角度が pos と一致しない");
    }
}

/// 角度は `pos * theta^(-2d/dh)`。dh=4 の d=1 は theta^(-1/2)。
/// 小さい pos で f64 参照と比べる (f32 の丸めのみ許容)。
#[test]
fn angle_follows_theta_exponent() {
    let (heads, dh, pos) = (1usize, 4usize, 2usize);
    let mut rng = Lcg::new(1639_5);
    let src = rng.vec(dh);
    let mut x = src.clone();
    rope_row(&mut x, heads, dh, pos, THETA_QWEN);

    for d in 0..dh / 2 {
        let ang = pos as f64 * (THETA_QWEN as f64).powf(-2.0 * d as f64 / dh as f64);
        let (c, s) = (ang.cos(), ang.sin());
        let (x1, x2) = (src[d] as f64, src[d + dh / 2] as f64);
        let e0 = x1 * c - x2 * s;
        let e1 = x1 * s + x2 * c;
        assert!(
            (x[d] as f64 - e0).abs() <= 1e-6 && (x[d + dh / 2] as f64 - e1).abs() <= 1e-6,
            "d={d}: 参照式と一致しない"
        );
    }
}

/// ヘッドごとに同じ角度で独立に回転する (データは混ざらない)。
#[test]
fn heads_rotate_independently_with_same_angle() {
    let (heads, dh, pos) = (3usize, 4usize, 11usize);
    let mut rng = Lcg::new(1639_6);
    let src = rng.vec(heads * dh);
    let mut x = src.clone();
    rope_row(&mut x, heads, dh, pos, THETA_QWEN);

    for h in 0..heads {
        let mut head = src[h * dh..(h + 1) * dh].to_vec();
        rope_row(&mut head, 1, dh, pos, THETA_QWEN);
        assert_eq!(
            bits(&x[h * dh..(h + 1) * dh]),
            bits(&head),
            "head={h}: 単独回転とビット一致しない"
        );
    }
}

/// 0 行・零ベクトルは安全 (パニックせず、零は零のまま)。
#[test]
fn empty_and_zero_inputs_are_safe() {
    let mut empty: Vec<f32> = Vec::new();
    rope_rows(&mut empty, 0, 4, 8, 32_768, THETA_QWEN);
    assert!(empty.is_empty());

    let mut zeros = vec![0.0f32; 4 * 8];
    rope_rows(&mut zeros, 4, 1, 8, 32_768, THETA_QWEN);
    assert!(zeros.iter().all(|v| *v == 0.0), "零ベクトルが零でなくなった");
}

/// pos で 2 回回すのは 2*pos で 1 回回すのと一致する (角度の加法性)。
/// 大きな位置では f32 の cos/sin 評価自体の誤差が乗るため、
/// 実測した上限 (下の許容値) で固定する。
#[test]
fn composing_two_rotations_matches_double_position() {
    let (heads, dh) = (2usize, 16usize);
    let mut rng = Lcg::new(1639_7);
    let src = rng.vec(heads * dh);

    for &pos in &[64usize, 1024, 8192] {
        let mut twice = src.clone();
        rope_row(&mut twice, heads, dh, pos, THETA_QWEN);
        rope_row(&mut twice, heads, dh, pos, THETA_QWEN);

        let mut direct = src.clone();
        rope_row(&mut direct, heads, dh, pos * 2, THETA_QWEN);

        for (i, (a, b)) in twice.iter().zip(&direct).enumerate() {
            assert!(
                (a - b).abs() <= 2e-4,
                "pos={pos} 添字 {i}: 合成回転と 2*pos が離れた ({a} vs {b})"
            );
        }
    }
}
