//! task 1644: `denno_engine::tensor` の要素演算の極値・境界テスト。
//!
//! 対象は公開 API の純関数のみ (`softmax` / `rmsnorm` / `rmsnorm_into` /
//! `layernorm` / `silu_inplace` / `gelu_inplace` / `qknorm_rows_inplace` /
//! `add_bias`)。次を縛る:
//! - 全ゼロベクトル: 出力が 0 / 一様 / バイアスのみになる。
//! - 単一要素ベクトル: softmax = 1、rmsnorm / silu / gelu の閉形式。
//! - 大きな振幅 (±1000): 内部の exp が f32 で溢れても NaN/Inf を出さない
//!   (softmax の max 減算、silu/gelu の飽和)。rmsnorm は 2 乗和が有限域。
//! - 極小 eps と subnormal 入力: 2 乗和が 0 に潰れても有限値を返す。
//! - 行分割 (layernorm / rmsnorm / qknorm) が行ごとに独立であること。
//! - 同じ入力で 2 回呼ぶとビット一致 (決定論)。
//!
//! CPU-only・高速。`cargo test --release -p denno-engine --test tensor_math_edge_1644`。

use denno_engine::tensor::{
    add_bias, gelu_inplace, layernorm, qknorm_rows_inplace, rmsnorm, rmsnorm_into, silu_inplace,
    softmax,
};

/// 絶対誤差で比較する (極値テストなので相対誤差より読みやすい)。
#[track_caller]
fn approx(got: f32, want: f32, tol: f32, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got={got}, want={want}, tol={tol}"
    );
}

/// NaN / ±Inf が 1 つも無いことを確認する。
#[track_caller]
fn assert_finite(xs: &[f32], what: &str) {
    for (i, v) in xs.iter().enumerate() {
        assert!(v.is_finite(), "{what}: [{i}] が有限でない: {v}");
    }
}

/// ビット一致 (決定論) を確認する。
#[track_caller]
fn assert_bit_eq(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: 長さが違う");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{what}: [{i}] が不一致: {x} vs {y}"
        );
    }
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

#[test]
fn softmax_all_zero_is_uniform() {
    for n in [1usize, 2, 7, 64] {
        let mut x = vec![0.0f32; n];
        softmax(&mut x);
        assert_finite(&x, "softmax(all-zero)");
        let sum: f32 = x.iter().sum();
        approx(sum, 1.0, 1e-6, "softmax(all-zero) の総和");
        for &v in &x {
            approx(v, 1.0 / n as f32, 1e-6, "softmax(all-zero) の各要素");
        }
    }
}

#[test]
fn softmax_single_element_is_one() {
    for v in [0.0f32, 1.0, -1.0, 1000.0, -1000.0, f32::MIN_POSITIVE] {
        let mut x = [v];
        softmax(&mut x);
        assert_finite(&x, "softmax(1 要素)");
        approx(x[0], 1.0, 1e-6, "softmax(1 要素) の値");
    }
}

#[test]
fn softmax_large_magnitude_never_overflows() {
    // exp(1000) は f32 (最大 ≈3.4e38) を溢れる。max 減算が効いていれば
    // exp(0)=1 が最大で、結果は有限かつ総和 1。
    let mut big = [1000.0f32, 1000.0, 1000.0];
    softmax(&mut big);
    assert_finite(&big, "softmax(+1000 一様)");
    approx(big[0], 1.0 / 3.0, 1e-6, "softmax(+1000 一様) の値");
    approx(big.iter().sum(), 1.0, 1e-6, "softmax(+1000 一様) の総和");

    let mut spread = [1000.0f32, 999.0, 998.0];
    softmax(&mut spread);
    assert_finite(&spread, "softmax(+1000 分布)");
    approx(spread.iter().sum(), 1.0, 1e-6, "softmax(+1000 分布) の総和");
    assert!(spread[0] > spread[1] && spread[1] > spread[2], "順序が崩れた: {spread:?}");

    let mut neg = [-1000.0f32, -1001.0, -1002.0];
    softmax(&mut neg);
    assert_finite(&neg, "softmax(-1000 分布)");
    approx(neg.iter().sum(), 1.0, 1e-6, "softmax(-1000 分布) の総和");
    assert!(neg[0] > neg[1] && neg[1] > neg[2], "順序が崩れた: {neg:?}");

    // +1000 の一様シフトで確率が変わらない (シフト不変性)。
    let mut base = [0.5f32, 1.5, -0.25, 2.0];
    let mut shifted = [1000.5f32, 1001.5, 999.75, 1002.0];
    softmax(&mut base);
    softmax(&mut shifted);
    for i in 0..base.len() {
        approx(shifted[i], base[i], 1e-6, "softmax の +1000 シフト不変性");
    }
}

#[test]
fn softmax_empty_is_noop() {
    let mut x: Vec<f32> = Vec::new();
    softmax(&mut x);
    assert!(x.is_empty(), "空入力で要素が増えた");
}

#[test]
fn softmax_is_deterministic() {
    let src = [0.0f32, 1000.0, -1000.0, 3.5, -2.25, 0.125, 999.0, -999.0];
    let mut a = src;
    let mut b = src;
    softmax(&mut a);
    softmax(&mut b);
    assert_bit_eq(&a, &b, "softmax");
}

// ---------------------------------------------------------------------------
// rmsnorm / rmsnorm_into
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_all_zero_is_zero() {
    let w = [0.5f32, 1.0, -2.0, 3.0];
    let x = [0.0f32; 4];
    let out = rmsnorm(&x, &w, 4, 1e-6);
    assert_finite(&out, "rmsnorm(all-zero)");
    for (i, &v) in out.iter().enumerate() {
        assert_eq!(v, 0.0, "rmsnorm(all-zero): [{i}] = {v}");
    }
}

#[test]
fn rmsnorm_single_element_closed_form() {
    let w = [2.0f32];
    // [1000] * (1/sqrt(1e6 + eps)) * 2 ≈ 2。
    let out = rmsnorm(&[1000.0], &w, 1, 1e-6);
    assert_finite(&out, "rmsnorm(1 要素, 1000)");
    approx(out[0], 2.0, 1e-5, "rmsnorm(1 要素, 1000)");

    // [0] は 0。
    let z = rmsnorm(&[0.0], &w, 1, 1e-6);
    assert_eq!(z[0], 0.0, "rmsnorm(1 要素, 0)");

    // [1] は 1/sqrt(1+eps) * w の閉形式。
    let eps = 1e-6f32;
    let one = rmsnorm(&[1.0], &w, 1, eps);
    let want = 1.0 / (1.0 + eps).sqrt() * 2.0;
    approx(one[0], want, 1e-6, "rmsnorm(1 要素, 1)");
}

#[test]
fn rmsnorm_large_magnitude_stays_bounded() {
    // 交互符号 ±1000。各要素の |out| = |x| / rms(x) = 1 (w=1 のとき)。
    let x = [1000.0f32, -1000.0, 1000.0, -1000.0];
    let w = [1.0f32; 4];
    let out = rmsnorm(&x, &w, 4, 1e-6);
    assert_finite(&out, "rmsnorm(±1000)");
    for (i, &v) in out.iter().enumerate() {
        approx(v.abs(), 1.0, 1e-5, "rmsnorm(±1000) の絶対値");
        assert_eq!(
            v.signum(),
            x[i].signum(),
            "rmsnorm(±1000): [{i}] の符号が反転した"
        );
    }
}

#[test]
fn rmsnorm_tiny_eps_and_subnormal_input() {
    // 1e-40 は f32 の subnormal。2 乗和は 0 に潰れるが、
    // inv = 1/sqrt(eps) は有限なので出力も有限であること。
    let x = [1e-40f32, -1e-40f32];
    let w = [1.0f32, 1.0];
    let out = rmsnorm(&x, &w, 2, 1e-12);
    assert_finite(&out, "rmsnorm(subnormal)");
    for &v in &out {
        assert!(v.abs() < 1e-30, "rmsnorm(subnormal) の値が想定外: {v}");
    }

    // 極小 eps でも通常入力は壊れない。
    let n = rmsnorm(&[1.0f32, 2.0, 3.0], &[1.0f32; 3], 3, 1e-30);
    assert_finite(&n, "rmsnorm(eps=1e-30)");
    let ss = (1.0f32 + 4.0 + 9.0) / 3.0;
    let inv = 1.0 / (ss + 1e-30).sqrt();
    for (i, &v) in n.iter().enumerate() {
        approx(v, (i as f32 + 1.0) * inv, 1e-6, "rmsnorm(eps=1e-30) の値");
    }
}

#[test]
fn rmsnorm_rows_are_independent() {
    let cols = 4usize;
    let w = [0.5f32, 1.0, 1.5, 2.0];
    let zero_row = [0.0f32; 4];
    let live_row = [3.0f32, -1.0, 2.0, 0.5];
    let mut x = Vec::new();
    x.extend_from_slice(&zero_row);
    x.extend_from_slice(&live_row);
    let out = rmsnorm(&x, &w, cols, 1e-6);
    let alone = rmsnorm(&live_row, &w, cols, 1e-6);
    for i in 0..cols {
        assert_eq!(out[i], 0.0, "ゼロ行 [{i}] が 0 でない: {}", out[i]);
        assert_bit_eq(&out[cols + i..cols + i + 1], &alone[i..i + 1], "rmsnorm 行独立性");
    }
}

#[test]
fn rmsnorm_into_matches_allocating_version() {
    let x = [2.0f32, -3.0, 0.5, 7.0, -1.5, 0.0];
    let w = [1.0f32, 0.5, 2.0, -1.0, 0.25, 3.0];
    let want = rmsnorm(&x, &w, 3, 1e-6);
    let mut got = vec![f32::NAN; x.len()];
    rmsnorm_into(&x, &w, 3, 1e-6, &mut got);
    assert_bit_eq(&got, &want, "rmsnorm_into vs rmsnorm");
}

// ---------------------------------------------------------------------------
// layernorm
// ---------------------------------------------------------------------------

#[test]
fn layernorm_all_zero_equals_bias() {
    let cols = 4usize;
    let w = [1.0f32, 2.0, 3.0, 4.0];
    let b = [0.5f32, -0.5, 1.5, -2.0];
    let x = vec![0.0f32; cols * 3];
    let out = layernorm(&x, &w, &b, cols, 1e-5);
    assert_finite(&out, "layernorm(all-zero)");
    for row in out.chunks(cols) {
        for (i, &v) in row.iter().enumerate() {
            // 平均 0・分散 0 → 正規化項は 0 なので bias だけが残る。
            approx(v, b[i], 1e-6, "layernorm(all-zero) = bias");
        }
    }
}

#[test]
fn layernorm_single_element_equals_bias() {
    // 1 要素では平均 = その値、分散 = 0 → 常に bias。
    let out = layernorm(&[123.0f32], &[2.0f32], &[3.0f32], 1, 1e-5);
    assert_finite(&out, "layernorm(1 要素)");
    approx(out[0], 3.0, 1e-6, "layernorm(1 要素) = bias");
}

#[test]
fn layernorm_large_magnitude_stays_bounded() {
    // 交互符号 ±1000。平均 0・分散 1e6 → inv ≈ 1e-3、出力 ≈ ±w。
    let x = [1000.0f32, -1000.0, 1000.0, -1000.0];
    let w = [1.0f32; 4];
    let b = [0.0f32; 4];
    let out = layernorm(&x, &w, &b, 4, 1e-5);
    assert_finite(&out, "layernorm(±1000)");
    for (i, &v) in out.iter().enumerate() {
        approx(v.abs(), 1.0, 1e-4, "layernorm(±1000) の絶対値");
        assert_eq!(v.signum(), x[i].signum(), "layernorm(±1000): [{i}] の符号");
    }
}

#[test]
fn layernorm_rows_are_independent_and_deterministic() {
    let cols = 3usize;
    let w = [0.5f32, 1.0, 2.0];
    let b = [0.1f32, -0.2, 0.3];
    let a = [1.0f32, 2.0, 3.0];
    let c = [-4.0f32, 0.0, 8.0];
    let mut x = Vec::new();
    x.extend_from_slice(&a);
    x.extend_from_slice(&c);
    let out1 = layernorm(&x, &w, &b, cols, 1e-5);
    let out2 = layernorm(&x, &w, &b, cols, 1e-5);
    assert_bit_eq(&out1, &out2, "layernorm 決定論");
    let alone = layernorm(&c, &w, &b, cols, 1e-5);
    for i in 0..cols {
        assert_bit_eq(&out1[cols + i..cols + i + 1], &alone[i..i + 1], "layernorm 行独立性");
    }
}

#[test]
fn layernorm_empty_is_noop() {
    let out = layernorm(&[], &[1.0f32], &[0.0f32], 1, 1e-5);
    assert!(out.is_empty(), "空入力で要素が増えた");
}

// ---------------------------------------------------------------------------
// silu / gelu
// ---------------------------------------------------------------------------

#[test]
fn silu_zero_and_single() {
    let mut z = [0.0f32];
    silu_inplace(&mut z);
    assert_eq!(z[0], 0.0, "silu(0)");

    // silu(1) = 1 / (1 + e^-1)。
    let mut one = [1.0f32];
    silu_inplace(&mut one);
    approx(one[0], 1.0 / (1.0 + (-1.0f32).exp()), 1e-6, "silu(1)");

    // 空入力は no-op。
    let mut empty: [f32; 0] = [];
    silu_inplace(&mut empty);
}

#[test]
fn silu_large_magnitude_saturates() {
    // +1000: e^-1000 = 0 → 1000。
    let mut pos = [1000.0f32];
    silu_inplace(&mut pos);
    assert_finite(&pos, "silu(+1000)");
    approx(pos[0], 1000.0, 1e-3, "silu(+1000)");

    // -1000: e^1000 = Inf → -1000/Inf = -0。NaN でも Inf でもない。
    let mut neg = [-1000.0f32];
    silu_inplace(&mut neg);
    assert_finite(&neg, "silu(-1000)");
    assert!(neg[0].abs() <= 1e-6, "silu(-1000) が 0 に飽和しない: {}", neg[0]);
}

#[test]
fn silu_subnormal_input_stays_finite() {
    let mut x = [1e-40f32, -1e-40f32, f32::MIN_POSITIVE];
    silu_inplace(&mut x);
    assert_finite(&x, "silu(subnormal)");
    // 小さい正値はほぼ x/2。
    approx(x[0], 0.5e-40, 1e-42, "silu(+subnormal)");
}

#[test]
fn gelu_zero_single_and_large() {
    let mut z = [0.0f32];
    gelu_inplace(&mut z);
    assert_eq!(z[0], 0.0, "gelu(0)");

    // gelu(1) ≈ 0.841192 (tanh 近似の解析値)。
    let mut one = [1.0f32];
    gelu_inplace(&mut one);
    approx(one[0], 0.841192, 1e-6, "gelu(1)");

    // ±1000 は飽和して 1000 / -0 になる (tanh の引数は有限域)。
    let mut big = [1000.0f32, -1000.0];
    gelu_inplace(&mut big);
    assert_finite(&big, "gelu(±1000)");
    approx(big[0], 1000.0, 1e-3, "gelu(+1000)");
    assert!(big[1].abs() <= 1e-6, "gelu(-1000) が 0 に飽和しない: {}", big[1]);
}

#[test]
fn silu_gelu_deterministic() {
    let src = [0.0f32, 1000.0, -1000.0, 0.25, -0.25, 3.0, -3.0, f32::MIN_POSITIVE];
    let mut s1 = src;
    let mut s2 = src;
    silu_inplace(&mut s1);
    silu_inplace(&mut s2);
    assert_bit_eq(&s1, &s2, "silu 決定論");

    let mut g1 = src;
    let mut g2 = src;
    gelu_inplace(&mut g1);
    gelu_inplace(&mut g2);
    assert_bit_eq(&g1, &g2, "gelu 決定論");
}

// ---------------------------------------------------------------------------
// qknorm
// ---------------------------------------------------------------------------

#[test]
fn qknorm_all_zero_is_zero() {
    let w = [0.5f32, 1.0];
    let mut x = [0.0f32; 8];
    qknorm_rows_inplace(&mut x, 2, 2, 2, &w, 1e-6);
    for (i, &v) in x.iter().enumerate() {
        assert_eq!(v, 0.0, "qknorm(all-zero): [{i}] = {v}");
    }
}

#[test]
fn qknorm_single_element_and_large_magnitude() {
    // rows=1, heads=1, dh=1: rmsnorm と同じ閉形式。
    let mut one = [1000.0f32];
    qknorm_rows_inplace(&mut one, 1, 1, 1, &[2.0], 1e-6);
    assert_finite(&one, "qknorm(1 要素)");
    approx(one[0], 2.0, 1e-5, "qknorm(1 要素, 1000)");

    // dh=2 の ±1000 ヘッドは ±1 に収まる。
    let mut head = [1000.0f32, -1000.0];
    qknorm_rows_inplace(&mut head, 1, 1, 2, &[1.0, 1.0], 1e-6);
    assert_finite(&head, "qknorm(±1000)");
    approx(head[0], 1.0, 1e-5, "qknorm(±1000)[0]");
    approx(head[1], -1.0, 1e-5, "qknorm(±1000)[1]");
}

#[test]
fn qknorm_heads_are_independent() {
    let dh = 2usize;
    let heads = 2usize;
    let w = [0.5f32, 2.0];
    // 1 ヘッド目はゼロ、2 ヘッド目は非ゼロ。
    let mut x = [0.0f32, 0.0, 3.0, -4.0];
    qknorm_rows_inplace(&mut x, 1, heads, dh, &w, 1e-6);
    assert_eq!((x[0], x[1]), (0.0, 0.0), "ゼロヘッドが 0 でない");
    // 2 ヘッド目は単独で回した値とビット一致。
    let mut alone = [3.0f32, -4.0];
    qknorm_rows_inplace(&mut alone, 1, 1, dh, &w, 1e-6);
    assert_bit_eq(&x[2..4], &alone, "qknorm ヘッド独立性");
}

#[test]
fn qknorm_deterministic() {
    let w = [0.25f32, -0.5, 1.5, 2.0];
    let src = [1.0f32, -2.0, 3.0, -4.0, 1000.0, -1000.0, 0.5, f32::MIN_POSITIVE];
    let mut a = src;
    let mut b = src;
    qknorm_rows_inplace(&mut a, 1, 2, 4, &w, 1e-6);
    qknorm_rows_inplace(&mut b, 1, 2, 4, &w, 1e-6);
    assert_bit_eq(&a, &b, "qknorm 決定論");
}

// ---------------------------------------------------------------------------
// add_bias
// ---------------------------------------------------------------------------

#[test]
fn add_bias_all_zero_equals_bias() {
    let cols = 3usize;
    let bias = [1.0f32, -2.0, 0.5];
    let mut x = vec![0.0f32; cols * 4];
    add_bias(&mut x, &bias, cols);
    for row in x.chunks(cols) {
        for (i, &v) in row.iter().enumerate() {
            assert_eq!(v, bias[i], "add_bias(all-zero) 行 {i}");
        }
    }
}

#[test]
fn add_bias_single_element_and_large_magnitude() {
    let mut one = [0.0f32];
    add_bias(&mut one, &[5.0], 1);
    assert_eq!(one[0], 5.0, "add_bias(1 要素)");

    let mut big = [1000.0f32, -1000.0];
    add_bias(&mut big, &[1000.0, -1000.0], 2);
    assert_finite(&big, "add_bias(±1000)");
    approx(big[0], 2000.0, 1e-3, "add_bias(+1000)");
    approx(big[1], -2000.0, 1e-3, "add_bias(-1000)");
}

#[test]
fn add_bias_empty_is_noop() {
    let mut x: [f32; 0] = [];
    add_bias(&mut x, &[1.0, 2.0], 2);
    assert!(x.is_empty());
}
