//! P-MOE-D1: hyper-connection の CPU 参照実装 (`denno_engine::model::hyper`) のテスト。
//!
//! GPU / CUDA / 推論経路は一切通さない。実モデルの重みも読まない
//! (形状は実ファイルに合わせた小さな合成データで検証する)。
//!
//! ビット一致を見るテストは `f32::to_bits` で比較する
//! (丸め順序が変われば最下位ビットが動くので、`==` では不十分な箇所がある)。

use denno_engine::model::hyper::{
    hc_combine, hc_head, hc_mix, sigmoid, silu, HyperParams, HyperStreams,
};
use denno_engine::tensor::rmsnorm;

/// 決定的な数列 (テスト内で固定 seed)。プラットフォーム非依存。
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// [-1, 1) の f32。上位 24 bit を使うので常に同じ値になる。
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 40) as u32; // 24 bit
        u as f32 / 8388608.0 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn assert_bits_eq(actual: &[f32], expected: &[f32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: 長さが違う");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            e.to_bits(),
            "{what}: index {i} がビット一致しない (actual={a:e} expected={e:e})"
        );
    }
}

fn params(hc: usize, n_embd: usize, low_rank: usize, eps: f32) -> HyperParams {
    HyperParams { hc, n_embd, low_rank, eps }
}

// ---------------------------------------------------------------------------
// 1. inject = 0 のとき hc_combine は素の residual 加算 (w = 1)
// ---------------------------------------------------------------------------

#[test]
fn combine_zero_inject_is_plain_residual_add() {
    let p = params(3, 4, 2, 1e-6);
    let residual = HyperStreams {
        // -0.0 は入れない (residual + 0.0 で符号が変わり得るため)
        data: vec![
            0.25, -1.5, 3.0, 0.0, //
            -0.75, 2.5, -4.25, 1.0, //
            7.5, -0.5, 0.125, -2.0,
        ],
        hc: 3,
        n_embd: 4,
    };
    let inject = vec![0.0f32; 3];

    // (a) block_out = 0 なら residual とビット一致
    let zero = vec![0.0f32; 4];
    let out = hc_combine(&residual, &zero, &inject, &p);
    assert_eq!(out.hc, 3);
    assert_eq!(out.n_embd, 4);
    assert_bits_eq(&out.data, &residual.data, "inject=0, block_out=0");

    // (b) block_out があっても「素の加算」とビット一致 (w = 2*sigmoid(0) = 1.0)
    let block = vec![1.0f32, -2.0, 0.5, 4.0];
    let out = hc_combine(&residual, &block, &inject, &p);
    let mut expect = Vec::with_capacity(12);
    for s in 0..3 {
        for d in 0..4 {
            expect.push(residual.data[s * 4 + d] + block[d]);
        }
    }
    assert_bits_eq(&out.data, &expect, "inject=0, block_out あり");

    // (c) inject が非 0 なら結果は変わる (w が効いていることの確認)
    let inject_nz = vec![0.5f32, -0.25, 2.0];
    let out_nz = hc_combine(&residual, &block, &inject_nz, &p);
    assert_ne!(out_nz.data, out.data, "inject 非 0 でも同じ値になっている");

    // w = 2*sigmoid(inject) でストリームごとに重みが違う
    for s in 0..3 {
        let w = 2.0f32 * sigmoid(inject_nz[s]);
        for d in 0..4 {
            let got = out_nz.data[s * 4 + d];
            let want = residual.data[s * 4 + d] + block[d] * w;
            assert_eq!(got.to_bits(), want.to_bits(), "stream {s} dim {d}");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. from_embed は hc 本すべて同じ内容
// ---------------------------------------------------------------------------

#[test]
fn from_embed_repeats_hc_copies() {
    let p = params(4, 5, 3, 1e-6);
    let embd = vec![0.5f32, -1.25, 0.0, 3.5, -0.125];
    let x = HyperStreams::from_embed(&embd, &p);
    assert_eq!(x.hc, 4);
    assert_eq!(x.n_embd, 5);
    assert_eq!(x.data.len(), 20);
    for s in 0..4 {
        assert_bits_eq(x.stream(s), &embd, &format!("stream {s}"));
    }
}

// ---------------------------------------------------------------------------
// 3. mean は「s の昇順に逐次加算 → 1/hc 倍」とビット一致
// ---------------------------------------------------------------------------

#[test]
fn mean_is_ascending_sum_then_scale() {
    let mut rng = Lcg::new(0x51ED_0001);
    let data = rng.vec(12);
    let x = HyperStreams { data: data.clone(), hc: 4, n_embd: 3 };

    let got = x.mean();
    // 期待値: 参照実装 hc_stream_mean と同じ順序 (acc = s0; acc += s1; ... ; *1/hc)
    let mut expect = Vec::with_capacity(3);
    for d in 0..3 {
        let mut acc = data[d];
        for s in 1..4 {
            acc += data[s * 3 + d];
        }
        expect.push(acc * (1.0f32 / 4.0));
    }
    assert_bits_eq(&got, &expect, "mean");

    // 1/hc が 2 の冪でない hc でも同じ順序であること (hc=3)
    let data3 = rng.vec(9);
    let x3 = HyperStreams { data: data3.clone(), hc: 3, n_embd: 3 };
    let mut expect3 = Vec::with_capacity(3);
    for d in 0..3 {
        let mut acc = data3[d];
        for s in 1..3 {
            acc += data3[s * 3 + d];
        }
        expect3.push(acc * (1.0f32 / 3.0));
    }
    assert_bits_eq(&x3.mean(), &expect3, "mean (hc=3)");
}

// ---------------------------------------------------------------------------
// 4. gamma = 1 / w_down = w_up = 0 → gate は 0、mixed は入力の平均の 0.5 倍
// ---------------------------------------------------------------------------

/// タスク仕様の「w_down/w_up を 0 にすると mixed = 入力の平均」は、
/// gate = 0 から `sigmoid(0) = 0.5` が掛かるため **0.5 倍**が正しい
/// (`gated = xn * 0.5`)。ここでは契約どおりの値
/// `mixed[d] = 0.5 * (1/hc) * sum_s xn[s][d]` をビット一致で固定する。
#[test]
fn zero_weights_give_half_mean_of_normed_input() {
    let p = params(3, 4, 2, 0.0);
    let embd = vec![1.0f32, 1.0, 1.0, 1.0];
    let x = HyperStreams::from_embed(&embd, &p);
    let w_norm = vec![1.0f32; 12];
    let w_down = vec![0.0f32; 2 * 12];
    let w_up = vec![0.0f32; 12 * 2];

    let (mixed, inject) = hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);

    // eps = 0 で全要素 1.0 のストリームは RMSNorm 後も 1.0 (mean(x^2) = 1)
    let xn = vec![1.0f32; 12];
    let mut expect = Vec::with_capacity(4);
    for d in 0..4 {
        let mut acc = xn[d] * 0.5;
        for s in 1..3 {
            acc += xn[s * 4 + d] * 0.5;
        }
        expect.push(acc * (1.0f32 / 3.0));
    }
    assert_bits_eq(&mixed, &expect, "zero weights mixed");
    assert_bits_eq(&mixed, &vec![0.5f32; 4], "zero weights mixed = 0.5");
    assert_bits_eq(&inject, &vec![0.0f32; 3], "inject は 0 で返る");

    // ストリームの内容が違っても「0.5 × 平均(正規化後)」であること
    let mut rng = Lcg::new(0x51ED_0004);
    let data = rng.vec(12);
    let x2 = HyperStreams { data: data.clone(), hc: 3, n_embd: 4 };
    let (mixed2, _) = hc_mix(&x2, &w_norm, &w_down, &w_up, None, &p);
    let mut expect2 = Vec::with_capacity(4);
    for d in 0..4 {
        // xn は crate::tensor::rmsnorm と同じ順序 (x * inv * w、w = 1)
        let mut acc = 0.0f32;
        for s in 0..3 {
            let xn_s = rmsnorm(&data[s * 4..(s + 1) * 4], &[1.0f32; 4], 4, 0.0);
            acc += xn_s[d] * 0.5;
        }
        expect2.push(acc * (1.0f32 / 3.0));
    }
    assert_bits_eq(&mixed2, &expect2, "zero weights mixed (異なるストリーム)");
}

// ---------------------------------------------------------------------------
// 5. w_down_inject あり / なしで lo がビット一致 (分割の正しさ)
// ---------------------------------------------------------------------------

#[test]
fn down_inject_split_keeps_lo_bit_exact() {
    let p = params(2, 4, 3, 1e-6);
    let mut rng = Lcg::new(0x51ED_0005);
    let embd = rng.vec(4);
    let x = HyperStreams::from_embed(&embd, &p);
    let w_norm = rng.vec(8);
    let w_down = rng.vec(3 * 8); // [low_rank=3][hc_dim=8]
    let w_up = rng.vec(8 * 3); // [hc_dim=8][low_rank=3]

    // 結合行列: 行 0..low_rank が w_down、行 low_rank..low_rank+hc が inject 行
    let inject_rows = rng.vec(2 * 8);
    let mut combined = Vec::with_capacity(5 * 8);
    combined.extend_from_slice(&w_down);
    combined.extend_from_slice(&inject_rows);

    let (mixed_plain, inject_plain) = hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);
    let (mixed_comb, inject_comb) = hc_mix(&x, &w_norm, &w_down, &w_up, Some(&combined), &p);

    // lo (→ gate → mixed) は分割前後でビット一致
    assert_bits_eq(&mixed_comb, &mixed_plain, "w_down_inject あり/なしの mixed");

    // なしのとき inject は 0
    assert_bits_eq(&inject_plain, &vec![0.0f32; 2], "w_down_inject なしの inject");

    // ありのとき inject = dot(inject 行, xn) * 1/hc (silu は掛けない)
    let e = 4;
    let mut xn = Vec::with_capacity(8);
    for s in 0..2 {
        xn.extend_from_slice(&rmsnorm(x.stream(s), &w_norm[s * e..(s + 1) * e], e, p.eps));
    }
    for r in 0..2 {
        let row = &inject_rows[r * 8..(r + 1) * 8];
        let mut acc = 0.0f32;
        for i in 0..8 {
            acc += row[i] * xn[i];
        }
        let want = acc * (1.0f32 / 2.0);
        assert_eq!(inject_comb[r].to_bits(), want.to_bits(), "inject[{r}]");
        // silu を掛けていないこと (silu(want) と want が区別できる値のときだけ比較する)
        if silu(want).to_bits() != want.to_bits() {
            assert_ne!(
                inject_comb[r].to_bits(),
                silu(want).to_bits(),
                "inject[{r}] に silu が掛かっている"
            );
        }
    }

    // hc_head は hc_mix の mixed と同じ (inject を捨てるだけ)
    let head = hc_head(&x, &w_norm, &w_down, &w_up, &p);
    assert_bits_eq(&head, &mixed_plain, "hc_head == hc_mix の mixed");
}

// ---------------------------------------------------------------------------
// 6. hc = 1 の境界
// ---------------------------------------------------------------------------

#[test]
fn hc_one_is_passthrough() {
    let p = params(1, 5, 2, 1e-6);
    let embd = vec![0.5f32, -1.25, 2.0, 0.0, -0.125];
    let x = HyperStreams::from_embed(&embd, &p);
    assert_eq!(x.data.len(), 5);
    assert_bits_eq(&x.data, &embd, "hc=1 の from_embed");
    assert_bits_eq(x.stream(0), &embd, "hc=1 の stream(0)");
    // mean は「1 要素の総和 × 1/1」= 素通し (1.0 倍は無誤差)
    assert_bits_eq(&x.mean(), &embd, "hc=1 の mean");

    // combine も inject = 0 なら素の加算
    let block = vec![1.0f32, 2.0, -3.0, 0.5, 4.0];
    let out = hc_combine(&x, &block, &[0.0], &p);
    let mut expect = Vec::with_capacity(5);
    for d in 0..5 {
        expect.push(embd[d] + block[d]);
    }
    assert_bits_eq(&out.data, &expect, "hc=1 の combine");
}

// ---------------------------------------------------------------------------
// 7. silu / sigmoid の既知値
// ---------------------------------------------------------------------------

#[test]
fn silu_sigmoid_known_values() {
    assert_eq!(silu(0.0).to_bits(), 0.0f32.to_bits(), "silu(0)");
    assert_eq!(sigmoid(0.0).to_bits(), 0.5f32.to_bits(), "sigmoid(0) = 0.5 は厳密");
    let s1 = silu(1.0);
    let sm1 = silu(-1.0);
    assert!((s1 - 0.7310586f32).abs() < 1e-7, "silu(1) = {s1}");
    assert!((sm1 - -0.2689414f32).abs() < 1e-7, "silu(-1) = {sm1}");
    // silu(x) = x * sigmoid(x) の定義どおりであること
    for v in [-3.0f32, -0.5, 0.25, 2.0] {
        assert_eq!(silu(v).to_bits(), (v * sigmoid(v)).to_bits(), "silu({v})");
    }
}

// ---------------------------------------------------------------------------
// 8. ランダム重みでの決定的な参照値 (期待値はハードコード)
// ---------------------------------------------------------------------------

/// 固定 seed の数列から重みを作り、mixed / inject / combine の参照値を固定する。
/// 丸め順序 (dot の逐次加算・1/hc の掛ける位置・mean の加算順) を変えると落ちる。
///
/// 期待値は本実装から採取したうえで、**同じ演算順序を f32 で厳密に写した独立実装
/// (Python、逐次加算・f32 丸めを各演算ごとに適用) でもビット一致することを確認**して
/// 固定している (自己一致だけのアンカーにしないため)。
#[test]
fn anchor_random_weights() {
    let p = params(2, 4, 3, 1e-6);
    let mut rng = Lcg::new(0x51ED_0008);
    let embd = rng.vec(4);
    let x = HyperStreams::from_embed(&embd, &p);
    let w_norm = rng.vec(8);
    let w_down = rng.vec(3 * 8);
    let w_up = rng.vec(8 * 3);
    let inject_rows = rng.vec(2 * 8);
    let mut combined = Vec::with_capacity(5 * 8);
    combined.extend_from_slice(&w_down);
    combined.extend_from_slice(&inject_rows);

    let (mixed, inject) = hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);
    let (mixed_comb, inject_comb) = hc_mix(&x, &w_norm, &w_down, &w_up, Some(&combined), &p);
    let head = hc_head(&x, &w_norm, &w_down, &w_up, &p);

    let expect_mixed: [f32; 4] = ANCHOR_MIXED;
    let expect_inject_comb: [f32; 2] = ANCHOR_INJECT;

    assert_bits_eq(&mixed, &expect_mixed, "anchor mixed");
    assert_bits_eq(&inject, &[0.0f32; 2], "anchor inject (w_down_inject なし)");
    assert_bits_eq(&mixed_comb, &expect_mixed, "anchor mixed (w_down_inject あり)");
    assert_bits_eq(&inject_comb, &expect_inject_comb, "anchor inject");
    assert_bits_eq(&head, &expect_mixed, "anchor hc_head");

    // combine も固定する
    let block = [0.5f32, -1.5, 2.25, -0.75];
    let out = hc_combine(&x, &block, &expect_inject_comb, &p);
    let expect_combine: [f32; 8] = ANCHOR_COMBINE;
    assert_bits_eq(&out.data, &expect_combine, "anchor combine");
}

const ANCHOR_MIXED: [f32; 4] = [-0.2145072, -0.1952698, 0.053666078, -0.19590339];
const ANCHOR_INJECT: [f32; 2] = [-0.0070681423, 0.17874321];
const ANCHOR_COMBINE: [f32; 8] = [
    -0.050585628,
    -0.6392422,
    1.6860697,
    -0.19153804,
    -0.004251361,
    -0.778245,
    1.8945739,
    -0.26103944,
];

// ---------------------------------------------------------------------------
// 9. 実ファイルの形状 (hc=4, n_embd=2560, low_rank=320) でのスモーク
// ---------------------------------------------------------------------------

/// 実ファイルの形状で落ちないこと・決定的であることだけを見る (値の正しさは
/// 上の小さな形状のテストで固定済み)。重みは LCG の合成データ。
#[test]
fn real_shape_smoke() {
    let p = params(4, 2560, 320, 1e-6);
    let hc_dim = p.hc * p.n_embd;
    let mut rng = Lcg::new(0x51ED_0009);
    let embd = rng.vec(p.n_embd);
    let x = HyperStreams::from_embed(&embd, &p);
    let w_norm = rng.vec(p.hc * p.n_embd);
    let w_down = rng.vec(p.low_rank * hc_dim);
    let w_up = rng.vec(hc_dim * p.low_rank);
    let mut combined = Vec::with_capacity((p.low_rank + p.hc) * hc_dim);
    combined.extend_from_slice(&w_down);
    combined.extend_from_slice(&rng.vec(p.hc * hc_dim));

    let (mixed, inject) = hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);
    assert_eq!(mixed.len(), p.n_embd);
    assert_eq!(inject.len(), p.hc);
    assert!(mixed.iter().all(|v| v.is_finite()), "mixed に非有限値");
    assert!(inject.iter().all(|v| v.is_finite()), "inject に非有限値");

    // 同じ入力なら 2 回目もビット一致 (決定性)
    let (mixed2, _) = hc_mix(&x, &w_norm, &w_down, &w_up, None, &p);
    assert_bits_eq(&mixed2, &mixed, "real shape の決定性");

    // 結合行列経由でも mixed は同じ
    let (mixed_comb, inject_comb) = hc_mix(&x, &w_norm, &w_down, &w_up, Some(&combined), &p);
    assert_bits_eq(&mixed_comb, &mixed, "real shape の w_down_inject 一致");
    assert!(inject_comb.iter().all(|v| v.is_finite()));

    let out = hc_combine(&x, &mixed, &inject_comb, &p);
    assert_eq!(out.data.len(), p.hc * p.n_embd);
    assert!(out.data.iter().all(|v| v.is_finite()), "combine に非有限値");
}

