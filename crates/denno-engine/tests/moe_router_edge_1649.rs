//! task 1649: MoE ルータの top-k 選択・再正規化の端 (同値タイブレーク・k=1・k=n・
//! k>32・負 logits・0/非正規数の重み) を、公開 API `moe::moe_ffn` 越しに固定する。
//!
//! 何を見ているか:
//! - `denno_engine::model::moe::moe_ffn` は公開 API で、内部で
//!   logits -> softmax -> top-k -> 再正規化 -> expert FFN をこの順に回す。
//!   選択関数 (`select_topk*`) と再正規化 (`router_renorm*`) は `pub(crate)` なので
//!   `tests/` からは直接呼べない。代わりに **expert の出力を観測可能な形に固定** し、
//!   出力から「どの expert が選ばれ、どの重みで足されたか」を読み取る。
//! - 期待値はこのファイル内の独立参照実装 (softmax / 走査 top-k / 再正規化) で作り、
//!   実装の関数を呼んで自分自身と比較しない。演算順序を実装と同じに揃えてあるので
//!   比較はビット一致で行う。
//!
//! 観測の作り方 (n_embd = n_expert = ne, n_ffn = 1):
//! - `gate_inp` は全 0 + `gate_inp_bias = logits` → logits がそのまま手で読める
//!   (0.0 * x の逐次加算は厳密に 0.0、そこに bias を足すだけ)。
//! - expert e の FFN は `gate_exps[e][0] = [1.0, 0, ...]` / `up_exps[e][0] = [1.0, 0, ...]`
//!   / `down_exps[e][j] = (j == e)` なので `h = A * e_e`、`A = silu(x[0]) * x[0]`。
//! - よって `out[e] = weight_e * A` (未選択の expert は厳密に +0.0)。
//!
//! 一様フォールバック (選択和が 0 / 非有限のとき 1/k で割る規則) は
//! `prefill::router_renorm_sum` の側にあり、`pub(crate)` で `tests/` からは呼べない。
//! さらに `moe_ffn` の再正規化は選択和が常に正 (最大確率の expert は必ず選ばれる) と
//! いう前提でフォールバックを持たないため、有限 logits では到達しない。
//! ここでは到達可能な範囲として「選択確率が同値 → 重みが一様 1/k」を固定し、
//! あわせてフォールバック不要の前提 (選択和が有限かつ正) を各ケースで検査する。
//!
//! CPU のみ・単スレッド・決定論的。GPU / CUDA / ファイル / 環境変数は触らない。

use denno_engine::model::moe::{moe_ffn, MoeParams};

/// 観測用の重み一式 (n_embd = n_expert = ne, n_ffn = 1)。
struct Holder {
    x: Vec<f32>,
    gate_inp: Vec<f32>,
    bias: Vec<f32>,
    gate_exps: Vec<f32>,
    up_exps: Vec<f32>,
    down_exps: Vec<f32>,
}

impl Holder {
    /// `logits` がそのままルータ logits になる最小構成を作る。
    fn new(logits: &[f32]) -> Self {
        let ne = logits.len();
        assert!(ne >= 1, "expert は 1 個以上");
        let mut x = vec![0.0f32; ne];
        x[0] = 2.0; // silu(2) * 2 = A (> 0, 非ゼロ)
        let mut gate_exps = vec![0.0f32; ne * ne]; // [e][r=0][c]
        let mut up_exps = vec![0.0f32; ne * ne]; // [e][r=0][c]
        let mut down_exps = vec![0.0f32; ne * ne]; // [e][j=0][c]
        for e in 0..ne {
            // expert e の唯一の FFN 行は入力の x[0] だけを見る。
            gate_exps[e * ne] = 1.0;
            up_exps[e * ne] = 1.0;
            // down は expert e 番目の出力成分だけに流す (h = A * e_e)。
            down_exps[e * ne + e] = 1.0;
        }
        // gate_exps は [e][r=0][c=0] = 1.0 (n_ffn = 1 なので e 番目そのもの)。
        Self {
            x,
            gate_inp: vec![0.0f32; ne * ne],
            bias: logits.to_vec(),
            gate_exps,
            up_exps,
            down_exps,
        }
    }

    fn params(&self, k: usize, norm: bool) -> MoeParams<'_> {
        let ne = self.bias.len();
        MoeParams {
            n_embd: ne,
            n_ffn: 1,
            n_expert: ne,
            n_expert_used: k,
            norm_topk_prob: norm,
            gate_inp: &self.gate_inp,
            gate_inp_bias: Some(&self.bias),
            gate_exps: &self.gate_exps,
            up_exps: &self.up_exps,
            down_exps: &self.down_exps,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        }
    }

    fn run(&self, k: usize, norm: bool) -> Vec<f32> {
        let p = self.params(k, norm);
        let mut out = vec![0.0f32; self.x.len()];
        moe_ffn(&self.x, &p, &mut out);
        out
    }
}

/// 1 expert あたりの出力ゲイン `A = silu(2.0) * 2.0`。
/// 実装は `act[0] = 2.0` → `v/(1+exp(-v))` → `act[0] *= 2.0` の順なので同じ順で書く。
fn gain() -> f32 {
    let s = 2.0f32 / (1.0 + (-2.0f32).exp());
    s * 2.0
}

/// 仕様の計算手順をそのまま f32 で書いた参照 softmax (`tensor::softmax` を呼ばない)。
fn softmax_ref(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in x.iter_mut() {
            *v /= sum;
        }
    }
}

/// 規則の参照実装: 走査を index 昇順に行い、厳密比較 `>` で更新する
/// (同値は小さい index が勝つ)。返り値は index 昇順。
fn topk_ref(probs: &[f32], k: usize) -> Vec<usize> {
    assert!(k >= 1 && k <= probs.len(), "k の範囲外");
    let mut used = vec![false; probs.len()];
    let mut sel = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best: Option<usize> = None;
        for i in 0..probs.len() {
            if used[i] {
                continue;
            }
            match best {
                None => best = Some(i),
                Some(b) => {
                    if probs[i] > probs[b] {
                        best = Some(i);
                    }
                }
            }
        }
        let b = best.expect("k <= n なので候補は残る");
        used[b] = true;
        sel.push(b);
    }
    sel.sort_unstable();
    sel
}

/// 規則の参照実装: 選択 index 昇順に逐次加算した和で割る (0/0 のフォールバックつき)。
fn renorm_ref(probs: &[f32], sel: &[usize]) -> Vec<f32> {
    let mut sum = 0.0f32;
    for &i in sel {
        sum += probs[i];
    }
    if sum.is_finite() && sum > 0.0 {
        sel.iter().map(|&i| probs[i] / sum).collect()
    } else {
        let u = 1.0f32 / sel.len().max(1) as f32;
        vec![u; sel.len()]
    }
}

/// 参照側の logits -> softmax -> top-k (確率と選択 index)。
/// 重みは再正規化の on/off で変わるので呼び出し側で作る。
fn route_probs_sel(logits: &[f32], k: usize) -> (Vec<f32>, Vec<usize>) {
    let mut probs = logits.to_vec();
    softmax_ref(&mut probs);
    let sel = topk_ref(&probs, k);
    (probs, sel)
}

/// 期待出力: 選択 expert の位置だけ `w * A`、他は厳密に +0.0。
fn expected_out(ne: usize, sel: &[usize], w: &[f32]) -> Vec<f32> {
    let a = gain();
    let mut out = vec![0.0f32; ne];
    for (t, &i) in sel.iter().enumerate() {
        out[i] = w[t] * a;
    }
    out
}

/// ビット一致の比較 (差が出たら位置と bit も出す)。
fn expect_bits(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: 長さが違う");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{what}: [{i}] got {g} ({:#010x}) != want {w} ({:#010x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

/// 出力から「重みが正の選択 expert」を index 昇順で読む (未選択は +0.0)。
fn nonzero_indices(out: &[f32]) -> Vec<usize> {
    out.iter()
        .enumerate()
        .filter(|(_, &v)| v != 0.0)
        .map(|(i, _)| i)
        .collect()
}

/// 1 ケースを回して、選択・重み・出力の 3 つを参照と突き合わせる。
fn check_case(label: &str, logits: &[f32], k: usize, norm: bool) -> Vec<f32> {
    let ne = logits.len();
    let h = Holder::new(logits);
    let out = h.run(k, norm);
    let (probs, sel) = route_probs_sel(logits, k);

    assert_eq!(sel.len(), k, "{label}: 選択数");
    // 選択和は有限かつ正 (一様フォールバックに入らない前提)。有限 logits では
    // 最大確率の expert が必ず選ばれるので常に成り立つ。
    let mut sum = 0.0f32;
    for &i in &sel {
        sum += probs[i];
    }
    assert!(
        sum.is_finite() && sum > 0.0,
        "{label}: 選択和が有限正でない (={sum})"
    );

    if norm {
        let w = renorm_ref(&probs, &sel);
        let ws: f32 = w.iter().sum();
        assert!(
            (ws - 1.0).abs() <= 1e-6,
            "{label}: 重みの和 {ws} が 1 から外れすぎ"
        );
        let want = expected_out(ne, &sel, &w);
        expect_bits(&out, &want, label);
    } else {
        // 再正規化しないときは softmax 確率そのもの (和は 1 未満)。
        let w: Vec<f32> = sel.iter().map(|&i| probs[i]).collect();
        let ws: f32 = w.iter().sum();
        assert!(ws <= 1.0 + 1e-6, "{label}: 未正規化重みの和 {ws} が 1 超");
        let want = expected_out(ne, &sel, &w);
        expect_bits(&out, &want, label);
    }

    // 全要素が有限 (0/0 も inf も出ていない)。
    assert!(
        out.iter().all(|v| v.is_finite()),
        "{label}: 非有限の出力 {:?}",
        out
    );

    // 決定論: 2 回目も同じビット。
    let out2 = h.run(k, norm);
    expect_bits(&out2, &out, label);
    out
}

// ---------------------------------------------------------------------------
// 同値 (タイブレークは小さい index が勝つ)
// ---------------------------------------------------------------------------

/// 全 logits 同値 → 選択は先頭 k 個 (index 昇順)、再正規化は一様 1/k。
#[test]
fn all_equal_logits_pick_lowest_indices_and_uniform_weights() {
    let logits = [0.0f32; 6];
    let out = check_case("all_equal", &logits, 3, true);
    assert_eq!(nonzero_indices(&out), vec![0, 1, 2], "同値は小さい index");
    let a = gain();
    let want = a / 3.0;
    for (i, &v) in out.iter().enumerate() {
        if i < 3 {
            assert!(
                (v - want).abs() <= 4.0 * f32::EPSILON * want.abs(),
                "一様重み: [{i}] {v} != {want}"
            );
        } else {
            assert_eq!(v.to_bits(), 0.0f32.to_bits(), "未選択は +0.0");
        }
    }
}

/// 0 の同値が 3 本 + 負値: k=3 は 0 の同値 3 本 (index 0, 2, 3) を取る。
#[test]
fn tied_zero_logits_beat_negative_ones() {
    let logits = [0.0f32, -1.0, 0.0, 0.0, -2.0, 0.0];
    let h = Holder::new(&logits);
    check_case("tied_zero", &logits, 3, true);
    let out = h.run(3, true);
    assert_eq!(nonzero_indices(&out), vec![0, 2, 3]);
}

/// 最大値が同値 (k=1): 最小 index が勝ち、重みは厳密に 1.0。
#[test]
fn argmax_tie_k1_takes_lowest_index_weight_one() {
    let logits = [1.0f32, 3.0, 3.0, 2.0];
    let out = check_case("argmax_tie_k1", &logits, 1, true);
    assert_eq!(nonzero_indices(&out), vec![1], "同値最大は小さい index");
    expect_bits(&out, &{
        let mut v = vec![0.0f32; 4];
        v[1] = gain();
        v
    }, "k1_weight_one");
}

/// k>32 で k 番目の境界に同値がある場合: 小さい index が残り、選択は 0..36。
#[test]
fn k_above_32_with_tie_at_cut_boundary() {
    // logits[i] = -i。i=35 と i=36 を同値 (-35.0) にして k=36 の境界で競合させる。
    let mut logits: Vec<f32> = (0..40).map(|i| -(i as f32)).collect();
    logits[36] = -35.0;
    let out = check_case("k_gt_32", &logits, 36, true);
    let sel: Vec<usize> = (0..36).collect();
    assert_eq!(nonzero_indices(&out), sel, "k>32 の選択集合");
}

/// k=n: 全 expert を index 昇順に取り、再正規化後の和は 1。
#[test]
fn k_equals_n_selects_all_in_index_order() {
    let logits = [0.5f32, 0.5, -1.0, 2.0];
    let out = check_case("k_n", &logits, 4, true);
    assert_eq!(nonzero_indices(&out), vec![0, 1, 2, 3]);
    let sum: f32 = out.iter().sum();
    let want = gain(); // 重みの和が 1 なので A 倍
    assert!(
        (sum - want).abs() <= 8.0 * f32::EPSILON * want,
        "k=n の総和 {sum} != {want}"
    );
}

/// 全 logits 負でも順序は保たれる (0 に近い = 大きい が勝つ)。
#[test]
fn all_negative_logits_keep_ordering() {
    let logits = [-1.0f32, -3.0, -1.0, -2.0, -1.0];
    let out = check_case("all_negative", &logits, 2, true);
    assert_eq!(nonzero_indices(&out), vec![0, 2], "-1.0 の同値のうち小さい 2 本");
}

// ---------------------------------------------------------------------------
// 0 / 非正規数の重み
// ---------------------------------------------------------------------------

/// exp のアンダーフローで確率が厳密に 0 になる expert: 重み 0、出力 0、NaN なし。
#[test]
fn zero_probability_experts_get_zero_weight() {
    let logits = [0.0f32, -200.0, -200.0, -200.0];
    let out = check_case("zero_prob", &logits, 4, true);
    assert_eq!(out[0].to_bits(), gain().to_bits(), "最大 expert の重みは 1");
    for i in 1..4 {
        assert_eq!(out[i].to_bits(), 0.0f32.to_bits(), "確率 0 の expert は +0.0");
        assert!(out[i].is_sign_positive(), "{i}: -0.0 になっている");
    }
}

/// 非正規数まで落ちる確率でも有限・非負を保ち、出力が単調に減る。
#[test]
fn subnormal_probability_weights_stay_finite() {
    let logits = [0.0f32, -100.0, -80.0, -60.0];
    let out = check_case("subnormal", &logits, 4, true);
    assert!(out[1] > 0.0, "非正規数の確率が消えている");
    assert!(
        out[1] < f32::MIN_POSITIVE,
        "非正規数の領域でない: {:e}",
        out[1]
    );
    assert!(
        out[1] < out[2] && out[2] < out[3] && out[3] < out[0],
        "logits の単調性が壊れている: {:?}",
        out
    );
}

// ---------------------------------------------------------------------------
// 再正規化の on / off
// ---------------------------------------------------------------------------

/// norm_topk_prob = false では softmax 確率をそのまま使い、和は 1 未満になる。
#[test]
fn no_renorm_uses_raw_softmax_probabilities() {
    let logits = [0.0f32, 1.0, 2.0, 3.0, 3.0, -1.0];
    let h = Holder::new(&logits);
    let out = h.run(2, false);
    check_case("no_renorm", &logits, 2, false);
    let mut probs = logits.to_vec();
    softmax_ref(&mut probs);
    // 同値最大 3.0 の 2 本 (index 3, 4) が選ばれる。
    assert_eq!(nonzero_indices(&out), vec![3, 4]);
    let ws: f32 = probs.iter().sum();
    assert!((ws - 1.0).abs() <= 1e-6, "softmax の和 {ws}");
    let sel_sum = probs[3] + probs[4];
    assert!(
        sel_sum < 1.0,
        "k<n の選択和が 1 以上: 再正規化していない前提が崩れている"
    );
    let got_sum: f32 = out.iter().sum();
    let want = sel_sum * gain();
    assert!(
        (got_sum - want).abs() <= 8.0 * f32::EPSILON * want,
        "未正規化の総和 {got_sum} != {want}"
    );
}
