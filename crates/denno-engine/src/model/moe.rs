//! MoE (Mixture-of-Experts) の CPU 参照実装 (P-MOE-A2)。
//!
//! 出典: ルーティングの演算順序 (softmax -> top-k -> 再正規化 -> shared expert の加算) は
//! llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! Qwen3.8-Flash-Next 型 (多数の expert / top-k ルーティング / shared expert) を
//! **f32 の逐次演算だけ**で計算する参照実装。GPU カーネル側とビット一致させる
//! 基準になるので、加算順序を固定し、f64 を使わず、並列化もしない
//! (まずは単スレッド。並列化するなら加算順序を固定すること)。
//!
//! 演算順序 (この順序自体が数値契約):
//!
//! 1. `logits[e] = dot(gate_inp[e], x)`、`gate_inp_bias` があれば加算
//! 2. 全 expert について softmax (最大値減算つき。`crate::tensor::softmax` と同一)
//! 3. top-k 選択。同値のときは **expert index 昇順** (小さい index が勝つ)
//! 4. `norm_topk_prob` が true なら選択 k 個の確率の和で再正規化
//!    (和は選択 index 昇順に逐次加算)
//! 5. 選択 expert を **index 昇順**に処理し、
//!    `out += p_e * down_e( silu(gate_e(x)) * up_e(x) )`
//! 6. shared expert があれば最後に 1 回 `out += down_s( silu(gate_s(x)) * up_s(x) )`
//!
//! `silu` は `crate::tensor::silu_inplace` (= `v / (1 + exp(-v))`) をそのまま使う
//! (dense 経路の swiglu と同じ式・同じ exp)。
//!
//! 重みの読み出し・量子型の展開は呼び出し側の責務。ここは f32 スライスだけを見る。

use crate::tensor::{silu_inplace, softmax};

/// MoE FFN のパラメータ (すべて f32 スライス・行優先)。
///
/// 形状:
/// - `gate_inp`: `[n_expert][n_embd]`
/// - `gate_exps` / `up_exps`: `[n_expert][n_ffn][n_embd]`
/// - `down_exps`: `[n_expert][n_embd][n_ffn]`
/// - `shexp_gate` / `shexp_up`: `[n_shared_ffn][n_embd]`
/// - `shexp_down`: `[n_embd][n_shared_ffn]`
///
/// `n_shared_ffn` はスライス長から決まる (`shexp_gate.len() / n_embd`) ので
/// フィールドに持たない。shared expert は 3 本とも `Some` のときだけ使う。
pub struct MoeParams<'a> {
    pub n_embd: usize,
    pub n_ffn: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// true: top-k 選択後に和を 1 に再正規化。false: softmax 確率をそのまま使う。
    pub norm_topk_prob: bool,
    pub gate_inp: &'a [f32],
    pub gate_inp_bias: Option<&'a [f32]>,
    pub gate_exps: &'a [f32],
    pub up_exps: &'a [f32],
    pub down_exps: &'a [f32],
    pub shexp_gate: Option<&'a [f32]>,
    pub shexp_up: Option<&'a [f32]>,
    pub shexp_down: Option<&'a [f32]>,
}

/// 行ベクトル同士の内積 (f32 逐次加算)。
///
/// ビット一致の基準なので 4 分割アキュムレータや FMA 化をしない
/// (`#[inline(never)]` は最適化で演算列が変わらないようにするための目印)。
#[inline(never)]
fn dot_row(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += w[i] * x[i];
    }
    acc
}

/// 上位 `k` 個の expert index を **昇順** で返す。
///
/// 同値のタイブレーク: 走査を index 昇順に行い、`probs[i] > probs[best]` の
/// 厳密比較で更新するので、同値なら先に見つかった (= 小さい index) が残る。
/// NaN は比較が常に false なので選ばれない (候補が NaN だけのときのみ選ばれる)。
///
/// P-MOE-F1: 量子化重みの経路 (`model::qwen4exp`) も同じタイブレーク規則を
/// 使うので `pub(crate)` にしてある (規則を 2 箇所に書かないため)。
pub(crate) fn select_topk(probs: &[f32], k: usize) -> Vec<usize> {
    let n = probs.len();
    assert!(k >= 1 && k <= n, "k={} は 1..={} の範囲外", k, n);
    let mut used = vec![false; n];
    let mut chosen = Vec::with_capacity(k);
    for _ in 0..k {
        let mut best: Option<usize> = None;
        for i in 0..n {
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
        let b = best.expect("k <= n なので候補は必ず残る");
        used[b] = true;
        chosen.push(b);
    }
    chosen.sort_unstable();
    chosen
}

/// 1 つの expert の FFN 出力 `h = down( silu(gate(x)) * up(x) )` を書く。
///
/// `act` は長さ `n_ffn` の作業域 (呼び出し側が確保して使い回す)。
fn expert_ffn_into(
    x: &[f32],
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    n_embd: usize,
    n_ffn: usize,
    act: &mut [f32],
    h: &mut [f32],
) {
    for r in 0..n_ffn {
        act[r] = dot_row(&gate_w[r * n_embd..(r + 1) * n_embd], x);
    }
    // dense 経路 (llama/qwen35) と同じ: silu を in-place で当ててから up を掛ける。
    silu_inplace(&mut act[..n_ffn]);
    for r in 0..n_ffn {
        act[r] *= dot_row(&up_w[r * n_embd..(r + 1) * n_embd], x);
    }
    for j in 0..n_embd {
        h[j] = dot_row(&down_w[j * n_ffn..(j + 1) * n_ffn], &act[..n_ffn]);
    }
}

/// MoE FFN を計算して `out` に書く。
///
/// **`out` は本関数がゼロクリアしてから積む**(呼び出し側の事前ゼロ埋めは不要。
/// 既存値は捨てられる)。長さは `n_embd`。
///
/// 決定性: 同一入力・同一パラメータなら常にビット一致 (単スレッド・固定順序)。
pub fn moe_ffn(x: &[f32], p: &MoeParams, out: &mut [f32]) {
    let e = p.n_embd;
    let ff = p.n_ffn;
    let ne = p.n_expert;
    let k = p.n_expert_used;

    assert_eq!(x.len(), e, "x.len() は n_embd と一致する必要がある");
    assert_eq!(out.len(), e, "out.len() は n_embd と一致する必要がある");
    assert!(ne >= 1, "n_expert が 0");
    assert!(k >= 1 && k <= ne, "n_expert_used={} は 1..={} の範囲外", k, ne);
    assert_eq!(p.gate_inp.len(), ne * e, "gate_inp は [n_expert][n_embd]");
    assert_eq!(p.gate_exps.len(), ne * ff * e, "gate_exps は [n_expert][n_ffn][n_embd]");
    assert_eq!(p.up_exps.len(), ne * ff * e, "up_exps は [n_expert][n_ffn][n_embd]");
    assert_eq!(p.down_exps.len(), ne * e * ff, "down_exps は [n_expert][n_embd][n_ffn]");
    if let Some(b) = p.gate_inp_bias {
        assert_eq!(b.len(), ne, "gate_inp_bias は [n_expert]");
    }

    // 1. ルータ logits
    let mut probs = vec![0.0f32; ne];
    for i in 0..ne {
        let mut v = dot_row(&p.gate_inp[i * e..(i + 1) * e], x);
        if let Some(b) = p.gate_inp_bias {
            v += b[i];
        }
        probs[i] = v;
    }

    // 2. softmax (既存実装を流用: 最大値減算 → exp → 総和で除算)
    softmax(&mut probs);

    // 3. top-k (index 昇順で返る)
    let sel = select_topk(&probs, k);

    // 4. 再正規化 (選択 index 昇順に逐次加算した和で割る)
    let mut weights = vec![0.0f32; k];
    if p.norm_topk_prob {
        let mut sum = 0.0f32;
        for &i in &sel {
            sum += probs[i];
        }
        for (t, &i) in sel.iter().enumerate() {
            weights[t] = probs[i] / sum;
        }
    } else {
        for (t, &i) in sel.iter().enumerate() {
            weights[t] = probs[i];
        }
    }

    // 5. 選択 expert を index 昇順に積む
    out.fill(0.0);
    let mut act = vec![0.0f32; ff];
    let mut h = vec![0.0f32; e];
    for (t, &ei) in sel.iter().enumerate() {
        let base_g = ei * ff * e;
        let base_d = ei * e * ff;
        expert_ffn_into(
            x,
            &p.gate_exps[base_g..base_g + ff * e],
            &p.up_exps[base_g..base_g + ff * e],
            &p.down_exps[base_d..base_d + e * ff],
            e,
            ff,
            &mut act,
            &mut h,
        );
        let pe = weights[t];
        for j in 0..e {
            out[j] += pe * h[j];
        }
    }

    // 6. shared expert (係数 1.0 で最後に 1 回)
    if let (Some(sg), Some(su), Some(sd)) = (p.shexp_gate, p.shexp_up, p.shexp_down) {
        assert_eq!(sg.len() % e, 0, "shexp_gate は [n_shared_ffn][n_embd]");
        let sff = sg.len() / e;
        assert_eq!(su.len(), sff * e, "shexp_up は [n_shared_ffn][n_embd]");
        assert_eq!(sd.len(), e * sff, "shexp_down は [n_embd][n_shared_ffn]");
        let mut sact = vec![0.0f32; sff];
        for r in 0..sff {
            sact[r] = dot_row(&sg[r * e..(r + 1) * e], x);
        }
        silu_inplace(&mut sact);
        for r in 0..sff {
            sact[r] *= dot_row(&su[r * e..(r + 1) * e], x);
        }
        for j in 0..e {
            let v = dot_row(&sd[j * sff..(j + 1) * sff], &sact);
            out[j] += v;
        }
    }
}

#[cfg(test)]
mod tests {
    //! 期待値はすべてテスト内に数値で直書きする (実装を呼んで自分自身と比較しない)。
    //!
    //! 期待値の出所: 仕様の計算手順をそのまま f32 で書いた独立の C 参照実装
    //! (libm の expf を使う) の出力を `%.9g` で書き出したもの。丸めの入る箇所は
    //! 手計算でも突き合わせ済み (下の各テストのコメントに中間値 h[e] を書く)。
    //!
    //! 共通の合成テンソル (n_embd=8 / n_ffn=4 / n_expert=4):
    //! - `x = [0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 1.0]`
    //! - `gate_exps[e][r]` は列 `(r+e)%8` だけ 1.0 → pre_act[r] = x[(r+e)%8]
    //! - `up_exps[e][r]` は列 `(r+e+3)%8` だけ 1.0 → up[r] = x[(r+e+3)%8]
    //! - `down_exps[e][j][c] = ((e + 2j + 3c) % 7) * 0.25 - 0.75`
    //! - `gate_inp` は 1 列だけ非ゼロ → logits が手で読める

    use super::*;

    const E: usize = 8;
    const FF: usize = 4;
    const NE: usize = 4;

    const X: [f32; E] = [0.5, -1.0, 2.0, 0.25, -0.5, 1.5, -2.0, 1.0];

    /// expert 重み 3 種 (gate / up / down) を上の規約で作る。
    fn expert_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut gate = vec![0.0f32; NE * FF * E];
        let mut up = vec![0.0f32; NE * FF * E];
        let mut down = vec![0.0f32; NE * E * FF];
        for e in 0..NE {
            for r in 0..FF {
                gate[e * FF * E + r * E + (r + e) % E] = 1.0;
                up[e * FF * E + r * E + (r + e + 3) % E] = 1.0;
            }
            for j in 0..E {
                for c in 0..FF {
                    down[e * E * FF + j * FF + c] =
                        ((e + 2 * j + 3 * c) % 7) as f32 * 0.25 - 0.75;
                }
            }
        }
        (gate, up, down)
    }

    /// case A/B/D/F のルータ重み: logits = {1.0, 2.0, 0.5, -0.5}
    fn gate_inp_main() -> Vec<f32> {
        let mut g = vec![0.0f32; NE * E];
        g[0 * E + 7] = 1.0; // x[7] = 1.0
        g[1 * E + 2] = 1.0; // x[2] = 2.0
        g[2 * E + 0] = 1.0; // x[0] = 0.5
        g[3 * E + 1] = 0.5; // 0.5 * x[1] = -0.5
        g
    }

    /// case C のルータ重み: logits = {2.0, 1.0, 1.0, -1.0} (expert 1 と 2 が同値)
    fn gate_inp_tie() -> Vec<f32> {
        let mut g = vec![0.0f32; NE * E];
        g[0 * E + 7] = 2.0; // 2.0 * x[7] = 2.0
        g[1 * E + 7] = 1.0; // 1.0
        g[2 * E + 7] = 1.0; // 1.0 (同値)
        g[3 * E + 1] = 1.0; // x[1] = -1.0
        g
    }

    fn run(p: &MoeParams, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; x.len()];
        moe_ffn(x, p, &mut out);
        out
    }

    /// ビット一致を要求する比較 (差が出たら bit も出す)。
    fn expect_bits(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "長さが違う");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "out[{}]: got {} ({:#010x}) != want {} ({:#010x})",
                i,
                g,
                g.to_bits(),
                w,
                w.to_bits()
            );
        }
    }

    /// 全 expert の h (手計算の突き合わせ用の中間値)。
    #[allow(dead_code)]
    const H: [[f32; E]; NE] = [
        [
            1.99370992,
            -1.3436842,
            -0.292217374,
            1.48647761,
            -1.98707938,
            -0.700288892,
            0.843082368,
            1.99370992,
        ],
        [
            0.804178655,
            1.95768023,
            -1.18265462,
            -0.264476657,
            1.38092947,
            -2.08975339,
            -0.605903745,
            0.804178655,
        ],
        [
            -0.553461552,
            1.33129954,
            1.65108979,
            -1.24988997,
            0.142966792,
            0.954661608,
            -2.2766664,
            -0.553461552,
        ],
        [
            -0.175670028,
            -0.0716681406,
            0.611100316,
            -0.271102041,
            0.25011006,
            0.0237638727,
            -0.366534054,
            -0.175670028,
        ],
    ];

    /// 1. 基本形: k=2 / norm_topk_prob=true / shared 無し。
    ///
    /// logits = {1,2,0.5,-0.5} → softmax = {0.219879642, 0.597694814, 0.133363739,
    /// 0.0490617789} → top-2 = {0,1} → 再正規化 w = {0.268941432, 0.731058598}
    /// (= sigmoid(-1), sigmoid(1))。out = w0*h0 + w1*h1。
    #[test]
    fn basic_top2_with_renorm() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: true,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        };
        let out = run(&p, &X);
        expect_bits(
            &out,
            &[
                1.12409294,
                1.06980658,
                -0.94317919,
                0.206427485,
                0.475132346,
                -1.71606886,
                -0.216211364,
                1.12409294,
            ],
        );
        // 手計算の突き合わせ: out[0] = 0.268941432*1.99370992 + 0.731058598*0.804178655
        assert!((out[0] - (0.268941432f32 * H[0][0] + 0.731058598f32 * H[1][0])).abs() < 1e-6);
    }

    /// 2. norm_topk_prob=false では再正規化しない。
    ///
    /// 同じ選択 {0,1} だが重みが softmax 確率そのもの (0.219879642 / 0.597694814)
    /// なので、出力は 1 の (p0+p1)=0.817574 倍になる。
    #[test]
    fn norm_topk_prob_false_keeps_softmax_probs() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: false,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        };
        let out = run(&p, &X);
        expect_bits(
            &out,
            &[
                0.919029593,
                0.874646544,
                -0.771119237,
                0.168769822,
                0.388456047,
                -1.40301406,
                -0.176768869,
                0.919029593,
            ],
        );
        // 1 の出力 (再正規化あり) との比が選択確率の和になっている
        let a0 = 1.12409294f32;
        assert!((out[0] / a0 - 0.817574).abs() < 1e-5);
    }

    /// 3. top-k の同値タイブレークは expert index 昇順。
    ///
    /// logits = {2,1,1,-1} → expert 1 と 2 が同値。k=2 なら {0,1} を選ぶ
    /// (index 降順側に倒すと {0,2} になり、値は C_WRONG になる)。
    #[test]
    fn topk_tie_breaks_to_lower_index() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_tie();
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: true,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        };
        let out = run(&p, &X);
        let want = [
            1.67379582,
            -0.455810606,
            -0.531692863,
            1.0155735,
            -1.08128226,
            -1.07397342,
            0.453390002,
            1.67379582,
        ];
        expect_bits(&out, &want);
        // 反例 (index 降順 = expert 2 を選ぶ) とは一致しないこと
        let wrong: [f32; E] = [
            1.30867004,
            -0.62427032,
            0.230418414,
            0.750555038,
            -1.41422176,
            -0.255204171,
            0.00405269861,
            1.30867004,
        ];
        for (i, (g, w)) in out.iter().zip(wrong.iter()).enumerate() {
            assert_ne!(g.to_bits(), w.to_bits(), "out[{}] が反例と一致した", i);
        }
    }

    /// select_topk 単体のタイブレーク (同値は小さい index)。
    #[test]
    fn select_topk_prefers_lower_index_on_ties() {
        let probs = [0.1f32, 0.3, 0.3, 0.3];
        assert_eq!(select_topk(&probs, 1), vec![1]);
        assert_eq!(select_topk(&probs, 2), vec![1, 2]);
        assert_eq!(select_topk(&probs, 4), vec![0, 1, 2, 3]);
        // 戻り値は常に index 昇順
        let probs2 = [0.05f32, 0.4, 0.15, 0.4];
        assert_eq!(select_topk(&probs2, 3), vec![1, 2, 3]);
    }

    /// 4. k == n_expert なら全 expert を使い、dense 合成と一致する。
    ///
    /// f32 では softmax の総和が厳密に 1.0 にならないため、norm_topk_prob=true で
    /// k=n_expert にすると再正規化が ~1ulp の摂動を入れる。ここでは dense 合成
    /// (sum_e p_e * h_e) と厳密比較するために norm_topk_prob=false を使う。
    #[test]
    fn k_equals_n_expert_matches_dense_composition() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: NE,
            norm_topk_prob: false,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        };
        let out = run(&p, &X);
        expect_bits(
            &out,
            &[
                0.836599171,
                1.04867744,
                -0.520942032,
                -0.011220919,
                0.419793457,
                -1.27453101,
                -0.498376429,
                0.836599171,
            ],
        );
        // 手計算: out[0] = Σ_e p_e * h_e[0]
        let p_e = [0.219879642f32, 0.597694814, 0.133363739, 0.0490617789];
        let mut dense = 0.0f32;
        for e in 0..NE {
            dense += p_e[e] * H[e][0];
        }
        assert!((out[0] - dense).abs() < 1e-6, "out[0]={} dense={}", out[0], dense);
    }

    /// 5. shared expert の寄与が最後に 1 回だけ加算される。
    ///
    /// n_shared_ffn = 2 (スライス長から決まる)。shared の act は
    /// [silu(x[3]) * x[4], silu(0.5*x[5]) * x[6]] = [-0.0702720657, -1.01876807]。
    /// shared の down は j<4 → act0、j>=4 → act1。
    #[test]
    fn shared_expert_adds_contribution() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let (sg, su, sd) = shared_weights();
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: true,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: Some(&sg),
            shexp_up: Some(&su),
            shexp_down: Some(&sd),
        };
        let out = run(&p, &X);
        expect_bits(
            &out,
            &[
                1.05382085,
                0.999534488,
                -1.01345122,
                0.136155427,
                -0.543635726,
                -2.73483706,
                -1.23497939,
                0.105324864,
            ],
        );
    }

    /// 5b. shared の重みを全部ゼロにすると 1 (shared 無し) とビット一致する。
    #[test]
    fn shared_expert_with_zero_weights_is_identity() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let sg = vec![0.0f32; 2 * E];
        let su = vec![0.0f32; 2 * E];
        let sd = vec![0.0f32; E * 2];
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: true,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: Some(&sg),
            shexp_up: Some(&su),
            shexp_down: Some(&sd),
        };
        let out = run(&p, &X);
        let base = [
            1.12409294f32,
            1.06980658,
            -0.94317919,
            0.206427485,
            0.475132346,
            -1.71606886,
            -0.216211364,
            1.12409294,
        ];
        expect_bits(&out, &base);
    }

    /// 6. gate_inp_bias が logits に加算され、選択が変わる。
    ///
    /// bias = {0,0,3,0} → logits = {1,2,3.5,-0.5} → top-2 = {1,2} (index 昇順)。
    #[test]
    fn gate_inp_bias_shifts_selection() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let bias = [0.0f32, 0.0, 3.0, 0.0];
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: true,
            gate_inp: &ginp,
            gate_inp_bias: Some(&bias),
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        };
        let out = run(&p, &X);
        expect_bits(
            &out,
            &[
                -0.305793315,
                1.44556737,
                1.1341424,
                -1.07012546,
                0.368802756,
                0.399282575,
                -1.9718765,
                -0.305793315,
            ],
        );
    }

    /// 決定性: 同じ入力で 2 回呼ぶとビット一致。`out` は上書きなので
    /// 非ゼロで汚れたバッファから呼んでも同じ結果になる。
    #[test]
    fn deterministic_and_overwrites_out() {
        let (gate, up, down) = expert_weights();
        let ginp = gate_inp_main();
        let p = MoeParams {
            n_embd: E,
            n_ffn: FF,
            n_expert: NE,
            n_expert_used: 2,
            norm_topk_prob: true,
            gate_inp: &ginp,
            gate_inp_bias: None,
            gate_exps: &gate,
            up_exps: &up,
            down_exps: &down,
            shexp_gate: None,
            shexp_up: None,
            shexp_down: None,
        };
        let a = run(&p, &X);
        let b = run(&p, &X);
        expect_bits(&a, &b);
        let mut dirty = vec![123.456f32; E];
        moe_ffn(&X, &p, &mut dirty);
        expect_bits(&dirty, &a);
    }

    /// shared expert の重み (n_shared_ffn=2)。
    fn shared_weights() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let sff = 2usize;
        let mut sg = vec![0.0f32; sff * E];
        let mut su = vec![0.0f32; sff * E];
        sg[0 * E + 3] = 1.0; // x[3] = 0.25
        sg[1 * E + 5] = 0.5; // 0.5 * x[5] = 0.75
        su[0 * E + 4] = 1.0; // x[4] = -0.5
        su[1 * E + 6] = 1.0; // x[6] = -2.0
        let mut sd = vec![0.0f32; E * sff];
        for j in 0..E {
            sd[j * sff + 0] = if j < 4 { 1.0 } else { 0.0 };
            sd[j * sff + 1] = if j < 4 { 0.0 } else { 1.0 };
        }
        (sg, su, sd)
    }
}
