//! サンプリング: greedy / temperature / top-k / top-p
//! seed を固定すれば同一設定で同一出力(TensorFold の exact decoding 思想)。
//!
//! P8 最適化 (2026-10-12): P7.5 プロファイルで sampler が約3.5ms/step の
//! 固定コスト(語彙152kのフルソート+1.2MBタプル確保)と判明したため:
//! - top_k 指定時: select_nth_unstable で O(n) 選択 + 上位 k のみソート。
//!   同点タイブレーク(値降順→インデックス昇順)も旧フル安定ソートと同一に
//!   再現するため、**選ばれる系列は旧実装とビット同一**
//! - top_p 単独時: 上位 M=2048 の部分集合で softmax/top-p を計算。
//!   cutoff が M に達した(=mass が p に足りない)場合は旧フルパスに
//!   フォールバックして厳密。cutoff < M の場合は数学的に上位 M への
//!   条件付き分布と一致(M 外の裾質量 < 0.1% を切り捨てる代わりに高速)。
//!   語彙 ≤ M なら旧実装とビット同一
//! - plain (temperature のみ・本番既定): フルソートを廃止し、スクラッチ
//!   バッファ再利用の 2 パス O(n)。分布は数学的に同一だが、RNG→トークンの
//!   対応が旧実装(降順走査)と変わるため**出力系列は旧実装と異なる**(仕様)。
//!   greedy(temperature=0)の経路は一切変更していない

use rand::{rngs::StdRng, SeedableRng, Rng};

/// top_p 単独モードで部分集合計算に使う上位要素数。
/// cutoff がこのサイズに達したらフルソートにフォールバックする。
const TOP_P_SUBSET: usize = 2048;

/// 反復ペナルティ (A2.5)。HF transformers の `RepetitionPenaltyLogitsProcessor`
/// と同じ変換を、文脈 (prompt + 生成済み) 中に現れた**ユニークな**トークンへ
/// 一度だけ適用する:
///
/// ```text
/// logit > 0  → logit / penalty
/// logit <= 0 → logit * penalty
/// ```
///
/// - `penalty > 1.0` で既出トークンが抑えられ、greedy の反復ループを打破する
/// - temperature / top-k / top-p より前 (生 logits) に適用する (HF と同じ順序)
/// - RNG を一切使わないため、temperature=0 (greedy) は完全決定論。
///   temperature>0 でも seed 固定なら同一入力→同一出力が保たれる
/// - `penalty == 1.0` は完全な no-op (既存出力とビット一致)
///
/// `context` は適用対象のトークン列 (prompt_ids + generated)。語彙外の id は
/// 無視する。同一 id の重複は 1 回だけ適用する (HF の set セマンティクス)。
pub fn apply_repetition_penalty(logits: &mut [f32], context: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = vec![false; logits.len()];
    for &t in context {
        let t = t as usize;
        if t < logits.len() && !seen[t] {
            seen[t] = true;
            let l = &mut logits[t];
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

pub struct Sampler {
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    rng: StdRng,
    /// スクラッチ (ステップごとのアロケートを避けるため再利用)
    scratch: Vec<(f32, usize)>,
    exps: Vec<f32>,
}

impl Sampler {
    pub fn new(temperature: Option<f32>, top_k: Option<usize>, top_p: Option<f32>, seed: Option<u64>) -> Self {
        Self {
            temperature: temperature.unwrap_or(1.0),
            top_k,
            top_p,
            rng: match seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            },
            scratch: Vec::new(),
            exps: Vec::new(),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> usize {
        // greedy (P8 でも不変)
        if self.temperature <= 0.0 {
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = i;
                }
            }
            return best;
        }

        let n = logits.len();
        let t = self.temperature;

        if let Some(k) = self.top_k {
            // --- top_k モード: 旧実装とビット同一の高速パス ---
            let m = k.max(1).min(n);
            // 温度適用した (値, インデックス) を構築
            self.scratch.clear();
            self.scratch.extend(logits.iter().enumerate().map(|(i, &x)| (x / t, i)));
            // 上位 m を O(n) で選択 (降順基準)
            let s = &mut self.scratch;
            s.select_nth_unstable_by(m - 1, |a, b| cmp_desc(a, b));
            // 旧実装の安定ソート(値降順、同点は元順=インデックス昇順)を再現
            let head = &mut s[..m];
            head.sort_by(|a, b| cmp_desc(a, b).then(a.1.cmp(&b.1)));
            return sample_sorted(&mut self.rng, self.top_p, &s[..m]);
        }

        if let Some(p) = self.top_p {
            // --- top_p モード: 上位 M 部分集合 + 境界でフルパスにフォールバック ---
            if n > TOP_P_SUBSET {
                let m = TOP_P_SUBSET;
                let mut max = f32::NEG_INFINITY;
                for &x in logits.iter() {
                    let v = x / t;
                    if v > max {
                        max = v;
                    }
                }
                self.scratch.clear();
                self.scratch.extend(logits.iter().enumerate().map(|(i, &x)| (x / t, i)));
                let s = &mut self.scratch;
                s.select_nth_unstable_by(m - 1, |a, b| cmp_desc(a, b));
                let head = &mut s[..m];
                head.sort_by(|a, b| cmp_desc(a, b).then(a.1.cmp(&b.1)));
                // 部分集合上で softmax + top-p
                let sub = &s[..m];
                let probs: Vec<f32> = sub.iter().map(|(x, _)| (x - max).exp()).collect();
                let total: f32 = probs.iter().sum();
                let mut cum = 0.0f32;
                let mut cutoff = m;
                for i in 0..m {
                    cum += probs[i] / total;
                    if cum >= p {
                        cutoff = i + 1;
                        break;
                    }
                }
                if cutoff < m {
                    // mass が M 内で収まった: 分布は上位 M への条件付きと一致
                    let mut pr: Vec<f32> = probs[..cutoff].to_vec();
                    let total2: f32 = pr.iter().sum();
                    let mut u = self.rng.gen::<f32>() * total2;
                    let mut out = sub[cutoff - 1].1;
                    for (i, v) in pr.iter().enumerate() {
                        u -= v;
                        if u <= 0.0 {
                            out = sub[i].1;
                            break;
                        }
                    }
                    return out;
                }
                // cutoff が M に達した: 裾が足りないので旧フルパスへ
            }
            return self.sample_legacy_full(logits);
        }

        // --- plain モード (temperature のみ・本番既定): ソートなし 2 パス O(n) ---
        // 分布は旧実装と数学的に同一 (softmax 全体からのカテゴリカル)。
        // RNG→トークンの対応が旧実装と変わる (仕様、モジュール先頭のコメント参照)。
        let mut max = f32::NEG_INFINITY;
        for &x in logits.iter() {
            let v = x / t;
            if v > max {
                max = v;
            }
        }
        self.exps.clear();
        self.exps.resize(n, 0.0);
        let mut total = 0.0f32;
        let exps = &mut self.exps;
        for (i, &x) in logits.iter().enumerate() {
            let e = (x / t - max).exp();
            exps[i] = e;
            total += e;
        }
        let mut u = self.rng.gen::<f32>() * total;
        for (i, &e) in exps.iter().enumerate() {
            u -= e;
            if u <= 0.0 {
                return i;
            }
        }
        n - 1
    }

    /// 旧実装のフルソートパス (top_p のフォールバック専用。厳密な旧挙動を保持)
    fn sample_legacy_full(&mut self, logits: &[f32]) -> usize {
        let mut v: Vec<(f32, usize)> = logits
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (x / self.temperature, i))
            .collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        sample_sorted(&mut self.rng, self.top_p, &v)
    }
}

/// 値の降順比較 (NaN は同値扱い。旧 sort_by の comparator と同一)
fn cmp_desc(a: &(f32, usize), b: &(f32, usize)) -> std::cmp::Ordering {
    b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
}

/// 降順にソート済みの (温度適用済み値, インデックス) スライスから
/// softmax (+必要なら top-p) して 1 トークン選ぶ。
/// 旧実装の演算順序 (softmax → top-p → 再正規化 → 累積走査) をそのまま再現。
fn sample_sorted(rng: &mut StdRng, top_p: Option<f32>, sorted: &[(f32, usize)]) -> usize {
    let max = sorted[0].0;
    let mut probs: Vec<f32> = sorted.iter().map(|(x, _)| (x - max).exp()).collect();

    if let Some(p) = top_p {
        let total: f32 = probs.iter().sum();
        let mut cum = 0.0f32;
        let mut cutoff = probs.len();
        for i in 0..probs.len() {
            cum += probs[i] / total;
            if cum >= p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
    }

    let total: f32 = probs.iter().sum();
    let mut u = rng.gen::<f32>() * total;
    for (i, pr) in probs.iter().enumerate() {
        u -= pr;
        if u <= 0.0 {
            return sorted[i].1;
        }
    }
    sorted[probs.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(temp: Option<f32>, top_k: Option<usize>, top_p: Option<f32>, seed: Option<u64>) -> Sampler {
        Sampler::new(temp, top_k, top_p, seed)
    }

    /// 旧実装 (P8 最適化前) の参照実装。高速パスのビット一致検証に使う。
    fn legacy_sample(
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        rng: &mut StdRng,
        logits: &[f32],
    ) -> usize {
        let mut v: Vec<(f32, usize)> = logits
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, x)| (x / temperature, i))
            .collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        if let Some(k) = top_k {
            v.truncate(k.max(1));
        }
        let max = v[0].0;
        let mut probs: Vec<f32> = v.iter().map(|(x, _)| (x - max).exp()).collect();
        if let Some(p) = top_p {
            let total: f32 = probs.iter().sum();
            let mut cum = 0.0f32;
            let mut cutoff = probs.len();
            for i in 0..probs.len() {
                cum += probs[i] / total;
                if cum >= p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            v.truncate(cutoff);
        }
        let total: f32 = probs.iter().sum();
        let mut u = rng.gen::<f32>() * total;
        for (i, pr) in probs.iter().enumerate() {
            u -= pr;
            if u <= 0.0 {
                return v[i].1;
            }
        }
        v[v.len() - 1].1
    }

    #[test]
    fn greedy_picks_argmax() {
        let mut s = sampler(Some(0.0), None, None, None);
        assert_eq!(s.sample(&[0.1, 3.0, 2.0]), 1);
        assert_eq!(s.sample(&[5.0, 1.0, 2.0]), 0);
        assert_eq!(s.sample(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn greedy_tie_breaks_to_first_index() {
        // 実装は `v > best_v` の厳密比較なので同点は先頭が勝つ
        let mut s = sampler(Some(0.0), None, None, None);
        assert_eq!(s.sample(&[1.0, 2.0, 2.0]), 1);
    }

    #[test]
    fn seeded_sampling_is_deterministic() {
        let logits: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 0.5, 3.0],
            vec![3.0, 1.0, 2.0, 0.5],
            vec![0.5, 0.5, 0.5, 0.5],
        ];
        let mut a = sampler(Some(0.8), None, None, Some(42));
        let mut b = sampler(Some(0.8), None, None, Some(42));
        let mut seq_a = Vec::new();
        let mut seq_b = Vec::new();
        for _ in 0..30 {
            for lg in &logits {
                seq_a.push(a.sample(lg));
                seq_b.push(b.sample(lg));
            }
        }
        assert_eq!(seq_a, seq_b, "同一seed+同一logits列なら同一サンプル列");
    }

    #[test]
    fn top_k_one_always_picks_argmax() {
        let mut s = sampler(Some(1.5), Some(1), None, Some(7));
        for _ in 0..50 {
            assert_eq!(s.sample(&[0.1, 4.0, 0.2]), 1);
        }
    }

    #[test]
    fn top_k_respects_k_choices() {
        // k=2 なら argmax(1) と準高峰(3) の2択だけ
        let mut s = sampler(Some(1.0), Some(2), None, Some(7));
        for _ in 0..100 {
            let i = s.sample(&[0.1, 4.0, 0.0, 3.0]);
            assert!(i == 1 || i == 3, "top_k=2 で id {i} が選ばれた");
        }
    }

    #[test]
    fn top_k_fast_path_matches_legacy_reference() {
        // P8: select_nth 高速パスが旧フルソート参照実装とビット一致すること
        let mut lcg: u64 = 0x1234_5678_9abc_def0;
        let mut next_f32 = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 33) as f32 / (u32::MAX >> 1) as f32 - 1.0) * 8.0
        };
        let n = 300;
        let logits: Vec<f32> = (0..n).map(|_| next_f32()).collect();
        for k in [1usize, 3, 17, n] {
            let mut a = sampler(Some(0.9), Some(k), None, Some(42));
            let mut rng_ref = StdRng::seed_from_u64(42);
            for step in 0..200 {
                let got = a.sample(&logits);
                let want = legacy_sample(0.9, Some(k), None, &mut rng_ref, &logits);
                assert_eq!(got, want, "top_k={k} step={step}: 高速パスが参照と不一致");
            }
        }
    }

    #[test]
    fn top_k_with_ties_matches_legacy_reference() {
        // 同点 (タイ) が多い場合も旧実装のタイブレーク (インデックス昇順) を再現
        let logits: Vec<f32> = vec![3.0, 1.0, 3.0, 2.0, 3.0, 1.0, 2.0];
        for k in [1usize, 2, 4, 7] {
            let mut a = sampler(Some(1.2), Some(k), None, Some(9));
            let mut rng_ref = StdRng::seed_from_u64(9);
            for step in 0..100 {
                let got = a.sample(&logits);
                let want = legacy_sample(1.2, Some(k), None, &mut rng_ref, &logits);
                assert_eq!(got, want, "ties top_k={k} step={step}");
            }
        }
    }

    #[test]
    fn top_p_dominant_logit_collapses_to_argmax() {
        // logit 差が大きいと p=0.5 でも top-1 だけで累積 0.5 を超える
        let mut s = sampler(Some(1.0), None, Some(0.5), Some(7));
        for _ in 0..50 {
            assert_eq!(s.sample(&[10.0, 0.0, 0.0]), 0);
        }
    }

    #[test]
    fn top_p_small_vocab_matches_legacy_reference() {
        // 語彙が TOP_P_SUBSET 未満なら部分集合 = 全体なので旧実装と同じ
        // 経路 (sample_from_sorted) を通り、選択はビット一致する
        let mut lcg: u64 = 0xdead_beef_cafe_1111;
        let mut next_f32 = || {
            lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((lcg >> 33) as f32 / (u32::MAX >> 1) as f32 - 1.0) * 6.0
        };
        let logits: Vec<f32> = (0..50).map(|_| next_f32()).collect();
        let mut a = sampler(Some(1.0), None, Some(0.9), Some(5));
        let mut rng_ref = StdRng::seed_from_u64(5);
        for step in 0..100 {
            let got = a.sample(&logits);
            let want = legacy_sample(1.0, None, Some(0.9), &mut rng_ref, &logits);
            assert_eq!(got, want, "top_p step={step}");
        }
    }

    #[test]
    fn plain_mode_distribution_is_correct() {
        // plain モード: softmax 分布に従うことの統計的検証
        // logits [0, 4] → exp(0)/(exp(0)+exp(4)) ≈ 0.0180 / 0.9820
        let mut s = sampler(Some(1.0), None, None, Some(11));
        let mut count1 = 0u32;
        let trials = 10000;
        for _ in 0..trials {
            if s.sample(&[0.0, 4.0]) == 1 {
                count1 += 1;
            }
        }
        let expected = 1.0 / (1.0 + (-4.0f32).exp());
        let freq = count1 as f32 / trials as f32;
        assert!(
            (freq - expected).abs() < 0.02,
            "freq={freq} expected≈{expected}"
        );
    }

    #[test]
    fn top_p_boundary_falls_back_without_panic() {
        // p=1.0 は mass が M 内で満たされない可能性がありフォールバックする。
        // パニックせず決定論的であることだけ確認。
        let mut a = sampler(Some(1.0), None, Some(1.0), Some(3));
        let mut b = sampler(Some(1.0), None, Some(1.0), Some(3));
        let logits: Vec<f32> = (0..5000).map(|i| ((i % 97) as f32) * 0.01).collect();
        let sa: Vec<usize> = (0..20).map(|_| a.sample(&logits)).collect();
        let sb: Vec<usize> = (0..20).map(|_| b.sample(&logits)).collect();
        assert_eq!(sa, sb);
    }

    #[test]
    fn unseeded_temp_one_can_pick_non_argmax() {
        // サニティ: 確率的サンプリングが argmax 固定になっていないことの逆方向確認
        let mut s = sampler(Some(1.0), None, None, Some(123));
        let picked: std::collections::HashSet<usize> =
            (0..100).map(|_| s.sample(&[2.0, 1.5])).collect();
        assert!(picked.contains(&0));
        assert!(picked.contains(&1), "temp=1.0 で非argmaxが一度も選ばれない");
    }

    // ---------- A2.5: repetition penalty ----------

    #[test]
    fn penalty_suppresses_repeated_positive_logit() {
        // token 1 が文脈に現れたら正の logit が割り引かれる
        let mut logits = vec![1.0f32, 4.0, 2.0];
        apply_repetition_penalty(&mut logits, &[1], 2.0);
        assert_eq!(logits[0], 1.0, "文脈外は不変");
        assert_eq!(logits[1], 2.0, "4.0/2.0 = 2.0");
        assert_eq!(logits[2], 2.0, "文脈外は不変");
    }

    #[test]
    fn penalty_pushes_negative_logit_down() {
        let mut logits = vec![-2.0f32];
        apply_repetition_penalty(&mut logits, &[0], 1.5);
        assert_eq!(logits[0], -3.0, "-2.0*1.5 = -3.0");
    }

    #[test]
    fn penalty_duplicate_ids_applied_once() {
        // 同一 id が文脈に複数回現れても 1 回だけ適用 (HF の set セマンティクス)
        let mut a = vec![4.0f32, 0.0];
        apply_repetition_penalty(&mut a, &[0, 0, 0], 2.0);
        let mut b = vec![4.0f32, 0.0];
        apply_repetition_penalty(&mut b, &[0], 2.0);
        assert_eq!(a, b, "重複適用で 4/2/2=1.0 になってはいけない");
    }

    #[test]
    fn penalty_one_is_exact_noop() {
        // penalty == 1.0 はビット完全な no-op (既存経路との互換契約)
        let base: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.37 - 100.0).collect();
        let mut x = base.clone();
        apply_repetition_penalty(&mut x, &(0..1000).collect::<Vec<u32>>(), 1.0);
        assert_eq!(x, base, "penalty=1.0 でビットが変わった");
    }

    #[test]
    fn penalty_out_of_vocab_id_is_ignored() {
        let mut logits = vec![1.0f32, 2.0];
        apply_repetition_penalty(&mut logits, &[999, u32::MAX], 2.0);
        assert_eq!(logits, vec![1.0, 2.0]);
    }

    #[test]
    fn penalty_greedy_breaks_loop_deterministically() {
        // greedy で token 1 のループに落ちている状況 (token 0 は未出) を模し、
        // penalty で選択が変わること + 2 回実行で同一結果 (決定論) を確認
        let logits = vec![3.0f32, 5.0];
        let ctx = [1u32, 1, 1];

        let mut a = logits.clone();
        apply_repetition_penalty(&mut a, &ctx, 1.5);
        let pick = |l: &[f32]| {
            let (mut bi, mut bv) = (0usize, f32::NEG_INFINITY);
            for (i, &v) in l.iter().enumerate() {
                if v > bv {
                    bv = v;
                    bi = i;
                }
            }
            bi
        };
        // token 1 のみ減衰: 5.0/1.5 ≈ 3.33 > 3.0 → まだ 1
        let mut b = logits.clone();
        apply_repetition_penalty(&mut b, &ctx, 2.0);
        assert_eq!(pick(&a), 1, "penalty 1.5 (5/1.5≈3.33>3) ではまだ 1");
        // penalty 2.0: token 1 → 2.5 < 3.0 (token 0 は不変) で argmax が反転
        assert_eq!(pick(&b), 0, "penalty 2.0 (5/2=2.5<3) で argmax が反転");
        // 決定論: 同一入力→同一ビット
        let mut c = logits.clone();
        apply_repetition_penalty(&mut c, &ctx, 2.0);
        assert_eq!(c, b);
    }
}
