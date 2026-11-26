//! Sampler の境界・決定論テスト (task 1434)。
//!
//! `src/sampler.rs` の単体テストを補完し、**公開 API のみ**で次を固定する:
//! - temperature<=0 (greedy) の完全決定論とタイブレーク (先頭添字)
//! - seed 固定時の再現性 (plain / top-k / top-p) と seed が効いていること
//! - NaN / -inf の logit の扱い (パニックせず、決定論的で、範囲内)
//! - top-k / top-p の境界 (k=0, k>語彙, p<=0, p>=1)
//! - min-p (spec_ngram::offer_draft の閾値) の境界
//! - repetition_penalty の単調性 (正/負/ゼロ logit・penalty<1・argmax 反転)
//!
//! CPU-only・決定的・乱数は自前の LCG のみ (プロセス外状態に依存しない)。
//! 語彙長 0 は Sampler の契約外 (実際の語彙長は常に 1 以上) なので扱わない。

use denno_engine::sampler::{
    apply_repetition_penalty, argmax_first, Sampler,
};
use denno_engine::spec_ngram::{offer_draft, NgramIndex};

/// 再現性テスト用の決定的な logits 生成器 (線形合同法)。
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
        // [-6, 6) 程度
        ((self.0 >> 33) as f32 / (u32::MAX >> 1) as f32 - 1.0) * 6.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn sampler(temp: Option<f32>, top_k: Option<usize>, top_p: Option<f32>, seed: Option<u64>) -> Sampler {
    Sampler::new(temp, top_k, top_p, seed)
}

/// 素直な比較鎖の argmax (テストの参照実装。同点は先頭・NaN 無視)。
fn naive_argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best
}

/// 系列を n 回サンプルする。
fn seq(s: &mut Sampler, logits: &[f32], n: usize) -> Vec<usize> {
    (0..n).map(|_| s.sample(logits)).collect()
}

// ------------------------------------------------------------ greedy (temp=0)

#[test]
fn greedy_is_deterministic_regardless_of_seed() {
    let logits = Lcg::new(0x1434).vec(512);
    let want = naive_argmax(&logits);

    for seed in [None, Some(0), Some(7), Some(u64::MAX)] {
        let mut s = sampler(Some(0.0), None, None, seed);
        for step in 0..32 {
            assert_eq!(s.sample(&logits), want, "seed={seed:?} step={step} で argmax と不一致");
        }
    }
}

#[test]
fn greedy_ties_break_to_lowest_index() {
    // 同点が 3 箇所・最大値が複数
    let cases: Vec<(Vec<f32>, usize)> = vec![
        (vec![1.0, 2.0, 2.0], 1),
        (vec![2.0, 2.0, 2.0], 0),
        (vec![0.0, -0.0, 3.5, 3.5, 3.5], 2),
        (vec![f32::MIN, 0.0, 0.0, f32::MIN], 1),
    ];
    for (logits, want) in cases {
        let mut s = sampler(Some(0.0), None, None, None);
        for _ in 0..8 {
            assert_eq!(s.sample(&logits), want, "logits={logits:?}");
        }
        assert_eq!(argmax_first(&logits), want, "argmax_first も同じタイブレーク");
    }
}

#[test]
fn non_positive_temperature_is_greedy() {
    let logits = Lcg::new(0xbeef).vec(64);
    let want = naive_argmax(&logits);
    for temp in [0.0f32, -0.0, -1.0, -1e-30] {
        let mut s = sampler(Some(temp), None, None, Some(9));
        assert_eq!(s.sample(&logits), want, "temp={temp} が greedy になっていない");
    }
}

#[test]
fn greedy_ignores_nan() {
    // NaN は比較が常に偽なので無視され、有限値の最大が選ばれる
    let mut s = sampler(Some(0.0), None, None, None);
    assert_eq!(s.sample(&[f32::NAN, 1.0, 2.0, f32::NAN]), 2);
    assert_eq!(s.sample(&[3.0, f32::NAN, f32::NAN]), 0);
    // 全て NaN: 有効な最大が無いので先頭に倒れる (パニックしない)
    assert_eq!(s.sample(&[f32::NAN, f32::NAN]), 0);
    assert_eq!(argmax_first(&[f32::NAN, f32::NAN]), 0);
}

#[test]
fn greedy_all_negative_infinity_falls_back_to_first() {
    let mut s = sampler(Some(0.0), None, None, None);
    assert_eq!(s.sample(&[f32::NEG_INFINITY; 5]), 0);
    // -inf と有限値が混在: 有限値が勝つ (-inf は最小でも最大でもない)
    assert_eq!(s.sample(&[f32::NEG_INFINITY, -1000.0, f32::NEG_INFINITY]), 1);
}

// ------------------------------------------------------------ seed 再現性

#[test]
fn seeded_plain_sampling_is_reproducible_and_seed_sensitive() {
    let mut gen = Lcg::new(0x5157);
    let steps: Vec<Vec<f32>> = (0..20).map(|_| gen.vec(300)).collect();

    let run = |seed: u64| {
        let mut s = sampler(Some(1.1), None, None, Some(seed));
        steps.iter().map(|l| s.sample(l)).collect::<Vec<usize>>()
    };
    let a = run(42);
    let b = run(42);
    assert_eq!(a, b, "同一 seed + 同一 logits 列で系列が割れた");

    // 20 ステップすべてが一致する確率は 1/300^19 相当なので、seed が効いて
    // いなければ実質確実に落ちる。固定 seed 同士の比較なので flaky ではない。
    let c = run(43);
    assert_ne!(a, c, "seed を変えても系列が同じ = seed がサンプリングに効いていない");
}

#[test]
fn seeded_top_k_sampling_is_reproducible() {
    let mut gen = Lcg::new(0x6b6b);
    let logits = gen.vec(500);
    let run = |seed: u64| {
        let mut s = sampler(Some(0.8), Some(16), None, Some(seed));
        seq(&mut s, &logits, 50)
    };
    assert_eq!(run(11), run(11), "top-k 経路が seed 固定で再現しない");
    assert_ne!(run(11), run(12), "top-k 経路で seed が効いていない");
}

#[test]
fn seeded_top_p_sampling_is_reproducible() {
    let mut gen = Lcg::new(0x7c7c);
    let logits = gen.vec(50); // 語彙 <= TOP_P_SUBSET (2048) の厳密経路
    let run = |seed: u64| {
        let mut s = sampler(Some(0.9), None, Some(0.95), Some(seed));
        seq(&mut s, &logits, 50)
    };
    assert_eq!(run(5), run(5), "top-p 経路が seed 固定で再現しない");
    assert_ne!(run(5), run(6), "top-p 経路で seed が効いていない");
}

// ------------------------------------------------------------ NaN / -inf

#[test]
fn plain_mode_nan_is_deterministic_and_in_range() {
    // plain (temperature のみ) では NaN が total に伝播しないようにしてある
    // (task 942 で修正: NaN は確率 0 = 選ばれない)。パニックせず決定論的で、
    // 系列は通常どおり seed に依存する。
    let logits = vec![0.5f32, f32::NAN, 1.5, f32::NEG_INFINITY, 1.0];
    let run = |seed: u64| {
        let mut s = sampler(Some(1.0), None, None, Some(seed));
        seq(&mut s, &logits, 16)
    };
    let a = run(3);
    let b = run(3);
    assert_eq!(a, b, "NaN 入り logits で系列が再現しない");
    for &i in &a {
        assert!(i < logits.len(), "範囲外の添字 {i}");
    }
    assert!(!a.iter().any(|&i| logits[i].is_nan()), "NaN の語彙が選ばれた");
    // NaN は候補から外れるので、RNG は通常どおり消費される (seed が効く)。
    // 16 ステップすべて一致する確率は 4^-16 相当なので flaky ではない。
    assert_ne!(a, run(4), "NaN の扱いで RNG 消費が止まっている (旧挙動)");
}

#[test]
fn negative_infinity_is_never_sampled_in_plain_mode() {
    // 1 個だけ -inf を混ぜ、他は十分大きな有限値。200 回中 -inf 添字は 0 回。
    let mut logits = vec![0.0f32; 8];
    logits[3] = f32::NEG_INFINITY;
    for i in [0usize, 1, 2, 4, 5, 6, 7] {
        logits[i] = 2.0;
    }
    let mut s = sampler(Some(1.0), None, None, Some(21));
    let picks = seq(&mut s, &logits, 200);
    assert!(!picks.contains(&3), "-inf のトークンが選ばれた: {picks:?}");
    assert!(picks.iter().any(|&i| i != 3), "有限値が一度も選ばれない");
}

#[test]
fn negative_infinity_is_never_sampled_in_top_k_and_top_p() {
    let mut logits = vec![1.0f32; 10];
    logits[7] = f32::NEG_INFINITY;
    logits[8] = f32::NEG_INFINITY;

    let mut s = sampler(Some(0.7), Some(4), None, Some(22));
    let picks = seq(&mut s, &logits, 200);
    assert!(!picks.iter().any(|&i| i == 7 || i == 8), "top-k で -inf が選ばれた");
    assert!(picks.iter().all(|&i| i < logits.len()), "top-k で範囲外の添字");

    let mut s = sampler(Some(0.7), None, Some(0.8), Some(23));
    let picks = seq(&mut s, &logits, 200);
    assert!(!picks.iter().any(|&i| i == 7 || i == 8), "top-p で -inf が選ばれた");
    assert!(picks.iter().all(|&i| i < logits.len()), "top-p で範囲外の添字");
}

#[test]
fn all_negative_infinity_top_p_does_not_panic() {
    // 全 -inf: softmax が NaN になるが、パニックせず範囲内の添字を返す。
    let logits = vec![f32::NEG_INFINITY; 6];
    let mut a = sampler(Some(1.0), None, Some(0.9), Some(1));
    let mut b = sampler(Some(1.0), None, Some(0.9), Some(1));
    let sa = seq(&mut a, &logits, 8);
    let sb = seq(&mut b, &logits, 8);
    assert_eq!(sa, sb, "全 -inf で決定論的でない");
    assert!(sa.iter().all(|&i| i < logits.len()), "全 -inf で範囲外の添字");
}

// ------------------------------------------------------------ top-k 境界

#[test]
fn top_k_zero_behaves_like_one() {
    // k=0 は `k.max(1)` で 1 に正規化される = argmax 固定。
    let logits = vec![0.2f32, 5.0, 1.0, 4.9];
    let mut s = sampler(Some(1.3), Some(0), None, Some(31));
    for _ in 0..64 {
        assert_eq!(s.sample(&logits), 1, "k=0 が argmax 固定になっていない");
    }
}

#[test]
fn top_k_larger_than_vocab_covers_whole_vocab() {
    // k > n は n に丸められ、全語彙が候補になる (各添字が少なくとも 1 回出る)。
    let logits = vec![0.0f32, 0.0, 0.0, 0.0];
    let mut s = sampler(Some(1.0), Some(1000), None, Some(32));
    let mut seen = [false; 4];
    for _ in 0..400 {
        seen[s.sample(&logits)] = true;
    }
    assert!(seen.iter().all(|&b| b), "k>n で全語彙が候補になっていない: {seen:?}");
}

#[test]
fn top_k_candidates_are_the_k_largest() {
    // 上位 k 以外が一度も選ばれないこと (k=3, 値の重複なし)。
    let logits = vec![0.0f32, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0];
    let allowed = [1usize, 3, 5];
    let mut s = sampler(Some(1.0), Some(3), None, Some(33));
    for _ in 0..300 {
        let i = s.sample(&logits);
        assert!(allowed.contains(&i), "上位 3 以外の {i} が選ばれた");
    }
}

// ------------------------------------------------------------ top-p 境界

#[test]
fn top_p_zero_and_negative_collapse_to_argmax() {
    let logits = vec![0.5f32, 4.0, 1.0, 3.9];
    for p in [0.0f32, -0.5, -0.0] {
        let mut s = sampler(Some(1.0), None, Some(p), Some(34));
        for _ in 0..32 {
            assert_eq!(s.sample(&logits), 1, "p={p} が argmax に潰れていない");
        }
    }
}

#[test]
fn top_p_one_keeps_full_support() {
    // p=1.0 は全質量を残す = 両トークンが観測される。
    let logits = vec![0.0f32, 0.0];
    for p in [1.0f32, 1.5] {
        let mut s = sampler(Some(1.0), None, Some(p), Some(35));
        let picks = seq(&mut s, &logits, 100);
        assert!(picks.contains(&0) && picks.contains(&1), "p={p} で分布が潰れた: {picks:?}");
    }
}

#[test]
fn top_p_single_dominant_token_excludes_others() {
    // 累積質量が最初の要素で p を超えるなら argmax 固定。
    let mut s = sampler(Some(1.0), None, Some(0.9), Some(36));
    for _ in 0..64 {
        assert_eq!(s.sample(&[20.0, 0.0, -20.0]), 0);
    }
}

// ------------------------------------------------------------ min-p 境界

/// draft が [3, 4] を返す辞書を作る (キー [1] の継続)。
fn minp_fixture() -> NgramIndex {
    let mut idx = NgramIndex::new(3, 100, 8, 4);
    // 位置 1 の手前 [1] の継続は [3, 4]。feed は i ごとに suffix を登録する。
    idx.feed(&[1, 3, 4]);
    idx
}

#[test]
fn min_p_fixture_returns_expected_draft() {
    let idx = minp_fixture();
    let draft = offer_draft(&idx, &[], 1, 4, &[0.0; 5], 0.0);
    assert_eq!(draft, vec![3, 4], "前提: 辞書の draft が想定と違う");
}

#[test]
fn min_p_non_positive_is_disabled() {
    let idx = minp_fixture();
    // token 3 の確率が極小でも、min_p <= 0 なら draft を素通しする。
    let logits = vec![0.0f32, 0.0, 0.0, -50.0, 0.0];
    for min_p in [0.0f32, -1.0, -0.0] {
        assert_eq!(
            offer_draft(&idx, &[], 1, 4, &logits, min_p),
            vec![3, 4],
            "min_p={min_p} が無効になっていない"
        );
    }
}

#[test]
fn min_p_probability_boundary_is_inclusive() {
    // logits 5 値の一様分布では各確率はちょうど 0.2 (f32 で正確)。
    // 実装は `p < min_p` なので min_p == 0.2 は通過、0.2 をわずかに超えると落ちる。
    let idx = minp_fixture();
    let logits = vec![0.0f32, 0.0, 0.0, 0.0, 0.0]; // token 3 の確率 = 1/5 = 0.2
    assert_eq!(offer_draft(&idx, &[], 1, 4, &logits, 0.2), vec![3, 4], "p == min_p は通過");
    assert!(
        offer_draft(&idx, &[], 1, 4, &logits, 0.2000001).is_empty(),
        "p < min_p は draft を落とす"
    );
}

#[test]
fn min_p_one_requires_certainty() {
    // min_p=1.0 は「確率 1.0 の候補」だけ通す。
    let idx = minp_fixture();
    let flat = vec![0.0f32; 5];
    assert!(offer_draft(&idx, &[], 1, 4, &flat, 1.0).is_empty(), "min_p=1.0 で平坦分布が通った");

    // token 3 だけ極端に高い (確率 ≈ 1.0) → 通る
    let mut peaked = vec![-1000.0f32; 5];
    peaked[3] = 1000.0;
    // f32 の exp(-2000) は 0 になるため確率はちょうど 1.0 になる。
    assert_eq!(offer_draft(&idx, &[], 1, 4, &peaked, 1.0), vec![3, 4], "確率 1.0 の候補が落ちた");
}

#[test]
fn min_p_only_inspects_first_draft_token() {
    // draft[0]=3 は高確率・draft[1]=4 はほぼゼロでも、判定は先頭のみなので通る。
    let idx = minp_fixture();
    let mut logits = vec![-30.0f32; 5];
    logits[3] = 30.0;
    assert_eq!(offer_draft(&idx, &[], 1, 4, &logits, 0.5), vec![3, 4], "2 番目以降を見て落としている");
}

#[test]
fn min_p_out_of_range_draft_token_bypasses_filter() {
    // 語彙 3 に対し draft[0]=3 は範囲外 → 確率計算せず draft をそのまま返す。
    let mut idx = NgramIndex::new(3, 100, 8, 4);
    idx.feed(&[1, 3, 4]);
    let logits = vec![0.0f32; 3];
    assert_eq!(offer_draft(&idx, &[], 1, 4, &logits, 1.0), vec![3, 4], "語彙外 draft が落とされた");
}

#[test]
fn min_p_nan_does_not_filter() {
    // 契約: min_p は `<= 0.0` で無効化され、比較は `p < min_p`。NaN はどちらも
    // 偽なので「有効だが何も落とさない」= draft 素通し (パニックしない)。
    let idx = minp_fixture();
    let logits = vec![0.0f32, 0.0, 0.0, -50.0, 0.0];
    assert_eq!(offer_draft(&idx, &[], 1, 4, &logits, f32::NAN), vec![3, 4]);
}

// ------------------------------------------------ repetition penalty 単調性

#[test]
fn repetition_penalty_positive_logits_are_monotonically_decreasing() {
    let base = 5.0f32;
    let mut prev = f32::INFINITY;
    for penalty in [1.0f32, 1.1, 1.5, 2.0, 3.0, 10.0] {
        let mut logits = vec![base, 1.0];
        apply_repetition_penalty(&mut logits, &[0], penalty);
        assert!(logits[0] <= prev, "penalty={penalty} で単調減少が破れた: {prev} -> {}", logits[0]);
        assert_eq!(logits[0], base / penalty, "penalty={penalty} の除算値が違う");
        assert_eq!(logits[1], 1.0, "文脈外が変わった");
        prev = logits[0];
    }
}

#[test]
fn repetition_penalty_negative_logits_are_monotonically_decreasing() {
    let base = -2.0f32;
    let mut prev = f32::INFINITY;
    for penalty in [1.0f32, 1.2, 2.0, 5.0] {
        let mut logits = vec![base];
        apply_repetition_penalty(&mut logits, &[0], penalty);
        assert!(logits[0] <= prev, "penalty={penalty} で単調減少が破れた");
        assert_eq!(logits[0], base * penalty, "penalty={penalty} の乗算値が違う");
        prev = logits[0];
    }
}

#[test]
fn repetition_penalty_keeps_zero_logit_at_zero() {
    // `> 0.0` が偽なので 0.0 * penalty = 0.0 (符号も変わらない)。
    for penalty in [1.0f32, 1.5, 4.0] {
        let mut logits = vec![0.0f32, 1.0];
        apply_repetition_penalty(&mut logits, &[0], penalty);
        assert_eq!(logits[0].to_bits(), 0.0f32.to_bits(), "penalty={penalty} で 0 が動いた");
    }
}

#[test]
fn repetition_penalty_below_one_amplifies_monotonically() {
    // HF 規則では penalty<1 は「既出トークンの強調」。正の logit は増加する。
    let base = 2.0f32;
    let mut prev = f32::NEG_INFINITY;
    for penalty in [1.0f32, 0.9, 0.5, 0.25] {
        let mut logits = vec![base];
        apply_repetition_penalty(&mut logits, &[0], penalty);
        assert!(logits[0] >= prev, "penalty={penalty} で単調増加が破れた");
        assert_eq!(logits[0], base / penalty);
        prev = logits[0];
    }
}

#[test]
fn repetition_penalty_greedy_flip_is_monotone() {
    // logits [3.0(未出), 5.0(既出)] を penalty で徐々に抑えると、
    // argmax は「1 → 0」の 1 回だけ反転し、決して 1 に戻らない。
    let mut flipped = false;
    let mut last = 1usize;
    for penalty in [1.0f32, 1.2, 1.4, 1.5, 1.6, 2.0, 3.0, 10.0] {
        let mut logits = vec![3.0f32, 5.0];
        apply_repetition_penalty(&mut logits, &[1], penalty);
        let pick = naive_argmax(&logits);
        if flipped {
            assert_eq!(pick, 0, "penalty={penalty} で argmax が 1 に戻った");
        } else if pick == 0 {
            flipped = true;
        }
        assert!(pick == last || pick == 0, "未定義の遷移 {last} -> {pick}");
        last = pick;
    }
    assert!(flipped, "penalty を上げても argmax が反転しなかった");
}

#[test]
fn repetition_penalty_context_gap_widens_monotonically() {
    // 既出トークンと未出トークンの差 (未出 - 既出) が penalty とともに広がる。
    let mut prev_gap = f32::NEG_INFINITY;
    for penalty in [1.0f32, 1.5, 2.0, 4.0] {
        let mut logits = vec![1.0f32, 4.0];
        apply_repetition_penalty(&mut logits, &[1], penalty);
        let gap = logits[0] - logits[1];
        assert!(gap >= prev_gap, "penalty={penalty} で差が縮んだ: {prev_gap} -> {gap}");
        prev_gap = gap;
    }
}

#[test]
fn repetition_penalty_greedy_selection_is_deterministic() {
    // penalty 適用後の greedy は seed にも RNG にも依存しない。
    let mut logits = vec![3.0f32, 5.0, 4.5];
    apply_repetition_penalty(&mut logits, &[1], 1.5);
    let a = {
        let mut s = sampler(Some(0.0), None, None, None);
        s.sample(&logits)
    };
    let b = {
        let mut s = sampler(Some(0.0), None, None, Some(u64::MAX));
        s.sample(&logits)
    };
    assert_eq!(a, b, "penalty 後の greedy が seed 依存");
    assert_eq!(a, naive_argmax(&logits));
}
