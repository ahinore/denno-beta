//! Sampler の決定論・安全性テスト (task 942)。
//!
//! `src/sampler.rs` の実装を **公開 API のみ** で監査し、次を固定する:
//!
//! 1. クロスラン seed 決定論: 固定 seed + 固定 logits 列 → 固定のトークン列
//!    (golden)。プロセスを跨いでも同じ列になることを、値をそのまま焼き込んで
//!    検証する (rand の RNG 実装・サンプリング経路が黙って変われば落ちる)。
//! 2. タイブレークの安定性: 同点は「値降順 → 添字昇順」で、top-k / top-p の
//!    候補集合もその規則に従う (全同値なら最小 k 添字、±0.0 は同値)。
//! 3. NaN / Inf の安全性: パニックしない・決定論的・範囲内。NaN は top-k の
//!    候補から外れ、+inf は softmax 支配トークンとして選ばれる。
//!    (task 942 以前は NaN 混入 + k == 語彙長で `sort_by` が
//!    "comparison function does not correctly implement a total order" で
//!     パニックした。ここが回帰テスト。)
//! 4. 反復ペナルティの窓の端: 空文脈・先頭/末尾添字・語彙境界・語彙外 id・
//!    重複・n-gram 窓の最初/最後の一致。
//!
//! CPU-only・決定的。乱数は自前の LCG のみ (プロセス外状態に依存しない)。
//! 語彙長 0 は `Sampler` の契約外なので扱わない。

use denno_engine::sampler::{
    apply_freq_presence_penalty, apply_no_repeat_ngram, apply_repetition_penalty, argmax_first,
    context_counts, AntiRepetition, FreqPresencePenalty, Sampler,
};
use denno_engine::spec_ngram::{offer_draft, NgramIndex};

/// 決定的な logits 生成器 (線形合同法)。テスト内で再生成できるよう、
/// 生成規則そのものをここに固定する。
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

fn seq(s: &mut Sampler, logits: &[f32], n: usize) -> Vec<usize> {
    (0..n).map(|_| s.sample(logits)).collect()
}

fn bits_of(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 素直な比較鎖の argmax (参照実装。同点は先頭・NaN は無視)。
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

// ==================================================== 1. クロスラン seed 決定論

/// `Lcg::new(0x51).vec(64)` を seed 42 / temperature 1.0 で 24 回サンプルした列。
/// 焼き込み値: 別プロセス・別日に実行しても同じ列であることの錨。
const GOLDEN_PLAIN_SEED42: [usize; 24] = [
    7, 29, 13, 30, 53, 35, 62, 22, 61, 3, 35, 22, 15, 39, 10, 53, 35, 7, 26, 3, 46, 55, 35, 27,
];

/// 同じ logits を temperature 0.8 / top_k 8 / seed 7 で 24 回。
const GOLDEN_TOPK_SEED7: [usize; 24] = [
    15, 35, 35, 53, 35, 35, 3, 22, 22, 53, 3, 7, 22, 35, 3, 53, 15, 35, 15, 3, 22, 3, 15, 35,
];

/// 同じ logits を temperature 1.0 / top_p 0.9 / seed 5 で 24 回 (語彙 <= 2048 の厳密経路)。
const GOLDEN_TOPP_SEED5: [usize; 24] = [
    53, 22, 53, 35, 38, 13, 14, 40, 15, 10, 38, 3, 22, 53, 39, 40, 3, 13, 53, 61, 7, 38, 3, 4,
];

fn golden_logits() -> Vec<f32> {
    Lcg::new(0x51).vec(64)
}

#[test]
fn golden_plain_sequence_is_stable_across_runs() {
    let logits = golden_logits();
    let got = seq(&mut sampler(Some(1.0), None, None, Some(42)), &logits, 24);
    assert_eq!(got, GOLDEN_PLAIN_SEED42, "plain 経路のクロスラン列が変わった");
}

#[test]
fn golden_top_k_sequence_is_stable_across_runs() {
    let logits = golden_logits();
    let got = seq(&mut sampler(Some(0.8), Some(8), None, Some(7)), &logits, 24);
    assert_eq!(got, GOLDEN_TOPK_SEED7, "top-k 経路のクロスラン列が変わった");
}

#[test]
fn golden_top_p_sequence_is_stable_across_runs() {
    let logits = golden_logits();
    let got = seq(&mut sampler(Some(1.0), None, Some(0.9), Some(5)), &logits, 24);
    assert_eq!(got, GOLDEN_TOPP_SEED5, "top-p 経路のクロスラン列が変わった");
}

/// 同じ seed の 2 インスタンスは、途中で交互に呼んでも互いに干渉しない
/// (RNG 状態がインスタンス固有 = グローバル状態が無い)。
#[test]
fn independent_samplers_do_not_share_rng_state() {
    let logits = golden_logits();
    let mut a = sampler(Some(1.0), None, None, Some(42));
    let mut b = sampler(Some(1.0), None, None, Some(42));
    // まず 7 回ずつ進める (golden の先頭 7 個と一致)
    assert_eq!(seq(&mut a, &logits, 7), GOLDEN_PLAIN_SEED42[..7]);
    assert_eq!(seq(&mut b, &logits, 7), GOLDEN_PLAIN_SEED42[..7]);
    // 片方だけ先に進めてから交互に呼ぶ。互いの RNG 状態を乱さない。
    let _ = seq(&mut b, &logits, 5);
    let rest_a = seq(&mut a, &logits, 17);
    assert_eq!(rest_a, GOLDEN_PLAIN_SEED42[7..], "他インスタンスの進行で列が変わった");
    let rest_b = seq(&mut b, &logits, 12);
    assert_eq!(rest_b, GOLDEN_PLAIN_SEED42[12..], "先行実行の続きが同じでない");
}

/// greedy は seed にも RNG にも依存しない (golden: argmax 固定)。
#[test]
fn greedy_sequence_is_seed_independent() {
    let logits = golden_logits();
    let want = naive_argmax(&logits);
    for seed in [None, Some(0), Some(u64::MAX)] {
        let got = seq(&mut sampler(Some(0.0), None, None, seed), &logits, 3);
        assert_eq!(got, vec![want; 3], "seed={seed:?} で greedy が動いた");
    }
}

// ==================================================== 2. タイブレークの安定性

#[test]
fn all_equal_top_k_candidates_are_the_lowest_indices() {
    // 全同値では「値降順 → 添字昇順」の規則から、候補は必ず最小 k 添字。
    let logits = vec![2.5f32; 9];
    for k in 1usize..=8 {
        let mut s = sampler(Some(1.0), Some(k), None, Some(1));
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..2000 {
            seen.insert(s.sample(&logits));
        }
        let want: std::collections::BTreeSet<usize> = (0..k).collect();
        assert_eq!(seen, want, "k={k} で候補集合が最小 k 添字でない");
    }
}

#[test]
fn ties_at_the_top_k_boundary_pick_lowest_indices() {
    // 最大値が 4 個 (添字 0..3)、残りは小さい。[5,5,5,5,1,1]
    let logits = vec![5.0f32, 5.0, 5.0, 5.0, 1.0, 1.0];
    for (k, want) in [(2usize, vec![0usize, 1]), (3, vec![0, 1, 2]), (5, vec![0, 1, 2, 3, 4])] {
        let mut s = sampler(Some(1.0), Some(k), None, Some(1));
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..4000 {
            seen.insert(s.sample(&logits));
        }
        assert_eq!(seen.into_iter().collect::<Vec<_>>(), want, "k={k} の境界タイ");
    }
}

#[test]
fn signed_zero_ties_are_equivalent_in_top_k() {
    // +0.0 と -0.0 は同値 (添字昇順)。候補は最小添字から。
    let logits = vec![0.0f32, -0.0, 0.0, -0.0, -3.0];
    let mut s = sampler(Some(1.0), Some(2), None, Some(3));
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..2000 {
        seen.insert(s.sample(&logits));
    }
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), vec![0, 1], "±0.0 のタイ");
}

#[test]
fn repeated_tie_sampling_is_bit_identical() {
    // 同点だらけの logits でも 2 インスタンスの系列は一致する。
    let logits = vec![1.0f32, 1.0, 0.0, 0.0, 1.0, -0.0];
    for (t, k, p) in [
        (1.0f32, None, None),
        (1.0, Some(3), None),
        (1.0, Some(6), None),
        (1.0, None, Some(0.6)),
    ] {
        let a = seq(&mut sampler(Some(t), k, p, Some(9)), &logits, 40);
        let b = seq(&mut sampler(Some(t), k, p, Some(9)), &logits, 40);
        assert_eq!(a, b, "タイ入り logits で系列が割れた: k={k:?} p={p:?}");
    }
}

#[test]
fn top_p_tie_cutoff_is_lowest_indices() {
    // 全同値 8 個・p=0.75 → 累積は 6 個目で 0.75 に達する → 候補は 0..5。
    let logits = vec![1.0f32; 8];
    let mut s = sampler(Some(1.0), None, Some(0.75), Some(4));
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..3000 {
        seen.insert(s.sample(&logits));
    }
    assert_eq!(seen.into_iter().collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
}

// ==================================================== 3. NaN / Inf の安全性

/// 回帰: NaN 混入 + k == 語彙長 で `sort_by` がパニックしていた (task 942)。
#[test]
fn nan_with_full_vocab_top_k_does_not_panic() {
    let mut logits: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
    for i in (0..64).step_by(5) {
        logits[i] = f32::NAN;
    }
    let mut s = sampler(Some(1.0), Some(64), None, Some(2));
    let picks = seq(&mut s, &logits, 64);
    assert!(picks.iter().all(|&i| i < logits.len()), "範囲外の添字");
    assert!(!picks.iter().any(|&i| logits[i].is_nan()), "NaN の語彙が選ばれた");

    // top_p のフルパス・フォールバック経路 (語彙 > 2048) でもパニックしない。
    let mut big: Vec<f32> = (0..3000).map(|i| (i as f32) * 0.001).collect();
    big[0] = f32::NAN;
    big[1500] = f32::NAN;
    let a = seq(&mut sampler(Some(1.0), None, Some(0.9), Some(2)), &big, 16);
    let b = seq(&mut sampler(Some(1.0), None, Some(0.9), Some(2)), &big, 16);
    assert_eq!(a, b, "NaN 入り大語彙 top_p が再現しない");
    assert!(a.iter().all(|&i| i < big.len() && !big[i].is_nan()));
}

#[test]
fn nan_is_never_sampled_in_top_k() {
    let logits = vec![1.0f32, f32::NAN, 3.0, 2.0, f32::NAN];
    for k in [1usize, 3, 5] {
        let picks = seq(&mut sampler(Some(1.0), Some(k), None, Some(2)), &logits, 300);
        assert!(!picks.iter().any(|&i| i == 1 || i == 4), "k={k} で NaN が選ばれた");
    }
}

/// NaN は「順位なし」= 降順の末尾。最大の有限値が選ばれる。
#[test]
fn nan_sorts_after_all_finite_values() {
    let mut s = sampler(Some(0.0), None, None, None);
    assert_eq!(s.sample(&[f32::NAN, 1.0, 2.0, f32::NAN]), 2, "greedy は NaN 無視");
    // top_k=1 も同じ (NaN は候補外)
    let mut s = sampler(Some(1.0), Some(1), None, Some(1));
    for _ in 0..16 {
        assert_eq!(s.sample(&[f32::NAN, 1.0, 2.0, f32::NAN]), 2);
    }
}

#[test]
fn positive_infinity_is_selected_as_dominant_token() {
    // +inf は softmax で確率 1。全経路で同じ添字 (同値なら最小) を返す。
    let logits = vec![1.0f32, f32::INFINITY, 3.0, 2.0];
    for (k, p) in [
        (None::<usize>, None::<f32>),
        (Some(3), None),
        (Some(4), None),
        (None, Some(0.9)),
        (Some(2), Some(0.5)),
    ] {
        let picks = seq(&mut sampler(Some(1.0), k, p, Some(3)), &logits, 32);
        assert_eq!(picks, vec![1usize; 32], "+inf が選ばれない: k={k:?} p={p:?}");
    }
    // 複数の +inf は最小添字。
    let two = vec![0.0f32, f32::INFINITY, f32::INFINITY];
    let picks = seq(&mut sampler(Some(1.0), None, None, Some(3)), &two, 8);
    assert_eq!(picks, vec![1usize; 8]);
    // 語彙 > 2048 の top_p 部分集合経路でも同じ。
    let mut big = vec![0.0f32; 3000];
    big[2999] = f32::INFINITY;
    let picks = seq(&mut sampler(Some(1.0), None, Some(0.9), Some(3)), &big, 8);
    assert_eq!(picks, vec![2999usize; 8]);
}

#[test]
fn all_non_finite_logits_are_deterministic_and_in_range() {
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("all -inf", vec![f32::NEG_INFINITY; 8]),
        ("all +inf", vec![f32::INFINITY; 8]),
        ("all NaN", vec![f32::NAN; 8]),
        (
            "mixed",
            vec![f32::NAN, f32::NEG_INFINITY, f32::INFINITY, f32::NAN, 0.0, -0.0, 1.0, f32::NAN],
        ),
    ];
    for (name, logits) in cases {
        for (k, p) in [
            (None::<usize>, None::<f32>),
            (Some(1), None),
            (Some(4), None),
            (Some(8), None),
            (None, Some(0.9)),
            (Some(3), Some(0.5)),
        ] {
            let a = seq(&mut sampler(Some(1.0), k, p, Some(6)), &logits, 12);
            let b = seq(&mut sampler(Some(1.0), k, p, Some(6)), &logits, 12);
            assert_eq!(a, b, "{name}: k={k:?} p={p:?} で再現しない");
            assert!(a.iter().all(|&i| i < logits.len()), "{name}: 範囲外の添字");
        }
    }
}

#[test]
fn extreme_temperature_is_deterministic() {
    let logits: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
    let argmax = naive_argmax(&logits);
    // 極小 temperature は softmax が飽和して argmax 固定 (x/t が inf になっても安全)
    for t in [f32::MIN_POSITIVE, 1e-30f32] {
        for (k, p) in [(None::<usize>, None::<f32>), (Some(3), None), (None, Some(0.9))] {
            let picks = seq(&mut sampler(Some(t), k, p, Some(2)), &logits, 8);
            assert_eq!(picks, vec![argmax; 8], "t={t} k={k:?} p={p:?} で飽和していない");
        }
    }
    // 巨大 temperature / inf / NaN 温度は「パニックせず決定論的」だけを契約とする。
    for t in [1e30f32, f32::INFINITY, f32::NAN] {
        let a = seq(&mut sampler(Some(t), None, None, Some(2)), &logits, 8);
        let b = seq(&mut sampler(Some(t), None, None, Some(2)), &logits, 8);
        assert_eq!(a, b, "t={t} で再現しない");
        assert!(a.iter().all(|&i| i < logits.len()));
    }
}

// ==================================================== 4. 反復ペナルティの窓の端

/// 「厄介な」logits: 0.0 / -0.0 / 負 / 非正規化 / -inf / +inf を含む。
fn tricky_logits(n: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..n)
        .map(|i| match i % 6 {
            0 => 0.0,
            1 => -0.0,
            2 => -(i as f32) * 0.125,
            3 => f32::MIN_POSITIVE / 4.0,
            4 => i as f32 * 0.37,
            _ => -1e30_f32.recip(),
        })
        .collect();
    if n > 3 {
        v[3] = f32::NEG_INFINITY;
    }
    if n > 4 {
        v[4] = f32::INFINITY;
    }
    v
}

fn all_on() -> AntiRepetition {
    AntiRepetition {
        repetition_penalty: Some(1.3),
        frequency_penalty: Some(0.4),
        presence_penalty: Some(0.2),
        no_repeat_ngram_size: Some(3),
    }
}

/// 空文脈は 4 つ全部有効でもビット完全 no-op (窓の下端)。
#[test]
fn empty_context_is_bit_identical_noop() {
    let base = tricky_logits(48);
    let mut x = base.clone();
    all_on().apply(&mut x, &[]);
    assert_eq!(bits_of(&x), bits_of(&base), "空文脈でビットが変わった");
    // 個別関数でも同じ
    let mut y = base.clone();
    apply_repetition_penalty(&mut y, &[], 2.0);
    apply_freq_presence_penalty(&mut y, &context_counts(&[], base.len()), &FreqPresencePenalty { frequency: 1.0, presence: 1.0 });
    apply_no_repeat_ngram(&mut y, &[], 3);
    assert_eq!(bits_of(&y), bits_of(&base));
}

/// 窓の両端の添字 (0 と len-1) だけが変わり、他はビット不変。
#[test]
fn penalty_touches_only_first_and_last_indices() {
    let n = 8usize;
    let mut x = vec![0.0f32; n];
    x[0] = 2.0;
    x[n - 1] = -2.0;
    let base = x.clone();
    let anti = AntiRepetition { repetition_penalty: Some(2.0), ..Default::default() };
    anti.apply(&mut x, &[0, n as u32 - 1]);
    assert_eq!(x[0], 1.0, "先頭添字 2.0/2.0");
    assert_eq!(x[n - 1], -4.0, "末尾添字 -2.0*2.0");
    for i in 1..n - 1 {
        assert_eq!(x[i].to_bits(), base[i].to_bits(), "添字 {i} が動いた");
    }
}

/// 語彙境界: 有効な最後の id (len-1) は触り、len 以上 / u32::MAX は無視。
#[test]
fn penalty_vocab_boundary_ids() {
    let mut x = vec![3.0f32, 3.0, 3.0, 3.0];
    let base = x.clone();
    apply_repetition_penalty(&mut x, &[3, 4, u32::MAX], 1.5);
    assert_eq!(x[0], 3.0, "添字 0 は文脈外");
    assert_eq!(x[1], 3.0);
    assert_eq!(x[2], 3.0);
    assert_eq!(x[3], 2.0, "添字 3 = len-1 は適用される");
    assert_ne!(x, base);

    // counts も同様: len-1 は数え、len 以上は数えない
    let counts = context_counts(&[3u32, 4, u32::MAX], 4);
    assert_eq!(counts, vec![0, 0, 0, 1]);
}

/// 文脈の重複は repetition では 1 回、frequency では回数だけ効く (窓の端で重複)。
#[test]
fn duplicate_ids_at_window_edges() {
    let mut x = vec![2.0f32, 2.0];
    apply_repetition_penalty(&mut x, &[0, 0, 1, 1, 1], 2.0);
    assert_eq!(x, vec![1.0, 1.0], "重複しても 1 回だけ (2.0/2.0)");

    let counts = context_counts(&[0u32, 0, 1, 1, 1], 2);
    assert_eq!(counts, vec![2, 3]);
    let mut y = vec![0.0f32, 0.0];
    apply_freq_presence_penalty(&mut y, &counts, &FreqPresencePenalty { frequency: 0.5, presence: 0.1 });
    assert_eq!(y[0], -(0.5f32 * 2.0 + 0.1), "先頭: 2 回");
    assert_eq!(y[1], -(0.5f32 * 3.0 + 0.1), "末尾: 3 回");
}

/// counts が logits より短い/長くても min で切り、パニックしない。
#[test]
fn freq_presence_counts_length_mismatch_is_clamped() {
    let mut a = vec![1.0f32, 1.0, 1.0];
    apply_freq_presence_penalty(&mut a, &[1u32], &FreqPresencePenalty { frequency: 1.0, presence: 0.0 });
    assert_eq!(a, vec![0.0, 1.0, 1.0], "短い counts は添字 0 から min 個だけ");

    let mut b = vec![1.0f32, 1.0];
    apply_freq_presence_penalty(&mut b, &[1u32, 1, 1], &FreqPresencePenalty { frequency: 1.0, presence: 0.0 });
    assert_eq!(b, vec![0.0, 0.0], "長い counts の余りは無視 (先頭 2 個だけ)");
}

/// n-gram 窓の端: 最初の窓 (i=0) の一致で添字 0 を、最後の窓
/// (i = len-n) の一致で添字 len-1 を禁止できる。
#[test]
fn no_repeat_ngram_bans_at_both_window_edges() {
    // 文脈 [0,1,2, 0,1, 0,1] n=3 / 末尾接頭辞 (0,1)
    //   i=0: (0,1,2) 一致 → 2 を禁止
    //   i=3: (0,1, ...) は i+3=6 <= 5? last_start = 7-3 = 4 なので i=3 で (0,1,0) → 0 を禁止
    let mut logits = vec![0.0f32; 3];
    apply_no_repeat_ngram(&mut logits, &[0u32, 1, 2, 0, 1, 0, 1], 3);
    assert_eq!(logits[2], f32::NEG_INFINITY, "i=0 の窓の続き 2");
    assert_eq!(logits[0], f32::NEG_INFINITY, "i=3 の窓の続き 0");
    assert_eq!(logits[1], 0.0, "1 は禁止されない");

    // 最後の窓 (i = len-n) が末尾接頭辞に一致する形で len-1 を禁止する。
    // 文脈 [5,1,2, 5,1,2] n=3 → 末尾接頭辞 (1,2)、i=1 と i=4 が該当するが
    // i=4 は i+2 = 6 で文脈外 → 続き無し。i=1 の続きは 2? (1,2,5) → 5。
    let mut lg = vec![0.0f32; 6];
    apply_no_repeat_ngram(&mut lg, &[5u32, 1, 2, 5, 1, 2], 3);
    assert_eq!(lg[5], f32::NEG_INFINITY, "接頭辞 (1,2) の続き 5");
    // n == 文脈長は窓が 1 個だけで続きを持たない → no-op
    let base = tricky_logits(8);
    let mut x = base.clone();
    apply_no_repeat_ngram(&mut x, &[0u32, 7, 3], 3);
    assert_eq!(bits_of(&x), bits_of(&base), "n == 文脈長は no-op");
    // n > 文脈長も no-op
    let mut y = base.clone();
    apply_no_repeat_ngram(&mut y, &[0u32, 7], 3);
    assert_eq!(bits_of(&y), bits_of(&base), "n > 文脈長は no-op");
}

/// n=2 の窓の端: 末尾の直前トークン自身は禁止せず、その過去の続きだけを禁止。
#[test]
fn no_repeat_ngram_n2_edges() {
    // 文脈 [0, 1, 0] → 末尾 0 の過去の続き 1 を禁止
    let mut l = vec![0.0f32; 2];
    apply_no_repeat_ngram(&mut l, &[0u32, 1, 0], 2);
    assert_eq!(l[1], f32::NEG_INFINITY);
    assert_eq!(l[0], 0.0, "直前トークン 0 自身は禁止しない");

    // 先頭添字 0 の禁止: 文脈 [2, 0, 2] → 接頭辞 2 の続き 0 を禁止
    let mut l = vec![0.0f32; 3];
    apply_no_repeat_ngram(&mut l, &[2u32, 0, 2], 2);
    assert_eq!(l[0], f32::NEG_INFINITY, "先頭添字 0 が禁止されていない");
    assert_eq!(l[2], 0.0);

    // 末尾添字 (len-1) の禁止: 文脈 [5, 1, 3, 3] → 接頭辞 3 の続き 3 を禁止
    let mut l = vec![0.0f32; 4];
    apply_no_repeat_ngram(&mut l, &[5u32, 1, 3, 3], 2);
    assert_eq!(l[3], f32::NEG_INFINITY, "末尾添字 (len-1) が禁止されていない");
    assert_eq!(l[0], 0.0);

    // 文脈 1 トークンは n=2 で no-op
    let base = tricky_logits(4);
    let mut x = base.clone();
    apply_no_repeat_ngram(&mut x, &[0u32], 2);
    assert_eq!(bits_of(&x), bits_of(&base));
}

/// ペナルティ全体 (固定順) の適用は 2 回でビット同一・RNG 不関与。
#[test]
fn anti_repetition_on_window_edges_is_deterministic() {
    let base = tricky_logits(32);
    // 窓の端ばかりの文脈 (0 と 31 を交互 + 重複)
    let ctx: Vec<u32> = vec![0, 31, 0, 31, 0, 31, 1, 30];
    let anti = all_on();
    let mut a = base.clone();
    let mut b = base.clone();
    anti.apply(&mut a, &ctx);
    anti.apply(&mut b, &ctx);
    assert_eq!(bits_of(&a), bits_of(&b), "2 回で不一致");
    // 空文脈版との差は「適用されている」ことの裏取り
    let mut c = base.clone();
    anti.apply(&mut c, &[]);
    assert_ne!(bits_of(&a), bits_of(&c), "端の文脈が何も変えていない");

    // 同じサンプラに「適用済み logits」を与えた場合と、毎回適用する場合で
    // 系列が一致する = ペナルティが RNG を消費しない。
    let mut s1 = sampler(Some(1.0), None, None, Some(13));
    let mut s2 = sampler(Some(1.0), None, None, Some(13));
    let seq1 = seq(&mut s1, &a, 24);
    let seq2: Vec<usize> = (0..24)
        .map(|_| {
            let mut x = base.clone();
            anti.apply(&mut x, &ctx);
            s2.sample(&x)
        })
        .collect();
    assert_eq!(seq1, seq2, "ペナルティ適用が RNG 消費を変えた");
}

/// `argmax_first` は公開 API としても同値 (タイブレーク・NaN 無視)。
#[test]
fn argmax_first_matches_naive_reference() {
    for n in [1usize, 2, 7, 8, 9, 64, 1023] {
        let v = Lcg::new(0x942 + n as u64).vec(n);
        assert_eq!(argmax_first(&v), naive_argmax(&v), "n={n}");
    }
    let v = vec![f32::NAN, 1.0, 1.0, f32::NAN];
    assert_eq!(argmax_first(&v), 1, "同点は先頭・NaN は無視");
}

// ==================================================== 5. min-p (spec_ngram)

/// draft が [3, 4] を返す辞書 (キー [1] の継続)。
fn minp_fixture() -> NgramIndex {
    let mut idx = NgramIndex::new(3, 100, 8, 4);
    idx.feed(&[1, 3, 4]);
    idx
}

/// min-p の判定は同じ入力なら何度呼んでも同一 (決定論的・状態を持たない)。
#[test]
fn min_p_filter_is_deterministic() {
    let idx = minp_fixture();
    // token 3 の softmax 確率 ≈ 0.649 (exp(2)/(4+exp(2)))
    let logits = vec![0.0f32, 0.0, 0.0, 2.0, 0.0];
    let first = offer_draft(&idx, &[], 1, 4, &logits, 0.5);
    for _ in 0..16 {
        assert_eq!(offer_draft(&idx, &[], 1, 4, &logits, 0.5), first, "min-p が非決定");
    }
    assert_eq!(first, vec![3, 4], "閾値 0.5 を超える候補が落ちた");
    assert!(offer_draft(&idx, &[], 1, 4, &logits, 0.7).is_empty(), "閾値未満が通った");
}

/// 非有限 logits の min-p はパニックせず draft を素通しする
/// (`total` が NaN/inf になり `p < min_p` が偽 = フィルタしない)。
#[test]
fn min_p_with_non_finite_logits_does_not_panic() {
    let idx = minp_fixture();
    let cases: Vec<(&str, Vec<f32>)> = vec![
        ("nan", vec![f32::NAN; 5]),
        ("nan-mixed", vec![0.0, 0.0, 0.0, f32::NAN, 0.0]),
        ("+inf", vec![0.0, 0.0, 0.0, f32::INFINITY, 0.0]),
        ("all -inf", vec![f32::NEG_INFINITY; 5]),
    ];
    for (name, logits) in cases {
        let a = offer_draft(&idx, &[], 1, 4, &logits, 0.9);
        let b = offer_draft(&idx, &[], 1, 4, &logits, 0.9);
        assert_eq!(a, b, "{name}: min-p が非決定");
        assert_eq!(a, vec![3, 4], "{name}: 非有限 logits で draft が落ちた");
    }
}
