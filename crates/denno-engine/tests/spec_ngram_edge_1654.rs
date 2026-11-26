//! task 1654: 投機的デコード用 n-gram 辞書 (`spec_ngram`) の境界を、公開 API
//! (`NgramIndex` / `offer_draft`) 越しに固定する。
//!
//! 何を見ているか:
//! - 空の履歴・`feed` 前の辞書 (常に空 draft = 通常 decode にフォールバック)。
//! - n-gram 次数より短い履歴 (登録できる suffix は `min(max_suffix, i)` まで。
//!   末尾の suffix が辞書に無ければ空、あれば短い suffix で引ける)。
//! - 周期パターン (末尾の継続は feed 末尾で切り詰められるので、draft はその
//!   切り詰め長まで。`k` を超える要求は候補長で頭打ち)。
//! - draft 長のクランプ (`k = 0`、`k` > 継続長、`NgramIndex::max_draft`)。
//! - `min_p` の端 (0.0 / 負 / 1.0 / 1.0 直上 / NaN / +inf / -inf logits /
//!   logits 長不足)。フィルタは候補の生成だけに効き、辞書状態は変えない。
//!
//! 期待値はすべてこのファイル内で手計算した固定値 (実装の関数を参照実装として
//! 呼び直さない)。CPU のみ・単スレッド・決定論的。GPU / CUDA / ファイル /
//! 環境変数は触らない。

use denno_engine::spec_ngram::{offer_draft, NgramIndex};

/// テスト用の短い名前 (引数は `new` と同じ順)。
fn idx(max_suffix: usize, max_entries: usize, cand_keep: usize, max_draft: usize) -> NgramIndex {
    NgramIndex::new(max_suffix, max_entries, cand_keep, max_draft)
}

// ---------------------------------------------------------------- 空の履歴

#[test]
fn empty_index_never_drafts() {
    let d = idx(3, 100, 8, 4);
    assert_eq!(d.len(), 0, "feed 前はエントリ 0");
    assert!(d.is_empty());
    assert_eq!(d.fed(), 0, "feed 前は fed=0");

    // 空トークン列でも、未登録の履歴でも draft は空。
    assert!(d.draft(&[], 4).is_empty(), "空履歴");
    assert!(d.draft(&[1, 2, 3], 4).is_empty(), "未 feed の履歴");
    assert!(d.draft(&[1, 2, 3], 0).is_empty(), "k=0");

    // offer_draft も同じ (min_p を掛けても空のまま)。
    let logits = [0.0f32; 4];
    assert!(offer_draft(&d, &[], 7, 4, &logits, 0.0).is_empty());
    assert!(offer_draft(&d, &[1, 2], 3, 4, &logits, 0.5).is_empty());
    assert!(offer_draft(&d, &[], 7, 4, &logits, 1.0).is_empty());
}

#[test]
fn feeding_empty_slice_keeps_index_empty() {
    let mut d = idx(3, 100, 8, 4);
    d.feed(&[]);
    d.feed(&[]);
    assert_eq!(d.fed(), 0, "空 feed は fed を進めない");
    assert!(d.is_empty(), "空 feed はエントリを作らない");
    assert!(d.draft(&[], 4).is_empty());
}

// ------------------------------------------- n-gram 次数より短い履歴

#[test]
fn single_token_history_registers_no_entry() {
    // キーは「位置 i の手前」なので i=0 では登録対象が無い。
    let mut d = idx(3, 100, 8, 4);
    d.feed(&[5]);
    assert_eq!(d.fed(), 1);
    assert!(d.is_empty(), "1 トークンだけでは suffix が 1 つも作れない");
    assert!(d.draft(&[5], 4).is_empty());
    // 履歴が次数より短くても空 draft (呼び出し側は通常 decode へ)。
    assert!(offer_draft(&d, &[], 5, 4, &[0.0f32; 8], 0.0).is_empty());
}

#[test]
fn two_token_history_registers_only_length_one_suffix() {
    let mut d = idx(3, 100, 8, 4);
    d.feed(&[5, 9]);
    // i=1 で作れる suffix は min(3, 1) = 1 種類だけ: [5] -> [9]。
    assert_eq!(d.len(), 1);
    assert_eq!(d.draft(&[5], 4), vec![9]);
    // 末尾 [5,9] はキーとして登録されていない (継続が無い位置) ので空。
    assert!(d.draft(&[5, 9], 4).is_empty(), "2-gram は未登録のまま");
    assert_eq!(offer_draft(&d, &[], 5, 4, &[0.0f32; 8], 0.0), vec![9]);
    assert!(offer_draft(&d, &[5], 9, 4, &[0.0f32; 8], 0.0).is_empty());
}

#[test]
fn three_token_history_builds_up_to_two_gram() {
    let mut d = idx(3, 100, 8, 4);
    d.feed(&[5, 9, 7]);
    // i=1: [5] -> [9,7] / i=2: [5,9] -> [7] と [9] -> [7]。
    assert_eq!(d.len(), 3, "1-gram 2 種 + 2-gram 1 種");
    assert_eq!(d.draft(&[5, 9], 4), vec![7], "2-gram で最長一致");
    assert_eq!(d.draft(&[5], 4), vec![9, 7], "1-gram の継続は 2 トークン");
    // 3-gram は一度も登録されていない (継続の無い末尾はキーにならない)。
    assert!(d.draft(&[5, 9, 7], 4).is_empty(), "3-gram は未登録");
}

#[test]
fn shorter_history_with_two_token_max_suffix() {
    let mut d = idx(2, 100, 8, 4);
    d.feed(&[5, 9, 7]);
    assert_eq!(d.draft(&[5, 9], 4), vec![7]);
    // max_suffix=2 でも 1-gram へのフォールバックは効く。
    assert_eq!(d.draft(&[9], 4), vec![7]);
    assert!(d.draft(&[7], 4).is_empty(), "末尾トークンはキーにならない");
}

// ------------------------------------------------------ 周期パターン

#[test]
fn periodic_pattern_clamps_to_trailing_continuation() {
    let mut d = idx(3, 100, 8, 4);
    let seq: Vec<u32> = (0..5).flat_map(|_| [1u32, 2, 3]).collect(); // 15 トークン
    d.feed(&seq);

    // 末尾の [1,2,3] は feed 末尾に接しているので継続は [1,2,3] で切れる。
    // k は候補長で頭打ち、k=1/2 はその前置き。
    assert_eq!(d.draft(&[1, 2, 3], 10), vec![1, 2, 3], "候補長で頭打ち");
    assert_eq!(d.draft(&[1, 2, 3], 3), vec![1, 2, 3]);
    assert_eq!(d.draft(&[1, 2, 3], 2), vec![1, 2]);
    assert_eq!(d.draft(&[1, 2, 3], 1), vec![1]);
    assert!(d.draft(&[1, 2, 3], 0).is_empty(), "k=0");
}

#[test]
fn periodic_pattern_with_period_four_yields_full_cycle() {
    let mut d = idx(3, 100, 8, 4);
    let seq: Vec<u32> = (0..3).flat_map(|_| [7u32, 8, 9, 10]).collect(); // 12 トークン
    d.feed(&seq);
    assert_eq!(d.draft(&[9, 10], 4), vec![7, 8, 9, 10], "次周期の先頭から 4 個");
    assert_eq!(d.draft(&[9, 10], 100), vec![7, 8, 9, 10]);
    // 3-gram [7,9,10] は未登録なので 2-gram [9,10] まで短縮して引く。
    assert_eq!(d.draft(&[7, 9, 10], 4), vec![7, 8, 9, 10]);
    // min_p 無効時は offer_draft も同じ列を返す。
    let logits = [0.0f32; 16];
    assert_eq!(
        offer_draft(&d, &[9], 10, 4, &logits, 0.0),
        vec![7, 8, 9, 10]
    );
}

#[test]
fn refeeding_same_prefix_does_not_grow_index() {
    let mut d = idx(3, 100_000, 8, 4);
    let seq: Vec<u32> = (0..2).flat_map(|_| [1u32, 2, 3, 4, 5, 6]).collect();
    d.feed(&seq);
    let (len, fed) = (d.len(), d.fed());
    let first = d.draft(&[1, 2], 4);
    d.feed(&seq); // 差分ゼロ: 何も登録し直さない。
    assert_eq!(d.len(), len, "再 feed でエントリは増えない");
    assert_eq!(d.fed(), fed, "fed は進まない");
    assert_eq!(d.draft(&[1, 2], 4), first);
    assert_eq!(first, vec![3, 4, 5, 6]);
}

// -------------------------------------------------- draft 長のクランプ

#[test]
fn draft_k_clamps_to_candidate_length_prefix_consistently() {
    let mut d = idx(2, 100, 8, 4);
    let seq: Vec<u32> = (0..2).flat_map(|_| [1u32, 2, 3, 4, 5, 6]).collect();
    d.feed(&seq);
    let full = d.draft(&[1, 2], 100);
    assert_eq!(full, vec![3, 4, 5, 6], "継続は max_draft=4 で切れる");
    for k in 1..=full.len() {
        assert_eq!(d.draft(&[1, 2], k), full[..k].to_vec(), "k={k} は前置き");
    }
    assert!(d.draft(&[1, 2], 0).is_empty(), "k=0 は常に空");
    assert_eq!(d.draft(&[1, 2], 4), full, "k=候補長ちょうど");
    assert_eq!(d.draft(&[1, 2], 5), full, "k>候補長でも伸びない");
}

#[test]
fn max_draft_bounds_candidate_even_with_large_k() {
    let mut d = idx(2, 100, 8, 2);
    let seq: Vec<u32> = (0..2).flat_map(|_| [1u32, 2, 3, 4, 5, 6]).collect();
    d.feed(&seq);
    assert_eq!(d.draft(&[1, 2], 100), vec![3, 4], "max_draft=2 で切れる");
    assert_eq!(d.draft(&[1, 2], 1), vec![3]);

    // max_draft=0 は「何も引けない辞書」(draft は常に空)。
    let mut zero = idx(2, 100, 8, 0);
    zero.feed(&seq);
    assert!(zero.draft(&[1, 2], 100).is_empty());
    assert!(offer_draft(&zero, &[1], 2, 4, &[0.0f32; 8], 0.0).is_empty());
}

#[test]
fn offer_draft_k_zero_returns_empty_without_touching_min_p() {
    let mut d = idx(3, 100, 8, 4);
    d.feed(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]);
    let ctx = [1u32, 2, 3, 4, 1, 2, 3];
    let logits = [0.0f32; 8];
    assert!(offer_draft(&d, &ctx, 4, 0, &logits, 0.0).is_empty());
    assert!(offer_draft(&d, &ctx, 4, 0, &logits, 1.0).is_empty());
}

#[test]
fn default_index_uses_documented_limits() {
    let mut d = NgramIndex::default();
    assert!(d.is_empty());
    // Default は max_draft=4 (継続は 4 トークンで切れる)。
    let seq: Vec<u32> = (0..2).flat_map(|_| [1u32, 2, 3, 4, 5, 6]).collect();
    d.feed(&seq);
    assert_eq!(d.draft(&[1, 2], 100), vec![3, 4, 5, 6]);
    // max_suffix=3: 2-gram まで登録され 3-gram はされない。
    assert_eq!(d.draft(&[1, 2, 3], 100), vec![4, 5, 6]);
    assert!(d.draft(&[6, 1], 100).len() <= 4);
}

// ------------------------------------------------------------ min_p の端

/// offer_draft 用の固定辞書: 履歴 [1,2,3,4] の繰り返し、q0=4 で draft は [1,2,3]。
fn min_p_fixture() -> NgramIndex {
    let mut d = idx(3, 100, 8, 4);
    d.feed(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]);
    d
}

const MIN_P_CTX: [u32; 7] = [1, 2, 3, 4, 1, 2, 3];

#[test]
fn min_p_zero_and_negative_disable_filter() {
    let d = min_p_fixture();
    // p(1) = e^-5 / (7 + e^-5) ≈ 9.6e-4 — 閾値 0 / 負なら落とさない。
    let mut logits = [0.0f32; 8];
    logits[1] = -5.0;
    assert_eq!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &logits, 0.0),
        vec![1, 2, 3]
    );
    assert_eq!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &logits, -0.5),
        vec![1, 2, 3],
        "負の min_p は無効扱い"
    );
    assert_eq!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &logits, 0.0009),
        vec![1, 2, 3],
        "確率が閾値以上なら出す"
    );
    assert!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &logits, 0.5).is_empty(),
        "確率が閾値未満なら出さない"
    );
}

#[test]
fn min_p_one_only_passes_exact_certainty() {
    let d = min_p_fixture();
    // 一様 logits では p(1) = 1/8 < 1.0 なので落ちる。
    let flat = [0.0f32; 8];
    assert!(offer_draft(&d, &MIN_P_CTX, 4, 4, &flat, 1.0).is_empty());
    // draft[0]=1 を唯一の非 -inf にして p(1) を厳密に 1.0 にする。
    // 比較は `p < min_p` なので 1.0 は通る (境界は包含)。
    let mut sharp = [f32::NEG_INFINITY; 8];
    sharp[1] = 0.0;
    assert_eq!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &sharp, 1.0),
        vec![1, 2, 3],
        "p=1.0 は min_p=1.0 を通る"
    );
    // 1.0 の直上 (f32 の次の値) は落ちる。
    let just_above_one = f32::from_bits(1.0f32.to_bits() + 1);
    assert!(just_above_one > 1.0);
    assert!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &sharp, just_above_one).is_empty(),
        "p=1.0 < 1.0+ε なので落ちる"
    );
    // +inf も同じ (常に落ちる)。
    assert!(offer_draft(&d, &MIN_P_CTX, 4, 4, &sharp, f32::INFINITY).is_empty());
}

#[test]
fn min_p_nan_behaves_as_disabled_filter() {
    let d = min_p_fixture();
    let mut logits = [0.0f32; 8];
    logits[1] = -5.0; // わざと低確率にする
    // `min_p <= 0.0` は NaN で false、`p < NaN` も false → フィルタは掛からない。
    let out = offer_draft(&d, &MIN_P_CTX, 4, 4, &logits, f32::NAN);
    assert_eq!(out, vec![1, 2, 3], "NaN はフィルタ無効として扱う");
}

#[test]
fn min_p_with_masked_and_short_logits() {
    let d = min_p_fixture();
    // draft[0]=1 が -inf (マスク済み) → p(1)=0。閾値 0 は無効なので出る。
    let mut masked = [0.0f32; 8];
    masked[1] = f32::NEG_INFINITY;
    assert_eq!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &masked, 0.0),
        vec![1, 2, 3],
        "min_p=0 は確率を見ない"
    );
    assert!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &masked, 0.5).is_empty(),
        "p=0 は正の閾値で落ちる"
    );

    // logits 長が draft[0] 以下なら確率を計算せず draft をそのまま返す。
    let short = [0.0f32; 1];
    assert_eq!(
        offer_draft(&d, &MIN_P_CTX, 4, 4, &short, 0.9),
        vec![1, 2, 3],
        "draft[0] が logits 範囲外なら素通し"
    );
    let empty: [f32; 0] = [];
    assert_eq!(offer_draft(&d, &MIN_P_CTX, 4, 4, &empty, 0.9), vec![1, 2, 3]);
}

#[test]
fn min_p_filter_does_not_mutate_dictionary() {
    let d = min_p_fixture();
    let (len, fed) = (d.len(), d.fed());
    let flat = [0.0f32; 8];
    let _ = offer_draft(&d, &MIN_P_CTX, 4, 4, &flat, 1.0); // 落ちる側
    let _ = offer_draft(&d, &MIN_P_CTX, 4, 4, &flat, 0.0); // 出る側
    assert_eq!(d.len(), len, "フィルタはエントリを増減しない");
    assert_eq!(d.fed(), fed, "フィルタは fed を動かさない");
    assert_eq!(d.draft(&ctx_with_q0(), 4), vec![1, 2, 3]);
}

/// `MIN_P_CTX` に q0=4 を足した列 (draft の直接呼び出し用)。
fn ctx_with_q0() -> Vec<u32> {
    let mut v = MIN_P_CTX.to_vec();
    v.push(4);
    v
}
