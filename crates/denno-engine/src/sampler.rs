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

/// s56 (T2): frequency / presence penalty の係数を組にする (OpenAI 互換)。
///
/// - `frequency_penalty` f: 文脈中の**出現回数** × f を logit から減算する
/// - `presence_penalty` p: 文脈中に**1 回以上**現れたトークンから p を 1 回減算する
///
/// いずれも A2.5 の `repetition_penalty` (正負で除算/乗算を切り替える HF 規則)
/// とは**独立・別物**で、単純減算のみ。適用順は
/// 「生 logits → repetition_penalty → frequency → presence → no_repeat_ngram
///  → temperature → top-k → top-p → sample」(docs/api.md にも明記)。
///
/// 既定 (`0.0`) は両方とも完全な no-op (加算もしない = ビット不変)。
/// RNG 不使用。語彙外の id は無視する。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FreqPresencePenalty {
    pub frequency: f32,
    pub presence: f32,
}

impl FreqPresencePenalty {
    /// 指定値から作る。None は 0.0 扱い。どちらも 0.0 なら no-op 経路。
    pub fn new(frequency: Option<f32>, presence: Option<f32>) -> Self {
        Self {
            frequency: frequency.unwrap_or(0.0),
            presence: presence.unwrap_or(0.0),
        }
    }

    /// 両方 0.0 (= 無効) か。呼び出し側はこれで適用自体を省ける。
    #[inline]
    pub fn is_noop(&self) -> bool {
        self.frequency == 0.0 && self.presence == 0.0
    }
}

/// frequency / presence penalty を適用する (s56)。係数 0 はビット完全 no-op。
///
/// `counts[t]` = 文脈中の t の出現回数。`logits[t] -= f * counts[t] + p * (counts[t] > 0)`。
/// 減算は 32bit float の素の減算で、係数が 0 の項は `+=`/`-=` を一切行わない
/// (0 を引いても -0.0 の符号が変わり得るため、無効時に触らないことを保証する)。
pub fn apply_freq_presence_penalty(logits: &mut [f32], counts: &[u32], p: &FreqPresencePenalty) {
    if p.is_noop() {
        return;
    }
    let n = logits.len().min(counts.len());
    for (l, &c) in logits.iter_mut().zip(counts.iter()).take(n) {
        if c == 0 {
            continue;
        }
        let mut sub = 0.0f32;
        if p.frequency != 0.0 {
            sub += p.frequency * c as f32;
        }
        if p.presence != 0.0 {
            sub += p.presence;
        }
        *l -= sub;
    }
}

/// 文脈の出現回数表を作る (語彙サイズ `vocab`。語彙外 id は数えない)。
pub fn context_counts(context: &[u32], vocab: usize) -> Vec<u32> {
    let mut counts = vec![0u32; vocab];
    for &t in context {
        let t = t as usize;
        if t < vocab {
            counts[t] += 1;
        }
    }
    counts
}

/// no_repeat_ngram (s56): 直近文脈に既出の n-gram について、その続き (= 最終
/// トークン) を禁止する (-inf)。llama.cpp / HF `NoRepeatNGramLogitsProcessor` 準拠。
///
/// - 文脈長 < n なら no-op (n-gram が 1 つも取れない)
/// - `n <= 1` も no-op (`n == 1` の「長さ 1 の n-gram の続き」は直前トークン自身を
///   禁じる意味になるが、HF 実装も `n <= 1` では生成しない。ここでは「n-gram の
///   接頭辞が空で常に一致する」という退化を避けるため明示的に no-op とする)
/// - 禁止するのは「既出の (n-1)-gram 接頭辞に一致する文脈末尾」に対する続きの
///   トークンのみ (文脈全体を走査し、末尾 (n-1) トークンのパターンに一致する
///   各位置の次トークンを集める)
/// - `-inf` を書くので softmax で確率 0 になる (temperature/top-k/top-p の前に適用)
/// - RNG 不使用
///
/// 文脈 (prompt + 生成済み) をそのまま渡す。語彙外の id は無視する。
pub fn apply_no_repeat_ngram(logits: &mut [f32], context: &[u32], n: usize) {
    if n <= 1 || context.len() < n {
        return;
    }
    // 接頭辞 = context[len-n+1 .. len] (長さ n-1) に一致する全位置の続きを禁止。
    let prefix = &context[context.len() - (n - 1)..];
    let mut banned: Vec<usize> = Vec::new();
    // i は接頭辞の開始位置。続きは context[i + n - 1]。
    let last_start = context.len() - n; // [i, i+n) が文脈に収まる最大 i
    for i in 0..=last_start {
        if &context[i..i + n - 1] == prefix {
            let follow = context[i + n - 1] as usize;
            if follow < logits.len() {
                banned.push(follow);
            }
        }
    }
    for t in banned {
        logits[t] = f32::NEG_INFINITY;
    }
}

/// s56 (T2): 反復抑止 3 種 + 既存 repetition_penalty を **1 か所で固定順**に適用する。
///
/// 適用順 (docs/api.md と同一。変えるとビット契約が壊れる):
/// ```text
/// 生 logits
///   → ① repetition_penalty   (A2.5 の既存処理・意味/位置は不変)
///   → ② frequency_penalty    (出現回数 × 係数の減算)
///   → ③ presence_penalty     (出現有無につき係数を 1 回減算)
///   → ④ no_repeat_ngram_size (既出 n-gram の続きを -inf)
///   → temperature → top-k → top-p → sample
/// ```
///
/// ①〜④ がすべて無効 (None / 0 / n<=1) のときは logits に一切触らない
/// = 呼ぶ前と 1 bit も変わらない。逐次 decode (`step_chat`) とバッチ decode
/// (`step_chat_batch`) は **この関数だけ**を通す (片経路だけに入れない)。
/// RNG を一切使わない (決定論契約は不変)。
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AntiRepetition {
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub no_repeat_ngram_size: Option<usize>,
}

impl AntiRepetition {
    /// 4 つすべて無効なら true。呼び出し側は適用自体を省ける
    /// (context の構築・counts の確保すらしない = 既定経路のコストも不変)。
    #[inline]
    pub fn is_noop(&self) -> bool {
        self.repetition_penalty.map_or(true, |p| p == 1.0)
            && self.frequency_penalty.map_or(true, |f| f == 0.0)
            && self.presence_penalty.map_or(true, |p| p == 0.0)
            && self.no_repeat_ngram_size.map_or(true, |n| n <= 1)
    }

    /// `logits` (生・語彙長) と文脈 (prompt_ids + generated) に対して固定順に適用する。
    pub fn apply(&self, logits: &mut [f32], context: &[u32]) {
        if self.is_noop() {
            return;
        }
        // ① 既存 repetition_penalty (意味・位置を変えない — HF 互換の除算/乗算)
        if let Some(p) = self.repetition_penalty {
            apply_repetition_penalty(logits, context, p);
        }
        // ②③ frequency → presence
        let fp = FreqPresencePenalty::new(self.frequency_penalty, self.presence_penalty);
        if !fp.is_noop() {
            let counts = context_counts(context, logits.len());
            apply_freq_presence_penalty(logits, &counts, &fp);
        }
        // ④ no_repeat_ngram (-inf マスクは最後 = 前段の減算結果に依存しない)
        if let Some(n) = self.no_repeat_ngram_size {
            apply_no_repeat_ngram(logits, context, n);
        }
    }

    /// `n <= 1` の no_repeat_ngram_size は「無効」として正規化する
    /// (受理時点の検証は server 側。ここは内部の防御)。
    pub fn normalized(mut self) -> Self {
        if let Some(n) = self.no_repeat_ngram_size {
            if n <= 1 {
                self.no_repeat_ngram_size = None;
            }
        }
        self
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

    // ---------- s56 (T2): frequency / presence / no_repeat_ngram ----------

    fn bits_of(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// 代表的な「厄介な」logits: 0.0 / -0.0 / 負 / 非正規化 / 大きな正負。
    fn tricky_logits(n: usize) -> Vec<f32> {
        let mut v: Vec<f32> = (0..n)
            .map(|i| match i % 6 {
                0 => 0.0,
                1 => -0.0,
                2 => -(i as f32) * 0.125,
                3 => f32::MIN_POSITIVE / 4.0,
                4 => i as f32 * 0.37,
                _ => 1e30_f32.recip() * -1.0,
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

    /// `is_noop()` の契約: 「None / 1.0 / 0.0 / n<=1」の任意の組み合わせで 4 つとも
    /// 無効と判定し、1 つでも有効な値があれば false。呼び出し側 (engine / CLI) は
    /// これを見て **文脈構築・counts 確保を完全に省く** ので、既定経路のコストと
    /// ビットの両方を守るのはこの判定。
    ///
    /// なお `apply()` 内の同じガードは「備え」で、仮に外しても下位の各関数が
    /// 値 0 / 1.0 / n<=1 で何もしないためビットは変わらない (変異テストで確認済み)。
    #[test]
    fn anti_repetition_is_noop_covers_every_disabled_combination() {
        let dis = |p: Option<f32>| p.map_or(true, |v| v == 0.0);
        let dis_rep = |p: Option<f32>| p.map_or(true, |v| v == 1.0);
        // 全 2^4 通りの「無効値」組み合わせで no-op と判定されること
        for bits in 0u32..16 {
            let pick = |b: u32, on: Option<f32>, off: Option<f32>| if bits & b != 0 { on } else { off };
            let a = AntiRepetition {
                repetition_penalty: pick(1, Some(1.0), None),
                frequency_penalty: pick(2, Some(0.0), None),
                presence_penalty: pick(4, Some(-0.0), None),
                no_repeat_ngram_size: if bits & 8 != 0 { Some(1) } else { None },
            };
            assert!(a.is_noop(), "無効値の組み合わせが no-op でない: bits={bits} {a:?}");
            assert!(dis(a.frequency_penalty) && dis(a.presence_penalty) && dis_rep(a.repetition_penalty));
        }
        // どれか 1 つでも有効値なら no-op ではない
        for a in [
            AntiRepetition { repetition_penalty: Some(1.1), ..Default::default() },
            AntiRepetition { frequency_penalty: Some(0.01), ..Default::default() },
            AntiRepetition { frequency_penalty: Some(-0.01), ..Default::default() },
            AntiRepetition { presence_penalty: Some(0.01), ..Default::default() },
            AntiRepetition { no_repeat_ngram_size: Some(2), ..Default::default() },
        ] {
            assert!(!a.is_noop(), "有効値なのに no-op 扱い: {a:?}");
        }
    }
    #[test]
    fn anti_repetition_default_is_bit_identical_noop() {
        let base = tricky_logits(64);
        // 文脈は語彙内の全 id を 2 回ずつ (負のゼロを含む -0.0 の位置も必ず入る)
        let ctx: Vec<u32> = (0..64u32).chain(0..64).collect();
        assert!(
            base.iter().any(|x| x.to_bits() == (-0.0f32).to_bits()),
            "前提: テスト logits に -0.0 が含まれる (符号反転を検出可能にする)"
        );
        let a = AntiRepetition::default();
        assert!(a.is_noop(), "既定は no-op と認識されること");
        let mut x = base.clone();
        a.apply(&mut x, &ctx);
        assert_eq!(bits_of(&x), bits_of(&base), "既定で logits のビットが変わった");
    }

    /// 明示的な 0.0 / 1.0 / n<=1 も「無効」= ビット完全 no-op (既定と同じ)。
    #[test]
    fn anti_repetition_explicit_zeros_are_bit_identical_noop() {
        let base = tricky_logits(64);
        let ctx: Vec<u32> = (0..64u32).chain(0..64).collect(); // 重複あり
        for a in [
            AntiRepetition { repetition_penalty: Some(1.0), frequency_penalty: Some(0.0), presence_penalty: Some(-0.0), no_repeat_ngram_size: None },
            AntiRepetition { repetition_penalty: None, frequency_penalty: Some(0.0), presence_penalty: Some(0.0), no_repeat_ngram_size: Some(1) },
            AntiRepetition { repetition_penalty: Some(1.0), frequency_penalty: None, presence_penalty: None, no_repeat_ngram_size: Some(0) },
        ] {
            assert!(a.is_noop(), "無効値の組が no-op と認識されない: {a:?}");
            let mut x = base.clone();
            a.apply(&mut x, &ctx);
            assert_eq!(bits_of(&x), bits_of(&base), "無効値でビットが変わった: {a:?}");
        }
    }

    /// 既定の逐次サンプリング (seed 固定) は AntiRepetition を挟んでも同一系列
    /// (greedy / temperature / top-k / top-p の全経路)。
    #[test]
    fn anti_repetition_default_keeps_sampling_sequence() {
        let logits: Vec<f32> = (0..300).map(|i| ((i * 37) % 101) as f32 * 0.03 - 1.0).collect();
        for (t, k, p) in [
            (Some(0.0), None, None),           // greedy
            (Some(1.0), None, None),           // plain
            (Some(0.7), Some(20), None),       // top-k
            (Some(0.9), None, Some(0.95)),     // top-p
            (Some(0.8), Some(50), Some(0.9)),  // top-k + top-p
        ] {
            let mut a = sampler(t, k, p, Some(42));
            let mut b = sampler(t, k, p, Some(42));
            let anti = AntiRepetition::default();
            let seq_a: Vec<usize> = (0..30).map(|_| a.sample(&logits)).collect();
            let seq_b: Vec<usize> = (0..30)
                .map(|_| {
                    let mut l = logits.clone();
                    anti.apply(&mut l, &[1, 2, 3]);
                    b.sample(&l)
                })
                .collect();
            assert_eq!(seq_a, seq_b, "既定 no-op で系列が変わった: t={t:?} k={k:?} p={p:?}");
        }
    }

    /// frequency: 出現回数 × 係数を減算 (手計算値)。重複は回数として数える。
    #[test]
    fn frequency_penalty_arithmetic() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        // 文脈: t0 が 3 回・t2 が 1 回 (t1/t3 は未出)
        let ctx = [0u32, 0, 0, 2];
        let counts = context_counts(&ctx, logits.len());
        assert_eq!(counts, vec![3, 0, 1, 0]);
        apply_freq_presence_penalty(
            &mut logits,
            &counts,
            &FreqPresencePenalty { frequency: 0.5, presence: 0.0 },
        );
        assert_eq!(logits, vec![1.0 - 1.5, 2.0, 3.0 - 0.5, 4.0], "3回×0.5 / 1回×0.5");
    }

    /// presence: 出現有無につき 1 回だけ減算 (回数に依存しない)。
    #[test]
    fn presence_penalty_arithmetic() {
        let mut logits = vec![1.0f32, 2.0, 3.0];
        let ctx = [0u32, 0, 0, 0, 0]; // t0 は 5 回出現
        let counts = context_counts(&ctx, logits.len());
        apply_freq_presence_penalty(
            &mut logits,
            &counts,
            &FreqPresencePenalty { frequency: 0.0, presence: 0.25 },
        );
        assert_eq!(logits, vec![0.75, 2.0, 3.0], "5 回でも減算は 1 回だけ");
    }

    /// frequency + presence 併用: -(f*count + p) を一度に減算 (同一トークン)。
    /// 負の係数もそのまま (加算方向)。
    #[test]
    fn frequency_and_presence_combine_and_allow_negative() {
        let mut logits = vec![0.0f32, 0.0];
        let counts = context_counts(&[1u32, 1, 1], 2); // t1: 3 回
        apply_freq_presence_penalty(
            &mut logits,
            &counts,
            &FreqPresencePenalty { frequency: 0.1, presence: 0.3 },
        );
        // -(0.1*3 + 0.3) = -0.6。0.1*3.0 = 0.30000001192... の f32 加算後の減算値。
        let expect = 0.0f32 - (0.1f32 * 3.0f32 + 0.3f32);
        assert_eq!(logits[1].to_bits(), expect.to_bits(), "0.0-(0.1*3+0.3)");
        assert_eq!(logits[0], 0.0, "未出トークンは不変");
        let mut l2 = vec![0.0f32];
        apply_freq_presence_penalty(&mut l2, &counts[1..2], &FreqPresencePenalty { frequency: 0.0, presence: -0.5 });
        assert_eq!(l2[0], 0.5, "負の presence は加算");
    }

    /// 語彙外 id は counts に入らず、logits も触らない (u32::MAX でも安全)。
    #[test]
    fn freq_presence_out_of_vocab_ignored() {
        let mut logits = vec![5.0f32, 6.0];
        let counts = context_counts(&[7u32, u32::MAX], 2);
        assert_eq!(counts, vec![0, 0]);
        apply_freq_presence_penalty(&mut logits, &counts, &FreqPresencePenalty { frequency: 1.0, presence: 1.0 });
        assert_eq!(logits, vec![5.0, 6.0]);
    }

    /// no_repeat_ngram: 既出 n-gram の続きだけが -inf になり、他は不変。
    #[test]
    fn no_repeat_ngram_bans_seen_continuation() {
        // 文脈 [1,2,3, 5, 1,2,3, 5, 1,2,3] / n=3 → 末尾接頭辞 = (2,3)
        //   i=1: (2,3,5) ✓ 続き 5
        //   i=5: (2,3,5) ✓ 続き 5
        //   i=7: (2,3) は末尾そのもの (i + n > len) → 続きを作らない
        // → 5 だけ禁止。1/2/3/4 は不変。
        let mut logits = vec![0.0f32; 6];
        let ctx = [1u32, 2, 3, 5, 1, 2, 3, 5, 1, 2, 3];
        apply_no_repeat_ngram(&mut logits, &ctx, 3);
        assert_eq!(logits[5], f32::NEG_INFINITY, "(2,3) の続き 5 を禁止");
        for (i, &l) in logits.iter().enumerate() {
            if i != 5 {
                assert_eq!(l, 0.0, "禁止対象外 {i} は不変");
            }
        }
        // 禁止対象が無い文脈は完全 no-op (境界: 末尾 n-gram は過去の接頭辞と
        // 一致しない)
        let mut x = vec![0.0f32; 6];
        let base = x.clone();
        apply_no_repeat_ngram(&mut x, &[1u32, 2, 3, 1, 2, 4], 3);
        assert_eq!(bits_of(&x), bits_of(&base), "末尾接頭辞 (2,4) に続きは無い");
    }

    /// 複数の続きが禁止される (接頭辞が 2 回以上・続きが異なる場合)。
    #[test]
    fn no_repeat_ngram_bans_all_seen_continuations() {
        // 文脈 [1,2,3, 5, 1,2,4, 1,2] (末尾接頭辞 = (1,2))
        // (1,2) の続き = 3 と 4 → 両方禁止。末尾自身は続きを持たない (文脈外)。
        let mut logits = vec![0.0f32; 8];
        let ctx = [1u32, 2, 3, 5, 1, 2, 4, 1, 2];
        apply_no_repeat_ngram(&mut logits, &ctx, 3);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(logits[4], f32::NEG_INFINITY);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], 0.0);
    }

    /// 文脈長 < n は no-op / n <= 1 は no-op (退化を禁止)。
    #[test]
    fn no_repeat_ngram_short_context_and_small_n_are_noop() {
        let base = tricky_logits(16);
        for n in [0usize, 1] {
            let mut x = base.clone();
            apply_no_repeat_ngram(&mut x, &[1, 2, 3, 1, 2], n);
            assert_eq!(bits_of(&x), bits_of(&base), "n={n} は no-op");
        }
        // 文脈 2 トークン・n=3 → no-op
        let mut x = base.clone();
        apply_no_repeat_ngram(&mut x, &[1, 2], 3);
        assert_eq!(bits_of(&x), bits_of(&base), "文脈長 < n は no-op");
        // ちょうど文脈長 == n: 1 個の n-gram しか取れず、その接頭辞は文脈内に
        // 「続きを持つ位置」として現れない (末尾自身) → no-op
        let mut x = base.clone();
        apply_no_repeat_ngram(&mut x, &[1, 2, 3], 3);
        assert_eq!(bits_of(&x), bits_of(&base), "文脈長 == n は no-op");
    }

    /// 文脈末尾境界: 直前 1 トークンだけで決まる n=2 の挙動
    /// (「直前トークン a の次に来たトークン」を全部禁止)。
    #[test]
    fn no_repeat_ngram_boundary_n2() {
        let mut logits = vec![0.0f32; 6];
        // 4 の続き: 位置 1 の次は 5、位置 3 は末尾そのもの → 5 を禁止
        let ctx = [7u32, 4, 5, 4];
        apply_no_repeat_ngram(&mut logits, &ctx, 2);
        assert_eq!(logits[5], f32::NEG_INFINITY, "4 の次の 5 を禁止");
        for (i, &l) in logits.iter().enumerate() {
            if i != 5 {
                assert_eq!(l, 0.0, "禁止対象外 {i} は不変");
            }
        }
        assert_eq!(logits[4], 0.0, "token 4 自体 (直前トークン) は禁止しない");
        // 別の文脈 [1,2,3]: 末尾トークン 3 は 1 回しか現れず続きが無い → no-op。
        // 一方 [1,2,1]: 末尾 1 の過去の続きは 2 → 2 を禁止 (直前トークンの
        // 出現位置そのものが末尾だけ = 続き無し、の境界)。
        let mut x = vec![0.0f32; 5];
        let xb = x.clone();
        apply_no_repeat_ngram(&mut x, &[1, 2, 3], 2);
        assert_eq!(bits_of(&x), bits_of(&xb), "末尾トークンの続きは存在しない");
        let mut y = vec![0.0f32; 5];
        apply_no_repeat_ngram(&mut y, &[1, 2, 1], 2);
        assert_eq!(y[2], f32::NEG_INFINITY, "末尾 1 の過去の続き 2 を禁止");
        assert_eq!(y[1], 0.0);
    }

    /// 適用順: repetition_penalty → frequency → presence → no_repeat_ngram。
    /// 順序を入れ替えると値が変わる入力で、固定順の結果を手計算で固定する。
    #[test]
    fn anti_repetition_applies_in_fixed_order() {
        // 文脈 [1, 1, 2] / logits: t0=1.0 t1=4.0 t2=2.0 t3=0.5
        // ① repetition 2.0: t1(4.0>0)→2.0, t2(2.0>0)→1.0 (t0 未出)
        // ② frequency 0.5: t1 は 2 回 → -1.0 → 1.0 / t2 は 1 回 → -0.5 → 0.5
        // ③ presence 0.25: t1 → 0.75 / t2 → 0.25
        // ④ n_repeat 3: 接頭辞 (1,2) の過去の続きは無い → 何も禁止しない
        let mut logits = vec![1.0f32, 4.0, 2.0, 0.5];
        let ctx = [1u32, 1, 2];
        let anti = AntiRepetition {
            repetition_penalty: Some(2.0),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.25),
            no_repeat_ngram_size: Some(3),
        };
        anti.apply(&mut logits, &ctx);
        assert_eq!(logits[0], 1.0, "文脈外は不変");
        assert_eq!(logits[1], 0.75, "4/2 - 2*0.5 - 0.25");
        assert_eq!(logits[2], 0.25, "2/2 - 1*0.5 - 0.25");
        assert_eq!(logits[3], 0.5, "文脈外は不変");
        // 逆順 (frequency/presence → repetition) では別の値になること:
        // (4 - 1.25)/2 = 1.375 ≠ 0.75 → 順序が結果に効いている裏取り。
        let mut swapped = vec![1.0f32, 4.0, 2.0, 0.5];
        apply_freq_presence_penalty(
            &mut swapped,
            &context_counts(&ctx, 4),
            &FreqPresencePenalty { frequency: 0.5, presence: 0.25 },
        );
        apply_repetition_penalty(&mut swapped, &ctx, 2.0);
        assert_eq!(swapped[1], (4.0f32 - 1.25) / 2.0);
        assert_ne!(swapped[1], logits[1], "順序を変えても同じ値 = テストが順序を検証していない");
    }

    /// 適用順 (② frequency → ③ presence) を固定する。f32 では減算の順序が
    /// 結果のビットを変える入力が実在するため、順序を入れ替えるとこのテストが落ちる。
    ///
    /// logit 0.3 / frequency 0.01 (count 1) / presence 0.05 のとき:
    ///   frequency → presence (2 段階): (0.3 - 0.01) - 0.05 = 0x1.ed999e...p-3
    ///   presence → frequency (2 段階): (0.3 - 0.05) - 0.01 = 0x1.eb851e...p-3
    ///   (どちらとも異なる)
    /// 現実装は「合計を 1 回だけ減算」する: 0.3 - (0.01 + 0.05)。
    #[test]
    fn frequency_and_presence_order_is_observable_in_f32() {
        let counts = [1u32]; // t0 は 1 回出現
        let mut x = [0.3f32];
        apply_freq_presence_penalty(
            &mut x,
            &counts,
            &FreqPresencePenalty { frequency: 0.01, presence: 0.05 },
        );
        let freq_then_presence = (0.3f32 - 0.01f32) - 0.05f32;
        let presence_then_freq = (0.3f32 - 0.05f32) - 0.01f32;
        let one_subtraction = 0.3f32 - (0.01f32 + 0.05f32);
        // 前提: 3 つが互いに異なるビットであること (そうでなければ順序を検出できない)
        assert_ne!(freq_then_presence.to_bits(), presence_then_freq.to_bits());
        assert_ne!(one_subtraction.to_bits(), freq_then_presence.to_bits());
        assert_ne!(one_subtraction.to_bits(), presence_then_freq.to_bits());
        // 実装は合計 1 回減算 (② の減算 → ③ の減算を別々にやらない)
        assert_eq!(x[0].to_bits(), one_subtraction.to_bits(), "合計 1 回の減算になっていない");
        assert_ne!(x[0].to_bits(), presence_then_freq.to_bits(), "presence が先に適用されている");
    }

    /// no_repeat_ngram は前段の減算結果を上書きする (-inf は最後)。
    /// 文脈 [1,2,3,1,2] n=3 → 接頭辞 (1,2) の続き 3 を禁止。
    #[test]
    fn no_repeat_ngram_masks_after_penalties() {
        let mut logits = vec![0.0f32, 0.0, 0.0, 5.0];
        let ctx = [1u32, 2, 3, 1, 2];
        let anti = AntiRepetition {
            repetition_penalty: Some(1.5),
            frequency_penalty: Some(0.1),
            presence_penalty: Some(0.1),
            no_repeat_ngram_size: Some(3),
        };
        anti.apply(&mut logits, &ctx);
        assert_eq!(logits[3], f32::NEG_INFINITY, "penalty 適用後に -inf で確定");
        assert!(logits[1] < 0.0, "t1 は repetition で 0/1.5=0 → freq/presence で負");
    }

    /// 決定論: 同一入力 2 回で同一ビット・RNG 不関与 (呼んでもサンプラ列が変わらない)。
    #[test]
    fn anti_repetition_is_deterministic_and_rng_free() {
        let base = tricky_logits(32);
        let ctx: Vec<u32> = (0..40u32).map(|i| (i * 7) % 32).collect();
        let anti = AntiRepetition {
            repetition_penalty: Some(1.2),
            frequency_penalty: Some(0.3),
            presence_penalty: Some(0.2),
            no_repeat_ngram_size: Some(4),
        };
        let mut a = base.clone();
        let mut b = base.clone();
        anti.apply(&mut a, &ctx);
        anti.apply(&mut b, &ctx);
        assert_eq!(bits_of(&a), bits_of(&b), "2 回で不一致");

        // RNG 不関与: 同じ seed のサンプラで penalty 適用の有無が RNG 消費に
        // 影響しない (適用後の logits をそのまま与えた場合と、適用してから
        // 与えた場合で系列が一致する = 次の RNG 状態が同じ)。
        let logits: Vec<f32> = (0..200).map(|i| ((i * 13) % 71) as f32 * 0.05).collect();
        let mut s1 = sampler(Some(1.0), None, None, Some(7));
        let mut s2 = sampler(Some(1.0), None, None, Some(7));
        let mut l = logits.clone();
        anti.apply(&mut l, &ctx);
        let seq1: Vec<usize> = (0..20).map(|_| s1.sample(&l)).collect();
        let seq2: Vec<usize> = (0..20)
            .map(|_| {
                let mut x = logits.clone();
                anti.apply(&mut x, &ctx);
                s2.sample(&x)
            })
            .collect();
        assert_eq!(seq1, seq2, "penalty 適用が RNG 消費を変えた");
    }

    /// normalized(): n<=1 を無効に落とす (内部防御)。
    #[test]
    fn anti_repetition_normalized_drops_small_n() {
        for n in [0usize, 1] {
            let a = AntiRepetition { no_repeat_ngram_size: Some(n), ..Default::default() };
            assert_eq!(a.normalized().no_repeat_ngram_size, None);
        }
        let a = AntiRepetition { no_repeat_ngram_size: Some(2), ..Default::default() };
        assert_eq!(a.normalized().no_repeat_ngram_size, Some(2));
        assert!(!a.is_noop());
    }
}
