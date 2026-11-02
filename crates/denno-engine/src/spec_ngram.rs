//! P109: n-gram 投機的デコード (prompt-lookup) 用の辞書。
//!
//! コンテキスト + 生成済みトークンから「末尾の suffix で始まる継続」を引く。
//! トークンは [`u32`] で扱う (日本語のマルチバイト文字がトークン境界で
//! 分断されても辞書は壊れない — テキストには触らない)。
//!
//! 登録は「位置 i の suffix (長さ 1..=max_suffix) → 位置 i からの継続列
//! (最大 [`Self::max_draft`] トークン)」。同じキーに同じ継続が再度出現したら
//! 末尾へ移動する (最新の継続ほど draft に採用されやすい)。
//!
//! 上限: エントリ (キー) 数 [`Self::max_entries`] を超えたら**挿入順**で
//! 先頭から落とす (FIFO 駆逐。キーの再出現で順序を更新しない —
//! decode ステップごとに 3 回の登録しか起きないので O(1) を保つ)。
//! 1 キーの候補数は [`Self::cand_keep`] まで (古いものから落とす)。

use std::collections::{HashMap, VecDeque};

/// suffix の最長 (既定 3。PLE の ngram_size と同じ値に合わせた)。
pub const DEFAULT_MAX_SUFFIX: usize = 3;
/// エントリ (キー) 数の上限 (既定 10 万。タスクの指示値)。
pub const DEFAULT_MAX_ENTRIES: usize = 100_000;
/// 1 キーあたり保持する継続候補の数。
pub const DEFAULT_CAND_KEEP: usize = 8;

/// n-gram 辞書 (コンテキスト + 生成済みトークンの suffix → 継続)。
#[derive(Debug)]
pub struct NgramIndex {
    /// suffix (古い順) → 継続候補 (新しいほど末尾)。
    map: HashMap<Box<[u32]>, VecDeque<Box<[u32]>>>,
    /// 挿入順のキー列 (駆逐は先頭から)。
    order: VecDeque<Box<[u32]>>,
    /// suffix の最長。
    max_suffix: usize,
    /// エントリ数の上限。
    max_entries: usize,
    /// 1 キーの候補保持数。
    cand_keep: usize,
    /// 継続列の最大長 (draft の最大長)。
    max_draft: usize,
    /// feed 済みトークン数 (差分 feed の重複登録を防ぐ)。
    fed: usize,
}

impl Default for NgramIndex {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_SUFFIX, DEFAULT_MAX_ENTRIES, DEFAULT_CAND_KEEP, 4)
    }
}

impl NgramIndex {
    /// 上限を明示して作る。`max_draft` は継続列の最大長 (draft の k)。
    ///
    /// `max_suffix == 0` / `max_entries == 0` / `max_draft == 0` はエラーではなく
    /// 「何も引けない辞書」になる (draft は常に空 = 通常 decode にフォールバック)。
    pub fn new(max_suffix: usize, max_entries: usize, cand_keep: usize, max_draft: usize) -> Self {
        NgramIndex {
            map: HashMap::new(),
            order: VecDeque::new(),
            max_suffix,
            max_entries,
            cand_keep,
            max_draft,
            fed: 0,
        }
    }

    /// 現在のエントリ (キー) 数。
    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// feed 済みトークン数 (テスト・診断用)。
    pub fn fed(&self) -> usize {
        self.fed
    }

    /// トークン列を登録する。
    ///
    /// 初回は全体を、以降は追加分だけを渡すのを想定する。`toks` が既に
    /// feed 済み (長さ `<= self.fed()`) なら何もしない。**途中を飛ばした feed**
    /// (前回より短い列から長い列への切り替え) も安全側で全体を登録し直す
    /// (重複登録は「末尾への移動」になるだけなので辞書の意味は変わらない)。
    pub fn feed(&mut self, toks: &[u32]) {
        // 差分の起点。縮んだら全体を見直す (fed は末尾へ進むだけ)。
        let start = if toks.len() >= self.fed { self.fed } else { 0 };
        for i in start..toks.len() {
            let n = self.max_suffix.min(i);
            // suffix の長さが長い順に登録 (draft は最長一致から引くため)。
            for s in 1..=n {
                let key = toks[i - s..i].to_vec().into_boxed_slice();
                let cont = toks[i..(i + self.max_draft).min(toks.len())].to_vec().into_boxed_slice();
                self.insert(key, cont);
            }
        }
        self.fed = self.fed.max(toks.len());
    }

    /// キーに継続候補を 1 件登録する (重複は末尾へ移動)。
    fn insert(&mut self, key: Box<[u32]>, cont: Box<[u32]>) {
        if self.cand_keep == 0 {
            return;
        }
        match self.map.get_mut(&key) {
            Some(cands) => {
                // 同じ継続が既にあれば末尾へ移動 (最新化)。無ければ push。
                if let Some(pos) = cands.iter().position(|c| c == &cont) {
                    let c = cands.remove(pos).expect("pos は有効");
                    cands.push_back(c);
                } else {
                    if cands.len() >= self.cand_keep {
                        cands.pop_front();
                    }
                    cands.push_back(cont);
                }
            }
            None => {
                // 上限に達していたら最古のキーを落とす (挿入順 FIFO)。
                while self.map.len() >= self.max_entries {
                    let Some(oldest) = self.order.pop_front() else {
                        self.map.clear();
                        break;
                    };
                    self.map.remove(&oldest);
                }
                if self.map.len() < self.max_entries {
                    let mut cands = VecDeque::with_capacity(1);
                    cands.push_back(cont);
                    self.map.insert(key.clone(), cands);
                    self.order.push_back(key);
                }
            }
        }
    }

    /// 末尾の suffix で引いて draft トークン列 (最大 `k` 個) を返す。
    ///
    /// 最長一致 (`min(max_suffix, toks.len())`) から短くして探し、最初に見つかった
    /// キーの**最新の候補** (末尾) から最大 `k` トークンを返す。どの長さでも
    /// 見つからなければ空 (呼び出し側は通常 decode にフォールバックする)。
    pub fn draft(&self, toks: &[u32], k: usize) -> Vec<u32> {
        if k == 0 || self.max_draft == 0 {
            return Vec::new();
        }
        let n = self.max_suffix.min(toks.len());
        for s in (1..=n).rev() {
            let key = &toks[toks.len() - s..];
            if let Some(cands) = self.map.get(key) {
                // 最新の候補から (末尾)。空の継続は飛ばす (feed の末尾位置は
                // 継続がまだ無い — 登録はされるが 0 長)。
                for c in cands.iter().rev() {
                    if !c.is_empty() {
                        let take = c.len().min(k);
                        return c[..take].to_vec();
                    }
                }
            }
        }
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(max_suffix: usize, entries: usize, cand: usize, max_draft: usize) -> NgramIndex {
        NgramIndex::new(max_suffix, entries, cand, max_draft)
    }

    #[test]
    fn feed_then_draft_returns_continuation() {
        let mut d = idx(3, 100, 8, 4);
        d.feed(&[10, 11, 12, 13, 14, 15]);
        // 末尾 [13,14,15] の継続は存在しない (13 以降が末尾) → 14 の suffix
        // [11,12,13] の継続はあるが現在の末尾は [13,14,15]。
        // 現在末尾 [13,14,15] に一致するキーは無いので draft は空。
        assert!(d.draft(&[10, 11, 12, 13, 14, 15], 4).is_empty());
        // [14,15] だけでも無い (2 トークンの suffix として登録されていない末尾)。
        assert!(d.draft(&[10, 11, 12, 13, 14, 15], 4).is_empty());
    }

    #[test]
    fn repeated_pattern_gives_long_draft() {
        let mut d = idx(3, 100, 8, 4);
        let toks: Vec<u32> = (0..12).map(|i| (i % 5) as u32).collect();
        d.feed(&toks);
        // 繰り返し列なら最長一致 (3) で引ける。末尾が [.., 1, 2] (i=10,11) の
        // suffix は [0,1,2] (i=7..9) に一致し、継続は [3,4,0,..]。
        let draft = d.draft(&toks, 4);
        assert_eq!(draft, vec![3, 4, 0, 1], "{draft:?}");
    }

    #[test]
    fn falls_back_to_shorter_suffix() {
        let mut d = idx(3, 100, 8, 4);
        // 末尾 [41,42] は 1 回しか出ないが [42] の継続なら引ける。
        d.feed(&[7, 8, 41, 42, 9, 7, 8, 43]);
        let q = [41, 42];
        // suffix [41,42] はキーとして存在しない (feed した位置 i=5 の suffix は
        // [8,43] / [7,8] / [7] — 41,42 は継続側にしか出ない)。
        // 末尾を [42] にすると suffix [42] (i=3 の 1 トークン suffix) で引ける。
        let draft = d.draft(&[41, 42], 4);
        assert_eq!(draft, vec![9, 7, 8, 43], "{draft:?}");
        assert!(d.draft(&q, 4).is_empty());
    }

    #[test]
    fn latest_candidate_wins() {
        let mut d = idx(1, 100, 4, 2);
        d.feed(&[1, 100, 2, 1, 200, 3, 1, 100, 5]);
        // suffix [1] の継続は [100,2] → [200,3] → [100,5] の順。最新は [100,5]。
        let draft = d.draft(&[1], 2);
        assert_eq!(draft, vec![100, 5], "{draft:?}");
        // 3 回目の [1,100,..] で [100] の継続が [5] に更新される。
        let draft = d.draft(&[100], 1);
        assert_eq!(draft, vec![5], "{draft:?}");
    }

    #[test]
    fn fifo_eviction_drops_oldest_entry() {
        let mut d = idx(1, 3, 1, 1);
        d.feed(&[1, 10, 2, 20, 3, 30, 4]);
        // キー [1],[2],[3] が入る。次の feed で [4] が入り [1] が追い出される。
        d.feed(&[1, 10, 2, 20, 3, 30, 4, 40, 5]);
        assert_eq!(d.len(), 3, "エントリ上限 3");
        assert!(d.draft(&[1], 1).is_empty(), "最古の [1] は落ちている");
        assert_eq!(d.draft(&[2], 1), vec![20]);
        assert_eq!(d.draft(&[4], 1), vec![40]);
    }

    #[test]
    fn candidate_keep_limits_per_key() {
        let mut d = idx(1, 100, 2, 1);
        d.feed(&[1, 10, 1, 11, 1, 12, 1, 13]);
        // suffix [1] の候補は [11] と [13] だけ (cand_keep=2、古い [10] と [12] は落ちる)。
        let draft = d.draft(&[1], 1);
        assert_eq!(draft, vec![13], "{draft:?}");
    }

    #[test]
    fn big_u32_tokens_do_not_corrupt() {
        // 日本語トークン相当の大きな ID (u32 全域) でも辞書は壊れない。
        let big: Vec<u32> = [4_294_967_295u32, 100_000, 65_536, 4_294_967_294, 1, 2]
            .iter()
            .copied()
            .chain([4_294_967_295u32, 100_000, 65_536].iter().copied())
            .collect();
        let mut d = idx(3, 100, 4, 3);
        d.feed(&big);
        let draft = d.draft(&big, 3);
        assert_eq!(draft, vec![4_294_967_294, 1, 2], "{draft:?}");
    }

    #[test]
    fn empty_and_degenerate_configs_return_empty_draft() {
        let mut d = idx(3, 100, 4, 4);
        assert!(d.draft(&[], 4).is_empty(), "空トークン列");
        let mut zero = idx(0, 100, 4, 4);
        zero.feed(&[1, 2, 3, 4]);
        assert!(zero.draft(&[1, 2, 3, 4], 4).is_empty(), "max_suffix=0");
        let mut nocont = idx(3, 100, 4, 0);
        nocont.feed(&[1, 2, 3, 4]);
        assert!(nocont.draft(&[1, 2, 3, 4], 4).is_empty(), "max_draft=0");
        let mut nocand = idx(3, 100, 0, 4);
        nocand.feed(&[1, 2, 3, 4]);
        assert!(nocand.draft(&[1, 2, 3, 4], 4).is_empty(), "cand_keep=0");
    }

    #[test]
    fn shrink_feed_resets_baseline() {
        let mut d = idx(2, 100, 4, 2);
        d.feed(&[1, 2, 3, 4]);
        let fed1 = d.fed();
        // 短い列を渡す (呼び出し側の誤用に相当)。全体を見直して fed は進む。
        d.feed(&[1, 2]);
        assert!(d.fed() >= fed1);
        assert_eq!(d.draft(&[1, 2], 2), vec![3, 4]);
    }
}
