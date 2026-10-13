//! マルチエントリ LRU prefix cache。
//!
//! 直前1会話分のみキャッシュする単一エントリ方式 (engine.rs の cached_prompt +
//! KvCache) に対し、過去の会話の KV をスナップショットとして複数保持する。
//! 複数セッションが交互に来ても prefix prefill を省略できるようにするのが目的。
//!
//! - 保持上限: 全エントリのトークン数合計 (`max_tokens`)。超過時は LRU で eviction
//! - 選択: リクエスト prompt との LCP が最大のエントリ
//! - メモリ: 約 24.6KB/トークン (Qwen2.5-0.5B, 24層×kv_dim 256×2×f32) × max_tokens

use crate::kv_cache::KvCache;

/// 1会話分のスナップショット。KV の行 i はトークン列 prompt (+生成分) の
/// 位置 i の KV に対応する (prompt 自体も位置整合する接頭辞)。
#[derive(Debug, Clone)]
pub struct PrefixEntry {
    pub prompt: Vec<u32>,
    /// 層ごとの K バッファ (有効なのは先頭 kv_len 行 = kv_len * kv_dim 要素)
    pub k: Vec<Vec<f32>>,
    /// 層ごとの V バッファ
    pub v: Vec<Vec<f32>>,
    /// スナップショットに含まれるトークン数 (prompt + 生成トークン)
    pub kv_len: usize,
    /// LRU 用の最終使用時刻 (単調増加カウンタ)
    last_used: u64,
}

impl PrefixEntry {
    /// prompt との LCP (最長共通接頭辞) を返す
    pub fn lcp_with(&self, prompt: &[u32]) -> usize {
        prompt
            .iter()
            .zip(self.prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
    }
}

/// マルチエントリ prefix cache 本体。
#[derive(Debug)]
pub struct PrefixCache {
    entries: Vec<PrefixEntry>,
    /// 全エントリの kv_len 合計の上限 (LRU eviction の閾値)
    pub max_tokens: usize,
    /// LRU 用の単調増加クロック
    clock: u64,
}

impl PrefixCache {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_tokens,
            clock: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 全エントリのトークン数合計
    pub fn total_tokens(&self) -> usize {
        self.entries.iter().map(|e| e.kv_len).sum()
    }

    /// prompt との LCP が最大のエントリを返す (同点は最後に使用した方)。
    /// LCP が 0 のエントリしか無ければ None。
    pub fn best_match(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None; // (index, lcp)
        for (i, e) in self.entries.iter().enumerate() {
            let lcp = e.lcp_with(prompt);
            if lcp == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, bl)) => lcp > bl, // 同点は先に見つかった方 (= より最近使われた方) を維持
            };
            if better {
                best = Some((i, lcp));
            }
        }
        best
    }

    /// エントリを使用済みとしてマークする (LRU 更新)
    pub fn touch(&mut self, index: usize) {
        self.clock += 1;
        if let Some(e) = self.entries.get_mut(index) {
            e.last_used = self.clock;
        }
    }

    /// 同一 prompt のエントリを既に保持しているか
    pub fn contains_prompt(&self, prompt: &[u32]) -> bool {
        self.entries.iter().any(|e| e.prompt == prompt)
    }

    /// live の KvCache からスナップショットを挿入する。
    /// - 同一 prompt のエントリが既にあれば置き換え (新鮮な KV へ更新)
    /// - 単体で上限を超えるエントリは格納しない
    /// - 挿入後、合計トークン数が上限を超えていれば LRU 順に eviction
    pub fn insert(&mut self, prompt: Vec<u32>, cache: &KvCache, kv_len: usize) {
        if kv_len == 0 {
            return;
        }
        let (k, v) = cache.snapshot(kv_len);
        self.insert_snapshot(prompt, k, v, kv_len);
    }

    /// 所有済みスナップショットを挿入する (insert の実体)。
    /// live を退避してから別エントリを復元する順序 (restore → insert) で使う。
    pub fn insert_snapshot(
        &mut self,
        prompt: Vec<u32>,
        k: Vec<Vec<f32>>,
        v: Vec<Vec<f32>>,
        kv_len: usize,
    ) {
        if prompt.is_empty() || kv_len == 0 || kv_len > self.max_tokens {
            return;
        }
        self.clock += 1;
        let last_used = self.clock;

        // 同一 prompt の重複は置き換え
        if let Some(e) = self.entries.iter_mut().find(|e| e.prompt == prompt) {
            e.k = k;
            e.v = v;
            e.kv_len = kv_len;
            e.last_used = last_used;
        } else {
            self.entries.push(PrefixEntry {
                prompt,
                k,
                v,
                kv_len,
                last_used,
            });
        }
        self.evict_to_limit();
    }

    /// 上限を変更し、超過分を LRU 順に eviction する
    pub fn set_max_tokens(&mut self, max_tokens: usize) {
        self.max_tokens = max_tokens;
        self.evict_to_limit();
    }

    /// 合計トークン数が上限以下になるまで、最も古く使われたエントリから捨てる。
    /// (insert 経路では新規エントリは単体で上限以下なので、ストアが空になることはない。
    ///  set_max_tokens で上限を下げた場合のみ空になり得る)
    fn evict_to_limit(&mut self) {
        while self.total_tokens() > self.max_tokens {
            let victim = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .unwrap();
            self.entries.remove(victim);
        }
    }

    /// エントリの先頭 n 行を KvCache へ復元する。
    /// エントリが無ければ false を返す (呼び出し側は full prefill にフォールバック)。
    pub fn restore_into(&self, index: usize, n: usize, cache: &mut KvCache) -> bool {
        match self.entries.get(index) {
            Some(e) if e.prompt.len() >= n && e.kv_len >= n && n > 0 => {
                cache.restore(&e.k, &e.v, n);
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with(len: usize) -> KvCache {
        let mut c = KvCache::new(1, 2, 64);
        // P17e: 需要ベース確保なので、事前に len トークン分のバッファを成長させる
        let _ = c.kv_bufs(0, len);
        c.advance(len);
        c
    }

    #[test]
    fn best_match_picks_max_lcp() {
        let mut pc = PrefixCache::new(100);
        pc.insert(vec![1, 2, 3, 4], &cache_with(4), 4);
        pc.insert(vec![1, 2, 9, 9, 9], &cache_with(5), 5);

        // [1,2,3,...] とは LCP 3、[1,2,9,...] とは LCP 2 → 前者が選ばれる
        let (i, lcp) = pc.best_match(&[1, 2, 3, 8]).unwrap();
        assert_eq!(i, 0);
        assert_eq!(lcp, 3);

        let (i, lcp) = pc.best_match(&[1, 2, 9, 1]).unwrap();
        assert_eq!(i, 1);
        assert_eq!(lcp, 3); // [1,2,3,4] とは LCP 2、[1,2,9,9,9] とは LCP 3
    }

    #[test]
    fn best_match_returns_none_on_no_overlap() {
        let mut pc = PrefixCache::new(100);
        pc.insert(vec![1, 2, 3], &cache_with(3), 3);
        assert!(pc.best_match(&[7, 8, 9]).is_none());
        assert!(pc.best_match(&[1]).is_some());
    }

    #[test]
    fn insert_dedupes_same_prompt() {
        let mut pc = PrefixCache::new(100);
        pc.insert(vec![1, 2], &cache_with(2), 2);
        assert!(pc.contains_prompt(&[1, 2]));
        assert!(!pc.contains_prompt(&[9, 9]));
        pc.insert(vec![1, 2], &cache_with(5), 5); // 同一 prompt → 置き換え
        assert_eq!(pc.len(), 1);
        assert_eq!(pc.total_tokens(), 5);
    }

    #[test]
    fn insert_skips_entry_larger_than_limit() {
        let mut pc = PrefixCache::new(4);
        pc.insert(vec![1, 2, 3, 4, 5], &cache_with(5), 5);
        assert!(pc.is_empty());
        // 上限ちょうどは入る
        pc.insert(vec![1, 2, 3, 4], &cache_with(4), 4);
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn evicts_least_recently_used_to_fit_budget() {
        let mut pc = PrefixCache::new(10);
        pc.insert(vec![1], &cache_with(4), 4); // A: 4
        pc.insert(vec![2], &cache_with(4), 4); // B: 4 (計8)
        pc.touch(0); // A を最近使った
        pc.insert(vec![3], &cache_with(4), 4); // C: 4 → 計12 > 10 → LRU な B が捨てられる
        assert_eq!(pc.len(), 2);
        assert_eq!(pc.total_tokens(), 8);
        // B(vec![2]) は無く、A と C が残る
        assert!(pc.best_match(&[1]).is_some());
        assert!(pc.best_match(&[2]).is_none());
        assert!(pc.best_match(&[3]).is_some());
    }

    #[test]
    fn evict_keeps_newest_entries_until_budget_fits() {
        let mut pc = PrefixCache::new(6);
        pc.insert(vec![1], &cache_with(3), 3);
        pc.insert(vec![2], &cache_with(3), 3);
        pc.insert(vec![3], &cache_with(3), 3); // 計9 > 6 → 最も古い A が捨てられ 6 で停止
        assert_eq!(pc.len(), 2);
        assert_eq!(pc.total_tokens(), 6);
        assert!(pc.best_match(&[1]).is_none());
        assert!(pc.best_match(&[2]).is_some());
        assert!(pc.best_match(&[3]).is_some());
    }

    #[test]
    fn restore_into_copies_prefix_rows() {
        let mut src = KvCache::new(1, 2, 64);
        {
            let (k, _) = src.kv_bufs(0, 3);
            k[0..6].copy_from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        }
        src.advance(3);
        let mut pc = PrefixCache::new(100);
        pc.insert(vec![1, 2, 3], &src, 3);

        let mut dst = KvCache::new(1, 2, 64);
        assert!(pc.restore_into(0, 2, &mut dst)); // LCP 2 分だけ復元
        assert_eq!(dst.len(), 2);
        let (k, _) = dst.kv_bufs(0, 1); // 3 行目まで成長させて stale 行を確認
        assert_eq!(k[0..4], [1.0, 1.0, 2.0, 2.0]);
        assert_eq!(k[4..6], [0.0, 0.0]);
    }

    #[test]
    fn restore_into_returns_false_for_bad_index_or_n() {
        let mut pc = PrefixCache::new(100);
        pc.insert(vec![1, 2, 3], &cache_with(3), 3);
        let mut dst = KvCache::new(1, 2, 64);
        assert!(!pc.restore_into(9, 2, &mut dst)); // 存在しない index
        assert!(!pc.restore_into(0, 0, &mut dst)); // n = 0 は復元でなく reset で対応
        assert!(!pc.restore_into(0, 99, &mut dst)); // kv_len を超える n
    }
}
