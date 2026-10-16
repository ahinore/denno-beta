//! マルチエントリ LRU prefix cache。
//!
//! 直前1会話分のみキャッシュする単一エントリ方式 (engine.rs の cached_prompt + KvCache)
//! に対し、過去の会話の KV をスナップショットとして複数保持する。
//! 複数セッションが交互に来ても prefix prefill を省略できるようにするのが目的。
//!
//! - 保持上限: 全エントリの重み合計 (`max_tokens`)。重み = KV トークン数 +
//!   GDN 状態スナップショットのトークン換算 (A4)。超過時は LRU で eviction
//! - 選択: リクエスト prompt との LCP が最大のエントリ。再帰モデル (Qwen3.5 GDN)
//!   向けには `best_full_match` (prompt 全長一致のエントリのみ) も提供する
//! - メモリ: KV 約 24.6KB/トークン (Qwen2.5-0.5B, 24層×kv_dim 256×2×f32) ×
//!   max_tokens + GDN 状態スナップショット (1 エントリあたりモデルの全 GDN 層の
//!   conv + SSM 状態 1 回分)
//!
//! ## A4: GDN 状態スナップショット
//!
//! 再帰モデルの GDN 状態 (conv + SSM) は「トークン位置ごと」ではなく
//! **シーケンス全体の関数**なので、スナップショットは 1 つの位置でしか有効でない。
//! 本実装は **prompt 境界 (prefill 直後・decode 前)** の状態だけを保存する
//! (`gdn: Option<...>`・engine が `KvCache::gdn_snapshot()` で取得)。したがって:
//!
//! - 復元は **新しい prompt がエントリの prompt 全長を接頭辞として含む場合のみ**
//!   (`restore_into` の `n == entry.prompt.len()` 条件)。それ以外は `false` を返し、
//!   呼び出し側は full prefill にフォールバックする
//! - KV と GDN 状態は常にペアで保存・復元する (片方だけ復元して再帰状態を
//!   不整合にする経路は存在しない)
//! - 復元後の継続 forward がフル prefill と bit 一致になることは、
//!   キャッシュ連続性 (forward(a)+forward(b) == forward(a++b) bit 一致) と
//!   状態の bit コピーから従う (qwen35.rs のテストで検証)

use crate::kv_cache::KvCache;

/// GDN 層の状態スナップショット (層 index 順)。GDN 層は `Some((conv, ssm))`・
/// full attention 層は `None`。非再帰モデルのエントリは `None` (KV のみ)。
pub type GdnStateSnapshot = Vec<Option<(Vec<f32>, Vec<f32>)>>;

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
    /// GDN 状態スナップショット (A4)。`Some` の場合 prompt 境界位置の状態を保持し、
    /// 復元は `n == prompt.len()` のときにのみ行う
    pub gdn: Option<GdnStateSnapshot>,
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
    /// 全エントリの重み合計の上限 (LRU eviction の閾値)。
    /// 重み = KV トークン数 + GDN 状態のトークン換算 (A4)
    pub max_tokens: usize,
    /// KV 1 トークンあたりのバイト数 (GDN 状態をトークン換算するためのスケール)。
    /// `KvCache::bytes_per_token()` と同じ値 (GDN 層の KV は除外済み) を渡す
    bytes_per_token: usize,
    /// LRU 用の単調増加クロック
    clock: u64,
}

impl PrefixCache {
    pub fn new(max_tokens: usize, bytes_per_token: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_tokens,
            // 0 除算ガード (KV を全く持たないモデルでも動くように)
            bytes_per_token: bytes_per_token.max(1),
            clock: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 全エントリの重み合計 (KV トークン数 + GDN 状態のトークン換算)
    pub fn total_weight(&self) -> usize {
        self.entries.iter().map(|e| self.weight_of(e)).sum()
    }

    /// エントリ 1 つの重み。GDN 状態スナップショットはそのバイト数を
    /// KV 1 トークンあたりのバイト数で割ってトークン換算する (A4)。
    /// これにより「KV 予算」に状態スナップショットの実メモリも反映され、
    /// 状態の大きいモデル (27B で ~150MB/エントリ) でも合計メモリが予算内に収まる。
    fn weight_of(&self, e: &PrefixEntry) -> usize {
        self.weight_of_parts(e.kv_len, &e.gdn)
    }

    /// kv_len と GDN 状態から重みを計算する (挿入前の上限判定にも使う)。
    fn weight_of_parts(&self, kv_len: usize, gdn: &Option<GdnStateSnapshot>) -> usize {
        let gdn_bytes = match gdn {
            None => 0,
            Some(states) => states
                .iter()
                .flatten()
                .map(|(c, s)| (c.len() + s.len()) * std::mem::size_of::<f32>())
                .sum::<usize>(),
        };
        kv_len + (gdn_bytes + self.bytes_per_token - 1) / self.bytes_per_token
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

    /// prompt を**完全に**接頭辞として含む (LCP == 自身の prompt 長) エントリを
    /// 返す (A4・再帰モデル用)。
    ///
    /// GDN 状態スナップショットはエントリの prompt 境界位置でしか取得していない
    /// ため、再帰モデルの復元は「エントリ prompt 全長との一致」が必須。
    /// 完全一致候補が複数ある場合は**最長のもの**を返す (s46: エントリは
    /// generation prompt 前までの「base」で保存されるため、マルチターン会話では
    /// turn1..turnN の base がすべて完全一致候補になる。最長を選ばないと
    /// turn_{n} の継続で turn_1 の base が復元され、再計算が減らない)。
    /// 戻り値の lcp は `entry.prompt.len()` そのものになる。
    pub fn best_full_match(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if e.prompt.is_empty() || e.lcp_with(prompt) != e.prompt.len() {
                continue;
            }
            // 完全一致候補のうち最長のもの (同長は走査順で先に見つかった方)
            let better = match best {
                None => true,
                Some((_, bl)) => e.prompt.len() > bl,
            };
            if better {
                best = Some((i, e.prompt.len()));
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

    /// live の KvCache から KV のみスナップショットを挿入する (GDN 状態なし)。
    /// 単体テスト用。再帰モデルのエントリは必ず GDN 状態をペアで持つ必要がある
    /// (`insert_snapshot` に `Some(states)` を渡すこと)。
    pub fn insert(&mut self, prompt: Vec<u32>, cache: &KvCache, kv_len: usize) {
        if kv_len == 0 {
            return;
        }
        let (k, v) = cache.snapshot(kv_len);
        self.insert_snapshot(prompt, k, v, kv_len, None);
    }

    /// 所有済みスナップショットを挿入する (insert の実体)。
    /// - 同一 prompt のエントリが既にあれば置き換え (新鮮な KV/状態へ更新)
    /// - 単体で上限を超えるエントリは格納しない
    /// - 挿入後、合計重みが上限を超えていれば LRU 順に eviction
    ///
    /// `gdn` には `KvCache::gdn_snapshot()` の結果 (prompt 境界位置の状態) を渡す。
    /// 非再帰モデルは `None`。
    pub fn insert_snapshot(
        &mut self,
        prompt: Vec<u32>,
        k: Vec<Vec<f32>>,
        v: Vec<Vec<f32>>,
        kv_len: usize,
        gdn: Option<GdnStateSnapshot>,
    ) {
        if prompt.is_empty() || kv_len == 0 || self.weight_of_parts(kv_len, &gdn) > self.max_tokens {
            return;
        }
        self.clock += 1;
        let last_used = self.clock;

        // 同一 prompt の重複は置き換え
        if let Some(e) = self.entries.iter_mut().find(|e| e.prompt == prompt) {
            e.k = k;
            e.v = v;
            e.kv_len = kv_len;
            e.gdn = gdn;
            e.last_used = last_used;
        } else {
            self.entries.push(PrefixEntry {
                prompt,
                k,
                v,
                kv_len,
                gdn,
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

    /// 合計重みが上限以下になるまで、最も古く使われたエントリから捨てる。
    /// (insert 経路では新規エントリは単体で上限以下なので、ストアが空になることはない。
    ///  set_max_tokens で上限を下げた場合のみ空になり得る)
    fn evict_to_limit(&mut self) {
        while self.total_weight() > self.max_tokens {
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
    ///
    /// A4: GDN 状態スナップショットを持つエントリ (`gdn: Some`) は、
    /// `n == entry.prompt.len()` (prompt 境界 == 状態の取得位置) のときにのみ
    /// **KV + GDN 状態をペアで**復元する。それ以外の n では再帰状態を正しく
    /// 用意できないため、KV のみの復元も行わず `false` を返す
    /// (片方だけ復元した場合の静かな出力変化を構造的に防ぐ)。
    pub fn restore_into(&self, index: usize, n: usize, cache: &mut KvCache) -> bool {
        let Some(e) = self.entries.get(index) else {
            return false;
        };
        if e.prompt.is_empty() || e.kv_len < n || n == 0 || n > e.prompt.len() {
            return false;
        }
        if let Some(gdn) = &e.gdn {
            if n != e.prompt.len() {
                // 状態を持てない位置での部分復元は拒否 (full prefill にフォールバック)
                return false;
            }
            cache.gdn_restore(gdn);
        }
        cache.restore(&e.k, &e.v, n);
        true
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

    fn pc(max_tokens: usize) -> PrefixCache {
        // cache_with の KvCache (1層 × kv_dim 2) の bytes_per_token = 16
        PrefixCache::new(max_tokens, 16)
    }

    #[test]
    fn best_match_picks_max_lcp() {
        let mut pc = pc(100);
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
        let mut pc = pc(100);
        pc.insert(vec![1, 2, 3], &cache_with(3), 3);
        assert!(pc.best_match(&[7, 8, 9]).is_none());
        assert!(pc.best_match(&[1]).is_some());
    }

    /// A4: best_full_match は「prompt 全長一致」のエントリだけを返す
    #[test]
    fn best_full_match_requires_full_prefix() {
        let mut pc = pc(100);
        pc.insert(vec![1, 2, 3], &cache_with(3), 3);
        pc.insert(vec![1, 2, 9], &cache_with(3), 3);

        // [1,2,3,4] はエントリ [1,2,3] を完全に含む
        let (i, lcp) = pc.best_full_match(&[1, 2, 3, 4]).unwrap();
        assert_eq!(i, 0);
        assert_eq!(lcp, 3);

        // [1,2,4] はどのエントリも完全には含まない (LCP 2 < 3)
        assert!(pc.best_full_match(&[1, 2, 4]).is_none());
        // best_match なら部分一致が返る (非再帰経路との対比)
        assert!(pc.best_match(&[1, 2, 4]).is_some());
        // 同一 prompt も完全一致
        assert!(pc.best_full_match(&[1, 2, 3]).is_some());
    }

    /// s46: 完全一致候補が複数ある場合は最長のものを選ぶ。
    /// (base エントリ化により、マルチターン会話では turn1..turnN の base が
    /// すべて完全一致候補になる。最長を選ばないと turn_{n} の継続で turn_1 の
    /// base が復元され、再計算が減らない)
    #[test]
    fn best_full_match_picks_longest_full_match() {
        let mut pc = pc(100);
        pc.insert(vec![1, 2], &cache_with(2), 2); // turn1 の base
        pc.insert(vec![1, 2, 3, 4], &cache_with(4), 4); // turn2 の base (turn1 base を完全に含む)

        // どちらのエントリも prompt を完全に含む → 最長 (4) が選ばれる
        let (i, lcp) = pc.best_full_match(&[1, 2, 3, 4, 5]).unwrap();
        assert_eq!(i, 1);
        assert_eq!(lcp, 4);

        // 短い方しか完全に含まない場合は短い方が選ばれる
        let (i, lcp) = pc.best_full_match(&[1, 2, 8, 8]).unwrap();
        assert_eq!(i, 0);
        assert_eq!(lcp, 2);

        // 完全一致候補が無い場合
        assert!(pc.best_full_match(&[1, 9, 9]).is_none());
    }

    #[test]
    fn insert_dedupes_same_prompt() {
        let mut pc = pc(100);
        pc.insert(vec![1, 2], &cache_with(2), 2);
        assert!(pc.contains_prompt(&[1, 2]));
        assert!(!pc.contains_prompt(&[9, 9]));
        pc.insert(vec![1, 2], &cache_with(5), 5); // 同一 prompt → 置き換え
        assert_eq!(pc.len(), 1);
        assert_eq!(pc.total_weight(), 5);
    }

    #[test]
    fn insert_skips_entry_larger_than_limit() {
        let mut pc = pc(4);
        pc.insert(vec![1, 2, 3, 4, 5], &cache_with(5), 5);
        assert!(pc.is_empty());
        // 上限ちょうどは入る
        pc.insert(vec![1, 2, 3, 4], &cache_with(4), 4);
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn evicts_least_recently_used_to_fit_budget() {
        let mut pc = pc(10);
        pc.insert(vec![1], &cache_with(4), 4); // A: 4
        pc.insert(vec![2], &cache_with(4), 4); // B: 4 (計8)
        pc.touch(0); // A を最近使った
        pc.insert(vec![3], &cache_with(4), 4); // C: 4 → 計12 > 10 → LRU な B が捨てられる
        assert_eq!(pc.len(), 2);
        assert_eq!(pc.total_weight(), 8);
        // B(vec![2]) は無く、A と C が残る
        assert!(pc.best_match(&[1]).is_some());
        assert!(pc.best_match(&[2]).is_none());
        assert!(pc.best_match(&[3]).is_some());
    }

    #[test]
    fn evict_keeps_newest_entries_until_budget_fits() {
        let mut pc = pc(6);
        pc.insert(vec![1], &cache_with(3), 3);
        pc.insert(vec![2], &cache_with(3), 3);
        pc.insert(vec![3], &cache_with(3), 3); // 計9 > 6 → 最も古い A が捨てられ 6 で停止
        assert_eq!(pc.len(), 2);
        assert_eq!(pc.total_weight(), 6);
        assert!(pc.best_match(&[1]).is_none());
        assert!(pc.best_match(&[2]).is_some());
        assert!(pc.best_match(&[3]).is_some());
    }

    /// A4: GDN 状態スナップショットのバイト数も重みに計上される
    /// (conv 2 + ssm 4 = 6 f32 = 24B → bpt 16 で 2 トークン換算)
    #[test]
    fn gdn_state_counts_into_eviction_weight() {
        let mut store = pc(10);
        let mut cache = cache_with(4);
        cache.attach_gdn_slot(0, 2, 4);
        // conv 2 + ssm 4 = 6 f32 = 24B → 24/16 = 切り上げ 2 トークン換算
        let gdn = cache.gdn_snapshot();
        assert!(gdn[0].is_some());
        store.insert_snapshot(vec![1], cache.snapshot(4).0, cache.snapshot(4).1, 4, Some(gdn));
        // KV 4 + 状態換算 2 = 6
        assert_eq!(store.total_weight(), 6);

        // KV のみのエントリ (同 kv_len) は重み 4 → 状態付きより軽い
        let mut store2 = pc(10);
        store2.insert(vec![1], &cache_with(4), 4);
        assert_eq!(store2.total_weight(), 4);

        // 状態付きエントリが上限を超える場合は格納されない
        let mut store3 = pc(5); // 状態付きの重み 6 > 5
        let mut cache3 = cache_with(4);
        cache3.attach_gdn_slot(0, 2, 4);
        store3.insert_snapshot(
            vec![1],
            cache3.snapshot(4).0,
            cache3.snapshot(4).1,
            4,
            Some(cache3.gdn_snapshot()),
        );
        assert!(store3.is_empty(), "状態込みで上限超過のエントリは拒否");
    }

    #[test]
    fn restore_into_copies_prefix_rows() {
        let mut src = KvCache::new(1, 2, 64);
        {
            let (k, _) = src.kv_bufs(0, 3);
            k[0..6].copy_from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        }
        src.advance(3);
        let mut pc = pc(100);
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
        let mut pc = pc(100);
        pc.insert(vec![1, 2, 3], &cache_with(3), 3);
        let mut dst = KvCache::new(1, 2, 64);
        assert!(!pc.restore_into(9, 2, &mut dst)); // 存在しない index
        assert!(!pc.restore_into(0, 0, &mut dst)); // n = 0 は復元でなく reset で対応
        assert!(!pc.restore_into(0, 99, &mut dst)); // kv_len を超える n
    }

    /// A4: 状態付きエントリの復元は「prompt 境界位置」でのみ KV + 状態が
    /// ペアで復元される。部分復元 (n < prompt.len) は拒否され、キャッシュも
    /// 呼び出し前のまま (KV のみ復元して状態を不整合にする経路がない)。
    #[test]
    fn gdn_entry_restores_state_only_at_prompt_boundary() {
        // 層 1 を GDN 層、層 0 を KV 層にする (GDN 層の KV は A3 で除外される)
        let mut src = KvCache::new(2, 2, 64);
        src.attach_gdn_slot(1, 3, 5);
        {
            let (k, _) = src.kv_bufs(0, 4);
            k[0..8].copy_from_slice(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
            let (cv, ssm) = src.gdn_state_mut(1).unwrap();
            cv.copy_from_slice(&[9.0, 8.0, 7.0]);
            ssm.copy_from_slice(&[0.5, 0.25, -0.5, -0.25, 1.0]);
        }
        src.advance(4);
        let (k, v) = src.snapshot(4);
        assert!(!k[0].is_empty(), "KV 層 (層 0) はスナップショット対象");
        let gdn = src.gdn_snapshot();
        assert!(gdn[1].is_some() && gdn[0].is_none());

        let mut store = pc(100);
        store.insert_snapshot(vec![1, 2, 3, 4], k, v, 4, Some(gdn));

        // prompt 境界 (n == 4): KV + 状態が復元される
        let mut dst = KvCache::new(2, 2, 64);
        dst.attach_gdn_slot(1, 3, 5);
        assert!(store.restore_into(0, 4, &mut dst));
        assert_eq!(dst.len(), 4);
        let (k2, _) = dst.kv_bufs_ref(0);
        assert_eq!(k2[0..8], [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
        let (cv, ssm) = dst.gdn_state_mut(1).unwrap();
        assert_eq!(cv.to_vec(), vec![9.0, 8.0, 7.0]);
        assert_eq!(ssm.to_vec(), vec![0.5, 0.25, -0.5, -0.25, 1.0]);

        // 部分復元 (n < prompt.len): 拒否されキャッシュは無変更
        let mut dst2 = KvCache::new(2, 2, 64);
        dst2.attach_gdn_slot(1, 3, 5);
        dst2.advance(1); // 復元が起きないことを見るため事前に len を設定
        assert!(!store.restore_into(0, 2, &mut dst2));
        assert_eq!(dst2.len(), 1, "拒否時にキャッシュが変更されてはならない");
        let (cv2, ssm2) = dst2.gdn_state_mut(1).unwrap();
        assert!(cv2.iter().all(|&x| x == 0.0) && ssm2.iter().all(|&x| x == 0.0));
    }

    /// A4: 同一 prompt の置き換えで GDN 状態も更新される
    #[test]
    fn insert_snapshot_replaces_gdn_state_on_dedupe() {
        let mut pc = pc(100);
        let mut c1 = KvCache::new(1, 2, 64);
        c1.attach_gdn_slot(0, 1, 1);
        {
            let (_, s) = c1.gdn_state_mut(0).unwrap();
            s[0] = 1.0;
        }
        c1.advance(2);
        pc.insert_snapshot(vec![1, 2], c1.snapshot(2).0, c1.snapshot(2).1, 2, Some(c1.gdn_snapshot()));

        let mut c2 = KvCache::new(1, 2, 64);
        c2.attach_gdn_slot(0, 1, 1);
        {
            let (_, s) = c2.gdn_state_mut(0).unwrap();
            s[0] = -3.5;
        }
        c2.advance(3);
        pc.insert_snapshot(vec![1, 2], c2.snapshot(3).0, c2.snapshot(3).1, 3, Some(c2.gdn_snapshot()));
        assert_eq!(pc.len(), 1);

        let mut dst = KvCache::new(1, 2, 64);
        dst.attach_gdn_slot(0, 1, 1);
        assert!(pc.restore_into(0, 2, &mut dst));
        assert_eq!(dst.len(), 2, "kv_len は置き換え後の値 (3) だが復元 n は prompt.len でクランプ");
        let (_, s) = dst.gdn_state_mut(0).unwrap();
        assert_eq!(s[0], -3.5, "GDN 状態も置き換え後のもの");
    }
}
