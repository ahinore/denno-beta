//! task 866: prefix KV スロットの radix (パス圧縮トライ) 索引。
//!
//! 目的と限界 (監査の結論) — 詳しい経緯は `serve_flash.rs` の P812/task 866 の
//! doc にある:
//!
//! - KV / GDN の系列状態は「丸ごと」しか持ち回れない。full attention の KV は
//!   層ごとの長さを戻せる (`AttnCache::truncate_layer`) が、GDN (ssm) の再帰状態は
//!   巻き戻しに層ごとの clone が要る (spec decode の `SpecGdnSnap` がそれ)。
//!   1 文脈あたり 117 MiB の clone を毎リクエスト取るのは、この task の範囲
//!   (索引と選択) の外なので、**部分一致の KV 再利用はしない**。
//!   再利用は従来どおり「キャッシュ全体がプロンプトの前置き」のときだけ。
//! - したがってトライが共有するのは **トークン列の索引だけ** で、KV は共有しない。
//!   それでも、スロット数 N に対する共通前置き長の計算が
//!   O(N × プロンプト長) から O(プロンプト長 + N) になり、共通 system プロンプトの
//!   ような長い前置きを持つ会話が増えても選択のコストが増えない。
//! - さらに「どのスロットを追い出すか」を前置きの近さで決められる
//!   ([`evict_victim`])。分岐会話 (同じ前置きから別の続き) が親の文脈を
//!   追い出さないための heuristic で、再利用の可否 (出力) には影響しない。
//!
//! 既定 off (`DENNO_KV_RADIX=1` で有効)。索引はリクエストごとにスロット列から
//! 作り直す (最大 8 本 × 文脈長。prefill の秒に対して無視できる) ので、
//! スロットと索引がずれる状態が存在しない。

/// パス圧縮トライの 1 ノード。
#[derive(Debug)]
struct Node {
    /// 親からの辺のラベル (空なのは根だけ)。
    label: Vec<u32>,
    /// 根からのトークン数。
    depth: usize,
    parent: Option<usize>,
    children: Vec<usize>,
    /// このノードの表すトークン列 **ちょうど** を保存しているスロット。
    slots: Vec<usize>,
}

/// スロットのトークン列を索引する radix 木。
///
/// 同じ前置き (共通 system プロンプトなど) を持つスロットは 1 本のパスを共有する。
#[derive(Debug, Default)]
pub(crate) struct KvRadix {
    nodes: Vec<Node>,
}

impl KvRadix {
    pub(crate) fn new() -> Self {
        KvRadix {
            nodes: vec![Node {
                label: Vec::new(),
                depth: 0,
                parent: None,
                children: Vec::new(),
                slots: Vec::new(),
            }],
        }
    }

    /// スロット列 (`None` は空きスロット) から索引を作る。
    pub(crate) fn from_slots<'a>(slots: impl Iterator<Item = Option<&'a [u32]>>) -> Self {
        let mut r = Self::new();
        for (i, s) in slots.enumerate() {
            if let Some(toks) = s {
                r.insert(toks, i);
            }
        }
        r
    }

    /// ノード数 (テストとログ用)。根を含む。
    pub(crate) fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// 索引しているトークン列の総和 (ノード数の比較用)。
    pub(crate) fn total_tokens(&self) -> usize {
        self.nodes.iter().map(|n| n.label.len()).sum()
    }

    fn new_node(&mut self, label: Vec<u32>, parent: usize, depth: usize) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            label,
            depth,
            parent: Some(parent),
            children: Vec::new(),
            slots: Vec::new(),
        });
        id
    }

    /// `seq` を `slot` として索引へ入れる (同じ `slot` を 2 回入れない前提:
    /// 索引はリクエストごとに作り直す)。
    pub(crate) fn insert(&mut self, seq: &[u32], slot: usize) {
        let mut node = 0usize;
        let mut pos = 0usize;
        while pos < seq.len() {
            let tok = seq[pos];
            let child = self.nodes[node]
                .children
                .iter()
                .copied()
                .find(|&c| self.nodes[c].label.first() == Some(&tok));
            let Some(child) = child else {
                let id = {
                    let depth = self.nodes[node].depth + seq.len() - pos;
                    self.new_node(seq[pos..].to_vec(), node, depth)
                };
                self.nodes[id].slots.push(slot);
                self.nodes[node].children.push(id);
                return;
            };
            let m = common_prefix(&self.nodes[child].label, &seq[pos..]);
            if m == self.nodes[child].label.len() {
                node = child;
                pos += m;
                continue;
            }
            // 辺の途中で分岐する: 中間ノードを作って辺を割る。
            let label = std::mem::take(&mut self.nodes[child].label);
            let head = label[..m].to_vec();
            let tail = label[m..].to_vec();
            let mid_depth = self.nodes[node].depth + m;
            let mid = self.new_node(head, node, mid_depth);
            self.nodes[mid].children.push(child);
            self.nodes[child].label = tail;
            self.nodes[child].parent = Some(mid);
            self.nodes[child].depth = mid_depth + self.nodes[child].label.len();
            let at = self.nodes[node]
                .children
                .iter()
                .position(|&c| c == child)
                .expect("child は親の children に居る");
            self.nodes[node].children[at] = mid;
            pos += m;
            if pos == seq.len() {
                self.nodes[mid].slots.push(slot);
                return;
            }
            let id = {
                let depth = mid_depth + seq.len() - pos;
                self.new_node(seq[pos..].to_vec(), mid, depth)
            };
            self.nodes[id].slots.push(slot);
            self.nodes[mid].children.push(id);
            return;
        }
        self.nodes[node].slots.push(slot);
    }

    /// `prompt` と各スロットのトークン列の共通前置き長 (スロット数ぶんの Vec)。
    ///
    /// 索引していないスロットは 0。値は `first_mismatch(slot_toks, prompt)` と
    /// 完全に一致する (テスト `lcps_match_bruteforce` がランダム列で突き合わせる)。
    /// プロンプトを 1 回なぞるだけで全スロット分が埋まる。
    pub(crate) fn lcps(&self, prompt: &[u32], n_slots: usize) -> Vec<usize> {
        let mut out = vec![0usize; n_slots];
        let mut node = 0usize;
        let mut pos = 0usize;
        loop {
            // このノードの部分木 (ノード自身の payload を含む) は、少なくとも
            // ここまで (pos) は一致している。降りるたびに大きい値で上書きする。
            self.assign_subtree(node, pos, &mut out);
            if pos >= prompt.len() {
                break;
            }
            let child = self.nodes[node]
                .children
                .iter()
                .copied()
                .find(|&c| self.nodes[c].label.first() == Some(&prompt[pos]));
            let Some(child) = child else {
                // このノードの先は一致しない (部分木は pos までで確定)。
                break;
            };
            let m = common_prefix(&self.nodes[child].label, &prompt[pos..]);
            if m == self.nodes[child].label.len() {
                node = child;
                pos += m;
                continue;
            }
            // 辺の途中で食い違う: その子の部分木は pos+m まで一致。
            self.assign_subtree(child, pos + m, &mut out);
            break;
        }
        out
    }

    fn assign_subtree(&self, node: usize, lcp: usize, out: &mut [usize]) {
        let mut stack = vec![node];
        while let Some(n) = stack.pop() {
            for &s in &self.nodes[n].slots {
                if s < out.len() {
                    out[s] = lcp;
                }
            }
            stack.extend(self.nodes[n].children.iter().copied());
        }
    }
}

/// 2 列の共通前置き長 (短い方で頭打ち)。
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// 追い出すスロットを選ぶ (task 866、radix 有効時のみ)。
///
/// `metas` は `serve_flash` の [`SlotMeta`](super::SlotMeta) と同じ並び。
/// 規則: 今回のプロンプトとの共通前置きが **一番短い** スロット。同値なら
/// 最終使用が一番古いもの、さらに同値なら最小 index。
///
/// 分岐会話 (同じ system / 同じ履歴から別の続き) では、来たプロンプトと
/// 長い前置きを共有する文脈は「次に延長される見込みが高い」ので残し、
/// 無関係な文脈を先に捨てる。再利用の可否は変えない (再利用は従来どおり
/// 「キャッシュ全体が前置き」のときだけ) ので、出力は radix の有無で変わらない。
pub(crate) fn evict_victim(lcps: &[usize], used_at: &[u64]) -> Option<usize> {
    debug_assert_eq!(lcps.len(), used_at.len());
    let mut best: Option<usize> = None;
    for i in 0..lcps.len() {
        let better = match best {
            None => true,
            Some(b) => {
                lcps[i] < lcps[b]
                    || (lcps[i] == lcps[b]
                        && (used_at[i] < used_at[b] || (used_at[i] == used_at[b] && i < b)))
            }
        };
        if better {
            best = Some(i);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcp_brute(a: &[u32], b: &[u32]) -> usize {
        common_prefix(a, b)
    }

    /// 辺の分割 (共通前置き + 分岐 + 完全一致) が期待どおりの木になる。
    #[test]
    fn insert_splits_and_shares_edges() {
        let mut r = KvRadix::new();
        r.insert(&[1, 2, 3], 0);
        assert_eq!(r.node_count(), 2, "根 + 1 ノード");
        r.insert(&[1, 2, 4], 1);
        // 根 - [1,2] - (3) / (4)
        assert_eq!(r.node_count(), 4);
        assert_eq!(r.total_tokens(), 2 + 1 + 1, "共有した [1,2] は 1 回だけ数える");
        r.insert(&[1, 2], 2);
        // [1,2] のノードがスロット 2 を持つ (ノードは増えない)。
        assert_eq!(r.node_count(), 4);
        // 完全一致 (同じ列を別スロットとして入れる)。
        r.insert(&[1, 2, 3], 3);
        assert_eq!(r.node_count(), 4, "同じ列はノードを増やさない");

        let l = r.lcps(&[1, 2, 9], 4);
        assert_eq!(l, vec![2, 2, 2, 2]);
        let l = r.lcps(&[1, 2, 3, 5], 4);
        assert_eq!(l, vec![3, 2, 2, 3]);
        let l = r.lcps(&[9], 4);
        assert_eq!(l, vec![0, 0, 0, 0]);
        let l = r.lcps(&[], 4);
        assert_eq!(l, vec![0, 0, 0, 0]);
        let l = r.lcps(&[1, 2, 3], 4);
        assert_eq!(l, vec![3, 2, 2, 3], "プロンプトを使い切っても全長一致");
    }

    /// 空きスロット (`None`) は索引しない。
    #[test]
    fn empty_slots_are_not_indexed() {
        let r = KvRadix::from_slots(
            vec![Some(&[7u32, 8][..]), None, Some(&[7, 8, 9][..])].into_iter(),
        );
        let l = r.lcps(&[7, 8, 9, 10], 3);
        assert_eq!(l, vec![2, 0, 3]);
    }

    /// ランダムな列で `lcps` が総当りと一致する (索引の正しさの担保)。
    #[test]
    fn lcps_match_bruteforce() {
        // 決定的な LCG (テストの再現性のため)。語彙は 1..=3。
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut next = |m: u32| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % m as u64) as u32
        };
        for trial in 0..40 {
            let n_slots = 1 + (trial % 5);
            let seqs: Vec<Vec<u32>> = (0..n_slots)
                .map(|_| {
                    let n = next(12) as usize;
                    (0..n).map(|_| 1 + next(3)).collect()
                })
                .collect();
            let r = KvRadix::from_slots(seqs.iter().map(|v| Some(v.as_slice())));
            for _ in 0..20 {
                let n = next(14) as usize;
                let prompt: Vec<u32> = (0..n).map(|_| 1 + next(3)).collect();
                let want: Vec<usize> = seqs.iter().map(|q| lcp_brute(q, &prompt)).collect();
                assert_eq!(r.lcps(&prompt, n_slots), want, "trial {trial} prompt {prompt:?}");
            }
        }
    }

    /// 共通 system 前置きは 1 本のパスを共有し、ノード数が総トークン数より小さい。
    #[test]
    fn shared_system_prefix_shares_one_path() {
        let sys: Vec<u32> = (100..160).collect();
        let a: Vec<u32> = [sys.clone(), vec![1, 2, 3, 4]].concat();
        let b: Vec<u32> = [sys.clone(), vec![1, 2, 5, 6]].concat();
        let c: Vec<u32> = [sys.clone(), vec![9, 9, 9, 9]].concat();
        let r = KvRadix::from_slots([a.clone(), b.clone(), c.clone()].iter().map(|v| Some(v.as_slice())));
        // 総トークン - 重複して共有された分だけノードのラベルが短くなる。
        assert!(
            r.total_tokens() <= 60 + 4 + 4 + 4,
            "共有されていない: total_tokens={}",
            r.total_tokens()
        );
        assert!(
            r.node_count() <= 1 + 1 + 3 + 3,
            "ノードが多すぎる: {}",
            r.node_count()
        );
        // system だけのプロンプトは 3 本とも共通前置き 60。
        assert_eq!(r.lcps(&sys, 3), vec![60, 60, 60]);
        // 2 本が同じ枝 (1,2) を共有する。
        let p = [sys.clone(), vec![1, 2, 7]].concat();
        assert_eq!(r.lcps(&p, 3), vec![62, 62, 60]);
    }

    /// 追い出しは「前置きが一番遠い」→「古い」→「index 小」。
    #[test]
    fn evict_prefers_least_related() {
        assert_eq!(evict_victim(&[5, 0, 5], &[1, 2, 3]), Some(1), "無関係を先に");
        assert_eq!(evict_victim(&[4, 4, 4], &[9, 2, 5]), Some(1), "同値なら最古");
        assert_eq!(evict_victim(&[4, 4], &[7, 7]), Some(0), "同値なら index 小");
        assert_eq!(evict_victim(&[], &[]), None);
    }
}
