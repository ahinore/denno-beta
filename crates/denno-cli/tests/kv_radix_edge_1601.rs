//! task 1601: prefix KV スロットの radix 索引 (`src/kv_radix.rs`) の境界試験と、
//! turn-2 の短い差分 prefill (nt = 19..50) がエンジンの常駐経路の閾値
//! (`PF_RESIDENT_MIN_TOK = 64`) に対してどこに落ちるかの監査ピン。
//!
//! denno-cli は bin クレートなので、lib 経由では `kv_radix` を取り込めない。
//! 既存の `tests/config_906.rs` と同じく `#[path]` で実ファイルをそのまま
//! 取り込み、**実装そのもの** を叩く (再実装した写しを試験しない)。
//!
//! 固定する契約:
//!
//! 1. 索引 (`KvRadix`): 共通前置きの共有 / 辺の分割 / 完全一致 / 空列 /
//!    空きスロット (`None`) / 範囲外 slot index / 8 本 × 長い共通前置き。
//!    `lcps` は総当りの `common_prefix` と全一致 (決定的な擬似乱数で 100 回超)。
//! 2. 追い出し (`evict_victim`): 前置きが最短 → 最終使用が最古 → index 最小、
//!    という順序を独立に計算した答えと突き合わせる (単発ケース + ランダム)。
//! 3. フラグ (`kv_radix_enabled`): 既定 off。`1/true/yes/on` だけ on。
//! 4. 監査ピン: `DENNO_KV_RADIX` が効くのはスロット 3 本以上のときだけで、
//!    エンジンの常駐 prefill は「最初のチャンクが 64 トークン未満」を
//!    見送る。turn-2 の差分 nt=19..50 はこの閾値の下に必ず落ちる。
//!
//! # 昇格 (DENNO_KV_RADIX を既定 on にする) の条件 — 監査の結論
//!
//! この試験で CPU 側の正しさ (索引の値・追い出しの順序・フラグの既定 off) は
//! 固定できる。それでも既定 off を維持する理由と、昇格に必要な残りの条件:
//!
//! - radix の有無で **出力は変わらない** (再利用は従来どおり「キャッシュ全体が
//!   プロンプトの前置き」のときだけで、変わるのは追い出しの相手と
//!   lcp の求め方だけ)。したがって昇格の判定は「速くなるか」ではなく
//!   「同じ出力で、選択ログとスロットの残り方が改善するか」で行う。
//! - 昇格の前提: (a) `--kv-slots` が 3 以上であること (2 本では不活性)、
//!   (b) 同一プロンプト列で radix on/off の生成 md5 が一致することのラボ実測、
//!   (c) `[flash] ctx: slots=[...] pick=... radix(nodes=.. tok_sum=..)` の行が
//!   prompt 長に対して O(prompt+N) で出ることの確認 (巨大 prefix でも増えない)。
//! - 索引はリクエストごとに作り直すので、常駐メモリは増えない (ノードは
//!   最大 8 本 × 文脈長の `u32` 相当、prefill の秒に対して無視できる)。
//! - 既定 on に切り替えるときも `DENNO_KV_RADIX=0` で戻せる形を残す
//!   (このフラグは `0` / `false` / `no` / typo を off と読む)。
//!
//! # 短い prefill の閾値 (PF_RESIDENT_MIN_TOK) を下げる条件 — 監査の結論
//!
//! - turn-2 の差分は `reuse` を引いた残り (`prompt[reused..]`) で、実測の
//!   19..50 トークンは常駐経路の判定 `first_nt < 64` に必ず引っかかり、
//!   従来の CPU バッチ経路に落ちる (この試験が境界を 64 として pin する)。
//! - 閾値は「GEMM の起動 + 同期 + 往復の固定費」と「CPU の要素演算」の
//!   損益分岐で、task 145/154 の実測で 64。下げる判断は CPU 側の試験では
//!   出せない (GPU 実測が要る) ので、ここでは **下げない**。
//! - 下げる場合の条件: (a) `first_nt` を 19..50 にしたときの常駐経路の
//!   wall が CPU 経路より短いことの GPU ラボ実測 (同一バイナリ・同一 env で
//!   交互比較)、(b) `PF_RESIDENT_MIN_TOK` を env で戻せるゲートを足すこと
//!   (現在は定数。A/B と切り戻しのため)、(c) 64 未満で常駐経路に載ったときの
//!   logits が CPU 経路と一致すること (丸め順が変わるので許容差の根拠を
//!   REPORT に書く)。
//!
//! 試験は決定的・CPU のみ・環境変数を触らない。

#[path = "../src/kv_radix.rs"]
mod kv_radix;

use kv_radix::{evict_victim, kv_radix_enabled, KvRadix};

/// 2 列の共通前置き長 (総当り。実装の `common_prefix` の写しではなく、
/// 試験側の独立した期待値)。
fn common(a: &[u32], b: &[u32]) -> usize {
    let mut i = 0;
    while i < a.len() && i < b.len() && a[i] == b[i] {
        i += 1;
    }
    i
}

/// 決定的な擬似乱数 (xorshift64*)。テストの再現性のためだけに使う。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// `0..m` の値。
    fn below(&mut self, m: usize) -> usize {
        (self.next() % m as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// 1. 索引: 共有 / 分割 / 完全一致 / 空列
// ---------------------------------------------------------------------------

/// 共通前置きは 1 本の辺に圧縮され、分岐点で辺が割れる。
#[test]
fn insert_shares_prefix_then_splits_edge() {
    let mut r = KvRadix::new();
    r.insert(&[1, 2, 3], 0);
    assert_eq!(r.node_count(), 2, "根 + 1 ノード");
    assert_eq!(r.total_tokens(), 3);

    r.insert(&[1, 2, 4], 1);
    assert_eq!(r.node_count(), 4, "根 - [1,2] - (3)/(4)");
    assert_eq!(r.total_tokens(), 2 + 1 + 1, "[1,2] は 1 回だけ数える");

    // 分岐点のノード自身に載る列 (辺の途中で終わる)。
    r.insert(&[1, 2], 2);
    assert_eq!(r.node_count(), 4, "ノードは増えない");
    assert_eq!(r.lcps(&[1, 2, 3, 9], 3), vec![3, 2, 2]);
}

/// 同じ列を別スロットとして入れてもノードは増えない。
#[test]
fn insert_duplicate_sequence_adds_slot_only() {
    let mut r = KvRadix::new();
    r.insert(&[5, 6, 7], 0);
    let nodes = r.node_count();
    r.insert(&[5, 6, 7], 3);
    assert_eq!(r.node_count(), nodes);
    assert_eq!(r.lcps(&[5, 6, 7], 4), vec![3, 0, 0, 3], "空きスロットは 0");
    assert_eq!(r.lcps(&[5, 6], 4), vec![2, 0, 0, 2], "短いプロンプトで頭打ち");
}

/// 空列は根に載る (= lcp 0)。serve_flash は空の文脈を保存しないが、
/// 索引側で落ちないことを固定する。
#[test]
fn insert_empty_sequence_indexes_root() {
    let mut r = KvRadix::new();
    r.insert(&[], 1);
    assert_eq!(r.node_count(), 1);
    assert_eq!(r.total_tokens(), 0);
    assert_eq!(r.lcps(&[1, 2], 2), vec![0, 0]);
    assert_eq!(r.lcps(&[], 2), vec![0, 0]);
}

/// 辺の途中で食い違うプロンプトは、その子の部分木を「食い違う直前まで」で
/// 確定させる (共有した分だけが lcp)。
#[test]
fn lcps_divergence_mid_edge_marks_whole_subtree() {
    let r = KvRadix::from_slots(
        [
            Some(&[10u32, 11, 12, 13, 14][..]),
            Some(&[10, 11, 12, 20][..]),
        ]
        .into_iter(),
    );
    // 根 - [10,11,12] - (13,14)/(20)
    assert_eq!(r.lcps(&[10, 11, 12, 99], 2), vec![3, 3]);
    assert_eq!(r.lcps(&[10, 11, 99], 2), vec![2, 2]);
    assert_eq!(r.lcps(&[10, 99], 2), vec![1, 1]);
    assert_eq!(r.lcps(&[99], 2), vec![0, 0]);
}

/// 空きスロット (`None`) と、`n_slots` より大きい index は無視され panic しない。
#[test]
fn lcps_ignores_none_and_out_of_range_slots() {
    let r = KvRadix::from_slots(
        vec![Some(&[7u32, 8][..]), None, Some(&[7, 8, 9][..])].into_iter(),
    );
    assert_eq!(r.lcps(&[7, 8, 9, 10], 3), vec![2, 0, 3]);

    // 索引は slot 5 を持つが n_slots=3。範囲外は書き込まない。
    let mut r2 = KvRadix::new();
    r2.insert(&[1, 2], 5);
    assert_eq!(r2.lcps(&[1, 2], 3), vec![0, 0, 0]);
    assert_eq!(r2.lcps(&[1, 2], 6), vec![0, 0, 0, 0, 0, 2]);
}

/// 全スロットがプロンプトより短い (= 前置きとして全部使える) ケース。
#[test]
fn lcps_prompt_longer_than_every_slot() {
    let r = KvRadix::from_slots(
        [
            Some(&[1u32, 2][..]),
            Some(&[1, 2, 3][..]),
            Some(&[1, 2, 3, 4][..]),
        ]
        .into_iter(),
    );
    assert_eq!(r.lcps(&[1, 2, 3, 4, 5, 6], 3), vec![2, 3, 4]);
}

/// 8 本が 512 トークンの system 前置きを共有しても、1 本のパスに圧縮される。
#[test]
fn eight_slots_share_long_prefix_in_one_path() {
    let sys: Vec<u32> = (1000..1512).collect();
    let mut slots: Vec<Vec<u32>> = Vec::new();
    for i in 0..8u32 {
        let mut v = sys.clone();
        v.extend_from_slice(&[i, 100 + i, 200 + i, 300 + i]);
        slots.push(v);
    }
    let r = KvRadix::from_slots(slots.iter().map(|v| Some(v.as_slice())));
    assert_eq!(r.total_tokens(), 512 + 8 * 4, "共有分は 1 回だけ");
    assert_eq!(r.node_count(), 1 + 1 + 8, "根 + 共有ノード + 8 本の枝");
    assert_eq!(r.lcps(&sys, 8), vec![512; 8], "system だけなら全スロット 512");
    let mut p = sys.clone();
    p.extend_from_slice(&[3, 103]);
    let l = r.lcps(&p, 8);
    assert_eq!(l[3], 514, "枝 3 の続き");
    for (i, &x) in l.iter().enumerate() {
        if i != 3 {
            assert_eq!(x, 512, "スロット {i} は system まで");
        }
    }
}

/// 索引の値は総当りと一致する (決定的な擬似乱数で 8 スロット × 長さ 40 まで)。
#[test]
fn lcps_matches_bruteforce_fuzz() {
    let mut rng = Rng::new(0x1601_1601_1601_1601);
    for trial in 0..60 {
        let n_slots = 1 + trial % 8;
        let seqs: Vec<Vec<u32>> = (0..n_slots)
            .map(|_| {
                let n = rng.below(41);
                (0..n).map(|_| 1 + rng.below(4) as u32).collect()
            })
            .collect();
        let r = KvRadix::from_slots(seqs.iter().map(|v| Some(v.as_slice())));
        for _ in 0..25 {
            let n = rng.below(46);
            let prompt: Vec<u32> = (0..n).map(|_| 1 + rng.below(4) as u32).collect();
            let want: Vec<usize> = seqs.iter().map(|q| common(q, &prompt)).collect();
            assert_eq!(
                r.lcps(&prompt, n_slots),
                want,
                "trial {trial} prompt {prompt:?} slots {seqs:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 2. 追い出し: 前置きが最短 → 最古 → index 最小
// ---------------------------------------------------------------------------

/// 単発ケース (規則を名指しで固定する)。
#[test]
fn evict_prefers_shortest_lcp_then_oldest_then_smallest_index() {
    assert_eq!(evict_victim(&[5, 0, 5], &[1, 2, 3]), Some(1), "無関係を先に");
    assert_eq!(evict_victim(&[4, 4, 4], &[9, 2, 5]), Some(1), "同値なら最古");
    assert_eq!(evict_victim(&[4, 4], &[7, 7]), Some(0), "同値なら index 小");
    assert_eq!(evict_victim(&[0], &[0]), Some(0), "1 本でも選べる");
    assert_eq!(evict_victim(&[], &[]), None, "空なら選ばない");
}

/// 独立に計算した argmin と一致する (ランダム 200 ケース)。
#[test]
fn evict_matches_independent_argmin() {
    let mut rng = Rng::new(0xDEAD_1601_0000_0001);
    for _ in 0..200 {
        let n = rng.below(9);
        let lcps: Vec<usize> = (0..n).map(|_| rng.below(17)).collect();
        let used: Vec<u64> = (0..n).map(|_| rng.below(6) as u64).collect();
        let want = (0..n).min_by_key(|&i| (lcps[i], used[i], i));
        assert_eq!(
            evict_victim(&lcps, &used),
            want,
            "lcps={lcps:?} used={used:?}"
        );
    }
}

/// 前置きを共有するスロット (lcp が大きい) は、他に短い相手がいれば選ばれない。
/// 分岐会話で親を残す、という radix の追い出し規則の要点。
#[test]
fn evict_keeps_the_slot_sharing_the_longest_prefix() {
    let sys: Vec<u32> = (0..64).collect();
    let parent: Vec<u32> = [sys.clone(), vec![7, 8, 9, 10]].concat();
    let other: Vec<u32> = (200..240).collect();
    let r = KvRadix::from_slots(
        [Some(parent.as_slice()), Some(other.as_slice())].into_iter(),
    );
    let mut prompt = parent.clone();
    prompt.extend_from_slice(&[11, 12]);
    let l = r.lcps(&prompt, 2);
    assert_eq!(l, vec![68, 0]);
    assert_eq!(evict_victim(&l, &[1, 2]), Some(1), "親ではなく無関係を捨てる");
}

// ---------------------------------------------------------------------------
// 3. フラグ: 既定 off (opt-in)
// ---------------------------------------------------------------------------

#[test]
fn radix_flag_is_opt_in_only() {
    assert!(!kv_radix_enabled(None), "既定は off");
    for v in ["", "0", "false", "no", "off", "x", "2", " on", "on "] {
        assert!(!kv_radix_enabled(Some(v)), "{v:?} は off");
    }
    for v in ["1", "true", "yes", "on"] {
        assert!(kv_radix_enabled(Some(v)), "{v:?} は on");
    }
}

// ---------------------------------------------------------------------------
// 4. 監査ピン: turn-2 の短い差分 prefill と常駐経路の閾値
// ---------------------------------------------------------------------------

/// エンジン側の定数を実ファイルから読む (bin クレートから import できないため)。
fn engine_source() -> String {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../denno-engine/src/model/qwen4exp/prefill.rs");
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{} を読めない: {e}", p.display()))
}

/// `pub(crate) const PF_RESIDENT_MIN_TOK: usize = N;` の N。
fn parse_resident_min_tok(src: &str) -> usize {
    let at = src
        .find("const PF_RESIDENT_MIN_TOK")
        .expect("PF_RESIDENT_MIN_TOK の定義が無い");
    let rest = &src[at..];
    let eq = rest.find('=').expect("= が無い");
    let tail = &rest[eq + 1..];
    let semi = tail.find(';').expect("; が無い");
    tail[..semi]
        .trim()
        .parse::<usize>()
        .expect("数値として読めない")
}

/// 常駐 prefill の閾値は 64 で、境界判定は `first_nt < 64` の 1 か所。
#[test]
fn resident_prefill_threshold_is_64_and_uses_first_nt() {
    let src = engine_source();
    assert_eq!(parse_resident_min_tok(&src), 64, "閾値が変わっている");
    let squashed: String = src.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        squashed.contains("iffirst_nt<Self::PF_RESIDENT_MIN_TOK{"),
        "常駐経路の見送り判定 (first_nt) が変わっている"
    );
    assert!(
        squashed.contains("letfirst_nt=chunk.min(ctx.len());"),
        "first_nt の求め方が変わっている"
    );
}

/// `DENNO_KV_RADIX` はスロット 3 本以上のときだけ選択を radix にする。
#[test]
fn radix_gate_requires_three_slots() {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/serve_flash.rs");
    let src = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let squashed: String = src.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        squashed.contains("letradix_on=kv_radix_enabled(")
            && squashed.contains("&&n_slots>=3;"),
        "radix の有効条件 (スロット 3 本以上) が変わっている"
    );
    // 既定 off の経路では索引を作らない (then の前が false なら構築されない)。
    assert!(
        squashed.contains("letradix=radix_on.then(||{"),
        "radix 索引の構築がフラグで gate されていない"
    );
}

/// turn-2 の差分 (nt = 19..=50) は、親を全長再利用したあとの残り。
/// その nt は必ず常駐経路の閾値 64 未満なので、エンジンは CPU バッチ経路に
/// 落ちる。境界 (64 の直前で切れる) を両側から pin する。
#[test]
fn turn2_incremental_tokens_always_below_resident_threshold() {
    let min_tok = parse_resident_min_tok(&engine_source());
    let sys: Vec<u32> = (1..=48).collect();
    // 親の会話 (turn 1 のプロンプト + assistant 1 ターン) を 400 トークンとする。
    let mut parent = sys.clone();
    parent.extend(1000u32..1352);
    assert_eq!(parent.len(), 400, "親の長さの前提");

    // 4 スロット (1 本が親、3 本は別会話)。
    let others: Vec<Vec<u32>> = (0..3u32)
        .map(|i| {
            let mut v = vec![900 + i];
            v.extend((0..80).map(|k| 2000 + i * 100 + k));
            v
        })
        .collect();
    let mut slots: Vec<Option<&[u32]>> = vec![Some(parent.as_slice())];
    slots.extend(others.iter().map(|v| Some(v.as_slice())));
    let r = KvRadix::from_slots(slots.into_iter());

    for nt in 19..=50usize {
        let mut prompt = parent.clone();
        prompt.extend((0..nt).map(|k| 5000 + k as u32));
        let l = r.lcps(&prompt, 4);
        assert_eq!(l[0], parent.len(), "nt={nt}: 親を全長再利用できていない");
        let incremental = prompt.len() - l[0];
        assert_eq!(incremental, nt);
        assert!(
            incremental < min_tok,
            "nt={nt}: 常駐経路の閾値 {min_tok} に届いた (監査の前提が崩れた)"
        );
    }
    // 閾値のすぐ下 (63) は落ち、64 で常駐経路の判定に入る (off-by-one の pin)。
    assert!(63 < min_tok);
    assert!(64 >= min_tok);
    // 実測の nt 上限 (50) と閾値の差。
    assert!(min_tok > 50, "nt=50 が閾値以上になっている");
}
