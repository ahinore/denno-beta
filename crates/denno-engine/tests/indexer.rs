//! P-MOE-D2: QSA indexer (k-pool 選択) の CPU 参照実装のテスト。
//!
//! GPU / CUDA / 推論経路は一切通さない。`--features cuda` 無しで回ること。
//! 参照値はハードコード (固定 seed の擬似乱数 + f32 の逐次演算で独立に算出)。

use denno_engine::model::indexer::{pool_keys, qsa_select, score_pools, select_cells, IndexerParams};

/// 決定的な擬似乱数 (xorshift32)。外部クレートに依存しない。
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// [0, 1) の f32 (上位 24 bit を使うので丸めが入らない)。
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16777216.0)
    }
    /// [-1, 1) の f32。
    fn next_signed(&mut self) -> f32 {
        self.next_f32() * 2.0 - 1.0
    }
}

fn params(n_head: usize, idx_dim: usize, top_k: usize, kpool: usize) -> IndexerParams {
    IndexerParams {
        n_head,
        idx_dim,
        top_k,
        kpool,
        eps: 1e-6,
    }
}

/// 8 セル / kpool 4 -> 2 ブロック。平均がセル昇順加算 -> 1/4 の乗算とビット一致。
#[test]
fn pool_keys_matches_ascending_mean() {
    let d = 3usize;
    let p = params(4, d, 2048, 4);
    let raw: Vec<f32> = (0..8 * d).map(|i| (i as f32) * 0.25 - 3.5).collect();
    let out = pool_keys(&raw, 8, &p);
    assert_eq!(out.len(), 2 * d);
    for j in 0..d {
        // ((a + b) + c) + d の順に足して 0.25 を掛ける。
        let want0 = (((raw[j] + raw[d + j]) + raw[2 * d + j]) + raw[3 * d + j]) * 0.25f32;
        let want1 = (((raw[4 * d + j] + raw[5 * d + j]) + raw[6 * d + j]) + raw[7 * d + j]) * 0.25f32;
        assert_eq!(out[j].to_bits(), want0.to_bits(), "ブロック 0 の次元 {}", j);
        assert_eq!(out[d + j].to_bits(), want1.to_bits(), "ブロック 1 の次元 {}", j);
    }
}

/// 6 セル (末尾が 2 セル) のとき、末尾ブロックは 2 要素平均になる。
#[test]
fn pool_keys_tail_block_uses_member_count() {
    let d = 2usize;
    let p = params(4, d, 2048, 4);
    let raw: Vec<f32> = vec![
        1.0, 2.0, // セル 0
        3.0, 4.0, // セル 1
        5.0, 6.0, // セル 2
        7.0, 8.0, // セル 3
        9.0, 10.0, // セル 4 (末尾ブロック)
        11.0, 12.0, // セル 5 (末尾ブロック)
    ];
    let out = pool_keys(&raw, 6, &p);
    assert_eq!(out.len(), 2 * d, "6 セル / kpool 4 は 2 ブロック");
    assert_eq!(out[0], 4.0);
    assert_eq!(out[1], 5.0);
    // (9 + 11) / 2, (10 + 12) / 2。kpool で割る (4 で割る) と 5.0/5.5 になり区別できる。
    assert_eq!(out[2], 10.0);
    assert_eq!(out[3], 11.0);
    assert_eq!(out[2].to_bits(), ((9.0f32 + 11.0f32) * 0.5f32).to_bits());
    assert_eq!(out[3].to_bits(), ((10.0f32 + 12.0f32) * 0.5f32).to_bits());
}

/// 負の dot は relu で 0 になり、スコアに寄与しない (head 間で符号は打ち消されない)。
#[test]
fn score_pools_negative_dot_is_rectified() {
    let d = 4usize;
    let p = params(2, d, 2048, 4);
    // head 0 は正、head 1 は負の dot になる。
    let q = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let pooled_neg = vec![1.0, -1.0, 0.0, 0.0];
    let pooled_zero = vec![1.0, 0.0, 0.0, 0.0];
    let s_neg = score_pools(&q, &pooled_neg, 1, &p);
    let s_zero = score_pools(&q, &pooled_zero, 1, &p);
    assert_eq!(s_neg.len(), 1);
    // head 1 の寄与が 0 なので、pooled の 2 次元目の符号はスコアを変えない。
    assert_eq!(s_neg[0].to_bits(), s_zero[0].to_bits());
    // 重みは 1/sqrt(4) = 0.5 で、head 0 の dot は 1.0。
    assert_eq!(s_neg[0], 0.5);
    // 絶対値の和 (1.0) でも全 head の和 (1.0 * 0.5 + (-1.0) * 0.5 = 0.0) でもない。
    assert_ne!(s_neg[0], 1.0);
    assert_ne!(s_neg[0], 0.0);
}

/// 手計算の既知値 (idx_dim=4, n_head=2) を 1e-6 で確認。
#[test]
fn score_pools_known_values() {
    let d = 4usize;
    let p = params(2, d, 2048, 4);
    let q = vec![1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0];
    let pooled = vec![
        // ブロック 0: dot0 = 1*1 + 2*0.5 = 2.0, dot1 = 1*2 + (-1)*1 = 1.0
        1.0, 0.5, 2.0, 1.0, // ブロック 1: dot0 = -1.0 -> 0, dot1 = 0.5 - 0.5 = 0
        -1.0, 0.0, 0.5, 0.5,
    ];
    let s = score_pools(&q, &pooled, 2, &p);
    // (relu(2.0) + relu(1.0)) * 1/sqrt(4) = 3.0 * 0.5 = 1.5
    assert!((s[0] - 1.5).abs() < 1e-6, "score[0]={}", s[0]);
    // (relu(-1.0) + relu(0.0)) * 0.5 = 0.0
    assert!((s[1] - 0.0).abs() < 1e-6, "score[1]={}", s[1]);
}

/// 同値スコアのときは小さいブロック番号が勝つ。
#[test]
fn select_cells_tie_prefers_lower_block() {
    let p = params(4, 8, 4, 4); // top_k / kpool = 1 ブロックだけ選ぶ
    let scores = [0.5f32, 0.5f32];
    let sel = select_cells(&scores, 8, &p);
    assert_eq!(sel, vec![0, 1, 2, 3], "同値ならブロック 0 が勝つ");
}

/// n_pool が top_k/kpool 以下のときは全ブロックが選ばれる。
#[test]
fn select_cells_keeps_all_when_budget_covers_pools() {
    let p = params(4, 8, 2048, 4); // 2048 / 4 = 512 ブロック分の予算
    let scores = [0.1f32, 0.9, -0.3];
    let sel = select_cells(&scores, 12, &p);
    assert_eq!(sel, (0u32..12).collect::<Vec<u32>>());
}

/// 出力は昇順で重複が無い。選ばれないブロックのセルは落ち、末尾の未完成ブロックは常に残る。
#[test]
fn select_cells_ascending_unique_and_keeps_tail() {
    let p = params(4, 8, 8, 4); // 予算 2 ブロック
    // 4 ブロック (0,1,2 が完成 / 3 が 2 セルの未完成)。スコアは 0 と 1 が上位。
    let scores = [0.9f32, 0.8, 0.1, 0.05];
    let sel = select_cells(&scores, 14, &p);
    assert_eq!(sel, vec![0, 1, 2, 3, 4, 5, 6, 7, 12, 13]);
    for w in sel.windows(2) {
        assert!(w[0] < w[1], "昇順でない: {:?}", w);
    }
    assert!(sel.iter().all(|&c| (c as usize) < 14));
}

/// 境界: n_cells = 0 で panic せず空を返す。
#[test]
fn empty_input_returns_empty() {
    let p = params(4, 8, 2048, 4);
    assert!(pool_keys(&[], 0, &p).is_empty());
    assert!(select_cells(&[], 0, &p).is_empty());
    let q = vec![0.0f32; 4 * 8];
    assert!(qsa_select(&[], &q, 0, &p).is_empty());
}

/// ランダム入力での決定的な参照値 (seed 99991)。
///
/// 4 head / idx_dim 8 / 14 セル / kpool 4 / top_k 8 -> n_pool 4, n_top 2。
/// 期待値は f32 の逐次演算 (セル昇順加算 -> 1/要素数 -> head 昇順 relu*dot) を
/// 別実装で回して得たビット列。
#[test]
fn qsa_select_deterministic_reference() {
    let (n_head, d, n_cells, kpool, top_k) = (4usize, 8usize, 14usize, 4usize, 8usize);
    let p = params(n_head, d, top_k, kpool);
    let mut rng = Rng::new(99991);
    let raw: Vec<f32> = (0..n_cells * d).map(|_| rng.next_signed()).collect();
    let q: Vec<f32> = (0..n_head * d).map(|_| rng.next_signed()).collect();

    let pooled = pool_keys(&raw, n_cells, &p);
    assert_eq!(pooled.len(), 4 * d);
    let scores = score_pools(&q, &pooled, 4, &p);
    let want_scores: [u32; 4] = [0x3def1c15, 0x3e6421b6, 0x3e7d2c73, 0x3e13f6c4];
    for (j, want) in want_scores.iter().enumerate() {
        assert_eq!(scores[j].to_bits(), *want, "score[{}]={} が参照値と不一致", j, scores[j]);
    }

    let sel = qsa_select(&raw, &q, n_cells, &p);
    assert_eq!(sel, vec![4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    // ブロック 0 (セル 0-3) は選ばれず落ちる。ブロック 3 は未完成の末尾なので残る。
    assert!(!sel.contains(&0));
    assert!(sel.contains(&12) && sel.contains(&13));
}

/// 実ファイルの形 (n_head 4 / idx_dim 128 / top_k 2048 / kpool 4) での本数確認。
#[test]
fn qsa_select_real_shape_cell_count() {
    let (n_head, d, n_cells, kpool, top_k) = (4usize, 128usize, 8192usize, 4usize, 2048usize);
    let p = params(n_head, d, top_k, kpool);
    let mut rng = Rng::new(20261026);
    let raw: Vec<f32> = (0..n_cells * d).map(|_| rng.next_signed()).collect();
    let q: Vec<f32> = (0..n_head * d).map(|_| rng.next_signed()).collect();
    let sel = qsa_select(&raw, &q, n_cells, &p);
    // n_pool = 2048, n_top = min(2048, 2048/4) = 512 -> 512 * 4 = 2048 セル。
    assert_eq!(sel.len(), 2048);
    for w in sel.windows(2) {
        assert!(w[0] < w[1]);
    }
    assert!(sel.iter().all(|&c| (c as usize) < n_cells));
}
