//! QSA indexer (k-pool 選択) の CPU 参照実装 (P-MOE-D2)。
//!
//! 出典: llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) の QSA indexer と、
//! 同リポジトリ `ggml/src/ggml-cpu/ops.cpp` の lightning indexer の式を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! QSA (indexer) は「full attention 層がどの KV セルを見るか」を決める。
//! 生キーを `kpool` セルごとに平均してブロックキーを作り、ブロック単位で
//! スコアを付けて上位ブロックを選び、選ばれたブロックのセル列を返す。
//! 推論経路には接続しない (GPU 実装のビット一致基準)。
//!
//! 演算順序 (この順序自体が数値契約):
//!
//! 1. ブロック集約: ブロック内の生キーを **セル昇順に逐次加算** し、
//!    `1/要素数` を **乗算** する (参照実装の `ggml_scale` と同じ。
//!    除算ではなく逆数の乗算)。末尾の未完成ブロックは要素数が `kpool` 未満で、
//!    その実要素数で割る。
//! 2. RMSNorm と RoPE は **呼び出し側の責務**(本モジュールは平均まで)。
//!    参照実装では集約後に `index_k_norm` と RoPE (ブロック先頭セルの位置) を当てる。
//!    クエリ側は `index_q_norm` と RoPE (トークン位置) を当ててから渡す。
//!    `IndexerParams::eps` はその呼び出し側 RMSNorm に渡す値で、本モジュール内では
//!    参照しない。
//! 3. スコア: `score[p] = sum_h relu(dot(q_h, pooled_p)) * (1/sqrt(idx_dim))`。
//!    加算は head 昇順、dot は k 昇順の逐次 f32 加算。relu は head ごとの dot の後。
//!    参照実装は全 head の重みを `1/sqrt(idx_dim)` に固定した lightning indexer。
//! 4. 選択: `n_top_pool = min(n_pool, top_k / kpool)` 個をスコア降順で選ぶ。
//!    同値はブロック番号の昇順 (小さい方が勝つ)。選ばれたブロックの全セルと、
//!    未完成の末尾ブロックのセルを残す。出力は昇順・重複なし。
//!
//! 決定性: 単スレッド・f64 不使用・固定加算順序なので、同一入力なら常にビット一致。
//! 並列化する場合は加算順序を変えないこと。

/// QSA indexer のパラメータ (GGUF メタデータ由来の値)。
#[derive(Debug, Clone, Copy)]
pub struct IndexerParams {
    /// indexer の head 数 (`indexer.n_head`)。実ファイルは 4。
    pub n_head: usize,
    /// head あたりのキー長 (`indexer.key_length`)。実ファイルは 128。
    pub idx_dim: usize,
    /// 残す生セル数の上限 (`indexer.top_k`)。ブロック数ではない。実ファイルは 2048。
    pub top_k: usize,
    /// ブロックサイズ (`compress_ratios` の QSA 層の値)。実ファイルは 4。
    pub kpool: usize,
    /// 呼び出し側 RMSNorm の eps (本モジュール内では参照しない)。
    pub eps: f32,
}

/// セル数からブロック数 (末尾の未完成ブロックを含む) を出す。
fn n_pools_of(n_cells: usize, kpool: usize) -> usize {
    if kpool == 0 {
        0
    } else {
        n_cells / kpool + usize::from(n_cells % kpool != 0)
    }
}

/// 生キー列 (セル順) をブロック平均する。末尾の未完成ブロックも平均する (要素数が kpool 未満)。
///
/// `raw` は `[n_cells][idx_dim]` の行優先。戻り値は `[n_pool][idx_dim]`
/// (`n_pool = ceil(n_cells / kpool)`)。`n_cells = 0` なら空を返す。
///
/// 平均は「ブロック内のセル昇順の逐次加算 -> `1/要素数` の乗算」。
/// 参照実装は `ggml_add` の鎖 + `ggml_scale(1/kpool)` で同じ順序。
pub fn pool_keys(raw: &[f32], n_cells: usize, p: &IndexerParams) -> Vec<f32> {
    assert!(p.kpool >= 1, "kpool が 0");
    assert_eq!(
        raw.len(),
        n_cells * p.idx_dim,
        "raw は [n_cells][idx_dim] (n_cells={}, idx_dim={})",
        n_cells,
        p.idx_dim
    );
    let d = p.idx_dim;
    let np = n_pools_of(n_cells, p.kpool);
    let mut out = vec![0.0f32; np * d];
    for b in 0..np {
        let base = b * p.kpool;
        let members = (n_cells - base).min(p.kpool);
        // 1 本目はコピー、以降はセル昇順に積む (0.0 から足し始めない: 参照実装と同じ)。
        let out_row = &mut out[b * d..(b + 1) * d];
        out_row.copy_from_slice(&raw[base * d..(base + 1) * d]);
        for m in 1..members {
            let row = &raw[(base + m) * d..(base + m + 1) * d];
            for i in 0..d {
                out_row[i] += row[i];
            }
        }
        let inv = 1.0f32 / members as f32;
        for v in out_row.iter_mut() {
            *v *= inv;
        }
    }
    out
}

/// スコア。`q` は `[n_head][idx_dim]`、`pooled` は `[n_pool][idx_dim]`。
///
/// `score[j] = sum_h relu(dot(q_h, pooled_j)) * (1/sqrt(idx_dim))`。
/// head 昇順に `relu * w` を積む。relu は head ごとの dot の後 (head 間で符号は
/// 打ち消されない)。マスクは呼び出し側の責務 (CPU 参照では keep 集合を返す)。
pub fn score_pools(q: &[f32], pooled: &[f32], n_pool: usize, p: &IndexerParams) -> Vec<f32> {
    assert_eq!(
        q.len(),
        p.n_head * p.idx_dim,
        "q は [n_head][idx_dim] (n_head={}, idx_dim={})",
        p.n_head,
        p.idx_dim
    );
    assert_eq!(
        pooled.len(),
        n_pool * p.idx_dim,
        "pooled は [n_pool][idx_dim] (n_pool={}, idx_dim={})",
        n_pool,
        p.idx_dim
    );
    let d = p.idx_dim;
    // 参照実装は head 重みを 1/sqrt(idx_dim) で埋めたテンソルを作ってから掛ける。
    let w = 1.0f32 / (d as f32).sqrt();
    let mut out = vec![0.0f32; n_pool];
    for (j, o) in out.iter_mut().enumerate() {
        let k = &pooled[j * d..(j + 1) * d];
        let mut acc = 0.0f32;
        for h in 0..p.n_head {
            let qh = &q[h * d..(h + 1) * d];
            let mut dot = 0.0f32;
            for i in 0..d {
                dot += qh[i] * k[i];
            }
            // MAX(dot, 0) と同じ: NaN は 0 側に落ちる。
            let relu = if dot > 0.0 { dot } else { 0.0 };
            acc += relu * w;
        }
        *o = acc;
    }
    out
}

/// 残すセルのインデックスを昇順で返す。
///
/// `scores` は `[n_pool]` (`n_pool = ceil(n_cells / kpool)`、`pool_keys` の出力ブロック数)。
/// `n_top_pool = min(n_pool, top_k / kpool)` 個のブロックをスコア降順で選び、
/// 同値はブロック番号の昇順で勝たせる。選ばれたブロックの全セルと、
/// 未完成の末尾ブロック (`n_cells % kpool` 個) を残す。`n_cells = 0` なら空。
pub fn select_cells(scores: &[f32], n_cells: usize, p: &IndexerParams) -> Vec<u32> {
    assert!(p.kpool >= 1, "kpool が 0");
    let np = n_pools_of(n_cells, p.kpool);
    assert_eq!(scores.len(), np, "scores は [n_pool] (n_pool={})", np);

    let n_top = (p.top_k / p.kpool).min(np);
    let mut used = vec![false; np];
    let mut keep = vec![false; n_cells];
    for _ in 0..n_top {
        // 走査をブロック番号昇順に行い、厳密比較 `>` で更新するので
        // 同値なら先に見つかった (= 小さい番号) が残る。
        let mut best: Option<usize> = None;
        for j in 0..np {
            if used[j] {
                continue;
            }
            match best {
                None => best = Some(j),
                Some(b) => {
                    if scores[j] > scores[b] {
                        best = Some(j);
                    }
                }
            }
        }
        let b = best.expect("n_top <= n_pool なので候補は必ず残る");
        used[b] = true;
        let base = b * p.kpool;
        let end = (base + p.kpool).min(n_cells);
        for c in base..end {
            keep[c] = true;
        }
    }

    // 未完成の末尾ブロックは選択に関わらず常に残す (indexer_kpool_select_tail)。
    let tail = (n_cells / p.kpool) * p.kpool;
    for c in tail..n_cells {
        keep[c] = true;
    }

    (0..n_cells)
        .filter(|&c| keep[c])
        .map(|c| c as u32)
        .collect()
}

/// 参照実装の順序を 1 関数にまとめたもの (pool -> score -> select)。
///
/// `raw_keys` は `[n_cells][idx_dim]` の生キー列 (norm も rope も受けない生の値)。
/// `q` は `[n_head][idx_dim]` で、呼び出し側が head ごとの RMSNorm と RoPE を
/// 済ませたものを渡す。戻り値は残すセルの昇順インデックス列。
pub fn qsa_select(raw_keys: &[f32], q: &[f32], n_cells: usize, p: &IndexerParams) -> Vec<u32> {
    let pooled = pool_keys(raw_keys, n_cells, p);
    let np = n_pools_of(n_cells, p.kpool);
    let scores = score_pools(q, &pooled, np, p);
    select_cells(&scores, n_cells, p)
}
