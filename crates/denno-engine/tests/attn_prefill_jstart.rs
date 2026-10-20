//! s122 (P1-2 候補 4): `dno_attention_prefill` の **causal 鍵範囲をタイル単位で
//! 切り詰めてよいか** を CPU で判定するモデル。GPU も cuda feature も使わない。
//!
//! 結論 (s122 の調査結果): **block 単位の切り詰め (`jstart`) は不可**。
//! 詳しくは `docs/TASKS.md` の P1-2 候補 4 と `REPORT_S122.md` を参照。
//!
//! 添字の意味 (カーネルとホスト起動側で一致):
//!  - `i0 = blockIdx.x * PF_BQ` は **q 側の行番号** (0 <= i0 < n)。host の grid は
//!    `(n.div_ceil(PF_BQ), heads)` なので block 番号は q 行方向のブロック。
//!  - block の row r の **絶対位置**は `pos0 + i0 + r`、q の行 i の絶対位置は `pos0 + i`。
//!  - key の絶対位置は `j`。カーネルのマスクは `ja < t_total && ja <= pos0 + ia`
//!    (`ia = i0 + r0`, `ja = j0 + k0`) で、CPU 参照 (`full_attn_attend_head` を
//!    `t_total = pos0 + i + 1` で呼ぶ) の causal 界 `j <= pos0 + i` と同一。
//!
//! つまり **row r が読める key は `j <= pos0 + i0 + r`** で、`j <= pos0 + i0` から
//! 始まるわけではない。`pos0 + i0` を block 全体の下界として扱う (jstart) と、
//! r > 0 の row が本来見るべき key (`pos0 + i0 < j <= pos0 + i0 + r`) を落とす。
//! ここではその事実を、通ってはいけない性質 (無い) ではなく
//! **正しい性質** として固定する。

const CU_SRC: &str = include_str!("../kernels/kernels_elem.cu");

/// `#define NAME VALUE` を i64 として取り出す (単純な数値リテラルのみ)。
fn define_i64(src: &str, name: &str) -> i64 {
    let needle = format!("#define {name} ");
    for line in src.lines() {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix(&needle) else {
            continue;
        };
        let val = rest
            .split_whitespace()
            .next()
            .unwrap_or_else(|| panic!("#define {name} の値が無い"));
        return val
            .parse()
            .unwrap_or_else(|_| panic!("#define {name} の値 {val:?} が数値でない"));
    }
    panic!("#define {name} が kernels_elem.cu に無い");
}

/// カーネル側の値。片側だけ変えた事故を検出するため `.cu` の `#define` と照合する。
const PF_BQ: i32 = 64;
const PF_BK: i32 = 8;

/// cold 計測の形 (5,686 token の prefill・pos0 = 0)。
const COLD_N: i32 = 5_686;

/// カーネルのマスク述語そのもの (`ja < t_total && ja <= pos0 + ia`)。
/// `i` は q 側の行番号 (0 <= i < n)、key `j` は絶対位置。
fn kernel_mask(j: i32, pos0: i32, i: i32, t_total: i32) -> bool {
    j < t_total && j <= pos0 + i
}

/// CPU 参照の causal 界 (row の絶対位置 pos_abs から見える key は `j <= pos_abs`)。
fn ref_mask(j: i32, pos_abs: i32) -> bool {
    j <= pos_abs
}

/// 「block 全体の下界」として提案されていた jstart (下方向整列)。
fn proposed_jstart(pos0: i32, i0: i32) -> i32 {
    ((pos0 + i0) / PF_BK) * PF_BK
}

#[test]
fn defines_match_cuda_rs_mirrors() {
    assert_eq!(
        define_i64(CU_SRC, "PF_BQ"),
        PF_BQ as i64,
        "PF_BQ が .cu と不一致"
    );
    assert_eq!(
        define_i64(CU_SRC, "PF_BK"),
        PF_BK as i64,
        "PF_BK が .cu と一致しない"
    );
}

/// 前提の確認: カーネルのマスクは CPU 参照と同じ causal 界を作る。
///
/// これが成り立たないと以降の話が意味を持たないので、まずこれを固定する。
/// (pos0 > 0 の継続 prefill でも成り立つ: row の絶対位置は pos0 + i。)
#[test]
fn kernel_mask_equals_causal_reference() {
    let mut checked = 0usize;
    for &n in &[1, 2, 7, 63, 64, 65, 129] {
        for &pos0 in &[0, 1, 5, 64, 1000] {
            let t_total = pos0 + n;
            for i in 0..n {
                for j in 0..t_total {
                    assert_eq!(
                        kernel_mask(j, pos0, i, t_total),
                        ref_mask(j, pos0 + i),
                        "n={n} pos0={pos0} row={i} key={j} でマスクが参照と食い違う"
                    );
                    checked += 1;
                }
            }
        }
    }
    println!("kernel mask == causal reference on {checked} (row, key) pairs");
    assert!(checked > 100_000);
}

/// **回帰の核**: block の row r > 0 は `j <= pos0 + i0 + r` まで見る。
///
/// したがって「block は `pos0 + i0` 未満の key を一切見ない」という主張は偽で、
/// jstart で先頭タイルを切り落とすと、切り落とした範囲に row r の可視 key が
/// 残っている (= 出力が変わる) ことを明示する。
#[test]
fn block_lower_bound_claim_is_false() {
    // cold 構成の 2 番目の block (i0 = 64, pos0 = 0)
    let (n, pos0, i0) = (COLD_N, 0, 64);
    let t_total = pos0 + n;
    let js = proposed_jstart(pos0, i0);
    assert!(js > 0, "cold の block 2 は jstart > 0 になるはず (js={js})");

    // 切り落とされる範囲 [0, js) に、可視の key を持つ row が存在する
    let mut lost = 0usize;
    for r in 1..PF_BQ {
        let i = i0 + r;
        if i >= n {
            break;
        }
        for j in 0..js {
            if kernel_mask(j, pos0, i, t_total) {
                lost += 1;
            }
        }
    }
    assert!(
        lost > 0,
        "block 2 の切り落とし範囲に可視 key が無いと主張している (前提が変わった?)"
    );
    println!("cold block i0={i0}: {lost} (row, key) pairs would be dropped");

    // 具体的な反例: row r = 8 は key j = 71 を見るが、jstart = 64 で落ちる。
    let (r, j) = (8, 71);
    assert!(j < js || j >= js, "自明");
    let i = i0 + r;
    assert!(kernel_mask(j, pos0, i, t_total), "row {i} は key {j} を見る");
    assert!(j >= js, "key {j} が切り落とし範囲に入っていない (js={js})");

    // jstart の根拠になっていた「全 row 共通の下界」自体が存在しないことを固定する。
    // causal なので row r = 0 (絶対位置 pos0 + i0) は key 0..=pos0+i0 を **全部** 見る。
    // したがって pos0 + i0 は下界ではなく、単に 1 行分の上界にすぎない。
    for j in (0..js).step_by(7) {
        assert!(
            kernel_mask(j, pos0, i0, t_total),
            "row 0 が key {j} を見ていない? (causal の前提が崩れる)"
        );
    }
}

/// **安全な切り詰め規則**: タイル `[j0, j0 + PF_BK)` を丸ごと落とせるのは、
/// block の全 row (r = 0..PF_BQ) でマスクされているときだけ。その判定は row の
/// 最大界 `pos0 + i0 + PF_BQ - 1` を使う (最小界ではない)。
///
/// この規則で落とせるのは「末尾」タイルだけなので、先頭ループの開始点を
/// 上げる形 (jstart) では表現できない — これが候補 4 が成立しない理由。
#[test]
fn only_tiles_above_the_max_row_bound_are_fully_masked() {
    let mut fully_masked_at_head = 0usize;
    let mut fully_masked_total = 0usize;
    let mut checked_blocks = 0usize;

    for &n in &[129, 257, COLD_N] {
        for &pos0 in &[0, 64, 1000] {
            let t_total = pos0 + n;
            let mut i0 = 0;
            while i0 < n {
                let je = t_total.min(pos0 + i0 + PF_BQ);
                let mut j0 = 0;
                while j0 + PF_BK <= je {
                    // 全 row でマスク = row の最大界より上
                    let fully = (0..PF_BK).all(|kk| {
                        (0..PF_BQ).all(|r| {
                            let i = i0 + r;
                            i >= n || !kernel_mask(j0 + kk, pos0, i, t_total)
                        })
                    });
                    if fully {
                        fully_masked_total += 1;
                        // 先頭から連続して全マスクになる (= jstart で表現できる) か
                        let all_before = (0..j0).all(|j| {
                            (0..PF_BK).all(|kk| {
                                (0..PF_BQ).all(|r| {
                                    let i = i0 + r;
                                    i >= n || !kernel_mask(j + kk, pos0, i, t_total)
                                })
                            })
                        });
                        if all_before {
                            fully_masked_at_head += 1;
                        }
                    }
                    j0 += PF_BK;
                }
                checked_blocks += 1;
                i0 += PF_BQ;
            }
        }
    }
    println!(
        "blocks={checked_blocks} fully-masked tiles={fully_masked_total} \
         (of which form a masked prefix: {fully_masked_at_head})"
    );
    // causal なので全マスクのタイルは末尾側にしか現れない = 先頭で落とせるものは無い
    assert_eq!(
        fully_masked_at_head, 0,
        "先頭から連続して全マスクのタイルがある = jstart 型の切り詰めが成立し得る"
    );
}

/// 数値の大きさ: 切り詰めると attention 出力がどれだけ変わるか (fp32 の toy 計算)。
///
/// 「ビット一致」どころか、緩和枠 (1e-3 abs, R3) すら大きく外れることを示す。
#[test]
fn truncation_changes_attention_far_beyond_the_relaxed_bound() {
    // 決定的な擬似乱数 (外部 crate を使わない)
    let mut s = 12345u64;
    let mut next = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    };
    let (n, pos0, dh) = (256usize as i32, 0i32, 8i32);
    let t_total = pos0 + n;
    let k: Vec<f32> = (0..t_total * dh).map(|_| next()).collect();
    let v: Vec<f32> = (0..t_total * dh).map(|_| next()).collect();
    let q: Vec<f32> = (0..n * dh).map(|_| next()).collect();

    let attend = |i: i32, from: i32| -> Vec<f32> {
        let hi = pos0 + i; // causal 界
        let mut scores = Vec::new();
        for j in from..=hi {
            let mut d = 0.0f32;
            for t in 0..dh {
                d += q[(i * dh + t) as usize] * k[(j * dh + t) as usize];
            }
            scores.push(d * (1.0 / (dh as f32).sqrt()));
        }
        let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = scores.iter().map(|x| (x - mx).exp()).collect();
        let sum: f32 = e.iter().sum();
        let mut out = vec![0.0f32; dh as usize];
        for (idx, j) in (from..=hi).enumerate() {
            let p = e[idx] / sum;
            for t in 0..dh {
                out[t as usize] += p * v[(j * dh + t) as usize];
            }
        }
        out
    };

    let mut worst = 0.0f32;
    let mut i0 = 0;
    while i0 < n {
        let js = proposed_jstart(pos0, i0);
        for r in 0..PF_BQ {
            let i = i0 + r;
            if i >= n {
                break;
            }
            let a = attend(i, 0);
            let b = attend(i, js);
            for t in 0..dh {
                worst = worst.max((a[t as usize] - b[t as usize]).abs());
            }
        }
        i0 += PF_BQ;
    }
    println!("max |att(full) - att(truncated)| = {worst}");
    assert!(
        worst > 1e-2,
        "切り詰めの差が小さすぎる ({worst}) — 数値の主張をやり直すこと"
    );
}