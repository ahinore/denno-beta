// s90 (P1-1): 2 launch スケジュールの帳簿 (ホスト側・GPU 不要)
//
// 背景:
//   chunked 経路は pass 1 (W/M) → pass 2 (Sinit の直列伝播) → pass 3 (逐次 AR)
//   の 3 launch + 3 sync。s89 は pass 2 を pass 3 の chunk 0 ブロックへ折り畳む
//   カーネル (`dno_gdn_chunk_scan_hoist`) を足し、chunk 境界の状態をグラフメモリ
//   `s_rows[t]` に置いて跨ごうとした。**しかし折り畳みの前提は常駐の連続性**で、
//   今のモデル forward にはそれが無い (下の「s90 の結論」)。
//
// 折り畳みの前提 (カーネルが正しいために必要なもの):
//   hoist 版の chunk ci (ci >= 1) は Sinit を行 `ci*chunk_len - 1` から読む。
//   その行を書くのは**別 launch** でなければならない (同一 launch 内のブロック間
//   には順序保証が無い)。書き手になり得るのは
//     (a) 前の層の同じ呼び出し (層 → 層の常駐), または
//     (b) 同じ層の直前の区間 (区間 → 区間の常駐)
//   のどちらかしかない。
//
// s90 の結論 (このツリーの実測に基づく・偽らない):
//   (a) 不成立。qwen35 の GDN 層は連続しておらず、隣り合う GDN 層のあいだには
//       full-attention 層がある。full 層は GDN の launch を出さないので、GDN 層 i
//       の s_rows を読む時点で書き手 launch は存在しない。
//   (b) 不成立。`KvCache::gdn` のホスト状態が `cache.advance(n)` のたびに区間
//       境界と一致するので、各層のスキャン区間はツリー内で必ず [lo, n) (n は
//       リクエストから見た現在の長さ) を覆い、区間の先頭は「前の forward の
//       末尾」= ホスト経由。すなわち隣接する区間は常にホストで切れる。
//   よって現状の forward では hoist 版を 1 回も走らせてはならない
//   (`route_hoist` は常に false を返す)。3 launch 経路が唯一の正しい経路である。
//
//   ここに帳簿 (どの行を誰が書く / 誰が読む) を残すのは、将来 (a) か (b) の
//   どちらかを実際に実装したときに、カーネルの式・バッファ確保・テストが同じ
//   定義を見るため。**帳簿が健全であることは経路を走らせる根拠にならない。**
//
// 帳簿の不変条件 (カーネルと同じ):
//   - chunk ci は行 `begin_row(ci) = ci*chunk_len - 1` を読む
//     (ci = 0 は `ssm_in` を読む・s_rows は読まない)。
//   - chunk ci が書く行は `[wstart, write_end)`:
//       非最終 chunk → 自分の最終行 t1-1 の 1 行だけ (次の chunk が読む)
//       最終 chunk   → [t0, n) の全部 (行 n-1 の書き手はこれ 1 つ)
//     中間行は誰も読まないので書かない。
//     **この規則は s89 の版からの訂正**。s89 は「t >= t0+chunk_len を書く」と
//     していたが、t1 = t0+chunk_len なので範囲が常に空で、境界行が 1 行も
//     書かれなかった (= 全 chunk が古い値から走る)。カーネル側も同じ規則に
//     直してある (`dno_gdn_chunk_scan_hoist` のコメント参照)。
//
// 注意 — 帳簿が健全でも経路は走らない:
//   「読む行を別 launch が書く」という前提は、下の (a)(b) がどちらも不成立
//   なので今は満たせない。さらに **中間行を書かない規則自体も、2 launch 経路の
//   前提の上でしか意味を持たない**。この前提が解けるまでは 3 launch 経路が
//   唯一の正しい経路である (`route_hoist` は常に false)。

use super::chunk_plan;

/// chunked 経路の chunk 分割 (CPU 参照と共有する唯一の定義)。
pub const CHUNK_MIN_TOKENS: usize = 2048;
pub const CHUNK_TOKENS: usize = 1024;
pub const CHUNK_MAX: usize = 16;

/// 2 launch 経路の境界行バッファ `s_rows = [n, nv, dk, dv]` の上限。
pub const SROWS_CAP_BYTES: usize = 1 << 20;

/// `scan_chunked_core` / GPU カーネルが使う分割。
pub fn plan(n: usize) -> (usize, usize) {
    chunk_plan(n, CHUNK_MIN_TOKENS, CHUNK_TOKENS, CHUNK_MAX)
}

/// 2 launch 経路の状態バッファ `[n, nv, dk, dv]` のバイト数。溢れたら None。
pub fn srows_bytes(n: usize, nv: usize, dk: usize, dv: usize) -> Option<usize> {
    n.checked_mul(nv)?
        .checked_mul(dk)?
        .checked_mul(dv)?
        .checked_mul(std::mem::size_of::<f32>())
}

/// 2 launch 経路の形式条件: 境界行バッファが [`SROWS_CAP_BYTES`] 以下。
pub fn fits(n: usize, nv: usize, dk: usize, dv: usize) -> bool {
    match srows_bytes(n, nv, dk, dv) {
        Some(b) => b <= SROWS_CAP_BYTES,
        None => false,
    }
}

/// chunk ci が書きはじめるトークン行 (この行以降だけ書いてよい)。
///
/// s90 訂正: 「非最終 chunk は t0+chunk_len 以降」では境界行が 1 行も書かれない
/// (t1 = t0+chunk_len なので範囲が空になる)。正しくは
///   - 非最終 chunk: 自分の最終行 t1-1 の 1 行だけ
///   - 最終 chunk:   [t0, n) の全部
/// を書く。戻り値は「書きはじめ」ので、非最終 chunk は t1-1、最終 chunk は t0。
/// 呼び出し側は [`write_end`] と組で使う (非最終 chunk は 1 行だけになる)。
pub fn wstart(ci: usize, chunk_len: usize, is_last: bool, n: usize) -> usize {
    if is_last {
        ci * chunk_len
    } else {
        n.min(ci * chunk_len + chunk_len).saturating_sub(1)
    }
}

/// chunk ci が書き終わる行 (排他)。
pub fn write_end(ci: usize, chunk_len: usize, is_last: bool, n: usize) -> usize {
    if is_last {
        n
    } else {
        n.min(ci * chunk_len + chunk_len).saturating_sub(1) + 1
    }
}

/// chunk ci が Sinit として読む行 (ci = 0 は None = `ssm_in` を読む)。
pub fn begin_row(ci: usize, chunk_len: usize) -> Option<usize> {
    if ci == 0 {
        None
    } else {
        Some(ci * chunk_len - 1)
    }
}

/// 2 launch 経路を使えるか。
///
/// `has_cross_launch_writer` は「この層の s_rows の行を、**別 launch** が書いて
/// くれること」を呼び出し側が保証できるか。今のモデル forward ではどの層でも
/// 保証できない (モジュール先頭の「s90 の結論」) ので、`gdn_scan_chunked_impl`
/// は常に false を渡す。将来 (a) 層常駐 か (b) 区間常駐 を実装したときだけ、
/// 実際に保証できる条件で true を渡す。
///
/// 形式条件 (`fits`) は「バッファが 1MiB に収まるか」だけを見る — これは
/// 保証の代わりにはならない (収まっても書き手が居なければ競合する)。
pub fn route_hoist(
    n: usize,
    nv: usize,
    dk: usize,
    dv: usize,
    has_cross_launch_writer: bool,
) -> bool {
    n > 0 && has_cross_launch_writer && fits(n, nv, dk, dv)
}

/// 層 i の tile を層 i+1 の tile へ そのまま渡す経路か (層 → 層の常駐)。
/// 同じ区間 [lo, n) で走り、かつ区間の先頭が 0 でない (= 前の層から状態を
/// 受け取っている) こと。先頭が 0 なら前段はホストで、末尾もホストへ戻る。
pub fn is_contig(lo: usize, n: usize, next_lo: usize, next_n: usize) -> bool {
    lo == next_lo && n == next_n && lo > 0
}

/// 層 i の次に状態をそのまま受け取れる層か (GDN 層を 0..n_gdn に詰めた順)。
/// 最後の GDN 層の次は無い (ホストへ戻す) ので false。
pub fn is_successor(layer: usize, n_gdn: usize, next_layer: usize) -> bool {
    next_layer < n_gdn && layer.checked_add(1) == Some(next_layer)
}

/// 行 t を書くのは 1 か所だけか (最終 chunk の tail を含む被覆検査)。
/// 戻り値: [n] の書き手数 (各要素 0 または 1 が正しい)。
pub fn write_counts(n: usize, chunk_len: usize, n_chunks: usize) -> Vec<u32> {
    let mut w = vec![0u32; n];
    for ci in 0..n_chunks {
        let last = ci == n_chunks - 1;
        let ws = wstart(ci, chunk_len, last, n);
        let we = write_end(ci, chunk_len, last, n).min(n);
        for slot in w.iter_mut().take(we).skip(ws) {
            *slot += 1;
        }
    }
    w
}

/// 同一 launch 内で「読む行」と「書く行」が交わらないことの検査用。
/// 戻り値: (読み手 chunk, 書き手 chunk, 行) の衝突一覧 (空なら健全)。
///
/// 2 launch 経路は読む行の書き手が**別 launch** であることを要求するので、
/// ここが空であることは必要条件にすぎない (経路を走らせてよい根拠にはならない)。
pub fn read_write_conflicts(
    n: usize,
    chunk_len: usize,
    n_chunks: usize,
) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    for ci in 0..n_chunks {
        let Some(row) = begin_row(ci, chunk_len) else {
            continue;
        };
        for wj in 0..n_chunks {
            let last = wj == n_chunks - 1;
            let ws = wstart(wj, chunk_len, last, n);
            let we = write_end(wj, chunk_len, last, n).min(n);
            if ws < we && row >= ws && row < we {
                out.push((ci, wj, row));
            }
        }
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;

    /// 実モデルの形に近い値で被覆を検査する (GPU 不要)。
    fn check_coverage(n: usize, nv: usize, dk: usize, dv: usize) -> (usize, usize) {
        let (n_chunks, chunk_len) = plan(n);
        assert!(n_chunks >= 1 && chunk_len >= 1);
        // chunk は隙間なく [0, n) を覆う (末尾 chunk 以外はちょうど chunk_len)
        for ci in 0..n_chunks {
            let t0 = ci * chunk_len;
            assert!(t0 < n, "n={n}: chunk {ci} が空 (n_chunks={n_chunks})");
            if ci + 1 < n_chunks {
                assert_eq!(n.min(t0 + chunk_len), t0 + chunk_len, "n={n}: 中間 chunk が短い");
            }
        }
        assert!(n_chunks * chunk_len >= n);
        // 書き手は「境界行」と「最終 chunk の tail」に限る (s90 の設計):
        //   - 行 n-1 は必ずちょうど 1 人が書く (最終 chunk)
        //   - 各 chunk ci (最後以外) の最終行 ci*chunk_len+chunk_len-1 は
        //     chunk ci が 1 人で書く (= 次の chunk が読む境界行)
        //   - どの行も 2 人以上は書かない (上書き競合が無い)
        //   - 中間行は誰も書かない (誰も読まないため。0 は正しい)
        let w = write_counts(n, chunk_len, n_chunks);
        assert_eq!(w.len(), n);
        for (t, c) in w.iter().enumerate() {
            assert!(*c <= 1, "n={n} chunk_len={chunk_len}: 行 {t} の書き手が {c} 人");
        }
        assert_eq!(w[n - 1], 1, "n={n}: 最終行の書き手が 1 人でない");
        for ci in 0..n_chunks - 1 {
            let boundary = ci * chunk_len + chunk_len - 1;
            assert!(boundary < n, "n={n}: 境界行 {boundary} が範囲外");
            assert_eq!(w[boundary], 1, "n={n}: 境界行 {boundary} の書き手が 1 人でない");
        }
        // 「読む行を別 launch が書く」前提の検査: 境界行の書き手は読む chunk の
        // **兄弟ではない** (別 launch 側) こと。今の forward ではこれが成立しない
        // ので、帳簿としては「兄弟と重なる」ことを明示的に記録しておく
        // (成立したら route_hoist の条件を見直す合図になる)。
        let conf = read_write_conflicts(n, chunk_len, n_chunks);
        for (reader, writer, row) in &conf {
            assert!(writer < reader, "n={n}: 行 {row} の書き手 {writer} が読み手 {reader} より後");
        }
        (n_chunks, chunk_len)
    }

    #[test]
    fn write_coverage_is_exclusive_and_covers_tail() {
        // plan() が単一 chunk になる境界・複数 chunk の端数あり/なし
        for n in [1usize, 2, 17, 1023, 2047, 2048, 2049, 2050, 3001, 4096, 4097, 8192] {
            check_coverage(n, 16, 128, 128);
        }
        // 端数が最終 chunk に集中する形 (n = k*chunk_len + r)
        for n in [2049usize, 2051, 3000, 4100, 5000] {
            let (nc, cl) = plan(n);
            if nc > 1 {
                check_coverage(n, 4, 64, 64);
                assert!(nc * cl - n < cl, "n={n}: 端数が最終 chunk に集中していない");
            }
        }
    }

    #[test]
    fn reader_rows_are_written_by_a_previous_chunk_or_leader() {
        // chunk ci が読む行は ci*chunk_len - 1 (ci=0 は ssm_in)
        for n in [2049usize, 3000, 4097, 8192] {
            let (nc, cl) = plan(n);
            assert!(nc > 1, "n={n} は複数 chunk のはず");
            assert_eq!(begin_row(0, cl), None);
            for ci in 1..nc {
                let row = begin_row(ci, cl).unwrap();
                assert_eq!(row, ci * cl - 1);
                // 読む行は必ずこの chunk の先頭より前 (= 兄弟の書き込み域と交わらない)
                assert!(row < ci * cl);
                // その行を書くのは「前の chunk」(最終 chunk が tail を書く場合も含む)
                let writers: Vec<usize> = (0..nc)
                    .filter(|&wj| {
                        let last = wj == nc - 1;
                        let ws = wstart(wj, cl, last, n);
                        let we = write_end(wj, cl, last, n).min(n);
                        ws < we && row >= ws && row < we
                    })
                    .collect();
                assert_eq!(
                    writers,
                    vec![ci - 1],
                    "n={n}: 行 {row} の書き手が想定外 (前の chunk 1 つだけのはず)"
                );
            }
        }
    }

    #[test]
    fn route_hoist_requires_a_cross_launch_writer() {
        // 書き手を保証できない限り、形が収まっていても 2 launch は使わない
        for n in [2049usize, 3000, 4097] {
            assert!(!route_hoist(n, 16, 128, 128, false));
        }
        // 書き手が保証されていても、バッファ上限を超えたら不可
        assert!(!route_hoist(4097, 1024, 128, 128, true));
    }

    #[test]
    fn srows_cap_boundaries() {
        // nv*dk*dv*4 (n=1) がちょうど 1MiB / 1MiB+1 になる形
        let at = SROWS_CAP_BYTES;
        assert!(fits(1, at / 4, 1, 1));
        assert!(!fits(2, at / 4, 1, 1)); // 2 行で 2MiB
        assert!(!fits(1, at / 4 + 1, 1, 1));
        // 代表形 (nv=16, dk=dv=128 → 1 行 1MiB) では n=1 のみ可
        assert_eq!(srows_bytes(1, 16, 128, 128), Some(at));
        assert!(fits(1, 16, 128, 128));
        assert!(!fits(2, 16, 128, 128));
        // 小さい形 (nv=4, dk=dv=64 → 64KiB/行) では n=16 まで可
        assert_eq!(srows_bytes(16, 4, 64, 64), Some(at));
        assert!(fits(16, 4, 64, 64));
        assert!(!fits(17, 4, 64, 64));
        // 溢れは None (wrap しない)
        assert_eq!(srows_bytes(usize::MAX, 2, 2, 2), None);
        assert!(!fits(usize::MAX, 2, 2, 2));
        assert_eq!(srows_bytes(3, 4, 5, 6), Some(3 * 4 * 5 * 6 * 4));
    }

    #[test]
    fn contiguity_and_successor_predicates() {
        // 同じ区間で、先頭が 0 でない → 層から層へそのまま渡せる形
        assert!(is_contig(2048, 5000, 2048, 5000));
        // 区間が違う / 先頭が 0 (前段はホスト) は不可
        assert!(!is_contig(2048, 5000, 2048, 5001));
        assert!(!is_contig(0, 5000, 0, 5000));
        // 隣の GDN 層だけが後継。最後の GDN 層の次は無い (= ホストへ戻す)
        assert!(is_successor(0, 24, 1));
        assert!(!is_successor(23, 24, 24));
        assert!(!is_successor(3, 24, 5));
        assert!(!is_successor(0, 1, 1));
    }

    #[test]
    fn chunk_plan_matches_cuda_routing_threshold() {
        // 閾値ちょうどで単一 chunk (= 逐次相当) から複数 chunk に変わる
        let (nc, cl) = plan(CHUNK_MIN_TOKENS);
        assert_eq!((nc, cl), (1, CHUNK_MIN_TOKENS));
        let (nc, cl) = plan(CHUNK_MIN_TOKENS + 1);
        assert!(nc >= 2);
        assert!(nc * cl >= CHUNK_MIN_TOKENS + 1);
    }
}
