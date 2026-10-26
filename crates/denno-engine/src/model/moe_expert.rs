//! MoE expert のスライス切り出しと GEMV (P-MOE-B1)。
//!
//! 出典: expert テンソルの配置 (`ne = [n_in, n_out, n_expert]`) と量子型の使い分けは
//! llama.cpp `src/models/qwen4exp.cpp` (commit `d854d98`、2026-10-26 取得、
//! MIT License, Copyright (c) 2023-2026 The ggml authors) を参照した。
//! 本ファイルは Rust による再実装であり、逐語コピーは含まない。
//! 行のデコードは本リポジトリの `kquant.rs` (同じく MIT 由来) を再利用している。
//! ライセンス台帳は `docs/history/80_licenses.md`。
//!
//! Flash-Next 系の expert テンソルは GGUF 上で 3 次元 (`ne = [n_in, n_out, n_expert]`、
//! ne[0] が最内 = 行長) で、1 ファイルに全 expert ぶんが連結されている。
//! 1 トークンで使うのは各層 top-k 個だけなので、expert 単位でバイト範囲を切り出して
//! GEMV する経路が要る。ここに置くのはその **CPU 参照実装**だけで、
//! GPU カーネルのビット一致比較の基準になる。推論経路には接続しない。
//!
//! 契約:
//!
//! - 対象量子型は実ファイルで使われている 4 種のみ
//!   (**Q4_K / Q5_K / Q5_1 / Q8_0**)。それ以外は `Err`。黙って別経路に落とさない。
//! - 行のデ量子化は [`crate::kquant::dequant_row_into`] そのものを使う
//!   ([`crate::kquant::dequant_row`] は同じ本体に `Vec` を渡すだけなので値は同一)。
//! - 内積は `0..n_in` の昇順に逐次加算。SIMD 化・分割アキュムレータ・FMA 化をしない。
//!   Rust は既定で浮動小数の再結合を許さないので、素直な `acc += w[i] * x[i]` は
//!   丸め順序が固定される (これがビット一致の根拠)。
//! - 行バッファは 1 本だけ確保して行ごとに使い回す (行ごとに `Vec` を作らない)。

use anyhow::{bail, Result};

use crate::kquant::{dequant_row_into, GgmlType};

/// 1 expert ぶんのバイト範囲 (テンソル先頭からの相対)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertSlice {
    /// テンソルのデータ先頭からのオフセット (バイト)
    pub off: usize,
    /// 1 expert ぶんのバイト数
    pub len: usize,
}

impl ExpertSlice {
    /// `ne = [n_in, n_out, n_expert]` の expert テンソルから expert `e` の範囲を返す。
    ///
    /// `off = e * n_out * row_bytes(n_in, ty)`、`len = n_out * row_bytes(n_in, ty)`。
    /// メモリ上は expert が最外なので、expert のバイト列は連続する。
    pub fn new(ty: GgmlType, n_in: usize, n_out: usize, n_expert: usize, e: usize) -> Result<Self> {
        let rb = row_bytes(ty, n_in)?;
        if n_out == 0 {
            bail!("n_out が 0 (行が無い)");
        }
        if n_expert == 0 {
            bail!("n_expert が 0");
        }
        if e >= n_expert {
            bail!("expert index {} が範囲外 (n_expert={})", e, n_expert);
        }
        let len = n_out
            .checked_mul(rb)
            .ok_or_else(|| anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={})", n_out))?;
        let off = e
            .checked_mul(len)
            .ok_or_else(|| anyhow::anyhow!("expert offset が usize を溢れる (e={})", e))?;
        Ok(ExpertSlice { off, len })
    }

    /// 範囲の終端 (排他)。テンソル総バイト数以下であることを呼び出し側で確認する。
    pub fn end(&self) -> usize {
        self.off + self.len
    }
}

/// 1 行 (`n_in` 要素) あたりのバイト数。`n_in` は量子型のブロックサイズで割り切れる必要がある。
///
/// 対象外の量子型は `Err` (Q4_K / Q5_K / Q5_1 / Q8_0 以外)。
pub fn row_bytes(ty: GgmlType, n_in: usize) -> Result<usize> {
    match ty {
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q5_1 | GgmlType::Q8_0 => {}
        other => bail!("expert GEMV の対象外の量子型: {:?}", other),
    }
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if n_in == 0 {
        bail!("n_in が 0");
    }
    if bs == 0 || n_in % bs != 0 {
        bail!("n_in={} がブロックサイズ {} で割り切れない (端数は扱わない)", n_in, bs);
    }
    Ok((n_in / bs) * ts)
}

/// 内積 (f32 逐次加算・`0..n` 昇順)。
///
/// ビット一致の基準なので分割アキュムレータや FMA 化をしない
/// (`#[inline(never)]` は最適化で演算列が畳まれないようにするための目印)。
#[inline(never)]
fn dot_seq(w: &[f32], x: &[f32]) -> f32 {
    debug_assert_eq!(w.len(), x.len());
    let mut acc = 0.0f32;
    for i in 0..x.len() {
        acc += w[i] * x[i];
    }
    acc
}

/// 1 expert ぶんの生バイト (`blocks`) に対する GEMV。`out[r] = dot(row_r, x)`。
///
/// - `blocks` は [`ExpertSlice`] で切り出した 1 expert ぶん (`len` バイトちょうど)。
/// - 行は `0..n_out` の昇順、内積は `0..n_in` の昇順に加算する。
/// - 行バッファは 1 本 (`n_in` 要素) だけ確保して使い回す。
pub fn expert_gemv(
    ty: GgmlType,
    blocks: &[u8],
    n_in: usize,
    n_out: usize,
    x: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let rb = row_bytes(ty, n_in)?;
    if n_out == 0 {
        bail!("n_out が 0 (行が無い)");
    }
    if x.len() != n_in {
        bail!("x の長さ {} が n_in={} と違う", x.len(), n_in);
    }
    if out.len() != n_out {
        bail!("out の長さ {} が n_out={} と違う", out.len(), n_out);
    }
    let want = n_out
        .checked_mul(rb)
        .ok_or_else(|| anyhow::anyhow!("1 expert のバイト数が usize を溢れる (n_out={})", n_out))?;
    if blocks.len() != want {
        bail!("blocks の長さ {} が n_out * row_bytes = {} と違う", blocks.len(), want);
    }

    // 行バッファは 1 本だけ。行ごとに Vec を確保しない。
    let mut row = vec![0.0f32; n_in];
    for r in 0..n_out {
        let src = &blocks[r * rb..(r + 1) * rb];
        dequant_row_into(ty, src, &mut row);
        out[r] = dot_seq(&row, x);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! 実ファイルはこのコンテナから見えないので、ブロックはテスト内で組み立てる。
    //! 期待値は「dequant_row + 素直な逐次内積」を独立に書き下した参照ループと
    //! `to_bits()` で比較する (Q8_0 の 1 本だけは手計算値も突き合わせる)。

    use super::*;
    use crate::kquant::dequant_row;

    const N_IN: usize = 256;
    const N_OUT: usize = 4;
    const N_EXPERT: usize = 3;

    /// テスト用の決定的バイト列 (0..=255)。
    fn pat(i: usize, seed: u32) -> u8 {
        let mut x = (i as u32)
            .wrapping_mul(2_654_435_761)
            .wrapping_add(seed.wrapping_mul(40_503));
        x ^= x >> 13;
        x = x.wrapping_mul(0x9E37_79B1);
        x ^= x >> 16;
        x as u8
    }

    fn f16_bytes(v: f32) -> [u8; 2] {
        half::f16::from_f32(v).to_le_bytes()
    }

    /// 1 ブロックを組み立てる。スケール類は有限値に固定する
    /// (生バイトをそのまま d に入れると f16 が NaN/Inf になり、テストが無意味になる)。
    fn make_block(ty: GgmlType, seed: u32) -> Vec<u8> {
        let ts = ty.type_size();
        let mut b = vec![0u8; ts];
        match ty {
            // d(2) + dmin(2) + scales[12] + qs[128]
            GgmlType::Q4K => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                b[2..4].copy_from_slice(&f16_bytes(0.005));
                for i in 4..16 {
                    // 下位 6bit = scale (0..63)、上位 2bit = min の上位ビット
                    b[i] = (pat(i, seed) & 0x3f) | ((pat(i + 7, seed) & 0x3) << 6);
                }
                for i in 16..144 {
                    b[i] = pat(i, seed);
                }
            }
            // d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
            GgmlType::Q5K => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                b[2..4].copy_from_slice(&f16_bytes(0.005));
                for i in 4..16 {
                    b[i] = (pat(i, seed) & 0x3f) | ((pat(i + 7, seed) & 0x3) << 6);
                }
                for i in 16..176 {
                    b[i] = pat(i, seed);
                }
            }
            // d(2) + m(2) + qh(4) + qs[16]
            GgmlType::Q5_1 => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                b[2..4].copy_from_slice(&f16_bytes(0.005));
                for i in 4..24 {
                    b[i] = pat(i, seed);
                }
            }
            // d(2) + qs[32]
            GgmlType::Q8_0 => {
                b[0..2].copy_from_slice(&f16_bytes(0.01));
                for i in 2..34 {
                    b[i] = pat(i, seed);
                }
            }
            other => panic!("テスト対象外の型: {:?}", other),
        }
        b
    }

    /// 1 expert ぶんのブロック列を、`(expert, row, block)` ごとに別 seed で組む。
    fn make_expert(ty: GgmlType, n_in: usize, n_out: usize, e: usize) -> Vec<u8> {
        let bs = ty.blck_size();
        let nblk = n_in / bs;
        let mut v = Vec::new();
        for r in 0..n_out {
            for blk in 0..nblk {
                let seed = (e * 10_000 + r * 100 + blk) as u32;
                v.extend_from_slice(&make_block(ty, seed));
            }
        }
        v
    }

    /// expert テンソル全体 (expert が最外で連結)。
    fn make_tensor(ty: GgmlType, n_in: usize, n_out: usize, n_expert: usize) -> Vec<u8> {
        let mut v = Vec::new();
        for e in 0..n_expert {
            v.extend_from_slice(&make_expert(ty, n_in, n_out, e));
        }
        v
    }

    fn make_x(n_in: usize) -> Vec<f32> {
        (0..n_in)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 64.0)
            .collect()
    }

    /// 参照: `dequant_row` (Vec 確保版) + 素直な逐次内積。
    fn ref_gemv(ty: GgmlType, blocks: &[u8], n_in: usize, n_out: usize, x: &[f32]) -> Vec<f32> {
        let rb = row_bytes(ty, n_in).unwrap();
        let mut out = vec![0.0f32; n_out];
        for r in 0..n_out {
            let w = dequant_row(ty, &blocks[r * rb..(r + 1) * rb], n_in);
            let mut acc = 0.0f32;
            for i in 0..n_in {
                acc += w[i] * x[i];
            }
            out[r] = acc;
        }
        out
    }

    fn expect_bits(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "長さが違う");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "row {} がビット一致しない: got={} ({:#010x}) want={} ({:#010x})",
                i,
                g,
                g.to_bits(),
                w,
                w.to_bits()
            );
        }
    }

    const ALL_TYPES: [GgmlType; 4] =
        [GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q5_1, GgmlType::Q8_0];

    // ---- 1. スライス切り出し ------------------------------------------------

    #[test]
    fn slice_matches_manual_expert_split() {
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            let total = tensor.len();
            let mut covered = 0usize;
            for e in 0..N_EXPERT {
                let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
                let want = make_expert(ty, N_IN, N_OUT, e);
                assert_eq!(sl.len, want.len(), "{:?}: expert {} の len が違う", ty, e);
                assert_eq!(
                    &tensor[sl.off..sl.end()],
                    &want[..],
                    "{:?}: expert {} の範囲が手で分割した範囲と違う",
                    ty,
                    e
                );
                covered += sl.len;
            }
            assert_eq!(covered, total, "{:?}: スライスが全体を過不足なく覆っていない", ty);
            // 最後の expert の終端がテンソル総バイト数に一致する
            let last = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT - 1).unwrap();
            assert_eq!(last.end(), total, "{:?}: 最終 expert の終端が総バイト数と違う", ty);
        }
    }

    #[test]
    fn slice_offset_is_contiguous_and_ordered() {
        let ty = GgmlType::Q4K;
        let rb = row_bytes(ty, N_IN).unwrap();
        let per = N_OUT * rb;
        for e in 0..N_EXPERT {
            let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
            assert_eq!(sl.off, e * per);
            assert_eq!(sl.len, per);
        }
    }

    // ---- 2. GEMV が dequant_row + 逐次内積とビット一致 -----------------------

    #[test]
    fn gemv_bits_equal_dequant_row_and_sequential_dot() {
        let x = make_x(N_IN);
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            for e in 0..N_EXPERT {
                let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
                let blocks = &tensor[sl.off..sl.end()];
                let mut got = vec![0.0f32; N_OUT];
                expert_gemv(ty, blocks, N_IN, N_OUT, &x, &mut got).unwrap();
                let want = ref_gemv(ty, blocks, N_IN, N_OUT, &x);
                expect_bits(&got, &want);
                // 縮退していないこと (全行 0 なら一致しても意味が無い)
                assert!(
                    got.iter().any(|v| *v != 0.0),
                    "{:?}: expert {} の出力が全 0",
                    ty,
                    e
                );
                assert!(got.iter().all(|v| v.is_finite()), "{:?}: 非有限値", ty);
            }
        }
    }

    #[test]
    fn gemv_on_extracted_slice_equals_gemv_on_standalone_bytes() {
        // ExpertSlice で切り出したバイト列と、expert 単体で組んだバイト列は同一のはず。
        let x = make_x(N_IN);
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            let standalone = make_expert(ty, N_IN, N_OUT, 1);
            let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, 1).unwrap();
            let mut a = vec![0.0f32; N_OUT];
            let mut b = vec![0.0f32; N_OUT];
            expert_gemv(ty, &tensor[sl.off..sl.end()], N_IN, N_OUT, &x, &mut a).unwrap();
            expert_gemv(ty, &standalone, N_IN, N_OUT, &x, &mut b).unwrap();
            expect_bits(&a, &b);
        }
    }

    #[test]
    fn gemv_q8_0_matches_hand_computed_value() {
        // d = 0.5、qs = 1..=32、x = 全 1.0 → out = 0.5 * (1+...+32) = 0.5 * 528 = 264。
        // 0.5 * q は f32 で厳密、和も 264 まで厳密なので丸めの余地が無い。
        let n_in = 32usize;
        let n_out = 1usize;
        let mut b = vec![0u8; GgmlType::Q8_0.type_size()];
        b[0..2].copy_from_slice(&f16_bytes(0.5));
        for j in 0..32 {
            b[2 + j] = (j as u8) + 1;
        }
        let x = vec![1.0f32; n_in];
        let mut out = vec![0.0f32; n_out];
        expert_gemv(GgmlType::Q8_0, &b, n_in, n_out, &x, &mut out).unwrap();
        assert_eq!(out[0].to_bits(), 264.0f32.to_bits(), "got {}", out[0]);
    }

    // ---- 3. n_in がブロックサイズで割り切れない -----------------------------

    #[test]
    fn row_bytes_rejects_ragged_n_in() {
        for (ty, bad) in [
            (GgmlType::Q4K, 257usize),
            (GgmlType::Q5K, 255),
            (GgmlType::Q5_1, 33),
            (GgmlType::Q8_0, 31),
        ] {
            assert!(row_bytes(ty, bad).is_err(), "{:?}: n_in={} が通ってしまった", ty, bad);
            assert!(
                ExpertSlice::new(ty, bad, N_OUT, N_EXPERT, 0).is_err(),
                "{:?}: ExpertSlice が端数を許した",
                ty
            );
            // gemv も同じ検査で落ちる (デ量子化の debug_assert より前に弾く)
            let blocks = vec![0u8; 4096];
            let x = vec![0.0f32; bad];
            let mut out = vec![0.0f32; N_OUT];
            assert!(expert_gemv(ty, &blocks, bad, N_OUT, &x, &mut out).is_err());
        }
        // 0 も弾く
        assert!(row_bytes(GgmlType::Q4K, 0).is_err());
    }

    // ---- 4. e >= n_expert ---------------------------------------------------

    #[test]
    fn slice_rejects_out_of_range_expert() {
        for ty in ALL_TYPES {
            assert!(ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT).is_err());
            assert!(ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT + 5).is_err());
            assert!(ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, N_EXPERT - 1).is_ok());
        }
        // n_expert = 0 は誰も選べない
        assert!(ExpertSlice::new(GgmlType::Q4K, N_IN, N_OUT, 0, 0).is_err());
    }

    // ---- 5. 範囲外読みが起きないこと ---------------------------------------

    #[test]
    fn last_expert_slice_fits_in_tensor() {
        for ty in ALL_TYPES {
            let tensor = make_tensor(ty, N_IN, N_OUT, N_EXPERT);
            for e in 0..N_EXPERT {
                let sl = ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, e).unwrap();
                assert!(
                    sl.end() <= tensor.len(),
                    "{:?}: expert {} の終端 {} がテンソル長 {} を超える",
                    ty,
                    e,
                    sl.end(),
                    tensor.len()
                );
                // 実際にスライスして GEMV しても落ちない (境界の実読み)
                let x = make_x(N_IN);
                let mut out = vec![0.0f32; N_OUT];
                expert_gemv(ty, &tensor[sl.off..sl.end()], N_IN, N_OUT, &x, &mut out).unwrap();
            }
        }
    }

    #[test]
    fn gemv_rejects_wrong_buffer_lengths() {
        let ty = GgmlType::Q4K;
        let blocks = make_expert(ty, N_IN, N_OUT, 0);
        let x = make_x(N_IN);
        let mut out = vec![0.0f32; N_OUT];
        // blocks が 1 バイト短い / 長い
        assert!(expert_gemv(ty, &blocks[..blocks.len() - 1], N_IN, N_OUT, &x, &mut out).is_err());
        let mut longer = blocks.clone();
        longer.push(0);
        assert!(expert_gemv(ty, &longer, N_IN, N_OUT, &x, &mut out).is_err());
        // x / out の長さ違い
        assert!(expert_gemv(ty, &blocks, N_IN, N_OUT, &x[..N_IN - 1], &mut out).is_err());
        let mut short_out = vec![0.0f32; N_OUT - 1];
        assert!(expert_gemv(ty, &blocks, N_IN, N_OUT, &x, &mut short_out).is_err());
        // n_out = 0
        assert!(expert_gemv(ty, &[], N_IN, 0, &x, &mut []).is_err());
    }

    // ---- 型の門番 -----------------------------------------------------------

    #[test]
    fn unsupported_types_are_rejected() {
        for ty in [
            GgmlType::F32,
            GgmlType::Q4_0,
            GgmlType::Q6K,
            GgmlType::Q8_0, // 下で「対象」であることを確認する対照
        ] {
            let want_ok = ty == GgmlType::Q8_0;
            assert_eq!(row_bytes(ty, N_IN).is_ok(), want_ok, "{:?}", ty);
            assert_eq!(
                ExpertSlice::new(ty, N_IN, N_OUT, N_EXPERT, 0).is_ok(),
                want_ok,
                "{:?}",
                ty
            );
        }
        let x = make_x(N_IN);
        let mut out = vec![0.0f32; N_OUT];
        assert!(expert_gemv(GgmlType::Q6K, &[0u8; 4096], N_IN, N_OUT, &x, &mut out).is_err());
        assert!(expert_gemv(GgmlType::F32, &[0u8; 4096], N_IN, N_OUT, &x, &mut out).is_err());
    }

    // ---- 実ファイルの形 (前提の実測値) との突き合わせ -----------------------

    #[test]
    fn row_bytes_match_flash_next_measurements() {
        // 前提表: ffn_{gate,up}_exps は [2560, 640, 512] で Q4_K の行が 1440 B、
        // ffn_down_exps は [640, 2560, 512] で Q5_1 の行が 480 B。
        assert_eq!(row_bytes(GgmlType::Q4K, 2560).unwrap(), 1440);
        assert_eq!(row_bytes(GgmlType::Q5K, 2560).unwrap(), 1760);
        assert_eq!(row_bytes(GgmlType::Q5_1, 640).unwrap(), 480);
        assert_eq!(row_bytes(GgmlType::Q8_0, 640).unwrap(), 680);
        // 1 expert ぶんのバイト数 (前提表の 921.7 KB / 1.23 MB に対応)
        assert_eq!(ExpertSlice::new(GgmlType::Q4K, 2560, 640, 512, 0).unwrap().len, 921_600);
        assert_eq!(ExpertSlice::new(GgmlType::Q5_1, 640, 2560, 512, 0).unwrap().len, 1_228_800);
        // 512 expert 全体のバイト数
        let sl = ExpertSlice::new(GgmlType::Q4K, 2560, 640, 512, 511).unwrap();
        assert_eq!(sl.off, 511 * 921_600);
        assert_eq!(sl.end(), 512 * 921_600);
    }
}
