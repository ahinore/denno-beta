//! P-MOE-E4: GGUF の量子化ブロックを**そのまま保持する**線形層。
//!
//! 既存の `Quant8Linear` / `Q4Linear` はロード時に int8 / Q4_0 へ量子化し直す
//! (27B の重みはその経路で作られている)。Flash-Next の重みは GGUF の中で既に
//! Q8_0 / Q4_K / Q5_K / Q5_1 / BF16 / F32 なので、**作り直さず生バイトのまま
//! 計算する**。量子化し直すと数値が変わり、アンカー比較の意味が無くなる。
//!
//! 数値契約:
//! - `y[r] = dot(row_r, x)` を r の昇順、内積は i の昇順に f32 で加算する
//!   (本体は `model::moe_expert::expert_gemv` を共有する。CPU と GPU の
//!   ビット一致はこの順序で決まる)。
//! - 行のデコードは `kquant::dequant_row_into` と同一 (GGML のブロック仕様)。
//!
//! GPU 経路はまだ繋いでいない (重みの常駐キャッシュと一緒に入れる)。現状は
//! 常に CPU 参照で計算する。Flash-Next の 1 トークンあたりの実測はこの経路で
//! 取れるが、120B を CPU で回すので秒/token 級になる。

use anyhow::{bail, Result};

use crate::kquant::{dequant_row_into, GgmlType};
use crate::model::moe_expert::{expert_gemv, expert_gemv_q8_fast};
use crate::quant::WeightBlocks;

/// 1 行のバイト数。`moe_expert::row_bytes` は量子型だけを受けるので、
/// F32 / F16 / BF16 も含めて受ける版をここに持つ (ブロック仕様は同じ)。
fn row_bytes_any(ty: GgmlType, k: usize) -> Result<usize> {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    if k == 0 || k % bs != 0 {
        bail!("GgufLinear: k={k} が {ty:?} のブロックサイズ {bs} の倍数でない");
    }
    Ok((k / bs) * ts)
}

/// GGUF のブロックを保持したまま GEMV する線形層 (n 行 k 列、y = W x)。
pub struct GgufLinear {
    uid: u64,
    ty: GgmlType,
    k: usize,
    n: usize,
    blocks: WeightBlocks,
    /// Q8_0 のときだけ、AVX2 融合カーネルを使うか。
    ///
    /// 既定は false (27B の Q8_0 テンソル 1 本がビット一致契約の対象なので、
    /// 明示的に選んだモデルだけ速い経路に載せる)。
    fast_q8: bool,
}

impl GgufLinear {
    /// 生バイトから作る。`blocks.len()` は `n * row_bytes(ty, k)` と一致すること。
    pub fn new(ty: GgmlType, k: usize, n: usize, blocks: WeightBlocks) -> Result<Self> {
        let rb = row_bytes_any(ty, k)?;
        if n == 0 || k == 0 {
            bail!("GgufLinear: 形状が 0 (k={k}, n={n})");
        }
        let want = n
            .checked_mul(rb)
            .ok_or_else(|| anyhow::anyhow!("GgufLinear: n * row_bytes が usize を溢れる"))?;
        if blocks.as_slice().len() != want {
            bail!(
                "GgufLinear: バイト数 {} が n * row_bytes = {} と違う (ty={ty:?}, k={k}, n={n})",
                blocks.as_slice().len(),
                want
            );
        }
        Ok(Self {
            uid: crate::quant::next_q8_uid(),
            ty,
            k,
            n,
            blocks,
            fast_q8: false,
        })
    }

    /// Q8_0 の AVX2 融合カーネルを有効にする (ビット一致は保証しない)。
    pub fn with_fast_q8(mut self, on: bool) -> Self {
        self.fast_q8 = on;
        self
    }

    /// Q8_0 の融合カーネルが有効か (既定 false)。
    pub fn fast_q8(&self) -> bool {
        self.fast_q8
    }

    /// スライスから作る (テスト・小さいテンソル用)。
    pub fn from_slice(ty: GgmlType, k: usize, n: usize, bytes: &[u8]) -> Result<Self> {
        Self::new(ty, k, n, WeightBlocks::from_vec(bytes.to_vec()))
    }

    pub fn uid(&self) -> u64 {
        self.uid
    }
    pub fn ty(&self) -> GgmlType {
        self.ty
    }
    /// 入力長 (列数)
    pub fn k(&self) -> usize {
        self.k
    }
    /// 出力長 (行数)
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn blocks(&self) -> &WeightBlocks {
        &self.blocks
    }
    pub fn weight_bytes(&self) -> usize {
        self.blocks.as_slice().len()
    }

    /// `out[r] = dot(row_r, x)`。`x.len() == k`、`out.len() == n` が必要。
    pub fn gemv(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != self.k {
            bail!("GgufLinear::gemv: x の長さ {} が k={} と違う", x.len(), self.k);
        }
        if out.len() != self.n {
            bail!("GgufLinear::gemv: out の長さ {} が n={} と違う", out.len(), self.n);
        }
        match self.ty {
            // 量子型は expert GEMV と本体を共有する (1 行ずつデコードして逐次内積)。
            GgmlType::Q8_0 | GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => {
                // P-MOE-E5: 重みが VRAM に常駐していれば GPU で計算する。
                // 対象外・GPU 無効・不調では false が返り、下の CPU 参照へ落ちる。
                #[cfg(feature = "cuda")]
                {
                    if crate::cuda::maybe_gpu_gguf_gemv_into(
                        self.uid,
                        self.ty,
                        self.blocks.as_slice(),
                        self.k,
                        self.n,
                        x,
                        out,
                    ) {
                        return Ok(());
                    }
                }
                if self.fast_q8 && self.ty == GgmlType::Q8_0 {
                    return expert_gemv_q8_fast(self.blocks.as_slice(), self.k, self.n, x, out);
                }
                expert_gemv(self.ty, self.blocks.as_slice(), self.k, self.n, x, out)
            }
            // 無量子型 (F32 / F16 / BF16) は expert GEMV の対象外なので同じ規則で回す。
            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => {
                let rb = row_bytes_any(self.ty, self.k)?;
                let b = self.blocks.as_slice();
                // P-MOE-G11: 行を rayon で分割する。行は独立で、行内の逐次加算順は
                // 変えないので、分割の有無で結果はビット一致する (従来の逐次版と
                // 同じ `dequant_row_into` + 同じ順の内積を通る)。
                // MoE のルータ (`ffn_gate_inp`, F32, 2560x512 = 5.2 MB) がここを通り、
                // 1 層あたり 1.07 ms かかっていた (moe の 42%)。
                #[cfg(feature = "par")]
                if self.n >= 64 && self.k >= 256 {
                    use rayon::prelude::*;
                    let threads = rayon::current_num_threads().max(1);
                    let rows = self.n.div_ceil(threads).max(32).min(self.n);
                    let bytes = rows * rb;
                    b.par_chunks(bytes).zip(out.par_chunks_mut(rows)).for_each(
                        |(bb, oo)| {
                            let mut row = vec![0f32; self.k];
                            for (r, slot) in oo.iter_mut().enumerate() {
                                dequant_row_into(self.ty, &bb[r * rb..(r + 1) * rb], &mut row);
                                let mut acc = 0f32;
                                for i in 0..self.k {
                                    acc += row[i] * x[i];
                                }
                                *slot = acc;
                            }
                        },
                    );
                    return Ok(());
                }
                let mut row = vec![0f32; self.k];
                for r in 0..self.n {
                    dequant_row_into(self.ty, &b[r * rb..(r + 1) * rb], &mut row);
                    let mut acc = 0f32;
                    for i in 0..self.k {
                        acc += row[i] * x[i];
                    }
                    out[r] = acc;
                }
                Ok(())
            }
            other => bail!("GgufLinear::gemv: 未対応の量子型 {other:?}"),
        }
    }

    /// 行 j を f32 に展開する (検証・デバッグ用)。
    pub fn dequant_row(&self, j: usize) -> Result<Vec<f32>> {
        if j >= self.n {
            bail!("GgufLinear::dequant_row: 行 {j} が n={} の範囲外", self.n);
        }
        let rb = row_bytes_any(self.ty, self.k)?;
        let b = self.blocks.as_slice();
        let mut out = vec![0f32; self.k];
        dequant_row_into(self.ty, &b[j * rb..(j + 1) * rb], &mut out);
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! 実 GGUF はこのコンテナから見えないのでブロックはテスト内で組み立てる。
    //! 期待値は「全行列を f32 へ展開してから素直な逐次内積」を独立に書き下した
    //! 参照ループと `to_bits()` で比較する (融合実装のレイアウト検算)。

    use super::*;
    use crate::kquant::dequant_row;

    const K: usize = 256;
    const N: usize = 5;

    /// 決定的な擬似乱数バイト。
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

    /// 1 ブロックを組み立てる。スケール欄は有限値に固定する。
    fn make_block(ty: GgmlType, seed: u32) -> Vec<u8> {
        let ts = ty.type_size();
        let mut b: Vec<u8> = (0..ts).map(|i| pat(i, seed)).collect();
        match ty {
            GgmlType::Q8_0 => b[0..2].copy_from_slice(&f16_bytes(0.125)),
            GgmlType::Q5_1 | GgmlType::Q4K | GgmlType::Q5K => {
                b[0..2].copy_from_slice(&f16_bytes(0.125));
                b[2..4].copy_from_slice(&f16_bytes(0.0625));
            }
            GgmlType::F32 => {
                for i in 0..(ts / 4) {
                    let v = ((pat(i, seed) as f32) / 128.0) - 1.0;
                    b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
            }
            GgmlType::F16 | GgmlType::Bf16 => {
                for i in 0..(ts / 2) {
                    let v = ((pat(i, seed) as f32) / 128.0) - 1.0;
                    if ty == GgmlType::F16 {
                        b[i * 2..i * 2 + 2].copy_from_slice(&f16_bytes(v));
                    } else {
                        let bits = (v.to_bits() >> 16) as u16;
                        b[i * 2..i * 2 + 2].copy_from_slice(&bits.to_le_bytes());
                    }
                }
            }
            _ => {}
        }
        b
    }

    fn make_matrix(ty: GgmlType, k: usize, n: usize, seed: u32) -> Vec<u8> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert_eq!(k % bs, 0);
        let nb = k / bs;
        let mut out = Vec::with_capacity(n * nb * ts);
        for r in 0..n {
            for blk in 0..nb {
                out.extend_from_slice(&make_block(ty, seed.wrapping_add((r * 31 + blk) as u32)));
            }
        }
        out
    }

    fn make_x(k: usize, seed: u32) -> Vec<f32> {
        (0..k)
            .map(|i| ((pat(i, seed) as f32) / 128.0) - 1.0)
            .collect()
    }

    /// 独立参照: 全行列を f32 に展開してから素直な逐次内積。
    fn naive_gemv(ty: GgmlType, w: &[u8], k: usize, n: usize, x: &[f32]) -> Vec<f32> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        let rb = (k / bs) * ts;
        let mut out = vec![0f32; n];
        for r in 0..n {
            let row = dequant_row(ty, &w[r * rb..(r + 1) * rb], k);
            let mut acc = 0f32;
            for i in 0..k {
                acc += row[i] * x[i];
            }
            out[r] = acc;
        }
        out
    }

    fn check(ty: GgmlType) {
        let w = make_matrix(ty, K, N, 7);
        let x = make_x(K, 11);
        let lin = GgufLinear::from_slice(ty, K, N, &w).expect("GgufLinear");
        assert_eq!(lin.k(), K);
        assert_eq!(lin.n(), N);
        assert_eq!(lin.ty(), ty);
        assert_eq!(lin.weight_bytes(), w.len());

        let mut got = vec![0f32; N];
        lin.gemv(&x, &mut got).expect("gemv");
        let want = naive_gemv(ty, &w, K, N, &x);
        for r in 0..N {
            assert_eq!(got[r].to_bits(), want[r].to_bits(), "{ty:?}: 行 {r}");
        }
    }

    #[test]
    fn q8_0_matches_naive() {
        check(GgmlType::Q8_0);
    }

    #[test]
    fn q5_1_matches_naive() {
        check(GgmlType::Q5_1);
    }

    #[test]
    fn q4_k_matches_naive() {
        check(GgmlType::Q4K);
    }

    #[test]
    fn q5_k_matches_naive() {
        check(GgmlType::Q5K);
    }

    #[test]
    fn f32_matches_naive() {
        check(GgmlType::F32);
    }

    #[test]
    fn f16_matches_naive() {
        check(GgmlType::F16);
    }

    #[test]
    fn bf16_matches_naive() {
        check(GgmlType::Bf16);
    }

    #[test]
    fn dequant_row_is_consistent_with_matrix_layout() {
        let w = make_matrix(GgmlType::Q8_0, K, N, 3);
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w).unwrap();
        for r in 0..N {
            let row = lin.dequant_row(r).unwrap();
            let bs = GgmlType::Q8_0.blck_size();
            let ts = GgmlType::Q8_0.type_size();
            let rb = (K / bs) * ts;
            let direct = dequant_row(GgmlType::Q8_0, &w[r * rb..(r + 1) * rb], K);
            assert_eq!(row, direct);
        }
        assert!(lin.dequant_row(N).is_err());
    }

    #[test]
    fn rejects_bad_shapes() {
        let w = make_matrix(GgmlType::Q8_0, K, N, 5);
        // バイト数不足
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w[..w.len() - 1]).is_err());
        // 行数とバイト数の不一致
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K, N + 1, &w).is_err());
        // ブロック境界でない k
        assert!(GgufLinear::from_slice(GgmlType::Q8_0, K + 1, N, &w).is_err());

        let lin = GgufLinear::from_slice(GgmlType::Q8_0, K, N, &w).unwrap();
        let x = make_x(K, 1);
        let mut out = vec![0f32; N];
        assert!(lin.gemv(&x[..K - 1], &mut out).is_err());
        assert!(lin.gemv(&x, &mut out[..N - 1]).is_err());
    }

    #[test]
    fn zero_input_gives_zero() {
        let w = make_matrix(GgmlType::Q4K, K, N, 9);
        let lin = GgufLinear::from_slice(GgmlType::Q4K, K, N, &w).unwrap();
        let x = vec![0f32; K];
        let mut out = vec![1.0f32; N];
        lin.gemv(&x, &mut out).unwrap();
        for r in 0..N {
            assert_eq!(out[r], 0.0, "行 {r}");
        }
    }
}
