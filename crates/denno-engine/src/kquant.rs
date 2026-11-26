//! GGML ブロック量子 (GGUF) の f32 へのデ量子化 (P19)。
//!
//! llama.cpp `ggml-quants.c` の dequantize_row_* を忠実に移植したもの。
//! **契約: llama.cpp とのビット一致**(同一ブロック列 → 同一 f32 列)。
//! 検証は C リファレンス(ref/ggml からビルドした dequant_ref)との比較テストで行う。
//!
//! 参照実装: llama.cpp ggml (https://github.com/ggml-org/llama.cpp)
//! MIT License — Copyright (c) 2023-2026 The ggml authors。
//! ビット一致契約のため演算を ggml に忠実に移植している (MIT 表示義務を満たすため
//! 本表示を保持する)。
//!
//! テーブル (KQUANT_TABLES) は ggml-common.h から自動生成 (gen_kquant_tables.py)。
//! 手で編集しない。

/// IQ2_S の格子 (8 要素 = u64) を外へ出す。融合カーネルが 8 バイトを直接読む。
#[inline]
pub fn iq2s_grid() -> &'static [u64; 1024] {
    &crate::kquant_tables::IQ2S_GRID
}

/// IQ3_XXS の格子 (4 要素 = u32) を外へ出す。
#[inline]
pub fn iq3xxs_grid() -> &'static [u32; 256] {
    &crate::kquant_tables::IQ3XXS_GRID
}

/// IQ2_S / IQ3_XXS の符号表 (7 ビット -> 8 レーンの符号マスク)。
#[inline]
pub fn ksigns_iq2xs() -> &'static [u8; 128] {
    &crate::kquant_tables::KSIGNS_IQ2XS
}

/// IQ4_NL の 4 ビット表 (16 エントリ) を外へ出す。融合カーネルが `pshufb` で引く。
#[inline]
pub fn kvalues_iq4nl() -> &'static [i8; 16] {
    &crate::kquant_tables::KVALUES_IQ4NL
}

use crate::kquant_tables::*;

/// GGML 型 ID (ggml.h enum ggml_type と同一値)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,     // 0
    F16,     // 1
    Q4_0,    // 2
    Q4_1,    // 3
    Q5_0,    // 6
    Q5_1,    // 7
    Q8_0,    // 8
    Q2K,     // 10
    Q3K,     // 11
    Q4K,     // 12
    Q5K,     // 13
    Q6K,     // 14
    IQ2Xxs,  // 16
    IQ2Xs,   // 17
    IQ3Xxs,  // 18
    IQ1S,    // 19
    IQ4Nl,   // 20
    IQ3S,    // 21
    IQ2S,    // 22
    IQ4Xs,   // 23
    Iq1M,    // 29
    Bf16,    // 30
    Tq1_0,   // 34
    Mxfp4,   // 39
    Q2_0,    // 42 — ブロック 64 要素 (QK2_0)、Swift 1.5 の ffn_down_exps が使う
}

impl GgmlType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            16 => GgmlType::IQ2Xxs,
            17 => GgmlType::IQ2Xs,
            18 => GgmlType::IQ3Xxs,
            19 => GgmlType::IQ1S,
            20 => GgmlType::IQ4Nl,
            21 => GgmlType::IQ3S,
            22 => GgmlType::IQ2S,
            23 => GgmlType::IQ4Xs,
            // IQ1_M = 29 / TQ1_0 = 34 は ggml.h の enum と gguf-py の
            // constants.py で一致を確認した値 (2026-10-21)。type 24 は I8、
            // 29 は IQ1_M であって TQ1_0 ではない。番号を入れ替えないこと。
            29 => GgmlType::Iq1M,
            30 => GgmlType::Bf16,
            34 => GgmlType::Tq1_0,
            39 => GgmlType::Mxfp4,
            42 => GgmlType::Q2_0,
            _ => return None,
        })
    }

    /// 型の短い名前 (カウンタ・テレメトリの表示用)。ggml.h の enum 名に合わせる。
    ///
    /// `Debug` (`IQ4Xs` / `Q4K` など Rust の variant 名) とは綴りが違うので、
    /// 実機の GGUF ヘッダと照合するときはこちらを使う。
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q2K => "Q2_K",
            GgmlType::Q3K => "Q3_K",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
            GgmlType::IQ2Xxs => "IQ2_XXS",
            GgmlType::IQ2Xs => "IQ2_XS",
            GgmlType::IQ3Xxs => "IQ3_XXS",
            GgmlType::IQ1S => "IQ1_S",
            GgmlType::IQ4Nl => "IQ4_NL",
            GgmlType::IQ3S => "IQ3_S",
            GgmlType::IQ2S => "IQ2_S",
            GgmlType::IQ4Xs => "IQ4_XS",
            GgmlType::Iq1M => "IQ1_M",
            GgmlType::Bf16 => "BF16",
            GgmlType::Tq1_0 => "TQ1_0",
            GgmlType::Mxfp4 => "MXFP4",
            GgmlType::Q2_0 => "Q2_0",
        }
    }

    /// 型 id (ggml.h の enum 値)。[`GgmlType::from_id`] と対になる
    /// (カウンタに積んで後で [`GgmlType`] に戻すのに使う)。
    pub fn id(self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q2K => 10,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::IQ2Xxs => 16,
            GgmlType::IQ2Xs => 17,
            GgmlType::IQ3Xxs => 18,
            GgmlType::IQ1S => 19,
            GgmlType::IQ4Nl => 20,
            GgmlType::IQ3S => 21,
            GgmlType::IQ2S => 22,
            GgmlType::IQ4Xs => 23,
            GgmlType::Iq1M => 29,
            GgmlType::Bf16 => 30,
            GgmlType::Tq1_0 => 34,
            GgmlType::Mxfp4 => 39,
            GgmlType::Q2_0 => 42,
        }
    }

    /// ブロックあたり要素数 (ggml_blck_size)
    pub fn blck_size(self) -> usize {
        match self {
            GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => 1,
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1
            | GgmlType::Q8_0 | GgmlType::IQ4Nl | GgmlType::Mxfp4 => 32,
            // Q2_0 は K 系ではなく 64 要素ブロック (ggml QK2_0)
            GgmlType::Q2_0 => 64,
            // K 系・IQ 系・TQ 系スーパーブロック
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K
            | GgmlType::IQ2Xxs | GgmlType::IQ2Xs | GgmlType::IQ3Xxs | GgmlType::IQ1S
            | GgmlType::IQ3S | GgmlType::IQ2S | GgmlType::IQ4Xs | GgmlType::Iq1M
            | GgmlType::Tq1_0 => 256,
        }
    }

    /// ブロックあたりバイト数 (ggml_type_size)
    pub fn type_size(self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 | GgmlType::Bf16 => 2,
            GgmlType::Q4_0 | GgmlType::IQ4Nl | GgmlType::Mxfp4 => 18,
            GgmlType::Q4_1 => 20,
            GgmlType::Q5_0 => 22,
            GgmlType::Q5_1 => 24,
            GgmlType::Q8_0 => 34,
            // block_q2_0 = d(f16) + qs[16] = 18 (QK2_0 = 64)
            GgmlType::Q2_0 => 18,
            GgmlType::Q2K => 84,
            GgmlType::Q3K => 110,
            GgmlType::Q4K => 144,
            GgmlType::Q5K => 176,
            GgmlType::Q6K => 210,
            GgmlType::IQ2Xxs => 66,
            GgmlType::IQ2Xs => 74,
            GgmlType::IQ3Xxs => 98,
            GgmlType::IQ1S => 50,
            GgmlType::IQ3S => 110,
            GgmlType::IQ2S => 82,
            GgmlType::IQ4Xs => 136,
            // block_iq1_m = qs[32] + qh[16] + scales[8] = 56 (d は scales に分散)
            GgmlType::Iq1M => 56,
            // block_tq1_0 = qs[48] + qh[4] + d(f16) = 54
            GgmlType::Tq1_0 => 54,
        }
    }
}

const QK_K: usize = 256;
/// ggml-common.h: #define IQ1S_DELTA 0.125f
const IQ1S_DELTA: f32 = 0.125;

#[inline]
fn f16_to_f32(bytes: &[u8]) -> f32 {
    half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32()
}

#[inline]
fn rd_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// [`dequant_block_into`] が扱える型か (スカラー経路に流せるかの判定)。
///
/// 2026-10-31: 旧実装は `GgufLinear` / attn / qsa / ssm の各所に「融合カーネルが
/// ある型だけ」を通す閉じた whitelist を持っていた。Swift 1.5 の重みは層ごとに
/// Q4_K / Q5_K / Q6_K / IQ4_XS が混在し、expert も IQ3_S / IQ2_S / IQ2_0 と
/// ばらけるため、**デコードさえできれば通す** 方針へ変えた。
pub fn dequant_supported(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::Bf16
            | GgmlType::Q2_0
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Q8_0
            | GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::IQ2Xxs
            | GgmlType::IQ2Xs
            | GgmlType::IQ2S
            | GgmlType::IQ3Xxs
            | GgmlType::IQ3S
            | GgmlType::IQ1S
            | GgmlType::IQ4Nl
            | GgmlType::IQ4Xs
            | GgmlType::Iq1M
            | GgmlType::Tq1_0
            | GgmlType::Mxfp4
    )
}

/// 1 行 (n 要素・n はブロックサイズの倍数) を f32 へデ量子化する。
/// `blocks` は n / blck_size ブロック分の連続バイト列。
pub fn dequant_row(ty: GgmlType, blocks: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; n];
    dequant_row_into(ty, blocks, &mut out);
    out
}

/// P42: [`dequant_row`] の「出力先は呼び出し側が持つ」版。行長は `out.len()`。
///
/// 本体は 1 つだけ (`dequant_row` はこれに `vec![0f32; n]` を渡すだけ) なので、
/// 行ごとに `Vec` を確保しない呼び出しでも **`dequant_row` とビット一致**する。
/// `token_embd` のオンデマンド gather (EmbedTable::Q4Rows) が decode/prefill の
/// 1 行復元に使う (行バッファを使い回す)。
pub fn dequant_row_into(ty: GgmlType, blocks: &[u8], out: &mut [f32]) {
    let n = out.len();
    let bs = ty.blck_size();
    let ts = ty.type_size();
    debug_assert_eq!(blocks.len(), (n / bs) * ts, "dequant_row: size mismatch");
    match ty {
        GgmlType::F32 => {
            for (o, c) in out.iter_mut().zip(blocks.chunks_exact(4)) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        GgmlType::F16 => {
            for (o, c) in out.iter_mut().zip(blocks.chunks_exact(2)) {
                *o = f16_to_f32(c);
            }
        }
        GgmlType::Bf16 => {
            // bf16 は f32 の上位 16bit
            for (o, c) in out.iter_mut().zip(blocks.chunks_exact(2)) {
                *o = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        _ => {
            for (i, blk) in blocks.chunks_exact(ts).enumerate() {
                dequant_block(ty, blk, &mut out[i * bs..(i + 1) * bs]);
            }
        }
    }
}

/// 1 ブロックだけを f32 へ開く ([`dequant_row_into`] の 1 ブロック版)。
///
/// 中身は [`dequant_row_into`] が行ごとに呼ぶのと同じ [`dequant_block`] なので、
/// 同じバイト列からは同じ f32 列になる (ビット一致)。バッチ GEMV が
/// 「ブロックを 1 回だけ開いて全トークンへ配る」ために使う。
pub fn dequant_block_into(ty: GgmlType, block: &[u8], out: &mut [f32]) {
    let bs = ty.blck_size();
    let ts = ty.type_size();
    debug_assert_eq!(block.len(), ts, "dequant_block_into: ブロック長");
    debug_assert_eq!(out.len(), bs, "dequant_block_into: 出力長");
    match ty {
        GgmlType::F32 => out[0] = f32::from_le_bytes([block[0], block[1], block[2], block[3]]),
        GgmlType::F16 => out[0] = f16_to_f32(&block[0..2]),
        GgmlType::Bf16 => {
            out[0] = f32::from_bits((u16::from_le_bytes([block[0], block[1]]) as u32) << 16)
        }
        _ => dequant_block(ty, block, out),
    }
}

/// 1 ブロックのデ量子化 (llama.cpp のループ本体と 1:1 対応・出力順も同一)。
fn dequant_block(ty: GgmlType, x: &[u8], y: &mut [f32]) {
    match ty {
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => unreachable!(),

        // ---- Q2_0: d + qs[16] (64 要素、1 要素 2 bit、値は (q-1)*d) -----------
        // llama.cpp ggml-quants.c の dequantize_row_q2_0 と 1:1。
        GgmlType::Q2_0 => {
            let d = f16_to_f32(&x[0..2]);
            for j in 0..64 {
                let byte = x[2 + j / 4];
                let q = ((byte >> ((j % 4) * 2)) & 0x03) as i32;
                y[j] = (q - 1) as f32 * d;
            }
        }

        // ---- Q4_0: d + qs[16] -------------------------------------------------
        GgmlType::Q4_0 => {
            let d = f16_to_f32(&x[0..2]);
            for j in 0..16 {
                let x0 = (x[2 + j] & 0x0f) as i32 - 8;
                let x1 = (x[2 + j] >> 4) as i32 - 8;
                y[j] = x0 as f32 * d;
                y[j + 16] = x1 as f32 * d;
            }
        }

        // ---- Q4_1: d + m + qs[16] --------------------------------------------
        GgmlType::Q4_1 => {
            let d = f16_to_f32(&x[0..2]);
            let m = f16_to_f32(&x[2..4]);
            for j in 0..16 {
                let x0 = (x[4 + j] & 0x0f) as i32;
                let x1 = (x[4 + j] >> 4) as i32;
                y[j] = x0 as f32 * d + m;
                y[j + 16] = x1 as f32 * d + m;
            }
        }

        // ---- Q5_0: d + qh[4] + qs[16] ----------------------------------------
        GgmlType::Q5_0 => {
            let d = f16_to_f32(&x[0..2]);
            let qh = rd_u32(x, 2);
            for j in 0..16 {
                let xh0 = (((qh >> j) << 4) & 0x10) as u8 as i32;
                let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
                let x0 = (((x[6 + j] & 0x0f) as i32) | xh0) - 16;
                let x1 = (((x[6 + j] >> 4) as i32) | xh1) - 16;
                y[j] = x0 as f32 * d;
                y[j + 16] = x1 as f32 * d;
            }
        }

        // ---- Q5_1: d + m + qh[4] + qs[16] ------------------------------------
        GgmlType::Q5_1 => {
            let d = f16_to_f32(&x[0..2]);
            let m = f16_to_f32(&x[2..4]);
            let qh = rd_u32(x, 4);
            for j in 0..16 {
                let xh0 = (((qh >> j) << 4) & 0x10) as u8 as i32;
                let xh1 = ((qh >> (j + 12)) & 0x10) as i32;
                let x0 = ((x[8 + j] & 0x0f) as i32) | xh0;
                let x1 = ((x[8 + j] >> 4) as i32) | xh1;
                y[j] = x0 as f32 * d + m;
                y[j + 16] = x1 as f32 * d + m;
            }
        }

        // ---- Q8_0: d + qs[32] -------------------------------------------------
        GgmlType::Q8_0 => {
            let d = f16_to_f32(&x[0..2]);
            for j in 0..32 {
                y[j] = x[2 + j] as i8 as f32 * d;
            }
        }

        // ---- Q2_K: scales[16] + qs[64] + d + dmin ------------------------------
        GgmlType::Q2K => {
            let d = f16_to_f32(&x[80..82]);
            let dmin = f16_to_f32(&x[82..84]);
            let mut yi = 0usize;
            let mut is = 0usize;
            for n in (0..QK_K).step_by(128) {
                let q = &x[16 + n / 4..16 + n / 4 + 32];
                let mut shift = 0u32;
                for _ in 0..4 {
                    let sc = x[is];
                    is += 1;
                    let (dl, ml) = (d * ((sc & 0xf) as f32), dmin * ((sc >> 4) as f32));
                    for l in 0..16 {
                        let v = ((q[l] >> shift) & 3) as i8;
                        y[yi] = dl * v as f32 - ml;
                        yi += 1;
                    }
                    let sc = x[is];
                    is += 1;
                    let (dl, ml) = (d * ((sc & 0xf) as f32), dmin * ((sc >> 4) as f32));
                    for l in 0..16 {
                        let v = ((q[l + 16] >> shift) & 3) as i8;
                        y[yi] = dl * v as f32 - ml;
                        yi += 1;
                    }
                    shift += 2;
                }
            }
        }

        // ---- Q3_K: hmask[32] + qs[64] + scales[12] + d -------------------------
        GgmlType::Q3K => {
            let d_all = f16_to_f32(&x[108..110]);
            let kmask1: u32 = 0x03030303;
            let kmask2: u32 = 0x0f0f0f0f;
            // scales[12] → aux32[4] (LE) + 6bit 詰め替え (C コードと同一)
            let mut aux = [
                rd_u32(x, 96),
                rd_u32(x, 100),
                rd_u32(x, 104),
                0u32,
            ];
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            let scales: [i8; 16] = {
                let mut s = [0i8; 16];
                for (i, a) in aux.iter().enumerate() {
                    s[i * 4] = *a as u8 as i8;
                    s[i * 4 + 1] = (*a >> 8) as u8 as i8;
                    s[i * 4 + 2] = (*a >> 16) as u8 as i8;
                    s[i * 4 + 3] = (*a >> 24) as u8 as i8;
                }
                s
            };
            let mut yi = 0usize;
            let mut is = 0usize;
            let mut m: u8 = 1;
            // hm は n/j で進めず m ビットだけで全 256 要素をカバーする (C コード準拠)
            let hm = &x[0..32];
            for n in (0..QK_K).step_by(128) {
                let q = &x[32 + n / 4..32 + n / 4 + 32];
                let mut shift = 0u32;
                for _ in 0..4 {
                    let dl = d_all * (scales[is] - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let v = ((q[l] >> shift) & 3) as i8 - if hm[l] & m != 0 { 0 } else { 4 };
                        y[yi] = dl * v as f32;
                        yi += 1;
                    }
                    let dl = d_all * (scales[is] - 32) as f32;
                    is += 1;
                    for l in 0..16 {
                        let v = ((q[l + 16] >> shift) & 3) as i8
                            - if hm[l + 16] & m != 0 { 0 } else { 4 };
                        y[yi] = dl * v as f32;
                        yi += 1;
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
        }

        // ---- Q4_K: d + dmin + scales[12] + qs[128] ------------------------------
        GgmlType::Q4K => {
            let d = f16_to_f32(&x[0..2]);
            let dmin = f16_to_f32(&x[2..4]);
            let mut yi = 0usize;
            let mut is = 0usize;
            for j in (0..QK_K).step_by(64) {
                let q = &x[16 + j / 2..16 + j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &x[4..16]);
                let (d1, m1) = (d * sc as f32, dmin * m as f32);
                let (sc, m) = get_scale_min_k4(is + 1, &x[4..16]);
                let (d2, m2) = (d * sc as f32, dmin * m as f32);
                for l in 0..32 {
                    y[yi] = d1 * ((q[l] & 0xf) as f32) - m1;
                    yi += 1;
                }
                for l in 0..32 {
                    y[yi] = d2 * ((q[l] >> 4) as f32) - m2;
                    yi += 1;
                }
                is += 2;
            }
        }

        // ---- Q5_K: d + dmin + scales[12] + qh[32] + qs[128] ---------------------
        GgmlType::Q5K => {
            let d = f16_to_f32(&x[0..2]);
            let dmin = f16_to_f32(&x[2..4]);
            // レイアウト: d(2) + dmin(2) + scales[12] + qh[32] + qs[128]
            let qh = &x[16..48];
            let mut yi = 0usize;
            let mut is = 0usize;
            let mut u1: u8 = 1;
            let mut u2: u8 = 2;
            for j in (0..QK_K).step_by(64) {
                let ql = &x[48 + j / 2..48 + j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &x[4..16]);
                let (d1, m1) = (d * sc as f32, dmin * m as f32);
                let (sc, m) = get_scale_min_k4(is + 1, &x[4..16]);
                let (d2, m2) = (d * sc as f32, dmin * m as f32);
                for l in 0..32 {
                    let v = ((ql[l] & 0xf) as i32 + if qh[l] & u1 != 0 { 16 } else { 0 }) as f32;
                    y[yi] = d1 * v - m1;
                    yi += 1;
                }
                for l in 0..32 {
                    let v = ((ql[l] >> 4) as i32 + if qh[l] & u2 != 0 { 16 } else { 0 }) as f32;
                    y[yi] = d2 * v - m2;
                    yi += 1;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }

        // ---- Q6_K: ql[128] + qh[64] + scales i8[16] + d -------------------------
        GgmlType::Q6K => {
            let d = f16_to_f32(&x[208..210]);
            let mut yi = 0usize;
            let sc_base = QK_K / 2 + QK_K / 4; // 192
            for n in (0..QK_K).step_by(128) {
                let ql = &x[n / 2..n / 2 + 64];
                let qh = &x[QK_K / 2 + n / 4..QK_K / 2 + n / 4 + 32];
                // sc は n あたり +8 (scales 16 バイトを 2 回に分けて使用)
                let sc = &x[sc_base + n / 16..sc_base + n / 16 + 8];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xf) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
                    let q2 = ((ql[l + 32] & 0xf) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                    let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                    let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                    y[yi + l] = d * sc[is] as i8 as f32 * q1 as f32;
                    y[yi + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                    y[yi + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                    y[yi + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                }
                yi += 128;
            }
        }

        // ---- IQ2_XXS: d + qs (u16[32]) ------------------------------------------
        GgmlType::IQ2Xxs => {
            let d = f16_to_f32(&x[0..2]);
            let qs = &x[2..66]; // u16[32] = 64 bytes
            let mut yi = 0usize;
            for ib32 in 0..QK_K / 32 {
                let base = ib32 * 8;
                let aux0 = rd_u32(qs, base);
                let aux1 = rd_u32(qs, base + 4);
                let aux8 = aux0.to_le_bytes();
                let db = d * (0.5f32 + ((aux1 >> 28) as f32)) * 0.25f32;
                for l in 0..4 {
                    let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        let sgn = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                        y[yi + j] = db * grid[j] as f32 * sgn;
                    }
                    yi += 8;
                }
            }
        }

        // ---- IQ2_XS: d + qs (u16[32]) + scales[8] --------------------------------
        GgmlType::IQ2Xs => {
            let d = f16_to_f32(&x[0..2]);
            let qs = &x[2..66];
            let scales = &x[66..74];
            let mut yi = 0usize;
            for ib32 in 0..QK_K / 32 {
                let db0 = d * (0.5f32 + (scales[ib32] & 0xf) as f32) * 0.25f32;
                let db1 = d * (0.5f32 + (scales[ib32] >> 4) as f32) * 0.25f32;
                for l in 0..4 {
                    let idx = rd_u16(qs, ib32 * 8 + 2 * l) as usize;
                    let grid = IQ2XS_GRID[idx & 511].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[idx >> 9];
                    let db = if l / 2 == 0 { db0 } else { db1 };
                    for j in 0..8 {
                        let sgn = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                        y[yi + j] = db * grid[j] as f32 * sgn;
                    }
                    yi += 8;
                }
            }
        }

        // ---- IQ2_S: d + qs[64] + qh[8] + scales[8] --------------------------------
        GgmlType::IQ2S => {
            let d = f16_to_f32(&x[0..2]);
            let qs = &x[2..66];
            let qh = &x[66..74];
            // signs は qs + QK_K/8 = ブロックオフセット 34 (qs 配列の後半 32 バイト)
            let signs = &x[34..66];
            let scales = &x[74..82];
            let mut yi = 0usize;
            for ib32 in 0..QK_K / 32 {
                let db0 = d * (0.5f32 + (scales[ib32] & 0xf) as f32) * 0.25f32;
                let db1 = d * (0.5f32 + (scales[ib32] >> 4) as f32) * 0.25f32;
                for l in 0..4 {
                    let idx = (qs[ib32 * 4 + l] as usize)
                        | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                    let grid = IQ2S_GRID[idx].to_le_bytes();
                    let sg = signs[ib32 * 4 + l];
                    let db = if l / 2 == 0 { db0 } else { db1 };
                    for j in 0..8 {
                        let sgn = if sg & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                        y[yi + j] = db * grid[j] as f32 * sgn;
                    }
                    yi += 8;
                }
            }
        }

        // ---- IQ3_XXS: d + qs[96] (後半 32 バイトが scales_and_signs) ---------------
        GgmlType::IQ3Xxs => {
            let d = f16_to_f32(&x[0..2]);
            let qs = &x[2..98];
            let ss = &x[2 + QK_K / 4..2 + QK_K / 4 + 32];
            let mut yi = 0usize;
            for ib32 in 0..QK_K / 32 {
                let aux32 = rd_u32(ss, ib32 * 4);
                let db = d * (0.5f32 + ((aux32 >> 28) as f32)) * 0.5f32;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l] as usize].to_le_bytes();
                    let g2 = IQ3XXS_GRID[qs[ib32 * 8 + 2 * l + 1] as usize].to_le_bytes();
                    for j in 0..4 {
                        let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                        let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                        y[yi + j] = db * g1[j] as f32 * s0;
                        y[yi + j + 4] = db * g2[j] as f32 * s1;
                    }
                    yi += 8;
                }
            }
        }

        // ---- IQ3_S: d + qs[64] + qh[8] + signs[32] + scales[4] --------------------
        GgmlType::IQ3S => {
            let d = f16_to_f32(&x[0..2]);
            let qs = &x[2..66];
            let qh = &x[66..74];
            let signs = &x[74..106];
            let scales = &x[106..110];
            let mut yi = 0usize;
            let mut qsi = 0usize;
            let mut sgi = 0usize;
            for ib32 in (0..QK_K / 32).step_by(2) {
                let db1 = d * (1.0f32 + 2.0f32 * (scales[ib32 / 2] & 0xf) as f32);
                let db2 = d * (1.0f32 + 2.0f32 * (scales[ib32 / 2] >> 4) as f32);
                for l in 0..4 {
                    let g1 = IQ3S_GRID
                        [(qs[qsi + 2 * l] as usize) | (((qh[ib32] as usize) << (8 - 2 * l)) & 256)]
                        .to_le_bytes();
                    let g2 = IQ3S_GRID
                        [(qs[qsi + 2 * l + 1] as usize)
                            | (((qh[ib32] as usize) << (7 - 2 * l)) & 256)]
                        .to_le_bytes();
                    let sg = signs[sgi + l];
                    for j in 0..4 {
                        let s0 = if sg & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                        let s1 = if sg & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                        y[yi + j] = db1 * g1[j] as f32 * s0;
                        y[yi + j + 4] = db1 * g2[j] as f32 * s1;
                    }
                    yi += 8;
                }
                qsi += 8;
                sgi += 4;
                for l in 0..4 {
                    let g1 = IQ3S_GRID
                        [(qs[qsi + 2 * l] as usize)
                            | (((qh[ib32 + 1] as usize) << (8 - 2 * l)) & 256)]
                        .to_le_bytes();
                    let g2 = IQ3S_GRID
                        [(qs[qsi + 2 * l + 1] as usize)
                            | (((qh[ib32 + 1] as usize) << (7 - 2 * l)) & 256)]
                        .to_le_bytes();
                    let sg = signs[sgi + l];
                    for j in 0..4 {
                        let s0 = if sg & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                        let s1 = if sg & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                        y[yi + j] = db2 * g1[j] as f32 * s0;
                        y[yi + j + 4] = db2 * g2[j] as f32 * s1;
                    }
                    yi += 8;
                }
                qsi += 8;
                sgi += 4;
            }
        }

        // ---- IQ1_S: d + qs[32] + qh (u16[8]) --------------------------------------
        GgmlType::IQ1S => {
            let d = f16_to_f32(&x[0..2]);
            let qs = &x[2..34];
            let mut yi = 0usize;
            for ib in 0..QK_K / 32 {
                let qh = rd_u16(x, 34 + 2 * ib);
                let dl = d * (2.0f32 * (((qh >> 12) & 7) as f32) + 1.0f32);
                let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for l in 0..4 {
                    let idx = (qs[ib * 4 + l] as usize) | ((((qh >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    for j in 0..8 {
                        y[yi + j] = dl * ((grid[j] as i8 as f32) + delta);
                    }
                    yi += 8;
                }
            }
        }

        // ---- TQ1_0: qs[48] + qh[4] + d (f16) --------------------------------------
        //
        // 形式は llama.cpp ggml-common.h / ggml-quants.c (MIT) 準拠・実装は独自。
        // 1 バイトに三値 (-1/0/+1) を 5 つ詰める。バイトは 0..242 の三進数を
        // 0..256 へ切り上げで写したもので、値は byte/256 の三進小数の桁
        // (最上位が桁 0) を -1/0/+1 に戻したもの。ggml は割り算を避けて
        //   xi = ((u8)(byte * 3^n) * 3) >> 8
        // で桁を取り出す。ここも同じ整数手順を使う (ビット一致のため)。
        // 要素順: qs の先頭 32 バイト x 5 桁 -> 残り 16 バイト x 5 桁 -> qh 4 x 4 桁。
        GgmlType::Tq1_0 => {
            const POW3: [u32; 5] = [1, 3, 9, 27, 81];
            let d = f16_to_f32(&x[52..54]);
            let mut yi = 0usize;
            for (start, count) in [(0usize, 32usize), (32, 16)] {
                for n in 0..5 {
                    for m in 0..count {
                        // C: uint8_t q = qs * pow3[n]; (int で乗じて u8 へ切り詰め)
                        let q = (x[start + m] as u32 * POW3[n]) as u8;
                        let xi = ((q as u16 * 3) >> 8) as i32;
                        y[yi] = (xi - 1) as f32 * d;
                        yi += 1;
                    }
                }
            }
            for n in 0..4 {
                for j in 0..4 {
                    let q = (x[48 + j] as u32 * POW3[n]) as u8;
                    let xi = ((q as u16 * 3) >> 8) as i32;
                    y[yi] = (xi - 1) as f32 * d;
                    yi += 1;
                }
            }
            debug_assert_eq!(yi, QK_K);
        }

        // ---- IQ1_M: qs[32] + qh[16] + scales[8] -----------------------------------
        //
        // 形式は llama.cpp ggml-common.h / ggml-quants.c (MIT) 準拠・実装は独自。
        // ブロックに d は無く、scales[8] を u16[4] (LE) として読んだ上位ニブルに
        // f16 の d が 4 分割で入っている。32 要素ごとに 2 つの 3bit scale
        // (2*s+1 倍)、8 要素グリッド iq1s_grid (2048 エントリ) の 11bit
        // インデックス、±0.125 のデルタで値を組み立てる。
        GgmlType::Iq1M => {
            let qs = &x[0..32];
            let qh = &x[32..48];
            let sc = [rd_u16(x, 48), rd_u16(x, 50), rd_u16(x, 52), rd_u16(x, 54)];
            // d は 4 ワードの上位ニブルに分散している (iq1m_scale_t の組み立てと同一)
            let d_bits = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00f0)
                | ((sc[2] >> 4) & 0x0f00)
                | (sc[3] & 0xf000);
            let d = half::f16::from_bits(d_bits).to_f32();
            let mut yi = 0usize;
            for ib in 0..QK_K / 32 {
                let s = sc[ib / 2];
                let shift = 6 * (ib % 2);
                let dl1 = d * ((2 * ((s >> shift) & 7) + 1) as f32);
                let dl2 = d * ((2 * ((s >> (shift + 3)) & 7) + 1) as f32);
                let (h0, h1) = (qh[2 * ib], qh[2 * ib + 1]);
                // 11bit グリッド番号: qs の 8bit + qh の 3bit (上位 / 中位)
                let idx = [
                    qs[4 * ib] as usize | (((h0 as usize) << 8) & 0x700),
                    qs[4 * ib + 1] as usize | (((h0 as usize) << 4) & 0x700),
                    qs[4 * ib + 2] as usize | (((h1 as usize) << 8) & 0x700),
                    qs[4 * ib + 3] as usize | (((h1 as usize) << 4) & 0x700),
                ];
                let delta = [
                    if h0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if h0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if h1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if h1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                ];
                for l in 0..4 {
                    let dl = if l < 2 { dl1 } else { dl2 };
                    let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                    for j in 0..8 {
                        y[yi + j] = dl * ((grid[j] as i8 as f32) + delta[l]);
                    }
                    yi += 8;
                }
            }
            debug_assert_eq!(yi, QK_K);
        }

        // ---- IQ4_NL: d + qs[16] (32 要素) ------------------------------------------
        GgmlType::IQ4Nl | GgmlType::Mxfp4 => {
            // MXFP4 は同じ 4bit テーブル構造だがテーブルが異なる (kvalues_mxfp4)。
            // UD-GGUF では未使用のため IQ4_NL テーブルのみ実装し、MXFP4 はエラーにする。
            if ty == GgmlType::Mxfp4 {
                unreachable!("MXFP4 は未対応 (UD-GGUF では未使用)");
            }
            let d = f16_to_f32(&x[0..2]);
            for j in 0..16 {
                y[j] = d * KVALUES_IQ4NL[(x[2 + j] & 0xf) as usize] as f32;
                y[j + 16] = d * KVALUES_IQ4NL[(x[2 + j] >> 4) as usize] as f32;
            }
        }

        // ---- IQ4_XS: d + scales_h(u16) + scales_l[4] + qs[128] ---------------------
        GgmlType::IQ4Xs => {
            let d = f16_to_f32(&x[0..2]);
            let scales_h = rd_u16(x, 2);
            let scales_l = &x[4..8];
            let qs = &x[8..136];
            let mut yi = 0usize;
            for ib in 0..QK_K / 32 {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as usize
                    | (((scales_h >> (2 * ib)) & 3) as usize) << 4;
                let dl = d * (ls as i32 - 32) as f32;
                for j in 0..16 {
                    y[yi + j] = dl * KVALUES_IQ4NL[(qs[ib * 16 + j] & 0xf) as usize] as f32;
                    y[yi + j + 16] = dl * KVALUES_IQ4NL[(qs[ib * 16 + j] >> 4) as usize] as f32;
                }
                yi += 32;
            }
        }
    }
}

/// q4_K/q5_K の 6bit scale/min 取り出し (ggml-quants.c get_scale_min_k4 と同一)。
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xf) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

// ---------------------------------------------------------------------------
// task 927: 全型 x 境界ブロックの f64 参照突き合わせ (テスト専用)
// ---------------------------------------------------------------------------
//
// `dequant_block` の正しさは `tests/dequant_ref.rs` が llama.cpp の C 実装と
// のビット一致で固定している (pseudo-random なブロック列 + fixture)。ここに
// 置くのは別系統の参照で、**ブロック形式の定義から要素番号を引き直して書いた
// f64 版**と、その比較行列である。役割は 2 つ。
//
//   1. fixture では踏みにくい境界値 (全ゼロ / f16 最大 / 負のゼロ / f16 最小
//      非正規 / 符号が交互のデータ) で全型を突き合わせる。fixture は C 側の
//      再ビルドが要るので、境界値の追加はどうしてもここに置くことになる。
//   2. スケールの位置やニブルの向きを取り違える編集を、fixture を触らずに落とす。
//
// 算術は f64。f16 スケールと小さい整数の積は f32 でも厳密なので、一致は
// ほぼビット一致になる (許容は AGENTS.md の絶対 1e-4 + 相対 1e-5)。
// 格子・符号表は本体と同じ `kquant_tables` を使う (表は gen_kquant_tables.py が
// ggml-common.h から生成しており、この検査の対象外)。
#[cfg(test)]
pub(crate) mod refdev {
    use super::*;
    use crate::quant_matrix::ALL_TYPES;

    #[inline]
    pub(crate) fn f16_at(x: &[u8], off: usize) -> f64 {
        half::f16::from_bits(u16::from_le_bytes([x[off], x[off + 1]])).to_f32() as f64
    }

    #[inline]
    fn u16_at(x: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([x[off], x[off + 1]])
    }

    #[inline]
    fn u32_at(x: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([x[off], x[off + 1], x[off + 2], x[off + 3]])
    }

    /// 符号表 1 バイト -> 1 要素の符号 (ビットが立っていれば負)。
    #[inline]
    fn sgn(byte: u8, j: usize) -> f64 {
        if byte & KMASK_IQ2XS[j] != 0 {
            -1.0
        } else {
            1.0
        }
    }

    /// q4_K/q5_K の 6bit scale / min (ビット配置から書き直した版)。
    /// `sc` は scales[12]。
    fn scale_min_k4(j: usize, sc: &[u8]) -> (f64, f64) {
        if j < 4 {
            ((sc[j] & 63) as f64, (sc[j + 4] & 63) as f64)
        } else {
            // j >= 4: low 4bit は sc[j+4] の下位、上位 2bit は前の 4 バイトの
            // 上位 2bit に散っている。
            (
                ((sc[j + 4] & 0x0f) | ((sc[j - 4] >> 6) << 4)) as f64,
                ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f64,
            )
        }
    }

    /// 1 ブロックを f64 で復元する独立参照。
    ///
    /// 索引は「要素番号からブロック内の位置を引き直す」形で書いてある
    /// (本体は ggml と同じループ順なので、同じ取り違えを写しにくくするため)。
    pub(crate) fn dequant_ref(ty: GgmlType, x: &[u8]) -> Vec<f64> {
        let n = ty.blck_size();
        let mut y = vec![0f64; n];
        match ty {
            GgmlType::F32 => y[0] = f32::from_le_bytes([x[0], x[1], x[2], x[3]]) as f64,
            GgmlType::F16 => y[0] = f16_at(x, 0),
            GgmlType::Bf16 => y[0] = f32::from_bits((u16_at(x, 0) as u32) << 16) as f64,

            // ---- 32 要素ブロック -------------------------------------------------
            GgmlType::Q2_0 => {
                let d = f16_at(x, 0);
                for i in 0..64 {
                    let q = ((x[2 + i / 4] >> (2 * (i % 4))) & 3) as i32;
                    y[i] = (q - 1) as f64 * d;
                }
            }
            GgmlType::Q4_0 => {
                let d = f16_at(x, 0);
                for i in 0..32 {
                    let q = if i < 16 {
                        x[2 + i] & 0x0f
                    } else {
                        x[2 + i - 16] >> 4
                    };
                    y[i] = (q as f64 - 8.0) * d;
                }
            }
            GgmlType::Q4_1 => {
                let (d, m) = (f16_at(x, 0), f16_at(x, 2));
                for i in 0..32 {
                    let q = if i < 16 {
                        x[4 + i] & 0x0f
                    } else {
                        x[4 + i - 16] >> 4
                    };
                    y[i] = q as f64 * d + m;
                }
            }
            GgmlType::Q5_0 | GgmlType::Q5_1 => {
                let five = ty == GgmlType::Q5_0;
                let d = f16_at(x, 0);
                let m = if five { 0.0 } else { f16_at(x, 2) };
                let qh_off = if five { 2 } else { 4 };
                let qs_off = qh_off + 4;
                let qh = u32_at(x, qh_off);
                for i in 0..32 {
                    let bit = ((qh >> i) & 1) as u8;
                    let nib = (x[qs_off + i % 16] >> (4 * (i / 16))) & 0x0f;
                    let q = (nib | (bit << 4)) as i32;
                    y[i] = if five {
                        (q - 16) as f64 * d
                    } else {
                        q as f64 * d + m
                    };
                }
            }
            GgmlType::Q8_0 => {
                let d = f16_at(x, 0);
                for i in 0..32 {
                    y[i] = x[2 + i] as i8 as f64 * d;
                }
            }
            GgmlType::IQ4Nl => {
                let d = f16_at(x, 0);
                for i in 0..32 {
                    let nib = (x[2 + i % 16] >> (4 * (i / 16))) & 0x0f;
                    y[i] = d * KVALUES_IQ4NL[nib as usize] as f64;
                }
            }

            // ---- K 系 (256 要素スーパーブロック) ---------------------------------
            GgmlType::Q2K => {
                let d = f16_at(x, 80);
                let dmin = f16_at(x, 82);
                for i in 0..256 {
                    let chunk = i / 128; // 2 つの 128 要素半ブロック
                    let j = (i % 128) / 32; // 4 つの 32 要素グループ
                    let g = (i % 32) / 16; // 16 要素でスケールが切り替わる
                    let l = i % 16;
                    let sc = x[chunk * 8 + j * 2 + g];
                    // qs は半ブロックあたり 32 バイト (16 要素 x 2)
                    let qb = x[16 + chunk * 32 + g * 16 + l];
                    let q = (qb >> (2 * j)) & 3;
                    y[i] = d * (sc & 0x0f) as f64 * q as f64 - dmin * (sc >> 4) as f64;
                }
            }
            GgmlType::Q3K => {
                let d_all = f16_at(x, 108);
                let sc = &x[96..108]; // scales[12]
                for i in 0..256 {
                    let chunk = i / 128;
                    let j = (i % 128) / 32;
                    let g = (i % 32) / 16;
                    let l = i % 16;
                    // 16 個の 6bit scale: 下位 4bit は sc[0..8] の 8 バイト
                    // (i<8 は下ニブル / i>=8 は上ニブル)、上位 2bit は sc[8..12]
                    // の 2bit x 4 段。
                    let idx = chunk * 8 + j * 2 + g;
                    let low = if idx < 8 {
                        sc[idx] & 0x0f
                    } else {
                        sc[idx - 8] >> 4
                    };
                    let high2 = (sc[8 + idx % 4] >> (2 * (idx / 4))) & 3;
                    let s = ((low | (high2 << 4)) as i8) as f64 - 32.0;
                    let hbit = 1u8 << (4 * chunk + j);
                    let qb = x[32 + chunk * 32 + g * 16 + l];
                    let q = ((qb >> (2 * j)) & 3) as f64;
                    let sub = if x[l + 16 * g] & hbit != 0 { 0.0 } else { 4.0 };
                    y[i] = d_all * s * (q - sub);
                }
            }
            GgmlType::Q4K | GgmlType::Q5K => {
                let five = ty == GgmlType::Q5K;
                let (d, dmin) = (f16_at(x, 0), f16_at(x, 2));
                let sc = &x[4..16];
                let (qh_off, qs_off) = if five { (16usize, 48usize) } else { (0, 16) };
                for i in 0..256 {
                    let j4 = i / 64; // 64 要素 x 4
                    let sub = (i % 64) / 32; // 32 要素ごとに 1 つの scale/min
                    let l = i % 32;
                    let (s, m) = scale_min_k4(j4 * 2 + sub, sc);
                    let qb = x[qs_off + j4 * 32 + l];
                    let q = if sub == 0 { qb & 0x0f } else { qb >> 4 } as f64;
                    let hi = if five {
                        let bit = 1u8 << (2 * j4 + sub);
                        if x[qh_off + l] & bit != 0 {
                            16.0
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                    y[i] = d * s * (q + hi) - dmin * m;
                }
            }
            GgmlType::Q6K => {
                let d = f16_at(x, 208);
                for i in 0..256 {
                    let chunk = i / 128;
                    let quad = (i % 128) / 32; // 32 要素 x 4 (ql/qh の使う場所が変わる)
                    let l = i % 32;
                    let is = l / 16;
                    let ql = x[chunk * 64 + if quad % 2 == 1 { 32 } else { 0 } + l];
                    let qh = x[128 + chunk * 32 + l];
                    let nib = if quad < 2 { ql & 0x0f } else { ql >> 4 };
                    let hi = (qh >> (2 * quad)) & 3;
                    let q = (nib as i32 | ((hi as i32) << 4)) - 32;
                    let s = x[192 + chunk * 8 + quad * 2 + is] as i8 as f64;
                    y[i] = d * s * q as f64;
                }
            }

            // ---- IQ 系 ------------------------------------------------------------
            GgmlType::IQ2Xxs => {
                let d = f16_at(x, 0);
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let aux0 = u32_at(x, 2 + ib * 8);
                    let aux1 = u32_at(x, 2 + ib * 8 + 4);
                    let db = d * (0.5 + ((aux1 >> 28) as f64)) * 0.25;
                    let gi = aux0.to_le_bytes()[l] as usize;
                    let grid = IQ2XXS_GRID[gi].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                    y[i] = db * grid[j] as f64 * sgn(signs, j);
                }
            }
            GgmlType::IQ2Xs => {
                let d = f16_at(x, 0);
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let idx = u16_at(x, 2 + ib * 8 + 2 * l) as usize;
                    let scb = x[66 + ib];
                    let nib = if l < 2 { scb & 0x0f } else { scb >> 4 };
                    let db = d * (0.5 + nib as f64) * 0.25;
                    let grid = IQ2XS_GRID[idx & 511].to_le_bytes();
                    let signs = KSIGNS_IQ2XS[idx >> 9];
                    y[i] = db * grid[j] as f64 * sgn(signs, j);
                }
            }
            GgmlType::IQ2S => {
                let d = f16_at(x, 0);
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let qb = x[2 + ib * 4 + l] as usize;
                    let idx = qb | (((x[66 + ib] as usize) << (8 - 2 * l)) & 0x300);
                    let scb = x[74 + ib];
                    let nib = if l < 2 { scb & 0x0f } else { scb >> 4 };
                    let db = d * (0.5 + nib as f64) * 0.25;
                    let grid = IQ2S_GRID[idx].to_le_bytes();
                    let signs = x[2 + 32 + ib * 4 + l];
                    y[i] = db * grid[j] as f64 * sgn(signs, j);
                }
            }
            GgmlType::IQ3Xxs => {
                let d = f16_at(x, 0);
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let aux = u32_at(x, 66 + ib * 4);
                    let db = d * (0.5 + ((aux >> 28) as f64)) * 0.5;
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                    // 8 要素 = 4bit グリッド 2 本 (前半 4 要素 / 後半 4 要素)
                    let gi = x[2 + ib * 8 + 2 * l + j / 4] as usize;
                    let grid = IQ3XXS_GRID[gi].to_le_bytes();
                    y[i] = db * grid[j % 4] as f64 * sgn(signs, j);
                }
            }
            GgmlType::IQ3S => {
                let d = f16_at(x, 0);
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let second = j >= 4;
                    let qb = x[2 + ib * 8 + 2 * l + second as usize] as usize;
                    let sh = if second { 7 - 2 * l } else { 8 - 2 * l };
                    let idx = qb | (((x[66 + ib] as usize) << sh) & 256);
                    let scb = x[106 + ib / 2];
                    let nib = if ib % 2 == 0 { scb & 0x0f } else { scb >> 4 };
                    let db = d * (1.0 + 2.0 * nib as f64);
                    let grid = IQ3S_GRID[idx].to_le_bytes();
                    let signs = x[74 + ib * 4 + l];
                    y[i] = db * grid[j % 4] as f64 * sgn(signs, j);
                }
            }
            GgmlType::IQ1S => {
                let d = f16_at(x, 0);
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let qh = u16_at(x, 34 + 2 * ib);
                    let dl = d * (2.0 * ((qh >> 12) & 7) as f64 + 1.0);
                    let delta = if qh & 0x8000 != 0 {
                        -IQ1S_DELTA as f64
                    } else {
                        IQ1S_DELTA as f64
                    };
                    let idx = (x[2 + ib * 4 + l] as usize) | ((((qh >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    y[i] = dl * (grid[j] as i8 as f64 + delta);
                }
            }
            GgmlType::IQ4Xs => {
                let d = f16_at(x, 0);
                let sh = u16_at(x, 2);
                for i in 0..256 {
                    let ib = i / 32;
                    let half = (i % 32) / 16;
                    let j = i % 16;
                    let low = (x[4 + ib / 2] >> (4 * (ib % 2))) & 0x0f;
                    let ls = (low as i32 | (((sh >> (2 * ib)) & 3) as i32) << 4) - 32;
                    let nib = (x[8 + ib * 16 + j] >> (4 * half)) & 0x0f;
                    y[i] = d * ls as f64 * KVALUES_IQ4NL[nib as usize] as f64;
                }
            }
            GgmlType::Iq1M => {
                let w = [u16_at(x, 48), u16_at(x, 50), u16_at(x, 52), u16_at(x, 54)];
                let d_bits = (w[0] >> 12)
                    | ((w[1] >> 8) & 0x00f0)
                    | ((w[2] >> 4) & 0x0f00)
                    | (w[3] & 0xf000);
                let d = half::f16::from_bits(d_bits).to_f32() as f64;
                for i in 0..256 {
                    let ib = i / 32;
                    let l = (i % 32) / 8;
                    let j = i % 8;
                    let s = w[ib / 2];
                    let shift = 6 * (ib % 2);
                    let nib = (s >> (shift + 3 * (l / 2))) & 7;
                    let dl = d * (2.0 * nib as f64 + 1.0);
                    let hb = x[32 + 2 * ib + l / 2];
                    let idx = (x[4 * ib + l] as usize)
                        | (((hb as usize) << (if l % 2 == 0 { 8 } else { 4 })) & 0x700);
                    let delta = if hb & if l % 2 == 0 { 0x08 } else { 0x80 } != 0 {
                        -IQ1S_DELTA as f64
                    } else {
                        IQ1S_DELTA as f64
                    };
                    let grid = IQ1S_GRID[idx].to_le_bytes();
                    y[i] = dl * (grid[j] as i8 as f64 + delta);
                }
            }
            GgmlType::Tq1_0 => {
                const POW3: [u32; 5] = [1, 3, 9, 27, 81];
                let d = f16_at(x, 52);
                for i in 0..256 {
                    // 先頭 32 バイト x 5 桁 -> 次の 16 バイト x 5 桁 -> qh 4 x 4 桁
                    let (src, n) = if i < 160 {
                        (x[i % 32], i / 32)
                    } else if i < 240 {
                        let k = i - 160;
                        (x[32 + k % 16], k / 16)
                    } else {
                        let k = i - 240;
                        (x[48 + k % 4], k / 4)
                    };
                    let q = (src as u32 * POW3[n]) as u8;
                    let xi = ((q as u16 * 3) >> 8) as i32;
                    y[i] = (xi - 1) as f64 * d;
                }
            }

            // 未対応 (dequant_block が unreachable!)。テストからは呼ばない。
            GgmlType::Mxfp4 => panic!("MXFP4 の参照実装は無い (未対応の型)"),
        }
        y
    }
    /// 境界ブロックの 1 バリエーション。
    pub(crate) struct EdgeCase {
        pub name: &'static str,
        /// f16 スケールに書く値 (スケールを持たない型では使わない)。
        pub bits: u16,
        /// 偶数バイト / 奇数バイトの値。交互パターンでニブルと符号を散らす。
        pub even: u8,
        pub odd: u8,
    }

    /// 全型に共通の境界ブロック (task 927 の 5 種 + データ最大)。
    pub(crate) const EDGE_CASES: [EdgeCase; 6] = [
        EdgeCase {
            name: "all_zero",
            bits: 0x0000,
            even: 0x00,
            odd: 0x00,
        },
        EdgeCase {
            name: "max_scale",
            bits: 0x7bff,
            even: 0xa5,
            odd: 0x5a,
        },
        EdgeCase {
            name: "neg_zero_scale",
            bits: 0x8000,
            even: 0x5a,
            odd: 0xa5,
        },
        EdgeCase {
            name: "subnormal_scale",
            bits: 0x0001,
            even: 0xa5,
            odd: 0x5a,
        },
        EdgeCase {
            name: "alt_signs",
            bits: 0x3c00,
            even: 0x5a,
            odd: 0xa5,
        },
        EdgeCase {
            name: "max_data",
            bits: 0x3c00,
            even: 0xff,
            odd: 0xff,
        },
    ];

    /// ブロック内で f16 スケールとして扱われるフィールドの位置。
    pub(crate) fn f16_scale_offsets(ty: GgmlType) -> &'static [usize] {
        use GgmlType as T;
        match ty {
            T::Q2K => &[80, 82],        // d + dmin (末尾)
            T::Q3K => &[108],           // d (末尾)
            T::Q4K | T::Q5K => &[0, 2], // d + dmin (先頭)
            T::Q6K => &[208],           // d (末尾)
            T::Tq1_0 => &[52],          // d (末尾)
            T::F32 | T::F16 | T::Bf16 | T::Iq1M | T::Mxfp4 => &[],
            _ => &[0], // ブロック先頭の d
        }
    }

    /// スケールを持たない生 float 型か。
    pub(crate) fn is_raw(ty: GgmlType) -> bool {
        matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16)
    }

    /// IQ1_M は d が scales[8] (u16 x 4) の上位ニブルに 4bit ずつ分散している。
    fn write_iq1m_d(b: &mut [u8], bits: u16) {
        for w in 0..4 {
            let off = 48 + 2 * w;
            let cur = u16::from_le_bytes([b[off], b[off + 1]]);
            let v = (cur & 0x0fff) | (((bits >> (4 * w)) & 0xf) << 12);
            b[off..off + 2].copy_from_slice(&v.to_le_bytes());
        }
    }

    /// 1 ブロックを境界値で埋める。
    pub(crate) fn build_block(ty: GgmlType, case: &EdgeCase) -> Vec<u8> {
        let ts = ty.type_size();
        let mut b: Vec<u8> = (0..ts)
            .map(|k| if k % 2 == 0 { case.even } else { case.odd })
            .collect();
        for &off in f16_scale_offsets(ty) {
            b[off..off + 2].copy_from_slice(&case.bits.to_le_bytes());
        }
        if ty == GgmlType::Iq1M {
            write_iq1m_d(&mut b, case.bits);
        }
        b
    }

    /// 本体と f64 参照を 1 ブロック突き合わせ、比較した要素数を返す。
    ///
    /// 許容は AGENTS.md の方針 (絶対 1e-4 + 相対 1e-5 のどちらかを満たす)。
    /// f16 スケール x 小さい整数は f32 でも厳密なので実際はほぼビット一致する。
    /// ゼロは符号まで比べる (f32 と f64 で符号則が同じことの確認)。
    pub(crate) fn check_block_parity(ty: GgmlType, block: &[u8], ctx: &str) -> usize {
        let n = ty.blck_size();
        let want = dequant_ref(ty, block);
        assert_eq!(want.len(), n, "{ctx}: 参照の要素数");
        let mut got = vec![0f32; n];
        dequant_row_into(ty, block, &mut got);
        for i in 0..n {
            let w = want[i];
            let g = got[i] as f64;
            if w.is_nan() {
                assert!(g.is_nan(), "{ctx}[{i}]: 参照は NaN、本体は {g:e}");
                continue;
            }
            assert!(g.is_finite(), "{ctx}[{i}]: 本体が非有限 {g:e} (参照 {w:e})");
            let abs = (g - w).abs();
            let rel = if w != 0.0 { abs / w.abs() } else { 0.0 };
            assert!(
                abs <= 1e-4 || rel <= 1e-5,
                "{ctx}[{i}]: 本体 {g:e} / 参照 {w:e} (絶対 {abs:e} 相対 {rel:e})"
            );
            if w == 0.0 {
                assert_eq!(
                    g.is_sign_negative(),
                    w.is_sign_negative(),
                    "{ctx}[{i}]: ゼロの符号が違う (本体 {g:e} / 参照 {w:e})"
                );
            }
        }
        n
    }

    /// パリティ行列の対象型 (CPU で復元できる全型から MXFP4 を除いたもの)。
    pub(crate) fn parity_types() -> Vec<GgmlType> {
        ALL_TYPES
            .iter()
            .copied()
            .filter(|t| dequant_supported(*t) && *t != GgmlType::Mxfp4)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::refdev::*;
    use super::*;
    use GgmlType as T;

    /// 全型 x 全境界ブロックを f64 参照と突き合わせる (task 927 の本丸)。
    #[test]
    fn f64_parity_matrix_covers_every_supported_type() {
        let mut seen: Vec<GgmlType> = Vec::new();
        let mut blocks = 0usize;
        let mut cmp = 0usize;
        for ty in parity_types() {
            if is_raw(ty) {
                continue; // 生 float は raw_float_types_decode_exactly が見る
            }
            for case in &EDGE_CASES {
                let blk = build_block(ty, case);
                cmp += check_block_parity(ty, &blk, &format!("{} / {}", ty.name(), case.name));
                blocks += 1;
            }
            seen.push(ty);
        }
        let want: Vec<GgmlType> = parity_types()
            .into_iter()
            .filter(|t| !is_raw(*t))
            .collect();
        assert_eq!(seen, want, "行列が対象型を全部は通っていない");
        assert_eq!(blocks, want.len() * EDGE_CASES.len(), "ブロック数");
        let want_elems: usize = want.iter().map(|t| t.blck_size() * EDGE_CASES.len()).sum();
        assert_eq!(cmp, want_elems, "比較した要素数");
        println!(
            "f64 parity: {} 型 / {} ブロック / {} 要素",
            seen.len(),
            blocks,
            cmp
        );
    }

    /// 全ゼロブロックは全型で 0 になる (スケールもデータも 0)。
    #[test]
    fn all_zero_block_is_zero_for_every_type() {
        for ty in parity_types() {
            if is_raw(ty) {
                continue;
            }
            let blk = build_block(ty, &EDGE_CASES[0]);
            let out = dequant_row(ty, &blk, ty.blck_size());
            assert!(
                out.iter().all(|v| *v == 0.0),
                "{}: 全ゼロブロックが 0 にならない",
                ty.name()
            );
        }
    }

    /// 負のゼロのスケールで、実際に -0.0 が出ていること (行列テストが
    /// 「+0.0 同士の比較」に退化していないことの確認)。
    #[test]
    fn neg_zero_scale_yields_signed_zeros() {
        let case = EDGE_CASES
            .iter()
            .find(|c| c.name == "neg_zero_scale")
            .expect("neg_zero_scale ケース");
        let mut neg = 0usize;
        let mut total = 0usize;
        for ty in parity_types() {
            if is_raw(ty) {
                continue;
            }
            let blk = build_block(ty, case);
            check_block_parity(ty, &blk, &format!("{} / neg_zero", ty.name()));
            for v in dequant_row(ty, &blk, ty.blck_size()) {
                total += 1;
                if v == 0.0 && v.is_sign_negative() {
                    neg += 1;
                }
            }
        }
        assert!(neg > 0, "-0.0 が 1 つも出ていない (テストが退化している)");
        println!("neg_zero: {neg} / {total} 要素が -0.0");
    }

    /// 生 float 型はバイトをそのまま f32 へ写す (丸めも変換もしない)。
    #[test]
    fn raw_float_types_decode_exactly() {
        for bits in [
            0x0000_0000u32,
            0x8000_0000,
            0x0000_0001, // 最小非正規
            0x7f7f_ffff, // 最大有限
            0x3f00_0000,
            0xbf80_0000,
            0x4050_0000,
        ] {
            let want = f32::from_bits(bits);
            let raw = bits.to_le_bytes();
            assert_eq!(dequant_row(T::F32, &raw, 1)[0].to_bits(), want.to_bits());
            assert_eq!(dequant_ref(T::F32, &raw)[0], want as f64);
            // 参照実装も一致 (f64 へ広げるだけ)。
            check_block_parity(T::F32, &raw, "F32");
        }
        // F16: 0 (両符号) / 最小非正規 / 最大有限 / 0.5 / -1 / 3.25
        for bits in [0x0000u16, 0x8000, 0x0001, 0x7bff, 0x3800, 0xbc00, 0x4280] {
            let want = half::f16::from_bits(bits).to_f32();
            let raw = bits.to_le_bytes();
            assert_eq!(
                dequant_row(T::F16, &raw, 1)[0].to_bits(),
                want.to_bits(),
                "F16 {bits:#06x}"
            );
            check_block_parity(T::F16, &raw, "F16");
        }
        // BF16: f32 の上位 16bit
        for bits in [0x0000u16, 0x8000, 0x0001, 0x7f7f, 0x3f00, 0xbf80, 0x4050] {
            let want = f32::from_bits((bits as u32) << 16);
            let raw = bits.to_le_bytes();
            assert_eq!(
                dequant_row(T::Bf16, &raw, 1)[0].to_bits(),
                want.to_bits(),
                "BF16 {bits:#06x}"
            );
            check_block_parity(T::Bf16, &raw, "BF16");
        }
    }

    /// 行 API と 1 ブロック API がビット一致し、2 回目も同じ値になる。
    #[test]
    fn row_equals_blockwise_and_is_deterministic() {
        for ty in parity_types() {
            for case in &EDGE_CASES {
                let blk = build_block(ty, case);
                let bs = ty.blck_size();
                let row = dequant_row(ty, &blk, bs);
                let mut blkwise = vec![0f32; bs];
                let mut tmp = vec![0f32; bs];
                let ts = ty.type_size();
                for (i, ch) in blk.chunks_exact(ts).enumerate() {
                    dequant_block_into(ty, ch, &mut tmp);
                    blkwise[i * bs..(i + 1) * bs].copy_from_slice(&tmp);
                }
                for (i, (a, b)) in row.iter().zip(blkwise.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{} / {}[{i}]: 行とブロックで不一致",
                        ty.name(),
                        case.name
                    );
                }
                // 2 回目も同じビット列 (共有バッファ・乱数を使っていない)。
                let again = dequant_row(ty, &blk, bs);
                for (i, (a, b)) in row.iter().zip(again.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{} / {}[{i}]: 非決定的",
                        ty.name(),
                        case.name
                    );
                }
            }
        }
    }

    /// MXFP4 は `dequant_supported` が true を返すのに `dequant_block` が
    /// `unreachable!()` で落ちる (UD-GGUF では未使用)。既知の穴として固定する。
    /// 実装するときはこのテストを消し、パリティ行列に加える。
    #[test]
    #[should_panic(expected = "MXFP4")]
    fn mxfp4_dequant_is_a_known_gap() {
        let _ = dequant_row(T::Mxfp4, &[0u8; 18], 32);
    }
}

