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
                let x0 = (((x[8 + j] & 0x0f) as i32) | xh0);
                let x1 = (((x[8 + j] >> 4) as i32) | xh1);
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
