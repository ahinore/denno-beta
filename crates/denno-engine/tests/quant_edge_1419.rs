//! task 1419: 量子型のブロック寸法・行バイト数・退化スケールの境界テスト (CPU のみ)。
//!
//! `kquant::GgmlType` のブロック定義と、`quant_matrix` の公開表、そして
//! 行バイト数を返す 3 つの公開関数 (`model::moe_expert::row_bytes` /
//! `model::exp_pf::row_bytes` / `model::exp_iq::row_bytes`) の整合を固定する。
//!
//! デコード側では「全ゼロ」「f16 最小非正規」「f16 最大」「NaN」のスケールで
//! panic しないこと、有限スケールでは有限値しか出さないことを確認する。
//!
//! MXFP4 は `dequant_block` が意図的に `unreachable!()` (UD-GGUF では未使用) で
//! 落ちる型なので、寸法と行バイト数の検査だけを行い、デコード検査からは外す。

use denno_engine::kquant::{dequant_row, dequant_supported, GgmlType};
use denno_engine::model::{exp_iq, exp_pf, moe_expert};
use denno_engine::quant_matrix::{ALL_TYPES, DENSE_GPU, EXPERT_IQ, PREFILL_EXPERT_K};

/// 実際に 1 ブロックをデコードできる型 (MXFP4 を除く)。
fn decodable_types() -> Vec<GgmlType> {
    ALL_TYPES
        .iter()
        .copied()
        .filter(|t| *t != GgmlType::Mxfp4)
        .collect()
}

/// ブロック先頭 (またはブロック内の既知位置) に単一の f16 スケールを持つ型の、
/// そのスケールのバイトオフセット。分散配置 (IQ1_M) や生 float 型は `None`。
fn single_f16_scale_offset(ty: GgmlType) -> Option<usize> {
    Some(match ty {
        GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::Q2_0
        | GgmlType::Q4K
        | GgmlType::Q5K
        | GgmlType::IQ2Xxs
        | GgmlType::IQ2Xs
        | GgmlType::IQ2S
        | GgmlType::IQ3Xxs
        | GgmlType::IQ3S
        | GgmlType::IQ1S
        | GgmlType::IQ4Nl
        | GgmlType::IQ4Xs => 0,
        GgmlType::Tq1_0 => 52,
        // block_q2_K: scales[16] + qs[64] + d + dmin
        GgmlType::Q2K => 80,
        // block_q3_K: hmask[32] + qs[64] + scales[12] + d
        GgmlType::Q3K => 108,
        // block_q6_K: ql[128] + qh[64] + scales[16] + d
        GgmlType::Q6K => 208,
        // 生 float と分散スケール型、未対応の MXFP4 は対象外。
        GgmlType::F32
        | GgmlType::F16
        | GgmlType::Bf16
        | GgmlType::Iq1M
        | GgmlType::Mxfp4 => return None,
    })
}

fn write_f16_bits(buf: &mut [u8], off: usize, bits: u16) {
    buf[off..off + 2].copy_from_slice(&bits.to_le_bytes());
}

#[test]
fn all_types_block_dimensions_are_consistent() {
    assert_eq!(ALL_TYPES.len(), 25, "GgmlType は 25 型");

    for ty in ALL_TYPES {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert!(bs > 0, "{ty:?}: blck_size が 0");
        assert!(ts > 0, "{ty:?}: type_size が 0");
        // ブロックは 1 / 32 / 64 / 256 要素のいずれか (ggml の定義)。
        assert!(
            matches!(bs, 1 | 32 | 64 | 256),
            "{ty:?}: 想定外の blck_size {bs}"
        );
        // ブロックは必ず 1 要素あたり 1 バイト以上 (圧縮型は 0.5 未満にならない)。
        assert!(
            ts >= bs.div_ceil(8),
            "{ty:?}: type_size {ts} が blck_size {bs} に対して小さすぎる"
        );
        // id <-> 型の往復。
        assert_eq!(
            GgmlType::from_id(ty.id()),
            Some(*ty),
            "{ty:?}: id 往復が壊れている"
        );
    }

    // 生 float 型の寸法。
    assert_eq!((GgmlType::F32.blck_size(), GgmlType::F32.type_size()), (1, 4));
    assert_eq!((GgmlType::F16.blck_size(), GgmlType::F16.type_size()), (1, 2));
    assert_eq!(
        (GgmlType::Bf16.blck_size(), GgmlType::Bf16.type_size()),
        (1, 2)
    );
    // Q2_0 だけが 64 要素ブロック。
    assert_eq!(GgmlType::Q2_0.blck_size(), 64);
    for ty in ALL_TYPES.iter().filter(|t| **t != GgmlType::Q2_0) {
        if ty.blck_size() != 1 {
            assert!(
                matches!(ty.blck_size(), 32 | 256),
                "{ty:?}: 32/256 以外の圧縮ブロック"
            );
        }
    }
}

#[test]
fn quant_matrix_tables_are_subsets_of_all_types() {
    assert_eq!(ALL_TYPES.len(), 25);
    for table in [DENSE_GPU, PREFILL_EXPERT_K, EXPERT_IQ] {
        let mut seen: Vec<GgmlType> = Vec::new();
        for ty in table {
            assert!(ALL_TYPES.contains(ty), "{ty:?}: ALL_TYPES に無い型が表にある");
            assert!(!seen.contains(ty), "{ty:?}: 表に重複がある");
            assert!(
                dequant_supported(*ty),
                "{ty:?}: CPU のデコード経路を持たない型が GPU 表にある"
            );
            seen.push(*ty);
        }
    }
    // dense の 4 型は expert prefill の K 系表に含まれる (同じカーネル系列)。
    for a in DENSE_GPU {
        assert!(
            PREFILL_EXPERT_K.contains(a),
            "{a:?}: dense 型が prefill K 表に無い"
        );
    }
    // IQ 系は expert 専用なので dense / prefill K とは重ならない。
    for a in EXPERT_IQ {
        assert!(!DENSE_GPU.contains(a), "{a:?}: IQ 型が dense 表にある");
        assert!(
            !PREFILL_EXPERT_K.contains(a),
            "{a:?}: IQ 型が prefill K 表にある"
        );
    }
}

#[test]
fn row_bytes_matches_block_definition_for_all_types() {
    for ty in ALL_TYPES {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        for blocks in 1usize..=4 {
            let k = blocks * bs;
            let want = blocks * ts;
            let moe = moe_expert::row_bytes(*ty, k)
                .unwrap_or_else(|e| panic!("{ty:?}: moe_expert::row_bytes({k}) が失敗: {e}"));
            let pf = exp_pf::row_bytes(*ty, k);
            let iq = exp_iq::row_bytes(*ty, k);
            assert_eq!(moe, want, "{ty:?}: moe_expert 行バイト数");
            assert_eq!(pf, Some(want), "{ty:?}: exp_pf 行バイト数");
            assert_eq!(iq, Some(want), "{ty:?}: exp_iq 行バイト数");
        }
    }
}

#[test]
fn row_bytes_rejects_zero_and_non_aligned_lengths() {
    for ty in ALL_TYPES {
        let bs = ty.blck_size();
        let ts = ty.type_size();

        // 0 は全経路で拒否。
        assert!(moe_expert::row_bytes(*ty, 0).is_err(), "{ty:?}: k=0 を受理");
        assert_eq!(exp_pf::row_bytes(*ty, 0), None, "{ty:?}: k=0 を受理");
        assert_eq!(exp_iq::row_bytes(*ty, 0), None, "{ty:?}: k=0 を受理");

        // 端数は全経路で拒否 (ブロック 1 要素の型には端数が存在しない)。
        if bs > 1 {
            let ragged = bs * 2 + 1;
            assert!(
                moe_expert::row_bytes(*ty, ragged).is_err(),
                "{ty:?}: k={ragged} (端数) を受理"
            );
            assert_eq!(
                exp_pf::row_bytes(*ty, ragged),
                None,
                "{ty:?}: 端数の k={ragged} を受理"
            );
            assert_eq!(
                exp_iq::row_bytes(*ty, ragged),
                None,
                "{ty:?}: 端数の k={ragged} を受理"
            );
            // 1 ブロックに満たない切り詰めも同じく拒否。
            assert!(moe_expert::row_bytes(*ty, bs - 1).is_err(), "{ty:?}");
        } else {
            // bs == 1 の型は k=1 が最小の受理値。
            assert_eq!(moe_expert::row_bytes(*ty, 1).unwrap(), ts);
        }
    }
}

#[test]
fn zero_scale_blocks_decode_to_zero_and_stay_finite() {
    for ty in decodable_types() {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        let block = vec![0u8; ts];
        let out = dequant_row(ty, &block, bs);
        assert_eq!(out.len(), bs, "{ty:?}: 出力長");
        assert!(
            out.iter().all(|v| *v == 0.0),
            "{ty:?}: 全ゼロブロックが 0 にならない"
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "{ty:?}: 全ゼロブロックが非有限値を出した"
        );
    }
}

#[test]
fn subnormal_and_extreme_scales_stay_finite() {
    // f16: 最小非正規 0x0001、最大有限 0x7bff。
    for (bits, label) in [(0x0001u16, "最小非正規"), (0x7bff, "最大有限")] {
        for ty in decodable_types() {
            let Some(off) = single_f16_scale_offset(ty) else {
                continue;
            };
            let bs = ty.blck_size();
            let ts = ty.type_size();
            let mut block = vec![0u8; ts];
            write_f16_bits(&mut block, off, bits);
            let out = dequant_row(ty, &block, bs);
            assert_eq!(out.len(), bs, "{ty:?}: 出力長");
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{ty:?}: スケール {label} (0x{bits:04x}) が非有限値を出した"
            );
        }
    }
}

#[test]
fn nan_scale_does_not_panic() {
    for ty in decodable_types() {
        let Some(off) = single_f16_scale_offset(ty) else {
            continue;
        };
        let bs = ty.blck_size();
        let ts = ty.type_size();
        let mut block = vec![0u8; ts];
        write_f16_bits(&mut block, off, 0x7e00); // f16 NaN
        let out = dequant_row(ty, &block, bs);
        // panic しないことが主眼。少なくとも 1 要素に NaN が伝播する。
        assert_eq!(out.len(), bs, "{ty:?}: 出力長");
        assert!(
            out.iter().any(|v| !v.is_finite()),
            "{ty:?}: NaN スケールが NaN を伝播しなかった"
        );
    }
}
