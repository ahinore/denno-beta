//! task 858: 量子型 (GGML quant) の GPU 経路カバレッジを 1 か所に集めた表。
//!
//! # なぜ要るか
//!
//! GPU に載せられる型は段 (stage) ごとに違う。decode の dense GEMV / prefill の
//! dense GEMM は **旧 4 型** (Q8_0 / Q5_1 / Q4_K / Q5_K) だけ、expert の prefill は
//! さらに Q6_K と IQ 系、expert の decode は IQ 系が別カーネル、というように
//! 表が 4 つに散っている。表が散っていると、型を足した / 消したときに
//! 「GPU を試みるが必ず落ちて CPU へ戻る」あるいは逆に「静かに CPU のまま」が
//! 起きても、テストが緑のまま通る。
//!
//! ここでは各段の対応型を **定数として 1 か所に書き**、テストで実際の dispatch
//! 関数 (`cuda::gguf_row_bytes` / `cuda::moe_pf::moe_batch_slot` /
//! `model::exp_pf::slot` / `model::exp_iq::slot`) と突き合わせる。dispatch を
//! 変えて表を更新し忘れればテストが落ちる。
//!
//! # 数値の契約
//!
//! - CPU の参照は `kquant::dequant_row_into` (全 25 型)。
//! - GPU カーネルのミラーは `crate::cuda::tests` と `model::exp_iq` にあり、
//!   f64 真値に対して「絶対 1e-4 + 相対 1e-5」で検証する (AGENTS.md の方針)。
//!   prefill だけは契約が緩い (FMA 融合 + 加算順が違う。既存 doc を参照)。
//!
//! # 実測と表の関係
//!
//! この表は「どの型が GPU カーネルを持つか」だけを持つ。速い / 遅いの判断は
//! lab 実測の担当で、ここには数字を置かない。

use crate::kquant::GgmlType;
use GgmlType as T;

/// 密行列 (dense linear) の GPU 経路が読める型。
///
/// decode の GEMV (`cuda::maybe_gpu_gguf_gemv_into` → `cuda::gguf_row_bytes`) と
/// prefill の GEMM (`cuda::maybe_gpu_gguf_gemm_into` → `CudaCtx::moe_gemm_dev`) は
/// **同じ 4 型**。ここが唯一の定義で、両方の入口がこの表を通る。
pub const DENSE_GPU: &[GgmlType] = &[T::Q8_0, T::Q5_1, T::Q4K, T::Q5K];

/// expert の prefill (m 行カーネル) が読める K 系の型。
///
/// `kernels_exp_k_pf.cu` / `kernels_exp_k_tc.cu` のスロット (model::exp_pf::K_SLOT0..)。
pub const PREFILL_EXPERT_K: &[GgmlType] = &[T::Q4K, T::Q5K, T::Q6K, T::Q5_1, T::Q8_0];

/// expert の decode / prefill が共通で使う IQ 系の型 (model::exp_iq::slot)。
pub const EXPERT_IQ: &[GgmlType] = &[T::IQ2S, T::IQ3Xxs, T::IQ3S, T::IQ4Xs, T::IQ4Nl, T::Q2_0];

/// `GgmlType` の全 25 型 (テストと文書の突き合わせ用)。
pub const ALL_TYPES: &[GgmlType] = &[
    T::F32,
    T::F16,
    T::Q4_0,
    T::Q4_1,
    T::Q5_0,
    T::Q5_1,
    T::Q8_0,
    T::Q2K,
    T::Q3K,
    T::Q4K,
    T::Q5K,
    T::Q6K,
    T::IQ2Xxs,
    T::IQ2Xs,
    T::IQ3Xxs,
    T::IQ1S,
    T::IQ4Nl,
    T::IQ3S,
    T::IQ2S,
    T::IQ4Xs,
    T::Iq1M,
    T::Bf16,
    T::Tq1_0,
    T::Mxfp4,
    T::Q2_0,
];

/// 密行列を GPU で回せる型か (decode GEMV と prefill GEMM の共通の門番)。
pub fn dense_gpu_supported(ty: GgmlType) -> bool {
    DENSE_GPU.contains(&ty)
}

/// expert の prefill を GPU で回せる型か (K 系 m 行カーネル + IQ 系)。
pub fn prefill_expert_gpu_supported(ty: GgmlType) -> bool {
    PREFILL_EXPERT_K.contains(&ty) || EXPERT_IQ.contains(&ty)
}

/// expert の IQ 系カーネル (`kernels_exp_iq*.cu`) を持つ型か。
pub fn expert_iq_supported(ty: GgmlType) -> bool {
    EXPERT_IQ.contains(&ty)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 表に重複が無く、全要素が [`ALL_TYPES`] に含まれる。
    #[test]
    fn tables_are_well_formed() {
        for tab in [DENSE_GPU, PREFILL_EXPERT_K, EXPERT_IQ] {
            let mut seen = vec![];
            for ty in tab {
                assert!(ALL_TYPES.contains(ty), "{ty:?} が ALL_TYPES に無い");
                assert!(!seen.contains(ty), "{ty:?} が表に重複");
                seen.push(*ty);
            }
        }
        // ALL_TYPES 自体も重複なし。
        let mut seen = vec![];
        for ty in ALL_TYPES {
            assert!(!seen.contains(ty), "{ty:?} が ALL_TYPES に重複");
            seen.push(*ty);
        }
        // 型 id の往復 (`from_id` と対) が全 25 型で成立する。
        for ty in ALL_TYPES {
            assert_eq!(GgmlType::from_id(ty.id()), Some(*ty), "{ty:?} の id 往復");
            assert!(ty.blck_size() > 0 && ty.type_size() > 0, "{ty:?} のブロック");
        }
    }

    /// CPU のデコードは全 25 型を扱える (GPU 表の全要素に参照がある)。
    #[test]
    fn every_type_has_cpu_reference_and_gpu_types_are_subset() {
        for ty in ALL_TYPES {
            assert!(
                crate::kquant::dequant_supported(*ty),
                "{ty:?} に CPU の参照 (dequant_row_into) が無い"
            );
        }
        for ty in DENSE_GPU.iter().chain(PREFILL_EXPERT_K).chain(EXPERT_IQ) {
            assert!(crate::kquant::dequant_supported(*ty), "{ty:?}");
        }
    }

    /// 表の型はすべて 1 行を実際にデコードできる (k はブロック長の倍数)。
    ///
    /// 「型だけ表にあってバイト列が読めない」を落とす。値は有限であることまで見る。
    #[test]
    fn gpu_table_types_dequantize_to_finite_values() {
        for ty in DENSE_GPU.iter().chain(PREFILL_EXPERT_K).chain(EXPERT_IQ) {
            let bs = ty.blck_size();
            let n = bs * 2;
            let ts = ty.type_size();
            let mut bytes = vec![0u8; (n / bs) * ts];
            // d は f16 の 1.0 (NaN / Inf を作らない)。他は決定的な模様。
            bytes[0] = 0x00;
            bytes[1] = 0x3c;
            let mut s = 12345u32;
            for b in bytes.iter_mut().skip(2) {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *b = (s >> 16) as u8;
            }
            let mut out = vec![0f32; n];
            crate::kquant::dequant_row_into(*ty, &bytes, &mut out);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{ty:?} のデコードが NaN / Inf を作った"
            );
        }
    }

    /// decode の GEMV と prefill の GEMM の門番が表と一致する。
    ///
    /// `gguf_row_bytes` は decode GEMV / prefill GEMM / expert の VRAM 常駐が
    /// **共通で** 使う唯一の型判定。ここが表とずれたら、GPU を試みて必ず落ちる
    /// (= 静かに CPU へ戻る) か、逆に CPU でしか回せない型を GPU に載せようとする。
    #[cfg(feature = "cuda")]
    #[test]
    fn dense_gpu_table_matches_cuda_dispatch() {
        for ty in ALL_TYPES {
            let k = 512usize;
            let row = crate::cuda::gguf_row_bytes(*ty, k);
            assert_eq!(
                row.is_some(),
                dense_gpu_supported(*ty),
                "{ty:?}: gguf_row_bytes と DENSE_GPU がずれた (row={row:?})"
            );
            // expert のバッチ (m 行) カーネルも同じ 4 型。
            assert_eq!(
                crate::cuda::moe_pf::moe_batch_slot(*ty).is_some(),
                dense_gpu_supported(*ty),
                "{ty:?}: moe_batch_slot と DENSE_GPU がずれた"
            );
        }
        // 行バイト数が型のブロック定義と一致する (4 型それぞれ)。
        for ty in DENSE_GPU {
            let k = 512usize;
            let rb = crate::cuda::gguf_row_bytes(*ty, k).expect("4 型は Some");
            assert_eq!(rb, (k / ty.blck_size()) * ty.type_size(), "{ty:?} の行バイト数");
        }
    }

    /// expert の prefill の門番が表と一致する。
    #[test]
    fn prefill_expert_table_matches_dispatch() {
        for ty in ALL_TYPES {
            assert_eq!(
                crate::model::exp_pf::slot(*ty).is_some(),
                prefill_expert_gpu_supported(*ty),
                "{ty:?}: exp_pf::slot と表がずれた"
            );
            assert_eq!(
                crate::model::exp_pf::supported(*ty),
                prefill_expert_gpu_supported(*ty),
                "{ty:?}: exp_pf::supported と表がずれた"
            );
        }
    }

    /// expert の IQ 系 (decode / prefill 共通) の門番が表と一致する。
    #[test]
    fn expert_iq_table_matches_dispatch() {
        for ty in ALL_TYPES {
            assert_eq!(
                crate::model::exp_iq::slot(*ty).is_some(),
                expert_iq_supported(*ty),
                "{ty:?}: exp_iq::slot と表がずれた"
            );
        }
    }

    /// 監査で確定した「GPU が無い型」を明示的に固定する。
    ///
    /// これらは CPU のスカラー経路 (`kquant::dequant_row_into` +
    /// `moe_expert::expert_gemv_*`) で計算される。**既定の挙動**であり、
    /// 遅い経路への「想定外の」フォールバックではない (task 858 の監査結果)。
    /// 型を GPU 対応にするときは、ここを直すのではなく表とカーネルを足す。
    #[test]
    fn known_cpu_only_types_are_pinned() {
        for ty in [
            T::Q4_0,
            T::Q4_1,
            T::Q5_0,
            T::Q2K,
            T::Q3K,
            T::IQ2Xxs,
            T::IQ2Xs,
            T::IQ1S,
            T::Iq1M,
            T::Tq1_0,
            T::Mxfp4,
            T::F32,
            T::F16,
            T::Bf16,
        ] {
            assert!(!dense_gpu_supported(ty), "{ty:?} が dense GPU 表に入った");
            assert!(!expert_iq_supported(ty), "{ty:?} が expert IQ 表に入った");
        }
        // Q6_K は dense では CPU、expert の prefill では GPU (非対称)。
        assert!(!dense_gpu_supported(T::Q6K), "Q6_K の dense GPU 経路は無い");
        assert!(
            prefill_expert_gpu_supported(T::Q6K),
            "Q6_K の expert prefill カーネルはある"
        );
        // IQ 系は expert 専用 (dense の GPU 経路は無い)。
        for ty in EXPERT_IQ {
            assert!(!dense_gpu_supported(*ty), "{ty:?} が dense GPU 表に入った");
        }
    }
}
