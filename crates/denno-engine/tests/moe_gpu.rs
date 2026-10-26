//! P-MOE-E1: Flash-Next の expert GEMV カーネル (`kernels_moe.cu`) と
//! CPU 参照 (`model::moe_expert::expert_gemv`) の**ビット一致**を確かめる。
//!
//! GPU が無い環境 (CI・cuda feature 無し) ではテスト自体が存在しない。
//! `DENNO_CUDA=1` が無いときは静かに skip する。
//!
//! 重みの生バイトは決定的な疑似乱数で作る。f16 のスケール欄だけは有限値に
//! 固定する (乱数のままだと NaN になり、NaN の伝播は CPU と GPU で
//! ペイロードが揃う保証が無いため、比較したいのは算術であって NaN ではない)。

#[cfg(feature = "cuda")]
mod gpu {
    use denno_engine::cuda::CudaCtx;
    use denno_engine::kquant::GgmlType;
    use denno_engine::model::moe_expert::{expert_gemv, row_bytes};

    /// f16 の 0.1 と 0.05 (有限値)。ブロック先頭の d / dmin に置く。
    const F16_0_1: [u8; 2] = [0x66, 0x2E];
    const F16_0_05: [u8; 2] = [0xCD, 0x2C];

    /// (1 ブロックのバイト数, 先頭で f16 を潰すバイト数)
    fn block_geo(ty: GgmlType) -> (usize, usize) {
        match ty {
            GgmlType::Q8_0 => (34, 2),
            GgmlType::Q5_1 => (24, 4),
            GgmlType::Q4K => (144, 4),
            GgmlType::Q5K => (176, 4),
            _ => unreachable!(),
        }
    }

    fn check(ty: GgmlType) {
        if std::env::var("DENNO_CUDA").as_deref() != Ok("1") {
            eprintln!("skip: DENNO_CUDA=1 未設定");
            return;
        }
        let n_in = 2560usize; // 32 と 256 の公倍数
        let n_out = 64usize;
        let rb = row_bytes(ty, n_in).expect("row_bytes");

        // xorshift64
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        let mut w: Vec<u8> = (0..n_out * rb).map(|_| (next() >> 24) as u8).collect();
        let (ts_blk, hdr) = block_geo(ty);
        assert_eq!(ts_blk, ty.type_size(), "ブロックのバイト数が kquant と違う");
        assert_eq!(rb % ts_blk, 0);
        for row in 0..n_out {
            for blk in 0..(rb / ts_blk) {
                // バイト位置は「ブロックのバイト数」で進める (要素数ではない)。
                let o = row * rb + blk * ts_blk;
                if hdr == 2 {
                    w[o..o + 2].copy_from_slice(&F16_0_1);
                } else {
                    w[o..o + 2].copy_from_slice(&F16_0_1);
                    w[o + 2..o + 4].copy_from_slice(&F16_0_05);
                }
            }
        }

        let x: Vec<f32> = (0..n_in)
            .map(|_| ((next() >> 40) as f32) / 16_777_216.0 - 0.5)
            .collect();

        let mut want = vec![0f32; n_out];
        expert_gemv(ty, &w, n_in, n_out, &x, &mut want).expect("CPU 参照");

        let ctx = CudaCtx::new().expect("CudaCtx::new");
        let mut got = vec![0f32; n_out];
        let used = ctx
            .moe_gemv(ty, &w, &x, n_in, n_out, &mut got)
            .expect("moe_gemv");
        assert!(used, "{ty:?}: GPU 経路が使われなかった (PTX 未ロード)");

        for r in 0..n_out {
            assert_eq!(
                got[r].to_bits(),
                want[r].to_bits(),
                "{ty:?}: 行 {r} で不一致 (gpu={} cpu={})",
                got[r],
                want[r]
            );
        }
        eprintln!("{ty:?}: {n_out} 行すべてビット一致");
    }

    #[test]
    #[ignore]
    fn q8_0_expert_gemv_bits_equal_cpu() {
        check(GgmlType::Q8_0);
    }

    #[test]
    #[ignore]
    fn q5_1_expert_gemv_bits_equal_cpu() {
        check(GgmlType::Q5_1);
    }

    #[test]
    #[ignore]
    fn q4_k_expert_gemv_bits_equal_cpu() {
        check(GgmlType::Q4K);
    }

    #[test]
    #[ignore]
    fn q5_k_expert_gemv_bits_equal_cpu() {
        check(GgmlType::Q5K);
    }
}
