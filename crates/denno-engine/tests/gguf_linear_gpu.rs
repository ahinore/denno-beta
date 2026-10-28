//! P-MOE-E5: `GgufLinear` の GPU 経路 (`cuda::maybe_gpu_gguf_gemv_into`) と
//! CPU 参照の**ビット一致**を確かめる。
//!
//! 重みは 1 度だけ VRAM へ上がり、2 回目以降は GEMV だけが走る (常駐レジストリ)。
//! ここでは「1 回目 = 転送あり」「2 回目 = 転送なし」の両方で同じ値が出ることを見る。
//!
//! GPU の無い環境ではテスト自体が存在しない。`DENNO_CUDA=1` が無ければ静かに skip。

#[cfg(feature = "cuda")]
mod gpu {
    use denno_engine::kquant::GgmlType;
    use denno_engine::model::linear_gguf::GgufLinear;

    const F16_0_1: [u8; 2] = [0x66, 0x2E];
    const F16_0_05: [u8; 2] = [0xCD, 0x2C];

    /// (1 ブロックのバイト数, 先頭で f16 を潰すバイト数)。
    fn block_geo(ty: GgmlType) -> (usize, usize) {
        match ty {
            GgmlType::Q8_0 => (34, 2),
            GgmlType::Q5_1 => (24, 4),
            GgmlType::Q4K => (144, 4),
            GgmlType::Q5K => (176, 4),
            _ => unreachable!(),
        }
    }

    /// 決定的な疑似乱数で行列を組み立て、f16 スケール欄だけ有限値に固定する。
    fn make_matrix(ty: GgmlType, k: usize, n: usize, seed: u64) -> Vec<u8> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert_eq!(k % bs, 0);
        assert_eq!(ts, block_geo(ty).0, "ブロックのバイト数が kquant と違う");
        let nb = k / bs;
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut w: Vec<u8> = (0..n * nb * ts).map(|_| (next() >> 24) as u8).collect();
        let (ts_blk, hdr) = block_geo(ty);
        let rb = nb * ts;
        for r in 0..n {
            for b in 0..nb {
                // バイト位置は「ブロックのバイト数」で進める (要素数ではない)。
                let o = r * rb + b * ts_blk;
                w[o..o + 2].copy_from_slice(&F16_0_1);
                if hdr == 4 {
                    w[o + 2..o + 4].copy_from_slice(&F16_0_05);
                }
            }
        }
        w
    }

    fn check(ty: GgmlType) {
        if std::env::var("DENNO_CUDA").as_deref() != Ok("1") {
            eprintln!("skip: DENNO_CUDA=1 未設定");
            return;
        }
        let k = 2560usize;
        let n = 64usize;
        let w = make_matrix(ty, k, n, 0x9e37_79b9_7f4a_7c15);

        let mut s: u64 = 0x1234_5678_9abc_def1;
        let x: Vec<f32> = (0..k)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32) / 16_777_216.0 - 0.5
            })
            .collect();

        // CPU 参照 (GPU 無効の GgufLinear をそのまま使う)
        let lin = GgufLinear::from_slice(ty, k, n, &w).expect("GgufLinear");
        let mut want = vec![0f32; n];
        denno_engine::model::moe_expert::expert_gemv(ty, &w, k, n, &x, &mut want).expect("CPU 参照");

        // GPU 経路 (1 回目 = 転送あり / 2 回目 = 常駐)
        for pass in 0..2 {
            let mut got = vec![0f32; n];
            let used = denno_engine::cuda::maybe_gpu_gguf_gemv_into(
                lin.uid(),
                ty,
                lin.blocks().as_slice(),
                k,
                n,
                &x,
                &mut got,
            );
            assert!(used, "{ty:?}: GPU 経路が使われなかった (pass {pass})");
            for r in 0..n {
                assert_eq!(
                    got[r].to_bits(),
                    want[r].to_bits(),
                    "{ty:?}: pass {pass} 行 {r} で不一致 (gpu={} cpu={})",
                    got[r],
                    want[r]
                );
            }
        }
        eprintln!("{ty:?}: 2 パスとも {n} 行ビット一致");

        // P-pf-moe-scratch: x / out の一時バッファは thread_local の grow-only
        // スクラッチを使い回す。**別の入力の後でも**同じ出力になること
        // (前回の残りを読んでいないこと) を固定する。x2 で 1 回走らせてから
        // x に戻し、上の CPU 参照 `want` とビット一致することを要求する。
        let x2: Vec<f32> = x.iter().map(|v| v * -0.5 + 0.25).collect();
        let mut got2 = vec![0f32; n];
        assert!(
            denno_engine::cuda::maybe_gpu_gguf_gemv_into(
                lin.uid(),
                ty,
                lin.blocks().as_slice(),
                k,
                n,
                &x2,
                &mut got2,
            ),
            "{ty:?}: x2 で GPU 経路が使われなかった"
        );
        let mut again = vec![0f32; n];
        assert!(
            denno_engine::cuda::maybe_gpu_gguf_gemv_into(
                lin.uid(),
                ty,
                lin.blocks().as_slice(),
                k,
                n,
                &x,
                &mut again,
            ),
            "{ty:?}: x に戻したときに GPU 経路が使われなかった"
        );
        for r in 0..n {
            assert_eq!(
                again[r].to_bits(),
                want[r].to_bits(),
                "{ty:?}: 別入力の後で行 {r} が変わった (gpu={} cpu={})",
                again[r],
                want[r]
            );
        }
        eprintln!("{ty:?}: 別入力の後でも {n} 行ビット一致");
    }

    #[test]
    #[ignore]
    fn gguf_linear_q8_0_gpu_matches_cpu() {
        check(GgmlType::Q8_0);
    }

    #[test]
    #[ignore]
    fn gguf_linear_q5_1_gpu_matches_cpu() {
        check(GgmlType::Q5_1);
    }

    #[test]
    #[ignore]
    fn gguf_linear_q4_k_gpu_matches_cpu() {
        check(GgmlType::Q4K);
    }

    #[test]
    #[ignore]
    fn gguf_linear_q5_k_gpu_matches_cpu() {
        check(GgmlType::Q5K);
    }
}
