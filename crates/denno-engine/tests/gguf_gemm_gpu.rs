//! P-PF-GB: prefill 用の量子 GEMM (`cuda::maybe_gpu_gguf_gemm_into`) の検算。
//!
//! prefill は 2026-10-28 のオペレーター指示でビット一致契約の外にある
//! (FMA 融合 + 加算順が違う)。ここでは **相対誤差** で CPU 参照と突き合わせる。
//!
//! GPU の無い環境では静かに skip。`DENNO_CUDA=1` が要る。

#[cfg(feature = "cuda")]
mod gpu {
    use denno_engine::kquant::GgmlType;
    use denno_engine::model::linear_gguf::GgufLinear;

    const F16_0_1: [u8; 2] = [0x66, 0x2E];
    const F16_0_05: [u8; 2] = [0xCD, 0x2C];

    fn block_geo(ty: GgmlType) -> (usize, usize) {
        match ty {
            GgmlType::Q8_0 => (34, 2),
            GgmlType::Q5_1 => (24, 4),
            GgmlType::Q4K => (144, 4),
            GgmlType::Q5K => (176, 4),
            _ => unreachable!(),
        }
    }

    fn make_matrix(ty: GgmlType, k: usize, n: usize, seed: u64) -> Vec<u8> {
        let bs = ty.blck_size();
        let ts = ty.type_size();
        assert_eq!(k % bs, 0);
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
                let o = r * rb + b * ts_blk;
                w[o..o + 2].copy_from_slice(&F16_0_1);
                if hdr == 4 {
                    w[o + 2..o + 4].copy_from_slice(&F16_0_05);
                }
            }
        }
        w
    }

    fn rand_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (((s >> 40) as f32) / 16_777_216.0 - 0.5) * scale
            })
            .collect()
    }

    fn check(ty: GgmlType, nt: usize, k: usize, n: usize) {
        if std::env::var("DENNO_CUDA").as_deref() != Ok("1") {
            eprintln!("skip: DENNO_CUDA=1 未設定");
            return;
        }
        let w = make_matrix(ty, k, n, 0x9e37_79b9_7f4a_7c15);
        let xs = rand_vec(nt * k, 0x1234_5678_9abc_def1, 1.0);

        // CPU 参照: 1 トークンずつ expert_gemv (= 逐次の定義)
        let mut want = vec![0f32; nt * n];
        for t in 0..nt {
            denno_engine::model::moe_expert::expert_gemv(
                ty,
                &w,
                k,
                n,
                &xs[t * k..(t + 1) * k],
                &mut want[t * n..(t + 1) * n],
            )
            .expect("CPU 参照");
        }

        let lin = GgufLinear::from_slice(ty, k, n, &w).expect("GgufLinear");
        for pass in 0..2 {
            let mut got = vec![0f32; nt * n];
            let used = denno_engine::cuda::maybe_gpu_gguf_gemm_into(
                lin.uid(),
                ty,
                lin.blocks().as_slice(),
                k,
                n,
                &xs,
                nt,
                &mut got,
            );
            assert!(used, "{ty:?} nt={nt}: GPU GEMM が使われなかった (pass {pass})");
            assert!(
                got.iter().any(|v| *v != 0.0),
                "{ty:?}: 出力が全 0 (経路の生死が分からない)"
            );
            let mut max_err = 0f32;
            let mut max_abs = 0f32;
            for i in 0..nt * n {
                max_err = max_err.max((got[i] - want[i]).abs());
                max_abs = max_abs.max(want[i].abs());
            }
            let rel = max_err / max_abs.max(1e-9);
            eprintln!("{ty:?} nt={nt} k={k} n={n} pass={pass}: 最大絶対誤差 {max_err:.3e} / 最大値 {max_abs:.3e} (相対 {rel:.3e})");
            assert!(
                rel < 2e-4,
                "{ty:?} nt={nt} pass={pass}: 相対誤差 {rel:.3e} が大きすぎる (FMA 融合の差の範囲を超えている)"
            );
        }
    }

    #[test]
    fn gemm_q8_0_matches_cpu_close() {
        check(GgmlType::Q8_0, 64, 2560, 64);
    }

    #[test]
    fn gemm_q5_1_matches_cpu_close() {
        check(GgmlType::Q5_1, 64, 2560, 64);
    }

    #[test]
    fn gemm_q4k_matches_cpu_close() {
        check(GgmlType::Q4K, 64, 2560, 64);
    }

    #[test]
    fn gemm_q5k_matches_cpu_close() {
        check(GgmlType::Q5K, 64, 2560, 64);
    }

    /// nt が 16 の倍数でない端数 (ngt=1 で lane が余る) も見る。
    #[test]
    fn gemm_odd_token_count() {
        check(GgmlType::Q8_0, 5, 2560, 64);
        check(GgmlType::Q4K, 37, 2560, 64);
    }
}
