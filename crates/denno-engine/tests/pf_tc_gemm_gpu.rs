//! task 169: prefill の BF16 テンソルコア GEMM (`kernels_gemm_tc.cu`) の検算。
//!
//! `kernels_gemm.cu` の既存カーネル (16 行 × 128 トークン・f32 FMA) と違い、
//! こちらは x と重みを BF16 に丸めて mma するので、CPU の逐次 GEMV とは
//! **丸めの差**が出る (演算の意味は同じ)。よって判定は相対誤差にする:
//! BF16 の仮数は 8 bit なので、K=2560 の内積で 2e-3 程度、余裕を見て 2e-2 未満。
//!
//! 実行条件 (GPU ホスト):
//!   DENNO_CUDA=1 DENNO_PF_TC_GEMM=1 DENNO_PF_TC_MIN_M=1 \
//!   DENNO_PF_TC_PTX=/path/to/kernels_gemm_tc.ptx \
//!   cargo test --release -p denno-engine --features cuda --test pf_tc_gemm_gpu -- --test-threads=1
//!
//! `DENNO_PF_TC_MIN_M=1` は M=1 / 17 のケースを TC カーネルに通すため
//! (既定 64 だと従来カーネルに落ちる)。このとき `used == true` は
//! 「TC カーネルが動いた」ことを意味する (PTX が無ければ false で落ちる)。
//!
//! GPU の無い環境では静かに skip する。

#[cfg(feature = "cuda")]
mod gpu {
    use denno_engine::kquant::GgmlType;
    use denno_engine::model::linear_gguf::GgufLinear;

    /// 1.0 の f16 (スケール)。量子値は ±127 の一様。
    const F16_0_1: [u8; 2] = [0x66, 0x2E];

    fn skip(reason: &str) -> bool {
        eprintln!("skip: {reason}");
        true
    }

    fn env_is(name: &str, want: &str) -> bool {
        std::env::var(name).as_deref() == Ok(want)
    }

    /// Q8_0 の合成行列 (34 B / 32 要素)。行ごとに量子値をずらす
    /// (全行同一だと行の取り違えが検出できない)。
    fn make_q8_0(k: usize, n: usize, seed: u64) -> Vec<u8> {
        assert_eq!(k % 32, 0);
        let nb = k / 32;
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut w = vec![0u8; n * nb * 34];
        for r in 0..n {
            for b in 0..nb {
                let o = (r * nb + b) * 34;
                w[o..o + 2].copy_from_slice(&F16_0_1);
                for j in 0..32 {
                    // 行とブロックで値を変える (0 を避ける)。
                    let v = ((next() >> 24) as i32 % 127) as i8;
                    w[o + 2 + j] = v as u8;
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

    /// 1 つの (M, K, N) を TC 経路で回し、選んだ行だけ CPU 逐次 GEMV と比べる。
    fn check(nt: usize, k: usize, n: usize) {
        let w = make_q8_0(k, n, 0x9e37_79b9_7f4a_7c15);
        let xs = rand_vec(nt * k, 0x1234_5678_9abc_def1, 1.0);
        let lin = GgufLinear::from_slice(GgmlType::Q8_0, k, n, &w).expect("GgufLinear");
        let mut got = vec![0f32; nt * n];
        let used = denno_engine::cuda::maybe_gpu_gguf_gemm_into(
            lin.uid(),
            GgmlType::Q8_0,
            lin.blocks().as_slice(),
            k,
            n,
            &xs,
            nt,
            &mut got,
        );
        assert!(
            used,
            "nt={nt} k={k} n={n}: GPU GEMM が使われなかった \
             (DENNO_PF_TC_GEMM=1 / DENNO_PF_TC_MIN_M=1 / PTX の存在を確認)"
        );
        assert!(
            got.iter().any(|v| *v != 0.0),
            "nt={nt} k={k} n={n}: 出力が全 0 (経路の生死が分からない)"
        );

        // 先頭・中央・末尾 (末尾は 128 の端数タイル) の 3 行を CPU で引き直す。
        let mut rows = vec![0usize, nt / 2, nt - 1];
        rows.sort_unstable();
        rows.dedup();
        let mut worst = 0f32;
        for r in rows {
            let mut want = vec![0f32; n];
            denno_engine::model::moe_expert::expert_gemv(
                GgmlType::Q8_0,
                &w,
                k,
                n,
                &xs[r * k..(r + 1) * k],
                &mut want,
            )
            .expect("CPU 参照");
            let mut max_err = 0f32;
            let mut max_abs = 0f32;
            for i in 0..n {
                max_err = max_err.max((got[r * n + i] - want[i]).abs());
                max_abs = max_abs.max(want[i].abs());
            }
            let rel = max_err / max_abs.max(1e-9);
            worst = worst.max(rel);
            eprintln!(
                "nt={nt} k={k} n={n} row={r}: 最大絶対誤差 {max_err:.3e} / 最大値 {max_abs:.3e} (相対 {rel:.3e})"
            );
        }
        assert!(
            worst < 2e-2,
            "nt={nt} k={k} n={n}: BF16 丸めの範囲を超える相対誤差 {worst:.3e}"
        );
    }

    fn ready(test: &str) -> bool {
        if !env_is("DENNO_CUDA", "1") {
            return skip("DENNO_CUDA=1 未設定");
        }
        if !env_is("DENNO_PF_TC_GEMM", "1") {
            return skip("DENNO_PF_TC_GEMM=1 未設定");
        }
        if !env_is("DENNO_PF_TC_MIN_M", "1") {
            return skip("DENNO_PF_TC_MIN_M=1 未設定 (M=1/17 を TC に通すため)");
        }
        eprintln!("--- {test}");
        false
    }

    /// ssm の qkv+gate (K=2560, N=16384) を M の全ケースで。
    #[test]
    fn tc_ssm_qkv_all_m() {
        if ready("tc_ssm_qkv_all_m") {
            return;
        }
        for nt in [1usize, 17, 512, 1024, 2048] {
            check(nt, 2560, 16384);
        }
    }

    /// full attention の q|k|v (K=2560, N=13312)。
    #[test]
    fn tc_full_qkv() {
        if ready("tc_full_qkv") {
            return;
        }
        for nt in [1usize, 17, 512, 1024, 2048] {
            check(nt, 2560, 13312);
        }
    }

    /// out_proj (K=6144, N=2560) と shared expert の down (K=640, N=2560)。
    #[test]
    fn tc_out_and_shared() {
        if ready("tc_out_and_shared") {
            return;
        }
        for nt in [1usize, 17, 512, 1024, 2048] {
            check(nt, 6144, 2560);
        }
        check(512, 640, 2560);
        check(17, 640, 2560);
    }
}
