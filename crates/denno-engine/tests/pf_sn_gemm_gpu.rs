//! task 181: prefill の N が小さい射影用 BF16 テンソルコア GEMM
//! (`kernels_gemm_sn.cu`。BN=64 のタイル) の検算。
//!
//! 比較の基準は CPU の逐次 GEMV (`moe_expert::expert_gemv`)。カーネルは x と
//! 重みを BF16 に丸めて mma する (FP32 アキュムレート) ので、CPU とは**丸めの
//! 差**が出る。BF16 の仮数は 8 bit なので K=2560/6144 の内積で相対 2e-3 程度、
//! 余裕を見て 2e-2 未満を合格にする (tc2 / br169 の検算と同じ基準)。
//!
//! 実行条件 (GPU ホスト):
//!   DENNO_CUDA=1 DENNO_PF_TC_SMALLN=1 DENNO_PF_TC_MIN_M=1 \
//!   DENNO_PF_SN_PTX=/path/to/kernels_gemm_sn.ptx \
//!   cargo test --release -p denno-engine --features cuda --test pf_sn_gemm_gpu -- --test-threads=1
//!
//! `DENNO_PF_TC_MIN_M=1` は M=17 をこのカーネルに通すため (既定 64 だと従来経路)。
//! `used == true` は「小さい N 用カーネルが動いた」ことを意味する (PTX が無ければ
//! 従来カーネルへ落ちて false)。
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

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Q8_0 の合成行列 (34 B / 32 要素)。行ごとに量子値をずらす
    /// (全行同一だと行の取り違えが検出できない)。
    fn make_q8_0(k: usize, n: usize, seed: u64) -> Vec<u8> {
        assert_eq!(k % 32, 0);
        let nb = k / 32;
        let mut rng = Rng(seed | 1);
        let mut w = vec![0u8; n * nb * 34];
        for r in 0..n {
            for b in 0..nb {
                let o = (r * nb + b) * 34;
                w[o..o + 2].copy_from_slice(&F16_0_1);
                for j in 0..32 {
                    let v = ((rng.next() >> 24) as i32 % 127) as i8;
                    w[o + 2 + j] = v as u8;
                }
            }
        }
        w
    }

    /// Q4_K の合成行列 (144 B / 256 要素)。スケール・min を妥当な値にして
    /// 6bit の scale/min と nibble をでたらめに振る。
    fn make_q4k(k: usize, n: usize, seed: u64) -> Vec<u8> {
        assert_eq!(k % 256, 0);
        let blk = 144usize;
        let nb = k / 256;
        let mut rng = Rng(seed | 1);
        let mut w = vec![0u8; n * nb * blk];
        // d = 0.0137 / dmin = 0.0021 の f16。
        let d = [0x66u8, 0x2Eu8];
        let dmin = [0x08u8, 0x1Cu8];
        for r in 0..n {
            for b in 0..nb {
                let o = (r * nb + b) * blk;
                w[o..o + 2].copy_from_slice(&d);
                w[o + 2..o + 4].copy_from_slice(&dmin);
                for i in 4..blk {
                    w[o + i] = (rng.next() >> 32) as u8;
                }
            }
        }
        w
    }

    fn rand_vec(len: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut rng = Rng(seed | 1);
        (0..len)
            .map(|_| (((rng.next() >> 40) as f32) / 16_777_216.0 - 0.5) * scale)
            .collect()
    }

    /// 1 つの (M, K, N) をこの経路で回し、先頭・中央・末尾の行を CPU と比べる。
    fn check(ty: GgmlType, nt: usize, k: usize, n: usize) {
        let w = match ty {
            GgmlType::Q8_0 => make_q8_0(k, n, 0x9e37_79b9_7f4a_7c15),
            GgmlType::Q4K => make_q4k(k, n, 0x2545_f491_4f6c_dd1d),
            _ => panic!("このカーネルは Q8_0 / Q4_K 専用"),
        };
        let xs = rand_vec(nt * k, 0x1234_5678_9abc_def1, 1.0);
        let lin = GgufLinear::from_slice(ty, k, n, &w).expect("GgufLinear");
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
        assert!(
            used,
            "{ty:?} nt={nt} k={k} n={n}: GPU GEMM が使われなかった \
             (DENNO_PF_TC_SMALLN=1 / DENNO_PF_TC_MIN_M=1 / PTX の存在を確認)"
        );
        assert!(
            got.iter().any(|v| *v != 0.0),
            "{ty:?} nt={nt} k={k} n={n}: 出力が全 0 (経路の生死が分からない)"
        );

        let mut rows = vec![0usize, nt / 2, nt - 1];
        rows.sort_unstable();
        rows.dedup();
        let mut worst = 0f32;
        for r in rows {
            let mut want = vec![0f32; n];
            denno_engine::model::moe_expert::expert_gemv(
                ty,
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
                "{ty:?} nt={nt} k={k} n={n} row={r}: 最大絶対誤差 {max_err:.3e} / \
                 最大値 {max_abs:.3e} (相対 {rel:.3e})"
            );
        }
        assert!(
            worst < 2e-2,
            "{ty:?} nt={nt} k={k} n={n}: BF16 丸めの範囲を超える相対誤差 {worst:.3e}"
        );
    }

    fn ready(test: &str) -> bool {
        if !env_is("DENNO_CUDA", "1") {
            return skip("DENNO_CUDA=1 未設定");
        }
        if !env_is("DENNO_PF_TC_SMALLN", "1") {
            return skip("DENNO_PF_TC_SMALLN=1 未設定");
        }
        if !env_is("DENNO_PF_TC_MIN_M", "1") {
            return skip("DENNO_PF_TC_MIN_M=1 未設定 (M=17 を通すため)");
        }
        eprintln!("--- {test}");
        false
    }

    // M=1 は `gemm_try` が `nt <= 1` で必ず従来経路へ落とす (prefill に 1 トークンの
    // GEMM は無い)。M=1 の形の条件と端数タイル (1 行) は cuda.rs の
    // `pf_tc_smalln_eligibility_real_shapes` / `pf_tc_smalln_tile_index_mirror` が押さえる。

    /// shared expert の gate (Q4_K) / up (Q8_0): K=2560, N=640。
    #[test]
    fn sn_shared_gate_up_all_m() {
        if ready("sn_shared_gate_up_all_m") {
            return;
        }
        for nt in [17usize, 512, 1024, 2048] {
            check(GgmlType::Q8_0, nt, 2560, 640);
            check(GgmlType::Q4K, nt, 2560, 640);
        }
    }

    /// shared expert の down: K=640, N=2560 (K が 640 = Q8_0 の 20 ブロック)。
    #[test]
    fn sn_shared_down_all_m() {
        if ready("sn_shared_down_all_m") {
            return;
        }
        for nt in [17usize, 512, 1024, 2048] {
            check(GgmlType::Q8_0, nt, 640, 2560);
        }
    }

    /// ssm out_proj: K=6144, N=2560 (N が小さい側の実形)。
    #[test]
    fn sn_ssm_out_all_m() {
        if ready("sn_ssm_out_all_m") {
            return;
        }
        for nt in [17usize, 512, 1024, 2048] {
            check(GgmlType::Q8_0, nt, 6144, 2560);
        }
    }
}
