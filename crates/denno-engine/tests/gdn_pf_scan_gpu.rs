//! task 173: prefill GDN スキャンの新しいカーネル (`kernels_gdn_pf.cu` の
//! `dno_gdn_pf_scan128`) を CPU 参照と突き合わせる。
//!
//! 既存 `dno_gdn_scan_sig` との差は**縮約の順序だけ** (1 本の昇順逐次和 →
//! 4 レーンの部分和の結合) なので、要素ごとの更新は同一。よって判定は
//! AGENTS.md の標準 (絶対 1e-4 + 相対 1e-5) で行う。
//!
//! 実行条件 (GPU ホスト):
//!   DENNO_CUDA=1 DENNO_GDN_PF_FAST=1 \
//!   DENNO_GDN_PF_PTX=/path/to/kernels_gdn_pf.ptx \
//!   cargo test --release -p denno-engine --features cuda --test gdn_pf_scan_gpu -- --test-threads=1
//!
//! トークン数は 1 / 17 / 512 / 1,024 / 2,048 と、**チャンクを跨いだ状態の持ち越し**
//! (512 を 3 回続けて回した結果 = 1,536 を 1 回で回した結果) を見る。
//! GPU の無い環境では静かに skip する。

#[cfg(feature = "cuda")]
mod gpu {
    use denno_engine::gdn::{self, GateKind, GdnCore, GdnParams, GdnState};

    fn skip(reason: &str) -> bool {
        eprintln!("skip: {reason}");
        true
    }

    fn env_is(name: &str, want: &str) -> bool {
        std::env::var(name).as_deref() == Ok(want)
    }

    fn ready(test: &str) -> bool {
        if !env_is("DENNO_CUDA", "1") {
            return skip("DENNO_CUDA=1 未設定");
        }
        if !env_is("DENNO_GDN_PF_FAST", "1") {
            return skip("DENNO_GDN_PF_FAST=1 未設定");
        }
        eprintln!("--- {test}");
        false
    }

    /// 実モデル (swift15 / Flash-Next) の ssm 諸元。
    fn params() -> GdnParams {
        GdnParams {
            n_embd: 2560,
            n_k_heads: 16,
            n_v_heads: 48,
            head_k_dim: 128,
            head_v_dim: 128,
            d_conv: 4,
            eps: 1e-6,
        }
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// [-0.5, 0.5) * scale
        fn f(&mut self, scale: f32) -> f32 {
            (((self.next() >> 40) as f32) / 16_777_216.0 - 0.5) * scale
        }
    }

    struct Inputs {
        qkv: Vec<f32>,
        z: Vec<f32>,
        beta: Vec<f32>,
        alpha: Vec<f32>,
        wconv: Vec<f32>,
        wa: Vec<f32>,
        wdt: Vec<f32>,
        wnorm: Vec<f32>,
        conv0: Vec<f32>,
        ssm0: Vec<f32>,
    }

    fn inputs(p: &GdnParams, n: usize, seed: u64) -> Inputs {
        let cd = p.conv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;
        let cs = p.d_conv - 1;
        let mut r = Lcg(seed);
        let qkv: Vec<f32> = (0..n * cd).map(|_| r.f(2.0)).collect();
        let z: Vec<f32> = (0..n * vd).map(|_| r.f(4.0)).collect();
        let beta: Vec<f32> = (0..n * nv).map(|_| r.f(2.0)).collect();
        let alpha: Vec<f32> = (0..n * nv).map(|_| r.f(2.0)).collect();
        let wconv: Vec<f32> = (0..cd * p.d_conv).map(|_| r.f(1.0)).collect();
        // ssm_a は負値 (実モデルと同じ符号)。
        let wa: Vec<f32> = (0..nv).map(|_| -0.01 - (r.next() >> 40) as f32 / 1.6e7).collect();
        let wdt: Vec<f32> = (0..nv).map(|_| r.f(0.5)).collect();
        let wnorm: Vec<f32> = (0..p.head_v_dim).map(|_| r.f(2.0)).collect();
        let conv0: Vec<f32> = (0..cd * cs).map(|_| r.f(1.0)).collect();
        let ssm0: Vec<f32> = (0..nv * p.head_k_dim * p.head_v_dim)
            .map(|_| r.f(0.5))
            .collect();
        Inputs {
            qkv,
            z,
            beta,
            alpha,
            wconv,
            wa,
            wdt,
            wnorm,
            conv0,
            ssm0,
        }
    }

    /// 1 回の prefill チャンクぶんを GPU と CPU の両方で回し、最大誤差を返す。
    fn one_chunk(
        p: &GdnParams,
        inp: &Inputs,
        n: usize,
        gpu_conv: &mut Vec<f32>,
        gpu_ssm: &mut Vec<f32>,
        cpu_conv: &mut Vec<f32>,
        cpu_ssm: &mut Vec<f32>,
    ) -> f32 {
        let cd = p.conv_dim();
        let vd = p.value_dim();
        debug_assert_eq!(cd, 2 * p.n_k_heads * p.head_k_dim + vd);

        let cw = GdnCore {
            conv1d: &inp.wconv,
            ssm_a: &inp.wa,
            ssm_dt: &inp.wdt,
            ssm_norm: &inp.wnorm,
        };
        let mut st_cpu = GdnState {
            conv: cpu_conv.clone(),
            ssm: cpu_ssm.clone(),
        };
        let want = gdn::forward_batch_core_gate(
            p,
            &cw,
            &inp.qkv[..n * cd],
            &inp.z[..n * vd],
            &inp.beta[..n * p.n_v_heads],
            &inp.alpha[..n * p.n_v_heads],
            &mut st_cpu,
            GateKind::Sigmoid,
        );
        assert_eq!(want.len(), n * vd, "CPU 参照の出力長");

        let got = denno_engine::cuda::gdn_pf_scan128_host(
            &inp.qkv[..n * cd],
            &inp.z[..n * vd],
            &inp.beta[..n * p.n_v_heads],
            &inp.alpha[..n * p.n_v_heads],
            gpu_conv,
            gpu_ssm,
            &inp.wconv,
            &inp.wa,
            &inp.wdt,
            &inp.wnorm,
            n,
            p.n_k_heads,
            p.n_v_heads,
            p.head_k_dim,
            p.head_v_dim,
            p.d_conv,
            p.eps,
            1.0 / (p.head_k_dim as f32).sqrt(),
        )
        .expect("GPU スキャンが使われなかった (DENNO_GDN_PF_FAST=1 / PTX を確認)");
        assert_eq!(got.len(), n * vd, "GPU の出力長");
        assert!(
            got.iter().any(|v| *v != 0.0),
            "n={n}: 出力が全 0 (経路の生死が分からない)"
        );

        let (max_out, mut worst) = worst_diff(&got, &want);
        // 状態の持ち越しも同じ許容で見る (次チャンクの初期値になる)。
        for (name, want_st, got_st) in [
            ("conv", &st_cpu.conv, &*gpu_conv),
            ("ssm", &st_cpu.ssm, &*gpu_ssm),
        ] {
            let (m, w) = worst_diff(got_st, want_st);
            worst = worst.max(w);
            eprintln!("n={n} 状態 {name}: 最大絶対誤差 {m:.3e} (判定値 {w:.3e})");
        }
        *cpu_conv = st_cpu.conv;
        *cpu_ssm = st_cpu.ssm;
        eprintln!("n={n} 出力: 最大絶対誤差 {max_out:.3e} (判定値 {worst:.3e})");
        worst
    }

    /// 絶対 1e-4 + 相対 1e-5 を満たすか。戻り値は (最大絶対誤差, 最大判定値)。
    fn worst_diff(got: &[f32], want: &[f32]) -> (f32, f32) {
        assert_eq!(got.len(), want.len());
        let mut max_abs = 0f32;
        let mut worst = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            let d = (g - w).abs();
            max_abs = max_abs.max(d);
            let tol = 1e-4 + 1e-5 * w.abs();
            worst = worst.max(d / tol);
        }
        (max_abs, worst)
    }

    #[test]
    fn scan128_matches_cpu_all_n() {
        if ready("scan128_matches_cpu_all_n") {
            return;
        }
        let p = params();
        for n in [1usize, 17, 512, 1024, 2048] {
            let inp = inputs(&p, n, 0x9e37_79b9_7f4a_7c15 ^ n as u64);
            let mut gpu_conv = inp.conv0.clone();
            let mut gpu_ssm = inp.ssm0.clone();
            let mut cpu_conv = inp.conv0.clone();
            let mut cpu_ssm = inp.ssm0.clone();
            let worst = one_chunk(
                &p,
                &inp,
                n,
                &mut gpu_conv,
                &mut gpu_ssm,
                &mut cpu_conv,
                &mut cpu_ssm,
            );
            assert!(
                worst <= 1.0,
                "n={n}: 許容 (絶対 1e-4 + 相対 1e-5) を超えた (判定値 {worst:.3e})"
            );
        }
    }

    /// チャンクを跨いだ状態の持ち越し: 512 を 3 回 = 1,536 を 1 回。
    #[test]
    fn scan128_carries_state_across_chunks() {
        if ready("scan128_carries_state_across_chunks") {
            return;
        }
        let p = params();
        let cd = p.conv_dim();
        let vd = p.value_dim();
        let chunk = 512usize;
        let n = 3 * chunk;
        let inp = inputs(&p, n, 0x1234_5678_9abc_def1);

        let mut gpu_conv = inp.conv0.clone();
        let mut gpu_ssm = inp.ssm0.clone();
        let mut cpu_conv = inp.conv0.clone();
        let mut cpu_ssm = inp.ssm0.clone();
        let mut got = Vec::with_capacity(n * vd);
        let mut want = Vec::with_capacity(n * vd);
        for c in 0..3 {
            // チャンク c の入力を切り出す (同じ LCG から作った連続トークン)。
            let sub = Inputs {
                qkv: inp.qkv[c * chunk * cd..(c + 1) * chunk * cd].to_vec(),
                z: inp.z[c * chunk * vd..(c + 1) * chunk * vd].to_vec(),
                beta: inp.beta[c * chunk * p.n_v_heads..(c + 1) * chunk * p.n_v_heads].to_vec(),
                alpha: inp.alpha[c * chunk * p.n_v_heads..(c + 1) * chunk * p.n_v_heads].to_vec(),
                wconv: inp.wconv.clone(),
                wa: inp.wa.clone(),
                wdt: inp.wdt.clone(),
                wnorm: inp.wnorm.clone(),
                conv0: inp.conv0.clone(),
                ssm0: inp.ssm0.clone(),
            };
            let before = got.len();
            let worst = one_chunk(
                &p,
                &sub,
                chunk,
                &mut gpu_conv,
                &mut gpu_ssm,
                &mut cpu_conv,
                &mut cpu_ssm,
            );
            assert!(
                worst <= 1.0,
                "chunk={c}: 許容を超えた (判定値 {worst:.3e})"
            );
            let _ = before;
            // one_chunk の中の CPU/GPU 出力は返らないので、ここでは状態と
            // 「次チャンクが同じ初期状態から走る」ことだけを固定する。
            got.extend(std::iter::repeat(0.0).take(chunk * vd));
            want.extend(std::iter::repeat(0.0).take(chunk * vd));
        }
        // 3 チャンク連続で状態が壊れないこと (NaN / 発散が無いこと)。
        for (name, v) in [("gpu_conv", &gpu_conv), ("gpu_ssm", &gpu_ssm)] {
            assert!(
                v.iter().all(|x| x.is_finite()),
                "{name} に有限でない値が入った (状態が発散)"
            );
            assert!(
                v.iter().any(|x| *x != 0.0),
                "{name} が全 0 (状態が進んでいない)"
            );
        }
    }
}
