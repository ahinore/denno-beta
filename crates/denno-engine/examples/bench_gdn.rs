//! GDN 層 1 本の単体ベンチ (27B 実形状)。
//!
//! モデルを起動せず、GDN スキャン 1 回のコストを測る。形状は 27B の実値
//! (nk 16 / nv 48 / dk dv 128 / conv 4 / hidden 5120)。n=1 が decode、
//! n=300 が prefill の 1 チャンクに相当する。
//!
//! 使い方:
//!   cargo run --release -p denno-engine --features cuda --example bench_gdn
//!   ... --example bench_gdn -- --layers 48 --iters 30
//!
//! 数値の確認: n=1 は CPU の `gdn::forward_token_core` と突き合わせる。

use std::time::Instant;

use denno_engine::gdn::{forward_token_core, GdnCore, GdnParams, GdnState};

fn arg_u(args: &[String], name: &str, dflt: usize) -> usize {
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                if let Ok(v) = v.parse() {
                    return v;
                }
            }
        }
    }
    dflt
}

/// 決定論的な擬似乱数 (xorshift)。重みと入力の生成に使う。
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (((self.0 >> 24) as u32) as f32 / 8_388_608.0) - 1.0
    }
    fn fill(&mut self, v: &mut [f32]) {
        for x in v.iter_mut() {
            *x = self.next_f32();
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let layers = arg_u(&args, "--layers", 48);
    let iters = arg_u(&args, "--iters", 30);
    std::env::set_var("DENNO_CUDA", "1");

    let p = GdnParams {
        n_embd: 5120,
        n_k_heads: 16,
        n_v_heads: 48,
        head_k_dim: 128,
        head_v_dim: 128,
        d_conv: 4,
        eps: 1e-6,
    };
    let conv_dim = p.conv_dim();
    let vd = p.value_dim();
    let (nv, dk, dv, dc) = (p.n_v_heads, p.head_k_dim, p.head_v_dim, p.d_conv);
    let state_mb = (p.conv_state_len() + p.ssm_state_len()) as f64 * 4.0 / 1e6;
    println!(
        "GDN layer: conv_dim {conv_dim}  nv {nv}  dk {dk}  dv {dv}  conv {dc}  \
         state {state_mb:.1} MB/layer  layers={layers} iters={iters}"
    );

    let mut rng = Rng(0x1234_5678_9abc_def0);
    let mut wconv = vec![0f32; conv_dim * dc];
    let mut wa = vec![0f32; nv];
    let mut wdt = vec![0f32; nv];
    let mut wnorm = vec![0f32; dv];
    rng.fill(&mut wconv);
    rng.fill(&mut wa);
    for v in wdt.iter_mut() {
        *v = *v * 0.1;
    }
    for v in wnorm.iter_mut() {
        *v = 1.0 + *v * 0.1;
    }
    // ssm_a は負値 (減衰)。ssm_dt は小さい正のオフセット。
    let ssm_a: Vec<f32> = (0..nv).map(|h| -0.5 - 0.5 * (h % 7) as f32 / 7.0).collect();

    let q_scale = 1.0 / (dk as f32).sqrt();
    let mut worst = 0.0f64;

    for &n in &[1usize, 8, 32, 300] {
        let mut qkv = vec![0f32; n * conv_dim];
        let mut z = vec![0f32; n * vd];
        let mut beta = vec![0f32; n * nv];
        let mut alpha = vec![0f32; n * nv];
        rng.fill(&mut qkv);
        rng.fill(&mut z);
        for v in beta.iter_mut() {
            *v = *v * 0.5;
        }
        for v in alpha.iter_mut() {
            *v = *v * 0.5;
        }

        // 状態は毎回ゼロから (kv を分けてデバイス正本を再 seed させる)
        let mut kv = 9_000_000u64 + n as u64;
        let mut conv = vec![0f32; p.conv_state_len()];
        let mut ssm = vec![0f32; p.ssm_state_len()];

        let t = Instant::now();
        let first = denno_engine::cuda::gdn_scan_batch(
            &qkv, &z, &beta, &alpha, &mut conv, &mut ssm, &wconv, &ssm_a, &wdt, &wnorm, n, p.n_k_heads,
            nv, dk, dv, dc, p.eps, q_scale, 0, kv,
        );
        let first_s = t.elapsed().as_secs_f64();
        let Some(y0) = first else {
            println!("n={n:>3}: GPU path not taken (CPU fallback)");
            continue;
        };

        let t = Instant::now();
        for _ in 0..iters {
            let ok = denno_engine::cuda::gdn_scan_batch(
                &qkv, &z, &beta, &alpha, &mut conv, &mut ssm, &wconv, &ssm_a, &wdt, &wnorm, n,
                p.n_k_heads, nv, dk, dv, dc, p.eps, q_scale, 0, kv,
            );
            assert!(ok.is_some(), "GPU path dropped out");
        }
        let per = t.elapsed().as_secs_f64() / iters as f64;

        // 決定論: 新しい kv で同じ入力をもう一度流し、出力が bit 一致するか
        kv += 1000;
        let mut conv2 = vec![0f32; p.conv_state_len()];
        let mut ssm2 = vec![0f32; p.ssm_state_len()];
        let y1 = denno_engine::cuda::gdn_scan_batch(
            &qkv, &z, &beta, &alpha, &mut conv2, &mut ssm2, &wconv, &ssm_a, &wdt, &wnorm, n, p.n_k_heads,
            nv, dk, dv, dc, p.eps, q_scale, 0, kv,
        )
        .expect("second run");
        let det = y0.iter().zip(&y1).all(|(a, b)| a.to_bits() == b.to_bits());

        println!(
            "n={n:>3}  {:.3} ms/call  first {:.3} s  x{layers} layers = {:.1} ms/token  det={}",
            per * 1e3,
            first_s,
            per * 1e3 * layers as f64,
            if det { "OK" } else { "NG" }
        );

        // n=1 は CPU 実装と突き合わせる
        if n == 1 {
            let core = GdnCore {
                conv1d: &wconv,
                ssm_a: &ssm_a,
                ssm_dt: &wdt,
                ssm_norm: &wnorm,
            };
            let mut st = GdnState::new(&p);
            let yc = forward_token_core(&p, &core, &qkv, &z, &beta, &alpha, &mut st);
            let mut mx = 0.0f64;
            for (a, b) in y0.iter().zip(&yc) {
                let e = (a - b).abs() as f64;
                if e > mx {
                    mx = e;
                }
            }
            worst = worst.max(mx);
            println!("      maxabs vs CPU = {mx:.2e}");
        }
    }
    if worst > 1e-3 {
        println!("WARN: deviation {worst:.2e} exceeds the R3 budget");
    }
}