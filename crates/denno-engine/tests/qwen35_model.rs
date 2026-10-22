//! Qwen3.5 モデルレベル統合テスト (実 GGUF・ignored)。
//! 環境変数: QWEN35_DIR (config.json + *.gguf を含むディレクトリ・必須)。
//!
//! 検証内容:
//! 1. ロード (18 GDN + 6 full_attention・tied)
//! 2. forward が有限 logits・スケール妥当
//! 3. 決定論: 同一入力 2 回で bit 一致
//! 4. キャッシュ連続性: forward(3 トークン)+forward(1) == forward(4) bit 一致

use denno_engine::model::{load_model, Model};
use denno_engine::quant::QuantMode;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore]
fn qwen35_model_load_and_forward() {
    // CI (-- --ignored) ではモデルが無いのでスキップ(パニックにしない)
    let Ok(dir) = std::env::var("QWEN35_DIR") else {
        eprintln!("skip: QWEN35_DIR 未設定");
        return;
    };
    let mut model = load_model(std::path::Path::new(&dir), QuantMode::Int8)
        .expect("モデルロード失敗");

    // 決定論的トークン列 (実在範囲)
    let toks: Vec<u32> = (0..8).map(|i| 1000 + i * 97).collect();

    let mut c1 = model.new_cache();
    let l1 = model.forward(&toks, &mut c1).expect("forward 失敗");
    assert_eq!(l1.len(), 248320, "logits 次元 = vocab");
    assert!(l1.iter().all(|v| v.is_finite()), "logits 非有限");
    let mx = l1.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(mx > 0.1 && mx < 1e4, "logits スケール異常 mx={mx}");
    let (mut best, mut bi) = (f32::MIN, 0usize);
    for (i, v) in l1.iter().enumerate() {
        if *v > best {
            best = *v;
            bi = i;
        }
    }
    eprintln!("last-token argmax={bi} val={best:.4} maxabs={mx:.4}");

    // 決定論
    let mut c2 = model.new_cache();
    let l2 = model.forward(&toks, &mut c2).expect("forward 失敗 (2)");
    assert_eq!(bits(&l1), bits(&l2), "決定論違反");

    // キャッシュ連続性: 3+1 == 4 (bit 一致)
    let mut c3 = model.new_cache();
    let _ = model.forward(&toks[..3], &mut c3).expect("forward 失敗 (3)");
    let l3 = model.forward(&[toks[3]], &mut c3).expect("forward 失敗 (4)");
    let mut c4 = model.new_cache();
    let l4 = model.forward(&toks[..4], &mut c4).expect("forward 失敗 (5)");
    assert_eq!(bits(&l3), bits(&l4), "キャッシュ連続性違反 (3+1 != 4)");

    // decode 継続 (5..8 を 1 トークンずつ)
    let mut c5 = model.new_cache();
    let _ = model.forward(&toks[..4], &mut c5).unwrap();
    for t in &toks[4..] {
        let l = model.forward(&[*t], &mut c5).unwrap();
        assert!(l.iter().all(|v| v.is_finite()));
    }
    let l_last = model.forward(&[toks[7]], &mut c5).unwrap();
    let _ = l_last;
}

fn argmax(v: &[f32]) -> usize {
    let mut best = f32::MIN;
    let mut bi = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > best {
            best = *x;
            bi = i;
        }
    }
    bi
}

fn top5(v: &[f32]) -> Vec<(usize, f32)> {
    let mut s: Vec<(usize, f32)> = v.iter().enumerate().map(|(i, x)| (i, *x)).collect();
    s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    s.truncate(5);
    s
}

#[test]
#[ignore]
fn qwen35_int8_vs_fp32_logits() {
    // CI (-- --ignored) ではモデルが無いのでスキップ(パニックにしない)
    let Ok(dir) = std::env::var("QWEN35_DIR") else {
        eprintln!("skip: QWEN35_DIR 未設定");
        return;
    };
    let toks: Vec<u32> = (0..8).map(|i| 1000 + i * 97).collect();
    let mut m32 = load_model(std::path::Path::new(&dir), QuantMode::Fp32).unwrap();
    let mut mi8 = load_model(std::path::Path::new(&dir), QuantMode::Int8).unwrap();
    let l32 = m32.forward(&toks, &mut m32.new_cache()).unwrap();
    let li8 = mi8.forward(&toks, &mut mi8.new_cache()).unwrap();
    let mx: f32 = l32
        .iter()
        .zip(li8.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    eprintln!(
        "logits maxabs diff = {mx:.6}  argmax fp32={} int8={}",
        argmax(&l32),
        argmax(&li8)
    );
    eprintln!("fp32 top5: {:?}", top5(&l32));
    eprintln!("int8 top5: {:?}", top5(&li8));
}

/// A1: バッチ prefill (n>1・投影一括 GEMM) がトークン逐次経路と bit-exact である
/// ことを検証する (i32 累積は順序不変 → m>1 の行が m==1 と一致する根拠)。
#[test]
#[ignore]
fn qwen35_prefill_batch_matches_tokenwise() {
    // CI (-- --ignored) ではモデルが無いのでスキップ(パニックにしない)
    let Ok(dir) = std::env::var("QWEN35_DIR") else {
        eprintln!("skip: QWEN35_DIR 未設定");
        return;
    };
    let toks: Vec<u32> = (0..12).map(|i| 1000 + i * 37).collect();

    // トークン逐次 (n==1 を 12 回)
    let mut m1 = load_model(std::path::Path::new(&dir), QuantMode::Int8).unwrap();
    let mut c1 = m1.new_cache();
    let mut l1 = Vec::new();
    for t in &toks {
        l1 = m1.forward(&[*t], &mut c1).unwrap();
    }

    // バッチ prefill (n==12 を 1 回)
    let mut m2 = load_model(std::path::Path::new(&dir), QuantMode::Int8).unwrap();
    let mut c2 = m2.new_cache();
    let l2 = m2.forward(&toks, &mut c2).unwrap();

    assert_eq!(l1.len(), l2.len());
    for (i, (a, b)) in l1.iter().zip(l2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "prefill batch != tokenwise at logits[{i}]"
        );
    }
    // キャッシュ連続性: どちらの経路でも以後の 1 トークン forward が一致
    let la = m1.forward(&[toks[0]], &mut c1).unwrap();
    let lb = m2.forward(&[toks[0]], &mut c2).unwrap();
    for (a, b) in la.iter().zip(lb.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    eprintln!("prefill batch == tokenwise (bit-exact, {} logits)", l1.len());
}

/// P19-A1: GDN GPU スキャンカーネルと CPU 逐次 (forward_token_core) の近似一致。
/// GDN は bit 一致不要契約 (maxabs < 1e-3 の緩和枠)。GPU が無い環境では skip。
/// cuda feature が無いビルド (CI 等) ではテスト自体が存在しない。
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn gdn_gpu_scan_close_to_cpu() {
    use denno_engine::cuda::gdn_scan_batch;
    use denno_engine::gdn::{forward_token_core, GdnCore, GdnParams, GdnState};

    // GPU が使えない環境 (CI 等) では静かに skip
    if std::env::var("DENNO_CUDA").as_deref() != Ok("1") {
        eprintln!("skip: DENNO_CUDA=1 未設定");
        return;
    }
    // 決定的な疑似ランダム (LCG)
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut rnd = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    let (nk, nv, dk, dv) = (2usize, 4usize, 8usize, 8usize);
    let d_conv = 4usize;
    let n = 12usize;
    let conv_dim = 2 * nk * dk + nv * dv;
    let p = GdnParams {
        n_embd: 16,
        n_k_heads: nk,
        n_v_heads: nv,
        head_k_dim: dk,
        head_v_dim: dv,
        d_conv,
        eps: 1e-6,
    };
    let wconv: Vec<f32> = (0..conv_dim * d_conv).map(|_| rnd()).collect();
    let ssm_a: Vec<f32> = (0..nv).map(|_| -1.0 - rnd().abs()).collect();
    let ssm_dt: Vec<f32> = (0..nv).map(|_| rnd()).collect();
    let ssm_norm: Vec<f32> = (0..dv).map(|_| 0.5 + rnd().abs()).collect();
    let qkv: Vec<f32> = (0..n * conv_dim).map(|_| rnd()).collect();
    let z: Vec<f32> = (0..n * nv * dv).map(|_| rnd()).collect();
    let beta: Vec<f32> = (0..n * nv).map(|_| rnd()).collect();
    let alpha: Vec<f32> = (0..n * nv).map(|_| rnd()).collect();
    let conv_st0: Vec<f32> = (0..conv_dim * (d_conv - 1)).map(|_| rnd()).collect();
    let ssm_st0: Vec<f32> = (0..nv * dk * dv).map(|_| rnd()).collect();

    // ---- CPU 逐次 (トークンごとに forward_token_core) ----
    let mut st = GdnState {
        conv: conv_st0.clone(),
        ssm: ssm_st0.clone(),
    };
    let cw = GdnCore {
        conv1d: &wconv,
        ssm_a: &ssm_a,
        ssm_dt: &ssm_dt,
        ssm_norm: &ssm_norm,
    };
    let nqkv = conv_dim;
    let mut ys_cpu = vec![0.0f32; n * nv * dv];
    for t in 0..n {
        let y = forward_token_core(
            &p,
            &cw,
            &qkv[t * nqkv..(t + 1) * nqkv],
            &z[t * nv * dv..(t + 1) * nv * dv],
            &beta[t * nv..(t + 1) * nv],
            &alpha[t * nv..(t + 1) * nv],
            &mut st,
        );
        ys_cpu[t * nv * dv..(t + 1) * nv * dv].copy_from_slice(&y);
    }

    // ---- GPU (1 launch) ----
    let mut conv_g = conv_st0.clone();
    let mut ssm_g = ssm_st0.clone();
    let ys_gpu = gdn_scan_batch(
        &qkv,
        &z,
        &beta,
        &alpha,
        &mut conv_g,
        &mut ssm_g,
        &wconv,
        &ssm_a,
        &ssm_dt,
        &ssm_norm,
        n,
        nk,
        nv,
        dk,
        dv,
        d_conv,
        p.eps,
        1.0 / (dk as f32).sqrt(),
        0, // li (s164: gdn_scan_batch の層番号 — 状態正本のキー)
        1, // kv (同・KV キャッシュ ID)
    )
    .expect("GPU スキャンが None (GPU 未初期化?)");

    let mut maxd = 0.0f32;
    for (a, b) in ys_cpu.iter().zip(ys_gpu.iter()) {
        maxd = maxd.max((a - b).abs());
    }
    // 状態も比較
    let mut maxd_s = 0.0f32;
    for (a, b) in st.ssm.iter().zip(ssm_g.iter()) {
        maxd_s = maxd_s.max((a - b).abs());
    }
    // conv 状態は整数値のロールで同一のはず (bit 一致)
    assert_eq!(st.conv, conv_g, "conv 状態が不一致");
    eprintln!("gdn gpu vs cpu: y maxabs={maxd:.3e} ssm maxabs={maxd_s:.3e}");
    assert!(maxd < 1e-3, "y maxabs diff {maxd}");
    assert!(maxd_s < 1e-3, "ssm maxabs diff {maxd_s}");
}

/// A1-r2: GDN chunked scan 経路 (n >= 2048 で 3-pass chunked カーネルにルーティング)
/// と CPU 逐次 (forward_token_core) の近似一致 + 決定論。
/// **緩和契約 (GDN)**: bit 一致不要・maxabs < 1e-3・cos > 0.999・決定論必須。
/// conv 状態は整数値ロールで bit 一致。GPU が無い環境では skip。
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn gdn_gpu_chunked_scan_close_to_cpu() {
    use denno_engine::cuda::gdn_scan_batch;
    use denno_engine::gdn::{forward_token_core, GdnCore, GdnParams, GdnState};

    // GPU が使えない環境 (CI 等) では静かに skip
    if std::env::var("DENNO_CUDA").as_deref() != Ok("1") {
        eprintln!("skip: DENNO_CUDA=1 未設定");
        return;
    }
    // 決定的な疑似ランダム (LCG) — 既存テストと同じ定数
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut rnd = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((rng >> 33) as f32) / (u32::MAX as f32) - 0.5
    };

    let (nk, nv, dk, dv) = (2usize, 4usize, 8usize, 8usize);
    let d_conv = 4usize;
    // n >= 2048 で chunked ルーティングが発動 (chunk_plan(5000, 2048, 1024, 16)
    // = 5 chunks × 1000)
    let n = 5000usize;
    let conv_dim = 2 * nk * dk + nv * dv;
    let p = GdnParams {
        n_embd: 16,
        n_k_heads: nk,
        n_v_heads: nv,
        head_k_dim: dk,
        head_v_dim: dv,
        d_conv,
        eps: 1e-6,
    };
    let wconv: Vec<f32> = (0..conv_dim * d_conv).map(|_| rnd()).collect();
    let ssm_a: Vec<f32> = (0..nv).map(|_| -1.0 - rnd().abs()).collect();
    let ssm_dt: Vec<f32> = (0..nv).map(|_| rnd()).collect();
    let ssm_norm: Vec<f32> = (0..dv).map(|_| 0.5 + rnd().abs()).collect();
    let qkv: Vec<f32> = (0..n * conv_dim).map(|_| rnd()).collect();
    let z: Vec<f32> = (0..n * nv * dv).map(|_| rnd()).collect();
    let beta: Vec<f32> = (0..n * nv).map(|_| rnd()).collect();
    let alpha: Vec<f32> = (0..n * nv).map(|_| rnd()).collect();
    let conv_st0: Vec<f32> = (0..conv_dim * (d_conv - 1)).map(|_| rnd()).collect();
    let ssm_st0: Vec<f32> = (0..nv * dk * dv).map(|_| rnd()).collect();

    // ---- CPU 逐次 (トークンごとに forward_token_core) ----
    let mut st = GdnState {
        conv: conv_st0.clone(),
        ssm: ssm_st0.clone(),
    };
    let cw = GdnCore {
        conv1d: &wconv,
        ssm_a: &ssm_a,
        ssm_dt: &ssm_dt,
        ssm_norm: &ssm_norm,
    };
    let nqkv = conv_dim;
    let mut ys_cpu = vec![0.0f32; n * nv * dv];
    for t in 0..n {
        let y = forward_token_core(
            &p,
            &cw,
            &qkv[t * nqkv..(t + 1) * nqkv],
            &z[t * nv * dv..(t + 1) * nv * dv],
            &beta[t * nv..(t + 1) * nv],
            &alpha[t * nv..(t + 1) * nv],
            &mut st,
        );
        ys_cpu[t * nv * dv..(t + 1) * nv * dv].copy_from_slice(&y);
    }

    // ---- GPU (chunked 3 カーネル: n=5000 >= 2048 でルーティング) ----
    let run_gpu = || {
        let mut conv_g = conv_st0.clone();
        let mut ssm_g = ssm_st0.clone();
        let ys = gdn_scan_batch(
            &qkv,
            &z,
            &beta,
            &alpha,
            &mut conv_g,
            &mut ssm_g,
            &wconv,
            &ssm_a,
            &ssm_dt,
            &ssm_norm,
            n,
            nk,
            nv,
            dk,
            dv,
            d_conv,
            p.eps,
            1.0 / (dk as f32).sqrt(),
            0, // li (s164: gdn_scan_batch の層番号 — 状態正本のキー)
            1, // kv (同・KV キャッシュ ID)
        )
        .expect("GPU スキャンが None (GPU 未初期化?)");
        (ys, conv_g, ssm_g)
    };
    let (ys_gpu, conv_g, ssm_g) = run_gpu();

    let mut maxd = 0.0f32;
    for (a, b) in ys_cpu.iter().zip(ys_gpu.iter()) {
        maxd = maxd.max((a - b).abs());
    }
    let mut maxd_s = 0.0f32;
    for (a, b) in st.ssm.iter().zip(ssm_g.iter()) {
        maxd_s = maxd_s.max((a - b).abs());
    }
    let dot: f32 = ys_cpu.iter().zip(ys_gpu.iter()).map(|(a, b)| a * b).sum();
    let na: f32 = ys_cpu.iter().map(|a| a * a).sum::<f32>().sqrt();
    let nb: f32 = ys_gpu.iter().map(|b| b * b).sum::<f32>().sqrt();
    let cos = dot / (na * nb);
    // conv 状態は整数値のロールで同一のはず (bit 一致)
    assert_eq!(st.conv, conv_g, "conv 状態が不一致");
    eprintln!(
        "gdn chunked gpu vs cpu: y maxabs={maxd:.3e} ssm maxabs={maxd_s:.3e} cos={cos:.8}"
    );
    assert!(maxd < 1e-3, "y maxabs diff {maxd}");
    assert!(maxd_s < 1e-3, "ssm maxabs diff {maxd_s}");
    assert!(cos > 0.999, "cosine {cos}");

    // ---- 決定論: 同一入力での 2 回目の chunked 呼び出しは bit 一致 ----
    let (ys_gpu2, conv_g2, ssm_g2) = run_gpu();
    for (a, b) in ys_gpu.iter().zip(ys_gpu2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "chunked scan 非決定 (y)");
    }
    for (a, b) in ssm_g.iter().zip(ssm_g2.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "chunked scan 非決定 (ssm)");
    }
    assert_eq!(conv_g, conv_g2, "chunked scan 非決定 (conv)");
    eprintln!("gdn chunked gpu deterministic: bit-exact across 2 runs");
}



/// A2.5: 9B int8 で repetition penalty 経路を検証 (実 GGUF・ignored)。
/// 環境変数: QWEN35_9B_DIR (config.json + *.gguf を含むディレクトリ・例: data/qwen35-9b)。
/// 未設定なら graceful skip (CI ではモデルが無いため常時 skip)。
///
/// 検証内容:
/// 1. greedy + penalty 1.1 が同一入力→同一出力 (連続 2 回 bit 一致・決定論契約)
/// 2. 反復退化の代理指標: 最頻 4-gram の出現数がテキスト長に対して過大でない
///    (penalty 無しの greedy は数トークンでフレーズ反復ループに落ちる)
#[test]
#[ignore]
fn qwen35_9b_int8_penalty_determinism() {
    use denno_engine::{ChatMessage, Engine, GenParams};
    let Ok(dir) = std::env::var("QWEN35_9B_DIR") else {
        eprintln!("skip: QWEN35_9B_DIR 未設定");
        return;
    };
    let mut e = Engine::load(
        std::path::Path::new(&dir),
        "q35-9b-test",
        Some(42),
        QuantMode::Int8,
    )
    .expect("9B ロード失敗");
    let msgs = vec![ChatMessage { role: "user".into(), content: "日本の首都は?".into() }];
    let params = GenParams {
        max_tokens: 32,
        temperature: Some(0.0),
        seed: Some(42),
        repetition_penalty: Some(1.1),
        ..Default::default()
    };
    let a = e.chat(None, &msgs, &params, |_| {}).unwrap();
    let b = e.chat(None, &msgs, &params, |_| {}).unwrap();
    assert_eq!(a.text, b.text, "greedy+penalty の決定論違反");
    assert_eq!(a.output_tokens, b.output_tokens);

    // 反復退化の代理指標: 文字 4-gram の最頻出現数が過大でないこと。
    // しきい値は緩く (長さの 1/6)、明らかなフレーズ反復ループだけを検出する。
    let chars: Vec<char> = a.text.chars().collect();
    if chars.len() >= 12 {
        let mut counts: std::collections::HashMap<Vec<char>, usize> =
            std::collections::HashMap::new();
        for w in chars.windows(4) {
            *counts.entry(w.to_vec()).or_default() += 1;
        }
        let top = counts.values().copied().max().unwrap_or(0);
        assert!(
            top * 6 <= chars.len(),
            "反復ループ検出: 最頻 4-gram が {top} 回 / 長さ {}",
            chars.len()
        );
    }
    eprintln!(
        "penalty determinism ok: out={} tokens, text={:?}",
        a.output_tokens,
        a.text
    );
}
