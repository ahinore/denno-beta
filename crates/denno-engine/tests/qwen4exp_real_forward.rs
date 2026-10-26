//! 実モデル（Qwen3.8-Flash-Next UD-Q4_K_XL）の 1 トークン forward スモーク。
//!
//! `DENNO_Q4_DIR` に 4 分割 GGUF のディレクトリを渡すと、48 層ぶんの実重みを
//! mmap 越しに読んで `Qwen4Exp::forward_token` を 1 回走らせ、
//! - logits が有限で `n_vocab` 個
//! - 層出口のノルムが有限で、層ごとに動いている
//! - 2 回目が同じ値（決定性）
//! を検査する。attention は恒等（`IdentityAttn`）なので、これは
//! hyper-connection / PLE gather / MoE routing / shared expert / 最終 mixer / head の
//! 実重み経路の検証であって、モデル出力の正しさの検証ではない。
//!
//! 実行:
//!   DENNO_Q4_DIR=<4 分割 GGUF のディレクトリ> \
//!   cargo test --release --test qwen4exp_real_forward -- --ignored --nocapture
//!
//! 1 トークンあたり CPU で数秒〜数十秒かかる（MoE の expert GEMV が支配的）。

use std::path::PathBuf;

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::Qwen4Exp;

fn parts() -> Option<Vec<PathBuf>> {
    let dir = PathBuf::from(std::env::var("DENNO_Q4_DIR").ok()?);
    let mut v: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().map(|x| x == "gguf").unwrap_or(false)
                && p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.contains("of-00004"))
                    .unwrap_or(false)
        })
        .collect();
    v.sort();
    if v.len() != 4 {
        return None;
    }
    Some(v)
}

#[test]
#[ignore]
fn real_model_one_token_forward() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };

    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let mut m = Qwen4Exp::new(&f).expect("モデルを組めない");
    let (n_vocab, n_layer) = (m.n_vocab(), m.n_layer());
    eprintln!("n_vocab={n_vocab} n_layer={n_layer}");

    // 合成のトークン列 (語彙内)。PLE の窓は末尾 3 個。
    let ctx: Vec<u32> = vec![1000, 2000, 3000, 4000];
    for &t in &ctx {
        assert!((t as usize) < n_vocab, "トークン {t} が語彙 {n_vocab} の外");
    }

    let mut norms: Vec<(usize, f32)> = Vec::new();
    let t0 = std::time::Instant::now();
    let logits = m
        .forward_token_with(&ctx, |il, res| {
            let mut s = 0.0f64;
            for &x in res {
                s += (x as f64) * (x as f64);
            }
            norms.push((il, s.sqrt() as f32));
            Ok(())
        })
        .expect("forward が失敗した");
    let dt = t0.elapsed();

    assert_eq!(logits.len(), n_vocab, "logits の長さ");
    let bad = logits.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "logits に非有限が {bad} 個");

    assert_eq!(norms.len(), n_layer, "層フックの回数");
    let nonfinite = norms.iter().filter(|(_, n)| !n.is_finite()).count();
    assert_eq!(nonfinite, 0, "層出口に非有限が {nonfinite} 個");
    // 全層が同じ値 = 層を回っていない
    let first = norms[0].1;
    assert!(
        norms.iter().any(|(_, n)| (*n - first).abs() > 0.0),
        "全層のノルムが同じ (層ループが効いていない)"
    );
    for (il, n) in norms.iter().take(4) {
        eprintln!("LAYER il={il} norm={n:.6}");
    }
    eprintln!("LAYER last norm={:.6}", norms[norms.len() - 1].1);

    // top-5
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("TOP5 {:?}", &idx[..5]);
    eprintln!("TOP5_LOGITS {:?}", idx[..5].iter().map(|&i| logits[i]).collect::<Vec<f32>>());
    eprintln!("forward_1token_ms={:.1}", dt.as_secs_f64() * 1e3);

    // 決定性: 同じ ctx でもう 1 回。状態 (KV / ssm) は IdentityAttn なので持たない。
    let again = m.forward_token(&ctx).expect("2 回目の forward");
    let diff = logits
        .iter()
        .zip(again.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(diff, 0, "2 回目で {diff} 個の logit がビット一致しない");
}
