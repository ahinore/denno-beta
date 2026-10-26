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
//!
//! 2026-10-26: `forward_token_with` が PLE の n-gram 窓を古い順のまま渡していたのを
//! 直した（`PleNgram::indices` は新しい順を要求する）。この修正の前後で下の数値は変わる。

use std::path::PathBuf;

use denno_engine::gguf::GgufFile;
use denno_engine::model::linear_gguf::GgufLinear;
use denno_engine::model::qwen4exp::{Qwen4Exp, Qwen4ExpParams};
use denno_engine::model::qwen4exp_decode::{DecoderConfig, Qwen4ExpDecoder};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_ssm::SsmAttn;

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


/// 実重みで attention も本物にして 1 トークン回す。
///
/// `real_model_one_token_forward` が attention 恒等で hyper-connection / PLE /
/// MoE / head を見るのに対し、こちらは ssm 36 層 + full attention 12 層の実経路を
/// 通す。**llama.cpp との数値一致はまだ取っていない**ので、ここで見るのは
/// 「落ちない・有限・決定性」までで、値の正しさの証明ではない。
#[test]
#[ignore]
fn real_model_one_token_forward_with_attention() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let max_ctx = 8usize;

    let t_load = std::time::Instant::now();
    let full = FullAttn::from_gguf(&f, &p, max_ctx).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let n_ssm = attn.n_ssm();
    let n_full = attn.n_full();
    let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");
    eprintln!(
        "LOAD attn_ms={:.1} n_ssm={n_ssm} n_full={n_full}",
        t_load.elapsed().as_secs_f64() * 1e3
    );

    let ctx: Vec<u32> = vec![1000, 2000, 3000, 4000];
    let n_vocab = m.n_vocab();
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
    assert_eq!(logits.len(), n_vocab);
    let bad = logits.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(bad, 0, "logits に非有限が {bad} 個");
    let nonfinite = norms.iter().filter(|(_, n)| !n.is_finite()).count();
    assert_eq!(nonfinite, 0, "層出口に非有限が {nonfinite} 個");
    eprintln!("ATTN_LAYER last norm={:.6}", norms[norms.len() - 1].1);
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!("ATTN_TOP5 {:?}", &idx[..5]);
    eprintln!(
        "ATTN_TOP5_LOGITS {:?}",
        idx[..5].iter().map(|&i| logits[i]).collect::<Vec<f32>>()
    );
    eprintln!("ATTN_forward_1token_ms={:.1}", dt.as_secs_f64() * 1e3);

    // 決定性。状態 (KV / ssm) を持つので、同じモデルを 2 回呼ぶと同じ入力でも
    // 位置が進んで別の値になる。組み直して同じ位置から回した値と比べる。
    let (logits2, norms2) = {
        let full2 = FullAttn::from_gguf(&f, &p, max_ctx).expect("FullAttn (2 回目)");
        let attn2 = SsmAttn::with_full(&f, &p, Box::new(full2)).expect("SsmAttn (2 回目)");
        let mut m2 = Qwen4Exp::with_attn(&f, Box::new(attn2)).expect("モデルを組めない (2 回目)");
        let mut n2: Vec<(usize, f32)> = Vec::new();
        let l2 = m2
            .forward_token_with(&ctx, |il, res| {
                let mut s = 0.0f64;
                for &x in res {
                    s += (x as f64) * (x as f64);
                }
                n2.push((il, s.sqrt() as f32));
                Ok(())
            })
            .expect("2 回目の forward");
        (l2, n2)
    };
    let diff = logits
        .iter()
        .zip(logits2.iter())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(diff, 0, "組み直した 2 回目で {diff} 個の logit がビット一致しない");
    let ndiff = norms
        .iter()
        .zip(norms2.iter())
        .filter(|(a, b)| a.1.to_bits() != b.1.to_bits())
        .count();
    assert_eq!(ndiff, 0, "組み直した 2 回目で {ndiff} 層のノルムがビット一致しない");

    // 同じモデルを続けて回すと位置が進む (状態を持つことの確認)。値は変わるのが正しい。
    let next = m.forward_token(&ctx).expect("2 トークン目の forward");
    let same = logits
        .iter()
        .zip(next.iter())
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    assert!(
        same < logits.len(),
        "位置を進めたのに全 logit が同じ (キャッシュが効いていない)"
    );
}


/// 実重み + 実 attention で貪欲に 4 トークン伸ばす。
///
/// プロンプトは `"1+1は?"` のトークン列（参照実装の `llama-tokenize` で得た
/// `[16, 10, 16, 14876, 30]`）。生成された id を参照実装の貪欲 decode と突き合わせる。
/// 文脈が 2048 token 以下なので QSA を入れなくても dense と一致する。
#[test]
#[ignore]
fn real_model_greedy_short_prompt() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let full = FullAttn::from_gguf(&f, &p, 32).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");

    let prompt: Vec<u32> = vec![16, 10, 16, 14876, 30];
    let t0 = std::time::Instant::now();
    let mut d = Qwen4ExpDecoder::new(m, DecoderConfig::new(32));
    let out = d.decode_greedy(&prompt, 4).expect("decode が失敗した");
    let dt = t0.elapsed();
    eprintln!("GREEDY_IDS {out:?}");
    eprintln!("GREEDY_tokens={} total_ms={:.0}", d.tokens().len(), dt.as_secs_f64() * 1e3);
    assert_eq!(out.len(), 4);
    assert!(d.tokens().len() == prompt.len() + 4);
}


/// 実重みで各ステップの top-5 を出す（参照実装の `llama-simple` と突き合わせるため）。
#[test]
#[ignore]
fn real_model_top5_per_step() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let full = FullAttn::from_gguf(&f, &p, 32).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");

    let prompt: Vec<u32> = vec![16, 10, 16, 14876, 30];
    let mut d = Qwen4ExpDecoder::new(m, DecoderConfig::new(32));
    let t0 = std::time::Instant::now();
    let mut logits = d.prefill(&prompt).expect("prefill");
    for step in 0..4 {
        let mut top: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let line: Vec<String> = top
            .iter()
            .take(5)
            .map(|(i, v)| format!("{i}:{v:.4}"))
            .collect();
        eprintln!("STEP {step} top5 {}", line.join(" "));
        let next = top[0].0 as u32;
        logits = d.step(next).expect("step");
    }
    eprintln!("TOTAL_ms {:.0}", t0.elapsed().as_secs_f64() * 1e3);
}


/// 実重みで 1 トークン forward の層別所要時間を出す（性能の当たりを付けるため）。
///
/// 層の出口フックの呼び出し間隔を測る。48 層ぶんの表と、合計を出す。
#[test]
#[ignore]
fn real_model_layer_timings() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let full = FullAttn::from_gguf(&f, &p, 32).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");

    // warm-up は捨てる（初回は重みのページインを含む）
    let _ = m.forward_token(&[16, 10, 16]).expect("warm-up");

    let mut marks: Vec<std::time::Instant> = Vec::with_capacity(64);
    marks.push(std::time::Instant::now());
    let t0 = std::time::Instant::now();
    let mut m2 = {
        let f2 = GgufFile::open_split(&parts).expect("再オープン");
        let p2 = Qwen4ExpParams::from_meta(&f2).expect("メタ");
        let full2 = FullAttn::from_gguf(&f2, &p2, 32).expect("FullAttn");
        let attn2 = SsmAttn::with_full(&f2, &p2, Box::new(full2)).expect("SsmAttn");
        Qwen4Exp::with_attn(&f2, Box::new(attn2)).expect("モデル")
    };
    let logits = m2
        .forward_token_with(&[16, 10, 16, 14876, 30], |_il, _res| {
            marks.push(std::time::Instant::now());
            Ok(())
        })
        .expect("forward");
    let total = t0.elapsed();
    assert_eq!(marks.len(), p.n_layer + 1);
    let mut sum = 0.0f64;
    for il in 0..p.n_layer {
        let ms = marks[il + 1].duration_since(marks[il]).as_secs_f64() * 1e3;
        sum += ms;
        eprintln!("LAYER {il:02} {ms:8.2} ms");
    }
    let tail = marks[p.n_layer].duration_since(marks[p.n_layer]).as_secs_f64();
    let _ = tail;
    eprintln!(
        "TOTAL_forward {:.0} ms (layers {:.0} ms, logits {})",
        total.as_secs_f64() * 1e3,
        sum,
        logits.len()
    );
}


/// 実重みで、プロンプト 5 トークンを 1 個ずつ食わせ、最後のトークンでの各層出口の
/// `res_hc` を出す（参照実装の `l_last-<il>` と突き合わせるため）。
///
/// 出すのは層ごとに、ストリーム 0 の先頭 3 要素と、ストリームごとの L2 ノルム。
#[test]
#[ignore]
fn real_model_dump_layers() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let full = FullAttn::from_gguf(&f, &p, 32).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");

    let prompt: Vec<u32> = vec![16, 10, 16, 14876, 30];
    let e = p.n_embd;
    let hc = p.n_hc;
    let mut last: Vec<(usize, [f32; 3], Vec<f32>)> = Vec::new();
    for i in 0..prompt.len() {
        let ctx = &prompt[..=i];
        let mut rows: Vec<(usize, [f32; 3], Vec<f32>)> = Vec::new();
        m.forward_token_with(ctx, |il, res| {
            let mut norms = Vec::with_capacity(hc);
            for s in 0..hc {
                let mut acc = 0.0f64;
                for d in 0..e {
                    let v = res[s * e + d] as f64;
                    acc += v * v;
                }
                norms.push(acc.sqrt() as f32);
            }
            rows.push((il, [res[0], res[1], res[2]], norms));
            Ok(())
        })
        .expect("forward");
        last = rows;
    }
    for (il, head, norms) in &last {
        let ns: Vec<String> = norms.iter().map(|v| format!("{v:.4}")).collect();
        eprintln!(
            "LL {il:02} s0 {:+.4} {:+.4} {:+.4} norms {}",
            head[0],
            head[1],
            head[2],
            ns.join(" ")
        );
    }
}


/// 実重みの `token_embd` の行 30 (プロンプト末尾のトークン) を出し、
/// 参照実装の `model.input_embed` と突き合わせる。
#[test]
#[ignore]
fn real_model_embedding_row() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let t = f.tensor("token_embd.weight").expect("token_embd が無い");
    eprintln!("token_embd: ty={:?} dims={:?}", t.ty, t.dims);
    let (file, off, len) = f.tensor_mapping_parts("token_embd.weight").expect("範囲");
    let map = f.mmap_handle_at(file).expect("mmap");
    let blocks = denno_engine::quant::WeightBlocks::mapped(map, off, len);
    let lin = GgufLinear::new(t.ty, t.dims[0] as usize, t.dims[1] as usize, blocks).expect("GgufLinear");
    let row = lin.dequant_row(30).expect("dequant_row");
    let mut acc = 0.0f64;
    for v in &row {
        acc += (*v as f64) * (*v as f64);
    }
    eprintln!("EMB30 first3 {:?} norm {:.6}", &row[..3], acc.sqrt());
}


/// 実重みで、指定層の中間値を出して参照実装のノードと突き合わせる。
///
/// `DENNO_DBG_LAYER` (既定 0) の層について、最後のトークンでの中間値の
/// 先頭 3 要素と L2 ノルムを出す。
#[test]
#[ignore]
fn real_model_debug_layer() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let layer: usize = std::env::var("DENNO_DBG_LAYER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let full = FullAttn::from_gguf(&f, &p, 32).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");

    let prompt: Vec<u32> = vec![16, 10, 16, 14876, 30];
    let mut sink = |name: &str, vals: &[f32]| {
        let mut acc = 0.0f64;
        for v in vals {
            acc += (*v as f64) * (*v as f64);
        }
        let n = vals.len().min(3);
        let head: Vec<String> = vals[..n].iter().map(|v| format!("{v:+.5}")).collect();
        eprintln!(
            "DBG L{layer:02} {name:16} n={:5} norm={:.5} head {}",
            vals.len(),
            acc.sqrt(),
            head.join(" ")
        );
    };
    m.debug_layer(&prompt, layer, &mut sink).expect("debug_layer");
}


/// 層 0 (ssm) の切り分け: 参照実装の `linear_attn_qkv_mixed` / `linear_attn_out`
/// と、こちらの qkv 投影・層出力をトークンごとに並べる。
#[test]
#[ignore]
fn real_model_ssm_stage0() {
    let parts = match parts() {
        Some(p) => p,
        None => {
            eprintln!("DENNO_Q4_DIR が無いので skip");
            return;
        }
    };
    let f = GgufFile::open_split(&parts).expect("分割 GGUF を開けない");
    let p = Qwen4ExpParams::from_meta(&f).expect("メタデータを読めない");
    let full = FullAttn::from_gguf(&f, &p, 32).expect("FullAttn");
    let attn = SsmAttn::with_full(&f, &p, Box::new(full)).expect("SsmAttn");
    let mut m = Qwen4Exp::with_attn(&f, Box::new(attn)).expect("モデルを組めない");

    let prompt: Vec<u32> = vec![16, 10, 16, 14876, 30];

    // 各 prefix について層 0 の入口 (hc_mixed) を取り出す。
    let mut mixed: Vec<Vec<f32>> = Vec::new();
    for i in 0..prompt.len() {
        let mut got: Option<Vec<f32>> = None;
        let mut sink = |name: &str, vals: &[f32]| {
            if name == "hc_mixed_attn" {
                got = Some(vals.to_vec());
            }
        };
        m.debug_layer(&prompt[..=i], 0, &mut sink).expect("debug_layer");
        mixed.push(got.expect("hc_mixed_attn が取れない"));
    }

    // まっさらな ssm 層に、その hc_mixed を順に食わせる。
    let mut fresh = SsmAttn::with_full(&f, &p, Box::new(FullAttn::from_gguf(&f, &p, 32).unwrap()))
        .expect("SsmAttn (fresh)");
    let layer = fresh.ssm_layer(0).expect("層 0 は ssm のはず");
    let (qe, ke, ve) = layer.qkv_split();
    eprintln!("qkv split q={qe} k={ke} v={ve}");
    for (i, x) in mixed.iter().enumerate() {
        let qkv = layer.project_qkv(x).expect("project_qkv");
        let head: Vec<String> = qkv[..3].iter().map(|v| format!("{v:+.5}")).collect();
        eprintln!("TOK {i} qkv {}", head.join(" "));
    }
    for (i, x) in mixed.iter().enumerate() {
        let out = match fresh.forward(0, x) {
            Ok(o) => o,
            Err(e) => {
                eprintln!("TOK {i} ssm err {e:?}");
                continue;
            }
        };
        let mut acc = 0.0f64;
        for v in &out {
            acc += (*v as f64) * (*v as f64);
        }
        let head: Vec<String> = out[..3].iter().map(|v| format!("{v:+.5}")).collect();
        eprintln!("TOK {i} out n={} norm={:.5} head {}", out.len(), acc.sqrt(), head.join(" "));
    }
}
