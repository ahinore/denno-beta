//! 27B/9B 出力品質調査用の logit/top-k プローブ (実 GGUF・ignored・QWEN35_27B_DIR で実行)。
//!
//! 目的: 27B の出力破綻 (早期 `<|im_end|>` → テンプレート断片ループ) の切り分けを続けられるよう、
//! chat プロンプト 1 本の prefill logits top-k と greedy 各ステップの argmax/top-k を
//! トークン単位で出力する。llama.cpp リファレンス (同一 GGUF) との差分比較に使う
//! (第13セッション: llama.cpp は同 GGUF で正答 → 本エンジン固有の差分あり)。
//!
//! 環境変数: QWEN35_27B_DIR (config.json + *.gguf を含むディレクトリ・未設定で graceful skip)。
//! 9B モデルを指定すれば 9B 経路の検証にも流用できる (第16セッション)。
//! 実行: `cargo test -p denno-engine --test qwen35_27b_logit_probe -- --ignored --nocapture`
//!
//! 既知の所見 (第12セッション): "What is the capital of France?" は greedy step 7 で
//! `<|im_end|>` (logit 19.25・runner-up 16.5) を出して思考を放棄する。prefill 自体は決定論的。
//! 第16セッション: 原因は GDN の k/q ヘッド対応 (blocked → タイル repeat に修正)。
//! 修正後は 27B も greedy で正答する (本テストは回帰監視として恒久化)。

use denno_engine::engine::ChatMessage;
use denno_engine::model::{load_model, Model};
use denno_engine::quant::QuantMode;
use denno_engine::tokenizer::DennoTokenizer;

fn topk(v: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut s: Vec<(usize, f32)> = v.iter().enumerate().map(|(i, x)| (i, *x)).collect();
    s.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    s.truncate(k);
    s
}

#[test]
#[ignore]
fn qwen35_27b_logit_probe() {
    let Ok(dir) = std::env::var("QWEN35_27B_DIR") else {
        eprintln!("skip: QWEN35_27B_DIR 未設定");
        return;
    };

    let mut model = load_model(std::path::Path::new(&dir), QuantMode::Int8)
        .expect("モデルロード失敗");
    let tok = DennoTokenizer::load(std::path::Path::new(&dir)).expect("tokenizer ロード失敗");

    let prompt_text = std::env::var("QWEN35_PROBE_PROMPT")
        .unwrap_or_else(|_| "What is the capital of France?".to_string());
    let prompt = tok
        .apply_chat_template(
            None,
            &[ChatMessage { role: "user".into(), content: prompt_text }],
            None,
            None,
        )
        .expect("template 失敗");
    eprintln!("PROMPT: {prompt:?}");
    let ids = tok.encode(&prompt).expect("encode 失敗");
    eprintln!("PROMPT_IDS: {ids:?} ({} tokens)", ids.len());

    // prefill
    let mut cache = model.new_cache();
    let logits = model.forward(&ids, &mut cache).expect("prefill 失敗");
    let mx = logits.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    let mean = logits.iter().sum::<f32>() / logits.len() as f32;
    eprintln!("PREFILL_LOGITS: maxabs={mx:.4} mean={mean:.4}");
    for (i, v) in topk(&logits, 10) {
        let piece = tok.decode(&[i as u32]).unwrap_or_default();
        eprintln!("  top[{i}] = {v:.4} {piece:?}");
    }

    // greedy 24 step
    let mut gen: Vec<u32> = Vec::new();
    let mut prev = logits.clone();
    for step in 0..24 {
        let (bi, bv) = prev
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, v)| (i as u32, *v))
            .unwrap();
        let top3 = topk(&prev, 3)
            .iter()
            .map(|(i, v)| (tok.decode(&[*i as u32]).unwrap_or_default(), *v))
            .collect::<Vec<_>>();
        eprintln!("step {step}: argmax={bi} ({bv:.4}) top3={top3:?}");
        gen.push(bi);
        if bi == 248044 || bi == 151645 {
            eprintln!("  (EOS)");
            break;
        }
        let l = model.forward(&[bi], &mut cache).expect("decode 失敗");
        prev = l;
    }
    let text = tok.decode(&gen).unwrap_or_default();
    eprintln!("GENERATED_TEXT: {text:?}");

    // 決定論確認: 同一 prefill をやり直して bit 一致 (層別診断時は [LCHK] 出力が
    // 2 パスにならないようスキップ)
    if std::env::var("DENNO_DEBUG_LAYER_CHKSUM").ok().as_deref() == Some("1") {
        eprintln!("DETERMINISM: skipped (layer checksum diag)");
        return;
    }
    let mut cache2 = model.new_cache();
    let logits2 = model.forward(&ids, &mut cache2).expect("prefill (2) 失敗");
    let same = logits
        .iter()
        .zip(&logits2)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    eprintln!("DETERMINISM: prefill bit-identical = {same}");
}
