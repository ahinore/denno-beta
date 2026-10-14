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
    let dir = std::env::var("QWEN35_DIR").expect("QWEN35_DIR を設定してください");
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
    let dir = std::env::var("QWEN35_DIR").expect("QWEN35_DIR を設定してください");
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
