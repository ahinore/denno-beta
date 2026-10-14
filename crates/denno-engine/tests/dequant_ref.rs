//! kquant dequant の llama.cpp ビット一致検証 (P19)。
//!
//! `ref/dequant_ref.c` (ggml-quants.c を直接リンク) が生成した擬似ランダム
//! ブロック列とその f32 出力を突き合わせる。**1 bit も違わないこと**が契約。
//!
//! 実行: fixture を生成してから
//!   DEQUANT_REF=<path> cargo test -p denno-engine --release --lib dequant_ref -- --ignored --nocapture

use std::env;

use denno_engine::kquant::{dequant_row, GgmlType};

fn type_from_name(name: &str) -> GgmlType {
    match name {
        "Q8_0" => GgmlType::Q8_0,
        "Q4_0" => GgmlType::Q4_0,
        "Q4_1" => GgmlType::Q4_1,
        "Q5_0" => GgmlType::Q5_0,
        "Q5_1" => GgmlType::Q5_1,
        "Q2_K" => GgmlType::Q2K,
        "Q3_K" => GgmlType::Q3K,
        "Q4_K" => GgmlType::Q4K,
        "Q5_K" => GgmlType::Q5K,
        "Q6_K" => GgmlType::Q6K,
        "IQ2_XXS" => GgmlType::IQ2Xxs,
        "IQ2_XS" => GgmlType::IQ2Xs,
        "IQ2_S" => GgmlType::IQ2S,
        "IQ3_XXS" => GgmlType::IQ3Xxs,
        "IQ3_S" => GgmlType::IQ3S,
        "IQ1_S" => GgmlType::IQ1S,
        "IQ4_NL" => GgmlType::IQ4Nl,
        "IQ4_XS" => GgmlType::IQ4Xs,
        other => panic!("未知のセクション {other}"),
    }
}

#[test]
#[ignore]
fn kquant_dequant_bitexact_vs_llama_cpp() {
    let path = env::var("DEQUANT_REF").unwrap_or_else(|_| "tests/data/dequant_ref.bin".into());
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("リファレンス {path} を読めない: {e}"));
    let mut p = 0usize;
    let rd_u32 = |p: &mut usize| {
        let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let n_secs = rd_u32(&mut p) as usize;
    assert!(n_secs >= 15, "セクション数 {n_secs} が少なすぎる");
    let mut checked = 0usize;
    for s in 0..n_secs {
        let nl = rd_u32(&mut p) as usize;
        let name = String::from_utf8(data[p..p + nl].to_vec()).unwrap();
        p += nl;
        let nb = rd_u32(&mut p) as usize;
        let raw = data[p..p + nb].to_vec();
        p += nb;
        let nf = rd_u32(&mut p) as usize;
        let mut want = vec![0f32; nf];
        for (i, w) in want.iter_mut().enumerate() {
            *w = f32::from_le_bytes(data[p..p + 4].try_into().unwrap());
            p += 4;
        }
        let ty = type_from_name(&name);
        let got = dequant_row(ty, &raw, nf);
        let mut bad = None;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if g.to_bits() != w.to_bits() {
                bad = Some((i, *g, *w));
                break;
            }
        }
        match bad {
            None => println!("{name}: BIT-EXACT ({nf} elems)"),
            Some((i, g, w)) => panic!(
                "{name}: 不一致 at {i}: got={g:.6}({:#x}) want={w:.6}({:#x})",
                g.to_bits(),
                w.to_bits()
            ),
        }
        checked += 1;
    }
    assert_eq!(p, data.len(), "余分なデータがある");
    assert!(checked >= 15);
}
