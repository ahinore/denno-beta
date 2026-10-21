//! kquant dequant の llama.cpp ビット一致検証 (P19 / K2)。
//!
//! `ref/dequant_ref.c` (ggml-quants.c を直接リンク) が生成した擬似ランダム
//! ブロック列とその f32 出力を突き合わせる。**1 bit も違わないこと**が契約。
//! 既定の `cargo test -p denno-engine --release --test dequant_ref` で走る
//! (fixture は `tests/data/dequant_ref.bin` に commit 済み・21 セクション)。
//!
//! 再生成するときだけ `DEQUANT_REF=<path>` で別の fixture を指せる。
//! 生成手順は `ref/README.md` を参照。

use std::env;

use denno_engine::kquant::{dequant_row, GgmlType};

/// fixture が持つべきセクション数 (18 型 + K2 の TQ1_0 / IQ1_M + K3 の Q4_0_EDGE)。
const N_SECTIONS: usize = 21;
/// このテストが必ず通す型。fixture が古くて欠けていたら明示的に落とす。
/// Q4_0 は 27B 本番が実際に serving している形式 (ggml type 2)。
const REQUIRED: [&str; 4] = ["Q4_0", "Q4_0_EDGE", "TQ1_0", "IQ1_M"];
const DEFAULT_FIXTURE: &str = "tests/data/dequant_ref.bin";

fn type_from_name(name: &str) -> GgmlType {
    match name {
        "Q8_0" => GgmlType::Q8_0,
        "Q4_0" => GgmlType::Q4_0,
        // K3: Q4_0 のエッジブロック列も同じ型 (type 2) のデコーダで読む
        "Q4_0_EDGE" => GgmlType::Q4_0,
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
        "IQ1_M" => GgmlType::Iq1M,
        "TQ1_0" => GgmlType::Tq1_0,
        other => panic!("未知のセクション {other} (fixture とテストの型名がずれている)"),
    }
}

#[test]
fn kquant_dequant_bitexact_vs_llama_cpp() {
    let path = env::var("DEQUANT_REF").unwrap_or_else(|_| DEFAULT_FIXTURE.into());
    let data = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "リファレンス {path} を読めない: {e}\n\
             (cwd={:?}) fixture は commit 済みのはず。再生成は ref/README.md の手順で行い、\n\
             `DEQUANT_REF=<生成した .bin>` でこのテストを再実行してから差し替える。",
            env::current_dir().ok()
        )
    });
    assert!(
        data.len() >= 8,
        "リファレンス {path} が小さすぎる ({} bytes)。壊れた fixture か、別ファイルを指している",
        data.len()
    );
    let mut p = 0usize;
    let rd_u32 = |p: &mut usize| {
        assert!(
            *p + 4 <= data.len(),
            "リファレンス {path} が途中で切れている (offset {p})"
        );
        let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let n_secs = rd_u32(&mut p) as usize;
    assert_eq!(
        n_secs, N_SECTIONS,
        "リファレンス {path} のセクション数が {n_secs} (期待 {N_SECTIONS})。\
         古い fixture を読んでいる可能性がある — TQ1_0 / IQ1_M を含む 20 型に再生成する手順は ref/README.md"
    );
    let mut checked = 0usize;
    let mut names: Vec<String> = Vec::new();
    for _s in 0..n_secs {
        let nl = rd_u32(&mut p) as usize;
        assert!(p + nl <= data.len(), "リファレンス {path}: 型名が切れている");
        let name = String::from_utf8(data[p..p + nl].to_vec()).unwrap();
        p += nl;
        let nb = rd_u32(&mut p) as usize;
        assert!(p + nb <= data.len(), "リファレンス {path}: {name} の生バイトが切れている");
        let raw = data[p..p + nb].to_vec();
        p += nb;
        let nf = rd_u32(&mut p) as usize;
        assert!(
            p + 4 * nf <= data.len(),
            "リファレンス {path}: {name} の f32 出力が切れている"
        );
        let mut want = vec![0f32; nf];
        for w in want.iter_mut() {
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
        names.push(name);
        checked += 1;
    }
    assert_eq!(p, data.len(), "余分なデータがある ({p} / {} bytes)", data.len());
    assert_eq!(checked, N_SECTIONS);
    for req in REQUIRED {
        assert!(
            names.iter().any(|n| n == req),
            "fixture {path} に {req} が無い (セクション: {names:?})。\
             K2 で追加した型を検証できないので、ref/README.md の手順で再生成する"
        );
    }
}

// ---------------------------------------------------------------------------
// Q4_0 (ggml type 2): 27B 本番が serving している形式の参照突き合わせ (Task K3)
// ---------------------------------------------------------------------------

/// fixture から 1 セクションだけ取り出す。無い / 壊れていれば理由付きで落とす。
fn fixture_section(want: &str) -> (Vec<u8>, Vec<f32>) {
    let path = env::var("DEQUANT_REF").unwrap_or_else(|_| DEFAULT_FIXTURE.into());
    let data = std::fs::read(&path)
        .unwrap_or_else(|e| panic!("リファレンス {path} を読めない: {e} (再生成は ref/README.md)"));
    let mut p = 0usize;
    let rd_u32 = |p: &mut usize| -> u32 {
        assert!(*p + 4 <= data.len(), "リファレンス {path} が途中で切れている");
        let v = u32::from_le_bytes(data[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let n_secs = rd_u32(&mut p) as usize;
    for _ in 0..n_secs {
        let nl = rd_u32(&mut p) as usize;
        assert!(p + nl <= data.len(), "リファレンス {path}: 型名が切れている");
        let name = String::from_utf8(data[p..p + nl].to_vec()).unwrap();
        p += nl;
        let nb = rd_u32(&mut p) as usize;
        assert!(p + nb <= data.len(), "リファレンス {path}: {name} の生バイトが切れている");
        let raw = data[p..p + nb].to_vec();
        p += nb;
        let nf = rd_u32(&mut p) as usize;
        assert!(p + 4 * nf <= data.len(), "リファレンス {path}: {name} の f32 出力が切れている");
        let mut y = vec![0f32; nf];
        for w in y.iter_mut() {
            *w = f32::from_le_bytes(data[p..p + 4].try_into().unwrap());
            p += 4;
        }
        if name == want {
            return (raw, y);
        }
    }
    panic!("fixture {path} に {want} セクションが無い (ref/README.md の手順で再生成する)");
}

/// Q4_0 (ggml type 2) のデ量子化を参照実装と **f32 バイト単位で** 突き合わせる。
///
/// 27B 本番の重みは Q4_0 なので、実際に serving している形式を llama.cpp の
/// `dequantize_row_q4_0` と比較するのがこのテストの目的。比較対象は
/// デコード後の f32 (`to_bits`) で、生バイトの一致ではない。
///
/// セクション `Q4_0_EDGE` は `ref/dequant_ref.c` の `emit_q4_0_edge` が
/// 6 ブロックを明示的に作ったもの (期待値は参照実装の出力そのもの):
/// 負の d / 全ゼロのニブル / 最大ニブル / 負の d + 最大ニブル / 上下で別パターン /
/// d = +0.0。擬似乱数の `Q4_0` セクションだけでは踏めない符号と +8 オフセットを
/// ここで確実に通す。
#[test]
fn q4_0_dequant_bitexact_vs_llama_cpp_edge_blocks() {
    let (raw, want) = fixture_section("Q4_0_EDGE");
    assert_eq!(raw.len(), 6 * 18, "Q4_0_EDGE は 6 ブロック (18 バイト/ブロック)");
    assert_eq!(want.len(), 6 * 32);

    // 前提: セクションが「負の d」「全ゼロのニブル」「最大ニブル」を実際に含むこと。
    // 含まなければこのテストは符号と +8 オフセットを踏めていない (fixture の退行)。
    let blocks: Vec<&[u8]> = raw.chunks_exact(18).collect();
    assert!(
        blocks.iter().any(|b| b[1] & 0x80 != 0),
        "負の d のブロックが無い"
    );
    assert!(
        blocks.iter().any(|b| b[2..].iter().all(|q| *q == 0x00)),
        "全ゼロのニブルのブロックが無い"
    );
    assert!(
        blocks.iter().any(|b| b[2..].iter().all(|q| *q == 0xff)),
        "最大ニブルのブロックが無い"
    );

    let got = dequant_row(GgmlType::Q4_0, &raw, want.len());
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "Q4_0_EDGE[{i}]: got {g:.6}({:#x}) want {w:.6}({:#x})",
            g.to_bits(),
            w.to_bits()
        );
    }

    // +8 オフセットの向き: 全ゼロのニブルは q = -8、最大ニブルは q = +7。
    // この 2 ブロックの d は 0.5 (f16 0x3800) で厳密なので -4.0 / +3.5 も厳密。
    assert!(
        want[32..64].iter().all(|v| v.to_bits() == (-4.0f32).to_bits()),
        "全ゼロのニブルが -8*d になっていない"
    );
    assert!(
        want[64..96].iter().all(|v| v.to_bits() == 3.5f32.to_bits()),
        "最大ニブルが +7*d になっていない"
    );
    // 符号付きゼロ: d = +0.0 のブロックの上位ニブル (q = -3) は -0.0 になる。
    assert!(
        want[5 * 32 + 16..6 * 32]
            .iter()
            .all(|v| v.to_bits() == (-0.0f32).to_bits()),
        "d = +0.0 で -0.0 になっていない"
    );
}

// ---------------------------------------------------------------------------
// TQ1_0 / IQ1_M: 自己完結の参照テスト (Task K1)
//
// ブロックは「形式の符号化規則」から組み立てたもの (TQ1_0 は三値パターン、
// IQ1_M は選んだ qs/qh/scales)。期待値は定義から独立に計算し、同じブロックを
// llama.cpp の dequantize_row_tq1_0 / dequantize_row_iq1_m に通して 256/256
// 一致することを確認した (2026-10-21、ローカルの C リファレンスで実行)。
// よって下の表は「llama.cpp の dequantize と同値になる手計算ベクトル」である。
// 形式は llama.cpp ggml-common.h / ggml-quants.c (MIT) 準拠・実装は独自。
// ---------------------------------------------------------------------------

/// TQ1_0: 三値パターンを 5 桁/バイトで詰めた 1 ブロック (d = f16 0x3A00 = 0.75)。
/// 期待値はパターン * 0.75 そのもの (手で追える)。
#[test]
fn tq1_0_matches_hand_computed_block() {
    let raw: [u8; 54] = [
        0x0f, 0x60, 0xc8, 0xc2, 0xdb, 0x60, 0xa0, 0xef, 0x87, 0x79, 0x25, 0xad, 0xe0, 0xd6, 0x82,
        0x2f, 0x51, 0x1b, 0xc8, 0x87, 0x1b, 0xef, 0xe7, 0x2a, 0x57, 0xad, 0x8c, 0x60, 0x19, 0x25,
        0x06, 0x40, 0xc3, 0x21, 0x78, 0x74, 0x2a, 0x3e, 0xfa, 0xf2, 0x97, 0x8b, 0xbb, 0x68, 0x68,
        0x40, 0xcb, 0x52, 0x53, 0xe1, 0x17, 0xfa, 0x00, 0x3a,
    ];
    #[rustfmt::skip]
    let want: [f32; 256] = [
        -0.75, 0.0, 0.75, 0.75, 0.75, 0.0, 0.0, 0.75,
        0.0, 0.0, -0.75, 0.75, 0.75, 0.75, 0.0, -0.75,
        -0.75, -0.75, 0.75, 0.0, -0.75, 0.75, 0.75, -0.75,
        0.0, 0.75, 0.0, 0.0, -0.75, -0.75, -0.75, -0.75,
        -0.75, -0.75, 0.0, -0.75, 0.0, -0.75, 0.75, 0.75,
        0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0,
        0.75, -0.75, 0.0, 0.0, -0.75, 0.75, 0.75, 0.0,
        -0.75, -0.75, 0.0, -0.75, -0.75, 0.0, -0.75, 0.75,
        0.0, 0.0, -0.75, 0.75, 0.75, 0.0, 0.0, 0.0,
        0.75, -0.75, -0.75, -0.75, 0.75, 0.0, 0.0, 0.0,
        0.75, 0.75, -0.75, 0.75, 0.75, 0.0, -0.75, 0.0,
        -0.75, -0.75, 0.75, 0.0, 0.75, -0.75, -0.75, -0.75,
        0.0, -0.75, -0.75, 0.0, -0.75, -0.75, 0.75, -0.75,
        -0.75, 0.75, 0.75, -0.75, 0.0, 0.0, 0.75, 0.75,
        0.0, 0.75, -0.75, -0.75, 0.75, -0.75, 0.0, 0.0,
        -0.75, -0.75, 0.75, -0.75, 0.0, 0.75, 0.0, 0.75,
        0.75, 0.0, -0.75, 0.0, -0.75, 0.0, 0.0, 0.0,
        0.75, -0.75, 0.75, 0.75, 0.75, 0.75, -0.75, 0.75,
        0.0, 0.0, -0.75, 0.75, 0.0, 0.0, -0.75, -0.75,
        0.0, 0.75, -0.75, 0.0, 0.75, 0.75, 0.75, -0.75,
        0.75, -0.75, 0.0, 0.0, -0.75, -0.75, 0.75, 0.75,
        0.0, 0.0, 0.75, 0.0, 0.0, -0.75, 0.75, -0.75,
        -0.75, 0.0, 0.0, 0.0, 0.0, 0.75, 0.75, 0.75,
        0.75, 0.0, -0.75, -0.75, -0.75, 0.75, 0.0, 0.75,
        0.75, -0.75, -0.75, -0.75, 0.0, -0.75, 0.75, 0.0,
        -0.75, 0.75, 0.0, 0.0, 0.0, -0.75, -0.75, 0.75,
        0.0, 0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 0.0,
        0.75, 0.0, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0,
        0.75, 0.0, 0.75, 0.75, -0.75, 0.0, -0.75, 0.0,
        0.75, 0.75, -0.75, 0.75, 0.75, -0.75, -0.75, 0.75,
        -0.75, 0.75, -0.75, 0.75, 0.75, 0.0, -0.75, 0.75,
        0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.0, 0.0,
    ];
    let got = dequant_row(GgmlType::Tq1_0, &raw, 256);
    assert_eq!(got.len(), 256);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "TQ1_0[{i}]: got {g} want {w}");
    }
}

/// IQ1_M: 選んだ qs/qh/scales の 1 ブロック。d は scales の上位ニブルに分散
/// していて、このブロックでは 0x02c0 (= 4.1961669921875e-05)。
#[test]
fn iq1_m_matches_hand_computed_block() {
    let raw: [u8; 56] = [
        0x00, 0x07, 0x0e, 0x15, 0x1d, 0x24, 0x2b, 0x32, 0x3a, 0x41, 0x48, 0x4f, 0x57, 0x5e, 0x65,
        0x6c, 0x74, 0x7b, 0x82, 0x89, 0x91, 0x98, 0x9f, 0xa6, 0xae, 0xb5, 0xbc, 0xc3, 0xcb, 0xd2,
        0xd9, 0xe0, 0x08, 0x91, 0x2a, 0xb3, 0x4c, 0xd5, 0x6e, 0xf7, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
        0xdd, 0xee, 0xff, 0x23, 0x01, 0x56, 0xc4, 0x89, 0x27, 0xbc, 0x0a,
    ];
    #[rustfmt::skip]
    let want: [f32; 256] = [
        -0.000330448151, -0.000330448151, -0.000330448151, -0.000330448151, -0.000330448151, -0.000330448151, -0.000330448151, -0.000330448151,
        -0.000257015228, -0.000257015228, 0.000330448151, -0.000257015228, -0.000257015228, -0.000257015228, -0.000257015228, -0.000257015228,
        4.72068787e-05, -0.000330448151, 0.000424861908, -0.000330448151, -0.000330448151, 4.72068787e-05, 4.72068787e-05, -0.000330448151,
        -4.72068787e-05, -0.000424861908, -4.72068787e-05, -4.72068787e-05, -0.000424861908, -4.72068787e-05, -4.72068787e-05, -0.000424861908,
        -0.000424861908, 0.000330448151, 0.000330448151, -4.72068787e-05, -4.72068787e-05, -4.72068787e-05, 0.000330448151, -0.000424861908,
        4.72068787e-05, -0.000330448151, 0.000424861908, 0.000424861908, 4.72068787e-05, 4.72068787e-05, 0.000424861908, -0.000330448151,
        5.24520874e-06, -3.67164612e-05, -3.67164612e-05, -3.67164612e-05, 5.24520874e-06, 4.72068787e-05, -3.67164612e-05, 5.24520874e-06,
        -4.72068787e-05, 3.67164612e-05, -5.24520874e-06, -4.72068787e-05, -5.24520874e-06, 3.67164612e-05, -4.72068787e-05, -5.24520874e-06,
        -0.000613689423, 0.000477313995, -0.000613689423, -6.81877136e-05, 0.000477313995, -6.81877136e-05, -6.81877136e-05, -6.81877136e-05,
        6.81877136e-05, 6.81877136e-05, 6.81877136e-05, 6.81877136e-05, 0.000613689423, 6.81877136e-05, 6.81877136e-05, 6.81877136e-05,
        0.000236034393, 0.000236034393, 2.62260437e-05, 2.62260437e-05, 0.000236034393, 2.62260437e-05, 0.000236034393, 2.62260437e-05,
        0.000183582306, 0.000183582306, -0.000236034393, 0.000183582306, 0.000183582306, -2.62260437e-05, 0.000183582306, -2.62260437e-05,
        0.000110149384, -0.000141620636, 0.000110149384, 0.000110149384, 0.000110149384, 0.000110149384, -0.000141620636, 0.000110149384,
        1.57356262e-05, 0.000141620636, 1.57356262e-05, -0.000110149384, -0.000110149384, -0.000110149384, 1.57356262e-05, 0.000141620636,
        2.62260437e-05, 0.000236034393, 2.62260437e-05, -0.000183582306, 2.62260437e-05, -0.000183582306, 0.000236034393, 0.000236034393,
        0.000183582306, 0.000183582306, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, -0.000236034393, 0.000183582306, 0.000183582306,
        0.000110149384, -1.57356262e-05, 0.000110149384, 0.000110149384, -1.57356262e-05, -1.57356262e-05, -0.000141620636, -0.000141620636,
        0.000110149384, -0.000141620636, -1.57356262e-05, -1.57356262e-05, 0.000110149384, -1.57356262e-05, -0.000141620636, -0.000141620636,
        -0.000141620636, -0.000141620636, 0.000110149384, -1.57356262e-05, -0.000141620636, 0.000110149384, -1.57356262e-05, -0.000141620636,
        -0.000141620636, -1.57356262e-05, -1.57356262e-05, 0.000110149384, -0.000141620636, 0.000110149384, -1.57356262e-05, -0.000141620636,
        -6.81877136e-05, 0.000477313995, 0.000477313995, -0.000613689423, -6.81877136e-05, -0.000613689423, -0.000613689423, -6.81877136e-05,
        0.000477313995, -6.81877136e-05, -6.81877136e-05, -6.81877136e-05, -6.81877136e-05, -0.000613689423, -0.000613689423, -6.81877136e-05,
        -0.000330448151, -0.000330448151, 0.000257015228, 0.000257015228, -3.67164612e-05, -0.000330448151, -3.67164612e-05, -3.67164612e-05,
        0.000257015228, 0.000257015228, -3.67164612e-05, -0.000330448151, 0.000257015228, -0.000330448151, -3.67164612e-05, -3.67164612e-05,
        -4.72068787e-05, -4.72068787e-05, 0.000330448151, -4.72068787e-05, 0.000330448151, 0.000330448151, -4.72068787e-05, -4.72068787e-05,
        0.000330448151, -4.72068787e-05, -4.72068787e-05, 0.000330448151, 0.000330448151, 0.000330448151, -4.72068787e-05, -4.72068787e-05,
        0.000550746918, -7.86781311e-05, 0.000550746918, -7.86781311e-05, -7.86781311e-05, -0.00070810318, -0.00070810318, 0.000550746918,
        -7.86781311e-05, -7.86781311e-05, 0.000550746918, 0.000550746918, -7.86781311e-05, -0.00070810318, -0.00070810318, 0.000550746918,
        0.000183582306, -2.62260437e-05, -0.000236034393, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, 0.000183582306,
        0.000183582306, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, -2.62260437e-05, 0.000183582306,
        -5.76972961e-05, -5.76972961e-05, -5.76972961e-05, 0.000403881073, -0.000519275665, 0.000403881073, 0.000403881073, 0.000403881073,
        -5.76972961e-05, 0.000403881073, -5.76972961e-05, -0.000519275665, -5.76972961e-05, 0.000403881073, 0.000403881073, 0.000403881073,
    ];
    let got = dequant_row(GgmlType::Iq1M, &raw, 256);
    assert_eq!(got.len(), 256);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "IQ1_M[{i}]: got {g} want {w}");
    }
}

/// 型 ID とブロック寸法。ID は ggml.h の enum と gguf-py constants.py で
/// 確認した値 (TQ1_0 = 34、IQ1_M = 29)。type 24 は I8 なので未対応のまま。
#[test]
fn tq1_0_and_iq1_m_ids_and_block_sizes() {
    assert_eq!(GgmlType::from_id(34), Some(GgmlType::Tq1_0));
    assert_eq!(GgmlType::from_id(29), Some(GgmlType::Iq1M));
    assert_eq!(GgmlType::from_id(24), None, "type 24 は I8 (IQ1_M ではない)");

    assert_eq!(GgmlType::Tq1_0.blck_size(), 256);
    assert_eq!(GgmlType::Tq1_0.type_size(), 54); // qs[48] + qh[4] + d(f16)
    assert_eq!(GgmlType::Iq1M.blck_size(), 256);
    assert_eq!(GgmlType::Iq1M.type_size(), 56); // qs[32] + qh[16] + scales[8]

    // ブロック長とバイト数の整合 (n 要素 = n / 256 ブロック分のバイト列)
    for ty in [GgmlType::Tq1_0, GgmlType::Iq1M] {
        let ts = ty.type_size();
        let raw = vec![0u8; 3 * ts];
        assert_eq!(dequant_row(ty, &raw, 3 * 256).len(), 3 * 256, "{ty:?}");
        assert_eq!((3 * 256 / ty.blck_size()) * ts, raw.len(), "{ty:?}");
    }
}

/// 全ゼロ・スケール 0・NaN/Inf のブロックで panic しないこと。
#[test]
fn degenerate_blocks_do_not_panic() {
    for ty in [GgmlType::Tq1_0, GgmlType::Iq1M] {
        let ts = ty.type_size();
        let zeros = vec![0u8; 2 * ts];
        let out = dequant_row(ty, &zeros, 2 * 256);
        assert_eq!(out.len(), 2 * 256);
        assert!(out.iter().all(|v| *v == 0.0), "{ty:?}: 全ゼロは 0 になる");
    }

    // f16 の NaN / +Inf / -Inf / 最小非正規 (最後の 1 つだけは有限値のまま)
    for (bits, nonfinite) in [
        (0x7e00u16, true),
        (0x7c00, true),
        (0xfc00, true),
        (0x0001, false),
    ] {
        // TQ1_0: d は末尾 2 バイト
        let mut b = vec![0x55u8; 54];
        b[52..54].copy_from_slice(&bits.to_le_bytes());
        let out = dequant_row(GgmlType::Tq1_0, &b, 256);
        assert_eq!(out.len(), 256);
        if nonfinite {
            assert!(out.iter().all(|v| !v.is_finite()), "TQ1_0 d=0x{bits:04x}");
        } else {
            assert!(out.iter().all(|v| v.is_finite()), "TQ1_0 d=0x{bits:04x}");
            assert!(out.iter().all(|v| v.abs() < 1e-6), "TQ1_0 d=0x{bits:04x}");
        }

        // IQ1_M: d は 4 つの scale ワードの上位ニブルに分散している
        let mut b = vec![0x55u8; 56];
        let low = 0x0123u16;
        let words = [
            ((bits & 0x000f) << 12) | low,
            ((bits & 0x00f0) << 8) | low,
            ((bits & 0x0f00) << 4) | low,
            (bits & 0xf000) | low,
        ];
        for (i, w) in words.iter().enumerate() {
            b[48 + 2 * i..50 + 2 * i].copy_from_slice(&w.to_le_bytes());
        }
        let out = dequant_row(GgmlType::Iq1M, &b, 256);
        assert_eq!(out.len(), 256);
        if nonfinite {
            assert!(out.iter().all(|v| !v.is_finite()), "IQ1_M d=0x{bits:04x}");
        } else {
            assert!(out.iter().all(|v| v.is_finite()), "IQ1_M d=0x{bits:04x}");
        }
    }
}

/// 未対応 ID は従来どおり None (TQ2_0 / NVFP4 / Q1_0 / Q2_0 も未実装)。
#[test]
fn unsupported_ids_stay_none() {
    for id in [
        4u32, 5, 9, 15, 24, 25, 26, 27, 28, 31, 32, 33, 35, 36, 37, 38, 40, 41, 42, 999,
    ] {
        assert_eq!(GgmlType::from_id(id), None, "id {id} は未対応のはず");
    }
}
