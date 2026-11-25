//! task 1039: バイナリワイヤ <-> NDJSON の等価往復と大規模ストリーム復帰テスト。
//!
//! 検証する不変条件:
//!
//! - 全 [`Frame`] variant が `Binary -> Frame -> NDJSON -> Frame -> Binary` を
//!   通り抜けても **バイト列が完全一致** する (encode は決定的)。
//! - 5,000 枚超の混在ストリームに空行・切り詰め JSON 行・不正 UTF-8 行を
//!   挟んでも停止せず、有効フレームを全て回収し、壊れた行を **1 始まりの行番号**
//!   付きで正確に報告する。
//! - 時刻の極値 (0 / u64::MAX) と Unicode 文字列が丸めずに保存される。
//!
//! ここでは公開 API だけを使う。「panic しない」ことは素で呼んでテストが
//! 落ちないことで検証する (catch_unwind は使わない)。
//!
//! 注: 並列配列 (`KvUsage::busy/t_lens` など) の長さが食い違うフレームは
//! encode が短い方に切り揃える仕様なので、ここでの往復入力は「正規化済み」
//! (長さが一致する) フレームだけを使う。切り揃え自体は wire_golden_994 が
//! 固定している。

use std::io::{BufReader, Cursor};

use denno_telemetry::{
    frame_to_ndjson, parse_frame_line, read_frames_ndjson, write_frames_ndjson, Frame,
    NdjsonFrames,
};

/// 全 17 variant を 1 枚ずつ。正規化済み (並列配列の長さが一致) の入力。
fn sample_frames() -> Vec<Frame> {
    vec![
        Frame::ServerInfo {
            model: "denno-日本語 \"quoted\" \\back\\slash".into(),
            model_id: "Qwen/Qwen2.5-0.5B-Instruct".into(),
            quant: "fp32".into(),
            slots: 4,
            context: 32768,
            layers: 24,
        },
        Frame::SessionAdmit { session: 1, slot: 0, prompt_tokens: 34 },
        Frame::SessionDone { session: 1, reason: u8::MAX, output_tokens: 26 },
        Frame::Prefill {
            session: 1,
            prompt_tokens: 34,
            ns: u64::MAX,
            tokens: vec![0, 972, 198, 3585, u32::MAX],
        },
        Frame::Token { session: 1, token: 42, pos: u32::MAX },
        Frame::Step { width: 8, ns: 1_234_567_890_123 },
        Frame::OpSummary { ops: vec![(0, 10), (3, u64::MAX), (31, 0)] },
        Frame::KvUsage { busy: vec![true, false, true], t_lens: vec![0, 512, u32::MAX] },
        Frame::Gap { lost: 7 },
        Frame::TopK {
            session: 1,
            chosen: 5,
            entropy_f16: 0xabcd,
            entries: vec![(7, 0x3c00), (3, 0x0000), (u32::MAX, 0xffff)],
        },
        Frame::LayerRun { layer: u16::MAX, ops: vec![(4, 1200), (5, 900)] },
        Frame::LayerAct {
            layer: 23,
            idx: vec![1, 2, 3, u32::MAX],
            val: vec![0, 0x3c00, 0x7bff, 0xffff],
        },
        Frame::RoleAct {
            layer: 2,
            attn: 0x3c00,
            gdn: 0,
            router: 0x4000,
            shex: 0x7bff,
            ple: u16::MAX,
        },
        Frame::ExpertResidency { layer: 5, in_vram: vec![0, 2, 7, u16::MAX] },
        Frame::TierStats {
            vram_used_mb: 1024,
            vram_total_mb: u32::MAX,
            ram_experts: 0,
            ssd_reads: 99,
        },
        Frame::RouterTopK {
            layer: 9,
            experts: vec![1, 4, 9],
            weights: vec![0x3c00, 0x3000, 0x2800],
        },
        Frame::ExpertSkip { layer: 9, experts: vec![3, 11] },
    ]
}

/// Binary -> Frame -> NDJSON -> Frame -> Binary の完全往復を 1 枚について検証する。
fn assert_wire_ndjson_parity(f: &Frame) {
    let wire = f.encode();

    let decoded = Frame::decode(&wire).unwrap_or_else(|e| panic!("wire decode 失敗: {e} ({f:?})"));
    assert_eq!(&decoded, f, "binary 往復でフレームが変わった: {f:?}");

    let line = frame_to_ndjson(&decoded);
    assert!(
        !line.contains('\n') && !line.contains('\r'),
        "1 フレームは 1 行に収まる必要がある: {line}"
    );
    let back = parse_frame_line(&line).unwrap_or_else(|e| panic!("ndjson parse 失敗: {e}: {line}"));
    assert_eq!(&back, &decoded, "NDJSON 往復でフレームが変わった: {line}");

    let wire2 = back.encode();
    assert_eq!(
        wire2, wire,
        "Binary -> NDJSON -> Binary でバイト列が変わった (kind {}): {line}",
        f.kind()
    );
}

#[test]
fn every_variant_roundtrips_binary_ndjson_binary_exactly() {
    let frames = sample_frames();

    for f in &frames {
        assert_wire_ndjson_parity(f);
    }

    // 17 variant すべてを踏んでいること (取りこぼし検出)。
    let mut kinds: Vec<u8> = frames.iter().map(Frame::kind).collect();
    kinds.sort_unstable();
    kinds.dedup();
    assert_eq!(kinds.len(), 17, "sample_frames が全 variant を網羅していない: {kinds:?}");
    assert_eq!(kinds, (1..=17).collect::<Vec<u8>>(), "kind 番号に穴がある");
}

#[test]
fn whole_stream_parity_is_byte_identical() {
    let frames = sample_frames();

    // 元のバイナリ列
    let mut wire = Vec::new();
    for f in &frames {
        wire.extend_from_slice(&f.encode());
    }

    // バイナリ -> NDJSON -> フレーム -> バイナリ
    let mut ndjson = Vec::new();
    assert_eq!(write_frames_ndjson(&mut ndjson, &frames).unwrap(), frames.len());

    let (back, errors) = read_frames_ndjson(BufReader::new(Cursor::new(&ndjson)));
    assert!(errors.is_empty(), "有効なストリームをエラーにした: {errors:?}");
    assert_eq!(back, frames, "ストリーム往復でフレーム列が変わった");

    let mut wire2 = Vec::new();
    for f in &back {
        wire2.extend_from_slice(&f.encode());
    }
    assert_eq!(wire2, wire, "ストリーム全体のバイト列が一致しない");

    // 決定的: 2 回書いても同じバイト列
    let mut ndjson2 = Vec::new();
    write_frames_ndjson(&mut ndjson2, &frames).unwrap();
    assert_eq!(ndjson2, ndjson, "NDJSON 出力が非決定的");
}

/// 混在ストリーム用のフレーム (i によって variant を切り替える)。
fn mixed_frame(i: u32) -> Frame {
    let layer = (i % 48) as u16;
    match i % 9 {
        0 => Frame::Token { session: i, token: i.wrapping_mul(7), pos: i },
        1 => Frame::Step { width: 1 + (i % 8), ns: (i as u64) * 1_000_003 },
        2 => Frame::Prefill {
            session: i,
            prompt_tokens: i % 512,
            ns: if i % 97 == 0 { u64::MAX } else { i as u64 },
            tokens: vec![i, i.wrapping_add(1), 0],
        },
        3 => Frame::LayerAct {
            layer,
            idx: vec![i, i.wrapping_add(1)],
            val: vec![0x3c00, (i % 0x7c00) as u16],
        },
        4 => Frame::RouterTopK {
            layer,
            experts: vec![(i % 512) as u16, ((i + 1) % 512) as u16],
            weights: vec![0x3c00, 0x3000],
        },
        5 => Frame::OpSummary { ops: vec![(0, i as u64), (31, u64::MAX - i as u64)] },
        6 => Frame::KvUsage {
            busy: vec![i % 2 == 0, true, false],
            t_lens: vec![i, 0, u32::MAX - i],
        },
        7 => Frame::ServerInfo {
            model: format!("モデル-{i}-\u{1F600}"),
            model_id: format!("id/{i}"),
            quant: "fp32".into(),
            slots: i % 8,
            context: 32768,
            layers: 48,
        },
        _ => Frame::ExpertResidency {
            layer,
            in_vram: vec![(i % 512) as u16, ((i + 7) % 512) as u16],
        },
    }
}

/// ストリームに挟む行の種類。
enum Item {
    Frame(Frame),
    /// 空行 (空白のみ)。読み飛ばされ、エラーにならない。
    Blank,
    /// JSON として途中で切れた行。1 件のエラー。
    Truncated(&'static str),
    /// 不正な UTF-8 バイトを含む行。1 件のエラー。
    BadUtf8(&'static [u8]),
}

/// エラーの種類 (行番号とメッセージの対応を厳密に見るため)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ErrClass {
    Utf8,
    Json,
}

#[test]
fn large_stream_recovers_and_reports_1based_line_numbers() {
    const N_FRAMES: u32 = 5_200;

    let mut items: Vec<Item> = Vec::new();
    let mut expected_frames: Vec<Frame> = Vec::new();

    // 先頭に空行を 1 本 (行 1)。エラーではない。
    items.push(Item::Blank);

    for i in 0..N_FRAMES {
        // 500 フレームごとに空行 + 壊れた行 + 空行を挟む。
        if i > 0 && i % 500 == 0 {
            items.push(Item::Blank);
            items.push(Item::Truncated("{\"kind\":\"token\",\"session\":1,\"tok"));
            items.push(Item::BadUtf8(b"{\"kind\":\"gap\",\"lost\":2}\xff\xfe"));
            items.push(Item::Blank);
        }
        let f = mixed_frame(i);
        items.push(Item::Frame(f.clone()));
        expected_frames.push(f);
    }

    // 末尾に改行なしで切れた行を置く (EOF 到達時に 1 件のエラー)。
    const TAIL: &[u8] = b"{\"kind\":\"step\",\"width\":";

    // 行の内容を直列化し、壊れた行の行番号を記録する。
    let mut text: Vec<u8> = Vec::new();
    let mut expected_errors: Vec<(u64, ErrClass)> = Vec::new();
    for (idx, item) in items.iter().enumerate() {
        let line_no = (idx + 1) as u64;
        match item {
            Item::Frame(f) => text.extend_from_slice(frame_to_ndjson(f).as_bytes()),
            Item::Blank => {}
            Item::Truncated(s) => {
                text.extend_from_slice(s.as_bytes());
                expected_errors.push((line_no, ErrClass::Json));
            }
            Item::BadUtf8(b) => {
                text.extend_from_slice(b);
                expected_errors.push((line_no, ErrClass::Utf8));
            }
        }
        text.push(b'\n');
    }
    // 末尾の切り詰め行 (改行なし)。
    let tail_line_no = (items.len() + 1) as u64;
    text.extend_from_slice(TAIL);
    expected_errors.push((tail_line_no, ErrClass::Json));

    let total_lines = items.len() as u64 + 1;

    // 行境界を跨ぐ読み取りを踏むため、小さめのバッファで読む。
    let reader = BufReader::with_capacity(61, Cursor::new(text.clone()));
    let (got, errors) = read_frames_ndjson(reader);

    assert_eq!(got.len(), expected_frames.len(), "回収できたフレーム数が違う");
    assert_eq!(got, expected_frames, "回収したフレーム列が入力と違う");

    let got_errs: Vec<(u64, ErrClass)> = errors
        .iter()
        .map(|e| {
            let class = if e.message.contains("invalid utf-8") {
                ErrClass::Utf8
            } else {
                ErrClass::Json
            };
            (e.line, class)
        })
        .collect();
    assert_eq!(got_errs, expected_errors, "エラー行番号 / 種類が入力と違う");
    assert!(
        errors.iter().all(|e| !e.message.is_empty()),
        "空のエラーメッセージがある"
    );

    // 1 始まりの行番号が昇順で、範囲内に収まっている。
    let mut prev = 0u64;
    for e in &errors {
        assert!(e.line >= 1, "行番号が 0 始まりになっている: {}", e.line);
        assert!(e.line > prev, "エラー行番号が単調増加でない: {}", e.line);
        assert!(e.line <= total_lines, "行番号が総行数を超える: {}", e.line);
        prev = e.line;
    }

    // イテレータでも同じ結果になり、読み切った行数が総行数と一致する。
    let mut it = NdjsonFrames::new(BufReader::with_capacity(53, Cursor::new(text.clone())));
    let mut ok = 0usize;
    let mut err_lines: Vec<u64> = Vec::new();
    for item in &mut it {
        match item {
            Ok(_) => ok += 1,
            Err(e) => err_lines.push(e.line),
        }
    }
    assert_eq!(ok, expected_frames.len());
    assert_eq!(
        err_lines,
        expected_errors.iter().map(|(l, _)| *l).collect::<Vec<_>>()
    );
    assert_eq!(it.line_no(), total_lines, "読み切った行数が総行数と違う");
}

#[test]
fn extreme_timestamps_and_unicode_strings_are_preserved() {
    // 時刻の極値: 0 と u64::MAX がどちらの形式でも丸まらない。
    let ts_cases = [
        Frame::Step { width: 1, ns: 0 },
        Frame::Step { width: u32::MAX, ns: u64::MAX },
        Frame::Prefill {
            session: u32::MAX,
            prompt_tokens: u32::MAX,
            ns: 0,
            tokens: vec![0, u32::MAX],
        },
        Frame::Prefill {
            session: 0,
            prompt_tokens: 0,
            ns: u64::MAX,
            tokens: Vec::new(),
        },
        Frame::OpSummary { ops: vec![(0, 0), (255, u64::MAX)] },
        Frame::LayerRun { layer: 0, ops: vec![(0, 0), (255, u64::MAX)] },
        Frame::TierStats {
            vram_used_mb: 0,
            vram_total_mb: u32::MAX,
            ram_experts: u32::MAX,
            ssd_reads: 0,
        },
    ];
    for f in &ts_cases {
        assert_wire_ndjson_parity(f);
        // NDJSON の行に u64::MAX を 10 進でそのまま書いていること。
        let line = frame_to_ndjson(f);
        if let Frame::Step { ns, .. } = f {
            if *ns == u64::MAX {
                assert!(
                    line.contains("18446744073709551615"),
                    "u64::MAX の 10 進表記が無い: {line}"
                );
            }
        }
    }

    // Unicode / 制御文字 / 引用符 / 円記号 / サロゲートペア。
    let model = "モデル \u{1F600} \"quoted\" \\back\\slash\nnew\ttab\rcr".to_string();
    let model_id = "識別子/ç/ñ/\u{1F680}".to_string();
    let quant = "量子化\u{0001}end".to_string();
    let unicode = Frame::ServerInfo {
        model: model.clone(),
        model_id: model_id.clone(),
        quant: quant.clone(),
        slots: u32::MAX,
        context: 0,
        layers: u32::MAX,
    };

    assert_wire_ndjson_parity(&unicode);

    // 往復後に文字列がバイト単位で一致する (エスケープの取りこぼし検出)。
    let wire = unicode.encode();
    let decoded = Frame::decode(&wire).expect("decode");
    let line = frame_to_ndjson(&decoded);
    let back = parse_frame_line(&line).expect("parse");
    match back {
        Frame::ServerInfo { model: m, model_id: mi, quant: q, slots, context, layers } => {
            assert_eq!(m, model);
            assert_eq!(mi, model_id);
            assert_eq!(q, quant);
            assert_eq!(slots, u32::MAX);
            assert_eq!(context, 0);
            assert_eq!(layers, u32::MAX);
        }
        other => panic!("ServerInfo に戻らなかった: {other:?}"),
    }

    // NDJSON は特殊文字をエスケープし、生の改行を 1 行の中に持ち込まない。
    assert!(!line.contains('\n'), "エスケープされていない改行がある: {line}");
    assert!(!line.contains('\r'), "エスケープされていない CR がある: {line}");
    assert!(line.contains("\\n"), "改行が \\n にエスケープされていない: {line}");
    assert!(line.contains("\\\""), "引用符がエスケープされていない: {line}");
    assert!(line.contains("\\\\"), "円記号がエスケープされていない: {line}");
    assert!(line.contains("モデル"), "非 ASCII 文字列が消えた: {line}");
}
