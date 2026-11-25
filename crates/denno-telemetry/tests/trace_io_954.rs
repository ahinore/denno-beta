//! task 954: NDJSON トレースの書き出し / 読み戻しテスト。
//!
//! 公開 API だけで検証する:
//! - 全 kind の決定的な往復 (フィールドが過不足なく一致する)
//! - 空ストリーム (フレーム 0 / エラー 0)
//! - 切り詰め行・壊れた行を panic せずにスキップし、行番号付きで報告する
//! - 1000 フレームのストリーミング読み戻し
//!
//! `read_frames_ndjson` は壊れた行で止まらないので、ここでの「panic しない」は
//! 素で呼んでテストが落ちないことで検証する (catch_unwind は使わない)。

use std::io::{BufReader, Cursor};

use denno_telemetry::{
    frame_to_ndjson, parse_frame_line, read_frames_ndjson, write_frame_ndjson,
    write_frames_ndjson, Frame, NdjsonFrames,
};

/// 全 17 kind を 1 枚ずつ。境界値 (u64 の ns, u32::MAX, 空配列, 非 ASCII) を含む。
fn sample_frames() -> Vec<Frame> {
    vec![
        Frame::ServerInfo {
            model: "denno-日本語 \"quoted\" \\ model".into(),
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

#[test]
fn roundtrip_all_variants_is_exact_and_deterministic() {
    let frames = sample_frames();

    // 1 行 1 フレーム
    for f in &frames {
        let line = frame_to_ndjson(f);
        assert!(!line.contains('\n'), "1 フレームは 1 行に収まる: {line}");
        let back = parse_frame_line(&line).expect("parse ok");
        assert_eq!(&back, f, "往復でフィールドが変わった: {line}");
    }

    // ストリーム全体の往復
    let mut buf = Vec::new();
    let n = write_frames_ndjson(&mut buf, &frames).unwrap();
    assert_eq!(n, frames.len());
    assert_eq!(buf.iter().filter(|&&b| b == b'\n').count(), frames.len());

    let (got, errors) = read_frames_ndjson(BufReader::new(Cursor::new(&buf)));
    assert!(errors.is_empty(), "有効な入力をエラーにした: {errors:?}");
    assert_eq!(got, frames);

    // 決定的: 2 回書くとバイト列が一致する
    let mut buf2 = Vec::new();
    write_frames_ndjson(&mut buf2, &frames).unwrap();
    assert_eq!(buf, buf2, "NDJSON 出力が非決定的");
}

#[test]
fn empty_stream_yields_nothing() {
    // 空ファイル
    let (frames, errors) = read_frames_ndjson(BufReader::new(Cursor::new(b"" as &[u8])));
    assert!(frames.is_empty());
    assert!(errors.is_empty());

    // 空行・CRLF の空行だけ
    let (frames, errors) = read_frames_ndjson(BufReader::new(Cursor::new(b"\n\r\n   \n" as &[u8])));
    assert!(frames.is_empty(), "空行はフレームではない: {frames:?}");
    assert!(errors.is_empty(), "空行はエラーではない: {errors:?}");

    // 空スライスの書き出しは 0 行
    let mut buf = Vec::new();
    assert_eq!(write_frames_ndjson(&mut buf, &[]).unwrap(), 0);
    assert!(buf.is_empty());
}

#[test]
fn truncated_line_is_reported_without_panic() {
    let good = Frame::Token { session: 1, token: 42, pos: 7 };

    // 1 行目は有効、2 行目は途中で切れた JSON (末尾改行なし)
    let mut text = frame_to_ndjson(&good);
    text.push('\n');
    text.push_str("{\"kind\":\"token\",\"session\":1,\"tok");

    let (frames, errors) = read_frames_ndjson(BufReader::new(Cursor::new(text.as_bytes())));
    assert_eq!(frames, vec![good.clone()]);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].line, 2);
    assert!(!errors[0].message.is_empty());

    // 3 行目が有効なら、壊れた行の後ろも読める (停止しない)
    let mut text = text;
    text.push('\n');
    text.push_str(&frame_to_ndjson(&Frame::Gap { lost: 3 }));
    let (frames, errors) = read_frames_ndjson(BufReader::new(Cursor::new(text.as_bytes())));
    assert_eq!(frames, vec![good, Frame::Gap { lost: 3 }]);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].line, 2);
}

#[test]
fn malformed_lines_are_skipped_and_reported() {
    // 種類の異なる壊れ方を 1 行ずつ。どれも Err で、panic しない。
    let bad = [
        "not json at all",
        "{\"kind\":\"token\",\"session\":1,\"token\":2}", // pos 欠落
        "{\"kind\":\"nope\"}",                          // 未知の kind
        "{\"kind\":\"token\",\"session\":-1,\"token\":2,\"pos\":3}", // 負の数
        "{\"kind\":\"gap\",\"lost\":1} trailing",       // 末尾の余分
        "{\"kind\":\"gap\",\"lost\":\"3\"}",            // 型違い
        "{\"kind\":\"kv_usage\",\"busy\":[true],\"t_lens\":[1]}{}", // 2 個の JSON
    ];
    for line in bad {
        assert!(parse_frame_line(line).is_err(), "壊れた行を受理した: {line}");
    }

    // ストリーム中に挟んでも、後続の有効な行は読める
    let mut text = String::new();
    text.push_str(&frame_to_ndjson(&Frame::Gap { lost: 1 }));
    text.push('\n');
    text.push_str("garbage\n");
    text.push_str(&frame_to_ndjson(&Frame::Gap { lost: 2 }));
    text.push('\n');

    let (frames, errors) = read_frames_ndjson(BufReader::new(Cursor::new(text.as_bytes())));
    assert_eq!(frames, vec![Frame::Gap { lost: 1 }, Frame::Gap { lost: 2 }]);
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].line, 2);

    // イテレータでも行番号が追える
    let it = NdjsonFrames::new(BufReader::new(Cursor::new(text.as_bytes())));
    let results: Vec<_> = it.collect();
    assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
    assert_eq!(results.len(), 3);
}

#[test]
fn thousand_frame_streaming_roundtrip() {
    let mut frames = Vec::with_capacity(1000);
    for i in 0..1000u32 {
        frames.push(match i % 4 {
            0 => Frame::Token { session: i, token: i.wrapping_mul(7), pos: i },
            1 => Frame::Step { width: 1 + (i % 8), ns: (i as u64) * 1_000_003 },
            2 => Frame::LayerAct {
                layer: (i % 48) as u16,
                idx: vec![i, i + 1],
                val: vec![0x3c00, i as u16],
            },
            _ => Frame::RouterTopK {
                layer: (i % 48) as u16,
                experts: vec![(i % 256) as u16, ((i + 1) % 256) as u16],
                weights: vec![0x3c00, 0x3000],
            },
        });
    }

    let mut buf = Vec::new();
    assert_eq!(write_frames_ndjson(&mut buf, &frames).unwrap(), 1000);

    // 小さめのバッファで 1 行ずつ読む (行境界を跨ぐ読み取りを踏む)
    let reader = BufReader::with_capacity(37, Cursor::new(buf.clone()));
    let (got, errors) = read_frames_ndjson(reader);
    assert!(errors.is_empty(), "有効な入力をエラーにした: {errors:?}");
    assert_eq!(got.len(), 1000);
    assert_eq!(got, frames);

    // イテレータで 1 枚ずつ取り出しても同じ
    let it = NdjsonFrames::new(BufReader::new(Cursor::new(buf.clone())));
    let streamed: Vec<Frame> = it.map(|r| r.expect("valid line")).collect();
    assert_eq!(streamed, frames);

    // 書き出しヘルパ単体も同じ 1 行を書く
    let mut one = Vec::new();
    write_frame_ndjson(&mut one, &frames[0]).unwrap();
    assert_eq!(one.len(), frame_to_ndjson(&frames[0]).len() + 1);
}
