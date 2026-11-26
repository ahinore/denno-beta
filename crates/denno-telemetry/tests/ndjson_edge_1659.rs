//! task 1659: NDJSON 直列化 / 復元の境界テスト。
//!
//! 検証する不変条件:
//!
//! - 全 17 フレーム種別が NDJSON 1 行 <-> [`Frame`] を往復し、バイナリ表現も
//!   バイト一致する (直列化は決定的)。
//! - すべての可変長配列が空のとき `[]` になり、往復で空のまま戻る。
//! - 文字列のエスケープ: 引用符 / 円記号 / 改行 / CR / タブ / すべての C0 制御
//!   文字 / DEL / 非 ASCII / サロゲートペア (astral)。1 行の中に生の制御文字を
//!   持ち込まない。
//! - 数値は常に 10 進整数として書かれ、`NaN` / `Inf` / `Infinity` のような
//!   浮動小数の特殊トークンを JSON に漏らさない (f16 は生ビットの u16 で送る)。
//! - 行フレーミング: `write_frame_ndjson` は末尾に `\n` を 1 個だけ足す。
//!   ストリーム読み込みは空行・CRLF・末尾改行なしを吸収する。
//!
//! ここでは公開 API だけを使う。壊れた入力に対して panic しないことは、
//! 素で呼んでテストが落ちないことで検証する。

use std::io::{BufReader, Cursor};

use denno_telemetry::{
    f32_to_f16_bits, frame_to_ndjson, parse_frame_line, read_frames_ndjson, write_frame_ndjson,
    write_frames_ndjson, Frame,
};

/// 全 17 種別を 1 枚ずつ、境界値を多用して並べる。
fn edge_frames() -> Vec<Frame> {
    vec![
        Frame::ServerInfo {
            model: "denno \u{1F600}".into(),
            model_id: "id/0".into(),
            quant: "fp32".into(),
            slots: 0,
            context: u32::MAX,
            layers: 0,
        },
        Frame::SessionAdmit { session: 0, slot: u32::MAX, prompt_tokens: 0 },
        Frame::SessionDone { session: u32::MAX, reason: u8::MAX, output_tokens: 0 },
        Frame::Prefill {
            session: u32::MAX,
            prompt_tokens: 0,
            ns: u64::MAX,
            tokens: vec![0, u32::MAX, 1],
        },
        Frame::Token { session: 0, token: u32::MAX, pos: 0 },
        Frame::Step { width: u32::MAX, ns: u64::MAX },
        Frame::OpSummary { ops: vec![(0, 0), (31, u64::MAX), (255, 1)] },
        Frame::KvUsage { busy: vec![true, false, true], t_lens: vec![0, u32::MAX, 1] },
        Frame::Gap { lost: u32::MAX },
        Frame::TopK {
            session: 0,
            chosen: u32::MAX,
            entropy_f16: 0xabcd,
            entries: vec![(0, 0x0000), (u32::MAX, 0xfbff)],
        },
        Frame::LayerRun { layer: u16::MAX, ops: vec![(0, 0), (255, u64::MAX)] },
        Frame::LayerAct {
            layer: 0,
            idx: vec![0, u32::MAX],
            val: vec![0x0000, 0x7bff],
        },
        Frame::RoleAct {
            layer: u16::MAX,
            attn: 0,
            gdn: u16::MAX,
            router: 1,
            shex: 0x7c00,
            ple: 0xfc00,
        },
        Frame::ExpertResidency { layer: 0, in_vram: vec![0, u16::MAX] },
        Frame::TierStats {
            vram_used_mb: u32::MAX,
            vram_total_mb: 0,
            ram_experts: u32::MAX,
            ssd_reads: 0,
        },
        Frame::RouterTopK {
            layer: u16::MAX,
            experts: vec![0, u16::MAX],
            weights: vec![0x3c00, 0x0000],
        },
        Frame::ExpertSkip { layer: 1, experts: vec![0, 1, u16::MAX] },
    ]
}

fn assert_one_line(line: &str) {
    assert!(!line.contains('\n'), "生の改行が行に混入した: {line:?}");
    assert!(!line.contains('\r'), "生の CR が行に混入した: {line:?}");
    assert!(line.starts_with('{'), "行が JSON オブジェクトで始まらない: {line}");
    assert!(line.ends_with('}'), "行が JSON オブジェクトで終わらない: {line}");
    for c in line.chars() {
        assert!(
            (c as u32) >= 0x20,
            "生の制御文字 {:#04x} が行に混入した: {line:?}",
            c as u32
        );
    }
}

/// 文字列リテラルの外側にある数値トークンを拾う。
fn numeric_tokens(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let mut i = 0usize;
    let mut in_str = false;
    let mut esc = false;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        match c {
            b'"' => {
                in_str = true;
                i += 1;
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                i += 1;
                while i < b.len()
                    && matches!(b[i], b'0'..=b'9' | b'.' | b'e' | b'E' | b'+' | b'-')
                {
                    i += 1;
                }
                out.push(&line[start..i]);
            }
            _ => i += 1,
        }
    }
    out
}

/// 数値トークンがすべて 10 進整数で、有限の f64 として解釈できることを見る。
fn assert_integer_numbers_only(line: &str) {
    let toks = numeric_tokens(line);
    assert!(!toks.is_empty(), "数値トークンが 1 個も無い: {line}");
    for t in toks {
        assert!(
            t.chars().all(|c| c.is_ascii_digit()),
            "整数以外の数値トークン {t:?} がある (浮動小数/符号): {line}"
        );
        let v: f64 = t.parse().unwrap_or_else(|_| panic!("数値 {t:?} が f64 として不正: {line}"));
        assert!(v.is_finite(), "非有限の数値 {t:?} が JSON に漏れた: {line}");
    }
}

/// 文字列リテラルの外側にある英字トークンを拾う (true/false 以外は異常)。
fn word_tokens_outside_strings(line: &str) -> Vec<&str> {
    let b = line.as_bytes();
    let mut i = 0usize;
    let mut in_str = false;
    let mut esc = false;
    let mut out = Vec::new();
    while i < b.len() {
        let c = b[i];
        if in_str {
            if esc {
                esc = false;
            } else if c == b'\\' {
                esc = true;
            } else if c == b'"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c.is_ascii_alphabetic() {
            let start = i;
            while i < b.len() && b[i].is_ascii_alphabetic() {
                i += 1;
            }
            out.push(&line[start..i]);
        } else {
            if c == b'"' {
                in_str = true;
            }
            i += 1;
        }
    }
    out
}

/// 特殊な浮動小数トークン (NaN/Inf/Infinity) が構造トークンとして出現しないこと。
/// 文字列リテラル内 (kind 名 "server_info" など) は対象外。
fn assert_no_float_special_tokens(line: &str) {
    for w in word_tokens_outside_strings(line) {
        assert!(
            w == "true" || w == "false",
            "数値位置に浮動小数の特殊トークン {w:?} が漏れた: {line}"
        );
    }
}

#[test]
fn all_17_kinds_roundtrip_through_one_line() {
    let frames = edge_frames();

    // kind 番号が 1..=17 を隙間なく覆っていること。
    let mut kinds: Vec<u8> = frames.iter().map(Frame::kind).collect();
    kinds.sort_unstable();
    assert_eq!(kinds, (1..=17).collect::<Vec<u8>>(), "17 種別を網羅していない: {kinds:?}");

    for f in &frames {
        let name = denno_telemetry::ndjson_kind_name(f.kind())
            .unwrap_or_else(|| panic!("kind {} に名前が無い", f.kind()));
        assert_eq!(
            denno_telemetry::ndjson_kind_from_name(name),
            Some(f.kind()),
            "kind 名 <-> バイトの往復が壊れている: {name}"
        );

        let line = frame_to_ndjson(f);
        assert_one_line(&line);
        assert!(line.contains(&format!("\"kind\":\"{name}\"")), "kind フィールドが無い: {line}");

        let back = parse_frame_line(&line).unwrap_or_else(|e| panic!("parse 失敗: {e}: {line}"));
        assert_eq!(&back, f, "NDJSON 往復でフレームが変わった: {line}");

        // バイナリ表現も一致する (直列化の決定性)。
        assert_eq!(back.encode(), f.encode(), "往復後のバイナリが変わった: {line}");
        assert_eq!(frame_to_ndjson(&back), line, "再直列化が非決定的: {line}");
    }
}

#[test]
fn empty_collections_serialize_as_empty_arrays() {
    let cases: Vec<(Frame, &str)> = vec![
        (Frame::Prefill { session: 0, prompt_tokens: 0, ns: 0, tokens: vec![] }, "\"tokens\":[]"),
        (Frame::OpSummary { ops: vec![] }, "\"ops\":[]"),
        (Frame::KvUsage { busy: vec![], t_lens: vec![] }, "\"busy\":[],\"t_lens\":[]"),
        (
            Frame::TopK { session: 0, chosen: 0, entropy_f16: 0, entries: vec![] },
            "\"entries\":[]",
        ),
        (Frame::LayerRun { layer: 0, ops: vec![] }, "\"ops\":[]"),
        (Frame::LayerAct { layer: 0, idx: vec![], val: vec![] }, "\"idx\":[],\"val\":[]"),
        (Frame::ExpertResidency { layer: 0, in_vram: vec![] }, "\"in_vram\":[]"),
        (
            Frame::RouterTopK { layer: 0, experts: vec![], weights: vec![] },
            "\"experts\":[],\"weights\":[]",
        ),
        (Frame::ExpertSkip { layer: 0, experts: vec![] }, "\"experts\":[]"),
    ];

    for (f, needle) in &cases {
        let line = frame_to_ndjson(f);
        assert!(line.contains(needle), "空配列が {needle} になっていない: {line}");
        let back = parse_frame_line(&line).unwrap_or_else(|e| panic!("parse 失敗: {e}: {line}"));
        assert_eq!(&back, f, "空配列の往復でフレームが変わった: {line}");
    }

    // 空文字列のフィールドも往復する。
    let empty = Frame::ServerInfo {
        model: String::new(),
        model_id: String::new(),
        quant: String::new(),
        slots: 0,
        context: 0,
        layers: 0,
    };
    let line = frame_to_ndjson(&empty);
    assert!(line.contains("\"model\":\"\""), "空文字列が \"\" になっていない: {line}");
    assert_eq!(parse_frame_line(&line).expect("parse"), empty);
}

#[test]
fn string_escaping_covers_c0_del_quotes_and_unicode() {
    // 0x00..=0x1f の C0 制御文字を全部含む文字列。
    let mut ctrl = String::new();
    for cp in 0x00u32..=0x1f {
        ctrl.push(char::from_u32(cp).expect("valid C0 code point"));
    }
    // DEL (0x7f)、引用符、円記号、スラッシュ、非 ASCII、astral。
    let tricky = "\u{7f}\"\\/\u{e9}\u{65e5}\u{672c}\u{8a9e}\u{1F600}\u{1F680}";
    let model = format!("{ctrl}{tricky}");
    let model_id = "識別子/\u{00e7}/\u{00f1}".to_string();
    // quant は C0 制御文字だけを含む短い文字列 (別経路の確認)。
    let quant = "\u{0001}\u{001f}\u{000b}".to_string();

    let frame = Frame::ServerInfo {
        model: model.clone(),
        model_id: model_id.clone(),
        quant: quant.clone(),
        slots: 7,
        context: 4096,
        layers: 48,
    };

    let line = frame_to_ndjson(&frame);
    assert_one_line(&line);

    // 生の制御文字は 1 個も残らない。
    for c in ctrl.chars() {
        assert!(
            !line.contains(c),
            "生の制御文字 {:#04x} が残っている: {line:?}",
            c as u32
        );
    }

    // 個別のエスケープ表記。
    assert!(line.contains("\\\""), "引用符が \\\" にエスケープされていない: {line}");
    assert!(line.contains("\\\\"), "円記号が \\\\ にエスケープされていない: {line}");
    assert!(line.contains("\\n"), "LF が \\n にエスケープされていない: {line}");
    assert!(line.contains("\\r"), "CR が \\r にエスケープされていない: {line}");
    assert!(line.contains("\\t"), "TAB が \\t にエスケープされていない: {line}");
    for cp in 0x00u32..=0x1f {
        // LF/CR/TAB は短縮表記、それ以外は \u00XX。
        let expected = match cp {
            0x09 => "\\t".to_string(),
            0x0a => "\\n".to_string(),
            0x0d => "\\r".to_string(),
            _ => format!("\\u{:04x}", cp),
        };
        assert!(
            line.contains(&expected),
            "制御文字 {cp:#04x} が {expected} にエスケープされていない: {line:?}"
        );
    }

    // 非 ASCII と astral はそのまま UTF-8 で出る (JSON として有効)。
    assert!(line.contains("日本語"), "非 ASCII 文字列が消えた: {line}");
    assert!(line.contains('\u{1F600}'), "astral 文字が消えた: {line}");

    // 往復でバイト単位一致。
    let back = parse_frame_line(&line).unwrap_or_else(|e| panic!("parse 失敗: {e}: {line}"));
    match back {
        Frame::ServerInfo { model: m, model_id: mi, quant: q, slots, context, layers } => {
            assert_eq!(m, model, "model の往復が壊れた");
            assert_eq!(mi, model_id, "model_id の往復が壊れた");
            assert_eq!(q, quant, "quant の往復が壊れた");
            assert_eq!((slots, context, layers), (7, 4096, 48));
        }
        other => panic!("ServerInfo に戻らなかった: {other:?}"),
    }
}

#[test]
fn parser_accepts_all_json_escape_forms() {
    // \/ \b \f \uXXXX (BMP) とサロゲートペア \uD83D\uDE00 を手書き JSON で与える。
    let line = concat!(
        r#"{"kind":"server_info","model":"a\/b\b\f\u0041\u0000\uD83D\uDE00","#,
        r#""model_id":"x","quant":"y","slots":0,"context":0,"layers":0}"#
    );
    let f = parse_frame_line(line).expect("escape を含む行の parse");
    match f {
        Frame::ServerInfo { model, .. } => {
            assert_eq!(
                model,
                "a/b\u{0008}\u{000c}A\u{0000}\u{1F600}",
                "各エスケープの解釈が違う"
            );
        }
        other => panic!("ServerInfo に戻らなかった: {other:?}"),
    }
}

#[test]
fn parser_rejects_malformed_escapes_and_raw_control_bytes() {
    // 文字列中の生の制御文字 (0x01) は拒否。
    let raw_ctrl = "{\"kind\":\"server_info\",\"model\":\"a\u{0001}b\",\"model_id\":\"x\",\"quant\":\"y\",\"slots\":0,\"context\":0,\"layers\":0}";
    assert!(parse_frame_line(raw_ctrl).is_err(), "生の制御文字を通した");

    // 不正なエスケープ。
    let bad_escape = r#"{"kind":"server_info","model":"a\qb","model_id":"x","quant":"y","slots":0,"context":0,"layers":0}"#;
    assert!(parse_frame_line(bad_escape).is_err(), "不正なエスケープを通した");

    // 孤立した上位サロゲート。
    let lone_hi = r#"{"kind":"server_info","model":"\ud800","model_id":"x","quant":"y","slots":0,"context":0,"layers":0}"#;
    assert!(parse_frame_line(lone_hi).is_err(), "孤立サロゲートを通した");

    // 上位サロゲートの直後が下位サロゲートでない。
    let bad_pair = r#"{"kind":"server_info","model":"\ud800\u0041","model_id":"x","quant":"y","slots":0,"context":0,"layers":0}"#;
    assert!(parse_frame_line(bad_pair).is_err(), "不正なサロゲートペアを通した");

    // 閉じられていない文字列。
    let unterminated = r#"{"kind":"server_info","model":"unterminated"#;
    assert!(parse_frame_line(unterminated).is_err(), "閉じていない文字列を通した");
}

#[test]
fn parser_rejects_structural_errors() {
    // 未知の kind。
    assert!(parse_frame_line(r#"{"kind":"nope"}"#).is_err(), "未知の kind を通した");
    // トップレベルが配列。
    assert!(parse_frame_line(r#"[]"#).is_err(), "トップレベル配列を通した");
    // 末尾に余分なバイト。
    let trailing = r#"{"kind":"gap","lost":1} extra"#;
    assert!(parse_frame_line(trailing).is_err(), "末尾の余分バイトを通した");
    // フィールド欠落。
    assert!(parse_frame_line(r#"{"kind":"gap"}"#).is_err(), "欠落フィールドを通した");
    // 型違い (数値フィールドに文字列)。
    assert!(
        parse_frame_line(r#"{"kind":"gap","lost":"1"}"#).is_err(),
        "型違いを通した"
    );
    // u32 オーバーフロー。
    assert!(
        parse_frame_line(r#"{"kind":"gap","lost":4294967296}"#).is_err(),
        "u32 オーバーフローを通した"
    );
    // u8 オーバーフロー (reason)。
    assert!(
        parse_frame_line(
            r#"{"kind":"session_done","session":0,"reason":256,"output_tokens":0}"#
        )
        .is_err(),
        "u8 オーバーフローを通した"
    );
}

#[test]
fn float_values_are_encoded_as_integer_f16_bits_never_nan_or_inf_tokens() {
    // f32 の特殊値は f16 の生ビット (u16) に落ち、JSON には 10 進整数で出る。
    let nan_bits = f32_to_f16_bits(f32::NAN);
    let inf_bits = f32_to_f16_bits(f32::INFINITY);
    let ninf_bits = f32_to_f16_bits(f32::NEG_INFINITY);
    assert_eq!(nan_bits, 0x7e00, "NaN が quiet NaN ビットにならない");
    assert_eq!(inf_bits, 0x7c00, "+Inf が Inf ビットにならない");
    assert_eq!(ninf_bits, 0xfc00, "-Inf が -Inf ビットにならない");

    let frame = Frame::LayerAct {
        layer: 3,
        idx: vec![1, 2, 3],
        val: vec![nan_bits, inf_bits, ninf_bits],
    };
    let line = frame_to_ndjson(&frame);

    // 生の NaN/Inf 文字列は出ない。f16 ビットが 10 進整数で出る。
    assert_no_float_special_tokens(&line);
    assert_integer_numbers_only(&line);
    for bits in [nan_bits, inf_bits, ninf_bits] {
        assert!(
            line.contains(&bits.to_string()),
            "f16 ビット {bits} が 10 進整数で出ていない: {line}"
        );
    }

    // 復元すると元のビット列に戻る。
    assert_eq!(parse_frame_line(&line).expect("parse"), frame);

    // 全 17 フレームの直列化にも浮動小数の特殊トークンが漏れない。
    for f in &edge_frames() {
        let line = frame_to_ndjson(f);
        assert_no_float_special_tokens(&line);
        assert_integer_numbers_only(&line);
    }
}

#[test]
fn write_frame_ndjson_adds_exactly_one_trailing_newline() {
    let f = Frame::Token { session: 1, token: 2, pos: 3 };
    let mut buf = Vec::new();
    write_frame_ndjson(&mut buf, &f).expect("write");
    assert!(buf.ends_with(b"\n"), "末尾に改行が無い: {buf:?}");
    assert_eq!(buf.iter().filter(|&&b| b == b'\n').count(), 1, "改行が 1 個でない");
    assert_eq!(buf.iter().filter(|&&b| b == b'\r').count(), 0, "生の CR が混入した");

    // 改行を除いた本体が 1 行の直列化と一致し、読み戻せる。
    let body = std::str::from_utf8(&buf[..buf.len() - 1]).expect("utf-8");
    assert_eq!(body, frame_to_ndjson(&f));
    assert_eq!(parse_frame_line(body).expect("parse"), f);
}

#[test]
fn write_frames_ndjson_one_line_per_frame() {
    let frames = edge_frames();
    let mut buf = Vec::new();
    let n = write_frames_ndjson(&mut buf, &frames).expect("write");
    assert_eq!(n, frames.len());
    assert_eq!(
        buf.iter().filter(|&&b| b == b'\n').count(),
        frames.len(),
        "フレーム数と改行数が一致しない"
    );
    assert_eq!(buf.iter().filter(|&&b| b == b'\r').count(), 0, "生の CR が混入した");
    assert!(buf.ends_with(b"\n"));

    // 空のフレーム列は 0 行を書く。
    let mut empty = Vec::new();
    assert_eq!(write_frames_ndjson(&mut empty, &[]).expect("write"), 0);
    assert!(empty.is_empty(), "空列で出力が出た: {empty:?}");

    // 決定性: 2 回書いても同じバイト列。
    let mut buf2 = Vec::new();
    write_frames_ndjson(&mut buf2, &frames).expect("write");
    assert_eq!(buf2, buf, "NDJSON 出力が非決定的");
}

#[test]
fn parse_frame_line_tolerates_trailing_whitespace_but_not_inside_strings() {
    let f = Frame::Gap { lost: 9 };
    let base = frame_to_ndjson(&f);

    // 末尾の空白 (CR/LF/TAB/SP) は読み飛ばされる。
    for suffix in ["\n", "\r\n", " ", "\t", "  \r\n"] {
        let line = format!("{base}{suffix}");
        assert_eq!(
            parse_frame_line(&line).unwrap_or_else(|e| panic!("parse 失敗 ({suffix:?}): {e}")),
            f,
            "末尾空白 {suffix:?} を吸収できていない"
        );
    }

    // 先頭の空白も読み飛ばされる。
    let leading = format!("   {base}");
    assert_eq!(parse_frame_line(&leading).expect("parse"), f);

    // 文字列の中の生の改行は制御文字として拒否される。
    let literal_nl = "{\"kind\":\"server_info\",\"model\":\"a\nb\",\"model_id\":\"x\",\"quant\":\"y\",\"slots\":0,\"context\":0,\"layers\":0}";
    assert!(parse_frame_line(literal_nl).is_err(), "文字列内の生改行を通した");
}

#[test]
fn stream_reading_handles_crlf_blank_lines_and_missing_final_newline() {
    let frames = edge_frames();

    // 先頭に空行、途中に空白のみ行、CRLF 改行、末尾に改行なしを混ぜる。
    let mut text = String::new();
    text.push_str("\r\n"); // 行 1: 空行
    text.push_str("   \t\r\n"); // 行 2: 空白のみ
    for (i, f) in frames.iter().enumerate() {
        text.push_str(&frame_to_ndjson(f));
        if i + 1 == frames.len() {
            // 最終フレームだけ改行を付けない。
        } else {
            text.push_str("\r\n");
        }
    }

    let (got, errors) = read_frames_ndjson(BufReader::with_capacity(37, Cursor::new(text.as_bytes())));
    assert!(errors.is_empty(), "有効なストリームをエラーにした: {errors:?}");
    assert_eq!(got, frames, "CRLF / 空行 / 末尾改行なしの往復が壊れた");

    // 行番号は空行も数える: フレームは 3 行目から始まる。
    let it = denno_telemetry::NdjsonFrames::new(BufReader::new(Cursor::new(text.as_bytes())));
    let collected: Vec<Frame> = it
        .map(|r| r.unwrap_or_else(|e| panic!("行の parse 失敗: {e}")))
        .collect();
    assert_eq!(collected, frames);
}
