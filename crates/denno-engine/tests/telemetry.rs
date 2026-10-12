//! P13: テレメトリ捕捉の統合テスト (ダミーモデル)。
//!
//! 検証すること:
//! - テレメトリ OFF (既定) ではフレームが一切出ない
//! - ON では SessionAdmit / Prefill / Token / TopK / KvUsage / SessionDone が流れる
//! - **テレメトリ ON/OFF で生成出力が同一** (捕捉が数値経路に触れないことの担保)
//! - 全フレームが encode→decode で可逆、チャネル満杯時はドロップして止まらない

use denno_engine::testutil::{dummy_engine, dummy_tokenizer};
use denno_engine::{ChatMessage, Engine, GenParams};
use denno_telemetry::{self as telem, Frame};

fn greedy(max_tokens: usize) -> GenParams {
    GenParams { max_tokens, temperature: Some(0.0), ..Default::default() }
}

fn msgs(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage { role: "user".into(), content: text.into() }]
}

/// ハブから今あるフレームを全部吸い出す (非ブロッキング)。
fn drain() -> Vec<Frame> {
    let mut out = Vec::new();
    while let Some(f) = telem::try_recv_nowait() {
        out.push(f);
    }
    out
}

/// ダミーモデル+ダミートークナイザの greedy 基準列 (観測値・決定的)。
/// プロンプト "t0" はダミートークナイザで t1..t4 に展開され、
/// DummyModel は t_n の次を常に t_{n+1} と予測するため出力は t5..t11。
fn expected_text() -> String {
    "t5 t6 t7 t8 t9 t10 t11".to_string()
}

#[test]
fn telemetry_off_then_on_lifecycle_and_output_invariance() {
    // --- OFF (既定) ---
    telem::set_enabled_for_tests(false);
    assert!(!telem::enabled(), "telemetry must be off");
    let want = expected_text();

    let mut e_off = dummy_engine("off", Some(42)).unwrap();
    let out_off = e_off.chat(None, &msgs("t0"), &greedy(20), |_| {}).unwrap();
    assert_eq!(out_off.text, want, "OFF 時の greedy 基準列");
    assert!(drain().is_empty(), "telemetry off でフレームが出た");

    // --- ON ---
    telem::set_enabled_for_tests(true);
    assert!(telem::enabled(), "telemetry must be on");
    let mut e_on = dummy_engine("on", Some(42)).unwrap();
    let out_on = e_on.chat(None, &msgs("t0"), &greedy(20), |_| {}).unwrap();

    // **テレメトリ ON でも出力は OFF と完全一致** (捕捉が数値経路に触れない担保)
    assert_eq!(out_on.text, out_off.text, "テレメトリ ON で出力が変わった");
    assert_eq!(out_on.output_tokens, out_off.output_tokens);
    assert_eq!(out_on.stop_reason, out_off.stop_reason);

    // 生成中にハブに溜まったフレームを吸い出す
    let frames = drain();
    assert!(!frames.is_empty(), "telemetry on でフレームが出ない");

    let has = |frames: &[Frame], pred: &dyn Fn(&Frame) -> bool| frames.iter().any(pred);
    assert!(has(&frames, &|f| matches!(f, Frame::SessionAdmit { .. })), "SessionAdmit 無し");
    assert!(has(&frames, &|f| matches!(f, Frame::Prefill { .. })), "Prefill 無し");
    assert!(has(&frames, &|f| matches!(f, Frame::Token { .. })), "Token 無し");
    assert!(has(&frames, &|f| matches!(f, Frame::TopK { .. })), "TopK 無し");
    assert!(has(&frames, &|f| matches!(f, Frame::KvUsage { .. })), "KvUsage 無し");
    assert!(has(&frames, &|f| matches!(f, Frame::SessionDone { .. })), "SessionDone 無し");
    assert!(has(&frames, &|f| matches!(f, Frame::OpSummary { .. })), "OpSummary 無し");

    // Token の pos は 1 から始まる連番 (1 セッション分)
    let toks: Vec<u32> = frames
        .iter()
        .filter_map(|f| match f {
            Frame::Token { pos, .. } => Some(*pos),
            _ => None,
        })
        .collect();
    assert_eq!(toks.first(), Some(&1));
    assert_eq!(toks.len(), out_on.output_tokens, "Token フレーム数 == 出力トークン数");
    for (a, b) in toks.iter().zip(toks.iter().skip(1)) {
        assert_eq!(b, &(a + 1), "pos が連番でない");
    }

    // Token フレームの token 列からテキストを復元できる。
    // 最終トークン (EOS) も Token フレームに含まれるため、復元テキストの末尾に
    // " eos" が付く (出力テキスト側には現れない)。
    let tok_ids: Vec<u32> = frames
        .iter()
        .filter_map(|f| match f {
            Frame::Token { token, .. } => Some(*token),
            _ => None,
        })
        .collect();
    let tok = dummy_tokenizer().unwrap();
    let decoded = tok.decode(&tok_ids).unwrap();
    let decoded = decoded.strip_suffix(" eos").unwrap_or(&decoded);
    assert_eq!(decoded, out_on.text, "Token フレームからテキスト復元不一致");

    // 全フレームが encode→decode で可逆
    for f in &frames {
        let enc = f.encode();
        assert_eq!(Frame::decode(&enc).unwrap(), *f);
    }

    // チャネルに過剰に詰めても推論は止まらない (try_send = 非ブロッキング)。
    // 溢れた分はドロップ計上される。
    let before = telem::dropped();
    for _ in 0..(telem::channel_cap() * 2) {
        telem::emit(Frame::Gap { lost: 1 });
    }
    assert!(telem::dropped() >= before + telem::channel_cap() as u64, "満杯時のドロップ計上");
    assert!(drain().len() >= telem::channel_cap() - 1);
}
