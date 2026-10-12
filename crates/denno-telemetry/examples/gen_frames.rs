//! JS デコーダ (denno-view/src/index.html) の検証用にフレームの 16 進表現を出力する。
//! 使い方: cargo run -p denno-telemetry --example gen_frames > frames.hex
fn main() {
    let f = denno_telemetry::f32_to_f16_bits;
    let frames = vec![
        denno_telemetry::Frame::ServerInfo {
            model: "denno-dev".into(),
            quant: "fp32".into(),
            slots: 4,
            context: 32768,
        },
        denno_telemetry::Frame::SessionAdmit { session: 1, slot: 0, prompt_tokens: 34 },
        denno_telemetry::Frame::Prefill { session: 1, prompt_tokens: 34, ns: 123_456_789 },
        denno_telemetry::Frame::Token { session: 1, token: 100, pos: 1 },
        denno_telemetry::Frame::TopK {
            session: 1,
            chosen: 100,
            entropy_f16: f(2.75),
            entries: vec![(100, f(0.5)), (200, f(0.25))],
        },
        denno_telemetry::Frame::Step { width: 1, ns: 42_000_000 },
        denno_telemetry::Frame::OpSummary { ops: vec![(4, 119_300_000), (10, 55_000_000)] },
        denno_telemetry::Frame::KvUsage {
            busy: vec![true, false, true, false],
            t_lens: vec![34, 0, 5, 0],
        },
        denno_telemetry::Frame::Gap { lost: 3 },
        denno_telemetry::Frame::SessionDone {
            session: 1,
            reason: denno_telemetry::done_reason::END_TURN,
            output_tokens: 26,
        },
    ];
    for fr in frames {
        let enc = fr.encode();
        println!("{}", enc.iter().map(|b| format!("{b:02x}")).collect::<String>());
    }
}
