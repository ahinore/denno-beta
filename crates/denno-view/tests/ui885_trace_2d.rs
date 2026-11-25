//! /2d のトレース記録 / オフライン再生 (task 885) の静的な結線の検査。
//!
//! ロジックは node --test (tests/trace2d.test.mjs) が見る。ここは
//! 「画面のアセットが純ロジックと噛み合っているか」だけを確かめる。
//!
//! 見ているもの:
//!   1. record / stop / export / load とシークバーの器 (id) が DOM にある
//!   2. 記録は既定で止まっていて、record を押したときだけ始まる
//!   3. 記録は 2,048 フレームで有界、使用量を画面に出す
//!   4. 再生は純ロジック (createReplay / replayStateAt) を呼ぶだけで、
//!      再生中はライブのフレームを地図に流さない
//!   5. view2d.mjs が記録・検証・再生の契約を export している
//!   6. 検証用フックがある

const HTML: &str = include_str!("../src/index2d.html");
const VIEW: &str = include_str!("../src/view2d.mjs");

#[test]
fn trace_controls_exist() {
    for id in [
        "tracebox", "tr-record", "tr-stop", "tr-export", "tr-file", "tr-info",
        "tr-seek", "tr-prev", "tr-next", "tr-pos", "tr-live", "tr-err",
    ] {
        assert!(HTML.contains(&format!("id=\"{id}\"")), "トレースの #{id} が無い");
    }
    assert!(HTML.contains("id=\"tr-seek\" type=\"range\""), "#tr-seek が range でない");
    assert!(HTML.contains("2,048 フレーム"), "上限の説明が無い");
    assert!(HTML.contains(".tracebar input[type=range]"), "シークバーの CSS が無い");
}

#[test]
fn recording_is_off_until_the_button_is_pressed() {
    // 起動時は空のバッファで、記録は止まっている
    assert!(HTML.contains("let trace = createTrace(shape);"), "記録バッファの初期化が無い");
    assert!(HTML.contains("if (trace.recording) {"), "記録中だけ積む分岐が無い");
    assert!(HTML.contains("traceStart(trace, Date.now())"), "record で開始していない");
    assert!(HTML.contains("traceStop(trace, { at: Date.now() })"), "stop で止めていない");
    // 既定の表示
    assert!(HTML.contains("trace: idle"), "初期表示が idle でない");
}

#[test]
fn recording_buffer_is_bounded_and_reported() {
    assert!(HTML.contains("TRACE_MAX_FRAMES"), "上限を使っていない");
    assert!(HTML.contains("{ maxFrames: TRACE_MAX_FRAMES }"), "バッファに上限を渡していない");
    assert!(HTML.contains("traceMemory(trace)"), "使用量を取っていない");
    assert!(HTML.contains("mem.frames + ' / ' + mem.maxFrames"), "フレーム数を出していない");
    assert!(HTML.contains("mem.kb"), "メモリ (KB) を出していない");
    assert!(HTML.contains("'dropped ' + mem.dropped"), "落としたフレームを出していない");
}

#[test]
fn replay_uses_the_pure_module_and_blocks_live_frames() {
    for name in ["createReplay(doc)", "replayStateAt(replay, tok)", "replayStep(replay,"] {
        assert!(HTML.contains(name), "画面が {name} を呼んでいない");
    }
    // 再生中はライブのフレームで地図を動かさない
    assert!(HTML.contains("if (replay) return;"), "再生中もライブを流している");
    // シーク (range / prev / next / ←→ / back to live)
    assert!(HTML.contains("$('tr-seek').addEventListener('input'"), "シークバーの handler が無い");
    assert!(HTML.contains("applyReplayAt(v)"), "シークが位置を反映していない");
    assert!(HTML.contains("$('tr-prev').addEventListener"), "prev の handler が無い");
    assert!(HTML.contains("$('tr-next').addEventListener"), "next の handler が無い");
    assert!(HTML.contains("e.key === 'ArrowLeft' || e.key === 'ArrowRight'"), "←→ で動かない");
    assert!(HTML.contains("$('tr-live').addEventListener('click', () => backToLive())"),
        "back to live が無い");
}

#[test]
fn load_and_export_go_through_validation() {
    assert!(HTML.contains("traceParse(text)"), "load で検証していない");
    assert!(HTML.contains("traceSerialize(doc)"), "export が直列化していない");
    assert!(HTML.contains("traceFileName(Date.now())"), "ファイル名を作っていない");
    assert!(HTML.contains(".denno-trace.json"), "拡張子の案内が無い");
    // 形の食い違いは断る (地図の寸法が変わるため)
    assert!(HTML.contains("shape mismatch (trace '"), "形の検査が無い");
    // load の失敗は赤字で理由を出す
    assert!(HTML.contains("$('tr-err').textContent"), "エラー表示が無い");
}

#[test]
fn view2d_exports_the_trace_contract() {
    for name in [
        "TRACE_SCHEMA", "TRACE_VERSION", "TRACE_MAX_FRAMES", "TRACE_SUFFIX",
        "traceFrame", "createTrace", "traceStart", "traceRecord", "traceStop",
        "traceDocument", "traceMemory", "traceBytes", "traceSerialize", "traceValidate",
        "traceParse", "traceFileName", "traceSummary", "createReplay",
        "replayTokenCount", "replaySeek", "replayStep", "replayTokenAt", "replayStateAt",
    ] {
        let pat = format!("export function {name}(");
        let pat2 = format!("export const {name} ");
        assert!(VIEW.contains(&pat) || VIEW.contains(&pat2),
            "view2d.mjs に {name} が無い");
    }
    assert!(VIEW.contains("export const TRACE_SCHEMA = 'denno-trace'"), "schema 名が違う");
    assert!(VIEW.contains("export const TRACE_MAX_FRAMES = 2048"), "既定の上限が 2,048 でない");
}

#[test]
fn verification_hooks_exist() {
    for name in [
        "recordTrace:", "stopTrace:", "traceText:", "loadTraceText:", "seekTrace:",
        "replay: () =>", "stopReplay:", "traceInfoText:",
    ] {
        assert!(HTML.contains(name), "検証用フック {name} が無い");
    }
}
