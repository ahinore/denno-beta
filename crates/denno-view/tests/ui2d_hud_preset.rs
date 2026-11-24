//! /2d のライブ性能 HUD とプリセット (task 809) の静的な結線の検査。
//!
//! denno-view はバイナリ crate なので、統合テストからは crate の中身を呼べない。
//! 代わりに「配信するアセットそのもの」を include_str! で読み、画面の DOM と
//! モジュールの export が噛み合っていることを確かめる (JS のロジックは
//! tests/hud2d.test.mjs が node --test で見る)。
//!
//! 見ているもの:
//!   1. 画面の id が重複しない / $('x') で参照する id が全部ある
//!   2. HUD 帯の 5 つの値を DOM に書き出す結線がある
//!   3. プリセットの export / import / apply の結線がある
//!   4. /hud2d.mjs が HUD とプリセットの関数を export している
//!   5. 構成図の描画に MTP の強調 (mtp:) が渡っている

const HTML: &str = include_str!("../src/index2d.html");
const HUD: &str = include_str!("../src/hud2d.mjs");

/// `prefix` の直後から `quote` までを集める (素朴な走査。HTML/JS の静的検査用)。
fn between(src: &str, prefix: &str, quote: char) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = src;
    while let Some(i) = rest.find(prefix) {
        let after = &rest[i + prefix.len()..];
        let Some(end) = after.find(quote) else { break };
        out.push(after[..end].to_string());
        rest = &after[end..];
    }
    out
}

fn ids(src: &str) -> Vec<String> {
    between(src, "id=\"", '"')
}

fn referenced_ids(src: &str) -> Vec<String> {
    between(src, "$('", '\'')
}

#[test]
fn page_ids_are_unique_and_every_dollar_id_exists() {
    let ids = ids(HTML);
    assert!(ids.len() > 100, "id が少なすぎる: {}", ids.len());
    let mut sorted = ids.clone();
    sorted.sort();
    for w in sorted.windows(2) {
        assert_ne!(w[0], w[1], "id が重複している: {}", w[0]);
    }
    let set: std::collections::HashSet<&str> = ids.iter().map(|s| s.as_str()).collect();
    let used = referenced_ids(HTML);
    assert!(used.len() > 50, "$('...') が少なすぎる: {}", used.len());
    for id in used {
        // $('t-' + which) のような動的な参照は接頭辞で受ける。
        let ok = set.contains(id.as_str()) || set.iter().any(|k| k.starts_with(id.as_str()));
        assert!(ok, "$('{id}') の id=\"{id}\" が DOM に無い");
    }
}

#[test]
fn hud_strip_is_wired_to_the_pure_module() {
    // 5 つの値の器 (task 809 の 1.)
    for id in ["hud", "hud-ttft", "hud-rate", "hud-vram", "hud-kv", "hud-mtp", "hudnote"] {
        assert!(HTML.contains(&format!("id=\"{id}\"")), "HUD の #{id} が無い");
    }
    // 画面は hud2d.mjs の状態機械を呼ぶだけ (数字を画面で作らない)
    for call in ["hudFields(hud)", "hudBeginRequest(hud", "hudEndRequest(hud", "hudOnToken(hud",
                 "hudOnStats(hud", "hudOnCache(hud", "hudOnFrame(hud", "hudOnRawKind(hud"] {
        assert!(HTML.contains(call), "HUD の結線が無い: {call}");
    }
    // 未知 kind のデコード失敗でも「投機が来た」ことを記録する
    assert!(HTML.contains("FrameDecodeError"), "frame のデコード失敗の経路が無い");
    assert!(HTML.contains("e.kindId"), "FrameDecodeError の kindId を見ていない");
}

#[test]
fn preset_export_import_apply_is_wired() {
    for id in ["presetbox", "preset-name", "preset-export", "preset-file", "preset-apply", "preset-clear",
               "preset-info", "preset-err"] {
        assert!(HTML.contains(&format!("id=\"{id}\"")), "preset の #{id} が無い");
    }
    for call in ["buildPreset(", "parsePreset(", "presetFileName(", "presetSummary(",
                 "validateInterveneBody(", "selectionToBody(sel)"] {
        assert!(HTML.contains(call), "preset の結線が無い: {call}");
    }
    // 読み込み → apply の 2 段 (ファイルを選んだだけではサーバーを変えない)
    assert!(HTML.contains("$('preset-apply').addEventListener"), "apply loaded の handler が無い");
    assert!(HTML.contains("a.download = presetFileName("), "export のダウンロードが無い");
}

#[test]
fn hud_module_exports_the_contract() {
    for name in ["createHud", "hudBeginRequest", "hudOnToken", "hudEndRequest", "hudOnStats",
                 "hudOnCache", "hudOnFrame", "hudOnRawKind", "hudFields", "buildPreset",
                 "parsePreset", "presetSummary", "presetFileName", "validateInterveneBody"] {
        assert!(HUD.contains(&format!("export function {name}(")), "hud2d.mjs に {name} が無い");
    }
    // kind 17 (ExpertSkip) を投機と取り違えない
    assert!(HUD.contains("SPEC_KIND_MIN"), "投機 kind の下限が無い");
    assert!(HUD.contains("import {") && HUD.contains("./view2d.mjs"), "view2d.mjs を使っていない");
}

#[test]
fn arch_draw_receives_the_mtp_highlight_flag() {
    assert!(HTML.contains("mtp: hud.mtp.seen ?"), "drawArch に mtp を渡していない");
    let arch = include_str!("../src/arch2d.mjs");
    assert!(arch.contains("D.mtp"), "drawArch が MTP の強調を見ていない");
    assert!(arch.contains("accept "), "MTP の採用率を描いていない");
    assert!(arch.contains("spec telemetry"), "数字が無いときの表示が無い");
}
