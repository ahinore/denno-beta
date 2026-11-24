//! task 813: ダッシュボード (/) のライブ計器 + API 案内と、3D ビューワ (/3d) の
//! カメラブックマーク + `?` 凡例の静的な結線の検査。
//!
//! denno-view はバイナリ crate なので統合テストから crate の中身は呼べない。
//! 配信するアセットそのものを include_str! で読み、DOM の id と module の
//! export / 画面からの呼び出しが噛み合っていることを確かめる。数値のロジックは
//! tests/dash813.test.mjs と tests/keys3d813.test.mjs が node --test で見る。

const INDEX: &str = include_str!("../src/index.html");
const INDEX3D: &str = include_str!("../src/index3d.html");
const DASH: &str = include_str!("../src/dash.mjs");
const KEYS3D: &str = include_str!("../src/keys3d.mjs");

#[test]
fn dashboard_has_the_three_sparklines_and_api_box() {
    for id in [
        "rates", "rate-decode", "rate-decode-v", "rate-decode-s", "rate-decode-p", "rate-decode-n",
        "rate-prefill", "rate-prefill-v", "rate-prefill-s", "rate-prefill-p", "rate-prefill-n",
        "rate-vram", "rate-vram-v", "rate-vram-s", "rate-vram-p", "rate-vram-n",
        "apis", "ep-openai", "ep-anthropic", "curl", "copy-note",
    ] {
        assert!(
            INDEX.contains(&format!("id=\"{id}\"")),
            "ダッシュボードに #{id} が無い"
        );
    }
    // 3 本のスパークラインは SVG path に書く (canvas ではない)
    assert_eq!(INDEX.matches("<svg class=\"spark\"").count(), 3, "sparkline が 3 本でない");
    // コピーの結線: 行ごとのボタン + curl のボタン
    for target in ["data-copy=\"ep-openai\"", "data-copy=\"ep-anthropic\"", "data-copy=\"curl\""] {
        assert!(INDEX.contains(target), "copy ボタンの結線が無い: {target}");
    }
}

#[test]
fn dashboard_wires_the_pure_module() {
    assert!(INDEX.contains("from \"/dash.mjs\""), "dash.mjs を import していない");
    for call in [
        "window.__dashAttach(",
        "noteFrame(dash, f, t)",
        "step(dash, performance.now())",
        "dashRows(dash, SPARK)",
        "apiEndpoints(origin)",
        "curlFor(\"anthropic\", origin, model)",
        "setAttribute(\"d\", r.path)",
    ] {
        assert!(INDEX.contains(call), "ダッシュボードの結線が無い: {call}");
    }
    // WS の受信でフレームを計器へ流す (アニメーションのキューとは別経路)
    assert!(INDEX.contains("dashSend(f, performance.now())"), "WS から計器への入口が無い");
    // 受信側のデコーダが計器に要る kind を読める
    for needle in [
        "case 14: { // ExpertResidency",
        "case 15: { // TierStats",
        "case 16: { // RouterTopK",
        "case 17: { // ExpertSkip",
        "const ROLE_NAMES =",
        "kind: \"Unknown\", kindId: kind",
    ] {
        assert!(INDEX.contains(needle), "デコーダに {needle} が無い");
    }
    // 未知 kind で throw しない (投機テレメトリが来ても計器が止まらない)
    assert!(
        !INDEX.contains("default: throw new Error(\"unknown kind:"),
        "未知 kind で throw している"
    );
}

#[test]
fn dash_module_exports_the_contract() {
    for name in [
        "createDash", "notePrefill", "noteToken", "step", "noteResidency", "noteRoute",
        "noteTierStats", "noteFrame", "fmtRate", "fmtPercent", "sparkPoints", "sparkPath",
        "dashRows", "apiEndpoints", "curlFor",
    ] {
        assert!(
            DASH.contains(&format!("export function {name}(")),
            "dash.mjs に {name} が無い"
        );
    }
    for name in ["SERIES_KEYS", "SERIES_LABELS", "API_ENDPOINTS", "DASH_SCHEMA"] {
        assert!(DASH.contains(&format!("export const {name}")), "dash.mjs に {name} が無い");
    }
    // OpenAI と Anthropic の両方の経路を出す
    assert!(DASH.contains("/v1/chat/completions"), "OpenAI の URL が無い");
    assert!(DASH.contains("/v1/messages"), "Anthropic の URL が無い");
}

#[test]
fn viewer3d_has_bookmarks_and_toggleable_legend() {
    for id in ["keys", "keys-rows", "camlabel"] {
        assert!(
            INDEX3D.contains(&format!("id=\"{id}\"")),
            "3D ビューワに #{id} が無い"
        );
    }
    assert!(INDEX3D.contains("from '/keys3d.mjs'"), "keys3d.mjs を import していない");
    for call in [
        "bookmarkForKey(e.key)",
        "bookmarkGoal(bm)",
        "camera.position.set(goal.pos[0]",
        "controls.target.set(goal.target[0]",
        "isLegendKey(e)",
        "toggleLegendUI()",
        "legendRows()",
        "legendFromQuery(Q.get('legend'))",
    ] {
        assert!(INDEX3D.contains(call), "3D ビューワの結線が無い: {call}");
    }
    // 1 キーで飛ぶ 4 つの見どころが凡例の文言にも出る
    for name in ["Overview", "Router / Experts", "Prediction Head", "KV Band"] {
        assert!(KEYS3D.contains(name), "keys3d.mjs に {name} が無い");
    }
    // `?` の扱い (Shift+/ も受ける)
    assert!(KEYS3D.contains("e.key === '?'"), "keys3d.mjs が ? を受けていない");
    assert!(KEYS3D.contains("shiftKey"), "keys3d.mjs が Shift+/ を見ていない");
}

#[test]
fn keys3d_module_exports_the_contract() {
    for name in [
        "bookmarkForKey", "bookmarkGoal", "legendRows", "toggleLegend", "isLegendKey",
        "legendFromQuery",
    ] {
        assert!(
            KEYS3D.contains(&format!("export function {name}(")),
            "keys3d.mjs に {name} が無い"
        );
    }
    for name in ["BOOKMARKS", "BOOKMARK_KEYS", "SHORTCUTS"] {
        assert!(KEYS3D.contains(&format!("export const {name}")), "keys3d.mjs に {name} が無い");
    }
}
