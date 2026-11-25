//! /2d の層ドリルダウン (task 857) の静的な結線の検査。
//!
//! ロジックは node --test (tests/drill2d.test.mjs) が見る。ここは
//! 「画面のアセットが純ロジックと噛み合っているか」だけを確かめる。
//!
//! 見ているもの:
//!   1. ドリルダウンの器 (id) が DOM にある
//!   2. 層のクリック (構成図の column / panel、層の壁) から開く
//!   3. 画面は arch2d.mjs の expertDrilldown / drillSparkBars を呼ぶだけ
//!   4. view2d.mjs が頻度用の履歴 (hist / raw / residencySeen) を保つ
//!   5. × と Esc で閉じる

const HTML: &str = include_str!("../src/index2d.html");
const ARCH: &str = include_str!("../src/arch2d.mjs");
const VIEW: &str = include_str!("../src/view2d.mjs");

#[test]
fn drill_container_ids_exist() {
    for id in ["drill", "drill-title", "drill-sub", "drill-sum", "drill-body", "drill-note", "drill-close"] {
        assert!(HTML.contains(&format!("id=\"{id}\"")), "ドリルダウンの #{id} が無い");
    }
    assert!(HTML.contains("class=\"drill\""), "#drill に .drill が付いていない");
    assert!(HTML.contains(".drill {"), "ドリルダウンの CSS が無い");
    assert!(HTML.contains("id=\"drill\" class=\"drill\" hidden"), "初期状態で閉じていない");
}

#[test]
fn layer_clicks_open_the_drilldown() {
    // 構成図: 層の箱 (column) と層パネル (panel)
    assert!(HTML.contains("openDrill(h.layer)"), "層のクリックで開かない");
    assert!(HTML.contains("h.type === 'column'"), "column の分岐が無い");
    assert!(HTML.contains("h.type === 'panel'"), "panel の分岐が無い");
    // 層の壁 (wall) でも同じ
    assert!(HTML.contains("if (d.hit.layer != null) openDrill(d.hit.layer)"),
        "wall のクリックで開かない");
    // 閉じる: × と Esc
    assert!(HTML.contains("$('drill-close').addEventListener"), "× の handler が無い");
    assert!(HTML.contains("if (e.key === 'Escape') { closeDrill();"), "Esc で閉じない");
}

#[test]
fn page_uses_the_pure_drilldown_module() {
    assert!(HTML.contains("expertDrilldown"), "expertDrilldown を呼んでいない");
    assert!(HTML.contains("drillSparkBars"), "スパークラインのデータを作っていない");
    assert!(HTML.contains("EXPERT_FREQ_WINDOW"), "頻度の窓を画面が持っていない");
    assert!(HTML.contains("DRILL_COLORS"), "hit / miss の色を画面が持っていない");
    // 4 Hz で描き直す (1 トークンで 48 層ぶんの RouterTopK が来る)
    assert!(HTML.contains("now - lastDrillRender > 250"), "描き直しの間引きが無い");
    // 検証用フック
    assert!(HTML.contains("openDrill: (l) =>"), "検証用フックが無い");
    assert!(HTML.contains("drill: () =>"), "検証用フック (drill) が無い");
}

#[test]
fn arch2d_exports_the_drilldown_contract() {
    for name in ["expertDrilldown", "expertFrequency", "drillSparkBars"] {
        assert!(ARCH.contains(&format!("export function {name}(")), "arch2d.mjs に {name} が無い");
    }
    assert!(ARCH.contains("export const DRILL_COLORS"), "DRILL_COLORS が無い");
    assert!(ARCH.contains("EXPERT_FREQ_WINDOW") && ARCH.contains("DRILL_MAX_ROWS"),
        "arch2d.mjs が view2d.mjs の窓 / 上限を使っていない");
}

#[test]
fn view2d_keeps_the_frequency_history() {
    for name in ["EXPERT_FREQ_WINDOW", "DRILL_MAX_ROWS"] {
        assert!(VIEW.contains(&format!("export const {name}")), "view2d.mjs に {name} が無い");
    }
    // 状態: 生の重み / 履歴 / residency の受信フラグ
    for field in ["raw: []", "hist: []", "residencySeen: []"] {
        assert!(VIEW.contains(field), "createMapState に {field} が無い");
    }
    assert!(VIEW.contains("st.hist[l].push(top.slice())"), "トークンごとに履歴を積んでいない");
    assert!(VIEW.contains("st.residencySeen[l] = true"), "ExpertResidency の受信を記録していない");
}
