//! task 904: /3d の層インスペクタ + カメラプリセット (1..5) + fps オーバーレイの
//! 静的な結線の検査。
//!
//! denno-view はバイナリ crate なので統合テストから crate の中身は呼べない。
//! 配信するアセットそのものを include_str! で読み、DOM の id と module の
//! export / 画面からの呼び出しが噛み合っていることを確かめる。数値のロジックは
//! tests/keys3d904.test.mjs が node --test で見る。

const INDEX3D: &str = include_str!("../src/index3d.html");
const KEYS3D: &str = include_str!("../src/keys3d.mjs");

#[test]
fn inspector_panel_has_all_the_slots() {
    for id in [
        "insp", "insp_idx", "insp_kind", "insp_exp", "insp_mag", "insp_bar",
    ] {
        assert!(
            INDEX3D.contains(&format!("id=\"{id}\"")),
            "層インスペクタに #{id} が無い"
        );
    }
    // 見出しに操作の導線 (クリック / [ ]) を出す
    assert!(INDEX3D.contains("layer inspector"), "インスペクタの見出しが無い");
    assert!(INDEX3D.contains("[ ]"), "キー操作の案内が無い");
}

#[test]
fn inspector_wires_the_pure_module() {
    for call in [
        "layerKind(il, INTERVAL)",
        "clampLayer(il)",
        "stepLayer(insp.sel, dstep, LAYERS)",
        "activationStats(w ? w.act : [])",
        "formatMagnitude(st)",
        "expertSummary(route, weights, 3)",
        "formatExpertSummary(",
        "isLayerStepKey(e)",
        "isFpsKey(e)",
        "overlayFromQuery(Q.get('fps'))",
        "selectLayer(",
        "updateInspector()",
    ] {
        assert!(INDEX3D.contains(call), "インスペクタの結線が無い: {call}");
    }
    // バーをクリックしたら層を選ぶ
    assert!(
        INDEX3D.contains("cell.addEventListener('click'"),
        "層バーのクリックが結線されていない"
    );
    // 選択層は白枠とは別に赤枠で示す (現在層と選択層を混同しない)
    assert!(INDEX3D.contains("'pick'"), "選択層のハイライトが無い");
    // ?insp=N で初期選択を固定できる (スクリーンショット検証用)
    assert!(INDEX3D.contains("Q.get('insp')"), "?insp= の入口が無い");
}

#[test]
fn camera_presets_and_smooth_interpolation_are_wired() {
    for call in [
        "presetForKey(e.key)",
        "presetGoal(preset)",
        "startCamAnim(camAnim, goal",
        "stepCamAnim(camAnim, dt)",
        "camAnimActive(camAnim)",
        "createCamAnim(0.45)",
    ] {
        assert!(INDEX3D.contains(call), "カメラプリセットの結線が無い: {call}");
    }
    // 補間の結果を camera と controls.target の両方に入れる
    assert!(INDEX3D.contains("camera.position.set(g.pos[0]"), "補間位置が camera に入っていない");
    assert!(INDEX3D.contains("controls.target.set(g.target[0]"), "補間注視点が controls に入っていない");
    // 813 のブックマーク経路は残す (既存の ?bm= と凡例の互換)
    assert!(INDEX3D.contains("bookmarkForKey(e.key)"), "813 のブックマーク経路が消えている");
}

#[test]
fn fps_overlay_is_toggleable_and_self_contained() {
    for id in ["fpsov", "fpsov_v", "fpsov_hist"] {
        assert!(
            INDEX3D.contains(&format!("id=\"{id}\"")),
            "fps オーバーレイに #{id} が無い"
        );
    }
    for call in [
        "toggleFpsOverlay()",
        "setFpsOverlay(",
        "paintFpsOverlay(",
        "noteFrame(fpsMeter, performance.now())",
        "fpsText(sum)",
        "fpsBars(sum, 24)",
    ] {
        assert!(INDEX3D.contains(call), "fps オーバーレイの結線が無い: {call}");
    }
    // 外部 CDN を足さない (importmap はローカルの vendor だけ)
    assert!(
        !INDEX3D.contains("cdn.") && !INDEX3D.contains("unpkg.com") && !INDEX3D.contains("jsdelivr"),
        "fps オーバーレイに外部 CDN が入っている"
    );
    assert!(INDEX3D.contains("/vendor/three.module.js"), "three がローカル vendor でない");
}

#[test]
fn keys3d_module_exports_the_904_contract() {
    for name in [
        "presetForKey", "presetGoal", "easeSmoothstep",
        "createCamAnim", "startCamAnim", "stepCamAnim", "camAnimActive",
        "layerKind", "clampLayer", "stepLayer",
        "activationStats", "formatMagnitude",
        "expertSummary", "formatExpertSummary",
        "isLayerStepKey", "isFpsKey", "overlayFromQuery",
        "createFpsMeter", "noteFrame", "fpsSummary", "fpsText", "fpsBars",
    ] {
        assert!(
            KEYS3D.contains(&format!("export function {name}(")),
            "keys3d.mjs に {name} が無い"
        );
    }
    for name in ["CAMERA_PRESETS", "PRESET_KEYS", "ATTENTION_INTERVAL"] {
        assert!(KEYS3D.contains(&format!("export const {name}")), "keys3d.mjs に {name} が無い");
    }
    // 5 つのプリセット名 (要件の並び)
    for name in ["Overview", "Front", "Top", "Side", "Reset"] {
        assert!(KEYS3D.contains(&format!("name: '{name}'")), "プリセット {name} が無い");
    }
    // 813 の契約はそのまま
    for name in ["BOOKMARKS", "BOOKMARK_KEYS", "SHORTCUTS"] {
        assert!(KEYS3D.contains(&format!("export const {name}")), "既存の {name} が消えている");
    }
    for name in ["bookmarkForKey", "bookmarkGoal", "legendRows", "toggleLegend", "isLegendKey", "legendFromQuery"] {
        assert!(
            KEYS3D.contains(&format!("export function {name}(")),
            "既存の {name} が消えている"
        );
    }
}

#[test]
fn no_debug_output_in_the_viewer_assets() {
    for (name, src) in [("index3d.html", INDEX3D), ("keys3d.mjs", KEYS3D)] {
        assert!(!src.contains("console.log"), "{name} に console.log が残っている");
        assert!(!src.contains("debugger"), "{name} に debugger が残っている");
    }
}
