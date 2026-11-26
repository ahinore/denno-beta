// denno-view の全 .mjs モジュールの export 契約スナップショット (task 1254)。
// 目的: 事故的な rename / 削除 / 追加を検出して、UI 側の import 破壊を防ぐ。
//   - 各モジュールの export 名の完全一致 (並びはソート済み)
//   - 関数 / クラス / オブジェクト定数 / プリミティブ定数の分類と値
// ここを更新するときは「意図した API 変更」であることを確認してから直す。
// node --test crates/denno-view/tests/ で走る。DOM には触らない。
import test from 'node:test';
import assert from 'node:assert/strict';

import * as frames from '../src/frames.mjs';
import * as dash from '../src/dash.mjs';
import * as keys3d from '../src/keys3d.mjs';
import * as chat from '../src/chat.mjs';
import * as arch2d from '../src/arch2d.mjs';
import * as nodes2d from '../src/nodes2d.mjs';
import * as view2d from '../src/view2d.mjs';
import * as hud2d from '../src/hud2d.mjs';
// task 1106: 再接続バックオフ / 接続状態の純関数。
import * as reconnect from '../src/reconnect.mjs';

const MODULES = { frames, dash, keys3d, chat, arch2d, nodes2d, view2d, hud2d, reconnect };

// スナップショット: 期待する export 面。キーは Object.keys(ns).sort() と一致させる。
const SNAPSHOT = {
  frames: {
    fns: ["concatFrames", "decodeFrame", "decodeFrames", "encodeFrame", "f16ToF32", "f32ToF16", "kindName"],
    classes: ["FrameDecodeError"],
    objects: ["KIND", "ROLE_NAMES"],
    consts: {"MAGIC": "DNTL", "VERSION": 1},
  },
  dash: {
    fns: ["apiEndpoints", "createDash", "createRing", "csvCell", "curlFor", "dashRows", "filterItems", "fmtPercent", "fmtRate", "gaugeBar", "kvGauge", "latencyPath", "latencyPercentiles", "latencyRow", "latencySummary", "matchesFilter", "metricsCsv", "metricsJson", "metricsRows", "noteFrame", "noteKv", "noteLatency", "notePrefill", "noteResidency", "noteRoute", "noteTierStats", "noteToken", "percentile", "quickNav", "ringLast", "ringPush", "sparkPath", "sparkPoints", "step", "toCsv", "vramGauge"],
    classes: [],
    objects: ["API_ENDPOINTS", "EXPORT_COLUMNS", "FILTER_KEYS", "PERCENTILES", "QUICK_NAV", "SERIES_KEYS", "SERIES_LABELS"],
    consts: {"DASH_SCHEMA": 1, "DEFAULT_CAP": 90, "DEFAULT_WINDOW_MS": 1000, "GAUGE_WIDTH": 10, "MAX_LAYERS": 512, "MAX_RING_CAP": 65536, "SERIES_CAP": 120},
  },
  keys3d: {
    fns: ["activationStats", "beginLayerJump", "bookmarkForKey", "bookmarkGoal", "camAnimActive", "cancelLayerJump", "clampLayer", "clampThemeIndex", "closeHelp", "createCamAnim", "createFpsMeter", "createHelpState", "createLayerJump", "createMotion", "cycleTheme", "digitFromKey", "easeSmoothstep", "effectiveReduced", "expertSummary", "fitCameraGoal", "formatExpertSummary", "formatHelpRow", "formatMagnitude", "fpsBars", "fpsSummary", "fpsText", "helpOpen", "helpRows", "hexToRgb", "isAutoFitKey", "isFpsKey", "isHelpCloseKey", "isHelpKey", "isLayerJumpKey", "isLayerStepKey", "isLegendKey", "isMotionKey", "isSnapshotKey", "isThemeKey", "jumpToLayer", "layerKind", "legendFromQuery", "legendRows", "noteFrame", "orbitEnabled", "overlayFromQuery", "parseLayerDigits", "presetForKey", "presetGoal", "pushLayerDigit", "rgbToHex", "sceneBounds", "setEnvReduced", "setHelpOpen", "snapshotFilename", "startCamAnim", "stepCamAnim", "stepLayer", "themeAt", "themeCssVars", "themeHexAt", "themeIndex", "themeName", "themeRgbAt", "toggleHelp", "toggleLegend", "toggleReducedMotion"],
    classes: [],
    objects: ["BOOKMARKS", "BOOKMARK_KEYS", "CAMERA_PRESETS", "FIT_BOUNDS", "FIT_DIR", "PRESET_KEYS", "SHORTCUTS", "THEMES"],
    consts: {"ATTENTION_INTERVAL": 4, "LAYER_COUNT": 48, "LAYER_JUMP_MAX_DIGITS": 3, "THEME_COUNT": 3},
  },
  chat: {
    fns: ["applyChatEvent", "buildChatBody", "chatBegin", "chatEnd", "chatFields", "chatOnToken", "chatRate", "clampMaxTokens", "clampTemperature", "createChatState", "createSseParser", "fmtMs", "fmtRate", "normaliseControls", "normaliseHistory", "parseSseBlock", "splitViewUrl", "systemPreset"],
    classes: [],
    objects: ["CHAT_DEFAULTS", "SYSTEM_PRESETS"],
    consts: {"CHAT_SCHEMA": 1, "CHAT_URL": "/chat", "MAX_TOKENS_CLAMP": 4096, "SPLIT_2D_URL": "/2d?panes=map", "TEMPERATURE_MAX": 2},
  },
  arch2d: {
    fns: ["archExpertRect", "archHitTest", "archItemsInRect", "archLayout", "archNodeRect", "drawArch", "drillSparkBars", "expertDrilldown", "expertFrequency", "fitView", "gridPixels", "streamCircle", "toWorld", "zoomAt"],
    classes: [],
    objects: ["ARCH", "ARCH_COLORS", "DRILL_COLORS", "PANEL"],
    consts: {},
  },
  nodes2d: {
    fns: ["ancestorIds", "buildTree", "createEnv", "dslSummary", "nodeCheckState", "nodeItems", "parseDsl", "runDsl", "tokenize", "visibleRows"],
    classes: ["DslError"],
    objects: ["DSL_ATTRS", "DSL_FUNCS", "GROUP_TYPES", "LEAF_TYPES"],
    consts: {"HYBRID_SIZE": 4},
  },
  view2d: {
    fns: ["ablitApplyBody", "ablitChart", "ablitCompareRows", "ablitEvalProgress", "ablitEvalPrompts", "ablitEvalRows", "ablitEvalTable", "ablitExportName", "ablitExportUrl", "ablitHybridValue", "ablitImportBody", "ablitLayerAt", "ablitRange", "ablitRangeFromDrag", "ablitRecipeBodyFromUi", "ablitRecipeFileName", "ablitRecipeFromText", "ablitRecipeObject", "ablitRecipeSummaryText", "ablitRecommendedRange", "ablitRecordBody", "ablitRefusalLike", "ablitSepScores", "ablitSnippet", "ablitStackBody", "ablitStackSummary", "ablitSummary", "actBuckets", "actLevel", "addItem", "clearSelection", "createMapState", "createMockStore", "createReplay", "createSelection", "createTrace", "depthValue", "entropyBits", "entropyColor", "expertCellRect", "expertCoverageText", "expertStatsView", "expertTierText", "fitExpertLimit", "hasItem", "heat", "hitTest", "itemsInRect", "layoutFromShape", "mapPaneSize", "mergeBodies", "mixerKind", "mockTokenFrames", "normaliseShape", "parsePromptList", "rampRGB", "removeItem", "replaySeek", "replayStateAt", "replayStep", "replayTokenAt", "replayTokenCount", "residencyBits", "roleForBlock", "routerWeights", "selectionCount", "selectionFromBody", "selectionItems", "selectionToBody", "skipBits", "steerApplyBody", "steerSummary", "streamValue", "toggleItem", "topkBarRows", "topkFrameView", "topkPush", "topkSummary", "traceBytes", "traceDocument", "traceFileName", "traceFrame", "traceMemory", "traceParse", "traceRecord", "traceSerialize", "traceStart", "traceStop", "traceSummary", "traceValidate", "wallExpertRect", "wallHitTest", "wallItemsInRect", "wallLayout"],
    classes: [],
    objects: ["BLOCKS", "CAPTURE_MODES", "DEFAULT_SHAPE", "DEFAULT_SHAPE_RAW", "GEOM", "MAP_CHROME", "STEER_MODES", "WALL_COLORS", "WALL_GEOM"],
    consts: {"ABLIT_A": "ablit_A", "ABLIT_ALPHA_MAX": 2, "ABLIT_ALPHA_MIN": 0, "ABLIT_B": "ablit_B", "ABLIT_CAPTURE_MODE": "last_prompt_token", "ABLIT_EVAL_MAX": 10, "ABLIT_EVAL_SNIPPET": 120, "ABLIT_STACK_MAX": 4, "ACT_BUCKETS": 64, "DRILL_MAX_ROWS": 16, "EXPERT_FREQ_WINDOW": 16, "RECIPE_MAX_TERMS": 16, "RECIPE_VERSION": 1, "TOPK_HISTORY_MAX": 16, "TRACE_MAX_ARRAY": 256, "TRACE_MAX_FRAMES": 2048, "TRACE_MAX_PAIRS": 128, "TRACE_SCHEMA": "denno-trace", "TRACE_SUFFIX": ".denno-trace.json", "TRACE_VERSION": 1, "VIEW_SCHEMA": 1},
  },
  hud2d: {
    fns: ["buildPreset", "checkedBody", "createHud", "fmtDuration", "fmtPercent", "fmtRate", "fmtTokens", "hudBeginRequest", "hudEndRequest", "hudFields", "hudOnCache", "hudOnFrame", "hudOnRawKind", "hudOnStats", "hudOnToken", "hudText", "isSpecKind", "parsePreset", "presetFileName", "presetSummary", "specSample", "validateInterveneBody"],
    classes: [],
    objects: ["HUD_LABELS", "SPEC_FRAME_KINDS"],
    consts: {"HUD_SCHEMA": 1, "PRESET_KIND": "denno-view/preset", "PRESET_SCHEMA": 1, "SPEC_KIND_MIN": 18},
  },
  reconnect: {
    fns: ["backoffDelay", "clampAttempt", "connectionBanner", "connectionStatus", "createReconnect", "formatRetry", "isOffline", "normaliseOptions"],
    classes: [],
    objects: ["RECONNECT_DEFAULTS"],
    consts: {"RECONNECT_SCHEMA": 1},
  },
};

const MODULE_NAMES = Object.keys(SNAPSHOT);

function namesOf(ns) {
  return Object.keys(ns).sort();
}

test('スナップショット対象は全 9 モジュールを網羅する', () => {
  assert.deepEqual([...MODULE_NAMES].sort(), ['arch2d', 'chat', 'dash', 'frames', 'hud2d', 'keys3d', 'nodes2d', 'reconnect', 'view2d']);
});

for (const name of MODULE_NAMES) {
  const spec = SNAPSHOT[name];
  const expectedAll = [...spec.fns, ...spec.classes, ...spec.objects, ...Object.keys(spec.consts)].sort();

  test(`${name}: export 名の完全一致 (rename / 削除 / 追加を検出)`, () => {
    assert.deepEqual(namesOf(MODULES[name]), expectedAll);
  });

  test(`${name}: export 数がスナップショットと一致する`, () => {
    assert.equal(namesOf(MODULES[name]).length, expectedAll.length);
    assert.equal(new Set(expectedAll).size, expectedAll.length, 'スナップショット内で名前が重複している');
  });

  test(`${name}: 関数 export の面`, () => {
    const actual = namesOf(MODULES[name]).filter((k) => typeof MODULES[name][k] === 'function'
      && !/^class\s/.test(Function.prototype.toString.call(MODULES[name][k])));
    assert.deepEqual(actual, spec.fns);
  });

  test(`${name}: クラス export の面`, () => {
    const actual = namesOf(MODULES[name]).filter((k) => typeof MODULES[name][k] === 'function'
      && /^class\s/.test(Function.prototype.toString.call(MODULES[name][k])));
    assert.deepEqual(actual, spec.classes);
  });

  test(`${name}: オブジェクト定数 export の面`, () => {
    const actual = namesOf(MODULES[name]).filter((k) => {
      const v = MODULES[name][k];
      return typeof v !== 'function' && (v === null || typeof v === 'object');
    });
    assert.deepEqual(actual, spec.objects);
  });

  test(`${name}: プリミティブ定数の名前と値`, () => {
    const actual = {};
    for (const k of namesOf(MODULES[name])) {
      const v = MODULES[name][k];
      if (typeof v !== 'function' && (v === null || typeof v !== 'object')) actual[k] = v;
    }
    assert.deepEqual(actual, spec.consts);
  });

  test(`${name}: export に undefined が混ざっていない`, () => {
    for (const k of namesOf(MODULES[name])) {
      assert.notEqual(MODULES[name][k], undefined, `${name}.${k} が undefined`);
    }
  });
}

test('総 export 数がスナップショットで固定される', () => {
  const total = MODULE_NAMES.reduce((n, m) => n + namesOf(MODULES[m]).length, 0);
  // 352 + reconnect.mjs の 10 export (task 1106)
  assert.equal(total, 362);
});
