// keys3d.mjs (task 1174) の全エクスポートを対象にした単体試験。
// 1. 公開定数が再帰的に凍結されていて、読み出しで変化しない (不変性)
// 2. 3 テーマのトークンが hex / ramp の契約を満たす (色の整合)
// 3. カメラ補間 / 層 / fps / snapshot / motion / 凡例 / 層ジャンプ の境界値
// node --test crates/denno-view/tests/*.test.mjs で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import * as K from '../src/keys3d.mjs';
import {
  BOOKMARKS, BOOKMARK_KEYS, CAMERA_PRESETS, PRESET_KEYS, SHORTCUTS, THEMES,
  THEME_COUNT, ATTENTION_INTERVAL, LAYER_COUNT, FIT_BOUNDS, FIT_DIR,
  LAYER_JUMP_MAX_DIGITS,
  bookmarkForKey, bookmarkGoal, presetForKey, presetGoal,
  easeSmoothstep, createCamAnim, startCamAnim, stepCamAnim, camAnimActive,
  layerKind, clampLayer, stepLayer, activationStats, formatMagnitude,
  expertSummary, formatExpertSummary, isLayerStepKey, isFpsKey, overlayFromQuery,
  createFpsMeter, noteFrame, fpsSummary, fpsText, fpsBars,
  legendRows, toggleLegend, isLegendKey, legendFromQuery,
  clampThemeIndex, themeAt, themeIndex, cycleTheme, themeName, isThemeKey,
  hexToRgb, rgbToHex, themeRgbAt, themeHexAt, themeCssVars,
  snapshotFilename, isSnapshotKey,
  createMotion, effectiveReduced, setEnvReduced, toggleReducedMotion, orbitEnabled, isMotionKey,
  createHelpState, helpOpen, setHelpOpen, toggleHelp, closeHelp, isHelpKey, isHelpCloseKey,
  helpRows, formatHelpRow,
  sceneBounds, fitCameraGoal, isAutoFitKey,
  createLayerJump, beginLayerJump, cancelLayerJump, isLayerJumpKey, digitFromKey,
  parseLayerDigits, pushLayerDigit, jumpToLayer,
} from '../src/keys3d.mjs';

const HEX6 = /^#[0-9a-f]{6}$/;

// このファイルが網羅する公開 API (export 名の完全一致を守るための一覧)。
const EXPORTED = [
  'ATTENTION_INTERVAL', 'BOOKMARKS', 'BOOKMARK_KEYS', 'CAMERA_PRESETS', 'FIT_BOUNDS',
  'FIT_DIR', 'LAYER_COUNT', 'LAYER_JUMP_MAX_DIGITS', 'PRESET_KEYS', 'SHORTCUTS',
  'THEMES', 'THEME_COUNT', 'activationStats', 'beginLayerJump', 'bookmarkForKey',
  'bookmarkGoal', 'camAnimActive', 'cancelLayerJump', 'clampLayer', 'clampThemeIndex',
  'closeHelp', 'createCamAnim', 'createFpsMeter', 'createHelpState', 'createLayerJump',
  'createMotion', 'cycleTheme', 'digitFromKey', 'easeSmoothstep', 'effectiveReduced',
  'expertSummary', 'fitCameraGoal', 'formatExpertSummary', 'formatHelpRow',
  'formatMagnitude', 'fpsBars', 'fpsSummary', 'fpsText', 'helpOpen', 'helpRows',
  'hexToRgb', 'isAutoFitKey', 'isFpsKey', 'isHelpCloseKey', 'isHelpKey',
  'isLayerJumpKey', 'isLayerStepKey', 'isLegendKey', 'isMotionKey', 'isSnapshotKey',
  'isThemeKey', 'jumpToLayer', 'layerKind', 'legendFromQuery', 'legendRows', 'noteFrame',
  'orbitEnabled', 'overlayFromQuery', 'parseLayerDigits', 'presetForKey', 'presetGoal',
  'pushLayerDigit', 'rgbToHex', 'sceneBounds', 'setEnvReduced', 'setHelpOpen',
  'snapshotFilename', 'startCamAnim', 'stepCamAnim', 'stepLayer', 'themeAt',
  'themeCssVars', 'themeHexAt', 'themeIndex', 'themeName', 'themeRgbAt', 'toggleHelp',
  'toggleLegend', 'toggleReducedMotion',
];

// ===== エクスポート面の完全一致 ==========================================
test('keys3d.mjs のエクスポートは既知の一覧と完全一致する', () => {
  assert.deepEqual(Object.keys(K).sort(), EXPORTED.slice().sort());
  for (const name of EXPORTED) assert.ok(name in K, name);
});

// ===== 不変性: 定数は再帰的に凍結される ==================================
test('公開定数は再帰的に凍結され、書き換えは TypeError になる', () => {
  for (const arr of [BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES, BOOKMARK_KEYS, PRESET_KEYS]) {
    assert.ok(Object.isFrozen(arr), 'array not frozen');
    assert.throws(() => { arr.push([0, 0, 0]); }, TypeError);
  }
  assert.ok(Object.isFrozen(FIT_BOUNDS));
  assert.ok(Object.isFrozen(FIT_DIR));
  assert.throws(() => { FIT_BOUNDS.fov = 1; }, TypeError);
  assert.throws(() => { FIT_DIR[0] = 9; }, TypeError);

  for (const arr of [BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES]) {
    for (const item of arr) {
      assert.ok(Object.isFrozen(item), 'item not frozen: ' + JSON.stringify(item));
      assert.throws(() => { item.key = '__x__'; }, TypeError);
    }
  }
  for (const bm of [...BOOKMARKS, ...CAMERA_PRESETS]) {
    assert.ok(Object.isFrozen(bm.pos), JSON.stringify(bm.name));
    assert.ok(Object.isFrozen(bm.target), JSON.stringify(bm.name));
    assert.throws(() => { bm.pos[0] = 999; }, TypeError);
    assert.throws(() => { bm.target[2] = 999; }, TypeError);
  }
  for (const th of THEMES) {
    assert.ok(Object.isFrozen(th.ramp), th.name);
    assert.throws(() => { th.ramp[0] = '#123456'; }, TypeError);
  }
});

test('定数は読み出しヘルパーを何度呼んでも変化しない', () => {
  const snap = JSON.stringify({ BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES, FIT_BOUNDS, FIT_DIR });
  for (let i = 0; i < 5; i++) {
    bookmarkForKey('2');
    presetForKey('3');
    legendRows();
    helpRows();
    themeAt(i);
    themeCssVars(i);
    assert.equal(JSON.stringify({ BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES, FIT_BOUNDS, FIT_DIR }), snap);
  }
});

test('FIT_DIR は長さ 1 の有限な単位ベクトル', () => {
  assert.equal(FIT_DIR.length, 3);
  for (const v of FIT_DIR) assert.ok(Number.isFinite(v));
  assert.ok(Math.abs(Math.hypot(FIT_DIR[0], FIT_DIR[1], FIT_DIR[2]) - 1) < 1e-9);
});

// ===== テーマトークンの契約 ==============================================
test('3 テーマの bg/fg/dim/accent と ramp 5 段はすべて小文字の #rrggbb', () => {
  assert.equal(THEME_COUNT, 3);
  assert.equal(THEME_COUNT, THEMES.length);
  assert.deepEqual(THEMES.map((t) => t.name), ['cyber', 'thermal', 'mono-high-contrast']);
  for (const th of THEMES) {
    for (const k of ['bg', 'fg', 'dim', 'accent']) {
      assert.match(th[k], HEX6, th.name + ' ' + k);
      assert.equal(th[k], th[k].toLowerCase(), th.name + ' ' + k);
    }
    assert.equal(th.ramp.length, 5, th.name + ' ramp');
    for (const c of th.ramp) assert.match(c, HEX6, th.name + ' ramp');
    // 前景と背景は同一でない (最低限の可読性)
    assert.notEqual(th.fg, th.bg, th.name);
  }
  // mono テーマは純白 / 純黒でハイコントラスト
  const mono = THEMES[2];
  assert.equal(mono.bg, '#000000');
  assert.equal(mono.fg, '#ffffff');
});

test('themeCssVars は accent を --vram に写し、bg/fg/dim をそのまま出す', () => {
  for (let i = 0; i < THEME_COUNT; i++) {
    const v = themeCssVars(i);
    assert.deepEqual(Object.keys(v).sort(), ['--bg', '--dim', '--fg', '--vram']);
    assert.equal(v['--bg'], THEMES[i].bg);
    assert.equal(v['--fg'], THEMES[i].fg);
    assert.equal(v['--dim'], THEMES[i].dim);
    assert.equal(v['--vram'], THEMES[i].accent);
  }
  // 添字は範囲を回り込む (0 と THEME_COUNT は同じテーマ)
  assert.deepEqual(themeCssVars(0), themeCssVars(THEME_COUNT));
  assert.deepEqual(themeCssVars(-1), themeCssVars(THEME_COUNT - 1));
});

test('themeAt / themeName / cycleTheme は 3 テーマを巡回し、常に有効なテーマを返す', () => {
  for (let i = 0; i < THEME_COUNT; i++) {
    assert.equal(clampThemeIndex(i), i);
    assert.equal(themeAt(i), THEMES[i]);
    assert.equal(themeName(i), THEMES[i].name);
  }
  for (const raw of [THEME_COUNT, 7, -1, -THEME_COUNT - 2, 1.9]) {
    const i = clampThemeIndex(raw);
    assert.ok(i >= 0 && i < THEME_COUNT, 'raw=' + raw + ' i=' + i);
    assert.ok(THEMES.includes(themeAt(raw)), 'raw=' + raw);
  }
  assert.equal(clampThemeIndex(NaN), 0);
  assert.equal(clampThemeIndex(1.9), 1);
  assert.equal(themeAt(-1).name, 'mono-high-contrast');
  assert.equal(themeAt(7).name, 'thermal');
  assert.equal(cycleTheme(0), 1);
  assert.equal(cycleTheme(1), 2);
  assert.equal(cycleTheme(2), 0);
  assert.equal(cycleTheme(-1), 0);
});

test('themeIndex は前後空白と大小文字を無視し、未知は -1', () => {
  assert.equal(themeIndex('cyber'), 0);
  assert.equal(themeIndex('Thermal'), 1);
  assert.equal(themeIndex(' MONO-HIGH-CONTRAST '), 2);
  for (const bad of ['nope', '', null, undefined, 42]) {
    assert.equal(themeIndex(bad), -1, String(bad));
  }
});

test('themeRgbAt / themeHexAt は 0..1 を ramp の両端へ張り付け、外側を clamp する', () => {
  for (let i = 0; i < THEME_COUNT; i++) {
    const first = hexToRgb(THEMES[i].ramp[0]);
    const last = hexToRgb(THEMES[i].ramp[4]);
    assert.deepEqual(themeRgbAt(i, 0), first, 'theme ' + i);
    assert.deepEqual(themeRgbAt(i, 1), last, 'theme ' + i);
    assert.deepEqual(themeRgbAt(i, -5), first, 'theme ' + i);
    assert.deepEqual(themeRgbAt(i, 99), last, 'theme ' + i);
    assert.deepEqual(themeRgbAt(i, NaN), first, 'theme ' + i);
    assert.deepEqual(themeRgbAt(i, 'x'), first, 'theme ' + i);
    assert.equal(themeHexAt(i, 0), THEMES[i].ramp[0], 'theme ' + i);
    assert.equal(themeHexAt(i, 1), THEMES[i].ramp[4], 'theme ' + i);
  }
  // 中間は補間され、必ず 0..255 の有限値
  for (let t = -0.3; t <= 1.3; t += 0.05) {
    for (let i = 0; i < THEME_COUNT; i++) {
      const c = themeRgbAt(i, t);
      for (const k of ['r', 'g', 'b']) {
        assert.ok(Number.isFinite(c[k]), 'finite ' + k);
        assert.ok(c[k] >= 0 && c[k] <= 255, k + ' in range: ' + c[k]);
      }
      assert.match(themeHexAt(i, t), HEX6, 'theme ' + i + ' t ' + t);
    }
  }
  // 3 テーマ目の ramp[2] は中間色そのもの (t=0.5 は整数添字に一致)
  assert.equal(themeHexAt(0, 0.5), THEMES[0].ramp[2]);
});

test('hexToRgb / rgbToHex は往復し、壊れた入力は null', () => {
  assert.deepEqual(hexToRgb('#00eaff'), { r: 0, g: 234, b: 255 });
  assert.deepEqual(hexToRgb('00EAFF'), { r: 0, g: 234, b: 255 });
  assert.deepEqual(hexToRgb('#ABC'), { r: 170, g: 187, b: 204 });
  for (const bad of ['', '#', '#12', '#12345', '#1234567', 'xyzxyz', null, undefined, 42]) {
    assert.equal(hexToRgb(bad), null, 'bad: ' + String(bad));
  }
  for (const th of THEMES) {
    for (const hex of [th.bg, th.fg, th.dim, th.accent, ...th.ramp]) {
      const c = hexToRgb(hex);
      assert.equal(rgbToHex(c.r, c.g, c.b), hex, hex);
    }
  }
  assert.equal(rgbToHex(-10, 300, 127.6), '#00ff80');
  assert.equal(rgbToHex(255.4, 0, 16), '#ff0010');
  assert.equal(rgbToHex(NaN, 0, 0), null);
  assert.equal(rgbToHex(0, Infinity, 0), null);
});

test('テーマ切替キーは c / C だけを受ける', () => {
  assert.equal(isThemeKey({ key: 'c' }), true);
  assert.equal(isThemeKey({ key: 'C' }), true);
  for (const bad of [{ key: 'x' }, { key: 's' }, null, undefined, {}]) {
    assert.equal(isThemeKey(bad), false, JSON.stringify(bad));
  }
});

// ===== カメラ補間の境界値 ==============================================
test('easeSmoothstep は 0..1 に clamp し、中央で 0.5 になる', () => {
  assert.equal(easeSmoothstep(0), 0);
  assert.equal(easeSmoothstep(1), 1);
  assert.equal(easeSmoothstep(0.5), 0.5);
  assert.equal(easeSmoothstep(-1), 0);
  assert.equal(easeSmoothstep(2), 1);
  assert.equal(easeSmoothstep(NaN), 0);
  assert.equal(easeSmoothstep(Infinity), 0);   // 非有限は 0
  assert.equal(easeSmoothstep('0.5'), 0);      // 文字列は受けない
  // 単調増加
  let prev = -1;
  for (let t = 0; t <= 1; t += 0.02) {
    const v = easeSmoothstep(t);
    assert.ok(v >= prev - 1e-12, 't=' + t);
    prev = v;
  }
});

test('createCamAnim は不正な duration を既定 0.45 に落とす', () => {
  for (const bad of [0, -1, NaN, Infinity, null, undefined, 'x']) {
    const st = createCamAnim(bad);
    assert.equal(st.duration, 0.45, String(bad));
    assert.equal(st.active, false);
    assert.equal(st.t, 0);
    assert.equal(st.fromPos, null);
    assert.equal(st.toPos, null);
  }
  assert.equal(createCamAnim(2).duration, 2);
});

test('startCamAnim は goal を写し取り、開始後の変更に影響されない', () => {
  const st = createCamAnim(1);
  const goal = { pos: [1, 2, 3], target: [4, 5, 6] };
  assert.equal(startCamAnim(st, goal), true);
  assert.equal(st.active, true);
  assert.equal(st.t, 0);
  assert.notEqual(st.toPos, goal.pos);      // 別配列
  goal.pos[0] = 99;
  goal.target[0] = 99;
  assert.deepEqual(st.toPos, [1, 2, 3]);
  assert.deepEqual(st.toTarget, [4, 5, 6]);
  // from を省略したら to を起点にする
  assert.deepEqual(st.fromPos, [1, 2, 3]);

  // 壊れた goal / state は false で状態を変えない
  const st2 = createCamAnim(1);
  for (const bad of [null, undefined, {}, { pos: [1, 2], target: [0, 0, 0] }, { pos: [1, 2, 3] }]) {
    assert.equal(startCamAnim(st2, bad), false, JSON.stringify(bad));
  }
  assert.equal(st2.active, false);
  assert.equal(startCamAnim(null, goal), false);
});

test('startCamAnim は正しい from だけを採用する', () => {
  const st = createCamAnim(1);
  const goal = { pos: [10, 10, 10], target: [0, 0, 0] };
  startCamAnim(st, goal, { pos: [1, 1, 1], target: [2, 2, 2] });
  assert.deepEqual(st.fromPos, [1, 1, 1]);
  assert.deepEqual(st.fromTarget, [2, 2, 2]);
  // 壊れた from は無視して goal を起点にする
  const st2 = createCamAnim(1);
  startCamAnim(st2, goal, { pos: [1, 2], target: [0, 0, 0] });
  assert.deepEqual(st2.fromPos, [10, 10, 10]);
});

test('stepCamAnim は途中を線形補間し、完了時に to を正確に返す', () => {
  const st = createCamAnim(1);
  startCamAnim(st, { pos: [10, 20, 30], target: [0, 0, 0] }, { pos: [0, 0, 0], target: [0, 0, 0] });
  assert.equal(camAnimActive(st), true);

  // dt 不正は 0 扱い (t は進まない)
  const zero = stepCamAnim(st, NaN);
  assert.equal(zero.done, false);
  assert.deepEqual(zero.pos, [0, 0, 0]);
  assert.equal(st.t, 0);

  // 半分 (duration/2) で u=0.5
  const half = stepCamAnim(st, 0.5);
  assert.equal(half.done, false);
  assert.deepEqual(half.pos, [5, 10, 15]);
  assert.equal(camAnimActive(st), true);

  // 完了: 正確に to、active が落ちる
  const done = stepCamAnim(st, 5);
  assert.equal(done.done, true);
  assert.deepEqual(done.pos, [10, 20, 30]);
  assert.deepEqual(done.target, [0, 0, 0]);
  assert.equal(camAnimActive(st), false);
  // 完了後の step は null
  assert.equal(stepCamAnim(st, 1), null);
});

test('stepCamAnim は duration 0 でも即完了し、非オブジェクト state で null', () => {
  const st = createCamAnim(1);
  st.duration = 0;
  startCamAnim(st, { pos: [1, 1, 1], target: [2, 2, 2] });
  const out = stepCamAnim(st, 0);
  assert.equal(out.done, true);
  assert.deepEqual(out.pos, [1, 1, 1]);
  assert.equal(stepCamAnim(null, 1), null);
  assert.equal(stepCamAnim({ active: false }, 1), null);
  assert.equal(camAnimActive(null), false);
  assert.equal(camAnimActive({}), false);
});

// ===== ブックマーク / プリセット ========================================
test('bookmarkForKey / presetForKey は割り当てたキーだけを引き、写しを返す', () => {
  assert.deepEqual(BOOKMARK_KEYS, ['1', '2', '3', '4']);
  assert.deepEqual(PRESET_KEYS, ['1', '2', '3', '4', '5']);
  for (const b of BOOKMARKS) assert.equal(bookmarkForKey(b.key), b);
  for (const p of CAMERA_PRESETS) assert.equal(presetForKey(p.key), p);
  for (const bad of ['0', '5', 'x', '', null, undefined]) {
    assert.equal(bookmarkForKey(bad), null, String(bad));
  }
  for (const bad of ['0', '6', 'x', '', null, undefined]) {
    assert.equal(presetForKey(bad), null, String(bad));
  }
});

test('bookmarkGoal / presetGoal はコピーを返し、壊れた入力は null', () => {
  for (const b of BOOKMARKS) {
    const g = bookmarkGoal(b);
    assert.deepEqual(g.pos, b.pos);
    assert.deepEqual(g.target, b.target);
    assert.notEqual(g.pos, b.pos);        // 別配列
    g.pos[0] = 999;
    assert.notEqual(b.pos[0], 999);
  }
  for (const p of CAMERA_PRESETS) {
    assert.deepEqual(presetGoal(p), bookmarkGoal(p));
  }
  for (const bad of [null, undefined, {}, { pos: [1, 2, 3] }, { pos: [1, 2], target: [0, 0, 0] },
    { pos: [1, 2, NaN], target: [0, 0, 0] }]) {
    assert.equal(bookmarkGoal(bad), null, JSON.stringify(bad));
    assert.equal(presetGoal(bad), null, JSON.stringify(bad));
  }
});

// ===== 層インスペクタの境界値 ==========================================
test('layerKind は 4 層に 1 層を FullAttn、負や非有限を GDN にする', () => {
  assert.equal(ATTENTION_INTERVAL, 4);
  assert.equal(LAYER_COUNT, 48);
  for (let i = 0; i < 12; i++) {
    assert.equal(layerKind(i), (i + 1) % 4 === 0 ? 'FullAttn' : 'GDN', 'layer ' + i);
  }
  assert.equal(layerKind(3), 'FullAttn');
  assert.equal(layerKind(3.9), 'FullAttn');   // trunc してから判定
  for (const bad of [-1, -4, NaN, Infinity, null, 'x']) {
    assert.equal(layerKind(bad), 'GDN', String(bad));
  }
  // interval を変えると周期が変わる / 壊れた interval は既定 4
  assert.equal(layerKind(5, 6), 'FullAttn');
  assert.equal(layerKind(3, 0), 'FullAttn');
  assert.equal(layerKind(3, 'x'), 'FullAttn');
});

test('clampLayer は 0..n-1 に丸め、非有限を 0 にする', () => {
  assert.equal(clampLayer(-5), 0);
  assert.equal(clampLayer(999), LAYER_COUNT - 1);
  assert.equal(clampLayer(NaN), 0);
  assert.equal(clampLayer(3.9), 3);
  assert.equal(clampLayer(0), 0);
  assert.equal(clampLayer(47), 47);
  assert.equal(clampLayer(10, 0), 10);      // n 不正 -> LAYER_COUNT
  assert.equal(clampLayer(999, 12), 11);
});

test('stepLayer は端で反対側へ回り込む', () => {
  assert.equal(stepLayer(0, -1), LAYER_COUNT - 1);
  assert.equal(stepLayer(LAYER_COUNT - 1, 1), 0);
  assert.equal(stepLayer(0, -LAYER_COUNT), 0);
  assert.equal(stepLayer(0, 1), 1);
  assert.equal(stepLayer(5, NaN), 5);       // delta 不正は据え置き
  assert.equal(stepLayer(NaN, 1), 1);       // 現在層不正は 0 から
  assert.equal(stepLayer(0, 2, 3), 2);
  assert.equal(stepLayer(0, 4, 3), 1);
});

test('activationStats は負 / NaN を除外しつつ count を全長で数える', () => {
  assert.deepEqual(activationStats([]), { count: 0, lit: 0, max: 0, mean: 0, rms: 0 });
  assert.deepEqual(activationStats(null), { count: 0, lit: 0, max: 0, mean: 0, rms: 0 });

  const a = activationStats([0.5, 0.5]);
  assert.equal(a.count, 2);
  assert.equal(a.max, 0.5);
  assert.equal(a.mean, 0.5);
  assert.equal(a.rms, 0.5);
  assert.equal(a.lit, 2);

  // 負・NaN・0 は集計から外れるが count には入る
  const b = activationStats([1, -1, NaN, 0]);
  assert.equal(b.count, 4);
  assert.equal(b.lit, 1);
  assert.equal(b.max, 1);
  assert.equal(b.mean, 0.25);              // sum=1 / n=4
  assert.equal(b.rms, 0.5);                // sqrt(1/4)

  // 0.01 未満は lit に数えない
  const c = activationStats([0.005, 0.009, 0.01]);
  assert.equal(c.lit, 1);
  assert.equal(c.max, 0.01);
  // 凍結入力でも落ちない (純粋)
  assert.doesNotThrow(() => activationStats(Object.freeze([0.1, 0.2])));
});

test('formatMagnitude は同じ統計に同じ文字列を返す', () => {
  assert.equal(formatMagnitude(activationStats([0.5, 0.5])), 'max 0.500 · rms 0.500 · lit 2');
  assert.equal(formatMagnitude(null), '--');
  assert.equal(formatMagnitude({ max: NaN }), '--');
  assert.equal(formatMagnitude({ max: 1, rms: 0, lit: 0 }), 'max 1.000 · rms 0.000 · lit 0');
});

test('expertSummary は重み降順・同点は index 昇順で並べ、残り件数を返す', () => {
  const s = expertSummary([3, 7, 1], [0.1, 0.9, 0.5], 3);
  assert.equal(s.count, 3);
  assert.equal(s.more, 0);
  assert.deepEqual(s.top, [
    { index: 7, weight: 0.9 },
    { index: 1, weight: 0.5 },
    { index: 3, weight: 0.1 },
  ]);

  // 同点は index 昇順
  const tie = expertSummary([5, 2], [0.2, 0.2], 2);
  assert.deepEqual(tie.top, [{ index: 2, weight: 0.2 }, { index: 5, weight: 0.2 }]);

  // limit を超えた分は more
  const many = expertSummary([1, 2, 3, 4, 5], [0.5, 0.4, 0.3, 0.2, 0.1], 2);
  assert.deepEqual(many.top, [{ index: 1, weight: 0.5 }, { index: 2, weight: 0.4 }]);
  assert.equal(many.more, 3);
  assert.equal(many.count, 5);

  // 非有限 expert は除外、重み欠落は 0
  const broken = expertSummary([NaN, 4], [0.5, 0.3]);
  assert.equal(broken.count, 1);
  assert.deepEqual(broken.top, [{ index: 4, weight: 0.3 }]);
  const noW = expertSummary([9], undefined);
  assert.deepEqual(noW.top, [{ index: 9, weight: 0 }]);

  // 壊れた limit は既定 3
  assert.equal(expertSummary([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1], NaN).top.length, 3);
  assert.equal(expertSummary([1, 2, 3, 4], [0.4, 0.3, 0.2, 0.1], 0).top.length, 3);
  // 空入力
  assert.deepEqual(expertSummary([], []), { top: [], more: 0, count: 0 });
  assert.deepEqual(expertSummary(null, null).count, 0);
});

test('formatExpertSummary は上位と残り件数を 1 行にする', () => {
  const sum = expertSummary([7, 1, 3], [0.9, 0.5, 0.1], 2);
  assert.equal(formatExpertSummary(sum), 'E7 0.90 · E1 0.50 · +1');
  assert.equal(formatExpertSummary(null), '--');
  assert.equal(formatExpertSummary({ top: [], more: 0 }), '--');
  assert.equal(formatExpertSummary({ top: [{ index: 4, weight: 0.2 }], more: 0 }), 'E4 0.20');
});

test('キー判定: [ / ] は層ステップ、p / P は fps', () => {
  assert.equal(isLayerStepKey({ key: '[' }), -1);
  assert.equal(isLayerStepKey({ key: ']' }), 1);
  for (const bad of [{ key: 'x' }, { key: '{' }, null, undefined]) {
    assert.equal(isLayerStepKey(bad), 0, JSON.stringify(bad));
  }
  assert.equal(isFpsKey({ key: 'p' }), true);
  assert.equal(isFpsKey({ key: 'P' }), true);
  for (const bad of [{ key: 'o' }, null, undefined]) assert.equal(isFpsKey(bad), false);
});

test('overlayFromQuery は 1/true/yes/on だけを真とする', () => {
  for (const t of ['1', 'true', 'TRUE', 'yes', 'on', 'On', 1]) {
    assert.equal(overlayFromQuery(t), true, String(t));
  }
  for (const f of ['0', '', 'false', 'no', 'off', null, undefined]) {
    assert.equal(overlayFromQuery(f), false, String(f));
  }
});

// ===== fps 計測器の境界値 ==============================================
test('createFpsMeter は不正な cap を既定 48 に落とす', () => {
  for (const bad of [0, -5, NaN, Infinity, null, undefined, 'x']) {
    const m = createFpsMeter(bad);
    assert.equal(m.cap, 48, String(bad));
    assert.equal(m.frames, 0);
    assert.equal(m.last, null);
    assert.deepEqual(m.hist, []);
  }
  assert.equal(createFpsMeter(3).cap, 3);
});

test('noteFrame は初回を起点にし、不正時刻と非正の差分を捨てる', () => {
  const m = createFpsMeter(10);
  assert.equal(noteFrame(m, 0).frames, 0);
  assert.equal(m.first, 0);
  assert.equal(m.last, 0);

  // 同じ時刻は無視
  assert.equal(noteFrame(m, 0).frames, 0);

  const s1 = noteFrame(m, 10);
  assert.equal(s1.frames, 1);
  assert.equal(s1.avgMs, 10);
  assert.equal(s1.fps, 100);

  // 巻き戻しは差分を捨てる (frames は増えない)
  assert.equal(noteFrame(m, 5).frames, 1);
  assert.equal(m.frames, 1);
  assert.equal(noteFrame(m, 15).frames, 2);   // 5 -> 15 の 10ms

  assert.equal(noteFrame(m, NaN), null);      // 不正時刻は null を返し状態も据え置き
  assert.equal(m.last, 15);
  assert.equal(noteFrame(null, 1), null);
  // hist を持たない素の object は fpsSummary で落ちる (実装が meter の形を前提にしている)
  assert.throws(() => noteFrame({}, 1), TypeError);
});

test('noteFrame は hist を cap で切り詰め、fpsSummary は hist を写し取る', () => {
  const m = createFpsMeter(3);
  // 差分が 10,20,30,40,50,60 ms になる時刻列
  const times = [0, 10, 30, 60, 100, 150, 210];
  for (const t of times) noteFrame(m, t);
  assert.equal(m.frames, 6);
  assert.equal(m.hist.length, 3);
  assert.deepEqual(m.hist, [40, 50, 60]);
  assert.equal(m.min, 10);
  assert.equal(m.max, 60);

  const sum = fpsSummary(m);
  assert.equal(sum.frames, 6);
  assert.equal(sum.avgMs, 210 / 6);
  assert.ok(Math.abs(sum.fps - 1000 / sum.avgMs) < 1e-9);
  assert.notEqual(sum.hist, m.hist);          // 写し
  sum.hist.push(999);
  assert.equal(m.hist.length, 3);
});

test('fpsSummary / fpsText は空や壊れた入力でも安全', () => {
  const empty = fpsSummary(null);
  assert.deepEqual(empty, { frames: 0, avgMs: 0, fps: 0, emaMs: 0, minMs: 0, maxMs: 0, hist: [] });
  assert.equal(fpsSummary({ frames: 0, sum: 0, ema: 0, min: 0, max: 0, hist: [] }).fps, 0);
  assert.equal(fpsText(empty), '--');
  assert.equal(fpsText(null), '--');
  // 60fps 相当
  assert.equal(fpsText(fpsSummary({ frames: 3, sum: 50.1, ema: 16.7, min: 16.6, max: 16.8, hist: [] })),
    'fps 60 · 16.7 ms');
});

test('fpsBars は 8..40ms を 0..1 に写し、外側を clamp する', () => {
  assert.deepEqual(fpsBars({ hist: [8, 16, 40, 100] }, 24), [0, 0.25, 1, 1]);
  assert.deepEqual(fpsBars({ hist: [-5, 0] }, 24), [0, 0]);
  assert.deepEqual(fpsBars({ hist: [NaN, 24] }, 24), [0, 0.5]);
  // n で末尾だけを切り出す
  assert.deepEqual(fpsBars({ hist: [8, 16, 24, 40] }, 2), [0.5, 1]);
  // 壊れた入力は空配列 / 既定 24 本
  assert.deepEqual(fpsBars(null), []);
  assert.deepEqual(fpsBars({ hist: 'nope' }), []);
  const many = Array.from({ length: 30 }, (_, i) => 8 + i);
  assert.equal(fpsBars({ hist: many }).length, 24);
  assert.equal(fpsBars({ hist: many }, 0).length, 24);   // n 不正 -> 24
});

// ===== 凡例 / ヘルプ ====================================================
test('legendRows は毎回新しい写しを返し、SHORTCUTS を汚さない', () => {
  const a = legendRows();
  const b = legendRows();
  const c = helpRows();
  assert.notEqual(a, b);
  assert.deepEqual(a, b);
  assert.deepEqual(c, a);
  assert.equal(a.length, SHORTCUTS.length);
  assert.notEqual(a[0], SHORTCUTS[0]);        // 行も写し
  a[0].label = '__mutated__';
  a.push({ key: 'x', label: 'x' });
  assert.equal(SHORTCUTS[0].label.includes('__mutated__'), false);
  assert.equal(legendRows().length, SHORTCUTS.length);
});

test('toggleLegend は状態を反転し、壊れた state でも決定的', () => {
  const st = { legend: false };
  assert.equal(toggleLegend(st), true);
  assert.equal(st.legend, true);
  assert.equal(toggleLegend(st), false);
  assert.equal(st.legend, false);
  // 非オブジェクトは無視して true (未定義の反転)
  assert.equal(toggleLegend(null), true);
  assert.equal(toggleLegend(undefined), true);
});

test('凡例キーは ? / Shift+/ を開閉として受ける', () => {
  assert.equal(isLegendKey({ key: '?' }), true);
  assert.equal(isLegendKey({ key: '/', shiftKey: true }), true);
  assert.equal(isLegendKey({ key: '/', shiftKey: false }), false);
  for (const bad of [{ key: 'h' }, { key: 'H' }, null, undefined]) {
    assert.equal(isLegendKey(bad), false, JSON.stringify(bad));
  }
  for (const t of ['1', 'true', 'yes', 'on']) assert.equal(legendFromQuery(t), true, t);
  for (const f of ['0', '', 'off', null, undefined]) assert.equal(legendFromQuery(f), false, String(f));
});

test('help modal の状態遷移 (? / H / Escape) は交互で冪等', () => {
  const st = createHelpState();
  assert.equal(helpOpen(st), false);
  assert.equal(toggleHelp(st), true);
  assert.equal(helpOpen(st), true);
  assert.equal(toggleHelp(st), false);
  assert.equal(closeHelp(st), false);
  for (let i = 0; i < 50; i++) {
    assert.equal(closeHelp(st), false);      // Escape の連打は冪等
    assert.equal(helpOpen(st), false);
  }
  assert.equal(setHelpOpen(st, true), true);
  assert.equal(helpOpen(st), true);
  assert.equal(setHelpOpen(st, false), false);
  // 壊れた state でも落ちない
  assert.equal(toggleHelp(null), true);
  assert.equal(helpOpen(null), false);
  assert.equal(setHelpOpen(null, true), false);
  assert.equal(closeHelp(undefined), false);
});

test('help キー判定: ? / Shift+/ / H は開閉、h はチャット用で受けない', () => {
  for (const e of [
    { key: '?', shiftKey: false }, { key: '?', shiftKey: true }, { key: '/', shiftKey: true },
    { key: 'H', shiftKey: true }, { key: 'h', shiftKey: true },
  ]) assert.equal(isHelpKey(e), true, JSON.stringify(e));
  for (const e of [{ key: '/', shiftKey: false }, { key: 'h' }, { key: 'H', shiftKey: false }]) {
    if (e.key === 'H') assert.equal(isHelpKey(e), true);   // H は shift に依らず開く
    else assert.equal(isHelpKey(e), false, JSON.stringify(e));
  }
  assert.equal(isHelpKey(null), false);
  assert.equal(isHelpCloseKey({ key: 'Escape' }), true);
  assert.equal(isHelpCloseKey({ key: 'Esc' }), false);
  assert.equal(isHelpCloseKey(null), false);
});

test('formatHelpRow は "key — label" を決定的に作る', () => {
  assert.equal(formatHelpRow({ key: 'F', label: 'camera: auto-fit' }), 'F — camera: auto-fit');
  assert.equal(formatHelpRow({ key: 'g' }), 'g');
  assert.equal(formatHelpRow({ label: 'only' }), 'only');
  assert.equal(formatHelpRow({}), '');
  assert.equal(formatHelpRow(null), '');
  const row = { key: 'p', label: 'fps' };
  const once = formatHelpRow(row);
  for (let i = 0; i < 5; i++) assert.equal(formatHelpRow(row), once);
});

// ===== snapshot / reduced motion ========================================
test('snapshotFilename はゼロ埋めした固定形式で、日付不正は -snapshot', () => {
  assert.equal(snapshotFilename(new Date(2026, 0, 2, 3, 4, 5)), 'denno3d-20260102-030405.png');
  assert.equal(snapshotFilename(new Date(2026, 11, 31, 23, 59, 9)), 'denno3d-20261231-235909.png');
  assert.equal(snapshotFilename(new Date(2026, 5, 7, 8, 9, 10), 'shot'), 'shot-20260607-080910.png');
  assert.equal(snapshotFilename(new Date(NaN)), 'denno3d-snapshot.png');
  assert.equal(snapshotFilename('not a date'), 'denno3d-snapshot.png');
  assert.equal(snapshotFilename(new Date(2026, 0, 2, 3, 4, 5), ''), 'denno3d-20260102-030405.png');
  assert.equal(snapshotFilename(new Date(2026, 0, 2, 3, 4, 5), null), 'denno3d-20260102-030405.png');
  assert.equal(isSnapshotKey({ key: 's' }), true);
  assert.equal(isSnapshotKey({ key: 'S' }), true);
  for (const bad of [{ key: 'p' }, null, undefined]) assert.equal(isSnapshotKey(bad), false);
});

test('reduced motion は env と手動フリーズの OR 合成', () => {
  const st = createMotion(false);
  assert.equal(effectiveReduced(st), false);
  assert.equal(orbitEnabled(st), true);

  assert.equal(toggleReducedMotion(st), true);
  assert.equal(st.manualFreeze, true);
  assert.equal(orbitEnabled(st), false);
  assert.equal(toggleReducedMotion(st), false);

  assert.equal(setEnvReduced(st, true), true);
  assert.equal(orbitEnabled(st), false);
  // env が真なら手動を戻しても止まったまま
  assert.equal(toggleReducedMotion(st), true);
  assert.equal(toggleReducedMotion(st), true);
  assert.equal(st.manualFreeze, false);
  assert.equal(setEnvReduced(st, false), false);
  assert.equal(orbitEnabled(st), true);

  // prefers-reduced-motion で初期化した状態
  const pre = createMotion(true);
  assert.equal(effectiveReduced(pre), true);
  assert.equal(orbitEnabled(pre), false);
  assert.equal(toggleReducedMotion(pre), true);
  assert.equal(setEnvReduced(pre, false), true);   // manual が残る
  assert.equal(toggleReducedMotion(pre), false);
  assert.equal(orbitEnabled(pre), true);

  // 壊れた state は落ちない
  assert.equal(effectiveReduced(null), false);
  assert.equal(orbitEnabled(undefined), true);
  assert.equal(toggleReducedMotion(null), false);
  assert.equal(setEnvReduced(null, true), false);

  assert.equal(isMotionKey({ key: 'm' }), true);
  assert.equal(isMotionKey({ key: 'M' }), true);
  for (const bad of [{ key: 'n' }, null, undefined]) assert.equal(isMotionKey(bad), false);
});

// ===== auto-fit の境界値 ================================================
test('sceneBounds は層数に比例した Z 範囲を返し、不正な層数は null', () => {
  const b1 = sceneBounds(1);
  assert.equal(b1.layers, 1);
  assert.deepEqual(b1.min, [-14.8, -7.0, -0.5]);
  assert.deepEqual(b1.max, [7.6, 7.0, 0.5]);
  assert.deepEqual(b1.size, [22.4, 14, 1]);
  assert.ok(Math.abs(b1.center[0] - (-3.6)) < 1e-9);
  assert.equal(b1.center[1], 0);
  assert.equal(b1.center[2], 0);

  const b48 = sceneBounds(48);
  assert.equal(b48.layers, 48);
  assert.ok(Math.abs(b48.min[2] - (-47 * 1.7 - 0.5)) < 1e-9);
  assert.equal(b48.size[2], 0.5 + 47 * 1.7 + 0.5);

  for (const bad of [0, -1, -999, NaN, Infinity, null, undefined, 'x', {}]) {
    assert.equal(sceneBounds(bad), null, String(bad));
  }
  // opts で zStep / halfDepth を上書きできる / 不正は既定
  assert.equal(sceneBounds(2, { zStep: 2 }).min[2], -2.5);
  assert.equal(sceneBounds(2, { zStep: 0 }).min[2], -2.2);    // 既定 1.7
  assert.equal(sceneBounds(2, { halfDepth: -3 }).size[2], 1.7); // 負は 0
});

test('fitCameraGoal は有限で距離範囲内、壊れた層数は null', () => {
  for (const n of [1, 2, 12, 48, 96, 128]) {
    const g = fitCameraGoal(n);
    assert.ok(g, 'N=' + n);
    assert.equal(g.layers, n);
    for (const v of [...g.pos, ...g.target, g.distance, g.radius]) {
      assert.ok(Number.isFinite(v), 'N=' + n + ' non-finite ' + v);
    }
    assert.ok(g.distance >= FIT_BOUNDS.minDistance - 1e-6, 'N=' + n);
    assert.ok(g.distance <= FIT_BOUNDS.maxDistance + 1e-6, 'N=' + n);
  }
  // 大きな N は maxDistance で頭打ち
  assert.equal(fitCameraGoal(128).distance, FIT_BOUNDS.maxDistance);
  // 壊れた入力は null
  for (const bad of [0, -1, NaN, Infinity, null, undefined, 'x', {}]) {
    assert.equal(fitCameraGoal(bad), null, String(bad));
  }
  // auto-fit キーは F (Shift+f) だけ、小文字 f は fly なので受けない
  assert.equal(isAutoFitKey({ key: 'F' }), true);
  assert.equal(isAutoFitKey({ code: 'KeyF', shiftKey: true }), true);
  assert.equal(isAutoFitKey({ key: 'f' }), false);
  assert.equal(isAutoFitKey({ code: 'KeyF', shiftKey: false }), false);
  assert.equal(isAutoFitKey(null), false);
});

// ===== 層ジャンプの境界値 ==============================================
test('digitFromKey は 1 文字の 0..9 だけを数値にする', () => {
  for (let d = 0; d <= 9; d++) assert.equal(digitFromKey({ key: String(d) }), d);
  for (const bad of [{ key: '10' }, { key: 'a' }, { key: '' }, { key: '-' }, { key: ' ' }, null, undefined, {}]) {
    assert.equal(digitFromKey(bad), null, JSON.stringify(bad));
  }
});

test('parseLayerDigits は数字列だけを受け、nLayers-1 に clamp する', () => {
  assert.equal(parseLayerDigits('0', 48), 0);
  assert.equal(parseLayerDigits('47', 48), 47);
  assert.equal(parseLayerDigits('48', 48), 47);
  assert.equal(parseLayerDigits('999', 48), 47);
  assert.equal(parseLayerDigits('047', 48), 47);
  for (const bad of ['', 'a', '1a', '-1', null, undefined, ' ']) {
    assert.equal(parseLayerDigits(bad, 48), null, String(bad));
  }
  for (const bad of [0, -1, NaN, null, undefined, 'x']) {
    assert.equal(parseLayerDigits('99', bad), LAYER_COUNT - 1, String(bad));
  }
});

test('createLayerJump は壊れた層数を LAYER_COUNT に落とす', () => {
  assert.deepEqual(createLayerJump(48), { layers: 48, digits: '', active: false });
  for (const bad of [-1, -48, 0, NaN, null, undefined, 'x']) {
    assert.equal(createLayerJump(bad).layers, LAYER_COUNT, String(bad));
  }
  assert.equal(createLayerJump(1.9).layers, 1);
});

test('pushLayerDigit は入力待ちのときだけ数値/数字列を受け、桁あふれを捨てる', () => {
  const st = createLayerJump(48);
  assert.equal(pushLayerDigit(st, 1), null);       // 未開始
  assert.equal(beginLayerJump(st), true);
  assert.equal(pushLayerDigit(st, 4), 4);
  assert.equal(pushLayerDigit(st, '7'), 47);
  for (const bad of [10, -1, 1.5, NaN, null, true, {}, 'x', '10']) {
    assert.equal(pushLayerDigit(st, bad), null, JSON.stringify(bad));
  }
  // 3 桁を超えたら古い方から捨てる
  const st2 = createLayerJump(48);
  beginLayerJump(st2);
  for (let i = 0; i < 10; i++) pushLayerDigit(st2, i % 10);
  assert.equal(st2.digits.length, LAYER_JUMP_MAX_DIGITS);
  assert.equal(st2.digits, '789');
  assert.equal(jumpToLayer(st2), 47);
  assert.equal(st2.active, false);
  assert.equal(st2.digits, '');
  assert.equal(beginLayerJump(null), false);
});

test('cancelLayerJump / jumpToLayer は入力を捨てて状態を戻す', () => {
  const st = createLayerJump(48);
  beginLayerJump(st);
  pushLayerDigit(st, 1);
  pushLayerDigit(st, 2);
  assert.equal(st.digits, '12');
  assert.equal(cancelLayerJump(st), false);
  assert.equal(st.active, false);
  assert.equal(st.digits, '');

  beginLayerJump(st);
  pushLayerDigit(st, 3);
  assert.equal(jumpToLayer(st), 3);
  assert.equal(jumpToLayer(st), null);            // 空なら null
  assert.equal(jumpToLayer(null), null);

  assert.equal(isLayerJumpKey({ key: 'g' }), true);
  assert.equal(isLayerJumpKey({ key: 'G' }), true);
  for (const bad of [{ key: 'h' }, null, undefined]) assert.equal(isLayerJumpKey(bad), false);
});

// ===== 純粋性: 凍結入力でも落ちない =====================================
test('純粋ヘルパーは凍結入力を書き換えない', () => {
  const route = Object.freeze([3, 7, 1]);
  const weights = Object.freeze([0.1, 0.9, 0.5]);
  assert.doesNotThrow(() => expertSummary(route, weights, 3));
  assert.deepEqual(route, [3, 7, 1]);
  assert.deepEqual(weights, [0.1, 0.9, 0.5]);

  const hist = Object.freeze([8, 16, 40]);
  assert.doesNotThrow(() => fpsBars(Object.freeze({ hist })));
  assert.doesNotThrow(() => fpsSummary(Object.freeze({ frames: 2, sum: 24, ema: 12, min: 8, max: 16, hist })));

  const bm = Object.freeze({ pos: Object.freeze([1, 2, 3]), target: Object.freeze([4, 5, 6]) });
  assert.doesNotThrow(() => bookmarkGoal(bm));
  assert.deepEqual(bm.pos, [1, 2, 3]);
});
