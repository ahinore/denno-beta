// keys3d.mjs (task 939) の純ロジック。
// カラーテーマ 3 種 / hex<->RGB の範囲 / snapshot のファイル名 / reduced motion。
// node --test crates/denno-view/tests/*.test.mjs で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  THEMES, THEME_COUNT, clampThemeIndex, themeAt, themeIndex, cycleTheme, themeName,
  isThemeKey, hexToRgb, rgbToHex, themeRgbAt, themeHexAt, themeCssVars,
  snapshotFilename, isSnapshotKey,
  createMotion, effectiveReduced, setEnvReduced, toggleReducedMotion, orbitEnabled, isMotionKey,
} from '../src/keys3d.mjs';

const HEX6 = /^#[0-9a-f]{6}$/;

test('3 つの curated テーマが揃っている', () => {
  assert.equal(THEME_COUNT, 3);
  assert.deepEqual(THEMES.map((t) => t.name), ['cyber', 'thermal', 'mono-high-contrast']);
  for (const th of THEMES) {
    for (const k of ['bg', 'fg', 'dim', 'accent']) {
      assert.match(th[k], HEX6, th.name + ' ' + k);
    }
    assert.equal(th.ramp.length, 5, th.name + ' ramp');
    for (const c of th.ramp) assert.match(c, HEX6, th.name + ' ramp color');
    // 両端は必ず固定 (範囲外の lookup が端に張り付く根拠)
    assert.equal(th.ramp[0], th.ramp[0].toLowerCase());
    assert.equal(th.ramp[4], th.ramp[4].toLowerCase());
  }
});

test('themeAt / clampThemeIndex は範囲を回り込む', () => {
  for (let i = 0; i < THEME_COUNT; i++) {
    assert.equal(clampThemeIndex(i), i);
    assert.equal(themeAt(i).name, THEMES[i].name);
  }
  assert.equal(clampThemeIndex(THEME_COUNT), 0);
  assert.equal(clampThemeIndex(THEME_COUNT + 1), 1);
  assert.equal(clampThemeIndex(-1), THEME_COUNT - 1);
  assert.equal(clampThemeIndex(-THEME_COUNT - 2), THEME_COUNT - 2);
  assert.equal(clampThemeIndex(NaN), 0);
  assert.equal(clampThemeIndex(1.9), 1);
  assert.equal(themeAt(-1).name, 'mono-high-contrast');
  assert.equal(themeAt(7).name, 'thermal');
});

test('themeIndex は名前を引き、未知は -1', () => {
  assert.equal(themeIndex('cyber'), 0);
  assert.equal(themeIndex('Thermal'), 1);
  assert.equal(themeIndex(' MONO-HIGH-CONTRAST '), 2);
  assert.equal(themeIndex('nope'), -1);
  assert.equal(themeIndex(null), -1);
  assert.equal(themeIndex(undefined), -1);
});

test('cycleTheme は cyber -> thermal -> mono -> cyber', () => {
  assert.equal(cycleTheme(0), 1);
  assert.equal(cycleTheme(1), 2);
  assert.equal(cycleTheme(2), 0);
  const seen = [];
  let i = 0;
  for (let n = 0; n < THEME_COUNT + 1; n++) { seen.push(themeName(i)); i = cycleTheme(i); }
  assert.deepEqual(seen, ['cyber', 'thermal', 'mono-high-contrast', 'cyber']);
});

test('hexToRgb は #rgb / #rrggbb を受け、壊れた値は null', () => {
  assert.deepEqual(hexToRgb('#00eaff'), { r: 0, g: 234, b: 255 });
  assert.deepEqual(hexToRgb('00EAFF'), { r: 0, g: 234, b: 255 });
  assert.deepEqual(hexToRgb('#fff'), { r: 255, g: 255, b: 255 });
  assert.deepEqual(hexToRgb('#000'), { r: 0, g: 0, b: 0 });
  for (const bad of ['', '#', '#12', '#12345', '#1234567', 'xyzxyz', null, undefined, 42]) {
    assert.equal(hexToRgb(bad), null, 'bad: ' + String(bad));
  }
});

test('rgbToHex は 0..255 に clamp して丸める', () => {
  assert.equal(rgbToHex(0, 234, 255), '#00eaff');
  assert.equal(rgbToHex(-10, 300, 127.6), '#00ff80');
  assert.equal(rgbToHex(255.4, 0, 16), '#ff0010');
  assert.equal(rgbToHex(NaN, 0, 0), null);
  assert.equal(rgbToHex(0, Infinity, 0), null);
});

test('themeRgbAt は 0..1 の外側を端に clamp する', () => {
  const first = hexToRgb(THEMES[0].ramp[0]);
  const last = hexToRgb(THEMES[0].ramp[4]);
  assert.deepEqual(themeRgbAt(0, -5), first);
  assert.deepEqual(themeRgbAt(0, 99), last);
  assert.deepEqual(themeRgbAt(0, NaN), first);
  assert.deepEqual(themeRgbAt(0, 'x'), first);
  for (let i = -0.25; i <= 1.25; i += 0.05) {
    const c = themeRgbAt(1, i);
    for (const k of ['r', 'g', 'b']) {
      assert.ok(Number.isFinite(c[k]), 'finite ' + k);
      assert.ok(c[k] >= 0 && c[k] <= 255, k + ' in range: ' + c[k]);
    }
  }
});

test('themeHexAt は常に #rrggbb を返す', () => {
  for (let t = -0.2; t <= 1.2; t += 0.1) {
    for (let i = 0; i < THEME_COUNT; i++) {
      const hex = themeHexAt(i, t);
      assert.match(hex, HEX6, 'theme ' + i + ' t ' + t);
    }
  }
  assert.equal(themeHexAt(0, 0), THEMES[0].ramp[0]);
  assert.equal(themeHexAt(0, 1), THEMES[0].ramp[4]);
});

test('themeCssVars は背景 / 文字 / 補助色をそのまま出す', () => {
  for (let i = 0; i < THEME_COUNT; i++) {
    const v = themeCssVars(i);
    assert.deepEqual(Object.keys(v).sort(), ['--bg', '--dim', '--fg', '--vram']);
    assert.equal(v['--bg'], THEMES[i].bg);
    assert.equal(v['--fg'], THEMES[i].fg);
    assert.equal(v['--vram'], THEMES[i].accent);
  }
  assert.deepEqual(themeCssVars(0), themeCssVars(THEME_COUNT));
});

test('snapshotFilename は決まった形式でゼロ埋めする', () => {
  assert.equal(snapshotFilename(new Date(2026, 0, 2, 3, 4, 5)), 'denno3d-20260102-030405.png');
  assert.equal(snapshotFilename(new Date(2026, 11, 31, 23, 59, 9)), 'denno3d-20261231-235909.png');
  assert.equal(snapshotFilename(new Date(2026, 5, 7, 8, 9, 10), 'shot'),
    'shot-20260607-080910.png');
  assert.equal(snapshotFilename(new Date(NaN)), 'denno3d-snapshot.png');
  assert.equal(snapshotFilename('not a date'), 'denno3d-snapshot.png');
  assert.equal(snapshotFilename(new Date(2026, 0, 2, 3, 4, 5), ''), 'denno3d-20260102-030405.png');
});

test('キー判定は c / s / m だけを受ける', () => {
  assert.equal(isThemeKey({ key: 'c' }), true);
  assert.equal(isThemeKey({ key: 'C' }), true);
  assert.equal(isThemeKey({ key: 'x' }), false);
  assert.equal(isThemeKey(null), false);
  assert.equal(isSnapshotKey({ key: 's' }), true);
  assert.equal(isSnapshotKey({ key: 'S' }), true);
  assert.equal(isSnapshotKey({ key: 'p' }), false);
  assert.equal(isMotionKey({ key: 'm' }), true);
  assert.equal(isMotionKey({ key: 'M' }), true);
  assert.equal(isMotionKey({ key: 'n' }), false);
  assert.equal(isMotionKey(undefined), false);
});

test('reduced motion の状態遷移 (env と手動の OR)', () => {
  const st = createMotion(false);
  assert.equal(effectiveReduced(st), false);
  assert.equal(orbitEnabled(st), true);

  // M の手動トグルだけで止まる / 戻る
  assert.equal(toggleReducedMotion(st), true);
  assert.equal(st.manualFreeze, true);
  assert.equal(orbitEnabled(st), false);
  assert.equal(toggleReducedMotion(st), false);
  assert.equal(orbitEnabled(st), true);

  // OS 設定 (env) が真なら手動を戻しても止まったまま
  assert.equal(setEnvReduced(st, true), true);
  assert.equal(orbitEnabled(st), false);
  assert.equal(toggleReducedMotion(st), true);   // manual=true, env=true
  assert.equal(toggleReducedMotion(st), true);   // manual=false でも env が残る
  assert.equal(st.manualFreeze, false);
  assert.equal(setEnvReduced(st, false), false);
  assert.equal(orbitEnabled(st), true);

  // prefers-reduced-motion で初期化した状態は最初から止まっている
  const pre = createMotion(true);
  assert.equal(effectiveReduced(pre), true);
  assert.equal(orbitEnabled(pre), false);
  assert.equal(toggleReducedMotion(pre), true);   // manual を足しても真のまま
  assert.equal(setEnvReduced(pre, false), true);  // env=false でも manual が残る
  assert.equal(toggleReducedMotion(pre), false);  // manual を戻せば解除
  assert.equal(orbitEnabled(pre), true);

  // 壊れた state でも落ちない
  assert.equal(effectiveReduced(null), false);
  assert.equal(orbitEnabled(undefined), true);
  assert.equal(toggleReducedMotion(null), false);
  assert.equal(setEnvReduced(null, true), false);
});
