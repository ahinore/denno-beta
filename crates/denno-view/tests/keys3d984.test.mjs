// keys3d.mjs (task 984) の純ロジック。
// ショートカット凡例モーダル / auto-fit camera / g + 数字の層ジャンプ。
// node --test crates/denno-view/tests/*.test.mjs で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SHORTCUTS, legendRows,
  createHelpState, helpOpen, setHelpOpen, toggleHelp, closeHelp,
  isHelpKey, isHelpCloseKey, helpRows, formatHelpRow,
  FIT_BOUNDS, FIT_DIR, sceneBounds, fitCameraGoal, isAutoFitKey,
  LAYER_JUMP_MAX_DIGITS, createLayerJump, beginLayerJump, cancelLayerJump,
  isLayerJumpKey, digitFromKey, parseLayerDigits, pushLayerDigit, jumpToLayer,
  LAYER_COUNT, clampLayer,
} from '../src/keys3d.mjs';

// index3d.html の PerspectiveCamera / OrbitControls と同じ値
const FOV = FIT_BOUNDS.fov;
const ASPECT = FIT_BOUNDS.aspect;
const MIN_DIST = FIT_BOUNDS.minDistance;
const MAX_DIST = FIT_BOUNDS.maxDistance;

function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function unit(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 0];
}

/// バウンディングボックスの 8 隅が goal の視錐台に入るか (実座標で確認する)。
function cornersInside(b, goal, fov, aspect) {
  const fwd = unit(sub(goal.target, goal.pos));
  const right = unit(cross(fwd, [0, 1, 0]));
  const up = cross(right, fwd);
  const tanV = Math.tan((fov * Math.PI) / 360);
  const tanH = Math.tan(Math.atan(tanV * aspect));
  const xs = [b.min[0], b.max[0]], ys = [b.min[1], b.max[1]], zs = [b.min[2], b.max[2]];
  for (const x of xs) for (const y of ys) for (const z of zs) {
    const v = sub([x, y, z], goal.pos);
    const d = dot(v, fwd);
    if (!(d > 0)) return false;                       // カメラの後ろ
    if (Math.abs(dot(v, right)) > d * tanH + 1e-6) return false;
    if (Math.abs(dot(v, up)) > d * tanV + 1e-6) return false;
  }
  return true;
}

// ===== 凡例モーダル ======================================================
test('凡例モーダル: 既定は閉じていて、開閉と Esc を扱える', () => {
  const st = createHelpState();
  assert.equal(helpOpen(st), false);
  assert.equal(helpOpen(createHelpState(true)), true);
  assert.equal(helpOpen(null), false);
  assert.equal(helpOpen(undefined), false);

  assert.equal(toggleHelp(st), true);
  assert.equal(helpOpen(st), true);
  assert.equal(toggleHelp(st), false);
  assert.equal(helpOpen(st), false);

  assert.equal(setHelpOpen(st, true), true);
  assert.equal(helpOpen(st), true);
  assert.equal(closeHelp(st), false);
  assert.equal(helpOpen(st), false);
  assert.equal(st.open, false);

  // state を渡さなくても落ちない
  assert.equal(toggleHelp(null), true);
  assert.equal(closeHelp(null), false);
});

test('isHelpKey は ? / Shift+/ / H を受け、小文字 h は受けない', () => {
  assert.equal(isHelpKey({ key: '?', shiftKey: false }), true);
  assert.equal(isHelpKey({ key: '?', shiftKey: true }), true);
  assert.equal(isHelpKey({ key: '/', shiftKey: true }), true);
  assert.equal(isHelpKey({ key: '/', shiftKey: false }), false);
  assert.equal(isHelpKey({ key: 'H', shiftKey: true }), true);
  assert.equal(isHelpKey({ key: 'h', shiftKey: true }), true);
  assert.equal(isHelpKey({ key: 'h', shiftKey: false }), false);   // チャット欄の h
  assert.equal(isHelpKey({ key: '1' }), false);
  assert.equal(isHelpKey(null), false);
});

test('isHelpCloseKey は Escape だけ', () => {
  assert.equal(isHelpCloseKey({ key: 'Escape' }), true);
  assert.equal(isHelpCloseKey({ key: 'Esc' }), false);
  assert.equal(isHelpCloseKey({ key: 'h' }), false);
  assert.equal(isHelpCloseKey(null), false);
});

test('helpRows は凡例の全行を写し、F / g + 数字 / ? / H が載る', () => {
  const rows = helpRows();
  assert.deepEqual(rows, legendRows());
  assert.equal(rows.length, SHORTCUTS.length);
  const keys = rows.map((r) => r.key);
  for (const k of ['1', '2', '3', '4', '5', 'F', '[ / ]', 'g + 数字', 'p',
    'c', 's', 'm', 'f', 'space', 'h', '?']) {
    assert.ok(keys.includes(k), 'missing ' + k);
  }
  // H でも開閉できることを凡例の文言で示す
  assert.ok(rows.some((r) => r.key === '?' && r.label.includes('H')));
  // 同じキーの行を二重に載せない
  assert.equal(new Set(keys).size, keys.length);
  for (const r of rows) assert.ok(String(r.label).length > 0, r.key);
});

test('formatHelpRow は "key — label" にし、欠けた入力でも落ちない', () => {
  assert.equal(formatHelpRow({ key: 'F', label: 'auto-fit' }), 'F — auto-fit');
  assert.equal(formatHelpRow({ key: 'F' }), 'F');
  assert.equal(formatHelpRow({ label: 'auto-fit' }), 'auto-fit');
  assert.equal(formatHelpRow({}), '');
  assert.equal(formatHelpRow(null), '');
});

// ===== auto-fit =========================================================
test('sceneBounds: 1 層は 1 層ぶんの厚み、N 層は Z に積み上がる', () => {
  const one = sceneBounds(1);
  assert.ok(one);
  assert.equal(one.layers, 1);
  assert.equal(one.min[2], -FIT_BOUNDS.halfDepth);
  assert.equal(one.max[2], FIT_BOUNDS.halfDepth);
  assert.equal(one.size[2], FIT_BOUNDS.halfDepth * 2);
  // X / Y はレイアウトの固定値
  assert.equal(one.min[0], FIT_BOUNDS.xMin);
  assert.equal(one.max[0], FIT_BOUNDS.xMax);
  assert.equal(one.min[1], FIT_BOUNDS.yMin);
  assert.equal(one.max[1], FIT_BOUNDS.yMax);

  const n = sceneBounds(LAYER_COUNT);
  assert.equal(n.min[2], -(LAYER_COUNT - 1) * FIT_BOUNDS.zStep - FIT_BOUNDS.halfDepth);
  assert.equal(n.max[2], FIT_BOUNDS.halfDepth);
  assert.equal(n.center[2], (n.min[2] + n.max[2]) / 2);
  // 半径は層数とともに広がる
  assert.ok(n.radius > one.radius);
  assert.ok(one.radius > 0);
  // 壊れた N は null
  for (const bad of [0, -3, NaN, Infinity, null, undefined, 'x', {}]) {
    assert.equal(sceneBounds(bad), null, String(bad));
  }
});

test('fitCameraGoal: 同じ N なら常に同じ target/position (決定論的)', () => {
  for (const n of [1, 2, 7, 12, 48, 64]) {
    const a = fitCameraGoal(n);
    const b = fitCameraGoal(n);
    assert.deepEqual(a, b, 'N=' + n);
    assert.equal(a.layers, n);
    // target はバウンディングボックスの中心
    const sb = sceneBounds(n);
    assert.deepEqual(a.target, sb.center.map((v) => Math.round(v * 10000) / 10000));
  }
  for (const bad of [0, -1, NaN, null, undefined, 'x']) {
    assert.equal(fitCameraGoal(bad), null, String(bad));
  }
});

test('fitCameraGoal: 距離は OrbitControls の範囲内で、向きは Overview と同じ', () => {
  for (let n = 1; n <= 64; n++) {
    const g = fitCameraGoal(n);
    const d = Math.hypot(g.pos[0] - g.target[0], g.pos[1] - g.target[1], g.pos[2] - g.target[2]);
    assert.ok(d >= MIN_DIST - 1e-6 && d <= MAX_DIST + 1e-6, 'N=' + n + ' d=' + d);
    assert.ok(Math.abs(d - g.distance) < 1e-3, 'N=' + n);
    const dir = unit(sub(g.pos, g.target));
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(dir[i] - FIT_DIR[i]) < 1e-3, 'N=' + n);
    for (const v of [...g.pos, ...g.target]) assert.ok(Number.isFinite(v), 'N=' + n);
  }
});

test('fitCameraGoal: 全層のバウンディングボックスが視錐台に収まる', () => {
  for (const n of [1, 2, 3, 12, 24, 48, 64]) {
    const g = fitCameraGoal(n);
    assert.ok(cornersInside(sceneBounds(n), g, FOV, ASPECT), 'N=' + n);
    // 縦長 (aspect < 1) の画面でも横方向が欠けない
    const tall = fitCameraGoal(n, { aspect: 0.6 });
    assert.ok(cornersInside(sceneBounds(n), tall, FOV, 0.6), 'N=' + n + ' aspect 0.6');
    assert.ok(tall.distance > g.distance, 'N=' + n + ' 縦長ほど遠い');
  }
});

test('fitCameraGoal: margin と距離 clamp が効く', () => {
  const base = fitCameraGoal(48);
  const wide = fitCameraGoal(48, { margin: 2.0 });
  assert.ok(wide.distance > base.distance);
  const near = fitCameraGoal(48, { maxDistance: 20 });
  assert.equal(near.distance, 20);
  const far = fitCameraGoal(1, { minDistance: 400, maxDistance: 900 });
  assert.equal(far.distance, 400);
  // 壊れたオプションは既定値へ落ちる
  const junk = fitCameraGoal(48, { fov: 'x', aspect: -1, margin: 0, minDistance: NaN });
  assert.ok(junk && junk.distance >= MIN_DIST && junk.distance <= MAX_DIST);
});

test('isAutoFitKey は大文字 F だけで、小文字 f は fly に残す', () => {
  assert.equal(isAutoFitKey({ key: 'F', code: 'KeyF', shiftKey: true }), true);
  assert.equal(isAutoFitKey({ key: 'f', code: 'KeyF', shiftKey: false }), false);
  assert.equal(isAutoFitKey({ key: 'F', code: 'KeyF', shiftKey: false }), true);
  assert.equal(isAutoFitKey({ key: 'a', code: 'KeyA', shiftKey: true }), false);
  assert.equal(isAutoFitKey(null), false);
});

// ===== 層ジャンプ =======================================================
test('createLayerJump は層数を丸め、壊れた値は LAYER_COUNT', () => {
  assert.equal(createLayerJump(48).layers, 48);
  assert.equal(createLayerJump(1).layers, 1);
  assert.equal(createLayerJump(48.9).layers, 48);
  for (const bad of [0, -1, NaN, null, undefined, 'x']) {
    assert.equal(createLayerJump(bad).layers, LAYER_COUNT, String(bad));
  }
  const st = createLayerJump(48);
  assert.equal(st.active, false);
  assert.equal(st.digits, '');
});

test('g + 数字: 1 桁 / 2 桁で層番号になり、範囲外は clamp', () => {
  const st = createLayerJump(48);
  assert.equal(pushLayerDigit(st, 3), null);        // g の前は受けない
  assert.equal(beginLayerJump(st), true);
  assert.equal(st.active, true);
  assert.equal(pushLayerDigit(st, 7), 7);
  assert.equal(pushLayerDigit(st, 9), 47);          // 79 -> 47 に丸める
  assert.equal(st.digits, '79');

  const st2 = createLayerJump(48);
  beginLayerJump(st2);
  assert.equal(pushLayerDigit(st2, 1), 1);
  assert.equal(pushLayerDigit(st2, 2), 12);

  // 47 はそのまま、48 以上は 47
  const st3 = createLayerJump(48);
  beginLayerJump(st3);
  pushLayerDigit(st3, 4);
  assert.equal(pushLayerDigit(st3, 8), 47);
  // 0 は 0
  const st4 = createLayerJump(48);
  beginLayerJump(st4);
  assert.equal(pushLayerDigit(st4, 0), 0);
});

test('pushLayerDigit は不正な入力では状態を変えない', () => {
  const st = createLayerJump(48);
  beginLayerJump(st);
  for (const bad of [-1, 10, 1.5, NaN, 'x', null, undefined, {}]) {
    assert.equal(pushLayerDigit(st, bad), null, String(bad));
  }
  assert.equal(st.digits, '');
  // 数字以外のキーは digitFromKey が null
  assert.equal(digitFromKey({ key: 'a' }), null);
  assert.equal(digitFromKey({ key: '10' }), null);
  assert.equal(digitFromKey({ key: '' }), null);
  assert.equal(digitFromKey(null), null);
  for (let d = 0; d <= 9; d++) assert.equal(digitFromKey({ key: String(d) }), d);
});

test('桁あふれは古い方から捨てる', () => {
  const st = createLayerJump(48);
  beginLayerJump(st);
  pushLayerDigit(st, 1);
  pushLayerDigit(st, 2);
  pushLayerDigit(st, 3);
  assert.equal(st.digits.length, LAYER_JUMP_MAX_DIGITS);
  assert.equal(pushLayerDigit(st, 4), 47);          // 234 -> clamp 47
  assert.equal(st.digits, '234');
});

test('jumpToLayer は確定して状態を戻し、空なら null', () => {
  const st = createLayerJump(48);
  assert.equal(jumpToLayer(st), null);
  beginLayerJump(st);
  assert.equal(jumpToLayer(st), null);              // g だけ押した
  assert.equal(st.active, false);

  beginLayerJump(st);
  pushLayerDigit(st, 2);
  pushLayerDigit(st, 3);
  assert.equal(jumpToLayer(st), 23);
  assert.equal(st.digits, '');
  assert.equal(st.active, false);
  assert.equal(jumpToLayer(null), null);
});

test('cancelLayerJump は入力を捨てる', () => {
  const st = createLayerJump(48);
  beginLayerJump(st);
  pushLayerDigit(st, 4);
  assert.equal(cancelLayerJump(st), false);
  assert.equal(st.active, false);
  assert.equal(st.digits, '');
  assert.equal(pushLayerDigit(st, 4), null);
});

test('parseLayerDigits は [0, n-1] に丸め、空 / 数字以外は null', () => {
  assert.equal(parseLayerDigits('0', 48), 0);
  assert.equal(parseLayerDigits('23', 48), 23);
  assert.equal(parseLayerDigits('47', 48), 47);
  assert.equal(parseLayerDigits('48', 48), 47);
  assert.equal(parseLayerDigits('999', 48), 47);
  assert.equal(parseLayerDigits('7', 1), 0);
  assert.equal(parseLayerDigits('007', 48), 7);
  assert.equal(parseLayerDigits('7', 0), 7);        // 壊れた層数は LAYER_COUNT
  for (const bad of ['', '-1', '1.5', 'x', null, undefined]) {
    assert.equal(parseLayerDigits(bad, 48), null, String(bad));
  }
  // clampLayer と同じ丸めになる
  for (const s of ['0', '5', '47', '48', '99']) {
    assert.equal(parseLayerDigits(s, 48), clampLayer(parseInt(s, 10), 48), s);
  }
});

test('isLayerJumpKey は g / G', () => {
  assert.equal(isLayerJumpKey({ key: 'g' }), true);
  assert.equal(isLayerJumpKey({ key: 'G' }), true);
  assert.equal(isLayerJumpKey({ key: 'j' }), false);
  assert.equal(isLayerJumpKey(null), false);
});
