// keys3d.mjs (task 1099) の edge / stress 試験。
// 1. auto-fit camera の極端な層数 / アスペクト比
// 2. 凡例モーダル (? / H / Escape) と層ジャンプ (g + 数字) の高速状態遷移
// 3. 公開定数と整形ヘルパーの不変性・決定論
// node --test crates/denno-view/tests/*.test.mjs で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BOOKMARKS, BOOKMARK_KEYS, CAMERA_PRESETS, PRESET_KEYS, SHORTCUTS, THEMES,
  FIT_BOUNDS, FIT_DIR, LAYER_COUNT, LAYER_JUMP_MAX_DIGITS,
  bookmarkForKey, presetForKey,
  sceneBounds, fitCameraGoal,
  createHelpState, helpOpen, toggleHelp, closeHelp, setHelpOpen,
  isHelpKey, isHelpCloseKey, legendRows, helpRows, formatHelpRow,
  createLayerJump, beginLayerJump, cancelLayerJump, isLayerJumpKey,
  pushLayerDigit, jumpToLayer, parseLayerDigits,
  activationStats, formatMagnitude, formatExpertSummary, fpsSummary, fpsText,
} from '../src/keys3d.mjs';

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
    if (!(d > 0)) return false;
    if (Math.abs(dot(v, right)) > d * tanH + 1e-6) return false;
    if (Math.abs(dot(v, up)) > d * tanV + 1e-6) return false;
  }
  return true;
}

// ===== auto-fit: 極端な層数 ==============================================
const COUNTS = [1, 2, 12, 48, 96, 128];

test('auto-fit: 極端な層数でも goal は有限で OrbitControls の距離範囲に収まる', () => {
  for (const n of COUNTS) {
    const g = fitCameraGoal(n);
    assert.ok(g, 'N=' + n);
    assert.equal(g.layers, n, 'N=' + n);
    for (const v of [...g.pos, ...g.target, g.distance, g.radius]) {
      assert.ok(Number.isFinite(v), 'N=' + n + ' non-finite ' + v);
    }
    assert.ok(g.distance >= MIN_DIST - 1e-6 && g.distance <= MAX_DIST + 1e-6,
      'N=' + n + ' distance ' + g.distance);
  }
});

test('auto-fit: 層数が増えるほど距離は単調に広がる (clamp まで)', () => {
  let prev = 0;
  for (const n of COUNTS) {
    const d = fitCameraGoal(n).distance;
    assert.ok(d >= prev - 1e-6, 'N=' + n + ' 距離が縮んだ prev=' + prev + ' d=' + d);
    prev = d;
  }
});

test('auto-fit: 非常に多い層 (128) は maxDistance で頭打ちになる', () => {
  const g = fitCameraGoal(128);
  assert.equal(g.distance, MAX_DIST);
  assert.equal(g.radius, Math.round(fitCameraGoal(128).radius * 10000) / 10000);
  // 半径は層数とともに増える
  assert.ok(g.radius > fitCameraGoal(96).radius);
  assert.ok(fitCameraGoal(96).radius > fitCameraGoal(48).radius);
});

test('auto-fit: 頭打ちでない層数では全 8 隅が視錐台に収まる', () => {
  for (const n of COUNTS) {
    const g = fitCameraGoal(n);
    // clamp されていない (= 幾何学的に収まる距離が使われた) 場合だけ隅を検査する
    if (g.distance < MAX_DIST - 1e-6) {
      assert.ok(cornersInside(sceneBounds(n), g, FOV, ASPECT), 'N=' + n);
    }
  }
});

// ===== auto-fit: 極端なアスペクト比 ======================================
test('auto-fit: ultrawide (3.5) は既定 (16:9) と同じく全隅が収まる', () => {
  for (const n of COUNTS) {
    const wide = fitCameraGoal(n, { aspect: 3.5 });
    if (wide.distance < MAX_DIST - 1e-6) {
      assert.ok(cornersInside(sceneBounds(n), wide, FOV, 3.5), 'N=' + n);
    }
    // 垂直 fov が律速なので ultrawide は既定と同じ距離になる
    assert.ok(Math.abs(wide.distance - fitCameraGoal(n).distance) < 1e-6, 'N=' + n);
  }
});

test('auto-fit: portrait (0.3) は縦長ぶん遠ざかり、全隅が収まる', () => {
  for (const n of COUNTS) {
    const base = fitCameraGoal(n);
    const tall = fitCameraGoal(n, { aspect: 0.3 });
    assert.ok(tall.distance >= base.distance - 1e-6, 'N=' + n);
    if (tall.distance < MAX_DIST - 1e-6) {
      assert.ok(cornersInside(sceneBounds(n), tall, FOV, 0.3), 'N=' + n);
    }
  }
});

test('auto-fit: 退化した aspect (0 / NaN / 負) は既定 16:9 に戻る', () => {
  for (const n of COUNTS) {
    const base = fitCameraGoal(n);
    for (const bad of [0, -1, NaN, Infinity, null, undefined, 'x']) {
      const g = fitCameraGoal(n, { aspect: bad });
      assert.deepEqual(g, base, 'N=' + n + ' aspect=' + String(bad));
      assert.ok(Number.isFinite(g.distance), 'N=' + n + ' aspect=' + String(bad));
    }
  }
});

test('auto-fit: 層数が壊れていれば aspect に依らず null', () => {
  for (const bad of [0, -1, -999, NaN, Infinity, null, undefined, 'x', {}]) {
    assert.equal(fitCameraGoal(bad), null, String(bad));
    assert.equal(fitCameraGoal(bad, { aspect: 3.5 }), null, String(bad));
  }
});

// ===== 凡例モーダルの高速状態遷移 ========================================
test('help modal: ? / H / Escape を高速に往復しても状態が壊れない', () => {
  const st = createHelpState();
  assert.equal(helpOpen(st), false);

  // ? で開き、H で閉じる
  st.open = toggleHelp(st);
  assert.equal(helpOpen(st), true);
  st.open = toggleHelp(st);
  assert.equal(helpOpen(st), false);

  // 閉じている状態で Escape を押しても閉じたまま (冪等)
  for (let i = 0; i < 100; i++) {
    closeHelp(st);
    assert.equal(helpOpen(st), false);
  }

  // 100 往復しても開閉が交互になる
  let expect = false;
  for (let i = 0; i < 100; i++) {
    expect = toggleHelp(st);
    assert.equal(expect, (i + 1) % 2 === 1, 'i=' + i);
    assert.equal(helpOpen(st), expect, 'i=' + i);
  }
});

test('help modal: isHelpKey と Escape の判定が入力で安定する', () => {
  const openers = [
    { key: '?', shiftKey: false },
    { key: '?', shiftKey: true },
    { key: '/', shiftKey: true },
    { key: 'H', shiftKey: true },
    { key: 'h', shiftKey: true },
  ];
  for (const e of openers) assert.equal(isHelpKey(e), true, JSON.stringify(e));
  for (const e of [{ key: '/', shiftKey: false }, { key: 'h' }, { key: '?' , shiftKey: undefined }]) {
    // '?' は shiftKey に依らず開く
    if (e.key === '?') assert.equal(isHelpKey(e), true);
    else assert.equal(isHelpKey(e), false, JSON.stringify(e));
  }
  assert.equal(isHelpCloseKey({ key: 'Escape' }), true);
  assert.equal(isHelpCloseKey({ key: 'Esc' }), false);

  // ? で開いて Escape で閉じる、を繰り返す
  const st = createHelpState();
  for (let i = 0; i < 50; i++) {
    if (isHelpKey({ key: '?', shiftKey: false })) setHelpOpen(st, true);
    assert.equal(helpOpen(st), true, 'i=' + i);
    if (isHelpCloseKey({ key: 'Escape' })) closeHelp(st);
    assert.equal(helpOpen(st), false, 'i=' + i);
  }
});

test('help modal: H は開閉し、Escape は閉じるだけ (開かない)', () => {
  const st = createHelpState();
  // H で開く
  if (isHelpKey({ key: 'H', shiftKey: true })) st.open = toggleHelp(st);
  assert.equal(helpOpen(st), true);
  // 開いている状態で H → 閉じる
  if (isHelpKey({ key: 'H', shiftKey: true })) st.open = toggleHelp(st);
  assert.equal(helpOpen(st), false);
  // 閉じている状態で Escape → 閉じたまま、開かない
  if (isHelpCloseKey({ key: 'Escape' })) setHelpOpen(st, false);
  assert.equal(helpOpen(st), false);
});

// ===== 層ジャンプの高速状態遷移 ==========================================
test('layer jump: g + 数字の入力途中で cancel すると捨てられる', () => {
  const st = createLayerJump(48);
  for (const n of [48, 12, 96, 128]) {
    st.layers = n;
    beginLayerJump(st);
    pushLayerDigit(st, 1);
    pushLayerDigit(st, 2);
    assert.equal(st.digits, '12');
    assert.equal(cancelLayerJump(st), false);
    assert.equal(st.active, false);
    assert.equal(st.digits, '');
    // cancel 後は数字を受けない
    assert.equal(pushLayerDigit(st, 3), null);
  }
});

test('layer jump: 連続 begin は途中の数字を捨ててやり直す', () => {
  const st = createLayerJump(48);
  beginLayerJump(st);
  pushLayerDigit(st, 4);
  pushLayerDigit(st, 7);
  assert.equal(st.digits, '47');
  beginLayerJump(st);                 // 途中でもう一度 g
  assert.equal(st.digits, '');
  assert.equal(pushLayerDigit(st, 5), 5);
  assert.equal(st.digits, '5');
});

test('layer jump: 範囲外の数字は常に nLayers-1 に clamp される', () => {
  for (const n of [1, 2, 12, 48, 96, 128]) {
    const st = createLayerJump(n);
    beginLayerJump(st);
    const last = pushLayerDigit(st, 9);
    assert.ok(last >= 0 && last <= n - 1, 'N=' + n + ' got ' + last);
    const jumped = jumpToLayer(st);
    assert.ok(jumped >= 0 && jumped <= n - 1, 'N=' + n + ' jump ' + jumped);
    assert.equal(st.active, false);
    assert.equal(st.digits, '');
  }
  // 具体的な丸め: 48 層で "999" -> 47
  const st = createLayerJump(48);
  beginLayerJump(st);
  pushLayerDigit(st, 9);
  pushLayerDigit(st, 9);
  assert.equal(pushLayerDigit(st, 9), 47);
  assert.equal(jumpToLayer(st), 47);
});

test('layer jump: 負 / NaN の nLayers は LAYER_COUNT にフォールバックする', () => {
  for (const bad of [-1, -48, NaN, null, undefined, 0, 'x']) {
    const st = createLayerJump(bad);
    assert.equal(st.layers, LAYER_COUNT, String(bad));
    beginLayerJump(st);
    assert.equal(pushLayerDigit(st, 9), 9, String(bad));
    assert.equal(pushLayerDigit(st, 9), 47, String(bad));   // 99 -> 47
    assert.equal(jumpToLayer(st), 47, String(bad));
  }
  // parseLayerDigits も同じフォールバック
  for (const bad of [-1, NaN, 0, null]) {
    assert.equal(parseLayerDigits('99', bad), 47, String(bad));
  }
});

test('layer jump: キー判定と桁数の上限が安定する', () => {
  assert.equal(isLayerJumpKey({ key: 'g' }), true);
  assert.equal(isLayerJumpKey({ key: 'G' }), true);
  assert.equal(isLayerJumpKey({ key: 'h' }), false);
  assert.equal(isLayerJumpKey(null), false);

  const st = createLayerJump(48);
  beginLayerJump(st);
  for (let i = 0; i < 10; i++) pushLayerDigit(st, i % 10);
  assert.equal(st.digits.length, LAYER_JUMP_MAX_DIGITS);
  assert.equal(st.digits, '789');     // 古い方から捨てられる
  assert.equal(jumpToLayer(st), 47);  // 789 -> clamp 47
});

// ===== 公開定数の不変性 ==================================================
test('定数は再帰的に凍結されていて、書き換えは TypeError になる', () => {
  // 配列そのもの
  for (const arr of [BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES, BOOKMARK_KEYS, PRESET_KEYS]) {
    assert.ok(Object.isFrozen(arr), 'array not frozen');
    assert.throws(() => { arr.push(1); }, TypeError);
  }
  assert.ok(Object.isFrozen(FIT_BOUNDS));
  assert.ok(Object.isFrozen(FIT_DIR));

  // ネストしたオブジェクト / 数値配列
  const deep = [BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES];
  for (const arr of deep) {
    for (const item of arr) {
      assert.ok(Object.isFrozen(item), 'item not frozen: ' + JSON.stringify(item));
      assert.throws(() => { item.key = '__x__'; }, TypeError);
    }
  }
  for (const bm of CAMERA_PRESETS) {
    assert.ok(Object.isFrozen(bm.pos));
    assert.ok(Object.isFrozen(bm.target));
    assert.throws(() => { bm.pos[0] = 999; }, TypeError);
  }
  assert.throws(() => { FIT_BOUNDS.zStep = 999; }, TypeError);
  assert.throws(() => { FIT_DIR[0] = 999; }, TypeError);
});

test('定数の値は読み出しで変化しない (連続参照で同一)', () => {
  const snap = JSON.stringify({ BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES, FIT_BOUNDS, FIT_DIR });
  for (let i = 0; i < 3; i++) {
    assert.equal(JSON.stringify({ BOOKMARKS, CAMERA_PRESETS, SHORTCUTS, THEMES, FIT_BOUNDS, FIT_DIR }), snap);
    bookmarkForKey('2');
    presetForKey('3');
    legendRows();
    helpRows();
  }
});

// ===== 整形ヘルパーの決定論と純粋性 ======================================
test('legendRows / helpRows は呼ぶたびに新しい配列を返し、定数を汚さない', () => {
  const a = legendRows();
  const b = legendRows();
  assert.notEqual(a, b);              // 別インスタンス
  assert.deepEqual(a, b);             // 同じ内容
  assert.equal(a.length, SHORTCUTS.length);
  for (const r of a) {
    assert.notEqual(r, SHORTCUTS.find((s) => s.key === r.key)); // 行も写し
  }
  // 返り値を書き換えても SHORTCUTS は変わらない
  a[0].label = '__mutated__';
  a.push({ key: 'x', label: 'x' });
  assert.equal(SHORTCUTS[0].label.includes('__mutated__'), false);
  assert.equal(legendRows().length, SHORTCUTS.length);
  assert.deepEqual(helpRows(), legendRows());
});

test('formatHelpRow は同じ入力に常に同じ文字列を返す', () => {
  const row = { key: 'F', label: 'camera: auto-fit' };
  const once = formatHelpRow(row);
  for (let i = 0; i < 5; i++) assert.equal(formatHelpRow(row), once);
  assert.equal(once, 'F — camera: auto-fit');
  // 欠けた入力でも決定的に落ちない
  assert.equal(formatHelpRow(null), '');
  assert.equal(formatHelpRow({}), '');
  assert.equal(formatHelpRow({ key: 'g' }), 'g');
});

test('formatMagnitude は同じ統計に同じ文字列を返す (決定論)', () => {
  const values = [0.4, 0.12, 0, -0.2, 0.9, 0.01, NaN, 0.55];
  const st = activationStats(values);
  const text = formatMagnitude(st);
  for (let i = 0; i < 5; i++) assert.equal(formatMagnitude(activationStats(values)), text);
  assert.equal(typeof text, 'string');
  assert.ok(text.startsWith('max '));
  // 壊れた入力は '--'
  assert.equal(formatMagnitude(null), '--');
  assert.equal(formatMagnitude({ max: NaN }), '--');
});

test('formatExpertSummary / fpsText も同じ入力に同じ文字列を返す', () => {
  // formatExpertSummary: 空 summary は '--'
  assert.equal(formatExpertSummary(null), '--');
  assert.equal(formatExpertSummary({ top: [], more: 0 }), '--');
  // fpsSummary -> fpsText は決定的
  const meter = { frames: 3, sum: 50.1, ema: 16.7, min: 16.6, max: 16.8, hist: [16.7, 16.7, 16.7] };
  const sum = fpsSummary(meter);
  const t1 = fpsText(sum);
  for (let i = 0; i < 5; i++) assert.equal(fpsText(fpsSummary(meter)), t1);
  assert.equal(fpsText(null), '--');
});
