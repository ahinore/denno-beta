// keys3d.mjs (task 813) の純ロジック。node --test crates/denno-view/tests/ で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BOOKMARKS, BOOKMARK_KEYS, SHORTCUTS, bookmarkForKey, bookmarkGoal,
  legendRows, toggleLegend, isLegendKey, legendFromQuery,
} from '../src/keys3d.mjs';

test('ブックマークは 1..4 と 4 つの見どころ', () => {
  assert.deepEqual(BOOKMARK_KEYS, ['1', '2', '3', '4']);
  assert.deepEqual(BOOKMARKS.map((b) => b.key), BOOKMARK_KEYS);
  assert.deepEqual(BOOKMARKS.map((b) => b.name), [
    'Overview', 'Router / Experts', 'Prediction Head', 'KV Band',
  ]);
  for (const b of BOOKMARKS) {
    assert.equal(b.pos.length, 3);
    assert.equal(b.target.length, 3);
    for (const v of [...b.pos, ...b.target]) assert.ok(Number.isFinite(v), b.name);
  }
});

test('bookmarkForKey は 1..4 だけを受ける', () => {
  assert.equal(bookmarkForKey('1').name, 'Overview');
  assert.equal(bookmarkForKey('4').name, 'KV Band');
  assert.equal(bookmarkForKey(2).name, 'Router / Experts');   // 数値でも受ける
  assert.equal(bookmarkForKey('5'), null);
  assert.equal(bookmarkForKey(' 1'), null);                   // 空白は正規化しない
  assert.equal(bookmarkForKey(''), null);
  assert.equal(bookmarkForKey(null), null);
});

test('bookmarkGoal はコピーを返し、距離が OrbitControls の範囲に収まる', () => {
  const b = bookmarkForKey('3');
  const g = bookmarkGoal(b);
  assert.deepEqual(g.pos, [...b.pos]);
  assert.deepEqual(g.target, [...b.target]);
  // 元データを壊さない
  g.pos[0] = 999;
  assert.notEqual(b.pos[0], 999);

  const MIN = 8, MAX = 380;   // index3d.html の controls.minDistance/maxDistance
  for (const bm of BOOKMARKS) {
    const dx = bm.pos[0] - bm.target[0];
    const dy = bm.pos[1] - bm.target[1];
    const dz = bm.pos[2] - bm.target[2];
    const d = Math.hypot(dx, dy, dz);
    assert.ok(d >= MIN && d <= MAX, bm.name + ' distance ' + d.toFixed(1));
  }
  assert.equal(bookmarkGoal(null), null);
  assert.equal(bookmarkGoal({ key: '9', pos: [0, 0], target: [0, 0, 0] }), null);
});

test('凡例はブックマーク 4 つと既存のキーを全部載せる', () => {
  const rows = legendRows();
  assert.equal(rows.length, SHORTCUTS.length);
  const keys = rows.map((r) => r.key);
  for (const k of ['1', '2', '3', '4', 'f', 'space', 'c', '?']) {
    assert.ok(keys.includes(k), 'missing ' + k);
  }
  for (const r of rows) {
    assert.equal(typeof r.label, 'string');
    assert.ok(r.label.length > 0);
  }
  // ブックマークの行は名前と一致する
  for (const b of BOOKMARKS) {
    const row = rows.find((r) => r.key === b.key);
    assert.ok(row.label.includes(b.name), row.label + ' vs ' + b.name);
  }
});

test('toggleLegend は開閉を反転させる', () => {
  const st = {};
  assert.equal(toggleLegend(st), true);
  assert.equal(st.legend, true);
  assert.equal(toggleLegend(st), false);
  assert.equal(st.legend, false);
  // state を渡さなくても落ちない
  assert.equal(toggleLegend(null), true);
});

test('isLegendKey は ? と Shift+/ を受ける', () => {
  assert.equal(isLegendKey({ key: '?', shiftKey: false }), true);
  assert.equal(isLegendKey({ key: '/', shiftKey: true }), true);
  assert.equal(isLegendKey({ key: '/', shiftKey: false }), false);
  assert.equal(isLegendKey({ key: '1' }), false);
  assert.equal(isLegendKey(null), false);
});

test('legendFromQuery は真値だけ true', () => {
  for (const v of ['1', 'true', 'yes', 'on', 'TRUE', 'On']) assert.equal(legendFromQuery(v), true, v);
  for (const v of ['0', '', null, undefined, 'no', 'off']) assert.equal(legendFromQuery(v), false, String(v));
});
