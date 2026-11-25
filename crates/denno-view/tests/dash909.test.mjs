// dash.mjs (task 909) の純ロジック: 120 本のリング系列、step レイテンシの
// スパークライン、KV / VRAM の利用量の棒、クイックナビ。
// node --test crates/denno-view/tests/ で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  SERIES_CAP, GAUGE_WIDTH, QUICK_NAV,
  createRing, ringPush, ringLast,
  createDash, noteFrame, step, noteToken, noteTierStats,
  sparkPath, sparkPoints, latencyPath, latencyRow,
  gaugeBar, kvGauge, vramGauge, quickNav,
} from '../src/dash.mjs';

test('SERIES_CAP は 120、GAUGE_WIDTH は 10', () => {
  assert.equal(SERIES_CAP, 120);
  assert.equal(GAUGE_WIDTH, 10);
});

test('リングは cap で古い値から落ちる', () => {
  const ring = createRing(3);
  assert.equal(ring.cap, 3);
  assert.deepEqual(ring.values, []);
  assert.equal(ringPush(ring, 1), true);
  ringPush(ring, 2);
  ringPush(ring, 3);
  assert.deepEqual(ring.values, [1, 2, 3]);
  ringPush(ring, 4);
  assert.deepEqual(ring.values, [2, 3, 4]);
  assert.equal(ringLast(ring), 4);
  assert.equal(ring.values.length, ring.cap);
});

test('リングは NaN / Infinity / null を積まない', () => {
  const ring = createRing(4);
  assert.equal(ringPush(ring, NaN), false);
  assert.equal(ringPush(ring, Infinity), false);
  assert.equal(ringPush(ring, -Infinity), false);
  assert.equal(ringPush(ring, null), false);
  assert.equal(ringPush(ring, undefined), false);
  assert.equal(ringPush(ring, ''), false);
  assert.deepEqual(ring.values, []);
  assert.equal(ringLast(ring), null);
  assert.equal(ringPush(null, 1), false);
  assert.equal(ringLast(null), null);
});

test('createRing の既定は 120 で、cap の下限は 2', () => {
  assert.equal(createRing().cap, SERIES_CAP);
  assert.equal(createRing(1).cap, 2);
  assert.equal(createRing('x').cap, SERIES_CAP);
});

test('decode のリングは 120 本で頭から落ちる', () => {
  const d = createDash({ windowMs: 1 });
  step(d, 0);
  for (let i = 1; i <= SERIES_CAP + 5; i++) {
    noteToken(d, 1, i);
    step(d, i);
  }
  assert.equal(d.decodeRing.values.length, SERIES_CAP);
  // 最後のサンプルは 1 ms 窓に 1 tok = 1000 tok/s
  assert.equal(ringLast(d.decodeRing), 1000);
});

test('sparkPath の境界: 空 / 1 点 / 一定 / NaN 混じり', () => {
  assert.equal(sparkPath([], {}), '');
  assert.equal(sparkPath([null, undefined, NaN, Infinity, -Infinity], {}), '');
  assert.deepEqual(sparkPoints([5], { w: 220, h: 34, pad: 2 }), [{ x: 218, y: 17 }]);
  assert.equal(sparkPath([5], { w: 220, h: 34, pad: 2 }), 'M 218 17');
  // 一定値は真ん中の水平線
  assert.equal(sparkPath([2, 2, 2], { w: 20, h: 10, pad: 0 }), 'M 0 5 L 10 5 L 20 5');
  // NaN / Inf は落として残りだけで描く
  assert.equal(sparkPath([NaN, 0, Infinity, 10], { w: 20, h: 10, pad: 0 }), 'M 0 10 L 20 0');
});

test('latencyPath / latencyRow は空でも壊れない', () => {
  const d = createDash();
  assert.equal(latencyPath(d, {}), '');
  const empty = latencyRow(d, { w: 220, h: 20, pad: 2 });
  assert.equal(empty.key, 'latency');
  assert.equal(empty.text, '-');
  assert.equal(empty.path, '');
  assert.equal(empty.samples, 0);

  // Step フレーム (ns) が ms になって積まれる
  assert.equal(noteFrame(d, { kind: 'Step', width: 1, ns: 12_500_000 }), true);
  assert.equal(ringLast(d.latencyRing), 12.5);
  const row = latencyRow(d, { w: 220, h: 20, pad: 2 });
  assert.equal(row.text, '12.5 ms');
  assert.equal(row.samples, 1);
  assert.ok(row.path.startsWith('M '));
  assert.equal(latencyPath(d, { w: 220, h: 20, pad: 2 }), row.path);
});

test('latencyRow は一定値でも水平線を返す', () => {
  const d = createDash();
  for (let i = 0; i < 4; i++) noteFrame(d, { kind: 'Step', ns: 5_000_000 });
  const row = latencyRow(d, { w: 220, h: 20, pad: 2 });
  assert.equal(row.samples, 4);
  assert.equal(row.text, '5.0 ms');
  // 一定値は y が真ん中で揃う
  const ys = row.path.split(' ').filter((_, i) => i % 3 === 2).map(Number);
  assert.equal(new Set(ys).size, 1);
});

test('gaugeBar は詰まった棒 + 割合を返す', () => {
  const g = gaugeBar(5, 10);
  assert.equal(g.ratio, 0.5);
  assert.equal(g.pct, '50%');
  assert.equal(g.filled, 5);
  assert.equal(g.width, 10);
  assert.equal(g.bar, '█████░░░░░');
  assert.equal(g.text, '█████░░░░░ 50%');
  // 幅は opts.w で変えられる
  assert.equal(gaugeBar(1, 4, { w: 4 }).bar, '█░░░');
  // 文字を差し替えられる
  const alt = gaugeBar(3, 4, { w: 4, full: '#', empty: '.' });
  assert.equal(alt.bar, '###.');
  assert.equal(alt.text, '###. 75%');
});

test('gaugeBar は壊れた値で空の棒と - を返す', () => {
  for (const [u, t] of [[null, 10], [1, null], [1, 0], [1, -5], [-1, 10], [NaN, 10], [1, Infinity]]) {
    const g = gaugeBar(u, t);
    assert.equal(g.ratio, null, `used=${u} total=${t}`);
    assert.equal(g.pct, '-');
    assert.equal(g.filled, 0);
    assert.equal(g.bar, '░'.repeat(GAUGE_WIDTH));
    assert.equal(g.text, '-');
  }
});

test('gaugeBar は 0..1 に丸める (過剰使用でも 100%)', () => {
  const over = gaugeBar(30, 10);
  assert.equal(over.ratio, 1);
  assert.equal(over.pct, '100%');
  assert.equal(over.filled, 10);
  assert.equal(gaugeBar(0, 10).bar, '░'.repeat(GAUGE_WIDTH));
});

test('kvGauge / vramGauge は計器の状態から棒を作る', () => {
  const d = createDash();
  assert.equal(kvGauge(d).text, '-');
  assert.equal(vramGauge(d).text, '-');
  assert.equal(noteFrame(d, { kind: 'KvUsage', busy: [true, false, true, false], t_lens: [1, 0, 2, 0] }), true);
  assert.deepEqual(d.kv, { busy: 2, slots: 4 });
  assert.equal(kvGauge(d, { w: 8 }).text, '████░░░░ 50%');
  // 空の KvUsage は無視
  assert.equal(noteFrame(d, { kind: 'KvUsage', busy: [] }), false);
  assert.deepEqual(d.kv, { busy: 2, slots: 4 });

  noteTierStats(d, { kind: 'TierStats', vramMB: 900, vramTotalMB: 1000, ramExperts: 0, ssdReads: 0 });
  assert.equal(vramGauge(d, { w: 10 }).text, '█████████░ 90%');
});

test('quickNav は 2 / 3 / R を割り当てる', () => {
  assert.equal(QUICK_NAV['2'], '/2d');
  assert.equal(QUICK_NAV['3'], '/3d');
  assert.equal(quickNav('2'), '/2d');
  assert.equal(quickNav('3'), '/3d');
  assert.equal(quickNav('r'), 'refresh');
  assert.equal(quickNav('R'), 'refresh');
  assert.equal(quickNav('x'), null);
  assert.equal(quickNav(null), null);
  assert.equal(quickNav(undefined), null);
});

test('quickNav は入力欄と修飾キー付きを無視する', () => {
  for (const tagName of ['INPUT', 'TEXTAREA', 'SELECT', 'input', 'textarea']) {
    assert.equal(quickNav('2', { tagName }), null, tagName);
  }
  assert.equal(quickNav('3', { contentEditable: true }), null);
  assert.equal(quickNav('3', { isContentEditable: true }), null);
  assert.equal(quickNav('r', { ctrlKey: true }), null);
  assert.equal(quickNav('r', { metaKey: true }), null);
  assert.equal(quickNav('r', { altKey: true }), null);
  // DIV などは通す
  assert.equal(quickNav('2', { tagName: 'DIV' }), '/2d');
  assert.equal(quickNav('2', null), '/2d');
});
