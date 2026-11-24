// denno-view /2d の構成図 (arch2d.mjs) の単体テスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SHAPE_RAW, createMapState } from '../src/view2d.mjs';
import {
  ARCH, PANEL, archLayout, archHitTest, archItemsInRect, archExpertRect, archNodeRect,
  streamCircle, fitView, zoomAt, toWorld, drawArch, gridPixels,
} from '../src/arch2d.mjs';
import { buildTree } from '../src/nodes2d.mjs';

const L = archLayout(DEFAULT_SHAPE_RAW);
const center = (r) => [r.x + r.w / 2, r.y + r.h / 2];
const inside = (a, b) => a.x >= b.x && a.y >= b.y && a.x + a.w <= b.x + b.w && a.y + a.h <= b.y + b.h;
const overlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

test('archLayout: 公式図の構成要素がすべて置かれる', () => {
  assert.equal(L.blocks.length, 12, 'Hybrid block × L/4');
  assert.equal(L.panels.length, 48);
  assert.equal(L.column.length, 48);
  assert.equal(L.groups.length, 12);
  for (const k of ['mtp', 'head', 'grFinal', 'hidFinal', 'vocab', 'ngram']) assert.ok(L[k] && L[k].w > 0, k);
  // 層パネルはブロックの中、部品はパネルの中で重ならない
  for (const p of L.panels) {
    assert.ok(inside(p, L.blocks[Math.floor(p.layer / 4)]), 'panel ' + p.layer);
    const parts = ['res0', 'res1', 'res2', 'readA', 'writeA', 'readB', 'writeB',
      'hidRA', 'hidWA', 'hidRB', 'hidWB', 'mixer', 'moe'].map((k) => [k, p[k]]);
    if (p.ple) parts.push(['ple', p.ple]);
    for (const [k, r] of parts) assert.ok(inside(r, p), k + ' in panel ' + p.layer);
    for (let i = 0; i < parts.length; i++) {
      for (let j = i + 1; j < parts.length; j++) {
        assert.ok(!overlap(parts[i][1], parts[j][1]), parts[i][0] + ' / ' + parts[j][0]);
      }
    }
    assert.ok(inside(p.grid, p.moe) && inside(p.shared, p.moe));
    assert.ok(!overlap(p.grid, p.shared));
  }
  assert.ok(L.panels[2].ple && !L.panels[3].ple, 'PLE は層 2 だけ');
  // パネル同士・ブロック同士は重ならない
  for (let i = 0; i < 48; i++) for (let j = i + 1; j < 48; j++) assert.ok(!overlap(L.panels[i], L.panels[j]));
  for (let i = 0; i < 12; i++) for (let j = i + 1; j < 12; j++) assert.ok(!overlap(L.blocks[i], L.blocks[j]));
  // 全部が bounds の中
  for (const r of [...L.panels, ...L.column, L.mtp, L.head, L.vocab, L.ngram]) assert.ok(inside(r, L.bounds));
});

test('archLayout: 流れは下 → 上 (層 0 が下、ブロック内も下から)', () => {
  for (let l = 1; l < 48; l++) assert.ok(L.column[l].y < L.column[l - 1].y, 'column ' + l);
  for (let h = 0; h < 12; h++) {
    const ps = L.panels.slice(h * 4, h * 4 + 4);
    for (let j = 1; j < 4; j++) assert.ok(ps[j].y < ps[j - 1].y);
    assert.equal(ps[3].kind, 'attn', 'ブロックの最上段が QSA');
    assert.ok(ps.slice(0, 3).every((p) => p.kind === 'gdn'));
  }
  // ブロックの 2 行目 (6..11) は 1 行目 (0..5) の上
  assert.ok(L.blocks[6].y < L.blocks[0].y);
  assert.ok(L.blocks[1].x > L.blocks[0].x);
  // 部品の縦の並び: 入力 residual → mixer → 中 residual → MoE → 出力 residual
  const p = L.panels[0];
  assert.ok(p.res0.y > p.mixer.y && p.mixer.y > p.res1.y && p.res1.y > p.moe.y && p.moe.y > p.res2.y);
  assert.ok(p.readA.y === p.res0.y && p.writeA.y === p.res1.y && p.writeB.y === p.res2.y);
  assert.ok(L.vocab.y > L.column[0].y && L.mtp.y < L.column[47].y && L.head.y < L.grFinal.y);
});

test('archHitTest: 部品 → 選択項目 / 情報', () => {
  const p = L.panels[5];
  const at = (r) => archHitTest(L, ...center(r));
  const e = archExpertRect(L, 5, 77);
  assert.deepEqual(at(e), { type: 'expert', layer: 5, expert: 77 });
  assert.deepEqual(at(p.mixer), { type: 'block', layer: 5, block: 'attn' });
  assert.deepEqual(at(p.shared), { type: 'block', layer: 5, block: 'shared' });
  assert.deepEqual(archHitTest(L, p.moe.x + 5, p.moe.y + 30), { type: 'block', layer: 5, block: 'moe' });
  assert.deepEqual(at(L.panels[2].ple), { type: 'block', layer: 2, block: 'ple' });
  for (const key of ['res0', 'res1', 'res2']) {
    const c = streamCircle(p[key], 2, 4);
    assert.deepEqual(archHitTest(L, c.cx, c.cy), { type: 'stream', layer: 5, stream: 2 }, key);
  }
  assert.deepEqual(at(p.readB), { type: 'info', layer: 5, part: 'readB', label: 'GR Read' });
  assert.deepEqual(at(p.writeA), { type: 'info', layer: 5, part: 'writeA', label: 'GR Write' });
  assert.deepEqual(archHitTest(L, p.x + 5, p.y + p.h - 5), { type: 'panel', layer: 5 });
  assert.deepEqual(at(L.column[9]), { type: 'column', layer: 9 });
  assert.deepEqual(archHitTest(L, L.blocks[4].x + 300, L.blocks[4].y + 20), { type: 'hybrid', hybrid: 4 });
  assert.deepEqual(at(L.head), { type: 'top', part: 'head' });
  assert.equal(archHitTest(L, -10, -10), null);
});

test('archItemsInRect: パネル全体 = その層の介入項目すべて', () => {
  const p = L.panels[3];
  const items = archItemsInRect(L, p.x, p.y, p.x + p.w, p.y + p.h);
  assert.equal(items.length, 512 + 3 + 4);
  assert.equal(items.filter((i) => i.type === 'expert').length, 512);
  assert.equal(items.filter((i) => i.type === 'stream').length, 4);
  // 格子の一部だけ: 2x3 のセル (router の moe は入らない)
  const a = archExpertRect(L, 3, 33), b = archExpertRect(L, 3, 67);
  const part = archItemsInRect(L, a.x + 1, a.y + 1, b.x + b.w - 1, b.y + b.h - 1);
  assert.deepEqual(part.map((i) => i.expert), [33, 34, 35, 65, 66, 67]);
  // 2 層にまたがる矩形 (residual の列) は両方の stream
  const two = archItemsInRect(L, L.panels[1].x, L.panels[1].y, L.panels[0].x + 400, L.panels[0].y + L.panels[0].h);
  assert.ok(two.every((i) => i.type === 'stream'));
  assert.deepEqual([...new Set(two.map((i) => i.layer))].sort(), [0, 1]);
});

test('archNodeRect: ツリーの全ノードに飛び先がある', () => {
  const t = buildTree(DEFAULT_SHAPE_RAW);
  for (const n of t.nodes) {
    if (n.type === 'expert' && n.index > 3) continue;   // 数が多いので一部
    const r = archNodeRect(L, n.id);
    assert.ok(r && r.w > 0 && r.h > 0, n.id);
  }
  assert.ok(overlap(archNodeRect(L, 'L9/e200'), archExpertRect(L, 9, 200)));
  assert.equal(archNodeRect(L, 'H3'), L.blocks[3]);
  assert.equal(archNodeRect(L, 'L4/mix'), L.panels[4].mixer);
});

test('fitView / zoomAt / toWorld: 画面の点を固定して拡大', () => {
  const pane = { w: 800, h: 600 };
  const v = fitView(L.bounds, pane, 10);
  // 全体が収まる
  assert.ok(L.bounds.w * v.s <= 780 + 1e-6 && L.bounds.h * v.s <= 580 + 1e-6);
  const w0 = toWorld(v, 300, 200);
  const z = zoomAt(v, 3, 300, 200);
  const w1 = toWorld(z, 300, 200);
  assert.ok(Math.abs(w0.x - w1.x) < 1e-6 && Math.abs(w0.y - w1.y) < 1e-6);
  assert.ok(Math.abs(z.s - v.s * 3) < 1e-9);
  assert.equal(zoomAt(v, 1e9, 0, 0, 0.01, 4).s, 4, '上限で止まる');
});

/// 描画は呼んだ命令を数えるだけの偽 context で、例外が出ないことと LOD を見る。
function fakeCtx() {
  const calls = { fillRect: 0, fillText: 0, drawImage: 0, arc: 0 };
  const noop = () => {};
  const g = new Proxy({}, {
    get(t, k) {
      if (k === 'calls') return calls;
      if (k in t) return t[k];
      if (k in calls) return () => { calls[k]++; };
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  return g;
}

test('drawArch: 全体表示は縮小画像、拡大すると expert を 1 つずつ・文字も描く', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame({ kind: 'Token' });
  st.onFrame({ kind: 'RouterTopK', layer: 0, experts: [1, 2], weights: [0.5, 0.5] });
  const D = {
    st, offExp: new Map([[0, new Set([5])]]), offBlk: new Set(['1:moe']), offStr: new Map(),
    isSel: () => false, dragKeys: null, hover: null, vramOn: true,
    gridImage: () => ({}), focusHybrid: 0, dpr: 1,
  };
  const pane = { w: 800, h: 600 };
  const whole = fitView(L.bounds, pane);
  const g1 = fakeCtx();
  drawArch(g1, L, Object.assign({}, whole, pane), D);
  assert.equal(g1.calls.drawImage, 48, '全体: 層ごとの縮小画像');
  const g2 = fakeCtx();
  drawArch(g2, L, Object.assign({}, fitView(L.panels[0].moe, pane), pane), D);
  assert.equal(g2.calls.drawImage, 0);
  assert.ok(g2.calls.fillRect >= 512, '拡大: セルを 1 つずつ');
  assert.ok(g2.calls.fillText >= 512, 'セルが大きいと番号も描く');
});

test('gridPixels: off は赤、skip は琥珀、moe off は暗い赤', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame({ kind: 'RouterTopK', layer: 0, experts: [1], weights: [1] });
  st.skip[0][2] = 1;
  const px = gridPixels(L, st, 0, new Uint8ClampedArray(512 * 4), new Set([3]), false);
  assert.deepEqual([...px.slice(3 * 4, 3 * 4 + 3)], [255, 59, 48]);
  assert.deepEqual([...px.slice(2 * 4, 2 * 4 + 3)], [255, 196, 77]);
  assert.ok(px[1 * 4 + 1] > 200, 'top-1 は明るい');
  const off = gridPixels(L, st, 0, new Uint8ClampedArray(512 * 4), null, true);
  assert.deepEqual([...off.slice(9 * 4, 9 * 4 + 3)], [58, 20, 24]);
});

test('PANEL / ARCH は固定値 (描画とテストで同じ値を使う)', () => {
  assert.ok(Object.isFrozen(PANEL) && Object.isFrozen(ARCH));
  assert.equal(PANEL.h, L.panels[0].h);
});

/// 描いた文字を全部覚える偽 context (task 809 の MTP 強調の確認用)。
function textCtx() {
  const texts = [];
  const noop = () => {};
  const g = new Proxy({ texts }, {
    get(t, k) {
      if (k in t) return t[k];
      if (k === 'fillText') return (s) => { texts.push(String(s)); };
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
  return g;
}

test('drawArch: 投機テレメトリが来ると MTP ブロックを強調し採用率を出す', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  const pane = { w: 800, h: 600 };
  const view = Object.assign({}, fitView(L.mtp, pane, 10), pane);
  const base = {
    st, offExp: new Map(), offBlk: new Set(), offStr: new Map(),
    isSel: () => false, dragKeys: null, hover: null, vramOn: false,
    gridImage: () => null, focusHybrid: null, dpr: 1,
  };
  const g0 = textCtx();
  drawArch(g0, L, view, Object.assign({}, base, { mtp: null }));
  assert.ok(g0.texts.includes('MTP Modules'), 'MTP の箱は常に描く');
  assert.ok(!g0.texts.some((t) => t.startsWith('accept') || t === 'spec telemetry'),
    'テレメトリが無いときは強調しない');

  const g1 = textCtx();
  drawArch(g1, L, view, Object.assign({}, base, { mtp: { seen: true, rate: 0.87 } }));
  assert.ok(g1.texts.includes('MTP Modules'));
  assert.ok(g1.texts.includes('accept 87%'), '採用率を MTP の下に出す');

  const g2 = textCtx();
  drawArch(g2, L, view, Object.assign({}, base, { mtp: { seen: true, rate: null } }));
  assert.ok(g2.texts.includes('spec telemetry'), '数字が無いときは数字を作らない');
  assert.ok(!g2.texts.some((t) => t.startsWith('accept')));
});
