// denno-view /2d 構成図 (arch2d.mjs) の当たり判定・枠の境界のテスト (task 1189)。
//
// 見る契約:
//   - 矩形は半開区間 [x, x+w) × [y, y+h)。左/上は入り、右/下は入らない。
//   - 負の座標・NaN・退化した view (s=0/NaN/負) でも例外を出さず null / [] を返す。
//   - 極端な拡大率でも画面 → 世界の往復で当たり判定が変わらない。
//   - 層 0 の形は何も置かず、どこを叩いても null / []。
//
// 実行: node --test crates/denno-view/tests/*.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SHAPE_RAW } from '../src/view2d.mjs';
import {
  archLayout, archHitTest, archItemsInRect, archExpertRect, streamCircle,
  fitView, zoomAt, toWorld,
} from '../src/arch2d.mjs';

const L = archLayout(DEFAULT_SHAPE_RAW);
const at = (x, y) => archHitTest(L, x, y);
const center = (r) => [r.x + r.w / 2, r.y + r.h / 2];

test('archHitTest: 箱の枠は左/上を含み、右/下を含まない (半開区間)', () => {
  const p = L.panels[5];
  const m = p.mixer;
  // 左上の角は mixer (block attn)
  assert.deepEqual(at(m.x, m.y), { type: 'block', layer: 5, block: 'attn' });
  // 右上の角: x が右端なので mixer を外れる
  assert.deepEqual(at(m.x + m.w, m.y), { type: 'panel', layer: 5 });
  // 左下の角: y が下端なので mixer を外れる
  assert.deepEqual(at(m.x, m.y + m.h), { type: 'panel', layer: 5 });
  // 右下の角: 両方外れる
  assert.deepEqual(at(m.x + m.w, m.y + m.h), { type: 'panel', layer: 5 });
  // 各辺の中点: 上/左は入り、右/下は外れる
  const [cx, cy] = center(m);
  assert.deepEqual(at(cx, m.y), { type: 'block', layer: 5, block: 'attn' });
  assert.deepEqual(at(m.x, cy), { type: 'block', layer: 5, block: 'attn' });
  assert.deepEqual(at(m.x + m.w, cy), { type: 'panel', layer: 5 });
  assert.deepEqual(at(cx, m.y + m.h), { type: 'panel', layer: 5 });
  // 内側の 1 点は当然 mixer
  assert.deepEqual(at(cx, cy), { type: 'block', layer: 5, block: 'attn' });
});

test('archHitTest: expert 格子のセル境界は次のセルへ送る', () => {
  const p = L.panels[5];
  const g = p.grid;
  // セル 0 の左上はセル 0
  assert.deepEqual(at(g.x, g.y), { type: 'expert', layer: 5, expert: 0 });
  // セル 0 の右端はセル 1 の左端 (同じ x は次へ)
  assert.deepEqual(at(g.x + g.cell, g.y + g.cell / 2), { type: 'expert', layer: 5, expert: 1 });
  // セル 0 の下端は次の行の同じ列
  assert.deepEqual(at(g.x + g.cell / 2, g.y + g.cell), { type: 'expert', layer: 5, expert: g.cols });
  // 格子の右端は格子の外 → MoE の箱 (router)
  assert.deepEqual(at(g.x + g.w, g.y + g.cell / 2), { type: 'block', layer: 5, block: 'moe' });
  // 最後の expert (511) の右下も MoE の箱
  const last = archExpertRect(L, 5, 511);
  assert.deepEqual(at(last.x + last.w, last.y + 1), { type: 'block', layer: 5, block: 'moe' });
  assert.deepEqual(at(last.x + 1, last.y + last.h), { type: 'block', layer: 5, block: 'moe' });
});

test('archHitTest: 左の列と Hybrid block の見出しの境界', () => {
  const c = L.column[9];
  assert.deepEqual(at(c.x, c.y + c.h / 2), { type: 'column', layer: 9 });
  assert.deepEqual(at(c.x + 5, c.y), { type: 'column', layer: 9 });
  // 右上の角は列の外 (パネルはもっと右)
  assert.equal(at(c.x + c.w, c.y + c.h / 2), null);
  assert.equal(at(c.x, c.y + c.h), null);
  // ブロックの見出し帯だけが hybrid。下端は外れる (パネルは padding の内側)
  const B = L.blocks[4];
  assert.deepEqual(at(B.x + 5, B.y), { type: 'hybrid', hybrid: 4 });
  assert.equal(at(B.x + 5, B.y + 80), null);
});

test('archHitTest: 負の座標では何も当たらない', () => {
  assert.equal(at(-1, -1), null);
  assert.equal(at(-1e6, -1e6), null);
  // 図の内側でも一方が負なら当たらない
  assert.equal(at(-10, L.panels[5].mixer.y + 10), null);
  assert.equal(at(L.panels[5].mixer.x + 10, -10), null);
  assert.equal(at(Number.NaN, L.panels[5].mixer.y), null);
  assert.equal(at(L.panels[5].mixer.x, Number.NaN), null);
  assert.equal(at(Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY), null);
});

test('archItemsInRect: 負・NaN の矩形は空、面積 0 の矩形は端に触れない', () => {
  assert.deepEqual(archItemsInRect(L, -100, -100, -1, -1), []);
  assert.deepEqual(archItemsInRect(L, -1e6, -1e6, -1e5, -1e5), []);
  assert.deepEqual(archItemsInRect(L, Number.NaN, Number.NaN, Number.NaN, Number.NaN), []);
  // 幅 0 / 高さ 0 の矩形は、ちょうど枠の上でも当たらない (hitsR は交差が要る)
  const g = L.panels[5].grid;
  assert.deepEqual(archItemsInRect(L, g.x, g.y, g.x, g.y), []);
  // 負の角を渡しても内側だけを拾う (順序は入れ替えても同じ)
  const p = L.panels[3];
  const a = archItemsInRect(L, p.x, p.y, p.x + p.w, p.y + p.h);
  const b = archItemsInRect(L, p.x + p.w, p.y + p.h, p.x, p.y);
  assert.deepEqual(b, a);
  assert.equal(a.length, 512 + 3 + 4);
  // 負の座標から正の座標へ伸びる矩形は、図の内側の項目だけ
  const fromNeg = archItemsInRect(L, -500, -500, p.x + p.w, p.y + p.h);
  assert.ok(fromNeg.every((i) => i.layer >= 0));
});

test('fitView / zoomAt / toWorld: 極端な拡大率でも画面の点が動かない', () => {
  const pane = { w: 1280, h: 800 };
  const v = Object.assign({}, fitView(L.bounds, pane, 16), pane);
  const sx = 640, sy = 400;
  const w0 = toWorld(v, sx, sy);

  // 上限 16 / 下限 0.005 で止まる
  assert.equal(zoomAt(v, 1e6, sx, sy).s, 16);
  assert.equal(zoomAt(v, 1e-9, sx, sy).s, 0.005);

  for (const factor of [1e6, 1e-9, 1e9, 0]) {
    const z = zoomAt(v, factor, sx, sy);
    assert.ok(Number.isFinite(z.s) && z.s >= 0.005 && z.s <= 16, 's=' + z.s);
    const w1 = toWorld(z, sx, sy);
    // 固定点は拡大縮小で動かない (浮動小数の誤差だけ)
    assert.ok(Math.abs(w1.x - w0.x) < 1e-9, 'x ' + factor);
    assert.ok(Math.abs(w1.y - w0.y) < 1e-9, 'y ' + factor);
  }
  // factor が NaN でも現在の view を有限のまま返す
  const zn = zoomAt(v, Number.NaN, sx, sy);
  assert.equal(zn.s, v.s);
  assert.ok(Number.isFinite(zn.tx) && Number.isFinite(zn.ty));
});

test('archHitTest: 画面座標を世界に戻して叩いても拡大率で結果が変わらない', () => {
  const pane = { w: 1280, h: 800 };
  const base = Object.assign({}, fitView(L.bounds, pane, 16), pane);
  const m = L.panels[5].mixer;
  // 枠の上ちょうどは浮動小数の往復で揺れるので、内側の点だけで往復を見る。
  const pts = [
    [m.x + m.w / 2, m.y + m.h / 2, { type: 'block', layer: 5, block: 'attn' }],
    [m.x + 1, m.y + 1, { type: 'block', layer: 5, block: 'attn' }],
  ];
  for (const s of [0.005, 16]) {
    const view = { s, tx: 640 - 9000 * s, ty: 400 - 7600 * s, w: pane.w, h: pane.h };
    assert.equal(view.s, s);
    for (const [wx, wy, want] of pts) {
      // 世界 → 画面 → 世界 (往復)
      const sx = wx * s + view.tx, sy = wy * s + view.ty;
      const back = toWorld(view, sx, sy);
      assert.ok(Math.abs(back.x - wx) < 1e-6 && Math.abs(back.y - wy) < 1e-6, 's=' + s);
      assert.deepEqual(archHitTest(L, back.x, back.y), want, 's=' + s);
    }
  }
  // 退化した view は原点を返すだけ (当たり判定に NaN を広げない)
  for (const bad of [{ s: 0, tx: 0, ty: 0 }, { s: Number.NaN, tx: 0, ty: 0 },
    { s: -4, tx: 10, ty: 10 }]) {
    assert.deepEqual(toWorld(bad, 123, 456), { x: 0, y: 0 });
  }
  // 座標が NaN でも 0 として扱い、NaN を返さない
  const nv = toWorld(base, Number.NaN, 0);
  assert.ok(Number.isFinite(nv.x) && Number.isFinite(nv.y));
});

test('層 0 の形: 何も置かず、どこを叩いても null / []', () => {
  const empty = archLayout({ ...DEFAULT_SHAPE_RAW, layers: 0, layer_kinds: [] });
  assert.equal(empty.width, 0);
  assert.equal(empty.height, 0);
  assert.deepEqual(empty.panels, []);
  assert.deepEqual(empty.column, []);
  assert.deepEqual(empty.blocks, []);
  for (const [x, y] of [[0, 0], [-1, -1], [1, 1], [1e6, 1e6], [Number.NaN, 0]]) {
    assert.equal(archHitTest(empty, x, y), null, x + ',' + y);
  }
  assert.deepEqual(archItemsInRect(empty, 0, 0, 1e6, 1e6), []);
  assert.deepEqual(archItemsInRect(empty, -10, -10, 10, 10), []);
  // 壊れた n_hc / n_expert でも streamCircle は有限の円を返す
  for (const n of [0, -3, Number.NaN, Number.POSITIVE_INFINITY]) {
    const c = streamCircle({ x: 0, y: 0, w: 100, h: 40 }, 0, n);
    assert.ok(Number.isFinite(c.cx) && Number.isFinite(c.cy) && Number.isFinite(c.r), String(n));
    assert.ok(c.r > 0 && c.r < 40);
  }
});
