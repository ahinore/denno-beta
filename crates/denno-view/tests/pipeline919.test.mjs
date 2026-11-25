// task 919: 2D テレメトリ / 構成図パイプラインの堅牢性テスト。
//   壊れたフレーム / NaN・Infinity の活性値 / 0 層・1 層の形 /
//   退化した viewport (0x0, 1x10000) / 1000 フレームの連続投入。
// 実行: node --test crates/denno-view/tests/*.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SHAPE_RAW, createMapState } from '../src/view2d.mjs';
import {
  decodeFrame, decodeFrames, concatFrames, KIND, FrameDecodeError, VERSION,
} from '../src/frames.mjs';
import {
  archLayout, archHitTest, archItemsInRect, archExpertRect, archNodeRect,
  streamCircle, fitView, zoomAt, toWorld, drawArch,
} from '../src/arch2d.mjs';
import { buildTree, createEnv, runDsl, visibleRows } from '../src/nodes2d.mjs';

// ---- 独立したフレーム組み立て (frames.test.mjs と同じ流儀) -----------------

const MAGIC = [0x44, 0x4e, 0x54, 0x4c];

function varint(v) {
  const out = [];
  let x = BigInt(v);
  if (x < 0n) throw new Error('varint: negative');
  while (x >= 0x80n) { out.push(Number(x & 0x7fn) | 0x80); x >>= 7n; }
  out.push(Number(x));
  return out;
}

function h16(bits) { return [bits & 0xff, (bits >> 8) & 0xff]; }

function frame(kindId, payload) {
  return Uint8Array.from([...MAGIC, VERSION, kindId, ...payload]);
}

const center = (r) => [r.x + r.w / 2, r.y + r.h / 2];

function fakeCtx() {
  const calls = { fillRect: 0, fillText: 0, drawImage: 0, arc: 0, strokeRect: 0 };
  const noop = () => {};
  return new Proxy({ calls }, {
    get(t, k) {
      if (k in t) return t[k];
      if (k in calls) return () => { calls[k]++; };
      return noop;
    },
    set(t, k, v) { t[k] = v; return true; },
  });
}

function viewData(st) {
  return {
    st, offExp: new Map(), offBlk: new Set(), offStr: new Map(),
    isSel: () => false, dragKeys: null, hover: null, vramOn: false,
    gridImage: () => null, focusHybrid: null, dpr: 1,
  };
}

// ---- フレーム: 壊れた入力 ---------------------------------------------------

test('decodeFrame: 不正な offset は FrameDecodeError (partial=false)', () => {
  const b = frame(KIND.GAP, varint(1));
  for (const bad of [-1, -0.5, NaN, Infinity, -Infinity]) {
    assert.throws(() => decodeFrame(b, bad), (e) => {
      assert.ok(e instanceof FrameDecodeError);
      assert.equal(e.partial, false);
      assert.match(e.message, /offset/);
      return true;
    }, 'offset ' + bad);
  }
  // null / 空は「短すぎる」で partial
  for (const bytes of [null, new Uint8Array(0), []]) {
    assert.throws(() => decodeFrame(bytes), (e) => e instanceof FrameDecodeError && e.partial === true);
  }
});

test('decodeFrame: 過大な count は巨大配列を作らず partial で弾く', () => {
  // 本体が無いのに count だけ 2^32-1 のフレーム (旧実装は new Array(2^32-1) を作った)。
  const cases = [
    frame(KIND.LAYER_ACT, [...varint(0), ...varint(0xffffffff)]),
    frame(KIND.EXPERT_RESIDENCY, [...varint(0), ...varint(0xffffffff)]),
    frame(KIND.EXPERT_SKIP, [...varint(0), ...varint(0xffffffff)]),
    frame(KIND.ROUTER_TOPK, [...varint(0), ...varint(0xffffffff)]),
    frame(KIND.OP_SUMMARY, [...varint(0xffffffff)]),
    frame(KIND.KV_USAGE, [...varint(0xffffffff)]),
    frame(KIND.PREFILL, [...varint(1), ...varint(1), ...varint(0), ...varint(0xffffffff)]),
    frame(KIND.SERVER_INFO, [...varint(0xffffffff)]),
  ];
  for (const b of cases) {
    assert.throws(() => decodeFrame(b), (e) => {
      assert.ok(e instanceof FrameDecodeError);
      assert.equal(e.partial, true, 'kind ' + b[5]);
      return true;
    });
  }
});

test('decodeFrame: 途中で切れた payload は partial=true のまま', () => {
  const good = frame(KIND.LAYER_ACT, [...varint(0), ...varint(2), ...varint(0), ...varint(1), ...h16(0x3c00), ...h16(0x3800)]);
  for (let cut = 6; cut < good.length; cut++) {
    assert.throws(() => decodeFrame(good.subarray(0, cut)), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'cut ' + cut);
      assert.equal(e.partial, true, 'cut ' + cut);
      return true;
    });
  }
  // 元のフレームはそのまま読める
  const f = decodeFrame(good).frame;
  assert.deepEqual(f.idx, [0, 1]);
  assert.deepEqual(f.val, [1, 0.5]);
});

test('decodeFrames: 途中まで読めた分を decoded に載せる (壊れた kind の後)', () => {
  const a = frame(KIND.GAP, varint(3));
  const bad = frame(99, []);
  const buf = Uint8Array.from([...a, ...bad]);
  assert.throws(() => decodeFrames(buf), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.decoded.length, 1);
    assert.deepEqual(e.decoded[0], { kind: 'Gap', lost: 3 });
    assert.equal(e.kindId, 99);
    return true;
  });
});

test('f16: NaN / Infinity は往復しても壊れない', () => {
  // 0x7e00 = NaN, 0x7c00 = +Inf, 0xfc00 = -Inf
  const payload = [...varint(0), ...h16(0x7e00), ...h16(0x7c00), ...h16(0xfc00), ...h16(0x3c00), ...h16(0x0000)];
  const f = decodeFrame(frame(KIND.ROLE_ACT, payload)).frame;
  assert.ok(Number.isNaN(f.roles.attn));
  assert.equal(f.roles.gdn, Infinity);
  assert.equal(f.roles.router, -Infinity);
  assert.equal(f.roles.shex, 1);
  // 表示状態に流しても例外を出さず、状態は有限のまま使える
  const st = createMapState({ layers: 1, n_expert: 4 });
  st.onFrame({ kind: 'Token' });
  st.onFrame(f);
  assert.equal(st.frames, 1);
  assert.equal(st.role[0].attn !== st.role[0].attn, true);   // NaN はそのまま保持
});

// ---- 構成図: 0 層 / 1 層 ---------------------------------------------------

test('archLayout: 0 層は空の配置 (例外なし・当たり判定も安全)', () => {
  const z = archLayout({ layers: 0 });
  assert.equal(z.shape.layers, 0);
  assert.deepEqual(z.panels, []);
  assert.deepEqual(z.blocks, []);
  assert.deepEqual(z.column, []);
  assert.deepEqual(z.groups, []);
  assert.equal(z.step, 0);
  assert.equal(z.width, 0);
  assert.deepEqual(z.bounds, { x: 0, y: 0, w: 0, h: 0 });
  assert.equal(archHitTest(z, 0, 0), null);
  assert.equal(archHitTest(z, NaN, NaN), null);
  assert.deepEqual(archItemsInRect(z, 0, 0, 100, 100), []);
  assert.equal(archNodeRect(z, 'L0'), null);
  assert.deepEqual(archNodeRect(z, 'model'), z.bounds);
  // それでも描画は落ちない (背景だけ)
  const g = fakeCtx();
  drawArch(g, z, Object.assign({ w: 800, h: 600 }, fitView(z.bounds, { w: 800, h: 600 })), viewData(null));
  assert.ok(g.calls.fillRect >= 1);
  assert.equal(g.calls.fillText, 0);
});

test('archLayout: 1 層は 1 パネル / 1 ブロックで、部品と当たり判定が動く', () => {
  const one = archLayout({ layers: 1, n_expert: 8, n_hc: 2, ple_layers: [0] });
  assert.equal(one.panels.length, 1);
  assert.equal(one.blocks.length, 1);
  assert.equal(one.column.length, 1);
  assert.equal(one.groups.length, 1);
  const p = one.panels[0];
  assert.equal(p.layer, 0);
  assert.ok(p.ple, 'PLE は層 0');
  for (const key of ['res0', 'res1', 'res2', 'mixer', 'moe', 'shared']) {
    assert.ok(Number.isFinite(p[key].x) && p[key].w > 0, key);
  }
  const e = archExpertRect(one, 0, 7);
  assert.deepEqual(archHitTest(one, e.x + e.w / 2, e.y + e.h / 2), { type: 'expert', layer: 0, expert: 7 });
  assert.deepEqual(archHitTest(one, ...center(one.column[0])), { type: 'column', layer: 0 });
  const items = archItemsInRect(one, p.x, p.y, p.x + p.w, p.y + p.h);
  assert.equal(items.filter((i) => i.type === 'expert').length, 8);
  assert.equal(items.filter((i) => i.type === 'stream').length, 2);
  // 描画も落ちない
  const st = createMapState({ layers: 1, n_expert: 8, n_hc: 2, ple_layers: [0] });
  const g = fakeCtx();
  drawArch(g, one, Object.assign({ w: 800, h: 600 }, fitView(one.panels[0], { w: 800, h: 600 })), viewData(st));
  assert.equal(g.calls.drawImage, 0);
  assert.ok(g.calls.fillRect > 0);
});

test('archLayout: NaN / Infinity の活性値・分離スコアでも描画が例外を出さない', () => {
  const L = archLayout(DEFAULT_SHAPE_RAW);
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame({ kind: 'Token' });
  st.role[0] = { attn: NaN, gdn: Infinity, router: -Infinity, shex: NaN, ple: Infinity };
  const D = Object.assign(viewData(st), {
    ablit: { scores: [Infinity, NaN, -Infinity, 0.5] },
    hover: { type: 'panel', layer: 0 },
  });
  const pane = { w: 800, h: 600 };
  const g = fakeCtx();
  drawArch(g, L, Object.assign({}, fitView(L.bounds, pane), pane), D);
  assert.ok(g.calls.fillRect > 0);
});

// ---- 退化した viewport -----------------------------------------------------

test('fitView: 0x0 / 1x10000 / 欠損 pane でも有限', () => {
  const L = archLayout(DEFAULT_SHAPE_RAW);
  const panes = [{ w: 0, h: 0 }, { w: 1, h: 10000 }, { w: 10000, h: 1 },
    { w: NaN, h: 600 }, { w: Infinity, h: 600 }, null, undefined];
  for (const pane of panes) {
    const v = fitView(L.bounds, pane);
    assert.ok(Number.isFinite(v.s) && v.s > 0, 's for ' + JSON.stringify(pane));
    assert.ok(Number.isFinite(v.tx) && Number.isFinite(v.ty), 't for ' + JSON.stringify(pane));
  }
  // 0 サイズの矩形でも 0 除算しない
  const v = fitView({ x: 0, y: 0, w: 0, h: 0 }, { w: 0, h: 0 });
  assert.ok(Number.isFinite(v.s) && v.s > 0);
});

test('zoomAt / toWorld: s=0 や NaN でも NaN を広げない', () => {
  const bad = [{ s: 0, tx: 0, ty: 0 }, { s: NaN, tx: 1, ty: 2 }, { s: -1, tx: 0, ty: 0 },
    { s: Infinity, tx: 0, ty: 0 }, null];
  for (const v of bad) {
    const z = zoomAt(v, 2, 100, 50);
    assert.ok(Number.isFinite(z.s) && z.s > 0 && Number.isFinite(z.tx) && Number.isFinite(z.ty),
      'zoom ' + JSON.stringify(v));
    const w = toWorld(v, 100, 50);
    assert.ok(Number.isFinite(w.x) && Number.isFinite(w.y), 'toWorld ' + JSON.stringify(v));
  }
  // 正常系は今までと同じ
  const v0 = { s: 2, tx: 10, ty: 20 };
  const z = zoomAt(v0, 3, 100, 50);
  assert.ok(Math.abs(z.s - 6) < 1e-9);
  assert.deepEqual(toWorld(v0, 14, 26), { x: 2, y: 3 });
  // 画面の点は固定される
  const before = toWorld(v0, 100, 50);
  const after = toWorld(z, 100, 50);
  assert.ok(Math.abs(before.x - after.x) < 1e-9 && Math.abs(before.y - after.y) < 1e-9);
});

test('zoomAt: factor / minS / maxS が壊れていても有限', () => {
  const v = { s: 1, tx: 0, ty: 0 };
  for (const f of [NaN, Infinity, -Infinity]) {
    const z = zoomAt(v, f, 10, 10);
    assert.ok(Number.isFinite(z.s) && z.s > 0);
  }
  const z = zoomAt(v, 1, 10, 10, NaN, NaN);
  assert.ok(Number.isFinite(z.s) && z.s > 0);
});

test('streamCircle: n=0 / 負 / NaN でも有限の円', () => {
  const box = { x: 0, y: 0, w: 300, h: 60 };
  for (const n of [0, -1, NaN, Infinity, 1]) {
    const c = streamCircle(box, 0, n);
    assert.ok(Number.isFinite(c.cx) && Number.isFinite(c.cy) && Number.isFinite(c.r), 'n=' + n);
    assert.ok(c.r >= 0);
  }
});

test('drawArch: 退化 view は描かずに戻る (例外なし)', () => {
  const L = archLayout(DEFAULT_SHAPE_RAW);
  const st = createMapState(DEFAULT_SHAPE_RAW);
  for (const view of [{ s: 0, tx: 0, ty: 0, w: 0, h: 0 }, { s: NaN, tx: 0, ty: 0, w: 800, h: 600 }]) {
    const g = fakeCtx();
    drawArch(g, L, view, viewData(st));
    assert.equal(g.calls.fillRect, 0, 'view ' + JSON.stringify(view));
  }
});

// ---- ノードツリー: 0 層 / 1 層 / NaN --------------------------------------

test('buildTree: 0 層は情報ノードだけの木', () => {
  const t = buildTree({ layers: 0 });
  assert.equal(t.shape.layers, 0);
  assert.equal(t.items.length, 0);
  assert.deepEqual(t.nodes.map((n) => n.id),
    ['model', 'emb', 'vocab', 'ngram', 'out', 'grfinal', 'head', 'mtp']);
  assert.deepEqual(visibleRows(t, null).map((n) => n.id), ['emb', 'out']);
  const env = createEnv(t, { st: null });
  assert.equal(env.count(t.root), 0);
  assert.equal(env.act(t.root), 0);
  assert.deepEqual(runDsl('*:', t, { st: null }).items, []);
});

test('buildTree: 1 層は L0 と 8 expert、DSL も動く', () => {
  const shape = { layers: 1, n_expert: 8, n_hc: 2 };
  const t = buildTree(shape);
  const count = (type) => t.nodes.filter((n) => n.type === type).length;
  assert.equal(count('hybrid'), 1);
  assert.equal(count('layer'), 1);
  assert.equal(count('expert'), 8);
  assert.equal(count('stream'), 2);
  assert.equal(t.items.length, 8 + 2 + 3);
  const st = createMapState(shape);
  st.onFrame({ kind: 'Token' });
  st.onFrame({ kind: 'RouterTopK', layer: 0, experts: [1], weights: [1] });
  assert.deepEqual(runDsl('expert: count > 0', t, { st }).items, [{ type: 'expert', layer: 0, expert: 1 }]);
  assert.deepEqual(runDsl('layer: layer == 0', t, { st }).items.length, 13);
});

test('createEnv: NaN / Infinity の活性値でも count/act/acc/rank は有限', () => {
  const shape = { layers: 2, n_expert: 8, n_hc: 2 };
  const st = createMapState(shape);
  st.onFrame({ kind: 'Token' });
  st.role[0] = { attn: NaN, gdn: Infinity, router: NaN, shex: -Infinity, ple: NaN };
  st.last[0][0] = NaN;
  st.last[0][1] = Infinity;
  st.accum[0][2] = NaN;
  const t = buildTree(shape);
  const env = createEnv(t, { st, off: {}, has: () => false });
  for (const n of t.nodes) {
    for (const fn of ['count', 'act', 'acc', 'rank']) {
      assert.ok(Number.isFinite(env[fn](n)), fn + ' ' + n.id);
    }
    assert.ok(Number.isFinite(env.flag(n, 'topk') + env.flag(n, 'off') + env.flag(n, 'sel')));
  }
  // DSL の比較も NaN に汚染されない
  const D = { st, off: {}, has: () => false };
  const ranked = runDsl('expert: act >= 0 and rank >= 1', t, D).items;
  assert.equal(ranked.length, 2 * 8);
  assert.ok(ranked.every((it) => it.type === 'expert'));
  assert.equal(runDsl('block: act >= 0', t, D).items.length, 2 * 3);
});

// ---- 1000 フレームの連続投入 ----------------------------------------------

test('1000 フレームの連続投入: 決定的で、パイプラインが例外なく回る', () => {
  const shape = DEFAULT_SHAPE_RAW;
  const frames = [];
  for (let k = 0; k < 1000; k++) {
    frames.push({
      kind: 'RouterTopK', layer: k % 48,
      experts: [k % 512, (k * 7 + 1) % 512], weights: [0.6, 0.4],
    });
  }
  const buf = concatFrames(frames);

  const run = () => {
    const decoded = decodeFrames(buf);
    assert.equal(decoded.length, 1000);
    const st = createMapState(shape);
    for (let k = 0; k < 1000; k++) {
      st.onFrame({ kind: 'Token' });
      st.onFrame(decoded[k]);
      st.onFrame({
        kind: 'RoleAct', layer: k % 48,
        roles: { attn: 0.5, gdn: NaN, router: Infinity, shex: 0, ple: 0 },
      });
      if (k % 11 === 0) st.onFrame({ kind: 'ExpertSkip', layer: k % 48, experts: [k % 512] });
      if (k % 13 === 0) st.onFrame({ kind: 'ExpertResidency', layer: k % 48, vram: [k % 8] });
    }
    assert.equal(st.token, 1000);
    const tree = buildTree(shape);
    const res = runDsl('expert: count > 0 and layer < 4 | top 3 by count per layer', tree, { st });
    const L = archLayout(shape);
    const hit = archHitTest(L, ...center(L.panels[0].mixer));
    const mx = L.panels[0].mixer;
    const picked = archItemsInRect(L, mx.x + 1, mx.y + 1, mx.x + mx.w - 1, mx.y + mx.h - 1);
    return {
      counts: [0, 1, 2, 3].map((l) => [...st.count[l].slice(0, 8)]),
      roleAny: [...st.roleCount.slice(0, 4)].map((r) => r.any),
      picked: res.items.length,
      hit,
      inRect: picked.length,
    };
  };

  const t0 = process.hrtime.bigint();
  const a = run();
  const b = run();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  assert.deepEqual(a, b, '同じ入力なら同じ結果');
  assert.deepEqual(a.hit, { type: 'block', layer: 0, block: 'attn' });
  assert.ok(a.picked > 0 && a.picked <= 12);
  assert.ok(a.inRect > 0);
  assert.ok(ms < 5000, '1000 フレーム × 2 回が 5 秒以内 (実測 ' + ms.toFixed(0) + ' ms)');
});

test('1000 枚のフレームを 1 本に連結しても、デコードが速く壊れない', () => {
  const parts = [];
  for (let k = 0; k < 1000; k++) {
    parts.push({ kind: 'RouterTopK', layer: k % 48, experts: [k % 512], weights: [0.5] });
  }
  const buf = concatFrames(parts);
  const t0 = process.hrtime.bigint();
  const out = decodeFrames(buf);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  assert.equal(out.length, 1000);
  assert.equal(out[999].layer, 999 % 48);
  assert.ok(ms < 2000, '実測 ' + ms.toFixed(0) + ' ms');
});
