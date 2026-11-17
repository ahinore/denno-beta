// denno-view /2d のノードツリーと選択式 (DSL) の単体テスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SHAPE_RAW, createMapState, createSelection, addItem, hasItem } from '../src/view2d.mjs';
import {
  buildTree, visibleRows, ancestorIds, nodeItems, nodeCheckState, createEnv,
  tokenize, parseDsl, runDsl, dslSummary, DslError, LEAF_TYPES,
} from '../src/nodes2d.mjs';

const SHAPE = DEFAULT_SHAPE_RAW;   // 48 層 / 512 expert / top-10 / 4 stream / PLE は層 2

/// 決まった活性を流した状態: トークン k ごとに、層 l の expert (l + k) % 4 と 100 + l を選ぶ。
/// RoleAct は gdn / router をいつも、shex は奇数層だけ、ple は層 2 だけ。
function fedState(tokens = 10) {
  const st = createMapState(SHAPE);
  for (let k = 0; k < tokens; k++) {
    st.onFrame({ kind: 'Token' });
    for (let l = 0; l < SHAPE.layers; l++) {
      st.onFrame({ kind: 'RouterTopK', layer: l, experts: [(l + k) % 4, 100 + l], weights: [0.6, 0.4] });
      st.onFrame({
        kind: 'RoleAct', layer: l,
        roles: { gdn: 0.5, attn: 0.5, router: 0.7, shex: l % 2 ? 0.3 : 0, ple: l === 2 ? 0.9 : 0 },
      });
    }
  }
  return st;
}

function data(st, extra = {}) {
  return Object.assign({ st, off: { experts_off: [], blocks_off: [], streams_off: [] }, has: () => false }, extra);
}

// ---- ツリー ---------------------------------------------------------------

test('buildTree: 構成要素がすべてノードになり、item は介入できる葉だけ', () => {
  const t = buildTree(SHAPE);
  const count = (type) => t.nodes.filter((n) => n.type === type).length;
  assert.equal(count('hybrid'), 12);
  assert.equal(count('layer'), 48);
  assert.equal(count('expert'), 48 * 512);
  assert.equal(count('stream'), 48 * 4);
  assert.equal(count('mixer'), 48);
  assert.equal(count('moe'), 48);
  assert.equal(count('shared'), 48);
  assert.equal(count('ple'), 1, 'PLE は ple_layers の層だけ');
  assert.equal(t.byId.get('L2/ple').layer, 2);
  // item の数 = 介入できる葉の総数
  assert.equal(t.items.length, 48 * (512 + 4 + 3) + 1);
  for (const n of t.items) assert.ok(LEAF_TYPES.includes(n.type));
  // 層の種別と表示名
  assert.equal(t.byId.get('L3').kind, 'attn');
  assert.match(t.byId.get('L3').label, /QSA/);
  assert.equal(t.byId.get('L3/mix').label, 'Qwen Sparse Attn');
  assert.equal(t.byId.get('L4/mix').label, 'Gated DeltaNet');
  // 情報だけのノード (embedding / head / MTP) は item を持たない
  for (const id of ['vocab', 'ngram', 'head', 'mtp', 'grfinal']) {
    const n = t.byId.get(id);
    assert.equal(n.hi - n.lo, 0, id);
  }
});

test('buildTree: items[lo..hi) は部分木の item と一致する', () => {
  const t = buildTree(SHAPE);
  const L5 = t.byId.get('L5');
  const items = nodeItems(t, L5);
  assert.equal(items.length, 512 + 4 + 3);
  assert.ok(items.every((it) => it.layer === 5));
  assert.equal(nodeItems(t, t.byId.get('H1')).length, 4 * (512 + 4 + 3));
  assert.deepEqual(nodeItems(t, t.byId.get('L7/e9')), [{ type: 'expert', layer: 7, expert: 9 }]);
  assert.deepEqual(nodeItems(t, t.byId.get('L7/mix')), [{ type: 'block', layer: 7, block: 'attn' }]);
  assert.equal(nodeItems(t, t.root).length, t.items.length);
});

test('visibleRows / ancestorIds: 開いたノードの子だけが並ぶ', () => {
  const t = buildTree(SHAPE);
  const closed = visibleRows(t, new Set());
  assert.deepEqual(closed.map((n) => n.id), ['emb', ...Array.from({ length: 12 }, (_, h) => 'H' + h), 'out']);
  const open = visibleRows(t, new Set(['H0', 'L1']));
  const ids = open.map((n) => n.id);
  assert.ok(ids.includes('L1/res') && ids.includes('L1/ffn') && ids.includes('L0'));
  assert.ok(!ids.includes('L0/res'), 'L0 は閉じている');
  assert.deepEqual(ancestorIds(t.byId.get('L6/e3')), ['model', 'H1', 'L6', 'L6/ffn', 'L6/ex']);
});

test('nodeCheckState: none / some / all / na', () => {
  const t = buildTree(SHAPE);
  const sel = createSelection();
  const has = (it) => hasItem(sel, it);
  const res = t.byId.get('L0/res');
  assert.equal(nodeCheckState(t, res, has), 'none');
  addItem(sel, { type: 'stream', layer: 0, stream: 1 });
  assert.equal(nodeCheckState(t, res, has), 'some');
  for (let i = 0; i < 4; i++) addItem(sel, { type: 'stream', layer: 0, stream: i });
  assert.equal(nodeCheckState(t, res, has), 'all');
  assert.equal(nodeCheckState(t, t.byId.get('vocab'), has), 'na');
});

// ---- 値 -------------------------------------------------------------------

test('createMapState: count / roleCount は decay しない活性化回数', () => {
  const st = fedState(10);
  assert.equal(st.token, 10);
  // 層 0: expert 0..3 は 10 トークン中それぞれ 2〜3 回、expert 100 は毎回
  assert.equal(st.count[0][100], 10);
  assert.equal(st.count[0][0] + st.count[0][1] + st.count[0][2] + st.count[0][3], 10);
  assert.equal(st.count[0][4], 0);
  assert.equal(st.roleCount[1].shex, 10);
  assert.equal(st.roleCount[0].shex, 0);
  assert.equal(st.roleCount[2].ple, 10);
  assert.equal(st.roleCount[5].any, 10);
  st.reset();
  assert.equal(st.count[0][100], 0);
  assert.equal(st.roleCount[1].shex, 0);
});

test('createEnv: count / act / rank / off / sel', () => {
  const t = buildTree(SHAPE);
  const st = fedState(10);
  const sel = createSelection();
  addItem(sel, { type: 'expert', layer: 4, expert: 100 });
  const env = createEnv(t, data(st, {
    off: { experts_off: [[3, 7]], blocks_off: [[5, 'moe']], streams_off: [[6, 2]] },
    has: (it) => hasItem(sel, it),
  }));
  const N = (id) => t.byId.get(id);
  assert.equal(env.count(N('L0/e100')), 10);
  assert.equal(env.count(N('L1/sh')), 10);
  assert.equal(env.count(N('L0/sh')), 0);
  assert.equal(env.count(N('L3/mix')), 10, 'QSA の mixer は attn の回数');
  assert.equal(env.count(N('L0/ex')), 20, 'routed experts は配下の和');
  assert.equal(env.rank(N('L0/e100')), 1);
  assert.equal(env.rank(N('L0/e500')), 6, '0 回の expert は 5 個の後ろで同順位');
  assert.equal(env.flag(N('L3/e7'), 'off'), 1);
  assert.equal(env.flag(N('L5/moe'), 'off'), 1);
  assert.equal(env.flag(N('L6/s2'), 'off'), 1);
  assert.equal(env.flag(N('L6/s1'), 'off'), 0);
  assert.equal(env.flag(N('L4/e100'), 'sel'), 1);
  assert.ok(Math.abs(env.act(N('L0/sh'))) < 1e-9);
  assert.ok(Math.abs(env.act(N('L1/sh')) - 0.3) < 1e-6);
});

// ---- 字句・構文 -----------------------------------------------------------

test('tokenize: 1..3 の範囲と小数・指数・文字列', () => {
  const t = tokenize('layer in 1..3 and w > .5e-1 or kind == "qsa"');
  const v = t.map((x) => x.v);
  assert.deepEqual(v.slice(0, 5), ['layer', 'in', 1, '..', 3]);
  assert.equal(v[8], 0.05);
  assert.equal(t.find((x) => x.t === 'str').v, 'qsa');
});

test('parseDsl: 種類の並び・段・複数クエリ', () => {
  const a = parseDsl('expert, mixer: count > 1 | top 3 by acc per layer | limit 9; stream:');
  assert.equal(a.queries.length, 2);
  assert.deepEqual(a.queries[0].types, ['expert', 'mixer']);
  assert.equal(a.queries[0].stages[0].op, 'top');
  assert.equal(a.queries[0].stages[0].n, 3);
  assert.equal(a.queries[0].stages[0].per, 'layer');
  assert.equal(a.queries[0].stages[1].op, 'limit');
  assert.equal(a.queries[1].expr, null);
  // 種類を省いた式は介入できる葉の全種類
  assert.deepEqual(parseDsl('count == 0').queries[0].types, LEAF_TYPES);
  assert.deepEqual(parseDsl('block: off').queries[0].types.sort(), ['mixer', 'moe', 'ple', 'shared']);
});

test('parseDsl: 誤りは DslError で位置つき', () => {
  const cases = [
    ['expert: cnt > 1', 8, /知らない名前/],
    ['foo: count', 0, /知らない種類/],
    ['expert: count >', 15, /途中で終わ/],
    ['expert: count > 1 2', 18, /書けない/],
    ['expert: sqrt(1, 2)', 8, /引数 1/],
    ['expert: lmax()', 8, /引数 1/],
    ['expert: x', 8, /知らない名前/],
    ['expert: "abc', 8, /閉じていない/],
    ['expert: | top x', 14, /整数/],
    ['', 0, /空/],
  ];
  for (const [src, pos, re] of cases) {
    assert.throws(() => parseDsl(src), (e) => e instanceof DslError && e.pos === pos && re.test(e.message), src);
  }
});

// ---- 評価 -----------------------------------------------------------------

test('runDsl: 活性化回数と層番号で絞る', () => {
  const t = buildTree(SHAPE);
  const st = fedState(10);
  const r = runDsl('expert: count == 0 and layer >= 40', t, data(st));
  // 層 40..47 の 512 個のうち、選ばれた 5 個 (0..3 と 100+l) 以外
  assert.equal(r.nodes.length, 8 * (512 - 5));
  assert.equal(r.items.length, r.nodes.length);
  assert.ok(r.items.every((it) => it.type === 'expert' && it.layer >= 40));
  assert.match(dslSummary(r), /^4056 nodes \(expert 4056\) → 4056 items$/);
});

test('runDsl: 算術・集約・連鎖比較・in', () => {
  const t = buildTree(SHAPE);
  const st = fedState(10);
  const D = data(st);
  const ids = (src) => runDsl(src, t, D).nodes.map((n) => n.id);
  // 層内最多 (= 10) の半分以上: expert 100+l と、2〜3 回の 0..3 は 5 未満なので入らない
  assert.deepEqual(ids('expert: count >= 0.5 * lmax(count) and layer < 2'), ['L0/e100', 'L1/e101']);
  assert.deepEqual(ids('expert: layer == 0 and count * 2 + 1 > 20'), ['L0/e100']);
  assert.deepEqual(ids('expert: 0 <= index < 3 and layer == 1'), ['L1/e0', 'L1/e1', 'L1/e2']);
  assert.deepEqual(ids('mixer: kind == "qsa" and layer % 8 == 3'),
    ['L3/mix', 'L11/mix', 'L19/mix', 'L27/mix', 'L35/mix', 'L43/mix']);
  assert.deepEqual(ids('moe: layer in [5, 9, 17]'), ['L5/moe', 'L9/moe', 'L17/moe']);
  assert.deepEqual(ids('stream: index == 0 and layer in 0..2'), ['L0/s0', 'L1/s0', 'L2/s0']);
  assert.deepEqual(ids('shared: act > 2 * gmean(act) and layer < 6'), ['L1/sh', 'L3/sh', 'L5/sh']);
  assert.deepEqual(ids('expert: rate == 1 and hybrid == 0'), ['L0/e100', 'L1/e101', 'L2/e102', 'L3/e103']);
  assert.deepEqual(ids('ple:'), ['L2/ple']);
  assert.deepEqual(ids('layer: not (layer < 46)'), ['L46', 'L47']);
  assert.deepEqual(ids('expert: layer == 0 and (index == 3 or index == 4) and -count < 0'), ['L0/e3']);
  // 0 割りは 0
  assert.deepEqual(ids('shared: count / 0 == 0 and layer == 0'), ['L0/sh']);
});

test('runDsl: top / bottom / per / limit と和集合', () => {
  const t = buildTree(SHAPE);
  const st = fedState(10);
  const D = data(st);
  const ids = (src) => runDsl(src, t, D).nodes.map((n) => n.id);
  assert.deepEqual(ids('expert: layer < 3 | top 1 by count per layer'), ['L0/e100', 'L1/e101', 'L2/e102']);
  // 既定の by は count。同点 (e0 と e1 が 3 回) はツリー順で前のもの
  assert.deepEqual(ids('expert: layer == 0 | top 2'), ['L0/e0', 'L0/e100']);
  assert.equal(ids('expert: | top 2 per hybrid').length, 24);
  assert.deepEqual(ids('expert: layer == 0 | bottom 2 by index'), ['L0/e0', 'L0/e1']);
  assert.deepEqual(ids('shared: | limit 3'), ['L0/sh', 'L1/sh', 'L2/sh']);
  // ; は和集合 (重複は 1 つ)
  assert.deepEqual(ids('moe: layer == 1; moe: layer <= 1; mixer: layer == 0'), ['L0/mix', 'L0/moe', 'L1/moe']);
});

test('runDsl: グループを選ぶと下の item を全部返す / off と sel で選べる', () => {
  const t = buildTree(SHAPE);
  const st = fedState(4);
  const sel = createSelection();
  addItem(sel, { type: 'stream', layer: 9, stream: 3 });
  const D = data(st, {
    off: { experts_off: [[3, 7], [3, 8]], blocks_off: [[5, 'moe']], streams_off: [] },
    has: (it) => hasItem(sel, it),
  });
  const r = runDsl('layer: layer == 47', t, D);
  assert.equal(r.nodes.length, 1);
  assert.equal(r.items.length, 512 + 4 + 3);
  assert.equal(runDsl('residual: layer < 2', t, D).items.length, 8);
  assert.deepEqual(runDsl('expert: off', t, D).items, [
    { type: 'expert', layer: 3, expert: 7 }, { type: 'expert', layer: 3, expert: 8 },
  ]);
  assert.deepEqual(runDsl('block: off', t, D).items, [{ type: 'block', layer: 5, block: 'moe' }]);
  assert.deepEqual(runDsl('sel', t, D).items, [{ type: 'stream', layer: 9, stream: 3 }]);
});

test('runDsl: 文字列の計算は位置つきの誤り', () => {
  const t = buildTree(SHAPE);
  assert.throws(() => runDsl('expert: kind + 1 > 0', t, data(fedState(1))),
    (e) => e instanceof DslError && /文字列/.test(e.message) && e.pos === 13);
});
