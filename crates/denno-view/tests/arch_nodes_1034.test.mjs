// denno-view /2d のノードツリー (nodes2d.mjs) と構成図 (arch2d.mjs) の
// トポロジー不変条件・ストレステスト (task 1034)。
//
// 小さな 2 層 dense / 48 層 hybrid GDN+QSA MoE / 96 層の大きなモデルで、
//   - 世界座標がすべて有限 (NaN / Infinity が 1 つも無い)
//   - 箱は正の大きさで、同じ階層の箱は重ならない
//   - 同じ形からは毎回同じ配置になる (決定的)
//   - ツリーの ID・親子・items[lo..hi) が参照整合
//   - ツリーの item と図の選択項目が層ごとに一致する
//   - 空 (layers=0) / 最小の形を壊さずに扱う
// を検査する。実行: node --test crates/denno-view/tests/*.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SHAPE_RAW } from '../src/view2d.mjs';
import { buildTree, visibleRows, nodeItems, LEAF_TYPES } from '../src/nodes2d.mjs';
import {
  PANEL, archLayout, archHitTest, archItemsInRect, archExpertRect, archNodeRect,
  streamCircle, fitView, zoomAt, toWorld, drawArch,
} from '../src/arch2d.mjs';

/// 多様なトポロジー。48 層は実モデル相当 (DEFAULT_SHAPE_RAW)、96 層は
/// expert 1024 / stream 8 の大きなモデル。
const SHAPES = Object.freeze([
  ['2-layer dense', {
    layers: 2, layer_kinds: ['gdn', 'attn'],
    n_expert: 1, n_expert_used: 1, n_hc: 1, n_embd: 64, ple_layers: [],
  }],
  ['48-layer hybrid MoE', DEFAULT_SHAPE_RAW],
  ['96-layer large', {
    layers: 96, n_expert: 1024, n_expert_used: 16,
    n_hc: 8, n_embd: 4096, ple_layers: [2, 50],
  }],
]);

const center = (r) => [r.x + r.w / 2, r.y + r.h / 2];
const inside = (a, b) => a.x >= b.x && a.y >= b.y && a.x + a.w <= b.x + b.w && a.y + a.h <= b.y + b.h;
const overlap = (a, b) => a.x < b.x + b.w && a.x + a.w > b.x && a.y < b.y + b.h && a.y + a.h > b.y;

/// 入れ子のどこかにある数値を全部集める ({path: number})。
function collectNumbers(v, out = [], path = '$') {
  if (typeof v === 'number') { out.push([path, v]); return out; }
  if (Array.isArray(v)) { v.forEach((x, i) => collectNumbers(x, out, path + '[' + i + ']')); return out; }
  if (v && typeof v === 'object') {
    for (const k of Object.keys(v)) collectNumbers(v[k], out, path + '.' + k);
  }
  return out;
}

/// レイアウトの全部の箱 (bounds とトップの箱、パネルと部品)。
function allRects(L) {
  const out = [
    ['bounds', L.bounds], ['grid', L.grid], ['mtp', L.mtp], ['grFinal', L.grFinal],
    ['hidFinal', L.hidFinal], ['head', L.head], ['vocab', L.vocab], ['ngram', L.ngram],
  ];
  L.blocks.forEach((r, i) => out.push(['block' + i, r]));
  L.panels.forEach((p, i) => out.push(['panel' + i, p]));
  L.column.forEach((r, i) => out.push(['column' + i, r]));
  L.groups.forEach((r, i) => out.push(['group' + i, r]));
  return out;
}

/// 層パネルの中の部品 (パネル自身は含めない)。
function panelParts(p) {
  const out = ['res0', 'res1', 'res2', 'readA', 'writeA', 'readB', 'writeB',
    'hidRA', 'hidWA', 'hidRB', 'hidWB', 'mixer', 'moe'].map((k) => [k, p[k]]);
  if (p.ple) out.push(['ple', p.ple]);
  return out;
}

/// 選択項目 → 比較用の文字列 (ツリーと図で同じ表現)。
function key(it) {
  if (it.type === 'expert') return 'expert:' + it.layer + ':' + it.expert;
  if (it.type === 'stream') return 'stream:' + it.layer + ':' + it.stream;
  return 'block:' + it.layer + ':' + it.block;
}

function fakeCtx() {
  const calls = { fillRect: 0, fillText: 0, drawImage: 0 };
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

// ---- 座標と箱 ---------------------------------------------------------------

test('多様なトポロジー: 座標はすべて有限・箱は正・同じ階層の箱は重ならない', () => {
  for (const [name, raw] of SHAPES) {
    const L = archLayout(raw);
    const s = L.shape;
    // 形は正規化されている (layers が反映される)
    assert.equal(s.layers, raw.layers, name);

    // NaN / Infinity が 1 つも無い
    for (const [path, n] of collectNumbers(L)) {
      assert.ok(Number.isFinite(n), name + ': 非有限の数 ' + path + ' = ' + n);
    }

    // 全部の箱が正の大きさ
    for (const [path, r] of allRects(L)) {
      assert.ok(r.w > 0 && r.h > 0, name + ': ' + path + ' が空の箱');
    }
    for (const p of L.panels) {
      for (const [k, r] of panelParts(p)) assert.ok(r.w > 0 && r.h > 0, name + ': panel ' + p.layer + '/' + k);
    }

    // 全体の bounds に全部が入る
    for (const [path, r] of allRects(L)) assert.ok(inside(r, L.bounds), name + ': ' + path + ' が bounds の外');

    // パネル同士・ブロック同士・列の箱同士は重ならない
    const noOverlap = (list, label) => {
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
          assert.ok(!overlap(list[i], list[j]), name + ': ' + label + ' ' + i + ' と ' + j + ' が重なる');
        }
      }
    };
    noOverlap(L.panels, 'panel');
    noOverlap(L.blocks, 'block');
    noOverlap(L.column, 'column');

    // パネルはブロックの中、部品はパネルの中で互いに重ならない
    for (const p of L.panels) {
      assert.ok(inside(p, L.blocks[Math.floor(p.layer / 4)]), name + ': panel ' + p.layer + ' がブロックの外');
      const parts = panelParts(p);
      for (const [k, r] of parts) assert.ok(inside(r, p), name + ': ' + k + ' が panel ' + p.layer + ' の外');
      for (let i = 0; i < parts.length; i++) {
        for (let j = i + 1; j < parts.length; j++) {
          assert.ok(!overlap(parts[i][1], parts[j][1]),
            name + ': panel ' + p.layer + ' の ' + parts[i][0] + ' / ' + parts[j][0] + ' が重なる');
        }
      }
      // MoE の格子と shared expert は MoE の中、互いに重ならない
      assert.ok(inside(p.grid, p.moe) && inside(p.shared, p.moe), name + ': panel ' + p.layer + ' の MoE の部品');
      assert.ok(!overlap(p.grid, p.shared), name + ': panel ' + p.layer + ' の格子と shared');
      assert.equal(p.grid.n, s.n_expert);
      assert.ok(p.grid.cols >= 1 && p.grid.rows >= 1 && p.grid.cell > 0, name + ': 格子の寸法');
      assert.equal(p.grid.cols * p.grid.rows >= s.n_expert, true, name + ': 格子の升目が expert を覆う');
    }

    // 下 → 上 の流れ: 入力 (embedding) が一番下、MTP と head が上
    assert.ok(L.vocab.y > L.grid.y + L.grid.h, name + ': embedding は格子の下');
    assert.ok(L.mtp.y + L.mtp.h <= L.column[s.layers - 1].y, name + ': MTP は列の上');
    assert.ok(L.head.y < L.grFinal.y, name + ': head は GR Read の上');
    for (let l = 1; l < s.layers; l++) {
      assert.ok(L.column[l].y < L.column[l - 1].y, name + ': 列 ' + l + ' は下から上');
      // 層パネルの上下はブロックの中だけで決まる (ブロックは行ごとに左 → 右)
      if (Math.floor(l / 4) === Math.floor((l - 1) / 4)) {
        assert.ok(L.panels[l].y <= L.panels[l - 1].y, name + ': panel ' + l + ' は下から上');
      }
    }
    // 2 行目以降のブロックは 1 行目の上。同じ行では右へ。
    const cols = L.blocks.filter((b) => b.y === L.blocks[0].y).length;
    for (let h = 1; h < L.blocks.length; h++) {
      if (h % cols === 0) assert.ok(L.blocks[h].y < L.blocks[h - 1].y, name + ': block ' + h + ' は 1 行上の左端');
      else assert.ok(L.blocks[h].x > L.blocks[h - 1].x, name + ': block ' + h + ' は右');
    }
  }
});

// ---- ツリーの参照整合 -------------------------------------------------------

test('多様なトポロジー: ツリーの ID・親子・items[lo..hi) が参照整合', () => {
  for (const [name, raw] of SHAPES) {
    const t = buildTree(raw);
    const s = t.shape;
    const idSet = new Set(t.nodes.map((n) => n.id));
    assert.equal(idSet.size, t.nodes.length, name + ': ノード id が重複');

    // 種類ごとの数
    const count = (type) => t.nodes.filter((n) => n.type === type).length;
    assert.equal(count('hybrid'), Math.ceil(s.layers / 4), name + ': hybrid');
    assert.equal(count('layer'), s.layers, name + ': layer');
    assert.equal(count('expert'), s.layers * s.n_expert, name + ': expert');
    assert.equal(count('stream'), s.layers * s.n_hc, name + ': stream');
    for (const type of ['mixer', 'moe', 'shared']) assert.equal(count(type), s.layers, name + ': ' + type);
    assert.equal(count('ple'), s.ple_layers.length, name + ': ple');
    assert.equal(t.items.length, s.layers * (s.n_expert + s.n_hc + 3) + s.ple_layers.length, name + ': items');

    // 親子は双方向に引ける。子の item 範囲は親の中、兄弟は隙間なく続く。
    for (const n of t.nodes) {
      if (n === t.root) {
        assert.equal(n.parent, null, name + ': root の親');
        assert.equal(n.lo, 0);
      } else {
        assert.equal(t.byId.get(n.parent.id), n.parent, name + ': ' + n.id + ' の親が byId に無い');
        assert.ok(n.parent.children.includes(n), name + ': ' + n.id + ' が親の children に無い');
        assert.equal(n.depth, n.parent.depth + 1, name + ': ' + n.id + ' の深さ');
        assert.ok(n.lo >= n.parent.lo && n.hi <= n.parent.hi, name + ': ' + n.id + ' の範囲が親の外');
      }
      assert.ok(n.lo >= 0 && n.lo <= n.hi && n.hi <= t.items.length, name + ': ' + n.id + ' の lo/hi');
      const kids = n.children || [];
      for (let i = 1; i < kids.length; i++) {
        assert.equal(kids[i - 1].hi, kids[i].lo, name + ': ' + n.id + ' の子 ' + i + ' に隙間');
      }
      if (kids.length) {
        assert.equal(kids[0].lo, n.lo + (n.item ? 1 : 0), name + ': ' + n.id + ' の最初の子の lo');
        assert.equal(kids[kids.length - 1].hi, n.hi, name + ': ' + n.id + ' の最後の子の hi');
      }
      if (n.item) assert.equal(n.hi - n.lo >= 1, true, name + ': ' + n.id + ' が item を持つのに空');
    }

    // items は item を持つノードの先行順と一致し、id で引ける
    const itemNodes = t.nodes.filter((n) => n.item);
    assert.equal(itemNodes.length, t.items.length, name + ': item ノード数');
    itemNodes.forEach((n, i) => assert.equal(t.items[i], n, name + ': items[' + i + '] の順序'));

    // item の中身は葉の種類だけで、値は範囲内
    for (const n of t.items) {
      assert.ok(LEAF_TYPES.includes(n.type), name + ': item の種類 ' + n.type);
      assert.ok(Number.isFinite(n.item.layer) && n.item.layer >= 0 && n.item.layer < s.layers, name + ': item の layer');
      if (n.type === 'expert') {
        assert.ok(Number.isInteger(n.item.expert) && n.item.expert >= 0 && n.item.expert < s.n_expert, name + ': expert 番号');
      } else if (n.type === 'stream') {
        assert.ok(Number.isInteger(n.item.stream) && n.item.stream >= 0 && n.item.stream < s.n_hc, name + ': stream 番号');
      } else if (n.type === 'mixer' || n.type === 'moe' || n.type === 'shared' || n.type === 'ple') {
        assert.ok(['attn', 'moe', 'shared', 'ple'].includes(n.item.block), name + ': block ' + n.item.block);
      } else {
        assert.fail(name + ': 想定外の item ノード ' + n.id);
      }
    }

    // ID の形 (代表): L<l> / L<l>/e<最後> / L<l>/s<最後> が引ける
    for (const l of [0, Math.floor(s.layers / 2), s.layers - 1]) {
      assert.equal(t.byId.get('L' + l).layer, l, name + ': L' + l);
      assert.equal(t.byId.get('L' + l + '/e' + (s.n_expert - 1)).index, s.n_expert - 1, name + ': L' + l + ' の最後の expert');
      assert.equal(t.byId.get('L' + l + '/s' + (s.n_hc - 1)).index, s.n_hc - 1, name + ': L' + l + ' の最後の stream');
      assert.equal(t.byId.get('L' + l + '/mix').item.block, 'attn', name + ': L' + l + ' の mixer');
    }
  }
});

// ---- ツリーと図の一致 -------------------------------------------------------

test('ツリーの item と図の選択項目は層ごとに一致する', () => {
  for (const [name, raw] of SHAPES) {
    const L = archLayout(raw);
    const t = buildTree(raw);
    const s = L.shape;
    for (let l = 0; l < s.layers; l++) {
      const p = L.panels[l];
      const want = new Set(nodeItems(t, t.byId.get('L' + l)).map(key));
      const got = archItemsInRect(L, p.x, p.y, p.x + p.w, p.y + p.h);
      const gotKeys = got.map(key);
      assert.equal(new Set(gotKeys).size, gotKeys.length, name + ': panel ' + l + ' の項目が重複');
      assert.equal(gotKeys.length, want.size, name + ': panel ' + l + ' の項目数');
      for (const k of gotKeys) assert.ok(want.has(k), name + ': panel ' + l + ' にツリーに無い項目 ' + k);
      const nPle = s.ple_layers.includes(l) ? 1 : 0;
      assert.equal(want.size, s.n_expert + s.n_hc + 3 + nPle, name + ': panel ' + l + ' の期待数');
      // archItemsInRect の順序は格子の行優先。expert だけは番号順に並ぶ。
      const experts = got.filter((it) => it.type === 'expert').map((it) => it.expert);
      assert.deepEqual(experts, Array.from({ length: s.n_expert }, (_, i) => i), name + ': panel ' + l + ' の expert 順');
    }
  }
});

test('ツリーのノード id → 図の矩形が引け、代表の点は同じ選択項目に当たる', () => {
  for (const [name, raw] of SHAPES) {
    const L = archLayout(raw);
    const t = buildTree(raw);
    const s = L.shape;

    // 情報ノードと層のノードは全部飛び先がある
    for (const n of t.nodes) {
      if (n.type === 'expert' && n.index > 0) continue;   // expert は代表だけ
      const r = archNodeRect(L, n.id);
      assert.ok(r && r.w > 0 && r.h > 0, name + ': archNodeRect(' + n.id + ')');
      for (const v of collectNumbers(r)) assert.ok(Number.isFinite(v[1]), name + ': ' + n.id + ' に非有限');
    }
    assert.equal(archNodeRect(L, 'model'), L.bounds, name + ': model は bounds');
    assert.equal(archNodeRect(L, 'H0'), L.blocks[0], name + ': H0');
    assert.equal(archNodeRect(L, 'H999'), null, name + ': 範囲外の H');
    assert.equal(archNodeRect(L, 'L' + s.layers), null, name + ': 範囲外の層');
    assert.equal(archNodeRect(L, 'unknown-id'), null, name + ': 知らない id');

    for (const l of [0, Math.floor(s.layers / 2), s.layers - 1]) {
      const p = L.panels[l];
      for (const e of [0, s.n_expert - 1]) {
        const r = archExpertRect(L, l, e);
        assert.ok(r.w > 0 && r.h > 0, name + ': expert のセル');
        assert.deepEqual(archHitTest(L, ...center(r)), { type: 'expert', layer: l, expert: e }, name + ': L' + l + '/e' + e);
        assert.equal(key(t.byId.get('L' + l + '/e' + e).item), key({ type: 'expert', layer: l, expert: e }));
        // 飛び先の矩形は expert のセルと重なる
        assert.ok(overlap(archNodeRect(L, 'L' + l + '/e' + e), r), name + ': L' + l + '/e' + e + ' の飛び先');
      }
      assert.throws(() => archExpertRect(L, l, -1), /out of range/, name + ': 負の expert');
      assert.throws(() => archExpertRect(L, l, s.n_expert), /out of range/, name + ': 範囲外の expert');

      for (const i of [0, s.n_hc - 1]) {
        const c = streamCircle(p.res0, i, s.n_hc);
        assert.ok(Number.isFinite(c.cx) && Number.isFinite(c.cy) && Number.isFinite(c.r) && c.r > 0,
          name + ': stream の円');
        assert.deepEqual(archHitTest(L, c.cx, c.cy), { type: 'stream', layer: l, stream: i }, name + ': L' + l + '/s' + i);
      }
      assert.deepEqual(archHitTest(L, ...center(p.mixer)), { type: 'block', layer: l, block: 'attn' }, name + ': mixer');
      assert.deepEqual(archHitTest(L, p.moe.x + 2, p.moe.y + 2), { type: 'block', layer: l, block: 'moe' }, name + ': moe');
      assert.deepEqual(archHitTest(L, ...center(p.shared)), { type: 'block', layer: l, block: 'shared' }, name + ': shared');
      if (p.ple) {
        assert.deepEqual(archHitTest(L, ...center(p.ple)), { type: 'block', layer: l, block: 'ple' }, name + ': ple');
      } else {
        assert.equal(s.ple_layers.includes(l), false);
      }
      // パネルの端 (部品の無い所) は panel
      assert.deepEqual(archHitTest(L, p.x + 2, p.y + 2), { type: 'panel', layer: l }, name + ': panel');
    }
  }
});

// ---- 決定性 -----------------------------------------------------------------

test('同じ形からは毎回同じ配置・同じツリーになる (決定的)', () => {
  for (const [name, raw] of SHAPES) {
    const a = archLayout(raw);
    const json = JSON.stringify(a);
    const b = archLayout(raw);
    assert.deepEqual(b, a, name + ': 2 回目の配置が違う');
    // 1 回目の戻り値を壊しても 3 回目は変わらない (形を共有していない)
    if (a.panels.length) a.panels[0].x = -999999;
    assert.equal(JSON.stringify(archLayout(raw)), json, name + ': 戻り値の書き換えが次の配置に漏れた');

    const ids = (tree) => JSON.stringify(tree.nodes.map((n) => [n.id, n.lo, n.hi, n.depth]));
    const t1 = buildTree(raw);
    const first = ids(t1);
    t1.byId.get('L0').label = 'broken';
    assert.equal(ids(buildTree(raw)), first, name + ': 2 回目のツリーが違う');
  }
});

// ---- 空・最小 ---------------------------------------------------------------

test('空の形 (layers=0) は「何も置かない」で通り、座標に NaN を出さない', () => {
  const L = archLayout({ layers: 0 });
  assert.equal(L.width, 0);
  assert.equal(L.height, 0);
  assert.equal(L.shape.layers, 0);
  for (const keyName of ['blocks', 'panels', 'column', 'groups']) assert.deepEqual(L[keyName], [], 'empty ' + keyName);
  assert.equal(L.step, 0);
  for (const [path, n] of collectNumbers(L)) assert.ok(Number.isFinite(n), 'empty: ' + path);
  for (const [path, r] of allRects(L)) {
    assert.equal(r.w, 0, 'empty: ' + path);
    assert.equal(r.h, 0, 'empty: ' + path);
  }
  assert.equal(L.input.x, 0);
  assert.equal(L.input.y, 0);
  assert.equal(archHitTest(L, 0, 0), null);
  assert.deepEqual(archItemsInRect(L, -100, -100, 100, 100), []);
  assert.equal(archNodeRect(L, 'model'), L.bounds);
  assert.equal(archNodeRect(L, 'L0'), null);
  assert.equal(archNodeRect(L, 'H0'), null);

  // 縮小・移動も有限
  for (const v of [fitView(L.bounds, { w: 800, h: 600 }), zoomAt({ s: 0, tx: 0, ty: 0 }, 2, 10, 20),
    zoomAt(undefined, NaN, NaN, NaN), toWorld({ s: 0 }, 5, 5), toWorld(undefined, 1, 2)]) {
    for (const [k, n] of Object.entries(v)) assert.ok(Number.isFinite(n), 'empty view.' + k);
  }
  // 描画は背景だけで止まる (例外を投げない)
  const g = fakeCtx();
  drawArch(g, L, { s: 1, tx: 0, ty: 0, w: 800, h: 600 }, {
    st: null, offExp: new Map(), offBlk: new Set(), offStr: new Map(),
    isSel: () => false, dragKeys: null, hover: null, vramOn: false, gridImage: () => null, dpr: 1,
  });
  assert.equal(g.calls.fillRect, 1);
  assert.equal(g.calls.drawImage, 0);

  // ツリーも情報ノードだけになる
  const t = buildTree({ layers: 0 });
  assert.equal(t.nodes.length, 8);
  assert.equal(t.items.length, 0);
  assert.equal(t.root.hi - t.root.lo, 0);
  assert.deepEqual(nodeItems(t, t.root), []);
  assert.deepEqual(visibleRows(t, new Set()).map((n) => n.id), ['emb', 'out']);
  for (const id of ['model', 'emb', 'vocab', 'ngram', 'out', 'grfinal', 'head', 'mtp']) {
    assert.ok(t.byId.has(id), 'empty tree: ' + id);
  }
  assert.equal(t.byId.get('L0'), undefined);
});

test('最小の形 (1 層 / 小さい expert) と既定値の埋め方を壊さない', () => {
  const raw = { layers: 1, n_expert: 2, n_expert_used: 1, n_hc: 2, n_embd: 8 };
  const L = archLayout(raw);
  const t = buildTree(raw);
  const s = L.shape;
  assert.equal(s.ple_layers.length, 0, '2 層以下は PLE 無しが既定');
  assert.equal(L.blocks.length, 1);
  assert.equal(L.panels.length, 1);
  assert.equal(L.column.length, 1);
  assert.equal(t.items.length, 2 + 2 + 3);
  assert.ok(archNodeRect(L, 'L0/e1').w > 0);
  assert.deepEqual(archHitTest(L, ...center(archExpertRect(L, 0, 1))), { type: 'expert', layer: 0, expert: 1 });
  assert.deepEqual(archHitTest(L, ...center(L.panels[0].mixer)), { type: 'block', layer: 0, block: 'attn' });
  for (const [path, n] of collectNumbers(L)) assert.ok(Number.isFinite(n), 'minimal: ' + path);

  // 空の配列や足りない layer_kinds は既定値で埋まる
  const t2 = buildTree({ layers: 2, layer_kinds: ['attn'], ple_layers: [] });
  assert.equal(t2.byId.get('L0').kind, 'attn');
  assert.equal(t2.byId.get('L1').kind, 'gdn');
  const t3 = buildTree({ layers: 4, layer_kinds: [], ple_layers: [0, 4, -1, 0] });
  assert.deepEqual(t3.shape.ple_layers, [0], '範囲外・重複の PLE は落とす');
  assert.equal(t3.byId.get('L3').kind, 'attn', '4 層ごとに QSA');
});

test('形でない入力は例外になり、NaN の配置を返さない', () => {
  for (const bad of [null, undefined, {}, [], 42, 'layers']) {
    assert.throws(() => archLayout(bad), (e) => e instanceof Error && /shape|layers/.test(e.message), 'archLayout ' + String(bad));
    assert.throws(() => buildTree(bad), (e) => e instanceof Error && /shape|layers/.test(e.message), 'buildTree ' + String(bad));
  }
  for (const bad of [{ layers: -1 }, { layers: 'many' }]) {
    assert.throws(() => archLayout(bad), /layers/, 'archLayout ' + JSON.stringify(bad));
    assert.throws(() => buildTree(bad), /layers/, 'buildTree ' + JSON.stringify(bad));
  }
  // 小数は切り捨て (intOr) てから層数になる
  assert.equal(archLayout({ layers: 1.5 }).shape.layers, 1);
  // NaN / Infinity の layers も 0 として扱われ例外になる
  assert.throws(() => archLayout({ layers: NaN }), /layers/);
  assert.throws(() => archLayout({ layers: Infinity }), /layers/);
});

test('streamCircle: 退化した n でも有限の円を返す', () => {
  const box = { x: 10, y: 20, w: 100, h: 40 };
  for (const n of [0, -3, NaN, Infinity, undefined, 'x']) {
    const c = streamCircle(box, 0, n);
    assert.ok(Number.isFinite(c.cx) && Number.isFinite(c.cy) && Number.isFinite(c.r), 'n=' + String(n));
    assert.ok(c.r > 0, 'n=' + String(n) + ': 半径は正');
    assert.ok(c.cx - c.r >= box.x - 1e-9 && c.cx + c.r <= box.x + box.w + 1e-9, 'n=' + String(n) + ': 箱の内外');
  }
  // 1 本なら箱の真ん中
  const one = streamCircle(box, 0, 1);
  assert.equal(one.cx, box.x + box.w / 2);
  assert.equal(one.cy, box.y + box.h / 2);
});

test('PANEL の寸法は全トポロジーで同じ (描画とテストが同じ値を使う)', () => {
  for (const [name, raw] of SHAPES) {
    const L = archLayout(raw);
    for (const p of L.panels) {
      assert.equal(p.w, PANEL.w, name + ': panel の幅');
      assert.equal(p.h, PANEL.h, name + ': panel の高さ');
    }
  }
});
