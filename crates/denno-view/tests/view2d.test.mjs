// denno-view /2d の純ロジックの単体テスト (DOM 非依存)。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの:
//   1. 形 → レイアウト (層数 / expert 格子 / stream / PLE)
//   2. テレメトリのフレーム → 色の値 (RouterTopK / RoleAct / LayerAct / ExpertResidency)
//   3. 選択 (クリック・ドラッグ) → POST /v1/intervene の本文
//   4. モック店の振る舞い (epoch / 記録 / snapshot / apply)
import test from 'node:test';
import assert from 'node:assert/strict';

import { encodeFrame } from '../src/frames.mjs';
import {
  ACT_BUCKETS, BLOCKS, DEFAULT_SHAPE, DEFAULT_SHAPE_RAW, GEOM, STEER_MODES,
  actBuckets, addItem, clearSelection, createMapState, createMockStore, createSelection,
  expertCellRect, hasItem, heat, hitTest, itemsInRect, layoutFromShape, mergeBodies,
  mixerKind, normaliseShape, rampRGB, residencyBits, roleForBlock, routerWeights,
  selectionCount, selectionFromBody, selectionItems, selectionToBody, steerApplyBody,
  steerSummary, streamValue, toggleItem, removeItem,
} from '../src/view2d.mjs';

// ---- 1. 形 → レイアウト ---------------------------------------------------

test('normaliseShape: 既定値で埋め、層種別は interval 4 で full になる', () => {
  const s = normaliseShape({ layers: 8, n_expert: 512, n_expert_used: 10, n_hc: 4, n_embd: 2560 });
  assert.equal(s.layers, 8);
  assert.equal(s.n_expert, 512);
  assert.equal(s.n_expert_used, 10);
  assert.equal(s.n_hc, 4);
  assert.equal(s.n_embd, 2560);
  assert.deepEqual([...s.layer_kinds], ['gdn', 'gdn', 'gdn', 'attn', 'gdn', 'gdn', 'gdn', 'attn']);
  assert.deepEqual([...s.ple_layers], [2], 'PLE は既定で層 2');

  // 欠けたキーは既定に落ちる (layers だけ必須)
  const d = normaliseShape({ layers: 3 });
  assert.equal(d.n_expert, DEFAULT_SHAPE_RAW.n_expert);
  assert.equal(d.n_expert_used, DEFAULT_SHAPE_RAW.n_expert_used);
  assert.equal(d.n_hc, DEFAULT_SHAPE_RAW.n_hc);
  assert.equal(d.n_embd, DEFAULT_SHAPE_RAW.n_embd);
  assert.deepEqual([...d.ple_layers], [2]);

  // 明示の layer_kinds を尊重し、表記ゆれを寄せる
  const k = normaliseShape({ layers: 4, layer_kinds: ['ssm', 'full_attention', 'G DN', 'Attention'] });
  assert.deepEqual([...k.layer_kinds], ['gdn', 'attn', 'gdn', 'attn']);
  assert.equal(mixerKind('full attention'), 'attn');
  assert.equal(mixerKind('QSA'), 'attn');
  assert.equal(mixerKind('gdn'), 'gdn');
  assert.equal(mixerKind(''), 'gdn');

  // 不正
  assert.throws(() => normaliseShape(null), /object required/);
  assert.throws(() => normaliseShape({ layers: 0 }), /layers/);
  assert.throws(() => normaliseShape({ layers: -3 }), /layers/);

  // 範囲外の PLE 指定は落とす
  const p = normaliseShape({ layers: 4, ple_layers: [-1, 1, 1, 9, 2] });
  assert.deepEqual([...p.ple_layers], [1, 2]);

  assert.equal(DEFAULT_SHAPE.layers, 48);
  assert.equal(DEFAULT_SHAPE.n_expert, 512);
  assert.equal(DEFAULT_SHAPE.layer_kinds.filter((x) => x === 'attn').length, 12);
});

test('layoutFromShape: 1 層 1 行、expert は n_expert 個、stream は n_hc 本', () => {
  const shape = { layers: 3, n_expert: 8, n_expert_used: 2, n_hc: 2, n_embd: 64, layer_kinds: ['gdn', 'gdn', 'full'] };
  const lay = layoutFromShape(shape);
  assert.equal(lay.rows.length, 3);
  assert.deepEqual(lay.rows.map((r) => r.layer), [0, 1, 2]);
  assert.deepEqual(lay.rows.map((r) => r.kind), ['gdn', 'gdn', 'attn']);

  for (const row of lay.rows) {
    assert.equal(row.experts.n, 8);
    assert.equal(row.experts.cols, 8, '8 本なら 1 行 8 列');
    assert.equal(row.experts.rows, 1);
    assert.equal(row.streams.length, 2);
    // 行の中に収まる
    assert.ok(row.experts.x >= 0);
    assert.ok(row.experts.x + row.experts.w <= lay.width);
    assert.ok(row.experts.y >= row.y);
    assert.ok(row.experts.y + row.experts.h <= row.y + row.h);
    assert.ok(row.strip.y >= row.experts.y + row.experts.h - 1);
    assert.ok(row.strip.y + row.strip.h <= row.y + row.h);
    // ブロックは重ならない (x 昇順)
    const xs = [row.mixer, row.router, row.experts, row.shared, row.ple];
    for (let i = 1; i < xs.length; i++) {
      assert.ok(xs[i - 1].x + xs[i - 1].w <= xs[i].x, 'block ' + i + ' overlaps');
    }
    assert.ok(row.streams[0].x >= row.ple.x + row.ple.w);
  }

  // PLE は層 2 だけ active
  assert.deepEqual(lay.rows.map((r) => r.ple.active), [false, false, true]);
  assert.equal(lay.height, GEOM.padY * 2 + 3 * lay.rowH);
  assert.ok(lay.width > 200 && lay.width < 700, 'width ' + lay.width);

  // expert のセルは行の格子に載る
  const r0 = expertCellRect(lay, 0, 0);
  assert.deepEqual(r0, { x: lay.rows[0].experts.x, y: lay.rows[0].experts.y, w: 3, h: 3 });
  const r7 = expertCellRect(lay, 0, 7);
  assert.equal(r7.x, lay.rows[0].experts.x + 7 * 4);
  assert.equal(r7.y, lay.rows[0].experts.y);
  assert.throws(() => expertCellRect(lay, 9, 0), /layer/);
  assert.throws(() => expertCellRect(lay, 0, 8), /expert/);
});

test('layoutFromShape: 実形 48 層 512 expert は 64x8 の格子に収まる', () => {
  const lay = layoutFromShape(DEFAULT_SHAPE);
  assert.equal(lay.rows.length, 48);
  assert.equal(lay.cols, 64);
  const ex = lay.rows[0].experts;
  assert.equal(ex.rows, 8);
  assert.equal(ex.cols * ex.rows, 512);
  // 最後のセルも格子の中
  const last = expertCellRect(lay, 47, 511);
  assert.ok(last.x + last.w <= ex.x + ex.w);
  assert.ok(last.y + last.h <= lay.rows[47].experts.y + lay.rows[47].experts.h);
  // 行の高さは格子 + strip + 余白
  assert.ok(lay.rowH >= ex.h + GEOM.strip.h);
  assert.equal(lay.height, GEOM.padY * 2 + 48 * lay.rowH);
  // 512 本でも 3px セルなので幅は 600px 未満 (3 カラムに収める)
  assert.ok(lay.width < 620, 'width ' + lay.width);
});

test('hitTest: セル / ブロック / stream を正しく指す', () => {
  const lay = layoutFromShape({ layers: 2, n_expert: 8, n_expert_used: 2, n_hc: 2, n_embd: 64 });
  const row = lay.rows[1];

  assert.deepEqual(hitTest(lay, row.experts.x + 1, row.experts.y + 1),
    { type: 'expert', layer: 1, expert: 0 });
  const c = expertCellRect(lay, 1, 5);
  assert.deepEqual(hitTest(lay, c.x + 1, c.y + 1), { type: 'expert', layer: 1, expert: 5 });
  assert.deepEqual(hitTest(lay, row.mixer.x + 3, row.mixer.y + 3), { type: 'block', layer: 1, block: 'attn' });
  assert.deepEqual(hitTest(lay, row.router.x + 3, row.router.y + 3), { type: 'block', layer: 1, block: 'moe' });
  assert.deepEqual(hitTest(lay, row.shared.x + 3, row.shared.y + 3), { type: 'block', layer: 1, block: 'shared' });
  assert.deepEqual(hitTest(lay, row.ple.x + 3, row.ple.y + 3), { type: 'block', layer: 1, block: 'ple' });
  assert.deepEqual(hitTest(lay, row.streams[1].x + 1, row.streams[1].y + 1),
    { type: 'stream', layer: 1, stream: 1 });

  assert.equal(hitTest(lay, -5, 5), null);
  assert.equal(hitTest(lay, 5, lay.height + 10), null);
});

test('itemsInRect: ドラッグの矩形の中の expert を全部返す (端は含む)', () => {
  const lay = layoutFromShape({ layers: 2, n_expert: 8, n_expert_used: 2, n_hc: 2, n_embd: 64 });
  const ex = lay.rows[0].experts;
  // 先頭 3 セル (0,1,2) を覆う
  const a = expertCellRect(lay, 0, 0);
  const c = expertCellRect(lay, 0, 2);
  const items = itemsInRect(lay, a.x, a.y, c.x + c.w, c.y + c.h);
  const experts = items.filter((x) => x.type === 'expert');
  assert.deepEqual(experts.map((x) => x.expert), [0, 1, 2]);
  assert.ok(items.every((x) => x.layer === 0));

  // 行全体 (ブロックも stream も含む)
  const all = itemsInRect(lay, 0, lay.rows[1].y, lay.width, lay.rows[1].y + lay.rowH);
  assert.equal(all.filter((x) => x.type === 'expert').length, 8);
  assert.deepEqual(all.filter((x) => x.type === 'block').map((x) => x.block), ['attn', 'moe', 'shared', 'ple']);
  assert.equal(all.filter((x) => x.type === 'stream').length, 2);
  // 空の矩形
  assert.deepEqual(itemsInRect(lay, -10, -10, -5, -5), []);
});

// ---- 2. フレーム → 色の値 -------------------------------------------------

test('routerWeights: top-k だけが立ち、層内最大で 1 に正規化される', () => {
  const w = routerWeights({ kind: 'RouterTopK', layer: 3, experts: [502, 7, 511], weights: [0.4, 0.2, 0.1] }, 512);
  assert.equal(w.length, 512);
  assert.equal(w[502], 1);
  assert.ok(Math.abs(w[7] - 0.5) < 1e-6);
  assert.ok(Math.abs(w[511] - 0.25) < 1e-6);
  // 選ばれていない expert は 0
  let nonzero = 0;
  for (const v of w) if (v > 0) nonzero++;
  assert.equal(nonzero, 3);
  // 範囲外・非数・重み 0 は捨てる
  const bad = routerWeights({ kind: 'RouterTopK', layer: 0, experts: [9999, 1, 2], weights: [1, NaN, 0] }, 512);
  assert.equal(bad[1], 0);
  assert.equal(bad[2], 0);
  // 別 kind は空
  assert.equal(routerWeights({ kind: 'Token', layer: 0 }, 8).length, 8);
});

test('actBuckets: idx を n_embd で 64 ビンに畳み、ビン内最大を取る', () => {
  const f = { kind: 'LayerAct', layer: 5, idx: [0, 5, 7, 511, 512], val: [0.5, 2.0, 1.0, 0.25, 9.0] };
  const b = actBuckets(f, 512);
  assert.equal(b.length, ACT_BUCKETS);
  // idx 0..7 はビン 0 (512/64 = 8 幅)。最大 2.0 が正規化されて 1。
  assert.equal(b[0], 1);
  assert.equal(b[63], 0.125);
  // 範囲外の idx は捨てる (n_embd=512 の idx 512 は入らない)
  let nonzero = 0;
  for (const v of b) if (v > 0) nonzero++;
  assert.equal(nonzero, 2);
  // 全部 0 のフレームは 0 のまま (0/0 を作らない)
  const z = actBuckets({ kind: 'LayerAct', layer: 0, idx: [1, 2], val: [0, 0] }, 512);
  assert.ok([...z].every((v) => v === 0));
});

test('residencyBits / roleForBlock / streamValue', () => {
  const bits = residencyBits({ kind: 'ExpertResidency', layer: 1, vram: [0, 3, 511, 9999] }, 512);
  assert.equal(bits[0], 1);
  assert.equal(bits[3], 1);
  assert.equal(bits[511], 1);
  assert.equal(bits[1], 0);
  let n = 0;
  for (const v of bits) n += v;
  assert.equal(n, 3);

  const role = { attn: 0.9, gdn: 0.2, router: 0.4, shex: 0.7, ple: 0.1 };
  assert.equal(roleForBlock(role, 'attn', 'attn'), 0.9);
  assert.equal(roleForBlock(role, 'attn', 'gdn'), 0.2);
  assert.equal(roleForBlock(role, 'moe', 'gdn'), 0.4);
  assert.equal(roleForBlock(role, 'shared', 'attn'), 0.7);
  assert.equal(roleForBlock(role, 'ple', 'attn'), 0.1);
  assert.equal(roleForBlock(null, 'attn', 'gdn'), 0);
  assert.equal(streamValue(role), 0.9);
  assert.equal(streamValue({}), 0);
});

test('heat / rampRGB: 0 は暗色、1 は赤、範囲外は端に張り付く', () => {
  assert.deepEqual(rampRGB(0), [8, 19, 28]);
  assert.deepEqual(rampRGB(1), [255, 59, 48]);
  assert.deepEqual(rampRGB(-3), rampRGB(0));
  assert.deepEqual(rampRGB(7), rampRGB(1));
  const mid = rampRGB(0.5);
  assert.ok(mid[2] > mid[0], '中間は青が強い: ' + mid.join(','));
  assert.equal(heat(0.5), 'rgb(' + mid.join(',') + ')');
  assert.equal(heat(2, 2), heat(1));
  assert.equal(heat(1, 0), heat(1), 'scale 0 は 1 とみなす');
});

test('createMapState: token モードは置換、accum は加算 + decay', () => {
  const shape = { layers: 2, n_expert: 8, n_expert_used: 2, n_hc: 1, n_embd: 64 };
  const st = createMapState(shape, { decay: 0.5 });
  assert.equal(st.mode, 'token');

  const f = { kind: 'RouterTopK', layer: 0, experts: [1, 4], weights: [1, 0.5] };
  st.onFrame(f);
  assert.equal(st.experts[0][1], 1);
  assert.equal(st.experts[0][4], 0.5);
  // token モード: 次のフレームで置き換わる
  st.onFrame({ kind: 'RouterTopK', layer: 0, experts: [2], weights: [0.25] });
  assert.equal(st.experts[0][1], 0);
  assert.equal(st.experts[0][2], 1, '置換 (層内最大で正規化)');
  assert.equal(st.experts[0][4], 0);

  // accum モード: 加算され、Token で減衰する
  const ac = createMapState(shape, { decay: 0.5, mode: 'accum' });
  ac.onFrame(f);
  assert.equal(ac.experts[0][1], 1);
  ac.onFrame({ kind: 'Token', session: 1, token: 3, pos: 1 });
  assert.ok(Math.abs(ac.experts[0][1] - 0.5) < 1e-6, 'decay で半分');
  assert.equal(ac.token, 1);
  ac.onFrame(f);
  assert.ok(Math.abs(ac.experts[0][1] - 1.5) < 1e-6, '前の残り + 新しい重み');

  // モード切替で消える (token と accum を混ぜない)
  ac.setMode('token');
  assert.equal(ac.mode, 'token');
  assert.equal(ac.experts[0][1], 0);

  // 層の範囲外は無視
  assert.equal(st.onFrame({ kind: 'RouterTopK', layer: 99, experts: [0], weights: [1] }), false);
  assert.equal(st.onFrame({ kind: 'Nope', layer: 0 }), false);
});

test('createMapState: RoleAct / LayerAct / ExpertResidency を取り込み、reset で消える', () => {
  const st = createMapState({ layers: 2, n_expert: 8, n_hc: 2, n_embd: 64 }, {});
  st.onFrame({ kind: 'RoleAct', layer: 1, roles: { attn: 0, gdn: 1, router: 0.5, shex: 0.25, ple: 0 } });
  assert.deepEqual(st.role[1], { attn: 0, gdn: 1, router: 0.5, shex: 0.25, ple: 0 });
  st.onFrame({ kind: 'LayerAct', layer: 0, idx: [0, 63], val: [1, 0.5] });
  assert.equal(st.act[0][0], 1);
  assert.equal(st.act[0][63], 0.5, 'idx 63 はビン 63、値は層内最大 1 で正規化');
  st.onFrame({ kind: 'ExpertResidency', layer: 0, vram: [2, 3] });
  assert.equal(st.residency[0][2], 1);
  st.token = 5;
  st.reset();
  assert.equal(st.token, 0);
  assert.equal(st.role[1].gdn, 0);
  assert.equal(st.residency[0][2], 0);
  assert.equal(st.act[0][0], 0);
});

test('フレームコーデックを通した RouterTopK でも同じ値になる (frames.mjs と接続)', () => {
  const bytes = encodeFrame({ kind: 'RouterTopK', layer: 2, experts: [3, 9], weights: [0.75, 0.25] });
  // decodeFrame を page 側で使う。ここでは encode → decode してから食わせる。
  return import('../src/frames.mjs').then(({ decodeFrames }) => {
    const f = decodeFrames(bytes)[0];
    const st = createMapState({ layers: 3, n_expert: 16, n_hc: 1, n_embd: 32 }, {});
    st.onFrame(f);
    assert.equal(st.experts[2][3], 1);
    assert.ok(Math.abs(st.experts[2][9] - 1 / 3) < 1e-3);
  });
});

// ---- 3. 選択 → リクエスト本文 ---------------------------------------------

test('selectionToBody: 決定的な順序・重複なし・3 配列', () => {
  const sel = createSelection();
  addItem(sel, { type: 'expert', layer: 5, expert: 9 });
  addItem(sel, { type: 'expert', layer: 5, expert: 9 });   // 重複は 1 つ
  addItem(sel, { type: 'expert', layer: 1, expert: 400 });
  addItem(sel, { type: 'block', layer: 5, block: 'moe' });
  addItem(sel, { type: 'block', layer: 0, block: 'attn' });
  addItem(sel, { type: 'stream', layer: 5, stream: 3 });
  addItem(sel, { type: 'stream', layer: 5, stream: 0 });
  assert.equal(selectionCount(sel), 6);
  assert.deepEqual(selectionToBody(sel), {
    experts_off: [[1, 400], [5, 9]],
    blocks_off: [[0, 'attn'], [5, 'moe']],
    streams_off: [[5, 0], [5, 3]],
  });
  // 本文 → 選択 → 本文で往復
  assert.deepEqual(selectionToBody(selectionFromBody(selectionToBody(sel))), selectionToBody(sel));
  // toggle
  assert.equal(toggleItem(sel, { type: 'expert', layer: 5, expert: 9 }), false);
  assert.equal(hasItem(sel, { type: 'expert', layer: 5, expert: 9 }), false);
  assert.equal(toggleItem(sel, { type: 'expert', layer: 5, expert: 9 }), true);
  // remove / clear
  removeItem(sel, { type: 'block', layer: 0, block: 'attn' });
  assert.deepEqual(selectionToBody(sel).blocks_off, [[5, 'moe']]);
  clearSelection(sel);
  assert.equal(selectionCount(sel), 0);
  assert.deepEqual(selectionToBody(sel), { experts_off: [], blocks_off: [], streams_off: [] });
  // 未知のブロックは例外
  assert.throws(() => addItem(createSelection(), { type: 'block', layer: 0, block: 'nope' }), /unknown block/);
});

test('mergeBodies: off 済みと選択の和 (重複は 1 つ)', () => {
  const a = { experts_off: [[1, 2]], blocks_off: [[0, 'moe']], streams_off: [] };
  const b = { experts_off: [[1, 2], [3, 4]], blocks_off: [], streams_off: [[3, 1]] };
  assert.deepEqual(mergeBodies(a, b), {
    experts_off: [[1, 2], [3, 4]],
    blocks_off: [[0, 'moe']],
    streams_off: [[3, 1]],
  });
  assert.deepEqual(mergeBodies(null, null), { experts_off: [], blocks_off: [], streams_off: [] });
});

test('ドラッグ (itemsInRect) → 選択 → POST /v1/intervene の本文', () => {
  const lay = layoutFromShape({ layers: 48, n_expert: 512, n_hc: 4, n_embd: 2560 });
  const row = lay.rows[10];
  // expert 格子の 2 列 x 2 行ぶんをドラッグ
  const a = expertCellRect(lay, 10, 0);
  const b = expertCellRect(lay, 10, 65);          // 2 行目の 1 列目
  const items = itemsInRect(lay, a.x, a.y, b.x + b.w, b.y + b.h);
  const sel = createSelection();
  for (const it of items) addItem(sel, it);
  const body = selectionToBody(sel);
  // 64 列なので expert 0,1,64,65
  assert.deepEqual(body.experts_off, [[10, 0], [10, 1], [10, 64], [10, 65]]);
  assert.deepEqual(body.blocks_off, []);
  assert.deepEqual(body.streams_off, []);
  // 選択の item 一覧も同じ
  assert.deepEqual(selectionItems(sel).map((x) => x.expert), [0, 1, 64, 65]);
});

// ---- 4. ステアリング本文とモック店 ----------------------------------------

test('steerApplyBody: 契約どおりの本文にする / 不正は例外', () => {
  assert.deepEqual(
    steerApplyBody({ terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -1 }], layers: [4, 47], alpha: 0.5, mode: 'project' }),
    { terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -1 }], layers: [4, 47], alpha: 0.5, mode: 'project' },
  );
  // 既定値
  assert.deepEqual(steerApplyBody({ terms: [{ snapshot: 'A', coef: '2' }], layers: [0, 3] }),
    { terms: [{ snapshot: 'A', coef: 2 }], layers: [0, 3], alpha: 1, mode: 'subtract' });

  assert.throws(() => steerApplyBody({ terms: [] }), /at least one/);
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: '', coef: 1 }] }), /snapshot/);
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef: 0 }] }), /non-zero/);
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef: 1 }], layers: [5, 2] }), /from must be <= to/);
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef: 1 }], layers: [-1, 2] }), />= 0/);
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef: 1 }], layers: [0, 2], alpha: -1 }), /alpha/);
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef: 1 }], layers: [0, 2], mode: 'add' }), /mode/);
  assert.deepEqual([...STEER_MODES], ['subtract', 'project']);

  assert.equal(
    steerSummary({ terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -1 }], layers: [0, 47], alpha: 1, mode: 'subtract' }),
    'A +1 · B -1 · subtract · α 1 · layers 0-47');
  assert.equal(steerSummary(null), '');
});

test('モック店: shape と intervene (POST 全置換 / DELETE 全消し / epoch)', () => {
  const m = createMockStore({ layers: 4, n_expert: 8, n_expert_used: 2, n_hc: 2, n_embd: 16 });

  const shape = m.handle('GET', '/v1/model/shape');
  assert.equal(shape.status, 200);
  assert.equal(shape.body.layers, 4);
  assert.equal(shape.body.n_expert, 8);
  assert.equal(shape.body.n_hc, 2);
  assert.equal(shape.body.n_embd, 16);
  assert.deepEqual(shape.body.layer_kinds, ['gdn', 'gdn', 'gdn', 'attn']);

  let iv = m.handle('GET', '/v1/intervene');
  assert.deepEqual(iv.body, { epoch: 0, experts_off: [], blocks_off: [], streams_off: [] });

  // POST は全置換 (前の内容は残らない)
  let r = m.handle('POST', '/v1/intervene', { experts_off: [[1, 2], [1, 2], [3, 7]], blocks_off: [[0, 'attn']], streams_off: [[2, 1]] });
  assert.equal(r.status, 200);
  assert.equal(r.body.epoch, 1);
  assert.deepEqual(r.body.experts_off, [[1, 2], [3, 7]], '重複は落ちる');
  r = m.handle('POST', '/v1/intervene', { experts_off: [[0, 1]] });
  assert.equal(r.body.epoch, 2);
  assert.deepEqual(r.body.blocks_off, []);
  assert.deepEqual(r.body.streams_off, []);
  iv = m.handle('GET', '/v1/intervene');
  assert.deepEqual(iv.body, { epoch: 2, experts_off: [[0, 1]], blocks_off: [], streams_off: [] });

  // DELETE は全消し + epoch
  r = m.handle('DELETE', '/v1/intervene');
  assert.equal(r.status, 200);
  assert.equal(r.body.epoch, 3);
  assert.deepEqual(m.handle('GET', '/v1/intervene').body.experts_off, []);

  // 不正は 400 で状態を変えない
  for (const bad of [
    { experts_off: [[9, 0]] },                       // layer 範囲外
    { experts_off: [[0, 99]] },                      // expert 範囲外
    { blocks_off: [[0, 'router']] },                 // 未知のブロック
    { streams_off: [[0, 2]] },                       // stream 範囲外 (n_hc=2)
    { experts_off: [[0]] },                          // 対になっていない
    { experts_off: [[0, 1]], blocks_off: 'x' },      // 配列でない
  ]) {
    const e = m.handle('POST', '/v1/intervene', bad);
    assert.equal(e.status, 400, JSON.stringify(bad));
    assert.ok(e.body.error.message, 'error message for ' + JSON.stringify(bad));
  }
  assert.equal(m.handle('GET', '/v1/intervene').body.epoch, 3, '400 は epoch を進めない');
});

test('モック店: 記録 → snapshot → 削除 → apply / clear', () => {
  const m = createMockStore({ layers: 4, n_expert: 8, n_hc: 1, n_embd: 16 }, { now: () => '2026-11-10T00:00:00.000Z' });

  assert.deepEqual(m.handle('GET', '/v1/steer').body, { recording: null, snapshots: [], apply: null });
  assert.equal(m.handle('POST', '/v1/steer/record/stop').status, 409, '記録していないのに stop');
  assert.equal(m.handle('POST', '/v1/steer/record', {}).status, 400, '名前なし');
  assert.equal(m.handle('POST', '/v1/steer/record', { name: 'A' }).status, 200);
  assert.equal(m.handle('POST', '/v1/steer/record', { name: 'B' }).status, 409, '二重開始');

  // トークンが進む
  m.tick(3);
  assert.equal(m.handle('GET', '/v1/steer').body.recording.tokens, 3);
  assert.deepEqual(m.handle('GET', '/v1/steer').body.recording, { name: 'A', tokens: 3, created: '2026-11-10T00:00:00.000Z' });

  const snap = m.handle('POST', '/v1/steer/record/stop');
  assert.equal(snap.status, 200);
  assert.deepEqual(snap.body, { name: 'A', tokens: 3, created: '2026-11-10T00:00:00.000Z', layers: [0, 3] });
  // tick は記録外では tokens を増やさない
  m.tick(2);
  assert.equal(m.handle('GET', '/v1/steer').body.snapshots[0].tokens, 3);

  // 2 つ目
  m.handle('POST', '/v1/steer/record', { name: 'B' });
  m.tick(1);
  m.handle('POST', '/v1/steer/record/stop');
  assert.deepEqual(m.handle('GET', '/v1/steer').body.snapshots.map((s) => s.name), ['A', 'B']);
  assert.equal(m.handle('POST', '/v1/steer/record', { name: 'A' }).status, 409, '同名は拒否');

  // apply: 本文をそのまま保存し、未知の snapshot は 404
  const body = { terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -1 }], layers: [0, 3], alpha: 0.5, mode: 'subtract' };
  assert.equal(m.handle('POST', '/v1/steer/apply', { ...body, terms: [{ snapshot: 'Z', coef: 1 }] }).status, 404);
  const ap = m.handle('POST', '/v1/steer/apply', body);
  assert.equal(ap.status, 200);
  assert.deepEqual(ap.body.terms, body.terms);
  assert.equal(ap.body.applied, '2026-11-10T00:00:00.000Z');
  assert.deepEqual(m.handle('GET', '/v1/steer').body.apply.terms, body.terms);
  assert.equal(m.handle('POST', '/v1/steer/apply', { ...body, mode: 'add' }).status, 400);

  // 削除
  assert.equal(m.handle('DELETE', '/v1/steer/snapshots/Z').status, 404);
  assert.equal(m.handle('DELETE', '/v1/steer/snapshots/A').status, 200);
  assert.deepEqual(m.handle('GET', '/v1/steer').body.snapshots.map((s) => s.name), ['B']);
  assert.equal(m.handle('DELETE', '/v1/steer/snapshots/A').status, 404);

  // clear
  assert.equal(m.handle('DELETE', '/v1/steer/apply').status, 200);
  assert.equal(m.handle('GET', '/v1/steer').body.apply, null);

  // 未知の経路
  assert.equal(m.handle('GET', '/v1/nope').status, 404);
  assert.equal(m.handle('PUT', '/v1/intervene').status, 404);
});

test('モック店: 返り値を書き換えても内部状態は壊れない (コピーを返す)', () => {
  const m = createMockStore({ layers: 2, n_expert: 4, n_hc: 1, n_embd: 8 });
  m.handle('POST', '/v1/intervene', { experts_off: [[0, 1]] });
  const got = m.handle('GET', '/v1/intervene');
  got.body.experts_off.push([1, 2]);
  assert.deepEqual(m.handle('GET', '/v1/intervene').body.experts_off, [[0, 1]]);
  const shape = m.handle('GET', '/v1/model/shape');
  shape.body.n_expert = 999;
  assert.equal(m.handle('GET', '/v1/model/shape').body.n_expert, 4);
});

test('BLOCKS と既定形の整合', () => {
  assert.deepEqual([...BLOCKS], ['attn', 'moe', 'shared', 'ple']);
  assert.equal(DEFAULT_SHAPE.n_hc, 4);
  assert.equal(DEFAULT_SHAPE.n_expert_used, 10);
  assert.equal(DEFAULT_SHAPE.n_expert_used <= DEFAULT_SHAPE.n_expert, true);
});
