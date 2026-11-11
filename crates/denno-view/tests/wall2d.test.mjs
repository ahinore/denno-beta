// denno-view /2d の「層の壁」(task 235) の純ロジックのテスト。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの:
//   1. 地図の実面積 → タイルの格子 (1920x1080 / 1366x768 で 8x6、狭い幅では縮退)
//   2. 画素 → (層, expert) のヒットテスト (タイルの四隅と全層)
//   3. 色の尺度 (暗い → 明るい 1 本、赤は off だけ)
//   4. 選択 → POST /v1/intervene の本文が今までと同じ形
//   5. 深さ方向の帯 (mixer / shared / PLE / LayerAct)
//   6. モックの合成テレメトリ (サーバー無しでも壁が動く)
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_SHAPE, DEFAULT_SHAPE_RAW, MAP_CHROME, WALL_COLORS, WALL_GEOM,
  actLevel, addItem, createMapState, createMockStore, createSelection, depthValue, heat,
  mapPaneSize, mockTokenFrames, rampRGB, selectionCount, selectionToBody, wallExpertRect,
  wallHitTest, wallItemsInRect, wallLayout,
} from '../src/view2d.mjs';

const FULL = wallLayout(DEFAULT_SHAPE, mapPaneSize({ w: 1366, h: 768 }));

// ---- 1. 実面積 → 格子 -----------------------------------------------------

test('mapPaneSize: 1920x1080 と 1366x768 の地図の実面積 (CSS と同じ値)', () => {
  const a = mapPaneSize({ w: 1920, h: 1080 });
  assert.deepEqual({ w: a.w, h: a.h }, { w: 1220, h: 965 });
  const b = mapPaneSize({ w: 1366, h: 768 });
  assert.deepEqual({ w: b.w, h: b.h }, { w: 666, h: 653 });
  // 列を畳むと地図が広くなる
  const c = mapPaneSize({ w: 1366, h: 768 }, { chat: false, ctrl: false });
  assert.ok(c.w > b.w + 600, 'collapsed width ' + c.w);
  assert.equal(c.h, b.h);
  assert.equal(MAP_CHROME.chatW + MAP_CHROME.ctrlW, 678);
});

test('wallLayout: 1920x1080 と 1366x768 は 8 列 x 6 行でスクロール無し', () => {
  for (const vp of [{ w: 1920, h: 1080 }, { w: 1366, h: 768 }]) {
    const pane = mapPaneSize(vp);
    const lay = wallLayout(DEFAULT_SHAPE, pane);
    assert.equal(lay.cols, 8, vp.w + ': cols');
    assert.equal(lay.rows, 6, vp.w + ': rows');
    assert.equal(lay.tiles.length, 48);
    assert.equal(lay.fits, true, vp.w + ': must fit');
    assert.ok(lay.cellW >= WALL_GEOM.minCell, vp.w + ': cellW ' + lay.cellW);
    assert.ok(lay.cellH >= WALL_GEOM.minCell, vp.w + ': cellH ' + lay.cellH);
    assert.ok(lay.width <= pane.w, vp.w + ': width ' + lay.width + ' > ' + pane.w);
    assert.ok(lay.height <= pane.h, vp.w + ': height ' + lay.height + ' > ' + pane.h);
    assert.equal(lay.heatCols, 32);
    assert.equal(lay.heatRows, 16);

    // reading order: 層 0 が左上、層 1 が右、層 8 が 1 つ下の行の左
    const t0 = lay.tiles[0], t1 = lay.tiles[1], t8 = lay.tiles[8], t47 = lay.tiles[47];
    assert.ok(t1.x > t0.x && Math.abs(t1.y - t0.y) < 1e-9, 'layer 1 is right of layer 0');
    assert.ok(t8.y > t0.y && Math.abs(t8.x - t0.x) < 1e-9, 'layer 8 starts row 2');
    assert.ok(t47.x > t0.x && t47.y > t0.y, 'layer 47 is bottom right');
    // 端は浮動小数の ulp だけ超えることがある (canvas 側は丸めない)
    assert.ok(t47.x + t47.w <= lay.width + 1e-6 && t47.y + t47.h <= lay.height + 1e-6,
      'last tile inside: ' + (t47.x + t47.w) + 'x' + (t47.y + t47.h) + ' vs ' + lay.width + 'x' + lay.height);

    // タイルの中身はタイルからはみ出さない
    for (const t of lay.tiles) {
      assert.ok(t.heat.x >= t.x && t.heat.x + t.heat.w <= t.x + t.w, 'heat inside tile');
      assert.ok(t.heat.y >= t.y && t.heat.y + t.heat.h <= t.y + t.h, 'heat inside tile');
      for (const b of t.bars) {
        assert.ok(b.x + b.w <= t.x + t.w + 1e-9, 'bars inside tile');
        assert.ok(b.y + b.h <= t.head.y + t.head.h + t.heat.h + 1e-9, 'bars inside heat');
      }
      for (const s of t.streams) {
        assert.ok(s.x + s.w <= t.x + t.w + 1e-9, 'stream inside tile');
        assert.ok(s.y + s.h <= t.y + t.h + 1e-9, 'stream inside tile');
      }
      assert.equal(t.streams.length, DEFAULT_SHAPE.n_hc);
      assert.equal(t.bars.length, 4);
      assert.ok(t.heat.cols * t.heat.rows >= t.heat.n);
    }
  }
});

test('wallLayout: 深さの帯は 4 本 x 48 セル、壁の上にある', () => {
  const d = FULL.depth;
  assert.equal(d.rows.length, 4);
  assert.deepEqual(d.rows.map((r) => r.key), ['mixer', 'shared', 'ple', 'act']);
  assert.deepEqual(d.rows.map((r) => r.label), ['mix', 'shr', 'ple', 'act']);
  assert.ok(Math.abs(d.cellW * 48 - d.w) < 1e-9, '48 cells fill the strip');
  assert.ok(d.y + d.h <= FULL.wall.y, 'depth above the wall');
  assert.ok(FULL.wall.y + FULL.wall.h <= FULL.height + 1e-9);
});

test('wallLayout: 狭い幅はセル 2px を床にして列数を落とし、必要ならスクロール', () => {
  const narrow = wallLayout(DEFAULT_SHAPE, { w: 420, h: 500 });
  assert.ok(narrow.cellW >= WALL_GEOM.minCell, 'cellW ' + narrow.cellW);
  assert.ok(narrow.cellH >= WALL_GEOM.minCell, 'cellH ' + narrow.cellH);
  assert.ok(narrow.cols < 8, 'must use fewer columns: ' + narrow.cols);
  assert.equal(narrow.tiles.length, 48);
  // 収まらないときは canvas が pane より大きい (= mapwrap がスクロールする)
  assert.equal(narrow.fits, false);
  assert.ok(narrow.width > 420 || narrow.height > 500,
    'overflow expected: ' + narrow.width + 'x' + narrow.height);
});

// ---- 2. ヒットテスト ------------------------------------------------------

test('wallHitTest: 全 48 タイルの四隅が正しい (層, expert) を指す', () => {
  for (const t of FULL.tiles) {
    const l = t.layer;
    const a = wallExpertRect(FULL, l, 0);
    const b = wallExpertRect(FULL, l, 31);
    const c = wallExpertRect(FULL, l, 480);
    const d = wallExpertRect(FULL, l, 511);
    assert.deepEqual(wallHitTest(FULL, a.x + 0.01, a.y + 0.01), { type: 'expert', layer: l, expert: 0 });
    assert.deepEqual(wallHitTest(FULL, b.x + b.w - 0.01, b.y + 0.01), { type: 'expert', layer: l, expert: 31 });
    assert.deepEqual(wallHitTest(FULL, c.x + 0.01, c.y + c.h - 0.01), { type: 'expert', layer: l, expert: 480 });
    assert.deepEqual(wallHitTest(FULL, d.x + d.w - 0.01, d.y + d.h - 0.01), { type: 'expert', layer: l, expert: 511 });
    // 中央付近も格子の割り算どおり
    const e = wallExpertRect(FULL, l, 100);
    assert.deepEqual(wallHitTest(FULL, e.x + e.w / 2, e.y + e.h / 2), { type: 'expert', layer: l, expert: 100 });
    // バッジは mixer ブロック (種別に関わらず介入 API の名前は attn)
    assert.deepEqual(wallHitTest(FULL, t.badge.x + 1, t.badge.y + 1),
      { type: 'block', layer: l, block: 'attn' });
  }
});

test('wallHitTest: バー / stream / 深さの帯 / 隙間', () => {
  const t = FULL.tiles[13];
  for (const b of t.bars) {
    assert.deepEqual(wallHitTest(FULL, b.x + 1, b.y + 1),
      { type: 'block', layer: 13, block: b.block }, b.key);
  }
  for (let i = 0; i < t.streams.length; i++) {
    const r = t.streams[i];
    assert.deepEqual(wallHitTest(FULL, r.x + 1, r.y + 1), { type: 'stream', layer: 13, stream: i });
  }
  // タイル同士の隙間と外側は null
  assert.equal(wallHitTest(FULL, t.x + t.w + 0.5, t.y + 20), null);
  assert.equal(wallHitTest(FULL, -1, 10), null);
  assert.equal(wallHitTest(FULL, 10, FULL.height + 5), null);
  // 深さの帯: 各セルが層、行が量
  for (const row of FULL.depth.rows) {
    for (const l of [0, 1, 23, 47]) {
      const x = FULL.depth.x + (l + 0.5) * FULL.depth.cellW;
      const y = FULL.depth.y + row.y + row.h / 2;
      assert.deepEqual(wallHitTest(FULL, x, y), { type: 'depth', layer: l, row: row.key });
    }
  }
});

test('wallItemsInRect: 1 タイル = 512 expert、2 タイルで 1024、バーも拾う', () => {
  const t = FULL.tiles[5];
  const one = wallItemsInRect(FULL, t.heat.x, t.heat.y, t.heat.x + t.heat.w, t.heat.y + t.heat.h);
  assert.equal(one.filter((x) => x.type === 'expert').length, 512);
  assert.ok(one.filter((x) => x.type === 'expert').every((x) => x.layer === 5));

  const t2 = FULL.tiles[6];
  const two = wallItemsInRect(FULL, t.heat.x, t.heat.y, t2.heat.x + t2.heat.w, t.heat.y + t.heat.h);
  const experts = two.filter((x) => x.type === 'expert');
  assert.equal(experts.length, 1024);
  assert.deepEqual([...new Set(experts.map((x) => x.layer))].sort((a, b) => a - b), [5, 6]);

  // バー 1 本だけを覆う矩形 (重複は出さない)
  const b = t.bars[1];
  const bars = wallItemsInRect(FULL, b.x, b.y, b.x + b.w, b.y + b.h);
  assert.deepEqual(bars, [{ type: 'block', layer: 5, block: 'moe' }]);

  // 空の矩形
  assert.deepEqual(wallItemsInRect(FULL, -20, -20, -10, -10), []);
});

// ---- 3. 色 ---------------------------------------------------------------

test('色: 活動は暗い → 明るい 1 本、赤は off、琥珀は skipped、白は選択', () => {
  let prev = -1;
  for (let i = 0; i <= 10; i++) {
    const c = rampRGB(i / 10);
    const lum = 0.30 * c[0] + 0.59 * c[1] + 0.11 * c[2];
    assert.ok(lum > prev, 'activity must get brighter: ' + i / 10);
    assert.ok(c[0] <= c[1], 'activity must not be red: ' + c.join(','));
    prev = lum;
  }
  assert.equal(heat(1), 'rgb(' + rampRGB(1).join(',') + ')');
  // off は赤、skipped は琥珀、選択は白 (固定色)
  const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const off = rgb(WALL_COLORS.off);
  assert.ok(off[0] > 200 && off[1] < 80 && off[2] < 80, 'off is red: ' + off.join(','));
  const skip = rgb(WALL_COLORS.skip);
  assert.ok(skip[0] > 200 && skip[1] > 150 && skip[2] < 120, 'skip is amber: ' + skip.join(','));
  assert.equal(WALL_COLORS.sel.toLowerCase(), '#ffffff');
});

// ---- 4. 選択 → リクエスト本文 ---------------------------------------------

test('選択: 壁のクリック → 今までと同じ POST /v1/intervene の本文', () => {
  const sel = createSelection();
  const hits = [
    wallHitTest(FULL, wallExpertRect(FULL, 3, 0).x + 0.5, wallExpertRect(FULL, 3, 0).y + 0.5),
    wallHitTest(FULL, wallExpertRect(FULL, 3, 511).x + 0.5, wallExpertRect(FULL, 3, 511).y + 0.5),
    wallHitTest(FULL, wallExpertRect(FULL, 10, 100).x + 0.5, wallExpertRect(FULL, 10, 100).y + 0.5),
    wallHitTest(FULL, FULL.tiles[5].badge.x + 1, FULL.tiles[5].badge.y + 1),
    wallHitTest(FULL, FULL.tiles[7].bars[1].x + 1, FULL.tiles[7].bars[1].y + 1),
    wallHitTest(FULL, FULL.tiles[9].streams[2].x + 1, FULL.tiles[9].streams[2].y + 1),
  ];
  for (const h of hits) addItem(sel, h);
  assert.equal(selectionCount(sel), 6);
  const body = selectionToBody(sel);
  assert.deepEqual(body, {
    experts_off: [[3, 0], [3, 511], [10, 100]],
    blocks_off: [[5, 'attn'], [7, 'moe']],
    streams_off: [[9, 2]],
  });
  // 実 API の検証 (モック店 = 同じ規則) を通る
  const m = createMockStore(DEFAULT_SHAPE_RAW);
  const r = m.handle('POST', '/v1/intervene', body);
  assert.equal(r.status, 200);
  assert.deepEqual(r.body.experts_off, body.experts_off);
  assert.deepEqual(r.body.blocks_off, body.blocks_off);
  assert.deepEqual(r.body.streams_off, body.streams_off);
  // 選択は本文 → 選択 → 本文で往復する
  assert.deepEqual(selectionToBody(sel), body);
});

// ---- 5. 深さの帯の値 ------------------------------------------------------

test('depthValue: mixer は層種別で attn / gdn、act は非ゼロの平均', () => {
  const st = createMapState(DEFAULT_SHAPE);
  st.onFrame({ kind: 'RoleAct', layer: 3, roles: { attn: 0.8, gdn: 0, router: 0.4, shex: 0.6, ple: 0.2 } });
  st.onFrame({ kind: 'RoleAct', layer: 2, roles: { attn: 0.7, gdn: 0.1, router: 0.4, shex: 0.3, ple: 0 } });
  assert.equal(depthValue(st, 3, 'mixer'), 0.8, 'layer 3 is full attention');
  assert.equal(depthValue(st, 2, 'mixer'), 0.1, 'layer 2 is gdn');
  assert.equal(depthValue(st, 3, 'shared'), 0.6);
  assert.equal(depthValue(st, 3, 'ple'), 0.2);
  assert.equal(depthValue(st, 2, 'ple'), 0);
  assert.equal(depthValue(st, 99, 'mixer'), 0);
  assert.equal(depthValue(null, 0, 'act'), 0);
  assert.ok(Math.abs(actLevel(new Float32Array([0, 1, 0, 0.5])) - 0.75) < 1e-9);
  assert.equal(actLevel(new Float32Array(4)), 0);
});

// ---- 6. 状態 (top-k / 累積) とモックの合成テレメトリ ----------------------

test('createMapState: top-k と累積は mode に依らず残り、reset で消える', () => {
  const st = createMapState(DEFAULT_SHAPE);
  st.onFrame({ kind: 'RouterTopK', layer: 0, experts: [3, 9], weights: [0.75, 0.25] });
  assert.equal(st.topk[0][3], 1);
  assert.equal(st.topk[0][9], 1);
  assert.equal(st.topk[0][4], 0);
  assert.equal(st.accum[0][3], 1);
  assert.ok(Math.abs(st.last[0][9] - 1 / 3) < 1e-6);

  st.onFrame({ kind: 'Token' });
  assert.ok(Math.abs(st.accum[0][3] - 0.9) < 1e-6, 'decay applies in token mode too');
  st.onFrame({ kind: 'RouterTopK', layer: 0, experts: [4], weights: [0.5] });
  assert.equal(st.experts[0][3], 0, 'token mode replaces the display');
  assert.ok(Math.abs(st.accum[0][3] - 0.9) < 1e-6, 'accum keeps the older count');
  assert.equal(st.accum[0][4], 1);
  assert.equal(st.topk[0][3], 0, 'top-k is the last token only');
  assert.equal(st.topk[0][4], 1);

  // accum モードは累積そのものを表示する (コピーを持たない)
  st.setMode('accum');
  assert.equal(st.experts[0], st.accum[0]);
  assert.ok(st.experts[0][3] > 0);
  st.setMode('token');
  assert.notEqual(st.experts[0], st.accum[0]);
  assert.equal(st.experts[0][3], 0);

  st.reset();
  assert.equal(st.accum[0][3], 0);
  assert.equal(st.topk[0][4], 0);
  assert.equal(st.last[0][4], 0);
});

test('mockTokenFrames: 決定的で、壁の状態を埋める (skip も作れる)', () => {
  const frames = mockTokenFrames(DEFAULT_SHAPE, 1);
  assert.equal(frames[0].kind, 'Token');
  const kinds = new Set(frames.map((f) => f.kind));
  for (const k of ['Token', 'RoleAct', 'RouterTopK', 'LayerAct', 'ExpertResidency']) {
    assert.ok(kinds.has(k), 'missing ' + k);
  }
  const rt = frames.filter((f) => f.kind === 'RouterTopK');
  assert.equal(rt.length, 48);
  for (const f of rt) {
    assert.ok(f.experts.length >= 1 && f.experts.length <= DEFAULT_SHAPE.n_expert_used);
    assert.equal(f.experts.length, f.weights.length);
    assert.equal(new Set(f.experts).size, f.experts.length);
  }
  // 同じ引数なら同じ列 (スクリーンショットが安定する)
  assert.deepEqual(mockTokenFrames(DEFAULT_SHAPE, 7), mockTokenFrames(DEFAULT_SHAPE, 7));
  assert.notDeepEqual(mockTokenFrames(DEFAULT_SHAPE, 7), mockTokenFrames(DEFAULT_SHAPE, 8));

  // 状態に食わせると top-k と累積が立つ
  const st = createMapState(DEFAULT_SHAPE);
  for (const f of frames) st.onFrame(f);
  const f0 = rt.find((f) => f.layer === 0);
  for (const e of f0.experts) {
    assert.equal(st.topk[0][e], 1);
    assert.ok(st.accum[0][e] > 0);
  }
  assert.equal(st.residency[0].reduce((a, b) => a + b, 0) > 0, true);

  // 介入で切った expert は RouterTopK から外れ、ExpertSkip で来る
  const offE = f0.experts[0];
  const withOff = mockTokenFrames(DEFAULT_SHAPE, 1, { off: [[0, offE]] });
  const skip = withOff.find((f) => f.kind === 'ExpertSkip' && f.layer === 0);
  assert.deepEqual(skip.experts, [offE]);
  const rt0 = withOff.find((f) => f.kind === 'RouterTopK' && f.layer === 0);
  assert.equal(rt0.experts.includes(offE), false);
  const st2 = createMapState(DEFAULT_SHAPE);
  for (const f of withOff) st2.onFrame(f);
  assert.equal(st2.skip[0][offE], 1, 'amber bit is set for the mock skip');
});
