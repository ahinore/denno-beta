// denno-view /2d の層ドリルダウン (task 857) の純ロジック。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの:
//   1. RouterTopK → 選ばれた expert の id / ゲート重み / top-k 内の割合
//   2. ExpertResidency → VRAM hit / CPU miss、来ていなければ null (不明)
//   3. ExpertSkip → skip の印
//   4. 直近トークンの頻度 (窓の上限・古い → 新しい順・reset で消える)
//   5. 壊れた / 最小のテレメトリでも落ちない (note に理由)
//   6. スパークラインの棒の形
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_SHAPE_RAW, EXPERT_FREQ_WINDOW, createMapState,
} from '../src/view2d.mjs';
import { DRILL_COLORS, drillSparkBars, expertDrilldown, expertFrequency } from '../src/arch2d.mjs';

const route = (layer, experts, weights) => ({ kind: 'RouterTopK', layer, experts, weights });
const resi = (layer, vram) => ({ kind: 'ExpertResidency', layer, vram });
const skip = (layer, experts) => ({ kind: 'ExpertSkip', layer, experts });
const close = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol;

test('expertDrilldown: テレメトリが無いときは空 + 理由 (落ちない)', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 0);
  assert.ok(d);
  assert.equal(d.layer, 0);
  assert.equal(d.n_used, 0);
  assert.deepEqual(d.experts, []);
  assert.match(d.note, /RouterTopK/);
  assert.equal(d.residency, false);
  assert.equal(d.window, 0);
  // 範囲外・不正は null
  assert.equal(expertDrilldown(st, DEFAULT_SHAPE_RAW, -1), null);
  assert.equal(expertDrilldown(st, DEFAULT_SHAPE_RAW, DEFAULT_SHAPE_RAW.layers), null);
  assert.equal(expertDrilldown(st, DEFAULT_SHAPE_RAW, 'x'), null);
  assert.equal(expertDrilldown(null, DEFAULT_SHAPE_RAW, 0), null);
});

test('expertDrilldown: RouterTopK の id / 重み / 割合 (重みの降順)', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame(route(5, [300, 7, 42], [0.2, 0.5, 0.3]));
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 5);
  assert.equal(d.n_used, 3);
  assert.deepEqual(d.experts.map((r) => r.id), [7, 42, 300], '重みの降順');
  assert.ok(close(d.experts[0].weight, 0.5, 1e-6));
  assert.ok(close(d.experts[1].weight, 0.3, 1e-6));
  assert.ok(close(d.experts[2].weight, 0.2, 1e-6));
  // 割合の合計は 1
  const sum = d.experts.reduce((a, r) => a + r.share, 0);
  assert.ok(close(sum, 1, 1e-5), 'share sum ' + sum);
  assert.ok(close(d.experts[0].share, 0.5, 1e-5));
  // residency が来ていないので hit は不明
  assert.deepEqual(d.experts.map((r) => r.hit), [null, null, null]);
  assert.match(d.note, /ExpertResidency/);
  assert.equal(d.kind, 'gdn', '層 5 は GDN (4 層ごとに QSA)');
});

test('expertDrilldown: ExpertResidency → VRAM / CPU の内訳と印', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame(route(3, [1, 2, 3], [0.4, 0.35, 0.25]));
  st.onFrame(resi(3, [2, 3]));
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 3);
  assert.equal(d.residency, true);
  assert.equal(d.hitVram, 2);
  assert.equal(d.hitCpu, 1);
  const by = new Map(d.experts.map((r) => [r.id, r.hit]));
  assert.equal(by.get(1), 'cpu');
  assert.equal(by.get(2), 'vram');
  assert.equal(by.get(3), 'vram');
  assert.equal(d.kind, 'attn');
  assert.equal(d.note, '');
});

test('expertDrilldown: ExpertSkip は skip の印になる', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame(route(9, [10, 11], [0.6, 0.4]));
  st.onFrame(skip(9, [11]));
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 9);
  const by = new Map(d.experts.map((r) => [r.id, r.skipped]));
  assert.equal(by.get(10), false);
  assert.equal(by.get(11), true);
});

test('頻度: 直近トークンの 0/1 が古い → 新しい順に並ぶ', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame(route(1, [5, 6], [0.5, 0.5]));
  st.onFrame(route(1, [6, 7], [0.5, 0.5]));
  st.onFrame(route(1, [5, 7], [0.5, 0.5]));
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 1);
  assert.equal(d.window, 3);
  // 直近トークンの top-k は [5, 7] なので、6 は行に出ない (count だけ残る)
  assert.deepEqual(d.experts.map((r) => r.id).sort((a, b) => a - b), [5, 7]);
  const by = new Map(d.experts.map((r) => [r.id, r]));
  assert.deepEqual(by.get(5).values, [1, 0, 1], 'e5: 3 トークン前に当たり、最後に当たり');
  assert.deepEqual(by.get(7).values, [0, 1, 1]);
  assert.equal(by.get(7).freqHits, 2);
  assert.ok(close(by.get(7).freqRate, 2 / 3, 1e-9));
  // 表示していない expert (6) の count も増えている (2 トークンで当たった)
  assert.equal(st.count[1][6], 2);
  assert.equal(by.get(5).count, 2);
});

test('頻度: 窓は EXPERT_FREQ_WINDOW 本で頭打ち、reset で空になる', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  for (let i = 0; i < EXPERT_FREQ_WINDOW + 5; i++) st.onFrame(route(2, [i % 3, 100], [0.5, 0.5]));
  assert.equal(st.hist[2].length, EXPERT_FREQ_WINDOW, '窓の上限');
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 2);
  assert.equal(d.window, EXPERT_FREQ_WINDOW);
  for (const r of d.experts) assert.equal(r.values.length, EXPERT_FREQ_WINDOW);
  st.reset();
  assert.equal(st.hist[2].length, 0);
  assert.equal(expertDrilldown(st, DEFAULT_SHAPE_RAW, 2).window, 0);
});

test('expertFrequency: ids の検証と窓の指定', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  st.onFrame(route(0, [4], [1]));
  st.onFrame(route(0, [4], [1]));
  const f = expertFrequency(st, 0, [4, -1, 'x', 999999]);
  assert.deepEqual(f.rows.map((r) => r.id), [4], '範囲外の id は落とす');
  assert.equal(f.window, 2);
  assert.deepEqual(f.rows[0].values, [1, 1]);
  // 窓を 1 にすると直近だけ
  const f1 = expertFrequency(st, 0, [4], { window: 1 });
  assert.equal(f1.window, 1);
  assert.deepEqual(f1.rows[0].values, [1]);
  assert.deepEqual(expertFrequency(st, 0, []).rows, []);
  assert.deepEqual(expertFrequency(null, 0, [1]).rows, []);
});

test('expertDrilldown: maxRows で並べる数を絞る (note に件数)', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  const experts = [];
  for (let i = 0; i < 12; i++) experts.push(i);
  st.onFrame(route(3, experts, experts.map((_, i) => 1 - i * 0.05)));
  const all = expertDrilldown(st, DEFAULT_SHAPE_RAW, 3);
  assert.equal(all.n_used, 12);
  assert.equal(all.shown, 12);
  const few = expertDrilldown(st, DEFAULT_SHAPE_RAW, 3, { maxRows: 4 });
  assert.equal(few.shown, 4);
  assert.equal(few.experts.length, 4);
  assert.match(few.note, /上位 4 件/);
});

test('expertDrilldown: 壊れた RouterTopK (experts 無し) でも落ちない', () => {
  const st = createMapState(DEFAULT_SHAPE_RAW);
  assert.equal(st.onFrame({ kind: 'RouterTopK', layer: 1 }), true);
  const d = expertDrilldown(st, DEFAULT_SHAPE_RAW, 1);
  assert.equal(d.n_used, 0);
  assert.deepEqual(d.experts, []);
  // 重複 id は最後の重みで上書き、負 id は落とす
  st.onFrame(route(1, [2, -3, 2], [0.4, 0.5, 0.6]));
  const d2 = expertDrilldown(st, DEFAULT_SHAPE_RAW, 1);
  assert.equal(d2.n_used, 1);
  assert.equal(d2.experts[0].id, 2);
  assert.ok(close(d2.experts[0].weight, 0.6, 1e-6));
});

test('drillSparkBars: 本数・範囲・0 の高さ', () => {
  assert.deepEqual(drillSparkBars([], 64, 12), []);
  const bars = drillSparkBars([1, 0, 1, 0.5], 40, 10);
  assert.equal(bars.length, 4);
  for (const b of bars) {
    assert.ok(b.x >= 0 && b.x + b.w <= 40 + 1e-9, 'x が枠内');
    assert.ok(b.y >= -1e-9 && b.y + b.h <= 10 + 1e-9, 'y が枠内');
    assert.ok(b.v >= 0 && b.v <= 1);
  }
  assert.equal(bars[1].v, 0);
  assert.equal(bars[1].h, 0);
  assert.ok(bars[0].h > 0);
  assert.ok(bars[2].h > bars[3].h, '大きい値ほど高い');
  // x は単調増加
  for (let i = 1; i < bars.length; i++) assert.ok(bars[i].x > bars[i - 1].x);
  // NaN は 0 に落ちる
  assert.equal(drillSparkBars([Number.NaN], 10, 10)[0].v, 0);
});

test('DRILL_COLORS: hit / miss / 不明の 3 色', () => {
  for (const k of ['vram', 'cpu', 'unknown']) assert.match(DRILL_COLORS[k], /^#[0-9a-f]{6}$/i);
});
