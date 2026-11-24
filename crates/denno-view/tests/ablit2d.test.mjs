// denno-view /2d の Abliterate 作業台 (task 825) の単体テスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
//
// 対象は純ロジックとモック (`view2d.mjs`) と、色モードの描画経路
// (`arch2d.mjs` の drawArch に D.ablit を渡したとき)。DOM は触らない。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ABLIT_A, ABLIT_B, ABLIT_ALPHA_MAX, ABLIT_CAPTURE_MODE, CAPTURE_MODES, DEFAULT_SHAPE_RAW,
  ablitApplyBody, ablitChart, ablitCompareRows, ablitExportName, ablitExportUrl,
  ablitHybridValue, ablitImportBody, ablitLayerAt, ablitRange, ablitRangeFromDrag,
  ablitRecommendedRange, ablitRecordBody, ablitSepScores, ablitSummary, createMockStore,
  parsePromptList, steerApplyBody,
} from '../src/view2d.mjs';
import { archLayout, drawArch, fitView } from '../src/arch2d.mjs';

const analysis = {
  pos: 'ablit_A',
  neg: 'ablit_B',
  n_layer: 8,
  peak_layer: 4,
  recommended_range: [2, 6],
  layers: [
    { layer: 0, diff_norm: 1, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.1 },
    { layer: 1, diff_norm: 2, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.25 },
    { layer: 2, diff_norm: 3, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.4 },
    { layer: 3, diff_norm: 4, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.55 },
    { layer: 4, diff_norm: 5, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.8 },
    { layer: 5, diff_norm: 4, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.6 },
    { layer: 6, diff_norm: 2, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.3 },
    { layer: 7, diff_norm: 1, norm_pos: 5, norm_neg: 5, cos_sim: 0.1, sep_score: 0.1 },
  ],
};

test('parsePromptList: 空行を捨て、前後の空白を落とし、上限で切る', () => {
  assert.deepEqual(parsePromptList('  a  \n\n b\r\nc\n'), ['a', 'b', 'c']);
  assert.deepEqual(parsePromptList(''), []);
  assert.deepEqual(parsePromptList(null), []);
  assert.deepEqual(parsePromptList('a\nb\nc\nd', { max: 2 }), ['a', 'b']);
  assert.deepEqual(parsePromptList('   \n \t\n'), []);
});

test('ablitSepScores: 層番号で並べ、欠けは 0、範囲外は捨てる', () => {
  assert.deepEqual(ablitSepScores(analysis, 8).map((x) => Math.round(x * 100)), [10, 25, 40, 55, 80, 60, 30, 10]);
  assert.deepEqual(ablitSepScores(analysis, 3), [0.1, 0.25, 0.4], '層数が少ないときは切る');
  assert.deepEqual(ablitSepScores(analysis, 10).slice(8), [0, 0], '足りない層は 0');
  assert.deepEqual(ablitSepScores(null, 2), [0, 0]);
  // 壊れた値は 0 に落とす (NaN を描かない)。
  const bad = { layers: [{ layer: 0, sep_score: 'x' }, { layer: 1, sep_score: 9 }, { layer: 'z', sep_score: 1 }] };
  assert.deepEqual(ablitSepScores(bad, 2), [0, 1]);
});

test('ablitRecommendedRange: 層 0 と最終層を避け、半値以上の連続窓', () => {
  assert.deepEqual(ablitRecommendedRange([]), [0, 0]);
  assert.deepEqual(ablitRecommendedRange([0.3]), [0, 1]);
  assert.deepEqual(ablitRecommendedRange([0.3, 0.4]), [0, 2]);
  assert.deepEqual(ablitRecommendedRange([0.9, 0.2, 0.5, 0.45, 0.1]), [2, 4]);
  assert.deepEqual(ablitRecommendedRange([0.1, 0.3, 0.6, 0.99]), [1, 3]);
  assert.deepEqual(ablitRecommendedRange([0, 0, 0, 0, 0]), [1, 4]);
  assert.deepEqual(ablitRecommendedRange([0, 0.4, 0.8, 0.5, 0.2, 0, 0]), [1, 4]);
  // engine (Rust) の `recommended_range` と同じ数字になること。
  const seps = ablitSepScores(analysis, 8);
  assert.deepEqual(ablitRecommendedRange(seps), [2, 6]);
});

test('ablitRange: 層数で丸め、壊れた値は [0, n]', () => {
  assert.deepEqual(ablitRange(analysis, 8), [2, 6]);
  assert.deepEqual(ablitRange(analysis, 5), [2, 5], '層数で切る');
  assert.deepEqual(ablitRange({ recommended_range: [6, 3] }, 8), [6, 6], 'from > to は詰める');
  assert.deepEqual(ablitRange({}, 4), [0, 4]);
  assert.deepEqual(ablitRange(null, 4), [0, 4]);
});

test('ablitChart: バーの位置と高さ・選択・ピーク', () => {
  const rect = { x: 10, y: 20, w: 80, h: 40 };
  const ch = ablitChart(analysis, rect, [3, 6], 8);
  assert.equal(ch.bars.length, 8);
  assert.equal(ch.n, 8);
  assert.deepEqual(ch.range, [3, 6]);
  assert.equal(ch.max, 0.8);
  // ピークのバーは高さいっぱい、値 0 は 0 (ただし値 > 0 は最低 1px)。
  const peak = ch.bars[4];
  assert.ok(Math.abs(peak.h - 40) < 1e-9, 'peak h ' + peak.h);
  assert.equal(peak.peak, true);
  assert.equal(ch.bars[3].selected, true);
  assert.equal(ch.bars[5].selected, true);
  assert.equal(ch.bars[6].selected, false);
  assert.ok(ch.bars.every((b) => b.x >= rect.x - 1e-9 && b.x + b.w <= rect.x + rect.w + 1e-6));
  // 全部 0 ならバーは 0 (最低 1px も付けない)。
  const flat = ablitChart({ layers: [] }, rect, [0, 0], 8);
  assert.ok(flat.bars.every((b) => b.h === 0));
  assert.equal(flat.max, 0);
});

test('ablitLayerAt / ablitRangeFromDrag: クリックとドラッグが層に落ちる', () => {
  const ch = ablitChart(analysis, { x: 0, y: 0, w: 80, h: 40 }, [3, 6], 8);
  assert.equal(ablitLayerAt(ch, 1), 0);
  assert.equal(ablitLayerAt(ch, 77), 7);
  assert.equal(ablitLayerAt(ch, -1), null);
  assert.equal(ablitLayerAt(ch, 81), null);
  assert.equal(ablitLayerAt(null, 5), null);
  assert.deepEqual(ablitRangeFromDrag(3, 6), [3, 7]);
  assert.deepEqual(ablitRangeFromDrag(6, 3), [3, 7]);
  assert.deepEqual(ablitRangeFromDrag(4, 4), [4, 5], '1 層は [4, 5)');
});

test('ablitApplyBody: A +1 / B -1 / project / [from,to) / alpha の検証', () => {
  const body = ablitApplyBody([3, 6], 1.0, { nLayers: 48 });
  assert.deepEqual(body.terms, [
    { snapshot: ABLIT_A, coef: 1 },
    { snapshot: ABLIT_B, coef: -1 },
  ]);
  assert.deepEqual(body.layers, [3, 6]);
  assert.equal(body.mode, 'project');
  assert.equal(body.alpha, 1);
  // 既定 alpha は 1。
  assert.equal(ablitApplyBody([0, 2], undefined, { nLayers: 48 }).alpha, 1);
  // 範囲の境界 (alpha 0 と 2.0) は通る。
  assert.equal(ablitApplyBody([0, 1], 0, { nLayers: 48 }).alpha, 0);
  assert.equal(ablitApplyBody([0, 1], ABLIT_ALPHA_MAX, { nLayers: 48 }).alpha, 2);
  // 不正
  assert.throws(() => ablitApplyBody([3, 3], 1), /from must be </);
  assert.throws(() => ablitApplyBody([4, 3], 1), /from must be </);
  assert.throws(() => ablitApplyBody([0, 49], 1, { nLayers: 48 }), /to must be <=/);
  assert.throws(() => ablitApplyBody([0, 2], 2.01, { nLayers: 48 }), /alpha/);
  assert.throws(() => ablitApplyBody([0, 2], -0.01, { nLayers: 48 }), /alpha/);
  assert.throws(() => ablitApplyBody([0, 2], NaN), /alpha/);
  assert.throws(() => ablitApplyBody([0], 1), /\[from, to\]/);
  assert.throws(() => ablitApplyBody(null, 1), /\[from, to\]/);
  // 実 API の本文と同じ形になる (steerApplyBody を通す)。
  assert.deepEqual(ablitApplyBody([3, 6], 0.5, { nLayers: 48 }),
    steerApplyBody({
      terms: [{ snapshot: ABLIT_A, coef: 1 }, { snapshot: ABLIT_B, coef: -1 }],
      layers: [3, 6], alpha: 0.5, mode: 'project',
    }, { nLayers: 48 }));
});

test('ablitHybridValue: Hybrid block は 4 層の平均', () => {
  const scores = ablitSepScores(analysis, 8);
  assert.ok(Math.abs(ablitHybridValue(scores, 0) - (0.1 + 0.25 + 0.4 + 0.55) / 4) < 1e-9);
  assert.ok(Math.abs(ablitHybridValue(scores, 1) - (0.8 + 0.6 + 0.3 + 0.1) / 4) < 1e-9);
  assert.equal(ablitHybridValue(scores, -1), 0);
  assert.equal(ablitHybridValue(scores, 9), 0, '範囲外は 0');
  assert.equal(ablitHybridValue(null, 0), 0);
});

test('ablitExportUrl / ablitExportName: クエリとファイル名', () => {
  assert.equal(ablitExportUrl('ablit_A'), '/api/v1/steer/export?name=ablit_A');
  assert.equal(ablitExportUrl('ablit_A', 'json'), '/api/v1/steer/export?name=ablit_A&format=json');
  assert.equal(ablitExportUrl('a b'), '/api/v1/steer/export?name=a%20b');
  assert.equal(ablitExportName('ablit_A'), 'ablit_A.dns');
  assert.equal(ablitExportName('ablit_A', 'json'), 'ablit_A.dnsteer.json');
  assert.equal(ablitExportName(null), 'snapshot.dns');
});

test('ablitImportBody: JSON / base64 / mean 配列の検証', () => {
  const ok = ablitImportBody(JSON.stringify({
    name: 'ablit_A', layers: 2, n_hc: 1, n_embd: 3, mean: [0, 1, 2, 3, 4, 5],
  }));
  assert.equal(ok.name, 'ablit_A');
  const b64 = ablitImportBody(JSON.stringify({ name: 'x', data_base64: 'bW9jaw==' }));
  assert.equal(b64.name, 'x');
  assert.throws(() => ablitImportBody(''), /empty/);
  assert.throws(() => ablitImportBody('  '), /empty/);
  assert.throws(() => ablitImportBody('not json'), /not JSON/);
  assert.throws(() => ablitImportBody('[]'), /must be an object/);
  assert.throws(() => ablitImportBody('null'), /must be an object/);
  assert.throws(() => ablitImportBody('{}'), /mean or data_base64/);
  assert.throws(() => ablitImportBody(JSON.stringify({ name: 'x', data_base64: 1 })), /must be a string/);
  assert.throws(() => ablitImportBody(JSON.stringify({
    name: 'x', layers: 2, n_hc: 1, n_embd: 3, mean: [0, 1],
  })), /length mismatch/);
  assert.throws(() => ablitImportBody(JSON.stringify({
    name: 'x', layers: 0, n_hc: 0, n_embd: 0, mean: [],
  })), /layers \/ n_hc \/ n_embd/);
  assert.throws(() => ablitImportBody(JSON.stringify({
    name: 'a.b', layers: 1, n_hc: 1, n_embd: 1, mean: [1],
  })), /name:/);
});

test('ablitSummary / ablitCompareRows: 1 行要約と OFF/ON の行', () => {
  assert.equal(ablitSummary(analysis), 'peak layer 4 · sep 0.800 · range 2-6');
  assert.equal(ablitSummary(null), '');
  assert.equal(ablitSummary({ layers: [] }), '');
  const rows = ablitCompareRows('before', 'after');
  assert.deepEqual(rows.map((r) => r.label), ['OFF', 'ON']);
  assert.equal(rows[0].text, 'before');
  assert.equal(rows[1].text, 'after');
  const long = ablitCompareRows('x'.repeat(50), null, { max: 16 });
  assert.equal(long[0].text, 'x'.repeat(16) + '…');
  assert.equal(long[1].text, '');
});

test('モック: analyze / export / import が実 API の検証規則で答える', () => {
  const m = createMockStore({ layers: 8, n_expert: 8, n_expert_used: 2, n_hc: 1, n_embd: 16 }, { now: () => 1762732800 });
  // スナップショットを作る (記録 → tick → stop)。
  m.handle('POST', '/v1/steer/record', { name: 'ablit_A' });
  m.tick(2);
  m.handle('POST', '/v1/steer/record/stop');
  m.handle('POST', '/v1/steer/record', { name: 'ablit_B' });
  m.tick(2);
  m.handle('POST', '/v1/steer/record/stop');

  const an = m.handle('POST', '/v1/steer/analyze', { pos: 'ablit_A', neg: 'ablit_B' });
  assert.equal(an.status, 200);
  assert.equal(an.body.layers.length, 8);
  assert.equal(an.body.recommended_range.length, 2);
  for (const l of an.body.layers) {
    assert.ok(l.sep_score >= 0 && l.sep_score <= 1, 'sep ' + l.sep_score);
    assert.ok(Number.isFinite(l.cos_sim));
  }
  assert.deepEqual(m.handle('POST', '/v1/steer/analyze', { terms: [
    { snapshot: 'ablit_A', coef: 1 }, { snapshot: 'ablit_B', coef: -1 },
  ] }).body.layers, an.body.layers);

  // 不正
  assert.equal(m.handle('POST', '/v1/steer/analyze', null).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/analyze', { pos: 'ablit_A' }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/analyze', { pos: 'nope', neg: 'ablit_B' }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/analyze', { terms: [] }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/analyze', { terms: [{ snapshot: 'ablit_A', coef: 1 }] }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/analyze', { pos: 'ablit_A', neg: 'ablit_B', x: 1 }).status, 400);

  // export (query 付き) / import
  const ex = m.handle('GET', '/v1/steer/export?name=ablit_A&format=json');
  assert.equal(ex.status, 200);
  assert.equal(ex.body.name, 'ablit_A');
  assert.equal(ex.body.layers, 8);
  assert.equal(m.handle('GET', '/v1/steer/export').status, 400);
  assert.equal(m.handle('GET', '/v1/steer/export?name=nope').status, 400);

  const im = m.handle('POST', '/v1/steer/import', { name: 'ablit_imp', layers: 8 });
  assert.equal(im.status, 200);
  assert.equal(im.body.imported.name, 'ablit_imp');
  assert.deepEqual(im.body.snapshots.map((s) => s.name), ['ablit_A', 'ablit_B', 'ablit_imp']);
  assert.equal(m.handle('POST', '/v1/steer/import', { name: 'a b' }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/import', { name: 'x', layers: 3 }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/import', null).status, 400);
  // 名前は query で上書きできる (実 API と同じ)。
  const im2 = m.handle('POST', '/v1/steer/import?name=ablit_q', { name: 'ignored', layers: 8 });
  assert.equal(im2.body.imported.name, 'ablit_q');
});

test('ablitRecordBody / CAPTURE_MODES: capture mode の本文と検証 (task 829)', () => {
  assert.deepEqual(CAPTURE_MODES, ['all', 'last_prompt_token']);
  assert.equal(ABLIT_CAPTURE_MODE, 'last_prompt_token');
  // 省略時は画面の既定 (last_prompt_token)。
  assert.deepEqual(ablitRecordBody('ablit_A'), { name: 'ablit_A', capture_mode: 'last_prompt_token' });
  assert.deepEqual(ablitRecordBody('ablit_B', 'all'), { name: 'ablit_B', capture_mode: 'all' });
  assert.deepEqual(ablitRecordBody('ablit_B', 'last_prompt_token'),
    { name: 'ablit_B', capture_mode: 'last_prompt_token' });
  assert.throws(() => ablitRecordBody('', 'all'), /name/);
  assert.throws(() => ablitRecordBody('a b', 'all'), /name/);
  assert.throws(() => ablitRecordBody('a'.repeat(65), 'all'), /name/);
  assert.throws(() => ablitRecordBody('ablit_A', 'bogus'), /capture_mode/);
  assert.throws(() => ablitRecordBody('ablit_A', 'ALL'), /capture_mode/);
});

test('モック: record が capture_mode を受け、status が返し、未知の値は 400 (task 829)', () => {
  const m = createMockStore({ layers: 8, n_expert: 8, n_expert_used: 2, n_hc: 1, n_embd: 16 },
    { now: () => 1762732800 });
  // 記録前は "all" (実 API の既定と同じ)。
  assert.equal(m.handle('GET', '/v1/steer').body.capture_mode, 'all');

  const r = m.handle('POST', '/v1/steer/record', ablitRecordBody('ablit_A'));
  assert.equal(r.status, 200);
  assert.equal(r.body.capture_mode, 'last_prompt_token');
  assert.equal(r.body.recording.capture_mode, 'last_prompt_token');
  assert.equal(m.handle('GET', '/v1/steer').body.capture_mode, 'last_prompt_token');
  m.tick(1);
  const stp = m.handle('POST', '/v1/steer/record/stop');
  assert.equal(stp.status, 200);
  // stop 後も直近の値が残る (実 API と同じ)。
  assert.equal(m.handle('GET', '/v1/steer').body.capture_mode, 'last_prompt_token');

  // "all" も通り、省略時は all。
  assert.equal(m.handle('POST', '/v1/steer/record', { name: 'ablit_B' }).status, 200);
  assert.equal(m.handle('GET', '/v1/steer').body.capture_mode, 'all');
  m.tick(1);
  m.handle('POST', '/v1/steer/record/stop');
  assert.equal(m.handle('POST', '/v1/steer/record',
    { name: 'ablit_B', capture_mode: 'all' }).status, 200);
  m.handle('POST', '/v1/steer/record/stop');

  // 不正は 400 (黙って all にしない)。
  assert.equal(m.handle('POST', '/v1/steer/record',
    { name: 'ablit_A', capture_mode: 'bogus' }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/record',
    { name: 'ablit_A', capture_mode: 1 }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/record',
    { name: 'ablit_A', capture_mode: 'last_prompt_token', x: 1 }).status, 400);
});

/// 描画は呼んだ命令を数えるだけの偽 context。
function fakeCtx() {
  const calls = { fillRect: 0, fillText: 0, drawImage: 0 };
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

test('drawArch: ablit のスコアを渡しても描画が壊れない (48 層 + 12 block を塗る)', () => {
  const L = archLayout(DEFAULT_SHAPE_RAW);
  const st = createMockStore(DEFAULT_SHAPE_RAW, {});
  const pane = { w: 900, h: 700 };
  const view = Object.assign({}, fitView(L.bounds, pane), pane);
  const base = {
    st, offExp: new Map(), offBlk: new Set(), offStr: new Map(),
    isSel: () => false, dragKeys: null, hover: null, vramOn: true,
    gridImage: () => ({}), focusHybrid: 0, dpr: 1,
  };
  const g0 = fakeCtx();
  drawArch(g0, L, view, base);
  const flatFills = g0.calls.fillRect;
  // 48 層ぶんのスコア (真ん中が山) を渡す。
  const scores = Array.from({ length: 48 }, (_, l) => Math.exp(-((l - 24) * (l - 24)) / 50));
  const g1 = fakeCtx();
  drawArch(g1, L, view, Object.assign({}, base, { ablit: { scores } }));
  assert.ok(g1.calls.fillRect > flatFills, 'ablit は柱と block を塗るので fillRect が増える');
  // 壊れた入れ物でも落ちない (無い扱い)。
  const g2 = fakeCtx();
  drawArch(g2, L, view, Object.assign({}, base, { ablit: { scores: [] } }));
  const g3 = fakeCtx();
  drawArch(g3, L, view, Object.assign({}, base, { ablit: 'nope' }));
});
