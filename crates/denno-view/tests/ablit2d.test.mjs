// denno-view /2d の Abliterate 作業台 (task 825) の単体テスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
//
// 対象は純ロジックとモック (`view2d.mjs`) と、色モードの描画経路
// (`arch2d.mjs` の drawArch に D.ablit を渡したとき)。DOM は触らない。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  ABLIT_A, ABLIT_B, ABLIT_ALPHA_MAX, ABLIT_CAPTURE_MODE, ABLIT_EVAL_MAX, CAPTURE_MODES,
  DEFAULT_SHAPE_RAW, RECIPE_VERSION,
  ablitApplyBody, ablitChart, ablitCompareRows, ablitEvalPrompts, ablitEvalProgress,
  ablitEvalRows, ablitEvalTable, ablitExportName, ablitExportUrl, ablitHybridValue,
  ablitImportBody, ablitLayerAt, ablitRange, ablitRangeFromDrag, ablitRecipeBodyFromUi,
  ablitRecipeFileName, ablitRecipeFromText, ablitRecipeObject, ablitRecipeSummaryText,
  ablitRecommendedRange, ablitRecordBody, ablitRefusalLike, ablitSepScores, ablitSnippet,
  ablitSummary, createMockStore, parsePromptList, steerApplyBody,
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

// ---- task 833: バッチ評価 ------------------------------------------------

test('ablitEvalPrompts / ablitEvalProgress: 1 行 1 プロンプトと進捗', () => {
  assert.deepEqual(ablitEvalPrompts('a\n\n b \nc'), ['a', 'b', 'c']);
  assert.deepEqual(ablitEvalPrompts(''), []);
  assert.equal(ABLIT_EVAL_MAX, 10);
  assert.equal(ablitEvalPrompts('a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk').length, 10, '10 本で切る');
  assert.equal(ablitEvalProgress(2, 5), 'eval 2/5…');
  assert.equal(ablitEvalProgress(2, 5, 'on'), 'eval 2/5 (on)…');
  assert.equal(ablitEvalProgress(9, 5), 'eval 5/5…', 'はみ出しは丸める');
  assert.equal(ablitEvalProgress(-1, 5), 'eval 0/5…');
});

test('ablitRefusalLike / ablitSnippet: 拒否っぽい言い回しと 1 行の抜粋', () => {
  assert.equal(ablitRefusalLike("I can't help with that."), true);
  assert.equal(ablitRefusalLike('I cannot assist with this request.'), true);
  assert.equal(ablitRefusalLike("As an AI, I don't have..."), true);
  assert.equal(ablitRefusalLike('I must decline.'), true);
  assert.equal(ablitRefusalLike('Here is how a lock works.'), false);
  assert.equal(ablitRefusalLike(''), false);
  assert.equal(ablitRefusalLike(null), false);
  // canned の前置きが付いても拾う (大文字小文字を無視)。
  assert.equal(ablitRefusalLike('Sorry. I WILL NOT do that.'), true);
  assert.equal(ablitSnippet('a\n\nb   c'), 'a b c');
  assert.equal(ablitSnippet('x'.repeat(20), 8), 'x'.repeat(8) + '…');
  assert.equal(ablitSnippet(null), '');
});

test('ablitEvalRows: OUT トークン数・差分・変化・拒否の印', () => {
  const rows = ablitEvalRows([
    { prompt: 'p1', off: "I can't do that.", offTokens: 5, on: 'Sure, here is how.', onTokens: 7 },
    { prompt: 'p2', off: 'same text', offTokens: 2, on: 'same  text', onTokens: 2 },
    { prompt: 'p3', off: 'abc', on: 'ab' },
    null,
  ]);
  assert.equal(rows.length, 4);
  assert.equal(rows[0].offLen, 5);
  assert.equal(rows[0].onLen, 7);
  assert.equal(rows[0].delta, 2);
  assert.equal(rows[0].deltaText, '+2');
  assert.equal(rows[0].changed, true);
  assert.equal(rows[0].offRefusal, true);
  assert.equal(rows[0].onRefusal, false);
  assert.equal(rows[1].delta, 0);
  assert.equal(rows[1].deltaText, '0');
  assert.equal(rows[1].changed, false, '空白の差だけは変化とみなさない');
  // トークン数が無ければ文字数。
  assert.equal(rows[2].offLen, 3);
  assert.equal(rows[2].onLen, 2);
  assert.equal(rows[2].deltaText, '-1');
  // 壊れた要素でも落ちない。
  assert.equal(rows[3].prompt, '');
  assert.equal(rows[3].deltaText, '0');
  assert.deepEqual(ablitEvalRows(null), []);
  // 抜粋は 1 行で上限つき。
  assert.equal(ablitEvalRows([{ prompt: 'p', off: 'a\nb', on: 'c' }], { max: 8 })[0].offSnippet, 'a b');
});

test('ablitEvalTable: prompt | OFF | ON | len Δ の表', () => {
  const rows = ablitEvalRows([
    { prompt: 'p1', off: "I can't.", offTokens: 2, on: 'ok', onTokens: 1 },
  ]);
  const t = ablitEvalTable(rows);
  const lines = t.split('\n');
  assert.equal(lines.length, 3, t);
  assert.ok(lines[0].indexOf('prompt') === 0, t);
  assert.ok(lines[0].includes('OFF') && lines[0].includes('ON') && lines[0].includes('len Δ'), t);
  assert.ok(lines[2].includes('p1'), t);
  assert.ok(lines[2].includes("[refuse]"), t);
  assert.ok(lines[2].includes('-1'), t);
  assert.equal(ablitEvalTable([]).split('\n').length, 2);
  assert.equal(ablitEvalTable(null).split('\n').length, 2);
  // 列は全部揃う (見出しと行)。
  for (const l of [lines[0], lines[2]]) assert.equal(l.split(' | ').length, 4, l);
});

// ---- task 833: レシピ ----------------------------------------------------

test('ablitRecipeBodyFromUi: 画面 → 保存用の本文', () => {
  const b = ablitRecipeBodyFromUi([3, 6], 0.5, { name: 'no_refusal' });
  assert.deepEqual(b, {
    name: 'no_refusal',
    embed_snapshots: false,
    terms: [{ snapshot: ABLIT_A, coef: 1 }, { snapshot: ABLIT_B, coef: -1 }],
    layers: [3, 6],
    alpha: 0.5,
    mode: 'project',
  });
  assert.equal(ablitRecipeBodyFromUi([0, 2], 1, { embed: true }).embed_snapshots, true);
  assert.equal(ablitRecipeBodyFromUi([0, 2], 1, { description: 'x' }).description, 'x');
  // from == to は空範囲 (engine の build_apply と同じで許す)。
  assert.deepEqual(ablitRecipeBodyFromUi([3, 3], 1).layers, [3, 3]);
  assert.throws(() => ablitRecipeBodyFromUi([4, 3], 1), /from must be/);
  assert.throws(() => ablitRecipeBodyFromUi([0, 2], NaN), /alpha/);
  assert.throws(() => ablitRecipeBodyFromUi([0, 2], 1, { name: 'a b' }), /name/);
  assert.throws(() => ablitRecipeBodyFromUi([0, 2], 1, { name: '' }), /name/);
});

test('ablitRecipeObject / ablitRecipeFromText: 実 API と同じ検証', () => {
  const ok = {
    version: RECIPE_VERSION, name: 'rc', mode: 'project', alpha: 0.5,
    layer_from: 1, layer_to: 3,
    terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -1 }],
  };
  const norm = ablitRecipeObject({ ...ok, description: 'd', model_id: 'm' });
  assert.equal(norm.version, 1);
  assert.equal(norm.description, 'd');
  assert.equal(norm.model_id, 'm');
  assert.equal(norm.terms.length, 2);
  // 既定 alpha。
  assert.equal(ablitRecipeObject({ ...ok, alpha: undefined }).alpha, 1);
  // 埋め込み。
  const emb = ablitRecipeObject({ ...ok, snapshots: { A: 'bW9jaw==' } });
  assert.deepEqual(Object.keys(emb.snapshots), ['A']);
  // 不正
  const bad = [
    null, [], 'x', {},
    { ...ok, version: 2 },
    { ...ok, name: 'a b' },
    { ...ok, bogus: 1 },
    { ...ok, mode: 'bogus' },
    { ...ok, alpha: 'x' },
    { ...ok, layer_from: -1 },
    { ...ok, layer_from: 4, layer_to: 3 },
    { ...ok, terms: [] },
    { ...ok, terms: 'x' },
    { ...ok, terms: [{ snapshot: 'A' }] },
    { ...ok, terms: [{ snapshot: 'a b', coef: 1 }] },
    { ...ok, terms: [{ snapshot: 'A', coef: 1, extra: 1 }] },
    { ...ok, terms: [{ snapshot: 'A', coef: 'x' }] },
    { ...ok, snapshots: [] },
    { ...ok, snapshots: { 'a b': 'x' } },
    { ...ok, snapshots: { A: 1 } },
    { ...ok, snapshots: { A: '' } },
    { ...ok, description: 1 },
    { ...ok, model_id: 1 },
  ];
  for (const v of bad) assert.throws(() => ablitRecipeObject(v), Error, JSON.stringify(v));
  // ファイル (テキスト) 経由。
  assert.equal(ablitRecipeFromText(JSON.stringify(ok)).name, 'rc');
  assert.throws(() => ablitRecipeFromText(''), /empty/);
  assert.throws(() => ablitRecipeFromText('   '), /empty/);
  assert.throws(() => ablitRecipeFromText('nope'), /not JSON/);
  assert.throws(() => ablitRecipeFromText('[]'), /JSON object/);
});

test('ablitRecipeSummaryText / ablitRecipeFileName', () => {
  const r = ablitRecipeObject({
    version: 1, name: 'rc', mode: 'project', alpha: 0.5, layer_from: 1, layer_to: 3,
    terms: [{ snapshot: 'A', coef: 1 }], snapshots: { A: 'bW9jaw==' }, model_id: 'm',
  });
  const s = ablitRecipeSummaryText(r);
  assert.ok(s.startsWith('rc · project · α 0.5 · layers 1-3'), s);
  assert.ok(s.includes('embedded 1'), s);
  assert.ok(s.includes('model m'), s);
  assert.equal(ablitRecipeSummaryText(null), '');
  assert.equal(ablitRecipeFileName('rc'), 'rc.recipe.json');
  assert.equal(ablitRecipeFileName(null), 'recipe.recipe.json');
});

test('モック: recipe export / import が実 API の形で答える (task 833)', () => {
  const m = createMockStore({ layers: 8, n_expert: 8, n_expert_used: 2, n_hc: 1, n_embd: 16 },
    { now: () => 1762732800 });
  m.handle('POST', '/v1/steer/record', { name: 'ablit_A' });
  m.tick(2);
  m.handle('POST', '/v1/steer/record/stop');
  m.handle('POST', '/v1/steer/record', { name: 'ablit_B' });
  m.tick(2);
  m.handle('POST', '/v1/steer/record/stop');
  // 適用が無いときは terms が要る。
  assert.equal(m.handle('POST', '/v1/steer/recipe/export', {}).status, 400);
  const body = ablitRecipeBodyFromUi([2, 6], 0.5, { name: 'rc', embed: true });
  const ex = m.handle('POST', '/v1/steer/recipe/export', body);
  assert.equal(ex.status, 200);
  assert.equal(ex.body.version, 1);
  assert.equal(ex.body.name, 'rc');
  assert.equal(ex.body.model_id, 'mock');
  assert.deepEqual(ex.body.snapshots && Object.keys(ex.body.snapshots), ['ablit_A', 'ablit_B']);
  // 検証済みのレシピとして読める (往復)。
  const back = ablitRecipeObject(ex.body);
  assert.equal(back.name, 'rc');
  assert.equal(back.alpha, 0.5);

  // import (検証のみ) は状態を変えない。
  const before = m.handle('GET', '/v1/steer').body.apply;
  const im = m.handle('POST', '/v1/steer/recipe/import', ex.body);
  assert.equal(im.status, 200);
  assert.equal(im.body.valid, true);
  assert.equal(im.body.applied, false);
  assert.deepEqual(m.handle('GET', '/v1/steer').body.apply, before);
  // ?apply=1 で適用。
  const ap = m.handle('POST', '/v1/steer/recipe/import?apply=1', ex.body);
  assert.equal(ap.body.applied, true);
  const cur = m.handle('GET', '/v1/steer').body.apply;
  assert.equal(cur.mode, 'project');
  assert.deepEqual(cur.layers, [2, 6]);
  assert.equal(cur.alpha, 0.5);
  // {recipe, apply} でも同じ。
  const ap2 = m.handle('POST', '/v1/steer/recipe/import', { recipe: ex.body, apply: true });
  assert.equal(ap2.body.applied, true);
  // 適用中の設定からも export できる (terms 無し)。
  const ex2 = m.handle('POST', '/v1/steer/recipe/export', { name: 'from_apply' });
  assert.equal(ex2.status, 200);
  assert.equal(ex2.body.layer_from, 2);
  assert.equal(ex2.body.layer_to, 6);
  assert.equal(ex2.body.mode, 'project');

  // 不正。
  for (const b of [null, [], {}, { bogus: 1 }, { ...ex.body, version: 2 },
    { ...ex.body, terms: [] }, { ...ex.body, layer_to: 99 }]) {
    assert.equal(m.handle('POST', '/v1/steer/recipe/import', b).status, 400, JSON.stringify(b));
  }
  // 参照先が無い (埋め込み無し) は 400。
  assert.equal(m.handle('POST', '/v1/steer/recipe/import', {
    version: 1, name: 'x', mode: 'project', alpha: 1, layer_from: 0, layer_to: 2,
    terms: [{ snapshot: 'ghost', coef: 1 }],
  }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/recipe/export', { name: 'a b' }).status, 400);
  assert.equal(m.handle('POST', '/v1/steer/recipe/export', { embed_snapshots: 'yes' }).status, 400);
});
