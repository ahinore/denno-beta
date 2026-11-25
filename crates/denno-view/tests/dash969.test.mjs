// dash.mjs (task 969) の純ロジック: レイテンシの分位点 (p50/p95/p99)、
// session / model のフィルタ、CSV / JSON のメトリクス書き出し。
// node --test crates/denno-view/tests/ で走る。DOM には触らない。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  PERCENTILES, EXPORT_COLUMNS, FILTER_KEYS,
  percentile, latencyPercentiles, latencySummary,
  matchesFilter, filterItems,
  metricsRows, csvCell, toCsv, metricsCsv, metricsJson,
  createDash, noteFrame, noteToken, step,
} from '../src/dash.mjs';

test('PERCENTILES / EXPORT_COLUMNS / FILTER_KEYS の契約', () => {
  assert.deepEqual(PERCENTILES, [50, 95, 99]);
  assert.deepEqual(EXPORT_COLUMNS, ['metric', 'value', 'samples', 'session', 'model']);
  assert.deepEqual(FILTER_KEYS, ['session', 'model']);
});

test('percentile は空配列で null', () => {
  assert.equal(percentile([], 50), null);
  assert.equal(percentile(null, 50), null);
  assert.equal(percentile(undefined, 99), null);
  assert.equal(percentile([], 50), null);
  // 全部 NaN / Infinity も空と同じ
  assert.equal(percentile([NaN, Infinity, -Infinity], 50), null);
  // p が非数なら null
  assert.equal(percentile([1, 2, 3], NaN), null);
  assert.equal(percentile([1, 2, 3], 'x'), null);
});

test('percentile は 1 本ならその値', () => {
  assert.equal(percentile([7], 0), 7);
  assert.equal(percentile([7], 50), 7);
  assert.equal(percentile([7], 100), 7);
  assert.equal(percentile([7], 99), 7);
});

test('percentile は整列済み配列で線形補間する (R-7)', () => {
  const sorted = [10, 20, 30, 40, 50];
  assert.equal(percentile(sorted, 0), 10);
  assert.equal(percentile(sorted, 50), 30);
  assert.equal(percentile(sorted, 100), 50);
  // h = (5-1)*0.95 = 3.8 → 40 + (50-40)*0.8 = 48
  assert.ok(Math.abs(percentile(sorted, 95) - 48) < 1e-12);
  // h = 3.96 → 40 + 10*0.96 = 49.6
  assert.ok(Math.abs(percentile(sorted, 99) - 49.6) < 1e-12);
  assert.equal(percentile(sorted, 25), 20);
});

test('percentile は未整列でも同じ結果で、入力を書き換えない', () => {
  const unsorted = [50, 10, 40, 20, 30];
  const copy = unsorted.slice();
  assert.equal(percentile(unsorted, 50), 30);
  assert.ok(Math.abs(percentile(unsorted, 95) - 48) < 1e-12);
  assert.deepEqual(unsorted, copy, '入力を並べ替えてはいけない');
});

test('percentile は p を 0..100 に丸める', () => {
  const v = [1, 2, 3, 4];
  assert.equal(percentile(v, -10), 1);
  assert.equal(percentile(v, 1000), 4);
  assert.equal(percentile(v, -10), percentile(v, 0));
  assert.equal(percentile(v, 1000), percentile(v, 100));
});

test('latencyPercentiles は p50/p95/p99 と count/min/max', () => {
  const empty = latencyPercentiles([]);
  assert.deepEqual(empty, { p50: null, p95: null, p99: null, count: 0, min: null, max: null });

  const one = latencyPercentiles([12.5]);
  assert.deepEqual(one, { p50: 12.5, p95: 12.5, p99: 12.5, count: 1, min: 12.5, max: 12.5 });

  const p = latencyPercentiles([40, 10, NaN, 30, 20, 50, Infinity]);
  assert.equal(p.count, 5);
  assert.equal(p.min, 10);
  assert.equal(p.max, 50);
  assert.equal(p.p50, 30);
  assert.ok(Math.abs(p.p95 - 48) < 1e-12);
  assert.ok(Math.abs(p.p99 - 49.6) < 1e-12);

  // 非配列でも壊れない
  assert.equal(latencyPercentiles(null).count, 0);
});

test('latencySummary は空のリングで - を並べる', () => {
  const d = createDash();
  const s = latencySummary(d);
  assert.equal(s.count, 0);
  assert.equal(s.p50, null);
  assert.equal(s.text, 'p50 - · p95 - · p99 -');
  // dash が null でも壊れない
  assert.equal(latencySummary(null).text, 'p50 - · p95 - · p99 -');
});

test('latencySummary は Step フレームから p50/p95/p99 を作る', () => {
  const d = createDash();
  for (const ms of [10, 20, 30, 40, 50]) {
    assert.equal(noteFrame(d, { kind: 'Step', ns: ms * 1e6 }), true);
  }
  const s = latencySummary(d);
  assert.equal(s.count, 5);
  assert.equal(s.p50, 30);
  assert.ok(Math.abs(s.p95 - 48) < 1e-12);
  assert.ok(Math.abs(s.p99 - 49.6) < 1e-12);
  assert.equal(s.text, 'p50 30.0 ms · p95 48.0 ms · p99 49.6 ms');
  // 単位を落とせる (バッジの幅が狭いとき用)
  assert.ok(!latencySummary(d, { unit: false }).text.includes(' ms'));
});

test('latencySummary は 100 以上を整数で出す (fmtRate と同じ丸め)', () => {
  const d = createDash();
  for (const ms of [100, 200, 300]) noteFrame(d, { kind: 'Step', ns: ms * 1e6 });
  const s = latencySummary(d);
  assert.equal(s.text, 'p50 200 ms · p95 290 ms · p99 298 ms');
});

test('matchesFilter は空の条件で全部通す', () => {
  const row = { metric: 'decode', session: 's1', model: 'm1' };
  assert.equal(matchesFilter(row), true);
  assert.equal(matchesFilter(row, {}), true);
  assert.equal(matchesFilter(row, null), true);
  assert.equal(matchesFilter(row, { session: '', model: '  ' }), true);
  assert.equal(matchesFilter(row, { session: null, model: undefined }), true);
});

test('matchesFilter は session / model の完全一致 (AND)', () => {
  const row = { metric: 'decode', session: 's1', model: 'm1' };
  assert.equal(matchesFilter(row, { session: 's1' }), true);
  assert.equal(matchesFilter(row, { model: 'm1' }), true);
  assert.equal(matchesFilter(row, { session: 's1', model: 'm1' }), true);
  assert.equal(matchesFilter(row, { session: 's2' }), false);
  assert.equal(matchesFilter(row, { model: 'm2' }), false);
  assert.equal(matchesFilter(row, { session: 's1', model: 'm2' }), false);
  // 前後の空白は無視する
  assert.equal(matchesFilter(row, { session: ' s1 ' }), true);
  // 大文字小文字は区別する
  assert.equal(matchesFilter(row, { session: 'S1' }), false);
  // 項目側に値が無ければ条件付きでは落とす
  assert.equal(matchesFilter({ metric: 'decode' }, { session: 's1' }), false);
  assert.equal(matchesFilter({ metric: 'decode' }, {}), true);
  // 壊れた項目は落とす
  assert.equal(matchesFilter(null, {}), false);
  assert.equal(matchesFilter('decode', {}), false);
});

test('filterItems は配列を絞り、非配列は空', () => {
  const rows = [
    { session: 's1', model: 'm1' },
    { session: 's1', model: 'm2' },
    { session: 's2', model: 'm1' },
  ];
  assert.equal(filterItems(rows, { session: 's1' }).length, 2);
  assert.equal(filterItems(rows, { model: 'm1' }).length, 2);
  assert.deepEqual(filterItems(rows, { session: 's1', model: 'm2' }), [{ session: 's1', model: 'm2' }]);
  assert.equal(filterItems(rows, {}).length, 3);
  assert.deepEqual(filterItems(null, {}), []);
  assert.deepEqual(filterItems(undefined, { session: 's1' }), []);
  // 元の配列は触らない
  assert.equal(rows.length, 3);
});

test('metricsRows は系列と分位点の行を session / model 付きで出す', () => {
  const d = createDash();
  assert.equal(noteFrame(d, { kind: 'Step', ns: 10e6 }), true);
  assert.equal(noteFrame(d, { kind: 'Step', ns: 30e6 }), true);
  noteToken(d, 3, 0);
  step(d, 1000);

  const rows = metricsRows(d, { session: 's1', model: 'm1' });
  assert.deepEqual(rows.map((r) => r.metric), [
    'decode', 'prefill', 'vram', 'latency_p50', 'latency_p95', 'latency_p99',
  ]);
  for (const r of rows) {
    assert.equal(r.session, 's1');
    assert.equal(r.model, 'm1');
  }
  const byMetric = Object.fromEntries(rows.map((r) => [r.metric, r]));
  assert.equal(byMetric.decode.value, 3);
  assert.equal(byMetric.decode.samples, 1);
  assert.equal(byMetric.prefill.value, null);
  assert.equal(byMetric.prefill.samples, 0);
  assert.equal(byMetric.vram.value, null);
  assert.equal(byMetric.latency_p50.value, 20);
  assert.equal(byMetric.latency_p50.samples, 2);
  assert.equal(byMetric.latency_p99.value, 29.8);

  // 空のダッシュでも 6 行 (値は null)
  const empty = metricsRows(createDash());
  assert.equal(empty.length, 6);
  for (const r of empty) {
    assert.equal(r.value, null);
    assert.equal(r.samples, 0);
    assert.equal(r.session, '');
    assert.equal(r.model, '');
  }
  // d が null でも壊れない
  assert.equal(metricsRows(null).length, 6);
});

test('csvCell は区切り・引用・改行を逃がす', () => {
  assert.equal(csvCell(null), '');
  assert.equal(csvCell(undefined), '');
  assert.equal(csvCell(12.5), '12.5');
  assert.equal(csvCell(0), '0');
  assert.equal(csvCell(NaN), '');
  assert.equal(csvCell(Infinity), '');
  assert.equal(csvCell('decode'), 'decode');
  assert.equal(csvCell('a,b'), '"a,b"');
  assert.equal(csvCell('he said "hi"'), '"he said ""hi"""');
  assert.equal(csvCell('line1\nline2'), '"line1\nline2"');
  assert.equal(csvCell('cr\rhere'), '"cr\rhere"');
  assert.equal(csvCell(''), '');
});

test('toCsv はヘッダ + 行を出し、列を差し替えられる', () => {
  const rows = [
    { metric: 'decode', value: 12.5, samples: 3, session: 's,1', model: 'm1' },
    { metric: 'latency_p50', value: null, samples: 0, session: '', model: '' },
  ];
  const csv = toCsv(rows);
  assert.deepEqual(csv.split('\n'), [
    'metric,value,samples,session,model',
    'decode,12.5,3,"s,1",m1',
    'latency_p50,,0,,',
  ]);
  assert.equal(toCsv([]), 'metric,value,samples,session,model');
  assert.equal(toCsv(null), 'metric,value,samples,session,model');
  // 列を絞る
  assert.equal(toCsv(rows, ['metric', 'value']).split('\n')[0], 'metric,value');
  assert.equal(toCsv(rows, ['metric', 'value']).split('\n')[1], 'decode,12.5');
  // 空の列指定は既定に戻す
  assert.equal(toCsv(rows, []).split('\n')[0], 'metric,value,samples,session,model');
});

test('metricsCsv は現在値を CSV にし、filter で絞る', () => {
  const d = createDash();
  noteFrame(d, { kind: 'Step', ns: 20e6 });
  const csv = metricsCsv(d, { session: 's1', model: 'm1' });
  const lines = csv.split('\n');
  assert.equal(lines[0], 'metric,value,samples,session,model');
  assert.equal(lines.length, 7);
  assert.ok(lines.includes('latency_p50,20,1,s1,m1'));

  // 一致しないフィルタはヘッダだけ (数字を作らない)
  const none = metricsCsv(d, { session: 's1', model: 'm1', filter: { session: 's9' } });
  assert.equal(none, 'metric,value,samples,session,model');

  // model だけのフィルタは通る
  const byModel = metricsCsv(d, { session: 's1', model: 'm1', filter: { model: ' m1 ' } });
  assert.equal(byModel.split('\n').length, 7);

  // 列を絞れる
  const narrow = metricsCsv(d, { columns: ['metric', 'value'] });
  assert.equal(narrow.split('\n')[0], 'metric,value');
});

test('metricsJson は schema と行を返し、filter で絞る', () => {
  const d = createDash();
  noteToken(d, 2, 0);
  step(d, 1000);
  const out = JSON.parse(metricsJson(d, { session: 's1', model: 'm1' }));
  assert.equal(out.schema, 1);
  assert.equal(out.metrics.length, 6);
  assert.equal(out.metrics[0].metric, 'decode');
  assert.equal(out.metrics[0].value, 2);
  assert.equal(out.metrics[0].session, 's1');

  const none = JSON.parse(metricsJson(d, { session: 's1', filter: { session: 'zzz' } }));
  assert.deepEqual(none.metrics, []);
  assert.equal(none.schema, 1);
});
