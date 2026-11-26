// dash.mjs の書き出し・分位点・フィルタの境界 (task 1184)。
//   - CSV の quoting / injection: カンマ・引用符・改行・日本語・数式らしき文字列
//   - percentile / latencyPercentiles: 空配列・1 要素・非有限値混在
//   - matchesFilter / filterItems: session + model の複数条件、余計なキー、型の揺れ
// node --test crates/denno-view/tests/ で走る。DOM には触らない。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  percentile, latencyPercentiles, latencySummary,
  csvCell, toCsv, metricsCsv, metricsJson, metricsRows,
  matchesFilter, filterItems,
  EXPORT_COLUMNS, FILTER_KEYS,
  createDash, noteFrame,
} from '../src/dash.mjs';

// ---- CSV の quoting / injection --------------------------------------------

test('csvCell: 区切り文字を含むセルは引用する', () => {
  assert.equal(csvCell('a,b'), '"a,b"');
  assert.equal(csvCell('a,b,c'), '"a,b,c"');
  assert.equal(csvCell(',leading'), '",leading"');
  assert.equal(csvCell('trailing,'), '"trailing,"');
  // セミコロンやタブは区切りではないので引用しない
  assert.equal(csvCell('a;b'), 'a;b');
  assert.equal(csvCell('a\tb'), 'a\tb');
});

test('csvCell: 引用符は二重化して包む', () => {
  assert.equal(csvCell('"'), '""""');           // 1 個の引用 -> "" を包む
  assert.equal(csvCell('""'), '""""""');         // 2 個 -> 4 個 + 包み
  assert.equal(csvCell('a"b'), '"a""b"');
  assert.equal(csvCell('he said "hi"'), '"he said ""hi"""');
  // 引用符とカンマが同時でも二重化 + 包み
  assert.equal(csvCell('a,"b"'), '"a,""b"""');
});

test('csvCell: 改行 (LF / CR / CRLF) を含むセルは引用し、中身は保つ', () => {
  assert.equal(csvCell('l1\nl2'), '"l1\nl2"');
  assert.equal(csvCell('l1\rl2'), '"l1\rl2"');
  assert.equal(csvCell('l1\r\nl2'), '"l1\r\nl2"');
  // 末尾の改行も引用対象
  assert.equal(csvCell('end\n'), '"end\n"');
});

test('csvCell: 日本語は引用せずそのまま、日本語 + 区切りは引用する', () => {
  assert.equal(csvCell('こんにちは'), 'こんにちは');
  assert.equal(csvCell('モデル名'), 'モデル名');
  // 全角カンマ「，」は ASCII の区切りではないので引用しない
  assert.equal(csvCell('東京，大阪'), '東京，大阪');
  // 半角カンマは引用する
  assert.equal(csvCell('東京,大阪'), '"東京,大阪"');
  // 日本語 + 引用符
  assert.equal(csvCell('「引用」'), '「引用」');
  assert.equal(csvCell('名"前"'), '"名""前"""');
});

test('csvCell: 数式 injection の文字列は現状そのまま通す (値は改変しない)', () => {
  // csvCell は数式の先頭文字を無害化しない (データを勝手に書き換えない方針)。
  // この挙動を契約として固定しておく。将来エスケープを足すならここを変える。
  assert.equal(csvCell('=SUM(A1:A2)'), '=SUM(A1:A2)');
  assert.equal(csvCell('+1+1'), '+1+1');
  assert.equal(csvCell('-1+1'), '-1+1');
  assert.equal(csvCell('@cmd'), '@cmd');
  // 数式 + カンマは quoting だけ効く
  assert.equal(csvCell('=SUM(A1,A2)'), '"=SUM(A1,A2)"');
  // 数式 + 引用符 + 改行
  assert.equal(csvCell('="a",\nb'), '"=""a"",\nb"');
});

test('csvCell: 数値は有限のみ文字列化し、非有限と非数は空', () => {
  assert.equal(csvCell(0), '0');
  assert.equal(csvCell(-0), '0');
  assert.equal(csvCell(3.14), '3.14');
  assert.equal(csvCell(-7), '-7');
  assert.equal(csvCell(1e21), '1e+21');
  assert.equal(csvCell(NaN), '');
  assert.equal(csvCell(Infinity), '');
  assert.equal(csvCell(-Infinity), '');
  assert.equal(csvCell(null), '');
  assert.equal(csvCell(undefined), '');
  // bool / 配列 / オブジェクトは String() 経由 (既存仕様)
  assert.equal(csvCell(true), 'true');
  assert.equal(csvCell([1, 2]), '"1,2"');   // String() の結果にカンマがあるので引用
  assert.equal(csvCell({}), '[object Object]');   // String() 経由
});

test('toCsv: 日本語・カンマ・改行を含む行を正しく組み立てる', () => {
  const rows = [
    { metric: 'decode', value: 1.5, samples: 1, session: 'セッション,1', model: 'モデル"A"' },
    { metric: 'prefill', value: null, samples: 0, session: '東京\n大阪', model: 'm2' },
  ];
  const lines = toCsv(rows).split('\n');
  assert.equal(lines[0], EXPORT_COLUMNS.join(','));
  // 改行を含むセルは引用されるので、行の分割は「論理行」と一致しない点に注意。
  // ここでは 1 行目 (ヘッダ) と、引用済みセルの断片を確認する。
  assert.ok(lines[0] === 'metric,value,samples,session,model');
  const joined = toCsv(rows);
  assert.ok(joined.includes('"セッション,1"'));
  assert.ok(joined.includes('"モデル""A"""'));
  assert.ok(joined.includes('"東京\n大阪"'));
});

test('toCsv: 空・null 行・欠損列は空セルで埋める', () => {
  assert.equal(
    toCsv([{}, null, { metric: 'decode' }]),
    ['metric,value,samples,session,model', ',,,,', ',,,,', 'decode,,,,'].join('\n'),
  );
  // 行に余計なキーがあっても列指定の分だけ出す
  assert.equal(
    toCsv([{ metric: 'decode', extra: 'x' }], ['metric', 'extra']),
    ['metric,extra', 'decode,x'].join('\n'),
  );
});

test('toCsv: 列指定に null セル・重複・存在しない列を混ぜても壊れない', () => {
  const rows = [{ metric: 'decode', value: 2 }];
  // 存在しない列は csvCell(undefined) -> 空
  assert.equal(toCsv(rows, ['metric', 'missing']), 'metric,missing\ndecode,');
  // 重複列はそのまま 2 回出る
  assert.equal(toCsv(rows, ['metric', 'metric']), 'metric,metric\ndecode,decode');
  // 非配列の columns は既定に戻す
  assert.equal(toCsv(rows, 'nope').split('\n')[0], EXPORT_COLUMNS.join(','));
});

// ---- 分位点の境界 -----------------------------------------------------------

test('percentile: 空・非配列・全非有限はすべて null', () => {
  for (const p of [0, 50, 95, 99, 100]) {
    assert.equal(percentile([], p), null);
    assert.equal(percentile(null, p), null);
    assert.equal(percentile(undefined, p), null);
    assert.equal(percentile([NaN, Infinity, -Infinity], p), null);
    assert.equal(percentile('not-array', p), null);
  }
});

test('percentile: 1 要素は p に関係なくその値 (非有限は落として判定)', () => {
  assert.equal(percentile([42], 0), 42);
  assert.equal(percentile([42], 50), 42);
  assert.equal(percentile([42], 100), 42);
  // 非有限を混ぜても有効値が 1 本ならその値
  assert.equal(percentile([NaN, 42, Infinity], 99), 42);
  // 1 本だけの 0 も返す (falsy を null と取り違えない)
  assert.equal(percentile([0], 50), 0);
  // 負値も有効 (分位点は符号を見ない)
  assert.equal(percentile([-5], 50), -5);
});

test('percentile: 2 要素の線形補間', () => {
  assert.equal(percentile([10, 20], 0), 10);
  assert.equal(percentile([10, 20], 50), 15);
  assert.equal(percentile([10, 20], 100), 20);
  assert.ok(Math.abs(percentile([10, 20], 25) - 12.5) < 1e-12);
  // 入力順に依存しない
  assert.equal(percentile([20, 10], 50), 15);
});

test('percentile: 負値を含む配列でも昇順ソートで補間する', () => {
  const v = [5, -5, 0, 10, -10];
  assert.equal(percentile(v, 0), -10);
  assert.equal(percentile(v, 50), 0);
  assert.equal(percentile(v, 100), 10);
  // h = (5-1)*0.25 = 1 -> -5
  assert.equal(percentile(v, 25), -5);
});

test('latencyPercentiles: 空は全 null、count 0', () => {
  const p = latencyPercentiles([]);
  assert.deepEqual(p, { p50: null, p95: null, p99: null, count: 0, min: null, max: null });
  assert.equal(latencyPercentiles(null).count, 0);
  // 全非有限も count 0
  const q = latencyPercentiles([NaN, Infinity, -Infinity]);
  assert.deepEqual(q, { p50: null, p95: null, p99: null, count: 0, min: null, max: null });
});

test('latencyPercentiles: 1 要素は p50/p95/p99/min/max が同じ値', () => {
  const p = latencyPercentiles([12.5]);
  assert.deepEqual(p, { p50: 12.5, p95: 12.5, p99: 12.5, count: 1, min: 12.5, max: 12.5 });
  // 0 でも null と区別して 0 を返す
  const z = latencyPercentiles([0]);
  assert.equal(z.count, 1);
  assert.equal(z.p50, 0);
  assert.equal(z.p99, 0);
});

test('latencyPercentiles: 非有限を除いた件数と min/max を返す', () => {
  const p = latencyPercentiles([3, NaN, 1, Infinity, 2, -1]);
  assert.equal(p.count, 4);
  assert.equal(p.min, -1);
  assert.equal(p.max, 3);
  assert.equal(p.p50, 1.5);   // [−1,1,2,3]: h=(4-1)*0.5=1.5 -> 1+(2-1)*0.5
});

test('latencySummary: 空リングの text はすべて - で 3 つ並ぶ', () => {
  const text = latencySummary(createDash()).text;
  assert.equal(text, 'p50 - · p95 - · p99 -');
  assert.equal(text.split('·').length, 3);
});

// ---- 複数条件のフィルタ -----------------------------------------------------

test('matchesFilter: session と model の AND を満たす行だけ通す', () => {
  const row = { session: 's1', model: 'm1' };
  assert.equal(matchesFilter(row, { session: 's1', model: 'm1' }), true);
  assert.equal(matchesFilter(row, { session: 's1', model: 'm9' }), false);
  assert.equal(matchesFilter(row, { session: 's9', model: 'm1' }), false);
  assert.equal(matchesFilter(row, { session: 's9', model: 'm9' }), false);
  // 片方だけの条件
  assert.equal(matchesFilter(row, { session: 's1' }), true);
  assert.equal(matchesFilter(row, { model: 'm1' }), true);
  assert.equal(matchesFilter(row, { session: '', model: 'm1' }), true);
});

test('matchesFilter: 指定した条件の順序や余計なキーは結果に影響しない', () => {
  const row = { session: 's1', model: 'm1', extra: 'x' };
  assert.equal(matchesFilter(row, { model: 'm1', session: 's1' }), true);
  // FILTER_KEYS 以外のキーは無視する
  assert.equal(matchesFilter(row, { session: 's1', other: 'zzz' }), true);
  assert.equal(matchesFilter(row, { unknown: 'zzz' }), true);
  assert.deepEqual(FILTER_KEYS, ['session', 'model']);
});

test('matchesFilter: 前後空白を無視し、空条件は全部通す', () => {
  const row = { session: '  s1  ', model: 'm1' };
  assert.equal(matchesFilter(row, { session: 's1' }), true);
  assert.equal(matchesFilter(row, { session: '  s1 ' }), true);
  assert.equal(matchesFilter(row, { session: '   ' }), true);    // 空白だけの条件は全部
  assert.equal(matchesFilter(row, { session: null, model: undefined }), true);
  // 大文字小文字は区別する
  assert.equal(matchesFilter({ session: 'S1' }, { session: 's1' }), false);
});

test('matchesFilter: 日本語・カンマを含む識別子を完全一致で比べる', () => {
  const row = { session: 'セッション,1', model: 'モデルA' };
  assert.equal(matchesFilter(row, { session: 'セッション,1' }), true);
  assert.equal(matchesFilter(row, { session: 'セッション' }), false);   // 部分一致しない
  assert.equal(matchesFilter(row, { model: 'モデル' }), false);
  assert.equal(matchesFilter(row, { session: 'セッション,1', model: 'モデルA' }), true);
});

test('matchesFilter: 数値の条件・項目は文字列化して比べる', () => {
  const row = { session: 1, model: 2 };
  assert.equal(matchesFilter(row, { session: 1 }), true);
  assert.equal(matchesFilter(row, { session: '1' }), true);
  assert.equal(matchesFilter(row, { session: 2 }), false);
  // 数値条件と数値項目
  assert.equal(matchesFilter({ session: 10 }, { session: '10' }), true);
});

test('matchesFilter: 条件付きで項目側に値が無ければ落とす', () => {
  assert.equal(matchesFilter({ metric: 'decode' }, { session: 's1' }), false);
  assert.equal(matchesFilter({ metric: 'decode', session: null }, { session: 's1' }), false);
  // 条件が空なら項目が欠けていても通す
  assert.equal(matchesFilter({ metric: 'decode' }, {}), true);
});

test('filterItems: 複数条件で絞り、入力配列を書き換えない', () => {
  const rows = [
    { id: 1, session: 's1', model: 'm1' },
    { id: 2, session: 's1', model: 'm2' },
    { id: 3, session: 's2', model: 'm1' },
    { id: 4, session: 's2', model: 'm2' },
    // 壊れた行は条件付きで落ちる
    null,
    { id: 5 },
  ];
  const snapshot = JSON.stringify(rows);
  assert.deepEqual(filterItems(rows, { session: 's1', model: 'm1' }).map((r) => r.id), [1]);
  assert.deepEqual(filterItems(rows, { session: 's1' }).map((r) => r.id), [1, 2]);
  assert.deepEqual(filterItems(rows, { model: 'm1' }).map((r) => r.id), [1, 3]);
  assert.equal(filterItems(rows, {}).length, 5);   // null は落ち、{id:5} は条件なしで通る
  assert.equal(JSON.stringify(rows), snapshot, '入力を書き換えてはいけない');
});

test('filterItems: 非配列は空配列', () => {
  assert.deepEqual(filterItems(null, {}), []);
  assert.deepEqual(filterItems(undefined, { session: 's1' }), []);
  assert.deepEqual(filterItems({ length: 1 }, {}), []);
  assert.deepEqual(filterItems('abc', {}), []);
});

// ---- CSV / JSON 書き出しとフィルタの組み合わせ -----------------------------

test('metricsCsv: session のフィルタは行全体を落とし、ヘッダだけ残る', () => {
  const d = createDash();
  noteFrame(d, { kind: 'Step', ns: 20e6 });
  const csv = metricsCsv(d, { session: 's1', model: 'm1', filter: { session: 's9' } });
  assert.equal(csv, EXPORT_COLUMNS.join(','));
  // 一致すれば 6 行 + ヘッダ
  const ok = metricsCsv(d, { session: 's1', model: 'm1' });
  assert.equal(ok.split('\n').length, 7);
  assert.ok(ok.includes('latency_p50,20,1,s1,m1'));
});

test('metricsCsv: session / model にカンマや引用符が入っても CSV が崩れない', () => {
  const d = createDash();
  noteFrame(d, { kind: 'Step', ns: 10e6 });
  const csv = metricsCsv(d, { session: 's,1', model: 'm"1' });
  assert.ok(csv.includes('"s,1"'));
  assert.ok(csv.includes('"m""1"'));
  // ヘッダは常に素の列名
  assert.equal(csv.split('\n')[0], EXPORT_COLUMNS.join(','));
});

test('metricsJson: filter は session/model の AND で効き、schema は保つ', () => {
  const d = createDash();
  noteFrame(d, { kind: 'Step', ns: 10e6 });
  const hit = JSON.parse(metricsJson(d, { session: 's1', model: 'm1', filter: { session: 's1', model: 'm1' } }));
  assert.equal(hit.schema, 1);
  assert.equal(hit.metrics.length, 6);
  const miss = JSON.parse(metricsJson(d, { session: 's1', model: 'm1', filter: { session: 's1', model: 'm9' } }));
  assert.equal(miss.schema, 1);
  assert.deepEqual(miss.metrics, []);
});

test('metricsRows: 空ダッシュでも 6 行、値は null で件数 0', () => {
  const rows = metricsRows(createDash(), { session: 's1', model: 'm1' });
  assert.equal(rows.length, 6);
  for (const r of rows) {
    assert.equal(r.value, null);
    assert.equal(r.samples, 0);
    assert.equal(r.session, 's1');
    assert.equal(r.model, 'm1');
  }
});
