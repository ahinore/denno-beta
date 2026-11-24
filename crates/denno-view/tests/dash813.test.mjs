// dash.mjs (task 813) の純ロジック。node --test crates/denno-view/tests/ で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DASH_SCHEMA, SERIES_KEYS, SERIES_LABELS, DEFAULT_CAP,
  createDash, notePrefill, noteToken, step, noteResidency, noteRoute,
  noteTierStats, noteFrame, fmtRate, fmtPercent, sparkPoints, sparkPath,
  dashRows, apiEndpoints, curlFor, API_ENDPOINTS,
} from '../src/dash.mjs';

test('createDash は空の系列と窓を持つ', () => {
  const d = createDash();
  assert.equal(d.schema, DASH_SCHEMA);
  assert.deepEqual(SERIES_KEYS, ['decode', 'prefill', 'vram']);
  for (const k of SERIES_KEYS) {
    assert.deepEqual(d.series[k], []);
    assert.equal(d.last[k], null);
  }
  assert.equal(d.winTokens, 0);
  assert.equal(d.tier, null);
});

test('notePrefill は「実際に食わせたトークン数 / 秒」を出す', () => {
  const d = createDash();
  // tokens 1000 本を 2.0 s で = 500 tok/s
  assert.equal(notePrefill(d, { kind: 'Prefill', prompt_tokens: 5000, ns: 2e9, tokens: new Array(1000).fill(7) }), true);
  assert.equal(d.series.prefill[0], 500);
  assert.equal(d.last.prefill, 500);
  assert.equal(d.prefills, 1);
  // tokens が無ければ prompt_tokens で代用する
  notePrefill(d, { kind: 'Prefill', prompt_tokens: 100, ns: 1e9, tokens: [] }, 0);
  assert.equal(d.series.prefill[1], 100);
  // ns=0 は割れないので何もしない (数字を作らない)
  assert.equal(notePrefill(d, { kind: 'Prefill', prompt_tokens: 10, ns: 0, tokens: [] }), false);
  assert.equal(d.series.prefill.length, 2);
});

test('decode は窓を締めたときだけサンプルを積む', () => {
  const d = createDash({ windowMs: 1000 });
  assert.equal(step(d, 0), false);          // 最初の step は窓の起点だけ
  assert.equal(d.winStart, 0);
  for (let i = 0; i < 20; i++) noteToken(d, 1, 500);
  assert.equal(step(d, 500), false);        // 窓の途中
  assert.equal(d.series.decode.length, 0);
  assert.equal(step(d, 1000), true);        // 1.0 s で 20 tok = 20 tok/s
  assert.equal(d.series.decode[0], 20);
  assert.equal(d.last.decode, 20);
  assert.equal(d.winStart, 1000);
  assert.equal(d.winTokens, 0);
  assert.equal(d.tokens, 20);
  // 次はトークン無し → 0 tok/s (アイドルも正直に出す)
  assert.equal(step(d, 2000), true);
  assert.equal(d.series.decode[1], 0);
  // 時計が戻ったら何もしない
  assert.equal(step(d, 1500), false);
});

test('decode は窓幅を超えた長さで割る (500 ms 窓)', () => {
  const d = createDash({ windowMs: 500 });
  step(d, 1000);
  noteToken(d, 7, 1100);
  assert.equal(step(d, 1600), true);        // 600 ms で 7 tok
  assert.ok(Math.abs(d.series.decode[0] - 7 * 1000 / 600) < 1e-9);
});

test('decode の系列は cap で古いものから落ちる', () => {
  const d = createDash({ cap: 3, windowMs: 10 });
  step(d, 0);
  for (let i = 1; i <= 5; i++) {
    noteToken(d, 1, i * 10);
    step(d, i * 10);
  }
  assert.equal(d.series.decode.length, 3);
  assert.equal(DEFAULT_CAP, 90);
});

test('vram ヒット率は「選択 expert のうち常駐している割合」', () => {
  const d = createDash();
  // 常駐が無い層のルートは何もしない
  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: [1, 2] }), false);
  assert.equal(d.series.vram.length, 0);
  assert.equal(noteResidency(d, { kind: 'ExpertResidency', layer: 3, vram: [1, 2, 3] }), true);
  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: [2, 3, 9] }), true);
  assert.ok(Math.abs(d.last.vram - 2 / 3) < 1e-12);
  assert.equal(d.hits, 1);
  // 常駐は置き換え (追記ではない)
  noteResidency(d, { kind: 'ExpertResidency', layer: 3, vram: [1] });
  noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: [1, 2] });
  assert.equal(d.last.vram, 0.5);
  // 空の選択は無視
  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: [] }), false);
  // 層が違えば常駐が無いので何もしない
  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 4, experts: [1] }), false);
});

test('TierStats は MB と disk を持つがヒット率の系列には入れない', () => {
  const d = createDash();
  assert.equal(noteTierStats(d, { kind: 'TierStats', vramMB: 900, vramTotalMB: 1000, ramExperts: 4, ssdReads: 2 }), true);
  assert.equal(d.tier.usedMB, 900);
  assert.equal(d.tier.totalMB, 1000);
  assert.equal(d.tier.fill, 0.9);
  assert.equal(d.series.vram.length, 0);
  // total 0 は割合を出さない
  noteTierStats(d, { kind: 'TierStats', vramMB: 0, vramTotalMB: 0, ramExperts: 0, ssdReads: 0 });
  assert.equal(d.tier.fill, null);
});

test('noteFrame は計器に要る kind だけ拾う', () => {
  const d = createDash();
  assert.equal(noteFrame(d, { kind: 'Token' }, 5), true);
  assert.equal(noteFrame(d, { kind: 'LayerRun', layer: 0, ops: [] }, 5), false);
  assert.equal(noteFrame(d, { kind: 'Gap', lost: 3 }, 5), false);
  assert.equal(noteFrame(d, null, 5), false);
  assert.equal(d.tokens, 1);
});

test('sparkPoints / sparkPath は境界を守る', () => {
  assert.deepEqual(sparkPoints([], { w: 220, h: 34, pad: 2 }), []);
  assert.equal(sparkPath([], {}), '');
  // 1 点は右端の点
  assert.deepEqual(sparkPoints([5], { w: 220, h: 34, pad: 2 }), [{ x: 218, y: 17 }]);
  // 2 点は左下 → 右上
  assert.deepEqual(sparkPoints([0, 10], { w: 220, h: 34, pad: 2 }), [
    { x: 2, y: 32 },
    { x: 218, y: 2 },
  ]);
  assert.equal(sparkPath([0, 10], { w: 220, h: 34, pad: 2 }), 'M 2 32 L 218 2');
  // 同じ値だけなら真ん中の水平線
  const flat = sparkPoints([3, 3, 3], { w: 20, h: 10, pad: 1 });
  assert.deepEqual(flat.map((p) => p.y), [5, 5, 5]);
  assert.deepEqual(flat.map((p) => p.x), [1, 10, 19]);
  // min/max を固定できる (系列の伸び方で縦が暴れないように)
  assert.deepEqual(sparkPoints([0, 10], { w: 20, h: 10, pad: 0, min: 0, max: 20 }), [
    { x: 0, y: 10 },
    { x: 20, y: 5 },
  ]);
  // 非有限値は落とす
  assert.deepEqual(sparkPoints([NaN, 1, Infinity], { w: 20, h: 10, pad: 0 }), [{ x: 20, y: 5 }]);
});

test('dashRows は 3 行をこの順で返し、空なら -', () => {
  const rows = dashRows(createDash(), { w: 220, h: 34, pad: 2 });
  assert.deepEqual(rows.map((r) => r.key), ['decode', 'prefill', 'vram']);
  assert.deepEqual(rows.map((r) => r.label), ['decode tok/s', 'prefill tok/s', 'vram hit']);
  for (const r of rows) {
    assert.equal(r.text, '-');
    assert.equal(r.path, '');
    assert.equal(r.samples, 0);
  }
  const d = createDash();
  notePrefill(d, { ns: 1e9, tokens: [1, 2, 3], prompt_tokens: 3 });
  const rows2 = dashRows(d);
  assert.equal(rows2[1].text, '3.0');
  assert.equal(rows2[1].samples, 1);
  assert.ok(rows2[1].path.startsWith('M '));
});

test('fmtRate / fmtPercent の丸め', () => {
  assert.equal(fmtRate(null), '-');
  assert.equal(fmtRate(-1), '-');
  assert.equal(fmtRate(12.345), '12.3');
  assert.equal(fmtRate(123.6), '124');
  assert.equal(fmtPercent(null), '-');
  assert.equal(fmtPercent(0.734), '73%');
  assert.equal(fmtPercent(1.5), '100%');
});

test('apiEndpoints は OpenAI と Anthropic の URL を出す', () => {
  assert.deepEqual(API_ENDPOINTS.map((e) => e.id), ['openai', 'anthropic']);
  const eps = apiEndpoints('http://127.0.0.1:8093/');
  assert.deepEqual(eps.map((e) => e.url), [
    'http://127.0.0.1:8093/v1/chat/completions',
    'http://127.0.0.1:8093/v1/messages',
  ]);
  // origin 無しなら path だけ (相対で貼れる)
  assert.deepEqual(apiEndpoints('').map((e) => e.url), ['/v1/chat/completions', '/v1/messages']);
  // 末尾スラッシュの二重化をしない
  assert.equal(apiEndpoints('http://h:1//')[0].url, 'http://h:1/v1/chat/completions');
});

test('curlFor はそのまま貼れる雛形を返す', () => {
  const oa = curlFor('openai', 'http://h:8093', 'denno-x');
  assert.ok(oa.includes('curl -s http://h:8093/v1/chat/completions'));
  assert.ok(oa.includes("content-type: application/json"));
  const body = oa.split("-d '")[1].replace(/'$/, '');
  const j = JSON.parse(body);
  assert.equal(j.model, 'denno-x');
  assert.equal(j.messages[0].role, 'user');
  assert.equal(j.max_tokens, 64);

  const an = curlFor('anthropic', 'http://h:8093');
  assert.ok(an.includes('http://h:8093/v1/messages'));
  assert.ok(an.includes('anthropic-version: 2023-06-01'));
  const ab = JSON.parse(an.split("-d '")[1].replace(/'$/, ''));
  assert.equal(ab.model, 'denno');
  assert.equal(ab.max_tokens, 64);

  assert.equal(curlFor('nope', 'http://h:1'), '');
});
