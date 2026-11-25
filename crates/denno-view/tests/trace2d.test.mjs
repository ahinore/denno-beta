// denno-view /2d のトレース記録・再生 (task 885) の純ロジック。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの:
//   1. フレームの正規化 (JSON 安全・配列の上限・未知 kind は捨てる)
//   2. 記録バッファ (上限で古いものを落とす・トークン数・バイト数)
//   3. 書き出し (.denno-trace.json) と読み込みの往復
//   4. スキーマ検証の失敗 (schema / version / shape / フレーム)
//   5. 再生のシーク (トークン区間・範囲外の丸め・preamble)
//   6. スクラブ描画用の状態 (トークンまでの再送でライブと同じ形になる)
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_SHAPE_RAW, TRACE_MAX_PAIRS, TRACE_SCHEMA, TRACE_SUFFIX, createMapState,
  TRACE_VERSION, createReplay, createTrace, replaySeek, replayStateAt, replayStep,
  replayTokenAt, replayTokenCount, traceDocument, traceFileName, traceFrame,
  traceMemory, traceParse, traceRecord, traceSerialize, traceStart, traceStop,
  traceSummary, traceValidate,
} from '../src/view2d.mjs';
import { mockTokenFrames } from '../src/view2d.mjs';

const frames1 = () => mockTokenFrames(DEFAULT_SHAPE_RAW, 1);
const frames2 = () => mockTokenFrames(DEFAULT_SHAPE_RAW, 2);

test('traceFrame: 地図に要る kind だけ残し、配列を上限で切る', () => {
  assert.equal(traceFrame(null), null);
  assert.equal(traceFrame({}), null);
  assert.equal(traceFrame({ kind: 'Mystery' }), null);
  // 未知の kind (投機の未来フレーム) は記録しない
  assert.equal(traceFrame({ kind: 'SpecDraft' }), null);

  const tok = traceFrame({ kind: 'Token', session: 3, token: 7, pos: 9, extra: 1 });
  assert.deepEqual(tok, { kind: 'Token', session: 3, token: 7, pos: 9 });

  const rt = traceFrame({ kind: 'RouterTopK', layer: 4, experts: [1, 2], weights: [0.25, 0.75] });
  assert.deepEqual(rt.experts, [1, 2]);
  assert.deepEqual(rt.weights, [0.25, 0.75]);

  // 重みが足りないときは 0 で埋め、多すぎるときは expert 数に合わせる
  const rt2 = traceFrame({ kind: 'RouterTopK', layer: 1, experts: [5, 6, 7], weights: [0.5] });
  assert.deepEqual(rt2.weights, [0.5, 0, 0]);

  // LayerAct は idx と val の短い方 + 上限
  const la = traceFrame({
    kind: 'LayerAct', layer: 2,
    idx: Array.from({ length: 400 }, (_, i) => i),
    val: Array.from({ length: 400 }, () => 0.5),
  });
  assert.equal(la.idx.length, TRACE_MAX_PAIRS);
  assert.equal(la.val.length, TRACE_MAX_PAIRS);

  // 非有限の数値は 0
  const bad = traceFrame({ kind: 'LayerAct', layer: 0, idx: [NaN], val: [Infinity] });
  assert.deepEqual(bad.idx, [0]);
  assert.deepEqual(bad.val, [0]);

  // RoleAct は 5 役割の数値だけ
  const role = traceFrame({ kind: 'RoleAct', layer: 3, roles: { attn: 1, router: NaN, junk: 9 } });
  assert.deepEqual(Object.keys(role.roles), ['attn', 'gdn', 'router', 'shex', 'ple']);
  assert.equal(role.roles.attn, 1);
  assert.equal(role.roles.router, 0);
});

test('traceRecord: 記録中だけ積み、上限で古いものを落として数える', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW, { maxFrames: 4 });
  assert.equal(traceRecord(rec, { kind: 'Token', token: 0 }), false, '開始前は積まない');
  traceStart(rec, 1000);
  assert.equal(rec.recording, true);
  assert.equal(rec.frames.length, 0);

  assert.equal(traceRecord(rec, { kind: 'Token', token: 0 }, 1001), true);
  assert.equal(traceRecord(rec, { kind: 'Nope' }), false);
  assert.equal(rec.tokens, 1);
  assert.ok(traceBytesOf(rec) > 0);

  for (let i = 1; i < 6; i++) traceRecord(rec, { kind: 'Token', token: i }, 1000 + i);
  assert.equal(rec.frames.length, 4, 'maxFrames で切る');
  assert.equal(rec.dropped, 2, '落とした数を数える');
  assert.equal(rec.tokens, 4, '落とした Token は数からも消す');
  assert.deepEqual(rec.frames.map((f) => f.token), [2, 3, 4, 5]);

  const mem = traceMemory(rec);
  assert.equal(mem.frames, 4);
  assert.equal(mem.maxFrames, 4);
  assert.ok(mem.bytes > 0);
  assert.equal(mem.pct, 1);
  assert.equal(mem.dropped, 2);

  const doc = traceStop(rec, { at: 5000 });
  assert.equal(rec.recording, false);
  assert.equal(doc.schema, TRACE_SCHEMA);
  assert.equal(doc.version, TRACE_VERSION);
  assert.equal(doc.created_at, 5000);
  assert.equal(doc.tokens, 4);
  assert.equal(doc.frames.length, 4);
  assert.equal(doc.shape.layers, DEFAULT_SHAPE_RAW.layers);
});

test('traceSerialize / traceParse: JSON の往復で形が変わらない', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW);
  traceStart(rec, 1);
  for (const f of frames1()) traceRecord(rec, f, 2);
  const doc = traceStop(rec, { at: 3 });
  const text = traceSerialize(doc);
  assert.ok(text.endsWith('\n'));
  const back = traceParse(text);
  assert.deepEqual(back, traceValidate(doc));
  assert.equal(back.tokens, 1);

  // 要約とファイル名
  const sum = traceSummary(back);
  assert.equal(sum.tokens, 1);
  assert.equal(sum.frames, doc.frames.length);
  assert.equal(sum.layers, DEFAULT_SHAPE_RAW.layers);
  assert.ok(sum.bytes > 0);
  assert.equal(traceFileName(Date.UTC(2026, 10, 25, 3, 4, 5)), 'trace-20261125-030405' + TRACE_SUFFIX);
});

test('traceValidate: 壊れた文書は理由つきで弾く', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW);
  traceStart(rec, 1);
  traceRecord(rec, { kind: 'Token', token: 0 });
  const good = JSON.parse(traceSerialize(traceStop(rec)));

  assert.throws(() => traceValidate(null), /object required/);
  assert.throws(() => traceValidate([]), /object required/);
  assert.throws(() => traceValidate(Object.assign({}, good, { schema: 'other' })), /schema/);
  assert.throws(() => traceValidate(Object.assign({}, good, { version: 2 })), /version/);
  assert.throws(() => traceValidate(Object.assign({}, good, { frames: {} })), /frames/);
  assert.throws(() => traceValidate(Object.assign({}, good, { shape: { layers: 0 } })), /layers/);
  assert.throws(() => traceValidate(Object.assign({}, good, { shape: null })), /shape|object/);
  assert.throws(
    () => traceValidate(Object.assign({}, good, { frames: [{ kind: 'SpecDraft' }] })),
    /frames\[0\]/);
  assert.throws(() => traceParse('{not json'), /invalid JSON/);

  // 空のトレースは通る (再生はトークン 0)
  const empty = traceValidate(Object.assign({}, good, { frames: [] }));
  assert.equal(empty.tokens, 0);
  assert.equal(createReplay(empty).groups.length, 0);
});

test('createReplay / replaySeek: Token ごとの区間と範囲の丸め', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW);
  traceStart(rec, 1);
  for (const f of frames1()) traceRecord(rec, f);
  for (const f of frames2()) traceRecord(rec, f);
  const doc = traceStop(rec);

  const rep = createReplay(doc);
  assert.equal(replayTokenCount(rep), 2);
  // 最初の Token より前の ServerInfo 等は preamble (このモックには無い)
  assert.equal(rep.groups[0].start, 0);
  assert.ok(rep.groups[0].end > rep.groups[0].start);
  assert.equal(rep.groups[1].start, rep.groups[0].end);
  assert.equal(rep.groups[1].end, doc.frames.length);

  const last = replaySeek(rep, 1);
  assert.equal(last.token, 1);
  assert.equal(last.end, doc.frames.length);
  assert.equal(last.frames.length, doc.frames.length);
  assert.equal(last.atEnd, true);

  const first = replaySeek(rep, 0);
  assert.equal(first.token, 0);
  assert.equal(first.end, rep.groups[0].end);
  assert.equal(first.atEnd, false);

  // 範囲外は端に丸める
  assert.equal(replaySeek(rep, -5).token, 0);
  assert.equal(replaySeek(rep, 99).token, 1);
  assert.equal(replaySeek(rep, 'x').token, 0);

  // 前後移動
  replaySeek(rep, 0);
  assert.equal(replayStep(rep, 1).token, 1);
  assert.equal(replayStep(rep, 1).token, 1, '端で止まる');
  assert.equal(replayStep(rep, -1).token, 0);

  // フレーム位置 → トークン
  assert.equal(replayTokenAt(rep, 0), 0);
  assert.equal(replayTokenAt(rep, rep.groups[1].start), 1);
  assert.equal(replayTokenAt(rep, doc.frames.length), 1);
  assert.equal(rep.token, 0, 'replayTokenAt は位置を動かさない');
});

test('replayStateAt: トークンまでの再送で地図の状態ができる', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW);
  traceStart(rec, 1);
  for (const f of frames1()) traceRecord(rec, f);
  for (const f of frames2()) traceRecord(rec, f);
  const rep = createReplay(traceStop(rec));

  const r0 = replayStateAt(rep, 0);
  const g0 = rep.frames.slice(0, rep.groups[0].end);
  const counted = g0.filter((f) => f.kind !== 'Token' && f.kind !== 'TopK').length;
  assert.equal(r0.token, 0);
  assert.equal(r0.state.token, 1);
  assert.equal(r0.state.frames, counted, '地図が数えるフレーム数');
  assert.equal(r0.topk.length, 1);
  assert.ok(r0.lastToken && r0.lastToken.kind === 'Token');

  const r1 = replayStateAt(rep, 1);
  assert.equal(r1.state.token, 2);
  assert.equal(r1.topk.length, 2);
  assert.equal(r1.atEnd, true);
  assert.ok(r1.frames > r0.frames);
  assert.equal(r1.state.frames, rep.frames.filter((f) => f.kind !== 'Token' && f.kind !== 'TopK').length);

  // ライブで全フレームを流した状態と同じ (累積の expert 出現回数)
  const ref = createMapState(DEFAULT_SHAPE_RAW);
  for (const f of rep.frames) ref.onFrame(f);
  assert.deepEqual([...r1.state.count[5]], [...ref.count[5]]);
  assert.deepEqual([...r1.state.count[40]], [...ref.count[40]]);

  // 空のトレースでも落ちない
  const empty = createReplay(traceValidate({
    schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW,
    frames: [], tokens: 0,
  }));
  const re = replayStateAt(empty, 0);
  assert.equal(re.token, -1);
  assert.equal(re.state.token, 0);
  assert.equal(replaySeek(empty, 3).token, -1);
  assert.deepEqual(replayStep(empty, 1).frames, []);
});

function traceBytesOf(rec) {
  // traceMemory 経由でバイト数を取る (内部カウンタの検査)
  return traceMemory(rec).bytes;
}
