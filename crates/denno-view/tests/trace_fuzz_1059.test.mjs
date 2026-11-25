// denno-view /2d のトレース記録・再生の境界 (task 1059)。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの:
//   1. TRACE_MAX_FRAMES (2048) ちょうど / はみ出しの巻き戻りと dropped の数え方
//   2. シーク位置の範囲外 (負・NaN・トークン数以上) と 0 トークンのトレース
//   3. 前後スクラブの決定性 (同じ位置なら毎回同じ状態)
//   4. 壊れた JSON 文書を例外で弾き、捕捉されない例外を投げないこと
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  DEFAULT_SHAPE_RAW, TRACE_MAX_ARRAY, TRACE_MAX_FRAMES, TRACE_MAX_PAIRS, TRACE_SCHEMA, TRACE_VERSION,
  createReplay, createTrace, mockTokenFrames, replaySeek, replayStateAt, replayStep,
  replayTokenAt, replayTokenCount, traceBytes, traceFrame, traceMemory, traceParse,
  traceRecord, traceStart, traceStop, traceValidate,
} from '../src/view2d.mjs';

function tokenFrames(n) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = { kind: 'Token', session: 1, token: i, pos: i };
  return out;
}

// mockTokenFrames(shape, n) は 1 トークン分。n 個ぶんを連結する。
function mockTokens(n, shape = DEFAULT_SHAPE_RAW) {
  const out = [];
  for (let i = 0; i < n; i++) out.push(...mockTokenFrames(shape, i));
  return out;
}

function replayOf(frames, shape = DEFAULT_SHAPE_RAW) {
  return createReplay({ schema: TRACE_SCHEMA, version: TRACE_VERSION, shape, frames, tokens: 0 });
}

test('TRACE_MAX_FRAMES: 2048 ちょうどまでは落とさない', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW);
  assert.equal(rec.maxFrames, TRACE_MAX_FRAMES);
  assert.equal(TRACE_MAX_FRAMES, 2048);
  traceStart(rec, 1);
  for (const f of tokenFrames(TRACE_MAX_FRAMES)) {
    assert.equal(traceRecord(rec, f, 2), true);
  }
  assert.equal(rec.frames.length, TRACE_MAX_FRAMES, '上限ちょうどは保持');
  assert.equal(rec.dropped, 0, '落としていない');
  assert.equal(rec.tokens, TRACE_MAX_FRAMES);
  assert.equal(traceMemory(rec).pct, 1);
  assert.equal(rec.frames[0].token, 0);
  assert.equal(rec.frames[TRACE_MAX_FRAMES - 1].token, TRACE_MAX_FRAMES - 1);
});

test('TRACE_MAX_FRAMES: はみ出すと古いものから落ち、dropped が正確', () => {
  const rec = createTrace(DEFAULT_SHAPE_RAW);
  traceStart(rec, 1);
  const extra = 12;
  for (const f of tokenFrames(TRACE_MAX_FRAMES + extra)) traceRecord(rec, f, 2);

  assert.equal(rec.frames.length, TRACE_MAX_FRAMES, '長さは上限のまま');
  assert.equal(rec.dropped, extra, '落とした数 = はみ出した数');
  assert.equal(rec.tokens, TRACE_MAX_FRAMES, '落とした Token もトークン数から消える');
  assert.equal(rec.frames[0].token, extra, '先頭ははみ出した分だけ進む');
  assert.equal(rec.frames[TRACE_MAX_FRAMES - 1].token, TRACE_MAX_FRAMES + extra - 1);

  // バイト数は残っているフレームの和と一致する (巻き戻しでずれない)
  const expectBytes = rec.frames.reduce((a, f) => a + JSON.stringify(f).length, 0);
  assert.equal(traceBytes(rec), expectBytes);

  // さらに積んでも dropped は単調に増え、長さは変わらない
  for (let i = 0; i < 5; i++) traceRecord(rec, { kind: 'Token', token: 90000 + i }, 3);
  assert.equal(rec.frames.length, TRACE_MAX_FRAMES);
  assert.equal(rec.dropped, extra + 5);

  // 記録を止めた文書も上限内で dropped を引き継ぐ
  traceStart(rec, 1);
  for (const f of tokenFrames(TRACE_MAX_FRAMES + 3)) traceRecord(rec, f, 2);
  const doc = traceStop(rec, { at: 9 });
  assert.equal(doc.frames.length, TRACE_MAX_FRAMES);
  assert.equal(doc.dropped, 3);
  assert.equal(doc.tokens, TRACE_MAX_FRAMES);
});

test('traceValidate: 2049 フレームの文書は理由つきで拒否', () => {
  const frames = tokenFrames(TRACE_MAX_FRAMES + 1);
  assert.throws(
    () => traceValidate({ schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW, frames }),
    /too many frames/);
  // 2048 ちょうどは通る
  const ok = traceValidate({
    schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW,
    frames: tokenFrames(TRACE_MAX_FRAMES),
  });
  assert.equal(ok.frames.length, TRACE_MAX_FRAMES);
  assert.equal(ok.tokens, TRACE_MAX_FRAMES);
});

test('replaySeek: 範囲外のシーク (負・NaN・トークン数以上) は端に丸める', () => {
  const r = replayOf(mockTokens(3));
  const n = replayTokenCount(r);
  assert.equal(n, 3);

  // NaN / Infinity は tInt が 0 に落としてから丸めるので先頭になる
  const cases = [
    [-1, 0], [-999, 0], [-Infinity, 0], [NaN, 0], ['x', 0], [null, 0], [Infinity, 0],
    [0.9, 0], [1.9, 1], [n - 1, n - 1], [n, n - 1], [n + 50, n - 1],
  ];
  for (const [input, want] of cases) {
    const out = replaySeek(r, input);
    assert.equal(out.token, want, 'seek(' + String(input) + ')');
    assert.equal(r.token, want, 'rep.token も同じ位置');
    // 返すフレームはその位置の区間の終端まで
    assert.equal(out.frames.length, out.end);
    assert.equal(out.atEnd, want === n - 1);
  }

  // 範囲内はそのまま、区間は単調
  for (let t = 0; t < n; t++) {
    const a = replaySeek(r, t);
    assert.equal(a.token, t);
    assert.ok(a.start >= 0 && a.start <= a.end);
    if (t > 0) assert.ok(a.start >= replaySeek(r, t - 1).end, '区間は重ならない');
  }
});

test('0 トークンのトレース: シークは -1 で全フレームを返し、落ちない', () => {
  // Token を含まない (preamble だけ) トレース。空配列とは別物として扱う。
  const preamble = [
    { kind: 'ServerInfo', model: 'm', layers: DEFAULT_SHAPE_RAW.layers },
    { kind: 'Prefill', session: 1, prompt_tokens: 2 },
    { kind: 'Gap', lost: 4 },
  ];
  const rep = replayOf(preamble);
  assert.equal(replayTokenCount(rep), 0);

  for (const at of [0, 1, -3, NaN, 99]) {
    const out = replaySeek(rep, at);
    assert.equal(out.token, -1);
    assert.equal(out.start, 0);
    assert.equal(out.end, rep.frames.length, '全フレームが見える');
    assert.equal(out.frames.length, rep.frames.length);
    assert.equal(out.atEnd, true);
  }
  assert.equal(replayTokenAt(rep, 0), -1, 'preamble は -1');
  assert.equal(replayTokenAt(rep, rep.frames.length), -1);

  const step = replayStep(rep, 1);
  assert.equal(step.token, -1);
  assert.deepEqual(step.frames, rep.frames);

  const st = replayStateAt(rep, 0);
  assert.equal(st.token, -1);
  assert.equal(st.lastToken, null, 'Token フレームは無い');
  assert.ok(st.state && typeof st.state.frames === 'number');
});

test('前後スクラブの決定性: 同じ位置なら毎回同じ状態になる', () => {
  const rep = replayOf(mockTokens(5));
  const n = replayTokenCount(rep);
  assert.equal(n, 5);

  const snap = (t) => {
    const s = replayStateAt(rep, t);
    return {
      token: s.token, frames: s.frames, atEnd: s.atEnd, topk: s.topk.length,
      stateToken: s.state.token, stateFrames: s.state.frames,
      last: s.lastToken ? s.lastToken.token : null,
    };
  };

  // 前方スクラブ → 後方スクラブ → もう一度前方、を 2 周
  const passes = [];
  for (let pass = 0; pass < 2; pass++) {
    const seen = [];
    for (let t = 0; t < n; t++) seen.push(snap(t));
    for (let t = n - 1; t >= 0; t--) {
      const again = snap(t);
      assert.deepEqual(again, seen[t], '後方スクラブで同じ状態 (pass ' + pass + ', t=' + t + ')');
    }
    passes.push(seen);
  }
  assert.deepEqual(passes[1], passes[0], '2 周目も同じ');

  // replayStep で往復しても同じ位置に戻る
  replaySeek(rep, 0);
  for (let i = 0; i < n - 1; i++) replayStep(rep, 1);
  assert.equal(rep.token, n - 1);
  for (let i = 0; i < n - 1; i++) replayStep(rep, -1);
  assert.equal(rep.token, 0);
  assert.deepEqual(snap(0), passes[0][0]);
});

test('traceValidate: スキーマ / バージョン / 形の欠落は捕捉可能な例外', () => {
  const base = { schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW, frames: [] };
  const bads = [
    [null, /object required/],
    [[], /object required/],
    ['x', /object required/],
    [{ ...base, schema: 'other' }, /schema/],
    [{ ...base, schema: undefined }, /schema/],
    [{ ...base, version: 2 }, /version/],
    [{ ...base, version: '1' }, /version/],
    [{ ...base, version: undefined }, /version/],
    [{ ...base, shape: undefined }, /shape|object/],
    [{ ...base, shape: null }, /shape|object/],
    [{ ...base, shape: [] }, /shape|object/],
    [{ ...base, shape: { layers: 0 } }, /layers/],
    [{ ...base, shape: { layers: -1 } }, /layers/],
    [{ ...base, shape: { layers: 'x' } }, /layers/],
    [{ ...base, shape: {} }, /layers/],
    [{ ...base, frames: {} }, /frames/],
    [{ ...base, frames: null }, /frames/],
    [{ ...base, frames: [null] }, /frames\[0\]/],
    [{ ...base, frames: ['Token'] }, /frames\[0\]/],
    [{ ...base, frames: [{ kind: 'Mystery' }] }, /frames\[0\]/],
    [{ ...base, frames: [{ kind: 42 }] }, /frames\[0\]/],
  ];
  for (const [doc, re] of bads) {
    let err = null;
    try { traceValidate(doc); } catch (e) { err = e; }
    assert.ok(err instanceof Error, '例外を投げる: ' + JSON.stringify(doc));
    assert.match(err.message, re);
  }
});

test('traceValidate: 1 フレームの巨大配列は上限で切って通す', () => {
  const big = 5000;
  const doc = traceValidate({
    schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW,
    frames: [{
      kind: 'LayerAct', layer: 0,
      idx: Array.from({ length: big }, (_, i) => i),
      val: Array.from({ length: big }, () => 0.5),
    }],
  });
  assert.equal(doc.frames[0].idx.length, TRACE_MAX_PAIRS);
  assert.equal(doc.frames[0].val.length, TRACE_MAX_PAIRS);
  // 配列が片方だけ / 非配列でも例外にせず空にする
  assert.deepEqual(traceFrame({ kind: 'LayerAct', idx: [1, 2], val: null }).idx, []);

  const rt = traceValidate({
    schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW,
    frames: [{ kind: 'RouterTopK', layer: 0, experts: Array(big).fill(1), weights: [0.5] }],
  });
  assert.equal(rt.frames[0].experts.length, TRACE_MAX_ARRAY);
  assert.equal(rt.frames[0].weights.length, rt.frames[0].experts.length, '足りない重みは 0 埋め');
});

test('非有限の値: フレーム内は 0 に落とし、JSON の NaN は invalid JSON', () => {
  const t = traceFrame({ kind: 'TopK', session: 1, chosen: NaN, entropy: Infinity, entries: [[0, NaN], [Infinity, -Infinity]] });
  assert.equal(t.chosen, 0);
  assert.equal(t.entropy, 0);
  assert.deepEqual(t.entries, [[0, 0], [0, 0]]);

  const la = traceFrame({ kind: 'LayerAct', layer: NaN, idx: [NaN, 1], val: [Infinity, 2] });
  assert.equal(la.layer, 0);
  assert.deepEqual(la.idx, [0, 1]);
  assert.deepEqual(la.val, [0, 2]);

  // JSON は NaN / Infinity を含められない → 構文エラーとして弾く
  assert.throws(
    () => traceParse('{"schema":"denno-trace","version":1,"shape":{"layers":1},"frames":[{"kind":"Step","ns":NaN}]}'),
    /invalid JSON/);
  assert.throws(() => traceParse('{"schema":"denno-trace","version":1,'), /invalid JSON/);
  assert.throws(() => traceParse(''), /invalid JSON/);

  // JSON 由来の null は 0 に正規化される
  const viaJson = traceValidate({
    schema: TRACE_SCHEMA, version: TRACE_VERSION, shape: DEFAULT_SHAPE_RAW,
    frames: [{ kind: 'Step', width: null, ns: null }],
  });
  assert.deepEqual(viaJson.frames[0], { kind: 'Step', width: 0, ns: 0 });
});
