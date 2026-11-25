// denno-view /2d の token probs / entropy 欄 (task 853) の純ロジック。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの:
//   1. TopK フレーム (kind 10) が wire → view に落ちる (f16 の確率・nat のエントロピー)
//   2. エントロピーの bit 変換と色 (単調)
//   3. 履歴の上限と順序、壊れたフレームの落とし方
//   4. 棒グラフの相対幅と chosen の印
//   5. モック生成器が 8 件の TopK を出す (画面をサーバー無しで確認できる)
import test from 'node:test';
import assert from 'node:assert/strict';

import { decodeFrame, encodeFrame } from '../src/frames.mjs';
import {
  DEFAULT_SHAPE_RAW, TOPK_HISTORY_MAX, entropyBits, entropyColor, mockTokenFrames,
  topkBarRows, topkFrameView, topkPush, topkSummary,
} from '../src/view2d.mjs';

const rawFrame = (over = {}) => ({
  kind: 'TopK', session: 1, chosen: 7, entropy: 1.2,
  entries: [[7, 0.5], [9, 0.25], [11, 0.125]],
  ...over,
});

test('TopK: wire (f16) → topkFrameView で id と確率になる', () => {
  const enc = encodeFrame({
    kind: 'TopK', session: 3, chosen: 42, entropy: 0.6931,
    entries: [[42, 0.5], [7, 0.25], [1000, 0.125]],
  });
  const f = decodeFrame(enc).frame;
  const v = topkFrameView(f);
  assert.equal(v.kind, 'TopK');
  assert.equal(v.chosen, 42);
  assert.ok(Math.abs(v.entropy - 0.6931) < 1e-3, 'nat のまま (f16)');
  assert.equal(v.entries.length, 3);
  assert.equal(v.entries[0].token, 42);
  assert.ok(Math.abs(v.entries[0].prob - 0.5) < 1e-3);
  assert.ok(Math.abs(v.entries[2].prob - 0.125) < 1e-3);
});

test('topkFrameView: TopK 以外は null、壊れた値は捨てる', () => {
  assert.equal(topkFrameView(null), null);
  assert.equal(topkFrameView({ kind: 'Token', token: 1 }), null);
  const v = topkFrameView(rawFrame({
    chosen: -3, entropy: Number.NaN,
    entries: [[4, 2.5], [-1, 0.2], [5, -0.5], 'x', [6]],
  }));
  assert.equal(v.chosen, 0, '負の chosen は 0');
  assert.equal(v.entropy, 0, 'NaN は 0');
  assert.deepEqual(v.entries, [{ token: 4, prob: 1 }, { token: 5, prob: 0 }],
    '範囲外の確率は 0..1 に丸め、形の壊れた entry は落とす');
});

test('entropyBits: nat → bit (ln2 = 1 bit), 0/NaN は 0', () => {
  assert.ok(Math.abs(entropyBits(Math.LN2) - 1) < 1e-9);
  assert.ok(Math.abs(entropyBits(2 * Math.LN2) - 2) < 1e-9);
  assert.equal(entropyBits(0), 0);
  assert.equal(entropyBits(-1), 0);
  assert.equal(entropyBits(Number.NaN), 0);
});

test('entropyColor: 低い = 緑 (hue 大) → 高い = 赤 (hue 小)、NaN は灰', () => {
  const hue = (css) => Number(/^hsl\((\d+)/.exec(css)[1]);
  const cold = hue(entropyColor(0));
  const warm = hue(entropyColor(1.0));
  const hot = hue(entropyColor(5));
  assert.ok(cold > warm && warm > hot, 'hue が単調に下がる');
  assert.equal(cold, 145);
  assert.equal(hot, 0);
  assert.equal(entropyColor(Number.NaN), 'rgb(120,140,150)');
});

test('topkPush: 古いものから落とし、TopK 以外は素通し (入力は壊さない)', () => {
  const base = [];
  const h1 = topkPush(base, rawFrame({ chosen: 1 }));
  assert.equal(base.length, 0, '入力配列を書き換えない');
  assert.equal(h1.length, 1);
  assert.equal(h1[0].chosen, 1);
  assert.equal(topkPush(h1, { kind: 'Token', token: 1 }), h1, 'TopK 以外は同じ配列を返す');

  let h = [];
  for (let i = 0; i < TOPK_HISTORY_MAX + 5; i++) h = topkPush(h, rawFrame({ chosen: i }));
  assert.equal(h.length, TOPK_HISTORY_MAX);
  assert.equal(h[0].chosen, 5, '先頭から落ちる');
  assert.equal(h[h.length - 1].chosen, TOPK_HISTORY_MAX + 4);
  assert.equal(topkPush([], rawFrame(), 2).length, 1, '上限 1 でも落ちない');
});

test('topkBarRows: 最大確率を 1 とした幅と chosen の印', () => {
  const rows = topkBarRows(rawFrame());
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((r) => r.token), [7, 9, 11]);
  assert.equal(rows[0].w, 1);
  assert.ok(Math.abs(rows[1].w - 0.5) < 1e-9);
  assert.ok(Math.abs(rows[2].w - 0.25) < 1e-9);
  assert.deepEqual(rows.map((r) => r.chosen), [true, false, false]);
  assert.deepEqual(topkBarRows({ kind: 'Token' }), []);
  // 確率が全部 0 でも NaN にしない
  const zero = topkBarRows(rawFrame({ entries: [[1, 0], [2, 0]] }));
  assert.deepEqual(zero.map((r) => r.w), [0, 0]);
});

test('topkSummary: bit 表示と先頭候補', () => {
  const s = topkSummary(rawFrame());
  assert.match(s, /^H 1\.73 bit · top 7 50\.0%$/, s);
  assert.equal(topkSummary({ kind: 'Gap', lost: 1 }), 'H -');
  // 正規化済みの view をそのまま渡しても同じ (画面は履歴の view を渡す)
  assert.equal(topkSummary(topkFrameView(rawFrame())), s);
});

test('mockTokenFrames: TopK (8 件・確率降順・chosen が先頭) を含む', () => {
  const frames = mockTokenFrames(DEFAULT_SHAPE_RAW, 3);
  const tk = frames.filter((f) => f.kind === 'TopK');
  assert.equal(tk.length, 1);
  assert.equal(tk[0].chosen, 3);
  assert.equal(tk[0].entries.length, 8);
  assert.equal(tk[0].entries[0][0], 3);
  for (let i = 1; i < 8; i++) {
    assert.ok(tk[0].entries[i][1] < tk[0].entries[i - 1][1], '確率が降順でない: ' + i);
  }
  assert.ok(tk[0].entropy >= 0.4 && tk[0].entropy <= 2.6 + 1e-9);
  // 決定的
  assert.deepEqual(mockTokenFrames(DEFAULT_SHAPE_RAW, 3), frames);
});
