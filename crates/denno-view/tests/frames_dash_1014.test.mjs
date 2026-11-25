// task 1014: frames.mjs / dash.mjs の壊れた入力に対する固めのテスト。
// node --test crates/denno-view/tests/ で走る。DOM には触らない。
//
// 対象:
//   - 途中で切れたバイナリフレーム (全 kind、任意の切断点)
//   - 未知の kind タグ (0 / 18 / 255 / 連結列の途中)
//   - NaN / Infinity / 負値のテレメトリサンプル
//   - リングバッファの折り返しと cap の暴走 (Infinity / 欠け / 巨大値)
//   - 大量サンプル (スプレッドによるスタック溢れを起こさないこと)
//   - 同じ入力から同じ出力になること (決定性)
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  decodeFrame, decodeFrames, encodeFrame, FrameDecodeError, KIND, VERSION,
} from '../src/frames.mjs';

import {
  SERIES_CAP, MAX_RING_CAP, MAX_LAYERS, GAUGE_WIDTH,
  createRing, ringPush, ringLast,
  createDash, notePrefill, noteToken, step,
  noteResidency, noteRoute, noteTierStats, noteLatency, noteKv, noteFrame,
  sparkPoints, sparkPath, latencyPercentiles, latencySummary, latencyRow,
  gaugeBar, metricsRows, dashRows,
} from '../src/dash.mjs';

// ---- 独立した書き手 (frames.test.mjs と同じ方針で自分で組む) ---------------

const MAGIC_BYTES = [0x44, 0x4e, 0x54, 0x4c];   // "DNTL"

function varint(v) {
  const out = [];
  let x = BigInt(v);
  if (x < 0n) throw new Error('varint: negative');
  while (x >= 0x80n) { out.push(Number(x & 0x7fn) | 0x80); x >>= 7n; }
  out.push(Number(x));
  return out;
}

function h16(bits) { return [bits & 0xff, (bits >> 8) & 0xff]; }

function str(s) {
  const b = [...new TextEncoder().encode(s)];
  return [...varint(b.length), ...b];
}

function frame(kindId, payload = []) {
  return Uint8Array.from([...MAGIC_BYTES, VERSION, kindId, ...payload]);
}

function cat(...parts) {
  const out = [];
  for (const p of parts) out.push(...p);
  return out;
}

function bytes(parts) {
  return Uint8Array.from(parts.flatMap((p) => [...p]));
}

/// 全 17 kind の 1 枚ずつ (切断点テスト用)。payload は lib.rs のレイアウト。
function oneFramePerKind() {
  return [
    frame(KIND.SERVER_INFO, cat(str('m'), str('id'), str('q'), varint(1), varint(2), varint(3))),
    frame(KIND.SESSION_ADMIT, cat(varint(1), varint(2), varint(3))),
    frame(KIND.SESSION_DONE, cat(varint(1), [7], varint(3))),
    frame(KIND.PREFILL, cat(varint(1), varint(2), varint(1_000_000_000), varint(2), varint(5), varint(6))),
    frame(KIND.TOKEN, cat(varint(1), varint(2), varint(3))),
    frame(KIND.STEP, cat(varint(1), varint(1_000_000_000))),
    frame(KIND.OP_SUMMARY, cat(varint(1), [4], varint(99))),
    frame(KIND.KV_USAGE, cat(varint(1), [1], varint(8))),
    frame(KIND.GAP, varint(1)),
    frame(KIND.TOP_K, cat(varint(1), varint(2), h16(0x3c00), varint(1), varint(3), h16(0x3800))),
    frame(KIND.LAYER_RUN, cat(varint(1), varint(1), [2], varint(9))),
    frame(KIND.LAYER_ACT, cat(varint(1), varint(1), varint(9), h16(0x3c00))),
    frame(KIND.ROLE_ACT, cat(varint(1), h16(0x3c00), h16(0x3c00), h16(0x3c00), h16(0x3c00), h16(0x3c00))),
    frame(KIND.EXPERT_RESIDENCY, cat(varint(1), varint(2), varint(9), varint(500))),
    frame(KIND.TIER_STATS, cat(varint(1), varint(2), varint(3), varint(4))),
    frame(KIND.ROUTER_TOPK, cat(varint(1), varint(2), varint(500), varint(501), h16(0x3c00), h16(0x3800))),
    frame(KIND.EXPERT_SKIP, cat(varint(1), varint(2), varint(300), varint(511))),
  ];
}

function expectDecodeError(fn, { partial, message, kindId } = {}) {
  let err = null;
  try { fn(); } catch (e) { err = e; }
  assert.ok(err, '例外が出なかった');
  assert.ok(err instanceof FrameDecodeError, 'FrameDecodeError ではない: ' + err.constructor.name);
  if (partial !== undefined) assert.equal(err.partial, partial, 'partial が違う: ' + err.message);
  if (message) assert.match(err.message, message);
  if (kindId !== undefined) assert.equal(err.kindId, kindId);
  assert.ok(Array.isArray(err.decoded), '.decoded が配列でない');
  return err;
}

// ---- 切断されたフレーム ---------------------------------------------------

test('全 17 kind: 6B..末尾のどの切断点でも FrameDecodeError (partial=true)', () => {
  for (const full of oneFramePerKind()) {
    const kindId = full[5];
    for (let cut = 6; cut < full.length; cut++) {
      const err = expectDecodeError(() => decodeFrame(full.slice(0, cut)), {
        partial: true,
      });
      assert.equal(err.offset, 0, 'kind ' + kindId + ' cut=' + cut + ' の offset');
      assert.deepEqual(err.decoded, [], 'kind ' + kindId + ' cut=' + cut + ' の decoded');
    }
  }
});

test('切断された 1 枚は decodeFrames でも FrameDecodeError (完全な prefix だけ decoded)', () => {
  const gap = frame(KIND.GAP, varint(9));
  const tail = frame(KIND.EXPERT_SKIP, cat(varint(3), varint(2), varint(1), varint(2)));
  const buf = bytes([gap, tail.slice(0, tail.length - 1)]);
  const err = expectDecodeError(() => decodeFrames(buf), { partial: true });
  assert.equal(err.decoded.length, 1);
  assert.deepEqual(err.decoded[0], { kind: 'Gap', lost: 9 });
  assert.equal(err.offset, gap.length);
});

test('count が巨大でも確保せず partial で弾く (LayerAct / ExpertSkip)', () => {
  const huge = 0xffffffff;
  for (const k of [KIND.LAYER_ACT, KIND.EXPERT_SKIP, KIND.ROUTER_TOPK]) {
    const b = frame(k, cat(varint(1), varint(huge), [0, 0]));
    expectDecodeError(() => decodeFrame(b), { partial: true });
  }
  // Prefill は session / prompt_tokens / ns の後ろが count
  const pre = frame(KIND.PREFILL, cat(varint(1), varint(1), varint(0), varint(huge)));
  expectDecodeError(() => decodeFrame(pre), { partial: true });
});

test('文字列長が残りバイトを超えたら partial (ServerInfo)', () => {
  const b = frame(KIND.SERVER_INFO, cat(varint(1000), [0x61, 0x62]));
  expectDecodeError(() => decodeFrame(b), { partial: true });
});

test('varint の桁あふれ / 長すぎは partial=false で弾く', () => {
  // u32 を超える 6 バイト varint (kind=Gap)
  const over = Uint8Array.from([...MAGIC_BYTES, VERSION, KIND.GAP,
    0xff, 0xff, 0xff, 0xff, 0xff, 0x7f]);
  expectDecodeError(() => decodeFrame(over), { partial: false, message: /overflows u32/ });

  // 終端しない varint (0x80 × 10)
  const tooLong = Uint8Array.from([...MAGIC_BYTES, VERSION, KIND.GAP,
    0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80]);
  expectDecodeError(() => decodeFrame(tooLong), { partial: false, message: /varint too long/ });
});

test('2^53 を超える ns は黙って丸めず partial=false で弾く (Prefill)', () => {
  const b = frame(KIND.PREFILL, cat(varint(1), varint(1), varint(2 ** 60), varint(0)));
  expectDecodeError(() => decodeFrame(b), { partial: false, message: /safe integer/ });
});

// ---- 未知の kind ----------------------------------------------------------

test('未知の kind タグ (0 / 18 / 19 / 255) は kindId 付き FrameDecodeError', () => {
  for (const k of [0, 18, 19, 255]) {
    const err = expectDecodeError(() => decodeFrame(frame(k, [0x00])), {
      partial: false, message: /unknown kind/, kindId: k,
    });
    assert.equal(err.offset, 0, 'kind ' + k + ' の offset はフレーム先頭');
  }
});

test('連結列の途中の未知 kind でも offset はそのフレームの先頭', () => {
  const gap = frame(KIND.GAP, varint(2));
  const buf = bytes([gap, frame(0x2a, [0x01, 0x02]), frame(KIND.GAP, varint(3))]);
  const err = expectDecodeError(() => decodeFrames(buf), { partial: false, kindId: 0x2a });
  assert.equal(err.offset, gap.length);
  assert.equal(err.decoded.length, 1);
});

test('未知 kind を decodeFrame の offset 付きで読んでもフレーム先頭を報告する', () => {
  const pad = [0xaa, 0xbb];
  const buf = bytes([pad, frame(0x7f, [0x00])]);
  const err = expectDecodeError(() => decodeFrame(buf, pad.length), { kindId: 0x7f });
  assert.equal(err.offset, pad.length);
});

// ---- offset / 入力型 ------------------------------------------------------

test('offset が端数・非数・負値なら bad offset (partial=false)', () => {
  const b = frame(KIND.GAP, varint(1));
  for (const off of [-1, 1.5, NaN, Infinity, -Infinity]) {
    expectDecodeError(() => decodeFrame(b, off), { partial: false, message: /bad offset/ });
  }
});

test('offset が末尾を超えたら partial (短すぎる扱い)', () => {
  const b = frame(KIND.GAP, varint(1));
  expectDecodeError(() => decodeFrame(b, b.length), { partial: true });
  expectDecodeError(() => decodeFrame(b, b.length + 100), { partial: true });
});

test('ArrayBuffer / 通常の配列 / Uint8Array のどれでも同じ結果', () => {
  const b = frame(KIND.GAP, varint(5));
  const fromArr = decodeFrame([...b]).frame;
  const fromU8 = decodeFrame(b).frame;
  const fromAb = decodeFrame(b.buffer.slice(0)).frame;
  assert.deepEqual(fromArr, { kind: 'Gap', lost: 5 });
  assert.deepEqual(fromU8, fromArr);
  assert.deepEqual(fromAb, fromArr);
});

// ---- 決定性 (同じ壊れた入力 → 同じ結果) ---------------------------------

test('壊れた入力の decodeFrames は同じ例外 (型/メッセージ/offset/partial) を返す', () => {
  const good = frame(KIND.GAP, varint(1));
  const inputs = [
    new Uint8Array(0),
    Uint8Array.from([0x00]),
    Uint8Array.from([...MAGIC_BYTES, VERSION, 0x63, 0x00]),
    Uint8Array.from([...good, 0x44, 0x4e]),
    Uint8Array.from([...good, ...good, 0x58, 0x4e, 0x54, 0x4c, VERSION, KIND.GAP]),
  ];
  for (const input of inputs) {
    const run = () => {
      try {
        return { ok: decodeFrames(input) };
      } catch (e) {
        return { err: { type: e.constructor.name, message: e.message, offset: e.offset, partial: e.partial } };
      }
    };
    const a = run();
    const b = run();
    assert.deepEqual(a, b, '同じ入力で結果が変わった');
    if (a.err) assert.equal(a.err.type, 'FrameDecodeError');
  }
});

test('決定的な擬似乱数バイト列でも FrameDecodeError 以外は出ない', () => {
  let s = 0x9e3779b9;
  const rnd = () => (s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  for (let iter = 0; iter < 400; iter++) {
    const n = Math.floor(rnd() * 96);
    const buf = new Uint8Array(n);
    for (let i = 0; i < n; i++) buf[i] = Math.floor(rnd() * 256);
    // 1/4 は先頭を正しい magic にして「payload だけ壊れている」を増やす
    if (iter % 4 === 0 && n >= 6) { buf[0] = 0x44; buf[1] = 0x4e; buf[2] = 0x54; buf[3] = 0x4c; buf[4] = VERSION; }
    let out = null, err = null;
    try { out = decodeFrames(buf); } catch (e) { err = e; }
    if (err) {
      assert.ok(err instanceof FrameDecodeError, 'iter ' + iter + ': ' + err.constructor.name);
      assert.ok(err.message.length > 0);
    } else {
      assert.ok(Array.isArray(out));
      for (const f of out) assert.equal(typeof f.kind, 'string');
    }
    // decodeFrames が成功したときは「消費バイト数ちょうど」まで進んでいる
    if (out) {
      let off = 0;
      for (let i = 0; i < out.length; i++) {
        const r = decodeFrame(buf, off);
        assert.ok(r.next > off, 'iter ' + iter + ': next が進まない');
        off = r.next;
      }
      assert.equal(off, buf.length);
    }
  }
});

// ---- dash: リングバッファ ------------------------------------------------

test('createRing の cap は [2, MAX_RING_CAP] に正規化される', () => {
  assert.equal(createRing().cap, SERIES_CAP);
  assert.equal(createRing(1).cap, 2);
  assert.equal(createRing(-5).cap, 2);
  assert.equal(createRing(3.9).cap, 3);
  assert.equal(createRing(0).cap, SERIES_CAP);          // 0 は既定 (従来どおり)
  assert.equal(createRing('x').cap, SERIES_CAP);
  assert.equal(createRing(NaN).cap, SERIES_CAP);
  assert.equal(createRing(Infinity).cap, SERIES_CAP);   // 非有限は既定に落とす
  assert.equal(createRing(1e12).cap, MAX_RING_CAP);
  assert.equal(MAX_RING_CAP, 65536);
});

test('ringPush は非有限・負値・空白文字列を積まない', () => {
  const ring = createRing(4);
  for (const bad of [NaN, Infinity, -Infinity, -1, -0.5, null, undefined, '', '   ', 'x', {}, []]) {
    assert.equal(ringPush(ring, bad), false, 'bad=' + String(bad));
  }
  assert.deepEqual(ring.values, []);
  // 数値文字列は受ける (WS から来る JSON の文字列を想定)
  assert.equal(ringPush(ring, ' 2.5 '), true);
  assert.deepEqual(ring.values, [2.5]);
  // bool / 配列は暗黙変換しない (false → 0 にしない)
  assert.equal(ringPush(ring, false), false);
  assert.equal(ringPush(ring, []), false);
  assert.deepEqual(ring.values, [2.5]);
});

test('ringPush は折り返して最後の cap 本だけを古い順に残す', () => {
  const ring = createRing(4);
  for (let i = 0; i < 10; i++) ringPush(ring, i);
  assert.deepEqual(ring.values, [6, 7, 8, 9]);
  assert.equal(ringLast(ring), 9);
  assert.equal(ring.values.length, ring.cap);
});

test('cap が欠けた / Infinity のリングでも青天井に伸びない', () => {
  const broken = { values: [] };                        // cap なし
  for (let i = 0; i < SERIES_CAP + 50; i++) ringPush(broken, i);
  assert.equal(broken.cap, SERIES_CAP);
  assert.equal(broken.values.length, SERIES_CAP);
  assert.equal(ringLast(broken), SERIES_CAP + 49);

  const inf = { cap: Infinity, values: [] };
  for (let i = 0; i < 7; i++) ringPush(inf, i);
  assert.equal(inf.cap, SERIES_CAP);
  assert.deepEqual(inf.values, [0, 1, 2, 3, 4, 5, 6]);
});

test('createDash の cap / windowMs も壊れた値で暴走しない', () => {
  assert.equal(createDash({ cap: Infinity }).cap, 90);
  assert.equal(createDash({ cap: 1e12 }).cap, MAX_RING_CAP);
  assert.equal(createDash({ cap: -3 }).cap, 2);
  assert.equal(createDash({ cap: 'x' }).cap, 90);
  assert.equal(createDash({ windowMs: Infinity }).windowMs, 1000);
  assert.equal(createDash({ windowMs: NaN }).windowMs, 1000);
  assert.equal(createDash({ windowMs: -5 }).windowMs, 1000);
  assert.equal(createDash({ windowMs: 250 }).windowMs, 250);
  assert.equal(createDash(null).windowMs, 1000);
});

// ---- dash: 集計ヘルパ ----------------------------------------------------

test('notePrefill は壊れた ns / tokens を捨てる', () => {
  const d = createDash();
  for (const fr of [
    { kind: 'Prefill', ns: NaN, tokens: [1] },
    { kind: 'Prefill', ns: Infinity, tokens: [1] },
    { kind: 'Prefill', ns: -1e9, tokens: [1] },
    { kind: 'Prefill', ns: 0, tokens: [1] },
    { kind: 'Prefill', ns: 1e9, tokens: [], prompt_tokens: NaN },
    { kind: 'Prefill', ns: 'x', tokens: [1, 2] },
  ]) {
    assert.equal(notePrefill(d, fr), false, JSON.stringify(fr));
  }
  // 桁あふれ (1e308 tok / 1 ns) は Infinity を作らず捨てる
  assert.equal(notePrefill(d, { kind: 'Prefill', ns: 1, prompt_tokens: 1e308, tokens: [] }), false);
  assert.deepEqual(d.series.prefill, []);
  assert.equal(d.last.prefill, null);
  assert.equal(d.prefills, 0);
});

test('noteToken は Infinity / NaN / 負値 / 0 を数えない', () => {
  const d = createDash();
  for (const n of [Infinity, -Infinity, NaN, -3, 0, 'x', null, {}]) {
    assert.equal(noteToken(d, n, 0), false, 'n=' + String(n));
  }
  assert.equal(d.winTokens, 0);
  assert.equal(d.tokens, 0);
  assert.equal(noteToken(d, 2, 0), true);
  assert.equal(d.winTokens, 2);
  // 壊れた状態からでも累積は有限に戻る
  d.winTokens = NaN;
  d.tokens = -10;
  assert.equal(noteToken(d, 3, 1), true);
  assert.equal(d.winTokens, 3);
  assert.equal(d.tokens, 3);
});

test('step は壊れた winTokens / 戻った時計で非有限サンプルを積まない', () => {
  const d = createDash({ windowMs: 1000 });
  step(d, 0);
  d.winTokens = Infinity;
  assert.equal(step(d, 1000), false);                   // 窓は畳むがサンプルは作らない
  assert.deepEqual(d.series.decode, []);
  assert.deepEqual(d.decodeRing.values, []);
  assert.equal(d.winTokens, 0);
  assert.equal(d.winStart, 1000);

  d.winTokens = NaN;
  assert.equal(step(d, 2000), false);
  assert.deepEqual(d.series.decode, []);

  // 壊れた状態からでも、次の正しい窓は普通に積める
  noteToken(d, 1, 3000);
  assert.equal(step(d, 3000), true);
  assert.deepEqual(d.series.decode, [1]);

  // 時計が戻ったら何もしない
  const before = d.series.decode.length;
  assert.equal(step(d, 1500), false);
  assert.equal(d.series.decode.length, before);
  for (const v of d.series.decode) assert.ok(Number.isFinite(v) && v >= 0);
});

test('noteResidency は NaN / Infinity / 負の expert id を落とす', () => {
  const d = createDash();
  assert.equal(noteResidency(d, { kind: 'ExpertResidency', layer: NaN, vram: [1] }), false);
  assert.equal(noteResidency(d, { kind: 'ExpertResidency', layer: -1, vram: [1] }), false);
  assert.equal(noteResidency(d, { kind: 'ExpertResidency', layer: 3, vram: 'x' }), false);
  assert.equal(noteResidency(d, {
    kind: 'ExpertResidency', layer: 3,
    vram: [1, NaN, Infinity, -3, '2', 1.9, null, undefined],
  }), true);
  assert.deepEqual([...d.residency.get(3)].sort((a, b) => a - b), [1, 2]);
  for (const id of d.residency.get(3)) assert.ok(Number.isSafeInteger(id) && id >= 0);
});

test('noteRoute は壊れた expert id を分母から外し、全部壊れていれば何もしない', () => {
  const d = createDash();
  noteResidency(d, { kind: 'ExpertResidency', layer: 3, vram: [1, 2] });

  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: [NaN, Infinity, -1] }), false);
  assert.deepEqual(d.series.vram, []);
  assert.deepEqual(d.residency.get(3), new Set([1, 2]), '常駐は壊れたルートで変わらない');

  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: [1, NaN, 9] }), true);
  assert.equal(d.last.vram, 0.5);
  assert.ok(d.last.vram >= 0 && d.last.vram <= 1);
  // layer が壊れていれば何もしない
  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: Infinity, experts: [1] }), false);
  assert.equal(noteRoute(d, { kind: 'RouterTopK', layer: 3, experts: 'x' }), false);
});

test('noteTierStats は負値 / 非有限を捨て、補助値も非負に揃える', () => {
  const d = createDash();
  assert.equal(noteTierStats(d, { kind: 'TierStats', vramMB: -1, vramTotalMB: 10 }), false);
  assert.equal(noteTierStats(d, { kind: 'TierStats', vramMB: 1, vramTotalMB: -10 }), false);
  assert.equal(noteTierStats(d, { kind: 'TierStats', vramMB: NaN, vramTotalMB: 10 }), false);
  assert.equal(noteTierStats(d, { kind: 'TierStats', vramMB: 5, vramTotalMB: Infinity }), false);
  assert.equal(d.tier, null);

  assert.equal(noteTierStats(d, {
    kind: 'TierStats', vramMB: 900, vramTotalMB: 1000, ramExperts: -4, ssdReads: Infinity,
  }), true);
  assert.deepEqual(d.tier, {
    usedMB: 900, totalMB: 1000, fill: 0.9, ramExperts: null, ssdReads: null,
  });
});

test('noteLatency は NaN / Infinity / 負値を捨てる', () => {
  const d = createDash();
  for (const v of [{ ns: NaN }, { ns: Infinity }, { ns: -1e6 }, NaN, Infinity, -2, 'x', null]) {
    assert.equal(noteLatency(d, v), false, JSON.stringify(v));
  }
  assert.deepEqual(d.latencyRing.values, []);
  assert.equal(noteLatency(d, { ns: 12_500_000 }), true);
  assert.equal(noteLatency(d, ' 3.5 '), true);
  assert.deepEqual(d.latencyRing.values, [12.5, 3.5]);
});

test('noteKv は busy が配列でなければ何もせず、真偽値だけ数える', () => {
  const d = createDash();
  assert.equal(noteKv(d, { kind: 'KvUsage' }), false);
  assert.equal(noteKv(d, { kind: 'KvUsage', busy: 'x' }), false);
  assert.equal(noteKv(d, { kind: 'KvUsage', busy: [] }), false);
  assert.equal(d.kv.busy, null);
  assert.equal(noteKv(d, { kind: 'KvUsage', busy: [true, false, 1, 0, 'x', null] }), true);
  assert.deepEqual(d.kv, { busy: 2, slots: 6 });
});

test('noteFrame は未知 kind と壊れた入力を黙って無視する', () => {
  const d = createDash();
  for (const fr of [null, undefined, 42, 'Frame', {}, { kind: 'Nope' }, { kind: 'kind99' }]) {
    assert.equal(noteFrame(d, fr, 0), false);
  }
  assert.equal(noteFrame(d, { kind: 'Prefill', ns: NaN }, 0), false);
  assert.equal(d.tokens, 0);
});

// ---- dash: 大量サンプル / 描画 ------------------------------------------

test('sparkPoints は 20 万点でもスプレッドで落ちない', () => {
  const n = 200000;
  const vals = new Array(n);
  for (let i = 0; i < n; i++) vals[i] = i % 97;
  const pts = sparkPoints(vals, { w: 220, h: 34, pad: 2 });
  assert.equal(pts.length, n);
  for (const p of pts) assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y));
  assert.equal(pts[0].x, 2);
  assert.ok(Math.abs(pts[n - 1].x - 218) < 1e-9);
  assert.ok(sparkPath(vals, { w: 20, h: 10, pad: 1 }).startsWith('M '));
});

test('sparkPoints は幅・高さ・余白の NaN / Infinity を既定値に落とす', () => {
  for (const opts of [
    { w: Infinity, h: Infinity, pad: Infinity },
    { w: NaN, h: NaN, pad: NaN },
    { w: -5, h: 0, pad: -1 },
    { w: 20, h: 10, pad: 1e9 },
  ]) {
    const pts = sparkPoints([1, 2, 3], opts);
    assert.equal(pts.length, 3);
    for (const p of pts) assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), JSON.stringify(opts));
  }
});

test('latencyPercentiles は 20 万点でも落ちない', () => {
  const n = 200000;
  const vals = new Array(n);
  for (let i = 0; i < n; i++) vals[i] = i;
  const p = latencyPercentiles(vals);
  assert.equal(p.count, n);
  assert.equal(p.min, 0);
  assert.equal(p.max, n - 1);
  assert.ok(Number.isFinite(p.p50) && Number.isFinite(p.p99));
});

test('gaugeBar は幅に Infinity / NaN / 0 / 負値を渡されても 1..40 に収める', () => {
  for (const w of [Infinity, NaN, 0, -3, 'x', null]) {
    const g = gaugeBar(1, 2, { w });
    assert.ok(Number.isInteger(g.width) && g.width >= 1 && g.width <= 40, 'w=' + String(w));
    assert.equal([...g.bar].length, g.width);
  }
  // 空文字の文字指定は既定のグリフに戻る
  const g = gaugeBar(1, 2, { w: 4, full: '', empty: '' });
  assert.equal(g.bar, '██░░');
  // 複数文字は 1 文字に切り詰めて幅を守る
  assert.equal(gaugeBar(1, 2, { w: 4, full: 'ab', empty: 'cd' }).bar, 'aacc');
});

test('系列/リングに直接 NaN が混ざっても描画と集計は落ちない', () => {
  const d = createDash();
  d.series.decode.push(NaN, 1, Infinity, 2);
  d.latencyRing.values.push(NaN, -5, 10, Infinity);
  d.last.decode = NaN;

  for (const r of dashRows(d)) {
    assert.equal(typeof r.text, 'string');
    assert.equal(typeof r.path, 'string');
    assert.ok(!r.path.includes('NaN') && !r.path.includes('Infinity'));
  }
  assert.ok(!latencyRow(d).text.includes(' ms') || latencyRow(d).text === '10.0 ms');
  assert.equal(latencySummary(d).count, 2);              // 有限値は 2 本
  const s = latencySummary(d);
  for (const k of ['p50', 'p95', 'p99']) assert.ok(Number.isFinite(s[k]), k);
  for (const row of metricsRows(d)) {
    assert.ok(row.value === null || Number.isFinite(row.value));
  }
});

// ---- dash: フレーム経由の決定性 ------------------------------------------

test('同じフレーム列を 2 回食わせると同じ計器状態になる (全系列が有限・非負)', () => {
  const capture = bytes([
    encodeFrame({ kind: 'ServerInfo', model: 'Flash', model_id: 'fn', quant: 'Q4_0', slots: 4, context: 65536, layers: 48 }),
    encodeFrame({ kind: 'SessionAdmit', session: 1, slot: 0, prompt_tokens: 300 }),
    encodeFrame({ kind: 'Prefill', session: 1, prompt_tokens: 1000, ns: 2e9, tokens: new Array(1000).fill(7) }),
    encodeFrame({ kind: 'Token', session: 1, token: 5, pos: 1 }),
    encodeFrame({ kind: 'Step', width: 4, ns: 10e6 }),
    encodeFrame({ kind: 'KvUsage', busy: [true, false, true], t_lens: [8, 0, 8] }),
    encodeFrame({ kind: 'ExpertResidency', layer: 3, vram: [1, 2] }),
    // 重みに NaN / Infinity を混ぜる (f16 で表現できる)
    encodeFrame({ kind: 'RouterTopK', layer: 3, experts: [1, 9], weights: [NaN, Infinity] }),
    encodeFrame({ kind: 'TierStats', vramMB: 900, vramTotalMB: 1000, ramExperts: 4, ssdReads: 2 }),
  ]);
  const frames = decodeFrames(capture);
  const run = () => {
    const d = createDash({ windowMs: 1000 });
    step(d, 0);
    for (let i = 0; i < frames.length; i++) noteFrame(d, frames[i], i * 500);
    step(d, 5000);
    return d;
  };
  const a = run(), b = run();

  // 決定性: 系列・直近値・リング・KV・Tier が一致する
  assert.deepEqual(a.series, b.series);
  assert.deepEqual(a.last, b.last);
  assert.deepEqual(a.decodeRing.values, b.decodeRing.values);
  assert.deepEqual(a.latencyRing.values, b.latencyRing.values);
  assert.deepEqual(a.kv, b.kv);
  assert.deepEqual(a.tier, b.tier);

  for (const key of ['decode', 'prefill', 'vram']) {
    for (const v of a.series[key]) assert.ok(Number.isFinite(v) && v >= 0, key + '=' + v);
  }
  for (const v of [...a.decodeRing.values, ...a.latencyRing.values]) {
    assert.ok(Number.isFinite(v) && v >= 0, 'ring=' + v);
  }
  assert.ok(a.last.vram === null || (a.last.vram >= 0 && a.last.vram <= 1));
  // RouterTopK の NaN 重みはヒット率に影響しない (expert 1 のみ常駐)
  assert.equal(a.last.vram, 0.5);
});

test('常駐表は壊れた層 id を連投されても MAX_LAYERS を超えて伸びない', () => {
  const d = createDash();
  const n = MAX_LAYERS + 100;
  for (let layer = 0; layer < n; layer++) {
    assert.equal(noteResidency(d, { kind: 'ExpertResidency', layer, vram: [1] }), true);
  }
  assert.ok(d.residency.size <= MAX_LAYERS, 'size=' + d.residency.size);
  // 直近の層は残っている (古い層から落ちる)
  assert.ok(d.residency.has(n - 1));
});

test('GAUGE_WIDTH は 10 のまま (表示の契約を変えていない)', () => {
  assert.equal(GAUGE_WIDTH, 10);
});
