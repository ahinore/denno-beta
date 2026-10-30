// denno-view のフレームコーデック単体テスト (DOM 非依存)。
// 実行: node --test crates/denno-view/tests/
//
// ワイヤ形式の正は crates/denno-telemetry/src/lib.rs (Frame::encode / Frame::decode)。
// ここでは encode 相当のバイト列を **このファイル内で独立に組み立て**、decodeFrame で
// 読み戻して突き合わせる (frames.mjs の Writer と往復させるだけでは、両側が同じ
// 勘違いをしていても通ってしまうため)。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  decodeFrame, decodeFrames, encodeFrame, concatFrames,
  f16ToF32, f32ToF16, FrameDecodeError, KIND, ROLE_NAMES, kindName, VERSION
} from '../src/frames.mjs';

// ---- 独立した書き手 (lib.rs の put_varint / to_le_bytes の写し) ------------

const MAGIC_BYTES = [0x44, 0x4e, 0x54, 0x4c];   // "DNTL"

/** LEB128 (unsigned)。BigInt で 2^53 超も正確に書く。 */
function varint(v) {
  const out = [];
  let x = BigInt(v);
  if (x < 0n) throw new Error('varint: negative');
  while (x >= 0x80n) { out.push(Number(x & 0x7fn) | 0x80); x >>= 7n; }
  out.push(Number(x));
  return out;
}

/** f16 のビット (u16) をリトルエンディアン 2 バイトで。 */
function h16(bits) { return [bits & 0xff, (bits >> 8) & 0xff]; }

/** 文字列 = varint 長 + UTF-8。 */
function str(s) {
  const b = [...new TextEncoder().encode(s)];
  return [...varint(b.length), ...b];
}

/** magic + version + kind + payload。 */
function frame(kindId, payload) {
  return Uint8Array.from([...MAGIC_BYTES, VERSION, kindId, ...payload]);
}

function cat(...parts) {
  const out = [];
  for (const p of parts) out.push(...p);
  return out;
}

// ---- 正常系: 各 kind を lib.rs のレイアウトで組み立てて読み戻す -------------

test('kind 1 ServerInfo: 3 文字列 + slots/context/layers', () => {
  const b = frame(KIND.SERVER_INFO, cat(
    str('Qwen3.8-Flash-Next'), str('flash-next'), str('Q4_0'),
    varint(4), varint(65536), varint(48)
  ));
  const { frame: f, next } = decodeFrame(b);
  assert.deepEqual(f, {
    kind: 'ServerInfo', model: 'Qwen3.8-Flash-Next', model_id: 'flash-next', quant: 'Q4_0',
    slots: 4, context: 65536, layers: 48
  });
  assert.equal(next, b.length);
});

test('kind 2/3 SessionAdmit / SessionDone', () => {
  const a = decodeFrame(frame(KIND.SESSION_ADMIT, cat(varint(7), varint(1), varint(300)))).frame;
  assert.deepEqual(a, { kind: 'SessionAdmit', session: 7, slot: 1, prompt_tokens: 300 });

  const d = decodeFrame(frame(KIND.SESSION_DONE, cat(varint(9), [2], varint(128)))).frame;
  assert.deepEqual(d, { kind: 'SessionDone', session: 9, reason: 2, output_tokens: 128 });
});

test('kind 4 Prefill: ns は u64、tokens は varint 列', () => {
  const b = frame(KIND.PREFILL, cat(
    varint(1), varint(3), varint(4_100_000_000), varint(3),
    varint(1), varint(2), varint(70000)
  ));
  const f = decodeFrame(b).frame;
  assert.equal(f.kind, 'Prefill');
  assert.equal(f.session, 1);
  assert.equal(f.prompt_tokens, 3);
  assert.equal(f.ns, 4_100_000_000);
  assert.deepEqual(f.tokens, [1, 2, 70000]);
});

test('kind 5 Token / 6 Step', () => {
  const t = decodeFrame(frame(KIND.TOKEN, cat(varint(2), varint(248046), varint(900)))).frame;
  assert.deepEqual(t, { kind: 'Token', session: 2, token: 248046, pos: 900 });

  const s = decodeFrame(frame(KIND.STEP, cat(varint(4), varint(80_000_000)))).frame;
  assert.deepEqual(s, { kind: 'Step', width: 4, ns: 80_000_000 });
});

test('kind 7 OpSummary / 11 LayerRun: op は u8 + ns は u64', () => {
  const o = decodeFrame(frame(KIND.OP_SUMMARY, cat(
    varint(2), [3], varint(1000), [200], varint(2 ** 40)
  ))).frame;
  assert.deepEqual(o, { kind: 'OpSummary', ops: [[3, 1000], [200, 2 ** 40]] });

  const l = decodeFrame(frame(KIND.LAYER_RUN, cat(
    varint(5), varint(1), [1], varint(123456)
  ))).frame;
  assert.deepEqual(l, { kind: 'LayerRun', layer: 5, ops: [[1, 123456]] });
});

test('kind 8 KvUsage / 9 Gap', () => {
  const k = decodeFrame(frame(KIND.KV_USAGE, cat(
    varint(2), [1, 0], varint(1024), varint(0)
  ))).frame;
  assert.deepEqual(k, { kind: 'KvUsage', busy: [true, false], t_lens: [1024, 0] });

  const g = decodeFrame(frame(KIND.GAP, varint(3))).frame;
  assert.deepEqual(g, { kind: 'Gap', lost: 3 });
});

test('kind 10 TopK: entropy と entries の確率は f16', () => {
  const b = frame(KIND.TOP_K, cat(
    varint(2), varint(7), h16(0x3e00),           // 1.5
    varint(2), varint(5), h16(0x3800),            // 0.5
    varint(500), h16(0x3400)                      // 0.25
  ));
  const f = decodeFrame(b).frame;
  assert.equal(f.kind, 'TopK');
  assert.equal(f.session, 2);
  assert.equal(f.chosen, 7);
  assert.equal(f.entropy, 1.5);
  assert.deepEqual(f.entries, [[5, 0.5], [500, 0.25]]);
});

test('kind 12 LayerAct: idx のあとに f16 × n', () => {
  const b = frame(KIND.LAYER_ACT, cat(
    varint(47), varint(3), varint(0), varint(128), varint(511),
    h16(0x3c00), h16(0x3800), h16(0x4000)
  ));
  const f = decodeFrame(b).frame;
  assert.deepEqual(f, { kind: 'LayerAct', layer: 47, idx: [0, 128, 511], val: [1, 0.5, 2] });
});

test('kind 13 RoleAct: varint layer + f16 × 5 (id バイトも count も無い)', () => {
  // 実データの形: 層 47、役割の強さは 0.0〜数倍 (ここでは 1.0/0.5/0.25/0.75/0.125)
  const payload = cat(varint(47),
    h16(0x3c00), h16(0x3800), h16(0x3400), h16(0x3a00), h16(0x3000));
  const b = frame(KIND.ROLE_ACT, payload);
  assert.equal(b.length, 6 + 1 + 10, 'RoleAct は 17 B (layer 1 + f16 10)');
  const { frame: f, next } = decodeFrame(b);
  assert.deepEqual(f, {
    kind: 'RoleAct', layer: 47,
    roles: { attn: 1.0, gdn: 0.5, router: 0.25, shex: 0.75, ple: 0.125 }
  });
  assert.equal(next, b.length);
  assert.deepEqual([...ROLE_NAMES], ['attn', 'gdn', 'router', 'shex', 'ple']);
});

test('kind 14 ExpertResidency: varint layer + varint n + varint × n', () => {
  const b = frame(KIND.EXPERT_RESIDENCY, cat(
    varint(4), varint(4), varint(0), varint(127), varint(128), varint(511)
  ));
  const f = decodeFrame(b).frame;
  assert.deepEqual(f, { kind: 'ExpertResidency', layer: 4, vram: [0, 127, 128, 511] });

  // 0 本 (全部 SSD に落ちた層) も読める
  const e = decodeFrame(frame(KIND.EXPERT_RESIDENCY, cat(varint(0), varint(0)))).frame;
  assert.deepEqual(e, { kind: 'ExpertResidency', layer: 0, vram: [] });
});

test('kind 15 TierStats: varint 4 個 (hit は載っていない)', () => {
  const b = frame(KIND.TIER_STATS, cat(
    varint(21000), varint(24576), varint(1200), varint(87)
  ));
  const { frame: f, next } = decodeFrame(b);
  assert.deepEqual(f, {
    kind: 'TierStats', vramMB: 21000, vramTotalMB: 24576, ramExperts: 1200, ssdReads: 87
  });
  assert.equal(next, b.length);
});

test('kind 16 RouterTopK: 実形 (層 47 / 512 本中 top-10) は 48 B', () => {
  const experts = [];
  for (let i = 0; i < 10; i++) experts.push(502 + i);          // 128 以上 = 2B varint
  const bits = [];
  for (let i = 0; i < 10; i++) bits.push(0x3c00 - i * 13);     // lib.rs のテストと同じ重み
  const b = frame(KIND.ROUTER_TOPK, cat(
    varint(47), varint(10),
    ...experts.map((e) => varint(e)),
    ...bits.map((h) => h16(h))
  ));
  assert.equal(b.length, 48, '実形の RouterTopK は 48 B (lib.rs と同じ)');
  const { frame: f, next } = decodeFrame(b);
  assert.equal(f.kind, 'RouterTopK');
  assert.equal(f.layer, 47);
  assert.deepEqual(f.experts, experts);
  assert.equal(f.weights.length, 10);
  for (let i = 0; i < 10; i++) assert.equal(f32ToF16(f.weights[i]), bits[i]);
  assert.equal(next, b.length);
});

// ---- next ポインタ / 連結 ------------------------------------------------

test('decodeFrame の next は次フレームの先頭を指す (2 フレーム連結)', () => {
  const a = frame(KIND.GAP, varint(1));
  const b = frame(KIND.TOKEN, cat(varint(1), varint(2), varint(3)));
  const buf = Uint8Array.from([...a, ...b]);
  const r1 = decodeFrame(buf, 0);
  assert.deepEqual(r1.frame, { kind: 'Gap', lost: 1 });
  assert.equal(r1.next, a.length);
  const r2 = decodeFrame(buf, r1.next);
  assert.deepEqual(r2.frame, { kind: 'Token', session: 1, token: 2, pos: 3 });
  assert.equal(r2.next, buf.length);
  // offset 付きでも同じ (前後に別データがあっても frame 境界で切れる)
  const padded = Uint8Array.from([0xaa, 0xbb, ...buf, 0xcc]);
  const r3 = decodeFrame(padded, 2);
  assert.equal(r3.next, 2 + a.length);
  const r4 = decodeFrame(padded, r3.next);
  assert.equal(r4.next, 2 + buf.length);
});

test('RouterTopK を 3 枚連結しても 3 枚とも読める (合成データ)', () => {
  const mk = (layer) => frame(KIND.ROUTER_TOPK, cat(
    varint(layer), varint(2), varint(500), varint(501), h16(0x3c00), h16(0x3800)
  ));
  const f1 = mk(0), f2 = mk(23), f3 = mk(47);
  const buf = Uint8Array.from([...f1, ...f2, ...f3]);

  const out = decodeFrames(buf);
  assert.equal(out.length, 3);
  assert.deepEqual(out.map((f) => f.layer), [0, 23, 47]);
  for (const f of out) {
    assert.equal(f.kind, 'RouterTopK');
    assert.deepEqual(f.experts, [500, 501]);
    assert.deepEqual(f.weights, [1, 0.5]);
  }

  // decodeFrame を手で回しても同じ位置で 3 枚に割れる
  let off = 0, n = 0;
  while (off < buf.length) {
    const r = decodeFrame(buf, off);
    assert.equal(r.next, off + (n === 2 ? f3.length : f1.length));
    off = r.next; n++;
  }
  assert.equal(n, 3);
});

// ---- 異常系 --------------------------------------------------------------

test('magic が違う → FrameDecodeError (partial=false)', () => {
  const b = frame(KIND.GAP, varint(1));
  b[0] = 0x58;
  assert.throws(() => decodeFrame(b), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.partial, false);
    assert.match(e.message, /bad magic/);
    return true;
  });
});

test('version が違う → FrameDecodeError', () => {
  const b = frame(KIND.GAP, varint(1));
  b[4] = 2;
  assert.throws(() => decodeFrame(b), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.match(e.message, /version/);
    return true;
  });
});

test('未知の kind → FrameDecodeError で kindId が入る', () => {
  const b = frame(0x63, [0x00]);
  assert.throws(() => decodeFrame(b), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.kindId, 0x63);
    assert.equal(e.partial, false);
    return true;
  });
  assert.equal(kindName(0x63), 'kind99');
});

test('短すぎる入力 (< 6B) は partial=true で FrameDecodeError', () => {
  for (const n of [0, 1, 4, 5]) {
    assert.throws(() => decodeFrame(new Uint8Array(n)), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'n=' + n);
      assert.equal(e.partial, true);
      return true;
    });
  }
});

test('どの位置で切れても例外は FrameDecodeError (部分デコードで他例外を投げない)', () => {
  // 全 kind のフレームを作り、6B..len-1 の全切断点で FrameDecodeError になること。
  const samples = [
    frame(KIND.SERVER_INFO, cat(str('m'), str('id'), str('q'), varint(1), varint(2), varint(3))),
    frame(KIND.SESSION_ADMIT, cat(varint(1), varint(2), varint(3))),
    frame(KIND.SESSION_DONE, cat(varint(1), [7], varint(3))),
    frame(KIND.PREFILL, cat(varint(1), varint(2), varint(1e9), varint(2), varint(5), varint(6))),
    frame(KIND.TOKEN, cat(varint(1), varint(2), varint(3))),
    frame(KIND.STEP, cat(varint(1), varint(1e9))),
    frame(KIND.OP_SUMMARY, cat(varint(1), [4], varint(99))),
    frame(KIND.KV_USAGE, cat(varint(1), [1], varint(8))),
    frame(KIND.GAP, varint(1)),
    frame(KIND.TOP_K, cat(varint(1), varint(2), h16(0x3c00), varint(1), varint(3), h16(0x3800))),
    frame(KIND.LAYER_RUN, cat(varint(1), varint(1), [2], varint(9))),
    frame(KIND.LAYER_ACT, cat(varint(1), varint(1), varint(9), h16(0x3c00))),
    frame(KIND.ROLE_ACT, cat(varint(1), h16(0x3c00), h16(0x3c00), h16(0x3c00), h16(0x3c00), h16(0x3c00))),
    frame(KIND.EXPERT_RESIDENCY, cat(varint(1), varint(2), varint(9), varint(500))),
    frame(KIND.TIER_STATS, cat(varint(1), varint(2), varint(3), varint(4))),
    frame(KIND.ROUTER_TOPK, cat(varint(1), varint(2), varint(500), varint(501), h16(0x3c00), h16(0x3800)))
  ];
  for (const full of samples) {
    const id = full[5];
    for (let cut = 6; cut < full.length; cut++) {
      const part = full.slice(0, cut);
      let err = null;
      try { decodeFrame(part); } catch (e) { err = e; }
      assert.ok(err, 'kind ' + id + ' cut=' + cut + ' が例外を投げない');
      assert.ok(err instanceof FrameDecodeError,
        'kind ' + id + ' cut=' + cut + ' が ' + err.constructor.name + ': ' + err.message);
      assert.equal(err.partial, true, 'kind ' + id + ' cut=' + cut + ' は partial');
    }
    // 1 バイト足りないだけでも読めない
    assert.ok(full.length > 6);
  }
});

test('decodeFrames は途中まで読めた分を decoded に載せて投げる', () => {
  const ok = frame(KIND.GAP, varint(3));
  const buf = Uint8Array.from([...ok, 0x00, 0x00, 0x00]);
  assert.throws(() => decodeFrames(buf), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.decoded.length, 1);
    assert.deepEqual(e.decoded[0], { kind: 'Gap', lost: 3 });
    assert.equal(e.offset, ok.length);
    return true;
  });
});

test('空の入力は空の結果', () => {
  assert.deepEqual(decodeFrames(new Uint8Array(0)), []);
});

// ---- f16 ----------------------------------------------------------------

test('f16: ビット列と値の対応 (IEEE 半精度)', () => {
  assert.equal(f16ToF32(0x0000), 0);
  assert.equal(f16ToF32(0x3c00), 1);
  assert.equal(f16ToF32(0x3800), 0.5);
  assert.equal(f16ToF32(0xc000), -2);
  assert.equal(f16ToF32(0x7bff), 65504);
  assert.equal(f16ToF32(0x0001), 2 ** -24);          // 最小の非正規
  assert.ok(Number.isNaN(f16ToF32(0x7e00)));
  assert.equal(f16ToF32(0x7c00), Infinity);
  assert.equal(f16ToF32(0xfc00), -Infinity);
  assert.equal(f32ToF16(1), 0x3c00);
  assert.equal(f32ToF16(-2), 0xc000);
  assert.equal(f32ToF16(2 ** -25), 0);                // RNE で 0
  assert.equal(f32ToF16(2 ** -24), 1);                // 最小の非正規
});

test('f16: 代表値が往復する (相対 1e-3 以内)', () => {
  for (const x of [0, 1, -1, 0.5, 2, 0.1, 65504, -65504, 5.960464477539063e-8]) {
    const back = f16ToF32(f32ToF16(x));
    if (x === 0) assert.equal(back, 0);
    else assert.ok(Math.abs(back - x) <= Math.abs(x) * 1e-3, `${x} -> ${back}`);
  }
});

// ---- エンコーダ (encodeFrame) の整合 -------------------------------------

test('encodeFrame: RoleAct / TierStats / RouterTopK が lib.rs のバイト列と一致する', () => {
  const role = encodeFrame({ kind: 'RoleAct', layer: 47, roles: { attn: 1, gdn: 0.5, router: 0.25, shex: 0.75, ple: 0.125 } });
  assert.deepEqual([...role], [...frame(KIND.ROLE_ACT, cat(
    varint(47), h16(0x3c00), h16(0x3800), h16(0x3400), h16(0x3a00), h16(0x3000)))],
  'RoleAct: id バイトも count も書かない');

  const tier = encodeFrame({ kind: 'TierStats', vramMB: 21000, vramTotalMB: 24576, ramExperts: 1200, ssdReads: 87 });
  assert.deepEqual([...tier], [...frame(KIND.TIER_STATS, cat(
    varint(21000), varint(24576), varint(1200), varint(87)))],
  'TierStats: varint 4 個だけ (hit は書かない)');

  const rk = encodeFrame({ kind: 'RouterTopK', layer: 47, experts: [502, 503], weights: [1, 0.5] });
  assert.deepEqual([...rk], [...frame(KIND.ROUTER_TOPK, cat(
    varint(47), varint(2), varint(502), varint(503), h16(0x3c00), h16(0x3800)))]);
});

test('encodeFrame → decodeFrame の往復 (全 kind)', () => {
  const frames = [
    { kind: 'ServerInfo', model: 'Flash', model_id: 'fn', quant: 'Q4_0', slots: 2, context: 4096, layers: 48 },
    { kind: 'SessionAdmit', session: 1, slot: 0, prompt_tokens: 300 },
    { kind: 'SessionDone', session: 1, reason: 3, output_tokens: 64 },
    { kind: 'Prefill', session: 1, prompt_tokens: 2, ns: 4.1e9, tokens: [1, 70000] },
    { kind: 'Token', session: 1, token: 248046, pos: 17 },
    { kind: 'Step', width: 4, ns: 80e6 },
    { kind: 'OpSummary', ops: [[1, 1000], [200, 2 ** 40]] },
    { kind: 'KvUsage', busy: [true, false, true], t_lens: [1024, 0, 8] },
    { kind: 'Gap', lost: 5 },
    { kind: 'TopK', session: 2, chosen: 7, entropy: 1.5, entries: [[5, 0.5], [500, 0.25]] },
    { kind: 'LayerRun', layer: 5, ops: [[1, 123456]] },
    { kind: 'LayerAct', layer: 12, idx: [0, 1, 2], val: [0.25, 0.5, 1.0] },
    { kind: 'RoleAct', layer: 9, roles: { attn: 1.0, gdn: 0.5, router: 0.25, shex: 0.75, ple: 0.125 } },
    { kind: 'ExpertResidency', layer: 4, vram: [3, 4, 511] },
    { kind: 'TierStats', vramMB: 21000, vramTotalMB: 24576, ramExperts: 1200, ssdReads: 87 },
    { kind: 'RouterTopK', layer: 47, experts: [502, 503, 511], weights: [0.5, 0.25, 0.125] }
  ];
  for (const f of frames) {
    const back = decodeFrames(encodeFrame(f));
    assert.equal(back.length, 1, f.kind);
    assert.equal(back[0].kind, f.kind);
    for (const k of Object.keys(f)) {
      if (k === 'kind') continue;
      assert.deepEqual(back[0][k], f[k], f.kind + '.' + k);
    }
  }
  // 連結しても壊れない
  const all = decodeFrames(concatFrames(frames));
  assert.equal(all.length, frames.length);
  assert.deepEqual(all.map((f) => f.kind), frames.map((f) => f.kind));
});

test('encodeFrame: 未知の kind は例外 (黙って別 kind を書かない)', () => {
  assert.throws(() => encodeFrame({ kind: 'Nope' }), /unknown kind/);
});

test('KIND の番号は lib.rs と一致する', () => {
  assert.deepEqual(KIND, {
    SERVER_INFO: 1, SESSION_ADMIT: 2, SESSION_DONE: 3, PREFILL: 4, TOKEN: 5, STEP: 6,
    OP_SUMMARY: 7, KV_USAGE: 8, GAP: 9, TOP_K: 10, LAYER_RUN: 11, LAYER_ACT: 12,
    ROLE_ACT: 13, EXPERT_RESIDENCY: 14, TIER_STATS: 15, ROUTER_TOPK: 16
  });
  assert.equal(kindName(13), 'ROLE_ACT');
  assert.equal(kindName(16), 'ROUTER_TOPK');
});
