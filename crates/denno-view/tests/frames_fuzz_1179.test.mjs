// denno-view のフレームコーデック (src/frames.mjs) の性質テストとファズ (task 1179)。
// 実行: node --test crates/denno-view/tests/
//
// 見ているもの (全 17 kind):
//   1. encodeFrame → decodeFrame の往復 (合成フレームが完全一致で戻る)
//   2. 連結列: decodeFrames の順序・境界・消費バイト数、decodeFrame の next 連鎖
//   3. offset スイープ: 各フレーム境界 / その内側 / 範囲外での扱い
//   4. 部分バッファ: 全切断点で partial=true になり、読めた前置きだけが decoded に残る
//   5. 500 ケースのファズ (決定論的 PRNG): ランダム連結列の往復と、改変・切断された
//      バイト列で FrameDecodeError 以外の例外を投げないこと
//
// PRNG は xorshift32 で固定シード。実行順やタイミングに依存しない。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  decodeFrame, decodeFrames, encodeFrame, concatFrames,
  f16ToF32, f32ToF16, FrameDecodeError, KIND, ROLE_NAMES, VERSION,
} from '../src/frames.mjs';

const FUZZ_CASES = 500;

// ---- 決定論的 PRNG -------------------------------------------------------

/** xorshift32。同じシードなら同じ列 (テストの再現性のため)。 */
function makeRng(seed) {
  let s = (seed >>> 0) || 0x1179;
  return function next() {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s;
  };
}

const pick = (r, arr) => arr[r() % arr.length];

function arrOf(r, n, pool) {
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = pick(r, pool);
  return out;
}

// ---- 値のプール (境界値を多めに) -----------------------------------------

// f16 で丸めなく往復する値だけを使う (往復比較を厳密にするため)。
const F16_EXACT = [0, 1, -1, 0.5, -0.5, 0.25, -0.25, 0.125, 1.5, 2, 4, 1024, 65504, 2 ** -24];
const U32_POOL = [0, 1, 2, 3, 7, 127, 128, 129, 255, 256, 300, 16383, 16384, 65535, 65536,
  1_000_000, 2 ** 31, 0xffffffff];
const NS_POOL = [0, 1, 127, 128, 16383, 16384, 2 ** 32 - 1, 2 ** 40, 2 ** 53 - 1];
const U8_POOL = [0, 1, 2, 7, 31, 63, 127, 200, 255];
const STR_POOL = ['', 'm', 'q4_0', 'Qwen3.8-Flash-Next', 'flash-next', '日本語モデル'];

// ---- 全 17 kind の合成フレーム生成 ---------------------------------------

function opsOf(r) {
  const n = r() % 3;
  const ops = [];
  for (let i = 0; i < n; i++) ops.push([pick(r, U8_POOL), pick(r, NS_POOL)]);
  return ops;
}

const GEN = {
  ServerInfo: (r) => ({
    kind: 'ServerInfo', model: pick(r, STR_POOL), model_id: pick(r, STR_POOL),
    quant: pick(r, STR_POOL), slots: pick(r, U32_POOL), context: pick(r, U32_POOL),
    layers: pick(r, U32_POOL),
  }),
  SessionAdmit: (r) => ({
    kind: 'SessionAdmit', session: pick(r, U32_POOL), slot: pick(r, U32_POOL),
    prompt_tokens: pick(r, U32_POOL),
  }),
  SessionDone: (r) => ({
    kind: 'SessionDone', session: pick(r, U32_POOL), reason: pick(r, U8_POOL),
    output_tokens: pick(r, U32_POOL),
  }),
  Prefill: (r) => ({
    kind: 'Prefill', session: pick(r, U32_POOL), prompt_tokens: pick(r, U32_POOL),
    ns: pick(r, NS_POOL), tokens: arrOf(r, r() % 4, U32_POOL),
  }),
  Token: (r) => ({
    kind: 'Token', session: pick(r, U32_POOL), token: pick(r, U32_POOL), pos: pick(r, U32_POOL),
  }),
  Step: (r) => ({ kind: 'Step', width: pick(r, U32_POOL), ns: pick(r, NS_POOL) }),
  OpSummary: (r) => ({ kind: 'OpSummary', ops: opsOf(r) }),
  KvUsage: (r) => {
    const n = r() % 4;
    const busy = [], t_lens = [];
    for (let i = 0; i < n; i++) { busy.push((r() & 1) === 1); t_lens.push(pick(r, U32_POOL)); }
    return { kind: 'KvUsage', busy, t_lens };
  },
  Gap: (r) => ({ kind: 'Gap', lost: pick(r, U32_POOL) }),
  TopK: (r) => {
    const n = r() % 3;
    const entries = [];
    for (let i = 0; i < n; i++) entries.push([pick(r, U32_POOL), pick(r, F16_EXACT)]);
    return {
      kind: 'TopK', session: pick(r, U32_POOL), chosen: pick(r, U32_POOL),
      entropy: pick(r, F16_EXACT), entries,
    };
  },
  LayerRun: (r) => ({ kind: 'LayerRun', layer: pick(r, U32_POOL), ops: opsOf(r) }),
  LayerAct: (r) => {
    const n = r() % 4;
    const idx = arrOf(r, n, U32_POOL), val = [];
    for (let i = 0; i < n; i++) val.push(pick(r, F16_EXACT));
    return { kind: 'LayerAct', layer: pick(r, U32_POOL), idx, val };
  },
  RoleAct: (r) => ({
    kind: 'RoleAct', layer: pick(r, U32_POOL),
    roles: {
      attn: pick(r, F16_EXACT), gdn: pick(r, F16_EXACT), router: pick(r, F16_EXACT),
      shex: pick(r, F16_EXACT), ple: pick(r, F16_EXACT),
    },
  }),
  ExpertResidency: (r) => ({
    kind: 'ExpertResidency', layer: pick(r, U32_POOL), vram: arrOf(r, r() % 4, U32_POOL),
  }),
  TierStats: (r) => ({
    kind: 'TierStats', vramMB: pick(r, U32_POOL), vramTotalMB: pick(r, U32_POOL),
    ramExperts: pick(r, U32_POOL), ssdReads: pick(r, U32_POOL),
  }),
  RouterTopK: (r) => {
    const n = r() % 4;
    const experts = arrOf(r, n, U32_POOL), weights = [];
    for (let i = 0; i < n; i++) weights.push(pick(r, F16_EXACT));
    return { kind: 'RouterTopK', layer: pick(r, U32_POOL), experts, weights };
  },
  ExpertSkip: (r) => ({
    kind: 'ExpertSkip', layer: pick(r, U32_POOL), experts: arrOf(r, r() % 4, U32_POOL),
  }),
};

const GEN_ORDER = [
  'ServerInfo', 'SessionAdmit', 'SessionDone', 'Prefill', 'Token', 'Step', 'OpSummary',
  'KvUsage', 'Gap', 'TopK', 'LayerRun', 'LayerAct', 'RoleAct', 'ExpertResidency',
  'TierStats', 'RouterTopK', 'ExpertSkip',
];

/** kind 名 → KIND 番号 (encodeFrame が書く kind バイトの期待値)。 */
const KIND_OF = {
  ServerInfo: KIND.SERVER_INFO, SessionAdmit: KIND.SESSION_ADMIT,
  SessionDone: KIND.SESSION_DONE, Prefill: KIND.PREFILL, Token: KIND.TOKEN,
  Step: KIND.STEP, OpSummary: KIND.OP_SUMMARY, KvUsage: KIND.KV_USAGE, Gap: KIND.GAP,
  TopK: KIND.TOP_K, LayerRun: KIND.LAYER_RUN, LayerAct: KIND.LAYER_ACT,
  RoleAct: KIND.ROLE_ACT, ExpertResidency: KIND.EXPERT_RESIDENCY,
  TierStats: KIND.TIER_STATS, RouterTopK: KIND.ROUTER_TOPK, ExpertSkip: KIND.EXPERT_SKIP,
};

const randomFrame = (r) => GEN[pick(r, GEN_ORDER)](r);

/** 全 17 kind を 1 枚ずつ (固定シード) 並べた連結列と、そのフレーム境界。 */
function capture(seed = 0x1179) {
  const r = makeRng(seed);
  const frames = GEN_ORDER.map((name) => GEN[name](r));
  const parts = frames.map(encodeFrame);
  const bounds = [0];
  for (const p of parts) bounds.push(bounds[bounds.length - 1] + p.length);
  return { frames, parts, bounds, buf: concatFrames(frames) };
}

/** 先頭から読み進め、末尾到達か例外で止まる。例外は握らずに返す。 */
function streamDecode(buf) {
  const frames = [];
  let p = 0;
  for (;;) {
    if (p >= buf.length) return { frames, p, err: null };
    let res;
    try {
      res = decodeFrame(buf, p);
    } catch (e) {
      return { frames, p, err: e };
    }
    assert.ok(res.next > p, 'next が進まない (p=' + p + ')');
    frames.push(res.frame);
    p = res.next;
  }
}

// ---- 1. 前提と往復 -------------------------------------------------------

test('前提: F16_EXACT の値は f32 -> f16 -> f32 で丸めなく往復する', () => {
  for (const v of F16_EXACT) {
    assert.equal(f16ToF32(f32ToF16(v)), v, 'value=' + String(v));
    assert.equal(f32ToF16(f16ToF32(f32ToF16(v))), f32ToF16(v), 'bits=' + String(v));
  }
});

test('property: 17 kind すべて encode -> decode が完全一致する (各 kind 20 例)', () => {
  const r = makeRng(0x1179);
  let cases = 0;
  for (const name of GEN_ORDER) {
    for (let i = 0; i < 20; i++) {
      const f = GEN[name](r);
      const b = encodeFrame(f);
      // ヘッダは magic + version + kind
      assert.deepEqual([...b.subarray(0, 4)], [0x44, 0x4e, 0x54, 0x4c], name);
      assert.equal(b[4], VERSION, name);
      assert.equal(b[5], KIND_OF[name], name + ' の kind バイト');
      const out = decodeFrames(b);
      assert.equal(out.length, 1, name + ' #' + i);
      assert.deepEqual(out[0], f, name + ' #' + i);
      // decodeFrame 単体でも同じで、next は末尾
      const one = decodeFrame(b);
      assert.deepEqual(one.frame, f, name + ' #' + i);
      assert.equal(one.next, b.length, name + ' #' + i);
      cases++;
    }
  }
  assert.equal(cases, GEN_ORDER.length * 20);
  assert.equal(Object.keys(GEN).length, 17);
  // 全 kind の番号が 1..17 をちょうど覆う
  assert.deepEqual([...new Set(Object.values(KIND_OF))].sort((a, b) => a - b),
    Array.from({ length: 17 }, (_, i) => i + 1));
});

test('property: 連結列 (1..6 枚 × 40 通り) は順序・境界・消費バイト数が保存される', () => {
  const r = makeRng(0x5eed);
  for (let c = 0; c < 40; c++) {
    const n = 1 + r() % 6;
    const frames = [];
    for (let i = 0; i < n; i++) frames.push(randomFrame(r));
    const parts = frames.map(encodeFrame);
    const buf = concatFrames(frames);
    const total = parts.reduce((a, p) => a + p.length, 0);
    assert.equal(buf.length, total, 'case ' + c);
    // 手組みの連結とも一致する (concatFrames のオフセット計算)
    const manual = Uint8Array.from(parts.flatMap((p) => [...p]));
    assert.deepEqual([...buf], [...manual], 'case ' + c);

    assert.deepEqual(decodeFrames(buf), frames, 'case ' + c);
    assert.deepEqual(decodeFrames(manual), frames, 'case ' + c);

    let off = 0;
    for (let i = 0; i < n; i++) {
      const res = decodeFrame(buf, off);
      assert.deepEqual(res.frame, frames[i], 'case ' + c + ' #' + i);
      assert.equal(res.next, off + parts[i].length, 'case ' + c + ' #' + i);
      off = res.next;
    }
    assert.equal(off, buf.length, 'case ' + c + ': 消費バイト数が入力長と一致');
  }
});

// ---- 2. offset スイープ --------------------------------------------------

test('offset スイープ: 境界は一致し、内側は magic 不一致なら bad magic、範囲外は partial', () => {
  const { frames, bounds, buf } = capture();
  const isMagicAt = (o) => buf[o] === 0x44 && buf[o + 1] === 0x4e &&
    buf[o + 2] === 0x54 && buf[o + 3] === 0x4c;

  // 各境界では必ずそのフレームが読める
  for (let i = 0; i < frames.length; i++) {
    const res = decodeFrame(buf, bounds[i]);
    assert.deepEqual(res.frame, frames[i], 'boundary ' + i);
    assert.equal(res.next, bounds[i + 1], 'boundary ' + i);
  }

  let interior = 0, skipped = 0;
  for (let i = 0; i < frames.length; i++) {
    for (let off = bounds[i] + 1; off < bounds[i + 1]; off++) {
      interior++;
      const remain = buf.length - off;
      if (remain < 6) {
        assert.throws(() => decodeFrame(buf, off), (e) => {
          assert.ok(e instanceof FrameDecodeError, 'off=' + off);
          assert.equal(e.partial, true, 'off=' + off);
          assert.equal(e.offset, off, 'off=' + off);
          return true;
        });
        continue;
      }
      if (isMagicAt(off)) {
        // 偶然 magic と一致した内側 offset: 例外か、前進する結果のどちらかでよい
        skipped++;
        try {
          const res = decodeFrame(buf, off);
          assert.ok(res.next > off && res.next <= buf.length, 'off=' + off);
        } catch (e) {
          assert.ok(e instanceof FrameDecodeError, 'off=' + off + ': ' + e);
        }
        continue;
      }
      assert.throws(() => decodeFrame(buf, off), (e) => {
        assert.ok(e instanceof FrameDecodeError, 'off=' + off);
        assert.equal(e.partial, false, 'off=' + off + ' は partial ではない: ' + e.message);
        assert.equal(e.offset, off, 'エラー位置はフレーム先頭');
        assert.match(e.message, /bad magic|version|unknown kind|corrupt/, 'off=' + off);
        return true;
      });
    }
  }
  assert.ok(interior > frames.length * 4, '内側の offset を十分に試した (interior=' + interior + ')');
  assert.ok(interior - skipped > 50, 'bad magic を見るケースが十分ある');

  // 範囲外 (末尾ちょうども含む) は「バイト不足」として partial
  for (const off of [buf.length, buf.length + 1, buf.length + 99]) {
    assert.throws(() => decodeFrame(buf, off), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'off=' + off);
      assert.equal(e.partial, true, 'off=' + off);
      assert.equal(e.offset, off, 'off=' + off);
      return true;
    });
  }
});

// ---- 3. 部分バッファ -----------------------------------------------------

test('部分バッファ: 全切断点で partial=true、decoded に読めた前置きだけが残る', () => {
  const { frames, bounds, buf } = capture();
  for (let cut = 0; cut < buf.length; cut++) {
    const part = buf.subarray(0, cut);
    const { frames: got, p, err } = streamDecode(part);

    // 切断点までに完全に収まるフレーム数
    let complete = 0;
    while (complete + 1 < bounds.length && bounds[complete + 1] <= cut) complete++;
    assert.equal(got.length, complete, 'cut=' + cut);
    assert.deepEqual(got, frames.slice(0, complete), 'cut=' + cut);

    const atBoundary = bounds[complete] === cut;
    if (atBoundary) {
      assert.equal(err, null, 'cut=' + cut + ' は境界なので例外なし');
      assert.equal(p, cut, 'cut=' + cut);
      assert.deepEqual(decodeFrames(part), frames.slice(0, complete), 'cut=' + cut);
      continue;
    }

    assert.ok(err instanceof FrameDecodeError, 'cut=' + cut + ': ' + err);
    assert.equal(err.partial, true, 'cut=' + cut + ': ' + err.message);
    assert.equal(p, bounds[complete], 'cut=' + cut + ': 欠けたフレームは消費しない');
    assert.equal(err.offset, p, 'cut=' + cut);

    // decodeFrames も同じ前置きを decoded に載せて投げる
    assert.throws(() => decodeFrames(part), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'cut=' + cut);
      assert.equal(e.partial, true, 'cut=' + cut);
      assert.equal(e.offset, bounds[complete], 'cut=' + cut);
      assert.deepEqual(e.decoded, frames.slice(0, complete), 'cut=' + cut);
      return true;
    });
  }
});

// ---- 4. 改変なしファズ 500 ケース ---------------------------------------

test('fuzz 500 ケース: ランダム連結列の roundtrip と境界再現', () => {
  const r = makeRng(0x1179f00d);
  let framesTotal = 0;
  for (let c = 0; c < FUZZ_CASES; c++) {
    const n = 1 + r() % 6;
    const frames = [];
    for (let i = 0; i < n; i++) frames.push(randomFrame(r));
    framesTotal += n;

    const parts = frames.map(encodeFrame);
    const buf = concatFrames(frames);
    assert.equal(buf.length, parts.reduce((a, p) => a + p.length, 0), 'case ' + c);

    assert.deepEqual(decodeFrames(buf), frames, 'case ' + c);

    let off = 0;
    for (let i = 0; i < n; i++) {
      const res = decodeFrame(buf, off);
      assert.deepEqual(res.frame, frames[i], 'case ' + c + ' #' + i);
      assert.equal(res.next, off + parts[i].length, 'case ' + c + ' #' + i);
      off = res.next;
    }
    assert.equal(off, buf.length, 'case ' + c);

    // 末尾から折り返す (リング再生) と同じ列が読める
    const lastStart = off - parts[parts.length - 1].length;
    const tail = decodeFrame(buf, lastStart);
    assert.deepEqual(tail.frame, frames[frames.length - 1], 'case ' + c);
    assert.equal(tail.next, buf.length, 'case ' + c);

    // 途中の境界から読んでも残りが同じ順で読める
    if (n > 1) {
      const mid = parts[0].length;
      assert.deepEqual(decodeFrames(buf.subarray(mid)), frames.slice(1), 'case ' + c);
    }
  }
  assert.ok(framesTotal > 1000, '合計フレーム数 (framesTotal=' + framesTotal + ')');
});

// ---- 5. 改変ファズ 500 ケース -------------------------------------------

test('fuzz 500 ケース: 改変・切断されたバイト列で FrameDecodeError 以外を投げない', () => {
  const r = makeRng(0xbadf00d);
  const KNOWN = new Set(GEN_ORDER);
  let errors = 0, ok = 0;

  for (let c = 0; c < FUZZ_CASES; c++) {
    const baseCount = 1 + r() % 4;
    const base = concatFrames(Array.from({ length: baseCount }, () => randomFrame(r)));
    let mut = Uint8Array.from(base);
    const op = r() % 4;
    if (op === 0) {
      mut = mut.slice(0, r() % (mut.length + 1));                       // 切断
    } else if (op === 1) {
      const at = r() % mut.length;
      mut[at] ^= 1 << (r() % 8);                                        // 1 ビット反転
    } else if (op === 2) {
      mut[r() % mut.length] = r() & 0xff;                               // バイト置換
    } else {
      const at = r() % (mut.length + 1), k = 1 + r() % 4;                // バイト挿入
      const ins = new Uint8Array(mut.length + k);
      ins.set(mut.subarray(0, at));
      for (let j = 0; j < k; j++) ins[at + j] = r() & 0xff;
      ins.set(mut.subarray(at), at + k);
      mut = ins;
    }

    const out = streamDecode(mut);
    for (const f of out.frames) {
      assert.ok(KNOWN.has(f.kind), 'case ' + c + ': 未知の kind 名 ' + f.kind);
    }
    if (out.err) {
      assert.ok(out.err instanceof FrameDecodeError,
        'case ' + c + ': ' + (out.err && out.err.constructor && out.err.constructor.name) +
        ' ' + (out.err && out.err.message));
      assert.equal(typeof out.err.partial, 'boolean', 'case ' + c);
      assert.ok(Number.isInteger(out.err.offset) && out.err.offset >= 0, 'case ' + c);
      errors++;
    } else {
      ok++;
    }

    // decodeFrames は同じ列 / 同じ前置きを報告する
    let df = null, derr = null;
    try { df = decodeFrames(mut); } catch (e) { derr = e; }
    if (derr) {
      assert.ok(derr instanceof FrameDecodeError, 'case ' + c);
      assert.deepEqual(derr.decoded, out.frames, 'case ' + c);
      assert.equal(derr.offset, out.p, 'case ' + c);
    } else {
      assert.deepEqual(df, out.frames, 'case ' + c);
    }
  }
  assert.equal(ok + errors, FUZZ_CASES);
  assert.ok(errors > 50, '壊れるケースが十分ある (errors=' + errors + ')');
  assert.ok(ok > 10, '壊れないケースもある (ok=' + ok + ')');
});

// ---- 6. offset / 数値 / kind の境界 -------------------------------------

test('offset の不正値は bad offset (partial=false)、大きな整数は partial に落ちる', () => {
  const b = encodeFrame({ kind: 'Gap', lost: 5 });
  for (const bad of [-1, -100, 1.5, 0.5, NaN, Infinity, -Infinity, 'x', 2 ** 40 + 0.25]) {
    assert.throws(() => decodeFrame(b, bad), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'offset=' + String(bad));
      assert.equal(e.partial, false, 'offset=' + String(bad));
      assert.equal(e.offset, 0, 'offset=' + String(bad));
      assert.match(e.message, /bad offset/, 'offset=' + String(bad));
      return true;
    });
  }
  for (const off of [b.length, b.length + 1, b.length + 4096, 1e21]) {
    assert.throws(() => decodeFrame(b, off), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'offset=' + String(off));
      assert.equal(e.partial, true, 'offset=' + String(off));
      assert.equal(e.offset, off, 'offset=' + String(off));
      return true;
    });
  }
  // undefined / null / 数値文字列は数値化して扱う
  assert.equal(decodeFrame(b).frame.lost, 5);
  assert.equal(decodeFrame(b, undefined).frame.lost, 5);
  assert.equal(decodeFrame(b, null).frame.lost, 5);
  assert.equal(decodeFrame(b, '0').frame.lost, 5);
  assert.throws(() => decodeFrame(b, '1'), /bad magic/);
});

test('varint の境界: u32 超過 / 2^53 超過 / 継続ビット過多は partial=false', () => {
  const head = (kind) => [0x44, 0x4e, 0x54, 0x4c, VERSION, kind];

  // 2^32 = u32 の上限 (0xffffffff) 超え
  assert.throws(() => decodeFrame(Uint8Array.from([...head(KIND.GAP), 0x80, 0x80, 0x80, 0x80, 0x10])), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.partial, false);
    assert.match(e.message, /overflows u32/);
    assert.equal(e.offset, 0);
    return true;
  });

  // 2^53 = ns の上限 (2^53-1) 超え
  const big = (n) => {
    const out = [];
    let x = n;
    while (x >= 0x80n) { out.push(Number(x & 0x7fn) | 0x80); x >>= 7n; }
    out.push(Number(x));
    return out;
  };
  assert.throws(() => decodeFrame(Uint8Array.from([...head(KIND.STEP), 0x01, ...big(2n ** 53n)])), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.partial, false);
    assert.match(e.message, /safe integer/);
    return true;
  });

  // 継続ビットが 10 バイト以上続く
  const long = new Uint8Array(6 + 12).fill(0x80);
  long.set(head(KIND.GAP));
  assert.throws(() => decodeFrame(long), (e) => {
    assert.ok(e instanceof FrameDecodeError);
    assert.equal(e.partial, false);
    assert.match(e.message, /varint too long/);
    return true;
  });

  // 境界の内側 (2^53-1) は往復する
  const okb = encodeFrame({ kind: 'Step', width: 0xffffffff, ns: 2 ** 53 - 1 });
  assert.deepEqual(decodeFrame(okb).frame, { kind: 'Step', width: 0xffffffff, ns: 2 ** 53 - 1 });
});

test('未知の kind (0 / 18 / 255) は kindId 付きで弾き、連結列では decoded を保つ', () => {
  const good = encodeFrame({ kind: 'Gap', lost: 1 });
  for (const k of [0, 18, 255]) {
    const b = Uint8Array.from([0x44, 0x4e, 0x54, 0x4c, VERSION, k, 0x00]);
    assert.throws(() => decodeFrame(b), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'kind=' + k);
      assert.equal(e.kindId, k, 'kind=' + k);
      assert.equal(e.partial, false, 'kind=' + k);
      assert.equal(e.offset, 0, 'kind=' + k);
      assert.match(e.message, /unknown kind/, 'kind=' + k);
      return true;
    });
    const buf = Uint8Array.from([...good, ...b]);
    assert.throws(() => decodeFrames(buf), (e) => {
      assert.ok(e instanceof FrameDecodeError, 'kind=' + k);
      assert.equal(e.offset, good.length, 'kind=' + k + ': offset はそのフレーム先頭');
      assert.equal(e.decoded.length, 1, 'kind=' + k);
      assert.deepEqual(e.decoded[0], { kind: 'Gap', lost: 1 }, 'kind=' + k);
      return true;
    });
  }
});

test('入力の型: 素の配列 / Buffer / subarray ビューでも同じ結果 (ROLE_NAMES も 5 個)', () => {
  const { frames, buf } = capture();
  assert.deepEqual(decodeFrames(Array.from(buf)), frames);
  assert.deepEqual(decodeFrames(Buffer.from(buf)), frames);

  const padded = Buffer.concat([Buffer.from([1, 2, 3]), Buffer.from(buf), Buffer.from([9])]);
  assert.deepEqual(decodeFrames(padded.subarray(3, 3 + buf.length)), frames);
  assert.deepEqual(decodeFrame(padded, 3).frame, frames[0]);
  // decodeFrame にビューを渡してもオフセットはビュー先頭から数える
  assert.deepEqual(decodeFrame(padded.subarray(3)).frame, frames[0]);

  assert.deepEqual([...ROLE_NAMES], ['attn', 'gdn', 'router', 'shex', 'ple']);
});
