// denno-view — テレメトリのフレームコーデック (v1)。DOM 非依存。
//
// ワイヤ形式の正は crates/denno-telemetry/src/lib.rs。ここはその写しで、
// ブラウザ (index3d.html) と node (tests/frames.test.mjs) の両方から使う。
//
//   magic "DNTL" (4B) | version (1B) | kind (1B) | payload...
//   payload 内の整数は LEB128 (unsigned)、f16 はリトルエンディアンの u16。
//
// kind 1..12 は lib.rs の Frame と 1:1。kind 13..15 (RoleAct / ExpertResidency /
// TierStats) は **lib.rs にまだ定義が無い** ので、このファイルで暫定仕様を決めて
// いる (lib.rs の流儀 = varint + f16 を踏襲)。エンジン側の送出が入ったら
// lib.rs に合わせて直すこと。暫定であることは REPORT_P55.md に書いた。

export const MAGIC = 'DNTL';
export const VERSION = 1;

export const KIND = Object.freeze({
  SERVER_INFO: 1,
  SESSION_ADMIT: 2,
  SESSION_DONE: 3,
  PREFILL: 4,
  TOKEN: 5,
  STEP: 6,
  OP_SUMMARY: 7,
  KV_USAGE: 8,
  GAP: 9,
  TOP_K: 10,
  LAYER_RUN: 11,
  LAYER_ACT: 12,
  ROLE_ACT: 13,          // lib.rs: Frame::RoleAct
  EXPERT_RESIDENCY: 14,  // lib.rs: Frame::ExpertResidency
  TIER_STATS: 15,        // lib.rs: Frame::TierStats
  ROUTER_TOPK: 16        // lib.rs: Frame::RouterTopK
});

const NAME_OF_KIND = Object.create(null);
for (const [k, v] of Object.entries(KIND)) NAME_OF_KIND[v] = k;

/** 役割 id → 名前。RoleAct のペイロードはこの並びの f16 5 個 (role バイトは無い)。 */
export const ROLE_NAMES = Object.freeze(['attn', 'gdn', 'router', 'shex', 'ple']);

/** フレームが壊れていたときに投げる。`.decoded` = そこまでに読めたフレーム列。 */
export class FrameDecodeError extends Error {
  constructor(message, decoded = [], offset = 0, partial = false) {
    super(message);
    this.name = 'FrameDecodeError';
    this.decoded = decoded;
    /** 壊れたフレームの先頭位置 (bytes 内) */
    this.offset = offset;
    /** true = 単にバイトが足りない (次のメッセージに繰り越せる) */
    this.partial = partial;
  }
}

// ---- f16 ---------------------------------------------------------------

const _f32 = new Float32Array(1);
const _u32 = new Uint32Array(_f32.buffer);

/** f16 ビット (u16) → f32。lib.rs の f16_bits_to_f32 と同じ値になる。 */
export function f16ToF32(h) {
  const sign = (h & 0x8000) ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const mant = h & 0x03ff;
  if (exp === 0) return sign * mant * 2 ** -24;            // 非正規 (0 を含む)
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** f32 → f16 ビット (u16)。丸めは round-to-nearest-even (lib.rs と同じ)。 */
export function f32ToF16(x) {
  _f32[0] = x;
  const bits = _u32[0];
  const sign = (bits >>> 16) & 0x8000;
  const exp = ((bits >>> 23) & 0xff) - 127;
  const mant = bits & 0x007fffff;
  if (exp === 128) return mant ? (sign | 0x7e00) : (sign | 0x7c00); // NaN / Inf
  if (exp > 15) return sign | 0x7c00;                               // オーバーフロー
  if (exp < -25) return sign;                                       // 2^-25 未満は 0
  if (exp < -14) {
    // 非正規: 2^-24 刻みに丸める
    const sh = -exp - 1;
    const m = mant | 0x00800000;
    let m16 = m >>> sh;
    const rem = m & ((1 << sh) - 1);
    const half = 1 << (sh - 1);
    if (rem > half || (rem === half && (m16 & 1) === 1)) m16 += 1;
    return sign | m16;
  }
  const half = 1 << 12;
  let m16 = mant >>> 13;
  const rem = mant & 0x1fff;
  if (rem > half || (rem === half && (m16 & 1) === 1)) m16 += 1;
  if (m16 === 0x800) return sign | ((exp + 16) << 10);              // 桁上がり
  return sign | ((exp + 15) << 10) | m16;
}

// ---- 読み書き ------------------------------------------------------------

/** 書き込み用の可変バイト列 (encode 専用)。 */
class Writer {
  constructor() { this.b = []; }
  u8(v) { this.b.push(v & 0xff); }
  u32(v) {
    // LEB128。2^32-1 まで。
    let x = v >>> 0;
    while (x >= 0x80) { this.b.push((x & 0x7f) | 0x80); x >>>= 7; }
    this.b.push(x);
  }
  u64(v) {
    // ns は 2^53 未満を前提 (2^53 ns ≈ 104 日)。
    let x = Number(v);
    while (x >= 0x80) { this.b.push((x % 0x80) | 0x80); x = Math.floor(x / 0x80); }
    this.b.push(x);
  }
  f16(x) { const h = f32ToF16(x); this.b.push(h & 0xff, h >> 8); }
  raw(arr) { for (let i = 0; i < arr.length; i++) this.b.push(arr[i] & 0xff); }
  bytes() { return Uint8Array.from(this.b); }
}

/** 読み出しカーソル。 */
class Reader {
  constructor(bytes, offset) { this.b = bytes; this.p = offset; this.start = offset; }
  need(n) {
    if (this.p + n > this.b.length) {
      // 位置は「フレームの先頭」を返す (呼び手が次のメッセージへ繰り越せるように)
      throw new FrameDecodeError('truncated frame', [], this.start, true);
    }
  }
  u8() { this.need(1); return this.b[this.p++]; }
  u32() {
    let v = 0, shift = 0, n = 0;
    for (;;) {
      this.need(1);
      const b = this.b[this.p++];
      n++;
      v += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) break;
      shift += 7;
      if (shift > 63) throw new FrameDecodeError('varint too long', [], this.start, false);
    }
    if (v > 0xffffffff) throw new FrameDecodeError('varint overflows u32', [], this.start, false);
    return v;
  }
  u64() { return this.u32max53(); }
  u32max53() {
    // ns 用。2^53 までは Number で厳密。
    let v = 0, shift = 0, n = 0;
    for (;;) {
      this.need(1);
      const b = this.b[this.p++];
      n++;
      v += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) break;
      shift += 7;
      if (shift > 63) throw new FrameDecodeError('varint too long', [], this.start, false);
    }
    return v;
  }
  f16() { this.need(2); const h = this.b[this.p] | (this.b[this.p + 1] << 8); this.p += 2; return h; }
  str() {
    const n = this.u32();
    this.need(n);
    const s = new TextDecoder().decode(this.b.subarray(this.p, this.p + n));
    this.p += n;
    return s;
  }
}

// ---- デコード ------------------------------------------------------------

/**
 * 1 フレームを offset から読む。返り値は { frame, next }。
 * 壊れていれば FrameDecodeError を投げる。
 */
export function decodeFrame(bytes, offset = 0) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (buf.length - offset < 6) {
    throw new FrameDecodeError('short frame', [], offset, true);
  }
  if (buf[offset] !== 0x44 || buf[offset + 1] !== 0x4e ||
      buf[offset + 2] !== 0x54 || buf[offset + 3] !== 0x4c) {
    throw new FrameDecodeError('bad magic', [], offset, false);
  }
  const ver = buf[offset + 4];
  if (ver !== VERSION) throw new FrameDecodeError('unsupported version: ' + ver, [], offset, false);
  const kind = buf[offset + 5];
  const r = new Reader(buf, offset + 6);
  r.start = offset;          // エラー位置はフレーム先頭で報告する
  const frame = decodePayload(kind, r);
  const next = r.p;
  return { frame, next };
}

function decodePayload(kind, r) {
  switch (kind) {
    case KIND.SERVER_INFO: {
      const model = r.str(), model_id = r.str(), quant = r.str();
      return { kind: 'ServerInfo', model, model_id, quant, slots: r.u32(), context: r.u32(), layers: r.u32() };
    }
    case KIND.SESSION_ADMIT:
      return { kind: 'SessionAdmit', session: r.u32(), slot: r.u32(), prompt_tokens: r.u32() };
    case KIND.SESSION_DONE:
      return { kind: 'SessionDone', session: r.u32(), reason: r.u8(), output_tokens: r.u32() };
    case KIND.PREFILL: {
      const session = r.u32(), prompt_tokens = r.u32(), ns = r.u64();
      const n = r.u32();
      const tokens = new Array(n);
      for (let i = 0; i < n; i++) tokens[i] = r.u32();
      return { kind: 'Prefill', session, prompt_tokens, ns, tokens };
    }
    case KIND.TOKEN:
      return { kind: 'Token', session: r.u32(), token: r.u32(), pos: r.u32() };
    case KIND.STEP:
      return { kind: 'Step', width: r.u32(), ns: r.u64() };
    case KIND.OP_SUMMARY: {
      const n = r.u32(), ops = new Array(n);
      for (let i = 0; i < n; i++) ops[i] = [r.u8(), r.u64()];
      return { kind: 'OpSummary', ops };
    }
    case KIND.KV_USAGE: {
      const n = r.u32();
      const busy = new Array(n), t_lens = new Array(n);
      for (let i = 0; i < n; i++) busy[i] = r.u8() !== 0;
      for (let i = 0; i < n; i++) t_lens[i] = r.u32();
      return { kind: 'KvUsage', busy, t_lens };
    }
    case KIND.GAP:
      return { kind: 'Gap', lost: r.u32() };
    case KIND.TOP_K: {
      const session = r.u32(), chosen = r.u32(), entropy = f16ToF32(r.f16());
      const n = r.u32(), entries = new Array(n);
      for (let i = 0; i < n; i++) entries[i] = [r.u32(), f16ToF32(r.f16())];
      return { kind: 'TopK', session, chosen, entropy, entries };
    }
    case KIND.LAYER_RUN: {
      const layer = r.u32(), n = r.u32(), ops = new Array(n);
      for (let i = 0; i < n; i++) ops[i] = [r.u8(), r.u64()];
      return { kind: 'LayerRun', layer, ops };
    }
    case KIND.LAYER_ACT: {
      const layer = r.u32(), n = r.u32();
      const idx = new Array(n), val = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = r.u32();
      for (let i = 0; i < n; i++) val[i] = f16ToF32(r.f16());
      return { kind: 'LayerAct', layer, idx, val };
    }
    case KIND.ROLE_ACT: {
      // lib.rs の encode に合わせる: varint layer のあと 5 値の f16 (id バイトは無い)
      const layer = r.u32();
      const roles = {};
      for (const name of ROLE_NAMES) roles[name] = f16ToF32(r.f16());
      return { kind: 'RoleAct', layer, roles };
    }
    case KIND.EXPERT_RESIDENCY: {
      const layer = r.u32(), n = r.u32();
      const idx = new Array(n);
      for (let i = 0; i < n; i++) idx[i] = r.u32();
      return { kind: 'ExpertResidency', layer, vram: idx };
    }
    case KIND.ROUTER_TOPK: {
      const layer = r.u32();
      const n = r.u32();
      const experts = new Array(n);
      for (let i = 0; i < n; i++) experts[i] = r.u32();
      const weights = new Array(n);
      for (let i = 0; i < n; i++) weights[i] = f16ToF32(r.f16());
      return { kind: 'RouterTopK', layer, experts, weights };
    }
    case KIND.TIER_STATS:
      // lib.rs の encode に合わせる: varint 4 個 (hit は載っていない)
      return {
        kind: 'TierStats',
        vramMB: r.u32(), vramTotalMB: r.u32(),
        ramExperts: r.u32(), ssdReads: r.u32()
      };
    default: {
      // 長さが分からないので後続には進めない (lib.rs の decode も未知 kind は Err)。
      const e = new FrameDecodeError('unknown kind: ' + kind, [], r.p - 1, false);
      e.kindId = kind;
      throw e;
    }
  }
}

/**
 * 連結されたフレーム列を全部デコードする。
 * 途中で壊れていたら FrameDecodeError (`.decoded` にそこまで、`.offset` に位置)。
 */
export function decodeFrames(bytes) {
  const buf = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const out = [];
  let p = 0;
  while (p < buf.length) {
    let res;
    try {
      res = decodeFrame(buf, p);
    } catch (e) {
      if (e instanceof FrameDecodeError && e.decoded.length === 0) {
        // 読めた分を載せ替えて投げ直す (kindId など付随情報は保つ)
        e.decoded = out;
        throw e;
      }
      throw e;
    }
    out.push(res.frame);
    p = res.next;
  }
  return out;
}

// ---- エンコード (テストと合成フレーム用) --------------------------------

/**
 * フレーム 1 個を v1 形式で書く。テストの往復と、実機が無いときの
 * 合成フレーム生成に使う。kind は文字列 (例 "LayerAct")。
 */
export function encodeFrame(frame) {
  const w = new Writer();
  w.u8(0x44); w.u8(0x4e); w.u8(0x54); w.u8(0x4c);   // "DNTL"
  w.u8(VERSION);
  switch (frame.kind) {
    case 'ServerInfo':
      w.u8(KIND.SERVER_INFO);
      w.u32(byteLen(frame.model)); w.raw(utf8(frame.model));
      w.u32(byteLen(frame.model_id)); w.raw(utf8(frame.model_id));
      w.u32(byteLen(frame.quant)); w.raw(utf8(frame.quant));
      w.u32(frame.slots); w.u32(frame.context); w.u32(frame.layers);
      break;
    case 'SessionAdmit':
      w.u8(KIND.SESSION_ADMIT); w.u32(frame.session); w.u32(frame.slot); w.u32(frame.prompt_tokens);
      break;
    case 'SessionDone':
      w.u8(KIND.SESSION_DONE); w.u32(frame.session); w.u8(frame.reason); w.u32(frame.output_tokens);
      break;
    case 'Prefill':
      w.u8(KIND.PREFILL);
      w.u32(frame.session); w.u32(frame.prompt_tokens); w.u64(frame.ns);
      w.u32(frame.tokens.length);
      for (const t of frame.tokens) w.u32(t);
      break;
    case 'Token':
      w.u8(KIND.TOKEN); w.u32(frame.session); w.u32(frame.token); w.u32(frame.pos);
      break;
    case 'Step':
      w.u8(KIND.STEP); w.u32(frame.width); w.u64(frame.ns);
      break;
    case 'OpSummary':
      w.u8(KIND.OP_SUMMARY); w.u32(frame.ops.length);
      for (const [op, ns] of frame.ops) { w.u8(op); w.u64(ns); }
      break;
    case 'KvUsage':
      w.u8(KIND.KV_USAGE); w.u32(frame.busy.length);
      for (const b of frame.busy) w.u8(b ? 1 : 0);
      for (const t of frame.t_lens) w.u32(t);
      break;
    case 'Gap':
      w.u8(KIND.GAP); w.u32(frame.lost);
      break;
    case 'TopK':
      w.u8(KIND.TOP_K); w.u32(frame.session); w.u32(frame.chosen); w.f16(frame.entropy);
      w.u32(frame.entries.length);
      for (const [id, p] of frame.entries) { w.u32(id); w.f16(p); }
      break;
    case 'LayerRun':
      w.u8(KIND.LAYER_RUN); w.u32(frame.layer); w.u32(frame.ops.length);
      for (const [op, ns] of frame.ops) { w.u8(op); w.u64(ns); }
      break;
    case 'LayerAct':
      w.u8(KIND.LAYER_ACT); w.u32(frame.layer); w.u32(frame.idx.length);
      for (const i of frame.idx) w.u32(i);
      for (const v of frame.val) w.f16(v);
      break;
    // ---- 暫定 (lib.rs 未定義) ------------------------------------------
    case 'RoleAct': {
      w.u8(KIND.ROLE_ACT); w.u32(frame.layer);
      const entries = Object.entries(frame.roles || {});
      w.u32(entries.length);
      for (const [name, v] of entries) {
        const id = ROLE_NAMES.indexOf(name);
        w.u8(id < 0 ? 0xff : id);
        w.f16(v);
      }
      break;
    }
    case 'ExpertResidency':
      w.u8(KIND.EXPERT_RESIDENCY); w.u32(frame.layer); w.u32(frame.vram.length);
      for (const i of frame.vram) w.u32(i);
      break;
    case 'TierStats':
      w.u8(KIND.TIER_STATS);
      w.u32(frame.vramMB); w.u32(frame.vramTotalMB);
      w.u32(frame.ramExperts); w.u32(frame.ssdReads); w.f16(frame.hit);
      break;
    default:
      throw new Error('encodeFrame: unknown kind ' + frame.kind);
  }
  return w.bytes();
}

function utf8(s) { return new TextEncoder().encode(String(s)); }
function byteLen(s) { return utf8(s).length; }

/** 複数フレームを 1 本のバイト列に連結する (WS 1 メッセージ相当の合成用)。 */
export function concatFrames(frames) {
  const parts = frames.map(encodeFrame);
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** kind バイトの名前 (HUD 表示用)。 */
export function kindName(kindId) {
  return NAME_OF_KIND[kindId] || ('kind' + kindId);
}
