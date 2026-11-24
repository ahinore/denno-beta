// denno-view /2d の性能 HUD とプリセットの純ロジック (DOM 非依存)。
//
// task 809:
//   1. ライブの性能 HUD — TTFT / 出力 tok/s / VRAM expert ヒット率 / KV の
//      使い回しトークン数 / MTP (投機) の採用率。
//   2. 介入 + ステアリングのプリセット — いま選択 (checked) しているノード、
//      選択式 (DSL)、off になっているノード、ステアリングの設定を JSON に
//      出し入れする。
//
// 画面は数字を作らない。HUD は「イベント (リクエスト開始 / トークン / 終了 /
// /v1/experts/stats / /v1/cache / 投機フレーム)」をこの状態機械に食わせて、
// 表示用の行を読むだけ。プリセットも検証規則をここにまとめ、画面は
// buildPreset / parsePreset を呼ぶだけにする。
//
// サーバー API の規則 (`/v1/intervene` は層あたり n_expert_used 本を残す、
// `/v1/steer/apply` の layers は [from, to)) は view2d.mjs の実装に合わせる。

import {
  BLOCKS, DEFAULT_SHAPE_RAW, addItem, createSelection, normaliseShape,
  selectionToBody, steerApplyBody,
} from './view2d.mjs';

/// HUD の状態の版。表示の形を変えたら上げる (プリセットとは別)。
export const HUD_SCHEMA = 1;

/// 投機 (MTP / speculative) のテレメトリとみなす最小の kind id。
/// 17 は既知の ExpertSkip なので含めない。18 以降は未知の kind として
/// 届いた時点で「投機のテレメトリが来た」とだけ記録する (payload は読まない)。
export const SPEC_KIND_MIN = 18;

/// 名前で届いた投機フレーム (frames.mjs が将来 kind を足したとき用)。
export const SPEC_FRAME_KINDS = Object.freeze([
  'MtpStats', 'Mtp', 'MTP', 'SpecStats', 'SpecAccept', 'SpecDecode', 'SpeculativeStats',
]);
const SPEC_KIND_SET = new Set(SPEC_FRAME_KINDS);

/// kind (数値 or 名前 or "kind18") → 投機のテレメトリか。
export function isSpecKind(kind) {
  if (typeof kind === 'number') {
    return Number.isInteger(kind) && kind >= SPEC_KIND_MIN;
  }
  const s = String(kind == null ? '' : kind);
  const m = /^kind(\d+)$/.exec(s);
  if (m) return Number(m[1]) >= SPEC_KIND_MIN;
  return SPEC_KIND_SET.has(s);
}

function numOrNull(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

// ---- 1. HUD ---------------------------------------------------------------

/// HUD の状態。opts は使っていないが、将来の窓幅 (移動平均) 用に受ける。
export function createHud(opts = {}) {
  return {
    schema: HUD_SCHEMA,
    window: Math.max(1, Number(opts.window) || 1),
    // いまのリクエスト
    reqActive: false,
    reqStart: null,
    firstTok: null,
    lastTok: null,
    reqTokens: 0,
    ttftMs: null,
    tokPerSec: null,
    lastSecs: null,
    // サーバーの状態 (最後に読んだ値)
    vramHit: null,       // 0..1 (expert の VRAM ヒット率)
    kvReuse: null,       // 次のターンで使い回せる main 文脈のトークン数
    kvEpoch: null,
    // 投機 / MTP
    mtp: {
      seen: false,       // 投機のテレメトリを 1 度でも見た
      frames: 0,
      accepted: 0,
      proposed: 0,
      rate: null,        // 0..1 (累積。accepted / proposed)
      last: null,        // 最後の 1 フレームぶん
      kind: null,        // 最後に見た kind (数値 or 名前)
    },
  };
}

/// 1 リクエストの開始。TTFT の時計をここから回す。
export function hudBeginRequest(hud, now) {
  if (!hud) return false;
  hud.reqActive = true;
  hud.reqStart = Number(now);
  hud.firstTok = null;
  hud.lastTok = null;
  hud.reqTokens = 0;
  hud.ttftMs = null;
  hud.tokPerSec = null;
  hud.lastSecs = null;
  return true;
}

/// トークン (ストリームの差分でもよい) を 1 つ以上食べる。
/// 最初のトークンで TTFT が確定し、2 つ目以降で tok/s が出る。
/// トークン/秒は「最初のトークン → 最後のトークン」の間で数える (TTFT を除く)。
export function hudOnToken(hud, now, n = 1) {
  if (!hud || !hud.reqActive) return false;
  const t = Number(now);
  if (!Number.isFinite(t)) return false;
  const add = Math.max(1, Math.trunc(Number(n) || 1));
  if (hud.firstTok === null) {
    hud.firstTok = t;
    hud.ttftMs = Math.max(0, t - hud.reqStart);
  }
  hud.lastTok = t;
  hud.reqTokens += add;
  if (hud.reqTokens >= 2 && hud.lastTok > hud.firstTok) {
    hud.tokPerSec = (hud.reqTokens - 1) / ((hud.lastTok - hud.firstTok) / 1000);
  }
  return true;
}

/// 1 リクエストの終了。usage の output_tokens があればトークン数をそれに直す
/// (ストリームの差分は usage と一致しないことがある)。
export function hudEndRequest(hud, now, outputTokens) {
  if (!hud || !hud.reqActive) return false;
  const t = Number(now);
  const out = numOrNull(outputTokens);
  if (out !== null && out > 0) hud.reqTokens = Math.trunc(out);
  if (hud.firstTok !== null && hud.lastTok !== null && hud.lastTok > hud.firstTok && hud.reqTokens >= 2) {
    hud.tokPerSec = (hud.reqTokens - 1) / ((hud.lastTok - hud.firstTok) / 1000);
  }
  hud.lastSecs = Number.isFinite(t) ? Math.max(0, t - hud.reqStart) / 1000 : null;
  hud.reqActive = false;
  return true;
}

/// `/v1/experts/stats` を expertStatsView に通した後の値。
export function hudOnStats(hud, statsView) {
  if (!hud) return false;
  const rows = statsView && Array.isArray(statsView.rows) ? statsView.rows : [];
  const hit = rows.length ? numOrNull(rows[0].hit) : null;
  const next = hit === null ? null : clamp01(hit);
  if (next === hud.vramHit) return false;
  hud.vramHit = next;
  return true;
}

/// `/v1/cache` の本文。main のトークン数が「次のターンで使い回す」量。
export function hudOnCache(hud, cacheBody) {
  if (!hud) return false;
  const main = cacheBody && typeof cacheBody === 'object' ? cacheBody.main : null;
  const tokens = main ? numOrNull(main.tokens) : null;
  const epoch = main ? numOrNull(main.epoch) : null;
  if (tokens === hud.kvReuse && epoch === hud.kvEpoch) return false;
  hud.kvReuse = tokens;
  hud.kvEpoch = epoch;
  return true;
}

/// 投機フレーム 1 枚 → {accepted, proposed, rate}。読める数字が無ければ null。
/// 名前は受けた側の実装に合わせて広めに受ける (accepted / accept / spec_accept など)。
export function specSample(frame) {
  const f = frame && typeof frame === 'object' ? frame : {};
  const accepted = numOrNull(
    f.accepted !== undefined ? f.accepted
      : f.accept !== undefined ? f.accept
        : f.spec_accept !== undefined ? f.spec_accept : f.accepted_tokens,
  );
  const proposed = numOrNull(
    f.proposed !== undefined ? f.proposed
      : f.draft !== undefined ? f.draft
        : f.spec_proposed !== undefined ? f.spec_proposed : f.proposed_tokens,
  );
  let rate = numOrNull(
    f.accept_rate !== undefined ? f.accept_rate
      : f.rate !== undefined ? f.rate : f.ratio,
  );
  if (rate !== null && rate > 1 && rate <= 100) rate /= 100;
  if (rate === null && accepted !== null && proposed !== null && proposed > 0) {
    rate = accepted / proposed;
  }
  if (rate === null && accepted === null && proposed === null) return null;
  const out = {
    accepted: accepted === null ? null : Math.max(0, Math.trunc(accepted)),
    proposed: proposed === null ? null : Math.max(0, Math.trunc(proposed)),
    rate: rate === null ? null : clamp01(rate),
  };
  return out;
}

/// 投機フレームを食べる。投機でないフレームは false。
/// accepted / proposed は累積に足し、rate は「累積の採用率」を優先する
/// (フレームが rate だけを運ぶときは、その値をそのまま累積の代わりに使う)。
export function hudOnFrame(hud, frame) {
  if (!hud || !frame || typeof frame !== 'object') return false;
  const kind = frame.kind;
  if (!isSpecKind(kind)) return false;
  const sm = hud.mtp;
  sm.seen = true;
  sm.frames++;
  sm.kind = kind;
  const s = specSample(frame);
  if (s) {
    sm.last = s;
    if (s.accepted !== null) sm.accepted += s.accepted;
    if (s.proposed !== null) sm.proposed += s.proposed;
    if (sm.proposed > 0) sm.rate = clamp01(sm.accepted / sm.proposed);
    else if (s.rate !== null) sm.rate = s.rate;
  }
  return true;
}

/// 未知の kind id (frames.mjs がデコードできない未来の kind) を見たとき。
/// 中身は読めないので「投機のテレメトリが来ている」ことだけ記録する。
export function hudOnRawKind(hud, kindId) {
  if (!hud || !isSpecKind(kindId)) return false;
  hud.mtp.seen = true;
  hud.mtp.frames++;
  hud.mtp.kind = Number(kindId);
  return true;
}

// ---- HUD の表示 -----------------------------------------------------------

export function fmtDuration(ms) {
  const v = numOrNull(ms);
  if (v === null || v < 0) return '-';
  return v < 1000 ? Math.round(v) + ' ms' : (v / 1000).toFixed(2) + ' s';
}

export function fmtRate(v) {
  const n = numOrNull(v);
  if (n === null || n < 0) return '-';
  return n >= 100 ? String(Math.round(n)) : n.toFixed(1);
}

export function fmtPercent(v) {
  const n = numOrNull(v);
  if (n === null) return '-';
  return Math.round(clamp01(n) * 100) + '%';
}

export function fmtTokens(v) {
  const n = numOrNull(v);
  if (n === null) return '-';
  return Math.trunc(n).toLocaleString('en-US');
}

/// HUD の行 (画面はこれをそのまま書く)。key は DOM の id の接尾辞。
export const HUD_LABELS = Object.freeze([
  Object.freeze({ key: 'ttft', label: 'ttft' }),
  Object.freeze({ key: 'rate', label: 'tok/s' }),
  Object.freeze({ key: 'vram', label: 'vram hit' }),
  Object.freeze({ key: 'kv', label: 'kv reuse' }),
  Object.freeze({ key: 'mtp', label: 'mtp accept' }),
]);

export function hudFields(hud) {
  const h = hud || createHud();
  const mtp = h.mtp || {};
  let mtpText = '-';
  if (mtp.seen) {
    mtpText = mtp.rate == null
      ? (mtp.frames > 0 ? 'telemetry' : '-')
      : fmtPercent(mtp.rate) + (mtp.proposed > 0 ? ' (' + mtp.accepted + '/' + mtp.proposed + ')' : '');
  }
  return HUD_LABELS.map((f) => {
    let value;
    if (f.key === 'ttft') value = fmtDuration(h.ttftMs);
    else if (f.key === 'rate') value = fmtRate(h.tokPerSec);
    else if (f.key === 'vram') value = fmtPercent(h.vramHit);
    else if (f.key === 'kv') value = fmtTokens(h.kvReuse);
    else value = mtpText;
    return { key: f.key, label: f.label, value };
  });
}

/// 1 行の要約 (ツールチップ / テスト用)。
export function hudText(hud) {
  return hudFields(hud).map((f) => f.label + ' ' + f.value).join(' · ');
}

// ---- 2. プリセット (介入 + 選択式 + ステアリング) ---------------------------

/// プリセット JSON の版。読む側はこの値が違えば拒否する。
export const PRESET_SCHEMA = 1;
/// ファイルの種類の目印。
export const PRESET_KIND = 'denno-view/preset';

const NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;
const SELECTOR_MAX = 4096;

function pairRows(rows, what) {
  if (rows == null) return [];
  if (!Array.isArray(rows)) throw new Error(what + ' must be an array');
  const out = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (!Array.isArray(r) || r.length !== 2) throw new Error(what + '[' + i + '] must be a pair');
    out.push([r[0], r[1]]);
  }
  return out;
}

/// `/v1/intervene` の本文を検証して正規化する (層 / expert / block / stream の
/// 範囲、層あたり n_expert_used 本の下限、重複除去、決定的な並び)。
/// 不正は例外。プリセットの読み込みと、画面から送る前の検査で同じものを使う。
export function validateInterveneBody(body, shape) {
  const s = normaliseShape(shape || DEFAULT_SHAPE_RAW);
  const b = body && typeof body === 'object' && !Array.isArray(body) ? body : {};
  const experts = pairRows(b.experts_off, 'experts_off');
  const blocks = pairRows(b.blocks_off, 'blocks_off');
  const streams = pairRows(b.streams_off, 'streams_off');
  for (const [l, e] of experts) {
    if (!Number.isInteger(l) || !Number.isInteger(e)) throw new Error('experts_off: entries must be integers');
    if (l < 0 || l >= s.layers) throw new Error('experts_off: layer out of range: ' + l);
    if (e < 0 || e >= s.n_expert) throw new Error('experts_off: expert out of range: ' + e);
  }
  for (const [l, name] of blocks) {
    if (!Number.isInteger(l)) throw new Error('blocks_off: layer must be an integer');
    if (l < 0 || l >= s.layers) throw new Error('blocks_off: layer out of range: ' + l);
    if (BLOCKS.indexOf(String(name)) < 0) throw new Error('blocks_off: unknown block: ' + name);
  }
  for (const [l, x] of streams) {
    if (!Number.isInteger(l) || !Number.isInteger(x)) throw new Error('streams_off: entries must be integers');
    if (l < 0 || l >= s.layers) throw new Error('streams_off: layer out of range: ' + l);
    if (x < 0 || x >= s.n_hc) throw new Error('streams_off: stream out of range: ' + x);
  }
  // 層あたり「残る expert」が top-k 未満だとサーバーが 400 にする。
  const perLayer = new Map();
  for (const [l] of experts) perLayer.set(l, (perLayer.get(l) || 0) + 1);
  if (s.n_expert_used > 0) {
    for (const [l, n] of perLayer) {
      if (s.n_expert - n < s.n_expert_used) {
        throw new Error('layer ' + l + ': remaining experts ' + (s.n_expert - n) +
          ' < n_expert_used ' + s.n_expert_used);
      }
    }
  }
  const uniq = (rows, cmp) => {
    const seen = new Set();
    const out = [];
    for (const r of rows) {
      const k = r.join(':');
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(r);
    }
    return out.sort(cmp);
  };
  return {
    experts_off: uniq(experts, (a, b2) => a[0] - b2[0] || a[1] - b2[1]),
    blocks_off: uniq(blocks, (a, b2) => a[0] - b2[0] || BLOCKS.indexOf(a[1]) - BLOCKS.indexOf(b2[1])),
    streams_off: uniq(streams, (a, b2) => a[0] - b2[0] || a[1] - b2[1]),
  };
}

/// 選択 (item の配列 or 本文) を本文に寄せる。
export function checkedBody(checked) {
  if (checked == null) return { experts_off: [], blocks_off: [], streams_off: [] };
  if (!Array.isArray(checked)) return checked;
  const sel = createSelection();
  for (const it of checked) addItem(sel, it);
  return selectionToBody(sel);
}

function normaliseName(name, dflt = 'preset') {
  const s = String(name == null ? '' : name).trim();
  if (!s) return dflt;
  return s.length > 64 ? s.slice(0, 64) : s;
}

function normaliseSelector(selector) {
  if (selector == null) return '';
  const s = String(selector);
  if (s.length > SELECTOR_MAX) throw new Error('selector: too long (' + s.length + ' > ' + SELECTOR_MAX + ')');
  return s;
}

function normaliseSteer(steer, nLayers) {
  if (steer == null) return null;
  if (typeof steer !== 'object' || Array.isArray(steer)) throw new Error('steer: object or null');
  const terms = Array.isArray(steer.terms) ? steer.terms : [];
  if (!terms.length) return null;
  const body = steerApplyBody({
    terms,
    layers: steer.layers,
    alpha: steer.alpha,
    mode: steer.mode,
  }, { nLayers });
  return body;
}

/// 画面の現在値をプリセットにする (opts.now で created を固定できる)。
export function buildPreset(input, opts = {}) {
  const src = input || {};
  const shape = normaliseShape(src.shape || DEFAULT_SHAPE_RAW);
  const checked = validateInterveneBody(checkedBody(src.checked), shape);
  const intervene = validateInterveneBody(src.intervene || src.disabled || {}, shape);
  const steer = normaliseSteer(src.steer, shape.layers);
  const created = typeof opts.now === 'string' ? opts.now : new Date().toISOString();
  return {
    schema: PRESET_SCHEMA,
    kind: PRESET_KIND,
    name: normaliseName(src.name),
    created,
    shape: {
      layers: shape.layers,
      n_expert: shape.n_expert,
      n_expert_used: shape.n_expert_used,
      n_hc: shape.n_hc,
      n_embd: shape.n_embd,
    },
    selector: normaliseSelector(src.selector),
    checked,
    intervene,
    steer,
    note: src.note == null ? '' : String(src.note).slice(0, 512),
  };
}

/// JSON 文字列 → 検証済みプリセット。不正は例外 (画面は赤字で出す)。
/// 未知の schema は拒否する (将来の版を古い画面が黙って読まないため)。
export function parsePreset(text, opts = {}) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('preset: file is empty');
  let v;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new Error('preset: not JSON: ' + ((e && e.message) ? e.message : String(e)));
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('preset: JSON must be an object');
  if (v.schema !== PRESET_SCHEMA) {
    throw new Error('preset: unsupported schema ' + JSON.stringify(v.schema) + ' (expected ' + PRESET_SCHEMA + ')');
  }
  if (v.kind !== PRESET_KIND) {
    throw new Error('preset: kind must be ' + JSON.stringify(PRESET_KIND));
  }
  if (!v.shape || typeof v.shape !== 'object' || Array.isArray(v.shape)) {
    throw new Error('preset: shape is required');
  }
  const shape = normaliseShape(v.shape);
  const checked = validateInterveneBody(v.checked || {}, shape);
  const intervene = validateInterveneBody(v.intervene || {}, shape);
  const steer = normaliseSteer(v.steer, shape.layers);
  const name = normaliseName(v.name, opts.defaultName || 'preset');
  return {
    schema: PRESET_SCHEMA,
    kind: PRESET_KIND,
    name: NAME_RE.test(name) ? name : 'preset',
    created: typeof v.created === 'string' ? v.created : '',
    shape: {
      layers: shape.layers,
      n_expert: shape.n_expert,
      n_expert_used: shape.n_expert_used,
      n_hc: shape.n_hc,
      n_embd: shape.n_embd,
    },
    selector: normaliseSelector(v.selector),
    checked,
    intervene,
    steer,
    note: v.note == null ? '' : String(v.note).slice(0, 512),
  };
}

/// 1 行の要約 ("checked 3 · off 5e/2b · selector … · steer 2 terms 0-12 project").
export function presetSummary(preset) {
  const p = preset || {};
  const n = (b) => {
    const x = b || {};
    return { e: (x.experts_off || []).length, b: (x.blocks_off || []).length, s: (x.streams_off || []).length };
  };
  const c = n(p.checked);
  const o = n(p.intervene);
  const parts = ['checked ' + (c.e + c.b + c.s) + ' (' + c.e + 'e/' + c.b + 'b/' + c.s + 's)'];
  parts.push('off ' + (o.e + o.b + o.s) + ' (' + o.e + 'e/' + o.b + 'b/' + o.s + 's)');
  const sel = String(p.selector || '').replace(/\s+/g, ' ').trim();
  parts.push('selector ' + (sel ? '"' + (sel.length > 40 ? sel.slice(0, 40) + '…' : sel) + '"' : '-'));
  if (p.steer) {
    parts.push('steer ' + p.steer.terms.length + ' terms ' + p.steer.layers[0] + '-' +
      p.steer.layers[1] + ' ' + p.steer.mode + ' α ' + p.steer.alpha);
  } else {
    parts.push('steer -');
  }
  return parts.join(' · ');
}

/// ダウンロードのファイル名。名前は英数と _ - だけに落とす。
export function presetFileName(name) {
  const raw = String(name == null ? '' : name).trim();
  const safe = raw.replace(/[^A-Za-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 64);
  return 'denno-preset-' + (safe || 'preset') + '.json';
}
