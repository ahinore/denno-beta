// denno-view — ダッシュボード (/) のライブ計器と API 案内の純ロジック (DOM 非依存)。
//
// task 813:
//   1. decode tok/s / prefill tok/s / VRAM expert ヒット率のローリング系列。
//      画面のスパークラインはこの系列から作った SVG path を描くだけにする。
//   2. OpenAI (/v1/chat/completions) と Anthropic (/v1/messages) の
//      エンドポイント URL と、そのまま貼れる curl の雛形。
//
// 画面は数字を作らない。WS のフレームを noteFrame に食わせ、step(now) を
// 回して系列を進め、dashRows の行を描く。値の丸めもここで固定する。

export const DASH_SCHEMA = 1;

/// 系列の並び (画面の行の順でもある)。
export const SERIES_KEYS = Object.freeze(['decode', 'prefill', 'vram']);

export const SERIES_LABELS = Object.freeze({
  decode: 'decode tok/s',
  prefill: 'prefill tok/s',
  vram: 'vram hit',
});

/// decode の窓幅 (ms) と、系列に残すサンプル数 (既定 90 = 90 秒)。
export const DEFAULT_WINDOW_MS = 1000;
export const DEFAULT_CAP = 90;

function numOrNull(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/// 計器の状態。residency は「層 → VRAM に居る expert の集合」。
export function createDash(opts = {}) {
  const windowMs = Math.max(1, Number(opts.windowMs) || DEFAULT_WINDOW_MS);
  const cap = Math.max(2, Math.trunc(Number(opts.cap) || DEFAULT_CAP));
  return {
    schema: DASH_SCHEMA,
    windowMs,
    cap,
    series: { decode: [], prefill: [], vram: [] },
    // decode の窓
    winStart: null,
    winTokens: 0,
    // expert の常駐 (層 → Set<expert id>)
    residency: new Map(),
    // 直近の値
    last: { decode: null, prefill: null, vram: null },
    // TierStats (MB と disk 読みの生値。ヒット率とは別物)
    tier: null,
    prefills: 0,
    tokens: 0,
    hits: 0,
  };
}

function pushSample(d, key, v) {
  const s = d.series[key];
  s.push(v);
  while (s.length > d.cap) s.shift();
  d.last[key] = v;
}

/// Prefill フレーム → prefill tok/s。ns はナノ秒、tokens は実際に食わせた分。
export function notePrefill(d, frame, now) {
  if (!d || !frame) return false;
  const ns = numOrNull(frame.ns);
  const fed = Array.isArray(frame.tokens) && frame.tokens.length
    ? frame.tokens.length
    : numOrNull(frame.prompt_tokens);
  if (ns === null || ns <= 0 || fed === null || fed <= 0) return false;
  pushSample(d, 'prefill', fed / (ns / 1e9));
  d.prefills++;
  return true;
}

/// Token フレーム n 個ぶん。窓は step() が締める (decode tok/s は時間で割る)。
export function noteToken(d, n = 1, now) {
  if (!d) return false;
  const add = Math.max(1, Math.trunc(Number(n) || 1));
  const t = numOrNull(now);
  if (d.winStart === null && t !== null) d.winStart = t;
  d.winTokens += add;
  d.tokens += add;
  return true;
}

/// 窓を締める。窓幅を超えていれば decode tok/s を 1 サンプル積む。
/// 戻り値は積んだかどうか。now が無い/戻った時計では何もしない。
export function step(d, now) {
  if (!d) return false;
  const t = numOrNull(now);
  if (t === null) return false;
  if (d.winStart === null) { d.winStart = t; return false; }
  const dt = t - d.winStart;
  if (dt < d.windowMs) return false;
  pushSample(d, 'decode', d.winTokens * 1000 / dt);
  d.winStart = t;
  d.winTokens = 0;
  return true;
}

/// ExpertResidency フレーム → その層の VRAM 常駐集合を置き換える。
export function noteResidency(d, frame) {
  if (!d || !frame) return false;
  const layer = numOrNull(frame.layer);
  const list = Array.isArray(frame.vram) ? frame.vram : null;
  if (layer === null || list === null) return false;
  d.residency.set(Math.trunc(layer), new Set(list.map((x) => Math.trunc(Number(x)))));
  return true;
}

/// RouterTopK フレーム → 「選ばれた expert のうち VRAM に居た割合」。
/// 常駐の情報がその層に無ければ何もしない (数字を作らない)。
export function noteRoute(d, frame) {
  if (!d || !frame) return false;
  const layer = numOrNull(frame.layer);
  const sel = Array.isArray(frame.experts) ? frame.experts : null;
  if (layer === null || !sel || !sel.length) return false;
  const res = d.residency.get(Math.trunc(layer));
  if (!res) return false;
  let hit = 0;
  for (const e of sel) if (res.has(Math.trunc(Number(e)))) hit++;
  pushSample(d, 'vram', hit / sel.length);
  d.hits++;
  return true;
}

/// TierStats フレーム → VRAM 使用量 (MB) と disk 読み。ヒット率の代用にはしない。
export function noteTierStats(d, frame) {
  if (!d || !frame) return false;
  const used = numOrNull(frame.vramMB);
  const total = numOrNull(frame.vramTotalMB);
  if (used === null || total === null) return false;
  d.tier = {
    usedMB: used,
    totalMB: total,
    fill: total > 0 ? clamp01(used / total) : null,
    ramExperts: numOrNull(frame.ramExperts),
    ssdReads: numOrNull(frame.ssdReads),
  };
  return true;
}

/// フレーム 1 枚を kind で振り分ける。計器に無関係な kind は false。
export function noteFrame(d, frame, now) {
  if (!d || !frame || typeof frame !== 'object') return false;
  switch (frame.kind) {
    case 'Prefill': return notePrefill(d, frame, now);
    case 'Token': return noteToken(d, 1, now);
    case 'ExpertResidency': return noteResidency(d, frame);
    case 'RouterTopK': return noteRoute(d, frame);
    case 'TierStats': return noteTierStats(d, frame);
    default: return false;
  }
}

/// 1 値を表示用の文字列にする。
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

/// 系列 → 点列。min/max を省くとデータから決める (全部同じ値なら真ん中の水平線)。
/// x は右端が最新。サンプル 1 個でも点を返す。
export function sparkPoints(values, opts = {}) {
  const w = Number(opts.w) > 0 ? Number(opts.w) : 220;
  const h = Number(opts.h) > 0 ? Number(opts.h) : 34;
  const pad = Number(opts.pad) >= 0 ? Number(opts.pad) : 2;
  const vals = (Array.isArray(values) ? values : []).filter((v) => Number.isFinite(v));
  if (!vals.length) return [];
  const lo = Number.isFinite(opts.min) ? Number(opts.min) : Math.min(...vals);
  const hi = Number.isFinite(opts.max) ? Number(opts.max) : Math.max(...vals);
  const span = hi - lo;
  const n = vals.length;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const x = n === 1 ? w - pad : pad + i * (w - 2 * pad) / (n - 1);
    const y = span > 0 ? h - pad - ((vals[i] - lo) / span) * (h - 2 * pad) : h / 2;
    pts.push({ x, y });
  }
  return pts;
}

const r2 = (x) => Math.round(x * 100) / 100;

/// 点列 → SVG path ("M x y L x y ...")。空なら空文字。
export function sparkPath(values, opts = {}) {
  const pts = sparkPoints(values, opts);
  if (!pts.length) return '';
  const head = 'M ' + r2(pts[0].x) + ' ' + r2(pts[0].y);
  return pts.slice(1).reduce((s, p) => s + ' L ' + r2(p.x) + ' ' + r2(p.y), head);
}

/// 画面の行 (key / label / 直近の表示値 / path)。画面はこれを描くだけ。
export function dashRows(d, opts = {}) {
  const src = d || createDash();
  return SERIES_KEYS.map((key) => {
    const values = src.series[key] || [];
    const last = values.length ? values[values.length - 1] : null;
    return {
      key,
      label: SERIES_LABELS[key],
      text: key === 'vram' ? fmtPercent(last) : fmtRate(last),
      path: sparkPath(values, opts),
      samples: values.length,
    };
  });
}

// ---- API の案内 -----------------------------------------------------------

/// denno は OpenAI 互換 (/v1/chat/completions) と Anthropic 互換 (/v1/messages)
/// の両方を出す。ビューワ自身の origin をそのまま見せる (ブラウザから叩ける)。
export const API_ENDPOINTS = Object.freeze([
  Object.freeze({ id: 'openai', label: 'OpenAI', path: '/v1/chat/completions' }),
  Object.freeze({ id: 'anthropic', label: 'Anthropic', path: '/v1/messages' }),
]);

function baseUrl(origin) {
  return String(origin == null ? '' : origin).trim().replace(/\/+$/, '');
}

/// エンドポイントの URL 一覧。origin が空なら path だけを返す。
export function apiEndpoints(origin) {
  const base = baseUrl(origin);
  return API_ENDPOINTS.map((e) => ({ id: e.id, label: e.label, path: e.path, url: base + e.path }));
}

/// そのまま貼れる curl。model は ServerInfo の model_id を入れる想定。
export function curlFor(id, origin, model = 'denno') {
  const base = baseUrl(origin);
  const m = String(model == null || model === '' ? 'denno' : model);
  if (id === 'openai') {
    const body = JSON.stringify({
      model: m,
      messages: [{ role: 'user', content: 'hi' }],
      max_tokens: 64,
    });
    return [
      "curl -s " + base + "/v1/chat/completions \\",
      "  -H 'content-type: application/json' \\",
      "  -d '" + body + "'",
    ].join('\n');
  }
  if (id === 'anthropic') {
    const body = JSON.stringify({
      model: m,
      max_tokens: 64,
      messages: [{ role: 'user', content: 'hi' }],
    });
    return [
      "curl -s " + base + "/v1/messages \\",
      "  -H 'content-type: application/json' \\",
      "  -H 'anthropic-version: 2023-06-01' \\",
      "  -d '" + body + "'",
    ].join('\n');
  }
  return '';
}
