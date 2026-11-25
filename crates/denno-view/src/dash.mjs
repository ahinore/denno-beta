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

/// スパークライン/リングバッファの保持数 (task 909: 120 サンプル)。
export const SERIES_CAP = 120;

/// リング/系列の上限 (cap に Infinity や巨大値を渡されても青天井にしない)。
export const MAX_RING_CAP = 65536;

/// 常駐表に残す層の上限 (壊れた層 id の連投で Map が伸び続けないように)。
export const MAX_LAYERS = 512;

/// 計器の棒 (KV / VRAM) の既定の幅 (文字セル数)。
export const GAUGE_WIDTH = 10;

/// 数値化。数値と数値文字列だけを受ける。NaN / Infinity / 空白だけの
/// 文字列 / bool / 配列 / オブジェクトは null (暗黙の型変換で数字を作らない)。
/// テレメトリの生値はここを通してから系列に入れる。
function numOrNull(v) {
  if (typeof v === 'number') return Number.isFinite(v) ? v : null;
  if (typeof v === 'string') {
    const s = v.trim();
    if (!s) return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/// cap を [2, MAX_RING_CAP] に正規化する。0 / 非数 / 未指定は既定値。
function normalizeCap(v, dflt) {
  const n = numOrNull(v);
  if (!n) return dflt;
  return Math.min(MAX_RING_CAP, Math.max(2, Math.trunc(n)));
}

/// リングバッファ (task 909)。古い値から落ちる固定長の系列。
/// 値は有限かつ非負の数だけを受ける (NaN / Infinity / 負値は捨てる)。
export function createRing(cap = SERIES_CAP) {
  return { cap: normalizeCap(cap, SERIES_CAP), values: [] };
}

export function ringPush(ring, v) {
  if (!ring || !Array.isArray(ring.values)) return false;
  // 手で組み立てたリング (cap 欠け / Infinity) でも暴走しないよう正規化する。
  const c = normalizeCap(ring.cap, SERIES_CAP);
  if (ring.cap !== c) ring.cap = c;
  const n = numOrNull(v);
  if (n === null || n < 0) return false;
  ring.values.push(n);
  while (ring.values.length > c) ring.values.shift();
  return true;
}

export function ringLast(ring) {
  if (!ring || !Array.isArray(ring.values) || !ring.values.length) return null;
  return ring.values[ring.values.length - 1];
}

/// 計器の状態。residency は「層 → VRAM に居る expert の集合」。
export function createDash(opts = {}) {
  const o = opts && typeof opts === 'object' ? opts : {};
  const wm = numOrNull(o.windowMs);
  const windowMs = wm !== null && wm > 0 ? wm : DEFAULT_WINDOW_MS;
  const cap = normalizeCap(o.cap, DEFAULT_CAP);
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
    // task 909: decode の tok/s と step の所要時間 (ms) を 120 本のリングに持つ。
    decodeRing: createRing(SERIES_CAP),
    latencyRing: createRing(SERIES_CAP),
    // KV スロットの使用状況 (KvUsage フレームの集計)
    kv: { busy: null, slots: null },
  };
}

function pushSample(d, key, v) {
  const s = d.series[key];
  if (!Array.isArray(s)) return false;
  // 系列は「率」なので非負の有限数だけ。壊れた値で last を汚さない。
  const n = numOrNull(v);
  if (n === null || n < 0) return false;
  const cap = normalizeCap(d.cap, DEFAULT_CAP);
  s.push(n);
  while (s.length > cap) s.shift();
  d.last[key] = n;
  return true;
}

/// Prefill フレーム → prefill tok/s。ns はナノ秒、tokens は実際に食わせた分。
export function notePrefill(d, frame, now) {
  if (!d || !frame) return false;
  const ns = numOrNull(frame.ns);
  const fed = Array.isArray(frame.tokens) && frame.tokens.length
    ? frame.tokens.length
    : numOrNull(frame.prompt_tokens);
  if (ns === null || ns <= 0 || fed === null || fed <= 0) return false;
  const rate = fed / (ns / 1e9);
  if (!Number.isFinite(rate)) return false;        // 桁あふれは捨てる
  if (!pushSample(d, 'prefill', rate)) return false;
  d.prefills++;
  return true;
}

/// Token フレーム n 個ぶん。窓は step() が締める (decode tok/s は時間で割る)。
export function noteToken(d, n = 1, now) {
  if (!d) return false;
  const raw = n === undefined ? 1 : numOrNull(n);
  const add = raw === null ? NaN : Math.trunc(raw);
  if (!Number.isSafeInteger(add) || add < 1) return false;
  const t = numOrNull(now);
  if (d.winStart === null && t !== null) d.winStart = t;
  const prevWin = numOrNull(d.winTokens);
  const prevAll = numOrNull(d.tokens);
  d.winTokens = (prevWin === null || prevWin < 0 ? 0 : prevWin) + add;
  d.tokens = (prevAll === null || prevAll < 0 ? 0 : prevAll) + add;
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
  if (!(dt >= d.windowMs)) return false;           // 戻った時計 / NaN は据え置き
  const wt = numOrNull(d.winTokens);
  if (wt === null) {
    // 窓の累積値が壊れている (NaN / Infinity)。数字を作らず窓だけ畳む。
    d.winStart = t;
    d.winTokens = 0;
    return false;
  }
  const rate = (wt < 0 ? 0 : wt) * 1000 / dt;
  if (!Number.isFinite(rate) || rate < 0) return false;
  pushSample(d, 'decode', rate);
  if (d.decodeRing) ringPush(d.decodeRing, rate);
  d.winStart = t;
  d.winTokens = 0;
  return true;
}

/// ExpertResidency フレーム → その層の VRAM 常駐集合を置き換える。
/// expert id は非負の整数だけを受ける (NaN / Infinity / 負値は落とす)。
export function noteResidency(d, frame) {
  if (!d || !frame) return false;
  const layer = numOrNull(frame.layer);
  const list = Array.isArray(frame.vram) ? frame.vram : null;
  if (layer === null || layer < 0 || list === null) return false;
  const ids = new Set();
  for (const x of list) {
    const id = numOrNull(x);
    if (id !== null && id >= 0) ids.add(Math.trunc(id));
  }
  // 層の数にも上限を置く (壊れた層 id で Map が伸び続けないように)。
  // 既存の層を更新するときは追い出さない。新しい層で満杯なら最古を落とす。
  if (d.residency.size >= MAX_LAYERS && !d.residency.has(Math.trunc(layer))) {
    const oldest = d.residency.keys().next().value;
    if (oldest !== undefined) d.residency.delete(oldest);
  }
  d.residency.set(Math.trunc(layer), ids);
  return true;
}

/// RouterTopK フレーム → 「選ばれた expert のうち VRAM に居た割合」。
/// 常駐の情報がその層に無ければ何もしない (数字を作らない)。
export function noteRoute(d, frame) {
  if (!d || !frame) return false;
  const layer = numOrNull(frame.layer);
  const sel = Array.isArray(frame.experts) ? frame.experts : null;
  if (layer === null || layer < 0 || !sel || !sel.length) return false;
  const res = d.residency.get(Math.trunc(layer));
  if (!res) return false;
  let hit = 0, count = 0;
  for (const e of sel) {
    const id = numOrNull(e);
    if (id === null || id < 0) continue;           // 壊れた id は分母からも外す
    count++;
    if (res.has(Math.trunc(id))) hit++;
  }
  if (!count) return false;
  if (!pushSample(d, 'vram', hit / count)) return false;
  d.hits++;
  return true;
}

/// TierStats フレーム → VRAM 使用量 (MB) と disk 読み。ヒット率の代用にはしない。
export function noteTierStats(d, frame) {
  if (!d || !frame) return false;
  const used = numOrNull(frame.vramMB);
  const total = numOrNull(frame.vramTotalMB);
  if (used === null || total === null || used < 0 || total < 0) return false;
  const nonNeg = (v) => { const n = numOrNull(v); return n !== null && n >= 0 ? n : null; };
  d.tier = {
    usedMB: used,
    totalMB: total,
    fill: total > 0 ? clamp01(used / total) : null,
    ramExperts: nonNeg(frame.ramExperts),
    ssdReads: nonNeg(frame.ssdReads),
  };
  return true;
}

/// Step フレーム (ns) / 生の ms → 1 ステップの所要時間 (ms) をリングに積む。
/// 負値・壊れた値は捨てる (数字を作らない)。
export function noteLatency(d, v) {
  if (!d || !d.latencyRing) return false;
  let ms = null;
  if (v && typeof v === 'object' && !Array.isArray(v)) {
    const ns = numOrNull(v.ns);
    if (ns !== null && ns >= 0) ms = ns / 1e6;
  } else {
    ms = numOrNull(v);
  }
  if (ms === null || ms < 0 || !Number.isFinite(ms)) return false;
  return ringPush(d.latencyRing, ms);
}

/// KvUsage フレーム → 使用中スロット数 / 全スロット数。比率は kvGauge が作る。
export function noteKv(d, frame) {
  if (!d || !frame || !Array.isArray(frame.busy)) return false;
  const slots = frame.busy.length;
  if (!slots) return false;
  let busy = 0;
  for (const b of frame.busy) if (b === true || b === 1) busy++;
  d.kv = { busy, slots };
  return true;
}

/// フレーム 1 枚を kind で振り分ける。計器に無関係な kind は false。
export function noteFrame(d, frame, now) {
  if (!d || !frame || typeof frame !== 'object') return false;
  switch (frame.kind) {
    case 'Prefill': return notePrefill(d, frame, now);
    case 'Token': return noteToken(d, 1, now);
    case 'Step': return noteLatency(d, frame);
    case 'KvUsage': return noteKv(d, frame);
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
/// 幅・高さ・余白に Infinity / NaN を渡されても既定値に落とす。
export function sparkPoints(values, opts = {}) {
  const o = opts && typeof opts === 'object' ? opts : {};
  const wv = numOrNull(o.w);
  const hv = numOrNull(o.h);
  const pv = numOrNull(o.pad);
  const w = wv !== null && wv > 0 ? wv : 220;
  const h = hv !== null && hv > 0 ? hv : 34;
  let pad = pv !== null && pv >= 0 ? pv : 2;
  pad = Math.min(pad, Math.min(w, h) / 2);          // 負の描画域を作らない
  const vals = (Array.isArray(values) ? values : []).filter((v) => Number.isFinite(v));
  if (!vals.length) return [];
  // 大量サンプルでも Math.min(...vals) のスタック溢れを起こさない。
  let dataLo = Infinity, dataHi = -Infinity;
  for (const v of vals) {
    if (v < dataLo) dataLo = v;
    if (v > dataHi) dataHi = v;
  }
  const lo = Number.isFinite(o.min) ? Number(o.min) : dataLo;
  const hi = Number.isFinite(o.max) ? Number(o.max) : dataHi;
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

/// ---- task 909: リング系列 → スパークライン / 利用量の棒 -------------------

/// latency のリング (120 本) → SVG path。空/一定/NaN 混じりでも安全。
export function latencyPath(d, opts = {}) {
  const ring = d && d.latencyRing;
  return sparkPath(ring && Array.isArray(ring.values) ? ring.values : [], opts);
}

/// 1 ステップの所要時間 (ms) の行。表示は画面がこれを描くだけ。
export function latencyRow(d, opts = {}) {
  const ring = d && d.latencyRing;
  const values = ring && Array.isArray(ring.values) ? ring.values : [];
  const last = values.length ? values[values.length - 1] : null;
  const ok = Number.isFinite(last) && last >= 0;
  return {
    key: 'latency',
    label: 'step latency ms',
    text: ok ? fmtRate(last) + ' ms' : '-',
    path: sparkPath(values, opts),
    samples: values.length,
  };
}

/// ---- task 969: レイテンシの分位点 / フィルタ / メトリクスの書き出し ------

/// 見る分位点 (パーセント)。画面のバッジも書き出しもこの並びを使う。
export const PERCENTILES = Object.freeze([50, 95, 99]);

/// 書き出す行の列。CSV のヘッダでもある。
export const EXPORT_COLUMNS = Object.freeze(['metric', 'value', 'samples', 'session', 'model']);

/// フィルタに使うキー。指定が空なら「全部」。
export const FILTER_KEYS = Object.freeze(['session', 'model']);

/// 線形補間 (R-7 / numpy 既定) の分位点。p は 0..100 に丸める。
/// 非有限値は落とし、1 本も残らなければ null。並べ替えはコピーで行う。
export function percentile(values, p) {
  const vals = (Array.isArray(values) ? values : []).filter((v) => Number.isFinite(v));
  if (!vals.length) return null;
  const q = Number(p);
  if (!Number.isFinite(q)) return null;
  const clamped = q < 0 ? 0 : q > 100 ? 100 : q;
  const sorted = vals.slice().sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const h = (sorted.length - 1) * clamped / 100;
  const lo = Math.floor(h);
  const hi = Math.ceil(h);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo);
}

/// 値の配列 → p50 / p95 / p99 (+ 件数 / 最小 / 最大)。空は全部 null。
export function latencyPercentiles(values) {
  const vals = (Array.isArray(values) ? values : []).filter((v) => Number.isFinite(v));
  const out = { p50: null, p95: null, p99: null, count: vals.length, min: null, max: null };
  if (!vals.length) return out;
  out.p50 = percentile(vals, 50);
  out.p95 = percentile(vals, 95);
  out.p99 = percentile(vals, 99);
  // 大量サンプルでもスプレッドでスタックを溢れさせない。
  let lo = Infinity, hi = -Infinity;
  for (const v of vals) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  out.min = lo;
  out.max = hi;
  return out;
}

/// 画面の Latency Summary バッジ。数字はここで作り、画面は text を書くだけ。
export function latencySummary(d, opts = {}) {
  const ring = d && d.latencyRing;
  const values = ring && Array.isArray(ring.values) ? ring.values : [];
  const p = latencyPercentiles(values);
  const ms = (v) => (v === null ? '-' : fmtRate(v) + (opts.unit === false ? '' : ' ms'));
  return {
    p50: p.p50,
    p95: p.p95,
    p99: p.p99,
    count: p.count,
    text: PERCENTILES.map((q) => 'p' + q + ' ' + ms(p['p' + q])).join(' · '),
  };
}

/// 1 項目がフィルタに合うか。filter は { session, model } の完全一致 (AND)。
/// 空文字・null・未指定の条件は「全部」。項目側に値が無ければ落とす。
export function matchesFilter(item, filter = {}) {
  if (!item || typeof item !== 'object') return false;
  const f = filter && typeof filter === 'object' ? filter : {};
  for (const key of FILTER_KEYS) {
    const want = f[key] == null ? '' : String(f[key]).trim();
    if (!want) continue;
    const have = item[key] == null ? '' : String(item[key]).trim();
    if (have !== want) return false;
  }
  return true;
}

/// 配列をフィルタで絞る。配列でなければ空。
export function filterItems(items, filter = {}) {
  if (!Array.isArray(items)) return [];
  return items.filter((it) => matchesFilter(it, filter));
}

const r3 = (x) => Math.round(x * 1000) / 1000;

/// ダッシュボードの現在値を書き出し用の行にする。
/// meta の session / model は各行に載せる (フィルタはこの 2 列で見る)。
export function metricsRows(d, meta = {}) {
  const src = d || createDash();
  const m = meta && typeof meta === 'object' ? meta : {};
  const session = m.session == null ? '' : String(m.session);
  const model = m.model == null ? '' : String(m.model);
  const rows = [];
  for (const key of SERIES_KEYS) {
    const values = src.series[key] || [];
    const last = values.length ? values[values.length - 1] : null;
    rows.push({
      metric: key,
      value: Number.isFinite(last) ? r3(last) : null,
      samples: values.length,
      session,
      model,
    });
  }
  const ring = src.latencyRing;
  const p = latencyPercentiles(ring && Array.isArray(ring.values) ? ring.values : []);
  for (const q of PERCENTILES) {
    const v = p['p' + q];
    rows.push({
      metric: 'latency_p' + q,
      value: v === null ? null : r3(v),
      samples: p.count,
      session,
      model,
    });
  }
  return rows;
}

/// CSV の 1 セル。null/未定義は空、非有限数も空、区切り・引用・改行を含む
/// 文字列は "..." で包み、内側の " は "" にする。
export function csvCell(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
  const s = String(v);
  return /[",\r\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/// 行の配列 → CSV (ヘッダ 1 行 + 改行は \n)。列は差し替えられる。
export function toCsv(rows, columns = EXPORT_COLUMNS) {
  const cols = Array.isArray(columns) && columns.length ? columns : EXPORT_COLUMNS;
  const lines = [cols.join(',')];
  for (const row of Array.isArray(rows) ? rows : []) {
    lines.push(cols.map((c) => csvCell(row ? row[c] : null)).join(','));
  }
  return lines.join('\n');
}

/// opts: { session, model, filter, columns }。filter は session / model で絞る。
export function metricsCsv(d, opts = {}) {
  const o = opts && typeof opts === 'object' ? opts : {};
  const rows = filterItems(metricsRows(d, o), o.filter);
  return toCsv(rows, o.columns);
}

/// 同じ行を JSON で返す (schema + metrics)。
export function metricsJson(d, opts = {}) {
  const o = opts && typeof opts === 'object' ? opts : {};
  const rows = filterItems(metricsRows(d, o), o.filter);
  return JSON.stringify({ schema: DASH_SCHEMA, metrics: rows }, null, 2);
}

/// KV / VRAM の利用量を「詰まった棒 + 割合」の 1 行にする。
/// used/total が壊れている (null, total<=0, used<0) ときは空の棒と '-'。
export function gaugeBar(used, total, opts = {}) {
  const o = opts && typeof opts === 'object' ? opts : {};
  const wn = numOrNull(o.w);
  const width = Math.min(40, Math.max(1, Math.trunc(wn !== null && wn > 0 ? wn : GAUGE_WIDTH)));
  const oneChar = (v, dflt) => {
    const s = v === undefined || v === null ? '' : String(v);
    return s === '' ? dflt : (Array.from(s)[0] || dflt);   // 幅 1 文字に固定
  };
  const full = oneChar(o.full, '█');
  const empty = oneChar(o.empty, '░');
  const blank = {
    ratio: null,
    pct: '-',
    filled: 0,
    width,
    bar: empty.repeat(width),
    text: '-',
  };
  const u = numOrNull(used);
  const t = numOrNull(total);
  if (u === null || t === null || !(t > 0) || u < 0) return blank;
  const ratio = clamp01(u / t);
  const filled = Math.round(ratio * width);
  const bar = full.repeat(filled) + empty.repeat(width - filled);
  const pct = Math.round(ratio * 100) + '%';
  return { ratio, pct, filled, width, bar, text: bar + ' ' + pct };
}

/// KV スロットの使用率 (KvUsage の busy / slots)。
export function kvGauge(d, opts = {}) {
  const kv = d && d.kv ? d.kv : null;
  return gaugeBar(kv ? kv.busy : null, kv ? kv.slots : null, opts);
}

/// VRAM の使用率 (TierStats の usedMB / totalMB)。
export function vramGauge(d, opts = {}) {
  const t = d && d.tier ? d.tier : null;
  return gaugeBar(t ? t.usedMB : null, t ? t.totalMB : null, opts);
}

/// ダッシュボードのクイックナビ。2 -> /2d, 3 -> /3d, R -> refresh。
/// 入力欄に打っている間と修飾キー付きは横取りしない。
export const QUICK_NAV = Object.freeze({
  '2': '/2d',
  '3': '/3d',
  'r': 'refresh',
});

export function quickNav(key, ctx = {}) {
  const c = ctx && typeof ctx === 'object' ? ctx : {};
  if (c.ctrlKey || c.metaKey || c.altKey) return null;
  const tag = String(c.tagName || '').toUpperCase();
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return null;
  if (c.contentEditable || c.isContentEditable) return null;
  const k = String(key == null ? '' : key).toLowerCase();
  return Object.prototype.hasOwnProperty.call(QUICK_NAV, k) ? QUICK_NAV[k] : null;
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
