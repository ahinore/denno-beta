// denno-view /2d の純ロジック (DOM 非依存)。
//
// ここには「形 → レイアウト」「テレメトリのフレーム → 色の値」「選択 → リクエスト本文」
// 「API のモック店」だけを置く。描画とイベントは index2d.html 側。node --test から
// 直接 import して検証する (crates/denno-view/tests/view2d.test.mjs)。
//
// 契約 (サーバー側 API。実装は task 217/218。まだ無いときはモックに落ちる):
//   GET    /v1/model/shape -> {layers, layer_kinds, n_expert, n_expert_used, n_hc, n_embd}
//   GET    /v1/intervene   -> {epoch, experts_off, blocks_off, streams_off}
//   POST   /v1/intervene    (上と同じ 3 配列。全置換)
//   DELETE /v1/intervene    (全消し)
//   GET    /v1/steer       -> {recording, snapshots, apply}
//   POST   /v1/steer/record {name} / POST /v1/steer/record/stop
//   DELETE /v1/steer/snapshots/<name>
//   POST   /v1/steer/apply  {terms:[{snapshot,coef}], layers:[from,to], alpha, mode}
//   DELETE /v1/steer/apply

export const VIEW_SCHEMA = 1;

/// 実モデル (Qwen3.8-Flash-Next) の形。shape が 404 のときの既定値でもある。
export const DEFAULT_SHAPE_RAW = Object.freeze({
  layers: 48,
  n_expert: 512,
  n_expert_used: 10,
  n_hc: 4,
  n_embd: 2560,
});

export const BLOCKS = Object.freeze(['attn', 'moe', 'shared', 'ple']);
export const STEER_MODES = Object.freeze(['subtract', 'project']);

/// レイアウトの寸法 (CSS px)。描画側とテストで同じ値を使う。
export const GEOM = Object.freeze({
  padX: 8,
  padY: 8,
  rowH: 46,
  gap: 5,
  label: { w: 26 },
  mixer: { w: 52, h: 13 },
  router: { w: 34, h: 13 },
  expert: { cell: 3, gap: 1, maxCols: 64 },
  shared: { w: 20, h: 13 },
  ple: { w: 18, h: 13 },
  stream: { w: 7, h: 10, gap: 3 },
  strip: { h: 3, gap: 2 },
});

/// LayerAct の idx (0..n_embd) を畳むビンの数。層あたりの細い帯になる。
export const ACT_BUCKETS = 64;

function intOr(v, dflt) {
  const n = Number(v);
  return Number.isFinite(n) ? Math.trunc(n) : dflt;
}

function clampInt(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

/// 層種別の表記ゆれを 'attn' (full attention) / 'gdn' (ssm) に寄せる。
export function mixerKind(kind) {
  const s = String(kind == null ? '' : kind).trim().toLowerCase().replace(/[_\s-]+/g, ' ');
  if (s === 'full' || s === 'full attn' || s === 'full attention' || s === 'attn' ||
      s === 'attention' || s === 'qsa' || s === 'fullattn' || s === 'self attention') return 'attn';
  return 'gdn';
}

/// shape を検証して既定値で埋める。layers だけは必須 (0 以下は例外)。
export function normaliseShape(raw) {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('shape: object required');
  }
  const layers = intOr(raw.layers, 0);
  if (!(layers > 0)) throw new Error('shape.layers must be a positive integer');
  const n_expert = Math.max(1, intOr(raw.n_expert, DEFAULT_SHAPE_RAW.n_expert));
  const n_expert_used = clampInt(intOr(raw.n_expert_used, DEFAULT_SHAPE_RAW.n_expert_used), 1, n_expert);
  const n_hc = Math.max(1, intOr(raw.n_hc, DEFAULT_SHAPE_RAW.n_hc));
  const n_embd = Math.max(1, intOr(raw.n_embd, DEFAULT_SHAPE_RAW.n_embd));

  const kindsRaw = Array.isArray(raw.layer_kinds) ? raw.layer_kinds : null;
  const layer_kinds = [];
  for (let l = 0; l < layers; l++) {
    if (kindsRaw && kindsRaw[l] != null && kindsRaw[l] !== '') layer_kinds.push(mixerKind(kindsRaw[l]));
    else layer_kinds.push((l + 1) % 4 === 0 ? 'attn' : 'gdn');
  }

  let ple_layers;
  if (Array.isArray(raw.ple_layers)) {
    ple_layers = raw.ple_layers
      .map((x) => intOr(x, -1))
      .filter((x) => x >= 0 && x < layers)
      .filter((x, i, a) => a.indexOf(x) === i)
      .sort((a, b) => a - b);
  } else {
    ple_layers = layers > 2 ? [2] : [];
  }

  return Object.freeze({
    layers,
    layer_kinds: Object.freeze(layer_kinds),
    n_expert,
    n_expert_used,
    n_hc,
    n_embd,
    ple_layers: Object.freeze(ple_layers),
  });
}

export const DEFAULT_SHAPE = normaliseShape(DEFAULT_SHAPE_RAW);

function rect(x, y, w, h) {
  return { x, y, w, h };
}

/// 形から 2D の配置を組む。1 層 = 1 行。行の中に
/// mixer / router / expert 格子 (n_expert) / shared / PLE / n_hc 本の stream を置く。
export function layoutFromShape(shape) {
  const s = normaliseShape(shape);
  const G = GEOM;
  const cols = Math.min(s.n_expert, G.expert.maxCols);
  const grows = Math.ceil(s.n_expert / cols);
  const expertW = cols * G.expert.cell + Math.max(0, cols - 1) * G.expert.gap;
  const expertH = grows * G.expert.cell + Math.max(0, grows - 1) * G.expert.gap;
  const streamW = s.n_hc * G.stream.w + Math.max(0, s.n_hc - 1) * G.stream.gap;
  const rowH = Math.max(G.rowH, expertH + G.strip.h + G.strip.gap + 12);

  const labelX = G.padX;
  const mixX = labelX + G.label.w + G.gap;
  const rtX = mixX + G.mixer.w + G.gap;
  const exX = rtX + G.router.w + G.gap;
  const shX = exX + expertW + G.gap;
  const pleX = shX + G.shared.w + G.gap;
  const stX = pleX + G.ple.w + G.gap;
  const width = stX + streamW + G.padX;
  const height = G.padY * 2 + s.layers * rowH;

  const rows = [];
  for (let l = 0; l < s.layers; l++) {
    const y = G.padY + l * rowH;
    const gy = y + Math.max(4, Math.round((rowH - (expertH + G.strip.h + G.strip.gap)) / 2));
    const midY = gy + Math.round((expertH - G.mixer.h) / 2);
    rows.push({
      layer: l,
      kind: s.layer_kinds[l],
      y,
      h: rowH,
      label: rect(labelX, y + Math.round((rowH - 10) / 2), G.label.w, 10),
      mixer: Object.assign(rect(mixX, midY, G.mixer.w, G.mixer.h), { block: 'attn' }),
      router: Object.assign(rect(rtX, midY, G.router.w, G.router.h), { block: 'moe' }),
      experts: {
        x: exX, y: gy, w: expertW, h: expertH,
        cols, rows: grows, cell: G.expert.cell, gap: G.expert.gap, n: s.n_expert,
      },
      shared: Object.assign(rect(shX, midY, G.shared.w, G.shared.h), { block: 'shared' }),
      ple: Object.assign(rect(pleX, midY, G.ple.w, G.ple.h), {
        block: 'ple', active: s.ple_layers.indexOf(l) >= 0,
      }),
      streams: Array.from({ length: s.n_hc }, (_, i) =>
        rect(stX + i * (G.stream.w + G.stream.gap),
          midY + Math.round((G.mixer.h - G.stream.h) / 2), G.stream.w, G.stream.h)),
      strip: rect(exX, gy + expertH + G.strip.gap, expertW, G.strip.h),
    });
  }
  return { shape: s, width, height, rowH, cols, rows, geom: G };
}

/// 層 l の expert e のセルの矩形。
export function expertCellRect(layout, layer, e) {
  const row = layout.rows[layer];
  if (!row) throw new Error('layer out of range: ' + layer);
  if (!(e >= 0 && e < row.experts.n)) throw new Error('expert out of range: ' + e);
  const ex = row.experts;
  const gx = ex.cell + ex.gap;
  const ci = e % ex.cols;
  const ri = Math.floor(e / ex.cols);
  return rect(ex.x + ci * gx, ex.y + ri * gx, ex.cell, ex.cell);
}

function rowAt(layout, y) {
  const l = Math.floor((y - layout.geom.padY) / layout.rowH);
  if (l < 0 || l >= layout.rows.length) return null;
  return layout.rows[l];
}

function inRect(r, x, y) {
  return x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
}

/// 点 → 選択できるもの。無ければ null。
/// 返り値は {type:'expert', layer, expert} | {type:'block', layer, block} |
/// {type:'stream', layer, stream}。
export function hitTest(layout, x, y) {
  const row = rowAt(layout, y);
  if (!row) return null;
  if (inRect(row.mixer, x, y)) return { type: 'block', layer: row.layer, block: 'attn' };
  if (inRect(row.router, x, y)) return { type: 'block', layer: row.layer, block: 'moe' };
  if (inRect(row.shared, x, y)) return { type: 'block', layer: row.layer, block: 'shared' };
  if (inRect(row.ple, x, y)) return { type: 'block', layer: row.layer, block: 'ple' };
  const ex = row.experts;
  if (inRect(ex, x, y)) {
    const gx = ex.cell + ex.gap;
    const ci = Math.floor((x - ex.x) / gx);
    const ri = Math.floor((y - ex.y) / gx);
    const idx = ri * ex.cols + ci;
    if (ci >= 0 && ci < ex.cols && ri >= 0 && ri < ex.rows && idx < ex.n) {
      return { type: 'expert', layer: row.layer, expert: idx };
    }
    return null;
  }
  for (let i = 0; i < row.streams.length; i++) {
    if (inRect(row.streams[i], x, y)) return { type: 'stream', layer: row.layer, stream: i };
  }
  return null;
}

/// 矩形 (ドラッグ) → 中の選択項目。expert はセルが矩形に触れたものを全部返す。
export function itemsInRect(layout, x0, y0, x1, y1) {
  const ax = Math.min(x0, x1), bx = Math.max(x0, x1);
  const ay = Math.min(y0, y1), by = Math.max(y0, y1);
  const out = [];
  const hits = (r) => r.x < bx && r.x + r.w > ax && r.y < by && r.y + r.h > ay;
  for (const row of layout.rows) {
    if (row.y + row.h <= ay || row.y >= by) continue;
    const ex = row.experts;
    if (hits(ex)) {
      const gx = ex.cell + ex.gap;
      const c0 = Math.max(0, Math.floor((ax - ex.x) / gx));
      const c1 = Math.min(ex.cols - 1, Math.floor((bx - ex.x) / gx));
      const r0 = Math.max(0, Math.floor((ay - ex.y) / gx));
      const r1 = Math.min(ex.rows - 1, Math.floor((by - ex.y) / gx));
      for (let ri = r0; ri <= r1; ri++) {
        for (let ci = c0; ci <= c1; ci++) {
          const idx = ri * ex.cols + ci;
          if (idx >= ex.n) continue;
          const cell = rect(ex.x + ci * gx, ex.y + ri * gx, ex.cell, ex.cell);
          if (hits(cell)) out.push({ type: 'expert', layer: row.layer, expert: idx });
        }
      }
    }
    const blocks = [['attn', row.mixer], ['moe', row.router], ['shared', row.shared], ['ple', row.ple]];
    for (const [name, r] of blocks) {
      if (hits(r)) out.push({ type: 'block', layer: row.layer, block: name });
    }
    for (let i = 0; i < row.streams.length; i++) {
      if (hits(row.streams[i])) out.push({ type: 'stream', layer: row.layer, stream: i });
    }
  }
  return out;
}

// ---- 選択 → /v1/intervene の本文 ------------------------------------------

const SEL_FIELD = { expert: 'experts', block: 'blocks', stream: 'streams' };

export function createSelection() {
  return { experts: new Set(), blocks: new Set(), streams: new Set() };
}

function itemKey(item) {
  if (item.type === 'expert') return 'e:' + item.layer + ':' + item.expert;
  if (item.type === 'block') {
    if (BLOCKS.indexOf(item.block) < 0) throw new Error('unknown block: ' + item.block);
    return 'b:' + item.layer + ':' + item.block;
  }
  if (item.type === 'stream') return 's:' + item.layer + ':' + item.stream;
  throw new Error('unknown item type: ' + item.type);
}

export function addItem(sel, item) {
  sel[SEL_FIELD[item.type]].add(itemKey(item));
}

export function removeItem(sel, item) {
  sel[SEL_FIELD[item.type]].delete(itemKey(item));
}

export function toggleItem(sel, item) {
  const set = sel[SEL_FIELD[item.type]];
  const k = itemKey(item);
  if (set.has(k)) { set.delete(k); return false; }
  set.add(k);
  return true;
}

export function hasItem(sel, item) {
  return sel[SEL_FIELD[item.type]].has(itemKey(item));
}

export function clearSelection(sel) {
  sel.experts.clear();
  sel.blocks.clear();
  sel.streams.clear();
}

export function selectionCount(sel) {
  return sel.experts.size + sel.blocks.size + sel.streams.size;
}

function parseKey(k, prefix) {
  const p = k.slice(prefix.length).split(':');
  return p.map((x) => (/^\d+$/.test(x) ? Number(x) : x));
}

/// 選択 → POST /v1/intervene の本文 (決定的な順序)。
export function selectionToBody(sel) {
  const experts = [...sel.experts].map((k) => parseKey(k, 'e:')).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const blocks = [...sel.blocks].map((k) => parseKey(k, 'b:'))
    .sort((a, b) => a[0] - b[0] || BLOCKS.indexOf(a[1]) - BLOCKS.indexOf(b[1]));
  const streams = [...sel.streams].map((k) => parseKey(k, 's:')).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return { experts_off: experts, blocks_off: blocks, streams_off: streams };
}

/// API の本文 → 選択 (画面の「off」表示に使う)。
export function selectionFromBody(body) {
  const sel = createSelection();
  const b = body || {};
  for (const [l, e] of b.experts_off || []) addItem(sel, { type: 'expert', layer: l | 0, expert: e | 0 });
  for (const [l, name] of b.blocks_off || []) addItem(sel, { type: 'block', layer: l | 0, block: String(name) });
  for (const [l, s] of b.streams_off || []) addItem(sel, { type: 'stream', layer: l | 0, stream: s | 0 });
  return sel;
}

/// 2 つの本文の和 (off 済み + 選択、など)。
export function mergeBodies(a, b) {
  const sel = selectionFromBody(a);
  for (const it of selectionItems(selectionFromBody(b))) addItem(sel, it);
  return selectionToBody(sel);
}

/// 選択を item の配列に戻す (テストとデバッグ用)。
export function selectionItems(sel) {
  const out = [];
  for (const [l, e] of selectionToBody(sel).experts_off) out.push({ type: 'expert', layer: l, expert: e });
  for (const [l, name] of selectionToBody(sel).blocks_off) out.push({ type: 'block', layer: l, block: name });
  for (const [l, s] of selectionToBody(sel).streams_off) out.push({ type: 'stream', layer: l, stream: s });
  return out;
}

// ---- フレーム → 色の値 ----------------------------------------------------

/// RouterTopK → expert ごとの重み (0..1、その層の最大で正規化)。
export function routerWeights(frame, n_expert) {
  const out = new Float32Array(n_expert);
  if (!frame || frame.kind !== 'RouterTopK') return out;
  const experts = frame.experts || [];
  const weights = frame.weights || [];
  let max = 0;
  for (let i = 0; i < experts.length; i++) {
    const e = experts[i];
    const w = Number(weights[i]);
    if (e >= 0 && e < n_expert && Number.isFinite(w) && w > max) max = w;
  }
  if (!(max > 0)) return out;
  for (let i = 0; i < experts.length; i++) {
    const e = experts[i];
    const w = Number(weights[i]);
    if (e >= 0 && e < n_expert && Number.isFinite(w) && w > 0) out[e] = w / max;
  }
  return out;
}

/// LayerAct → n_embd を ACT_BUCKETS に畳んだ値 (ビン内最大、フレーム内最大で正規化)。
export function actBuckets(frame, n_embd, buckets = ACT_BUCKETS) {
  const out = new Float32Array(buckets);
  if (!frame || frame.kind !== 'LayerAct') return out;
  const idx = frame.idx || [];
  const val = frame.val || [];
  let max = 0;
  for (let i = 0; i < idx.length; i++) {
    if (!(idx[i] >= 0 && idx[i] < n_embd)) continue;
    const v = Number(val[i]);
    if (Number.isFinite(v) && v > max) max = v;
  }
  if (!(max > 0)) return out;
  for (let i = 0; i < idx.length; i++) {
    const v = Number(val[i]);
    if (!Number.isFinite(v) || v <= 0) continue;
    if (!(idx[i] >= 0 && idx[i] < n_embd)) continue;
    const b = Math.max(0, Math.min(buckets - 1, Math.floor((idx[i] / n_embd) * buckets)));
    if (v / max > out[b]) out[b] = v / max;
  }
  return out;
}

/// ExpertResidency → expert ごとの 0/1 (VRAM 常駐)。
export function residencyBits(frame, n_expert) {
  const out = new Uint8Array(n_expert);
  if (!frame || frame.kind !== 'ExpertResidency') return out;
  for (const e of frame.vram || []) if (e >= 0 && e < n_expert) out[e] = 1;
  return out;
}

/// 行のどこが光るか。mixer は層種別で attn / gdn を選ぶ。
export function roleForBlock(role, block, kind) {
  const r = role || {};
  if (block === 'attn') return kind === 'attn' ? (r.attn || 0) : (r.gdn || 0);
  if (block === 'moe') return r.router || 0;
  if (block === 'shared') return r.shex || 0;
  if (block === 'ple') return r.ple || 0;
  return 0;
}

/// stream の明るさ = その層の役割の最大 (stream 自身のテレメトリは無い)。
export function streamValue(role) {
  const r = role || {};
  return Math.max(r.attn || 0, r.gdn || 0, r.router || 0, r.shex || 0, r.ple || 0);
}

/// 0..1 を色に。ramp: 暗い青 → 青緑 → シアン → 黄 → 赤。
const RAMP = [
  [8, 19, 28],
  [18, 80, 107],
  [0, 184, 212],
  [255, 225, 77],
  [255, 59, 48],
];

export function rampRGB(t) {
  let v = Number(t);
  if (!Number.isFinite(v)) v = 0;
  v = v < 0 ? 0 : v > 1 ? 1 : v;
  const x = v * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = RAMP[i], b = RAMP[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

export function heat(v, scale = 1) {
  const s = Number(scale) > 0 ? Number(scale) : 1;
  const t = Number(v) / s;
  const c = rampRGB(t);
  return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
}

// ---- 状態 (トークンごと / 会話の累積) -------------------------------------

/// フレームを食って地図の状態を更新する。DOM には触らない。
/// mode:
///   'token' — 最後の 1 トークンの重みで置き換える (アニメーション)
///   'accum' — 会話全体の和。Token ごとに decay を掛けて古いものを薄くする。
export function createMapState(shape, opts = {}) {
  const s = normaliseShape(shape);
  const decay = Number.isFinite(opts.decay) ? opts.decay : 0.9;
  const st = {
    shape: s,
    mode: opts.mode === 'accum' ? 'accum' : 'token',
    decay,
    token: 0,
    experts: [],
    residency: [],
    role: [],
    act: [],
    lastToken: null,
    frames: 0,
  };
  for (let l = 0; l < s.layers; l++) {
    st.experts.push(new Float32Array(s.n_expert));
    st.residency.push(new Uint8Array(s.n_expert));
    st.role.push({ attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 });
    st.act.push(new Float32Array(ACT_BUCKETS));
  }

  st.setMode = (mode) => {
    const m = mode === 'accum' ? 'accum' : 'token';
    if (m === st.mode) return;
    st.mode = m;
    for (let l = 0; l < s.layers; l++) st.experts[l].fill(0);
  };

  st.reset = () => {
    st.token = 0;
    st.frames = 0;
    st.lastToken = null;
    for (let l = 0; l < s.layers; l++) {
      st.experts[l].fill(0);
      st.residency[l].fill(0);
      st.role[l] = { attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 };
      st.act[l].fill(0);
    }
  };

  st.onFrame = (f) => {
    if (!f || typeof f !== 'object') return false;
    if (f.kind === 'Token') {
      st.token++;
      st.lastToken = f;
      if (st.mode === 'accum') {
        for (let i = 0; i < s.layers; i++) {
          const a = st.experts[i];
          for (let j = 0; j < a.length; j++) if (a[j] !== 0) a[j] *= decay;
        }
      }
      return true;
    }
    if (f.layer == null || f.layer >= s.layers) return false;
    const l = f.layer | 0;
    switch (f.kind) {
      case 'RouterTopK': {
        const w = routerWeights(f, s.n_expert);
        if (st.mode === 'accum') {
          const a = st.experts[l];
          for (let i = 0; i < w.length; i++) a[i] += w[i];
        } else {
          st.experts[l].set(w);
        }
        st.frames++;
        return true;
      }
      case 'RoleAct': {
        st.role[l] = Object.assign({ attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 }, f.roles || {});
        st.frames++;
        return true;
      }
      case 'LayerAct': {
        st.act[l].set(actBuckets(f, s.n_embd));
        st.frames++;
        return true;
      }
      case 'ExpertResidency': {
        st.residency[l].set(residencyBits(f, s.n_expert));
        st.frames++;
        return true;
      }
      default:
        return false;
    }
  };

  return st;
}

// ---- ステアリング ---------------------------------------------------------

/// 画面の入力 → POST /v1/steer/apply の本文。不正は例外 (画面は赤字で出す)。
export function steerApplyBody(spec) {
  const src = spec || {};
  const terms = src.terms;
  if (!Array.isArray(terms) || terms.length === 0) {
    throw new Error('terms: at least one snapshot is required');
  }
  const outTerms = terms.map((t, i) => {
    if (!t || typeof t.snapshot !== 'string' || !t.snapshot) {
      throw new Error('terms[' + i + '].snapshot is required');
    }
    const coef = Number(t.coef);
    if (!Number.isFinite(coef) || coef === 0) {
      throw new Error('terms[' + i + '].coef must be a non-zero number');
    }
    return { snapshot: t.snapshot, coef };
  });
  const layers = Array.isArray(src.layers) ? src.layers : [0, 0];
  if (layers.length !== 2) throw new Error('layers: [from, to] required');
  const from = intOr(layers[0], -1);
  const to = intOr(layers[1], -1);
  if (from < 0 || to < 0) throw new Error('layers: from/to must be >= 0');
  if (from > to) throw new Error('layers: from must be <= to');
  const alpha = src.alpha == null ? 1 : Number(src.alpha);
  if (!Number.isFinite(alpha) || alpha < 0) throw new Error('alpha: must be a number >= 0');
  const mode = src.mode == null ? 'subtract' : String(src.mode);
  if (STEER_MODES.indexOf(mode) < 0) throw new Error('mode: subtract or project');
  return { terms: outTerms, layers: [from, to], alpha, mode };
}

/// いま効いているステアリングの 1 行要約 (バナー用)。
export function steerSummary(apply) {
  if (!apply) return '';
  const parts = (apply.terms || []).map((t) => {
    const c = Number(t.coef);
    return t.snapshot + (c >= 0 ? ' +' : ' ') + c;
  });
  const layers = Array.isArray(apply.layers) ? apply.layers.join('-') : String(apply.layers);
  return parts.join(' · ') + ' · ' + apply.mode + ' · α ' + apply.alpha + ' · layers ' + layers;
}

// ---- モック店 (shape が 404 のとき) ---------------------------------------

function jsonClone(v) {
  return v == null ? v : JSON.parse(JSON.stringify(v));
}

/// shape が返らないときの API 代役。handle(method, path, body) -> {status, body}。
/// 実 API と同じ規則: POST /v1/intervene は全置換、DELETE は全消し、epoch は
/// 状態を変えるたびに 1 増える。
export function createMockStore(shape, opts = {}) {
  const s = normaliseShape(shape || DEFAULT_SHAPE_RAW);
  const now = typeof opts.now === 'function' ? opts.now : () => new Date().toISOString();
  const st = {
    epoch: 0,
    experts_off: [],
    blocks_off: [],
    streams_off: [],
    recording: null,
    snapshots: [],
    apply: null,
    tokens: 0,
  };
  let tick = 0;

  function err(status, message) {
    return { status, body: { error: { message } } };
  }

  function validIntervene(body) {
    if (!body || typeof body !== 'object') return 'body must be an object';
    const lists = [['experts_off', 2], ['blocks_off', 2], ['streams_off', 2]];
    for (const [name, arity] of lists) {
      const v = body[name] == null ? [] : body[name];
      if (!Array.isArray(v)) return name + ' must be an array';
      for (const entry of v) {
        if (!Array.isArray(entry) || entry.length !== arity) return name + ' entries must be [..] pairs';
      }
    }
    for (const [l, e] of body.experts_off || []) {
      if (!(l >= 0 && l < s.layers)) return 'experts_off: layer out of range: ' + l;
      if (!(e >= 0 && e < s.n_expert)) return 'experts_off: expert out of range: ' + e;
    }
    for (const [l, name] of body.blocks_off || []) {
      if (!(l >= 0 && l < s.layers)) return 'blocks_off: layer out of range: ' + l;
      if (BLOCKS.indexOf(name) < 0) return 'blocks_off: unknown block: ' + name;
    }
    for (const [l, x] of body.streams_off || []) {
      if (!(l >= 0 && l < s.layers)) return 'streams_off: layer out of range: ' + l;
      if (!(x >= 0 && x < s.n_hc)) return 'streams_off: stream out of range: ' + x;
    }
    return null;
  }

  function normaliseOff(body) {
    const uniq = (rows, cmp) => {
      const seen = new Set();
      const out = [];
      for (const r of rows) {
        const k = r.join(':');
        if (!seen.has(k)) { seen.add(k); out.push(r); }
      }
      return out.sort(cmp);
    };
    return {
      experts_off: uniq(body.experts_off || [], (a, b) => a[0] - b[0] || a[1] - b[1]),
      blocks_off: uniq(body.blocks_off || [], (a, b) => a[0] - b[0] || BLOCKS.indexOf(a[1]) - BLOCKS.indexOf(b[1])),
      streams_off: uniq(body.streams_off || [], (a, b) => a[0] - b[0] || a[1] - b[1]),
    };
  }

  const store = {
    shape: s,
    state: () => jsonClone(store.snapshot()),
    snapshot: () => ({
      shape: { layers: s.layers, layer_kinds: [...s.layer_kinds], n_expert: s.n_expert,
        n_expert_used: s.n_expert_used, n_hc: s.n_hc, n_embd: s.n_embd },
      epoch: st.epoch,
      intervene: { epoch: st.epoch, experts_off: st.experts_off, blocks_off: st.blocks_off, streams_off: st.streams_off },
      steer: { recording: st.recording, snapshots: st.snapshots, apply: st.apply },
      tokens: st.tokens,
    }),

    /// Token フレームのたびに呼ぶ。記録中ならトークン数を進める。
    tick(n = 1) {
      tick += n;
      if (st.recording) st.recording.tokens += n;
      st.tokens += n;
    },

    handle(method, path, body) {
      const m = String(method || 'GET').toUpperCase();
      const p = String(path || '');
      if (m === 'GET' && p === '/v1/model/shape') {
        return { status: 200, body: store.snapshot().shape };
      }
      if (m === 'GET' && p === '/v1/intervene') {
        return {
          status: 200,
          body: { epoch: st.epoch, experts_off: jsonClone(st.experts_off),
            blocks_off: jsonClone(st.blocks_off), streams_off: jsonClone(st.streams_off) },
        };
      }
      if (m === 'POST' && p === '/v1/intervene') {
        const bad = validIntervene(body);
        if (bad) return err(400, bad);
        const off = normaliseOff(body);
        st.experts_off = off.experts_off;
        st.blocks_off = off.blocks_off;
        st.streams_off = off.streams_off;
        st.epoch++;
        return { status: 200, body: { epoch: st.epoch, ...jsonClone(off) } };
      }
      if (m === 'DELETE' && p === '/v1/intervene') {
        st.experts_off = [];
        st.blocks_off = [];
        st.streams_off = [];
        st.epoch++;
        return { status: 200, body: { epoch: st.epoch, experts_off: [], blocks_off: [], streams_off: [] } };
      }
      if (m === 'GET' && p === '/v1/steer') {
        return {
          status: 200,
          body: { recording: jsonClone(st.recording), snapshots: jsonClone(st.snapshots), apply: jsonClone(st.apply) },
        };
      }
      if (m === 'POST' && p === '/v1/steer/record') {
        if (st.recording) return err(409, 'already recording: ' + st.recording.name);
        const name = body && typeof body.name === 'string' ? body.name.trim() : '';
        if (!name) return err(400, 'name is required');
        if (st.snapshots.some((x) => x.name === name)) return err(409, 'snapshot already exists: ' + name);
        st.recording = { name, tokens: 0, created: now() };
        return { status: 200, body: jsonClone(st.recording) };
      }
      if (m === 'POST' && p === '/v1/steer/record/stop') {
        if (!st.recording) return err(409, 'not recording');
        const snap = {
          name: st.recording.name,
          tokens: st.recording.tokens,
          created: now(),
          layers: [0, s.layers - 1],
        };
        st.recording = null;
        st.snapshots.push(snap);
        return { status: 200, body: jsonClone(snap) };
      }
      if (m === 'DELETE' && p.indexOf('/v1/steer/snapshots/') === 0) {
        const name = decodeURIComponent(p.slice('/v1/steer/snapshots/'.length));
        const i = st.snapshots.findIndex((x) => x.name === name);
        if (i < 0) return err(404, 'no such snapshot: ' + name);
        st.snapshots.splice(i, 1);
        return { status: 200, body: { deleted: name } };
      }
      if (m === 'POST' && p === '/v1/steer/apply') {
        let spec;
        try {
          spec = steerApplyBody(body);
        } catch (e) {
          return err(400, (e && e.message) || String(e));
        }
        for (const t of spec.terms) {
          if (!st.snapshots.some((x) => x.name === t.snapshot)) {
            return err(404, 'no such snapshot: ' + t.snapshot);
          }
        }
        st.apply = { ...spec, applied: now() };
        return { status: 200, body: jsonClone(st.apply) };
      }
      if (m === 'DELETE' && p === '/v1/steer/apply') {
        st.apply = null;
        return { status: 200, body: { apply: null } };
      }
      return err(404, 'not found: ' + m + ' ' + p);
    },
  };
  return store;
}
