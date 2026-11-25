// denno-view /2d の構成図 (DOM 非依存。描画は渡された canvas 2D context にだけ書く)。
//
// Qwen3.8-Flash-Next の公式構成図 (architecture.png) と同じ並びで、構成要素を全部置く:
//
//   左の列   : 入力 → Vocabulary embedding → 層 0..L-1 (GDN / QSA) → MTP modules
//              4 層ごとの点線 = Hybrid block (GDN ×3 + QSA ×1)。N-gram embedding は PLE 層へ
//   右の詳細 : Hybrid block ごとのパネル (×L/4)。中に 4 層ぶんの層パネル。層パネルは
//              Expanded residual (n_hc 本の stream) → GR Read → Hidden → Gated DeltaNet /
//              Qwen Sparse Attn → Hidden → GR Write → residual → GR Read → Hidden → MoE
//              (router・shared expert・n_expert 個の routed expert) → Hidden → GR Write → residual
//   上       : final GR Read → Hidden → Prediction head
//
// 座標は「世界座標」(1 単位 = 拡大率 1 の 1 CSS px)。拡大縮小は view = {s, tx, ty}
// (画面 = 世界 * s + t) で行い、描画は画面上の大きさで細部を出し分ける (文字は 6px 以上、
// expert のセルは 1.2px 以上で 1 つずつ、それ未満は層ごとの縮小画像)。

import {
  normaliseShape, roleForBlock, streamValue, rampRGB, heat, WALL_COLORS,
  EXPERT_FREQ_WINDOW, DRILL_MAX_ROWS,
} from './view2d.mjs';
import { HYBRID_SIZE } from './nodes2d.mjs';

/// 層パネルの中の配置 (パネル左上からの相対)。上へ流れる (入力が下)。
export const PANEL = Object.freeze({
  w: 1240, h: 930,
  res: Object.freeze({ x: 40, w: 300, h: 60 }),
  gr: Object.freeze({ x: 480, w: 200, h: 60 }),
  hid: Object.freeze({ x: 780, w: 80, h: 60 }),
  blk: Object.freeze({ x: 900, w: 310 }),
  trunkX: 190,           // residual から上へ伸びる幹の x
  r2: 70,                // 上: residual (出力) / GR Write (MoE 側) / hidden
  moeY: 170, moeH: 250,
  rB: 460,               // GR Read (MoE 側) / hidden
  r1: 580,               // 中: residual / GR Write (mixer 側) / hidden
  mixY: 680, mixH: 90,
  r0: 810,               // 下: residual (入力) / GR Read (mixer 側) / hidden
  labelY: 905,
});

export const ARCH = Object.freeze({
  pad: 200,
  colX: 200, colW: 520, colBoxFrac: 0.66,
  gridX: 1000, gridCols: 6,
  blockPad: 40, blockTitle: 80, layerGap: 70, blockGapX: 90, blockGapY: 110,
  topY: 120, topH: 520,
  bottomGap: 220, embH: 150,
});

function rect(x, y, w, h) { return { x, y, w, h }; }

function finiteNum(v, d) {
  const n = Number(v);
  return Number.isFinite(n) ? n : d;
}

/// 層が 0 の形 (テレメトリの ServerInfo が layers=0 を返した場合) 用の空 shape。
/// normaliseShape は layers が正でないと例外にするので、構成図だけは
/// 「何も置かない」で通す (view2d の契約は変えない)。
const EMPTY_SHAPE = Object.freeze({
  layers: 0, layer_kinds: Object.freeze([]), n_expert: 1, n_expert_used: 1,
  n_hc: 1, n_embd: 1, ple_layers: Object.freeze([]),
});

function layoutShape(raw) {
  try {
    return normaliseShape(raw);
  } catch (e) {
    if (raw && typeof raw === 'object' && !Array.isArray(raw) &&
        raw.layers !== undefined && raw.layers !== null && Number(raw.layers) === 0) {
      return EMPTY_SHAPE;
    }
    throw e;
  }
}

function emptyLayout() {
  const z = () => rect(0, 0, 0, 0);
  return {
    shape: EMPTY_SHAPE, width: 0, height: 0, bounds: z(),
    blocks: [], panels: [], column: [], groups: [], step: 0,
    grid: z(), mtp: z(), grFinal: z(), hidFinal: z(), head: z(),
    vocab: z(), ngram: z(), input: { x: 0, y: 0 }, colTop: 0,
  };
}

/// 形 → 構成図の配置 (世界座標)。
export function archLayout(shape) {
  const s = layoutShape(shape);
  if (s.layers === 0) return emptyLayout();
  const A = ARCH, P = PANEL;
  const L = s.layers;
  const nh = Math.ceil(L / HYBRID_SIZE);
  const cols = Math.min(A.gridCols, nh);
  const rows = Math.ceil(nh / cols);
  const blockW = P.w + 2 * A.blockPad;
  const blockH = A.blockTitle + HYBRID_SIZE * P.h + (HYBRID_SIZE - 1) * A.layerGap + A.blockPad;
  const gridY = A.topY + A.topH;
  const gridW = cols * blockW + (cols - 1) * A.blockGapX;
  const gridH = rows * blockH + (rows - 1) * A.blockGapY;
  const gridBottom = gridY + gridH;

  // Hybrid block のパネル: 下の行から左 → 右 (モデルの流れが下 → 上になるように)
  const blocks = [];
  for (let h = 0; h < nh; h++) {
    const r = Math.floor(h / cols);
    const c = h % cols;
    const x = A.gridX + c * (blockW + A.blockGapX);
    const y = gridY + (rows - 1 - r) * (blockH + A.blockGapY);
    const a = h * HYBRID_SIZE;
    const b = Math.min(L, a + HYBRID_SIZE) - 1;
    blocks.push({ hybrid: h, first: a, last: b, x, y, w: blockW, h: blockH });
  }

  // 層パネル: ブロックの中で下 (層 4h) → 上
  const panels = [];
  for (let l = 0; l < L; l++) {
    const B = blocks[Math.floor(l / HYBRID_SIZE)];
    const j = l % HYBRID_SIZE;
    const x = B.x + A.blockPad;
    const y = B.y + A.blockTitle + (HYBRID_SIZE - 1 - j) * (P.h + A.layerGap);
    panels.push(panelGeom(s, l, x, y));
  }

  // 左の列: 層 0 が下。列の高さを詳細グリッドに合わせる
  const step = gridH / L;
  const boxH = Math.max(20, step * A.colBoxFrac);
  const column = [];
  for (let l = 0; l < L; l++) {
    const yc = gridBottom - (l + 0.5) * step;
    column.push({ layer: l, kind: s.layer_kinds[l], ...rect(A.colX, yc - boxH / 2, A.colW, boxH) });
  }
  const groups = blocks.map((B) => {
    const lo = column[B.first], hi = column[B.last];
    const m = Math.max(10, (step - boxH) / 2 - 4);
    return { hybrid: B.hybrid, ...rect(A.colX - 30, hi.y - m, A.colW + 60, lo.y + lo.h - hi.y + 2 * m) };
  });

  // 上: MTP (列の上) / final GR Read → hidden → Prediction head
  const colTop = column[L - 1].y;
  const mtp = rect(A.colX - 40, A.topY, A.colW + 80, 150);
  const grFinal = rect(A.gridX + 400, A.topY + 300, 300, 90);
  const hidFinal = rect(grFinal.x + 440, grFinal.y, 110, 90);
  const head = rect(hidFinal.x + hidFinal.w / 2 - 380, A.topY, 760, 150);

  // 下: Vocabulary embedding / N-gram embedding / Input tokens
  const embY = gridBottom + A.bottomGap;
  const vocab = rect(A.colX - 80, embY, A.colW + 160, A.embH);
  const ngram = rect(A.gridX + 200, embY, gridW - 200, A.embH);
  const input = { x: A.colX + A.colW / 2, y: embY + A.embH + 200 };

  const width = A.gridX + gridW + A.pad;
  const height = input.y + 120;
  return {
    shape: s, width, height, bounds: rect(0, 0, width, height),
    blocks, panels, column, groups, step,
    grid: rect(A.gridX - 50, gridY - 60, gridW + 100, gridH + 120),
    mtp, grFinal, hidFinal, head, vocab, ngram, input, colTop,
  };
}

/// 層パネル 1 枚の部品の世界座標。
function panelGeom(s, l, x, y) {
  const P = PANEL;
  const kind = s.layer_kinds[l];
  const R = (dx, dy, w, h) => rect(x + dx, y + dy, w, h);
  const res = (ry) => R(P.res.x, ry, P.res.w, P.res.h);
  const gr = (ry) => R(P.gr.x, ry, P.gr.w, P.gr.h);
  const hid = (ry) => R(P.hid.x, ry, P.hid.w, P.hid.h);
  const moe = R(P.blk.x, P.moeY, P.blk.w, P.moeH);
  // MoE の中: 見出し行 (router) / shared expert / routed expert の格子
  const cols = Math.min(32, s.n_expert);
  const grows = Math.ceil(s.n_expert / cols);
  const cell = Math.min((moe.w - 40) / cols, (moe.h - 110) / grows);
  const gw = cell * cols, gh = cell * grows;
  const grid = {
    x: moe.x + (moe.w - gw) / 2, y: moe.y + moe.h - 14 - gh, w: gw, h: gh,
    cols, rows: grows, cell, n: s.n_expert,
  };
  const shared = rect(moe.x + moe.w - 150, moe.y + 12, 138, 40);
  const ple = s.ple_layers.indexOf(l) >= 0 ? R(P.gr.x, P.labelY - 30, P.gr.w, 44) : null;
  return {
    layer: l, kind, x, y, w: P.w, h: P.h,
    res0: res(P.r0), res1: res(P.r1), res2: res(P.r2),
    readA: gr(P.r0), writeA: gr(P.r1), readB: gr(P.rB), writeB: gr(P.r2),
    hidRA: hid(P.r0), hidWA: hid(P.r1), hidRB: hid(P.rB), hidWB: hid(P.r2),
    mixer: R(P.blk.x, P.mixY, P.blk.w, P.mixH),
    moe, shared, grid, ple,
  };
}

/// 残差の箱の中の stream i の円 (中心と半径)。n が 0 / 非有限でも割り算で
/// NaN や Infinity を返さない (壊れた shape でも描画を止めない)。
export function streamCircle(box, i, n) {
  const cnt = Number.isFinite(Number(n)) && n > 0 ? Math.floor(n) : 1;
  const w = box.w / cnt;
  return { cx: box.x + (i + 0.5) * w, cy: box.y + box.h / 2, r: Math.min(box.h * 0.32, w * 0.34) };
}

/// 層 l の expert e のセル (世界座標)。
export function archExpertRect(layout, layer, e) {
  const p = layout.panels[layer];
  if (!p) throw new Error('layer out of range: ' + layer);
  const g = p.grid;
  if (!(e >= 0 && e < g.n)) throw new Error('expert out of range: ' + e);
  return rect(g.x + (e % g.cols) * g.cell, g.y + Math.floor(e / g.cols) * g.cell, g.cell, g.cell);
}

function inR(r, x, y) { return !!r && x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h; }
function hitsR(r, a) { return !!r && r.x < a.x + a.w && r.x + r.w > a.x && r.y < a.y + a.h && r.y + r.h > a.y; }

const INFO_PARTS = Object.freeze([
  ['readA', 'GR Read'], ['writeA', 'GR Write'], ['readB', 'GR Read'], ['writeB', 'GR Write'],
  ['hidRA', 'hidden state'], ['hidWA', 'hidden state'], ['hidRB', 'hidden state'], ['hidWB', 'hidden state'],
]);

/// 世界座標の点 → 何があるか。選べるものは view2d.mjs の選択項目と同じ形:
///   {type:'expert', layer, expert} | {type:'block', layer, block} | {type:'stream', layer, stream}
/// 選べないもの: {type:'column', layer} | {type:'panel', layer} | {type:'hybrid', hybrid}
///   | {type:'info', layer, part, label} | {type:'top', part}
export function archHitTest(layout, x, y) {
  const s = layout.shape;
  for (const p of layout.panels) {
    if (!inR(p, x, y)) continue;
    const l = p.layer;
    const g = p.grid;
    if (inR(g, x, y)) {
      const e = Math.floor((y - g.y) / g.cell) * g.cols + Math.floor((x - g.x) / g.cell);
      if (e >= 0 && e < g.n) return { type: 'expert', layer: l, expert: e };
      return { type: 'block', layer: l, block: 'moe' };
    }
    if (inR(p.shared, x, y)) return { type: 'block', layer: l, block: 'shared' };
    if (inR(p.moe, x, y)) return { type: 'block', layer: l, block: 'moe' };
    if (inR(p.mixer, x, y)) return { type: 'block', layer: l, block: 'attn' };
    if (p.ple && inR(p.ple, x, y)) return { type: 'block', layer: l, block: 'ple' };
    for (const key of ['res0', 'res1', 'res2']) {
      const box = p[key];
      if (!inR(box, x, y)) continue;
      const i = Math.min(s.n_hc - 1, Math.max(0, Math.floor((x - box.x) / (box.w / s.n_hc))));
      return { type: 'stream', layer: l, stream: i };
    }
    for (const [key, label] of INFO_PARTS) {
      if (inR(p[key], x, y)) return { type: 'info', layer: l, part: key, label };
    }
    return { type: 'panel', layer: l };
  }
  for (const c of layout.column) if (inR(c, x, y)) return { type: 'column', layer: c.layer };
  for (const B of layout.blocks) {
    if (inR(rect(B.x, B.y, B.w, ARCH.blockTitle), x, y)) return { type: 'hybrid', hybrid: B.hybrid };
  }
  for (const k of ['mtp', 'grFinal', 'hidFinal', 'head', 'vocab', 'ngram']) {
    if (inR(layout[k], x, y)) return { type: 'top', part: k };
  }
  return null;
}

/// 世界座標の矩形 → 中の選択項目 (重複なし)。
export function archItemsInRect(layout, x0, y0, x1, y1) {
  const a = { x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) };
  const out = [];
  const s = layout.shape;
  for (const p of layout.panels) {
    if (!hitsR(p, a)) continue;
    const l = p.layer;
    const g = p.grid;
    if (hitsR(g, a)) {
      const c0 = Math.max(0, Math.floor((a.x - g.x) / g.cell));
      const c1 = Math.min(g.cols - 1, Math.floor((a.x + a.w - g.x) / g.cell));
      const r0 = Math.max(0, Math.floor((a.y - g.y) / g.cell));
      const r1 = Math.min(g.rows - 1, Math.floor((a.y + a.h - g.y) / g.cell));
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const e = r * g.cols + c;
          if (e < g.n) out.push({ type: 'expert', layer: l, expert: e });
        }
      }
    }
    if (hitsR(p.mixer, a)) out.push({ type: 'block', layer: l, block: 'attn' });
    // MoE の箱は格子を除いた見出しの帯に触れたときだけ (格子のドラッグで router まで切らない)
    if (hitsR(rect(p.moe.x, p.moe.y, p.shared.x - p.moe.x, g.y - p.moe.y), a)) {
      out.push({ type: 'block', layer: l, block: 'moe' });
    }
    if (hitsR(p.shared, a)) out.push({ type: 'block', layer: l, block: 'shared' });
    if (p.ple && hitsR(p.ple, a)) out.push({ type: 'block', layer: l, block: 'ple' });
    const seen = new Set();
    for (const key of ['res0', 'res1', 'res2']) {
      for (let i = 0; i < s.n_hc; i++) {
        const c = streamCircle(p[key], i, s.n_hc);
        if (seen.has(i)) continue;
        if (hitsR(rect(c.cx - c.r, c.cy - c.r, 2 * c.r, 2 * c.r), a)) {
          seen.add(i);
          out.push({ type: 'stream', layer: l, stream: i });
        }
      }
    }
  }
  return out;
}

/// ツリーのノード id → 世界座標の矩形 (ツリーから図へ飛ぶとき)。
export function archNodeRect(layout, id) {
  if (id === 'model' || id === 'emb' || id === 'out') return layout.bounds;
  if (id === 'vocab') return layout.vocab;
  if (id === 'ngram') return layout.ngram;
  if (id === 'head') return layout.head;
  if (id === 'mtp') return layout.mtp;
  if (id === 'grfinal') return layout.grFinal;
  let m = /^H(\d+)$/.exec(id);
  if (m) return layout.blocks[+m[1]] || null;
  m = /^L(\d+)(?:\/(\w+?)(\d*))?$/.exec(id);
  if (!m) return null;
  const p = layout.panels[+m[1]];
  if (!p) return null;
  const part = m[2] || '';
  const idx = m[3] === '' ? -1 : +m[3];
  if (!part) return p;
  if (part === 'res') return p.res0;
  if (part === 's') {
    const c = streamCircle(p.res0, idx, layout.shape.n_hc);
    return rect(c.cx - c.r * 3, c.cy - c.r * 3, c.r * 6, c.r * 6);
  }
  if (part === 'mix') return p.mixer;
  if (part === 'ffn' || part === 'moe' || part === 'ex') return p.moe;
  if (part === 'sh') return p.shared;
  if (part === 'ple') return p.ple || p;
  if (part === 'e') {
    const r = archExpertRect(layout, +m[1], idx);
    return rect(r.x - r.w * 6, r.y - r.h * 4, r.w * 13, r.h * 9);
  }
  return p;
}

// ---- 拡大縮小 ----------------------------------------------------------------

/// 矩形を pane に収める view。pane / r の値が欠けていても (0x0・NaN) 有限を返す。
export function fitView(r, pane, margin = 16) {
  const pw = finiteNum(pane && pane.w, 0), ph = finiteNum(pane && pane.h, 0);
  const m = finiteNum(margin, 0);
  const w = Math.max(1, pw - 2 * m), h = Math.max(1, ph - 2 * m);
  const rw = Math.max(1, finiteNum(r && r.w, 0)), rh = Math.max(1, finiteNum(r && r.h, 0));
  const rx = finiteNum(r && r.x, 0), ry = finiteNum(r && r.y, 0);
  const s = Math.min(w / rw, h / rh);
  return { s, tx: pw / 2 - (rx + rw / 2) * s, ty: ph / 2 - (ry + rh / 2) * s };
}

/// 画面の点 (sx, sy) を固定して factor 倍する。s は [minS, maxS] に収める。
/// view.s が 0 / 非有限・factor が NaN でも NaN を返さない。
export function zoomAt(v, factor, sx, sy, minS = 0.005, maxS = 16) {
  const cur = finiteNum(v && v.s, 0);
  const base = cur > 0 ? cur : 1;
  const f = finiteNum(factor, 1);
  const lo = finiteNum(minS, 0.005), hi = finiteNum(maxS, 16);
  const s = Math.max(lo, Math.min(hi, base * f));
  const k = s / base;
  const px = finiteNum(sx, 0), py = finiteNum(sy, 0);
  const tx = finiteNum(v && v.tx, 0), ty = finiteNum(v && v.ty, 0);
  return { s, tx: px - (px - tx) * k, ty: py - (py - ty) * k };
}

/// 画面 → 世界。s が 0 / 非有限の退化 view では原点を返す (NaN を広げない)。
export function toWorld(v, sx, sy) {
  const s = finiteNum(v && v.s, 0);
  const tx = finiteNum(v && v.tx, 0), ty = finiteNum(v && v.ty, 0);
  const px = finiteNum(sx, 0), py = finiteNum(sy, 0);
  if (!(s > 0)) return { x: 0, y: 0 };
  return { x: (px - tx) / s, y: (py - ty) / s };
}

// ---- 描画 -------------------------------------------------------------------

export const ARCH_COLORS = Object.freeze({
  bg: '#04080d', text: '#d8ecf7', dim: '#6d8796', line: '#7d97a6', arrow: '#9fb6c3',
  qsaPanel: '#0d1424', qsaEdge: '#4d6488', gdnPanel: '#0c1812', gdnEdge: '#4f7550',
  qsa: '#b46bff', gdn: '#00d9a0',
  grRead: '#f2a25c', grReadFill: '#24170b', grWrite: '#86b2e3', grWriteFill: '#0e1a2a',
  moe: '#e07a84', moeFill: '#231014', node: '#cfe3ee', nodeFill: '#09121a',
  head: '#e2cf9a', headFill: '#211c10', emb: '#b993d6', embFill: '#1a1224',
  dash: '#5f7a8a', sel: '#ffffff', off: WALL_COLORS.off, offFill: WALL_COLORS.offFill,
  skip: WALL_COLORS.skip, resident: WALL_COLORS.resident, none: WALL_COLORS.none,
});

function rgba(c, a) { return 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')'; }

/// 構成図を描く。
///   g    — canvas 2D context (setTransform は呼び出し側の dpr だけ。ここで view を掛ける)
///   view — {s, tx, ty, w, h} (w/h は CSS px の画面寸法)
///   D    — 表示データ:
///     st (createMapState), offExp: Map(l→Set), offBlk: Set("l:block"), offStr: Map(l→Set),
///     isSel(type, layer, key), dragKeys: Set|null ("e:l:i" など), hover: hit|null,
///     vramOn, gridImage(l) → 縮小画像 (canvas) | null, focusHybrid: number|null, dpr,
///     mtp: {seen, rate}|null (task 809。投機テレメトリが来ている間 MTP を強調)
export function drawArch(g, layout, view, D) {
  const C = ARCH_COLORS;
  const s = finiteNum(view && view.s, 0);
  if (!(s > 0)) return;                          // 退化した view (s=0 / NaN) は描かない
  const dpr = D.dpr || 1;
  const vw = finiteNum(view.w, 0), vh = finiteNum(view.h, 0);
  const vtx = finiteNum(view.tx, 0), vty = finiteNum(view.ty, 0);
  const vis = { x: -vtx / s, y: -vty / s, w: vw / s, h: vh / s };
  const px = (n) => n / s;                       // 画面 n px を世界の長さに
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.fillStyle = C.bg;
  g.fillRect(0, 0, vw, vh);
  g.setTransform(dpr * s, 0, 0, dpr * s, dpr * vtx, dpr * vty);
  g.lineJoin = 'round';
  // 0 層の形: 背景だけ描いて画面座標に戻す (列が無いので以下の配置を触らない)。
  if (!layout.column || !layout.column.length) {
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return;
  }

  // task 825: ablit 色モード (層 / Hybrid block を分離スコアで塗る)。
  const ABLIT = D.ablit && Array.isArray(D.ablit.scores) ? D.ablit : null;
  const ablitMax = ABLIT
    ? ABLIT.scores.reduce((a, v) => Math.max(a, Number(v) || 0), 0)
    : 0;
  const ablitNorm = (v) => (Number.isFinite(ablitMax) && ablitMax > 0
    ? Math.max(0, Math.min(1, finiteNum(v, 0) / ablitMax)) : 0);
  const ablitValue = (l) => (ABLIT ? ablitNorm(Number(ABLIT.scores[l]) || 0) : null);
  const ablitHybrid = (h) => {
    if (!ABLIT) return null;
    const per = 4;
    let sum = 0;
    let n = 0;
    for (let l = h * per; l < (h + 1) * per && l < ABLIT.scores.length; l++) {
      sum += Number(ABLIT.scores[l]) || 0;
      n++;
    }
    return n ? ablitNorm(sum / n) : 0;
  };

  const showText = (size) => size * s >= 6;
  const text = (str, x, y, size, color, opts = {}) => {
    if (!showText(size)) return;
    g.font = (opts.bold ? '600 ' : '') + size + 'px ui-sans-serif, system-ui, sans-serif';
    g.fillStyle = color;
    g.textAlign = opts.align || 'left';
    g.textBaseline = opts.base || 'middle';
    g.fillText(str, x, y);
  };
  const box = (r, fill, edge, lw = 1.4, radius = 8) => {
    const rr = Math.min(radius, r.w / 2, r.h / 2);
    g.beginPath();
    if (g.roundRect && rr * s > 1) g.roundRect(r.x, r.y, r.w, r.h, rr);
    else g.rect(r.x, r.y, r.w, r.h);
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (edge) { g.strokeStyle = edge; g.lineWidth = px(lw); g.stroke(); }
  };
  const glow = (r, v, radius = 8) => {
    if (!(v > 0)) return;
    const c = rampRGB(Math.min(1, v));
    box(r, rgba(c, 0.08 + 0.3 * Math.min(1, v)), null, 0, radius);
  };
  const arrowHead = (x, y, dx, dy) => {
    const L = Math.max(px(5), Math.min(14, px(9)));
    const W = L * 0.55;
    g.beginPath();
    g.moveTo(x, y);
    g.lineTo(x - dx * L - dy * W, y - dy * L + dx * W);
    g.lineTo(x - dx * L + dy * W, y - dy * L - dx * W);
    g.closePath();
    g.fillStyle = C.arrow;
    g.fill();
  };
  // 折れ線 + 終点の矢印
  const path = (pts, arrow = true, color = C.arrow) => {
    g.beginPath();
    g.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]);
    g.strokeStyle = color;
    g.lineWidth = px(1.3);
    g.stroke();
    if (arrow && pts.length >= 2) {
      const [x1, y1] = pts[pts.length - 1];
      const [x0, y0] = pts[pts.length - 2];
      const d = Math.hypot(x1 - x0, y1 - y0) || 1;
      arrowHead(x1, y1, (x1 - x0) / d, (y1 - y0) / d);
    }
  };
  const dashed = (r, color, dash = 10, lw = 1.6, radius = 0) => {
    g.save();
    g.setLineDash([px(dash), px(dash * 0.6)]);
    box(r, null, color, lw, radius);
    g.restore();
  };
  const visible = (r) => hitsR(r, vis);

  const st = D.st;
  const shape = layout.shape;
  const roleOf = (l) => (st && st.role && st.role[l]) || {};
  const blockOff = (l, b) => !!(D.offBlk && D.offBlk.has(l + ':' + b));
  const selOutline = (r, on, radius = 8) => { if (on) box(r, null, C.sel, 3, radius); };
  const offMark = (r, radius = 8) => {
    box(r, C.offFill, C.off, 2.4, radius);
    text('OFF', r.x + r.w - 10, r.y + 14, 16, '#ffb3b8', { align: 'right', bold: true });
  };
  const hov = D.hover;

  // ---- 上: MTP / final GR Read / hidden / Prediction head
  const top = layout;
  const colMid = top.column[0].x + top.column[0].w / 2;
  path([[colMid, top.colTop], [colMid, top.mtp.y + top.mtp.h]]);
  path([[colMid, top.grFinal.y + top.grFinal.h / 2], [top.grFinal.x, top.grFinal.y + top.grFinal.h / 2]]);
  path([[top.grFinal.x + top.grFinal.w, top.grFinal.y + 45], [top.hidFinal.x, top.hidFinal.y + 45]]);
  path([[top.hidFinal.x + top.hidFinal.w / 2, top.hidFinal.y], [top.hidFinal.x + top.hidFinal.w / 2, top.head.y + top.head.h]]);
  box(top.mtp, C.headFill, C.head, 2, 12);
  text('MTP Modules', top.mtp.x + top.mtp.w / 2, top.mtp.y + top.mtp.h / 2, 46, C.head, { align: 'center' });
  // task 809: 投機 (MTP / speculative) のテレメトリが届いている間は MTP の箱を
  // 明るい縁と内側の光で強調し、採用率を下に出す。数字がまだ無い (未知 kind だけ)
  // ときは "spec telemetry" とだけ書く (推測の数字は出さない)。
  const mtpTele = D.mtp && D.mtp.seen ? D.mtp : null;
  if (mtpTele) {
    g.save();
    g.globalAlpha = 0.22;
    box(top.mtp, 'rgba(127,227,255,.35)', null, 0, 14);
    g.restore();
    box(top.mtp, null, '#7fe3ff', 3.2, 12);
    const label = mtpTele.rate == null ? 'spec telemetry' : 'accept ' + Math.round(mtpTele.rate * 100) + '%';
    text(label, top.mtp.x + top.mtp.w / 2, top.mtp.y + top.mtp.h + 44, 34, '#7fe3ff',
      { align: 'center', bold: true });
  }
  box(top.head, C.headFill, C.head, 2, 12);
  text('Prediction Head', top.head.x + top.head.w / 2, top.head.y + top.head.h / 2, 50, C.head, { align: 'center' });
  box(top.grFinal, C.grReadFill, C.grRead, 2, 10);
  text('GR Read', top.grFinal.x + top.grFinal.w / 2, top.grFinal.y + 45, 38, C.grRead, { align: 'center' });
  box(top.hidFinal, C.nodeFill, C.node, 2, 8);
  g.beginPath();
  g.arc(top.hidFinal.x + 55, top.hidFinal.y + 45, 22, 0, Math.PI * 2);
  g.strokeStyle = C.node; g.lineWidth = px(1.6); g.stroke();

  // ---- 下: 入力 / Vocabulary embedding / N-gram embedding
  const vb = top.vocab, ng = top.ngram, inp = top.input;
  path([[inp.x, inp.y - 40], [inp.x, vb.y + vb.h]]);
  path([[inp.x, inp.y - 80], [ng.x + ng.w / 2, inp.y - 80], [ng.x + ng.w / 2, ng.y + ng.h]]);
  path([[colMid, vb.y], [colMid, top.column[0].y + top.column[0].h]]);
  box(vb, C.embFill, C.emb, 2, 12);
  text('Vocabulary Embedding', vb.x + vb.w / 2, vb.y + vb.h / 2, 46, C.emb, { align: 'center' });
  box(ng, C.embFill, C.emb, 2, 12);
  text('N-gram Embedding Layer', ng.x + ng.w / 2, ng.y + ng.h / 2 - 14, 52, C.emb, { align: 'center' });
  text('Layer ' + (shape.ple_layers.join(', ') || '-') + ' only', ng.x + ng.w / 2, ng.y + ng.h / 2 + 40,
    30, C.emb, { align: 'center', bold: true });
  text('Input Tokens', inp.x, inp.y + 10, 44, C.text, { align: 'center' });
  // N-gram → PLE 層 (列の箱の右側へ)
  for (const l of shape.ple_layers) {
    const c = top.column[l];
    if (!c) continue;
    const xm = c.x + c.w + 110;
    path([[ng.x, ng.y + ng.h / 2], [xm, ng.y + ng.h / 2], [xm, c.y + c.h / 2], [c.x + c.w, c.y + c.h / 2]]);
  }

  // ---- 左の列: 層の箱と Hybrid block の点線
  for (let l = 0; l < top.column.length; l++) {
    const c = top.column[l];
    if (l + 1 < top.column.length) {
      const n = top.column[l + 1];
      path([[colMid, c.y], [colMid, n.y + n.h]]);
    }
  }
  for (const gr of top.groups) {
    const focus = D.focusHybrid === gr.hybrid;
    dashed(gr, focus ? '#9fdcff' : C.dash, 9, focus ? 2.6 : 1.6, 6);
  }
  for (const c of top.column) {
    if (!visible(c)) continue;
    const l = c.layer;
    const qsa = c.kind === 'attn';
    const v = streamValue(roleOf(l));
    const hot = hov && hov.layer === l;
    // task 825: ablit 色モードでは 48 層の柱を層ごとの分離スコアで塗る。
    const av = D.ablit ? ablitValue(l) : null;
    const fill = av != null ? heat(av) : (qsa ? C.qsaPanel : C.gdnPanel);
    box(c, fill, hot ? '#cfefff' : (qsa ? C.qsa : C.gdn), hot ? 3 : 2, 8);
    if (av != null) {
      // 分離の強い層の柱は白い内枠で目立たせる (色だけだと暗い背景で沈む)。
      g.save();
      g.globalAlpha = 0.35 + 0.5 * av;
      box({ x: c.x + 2, y: c.y + 2, w: c.w - 4, h: c.h - 4 }, null, '#dff4ff', 1.2, 6);
      g.restore();
    }
    glow(c, v);
    // 中で切ってあるものがあれば左端に赤い帯
    const anyOff = (D.offExp && D.offExp.get(l) && D.offExp.get(l).size) ||
      ['attn', 'moe', 'shared', 'ple'].some((b) => blockOff(l, b)) ||
      (D.offStr && D.offStr.get(l) && D.offStr.get(l).size);
    if (anyOff) { g.fillStyle = C.off; g.fillRect(c.x, c.y, Math.max(px(3), 10), c.h); }
    text((qsa ? 'QSA' : 'GDN') + ' Layer', c.x + c.w / 2 + 30, c.y + c.h / 2, Math.min(44, c.h * 0.42),
      C.text, { align: 'center' });
    text(String(l).padStart(2, '0'), c.x + 24, c.y + c.h / 2, Math.min(34, c.h * 0.34), C.dim);
  }

  // ---- 右: Hybrid block の外枠と点線のつなぎ
  dashed(top.grid, C.dash, 16, 2, 0);
  text('Hybrid Block × ' + top.blocks.length + '  (L / 4)', top.grid.x + top.grid.w, top.grid.y - 34, 40,
    C.text, { align: 'right', bold: true });
  if (D.focusHybrid != null && top.groups[D.focusHybrid] && top.blocks[D.focusHybrid]) {
    const gr = top.groups[D.focusHybrid], B = top.blocks[D.focusHybrid];
    g.save();
    g.setLineDash([px(4), px(4)]);
    g.strokeStyle = '#9fdcff';
    g.lineWidth = px(1.4);
    g.beginPath();
    g.moveTo(gr.x + gr.w, gr.y); g.lineTo(B.x, B.y);
    g.moveTo(gr.x + gr.w, gr.y + gr.h); g.lineTo(B.x, B.y + B.h);
    g.stroke();
    g.restore();
  }
  for (const B of top.blocks) {
    if (!visible(B)) continue;
    const focus = D.focusHybrid === B.hybrid;
    // task 825: ablit 色モードでは 12 個の Hybrid block を 4 層の平均で塗る。
    const bv = ablitHybrid(B.hybrid);
    if (bv != null) {
      g.save();
      g.globalAlpha = 0.20 + 0.45 * bv;
      g.fillStyle = heat(bv);
      g.fillRect(B.x, B.y, B.w, B.h);
      g.restore();
    }
    dashed(B, focus ? '#9fdcff' : C.dash, 12, focus ? 2.6 : 1.8, 18);
    text('Hybrid block ' + B.hybrid + '  ·  layers ' + B.first + '–' + B.last,
      B.x + ARCH.blockPad, B.y + ARCH.blockTitle / 2 + 4, 40, C.text, { bold: true });
  }

  // ---- 層パネル
  for (const p of top.panels) {
    if (!visible(p)) continue;
    drawPanel(p);
  }

  function drawPanel(p) {
    const l = p.layer;
    const qsa = p.kind === 'attn';
    const role = roleOf(l);
    const hot = hov && hov.layer === l && (hov.type === 'panel' || hov.type === 'column');
    box(p, qsa ? C.qsaPanel : C.gdnPanel, hot ? '#cfefff' : (qsa ? C.qsaEdge : C.gdnEdge), hot ? 3 : 2, 34);
    text((qsa ? 'QSA' : 'GDN') + ' Layer  ' + String(l).padStart(2, '0'), p.x + 40, p.y + 36, 34, C.text, { bold: true });
    if (p.w * s < 40) return;     // 小さすぎるときは箱だけ

    const P = PANEL;
    const X = (dx) => p.x + dx, Y = (dy) => p.y + dy;
    const midY = (r) => r.y + r.h / 2;
    const tx = X(P.trunkX);
    const gx = X(P.gr.x + P.gr.w / 2);
    const bx = X(P.blk.x + P.blk.w / 2);
    // 線 (先に描いて、箱で上書きする)
    path([[tx, p.y + p.h], [tx, p.res0.y + p.res0.h]]);                  // 入力 → residual
    path([[p.res0.x + p.res0.w, midY(p.res0)], [p.readA.x, midY(p.readA)]]);
    path([[p.readA.x + p.readA.w, midY(p.readA)], [p.hidRA.x, midY(p.hidRA)]]);
    path([[p.hidRA.x + p.hidRA.w, midY(p.hidRA)], [bx, midY(p.hidRA)], [bx, p.mixer.y + p.mixer.h]]);
    path([[bx, p.mixer.y], [bx, midY(p.hidWA)], [p.hidWA.x + p.hidWA.w, midY(p.hidWA)]]);
    path([[p.hidWA.x, midY(p.hidWA)], [p.writeA.x + p.writeA.w, midY(p.writeA)]]);
    path([[p.writeA.x, midY(p.writeA)], [p.res1.x + p.res1.w, midY(p.res1)]]);
    path([[tx, p.res0.y], [tx, Y(725)], [gx, Y(725)], [gx, p.writeA.y + p.writeA.h]]);
    path([[tx, p.res1.y], [tx, Y(300)], [gx, Y(300)], [gx, p.writeB.y + p.writeB.h]]);
    path([[tx, midY(p.readB)], [p.readB.x, midY(p.readB)]]);
    path([[p.readB.x + p.readB.w, midY(p.readB)], [p.hidRB.x, midY(p.hidRB)]]);
    path([[p.hidRB.x + p.hidRB.w, midY(p.hidRB)], [bx, midY(p.hidRB)], [bx, p.moe.y + p.moe.h]]);
    path([[bx, p.moe.y], [bx, midY(p.hidWB)], [p.hidWB.x + p.hidWB.w, midY(p.hidWB)]]);
    path([[p.hidWB.x, midY(p.hidWB)], [p.writeB.x + p.writeB.w, midY(p.writeB)]]);
    path([[p.writeB.x, midY(p.writeB)], [p.res2.x + p.res2.w, midY(p.res2)]]);
    path([[tx, p.res2.y], [tx, p.y]], false);                              // 次の層へ
    if (p.ple) path([[p.ple.x, midY(p.ple)], [X(300), midY(p.ple)], [X(300), p.res0.y + p.res0.h]]);

    // residual の箱と stream の円
    const sv = streamValue(role);
    const offS = D.offStr && D.offStr.get(l);
    for (const key of ['res0', 'res1', 'res2']) {
      const r = p[key];
      box(r, C.nodeFill, C.node, 1.8, 10);
      for (let i = 0; i < shape.n_hc; i++) {
        const c = streamCircle(r, i, shape.n_hc);
        const off = offS && offS.has(i);
        const selNow = D.isSel('stream', l, i) || (D.dragKeys && D.dragKeys.has('s:' + l + ':' + i));
        g.beginPath();
        g.arc(c.cx, c.cy, c.r, 0, Math.PI * 2);
        g.fillStyle = off ? C.off : sv > 0 ? heat(sv) : C.nodeFill;
        g.fill();
        g.strokeStyle = selNow ? C.sel : off ? C.off : C.node;
        g.lineWidth = px(selNow ? 3 : 1.4);
        g.stroke();
      }
    }
    text('Expanded Residual', X(P.trunkX + 14), Y(P.labelY), 26, C.dim);
    text('Hidden States', X(P.hid.x - 20), Y(P.labelY), 26, C.dim);

    // GR Read / Write と hidden
    for (const key of ['readA', 'readB']) {
      box(p[key], C.grReadFill, C.grRead, 1.8, 8);
      text('GR Read', p[key].x + p[key].w / 2, midY(p[key]), 28, C.grRead, { align: 'center' });
    }
    for (const key of ['writeA', 'writeB']) {
      box(p[key], C.grWriteFill, C.grWrite, 1.8, 8);
      text('GR Write', p[key].x + p[key].w / 2, midY(p[key]), 28, C.grWrite, { align: 'center' });
    }
    const hv = { hidRA: sv, hidWA: roleForBlock(role, 'attn', p.kind), hidRB: sv, hidWB: role.router || 0 };
    for (const key of ['hidRA', 'hidWA', 'hidRB', 'hidWB']) {
      const r = p[key];
      box(r, C.nodeFill, C.node, 1.6, 8);
      g.beginPath();
      g.arc(r.x + r.w / 2, midY(r), 15, 0, Math.PI * 2);
      g.fillStyle = hv[key] > 0 ? heat(hv[key]) : C.nodeFill;
      g.fill();
      g.strokeStyle = C.node; g.lineWidth = px(1.3); g.stroke();
    }

    // mixer (Gated DeltaNet / Qwen Sparse Attn)
    const mv = roleForBlock(role, 'attn', p.kind);
    const mixColor = qsa ? C.qsa : C.gdn;
    if (blockOff(l, 'attn')) offMark(p.mixer, 10);
    else { box(p.mixer, '#0a1410', mixColor, 2.2, 10); glow(p.mixer, mv, 10); }
    text(qsa ? 'Qwen Sparse Attn' : 'Gated DeltaNet', bx, midY(p.mixer), 32, C.text, { align: 'center' });
    selOutline(p.mixer, D.isSel('block', l, 'attn') || (D.dragKeys && D.dragKeys.has('b:' + l + ':attn')), 10);

    // MoE (router 見出し / shared expert / routed expert の格子)
    const moeOff = blockOff(l, 'moe');
    if (moeOff) offMark(p.moe, 10);
    else { box(p.moe, C.moeFill, C.moe, 2.2, 10); glow(rect(p.moe.x, p.moe.y, p.moe.w, p.grid.y - p.moe.y - 6), role.router || 0, 10); }
    text('MoE', p.moe.x + 18, p.moe.y + 32, 32, C.moe, { bold: true });
    text('top-' + shape.n_expert_used + ' / ' + shape.n_expert, p.moe.x + 18, p.moe.y + 66, 18, C.dim);
    selOutline(p.moe, D.isSel('block', l, 'moe') || (D.dragKeys && D.dragKeys.has('b:' + l + ':moe')), 10);
    if (blockOff(l, 'shared')) offMark(p.shared, 6);
    else { box(p.shared, '#170b0e', C.moe, 1.6, 6); glow(p.shared, role.shex || 0, 6); }
    text('shared expert', p.shared.x + p.shared.w / 2, midY(p.shared), 18, C.text, { align: 'center' });
    selOutline(p.shared, D.isSel('block', l, 'shared') || (D.dragKeys && D.dragKeys.has('b:' + l + ':shared')), 6);
    drawGrid(p, moeOff);

    // PLE (N-gram 入力)
    if (p.ple) {
      if (blockOff(l, 'ple')) offMark(p.ple, 8);
      else { box(p.ple, C.embFill, C.emb, 1.8, 8); glow(p.ple, role.ple || 0, 8); }
      text('PLE (N-gram)', p.ple.x + p.ple.w / 2, midY(p.ple), 22, C.emb, { align: 'center' });
      selOutline(p.ple, D.isSel('block', l, 'ple') || (D.dragKeys && D.dragKeys.has('b:' + l + ':ple')), 8);
    }
  }

  function drawGrid(p, moeOff) {
    const l = p.layer;
    const gd = p.grid;
    const cellPx = gd.cell * s;
    g.fillStyle = C.none;
    g.fillRect(gd.x, gd.y, gd.w, gd.h);
    if (cellPx < 1.2 || !visible(gd)) {
      const img = D.gridImage ? D.gridImage(l) : null;
      if (img) {
        g.imageSmoothingEnabled = cellPx < 1;
        g.drawImage(img, gd.x, gd.y, gd.w, gd.h);
      }
      return;
    }
    const arr = st.experts[l];
    let scale = 1;
    if (st.mode === 'accum') { for (let i = 0; i < gd.n; i++) if (arr[i] > scale) scale = arr[i]; }
    const offE = D.offExp && D.offExp.get(l);
    const gap = cellPx >= 5 ? px(1) : 0;
    const c0 = Math.max(0, Math.floor((vis.x - gd.x) / gd.cell));
    const c1 = Math.min(gd.cols - 1, Math.floor((vis.x + vis.w - gd.x) / gd.cell));
    const r0 = Math.max(0, Math.floor((vis.y - gd.y) / gd.cell));
    const r1 = Math.min(gd.rows - 1, Math.floor((vis.y + vis.h - gd.y) / gd.cell));
    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        const e = r * gd.cols + c;
        if (e >= gd.n) continue;
        const x = gd.x + c * gd.cell, y = gd.y + r * gd.cell;
        const off = offE && offE.has(e);
        let col;
        if (st.skip[l][e] === 1) col = C.skip;
        else if (off || moeOff) col = off ? C.off : '#3a1418';
        else if (arr[e] > 0) col = heat(arr[e] / scale);
        else col = C.none;
        g.fillStyle = col;
        g.fillRect(x, y, gd.cell - gap, gd.cell - gap);
        if (D.vramOn && st.residency[l][e] === 1 && !off) {
          g.fillStyle = C.resident;
          g.fillRect(x, y, gd.cell * 0.3, gd.cell * 0.3);
        }
      }
    }
    if (cellPx >= 3) {
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const e = r * gd.cols + c;
          if (e >= gd.n) continue;
          const selNow = D.isSel('expert', l, e) || (D.dragKeys && D.dragKeys.has('e:' + l + ':' + e));
          if (!selNow && st.topk[l][e] !== 1) continue;
          g.strokeStyle = C.sel;
          g.lineWidth = px(selNow ? 2 : 1);
          g.strokeRect(gd.x + c * gd.cell, gd.y + r * gd.cell, gd.cell - gap, gd.cell - gap);
        }
      }
    }
    if (cellPx >= 16) {
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          const e = r * gd.cols + c;
          if (e >= gd.n) continue;
          text(String(e), gd.x + (c + 0.5) * gd.cell, gd.y + (r + 0.5) * gd.cell, gd.cell * 0.32,
            'rgba(230,245,255,.75)', { align: 'center' });
        }
      }
    }
  }

  // 画面座標に戻す (呼び出し側がドラッグ枠などを描く)
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
}

/// 層 l の expert 格子を 1 セル = 1 画素の RGBA に書く (縮小表示用)。
export function gridPixels(layout, st, l, out, offE, moeOff) {
  const p = layout.panels && layout.panels[l];
  if (!p || !st || !st.experts || !st.experts[l] || !st.skip || !st.skip[l]) return out;
  const gd = p.grid;
  const arr = st.experts[l];
  let scale = 1;
  if (st.mode === 'accum') { for (let i = 0; i < gd.n; i++) if (arr[i] > scale) scale = arr[i]; }
  const none = [7, 16, 25];
  for (let e = 0; e < gd.cols * gd.rows; e++) {
    let c;
    if (e >= gd.n) c = none;
    else if (st.skip[l][e] === 1) c = [255, 196, 77];
    else if (offE && offE.has(e)) c = [255, 59, 48];
    else if (moeOff) c = [58, 20, 24];
    else if (arr[e] > 0) c = rampRGB(arr[e] / scale);
    else c = none;
    out[e * 4] = c[0]; out[e * 4 + 1] = c[1]; out[e * 4 + 2] = c[2]; out[e * 4 + 3] = 255;
  }
  return out;
}

// ---- task 857: 層ドリルダウン -------------------------------------------------
//
// 層の箱 (column) か層パネルをクリックしたとき、その層の router が選んだ
// expert の id・ゲート重み・VRAM hit / CPU miss・直近トークンの頻度を 1 枚に
// まとめる。DOM には触らない (index2d.html がこの値を描く)。
//
// テレメトリは最小構成でも動く: RouterTopK だけなら重みは出て hit/miss は
// 「不明」(null)、ExpertResidency が来ていれば vram / cpu を出す。
// 何も来ていなければ note にその旨を 1 行入れて空の行を返す。

/// hit/miss の色 (ドリルダウンのバッジと頻度バーで共有)。
export const DRILL_COLORS = Object.freeze({
  vram: '#3fe08a', cpu: '#ff8a94', unknown: '#5b7280',
  bar: '#2f8aa8', weight: '#9fefff',
});

/// 直近のトークン列 (st.hist[l]) から expert ごとの 0/1 の系列を取る。
/// ids は expert id の配列。窓の古い → 新しい順。window は実際に残っている本数。
export function expertFrequency(st, layer, ids, opts = {}) {
  if (!st) return { window: 0, rows: [] };
  const l = Math.trunc(Number(layer));
  const n = st && st.shape && Number.isFinite(Number(st.shape.n_expert)) ? st.shape.n_expert | 0 : 0;
  const cap = Math.max(1, Math.trunc(Number(opts.window) || EXPERT_FREQ_WINDOW));
  const all = st && Array.isArray(st.hist) && Array.isArray(st.hist[l]) ? st.hist[l] : [];
  const win = all.slice(Math.max(0, all.length - cap));
  const rows = [];
  for (const raw of Array.isArray(ids) ? ids : []) {
    const e = Math.trunc(Number(raw));
    if (!(e >= 0 && (n <= 0 || e < n))) continue;
    const values = win.map((bits) => (bits && bits[e] === 1 ? 1 : 0));
    let hits = 0;
    for (const v of values) hits += v;
    rows.push({ id: e, values, hits, rate: win.length ? hits / win.length : 0 });
  }
  return { window: win.length, rows };
}

/// 層 1 枚のドリルダウンの中身。layer が範囲外・st が無いときは null。
/// opts.maxRows で並べる expert の上限を変えられる (既定 DRILL_MAX_ROWS)。
export function expertDrilldown(st, shape, layer, opts = {}) {
  const l = Math.trunc(Number(layer));
  if (!st || !Number.isFinite(l)) return null;
  let s;
  try { s = normaliseShape(st.shape || shape); } catch (e) { return null; }
  if (!(l >= 0 && l < s.layers)) return null;

  const topk = Array.isArray(st.topk) ? st.topk[l] : null;
  const last = Array.isArray(st.last) ? st.last[l] : null;
  const raw = Array.isArray(st.raw) ? st.raw[l] : null;
  const res = Array.isArray(st.residency) ? st.residency[l] : null;
  const seenRes = !!(Array.isArray(st.residencySeen) ? st.residencySeen[l] : false);
  const skip = Array.isArray(st.skip) ? st.skip[l] : null;
  const count = Array.isArray(st.count) ? st.count[l] : null;

  const ids = [];
  if (topk) for (let e = 0; e < s.n_expert; e++) if (topk[e] === 1) ids.push(e);
  // 重み: 生のゲート値があればそれ、無ければ正規化済みの最終重みで代用する。
  const weightOf = (e) => {
    if (raw && Number(raw[e]) > 0) return Number(raw[e]);
    return last && Number.isFinite(Number(last[e])) ? Number(last[e]) : 0;
  };
  let sum = 0;
  for (const e of ids) sum += weightOf(e);

  let rows = ids.map((e) => ({
    id: e,
    weight: weightOf(e),
    share: sum > 0 ? weightOf(e) / sum : 0,
    rel: last && Number.isFinite(Number(last[e])) ? Number(last[e]) : 0,
    hit: seenRes ? (res && res[e] === 1 ? 'vram' : 'cpu') : null,
    skipped: !!(skip && skip[e] === 1),
    count: count ? Number(count[e]) || 0 : 0,
  }));
  rows.sort((a, b) => (b.weight - a.weight) || (a.id - b.id));
  const limit = Math.max(1, Math.trunc(Number(opts.maxRows) || DRILL_MAX_ROWS));
  const shown = rows.slice(0, limit);

  // 頻度のスパークライン (表示する行だけ)
  const freq = expertFrequency(st, l, shown.map((r) => r.id), opts);
  for (let i = 0; i < shown.length; i++) {
    const f = freq.rows[i];
    shown[i].values = f ? f.values : [];
    shown[i].freqHits = f ? f.hits : 0;
    shown[i].freqRate = f ? f.rate : 0;
  }

  let hitVram = 0;
  let hitCpu = 0;
  for (const r of rows) {
    if (r.hit === 'vram') hitVram++;
    else if (r.hit === 'cpu') hitCpu++;
  }
  const notes = [];
  if (!rows.length) notes.push('router の top-k 待ち (RouterTopK がまだ来ていない)');
  else {
    if (!seenRes) notes.push('ExpertResidency 待ち (hit / miss は不明。DENNO_EXPERT_GPU=1 で出る)');
    if (rows.length > shown.length) notes.push('上位 ' + shown.length + ' 件だけ表示 (' + rows.length + ' 件)');
  }
  const note = notes.join(' · ');

  return {
    layer: l,
    kind: s.layer_kinds[l],
    n_expert: s.n_expert,
    n_used: rows.length,
    shown: shown.length,
    residency: seenRes,
    hitVram,
    hitCpu,
    experts: shown,
    window: freq.window,
    freqWindow: EXPERT_FREQ_WINDOW,
    tokens: Number(st.token) || 0,
    note,
  };
}

/// 0/1 (または 0..1) の系列 → 小さな棒の矩形 (SVG 用)。
/// 新しい値ほど右。w/h は CSS px。空の系列は []。
export function drillSparkBars(values, w = 64, h = 12) {
  const vals = Array.isArray(values) ? values : [];
  const n = vals.length;
  if (!n) return [];
  const slot = w / n;
  const bw = Math.max(0.5, slot - Math.min(1, slot * 0.25));
  const out = [];
  for (let i = 0; i < n; i++) {
    const v = Math.max(0, Math.min(1, Number(vals[i]) || 0));
    const bh = Math.max(v > 0 ? 1 : 0, v * h);
    out.push({ x: i * slot, y: h - bh, w: bw, h: bh, v });
  }
  return out;
}
