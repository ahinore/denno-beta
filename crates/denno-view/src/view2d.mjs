// denno-view /2d の純ロジック (DOM 非依存)。
//
// ここには「形 → レイアウト」「テレメトリのフレーム → 色の値」「選択 → リクエスト本文」
// 「API のモック店」だけを置く。描画とイベントは index2d.html 側。node --test から
// 直接 import して検証する (crates/denno-view/tests/view2d.test.mjs)。
//
// 契約 (サーバー側 API。実装は task 217/218。404 のときだけモックに落ちる):
//   GET    /v1/model/shape -> {layers, layer_kinds, n_expert, n_expert_used, n_hc, n_embd,
//                              ple_layers}
//   GET    /v1/intervene   -> {epoch, experts_off, blocks_off, streams_off}
//   POST   /v1/intervene    (上と同じ 3 配列。全置換)
//   DELETE /v1/intervene    (全消し)
//   GET    /v1/steer       -> {epoch, recording, snapshots, apply}
//   POST   /v1/steer/record {name} / POST /v1/steer/record/stop
//   DELETE /v1/steer/snapshots/<name>
//   POST   /v1/steer/apply  {terms:[{snapshot,coef}], layers:[from,to), alpha, mode}
//   DELETE /v1/steer/apply
//
// task 220: `/2d` はこの契約を `/api` プロキシ経由でそのまま使う。モック
// (`createMockStore`) は実 API の**検証規則と status まで写した**もので、shape が
// 404 のとき (と ?mock=1) だけ使う。実サーバーに対する往復は
// `crates/denno-view/examples/smoke_2d.sh` が検査する (lab t220a で md5 SAME /
// decode ±0.2% / prefill ±0.9%)。

export const VIEW_SCHEMA = 1;

/// 実モデル (Qwen3.8-Flash-Next) の形。shape が 404 のときの既定値でもある。
/// `ple_layers` は実サーバーの `GET /v1/model/shape` が返す (task 220。実モデルは層 2)。
export const DEFAULT_SHAPE_RAW = Object.freeze({
  layers: 48,
  n_expert: 512,
  n_expert_used: 10,
  n_hc: 4,
  n_embd: 2560,
  ple_layers: [2],
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

// ---- 層の壁 (task 235) ----------------------------------------------------
// 1 枚のタイル = 1 層。48 層を reading order (左上が層 0) に並べる。
// 列数とセル寸法は**地図の実面積**から決める (ページは ResizeObserver で pane を
// 渡す) ので、窓の大きさやチャット / 操作列の開閉に追従する。単位は CSS px。
//
// 1 枚の中身:
//   [層番号 + 種別バッジ]
//   [n_expert の 32x16 ヒートマップ][RoleAct のバー 4 本]
//   [n_hc 本の残差 stream の四角]
// 壁の上に深さ方向の帯 (mixer / shared / PLE / LayerAct) を置く。

export const WALL_GEOM = Object.freeze({
  tileGap: 2,      // タイル同士の隙間
  pad: 2,          // タイルの内側の余白
  headH: 11,       // 層番号 + 種別バッジの行
  gapY: 2,         // ヘッダ / ヒートマップ / stream の間
  barsW: 9,        // 右側の RoleAct バーの幅
  barsGap: 2,      // ヒートマップとバーの間
  barH: 3,         // バー 1 本の高さ
  barGap: 1,       // バー同士の隙間
  streamH: 7,      // 残差 stream の四角の一辺
  streamGap: 2,
  heatCols: 32,    // expert のヒートマップ (512 = 32x16)
  heatRows: 16,
  minCell: 2,      // expert セルの最小辺 (これ未満にはしない)
  prefCols: 8,     // 広い画面での列数 (8 x 6)
  minCols: 1,
  maxCols: 16,
  depth: Object.freeze({ labelW: 26, rowH: 6, rowGap: 2, gapBelow: 5, rows: 4 }),
});

/// ページの chrome (index2d.html の CSS と同じ値。テストの見積り用)。
export const MAP_CHROME = Object.freeze({
  headerH: 42, padY: 16, padX: 20, wrapBorder: 2,
  headH: 26, statH: 15, noteH: 14,
  chatW: 340, ctrlW: 338, mapMinW: 500,
});

/// ビューポート → 地図 (canvas) の実面積。列を畳んだら opts.chat / opts.ctrl を false。
export function mapPaneSize(viewport, opts = {}) {
  const C = MAP_CHROME;
  const vw = Math.max(1, Math.floor(Number((viewport && viewport.w) || 0)));
  const vh = Math.max(1, Math.floor(Number((viewport && viewport.h) || 0)));
  const chat = opts.chat !== false;
  const ctrl = opts.ctrl !== false;
  const colW = Math.max(C.mapMinW, vw - (chat ? C.chatW : 0) - (ctrl ? C.ctrlW : 0));
  const w = Math.max(1, colW - C.padX - C.wrapBorder);
  const h = Math.max(1, vh - C.headerH - C.padY - C.headH - C.statH - C.noteH - C.wrapBorder);
  return { w: Math.floor(w), h: Math.floor(h), colW };
}

/// 候補の列数ごとの寸法。fits は「両方向のセルが minCell 以上」の意味。
function wallGrid(area, layers, heatCols, heatRows, G) {
  const out = [];
  const hi = Math.min(G.maxCols, Math.max(1, layers));
  for (let cols = G.minCols; cols <= hi; cols++) {
    const rows = Math.ceil(layers / cols);
    const tileW = (area.w - (cols - 1) * G.tileGap) / cols;
    const tileH = (area.h - (rows - 1) * G.tileGap) / rows;
    const innerW = tileW - 2 * G.pad - G.barsW - G.barsGap;
    const innerH = tileH - 2 * G.pad - G.headH - 2 * G.gapY - G.streamH;
    const cellW = innerW / heatCols;
    const cellH = innerH / heatRows;
    out.push({
      cols, rows, tileW, tileH, cellW, cellH,
      fits: tileW > 8 && tileH > 12 && cellW >= G.minCell && cellH >= G.minCell,
    });
  }
  return out;
}

const WALL_BARS = Object.freeze([
  Object.freeze({ key: 'mixer', block: 'attn', label: 'm' }),
  Object.freeze({ key: 'router', block: 'moe', label: 'rt' }),
  Object.freeze({ key: 'shared', block: 'shared', label: 'sh' }),
  Object.freeze({ key: 'ple', block: 'ple', label: 'pl' }),
]);

/// 形 + 地図の実面積 → 壁の配置。fits=false のときは width/height が pane より
/// 大きくなる (ページは mapwrap をスクロールさせる)。それでもセルは minCell 以上。
export function wallLayout(shape, pane, opts = {}) {
  const s = normaliseShape(shape);
  const G = WALL_GEOM;
  const W = Math.max(1, Math.floor(Number((pane && pane.w) || 0)));
  const H = Math.max(1, Math.floor(Number((pane && pane.h) || 0)));
  const L = s.layers;
  const heatCols = Math.max(1, Math.min(G.heatCols, s.n_expert));
  const heatRows = Math.ceil(s.n_expert / heatCols);
  const D = G.depth;
  const depthH = D.rows * D.rowH + (D.rows - 1) * D.rowGap;
  const wallY = depthH + D.gapBelow;
  const area = { x: 0, y: wallY, w: W, h: Math.max(1, H - wallY) };
  const pref = Math.max(G.minCols, Math.min(G.maxCols, intOr(opts.prefCols, G.prefCols)));

  const cands = wallGrid(area, L, heatCols, heatRows, G);
  let pick = null;
  for (const c of cands) {
    if (!c.fits) continue;
    if (!pick || Math.abs(c.cols - pref) < Math.abs(pick.cols - pref)) pick = c;
  }
  const fits = !!pick;
  if (!pick) {
    // 収まらない: セルが最大になる列数を選び、あとは minCell で床止めしてスクロール。
    pick = cands[0];
    for (const c of cands) {
      const a = Math.min(c.cellW, c.cellH);
      const b = Math.min(pick.cellW, pick.cellH);
      if (a > b + 1e-9 ||
          (Math.abs(a - b) <= 1e-9 && Math.abs(c.cols - pref) < Math.abs(pick.cols - pref))) pick = c;
    }
  }

  const cellW = Math.max(G.minCell, pick.cellW);
  const cellH = Math.max(G.minCell, pick.cellH);
  const heatW = cellW * heatCols;
  const heatH = cellH * heatRows;
  const tileW = 2 * G.pad + heatW + G.barsGap + G.barsW;
  const tileH = 2 * G.pad + G.headH + 2 * G.gapY + heatH + G.streamH;
  const needW = pick.cols * tileW + (pick.cols - 1) * G.tileGap;
  const needH = wallY + pick.rows * tileH + (pick.rows - 1) * G.tileGap;
  const width = fits ? W : Math.max(W, Math.ceil(needW));
  const height = fits ? H : Math.max(H, Math.ceil(needH));

  // バーの縦位置: ヒートマップの高さの中央に 4 本
  const barsH = WALL_BARS.length * G.barH + (WALL_BARS.length - 1) * G.barGap;
  const barTop = G.pad + G.headH + G.gapY + Math.max(0, (heatH - barsH) / 2);
  // stream の四角は幅が足りなければ縮める (下限 2px)
  const streamSize = Math.max(2, Math.min(G.streamH,
    (tileW - 2 * G.pad - (s.n_hc - 1) * G.streamGap) / s.n_hc));
  const badgeW = Math.max(14, Math.min(26, Math.round(tileW * 0.30)));
  const badgeH = G.headH - 2;

  const tiles = [];
  for (let l = 0; l < L; l++) {
    const ci = l % pick.cols;
    const ri = Math.floor(l / pick.cols);
    const x = area.x + ci * (tileW + G.tileGap);
    const y = area.y + ri * (tileH + G.tileGap);
    const head = { x: x + G.pad, y: y + G.pad, w: tileW - 2 * G.pad, h: G.headH };
    const heat = {
      x: head.x, y: head.y + G.headH + G.gapY, w: heatW, h: heatH,
      cols: heatCols, rows: heatRows, cellW, cellH, n: s.n_expert,
    };
    const badges = { x: x + tileW - G.pad - badgeW, y: y + G.pad + 1, w: badgeW, h: badgeH };
    const bars = WALL_BARS.map((b, i) => ({
      key: b.key, block: b.block, label: b.label,
      x: heat.x + heatW + G.barsGap, y: y + barTop + i * (G.barH + G.barGap),
      w: G.barsW, h: G.barH,
    }));
    const streams = [];
    for (let i = 0; i < s.n_hc; i++) {
      streams.push({
        x: heat.x + i * (streamSize + G.streamGap),
        y: y + tileH - G.pad - streamSize,
        w: streamSize, h: streamSize,
      });
    }
    tiles.push({
      layer: l, kind: s.layer_kinds[l], x, y, w: tileW, h: tileH,
      head, badge: badges, heat, bars, streams,
      pleActive: s.ple_layers.indexOf(l) >= 0,
    });
  }

  const depthW = Math.max(1, width - D.labelW - 2);
  const depth = {
    x: D.labelW, y: 0, w: depthW, h: depthH,
    cellW: depthW / L,
    rows: [
      { key: 'mixer', label: 'mix', y: 0, h: D.rowH },
      { key: 'shared', label: 'shr', y: D.rowH + D.rowGap, h: D.rowH },
      { key: 'ple', label: 'ple', y: 2 * (D.rowH + D.rowGap), h: D.rowH },
      { key: 'act', label: 'act', y: 3 * (D.rowH + D.rowGap), h: D.rowH },
    ],
  };

  return {
    shape: s, pane: { w: W, h: H }, width, height, fits,
    cols: pick.cols, rows: pick.rows, cellW, cellH, tileW, tileH,
    heatCols, heatRows, geom: G, area,
    wall: { x: 0, y: wallY, w: width, h: Math.max(1, height - wallY) },
    depth, tiles,
  };
}

/// 層 l の expert e のセル矩形 (タイル座標)。
export function wallExpertRect(layout, layer, e) {
  const t = layout.tiles[layer];
  if (!t) throw new Error('layer out of range: ' + layer);
  if (!(e >= 0 && e < t.heat.n)) throw new Error('expert out of range: ' + e);
  const ex = t.heat;
  return {
    x: ex.x + (e % ex.cols) * ex.cellW,
    y: ex.y + Math.floor(e / ex.cols) * ex.cellH,
    w: ex.cellW, h: ex.cellH,
  };
}

/// 点 → 選択できるもの。深さの帯は {type:'depth', layer, row}。
/// 何も無ければ null。
export function wallHitTest(layout, x, y) {
  const d = layout.depth;
  if (x >= d.x && x < d.x + d.w && y >= d.y && y < d.y + d.h) {
    const layer = Math.floor((x - d.x) / d.cellW);
    let row = null;
    for (const r of d.rows) if (y >= d.y + r.y && y < d.y + r.y + r.h) row = r;
    if (layer >= 0 && layer < layout.shape.layers && row) {
      return { type: 'depth', layer, row: row.key };
    }
    return null;
  }
  for (const t of layout.tiles) {
    if (x < t.x || x >= t.x + t.w || y < t.y || y >= t.y + t.h) continue;
    if (inRect(t.badge, x, y)) return { type: 'block', layer: t.layer, block: 'attn' };
    const ex = t.heat;
    if (inRect(ex, x, y)) {
      const ci = Math.floor((x - ex.x) / ex.cellW);
      const ri = Math.floor((y - ex.y) / ex.cellH);
      if (ci >= 0 && ci < ex.cols && ri >= 0 && ri < ex.rows) {
        const idx = ri * ex.cols + ci;
        if (idx >= 0 && idx < ex.n) return { type: 'expert', layer: t.layer, expert: idx };
      }
      return null;
    }
    for (const b of t.bars) {
      if (inRect(b, x, y)) return { type: 'block', layer: t.layer, block: b.block };
    }
    for (let i = 0; i < t.streams.length; i++) {
      if (inRect(t.streams[i], x, y)) return { type: 'stream', layer: t.layer, stream: i };
    }
    return null;  // ヘッダ / 余白は選ばない
  }
  return null;
}

/// 矩形 (ドラッグ) → 中の選択項目。重複は落とす。
export function wallItemsInRect(layout, x0, y0, x1, y1) {
  const ax = Math.min(x0, x1), bx = Math.max(x0, x1);
  const ay = Math.min(y0, y1), by = Math.max(y0, y1);
  const out = [];
  const seen = new Set();
  const push = (item) => {
    const k = itemKey(item);
    if (seen.has(k)) return;
    seen.add(k);
    out.push(item);
  };
  const hits = (r) => r.x < bx && r.x + r.w > ax && r.y < by && r.y + r.h > ay;
  for (const t of layout.tiles) {
    if (t.x + t.w <= ax || t.x >= bx || t.y + t.h <= ay || t.y >= by) continue;
    const ex = t.heat;
    if (hits(ex)) {
      const c0 = Math.max(0, Math.floor((ax - ex.x) / ex.cellW));
      const c1 = Math.min(ex.cols - 1, Math.floor((bx - ex.x) / ex.cellW));
      const r0 = Math.max(0, Math.floor((ay - ex.y) / ex.cellH));
      const r1 = Math.min(ex.rows - 1, Math.floor((by - ex.y) / ex.cellH));
      for (let ri = r0; ri <= r1; ri++) {
        for (let ci = c0; ci <= c1; ci++) {
          const idx = ri * ex.cols + ci;
          if (idx >= ex.n) continue;
          const cell = {
            x: ex.x + ci * ex.cellW, y: ex.y + ri * ex.cellH,
            w: ex.cellW, h: ex.cellH,
          };
          if (hits(cell)) push({ type: 'expert', layer: t.layer, expert: idx });
        }
      }
    }
    if (hits(t.badge)) push({ type: 'block', layer: t.layer, block: 'attn' });
    for (const b of t.bars) {
      if (hits(b)) push({ type: 'block', layer: t.layer, block: b.block });
    }
    for (let i = 0; i < t.streams.length; i++) {
      if (hits(t.streams[i])) push({ type: 'stream', layer: t.layer, stream: i });
    }
  }
  return out;
}

/// 深さ方向の帯の値。mixer は層種別で attn / gdn を選ぶ。
export function actLevel(act) {
  if (!act) return 0;
  let sum = 0, n = 0;
  for (let i = 0; i < act.length; i++) {
    if (act[i] > 0) { sum += act[i]; n++; }
  }
  return n ? sum / n : 0;
}

export function depthValue(st, layer, key) {
  const role = (st && st.role && st.role[layer]) || {};
  const kind = st && st.shape && st.shape.layer_kinds ? st.shape.layer_kinds[layer] : 'gdn';
  if (key === 'mixer') return roleForBlock(role, 'attn', kind);
  if (key === 'shared') return role.shex || 0;
  if (key === 'ple') return role.ple || 0;
  if (key === 'act') return actLevel(st && st.act && st.act[layer]);
  return 0;
}

/// /2d の固定色。赤は off、琥珀は skipped、白は選択 / top-k、シアンは VRAM だけ。
export const WALL_COLORS = Object.freeze({
  off: '#ff3b30',
  offFill: '#4a0d14',
  skip: '#ffc44d',
  sel: '#ffffff',
  resident: '#00eaff',
  tile: '#08131c',
  tileLine: '#12283a',
  none: '#071019',
});

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

/// off にする本文を「層あたり残る expert ≥ n_expert_used」(実 API の規則、満たさないと
/// 400) に収める。ドラッグで層のタイルをほぼ覆うと、expert が全部 / ほぼ全部選ばれて
/// この規則に当たる。
///   - 層の expert が全部 off → その層の expert を外して `moe` ブロック (routed expert
///     全体の skip) を off にする。意味は同じ (routed expert を 1 つも使わない)。
///   - 残りが 1..k-1 → 今回新しく選んだ expert のうち番号の大きい方から、足りない数だけ
///     on のまま残す (前から off のものは触らない)。
/// `prev` は今の off (applied)。戻り値 { body, notes }。notes は層ごとの説明。
export function fitExpertLimit(body, shape, prev) {
  const s = normaliseShape(shape);
  const k = s.n_expert_used | 0;
  const ne = s.n_expert | 0;
  const out = {
    experts_off: (body.experts_off || []).map((p) => [p[0], p[1]]),
    blocks_off: (body.blocks_off || []).map((p) => [p[0], p[1]]),
    streams_off: (body.streams_off || []).map((p) => [p[0], p[1]]),
  };
  const notes = [];
  if (!(k > 0 && ne > 0) || out.experts_off.length === 0) return { body: out, notes };
  const before = new Set((prev && prev.experts_off ? prev.experts_off : []).map((p) => p[0] + ':' + p[1]));
  const byLayer = new Map();
  for (const [l, e] of out.experts_off) {
    if (!byLayer.has(l)) byLayer.set(l, []);
    byLayer.get(l).push(e);
  }
  const drop = new Set();
  for (const [l, list] of byLayer) {
    const remaining = ne - list.length;
    if (remaining >= k) continue;
    if (remaining <= 0) {
      for (const e of list) drop.add(l + ':' + e);
      if (!out.blocks_off.some((b) => b[0] === l && b[1] === 'moe')) out.blocks_off.push([l, 'moe']);
      notes.push('層 ' + l + ': expert を全部選んだので moe ブロックを off にした');
      continue;
    }
    const need = k - remaining;
    const fresh = list.filter((e) => !before.has(l + ':' + e)).sort((a, b) => b - a);
    const keep = fresh.slice(0, need);
    for (const e of keep) drop.add(l + ':' + e);
    notes.push('層 ' + l + ': 最低 ' + k + ' 個の expert が要るので ' + keep.length + ' 個を on のまま残した (' +
      keep.slice().sort((a, b) => a - b).join(',') + ')');
  }
  if (drop.size) {
    out.experts_off = out.experts_off.filter((p) => !drop.has(p[0] + ':' + p[1]));
    out.blocks_off.sort((a, b) => a[0] - b[0] || BLOCKS.indexOf(a[1]) - BLOCKS.indexOf(b[1]));
  }
  notes.sort((a, b) => Number(a.split(' ')[1]) - Number(b.split(' ')[1]));
  return { body: out, notes };
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

/// task 220: ExpertSkip → expert ごとの 0/1 (ルータが選びたかったが介入で切った)。
export function skipBits(frame, n_expert) {
  const out = new Uint8Array(n_expert);
  if (!frame || frame.kind !== 'ExpertSkip') return out;
  for (const e of frame.experts || []) if (e >= 0 && e < n_expert) out[e] = 1;
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

/// 0..1 を色に。task 235: /2d は「暗い → 明るい」の 1 本の尺度にする
/// (赤は off、琥珀は skipped だけに使うので、活動の上限はほぼ白)。
/// 明るさは単調増加 (0.0 約 17 → 1.0 約 240)。
const RAMP = [
  [8, 19, 28],
  [12, 58, 78],
  [0, 150, 178],
  [0, 214, 240],
  [214, 250, 255],
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
///
/// task 235: ツールチップと top-k の輪郭のために 3 本の配列を足した。
///   st.last[l]   — 最後のトークンの正規化重み (mode に依らない)
///   st.accum[l]  — decay 付きの選択回数 (mode に依らない。accum モードでは
///                  st.experts[l] がこの配列そのものを指す)
///   st.topk[l]   — 最後のトークンでルータが選んだ expert のビット
export function createMapState(shape, opts = {}) {
  const s = normaliseShape(shape);
  const decay = Number.isFinite(opts.decay) ? opts.decay : 0.9;
  const st = {
    shape: s,
    mode: opts.mode === 'accum' ? 'accum' : 'token',
    decay,
    token: 0,
    experts: [],
    accum: [],
    last: [],
    topk: [],
    residency: [],
    skip: [],
    role: [],
    act: [],
    lastToken: null,
    frames: 0,
  };
  for (let l = 0; l < s.layers; l++) {
    st.accum.push(new Float32Array(s.n_expert));
    // accum モードでは表示用の配列を累積そのものにして、2 重に持たない。
    st.experts.push(st.mode === 'accum' ? st.accum[l] : new Float32Array(s.n_expert));
    st.last.push(new Float32Array(s.n_expert));
    st.topk.push(new Uint8Array(s.n_expert));
    st.residency.push(new Uint8Array(s.n_expert));
    st.skip.push(new Uint8Array(s.n_expert));
    st.role.push({ attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 });
    st.act.push(new Float32Array(ACT_BUCKETS));
  }

  st.setMode = (mode) => {
    const m = mode === 'accum' ? 'accum' : 'token';
    if (m === st.mode) return;
    st.mode = m;
    for (let l = 0; l < s.layers; l++) {
      // accum → token は表示だけ空にする (累積はツールチップ用に残す)
      st.experts[l] = m === 'accum' ? st.accum[l] : new Float32Array(s.n_expert);
    }
  };

  st.reset = () => {
    st.token = 0;
    st.frames = 0;
    st.lastToken = null;
    for (let l = 0; l < s.layers; l++) {
      st.accum[l].fill(0);
      if (st.experts[l] !== st.accum[l]) st.experts[l].fill(0);
      st.last[l].fill(0);
      st.topk[l].fill(0);
      st.residency[l].fill(0);
      st.skip[l].fill(0);
      st.role[l] = { attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 };
      st.act[l].fill(0);
    }
  };

  st.onFrame = (f) => {
    if (!f || typeof f !== 'object') return false;
    if (f.kind === 'Token') {
      st.token++;
      st.lastToken = f;
      // task 235: 累積は mode に依らず常に decay する (ツールチップの count 用)。
      for (let i = 0; i < s.layers; i++) {
        const a = st.accum[i];
        for (let j = 0; j < a.length; j++) if (a[j] !== 0) a[j] *= decay;
      }
      return true;
    }
    if (f.layer == null || f.layer >= s.layers) return false;
    const l = f.layer | 0;
    switch (f.kind) {
      case 'RouterTopK': {
        const w = routerWeights(f, s.n_expert);
        // task 220: この層のルーティングが確定したので、前のトークンの
        // 「選びたかったが切った」は捨てる (ExpertSkip は直後に来る。空なら
        // そのトークンでは誰も選びたくなかった = amber を消す)。
        st.skip[l].fill(0);
        st.last[l].set(w);
        const top = st.topk[l];
        top.fill(0);
        const acc = st.accum[l];
        for (let i = 0; i < w.length; i++) {
          if (w[i] > 0) {
            top[i] = 1;
            acc[i] += w[i];
          }
        }
        if (st.mode === 'token') st.experts[l].set(w);
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
      case 'ExpertSkip': {
        // task 220: ルータが選びたかったが介入で切った expert。層ごとに置き換える
        // (RouterTopK と同じ「最後の 1 トークン」の意味)。
        st.skip[l].set(skipBits(f, s.n_expert));
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
///
/// 実 API (task 218) の規則に合わせる:
/// - `terms[].coef` は有限なら 0 も負も可 (0 は 1 ビットも変えない)
/// - `alpha` は有限なら負も可 (実 API は符号を見ない)
/// - `layers` は `[from, to)`。`to` は層数まで (含まない)。`from == to` は空範囲で可
/// - `opts.nLayers` を渡すと `to > nLayers` を弾く (実 API は 400)
export function steerApplyBody(spec, opts = {}) {
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
    if (!Number.isFinite(coef)) {
      throw new Error('terms[' + i + '].coef must be a finite number');
    }
    return { snapshot: t.snapshot, coef };
  });
  const layers = Array.isArray(src.layers) ? src.layers : [0, 0];
  if (layers.length !== 2) throw new Error('layers: [from, to] required');
  const from = intOr(layers[0], -1);
  const to = intOr(layers[1], -1);
  if (from < 0 || to < 0) throw new Error('layers: from/to must be >= 0');
  if (from > to) throw new Error('layers: from must be <= to');
  const nLayers = intOr(opts.nLayers, 0);
  if (nLayers > 0 && to > nLayers) {
    throw new Error('layers: to must be <= ' + nLayers + ' ([from, to) の to は層数まで)');
  }
  const alpha = src.alpha == null ? 1 : Number(src.alpha);
  if (!Number.isFinite(alpha)) throw new Error('alpha: must be a finite number');
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

/// モックモードの合成テレメトリ (task 235)。サーバー無しでも壁を動かして
/// 目で確認できるように、1 トークン分のフレームを決定的に作る。
/// opts.off に [[layer, expert], ...] を渡すと、その expert を ExpertSkip として
/// 送る (介入の amber をモックでも再現する)。
export function mockTokenFrames(shape, n, opts = {}) {
  const s = normaliseShape(shape);
  const seed0 = (intOr(opts.seed, 0) >>> 0) ^ Math.imul((n | 0) + 1, 2654435761);
  let x = seed0 >>> 0 || 1;
  const rnd = () => {
    x ^= x << 13; x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5; x >>>= 0;
    return x / 4294967296;
  };
  const off = new Set((opts.off || []).map((p) => (p[0] | 0) + ':' + (p[1] | 0)));
  const k = Math.max(1, Math.min(s.n_expert, s.n_expert_used));
  const frames = [{ kind: 'Token', session: 1, token: n | 0, pos: n | 0 }];

  for (let l = 0; l < s.layers; l++) {
    const kind = s.layer_kinds[l];
    const mix = 0.35 + rnd() * 0.65;
    frames.push({
      kind: 'RoleAct', layer: l,
      roles: {
        attn: kind === 'attn' ? mix : 0,
        gdn: kind === 'gdn' ? mix : 0,
        router: 0.3 + rnd() * 0.7,
        shex: 0.2 + rnd() * 0.8,
        ple: s.ple_layers.indexOf(l) >= 0 ? 0.4 + rnd() * 0.6 : 0,
      },
    });
    // 512 から重複なしで top-k を選ぶ
    const chosen = new Set();
    while (chosen.size < k) chosen.add(Math.floor(rnd() * s.n_expert));
    const experts = [...chosen].sort((a, b) => a - b);
    const weights = experts.map(() => 0.2 + rnd() * 0.8);
    const keepE = [], keepW = [], skipE = [];
    for (let i = 0; i < experts.length; i++) {
      if (off.has(l + ':' + experts[i])) skipE.push(experts[i]);
      else { keepE.push(experts[i]); keepW.push(weights[i]); }
    }
    frames.push({ kind: 'RouterTopK', layer: l, experts: keepE, weights: keepW });
    if (skipE.length) frames.push({ kind: 'ExpertSkip', layer: l, experts: skipE });
    const idx = [], val = [];
    for (let i = 0; i < 96; i++) {
      idx.push(Math.floor(rnd() * s.n_embd));
      val.push(rnd() * rnd());
    }
    frames.push({ kind: 'LayerAct', layer: l, idx, val });
  }

  // VRAM 常駐は形だけで決まる (トークンで変わらない)。
  let y = ((Math.imul(s.n_expert, 2246822519) ^ Math.imul(s.layers, 3266489917)) >>> 0) || 1;
  const rnd2 = () => {
    y ^= y << 13; y >>>= 0;
    y ^= y >>> 17;
    y ^= y << 5; y >>>= 0;
    return y / 4294967296;
  };
  const vram = [];
  for (let e = 0; e < s.n_expert; e++) if (rnd2() < 0.35) vram.push(e);
  for (let l = 0; l < s.layers; l++) frames.push({ kind: 'ExpertResidency', layer: l, vram });
  return frames;
}

function jsonClone(v) {
  return v == null ? v : JSON.parse(JSON.stringify(v));
}

/// 実 API (task 217/218) の検証と同じ規則。エラーは 400 で
/// `{type:'error', error:{type:'invalid_request_error', message}}`。
function apiError(message) {
  return { type: 'error', error: { type: 'invalid_request_error', message } };
}

/// ステアリングの snapshot 名 (engine の `steer::validate_name` と同じ)。
const NAME_RE = /^[A-Za-z0-9_-]{1,64}$/;

/// shape が 404 のときの API 代役。`handle(method, path, body)` -> `{status, body}`。
///
/// **実 API と同じ規則**にしてある (task 220 で差分を全部寄せた):
/// - `POST /v1/intervene` は 3 配列の全置換。未知フィールド・範囲外・層あたりの
///   残り expert < n_expert_used は 400。並びは (層, 種別) の昇順・重複なし。
/// - epoch は介入の set / clear、ステアリングの apply / clear、適用中の snapshot の
///   削除で 1 増える (記録の開始 / 停止では増えない)。
/// - ステアリングのエラーは全部 400 (実 API は 404/409 を使わない)。
/// - 同名 snapshot は上書きできる (engine の `push_snapshot` が名前で置換する)。
/// - snapshot の `layers` は層数 (数値)。apply の `layers` は `[from, to)` の配列。
export function createMockStore(shape, opts = {}) {
  const s = normaliseShape(shape || DEFAULT_SHAPE_RAW);
  // 実 API の `created` は UNIX 秒 (engine の `now_secs`)。ISO 文字列ではない。
  const now = typeof opts.now === 'function' ? opts.now : () => Math.floor(Date.now() / 1000);
  const st = {
    epoch: 0,
    experts_off: [],
    blocks_off: [],
    streams_off: [],
    recording: null,
    snapshots: [],
    apply: null,
    tokens: 0,
    // 実 API の main 文脈のトークン数 (GET / DELETE /v1/cache)。
    ctx_tokens: 0,
  };
  let tick = 0;

  function err(status, message) {
    return { status, body: apiError(message) };
  }

  function validIntervene(body) {
    if (!body || typeof body !== 'object' || Array.isArray(body)) return 'body must be an object';
    for (const k of Object.keys(body)) {
      if (['experts_off', 'blocks_off', 'streams_off'].indexOf(k) < 0) {
        return 'unknown field ' + JSON.stringify(k) + ' (experts_off / blocks_off / streams_off only)';
      }
    }
    const lists = [['experts_off', 2], ['blocks_off', 2], ['streams_off', 2]];
    for (const [name, arity] of lists) {
      const v = body[name] == null ? [] : body[name];
      if (!Array.isArray(v)) return name + ' must be an array';
      for (const entry of v) {
        if (!Array.isArray(entry) || entry.length !== arity) return name + ' entries must be [..] pairs';
      }
    }
    for (const [l, e] of body.experts_off || []) {
      if (!Number.isInteger(l) || !Number.isInteger(e)) return 'experts_off: entries must be integers';
      if (!(l >= 0 && l < s.layers)) return 'experts_off: layer out of range: ' + l;
      if (!(e >= 0 && e < s.n_expert)) return 'experts_off: expert out of range: ' + e;
    }
    for (const [l, name] of body.blocks_off || []) {
      if (!Number.isInteger(l)) return 'blocks_off: layer must be an integer';
      if (!(l >= 0 && l < s.layers)) return 'blocks_off: layer out of range: ' + l;
      if (BLOCKS.indexOf(name) < 0) return 'blocks_off: unknown block: ' + name;
    }
    for (const [l, x] of body.streams_off || []) {
      if (!Number.isInteger(l) || !Number.isInteger(x)) return 'streams_off: entries must be integers';
      if (!(l >= 0 && l < s.layers)) return 'streams_off: layer out of range: ' + l;
      if (!(x >= 0 && x < s.n_hc)) return 'streams_off: stream out of range: ' + x;
    }
    // 層あたり「残る expert」が top-k 未満だとルータが k 個選べない (実 API も 400)。
    if ((body.experts_off || []).length && s.n_expert_used > 0) {
      const cnt = new Array(Math.max(1, s.layers)).fill(0);
      for (const [l] of body.experts_off) cnt[l] += 1;
      for (let l = 0; l < s.layers; l++) {
        if (s.n_expert - cnt[l] < s.n_expert_used) {
          return 'layer ' + l + ': remaining experts ' + (s.n_expert - cnt[l]) +
            ' < n_expert_used ' + s.n_expert_used;
        }
      }
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

  function status() {
    return {
      epoch: st.epoch,
      recording: jsonClone(st.recording),
      snapshots: jsonClone(st.snapshots),
      apply: jsonClone(st.apply),
    };
  }

  function clearApply() {
    st.apply = null;
    st.epoch++;
  }

  function ctxStatus() {
    return {
      main: st.ctx_tokens > 0 ? { tokens: st.ctx_tokens, epoch: st.epoch } : null,
      aux: null,
    };
  }

  const store = {
    shape: s,
    state: () => jsonClone(store.snapshot()),
    snapshot: () => ({
      shape: {
        layers: s.layers, layer_kinds: [...s.layer_kinds], n_expert: s.n_expert,
        n_expert_used: s.n_expert_used, n_hc: s.n_hc, n_embd: s.n_embd,
        ple_layers: [...s.ple_layers],
      },
      epoch: st.epoch,
      intervene: {
        epoch: st.epoch, experts_off: st.experts_off,
        blocks_off: st.blocks_off, streams_off: st.streams_off,
      },
      steer: status(),
      tokens: st.tokens,
    }),

    /// Token フレームのたびに呼ぶ。記録中ならトークン数を進める。
    tick(n = 1) {
      tick += n;
      if (st.recording) st.recording.tokens += n;
      st.tokens += n;
      st.ctx_tokens += n;
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
          body: {
            epoch: st.epoch, experts_off: jsonClone(st.experts_off),
            blocks_off: jsonClone(st.blocks_off), streams_off: jsonClone(st.streams_off),
          },
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
        return {
          status: 200,
          body: {
            epoch: st.epoch, experts_off: jsonClone(st.experts_off),
            blocks_off: jsonClone(st.blocks_off), streams_off: jsonClone(st.streams_off),
          },
        };
      }
      if (m === 'DELETE' && p === '/v1/intervene') {
        st.experts_off = [];
        st.blocks_off = [];
        st.streams_off = [];
        st.epoch++;
        return {
          status: 200,
          body: { epoch: st.epoch, experts_off: [], blocks_off: [], streams_off: [] },
        };
      }
      if (m === 'POST' && p === '/v1/abort') {
        // モックには走っている生成が無い (実 API も生成中でなければ空振り)。
        return { status: 200, body: { aborted: false, generating: false } };
      }
      if (m === 'GET' && p === '/v1/cache') {
        return { status: 200, body: ctxStatus() };
      }
      if (m === 'DELETE' && p === '/v1/cache') {
        const dropped = st.ctx_tokens > 0 ? 1 : 0;
        const tokens = st.ctx_tokens;
        st.ctx_tokens = 0;
        // 実 API と同じく介入 epoch は動かさない。
        return { status: 200, body: Object.assign(ctxStatus(), { dropped, dropped_tokens: tokens }) };
      }
      if (m === 'GET' && p === '/v1/steer') {
        return { status: 200, body: status() };
      }
      if (m === 'POST' && p === '/v1/steer/record') {
        if (!body || typeof body !== 'object' || Array.isArray(body)) return err(400, 'body must be an object');
        for (const k of Object.keys(body)) {
          if (k !== 'name') return err(400, 'unknown field ' + JSON.stringify(k) + ' (name only)');
        }
        const name = typeof body.name === 'string' ? body.name : '';
        if (!name) return err(400, 'name is required');
        if (!NAME_RE.test(name)) return err(400, 'name: A-Z a-z 0-9 _ - only, at most 64 chars');
        if (st.recording) return err(400, 'already recording: ' + st.recording.name);
        st.recording = { name, tokens: 0, created: now() };
        return { status: 200, body: status() };
      }
      if (m === 'POST' && p === '/v1/steer/record/stop') {
        if (!st.recording) return err(400, 'not recording');
        const rec = st.recording;
        st.recording = null;
        // engine は 0 トークンの平均を保存しない (エラーにして記録も落とす)。
        if (!(rec.tokens > 0)) return err(400, 'recorded 0 tokens');
        const snap = { name: rec.name, tokens: rec.tokens, created: now(), layers: s.layers };
        const i = st.snapshots.findIndex((x) => x.name === snap.name);
        if (i >= 0) st.snapshots[i] = snap;
        else st.snapshots.push(snap);
        st.snapshots.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
        const out = status();
        out.saved = jsonClone(snap);
        return { status: 200, body: out };
      }
      if (m === 'DELETE' && p.indexOf('/v1/steer/snapshots/') === 0) {
        const name = decodeURIComponent(p.slice('/v1/steer/snapshots/'.length));
        if (!NAME_RE.test(name)) return err(400, 'name: A-Z a-z 0-9 _ - only, at most 64 chars');
        const i = st.snapshots.findIndex((x) => x.name === name);
        if (i < 0) return err(400, 'no such snapshot: ' + name);
        st.snapshots.splice(i, 1);
        // 適用中の snapshot を消すと適用は落ちる (engine の `delete` は clear_apply する)。
        if (st.apply && (st.apply.terms || []).some((t) => t.snapshot === name)) clearApply();
        return { status: 200, body: status() };
      }
      if (m === 'POST' && p === '/v1/steer/apply') {
        if (!body || typeof body !== 'object' || Array.isArray(body)) return err(400, 'body must be an object');
        for (const k of Object.keys(body)) {
          if (['terms', 'layers', 'alpha', 'mode'].indexOf(k) < 0) {
            return err(400, 'unknown field ' + JSON.stringify(k) + ' (terms / layers / alpha / mode only)');
          }
        }
        if (!Array.isArray(body.terms) || body.terms.length === 0) {
          return err(400, 'terms must be a non-empty array (use DELETE /v1/steer/apply to clear)');
        }
        for (let i = 0; i < body.terms.length; i++) {
          const t = body.terms[i];
          if (!t || typeof t !== 'object' || Array.isArray(t)) return err(400, 'terms[' + i + '] must be an object');
          for (const k of Object.keys(t)) {
            if (['snapshot', 'coef'].indexOf(k) < 0) return err(400, 'terms[' + i + ']: unknown field ' + JSON.stringify(k));
          }
          if (typeof t.snapshot !== 'string' || !t.snapshot) return err(400, 'terms[' + i + '].snapshot must be a string');
          if (!Number.isFinite(Number(t.coef))) return err(400, 'terms[' + i + '].coef must be a finite number');
        }
        if (!Array.isArray(body.layers) || body.layers.length !== 2) return err(400, 'layers must be [from, to]');
        const from = Number(body.layers[0]);
        const to = Number(body.layers[1]);
        if (!Number.isInteger(from) || !Number.isInteger(to) || from < 0 || to < 0) {
          return err(400, 'layers must be non-negative integers');
        }
        if (from > to) return err(400, 'layers: from must be <= to');
        if (to > s.layers) return err(400, 'layers: to must be <= ' + s.layers);
        const alpha = body.alpha == null ? 1 : Number(body.alpha);
        if (!Number.isFinite(alpha)) return err(400, 'alpha must be a finite number');
        const mode = body.mode == null ? 'subtract' : body.mode;
        if (STEER_MODES.indexOf(mode) < 0) return err(400, 'mode: subtract or project');
        for (const t of body.terms) {
          if (!st.snapshots.some((x) => x.name === t.snapshot)) {
            return err(400, 'no such snapshot: ' + t.snapshot);
          }
        }
        st.apply = {
          terms: body.terms.map((t) => ({ snapshot: t.snapshot, coef: Number(t.coef) })),
          layers: [from, to], alpha, mode,
        };
        st.epoch++;
        return { status: 200, body: status() };
      }
      if (m === 'DELETE' && p === '/v1/steer/apply') {
        clearApply();
        return { status: 200, body: status() };
      }
      return { status: 404, body: apiError('not found: ' + m + ' ' + p) };
    },
  };
  return store;
}
