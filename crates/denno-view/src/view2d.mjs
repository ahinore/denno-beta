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
//   GET    /v1/cache       -> {main:{tokens,epoch}|null, aux:{...}|null} / DELETE で全消し
//   GET    /v1/experts/stats -> {enabled, file, last_save, recorded, vram, ram, mmap,
//                                coverage, last_request, plan} (task 271)
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
const ROLE_KEYS = Object.freeze(['attn', 'gdn', 'router', 'shex', 'ple']);

function zeroRoleCount() {
  return { attn: 0, gdn: 0, router: 0, shex: 0, ple: 0, any: 0 };
}

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
    // 構成図 / ノード選択 DSL 用: リセットからの「活性化した回数」(decay なし)。
    //   count[l][e]   — expert e がルータの top-k に入ったトークン数
    //   roleCount[l]  — RoleAct の各役割が 0 より大きかったトークン数 (any はどれか 1 つ)
    count: [],
    roleCount: [],
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
    st.count.push(new Uint32Array(s.n_expert));
    st.roleCount.push(zeroRoleCount());
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
      st.count[l].fill(0);
      st.roleCount[l] = zeroRoleCount();
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
            st.count[l][i]++;
          }
        }
        if (st.mode === 'token') st.experts[l].set(w);
        st.frames++;
        return true;
      }
      case 'RoleAct': {
        st.role[l] = Object.assign({ attn: 0, gdn: 0, router: 0, shex: 0, ple: 0 }, f.roles || {});
        const rc = st.roleCount[l];
        let any = false;
        for (const k of ROLE_KEYS) {
          if (Number(st.role[l][k]) > 0) { rc[k]++; any = true; }
        }
        if (any) rc.any++;
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

// ---- abliteration の作業台 (task 825) ------------------------------------
//
// `/2d` の Abliterate タブの純ロジック。サーバーの `/v1/steer/analyze` が返す
// 層ごとの指標を、分離チャート・色モード・適用本文・比較表示に落とす。
// どれも DOM に触らない (node --test から直接呼ぶ)。

/// 対比に使うスナップショット名 (画面の既定)。
export const ABLIT_A = 'ablit_A';
export const ABLIT_B = 'ablit_B';
/// alpha スライダの範囲。
export const ABLIT_ALPHA_MIN = 0;
export const ABLIT_ALPHA_MAX = 2;

// ---- 残差の取り方 (task 829) --------------------------------------------
//
// `POST /v1/steer/record` の `capture_mode`。`all` は全プロンプトトークンの
// 平均 (従来)、`last_prompt_token` は prefill の最終トークンだけ。
// Abliterate タブの既定は `last_prompt_token` (difference-in-means の信号が濃い)。

/// 実 API が受ける値 (これ以外は 400)。
export const CAPTURE_MODES = ['all', 'last_prompt_token'];
/// Abliterate タブの既定。
export const ABLIT_CAPTURE_MODE = 'last_prompt_token';

/// `/v1/steer/record` の本文。name と capture_mode を実 API の規則で検証する。
export function ablitRecordBody(name, captureMode) {
  const s = typeof name === 'string' ? name : '';
  if (!s) throw new Error('name: required');
  if (!NAME_RE.test(s)) throw new Error('name: A-Z a-z 0-9 _ - only, at most 64 chars');
  const m = captureMode == null ? ABLIT_CAPTURE_MODE : String(captureMode);
  if (CAPTURE_MODES.indexOf(m) < 0) {
    throw new Error('capture_mode: ' + CAPTURE_MODES.join(' / ') + ' (got ' + m + ')');
  }
  return { name: s, capture_mode: m };
}

/// テキストエリアの改行区切り → プロンプトの配列 (空行は捨てる)。
export function parsePromptList(text, opts = {}) {
  if (typeof text !== 'string') return [];
  const max = Math.max(1, intOr(opts.max, 512));
  const out = [];
  for (const line of text.split(/\r?\n/)) {
    const p = line.trim();
    if (!p) continue;
    out.push(p);
    if (out.length >= max) break;
  }
  return out;
}

/// 分析結果 → 層ごとの分離スコア (0..1)。足りない層は 0。
export function ablitSepScores(analysis, nLayers) {
  const n = Math.max(0, intOr(nLayers, 0));
  const out = new Array(n).fill(0);
  const arr = analysis && Array.isArray(analysis.layers) ? analysis.layers : [];
  for (const l of arr) {
    const i = intOr(l && l.layer, -1);
    if (i < 0 || i >= out.length) continue;
    const v = Number(l.sep_score);
    out[i] = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
  }
  return out;
}

/// 分離スコアの配列 → 推奨区間 `[from, to)`。
/// engine の `steer::recommended_range` と同じ規則 (層 0 と最終層を避け、
/// ピークの半値以上が続く連続窓)。
export function ablitRecommendedRange(seps) {
  const arr = Array.isArray(seps) ? seps : [];
  const n = arr.length;
  if (!n) return [0, 0];
  if (n < 3) return [0, n];
  const lo = 1;
  const hi = n - 1;
  let peak = lo;
  let best = Number(arr[lo]) || 0;
  for (let i = lo; i < hi; i++) {
    const v = Number(arr[i]) || 0;
    if (v > best) { best = v; peak = i; }
  }
  if (!(best > 0)) return [lo, hi];
  const thr = best * 0.5;
  let from = peak;
  while (from > lo && (Number(arr[from - 1]) || 0) >= thr) from--;
  let to = peak + 1;
  while (to < hi && (Number(arr[to]) || 0) >= thr) to++;
  return [from, to];
}

/// 分析結果の推奨区間 (層数で丸める)。
export function ablitRange(analysis, nLayers) {
  const n = Math.max(0, intOr(nLayers, 0));
  const r = analysis && Array.isArray(analysis.recommended_range) ? analysis.recommended_range : null;
  if (!r || r.length !== 2) return [0, n];
  let f = Math.max(0, Math.min(n, intOr(r[0], 0)));
  const t = Math.max(f, Math.min(n, intOr(r[1], 0)));
  return [f, t];
}

/// 分離チャートのバー (層の数だけ)。`value` は 0..1 の生値、`h` は max で正規化。
export function ablitChart(analysis, rect, range, nLayers) {
  const n = Math.max(0, intOr(nLayers, 0));
  const R = rect && typeof rect === 'object' ? rect : { x: 0, y: 0, w: 1, h: 1 };
  const vals = ablitSepScores(analysis, n);
  const max = vals.reduce((a, v) => Math.max(a, v), 0);
  const rr = Array.isArray(range) ? range : ablitRange(analysis, n);
  const from = intOr(rr[0], 0);
  const to = intOr(rr[1], 0);
  const step = n > 0 ? R.w / n : 0;
  const gap = Math.min(2, step * 0.25);
  const bw = Math.max(1, step - gap);
  const peak = intOr(analysis && analysis.peak_layer, -1);
  const bars = vals.map((v, i) => {
    const h = max > 0 ? Math.max(v > 0 ? 1 : 0, (v / max) * R.h) : 0;
    return {
      layer: i,
      x: R.x + i * step,
      y: R.y + R.h - h,
      w: bw,
      h,
      value: v,
      selected: i >= from && i < to,
      peak: i === peak,
    };
  });
  return { bars, max, range: [from, to], n };
}

/// チャート上の x 座標 → 層 index。外は null。
export function ablitLayerAt(chart, x) {
  if (!chart || !Array.isArray(chart.bars) || !chart.bars.length) return null;
  const b0 = chart.bars[0];
  const bN = chart.bars[chart.bars.length - 1];
  const x0 = b0.x;
  const x1 = bN.x + bN.w;
  if (x < x0 || x > x1 || x1 <= x0) return null;
  const idx = Math.floor((x - x0) / ((x1 - x0) / chart.bars.length));
  return Math.max(0, Math.min(chart.bars.length - 1, idx));
}

/// ドラッグ選択 (anchor → いまの層) → `[from, to)`。同じ層なら 1 層。
export function ablitRangeFromDrag(anchor, layer) {
  const a = Math.max(0, intOr(anchor, 0));
  const b = Math.max(0, intOr(layer, 0));
  return [Math.min(a, b), Math.max(a, b) + 1];
}

/// ワンクリック適用の本文 (A +1 / B -1 / project)。
export function ablitApplyBody(range, alpha, opts = {}) {
  const nLayers = Math.max(0, intOr(opts.nLayers, 0));
  const r = Array.isArray(range) ? range : [];
  if (r.length !== 2) throw new Error('range: [from, to] required');
  const from = intOr(r[0], -1);
  const to = intOr(r[1], -1);
  if (from < 0 || to < 0) throw new Error('range: from/to must be >= 0');
  if (from >= to) throw new Error('range: from must be < to (空の範囲は掛けない)');
  if (nLayers > 0 && to > nLayers) throw new Error('range: to must be <= ' + nLayers);
  const a = alpha == null ? 1 : Number(alpha);
  if (!Number.isFinite(a)) throw new Error('alpha: must be a finite number');
  if (a < ABLIT_ALPHA_MIN || a > ABLIT_ALPHA_MAX) {
    throw new Error('alpha: ' + ABLIT_ALPHA_MIN + ' .. ' + ABLIT_ALPHA_MAX + ' (got ' + a + ')');
  }
  return steerApplyBody({
    terms: [
      { snapshot: opts.a || ABLIT_A, coef: 1 },
      { snapshot: opts.b || ABLIT_B, coef: -1 },
    ],
    layers: [from, to],
    alpha: a,
    mode: 'project',
  }, { nLayers });
}

/// Hybrid block (4 層) の分離スコア = その 4 層の平均。
export function ablitHybridValue(scores, hybrid, per = 4) {
  const arr = Array.isArray(scores) ? scores : [];
  const p = Math.max(1, intOr(per, 4));
  const h = intOr(hybrid, -1);
  if (h < 0) return 0;
  let s = 0;
  let n = 0;
  for (let l = h * p; l < (h + 1) * p && l < arr.length; l++) {
    s += Number(arr[l]) || 0;
    n++;
  }
  return n ? s / n : 0;
}

/// エクスポートの URL とファイル名 (ブラウザのダウンロード用)。
export function ablitExportUrl(name, format) {
  const q = '?name=' + encodeURIComponent(String(name || ''));
  return '/api/v1/steer/export' + q + (format === 'json' ? '&format=json' : '');
}

export function ablitExportName(name, format) {
  const n = String(name || 'snapshot');
  return format === 'json' ? n + '.dnsteer.json' : n + '.dns';
}

/// 取り込む JSON テキストの検証 (実 API の `import_json` と同じ規則)。不正は例外。
export function ablitImportBody(text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('file is empty');
  let v;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new Error('not JSON: ' + ((e && e.message) ? e.message : String(e)));
  }
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('JSON must be an object');
  if (v.data_base64 == null) {
    if (!Array.isArray(v.mean)) throw new Error('mean or data_base64 required');
    const layers = intOr(v.layers, 0);
    const nHc = intOr(v.n_hc, 0);
    const nEmbd = intOr(v.n_embd, 0);
    if (!(layers > 0 && nHc > 0 && nEmbd > 0)) throw new Error('layers / n_hc / n_embd required');
    if (v.mean.length !== layers * nHc * nEmbd) throw new Error('mean length mismatch');
  } else if (typeof v.data_base64 !== 'string') {
    throw new Error('data_base64 must be a string');
  }
  if (v.name != null && !NAME_RE.test(String(v.name))) {
    throw new Error('name: A-Z a-z 0-9 _ - only');
  }
  return v;
}

/// 分離チャートの 1 行要約 ("peak layer 21 · sep 0.433 · range 12-30")。
export function ablitSummary(analysis) {
  if (!analysis || !Array.isArray(analysis.layers) || !analysis.layers.length) return '';
  const n = analysis.layers.length;
  const rr = ablitRange(analysis, n);
  const pk = intOr(analysis.peak_layer, -1);
  const row = pk >= 0 && pk < n ? analysis.layers.find((l) => intOr(l && l.layer, -1) === pk) : null;
  const sep = row ? Number(row.sep_score) : NaN;
  return 'peak layer ' + pk + (Number.isFinite(sep) ? ' · sep ' + sep.toFixed(3) : '') +
    ' · range ' + rr[0] + '-' + rr[1];
}

/// A/B の比較の行 (OFF と ON を並べる)。長すぎる出力は末尾を落とす。
export function ablitCompareRows(offText, onText, opts = {}) {
  const max = Math.max(16, intOr(opts.max, 4000));
  const cut = (t) => {
    const s = typeof t === 'string' ? t : '';
    return s.length > max ? s.slice(0, max) + '…' : s;
  };
  return [
    { label: 'OFF', text: cut(offText) },
    { label: 'ON', text: cut(onText) },
  ];
}

// ---- バッチ評価 (task 833) ----------------------------------------------
//
// `compare off vs on` を 1 プロンプトから 1 行 1 プロンプトの一括評価へ広げる。
// 実行 (OFF を全部 → ON を全部、進捗とキャンセル) は DOM 側
// (`index2d.html`) が持ち、ここは行の組み立てと表の文字列化だけを持つ。

/// 一括評価のプロンプト数の上限 (画面は 3-10 を想定)。
export const ABLIT_EVAL_MAX = 10;
/// 表に載せる応答の最大文字数 (1 行に潰す)。
export const ABLIT_EVAL_SNIPPET = 120;

/// テキストエリア → 評価するプロンプト (空行は捨て、上限で切る)。
export function ablitEvalPrompts(text, opts = {}) {
  return parsePromptList(text, { max: intOr(opts.max, ABLIT_EVAL_MAX) });
}

/// 拒否っぽい応答か (素朴なヒューリスティック。判定ではない)。
const ABLIT_REFUSAL_RE =
  /(i can(?:no|')?t|i cannot|i'm sorry|i am sorry|as an ai|i won't|i will not|i'm not able|i am not able|cannot assist|can't assist|i must decline|unable to help|not able to help)/i;

export function ablitRefusalLike(text) {
  return typeof text === 'string' && ABLIT_REFUSAL_RE.test(text);
}

/// 応答を 1 行の抜粋にする (改行は空白へ、長すぎれば … で切る)。
export function ablitSnippet(text, max) {
  const s = typeof text === 'string' ? text : '';
  const one = s.replace(/\s+/g, ' ').trim();
  const m = Math.max(8, intOr(max, ABLIT_EVAL_SNIPPET));
  return one.length > m ? one.slice(0, m) + '…' : one;
}

/// 出力トークン数。分からなければ文字数で代用する (表示用)。
function evalLen(text, tokens) {
  const t = Number(tokens);
  if (Number.isFinite(t) && t >= 0) return Math.round(t);
  return typeof text === 'string' ? text.trim().length : 0;
}

/// `[{prompt, off, on, offTokens?, onTokens?}]` → 表の行。
///
/// - `offLen` / `onLen` は出力トークン数 (渡されなければ文字数)
/// - `delta` は `onLen - offLen`
/// - `changed` は空白を潰した応答が違うか
/// - `offRefusal` / `onRefusal` は拒否っぽい言い回しか
export function ablitEvalRows(items, opts = {}) {
  const arr = Array.isArray(items) ? items : [];
  const max = intOr(opts.max, ABLIT_EVAL_SNIPPET);
  return arr.map((it, i) => {
    const src = it && typeof it === 'object' ? it : {};
    const prompt = typeof src.prompt === 'string' ? src.prompt : '';
    const off = typeof src.off === 'string' ? src.off : '';
    const on = typeof src.on === 'string' ? src.on : '';
    const offLen = evalLen(off, src.offTokens);
    const onLen = evalLen(on, src.onTokens);
    const delta = onLen - offLen;
    return {
      index: i,
      prompt,
      off,
      on,
      offLen,
      onLen,
      delta,
      deltaText: (delta > 0 ? '+' : '') + delta,
      changed: off.replace(/\s+/g, ' ').trim() !== on.replace(/\s+/g, ' ').trim(),
      offRefusal: ablitRefusalLike(off),
      onRefusal: ablitRefusalLike(on),
      offSnippet: ablitSnippet(off, max),
      onSnippet: ablitSnippet(on, max),
    };
  });
}

/// 行 → 等幅の表 (`prompt | OFF | ON | len Δ`)。1 プロンプト 1 行。
export function ablitEvalTable(rows) {
  const arr = Array.isArray(rows) ? rows : [];
  const head = ['prompt', 'OFF', 'ON', 'len Δ'];
  const cells = arr.map((r) => [
    String(r.prompt == null ? '' : r.prompt).replace(/\s+/g, ' '),
    (r.offSnippet == null ? '' : String(r.offSnippet)) + (r.offRefusal ? ' [refuse]' : ''),
    (r.onSnippet == null ? '' : String(r.onSnippet)) + (r.onRefusal ? ' [refuse]' : ''),
    String(r.deltaText == null ? '' : r.deltaText),
  ]);
  const w = head.map((h, i) => Math.max(h.length, ...cells.map((c) => c[i].length)));
  const line = (c) => c.map((x, i) => x + ' '.repeat(w[i] - x.length)).join(' | ');
  return [line(head), w.map((n) => '-'.repeat(n)).join('-+-'), ...cells.map(line)].join('\n');
}

// ---- レシピ (task 833) ----------------------------------------------------
//
// `/2d` の設定を 1 ファイルにまとめ、`denno serve-flash --steer-recipe` で
// 起動時に掛け直せるようにする。検証規則は engine の
// `SteerRecipe::from_json` と同じ (版・名前・層範囲・有限の alpha/coef・
// 埋め込みは base64 文字列)。

/// レシピの JSON スキーマ版。
export const RECIPE_VERSION = 1;
/// レシピ 1 本の項数の上限。
export const RECIPE_MAX_TERMS = 16;

/// 画面の [from,to)・alpha・名前 → レシピの本文 (保存用)。
/// `opts.embed` はサーバーに埋め込ませる指定 (`embed_snapshots`)。
export function ablitRecipeBodyFromUi(range, alpha, opts = {}) {
  const o = opts && typeof opts === 'object' ? opts : {};
  const r = Array.isArray(range) ? range : [];
  if (r.length !== 2) throw new Error('range: [from, to] required');
  const from = intOr(r[0], -1);
  const to = intOr(r[1], -1);
  if (from < 0 || to < 0) throw new Error('range: from/to must be >= 0');
  if (from > to) throw new Error('range: from must be <= to');
  const a = alpha == null ? 1 : Number(alpha);
  if (!Number.isFinite(a)) throw new Error('alpha: must be a finite number');
  const name = o.name == null ? 'recipe' : String(o.name);
  if (!NAME_RE.test(name)) throw new Error('name: A-Z a-z 0-9 _ - only, at most 64 chars');
  const body = {
    name,
    embed_snapshots: o.embed === true,
    terms: [
      { snapshot: o.a || ABLIT_A, coef: 1 },
      { snapshot: o.b || ABLIT_B, coef: -1 },
    ],
    layers: [from, to],
    alpha: a,
    mode: 'project',
  };
  if (o.description != null) {
    if (typeof o.description !== 'string') throw new Error('description must be a string');
    body.description = o.description;
  }
  return body;
}

/// パース済みのレシピを検証して正規化する (実 API の `from_json` と同じ規則)。
/// 不正は例外。未知のフィールドは黙って捨てずに弾く。
export function ablitRecipeObject(v) {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('recipe must be a JSON object');
  const allowed = [
    'version', 'name', 'description', 'model_id', 'mode', 'alpha',
    'layer_from', 'layer_to', 'terms', 'snapshots',
  ];
  for (const k of Object.keys(v)) {
    if (allowed.indexOf(k) < 0) throw new Error('unknown field ' + JSON.stringify(k));
  }
  if (!Object.prototype.hasOwnProperty.call(v, 'version')) throw new Error('version: required (1)');
  if (v.version !== RECIPE_VERSION) throw new Error('version: expected ' + RECIPE_VERSION + ' (got ' + v.version + ')');
  if (typeof v.name !== 'string' || !NAME_RE.test(v.name)) {
    throw new Error('name: A-Z a-z 0-9 _ - only, at most 64 chars');
  }
  const out = { version: RECIPE_VERSION, name: v.name };
  if (v.description != null) {
    if (typeof v.description !== 'string') throw new Error('description must be a string');
    out.description = v.description;
  }
  if (v.model_id != null) {
    if (typeof v.model_id !== 'string') throw new Error('model_id must be a string');
    if (v.model_id) out.model_id = v.model_id;
  }
  if (typeof v.mode !== 'string' || STEER_MODES.indexOf(v.mode) < 0) {
    throw new Error('mode: ' + STEER_MODES.join(' / '));
  }
  out.mode = v.mode;
  const alpha = v.alpha == null ? 1 : Number(v.alpha);
  if (!Number.isFinite(alpha)) throw new Error('alpha: must be a finite number');
  out.alpha = alpha;
  const from = intOr(v.layer_from, -1);
  const to = intOr(v.layer_to, -1);
  if (from < 0 || to < 0) throw new Error('layer_from / layer_to: must be integers >= 0');
  if (from > to) throw new Error('layer_from must be <= layer_to');
  out.layer_from = from;
  out.layer_to = to;
  if (!Array.isArray(v.terms) || v.terms.length === 0) throw new Error('terms: at least one term is required');
  if (v.terms.length > RECIPE_MAX_TERMS) throw new Error('terms: at most ' + RECIPE_MAX_TERMS);
  out.terms = v.terms.map((t, i) => {
    if (!t || typeof t !== 'object' || Array.isArray(t)) throw new Error('terms[' + i + ']: must be an object');
    for (const k of Object.keys(t)) {
      if (k !== 'snapshot' && k !== 'coef') throw new Error('terms[' + i + ']: unknown field ' + JSON.stringify(k));
    }
    if (typeof t.snapshot !== 'string' || !NAME_RE.test(t.snapshot)) {
      throw new Error('terms[' + i + '].snapshot: A-Z a-z 0-9 _ - only');
    }
    const coef = Number(t.coef);
    if (!Number.isFinite(coef)) throw new Error('terms[' + i + '].coef: must be a finite number');
    return { snapshot: t.snapshot, coef };
  });
  if (v.snapshots != null) {
    if (typeof v.snapshots !== 'object' || Array.isArray(v.snapshots)) {
      throw new Error('snapshots: must be an object of { name: base64 }');
    }
    const outSnaps = {};
    for (const k of Object.keys(v.snapshots)) {
      if (!NAME_RE.test(k)) throw new Error('snapshots: bad name ' + JSON.stringify(k));
      const s = v.snapshots[k];
      if (typeof s !== 'string' || !s) throw new Error('snapshots[' + JSON.stringify(k) + ']: non-empty base64 string required');
      outSnaps[k] = s;
    }
    out.snapshots = outSnaps;
  }
  return out;
}

/// レシピのテキスト (ファイル) → 検証済みのレシピ。
export function ablitRecipeFromText(text) {
  if (typeof text !== 'string' || !text.trim()) throw new Error('file is empty');
  let v;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new Error('not JSON: ' + ((e && e.message) ? e.message : String(e)));
  }
  return ablitRecipeObject(v);
}

/// レシピの 1 行要約 ("no_refusal · project · α 0.5 · layers 1-3 · embedded 2")。
export function ablitRecipeSummaryText(r) {
  if (!r || typeof r !== 'object') return '';
  const parts = [String(r.name || '')];
  parts.push(String(r.mode || ''));
  parts.push('α ' + (r.alpha == null ? 1 : r.alpha));
  parts.push('layers ' + intOr(r.layer_from, 0) + '-' + intOr(r.layer_to, 0));
  const n = r.snapshots && typeof r.snapshots === 'object' ? Object.keys(r.snapshots).length : 0;
  if (n) parts.push('embedded ' + n);
  if (r.model_id) parts.push('model ' + r.model_id);
  return parts.join(' · ');
}

/// レシピの保存ファイル名。
export function ablitRecipeFileName(name) {
  const n = String(name || 'recipe');
  return n + '.recipe.json';
}

/// 評価の進捗表示 ("eval 2/5…" / "eval 2/5 (off)…")。
export function ablitEvalProgress(done, total, phase) {
  const t = Math.max(0, intOr(total, 0));
  const d = Math.max(0, Math.min(t, intOr(done, 0)));
  return 'eval ' + d + '/' + t + (phase ? ' (' + phase + ')' : '') + '…';
}

// ---- モック店 (shape が 404 のとき) ---------------------------------------

/// 名前 → 32 bit の決定的な種 (モックの合成スコア用)。
function hashName(str) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h;
}

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

// ---- expert tier (task 271) ----------------------------------------------

/// `/v1/experts/stats` の本文を state 欄の行に落とす (純関数)。
///
/// 実 API (denno-cli の `experts_stats_json`) の形:
///   { enabled, file, last_save:{unix,bytes}|null,
///     recorded:{dir,expert_bytes,n_layer,n_expert,experts,uses}|null,
///     vram:{experts,mb,cap_mb}, ram:{experts,slots,mb,pinned_mb},
///     mmap:{experts,mb}, coverage:{vram,ram,total},
///     last_request:{lookups,vram:{hits,hit_rate},ram:{...},mmap:{...}}|null,
///     plan:{...}|null }
///
/// 古いサーバー (404) や欠けたキーでも呼べるように、数値でない値は null に落とす。
/// `hit` は 0..1 (表示で % に丸める)。
export function expertStatsView(body) {
  const b = body && typeof body === 'object' && !Array.isArray(body) ? body : {};
  const num = (v) => {
    if (v === null || v === undefined || v === '') return null;
    const x = Number(v);
    return Number.isFinite(x) ? x : null;
  };
  const obj = (v) => (v && typeof v === 'object' && !Array.isArray(v) ? v : {});
  const last = obj(b.last_request);
  const hasLast = b.last_request && typeof b.last_request === 'object';
  const tier = (key) => {
    const t = obj(b[key]);
    const raw = hasLast ? num(obj(last[key]).hit_rate) : null;
    return {
      key,
      experts: num(t.experts),
      mb: num(t.mb),
      hit: raw == null ? null : Math.max(0, Math.min(1, raw)),
    };
  };
  const cov = obj(b.coverage);
  const save = obj(b.last_save);
  return {
    enabled: b.enabled === true,
    file: typeof b.file === 'string' && b.file ? b.file : null,
    savedUnix: num(save.unix),
    savedBytes: num(save.bytes),
    uses: num(obj(b.recorded).uses),
    rows: [tier('vram'), tier('ram'), tier('mmap')],
    coverage: { vram: num(cov.vram), ram: num(cov.ram), total: num(cov.total) },
  };
}

/// 1 tier の 1 行: "12,345 exp · 21,000 MB · last 91%"。欠けた値は "-"。
export function expertTierText(row) {
  const r = row || {};
  const n = (v) => (v == null ? '-' : Number(v).toLocaleString('en-US'));
  const p = r.hit == null ? '-' : Math.round(r.hit * 100) + '%';
  return n(r.experts) + ' exp · ' + n(r.mb) + ' MB · last ' + p;
}

/// 記録された使用回数のうち各 tier が覆う割合: "vram 70% · ram 20% · total 90%"。
export function expertCoverageText(cov) {
  const c = cov || {};
  const p = (v) => (v == null ? '-' : Math.round(Number(v) * 100) + '%');
  return 'vram ' + p(c.vram) + ' · ram ' + p(c.ram) + ' · total ' + p(c.total);
}

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
///
/// task 825: `/v1/steer/analyze` / `export` / `import` も受ける。分析は
/// エンジンを持たないので、名前から決定的に作った山を返す (形だけ実 API と同じ)。
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
    // task 829: 直近に使った capture_mode (実 API の GET /v1/steer と同じ)。
    last_capture_mode: 'all',
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
      capture_mode: st.recording ? st.recording.capture_mode : st.last_capture_mode,
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

  /// task 271: `/v1/experts/stats` の代役。形から決まる固定値で、トークンでは
  /// 変わらない (実 API の tier の本数も decode では大きく動かない)。
  /// 実 API と同じキーを全部返す (ビューワはここだけで描画を確かめられる)。
  function expertStats() {
    const total = s.layers * s.n_expert;
    const per = 3_072_000; // gate+up+down 1 本ぶんのバイト数 (合成の値)
    const mb = (n) => Math.floor((n * per) / (1024 * 1024));
    const vramExperts = Math.floor(total * 0.35);
    const ramExperts = Math.floor(total * 0.25);
    const mmapExperts = total - vramExperts - ramExperts;
    const looks = s.layers * s.n_expert_used;
    const vramHits = Math.round(looks * 0.9);
    const ramHits = Math.round(looks * 0.07);
    const mmapHits = Math.max(0, looks - vramHits - ramHits);
    const rate = (n) => (looks > 0 ? n / looks : 0);
    return {
      enabled: true,
      file: 'mock/dnnoexps-mock-' + per * total + '.bin',
      last_save: { unix: now(), bytes: 24 + 4 + 8 + total * 4 },
      recorded: {
        dir: 'mock', expert_bytes: per * total, n_layer: s.layers,
        n_expert: s.n_expert, experts: total, uses: looks,
      },
      vram: { experts: vramExperts, mb: mb(vramExperts), cap_mb: mb(vramExperts) + 512 },
      ram: { experts: ramExperts, slots: ramExperts, mb: mb(ramExperts), pinned_mb: mb(ramExperts) },
      mmap: { experts: mmapExperts, mb: mb(mmapExperts) },
      coverage: { vram: 0.7, ram: 0.2, total: 0.9 },
      last_request: {
        lookups: looks,
        vram: { hits: vramHits, hit_rate: rate(vramHits) },
        ram: { hits: ramHits, hit_rate: rate(ramHits) },
        mmap: { hits: mmapHits, hit_rate: rate(mmapHits) },
      },
      plan: null,
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
      if (m === 'GET' && p === '/v1/model/choices') {
        // モックは切り替えできない (実 API も DENNO_MODEL_CHOICES が無ければ同じ形)。
        return { status: 200, body: { switchable: false, current: null, current_dir: 'mock/model', choices: [] } };
      }
      if (m === 'POST' && p === '/v1/abort') {
        // モックには走っている生成が無い (実 API も生成中でなければ空振り)。
        return { status: 200, body: { aborted: false, generating: false } };
      }
      if (m === 'GET' && p === '/v1/cache') {
        return { status: 200, body: ctxStatus() };
      }
      if (m === 'GET' && p === '/v1/experts/stats') {
        // task 271: tier ごとの本数・MB、直前のリクエストの当たり (実 API と同じ形)。
        return { status: 200, body: expertStats() };
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
          if (k !== 'name' && k !== 'capture_mode') {
            return err(400, 'unknown field ' + JSON.stringify(k) + ' (name / capture_mode only)');
          }
        }
        const name = typeof body.name === 'string' ? body.name : '';
        if (!name) return err(400, 'name is required');
        if (!NAME_RE.test(name)) return err(400, 'name: A-Z a-z 0-9 _ - only, at most 64 chars');
        // task 829: 未知の capture_mode は実 API と同じく 400 (黙って all にしない)。
        const cm = body.capture_mode == null ? 'all' : body.capture_mode;
        if (typeof cm !== 'string' || CAPTURE_MODES.indexOf(cm) < 0) {
          return err(400, 'capture_mode: ' + CAPTURE_MODES.join(' / ') + ' (got ' + JSON.stringify(cm) + ')');
        }
        if (st.recording) return err(400, 'already recording: ' + st.recording.name);
        st.last_capture_mode = cm;
        st.recording = { name, tokens: 0, created: now(), capture_mode: cm };
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
      // task 825: Abliterate 作業台 (analyze / export / import)。
      // モックはエンジンを持たないので、層ごとの分離スコアは名前から
      // 決定的に作り、形と検証規則だけ実 API に合わせる。
      const qi = p.indexOf('?');
      const bare = qi < 0 ? p : p.slice(0, qi);
      const q = new URLSearchParams(qi < 0 ? '' : p.slice(qi + 1));
      if (m === 'POST' && bare === '/v1/steer/analyze') {
        if (!body || typeof body !== 'object' || Array.isArray(body)) return err(400, 'body must be an object');
        for (const k of Object.keys(body)) {
          if (['pos', 'neg', 'terms'].indexOf(k) < 0) {
            return err(400, 'unknown field ' + JSON.stringify(k) + ' (pos / neg / terms only)');
          }
        }
        let names;
        if (body.terms != null) {
          if (!Array.isArray(body.terms) || body.terms.length === 0) return err(400, 'terms must be a non-empty array');
          let np = 0;
          let nn = 0;
          for (let i = 0; i < body.terms.length; i++) {
            const t = body.terms[i];
            if (!t || typeof t !== 'object' || Array.isArray(t)) return err(400, 'terms[' + i + '] must be an object');
            for (const k of Object.keys(t)) {
              if (['snapshot', 'coef'].indexOf(k) < 0) return err(400, 'terms[' + i + ']: unknown field ' + JSON.stringify(k));
            }
            if (typeof t.snapshot !== 'string' || !t.snapshot) return err(400, 'terms[' + i + '].snapshot must be a string');
            if (!Number.isFinite(Number(t.coef))) return err(400, 'terms[' + i + '].coef must be a finite number');
            if (Number(t.coef) > 0) np++;
            else if (Number(t.coef) < 0) nn++;
          }
          if (!np || !nn) return err(400, 'both a positive and a negative coef are required');
          names = body.terms.map((t) => t.snapshot);
        } else {
          if (typeof body.pos !== 'string' || typeof body.neg !== 'string') return err(400, 'pos and neg must be strings');
          names = [body.pos, body.neg];
        }
        for (const n of names) {
          if (!st.snapshots.some((x) => x.name === n)) return err(400, 'no such snapshot: ' + n);
        }
        const seed = hashName(names.join('|'));
        const peak = 2 + (seed % Math.max(1, s.layers - 4));
        const layers = [];
        for (let l = 0; l < s.layers; l++) {
          const v = 0.15 + 0.7 * Math.exp(-((l - peak) * (l - peak)) / 18);
          layers.push({
            layer: l, diff_norm: Math.round(v * 10000) / 1000,
            norm_pos: 9, norm_neg: 8.5, cos_sim: -0.2 - v * 0.3, sep_score: v,
          });
        }
        const rr = ablitRecommendedRange(layers.map((x) => x.sep_score));
        return {
          status: 200,
          body: {
            pos: names[0], neg: names[names.length - 1], n_layer: s.layers,
            peak_layer: peak, recommended_range: rr, layers,
          },
        };
      }
      if (m === 'GET' && bare === '/v1/steer/export') {
        const name = q.get('name') || '';
        const snap = st.snapshots.find((x) => x.name === name);
        if (!snap) return err(400, name ? 'no such snapshot: ' + name : 'name is required (?name=...)');
        if (q.get('format') === 'json') {
          return {
            status: 200,
            body: {
              name: snap.name, tokens: snap.tokens, created: snap.created,
              model: 'mock', layers: s.layers, n_hc: s.n_hc, n_embd: s.n_embd,
              format: 'DNSTEER1', data_base64: 'bW9jaw==',
            },
          };
        }
        return { status: 200, body: { name: snap.name, tokens: snap.tokens, format: 'DNSTEER1', data_base64: 'bW9jaw==' } };
      }
      if (m === 'POST' && bare === '/v1/steer/import') {
        if (!body || typeof body !== 'object' || Array.isArray(body)) return err(400, 'body must be a JSON object');
        const name = q.get('name') || body.name;
        if (typeof name !== 'string' || !NAME_RE.test(name)) return err(400, 'name: A-Z a-z 0-9 _ - only, at most 64 chars');
        const layers = body.layers == null ? s.layers : Number(body.layers);
        if (layers !== s.layers) return err(400, 'layers: expected ' + s.layers);
        const snap = { name, tokens: Number(body.tokens) || 0, created: now(), layers };
        const i = st.snapshots.findIndex((x) => x.name === name);
        if (i >= 0) st.snapshots[i] = snap;
        else st.snapshots.push(snap);
        st.snapshots.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
        const out = status();
        out.imported = jsonClone(snap);
        return { status: 200, body: out };
      }
      // task 833: レシピ (持ち出し / 取り込み)。検証規則は実 API と同じ。
      if (m === 'POST' && bare === '/v1/steer/recipe/export') {
        if (!body || typeof body !== 'object' || Array.isArray(body)) return err(400, 'body must be an object');
        for (const k of Object.keys(body)) {
          if (['name', 'description', 'embed_snapshots', 'terms', 'layers', 'alpha', 'mode'].indexOf(k) < 0) {
            return err(400, 'unknown field ' + JSON.stringify(k));
          }
        }
        const name = body.name == null ? 'recipe' : body.name;
        if (typeof name !== 'string' || !NAME_RE.test(name)) {
          return err(400, 'name: A-Z a-z 0-9 _ - only, at most 64 chars');
        }
        if (body.embed_snapshots != null && typeof body.embed_snapshots !== 'boolean') {
          return err(400, 'embed_snapshots must be a boolean');
        }
        if (body.description != null && typeof body.description !== 'string') {
          return err(400, 'description must be a string');
        }
        let base;
        if (body.terms != null) {
          const draft = {
            version: RECIPE_VERSION, name,
            mode: body.mode == null ? 'project' : body.mode,
            alpha: body.alpha == null ? 1 : body.alpha,
            layer_from: Array.isArray(body.layers) ? body.layers[0] : 0,
            layer_to: Array.isArray(body.layers) ? body.layers[1] : s.layers,
            terms: body.terms,
          };
          let recipe;
          try {
            recipe = ablitRecipeObject(draft);
          } catch (e) {
            return err(400, (e && e.message) ? e.message : String(e));
          }
          base = recipe;
        } else {
          if (!st.apply) return err(400, 'no active steering (pass terms or apply one first)');
          base = {
            version: RECIPE_VERSION, name, mode: st.apply.mode, alpha: st.apply.alpha,
            layer_from: st.apply.layers[0], layer_to: st.apply.layers[1], terms: st.apply.terms,
          };
        }
        base.model_id = 'mock';
        if (body.description != null) base.description = body.description;
        if (body.embed_snapshots === true) {
          const snaps = {};
          for (const t of base.terms) {
            if (!st.snapshots.some((x) => x.name === t.snapshot)) {
              return err(400, 'no such snapshot: ' + t.snapshot);
            }
            // 実 API は DNSTEER1 を base64 で載せる。モックは中身を持たない。
            snaps[t.snapshot] = 'bW9jaw==';
          }
          base.snapshots = snaps;
        }
        return { status: 200, body: jsonClone(base) };
      }
      if (m === 'POST' && bare === '/v1/steer/recipe/import') {
        let rv = body;
        let wantApply = false;
        if (body && typeof body === 'object' && !Array.isArray(body)
          && Object.prototype.hasOwnProperty.call(body, 'recipe')) {
          for (const k of Object.keys(body)) {
            if (k !== 'recipe' && k !== 'apply') return err(400, 'unknown field ' + JSON.stringify(k));
          }
          if (body.apply != null && typeof body.apply !== 'boolean') return err(400, 'apply must be a boolean');
          rv = body.recipe;
          wantApply = body.apply === true;
        }
        let recipe;
        try {
          recipe = ablitRecipeObject(rv);
        } catch (e) {
          return err(400, (e && e.message) ? e.message : String(e));
        }
        const embedded = recipe.snapshots ? Object.keys(recipe.snapshots) : [];
        if (recipe.layer_to > s.layers) return err(400, 'layers: to must be <= ' + s.layers);
        for (const t of recipe.terms) {
          if (embedded.indexOf(t.snapshot) < 0 && !st.snapshots.some((x) => x.name === t.snapshot)) {
            return err(400, 'no such snapshot: ' + t.snapshot);
          }
        }
        // 埋め込みはメモリにだけ載せる (実 API と同じ)。
        for (const n of embedded) {
          if (!st.snapshots.some((x) => x.name === n)) {
            st.snapshots.push({ name: n, tokens: 0, created: now(), layers: s.layers });
          }
        }
        st.snapshots.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
        const doApply = wantApply || q.get('apply') === '1' || q.get('apply') === 'true';
        const out = {
          valid: true,
          applied: false,
          recipe: {
            name: recipe.name, description: recipe.description || null, model_id: recipe.model_id || null,
            mode: recipe.mode, alpha: recipe.alpha, layers: [recipe.layer_from, recipe.layer_to],
            terms: recipe.terms, embedded,
          },
        };
        if (doApply) {
          st.apply = {
            terms: recipe.terms.map((t) => ({ snapshot: t.snapshot, coef: t.coef })),
            layers: [recipe.layer_from, recipe.layer_to],
            alpha: recipe.alpha,
            mode: recipe.mode,
          };
          st.epoch++;
          out.applied = true;
          out.steer = status();
        }
        return { status: 200, body: out };
      }
      return { status: 404, body: apiError('not found: ' + m + ' ' + p) };
    },
  };
  return store;
}
