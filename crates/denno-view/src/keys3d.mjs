// denno-view /3d のカメラブックマークとキーボード凡例の純ロジック (DOM 非依存)。
//
// task 813:
//   1. 数字キー 1..4 でカメラを固定の見どころへ飛ばす
//      (1=全体, 2=router / experts, 3=prediction head, 4=KV 帯)。
//   2. `?` で開閉するキーボードショートカットの凡例。
//
// task 904:
//   3. 1..5 のカメラプリセット (Overview / Front / Top / Side / Reset) と、
//      遷移を滑らかにつなぐ補間状態 (createCamAnim / startCamAnim / stepCamAnim)。
//   4. 層インスペクタ用の純ロジック (layerKind / stepLayer / activationStats /
//      expertSummary など)。DOM は index3d.html が持つ。
//   5. fps オーバーレイ用のフレーム計測 (createFpsMeter / noteFrame / fpsSummary)。
//
// 座標は index3d.html のシーン配置 (X の並び / zOf(il) / KV 帯 / topZ) に合わせる。
// 画面は goal の pos/target を camera と controls.target に入れるだけ。

/// 見どころ。pos = カメラ位置、target = 注視点 (どちらも three.js の座標)。
export const BOOKMARKS = Object.freeze([
  Object.freeze({ key: '1', name: 'Overview', pos: Object.freeze([29, 30, 47]), target: Object.freeze([-4, -1, -25.6]) }),
  Object.freeze({ key: '2', name: 'Router / Experts', pos: Object.freeze([20, 21, -4]), target: Object.freeze([-3, 1.5, -26]) }),
  Object.freeze({ key: '3', name: 'Prediction Head', pos: Object.freeze([-6.6, 11, -58]), target: Object.freeze([-6.6, 0.5, -83]) }),
  Object.freeze({ key: '4', name: 'KV Band', pos: Object.freeze([-1.5, 13, 33]), target: Object.freeze([-1.5, 3.2, 10]) }),
]);

/// ブックマークに割り当てたキー (この並びで凡例にも出す)。
export const BOOKMARK_KEYS = Object.freeze(['1', '2', '3', '4']);

/// キー (文字列) → ブックマーク。無ければ null。数字キー以外は受けない。
export function bookmarkForKey(key) {
  const s = String(key == null ? '' : key);
  for (const b of BOOKMARKS) if (b.key === s) return b;
  return null;
}

function finite3(a) {
  return Array.isArray(a) && a.length === 3 && a.every((v) => Number.isFinite(v));
}

/// ブックマーク → カメラにそのまま入れる goal。壊れていれば null。
export function bookmarkGoal(bm) {
  if (!bm || !finite3(bm.pos) || !finite3(bm.target)) return null;
  return { pos: bm.pos.slice(), target: bm.target.slice() };
}

// ===== task 904: カメラプリセット (1..5) ==================================
// 「決め打ちの視点」を 5 つ。値は index3d.html のシーン座標系で固定する
// (乱数も時刻も使わないので、同じキーなら毎回同じ絵になる)。
export const CAMERA_PRESETS = Object.freeze([
  // 全体を斜め上から (起動時の視点と同じ)
  Object.freeze({ key: '1', name: 'Overview', pos: Object.freeze([29, 30, 47]), target: Object.freeze([-4, -1, -25.6]) }),
  // 残差レーンを正面から見る
  Object.freeze({ key: '2', name: 'Front', pos: Object.freeze([-4, 6, 58]), target: Object.freeze([-4, 0, -20]) }),
  // 真上から層の並び (Z 方向) を見る
  Object.freeze({ key: '3', name: 'Top', pos: Object.freeze([-4, 78, -20]), target: Object.freeze([-4, 0, -20]) }),
  // 真横 (X 方向) から expert グリッドを見る
  Object.freeze({ key: '4', name: 'Side', pos: Object.freeze([62, 9, -20]), target: Object.freeze([-4, 0, -20]) }),
  // 起動時のカメラへ戻す
  Object.freeze({ key: '5', name: 'Reset', pos: Object.freeze([29, 30, 47]), target: Object.freeze([-4, -1, -25.6]) }),
]);

/// プリセットに割り当てたキー (この並びで凡例にも出す)。
export const PRESET_KEYS = Object.freeze(['1', '2', '3', '4', '5']);

/// キー (文字列) → プリセット。無ければ null。
export function presetForKey(key) {
  const s = String(key == null ? '' : key);
  for (const p of CAMERA_PRESETS) if (p.key === s) return p;
  return null;
}

/// プリセット → カメラ goal (コピー)。壊れていれば null。
export function presetGoal(preset) {
  return bookmarkGoal(preset);
}

/// smoothstep (0..1 を 0..1 に)。両端で速度 0 になるので出だしと着地が滑らか。
export function easeSmoothstep(t) {
  const x = Number.isFinite(t) ? Math.min(1, Math.max(0, t)) : 0;
  return x * x * (3 - 2 * x);
}

/// 補間の状態。duration は秒。値はすべて素の数値配列 (three.js に依存しない)。
export function createCamAnim(duration) {
  const d = Number.isFinite(duration) && duration > 0 ? duration : 0.45;
  return { active: false, t: 0, duration: d, fromPos: null, fromTarget: null, toPos: null, toTarget: null };
}

/// from (省略時は to) から goal への遷移を開始する。壊れた goal は false。
export function startCamAnim(state, goal, from) {
  if (!state || typeof state !== 'object') return false;
  if (!goal || !finite3(goal.pos) || !finite3(goal.target)) return false;
  const f = from && finite3(from.pos) && finite3(from.target) ? from : null;
  state.fromPos = f ? f.pos.slice() : goal.pos.slice();
  state.fromTarget = f ? f.target.slice() : goal.target.slice();
  state.toPos = goal.pos.slice();
  state.toTarget = goal.target.slice();
  state.t = 0;
  state.active = true;
  return true;
}

/// 経過時間を進めて、その時点の pos/target を返す。動いていなければ null。
export function stepCamAnim(state, dt) {
  if (!state || !state.active) return null;
  const step = Number.isFinite(dt) && dt > 0 ? dt : 0;
  state.t = Math.min(state.duration, state.t + step);
  const done = state.t >= state.duration;
  const u = easeSmoothstep(state.duration > 0 ? state.t / state.duration : 1);
  const lerp3 = (a, b) => [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u];
  const out = done
    ? { pos: state.toPos.slice(), target: state.toTarget.slice(), done: true }
    : { pos: lerp3(state.fromPos, state.toPos), target: lerp3(state.fromTarget, state.toTarget), done: false };
  if (done) state.active = false;
  return out;
}

/// 遷移中か。
export function camAnimActive(state) {
  return !!(state && state.active);
}

// ===== task 904: 層インスペクタ ==========================================
/// full attention (QSA) が入る間隔。index3d.html の INTERVAL と同じ。
export const ATTENTION_INTERVAL = 4;
export const LAYER_COUNT = 48;

/// 層番号 → mixer の種類。4 層に 1 層が full attention。
export function layerKind(il, interval = ATTENTION_INTERVAL) {
  const n = Number.isFinite(interval) && interval > 0 ? interval : ATTENTION_INTERVAL;
  const i = Number(il);
  if (!Number.isFinite(i) || i < 0) return 'GDN';
  return (Math.trunc(i) + 1) % n === 0 ? 'FullAttn' : 'GDN';
}

/// 層番号を 0..n-1 に丸める (範囲外は端に寄せる)。
export function clampLayer(il, n = LAYER_COUNT) {
  const max = Number.isFinite(n) && n > 0 ? Math.trunc(n) : LAYER_COUNT;
  const i = Number.isFinite(Number(il)) ? Math.trunc(Number(il)) : 0;
  return Math.min(max - 1, Math.max(0, i));
}

/// 層番号を delta だけ動かす。端では反対側へ回り込む ([-] で 0 から 47 へ)。
export function stepLayer(il, delta, n = LAYER_COUNT) {
  const max = Number.isFinite(n) && n > 0 ? Math.trunc(n) : LAYER_COUNT;
  const cur = clampLayer(il, max);
  const d = Number.isFinite(Number(delta)) ? Math.trunc(Number(delta)) : 0;
  return ((cur + d) % max + max) % max;
}

/// 層の活性 (ニューロンの実値 0..1) の要約。max / mean / rms / lit を返す。
export function activationStats(values) {
  const n = values && typeof values.length === 'number' ? values.length : 0;
  let max = 0, sum = 0, sq = 0, lit = 0;
  for (let i = 0; i < n; i++) {
    const v = Number(values[i]);
    if (!Number.isFinite(v) || v <= 0) continue;
    if (v > max) max = v;
    sum += v; sq += v * v;
    if (v >= 0.01) lit++;
  }
  return {
    count: n,
    lit,
    max,
    mean: n > 0 ? sum / n : 0,
    rms: n > 0 ? Math.sqrt(sq / n) : 0,
  };
}

/// 活性の要約を 1 行の文字列にする (HUD 用)。
export function formatMagnitude(st) {
  if (!st || !Number.isFinite(st.max)) return '--';
  return 'max ' + st.max.toFixed(3) + ' · rms ' + st.rms.toFixed(3) + ' · lit ' + st.lit;
}

/// ルーティングされた expert の要約。重みの大きい順に limit 件 + 残り件数。
export function expertSummary(experts, weights, limit = 3) {
  const list = [];
  const n = experts && typeof experts.length === 'number' ? experts.length : 0;
  for (let i = 0; i < n; i++) {
    const k = Number(experts[i]);
    if (!Number.isFinite(k)) continue;
    const w = weights && Number.isFinite(Number(weights[i])) ? Number(weights[i]) : 0;
    list.push({ index: Math.trunc(k), weight: w });
  }
  list.sort((a, b) => (b.weight - a.weight) || (a.index - b.index));
  const cap = Number.isFinite(limit) && limit > 0 ? Math.trunc(limit) : 3;
  return { top: list.slice(0, cap), more: Math.max(0, list.length - cap), count: list.length };
}

/// expert の要約を 1 行にする。"E123 0.19 · E7 0.14 · +7" / 無ければ "--"。
export function formatExpertSummary(sum) {
  if (!sum || !sum.top || sum.top.length === 0) return '--';
  const head = sum.top.map((e) => 'E' + e.index + ' ' + e.weight.toFixed(2)).join(' · ');
  return sum.more > 0 ? head + ' · +' + sum.more : head;
}

/// [ / ] のキーイベント → 前後どちらへ動かすか (-1 / 0 / +1)。
export function isLayerStepKey(e) {
  if (!e) return 0;
  if (e.key === '[') return -1;
  if (e.key === ']') return 1;
  return 0;
}

/// キーイベントが fps オーバーレイの開閉か (p / P)。
export function isFpsKey(e) {
  if (!e) return false;
  return e.key === 'p' || e.key === 'P';
}

/// オーバーレイの初期表示 (?fps=1 など)。真値は 1 / true / yes / on。
function truthy(v) {
  const s = String(v == null ? '' : v).toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'on';
}

export function overlayFromQuery(v) {
  return truthy(v);
}

// ===== task 904: fps オーバーレイ ========================================
/// フレーム間隔の計測器。時刻は呼び出し側が渡す (performance.now() など)。
export function createFpsMeter(cap) {
  const n = Number.isFinite(cap) && cap > 0 ? Math.trunc(cap) : 48;
  return { last: null, first: null, frames: 0, sum: 0, ema: 0, min: 0, max: 0, hist: [], cap: n };
}

/// 1 フレーム分の時刻を記録し、その時点の要約を返す。壊れた時刻は捨てる。
export function noteFrame(meter, now) {
  if (!meter || typeof meter !== 'object') return null;
  const t = Number(now);
  if (!Number.isFinite(t)) return null;
  if (meter.last != null) {
    const d = t - meter.last;
    if (d > 0) {
      meter.frames++;
      meter.sum += d;
      meter.ema = meter.ema > 0 ? meter.ema * 0.9 + d * 0.1 : d;
      if (meter.min === 0 || d < meter.min) meter.min = d;
      if (d > meter.max) meter.max = d;
      meter.hist.push(d);
      while (meter.hist.length > meter.cap) meter.hist.shift();
    }
  } else {
    meter.first = t;
  }
  meter.last = t;
  return fpsSummary(meter);
}

/// 計測器の要約。fps は平均フレーム間隔から出す。
export function fpsSummary(meter) {
  if (!meter) return { frames: 0, avgMs: 0, fps: 0, emaMs: 0, minMs: 0, maxMs: 0, hist: [] };
  const avg = meter.frames > 0 ? meter.sum / meter.frames : 0;
  return {
    frames: meter.frames,
    avgMs: avg,
    fps: avg > 0 ? 1000 / avg : 0,
    emaMs: meter.ema,
    minMs: meter.min,
    maxMs: meter.max,
    hist: meter.hist.slice(),
  };
}

/// 要約を 1 行にする。"fps 60 · 16.7 ms"。まだ 2 フレーム未満なら "--"。
export function fpsText(sum) {
  if (!sum || !(sum.fps > 0)) return '--';
  return 'fps ' + sum.fps.toFixed(0) + ' · ' + sum.avgMs.toFixed(1) + ' ms';
}

/// 直近のフレーム間隔 → 0..1 のバー高さ (描画用)。速いほど低い。
export function fpsBars(sum, n = 24) {
  const s = sum && Array.isArray(sum.hist) ? sum.hist : [];
  const cap = Number.isFinite(n) && n > 0 ? Math.trunc(n) : 24;
  const tail = s.slice(-cap);
  const norm = (d) => {
    // 60 fps = 16.7 ms を 0.5 とし、8..40 ms を 0..1 に写す
    const x = (Number(d) - 8) / 32;
    return Math.min(1, Math.max(0, Number.isFinite(x) ? x : 0));
  };
  return tail.map(norm);
}

// ===== 凡例 ==============================================================
/// 全ショートカットの凡例。key は表示用の見出し。
/// task 984: `?` / `H` は同じモーダル、`F` は auto-fit、`g + 数字` は層ジャンプ。
export const SHORTCUTS = Object.freeze([
  Object.freeze({ key: '1', label: 'camera: Overview' }),
  Object.freeze({ key: '2', label: 'camera: Router / Experts (Front)' }),
  Object.freeze({ key: '3', label: 'camera: Prediction Head (Top)' }),
  Object.freeze({ key: '4', label: 'camera: KV Band (Side)' }),
  Object.freeze({ key: '5', label: 'camera: Reset' }),
  Object.freeze({ key: 'F', label: 'camera: auto-fit (層数に合わせて全層を収める)' }),
  Object.freeze({ key: '[ / ]', label: 'layer inspector: 層を前後' }),
  Object.freeze({ key: 'g + 数字', label: 'layer inspector: 層へ直接ジャンプ (0..47)' }),
  Object.freeze({ key: 'p', label: 'fps オーバーレイの表示切替' }),
  Object.freeze({ key: 'c', label: 'カラーテーマの切替 (cyber / thermal / mono-high-contrast)' }),
  Object.freeze({ key: 's', label: 'PNG スナップショットの保存 (fly 中は後退)' }),
  Object.freeze({ key: 'm', label: 'reduced motion の手動切替 (idle orbit を止める)' }),
  Object.freeze({ key: 'f', label: 'fly mode (WASD + mouse, Esc で戻る)' }),
  Object.freeze({ key: 'space', label: 'demo の一時停止 / 再開' }),
  Object.freeze({ key: 'h', label: 'チャット欄の表示切替' }),
  Object.freeze({ key: '?', label: 'この凡例の表示切替 (H でも開閉 · Esc で閉じる)' }),
]);

/// 画面に出す凡例の行 (そのままコピー。画面は DOM を作るだけ)。
export function legendRows() {
  return SHORTCUTS.map((s) => ({ key: s.key, label: s.label }));
}

/// 凡例を開閉する。state.legend を書き換え、新しい値を返す。
export function toggleLegend(state) {
  const s = state && typeof state === 'object' ? state : {};
  s.legend = !s.legend;
  return s.legend;
}

/// キーイベント ({key, shiftKey}) が凡例の開閉か。
/// US 配列では `?` は Shift+/ なので両方受ける。
export function isLegendKey(e) {
  if (!e) return false;
  if (e.key === '?') return true;
  return e.key === '/' && !!e.shiftKey;
}

/// `?legend=1` のようなクエリ値 → 初期表示。真値は 1 / true / yes / on。
export function legendFromQuery(v) {
  return truthy(v);
}

// ===== task 939: カラーテーマ (純ロジック) ================================
// 3 つの curated なプリセット。ramp は 0..1 の値を色へ写すための 5 段の
// アンカーで、両端は必ず ramp[0] / ramp[last] に張り付く (範囲外は端で clamp)。
// 値はすべて固定の hex なので、同じ入力からは常に同じ色が出る (乱数なし)。
export const THEMES = Object.freeze([
  Object.freeze({
    name: 'cyber', bg: '#04060b', fg: '#cfeeff', dim: '#4d6470', accent: '#00eaff',
    ramp: Object.freeze(['#061018', '#0a3a4a', '#00b8d4', '#7ff0ff', '#ffffff']),
  }),
  Object.freeze({
    name: 'thermal', bg: '#0a0503', fg: '#ffe9d6', dim: '#6b4a3a', accent: '#ff8c1a',
    ramp: Object.freeze(['#100704', '#4a1608', '#c2410c', '#ffb020', '#fff3c4']),
  }),
  Object.freeze({
    name: 'mono-high-contrast', bg: '#000000', fg: '#ffffff', dim: '#808080', accent: '#ffffff',
    ramp: Object.freeze(['#000000', '#404040', '#a0a0a0', '#e0e0e0', '#ffffff']),
  }),
]);
export const THEME_COUNT = THEMES.length;

/// 添字を 0..THEME_COUNT-1 に丸める (負や範囲外は回り込む)。壊れた値は 0。
export function clampThemeIndex(i) {
  const n = THEME_COUNT;
  const k = Number.isFinite(Number(i)) ? Math.trunc(Number(i)) : 0;
  return ((k % n) + n) % n;
}

/// 添字 → テーマ (範囲外は回り込む)。
export function themeAt(i) {
  return THEMES[clampThemeIndex(i)];
}

/// 名前 → 添字。無ければ -1。
export function themeIndex(name) {
  const s = String(name == null ? '' : name).trim().toLowerCase();
  for (let i = 0; i < THEMES.length; i++) if (THEMES[i].name === s) return i;
  return -1;
}

/// 次のテーマの添字 (cyber -> thermal -> mono -> cyber)。
export function cycleTheme(i) {
  return (clampThemeIndex(i) + 1) % THEME_COUNT;
}

/// 添字 → テーマ名。
export function themeName(i) {
  return themeAt(i).name;
}

/// キーイベントがテーマ切替か (c / C)。
export function isThemeKey(e) {
  return !!e && (e.key === 'c' || e.key === 'C');
}

/// `#rgb` / `#rrggbb` → {r,g,b} (0..255)。壊れていれば null。
export function hexToRgb(hex) {
  const s = String(hex == null ? '' : hex).trim().replace(/^#/, '');
  let full = null;
  if (s.length === 3) full = s[0] + s[0] + s[1] + s[1] + s[2] + s[2];
  else if (s.length === 6) full = s;
  if (!full || !/^[0-9a-fA-F]{6}$/.test(full)) return null;
  return {
    r: parseInt(full.slice(0, 2), 16),
    g: parseInt(full.slice(2, 4), 16),
    b: parseInt(full.slice(4, 6), 16),
  };
}

/// {r,g,b} → `#rrggbb`。各成分は 0..255 に丸めて clamp する。壊れた値は null。
export function rgbToHex(r, g, b) {
  const c = [r, g, b].map((v) => Number(v));
  if (c.some((v) => !Number.isFinite(v))) return null;
  return '#' + c.map((v) => Math.min(255, Math.max(0, Math.round(v))).toString(16).padStart(2, '0')).join('');
}

/// ramp 上の位置 t (0..1) → rgb。範囲外は端、NaN は先頭。
export function themeRgbAt(i, t) {
  const ramp = themeAt(i).ramp;
  const x = Number.isFinite(Number(t)) ? Math.min(1, Math.max(0, Number(t))) : 0;
  const seg = ramp.length - 1;
  const p = x * seg;
  const k = Math.min(seg - 1, Math.floor(p));
  const u = p - k;
  const a = hexToRgb(ramp[k]), b = hexToRgb(ramp[k + 1]);
  return {
    r: a.r + (b.r - a.r) * u,
    g: a.g + (b.g - a.g) * u,
    b: a.b + (b.b - a.b) * u,
  };
}

/// ramp 上の位置 t (0..1) → `#rrggbb`。
export function themeHexAt(i, t) {
  const c = themeRgbAt(i, t);
  return rgbToHex(c.r, c.g, c.b);
}

/// テーマ → CSS 変数 (画面はこれを root に入れるだけ)。
export function themeCssVars(i) {
  const th = themeAt(i);
  return { '--bg': th.bg, '--fg': th.fg, '--dim': th.dim, '--vram': th.accent };
}

// ===== task 939: PNG スナップショット ====================================
/// 撮影時刻 → ファイル名。`denno3d-YYYYMMDD-HHMMSS.png` (ローカル時刻)。
/// 分・秒までなので連打しても同じ秒なら同じ名前 (上書き確認はブラウザ任せ)。
export function snapshotFilename(date, prefix = 'denno3d') {
  const d = date instanceof Date ? date : new Date(date);
  const p = String(prefix == null || prefix === '' ? 'denno3d' : prefix);
  if (Number.isNaN(d.getTime())) return p + '-snapshot.png';
  const pad = (n) => String(n).padStart(2, '0');
  return p + '-' + d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) +
    '-' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds()) + '.png';
}

/// キーイベントがスナップショットか (s / S)。
export function isSnapshotKey(e) {
  return !!e && (e.key === 's' || e.key === 'S');
}

// ===== task 939: reduced motion =========================================
// envReduced は prefers-reduced-motion の現在値、manualFreeze は M キーの
// 手動トグル。どちらかが真なら idle orbit を止める (OR 合成)。
export function createMotion(prefersReduced) {
  return { envReduced: !!prefersReduced, manualFreeze: false };
}

/// 実際に reduced motion が効いているか。
export function effectiveReduced(state) {
  return !!(state && (state.envReduced || state.manualFreeze));
}

/// prefers-reduced-motion の変化を反映し、新しい実効値を返す。
export function setEnvReduced(state, flag) {
  if (state && typeof state === 'object') state.envReduced = !!flag;
  return effectiveReduced(state);
}

/// M キー: 手動フリーズを反転し、新しい実効値を返す。
export function toggleReducedMotion(state) {
  if (state && typeof state === 'object') state.manualFreeze = !state.manualFreeze;
  return effectiveReduced(state);
}

/// idle orbit を回してよいか (reduced motion でなければ true)。
export function orbitEnabled(state) {
  return !effectiveReduced(state);
}

/// キーイベントが reduced motion の手動切替か (m / M)。
export function isMotionKey(e) {
  return !!e && (e.key === 'm' || e.key === 'M');
}

// ===== task 984: ショートカット凡例モーダルの状態 ==========================
// 「開いているか」だけを持つ素の状態。DOM は index3d.html が持つ。
export function createHelpState(open) {
  return { open: !!open };
}

/// モーダルが開いているか。
export function helpOpen(state) {
  return !!(state && state.open);
}

/// 開閉を明示的に設定し、新しい値を返す。
export function setHelpOpen(state, on) {
  if (state && typeof state === 'object') state.open = !!on;
  return helpOpen(state);
}

/// 表示切替 (開 → 閉 / 閉 → 開)。新しい値を返す。
export function toggleHelp(state) {
  if (!state || typeof state !== 'object') return true;
  state.open = !state.open;
  return state.open;
}

/// 閉じる。新しい値を返す (常に false)。
export function closeHelp(state) {
  return setHelpOpen(state, false);
}

/// キーイベントが凡例モーダルの開閉か (`?` / Shift+/ / `H`)。
/// 小文字 `h` はチャット欄の表示切替なので受けない (index3d.html 側で分ける)。
export function isHelpKey(e) {
  if (!e) return false;
  if (e.key === '?') return true;
  if (e.key === '/' && !!e.shiftKey) return true;
  if (e.key === 'H') return true;
  return e.key === 'h' && !!e.shiftKey;
}

/// キーイベントがモーダルを閉じるキーか (Escape)。
export function isHelpCloseKey(e) {
  return !!e && e.key === 'Escape';
}

/// モーダルに並べる行 (SHORTCUTS の写し)。
export function helpRows() {
  return legendRows();
}

/// モーダルの 1 行 "key — label"。欠けた入力でも落ちない。
export function formatHelpRow(row) {
  if (!row) return '';
  const k = String(row.key == null ? '' : row.key);
  const l = String(row.label == null ? '' : row.label);
  if (k && l) return k + ' — ' + l;
  return k || l;
}

// ===== task 984: auto-fit (層数からカメラの target/position を決める) ======
// index3d.html のシーン配置 (X の並び / zOf(il) = -il * ZSTEP / 3 段の Y) に合わせた
// 固定のバウンディングボックス。乱数も時刻も使わないので、同じ N なら常に同じ絵になる。
//   Z: 層 0 が z=0、層 i が z=-i*zStep。1 層ぶんの厚みは 2*halfDepth。
export const FIT_BOUNDS = Object.freeze({
  xMin: -14.8, xMax: 7.6,     // deck() の x0..x1
  yMin: -7.0, yMax: 7.0,      // TIER_Y の下端..上端
  zStep: 1.7,                 // index3d.html の ZSTEP
  halfDepth: 0.5,
  margin: 1.18,               // バウンディング球に足す余白
  fov: 39,                    // PerspectiveCamera の垂直 fov (度)
  aspect: 16 / 9,
  minDistance: 8,             // controls.minDistance
  maxDistance: 380,           // controls.maxDistance
});

/// 既存の Overview 視点 (pos - target = [33, 31, 72.6]) と同じ向き。正規化して使う。
const FIT_DIR_RAW = Object.freeze([33, 31, 72.6]);

function numOr(v, d) {
  const x = Number(v);
  return Number.isFinite(x) ? x : d;
}

function posOr(v, d) {
  const x = Number(v);
  return Number.isFinite(x) && x > 0 ? x : d;
}

function norm3(v) {
  const l = Math.hypot(v[0], v[1], v[2]);
  return l > 0 ? [v[0] / l, v[1] / l, v[2] / l] : [0, 0, 1];
}

function round4(v) {
  return Math.round(v * 10000) / 10000;
}

/// auto-fit の視線方向 (単位ベクトル)。
export const FIT_DIR = Object.freeze(norm3(FIT_DIR_RAW));

/// 層数 N (>= 1) → シーンの軸平行バウンディングボックス。
/// min / max / center / size / radius を返す。N が壊れていれば null。
export function sceneBounds(nLayers, opts) {
  const n = Number(nLayers);
  if (!Number.isFinite(n) || Math.trunc(n) < 1) return null;
  const count = Math.trunc(n);
  const o = opts && typeof opts === 'object' ? opts : {};
  const zStep = posOr(o.zStep, FIT_BOUNDS.zStep);
  const half = Math.max(0, numOr(o.halfDepth, FIT_BOUNDS.halfDepth));
  const min = [FIT_BOUNDS.xMin, FIT_BOUNDS.yMin, -(count - 1) * zStep - half];
  const max = [FIT_BOUNDS.xMax, FIT_BOUNDS.yMax, half];
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return {
    layers: count,
    min,
    max,
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    size,
    radius: 0.5 * Math.hypot(size[0], size[1], size[2]),
  };
}

/// 層数 N (>= 1) → カメラの goal {pos, target}。
/// バウンディング球が縦横どちらの fov にも収まる距離を出し、OrbitControls の
/// 距離範囲に clamp する。N が壊れていれば null。
export function fitCameraGoal(nLayers, opts) {
  const b = sceneBounds(nLayers, opts);
  if (!b) return null;
  const o = opts && typeof opts === 'object' ? opts : {};
  const fov = Math.min(179, Math.max(1, numOr(o.fov, FIT_BOUNDS.fov)));
  const aspect = posOr(o.aspect, FIT_BOUNDS.aspect);
  const margin = posOr(o.margin, FIT_BOUNDS.margin);
  const minD = posOr(o.minDistance, FIT_BOUNDS.minDistance);
  const maxD = posOr(o.maxDistance, FIT_BOUNDS.maxDistance);
  const vHalf = (fov * Math.PI) / 360;
  const hHalf = Math.atan(Math.tan(vHalf) * aspect);
  const sinHalf = Math.sin(Math.max(1e-6, Math.min(vHalf, hHalf)));
  const want = (b.radius / sinHalf) * margin;
  const distance = Math.min(maxD, Math.max(minD, want));
  const d = FIT_DIR;
  const pos = [
    round4(b.center[0] + d[0] * distance),
    round4(b.center[1] + d[1] * distance),
    round4(b.center[2] + d[2] * distance),
  ];
  const target = [round4(b.center[0]), round4(b.center[1]), round4(b.center[2])];
  return {
    layers: b.layers,
    pos,
    target,
    distance: round4(distance),
    radius: round4(b.radius),
  };
}

/// キーイベントが auto-fit か (`F` = Shift+f)。小文字 `f` は fly mode なので受けない。
export function isAutoFitKey(e) {
  if (!e) return false;
  if (e.key === 'F') return true;
  return e.code === 'KeyF' && !!e.shiftKey;
}

// ===== task 984: 層ジャンプ (g + 数字) ====================================
/// 受け付ける桁数。47 までなら 2 桁で足りるが、層数が増えても壊れないよう 3 桁。
export const LAYER_JUMP_MAX_DIGITS = 3;

/// 層ジャンプの入力状態。layers は丸め先の層数。
export function createLayerJump(nLayers) {
  const n = Number(nLayers);
  const layers = Number.isFinite(n) && Math.trunc(n) >= 1 ? Math.trunc(n) : LAYER_COUNT;
  return { layers, digits: '', active: false };
}

/// `g` を押した: 入力待ちに入る (数字はまだ空)。
export function beginLayerJump(state) {
  if (!state || typeof state !== 'object') return false;
  state.digits = '';
  state.active = true;
  return true;
}

/// 入力待ちを捨てる。新しい active (常に false) を返す。
export function cancelLayerJump(state) {
  if (state && typeof state === 'object') {
    state.digits = '';
    state.active = false;
  }
  return false;
}

/// キーイベントが層ジャンプの開始か (g / G)。
export function isLayerJumpKey(e) {
  return !!e && (e.key === 'g' || e.key === 'G');
}

/// キーイベント → 0..9 の数字。数字キー以外は null。
export function digitFromKey(e) {
  if (!e) return null;
  const k = String(e.key == null ? '' : e.key);
  if (k.length !== 1 || k < '0' || k > '9') return null;
  return Number(k);
}

/// 数字列 → [0, nLayers-1] の層番号。空 / 数字以外は null。
export function parseLayerDigits(digits, nLayers = LAYER_COUNT) {
  const s = String(digits == null ? '' : digits);
  if (!/^[0-9]+$/.test(s)) return null;
  const n = Number(nLayers);
  const layers = Number.isFinite(n) && Math.trunc(n) >= 1 ? Math.trunc(n) : LAYER_COUNT;
  return Math.min(layers - 1, Math.max(0, parseInt(s, 10)));
}

/// 数字を 1 つ足して、その時点の層番号 (丸め済み) を返す。
/// 入力待ちでない / 数字でないときは null。桁あふれは古い方から捨てる。
export function pushLayerDigit(state, digit) {
  if (!state || typeof state !== 'object' || !state.active) return null;
  // 数値 (0..9) か 1 文字の数字列だけを受ける (null / true / {} は受けない)
  const d = typeof digit === 'number' ? digit
    : (typeof digit === 'string' && /^[0-9]$/.test(digit) ? Number(digit) : NaN);
  if (!Number.isInteger(d) || d < 0 || d > 9) return null;
  if (state.digits.length >= LAYER_JUMP_MAX_DIGITS) state.digits = state.digits.slice(1);
  state.digits += String(d);
  return parseLayerDigits(state.digits, state.layers);
}

/// 保留中の入力を確定して層番号を返し、状態をリセットする。空なら null。
export function jumpToLayer(state) {
  if (!state || typeof state !== 'object') return null;
  const il = parseLayerDigits(state.digits, state.layers);
  state.digits = '';
  state.active = false;
  return il;
}
