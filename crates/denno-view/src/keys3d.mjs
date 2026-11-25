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
export const SHORTCUTS = Object.freeze([
  Object.freeze({ key: '1', label: 'camera: Overview' }),
  Object.freeze({ key: '2', label: 'camera: Router / Experts (Front)' }),
  Object.freeze({ key: '3', label: 'camera: Prediction Head (Top)' }),
  Object.freeze({ key: '4', label: 'camera: KV Band (Side)' }),
  Object.freeze({ key: '5', label: 'camera: Reset' }),
  Object.freeze({ key: '[ / ]', label: 'layer inspector: 層を前後' }),
  Object.freeze({ key: 'p', label: 'fps オーバーレイの表示切替' }),
  Object.freeze({ key: 'f', label: 'fly mode (WASD + mouse, Esc で戻る)' }),
  Object.freeze({ key: 'space', label: 'demo の一時停止 / 再開' }),
  Object.freeze({ key: 'c', label: 'チャット欄の表示切替' }),
  Object.freeze({ key: '?', label: 'この凡例の表示切替' }),
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
