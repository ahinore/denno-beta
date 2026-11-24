// denno-view /3d のカメラブックマークとキーボード凡例の純ロジック (DOM 非依存)。
//
// task 813:
//   1. 数字キー 1..4 でカメラを固定の見どころへ飛ばす
//      (1=全体, 2=router / experts, 3=prediction head, 4=KV 帯)。
//   2. `?` で開閉するキーボードショートカットの凡例。
//
// 座標は index3d.html のシーン配置 (X の並び / zOf(il) / KV 帯 / topZ) に合わせる。
// 画面は bookmarkForKey の pos/target を camera と controls.target に入れるだけ。

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

/// 全ショートカットの凡例。key は表示用の見出し。
export const SHORTCUTS = Object.freeze([
  Object.freeze({ key: '1', label: 'camera: Overview' }),
  Object.freeze({ key: '2', label: 'camera: Router / Experts' }),
  Object.freeze({ key: '3', label: 'camera: Prediction Head' }),
  Object.freeze({ key: '4', label: 'camera: KV Band' }),
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
  const s = String(v == null ? '' : v).toLowerCase();
  return s === '1' || s === 'true' || s === 'yes' || s === 'on';
}
