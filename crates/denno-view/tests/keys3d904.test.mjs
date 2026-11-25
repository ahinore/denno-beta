// keys3d.mjs (task 904) の純ロジック。
// カメラプリセット (1..5) / 補間状態 / 層インスペクタ / fps オーバーレイ。
// node --test crates/denno-view/tests/*.test.mjs で走る。
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CAMERA_PRESETS, PRESET_KEYS, BOOKMARKS,
  presetForKey, presetGoal, easeSmoothstep,
  createCamAnim, startCamAnim, stepCamAnim, camAnimActive,
  ATTENTION_INTERVAL, LAYER_COUNT, layerKind, clampLayer, stepLayer,
  activationStats, formatMagnitude, expertSummary, formatExpertSummary,
  isLayerStepKey, isFpsKey, overlayFromQuery,
  createFpsMeter, noteFrame, fpsSummary, fpsText, fpsBars,
  SHORTCUTS, legendRows,
} from '../src/keys3d.mjs';

const MIN_DIST = 8, MAX_DIST = 380;   // index3d.html の controls.minDistance/maxDistance

function dist(pos, target) {
  return Math.hypot(pos[0] - target[0], pos[1] - target[1], pos[2] - target[2]);
}

test('カメラプリセットは 1..5 の 5 視点で、名前が決まっている', () => {
  assert.deepEqual(PRESET_KEYS, ['1', '2', '3', '4', '5']);
  assert.deepEqual(CAMERA_PRESETS.map((p) => p.key), PRESET_KEYS);
  assert.deepEqual(CAMERA_PRESETS.map((p) => p.name), [
    'Overview', 'Front', 'Top', 'Side', 'Reset',
  ]);
});

test('プリセットの座標は有限で、OrbitControls の距離範囲に収まる', () => {
  for (const p of CAMERA_PRESETS) {
    assert.equal(p.pos.length, 3, p.name);
    assert.equal(p.target.length, 3, p.name);
    for (const v of [...p.pos, ...p.target]) assert.ok(Number.isFinite(v), p.name);
    const d = dist(p.pos, p.target);
    assert.ok(d >= MIN_DIST && d <= MAX_DIST, p.name + ' distance ' + d.toFixed(1));
  }
  // 5 つのプリセットは同じ絵にならない (Top と Side は視点が違う)
  const topside = CAMERA_PRESETS.find((p) => p.name === 'Top');
  const side = CAMERA_PRESETS.find((p) => p.name === 'Side');
  assert.notDeepEqual(topside.pos, side.pos);
  // Reset は起動時のカメラ (index3d.html の camera.position.set(29,30,47))
  const reset = CAMERA_PRESETS.find((p) => p.name === 'Reset');
  assert.deepEqual([...reset.pos], [29, 30, 47]);
});

test('presetForKey は 1..5 だけで、6 以上や空白は受けない', () => {
  assert.equal(presetForKey('1').name, 'Overview');
  assert.equal(presetForKey('5').name, 'Reset');
  assert.equal(presetForKey(3).name, 'Top');
  assert.equal(presetForKey('6'), null);
  assert.equal(presetForKey(' 1'), null);
  assert.equal(presetForKey(''), null);
  assert.equal(presetForKey(null), null);
  assert.equal(presetForKey(undefined), null);
});

test('presetGoal はコピーを返し、壊れた入力は null', () => {
  const p = presetForKey('2');
  const g = presetGoal(p);
  assert.deepEqual(g.pos, [...p.pos]);
  g.pos[0] = 999;
  assert.notEqual(p.pos[0], 999);
  assert.equal(presetGoal(null), null);
  assert.equal(presetGoal({ key: '1', pos: [0, 0], target: [0, 0, 0] }), null);
  assert.equal(presetGoal({ key: '1', pos: [0, 0, 0], target: [0, 0, NaN] }), null);
});

test('プリセットは 813 のブックマークと同じ名前空間を壊さない', () => {
  // BOOKMARKS は 1..4 のまま (task 813 の契約)
  assert.deepEqual(BOOKMARKS.map((b) => b.key), ['1', '2', '3', '4']);
  // 1 番は 813 の Overview と同じ視点
  const b1 = BOOKMARKS[0], p1 = presetForKey('1');
  assert.deepEqual([...b1.pos], [...p1.pos]);
  assert.deepEqual([...b1.target], [...p1.target]);
});

test('easeSmoothstep は両端 0/1 で単調増加', () => {
  assert.equal(easeSmoothstep(0), 0);
  assert.equal(easeSmoothstep(1), 1);
  assert.equal(easeSmoothstep(-1), 0);
  assert.equal(easeSmoothstep(2), 1);
  assert.equal(easeSmoothstep(NaN), 0);
  let prev = -1;
  for (let i = 0; i <= 10; i++) {
    const v = easeSmoothstep(i / 10);
    assert.ok(v >= prev, 'not monotonic at ' + i);
    prev = v;
  }
  assert.ok(Math.abs(easeSmoothstep(0.5) - 0.5) < 1e-12);
});

test('startCamAnim / stepCamAnim は決め打ちの経路を補間する', () => {
  const st = createCamAnim(0.5);
  assert.equal(st.active, false);
  assert.equal(camAnimActive(st), false);
  assert.equal(stepCamAnim(st, 0.016), null);      // 動いていない

  const goal = presetGoal(presetForKey('3'));
  const from = { pos: [0, 0, 0], target: [0, 0, 0] };
  assert.equal(startCamAnim(st, goal, from), true);
  assert.equal(camAnimActive(st), true);

  const first = stepCamAnim(st, 0.1);
  assert.equal(first.done, false);
  // 出だしは from の近く (smoothstep)
  assert.ok(first.pos[1] > 0 && first.pos[1] < goal.pos[1] * 0.5, String(first.pos[1]));

  const last = stepCamAnim(st, 1.0);               // duration を超えて進める
  assert.equal(last.done, true);
  assert.deepEqual(last.pos, [...goal.pos]);
  assert.deepEqual(last.target, [...goal.target]);
  assert.equal(camAnimActive(st), false);

  // 同じ入力なら毎回同じ経路 (途中の値も一致する)
  const a = createCamAnim(0.5), b = createCamAnim(0.5);
  const g = presetGoal(presetForKey('2'));
  startCamAnim(a, g, from); startCamAnim(b, g, from);
  for (let i = 0; i < 5; i++) {
    assert.deepEqual(stepCamAnim(a, 0.07).pos, stepCamAnim(b, 0.07).pos);
  }
});

test('stepCamAnim は dt が壊れていても落ちない', () => {
  const st = createCamAnim(0.2);
  startCamAnim(st, presetGoal(presetForKey('1')), { pos: [1, 2, 3], target: [0, 0, 0] });
  const r = stepCamAnim(st, NaN);
  assert.equal(r.done, false);
  assert.deepEqual(r.pos, [1, 2, 3]);
});

test('startCamAnim は goal が壊れていれば false で状態を変えない', () => {
  const st = createCamAnim();
  assert.equal(startCamAnim(st, null), false);
  assert.equal(startCamAnim(st, { pos: [0, 0], target: [0, 0, 0] }), false);
  assert.equal(st.active, false);
  assert.equal(startCamAnim(null, { pos: [0, 0, 0], target: [0, 0, 0] }), false);
});

test('layerKind は 4 層に 1 層を FullAttn にする', () => {
  assert.equal(ATTENTION_INTERVAL, 4);
  assert.equal(LAYER_COUNT, 48);
  assert.equal(layerKind(0), 'GDN');
  assert.equal(layerKind(1), 'GDN');
  assert.equal(layerKind(2), 'GDN');
  assert.equal(layerKind(3), 'FullAttn');
  assert.equal(layerKind(7), 'FullAttn');
  assert.equal(layerKind(47), 'FullAttn');
  // index3d.html の isQSA (il+1)%4===0 と一致する
  for (let il = 0; il < LAYER_COUNT; il++) {
    assert.equal(layerKind(il), (il + 1) % 4 === 0 ? 'FullAttn' : 'GDN', 'L' + il);
  }
});

test('clampLayer と stepLayer は端で回り込む', () => {
  assert.equal(clampLayer(-5), 0);
  assert.equal(clampLayer(99), 47);
  assert.equal(clampLayer(12.9), 12);
  assert.equal(clampLayer(NaN), 0);
  assert.equal(stepLayer(0, -1), 47);
  assert.equal(stepLayer(47, 1), 0);
  assert.equal(stepLayer(10, 1), 11);
  assert.equal(stepLayer(10, -1), 9);
  assert.equal(stepLayer(10, 0), 10);
  assert.equal(stepLayer(10, 48), 10);
  assert.equal(stepLayer('x', 1), 1);
});

test('activationStats は max / mean / rms / lit を出す', () => {
  const st = activationStats([0, 0.5, 1, 0.25, -3, NaN]);
  assert.equal(st.count, 6);
  assert.equal(st.max, 1);
  assert.equal(st.lit, 3);
  assert.ok(Math.abs(st.mean - 1.75 / 6) < 1e-12);
  assert.ok(st.rms > st.mean, 'rms は平均以上');
  assert.equal(formatMagnitude(st), 'max 1.000 · rms 0.468 · lit 3');

  const empty = activationStats([]);
  assert.deepEqual([empty.max, empty.mean, empty.rms, empty.lit, empty.count], [0, 0, 0, 0, 0]);
  assert.equal(activationStats(null).count, 0);
  assert.equal(formatMagnitude(null), '--');
});

test('expertSummary は重みの大きい順に limit 件 + 残り件数', () => {
  const sum = expertSummary([17, 233, 41, 9], [0.11, 0.31, 0.21, 0.05], 3);
  assert.deepEqual(sum.top.map((e) => e.index), [233, 41, 17]);
  assert.equal(sum.more, 1);
  assert.equal(sum.count, 4);
  assert.equal(formatExpertSummary(sum), 'E233 0.31 · E41 0.21 · E17 0.11 · +1');

  // 同点は index 昇順で決める (決め打ち)
  const tie = expertSummary([9, 4], [0.2, 0.2], 2);
  assert.deepEqual(tie.top.map((e) => e.index), [4, 9]);

  assert.equal(formatExpertSummary(expertSummary([], [], 3)), '--');
  assert.equal(formatExpertSummary(null), '--');
  // 重みが無くても index は出す
  assert.equal(formatExpertSummary(expertSummary([7], undefined, 3)), 'E7 0.00');
});

test('[ ] と p のキー判定', () => {
  assert.equal(isLayerStepKey({ key: '[' }), -1);
  assert.equal(isLayerStepKey({ key: ']' }), 1);
  assert.equal(isLayerStepKey({ key: '[' , shiftKey: true }), -1);
  assert.equal(isLayerStepKey({ key: '{' }), 0);
  assert.equal(isLayerStepKey({ key: '1' }), 0);
  assert.equal(isLayerStepKey(null), 0);

  assert.equal(isFpsKey({ key: 'p' }), true);
  assert.equal(isFpsKey({ key: 'P' }), true);
  assert.equal(isFpsKey({ key: 'f' }), false);
  assert.equal(isFpsKey({ code: 'KeyP' }), false);
  assert.equal(isFpsKey(null), false);
});

test('overlayFromQuery は真値だけ true', () => {
  for (const v of ['1', 'true', 'yes', 'on', 'On']) assert.equal(overlayFromQuery(v), true, String(v));
  for (const v of ['0', '', null, undefined, 'no', 'off']) assert.equal(overlayFromQuery(v), false, String(v));
});

test('fps 計測器は平均フレーム間隔から fps を出す', () => {
  const m = createFpsMeter(4);
  assert.equal(fpsText(fpsSummary(m)), '--');            // 2 フレーム未満
  assert.equal(noteFrame(m, NaN), null);                 // 壊れた時刻は捨てる
  let s = noteFrame(m, 1000);
  assert.equal(s.frames, 0);
  s = noteFrame(m, 1016.7);
  assert.equal(s.frames, 1);
  s = noteFrame(m, 1033.4);
  assert.equal(s.frames, 2);
  assert.ok(Math.abs(s.fps - 59.9) < 0.5, String(s.fps));
  assert.equal(fpsText(s), 'fps 60 · 16.7 ms');
  // 逆行する時刻はフレームとして数えない
  const before = m.frames;
  noteFrame(m, 1000);
  assert.equal(m.frames, before);
  // 履歴は cap で切る
  for (let i = 0; i < 10; i++) noteFrame(m, 2000 + i * 16);
  assert.equal(m.hist.length, 4);
  assert.equal(fpsSummary(m).hist.length, 4);
});

test('fpsBars は 0..1 に収まる', () => {
  const sum = { hist: [16.7, 8, 40, 24] };
  const bars = fpsBars(sum, 24);
  assert.equal(bars.length, 4);
  assert.ok(Math.abs(bars[0] - (16.7 - 8) / 32) < 1e-9);
  assert.equal(bars[1], 0);
  assert.equal(bars[2], 1);
  for (const b of bars) assert.ok(b >= 0 && b <= 1);
  assert.deepEqual(fpsBars(null), []);
  assert.deepEqual(fpsBars({ hist: [] }), []);
});

test('凡例は 5 つのプリセット名と新しいキーを載せる', () => {
  const rows = legendRows();
  const keys = rows.map((r) => r.key);
  for (const k of ['1', '2', '3', '4', '5', '[ / ]', 'p', 'f', 'space', 'c', '?']) {
    assert.ok(keys.includes(k), 'missing ' + k);
  }
  assert.equal(rows.length, SHORTCUTS.length);
  // 各プリセット名がどこかの行に出る
  for (const p of CAMERA_PRESETS) {
    assert.ok(rows.some((r) => r.label.includes(p.name)), 'no legend row for ' + p.name);
  }
  // 813 のブックマーク名も残っている (既存の凡例を壊さない)
  for (const name of ['Router / Experts', 'Prediction Head', 'KV Band']) {
    assert.ok(rows.some((r) => r.label.includes(name)), 'lost ' + name);
  }
});
