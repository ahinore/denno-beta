// denno-view /2d の HUD とプリセット (hud2d.mjs) の単体テスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
//
// 見ているもの:
//   1. HUD の状態機械 (TTFT / tok/s は TTFT を除いた区間で数える / usage で直す)
//   2. VRAM ヒット率と KV 使い回しトークンの取り込み
//   3. 投機 (MTP) の採用率 — ExpertSkip (kind 17) と未来 kind (>= 18) の区別
//   4. プリセットの保存 → 読み込みの往復と、サーバーと同じ検証規則
import test from 'node:test';
import assert from 'node:assert/strict';

import { DEFAULT_SHAPE_RAW, addItem, createSelection, selectionToBody } from '../src/view2d.mjs';
import {
  HUD_LABELS, PRESET_KIND, PRESET_SCHEMA, SPEC_KIND_MIN,
  buildPreset, checkedBody, createHud, fmtDuration, fmtPercent, fmtRate, fmtTokens,
  hudBeginRequest, hudEndRequest, hudFields, hudOnCache, hudOnFrame, hudOnRawKind,
  hudOnStats, hudOnToken, hudText, isSpecKind, parsePreset, presetFileName,
  presetSummary, specSample, validateInterveneBody,
} from '../src/hud2d.mjs';

// ---- 1. HUD の状態機械 ----------------------------------------------------

test('createHud: 初期値は全部 "-" で mtp は未受信', () => {
  const h = createHud();
  assert.equal(h.ttftMs, null);
  assert.equal(h.tokPerSec, null);
  assert.equal(h.mtp.seen, false);
  const vals = hudFields(h).map((f) => f.value);
  assert.deepEqual(vals, ['-', '-', '-', '-', '-']);
  assert.deepEqual(hudFields(h).map((f) => f.label), ['ttft', 'tok/s', 'vram hit', 'kv reuse', 'mtp accept']);
  assert.equal(HUD_LABELS.length, 5);
});

test('TTFT は最初のトークンで確定し、tok/s は TTFT を除いた区間で数える', () => {
  const h = createHud();
  assert.equal(hudBeginRequest(h, 1000), true);
  assert.equal(h.ttftMs, null);
  hudOnToken(h, 1400);                       // 最初のトークン
  assert.equal(h.ttftMs, 400);
  assert.equal(h.tokPerSec, null, '1 トークンでは速度を出さない');
  hudOnToken(h, 1600);
  hudOnToken(h, 1800);
  assert.equal(h.reqTokens, 3);
  assert.ok(Math.abs(h.tokPerSec - 5) < 1e-9, '3 tok / 0.4 s = 5 tok/s (2 つ目以降)');
  assert.equal(hudEndRequest(h, 2200, 11), true);
  assert.ok(Math.abs(h.tokPerSec - 25) < 1e-9, 'usage の 11 tok で数え直す (10 / 0.4 s)');
  assert.ok(Math.abs(h.lastSecs - 1.2) < 1e-9);
  assert.equal(h.reqActive, false);
});

test('usage が無い / トークン 0 の応答でも壊れない', () => {
  const h = createHud();
  hudBeginRequest(h, 0);
  hudEndRequest(h, 500, 0);
  assert.equal(h.ttftMs, null);
  assert.equal(h.tokPerSec, null);
  assert.ok(Math.abs(h.lastSecs - 0.5) < 1e-9);
  assert.equal(hudOnToken(h, 600), false, '終了後のトークンは無視');
  assert.equal(hudEndRequest(h, 700, 5), false, '二重終了は無視');
});

test('n 個まとめてのトークンと、同時刻のトークン (0 除算しない)', () => {
  const h = createHud();
  hudBeginRequest(h, 0);
  hudOnToken(h, 100, 5);
  assert.equal(h.ttftMs, 100);
  hudOnToken(h, 100, 5);                     // 同じ時刻
  assert.equal(h.tokPerSec, null, '区間 0 では速度を出さない');
  hudOnToken(h, 600, 0);                     // 0 は 1 に丸める
  assert.equal(h.reqTokens, 11);
  assert.ok(h.tokPerSec > 0);
});

// ---- 2. サーバーの状態 ----------------------------------------------------

test('hudOnStats: VRAM tier のヒット率を % 表示に (変化が無ければ false)', () => {
  const h = createHud();
  const view = { rows: [{ key: 'vram', hit: 0.9123 }, { key: 'ram', hit: 0.1 }, { key: 'mmap', hit: 0 }] };
  assert.equal(hudOnStats(h, view), true);
  assert.equal(h.vramHit, 0.9123);
  assert.equal(hudFields(h).find((f) => f.key === 'vram').value, '91%');
  assert.equal(hudOnStats(h, view), false, '同じ値なら変わらない');
  assert.equal(hudOnStats(h, { rows: [{ hit: null }] }), true);
  assert.equal(hudFields(h).find((f) => f.key === 'vram').value, '-', 'hit が無ければ -');
});

test('hudOnCache: main のトークン数 = 次のターンで使い回す KV', () => {
  const h = createHud();
  assert.equal(hudOnCache(h, { main: { tokens: 3826, epoch: 4 }, aux: null }), true);
  assert.equal(hudFields(h).find((f) => f.key === 'kv').value, '3,826');
  assert.equal(hudOnCache(h, { main: { tokens: 3826, epoch: 4 } }), false);
  assert.equal(hudOnCache(h, { main: { tokens: 0, epoch: 5 } }), true);
  assert.equal(hudFields(h).find((f) => f.key === 'kv').value, '0');
  hudOnCache(h, { main: null });
  assert.equal(hudFields(h).find((f) => f.key === 'kv').value, '-');
});

// ---- 3. 投機 (MTP) --------------------------------------------------------

test('isSpecKind: 17 は ExpertSkip なので投機ではない / 18 以降は投機', () => {
  assert.equal(SPEC_KIND_MIN, 18);
  assert.equal(isSpecKind(17), false);
  assert.equal(isSpecKind('ExpertSkip'), false);
  assert.equal(isSpecKind(18), true);
  assert.equal(isSpecKind('kind21'), true);
  assert.equal(isSpecKind('kind2'), false);
  assert.equal(isSpecKind('MtpStats'), true);
  assert.equal(isSpecKind('SpecAccept'), true);
  assert.equal(isSpecKind('ServerInfo'), false);
  assert.equal(isSpecKind(null), false);
});

test('specSample: accepted/proposed と rate (%) の両方を受ける', () => {
  assert.deepEqual(specSample({ accepted: 7, proposed: 10 }), { accepted: 7, proposed: 10, rate: 0.7 });
  assert.deepEqual(specSample({ accept_rate: 87.5 }), { accepted: null, proposed: null, rate: 0.875 });
  assert.deepEqual(specSample({ rate: 0.25 }), { accepted: null, proposed: null, rate: 0.25 });
  assert.deepEqual(specSample({ proposed: 0, accepted: 0 }), { accepted: 0, proposed: 0, rate: null });
  assert.equal(specSample({ foo: 1 }), null);
  assert.equal(specSample(null), null);
  assert.deepEqual(specSample({ accepted: -3, proposed: 4 }), { accepted: 0, proposed: 4, rate: 0 });
});

test('hudOnFrame: 累積の採用率を出し、ExpertSkip では反応しない', () => {
  const h = createHud();
  assert.equal(hudOnFrame(h, { kind: 'ExpertSkip', layer: 0, experts: [1] }), false);
  assert.equal(h.mtp.seen, false);
  assert.equal(hudOnFrame(h, { kind: 'MtpStats', accepted: 3, proposed: 4 }), true);
  assert.equal(h.mtp.seen, true);
  assert.equal(h.mtp.frames, 1);
  assert.ok(Math.abs(h.mtp.rate - 0.75) < 1e-9);
  assert.equal(hudFields(h).find((f) => f.key === 'mtp').value, '75% (3/4)');
  hudOnFrame(h, { kind: 'MtpStats', accepted: 1, proposed: 2 });
  assert.ok(Math.abs(h.mtp.rate - 4 / 6) < 1e-9, '累積 4/6');
  assert.equal(hudFields(h).find((f) => f.key === 'mtp').value, '67% (4/6)');
  // rate だけのフレームは累積の代わりにその値
  const h2 = createHud();
  hudOnFrame(h2, { kind: 'SpecStats', accept_rate: 0.5 });
  assert.equal(h2.mtp.rate, 0.5);
  assert.equal(h2.mtp.proposed, 0);
  assert.equal(hudFields(h2).find((f) => f.key === 'mtp').value, '50%');
});

test('hudOnRawKind: デコードできない未来 kind は「来ている」だけ記録', () => {
  const h = createHud();
  assert.equal(hudOnRawKind(h, 17), false, '17 は既知 (ExpertSkip)');
  assert.equal(hudOnRawKind(h, 18), true);
  assert.equal(h.mtp.seen, true);
  assert.equal(h.mtp.rate, null);
  assert.equal(hudFields(h).find((f) => f.key === 'mtp').value, 'telemetry');
  assert.equal(hudOnRawKind(h, 'ExpertSkip'), false);
});

// ---- 表示の書式 -----------------------------------------------------------

test('fmt* は欠けた値を "-" にし、桁を丸める', () => {
  assert.equal(fmtDuration(null), '-');
  assert.equal(fmtDuration(412), '412 ms');
  assert.equal(fmtDuration(1912.7), '1.91 s');
  assert.equal(fmtRate(null), '-');
  assert.equal(fmtRate(5), '5.0');
  assert.equal(fmtRate(12.34), '12.3');
  assert.equal(fmtRate(123.4), '123');
  assert.equal(fmtPercent(0.916), '92%');
  assert.equal(fmtPercent(null), '-');
  assert.equal(fmtTokens(3826), '3,826');
  assert.equal(fmtTokens(null), '-');
  assert.equal(hudText(createHud()), 'ttft - · tok/s - · vram hit - · kv reuse - · mtp accept -');
});

// ---- 4. プリセット --------------------------------------------------------

const SHAPE = { layers: 48, n_expert: 512, n_expert_used: 10, n_hc: 4, n_embd: 2560 };
const AT = '2026-11-25T00:00:00.000Z';

function sampleInput() {
  return {
    name: 'dbg_A',
    selector: 'expert: count == 0 and layer >= 40',
    checked: [{ type: 'expert', layer: 41, expert: 7 }, { type: 'block', layer: 3, block: 'moe' }],
    intervene: {
      experts_off: [[41, 0], [3, 9], [41, 1]],
      blocks_off: [[3, 'shared']],
      streams_off: [[5, 2]],
    },
    steer: {
      terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -0.5 }],
      layers: [4, 20], alpha: 0.75, mode: 'project',
    },
    shape: SHAPE,
  };
}

test('buildPreset → parsePreset の往復 (checked は昇順・重複なしになる)', () => {
  const p = buildPreset(sampleInput(), { now: AT });
  assert.equal(p.schema, PRESET_SCHEMA);
  assert.equal(p.kind, PRESET_KIND);
  assert.equal(p.name, 'dbg_A');
  assert.equal(p.created, AT);
  assert.deepEqual(p.checked.experts_off, [[41, 7]]);
  assert.deepEqual(p.checked.blocks_off, [[3, 'moe']]);
  assert.deepEqual(p.intervene.experts_off, [[3, 9], [41, 0], [41, 1]], '層 → expert の昇順');
  assert.deepEqual(p.intervene.blocks_off, [[3, 'shared']]);
  assert.deepEqual(p.steer, {
    terms: [{ snapshot: 'A', coef: 1 }, { snapshot: 'B', coef: -0.5 }],
    layers: [4, 20], alpha: 0.75, mode: 'project',
  });
  const q = parsePreset(JSON.stringify(p));
  assert.deepEqual(q, p);
  assert.equal(checkedBody(p.checked).experts_off.length, 1);
});

test('選択 (Set) を checkedBody で本文に寄せられる', () => {
  const sel = createSelection();
  addItem(sel, { type: 'expert', layer: 2, expert: 300 });
  addItem(sel, { type: 'stream', layer: 1, stream: 3 });
  const p = buildPreset({ shape: SHAPE, checked: selectionToBody(sel) }, { now: AT });
  assert.deepEqual(p.checked, { experts_off: [[2, 300]], blocks_off: [], streams_off: [[1, 3]] });
});

test('validateInterveneBody: 範囲・block 名・top-k の下限を弾く', () => {
  const ok = validateInterveneBody({ experts_off: [[0, 1], [0, 1]] }, SHAPE);
  assert.deepEqual(ok.experts_off, [[0, 1]], '重複は落とす');
  assert.throws(() => validateInterveneBody({ experts_off: [[48, 1]] }, SHAPE), /layer out of range/);
  assert.throws(() => validateInterveneBody({ experts_off: [[0, 512]] }, SHAPE), /expert out of range/);
  assert.throws(() => validateInterveneBody({ blocks_off: [[0, 'nope']] }, SHAPE), /unknown block/);
  assert.throws(() => validateInterveneBody({ streams_off: [[0, 4]] }, SHAPE), /stream out of range/);
  assert.throws(() => validateInterveneBody({ experts_off: [[0, 'x']] }, SHAPE), /integers/);
  // 層 0 の 503 本を off = 残り 9 < top-10 → サーバーと同じく拒否
  const tooMany = [];
  for (let e = 0; e < 503; e++) tooMany.push([0, e]);
  assert.throws(() => validateInterveneBody({ experts_off: tooMany }, SHAPE), /remaining experts 9/);
  // 502 本なら残り 10 = ちょうど通る
  assert.equal(validateInterveneBody({ experts_off: tooMany.slice(0, 502) }, SHAPE).experts_off.length, 502);
});

test('parsePreset: 壊れた JSON / 版違い / 種類違い / 範囲外を拒否', () => {
  assert.throws(() => parsePreset(''), /empty/);
  assert.throws(() => parsePreset('{nope'), /not JSON/);
  assert.throws(() => parsePreset('[]'), /must be an object/);
  const p = buildPreset(sampleInput(), { now: AT });
  assert.throws(() => parsePreset(JSON.stringify(Object.assign({}, p, { schema: 2 }))), /unsupported schema/);
  assert.throws(() => parsePreset(JSON.stringify(Object.assign({}, p, { kind: 'other' }))), /kind must be/);
  const noShape = Object.assign({}, p);
  delete noShape.shape;
  assert.throws(() => parsePreset(JSON.stringify(noShape)), /shape is required/);
  assert.throws(() => parsePreset(JSON.stringify(Object.assign({}, p, {
    intervene: { experts_off: [[99, 1]] },
  }))), /layer out of range/);
  assert.throws(() => parsePreset(JSON.stringify(Object.assign({}, p, { selector: 'x'.repeat(5000) }))), /too long/);
});

test('steer は空 terms なら null、範囲外の layers は例外', () => {
  const p = buildPreset({ shape: SHAPE, steer: { terms: [] } }, { now: AT });
  assert.equal(p.steer, null);
  assert.throws(() => buildPreset({
    shape: SHAPE,
    steer: { terms: [{ snapshot: 'A', coef: 1 }], layers: [0, 49], alpha: 1, mode: 'subtract' },
  }), /to must be <= 48/);
  assert.throws(() => buildPreset({
    shape: SHAPE,
    steer: { terms: [{ snapshot: 'A', coef: 1 }], layers: [0, 4], alpha: 1, mode: 'nope' },
  }), /subtract or project/);
});

test('presetSummary と presetFileName', () => {
  const p = buildPreset(sampleInput(), { now: AT });
  const s = presetSummary(p);
  assert.match(s, /^checked 2 \(1e\/1b\/0s\)/);
  assert.match(s, /off 5 \(3e\/1b\/1s\)/);
  assert.match(s, /selector "expert: count == 0 and layer >= 40"/);
  assert.match(s, /steer 2 terms 4-20 project α 0.75$/);
  assert.equal(presetFileName('dbg_A'), 'denno-preset-dbg_A.json');
  assert.equal(presetFileName('a b/c:d'), 'denno-preset-a_b_c_d.json');
  assert.equal(presetFileName(''), 'denno-preset-preset.json');
  assert.equal(presetFileName('///'), 'denno-preset-preset.json');
});
