// denno-view /2d の HUD と多ベクトル ステアリング (hud2d.mjs / view2d.mjs) の
// 境界値テスト (task 1124)。実行: node --test crates/denno-view/tests/*.test.mjs
//
// 既存の hud2d.test.mjs / ablit2d.test.mjs が見ていない範囲だけを足す:
//   1. 0 トークン / 1 トークン / NaN / Infinity / 時計が逆戻りしたときの
//      HUD の数値と書式 (画面に NaN や Infinity を出さない)
//   2. 多ベクトル ステアリング規則 (rules[]) の検証と直列化の境界
//      — 空の名前、極端な alpha / coef、同じ層範囲の重複、範囲外の層
//   3. 壊れた状態更新を食わせても例外を投げないこと / 同じ入力列なら同じ表示
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  createHud, fmtDuration, fmtPercent, fmtRate, fmtTokens,
  hudBeginRequest, hudEndRequest, hudFields, hudOnCache, hudOnFrame,
  hudOnRawKind, hudOnStats, hudOnToken, hudText, specSample,
} from '../src/hud2d.mjs';
import {
  ablitStackBody, ablitStackSummary, createMockStore, steerApplyBody,
} from '../src/view2d.mjs';

// ---- 1. 書式: NaN / Infinity / 負 を画面に出さない -------------------------

test('fmt*: NaN / Infinity / 負の値は "-" (または 0 に丸める)', () => {
  for (const v of [NaN, Infinity, -Infinity, null, undefined, '', 'x', {}]) {
    assert.equal(fmtDuration(v), '-', 'duration ' + String(v));
    assert.equal(fmtRate(v), '-', 'rate ' + String(v));
    assert.equal(fmtPercent(v), '-', 'percent ' + String(v));
    assert.equal(fmtTokens(v), '-', 'tokens ' + String(v));
  }
  // 負の duration / rate は不正値 (負の百分率だけは 0% に寄せる)。
  assert.equal(fmtDuration(-1), '-');
  assert.equal(fmtDuration(-0.4), '-');
  assert.equal(fmtRate(-0.001), '-');
  assert.equal(fmtPercent(-1), '0%');
  assert.equal(fmtPercent(2), '100%', '1 を超える割合は 100% で頭打ち');
  // 0 は "-" ではなく 0 を出す。
  assert.equal(fmtDuration(0), '0 ms');
  assert.equal(fmtRate(0), '0.0');
  assert.equal(fmtPercent(0), '0%');
  assert.equal(fmtTokens(0), '0');
  // 境界: 999.6 は四捨五入で 1000 ms (まだ ms 表記)、1000 から s 表記。
  assert.equal(fmtDuration(999.6), '1000 ms');
  assert.equal(fmtDuration(1000), '1.00 s');
  assert.equal(fmtDuration('412.7'), '413 ms', '数値文字列は受ける');
  // fmtTokens は切り捨てて桁区切り。負はそのまま (状態が壊れている印)。
  assert.equal(fmtTokens('3826.9'), '3,826');
  assert.equal(fmtTokens(-5), '-5');
  assert.equal(fmtPercent('0.5'), '50%');
});

test('hudFields / hudText: 壊れた hud / mtp でも "-" 行を返す', () => {
  for (const bad of [null, undefined, {}, { mtp: null }, { mtp: {} }, []]) {
    const rows = hudFields(bad);
    assert.equal(rows.length, 5);
    assert.deepEqual(rows.map((f) => f.value), ['-', '-', '-', '-', '-']);
    assert.equal(hudText(bad), 'ttft - · tok/s - · vram hit - · kv reuse - · mtp accept -');
  }
});

// ---- 1b. 0 トークン / 1 トークンのリクエスト -------------------------------

test('0 トークン: TTFT も tok/s も出さず、usage 0 ではトークン数を変えない', () => {
  const h = createHud();
  hudBeginRequest(h, 100);
  assert.equal(hudEndRequest(h, 400, 0), true);
  assert.equal(h.reqTokens, 0);
  assert.equal(h.ttftMs, null);
  assert.equal(h.tokPerSec, null);
  assert.ok(Math.abs(h.lastSecs - 0.3) < 1e-9);
  assert.deepEqual(
    hudFields(h).map((f) => f.value).slice(0, 2),
    ['-', '-'],
  );

  // usage を渡さない終了も同じ (tokens 0 のまま)。
  const h2 = createHud();
  hudBeginRequest(h2, 0);
  hudEndRequest(h2, 250);
  assert.equal(h2.reqTokens, 0);
  assert.equal(h2.tokPerSec, null);
  assert.ok(Math.abs(h2.lastSecs - 0.25) < 1e-9);
});

test('1 トークン: TTFT だけ確定し、usage 1 でも tok/s は出さない', () => {
  const h = createHud();
  hudBeginRequest(h, 1000);
  hudOnToken(h, 1250);
  assert.equal(h.ttftMs, 250);
  assert.equal(hudFields(h)[0].value, '250 ms');
  assert.equal(h.tokPerSec, null, '区間が 0 なので速度を出さない');
  hudEndRequest(h, 1300, 1);
  assert.equal(h.reqTokens, 1);
  assert.equal(h.tokPerSec, null);
  assert.equal(hudFields(h)[1].value, '-');
});

test('usage 文字列は切り捨て、n が 0 / 負でも 1 トークンに丸める', () => {
  const h = createHud();
  hudBeginRequest(h, 0);
  hudOnToken(h, 100, 0);
  hudOnToken(h, 200, -3);
  assert.equal(h.reqTokens, 2, '0 と負は 1 に丸める');
  assert.ok(Math.abs(h.tokPerSec - 10) < 1e-9);
  hudEndRequest(h, 300, '7.9');
  assert.equal(h.reqTokens, 7);
  assert.ok(Math.abs(h.tokPerSec - 60) < 1e-9, 'usage 7 で数え直す (6 / 0.1 s)');
});

// ---- 1c. NaN / Infinity の時計 --------------------------------------------

test('reqStart が NaN: TTFT は NaN のまま "-" 表示、例外は投げない', () => {
  const h = createHud();
  hudBeginRequest(h, NaN);
  assert.equal(hudOnToken(h, 100), true);
  assert.ok(Number.isNaN(h.ttftMs));
  assert.equal(hudFields(h)[0].value, '-', 'NaN は画面に出さない');
  // 以降のトークンは正常 (区間は 100 → 200 で測れる)。
  hudOnToken(h, 200);
  assert.ok(Math.abs(h.tokPerSec - 10) < 1e-9);
  // 終了時刻が読めても reqStart が NaN なら経過秒も NaN。表示経路は無いが、
  // 書式に通しても "-" に落ちる (画面に NaN は出ない)。
  hudEndRequest(h, 300, NaN);
  assert.ok(Number.isNaN(h.lastSecs));
  assert.equal(fmtDuration(h.lastSecs), '-');
  assert.equal(hudFields(h)[1].value, '10.0');
  // 終了時刻そのものが NaN なら経過秒は null (usage Infinity も無い扱い)。
  const h2 = createHud();
  hudBeginRequest(h2, 1000);
  hudEndRequest(h2, NaN, Infinity);
  assert.equal(h2.lastSecs, null);
});

test('Infinity の時刻はトークンとして捨て、n=Infinity でも描画は "-"', () => {
  const h = createHud();
  hudBeginRequest(h, 0);
  assert.equal(hudOnToken(h, Infinity), false);
  assert.equal(h.ttftMs, null);
  assert.equal(h.reqTokens, 0);
  assert.equal(hudOnToken(h, -Infinity), false);
  assert.equal(hudOnToken(h, NaN), false);

  // n=Infinity はトークン数が Infinity になるが、速度表示は "-" に落ちる。
  assert.equal(hudOnToken(h, 100, Infinity), true);
  assert.equal(h.reqTokens, Infinity);
  assert.equal(hudOnToken(h, 200), true);
  assert.equal(h.tokPerSec, Infinity);
  assert.equal(hudFields(h)[1].value, '-');
  // 終了しても壊れた値は増えない。
  assert.equal(hudEndRequest(h, 300, Infinity), true);
  assert.equal(hudFields(h)[1].value, '-', 'usage Infinity は無い扱い');
});

// ---- 1d. 時計が逆戻りする ------------------------------------------------

test('時計の逆戻り: TTFT / 経過秒は 0 で床止め、速度は出さない', () => {
  const h = createHud();
  hudBeginRequest(h, 1000);
  hudOnToken(h, 900);
  assert.equal(h.ttftMs, 0, '負の TTFT は 0');
  assert.equal(hudFields(h)[0].value, '0 ms');
  hudOnToken(h, 800);                        // 最初のトークンより前
  assert.equal(h.reqTokens, 2);
  assert.equal(h.tokPerSec, null, '逆順の区間では速度を出さない');
  hudEndRequest(h, 500, 3);
  assert.equal(h.lastSecs, 0, '負の経過秒は 0');
  assert.equal(h.tokPerSec, null, '終了時の再計算でも逆順は無視');
  assert.equal(hudFields(h)[1].value, '-');
});

// ---- 1e. 投機 (MTP) の NaN / Infinity ------------------------------------

test('specSample: NaN / Infinity は「読めない」= null に落とす', () => {
  assert.deepEqual(specSample({ accepted: NaN, proposed: 10 }), { accepted: null, proposed: 10, rate: null });
  assert.equal(specSample({ accepted: Infinity, proposed: Infinity }), null);
  assert.equal(specSample({ rate: NaN }), null);
  assert.equal(specSample({ rate: Infinity }), null);
  // 割合の範囲外は clamp (500 は 100%、負は 0)。
  assert.deepEqual(specSample({ rate: 500 }), { accepted: null, proposed: null, rate: 1 });
  assert.deepEqual(specSample({ rate: -3 }), { accepted: null, proposed: null, rate: 0 });
  assert.deepEqual(specSample({ accepted: 3, proposed: 0 }), { accepted: 3, proposed: 0, rate: null });
});

test('hudOnFrame: accepted が NaN なら 0 として累積に載る (表示は 0%)', () => {
  const h = createHud();
  assert.equal(hudOnFrame(h, { kind: 18, accepted: NaN, proposed: 10 }), true);
  assert.equal(h.mtp.seen, true);
  assert.equal(h.mtp.accepted, 0);
  assert.equal(h.mtp.proposed, 10);
  assert.equal(h.mtp.rate, 0);
  assert.equal(hudFields(h)[4].value, '0% (0/10)');
  // proposed だけ Infinity のフレームは読める数字が無い → telemetry 表示のみ。
  const h2 = createHud();
  assert.equal(hudOnFrame(h2, { kind: 'kind18', proposed: Infinity }), true);
  assert.equal(h2.mtp.last, null);
  assert.equal(h2.mtp.rate, null);
  assert.equal(hudFields(h2)[4].value, 'telemetry');
});

test('hudOnRawKind: 整数でない / 負の kind は投機として数えない', () => {
  const h = createHud();
  assert.equal(hudOnRawKind(h, 18.5), false);
  assert.equal(hudOnRawKind(h, NaN), false);
  assert.equal(hudOnRawKind(h, -18), false);
  assert.equal(hudOnRawKind(h, 'kind-1'), false);
  assert.equal(hudOnRawKind(h, 'kind18'), true);
  assert.equal(h.mtp.frames, 1);
});

// ---- 3a. 壊れた状態更新で例外を投げない ----------------------------------

test('壊れた状態更新: null / undefined / 配列 / 数値 / 文字列で落ちない', () => {
  const junk = [null, undefined, 0, 1, 'x', '', [], [null], {}, true];
  const sinks = [hudOnToken, hudOnStats, hudOnCache, hudOnFrame, hudOnRawKind, hudEndRequest];
  for (const fn of sinks) {
    for (const j of junk) {
      assert.doesNotThrow(() => fn(createHud(), j), fn.name + ' / ' + String(j));
    }
  }
  // 更新系は false を返して「何もしない」ことを示す (reqActive でない / 読めない)。
  assert.equal(hudOnToken(createHud(), 1), false);
  assert.equal(hudOnFrame(createHud(), []), false);
  assert.equal(hudOnStats(createHud(), 'x'), false);
  assert.equal(hudOnCache(createHud(), 'x'), false);
  assert.equal(hudOnRawKind(createHud(), {}), false);
  assert.equal(hudEndRequest(createHud(), 1), false);
  // stats / cache の中身が欠けていても null に落ちる。
  const h = createHud();
  assert.equal(hudOnStats(h, { rows: [{}] }), false, '既に null なので変化なし');
  assert.equal(hudOnStats(h, { rows: [{ hit: 1.7 }] }), true);
  assert.equal(h.vramHit, 1);
  assert.equal(hudOnStats(h, { rows: [{ hit: -0.5 }] }), true);
  assert.equal(h.vramHit, 0);
  assert.equal(hudOnCache(h, { main: { tokens: NaN, epoch: NaN } }), false);
  assert.equal(hudOnCache(h, { main: 5 }), false);
});

// ---- 3b. 決定性 -----------------------------------------------------------

test('同じイベント列 → 同じ HUD 状態と表示 (読み出しは状態を変えない)', () => {
  const run = () => {
    const h = createHud();
    hudBeginRequest(h, 100);
    hudOnToken(h, 160);
    hudOnToken(h, 260, 3);
    hudOnStats(h, { rows: [{ hit: 0.87 }] });
    hudOnCache(h, { main: { tokens: 1200, epoch: 2 } });
    hudOnFrame(h, { kind: 'MtpStats', accepted: 5, proposed: 8 });
    hudOnFrame(h, { kind: 'MtpStats', accepted: 1, proposed: 2 });
    hudEndRequest(h, 900, 9);
    return h;
  };
  const a = run();
  const b = run();
  assert.deepEqual(a, b);
  assert.equal(hudText(a), hudText(b));
  const first = hudText(a);
  for (let i = 0; i < 3; i++) assert.equal(hudText(a), first);
  assert.deepEqual(hudFields(a), hudFields(a));
  // 同じ stats をもう一度食わせても変化なし (冪等)。
  const snap = JSON.parse(JSON.stringify(a));
  assert.equal(hudOnStats(a, { rows: [{ hit: 0.87 }] }), false);
  assert.deepEqual(a, snap);
  // 入力オブジェクトは書き換えない。
  const frame = Object.freeze({ kind: 18, accepted: 2, proposed: 4 });
  const h = createHud();
  assert.doesNotThrow(() => hudOnFrame(h, frame));
  assert.equal(h.mtp.accepted, 2);
});

// ---- 2. 多ベクトル ステアリング規則 --------------------------------------

const BASE = { pos: 'ablit_A', neg: 'ablit_B', layers: [0, 4] };

test('ablitStackBody: 空 / 不正な snapshot 名を弾く', () => {
  assert.throws(() => ablitStackBody([{ ...BASE, pos: '' }], {}), /pos: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, pos: '   ' }], {}), /pos: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, pos: 'a b' }], {}), /pos: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, pos: 'A'.repeat(65) }], {}), /pos: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, pos: 5 }], {}), /pos: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, neg: '' }], {}), /neg: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, neg: 'A B' }], {}), /neg: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, neg: null }], {}), /neg: snapshot name/);
  assert.throws(() => ablitStackBody([{ ...BASE, pos: 'same', neg: 'same' }], {}), /different/);
  // 64 文字ちょうどは通る。
  assert.equal(ablitStackBody([{ ...BASE, pos: 'A'.repeat(64) }], {}).rules[0].terms[0].snapshot.length, 64);
  // 空の配列 / 全部 disable は「有効な規則が無い」。
  assert.throws(() => ablitStackBody([], {}), /at least one/);
  assert.throws(() => ablitStackBody(null, {}), /at least one/);
});

test('ablitStackBody: 極端な alpha を弾き、数値文字列は受ける', () => {
  for (const alpha of [2.0000001, 1e308, -1e308, NaN, Infinity, -Infinity]) {
    assert.throws(() => ablitStackBody([{ ...BASE, alpha }], {}), /alpha/, 'alpha ' + String(alpha));
  }
  assert.throws(() => ablitStackBody([{ ...BASE, alpha: 'x' }], {}), /alpha/);
  // 境界の 0 と 2 は通り、文字列 '1.5' は数値に直す。
  assert.equal(ablitStackBody([{ ...BASE, alpha: 0 }], {}).rules[0].alpha, 0);
  assert.equal(ablitStackBody([{ ...BASE, alpha: 2 }], {}).rules[0].alpha, 2);
  assert.equal(ablitStackBody([{ ...BASE, alpha: '1.5' }], {}).rules[0].alpha, 1.5);
  assert.equal(ablitStackBody([{ ...BASE, alpha: null }], {}).rules[0].alpha, 1, 'null は既定 1');
});

test('ablitStackBody: 層範囲の境界と範囲外', () => {
  assert.deepEqual(ablitStackBody([{ ...BASE, layers: [47, 48] }], { nLayers: 48 }).rules[0].layers, [47, 48]);
  assert.deepEqual(ablitStackBody([{ ...BASE, layers: ['3', '6'] }], { nLayers: 48 }).rules[0].layers, [3, 6]);
  assert.deepEqual(ablitStackBody([{ ...BASE, layers: [0.9, 2.9] }], { nLayers: 48 }).rules[0].layers, [0, 2],
    '小数は切り捨て (intOr)');
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [0, 49] }], { nLayers: 48 }), /to must be <=/);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [48, 49] }], { nLayers: 48 }), /to must be <=/);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [-1, 3] }], {}), /from\/to must be >= 0/);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [3, 2] }], {}), /from must be </);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [2, 2] }], {}), /from must be </);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [0] }], {}), /\[from, to\]/);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: 'x' }], {}), /\[from, to\]/);
  assert.throws(() => ablitStackBody([{ ...BASE, layers: [NaN, 2] }], {}), /from\/to must be >= 0/);
  // nLayers 未指定 (0) のときは上限を見ない (画面側の見積りだけ)。
  assert.deepEqual(ablitStackBody([{ ...BASE, layers: [0, 999] }], {}).rules[0].layers, [0, 999]);
});

test('ablitStackBody: 同じ層範囲の重複はそのまま残し、順序と本文が決定的', () => {
  const stack = [
    { pos: 'ablit_A', neg: 'ablit_B', layers: [4, 8], alpha: 1, mode: 'project' },
    { pos: 'ablit_C', neg: 'ablit_D', layers: [4, 8], alpha: 0.5, mode: 'subtract' },
  ];
  const b1 = ablitStackBody(stack, { nLayers: 48 });
  const b2 = ablitStackBody(stack, { nLayers: 48 });
  assert.deepEqual(b1, b2);
  assert.notEqual(b1, b2, '呼ぶたびに新しい本文');
  assert.equal(b1.rules.length, 2, '重複した層範囲でも 1 本ずつ残す');
  assert.deepEqual(b1.rules.map((r) => r.layers), [[4, 8], [4, 8]]);
  assert.deepEqual(b1.rules.map((r) => r.mode), ['project', 'subtract']);
  assert.deepEqual(b1.rules.map((r) => r.alpha), [1, 0.5]);
  // 直列化 → 読み直しても同じ (JSON の往復で壊れない)。
  assert.deepEqual(JSON.parse(JSON.stringify(b1)), b1);
  // 余計なフィールド (on / note) は本文に出さない。
  const withExtra = ablitStackBody([{ ...BASE, on: true, note: 'x', bogus: 1 }], {});
  assert.deepEqual(Object.keys(withExtra.rules[0]).sort(), ['alpha', 'layers', 'mode', 'terms']);
  // 入力は凍結しても触らない (純粋)。
  const frozen = Object.freeze([
    Object.freeze({ ...BASE, layers: Object.freeze([0, 4]) }),
  ]);
  assert.doesNotThrow(() => ablitStackBody(frozen, { nLayers: 48 }));
  // 要約は規則ごとに層範囲を出す (重複も 2 回)。
  const s = ablitStackSummary(b1);
  assert.equal(s,
    'ablit_A +1 / ablit_B -1 · project · α 1 · layers 4-8 + ' +
    'ablit_C +1 / ablit_D -1 · subtract · α 0.5 · layers 4-8');
});

test('モック: 同じ層範囲の 2 規則を受け、範囲外は 400', () => {
  const m = createMockStore({ layers: 48, n_expert: 8, n_expert_used: 2, n_hc: 1, n_embd: 16 },
    { now: () => 1762732800 });
  for (const name of ['ablit_A', 'ablit_B', 'ablit_C', 'ablit_D']) {
    m.handle('POST', '/v1/steer/record', { name });
    m.tick(1);
    m.handle('POST', '/v1/steer/record/stop');
  }
  const dup = ablitStackBody([
    { pos: 'ablit_A', neg: 'ablit_B', layers: [4, 8] },
    { pos: 'ablit_C', neg: 'ablit_D', layers: [4, 8], alpha: 0.5, mode: 'subtract' },
  ], { nLayers: 48 });
  const r = m.handle('POST', '/v1/steer/apply', dup);
  assert.equal(r.status, 200);
  const st = m.handle('GET', '/v1/steer').body;
  assert.equal(st.apply.rules.length, 2);
  assert.deepEqual(st.apply.rules.map((x) => x.layers), [[4, 8], [4, 8]]);
  assert.deepEqual(st.apply.layers, [4, 8], '包みは全規則の外側 (同じなら同じ)');

  // 範囲外の層は実 API と同じく 400。
  assert.equal(m.handle('POST', '/v1/steer/apply', {
    rules: [{ terms: [{ snapshot: 'ablit_A', coef: 1 }], layers: [0, 49], alpha: 1, mode: 'project' }],
  }).status, 400);
  // 空の名前の規則は 400。
  assert.equal(m.handle('POST', '/v1/steer/apply', {
    rules: [{ terms: [{ snapshot: '', coef: 1 }], layers: [0, 1], alpha: 1, mode: 'project' }],
  }).status, 400);
});

// ---- 2b. 単一ベクトル本文の極端な coef / alpha ---------------------------

test('steerApplyBody: 有限なら極端な coef / alpha もそのまま通す', () => {
  const b = steerApplyBody({
    terms: [{ snapshot: 'A', coef: 1e308 }, { snapshot: 'B', coef: -1e-300 }],
    layers: [0, 1], alpha: 1e308, mode: 'subtract',
  });
  assert.equal(b.terms[0].coef, 1e308);
  assert.equal(b.terms[1].coef, -1e-300);
  assert.equal(b.alpha, 1e308);
  // 文字列 coef は数値に直し、null は 0。
  assert.equal(steerApplyBody({ terms: [{ snapshot: 'A', coef: '3' }] }).terms[0].coef, 3);
  assert.equal(steerApplyBody({ terms: [{ snapshot: 'A', coef: null }] }).terms[0].coef, 0);
  // 非有限は拒否。
  for (const coef of [NaN, Infinity, -Infinity]) {
    assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef }] }), /coef must be a finite/);
  }
  for (const alpha of [NaN, Infinity, -Infinity]) {
    assert.throws(() => steerApplyBody({ terms: [{ snapshot: 'A', coef: 1 }], alpha }), /alpha: must be a finite/);
  }
  // 空の snapshot / terms 無し / 壊れた body は例外 (黙って通さない)。
  assert.throws(() => steerApplyBody({ terms: [{ snapshot: '', coef: 1 }] }), /snapshot is required/);
  assert.throws(() => steerApplyBody({ terms: [] }), /at least one/);
  assert.throws(() => steerApplyBody(null), /terms/);
  assert.throws(() => steerApplyBody({ terms: 'x' }), /terms/);
  assert.throws(() => steerApplyBody({ terms: [null] }), /snapshot is required/);
});
