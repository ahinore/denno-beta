// denno-view /chat (task 1194) の履歴正規化と操作値の優先順位のテスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
//
// 見ているもの:
//   1. 複数ターンの履歴 (user/assistant の並び) を順序どおり落とさない
//   2. normaliseHistory が壊れた role / content だけを捨てる
//   3. プリセット本文と手書き system の優先順位 (custom が空でなければ custom)
//   4. buildChatBody が空の system を送らず、履歴の掃除を通す
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CHAT_DEFAULTS, SYSTEM_PRESETS,
  buildChatBody, normaliseControls, normaliseHistory, systemPreset,
} from '../src/chat.mjs';

// ---- 1. 複数ターンの履歴 --------------------------------------------------

test('normaliseHistory: user/assistant の複数ターンを順序どおり保つ', () => {
  const history = [
    { role: 'user', content: 'q1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'q2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'q3' },
  ];
  assert.deepEqual(normaliseHistory(history), [
    { role: 'user', content: 'q1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'q2' },
    { role: 'assistant', content: 'a2' },
    { role: 'user', content: 'q3' },
  ]);
});

test('normaliseHistory: 元の配列と要素を書き換えない', () => {
  const history = [
    { role: 'user', content: 'hi', extra: 1 },
    { role: 'system', content: 'drop me' },
  ];
  const snapshot = JSON.parse(JSON.stringify(history));
  const out = normaliseHistory(history);
  // 返る要素は role と content だけを持つ新しいオブジェクト
  assert.deepEqual(out, [{ role: 'user', content: 'hi' }]);
  assert.notEqual(out[0], history[0]);
  assert.deepEqual(history, snapshot, '入力を破壊した');
});

test('normaliseHistory: 隣接する同じ role を勝手に統合しない', () => {
  const out = normaliseHistory([
    { role: 'user', content: 'a' },
    { role: 'user', content: 'b' },
    { role: 'assistant', content: 'c' },
  ]);
  assert.equal(out.length, 3);
  assert.deepEqual(out.map((m) => m.content), ['a', 'b', 'c']);
});

test('normaliseHistory: system ロールは履歴から落ちる (本文は別枠)', () => {
  const out = normaliseHistory([
    { role: 'system', content: 'you are nice' },
    { role: 'user', content: 'hi' },
  ]);
  assert.deepEqual(out, [{ role: 'user', content: 'hi' }]);
});

// ---- 2. 壊れた role / content --------------------------------------------

test('normaliseHistory: 未知の role (大小文字違い・tool・空) は捨てる', () => {
  const bad = [
    { role: 'User', content: 'x' },
    { role: 'USER', content: 'x' },
    { role: 'Assistant', content: 'x' },
    { role: 'tool', content: 'x' },
    { role: '', content: 'x' },
    { role: ' system ', content: 'x' },
    { content: 'x' },
    { role: null, content: 'x' },
    { role: 1, content: 'x' },
  ];
  const out = normaliseHistory([...bad, { role: 'user', content: 'keep' }]);
  assert.deepEqual(out, [{ role: 'user', content: 'keep' }]);
});

test('normaliseHistory: 文字列でない content は捨てる', () => {
  const bad = [
    { role: 'user' },
    { role: 'user', content: null },
    { role: 'user', content: undefined },
    { role: 'user', content: 0 },
    { role: 'user', content: 42 },
    { role: 'user', content: true },
    { role: 'user', content: ['a', 'b'] },
    { role: 'user', content: { text: 'obj' } },
    { role: 'assistant', content: ['x'] },
  ];
  assert.deepEqual(normaliseHistory(bad), []);
  assert.deepEqual(
    normaliseHistory([...bad, { role: 'assistant', content: 'ok' }]),
    [{ role: 'assistant', content: 'ok' }],
  );
});

test('normaliseHistory: 空文字の content だけを落とし、空白だけは残す', () => {
  const out = normaliseHistory([
    { role: 'user', content: '' },
    { role: 'user', content: ' ' },
    { role: 'assistant', content: '\n' },
    { role: 'user', content: '   spaced   ' },
  ]);
  assert.deepEqual(out, [
    { role: 'user', content: ' ' },
    { role: 'assistant', content: '\n' },
    { role: 'user', content: '   spaced   ' },
  ]);
});

test('normaliseHistory: null / 非配列の履歴は空配列', () => {
  assert.deepEqual(normaliseHistory(null), []);
  assert.deepEqual(normaliseHistory(undefined), []);
  assert.deepEqual(normaliseHistory('nope'), []);
  assert.deepEqual(normaliseHistory(42), []);
  assert.deepEqual(normaliseHistory({ length: 1 }), []);
  assert.deepEqual(normaliseHistory([null, undefined, 0, 'x', []]), []);
});

// ---- 3. プリセット vs 手書き system の優先順位 ----------------------------

test('normaliseControls: 手書きが空でなければプリセットより優先', () => {
  const preset = systemPreset('code');
  assert.equal(preset.text.length > 0, true, '前提: code は本文を持つ');

  const a = normaliseControls({ presetId: 'code' });
  assert.equal(a.systemText, preset.text, 'preset も custom も無ければ preset');

  const b = normaliseControls({ presetId: 'code', system: 'my own prompt' });
  assert.equal(b.systemText, 'my own prompt', 'custom が優先されていない');

  const c = normaliseControls({ presetId: 'code', system: '  padded  ' });
  assert.equal(c.systemText, 'padded', 'custom を trim していない');
});

test('normaliseControls: 空白だけの custom はプリセットに戻る', () => {
  const preset = systemPreset('tutor');
  for (const blank of ['', '   ', '\t', '\n', ' \n ']) {
    const r = normaliseControls({ presetId: 'tutor', system: blank });
    assert.equal(r.systemText, preset.text, `custom=${JSON.stringify(blank)} で preset に戻らない`);
  }
});

test('normaliseControls: 文字列でない custom は無視してプリセットを使う', () => {
  const preset = systemPreset('ja');
  for (const bad of [null, undefined, 0, 42, true, ['x'], { text: 'x' }]) {
    const r = normaliseControls({ presetId: 'ja', system: bad });
    assert.equal(r.systemText, preset.text, `custom=${String(bad)} が混入した`);
  }
});

test('normaliseControls: 未知の preset は custom が無ければ空 (plain 相当)', () => {
  const none = normaliseControls({ presetId: 'does-not-exist' });
  assert.equal(none.systemText, '');
  assert.equal(none.temperature, CHAT_DEFAULTS.temperature);
  assert.equal(none.maxTokens, CHAT_DEFAULTS.maxTokens);

  const withCustom = normaliseControls({ presetId: 'does-not-exist', system: 'explicit' });
  assert.equal(withCustom.systemText, 'explicit', '未知 preset でも custom は使う');

  // plain は本文が空なので custom 無しでは空
  const plain = normaliseControls({ presetId: 'plain' });
  assert.equal(plain.systemText, '');
});

test('normaliseControls: 全プリセットで custom が常に勝つ', () => {
  for (const p of SYSTEM_PRESETS) {
    const custom = `custom for ${p.id}`;
    const r = normaliseControls({ presetId: p.id, system: custom });
    assert.equal(r.systemText, custom, `${p.id} で custom が負けた`);
  }
});

// ---- 4. buildChatBody との結線 -------------------------------------------

test('buildChatBody: 複数ターン履歴をそのまま本文へ写す', () => {
  const messages = [
    { role: 'user', content: 'q1' },
    { role: 'assistant', content: 'a1' },
    { role: 'user', content: 'q2' },
  ];
  const body = buildChatBody({ messages, maxTokens: 64, temperature: 0.5 });
  assert.deepEqual(body.messages, messages);
  assert.equal(body.max_tokens, 64);
  assert.equal(body.temperature, 0.5);
  assert.equal(body.stream, true);
  assert.equal('system' in body, false, '空の system を送っている');
});

test('buildChatBody: 壊れた履歴要素は落として正しい形で送る', () => {
  const body = buildChatBody({
    messages: [
      { role: 'user', content: 'ok' },
      { role: 'system', content: 'drop' },
      { role: 'assistant', content: 5 },
      null,
      { role: 'assistant', content: 'kept' },
    ],
  });
  assert.deepEqual(body.messages, [
    { role: 'user', content: 'ok' },
    { role: 'assistant', content: 'kept' },
  ]);
});

test('buildChatBody: normaliseControls の結果を system として送る', () => {
  // preset だけ → 空 (plain) なら system を送らない
  const plain = normaliseControls({ presetId: 'plain' });
  const b1 = buildChatBody({ system: plain.systemText, messages: [{ role: 'user', content: 'x' }] });
  assert.equal('system' in b1, false, 'plain で system を送っている');

  // 本文のある preset → その本文を送る
  const concise = normaliseControls({ presetId: 'concise' });
  const b2 = buildChatBody({ system: concise.systemText, messages: [{ role: 'user', content: 'x' }] });
  assert.equal(b2.system, systemPreset('concise').text);

  // custom が preset を上書きした結果を送る
  const custom = normaliseControls({ presetId: 'concise', system: '  override me  ' });
  const b3 = buildChatBody({ system: custom.systemText, messages: [{ role: 'user', content: 'x' }] });
  assert.equal(b3.system, 'override me');

  // 空白だけの system は trim で空になり送られない
  const b4 = buildChatBody({ system: '   ', messages: [] });
  assert.equal('system' in b4, false);
  assert.deepEqual(b4.messages, []);
});
