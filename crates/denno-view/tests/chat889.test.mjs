// denno-view /chat (task 889) の純ロジックのテスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
//
// 見ているもの:
//   1. システムプロンプトのプリセット (重複なし・空の plain・引き当て)
//   2. temperature / max_tokens のクランプ (サーバーの上限 4096 / 2.0)
//   3. POST /chat の本文 (空の system は送らない・履歴の掃除・continue)
//   4. TTFT / 出力 tok/s の状態機械 (usage で数え直す)
//   5. SSE の増分パーサ (チャンク境界・CRLF・複数 data 行・flush)
//   6. event → 本文/usage/停止理由/エラー (Anthropic と OpenAI の差分配列)
//   7. 画面 (chat.html) と /2d のリンクの結線
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import {
  CHAT_DEFAULTS, CHAT_SCHEMA, CHAT_URL, MAX_TOKENS_CLAMP, SPLIT_2D_URL,
  SYSTEM_PRESETS, TEMPERATURE_MAX,
  applyChatEvent, buildChatBody, chatBegin, chatEnd, chatFields, chatOnToken,
  chatRate, clampMaxTokens, clampTemperature, createChatState, createSseParser,
  fmtMs, fmtRate, normaliseControls, normaliseHistory, parseSseBlock, splitViewUrl,
  systemPreset,
} from '../src/chat.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(HERE, '../src/chat.html'), 'utf8');
const index2d = readFileSync(join(HERE, '../src/index2d.html'), 'utf8');

// ---- 1. プリセット --------------------------------------------------------

test('SYSTEM_PRESETS: id は一意で plain は本文が空', () => {
  assert.equal(CHAT_SCHEMA, 1);
  const ids = SYSTEM_PRESETS.map((p) => p.id);
  assert.equal(new Set(ids).size, ids.length, 'id が重複: ' + ids.join(','));
  assert.ok(ids.includes('plain'));
  assert.equal(systemPreset('plain').text, '');
  for (const p of SYSTEM_PRESETS) {
    assert.equal(typeof p.label, 'string');
    assert.ok(p.label.length > 0, p.id + ' の label が空');
    assert.equal(typeof p.text, 'string');
  }
  // 未知の id と null は null (画面は plain のまま)
  assert.equal(systemPreset('nope'), null);
  assert.equal(systemPreset(null), null);
  assert.equal(systemPreset(undefined), null);
  // 本文のあるプリセットが 1 つ以上ある (選択が意味を持つ)
  assert.ok(SYSTEM_PRESETS.some((p) => p.text.length > 0));
});

// ---- 2. クランプ ----------------------------------------------------------

test('clampTemperature: 0..2 に丸め、非数は既定', () => {
  assert.equal(clampTemperature(0.3), 0.3);
  assert.equal(clampTemperature(-1), 0);
  assert.equal(clampTemperature(9), TEMPERATURE_MAX);
  assert.equal(clampTemperature('0.5'), 0.5);
  assert.equal(clampTemperature('abc'), CHAT_DEFAULTS.temperature);
  assert.equal(clampTemperature(NaN), CHAT_DEFAULTS.temperature);
  // Infinity は数として扱わない (入力欄が壊れたときの既定)
  assert.equal(clampTemperature(Infinity), CHAT_DEFAULTS.temperature);
});

test('clampMaxTokens: [0, limit] に丸め、負や非数は既定', () => {
  assert.equal(clampMaxTokens(128), 128);
  assert.equal(clampMaxTokens(0), 0, '0 = prefill のみは有効');
  assert.equal(clampMaxTokens(-5), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(99999), MAX_TOKENS_CLAMP);
  assert.equal(clampMaxTokens(99999, 128, 1024), 1024);
  assert.equal(clampMaxTokens('64'), 64);
  assert.equal(clampMaxTokens('x'), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(10.9), 10);
});

test('normaliseControls: 手書きの system がプリセットより優先', () => {
  const preset = systemPreset('concise');
  const a = normaliseControls({ presetId: 'concise', temperature: 1, maxTokens: 32 });
  assert.equal(a.systemText, preset.text);
  assert.equal(a.temperature, 1);
  assert.equal(a.maxTokens, 32);

  const b = normaliseControls({ presetId: 'concise', system: '  custom text  ', temperature: 0.1, maxTokens: 8 });
  assert.equal(b.systemText, 'custom text');

  // 空白だけの手書きはプリセットへ戻る
  const c = normaliseControls({ presetId: 'code', system: '   ' });
  assert.equal(c.systemText, systemPreset('code').text);

  // 未知の preset は plain (system 無し)
  const d = normaliseControls({ presetId: 'zzz' });
  assert.equal(d.systemText, '');
  assert.equal(d.temperature, CHAT_DEFAULTS.temperature);
});

// ---- 3. リクエスト本文 ----------------------------------------------------

test('normaliseHistory: 空の content と未知の role を落とす', () => {
  const h = normaliseHistory([
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: '' },
    { role: 'system', content: 'ignored' },
    null,
    { role: 'assistant', content: 'yo' },
  ]);
  assert.deepEqual(h, [
    { role: 'user', content: 'hi' },
    { role: 'assistant', content: 'yo' },
  ]);
  assert.deepEqual(normaliseHistory(null), []);
  assert.deepEqual(normaliseHistory('nope'), []);
});

test('buildChatBody: 空の system は送らず、指定は素通しする', () => {
  const base = buildChatBody({
    messages: [{ role: 'user', content: 'hi' }],
    temperature: 0.2,
    maxTokens: 40,
  });
  assert.equal(base.model, 'denno');
  assert.equal(base.max_tokens, 40);
  assert.equal(base.temperature, 0.2);
  assert.equal(base.stream, true);
  assert.equal('system' in base, false, '空の system を送っている');
  assert.equal('continue' in base, false);

  const withSys = buildChatBody({
    system: '  be brief ',
    messages: [{ role: 'user', content: 'hi' }],
    maxTokens: 99999,
    temperature: 9,
    stream: false,
  });
  assert.equal(withSys.system, 'be brief');
  assert.equal(withSys.max_tokens, MAX_TOKENS_CLAMP);
  assert.equal(withSys.temperature, TEMPERATURE_MAX);
  assert.equal(withSys.stream, false);

  const cont = buildChatBody({ messages: [{ role: 'user', content: 'x' }], continue: true });
  assert.equal(cont.continue, true);
  // モデル名は差し替えられる
  assert.equal(buildChatBody({ model: 'm', messages: [] }).model, 'm');
  assert.deepEqual(buildChatBody({ messages: [] }).messages, []);
});

// ---- 4. TTFT / tok/s ------------------------------------------------------

test('TTFT は最初のトークンで確定し、tok/s は TTFT を除く区間で数える', () => {
  const st = createChatState();
  assert.deepEqual(chatFields(st).map((f) => f.value), ['-', '-']);

  chatBegin(st, 1000);
  assert.equal(st.ttftMs, null);
  chatOnToken(st, 1400);                       // 1 個目 = TTFT 400 ms
  assert.equal(st.ttftMs, 400);
  assert.equal(chatRate(st), null, '1 トークンでは速度を出さない');
  // 2 個目以降: (n-1) 個を経過時間で割る
  chatOnToken(st, 1500);
  chatOnToken(st, 1600);
  assert.equal(st.tokens, 3);
  assert.ok(Math.abs(chatRate(st) - 2 / 0.2) < 1e-9);
  assert.deepEqual(chatFields(st).map((f) => f.value), ['400 ms', '10.0']);
  // 順序とラベルは固定
  assert.deepEqual(chatFields(st).map((f) => f.label), ['ttft', 'tok/s']);
  assert.deepEqual(chatFields(st).map((f) => f.id), ['ttft', 'rate']);
});

test('chatEnd: usage.output_tokens で数え直し、終了後は 0.0 を出す', () => {
  const st = createChatState();
  chatBegin(st, 0);
  chatOnToken(st, 100);
  chatOnToken(st, 200);
  chatEnd(st, 500, 2);
  assert.equal(st.done, true);
  assert.equal(st.tokens, 2);
  assert.equal(st.outputTokens, 2);

  // 1 トークンだけの応答 → 速度は出せないが、終了後は 0.0 を出す
  const one = createChatState();
  chatBegin(one, 0);
  chatOnToken(one, 120);
  chatEnd(one, 130, 1);
  assert.deepEqual(chatFields(one).map((f) => f.value), ['120 ms', '0.0']);

  // usage が無ければ計器の値をそのまま使う (出力トークン数は不明)
  const noUsage = createChatState();
  chatBegin(noUsage, 0);
  chatOnToken(noUsage, 50);
  chatOnToken(noUsage, 60);
  chatEnd(noUsage, 70, null);
  assert.equal(noUsage.tokens, 2);
  assert.equal(noUsage.outputTokens, null);
});

test('時刻の書式: fmtMs / fmtRate', () => {
  assert.equal(fmtMs(null), '-');
  assert.equal(fmtMs(412), '412 ms');
  assert.equal(fmtMs(1234), '1.23 s');
  assert.equal(fmtRate(null), '-');
  assert.equal(fmtRate(9.876), '9.9');
  assert.equal(fmtRate(123.4), '123');
});

// ---- 5. SSE パーサ --------------------------------------------------------

test('createSseParser: チャンク境界をまたいでも event を落とさない', () => {
  const p = createSseParser();
  assert.deepEqual(p.push('data: {"a"'), []);      // 未完成
  assert.equal(p.pending(), 'data: {"a"');
  const out = p.push(':1}\n\n');
  assert.equal(out.length, 1);
  assert.equal(out[0].event, 'message');
  assert.equal(out[0].data, '{"a":1}');
  assert.equal(p.pending(), '');
});

test('createSseParser: event 名・複数 data 行・CRLF・コメント行', () => {
  const p = createSseParser();
  const out = p.push(': keepalive\r\nevent: ping\r\ndata: one\r\ndata: two\r\n\r\n');
  assert.equal(out.length, 1);
  assert.equal(out[0].event, 'ping');
  assert.equal(out[0].data, 'one\ntwo');

  // data 行が無い event は捨てる
  assert.deepEqual(p.push('event: empty\n\n'), []);
});

test('createSseParser: 改行で終わらない最後の event は flush で出す', () => {
  const p = createSseParser();
  assert.deepEqual(p.push('data: tail'), []);
  const out = p.flush();
  assert.equal(out.length, 1);
  assert.equal(out[0].data, 'tail');
  assert.deepEqual(p.flush(), [], 'flush は冪等');
});

test('parseSseBlock: data: の空白は 1 個だけ落とす', () => {
  assert.deepEqual(parseSseBlock('data:  x'), { event: 'message', data: ' x' });
  assert.equal(parseSseBlock(''), null);
  assert.equal(parseSseBlock(': only comment'), null);
});

// ---- 6. event → 意味 ------------------------------------------------------

function ev(data) {
  return { event: 'message', data: JSON.stringify(data) };
}

test('applyChatEvent: Anthropic の delta / message_delta / error', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'content_block_delta', delta: { type: 'text_delta', text: 'hi' } })),
    { text: 'hi' },
  );
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'message_delta', delta: { stop_reason: 'end_turn' }, usage: { output_tokens: 7 } })),
    { usage: { output_tokens: 7 }, stopReason: 'end_turn' },
  );
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'error', error: { message: 'boom' } })),
    { error: 'boom' },
  );
  // 本文の無いブロック開始などは扱わない
  assert.equal(applyChatEvent({}, ev({ type: 'content_block_start', index: 0 })), null);
  // JSON でない data は捨てる
  assert.equal(applyChatEvent({}, { event: 'message', data: 'not json' }), null);
});

test('applyChatEvent: OpenAI 互換の delta と usage も受ける', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ choices: [{ delta: { content: 'yo' } }] })),
    { text: 'yo' },
  );
  assert.deepEqual(
    applyChatEvent({}, ev({ choices: [{ delta: {} }], usage: { completion_tokens: 12 } })),
    { usage: { output_tokens: 12 }, stopReason: null },
  );
  assert.equal(applyChatEvent({}, ev({ choices: [{ delta: { content: '' } }] })), null);
});

// ---- 7. 画面との結線 ------------------------------------------------------

test('chat.html: chat.mjs を読み、操作の器と分割表示を持つ', () => {
  assert.ok(html.includes("./chat.mjs"), 'chat.mjs を import していない');
  for (const id of [
    'sys-preset', 'sys-text', 'temperature', 'max-tokens', 'send', 'stop',
    'badge-ttft', 'badge-rate', 'log', 'split-toggle', 'open-2d',
  ]) {
    assert.ok(html.includes(`id="${id}"`), `#${id} が無い`);
  }
  // 数字は画面で作らず、純ロジックの値と本文を使う
  for (const call of ['buildChatBody(', 'createSseParser(', 'chatOnToken(', 'chatFields(', 'normaliseControls(']) {
    assert.ok(html.includes(call), `画面が ${call} を呼んでいない`);
  }
  // 上流は既存のプロキシ (POST /chat) で、SSE をそのまま読む
  assert.ok(html.includes("fetch(CHAT_URL"), 'CHAT_URL を使っていない');
  assert.ok(html.includes("getReader()"), 'ストリームを読んでいない');
  // 分割表示は /2d?panes=map (地図だけ)
  assert.ok(html.includes('splitViewUrl()'), '分割表示の URL を使っていない');
  // 停止は /api/v1/abort (既存の api プロキシ)
  assert.ok(html.includes('/api/v1/abort'), '停止の経路が無い');
  assert.equal(SPLIT_2D_URL, splitViewUrl());
});

test('/2d から /chat へのリンクがある', () => {
  assert.ok(index2d.includes('id="chatpage"'), 'リンクの器が無い');
  assert.ok(index2d.includes('href="/chat"'), '/chat への href が無い');
  assert.equal(CHAT_URL, '/chat');
});
