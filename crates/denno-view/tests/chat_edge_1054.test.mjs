// denno-view /chat (task 1054) の SSE / 状態機械の端のテスト。
// 実行: node --test crates/denno-view/tests/*.test.mjs
//
// chat889.test.mjs が見ている「普通の経路」に対して、ここは壊れた入力・
// 極端な入力だけを見る:
//   1. SSE フレームを 1 バイトずつ / 任意の位置で切る (LF・CRLF・混在)
//   2. コメント行・keep-alive・data 行の無いブロック・壊れた JSON
//   3. Anthropic (content_block_delta / message_delta / error) と
//      OpenAI (choices[].delta.content / [DONE]) の差分配列
//   4. TTFT / tok/s の 0 トークン・1 トークン・同時刻・NaN・負の時刻
//   5. 操作値のクランプ (NaN / Infinity / 負の temperature と max_tokens)
import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CHAT_DEFAULTS, MAX_TOKENS_CLAMP,
  applyChatEvent, chatBegin, chatEnd, chatFields, chatOnToken, chatRate,
  clampMaxTokens, clampTemperature, createChatState, createSseParser,
  fmtMs, fmtRate, normaliseControls, parseSseBlock,
} from '../src/chat.mjs';

// ---- 1. チャンク分割 ------------------------------------------------------

// ストリームを 1 文字ずつ流し、完成した event を全部集める。
function pushBytewise(stream) {
  const p = createSseParser();
  const got = [];
  for (const ch of stream) got.push(...p.push(ch));
  got.push(...p.flush());
  return got;
}

test('SSE: 1 バイトずつ流しても LF のフレームを落とさない', () => {
  const stream =
    'event: ping\ndata: {"a":1}\n\n' +
    'data: {"type":"content_block_delta","delta":{"text":"x"}}\n\n' +
    'data: [DONE]\n\n';
  const got = pushBytewise(stream);
  assert.equal(got.length, 3);
  assert.deepEqual(got[0], { event: 'ping', data: '{"a":1}' });
  assert.deepEqual(got[1], { event: 'message', data: '{"type":"content_block_delta","delta":{"text":"x"}}' });
  assert.deepEqual(got[2], { event: 'message', data: '[DONE]' });
});

test('SSE: 1 バイトずつ流しても CRLF のフレームを落とさない', () => {
  const stream =
    ': keepalive\r\n' +
    'event: delta\r\n' +
    'data: one\r\n' +
    'data: two\r\n' +
    '\r\n' +
    'data: {"b":2}\r\n\r\n';
  const got = pushBytewise(stream);
  assert.equal(got.length, 2);
  assert.deepEqual(got[0], { event: 'delta', data: 'one\ntwo' });
  assert.deepEqual(got[1], { event: 'message', data: '{"b":2}' });
});

test('SSE: 任意の位置で切っても event の数と中身が変わらない', () => {
  const stream =
    'data: {"type":"content_block_delta","delta":{"text":"hello"}}\r\n\r\n' +
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}\r\n\r\n' +
    'data: [DONE]\n\n';

  const whole = createSseParser().push(stream);
  assert.equal(whole.length, 3);

  // すべての分割点で同じ結果になること (2 分割の全組み合わせ)
  for (let i = 0; i <= stream.length; i += 1) {
    const p = createSseParser();
    const got = [...p.push(stream.slice(0, i)), ...p.push(stream.slice(i))];
    got.push(...p.flush());
    assert.deepEqual(got, whole, `分割位置 ${i} で結果が変わった`);
  }
});

test('SSE: LF と CRLF が混ざったストリームでも区切れる', () => {
  const got = pushBytewise('data: a\n\ndata: b\r\n\r\ndata: c\n\r\n');
  assert.deepEqual(got.map((e) => e.data), ['a', 'b', 'c']);
});

// ---- 2. 壊れたフレームと keep-alive --------------------------------------

test('SSE: コメントだけのブロックと keep-alive は event にならない', () => {
  const p = createSseParser();
  assert.deepEqual(p.push(': ping\n\n'), []);
  assert.deepEqual(p.push(':keepalive\r\n\r\n'), []);
  assert.deepEqual(p.push(': a\n: b\n\n'), []);
  // コメント行は data 行と同居しても捨てる
  const out = p.push(': note\ndata: ok\n\n');
  assert.deepEqual(out, [{ event: 'message', data: 'ok' }]);
  assert.equal(p.pending(), '');
});

test('SSE: data 行の無いブロックと空ブロックは捨てる', () => {
  const p = createSseParser();
  assert.deepEqual(p.push('event: empty\n\n'), []);
  assert.deepEqual(p.push('\n\n'), []);
  assert.deepEqual(p.push('\r\n\r\n'), []);
  assert.deepEqual(p.push('id: 7\nretry: 100\n\n'), []);
  assert.equal(p.pending(), '');
});

test('SSE: 未完成のフレームは pending に残り、flush で 1 回だけ出る', () => {
  const p = createSseParser();
  assert.deepEqual(p.push('event: ping\ndata: {"x":1}'), []);
  assert.equal(p.pending(), 'event: ping\ndata: {"x":1}');
  const out = p.flush();
  assert.deepEqual(out, [{ event: 'ping', data: '{"x":1}' }]);
  assert.equal(p.pending(), '');
  assert.deepEqual(p.flush(), []);
});

test('SSE: 壊れた JSON の payload は applyChatEvent が null で捨てる', () => {
  const p = createSseParser();
  const frames = [
    'data: {"type":"content_block_delta",\n\n', // 途中で切れた JSON
    'data: not json\n\n',
    'data: \n\n',
    'data: {"type":"content_block_delta","delta":}\n\n',
  ];
  const out = [];
  for (const f of frames) out.push(...p.push(f));
  // 空 data のブロックは push が data 空を見て落とす (3 番目)
  assert.equal(out.length, 3);
  for (const ev of out) assert.equal(applyChatEvent({}, ev), null);

  // 複数 data 行の連結が壊れた JSON になる場合も null
  const joined = parseSseBlock('data: {"type":"x",\ndata: "y":1}');
  assert.equal(applyChatEvent({}, joined), null);
});

test('parseSseBlock: data: の直後の空白は 1 個だけ、空 data は data 空のブロック', () => {
  assert.deepEqual(parseSseBlock('data:no space'), { event: 'message', data: 'no space' });
  assert.deepEqual(parseSseBlock('data:  two'), { event: 'message', data: ' two' });
  // 空の data 行はブロックとしては返るが、パーサの push が捨てる (下のテスト)
  assert.deepEqual(parseSseBlock('data:'), { event: 'message', data: '' });
  assert.equal(parseSseBlock('event: message'), null);
  assert.equal(parseSseBlock(null), null);
});

// ---- 3. Anthropic / OpenAI のイベント ------------------------------------

function ev(data) {
  return { event: 'message', data: typeof data === 'string' ? data : JSON.stringify(data) };
}

test('Anthropic: content_block_delta は text だけを返す', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'content_block_delta', delta: { type: 'text_delta', text: '' } })),
    { text: '' }, // 空文字も「1 トークンぶんの差分」として通す
  );
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'content_block_delta', delta: { type: 'text_delta', text: 'multi word' } })),
    { text: 'multi word' },
  );
  // text が文字列でない差分は扱わない
  assert.equal(applyChatEvent({}, ev({ type: 'content_block_delta', delta: { text: 5 } })), null);
  assert.equal(applyChatEvent({}, ev({ type: 'content_block_delta' })), null);
});

test('Anthropic: message_delta は usage と stop_reason を返す (欠けても壊れない)', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'message_delta', delta: { stop_reason: 'max_tokens' }, usage: { output_tokens: 0 } })),
    { usage: { output_tokens: 0 }, stopReason: 'max_tokens' },
  );
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'message_delta' })),
    { usage: null, stopReason: null },
  );
});

test('Anthropic: error は message を返し、無ければ "error"', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ type: 'error', error: { type: 'overloaded_error', message: 'busy' } })),
    { error: 'busy' },
  );
  assert.deepEqual(applyChatEvent({}, ev({ type: 'error' })), { error: 'error' });
  assert.deepEqual(applyChatEvent({}, ev({ type: 'error', error: {} })), { error: 'error' });
});

test('OpenAI: choices[0].delta.content を取り、空差分は無視する', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ choices: [{ delta: { content: 'hi' } }] })),
    { text: 'hi' },
  );
  assert.equal(applyChatEvent({}, ev({ choices: [{ delta: { content: '' } }] })), null);
  assert.equal(applyChatEvent({}, ev({ choices: [{ delta: { role: 'assistant' } }] })), null);
  assert.equal(applyChatEvent({}, ev({ choices: [] })), null);
  assert.equal(applyChatEvent({}, ev({ choices: [{ delta: { content: ['x'] } }] })), null);
  // [DONE] は JSON ではないので null (呼び側が終端として扱う)
  assert.equal(applyChatEvent({}, ev('[DONE]')), null);
});

test('OpenAI: usage.completion_tokens を output_tokens に写す', () => {
  assert.deepEqual(
    applyChatEvent({}, ev({ choices: [{ delta: {} }], usage: { completion_tokens: 12 } })),
    { usage: { output_tokens: 12 }, stopReason: null },
  );
  // 文字列の数字も受ける
  assert.deepEqual(
    applyChatEvent({}, ev({ choices: [{ delta: {} }], usage: { completion_tokens: '3' } })),
    { usage: { output_tokens: 3 }, stopReason: null },
  );
  // 数でない usage は捨てる
  assert.equal(applyChatEvent({}, ev({ usage: { completion_tokens: 'x' } })), null);
  assert.equal(applyChatEvent({}, ev({ usage: {} })), null);
});

// ---- 4. TTFT / tok/s の端 ------------------------------------------------

test('TTFT: 0 トークンで終わっても計器が壊れない', () => {
  const st = createChatState();
  chatBegin(st, 1000);
  chatEnd(st, 1250, 0);
  assert.equal(st.tokens, 0);
  assert.equal(st.outputTokens, 0);
  assert.equal(st.ttftMs, null);
  assert.equal(chatRate(st), null);
  assert.deepEqual(chatFields(st).map((f) => f.value), ['-', '0.0']);
});

test('TTFT: 1 トークンは TTFT だけ確定し、tok/s は 0.0 (終了後)', () => {
  const st = createChatState();
  chatBegin(st, 1000);
  chatOnToken(st, 1100);
  assert.equal(st.ttftMs, 100);
  assert.equal(chatRate(st), null, '1 トークン以下では速度を出さない');
  chatEnd(st, 1100, 1);
  assert.deepEqual(chatFields(st).map((f) => f.value), ['100 ms', '0.0']);
});

test('TTFT: 全トークンが同時刻なら TTFT 0 ms、速度は出さない', () => {
  const st = createChatState();
  chatBegin(st, 500);
  chatOnToken(st, 500);
  chatOnToken(st, 500);
  chatOnToken(st, 500);
  assert.equal(st.tokens, 3);
  assert.equal(st.ttftMs, 0);
  assert.equal(chatRate(st), null, 'span 0 では割らない');
  assert.deepEqual(chatFields(st).map((f) => f.value), ['0 ms', '-']);
  chatEnd(st, 500, 3);
  assert.deepEqual(chatFields(st).map((f) => f.value), ['0 ms', '0.0']);
});

test('TTFT: 負の経過時間は 0 に丸める', () => {
  const st = createChatState();
  chatBegin(st, 1000);
  chatOnToken(st, 400); // 時計が逆行した
  assert.equal(st.ttftMs, 0);
  assert.equal(fmtMs(st.ttftMs), '0 ms');
});

test('TTFT: NaN の時刻でも表示は "-" に落ちる (例外を投げない)', () => {
  const st = createChatState();
  chatBegin(st, NaN);
  chatOnToken(st, NaN);
  assert.equal(fmtMs(st.ttftMs), '-');
  assert.equal(fmtRate(chatRate(st)), '-');
  assert.deepEqual(chatFields(st).map((f) => f.value), ['-', '-']);
  chatEnd(st, NaN, 1);
  assert.equal(fmtMs(st.ttftMs), '-');
});

test('TTFT: 開始前のトークンは数えない', () => {
  const st = createChatState();
  chatOnToken(st, 10); // chatBegin していない
  assert.equal(st.tokens, 0);
  assert.equal(st.firstTokenAt, null);
  assert.equal(st.ttftMs, null);
});

test('chatEnd: 不正な usage は計器の値を上書きしない', () => {
  const cases = [null, undefined, '', 'abc', NaN, -1, -3.5];
  for (const bad of cases) {
    const st = createChatState();
    chatBegin(st, 0);
    chatOnToken(st, 10);
    chatOnToken(st, 20);
    chatEnd(st, 30, bad);
    assert.equal(st.tokens, 2, `usage=${String(bad)} で上書きされた`);
    assert.equal(st.outputTokens, null);
  }
  // 小数の usage は切り捨てる
  const st = createChatState();
  chatBegin(st, 0);
  chatOnToken(st, 10);
  chatOnToken(st, 20);
  chatEnd(st, 30, 2.9);
  assert.equal(st.tokens, 2);
  assert.equal(st.outputTokens, 2);
});

test('chatEnd: トークンが来なくても lastTokenAt を now で埋める', () => {
  const st = createChatState();
  chatBegin(st, 0);
  chatEnd(st, 75, null);
  assert.equal(st.lastTokenAt, 75);
  assert.equal(st.firstTokenAt, null);
  assert.equal(chatRate(st), null);
});

// ---- 5. 操作値のクランプ -------------------------------------------------

test('clampTemperature: NaN / Infinity / 負は既定か 0 に落ちる', () => {
  assert.equal(clampTemperature(NaN), CHAT_DEFAULTS.temperature);
  assert.equal(clampTemperature(Infinity), CHAT_DEFAULTS.temperature);
  assert.equal(clampTemperature(-Infinity), CHAT_DEFAULTS.temperature);
  assert.equal(clampTemperature(null), 0, 'Number(null) は 0');
  assert.equal(clampTemperature(undefined), CHAT_DEFAULTS.temperature);
  assert.equal(clampTemperature(-0.001), 0);
  assert.equal(clampTemperature(2.0001), 2);
  assert.equal(clampTemperature(0), 0);
  assert.equal(clampTemperature(2), 2);
  // fallback は呼び側が指定できる
  assert.equal(clampTemperature(NaN, 1), 1);
  assert.equal(clampTemperature(9, 1), 2, 'clamp は fallback より優先');
});

test('clampMaxTokens: NaN / Infinity / 負は既定、上限は 0 未満にしない', () => {
  assert.equal(clampMaxTokens(NaN), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(Infinity), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(-Infinity), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(-1), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(undefined), CHAT_DEFAULTS.maxTokens);
  assert.equal(clampMaxTokens(null), 0, 'Number(null) は 0 (prefill のみ)');
  assert.equal(clampMaxTokens(4097), MAX_TOKENS_CLAMP);
  assert.equal(clampMaxTokens(4096), MAX_TOKENS_CLAMP);
  // limit が負なら 0 に丸めた上限で止める
  assert.equal(clampMaxTokens(100, 8, -5), 0);
  // 小数は切り捨て
  assert.equal(clampMaxTokens(12.99), 12);
});

test('normaliseControls: 壊れた操作値でも既定に戻る', () => {
  const nan = normaliseControls({ presetId: 'plain', temperature: NaN, maxTokens: NaN });
  assert.equal(nan.temperature, CHAT_DEFAULTS.temperature);
  assert.equal(nan.maxTokens, CHAT_DEFAULTS.maxTokens);

  const inf = normaliseControls({ presetId: 'plain', temperature: Infinity, maxTokens: Infinity });
  assert.equal(inf.temperature, CHAT_DEFAULTS.temperature);
  assert.equal(inf.maxTokens, CHAT_DEFAULTS.maxTokens);

  const neg = normaliseControls({ presetId: 'plain', temperature: -3, maxTokens: -9 });
  assert.equal(neg.temperature, 0);
  assert.equal(neg.maxTokens, CHAT_DEFAULTS.maxTokens);

  // limit が指定されていればそちらで頭打ち
  const lim = normaliseControls({ presetId: 'plain', temperature: 1, maxTokens: 9000, limit: 700 });
  assert.equal(lim.maxTokens, 700);
});
