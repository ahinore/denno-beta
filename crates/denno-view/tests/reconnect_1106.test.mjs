// task 1106: 再接続バックオフと接続状態の純関数 (src/reconnect.mjs) の単体テスト。
// DOM には触らない。node --test crates/denno-view/tests/ で走る。
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import {
  RECONNECT_SCHEMA,
  RECONNECT_DEFAULTS,
  normaliseOptions,
  clampAttempt,
  backoffDelay,
  isOffline,
  formatRetry,
  connectionStatus,
  connectionBanner,
  createReconnect,
} from '../src/reconnect.mjs';

const SRC = readFileSync(new URL('../src/reconnect.mjs', import.meta.url), 'utf8');
// コメント行は除いて、実コードだけを見る (禁止語チェック用)。
const CODE = SRC.split('\n').filter((line) => !line.trim().startsWith('//')).join('\n');

test('schema と既定値の形が固定されている', () => {
  assert.equal(RECONNECT_SCHEMA, 1);
  assert.deepEqual(RECONNECT_DEFAULTS, {
    baseMs: 500,
    maxMs: 15000,
    factor: 2,
    maxAttempts: 12,
    offlineAfter: 3,
  });
  assert.ok(Object.isFrozen(RECONNECT_DEFAULTS), '既定値は凍結されている');
});

test('backoff は 0.5 s から 2 倍で伸び、15 s で頭打ち', () => {
  assert.equal(backoffDelay(1), 500);
  assert.equal(backoffDelay(2), 1000);
  assert.equal(backoffDelay(3), 2000);
  assert.equal(backoffDelay(4), 4000);
  assert.equal(backoffDelay(5), 8000);
  assert.equal(backoffDelay(6), 15000);
  assert.equal(backoffDelay(12), 15000);
  assert.equal(backoffDelay(100), 15000);
});

test('attempt 0 / 負 / NaN は待たない (0 ms)', () => {
  assert.equal(backoffDelay(0), 0);
  assert.equal(backoffDelay(-3), 0);
  assert.equal(backoffDelay(Number.NaN), 0);
  assert.equal(backoffDelay('2'), 0);
});

test('backoff は単調非減少で maxMs を超えず、jitter が無い (2 回呼んで同値)', () => {
  let prev = 0;
  for (let n = 0; n <= 20; n += 1) {
    const d = backoffDelay(n);
    assert.ok(d >= prev, `attempt ${n} で待ち時間が縮んだ`);
    assert.ok(d <= RECONNECT_DEFAULTS.maxMs, `attempt ${n} で上限超過: ${d}`);
    assert.equal(d, backoffDelay(n), '同じ入力で同じ値 (決定性)');
    prev = d;
  }
  assert.ok(!CODE.includes('Math.random'), 'jitter 用の乱数を使っていない');
});

test('opts で base / factor / max を差し替えられる', () => {
  const o = { baseMs: 100, factor: 3, maxMs: 1000, maxAttempts: 8, offlineAfter: 2 };
  assert.equal(backoffDelay(1, o), 100);
  assert.equal(backoffDelay(2, o), 300);
  assert.equal(backoffDelay(3, o), 900);
  assert.equal(backoffDelay(4, o), 1000);
  assert.equal(backoffDelay(8, o), 1000);
});

test('不正な opts は既定へ落ちる', () => {
  const cfg = normaliseOptions({ baseMs: -1, factor: 0.5, maxMs: 10 });
  assert.equal(cfg.baseMs, RECONNECT_DEFAULTS.baseMs);
  assert.equal(cfg.maxMs, RECONNECT_DEFAULTS.baseMs, 'maxMs < baseMs は baseMs に揃える');
  assert.equal(cfg.factor, 1, 'factor < 1 は 1 に丸める');
  const nan = normaliseOptions({ baseMs: Number.NaN });
  assert.equal(nan.baseMs, RECONNECT_DEFAULTS.baseMs);
});

test('clampAttempt は [0, maxAttempts] に丸める', () => {
  assert.equal(clampAttempt(0), 0);
  assert.equal(clampAttempt(-2), 0);
  assert.equal(clampAttempt(2.9), 2);
  assert.equal(clampAttempt(12), 12);
  assert.equal(clampAttempt(99), 12);
  assert.equal(clampAttempt(99, { maxAttempts: 4 }), 4);
});

test('offline 判定は offlineAfter 回目の失敗から', () => {
  assert.equal(isOffline(0), false);
  assert.equal(isOffline(2), false);
  assert.equal(isOffline(3), true);
  assert.equal(isOffline(9), true);
  assert.equal(isOffline(1, { offlineAfter: 1 }), true);
});

test('formatRetry の表記', () => {
  assert.equal(formatRetry(0), 'now');
  assert.equal(formatRetry(-5), 'now');
  assert.equal(formatRetry(Number.NaN), 'now');
  assert.equal(formatRetry(500), '0.5 s');
  assert.equal(formatRetry(1500), '1.5 s');
  assert.equal(formatRetry(2000), '2 s');
  assert.equal(formatRetry(15000), '15 s');
});

test('connectionStatus: 接続中は帯を出さない', () => {
  const s = connectionStatus({ connected: true, attempt: 0 });
  assert.deepEqual(s, {
    level: 'online', label: 'connected', banner: '', showBanner: false, retryInMs: 0, attempt: 0,
  });
});

test('connectionStatus: 未接続 + attempt 0 は connecting', () => {
  const s = connectionStatus({});
  assert.equal(s.level, 'connecting');
  assert.equal(s.label, 'connecting');
  assert.equal(s.showBanner, false);
});

test('connectionStatus: 1-2 回目の失敗は reconnecting (帯なし)', () => {
  const s = connectionStatus({ connected: false, attempt: 2 });
  assert.equal(s.level, 'reconnecting');
  assert.equal(s.attempt, 2);
  assert.equal(s.retryInMs, 1000);
  assert.equal(s.label, 'disconnected - retry #2 in 1 s');
  assert.equal(s.banner, 'disconnected - retrying in 1 s');
  assert.equal(s.showBanner, false);
});

test('connectionStatus: offlineAfter 回目から offline (帯あり)', () => {
  const s = connectionStatus({ connected: false, attempt: 3 });
  assert.equal(s.level, 'offline');
  assert.equal(s.retryInMs, 2000);
  assert.equal(s.label, 'offline - retry #3 in 2 s');
  assert.equal(s.banner, 'offline - telemetry unreachable, retrying in 2 s');
  assert.equal(s.showBanner, true);
});

test('connectionStatus: 接続中でも polling が落ちていれば degraded', () => {
  const s = connectionStatus({ connected: true, pollFailing: true });
  assert.equal(s.level, 'degraded');
  assert.equal(s.showBanner, true);
  assert.equal(s.banner, 'telemetry polling is failing - will retry');
});

test('connectionStatus: opts が offline の閾値を変える', () => {
  const o = { offlineAfter: 1, baseMs: 250, maxMs: 4000 };
  const s = connectionStatus({ connected: false, attempt: 1 }, o);
  assert.equal(s.level, 'offline');
  assert.equal(s.retryInMs, 250);
});

test('connectionBanner は status.banner と一致する', () => {
  const input = { connected: false, attempt: 4 };
  assert.equal(connectionBanner(input), connectionStatus(input).banner);
  assert.equal(connectionBanner({ connected: true }), '');
});

test('createReconnect: open で attempt が 0 に戻る', () => {
  const rc = createReconnect();
  assert.equal(rc.status().level, 'connecting');
  rc.onClose();
  rc.onClose();
  assert.equal(rc.attempt, 2);
  const s = rc.onOpen();
  assert.equal(s.level, 'online');
  assert.equal(rc.attempt, 0);
  assert.equal(rc.connected, true);
});

test('createReconnect: onClose が返す待ち時間は増えて上限で止まる', () => {
  const rc = createReconnect({ maxAttempts: 5, maxMs: 4000 });
  const got = [];
  for (let i = 0; i < 7; i += 1) got.push(rc.onClose());
  assert.deepEqual(got, [500, 1000, 2000, 4000, 4000, 4000, 4000]);
  assert.equal(rc.status().level, 'offline');
  assert.equal(rc.offline, true);
});

test('createReconnect: poll の失敗と回復', () => {
  const rc = createReconnect();
  rc.onOpen();
  assert.equal(rc.onPollError().level, 'degraded');
  assert.equal(rc.pollFailing, true);
  assert.equal(rc.onPollOk().level, 'online');
  assert.equal(rc.pollFailing, false);
});

test('createReconnect: reset で初期状態に戻る', () => {
  const rc = createReconnect();
  rc.onClose();
  rc.onClose();
  rc.onClose();
  assert.equal(rc.offline, true);
  rc.reset();
  assert.equal(rc.attempt, 0);
  assert.equal(rc.offline, false);
  assert.equal(rc.status().level, 'connecting');
});

test('createReconnect: options は既定値の写しで、外から壊せない', () => {
  const rc = createReconnect({ baseMs: 200 });
  const o = rc.options;
  assert.equal(o.baseMs, 200);
  o.baseMs = 9999;
  assert.equal(rc.options.baseMs, 200, '内部設定が書き換わった');
});

// コメント行は除いて、実際のコードだけを見る。
test('モジュールは DOM / タイマー / 乱数に依存しない (純関数の証明)', () => {
  for (const bad of ['document', 'window', 'fetch(', 'WebSocket', 'setTimeout', 'setInterval', 'Math.random', 'console.']) {
    assert.ok(!CODE.includes(bad), `reconnect.mjs のコードが ${bad} に触っている`);
  }
});
