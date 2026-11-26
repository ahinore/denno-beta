// denno-view の接続状態と再接続バックオフ (task 1106)。
// 純関数だけを置く。DOM / fetch / WebSocket / setTimeout / Math.random には触らない。
// 3 画面 (/, /2d, /3d) が同じ規則で「切断 → 再試行 → オフライン」を表示できるように、
// 待ち時間の計算と文言をここに集める。jitter は入れない (決定性を優先)。

export const RECONNECT_SCHEMA = 1;

// 既定の再試行規則。0.5 s から 2 倍で伸ばし、15 s で頭打ち。
// offlineAfter 回以上続けて失敗したら「オフライン」として帯を出す。
export const RECONNECT_DEFAULTS = Object.freeze({
  baseMs: 500,
  maxMs: 15000,
  factor: 2,
  maxAttempts: 12,
  offlineAfter: 3,
});

// 部分指定の opts を既定値で埋める。不正な値は既定へ落とす。
export function normaliseOptions(opts) {
  const out = {
    baseMs: RECONNECT_DEFAULTS.baseMs,
    maxMs: RECONNECT_DEFAULTS.maxMs,
    factor: RECONNECT_DEFAULTS.factor,
    maxAttempts: RECONNECT_DEFAULTS.maxAttempts,
    offlineAfter: RECONNECT_DEFAULTS.offlineAfter,
  };
  if (opts && typeof opts === 'object') {
    for (const key of Object.keys(out)) {
      const v = opts[key];
      if (typeof v === 'number' && Number.isFinite(v) && v > 0) out[key] = v;
    }
  }
  out.maxAttempts = Math.max(1, Math.floor(out.maxAttempts));
  out.offlineAfter = Math.max(1, Math.floor(out.offlineAfter));
  if (out.maxMs < out.baseMs) out.maxMs = out.baseMs;
  if (out.factor < 1) out.factor = 1;
  if (out.offlineAfter > out.maxAttempts) out.offlineAfter = out.maxAttempts;
  return out;
}

// 失敗回数を [0, maxAttempts] に丸める。0 は「まだ失敗していない」。
export function clampAttempt(attempt, opts) {
  const cfg = normaliseOptions(opts);
  const n = typeof attempt === 'number' && Number.isFinite(attempt) ? Math.floor(attempt) : 0;
  if (n <= 0) return 0;
  return Math.min(n, cfg.maxAttempts);
}

// n 回目の失敗のあとに待つ時間 (ms)。1 回目が baseMs、以降 factor 倍、maxMs で頭打ち。
export function backoffDelay(attempt, opts) {
  const cfg = normaliseOptions(opts);
  const n = clampAttempt(attempt, cfg);
  if (n === 0) return 0;
  return Math.min(cfg.maxMs, Math.round(cfg.baseMs * Math.pow(cfg.factor, n - 1)));
}

// 連続失敗が offlineAfter 以上なら、利用者には「オフライン」と見せる。
export function isOffline(attempt, opts) {
  const cfg = normaliseOptions(opts);
  return clampAttempt(attempt, cfg) >= cfg.offlineAfter;
}

// "0.5 s" / "2 s" / "15 s" / "now"。端数は 1 桁だけ出す。
export function formatRetry(ms) {
  const n = typeof ms === 'number' && Number.isFinite(ms) && ms > 0 ? ms : 0;
  if (n === 0) return 'now';
  const s = n / 1000;
  return (Number.isInteger(s) ? String(s) : s.toFixed(1)) + ' s';
}

// 接続の見え方 (状態レベル / ラベル / 帯の文言) を 1 か所で決める。
// input: { connected, attempt, pollFailing }。opts は backoff と同じ。
export function connectionStatus(input, opts) {
  const cfg = normaliseOptions(opts);
  const src = input && typeof input === 'object' ? input : {};
  const attempt = clampAttempt(src.attempt, cfg);
  const connected = src.connected === true;
  const pollFailing = src.pollFailing === true;
  const next = backoffDelay(attempt, cfg);
  const retry = formatRetry(next);

  if (connected && !pollFailing) {
    return {
      level: 'online',
      label: 'connected',
      banner: '',
      showBanner: false,
      retryInMs: 0,
      attempt,
    };
  }
  if (connected && pollFailing) {
    return {
      level: 'degraded',
      label: 'connected (polling failing)',
      banner: 'telemetry polling is failing - will retry',
      showBanner: true,
      retryInMs: 0,
      attempt,
    };
  }
  if (attempt === 0) {
    return {
      level: 'connecting',
      label: 'connecting',
      banner: '',
      showBanner: false,
      retryInMs: 0,
      attempt,
    };
  }
  if (isOffline(attempt, cfg)) {
    return {
      level: 'offline',
      label: 'offline - retry #' + attempt + ' in ' + retry,
      banner: 'offline - telemetry unreachable, retrying in ' + retry,
      showBanner: true,
      retryInMs: next,
      attempt,
    };
  }
  return {
    level: 'reconnecting',
    label: 'disconnected - retry #' + attempt + ' in ' + retry,
    banner: 'disconnected - retrying in ' + retry,
    showBanner: false,
    retryInMs: next,
    attempt,
  };
}

// 帯に出す 1 行だけが欲しいとき用。
export function connectionBanner(input, opts) {
  return connectionStatus(input, opts).banner;
}

// 画面が持つ小さな状態機械。タイマーは持たず、待ち時間を返すだけ
// (実際の setTimeout は呼び出し側 = 各画面)。
export function createReconnect(opts) {
  const cfg = normaliseOptions(opts);
  let attempt = 0;
  let open = false;
  let pollFailing = false;

  const self = {
    onOpen() {
      open = true;
      attempt = 0;
      pollFailing = false;
      return self.status();
    },
    // 切断を見た。次の待ち時間 (ms) を返す。
    onClose() {
      open = false;
      attempt = clampAttempt(attempt + 1, cfg);
      return backoffDelay(attempt, cfg);
    },
    onPollOk() {
      pollFailing = false;
      return self.status();
    },
    onPollError() {
      pollFailing = true;
      return self.status();
    },
    reset() {
      open = false;
      attempt = 0;
      pollFailing = false;
    },
    status() {
      return connectionStatus({ connected: open, attempt, pollFailing }, cfg);
    },
    banner() {
      return self.status().banner;
    },
  };

  Object.defineProperties(self, {
    attempt: { get: () => attempt, enumerable: true },
    connected: { get: () => open, enumerable: true },
    pollFailing: { get: () => pollFailing, enumerable: true },
    nextDelayMs: { get: () => backoffDelay(attempt, cfg), enumerable: true },
    offline: { get: () => isOffline(attempt, cfg), enumerable: true },
    options: { get: () => Object.assign({}, cfg), enumerable: true },
  });
  return self;
}
