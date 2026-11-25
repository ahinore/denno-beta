// denno-view /chat の純ロジック (task 889)。DOM に触らない。
//
// 画面 (chat.html) はここだけを呼ぶ。ここで決めているもの:
//   1. システムプロンプトのプリセット (選択肢と本文)
//   2. temperature / max_tokens の検証とクランプ (サーバーの上限に合わせる)
//   3. POST /chat (= 上流 /v1/messages) の本文の組み立て
//   4. SSE の解釈 (data: 行の連結 → event 列)
//   5. TTFT / 出力 tok/s の状態機械 (画面は数字を作らない)
//
// サーバーの規則は denno-server の MessagesRequest に合わせる:
//   system は文字列 (空なら送らない) / temperature は 0..2 /
//   max_tokens は MAX_TOKENS_CLAMP (4096) で clamp。

export const CHAT_SCHEMA = 1;

/// 上流 (/v1/messages) への入口。denno-view の既存プロキシ。
export const CHAT_URL = '/chat';
/// 分割表示で開く /2d (地図だけの並び)。
export const SPLIT_2D_URL = '/2d?panes=map';

/// システムプロンプトのプリセット。text が空 = システムプロンプト無し。
/// 画面の <option> はこの配列から作る (HTML に本文を二重に書かない)。
export const SYSTEM_PRESETS = Object.freeze([
  { id: 'plain', label: 'plain (no system prompt)', text: '' },
  { id: 'concise', label: 'concise', text: 'Answer in one short paragraph. No preamble.' },
  { id: 'code', label: 'code assistant', text: 'You are a careful coding assistant. Show code, then a one-line explanation.' },
  { id: 'ja', label: 'ja 簡潔', text: '日本語で簡潔に答えてください。前置きは書かない。' },
  { id: 'tutor', label: 'tutor', text: 'Explain step by step for a beginner, using small numbered steps.' },
]);

/// 既定値。maxTokens は /v1/limits が取れたら画面が上書きする。
export const CHAT_DEFAULTS = Object.freeze({
  presetId: 'plain',
  temperature: 0.7,
  maxTokens: 512,
  model: 'denno',
});

export const TEMPERATURE_MAX = 2;
export const MAX_TOKENS_CLAMP = 4096;

/// プリセット id → プリセット。未知の id は null。
export function systemPreset(id) {
  const want = String(id == null ? '' : id);
  for (const p of SYSTEM_PRESETS) if (p.id === want) return p;
  return null;
}

/// 温度を [0, TEMPERATURE_MAX] に丸める。数でない / 非有限は既定へ。
export function clampTemperature(v, fallback = CHAT_DEFAULTS.temperature) {
  const n = Number(v);
  if (!Number.isFinite(n)) return fallback;
  if (n < 0) return 0;
  if (n > TEMPERATURE_MAX) return TEMPERATURE_MAX;
  return n;
}

/// 生成上限を [0, clamp] に丸める (0 = prefill のみ、はサーバーも許す)。
export function clampMaxTokens(v, fallback = CHAT_DEFAULTS.maxTokens, limit = MAX_TOKENS_CLAMP) {
  const n = Math.trunc(Number(v));
  if (!Number.isFinite(n) || n < 0) return fallback;
  const hi = Math.max(0, Math.trunc(Number(limit)));
  return n > hi ? hi : n;
}

/// 画面の入力 → 送る値。system は「プリセットの本文」と「手で書き換えた欄」の
/// どちらか。custom が空でなければそちらを優先する (プリセット選択は下敷き)。
export function normaliseControls(opts) {
  const o = opts || {};
  const preset = systemPreset(o.presetId);
  const presetText = preset ? preset.text : '';
  const custom = typeof o.system === 'string' ? o.system : '';
  const text = custom.trim() ? custom : presetText;
  return {
    systemText: text.trim(),
    temperature: clampTemperature(o.temperature),
    maxTokens: clampMaxTokens(o.maxTokens, CHAT_DEFAULTS.maxTokens, o.limit),
  };
}

/// 履歴を API の形に落とす。空の content と未知の role は捨てる。
export function normaliseHistory(history) {
  const out = [];
  for (const m of Array.isArray(history) ? history : []) {
    if (!m) continue;
    const role = m.role === 'assistant' ? 'assistant' : m.role === 'user' ? 'user' : null;
    if (!role) continue;
    const content = typeof m.content === 'string' ? m.content : '';
    if (!content) continue;
    out.push({ role, content });
  }
  return out;
}

/// POST /chat の本文。maxTokens が 0 のときも送る (prefill のみは有効な指定)。
/// continue は /2d の「続きを生成」と同じ拡張。
export function buildChatBody(opts) {
  const o = opts || {};
  const messages = normaliseHistory(o.messages);
  const body = {
    model: o.model || CHAT_DEFAULTS.model,
    max_tokens: clampMaxTokens(o.maxTokens, CHAT_DEFAULTS.maxTokens, o.limit),
    temperature: clampTemperature(o.temperature),
    stream: o.stream !== false,
    messages,
  };
  const system = typeof o.system === 'string' ? o.system.trim() : '';
  if (system) body.system = system;
  if (o.continue) body.continue = true;
  return body;
}

// ---- TTFT / 出力 tok/s ----------------------------------------------------

/// 1 リクエストぶんの計器。画面は chatFields の value を貼るだけ。
export function createChatState() {
  return {
    startedAt: null,
    firstTokenAt: null,
    lastTokenAt: null,
    tokens: 0,
    ttftMs: null,
    outputTokens: null,
    done: false,
  };
}

/// リクエスト開始。
export function chatBegin(st, now) {
  st.startedAt = now;
  st.firstTokenAt = null;
  st.lastTokenAt = null;
  st.tokens = 0;
  st.ttftMs = null;
  st.outputTokens = null;
  st.done = false;
  return st;
}

/// 出力の差分 1 つ = 1 トークン。最初の 1 つで TTFT が確定する。
export function chatOnToken(st, now) {
  if (st.startedAt == null) return st;
  st.tokens += 1;
  st.lastTokenAt = now;
  if (st.firstTokenAt == null) {
    st.firstTokenAt = now;
    st.ttftMs = Math.max(0, now - st.startedAt);
  }
  return st;
}

/// 終了。usage.output_tokens があればトークン数をそれで直す
/// (差分の数とずれることがある。表示は usage を正とする)。
export function chatEnd(st, now, outputTokens) {
  st.done = true;
  // null/undefined は「usage が無い」。Number(null) は 0 になるので先に弾く。
  if (outputTokens != null && outputTokens !== '') {
    const n = Number(outputTokens);
    if (Number.isFinite(n) && n >= 0) {
      st.tokens = Math.trunc(n);
      st.outputTokens = st.tokens;
    }
  }
  if (st.lastTokenAt == null && st.startedAt != null) st.lastTokenAt = now;
  return st;
}

/// 出力 tok/s。TTFT を除いた区間で、最初の 1 トークンは数えない
/// (hud2d.mjs と同じ規約)。1 トークン以下では出さない。
export function chatRate(st) {
  if (st.firstTokenAt == null || st.lastTokenAt == null) return null;
  const span = (st.lastTokenAt - st.firstTokenAt) / 1000;
  if (span <= 0) return null;
  const n = st.tokens - 1;
  if (n <= 0) return null;
  return n / span;
}

export function fmtMs(v) {
  if (v == null || v === '') return '-';
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  if (n >= 1000) return (n / 1000).toFixed(2) + ' s';
  return Math.round(n) + ' ms';
}

export function fmtRate(v) {
  if (v == null || v === '') return '-';
  const n = Number(v);
  if (!Number.isFinite(n)) return '-';
  if (n >= 100) return n.toFixed(0);
  return n.toFixed(1);
}

/// 画面のバッジ 2 つ (ttft / tok/s)。順序とラベルはテストで固定する。
export function chatFields(st) {
  return [
    { id: 'ttft', label: 'ttft', value: st.ttftMs == null ? '-' : fmtMs(st.ttftMs) },
    { id: 'rate', label: 'tok/s', value: st.done && chatRate(st) == null ? '0.0' : fmtRate(chatRate(st)) },
  ];
}

// ---- SSE -----------------------------------------------------------------

/// SSE の増分パーサ。push(text) は受け取ったチャンクを足し、完成した
/// event (data 行の連結) の配列を返す。区切りは空行 (\n\n)、CRLF も許す。
/// event 名は `event:` 行から取り、無ければ 'message'。
export function createSseParser() {
  let buf = '';
  return {
    /// 受信チャンク → 完成した event の配列 [{ event, data }]。
    push(chunk) {
      buf += String(chunk == null ? '' : chunk);
      const out = [];
      for (;;) {
        // 空行 (CRLF 可) を区切りとして探す
        const m = /\r?\n\r?\n/.exec(buf);
        if (!m) break;
        const raw = buf.slice(0, m.index);
        buf = buf.slice(m.index + m[0].length);
        const ev = parseSseBlock(raw);
        if (ev && ev.data) out.push(ev);
      }
      return out;
    },
    /// ストリーム終端。区切り無しで残った最後の event を返す。
    flush() {
      const raw = buf;
      buf = '';
      const ev = parseSseBlock(raw);
      return ev && ev.data ? [ev] : [];
    },
    /// 未完成のバッファ (テスト用)。
    pending() {
      return buf;
    },
  };
}

/// 1 ブロック (空行まで) → { event, data }。data 行は \n で連結する。
export function parseSseBlock(raw) {
  let event = 'message';
  const data = [];
  for (const line of String(raw == null ? '' : raw).split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) {
      event = line.slice(6).trim() || 'message';
    } else if (line.startsWith('data:')) {
      let d = line.slice(5);
      if (d.startsWith(' ')) d = d.slice(1);
      data.push(d);
    }
  }
  if (!data.length) return null;
  return { event, data: data.join('\n') };
}

/// SSE event → Anthropic 相当の意味。未知の type は素通しで null を返す。
///   content_block_delta (text_delta) → { text }
///   message_delta                   → { usage, stopReason }
///   error                           → { error }
/// OpenAI 互換 (`choices[0].delta.content`) も受ける。
export function applyChatEvent(st, ev) {
  let j;
  try {
    j = JSON.parse(ev.data);
  } catch (e) {
    return null;
  }
  if (j && j.type === 'content_block_delta' && j.delta && typeof j.delta.text === 'string') {
    return { text: j.delta.text };
  }
  if (j && j.type === 'message_delta') {
    return {
      usage: j.usage || null,
      stopReason: (j.delta && j.delta.stop_reason) || null,
    };
  }
  if (j && j.type === 'error') {
    return { error: (j.error && j.error.message) || 'error' };
  }
  // OpenAI chat.completions のストリーム (中継先が変わっても壊れないように)
  const ch = j && Array.isArray(j.choices) ? j.choices[0] : null;
  if (ch && ch.delta && typeof ch.delta.content === 'string' && ch.delta.content) {
    return { text: ch.delta.content };
  }
  if (j && j.usage && Number.isFinite(Number(j.usage.completion_tokens))) {
    return { usage: { output_tokens: Number(j.usage.completion_tokens) }, stopReason: null };
  }
  return null;
}

/// 分割表示の URL (2D 地図だけを並べる)。
export function splitViewUrl() {
  return SPLIT_2D_URL;
}
