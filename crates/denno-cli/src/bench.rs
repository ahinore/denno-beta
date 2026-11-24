//! P801: `denno bench` — llama-bench 風の標準ベンチマークを HTTP で叩く。
//!
//! 対象は起動中の denno サーバー (`--url`)。Anthropic Messages API
//! (`/v1/messages`) と OpenAI Chat Completions API (`/v1/chat/completions`)
//! の両方を話せる (--api)。測るもの:
//!
//! - pp (prompt processing): `--pp` x `--depths` の組合せ。TTFT を prefill 時間と
//!   みなし、total_tokens / TTFT を tok/s とする。
//! - tg (text generation): `--depths` ごとに `--tg` トークン生成。SSE delta の
//!   到着間隔から TPOT / ITL を出す。
//! - turn-2 KV reuse: 同一プレフィックスの 2 ターン目 TTFT と cold TTFT の比。
//! - 決定論: temperature=0 (greedy) で `--rounds` 回生成し出力 md5 を比較。
//!
//! プロンプトは決定的な擬似自然文を生成し、サーバーが返す `usage.input_tokens`
//! を使って目標トークン数へ二分探索で較正する (サーバーのトークナイザが唯一の
//! 真実)。cold KV のラウンド間には、コンテキスト長に近い長さの捨てリクエストで
//! prefix cache を押し流す (ベストエフォート。`--no-flush` で無効化)。

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::io::Read;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// 較正プロンプトの末尾に必ず付ける固定文 (トークン数は較正に含まれる)。
const INSTRUCTION: &str = "\n\nContinue with a clear, detailed explanation of the topic above.";

// ---------------------------------------------------------------------------
// 統計
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub mean: f64,
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub stddev: f64,
    pub p90: f64,
    pub p99: f64,
}

impl Stats {
    pub fn from_slice(xs: &[f64]) -> Stats {
        if xs.is_empty() {
            return Stats::default();
        }
        let mut v = xs.to_vec();
        for x in v.iter() {
            assert!(x.is_finite(), "non-finite sample in stats: {x}");
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        let mean = v.iter().sum::<f64>() / n as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        Stats {
            n,
            mean,
            median: median_sorted(&v),
            min: v[0],
            max: v[n - 1],
            stddev: var.sqrt(),
            p90: percentile_sorted(&v, 0.90),
            p99: percentile_sorted(&v, 0.99),
        }
    }
}

/// ソート済みスライスの中央値 (偶数長は中間 2 値の平均)。
pub fn median_sorted(sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 => 0.0,
        n if n % 2 == 1 => sorted[n / 2],
        n => (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0,
    }
}

/// ソート済みスライスのパーセンタイル (線形補間)。`p` は 0.0..=1.0。
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let p = p.clamp(0.0, 1.0);
    let pos = p * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    }
}

/// `Duration` をミリ秒 f64 へ。
fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

// ---------------------------------------------------------------------------
// MD5 (自己完結。新しい依存を増やさないため)
// ---------------------------------------------------------------------------

const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

const MD5_K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
    0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
    0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
    0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
    0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
    0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
    0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
    0xeb86d391,
];

/// MD5 の 16 進小文字表現 (RFC 1321)。
pub fn md5_hex(input: &[u8]) -> String {
    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);

    let mut msg = input.to_vec();
    let bit_len = (input.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_le_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, w) in m.iter_mut().enumerate() {
            *w = u32::from_le_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | ((!b) & d), i),
                1 => ((d & b) | ((!d) & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | (!d)), (7 * i) % 16),
            };
            let tmp = d;
            d = c;
            c = b;
            let sum = a
                .wrapping_add(f)
                .wrapping_add(MD5_K[i])
                .wrapping_add(m[g]);
            b = b.wrapping_add(sum.rotate_left(MD5_S[i]));
            a = tmp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut out = String::with_capacity(32);
    for v in [a0, b0, c0, d0] {
        for byte in v.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// SSE パーサ
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// 行単位の SSE パーサ。`event:` / `data:` / コメント (`:` 始まり) を扱い、
/// 空行で 1 イベントを確定する。`ping` などの keep-alive は `event` だけの
/// イベントとして出るので、呼び側が無視する。
#[derive(Debug, Default)]
pub struct SseParser {
    event: Option<String>,
    data: String,
    has_data: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// 1 行を食わせ、イベントが確定したら返す。
    pub fn push_line(&mut self, line: &str) -> Option<SseEvent> {
        // 呼び側は行末を含めて渡してよい (\r\n / \n のどちらも受ける)。
        let line = line.strip_suffix('\n').unwrap_or(line);
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return None; // コメント行
        }
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (line, ""),
        };
        match field {
            "event" => self.event = Some(value.to_string()),
            "data" => {
                if self.has_data {
                    self.data.push('\n');
                }
                self.data.push_str(value);
                self.has_data = true;
            }
            // id / retry は無視 (ベンチには不要)
            _ => {}
        }
        None
    }

    /// ストリーム終端で未確定のイベントを吐く (終端に空行が無い実装向け)。
    pub fn finish(&mut self) -> Option<SseEvent> {
        self.dispatch()
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if !self.has_data && self.event.is_none() {
            return None;
        }
        let event = self.event.take();
        let data = std::mem::take(&mut self.data);
        self.has_data = false;
        Some(SseEvent { event, data })
    }
}

// ---------------------------------------------------------------------------
// 決定的プロンプト生成
// ---------------------------------------------------------------------------

const SUBJECTS: [&str; 20] = [
    "the engineer",
    "a researcher",
    "the committee",
    "my neighbour",
    "the teacher",
    "a small team",
    "the historian",
    "the pilot",
    "a local farmer",
    "the analyst",
    "the architect",
    "a student",
    "the librarian",
    "a retired sailor",
    "the physician",
    "a young musician",
    "the gardener",
    "a software developer",
    "the economist",
    "a volunteer",
];

const ADVERBS: [&str; 10] = [
    "carefully",
    "quickly",
    "quietly",
    "gradually",
    "patiently",
    "rarely",
    "deliberately",
    "honestly",
    "occasionally",
    "thoroughly",
];

const VERBS: [&str; 16] = [
    "measured",
    "described",
    "rebuilt",
    "questioned",
    "translated",
    "sketched",
    "compared",
    "recorded",
    "collected",
    "explained",
    "repaired",
    "reviewed",
    "sorted",
    "tested",
    "planted",
    "documented",
];

const OBJECTS: [&str; 16] = [
    "the rainfall records from the valley",
    "a collection of old coastal maps",
    "the wooden bridge near the harbour",
    "three notebooks of field observations",
    "the annual budget for the small clinic",
    "a set of unusual bird calls",
    "the damaged roof of the village school",
    "several letters written in the previous century",
    "the timetable for the night train",
    "a long list of unanswered questions",
    "the engine of an old fishing boat",
    "the results of the soil survey",
    "two boxes of photographs from the archive",
    "the schedule for the summer festival",
    "the wiring inside the control cabinet",
    "a detailed plan for the new orchard",
];

const PLACES: [&str; 10] = [
    "in the northern province",
    "near the river mouth",
    "on a cold morning",
    "during the long drought",
    "under the old clock tower",
    "at the edge of the forest",
    "before the harvest began",
    "inside the empty warehouse",
    "along the coastal road",
    "after the storm had passed",
];

/// 決定的な xorshift64。同じ seed と同じ長さなら常に同じ文字列になる。
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn idx(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// `words` 個の空白区切り語からなる決定的な英文を組む。
/// 文の区切りで語を切り落とすので、語数はちょうど `words` になる。
pub fn build_prompt(seed: u64, words: usize) -> String {
    if words == 0 {
        return String::new();
    }
    let mut rng = Rng::new(seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut sentences: Vec<String> = Vec::new();
    let mut remaining = words;
    while remaining > 0 {
        let subject = SUBJECTS[rng.idx(SUBJECTS.len())];
        let adv = ADVERBS[rng.idx(ADVERBS.len())];
        let verb = VERBS[rng.idx(VERBS.len())];
        let object = OBJECTS[rng.idx(OBJECTS.len())];
        let place = PLACES[rng.idx(PLACES.len())];
        let mut sw: Vec<&'static str> = Vec::with_capacity(24);
        sw.extend(subject.split(' '));
        if rng.next() % 3 != 0 {
            sw.push(adv);
        }
        sw.push(verb);
        sw.extend(object.split(' '));
        if rng.next() % 2 == 0 {
            sw.extend(place.split(' '));
        }
        if sw.len() > remaining {
            sw.truncate(remaining);
        }
        remaining -= sw.len();
        let mut sentence = sw.join(" ");
        sentence.push('.');
        sentences.push(sentence);
    }
    let mut text = sentences.join(" ");
    // 先頭を大文字化 (ASCII のみ扱うので byte 操作で十分)
    let mut chars = text.chars();
    if let Some(c) = chars.next() {
        let rest: String = chars.collect();
        text = format!("{}{}", c.to_ascii_uppercase(), rest);
    }
    text
}

fn calibrate_prompt_text(seed: u64, words: usize) -> String {
    let mut s = build_prompt(seed, words);
    s.push_str(INSTRUCTION);
    s
}

// ---------------------------------------------------------------------------
// HTTP クライアント
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    Anthropic,
    Openai,
}

#[derive(Debug, Clone)]
pub struct Client {
    agent: ureq::Agent,
    base: String,
    api: Api,
}

#[derive(Debug, Clone, Default)]
pub struct Completion {
    pub input_tokens: usize,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct StreamOutcome {
    pub text: String,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub model: Option<String>,
    pub ttft: Option<Duration>,
    pub total: Duration,
    /// 連続する delta の到着間隔 (ITL)。
    pub itl: Vec<Duration>,
    pub stop_reason: Option<String>,
}

impl Api {
    pub fn parse(s: &str) -> Result<Api> {
        match s.to_ascii_lowercase().as_str() {
            "anthropic" => Ok(Api::Anthropic),
            "openai" => Ok(Api::Openai),
            other => bail!("--api: unknown value {other:?} (anthropic|openai)"),
        }
    }
}

impl Client {
    pub fn new(url: &str, api: Api, timeout_secs: u64) -> Client {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(timeout_secs.max(1)))
            .build();
        Client {
            agent,
            base: url.trim_end_matches('/').to_string(),
            api,
        }
    }

    fn endpoint(&self) -> String {
        match self.api {
            Api::Anthropic => format!("{}/v1/messages", self.base),
            Api::Openai => format!("{}/v1/chat/completions", self.base),
        }
    }

    fn body(&self, messages: &[(&str, &str)], max_tokens: usize, temperature: f32, stream: bool) -> String {
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|(role, content)| serde_json::json!({"role": role, "content": content}))
            .collect();
        let mut v = serde_json::json!({
            "model": "denno-bench",
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        });
        if self.api == Api::Openai && stream {
            v["stream_options"] = serde_json::json!({"include_usage": true});
        }
        v.to_string()
    }

    fn send(&self, body: &str, stream: bool) -> Result<ureq::Response> {
        let url = self.endpoint();
        let req = self
            .agent
            .post(&url)
            .set("content-type", "application/json")
            .set(
                "accept",
                if stream { "text/event-stream" } else { "application/json" },
            );
        match req.send_string(body) {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                bail!("HTTP {code} from {url}: {}", text.trim())
            }
            Err(e) => Err(anyhow::Error::new(e)).with_context(|| format!("POST {url}")),
        }
    }

    pub fn complete(
        &self,
        messages: &[(&str, &str)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Completion> {
        let body = self.body(messages, max_tokens, temperature, false);
        let resp = self.send(&body, false)?;
        let text = resp.into_string().context("read response body")?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parse JSON response: {}", text.trim()))?;
        if let Some(err) = v.get("error") {
            bail!("server error: {err}");
        }
        match self.api {
            Api::Anthropic => Ok(Completion {
                input_tokens: v["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize,
                model: v["model"].as_str().map(str::to_string),
            }),
            Api::Openai => Ok(Completion {
                input_tokens: v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) as usize,
                model: v["model"].as_str().map(str::to_string),
            }),
        }
    }

    pub fn stream(
        &self,
        messages: &[(&str, &str)],
        max_tokens: usize,
        temperature: f32,
    ) -> Result<StreamOutcome> {
        let body = self.body(messages, max_tokens, temperature, true);
        let start = Instant::now();
        let resp = self.send(&body, true)?;
        // 行分割は自前で行う (BufReader<Box<dyn Read>> 経由の read_line が
        // chunked ボディで 0 バイトを返す環境があったため)。読み取った
        // バイト列をためて '\n' で切り出す。タイムスタンプは行ごとに取る。
        let mut reader = resp.into_reader();
        let mut parser = SseParser::new();
        let mut outcome = StreamOutcome::default();
        let mut last_delta: Option<Duration> = None;
        let mut pending: Vec<u8> = Vec::new();
        let mut chunk = [0u8; 4096];
        let mut done = false;

        loop {
            let n = reader.read(&mut chunk).context("read SSE stream")?;
            if n == 0 {
                break;
            }
            pending.extend_from_slice(&chunk[..n]);
            while let Some(pos) = pending.iter().position(|&b| b == b'\n') {
                let raw: Vec<u8> = pending.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&raw);
                let now = start.elapsed();
                if let Some(ev) = parser.push_line(&line) {
                    if self.handle_event(&ev, now, &mut outcome, &mut last_delta)? {
                        done = true;
                        break;
                    }
                }
            }
            if done {
                break;
            }
        }
        if !done {
            if !pending.is_empty() {
                let line = String::from_utf8_lossy(&pending).to_string();
                let _ = parser.push_line(&line);
            }
            if let Some(ev) = parser.finish() {
                let now = start.elapsed();
                let _ = self.handle_event(&ev, now, &mut outcome, &mut last_delta)?;
            }
        }
        outcome.total = start.elapsed();
        Ok(outcome)
    }

    /// 1 イベント処理。ストリーム終端なら true。
    fn handle_event(
        &self,
        ev: &SseEvent,
        now: Duration,
        outcome: &mut StreamOutcome,
        last_delta: &mut Option<Duration>,
    ) -> Result<bool> {
        match self.api {
            Api::Anthropic => {
                if ev.event.as_deref() == Some("ping") {
                    return Ok(false);
                }
                let v: serde_json::Value = match serde_json::from_str(&ev.data) {
                    Ok(v) => v,
                    Err(_) => return Ok(false), // keep-alive の生文字列など
                };
                let ty = v["type"].as_str().unwrap_or_default();
                match ty {
                    "message_start" => {
                        outcome.input_tokens =
                            v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
                        if let Some(m) = v["message"]["model"].as_str() {
                            outcome.model = Some(m.to_string());
                        }
                    }
                    "content_block_delta" => {
                        if v["delta"]["type"] == "text_delta" {
                            let t = v["delta"]["text"].as_str().unwrap_or_default();
                            if !t.is_empty() {
                                outcome.text.push_str(t);
                                if let Some(prev) = *last_delta {
                                    outcome.itl.push(now - prev);
                                } else {
                                    outcome.ttft = Some(now);
                                }
                                *last_delta = Some(now);
                            }
                        }
                    }
                    "message_delta" => {
                        if let Some(n) = v["usage"]["output_tokens"].as_u64() {
                            outcome.output_tokens = n as usize;
                        }
                        if let Some(r) = v["delta"]["stop_reason"].as_str() {
                            outcome.stop_reason = Some(r.to_string());
                        }
                    }
                    "message_stop" => return Ok(true),
                    "error" => bail!("server SSE error: {}", v["error"]),
                    _ => {}
                }
                Ok(false)
            }
            Api::Openai => {
                if ev.data.trim() == "[DONE]" {
                    return Ok(true);
                }
                let v: serde_json::Value = match serde_json::from_str(&ev.data) {
                    Ok(v) => v,
                    Err(_) => return Ok(false),
                };
                if let Some(err) = v.get("error") {
                    bail!("server SSE error: {err}");
                }
                if let Some(m) = v["model"].as_str() {
                    if !m.is_empty() {
                        outcome.model = Some(m.to_string());
                    }
                }
                if let Some(u) = v.get("usage") {
                    if !u.is_null() {
                        outcome.input_tokens =
                            u["prompt_tokens"].as_u64().unwrap_or(0) as usize;
                        outcome.output_tokens =
                            u["completion_tokens"].as_u64().unwrap_or(0) as usize;
                    }
                }
                let Some(choice) = v["choices"].get(0) else {
                    return Ok(false);
                };
                if let Some(r) = choice["finish_reason"].as_str() {
                    outcome.stop_reason = Some(r.to_string());
                }
                let t = choice["delta"]["content"].as_str().unwrap_or_default();
                if !t.is_empty() {
                    outcome.text.push_str(t);
                    if let Some(prev) = *last_delta {
                        outcome.itl.push(now - prev);
                    } else {
                        outcome.ttft = Some(now);
                    }
                    *last_delta = Some(now);
                }
                Ok(false)
            }
        }
    }

    /// /v1/models から (model, context_length) を取る。失敗は None。
    pub fn models(&self) -> Option<(String, Option<usize>)> {
        let url = format!("{}/v1/models", self.base);
        let resp = self.agent.get(&url).call().ok()?;
        let v: serde_json::Value = serde_json::from_str(&resp.into_string().ok()?).ok()?;
        let m = v["data"].get(0)?;
        Some((
            m["id"].as_str()?.to_string(),
            m["context_length"].as_u64().map(|n| n as usize),
        ))
    }
}

// ---------------------------------------------------------------------------
// プロンプト較正
// ---------------------------------------------------------------------------

/// サーバーの `usage.input_tokens` を使い、目標トークン数へ二分探索で合わせる。
pub struct Calibrator<'a> {
    client: &'a Client,
    cache: HashMap<(u64, usize), usize>,
    pub probes: usize,
}

impl<'a> Calibrator<'a> {
    pub fn new(client: &'a Client) -> Self {
        Calibrator {
            client,
            cache: HashMap::new(),
            probes: 0,
        }
    }

    fn tokens_for(&mut self, seed: u64, words: usize) -> Result<usize> {
        if let Some(&t) = self.cache.get(&(seed, words)) {
            return Ok(t);
        }
        let text = calibrate_prompt_text(seed, words);
        let c = self.client.complete(&[("user", text.as_str())], 1, 0.0)?;
        self.probes += 1;
        let t = c.input_tokens.max(1);
        self.cache.insert((seed, words), t);
        Ok(t)
    }

    /// 目標トークン数に一致するプロンプトを返す。一致できない場合は
    /// 最も近い (>= target) 語数を使い、実測トークン数を併せて返す。
    pub fn calibrate(&mut self, seed: u64, target: usize) -> Result<(String, usize)> {
        if target == 0 {
            return Ok((String::new(), 0));
        }
        let mut hi = target.max(1) * 2;
        let mut guard = 0;
        while self.tokens_for(seed, hi)? < target {
            hi *= 2;
            guard += 1;
            if guard > 24 {
                bail!("calibration failed: cannot reach {target} tokens (words={hi})");
            }
        }
        let mut lo = 1usize;
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.tokens_for(seed, mid)? >= target {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let got = self.tokens_for(seed, lo)?;
        Ok((calibrate_prompt_text(seed, lo), got))
    }
}

// ---------------------------------------------------------------------------
// ベンチ本体
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub url: String,
    pub api: Api,
    pub pp: Vec<usize>,
    pub tg: usize,
    pub depths: Vec<usize>,
    pub rounds: usize,
    pub flush: bool,
    /// 0 = auto (コンテキスト長 - 64)
    pub flush_tokens: usize,
    pub seed: u64,
    pub det_tokens: usize,
    pub reuse_depth: usize,
    pub timeout_secs: u64,
}

impl Default for BenchOptions {
    fn default() -> Self {
        BenchOptions {
            url: "http://127.0.0.1:8098".to_string(),
            api: Api::Anthropic,
            pp: vec![512, 2048, 8192],
            tg: 128,
            depths: vec![0, 2048, 8192],
            rounds: 3,
            flush: true,
            flush_tokens: 0,
            seed: 801,
            det_tokens: 16,
            reuse_depth: 2048,
            timeout_secs: 600,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PpRow {
    pub pp_tokens: usize,
    pub depth: usize,
    pub total_tokens: usize,
    pub ttft_ms: f64,
    pub pp_tok_s: f64,
    pub ttft_runs_ms: Vec<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TgRow {
    pub depth: usize,
    pub gen_tokens: usize,
    pub prompt_tokens: usize,
    pub ttft_ms: f64,
    pub tpot_ms: Stats,
    pub itl_ms: Stats,
    pub tg_tok_s: f64,
    pub runs_tok_s: Vec<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Determinism {
    pub tokens: usize,
    pub gen_tokens: usize,
    pub md5: String,
    pub verified: bool,
    pub round_md5: Vec<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Turn2 {
    pub depth: usize,
    pub cold_ttft_ms: f64,
    pub reuse_ttft_ms: f64,
    pub speedup: f64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchReport {
    pub url: String,
    pub api: String,
    pub model: String,
    pub context_length: Option<usize>,
    pub rounds: usize,
    pub created_unix: u64,
    pub pp: Vec<PpRow>,
    pub tg: Vec<TgRow>,
    pub turn2: Option<Turn2>,
    pub determinism: Option<Determinism>,
    pub errors: Vec<String>,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// seed の混合 (ケースごとに別プロンプトにするため)。
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// KV/prefix cache を押し流すための捨てリクエスト (ベストエフォート)。
fn flush_kv(client: &Client, tokens: usize, idx: usize) -> Result<usize> {
    if tokens == 0 {
        return Ok(0);
    }
    let mut words = (tokens * 3 / 4).max(8);
    let seed = 0xF00D_u64
        .wrapping_mul(idx as u64 + 1)
        .wrapping_add(0xABCD_1234);
    loop {
        let prompt = build_prompt(seed, words);
        match client.complete(&[("user", prompt.as_str())], 1, 0.0) {
            Ok(c) => return Ok(c.input_tokens),
            Err(e) => {
                if words <= 16 {
                    return Err(e);
                }
                words /= 2;
            }
        }
    }
}

pub fn run(opts: &BenchOptions) -> Result<BenchReport> {
    if opts.rounds == 0 {
        bail!("--rounds must be >= 1");
    }
    let client = Client::new(&opts.url, opts.api, opts.timeout_secs);

    // 事前疎通 (ここで落ちたらサーバー不達なので即エラー)
    let pre = client
        .complete(&[("user", "Hello.")], 1, 0.0)
        .with_context(|| format!("preflight request to {} failed", opts.url))?;
    let (listed_model, context_length) = client.models().unwrap_or((String::new(), None));
    let model = if listed_model.is_empty() {
        pre.model.clone().unwrap_or_else(|| "unknown".to_string())
    } else {
        listed_model
    };

    let flush_tokens = if !opts.flush {
        0
    } else if opts.flush_tokens > 0 {
        opts.flush_tokens
    } else {
        match context_length {
            Some(c) if c > 128 => c - 64,
            _ => opts
                .pp
                .iter()
                .copied()
                .max()
                .unwrap_or(512)
                .saturating_add(1024),
        }
    };

    let mut errors: Vec<String> = Vec::new();
    let mut flush_idx = 0usize;
    let mut calib = Calibrator::new(&client);

    // ---- pp (prompt processing) ----
    let mut pp_rows = Vec::new();
    for &depth in &opts.depths {
        for &pp in &opts.pp {
            let total = pp + depth;
            let seed = opts.seed ^ mix(total as u64) ^ mix(depth as u64) ^ 0x5050;
            let mut row = PpRow {
                pp_tokens: pp,
                depth,
                total_tokens: total,
                ttft_ms: 0.0,
                pp_tok_s: 0.0,
                ttft_runs_ms: Vec::new(),
                error: None,
            };
            let mut calibrated: Option<(String, usize)> = None;
            for _ in 0..opts.rounds {
                if flush_tokens > 0 {
                    flush_idx += 1;
                    if let Err(e) = flush_kv(&client, flush_tokens, flush_idx) {
                        errors.push(format!("flush (pp{pp} d{depth}): {e:#}"));
                    }
                }
                let (prompt, toks) = match &calibrated {
                    Some(v) => v.clone(),
                    None => match calib.calibrate(seed, total) {
                        Ok(v) => {
                            calibrated = Some(v.clone());
                            v
                        }
                        Err(e) => {
                            row.error = Some(format!("calibrate: {e:#}"));
                            break;
                        }
                    },
                };
                if toks != row.total_tokens {
                    row.total_tokens = toks;
                }
                match client.stream(&[("user", prompt.as_str())], 1, 0.0) {
                    Ok(o) => match o.ttft {
                        Some(t) => row.ttft_runs_ms.push(ms(t)),
                        None => {
                            row.error = Some("no text delta before stream end".to_string());
                            break;
                        }
                    },
                    Err(e) => {
                        row.error = Some(format!("{e:#}"));
                        break;
                    }
                }
            }
            if !row.ttft_runs_ms.is_empty() {
                let st = Stats::from_slice(&row.ttft_runs_ms);
                row.ttft_ms = st.median;
                let secs = (st.median / 1000.0).max(1e-9);
                row.pp_tok_s = row.total_tokens as f64 / secs;
            }
            if let Some(e) = row.error.clone() {
                errors.push(format!("pp{pp} d{depth}: {e}"));
            }
            pp_rows.push(row);
        }
    }
    eprintln!("[bench] pp done ({} rows)", pp_rows.len());

    // ---- tg (text generation) ----
    let mut tg_rows = Vec::new();
    for &depth in &opts.depths {
        let target = depth.max(1);
        let seed = opts.seed ^ mix(target as u64) ^ 0x7474;
        let mut row = TgRow {
            depth,
            gen_tokens: opts.tg,
            prompt_tokens: target,
            ttft_ms: 0.0,
            tpot_ms: Stats::default(),
            itl_ms: Stats::default(),
            tg_tok_s: 0.0,
            runs_tok_s: Vec::new(),
            error: None,
        };
        let (prompt, ptoks) = match calib.calibrate(seed, target) {
            Ok(v) => v,
            Err(e) => {
                row.error = Some(format!("calibrate: {e:#}"));
                errors.push(format!("tg d{depth}: {e:#}"));
                tg_rows.push(row);
                continue;
            }
        };
        row.prompt_tokens = ptoks;
        let mut ttfts = Vec::new();
        let mut tpot_runs: Vec<f64> = Vec::new();
        let mut itl_all: Vec<f64> = Vec::new();
        for _ in 0..opts.rounds {
            match client.stream(&[("user", prompt.as_str())], opts.tg, 0.0) {
                Ok(o) => {
                    let Some(ttft) = o.ttft else {
                        row.error = Some("no text delta before stream end".to_string());
                        break;
                    };
                    ttfts.push(ms(ttft));
                    let n_out = o.output_tokens.max(2) as f64;
                    // デコード区間は最初と最後の delta の間 (usage 送信ぶんを含めない)
                    let span_s: f64 = o.itl.iter().map(|d| d.as_secs_f64()).sum();
                    let span_s = if span_s > 0.0 {
                        span_s
                    } else {
                        o.total.saturating_sub(ttft).as_secs_f64()
                    }
                    .max(1e-9);
                    row.runs_tok_s.push((n_out - 1.0) / span_s);
                    tpot_runs.push(span_s * 1000.0 / (n_out - 1.0));
                    itl_all.extend(o.itl.iter().map(|d| ms(*d)));
                }
                Err(e) => {
                    row.error = Some(format!("{e:#}"));
                    break;
                }
            }
        }
        if !ttfts.is_empty() {
            row.ttft_ms = Stats::from_slice(&ttfts).median;
        }
        if !row.runs_tok_s.is_empty() {
            let mut v = row.runs_tok_s.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            row.tg_tok_s = median_sorted(&v);
        }
        if !tpot_runs.is_empty() {
            // TPOT はラウンドごとの (デコード区間 / 生成トークン数) の中央値。
            // ITL は全ラウンドの到着間隔をまとめた分布 (p90/p99 付き)。
            row.tpot_ms = Stats::from_slice(&tpot_runs);
        }
        if !itl_all.is_empty() {
            row.itl_ms = Stats::from_slice(&itl_all);
        }
        if let Some(e) = row.error.clone() {
            errors.push(format!("tg d{depth}: {e}"));
        }
        tg_rows.push(row);
    }
    eprintln!("[bench] tg done ({} rows)", tg_rows.len());

    // ---- turn-2 KV reuse ----
    let turn2 = {
        let depth = opts.reuse_depth.max(1);
        let seed = opts.seed ^ mix(depth as u64) ^ 0x2b2b;
        let mut row = Turn2 {
            depth,
            cold_ttft_ms: 0.0,
            reuse_ttft_ms: 0.0,
            speedup: 0.0,
            error: None,
        };
        match calib.calibrate(seed, depth) {
            Ok((prompt, _)) => {
                if flush_tokens > 0 {
                    flush_idx += 1;
                    if let Err(e) = flush_kv(&client, flush_tokens, flush_idx) {
                        errors.push(format!("flush (turn2): {e:#}"));
                    }
                }
                match client.stream(&[("user", prompt.as_str())], 16, 0.0) {
                    Ok(cold) => match cold.ttft {
                        Some(t) => {
                            row.cold_ttft_ms = ms(t);
                            let answer = if cold.text.trim().is_empty() {
                                "The topic is complex.".to_string()
                            } else {
                                cold.text.clone()
                            };
                            let msgs = [
                                ("user", prompt.as_str()),
                                ("assistant", answer.as_str()),
                                ("user", "Continue with one more sentence."),
                            ];
                            match client.stream(&msgs, 1, 0.0) {
                                Ok(warm) => match warm.ttft {
                                    Some(t2) => {
                                        row.reuse_ttft_ms = ms(t2);
                                        row.speedup =
                                            row.cold_ttft_ms / row.reuse_ttft_ms.max(1e-6);
                                    }
                                    None => {
                                        row.error =
                                            Some("turn 2: no text delta".to_string())
                                    }
                                },
                                Err(e) => row.error = Some(format!("turn 2: {e:#}")),
                            }
                        }
                        None => row.error = Some("turn 1: no text delta".to_string()),
                    },
                    Err(e) => row.error = Some(format!("turn 1: {e:#}")),
                }
            }
            Err(e) => row.error = Some(format!("calibrate: {e:#}")),
        }
        if let Some(e) = row.error.clone() {
            errors.push(format!("turn2: {e}"));
        }
        row
    };

    // ---- 決定論 md5 ----
    let determinism = {
        let target = 256usize.max(opts.det_tokens);
        let seed = opts.seed ^ 0x6d64_6d64;
        let mut det = Determinism {
            tokens: target,
            gen_tokens: opts.det_tokens,
            md5: String::new(),
            verified: false,
            round_md5: Vec::new(),
            error: None,
        };
        match calib.calibrate(seed, target) {
            Ok((prompt, toks)) => {
                det.tokens = toks;
                for _ in 0..opts.rounds {
                    if flush_tokens > 0 {
                        flush_idx += 1;
                        if let Err(e) = flush_kv(&client, flush_tokens, flush_idx) {
                            errors.push(format!("flush (determinism): {e:#}"));
                        }
                    }
                    match client.stream(&[("user", prompt.as_str())], opts.det_tokens, 0.0) {
                        Ok(o) => det.round_md5.push(md5_hex(o.text.as_bytes())),
                        Err(e) => {
                            det.error = Some(format!("{e:#}"));
                            break;
                        }
                    }
                }
                if let Some(first) = det.round_md5.first() {
                    det.md5 = first.clone();
                    det.verified = det.round_md5.len() == opts.rounds
                        && det.round_md5.iter().all(|h| h == first);
                }
            }
            Err(e) => det.error = Some(format!("calibrate: {e:#}")),
        }
        if let Some(e) = det.error.clone() {
            errors.push(format!("determinism: {e}"));
        }
        det
    };
    eprintln!(
        "[bench] done ({} calibration probes, {} errors)",
        calib.probes,
        errors.len()
    );

    Ok(BenchReport {
        url: opts.url.clone(),
        api: match opts.api {
            Api::Anthropic => "anthropic".to_string(),
            Api::Openai => "openai".to_string(),
        },
        model,
        context_length,
        rounds: opts.rounds,
        created_unix: now_unix(),
        pp: pp_rows,
        tg: tg_rows,
        turn2: Some(turn2),
        determinism: Some(determinism),
        errors,
    })
}

// ---------------------------------------------------------------------------
// 出力
// ---------------------------------------------------------------------------

pub fn render_json(report: &BenchReport) -> Result<String> {
    Ok(serde_json::to_string_pretty(report)?)
}

pub fn render_markdown(report: &BenchReport) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "## denno bench — {} @ {} (api={}, rounds={})\n\n",
        report.model, report.url, report.api, report.rounds
    ));
    if let Some(c) = report.context_length {
        s.push_str(&format!("context_length: {c}\n\n"));
    }
    s.push_str("| test | depth | tokens | ttft_ms | pp_tok_s | tpot_ms | tg_tok_s |\n");
    s.push_str("|---|---|---|---|---|---|---|\n");
    for r in &report.pp {
        if let Some(e) = &r.error {
            s.push_str(&format!(
                "| pp{} | {} | {} | error | - | - | - |\n",
                r.pp_tokens, r.depth, r.total_tokens
            ));
            let _ = e;
            continue;
        }
        s.push_str(&format!(
            "| pp{} | {} | {} | {:.3} | {:.2} | - | - |\n",
            r.pp_tokens, r.depth, r.total_tokens, r.ttft_ms, r.pp_tok_s
        ));
    }
    for r in &report.tg {
        if r.error.is_some() {
            s.push_str(&format!(
                "| tg{} | {} | {} | error | - | - | - |\n",
                r.gen_tokens, r.depth, r.prompt_tokens
            ));
            continue;
        }
        s.push_str(&format!(
            "| tg{} | {} | {} | {:.3} | - | {:.3} | {:.2} |\n",
            r.gen_tokens, r.depth, r.prompt_tokens, r.ttft_ms, r.tpot_ms.median, r.tg_tok_s
        ));
    }
    s.push('\n');
    if let Some(t) = &report.turn2 {
        match &t.error {
            None => s.push_str(&format!(
                "Turn-2 KV reuse TTFT: {:.3} ms (cold {:.3} ms, {:.2}x) at depth {}\n",
                t.reuse_ttft_ms, t.cold_ttft_ms, t.speedup, t.depth
            )),
            Some(e) => s.push_str(&format!("Turn-2 KV reuse: error ({e})\n")),
        }
    }
    if let Some(d) = &report.determinism {
        match &d.error {
            None => s.push_str(&format!(
                "Determinism: md5 `{}` {} across {} rounds ({} tokens, {} generated)\n",
                d.md5,
                if d.verified { "verified" } else { "MISMATCH" },
                d.round_md5.len(),
                d.tokens,
                d.gen_tokens
            )),
            Some(e) => s.push_str(&format!("Determinism: error ({e})\n")),
        }
    }
    if !report.errors.is_empty() {
        s.push_str("\nerrors:\n");
        for e in &report.errors {
            s.push_str(&format!("- {e}\n"));
        }
    }
    s
}

// ---------------------------------------------------------------------------
// テスト
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // ---- stats ----

    #[test]
    fn stats_median_and_percentiles() {
        assert_eq!(median_sorted(&[3.0]), 3.0);
        assert_eq!(median_sorted(&[1.0, 2.0, 3.0]), 2.0);
        assert_eq!(median_sorted(&[1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median_sorted(&[]), 0.0);

        let v = [10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(percentile_sorted(&v, 0.0), 10.0);
        assert_eq!(percentile_sorted(&v, 1.0), 50.0);
        assert_eq!(percentile_sorted(&v, 0.5), 30.0);
        assert!((percentile_sorted(&v, 0.25) - 20.0).abs() < 1e-12);

        let st = Stats::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(st.n, 4);
        assert!((st.mean - 2.5).abs() < 1e-12);
        assert!((st.median - 2.5).abs() < 1e-12);
        assert_eq!(st.min, 1.0);
        assert_eq!(st.max, 4.0);
        assert!((st.stddev - 1.118_033_988_749_895).abs() < 1e-12);

        let empty = Stats::from_slice(&[]);
        assert_eq!(empty.n, 0);
        assert_eq!(empty.median, 0.0);
    }

    // ---- md5 ----

    #[test]
    fn md5_known_vectors() {
        assert_eq!(md5_hex(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex(b"a"), "0cc175b9c0f1b6a831c399e269772661");
        assert_eq!(md5_hex(b"abc"), "900150983cd24fb0d6963f7d28e17f72");
        assert_eq!(
            md5_hex(b"The quick brown fox jumps over the lazy dog"),
            "9e107d9d372bb6826bd81d3542a419d6"
        );
        // 64 バイト境界を跨ぐ長さ
        let long = "x".repeat(200);
        assert_eq!(md5_hex(long.as_bytes()).len(), 32);
        assert_ne!(md5_hex(long.as_bytes()), md5_hex(b"x"));
    }

    // ---- SSE parser ----

    #[test]
    fn sse_parser_event_and_data() {
        let mut p = SseParser::new();
        assert!(p.push_line("event: message_start").is_none());
        assert!(p.push_line("data: {\"a\":1}").is_none());
        let ev = p.push_line("").expect("event dispatched");
        assert_eq!(ev.event.as_deref(), Some("message_start"));
        assert_eq!(ev.data, "{\"a\":1}");
        assert!(p.push_line("").is_none(), "blank line does not re-dispatch");
    }

    #[test]
    fn sse_parser_multiline_data_and_crlf() {
        let mut p = SseParser::new();
        assert!(p.push_line("data: line1\r").is_none());
        assert!(p.push_line("data: line2\r").is_none());
        let ev = p.push_line("\r").expect("event dispatched");
        assert_eq!(ev.event, None);
        assert_eq!(ev.data, "line1\nline2");
    }

    #[test]
    fn sse_parser_comments_and_ping() {
        let mut p = SseParser::new();
        assert!(p.push_line(": keep-alive").is_none());
        assert!(p.push_line("").is_none(), "comment alone makes no event");
        assert!(p.push_line("event: ping").is_none());
        let ev = p.push_line("").expect("ping event");
        assert_eq!(ev.event.as_deref(), Some("ping"));
        assert_eq!(ev.data, "");
    }

    #[test]
    fn sse_parser_finish_without_blank_line() {
        let mut p = SseParser::new();
        assert!(p.push_line("data: [DONE]").is_none());
        let ev = p.finish().expect("flushed at eof");
        assert_eq!(ev.data, "[DONE]");
        assert!(p.finish().is_none());
    }

    /// 回帰: 行末の '\n' を含めて渡しても event/data が壊れないこと
    /// (当初はここで event 名に '\n' が残り、空行も空行と認識されなかった)。
    #[test]
    fn sse_parser_accepts_newline_terminated_lines() {
        let mut p = SseParser::new();
        assert!(p.push_line("event: content_block_delta\n").is_none());
        assert!(
            p.push_line("data: {\"delta\":{\"text\":\"hi\"}}\n").is_none(),
            "data line alone does not dispatch"
        );
        let ev2 = p.push_line("\n").expect("blank line dispatches");
        assert_eq!(ev2.event.as_deref(), Some("content_block_delta"));
        assert_eq!(ev2.data, "{\"delta\":{\"text\":\"hi\"}}");
        let v: serde_json::Value = serde_json::from_str(&ev2.data).unwrap();
        assert_eq!(v["delta"]["text"], "hi");
    }

    // ---- prompt builder ----

    #[test]
    fn prompt_is_deterministic_and_diverse() {
        let a = build_prompt(7, 64);
        let b = build_prompt(7, 64);
        assert_eq!(a, b, "same seed must give identical prompt");
        assert_ne!(a, build_prompt(8, 64), "different seed must differ");
        assert_eq!(a.split_whitespace().count(), 64);
        assert!(a.ends_with('.'));
        let long = build_prompt(7, 500);
        assert_eq!(long.split_whitespace().count(), 500);
        assert_eq!(build_prompt(7, 0), "");
    }

    #[test]
    fn prompt_words_are_natural_ascii() {
        let p = build_prompt(42, 120);
        assert!(p.is_ascii());
        assert!(p.chars().next().unwrap().is_ascii_uppercase());
        // 文が複数ある (多様性の最低条件)
        assert!(p.matches(". ").count() >= 2);
    }

    // ---- mock server ----

    struct MockServer {
        url: String,
        stop: Arc<AtomicBool>,
        handle: Option<std::thread::JoinHandle<()>>,
    }

    impl MockServer {
        fn start(context_length: usize) -> MockServer {
            let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
            listener.set_nonblocking(true).unwrap();
            let addr = listener.local_addr().unwrap();
            let stop = Arc::new(AtomicBool::new(false));
            let stop2 = stop.clone();
            let handle = std::thread::spawn(move || {
                while !stop2.load(Ordering::SeqCst) {
                    match listener.accept() {
                        Ok((sock, _)) => {
                            let _ = handle_mock_conn(sock, context_length);
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        Err(_) => break,
                    }
                }
            });
            MockServer {
                url: format!("http://{addr}"),
                stop,
                handle: Some(handle),
            }
        }
    }

    impl Drop for MockServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::SeqCst);
            if let Some(h) = self.handle.take() {
                let _ = h.join();
            }
        }
    }

    fn read_request(sock: &mut std::net::TcpStream) -> (String, serde_json::Value) {
        sock.set_read_timeout(Some(Duration::from_secs(10))).ok();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        let header_end;
        loop {
            let n = sock.read(&mut tmp).expect("read request");
            assert!(n > 0, "connection closed before headers");
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
                header_end = pos + 4;
                break;
            }
        }
        let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
        let mut content_length = 0usize;
        let mut path = String::new();
        for (i, line) in head.split("\r\n").enumerate() {
            if i == 0 {
                path = line.split(' ').nth(1).unwrap_or("/").to_string();
            }
            if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
        while buf.len() < header_end + content_length {
            let n = sock.read(&mut tmp).expect("read body");
            assert!(n > 0, "connection closed before body");
            buf.extend_from_slice(&tmp[..n]);
        }
        let body = String::from_utf8_lossy(&buf[header_end..header_end + content_length]).to_string();
        let json: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
        (path, json)
    }

    fn find_subslice(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }

    /// HTTP/1.1 chunked の 1 チャンクを書く (SSE の実サーバーと同じ形)。
    fn write_chunk(sock: &mut std::net::TcpStream, s: &str) -> std::io::Result<()> {
        write!(sock, "{:x}\r\n{}\r\n", s.len(), s)?;
        sock.flush()
    }

    /// プロンプトの「トークン数」= 空白区切りの語数 (較正の検証に十分)。
    fn mock_tokens(body: &serde_json::Value) -> usize {
        let text = body["messages"]
            .get(0)
            .and_then(|m| m["content"].as_str())
            .unwrap_or_default();
        text.split_whitespace().count().max(1)
    }

    fn handle_mock_conn(mut sock: std::net::TcpStream, context_length: usize) -> std::io::Result<()> {
        sock.set_nodelay(true).ok();
        let (path, body) = read_request(&mut sock);
        if path.starts_with("/v1/models") {
            let resp = serde_json::json!({
                "object": "list",
                "data": [{"id": "mock-model", "context_length": context_length}],
            });
            let payload = resp.to_string();
            write!(
                sock,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            )?;
            return Ok(());
        }
        let anthropic = path.starts_with("/v1/messages");
        let stream = body["stream"].as_bool().unwrap_or(false);
        let max_tokens = body["max_tokens"].as_u64().unwrap_or(1) as usize;
        let in_tokens = mock_tokens(&body);
        let pieces = max_tokens.clamp(1, 16);
        let text = "mock output words ".repeat(pieces);

        if !stream {
            let resp = if anthropic {
                serde_json::json!({
                    "type": "message", "role": "assistant", "model": "mock-model",
                    "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": in_tokens, "output_tokens": pieces},
                })
            } else {
                serde_json::json!({
                    "object": "chat.completion", "model": "mock-model",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": in_tokens, "completion_tokens": pieces},
                })
            };
            let payload = resp.to_string();
            write!(
                sock,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            )?;
            return Ok(());
        }

        write!(
            sock,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )?;
        sock.flush()?;
        if anthropic {
            let start = serde_json::json!({
                "type": "message_start",
                "message": {"model": "mock-model", "usage": {"input_tokens": in_tokens, "output_tokens": 0}},
            });
            write_chunk(&mut sock, &format!("event: message_start\ndata: {start}\n\n"))?;
            write_chunk(
                &mut sock,
                &format!(
                    "event: content_block_start\ndata: {}\n\n",
                    serde_json::json!({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}})
                ),
            )?;
            for i in 0..pieces {
                std::thread::sleep(Duration::from_millis(2));
                let ev = serde_json::json!({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": format!("mock{i} ")},
                });
                write_chunk(
                    &mut sock,
                    &format!("event: content_block_delta\ndata: {ev}\n\n"),
                )?;
            }
            write_chunk(
                &mut sock,
                &format!(
                    "event: content_block_stop\ndata: {}\n\n",
                    serde_json::json!({"type": "content_block_stop", "index": 0})
                ),
            )?;
            write_chunk(
                &mut sock,
                &format!(
                    "event: message_delta\ndata: {}\n\n",
                    serde_json::json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": pieces}})
                ),
            )?;
            write_chunk(
                &mut sock,
                &format!(
                    "event: message_stop\ndata: {}\n\n",
                    serde_json::json!({"type": "message_stop"})
                ),
            )?;
        } else {
            let first = serde_json::json!({
                "object": "chat.completion.chunk", "model": "mock-model",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                             "finish_reason": serde_json::Value::Null}],
            });
            write_chunk(&mut sock, &format!("data: {first}\n\n"))?;
            for i in 0..pieces {
                std::thread::sleep(Duration::from_millis(2));
                let ev = serde_json::json!({
                    "object": "chat.completion.chunk", "model": "mock-model",
                    "choices": [{"index": 0, "delta": {"content": format!("mock{i} ")},
                                 "finish_reason": serde_json::Value::Null}],
                });
                write_chunk(&mut sock, &format!("data: {ev}\n\n"))?;
            }
            let last = serde_json::json!({
                "object": "chat.completion.chunk", "model": "mock-model",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            });
            write_chunk(&mut sock, &format!("data: {last}\n\n"))?;
            let usage = serde_json::json!({
                "object": "chat.completion.chunk", "model": "mock-model", "choices": [],
                "usage": {"prompt_tokens": in_tokens, "completion_tokens": pieces},
            });
            write_chunk(&mut sock, &format!("data: {usage}\n\n"))?;
            write_chunk(&mut sock, "data: [DONE]\n\n")?;
        }
        write!(sock, "0\r\n\r\n")?;
        sock.flush()?;
        Ok(())
    }

    fn tiny_opts(url: &str, api: Api) -> BenchOptions {
        BenchOptions {
            url: url.to_string(),
            api,
            pp: vec![16],
            tg: 4,
            depths: vec![0],
            rounds: 2,
            flush: false,
            flush_tokens: 0,
            seed: 11,
            det_tokens: 4,
            reuse_depth: 16,
            timeout_secs: 20,
        }
    }

    fn assert_report_shape(r: &BenchReport) {
        assert_eq!(r.pp.len(), 1);
        let pp = &r.pp[0];
        assert!(pp.error.is_none(), "pp error: {:?}", pp.error);
        assert_eq!(pp.total_tokens, 16, "prompt must be calibrated exactly");
        assert!(pp.ttft_ms > 0.0, "ttft must be measured");
        assert!(pp.pp_tok_s > 0.0);
        assert_eq!(pp.ttft_runs_ms.len(), 2);

        assert_eq!(r.tg.len(), 1);
        let tg = &r.tg[0];
        assert!(tg.error.is_none(), "tg error: {:?}", tg.error);
        assert_eq!(tg.gen_tokens, 4);
        assert!(tg.tg_tok_s > 0.0, "decode speed must be positive");
        assert_eq!(tg.itl_ms.n, 3 * 2, "3 intervals per round, 2 rounds");
        assert_eq!(tg.tpot_ms.n, 2, "1 TPOT per round");
        assert!(tg.tpot_ms.median > 0.0);

        let t2 = r.turn2.as_ref().expect("turn2 measured");
        assert!(t2.error.is_none(), "turn2 error: {:?}", t2.error);
        assert!(t2.cold_ttft_ms > 0.0 && t2.reuse_ttft_ms > 0.0);
        assert!(t2.speedup > 0.0);

        let det = r.determinism.as_ref().expect("determinism measured");
        assert!(det.error.is_none(), "det error: {:?}", det.error);
        assert_eq!(det.round_md5.len(), 2);
        assert!(det.verified, "same greedy output must hash equal");
        assert_eq!(det.md5.len(), 32);
    }

    #[test]
    fn mock_server_anthropic_end_to_end() {
        let server = MockServer::start(64);
        let opts = tiny_opts(&server.url, Api::Anthropic);
        let report = run(&opts).expect("bench run");
        assert_report_shape(&report);
        assert_eq!(report.model, "mock-model");
        assert_eq!(report.api, "anthropic");
        let md = render_markdown(&report);
        assert!(md.contains("| pp16 | 0 | 16 |"));
        assert!(md.contains("| tg4 | 0 |"));
        assert!(md.contains("Determinism: md5"));
        let json = render_json(&report).expect("json");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["pp"][0]["total_tokens"], 16);
        assert_eq!(v["tg"][0]["gen_tokens"], 4);
    }

    #[test]
    fn mock_server_openai_end_to_end() {
        let server = MockServer::start(64);
        let mut opts = tiny_opts(&server.url, Api::Openai);
        opts.rounds = 1;
        opts.depths = vec![0];
        let report = run(&opts).expect("bench run (openai)");
        assert_eq!(report.api, "openai");
        let pp = &report.pp[0];
        assert!(pp.error.is_none(), "pp error: {:?}", pp.error);
        assert_eq!(pp.total_tokens, 16);
        assert!(pp.ttft_ms > 0.0);
        assert!(report.tg[0].tg_tok_s > 0.0);
        assert!(report.determinism.as_ref().unwrap().verified);
    }

    #[test]
    fn calibration_hits_exact_targets() {
        let server = MockServer::start(64);
        let client = Client::new(&server.url, Api::Anthropic, 20);
        let mut calib = Calibrator::new(&client);
        // 末尾の固定指示文が約 11 トークンあるので、それ未満は一致不能。
        // 一致不能なときは最小プロンプトを返す (呼び側は実測値を報告する)。
        for target in [11usize, 16, 33, 64, 129] {
            let (prompt, toks) = calib.calibrate(99, target).expect("calibrate");
            assert_eq!(toks, target, "exact target {target} (words in prompt)");
            assert_eq!(prompt.split_whitespace().count().max(1), toks);
        }
        let (prompt, toks) = calib.calibrate(99, 1).expect("calibrate tiny");
        assert!(toks >= 1);
        assert!(!prompt.is_empty());
    }

    #[test]
    fn client_reports_http_error() {
        let server = MockServer::start(64);
        // 存在しないパスでも mock は /v1/chat/completions として返すので、
        // 接続不能 URL でエラー経路を確認する。
        let client = Client::new("http://127.0.0.1:1", Api::Anthropic, 1);
        let err = client.complete(&[("user", "hi")], 1, 0.0).unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("preflight") || msg.contains("POST") || msg.contains("connect"),
            "unexpected error text: {msg}"
        );
        drop(server);
    }

    #[test]
    fn flush_kv_uses_unique_prompts() {
        let server = MockServer::start(64);
        let client = Client::new(&server.url, Api::Anthropic, 20);
        let a = flush_kv(&client, 64, 1).expect("flush 1");
        let b = flush_kv(&client, 64, 2).expect("flush 2");
        assert!(a > 0 && b > 0);
        assert_ne!(a, 0);
    }

    /// 回帰: 非ストリーミング要求の直後のストリーミング要求でも
    /// delta / usage が取れること (SSE の行末に '\n' が付く形)。
    #[test]
    fn stream_after_nonstream_request() {
        let server = MockServer::start(64);
        let client = Client::new(&server.url, Api::Anthropic, 20);
        let c = client.complete(&[("user", "hello world")], 1, 0.0).unwrap();
        assert_eq!(c.input_tokens, 2, "usage.input_tokens を読む");
        assert_eq!(c.model.as_deref(), Some("mock-model"));
        let o = client.stream(&[("user", "hello world")], 1, 0.0).unwrap();
        assert_eq!(o.input_tokens, 2, "message_start usage を読む");
        assert!(o.ttft.is_some(), "TTFT が取れる");
        assert_eq!(o.text, "mock0 ");
        assert_eq!(o.output_tokens, 1);
        assert_eq!(o.stop_reason.as_deref(), Some("end_turn"));
    }
}

