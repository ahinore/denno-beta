//! task 861: `denno chat` — 起動中の denno サーバーと話す対話 REPL。
//!
//! `--api anthropic` は `/v1/messages`、`--api openai` は
//! `/v1/chat/completions` を SSE (`stream: true`) で叩き、届いた text delta を
//! その場で stdout へ書く (バッファに貯めて最後に出すことはしない)。
//! REPL のプロンプト・ヘルプ・統計は stderr に出すので、stdout は
//! アシスタントのトークンだけになる (パイプで使いやすい)。
//!
//! スラッシュコマンド: `/help` `/reset` `/stats` `/save <file.json>`
//! `/load <file.json>` `/steer` `/exit`。`/steer` はサーバーの `GET /v1/steer`
//! (現在記録中/適用中のステアリングの状態) をそのまま表示する。
//!
//! モデル名は `--model` で上書きできる。未指定なら `GET /v1/models` の先頭 id を
//! 使う (Anthropic 経路は model 名の一致を要求するため)。解決に失敗しても
//! 空文字で送る (OpenAI 経路は空を許す)。

use crate::bench::{Api, SseEvent, SseParser};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, Read, Write};
use std::path::Path;
use std::time::{Duration, Instant};

/// `denno chat` の設定。
#[derive(Debug, Clone)]
pub struct ChatOptions {
    pub url: String,
    pub api: Api,
    pub system: Option<String>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub timeout_secs: u64,
    /// 空なら `GET /v1/models` から解決する。
    pub model: Option<String>,
    pub api_key: Option<String>,
}

/// 1 ターン分の実測値。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TurnStats {
    /// リクエスト送信から最初の text delta まで。
    pub ttft_ms: f64,
    /// 最初の delta から最後の delta まで / (出力トークン - 1)。
    pub tpot_ms: f64,
    /// リクエスト送信からストリーム終端まで。
    pub elapsed_ms: f64,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chars: usize,
    pub stop_reason: Option<String>,
}

/// 会話 1 発話。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    fn user(content: &str) -> Message {
        Message {
            role: "user".into(),
            content: content.to_string(),
        }
    }

    fn assistant(content: &str) -> Message {
        Message {
            role: "assistant".into(),
            content: content.to_string(),
        }
    }
}

/// `/save` が書くファイル。`/load` はこの形 (version は任意) を読む。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationFile {
    #[serde(default = "default_version")]
    pub version: u32,
    #[serde(default)]
    pub api: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub messages: Vec<Message>,
}

fn default_version() -> u32 {
    1
}

/// 対話 1 セッション。HTTP クライアントと履歴を持つ。
pub struct Session {
    opts: ChatOptions,
    agent: ureq::Agent,
    base: String,
    api: Api,
    model: String,
    history: Vec<Message>,
    last_stats: Option<TurnStats>,
}

impl Session {
    pub fn new(opts: ChatOptions) -> Session {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(opts.timeout_secs.max(1)))
            .build();
        let base = opts.url.trim_end_matches('/').to_string();
        let api = opts.api;
        let model = opts.model.clone().unwrap_or_default();
        Session {
            opts,
            agent,
            base,
            api,
            model,
            history: Vec::new(),
            last_stats: None,
        }
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn endpoint(&self) -> String {
        match self.api {
            Api::Anthropic => format!("{}/v1/messages", self.base),
            Api::Openai => format!("{}/v1/chat/completions", self.base),
        }
    }

    /// 認証ヘッダを付ける。`--api-key` 未指定なら何もしない。
    fn authed(&self, req: ureq::Request) -> ureq::Request {
        match &self.opts.api_key {
            Some(k) => req.set("x-api-key", k).set("authorization", &format!("Bearer {k}")),
            None => req,
        }
    }

    /// `GET /v1/models` から先頭の id を取る。失敗は None。
    pub fn resolve_model(&mut self) -> Option<String> {
        if self.model.is_empty() {
            if let Some(id) = self.fetch_first_model() {
                self.model = id;
            }
        }
        if self.model.is_empty() {
            None
        } else {
            Some(self.model.clone())
        }
    }

    fn fetch_first_model(&self) -> Option<String> {
        let url = format!("{}/v1/models", self.base);
        let resp = self.authed(self.agent.get(&url)).call().ok()?;
        let v: Value = serde_json::from_str(&resp.into_string().ok()?).ok()?;
        Some(v["data"].get(0)?["id"].as_str()?.to_string())
    }

    fn request_messages(&self) -> Vec<Message> {
        let mut msgs = Vec::with_capacity(self.history.len() + 1);
        if let Some(sys) = &self.opts.system {
            msgs.push(Message {
                role: "system".into(),
                content: sys.clone(),
            });
        }
        msgs.extend(self.history.iter().cloned());
        msgs
    }

    fn build_body(&self, user_text: &str) -> String {
        let mut msgs = self.request_messages();
        msgs.push(Message::user(user_text));
        let msgs_json: Vec<Value> = msgs
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        let mut body = json!({
            "model": self.model,
            "messages": msgs_json,
            "max_tokens": self.opts.max_tokens,
            "temperature": self.opts.temperature,
            "stream": true,
        });
        match self.api {
            Api::Anthropic => {
                // Anthropic は system を messages ではなくトップレベルに置く。
                // 互換サーバーはどちらでも受けるが、実サーバーに合わせる。
                if let Some(sys) = &self.opts.system {
                    body["system"] = json!(sys);
                }
                if let Some(arr) = body["messages"].as_array_mut() {
                    arr.retain(|m| m["role"] != "system");
                }
            }
            Api::Openai => {
                body["stream_options"] = json!({"include_usage": true});
            }
        }
        body.to_string()
    }

    /// 1 ターン送る。`out` へ text delta を到着順に書き、flush する。
    pub fn stream_turn(&mut self, user_text: &str, out: &mut dyn Write) -> Result<TurnStats> {
        let url = self.endpoint();
        let body = self.build_body(user_text);
        let start = Instant::now();
        let req = self
            .agent
            .post(&url)
            .set("content-type", "application/json")
            .set("accept", "text/event-stream");
        let resp = match self.authed(req).send_string(&body) {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                bail!("HTTP {code} from {url}: {}", text.trim())
            }
            Err(e) => return Err(anyhow::Error::new(e)).with_context(|| format!("POST {url}")),
        };

        let mut reader = resp.into_reader();
        let mut parser = SseParser::new();
        let mut text = String::new();
        let mut stats = TurnStats::default();
        let mut first_delta: Option<Duration> = None;
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
                let line = String::from_utf8_lossy(&raw).to_string();
                if let Some(ev) = parser.push_line(&line) {
                    let now = start.elapsed();
                    if handle_event(self.api, &ev, now, &mut stats, &mut text, &mut first_delta, &mut last_delta, out)? {
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
                let _ = handle_event(self.api, &ev, now, &mut stats, &mut text, &mut first_delta, &mut last_delta, out)?;
            }
        }

        stats.elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        stats.chars = text.chars().count();
        match (first_delta, last_delta) {
            (Some(f), Some(l)) => {
                stats.ttft_ms = f.as_secs_f64() * 1000.0;
                let n = stats.output_tokens.max(2) - 1;
                stats.tpot_ms = (l.saturating_sub(f)).as_secs_f64() * 1000.0 / n as f64;
            }
            _ => {
                if let Some(f) = first_delta {
                    stats.ttft_ms = f.as_secs_f64() * 1000.0;
                }
            }
        }

        // 履歴は成功したターンだけ積む (失敗したら会話を汚さない)。
        self.history.push(Message::user(user_text));
        self.history.push(Message::assistant(&text));
        self.last_stats = Some(stats.clone());
        Ok(stats)
    }

    /// `GET /v1/steer` を取って pretty JSON を返す。
    pub fn fetch_steer(&self) -> Result<String> {
        let url = format!("{}/v1/steer", self.base);
        let resp = match self.authed(self.agent.get(&url)).call() {
            Ok(r) => r,
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                bail!(
                    "GET /v1/steer: HTTP {code}{}",
                    if text.trim().is_empty() {
                        String::new()
                    } else {
                        format!(": {}", text.trim())
                    }
                )
            }
            Err(e) => return Err(anyhow::Error::new(e)).context("GET /v1/steer"),
        };
        let text = resp.into_string().context("read /v1/steer body")?;
        let v: Value = serde_json::from_str(&text)
            .with_context(|| format!("parse /v1/steer JSON: {}", text.trim()))?;
        Ok(serde_json::to_string_pretty(&v)?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let file = ConversationFile {
            version: 1,
            api: match self.api {
                Api::Anthropic => "anthropic".into(),
                Api::Openai => "openai".into(),
            },
            model: self.model.clone(),
            system: self.opts.system.clone(),
            messages: self.history.clone(),
        };
        let text = serde_json::to_string_pretty(&file)?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))?;
        Ok(())
    }

    pub fn load(&mut self, path: &Path) -> Result<usize> {
        let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let file: ConversationFile = serde_json::from_str(&text)
            .with_context(|| format!("parse {} (expected /save format)", path.display()))?;
        for m in &file.messages {
            match m.role.as_str() {
                "user" | "assistant" | "system" => {}
                other => bail!("{}: unknown role {other:?}", path.display()),
            }
        }
        self.history = file.messages;
        self.last_stats = None;
        if self.model.is_empty() && !file.model.is_empty() {
            self.model = file.model;
        }
        Ok(self.history.len())
    }

    /// スラッシュコマンドを処理する。`Ok(false)` は `/exit`。
    pub fn handle_slash(&mut self, line: &str, ui: &mut dyn Write) -> Result<bool> {
        let line = line.trim();
        let (cmd, arg) = match line.split_once(char::is_whitespace) {
            Some((c, a)) => (c, a.trim()),
            None => (line, ""),
        };
        match cmd {
            "/help" => {
                writeln!(ui, "slash commands:")?;
                writeln!(ui, "  /help             このヘルプ")?;
                writeln!(ui, "  /reset            履歴と統計を消す")?;
                writeln!(ui, "  /stats            直前ターンの TTFT/TPOT/トークン数")?;
                writeln!(ui, "  /save <file.json> 履歴を JSON へ保存")?;
                writeln!(ui, "  /load <file.json> 履歴を JSON から読む")?;
                writeln!(ui, "  /steer            GET /v1/steer (適用中のステアリング)")?;
                writeln!(ui, "  /exit             終了 (Ctrl-D でも終了)")?;
            }
            "/reset" => {
                self.history.clear();
                self.last_stats = None;
                writeln!(ui, "history cleared")?;
            }
            "/stats" => match &self.last_stats {
                Some(s) => {
                    writeln!(
                        ui,
                        "ttft {:.1} ms / tpot {:.2} ms / in {} / out {} / {:.0} ms / {} chars{}",
                        s.ttft_ms,
                        s.tpot_ms,
                        s.input_tokens,
                        s.output_tokens,
                        s.elapsed_ms,
                        s.chars,
                        s.stop_reason
                            .as_deref()
                            .map(|r| format!(" / stop {r}"))
                            .unwrap_or_default()
                    )?;
                }
                None => writeln!(ui, "no stats yet")?,
            },
            "/save" => {
                if arg.is_empty() {
                    writeln!(ui, "usage: /save <file.json>")?;
                } else {
                    self.save(Path::new(arg))?;
                    writeln!(ui, "saved {} messages to {}", self.history.len(), arg)?;
                }
            }
            "/load" => {
                if arg.is_empty() {
                    writeln!(ui, "usage: /load <file.json>")?;
                } else {
                    let n = self.load(Path::new(arg))?;
                    writeln!(ui, "loaded {n} messages from {arg}")?;
                }
            }
            "/steer" => match self.fetch_steer() {
                Ok(s) => writeln!(ui, "{s}")?,
                Err(e) => writeln!(ui, "steer unavailable: {e:#}")?,
            },
            "/exit" | "/quit" => return Ok(false),
            other => writeln!(ui, "unknown command {other:?} (try /help)")?,
        }
        Ok(true)
    }
}

/// SSE 1 イベントを処理する。ストリーム終端なら true。
#[allow(clippy::too_many_arguments)]
fn handle_event(
    api: Api,
    ev: &SseEvent,
    now: Duration,
    stats: &mut TurnStats,
    text: &mut String,
    first_delta: &mut Option<Duration>,
    last_delta: &mut Option<Duration>,
    out: &mut dyn Write,
) -> Result<bool> {
    match api {
        Api::Anthropic => {
            if ev.event.as_deref() == Some("ping") {
                return Ok(false);
            }
            let v: Value = match serde_json::from_str(&ev.data) {
                Ok(v) => v,
                Err(_) => return Ok(false),
            };
            match v["type"].as_str().unwrap_or_default() {
                "message_start" => {
                    stats.input_tokens =
                        v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0) as usize;
                }
                "content_block_delta" => {
                    if v["delta"]["type"] == "text_delta" {
                        let t = v["delta"]["text"].as_str().unwrap_or_default();
                        if !t.is_empty() {
                            out.write_all(t.as_bytes())?;
                            out.flush()?;
                            text.push_str(t);
                            if first_delta.is_none() {
                                *first_delta = Some(now);
                            }
                            *last_delta = Some(now);
                        }
                    }
                }
                "message_delta" => {
                    if let Some(n) = v["usage"]["output_tokens"].as_u64() {
                        stats.output_tokens = n as usize;
                    }
                    if let Some(r) = v["delta"]["stop_reason"].as_str() {
                        stats.stop_reason = Some(r.to_string());
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
            let v: Value = match serde_json::from_str(&ev.data) {
                Ok(v) => v,
                Err(_) => return Ok(false),
            };
            if let Some(err) = v.get("error") {
                bail!("server SSE error: {err}");
            }
            if let Some(u) = v.get("usage") {
                if !u.is_null() {
                    stats.input_tokens = u["prompt_tokens"].as_u64().unwrap_or(0) as usize;
                    stats.output_tokens = u["completion_tokens"].as_u64().unwrap_or(0) as usize;
                }
            }
            let Some(choice) = v["choices"].get(0) else {
                return Ok(false);
            };
            if let Some(r) = choice["finish_reason"].as_str() {
                stats.stop_reason = Some(r.to_string());
            }
            let t = choice["delta"]["content"].as_str().unwrap_or_default();
            if !t.is_empty() {
                out.write_all(t.as_bytes())?;
                out.flush()?;
                text.push_str(t);
                if first_delta.is_none() {
                    *first_delta = Some(now);
                }
                *last_delta = Some(now);
            }
            Ok(false)
        }
    }
}

/// `denno chat` の本体。stdin から 1 行ずつ読み、対話する。
pub fn run(opts: ChatOptions) -> Result<()> {
    let mut session = Session::new(opts);
    if session.model().is_empty() {
        let _ = session.resolve_model();
    }
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    loop {
        {
            let mut err = std::io::stderr();
            write!(err, "you> ")?;
            err.flush()?;
        }
        let mut line = String::new();
        let n = stdin.read_line(&mut line).context("read stdin")?;
        if n == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('/') {
            let mut err = std::io::stderr();
            match session.handle_slash(line, &mut err) {
                Ok(true) => continue,
                Ok(false) => break,
                Err(e) => {
                    writeln!(err, "error: {e:#}")?;
                    continue;
                }
            }
        }
        match session.stream_turn(line, &mut out) {
            Ok(_) => {
                writeln!(out)?;
                out.flush()?;
            }
            Err(e) => {
                let mut err = std::io::stderr();
                writeln!(err, "error: {e:#}")?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> ChatOptions {
        ChatOptions {
            url: "http://127.0.0.1:1".into(),
            api: Api::Anthropic,
            system: Some("be brief".into()),
            max_tokens: 64,
            temperature: 0.2,
            timeout_secs: 2,
            model: Some("m".into()),
            api_key: None,
        }
    }

    #[test]
    fn anthropic_body_hoists_system_and_keeps_history() {
        let mut s = Session::new(opts());
        s.history.push(Message::user("hi"));
        s.history.push(Message::assistant("hello"));
        let body: Value = serde_json::from_str(&s.build_body("again")).unwrap();
        assert_eq!(body["system"], "be brief");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3, "{msgs:?}");
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[2]["content"], "again");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 64);
    }

    #[test]
    fn openai_body_keeps_system_in_messages_and_asks_usage() {
        let mut o = opts();
        o.api = Api::Openai;
        let s = Session::new(o);
        let body: Value = serde_json::from_str(&s.build_body("hi")).unwrap();
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["content"], "hi");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("denno-chat-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("c.json");
        let mut s = Session::new(opts());
        s.history.push(Message::user("q"));
        s.history.push(Message::assistant("a"));
        s.save(&path).unwrap();
        let mut t = Session::new(opts());
        assert_eq!(t.load(&path).unwrap(), 2);
        assert_eq!(t.history, s.history);
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\"version\": 1"));
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn slash_reset_and_unknown_and_exit() {
        let mut s = Session::new(opts());
        s.history.push(Message::user("q"));
        let mut ui = Vec::new();
        assert!(s.handle_slash("/reset", &mut ui).unwrap());
        assert!(s.history.is_empty());
        ui.clear();
        assert!(s.handle_slash("/bogus", &mut ui).unwrap());
        assert!(String::from_utf8(ui.clone()).unwrap().contains("unknown command"));
        ui.clear();
        assert!(!s.handle_slash("/exit", &mut ui).unwrap());
    }

    #[test]
    fn load_rejects_unknown_role() {
        let dir = std::env::temp_dir().join(format!("denno-chat-role-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.json");
        std::fs::write(&path, r#"{"messages":[{"role":"tool","content":"x"}]}"#).unwrap();
        let mut s = Session::new(opts());
        assert!(s.load(&path).is_err());
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
