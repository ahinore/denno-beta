//! Qwen3.8-Flash-Next (arch `qwen4exp`) 用の最小 HTTP サーバー。
//!
//! 27B 用の `denno serve` (denno-server / axum) とは別経路。Flash-Next は
//! MoE + hyper-connection + QSA + PLE の 4 要素を持つ別アーキで、`Engine`
//! 実装を持たないため、ここで直接 `Qwen4ExpDecoder` を回す。
//!
//! - `POST /v1/messages` (Anthropic Messages API、`stream` の有無で SSE / JSON)
//! - `GET /v1/models`
//! - `GET /health`
//!
//! 直前のリクエストの状態 (KV / GDN / PLE の窓) を 1 つだけ保持し、次の
//! プロンプトが**そのトークン列の続き**なら差分だけ食わせる。会話のたびに
//! システムプロンプトを読み直さないための仕組みで、前置きが一致しないときは
//! モデルを組み直して最初から prefill する。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp::{Qwen4Exp, Qwen4ExpParams, TapTokens};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_decode::{DecoderConfig, Qwen4ExpDecoder};
use denno_engine::model::qwen4exp_ssm::SsmAttn;
use denno_engine::sampler::Sampler;
use denno_engine::tok_gguf::GgufTokenizer;
use denno_telemetry::{self as telem, Frame};
use serde_json::{json, Value};

use crate::flash_telemetry::{ActTap, Telemetry};

pub struct FlashConfig {
    pub dir: PathBuf,
    pub host: String,
    pub port: u16,
    pub max_ctx: usize,
    pub name: String,
    /// 既定の生成上限 (リクエストの max_tokens が無いとき)
    pub max_tokens: usize,
}

fn shards(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("{} を読めない", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map(|x| x == "gguf").unwrap_or(false))
        .collect();
    v.sort();
    if v.is_empty() {
        bail!("{} に .gguf が無い", dir.display());
    }
    Ok(v)
}

/// 1 リクエストぶんの生成要求。
struct GenReq {
    system: Option<String>,
    messages: Vec<(String, String)>,
    tools: Option<Value>,
    max_tokens: usize,
    temperature: f32,
    top_p: Option<f32>,
    top_k: Option<usize>,
    stop_sequences: Vec<String>,
    enable_thinking: bool,
}

/// 環境変数が 1/true のときだけ有効。
fn env_flag(k: &str) -> bool {
    matches!(std::env::var(k).as_deref(), Ok("1") | Ok("true") | Ok("yes"))
}

impl GenReq {
    fn from_json(v: &Value, default_max: usize) -> Result<Self> {
        let mut system = match v.get("system") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Array(a)) => {
                let mut s = String::new();
                for b in a {
                    if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                        if !s.is_empty() {
                            s.push('\n');
                        }
                        s.push_str(t);
                    }
                }
                Some(s)
            }
            _ => None,
        };
        let mut messages = Vec::new();
        if let Some(Value::Array(ms)) = v.get("messages") {
            for m in ms {
                let role = m
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let text = match m.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(a)) => {
                        let mut s = String::new();
                        for b in a {
                            match b.get("type").and_then(|t| t.as_str()) {
                                Some("text") => {
                                    if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                        if !s.is_empty() {
                                            s.push('\n');
                                        }
                                        s.push_str(t);
                                    }
                                }
                                Some("tool_result") => {
                                    if let Some(t) = b.get("content").and_then(|c| c.as_str()) {
                                        if !s.is_empty() {
                                            s.push('\n');
                                        }
                                        s.push_str(t);
                                    }
                                }
                                _ => {}
                            }
                        }
                        s
                    }
                    _ => String::new(),
                };
                messages.push((role, text));
            }
        }
        if messages.is_empty() {
            bail!("messages が空");
        }

        // 推論サーバ側で、会話に不要なツール定義を落とす (既定 off)。
        // クライアントの MCP / tool 定義は数千トークンになり prefill を直接押し上げる。
        // `DENNO_STRIP_TOOLS=1` で落とす。`DENNO_MAX_SYSTEM=N` なら system の先頭 N 文字だけ残す。
        let mut tools = v.get("tools").cloned();
        if env_flag("DENNO_STRIP_TOOLS") {
            if let Some(t) = tools.take() {
                let bytes = serde_json::to_string(&t).map(|s| s.len()).unwrap_or(0);
                println!("[flash] strip tools: {bytes} bytes dropped");
            }
        }
        if let Some(maxc) = std::env::var("DENNO_MAX_SYSTEM")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        {
            if let Some(s) = system.as_mut() {
                let n = s.chars().count();
                if n > maxc {
                    let cut = s.char_indices().nth(maxc).map(|(i, _)| i).unwrap_or(s.len());
                    s.truncate(cut);
                    println!("[flash] strip system: {n} -> {maxc} chars");
                }
            }
        }
        let max_tokens = v
            .get("max_tokens")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(default_max)
            .min(default_max);
        let temperature = v
            .get("temperature")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(0.7);
        let top_p = v.get("top_p").and_then(|x| x.as_f64()).map(|x| x as f32);
        let top_k = v
            .get("top_k")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize);
        let stop_sequences = v
            .get("stop_sequences")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        let enable_thinking = v
            .get("enable_thinking")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        Ok(GenReq {
            system,
            messages,
            tools,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stop_sequences,
            enable_thinking,
        })
    }
}

/// 直前のリクエストの状態。`toks` は `dec` が食ったトークン列 (古い順)。
struct Cached {
    dec: Qwen4ExpDecoder,
    toks: Vec<u32>,
    logits: Vec<f32>,
}

struct Flash {
    parts: Vec<PathBuf>,
    max_ctx: usize,
    name: String,
    default_max: usize,
    tok: GgufTokenizer,
    /// 層優先のバッチ prefill を使うか (`DENNO_PREFILL_BATCH=0` で切れる)
    batch: bool,
    cache: Option<Cached>,
    /// P51: 推論テレメトリの送出先 (`DENNO_TELEMETRY=0` で無効)。
    telem: Telemetry,
}

/// 計測のみ (`DENNO_Q4_PROFILE=1` / `DENNO_DEV_PROFILE=1`)。
///
/// - `devprof` (T63): op 別の時間・転送量・同期。GPU の GEMV 1 回ごとの
///   内訳を役割名で出す。`DENNO_DEV_PROFILE` 未設定なら 1 行も出さない。
/// - `q4prof`: Flash-Next の層内フェーズを 1 行にまとめて出す。
///
/// どちらも既定 off では分岐 1 回だけで、数値経路には触れない。
fn prof_flash(tag: &str, tok: usize) {
    for line in denno_engine::model::devprof::report(tag, tok) {
        eprintln!("{line}");
    }
    prof_flash_q4(tag, tok);
}

/// `q4prof` は 27B の `DENNO_PROFILE` (表 C) とは別系統で、このモデル専用の軽量な
/// カウンタ。以前はテストからしか読めず serve では取れなかったので、リクエストの
/// 区切り (prefill 終端 / decode 終端) でここから出す。既定 off では
/// `enabled()` の分岐 1 回だけで、数値経路には触れない。
fn prof_flash_q4(tag: &str, tok: usize) {
    use denno_engine::model::qwen4exp::{moe_sub_take, q4prof};
    // decode の GDN コア内訳 (`DENNO_GDN_KPROF=1` のときだけ 1 行)。
    // q4prof とは独立に出す (`DENNO_Q4_PROFILE` を立てなくても読める)。
    // 未設定なら `line()` が None = 出力ゼロ・積算もゼロのまま。
    if let Some(line) = denno_engine::gdn::kprof::line() {
        eprintln!("{line}");
        denno_engine::gdn::kprof::reset();
    }
    if !q4prof::enabled() {
        return;
    }
    let (ple, mix, attn, moe, n) = q4prof::dump();
    let (qkv, gate, gdn, aout, full) = q4prof::dump_attn();
    let (rtr, exp, shx) = moe_sub_take();
    let d = n.max(1) as f64;
    eprintln!(
        "[q4prof] {tag} tok={tok} layer_exec={n} ms total={:.1} | per-layer ple={:.3} mix={:.3} attn={:.3} moe={:.3}",
        ple + mix + attn + moe,
        ple / d,
        mix / d,
        attn / d,
        moe / d
    );
    eprintln!(
        "[q4prof] {tag} ms moe router={rtr:.1} experts={exp:.1} shared={shx:.1} | attn qkv={qkv:.1} gate={gate:.1} gdn={gdn:.1} out={aout:.1} full={full:.1}"
    );
    q4prof::reset();
}

impl Flash {
    /// モデルを組む (mmap 越し)。
    fn build_model(&self) -> Result<Qwen4Exp> {
        let f = GgufFile::open_split(&self.parts).context("分割 GGUF を開けない")?;
        let p = Qwen4ExpParams::from_meta(&f).context("メタデータを読めない")?;
        let full = FullAttn::from_gguf(&f, &p, self.max_ctx).context("FullAttn")?;
        let attn = SsmAttn::with_full(&f, &p, Box::new(full)).context("SsmAttn")?;
        Qwen4Exp::with_attn(&f, Box::new(attn)).context("モデルを組めない")
    }

    /// P53: 層活性の送出が有効か。テレメトリ本体が有効で、かつ
    /// `DENNO_TELEMETRY_ACT` が立っているときだけ (既定 off)。
    fn act_on(&self) -> bool {
        self.telem.on() && denno_telemetry::act_enabled()
    }

    /// P53: prefill の logits。act が有効なら層の出口をタップする
    /// (`TapTokens::Last` = チャンク末尾の 1 トークンだけ)。無効時は従来の
    /// `prefill_batched` をそのまま呼ぶ (フックを付けないので数値経路は不変)。
    fn prefill_logits(&self, dec: &mut Qwen4ExpDecoder, prompt: &[u32]) -> Result<Vec<f32>> {
        if !self.act_on() {
            return Ok(dec.prefill_batched(prompt, self.batch)?);
        }
        let mut tap = ActTap::new(&self.telem, true);
        let logits =
            dec.prefill_batched_tapped(prompt, self.batch, TapTokens::Last, |o| tap.on_layer(o))?;
        tap.report_summary("prefill");
        Ok(logits)
    }

    /// 生成して、確定した本文を返す。`emit` は差分テキストを受け取る。
    fn generate(&mut self, req: &GenReq, mut emit: impl FnMut(&str) -> Result<()>) -> Result<GenOut> {
        let prompt = self
            .tok
            .encode_messages(
                req.system.as_deref(),
                &req.messages,
                req.tools.as_ref(),
                req.enable_thinking,
            )
            .context("プロンプトを組めない")?;
        if prompt.is_empty() {
            bail!("プロンプトが空になった");
        }
        if prompt.len() >= self.max_ctx {
            bail!(
                "プロンプト {} トークンが上限 {} に収まらない",
                prompt.len(),
                self.max_ctx
            );
        }
        let n_prompt = prompt.len();

        // P51: テレメトリ (観測のみ)。無効時はこの bool 1 個を見るだけで、
        // Instant も確保もしない。セッション ID は OFF でも採番する (原子加算 1 回)。
        let t_on = self.telem.on();
        let sid = telem::next_session_id();

        // 直前の状態が使えるなら差分だけ食わせる。
        let t1 = Instant::now();
        let mut reused = 0usize;
        let (mut dec, mut logits) = match self.cache.take() {
            Some(mut c) if c.toks.len() <= prompt.len() && prompt[..c.toks.len()] == c.toks[..] => {
                reused = c.toks.len();
                if reused < prompt.len() {
                    let lg = self.prefill_logits(&mut c.dec, &prompt[reused..])?;
                    (c.dec, lg)
                } else {
                    (c.dec, c.logits)
                }
            }
            _ => {
                let m = self.build_model()?;
                let mut d = Qwen4ExpDecoder::new(m, DecoderConfig::new(self.max_ctx));
                let lg = self.prefill_logits(&mut d, &prompt)?;
                (d, lg)
            }
        };
        let t_prefill = t1.elapsed().as_secs_f64();

        if t_on {
            self.telem.emit(Frame::SessionAdmit {
                session: sid,
                slot: 0,
                prompt_tokens: n_prompt as u32,
            });
            // 実際に食わせた分 (再利用した前置きは含めない) とその所要
            self.telem.emit(Frame::Prefill {
                session: sid,
                prompt_tokens: (n_prompt - reused) as u32,
                ns: (t_prefill * 1e9) as u64,
                tokens: prompt[reused..].to_vec(),
            });
            self.telem.emit(Frame::KvUsage {
                busy: vec![true],
                t_lens: vec![dec.tokens().len() as u32],
            });
        }

        prof_flash("prefill", n_prompt - reused);
        let mut sampler = Sampler::new(
            if req.temperature <= 0.0 {
                None
            } else {
                Some(req.temperature)
            },
            req.top_k,
            req.top_p,
            Some(42),
        );
        let eos = self.tok.eos();
        let mut out: Vec<u32> = Vec::new();
        let mut text = String::new();
        let mut stop_hit: Option<String> = None;
        let mut failed: Option<anyhow::Error> = None;
        // P53: decode は 1 トークンあたり全層ぶん流す (act が有効なときだけ)。
        let mut act = if self.act_on() {
            Some(ActTap::new(&self.telem, false))
        } else {
            None
        };
        let t2 = Instant::now();
        for _ in 0..req.max_tokens {
            // P58: 選ぶ前の分布を top-8 で送る (3D ビューワの候補表示用)。
            // テレメトリ OFF なら計算しない。数値経路には触れない。
            let cand = if t_on { Some(denno_telemetry::topk_softmax(&logits, 8)) } else { None };
            let id = sampler.sample(&logits) as u32;
            if Some(id) == eos {
                break;
            }
            out.push(id);
            let full = self.tok.decode(&out).unwrap_or_default();
            if full.len() > text.len() && full.starts_with(&text) {
                // 停止語が本文に現れたら、そこまでで切る
                let mut cut: Option<usize> = None;
                for s in &req.stop_sequences {
                    if let Some(i) = full.find(s.as_str()) {
                        cut = Some(cut.map_or(i, |c: usize| c.min(i)));
                    }
                }
                if let Some(c) = cut {
                    let keep = &full[..c];
                    if keep.len() > text.len() {
                        if let Err(e) = emit(&keep[text.len()..]) {
                            failed = Some(e);
                        }
                    }
                    text = keep.to_string();
                    stop_hit = Some("stop_sequence".into());
                    break;
                }
                if let Err(e) = emit(&full[text.len()..]) {
                    failed = Some(e);
                    break;
                }
                text = full;
            } else {
                text = full;
            }
            let t_step = t_on.then(Instant::now);
            let pos = dec.tokens().len() as u32;
            let stepped = match act.as_mut() {
                Some(tap) => dec.step_tapped(id, |o| tap.on_layer(o)),
                None => dec.step(id),
            };
            match stepped {
                Ok(lg) => logits = lg,
                Err(e) => {
                    failed = Some(e.into());
                    break;
                }
            }
            if let Some(t0) = t_step {
                self.telem.emit(Frame::Step {
                    width: 1,
                    ns: t0.elapsed().as_nanos() as u64,
                });
                self.telem.emit(Frame::Token { session: sid, token: id, pos });
                if let Some(tk) = cand.as_ref() {
                    self.telem.emit(Frame::TopK {
                        session: sid,
                        chosen: id,
                        entropy_f16: denno_telemetry::f32_to_f16_bits(tk.entropy),
                        entries: tk
                            .entries
                            .iter()
                            .map(|&(tid, p)| (tid, denno_telemetry::f32_to_f16_bits(p)))
                            .collect(),
                    });
                }
                self.telem.emit(Frame::KvUsage {
                    busy: vec![true],
                    t_lens: vec![dec.tokens().len() as u32],
                });
            }
        }
        if let Some(tap) = act.as_ref() {
            tap.report_summary("decode");
        }
        let t_decode = t2.elapsed().as_secs_f64();
        eprintln!(
            "[flash] prompt={n_prompt} reuse={reused} out={} prefill={t_prefill:.2}s ({:.0} ms/tok) decode={t_decode:.2}s ({:.0} ms/tok)",
            out.len(),
            t_prefill * 1e3 / (n_prompt - reused).max(1) as f64,
            t_decode * 1e3 / out.len().max(1) as f64
        );
        prof_flash("decode", out.len());
        // 途中で失敗したら状態は信用しない (次のリクエストで組み直す)。
        let stop_reason = stop_hit.unwrap_or_else(|| {
            if out.len() >= req.max_tokens {
                "max_tokens".into()
            } else {
                "end_turn".into()
            }
        });
        if t_on {
            let reason = if failed.is_some() {
                telem::done_reason::ERROR
            } else if stop_reason == "stop_sequence" {
                telem::done_reason::STOP_SEQUENCE
            } else if stop_reason == "max_tokens" {
                telem::done_reason::MAX_TOKENS
            } else {
                telem::done_reason::END_TURN
            };
            self.telem.emit(Frame::SessionDone {
                session: sid,
                reason,
                output_tokens: out.len() as u32,
            });
        }
        if let Some(e) = failed {
            return Err(e);
        }
        let toks = dec.tokens().to_vec();
        self.cache = Some(Cached { dec, toks, logits });
        Ok(GenOut {
            text,
            n_prompt,
            n_out: out.len(),
            stop_reason,
        })
    }
}

struct GenOut {
    text: String,
    n_prompt: usize,
    n_out: usize,
    stop_reason: String,
}

struct HttpReq {
    method: String,
    path: String,
    /// ヘッダ (名前は小文字、値は trim 済み)。P51 で WS の Upgrade 判定に使う。
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl HttpReq {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

fn read_req(stream: &mut BufReader<TcpStream>) -> Result<Option<HttpReq>> {
    let mut line = String::new();
    if stream.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    let mut it = line.split_whitespace();
    let method = it.next().unwrap_or("").to_string();
    let path = it.next().unwrap_or("/").to_string();
    let mut content_length = 0usize;
    let mut headers: Vec<(String, String)> = Vec::new();
    loop {
        let mut h = String::new();
        if stream.read_line(&mut h)? == 0 {
            break;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().to_string();
            if key == "content-length" {
                content_length = val.parse().unwrap_or(0);
            }
            headers.push((key, val));
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        stream.read_exact(&mut body)?;
    }
    Ok(Some(HttpReq {
        method,
        path,
        headers,
        body,
    }))
}

fn head(stream: &mut TcpStream, status: &str, ctype: &str, extra: &str) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n{extra}\r\n"
    )?;
    Ok(())
}

fn send_json(stream: &mut TcpStream, status: &str, v: &Value) -> Result<()> {
    let body = serde_json::to_vec(v)?;
    head(
        stream,
        status,
        "application/json",
        &format!("Content-Length: {}\r\n", body.len()),
    )?;
    stream.write_all(&body)?;
    stream.flush()?;
    Ok(())
}

fn send_text(stream: &mut TcpStream, status: &str, s: &str) -> Result<()> {
    head(
        stream,
        status,
        "text/plain; charset=utf-8",
        &format!("Content-Length: {}\r\n", s.len()),
    )?;
    stream.write_all(s.as_bytes())?;
    stream.flush()?;
    Ok(())
}

fn sse_event(stream: &mut TcpStream, name: &str, data: &Value) -> Result<()> {
    write!(stream, "event: {name}\ndata: {}\n\n", data)?;
    stream.flush()?;
    Ok(())
}

/// GET の固定応答。P51 でテレメトリ経路を足したときに既存の応答が変わって
/// いないことをテストで固定するため、JSON 組み立てをここへ切り出してある。
fn get_static(path: &str, name: &str) -> Option<Value> {
    match path {
        "/health" | "/healthz" => Some(json!({"ok": true, "model": name})),
        "/v1/models" => Some(json!({"object": "list", "data": [{
            "id": name, "object": "model", "type": "model",
            "display_name": name, "created_at": "2026-10-27T00:00:00Z"
        }]})),
        _ => None,
    }
}

/// `GET /v1/telemetry/ws` の Upgrade 要求から `Sec-WebSocket-Key` を取り出す。
/// Upgrade でない / 鍵が無い / 別経路なら None (呼び出し側が 400 を返す)。
fn ws_upgrade_key<'a>(req: &'a HttpReq, path: &str) -> Option<&'a str> {
    if req.method != "GET" || path != "/v1/telemetry/ws" {
        return None;
    }
    if !req.header("upgrade")?.eq_ignore_ascii_case("websocket") {
        return None;
    }
    match req.header("sec-websocket-key") {
        Some(k) if !k.is_empty() => Some(k),
        _ => None,
    }
}

fn handle(flash: &mut Flash, mut stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true).ok();
    let mut r = BufReader::new(stream.try_clone()?);
    let Some(req) = read_req(&mut r)? else {
        return Ok(());
    };
    let path = req.path.split('?').next().unwrap_or("/").to_string();

    // P51: 推論テレメトリの WebSocket (denno-view の購読先)。101 を返して
    // クライアントを登録し、以後は relay スレッドが書く。推論は待たない。
    if req.method == "GET" && path == "/v1/telemetry/ws" {
        let Some(key) = ws_upgrade_key(&req, &path).map(|k| k.to_string()) else {
            return send_json(
                &mut stream,
                "400 Bad Request",
                &json!({"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "Upgrade: websocket と Sec-WebSocket-Key が要る"
                }}),
            );
        };
        return flash.telem.accept(stream, r, &key);
    }
    if req.method == "GET" {
        if let Some(v) = get_static(&path, &flash.name) {
            return send_json(&mut stream, "200 OK", &v);
        }
    }
    if req.method != "POST" || path != "/v1/messages" {
        return send_json(
            &mut stream,
            "404 Not Found",
            &json!({"type": "error", "error": {"type": "not_found_error", "message": path}}),
        );
    }

    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => {
            return send_json(
                &mut stream,
                "400 Bad Request",
                &json!({"type": "error", "error": {"type": "invalid_request_error", "message": e.to_string()}}),
            )
        }
    };
    let greq = match GenReq::from_json(&body, flash.default_max) {
        Ok(g) => g,
        Err(e) => {
            return send_json(
                &mut stream,
                "400 Bad Request",
                &json!({"type": "error", "error": {"type": "invalid_request_error", "message": e.to_string()}}),
            )
        }
    };
    let want_stream = body
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);
    let id = format!("msg_{}", now_ms());

    if want_stream {
        head(
            &mut stream,
            "200 OK",
            "text/event-stream",
            "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n",
        )?;
        sse_event(
            &mut stream,
            "message_start",
            &json!({"type": "message_start", "message": {
                "id": id, "type": "message", "role": "assistant", "model": flash.name,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 1}
            }}),
        )?;
        sse_event(
            &mut stream,
            "content_block_start",
            &json!({"type": "content_block_start", "index": 0,
                    "content_block": {"type": "text", "text": ""}}),
        )?;
        let mut err: Option<String> = None;
        let out = flash.generate(&greq, |delta| {
            sse_event(
                &mut stream,
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": delta}}),
            )
        });
        let out = match out {
            Ok(o) => o,
            Err(e) => {
                err = Some(format!("{e:#}"));
                GenOut {
                    text: String::new(),
                    n_prompt: 0,
                    n_out: 0,
                    stop_reason: "error".into(),
                }
            }
        };
        if let Some(e) = err {
            let _ = sse_event(
                &mut stream,
                "error",
                &json!({"type": "error", "error": {"type": "api_error", "message": e}}),
            );
            return Ok(());
        }
        sse_event(
            &mut stream,
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": 0}),
        )?;
        sse_event(
            &mut stream,
            "message_delta",
            &json!({"type": "message_delta",
                    "delta": {"stop_reason": out.stop_reason, "stop_sequence": null},
                    "usage": {"input_tokens": out.n_prompt, "output_tokens": out.n_out}}),
        )?;
        sse_event(&mut stream, "message_stop", &json!({"type": "message_stop"}))?;
        return Ok(());
    }

    // 非ストリーム
    let mut chunks = String::new();
    let out = match flash.generate(&greq, |d| {
        chunks.push_str(d);
        Ok(())
    }) {
        Ok(o) => o,
        Err(e) => {
            return send_json(
                &mut stream,
                "500 Internal Server Error",
                &json!({"type": "error", "error": {"type": "api_error", "message": format!("{e:#}")}}),
            )
        }
    };
    send_json(
        &mut stream,
        "200 OK",
        &json!({
            "id": id, "type": "message", "role": "assistant", "model": flash.name,
            "content": [{"type": "text", "text": out.text}],
            "stop_reason": out.stop_reason, "stop_sequence": null,
            "usage": {"input_tokens": out.n_prompt, "output_tokens": out.n_out}
        }),
    )
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

pub fn run(cfg: FlashConfig) -> Result<()> {
    let parts = shards(&cfg.dir)?;
    eprintln!(
        "[flash] {} 分割ファイル: {:?}",
        parts.len(),
        parts
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().to_string())
            .collect::<Vec<_>>()
    );
    let t = Instant::now();
    let f = GgufFile::open_split(&parts).context("分割 GGUF を開けない")?;
    let tok = GgufTokenizer::from_gguf(&f).context("トークナイザを組めない")?;
    // P-pf-arena: どの項が VRAM を食うかを実測前に 1 行で出す。
    // デバイス側 (weights / staging / arena) に max_ctx は入らず、
    // max_ctx に比例するのはホスト側の KV だけ (式は gpu_budget の doc)。
    // P51: テレメトリの ServerInfo 用 (層数)。モデルは実行中に変わらない。
    let n_layer;
    {
        let p = Qwen4ExpParams::from_meta(&f).context("メタデータを読めない")?;
        let budget = denno_engine::gpu_budget::flash_budget_from_gguf(&f, &p, cfg.max_ctx);
        eprintln!("{}", denno_engine::gpu_budget::budget_line(&budget));
        n_layer = p.n_layer;
    }
    drop(f);
    let batch = std::env::var("DENNO_PREFILL_BATCH").ok().as_deref() != Some("0");
    let chunk = std::env::var("DENNO_PREFILL_BATCH_CHUNK").unwrap_or_else(|_| "8".into());

    // P51: 推論テレメトリ (denno-view が購読する WS)。既定で有効で、
    // `DENNO_TELEMETRY=0` のときだけ無効。無効時はチャネルもスレッドも作らない。
    let telem_on = std::env::var("DENNO_TELEMETRY").ok().as_deref() != Some("0");
    // エンジン側の emit() も同じ判定にする (未設定なら有効、ここでは off のみ無効)。
    denno_telemetry::set_enabled(telem_on);
    let (telemetry, telem_rx) = Telemetry::new(
        telem_on,
        Frame::ServerInfo {
            model: cfg.name.clone(),
            model_id: cfg
                .dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
            quant: "gguf".into(),
            slots: 1,
            context: cfg.max_ctx as u32,
            layers: n_layer as u32,
        },
    );
    if let Some(rx) = telem_rx {
        telemetry.start_relay(rx).context("telemetry relay を起動できない")?;
    }
    telemetry
        .start_engine_bridge()
        .context("telemetry の engine bridge を起動できない")?;
    eprintln!(
        "[flash] telemetry: {} (GET /v1/telemetry/ws)",
        if telem_on {
            "on"
        } else {
            "off (DENNO_TELEMETRY=0)"
        }
    );
    // P53: 層活性 (DENNO_TELEMETRY_ACT, 既定 off)。テレメトリ本体が無効なら動かない。
    let act_on = telem_on && denno_telemetry::act_enabled();
    eprintln!(
        "[flash] telemetry act: {}",
        if act_on {
            format!(
                "on k={} dbg={} (decode=全層 / prefill=チャンク末尾)",
                denno_telemetry::act_topk(),
                denno_telemetry::act_dbg()
            )
        } else {
            "off (DENNO_TELEMETRY_ACT=1 で有効)".to_string()
        }
    );

    let mut flash = Flash {
        parts,
        max_ctx: cfg.max_ctx,
        name: cfg.name.clone(),
        default_max: cfg.max_tokens,
        tok,
        batch,
        cache: None,
        telem: telemetry,
    };
    // 起動時に 1 回組んで、読めないならここで落とす
    let m = flash.build_model().context("起動時のロード")?;
    drop(m);
    eprintln!(
        "[flash] load ok {:.1}s eos={:?} batch={} chunk={}",
        t.elapsed().as_secs_f64(),
        flash.tok.eos(),
        flash.batch,
        chunk
    );

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).with_context(|| format!("{addr} を listen できない"))?;
    eprintln!("[flash] listening on http://{addr} model={}", cfg.name);
    for s in listener.incoming() {
        match s {
            Ok(stream) => {
                let t = Instant::now();
                // 逐次処理 (モデルは 1 つ、同時リクエストは受けない)
                if let Err(e) = handle(&mut flash, stream) {
                    eprintln!("[flash] リクエスト失敗: {e:#}");
                }
                eprintln!("[flash] request done {:.1}s", t.elapsed().as_secs_f64());
                // P-pf-arena: 重み登録の成否をリクエスト単位で 1 行 (登録を試みた
                // ときだけ)。GPU_BROKEN のラッチはプロセス寿命なので reset しない。
                let st = denno_engine::gpu_budget::weight_upload_stats();
                if st.attempted() > 0 {
                    eprintln!("{}", denno_engine::gpu_budget::summary_line(&st));
                    denno_engine::gpu_budget::reset_weight_upload_stats();
                }
            }
            Err(e) => eprintln!("[flash] accept 失敗: {e}"),
        }
    }
    Ok(())
}

// ---------- テスト ----------

#[cfg(test)]
mod flash_tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::Duration;

    fn loopback() -> (TcpStream, TcpStream) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let cli = TcpStream::connect(addr).unwrap();
        let (srv, _) = l.accept().unwrap();
        cli.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        (srv, cli)
    }

    fn read_all(s: &mut TcpStream) -> String {
        let mut buf = Vec::new();
        let _ = s.read_to_end(&mut buf);
        String::from_utf8_lossy(&buf).to_string()
    }

    /// P51: 従来からの GET 応答はバイト列まで変えない (テレメトリの有無に依らない)。
    #[test]
    fn static_get_responses_are_byte_identical() {
        let (mut srv, mut cli) = loopback();
        send_json(&mut srv, "200 OK", &get_static("/health", "m").unwrap()).unwrap();
        drop(srv);
        assert_eq!(
            read_all(&mut cli),
            concat!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n",
                "Access-Control-Allow-Origin: *\r\nConnection: close\r\n",
                "Content-Length: 23\r\n\r\n",
                "{\"model\":\"m\",\"ok\":true}"
            )
        );

        let (mut srv, mut cli) = loopback();
        send_json(&mut srv, "200 OK", &get_static("/v1/models", "m").unwrap()).unwrap();
        drop(srv);
        assert_eq!(
            read_all(&mut cli),
            concat!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n",
                "Access-Control-Allow-Origin: *\r\nConnection: close\r\n",
                "Content-Length: 124\r\n\r\n",
                "{\"data\":[{\"created_at\":\"2026-10-27T00:00:00Z\",\"display_name\":\"m\",",
                "\"id\":\"m\",\"object\":\"model\",\"type\":\"model\"}],\"object\":\"list\"}"
            )
        );
    }

    #[test]
    fn static_get_covers_only_the_old_routes() {
        assert_eq!(get_static("/healthz", "x"), Some(json!({"ok": true, "model": "x"})));
        assert!(get_static("/", "x").is_none());
        assert!(get_static("/v1/telemetry/ws", "x").is_none());
        assert!(get_static("/v1/messages", "x").is_none());
    }

    /// P51: read_req が WS の Upgrade ヘッダを拾える (名前は小文字で保持)。
    #[test]
    fn read_req_captures_ws_headers() {
        let (srv, mut cli) = loopback();
        cli.write_all(
            b"GET /v1/telemetry/ws?x=1 HTTP/1.1\r\nHost: h\r\nUpgrade: WebSocket\r\n\
              Connection: upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
        )
        .unwrap();
        let mut r = BufReader::new(srv);
        let req = read_req(&mut r).unwrap().unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/v1/telemetry/ws?x=1");
        assert_eq!(req.header("upgrade"), Some("WebSocket"));
        assert_eq!(req.header("sec-websocket-key"), Some("dGhlIHNhbXBsZSBub25jZQ=="));
        assert_eq!(req.header("connection"), Some("upgrade"));
    }

    /// POST のボディ読み取りは従来どおり。
    #[test]
    fn read_req_still_reads_body() {
        let (srv, mut cli) = loopback();
        cli.write_all(b"POST /v1/messages HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello")
            .unwrap();
        let mut r = BufReader::new(srv);
        let req = read_req(&mut r).unwrap().unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.body, b"hello");
    }

    fn req(method: &str, path: &str, headers: &[(&str, &str)]) -> HttpReq {
        HttpReq {
            method: method.to_string(),
            path: path.to_string(),
            headers: headers
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            body: Vec::new(),
        }
    }

    /// P51: WS 経路の判定 (Upgrade でない/鍵が無い/別経路は 400 側)。
    #[test]
    fn ws_upgrade_key_gate() {
        let ok = req(
            "GET",
            "/v1/telemetry/ws",
            &[("upgrade", "WebSocket"), ("sec-websocket-key", "abc")],
        );
        assert_eq!(ws_upgrade_key(&ok, "/v1/telemetry/ws"), Some("abc"));
        let no_key = req("GET", "/v1/telemetry/ws", &[("upgrade", "websocket")]);
        assert_eq!(ws_upgrade_key(&no_key, "/v1/telemetry/ws"), None);
        let no_up = req("GET", "/v1/telemetry/ws", &[("sec-websocket-key", "abc")]);
        assert_eq!(ws_upgrade_key(&no_up, "/v1/telemetry/ws"), None);
        let empty = req(
            "GET",
            "/v1/telemetry/ws",
            &[("upgrade", "websocket"), ("sec-websocket-key", "")],
        );
        assert_eq!(ws_upgrade_key(&empty, "/v1/telemetry/ws"), None);
        assert_eq!(ws_upgrade_key(&ok, "/v1/models"), None);
        let post = req(
            "POST",
            "/v1/telemetry/ws",
            &[("upgrade", "websocket"), ("sec-websocket-key", "abc")],
        );
        assert_eq!(ws_upgrade_key(&post, "/v1/telemetry/ws"), None);
    }
}
