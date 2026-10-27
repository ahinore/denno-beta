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
use denno_engine::model::qwen4exp::{Qwen4Exp, Qwen4ExpParams};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_decode::{DecoderConfig, Qwen4ExpDecoder};
use denno_engine::model::qwen4exp_ssm::SsmAttn;
use denno_engine::sampler::Sampler;
use denno_engine::tok_gguf::GgufTokenizer;
use serde_json::{json, Value};

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

impl GenReq {
    fn from_json(v: &Value, default_max: usize) -> Result<Self> {
        let system = match v.get("system") {
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
            tools: v.get("tools").cloned(),
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

        // 直前の状態が使えるなら差分だけ食わせる。
        let t1 = Instant::now();
        let mut reused = 0usize;
        let (mut dec, mut logits) = match self.cache.take() {
            Some(mut c) if c.toks.len() <= prompt.len() && prompt[..c.toks.len()] == c.toks[..] => {
                reused = c.toks.len();
                if reused < prompt.len() {
                    let lg = c.dec.prefill_batched(&prompt[reused..], self.batch)?;
                    (c.dec, lg)
                } else {
                    (c.dec, c.logits)
                }
            }
            _ => {
                let m = self.build_model()?;
                let mut d = Qwen4ExpDecoder::new(m, DecoderConfig::new(self.max_ctx));
                let lg = d.prefill_batched(&prompt, self.batch)?;
                (d, lg)
            }
        };
        let t_prefill = t1.elapsed().as_secs_f64();

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
        let t2 = Instant::now();
        for _ in 0..req.max_tokens {
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
            match dec.step(id) {
                Ok(lg) => logits = lg,
                Err(e) => {
                    failed = Some(e.into());
                    break;
                }
            }
        }
        let t_decode = t2.elapsed().as_secs_f64();
        eprintln!(
            "[flash] prompt={n_prompt} reuse={reused} out={} prefill={t_prefill:.2}s ({:.0} ms/tok) decode={t_decode:.2}s ({:.0} ms/tok)",
            out.len(),
            t_prefill * 1e3 / (n_prompt - reused).max(1) as f64,
            t_decode * 1e3 / out.len().max(1) as f64
        );
        // 途中で失敗したら状態は信用しない (次のリクエストで組み直す)。
        if let Some(e) = failed {
            return Err(e);
        }
        let toks = dec.tokens().to_vec();
        self.cache = Some(Cached { dec, toks, logits });
        Ok(GenOut {
            text,
            n_prompt,
            n_out: out.len(),
            stop_reason: stop_hit.unwrap_or_else(|| {
                if out.len() >= req.max_tokens {
                    "max_tokens".into()
                } else {
                    "end_turn".into()
                }
            }),
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
    body: Vec<u8>,
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
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            }
        }
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        stream.read_exact(&mut body)?;
    }
    Ok(Some(HttpReq {
        method,
        path,
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

fn handle(flash: &mut Flash, mut stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true).ok();
    let mut r = BufReader::new(stream.try_clone()?);
    let Some(req) = read_req(&mut r)? else {
        return Ok(());
    };
    let path = req.path.split('?').next().unwrap_or("/").to_string();

    if req.method == "GET" && (path == "/health" || path == "/healthz") {
        return send_json(&mut stream, "200 OK", &json!({"ok": true, "model": flash.name}));
    }
    if req.method == "GET" && path == "/v1/models" {
        return send_json(
            &mut stream,
            "200 OK",
            &json!({"object": "list", "data": [{
                "id": flash.name, "object": "model", "type": "model",
                "display_name": flash.name, "created_at": "2026-10-27T00:00:00Z"
            }]}),
        );
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
    {
        let p = Qwen4ExpParams::from_meta(&f).context("メタデータを読めない")?;
        let budget = denno_engine::gpu_budget::flash_budget_from_gguf(&f, &p, cfg.max_ctx);
        eprintln!("{}", denno_engine::gpu_budget::budget_line(&budget));
    }
    drop(f);
    let batch = std::env::var("DENNO_PREFILL_BATCH").ok().as_deref() != Some("0");
    let chunk = std::env::var("DENNO_PREFILL_BATCH_CHUNK").unwrap_or_else(|_| "8".into());
    let mut flash = Flash {
        parts,
        max_ctx: cfg.max_ctx,
        name: cfg.name.clone(),
        default_max: cfg.max_tokens,
        tok,
        batch,
        cache: None,
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
