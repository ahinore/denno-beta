//! task 865: `denno abliterate` — ヘッドレスな abliteration 作業台。
//!
//! 起動中の denno サーバーに対して HTTP で次を順に行う:
//!
//! 1. pos / neg のプロンプト集を 1 本ずつ prefill させて残差を記録する
//!    (`POST /v1/steer/record` → 生成リクエスト → `POST /v1/steer/record/stop`)
//! 2. `POST /v1/steer/analyze` で層ごとの分離スコアを取り、層の ASCII 棒グラフを
//!    標準出力へ描く (推奨区間 `[from, to)` を `*` で示す)
//! 3. `GET /v1/steer/export` で pos / neg のスナップショットを `.dns` として保存
//! 4. `POST /v1/steer/apply` で pos - neg を適用し、必要なら
//!    `POST /v1/steer/recipe/export` のレシピ JSON を書き出す
//! 5. `--eval-file` があれば OFF (未適用) と ON (適用) の生成を比べる
//!
//! 進捗は stderr、グラフと評価表は stdout に出す (パイプで使いやすい)。
//! 適用は ON のまま残す (終了後にそのまま対話できる)。

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// `denno abliterate` の設定。
#[derive(Debug, Clone)]
pub struct AbliterateOptions {
    pub url: String,
    pub pos_file: PathBuf,
    pub neg_file: PathBuf,
    pub eval_file: Option<PathBuf>,
    /// `all` か `last_prompt_token`。
    pub capture_mode: String,
    pub alpha: f32,
    pub out_dir: PathBuf,
    pub out_recipe: Option<PathBuf>,
    /// スナップショット名の接頭辞 (`<tag>_pos` / `<tag>_neg`)。
    pub tag: String,
    pub model: Option<String>,
    pub eval_max_tokens: usize,
    pub embed_snapshots: bool,
    pub timeout_secs: u64,
}

// ---------------------------------------------------------------------------
// 純粋な部分 (テスト対象)
// ---------------------------------------------------------------------------

/// プロンプト集を読む。1 行 1 本。空行と `#` 始まりの行は捨てる。
pub fn parse_prompts(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.trim_end_matches('\r').trim())
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.to_string())
        .collect()
}

/// `capture_mode` の検証。サーバーの `record_params` と同じ 2 値だけ受ける。
pub fn check_capture_mode(s: &str) -> Result<()> {
    match s {
        "all" | "last_prompt_token" => Ok(()),
        other => bail!("未知の capture_mode {other:?} (all / last_prompt_token)"),
    }
}

/// `/v1/steer/analyze` の 1 層分。
#[derive(Debug, Clone, PartialEq)]
pub struct LayerScore {
    pub layer: usize,
    pub sep: f32,
    pub diff_norm: f32,
    pub cos: f32,
}

/// `/v1/steer/analyze` の応答。
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    pub pos: String,
    pub neg: String,
    /// 推奨区間 `[from, to)` (サーバーの値。空なら全層に直す)。
    pub range: (usize, usize),
    pub peak_layer: usize,
    pub layers: Vec<LayerScore>,
}

impl Analysis {
    /// サーバーの JSON を読む。形が違えばエラー (黙って 0 埋めしない)。
    pub fn from_json(v: &Value) -> Result<Analysis> {
        let pos = v["pos"]
            .as_str()
            .ok_or_else(|| anyhow!("analyze 応答に pos が無い: {v}"))?
            .to_string();
        let neg = v["neg"]
            .as_str()
            .ok_or_else(|| anyhow!("analyze 応答に neg が無い: {v}"))?
            .to_string();
        let arr = v["layers"]
            .as_array()
            .ok_or_else(|| anyhow!("analyze 応答に layers が無い: {v}"))?;
        if arr.is_empty() {
            bail!("analyze 応答の layers が空");
        }
        let mut layers = Vec::with_capacity(arr.len());
        for (i, l) in arr.iter().enumerate() {
            let layer = l["layer"]
                .as_u64()
                .ok_or_else(|| anyhow!("layers[{i}].layer が整数でない: {l}"))?
                as usize;
            let num = |name: &str| -> Result<f32> {
                let x = l[name]
                    .as_f64()
                    .ok_or_else(|| anyhow!("layers[{i}].{name} が数値でない: {l}"))?;
                if !x.is_finite() {
                    bail!("layers[{i}].{name} が有限でない: {l}");
                }
                Ok(x as f32)
            };
            layers.push(LayerScore {
                layer,
                sep: num("sep_score")?,
                diff_norm: num("diff_norm")?,
                cos: num("cos_sim")?,
            });
        }
        layers.sort_by_key(|l| l.layer);
        let n = layers.len();
        let (mut from, mut to) = match v["recommended_range"].as_array() {
            Some(r) if r.len() == 2 => (
                r[0].as_u64().unwrap_or(0) as usize,
                r[1].as_u64().unwrap_or(0) as usize,
            ),
            _ => (0, n),
        };
        if from >= to {
            // サーバーの推奨が空でも作業を止めない (全層に掛ける)。
            eprintln!("[abliterate] 警告: recommended_range [{from}, {to}) が空。全層 [0, {n}) を使う");
            from = 0;
            to = n;
        }
        let peak_layer = v["peak_layer"]
            .as_u64()
            .map(|x| x as usize)
            .unwrap_or_else(|| {
                layers
                    .iter()
                    .max_by(|a, b| a.sep.partial_cmp(&b.sep).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|l| l.layer)
                    .unwrap_or(0)
            });
        Ok(Analysis {
            pos,
            neg,
            range: (from, to),
            peak_layer,
            layers,
        })
    }

    /// pos の平均ノルム (表示用)。
    pub fn mean_diff_norm(&self) -> f32 {
        if self.layers.is_empty() {
            return 0.0;
        }
        self.layers.iter().map(|l| l.diff_norm).sum::<f32>() / self.layers.len() as f32
    }
}

/// 層ごとの分離スコアの ASCII 棒グラフ。推奨区間の行に `*` を付ける。
pub fn render_chart(a: &Analysis, width: usize) -> String {
    let max = a.layers.iter().map(|l| l.sep).fold(0.0f32, f32::max);
    let scale = if max > 0.0 { max } else { 1.0 };
    let width = width.max(8);
    let mut out = String::new();
    out.push_str(&format!(
        "per-layer separation ({} layers, bar 0..{:.3}, * = recommended [{}, {}))\n",
        a.layers.len(),
        scale,
        a.range.0,
        a.range.1
    ));
    for l in &a.layers {
        let n = ((l.sep.clamp(0.0, scale) / scale) * width as f32).round() as usize;
        let mark = if l.layer >= a.range.0 && l.layer < a.range.1 {
            "  *"
        } else {
            ""
        };
        out.push_str(&format!(
            "{:>3} |{:<width$}| {:6.3}{}\n",
            l.layer,
            "#".repeat(n.min(width)),
            l.sep,
            mark,
            width = width
        ));
    }
    out.push_str(&format!(
        "peak layer {} (sep {:.3}), mean |diff| {:.4}\n",
        a.peak_layer,
        a.layers
            .iter()
            .find(|l| l.layer == a.peak_layer)
            .map(|l| l.sep)
            .unwrap_or(0.0),
        a.mean_diff_norm()
    ));
    out
}

/// 文字の LCS による類似度 (1.0 = 完全一致)。長すぎる出力は先頭 2000 文字で切る。
pub fn similarity(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().take(2000).collect();
    let b: Vec<char> = b.chars().take(2000).collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = 0;
        for j in 1..=b.len() {
            cur[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(cur[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    let lcs = prev[b.len()];
    2.0 * lcs as f32 / (a.len() + b.len()) as f32
}

/// 最初に食い違う文字位置 (文字数)。接頭辞なら短い方の長さ。
pub fn first_diff(a: &str, b: &str) -> Option<usize> {
    if a == b {
        return None;
    }
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let n = av.len().min(bv.len());
    for i in 0..n {
        if av[i] != bv[i] {
            return Some(i);
        }
    }
    Some(n)
}

/// `POST /v1/steer/record` の body。
pub fn record_body(name: &str, capture_mode: &str) -> Value {
    json!({"name": name, "capture_mode": capture_mode})
}

/// `POST /v1/steer/apply` の body (pos - neg を `[from, to)` に掛ける)。
pub fn apply_body(pos: &str, neg: &str, range: (usize, usize), alpha: f32) -> Value {
    json!({
        "terms": [
            {"snapshot": pos, "coef": 1.0},
            {"snapshot": neg, "coef": -1.0},
        ],
        "layers": [range.0, range.1],
        "alpha": alpha,
        "mode": "subtract",
    })
}

/// `POST /v1/steer/recipe/export` の body。
pub fn recipe_body(
    name: &str,
    pos: &str,
    neg: &str,
    range: (usize, usize),
    alpha: f32,
    embed: bool,
) -> Value {
    json!({
        "name": name,
        "description": format!("abliterate {pos} - {neg}"),
        "terms": [
            {"snapshot": pos, "coef": 1.0},
            {"snapshot": neg, "coef": -1.0},
        ],
        "layers": [range.0, range.1],
        "alpha": alpha,
        "mode": "subtract",
        "embed_snapshots": embed,
    })
}

/// `POST /v1/chat/completions` (非ストリーム) の body。
pub fn completion_body(model: &str, prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": false,
    })
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

struct Client {
    agent: ureq::Agent,
    base: String,
}

impl Client {
    fn new(url: &str, timeout_secs: u64) -> Client {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(timeout_secs.max(1)))
            .build();
        Client {
            agent,
            base: url.trim_end_matches('/').to_string(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base, path)
    }

    /// 応答を (本文, Content-Type) で返す。HTTP エラーは本文付きで Err。
    fn finish(
        resp: Result<ureq::Response, ureq::Error>,
        what: &str,
    ) -> Result<(Vec<u8>, Option<String>)> {
        match resp {
            Ok(r) => {
                let ctype = r.header("content-type").map(|s| s.to_string());
                let mut buf = Vec::new();
                r.into_reader()
                    .read_to_end(&mut buf)
                    .with_context(|| format!("read body of {what}"))?;
                Ok((buf, ctype))
            }
            Err(ureq::Error::Status(code, r)) => {
                let text = r.into_string().unwrap_or_default();
                if text.trim().is_empty() {
                    bail!("{what}: HTTP {code}")
                } else {
                    bail!("{what}: HTTP {code}: {}", text.trim())
                }
            }
            Err(e) => Err(anyhow::Error::new(e)).with_context(|| what.to_string()),
        }
    }

    fn get(&self, path: &str) -> Result<(Vec<u8>, Option<String>)> {
        let url = self.url(path);
        Self::finish(self.agent.get(&url).call(), &format!("GET {path}"))
    }

    fn get_json(&self, path: &str) -> Result<Value> {
        let (buf, _) = self.get(path)?;
        let text = String::from_utf8_lossy(&buf).to_string();
        serde_json::from_str(&text)
            .with_context(|| format!("parse JSON of GET {path}: {}", text.trim()))
    }

    fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let url = self.url(path);
        let resp = self
            .agent
            .post(&url)
            .set("content-type", "application/json")
            .set("accept", "application/json")
            .send_string(&body.to_string());
        let (buf, _) = Self::finish(resp, &format!("POST {path}"))?;
        let text = String::from_utf8_lossy(&buf).to_string();
        serde_json::from_str(&text)
            .with_context(|| format!("parse JSON of POST {path}: {}", text.trim()))
    }

    fn delete_json(&self, path: &str) -> Result<Value> {
        let url = self.url(path);
        let (buf, _) = Self::finish(self.agent.delete(&url).call(), &format!("DELETE {path}"))?;
        let text = String::from_utf8_lossy(&buf).to_string();
        serde_json::from_str(&text)
            .with_context(|| format!("parse JSON of DELETE {path}: {}", text.trim()))
    }

    /// モデル ID を解決する (フラグ → `GET /v1/models` の先頭 → 空文字)。
    fn resolve_model(&self, want: Option<&str>) -> String {
        if let Some(m) = want {
            return m.to_string();
        }
        match self.get_json("/v1/models") {
            Ok(v) => v["data"]
                .get(0)
                .and_then(|m| m["id"].as_str())
                .unwrap_or("")
                .to_string(),
            Err(e) => {
                eprintln!("[abliterate] /v1/models を取れない ({e:#})。model は空で送る");
                String::new()
            }
        }
    }

    /// 非ストリームの 1 生成。返り値は本文テキスト。
    fn complete(&self, model: &str, prompt: &str, max_tokens: usize) -> Result<String> {
        let v = self.post_json(
            "/v1/chat/completions",
            &completion_body(model, prompt, max_tokens),
        )?;
        let choice = v["choices"]
            .get(0)
            .ok_or_else(|| anyhow!("completion に choices が無い: {v}"))?;
        if let Some(t) = choice["message"]["content"].as_str() {
            return Ok(t.to_string());
        }
        if let Some(t) = choice["text"].as_str() {
            return Ok(t.to_string());
        }
        bail!("completion に content が無い: {v}")
    }
}

// ---------------------------------------------------------------------------
// 本体
// ---------------------------------------------------------------------------

fn read_prompts(path: &Path) -> Result<Vec<String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read prompts {}", path.display()))?;
    let prompts = parse_prompts(&text);
    if prompts.is_empty() {
        bail!(
            "{}: 有効なプロンプトが 0 本 (空行と # 始まりは無視)",
            path.display()
        );
    }
    Ok(prompts)
}

/// 記録 → プロンプトを流す → stop。返り値は記録トークン数。
fn record_snapshot(
    c: &Client,
    model: &str,
    name: &str,
    capture_mode: &str,
    prompts: &[String],
) -> Result<u64> {
    c.post_json("/v1/steer/record", &record_body(name, capture_mode))?;
    for (i, p) in prompts.iter().enumerate() {
        // 記録は prefill で起きるので生成は 1 トークンで足りる。
        c.complete(model, p, 1)
            .with_context(|| format!("record {name}: prompt {} of {}", i + 1, prompts.len()))?;
    }
    let v = c.post_json("/v1/steer/record/stop", &json!({}))?;
    let tokens = v["saved"]["tokens"].as_u64().unwrap_or(0);
    eprintln!(
        "[abliterate] snapshot {name}: {} prompts, {tokens} tokens (capture_mode={capture_mode})",
        prompts.len()
    );
    Ok(tokens)
}

struct EvalRow {
    prompt: String,
    sim: f32,
    off_chars: usize,
    on_chars: usize,
    first_diff: Option<usize>,
    off_head: String,
    on_head: String,
}

fn head(s: &str, n: usize) -> String {
    let one_line: String = s.chars().map(|c| if c == '\n' { ' ' } else { c }).collect();
    let mut out: String = one_line.chars().take(n).collect();
    if one_line.chars().count() > n {
        out.push_str("...");
    }
    out
}

fn eval_off_on(
    c: &Client,
    model: &str,
    prompts: &[String],
    max_tokens: usize,
    apply_v: &Value,
) -> Result<()> {
    c.delete_json("/v1/steer/apply")?;
    eprintln!("[abliterate] eval: OFF (適用なし) を {} 本", prompts.len());
    let mut off = Vec::with_capacity(prompts.len());
    for p in prompts {
        off.push(c.complete(model, p, max_tokens)?);
    }
    c.post_json("/v1/steer/apply", apply_v)?;
    eprintln!("[abliterate] eval: ON (適用あり) を {} 本", prompts.len());
    let mut on = Vec::with_capacity(prompts.len());
    for p in prompts {
        on.push(c.complete(model, p, max_tokens)?);
    }

    let rows: Vec<EvalRow> = prompts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let (o, n) = (&off[i], &on[i]);
            EvalRow {
                prompt: p.clone(),
                sim: similarity(o, n),
                off_chars: o.chars().count(),
                on_chars: n.chars().count(),
                first_diff: first_diff(o, n),
                off_head: head(o, 28),
                on_head: head(n, 28),
            }
        })
        .collect();

    println!("\nOFF vs ON ({} prompts, {} tokens/prompt)", rows.len(), max_tokens);
    println!("  {:>3}  {:>6}  {:>5} {:>5}  {:>5}  {}", "#", "sim", "off", "on", "diff@", "output (off => on)");
    for (i, r) in rows.iter().enumerate() {
        println!(
            "  {:>3}  {:>6.3}  {:>5} {:>5}  {:>5}  {} => {}",
            i,
            r.sim,
            r.off_chars,
            r.on_chars,
            r.first_diff.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
            r.off_head,
            r.on_head
        );
        let _ = &r.prompt;
    }
    let same = rows.iter().filter(|r| r.sim >= 1.0).count();
    let mean = if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(|r| r.sim).sum::<f32>() / rows.len() as f32
    };
    println!(
        "identical {same} / {}, changed {}, mean similarity {mean:.3}",
        rows.len(),
        rows.len() - same
    );
    Ok(())
}

/// `denno abliterate` の本体。
pub fn run(opts: AbliterateOptions) -> Result<()> {
    check_capture_mode(&opts.capture_mode)?;
    if !opts.alpha.is_finite() {
        bail!("alpha が有限でない: {}", opts.alpha);
    }
    if opts.eval_max_tokens == 0 {
        bail!("eval_max_tokens が 0");
    }
    let pos_prompts = read_prompts(&opts.pos_file)?;
    let neg_prompts = read_prompts(&opts.neg_file)?;
    let eval_prompts = match &opts.eval_file {
        Some(p) => Some(read_prompts(p)?),
        None => None,
    };

    let pos_name = format!("{}_pos", opts.tag);
    let neg_name = format!("{}_neg", opts.tag);
    let c = Client::new(&opts.url, opts.timeout_secs);
    let model = c.resolve_model(opts.model.as_deref());
    eprintln!(
        "[abliterate] {} pos={} ({} 本) neg={} ({} 本) capture_mode={}",
        c.base,
        pos_name,
        pos_prompts.len(),
        neg_name,
        neg_prompts.len(),
        opts.capture_mode
    );

    // 1. 記録
    record_snapshot(&c, &model, &pos_name, &opts.capture_mode, &pos_prompts)?;
    record_snapshot(&c, &model, &neg_name, &opts.capture_mode, &neg_prompts)?;

    // 2. 分析とチャート
    let an = Analysis::from_json(&c.post_json(
        "/v1/steer/analyze",
        &json!({"pos": pos_name, "neg": neg_name}),
    )?)?;
    print!("{}", render_chart(&an, 48));

    // 3. .dns の書き出し
    std::fs::create_dir_all(&opts.out_dir)
        .with_context(|| format!("create out dir {}", opts.out_dir.display()))?;
    for name in [&pos_name, &neg_name] {
        let (buf, ctype) = c.get(&format!("/v1/steer/export?name={name}"))?;
        let path = opts.out_dir.join(format!("{name}.dns"));
        std::fs::write(&path, &buf).with_context(|| format!("write {}", path.display()))?;
        eprintln!(
            "[abliterate] wrote {} ({} bytes, {})",
            path.display(),
            buf.len(),
            ctype.as_deref().unwrap_or("?")
        );
    }

    // 4. 適用 (ON はここから先ずっと ON)
    let apply_v = apply_body(&pos_name, &neg_name, an.range, opts.alpha);
    c.post_json("/v1/steer/apply", &apply_v)?;
    eprintln!(
        "[abliterate] applied {pos_name} - {neg_name} on layers [{}, {}) alpha={}",
        an.range.0, an.range.1, opts.alpha
    );

    // 5. レシピ
    if let Some(path) = &opts.out_recipe {
        let body = recipe_body(
            &format!("{}-recipe", opts.tag),
            &pos_name,
            &neg_name,
            an.range,
            opts.alpha,
            opts.embed_snapshots,
        );
        let v = c.post_json("/v1/steer/recipe/export", &body)?;
        let text = serde_json::to_string_pretty(&v)?;
        std::fs::write(path, format!("{text}\n"))
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("[abliterate] wrote recipe {}", path.display());
    }

    // 6. OFF vs ON
    if let Some(prompts) = eval_prompts {
        eval_off_on(&c, &model, &prompts, opts.eval_max_tokens, &apply_v)?;
    }

    eprintln!("[abliterate] done (apply は ON のまま。戻すなら DELETE /v1/steer/apply)");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis_json() -> Value {
        let seps = [0.05, 0.4, 0.8, 0.3, 0.01];
        let layers: Vec<Value> = (0..5)
            .map(|i| {
                json!({
                    "layer": i,
                    "diff_norm": 0.1 * (i as f32 + 1.0),
                    "norm_pos": 1.0,
                    "norm_neg": 1.0,
                    "cos_sim": 0.9 - 0.1 * i as f32,
                    "sep_score": seps[i],
                })
            })
            .collect();
        json!({
            "pos": "ablit_pos",
            "neg": "ablit_neg",
            "n_layer": 5,
            "peak_layer": 2,
            "recommended_range": [1, 3],
            "layers": layers,
        })
    }

    #[test]
    fn parse_prompts_skips_blank_and_comments_and_trims_crlf() {
        let text = "first\r\n\r\n# a comment\n  second  \n\t\nthird\n";
        assert_eq!(parse_prompts(text), vec!["first", "second", "third"]);
        assert!(parse_prompts("\n\n# x\n").is_empty());
    }

    #[test]
    fn capture_mode_only_accepts_two_values() {
        assert!(check_capture_mode("all").is_ok());
        assert!(check_capture_mode("last_prompt_token").is_ok());
        let e = check_capture_mode("last").unwrap_err().to_string();
        assert!(e.contains("last_prompt_token"), "{e}");
    }

    #[test]
    fn analysis_parses_layers_and_range() {
        let a = Analysis::from_json(&analysis_json()).unwrap();
        assert_eq!(a.pos, "ablit_pos");
        assert_eq!(a.neg, "ablit_neg");
        assert_eq!(a.range, (1, 3));
        assert_eq!(a.peak_layer, 2);
        assert_eq!(a.layers.len(), 5);
        assert!((a.layers[2].sep - 0.8).abs() < 1e-6);
        assert!((a.mean_diff_norm() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn analysis_falls_back_to_all_layers_when_range_empty() {
        let mut v = analysis_json();
        v["recommended_range"] = json!([2, 2]);
        let a = Analysis::from_json(&v).unwrap();
        assert_eq!(a.range, (0, a.layers.len()));
    }

    #[test]
    fn analysis_falls_back_to_peak_when_missing() {
        let mut v = analysis_json();
        v.as_object_mut().unwrap().remove("peak_layer");
        let a = Analysis::from_json(&v).unwrap();
        assert_eq!(a.peak_layer, 2);
    }

    #[test]
    fn analysis_rejects_malformed_layer() {
        let mut v = analysis_json();
        v["layers"][1].as_object_mut().unwrap().remove("sep_score");
        assert!(Analysis::from_json(&v).is_err());
    }

    #[test]
    fn chart_marks_only_recommended_layers_and_has_a_line_per_layer() {
        let a = Analysis::from_json(&analysis_json()).unwrap();
        let chart = render_chart(&a, 40);
        let lines: Vec<&str> = chart.lines().collect();
        // 見出し + 5 層 + 末尾の 1 行
        assert_eq!(lines.len(), 7, "{chart}");
        assert!(lines[0].contains("[1, 3)"), "{chart}");
        let starred: Vec<&str> = lines[1..6].iter().copied().filter(|l| l.ends_with('*')).collect();
        assert_eq!(starred.len(), 2, "{chart}");
        assert!(starred[0].starts_with("  1 |"), "{chart}");
        assert!(starred[1].starts_with("  2 |"), "{chart}");
        // ピークの層が最も長い棒
        let bars = |l: &str| l.matches('#').count();
        assert!(bars(lines[3]) >= bars(lines[1]));
        assert!(bars(lines[3]) > bars(lines[5]));
        assert!(lines[6].contains("peak layer 2"), "{chart}");
    }

    #[test]
    fn chart_survives_all_zero_scores() {
        let mut v = analysis_json();
        for l in v["layers"].as_array_mut().unwrap() {
            l["sep_score"] = json!(0.0);
        }
        let a = Analysis::from_json(&v).unwrap();
        let chart = render_chart(&a, 20);
        assert_eq!(chart.lines().count(), 7);
        assert!(!chart.contains('#'));
    }

    #[test]
    fn similarity_and_first_diff() {
        assert_eq!(similarity("abc", "abc"), 1.0);
        assert_eq!(similarity("", ""), 1.0);
        assert_eq!(similarity("abc", ""), 0.0);
        assert!(similarity("abcd", "abxd") > 0.7);
        assert!(similarity("abcd", "dcba") < 0.6);
        assert_eq!(first_diff("abc", "abc"), None);
        assert_eq!(first_diff("abc", "abx"), Some(2));
        assert_eq!(first_diff("abc", "abcdef"), Some(3));
    }

    #[test]
    fn bodies_match_the_server_schema() {
        assert_eq!(
            record_body("A", "last_prompt_token"),
            json!({"name": "A", "capture_mode": "last_prompt_token"})
        );
        let a = apply_body("P", "N", (3, 9), 0.75);
        assert_eq!(a["terms"][0], json!({"snapshot": "P", "coef": 1.0}));
        assert_eq!(a["terms"][1], json!({"snapshot": "N", "coef": -1.0}));
        assert_eq!(a["layers"], json!([3, 9]));
        assert_eq!(a["mode"], "subtract");
        assert!((a["alpha"].as_f64().unwrap() - 0.75).abs() < 1e-6);

        let r = recipe_body("rc", "P", "N", (3, 9), 0.75, true);
        assert_eq!(r["name"], "rc");
        assert_eq!(r["embed_snapshots"], true);
        assert_eq!(r["layers"], json!([3, 9]));

        let b = completion_body("m", "hi", 4);
        assert_eq!(b["messages"][0]["content"], "hi");
        assert_eq!(b["max_tokens"], 4);
        assert_eq!(b["stream"], false);
        assert_eq!(b["temperature"], 0.0);
    }

    #[test]
    fn head_is_single_line_and_truncates() {
        assert_eq!(head("a\nb", 10), "a b");
        assert_eq!(head("abcdef", 3), "abc...");
        assert_eq!(head("abc", 3), "abc");
    }
}
