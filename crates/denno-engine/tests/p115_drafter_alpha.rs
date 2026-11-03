//! task 115: drafter feasibility harness — 9B drafter の token-level 受理率 alpha を測る。
//!
//! 測定のみ (投機的デコードの実装はしない)。1 プロセスで:
//!   a. swift15 ターゲット (qwen4exp 分割 GGUF) を貪欲デコード (argmax、
//!      `qwen4exp_decode::argmax` と同じ規約 = f32 生値比較・同点は添字小) して
//!      プロンプトごとに 256 トークン生成する。
//!   b. 9B drafter (qwen35) に prompt + ターゲットトークン列を teacher forcing し、
//!      各位置の drafter argmax を記録する。alpha(位置) = argmax がターゲット
//!      トークンと一致したら 1、 else 0。
//!   c. プロンプトごとに alpha 全体 / 位置 1..64 / 65..128 / 129..256 (1 始まり、
//!      0 始まりの半開区間 [0,64) [64,128) [128,256) と同じ) / 位置 0 からの
//!      最長共通接頭辞 (LCP) を出す。
//!
//! モード (環境変数 DENNO_P115_TARGET_DIR の有無で切替):
//! - FULL: ターゲット + drafter を同じプロセスで読み、本命の alpha を測る。
//!   ターゲットのトークナイザで chat template 展開した prompt ids を drafter に
//!   そのまま食わせる (specdec 本番と同じ「ターゲットの id 空間を drafter が消費」
//!   する形)。ターゲットと drafter のトークナイザが同じテキストから同じ id 列を
//!   作るかも検査し、崩れていたら大音量で報告する (その場合 specdec は不成立)。
//! - SELFTEST (ターゲットが無い環境): drafter 自身の貪欲出力を「ターゲット列」
//!   として同じ teacher forcing 経路に食わせる。簿記 (位置合わせ・argmax 規約・
//!   bucket 集計) が正しければ alpha は 1.0 になるはずで、ここが 1.0 からずれる
//!   ときは harness 側の欠陥を意味する。drafter の CPU ms/token もこの経路で
//!   実測する (FULL モードの経済性の項で使う数字と同じ経路)。
//!
//! ターゲットが無い環境でのトークナイザ照合は、代わりに DENNO_P115_REF_GGUF
//! (Flash-Next 系と同一のトークナイザ メタデータを持つ GGUF) を比較相手に使う。
//!
//! 実行 (FULL):
//!   DENNO_P115_TARGET_DIR=/models/qwen38-swift15-abl/IQ3_S \
//!   cargo test -p denno-engine --release --test p115_drafter_alpha -- --ignored --nocapture
//!
//! 実行 (SELFTEST、ターゲット無し):
//!   DENNO_P115_REF_GGUF=/work/models/Qwen3.8-27B-Q4_0.gguf \
//!   cargo test -p denno-engine --release --test p115_drafter_alpha -- --ignored --nocapture
//!
//! 環境変数:
//!   DENNO_P115_TARGET_DIR      ターゲット分割 GGUF のディレクトリ (無ければ SELFTEST)
//!   DENNO_P115_DRAFTER_DIR     drafter ディレクトリ (既定: data/qwen35-9b)
//!   DENNO_P115_REF_GGUF        SELFTEST 時のトークナイザ比較相手 (単一ファイル GGUF)
//!   DENNO_P115_N_TOKENS        生成トークン数 (既定 256)
//!   DENNO_P115_MAX_PROMPTS     使うプロンプト数 (既定 5。SELFTEST は既定 2)
//!   DENNO_P115_SELFTEST_PROMPTS  SELFTEST の既定プロンプト数 (既定 2)
//!   DENNO_P115_OUT             結果 JSON の出力先 (既定 /tmp/p115_alpha.json)
//!
//! 数値の注記: drafter は serve 本番と同じ QuantMode::Int8 で読む (9B CPU 経路)。
//! 重みは int8 への再量子化で Q8_0 GGUF と厳密には異なるが、本命の alpha も
//! 「CPU drafter を int8 で動かしたときの値」が意味を持つのでこの経路で測る。

use std::path::{Path, PathBuf};
use std::time::Instant;

use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen35::Qwen35;
use denno_engine::model::qwen4exp::{Qwen4Exp, Qwen4ExpParams};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_decode::{argmax, DecoderConfig, Qwen4ExpDecoder};
use denno_engine::model::qwen4exp_ssm::SsmAttn;
use denno_engine::model::Model;
use denno_engine::quant::QuantMode;
use denno_engine::tok_gguf::GgufTokenizer;

/// プロンプト (本番トラフィックに近い自然な日本語・1 ターン user)。
const PROMPTS: [&str; 5] = [
    // SF 小説の書き出し依頼
    "小説書いてみてください。ジャンルはSFでお願いします",
    // 技術的な質問
    "RustのArc<Mutex<T>>とArc<RwLock<T>>はどんなときに使い分けるべきですか?",
    // コード書いて系
    "PythonでCSVを読み込んで列ごとの平均を出すスクリプトを書いてください",
    // 会話を書いて系
    "喫茶店での二人の会話をセリフだけで書いてください。テーマは引っ越しです",
    // 事実 QA
    "日本で一番標高の高い山は何ですか?",
];

struct Config {
    target_dir: Option<PathBuf>,
    drafter_dir: PathBuf,
    ref_gguf: Option<PathBuf>,
    n_tokens: usize,
    max_prompts: usize,
    out_path: PathBuf,
}

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn read_config() -> Config {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let drafter_dir =
        env_path("DENNO_P115_DRAFTER_DIR").unwrap_or_else(|| manifest.join("../../data/qwen35-9b"));
    Config {
        target_dir: env_path("DENNO_P115_TARGET_DIR"),
        drafter_dir,
        ref_gguf: env_path("DENNO_P115_REF_GGUF"),
        n_tokens: env_usize("DENNO_P115_N_TOKENS", 256),
        max_prompts: env_usize("DENNO_P115_MAX_PROMPTS", 5),
        out_path: env_path("DENNO_P115_OUT")
            .unwrap_or_else(|| PathBuf::from("/tmp/p115_alpha.json")),
    }
}

/// ディレクトリ内の .gguf をソートして返す (serve_flash::shards と同じ規約)。
fn ggufs_in(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("gguf"))
        .collect();
    v.sort();
    anyhow::ensure!(!v.is_empty(), "{} に .gguf が無い", dir.display());
    Ok(v)
}

/// teacher forcing の結果。
struct ForceOut {
    /// hits[j] = 「prompt + target[..j] を見た drafter の argmax == target[j]」。
    hits: Vec<bool>,
    /// 不一致位置の drafter argmax (診断用)。全部一致なら None。
    miss: Option<(usize, u32, u32)>,
    /// decode 1 ステップ平均 (ms/token)。ターゲット列を 1 個食うごとの实测。
    ms_per_tok: f64,
    prefill_ms: f64,
}

/// drafter を読む (serve の 9B 経路と同じ QuantMode::Int8)。
fn load_drafter(dir: &Path) -> anyhow::Result<(Qwen35, GgufTokenizer)> {
    let cfg_path = dir.join("config.json");
    let cfg: serde_json::Value = serde_json::from_reader(std::fs::File::open(&cfg_path)?)?;
    let m = Qwen35::load(dir, &cfg, QuantMode::Int8)?;
    // drafter 側トークナイザは自分の GGUF メタデータから (crosscheck 専用。
    // teacher forcing はターゲットの id をそのまま食うので使わない)。
    let gguf = ggufs_in(dir)?.remove(0);
    let f = GgufFile::open(&gguf)?;
    let tok = GgufTokenizer::from_gguf(&f)?;
    Ok((m, tok))
}

/// teacher forcing: prompt を prefill してから target の各トークンを食わせ、
/// 直前 logits の argmax が target トークンと一致したかを記録する。
fn teacher_force(m: &mut Qwen35, prompt: &[u32], target: &[u32]) -> anyhow::Result<ForceOut> {
    let mut cache = m.new_cache();
    let t0 = Instant::now();
    let mut logits = m.forward(prompt, &mut cache)?;
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut hits = Vec::with_capacity(target.len());
    let mut miss: Option<(usize, u32, u32)> = None;
    let mut decode_ns: u128 = 0;
    for (j, &t) in target.iter().enumerate() {
        let a = argmax(&logits)?;
        let a = u32::try_from(a).map_err(|_| anyhow::anyhow!("argmax {a} が u32 に収まらない"))?;
        hits.push(a == t);
        if miss.is_none() && a != t {
            miss = Some((j, t, a));
        }
        let s = Instant::now();
        // teacher forcing: 自分の argmax ではなくターゲットのトークンを食う
        // (specdec 本番の verify がターゲット列を drafter に食わせるのと同じ形)。
        logits = m.forward(&[t], &mut cache)?;
        decode_ns += s.elapsed().as_nanos();
    }
    let n = target.len().max(1) as f64;
    Ok(ForceOut {
        hits,
        miss,
        ms_per_tok: decode_ns as f64 / 1e6 / n,
        prefill_ms,
    })
}

/// drafter 単体で貪欲生成 (SELFTEST の「ターゲット列」を作る)。
/// argmax 規約は FULL のターゲット生成と同じ (f32 生値・同点は添字小)。
fn drafter_greedy(
    m: &mut Qwen35,
    prompt: &[u32],
    n: usize,
    eos: Option<u32>,
) -> anyhow::Result<Vec<u32>> {
    let mut cache = m.new_cache();
    let mut logits = m.forward(prompt, &mut cache)?;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let id = argmax(&logits)? as u32;
        if Some(id) == eos {
            break;
        }
        out.push(id);
        logits = m.forward(&[id], &mut cache)?;
    }
    Ok(out)
}

/// ターゲット (qwen4exp) を serve と同じ経路で貪欲生成する。
/// 戻り値: (prompt ids, 生成 ids, ターゲット トークナイザ)。
fn target_greedy(
    dir: &Path,
    text: &str,
    n: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>, GgufTokenizer)> {
    let parts = ggufs_in(dir)?;
    let f = GgufFile::open_split(&parts)?;
    let tok = GgufTokenizer::from_gguf(&f)?;
    // serve の既定 (enable_thinking=false・system 無し・1 ターン user) と同じ形。
    let msgs = vec![("user".to_string(), text.to_string())];
    let prompt = tok.encode_messages(None, &msgs, None, false)?;
    anyhow::ensure!(!prompt.is_empty(), "プロンプトが空");
    let max_ctx = prompt.len() + n + 64;
    let p = Qwen4ExpParams::from_meta(&f)?;
    let full = FullAttn::from_gguf(&f, &p, max_ctx)?;
    let attn = SsmAttn::with_full(&f, &p, Box::new(full))?;
    let m = Qwen4Exp::with_attn(&f, Box::new(attn))?;
    let mut dec = Qwen4ExpDecoder::new(m, DecoderConfig::new(max_ctx));
    let mut logits = dec.prefill_batched(&prompt, true)?;
    let eos = tok.eos();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let id = argmax(&logits)? as u32;
        if Some(id) == eos {
            break;
        }
        out.push(id);
        logits = dec.step(id)?;
    }
    Ok((prompt, out, tok))
}

/// 2 つのトークナイザが同じテキストから同じ id 列を作るか検査する。
/// 戻り値: (raw 一致, chat 一致, 相手語彙で未定義になる id があったか)。
fn tokenizer_crosscheck(
    label_a: &str,
    a: &GgufTokenizer,
    label_b: &str,
    b: &GgufTokenizer,
) -> (Vec<bool>, Vec<bool>, bool) {
    let mut raw_eq = Vec::new();
    let mut chat_eq = Vec::new();
    let mut undef = false;
    for text in PROMPTS {
        let ia = a.encode(text).expect("encode a");
        let ib = b.encode(text).expect("encode b");
        raw_eq.push(ia == ib);
        // ターゲット id を drafter の語彙で解釈できるか (id 空間共有の必要条件)。
        let dec = b.decode(&ia).unwrap_or_default();
        if dec.contains('\u{FFFD}') {
            undef = true;
        }
        let msgs = vec![("user".to_string(), text.to_string())];
        let ca = a.encode_messages(None, &msgs, None, false).expect("chat a");
        let cb = b.encode_messages(None, &msgs, None, false).expect("chat b");
        chat_eq.push(ca == cb);
        println!(
            "  tok[{label_a}] raw={} chat={} | tok[{label_b}] raw={} chat={} | raw_eq={} chat_eq={}",
            ia.len(),
            ca.len(),
            ib.len(),
            cb.len(),
            ia == ib,
            ca == cb
        );
        if ia != ib {
            let k = ia
                .iter()
                .zip(ib.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            println!(
                "    DIFF at raw[{k}]: {label_a}={} vs {label_b}={} (text={text:?})",
                ia.get(k).copied().unwrap_or(u32::MAX),
                ib.get(k).copied().unwrap_or(u32::MAX)
            );
        }
        if ca != cb {
            let k = ca
                .iter()
                .zip(cb.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(0);
            println!(
                "    CHAT DIFF at [{k}]: {label_a}={} vs {label_b}={}",
                ca.get(k).copied().unwrap_or(u32::MAX),
                cb.get(k).copied().unwrap_or(u32::MAX)
            );
        }
    }
    (raw_eq, chat_eq, undef)
}

/// プロンプト結果。
struct PromptResult {
    text: String,
    n_prompt: usize,
    /// 測定対象のトークン列 (FULL: ターゲット生成 / SELFTEST: drafter 自己生成)。
    target_ids: Vec<u32>,
    hits: Vec<bool>,
    miss: Option<(usize, u32, u32)>,
    /// 位置 0 から連続して一致している長さ。
    lcp: usize,
    ms_per_tok: f64,
    prefill_ms: f64,
}

impl PromptResult {
    /// 半開区間 [lo, hi) の alpha。区間が空 (生成が届かない) なら None。
    fn alpha(&self, lo: usize, hi: usize) -> Option<f64> {
        let hi = hi.min(self.hits.len());
        if lo >= hi {
            return None;
        }
        let hit = self.hits[lo..hi].iter().filter(|&&h| h).count();
        Some(hit as f64 / (hi - lo) as f64)
    }

    fn alpha_all(&self) -> f64 {
        if self.hits.is_empty() {
            return 0.0;
        }
        let hit = self.hits.iter().filter(|&&h| h).count();
        hit as f64 / self.hits.len() as f64
    }

    fn lcp(hits: &[bool]) -> usize {
        hits.iter().take_while(|&&h| h).count()
    }
}

fn print_result(tag: &str, r: &PromptResult) {
    let f3 = |v: Option<f64>| v.map(|v| format!("{v:.3}")).unwrap_or("-".into());
    let head: String = r.text.chars().take(24).collect();
    println!(
        "[{tag}] prompt={head:?} n_prompt={} n_gen={} alpha_all={:.3} a[1..64]={} a[65..128]={} a[129..256]={} LCP={} drafter={:.1} ms/tok prefill={:.0} ms",
        r.n_prompt,
        r.target_ids.len(),
        r.alpha_all(),
        f3(r.alpha(0, 64)),
        f3(r.alpha(64, 128)),
        f3(r.alpha(128, 256)),
        r.lcp,
        r.ms_per_tok,
        r.prefill_ms
    );
    if let Some((j, t, d)) = r.miss {
        println!("    first_miss at gen[{j}]: target={t} drafter_argmax={d}");
    }
}

#[test]
#[ignore]
fn p115_drafter_feasibility() {
    let cfg = read_config();
    let mode_full = cfg.target_dir.is_some();
    println!(
        "=== p115 mode={} n_tokens={} prompts<={} drafter={}",
        if mode_full { "FULL" } else { "SELFTEST" },
        cfg.n_tokens,
        cfg.max_prompts,
        cfg.drafter_dir.display()
    );

    // ---- drafter を読む (どちらのモードでも必須) ----
    let t0 = Instant::now();
    let (mut drafter, dtok) = load_drafter(&cfg.drafter_dir).expect("drafter ロード失敗");
    let load_s = t0.elapsed().as_secs_f64();
    println!("drafter loaded in {load_s:.1} s (QuantMode::Int8)");
    let deos = dtok.eos();

    let mut results: Vec<PromptResult> = Vec::new();
    let mut tok_raw: Option<Vec<bool>> = None;
    let mut tok_chat: Option<Vec<bool>> = None;
    let mut tok_undef = false;
    let mut tok_label = String::new();

    if mode_full {
        // ---- FULL: ターゲット生成 → drafter teacher forcing ----
        let tdir = cfg.target_dir.as_ref().unwrap();
        let n = cfg.n_tokens;
        for (pi, &text) in PROMPTS.iter().enumerate().take(cfg.max_prompts) {
            println!("--- prompt {pi}: {text:?}");
            let t0 = Instant::now();
            let (prompt, gen, ttok) = target_greedy(tdir, text, n).expect("ターゲット生成失敗");
            let gen_s = t0.elapsed().as_secs_f64();
            println!(
                "  target greedy: n_prompt={} n_gen={} in {gen_s:.1} s (eos={:?})",
                prompt.len(),
                gen.len(),
                ttok.eos()
            );
            if pi == 0 {
                // ターゲット vs drafter のトークナイザ照合は本命の検査。
                let (raw, chat, undef) = tokenizer_crosscheck("target", &ttok, "drafter", &dtok);
                tok_raw = Some(raw);
                tok_chat = Some(chat);
                tok_undef = undef;
                tok_label = "target-vs-drafter".into();
            }
            let t1 = Instant::now();
            let out = teacher_force(&mut drafter, &prompt, &gen).expect("teacher forcing 失敗");
            let tf_s = t1.elapsed().as_secs_f64();
            println!("  teacher forcing: {tf_s:.1} s");
            let lcp = PromptResult::lcp(&out.hits);
            results.push(PromptResult {
                text: text.to_string(),
                n_prompt: prompt.len(),
                target_ids: gen,
                hits: out.hits,
                miss: out.miss,
                lcp,
                ms_per_tok: out.ms_per_tok,
                prefill_ms: out.prefill_ms,
            });
        }
    } else {
        // ---- SELFTEST: drafter 自己生成 → 同じ teacher forcing 経路へ ----
        // 簿記検証: 位置合わせと argmax 規約が正しければ alpha は 1.0 になる。
        let max_p = env_usize("DENNO_P115_SELFTEST_PROMPTS", 2).min(cfg.max_prompts);
        for (pi, &text) in PROMPTS.iter().enumerate().take(max_p) {
            println!("--- prompt {pi}: {text:?}");
            let msgs = vec![("user".to_string(), text.to_string())];
            let prompt = dtok
                .encode_messages(None, &msgs, None, false)
                .expect("drafter chat encode 失敗");
            let t0 = Instant::now();
            let gen = drafter_greedy(&mut drafter, &prompt, cfg.n_tokens, deos)
                .expect("drafter 生成失敗");
            let gen_s = t0.elapsed().as_secs_f64();
            println!(
                "  drafter greedy: n_prompt={} n_gen={} in {gen_s:.1} s (own eos={deos:?})",
                prompt.len(),
                gen.len()
            );
            let t1 = Instant::now();
            let out = teacher_force(&mut drafter, &prompt, &gen).expect("teacher forcing 失敗");
            let tf_s = t1.elapsed().as_secs_f64();
            println!("  teacher forcing: {tf_s:.1} s");
            let lcp = PromptResult::lcp(&out.hits);
            results.push(PromptResult {
                text: text.to_string(),
                n_prompt: prompt.len(),
                target_ids: gen,
                hits: out.hits,
                miss: out.miss,
                lcp,
                ms_per_tok: out.ms_per_tok,
                prefill_ms: out.prefill_ms,
            });
        }
        // トークナイザ照合の代わり (ターゲット GGUF が無い環境):
        // Flash-Next 系と同一メタデータの参照 GGUF があれば drafter と比べる。
        if let Some(refp) = &cfg.ref_gguf {
            if refp.exists() {
                let f = GgufFile::open(refp).expect("ref gguf を開けない");
                let rtok = GgufTokenizer::from_gguf(&f).expect("ref トークナイザを組めない");
                let (raw, chat, undef) = tokenizer_crosscheck("ref", &rtok, "drafter", &dtok);
                tok_raw = Some(raw);
                tok_chat = Some(chat);
                tok_undef = undef;
                tok_label = format!("ref({})-vs-drafter", refp.display());
            }
        }
    }

    // ---- サマリ ----
    println!("=== per-prompt results");
    for r in &results {
        print_result(if mode_full { "FULL" } else { "SELF" }, r);
    }
    let alpha_mean = if results.is_empty() {
        f64::NAN
    } else {
        let w: f64 = results
            .iter()
            .map(|r| r.alpha_all() * r.target_ids.len() as f64)
            .sum();
        let n: usize = results.iter().map(|r| r.target_ids.len()).sum();
        w / n.max(1) as f64
    };
    let lcp_mean = if results.is_empty() {
        0.0
    } else {
        results.iter().map(|r| r.lcp as f64).sum::<f64>() / results.len() as f64
    };
    let ms_tok_mean = if results.is_empty() {
        f64::NAN
    } else {
        results.iter().map(|r| r.ms_per_tok).sum::<f64>() / results.len() as f64
    };
    println!(
        "=== SUMMARY mode={} alpha_mean(token-weighted)={alpha_mean:.4} lcp_mean={lcp_mean:.1} drafter_cpu_ms_per_tok={ms_tok_mean:.1} drafter_load_s={load_s:.1}",
        if mode_full { "FULL" } else { "SELFTEST" }
    );
    match (&tok_raw, &tok_chat) {
        (Some(raw), Some(chat)) => {
            println!(
                "=== TOKENIZER {tok_label}: raw_equal={raw:?} chat_equal={chat:?} undef_in_drafter={tok_undef}"
            );
            if raw.iter().any(|&e| !e) || chat.iter().any(|&e| !e) || tok_undef {
                println!("!!! TOKENIZER MISMATCH: ターゲットの id 空間を drafter が共有していない。specdec はこのままだと不成立");
            }
        }
        _ => {
            println!("=== TOKENIZER: 比較相手が無くスキップ (DENNO_P115_TARGET_DIR か DENNO_P115_REF_GGUF を設定)");
        }
    }

    // ---- JSON 出力 ----
    let j = serde_json::json!({
        "mode": if mode_full { "full" } else { "selftest" },
        "drafter_dir": cfg.drafter_dir.display().to_string(),
        "target_dir": cfg.target_dir.as_ref().map(|p| p.display().to_string()),
        "n_tokens_requested": cfg.n_tokens,
        "drafter_load_s": load_s,
        "drafter_cpu_ms_per_tok_mean": ms_tok_mean,
        "alpha_mean_token_weighted": alpha_mean,
        "lcp_mean": lcp_mean,
        "tokenizer": {
            "label": tok_label,
            "raw_equal": tok_raw,
            "chat_equal": tok_chat,
            "undef_in_drafter": tok_undef,
        },
        "prompts": results.iter().map(|r| serde_json::json!({
            "text": r.text,
            "n_prompt": r.n_prompt,
            "n_gen": r.target_ids.len(),
            "alpha_all": r.alpha_all(),
            "alpha_1_64": r.alpha(0, 64),
            "alpha_65_128": r.alpha(64, 128),
            "alpha_129_256": r.alpha(128, 256),
            "lcp": r.lcp,
            "drafter_ms_per_tok": r.ms_per_tok,
            "prefill_ms": r.prefill_ms,
            "gen_head": r.target_ids.iter().take(16).copied().collect::<Vec<u32>>(),
        })).collect::<Vec<_>>(),
    });
    if let Some(parent) = cfg.out_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(
        &cfg.out_path,
        serde_json::to_string_pretty(&j).unwrap() + "\n",
    )
    .expect("結果 JSON を書けない");
    println!("=== JSON written to {}", cfg.out_path.display());

    // ---- 恒等検査 ----
    // SELFTEST のときは簿記が正しければ alpha は 1.0 (自己生成を自分で teacher
    // forcing するので、同一プロセス・同一経路なら 1 つも外れない)。
    if !mode_full {
        for r in &results {
            assert_eq!(
                r.alpha_all(),
                1.0,
                "SELFTEST で alpha != 1.0 (prompt={:?}): harness の簿記か argmax 規約が壊れている",
                r.text
            );
            assert_eq!(r.lcp, r.target_ids.len(), "SELFTEST で LCP != n_gen");
        }
        println!("=== SELFTEST identity check OK (alpha == 1.0 on all prompts)");
    }
}
