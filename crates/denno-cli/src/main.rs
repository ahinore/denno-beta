//! denno: TensorFold 型の単一バイナリ CLI (pull / models / info / generate / serve)

mod pull;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "denno", version, about = "最小限の Anthropic互換 LLM サーバー")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Hugging Face からモデルをキャッシュにダウンロード
    Pull { repo: String },
    /// キャッシュ済みモデル一覧
    Models,
    /// モデルの情報表示
    Info { repo: String },
    /// 単発生成(デバッグ用)
    Generate {
        repo: String,
        #[arg(long)] prompt: String,
        #[arg(long)] system: Option<String>,
        #[arg(long, default_value_t = 128)] max_tokens: usize,
        #[arg(long)] temperature: Option<f32>,
        #[arg(long)] seed: Option<u64>,
        /// A2.5: 反復ペナルティ (例: 1.1。未指定 = 無効)
        #[arg(long)] repetition_penalty: Option<f32>,
        /// s56: frequency penalty (OpenAI 互換。出現回数 × 係数を減算。未指定/0 = 無効)
        #[arg(long)] frequency_penalty: Option<f32>,
        /// s56: presence penalty (OpenAI 互換。出現したトークンから係数を 1 回減算)
        #[arg(long)] presence_penalty: Option<f32>,
        /// s56: no_repeat_ngram (既出 n-gram の続きを禁止。2 以上。未指定 = 無効)
        #[arg(long)] no_repeat_ngram_size: Option<usize>,
        /// 重みの量子化モード (fp32 / int8)。int8 は決定論を維持するが
        /// fp32 と同一出力にはならない
        #[arg(long, default_value = "fp32")] quant: String,
        /// chat template を適用しない(生プロンプト)
        #[arg(long)] raw: bool,
    },
    /// HTTP サーバー起動 (Anthropic Messages API)
    Serve {
        repo: String,
        #[arg(long, default_value = "127.0.0.1")] host: String,
        #[arg(long, default_value_t = 8080)] port: u16,
        /// クライアントに広告するモデルID
        #[arg(long)] name: Option<String>,
        /// 設定すると x-api-key 認証を要求
        #[arg(long)] api_key: Option<String>,
        #[arg(long, default_value_t = 256)] max_tokens: usize,
        /// デフォルト seed(固定すると決定論的)
        #[arg(long)] seed: Option<u64>,
        /// マルチエントリ prefix cache のトークン上限
        /// (未指定時は DENNO_PREFIX_CACHE_MAX_TOKENS、それも無ければコンテキスト長)
        #[arg(long)] prefix_cache_tokens: Option<usize>,
        /// 重みの量子化モード (fp32 / int8、デフォルト fp32)。
        /// int8 は exact decoding の決定論 (同seed→同出力) を維持するが
        /// fp32 と同一出力ではない (丸め誤差)
        #[arg(long, default_value = "fp32")] quant: String,
        /// P4: リクエストキューの上限 (実行中 1 件の他に待ち行列に入れられる数)。
        /// 同時受け付けがこれを超えると 503 (overloaded_error) を返す
        #[arg(long, default_value_t = 32)] max_queue: usize,
        /// s56b: serve 既定の repetition_penalty (リクエスト未指定時のみ適用)。
        /// 未指定なら従来と完全に同一のサンプリング経路 (出力ビット不変)。
        /// Qwen3.5-9B int8 + 長い system + tools の崩壊対策として 1.05 を推奨
        #[arg(long)] repetition_penalty: Option<f32>,
    },
}

pub fn cache_root() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache").join("denno").join("hub")
}

pub fn model_dir(repo: &str) -> PathBuf {
    // "org/name" → models--org--name (HF cache 風)
    cache_root().join(format!("models--{}", repo.replace('/', "--")))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // s44: serve のみ非同期ロギングを使う (logging.rs — stdout 消費者の凍結で
    // サーバスレッドがブロックしない)。他サブコマンドは従来どおりの同期ロギング。
    if matches!(cli.command, Commands::Serve { .. }) {
        denno_server::init_serve_logging();
    } else {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("info")
        });
        tracing_subscriber::fmt().init();
    }

    match cli.command {
        Commands::Pull { repo } => {
            let dir = model_dir(&repo);
            pull::pull(&repo, &dir)?;
            println!("done: {}", dir.display());
        }
        Commands::Models => {
            let root = cache_root();
            if !root.exists() {
                println!("(cache is empty: {})", root.display());
                return Ok(());
            }
            for entry in std::fs::read_dir(&root)? {
                let entry = entry?;
                if entry.file_type()?.is_dir() {
                    println!("{}", entry.file_name().to_string_lossy().replacen("models--", "", 1).replace("--", "/"));
                }
            }
        }
        Commands::Info { repo } => {
            let dir = model_dir(&repo);
            if !dir.exists() {
                bail!("not cached: {} (run `denno pull {}` first)", dir.display(), repo);
            }
            let cfg: serde_json::Value = serde_json::from_reader(std::fs::File::open(dir.join("config.json"))?)?;
            println!("repo:      {}", repo);
            println!("family:    {}", cfg["model_type"].as_str().unwrap_or("?"));
            println!("layers:    {}", cfg["num_hidden_layers"].as_u64().unwrap_or(cfg["n_layer"].as_u64().unwrap_or(0)));
            println!("hidden:    {}", cfg["hidden_size"].as_u64().unwrap_or(cfg["n_embd"].as_u64().unwrap_or(0)));
            println!("vocab:     {}", cfg["vocab_size"].as_u64().unwrap_or(0));
            let ctx = cfg["max_position_embeddings"].as_u64().unwrap_or(cfg["n_ctx"].as_u64().unwrap_or(0));
            println!("context:   {}", ctx);
            let index_path = dir.join("model.safetensors.index.json");
            let st_path = dir.join("model.safetensors");
            if index_path.exists() {
                // 分割 checkpoint: weight_map が参照する shard ファイルの合計
                if let Ok(index) =
                    serde_json::from_reader::<_, serde_json::Value>(std::fs::File::open(&index_path)?)
                {
                    let mut files: Vec<&str> = index
                        .get("weight_map")
                        .and_then(|v| v.as_object())
                        .map(|m| m.values().filter_map(|v| v.as_str()).collect())
                        .unwrap_or_default();
                    files.sort_unstable();
                    files.dedup();
                    let total: u64 = files
                        .iter()
                        .filter_map(|f| std::fs::metadata(dir.join(f)).ok())
                        .map(|m| m.len())
                        .sum();
                    println!("weights:   {:.1} MB ({} shard(s))", total as f64 / 1e6, files.len());
                }
            } else if st_path.exists() {
                let bytes = std::fs::metadata(&st_path)?.len();
                println!("weights:   {:.1} MB", bytes as f64 / 1e6);
            }
        }
        Commands::Generate { repo, prompt, system, max_tokens, temperature, seed, repetition_penalty, frequency_penalty, presence_penalty, no_repeat_ngram_size, quant, raw } => {
            // s56: 引数の検証 (サーバー 400 と同じ規則。失敗は引数エラーで止める)
            if let Some(p) = frequency_penalty {
                if !p.is_finite() { bail!("--frequency-penalty は有限値 (got {p})"); }
            }
            if let Some(p) = presence_penalty {
                if !p.is_finite() { bail!("--presence-penalty は有限値 (got {p})"); }
            }
            if let Some(n) = no_repeat_ngram_size {
                if n < 2 { bail!("--no-repeat-ngram-size は 2 以上 (got {n})"); }
                if n > 1024 { bail!("--no-repeat-ngram-size は 1024 以下 (got {n})"); }
            }
            let dir = model_dir(&repo);
            if !dir.exists() {
                bail!("not cached: {} (run `denno pull {}` first)", dir.display(), repo);
            }
            let quant_mode = denno_engine::QuantMode::parse(&quant)?;
            let mut engine = denno_engine::Engine::load(&dir, repo.clone(), seed, quant_mode)?;
            let params = denno_engine::GenParams {
                max_tokens,
                temperature,
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                no_repeat_ngram_size,
                ..Default::default()
            };
            // s56: raw 経路も chat 経路と同じ順序/関数を通す
            let anti = denno_engine::sampler::AntiRepetition {
                repetition_penalty,
                frequency_penalty,
                presence_penalty,
                no_repeat_ngram_size,
            }
            .normalized();
            let out = if raw {
                // 生プロンプトで生成
                let ids = engine.tok.encode(&prompt)?;
                engine.caches[0].reset();
                let mut logits = engine.model.forward(&ids, &mut engine.caches[0])?;
                use denno_engine::sampler::Sampler;
                let mut sampler = Sampler::new(temperature, None, None, seed);
                let mut generated = Vec::new();
                for _ in 0..max_tokens {
                    // A2.5/s56: 反復抑止は chat 経路と同じ関数・同じ順序で適用
                    // (4 つすべて無効なら logits に触らない = 既定はビット不変)
                    if !anti.is_noop() {
                        let mut ctx = Vec::with_capacity(ids.len() + generated.len());
                        ctx.extend_from_slice(&ids);
                        ctx.extend_from_slice(&generated);
                        anti.apply(&mut logits, &ctx);
                    }
                    let t = sampler.sample(&logits) as u32;
                    if engine.model.eos_ids().contains(&t) { break; }
                    generated.push(t);
                    logits = engine.model.forward(&[t], &mut engine.caches[0])?;
                }
                engine.tok.decode(&generated)?
            } else {
                let msgs = vec![denno_engine::ChatMessage { role: "user".into(), content: prompt }];
                engine.chat(system.as_deref(), &msgs, &params, |_| {})? .text
            };
            println!("{}", out);
        }
        Commands::Serve { repo, host, port, name, api_key, max_tokens: _max_tokens, seed, prefix_cache_tokens, quant, max_queue, repetition_penalty } => {
            let dir = model_dir(&repo);
            if !dir.exists() {
                bail!("not cached: {} (run `denno pull {}` first)", dir.display(), repo);
            }
            let quant_mode = denno_engine::QuantMode::parse(&quant)?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(denno_server::run(denno_server::ServeConfig {
                model_dir: dir,
                name: name.unwrap_or_else(|| repo.clone()),
                model_id: repo.clone(),
                host,
                port,
                api_key,
                default_seed: seed,
                prefix_cache_tokens,
                quant: quant_mode,
                max_queue,
                repetition_penalty,
            }))?;
        }
    }
    Ok(())
}
