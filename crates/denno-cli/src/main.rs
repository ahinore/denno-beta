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
    tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new("info")
    });
    tracing_subscriber::fmt().init();

    match Cli::parse().command {
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
            let st_path = dir.join("model.safetensors");
            if st_path.exists() {
                let bytes = std::fs::metadata(&st_path)?.len();
                println!("weights:   {:.1} MB", bytes as f64 / 1e6);
            }
        }
        Commands::Generate { repo, prompt, system, max_tokens, temperature, seed, quant, raw } => {
            let dir = model_dir(&repo);
            if !dir.exists() {
                bail!("not cached: {} (run `denno pull {}` first)", dir.display(), repo);
            }
            let quant_mode = denno_engine::QuantMode::parse(&quant)?;
            let mut engine = denno_engine::Engine::load(&dir, repo.clone(), seed, quant_mode)?;
            let params = denno_engine::GenParams {
                max_tokens,
                temperature,
                ..Default::default()
            };
            let out = if raw {
                // 生プロンプトで生成
                let ids = engine.tok.encode(&prompt)?;
                engine.cache.reset();
                let mut logits = engine.model.forward(&ids, &mut engine.cache)?;
                use denno_engine::sampler::Sampler;
                let mut sampler = Sampler::new(temperature, None, None, seed);
                let mut generated = Vec::new();
                for _ in 0..max_tokens {
                    let t = sampler.sample(&logits) as u32;
                    if engine.model.eos_ids().contains(&t) { break; }
                    generated.push(t);
                    logits = engine.model.forward(&[t], &mut engine.cache)?;
                }
                engine.tok.decode(&generated)?
            } else {
                let msgs = vec![denno_engine::ChatMessage { role: "user".into(), content: prompt }];
                engine.chat(system.as_deref(), &msgs, &params, |_| {})? .text
            };
            println!("{}", out);
        }
        Commands::Serve { repo, host, port, name, api_key, max_tokens: _max_tokens, seed, prefix_cache_tokens, quant } => {
            let dir = model_dir(&repo);
            if !dir.exists() {
                bail!("not cached: {} (run `denno pull {}` first)", dir.display(), repo);
            }
            let quant_mode = denno_engine::QuantMode::parse(&quant)?;
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(denno_server::run(denno_server::ServeConfig {
                model_dir: dir,
                name: name.unwrap_or_else(|| repo.clone()),
                host,
                port,
                api_key,
                default_seed: seed,
                prefix_cache_tokens,
                quant: quant_mode,
            }))?;
        }
    }
    Ok(())
}
