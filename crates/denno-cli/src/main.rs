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
        /// s67b: temperature の上限。指定すると実効 temperature をこの値で
        /// 頭打ちにする (未指定リクエストは sampler 既定 1.0 を cap でクランプ)。
        /// 未指定なら従来と完全に同一 (出力ビット不変)。
        /// thinking 有効時に temperature=1 を機械的に送るクライアント
        /// (Hermes 等) で小型モデルが崩壊する対策として 0.7 を推奨
        #[arg(long)] temperature_cap: Option<f32>,
    },
}

/// s156: `denno info` の設定解決結果 (テスト可能にするため main から切り出し)
struct ModelInfo {
    family: String,
    layers: u64,
    hidden: u64,
    vocab: u64,
    context: u64,
}

/// s156: マルチモーダル wrapper の config.json では言語モデルの次元が
/// 入れ子の "text_config" 等に入っている。探索順は text_config / llm_config /
/// language_config / トップレベル。u64 以外 (欠落・0 以外の型) は次へ送る。
const NESTED_CONFIG_KEYS: [&str; 3] = ["text_config", "llm_config", "language_config"];

fn info_lookup_u64(cfg: &serde_json::Value, key: &str) -> Option<u64> {
    NESTED_CONFIG_KEYS
        .iter()
        .filter_map(|k| cfg.get(*k))
        .chain(std::iter::once(cfg))
        .find_map(|obj| obj.get(key).and_then(|v| v.as_u64()))
}

fn info_lookup_str<'a>(cfg: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    NESTED_CONFIG_KEYS
        .iter()
        .filter_map(|k| cfg.get(*k))
        .chain(std::iter::once(cfg))
        .find_map(|obj| obj.get(key).and_then(|v| v.as_str()))
}

/// 既存のフラットな別名 (n_layer / n_embd / n_ctx) も維持する
fn info_dims(cfg: &serde_json::Value) -> ModelInfo {
    ModelInfo {
        family: info_lookup_str(cfg, "model_type").unwrap_or("?").to_string(),
        layers: info_lookup_u64(cfg, "num_hidden_layers")
            .or_else(|| info_lookup_u64(cfg, "n_layer"))
            .unwrap_or(0),
        hidden: info_lookup_u64(cfg, "hidden_size")
            .or_else(|| info_lookup_u64(cfg, "n_embd"))
            .unwrap_or(0),
        vocab: info_lookup_u64(cfg, "vocab_size").unwrap_or(0),
        context: info_lookup_u64(cfg, "max_position_embeddings")
            .or_else(|| info_lookup_u64(cfg, "n_ctx"))
            .unwrap_or(0),
    }
}

/// s156: repo 引数が実在のディレクトリならそこを直接見る (hub cache 不要)。
/// そうでなければ従来どおりの hub cache 経路 (エラー文言も従来どおり)。
fn resolve_model_dir(repo: &str) -> Result<PathBuf> {
    let direct = PathBuf::from(repo);
    if direct.is_dir() {
        return Ok(direct);
    }
    let dir = model_dir(repo);
    if !dir.exists() {
        bail!("not cached: {} (run `denno pull {}` first)", dir.display(), repo);
    }
    Ok(dir)
}

/// s156: `<dir>/config.json` を読む。無い/壊れている場合は panic せず clean な Err。
fn read_config(dir: &std::path::Path) -> Result<serde_json::Value> {
    let cfg_path = dir.join("config.json");
    let file = std::fs::File::open(&cfg_path)
        .with_context(|| format!("config.json を開けません: {}", cfg_path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("config.json を解釈できません: {}", cfg_path.display()))
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
            let dir = resolve_model_dir(&repo)?;
            let cfg = read_config(&dir)?;
            let info = info_dims(&cfg);
            println!("repo:      {}", repo);
            println!("family:    {}", info.family);
            println!("layers:    {}", info.layers);
            println!("hidden:    {}", info.hidden);
            println!("vocab:     {}", info.vocab);
            println!("context:   {}", info.context);
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
        Commands::Serve { repo, host, port, name, api_key, max_tokens: _max_tokens, seed, prefix_cache_tokens, quant, max_queue, repetition_penalty, temperature_cap } => {
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
                temperature_cap,
            }))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// temp_dir 配下に一意なディレクトリを作る (drop で削除)
    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let path = std::env::temp_dir().join(format!(
                "denno_cli_info_{}_{}_{}",
                tag,
                std::process::id(),
                nanos
            ));
            std::fs::create_dir_all(&path).expect("create temp dir");
            TempDir { path }
        }

        fn write_config(&self, body: &str) -> PathBuf {
            let p = self.path.join("config.json");
            let mut f = std::fs::File::create(&p).expect("create config.json");
            f.write_all(body.as_bytes()).expect("write config.json");
            p
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    const QWEN3_5_WRAPPER: &str = r#"{
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "image_token_id": 248056,
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "vocab_size": 248320,
            "max_position_embeddings": 262144
        },
        "vision_config": {"depth": 27, "hidden_size": 1152}
    }"#;

    const FLAT_LLAMA: &str = r#"{
        "model_type": "llama",
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "vocab_size": 128256,
        "max_position_embeddings": 131072
    }"#;

    // a. 入れ子の qwen3_5 形式
    #[test]
    fn nested_text_config_dims() {
        let cfg: serde_json::Value = serde_json::from_str(QWEN3_5_WRAPPER).unwrap();
        let info = info_dims(&cfg);
        assert_eq!(info.family, "qwen3_5_text");
        assert_eq!(info.layers, 64);
        assert_eq!(info.hidden, 5120);
        assert_eq!(info.vocab, 248320);
        assert_eq!(info.context, 262144);
    }

    // b. フラットな llama 形式は従来どおり (別名 n_layer/n_embd/n_ctx も同じ)
    #[test]
    fn flat_llama_dims_unchanged() {
        let cfg: serde_json::Value = serde_json::from_str(FLAT_LLAMA).unwrap();
        let info = info_dims(&cfg);
        assert_eq!(info.family, "llama");
        assert_eq!(info.layers, 32);
        assert_eq!(info.hidden, 4096);
        assert_eq!(info.vocab, 128256);
        assert_eq!(info.context, 131072);

        let alias: serde_json::Value = serde_json::from_str(
            r#"{"model_type":"gpt2","n_layer":12,"n_embd":768,"n_ctx":1024}"#,
        )
        .unwrap();
        let info = info_dims(&alias);
        assert_eq!(info.layers, 12);
        assert_eq!(info.hidden, 768);
        assert_eq!(info.vocab, 0);
        assert_eq!(info.context, 1024);
    }

    // 探索順: text_config が先、無ければ llm_config / language_config、最後にトップレベル
    #[test]
    fn nested_lookup_order() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{
                "llm_config": {"model_type": "llm_x", "num_hidden_layers": 7},
                "language_config": {"model_type": "lang_y", "num_hidden_layers": 9,
                                    "hidden_size": 2048}
            }"#,
        )
        .unwrap();
        let info = info_dims(&cfg);
        assert_eq!(info.family, "llm_x");
        assert_eq!(info.layers, 7);
        assert_eq!(info.hidden, 2048);

        let top: serde_json::Value =
            serde_json::from_str(r#"{"model_type":"top_z","vocab_size":32000}"#).unwrap();
        let info = info_dims(&top);
        assert_eq!(info.family, "top_z");
        assert_eq!(info.vocab, 32000);
    }

    // c. config.json の無いディレクトリ -> clean な Err (panic しない)
    #[test]
    fn dir_without_config_is_err_not_panic() {
        let td = TempDir::new("no_cfg");
        let dir = resolve_model_dir(td.path.to_str().unwrap()).expect("dir must resolve");
        assert_eq!(dir, td.path);
        let err = read_config(&dir).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("config.json"), "unexpected message: {msg}");
    }

    // d. ディレクトリ引数で同じ値が出る (hub cache 不要)。解決自体は resolve_model_dir で確認
    #[test]
    fn directory_argument_resolves_same_values() {
        let td = TempDir::new("dirarg");
        td.write_config(QWEN3_5_WRAPPER);
        let dir = resolve_model_dir(td.path.to_str().unwrap()).expect("dir must resolve");
        let cfg = read_config(&dir).expect("config must parse");
        let info = info_dims(&cfg);
        assert_eq!(info.family, "qwen3_5_text");
        assert_eq!(
            (info.layers, info.hidden, info.vocab, info.context),
            (64, 5120, 248320, 262144)
        );
    }
}
