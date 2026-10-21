//! denno: TensorFold 型の単一バイナリ CLI (pull / models / info / generate / serve)

mod pull;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use denno_engine::kquant::GgmlType;
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
/// s158 (P2-1): 次元は `Option`。未知は `?` と表示し、`0` を事実として出さない
/// (未実装値と本物の 0 を区別できないため)。
struct ModelInfo {
    family: String,
    layers: Option<u64>,
    hidden: Option<u64>,
    vocab: Option<u64>,
    context: Option<u64>,
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

/// s158 (P2-1): 浮動小数のキー (rope_theta / partial_rotary_factor)。
/// 整数で書かれていても `as_f64` で拾う (JSON の 1 と 1.0 を区別しない)。
fn info_lookup_f64(cfg: &serde_json::Value, key: &str) -> Option<f64> {
    NESTED_CONFIG_KEYS
        .iter()
        .filter_map(|k| cfg.get(*k))
        .chain(std::iter::once(cfg))
        .find_map(|obj| obj.get(key).and_then(|v| v.as_f64()))
}

/// s158 (P2-1): 入れ子の探索順で「オブジェクト」を引く (rope_scaling /
/// quantization_config 用)。null や配列・スカラは「無い」と同じ扱い。
fn info_lookup_obj<'a>(cfg: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
    NESTED_CONFIG_KEYS
        .iter()
        .filter_map(|k| cfg.get(*k))
        .chain(std::iter::once(cfg))
        .find_map(|obj| obj.get(key).filter(|v| v.is_object()))
}

/// 既存のフラットな別名 (n_layer / n_embd / n_ctx) も維持する
fn info_dims(cfg: &serde_json::Value) -> ModelInfo {
    ModelInfo {
        family: info_lookup_str(cfg, "model_type").unwrap_or("?").to_string(),
        layers: info_lookup_u64(cfg, "num_hidden_layers")
            .or_else(|| info_lookup_u64(cfg, "n_layer")),
        hidden: info_lookup_u64(cfg, "hidden_size")
            .or_else(|| info_lookup_u64(cfg, "n_embd")),
        vocab: info_lookup_u64(cfg, "vocab_size"),
        context: info_lookup_u64(cfg, "max_position_embeddings")
            .or_else(|| info_lookup_u64(cfg, "n_ctx")),
    }
}

/// s158 (P2-1): config.json 由来の追加表示 (rope / attention / dtype / 量子化)。
/// すべて `Option` のまま持ち、文字列化の段で `?` にする。既定値は捏造しない。
#[derive(Debug, Default, PartialEq)]
struct ModelExtras {
    rope_theta: Option<f64>,
    /// `rope_scaling` がオブジェクトとして存在したか (無い/マップでない = false)
    rope_scaling_present: bool,
    rope_scaling_type: Option<String>,
    rope_scaling_factor: Option<f64>,
    partial_rotary_factor: Option<f64>,
    num_attention_heads: Option<u64>,
    num_key_value_heads: Option<u64>,
    head_dim: Option<u64>,
    torch_dtype: Option<String>,
    /// `quantization_config` が存在したか (無ければ表示は `none`)
    quant_config_present: bool,
    quant_method: Option<String>,
    quant_bits: Option<u64>,
}

/// `rope_theta`。Qwen3.5 などの新しい HF config は同じ事実を
/// `rope_parameters.rope_theta` に置くので、そこも読む (既定値は作らない)。
fn info_rope_theta(cfg: &serde_json::Value) -> Option<f64> {
    info_lookup_f64(cfg, "rope_theta").or_else(|| {
        info_lookup_obj(cfg, "rope_parameters")
            .and_then(|v| v.get("rope_theta"))
            .and_then(|v| v.as_f64())
    })
}

fn info_extras(cfg: &serde_json::Value) -> ModelExtras {
    let rope_scaling = info_lookup_obj(cfg, "rope_scaling");
    let quant = info_lookup_obj(cfg, "quantization_config");
    ModelExtras {
        rope_theta: info_rope_theta(cfg),
        rope_scaling_present: rope_scaling.is_some(),
        // 新しい HF config は同じ事実を "rope_type" に書く (別名として読む)
        rope_scaling_type: rope_scaling
            .and_then(|v| v.get("type").or_else(|| v.get("rope_type")))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        rope_scaling_factor: rope_scaling
            .and_then(|v| v.get("factor"))
            .and_then(|v| v.as_f64()),
        partial_rotary_factor: info_lookup_f64(cfg, "partial_rotary_factor"),
        num_attention_heads: info_lookup_u64(cfg, "num_attention_heads"),
        num_key_value_heads: info_lookup_u64(cfg, "num_key_value_heads"),
        head_dim: info_lookup_u64(cfg, "head_dim"),
        torch_dtype: info_lookup_str(cfg, "torch_dtype").map(str::to_string),
        quant_config_present: quant.is_some(),
        quant_method: quant
            .and_then(|v| v.get("quant_method"))
            .and_then(|v| v.as_str())
            .map(str::to_string),
        quant_bits: quant.and_then(|v| v.get("bits")).and_then(|v| v.as_u64()),
    }
}

/// 表示ラベルの幅 (値の開始桁を揃える)。ラベルはコロン込みで渡す。
const INFO_LABEL_WIDTH: usize = 13;

fn info_line(label: &str, value: impl std::fmt::Display) -> String {
    format!("{:<w$}{}", label, value, w = INFO_LABEL_WIDTH)
}

/// 未知は `?`。`0` を事実として出さない。
fn info_fmt_u64(v: Option<u64>) -> String {
    match v {
        Some(x) => x.to_string(),
        None => "?".to_string(),
    }
}

/// 未知・非有限は `?`。整数値は小数点を付けない (1e6 は "1000000")。
fn info_fmt_f64(v: Option<f64>) -> String {
    match v {
        Some(x) if x.is_finite() => format!("{x}"),
        _ => "?".to_string(),
    }
}

fn info_fmt_str(v: Option<&str>) -> String {
    match v {
        Some(s) => s.to_string(),
        None => "?".to_string(),
    }
}

/// `rope_scaling` の表示。マップでない/欠落は `?`、マップなら type と factor
/// (片方しか無ければその片方が `?`)。
fn info_fmt_rope_scaling(e: &ModelExtras) -> String {
    if !e.rope_scaling_present {
        return "?".to_string();
    }
    if e.rope_scaling_type.is_none() && e.rope_scaling_factor.is_none() {
        return "?".to_string();
    }
    format!(
        "{} factor={}",
        info_fmt_str(e.rope_scaling_type.as_deref()),
        info_fmt_f64(e.rope_scaling_factor)
    )
}

/// `quantization_config` が無い = 「量子化設定なし」なので `none`
/// (キーはあるが値が読めない場合は `?`)。
fn info_fmt_quant_method(e: &ModelExtras) -> String {
    if !e.quant_config_present {
        return "none".to_string();
    }
    info_fmt_str(e.quant_method.as_deref())
}

fn info_fmt_quant_bits(e: &ModelExtras) -> String {
    if !e.quant_config_present {
        return "none".to_string();
    }
    info_fmt_u64(e.quant_bits)
}

/// config.json から出せる表示行 (weights / gguf の行は main 側で足す)。
fn info_config_lines(repo: &str, info: &ModelInfo, extras: &ModelExtras) -> Vec<String> {
    vec![
        info_line("repo:", repo),
        info_line("family:", &info.family),
        info_line("layers:", info_fmt_u64(info.layers)),
        info_line("hidden:", info_fmt_u64(info.hidden)),
        info_line("vocab:", info_fmt_u64(info.vocab)),
        info_line("context:", info_fmt_u64(info.context)),
        info_line("rope_theta:", info_fmt_f64(extras.rope_theta)),
        info_line("rope_scaling:", info_fmt_rope_scaling(extras)),
        info_line("part_rotary:", info_fmt_f64(extras.partial_rotary_factor)),
        info_line("attn_heads:", info_fmt_u64(extras.num_attention_heads)),
        info_line("kv_heads:", info_fmt_u64(extras.num_key_value_heads)),
        info_line("head_dim:", info_fmt_u64(extras.head_dim)),
        info_line("torch_dtype:", info_fmt_str(extras.torch_dtype.as_deref())),
        info_line("quant_method:", info_fmt_quant_method(extras)),
        info_line("quant_bits:", info_fmt_quant_bits(extras)),
    ]
}

/// s158 (P2-1): ggml type の人間可読名。数値 id は `ggml_type_id`。
fn ggml_type_name(ty: GgmlType) -> &'static str {
    match ty {
        GgmlType::F32 => "F32",
        GgmlType::F16 => "F16",
        GgmlType::Q4_0 => "Q4_0",
        GgmlType::Q4_1 => "Q4_1",
        GgmlType::Q5_0 => "Q5_0",
        GgmlType::Q5_1 => "Q5_1",
        GgmlType::Q8_0 => "Q8_0",
        GgmlType::Q2K => "Q2_K",
        GgmlType::Q3K => "Q3_K",
        GgmlType::Q4K => "Q4_K",
        GgmlType::Q5K => "Q5_K",
        GgmlType::Q6K => "Q6_K",
        GgmlType::IQ2Xxs => "IQ2_XXS",
        GgmlType::IQ2Xs => "IQ2_XS",
        GgmlType::IQ3Xxs => "IQ3_XXS",
        GgmlType::IQ1S => "IQ1_S",
        GgmlType::IQ4Nl => "IQ4_NL",
        GgmlType::IQ3S => "IQ3_S",
        GgmlType::IQ2S => "IQ2_S",
        GgmlType::IQ4Xs => "IQ4_XS",
        GgmlType::Iq1M => "IQ1_M",
        GgmlType::Bf16 => "BF16",
        GgmlType::Tq1_0 => "TQ1_0",
        GgmlType::Mxfp4 => "MXFP4",
    }
}

/// s158 (P2-1): ggml type の数値 id (`ggml.h` の enum 値そのまま)。
/// `GgmlType::from_id` と往復することをテストで固定する (表の二重管理よけ)。
fn ggml_type_id(ty: GgmlType) -> u32 {
    match ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Q4_0 => 2,
        GgmlType::Q4_1 => 3,
        GgmlType::Q5_0 => 6,
        GgmlType::Q5_1 => 7,
        GgmlType::Q8_0 => 8,
        GgmlType::Q2K => 10,
        GgmlType::Q3K => 11,
        GgmlType::Q4K => 12,
        GgmlType::Q5K => 13,
        GgmlType::Q6K => 14,
        GgmlType::IQ2Xxs => 16,
        GgmlType::IQ2Xs => 17,
        GgmlType::IQ3Xxs => 18,
        GgmlType::IQ1S => 19,
        GgmlType::IQ4Nl => 20,
        GgmlType::IQ3S => 21,
        GgmlType::IQ2S => 22,
        GgmlType::IQ4Xs => 23,
        GgmlType::Iq1M => 29,
        GgmlType::Bf16 => 30,
        GgmlType::Tq1_0 => 34,
        GgmlType::Mxfp4 => 39,
    }
}

/// ディレクトリ直下の `*.gguf`。複数あるときは名前順の先頭 (実行ごとに変えない)。
fn find_gguf(dir: &std::path::Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .and_then(|x| x.to_str())
                    .map_or(false, |x| x.eq_ignore_ascii_case("gguf"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// テンソル型の最頻値 (型, 個数)。同数は ggml type id の小さい方 (決定論)。
fn most_common_type(types: impl Iterator<Item = GgmlType>) -> Option<(GgmlType, usize)> {
    let mut counts: std::collections::HashMap<u32, (GgmlType, usize)> =
        std::collections::HashMap::new();
    for ty in types {
        let e = counts.entry(ggml_type_id(ty)).or_insert((ty, 0));
        e.1 += 1;
    }
    counts
        .into_values()
        .max_by_key(|(ty, n)| (*n, std::cmp::Reverse(ggml_type_id(*ty))))
}

/// gguf を開いてテンソル型の最頻値を返す。開けない・テンソル無しは None。
fn gguf_common_type(path: &std::path::Path) -> Option<(GgmlType, usize)> {
    let f = denno_engine::gguf::GgufFile::open(path).ok()?;
    let types = f
        .tensor_order
        .iter()
        .filter_map(|n| f.tensors.get(n).map(|t| t.ty));
    most_common_type(types)
}

/// `quant:` の値。開けない/読めないときは `?` (panic しない)。
fn gguf_quant_value(path: &std::path::Path) -> String {
    match gguf_common_type(path) {
        Some((ty, _count)) => format!("{} ({})", ggml_type_name(ty), ggml_type_id(ty)),
        None => "?".to_string(),
    }
}

/// GGUF があるディレクトリでだけ `quant:` 行を返す。無ければ None = 行を出さない。
fn info_quant_line(dir: &std::path::Path) -> Option<String> {
    find_gguf(dir).map(|p| info_line("quant:", gguf_quant_value(&p)))
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
            let extras = info_extras(&cfg);
            for line in info_config_lines(&repo, &info, &extras) {
                println!("{line}");
            }
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
                    println!(
                        "{}",
                        info_line(
                            "weights:",
                            format!("{:.1} MB ({} shard(s))", total as f64 / 1e6, files.len())
                        )
                    );
                }
            } else if st_path.exists() {
                let bytes = std::fs::metadata(&st_path)?.len();
                println!(
                    "{}",
                    info_line("weights:", format!("{:.1} MB", bytes as f64 / 1e6))
                );
            }
            // s158 (P2-1): GGUF があるときだけ 1 行 (無ければ行自体を出さない)
            if let Some(line) = info_quant_line(&dir) {
                println!("{line}");
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

        /// s158: 任意のファイル (gguf のダミー等) を置く
        fn write_file(&self, name: &str, bytes: &[u8]) -> PathBuf {
            let p = self.path.join(name);
            let mut f = std::fs::File::create(&p).expect("create file");
            f.write_all(bytes).expect("write file");
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
        assert_eq!(info.layers, Some(64));
        assert_eq!(info.hidden, Some(5120));
        assert_eq!(info.vocab, Some(248320));
        assert_eq!(info.context, Some(262144));
    }

    // b. フラットな llama 形式は従来どおり (別名 n_layer/n_embd/n_ctx も同じ)
    #[test]
    fn flat_llama_dims_unchanged() {
        let cfg: serde_json::Value = serde_json::from_str(FLAT_LLAMA).unwrap();
        let info = info_dims(&cfg);
        assert_eq!(info.family, "llama");
        assert_eq!(info.layers, Some(32));
        assert_eq!(info.hidden, Some(4096));
        assert_eq!(info.vocab, Some(128256));
        assert_eq!(info.context, Some(131072));

        let alias: serde_json::Value = serde_json::from_str(
            r#"{"model_type":"gpt2","n_layer":12,"n_embd":768,"n_ctx":1024}"#,
        )
        .unwrap();
        let info = info_dims(&alias);
        assert_eq!(info.layers, Some(12));
        assert_eq!(info.hidden, Some(768));
        // s158: 未知は None (以前は 0 を事実として表示していた)
        assert_eq!(info.vocab, None);
        assert_eq!(info.context, Some(1024));
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
        assert_eq!(info.layers, Some(7));
        assert_eq!(info.hidden, Some(2048));

        let top: serde_json::Value =
            serde_json::from_str(r#"{"model_type":"top_z","vocab_size":32000}"#).unwrap();
        let info = info_dims(&top);
        assert_eq!(info.family, "top_z");
        assert_eq!(info.vocab, Some(32000));
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
            (Some(64), Some(5120), Some(248320), Some(262144))
        );
    }

    // ---- s158 (P2-1): 追加表示 (rope / attention / dtype / 量子化 / gguf) ----

    /// 表示行からラベルを除いた値だけを取り出す
    fn line_value(lines: &[String], label: &str) -> String {
        lines
            .iter()
            .find(|l| l.starts_with(label))
            .unwrap_or_else(|| panic!("no line for {label}: {lines:?}"))
            .trim_start_matches(label)
            .trim()
            .to_string()
    }

    /// 入れ子 text_config に rope / attention / dtype が入っている形
    const ROPE_WRAPPER: &str = r#"{
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 64,
            "hidden_size": 5120,
            "rope_theta": 10000000.0,
            "rope_scaling": {"type": "linear", "factor": 4.0},
            "partial_rotary_factor": 0.25,
            "num_attention_heads": 40,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "torch_dtype": "bfloat16"
        }
    }"#;

    // a'. 入れ子 text_config から rope / attention / dtype が解決できる
    #[test]
    fn nested_text_config_rope_and_attention() {
        let cfg: serde_json::Value = serde_json::from_str(ROPE_WRAPPER).unwrap();
        let extras = info_extras(&cfg);
        assert_eq!(extras.rope_theta, Some(10000000.0));
        assert!(extras.rope_scaling_present);
        assert_eq!(extras.rope_scaling_type.as_deref(), Some("linear"));
        assert_eq!(extras.rope_scaling_factor, Some(4.0));
        assert_eq!(extras.partial_rotary_factor, Some(0.25));
        assert_eq!(extras.num_attention_heads, Some(40));
        assert_eq!(extras.num_key_value_heads, Some(8));
        assert_eq!(extras.head_dim, Some(128));
        assert_eq!(extras.torch_dtype.as_deref(), Some("bfloat16"));

        let info = info_dims(&cfg);
        let lines = info_config_lines("local/x", &info, &extras);
        assert_eq!(line_value(&lines, "rope_theta:"), "10000000");
        assert_eq!(line_value(&lines, "rope_scaling:"), "linear factor=4");
        assert_eq!(line_value(&lines, "part_rotary:"), "0.25");
        assert_eq!(line_value(&lines, "attn_heads:"), "40");
        assert_eq!(line_value(&lines, "kv_heads:"), "8");
        assert_eq!(line_value(&lines, "head_dim:"), "128");
        assert_eq!(line_value(&lines, "torch_dtype:"), "bfloat16");
        assert_eq!(line_value(&lines, "layers:"), "64");
    }

    // a''. 新しい HF config の `rope_parameters.rope_theta` からも theta を読む
    // (同じ事実の別の置き場所。無ければ '?' のまま)
    #[test]
    fn rope_theta_from_rope_parameters() {
        let cfg: serde_json::Value = serde_json::from_str(
            r#"{"text_config":{"rope_parameters":{"rope_type":"default","rope_theta":10000000}}}"#,
        )
        .unwrap();
        let extras = info_extras(&cfg);
        assert_eq!(extras.rope_theta, Some(10000000.0));
        // rope_parameters は rope_scaling ではないので、その行は '?' のまま
        assert!(!extras.rope_scaling_present);
        let lines = info_config_lines("local/z", &info_dims(&cfg), &extras);
        assert_eq!(line_value(&lines, "rope_theta:"), "10000000");
        assert_eq!(line_value(&lines, "rope_scaling:"), "?");

        // どちらも無ければ '?'
        let none: serde_json::Value = serde_json::from_str(r#"{"model_type":"x"}"#).unwrap();
        assert_eq!(info_extras(&none).rope_theta, None);
    }

    // b'. 未知のキーは '?' (0 ではない)。quantization_config が無ければ 'none'
    #[test]
    fn unknown_keys_print_question_mark_not_zero() {
        let cfg: serde_json::Value =
            serde_json::from_str(r#"{"model_type":"mystery"}"#).unwrap();
        let info = info_dims(&cfg);
        assert_eq!(info.layers, None);
        let extras = info_extras(&cfg);
        let lines = info_config_lines("local/y", &info, &extras);

        for label in [
            "layers:",
            "hidden:",
            "vocab:",
            "context:",
            "rope_theta:",
            "rope_scaling:",
            "part_rotary:",
            "attn_heads:",
            "kv_heads:",
            "head_dim:",
            "torch_dtype:",
        ] {
            assert_eq!(line_value(&lines, label), "?", "{label} must be '?'");
        }
        // 量子化は「設定が無い」= none (未知の '?' とは別の事実)
        assert_eq!(line_value(&lines, "quant_method:"), "none");
        assert_eq!(line_value(&lines, "quant_bits:"), "none");
        // 0 を事実として出していないこと
        for line in &lines {
            assert!(!line.trim_end().ends_with(" 0"), "unexpected 0: {line}");
        }
    }

    // c'. 量子化設定がある場合 / 無い場合 / あるが中身が読めない場合
    #[test]
    fn quantization_config_present_and_absent() {
        let present: serde_json::Value = serde_json::from_str(
            r#"{"model_type":"q","quantization_config":{"quant_method":"gptq","bits":4}}"#,
        )
        .unwrap();
        let extras = info_extras(&present);
        assert!(extras.quant_config_present);
        assert_eq!(extras.quant_method.as_deref(), Some("gptq"));
        assert_eq!(extras.quant_bits, Some(4));
        let lines = info_config_lines("local/q", &info_dims(&present), &extras);
        assert_eq!(line_value(&lines, "quant_method:"), "gptq");
        assert_eq!(line_value(&lines, "quant_bits:"), "4");

        // 入れ子 (text_config の中) でも同じ探索順で拾う
        let nested: serde_json::Value = serde_json::from_str(
            r#"{"text_config":{"quantization_config":{"quant_method":"awq","bits":8}}}"#,
        )
        .unwrap();
        let extras = info_extras(&nested);
        assert_eq!(extras.quant_method.as_deref(), Some("awq"));
        assert_eq!(extras.quant_bits, Some(8));

        // 無い場合は none (どちらの行も)
        let absent: serde_json::Value = serde_json::from_str(r#"{"model_type":"q"}"#).unwrap();
        let extras = info_extras(&absent);
        assert!(!extras.quant_config_present);
        let lines = info_config_lines("local/q", &info_dims(&absent), &extras);
        assert_eq!(line_value(&lines, "quant_method:"), "none");
        assert_eq!(line_value(&lines, "quant_bits:"), "none");

        // あるが中身が読めない場合は '?'
        let empty: serde_json::Value =
            serde_json::from_str(r#"{"quantization_config":{}}"#).unwrap();
        let extras = info_extras(&empty);
        let lines = info_config_lines("local/q", &info_dims(&empty), &extras);
        assert_eq!(line_value(&lines, "quant_method:"), "?");
        assert_eq!(line_value(&lines, "quant_bits:"), "?");
    }

    // d'. gguf の無いディレクトリでは quant 行を出さない (panic もしない)
    #[test]
    fn dir_without_gguf_has_no_quant_line() {
        let td = TempDir::new("no_gguf");
        td.write_config(FLAT_LLAMA);
        assert!(find_gguf(&td.path).is_none());
        assert!(info_quant_line(&td.path).is_none());
    }

    // gguf があっても開けなければ '?' (panic しない)。拡張子の大小は問わない
    #[test]
    fn unreadable_gguf_prints_question_mark() {
        let td = TempDir::new("bad_gguf");
        let p = td.write_file("broken.GGUF", b"not a gguf at all");
        assert_eq!(find_gguf(&td.path).as_deref(), Some(p.as_path()));
        assert_eq!(gguf_quant_value(&p), "?");
        let line = info_quant_line(&td.path).expect("gguf exists so the line exists");
        assert_eq!(line_value(&[line], "quant:"), "?");
    }

    // 最頻値の選択 (同数は型 id の小さい方 = 決定論)
    #[test]
    fn most_common_type_picks_mode_with_deterministic_tie_break() {
        use GgmlType::{F16, Q4K, Q4_0, Q8_0};
        assert_eq!(most_common_type(vec![Q4_0, Q8_0, Q4_0].into_iter()), Some((Q4_0, 2)));
        // 1 : 1 の同数 -> 小さい id (Q4_0 = 2) を選ぶ
        assert_eq!(most_common_type(vec![Q8_0, Q4_0].into_iter()), Some((Q4_0, 1)));
        assert_eq!(most_common_type(Vec::<GgmlType>::new().into_iter()), None);
        assert_eq!(most_common_type(vec![F16, Q4K, Q4K].into_iter()), Some((Q4K, 2)));
    }

    // 名前と数値 id の表が GgmlType::from_id と一致すること (数値の出所は ggml.h)
    #[test]
    fn ggml_type_ids_roundtrip() {
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::IQ2Xxs,
            GgmlType::IQ2Xs,
            GgmlType::IQ3Xxs,
            GgmlType::IQ1S,
            GgmlType::IQ4Nl,
            GgmlType::IQ3S,
            GgmlType::IQ2S,
            GgmlType::IQ4Xs,
            GgmlType::Iq1M,
            GgmlType::Bf16,
            GgmlType::Tq1_0,
            GgmlType::Mxfp4,
        ] {
            let id = ggml_type_id(ty);
            assert_eq!(GgmlType::from_id(id), Some(ty), "id {id} for {ty:?}");
            assert!(!ggml_type_name(ty).is_empty());
        }
        // K1 で確定した値 (ggml.h) をこの経路でも固定する
        assert_eq!(ggml_type_id(GgmlType::Iq1M), 29);
        assert_eq!(ggml_type_id(GgmlType::Tq1_0), 34);
        assert_eq!(ggml_type_id(GgmlType::Q4_0), 2);
        assert_eq!(ggml_type_name(GgmlType::Q4_0), "Q4_0");
    }
}
