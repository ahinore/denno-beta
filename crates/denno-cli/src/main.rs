//! denno: TensorFold 型の単一バイナリ CLI (pull / models / info / generate / serve)

mod abliterate;
mod bench;
mod chat;
mod config;
mod doctor;
mod flash_telemetry;
mod inspect;
mod pull;
mod serve_flash;
mod tokenize;

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
    /// Qwen3.8-Flash-Next (arch qwen4exp) の最小サーバー。
    /// 27B 用の `serve` とは別経路 (Engine 実装を持たない別アーキのため)。
    ServeFlash {
        /// 分割 GGUF のあるディレクトリ (未指定なら `DENNO_FLASH_DIR` /
        /// 設定ファイルの `dir`)
        #[arg(long)]
        dir: Option<PathBuf>,
        /// 待ち受けアドレス (既定 127.0.0.1)
        #[arg(long)]
        host: Option<String>,
        /// 待ち受けポート (既定 8098)
        #[arg(long)]
        port: Option<u16>,
        /// コンテキスト長 (KV / GDN 状態の容量、既定 4096)
        #[arg(long)]
        max_ctx: Option<usize>,
        /// クライアントに広告するモデルID (既定 qwen3.8-flash-next)
        #[arg(long)]
        name: Option<String>,
        /// 1 リクエストの生成上限 (max_tokens 未指定時の既定でもある。既定 512)
        #[arg(long)]
        max_tokens: Option<usize>,
        /// task 796: MTP (nextn) head の safetensors。未指定なら
        /// 環境変数 `DENNO_MTP_PATH` を見る。どちらも無ければ従来どおり
        /// (MTP を読まない) 起動する。config.json は同じディレクトリに置く。
        #[arg(long)]
        mtp_path: Option<PathBuf>,
        /// P812: 文脈 (KV / GDN) スロット数。既定 2 (従来の main + aux)。
        /// 3 以上で LCP 一致 + LRU の汎用 prefix cache。未指定なら
        /// 環境変数 `DENNO_KV_SLOTS`、それも無ければ 2。
        #[arg(long)]
        kv_slots: Option<usize>,
        /// task 833: 起動時に掛けるステアリングのレシピ (.json)。未指定なら
        /// 環境変数 `DENNO_STEER_RECIPE` を見る。どちらも無ければ何もしない
        /// (既定の推論は不変)。
        #[arg(long)]
        steer_recipe: Option<PathBuf>,
        /// P818: カードの VRAM から `DENNO_*` の既定を選ぶ。`auto` で起動時に
        /// VRAM / RAM / モデルサイズを引き、未設定の env だけを埋める
        /// (明示 env は常に勝つ)。数値 (MiB) を渡すと総 VRAM を上書きする。
        /// 未指定なら環境変数 `DENNO_AUTO_CONFIG`。
        #[arg(long)]
        vram_budget: Option<String>,
        /// task 906: TOML 設定ファイル。解決順は
        /// CLI 引数 > `DENNO_*` 環境変数 > 設定ファイル > 既定値。
        #[arg(long, value_name = "PATH")]
        config: Option<PathBuf>,
        /// task 906: 解決結果を表示して終了する (サーバーは起動しない)。
        #[arg(long)]
        print_config: bool,
    },
    /// task 861: 起動中のサーバーと話す対話 REPL (`/help` で一覧)。
    Chat {
        /// ベース URL
        #[arg(long, default_value = "http://127.0.0.1:8098")]
        url: String,
        /// 話す API (anthropic = /v1/messages, openai = /v1/chat/completions)
        #[arg(long, default_value = "anthropic")]
        api: String,
        /// system プロンプト
        #[arg(long)]
        system: Option<String>,
        /// 1 ターンの生成上限
        #[arg(long, default_value_t = 512)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// モデル ID (未指定なら GET /v1/models の先頭)
        #[arg(long)]
        model: Option<String>,
        /// x-api-key / Bearer で送る API キー (サーバーが要求する場合)
        #[arg(long)]
        api_key: Option<String>,
        /// HTTP 読み取りタイムアウト秒
        #[arg(long, default_value_t = 600)]
        timeout: u64,
    },
    /// task 865: A/B 残差の記録 → 分離チャート → `.dns`/レシピ書き出し → OFF/ON 差分評価。
    /// 起動中のサーバーに HTTP で指示する (推論の既定経路は変えない)。
    Abliterate {
        /// ベース URL
        #[arg(long, default_value = "http://127.0.0.1:8098")]
        url: String,
        /// positive (消したい概念) のプロンプト集 (1 行 1 本、# 始まりは無視)
        #[arg(long)]
        pos_file: PathBuf,
        /// negative のプロンプト集
        #[arg(long)]
        neg_file: PathBuf,
        /// OFF/ON の差分評価に使うプロンプト集 (省略なら評価しない)
        #[arg(long)]
        eval_file: Option<PathBuf>,
        /// 残差の記録モード (all = 全トークン平均 / last_prompt_token = prefill 末尾)
        #[arg(long, default_value = "all")]
        capture_mode: String,
        /// 適用係数
        #[arg(long, default_value_t = 1.0)]
        alpha: f32,
        /// `.dns` の書き出し先ディレクトリ
        #[arg(long, default_value = ".")]
        out_dir: PathBuf,
        /// レシピ JSON の書き出し先
        #[arg(long)]
        out_recipe: Option<PathBuf>,
        /// スナップショット名の接頭辞 (`<tag>_pos` / `<tag>_neg`)
        #[arg(long, default_value = "ablit")]
        tag: String,
        /// モデル ID (未指定なら GET /v1/models の先頭)
        #[arg(long)]
        model: Option<String>,
        /// 差分評価の生成上限
        #[arg(long, default_value_t = 32)]
        eval_max_tokens: usize,
        /// レシピに `.dns` を埋め込む (持ち運び用)
        #[arg(long)]
        embed_snapshots: bool,
        /// HTTP 読み取りタイムアウト秒
        #[arg(long, default_value_t = 600)]
        timeout: u64,
        /// 引数と入力ファイルだけ検証し、サーバーへは何も送らない
        #[arg(long)]
        dry_run: bool,
    },
    /// P801: 起動中のサーバーに対する再現可能ベンチ (llama-bench 風 pp/tg)。
    /// プロンプトはサーバーの usage.input_tokens で目標トークン数へ較正する。
    Bench {
        /// ベンチ対象のベース URL
        #[arg(long, default_value = "http://127.0.0.1:8098")]
        url: String,
        /// 話す API (anthropic = /v1/messages, openai = /v1/chat/completions)
        #[arg(long, default_value = "anthropic")]
        api: String,
        /// pp (prompt processing) のトークン数。カンマ区切り
        #[arg(long, value_delimiter = ',', default_value = "512,2048,8192")]
        pp: Vec<usize>,
        /// tg (生成) のトークン数
        #[arg(long, default_value_t = 128)]
        tg: usize,
        /// 文脈深さ (プロンプトのトークン数)。カンマ区切り
        #[arg(long, value_delimiter = ',', default_value = "0,2048,8192")]
        depths: Vec<usize>,
        /// 各ケースの反復回数 (中央値を採る)
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        /// JSON レポートの書き出し先
        #[arg(long)]
        json: Option<PathBuf>,
        /// Markdown サマリを表示する (既定 on。--json 併用時の明示用)
        #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
        markdown: bool,
        /// ラウンド間の KV/prefix cache 押し流しを行わない
        #[arg(long)]
        no_flush: bool,
        /// 押し流しプロンプトのトークン数 (0 = 自動: コンテキスト長 - 64)
        #[arg(long, default_value_t = 0)]
        flush_tokens: usize,
        /// プロンプト生成と較正の基準 seed
        #[arg(long, default_value_t = 801)]
        seed: u64,
        /// 決定論チェックの生成トークン数
        #[arg(long, default_value_t = 16)]
        det_tokens: usize,
        /// turn-2 KV reuse を測る文脈長
        #[arg(long, default_value_t = 2048)]
        reuse_depth: usize,
        /// HTTP 読み取りタイムアウト秒
        #[arg(long, default_value_t = 600)]
        timeout: u64,
        /// P897: 2 つの JSON レポートを比較して差分表を出す (ベンチは実行しない)。
        /// 使い方: --compare base.json new.json
        #[arg(long, num_args = 2, value_names = ["BASE", "NEW"])]
        compare: Option<Vec<PathBuf>>,
        /// P897: --compare 時に、この % を超える悪化を退行とみなす非 0 で終わる
        #[arg(long)]
        fail_on_regress_pct: Option<f64>,
        /// task 1021: --compare の要約を JSON で書き出す (`-` は stdout)
        #[arg(long, value_name = "PATH")]
        compare_json: Option<PathBuf>,
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
        /// task 951: 複数の API キー (カンマ区切り)。`--api-key` と
        /// `DENNO_API_KEYS` の和が受理される (どれか 1 つに一致すれば通る)。
        /// 未指定かつ他も空なら従来どおり認証なし。
        #[arg(long, value_name = "KEYS")] api_keys: Option<String>,
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
        /// 同時受け付けがこれを超えると過負荷応答 (Anthropic 503 /
        /// OpenAI 429 / Ollama 503、いずれも Retry-After 付き) を返す。
        /// 未指定時は DENNO_MAX_QUEUE_DEPTH、それも無ければ 32
        #[arg(long = "max-queue-depth", visible_alias = "max-queue", value_name = "N")]
        max_queue_depth: Option<usize>,
        /// P873: リクエストの壁時計上限 (秒)。0 = 無効 (既定)。
        /// 超えたリクエストは生成を打ち切り、KV スロットを解放してエラーを返す。
        /// 未指定時は DENNO_REQUEST_TIMEOUT_SECS
        #[arg(long, value_name = "SECS")]
        request_timeout_secs: Option<u64>,
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
        /// P877: アクセスログの形式。`text` (既定) は従来どおり 2 行、
        /// `json` は 1 リクエスト 1 行の JSON (`ts_ms` / `method` / `path` /
        /// `status` / `duration_ms`)。
        #[arg(long, default_value = "text", value_parser = ["text", "json"])]
        log_format: String,
        /// task 916: CORS の許可オリジン。`*` (任意) またはカンマ区切りの
        /// オリジン一覧 (例 `https://app.example.com,http://localhost:5173`)。
        /// 未指定時は DENNO_CORS_ORIGIN、それも無ければループバック
        /// (localhost / 127.0.0.1 / [::1]) の任意ポートだけを許可する。
        #[arg(long, value_name = "ORIGINS")]
        cors_origin: Option<String>,
        /// task 946: HTTP リクエストボディの上限 (MiB)。上限を超えると
        /// 413 + JSON エラー封筒で返す。未指定時は DENNO_MAX_BODY_MB、
        /// それも無ければ 16 MiB。
        #[arg(long, value_name = "MB", value_parser = clap::value_parser!(u64).range(1..))]
        max_body_mb: Option<u64>,
    },
    /// task 921: GGUF のヘッダだけを読み、アーキ情報 / 量子型ヒストグラム /
    /// VRAM 見積りを表示する (重みは読まない)。
    Inspect {
        /// 単一 or 分割 GGUF の 1 パート (分割は各パートのヘッダを見る)
        path: PathBuf,
        /// 機械可読な JSON で出力する
        #[arg(long)]
        json: bool,
        /// dense + expert キャッシュ見積りで expert に許す MiB (既定 4096)
        #[arg(long, default_value_t = inspect::DEFAULT_EXPERT_CACHE_MIB)]
        expert_cache_mib: u64,
    },
    /// task 941: GGUF のトークナイザ・メタデータだけで文字列をトークン化する
    /// (重みは読まない)。トークン数 / id / 復号した断片を出す。
    Tokenize {
        /// 単一の GGUF か、分割 GGUF のあるディレクトリ
        #[arg(long)]
        dir: PathBuf,
        /// トークン化する文字列 (空文字列 `""` も指定できる)
        #[arg(long, conflicts_with = "stdin")]
        text: Option<String>,
        /// 標準入力から UTF-8 のテキストを読む
        #[arg(long)]
        stdin: bool,
        /// 機械可読な JSON で出力する
        #[arg(long)]
        json: bool,
    },
    /// task 881: 環境とハードウェアの自己診断 (CPU SIMD / RAM / CUDA /
    /// Level Zero / GGUF の欠け)。GPU は実行しない (存在と /proc だけ見る)。
    Doctor {
        /// 分割 GGUF のあるディレクトリ。省略するとモデル検査をしない
        #[arg(long)]
        dir: Option<PathBuf>,
        /// 機械可読な JSON で出力する (自動化向け)
        #[arg(long)]
        json: bool,
    },
}

/// s156: `denno info` の設定解決結果 (テスト可能にするため main から切り出し)
/// s158 (P2-1): 次元は `Option`。未知は `?` と表示し、`0` を事実として出さない
/// (未実装値と本物の 0 を区別できないため)。
/// s164 (K5): `Debug` はテストの `unwrap_err` 用 (表示には使わない)。
#[derive(Debug)]
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
        GgmlType::Q2_0 => "Q2_0",
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
        GgmlType::Q2_0 => 42,
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

/// s168 (P7): テンソル型の「支配的な型」。
///
/// F32 は量子化されない置き場 (rmsnorm / bias) で、実ファイルでは本数が量子化された
/// 重みを上回る。Qwen3.8-27B-Q4_0.gguf の実測 (ヘッダを独立に読んだ本数) は
/// F32 456 / Q4_0 352 / Q5K 48 / Q4_1 8 / Q6K 1 / Q8_0 1 で、全テンソルを投票すると
/// 「F32 (0)」になる。量子化の種類を聞かれている行なので、F32 以外が 1 本でもあれば
/// F32 を投票から外す。F32 しか無いファイル (本当に F32 の GGUF) は F32 を返す。
fn dominant_tensor_type(types: impl Iterator<Item = GgmlType>) -> Option<(GgmlType, usize)> {
    let all: Vec<GgmlType> = types.collect();
    let voted: Vec<GgmlType> = if all.iter().any(|t| *t != GgmlType::F32) {
        all.into_iter().filter(|t| *t != GgmlType::F32).collect()
    } else {
        all
    };
    most_common_type(voted.into_iter())
}

/// gguf を開いてテンソル型の最頻値を返す。開けない・テンソル無しは None。
fn gguf_common_type(path: &std::path::Path) -> Option<(GgmlType, usize)> {
    let f = denno_engine::gguf::GgufFile::open(path).ok()?;
    let types = f
        .tensor_order
        .iter()
        .filter_map(|n| f.tensors.get(n).map(|t| t.ty));
    dominant_tensor_type(types)
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

/// s164 (K5): GGUF のメタデータから次元を読む (config.json が無い/キーが欠けたとき用)。
/// arch 名は `general.architecture` の値から組み立てる (`qwen` 等を決め打ちしない)。
/// 開けない・arch が無い・値が 0 のものは None のまま (事実として 0 を出さない)。
///
/// s169 (P19): layers は推論トランクの層数にする。GGUF の `<arch>.block_count` は
/// MTP (nextn) 層を含み得る (27B: block_count=65 = trunk 64 + nextn 1)。エンジンは
/// config の `num_hidden_layers` を正として超過分の blk.{n_layer}.. を nextn 層として
/// 読み飛ばす (model/qwen35.rs の層数整合確認) ので、config.json が無いときの表示も
/// 同じ層数 = `block_count - nextn_predict_layers` にする。キーが無ければ block_count
/// そのまま。差し引きが 0 以下になる異常値は None (`?`) にする (0 を事実にしない)。
fn gguf_dims(dir: &std::path::Path) -> ModelInfo {
    let empty = ModelInfo {
        family: "?".to_string(),
        layers: None,
        hidden: None,
        vocab: None,
        context: None,
    };
    let Some(path) = find_gguf(dir) else {
        return empty;
    };
    let Ok(f) = denno_engine::gguf::GgufFile::open(&path) else {
        return empty;
    };
    let Some(arch) = f.get("general.architecture").and_then(|v| v.as_str()) else {
        return empty;
    };
    let dim = |suffix: &str| {
        f.get(&format!("{arch}.{suffix}"))
            .and_then(|v| v.as_u64())
            .filter(|n| *n > 0)
    };
    let layers = dim("block_count").and_then(|block_count| {
        let nextn = f
            .get(&format!("{arch}.nextn_predict_layers"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        block_count.checked_sub(nextn).filter(|n| *n > 0)
    });
    ModelInfo {
        // family は config.json からのみ (GGUF の arch 名は family ではない)
        family: "?".to_string(),
        layers,
        hidden: dim("embedding_length"),
        context: dim("context_length"),
        // vocab はトークン表の配列長。無ければ `<arch>.vocab_size`
        vocab: f
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array_len())
            .map(|n| n as u64)
            .filter(|n| *n > 0)
            .or_else(|| dim("vocab_size")),
    }
}

/// s164 (K5): `denno info` の次元解決。config.json があればそれが優先で、
/// 欠けている次元だけ GGUF のメタデータで埋める (同じ次元を GGUF で上書きしない)。
/// config.json が読めず、GGUF からも次元が取れないときは `read_config` の Err を
/// そのまま返す (Err 契約・文言は従来どおり)。
fn info_resolve(dir: &std::path::Path) -> Result<(ModelInfo, ModelExtras)> {
    let gguf = gguf_dims(dir);
    match read_config(dir) {
        Ok(cfg) => {
            let from_cfg = info_dims(&cfg);
            let merged = ModelInfo {
                family: from_cfg.family,
                layers: from_cfg.layers.or(gguf.layers),
                hidden: from_cfg.hidden.or(gguf.hidden),
                vocab: from_cfg.vocab.or(gguf.vocab),
                context: from_cfg.context.or(gguf.context),
            };
            Ok((merged, info_extras(&cfg)))
        }
        Err(e) => {
            let any_dim = gguf.layers.is_some()
                || gguf.hidden.is_some()
                || gguf.vocab.is_some()
                || gguf.context.is_some();
            if !any_dim {
                return Err(e);
            }
            Ok((gguf, ModelExtras::default()))
        }
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

/// task 906: 設定ファイル経路は文字列で値を持つため、パスは UTF-8 だけ受ける。
fn flash_cfg_path(v: PathBuf, key: &str) -> Result<String> {
    v.into_os_string().into_string().map_err(|p| {
        anyhow::anyhow!(
            "`{key}` のパスが UTF-8 でない ({:?})。--config 経由の設定は UTF-8 のパスだけ扱う",
            p
        )
    })
}

/// task 906: `serve-flash` の設定を CLI 明示値 > `DENNO_*` > 設定ファイル > 既定値
/// の順で解決する。`--print-config` のときは表示して `Ok(None)` を返す。
#[allow(clippy::too_many_arguments)]
fn resolve_flash_config(
    dir: Option<PathBuf>,
    host: Option<String>,
    port: Option<u16>,
    max_ctx: Option<usize>,
    name: Option<String>,
    max_tokens: Option<usize>,
    mtp_path: Option<PathBuf>,
    kv_slots: Option<usize>,
    steer_recipe: Option<PathBuf>,
    vram_budget: Option<String>,
    config_path: Option<PathBuf>,
    print_config: bool,
) -> Result<Option<serve_flash::FlashConfig>> {
    // clap の既定値は使っていない (Option) ので、ここに入るのは明示値だけ。
    let mut cli_vals: Vec<(&'static str, String)> = Vec::new();
    if let Some(v) = dir {
        cli_vals.push(("dir", flash_cfg_path(v, "dir")?));
    }
    if let Some(v) = host {
        cli_vals.push(("host", v));
    }
    if let Some(v) = port {
        cli_vals.push(("port", v.to_string()));
    }
    if let Some(v) = max_ctx {
        cli_vals.push(("max_ctx", v.to_string()));
    }
    if let Some(v) = name {
        cli_vals.push(("name", v));
    }
    if let Some(v) = max_tokens {
        cli_vals.push(("max_tokens", v.to_string()));
    }
    if let Some(v) = mtp_path {
        cli_vals.push(("mtp_path", flash_cfg_path(v, "mtp_path")?));
    }
    if let Some(v) = kv_slots {
        cli_vals.push(("kv_slots", v.to_string()));
    }
    if let Some(v) = steer_recipe {
        cli_vals.push(("steer_recipe", flash_cfg_path(v, "steer_recipe")?));
    }
    if let Some(v) = vram_budget {
        cli_vals.push(("vram_budget", v));
    }

    let file = match &config_path {
        Some(p) => Some(config::load(p).map_err(|e| anyhow::anyhow!("{e}"))?),
        None => None,
    };
    let env: std::collections::BTreeMap<String, String> = std::env::vars().collect();
    let resolved = config::resolve(&cli_vals, &env, file.as_ref())
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    if print_config {
        print!("{}", resolved.render());
        return Ok(None);
    }

    let dir = resolved.path("dir").ok_or_else(|| {
        anyhow::anyhow!(
            "`dir` が決まらない: --dir <path>、環境変数 DENNO_FLASH_DIR、\
             設定ファイルの `dir = \"...\"` のどれかを指定する"
        )
    })?;
    let host = resolved
        .string("host")
        .ok_or_else(|| anyhow::anyhow!("`host` の既定値が無い (config.rs の SPEC を確認)"))?;
    let port = resolved
        .u16("port")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("`port` の既定値が無い (config.rs の SPEC を確認)"))?;
    let max_ctx = resolved
        .usize("max_ctx")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("`max_ctx` の既定値が無い (config.rs の SPEC を確認)"))?;
    let max_tokens = resolved
        .usize("max_tokens")
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("`max_tokens` の既定値が無い (config.rs の SPEC を確認)"))?;
    let name = resolved
        .string("name")
        .ok_or_else(|| anyhow::anyhow!("`name` の既定値が無い (config.rs の SPEC を確認)"))?;
    // 0 は従来の `DENNO_KV_SLOTS=0` と同じ「未指定」。serve_flash 側で
    // 既定 2 に落ちるよう None に正規化する。
    let kv_slots = match resolved.usize("kv_slots").map_err(|e| anyhow::anyhow!("{e}"))? {
        Some(0) => None,
        other => other,
    };

    Ok(Some(serve_flash::FlashConfig {
        dir,
        host,
        port,
        max_ctx,
        name,
        max_tokens,
        mtp_path: resolved.path("mtp_path"),
        kv_slots,
        steer_recipe: resolved.path("steer_recipe"),
        vram_budget: resolved.string("vram_budget"),
    }))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // s44: serve のみ非同期ロギングを使う (logging.rs — stdout 消費者の凍結で
    // サーバスレッドがブロックしない)。他サブコマンドは従来どおりの同期ロギング。
    if matches!(cli.command, Commands::Serve { .. }) {
        denno_server::init_serve_logging();
        // P877: `--log-format json` ならアクセスログだけ JSON にする。
        if let Commands::Serve { log_format, .. } = &cli.command {
            denno_server::set_access_log_json(log_format == "json");
        }
    } else {
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("info")
        });
        tracing_subscriber::fmt().init();
    }

    match cli.command {
        Commands::Abliterate {
            url,
            pos_file,
            neg_file,
            eval_file,
            capture_mode,
            alpha,
            out_dir,
            out_recipe,
            tag,
            model,
            eval_max_tokens,
            embed_snapshots,
            timeout,
            dry_run,
        } => {
            abliterate::run(abliterate::AbliterateOptions {
                url,
                pos_file,
                neg_file,
                eval_file,
                capture_mode,
                alpha,
                out_dir,
                out_recipe,
                tag,
                model,
                eval_max_tokens,
                embed_snapshots,
                timeout_secs: timeout,
                dry_run,
            })?;
        }
        Commands::Bench {
            url,
            api,
            pp,
            tg,
            depths,
            rounds,
            json,
            markdown,
            no_flush,
            flush_tokens,
            seed,
            det_tokens,
            reuse_depth,
            timeout,
            compare,
            fail_on_regress_pct,
            compare_json,
        } => {
            // P897: --compare はベンチを実行せずに 2 つのレポートを比べる。
            if let Some(paths) = &compare {
                if paths.len() != 2 {
                    bail!("--compare needs exactly 2 paths (BASE NEW)");
                }
                let out = bench::run_compare(&paths[0], &paths[1], fail_on_regress_pct)?;
                if markdown {
                    print!("{}", out.markdown);
                }
                let color = std::env::var_os("NO_COLOR").is_none();
                print!("{}", bench::render_compare_ansi(&out, color));
                // task 1021: 要約 JSON (機械可読)。`-` なら stdout。
                if let Some(path) = &compare_json {
                    let text = bench::render_compare_json(&out)?;
                    if path.as_os_str() == "-" {
                        println!("{text}");
                    } else {
                        std::fs::write(path, text)
                            .with_context(|| format!("write {}", path.display()))?;
                        eprintln!("wrote {}", path.display());
                    }
                }
                if out.exit_fail {
                    bail!(
                        "bench compare: {} regression(s), {} missing, determinism {:?}",
                        out.regressions,
                        out.missing,
                        out.determinism_match
                    );
                }
                return Ok(());
            }
            if compare_json.is_some() {
                bail!("--compare-json requires --compare BASE NEW");
            }
            let opts = bench::BenchOptions {
                url,
                api: bench::Api::parse(&api)?,
                pp,
                tg,
                depths,
                rounds,
                flush: !no_flush,
                flush_tokens,
                seed,
                det_tokens,
                reuse_depth,
                timeout_secs: timeout,
            };
            let report = bench::run(&opts)?;
            if markdown {
                println!("{}", bench::render_markdown(&report));
            }
            if let Some(path) = &json {
                std::fs::write(path, bench::render_json(&report)?)
                    .with_context(|| format!("write {}", path.display()))?;
                eprintln!("wrote {}", path.display());
            }
        }
        Commands::Chat {
            url,
            api,
            system,
            max_tokens,
            temperature,
            model,
            api_key,
            timeout,
        } => {
            chat::run(chat::ChatOptions {
                url,
                api: bench::Api::parse(&api)?,
                system,
                max_tokens,
                temperature,
                timeout_secs: timeout,
                model,
                api_key,
            })?;
        }
        Commands::ServeFlash {
            dir,
            host,
            port,
            max_ctx,
            name,
            max_tokens,
            mtp_path,
            kv_slots,
            steer_recipe,
            vram_budget,
            config,
            print_config,
        } => {
            // task 906: CLI 明示値 > DENNO_* > 設定ファイル > 既定値。
            // --print-config のときはここで表示して終わる (サーバーは起動しない)。
            if let Some(cfg) = resolve_flash_config(
                dir,
                host,
                port,
                max_ctx,
                name,
                max_tokens,
                mtp_path,
                kv_slots,
                steer_recipe,
                vram_budget,
                config,
                print_config,
            )? {
                serve_flash::run(cfg)?;
            }
        }
        Commands::Pull { repo } => {
            let dir = model_dir(&repo);
            pull::pull(&repo, &dir)?;
            println!("done: {}", dir.display());
        }
        Commands::Inspect { path, json, expert_cache_mib } => {
            inspect::run(&path, json, expert_cache_mib)?;
        }
        Commands::Tokenize { dir, text, stdin, json } => {
            // task 941: 失敗は終了コードを分けて返す (2=使い方 / 3=GGUF /
            // 4=標準入力が UTF-8 でない)。main の `?` だとすべて 1 になるため
            // ここだけ自前で stderr に出して exit する。
            if let Err(e) = tokenize::run(&dir, text.as_deref(), stdin, json) {
                use std::io::Write;
                let _ = writeln!(std::io::stderr(), "denno tokenize: {}", e.message);
                std::process::exit(e.code);
            }
        }
        Commands::Doctor { dir, json } => {
            doctor::run(dir.as_deref(), json)?;
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
            // s164 (K5): config.json が無い/次元キーが欠けているときは GGUF の
            // メタデータで埋める (config.json が優先)。どちらも無ければ従来どおり Err。
            let (info, extras) = info_resolve(&dir)?;
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
        Commands::Serve { repo, host, port, name, api_key, api_keys, max_tokens: _max_tokens, seed, prefix_cache_tokens, quant, max_queue_depth, request_timeout_secs, repetition_penalty, temperature_cap, log_format: _log_format, cors_origin, max_body_mb } => {
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
                api_keys,
                default_seed: seed,
                prefix_cache_tokens,
                quant: quant_mode,
                max_queue: denno_server::DEFAULT_MAX_QUEUE_DEPTH,
                max_queue_depth,
                request_timeout_secs,
                repetition_penalty,
                temperature_cap,
                cors_origin,
                max_body_mb,
            }))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// task 916: `serve --cors-origin` の受け取り。未指定は None (env / 既定へ)。
    #[test]
    fn serve_cors_origin_flag_is_parsed() {
        let cli = Cli::try_parse_from(["denno", "serve", "repo/x"]).unwrap();
        match cli.command {
            Commands::Serve { cors_origin, .. } => assert_eq!(cors_origin, None),
            _ => panic!("serve でない"),
        }
        let cli = Cli::try_parse_from([
            "denno",
            "serve",
            "repo/x",
            "--cors-origin",
            "https://a.example,http://localhost:5173",
        ])
        .unwrap();
        match cli.command {
            Commands::Serve { cors_origin, .. } => assert_eq!(
                cors_origin.as_deref(),
                Some("https://a.example,http://localhost:5173")
            ),
            _ => panic!("serve でない"),
        }
        let cli =
            Cli::try_parse_from(["denno", "serve", "repo/x", "--cors-origin", "*"]).unwrap();
        match cli.command {
            Commands::Serve { cors_origin, .. } => assert_eq!(cors_origin.as_deref(), Some("*")),
            _ => panic!("serve でない"),
        }
    }

    /// task 951: `serve --api-keys` の受け取り。未指定は None (他と env の和へ)。
    /// `--api-key` と併用できる。
    #[test]
    fn serve_api_keys_flag_is_parsed() {
        let cli = Cli::try_parse_from(["denno", "serve", "repo/x"]).unwrap();
        match cli.command {
            Commands::Serve { api_keys, api_key, .. } => {
                assert_eq!(api_keys, None);
                assert_eq!(api_key, None);
            }
            _ => panic!("serve でない"),
        }
        let cli = Cli::try_parse_from([
            "denno",
            "serve",
            "repo/x",
            "--api-keys",
            "alpha,beta",
            "--api-key",
            "solo",
        ])
        .unwrap();
        match cli.command {
            Commands::Serve { api_keys, api_key, .. } => {
                assert_eq!(api_keys.as_deref(), Some("alpha,beta"));
                assert_eq!(api_key.as_deref(), Some("solo"));
            }
            _ => panic!("serve でない"),
        }
    }

    /// task 946: `serve --max-body-mb` の受け取り。未指定は None (env / 既定へ)。
    /// 0 は clap の range で拒否する。
    #[test]
    fn serve_max_body_mb_flag_is_parsed() {
        let cli = Cli::try_parse_from(["denno", "serve", "repo/x"]).unwrap();
        match cli.command {
            Commands::Serve { max_body_mb, .. } => assert_eq!(max_body_mb, None),
            _ => panic!("serve でない"),
        }
        let cli =
            Cli::try_parse_from(["denno", "serve", "repo/x", "--max-body-mb", "64"]).unwrap();
        match cli.command {
            Commands::Serve { max_body_mb, .. } => assert_eq!(max_body_mb, Some(64)),
            _ => panic!("serve でない"),
        }
        assert!(
            Cli::try_parse_from(["denno", "serve", "repo/x", "--max-body-mb", "0"]).is_err(),
            "--max-body-mb 0 は拒否するべき"
        );
    }

    /// s164 (K5): テスト用 GGUF メタデータ値 (最小の書き手用)
    enum GgufMeta<'a> {
        Str(&'a str),
        U32(u32),
        /// 文字列配列 (`tokenizer.ggml.tokens` 用)
        ArrStr(&'a [&'a str]),
    }

    /// GGUF の magic ("GGUF" を LE で読んだ値)
    const GGUF_MAGIC: u32 = 0x4655_4747;

    fn gguf_push_str(b: &mut Vec<u8>, s: &str) {
        b.extend_from_slice(&(s.len() as u64).to_le_bytes());
        b.extend_from_slice(s.as_bytes());
    }

    /// s168 (P7): メタデータの key/value 列を GGUF v3 形式で書き出す
    fn gguf_push_kv(b: &mut Vec<u8>, kv: &[(&str, GgufMeta<'_>)]) {
        for (key, val) in kv {
            gguf_push_str(b, key);
            match val {
                GgufMeta::Str(s) => {
                    b.extend_from_slice(&8u32.to_le_bytes()); // GT_STRING
                    gguf_push_str(b, s);
                }
                GgufMeta::U32(n) => {
                    b.extend_from_slice(&4u32.to_le_bytes()); // GT_U32
                    b.extend_from_slice(&n.to_le_bytes());
                }
                GgufMeta::ArrStr(items) => {
                    b.extend_from_slice(&9u32.to_le_bytes()); // GT_ARRAY
                    b.extend_from_slice(&8u32.to_le_bytes()); // elem = string
                    b.extend_from_slice(&(items.len() as u64).to_le_bytes());
                    for s in items.iter() {
                        gguf_push_str(b, s);
                    }
                }
            }
        }
    }

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

        /// s164 (K5): メタデータだけの最小 GGUF v3 (テンソル 0 個) を置く。
        /// データ部が無いので、32 byte 整列のぶんだけ 0 で埋める。
        fn write_gguf_meta(&self, name: &str, kv: &[(&str, GgufMeta<'_>)]) -> PathBuf {
            let mut b = Vec::new();
            b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            b.extend_from_slice(&3u32.to_le_bytes()); // version
            b.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
            gguf_push_kv(&mut b, kv);
            while b.len() % 32 != 0 {
                b.push(0);
            }
            self.write_file(name, &b)
        }

        /// s168 (P7): テンソルを数本だけ持つ最小 GGUF v3。データ部は置かない
        /// (open はヘッダの整列だけを見るので、32 byte 詰めで通る)。
        fn write_gguf_tensors(
            &self,
            name: &str,
            kv: &[(&str, GgufMeta<'_>)],
            tensors: &[(&str, GgmlType)],
        ) -> PathBuf {
            let mut b = Vec::new();
            b.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            b.extend_from_slice(&3u32.to_le_bytes()); // version
            b.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
            b.extend_from_slice(&(kv.len() as u64).to_le_bytes());
            gguf_push_kv(&mut b, kv);
            for (tname, ty) in tensors {
                gguf_push_str(&mut b, tname);
                b.extend_from_slice(&1u32.to_le_bytes()); // n_dims
                b.extend_from_slice(&1u64.to_le_bytes()); // ne[0] = 1
                b.extend_from_slice(&ggml_type_id(*ty).to_le_bytes());
                b.extend_from_slice(&0u64.to_le_bytes()); // offset
            }
            while b.len() % 32 != 0 {
                b.push(0);
            }
            self.write_file(name, &b)
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

    // s168 (P7): F32 は量子化されない置き場 (rmsnorm / bias) で本数が量子化された重みを
    // 上回る (実ファイル Qwen3.8-27B-Q4_0.gguf は F32 456 / Q4_0 352)。量子化の種類を
    // 出す行なので、F32 以外が 1 本でもあれば F32 を投票から外す。
    #[test]
    fn dominant_type_ignores_f32_when_quantised_type_exists() {
        use GgmlType::{F32, Q4_0};
        let mut types = vec![F32; 5];
        types.extend(vec![Q4_0; 2]);
        // 素の最頻値は F32 (5) で、これが「quant: F32 (0)」の原因だった
        assert_eq!(most_common_type(types.iter().copied()), Some((F32, 5)));
        assert_eq!(dominant_tensor_type(types.into_iter()), Some((Q4_0, 2)));
        // 量子化型が 1 本だけでも F32 は負けない (実ファイルの F32 456 本でも同じ)
        let mut one = vec![F32; 456];
        one.push(Q4_0);
        assert_eq!(dominant_tensor_type(one.into_iter()), Some((Q4_0, 1)));
    }

    // s168 (P7): 本当に F32 だけの GGUF は従来どおり F32 を返す (空投票にしない)
    #[test]
    fn dominant_type_keeps_f32_for_f32_only_file() {
        use GgmlType::F32;
        assert_eq!(dominant_tensor_type(vec![F32, F32, F32].into_iter()), Some((F32, 3)));
        assert_eq!(dominant_tensor_type(vec![F32].into_iter()), Some((F32, 1)));
    }

    // s168 (P7): F32 が無いファイルの答えは素の最頻値と一致する (挙動不変)
    #[test]
    fn dominant_type_matches_mode_without_f32() {
        use GgmlType::{Q4K, Q5K, Q8_0};
        let types = vec![Q8_0, Q4K, Q4K, Q5K];
        assert_eq!(
            dominant_tensor_type(types.iter().copied()),
            most_common_type(types.iter().copied())
        );
        assert_eq!(dominant_tensor_type(types.into_iter()), Some((Q4K, 2)));
        // 同数は型 id の小さい方 (決定論) も素の最頻値と同じ -> Q8_0 (8) < Q4K (12)
        assert_eq!(dominant_tensor_type(vec![Q4K, Q8_0].into_iter()), Some((Q8_0, 1)));
        assert_eq!(dominant_tensor_type(vec![Q5K, Q5K].into_iter()), Some((Q5K, 2)));
    }

    // s168 (P7): テンソル 0 本は None (panic しない)。F32 を外して空になる経路は無い
    #[test]
    fn dominant_type_empty_is_none() {
        assert_eq!(dominant_tensor_type(Vec::<GgmlType>::new().into_iter()), None);
        assert!(dominant_tensor_type(vec![GgmlType::F32].into_iter()).is_some());
    }

    // s168 (P7): 実ファイル経路 (gguf を開いて tensor_order を投票する) でも同じ答え。
    // ヘッダだけの最小 GGUF なのでデータ部は無い (open は整列だけを見る)。
    #[test]
    fn gguf_quant_value_skips_f32_storage_tensors() {
        let td = TempDir::new("dominant_gguf");
        let p = td.write_gguf_tensors(
            "q.gguf",
            &[("general.architecture", GgufMeta::Str("qwen3"))],
            &[
                ("blk.0.attn_norm.weight", GgmlType::F32),
                ("blk.0.ffn_norm.weight", GgmlType::F32),
                ("output_norm.weight", GgmlType::F32),
                ("blk.0.attn_q.weight", GgmlType::Q4_0),
                ("blk.0.ffn_down.weight", GgmlType::Q4_0),
            ],
        );
        // F32 3 本 vs Q4_0 2 本: 全テンソル投票なら F32 (0) になる並び
        let f = denno_engine::gguf::GgufFile::open(&p).expect("open test gguf");
        let all = f
            .tensor_order
            .iter()
            .filter_map(|n| f.tensors.get(n).map(|t| t.ty));
        assert_eq!(most_common_type(all), Some((GgmlType::F32, 3)));
        // 実際に info が出す値は量子化型
        assert_eq!(gguf_quant_value(&p), "Q4_0 (2)");
        let line = info_quant_line(&td.path).expect("gguf exists so the line exists");
        assert_eq!(line_value(&[line], "quant:"), "Q4_0 (2)");
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

    // ---- s164 (K5): GGUF だけのディレクトリから次元を埋める ----

    // e. config.json が無く GGUF しか無いディレクトリでも 4 次元が非 0 で出る
    #[test]
    fn gguf_only_dir_fills_dims() {
        let td = TempDir::new("gguf_only");
        let tokens: Vec<&str> = vec!["a", "b", "c", "d"];
        let p = td.write_gguf_meta(
            "model.gguf",
            &[
                ("general.architecture", GgufMeta::Str("qwen3")),
                ("qwen3.block_count", GgufMeta::U32(64)),
                ("qwen3.embedding_length", GgufMeta::U32(5120)),
                ("qwen3.context_length", GgufMeta::U32(262144)),
                ("tokenizer.ggml.tokens", GgufMeta::ArrStr(&tokens)),
            ],
        );
        assert_eq!(find_gguf(&td.path).as_deref(), Some(p.as_path()));
        // config.json は無い (read_config の Err 契約はそのまま)
        assert!(read_config(&td.path).is_err());

        let (info, extras) = info_resolve(&td.path).expect("gguf metadata must fill dims");
        assert_eq!(info.layers, Some(64));
        assert_eq!(info.hidden, Some(5120));
        assert_eq!(info.vocab, Some(4)); // tokenizer.ggml.tokens の配列長
        assert_eq!(info.context, Some(262144));
        // family は config.json からのみ (GGUF の arch 名は family ではない)
        assert_eq!(info.family, "?");

        // 表示行のラベル・書式は従来どおりで、値が非 0
        let lines = info_config_lines("local/g", &info, &extras);
        assert_eq!(line_value(&lines, "layers:"), "64");
        assert_eq!(line_value(&lines, "hidden:"), "5120");
        assert_eq!(line_value(&lines, "vocab:"), "4");
        assert_eq!(line_value(&lines, "context:"), "262144");
        for label in ["layers:", "hidden:", "vocab:", "context:"] {
            assert_ne!(line_value(&lines, label), "0", "{label} must be non-zero");
        }
        // GGUF があるので quant 行は従来どおり出る
        assert!(info_quant_line(&td.path).is_some());
    }

    // s169 (P19): GGUF だけの 27B は block_count=65 (= trunk 64 + MTP 1) なので、
    // エンジンが実際に走らせる層数 (config の num_hidden_layers = 64) と一致させる
    // ため `<arch>.nextn_predict_layers` を差し引く。hidden / vocab は従来のキー。
    #[test]
    fn gguf_only_mtp_layer_count_matches_engine() {
        let td = TempDir::new("gguf_mtp");
        let tokens: Vec<&str> = vec!["a", "b", "c", "d"];
        td.write_gguf_meta(
            "model.gguf",
            &[
                ("general.architecture", GgufMeta::Str("qwen35")),
                ("qwen35.block_count", GgufMeta::U32(65)),
                ("qwen35.nextn_predict_layers", GgufMeta::U32(1)),
                ("qwen35.embedding_length", GgufMeta::U32(5120)),
                ("qwen35.context_length", GgufMeta::U32(262144)),
                ("tokenizer.ggml.tokens", GgufMeta::ArrStr(&tokens)),
            ],
        );
        assert!(read_config(&td.path).is_err(), "GGUF だけのディレクトリ");
        let (info, extras) = info_resolve(&td.path).expect("gguf metadata must fill dims");
        // 65 - 1 = 64 (block_count をそのまま出さない)
        assert_eq!(info.layers, Some(64));
        assert_eq!(info.hidden, Some(5120));
        assert_eq!(info.vocab, Some(4));
        assert_eq!(info.context, Some(262144));

        let lines = info_config_lines("local/mtp", &info, &extras);
        assert_eq!(line_value(&lines, "layers:"), "64");
        assert_eq!(line_value(&lines, "hidden:"), "5120");
        assert_eq!(line_value(&lines, "vocab:"), "4");
        for label in ["layers:", "hidden:", "vocab:", "context:"] {
            assert_ne!(line_value(&lines, label), "0", "{label} must be non-zero");
        }
    }

    // s169 (P19): nextn の差し引きの境界と、config.json 優先の確認。
    #[test]
    fn gguf_nextn_edges_and_config_precedence() {
        // (a) nextn == block_count のような異常値は 0 を出さず `?`
        let bad = TempDir::new("gguf_nextn_bad");
        bad.write_gguf_meta(
            "m.gguf",
            &[
                ("general.architecture", GgufMeta::Str("acme")),
                ("acme.block_count", GgufMeta::U32(2)),
                ("acme.nextn_predict_layers", GgufMeta::U32(2)),
                // 層数以外の次元はあるので GGUF 経路は成立する (layers だけ `?`)
                ("acme.embedding_length", GgufMeta::U32(1024)),
            ],
        );
        let (info, _extras) = info_resolve(&bad.path).expect("block_count is enough");
        assert_eq!(info.layers, None);
        assert_eq!(info.hidden, Some(1024));
        let lines = info_config_lines("local/bad", &info, &ModelExtras::default());
        assert_eq!(line_value(&lines, "layers:"), "?");
        assert_eq!(line_value(&lines, "hidden:"), "1024");

        // (b) nextn キーが無ければ block_count そのまま (差し引かない)
        let plain = TempDir::new("gguf_no_nextn");
        plain.write_gguf_meta(
            "m.gguf",
            &[
                ("general.architecture", GgufMeta::Str("acme")),
                ("acme.block_count", GgufMeta::U32(12)),
            ],
        );
        let (info, _extras) = info_resolve(&plain.path).expect("block_count is enough");
        assert_eq!(info.layers, Some(12));

        // (c) config.json があればそちらが勝つ (GGUF の 65/1 で上書きしない)
        let both = TempDir::new("cfg_vs_mtp");
        both.write_config(r#"{"model_type":"qwen3_5_text","num_hidden_layers":60}"#);
        both.write_gguf_meta(
            "m.gguf",
            &[
                ("general.architecture", GgufMeta::Str("qwen35")),
                ("qwen35.block_count", GgufMeta::U32(65)),
                ("qwen35.nextn_predict_layers", GgufMeta::U32(1)),
            ],
        );
        let (info, _extras) = info_resolve(&both.path).expect("config must parse");
        assert_eq!(info.layers, Some(60));
    }

    // f. config.json と GGUF の両方があるときは config.json が勝つ
    #[test]
    fn config_json_wins_over_gguf() {
        let td = TempDir::new("both");
        td.write_config(FLAT_LLAMA); // 32 / 4096 / 128256 / 131072
        let tokens: Vec<&str> = vec!["a", "b"];
        td.write_gguf_meta(
            "model.gguf",
            &[
                ("general.architecture", GgufMeta::Str("qwen3")),
                ("qwen3.block_count", GgufMeta::U32(64)),
                ("qwen3.embedding_length", GgufMeta::U32(5120)),
                ("qwen3.context_length", GgufMeta::U32(262144)),
                ("tokenizer.ggml.tokens", GgufMeta::ArrStr(&tokens)),
            ],
        );
        let (info, _extras) = info_resolve(&td.path).expect("config must parse");
        assert_eq!(info.family, "llama");
        assert_eq!(info.layers, Some(32));
        assert_eq!(info.hidden, Some(4096));
        assert_eq!(info.vocab, Some(128256));
        assert_eq!(info.context, Some(131072));
    }

    // g. config.json にキーが欠けているときは、欠けた次元だけ GGUF で埋める
    // (同じ次元を GGUF で上書きしない = 出所を混ぜない)
    #[test]
    fn config_missing_keys_filled_from_gguf() {
        let td = TempDir::new("partial");
        td.write_config(r#"{"model_type":"acme","vocab_size":999}"#);
        let tokens: Vec<&str> = vec!["a", "b", "c"];
        td.write_gguf_meta(
            "m.gguf",
            &[
                ("general.architecture", GgufMeta::Str("acme")),
                ("acme.block_count", GgufMeta::U32(12)),
                ("acme.embedding_length", GgufMeta::U32(768)),
                ("acme.context_length", GgufMeta::U32(4096)),
                ("acme.vocab_size", GgufMeta::U32(32000)),
                ("tokenizer.ggml.tokens", GgufMeta::ArrStr(&tokens)),
            ],
        );
        let (info, _extras) = info_resolve(&td.path).expect("config must parse");
        assert_eq!(info.family, "acme");
        assert_eq!(info.layers, Some(12)); // config に無い -> GGUF
        assert_eq!(info.hidden, Some(768)); // config に無い -> GGUF
        assert_eq!(info.context, Some(4096)); // config に無い -> GGUF
        assert_eq!(info.vocab, Some(999)); // config.json が勝つ (配列長 3 ではない)
    }

    // h. トークン表が無い GGUF では `<arch>.vocab_size` にフォールバックする
    // (arch 名はメタデータから組み立てる。決め打ちしない)
    #[test]
    fn gguf_vocab_from_arch_vocab_size() {
        let td = TempDir::new("gguf_vocab");
        td.write_gguf_meta(
            "m.gguf",
            &[
                ("general.architecture", GgufMeta::Str("acme")),
                ("acme.vocab_size", GgufMeta::U32(32000)),
            ],
        );
        let (info, _extras) = info_resolve(&td.path).expect("vocab_size is enough");
        assert_eq!(info.vocab, Some(32000));
        assert_eq!(info.layers, None);
        assert_eq!(info.hidden, None);
        assert_eq!(info.context, None);
    }

    // i. config.json も GGUF のメタデータも無いときは従来どおり Err (panic しない)
    #[test]
    fn no_config_and_no_gguf_metadata_is_err_not_panic() {
        // 空のディレクトリ
        let empty = TempDir::new("nothing");
        let err = info_resolve(&empty.path).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("config.json"), "unexpected message: {msg}");

        // 壊れた GGUF しか無いディレクトリ
        let broken = TempDir::new("broken_meta");
        broken.write_file("x.gguf", b"not a gguf");
        let err = info_resolve(&broken.path).unwrap_err();
        let msg = format!("{:#}", err);
        assert!(msg.contains("config.json"), "unexpected message: {msg}");

        // general.architecture が無い GGUF しか無いディレクトリ
        let noarch = TempDir::new("no_arch");
        let none: Vec<&str> = Vec::new();
        noarch.write_gguf_meta("y.gguf", &[("tokenizer.ggml.tokens", GgufMeta::ArrStr(&none))]);
        assert!(info_resolve(&noarch.path).is_err());
    }
}
