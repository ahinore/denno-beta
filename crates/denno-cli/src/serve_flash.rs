//! Qwen3.8-Flash-Next (arch `qwen4exp`) 用の最小 HTTP サーバー。
//!
//! 27B 用の `denno serve` (denno-server / axum) とは別経路。Flash-Next は
//! MoE + hyper-connection + QSA + PLE の 4 要素を持つ別アーキで、`Engine`
//! 実装を持たないため、ここで直接 `Qwen4ExpDecoder` を回す。
//!
//! - `POST /v1/messages` (Anthropic Messages API、`stream` の有無で SSE / JSON)
//! - `GET /v1/models`
//! - `GET /health`
//! - `GET /v1/model/shape` (task 217: ビューワが層と MoE の形を読む)
//! - `GET|POST|DELETE /v1/intervene` (task 217: expert / block / 残差ストリームの介入)
//!
//! 直前のリクエストの状態 (KV / GDN / PLE の窓) を 1 つだけ保持し、次の
//! プロンプトが**そのトークン列の続き**なら差分だけ食わせる。会話のたびに
//! システムプロンプトを読み直さないための仕組みで、前置きが一致しないときは
//! モデルを組み直して最初から prefill していた。
//!
//! P118: 前置きが一致しないときも、decoder が残っていれば `build_model` を
//! 呼ばずに系列状態だけ消して (`Qwen4ExpDecoder::reset`) 使い回す。組み直すと
//! `GgufLinear` に新しい uid が振られて GPU 常駐の重みを全部再 upload する
//! (cache-miss 1 回あたり 266 本 / 3.4 秒の無駄)。reset は uid に触れないので
//! 重みは常駐したまま、logits は新規構築とビット一致する。
//! `DENNO_FLASH_MODEL_REUSE=0` で組み直しに戻す (既定は使い回す)。
//!
//! task 123: クライアント切断 (emit = HTTP 応答の差分書き出しの失敗、broken
//! pipe など) のときも decoder を `self.cache` に残す。engine 側の失敗
//! (prefill / forward step) のときは従来どおり捨てる。切断の直後のリクエストは
//! 組み直さないので、GPU 重みの再 upload (266 本 / 約 3 秒) が避けられ、
//! TTFT が 5.0 秒 → 通常の reuse=0 prefill と同じ (推定 2.3 秒) に戻る。
//!
//! task 131: chat_template が assistant の content を `|trim` するため、
//! 生成テキスト末尾の空白は次ターンのプロンプトから落ちる。そこで前置きが
//! 1 トークンだけずれて reuse=0 (= full prefill) になっていた (本番 462
//! トークンで 18.19 秒)。前置きが一致しないときは、前回の生成トークンを
//! そのまま前置きに残して後ろだけ食う splice を試す (既定 on、
//! `DENNO_FLASH_PREFIX_SPLICE=0` で切れる)。splice できないときは従来どおり
//! full prefill に落ち、`[flash] prefix miss: ...` を 1 行出す。
//!
//! task 141: その splice は「最後に返した assistant の直後」を継ぐ形だったので、
//! 一度 splice したキャッシュは古い assistant の末尾空白を抱えたままになり、
//! 以降の描画とはその位置 (本番 443) でずれ続ける。いまは描画済みプロンプトと
//! キャッシュをトークン単位で突き合わせ、キャッシュ側にだけ残っている空白
//! (`|trim` が落とした分) を吸収して食い残しを求める
//! ([`GgufTokenizer::align_trimmed_prefix`])。何ターン前の assistant でも同じ
//! 扱いになり、空白以外の不一致は full prefill のまま。整列できない形
//! (content が空白で *始まる* など、`|trim` が先頭も落とすケース) は従来の
//! sentinel splice に落ちる。返送本文の照合も、テンプレートが端の空白を
//! 落とすのに合わせて端の空白差を許す。

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::time::Instant;

// task 347: Prometheus `/metrics` の計器と本文の組み立て。
#[path = "serve_metrics.rs"]
mod metrics;

// task 866: prefix KV スロットの radix (パス圧縮トライ) 索引。既定 off。
#[path = "kv_radix.rs"]
mod kv_radix;

use anyhow::{bail, Context, Result};
use denno_engine::gguf::GgufFile;
use denno_engine::model::prekprof;
use denno_engine::model::q4ledger;
use denno_engine::model::intervene as iv;
use denno_engine::model::steer;
use denno_engine::model::qwen4exp::{
    expert_w_stats_request_end, q4prof, LayerKind, Qwen4Exp, Qwen4ExpParams, TapTokens,
};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_decode::{
    argmax, DecoderConfig, DecoderSeqState, Qwen4ExpDecoder,
};
use denno_engine::spec_ngram::{
    offer_draft, NgramIndex, DEFAULT_CAND_KEEP, DEFAULT_MAX_ENTRIES, DEFAULT_MAX_SUFFIX,
};
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
    /// task 796: MTP (nextn) head の safetensors。`None` かつ
    /// `DENNO_MTP_PATH` 未設定なら MTP を読まない (既定起動は不変)。
    pub mtp_path: Option<PathBuf>,
    /// P812: 文脈スロット数 (`--kv-slots` / `DENNO_KV_SLOTS`、既定 2)。
    /// 2 は従来の main + aux と同じ、3 以上は LCP 一致 + LRU の汎用経路。
    pub kv_slots: Option<usize>,
    /// task 833: 起動時に掛けるレシピ (`--steer-recipe` / `DENNO_STEER_RECIPE`)。
    /// `None` かつ env 未設定なら何もしない (既定の推論は不変)。
    pub steer_recipe: Option<PathBuf>,
    /// P818: `--vram-budget` の生値 (`auto` か MiB の数値)。`None` なら
    /// `DENNO_AUTO_CONFIG` を見る。auto は起動時に VRAM / RAM を引き、
    /// 未設定の `DENNO_*` だけを計画値で埋める。
    pub vram_budget: Option<String>,
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
    /// 継続生成: 最後の assistant メッセージを閉じずにプロンプトへ含める。
    cont: bool,
}

/// 環境変数が 1/true のときだけ有効。
fn env_flag(k: &str) -> bool {
    matches!(std::env::var(k).as_deref(), Ok("1") | Ok("true") | Ok("yes"))
}

/// P118: `DENNO_FLASH_MODEL_REUSE` の値から「no-reuse 経路でモデルを使い回すか」
/// を決める (純関数。真偽表テスト `model_reuse_truth_table` が規約を固定する)。
///
/// 既定は `true` (= 使い回す)。`0` / `false` / `no` のときだけ `false`
/// (従来どおりリクエストごとに `build_model` する)。未設定・空・その他の値は
/// 既定に従う (typo で黙って毎回 3.4 秒かかるより、既定の速い経路に落ちる)。
fn model_reuse_enabled(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false") | Some("no"))
}

/// P118: テスト専用の `build_model` 呼び出しカウンタ (firing test 用)。
/// `cfg(test)` なので本番ビルドには存在しない。数値経路には触れない。
#[cfg(test)]
static TEST_BUILD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// task 123: テスト専用の直近リクエストの再利用トークン数 (firing test 用)。
/// `cfg(test)` なので本番ビルドには存在しない。数値経路には触れない。
#[cfg(test)]
static TEST_LAST_REUSE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// task 206: テスト専用の直近リクエストの [`CtxPick`] (発火テスト用)。
/// `cfg(test)` なので本番ビルドには存在しない。数値経路には触れない。
#[cfg(test)]
static TEST_LAST_PICK: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// task 123: テスト専用の engine 側 step 失敗の注入スイッチ (firing test 用)。
/// `cfg(test)` なので本番ビルドには存在しない。数値経路には触れない。
#[cfg(test)]
static TEST_FAIL_STEP: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// task 123: 生成ループ中の失敗の種別。
///
/// - [`GenFail::Client`] — emit (HTTP 応答の差分書き出し) の失敗 = クライアント
///   切断 (broken pipe など)。decoder は一貫している (emit は step の**前に**
///   呼ばれるので、失敗した差分のトークンはまだ step していない。spec 経路は
///   `spec_verify` が確定トークンを先に step するが、`tokens` と内部状態は
///   常に一致する)。生成を即やめて `self.cache` に保存し、次のリクエストが
///   使い回す / reset できるようにする。呼び出し元には今日どおり同じエラーを
///   返す。
/// - [`GenFail::Engine`] — prefill / forward step の失敗。半端に進んだ decoder
///   は信用できない (`Qwen4ExpDecoder::step` は forward が途中で失敗すると
///   `tokens` にそのトークンを残す) ので、今日どおり捨てて次のリクエストで
///   組み直す。
enum GenFail {
    Client(anyhow::Error),
    Engine(anyhow::Error),
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
        let cont = v
            .get("continue")
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
            cont,
        })
    }
}

/// 1 つの文脈 (会話) の状態。`toks` は `state` が食ったトークン列 (古い順)。
///
/// task 206: モデル (重み) は [`Flash::dec`] に 1 つだけ持ち、ここには系列状態
/// (KV / GDN / PLE conv / トークン列) だけを置く。状態の差し替えは move なので、
/// 文脈を入れ替えても KV のコピーは起きない。
struct Cached {
    state: DecoderSeqState,
    toks: Vec<u32>,
    /// task 217: この文脈を計算したときの介入 epoch。epoch が変わったら
    /// この文脈は使わない (別のモデルと同じなので)。
    epoch: u64,
    logits: Vec<f32>,
    /// `toks` のうちプロンプト部の長さ (`toks[prompt_len..]` が生成部)。
    /// task 131 の splice が「生成テキスト」を取り出すのに使う。
    prompt_len: usize,
    /// プロンプトが chat_template の描画結果と対応しているか
    /// (`encode_messages_ex` の bool)。描画に落ちたプロンプトは splice できない。
    from_render: bool,
    /// このキャッシュを生んだリクエストの入力 (task 131 の splice 判定用)。
    req: CachedReq,
}

/// キャッシュを生んだリクエストのうち、プロンプト構築に効く入力だけを持つ。
///
/// 次ターンのリクエストが「この入力 + 生成テキストの返送 + 続き」に一致する
/// ときだけ splice する (テンプレート描画はこの 4 つだけで決まる)。
#[derive(Clone, PartialEq)]
struct CachedReq {
    system: Option<String>,
    messages: Vec<(String, String)>,
    tools: Option<Value>,
    enable_thinking: bool,
    /// `continue: true` のプロンプトは描画結果と対応しないので splice しない。
    cont: bool,
}

/// task 217: `GET /v1/model/shape` が返す形 (ビューワ用)。
///
/// `layers` / `layer_kinds` は GGUF のメタデータから起動時に 1 回だけ作る。
/// `gdn` は GDN (ssm) 層、`full` は full attention 層。
#[derive(Clone, Debug)]
struct ModelShape {
    layers: usize,
    layer_kinds: Vec<&'static str>,
    n_expert: usize,
    n_expert_used: usize,
    n_hc: usize,
    n_embd: usize,
    /// task 220: PLE を注入する層 (実モデルは 1 層)。ビューワが `ple` を点灯する層。
    ple_layers: Vec<usize>,
}

impl ModelShape {
    fn json(&self) -> Value {
        json!({
            "layers": self.layers,
            "layer_kinds": self.layer_kinds,
            "n_expert": self.n_expert,
            "n_expert_used": self.n_expert_used,
            "n_hc": self.n_hc,
            "n_embd": self.n_embd,
            "ple_layers": self.ple_layers,
        })
    }

    /// エンジン側の検証に渡す形。
    fn engine_shape(&self) -> iv::Shape {
        iv::Shape {
            n_layer: self.layers,
            n_expert: self.n_expert,
            n_expert_used: self.n_expert_used,
            n_hc: self.n_hc,
        }
    }
}

/// task 217: このリクエストをどの文脈に載せるか。
///
/// 決め方は [`pick_ctx`] の doc にある。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CtxPick {
    /// 主の文脈 (長い会話) を延長する / 置き換える。
    Main,
    /// 短い脇リクエスト用の小さな文脈に載せる (主の文脈はそのまま残す)。
    Aux,
    /// 使える文脈が無い (新規)。主の枠に入れる。
    Fresh,
    /// 短い脇リクエストだが脇の容量 (`DENNO_FLASH_AUX_TOK`) を超える。
    /// 脇の文脈は捨てて主へ載せる (メモリ上限での追い出し)。
    MainAfterEvict,
}

impl CtxPick {
    fn name(self) -> &'static str {
        match self {
            CtxPick::Main => "main",
            CtxPick::Aux => "aux",
            CtxPick::Fresh => "fresh",
            CtxPick::MainAfterEvict => "main+evict",
        }
    }
}

/// task 206: `DENNO_FLASH_AUX_CTX` (既定 on)。`0` / `false` / `no` で切る。
fn aux_ctx_enabled(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false") | Some("no"))
}

/// task 206: `DENNO_FLASH_AUX_TOK` — 脇の文脈が持てるトークン数の上限 (既定 2048)。
///
/// 実モデルでは 1 トークンあたり KV が 48 KiB (full attention 12 層 ×
/// k/v 512 要素 × f32) なので、2048 トークンで約 96 MiB。GDN の状態
/// (文脈長に依存しない 117 MiB) が別に乗る。
fn aux_ctx_max_tok(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.max(8))
        .unwrap_or(2048)
}

/// task 206: `DENNO_FLASH_AUX_RATIO` — 主の 1/N 以下の長さなら脇の文脈へ (既定 4)。
/// 0 で「長さでは脇へ回さない」。
fn aux_ctx_ratio(v: Option<&str>) -> usize {
    v.and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(4)
}

/// P812: 文脈スロットの上限 (これ以上は端数を切る)。
pub const KV_SLOTS_MAX: usize = 8;

/// P812: `--kv-slots` / `DENNO_KV_SLOTS` の値からスロット数を決める
/// (純関数。真偽表テスト `kv_slots_truth_table` が規約を固定する)。
///
/// 既定は 2 (従来の main + aux とメモリ量・挙動が同じ)。`1` で単一スロット、
/// `3..=8` で N スロット LRU。0・typo・上限超えは既定 2 に落とす (黙って
/// メモリを食わないため)。`DENNO_KV_SLOTS` より CLI 引数が優先。
fn kv_slots_from(v: Option<&str>) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) if (1..=KV_SLOTS_MAX).contains(&n) => n,
        _ => 2,
    }
}

/// P812: N スロット経路で 1 スロットが持てるトークン数の上限
/// (`DENNO_KV_SLOT_TOK`。0 / 未設定 / typo は「コンテキスト長まで」)。
///
/// メモリの目安は 1 スロットあたり GDN 状態 117 MiB + KV 48 KiB/token。
fn kv_slot_max_tok(v: Option<&str>, max_ctx: usize) -> usize {
    match v.and_then(|s| s.trim().parse::<usize>().ok()) {
        Some(n) if n >= 8 => n.min(max_ctx),
        _ => max_ctx,
    }
}

/// task 866: `DENNO_KV_RADIX` (既定 off)。`1` / `true` / `yes` / `on` で、
/// スロット選択を radix (パス圧縮トライ) 索引 + 前置き優先の追い出しにする。
///
/// 他のフラグと違い **既定 off** (task の規則: 実験経路は明示フラグの裏)。
/// 既定 off では索引を 1 度も作らず、ログにも radix の欄を出さない
/// (既定の挙動とログ行を byte 単位で変えないため)。スロット数 2 では
/// 従来の main/aux heuristic (`pick_ctx`) をそのまま使うので不活性
/// (radix は `--kv-slots` / `DENNO_KV_SLOTS` が 3 以上のときだけ効く)。
fn kv_radix_enabled(v: Option<&str>) -> bool {
    matches!(v, Some("1") | Some("true") | Some("yes") | Some("on"))
}

/// P812: N スロット選択の入力 (1 スロットぶん)。
///
/// `len` は文脈のトークン数 (空きスロットは `None`)、`lcp` は今回のプロンプト
/// との共通前置き長、`used_at` は最終使用時刻 (単調増加。大きいほど新しい)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SlotMeta {
    len: Option<usize>,
    lcp: usize,
    used_at: u64,
    /// P812: 途中までの一致でも splice (task 131) で継げる文脈か。
    /// assistant の末尾空白が `|trim` で落ちるため、延長できないことが多い。
    splice: bool,
}

/// P812: どのスロットに載せるか (スロット数 ≥ 3 の汎用経路)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotPick {
    /// そのまま延長できる (キャッシュ全体がプロンプトの前置き)。
    Reuse(usize),
    /// 空きスロット (idx) に載せる。
    Fresh(usize),
    /// 一番古いスロット (idx) を捨てて載せる (LRU)。
    Evict(usize),
}

impl SlotPick {
    fn idx(self) -> usize {
        match self {
            SlotPick::Reuse(i) | SlotPick::Fresh(i) | SlotPick::Evict(i) => i,
        }
    }

    fn name(self) -> String {
        match self {
            SlotPick::Reuse(i) => format!("reuse{i}"),
            SlotPick::Fresh(i) => format!("fresh{i}"),
            SlotPick::Evict(i) => format!("evict{i}"),
        }
    }
}

/// task 866 監査: 既存の multi-slot prefix cache (P812) と radix 化の限界。
///
/// 既存の形: スロット 1 本 = 系列状態 1 つ (`Cached`)。選択は
/// [`pick_ctx`] (2 本) と [`pick_slot_n`] (3 本以上) で、
/// 1. 「キャッシュ全体がプロンプトの前置き」のスロットを最長優先で再利用、
/// 2. 部分一致は splice (task 131) で継げるものだけ、
/// 3. 空き → LRU 追い出し。
/// 状態は move でしか動かないので、スロット間で KV は共有されない。
///
/// task 866 で radix (パス圧縮トライ) を入れた範囲と、入れなかった範囲:
/// - 入れた: 全スロットの lcp を 1 回の走査で求める索引 ([`kv_radix::KvRadix`])。
///   共通 system プロンプトのような長い前置きを N 本が共有していても、
///   選択のコストが O(N × プロンプト長) から O(プロンプト長 + N) になる。
///   追い出しは「今回のプロンプトと前置きを共有しない文脈」を優先する。
/// - 入れなかった: 部分一致 (LCP の途中) からの KV 再利用 = copy-on-write の
///   分岐。理由は 2 つ。(a) full attention の KV は層長を戻せる
///   (`AttnCache::truncate_layer`) が、GDN (ssm) の再帰状態は層ごとの clone が
///   要る (spec decode の `SpecGdnSnap` と同じ形。1 文脈 117 MiB)。毎リクエスト
///   clone するコストはこの task では測っていない。(b) 部分再利用を入れると
///   「同じ前置きでも計算経路が変わる」ので、md5 一致の検証が重くなる。
///   したがって **再利用は従来どおり「キャッシュ全体が前置き」のときだけ** で、
///   出力は radix の有無で変わらない (ラボで md5 一致を確認する前提)。
///
/// P812: N スロットから 1 つ選ぶ純関数 (テストが規約を固定する)。
///
/// 優先順:
/// 1. キャッシュ全体がプロンプトの前置きになっているスロットのうち最長
///    (同じ長さなら最近使ったもの)。部分一致はそのままでは再利用できない
///    (KV は縮められず、GDN は畳み込み履歴を持つ) — task 206 と同じ規約。
/// 2. 1 が無く、splice (task 131) で継げるスロットがあれば、共通前置きが
///    一番長いもの (assistant の末尾空白が `|trim` で落ちるぶんだけずれる)。
/// 3. 空きスロット (最小 index)。
/// 4. 全部埋まっていれば最終使用が一番古いスロット (LRU。同値は最小 index)。
fn pick_slot_n(prompt_len: usize, slots: &[SlotMeta]) -> SlotPick {
    let mut best: Option<(usize, usize, u64)> = None;
    let mut splice_best: Option<(usize, usize, u64)> = None;
    for (i, s) in slots.iter().enumerate() {
        let Some(n) = s.len else { continue };
        let extendable = n <= prompt_len && s.lcp == n;
        if extendable {
            let better = match best {
                None => true,
                Some((bn, _, bu)) => n > bn || (n == bn && s.used_at >= bu),
            };
            if better {
                best = Some((n, i, s.used_at));
            }
        } else if s.splice && s.lcp > 0 {
            let better = match splice_best {
                None => true,
                Some((bl, _, bu)) => s.lcp > bl || (s.lcp == bl && s.used_at >= bu),
            };
            if better {
                splice_best = Some((s.lcp, i, s.used_at));
            }
        }
    }
    if let Some((_, i, _)) = best {
        return SlotPick::Reuse(i);
    }
    if let Some((_, i, _)) = splice_best {
        return SlotPick::Reuse(i);
    }
    if let Some((i, _)) = slots.iter().enumerate().find(|(_, s)| s.len.is_none()) {
        return SlotPick::Fresh(i);
    }
    let mut victim = 0usize;
    for (i, s) in slots.iter().enumerate() {
        if s.used_at < slots[victim].used_at {
            victim = i;
        }
    }
    SlotPick::Evict(victim)
}

/// task 206: どの文脈に載せるかを決める (純関数。テストが規約を固定する)。
///
/// 引数は `(トークン数, 共通前置き長)`。規則は優先順に:
///
/// 1. キャッシュ全体がプロンプトの前置きになっている文脈 (= そのまま延長できる)
///    のうち、一番長いものを選ぶ。同じ長さなら主。
/// 2. 延長できる文脈が無く、主の文脈が無ければ [`CtxPick::Fresh`]。
/// 3. 脇の文脈が有効で、プロンプトが主の `1/ratio` 以下
///    (かつ脇の容量内) なら [`CtxPick::Aux`] — 主へ食わせると主が消えるため。
/// 4. それ以外は [`CtxPick::Main`]。主より短い脇リクエストが容量を超えるときだけ
///    [`CtxPick::MainAfterEvict`] (脇の状態を捨ててから主へ)。
///
/// 「共通前置きが長い方」ではなく「延長できる方」を優先するのは、途中までしか
/// 一致しない文脈へ食わせるとその文脈が消えるから (それが task 206 の症状)。
/// KV は短くできない (GDN は畳み込みの履歴を持つ) ので、部分一致の再利用は
/// どのみち 0 トークンになる。
fn pick_ctx(
    prompt_len: usize,
    main: Option<(usize, usize)>,
    aux: Option<(usize, usize)>,
    aux_on: bool,
    aux_max_tok: usize,
    ratio: usize,
) -> CtxPick {
    let extendable = |c: Option<(usize, usize)>| -> Option<(usize, usize)> {
        c.filter(|&(n, lcp)| n <= prompt_len && lcp == n)
    };
    let m = extendable(main);
    let a = if aux_on { extendable(aux) } else { None };
    match (m, a) {
        (Some((mn, _)), Some((an, _))) => {
            return if an > mn { CtxPick::Aux } else { CtxPick::Main }
        }
        (Some(_), None) => return CtxPick::Main,
        (None, Some(_)) => return CtxPick::Aux,
        (None, None) => {}
    }
    let Some((main_len, _)) = main else {
        return CtxPick::Fresh;
    };
    if !aux_on {
        return CtxPick::Main;
    }
    let much_shorter = ratio > 0 && prompt_len.saturating_mul(ratio) <= main_len;
    if much_shorter {
        if prompt_len <= aux_max_tok {
            return CtxPick::Aux;
        }
        return CtxPick::MainAfterEvict;
    }
    CtxPick::Main
}

/// task 206: 診断用の `長さ/共通前置き長` (文脈が無ければ `-`)。
fn ctx_len_lcp(c: Option<&Cached>, lcp: Option<usize>) -> String {
    match (c, lcp) {
        (Some(c), Some(l)) => format!("{}/{}", c.toks.len(), l),
        _ => "-".to_string(),
    }
}

/// task 131: `DENNO_FLASH_PREFIX_SPLICE` の値から splice を試すかを決める
/// (純関数。真偽表テスト `prefix_splice_truth_table` が規約を固定する)。
///
/// 既定は `true`。`0` / `false` / `no` のときだけ `false` (従来どおり
/// 前置きが 1 トークンでもずれたら full prefill)。未設定・空・その他の値は
/// 既定に従う ([`model_reuse_enabled`] と同じ規約)。
fn prefix_splice_enabled(v: Option<&str>) -> bool {
    !matches!(v, Some("0") | Some("false") | Some("no"))
}

/// 2 列の最初の不一致位置 (短い方の長さで頭打ち)。
fn first_mismatch(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// task 926: prefix cache の利用実績 (累積)。サーバ本体は 1 本のスレッドが
/// `&mut Flash` を持つが、テストは複数スレッドで回すため原子カウンタにする
/// (Relaxed の加算 1 回だけで、ロックも確保もしない)。
static CACHE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CACHE_MISSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static CACHE_REUSED_TOKENS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// task 926: prefix cache の利用実績を 1 リクエストにつき 1 回だけ数える。
/// `reused` は前置きを使い回したトークン数 (0 ならフル prefill = miss)。
fn record_cache_reuse(reused: usize) {
    use std::sync::atomic::Ordering::Relaxed;
    if reused > 0 {
        CACHE_HITS.fetch_add(1, Relaxed);
        CACHE_REUSED_TOKENS.fetch_add(reused as u64, Relaxed);
    } else {
        CACHE_MISSES.fetch_add(1, Relaxed);
    }
}

/// task 926: 累積カウンタ `(hits, misses, total_reused_tokens)`。
fn cache_counters() -> (u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        CACHE_HITS.load(Relaxed),
        CACHE_MISSES.load(Relaxed),
        CACHE_REUSED_TOKENS.load(Relaxed),
    )
}

/// task 926: `GET /v1/cache/stats` の本文 (純関数。テストが形を固定する)。
///
/// - `slots_used` / `slots_capacity`: 文脈スロットの使用本数と総本数
/// - `hits` / `misses`: 完了したリクエストのうち前置きを使い回した / しなかった本数
/// - `hit_rate`: `hits / (hits + misses)`。リクエストが 0 なら 0
/// - `total_reused_tokens`: 使い回した先頭トークン数の累計
fn cache_stats_json(used: usize, capacity: usize, hits: u64, misses: u64, reused: u64) -> Value {
    let lookups = hits + misses;
    let hit_rate = if lookups == 0 {
        0.0
    } else {
        hits as f64 / lookups as f64
    };
    json!({
        "slots_used": used,
        "slots_capacity": capacity,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "total_reused_tokens": reused,
    })
}

/// task 131: 前置きが一致せず reuse=0 になったときの 1 行 (原因追跡用)。
///
/// 本番で「なぜ reuse=0 になったか」を後から説明できるように、最初の不一致
/// 位置とその両側のトークン (id と短い復号) を出す。1 行を保つため
/// 復号は `{:?}` でエスケープする (改行は `\n` になる)。
fn prefix_miss_line(tok: &GgufTokenizer, cached: &[u32], prompt: &[u32]) -> String {
    let lcp = first_mismatch(cached, prompt);
    let show = |t: Option<&u32>| match t {
        Some(&id) => match tok.token_text(id) {
            Some(s) => format!("{id}({s:?})"),
            None => format!("{id}(?)"),
        },
        None => "-".to_string(),
    };
    format!(
        "[flash] prefix miss: cache={} prompt={} lcp={} cached_tok={} prompt_tok={}",
        cached.len(),
        prompt.len(),
        lcp,
        show(cached.get(lcp)),
        show(prompt.get(lcp))
    )
}

/// task 141: 返送された本文が「食った生成トークンの復号」と同じ会話か。
///
/// テンプレートは assistant の content を `|trim` するので、両端の空白は
/// 描画に残らない。クライアントが端の空白を落として返しても同じ会話として
/// 扱う (描画結果が同じになるため)。空文字は不可 (内容ごと落とした可能性を
/// 区別できないので full prefill に落とす)。
fn echo_matches(tok: &GgufTokenizer, gen: &[u32], echo: &str) -> bool {
    if echo.is_empty() {
        return false;
    }
    let gen_text = tok.decode(gen).unwrap_or_default();
    if gen_text.is_empty() {
        return false;
    }
    echo.trim() == gen_text.trim()
}

/// P812: 1 つの文脈スロット。
///
/// スロット 0 が従来の「主」(長い会話)、スロット 1 が従来の「aux」に当たる
/// (スロット数 2 のときは挙動も従来どおり)。`used_at` は LRU 用の最終使用時刻で、
/// 0 は「未使用」。
struct KvSlot {
    ctx: Option<Cached>,
    used_at: u64,
}

impl KvSlot {
    fn empty() -> Self {
        KvSlot {
            ctx: None,
            used_at: 0,
        }
    }
}

struct Flash {
    parts: Vec<PathBuf>,
    max_ctx: usize,
    name: String,
    default_max: usize,
    tok: GgufTokenizer,
    /// 層優先のバッチ prefill を使うか (`DENNO_PREFILL_BATCH=0` で切れる)
    batch: bool,
    /// task 217: `/v1/model/shape` と介入の検証に使う形。
    shape: ModelShape,
    /// P118/task 206: 重みを持つ decoder。リクエストの間は系列状態が空。
    /// `DENNO_FLASH_MODEL_REUSE=0` でも成功したリクエストのあとは残す (今日と同じ)。
    dec: Option<Qwen4ExpDecoder>,
    /// P812: 文脈スロット列。`slots[0]` が主、`slots[1]` が脇 (スロット数 2 の
    /// ときは task 206 と同じ)。3 本以上では LCP 一致 + LRU で選ぶ。
    /// 各スロットの系列状態 (KV / GDN / PLE conv) は move でしか出入りしないので、
    /// スロット間に共有は無い。
    slots: Vec<KvSlot>,
    /// P812: LRU の論理時計 (スロットを使うたびに 1 進める)。
    tick: u64,
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
fn prof_flash(tag: &str, tok: usize, wall_ns: u64) {
    for line in denno_engine::model::devprof::report(tag, tok) {
        eprintln!("{line}");
    }
    prof_flash_q4(tag, tok, wall_ns);
}

/// `q4prof` は 27B の `DENNO_PROFILE` (表 C) とは別系統で、このモデル専用の軽量な
/// カウンタ。以前はテストからしか読めず serve では取れなかったので、リクエストの
/// 区切り (prefill 終端 / decode 終端) でここから出す。既定 off では
/// `enabled()` の分岐 1 回だけで、数値経路には触れない。
fn prof_flash_q4(tag: &str, tok: usize, wall_ns: u64) {
    use denno_engine::model::qwen4exp::{moe_sub_take, q4prof};
    // decode の GDN コア内訳 (`DENNO_GDN_KPROF=1` のときだけ 1 行)。
    // q4prof とは独立に出す (`DENNO_Q4_PROFILE` を立てなくても読める)。
    // 未設定なら `line()` が None = 出力ゼロ・積算もゼロのまま。
    if let Some(line) = denno_engine::gdn::kprof::line() {
        eprintln!("{line}");
        denno_engine::gdn::kprof::reset();
    }
    // decode の mix 内訳 (`DENNO_MIX_KPROF=1` のときだけ 1 行)。
    // こちらも q4prof とは独立に出す (`DENNO_Q4_PROFILE` を立てなくても読める)。
    if let Some(line) = denno_engine::model::qwen4exp::mix_kprof::line() {
        eprintln!("{line}");
        denno_engine::model::qwen4exp::mix_kprof::reset();
    }
    // task 87: decode の ssm 層 attention 投影 (qkv / gate / out GEMV) の経路内訳
    // (`DENNO_ATTN_KPROF=1` のときだけ 1 行)。GPU intercept を通った発数 / CPU へ
    // 落ちた発数が読める (`mix_kprof` と同じく q4prof とは独立に出す)。
    if let Some(line) = denno_engine::model::qwen4exp::attn_kprof::line() {
        eprintln!("{line}");
        denno_engine::model::qwen4exp::attn_kprof::reset();
    }
    // task 133: decode の full attention を GPU で回したぶんの内訳
    // (`DENNO_Q4_PROFILE` / `DENNO_PF_ATT_DBG` のときだけ 1 行)。
    // 遅延シード / 追記の H2D、launch、カーネル待ち、D2H、ミラー使用量が読める。
    if let Some(line) = q4prof::attn_gpu_line() {
        eprintln!("{line}");
        q4prof::attn_gpu_reset();
    }
    // expert GEMV の型別内訳 (`DENNO_EXP_KPROF=1` のときだけ 1 レポート)。
    // task 95: prefill 終端では積むだけにして、decode 終端 (= リクエストの終端) で
    // decode (m=1) と prefill (m>1) を分けた 1 レポートを出して reset する
    // (prefill だけで終わるリクエストでは次のリクエストに持ち越す)。
    if tag == "decode" {
        if let Some(line) = denno_engine::model::exp_kprof::line() {
            eprintln!("{line}");
            denno_engine::model::exp_kprof::reset();
        }
    }
    // prefill の expert GEMM (gemm_iq 系) の 1 行カウンタ
    // (`DENNO_GEMM_IQ_KPROF=1` のときだけ 1 行)。decode は exp_kprof、
    // prefill はこちら (gemm_iq_v2 / gemm_iq の IQ 系バッチ経路だけを積む)。
    if let Some(line) = denno_engine::model::gemm_iq_kprof::line() {
        eprintln!("{line}");
        denno_engine::model::gemm_iq_kprof::reset();
    }
    // task 138: decode 1 トークンの GPU 往復台帳 (`DENNO_Q4_PROFILE`)。
    // 区間ごとの発行数・同期回数と待ち・H2D/D2H バイト・壁時計を 1 行ずつ。
    // 値は 1 トークンあたり (この窓の decode トークン数で割る)。
    // 窓は prefill 終端でも切る (prefill の転送を decode の数字に混ぜない)。
    if tag == "decode" {
        for line in denno_engine::model::q4ledger::line(tok, wall_ns) {
            eprintln!("{line}");
        }
    } else {
        // task 145: prefill の台帳 (区間ごとの合計 ms と割合)。decode と同じ
        // 区間名を使うが、こちらは合計 (prefill 1 回 = tok トークン)。
        for line in denno_engine::model::q4ledger::line_prefill(tok) {
            eprintln!("{line}");
        }
        // task 254: 14k cold prefill のホスト台帳 (experts 区間の中の内訳)。
        for line in denno_engine::model::pf_host::report("prefill", tok) {
            eprintln!("{line}");
        }
    }
    // task 382: silu + up + hc_mixed の融合版 (`DENNO_MIX_UPQ8`) の発火回数と、
    // `=check` の比較結果 (積んでいるときだけ)。
    {
        let n = denno_engine::gpu_report::mix_upq8_hits();
        if n != 0 {
            eprintln!("[mix-upq8] fused={n} ({tag})");
            denno_engine::gpu_report::mix_upq8_reset();
        }
        // task 437: 1 対 = 4 レーンの up (`DENNO_MIX_UPQ8=u4` 系) の発火回数。
        let n4 = denno_engine::gpu_report::mix_up4_hits();
        if n4 != 0 {
            eprintln!("[mix-up4] up4={n4} ({tag})");
            denno_engine::gpu_report::mix_up4_reset();
        }
        if let Some(line) = denno_engine::gpu_report::mix_upq8_check_line() {
            eprintln!("{line}");
        }
    }
    // task 382: mix の鎖から H2D / D2H を消した回数 (`DENNO_MIX_ZC` のときだけ)。
    // ここが 0 のままなら鎖は従来経路で、測った数字は ZC の効果ではない。
    {
        let (zx, zo) = denno_engine::gpu_report::mix_zc_hits();
        if zx != 0 || zo != 0 {
            eprintln!("[mix-zc] no_h2d={zx} no_d2h={zo} ({tag})");
            denno_engine::gpu_report::mix_zc_reset();
        }
    }
    // task 187: decode の IQ expert 段の層内 GPU 時間 (PROFILE のときだけ)。
    if tag == "decode" {
        for line in denno_engine::model::q4ledger::dec_exp_kprof_lines() {
            eprintln!("{line}");
        }
        denno_engine::model::q4ledger::dec_exp_kprof_reset();
    }
    // task 173: GDN prefill 連鎖の段ごとの GPU 時間 (`DENNO_GDN_PF_KPROF=1`)。
    // 台帳の `qkv_pf` / `scan_pf` / `out_pf` は「発行 + 同期の待ち」の壁時計なので、
    // イベントで測った段の GPU 時間と並べると「待ち」が分離できる。
    if tag != "decode" {
        for line in denno_engine::model::q4ledger::gdn_pf_kprof_lines() {
            eprintln!("{line}");
        }
        denno_engine::model::q4ledger::gdn_pf_kprof_reset();
        // task 189: mix 1 呼び出しの段ごとの内訳 (`DENNO_PF_MIX_KPROF=1`)。
        // 台帳の `mix` 行は「発行 + 同期の待ち」なので、段の GPU 時間と
        // `sync_own` / `sync_other` を並べて「誰を待っていたか」を分離する。
        for line in denno_engine::model::q4ledger::pf_mix_kprof_lines() {
            eprintln!("{line}");
        }
        denno_engine::model::q4ledger::pf_mix_kprof_reset();
    }
    denno_engine::model::q4ledger::reset();
    // task 144: IQ expert のヒット率 / GPU で回した層と本数 / 上げ直し / VRAM
    // (`DENNO_EXPERT_IQ_GPU=1` のときだけ 1 行)。リクエスト単位で読む。
    if let Some(line) = denno_engine::model::exp_iq::stats_line() {
        eprintln!("{line}");
    }
    // task 794: VRAM expert キャッシュの載せ替え方針と頻度ゲートの発火数
    // (`DENNO_EXPERT_CACHE_POLICY=tinylfu` のときだけ 1 行)。
    if let Some(line) = denno_engine::model::expert_vram::policy_line() {
        eprintln!("{line}");
    }
    // task 830: P45 ゲートの判定内訳 (ゲートが在るときだけ 1 行)。
    // `bypass_gpu` が増えて `cpu` が 0 なら、TinyLFU の下で降格が起きていない。
    if let Some(line) = denno_engine::model::qwen4exp::expert_gate_line() {
        eprintln!("{line}");
    }
    // task 260: ホスト RAM tier の当たり (`DENNO_EXPERT_RAM_MB` のときだけ 1 行)。
    if let Some(line) = denno_engine::model::expert_ram::stats_line() {
        eprintln!("{line}");
    }
    // task 271: 直前のリクエストの tier 別ヒット率をエンドポイント用に写す
    // (この直後に両方の積算を 0 に戻す)。
    record_last_tier_hits();
    denno_engine::model::expert_ram::stats_reset();
    // task 260 step 1: ホスト側のバイトの出所 (minflt / majflt / 常駐量)。
    if let Some(line) = denno_engine::model::expert_ram::probe_line(tag) {
        eprintln!("{line}");
    }
    // task 514: リクエスト終端で expert 範囲の page cache を落とす
    // (`DENNO_PF_FADV_END=1` のときだけ 1 行)。pread の読み残しだけが消える。
    if let Some(line) = denno_engine::model::expert_ram::drop_expert_pages_end(tag) {
        eprintln!("{line}");
    }
    // task 190: ヒット本数ごとのミス CPU / 上げ直しの表 (PROFILE のときだけ)。
    if tag == "decode" {
        for line in denno_engine::model::exp_iq::hit_lines(128) {
            eprintln!("{line}");
        }
    }
    denno_engine::model::exp_iq::stats_reset();
    // task 148: prefill の m 行 expert (ヒット率 / GPU 層 / 行数 / D2H と足し込みの
    // 時間 / 上げ直し)。`DENNO_EXPERT_IQ_GPU=1` かつ PF ゲート on のときだけ 1 行。
    if let Some(line) = denno_engine::model::exp_pf::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_pf::stats_reset();
    // task 261: 旧 4 型 expert の prefill GPU 経路 (1 層も回っていなければ出さない)。
    if let Some(line) = denno_engine::model::exp_pf::quant_pf_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_pf::quant_pf_reset();
    // task 158: decode のミス expert を GPU で回したぶんの内訳 (`DENNO_DEC_MISS_UP`
    // が on のときだけ 1 行)。1 本あたりのホスト写し時間と実効レートが読める。
    if let Some(line) = denno_engine::model::exp_miss_up::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_miss_up::stats_reset();
    // task 214: 背景スレッドの上げ直し (`DENNO_DEC_UP_WORKER` のときだけ 1 行)。
    if let Some(line) = denno_engine::model::exp_up_worker::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_up_worker::stats_reset();
    // task 356: expert 段の D2H を専用ストリームへ出した本数
    // (`DENNO_DEC_EXP_D2H_STREAM=1` のときだけ 1 行 = 分岐が発火した証拠)。
    if let Some(line) = denno_engine::model::exp_iq::d2h_stream_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_iq::d2h_stream_reset();
    // task 356: 和を pinned のホストへ直接書いた層数
    // (`DENNO_DEC_EXP_SUM_ZC=1` のときだけ 1 行 = 分岐が発火した証拠)。
    if let Some(line) = denno_engine::model::exp_iq::sum_zc_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_iq::sum_zc_reset();
    // task 388: ミスの CPU 計算を GPU の起動と重ねた層数 (ゲート on のときだけ 1 行)。
    if let Some(line) = denno_engine::model::exp_iq::miss_spawn_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_iq::miss_spawn_reset();
    // task 392: GDN の ba matvec を qkv GEMV と重ねた層数
    // (`DENNO_GDN_BA_PAR=1` のときだけ 1 行 = 分岐が発火した証拠)。
    if let Some(line) = denno_engine::gpu_report::gdn_ba_par_line() {
        eprintln!("{line}");
    }
    denno_engine::gpu_report::gdn_ba_par_reset();
    // task 392: GDN の入力 cur の H2D を mix のデバイス出力で置き換えた層数
    // (`DENNO_GDN_X_MIX=1` のときだけ 1 行 = 分岐が発火した証拠)。
    if let Some(line) = denno_engine::gpu_report::gdn_x_mix_line() {
        eprintln!("{line}");
    }
    // task 464: mix の鎖の D2H と待ちを省いた (ssm 層) 数。控えを使った層数
    // (`[gdn-x-mix]` と同じ数) も並べるので、`gdn_x_mix_reset` より**先**に出す。
    // ゲート on で 1 層も通らなければ行は出ない (= 分岐が発火した証拠)。
    if let Some(line) = denno_engine::gpu_report::mix_nowait_line() {
        eprintln!("{line}");
    }
    denno_engine::gpu_report::mix_nowait_reset();
    // task 468: full 層の attn mix も defer した層数 (`DENNO_MIX_NOWAIT=full` の
    // ときだけ 1 行 = 分岐が発火した証拠)。
    if let Some(line) = denno_engine::gpu_report::mix_nowait_full_line() {
        eprintln!("{line}");
    }
    denno_engine::gpu_report::mix_nowait_full_reset();
    denno_engine::gpu_report::gdn_x_mix_reset();
    // task 368: decode の待ちを null ストリームのイベントへ狭めた回数
    // (`DENNO_DEC_SYNC_STREAM=1` のときだけ 1 行 = 分岐が発火した証拠)。
    if let Some(line) = denno_engine::gpu_report::dec_sync_stream_line() {
        eprintln!("{line}");
    }
    denno_engine::gpu_report::dec_sync_stream_reset();
    // task 266: decode の先読み (`DENNO_DEC_PREF` のときだけ 1 行)。
    if let Some(line) = denno_engine::model::exp_prefetch::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_prefetch::stats_reset();
    // task 850: 層間共起の先読み (`DENNO_EXPERT_MARKOV_PREFETCH=1` のときだけ 1 行)。
    if let Some(line) = denno_engine::model::exp_markov::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_markov::stats_reset();
    // task 266: 量子型 mix の fused chain が走った回数 (`DENNO_MIX_QCHAIN` のときだけ)。
    if let Some(line) = denno_engine::model::qwen4exp::mix_qchain_line() {
        eprintln!("{line}");
    }
    denno_engine::model::qwen4exp::mix_qchain_reset();
    // task 576: shared expert の gate+up+silu 融合 (`DENNO_DEC_SH_GU` のときだけ 1 行)。
    if let Some(line) = denno_engine::gpu_report::sh_gu_line() {
        eprintln!("{line}");
    }
    // task 358: pinned バウンス (`DENNO_PF_PIN_STAGE`) が流した量 (発火しなければ出ない)。
    if let Some(line) = denno_engine::gpu_report::pf_pin_line() {
        eprintln!("{line}");
    }
    denno_engine::gpu_report::pf_pin_reset();
    // task 822: prefill の天井 (計算 vs ディスク) を 1 行で出す
    // (`DENNO_PF_ROOFLINE=1` のときだけ。off は何もしない)。
    // q4prof のゲートとは独立に出す (`DENNO_Q4_PROFILE` を立てなくても読める)。
    if let Some(line) = denno_engine::model::qwen4exp::roofline::take_line(tag) {
        eprintln!("{line}");
    }
    denno_engine::model::qwen4exp::roofline::reset();
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
    // task 132: decode の層の外の内訳 (decode の壁時計 − 層実行 を、個別に
    // 測った区間に割る)。prefill は `DENNO_PRE_KPROF` の [pre-kprof] が持つ。
    if tag == "decode" {
        if let Some(line) = q4prof::decode_outside_line(tok, wall_ns) {
            eprintln!("{line}");
        }
    }
    q4prof::reset();
}

/// P109: 貪欲 (argmax) のトークン id。spec の照合は greedy が前提。
///
/// task 112 で serve の Sampler も temperature <= 0 で greedy になった
/// (`Sampler::new` に temperature をそのまま渡す。以前は None を渡していて
/// temperature 1.0 扱い = temp 0 でも確率サンプリングになっていた)。
/// `Sampler::sample` の greedy 経路と選択規則は同じ (f32 生値の `>`、
/// 同点は添字小) だが、こちらは engine の argmax で NaN をエラーにする。
fn greedy_u32(logits: &[f32]) -> Result<u32> {
    let i = argmax(logits).map_err(|e| anyhow::anyhow!("{e}"))?;
    u32::try_from(i).map_err(|_| anyhow::anyhow!("argmax の {i} が u32 に収まらない"))
}

/// 確定トークン 1 個ぶんの本文差分処理 (P109 で spec/通常の両経路から呼ぶ)。
///
/// 従来の decode ループ内にあった処理をそのまま関数にしたもの
/// (UTF-8 未完成の末尾保留・停止語の切り詰め・差分 emit)。戻り値は
/// 「生成を続けるなら true」。停止語に当たった / emit が失敗したら false
/// (emit の失敗は `failed` に [`GenFail::Client`] として入れる。クライアント
/// 切断なので decoder は一貫したまま残る。リクエスト全体の失敗として
/// 最後に返す)。
fn emit_confirmed(
    tok: &GgufTokenizer,
    out: &mut Vec<u32>,
    text: &mut String,
    last_full: &mut Option<String>,
    stop_hit: &mut Option<String>,
    failed: &mut Option<GenFail>,
    stop_sequences: &[String],
    id: u32,
    emit: &mut dyn FnMut(&str) -> Result<()>,
) -> bool {
    out.push(id);
    // task 132: detokenize と SSE write を層の外の内訳として別々に測る。
    let d_detok = q4prof::dstamp();
    let full = tok.decode(out).unwrap_or_default();
    q4prof::dmark(q4prof::D_DETOK, d_detok);
    // 未完成の UTF-8 末尾は lossy デコードで U+FFFD になる。それをそのまま
    // emit すると、次のトークンで文字が完成したときに starts_with 照合が
    // 壊れてその文字が欠落する。末尾に連続する U+FFFD は保留し、文字が
    // 完成してから出す (本物の U+FFFD でも 1 トークン遅れで出るだけ)。
    if full.len() > text.len() && full.starts_with(text.as_str()) {
        let mut emit_end = full.len();
        while emit_end >= text.len() + 3 && full[..emit_end].ends_with('\u{FFFD}') {
            emit_end -= 3;
        }
        // 停止語が本文に現れたら、そこまでで切る
        let mut cut: Option<usize> = None;
        for s in stop_sequences {
            if let Some(i) = full.find(s.as_str()) {
                cut = Some(cut.map_or(i, |c: usize| c.min(i)));
            }
        }
        if let Some(c) = cut {
            let keep = &full[..c];
            if keep.len() > text.len() {
                let d_emit = q4prof::dstamp();
                let r = emit(&keep[text.len()..]);
                q4prof::dmark(q4prof::D_EMIT, d_emit);
                if let Err(e) = r {
                    *failed = Some(GenFail::Client(e));
                }
            }
            *text = keep.to_string();
            *last_full = Some(text.clone());
            *stop_hit = Some("stop_sequence".into());
            return false;
        }
        if emit_end > text.len() {
            let d_emit = q4prof::dstamp();
            let r = emit(&full[text.len()..emit_end]);
            q4prof::dmark(q4prof::D_EMIT, d_emit);
            if let Err(e) = r {
                *failed = Some(GenFail::Client(e));
                return false;
            }
        }
        *text = full[..emit_end].to_string();
        *last_full = Some(full);
    } else {
        *last_full = Some(full.clone());
        *text = full;
    }
    true
}

impl Flash {
    /// task 141: 描画済みプロンプトをキャッシュに整列し、食い残しだけを継いだ
    /// プロンプトを返す。
    ///
    /// 原因 (task 131): chat_template が assistant の content を `|trim` するので、
    /// 生成テキストの末尾の空白は次ターンの描画から落ちる。task 131 の splice は
    /// 「最後に返した assistant の直後」だけを継いでいたため、一度 splice した
    /// キャッシュは古い assistant の末尾空白を抱えたままになり、以降の描画とは
    /// その位置 (本番 4 ターン目で lcp=443) でずれ続けていた。
    ///
    /// ここでは描画済みトークン列 `prompt` そのものをキャッシュと突き合わせ、
    /// キャッシュ側にだけ残っている空白トークン (= `|trim` が落とした分) を
    /// 吸収する。吸収は [`GgufTokenizer::align_trimmed_prefix`] が「空白だけ」かつ
    /// 「直後が content の終端 (特殊トークン)」に限って行うので、本文が変わった
    /// 場合や別の会話は必ず full prefill に落ちる。
    ///
    /// 返り値は `c.toks ++ prompt[i..]`。`None` のとき呼び出し側は full prefill。
    /// task 131 / P812: この文脈に splice で継げるリクエストか (安い判定)。
    ///
    /// 前回のメッセージ列がそのまま前置きで、その直後に assistant の返送があり、
    /// 返送本文が実際に食った生成トークンの復号と同じときだけ真。
    fn splice_compatible(&self, c: &Cached, req: &GenReq) -> bool {
        if !c.from_render || c.prompt_len > c.toks.len() || c.req.cont || req.cont {
            return false;
        }
        if c.req.system != req.system
            || c.req.tools != req.tools
            || c.req.enable_thinking != req.enable_thinking
        {
            return false;
        }
        let n = c.req.messages.len();
        if req.messages.len() <= n || req.messages[..n] != c.req.messages[..] {
            return false;
        }
        let (role, text) = &req.messages[n];
        if role != "assistant" {
            return false;
        }
        echo_matches(&self.tok, &c.toks[c.prompt_len..], text)
    }

    fn splice_prompt(&self, c: &Cached, req: &GenReq, prompt: &[u32]) -> Result<Option<Vec<u32>>> {
        if !self.splice_compatible(c, req) {
            return Ok(None);
        }
        // 前回のメッセージ列がそのまま前置きで、その直後に assistant の返送がある。
        let n = c.req.messages.len();
        // 1. 描画済みプロンプトをキャッシュに整列する (空白の吸収だけを許す)。
        //    整列できれば、食うのは描画の食い残しだけになり、前置きが描画と
        //    対応していることも検証できる。何ターン前の assistant の末尾空白でも
        //    同じ扱いになる (task 141)。
        if let Some(i) = self.tok.align_trimmed_prefix(&c.toks, prompt) {
            if i < prompt.len() {
                let mut p = Vec::with_capacity(c.toks.len() + prompt.len() - i);
                p.extend_from_slice(&c.toks);
                p.extend_from_slice(&prompt[i..]);
                return Ok(Some(p));
            }
        }
        // 2. 整列できないとき (assistant の content が空白で *始まる* 場合など、
        //    `|trim` が先頭も落とすケース) は、これまでどおり sentinel を置いて
        //    content より後ろだけを継ぐ。前置きはキャッシュのまま。
        let spliced = self.tok.splice_after_content(
            req.system.as_deref(),
            &req.messages,
            req.tools.as_ref(),
            req.enable_thinking,
            n,
            &c.toks,
        )?;
        if let Some(p) = spliced.as_ref() {
            debug_assert_eq!(&p[..c.toks.len()], &c.toks[..], "splice が前置きを壊した");
        }
        Ok(spliced)
    }

    /// モデルを組む (mmap 越し)。
    fn build_model(&self) -> Result<Qwen4Exp> {
        #[cfg(test)]
        TEST_BUILD_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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

    /// task 155: 層活性の**タップを作るか**。
    ///
    /// [`Self::act_on`] に加えて「WS の購読者が 1 人以上いる」ことを要求する。
    /// ビューワが居ないときの層活性 (decode は 48 層 × トークンごと) は誰も
    /// 読まないので、組み立ても送出もしない。`act_on` 自体は変えない
    /// (spec の可否など、タップ以外の判断に使われているため)。
    fn tap_on(&self) -> bool {
        self.act_on() && self.telem.has_clients()
    }

    /// P53: prefill の logits。act が有効なら層の出口をタップする
    /// (`TapTokens::Last` = チャンク末尾の 1 トークンだけ)。無効時は従来の
    /// `prefill_batched` をそのまま呼ぶ (フックを付けないので数値経路は不変)。
    fn prefill_logits(&self, dec: &mut Qwen4ExpDecoder, prompt: &[u32]) -> Result<Vec<f32>> {
        // task 248: prefill の間だけ確保の台帳のピークを控える
        // (`DENNO_Q4_PROFILE` が無ければ guard は何もしない)。
        let _peak = denno_engine::mem_audit::peak_window("prefill");
        // task 256: prefill の作業域を先に予約する (`DENNO_PREFILL_WORK_MB`、
        // 既定 0 = 何もしない)。空きが足りなければチャンクを丸ごと空けて場所を作り、
        // 終わったら release する (解放したチャンクは通常の LRU が埋め直す)。
        let _work = denno_engine::model::expert_vram::reserve_work_area(
            denno_engine::model::expert_vram::prefill_work_bytes(),
        );
        if !self.tap_on() {
            return Ok(dec.prefill_batched(prompt, self.batch)?);
        }
        let mut tap = ActTap::new(&self.telem, true);
        let logits =
            dec.prefill_batched_tapped(prompt, self.batch, TapTokens::Last, |o| tap.on_layer(o))?;
        tap.report_summary("prefill");
        Ok(logits)
    }

    /// P812: スロットの文脈を借用で見る (`i == 0` が従来の主、`1` が脇)。
    fn slot_ctx(&self, i: usize) -> Option<&Cached> {
        self.slots.get(i).and_then(|s| s.ctx.as_ref())
    }

    /// P812: 従来の「主の文脈」(スロット 0)。
    fn main_ctx(&self) -> Option<&Cached> {
        self.slot_ctx(0)
    }

    /// P812: 従来の「脇の文脈」(スロット 1。スロット数 1 なら常に None)。
    fn aux_ctx(&self) -> Option<&Cached> {
        self.slot_ctx(1)
    }

    /// P812: スロットから系列状態を取り出す (move)。LRU の最終使用時刻を進める。
    fn take_slot(&mut self, i: usize) -> Option<Cached> {
        self.tick += 1;
        let t = self.tick;
        self.slots.get_mut(i).and_then(|s| {
            s.used_at = t;
            s.ctx.take()
        })
    }

    /// P812: スロットの文脈を捨てる (空になる)。
    fn drop_slot(&mut self, i: usize) {
        if let Some(s) = self.slots.get_mut(i) {
            s.ctx = None;
        }
    }

    /// P812: 全スロットの文脈を捨てる (decoder の系列状態は別)。
    fn drop_all_ctx(&mut self) {
        for s in &mut self.slots {
            s.ctx = None;
        }
    }

    /// P812: いま持っている文脈の本数とトークン数の合計 (ログ / ビューワ用)。
    fn ctx_totals(&self) -> (usize, usize) {
        self.slots
            .iter()
            .filter_map(|s| s.ctx.as_ref())
            .fold((0, 0), |(n, t), c| (n + 1, t + c.toks.len()))
    }

    /// task 206 / P812: decoder の系列状態を抜き出して、行き先のスロットへ保存する。
    ///
    /// 取り出しは move (コピーしない)。`max_tok` を超えた文脈は保存しない =
    /// その場で捨てる (メモリ上限での追い出し)。系列状態を差し替えられない
    /// モデル (QSA 有効) では文脈を残さず、decoder を空に戻す。
    #[allow(clippy::too_many_arguments)]
    fn save_ctx(
        &mut self,
        dec: &mut Qwen4ExpDecoder,
        dest: usize,
        logits: Vec<f32>,
        n_prompt: usize,
        from_render: bool,
        req: CachedReq,
        max_tok: usize,
        epoch: u64,
    ) {
        let Some(state) = dec.take_seq_state() else {
            dec.reset();
            self.drop_all_ctx();
            return;
        };
        let toks = state.tokens().to_vec();
        let c = Cached {
            state,
            toks,
            epoch,
            logits,
            prompt_len: n_prompt,
            from_render,
            req,
        };
        if c.toks.len() > max_tok || dest >= self.slots.len() {
            self.drop_slot(dest.min(self.slots.len().saturating_sub(1)));
            return;
        }
        self.tick += 1;
        let t = self.tick;
        let s = &mut self.slots[dest];
        s.used_at = t;
        s.ctx = Some(c);
    }

    /// task 217 / P812: 介入 epoch が違う文脈を捨てる (全スロット)。
    ///
    /// prefix cache の状態は「その設定で計算したモデル」の状態なので、
    /// 設定が変わったら再利用してはいけない。戻り値は捨てた本数。
    fn drop_stale_ctx(&mut self, epoch: u64) -> usize {
        let mut n = 0;
        for s in &mut self.slots {
            if s.ctx.as_ref().is_some_and(|c| c.epoch != epoch) {
                s.ctx = None;
                n += 1;
            }
        }
        n
    }

    /// task 926: `GET /v1/cache/stats` の本文 (読み取りのみ。状態は変えない)。
    fn cache_stats(&self) -> Value {
        let (used, _tokens) = self.ctx_totals();
        let (hits, misses, reused) = cache_counters();
        cache_stats_json(used, self.slots.len(), hits, misses, reused)
    }

    /// ビューワ用: いま持っている文脈 (main / aux / 全スロット) のトークン数と epoch。
    /// 文脈 = full attention の KV と GDN の再帰状態・conv 履歴・PLE conv 履歴。
    /// 残差 stream そのものはトークンごとに作り直すので、リクエストの間には残らない。
    fn ctx_json(&self) -> Value {
        let one = |c: Option<&Cached>| match c {
            Some(c) => json!({"tokens": c.toks.len(), "epoch": c.epoch}),
            None => Value::Null,
        };
        let all: Vec<Value> = self.slots.iter().map(|s| one(s.ctx.as_ref())).collect();
        json!({
            "main": one(self.main_ctx()),
            "aux": one(self.aux_ctx()),
            "n_slots": self.slots.len(),
            "slots": all,
        })
    }

    /// ビューワ用: 文脈 (全スロット) を捨て、decoder の系列状態も空に戻す。
    /// 次のリクエストは前置きを再利用せず、最初から prefill する。
    /// 介入 epoch は動かさない (介入の設定は変えていないので)。
    fn clear_ctx(&mut self) -> Value {
        let (n, tokens) = self.ctx_totals();
        self.drop_all_ctx();
        if let Some(d) = self.dec.as_mut() {
            d.reset();
        }
        eprintln!("[flash] cache clear: 文脈を {n} 本捨てた ({tokens} tok, KV / GDN 状態)");
        let mut v = self.ctx_json();
        v["dropped"] = json!(n);
        v["dropped_tokens"] = json!(tokens);
        v
    }

    /// 生成して、確定した本文を返す。`emit` は差分テキストを受け取る。
    fn generate(&mut self, req: &GenReq, mut emit: impl FnMut(&str) -> Result<()>) -> Result<GenOut> {
        // task 217: このリクエストの間は介入設定を固定する。生成中に
        // `POST /v1/intervene` が来ても、いま走っている生成の数値は変わらない
        // (次のリクエストから効く)。forward 側はピン留めを読むだけ。
        let _iv_pin = iv::pin_request();
        // 生成中の印 (`POST /v1/abort` はこの間だけ効く)。始めに古い停止要求を捨てる。
        let _gen = GenGuard::enter();
        // task 347: `num_requests_running` とリクエスト全体の時計 (TTFT / e2e)。
        let _running = metrics::RunningGuard::enter();
        let t_req = Instant::now();
        // task 218: `DENNO_STEER_RECORD` のときは 1 リクエストごとに平均を保存する
        // (停止は不要。プロセスが落ちても直前のリクエストまでが残る)。
        let _steer_flush = SteerFlushGuard;
        let iv_epoch = iv::current_epoch();
        // 別 epoch で計算した文脈は使わない (main / aux の両方)。
        let dropped = self.drop_stale_ctx(iv_epoch);
        if dropped > 0 {
            eprintln!(
                "[flash] intervene epoch {iv_epoch}: 古い文脈を {dropped} 本捨てた (prefix cache 無効化)"
            );
        }
        // task 146: 背景の事前ロードに「いま推論中」を知らせる (ロックを奪い合わない)。
        // ガードは戻るとき (失敗経路も) に下ろす。
        let _busy = denno_engine::model::exp_stats::busy_guard();
        // task 260 診断 (`DENNO_EXPERT_RAM_DROP=1`): リクエストの先頭で expert の
        // page cache を落とす (dense 重みは落とさない)。冷えた機械の再現用。
        for line in denno_engine::model::expert_ram::drop_pages_if_enabled() {
            eprintln!("{line}");
        }
        // P78: 層の外の内訳 (tokenize は prefill wall の外なので別に積む)。
        let t_tok = prekprof::stamp();
        // 継続生成 (continue: true): 最後が assistant なら、その分を閉じずに
        // 生成プロンプトの末尾へ生テキストとして継ぎ、モデルに続きを書かせる。
        // チャットテンプレートは最後の assistant ターンを <|im_end|> で閉じて
        // しまうため、そのままでは続きにならない。
        let (msgs_for_tpl, partial) =
            if req.cont && req.messages.last().map(|(r, _)| r == "assistant").unwrap_or(false) {
                let n = req.messages.len();
                (&req.messages[..n - 1], Some(req.messages[n - 1].1.clone()))
            } else {
                (&req.messages[..], None)
            };
        let mut prompt = self
            .tok
            .encode_messages_ex(
                req.system.as_deref(),
                msgs_for_tpl,
                req.tools.as_ref(),
                req.enable_thinking,
            )
            .context("プロンプトを組めない")?;
        let is_cont = partial.is_some();
        if let Some(p) = partial {
            prompt.0.extend(
                self.tok
                    .encode(&p)
                    .context("継続テキストをトークン化できない")?,
            );
        }
        let (mut prompt, mut prompt_from_render) = prompt;
        if is_cont {
            // continue 経路のプロンプトは描画結果 + 生テキストなので、
            // メッセージ列の描画とは対応しない (splice 不可)。
            prompt_from_render = false;
        }
        prekprof::mark(t_tok, prekprof::TOKENIZE);
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
        // P51: テレメトリ (観測のみ)。無効時はこの bool 1 個を見るだけで、
        // Instant も確保もしない。セッション ID は OFF でも採番する (原子加算 1 回)。
        let t_on = self.telem.on();
        let sid = telem::next_session_id();

        // 直前の状態が使えるなら差分だけ食わせる。
        // P118: モデルの使い回しはここで 1 回だけ読む (`DENNO_FLASH_MODEL_REUSE`)。
        let model_reuse = model_reuse_enabled(std::env::var("DENNO_FLASH_MODEL_REUSE").ok().as_deref());
        // task 131: テンプレートが assistant の content を `|trim` するため、
        // 生成テキスト末尾の空白は次ターンのプロンプトから落ちる。落ちたぶん
        // だけキャッシュの前置きとずれて reuse=0 (本番 462 トークンで 18 秒) に
        // なっていた。生成済みトークンをそのまま前置きに残す経路を既定で試す。
        let splice_on = model_reuse
            && prefix_splice_enabled(std::env::var("DENNO_FLASH_PREFIX_SPLICE").ok().as_deref());
        // task 206: 短い脇リクエストで主の文脈を壊さないための小さな文脈。
        let aux_on = aux_ctx_enabled(std::env::var("DENNO_FLASH_AUX_CTX").ok().as_deref());
        let aux_max_tok = aux_ctx_max_tok(std::env::var("DENNO_FLASH_AUX_TOK").ok().as_deref());
        let aux_ratio = aux_ctx_ratio(std::env::var("DENNO_FLASH_AUX_RATIO").ok().as_deref());
        let t1 = Instant::now();
        // 系列状態を差し替えられないモデル (DENNO_QSA=1) では文脈を持ち回れない。
        // 判定は前のリクエストの decoder で行う (1 本目は文脈が無いので同じ)。
        let swap_ok = self
            .dec
            .as_ref()
            .map(|d| d.seq_state_supported())
            .unwrap_or(true);
        let aux_on = aux_on && swap_ok && self.slots.len() > 1;
        // P812: スロット数 2 までは従来の task 206 の選び方 (main + aux) をそのまま
        // 使う (既定 --kv-slots 2 の挙動・メモリ量を変えないため)。3 本以上では
        // LCP 一致 + LRU の汎用経路。
        let n_slots = self.slots.len();
        // task 866: radix (パス圧縮トライ) 索引。既定 off。スロット 2 本では
        // 従来の main/aux 経路 (pick_ctx) をそのまま使うので不活性。
        let radix_on =
            kv_radix_enabled(std::env::var("DENNO_KV_RADIX").ok().as_deref()) && n_slots >= 3;
        let legacy_pick = if n_slots <= 2 {
            let lcp_main = self.main_ctx().map(|c| first_mismatch(&c.toks, &prompt));
            let lcp_aux = self.aux_ctx().map(|c| first_mismatch(&c.toks, &prompt));
            let pick = pick_ctx(
                prompt.len(),
                self.main_ctx().map(|c| (c.toks.len(), lcp_main.unwrap_or(0))),
                self.aux_ctx().map(|c| (c.toks.len(), lcp_aux.unwrap_or(0))),
                aux_on,
                aux_max_tok,
                aux_ratio,
            );
            // task 206 (3): 次のターンが実際にどこまで一致したかを毎回 1 行残す。
            // 前置きの途中 (時刻など) が変わっていれば `main=8988/5` のように出る。
            if aux_on {
                eprintln!(
                    "[flash] ctx: main={} aux={} pick={} plen={}",
                    ctx_len_lcp(self.main_ctx(), lcp_main),
                    ctx_len_lcp(self.aux_ctx(), lcp_aux),
                    pick.name(),
                    prompt.len()
                );
            }
            if pick == CtxPick::MainAfterEvict {
                // 脇の容量を超える短いリクエスト: 脇を捨ててから主へ (メモリ上限)。
                self.drop_slot(1);
            }
            #[cfg(test)]
            TEST_LAST_PICK.store(pick as u8, std::sync::atomic::Ordering::Relaxed);
            Some(pick)
        } else {
            None
        };
        // P812: 使用スロット。legacy 経路では 0 (主) / 1 (脇)。
        let mut dest_slot = match legacy_pick {
            Some(CtxPick::Aux) => 1,
            Some(_) => 0,
            None => 0,
        };
        let mut src: Option<Cached> = match legacy_pick {
            Some(CtxPick::Main | CtxPick::MainAfterEvict) => self.take_slot(0),
            Some(CtxPick::Aux) => self.take_slot(1),
            Some(CtxPick::Fresh) => None,
            None => {
                // 汎用経路: 全スロットの共通前置き長を見て 1 本選ぶ。
                // task 866: `DENNO_KV_RADIX=1` ではパス圧縮トライを 1 回なぞって
                // 全スロットの lcp を求める (プロンプト長 1 回ぶんの走査)。
                // 値は `first_mismatch` と一致する (下の debug_assert とテスト
                // `lcps_match_bruteforce` が担保)。
                let radix = radix_on.then(|| {
                    kv_radix::KvRadix::from_slots(
                        self.slots
                            .iter()
                            .map(|s| s.ctx.as_ref().map(|c| c.toks.as_slice())),
                    )
                });
                let lcps: Vec<usize> = match radix.as_ref() {
                    Some(r) => r.lcps(&prompt, n_slots),
                    None => self
                        .slots
                        .iter()
                        .map(|s| {
                            s.ctx
                                .as_ref()
                                .map(|c| first_mismatch(&c.toks, &prompt))
                                .unwrap_or(0)
                        })
                        .collect(),
                };
                debug_assert!(
                    !radix_on
                        || self
                            .slots
                            .iter()
                            .zip(&lcps)
                            .all(|(s, &l)| l
                                == s.ctx
                                    .as_ref()
                                    .map(|c| first_mismatch(&c.toks, &prompt))
                                    .unwrap_or(0)),
                    "radix の lcp が総当りと違う"
                );
                let metas: Vec<SlotMeta> = self
                    .slots
                    .iter()
                    .zip(&lcps)
                    .map(|(s, &lcp)| SlotMeta {
                        len: s.ctx.as_ref().map(|c| c.toks.len()),
                        lcp,
                        used_at: s.used_at,
                        splice: splice_on
                            && prompt_from_render
                            && s.ctx
                                .as_ref()
                                .is_some_and(|c| self.splice_compatible(c, req)),
                    })
                    .collect();
                let mut sp = pick_slot_n(prompt.len(), &metas);
                if radix_on {
                    // task 866: 追い出しだけは前置き優先にする (今回のプロンプトと
                    // 前置きを共有しない文脈から捨てる)。再利用の可否は変えない。
                    if let SlotPick::Evict(_) = sp {
                        let l: Vec<usize> = metas.iter().map(|m| m.lcp).collect();
                        let u: Vec<u64> = metas.iter().map(|m| m.used_at).collect();
                        if let Some(v) = kv_radix::evict_victim(&l, &u) {
                            sp = SlotPick::Evict(v);
                        }
                    }
                }
                dest_slot = sp.idx();
                eprintln!(
                    "[flash] ctx: slots=[{}] pick={} plen={}{}",
                    metas
                        .iter()
                        .map(|m| match m.len {
                            Some(n) => format!("{n}/{}", m.lcp),
                            None => "-".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                    sp.name(),
                    prompt.len(),
                    match radix.as_ref() {
                        Some(r) => format!(
                            " radix(nodes={} tok_sum={})",
                            r.node_count(),
                            r.total_tokens()
                        ),
                        None => String::new(),
                    }
                );
                match sp {
                    SlotPick::Reuse(i) => self.take_slot(i),
                    SlotPick::Fresh(i) => {
                        // 空きスロット (念のため残っていれば捨てる)。
                        self.drop_slot(i);
                        None
                    }
                    SlotPick::Evict(i) => {
                        let old = self.take_slot(i).map(|c| c.toks.len());
                        if let Some(n) = old {
                            eprintln!(
                                "[flash] slot {i}: {} ({} tok, スロット {n_slots} 本)",
                                if radix_on {
                                    "前置き優先の追い出し"
                                } else {
                                    "LRU 追い出し"
                                },
                                n
                            );
                        }
                        None
                    }
                }
            }
        };
        let mut reused = 0usize;
        // P812: 保存先スロットのトークン上限。従来の脇 (スロット 1) は
        // `DENNO_FLASH_AUX_TOK`、主はコンテキスト長、N スロット経路は
        // `DENNO_KV_SLOT_TOK` (既定 = コンテキスト長)。
        let save_max_tok = match legacy_pick {
            Some(CtxPick::Aux) => aux_max_tok,
            Some(_) => self.max_ctx,
            None => kv_slot_max_tok(
                std::env::var("DENNO_KV_SLOT_TOK").ok().as_deref(),
                self.max_ctx,
            ),
        };
        // キャッシュ全体がプロンプトの前置きなら、そのまま延長できる。
        let mut extend = src
            .as_ref()
            .map(|c| c.toks.len() <= prompt.len() && prompt[..c.toks.len()] == c.toks[..])
            .unwrap_or(false);
        // task 131/141: 描画済みプロンプトをキャッシュに整列し、食い残しだけを食う。
        let mut spliced: Option<Vec<u32>> = None;
        if !extend && splice_on && prompt_from_render {
            if let Some(c) = src.as_ref() {
                spliced = self.splice_prompt(c, req, &prompt)?;
            }
        }
        // モデル (重み) は decoder に 1 つだけ持ち、リクエストを跨いで使い回す。
        // 前置きを延ばす / splice するリクエストは必ず使い回す。それ以外は
        // `DENNO_FLASH_MODEL_REUSE=0` のとき従来どおり組み直す (gate の意味を保つ)。
        // (P118: 組み直すと `GgufLinear` に新しい uid が振られ、GPU 常駐の重みを
        // 全部再 upload する = cache-miss 1 回あたり 266 本 / 3.42 秒の無駄。)
        let reuse_model = model_reuse || extend || spliced.is_some();
        let mut dec = match self.dec.take() {
            Some(d) if reuse_model => d,
            other => {
                drop(other);
                Qwen4ExpDecoder::new(self.build_model()?, DecoderConfig::new(self.max_ctx))
            }
        };
        if !dec.seq_state_supported() {
            // 文脈を持ち回れない: 選んだ文脈も捨てて full prefill に落ちる。
            static WARNED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "[flash] 系列状態を差し替えられない (DENNO_QSA=1) ので prefix cache を使わない"
                );
            }
            self.drop_all_ctx();
            drop(src.take());
            extend = false;
            spliced = None;
        }
        let mut logits;
        if let Some(np) = spliced {
            if np.len() >= self.max_ctx {
                bail!(
                    "プロンプト {} トークンが上限 {} に収まらない",
                    np.len(),
                    self.max_ctx
                );
            }
            let Cached {
                state,
                toks,
                logits: _,
                ..
            } = src.take().expect("splice は src があるときだけ返る");
            reused = toks.len();
            prompt = np;
            dec.put_seq_state(state)
                .map_err(|e| anyhow::anyhow!("系列状態を戻せない: {e}"))?;
            logits = self.prefill_logits(&mut dec, &prompt[reused..])?;
        } else if extend {
            let Cached {
                state,
                toks,
                logits: cached_logits,
                ..
            } = src.take().expect("extend は src があるときだけ");
            reused = toks.len();
            dec.put_seq_state(state)
                .map_err(|e| anyhow::anyhow!("系列状態を戻せない: {e}"))?;
            logits = if reused < prompt.len() {
                self.prefill_logits(&mut dec, &prompt[reused..])?
            } else {
                cached_logits
            };
        } else {
            // 落ちた原因を 1 行残す (空でないキャッシュで reuse=0 のときだけ)。
            if let Some(c) = src.as_ref() {
                if !c.toks.is_empty() {
                    eprintln!("{}", prefix_miss_line(&self.tok, &c.toks, &prompt));
                }
            }
            // 置き換え: 選んだ文脈の状態は捨てる (decoder は空のまま)。
            // P118: `build_model` は呼ばない — 重みはそのまま常駐する。
            drop(src.take());
            debug_assert!(
                dec.tokens().is_empty(),
                "置き換え経路で decoder に状態が残っている"
            );
            let t_model = prekprof::stamp();
            prekprof::mark(t_model, prekprof::MODEL);
            logits = self.prefill_logits(&mut dec, &prompt)?;
        }
        let t_prefill = t1.elapsed().as_secs_f64();
        // task 926: prefix cache の利用実績をこのリクエスト 1 本ぶん数える
        // (前置きを使い回したら hit、フル prefill なら miss)。整数加算だけで、
        // ホットパスの数値経路には触れない。
        record_cache_reuse(reused);
        let n_prompt = prompt.len();
        // 次のリクエストが同じ入力を送ってきたら splice できるように、
        // このリクエストのプロンプト構築入力を残す (task 131)。
        let req_sig = CachedReq {
            system: req.system.clone(),
            messages: req.messages.clone(),
            tools: req.tools.clone(),
            enable_thinking: req.enable_thinking,
            cont: req.cont,
        };
        // task 123: テスト用の再利用トークン数の記録 (cfg(test) のみ)。
        #[cfg(test)]
        TEST_LAST_REUSE.store(reused, std::sync::atomic::Ordering::Relaxed);

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

        prof_flash("prefill", n_prompt - reused, (t_prefill * 1e9) as u64);
        // task 112: temperature をそのまま渡す。temperature <= 0 は
        // `Sampler::sample` 内の greedy 経路 (argmax、同点は添字小) になる。
        // 以前は None を渡していて `Sampler::new` が temperature 1.0 扱いにして
        // いたため、spec off の temp 0 が貪欲にならず、spec on (engine の
        // argmax) と出力がずれていた (accept=0 でも不一致になる原因)。
        let mut sampler = Sampler::new(Some(req.temperature), req.top_k, req.top_p, Some(42));
        let eos = self.tok.eos();
        let mut out: Vec<u32> = Vec::new();
        let mut text = String::new();
        // 最後に decode(&out) した全文 (保留した末尾 FFFD を含む)。
        // ループを抜けたとき text へ反映するのに使う。
        let mut last_full: Option<String> = None;
        let mut stop_hit: Option<String> = None;
        let mut failed: Option<GenFail> = None;
        // P53: decode は 1 トークンあたり全層ぶん流す (act が有効で、かつ
        // ビューワが繋がっているときだけ)。
        let mut act = if self.tap_on() {
            Some(ActTap::new(&self.telem, false))
        } else {
            None
        };
        // P109: n-gram 投機的デコード (greedy 専用、既定 off)。
        // - `DENNO_SPEC=1` で有効
        // - greedy (temperature <= 0) のときだけ (spec は貪欲照合が前提)
        // - テレメトリの層活性タップ (ACT) 有効時は off (tap は Step 幅 1 前提)
        // - `DENNO_SPEC_K` で draft の最大長 (既定 4、上限 16)
        // - `DENNO_SPEC_MIN_P` で draft 候補の先頭トークン確率の下限
        //   (既定 0 = 無効。task 112。候補の生成だけに効く)
        let spec_on = env_flag("DENNO_SPEC") && req.temperature <= 0.0 && !self.act_on();
        let spec_k = if spec_on {
            std::env::var("DENNO_SPEC_K")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .map(|k| k.clamp(1, 16))
                .unwrap_or(4)
        } else {
            0
        };
        let spec_min_p = if spec_on {
            std::env::var("DENNO_SPEC_MIN_P")
                .ok()
                .and_then(|s| s.parse::<f32>().ok())
                .map(|p| p.clamp(0.0, 1.0))
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let mut ngram =
            NgramIndex::new(DEFAULT_MAX_SUFFIX, DEFAULT_MAX_ENTRIES, DEFAULT_CAND_KEEP, spec_k);
        if spec_on {
            ngram.feed(dec.tokens());
        }
        // P109: 受理率の計測 (DENNO_SPEC_STATS=1 のときだけ生成終了時に 1 行出す)。
        // 加算だけなので数値経路には触れない。
        let mut spec_steps = 0usize;
        let mut spec_confirmed = 0usize;
        let mut spec_accepted = 0usize;
        let mut spec_offered = 0usize;
        let t2 = Instant::now();
        // task 347: 最初の 1 トークンが出た時刻 (TTFT / inter-token)。
        // ホットループでは分岐 1 回と `Option` の代入だけで、ロックも確保もしない。
        let mut t_first: Option<Instant> = None;
        // task 153: decode の窓で CUDA イベントの時系列を取る (`DENNO_Q4_TLINE`)。
        // off では atomic load 1 回で、イベントは 1 本も打たない。
        q4ledger::tline_begin();
        'gen: while out.len() < req.max_tokens {
            // `POST /v1/abort`: 次のトークンを作らずに終わる。ここでは out の全トークンが
            // step 済みなので、文脈は max_tokens で止まったときと同じく一貫している。
            if take_abort() {
                eprintln!("[flash] abort: {} トークンで停止", out.len());
                break 'gen;
            }
            if spec_on {
                // P109: n-gram 投機ステップ。q0 は engine の argmax (greedy)。
                let d_topk = q4prof::dstamp();
                // task 155: top-8 はビューワ (WS) が繋がっているときだけ作る。
                // 読む相手が居ないときの出力は従来と同じ (誰も受け取らない)。
                let cand =
                    if t_on && self.telem.has_clients() && denno_telemetry::topk_telemetry_enabled()
                    {
                        Some(denno_telemetry::topk_softmax(&logits, 8))
                    } else {
                        None
                    };
                q4prof::dmark(q4prof::D_TOPK, d_topk);
                let t_s = prekprof::stamp();
                let d_sample = q4prof::dstamp();
                let q0 = match greedy_u32(&logits) {
                    Ok(v) => v,
                    Err(e) => {
                        failed = Some(GenFail::Engine(e));
                        break;
                    }
                };
                prekprof::mark(t_s, prekprof::SAMPLE);
                q4prof::dmark(q4prof::D_SAMPLE, d_sample);
                prekprof::note_sample();
                if Some(q0) == eos {
                    break;
                }
                // task 112: 辞書は「q0 を含む末尾」で引く。spec_verify は q0 を
                // 先に食い、draft[0] を「q0 の次」の貪欲予測と照合するので、
                // q0 を含めないで引くと継続の先頭 (= q0 スロットのトークン) が
                // draft[0] になり、パターンが繰り返すほど最初の照合で必ず壊れる
                // (accept=0 の原因)。min_p > 0 なら低確率の候補は出さない。
                let draft = offer_draft(&ngram, dec.tokens(), q0, spec_k, &logits, spec_min_p);
                let t_step = t_on.then(Instant::now);
                let outcome = match dec.spec_verify(q0, &draft, eos) {
                    Ok(o) => o,
                    Err(e) => {
                        failed = Some(GenFail::Engine(e.into()));
                        break;
                    }
                };
                // task 123: emit 失敗で抜けるときも logits を decoder 状態に
                // 揃える (spec_verify は確定トークンを先に step するので、古い
                // logits のままだと cache の logits と decoder がずれる)。
                // 成功終端で代入していたのを前倒ししただけで、値は同じ。
                logits = outcome.logits;
                ngram.feed(dec.tokens());
                spec_steps += 1;
                spec_confirmed += outcome.confirmed.len();
                spec_accepted += outcome.n_accepted;
                spec_offered += draft.len();
                let pos0 = dec.tokens().len() - outcome.confirmed.len();
                // max_tokens に収まるぶんだけ出す (はみ出した確定トークンはモデルの
                // 状態には残るが本文には出さない = max_tokens を厳守する)。
                let room = req.max_tokens - out.len();
                let n_emit = outcome.confirmed.len().min(room);
                for &id in outcome.confirmed[..n_emit].iter() {
                    if !emit_confirmed(
                        &self.tok,
                        &mut out,
                        &mut text,
                        &mut last_full,
                        &mut stop_hit,
                        &mut failed,
                        &req.stop_sequences,
                        id,
                        &mut emit,
                    ) {
                        break 'gen;
                    }
                }
                // task 347: 最初の 1 トークン (TTFT)。分岐 1 回だけで、
                // ヒストグラムへの加算もリクエストあたり 1 回しかしない。
                if t_first.is_none() && !out.is_empty() {
                    let now = Instant::now();
                    t_first = Some(now);
                    metrics::note_first_token((now - t_req).as_secs_f64());
                }
                if t_on {
                    if let Some(t0) = t_step {
                        // Step の width は確定トークン数 (通常経路は 1)。
                        self.telem.emit(Frame::Step {
                            width: n_emit as u32,
                            ns: t0.elapsed().as_nanos() as u64,
                        });
                    }
                    for (i, &id) in outcome.confirmed[..n_emit].iter().enumerate() {
                        self.telem.emit(Frame::Token {
                            session: sid,
                            token: id,
                            pos: (pos0 + i + 1) as u32,
                        });
                    }
                    if let Some(tk) = cand.as_ref() {
                        self.telem.emit(Frame::TopK {
                            session: sid,
                            chosen: q0,
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
                if n_emit < outcome.confirmed.len() {
                    // max_tokens で満杯 (残りの確定トークンは出さない)。
                    break;
                }
                if outcome.eos {
                    break;
                }
                continue;
            }
            // ---- 通常経路 (P108 までと同じ。emit の差分処理を helper 化) ----
            // P58: 選ぶ前の分布を top-8 で送る (3D ビューワの候補表示用)。
            // テレメトリ OFF なら計算しない。数値経路には触れない。
            let d_topk = q4prof::dstamp();
            let cand = {
                // task 153: 台帳の区間 (topk)。テレメトリ用の top-8 を選ぶだけの時間。
                // task 155: ビューワ (WS) が繋がっていないときは作らない
                // (248k logits の 1 パスが毎トークン ~1 ms かかる)。繋がって
                // いる間の出力は従来と同一。
                let _l = q4ledger::enter(q4ledger::TOPK);
                if t_on && self.telem.has_clients() && denno_telemetry::topk_telemetry_enabled() {
                    Some(denno_telemetry::topk_softmax(&logits, 8))
                } else {
                    None
                }
            };
            q4prof::dmark(q4prof::D_TOPK, d_topk);
            // P78: サンプリングは decode 側 (prefill wall の外)。
            let t_s = prekprof::stamp();
            let d_sample = q4prof::dstamp();
            // task 153: 台帳の区間 (sample) と (step)。step の中の名前付き区間は
            // 入れ子で入るので、step 行は「層と層の外のうち名前が付いていない分」になる。
            let id = {
                let _l = q4ledger::enter(q4ledger::SAMPLE);
                sampler.sample(&logits) as u32
            };
            prekprof::mark(t_s, prekprof::SAMPLE);
            q4prof::dmark(q4prof::D_SAMPLE, d_sample);
            prekprof::note_sample();
            if Some(id) == eos {
                break;
            }
            let ok = {
                // task 153: 台帳の区間 (detok)。出力トークンの切り出しと本文への反映。
                let _l = q4ledger::enter(q4ledger::DETOK);
                emit_confirmed(
                    &self.tok,
                    &mut out,
                    &mut text,
                    &mut last_full,
                    &mut stop_hit,
                    &mut failed,
                    &req.stop_sequences,
                    id,
                    &mut emit,
                )
            };
            if !ok {
                break;
            }
            // task 347: 最初の 1 トークン (TTFT)。分岐 1 回だけ。
            if t_first.is_none() {
                let now = Instant::now();
                t_first = Some(now);
                metrics::note_first_token((now - t_req).as_secs_f64());
            }
            let t_step = t_on.then(Instant::now);
            let pos = dec.tokens().len() as u32;
            // task 123: engine 側失敗の注入 (cfg(test) のみ。本番では存在しない)。
            // emit 成功後・step 前に落として、cache が残らないことを確かめる。
            #[cfg(test)]
            if TEST_FAIL_STEP.load(std::sync::atomic::Ordering::Relaxed) {
                failed = Some(GenFail::Engine(anyhow::anyhow!("test: step 失敗を注入")));
                break;
            }
            let stepped = match act.as_mut() {
                Some(tap) => {
                    let _l = q4ledger::enter(q4ledger::STEP);
                    dec.step_tapped(id, |o| {
                        // task 153: 層ごとの活性タップ (LayerAct の組み立て) を区間にする。
                        let _t = q4ledger::enter(q4ledger::TAP);
                        tap.on_layer(o)
                    })
                }
                None => {
                    let _l = q4ledger::enter(q4ledger::STEP);
                    dec.step(id)
                }
            };
            match stepped {
                Ok(lg) => logits = lg,
                Err(e) => {
                    failed = Some(GenFail::Engine(e.into()));
                    break;
                }
            }
            // task 132: テレメトリのフレーム送出 (Step / Token / TopK / KvUsage)
            // も層の外。on でなければブロックごと無い (タイマーは 1 回だけ)。
            let d_telem = q4prof::dstamp();
            if let Some(t0) = t_step {
                // task 153: 台帳の区間 (telem)。WS クライアントへのフレーム組み立てと送出。
                let _l = q4ledger::enter(q4ledger::TELEM);
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
            q4prof::dmark(q4prof::D_TELEM, d_telem);
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
        // task 347: `/metrics` の計器 (リクエスト終端 = ホットループの外)。
        // ゲート off なら atomic load 1 回だけで、以下のサンプルは走らない。
        if metrics::enabled() {
            let e2e = t_req.elapsed().as_secs_f64();
            let n_out = out.len();
            let inter = match t_first {
                Some(t) if n_out >= 2 => {
                    // decode 窓の終わり (t2 から) から最初の 1 トークンまでを引く。
                    let span = t2.elapsed().saturating_sub(t - t2);
                    Some(span.as_secs_f64() / (n_out - 1) as f64)
                }
                _ => None,
            };
            // expert の VRAM ヒット率 (直前のリクエスト)。
            let (lookups, hits) = denno_engine::model::exp_iq::lookups_hits();
            if lookups > 0 {
                metrics::record_expert_vram(Some(hits.min(lookups) as f64 / lookups as f64));
            }
            // VRAM / RAM tier のゲージとディスク読み (ここでだけロックを取る)。
            metrics::sample_tiers();
            metrics::sample_disk_reads();
            metrics::record_request(&metrics::ReqSample {
                prompt_tokens: n_prompt as u64,
                reused_tokens: reused as u64,
                output_tokens: n_out as u64,
                prefill_secs: t_prefill,
                e2e_secs: e2e,
                inter_token_secs: inter,
                decode_ms_per_token: if n_out > 0 {
                    Some(t_decode * 1e3 / n_out as f64)
                } else {
                    None
                },
                failed: failed.is_some(),
                ctx_tokens: dec.tokens().len() as u64,
            });
        }
        // P109: `DENNO_SPEC_STATS=1` のときだけ spec ステップの受理率を 1 行出す
        // (恒久診断。env 未指定では何も出さない。加算だけなので出力は変わらない)。
        if spec_on && env_flag("DENNO_SPEC_STATS") {
            let rate = if spec_offered > 0 {
                spec_accepted as f64 / spec_offered as f64
            } else {
                0.0
            };
            eprintln!(
                "[SPEC] steps={spec_steps} confirmed={spec_confirmed} accepted={spec_accepted} offered={spec_offered} accept={rate:.3} tok/step={:.2}",
                spec_confirmed as f64 / spec_steps.max(1) as f64
            );
        }
        // task 120: `DENNO_EXPERT_W_STATS=1` のときだけ decode の重み統計を 1 行出す
        // (リクエスト終端。env 未指定では分岐 1 回で何も出さない)。
        expert_w_stats_request_end();
        // task 153: 時系列の記録を閉じる (decode の台帳を出す直前に 1 回)。
        q4ledger::tline_end();
        prof_flash("decode", out.len(), (t_decode * 1e9) as u64);
        // P78: `DENNO_PRE_KPROF=1` のときだけ、層の外の区間を 2 行で出す
        // (既定 off では `report` が空を返す = 出力ゼロ)。
        for line in
            prekprof::report("prefill", (n_prompt - reused).max(1), (t_prefill * 1e9) as u64)
        {
            eprintln!("{line}");
        }
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
        // task 123: 失敗の種別で後始末を分ける。
        // - Client (emit 断線): decoder は完了した step まで一貫している
        //   (通常経路は emit → step の順なので、失敗した差分のトークンはまだ
        //   step していない。spec 経路は spec_verify が確定トークンを先に
        //   step するが、tokens と内部状態は一致している) ので cache に保存
        //   する。次のリクエストが前置きを延長していれば差分 prefill で
        //   使い回し、一致しなければ reset する (どちらも build_model は
        //   走らない)。gate (`DENNO_FLASH_MODEL_REUSE=0`) off のときは従来
        //   どおり捨てる (gate off は前置き一致経路も使わない、という今日の
        //   挙動を保つ)。
        // - Engine: 半端に進んだ decoder は信用しない。従来どおり捨てる
        //   (cache は None のまま = 次のリクエストで組み直す)。
        if let Some(GenFail::Client(e)) = failed {
            if model_reuse {
                self.save_ctx(
                    &mut dec,
                    dest_slot,
                    logits,
                    n_prompt,
                    prompt_from_render,
                    req_sig.clone(),
                    save_max_tok,
                    iv_epoch,
                );
                self.dec = Some(dec);
            } else {
                self.drop_all_ctx();
                self.dec = None;
            }
            return Err(e);
        }
        if let Some(GenFail::Engine(e)) = failed {
            // 半端に進んだ decoder は信用しない: モデルごと捨てて次で組み直す。
            // (他の文脈も一緒に捨てる。task 123 の従来の後始末と同じ。)
            self.drop_all_ctx();
            self.dec = None;
            return Err(e);
        }
        // 保留していた末尾 (文字が完成しないまま終わった FFFD を含む) を最終本文へ反映。
        // stop_sequence で切ったときは last_full = text にしてあるので伸びない。
        if let Some(f) = last_full {
            if f.len() > text.len() && f.starts_with(&text) {
                text = f;
            }
        }
        self.save_ctx(
            &mut dec,
            dest_slot,
            logits,
            n_prompt,
            prompt_from_render,
            req_sig,
            save_max_tok,
            iv_epoch,
        );
        self.dec = Some(dec);
        // task 146: リクエスト終端で使用回数を書く (decode の外・原子的)。
        if let Some(line) = denno_engine::model::exp_stats::save_now() {
            eprintln!("{line}");
        }
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

/// task 347: Prometheus の exposition (text format 0.0.4)。
fn send_metrics(stream: &mut TcpStream, body: &str) -> Result<()> {
    head(
        stream,
        "200 OK",
        "text/plain; version=0.0.4; charset=utf-8",
        &format!("Content-Length: {}\r\n", body.len()),
    )?;
    stream.write_all(body.as_bytes())?;
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
///
/// task 347: sparkDash (vLLM の reader) が「denno のエンジン」と判別できるように
/// `/v1/models` の要素へ `owned_by: "denno"` を足す (`DENNO_MODELS_OWNED_BY=0` で
/// 従来の形に戻る)。他のフィールドは変えない。
///
/// task 817: 運用のためのメタデータ (`context_length` / `capabilities`) を足す。
/// 既存クライアントは未知フィールドを無視する。
fn get_static(path: &str, name: &str, max_ctx: usize) -> Option<Value> {
    match path {
        "/health" | "/healthz" => Some(json!({"ok": true, "model": name})),
        "/v1/models" => {
            let mut m = json!({
                "id": name, "object": "model", "type": "model",
                "display_name": name, "created_at": "2026-10-27T00:00:00Z",
                "context_length": max_ctx,
                "capabilities": {
                    "anthropic": true,
                    "openai": true,
                    "tools": true,
                    "streaming": true,
                    "telemetry": telem::enabled(),
                },
            });
            if metrics::enabled_from_value(std::env::var(metrics::ENV_OWNED_BY).ok().as_deref()) {
                m["owned_by"] = json!("denno");
            }
            Some(json!({"object": "list", "data": [m]}))
        }
        _ => None,
    }
}

/// P817: `GET /ready` の応答 (本文 + HTTP ステータス)。モデルのロードは
/// listen 前に終わっているので、ready は「drain していない」で決まる。
/// drain 中は 503 (オーケストレータが新規ルーティングを止める)。
fn ready_response(name: &str, max_ctx: usize) -> (&'static str, Value) {
    let draining = is_draining();
    let in_flight = IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst);
    let body = json!({
        "status": if draining { "draining" } else { "ready" },
        "ready": !draining,
        "draining": draining,
        "model_loaded": true,
        "model": name,
        "context_length": max_ctx,
        "in_flight": in_flight,
    });
    if draining { ("503 Service Unavailable", body) } else { ("200 OK", body) }
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


// ---------------------------------------------------------------------------
// task 217: 介入の HTTP (<-> engine::intervene)
// ---------------------------------------------------------------------------

/// 現在の設定を `GET /v1/intervene` の形にする。
fn intervene_to_json(cfg: &iv::Config) -> Value {
    let experts: Vec<Value> = cfg
        .experts_off()
        .iter()
        .map(|&(l, e)| json!([l, e]))
        .collect();
    let blocks: Vec<Value> = cfg
        .blocks_off()
        .iter()
        .map(|&(l, b)| json!([l, b.name()]))
        .collect();
    let streams: Vec<Value> = cfg
        .streams_off()
        .iter()
        .map(|&(l, s)| json!([l, s]))
        .collect();
    json!({
        "epoch": cfg.epoch(),
        "experts_off": experts,
        "blocks_off": blocks,
        "streams_off": streams,
    })
}

fn as_index(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("{what} は 0 以上の整数"))
}

fn as_pair_array<'a>(v: &'a Value, what: &str) -> Result<&'a Vec<Value>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} は配列 [[layer, x], ...]"))
}

/// `POST /v1/intervene` の body を検証して設定を置き換える。
///
/// 失敗はメッセージ (HTTP 400) にする。**panic しない**のが契約。
fn apply_intervene(shape: &ModelShape, body: &Value) -> Result<Value, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "body は JSON オブジェクト".to_string())?;
    for k in obj.keys() {
        if !matches!(k.as_str(), "experts_off" | "blocks_off" | "streams_off") {
            return Err(format!(
                "未知のフィールド {k:?} (experts_off / blocks_off / streams_off のみ)"
            ));
        }
    }
    let mut experts: Vec<(usize, usize)> = Vec::new();
    if let Some(v) = obj.get("experts_off") {
        for (i, item) in as_pair_array(v, "experts_off")?.iter().enumerate() {
            let p = item
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("experts_off[{i}] は [layer, expert]"))?;
            experts.push((as_index(&p[0], "layer")?, as_index(&p[1], "expert")?));
        }
    }
    let mut blocks: Vec<(usize, iv::Block)> = Vec::new();
    if let Some(v) = obj.get("blocks_off") {
        for (i, item) in as_pair_array(v, "blocks_off")?.iter().enumerate() {
            let p = item
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("blocks_off[{i}] は [layer, \"attn\"|\"moe\"|\"shared\"|\"ple\"]"))?;
            let name = p[1]
                .as_str()
                .ok_or_else(|| format!("blocks_off[{i}][1] は文字列"))?;
            let b = iv::Block::parse(name).ok_or_else(|| {
                format!("blocks_off[{i}]: 未知の block {name:?} (attn / moe / shared / ple)")
            })?;
            blocks.push((as_index(&p[0], "layer")?, b));
        }
    }
    let mut streams: Vec<(usize, usize)> = Vec::new();
    if let Some(v) = obj.get("streams_off") {
        for (i, item) in as_pair_array(v, "streams_off")?.iter().enumerate() {
            let p = item
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("streams_off[{i}] は [layer, stream]"))?;
            streams.push((as_index(&p[0], "layer")?, as_index(&p[1], "stream")?));
        }
    }
    let cfg = iv::set_config(&shape.engine_shape(), experts, blocks, streams)?;
    Ok(intervene_to_json(&cfg))
}

// ---------------------------------------------------------------------------
// task 218: ステアリング (残差の平均の記録 / 適用) の HTTP
// ---------------------------------------------------------------------------

/// `GET /v1/steer` の形。
fn steer_status_to_json() -> Value {
    let recording = match steer::rec_status() {
        Some((name, tokens)) => json!({"name": name, "tokens": tokens}),
        None => Value::Null,
    };
    let snaps: Vec<Value> = steer::snapshots()
        .iter()
        .map(|s| {
            json!({
                "name": s.name,
                "tokens": s.tokens,
                "created": s.created,
                "layers": s.layers,
            })
        })
        .collect();
    let apply = match steer::apply_status() {
        Some(a) => json!({
            "terms": a
                .terms
                .iter()
                .map(|t| json!({"snapshot": t.snapshot, "coef": t.coef}))
                .collect::<Vec<_>>(),
            "layers": [a.from, a.to],
            "alpha": a.alpha,
            "mode": a.mode.name(),
            // task 849: 重ね掛けの規則 (空配列なら従来の単一規則)。
            "rules": a
                .rules
                .iter()
                .map(|r| json!({
                    "terms": r
                        .terms
                        .iter()
                        .map(|t| json!({"snapshot": t.snapshot, "coef": t.coef}))
                        .collect::<Vec<_>>(),
                    "layers": [r.from, r.to],
                    "alpha": r.alpha,
                    "mode": r.mode.name(),
                }))
                .collect::<Vec<_>>(),
        }),
        None => Value::Null,
    };
    json!({
        "epoch": iv::current_epoch(),
        "capture_mode": steer::capture_mode().name(),
        "recording": recording,
        "snapshots": snaps,
        "apply": apply,
    })
}

/// 400 の本文 (task 217 と同じ形)。
fn err_json(msg: &str) -> Value {
    json!({"type": "error", "error": {"type": "invalid_request_error", "message": msg}})
}

fn as_f32(v: &Value, what: &str) -> Result<f32, String> {
    let x = v
        .as_f64()
        .ok_or_else(|| format!("{what} は数値 (JSON の数)"))?;
    if !x.is_finite() || x.abs() > 1e30 {
        return Err(format!("{what} が有限の範囲に無い"));
    }
    Ok(x as f32)
}

/// task 849: `/v1/steer/apply` の `rules` の上限 (画面は 4 まで)。
const MAX_STEER_RULES: usize = 8;

/// `POST /v1/steer/apply` の body を検証して適用する。**panic しない**。
fn apply_steer(body: &Value) -> Result<Value, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "body は JSON オブジェクト".to_string())?;
    for k in obj.keys() {
        if !matches!(k.as_str(), "terms" | "layers" | "alpha" | "mode" | "rules") {
            return Err(format!(
                "未知のフィールド {k:?} (terms / layers / alpha / mode / rules のみ)"
            ));
        }
    }
    // task 849: 複数概念の重ね掛け。`rules: [{terms, layers, alpha, mode}]` は
    // 単一規則の `terms` / `layers` / ... と同時には使えない (規則ごとに持つ)。
    if let Some(rv) = obj.get("rules") {
        for k in ["terms", "layers", "alpha", "mode"] {
            if obj.contains_key(k) {
                return Err(format!(
                    "rules と {k} は同時に使えない (層範囲・alpha・mode は規則ごとに書く)"
                ));
            }
        }
        let arr = rv
            .as_array()
            .ok_or_else(|| "rules は配列".to_string())?;
        if arr.is_empty() {
            return Err("rules が空 (消すなら DELETE /v1/steer/apply)".into());
        }
        if arr.len() > MAX_STEER_RULES {
            return Err(format!(
                "rules が多すぎる ({} 個 / 上限 {MAX_STEER_RULES})",
                arr.len()
            ));
        }
        let mut specs = Vec::with_capacity(arr.len());
        for (i, r) in arr.iter().enumerate() {
            let o = r
                .as_object()
                .ok_or_else(|| format!("rules[{i}] はオブジェクト"))?;
            for k in o.keys() {
                if !matches!(k.as_str(), "terms" | "layers" | "alpha" | "mode") {
                    return Err(format!("rules[{i}]: 未知のフィールド {k:?}"));
                }
            }
            let tv = o
                .get("terms")
                .ok_or_else(|| format!("rules[{i}].terms が無い"))?
                .as_array()
                .ok_or_else(|| format!("rules[{i}].terms は配列"))?;
            if tv.is_empty() {
                return Err(format!("rules[{i}].terms が空"));
            }
            let mut terms = Vec::with_capacity(tv.len());
            for (j, t) in tv.iter().enumerate() {
                let to = t
                    .as_object()
                    .ok_or_else(|| format!("rules[{i}].terms[{j}] はオブジェクト"))?;
                for k in to.keys() {
                    if !matches!(k.as_str(), "snapshot" | "coef") {
                        return Err(format!(
                            "rules[{i}].terms[{j}]: 未知のフィールド {k:?}"
                        ));
                    }
                }
                let name = to
                    .get("snapshot")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("rules[{i}].terms[{j}].snapshot は文字列"))?;
                let coef = as_f32(
                    to.get("coef")
                        .ok_or_else(|| format!("rules[{i}].terms[{j}].coef が無い"))?,
                    &format!("rules[{i}].terms[{j}].coef"),
                )?;
                terms.push(steer::Term {
                    snapshot: name.to_string(),
                    coef,
                });
            }
            let lv = o
                .get("layers")
                .ok_or_else(|| format!("rules[{i}].layers が無い"))?
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| format!("rules[{i}].layers は [from, to]"))?;
            let from = as_index(&lv[0], &format!("rules[{i}].layers[0]"))?;
            let to = as_index(&lv[1], &format!("rules[{i}].layers[1]"))?;
            let alpha = match o.get("alpha") {
                Some(v) => as_f32(v, &format!("rules[{i}].alpha"))?,
                None => 1.0,
            };
            let mode = match o.get("mode") {
                Some(v) => {
                    let s = v
                        .as_str()
                        .ok_or_else(|| format!("rules[{i}].mode は文字列"))?;
                    steer::Mode::parse(s)
                        .ok_or_else(|| format!("rules[{i}]: 未知の mode {s:?} (subtract / project)"))?
                }
                None => steer::Mode::Subtract,
            };
            specs.push(steer::RuleSpec {
                terms,
                from,
                to,
                alpha,
                mode,
            });
        }
        steer::set_rules(specs)?;
        return Ok(steer_status_to_json());
    }
    let tv = obj
        .get("terms")
        .ok_or_else(|| "terms が無い".to_string())?
        .as_array()
        .ok_or_else(|| "terms は配列".to_string())?;
    if tv.is_empty() {
        return Err("terms が空 (消すなら DELETE /v1/steer/apply)".into());
    }
    let mut terms = Vec::with_capacity(tv.len());
    for (i, t) in tv.iter().enumerate() {
        let o = t
            .as_object()
            .ok_or_else(|| format!("terms[{i}] はオブジェクト"))?;
        for k in o.keys() {
            if !matches!(k.as_str(), "snapshot" | "coef") {
                return Err(format!("terms[{i}]: 未知のフィールド {k:?}"));
            }
        }
        let name = o
            .get("snapshot")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("terms[{i}].snapshot は文字列"))?;
        let coef = as_f32(
            o.get("coef")
                .ok_or_else(|| format!("terms[{i}].coef が無い"))?,
            &format!("terms[{i}].coef"),
        )?;
        terms.push(steer::Term {
            snapshot: name.to_string(),
            coef,
        });
    }
    let lv = obj
        .get("layers")
        .ok_or_else(|| "layers が無い".to_string())?
        .as_array()
        .filter(|a| a.len() == 2)
        .ok_or_else(|| "layers は [from, to]".to_string())?;
    let from = as_index(&lv[0], "layers[0]")?;
    let to = as_index(&lv[1], "layers[1]")?;
    let alpha = match obj.get("alpha") {
        Some(v) => as_f32(v, "alpha")?,
        None => 1.0,
    };
    let mode = match obj.get("mode") {
        Some(v) => {
            let s = v.as_str().ok_or_else(|| "mode は文字列".to_string())?;
            steer::Mode::parse(s)
                .ok_or_else(|| format!("未知の mode {s:?} (subtract / project)"))?
        }
        None => steer::Mode::Subtract,
    };
    steer::set_apply(terms, from, to, alpha, mode)?;
    Ok(steer_status_to_json())
}

/// `POST /v1/steer/record` の body から名前と capture_mode を取る。
///
/// `capture_mode` は省略時 `"all"`。`"all"` と `"last_prompt_token"` を受け、
/// それ以外は 400 (未知の値を黙って `all` にしない)。
fn record_params(body: &Value) -> Result<(String, steer::CaptureMode), String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "body は JSON オブジェクト".to_string())?;
    for k in obj.keys() {
        if !matches!(k.as_str(), "name" | "capture_mode") {
            return Err(format!(
                "未知のフィールド {k:?} (name / capture_mode のみ)"
            ));
        }
    }
    // task 829: `last_prompt_token` は prefill の最終トークンだけを記録する。
    let mode = match obj.get("capture_mode") {
        Some(m) => {
            let s = m
                .as_str()
                .ok_or_else(|| "capture_mode は文字列".to_string())?;
            steer::CaptureMode::parse(s).ok_or_else(|| {
                format!("未知の capture_mode {s:?} (all / last_prompt_token)")
            })?
        }
        None => steer::CaptureMode::All,
    };
    let name = obj
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "name は文字列".to_string())?;
    Ok((name.to_string(), mode))
}

// ---------------------------------------------------------------------------
// task 825: abliteration の作業台 (analyze / export / import)
// ---------------------------------------------------------------------------

/// 分析結果の JSON (ビューワの分離チャートがそのまま読む)。
fn analysis_to_json(a: &steer::Analysis) -> Value {
    json!({
        "pos": a.pos,
        "neg": a.neg,
        "n_layer": a.layers.len(),
        "peak_layer": a.peak_layer,
        "recommended_range": [a.recommended_range.0, a.recommended_range.1],
        "layers": a
            .layers
            .iter()
            .map(|l| json!({
                "layer": l.layer,
                "diff_norm": l.diff_norm,
                "norm_pos": l.norm_pos,
                "norm_neg": l.norm_neg,
                "cos_sim": l.cos,
                "sep_score": l.sep,
            }))
            .collect::<Vec<_>>(),
    })
}

/// `POST /v1/steer/analyze` の body を検証して分析する。**panic しない**。
///
/// `{pos, neg}` (名前 2 つ) か `{terms:[{snapshot,coef}...]}` (符号で A/B に分ける)。
fn analyze_steer(body: &Value) -> Result<Value, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "body は JSON オブジェクト".to_string())?;
    for k in obj.keys() {
        if !matches!(k.as_str(), "pos" | "neg" | "terms") {
            return Err(format!(
                "未知のフィールド {k:?} (pos / neg / terms のみ)"
            ));
        }
    }
    let an = if let Some(tv) = obj.get("terms") {
        let arr = tv
            .as_array()
            .ok_or_else(|| "terms は配列".to_string())?;
        if arr.is_empty() {
            return Err("terms が空".into());
        }
        let mut terms = Vec::with_capacity(arr.len());
        for (i, t) in arr.iter().enumerate() {
            let o = t
                .as_object()
                .ok_or_else(|| format!("terms[{i}] はオブジェクト"))?;
            for k in o.keys() {
                if !matches!(k.as_str(), "snapshot" | "coef") {
                    return Err(format!("terms[{i}]: 未知のフィールド {k:?}"));
                }
            }
            let name = o
                .get("snapshot")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("terms[{i}].snapshot は文字列"))?;
            let coef = as_f32(
                o.get("coef")
                    .ok_or_else(|| format!("terms[{i}].coef が無い"))?,
                &format!("terms[{i}].coef"),
            )?;
            terms.push(steer::Term {
                snapshot: name.to_string(),
                coef,
            });
        }
        steer::analyze_terms(&terms)?
    } else {
        let pos = obj
            .get("pos")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "pos は文字列 (か terms)".to_string())?;
        let neg = obj
            .get("neg")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "neg は文字列".to_string())?;
        steer::analyze(pos, neg)?
    };
    Ok(analysis_to_json(&an))
}

/// `?k=v&...` から 1 つ取る (パーセント復号はしない。名前は A-Z a-z 0-9 _ - のみ)。
fn query_param(path: &str, key: &str) -> Option<String> {
    let q = path.split_once('?')?.1;
    for kv in q.split('&') {
        if let Some((k, v)) = kv.split_once('=') {
            if k == key {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// `GET /v1/steer/export` の応答 (本文と Content-Type)。
fn export_body(path: &str) -> Result<(Vec<u8>, &'static str, String), String> {
    let name = query_param(path, "name").ok_or_else(|| "name が要る (?name=...)".to_string())?;
    if name.is_empty() {
        return Err("name が空".into());
    }
    let json_wanted = matches!(query_param(path, "format").as_deref(), Some("json"));
    if json_wanted {
        let v = steer::export_json(&name)?;
        let body = serde_json::to_vec_pretty(&v).map_err(|e| e.to_string())?;
        Ok((
            body,
            "application/json",
            format!("{name}.dnsteer.json"),
        ))
    } else {
        let body = steer::export_raw(&name)?;
        Ok((
            body,
            "application/octet-stream",
            format!("{name}.{}", steer::FILE_EXT),
        ))
    }
}

/// `POST /v1/steer/import` の本文 (DNSTEER1 そのまま / JSON) を取り込む。
fn import_steer(body: &[u8], rename: Option<&str>) -> Result<Value, String> {
    if body.is_empty() {
        return Err("body が空".into());
    }
    let snap = if body.starts_with(steer::FILE_MAGIC.as_bytes()) {
        steer::import_bytes(body, rename)?
    } else {
        let v: Value = serde_json::from_slice(body).map_err(|e| format!("JSON でない: {e}"))?;
        let renamed = match rename {
            Some(n) => {
                let mut v = v;
                if let Some(o) = v.as_object_mut() {
                    o.insert("name".into(), json!(n));
                }
                v
            }
            None => v,
        };
        steer::import_json(&renamed)?
    };
    let mut out = steer_status_to_json();
    out["imported"] = json!({
        "name": snap.name,
        "tokens": snap.tokens,
        "created": snap.created,
        "layers": snap.layers,
        "n_hc": snap.n_hc,
        "n_embd": snap.n_embd,
    });
    Ok(out)
}

// ---------------------------------------------------------------------------
// task 833: レシピ (持ち運びできるステアリング設定) の HTTP
// ---------------------------------------------------------------------------

/// `--steer-recipe` / `DENNO_STEER_RECIPE` のパス (フラグが優先)。
fn steer_recipe_path(cfg: &FlashConfig) -> Option<PathBuf> {
    if let Some(p) = &cfg.steer_recipe {
        return Some(p.clone());
    }
    let s = std::env::var("DENNO_STEER_RECIPE").ok()?;
    let s = s.trim();
    if s.is_empty() {
        None
    } else {
        Some(PathBuf::from(s))
    }
}

/// レシピを読んで検証し、適用する。返り値は起動ログの 1 行。
fn apply_steer_recipe(path: &Path) -> Result<String, String> {
    let r = steer::SteerRecipe::from_file(path)?;
    steer::SteerState::new().apply_recipe(&r)?;
    Ok(format!(
        "[flash] steer recipe loaded: {} mode={} alpha={} layers={}..{}",
        r.name,
        r.mode.name(),
        r.alpha,
        r.layer_from,
        r.layer_to
    ))
}

/// レシピの要約 (`/v1/steer/recipe/*` の応答に載せる)。
fn recipe_summary(r: &steer::SteerRecipe) -> Value {
    json!({
        "name": r.name,
        "description": r.description,
        "model_id": r.model_id,
        "mode": r.mode.name(),
        "alpha": r.alpha,
        "layers": [r.layer_from, r.layer_to],
        "terms": r
            .terms
            .iter()
            .map(|t| json!({"snapshot": t.snapshot, "coef": t.coef}))
            .collect::<Vec<_>>(),
        "embedded": r.snapshots.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
    })
}

/// `POST /v1/steer/recipe/export`。適用中の設定か、body で指定した設定を
/// レシピ JSON にして返す。**panic しない**。
fn recipe_export(body: &Value) -> Result<Value, String> {
    let obj = body
        .as_object()
        .ok_or_else(|| "body は JSON オブジェクト".to_string())?;
    for k in obj.keys() {
        if !matches!(
            k.as_str(),
            "name" | "description" | "embed_snapshots" | "terms" | "layers" | "alpha" | "mode"
        ) {
            return Err(format!(
                "未知のフィールド {k:?} (name / description / embed_snapshots / terms / layers / alpha / mode のみ)"
            ));
        }
    }
    let name = match obj.get("name") {
        Some(v) => v
            .as_str()
            .ok_or_else(|| "name は文字列".to_string())?
            .to_string(),
        None => "recipe".to_string(),
    };
    let embed = match obj.get("embed_snapshots") {
        Some(v) => v
            .as_bool()
            .ok_or_else(|| "embed_snapshots は真偽値".to_string())?,
        None => false,
    };
    let model_id = match steer::meta() {
        Some((_, m)) if !m.is_empty() => Some(m),
        _ => None,
    };
    let mut r = if obj.contains_key("terms") {
        // body で指定された設定 (まだ適用していなくてもよい)。
        let tv = obj["terms"]
            .as_array()
            .ok_or_else(|| "terms は配列".to_string())?;
        if tv.is_empty() {
            return Err("terms が空".into());
        }
        let mut terms = Vec::with_capacity(tv.len());
        for (i, t) in tv.iter().enumerate() {
            let o = t
                .as_object()
                .ok_or_else(|| format!("terms[{i}] はオブジェクト"))?;
            for k in o.keys() {
                if !matches!(k.as_str(), "snapshot" | "coef") {
                    return Err(format!("terms[{i}]: 未知のフィールド {k:?}"));
                }
            }
            let snap = o
                .get("snapshot")
                .and_then(|v| v.as_str())
                .ok_or_else(|| format!("terms[{i}].snapshot は文字列"))?;
            let coef = as_f32(
                o.get("coef")
                    .ok_or_else(|| format!("terms[{i}].coef が無い"))?,
                &format!("terms[{i}].coef"),
            )?;
            terms.push(steer::Term {
                snapshot: snap.to_string(),
                coef,
            });
        }
        let (from, to) = match obj.get("layers") {
            Some(v) => {
                let lv = v
                    .as_array()
                    .filter(|a| a.len() == 2)
                    .ok_or_else(|| "layers は [from, to]".to_string())?;
                (as_index(&lv[0], "layers[0]")?, as_index(&lv[1], "layers[1]")?)
            }
            None => {
                let n = steer::meta().map(|(s, _)| s.n_layer).unwrap_or(0);
                (0, n)
            }
        };
        let alpha = match obj.get("alpha") {
            Some(v) => as_f32(v, "alpha")?,
            None => 1.0,
        };
        let mode = match obj.get("mode") {
            Some(v) => {
                let s = v.as_str().ok_or_else(|| "mode は文字列".to_string())?;
                steer::Mode::parse(s)
                    .ok_or_else(|| format!("未知の mode {s:?} (subtract / project)"))?
            }
            None => steer::Mode::Project,
        };
        steer::SteerRecipe {
            name: name.clone(),
            description: None,
            model_id: model_id.clone(),
            mode,
            alpha,
            layer_from: from,
            layer_to: to,
            terms,
            snapshots: Vec::new(),
        }
    } else {
        let mut r = steer::current_recipe(&name, false)?;
        r.model_id = model_id;
        r
    };
    r.description = match obj.get("description") {
        Some(v) => Some(
            v.as_str()
                .ok_or_else(|| "description は文字列".to_string())?
                .to_string(),
        ),
        None => None,
    };
    if embed {
        r.embed_snapshots()?;
    }
    Ok(r.to_json())
}

/// `POST /v1/steer/recipe/import`。レシピ JSON を検証し、`apply` が真なら
/// 適用する (埋め込みはメモリにだけ載せる)。**panic しない**。
fn recipe_import(body: &Value, apply_query: bool) -> Result<Value, String> {
    let (rv, want_apply) = match body.as_object() {
        Some(o) if o.contains_key("recipe") => {
            for k in o.keys() {
                if !matches!(k.as_str(), "recipe" | "apply") {
                    return Err(format!(
                        "未知のフィールド {k:?} (recipe / apply のみ。レシピそのものを送るなら recipe を包まない)"
                    ));
                }
            }
            let a = match o.get("apply") {
                Some(v) => v
                    .as_bool()
                    .ok_or_else(|| "apply は真偽値".to_string())?,
                None => false,
            };
            (
                o.get("recipe")
                    .ok_or_else(|| "recipe が無い".to_string())?,
                a,
            )
        }
        _ => (body, false),
    };
    let r = steer::SteerRecipe::from_json(rv)?;
    let do_apply = apply_query || want_apply;
    let mut out = json!({
        "valid": true,
        "applied": false,
        "recipe": recipe_summary(&r),
    });
    if do_apply {
        steer::SteerState::new().apply_recipe(&r)?;
        out["applied"] = json!(true);
        out["steer"] = steer_status_to_json();
    }
    Ok(out)
}

/// 生のバイト列を返す (ダウンロード用。Content-Disposition 付き)。
fn send_bytes(
    stream: &mut TcpStream,
    status: &str,
    ctype: &str,
    filename: &str,
    body: &[u8],
) -> Result<()> {
    head(
        stream,
        status,
        ctype,
        &format!(
            "Content-Length: {}\r\nContent-Disposition: attachment; filename=\"{}\"\r\n",
            body.len(),
            filename
        ),
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

/// task 218: `DENNO_STEER_RECORD` の記録を 1 リクエストごとにディスクへ書く。
///
/// 生成の途中で return してもドロップで走る (記録は止めない)。
struct SteerFlushGuard;

impl Drop for SteerFlushGuard {
    fn drop(&mut self) {
        if steer::env_recording() {
            if let Err(e) = steer::flush() {
                eprintln!("[steer] 記録の保存に失敗: {e}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 生成の強制停止 (`POST /v1/abort`)
// ---------------------------------------------------------------------------

/// 停止要求。decode のトークンごとに見る (立っていたら次のトークンを作らずに終わる)。
static ABORT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// いま生成中か (停止要求は生成中だけ受ける。空振りの要求を次の生成に持ち越さない)。
static GENERATING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 生成の間だけ GENERATING を立てる。始めに古い停止要求を捨てる。
struct GenGuard;

impl GenGuard {
    fn enter() -> Self {
        use std::sync::atomic::Ordering::SeqCst;
        ABORT.store(false, SeqCst);
        GENERATING.store(true, SeqCst);
        GenGuard
    }
}

impl Drop for GenGuard {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::SeqCst;
        GENERATING.store(false, SeqCst);
        ABORT.store(false, SeqCst);
    }
}

/// テスト: ABORT / GENERATING はプロセス全体の状態なので、触るテストはこれで直列にする。
#[cfg(test)]
static ABORT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// 停止要求を出す。生成中でなければ何もしない。戻り値は「生成中に要求した」か。
fn request_abort() -> bool {
    use std::sync::atomic::Ordering::SeqCst;
    if GENERATING.load(SeqCst) {
        ABORT.store(true, SeqCst);
        true
    } else {
        false
    }
}

/// 停止要求が立っていれば下ろして true。
fn take_abort() -> bool {
    ABORT.swap(false, std::sync::atomic::Ordering::SeqCst)
}

/// `POST /v1/abort` か (生成中も受け付ける経路。主スレッドを待たない)。
fn is_abort_req(req: &HttpReq) -> bool {
    req.method == "POST" && req.path.split('?').next() == Some("/v1/abort")
}

fn send_abort_reply(stream: &mut TcpStream) -> Result<()> {
    let generating = request_abort();
    if generating {
        eprintln!("[flash] abort: 停止要求を受けた (次のトークンで止める)");
    }
    send_json(stream, "200 OK", &json!({"aborted": generating, "generating": generating}))
}

// ---------------------------------------------------------------------------
// モデルの選択 (`GET /v1/model/choices` / `POST /v1/model/select`)
// ---------------------------------------------------------------------------
//
// GPU と RAM に載るモデルは 1 つだけなので、選び直しは「選択を書いて終了 →
// 外側の起動ループ (serve_loop.sh) が選ばれたモデルで起動し直す」で行う。
// 一覧は `DENNO_MODEL_CHOICES` のファイル (1 行 1 モデル `id|dir|label|env`、
// `#` はコメント、先頭の行が既定)、選択は `DENNO_MODEL_SELECT` のファイルに id を書く。
// どちらかが未設定なら切り替えは無効 (今まで通り)。

/// 切り替えのための終了コード (serve_loop.sh はこれを見て起動し直す)。
pub const EXIT_MODEL_SWITCH: i32 = 75;

/// 主ループがリクエストの後に見る終了要求 (0 = なし)。
static EXIT_AFTER: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// P817: 停止要求 (SIGTERM / SIGINT) を受けたか。立ったら新規の生成要求は
/// 503 で受けず、進行中の 1 本が終わるのを短い上限まで待ってから終了する。
static DRAINING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// P817: いま生成中の要求数 (主ループは 1 本ずつ処理するので 0 / 1)。
static IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// P817: 停止時に進行中の生成を待つ上限 (ms)。`DENNO_DRAIN_TIMEOUT_MS`
/// (既定 10 s、0〜600000 にクランプ) で上書きできる。0 は「待たない」。
fn drain_timeout() -> std::time::Duration {
    let ms = std::env::var("DENNO_DRAIN_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(10_000)
        .min(600_000);
    std::time::Duration::from_millis(ms)
}

fn is_draining() -> bool {
    DRAINING.load(std::sync::atomic::Ordering::SeqCst)
}

#[derive(Clone, Debug, PartialEq)]
struct ModelChoice {
    id: String,
    dir: String,
    label: String,
    /// 起動ループが付ける env (`K=V` を空白区切り)。表示だけに使う。
    env: String,
}

fn parse_choices(text: &str) -> Vec<ModelChoice> {
    let mut out: Vec<ModelChoice> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('|').map(|s| s.trim()).collect();
        if f.len() < 2 || f[0].is_empty() || f[1].is_empty() {
            continue;
        }
        if out.iter().any(|c| c.id == f[0]) {
            continue;
        }
        out.push(ModelChoice {
            id: f[0].to_string(),
            dir: f[1].to_string(),
            label: f.get(2).map(|s| s.to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| f[0].to_string()),
            env: f.get(3).map(|s| s.to_string()).unwrap_or_default(),
        });
    }
    out
}

fn choices_paths() -> Option<(PathBuf, PathBuf)> {
    let c = std::env::var("DENNO_MODEL_CHOICES").ok().filter(|s| !s.is_empty())?;
    let s = std::env::var("DENNO_MODEL_SELECT").ok().filter(|s| !s.is_empty())?;
    Some((PathBuf::from(c), PathBuf::from(s)))
}

fn same_dir(a: &Path, b: &Path) -> bool {
    let ca = std::fs::canonicalize(a).unwrap_or_else(|_| a.to_path_buf());
    let cb = std::fs::canonicalize(b).unwrap_or_else(|_| b.to_path_buf());
    ca == cb
}

/// いま動いているモデルのディレクトリ (最初の part の親)。
fn current_model_dir(flash: &Flash) -> Option<PathBuf> {
    flash.parts.first().and_then(|p| p.parent()).map(|p| p.to_path_buf())
}

fn dir_gguf_bytes(dir: &Path) -> Option<u64> {
    let rd = std::fs::read_dir(dir).ok()?;
    let mut n = 0u64;
    let mut any = false;
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().map(|x| x == "gguf").unwrap_or(false) {
            n += e.metadata().map(|m| m.len()).unwrap_or(0);
            any = true;
        }
    }
    any.then_some(n)
}

fn model_choices_json(flash: &Flash) -> Value {
    let cur_dir = current_model_dir(flash);
    let Some((cpath, _)) = choices_paths() else {
        return json!({
            "switchable": false,
            "current": Value::Null,
            "current_dir": cur_dir.map(|d| d.display().to_string()),
            "choices": [],
        });
    };
    let list = std::fs::read_to_string(&cpath).map(|t| parse_choices(&t)).unwrap_or_default();
    let mut current: Option<String> = None;
    let rows: Vec<Value> = list
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let here = cur_dir.as_deref().is_some_and(|d| same_dir(d, Path::new(&c.dir)));
            if here && current.is_none() {
                current = Some(c.id.clone());
            }
            let bytes = dir_gguf_bytes(Path::new(&c.dir));
            json!({
                "id": c.id, "label": c.label, "dir": c.dir, "env": c.env,
                "default": i == 0,
                "available": bytes.is_some(),
                "size_gb": bytes.map(|b| (b as f64 / 1e9 * 10.0).round() / 10.0),
            })
        })
        .collect();
    json!({
        "switchable": true,
        "current": current,
        "current_dir": cur_dir.map(|d| d.display().to_string()),
        "choices": rows,
    })
}

/// 選択を検証して書く。切り替えるなら Ok(true) (呼び出し側が応答後に終了を予約)。
fn select_model(flash: &Flash, body: &Value) -> Result<(bool, String), String> {
    let (cpath, spath) = choices_paths().ok_or("このサーバーはモデルの切り替えに対応していない (DENNO_MODEL_CHOICES / DENNO_MODEL_SELECT が無い)")?;
    let id = body
        .get("id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or("id (文字列) が要る")?
        .to_string();
    let text = std::fs::read_to_string(&cpath)
        .map_err(|e| format!("{} を読めない: {e}", cpath.display()))?;
    let list = parse_choices(&text);
    let c = list
        .iter()
        .find(|c| c.id == id)
        .ok_or_else(|| format!("未知のモデル {id:?}"))?;
    if dir_gguf_bytes(Path::new(&c.dir)).is_none() {
        return Err(format!("{} に .gguf が無い", c.dir));
    }
    if current_model_dir(flash).is_some_and(|d| same_dir(&d, Path::new(&c.dir))) {
        return Ok((false, id));
    }
    std::fs::write(&spath, format!("{id}\n"))
        .map_err(|e| format!("{} に書けない: {e}", spath.display()))?;
    Ok((true, id))
}

/// task 271: 直前のリクエスト (decode) の tier 別の当たり。
///
/// `exp_iq` の参照/ヒットと `expert_ram` の VRAM ミス転送元はリクエスト終端で
/// 0 に戻るので、戻す直前にここへ写す。`/v1/experts/stats` はこの写しを返す
/// (積算ではなく「直前の 1 リクエスト」)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LastTierHits {
    /// decode で参照した expert の本数
    lookups: u64,
    /// そのうち VRAM に載っていた本数
    vram: u64,
    /// VRAM ミスのうち RAM tier から上げた本数
    ram: u64,
    /// VRAM ミスのうち mmap (ファイル) から読んだ本数
    mmap: u64,
}

static LAST_TIER: std::sync::Mutex<Option<LastTierHits>> = std::sync::Mutex::new(None);

/// リクエスト終端のカウンタを [`LAST_TIER`] へ写す (0 に戻す前に呼ぶ)。
///
/// 参照 0 件 (prefill だけで終わったリクエスト) では書き換えない。
/// ミス本数と転送元の積算は別の計器なので、`lookups` を超えないように丸める。
fn record_last_tier_hits() {
    let (lookups, hits) = denno_engine::model::exp_iq::lookups_hits();
    if lookups == 0 {
        return;
    }
    let (miss_ram, miss_mmap) = denno_engine::model::expert_ram::miss_stats();
    let vram = hits.min(lookups);
    let miss = lookups - vram;
    let ram = miss_ram.min(miss);
    let mmap = miss_mmap.min(miss - ram);
    *LAST_TIER.lock().unwrap_or_else(|e| e.into_inner()) = Some(LastTierHits {
        lookups,
        vram,
        ram,
        mmap,
    });
}

/// task 263: `GET /v1/experts/stats` の形。
///
/// tier ごとの本数と MB、記録された使用回数のうち各 tier が覆う割合、
/// 記録ファイルと最後に保存できた時刻。ビューワが後で描くためのもの。
/// task 271: mmap tier と、直前のリクエストの tier 別ヒット率を足した。
fn experts_stats_json() -> Value {
    use denno_engine::model::{exp_stats, expert_ram, expert_vram};
    let counts = exp_stats::current_counts();
    let vkeys: Vec<(u16, u16)> = expert_vram::keys()
        .iter()
        .map(|k| (k.layer, k.expert))
        .collect();
    let rkeys: Vec<(u16, u16)> = expert_ram::keys()
        .iter()
        .map(|k| (k.layer, k.expert))
        .collect();
    let (cov_v, cov_r, total) = match &counts {
        Some(c) => {
            let (a, t) = exp_stats::coverage(c, &vkeys);
            let (b, _) = exp_stats::coverage(c, &rkeys);
            (a, b, t)
        }
        None => (0, 0, 0),
    };
    let frac = |a: u64| {
        if total > 0 {
            a.min(total) as f64 / total as f64
        } else {
            0.0
        }
    };
    let vram = expert_vram::cache_usage();
    let ram = expert_ram::status();
    let plan = exp_stats::plan();
    // task 271: mmap tier = 記録された expert のうち VRAM にも RAM にも載って
    // いない本数。1 本のバイト数は expert_bytes / (n_layer * n_expert)。
    let (mmap_experts, mmap_bytes) = match &counts {
        Some(c) => {
            let cells = (c.n_layer as usize).saturating_mul(c.n_expert as usize);
            let per = if cells > 0 {
                c.expert_bytes / cells as u64
            } else {
                0
            };
            let resident: std::collections::HashSet<(u16, u16)> =
                vkeys.iter().chain(rkeys.iter()).copied().collect();
            let mut n = 0usize;
            for layer in 0..c.n_layer as usize {
                for e in 0..c.n_expert as usize {
                    let i = layer * c.n_expert as usize + e;
                    if c.cells.get(i).copied().unwrap_or(0) > 0
                        && !resident.contains(&(layer as u16, e as u16))
                    {
                        n += 1;
                    }
                }
            }
            (n, n as u64 * per)
        }
        None => (0, 0),
    };
    let last = *LAST_TIER.lock().unwrap_or_else(|e| e.into_inner());
    let last_json = last.map(|l| {
        let rate = |n: u64| {
            if l.lookups > 0 {
                n.min(l.lookups) as f64 / l.lookups as f64
            } else {
                0.0
            }
        };
        json!({
            "lookups": l.lookups,
            "vram": {"hits": l.vram, "hit_rate": rate(l.vram)},
            "ram": {"hits": l.ram, "hit_rate": rate(l.ram)},
            "mmap": {"hits": l.mmap, "hit_rate": rate(l.mmap)},
        })
    });
    json!({
        "enabled": exp_stats::enabled(),
        "file": exp_stats::stats_path().map(|p| p.display().to_string()),
        "last_save": exp_stats::last_save().map(|(t, b)| json!({"unix": t, "bytes": b})),
        "recorded": counts.as_ref().map(|c| json!({
            "dir": c.dir,
            "expert_bytes": c.expert_bytes,
            "n_layer": c.n_layer,
            "n_expert": c.n_expert,
            "experts": c.nonzero(),
            "uses": total,
        })),
        "vram": {
            "experts": vkeys.len(),
            "mb": vram.map(|(u, _)| u >> 20),
            "cap_mb": vram.map(|(_, c)| c >> 20),
        },
        "ram": {
            "experts": ram.map(|r| r.0),
            "slots": ram.map(|r| r.1),
            "mb": ram.map(|r| (r.3 >> 20) as usize),
            "pinned_mb": ram.map(|r| (r.2 >> 20) as usize),
        },
        "mmap": {
            "experts": mmap_experts,
            "mb": mmap_bytes >> 20,
        },
        "last_request": last_json,
        "coverage": {
            "vram": frac(cov_v),
            "ram": frac(cov_r),
            "total": frac(cov_v.saturating_add(cov_r)),
        },
        "plan": plan.map(|p| json!({
            "at": p.at,
            "planned_experts": p.planned_experts,
            "vram_experts": p.vram_experts,
            "vram_mb": p.vram_bytes >> 20,
            "ram_experts": p.ram_experts,
            "ram_mb": p.ram_bytes >> 20,
            "cover_vram": p.cover_vram,
            "cover_ram": p.cover_ram,
            "cover_total": p.cover_total,
            "secs": p.secs,
            "promoted": p.promoted,
            "dropped": p.dropped,
        })),
    })
}

/// 1 接続ぶんを読んでから処理する (テストと単発用)。本番の listen は
/// [`run`] が読み取りを別スレッドで行い、`/v1/abort` だけはそこで即答する。
#[cfg(test)]
fn handle(flash: &mut Flash, stream: TcpStream) -> Result<()> {
    stream.set_nodelay(true).ok();
    let mut r = BufReader::new(stream.try_clone()?);
    let Some(req) = read_req(&mut r)? else {
        return Ok(());
    };
    handle_req(flash, stream, r, req)
}

fn handle_req(
    flash: &mut Flash,
    mut stream: TcpStream,
    r: BufReader<TcpStream>,
    req: HttpReq,
) -> Result<()> {
    let path = req.path.split('?').next().unwrap_or("/").to_string();
    // 生成中でない間に来た停止要求 (読み取りスレッドを通らないテスト経路など)。
    if is_abort_req(&req) {
        return send_abort_reply(&mut stream);
    }
    // P817: drain 中は新規の要求 (POST) を受けない。GET (プローブ / メトリクス /
    // 状態の読み出し) は止めない — オーケストレータが /ready を読めなくなる。
    if is_draining() && req.method != "GET" {
        return send_json(
            &mut stream,
            "503 Service Unavailable",
            &json!({"type": "error", "error": {
                "type": "overloaded_error",
                "message": "server is draining (shutdown signal received); the request was not accepted"
            }}),
        );
    }
    // P817: readiness。モデルのロードは listen 前に終わっているので、
    // ここでの判定は「drain していない」。
    if req.method == "GET" && path == "/ready" {
        let (status, body) = ready_response(&flash.name, flash.max_ctx);
        return send_json(&mut stream, status, &body);
    }

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
        if let Some(v) = get_static(&path, &flash.name, flash.max_ctx) {
            return send_json(&mut stream, "200 OK", &v);
        }
    }
    // task 347: Prometheus `/metrics`。ゲート off (`DENNO_METRICS=0`) のときは
    // 経路ごと無い (下の 404 に落ちる)。
    if req.method == "GET" && path == "/metrics" && metrics::enabled() {
        return send_metrics(&mut stream, &metrics::render());
    }
    // GET /v1/limits — 生成上限の問い合わせ。ビューワが値をハードコードせずに
    // ここから読む (取得できないクライアント側の既定は 4096)。
    if req.method == "GET" && path == "/v1/limits" {
        return send_json(
            &mut stream,
            "200 OK",
            &json!({
                "max_tokens_default": flash.default_max,
                "max_tokens_limit": flash.default_max,
                "max_context_tokens": flash.max_ctx
            }),
        );
    }
    // task 217: ビューワ用の形と介入の get / set / clear。
    if req.method == "GET" && path == "/v1/model/shape" {
        return send_json(&mut stream, "200 OK", &flash.shape.json());
    }
    if req.method == "GET" && path == "/v1/intervene" {
        return send_json(&mut stream, "200 OK", &intervene_to_json(&iv::snapshot()));
    }
    if req.method == "POST" && path == "/v1/intervene" {
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
        return match apply_intervene(&flash.shape, &body) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(msg) => send_json(
                &mut stream,
                "400 Bad Request",
                &json!({"type": "error", "error": {"type": "invalid_request_error", "message": msg}}),
            ),
        };
    }
    if req.method == "DELETE" && path == "/v1/intervene" {
        let cfg = iv::clear(&flash.shape.engine_shape());
        return send_json(&mut stream, "200 OK", &intervene_to_json(&cfg));
    }
    // ビューワ用: モデルの一覧と選択 (選び直しはプロセスの再起動で行う)。
    if req.method == "GET" && path == "/v1/model/choices" {
        return send_json(&mut stream, "200 OK", &model_choices_json(flash));
    }
    if req.method == "POST" && path == "/v1/model/select" {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return send_json(&mut stream, "400 Bad Request", &err_json(&e.to_string())),
        };
        return match select_model(flash, &body) {
            Ok((true, id)) => {
                eprintln!("[flash] model select: {id} を選んだ。応答のあと終了して起動し直す (exit {EXIT_MODEL_SWITCH})");
                // task 271: プロセスが終わる前に、いまのモデルの使用回数を書く
                // (切り替えで直前の数分の回数を失わない)。
                save_stats_on_stop("model select");
                EXIT_AFTER.store(EXIT_MODEL_SWITCH, std::sync::atomic::Ordering::SeqCst);
                send_json(&mut stream, "202 Accepted", &json!({"switching": true, "id": id}))
            }
            Ok((false, id)) => send_json(&mut stream, "200 OK", &json!({"switching": false, "id": id})),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    // task 926: prefix cache の利用実績 (ヒット / ミス / 再利用トークン) の要約。
    // ビューワ向けの詳細 (task 217 の /v1/cache) とは別に、監視用の平たい JSON を返す。
    if req.method == "GET" && path == "/v1/cache/stats" {
        return send_json(&mut stream, "200 OK", &flash.cache_stats());
    }
    // ビューワ用: 保持している文脈 (KV / GDN 状態) の確認と破棄。
    if req.method == "GET" && path == "/v1/cache" {
        return send_json(&mut stream, "200 OK", &flash.ctx_json());
    }
    // task 263: expert の常駐状態 (tier ごとの本数・MB・覆盖率・記録ファイル)。
    if req.method == "GET" && path == "/v1/experts/stats" {
        return send_json(&mut stream, "200 OK", &experts_stats_json());
    }
    if req.method == "DELETE" && path == "/v1/cache" {
        let v = flash.clear_ctx();
        return send_json(&mut stream, "200 OK", &v);
    }
    // task 218: 残差の平均の記録 / 適用 (ビューワが使う)。
    if req.method == "GET" && path == "/v1/steer" {
        return send_json(&mut stream, "200 OK", &steer_status_to_json());
    }
    if req.method == "POST" && path == "/v1/steer/record" {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return send_json(&mut stream, "400 Bad Request", &err_json(&e.to_string())),
        };
        return match record_params(&body)
            .and_then(|(n, m)| steer::start_record_mode(&n, m))
        {
            Ok(_) => send_json(&mut stream, "200 OK", &steer_status_to_json()),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "POST" && path == "/v1/steer/record/stop" {
        return match steer::stop_record() {
            Ok(s) => {
                let mut v = steer_status_to_json();
                v["saved"] = json!({
                    "name": s.name, "tokens": s.tokens, "created": s.created, "layers": s.layers
                });
                send_json(&mut stream, "200 OK", &v)
            }
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "DELETE" && path.starts_with("/v1/steer/snapshots/") {
        let name = &path["/v1/steer/snapshots/".len()..];
        return match steer::delete(name) {
            Ok(()) => send_json(&mut stream, "200 OK", &steer_status_to_json()),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "POST" && path == "/v1/steer/apply" {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return send_json(&mut stream, "400 Bad Request", &err_json(&e.to_string())),
        };
        return match apply_steer(&body) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "DELETE" && path == "/v1/steer/apply" {
        steer::clear_apply();
        return send_json(&mut stream, "200 OK", &steer_status_to_json());
    }
    // task 825: Abliterate 作業台 (対比の分析 / 持ち出し / 取り込み)。
    if req.method == "POST" && path == "/v1/steer/analyze" {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return send_json(&mut stream, "400 Bad Request", &err_json(&e.to_string())),
        };
        return match analyze_steer(&body) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "GET" && path == "/v1/steer/export" {
        return match export_body(&req.path) {
            Ok((body, ctype, filename)) => send_bytes(&mut stream, "200 OK", ctype, &filename, &body),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "POST" && path == "/v1/steer/import" {
        let rename = query_param(&req.path, "name");
        return match import_steer(&req.body, rename.as_deref()) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    // task 833: レシピ (1 ファイルのステアリング設定) の持ち出し / 取り込み。
    // import は既定で検証だけ。`?apply=1` か body の `apply: true` で適用する。
    if req.method == "POST" && path == "/v1/steer/recipe/export" {
        let body: Value = if req.body.is_empty() || req.body == b"{}" {
            json!({})
        } else {
            match serde_json::from_slice(&req.body) {
                Ok(v) => v,
                Err(e) => return send_json(&mut stream, "400 Bad Request", &err_json(&e.to_string())),
            }
        };
        return match recipe_export(&body) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
    }
    if req.method == "POST" && path == "/v1/steer/recipe/import" {
        let body: Value = match serde_json::from_slice(&req.body) {
            Ok(v) => v,
            Err(e) => return send_json(&mut stream, "400 Bad Request", &err_json(&e.to_string())),
        };
        let apply = matches!(
            query_param(&req.path, "apply").as_deref(),
            Some("1") | Some("true")
        );
        return match recipe_import(&body, apply) {
            Ok(v) => send_json(&mut stream, "200 OK", &v),
            Err(msg) => send_json(&mut stream, "400 Bad Request", &err_json(&msg)),
        };
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

/// task 119b: `RAYON_NUM_THREADS` 未設定のときの既定 rayon スレッド数 (純粋関数)。
///
/// `max(1, 論理 CPU - 2)`。rayon ワーカ + メインスレッドが論理 CPU 全数を埋めると
/// オーバーサブスクライブになる (メインセッション実測, swift15-abl IQ3_S 128 tok:
/// decode 16T 118 / 14T 114 / 12T 113 / 10T 112.5 ms/tok、prefill 28 tok は
/// 16T 2.5 s / 14T 2.55 s / 12T 2.77 s。14 が折れ点)。`RAYON_NUM_THREADS` が
/// 設定されているときはこの関数は使われない (そちらが勝つ)。
fn default_rayon_threads(logical_cpus: usize) -> usize {
    logical_cpus.saturating_sub(2).max(1)
}

/// task 119b: rayon の大域プールを既定スレッド数で立てる (初回使用の前)。
///
/// `RAYON_NUM_THREADS` が設定済み (空でない) なら何もしない (rayon が env を読む)。
/// 未設定なら [`default_rayon_threads`] で `build_global` する。既にプールが立って
/// いる (二重呼び出し) 場合のエラーは無視する。[`run`] の最初で呼ぶ。
fn ensure_rayon_pool() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some_and(|v| !v.is_empty()) {
        return;
    }
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let n = default_rayon_threads(cpus);
    if rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .is_ok()
    {
        eprintln!("[flash] rayon pool: {n} threads (論理 CPU {cpus} - 2)");
    }
}

/// task 146: `run` が正常に戻るときに使用回数を書き残す (Drop)。
///
/// SIGKILL・電源断では走らないので、通常はリクエスト終端と 20 秒ごとの保存が
/// 正規の経路 (こちらは最後の 1 回を確実にするための保険)。
struct StatsSaveOnExit;

impl Drop for StatsSaveOnExit {
    fn drop(&mut self) {
        if let Some(line) = denno_engine::model::exp_stats::save_now() {
            eprintln!("{line}");
        }
    }
}

/// task 271: 終了の直前に expert 使用回数を書く (SIGTERM / モデル切り替え)。
///
/// 20 秒ごとの保存とは別に、終了経路でもう一度書く。`save_now` は新しい
/// 回数が無ければ何もしないので、二重に呼んでも安全 (一時ファイル → rename の
/// 原子的な 1 手順は `save_now` の中)。
fn save_stats_on_stop(why: &str) {
    if let Some(line) = denno_engine::model::exp_stats::save_now() {
        eprintln!("[flash] {why}: {line}");
    }
}

/// task 271 / task 817: SIGTERM・SIGINT で drain してから使用回数を保存して終了する。
///
/// 既定の SIGTERM はその場でプロセスを殺すので、20 秒ごとの保存の谷間の
/// 回数が消える。task 817 では「進行中の 1 本を打ち切らない」ように、
/// 1. `DRAINING` を立てる (以後の POST は 503)
/// 2. `IN_FLIGHT` が 0 になるのを `drain_timeout()` まで待つ
/// 3. expert 使用回数を保存
/// の順に行う。tokio の signal ハンドラ本体は async-signal-safe な書き込みだけで、
/// 待ちと保存は通常の文脈 (このスレッド) で行う。
#[cfg(unix)]
fn spawn_sigterm_saver() {
    let _ = std::thread::Builder::new()
        .name("sigterm-save".to_string())
        .spawn(|| {
            let Ok(rt) = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            else {
                return;
            };
            rt.block_on(async {
                use tokio::signal::unix::{signal, SignalKind};
                let Ok(mut term) = signal(SignalKind::terminate()) else {
                    return;
                };
                let Ok(mut int) = signal(SignalKind::interrupt()) else {
                    return;
                };
                let sig = tokio::select! {
                    _ = term.recv() => "SIGTERM",
                    _ = int.recv() => "SIGINT",
                };
                DRAINING.store(true, std::sync::atomic::Ordering::SeqCst);
                let timeout = drain_timeout();
                eprintln!(
                    "[flash] {sig}: drain 開始 (新規 POST は 503、進行中を最大 {:.1}s 待つ)",
                    timeout.as_secs_f64()
                );
                let start = Instant::now();
                loop {
                    let n = IN_FLIGHT.load(std::sync::atomic::Ordering::SeqCst);
                    if n == 0 {
                        break;
                    }
                    if start.elapsed() >= timeout {
                        eprintln!("[flash] drain 上限超過: 進行中 {n} 本のまま終了する");
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                }
                save_stats_on_stop(sig);
                std::process::exit(0);
            });
        });
}

/// 非 unix では SIGTERM が無いので何もしない。
#[cfg(not(unix))]
fn spawn_sigterm_saver() {}

/// 要求の読み取りは接続ごとのスレッドで行い、読めた要求を主スレッドへ渡す
/// (モデルを触る処理は今まで通り主スレッドで 1 本ずつ)。`POST /v1/abort` だけは
/// 読み取りスレッドで即答するので、生成中 (主スレッドが塞がっている間) でも止められる。
type Job = (TcpStream, BufReader<TcpStream>, HttpReq);

fn spawn_acceptor(listener: TcpListener) -> std::sync::mpsc::Receiver<std::io::Result<Job>> {
    let (tx, rx) = std::sync::mpsc::channel::<std::io::Result<Job>>();
    std::thread::spawn(move || {
        for s in listener.incoming() {
            let tx = tx.clone();
            let stream = match s {
                Ok(s) => s,
                Err(e) => {
                    if tx.send(Err(e)).is_err() {
                        return;
                    }
                    continue;
                }
            };
            std::thread::spawn(move || {
                stream.set_nodelay(true).ok();
                let Ok(clone) = stream.try_clone() else { return };
                let mut r = BufReader::new(clone);
                let req = match read_req(&mut r) {
                    Ok(Some(req)) => req,
                    Ok(None) => return,
                    Err(e) => {
                        eprintln!("[flash] 要求を読めない: {e:#}");
                        return;
                    }
                };
                if is_abort_req(&req) {
                    let mut stream = stream;
                    if let Err(e) = send_abort_reply(&mut stream) {
                        eprintln!("[flash] abort の応答に失敗: {e:#}");
                    }
                    return;
                }
                // task 347: `/metrics` は**読み取りスレッドで即答**する。生成中でも
                // スクレイプが待たされない (計器は atomic だけなのでロック不要)。
                if req.method == "GET"
                    && req.path.split('?').next() == Some("/metrics")
                    && metrics::enabled()
                {
                    let mut stream = stream;
                    if let Err(e) = send_metrics(&mut stream, &metrics::render()) {
                        eprintln!("[flash] metrics の応答に失敗: {e:#}");
                    }
                    return;
                }
                let _ = tx.send(Ok((stream, r, req)));
            });
        }
    });
    rx
}

/// P818: `--vram-budget auto` / `DENNO_AUTO_CONFIG=1` の本体。
///
/// カードの総 VRAM とホスト RAM、モデルのファイル長、expert の量子型から
/// `denno_engine::auto_config::plan_gpu_config` で計画を組み、**未設定の
/// `DENNO_*` だけ**を埋める。明示 env は 1 つも触らない。
///
/// モデルを組む前に呼ぶ (expert の常駐予算は構築時に 1 度だけ読まれる)。
fn apply_auto_config(
    budget: Option<&str>,
    parts: &[PathBuf],
    f: &GgufFile,
) -> Result<String> {
    use denno_engine::auto_config as ac;
    let req = budget
        .and_then(ac::VramBudget::parse)
        .unwrap_or(ac::VramBudget::Auto);
    let vram = match req.vram_mb() {
        Some(n) => n,
        None => ac::probe_vram_mb().ok_or_else(|| {
            anyhow::anyhow!(
                "VRAM を検出できない (cuda feature 無し / libcuda 無し)。\
                 `--vram-budget <MiB>` で総 VRAM を明示できる"
            )
        })?,
    };
    let ram = ac::host_ram_mb();
    let model_bytes = ac::sum_model_bytes(
        parts
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len()),
    );
    let types = f
        .tensor_order
        .iter()
        .filter(|n| denno_engine::gpu_budget::is_expert_weight(n))
        .filter_map(|n| f.tensors.get(n).map(|t| t.ty));
    // expert が無いモデルでも dense 側の予算は要るので Q4 とみなす
    // (dense の推定割合が小さくなる側 = expert 枠を多めに取る)。
    let quant = ac::quant_from_expert_types(types).unwrap_or(ac::ModelQuant::Q4);
    let plan = ac::plan_gpu_config(vram, ram, model_bytes, quant);
    let (applied, kept) = ac::apply_env_pairs(&ac::plan_env_pairs(&plan));
    let mut line = ac::plan_line(&plan, &applied, &kept);
    if let Some((maj, min)) = ac::probe_sm() {
        line.push_str(&format!(" device=sm{maj}.{min}"));
    }
    Ok(line)
}

pub fn run(cfg: FlashConfig) -> Result<()> {
    // task 347: `/metrics` のゲート (既定 on) と、`model_name` ラベルの材料。
    metrics::init_from_env();
    metrics::set_model(&cfg.name, cfg.max_ctx);
    // task 271: SIGTERM でも使用回数を失わない (保存してから終了する)。
    // モデルの読み込みより前に仕掛ける (読み込み中の停止でも 1 回は走る)。
    spawn_sigterm_saver();
    // task 119b: プールの初回使用 (モデル読み込み) の前に既定スレッド数を決める。
    ensure_rayon_pool();
    // task 796: MTP (nextn) head のロード。`--mtp-path` か `DENNO_MTP_PATH` が
    // あるときだけ読む (どちらも無ければ従来どおり読まない = 既定起動は不変)。
    // ここでは safetensors の検証と f32 展開までで、decode ループへの
    // draft 統合は次のタスク (`MtpModel::draft` が入口)。
    let _mtp = {
        let path = cfg
            .mtp_path
            .clone()
            .or_else(denno_engine::model::mtp::path_from_env);
        match path {
            Some(p) => {
                let t = Instant::now();
                let m = denno_engine::model::mtp::MtpModel::open(&p)
                    .with_context(|| format!("MTP 重み {} を読めない", p.display()))?;
                eprintln!(
                    "[flash] mtp: {} tensors, {:.1} MB, host f32 {:.0} MB, {:.2} s ({})",
                    m.tensor_count(),
                    m.file_bytes() as f64 / 1e6,
                    m.host_f32_bytes() as f64 / 1e6,
                    t.elapsed().as_secs_f64(),
                    p.display()
                );
                Some(m)
            }
            None => {
                eprintln!(
                    "[flash] mtp: off (--mtp-path / {} 未設定)",
                    denno_engine::model::mtp::MTP_PATH_ENV
                );
                None
            }
        }
    };
    // task 361: ミス経路の行幅 (`DENNO_EXP_MISS_ROWS`、既定 16) を起動時に 1 行で出す。
    eprintln!(
        "[flash] exp miss rows: {} (DENNO_EXP_MISS_ROWS)",
        denno_engine::model::qwen4exp::exp_miss_rows_report()
    );
    // task 448: decode の flash decoding の実装選択を 1 行で出す (COND の env が
    // 届いたか、ビルドが新しいかをログで確かめる。数値経路には触らない)。
    if let Some(line) = denno_engine::gpu_report::attn_fdec_boot_line() {
        eprintln!("{line}");
    }
    // task 356: expert 段の寄与 D2H を専用ストリームへ出すゲート
    // (`DENNO_DEC_EXP_D2H_STREAM`、既定 off)。COND の env が届いたかをログで確かめる。
    eprintln!(
        "[flash] exp d2h stream: {} (DENNO_DEC_EXP_D2H_STREAM)",
        if denno_engine::model::exp_iq::dec_d2h_stream_enabled() {
            "on"
        } else {
            "off"
        }
    );
    // task 356: 和の出力を pinned のホストへ直接書くゲート
    // (`DENNO_DEC_EXP_SUM_ZC`、既定 off)。D2H を消す経路。
    eprintln!(
        "[flash] exp sum zc: {} (DENNO_DEC_EXP_SUM_ZC)",
        if denno_engine::model::exp_iq::sum_zc_enabled() {
            "on"
        } else {
            "off"
        }
    );
    // task 498: seg 層の和 (10 KB) の D2H を pinned バウンス経由にするゲート
    // (`DENNO_SEG_SUM_PIN`、既定 off)。COND の env が届いたかを確かめる。
    eprintln!("{}", denno_engine::model::seg_prof::seg_sum_pin_boot_line());
    // task 502: seg 層の和の D2H を非同期にするゲート (`DENNO_SEG_SUM_ASYNC`、既定 off)。
    eprintln!("{}", denno_engine::model::seg_prof::seg_sum_async_boot_line());
    // task 213: 既定 on のゲートが必要とする PTX を起動時に監査する。
    // PTX が無いとその機能は**静かに**従来経路へ落ちる (REPORT_P204 の fold/upw)。
    // 起動は止めず、summary 1 行 + 欠落ごとに 1 行を出す。
    for line in denno_engine::gate_audit::audit_lines() {
        eprintln!("{line}");
    }
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
    let n_expert;
    let expert_bytes;
    let fingerprint;
    let shape;
    {
        let p = Qwen4ExpParams::from_meta(&f).context("メタデータを読めない")?;
        // task 152: **全パート**を数える。t149c の `[flash] vram budget` は 2 分割の
        // part-1 しか数えておらず、実際に VRAM へ上がった dense 重みの 13% だった。
        match denno_engine::gpu_budget::flash_budget_from_parts(&parts, &p, cfg.max_ctx) {
            Ok(b) => eprintln!(
                "{}",
                denno_engine::gpu_budget::budget_line_with_parts(&b, parts.len())
            ),
            Err(e) => eprintln!("[flash] vram budget: 全パートを数えられない: {e:#}"),
        }
        // P818: `--vram-budget auto` / `DENNO_AUTO_CONFIG=1`。未設定の
        // `DENNO_*` だけを計画値で埋める (明示 env は上書きしない)。
        // モデルを組む前 = expert の予算が読まれる前に適用する。
        if denno_engine::auto_config::auto_requested(
            cfg.vram_budget.as_deref(),
            std::env::var(denno_engine::auto_config::AUTO_ENV)
                .ok()
                .as_deref(),
        ) {
            match apply_auto_config(cfg.vram_budget.as_deref(), &parts, &f) {
                Ok(line) => eprintln!("{line}"),
                Err(e) => eprintln!("[auto-config] skip: {e:#}"),
            }
        }
        n_layer = p.n_layer;
        n_expert = p.n_expert;
        // task 146: 使用回数ファイルのモデル同一性 (全層の expert 3 行列の合計バイト)。
        expert_bytes = denno_engine::model::qwen4exp::expert_identity_bytes(&f, &p);
        // task 798: テンソル目録の指紋。同じ名前 + 同じバイト数でも別リビジョンの
        // プロファイルは読まない (保存ファイルにも載る)。
        fingerprint = denno_engine::model::exp_stats::fingerprint_tensors(&f);
        // task 243: 冷えた page cache を起動時に温める (expert 行列だけ)。
        // 最初のリクエストが SSD から読み直すぶん (lab 実測で約 14 s) を
        // `listening` の前に前払いする。数値経路には触らない。
        // `DENNO_START_DROP_CACHE=1` は診断用 (冷えた機械を再現する)。
        if denno_engine::model::start_warm::drop_enabled() {
            eprintln!(
                "{}",
                denno_engine::model::start_warm::drop_cache(&parts)
            );
        }
        if denno_engine::model::start_warm::enabled() {
            let mut ranges = denno_engine::model::start_warm::expert_ranges(
                f.tensor_order.iter().map(|s| s.as_str()),
                |name| {
                    f.tensor_mapping_parts(name)
                        .map(|(file, off, len)| (file, off as u64, len as u64))
                },
            );
            // expert のあとに PLE の表とホスト型の重み (lab t243b の残差)。
            // 後に読むほど page cache の LRU で残りやすい。
            ranges.extend(denno_engine::model::start_warm::extra_ranges(
                f.tensor_order.iter().map(|s| s.as_str()),
                |name| f.tensors.get(name).map(|t| (t.ty, t.nbytes)),
                |name| {
                    f.tensor_mapping_parts(name)
                        .map(|(file, off, len)| (file, off as u64, len as u64))
                },
                denno_engine::model::start_warm::extra_budget(),
            ));
            let rep = denno_engine::model::start_warm::warm(
                &parts,
                &ranges,
                denno_engine::model::start_warm::budget_secs(),
                denno_engine::model::start_warm::read_threads(),
            );
            eprintln!("{}", rep.line());
        } else {
            eprintln!(
                "[start-prefetch] off ({}=0 で明示 off)",
                denno_engine::model::start_warm::ENV_ON
            );
        }
        // task 349: 起動後の非同期温め (既定 off)。`listening` より前に
        // 戻りを待たないので、起動時間も最初のリクエストも待たせない。
        // extra (PLE + ホスト重み) を先、expert を後 (`DENNO_PF_BG_WARM_MB` まで)。
        if denno_engine::model::start_warm::bg_on() {
            let mut ranges = denno_engine::model::start_warm::extra_ranges(
                f.tensor_order.iter().map(|s| s.as_str()),
                |name| f.tensors.get(name).map(|t| (t.ty, t.nbytes)),
                |name| {
                    f.tensor_mapping_parts(name)
                        .map(|(file, off, len)| (file, off as u64, len as u64))
                },
                denno_engine::model::start_warm::extra_budget(),
            );
            let extra_bytes: u64 = ranges.iter().map(|r| r.2).sum();
            let mut experts = denno_engine::model::start_warm::expert_ranges(
                f.tensor_order.iter().map(|s| s.as_str()),
                |name| {
                    f.tensor_mapping_parts(name)
                        .map(|(file, off, len)| (file, off as u64, len as u64))
                },
            );
            let cap = (denno_engine::model::start_warm::bg_mb() as u64) << 20;
            denno_engine::model::start_warm::cap_ranges(&mut experts, cap);
            eprintln!(
                "[pf-bg-warm] plan extra={:.1} MB experts={:.1} MB (cap {:.1} MB)",
                extra_bytes as f64 / 1e6,
                experts.iter().map(|r| r.2).sum::<u64>() as f64 / 1e6,
                cap as f64 / 1e6
            );
            ranges.extend(experts);
            let _ = denno_engine::model::start_warm::spawn_background(
                parts.clone(),
                ranges,
                denno_engine::model::start_warm::bg_secs(),
                denno_engine::model::start_warm::bg_threads(),
            );
        }
        // task 217: ビューワが層を描けるように、層の種別と MoE / hc の形を残す。
        let layer_kinds = (0..p.n_layer)
            .map(|il| match p.layer_kind(il) {
                Ok(LayerKind::FullAttn) => "full",
                _ => "gdn",
            })
            .collect();
        shape = ModelShape {
            layers: p.n_layer,
            layer_kinds,
            n_expert: p.n_expert,
            n_expert_used: p.n_expert_used,
            n_hc: p.n_hc,
            n_embd: p.n_embd,
            // task 220: PLE の層は GGUF のメタデータ (実モデルは 1 層)。
            ple_layers: p.ple_layers.clone(),
        };
        // task 218: ステアリングの形とモデル id を**モデルを組む前**に決める
        // (記録 / 適用の HTTP は最初の generate より前に来る)。ここで決めないと
        // `GET /v1/steer` と `POST /v1/steer/record` が「未初期化」になる。
        // モデル id はテレメトリと同じ `cfg.dir` の末尾。
        let steer_model = match std::env::var("DENNO_STEER_MODEL") {
            Ok(m) if !m.trim().is_empty() => m.trim().to_string(),
            _ => cfg
                .dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default(),
        };
        steer::init(
            steer::Shape {
                n_layer: p.n_layer,
                n_hc: p.n_hc,
                n_embd: p.n_embd,
            },
            &steer_model,
        );
    }
    drop(f);
    // task 833: 起動時のレシピ。`--steer-recipe` か `DENNO_STEER_RECIPE` が
    // あるときだけ読む (どちらも無ければ従来どおり = 既定の推論は不変)。
    if let Some(path) = steer_recipe_path(&cfg) {
        match apply_steer_recipe(&path) {
            Ok(line) => eprintln!("{line}"),
            Err(e) => bail!("--steer-recipe {}: {e}", path.display()),
        }
    }
    let batch = std::env::var("DENNO_PREFILL_BATCH").ok().as_deref() != Some("0");
    let chunk = std::env::var("DENNO_PREFILL_BATCH_CHUNK").unwrap_or_else(|_| "8".into());

    // task 146/263: (層, expert) の使用回数と、起動プラン (VRAM → RAM)。
    // 記録先は `DENNO_EXPERT_IQ_STATS_FILE` (明示) か `DENNO_EXPERT_STATS_DIR`
    // (モデルごとのファイル)。どちらも無ければ丸ごと off。
    // **起動時のモデル構築より前に起こす**のが大事で、構築 (数秒) の間に
    // 上位 expert を載せておかないと、最初のリクエストが来た時点で数本しか
    // 載っていない (lab t146b)。背景スレッドなので `/v1/models` は遅れない。
    let model_dir_name = cfg
        .dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    let stats_path = denno_engine::model::exp_stats::set_identity(
        &model_dir_name,
        expert_bytes,
        fingerprint,
        n_layer,
        n_expert,
    );
    let _stats_guard = if denno_engine::model::exp_stats::enabled() {
        eprintln!(
            "[exp-stats] on file={} (iq_gpu={} vram={} ram={} fp={:016x} warm_experts={} warm_mb={})",
            stats_path
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            u8::from(denno_engine::model::exp_iq::iq_gpu_enabled()),
            std::env::var("DENNO_EXPERT_VRAM_MB").unwrap_or_default(),
            std::env::var("DENNO_EXPERT_RAM_MB").unwrap_or_default(),
            fingerprint,
            std::env::var(denno_engine::model::exp_stats::WARM_EXPERTS_ENV).unwrap_or_default(),
            std::env::var(denno_engine::model::exp_stats::WARM_MB_ENV).unwrap_or_default()
        );
        // 起動プラン: 回数の多い順に VRAM tier → RAM tier。decode 中は譲り、
        // 走行中は RAM tier を回数に追従させる (低優先度)。
        denno_engine::model::qwen4exp::spawn_expert_plan(parts.clone(), model_dir_name.clone());
        // 保存: 新しい回数があるときだけ、20 秒ごとに原子的に書く (decode の外)。
        std::thread::spawn(|| loop {
            std::thread::sleep(std::time::Duration::from_secs(
                denno_engine::model::exp_stats::SAVE_PERIOD_SECS,
            ));
            if let Some(line) = denno_engine::model::exp_stats::save_now() {
                eprintln!("{line}");
            }
        });
        Some(StatsSaveOnExit)
    } else {
        None
    };

    // P51: 推論テレメトリ (denno-view が購読する WS)。既定で有効で、
    // `DENNO_TELEMETRY=0` のときだけ無効。無効時はチャネルもスレッドも作らない。
    // P812: 文脈スロット数 (CLI 引数 > `DENNO_KV_SLOTS` > 既定 2)。
    let kv_slots = cfg
        .kv_slots
        .unwrap_or_else(|| kv_slots_from(std::env::var("DENNO_KV_SLOTS").ok().as_deref()));
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
            slots: kv_slots as u32,
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

    // task 853: sampler の top-k / エントロピー (DENNO_TELEMETRY_TOPK, 既定 off)。
    // ビューワ (/2d の token probs 欄、3D の候補表示) が要る時だけ有効にする。
    let topk_on = telem_on && denno_telemetry::topk_telemetry_enabled();
    eprintln!(
        "[flash] telemetry topk: {}",
        if topk_on { "on (top-8 を毎トークン)" } else { "off (DENNO_TELEMETRY_TOPK=1 で有効)" }
    );

    // P812: 文脈スロット数 (CLI 引数 > `DENNO_KV_SLOTS` > 既定 2)。
    // スロット 1 本あたりのメモリ目安 (GDN 状態 117 MiB + KV 48 KiB/token)。
    if kv_slots > 2 {
        eprintln!(
            "[flash] kv-slots={kv_slots} (LCP + LRU)。追加 {} 本ぶんのメモリ目安: GDN 状態 {} MiB + KV {} MiB (max_ctx={})",
            kv_slots - 2,
            (kv_slots - 2) * 117,
            (kv_slots - 2) * cfg.max_ctx * 48 / 1024,
            cfg.max_ctx
        );
    }
    // task 866: radix 索引の状態を 1 行だけ出す (既定 off なので既定ログは不変)。
    if kv_radix_enabled(std::env::var("DENNO_KV_RADIX").ok().as_deref()) {
        if kv_slots >= 3 {
            eprintln!(
                "[flash] kv-radix=on: パス圧縮トライ + 前置き優先の追い出し (slots={kv_slots})"
            );
        } else {
            eprintln!(
                "[flash] kv-radix: slots={kv_slots} では不活性 (main/aux 経路のまま。--kv-slots 3 以上で有効)"
            );
        }
    }
    let mut flash = Flash {
        parts,
        max_ctx: cfg.max_ctx,
        name: cfg.name.clone(),
        default_max: cfg.max_tokens,
        tok,
        batch,
        shape,
        dec: None,
        slots: (0..kv_slots).map(|_| KvSlot::empty()).collect(),
        tick: 0,
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
    // task 347: 起動直後の VRAM / RAM tier のゲージとディスク読みを 1 回サンプル。
    metrics::sample_tiers();
    metrics::sample_disk_reads();
    eprintln!(
        "[flash] metrics: {} (GET /metrics, {}={})",
        if metrics::enabled() { "on" } else { "off" },
        metrics::ENV_ON,
        if metrics::enabled() { "既定" } else { "0" }
    );

    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = TcpListener::bind(&addr).with_context(|| format!("{addr} を listen できない"))?;
    eprintln!("[flash] listening on http://{addr} model={}", cfg.name);
    let rx = spawn_acceptor(listener);
    for s in rx {
        match s {
            Ok((stream, r, req)) => {
                let t = Instant::now();
                // 逐次処理 (モデルは 1 つ、同時リクエストは受けない)
                // P817: drain は「この 1 本が終わるまで」待つので、処理中を数える。
                IN_FLIGHT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let handled = handle_req(&mut flash, stream, r, req);
                IN_FLIGHT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                if let Err(e) = handled {
                    eprintln!("[flash] リクエスト失敗: {e:#}");
                }
                eprintln!("[flash] request done {:.1}s", t.elapsed().as_secs_f64());
                // task 498: seg 和の D2H の pinned バウンスが発火した回数 (ゲート on の
                // ときだけ)。0 なら分岐が通っていないので採否に使わない。
                if let Some(line) = denno_engine::model::seg_prof::seg_sum_pin_report() {
                    eprintln!("{line}");
                }
                // task 502: seg 和の D2H の非同期コピーが発火した回数 (ゲート on の
                // ときだけ)。0 なら分岐が通っていないので採否に使わない。
                if let Some(line) = denno_engine::model::seg_prof::seg_sum_async_report() {
                    eprintln!("{line}");
                }
                // task 594: seg prologue の H2D を pinned + 非同期にした回数
                // (ゲート on のときだけ)。0 なら分岐が通っていない。
                if let Some(line) = denno_engine::model::seg_prof::seg_h2d_report() {
                    eprintln!("{line}");
                }
                // task 534: 8 バイトロード版 (gu8 / dn8) の起動回数 (ゲート on の
                // ときだけ)。0 なら新しいカーネルは発火していない。
                if let Some(line) = denno_engine::model::seg_prof::seg_fast_report() {
                    eprintln!("{line}");
                }
                // task 149: リクエスト境界。prefill の一時領域 (arena) を返し、
                // 確保の内訳を 1 表だけ出す (1 本目と長い文脈の最初の 1 回)。
                // `DENNO_SCRATCH_FREE=0` / `DENNO_Q4_PROFILE` なしでは何もしない。
                let ctx_tokens = flash
                    .slots
                    .iter()
                    .filter_map(|s| s.ctx.as_ref())
                    .map(|c| c.toks.len())
                    .max()
                    .unwrap_or(0);
                denno_engine::mem_audit::request_boundary(ctx_tokens, 4096);
                // P-pf-arena: 重み登録の成否をリクエスト単位で 1 行 (登録を試みた
                // ときだけ)。GPU_BROKEN のラッチはプロセス寿命なので reset しない。
                let st = denno_engine::gpu_budget::weight_upload_stats();
                if st.attempted() > 0 {
                    eprintln!("{}", denno_engine::gpu_budget::summary_line(&st));
                    denno_engine::gpu_budget::reset_weight_upload_stats();
                }
                // モデルの選び直し: 応答を返したあとで終了し、起動ループに任せる。
                let code = EXIT_AFTER.load(std::sync::atomic::Ordering::SeqCst);
                if code != 0 {
                    // 記録中のときだけ保存する (記録していないときの flush はエラーを返す)。
                    if steer::rec_status().is_some() {
                        if let Err(e) = steer::flush() {
                            eprintln!("[steer] 記録の保存に失敗: {e}");
                        }
                    }
                    // task 271: 使用回数も終了前にもう一度書く (選択の応答で
                    // 保存済みなら save_now は何もしない)。
                    save_stats_on_stop("model select");
                    eprintln!("[flash] exit {code} (モデルの切り替え)");
                    std::process::exit(code);
                }
            }
            Err(e) => eprintln!("[flash] accept 失敗: {e}"),
        }
    }
    Ok(())
}

// ---------- テスト ----------

/// P118: 合成モデルで `generate()` を回す発火テスト (cfg(test) のみ)。
#[cfg(test)]
#[path = "serve_flash_synth.rs"]
mod flash_synth;

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

    /// task 926: `/v1/cache/stats` の本文 (キーと率) を固定する。
    #[test]
    fn cache_stats_json_shape_and_rates() {
        let v = cache_stats_json(2, 4, 3, 1, 120);
        assert_eq!(v["slots_used"], 2);
        assert_eq!(v["slots_capacity"], 4);
        assert_eq!(v["hits"], 3);
        assert_eq!(v["misses"], 1);
        assert!((v["hit_rate"].as_f64().unwrap() - 0.75).abs() < 1e-12);
        assert_eq!(v["total_reused_tokens"], 120);
        // リクエスト 0 本なら率は 0 (NaN を返さない)。
        let z = cache_stats_json(0, 2, 0, 0, 0);
        assert_eq!(z["hit_rate"].as_f64().unwrap(), 0.0);
    }

    /// task 926: 再利用の計上 (hit / miss / トークン) が累積する。
    /// 同じバイナリの他のテストも generate() 経由で計上するので、差分は下限で見る。
    #[test]
    fn record_cache_reuse_accumulates() {
        let (h0, m0, r0) = cache_counters();
        record_cache_reuse(7);
        record_cache_reuse(0);
        let (h, m, r) = cache_counters();
        assert!(h > h0, "hit が増えていない: {h0} -> {h}");
        assert!(m > m0, "miss が増えていない: {m0} -> {m}");
        assert!(r >= r0 + 7, "再利用トークンが増えていない: {r0} -> {r}");
    }

    /// 生成中 (主スレッドが要求を取りに来ない間) でも `POST /v1/abort` は読み取り
    /// スレッドで即答し、停止要求を立てる。ほかの要求は主スレッドへ順に渡る。
    #[test]
    fn acceptor_answers_abort_while_main_is_busy() {
        use std::io::Write as _;
        let _lk = ABORT_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let rx = spawn_acceptor(l);
        // 主スレッドが生成中 (rx を読まない) のつもり
        let _gen = GenGuard::enter();
        let mut a = TcpStream::connect(addr).unwrap();
        a.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        a.write_all(b"POST /v1/abort HTTP/1.1\r\nHost: h\r\n\r\n").unwrap();
        let resp = read_all(&mut a);
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
        assert!(resp.contains("\"aborted\":true"), "{resp}");
        assert!(take_abort(), "停止要求が立っていない");
        // ふつうの要求は主スレッド側の channel に届く (読み取りスレッドでは答えない)
        let mut b = TcpStream::connect(addr).unwrap();
        b.write_all(b"GET /v1/models HTTP/1.1\r\nHost: h\r\n\r\n").unwrap();
        let job = rx.recv_timeout(Duration::from_secs(5)).expect("要求が主スレッドに届かない").unwrap();
        assert_eq!(job.2.method, "GET");
        assert_eq!(job.2.path, "/v1/models");
        drop(_gen);
        // 生成していないときは空振り
        let mut c = TcpStream::connect(addr).unwrap();
        c.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        c.write_all(b"POST /v1/abort HTTP/1.1\r\nHost: h\r\n\r\n").unwrap();
        assert!(read_all(&mut c).contains("\"aborted\":false"));
        assert!(!take_abort());
    }

    /// P51: 従来からの GET 応答はバイト列まで変えない (テレメトリの有無に依らない)。
    /// task 817: `/v1/models` は運用メタデータを足したので、旧フィールドが
    /// 残っていることを JSON で確認する (バイト列の固定は /health だけ)。
    #[test]
    fn static_get_responses_are_byte_identical() {
        let (mut srv, mut cli) = loopback();
        send_json(&mut srv, "200 OK", &get_static("/health", "m", 4096).unwrap()).unwrap();
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
        send_json(&mut srv, "200 OK", &get_static("/v1/models", "m", 4096).unwrap()).unwrap();
        drop(srv);
        // task 347: `owned_by: "denno"` が付く (既定)。task 817: context_length と
        // capabilities が付く。既存フィールドは残る。
        let raw = read_all(&mut cli);
        let body = raw.split("\r\n\r\n").nth(1).expect("本文が無い");
        let v: Value = serde_json::from_str(body).expect("JSON ではない");
        let m = &v["data"][0];
        assert_eq!(m["id"], "m");
        assert_eq!(m["object"], "model");
        assert_eq!(m["type"], "model");
        assert_eq!(m["display_name"], "m");
        assert_eq!(m["created_at"], "2026-10-27T00:00:00Z");
        assert_eq!(m["owned_by"], "denno");
        assert_eq!(m["context_length"], 4096);
        assert_eq!(m["capabilities"]["anthropic"], true);
        assert_eq!(m["capabilities"]["openai"], true);
        assert_eq!(m["capabilities"]["tools"], true);
        assert!(m["capabilities"]["telemetry"].is_boolean());
    }

    #[test]
    fn static_get_covers_only_the_old_routes() {
        assert_eq!(
            get_static("/healthz", "x", 4096),
            Some(json!({"ok": true, "model": "x"}))
        );
        assert!(get_static("/", "x", 4096).is_none());
        assert!(get_static("/v1/telemetry/ws", "x", 4096).is_none());
        assert!(get_static("/v1/messages", "x", 4096).is_none());
    }

    /// task 817: `/ready` は drain していないとき 200、`DRAINING` が立つと 503。
    #[test]
    fn ready_reflects_draining() {
        // このテストはグローバルな DRAINING を触るので、前後で戻す。
        DRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
        let (status, body) = ready_response("m", 4096);
        assert_eq!(status, "200 OK");
        assert_eq!(body["ready"], true);
        assert_eq!(body["draining"], false);
        assert_eq!(body["context_length"], 4096);

        DRAINING.store(true, std::sync::atomic::Ordering::SeqCst);
        let (status, body) = ready_response("m", 4096);
        assert_eq!(status, "503 Service Unavailable");
        assert_eq!(body["ready"], false);
        assert_eq!(body["draining"], true);
        assert_eq!(body["status"], "draining");
        DRAINING.store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// task 263: `/v1/experts/stats` の形。env 未設定のテストでは off で、
    /// file / recorded / plan は null、tier の数字は 0 (キーは必ず在る)。
    #[test]
    fn experts_stats_json_has_every_tier() {
        let v = experts_stats_json();
        for k in [
            "enabled",
            "file",
            "last_save",
            "recorded",
            "vram",
            "ram",
            "mmap",
            "last_request",
            "coverage",
            "plan",
        ] {
            assert!(v.get(k).is_some(), "{k} が無い: {v}");
        }
        // 形だけを見る (task 271 で記録先を設定するテストが同居するため、
        // enabled / file の値は「型が正しい」ことだけ確かめる)。
        assert!(v["enabled"].is_boolean(), "{v}");
        assert!(v["file"].is_string() || v["file"].is_null(), "{v}");
        for t in ["vram", "ram", "mmap"] {
            for k in ["experts", "mb"] {
                assert!(v[t].get(k).is_some(), "{t}.{k} が無い: {v}");
            }
        }
        for k in ["vram", "ram", "total"] {
            let x = v["coverage"][k].as_f64().expect("数値");
            assert!((0.0..=1.0).contains(&x), "{k}={x}");
        }
        // JSON にできる (NaN が混ざっていない)
        assert!(serde_json::to_string(&v).is_ok());
    }

    /// task 271: 直前のリクエストの tier 別ヒット率。exp_iq の参照/ヒットと
    /// expert_ram のミス転送元から写し、lookups を超えない (率は 0..1)。
    ///
    /// 共有カウンタを触るので、このテストの中だけで 0 に戻してから使う。
    #[test]
    fn last_tier_hits_split_vram_ram_mmap() {
        use denno_engine::model::{exp_iq, expert_ram};
        exp_iq::stats_reset();
        expert_ram::miss_reset();
        // 10 参照中 6 が VRAM、ミス 4 のうち 3 が RAM tier、1 が mmap。
        exp_iq::note_lookup(10, 6);
        expert_ram::note_miss_source(true);
        expert_ram::note_miss_source(true);
        expert_ram::note_miss_source(true);
        expert_ram::note_miss_source(false);
        record_last_tier_hits();
        let got = *LAST_TIER.lock().unwrap();
        assert_eq!(
            got,
            Some(LastTierHits {
                lookups: 10,
                vram: 6,
                ram: 3,
                mmap: 1,
            })
        );
        // エンドポイントにも同じ数字が載る
        let v = experts_stats_json();
        assert_eq!(v["last_request"]["lookups"], 10);
        assert_eq!(v["last_request"]["vram"]["hits"], 6);
        assert_eq!(v["last_request"]["ram"]["hits"], 3);
        assert_eq!(v["last_request"]["mmap"]["hits"], 1);
        for (t, want) in [("vram", 0.6), ("ram", 0.3), ("mmap", 0.1)] {
            let x = v["last_request"][t]["hit_rate"].as_f64().expect("数値");
            assert!((x - want).abs() < 1e-9, "{t} hit_rate={x} (期待 {want})");
        }

        // 参照 0 件 (prefill だけで終わったリクエスト) では前の値を残す。
        exp_iq::stats_reset();
        expert_ram::miss_reset();
        record_last_tier_hits();
        assert_eq!(*LAST_TIER.lock().unwrap(), got);

        // 転送元の積算がミス本数を超えても率は 1 を超えない (丸める)。
        exp_iq::note_lookup(2, 0);
        for _ in 0..5 {
            expert_ram::note_miss_source(true);
        }
        record_last_tier_hits();
        let v = experts_stats_json();
        assert_eq!(v["last_request"]["ram"]["hits"], 2);
        assert_eq!(v["last_request"]["mmap"]["hits"], 0);
        assert_eq!(v["last_request"]["ram"]["hit_rate"], 1.0);
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

    /// task 119b: rayon 既定スレッド数の純粋関数 (max(1, 論理 CPU - 2))。
    #[test]
    fn default_rayon_threads_is_cpus_minus_two_floor_one() {
        assert_eq!(default_rayon_threads(16), 14);
        assert_eq!(default_rayon_threads(14), 12);
        assert_eq!(default_rayon_threads(3), 1);
        assert_eq!(default_rayon_threads(2), 1);
        assert_eq!(default_rayon_threads(1), 1);
        assert_eq!(default_rayon_threads(0), 1);
    }

    // ---- task 825: analyze / export / import のハンドラ -------------------

    /// steer のグローバル (init / スナップショット) を触るテストの直列化。
    static STEER825_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// 小さな形 (4 層) で A/B の 2 スナップショットを用意する。
    fn steer825_fixture() -> std::path::PathBuf {
        let n = {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        };
        let dir = std::env::temp_dir().join(format!("denno-cli-825-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("DENNO_STEER_DIR", &dir);
        let s = steer::Shape {
            n_layer: 4,
            n_hc: 2,
            n_embd: 3,
        };
        steer::init(s, "cli-825");
        let st = s.stride();
        for (name, seed) in [("ablit_A", 0.5f32), ("ablit_B", -0.5f32)] {
            steer::start_record(name).expect("記録開始");
            for t in 0..2usize {
                for il in 0..s.n_layer {
                    let row: Vec<f32> = (0..st)
                        .map(|i| ((t * st + i) as f32 * 0.125) + seed)
                        .collect();
                    steer::acc_host(il, &row);
                }
            }
            steer::stop_record().expect("保存");
        }
        dir
    }

    /// `/v1/steer/analyze` の JSON が層ごとの指標と推奨区間を返す。
    #[test]
    fn analyze_handler_reports_per_layer_metrics() {
        let _g = STEER825_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = steer825_fixture();
        let v = analyze_steer(&json!({"pos": "ablit_A", "neg": "ablit_B"})).expect("analyze");
        assert!(serde_json::to_string(&v).is_ok(), "JSON にできない: {v}");
        assert_eq!(v["pos"], "ablit_A");
        assert_eq!(v["neg"], "ablit_B");
        assert_eq!(v["n_layer"], 4);
        let layers = v["layers"].as_array().expect("layers");
        assert_eq!(layers.len(), 4);
        for (i, l) in layers.iter().enumerate() {
            assert_eq!(l["layer"], i);
            let sep = l["sep_score"].as_f64().expect("sep_score");
            assert!((0.0..=1.0).contains(&sep), "sep_score {sep}");
            let cos = l["cos_sim"].as_f64().expect("cos_sim");
            assert!((-1.0..=1.0).contains(&cos), "cos_sim {cos}");
            assert!(l["diff_norm"].as_f64().expect("diff_norm") > 0.0);
        }
        let r = v["recommended_range"].as_array().expect("range");
        assert_eq!(r.len(), 2);
        let (f, t) = (
            r[0].as_u64().unwrap() as usize,
            r[1].as_u64().unwrap() as usize,
        );
        assert!(f >= 1 && t > f && t <= 3, "range {f}-{t}");
        let peak = v["peak_layer"].as_u64().unwrap() as usize;
        assert!((1..3).contains(&peak), "peak {peak}");

        // terms 形式 (A +1 / B -1) は名前 2 つの分析と同じ数字になる。
        let w = analyze_steer(&json!({
            "terms": [
                {"snapshot": "ablit_A", "coef": 1.0},
                {"snapshot": "ablit_B", "coef": -1.0},
            ]
        }))
        .expect("analyze terms");
        assert_eq!(v["layers"], w["layers"]);
        assert_eq!(v["recommended_range"], w["recommended_range"]);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 不正な body は Err (panic しない)。
    #[test]
    fn analyze_handler_never_panics_on_bad_input() {
        let _g = STEER825_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = steer825_fixture();
        for body in [
            json!(null),
            json!([1, 2]),
            json!({}),
            json!({"pos": "ablit_A"}),
            json!({"neg": "ablit_B"}),
            json!({"pos": 1, "neg": 2}),
            json!({"pos": "ablit_A", "neg": "ablit_B", "bogus": 1}),
            json!({"pos": "nope", "neg": "ablit_B"}),
            json!({"pos": "ablit_A", "neg": "nope"}),
            json!({"terms": []}),
            json!({"terms": "x"}),
            json!({"terms": [1]}),
            json!({"terms": [{"snapshot": "ablit_A"}]}),
            json!({"terms": [{"snapshot": 1, "coef": 1}]}),
            json!({"terms": [{"snapshot": "ablit_A", "coef": 1, "x": 0}]}),
            json!({"terms": [{"snapshot": "ablit_A", "coef": "1"}]}),
            json!({"terms": [{"snapshot": "ablit_A", "coef": 1}]}),
            json!({"terms": [{"snapshot": "nope", "coef": 1}, {"snapshot": "ablit_B", "coef": -1}]}),
        ] {
            assert!(analyze_steer(&body).is_err(), "通ってしまった: {body}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// export (バイナリ / JSON) と import (同じ本文を戻す) の往復。
    #[test]
    fn export_and_import_handlers_round_trip() {
        let _g = STEER825_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = steer825_fixture();
        let (body, ctype, fname) =
            export_body("/v1/steer/export?name=ablit_A").expect("export raw");
        assert_eq!(ctype, "application/octet-stream");
        assert_eq!(fname, "ablit_A.dns");
        assert!(body.starts_with(steer::FILE_MAGIC.as_bytes()));

        // ヘッダ (Content-Disposition と長さ) まで含めて送れる。
        let (mut srv, mut cli) = loopback();
        send_bytes(&mut srv, "200 OK", ctype, &fname, &body).expect("send");
        drop(srv);
        let resp = read_all(&mut cli);
        assert!(resp.starts_with("HTTP/1.1 200 OK"), "{resp}");
        assert!(
            resp.contains("Content-Disposition: attachment; filename=\"ablit_A.dns\""),
            "{resp}"
        );
        assert!(resp.contains(&format!("Content-Length: {}", body.len())));

        // 名前を変えて取り込む (ディスク + メモリ)。
        let out = import_steer(&body, Some("ablit_copy")).expect("import raw");
        assert_eq!(out["imported"]["name"], "ablit_copy");
        assert_eq!(out["imported"]["layers"], 4);
        assert!(out["snapshots"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s["name"] == "ablit_copy"));

        // JSON (base64) の往復。
        let (jbody, jctype, jname) =
            export_body("/v1/steer/export?name=ablit_A&format=json").expect("export json");
        assert_eq!(jctype, "application/json");
        assert!(jname.ends_with(".json"), "{jname}");
        let j: Value = serde_json::from_slice(&jbody).expect("JSON");
        assert_eq!(j["name"], "ablit_A");
        assert!(j["data_base64"].as_str().unwrap().len() > 16);
        let out = import_steer(&jbody, Some("ablit_json")).expect("import json");
        assert_eq!(out["imported"]["name"], "ablit_json");

        // 不正な入力は Err。
        assert!(import_steer(b"", None).is_err());
        assert!(import_steer(b"nope", None).is_err());
        assert!(import_steer(b"{}", None).is_err());
        assert!(import_steer(&body, Some("a.b")).is_err(), "名前の検証が効いていない");
        assert!(export_body("/v1/steer/export").is_err());
        assert!(export_body("/v1/steer/export?name=").is_err());
        assert!(export_body("/v1/steer/export?name=nope").is_err());
        assert!(export_body("/v1/steer/export?name=nope&format=json").is_err());

        // query の読み方 (順序・複数)。
        assert_eq!(
            query_param("/v1/steer/export?format=json&name=A", "name").as_deref(),
            Some("A")
        );
        assert_eq!(query_param("/v1/steer/export?name=A", "format"), None);
        assert_eq!(query_param("/v1/steer/export", "name"), None);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// task 829: `/v1/steer/record` は "all" と "last_prompt_token" を受け、
    /// 未知の値は 400。`GET /v1/steer` は capture_mode を返す。
    #[test]
    fn record_params_accepts_both_capture_modes() {
        assert_eq!(
            record_params(&json!({"name": "A"})).expect("name だけ"),
            ("A".to_string(), steer::CaptureMode::All)
        );
        assert_eq!(
            record_params(&json!({"name": "A", "capture_mode": "all"})).expect("all"),
            ("A".to_string(), steer::CaptureMode::All)
        );
        assert_eq!(
            record_params(&json!({"name": "A", "capture_mode": "last_prompt_token"}))
                .expect("last_prompt_token"),
            ("A".to_string(), steer::CaptureMode::LastPromptToken)
        );
        assert!(record_params(&json!({"name": "A", "capture_mode": "bogus"})).is_err());
        assert!(record_params(&json!({"name": "A", "capture_mode": 1})).is_err());
        assert!(record_params(&json!({"name": "A", "bogus": 1})).is_err());
        assert!(record_params(&json!({})).is_err());
        assert!(record_params(&json!(null)).is_err());
    }

    /// task 829: 記録中の capture_mode が status に出て、stop 後も直近値が残る。
    #[test]
    fn steer_status_exposes_capture_mode() {
        let _g = STEER825_LOCK.lock().unwrap();
        let _ = steer825_fixture();
        steer::start_record_mode("cm_A", steer::CaptureMode::LastPromptToken).expect("記録開始");
        assert_eq!(
            steer_status_to_json()["capture_mode"],
            json!("last_prompt_token")
        );
        let st = steer::Shape {
            n_layer: 4,
            n_hc: 2,
            n_embd: 3,
        };
        let row = vec![0.5f32; st.stride()];
        for il in 0..st.n_layer {
            steer::acc_host_prefill(il, &row, 3, 4);
        }
        assert_eq!(steer::rec_status().map(|(_, t)| t), Some(1));
        // decode のトークン (acc_host) は LastPromptToken では載らない。
        for il in 0..st.n_layer {
            steer::acc_host(il, &row);
        }
        assert_eq!(steer::rec_status().map(|(_, t)| t), Some(1));
        let snap = steer::stop_record().expect("保存");
        assert_eq!(snap.tokens, 1);
        assert_eq!(
            steer_status_to_json()["capture_mode"],
            json!("last_prompt_token")
        );
        // 従来どおりの start_record は "all"。
        steer::start_record("cm_all").expect("記録開始");
        assert_eq!(steer_status_to_json()["capture_mode"], json!("all"));
        steer::stop_record().ok();
    }

    // ---- task 833: レシピのハンドラ ---------------------------------------

    /// A +1 / B -1 / project を適用した状態を作る。
    fn recipe833_apply() {
        steer::set_apply(
            vec![
                steer::Term {
                    snapshot: "ablit_A".into(),
                    coef: 1.0,
                },
                steer::Term {
                    snapshot: "ablit_B".into(),
                    coef: -1.0,
                },
            ],
            1,
            3,
            0.5,
            steer::Mode::Project,
        )
        .expect("適用");
    }

    /// 適用中の設定を export し、import (検証のみ / 適用) で戻せる。
    #[test]
    fn recipe_export_import_handlers_round_trip() {
        // ステアリングのグローバルを触るので、合成モデルの HTTP テスト
        // (`flash_synth`) と同じロックを同じ順で取る。
        let _g = STEER825_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _r = super::flash_synth::REUSE_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _i = super::flash_synth::IV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = steer825_fixture();
        recipe833_apply();

        // 適用中の設定から (埋め込み無し)。
        let v = recipe_export(&json!({"name": "rc1"})).expect("export current");
        assert_eq!(v["version"], json!(steer::RECIPE_VERSION));
        assert_eq!(v["name"], "rc1");
        assert_eq!(v["mode"], "project");
        assert_eq!(v["layer_from"], 1);
        assert_eq!(v["layer_to"], 3);
        assert!((v["alpha"].as_f64().unwrap() - 0.5).abs() < 1e-6);
        assert_eq!(v["model_id"], "cli-825");
        assert_eq!(v["terms"].as_array().unwrap().len(), 2);
        assert!(v.get("snapshots").is_none(), "既定では埋め込まない");

        // 埋め込み付きは自分だけで完結する。
        let emb = recipe_export(&json!({"name": "rc1", "embed_snapshots": true}))
            .expect("export embedded");
        let snaps = emb["snapshots"].as_object().expect("snapshots");
        assert_eq!(snaps.len(), 2);
        assert!(snaps.contains_key("ablit_A") && snaps.contains_key("ablit_B"));
        assert!(snaps["ablit_A"].as_str().unwrap().len() > 16);

        // 検証だけの import は状態を変えない。
        let before = steer::apply_status().expect("apply").terms;
        let out = recipe_import(&emb, false).expect("validate only");
        assert_eq!(out["valid"], true);
        assert_eq!(out["applied"], false);
        assert_eq!(out["recipe"]["embedded"].as_array().unwrap().len(), 2);
        assert_eq!(steer::apply_status().expect("apply").terms, before);

        // apply=true で適用される (埋め込みだけでも = ディスクは使わない)。
        steer::clear_apply();
        let out = recipe_import(&json!({"recipe": emb, "apply": true}), false).expect("apply");
        assert_eq!(out["applied"], true);
        assert_eq!(out["steer"]["apply"]["mode"], "project");
        assert_eq!(out["steer"]["apply"]["layers"], json!([1, 3]));
        assert!(steer::apply_status().is_some(), "適用されていない");

        // body で指定した設定 (未適用でもよい) の export。
        let req = recipe_export(&json!({
            "name": "rc2",
            "terms": [
                {"snapshot": "ablit_A", "coef": 1.0},
                {"snapshot": "ablit_B", "coef": -1.0},
            ],
            "layers": [0, 2],
            "alpha": 0.25,
            "mode": "subtract",
            "description": "requested",
        }))
        .expect("export requested");
        assert_eq!(req["name"], "rc2");
        assert_eq!(req["mode"], "subtract");
        assert_eq!(req["layer_from"], 0);
        assert_eq!(req["layer_to"], 2);
        assert_eq!(req["description"], "requested");
        assert!(req.get("snapshots").is_none());

        // 不正は Err (panic しない)。
        for body in [
            json!(null),
            json!([1]),
            json!({"bogus": 1}),
            json!({"embed_snapshots": "yes"}),
            json!({"name": 1}),
            json!({"terms": []}),
            json!({"terms": "x"}),
            json!({"terms": [{"snapshot": "ablit_A", "coef": 1}], "layers": [0]}),
            json!({"terms": [{"snapshot": "ablit_A", "coef": 1}], "alpha": "x"}),
            json!({"terms": [{"snapshot": "ablit_A", "coef": 1}], "mode": "bogus"}),
            // 埋め込み指定で参照先が無い → Err。
            json!({"terms": [{"snapshot": "ghost", "coef": 1}], "embed_snapshots": true}),
        ] {
            assert!(recipe_export(&body).is_err(), "通ってしまった: {body}");
        }
        // 埋め込み無しなら参照先の有無は見ない (後で置いてもよいレシピ)。
        let late = recipe_export(&json!({
            "terms": [{"snapshot": "later", "coef": 1.0}],
            "layers": [0, 2],
        }))
        .expect("参照先がまだ無いレシピ");
        assert_eq!(late["terms"][0]["snapshot"], "later");
        // import の不正。
        for body in [
            json!(null),
            json!([1]),
            json!({}),
            json!({"version": 2, "name": "x", "mode": "project", "layer_from": 0, "layer_to": 1,
                   "terms": [{"snapshot": "ablit_A", "coef": 1}]}),
            json!({"recipe": {"version": 1}, "extra": 1}),
            json!({"recipe": emb, "apply": "yes"}),
        ] {
            assert!(recipe_import(&body, false).is_err(), "通ってしまった: {body}");
        }
        steer::clear_apply();
        std::fs::remove_dir_all(&dir).ok();
    }

    /// 起動時のレシピ: ファイルを読み、1 行を返し、実際に適用する。
    #[test]
    fn apply_steer_recipe_loads_file_and_logs_one_line() {
        let _g = STEER825_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let _r = super::flash_synth::REUSE_ENV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _i = super::flash_synth::IV_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let dir = steer825_fixture();
        recipe833_apply();
        let emb = recipe_export(&json!({"name": "rc_start", "embed_snapshots": true}))
            .expect("export");

        // 別プロセス相当: メモリのスナップショットを捨て、ディスクも空にする。
        steer::reset_for_test();
        let s = steer::Shape {
            n_layer: 4,
            n_hc: 2,
            n_embd: 3,
        };
        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        std::env::set_var("DENNO_STEER_DIR", &empty);
        steer::init(s, "cli-825");
        assert!(steer::snapshots().is_empty(), "ディスクに残っている");

        let path = dir.join("rc_start.json");
        std::fs::write(&path, serde_json::to_string_pretty(&emb).unwrap()).unwrap();
        let line = apply_steer_recipe(&path).expect("起動時適用");
        assert_eq!(
            line,
            format!(
                "[flash] steer recipe loaded: rc_start mode=project alpha={} layers=1..3",
                0.5f32
            )
        );
        let info = steer::apply_status().expect("適用されていない");
        assert_eq!((info.from, info.to), (1, 3));
        assert_eq!(info.mode, steer::Mode::Project);
        assert_eq!(steer::snapshots().len(), 2, "埋め込みがメモリに載っていない");

        // 壊れたファイル / 無いファイルは Err。
        let bad = dir.join("bad.json");
        std::fs::write(&bad, b"{not json").unwrap();
        assert!(apply_steer_recipe(&bad).is_err());
        assert!(apply_steer_recipe(&dir.join("missing.json")).is_err());

        // フラグは env より優先。どちらも無ければ None。
        let cfg = |p: Option<std::path::PathBuf>| FlashConfig {
            dir: dir.clone(),
            host: "127.0.0.1".into(),
            port: 0,
            max_ctx: 4096,
            name: "t".into(),
            max_tokens: 16,
            mtp_path: None,
            kv_slots: None,
            steer_recipe: p,
            vram_budget: None,
        };
        std::env::remove_var("DENNO_STEER_RECIPE");
        assert!(steer_recipe_path(&cfg(None)).is_none());
        std::env::set_var("DENNO_STEER_RECIPE", " /tmp/env_rc.json ");
        assert_eq!(
            steer_recipe_path(&cfg(None)),
            Some(std::path::PathBuf::from("/tmp/env_rc.json"))
        );
        assert_eq!(
            steer_recipe_path(&cfg(Some(path.clone()))),
            Some(path.clone()),
            "フラグが優先されていない"
        );
        std::env::remove_var("DENNO_STEER_RECIPE");
        // 空文字の env は「未設定」と同じ。
        std::env::set_var("DENNO_STEER_RECIPE", "  ");
        assert!(steer_recipe_path(&cfg(None)).is_none());
        std::env::remove_var("DENNO_STEER_RECIPE");
        steer::clear_apply();
        std::fs::remove_dir_all(&dir).ok();
    }
}
