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

use anyhow::{bail, Context, Result};
use denno_engine::gguf::GgufFile;
use denno_engine::model::prekprof;
use denno_engine::model::q4ledger;
use denno_engine::model::qwen4exp::{
    expert_w_stats_request_end, q4prof, Qwen4Exp, Qwen4ExpParams, TapTokens,
};
use denno_engine::model::qwen4exp_attn::FullAttn;
use denno_engine::model::qwen4exp_decode::{argmax, DecoderConfig, Qwen4ExpDecoder};
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

/// 直前のリクエストの状態。`toks` は `dec` が食ったトークン列 (古い順)。
struct Cached {
    dec: Qwen4ExpDecoder,
    toks: Vec<u32>,
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
    }
    denno_engine::model::q4ledger::reset();
    // task 144: IQ expert のヒット率 / GPU で回した層と本数 / 上げ直し / VRAM
    // (`DENNO_EXPERT_IQ_GPU=1` のときだけ 1 行)。リクエスト単位で読む。
    if let Some(line) = denno_engine::model::exp_iq::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_iq::stats_reset();
    // task 148: prefill の m 行 expert (ヒット率 / GPU 層 / 行数 / D2H と足し込みの
    // 時間 / 上げ直し)。`DENNO_EXPERT_IQ_GPU=1` かつ PF ゲート on のときだけ 1 行。
    if let Some(line) = denno_engine::model::exp_pf::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_pf::stats_reset();
    // task 158: decode のミス expert を GPU で回したぶんの内訳 (`DENNO_DEC_MISS_UP`
    // が on のときだけ 1 行)。1 本あたりのホスト写し時間と実効レートが読める。
    if let Some(line) = denno_engine::model::exp_miss_up::stats_line() {
        eprintln!("{line}");
    }
    denno_engine::model::exp_miss_up::stats_reset();
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
    fn splice_prompt(&self, c: &Cached, req: &GenReq, prompt: &[u32]) -> Result<Option<Vec<u32>>> {
        if !c.from_render || c.prompt_len > c.toks.len() || c.req.cont || req.cont {
            return Ok(None);
        }
        if c.req.system != req.system
            || c.req.tools != req.tools
            || c.req.enable_thinking != req.enable_thinking
        {
            return Ok(None);
        }
        // 前回のメッセージ列がそのまま前置きで、その直後に assistant の返送がある。
        let n = c.req.messages.len();
        if req.messages.len() <= n || req.messages[..n] != c.req.messages[..] {
            return Ok(None);
        }
        let (role, text) = &req.messages[n];
        if role != "assistant" {
            return Ok(None);
        }
        // 実際に食った生成トークンの復号と、返送された本文が同じ会話のときだけ。
        if !echo_matches(&self.tok, &c.toks[c.prompt_len..], text) {
            return Ok(None);
        }
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
        if !self.tap_on() {
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
        // task 146: 背景の事前ロードに「いま推論中」を知らせる (ロックを奪い合わない)。
        // ガードは戻るとき (失敗経路も) に下ろす。
        let _busy = denno_engine::model::exp_stats::busy_guard();
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
        let t1 = Instant::now();
        let mut reused = 0usize;
        let cached = self.cache.take();
        let (mut dec, mut logits) = match cached {
            Some(mut c) if c.toks.len() <= prompt.len() && prompt[..c.toks.len()] == c.toks[..] => {
                reused = c.toks.len();
                if reused < prompt.len() {
                    let lg = self.prefill_logits(&mut c.dec, &prompt[reused..])?;
                    (c.dec, lg)
                } else {
                    (c.dec, c.logits)
                }
            }
            other => {
                // task 131/141: 描画済みプロンプトをキャッシュに整列し、
                // 食い残しだけを食う (成立しなければ従来どおり full prefill)。
                let spliced = match (other.as_ref(), splice_on && prompt_from_render) {
                    (Some(c), true) => self.splice_prompt(c, req, &prompt)?,
                    _ => None,
                };
                let mut other = other;
                if let Some(np) = spliced {
                    let mut c = other.take().expect("splice は cache があるときだけ返る");
                    if np.len() >= self.max_ctx {
                        bail!(
                            "プロンプト {} トークンが上限 {} に収まらない",
                            np.len(),
                            self.max_ctx
                        );
                    }
                    reused = c.toks.len();
                    prompt = np;
                    let lg = self.prefill_logits(&mut c.dec, &prompt[reused..])?;
                    (c.dec, lg)
                } else {
                    // 落ちた原因を 1 行残す (空でないキャッシュで reuse=0 のときだけ)。
                    if let Some(c) = other.as_ref() {
                        if !c.toks.is_empty() {
                            eprintln!("{}", prefix_miss_line(&self.tok, &c.toks, &prompt));
                        }
                    }
                    // P118: 前置きが一致しないときも、decoder が残っていれば
                    // `build_model` を呼ばずに系列状態だけ消して使い回す。
                    // 組み直すと `GgufLinear` に新しい uid が振られ、uid 引きの
                    // GPU 常駐重みが全部再 upload される (cache-miss 1 回あたり
                    // 266 本 / 3.42 秒)。reset は uid に触れないので重みは常駐した
                    // ままで、数値も新規構築とビット一致する (engine 側
                    // `reset_matches_fresh_decoder_bitexact`)。前のリクエストが
                    // engine 側の失敗 (prefill / step) で落ちたときは cache が
                    // None のままなので、この経路は通らず組み直す (半端に進んだ
                    // decoder を使い回さない)。クライアント切断 (emit 失敗) の
                    // ときは task 123 で cache が残るので、この reset 経路で
                    // 重みの再 upload なしに使い回す。
                    let t_model = prekprof::stamp();
                    let mut d = match other {
                        Some(c) if model_reuse => {
                            let mut d = c.dec;
                            d.reset();
                            d
                        }
                        _ => {
                            // P78: mmap と attention/KV 状態の確保は「層の外」でも重い候補。
                            let m = self.build_model()?;
                            Qwen4ExpDecoder::new(m, DecoderConfig::new(self.max_ctx))
                        }
                    };
                    prekprof::mark(t_model, prekprof::MODEL);
                    let lg = self.prefill_logits(&mut d, &prompt)?;
                    (d, lg)
                }
            }
        };
        let t_prefill = t1.elapsed().as_secs_f64();
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
        // task 153: decode の窓で CUDA イベントの時系列を取る (`DENNO_Q4_TLINE`)。
        // off では atomic load 1 回で、イベントは 1 本も打たない。
        q4ledger::tline_begin();
        'gen: while out.len() < req.max_tokens {
            if spec_on {
                // P109: n-gram 投機ステップ。q0 は engine の argmax (greedy)。
                let d_topk = q4prof::dstamp();
                // task 155: top-8 はビューワ (WS) が繋がっているときだけ作る。
                // 読む相手が居ないときの出力は従来と同じ (誰も受け取らない)。
                let cand =
                    if t_on && self.telem.has_clients() {
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
                if t_on && self.telem.has_clients() {
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
                let toks = dec.tokens().to_vec();
                self.cache = Some(Cached {
                    dec,
                    toks,
                    logits,
                    prompt_len: n_prompt,
                    from_render: prompt_from_render,
                    req: req_sig.clone(),
                });
            }
            return Err(e);
        }
        if let Some(GenFail::Engine(e)) = failed {
            return Err(e);
        }
        // 保留していた末尾 (文字が完成しないまま終わった FFFD を含む) を最終本文へ反映。
        // stop_sequence で切ったときは last_full = text にしてあるので伸びない。
        if let Some(f) = last_full {
            if f.len() > text.len() && f.starts_with(&text) {
                text = f;
            }
        }
        let toks = dec.tokens().to_vec();
        self.cache = Some(Cached {
            dec,
            toks,
            logits,
            prompt_len: n_prompt,
            from_render: prompt_from_render,
            req: req_sig,
        });
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

pub fn run(cfg: FlashConfig) -> Result<()> {
    // task 119b: プールの初回使用 (モデル読み込み) の前に既定スレッド数を決める。
    ensure_rayon_pool();
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
        n_layer = p.n_layer;
        n_expert = p.n_expert;
        // task 146: 使用回数ファイルのモデル同一性 (全層の expert 3 行列の合計バイト)。
        expert_bytes = denno_engine::model::qwen4exp::expert_identity_bytes(&f, &p);
    }
    drop(f);
    let batch = std::env::var("DENNO_PREFILL_BATCH").ok().as_deref() != Some("0");
    let chunk = std::env::var("DENNO_PREFILL_BATCH_CHUNK").unwrap_or_else(|_| "8".into());

    // task 146: (層, expert) の使用回数と、起動時の事前ロード。
    // `DENNO_EXPERT_IQ_GPU=1` かつ `DENNO_EXPERT_IQ_STATS_FILE` が設定されている
    // ときだけ動く。**起動時のモデル構築より前に起こす**のが大事で、構築 (数秒) の
    // 間に上位 expert を載せておかないと、最初のリクエストが来た時点で数本しか
    // 載っていない (lab t146b: 昇順で下位 3,212 本しか載らず、ヒット率が素と同じ)。
    // どちらも背景スレッドなので `/v1/models` が返るまでの時間は変わらない。
    let _stats_guard = if denno_engine::model::exp_stats::enabled() {
        let dir_name = cfg
            .dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        denno_engine::model::exp_stats::set_identity(&dir_name, expert_bytes, n_layer, n_expert);
        // 事前ロード: 回数の多い順に VRAM へ載せる。decode 中は譲る。
        let parts_pre = parts.clone();
        std::thread::spawn(move || {
            if let Some(line) =
                denno_engine::model::qwen4exp::preload_experts_from_stats(&parts_pre, &dir_name)
            {
                eprintln!("{line}");
            }
        });
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
                // task 149: リクエスト境界。prefill の一時領域 (arena) を返し、
                // 確保の内訳を 1 表だけ出す (1 本目と長い文脈の最初の 1 回)。
                // `DENNO_SCRATCH_FREE=0` / `DENNO_Q4_PROFILE` なしでは何もしない。
                let ctx_tokens = flash.cache.as_ref().map(|c| c.toks.len()).unwrap_or(0);
                denno_engine::mem_audit::request_boundary(ctx_tokens, 4096);
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
}
