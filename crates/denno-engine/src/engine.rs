//! エンジン本体: モデル+トークナイザ+KVキャッシュ+サンプラを束ね、
//! prefill → decode ループ(stop_sequences / EOS / max_tokens 処理)を実行する。

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::model::{load_model, Model};
use crate::prefix_cache::PrefixCache;
use crate::quant::QuantMode;
use denno_telemetry as telem;
use denno_telemetry::Frame;
use crate::sampler::Sampler;
use crate::tokenizer::DennoTokenizer;

/// prefix cache のトークン上限を指定する環境変数。
/// 未設定時は KvCache の max_len (コンテキスト長) が使われる。
pub const PREFIX_CACHE_ENV: &str = "DENNO_PREFIX_CACHE_MAX_TOKENS";

/// 直前のリクエストで prefix cache をどう使ったか (ログ/観測/テスト用)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixHit {
    /// ヒット無し (full prefill)
    None,
    /// 直前リクエストの live キャッシュを再利用 (truncate のみ)
    Live,
    /// ストア内の別セッションのエントリから KV を復元
    Entry,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// サンプリング/生成パラメータ
#[derive(Debug, Clone)]
pub struct GenParams {
    pub max_tokens: usize,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub stop_sequences: Vec<String>,
    pub seed: Option<u64>,
    /// A2.5: 反復ペナルティ (None = 無効・既定)。1.0 も無効と同一 (no-op)。
    /// 文脈 (prompt + 生成済み) の既出トークンの logit を HF と同じ規則で
    /// 抑える (詳細は `sampler::apply_repetition_penalty`)。RNG 不関与のため
    /// greedy は完全決定論・seed 固定時の出力系列も bit 再現可能。
    pub repetition_penalty: Option<f32>,
    /// s56 (T2): OpenAI 互換 frequency penalty (None / 0.0 = 無効・既定)。
    /// 文脈中の出現回数 × 係数を logit から減算する。repetition_penalty とは独立。
    pub frequency_penalty: Option<f32>,
    /// s56 (T2): OpenAI 互換 presence penalty (None / 0.0 = 無効・既定)。
    /// 文脈に 1 回以上現れたトークンから係数を 1 回減算する。
    pub presence_penalty: Option<f32>,
    /// s56 (T2): no_repeat_ngram (None / n<=1 = 無効・既定)。文脈に既出の
    /// n-gram の続きを -inf で禁止する (llama.cpp / HF 準拠)。
    pub no_repeat_ngram_size: Option<usize>,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: Some(1.0),
            top_p: None,
            top_k: None,
            stop_sequences: Vec::new(),
            seed: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            no_repeat_ngram_size: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
}

impl StopReason {
    pub fn as_str(&self) -> &'static str {
        match self {
            StopReason::EndTurn => "end_turn",
            StopReason::MaxTokens => "max_tokens",
            StopReason::StopSequence => "stop_sequence",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatOutput {
    pub text: String,
    pub stop_reason: StopReason,
    pub stop_sequence: Option<String>,
    pub input_tokens: usize,
    pub output_tokens: usize,
}

pub struct Engine {
    pub model: Box<dyn Model>,
    pub tok: DennoTokenizer,
    /// KV キャッシュのスロットプール (P6 バッチ並列 decode 用)。
    /// スロット 1 つ = 1 セッション分の KV。セッションは完了まで自身の
    /// スロットに常駐し、全スロットの decode を 1 forward にまとめる。
    /// スロット数は環境変数 `DENNO_BATCH_SLOTS` (デフォルト 4)。
    pub caches: Vec<KvCache>,
    /// slot ごとの「KV に入っているプロンプトトークン列」。
    /// cached_prompts[s][i] の KV は caches[s] の i 行目にある。
    /// (生成トークンの KV も cache には残るが cached_prompts には含めない)
    cached_prompts: Vec<Vec<u32>>,
    /// slot の使用中フラグ (install_session が取得、release_session が解放)
    slot_busy: Vec<bool>,
    /// slot ごとの KV 予約トークン数 (prompt + max_tokens。P17e の予算ハード保証用)
    slot_reserved: Vec<usize>,
    pub name: String,
    pub default_seed: Option<u64>,
    /// 過去会話の KV スナップショット群 (マルチエントリ LRU prefix cache)
    prefix_cache: PrefixCache,
    /// 直前のリクエストでの prefix cache ヒット状況
    last_prefix_hit: PrefixHit,
    /// 重みの量子化モード (テレメトリの ServerInfo 用に保持。数値経路には不変)
    pub quant: QuantMode,
}

/// KV キャッシュスロット数の上限 (環境変数 `DENNO_BATCH_SLOTS` で変更可)。
/// P17e から KV は需要ベース確保 (初期 0・書き込みで倍々成長) なので、
/// スロット数は「1 セッションあたりの目安予約 [`SESSION_RESERVE_TOKENS`]
/// トークン分が予算に収まる数」を既定とし、この上限で抑える。
pub const DEFAULT_BATCH_SLOTS: usize = 32;
pub const BATCH_SLOTS_ENV: &str = "DENNO_BATCH_SLOTS";

/// 1 セッションの目安予約トークン数 (P17e)。スロット数の既定計算と
/// install 時の KV 予算ハード保証に使う (prompt + max_tokens の合計が
/// この値なら、全スロットが同時に埋まっても KV は予算内に収まる)。
pub const SESSION_RESERVE_TOKENS: usize = 4096;

/// P15: KV メモリ予算の既定値 (MB) と環境変数。
/// バッチスロット + prefix cache の合計 KV の目安。巨大 KV モデル
/// (例: Qwen3-0.6B は kv_heads 8 × head_dim 128 × 28 層で 1 スロット
/// ~9.4GB) で既定構成の事前確保が物理メモリを超えないよう、予算から
/// スロット数と prefix cache トークン数を絞る。明示的な環境変数
/// (`DENNO_BATCH_SLOTS` / `DENNO_PREFIX_CACHE_TOKENS`) は予算より優先。
pub const KV_BUDGET_ENV: &str = "DENNO_KV_BUDGET_MB";

/// 1 リクエストのプロンプト長の上限 (token)。既定 12,288。
///
/// s124 (P4-6/P4-7): Q4 GPU 経路の作業領域は「プロンプト長 × 層数」で残るため
/// (thread_local の grow-only scratch)、27B/32GB では 1 万 token 超で VRAM の
/// 余裕が 3GB を切る。上限を超えたら cuMemAlloc 失敗 → CPU フォールバック
/// (秒/token) に落ちる前に、明示的な 400 で拒否する。P4-7 で解消したら上げる。
pub const MAX_PROMPT_ENV: &str = "DENNO_MAX_PROMPT_TOKENS";
pub const DEFAULT_MAX_PROMPT_TOKENS: usize = 32768;

fn max_prompt_tokens() -> usize {
    std::env::var(MAX_PROMPT_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_PROMPT_TOKENS)
}
pub const DEFAULT_KV_BUDGET_MB: usize = 12288;

fn kv_budget_bytes() -> usize {
    std::env::var(KV_BUDGET_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(DEFAULT_KV_BUDGET_MB)
        .saturating_mul(1024 * 1024)
}

/// スロット数と prefix cache トークン上限を決める (純粋関数: 単体テストあり)。
///
/// P17e: KV は需要ベース確保なので「スロット 1 個 = ctx 全幅の事前確保」
/// ではなく「1 セッション = [`SESSION_RESERVE_TOKENS`] トークンの予約」が
/// 予算単位になる。
///
/// - `explicit_slots` / `explicit_prefix`: 環境変数による明示指定 (最優先)
/// - 明示指定が無い場合: スロット数 = 予算内に目安予約が収まる数
///   ([`DEFAULT_BATCH_SLOTS`] 上限)。残り予算を prefix cache に充てる
///   (最低でも 2048 トークンは確保)
fn resolve_kv_budget(
    slot_max_len: usize,
    bytes_per_token: usize,
    explicit_slots: Option<usize>,
    explicit_prefix: Option<usize>,
    budget_bytes: usize,
) -> (usize, usize) {
    let bpt = bytes_per_token.max(1);
    let reserve_bytes = SESSION_RESERVE_TOKENS.min(slot_max_len).saturating_mul(bpt);
    let slots = match explicit_slots {
        Some(n) => n.max(1),
        None => DEFAULT_BATCH_SLOTS.min((budget_bytes / reserve_bytes.max(1)).max(1)),
    };
    let reserved = slots.saturating_mul(reserve_bytes);
    let prefix = match explicit_prefix {
        Some(n) => n,
        None => {
            let floor = 2048.min(slot_max_len);
            if reserved >= budget_bytes {
                floor
            } else {
                let rest = budget_bytes - reserved;
                slot_max_len.min(rest / bpt).max(floor)
            }
        }
    };
    (slots, prefix)
}

/// P4 ステッパ: 1リクエスト分の生成状態。
///
/// P6 からは各セッションが KV キャッシュスロットを 1 つ占有して完了まで
/// 常駐する (`Engine::install_session` でスロット取得+prefix cache 経路選択、
/// `Engine::release_session` でスロット解放+prefix store へ退避)。
/// 複数セッションの decode は [`Engine::step_chat_batch`] で 1 forward に
/// まとめて進める。
pub struct ChatSession {
    prompt_ids: Vec<u32>,
    /// prefill で再利用した prefix 長 (KV に既にある先頭トークン数)
    lcp: usize,
    prefill_done: bool,
    /// 直近の forward の logits (最終位置)
    logits: Vec<f32>,
    generated: Vec<u32>,
    text: String,
    /// s53: ストリームへ送出済みのバイト数 (sent_len)。
    /// 不変条件: `text[..sent_len]` = 送出済み、`text[sent_len..]` = holdback 中
    /// (未送出)。holdback は「ロールラベル停止マーカーの prefix になり得る」
    /// 末尾のみで、生成終了時 (flush_stream 後) は必ず `sent_len == text.len()`
    /// になる (通常応答の最終テキスト不変)。
    sent_len: usize,
    max_tokens: usize,
    stop_sequences: Vec<String>,
    sampler: Sampler,
    stop_reason: StopReason,
    stop_sequence: Option<String>,
    /// 占有中の KV キャッシュスロット (None = 未 install / release 済み)
    slot: Option<usize>,
    input_tokens: usize,
    /// A2.5/s56: 反復抑止 (repetition_penalty + frequency/presence/no_repeat_ngram)。
    /// すべて無効なら `is_noop()` が真で、適用自体を省く (既定は挙動不変)。
    anti_rep: crate::sampler::AntiRepetition,
    /// A5/s46: 末尾 generation prompt のトークン境界 (prompt_ids[..境界] が
    /// 「base」= 次ターンにも現れる履歴部分)。テンプレート描画から検出できた
    /// 場合のみ Some (非 recurrent モデル・fallback テンプレート・base が
    /// トークン接頭辞にならなかった場合は None = 従来経路)。
    gen_prompt_start: Option<usize>,
    /// 第21: tool_choice 拘束デコード (None = 無効・無マスク経路とビット同一)。
    /// Some のとき decode 毎に sample 直前に logits マスクを適用し、
    /// サンプル後に状態機械を前進させる (toolmask.rs)。
    tool_mask: Option<crate::toolmask::ToolMask>,
    /// テレメトリ用のセッション ID (採番順。数値経路には無関係)
    pub id: u32,
}

impl ChatSession {
    /// プロンプトのトークン数 (SSE message_start の usage 用)
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
    }

    /// prefill 済みか (decode 待ちか)。スケジューラが prefill/decode の
    /// 振り分けに使う。
    pub fn prefill_done(&self) -> bool {
        self.prefill_done
    }

    /// 生成結果を取り出す (step_chat が true を返した後)
    pub fn into_output(self) -> ChatOutput {
        ChatOutput {
            text: self.text,
            stop_reason: self.stop_reason,
            stop_sequence: self.stop_sequence,
            input_tokens: self.input_tokens,
            output_tokens: self.generated.len(),
        }
    }
}

impl Engine {
    /// モデルディレクトリ(HF形式: config.json + model.safetensors + tokenizer.json
    /// + tokenizer_config.json)からロードする。
    ///
    /// `quant` は重みの量子化モード (P3):
    /// - `QuantMode::Fp32` (デフォルト): 従来動作。HANDOFF §5 の exactness 基準は
    ///   このモードのもの。
    /// - `QuantMode::Int8`: 線形層の重みをロード時に per-channel scale の int8 へ
    ///   量子化する。**同一 seed + 温度 0 → 同一出力の決定論は維持されるが、
    ///   fp32 モードと同一の出力にはならない** (丸め誤差)。CLI でのみ有効化
    ///   (`denno serve --quant int8`)。
    pub fn load(
        dir: &Path,
        name: impl Into<String>,
        default_seed: Option<u64>,
        quant: QuantMode,
    ) -> Result<Self> {
        let model = load_model(dir, quant).context("モデルのロードに失敗")?;
        let tok = DennoTokenizer::load(dir).context("トークナイザのロードに失敗")?;
        let slot0 = model.new_cache();
        // マルチエントリ prefix cache のトークン上限 (P15: KV メモリ予算から
        // スロット数と一緒に決める。明示 env は予算より優先)。
        let explicit_slots = std::env::var(BATCH_SLOTS_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1);
        let explicit_prefix = std::env::var(PREFIX_CACHE_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        let (slots, max_tokens) = resolve_kv_budget(
            slot0.max_len(),
            slot0.bytes_per_token(),
            explicit_slots,
            explicit_prefix,
            kv_budget_bytes(),
        );
        tracing::info!(
            "prefix cache: multi-entry LRU, budget = {} tokens (~{:.0} MB KV)",
            max_tokens,
            max_tokens as f64 * slot0.bytes_per_token() as f64 / 1e6
        );
        tracing::info!(
            "batch slots: {} (KV 予約 ~{:.0} MB/slot @{} tok, env {BATCH_SLOTS_ENV}; total KV budget {} MB, env {KV_BUDGET_ENV})",
            slots,
            SESSION_RESERVE_TOKENS.min(slot0.max_len()) as f64 * slot0.bytes_per_token() as f64 / 1e6,
            SESSION_RESERVE_TOKENS,
            kv_budget_bytes() / (1024 * 1024)
        );
        tracing::info!(
            "quant mode: {}{}",
            quant.label(),
            if quant == QuantMode::Int8 {
                " (決定論は維持されるが fp32 と同一出力ではない)"
            } else {
                ""
            }
        );
        let caches: Vec<KvCache> = (0..slots).map(|_| model.new_cache()).collect();
        let cached_prompts = vec![Vec::new(); slots];
        let slot_busy = vec![false; slots];
        let slot_reserved = vec![0usize; slots];
        // A4: GDN 状態スナップショットをトークン換算するためのスケール
        let kv_bytes_per_token = caches[0].bytes_per_token();
        // P58-5: 起動直後の重みメモリ内訳を 1 行 (ヒープ実体 vs mmap スライス)。
        tracing::info!("{}", crate::quant::weight_memory_line("startup"));
        Ok(Self {
            model,
            tok,
            caches,
            cached_prompts,
            slot_busy,
            slot_reserved,
            name: name.into(),
            default_seed,
            prefix_cache: PrefixCache::new(max_tokens, kv_bytes_per_token),
            last_prefix_hit: PrefixHit::None,
            quant,
        })
    }

    pub fn context(&self) -> usize {
        self.model.context()
    }

    /// テレメトリ: KV スロット使用状況フレームを作る (enabled 時のみ呼ばれる)。
    fn kv_usage_frame(&self) -> Frame {
        Frame::KvUsage {
            busy: self.slot_busy.clone(),
            t_lens: self.caches.iter().map(|c| c.len() as u32).collect(),
        }
    }

    /// テレメトリ: prof フックで累積した op 時間を OpSummary 1 フレームにして流す。
    fn flush_ops_frame(&self) {
        let ops = telem::take_ops();
        if !ops.is_empty() {
            telem::emit(Frame::OpSummary { ops });
        }
    }

    /// テスト用 (test-utils feature): モデルファイル無しで Engine を直接構築する。
    /// ダミーモデル + ダミートークナイザによるワイヤテスト (denno-server) で使う。
    #[cfg(any(test, feature = "test-utils"))]
    pub fn from_parts(
        model: Box<dyn Model>,
        tok: DennoTokenizer,
        name: impl Into<String>,
        default_seed: Option<u64>,
    ) -> Self {
        let slot0 = model.new_cache();
        let max_tokens = slot0.max_len();
        let slots = {
            let explicit_slots = std::env::var(BATCH_SLOTS_ENV)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .filter(|&n| n >= 1);
            let explicit_prefix = std::env::var(PREFIX_CACHE_ENV)
                .ok()
                .and_then(|s| s.parse::<usize>().ok());
            let (slots, _prefix_tokens) = resolve_kv_budget(
                slot0.max_len(),
                slot0.bytes_per_token(),
                explicit_slots,
                explicit_prefix,
                kv_budget_bytes(),
            );
            slots
        };
        let caches: Vec<KvCache> = (0..slots).map(|_| model.new_cache()).collect();
        let cached_prompts = vec![Vec::new(); slots];
        let slot_busy = vec![false; slots];
        let slot_reserved = vec![0usize; slots];
        // A4: GDN 状態スナップショットをトークン換算するためのスケール
        let kv_bytes_per_token = caches[0].bytes_per_token();
        Self {
            model,
            tok,
            caches,
            cached_prompts,
            slot_busy,
            slot_reserved,
            name: name.into(),
            default_seed,
            prefix_cache: PrefixCache::new(max_tokens, kv_bytes_per_token),
            last_prefix_hit: PrefixHit::None,
            quant: QuantMode::Fp32,
        }
    }

    /// マルチエントリ prefix cache のトークン上限を変更する (env の代わりに引数で指定する用)。
    /// 現在保持中のエントリが新上限を超える分は LRU 順に eviction される。
    pub fn set_prefix_cache_max_tokens(&mut self, max_tokens: usize) {
        self.prefix_cache.set_max_tokens(max_tokens);
    }

    pub fn prefix_cache_max_tokens(&self) -> usize {
        self.prefix_cache.max_tokens
    }

    /// 現在保持している prefix cache エントリ数
    pub fn prefix_cache_entries(&self) -> usize {
        self.prefix_cache.len()
    }

    /// 直前のリクエストでの prefix cache ヒット状況
    pub fn last_prefix_hit(&self) -> PrefixHit {
        self.last_prefix_hit
    }

    /// P6 ステッパ: セッションを KV キャッシュスロットに常駐させる。
    ///
    /// - 空きスロットの中で「退避済み KV との共通接頭辞 (live 再利用候補)」が
    ///   最長のスロットを選ぶ (旧単一キャッシュの live 再利用を再現)
    /// - 旧 `chat()` と同一の prefix cache 経路選択 (live LCP vs マルチエントリ
    ///   ストアの LCP、LCP が最大の候補を採用) で `sess.lcp` に再利用長を設定
    /// - スロットが全て使用中ならエラー (スケジューラは同時実行数を
    ///   スロット数以下に制限するため通常起こらない)
    ///
    /// P17e: install 時に `prompt + max_tokens` を KV 予約として登録し、
    /// 2 つのハード保証を行う。
    /// 1. 予約合計がスロットの `max_len` を超えない (KV 書き込みが
    ///    物理的に入る保証。超過は 400 相当のエラーで拒否)
    /// 2. 全 busy スロットの予約合計 × bytes_per_token が KV 予算
    ///    (`DENNO_KV_BUDGET_MB`) を超えない
    pub fn install_session(&mut self, sess: &mut ChatSession) -> Result<()> {
        if sess.slot.is_some() {
            return Ok(()); // 既に常駐済み
        }
        // --- s124: プロンプト長の上限 (作業領域の n 比例分を守る) ---
        let plen = sess.prompt_ids.len();
        let cap = max_prompt_tokens();
        if plen > cap {
            anyhow::bail!(
                "invalid_request_error: prompt ({} tokens) が上限 {} を超えます。{} を上げるか入力を短くしてください",
                plen,
                cap,
                MAX_PROMPT_ENV
            );
        }
        // --- P17e: 予約トークン数 (prompt + max_tokens) のハード保証 ---
        let need = sess.prompt_ids.len() + sess.max_tokens;
        let budget = self
            .caches
            .first()
            .map(|c| (c.max_len(), c.bytes_per_token()))
            .unwrap_or((0, 0));
        if need > budget.0 {
            anyhow::bail!(
                "invalid_request_error: prompt ({} tokens) + max_tokens ({}) が KV スロットの上限 {} を超えます",
                sess.prompt_ids.len(),
                sess.max_tokens,
                budget.0
            );
        }
        let reserved_total: usize = self.slot_reserved.iter().sum();
        let budget_bytes = kv_budget_bytes();
        if (reserved_total + need).saturating_mul(budget.1) > budget_bytes {
            anyhow::bail!(
                "invalid_request_error: KV 予算超過 (予約 {} + {} tokens × {} B/token > {} MB)。同時リクエストを減らすか {} を上げてください",
                reserved_total,
                need,
                budget.1,
                budget_bytes / (1024 * 1024),
                KV_BUDGET_ENV
            );
        }
        // --- 空きスロットを選ぶ (live 再利用候補が最長のもの) ---
        let cap = sess.prompt_ids.len().saturating_sub(1);
        // logits は forward した最終位置からしか得られないので、
        // 少なくとも最後の 1 トークンは必ず forward する
        let mut best_slot = None;
        let mut best_live_lcp = 0usize;
        for s in 0..self.caches.len() {
            if self.slot_busy[s] {
                continue;
            }
            let lcp = sess
                .prompt_ids
                .iter()
                .zip(self.cached_prompts[s].iter())
                .take_while(|(a, b)| a == b)
                .count()
                .min(cap)
                .min(self.caches[s].len());
            if best_slot.is_none() || lcp > best_live_lcp {
                best_slot = Some(s);
                best_live_lcp = lcp;
            }
        }
        let slot = best_slot
            .ok_or_else(|| anyhow::anyhow!("no free KV cache slot ({} slots)", self.caches.len()))?;
        self.slot_busy[slot] = true;
        self.slot_reserved[slot] = need;
        sess.slot = Some(slot);
        if telem::enabled() {
            telem::emit(Frame::SessionAdmit {
                session: sess.id,
                slot: slot as u32,
                prompt_tokens: sess.prompt_ids.len() as u32,
            });
        }

        // --- prefix cache 経路選択 (旧 chat() と同一) ---
        let lcp;
        let hit;
        // A4: 再帰状態を持つモデル (Qwen3.5 GDN) も「GDN 状態スナップショット付き
        // エントリ」からの復元で KV + 状態をペア復元できる。状態はエントリの
        // prompt 境界 (prefill 直後) でしか取得していないため、新しい prompt が
        // エントリ prompt 全長を接頭辞として含む場合 (best_full_match) に限り有効。
        // live 再利用は従来どおり純拡張 (lcp == 現在長) のみで、truncate は不可。
        let recurrent = self.model.is_recurrent();
        let best_entry = if recurrent {
            self.prefix_cache
                .best_full_match(&sess.prompt_ids)
                .filter(|&(_, l)| l <= cap)
        } else {
            self.prefix_cache
                .best_match(&sess.prompt_ids)
                .map(|(i, l)| (i, l.min(cap)))
        };
        // recurrent では live truncate が不可能なため、live が純拡張で使える
        // (コピー無しで最良) 場合を除きエントリ復元を優先する。
        // 非 recurrent は従来どおり「エントリが厳に長い」場合のみエントリを選ぶ。
        let prefer_entry = recurrent && best_live_lcp != self.caches[slot].len();
        if let Some((idx, entry_lcp)) =
            best_entry.filter(|&(_, l)| l > best_live_lcp || prefer_entry)
        {
            // 別セッションのエントリから KV (+ GDN 状態) を復元
            let restored = self
                .prefix_cache
                .restore_into(idx, entry_lcp, &mut self.caches[slot]);
            if restored {
                lcp = entry_lcp;
                hit = PrefixHit::Entry;
            } else {
                // 状態を持てない位置での復元要求など → フル prefill にフォールバック
                self.caches[slot].reset();
                self.cached_prompts[slot].clear();
                lcp = 0;
                hit = PrefixHit::None;
            }
        } else if best_live_lcp > 0 && (!recurrent || best_live_lcp == self.caches[slot].len()) {
            // スロット残留 KV の live 再利用 (truncate のみ、コピー無し)。
            // recurrent モデルでは純拡張 (lcp == len・truncate 不要) のみ到達。
            if best_live_lcp < self.caches[slot].len() {
                self.caches[slot].truncate(best_live_lcp);
            }
            lcp = best_live_lcp;
            hit = PrefixHit::Live;
        } else {
            // 共通接頭辞なし: フル prefill
            self.caches[slot].reset();
            self.cached_prompts[slot].clear();
            lcp = 0;
            hit = PrefixHit::None;
        }
        if lcp > 0 {
            tracing::info!(
                "prefix cache hit: reuse {} / {} tokens (recompute {})",
                lcp,
                sess.prompt_ids.len(),
                sess.prompt_ids.len() - lcp
            );
        }
        sess.lcp = lcp;
        self.last_prefix_hit = hit;
        Ok(())
    }

    /// P6 ステッパ: 完了 (またはエラー) したセッションのスロットを解放する。
    /// KV スナップショットの prefix store への退避は **prefill 直後**
    /// (`step_chat` の prefill 分岐・A4) に行われる。GDN 状態はシーケンス全体の
    /// 関数なので、KV と状態を同じ位置 (prompt 境界) でペア保存する必要があり、
    /// 生成トークンで状態が進んだ後の release 時点では退避できないため。
    ///
    /// スロットの KV 行データと cached_prompts[slot] は**そのまま残す**。
    /// 次にこのスロットを取得したセッションが同じ接頭辞を持てば
    /// install_session の live 再利用経路 (truncate のみ、コピー無し) で
    /// 使える — 旧単一キャッシュ構成の live 再利用を再現する。
    pub fn release_session(&mut self, sess: &mut ChatSession) {
        if telem::enabled() {
            let reason = match sess.stop_reason {
                StopReason::EndTurn => telem::done_reason::END_TURN,
                StopReason::MaxTokens => telem::done_reason::MAX_TOKENS,
                StopReason::StopSequence => telem::done_reason::STOP_SEQUENCE,
            };
            telem::emit(Frame::SessionDone {
                session: sess.id,
                reason,
                output_tokens: sess.generated.len() as u32,
            });
        }
        let Some(slot) = sess.slot.take() else { return };
        self.slot_busy[slot] = false;
        self.slot_reserved[slot] = 0;
    }

    /// s146 (B8-3): クライアント向けエラー分類のために KV スロット数を読む
    /// (読み取り専用。数値経路には触れない)。
    pub fn slot_count(&self) -> usize {
        self.caches.len()
    }

    /// s146 (B8-3): 実行中 (busy) のスロット数。分類メッセージ用の読み取りのみ。
    pub fn slot_busy_count(&self) -> usize {
        self.slot_busy.iter().filter(|b| **b).count()
    }

    /// P4 ステッパ: リクエスト状態を準備する (トークン化 + max_tokens クランプ +
    /// サンプラ構築)。KV キャッシュには触れない。
    /// 以後 `install_session` → `step_chat` (true が返るまで) の順に進める。
    pub fn begin_chat(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        params: &GenParams,
    ) -> Result<ChatSession> {
        self.begin_chat_with_tools(system, messages, params, None, None, None)
    }

    /// tools 付きの begin_chat。`tools` は chat template の `tools` 変数へ
    /// そのまま渡される (None = tools 無し。従来経路と完全同一)。
    /// `enable_thinking` (s47) は chat template の `enable_thinking` 変数として
    /// 渡すか否か (None = 変数無し = 従来経路と同一。Some(false) = thinking off。
    /// 描画に使うテンプレートが無い場合は反映されない)。
    /// `constraint` (第21) は tool_choice any/Named の拘束デコード。
    /// None なら無マスク (auto / tools 無し経路とビット同一)。
    pub fn begin_chat_with_tools(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        params: &GenParams,
        tools: Option<&serde_json::Value>,
        constraint: Option<&crate::toolmask::ToolConstraint>,
        enable_thinking: Option<bool>,
    ) -> Result<ChatSession> {
        let prompt = self.tok.apply_chat_template(system, messages, tools, enable_thinking)?;
        let prompt_ids = self.tok.encode(&prompt)?;

        // A5/s46: 末尾 generation prompt のトークン境界を検出する (recurrent
        // モデルのみ・prefix cache の base エントリ用)。
        // chat template は毎回 prompt を generation prompt
        // (Qwen3.5 なら `<|im_start|>assistant\n\n\n`) で終えるが、次ターンの
        // 描画ではこの部分が assistant メッセージ本文に置き換わるため、
        // **prompt 全体は次ターンのトークン接頭辞にならない** (実 9B で確認:
        // turn2 の prompt は turn1 の末尾 2 トークン `<think>\n` で分岐)。
        // これが prefix cache が 1 度もヒットしない根因。
        // 「generation prompt 直前まで」(base) は次ターンにもそのまま現れるため、
        // base 境界を記録して prefill を base + generation prompt の 2 forward に
        // 分割し、base 境界で KV+GDN スナップショットを退避できるようにする
        // (step_chat の split 経路参照)。
        let gen_prompt_start = if self.model.is_recurrent() {
            self.tok
                .apply_chat_template_base(system, messages, tools, enable_thinking)
                .ok()
                .and_then(|base| self.tok.encode(&base).ok())
                .filter(|base_ids| {
                    !base_ids.is_empty()
                        && base_ids.len() < prompt_ids.len()
                        && prompt_ids.starts_with(base_ids.as_slice())
                })
                .map(|base_ids| base_ids.len())
        } else {
            None
        };

        let max_tokens = params.max_tokens;
        // Hermes 等のクライアントは context_length から大きな max_tokens (例: 65535) を
        // 送ってくることがある。Anthropic API なら 400 を返す場面だが、このサーバーは
        // 学習用ツールなので「コンテキストに収まる範囲へクランプ」して続行する。
        let max_tokens = if prompt_ids.len() + max_tokens > self.model.context() {
            let clamped = self.model.context().saturating_sub(prompt_ids.len());
            tracing::warn!(
                "max_tokens {} clamped to {} to fit context {} (prompt {} tokens)",
                max_tokens,
                clamped,
                self.model.context(),
                prompt_ids.len()
            );
            clamped
        } else {
            max_tokens
        };
        if prompt_ids.len() + max_tokens > self.model.context() {
            bail!(
                "invalid_request_error: prompt ({} tokens) + max_tokens ({}) がモデルのコンテキスト長 {} を超えます",
                prompt_ids.len(),
                max_tokens,
                self.model.context()
            );
        }

        let seed = params.seed.or(self.default_seed);
        let sampler = Sampler::new(params.temperature, params.top_k, params.top_p, seed);
        // 第21: 拘束デコード (tool_choice any/Named)。names が実質空なら無効
        let tool_mask = constraint
            .filter(|c| !c.names.is_empty())
            .and_then(|c| crate::toolmask::ToolMask::new(c.names.clone()));

        Ok(ChatSession {
            input_tokens: prompt_ids.len(),
            prompt_ids,
            lcp: 0,
            prefill_done: false,
            logits: Vec::new(),
            generated: Vec::new(),
            text: String::new(),
            sent_len: 0,
            max_tokens,
            stop_sequences: params.stop_sequences.clone(),
            sampler,
            stop_reason: StopReason::MaxTokens,
            stop_sequence: None,
            slot: None,
            anti_rep: crate::sampler::AntiRepetition {
                repetition_penalty: params.repetition_penalty,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
                no_repeat_ngram_size: params.no_repeat_ngram_size,
            }
            .normalized(),
            gen_prompt_start,
            tool_mask,
            id: telem::next_session_id(),
        })
    }

    /// A2.5/s56: 反復抑止をセッションの logits に適用する (サンプル直前)。
    /// 適用対象は prompt_ids + generated。順序は
    /// repetition_penalty → frequency → presence → no_repeat_ngram で固定
    /// (`sampler::AntiRepetition::apply`)。4 つすべて無効なら logits に触らない
    /// = 従来経路とビット一致。逐次・バッチ両 decode がこの関数だけを通る。
    fn apply_session_penalty(sess: &mut ChatSession) {
        if sess.anti_rep.is_noop() {
            return;
        }
        let mut ctx = Vec::with_capacity(sess.prompt_ids.len() + sess.generated.len());
        ctx.extend_from_slice(&sess.prompt_ids);
        ctx.extend_from_slice(&sess.generated);
        sess.anti_rep.apply(&mut sess.logits, &ctx);
    }

    /// P4 ステッパ: 生成を 1 forward 進める (prefill 全体 or decode 1 トークン)。
    /// 完了 (EOS / stop sequence / max_tokens) で true を返す。
    /// 呼び出し前にセッションが resident (`install_session` 済み) であること。
    /// (P6: 複数セッションの decode をまとめる場合は `step_chat_batch` を使う)
    pub fn step_chat(
        &mut self,
        sess: &mut ChatSession,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<bool> {
        // s148 (Task B10・計測のみ): scheduler は prefill を `step_chat` で、decode を
        // `step_chat_batch` で 1 歩ずつ進める。片方だけに `sl_step` を置くと
        // 「steps = サンプル」が片経路で崩れるので、**両方**の入口で 1 サンプル積む
        // (計測 OFF は `step_begin` が `None` = クロック読みなし)。
        let st = crate::prof::step_begin();
        let r = self.step_chat_inner(sess, on_delta);
        crate::prof::step_end(st);
        if matches!(&r, Ok(true)) {
            Self::log_weights_after_first_request();
        }
        r
    }

    /// P58-5: 最初のリクエストが完了した時点で重みメモリ内訳を 1 回だけ出す
    /// (起動直後の行と同じ体裁)。1 リクエストで増えた重み関連のヒープ確保が
    /// 無いこと (= 推論が重みを実体化し直していないこと) をログで確認できる。
    fn log_weights_after_first_request() {
        static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        tracing::info!("{}", crate::quant::weight_memory_line("after 1 request"));
    }

    /// s148: [`step_chat`] の本体 (表 C の step 計測を外したもの)。
    fn step_chat_inner(
        &mut self,
        sess: &mut ChatSession,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<bool> {
        if !sess.prefill_done {
            // prefill: 旧 chat() と同一形状 (残りプロンプトを 1 回の forward で処理)。
            // A5/s46: recurrent モデルで generation prompt 境界が検出できた場合のみ、
            // prefill を [base 部分][generation prompt] の 2 forward に分割する。
            // forward(a) + forward(b) は forward(a++b) と bit 一致 (GEMM は行独立の
            // i32 累積・GDN/attention はトークン逐次のため分割で蓄積順が変わらない。
            // 契約テスト: qwen35::qwen38_27b_shape_split_prefill_bit_exact)。
            // 分割により base 境界 (= 次ターンの prompt にそのまま現れる履歴部分の
            // 末尾) で KV+GDN スナップショットを退避でき、次ターン以降の
            // best_full_match → ペア復元が成立する。境界不明 (fallback テンプレート
            // 等) や非 recurrent は従来どおり単一 forward。
            let slot = sess.slot.expect("install_session 済みのセッション");
            let t0 = telem::enabled().then(std::time::Instant::now);
            let split_base = match sess.gen_prompt_start {
                Some(b)
                    if self.model.is_recurrent()
                        && b > 0
                        && b >= sess.lcp
                        && b < sess.prompt_ids.len() =>
                {
                    Some(b)
                }
                _ => None,
            };
            let logits = if let Some(b) = split_base {
                // (1) base 部分: 復元済み lcp から base 境界まで (lcp == b なら skip。
                //     エントリ復元で KV+GDN 状態が既に base 境界にあるため)
                if b > sess.lcp {
                    let _ = self
                        .model
                        .forward(&sess.prompt_ids[sess.lcp..b], &mut self.caches[slot])?;
                }
                // (2) base 境界でスナップショット退避 (A4 と同一の KV+GDN ペア保存。
                //     kv_len = cache.len() == b: (1) 完了直後 or lcp == b の復元直後)
                debug_assert_eq!(self.caches[slot].len(), b);
                {
                    let kv_len = self.caches[slot].len();
                    let gdn = self.model.is_recurrent().then(|| {
                        self.caches[slot].flush_gdn_device_state();
                        self.caches[slot].gdn_snapshot()
                    });
                    let (k, v) = self.caches[slot].snapshot(kv_len);
                    self.prefix_cache.insert_snapshot(
                        sess.prompt_ids[..b].to_vec(),
                        k,
                        v,
                        kv_len,
                        gdn,
                    );
                }
                // (3) generation prompt 部分: base 境界から prompt 末尾まで。
                //     logits は最終位置の forward から得る
                self.model
                    .forward(&sess.prompt_ids[b..], &mut self.caches[slot])?
            } else {
                self.model
                    .forward(&sess.prompt_ids[sess.lcp..], &mut self.caches[slot])?
            };
            if let Some(t0) = t0 {
                telem::emit(Frame::Prefill {
                    session: sess.id,
                    prompt_tokens: (sess.prompt_ids.len() - sess.lcp) as u32,
                    ns: t0.elapsed().as_nanos() as u64,
                    tokens: sess.prompt_ids[sess.lcp..].to_vec(),
                });
                telem::emit(self.kv_usage_frame());
                self.flush_ops_frame();
            }
            self.cached_prompts[slot] = sess.prompt_ids.clone();
            // A4: prefill 直後のスナップショットを prefix store へ退避。
            // split 経路では上で base 境界に退避済み (prompt 全体は次ターンの
            // トークン接頭辞にならないため、全体のエントリを作ってもヒットせず
            // 退避しない)。GDN 状態はシーケンス全体の関数なので、KV と状態を
            // 同じ位置 (cache.len() == prompt 長) でペア保存する。decode 中は
            // 状態位置が進むため退避しない (release 時点の状態は prompt 境界の
            // ものではない)。KV 行は prompt ちょうどの長さで保存する (restore は
            // n <= prompt.len に制限されるため生成分の行を保存しても復元に使えない)。
            if split_base.is_none() {
                debug_assert_eq!(self.caches[slot].len(), sess.prompt_ids.len());
                {
                    let kv_len = self.caches[slot].len();
                    let gdn = self.model.is_recurrent().then(|| {
                        self.caches[slot].flush_gdn_device_state();
                        self.caches[slot].gdn_snapshot()
                    });
                    let (k, v) = self.caches[slot].snapshot(kv_len);
                    self.prefix_cache
                        .insert_snapshot(sess.prompt_ids.clone(), k, v, kv_len, gdn);
                }
            }
            debug_assert_eq!(self.caches[slot].len(), sess.prompt_ids.len());
            sess.logits = logits;
            sess.prefill_done = true;
            return Ok(false);
        }

        if sess.generated.len() >= sess.max_tokens {
            // max_tokens = 0 (prefill のみで確定) の場合
            sess.stop_reason = StopReason::MaxTokens;
            return Ok(true);
        }

        // decode 1 ステップ (旧 chat() のループ 1 回分と同一)
        // A2.5: penalty 指定時のみ既出トークンを抑制 (無効時はビット不変)
        Self::apply_session_penalty(sess);
        // 第21: tool_choice 拘束 (Some のとき sample 直前に許容集合マスク)。
        // 拘束完了後は no-op = 無マスク経路とビット同一
        if sess.tool_mask.is_some() {
            let eos = self.model.eos_ids().to_vec();
            sess.tool_mask
                .as_mut()
                .unwrap()
                .apply(&mut sess.logits, &self.tok, &eos);
        }
        // s148: 表 C の `sl_step_sampler` (penalty/拘束マスク適用は含めない — 拘束
        // マスクは logits 側の処理なのでここでは sample の呼び出しだけを測る)。
        let samp_t = crate::prof::span_begin();
        let next =
            crate::prof::time(crate::prof::SAMPLER, || sess.sampler.sample(&sess.logits)) as u32;
        crate::prof::span_end(samp_t, crate::prof::SL_STEP_SAMPLER);
        sess.generated.push(next);
        // 第21: 拘束状態の前進 (サンプル済みトークンの piece で状態遷移)
        if let Some(mask) = sess.tool_mask.as_mut() {
            mask.feed(&self.tok, next);
        }
        if telem::enabled() {
            telem::emit(Frame::Token { session: sess.id, token: next, pos: sess.generated.len() as u32 });
            let tk = telem::topk_softmax(&sess.logits, 8);
            telem::emit(Frame::TopK {
                session: sess.id,
                chosen: next,
                entropy_f16: telem::f32_to_f16_bits(tk.entropy),
                entries: tk
                    .entries
                    .iter()
                    .map(|&(id, p)| (id, telem::f32_to_f16_bits(p)))
                    .collect(),
            });
        }

        if self.model.eos_ids().contains(&next) {
            sess.stop_reason = StopReason::EndTurn;
            self.flush_stream(sess, 1, on_delta)?;
            return Ok(true);
        }

        if self.emit_delta_and_check_stop(sess, on_delta)? {
            return Ok(true);
        }

        if sess.generated.len() < sess.max_tokens {
            let slot = sess.slot.expect("install_session 済みのセッション");
            let logits = self.model.forward(&[next], &mut self.caches[slot])?;
            if telem::enabled() {
                telem::emit(self.kv_usage_frame());
                self.flush_ops_frame();
            }
            sess.logits = logits;
            return Ok(false);
        }
        sess.stop_reason = StopReason::MaxTokens;
        self.flush_stream(sess, 0, on_delta)?;
        Ok(true)
    }

    /// 生成終了時のフラッシュ: decode の保留分 (末尾がマルチバイト文字の
    /// 途中で終わるトークン) と s53 の holdback (`sess.text[sess.sent_len..]`)
    /// をまとめて確定させて流す。
    /// EOS / max_tokens / stop sequence / ロールラベル停止で打ち切られた
    /// ステップでは通常の emit が行われない (または保留が起きる) ため、
    /// 完了時に必ず呼ぶ。`exclude_last` は decode から除外する末尾トークン数
    /// (EOS 分 = 1)。
    ///
    /// 不変条件: 送出すべきは `text[sent_len..]` のみ (既送出分は再送しない)。
    /// 呼び出し後 `sent_len == text.len()`。
    fn flush_stream(
        &self,
        sess: &mut ChatSession,
        exclude_last: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let n = sess.generated.len().saturating_sub(exclude_last);
        let full = match self.tok.decode(&sess.generated[..n]) {
            Ok(t) => t, // lossy (不正バイトは U+FFFD)
            // decode 失敗時も holdback は必ず解放する (送出は確定済み分のみ)
            Err(e) => {
                if sess.sent_len < sess.text.len() {
                    let start = sess.sent_len;
                    sess.sent_len = sess.text.len();
                    on_delta(&sess.text[start..]);
                }
                return Err(e);
            }
        };
        // 既に確定済みの text (holdback 分も含む) に続きがあれば text へ取り込み
        let confirmed = sess.sent_len;
        if full.len() > confirmed && full.is_char_boundary(confirmed) {
            sess.text.truncate(confirmed);
            sess.text.push_str(&full[confirmed..]);
        }
        if sess.sent_len < sess.text.len() {
            let start = sess.sent_len;
            sess.sent_len = sess.text.len();
            on_delta(&sess.text[start..]);
        }
        Ok(())
    }

    /// decode 1 ステップの共通後処理: 差分ストリーミング (s53 holdback 付き) +
    /// stop_sequences / ロールラベル停止のチェック。
    /// 打ち切るときは sess の stop_reason/stop_sequence を設定し true を返す。
    /// step_chat / step_chat_batch の両方から使う。
    fn emit_delta_and_check_stop(
        &self,
        sess: &mut ChatSession,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<bool> {
        // 差分だけストリーミング(部分的なUTF-8を避けるため全decode→diff)。
        // バイトレベルBPE (GPT-2 等) ではマルチバイト文字がトークン境界で
        // 分割され、未完成バイト列は decode が U+FFFD に置き換える。その
        // U+FFFD を確定済みテキスト (sess.text) に混入させると以後のバイト
        // 差分がずれて文字が欠落するため、**末尾が U+FFFD になるトークンは
        // 保留**し、文字が完成したステップでまとめて差分として流す。
        // s148: 表 C の `sl_step_detok` (差分テキスト生成 = detok の全区間)。
        let detok_t = crate::prof::span_begin();
        let full = crate::prof::time(crate::prof::DETOK, || -> Result<String> {
            let mut n = sess.generated.len();
            let mut full = self.tok.decode(&sess.generated[..n])?;
            while n > 0 && full.ends_with('\u{FFFD}') {
                n -= 1;
                full = self.tok.decode(&sess.generated[..n])?;
            }
            Ok(full)
        })?;
        crate::prof::span_end(detok_t, crate::prof::SL_STEP_DETOK);
        // 確定済み分 (送出于定の範囲) に新しい差分を追記する。
        // 不変条件: sent_len は追記後に変えない (= 送出済み境界は固定)。
        let confirmed = sess.sent_len;
        if full.len() > confirmed && full.is_char_boundary(confirmed) {
            sess.text.truncate(confirmed);
            sess.text.push_str(&full[confirmed..]);
        }
        // s53: 生成テキストにロールラベル (チャットテンプレート由来の
        // `user` / `assistant` / `system` 行・特殊トークン) が現れたら、
        // マーカー自体を出力せず打ち切る。判定はトークンではなくテキスト列に
        // 対して行う (決定的)。判定対象は未送出部分 (sent_len 以降)。
        // マーカー長は最長 12 バイト程度なので、末尾の「マーカーの prefix に
        // なり得る部分」は holdback で保持し、確定後に送出する。
        if let Some((at, marker)) = find_stop_marker(&sess.text, sess.sent_len) {
            sess.text.truncate(at);
            sess.sent_len = sess.text.len();
            sess.stop_reason = StopReason::StopSequence;
            sess.stop_sequence = Some(marker);
            return Ok(true);
        }

        // stop_sequences チェック(未送出部分の中で最も早い出現で打ち切り)。
        // s53: クライアント指定 stop は上記ロールラベルと併用し、テキスト上の
        // 最も早い停止位置で打ち切る (既送出分に打ち切り位置が来ることはない)。
        if let Some((idx, seq)) = sess
            .stop_sequences
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| sess.text[sess.sent_len..].find(s.as_str()).map(|i| (i, s.clone())))
            .min_by_key(|(i, _)| *i)
        {
            let at = sess.sent_len + idx;
            sess.text.truncate(at);
            sess.sent_len = sess.text.len();
            sess.stop_reason = StopReason::StopSequence;
            sess.stop_sequence = Some(seq);
            return Ok(true);
        }

        // s53: 送出タイミング制御。未送出テキストが
        //   - マーカーに一致したら送出せず打ち切る (生成ループも終了)
        //   - マーカーを含まず・その prefix にもなり得ないなら全文送出
        //   - prefix になり得るならその手前まで送出し、末尾は保持
        let start = sess.sent_len;
        let at_line_start = start == 0 || sess.text[..start].ends_with('\n');
        match overall_stop_status(&sess.text[start..], at_line_start) {
            StopStatus::Matched(idx, marker) => {
                let at = start + idx;
                sess.text.truncate(at);
                sess.sent_len = sess.text.len();
                sess.stop_reason = StopReason::StopSequence;
                sess.stop_sequence = Some(marker);
                return Ok(true);
            }
            StopStatus::Partial(k) => {
                // k は text 側の char 境界に丸める (安全弁・通常は境界一致)
                let mut end = (start + k).min(sess.text.len());
                while end > start && !sess.text.is_char_boundary(end) {
                    end -= 1;
                }
                if end > start {
                    on_delta(&sess.text[start..end]);
                }
                sess.sent_len = end;
            }
            StopStatus::Clear => {
                if start < sess.text.len() {
                    on_delta(&sess.text[start..]);
                }
                sess.sent_len = sess.text.len();
            }
        }
        Ok(false)
    }

    /// P6: 複数セッションの decode 1 ステップを **1 回の forward_batch** で進める。
    ///
    /// 各セッション: サンプル → EOS/stop_sequences/max_tokens 判定 → 差分ストリーム。
    /// 継続するセッションだけ全て 1 つの `forward_batch` 呼び出しにまとめ、
    /// 各セッションの logits を更新する。
    ///
    /// **exactness 契約**: `forward_batch` は各セッションの KV/演算列が単独
    /// decode とビット一致するよう実装されている (`model::llama` の契約を参照)。
    /// サンプラは各セッションが独立した RNG を持つため、バッチに加わる他の
    /// セッションの存在は各セッションの出力に一切影響しない
    /// (**どのセッションと同時に実行しても単独実行とビット同一の出力**)。
    ///
    /// 戻り値: 各セッションが完了したか (true = `release_session` を呼んで終了)。
    /// エラー時はバッチ全体が失敗する (detok エラー等は実運用でほぼ起きない)。
    pub fn step_chat_batch(
        &mut self,
        batch: &mut [&mut ChatSession],
        // 第27: scheduler が Run から取り出した SSE アセンブラへの &mut を
        // クロージャに捕ませるため短命トレイトオブジェクトを許す
        // (Box<dyn FnMut> は既定で 'static 要求になるため明示的に '_ を付ける)
        on_deltas: &mut [Box<dyn FnMut(&str) + '_>],
    ) -> Result<Vec<bool>> {
        // s148 (Task B10・計測のみ): serve の decode 1 ステップの全体を表 C に積む。
        // サーバのループはこの関数を 1 回呼ぶごとに 1 トークン進めるので、これが
        // 「1 step = 1 サンプル」の最も内側の点。計測 OFF は `step_begin` が `None`
        // を返し、`step_end` は分岐 1 回で戻る (時刻も読まない)。
        let st = crate::prof::step_begin();
        let r = self.step_chat_batch_inner(batch, on_deltas);
        crate::prof::step_end(st);
        if matches!(&r, Ok(v) if v.iter().any(|&d| d)) {
            Self::log_weights_after_first_request();
        }
        // s155 (Task B16): `sl_step_other` の確定は **forward の戻りの後**。
        // `sl_step` の内訳 (`sl_step_lay_other` / `sl_so_*` / 次の `sl_lin`) は
        // step の後半で閉じるので、`step_end` の時点では揃っていない。
        crate::prof::step_end_tail();
        r
    }

    /// s148: [`step_chat_batch`] の本体 (表 C の step 計測を外したもの)。
    fn step_chat_batch_inner(
        &mut self,
        batch: &mut [&mut ChatSession],
        on_deltas: &mut [Box<dyn FnMut(&str) + '_>],
    ) -> Result<Vec<bool>> {
        debug_assert_eq!(batch.len(), on_deltas.len());
        let mut done = vec![false; batch.len()];
        let mut toks: Vec<u32> = Vec::new();
        let mut need_fwd: Vec<usize> = Vec::new();

        // 1) 各セッションのサンプルと終了判定 (モデルには触れない)
        for (i, sess) in batch.iter_mut().enumerate() {
            if sess.generated.len() >= sess.max_tokens {
                // max_tokens = 0 (prefill のみで確定) の場合を含む
                sess.stop_reason = StopReason::MaxTokens;
                done[i] = true;
                continue;
            }
            // s155 (Task B16): `sl_so_sample_pen` = サンプラ前の前処理 (penalty + 拘束マスク)。
            // s154 (B15) は行だけ作ってサイトを置いていなかった (実測 0 call は
            // 空振りではなく未計装)。両方とも条件付き (penalty 指定時 / tool_choice
            // 指定時) なので、**区間は条件の外側に必ず 1 本**置く — 何もしなかった
            // 区間は (ほぼ) 0 ns として積まれ、step 数と号数が一致する。
            let pen_t = crate::prof::span_begin();
            // A2.5: penalty 指定時のみ既出トークンを抑制 (無効時はビット不変)
            Self::apply_session_penalty(sess);
            // 第21: tool_choice 拘束 (Some のとき sample 直前に許容集合マスク)。
            // 拘束完了後は no-op = 無マスク経路とビット同一
            if sess.tool_mask.is_some() {
                let eos = self.model.eos_ids().to_vec();
                sess.tool_mask
                    .as_mut()
                    .unwrap()
                    .apply(&mut sess.logits, &self.tok, &eos);
            }
            crate::prof::span_end(pen_t, crate::prof::SL_SO_SAMPLE_PEN);
            // s148: 表 C の `sl_step_sampler` (1 セッション 1 サンプル)。
            let samp_t = crate::prof::span_begin();
            let next = crate::prof::time(crate::prof::SAMPLER, || {
                sess.sampler.sample(&sess.logits)
            }) as u32;
            crate::prof::span_end(samp_t, crate::prof::SL_STEP_SAMPLER);
            sess.generated.push(next);
            // 第21: 拘束状態の前進 (サンプル済みトークンの piece で状態遷移)
            if let Some(mask) = sess.tool_mask.as_mut() {
                mask.feed(&self.tok, next);
            }
            if telem::enabled() {
                telem::emit(Frame::Token {
                    session: sess.id,
                    token: next,
                    pos: sess.generated.len() as u32,
                });
                let tk = telem::topk_softmax(&sess.logits, 8);
                telem::emit(Frame::TopK {
                    session: sess.id,
                    chosen: next,
                    entropy_f16: telem::f32_to_f16_bits(tk.entropy),
                    entries: tk
                        .entries
                        .iter()
                        .map(|&(id, p)| (id, telem::f32_to_f16_bits(p)))
                        .collect(),
                });
            }

            // s154 (Task B15): `sl_step_other` の child `sl_so_stream` = detok /
            // delta 送出 / stop 判定 (この 3 分岐)。区間は分岐の手前で 1 回開き、
            // **どの出口でも閉じる** (下の 3 つの `continue` はそれぞれ閉じてから
            // 抜ける) ので 1 step 1 号。計測 OFF は `span_begin` が None で
            // `span_end` は素通し (数値経路・既定挙動は不変)。
            let stream_t = crate::prof::span_begin();
            if self.model.eos_ids().contains(&next) {
                sess.stop_reason = StopReason::EndTurn;
                self.flush_stream(sess, 1, on_deltas[i].as_mut())?;
                crate::prof::span_end(stream_t, crate::prof::SL_SO_STREAM);
                done[i] = true;
                continue;
            }
            if self.emit_delta_and_check_stop(sess, on_deltas[i].as_mut())? {
                crate::prof::span_end(stream_t, crate::prof::SL_SO_STREAM);
                done[i] = true;
                continue;
            }
            if sess.generated.len() >= sess.max_tokens {
                sess.stop_reason = StopReason::MaxTokens;
                self.flush_stream(sess, 0, on_deltas[i].as_mut())?;
                crate::prof::span_end(stream_t, crate::prof::SL_SO_STREAM);
                done[i] = true;
                continue;
            }
            crate::prof::span_end(stream_t, crate::prof::SL_SO_STREAM);
            toks.push(next);
            need_fwd.push(i);
        }

        // 2) 継続セッションだけまとめて 1 forward。
        //    スロット昇順に並べ替えて caches から非重複の &mut を取り出す
        //    (結果は行位置 → 元のセッション順に戻して代入)。
        if !need_fwd.is_empty() {
            let slots: Vec<usize> = need_fwd.iter().map(|&i| {
                batch[i].slot.expect("install_session 済みのセッション")
            }).collect();
            let mut order: Vec<usize> = (0..need_fwd.len()).collect();
            order.sort_by_key(|&k| slots[k]);
            let sorted_toks: Vec<u32> = order.iter().map(|&k| toks[k]).collect();
            let t0 = telem::enabled().then(std::time::Instant::now);
            let model = &mut *self.model;
            let caches = &mut *self.caches;
            let mut cache_refs = pick_caches(caches, &slots, &order);
            // s57 (T4): バッチ decode の計測フェーズを明示 (PROFILE 無効時 no-op)。
            // 各モデルの forward も入口で自身のフェーズを設定する (両方一致)。
            let outs = crate::prof::with_phase(crate::prof::Phase::Decode, || {
                model.forward_batch(&sorted_toks, &mut cache_refs)
            })?;
            if let Some(t0) = t0 {
                telem::emit(Frame::Step {
                    width: need_fwd.len() as u32,
                    ns: t0.elapsed().as_nanos() as u64,
                });
            }
            for (row, &k) in outs.into_iter().zip(&order) {
                let i = need_fwd[k];
                batch[i].logits = row;
            }
        }
        // 空バッチ (アイドルポーリング) では発出しない — アイドル中の
        // KvUsage フレーム洪水を防ぐ (P13 e2e で発見)
        if telem::enabled() && !batch.is_empty() {
            telem::emit(self.kv_usage_frame());
            self.flush_ops_frame();
        }
        // s154 (Task B15・計測のみ): `sl_step_other` の child。最後の forward の
        // 戻り (logits 反映) は次の `step_end` まで child が積まれない区間なので、
        // `sl_so_end` として毎 step 1 回閉じる。計測 OFF は `span_begin` が None。
        let end_t = crate::prof::span_begin();
        crate::prof::span_end(end_t, crate::prof::SL_SO_END);
        Ok(done)
    }

    /// チャット実行 (単発 API)。トークン生成ごとに on_delta(差分テキスト) が呼ばれる
    /// (ストリーミング用。非ストリーミングでは空クロージャでよい)。
    ///
    /// 1 セッションを通しで実行する経路は旧実装と同一の操作列 (同一形状の
    /// forward 呼び出し、同一の prefix cache 経路選択) なので挙動は完全に変わらない。
    pub fn chat(
        &mut self,
        system: Option<&str>,
        messages: &[ChatMessage],
        params: &GenParams,
        mut on_delta: impl FnMut(&str),
    ) -> Result<ChatOutput> {
        let mut sess = self.begin_chat(system, messages, params)?;
        self.install_session(&mut sess)?;
        while !self.step_chat(&mut sess, &mut on_delta)? {}
        self.release_session(&mut sess);
        Ok(sess.into_output())
    }
}

/// スロット群から指定位置の KvCache を非重複に取り出す
/// (`split_at_mut` の繰り返し。`slots` は任意順でよい)。
fn pick_caches<'a>(
    caches: &'a mut [KvCache],
    slots: &[usize],
    order: &[usize],
) -> Vec<&'a mut KvCache> {
    // order: slots の添字を「forward に渡す並び」で並べたもの。
    // cache_refs[k] は slots[order[k]] 番スロットへの &mut になる。
    let mut sorted: Vec<usize> = slots.to_vec();
    sorted.sort_unstable();
    let mut picked_sorted: Vec<&mut KvCache> = Vec::with_capacity(sorted.len());
    let mut rest: &mut [KvCache] = caches;
    let mut offset = 0usize;
    for &s in &sorted {
        let idx = s - offset;
        let (head, tail) = rest.split_at_mut(idx + 1);
        picked_sorted.push(&mut head[idx]);
        rest = tail;
        offset = s + 1;
    }
    // picked_sorted は slot 昇順。order に従って forward 順に並べ替える。
    // (sort_unstable_by_key + index 参照で安全に再配置する)
    let mut refs: Vec<Option<&mut KvCache>> = picked_sorted.into_iter().map(Some).collect();
    let mut result = Vec::with_capacity(order.len());
    for &k in order {
        let pos = sorted.binary_search(&slots[k]).expect("slot must be in sorted");
        let r = refs[pos].take().expect("each slot used exactly once");
        result.push(r);
    }
    result
}

/// s53: ロールラベル停止マーカーの最長バイト長。
///
/// 最長は特殊トークン文字列 `<|im_start|>` (12 バイト)。holdback の上限と
/// 「マーカーが完成し得る末尾」の計算に使う (これより短い prefix だけを保持する)。
pub const MAX_STOP_MARKER_LEN: usize = 12;

/// s53: 特殊トークン文字列マーカー (出現位置を問わず即停止)。
const STOP_SPECIAL_MARKERS: [&str; 2] = ["<|im_start|>", "<|im_end|>"];

/// s53: 単独行のロールラベル名 (行頭の空白許容・直後は行末 or `:` / `：` のみ)。
const STOP_LINE_LABELS: [&str; 3] = ["user", "assistant", "system"];

/// `s[i..]` の先頭で `needle` の途中まで (1 文字以上) 一致している最大長を返す
/// (完全一致は含まない = 完全一致を「部分」と誤判定しない)。
fn partial_prefix_len(s: &str, needle: &str) -> usize {
    let max = s.len().min(needle.len().saturating_sub(1));
    let mut best = 0usize;
    for (i, ch) in s.char_indices() {
        let end = i + ch.len_utf8();
        if end > max {
            break;
        }
        if needle.is_char_boundary(end) && needle[..end] == s[..end] {
            best = end;
        }
    }
    best
}

/// 単独行マーカー (`user` / `assistant` / `system`) がテキスト位置 `i` で
/// 開始しているかを判定する。成立時はマーカー名を返す。
///
/// - 行頭 (テキスト先頭 or 直前が `\n`) + 行頭の空白 (` ` / `\t`) は許容
/// - 直後は行末 / `\r\n` / `:` / `：` のみ (前後に空白許容)
fn line_label_at(text: &str, i: usize) -> Option<&'static str> {
    STOP_LINE_LABELS
        .iter()
        .find(|label| tail_is_line_end(&text[i..], label).is_some())
        .copied()
}

/// `s` (行頭から始まる文字列) が `label` + 許可される末尾で **確定** して
/// いるかを判定する。`label` 自体が `s` より長くても、`s` が `label` の
/// prefix ならば「まだ確定していないが候補」として扱わない (確定のみ判定)。
fn tail_is_line_end(s: &str, label: &str) -> Option<usize> {
    let body = s.trim_start_matches([' ', '\t']);
    let after = body.strip_prefix(label)?;
    let tail = after.trim_start_matches([' ', '\t']);
    let end = tail.len();
    let ok = end == 0
        || tail == "\n"
        || tail == "\r\n"
        || tail.starts_with(":\n")
        || tail.starts_with(":\r")
        || tail.starts_with("：\n")
        || tail.starts_with("：\r");
    ok.then_some(body.len() - tail.len())
}

/// `s` が「マーカーの prefix になり得る」場合、保持すべき先頭 (= マーカー候補の
/// 開始位置) を返す。確定済みのマーカー・無関係なテキストでは `None`。
///
/// `s` は「行頭からの suffix」として渡す。マーカー候補の開始は行頭の空白を
/// 読み飛ばした位置 (= 送出してよい手前) になる。
fn partial_holdback(s: &str) -> Option<usize> {
    let body_off = s.len() - s.trim_start_matches([' ', '\t']).len();
    let body = &s[body_off..];
    if body.is_empty() {
        return None;
    }
    if STOP_SPECIAL_MARKERS
        .iter()
        .any(|m| partial_prefix_len(body, m) > 0)
    {
        return Some(body_off);
    }
    for label in STOP_LINE_LABELS {
        // ラベル名の途中まで一致
        if partial_prefix_len(body, label) > 0 {
            return Some(body_off);
        }
        let Some(after) = body.strip_prefix(label) else {
            continue;
        };
        let tail = after.trim_start_matches([' ', '\t']);
        if tail.is_empty() {
            // ラベル直後 (行末 / `:` が続くか未確定)
            return Some(body_off);
        }
        if tail == "\r" {
            // `\r\n` の `\n` が未確定
            return Some(body_off);
        }
        // `:` / `：` の途中まで一致 (2 バイト文字の分断を含む)
        if partial_prefix_len(tail, ":") > 0 || partial_prefix_len(tail, "：") > 0 {
            return Some(body_off);
        }
        // `:` / `：` の直後 (改行が続くか未確定)
        if tail == ":" || tail == "：" {
            return Some(body_off);
        }
    }
    None
}

/// テキスト `text` の中で (まだ確定していない位置 `i` 以降で) 最初に現れる
/// ロールラベル停止マーカーの (開始バイト位置, マーカー文字列) を返す。
///
/// 本関数は **確定済みテキスト全体** を渡す前提。行頭判定が行をまたいで
/// 正しく効くようにするため、呼び出し側は holdback を含む text 全体を渡す。
fn find_stop_marker(text: &str, start: usize) -> Option<(usize, String)> {
    if start > text.len() {
        return None;
    }
    // 特殊トークン文字列: 出現位置を問わず即停止 (絶対位置で返す)
    if let Some((i, m)) = STOP_SPECIAL_MARKERS
        .iter()
        .filter_map(|m| text[start..].find(m).map(|i| (start + i, *m)))
        .min_by_key(|(i, _)| *i)
    {
        return Some((i, m.to_string()));
    }
    // 行頭 = テキスト先頭 or `\n` の直後 (text 全体基準)。
    // s53b: `start` はチャンク境界で単語の途中に落ち得るため、`start` を含む行
    // から走査し、`start` より前の行頭は判定しない (旧実装は
    // `&text[sent_len..]` を渡して本文中の "system" を誤検知していた)。
    let mut line_start = text[..start].rfind('\n').map(|p| p + 1).unwrap_or(0);
    loop {
        if line_start >= start {
            if let Some(label) = line_label_at(text, line_start) {
                return Some((line_start, label.to_string()));
            }
        }
        match text[line_start..].find('\n') {
            Some(k) => line_start += k + 1,
            None => break,
        }
        if line_start >= text.len() {
            break;
        }
    }
    None
}

/// s53: 末尾テキストの停止可能性 (holdback 判定)。
#[derive(Debug, Clone, PartialEq, Eq)]
enum StopStatus {
    /// 位置 `usize` でマーカーが確定した (マーカー文字列付き)。
    Matched(usize, String),
    /// 長さ `usize` はマーカーの prefix になり得る → 送出せず保持する。
    Partial(usize),
    /// 停止マーカーとは無関係 → 全文送出してよい。
    Clear,
}

/// 末尾テキスト `s` (「この suffix が送信済みでない未確定部分」として渡す) の
/// 停止可能性を判定する。
///
/// `s` がマーカーを含む場合はその開始位置 (s 先頭からのバイト位置) を返す。
/// 含まない場合、`s` の末尾の「マーカーの prefix になり得る部分」の開始
/// (= 保持すべき部分の開始バイト位置) を返す。どちらでもなければ `Clear`。
///
/// s53b: `at_line_start` は `s` がテキスト全体の真の行頭から始まるかを示す。
/// `s` はチャンク境界 (sent_len) から切り出されるため単語の途中から始まり得る。
/// その場合に単独行ラベルの Matched を返すと本文中の "system" 等を誤検知する
/// ため、行頭からのときのみラベル判定を行う。
fn overall_stop_status(s: &str, at_line_start: bool) -> StopStatus {
    // 1) 特殊トークン文字列が既に現れている
    if let Some((i, m)) = STOP_SPECIAL_MARKERS
        .iter()
        .filter_map(|m| s.find(m).map(|i| (i, *m)))
        .min_by_key(|(i, _)| *i)
    {
        return StopStatus::Matched(i, m.to_string());
    }
    // 2) 確定した単独行マーカー (行頭ごとに判定・真の行頭からのみ)
    let mut line_starts: Vec<usize> = if at_line_start { vec![0] } else { Vec::new() };
    for (i, ch) in s.char_indices() {
        if ch == '\n' {
            line_starts.push(i + 1);
        }
    }
    for &st in &line_starts {
        if st >= s.len() {
            continue;
        }
        if let Some(label) = line_label_at(s, st) {
            return StopStatus::Matched(st, label.to_string());
        }
    }
    // 3) 最終行 (= まだ伸び得る唯一の行) の末尾がマーカーの prefix か
    // s53b: at_line_start=false かつ改行無しのチャンクでは空 (先頭行扱い)。
    let last_start = line_starts.last().copied().unwrap_or(0);
    partial_holdback(&s[last_start..])
        .map(|k| StopStatus::Partial(last_start + k))
        .unwrap_or(StopStatus::Clear)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{
        dummy_engine, recurrent_engine, recurrent_qwen_style_engine, thinking_sim_engine,
        DUMMY_CONTEXT,
    };

    fn params(max_tokens: usize) -> GenParams {
        GenParams {
            max_tokens,
            temperature: Some(0.0),
            ..Default::default()
        }
    }

    fn msgs(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage { role: "user".into(), content: text.into() }]
    }

    /// ダミーモデル (fallback テンプレート: "user: t0\n\nassistant:") の
    /// greedy 生成は t5 → t6 → ... → t11 → EOS の順に進む。
    /// (プロンプトの最終トークンは常に "assistant:" → [UNK]=20 → 次ピークは 5)
    const EXPECTED_GREEDY_TEXT: &str = "t5 t6 t7 t8 t9 t10 t11";

    #[test]
    fn greedy_decoding_is_deterministic_across_engines() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut a = dummy_engine("m", None).unwrap();
        let mut b = dummy_engine("m", None).unwrap();
        let out_a = a.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        let out_b = b.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(out_a.text, EXPECTED_GREEDY_TEXT);
        assert_eq!(out_a.text, out_b.text);
        assert_eq!(out_a.stop_reason, StopReason::EndTurn);
        // EOS トークンも generated には数えられる
        assert_eq!(out_a.output_tokens, out_a.output_tokens.max(1));
    }

    #[test]
    fn seeded_sampling_is_deterministic_within_and_across_engines() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut p = params(30);
        p.temperature = Some(1.0);
        p.seed = Some(42);
        let mut a = dummy_engine("m", None).unwrap();
        let mut b = dummy_engine("m", None).unwrap();
        let out_a1 = a.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        let out_a2 = a.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        let out_b = b.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        // 同一 seed → 同一出力 (エンジンをまたいでも、同一エンジン2回目でも)
        assert_eq!(out_a1.text, out_a2.text);
        assert_eq!(out_a1.text, out_b.text);
        assert_eq!(out_a1.output_tokens, out_b.output_tokens);
    }

    #[test]
    fn prefix_cache_reuses_prompt_and_keeps_exact_output() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        let out1 = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();

        // スロット 0 の cached_prompt にプロンプト全体が入っている
        let prompt = e
            .tok
            .apply_chat_template(None, &msgs("t0"), None, None)
            .unwrap();
        let prompt_ids = e.tok.encode(&prompt).unwrap();
        assert_eq!(e.cached_prompts[0], prompt_ids);

        // 2回目 (warm cache): LCP 再利用後も同一出力
        let out2 = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(out1.text, out2.text);
        assert_eq!(out1.input_tokens, out2.input_tokens);
        assert_eq!(out1.output_tokens, out2.output_tokens);

        // 3回目: プロンプトの共通接頭辞が短い場合も同一出力
        let out3 = e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        assert_eq!(out3.text, EXPECTED_GREEDY_TEXT);
    }

    /// ダミートークナイザの fallback テンプレートでは "t0" と "t15" のプロンプトは
    /// [UNK, tN, UNK] というトークン列になり、共通接頭辞は 1 トークンのみ。
    /// 別セッションとして扱える。
    #[test]
    fn multi_session_alternation_restores_entries_and_keeps_exact_output() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();

        // 基準出力 (キャッシュ無しの新規エンジン)
        let mut fresh_a = dummy_engine("a", None).unwrap();
        let mut fresh_b = dummy_engine("b", None).unwrap();
        let ref_a = fresh_a.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        let ref_b = fresh_b.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();

        // req1: セッションA (初回 → full prefill)
        let out1 = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);
        // req2: セッションB (live との共通接頭辞は1トークン → live 再利用。
        // req1 完了時に A が、req2 完了時に B がストアへ退避される → 2 エントリ)
        let out2 = e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Live);
        assert_eq!(e.prefix_cache_entries(), 2);
        // req3: セッションAに戻る → live との LCP(1) よりエントリの LCP(全体) が大きいので復元
        let out3 = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);
        assert_eq!(e.prefix_cache_entries(), 2);
        // req4: 再びセッションB → エントリから復元
        let out4 = e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);

        // 交互に投げても出力は単独実行と完全一致 (exact decoding)
        assert_eq!(out1.text, ref_a.text);
        assert_eq!(out3.text, ref_a.text);
        assert_eq!(out1, ref_a);
        assert_eq!(out3, ref_a);
        assert_eq!(out2, ref_b);
        assert_eq!(out4, ref_b);
    }

    #[test]
    fn prefix_cache_budget_is_configurable_and_evicts() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        assert_eq!(e.prefix_cache_max_tokens(), DUMMY_CONTEXT);

        // 2会話分ストアに入れる (エントリ数 2 を確認)
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap(); // B退避で2エントリ
        assert_eq!(e.prefix_cache_entries(), 2);

        // 上限を引き下げると LRU eviction される
        e.set_prefix_cache_max_tokens(1);
        assert_eq!(e.prefix_cache_max_tokens(), 1);
        assert_eq!(e.prefix_cache_entries(), 0);

        // eviction 後も出力は不変 (full prefill にフォールバック)
        let out = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(out.text, EXPECTED_GREEDY_TEXT);
    }

    #[test]
    fn prefix_cache_hit_via_live_after_entry_restore() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);
        // 同一セッション連続は live 経路 (単一エントリ時代と同じ、コピー無し)
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Live);
    }

    // ===== A4: 再帰モデル (GDN 状態スナップショット) の prefix cache 復元 =====

    /// (role, content) の列からメッセージ列を作る
    fn msgs_multi(turns: &[(&str, &str)]) -> Vec<ChatMessage> {
        turns
            .iter()
            .map(|(role, content)| ChatMessage {
                role: (*role).into(),
                content: (*content).into(),
            })
            .collect()
    }

    /// A4: 再帰モデルのマルチターン会話で、エントリ復元 (KV + GDN 状態の
    /// ペア復元) 後の出力が「フル prefill の参照エンジン」と完全一致する。
    /// fallback テンプレートにより turn_{n+1} の prompt は turn_n の prompt を
    /// トークン列として完全に含む ("user: t0\n\nassistant: <生成>\n\nuser: ...")
    /// → best_full_match が成立し GDN 状態スナップショットが使われる。
    /// RecurStateModel は logits が GDN 状態 + KV 両方に依存するため、
    /// どちらかの復元が欠けると生成列が一致しなくなる。
    #[test]
    fn recurrent_entry_restore_matches_full_prefill() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_engine("recur", None).unwrap();
        let mut ref2 = recurrent_engine("recur-ref2", None).unwrap();
        let mut ref3 = recurrent_engine("recur-ref3", None).unwrap();

        // turn1: フル prefill (prefill 直後にエントリが作られる)
        let out1 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        assert_eq!(e.prefix_cache_entries(), 1);
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);

        // turn2: turn1 の prompt を完全に含むマルチターン prompt
        let m2 = msgs_multi(&[("user", "t0"), ("assistant", &out1.text), ("user", "t15")]);
        let expect2 = ref2.chat(None, &m2, &params(3), |_| {}).unwrap();
        let out2 = e.chat(None, &m2, &params(3), |_| {}).unwrap();
        assert_eq!(
            e.last_prefix_hit(),
            PrefixHit::Entry,
            "GDN 状態スナップショット経由で復元されるべき"
        );
        assert_eq!(out2, expect2, "エントリ復元 (KV+GDN 状態) 後の出力がフル prefill と不一致");

        // turn3: turn2 の prompt をさらに完全に含む (スナップショットの連鎖)
        let m3 = msgs_multi(&[
            ("user", "t0"),
            ("assistant", &out1.text),
            ("user", "t15"),
            ("assistant", &out2.text),
            ("user", "t13"),
        ]);
        let expect3 = ref3.chat(None, &m3, &params(3), |_| {}).unwrap();
        let out3 = e.chat(None, &m3, &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);
        assert_eq!(out3, expect3, "2 回目のエントリ復元後の出力がフル prefill と不一致");
    }

    /// A4: 部分一致 (LCP < エントリ prompt 長) の prompt では再帰モデルは
    /// 状態を復元できず full prefill にフォールバックする
    /// (KV のみ復元して静かに出力を変える経路は存在しない)。
    #[test]
    fn recurrent_partial_prefix_falls_back_to_full_prefill() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_engine("recur", None).unwrap();
        let mut r = recurrent_engine("recur-ref", None).unwrap();
        e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        // "t1" は "t0" とトークン接頭辞が UNK 1 個分しか一致しない
        // → 完全一致エントリ無し → live も拡張でない → full prefill
        let expect = r.chat(None, &msgs("t1"), &params(3), |_| {}).unwrap();
        let out = e.chat(None, &msgs("t1"), &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);
        assert_eq!(out, expect);
    }

    /// A4: 同一 prompt の再実行は状態スナップショットを使えない
    /// (prompt 境界の状態では最終トークンの logits が作れない) ので full prefill。
    /// 出力はフル prefill と完全一致する (状態の整合は保たれる)。
    #[test]
    fn recurrent_identical_prompt_falls_back_to_full_prefill() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_engine("recur", None).unwrap();
        let mut r = recurrent_engine("recur-ref", None).unwrap();
        let out1 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        let out2 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        let expect = r.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        assert_ne!(e.last_prefix_hit(), PrefixHit::Entry);
        assert_eq!(out2, out1, "同一 prompt 再実行で出力が変わってはならない");
        assert_eq!(out2, expect);
    }

    // ===== s46: generation prompt 境界の split prefill + base エントリ復元 =====

    /// s46: Qwen3.5 風テンプレート (各 prompt を generation prompt
    /// `<|im_start|>assistant\n\n\n` = 末尾に `<think>` トークンで終える) では
    /// turn_{n+1} の prompt が turn_n の prompt をトークン接頭辞として含まない
    /// (generation prompt が assistant 本文で置き換わる = 実 Qwen3.5 と同一構造)。
    /// 修正前は best_full_match が永遠に成立せず recurrent モデルの prefix cache
    /// が 1 度もヒットしない (REPORT_NEXT19 の実機観察と同一)。
    /// 修正後: prefill が generation prompt 境界で分割され、base
    /// (= generation prompt 直前まで) の KV+GDN スナップショットが退避され、
    /// 次ターン以降で復元される。出力はフル prefill の参照エンジンと完全一致。
    #[test]
    fn recurrent_qwen_style_template_hits_entry_on_multiturn() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_qwen_style_engine("qwen-style", None).unwrap();
        let mut ref2 = recurrent_qwen_style_engine("qwen-style-ref2", None).unwrap();
        let mut ref3 = recurrent_qwen_style_engine("qwen-style-ref3", None).unwrap();

        // turn1: フル prefill。エントリは prompt 全体ではなく base
        // (generation prompt 直前まで) で保存される
        let out1 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        assert_eq!(e.prefix_cache_entries(), 1);
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);
        // base キーイングの確認: turn1 prompt 全体を渡しても完全一致するのは
        // base エントリ (長さ < prompt 全長) のみ
        let turn1_full = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &msgs("t0"), None, None).unwrap())
            .unwrap();
        let turn1_base = e
            .tok
            .encode(&e.tok.apply_chat_template_base(None, &msgs("t0"), None, None).unwrap())
            .unwrap();
        assert!(turn1_full.starts_with(&turn1_base), "base 描画がトークン接頭辞");
        assert!(turn1_base.len() < turn1_full.len(), "generation prompt で末尾が伸びる");
        let (_, base_len) = e
            .prefix_cache
            .best_full_match(&turn1_full)
            .expect("turn1 の base エントリが完全一致する");
        assert_eq!(base_len, turn1_base.len(), "エントリは base 境界で保存される");

        // turn2: turn1 の prompt を接頭辞に含まない (generation prompt が置換)。
        // 修正後は Entry 復元 + base 部の差分 forward で出力がフル prefill と一致
        let m2 = msgs_multi(&[("user", "t0"), ("assistant", &out1.text), ("user", "t15")]);
        let expect2 = ref2.chat(None, &m2, &params(3), |_| {}).unwrap();
        let out2 = e.chat(None, &m2, &params(3), |_| {}).unwrap();
        assert_eq!(
            e.last_prefix_hit(),
            PrefixHit::Entry,
            "generation prompt 境界の base 復元でヒットするべき"
        );
        assert_eq!(out2, expect2, "base 復元後の出力がフル prefill と不一致");
        assert_eq!(e.prefix_cache_entries(), 2, "turn2 の base も退避される");

        // turn3: 複数の完全一致候補 (turn1/turn2 の base) のうち最長 (turn2 base)
        // が選ばれる (turn3 自身の base はこの時点では未登録)
        let m3 = msgs_multi(&[
            ("user", "t0"),
            ("assistant", &out1.text),
            ("user", "t15"),
            ("assistant", &out2.text),
            ("user", "t13"),
        ]);
        let turn3_full = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &m3, None, None).unwrap())
            .unwrap();
        let turn2_full = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &m2, None, None).unwrap())
            .unwrap();
        let turn2_base = e
            .tok
            .encode(&e.tok.apply_chat_template_base(None, &m2, None, None).unwrap())
            .unwrap();
        assert!(turn3_full.starts_with(&turn2_full[..turn2_base.len()]), "turn2 base は turn3 の接頭辞");
        let (_, pick_len) = e
            .prefix_cache
            .best_full_match(&turn3_full)
            .expect("turn3 には完全一致候補がある");
        assert_eq!(pick_len, turn2_base.len(), "最長の完全一致 (turn2 の base) が選ばれる");
        let expect3 = ref3.chat(None, &m3, &params(3), |_| {}).unwrap();
        let out3 = e.chat(None, &m3, &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);
        assert_eq!(out3, expect3, "2 回目の base 復元後の出力がフル prefill と不一致");
    }

    /// s46: 同一 prompt (同一 messages) の再実行も base 復元でヒットする
    /// (generation prompt 部分のみ再 forward)。fallback テンプレートの
    /// recurrent_identical_prompt_falls_back_to_full_prefill と対照的
    /// (fallback は境界不明 = 従来経路のまま)。
    #[test]
    fn recurrent_qwen_style_template_identical_resend_hits_entry() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_qwen_style_engine("qwen-style", None).unwrap();
        let out1 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);
        let out2 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry, "同一 messages 再実行も base 復元でヒット");
        assert_eq!(out2, out1, "同一 messages 再実行で出力が変わってはならない");
        // 3 回目も同様にヒット (復元 → base 部 skip (lcp == b) → gen 部のみ forward)
        let out3 = e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);
        assert_eq!(out3, out1);
    }

    /// s46: Qwen3.5 風テンプレートでも prefix 崩れ以外の部分一致
    /// (LCP < エントリ base 長) は状態を復元できないので full prefill に
    /// フォールバックする (安全側経路の維持確認)。
    #[test]
    fn recurrent_qwen_style_template_partial_prefix_still_falls_back() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_qwen_style_engine("qwen-style", None).unwrap();
        let mut r = recurrent_qwen_style_engine("qwen-style-ref", None).unwrap();
        e.chat(None, &msgs("t0"), &params(3), |_| {}).unwrap();
        // "t1" は "t0" と共通接頭辞を持たない → base エントリとも完全一致しない
        let expect = r.chat(None, &msgs("t1"), &params(3), |_| {}).unwrap();
        let out = e.chat(None, &msgs("t1"), &params(3), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);
        assert_eq!(out, expect);
    }

    // ---------- s47: thinking 制御 ----------

    /// thinking=off を engine 経路 (begin_chat_with_tools) で回すヘルパ。
    fn run_chat_with_thinking(
        e: &mut Engine,
        messages: &[ChatMessage],
        max_tokens: usize,
        enable_thinking: Option<bool>,
    ) -> ChatOutput {
        let mut sess = e
            .begin_chat_with_tools(None, messages, &params(max_tokens), None, None, enable_thinking)
            .unwrap();
        e.install_session(&mut sess).unwrap();
        while !e.step_chat(&mut sess, &mut |_| {}).unwrap() {}
        e.release_session(&mut sess);
        sess.into_output()
    }

    /// s47: thinking=off (enable_thinking=Some(false)) は generation prompt を
    /// 実 Qwen3.5 と同じ空の think block に変える。従来経路 (None/Some(true)) の
    /// generation prompt は assistant (19) で終わる (開いた think block = 思考開始待ち)。
    /// off (Some(false)) は `<think>` (20) と `</think>` (22) を追加する (実 Qwen3.5 と
    /// 同一の「think 終了タグの直前が改行3つ」構造 = `assistant\n` + `<think>` + `\n\n\n` +
    /// `</think>`。2 タグは added_tokens 登録済みで単一トークン化・間/後続の改行は
    /// Whitespace 前処理で落ちる = prompt は 2 トークン伸びて off = full + [20, 22])。
    #[test]
    fn thinking_off_renders_empty_think_block_in_generation_prompt() {
        let e = recurrent_qwen_style_engine("qwen-style", None).unwrap();
        let m = msgs("t0");

        // 従来経路 (None = 変数無し): assistant (19) で終わる (開いた think block)
        let ids_full = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &m, None, None).unwrap())
            .unwrap();
        assert_eq!(ids_full.last().copied(), Some(19));
        assert!(!ids_full.contains(&22), "従来経路に空 think block は現れない");

        // off (Some(false)): 空の think block → 末尾が [<think> (20), </think> (22)]
        // (added_tokens 登録済みで単一トークン化・間の改行3つは前処理で落ちる)
        let ids_off = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &m, None, Some(false)).unwrap())
            .unwrap();
        assert_eq!(ids_off.last().copied(), Some(22));
        assert_eq!(ids_off.len(), ids_full.len() + 2, "空 think block で prompt が 2 トークン伸びる");
        assert_eq!(&ids_off[ids_full.len()..], &[20, 22], "off = full + [20, 22]");
        assert!(ids_off.starts_with(&ids_full), "off でも履歴部 (base) はトークン接頭辞のまま");

        // Some(true) は変数を渡すが思考 ON = 従来経路と同一の描画
        let ids_on = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &m, None, Some(true)).unwrap())
            .unwrap();
        assert_eq!(ids_on, ids_full);
    }

    /// s47: thinking=off は即答 (空 think block 直後で EOS・1 トークンで完答)、
    /// full (None) は従来どおり思考する (フィラ 3 トークン + EOS)。
    /// ThinkingSimModel が実モデルの「思考する/しない」をトークンレベルで模倣する。
    #[test]
    fn thinking_off_short_circuits_generation_and_full_thinks() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = thinking_sim_engine("think-sim").unwrap();
        let m = msgs("t0");

        let out_off = run_chat_with_thinking(&mut e, &m, 20, Some(false));
        assert_eq!(out_off.output_tokens, 1, "off は空 think block 直後で即 EOS (EOS のみ)");
        assert_eq!(out_off.stop_reason, StopReason::EndTurn);

        let out_full = run_chat_with_thinking(&mut e, &m, 20, None);
        assert_eq!(out_full.output_tokens, 4, "full は思考フィラ 3 トークン + EOS");
        assert_eq!(out_full.stop_reason, StopReason::EndTurn);
    }

    /// s47: thinking=off でも recurrent モデルの base 検出 (s46 split prefill) が
    /// 働く。base 描画は同一引数 (enable_thinking 込み) で行われるため、
    /// off のマルチターンでも PrefixHit::Entry になり出力はフル prefill と一致する。
    #[test]
    fn thinking_off_keeps_recurrent_base_split() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = recurrent_qwen_style_engine("qwen-style", None).unwrap();
        let mut r = recurrent_qwen_style_engine("qwen-style-ref", None).unwrap();

        let out1 = run_chat_with_thinking(&mut e, &msgs("t0"), 3, Some(false));
        assert_eq!(e.last_prefix_hit(), PrefixHit::None);
        assert_eq!(e.prefix_cache_entries(), 1, "off でも base エントリが退避される");

        let m2 = msgs_multi(&[("user", "t0"), ("assistant", &out1.text), ("user", "t15")]);
        let expect2 = run_chat_with_thinking(&mut r, &m2, 3, Some(false));
        let out2 = run_chat_with_thinking(&mut e, &m2, 3, Some(false));
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry, "off でも base 復元でヒットする");
        assert_eq!(out2, expect2, "off 経路の出力がフル prefill と不一致");
    }

    #[test]
    fn stop_sequence_truncates_at_first_occurrence() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        let full = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();

        let mut e2 = dummy_engine("m", None).unwrap();
        let mut p = params(20);
        p.stop_sequences = vec!["t7".into()];
        let stopped = e2.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        assert_eq!(stopped.stop_reason, StopReason::StopSequence);
        assert_eq!(stopped.stop_sequence.as_deref(), Some("t7"));
        assert!(full.text.starts_with(&stopped.text));
        assert!(!stopped.text.contains("t7"));

        // 生成テキストに現れない stop sequence は影響しない
        let mut e3 = dummy_engine("m", None).unwrap();
        let mut p = params(20);
        p.stop_sequences = vec!["t99".into()];
        let out = e3.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        assert_eq!(out.stop_reason, StopReason::EndTurn);
        assert_eq!(out.text, EXPECTED_GREEDY_TEXT);
    }

    #[test]
    fn max_tokens_is_clamped_to_fit_context() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        // コンテキスト (64) を大きく超える max_tokens はクランプされて続行する
        let out = e.chat(None, &msgs("t0"), &params(10_000), |_| {}).unwrap();
        assert!(out.input_tokens + out.output_tokens <= DUMMY_CONTEXT);
        assert_eq!(out.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn max_tokens_limit_stops_generation() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        // EOS に届く前に打ち切る
        let out = e.chat(None, &msgs("t0"), &params(2), |_| {}).unwrap();
        assert_eq!(out.stop_reason, StopReason::MaxTokens);
        assert_eq!(out.output_tokens, 2);
        assert_eq!(out.text, "t5 t6");
    }

    #[test]
    fn on_delta_receives_the_same_text_as_result() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        let mut streamed = String::new();
        let out = e
            .chat(None, &msgs("t0"), &params(20), |delta| streamed.push_str(delta))
            .unwrap();
        assert_eq!(streamed, out.text);
    }

    #[test]
    fn empty_messages_is_an_error() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        assert!(e.chat(None, &[], &params(4), |_| {}).is_err());
    }

    #[test]
    fn system_prompt_is_included_in_prompt_tokens() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let mut e = dummy_engine("m", None).unwrap();
        let with_sys = e.chat(Some("sys"), &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(with_sys.text, EXPECTED_GREEDY_TEXT);
        // fallback テンプレート "system: sys\n\nuser: t0\n\nassistant:" は
        // system 無しより多くのトークンを消費する
        let mut e2 = dummy_engine("m", None).unwrap();
        let without_sys = e2.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert!(with_sys.input_tokens > without_sys.input_tokens);
    }

    // ---------- P4 ステッパ (インターリーブ) ----------

    use crate::testutil::kv_sensitive_engine;

    /// 交互実行下 (A 常駐中に B を割り込ませ、B 完了後に A を続行) でも
    /// 各リクエストの出力は単独実行と完全一致する (P4 最重要要件)。
    /// P6 ではセッションは各々のスロットに常駐し続けるため、suspend せずに
    /// 別セッションを install して進める。
    /// DummyModel は KV を参照しないため、KV 依存の KvSensitiveModel を使う。
    #[test]
    fn interleaved_sessions_match_standalone_outputs() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let p = params(6);
        // 単独実行の基準 (それぞれ独立したエンジンで full prefill)
        let mut ra = kv_sensitive_engine("a", None).unwrap();
        let ref_a = ra.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        let mut rb = kv_sensitive_engine("b", None).unwrap();
        let ref_b = rb.chat(None, &msgs("t15"), &p, |_| {}).unwrap();
        assert_ne!(
            ref_a.text, ref_b.text,
            "テストの前提: プロンプトが異なれば出力も異なること"
        );

        let mut e = kv_sensitive_engine("m", None).unwrap();
        let mut sa = e.begin_chat(None, &msgs("t0"), &p).unwrap();
        let mut sb = e.begin_chat(None, &msgs("t15"), &p).unwrap();

        // A: prefill まで進める (スロット 0 に常駐)
        e.install_session(&mut sa).unwrap();
        assert!(!e.step_chat(&mut sa, &mut |_| {}).unwrap());

        // A を常駐させたまま B を割り込ませ、完了まで進める (スロット 1 に常駐)
        e.install_session(&mut sb).unwrap();
        for _ in 0..100 {
            if e.step_chat(&mut sb, &mut |_| {}).unwrap() {
                break;
            }
        }
        e.release_session(&mut sb);
        let out_b = sb.into_output();

        // A を続行 (KV はスロット 0 に残ったまま)
        for _ in 0..100 {
            if e.step_chat(&mut sa, &mut |_| {}).unwrap() {
                break;
            }
        }
        e.release_session(&mut sa);
        let out_a = sa.into_output();

        assert_eq!(out_a, ref_a, "インターリーブ後の A が単独実行と不一致");
        assert_eq!(out_b, ref_b, "割り込んだ B が単独実行と不一致");
    }

    /// s41: 同一エンジン (同一 KV スロット) で短い会話 A の後に長い別会話 B を
    /// 実行しても B の出力は単独実行と完全一致する (1b→2b 順序の再現テスト)。
    /// CPU 経路では GPU KV ミラーが介在しないため常に成立するアンカーだが、
    /// GPU 経路では「同一 kv_id のミラーが前会話の行 [0..have) を正規行として
    /// 再利用する」不具合 (s35 の 2b 2 値分岐) がこの順序でのみ顕在化した
    /// (fresh 初回は len=0 で全行再 seed・2 回目以降は rollback ガードで再 seed)。
    /// 修正 (KvCache epoch + with_mirror_ext の世代検証) 後は GPU 経路でも成立する。
    /// 併せて長→短 (rollback ガード経路) と prefix cache 再利用 (live/entry) 経路も
    /// 単独実行と一致することを固定する。
    #[test]
    fn short_then_long_conversation_matches_standalone_outputs() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let p = params(6);
        let short = "t0";
        let long = "t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t13 t14 t15 t0 t1 t2 t3";
        let ref_a = {
            let mut solo = kv_sensitive_engine("a", None).unwrap();
            solo.chat(None, &msgs(short), &p, |_| {}).unwrap()
        };
        let ref_b = {
            let mut solo = kv_sensitive_engine("b", None).unwrap();
            solo.chat(None, &msgs(long), &p, |_| {}).unwrap()
        };
        assert_ne!(
            ref_a.text, ref_b.text,
            "テストの前提: プロンプトが異なれば出力も異なること"
        );

        let mut e = kv_sensitive_engine("m", None).unwrap();
        // 短 → 長 (1b→2b 相当・問題の順序)
        let a1 = e.chat(None, &msgs(short), &p, |_| {}).unwrap();
        assert_eq!(a1, ref_a);
        let b1 = e.chat(None, &msgs(long), &p, |_| {}).unwrap();
        assert_eq!(b1, ref_b, "短い会話の後の長い会話が単独実行と不一致");
        // 長 → 長 (旧実装でも rollback ガードで再 seed されて安定していた経路)
        let b2 = e.chat(None, &msgs(long), &p, |_| {}).unwrap();
        assert_eq!(b2, ref_b);
        // 長 → 短 (rollback ガード経路)
        let a2 = e.chat(None, &msgs(short), &p, |_| {}).unwrap();
        assert_eq!(a2, ref_a);
        // 短 → 長 をもう一度 (切替は繰り返し発火する・entry restore 経路)
        let b3 = e.chat(None, &msgs(long), &p, |_| {}).unwrap();
        assert_eq!(b3, ref_b);
    }

    /// 4 セッション (同一プロンプト含む) を 1 ステップずつ round-robin で進めても
    /// 全て単独実行と完全一致。P6 では全員が同時にスロットに常駐する。
    #[test]
    fn round_robin_stepping_matches_standalone_outputs() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        let prompts = ["t0", "t15", "t0", "t13"];
        let p = params(6);
        // 単独実行の基準
        let refs: Vec<ChatOutput> = prompts
            .iter()
            .map(|t| {
                let mut solo = kv_sensitive_engine(t, None).unwrap();
                solo.chat(None, &msgs(t), &p, |_| {}).unwrap()
            })
            .collect();
        assert_eq!(refs[0].text, refs[2].text);

        let mut e = kv_sensitive_engine("m", None).unwrap();
        let mut sessions: Vec<ChatSession> = prompts
            .iter()
            .map(|t| e.begin_chat(None, &msgs(t), &p).unwrap())
            .collect();

        // 全員の prefill を順に済ませる (install → prefill step。以後常駐し続ける)
        for s in sessions.iter_mut() {
            e.install_session(s).unwrap();
            assert!(!e.step_chat(s, &mut |_| {}).unwrap());
        }

        // round-robin で decode (セッションは全員常駐のまま)
        let mut done = vec![false; sessions.len()];
        for _ in 0..200 {
            let active: Vec<usize> = (0..sessions.len()).filter(|&i| !done[i]).collect();
            if active.is_empty() {
                break;
            }
            for &i in &active {
                done[i] = e.step_chat(&mut sessions[i], &mut |_| {}).unwrap();
            }
        }
        assert!(done.iter().all(|&d| d), "全セッションが完了していない");

        for (i, s) in sessions.iter_mut().enumerate() {
            e.release_session(s);
        }
        for (i, s) in sessions.into_iter().enumerate() {
            assert_eq!(s.into_output(), refs[i], "session {i} が単独実行と不一致");
        }
    }

    /// P6: 複数セッションの decode を step_chat_batch でまとめて進めても、
    /// 各セッションの出力は単独実行と完全一致 (fp32 バッチカーネルの契約)。
    #[test]
    fn batch_decode_matches_standalone_outputs() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        use crate::testutil::dummy_engine;
        let prompts = ["t0", "t15", "t0", "t13"];
        let p = params(6);
        // 単独実行の基準 (chat は step_chat ベースの単独経路)
        let refs: Vec<ChatOutput> = prompts
            .iter()
            .map(|t| {
                let mut solo = dummy_engine(t, None).unwrap();
                solo.chat(None, &msgs(t), &p, |_| {}).unwrap()
            })
            .collect();
        assert_eq!(refs[0].text, refs[2].text);

        let mut e = dummy_engine("m", None).unwrap();
        let mut sessions: Vec<ChatSession> = prompts
            .iter()
            .map(|t| e.begin_chat(None, &msgs(t), &p).unwrap())
            .collect();
        // 全員 prefill (常駐)
        for s in sessions.iter_mut() {
            e.install_session(s).unwrap();
            e.step_chat(s, &mut |_| {}).unwrap();
        }

        // バッチ decode: 完了したセッションは release、残りで継続
        loop {
            let mut cbs: Vec<Box<dyn FnMut(&str)>> =
                (0..sessions.len()).map(|_| Box::new(|_: &str| {}) as _).collect();
            let mut refs_sess: Vec<&mut ChatSession> =
                sessions.iter_mut().collect();
            let done = e.step_chat_batch(&mut refs_sess, &mut cbs).unwrap();
            let mut all = true;
            for (i, d) in done.iter().enumerate() {
                if *d {
                    e.release_session(&mut sessions[i]);
                } else {
                    all = false;
                }
            }
            if all {
                break;
            }
        }
        for (i, s) in sessions.into_iter().enumerate() {
            assert_eq!(s.into_output(), refs[i], "session {i} がバッチ実行と不一致");
        }
    }

    /// P6: バッチ途中でセッションが完了しても残りの継続セッションの
    /// 出力は変わらない (完了セッションは forward から外れる)。
    #[test]
    fn batch_decode_handles_uneven_completion() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        use crate::testutil::dummy_engine;
        let p = params(2); // t0 は 2 トークンで打ち切り (EOS 届く前)
        let p_long = params(6);
        // 基準: 短い側 / 長い側を単独実行
        let mut ref_short = dummy_engine("a", None).unwrap();
        let ref_s = ref_short.chat(None, &msgs("t0"), &p, |_| {}).unwrap();
        let mut ref_long = dummy_engine("b", None).unwrap();
        let ref_l = ref_long.chat(None, &msgs("t15"), &p_long, |_| {}).unwrap();

        let mut e = dummy_engine("m", None).unwrap();
        let mut s0 = e.begin_chat(None, &msgs("t0"), &p).unwrap();
        let mut s1 = e.begin_chat(None, &msgs("t15"), &p_long).unwrap();
        e.install_session(&mut s0).unwrap();
        e.install_session(&mut s1).unwrap();
        e.step_chat(&mut s0, &mut |_| {}).unwrap();
        e.step_chat(&mut s1, &mut |_| {}).unwrap();

        let mut done_any = false;
        for _ in 0..20 {
            let mut cbs: Vec<Box<dyn FnMut(&str)>> = (0..2)
                .map(|_| Box::new(|_: &str| {}) as _)
                .collect();
            let mut batch: Vec<&mut ChatSession> = vec![&mut s0, &mut s1];
            let done = e.step_chat_batch(&mut batch, &mut cbs).unwrap();
            if done[0] {
                e.release_session(&mut s0);
                done_any = true;
            }
            if done[1] {
                e.release_session(&mut s1);
                break;
            }
        }
        assert!(done_any, "短い側がバッチ途中で完了していること");
        assert_eq!(s0.into_output(), ref_s);
        assert_eq!(s1.into_output(), ref_l);
    }

    // ---------- s56 (T2): 反復抑止の追加分が両 decode 経路で効くこと ----------

    /// 反復抑止のパラメータ違いの GenParams (greedy = temperature 0 で決定論)。
    fn anti_params(max_tokens: usize, f: Option<f32>, pr: Option<f32>, n: Option<usize>) -> GenParams {
        GenParams {
            max_tokens,
            temperature: Some(0.0),
            frequency_penalty: f,
            presence_penalty: pr,
            no_repeat_ngram_size: n,
            ..Default::default()
        }
    }

    /// 既定 (未指定) の出力は従来とビット同一 — 逐次 (step_chat) と バッチ
    /// (step_chat_batch) の両経路で、no-op の適用関数を挟んでも変わらない。
    #[test]
    fn anti_repetition_default_is_bit_identical_in_both_decode_paths() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        use crate::testutil::dummy_engine;
        let base = params(20);

        // 逐次経路 (chat = begin_chat + step_chat)
        let mut a = dummy_engine("m", Some(9)).unwrap();
        let out_a = a.chat(None, &msgs("t0"), &base, |_| {}).unwrap();
        let mut b = dummy_engine("m", Some(9)).unwrap();
        let p_off = GenParams {
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            no_repeat_ngram_size: None,
            repetition_penalty: Some(1.0),
            ..base.clone()
        };
        let out_b = b.chat(None, &msgs("t0"), &p_off, |_| {}).unwrap();
        assert_eq!(out_a, out_b, "無効値指定で逐次経路の出力が変わった");
        assert_eq!(out_a.text, EXPECTED_GREEDY_TEXT);

        // バッチ経路 (step_chat_batch)
        let mut e1 = dummy_engine("m", Some(9)).unwrap();
        let mut e2 = dummy_engine("m", Some(9)).unwrap();
        // バッチ経路は「install → 逐次 step で prefill → step_chat_batch で decode」
        // (scheduler と同じ手順。prefill を batch に入れると logits が無い)
        let run_batch = |e: &mut Engine, p: &GenParams| -> ChatOutput {
            let mut s = e.begin_chat(None, &msgs("t0"), p).unwrap();
            e.install_session(&mut s).unwrap();
            e.step_chat(&mut s, &mut |_| {}).unwrap();
            for _ in 0..40 {
                let mut cbs: Vec<Box<dyn FnMut(&str)>> =
                    vec![Box::new(|_: &str| {}) as Box<dyn FnMut(&str)>];
                let mut batch: Vec<&mut ChatSession> = vec![&mut s];
                let done = e.step_chat_batch(&mut batch, &mut cbs).unwrap();
                if done[0] {
                    break;
                }
            }
            e.release_session(&mut s);
            s.into_output()
        };
        let ob1 = run_batch(&mut e1, &base);
        let ob2 = run_batch(&mut e2, &p_off);
        assert_eq!(ob1, ob2, "無効値指定でバッチ経路の出力が変わった");
        assert_eq!(ob1, out_a, "バッチと逐次で既定出力が一致しない");
    }

    /// no_repeat_ngram が **両方の decode 経路** で効くこと (片経路だけに入れていない)。
    ///
    /// ScriptModel のスクリプト [5,6,5,6,...]・n=2 の step ごとの文脈
    /// (プロンプト末尾は [UNK]=13):
    ///   step0 ctx=[13]        禁止なし → 5
    ///   step1 ctx=[13,5]      禁止なし → 6
    ///   step2 ctx=[13,5,6]    接頭辞 (5,6) の過去の続きなし → 5
    ///   step3 ctx=[13,5,6,5]  接頭辞 (6,5) は step0-1 に在り続き 6 → 6 を禁止
    ///                         → fallback 7 (準ピーク)
    /// つまり「n=2 が無いと 5,6,5,6 だが、有ると 5,6,5,7」。
    #[test]
    fn no_repeat_ngram_applies_in_both_decode_paths() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        use crate::testutil::tool_script_engine;
        let script = vec![5u32, 6, 5, 6, 5, 6, 5, 6];
        // ベースライン (n 無し): 5,6,5,6,... がそのまま出る
        let mut e0 = tool_script_engine("m", None, script.clone(), 7).unwrap();
        let out0 = e0
            .chat(None, &msgs("t0"), &anti_params(4, None, None, None), |_| {})
            .unwrap();
        assert_eq!(out0.text, "t5t6t5t6", "ベースライン (n 無し)");

        // 逐次
        let mut e1 = tool_script_engine("m", None, script.clone(), 7).unwrap();
        let out1 = e1
            .chat(None, &msgs("t0"), &anti_params(4, None, None, Some(2)), |_| {})
            .unwrap();
        assert_eq!(out1.text, "t5t6t5t7", "逐次経路で n=2 が効いていない");

        // バッチ (スクリプトモデルは単一セッション専用 → 1 本でバッチ経路を通す)
        let mut e2 = tool_script_engine("m", None, script, 7).unwrap();
        let p = anti_params(4, None, None, Some(2));
        let mut s = e2.begin_chat(None, &msgs("t0"), &p).unwrap();
        e2.install_session(&mut s).unwrap();
        e2.step_chat(&mut s, &mut |_| {}).unwrap(); // prefill
        for _ in 0..20 {
            let mut cbs: Vec<Box<dyn FnMut(&str)>> = vec![Box::new(|_: &str| {})];
            let mut batch: Vec<&mut ChatSession> = vec![&mut s];
            if e2.step_chat_batch(&mut batch, &mut cbs).unwrap()[0] {
                break;
            }
        }
        e2.release_session(&mut s);
        let out2 = s.into_output();
        assert_eq!(out2.text, out1.text, "バッチ経路だけ n=2 が効いていない");
    }

    /// GenParams の 3 フィールドがセッションに伝わる (正規化込み)。
    #[test]
    fn gen_params_anti_repetition_reach_session() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        use crate::testutil::dummy_engine;
        let mut e = dummy_engine("m", None).unwrap();
        let p = GenParams {
            max_tokens: 4,
            repetition_penalty: Some(1.1),
            frequency_penalty: Some(0.25),
            presence_penalty: Some(0.5),
            no_repeat_ngram_size: Some(3),
            ..Default::default()
        };
        let s = e.begin_chat(None, &msgs("t0"), &p).unwrap();
        assert_eq!(
            s.anti_rep,
            crate::sampler::AntiRepetition {
                repetition_penalty: Some(1.1),
                frequency_penalty: Some(0.25),
                presence_penalty: Some(0.5),
                no_repeat_ngram_size: Some(3),
            }
        );
        // n<=1 は無効に正規化される (内部防御)
        let p2 = GenParams { no_repeat_ngram_size: Some(1), ..p.clone() };
        let s2 = e.begin_chat(None, &msgs("t0"), &p2).unwrap();
        assert_eq!(s2.anti_rep.no_repeat_ngram_size, None);
        // 既定は no-op と認識される
        let s3 = e.begin_chat(None, &msgs("t0"), &GenParams::default()).unwrap();
        assert!(s3.anti_rep.is_noop());
    }

    #[test]
    fn resolve_kv_budget_scales_slots_by_reservation() {
        // P17e: スロット数 = 予算内に 4096tok 予約が収まる数 (上限 32)。
        // Qwen2.5-0.5B 相当 (bpt 24,576): 予約 100.7MB → 予算 12GB で上限 32
        let (slots, prefix) = resolve_kv_budget(32768, 24_576, None, None, 12288 * 1024 * 1024);
        assert_eq!(slots, 32);
        assert_eq!(prefix, 32768);
        // Qwen2.5-1.5B 相当 (bpt 57,344): 予約 235MB → 予算 12GB で 52 → 上限 32
        let (slots, prefix) = resolve_kv_budget(32768, 57_344, None, None, 12288 * 1024 * 1024);
        assert_eq!(slots, 32);
        assert_eq!(prefix, 32768);
    }

    #[test]
    fn resolve_kv_budget_shrinks_for_large_kv_models() {
        // Qwen3-0.6B 相当 (bpt ~229KB): 予約 939.5MB → 13 スロット、
        // 残り予算 (671MB) を prefix に充てる
        let (slots, prefix) =
            resolve_kv_budget(40960, 229_376, None, None, 12288 * 1024 * 1024);
        assert_eq!(slots, 13);
        assert_eq!(prefix, 2925);
        // 中間 (bpt 114,688): 予約 469.8MB → 27 スロット。残り予算は 1755 トークン
        // 分だが prefix の floor (2048) が下限として効く
        let (slots, prefix) = resolve_kv_budget(32768, 114_688, None, None, 12288 * 1024 * 1024);
        assert_eq!(slots, 27);
        assert_eq!(prefix, 2048);
        // 予算 0 でも最低 1 スロット + floor 2048 は確保される
        let (slots, prefix) = resolve_kv_budget(40960, 229_376, None, None, 0);
        assert_eq!(slots, 1);
        assert_eq!(prefix, 2048);
    }

    #[test]
    fn resolve_kv_budget_respects_explicit_env() {
        // 明示指定は予算より優先
        let (slots, prefix) = resolve_kv_budget(40960, 229_376, Some(2), Some(1000), 0);
        assert_eq!(slots, 2);
        assert_eq!(prefix, 1000);
        // スロット 0 指定は 1 にクランプ
        let (slots, _) = resolve_kv_budget(32768, 24_576, Some(0), None, 12288 * 1024 * 1024);
        assert_eq!(slots, 1);
    }

    #[test]
    fn resolve_kv_budget_qwen38_27b_scale() {
        // A3: Qwen3.8-27B 実構成 (64 層 = 48 GDN + 16 full、kv_heads 4 ×
        // head_dim 256)。KV を使うのは full attention 16 層のみなので
        // bytes_per_token = 16 × 1024 × 2 × 4 = 131,072 B/token。
        // (A3 で GDN 層の KV バッファ確保を廃止する前は 64 層分の 524,288
        // B/token で計上され、4 倍に過大評価していた)
        let bpt = 16 * (4 * 256) * 2 * 4;
        assert_eq!(bpt, 131_072);
        // 予約 4096 tok × 128KB = 512MB/slot → 12GB / 512MB = 24 スロット (上限 32)
        let (slots, prefix) = resolve_kv_budget(262_144, bpt, None, None, 12288 * 1024 * 1024);
        assert_eq!(slots, 24);
        // 24 スロットの予約で予算を使い切るため prefix は floor (2048) が効く
        assert_eq!(prefix, 2048);
        // 単一リクエストは 予算 / bpt = 98,304 トークンまで (32K prefill は 4GB)
        let need_32k = 32768 * bpt;
        assert_eq!(need_32k, 4 * 1024 * 1024 * 1024);
    }

    // ---------- s53: 停止条件の強化 (ロールラベル停止・holdback・stop 併用) ----------

    /// スクリプト + トークナイザで生成を回し、送出 delta の列も返す。
    /// (ScriptModel は 1 セッション専用・決定論的)
    fn run_script_streamed(
        script: &[u32],
        extra_vocab: &[&str],
        stop_sequences: Vec<String>,
    ) -> (ChatOutput, Vec<String>) {
        use std::cell::RefCell;
        let tok = crate::testutil::tool_tokenizer_with(extra_vocab).unwrap();
        let model = Box::new(crate::testutil::ScriptModel::new(script.to_vec(), tid("t0")));
        let mut e = Engine::from_parts(model, tok, "m", None);
        let deltas = RefCell::new(Vec::<String>::new());
        let mut p = params(64);
        p.stop_sequences = stop_sequences;
        let mut sess = e.begin_chat(None, &msgs("t0"), &p).unwrap();
        e.install_session(&mut sess).unwrap();
        while !e
            .step_chat(&mut sess, &mut |d: &str| deltas.borrow_mut().push(d.to_string()))
            .unwrap()
        {}
        e.release_session(&mut sess);
        (sess.into_output(), deltas.into_inner())
    }

    /// s53: 追加語彙 (40 以降) + 固定語彙の piece → id 逆引き。
    fn vb(pieces: &[&str], piece: &str) -> u32 {
        if let Some((_, id)) = crate::testutil::TOOL_VOCAB.iter().find(|(p, _)| *p == piece) {
            return *id;
        }
        let mut next = 40u32;
        for p in pieces {
            let known = crate::testutil::TOOL_VOCAB.iter().any(|(t, _)| t == p);
            if known {
                continue;
            }
            if *p == piece {
                return next;
            }
            next += 1;
        }
        panic!("vocab missing: {piece}");
    }

    /// piece 列 + 追加語彙 から (id 列, 追加語彙) を作る (重複順で 40 以降)。
    fn script_extra<'a>(pieces: &[&'a str]) -> (Vec<u32>, Vec<&'a str>) {
        let mut extra: Vec<&str> = Vec::new();
        for p in pieces {
            if crate::testutil::TOOL_VOCAB.iter().any(|(t, _)| t == p) {
                continue;
            }
            if !extra.contains(p) {
                extra.push(p);
            }
        }
        let ids = pieces.iter().map(|p| vb(&extra, p)).collect();
        (ids, extra)
    }

    #[test]
    fn role_label_stops_and_excludes_marker() {
        // s163 (K4): run_script_streamed 経由で forward を回す (カウンタ窓と排他)。
        let _g = crate::prof::counters_guard();
        // 生成中に "\nuser\n" が現れたらマーカーを含めず打ち切る
        let (ids, extra) = script_extra(&["Hello", "Ċ", "user", "Ċ", "next", "eos"]);
        let out = run_script_streamed(&ids, &extra, Vec::new()).0;
        // 直後の改行も「ラベル行の開始 (prefix)」として holdback 中に破棄される
        assert_eq!(out.text, "Hello\n");
        assert_eq!(out.stop_reason, StopReason::StopSequence);
        assert_eq!(out.stop_sequence.as_deref(), Some("user"));
    }

    #[test]
    fn role_label_stops_on_assistant_colon_and_im_end() {
        // s163 (K4): run_script_streamed 経由で forward を回す (カウンタ窓と排他)。
        let _g = crate::prof::counters_guard();
        let (ids, extra) = script_extra(&["A", "Ċ", "assistant", ":", "Ċ", "B", "eos"]);
        let out = run_script_streamed(&ids, &extra, Vec::new()).0;
        assert_eq!(out.text, "A\n");
        assert_eq!(out.stop_reason, StopReason::StopSequence);
        assert_eq!(out.stop_sequence.as_deref(), Some("assistant"));

        // 特殊トークン文字列 <|im_end|> は行頭以外でも即停止
        let (ids2, extra2) = script_extra(&["AB", "<|im_end|>", "C", "eos"]);
        let out2 = run_script_streamed(&ids2, &extra2, Vec::new()).0;
        assert_eq!(out2.text, "AB");
        assert_eq!(out2.stop_reason, StopReason::StopSequence);
        assert_eq!(out2.stop_sequence.as_deref(), Some("<|im_end|>"));
    }

    #[test]
    fn partial_marker_is_not_streamed_until_confirmed() {
        // s163 (K4): run_script_streamed 経由で forward を回す (カウンタ窓と排他)。
        let _g = crate::prof::counters_guard();
        // "Hello" → "\n" → "user" (ここで確定 = 打ち切り) の間、
        // on_delta に "u" / "us" / "use" 等の prefix が漏れない。
        let (ids, extra) = script_extra(&["Hello", "Ċ", "user", "Ċ", "next", "eos"]);
        let (out, deltas) = run_script_streamed(&ids, &extra, Vec::new());
        let streamed: String = deltas.concat();
        assert_eq!(streamed, out.text);
        assert_eq!(streamed, "Hello\n");
        // 不変条件: 終了時 sent_len == text.len() (送出漏れなし)
        assert!(!streamed.contains('u'), "holdback leaked: {streamed:?}");
        assert!(!streamed.contains("user"), "marker leaked: {streamed:?}");

        // 打ち切られない通常応答では holdback 分も最終的に全て送出される
        let (ids2, extra2) = script_extra(&["Hello", "Ċ", "userland", "Ċ", "ok", "eos"]);
        let (out2, deltas2) = run_script_streamed(&ids2, &extra2, Vec::new());
        assert_eq!(out2.stop_reason, StopReason::EndTurn);
        assert_eq!(deltas2.concat(), out2.text);
        assert_eq!(out2.text, "Hello\nuserland\nok");
    }

    #[test]
    fn stop_marker_boundaries_crlf_and_fullwidth_colon() {
        // s163 (K4): run_script_streamed 経由で forward を回す (カウンタ窓と排他)。
        let _g = crate::prof::counters_guard();
        // MAX_STOP_MARKER_LEN (12) 境界: 11 バイトの prefix は保持・12 バイトで確定
        assert_eq!(MAX_STOP_MARKER_LEN, 12);
        // MAX_STOP_MARKER_LEN (12) の保持上限: 11 / 12 バイトは prefix で保持、
        // 13 バイト目が混ざると prefix でなくなる (= 保持しない)
        assert_eq!(partial_holdback("<|im_start|"), Some(0)); // 11 バイト
        assert_eq!(overall_stop_status("<|im_end", true), StopStatus::Partial(0)); // 8 バイト
        // "zzz qqq" は特殊トークン文字列でもラベル prefix でもない → 保持しない
        assert_eq!(overall_stop_status("zzz qqq", true), StopStatus::Clear);
        // 長さ 13 バイト以上でも「マーカー prefix で始まる最終行」なら保持される
        assert_eq!(overall_stop_status("abc xyz <|im_", true), StopStatus::Partial(0));
        assert_eq!(overall_stop_status("abc\n<|im_en", true), StopStatus::Partial(4));
        // prefix の判定は「行 (= 最終行) の先頭がマーカー開始か」で行う
        // (本文の途中から始まるマーカーは対象外・実際の停止判定は
        //  find_stop_marker / Matched 経路が行う)
        assert_eq!(partial_holdback("<|im_st"), Some(0));
                assert_eq!(
            overall_stop_status("a\nassistant:\n", true),
            StopStatus::Matched(2, "assistant".into())
        );

        // CRLF 行の "user" (tokenizer の ByteLevel デコーダは "\r" を落とすため
        // テキスト上は "A\nuser\n..." になる)。判定は find_stop_marker 単体でも確認。
        assert_eq!(
            find_stop_marker("A\n   user:\r\nB", 0),
            Some((2, "user".to_string()))
        );
        // s53b 回帰: 未送出境界 (start) が単語の途中にある場合、その語を行頭
        // ラベルと誤認しない (s53 の誤爆: "system" で 23-44 トークン切断)。
        assert_eq!(
            find_stop_marker("Constraint (from the system prompt)", 20),
            None
        );
        assert_eq!(find_stop_marker("abc\nsystem prompt)", 4), None);
        // 真の行頭 (start より後) に現れるラベル行は従来どおり停止する。
        assert_eq!(
            find_stop_marker("abc\nsystem\n", 4),
            Some((4, "system".to_string()))
        );
        // 行頭でも「ラベル行」でないもの (後続に本文が続く) は停止しない。
        assert_eq!(find_stop_marker("abc\nsystem\nnext", 4), None);
        // start より前の行にあるラベルは再判定されない (二重停止の防止)。
        assert_eq!(find_stop_marker("user:hello\nworld", 5), None);
        let (ids, extra) = script_extra(&["A", "Ċ", "user\r", "Ċ", "B", "eos"]);
        let out = run_script_streamed(&ids, &extra, Vec::new()).0;
        assert_eq!(out.text, "A\n");
        assert_eq!(out.stop_sequence.as_deref(), Some("user"));

        // 全角コロン "："
        let (ids2, extra2) = script_extra(&["A", "Ċ", "system：", "Ċ", "B", "eos"]);
        let out2 = run_script_streamed(&ids2, &extra2, Vec::new()).0;
        assert_eq!(out2.text, "A\n");
        assert_eq!(out2.stop_sequence.as_deref(), Some("system"));

        // 行頭でない "user" は停止しない (行末でない "user" も確定しない)
        let (ids3, extra3) = script_extra(&["trustee", "Ċ", "trusting", "eos"]);
        let out3 = run_script_streamed(&ids3, &extra3, Vec::new()).0;
        assert_eq!(out3.stop_reason, StopReason::EndTurn);
        assert_eq!(out3.text, "trustee\ntrusting");
    }

    #[test]
    fn client_stop_combines_with_role_label_earliest_wins() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        // クライアント stop が先に現れる → そちらで打ち切る
        let (ids, extra) = script_extra(&["He", "llo", "Ċ", "user", "Ċ", "eos"]);
        let out = run_script_streamed(&ids, &extra, vec!["llo".into()]).0;
        assert_eq!(out.text, "He");
        assert_eq!(out.stop_reason, StopReason::StopSequence);
        assert_eq!(out.stop_sequence.as_deref(), Some("llo"));

        // ロールラベルが先に現れる → そちらで打ち切る
        let (ids2, extra2) = script_extra(&["Hi", "Ċ", "user", "Ċ", "llo", "eos"]);
        let out2 = run_script_streamed(&ids2, &extra2, vec!["llo".into()]).0;
        assert_eq!(out2.text, "Hi\n");
        assert_eq!(out2.stop_reason, StopReason::StopSequence);
        assert_eq!(out2.stop_sequence.as_deref(), Some("user"));

        // 空文字列 stop は無視する (即停止・無限ループにならない)
        let (ids3, extra3) = script_extra(&["Hello", "eos"]);
        let out3 = run_script_streamed(&ids3, &extra3, vec!["".into()]).0;
        assert_eq!(out3.text, "Hello");
        assert_eq!(out3.stop_reason, StopReason::EndTurn);

        // 複数 stop は最早一致
        let (ids4, extra4) = script_extra(&["abc def", "eos"]);
        let out4 = run_script_streamed(&ids4, &extra4, vec!["def".into(), "c d".into()]).0;
        assert_eq!(out4.stop_sequence.as_deref(), Some("c d"));
    }

    // ---------- 第21: tool_choice 拘束デコード (engine 経路) ----------

    /// TOOL_VOCAB の piece → id (スクリプト可読性のための逆引き)。
    fn tid(piece: &str) -> u32 {
        crate::testutil::TOOL_VOCAB
            .iter()
            .find(|(p, _)| *p == piece)
            .map(|(_, id)| *id)
            .unwrap_or_else(|| panic!("vocab missing: {piece}"))
    }

    /// 完全な tool_call block を吐くスクリプト (第27: 閉じタグ `</tool_call>` 付き = 抽出対象)。
    /// `name` はモデルが「選ぼうとする」ツール名 (拘束の有無で分岐を見る)。
    fn block_script(name: &str) -> Vec<u32> {
        [
            "Checking.", "<tool_call>", "Ċ", "<function=", name, ">", "city", "Tokyo",
            "</parameter>", "</function>", "</tool_call>", "eos",
        ]
        .iter()
        .map(|p| tid(p))
        .collect()
    }

    /// ScriptModel + tool_tokenizer で begin_chat_with_tools → step まで回す
    /// (scheduler build_run と同一の呼び出し経路)。fallback は拘束で禁止された
    /// ピークの代わりに選ばれる準ピーク。
    fn run_script(
        script: &[u32],
        fallback: u32,
        constraint: Option<&crate::toolmask::ToolConstraint>,
    ) -> ChatOutput {
        let mut e =
            crate::testutil::tool_script_engine("m", None, script.to_vec(), fallback).unwrap();
        let mut sess = e
            .begin_chat_with_tools(None, &msgs("t0"), &params(40), None, constraint, None)
            .unwrap();
        e.install_session(&mut sess).unwrap();
        while !e.step_chat(&mut sess, &mut |_| {}).unwrap() {}
        e.release_session(&mut sess);
        sess.into_output()
    }

    #[test]
    fn tool_mask_off_matches_plain_generation() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        // 拘束なし = スクリプトをそのまま再生
        let out = run_script(&block_script("run_tool"), tid("city"), None);
        assert!(out.text.contains("<function=run_tool>"), "text = {}", out.text);
        assert_eq!(out.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn tool_mask_any_is_bit_identical_when_compliant() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        // スクリプトが許容集合に完全準拠なら拘束ありでも無しと同一出力
        let a = run_script(&block_script("get_weather"), tid("city"), None);
        let b = run_script(
            &block_script("get_weather"),
            tid("city"),
            Some(&crate::toolmask::ToolConstraint {
                names: vec!["get_weather".into(), "run_tool".into()],
            }),
        );
        assert_eq!(a.text, b.text);
        assert_eq!(a.stop_reason, b.stop_reason);
    }

    #[test]
    fn tool_mask_named_forces_allowed_tool_name() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        // モデルが指定外ツール名 (run_tool) をピークに出しても、
        // 拘束により許容名 (get_weather) へ強制される
        let forced = run_script(
            &block_script("run_tool"),
            tid("get_weather"),
            Some(&crate::toolmask::ToolConstraint { names: vec!["get_weather".into()] }),
        );
        assert!(forced.text.contains("<function=get_weather>"), "text = {}", forced.text);
        assert!(!forced.text.contains("run_tool"), "text = {}", forced.text);
        assert_eq!(forced.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn tool_mask_multibyte_name_split_tokens_pass() {
        // s163 (K4): prof のカウンタ窓と排他 (同じグローバルカウンタを汚し得る)。
        let _g = crate::prof::counters_guard();
        // マルチバイト名 (天気) を 1 文字ずつのトークン (天/気) で構築しても
        // byte-level 前提の許容計算で通る
        let script = vec![
            tid("Checking."), tid("<tool_call>"), tid("Ċ"), tid("<function="),
            tid("天"), tid("気"), tid(">"), tid("Tokyo"), tid("</tool_call>"), tid("eos"),
        ];
        let out = run_script(
            &script,
            tid("city"),
            Some(&crate::toolmask::ToolConstraint { names: vec!["天気".into()] }),
        );
        assert!(out.text.contains("<function=天気>"), "text = {}", out.text);
        assert_eq!(out.stop_reason, StopReason::EndTurn);
    }

    // ===== P4-4: 長文脈 (LONG_CONTEXT = 4096) の回帰 =====
    //
    // 既定ゲートの他のテストはすべて testutil::DUMMY_CONTEXT = 64 で回るため、
    // 「初期容量を超える KV 成長」「長い prompt での prefix cache 予算/退避」
    // 「コンテキスト容量の境界」は既定ゲートで一度も実行されていなかった。
    // 本番は 3,020 token の prefill とマルチターン会話を配信している。

    use crate::testutil::{long_context_engine, KvSensitiveModel, LONG_CONTEXT};

    /// 長文脈テスト用のプロンプト本文。fallback テンプレート
    /// ("user: <content>\n\nassistant:") は content を空白区切りの 1 語 1 token に
    /// し、"user:" と "assistant:" がそれぞれ [UNK] 2 token になるため、
    /// プロンプト長 = words + 4 (実測: ids = [20, 20, t1..., 20, 20])。
    fn long_content(words: usize, word: &str) -> String {
        let mut s = String::with_capacity(words * 3);
        for _ in 0..words {
            s.push_str(word);
            s.push(' ');
        }
        s
    }

    /// プロンプト長 -> content の語数 (上の式の逆。テストの前提を 1 箇所に固定する)
    fn long_prompt_words(len: usize) -> usize {
        assert!(len >= 4, "プロンプト長 {len} は短すぎる");
        len - 4
    }

    /// content の語数 -> プロンプト長
    fn long_prompt_len(words: usize) -> usize {
        words + 4
    }

    /// P4-4 (a): 長い prompt でも prefix cache の再利用は数値的に無害。
    /// 3,000 token の同一 prompt を「cold (full prefill)」と「同一エンジンで
    /// 2 回目 (prefix cache 再利用)」で実行し、prefill 直後の logits が
    /// **ビット一致**することを f32 のビット列で比較する (epsilon 比較ではない)。
    /// 同じ性質は既定ゲートで 64 token 以下でしか見ていなかった。
    #[test]
    fn long_context_prefix_cache_reuse_keeps_logits_bit_identical() {
        // s163 (Task K4): 3,000 token の forward は prof の厳密比較の窓を広げるため、
        // prof::tests と同じ 1 本のロックを取って双方向に排他する。
        let _g = crate::prof::counters_guard();
        let words = long_prompt_words(3000);
        let content = long_content(words, "t1");
        let p = params(1);
        let mut e = long_context_engine("long", None).unwrap();

        // 1 回目 (cold): full prefill。prefill 直後の logits を退避する
        let mut s1 = e.begin_chat(None, &msgs(&content), &p).unwrap();
        assert_eq!(s1.input_tokens(), long_prompt_len(words), "前提: プロンプト長");
        e.install_session(&mut s1).unwrap();
        assert!(!e.step_chat(&mut s1, &mut |_| {}).unwrap(), "prefill では完了しない");
        let cold = s1.logits.clone();
        assert_eq!(e.last_prefix_hit(), PrefixHit::None, "1 回目は full prefill");
        let slot = s1.slot.expect("install 済み");
        assert_eq!(e.caches[slot].len(), long_prompt_len(words));
        while !e.step_chat(&mut s1, &mut |_| {}).unwrap() {}
        e.release_session(&mut s1);

        // 2 回目 (同一 prompt・同一エンジン): 直前の KV を再利用する
        let mut s2 = e.begin_chat(None, &msgs(&content), &p).unwrap();
        e.install_session(&mut s2).unwrap();
        assert!(
            matches!(e.last_prefix_hit(), PrefixHit::Live | PrefixHit::Entry),
            "2 回目は prefix cache 経路であるべき (実際: {:?})",
            e.last_prefix_hit()
        );
        assert!(!e.step_chat(&mut s2, &mut |_| {}).unwrap());
        let warm = s2.logits.clone();
        assert_eq!(cold.len(), warm.len());
        for (i, (a, b)) in cold.iter().zip(warm.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "logits[{i}]: cold {a} vs prefix cache 再利用 {b} (ビット不一致)"
            );
        }
    }

    /// P4-4 (b): 容量境界は panic ではなくきれいなエラー。
    /// max_len (4096) ちょうどの prefill は受理され (最後の合法な append)、
    /// その 1 token 超の prompt は `begin_chat` が Err を返す — panic せず、
    /// 既存スロットの KV も書き換えない。kv_cache.rs の assert
    /// ("KV cache overflow") は engine が手前で弾くため到達しない。
    #[test]
    fn long_context_capacity_boundary_is_a_clean_error_not_a_panic() {
        // s163 (Task K4): prof のカウンタ窓と排他 (詳細は 1 本目のテスト)。
        let _g = crate::prof::counters_guard();
        let mut e = long_context_engine("long", None).unwrap();
        let max_len = e.caches[0].max_len();
        assert_eq!(max_len, LONG_CONTEXT);

        // (1) max_len - 1 token の prompt + 1 token 生成 = 境界の内側
        let words = long_prompt_words(max_len - 1);
        let content = long_content(words, "t1");
        let out = e.chat(None, &msgs(&content), &params(1), |_| {}).unwrap();
        assert_eq!(out.input_tokens, long_prompt_len(words));
        assert_eq!(e.caches[0].len(), max_len - 1);

        // (2) max_len ちょうど: 最後の合法な append が受理される
        //     (max_tokens は begin_chat が context - prompt にクランプする = 0)
        let words_full = long_prompt_words(max_len);
        let content_full = long_content(words_full, "t1");
        let out_full = e.chat(None, &msgs(&content_full), &params(4), |_| {}).unwrap();
        assert_eq!(out_full.input_tokens, max_len);
        assert_eq!(out_full.output_tokens, 0, "max_tokens は 0 にクランプされる");
        assert_eq!(e.caches[0].len(), max_len);
        assert_eq!(e.caches[0].max_len(), max_len);

        // (3) max_len + 1 token: Err (panic しない)・KV は不変
        let len_before = e.caches[0].len();
        let over = long_content(long_prompt_words(max_len + 1), "t1"); // prompt 長 = max_len + 1
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            e.begin_chat(None, &msgs(&over), &params(1))
        }));
        let err = match r.expect("容量超過で panic してはならない") {
            Ok(_) => panic!("容量超過は Err であるべき"),
            Err(err) => err,
        };
        assert!(
            format!("{err}").contains("コンテキスト長"),
            "エラー内容が想定と違う: {err}"
        );
        assert_eq!(e.caches[0].len(), len_before, "拒否時に KV を書き換えない");
        assert_eq!(e.caches[0].max_len(), max_len);
    }

    /// P4-4 (c): 成長は需要ベースで、既存行を壊さない。
    /// 3,000 token の prefill 直後は容量がちょうど 3,000 (max_len 4096 を
    /// 先取りしない) で、モデルが書いた行が式どおり読める。続けて同じ接頭辞を
    /// 4,000 token に伸ばすと倍々成長 (3000 -> 4096) が起き、realloc 後も
    /// 先頭 2,999 行がビット一致で残る。
    #[test]
    fn long_context_kv_growth_is_demand_driven_and_preserves_rows() {
        // s163 (Task K4): prof のカウンタ窓と排他 (詳細は 1 本目のテスト)。
        let _g = crate::prof::counters_guard();
        let mut e = long_context_engine("long", None).unwrap();
        let kv_dim = e.caches[0].kv_dim();
        let first_words = long_prompt_words(3000);
        let first = long_content(first_words, "t1");
        let p0 = params(0); // prefill のみ (decode で行を足さない)
        let out1 = e.chat(None, &msgs(&first), &p0, |_| {}).unwrap();
        assert_eq!(out1.input_tokens, long_prompt_len(first_words));

        let prompt1 = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &msgs(&first), None, None).unwrap())
            .unwrap();
        assert_eq!(prompt1.len(), 3000, "前提: 3,000 token の prompt");

        // (i) 需要ベース: 必要な 3,000 行だけ確保し、max_len を先取りしない
        assert_eq!(e.caches[0].len(), 3000);
        assert_eq!(e.caches[0].capacity(), 3000, "必要な行数ちょうどの確保");
        assert!(
            e.caches[0].capacity() < e.caches[0].max_len(),
            "コンテキスト全幅を事前確保してはならない"
        );
        // モデルが書いた行 (先頭・中間・末尾) が式どおりであること
        {
            let (k, v) = e.caches[0].kv_bufs_ref(0);
            for &pos in &[0usize, 1, 1000, 2999] {
                let t = prompt1[pos];
                for d in 0..kv_dim {
                    assert_eq!(
                        k[pos * kv_dim + d].to_bits(),
                        KvSensitiveModel::kv_k(t, pos, d, kv_dim).to_bits(),
                        "K 行 {pos} 列 {d}"
                    );
                    assert_eq!(
                        v[pos * kv_dim + d].to_bits(),
                        KvSensitiveModel::kv_v(t, pos, d, kv_dim).to_bits(),
                        "V 行 {pos} 列 {d}"
                    );
                }
            }
        }

        // (ii) 同じ接頭辞を 4,000 token に伸ばす -> 倍々成長 + realloc コピー
        let extended = format!("{}{}", first, long_content(1000, "t2"));
        let out2 = e.chat(None, &msgs(&extended), &p0, |_| {}).unwrap();
        assert_eq!(out2.input_tokens, 4000);
        assert_eq!(e.caches[0].len(), 4000);
        assert_eq!(
            e.caches[0].capacity(),
            LONG_CONTEXT,
            "3000 -> 4096 の倍々成長 (需要ベース)"
        );
        let prompt2 = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &msgs(&extended), None, None).unwrap())
            .unwrap();
        assert_eq!(prompt2.len(), 4000);
        // 共通接頭辞は「先頭 2 token ([UNK][UNK]) + t1 x 2,996」= 2,998。
        // 3,000 行目以降 (prompt1 の末尾 [UNK][UNK]) は上書きされる。
        assert_eq!(prompt1[..2998], prompt2[..2998], "前提: 先頭 2,998 token が共通接頭辞");
        assert_ne!(prompt1[2998], prompt2[2998], "前提: 2,998 行目で分岐する");
        // 成長前に書かれた行が realloc 後も同じ値で読めること
        {
            let (k, v) = e.caches[0].kv_bufs_ref(0);
            for &pos in &[0usize, 1, 1000, 2997] {
                let t = prompt1[pos];
                for d in 0..kv_dim {
                    assert_eq!(
                        k[pos * kv_dim + d].to_bits(),
                        KvSensitiveModel::kv_k(t, pos, d, kv_dim).to_bits(),
                        "成長後 K 行 {pos} 列 {d}"
                    );
                    assert_eq!(
                        v[pos * kv_dim + d].to_bits(),
                        KvSensitiveModel::kv_v(t, pos, d, kv_dim).to_bits(),
                        "成長後 V 行 {pos} 列 {d}"
                    );
                }
            }
        }
    }

    /// P4-4 (d): 長い prompt での prefix cache 予算。1 エントリだけが収まる
    /// 予算で 2 本目の長い prompt を入れると、予算超過ではなく LRU 退避が起きる。
    /// 退避された prompt を再実行するとエントリ復元は起きず (miss)、logits は
    /// full prefill と **ビット一致**になる (半端に復元されたキャッシュは無い)。
    /// prefix_cache.rs の単体テストは同じ算術を短い prompt で見ているが、
    /// 3,000 token 級 + engine 経路 (スロット残留 KV との競合を含む) は未覆。
    #[test]
    fn long_context_prefix_budget_evicts_and_post_eviction_is_a_clean_miss() {
        // s163 (Task K4): prof のカウンタ窓と排他 (詳細は 1 本目のテスト)。
        let _g = crate::prof::counters_guard();
        let words = long_prompt_words(3000);
        let prompt_len = long_prompt_len(words); // 3,000
        let budget = prompt_len + prompt_len / 2; // 1 本は入り 2 本は入らない
        let p0 = params(0);
        let a = long_content(words, "t1");
        let b = long_content(words, "t2");

        let mut e = long_context_engine("long", None).unwrap();
        e.set_prefix_cache_max_tokens(budget);
        assert_eq!(e.prefix_cache_max_tokens(), budget);

        let out_a = e.chat(None, &msgs(&a), &p0, |_| {}).unwrap();
        assert_eq!(e.prefix_cache_entries(), 1);
        assert_eq!(e.prefix_cache.total_weight(), prompt_len);
        assert!(e.prefix_cache.total_weight() <= budget);

        // 2 本目: 合計 6,000 > 4,500 -> LRU (A) が退避される
        let out_b = e.chat(None, &msgs(&b), &p0, |_| {}).unwrap();
        assert_eq!(e.prefix_cache_entries(), 1, "2 本目で予算超過せず退避する");
        assert_eq!(e.prefix_cache.total_weight(), prompt_len);
        assert!(e.prefix_cache.total_weight() <= budget, "予算を超えない");
        let ids_a = e
            .tok
            .encode(&e.tok.apply_chat_template(None, &msgs(&a), None, None).unwrap())
            .unwrap();
        assert!(!e.prefix_cache.contains_prompt(&ids_a), "A は退避済み");

        // 退避された A の再実行: エントリ復元は起きず (miss)、数値は full prefill と同一
        let mut s = e.begin_chat(None, &msgs(&a), &p0).unwrap();
        e.install_session(&mut s).unwrap();
        assert_ne!(
            e.last_prefix_hit(),
            PrefixHit::Entry,
            "退避済みエントリからの復元はあり得ない"
        );
        assert!(!e.step_chat(&mut s, &mut |_| {}).unwrap());
        let got = s.logits.clone();
        e.release_session(&mut s);

        let mut fresh = long_context_engine("long-ref", None).unwrap();
        let mut r = fresh.begin_chat(None, &msgs(&a), &p0).unwrap();
        fresh.install_session(&mut r).unwrap();
        assert_eq!(fresh.last_prefix_hit(), PrefixHit::None);
        assert!(!fresh.step_chat(&mut r, &mut |_| {}).unwrap());
        let want = r.logits.clone();
        assert_eq!(got.len(), want.len());
        for (i, (x, y)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "logits[{i}]: 退避後 {x} vs full prefill {y} (ビット不一致)"
            );
        }
        assert_eq!(out_a.input_tokens, prompt_len);
        assert_eq!(out_b.input_tokens, prompt_len);
    }
}
