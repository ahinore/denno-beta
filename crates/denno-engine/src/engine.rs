//! エンジン本体: モデル+トークナイザ+KVキャッシュ+サンプラを束ね、
//! prefill → decode ループ(stop_sequences / EOS / max_tokens 処理)を実行する。

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::model::{load_model, Model};
use crate::prefix_cache::PrefixCache;
use crate::quant::QuantMode;
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
    pub cache: KvCache,
    pub name: String,
    pub default_seed: Option<u64>,
    /// KVキャッシュに入っているプロンプトトークン列 (live prefix cache 用)。
    /// cached_prompt[i] の KV は cache の i 行目にある。
    /// (生成トークンの KV も cache には残るが cached_prompt には含めない)
    cached_prompt: Vec<u32>,
    /// 過去会話の KV スナップショット群 (マルチエントリ LRU prefix cache)
    prefix_cache: PrefixCache,
    /// 直前のリクエストでの prefix cache ヒット状況
    last_prefix_hit: PrefixHit,
}

/// P4 ステッパ: 1リクエスト分の生成状態。
///
/// 単一の KvCache を複数リクエストで時分割するため、実行中リクエストを
/// [`Engine::suspend_session`] で KV キャッシュから退避し (KV スナップショットを
/// この構造体に保持)、別リクエストの prefill/decode を割り込ませてから
/// [`Engine::install_session`] で復元できる。
/// KV の各行はトークン接頭辞のみに依存 (causal attention) するため、
/// 退避/復元は bit 完全に可逆 → インターリーブ下でも各リクエストの 出力は
/// 単独実行と完全一致する。
pub struct ChatSession {
    prompt_ids: Vec<u32>,
    /// prefill で再利用した prefix 長 (KV に既にある先頭トークン数)
    lcp: usize,
    prefill_done: bool,
    /// 直近の forward の logits (最終位置)
    logits: Vec<f32>,
    generated: Vec<u32>,
    text: String,
    max_tokens: usize,
    stop_sequences: Vec<String>,
    sampler: Sampler,
    stop_reason: StopReason,
    stop_sequence: Option<String>,
    /// suspend 中の KV スナップショット (None = 未 suspend)
    kv: Option<SuspendedKv>,
    input_tokens: usize,
}

/// suspend 中の KV スナップショット (KvCache::snapshot と同じ形式)。
/// 行 i はトークン列 prompt (+生成済み分) の位置 i の KV。
pub struct SuspendedKv {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    n: usize,
}

impl ChatSession {
    /// プロンプトのトークン数 (SSE message_start の usage 用)
    pub fn input_tokens(&self) -> usize {
        self.input_tokens
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
        let cache = model.new_cache();
        // マルチエントリ prefix cache のトークン上限。
        // デフォルトは KvCache の max_len (コンテキスト長) と同じ (Qwen2.5-0.5B なら 32k)。
        let max_tokens = std::env::var(PREFIX_CACHE_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| cache.max_len());
        tracing::info!(
            "prefix cache: multi-entry LRU, budget = {} tokens (~{:.0} MB KV)",
            max_tokens,
            max_tokens as f64 * cache.bytes_per_token() as f64 / 1e6
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
        Ok(Self {
            model,
            tok,
            cache,
            name: name.into(),
            default_seed,
            cached_prompt: Vec::new(),
            prefix_cache: PrefixCache::new(max_tokens),
            last_prefix_hit: PrefixHit::None,
        })
    }

    pub fn context(&self) -> usize {
        self.model.context()
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
        let cache = model.new_cache();
        let max_tokens = cache.max_len();
        Self {
            model,
            tok,
            cache,
            name: name.into(),
            default_seed,
            cached_prompt: Vec::new(),
            prefix_cache: PrefixCache::new(max_tokens),
            last_prefix_hit: PrefixHit::None,
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

    /// live (直前リクエスト) の KV をスナップショットとして取り出す。
    /// cache には prompt + 生成トークン分の行があり、cached_prompt はその接頭辞。
    fn snapshot_live(&self) -> Option<(Vec<u32>, Vec<Vec<f32>>, Vec<Vec<f32>>, usize)> {
        if self.cached_prompt.is_empty() || self.cache.is_empty() {
            return None;
        }
        let kv_len = self.cache.len();
        let (k, v) = self.cache.snapshot(kv_len);
        Some((self.cached_prompt.clone(), k, v, kv_len))
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
        let prompt = self.tok.apply_chat_template(system, messages)?;
        let prompt_ids = self.tok.encode(&prompt)?;

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

        Ok(ChatSession {
            input_tokens: prompt_ids.len(),
            prompt_ids,
            lcp: 0,
            prefill_done: false,
            logits: Vec::new(),
            generated: Vec::new(),
            text: String::new(),
            max_tokens,
            stop_sequences: params.stop_sequences.clone(),
            sampler,
            stop_reason: StopReason::MaxTokens,
            stop_sequence: None,
            kv: None,
        })
    }

    /// P4 ステッパ: セッションを live (エンジンの単一 KV キャッシュに乗った状態)
    /// にする。
    ///
    /// - 初回 (suspend 履歴なし): 旧 `chat()` と同一の prefix cache 経路選択
    ///   (live LCP vs マルチエントリストアの LCP、LCP が最大の候補を採用) で
    ///   KV を準備し、`sess.lcp` に再利用長を設定する
    /// - 再開 (suspend 済み): 退避しておいた KV スナップショットを bit 完全に復元する
    /// - いずれの場合も、その前に live に残っている完了済みリクエストの KV を
    ///   ストアへ退避する (旧 chat() が「次のリクエスト開始時」に退避するのと同じ)
    pub fn install_session(&mut self, sess: &mut ChatSession) {
        // live に残っている前リクエスト完了分の KV をストアへ退避。
        // (同一 prompt のエントリが既にあれば insert_snapshot が置き換え/保持する。
        //  KV 行はトークン接頭辞のみに依存するため、古いエントリでも
        //  共通接頭辞分の復元には bit 完全に使える → exactness に影響しない)
        if let Some((p, k, v, n)) = self.snapshot_live() {
            if !self.prefix_cache.contains_prompt(&p) {
                self.prefix_cache.insert_snapshot(p, k, v, n);
            }
        }

        if let Some(kv) = sess.kv.take() {
            // 再開: 退避済み KV (prompt + 生成済みトークン分) を復元
            self.cache.restore(&kv.k, &kv.v, kv.n);
            self.cached_prompt = sess.prompt_ids.clone();
            sess.lcp = sess.prompt_ids.len();
            sess.prefill_done = true;
            self.last_prefix_hit = PrefixHit::Live;
            return;
        }

        // --- 初回: 旧 chat() と同一の prefix cache 経路選択 ---
        let cap = sess.prompt_ids.len().saturating_sub(1);
        // logits は forward した最終位置からしか得られないので、
        // 少なくとも最後の 1 トークンは必ず forward する
        let live_lcp = sess
            .prompt_ids
            .iter()
            .zip(self.cached_prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
            .min(cap)
            .min(self.cache.len());
        let best_entry = self
            .prefix_cache
            .best_match(&sess.prompt_ids)
            .map(|(i, l)| (i, l.min(cap)));

        let lcp;
        let hit;
        if let Some((idx, entry_lcp)) = best_entry.filter(|&(_, l)| l > live_lcp) {
            // 別セッションのエントリへ切り替え (live の退避は冒頭で済み)
            let restored = self
                .prefix_cache
                .restore_into(idx, entry_lcp, &mut self.cache);
            if restored {
                lcp = entry_lcp;
                hit = PrefixHit::Entry;
            } else {
                // エントリが消えていたなど → フル prefill にフォールバック
                self.cache.reset();
                lcp = 0;
                hit = PrefixHit::None;
            }
        } else if live_lcp > 0 {
            // live 再利用 (truncate のみ、コピー無し)
            self.cache.truncate(live_lcp);
            lcp = live_lcp;
            hit = PrefixHit::Live;
        } else {
            // 共通接頭辞なし: フル prefill (live の退避は冒頭で済み)
            self.cache.reset();
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
    }

    /// P4 ステッパ: 実行中 (live) のセッションを KV キャッシュから退避する。
    /// 別のセッションを `install_session` で live にする前に呼ぶ。
    /// cache の行データは物理的には残るが `cached_prompt` を空にするため、
    /// 次の install では参照されない (上書き or reset)。
    pub fn suspend_session(&mut self, sess: &mut ChatSession) {
        debug_assert!(sess.kv.is_none(), "既に suspend 済みのセッション");
        debug_assert!(sess.prefill_done, "prefill 前のセッションは suspend できない");
        let kv_len = self.cache.len();
        let (k, v) = self.cache.snapshot(kv_len);
        sess.kv = Some(SuspendedKv { k, v, n: kv_len });
        self.cached_prompt.clear();
    }

    /// P4 ステッパ: 生成を 1 forward 進める (prefill 全体 or decode 1 トークン)。
    /// 完了 (EOS / stop sequence / max_tokens) で true を返す。
    /// 呼び出し前にセッションが live (`install_session` 済み) であること。
    pub fn step_chat(
        &mut self,
        sess: &mut ChatSession,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<bool> {
        if !sess.prefill_done {
            // prefill: 旧 chat() と同一形状 (残りプロンプトを 1 回の forward で処理)。
            // 分割すると GEMM の蓄積順が変わり exactness が壊れ得るため
            // 単一呼び出しの形状を維持する。
            let logits = self.model.forward(&sess.prompt_ids[sess.lcp..], &mut self.cache)?;
            self.cached_prompt = sess.prompt_ids.clone();
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
        let next = sess.sampler.sample(&sess.logits) as u32;
        sess.generated.push(next);

        if self.model.eos_ids().contains(&next) {
            sess.stop_reason = StopReason::EndTurn;
            return Ok(true);
        }

        // 差分だけストリーミング(部分的なUTF-8を避けるため全decode→diff)
        let full = self.tok.decode(&sess.generated)?;
        if full.len() > sess.text.len() && full.is_char_boundary(sess.text.len()) {
            let delta = &full[sess.text.len()..];
            on_delta(delta);
            sess.text.push_str(delta);
        }

        // stop_sequences チェック(生成テキスト中の最も早い出現で打ち切り)
        if let Some(seq) = sess
            .stop_sequences
            .iter()
            .find(|s| !s.is_empty() && sess.text.contains(s.as_str()))
        {
            let idx = sess
                .stop_sequences
                .iter()
                .filter_map(|s| sess.text.find(s.as_str()))
                .min()
                .unwrap_or(sess.text.len());
            sess.text.truncate(idx);
            sess.stop_reason = StopReason::StopSequence;
            sess.stop_sequence = Some(seq.clone());
            return Ok(true);
        }

        if sess.generated.len() < sess.max_tokens {
            let logits = self.model.forward(&[next], &mut self.cache)?;
            sess.logits = logits;
            return Ok(false);
        }
        sess.stop_reason = StopReason::MaxTokens;
        Ok(true)
    }

    /// チャット実行 (単発 API)。トークン生成ごとに on_delta(差分テキスト) が呼ばれる
    /// (ストリーミング用。非ストリーミングでは空クロージャでよい)。
    ///
    /// P4 で stepper (begin/install/step) ベースに書き換えたが、1 セッションを
    /// 通しで実行する経路は旧実装と同一の操作列 (同一形状の forward 呼び出し、
    /// 同一の prefix cache 経路選択) なので挙動は完全に変わらない。
    pub fn chat(
        &mut self,
        system: Option<&str>,
        messages: &[ChatMessage],
        params: &GenParams,
        mut on_delta: impl FnMut(&str),
    ) -> Result<ChatOutput> {
        let mut sess = self.begin_chat(system, messages, params)?;
        self.install_session(&mut sess);
        while !self.step_chat(&mut sess, &mut on_delta)? {}
        Ok(sess.into_output())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{dummy_engine, DUMMY_CONTEXT};

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
        let mut e = dummy_engine("m", None).unwrap();
        let out1 = e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();

        // cached_prompt にプロンプト全体が入っている
        let prompt = e
            .tok
            .apply_chat_template(None, &msgs("t0"))
            .unwrap();
        let prompt_ids = e.tok.encode(&prompt).unwrap();
        assert_eq!(e.cached_prompt, prompt_ids);

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
        // ただしセッションAはこの時点でストアに退避される)
        let out2 = e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Live);
        assert_eq!(e.prefix_cache_entries(), 1);
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
        let mut e = dummy_engine("m", None).unwrap();
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        e.chat(None, &msgs("t15"), &params(20), |_| {}).unwrap();
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Entry);
        // 同一セッション連続は live 経路 (単一エントリ時代と同じ、コピー無し)
        e.chat(None, &msgs("t0"), &params(20), |_| {}).unwrap();
        assert_eq!(e.last_prefix_hit(), PrefixHit::Live);
    }

    #[test]
    fn stop_sequence_truncates_at_first_occurrence() {
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
        let mut e = dummy_engine("m", None).unwrap();
        // コンテキスト (64) を大きく超える max_tokens はクランプされて続行する
        let out = e.chat(None, &msgs("t0"), &params(10_000), |_| {}).unwrap();
        assert!(out.input_tokens + out.output_tokens <= DUMMY_CONTEXT);
        assert_eq!(out.stop_reason, StopReason::EndTurn);
    }

    #[test]
    fn max_tokens_limit_stops_generation() {
        let mut e = dummy_engine("m", None).unwrap();
        // EOS に届く前に打ち切る
        let out = e.chat(None, &msgs("t0"), &params(2), |_| {}).unwrap();
        assert_eq!(out.stop_reason, StopReason::MaxTokens);
        assert_eq!(out.output_tokens, 2);
        assert_eq!(out.text, "t5 t6");
    }

    #[test]
    fn on_delta_receives_the_same_text_as_result() {
        let mut e = dummy_engine("m", None).unwrap();
        let mut streamed = String::new();
        let out = e
            .chat(None, &msgs("t0"), &params(20), |delta| streamed.push_str(delta))
            .unwrap();
        assert_eq!(streamed, out.text);
    }

    #[test]
    fn empty_messages_is_an_error() {
        let mut e = dummy_engine("m", None).unwrap();
        assert!(e.chat(None, &[], &params(4), |_| {}).is_err());
    }

    #[test]
    fn system_prompt_is_included_in_prompt_tokens() {
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

    /// インターリーブ下 (A prefill → A suspend → B 完了まで → A 再開) でも
    /// 各リクエストの出力は単独実行と完全一致する (P4 最重要要件)。
    /// DummyModel は KV を参照しないため、KV 依存の KvSensitiveModel を使う。
    #[test]
    fn interleaved_sessions_match_standalone_outputs() {
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

        // A: prefill まで進める
        e.install_session(&mut sa);
        assert!(!e.step_chat(&mut sa, &mut |_| {}).unwrap());

        // A を suspend して B を割り込ませ、完了まで進める
        e.suspend_session(&mut sa);
        e.install_session(&mut sb);
        for _ in 0..100 {
            if e.step_chat(&mut sb, &mut |_| {}).unwrap() {
                break;
            }
        }
        let out_b = sb.into_output();

        // A を再開 (KV 復元) して完了まで進める
        e.install_session(&mut sa);
        for _ in 0..100 {
            if e.step_chat(&mut sa, &mut |_| {}).unwrap() {
                break;
            }
        }
        let out_a = sa.into_output();

        assert_eq!(out_a, ref_a, "インターリーブ後の A が単独実行と不一致");
        assert_eq!(out_b, ref_b, "割り込んだ B が単独実行と不一致");
    }

    /// 4 セッション (同一プロンプト含む) を 1 ステップずつ round-robin で進めても
    /// 全て単独実行と完全一致。
    #[test]
    fn round_robin_stepping_matches_standalone_outputs() {
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

        // 全員の prefill を順に済ませる (install → prefill step → suspend)
        for s in sessions.iter_mut() {
            e.install_session(s);
            assert!(!e.step_chat(s, &mut |_| {}).unwrap());
            e.suspend_session(s);
        }

        // round-robin で decode (完了したセッションは suspend せず放置)
        let mut done = vec![false; sessions.len()];
        for _ in 0..200 {
            let active: Vec<usize> = (0..sessions.len()).filter(|&i| !done[i]).collect();
            if active.is_empty() {
                break;
            }
            for &i in &active {
                e.install_session(&mut sessions[i]);
                done[i] = e.step_chat(&mut sessions[i], &mut |_| {}).unwrap();
                if !done[i] {
                    e.suspend_session(&mut sessions[i]);
                }
            }
        }
        assert!(done.iter().all(|&d| d), "全セッションが完了していない");

        for (i, s) in sessions.into_iter().enumerate() {
            assert_eq!(s.into_output(), refs[i], "session {i} が単独実行と不一致");
        }
    }
}
