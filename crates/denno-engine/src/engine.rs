//! エンジン本体: モデル+トークナイザ+KVキャッシュ+サンプラを束ね、
//! prefill → decode ループ(stop_sequences / EOS / max_tokens 処理)を実行する。

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::model::{load_model, Model};
use crate::prefix_cache::PrefixCache;
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

impl Engine {
    /// モデルディレクトリ(HF形式: config.json + model.safetensors + tokenizer.json
    /// + tokenizer_config.json)からロードする。
    pub fn load(dir: &Path, name: impl Into<String>, default_seed: Option<u64>) -> Result<Self> {
        let model = load_model(dir).context("モデルのロードに失敗")?;
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

    /// チャット実行。トークン生成ごとに on_delta(差分テキスト) が呼ばれる
    /// (ストリーミング用。非ストリーミングでは空クロージャでよい)。
    pub fn chat(
        &mut self,
        system: Option<&str>,
        messages: &[ChatMessage],
        params: &GenParams,
        mut on_delta: impl FnMut(&str),
    ) -> Result<ChatOutput> {
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

        // --- prefill (マルチエントリ LRU prefix cache 付き) ---
        // 候補は2つ:
        //   live: 直前リクエストの KV (まだ KvCache に入っている、truncate だけで再利用)
        //   entry: 過去会話のスナップショット (KvCache へ復元して再利用)
        // LCP が最大の候補を採用する。Hermes 等は毎回同じ system prompt を送ってくる
        // ので、同一セッションの継続 (live) でも別セッションの復帰 (entry) でも
        // prefill の大半を省略できる。
        let cap = prompt_ids.len().saturating_sub(1);
        // logits は forward した最終位置からしか得られないので、
        // 少なくとも最後の 1 トークンは必ず forward する
        let live_lcp = prompt_ids
            .iter()
            .zip(self.cached_prompt.iter())
            .take_while(|(a, b)| a == b)
            .count()
            .min(cap)
            .min(self.cache.len());
        let best_entry = self
            .prefix_cache
            .best_match(&prompt_ids)
            .map(|(i, l)| (i, l.min(cap)));

        let lcp;
        let hit;
        if let Some((idx, entry_lcp)) = best_entry.filter(|&(_, l)| l > live_lcp) {
            // 別セッションのエントリへ切り替え: live の KV をスナップショットしてから
            // (復元が live の行を上書きするため先に退避)、エントリの先頭 entry_lcp 行を復元。
            let snap = self.snapshot_live();
            let restored = self
                .prefix_cache
                .restore_into(idx, entry_lcp, &mut self.cache);
            if let Some((p, k, v, n)) = snap {
                self.prefix_cache.insert_snapshot(p, k, v, n);
            }
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
            // live 再利用 (単一エントリ時代と同じ経路)。cache.len() >= live_lcp が
            // 常に成り立つ (生成トークン分だけ長いことがある) ので truncate で戻せる。
            // ただし live の会話がまだストアに無い場合は退避しておく:
            // live との共通接頭辞 (system prompt 等) だけでは別セッション固有の
            // suffix は再計算になってしまうため、ストアに無いと交互セッションで
            // エントリが永遠に増えない。
            if !self.prefix_cache.contains_prompt(&self.cached_prompt) {
                let snap = self.snapshot_live();
                if let Some((p, k, v, n)) = snap {
                    self.prefix_cache.insert_snapshot(p, k, v, n);
                }
            }
            self.cache.truncate(live_lcp);
            lcp = live_lcp;
            hit = PrefixHit::Live;
        } else {
            // 共通接頭辞なし: live をストアに退避してフル prefill
            let snap = self.snapshot_live();
            if let Some((p, k, v, n)) = snap {
                self.prefix_cache.insert_snapshot(p, k, v, n);
            }
            self.cache.reset();
            lcp = 0;
            hit = PrefixHit::None;
        }
        if lcp > 0 {
            tracing::info!(
                "prefix cache hit: reuse {} / {} tokens (recompute {})",
                lcp,
                prompt_ids.len(),
                prompt_ids.len() - lcp
            );
        }
        // truncate/restore された stale な行は forward 中に上書きされるか、
        // そもそも参照されない。
        let mut logits = self.model.forward(&prompt_ids[lcp..], &mut self.cache)?;
        self.cached_prompt = prompt_ids.clone();
        self.last_prefix_hit = hit;

        let seed = params.seed.or(self.default_seed);
        let mut sampler = Sampler::new(params.temperature, params.top_k, params.top_p, seed);

        let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
        let mut text = String::new();
        let mut stop_reason = StopReason::MaxTokens;
        let mut stop_sequence: Option<String> = None;

        for step in 0..max_tokens {
            let next = sampler.sample(&logits) as u32;
            generated.push(next);

            if self.model.eos_ids().contains(&next) {
                stop_reason = StopReason::EndTurn;
                break;
            }

            // 差分だけストリーミング(部分的なUTF-8を避けるため全decode→diff)
            let full = self.tok.decode(&generated)?;
            if full.len() > text.len() && full.is_char_boundary(text.len()) {
                let delta = &full[text.len()..];
                on_delta(delta);
                text.push_str(delta);
            }

            // stop_sequences チェック(生成テキスト中の最も早い出現で打ち切り)
            if let Some(seq) = params
                .stop_sequences
                .iter()
                .find(|s| !s.is_empty() && text.contains(s.as_str()))
            {
                let idx = params
                    .stop_sequences
                    .iter()
                    .filter_map(|s| text.find(s.as_str()))
                    .min()
                    .unwrap_or(text.len());
                text.truncate(idx);
                stop_reason = StopReason::StopSequence;
                stop_sequence = Some(seq.clone());
                break;
            }

            if step + 1 < max_tokens {
                logits = self.model.forward(&[next], &mut self.cache)?;
            }
        }

        Ok(ChatOutput {
            text,
            stop_reason,
            stop_sequence,
            input_tokens: prompt_ids.len(),
            output_tokens: generated.len(),
        })
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
}
