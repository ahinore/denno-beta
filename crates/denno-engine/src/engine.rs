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
    pub name: String,
    pub default_seed: Option<u64>,
    /// 過去会話の KV スナップショット群 (マルチエントリ LRU prefix cache)
    prefix_cache: PrefixCache,
    /// 直前のリクエストでの prefix cache ヒット状況
    last_prefix_hit: PrefixHit,
    /// 重みの量子化モード (テレメトリの ServerInfo 用に保持。数値経路には不変)
    pub quant: QuantMode,
}

/// KV キャッシュスロット数の既定値 (環境変数 `DENNO_BATCH_SLOTS` で変更可)。
pub const DEFAULT_BATCH_SLOTS: usize = 4;
pub const BATCH_SLOTS_ENV: &str = "DENNO_BATCH_SLOTS";

fn batch_slots() -> usize {
    std::env::var(BATCH_SLOTS_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(DEFAULT_BATCH_SLOTS)
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
    max_tokens: usize,
    stop_sequences: Vec<String>,
    sampler: Sampler,
    stop_reason: StopReason,
    stop_sequence: Option<String>,
    /// 占有中の KV キャッシュスロット (None = 未 install / release 済み)
    slot: Option<usize>,
    input_tokens: usize,
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
        // マルチエントリ prefix cache のトークン上限。
        // デフォルトは KvCache の max_len (コンテキスト長) と同じ (Qwen2.5-0.5B なら 32k)。
        let max_tokens = std::env::var(PREFIX_CACHE_ENV)
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or_else(|| slot0.max_len());
        let slots = batch_slots();
        tracing::info!(
            "prefix cache: multi-entry LRU, budget = {} tokens (~{:.0} MB KV)",
            max_tokens,
            max_tokens as f64 * slot0.bytes_per_token() as f64 / 1e6
        );
        tracing::info!(
            "batch slots: {} (KV ~{:.0} MB/slot, env {BATCH_SLOTS_ENV})",
            slots,
            slot0.max_len() as f64 * slot0.bytes_per_token() as f64 / 1e6
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
        Ok(Self {
            model,
            tok,
            caches,
            cached_prompts,
            slot_busy,
            name: name.into(),
            default_seed,
            prefix_cache: PrefixCache::new(max_tokens),
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
        let slots = batch_slots();
        let caches: Vec<KvCache> = (0..slots).map(|_| model.new_cache()).collect();
        let cached_prompts = vec![Vec::new(); slots];
        let slot_busy = vec![false; slots];
        Self {
            model,
            tok,
            caches,
            cached_prompts,
            slot_busy,
            name: name.into(),
            default_seed,
            prefix_cache: PrefixCache::new(max_tokens),
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
    pub fn install_session(&mut self, sess: &mut ChatSession) -> Result<()> {
        if sess.slot.is_some() {
            return Ok(()); // 既に常駐済み
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
        let best_entry = self
            .prefix_cache
            .best_match(&sess.prompt_ids)
            .map(|(i, l)| (i, l.min(cap)));
        if let Some((idx, entry_lcp)) = best_entry.filter(|&(_, l)| l > best_live_lcp) {
            // 別セッションのエントリから KV を復元
            let restored = self
                .prefix_cache
                .restore_into(idx, entry_lcp, &mut self.caches[slot]);
            if restored {
                lcp = entry_lcp;
                hit = PrefixHit::Entry;
            } else {
                // エントリが消えていたなど → フル prefill にフォールバック
                self.caches[slot].reset();
                self.cached_prompts[slot].clear();
                lcp = 0;
                hit = PrefixHit::None;
            }
        } else if best_live_lcp > 0 {
            // スロット残留 KV の live 再利用 (truncate のみ、コピー無し)
            self.caches[slot].truncate(best_live_lcp);
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
    /// KV スナップショットをマルチエントリ prefix store へ退避してから
    /// スロットを空きに戻す (旧 install 冒頭の退避と同一の操作)。
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
        if !self.cached_prompts[slot].is_empty() && !self.caches[slot].is_empty() {
            let p = self.cached_prompts[slot].clone();
            if !self.prefix_cache.contains_prompt(&p) {
                let kv_len = self.caches[slot].len();
                let (k, v) = self.caches[slot].snapshot(kv_len);
                self.prefix_cache.insert_snapshot(p, k, v, kv_len);
            }
        }
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
            slot: None,
            id: telem::next_session_id(),
        })
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
        if !sess.prefill_done {
            // prefill: 旧 chat() と同一形状 (残りプロンプトを 1 回の forward で処理)。
            // 分割すると GEMM の蓄積順が変わり exactness が壊れ得るため
            // 単一呼び出しの形状を維持する。
            let slot = sess.slot.expect("install_session 済みのセッション");
            let t0 = telem::enabled().then(std::time::Instant::now);
            let logits =
                self.model
                    .forward(&sess.prompt_ids[sess.lcp..], &mut self.caches[slot])?;
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
        let next =
            crate::prof::time(crate::prof::SAMPLER, || sess.sampler.sample(&sess.logits)) as u32;
        sess.generated.push(next);
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

    /// 生成終了時のフラッシュ: emit で保留中のトークン (末尾がマルチバイト
    /// 文字の途中で終わるバイト列) を最終 decode で確定させて流す。
    /// EOS / max_tokens で打ち切られたステップでは通常の emit が行われない
    /// (または保留が起きる) ため、完了時に必ず呼ぶ。
    /// `exclude_last` は decode から除外する末尾トークン数 (EOS 分 = 1)。
    fn flush_stream(
        &self,
        sess: &mut ChatSession,
        exclude_last: usize,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<()> {
        let n = sess.generated.len().saturating_sub(exclude_last);
        let full = self.tok.decode(&sess.generated[..n])?; // lossy (不正バイトは U+FFFD)
        if full.len() > sess.text.len() && full.is_char_boundary(sess.text.len()) {
            let delta = &full[sess.text.len()..];
            on_delta(delta);
            sess.text.push_str(delta);
        }
        Ok(())
    }

    /// decode 1 ステップの共通後処理: 差分ストリーミング + stop_sequences チェック。
    /// stop_sequence で打ち切るときは sess の stop_reason/stop_sequence を設定し
    /// true を返す。step_chat / step_chat_batch の両方から使う。
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
        let full = crate::prof::time(crate::prof::DETOK, || -> Result<String> {
            let mut n = sess.generated.len();
            let mut full = self.tok.decode(&sess.generated[..n])?;
            while n > 0 && full.ends_with('\u{FFFD}') {
                n -= 1;
                full = self.tok.decode(&sess.generated[..n])?;
            }
            Ok(full)
        })?;
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
        on_deltas: &mut [Box<dyn FnMut(&str)>],
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
            let next = crate::prof::time(crate::prof::SAMPLER, || {
                sess.sampler.sample(&sess.logits)
            }) as u32;
            sess.generated.push(next);
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

            if self.model.eos_ids().contains(&next) {
                sess.stop_reason = StopReason::EndTurn;
                self.flush_stream(sess, 1, on_deltas[i].as_mut())?;
                done[i] = true;
                continue;
            }
            if self.emit_delta_and_check_stop(sess, on_deltas[i].as_mut())? {
                done[i] = true;
                continue;
            }
            if sess.generated.len() >= sess.max_tokens {
                sess.stop_reason = StopReason::MaxTokens;
                self.flush_stream(sess, 0, on_deltas[i].as_mut())?;
                done[i] = true;
                continue;
            }
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
            let outs = model.forward_batch(&sorted_toks, &mut cache_refs)?;
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

        // スロット 0 の cached_prompt にプロンプト全体が入っている
        let prompt = e
            .tok
            .apply_chat_template(None, &msgs("t0"))
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

    /// 交互実行下 (A 常駐中に B を割り込ませ、B 完了後に A を続行) でも
    /// 各リクエストの出力は単独実行と完全一致する (P4 最重要要件)。
    /// P6 ではセッションは各々のスロットに常駐し続けるため、suspend せずに
    /// 別セッションを install して進める。
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

    /// 4 セッション (同一プロンプト含む) を 1 ステップずつ round-robin で進めても
    /// 全て単独実行と完全一致。P6 では全員が同時にスロットに常駐する。
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
}
