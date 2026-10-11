//! エンジン本体: モデル+トークナイザ+KVキャッシュ+サンプラを束ね、
//! prefill → decode ループ(stop_sequences / EOS / max_tokens 処理)を実行する。

use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::kv_cache::KvCache;
use crate::model::{load_model, Model};
use crate::sampler::Sampler;
use crate::tokenizer::DennoTokenizer;

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

#[derive(Debug, Clone)]
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
}

impl Engine {
    /// モデルディレクトリ(HF形式: config.json + model.safetensors + tokenizer.json
    /// + tokenizer_config.json)からロードする。
    pub fn load(dir: &Path, name: impl Into<String>, default_seed: Option<u64>) -> Result<Self> {
        let model = load_model(dir).context("モデルのロードに失敗")?;
        let tok = DennoTokenizer::load(dir).context("トークナイザのロードに失敗")?;
        let cache = model.new_cache();
        Ok(Self {
            model,
            tok,
            cache,
            name: name.into(),
            default_seed,
        })
    }

    pub fn context(&self) -> usize {
        self.model.context()
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
        if prompt_ids.len() + max_tokens > self.model.context() {
            bail!(
                "invalid_request_error: prompt ({} tokens) + max_tokens ({}) がモデルのコンテキスト長 {} を超えます",
                prompt_ids.len(),
                max_tokens,
                self.model.context()
            );
        }

        // prefill
        self.cache.reset();
        let mut logits = self.model.forward(&prompt_ids, &mut self.cache)?;

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
