//! テスト用ユーティリティ (feature = "test-utils" または cargo test 時のみビルド)。
//!
//! 小さなダミーモデル + ダミートークナイザで Engine を構築できるようにし、
//! denno-server のワイヤレベルテスト (SSE の event: 行検証など) を
//! モデルDLなしで・数秒で実行できるようにするのが目的。
//! 実モデルでの出力検証は tests/exactness.rs (#[ignore] 付き) を使う。

use crate::engine::Engine;
use crate::kv_cache::KvCache;
use crate::model::Model;
use crate::tokenizer::DennoTokenizer;
use anyhow::Result;

/// ダミーモデルの語彙数 (logits の長さ)
pub const DUMMY_VOCAB: usize = 16;
/// ダミーモデルの EOS トークンID
pub const DUMMY_EOS: u32 = 12;
/// ダミーモデルのコンテキスト長
pub const DUMMY_CONTEXT: usize = 64;

/// 最小構成の WordLevel トークナイザ。
/// "t0".."t11" が ID 0..11、"eos" が 12、未知語は "[UNK]" (ID 20)。
pub const TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "t0": 0, "t1": 1, "t2": 2, "t3": 3, "t4": 4, "t5": 5, "t6": 6,
      "t7": 7, "t8": 8, "t9": 9, "t10": 10, "t11": 11, "eos": 12,
      "t13": 13, "t14": 14, "t15": 15, "[UNK]": 20
    },
    "unk_token": "[UNK]"
  }
}"#;

/// 決定論的なダミーモデル。
///
/// logits は「直前トークン + 1」をピーク (5.0)、「+2」を準ピーク (1.0) にする。
/// greedy (temperature 0) なら t_n の次は常に t_{n+1}、t11 の次は EOS(12) で停止する。
/// KVキャッシュの中身は参照せず advance だけする (engine.rs の LCP/truncate
/// ロジックの動作検証には十分)。
pub struct DummyModel;

impl Model for DummyModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        logits[(last + 1) % DUMMY_VOCAB] = 5.0;
        logits[(last + 2) % DUMMY_VOCAB] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

/// ダミートークナイザ (chat template 無し → fallback テンプレートが使われる)
pub fn dummy_tokenizer() -> Result<DennoTokenizer> {
    DennoTokenizer::from_json_str(TOKENIZER_JSON, None)
}

/// ダミーモデル + ダミートークナイザで Engine を構築する
pub fn dummy_engine(name: &str, seed: Option<u64>) -> Result<Engine> {
    let tok = dummy_tokenizer()?;
    Ok(Engine::from_parts(Box::new(DummyModel), tok, name, seed))
}
