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

/// int8 量子化パスの生成決定論テスト用ダミーモデル (P3)。
///
/// logits = W @ a を `quant::Linear` 経由で計算する (a は直前トークンIDから
/// 決定論的に作る疎ベクトル)。重みは `QuantMode` に従い fp32 または
/// per-channel int8 に量子化される。丸め誤差で fp32 とは logits が僅かに
/// 異なる (= 量子化パスは fp32 と同一出力ではない仕様の確認用) が、
/// int8 パス自体は決定論的 (同一 seed → 同一出力)。
pub struct QuantMatModel {
    w: crate::quant::Linear, // [DUMMY_VOCAB, DUMMY_QDIM]
    /// 最後に forward が返した logits (テストで比較する用)
    pub last_logits: std::cell::RefCell<Vec<f32>>,
}

/// QuantMatModel の入力次元
pub const DUMMY_QDIM: usize = 8;

impl QuantMatModel {
    /// 決定論的な重みで構築。mode = Int8 ならロード時に量子化する。
    pub fn new(mode: crate::quant::QuantMode) -> Self {
        let n = DUMMY_VOCAB;
        let k = DUMMY_QDIM;
        let mut w = Vec::with_capacity(n * k);
        // LCG (tensor.rs/quant.rs の pseudo_random と同じ系列) で値を作る。
        // わざと 2 の冪できれいに割れない値にし、丸め誤差が必ず発生するようにする。
        let mut x = 1234u32;
        for _ in 0..n * k {
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            w.push(((x >> 16) % 2000) as f32 / 977.0 - 1.0); // [-1, 1) 非整数刻み
        }
        Self {
            w: crate::quant::Linear::new(w, k, n, mode),
            last_logits: std::cell::RefCell::new(Vec::new()),
        }
    }
}

impl Model for QuantMatModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0) as usize;
        let mut a = vec![0f32; DUMMY_QDIM];
        a[last % DUMMY_QDIM] = 1.0;
        a[(last + 3) % DUMMY_QDIM] = 0.5;
        let logits = self.w.apply_bt(&a, 1);
        *self.last_logits.borrow_mut() = logits.clone();
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

/// QuantMatModel (量子化パス用ダミー) で Engine を構築する
pub fn quant_dummy_engine(
    name: &str,
    seed: Option<u64>,
    mode: crate::quant::QuantMode,
) -> Result<Engine> {
    let tok = dummy_tokenizer()?;
    Ok(Engine::from_parts(Box::new(QuantMatModel::new(mode)), tok, name, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ChatMessage, GenParams};
    use crate::quant::QuantMode;

    fn greedy_params(max_tokens: usize) -> GenParams {
        GenParams {
            max_tokens,
            temperature: Some(0.0),
            ..Default::default()
        }
    }

    fn msgs(text: &str) -> Vec<ChatMessage> {
        vec![ChatMessage { role: "user".into(), content: text.into() }]
    }

    /// P3: 量子化パスでの生成決定論 (ダミーモデル)。
    /// 同一 seed + 温度 0 なら、エンジンをまたいでも・同一エンジンの再実行
    /// (prefix cache ヒット経路含む) でも完全に同一の出力になる。
    #[test]
    fn quant_dummy_generation_is_deterministic_in_int8_mode() {
        let mut a = quant_dummy_engine("a", Some(42), QuantMode::Int8).unwrap();
        let mut b = quant_dummy_engine("b", Some(42), QuantMode::Int8).unwrap();

        let out_a1 = a.chat(None, &msgs("t0"), &greedy_params(20), |_| {}).unwrap();
        let out_a2 = a.chat(None, &msgs("t0"), &greedy_params(20), |_| {}).unwrap();
        let out_b = b.chat(None, &msgs("t0"), &greedy_params(20), |_| {}).unwrap();

        assert_eq!(out_a1.text, out_a2.text, "同一エンジン再実行 (prefix cache 経路) で不一致");
        assert_eq!(out_a1.text, out_b.text, "エンジンをまたいで不一致");
        assert_eq!(out_a1.output_tokens, out_b.output_tokens);
        assert_eq!(out_a1.stop_reason, out_b.stop_reason);

        // seed 付きサンプリング (temperature > 0) でも決定論
        let mut p = greedy_params(20);
        p.temperature = Some(1.0);
        let s1 = a.chat(None, &msgs("t5"), &p, |_| {}).unwrap();
        let s2 = b.chat(None, &msgs("t5"), &p, |_| {}).unwrap();
        assert_eq!(s1.text, s2.text);
    }

    /// P3 仕様の確認: int8 モデルの logits は丸め誤差により fp32 と同一ではない。
    /// ただし誤差は小さく、greedy の選択 (argmax) はこのダミーでは変わらない。
    #[test]
    fn quant_dummy_int8_logits_differ_from_fp32_but_stay_close() {
        let mut f = QuantMatModel::new(QuantMode::Fp32);
        let mut q = QuantMatModel::new(QuantMode::Int8);
        let mut cf = f.new_cache();
        let mut cq = q.new_cache();
        let lf = f.forward(&[3], &mut cf).unwrap();
        let lq = q.forward(&[3], &mut cq).unwrap();

        assert_ne!(lf, lq, "int8 logits が fp32 と完全一致 (丸め誤差が無い = 仕様違反)");

        let mut max_abs = 0f32;
        for (g, w) in lq.iter().zip(&lf) {
            max_abs = max_abs.max((g - w).abs());
        }
        // per-channel int8 (scale = max|w|/127) の誤差オーダー: 入力 a のノルム ~1.1 なので
        // |Δlogits| ≲ Σ|a_d| * scale_max ≈ 1.1 * (1/127 * ~1) ≈ 0.01 程度に収まるはず
        assert!(max_abs < 0.05, "丸め誤差が大きすぎる: {max_abs}");

        let argmax_f = lf.iter().enumerate().max_by(|(_, x), (_, y)| x.total_cmp(y)).unwrap().0;
        let argmax_q = lq.iter().enumerate().max_by(|(_, x), (_, y)| x.total_cmp(y)).unwrap().0;
        assert_eq!(argmax_f, argmax_q, "ダミーモデルで argmax が変わった");
    }
}
