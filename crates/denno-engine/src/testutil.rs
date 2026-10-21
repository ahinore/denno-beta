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
/// P4-4: 長文脈回帰テスト用のコンテキスト長 (本番の 3,020 token prefill を
/// 覆う長さ。既定ゲートの他テストは DUMMY_CONTEXT のまま)。
pub const LONG_CONTEXT: usize = 4096;

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
        // P17e: 需要ベース確保。実モデルは KV 書き込み経路で成長するため、
        // ダミーも同じ契約 (書き込み前に容量確保) に合わせる
        let _ = cache.kv_bufs(0, tokens.len());
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

// ---------- 第21: tool use (streaming tool_use / 拘束デコード) の E2E テスト用 ----------

/// tool use E2E テスト用語彙 (**連番の密な id**)。
/// ByteLevel decoder と組み合わせるとデコード空間が byte-level BPE と同一になる
/// ("Ċ" → "\n")。tool_call スキャフォールド + 2 ツール名 + マルチバイト名を含む。
/// id は logits 長 48 の ScriptModel で使う。
pub const TOOL_VOCAB: &[(&str, u32)] = &[
    ("t0", 0),
    ("t1", 1),
    ("t2", 2),
    ("t3", 3),
    ("t4", 4),
    ("t5", 5),
    ("t6", 6),
    ("t7", 7),
    ("t8", 8),
    ("t9", 9),
    ("t10", 10),
    ("t11", 11),
    ("eos", 12), // DUMMY_EOS と一致させる
    ("[UNK]", 13),
    ("<", 14),
    ("`", 15),
    ("``", 16),
    ("Ċ", 17), // byte-level の "\n"
    ("<function=", 18),
    (">", 19),
    ("<parameter=", 20),
    ("</parameter>", 21),
    ("<tool_call>", 22),
    ("</tool_call>", 23),
    ("get_weather", 24),
    ("run_tool", 25),
    ("city", 26),
    ("Tokyo", 27),
    ("Checking.", 28),
    ("天", 29),
    ("気", 30),
    ("天気", 31),
    ("</function>", 32), // 第27: 完全な tool_call block (XML 閉じタグ) 用
];

/// tool use E2E テスト用トークナイザ (ByteLevel decoder + TOOL_VOCAB)。
pub fn tool_tokenizer() -> Result<DennoTokenizer> {
    tool_tokenizer_with(&[])
}

/// s53: TOOL_VOCAB に追加語彙を足したトークナイザ (ByteLevel decoder)。
///
/// WordLevel + 空白 pre-tokenizer のため、**1 語として現れる**追加 piece だけが
/// 追加語彙に一致する。id は 40 以降を重複順に割り当てる (決定的)。
pub fn tool_tokenizer_with(extra: &[&str]) -> Result<DennoTokenizer> {
    let mut vocab: Vec<(&str, u32)> = TOOL_VOCAB.to_vec();
    let mut next = 40u32;
    for &e in extra {
        if vocab.iter().any(|(t, _)| *t == e) {
            continue;
        }
        vocab.push((e, next));
        next += 1;
    }
    DennoTokenizer::testutil_vocab_tokenizer(
        &vocab,
        Some(serde_json::json!({
            "type": "ByteLevel", "add_prefix_space": false,
            "trim_offsets": true, "use_regex": true
        })),
    )
}

/// tool use E2E 用: 固定トークン列 (スクリプト) を one-hot logits で出す決定論モデル。
///
/// forward 1 呼び出し = スクリプト 1 位置前進 (prefill も 1 call として数える。
/// 少なくとも最終トークンは必ず forward されるため call 0 は常に prefill)。
/// **単一セッション専用** (call 数 = スクリプト位置のため複数セッション同時実行では
/// 位置が混在する。tool use E2E テストは 1 リクエストずつ実行する)。
/// `fallback` は one-hot がマスクで禁止されたときに選ばれる準ピーク (1.0) —
/// 拘束デコードが「指定外ツール名を禁止して別トークンに落ちる」ことを実証する。
pub struct ScriptModel {
    pub script: Vec<u32>,
    pub fallback: u32,
    /// logits 長 (スクリプトの最大 id より大きいこと)
    pub vocab: usize,
    calls: usize,
}

impl ScriptModel {
    pub fn new(script: Vec<u32>, fallback: u32) -> Self {
        Self { script, fallback, vocab: 48, calls: 0 }
    }
}

impl Model for ScriptModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let idx = self.calls.min(self.script.len().saturating_sub(1));
        self.calls += 1;
        let mut logits = vec![0.0f32; self.vocab];
        let peak = self.script.get(idx).copied();
        if let Some(t) = peak {
            logits[t as usize] = 5.0;
        }
        if (self.fallback as usize) < self.vocab && Some(self.fallback) != peak {
            logits[self.fallback as usize] = 1.0;
        }
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

/// ScriptModel + tool_tokenizer で Engine を構築する
pub fn tool_script_engine(
    name: &str,
    seed: Option<u64>,
    script: Vec<u32>,
    fallback: u32,
) -> Result<Engine> {
    let tok = tool_tokenizer()?;
    Ok(Engine::from_parts(Box::new(ScriptModel::new(script, fallback)), tok, name, seed))
}

/// ダミーモデル + ダミートークナイザで Engine を構築する
pub fn dummy_engine(name: &str, seed: Option<u64>) -> Result<Engine> {
    let tok = dummy_tokenizer()?;
    Ok(Engine::from_parts(Box::new(DummyModel), tok, name, seed))
}

/// KV キャッシュの内容に依存する決定論的ダミーモデル (P4 インターリーブ exactness 用)。
///
/// DummyModel は KV の中身を参照しないため、suspend/restore の切替えが壊れても
/// logits が変わらず検証にならない。このモデルは forward のたびに K/V 行
/// (トークン+位置に依存する決定論的値) を書き込み、logits の argmax を
/// 「キャッシュ内 K 全行の加重和」から決める。KV の退避/復元が壊れる
/// (行が欠落・入れ替わる) と logits が変わり生成トークン列が一致しなくなるため、
/// 単一 KV キャッシュの時分割切替えの exactness 検証に使える。
pub struct KvSensitiveModel {
    /// forward ごとの sleep (スケジューラのインターリーブを確実に起こす用)
    pub delay: std::time::Duration,
}

impl KvSensitiveModel {
    /// 遅延無しで構築
    pub fn new() -> Self {
        Self { delay: std::time::Duration::ZERO }
    }

    /// P4-4: 長文脈テストが「モデルが書いた行」を式から再計算して照合できるよう
    /// crate 内に公開する (挙動の変更ではない)。
    pub(crate) fn kv_k(t: u32, pos: usize, d: usize, kv_dim: usize) -> f32 {
        (((t as usize + 2 * d + pos) % 11) as f32) / 3.0 + ((t as usize * (d + 1)) % 7) as f32
    }

    pub(crate) fn kv_v(t: u32, pos: usize, d: usize, kv_dim: usize) -> f32 {
        (((t as usize + 3 * d + pos) % 13) as f32) / 5.0
    }
}

impl Default for KvSensitiveModel {
    fn default() -> Self {
        Self::new()
    }
}

impl Model for KvSensitiveModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        if !self.delay.is_zero() {
            std::thread::sleep(self.delay);
        }
        let pos = cache.len();
        let kv_dim = cache.kv_dim();
        for (i, &t) in tokens.iter().enumerate() {
            let p = pos + i;
            let (k, v) = cache.kv_bufs(0, tokens.len());
            for d in 0..kv_dim {
                k[p * kv_dim + d] = Self::kv_k(t, p, d, kv_dim);
                v[p * kv_dim + d] = Self::kv_v(t, p, d, kv_dim);
            }
        }
        cache.advance(tokens.len());
        // logits の argmax は K 全行の加重和に依存する (V は参照のみで exactness に
        // 影響しないが、壊れた restore を K 側で検出できれば十分)
        let len = cache.len();
        let acc;
        let acc2;
        {
            let (k, _) = cache.kv_bufs_ref(0);
            let mut a1 = 0.0f32;
            let mut a2 = 0.0f32;
            for p in 0..len {
                for d in 0..kv_dim {
                    let x = k[p * kv_dim + d];
                    a1 += x * ((p % 4 + 1) as f32);
                    a2 += x * ((p % 5 + 2) as f32);
                }
            }
            acc = a1;
            acc2 = a2;
        }
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = ((acc * 7.0).round() as i64).rem_euclid(DUMMY_VOCAB as i64) as usize;
        let second = ((acc2 * 3.0).round() as i64).rem_euclid(DUMMY_VOCAB as i64) as usize;
        logits[peak] = 5.0;
        logits[second] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

/// KvSensitiveModel (KV 依存ダミー) で Engine を構築する
pub fn kv_sensitive_engine(name: &str, seed: Option<u64>) -> Result<Engine> {
    let tok = dummy_tokenizer()?;
    Ok(Engine::from_parts(Box::new(KvSensitiveModel::new()), tok, name, seed))
}

/// P4-4: 長文脈回帰テスト用の KV 依存モデル。
///
/// 数値は `KvSensitiveModel` と**同一** (forward を委譲するので、KV 行の式も
/// logits の組み立ても 1 行も変わらない) で、コンテキスト長だけを
/// `LONG_CONTEXT` (4096) に広げる。既存ヘルパの挙動を変えないため、
/// `KvSensitiveModel::new()` は DUMMY_CONTEXT のまま据え置き、
/// 長文脈はこの型でだけ作る (既定ゲートの他テストは 64 token のまま)。
///
/// 用途: kv_cache の容量境界・需要ベース成長・長い prompt での prefix cache
/// 予算/退避を、本番と同じ「コンテキスト長 >> テスト長」の条件で回帰させる。
pub struct LongKvSensitiveModel {
    inner: KvSensitiveModel,
    context: usize,
}

impl LongKvSensitiveModel {
    /// コンテキスト長を指定して構築する (kv_dim は 4 のまま = テストを軽く保つ)。
    pub fn with_context(context: usize) -> Self {
        Self { inner: KvSensitiveModel::new(), context }
    }
}

impl Model for LongKvSensitiveModel {
    fn context(&self) -> usize {
        self.context
    }

    fn eos_ids(&self) -> &[u32] {
        self.inner.eos_ids()
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        // KvSensitiveModel::forward は cache.kv_dim() を使うため、長文脈でも
        // 行の式・logits は 64 token 版と完全に同じ経路を通る。
        self.inner.forward(tokens, cache)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, self.context)
    }
}

/// P4-4: 長文脈 (LONG_CONTEXT) 用エンジン。kv_dim 4 / 1 層なので
/// bytes_per_token = 32 B で、4096 token でも KV は 128 KB に収まる。
pub fn long_context_engine(name: &str, seed: Option<u64>) -> Result<Engine> {
    let tok = dummy_tokenizer()?;
    Ok(Engine::from_parts(
        Box::new(LongKvSensitiveModel::with_context(LONG_CONTEXT)),
        tok,
        name,
        seed,
    ))
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
        // P17e: DummyModel と同じく書き込み前に容量確保 (需要ベース契約)
        let _ = cache.kv_bufs(0, tokens.len());
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

/// A4: 再帰状態 (GDN スロット) と KV の**両方**に依存する決定論的ダミーモデル。
///
/// Qwen3.5 GDN 向け prefix cache 復元 (KV + GDN 状態のペア復元) を engine 経路で
/// 検証するためのもの。DummyModel は状態を参照せず、KvSensitiveModel は KV のみ
/// 参照するため、GDN 状態の退避/復元が壊れても logits が変わらず検証にならない。
/// このモデルは:
/// - GDN スロット (層 0: conv 2 / ssm 1) を持ち、トークンごとに `s ← s + t/8`、
///   `conv ← [直前トークン, t]` と更新する (状態はシーケンス全体の関数 —
///   収縮しないので、状態の欠落がスコア差として**減衰せず残り**、argmax が
///   必ずずれる: 差 Δs = Σt_prompt/8 に対し score 差 = Δs × 41 × 7 ≡ 一定 mod 16)
/// - KV 行も KvSensitiveModel 同様 (トークン + 位置に依存する決定論的値) に書く
/// - logits の argmax を「GDN 状態と KV 全行の加重和の合成スコア」から決める
///
/// → GDN 状態の復元が欠けると greedy 列の**先頭トークンから**変わり、
///   KV 復元が欠けると以降のステップで変わるため、ペア復元の検証に使える。
pub struct RecurStateModel;

impl Model for RecurStateModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        &[DUMMY_EOS]
    }

    fn is_recurrent(&self) -> bool {
        true
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let pos = cache.len();
        let kv_dim = cache.kv_dim();
        for (i, &t) in tokens.iter().enumerate() {
            let p = pos + i;
            // KV 行を書く (層 0 = KV 層・KvSensitiveModel と同じ決定論的値)
            {
                let (k, v) = cache.kv_bufs(0, tokens.len());
                for d in 0..kv_dim {
                    k[p * kv_dim + d] = KvSensitiveModel::kv_k(t, p, d, kv_dim);
                    v[p * kv_dim + d] = KvSensitiveModel::kv_v(t, p, d, kv_dim);
                }
            }
            // GDN 状態を更新 (層 1 = GDN 層・t に依存する決定論的遷移・
            // 非収縮なので過去トークンの寄与が状態差として残り続ける)
            {
                let (conv, ssm) = cache.gdn_state_mut(1).expect("GDN スロット未初期化");
                let prev = conv[1];
                ssm[0] += t as f32 * 0.125;
                conv.copy_from_slice(&[prev, t as f32]);
            }
        }
        cache.advance(tokens.len());
        // logits: GDN 状態 (s・conv) と KV 全行の加重和の合成スコアから決める。
        // 状態依存を強くするため s に大きい係数を掛け、状態が違えば argmax が
        // 最初のステップで変わるようにする。
        let (s, conv_last);
        {
            let (conv, ssm) = cache.gdn_state_mut(1).expect("GDN スロット未初期化");
            s = ssm[0];
            conv_last = conv[1];
        }
        let len = cache.len();
        let kv_acc;
        {
            let (k, _) = cache.kv_bufs_ref(0);
            let mut a = 0.0f32;
            for p in 0..len {
                for d in 0..kv_dim {
                    a += k[p * kv_dim + d] * ((p % 4 + 1) as f32);
                }
            }
            kv_acc = a;
        }
        let score = s * 41.0 + conv_last * 3.0 + kv_acc;
        let mut logits = vec![0.0f32; DUMMY_VOCAB];
        let peak = ((score * 7.0).round() as i64).rem_euclid(DUMMY_VOCAB as i64) as usize;
        let second = ((score * 5.0 + 3.0).round() as i64).rem_euclid(DUMMY_VOCAB as i64) as usize;
        logits[peak] = 5.0;
        logits[second] = 1.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        // 層 0 = KV 層・層 1 = GDN 層 (conv 2 / ssm 1)
        let mut c = KvCache::new(2, 4, DUMMY_CONTEXT);
        c.attach_gdn_slot(1, 2, 1);
        c
    }
}

/// RecurStateModel (状態依存ダミー) で Engine を構築する
pub fn recurrent_engine(name: &str, seed: Option<u64>) -> Result<Engine> {
    let tok = dummy_tokenizer()?;
    Ok(Engine::from_parts(Box::new(RecurStateModel), tok, name, seed))
}

/// s47: thinking 制御の挙動をトークンレベルで模倣する決定論的ダミーモデル
/// (qwen_style_tokenizer 前提・語彙 0..=22)。
///
/// - 直前トークンが </think> (22・thinking off の空 think block 末尾) なら即 EOS
///   (空の think block が閉じている = 思考済み → 即答)
/// - それ以外 (デフォルト/brief 経路の prompt 末尾 = assistant 直後の開いた
///   think block) はフィラ 3 トークン (t13→t14→t15) の後に EOS (思考してから答える)
///   (テスト用トークナイザは generation prompt 末尾の改行を落とすため、思考 ON の
///   prompt は assistant (19) で終わる = 開いた think block の扱い)
///
/// → thinking=off で「思考文が長く続かず短時間で完答する」ことと
///   off/full で prompt (→ 生成長) が変わることを出力トークン数で観測できる。
pub struct ThinkingSimModel;

/// qwen_style_tokenizer の語彙数 (t0..t15, eos, `<|im_start|>`, `<|im_end|>`,
/// user, assistant, <think>, [UNK], </think>)
pub const THINK_SIM_VOCAB: usize = 23;

impl Model for ThinkingSimModel {
    fn context(&self) -> usize {
        DUMMY_CONTEXT
    }

    fn eos_ids(&self) -> &[u32] {
        // qwen_style 語彙の "eos"
        &[12]
    }

    fn forward(&mut self, tokens: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let _ = cache.kv_bufs(0, tokens.len());
        cache.advance(tokens.len());
        let last = tokens.last().copied().unwrap_or(0);
        let mut logits = vec![0.0f32; THINK_SIM_VOCAB];
        let peak: usize = match last {
            22 => 12, // </think> (空 think block 末尾) → 即答 (EOS)
            20 => 13, // <think> → 思考開始 (フィラへ)
            13 => 14,
            14 => 15,
            15 => 12, // 思考終わり → EOS
            _ => 13,  // その他 (assistant 直後 = 開いた think block) → 思考開始
        };
        logits[peak] = 5.0;
        Ok(logits)
    }

    fn new_cache(&self) -> KvCache {
        KvCache::new(1, 4, DUMMY_CONTEXT)
    }
}

/// ThinkingSimModel + Qwen3.5 風テンプレートで Engine を構築する
/// (s47: thinking 制御の engine 経路検証用)。
pub fn thinking_sim_engine(name: &str) -> Result<Engine> {
    let tok = qwen_style_tokenizer()?;
    Ok(Engine::from_parts(Box::new(ThinkingSimModel), tok, name, None))
}

/// s46: Qwen3.5 風 chat template 用トークナイザ (WordLevel・whitespace 前処理)。
/// テンプレートは毎回 prompt を `<|im_start|>assistant\n\n\n` (→ 末尾に
/// `<think>` トークンが残る) で終えるため、**次ターンの描画は直前 prompt の
/// トークン接頭辞にならない** (実 Qwen3.5 と同じ prefix 崩れを再現する)。
/// 語彙: t0..t11=0..11, eos=12, t13..t15=13..15, `<|im_start|>`=16,
/// `<|im_end|>`=17, user=18, assistant=19, `<think>`=20, [UNK]=21,
/// s47: `</think>`=22 (thinking=off の空 think block 用・added_tokens にも登録 =
/// 実 Qwen3.5 と同一・special=false・id は語彙側と同一を再利用)。
/// added_tokens は前処理より先に照合されるため Whitespace 前処理で分割されず
/// 単一トークンになる = テンプレートが `<think>\n\n\n</think>` と連続で並んでも
/// [20, 22] として出る (id は語彙側と同一を再利用)。
pub const QWEN_STYLE_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 20, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 22, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "t0": 0, "t1": 1, "t2": 2, "t3": 3, "t4": 4, "t5": 5, "t6": 6,
      "t7": 7, "t8": 8, "t9": 9, "t10": 10, "t11": 11, "eos": 12,
      "t13": 13, "t14": 14, "t15": 15,
      "<|im_start|>": 16, "<|im_end|>": 17, "user": 18, "assistant": 19,
      "<think>": 20, "[UNK]": 21, "</think>": 22
    },
    "unk_token": "[UNK]"
  }
}"#;

/// Qwen3.5 風 chat template (ChatML + `<think>` 付き generation prompt)。
/// turn_{n+1} の描画は turn_n の prompt 末尾の generation prompt
/// (`<|im_start|>assistant\n\n\n` のうち `<think>` トークン) を assistant 本文で
/// 置き換えるため、トークン接頭辞が崩れる (s46 の根因と同一構造)。
/// s47: generation prompt は実 Qwen3.5 と同一の分岐構造を持つ
/// (`{% if enable_thinking is defined and enable_thinking is false %}`)。
/// 既定/brief/full: `<|im_start|>assistant\n\n\n` = assistant 改行 + 開いた
/// think block (s46 と同一のトークン列)。off: `assistant\n` + `<think>` +
/// `\n\n\n` + `</think>` + `\n\n` = **think 終了タグの直前が改行3つ** (実モデルの
/// chat_template.jinja と同一の anchor・間の改行はトークン化で落ちる)。
pub const QWEN_STYLE_TEMPLATE: &str = "{%- for m in messages %}\
<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
{%- endfor %}\
{%- if add_generation_prompt %}<|im_start|>assistant\n\
{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n\n</think>\n\n\
{%- else %}\n\n{%- endif %}\
{%- endif %}";

/// Qwen3.5 風テンプレートのトークナイザ
pub fn qwen_style_tokenizer() -> Result<DennoTokenizer> {
    DennoTokenizer::from_json_str(QWEN_STYLE_TOKENIZER_JSON, Some(QWEN_STYLE_TEMPLATE.into()))
}

/// RecurStateModel + Qwen3.5 風テンプレートで Engine を構築する
/// (s46: generation prompt 境界の split prefill + base エントリ復元の検証用)。
pub fn recurrent_qwen_style_engine(name: &str, seed: Option<u64>) -> Result<Engine> {
    let tok = qwen_style_tokenizer()?;
    Ok(Engine::from_parts(Box::new(RecurStateModel), tok, name, seed))
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
