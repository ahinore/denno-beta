//! P-MOE-F5: `qwen4exp` の貪欲 decode 駆動部。
//!
//! [`Qwen4Exp::forward_token`] は 1 トークンぶんの logits を返すところまで出来て
//! いる。このモジュールはその上に「プロンプトを食わせて N トークン伸ばす」層だけ
//! を足す。**新しい数式は書かない** — トークンの埋め込み・層ループ・logits は
//! すべて `Qwen4Exp` の中にある。
//!
//! attention の実装は [`crate::model::qwen4exp::LayerAttn`] 越しなので、
//! ここは `Qwen4Exp` が持っている `attn` をそのまま使う (F4 の full attention が
//! 入っても駆動部は変えなくてよい)。
//!
//! # 決定性
//!
//! 貪欲の規則は 3 つだけ:
//!
//! 1. logits は **f32 の生値**で比べる (丸めない・量子化しない)。
//! 2. 同値のときは **添字が小さい方**を採る (`>` で更新し `>=` にしない)。
//! 3. NaN が 1 つでも混じったらエラーにする (黙って 0 番を返さない)。
//!
//! # 文脈の渡し方
//!
//! [`Qwen4Exp::forward_token`] の契約は「`ctx` は古い順で末尾が現在のトークン」。
//! このモジュールは `tokens` (食わせた全トークン) の**末尾 `ple_window` 個**を
//! そのまま古い順で渡す。PLE の n-gram 索引は `ctx` の末尾 3 個しか見ないので、
//! 窓を切っても意味は変わらず、履歴長に依存しない。
//!
//! 注意: `ple.rs` の [`crate::ple::PleNgram::indices`] は「`ctx[0]` が現在の
//! トークン」(新しい順) を前提にしている。`qwen4exp.rs` の `forward_token_with`
//! は末尾 `n_gram` 個を反転してから `gather` に渡す (P-MOE-F7 で修正済み)。
//! このモジュールは古い順のまま渡してよい。

use crate::model::qwen4exp::{one, Qwen4Exp, Qwen4ExpError};

/// `qwen4exp.rs` と同じ「1 行 1 件」のエラー型を使う。
type Result<T> = std::result::Result<T, Qwen4ExpError>;

/// PLE 索引の窓の既定値 (実モデルの `ple.ngram_size` = 3)。
pub const DEFAULT_PLE_WINDOW: usize = 3;

/// decode の駆動設定。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecoderConfig {
    /// KV / ssm 状態を保持する上限トークン数 (プロンプト + 生成の合計)。
    /// 超えたらエラーにする (黙って古いトークンを捨てない。窓の管理は後で設計する)。
    pub max_ctx: usize,
    /// PLE 索引に渡す履歴の窓 (`qwen4exp.ple.ngram_size`)。既定は 3。
    pub ple_window: usize,
}

impl DecoderConfig {
    /// `max_ctx` だけ決めて、PLE の窓は既定 (`DEFAULT_PLE_WINDOW`) にする。
    pub fn new(max_ctx: usize) -> Self {
        DecoderConfig {
            max_ctx,
            ple_window: DEFAULT_PLE_WINDOW,
        }
    }
}

/// logits の argmax。**f32 の生値**で比べ、同値なら添字が小さい方を返す。
///
/// - `logits` が空ならエラー (0 番を返さない)
/// - NaN が 1 つでもあればエラー (位置をメッセージに出す)
/// - `+inf` / `-inf` は比較可能なのでそのまま扱う (NaN だけを弾く)
pub fn argmax(logits: &[f32]) -> Result<usize> {
    if logits.is_empty() {
        return Err(one("argmax: logits が空"));
    }
    let mut best = 0usize;
    let mut best_v = logits[0];
    if best_v.is_nan() {
        return Err(one("argmax: logits[0] が NaN"));
    }
    for (i, &v) in logits.iter().enumerate().skip(1) {
        if v.is_nan() {
            return Err(one(format!("argmax: logits[{i}] が NaN")));
        }
        // `>` で更新する (`>=` にすると同値のとき後ろの添字が勝つ)
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    Ok(best)
}

/// プロンプトを食わせて N トークン貪欲に伸ばす駆動部。
pub struct Qwen4ExpDecoder {
    model: Qwen4Exp,
    cfg: DecoderConfig,
    /// これまでに食わせたトークン (古い順)
    tokens: Vec<u32>,
}

impl Qwen4ExpDecoder {
    pub fn new(model: Qwen4Exp, cfg: DecoderConfig) -> Self {
        Qwen4ExpDecoder {
            model,
            cfg,
            tokens: Vec::new(),
        }
    }

    /// これまでに食わせたトークン (古い順)。
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// 使っている設定。
    pub fn cfg(&self) -> &DecoderConfig {
        &self.cfg
    }

    /// 1 トークン進めて logits を返す。`tokens` の末尾に積む。
    ///
    /// `max_ctx` を超えるときは **積む前に** エラーを返す。forward が途中で
    /// 失敗した場合は `tokens` にそのトークンが残り、ssm / attention の状態は
    /// 進んだところまで進んでいる (その decoder は作り直すこと)。
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>> {
        self.step_with(token, |_, _| Ok(()))
    }

    /// プロンプトを全部食わせて、最後の logits を返す。
    ///
    /// 1 トークンずつ [`Qwen4ExpDecoder::step`] を呼ぶ。**最後の 1 回だけ**
    /// logits を返す (途中の logits は捨てる)。空のプロンプトはエラー。
    pub fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>> {
        if prompt.is_empty() {
            return Err(one("prefill: プロンプトが空"));
        }
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.step(t)?;
        }
        Ok(logits)
    }

    /// 貪欲に `n_new` 個伸ばす。生成した id を返す (プロンプトは含めない)。
    ///
    /// `n_new == 0` は何も食わせずに空を返す (エラーにしない)。
    /// 最後の 1 トークンも forward する (`tokens` と状態を食い違わせない)。
    pub fn decode_greedy(&mut self, prompt: &[u32], n_new: usize) -> Result<Vec<u32>> {
        if n_new == 0 {
            return Ok(Vec::new());
        }
        let mut logits = self.prefill(prompt)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let id = argmax(&logits)?;
            let id = u32::try_from(id).map_err(|_| one(format!("argmax の {id} が u32 に収まらない")))?;
            out.push(id);
            logits = self.step(id)?;
        }
        Ok(out)
    }

    /// 貪欲に伸ばし、`stop` に入った id を出したら止める。
    ///
    /// 停止トークンも `out` に含める (呼び出し側が detokenize して落とせる)。
    /// `n_new` に達した場合も止まる。それ以外は [`Self::decode_greedy`] と同じ。
    pub fn decode_greedy_stop(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        stop: &[u32],
    ) -> Result<Vec<u32>> {
        if n_new == 0 {
            return Ok(Vec::new());
        }
        let mut logits = self.prefill(prompt)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let id = argmax(&logits)?;
            let id = u32::try_from(id).map_err(|_| one(format!("argmax の {id} が u32 に収まらない")))?;
            out.push(id);
            if stop.contains(&id) {
                break;
            }
            logits = self.step(id)?;
        }
        Ok(out)
    }

    /// 層別 trace 付きの貪欲 decode。
    ///
    /// `on_layer(step, il, norm, head)` を、生成した各トークンの各層の出口で
    /// 呼ぶ (`step` は生成の 0 始まり、`il` は層番号、`norm` は `res_hc` の
    /// L2 ノルム、`head` は先頭 4 要素)。**プロンプトの forward では呼ばない**
    /// ので、フックの回数は `n_new * n_layer`。
    ///
    /// `norm` は f64 で二乗和してから f32 に落とす (実モデルの `||res_hc||` は
    /// 1e2 級なので f32 の逐次加算では桁が落ちる)。NaN/inf が出たらエラー。
    pub fn decode_greedy_traced<F>(
        &mut self,
        prompt: &[u32],
        n_new: usize,
        mut on_layer: F,
    ) -> Result<Vec<u32>>
    where
        F: FnMut(usize, usize, f32, [f32; 4]) -> Result<()>,
    {
        if n_new == 0 {
            return Ok(Vec::new());
        }
        let mut logits = self.prefill(prompt)?;
        let mut out = Vec::with_capacity(n_new);
        for step in 0..n_new {
            let id = argmax(&logits)?;
            let id = u32::try_from(id).map_err(|_| one(format!("argmax の {id} が u32 に収まらない")))?;
            out.push(id);
            logits = self.step_with(id, |il, res| {
                let (norm, head) = layer_trace(step, il, res)?;
                on_layer(step, il, norm, head)
            })?;
        }
        Ok(out)
    }

    /// [`Qwen4ExpDecoder::step`] の層フック付き版 (内部用)。
    fn step_with<F>(&mut self, token: u32, mut on_layer: F) -> Result<Vec<f32>>
    where
        F: FnMut(usize, &[f32]) -> Result<()>,
    {
        let max = self.cfg.max_ctx;
        if max == 0 {
            return Err(one("max_ctx が 0 (1 トークンも食わせられない)"));
        }
        if self.tokens.len() >= max {
            return Err(one(format!(
                "max_ctx {max} を超える: すでに {} トークンを食わせた",
                self.tokens.len()
            )));
        }
        let win = self.cfg.ple_window;
        if win == 0 {
            return Err(one("ple_window が 0 (PLE 索引の窓が無い)"));
        }

        self.tokens.push(token);

        // 末尾 `win` 個を古い順のまま渡す (窓を切っても末尾の意味は変わらない)。
        let start = self.tokens.len().saturating_sub(win);
        let ctx: Vec<u32> = self.tokens[start..].to_vec();

        self.model
            .forward_token_with(&ctx, |il, res| on_layer(il, res))
    }
}

/// 層の出口の `res_hc` から (L2 ノルム, 先頭 4 要素) を作る。
///
/// `n_embd * n_hc >= 4` は `validate` が保証している (`n_embd >= 1`) が、
/// 念のため足りないときはエラーにする (添字 panic を起こさない)。
fn layer_trace(step: usize, il: usize, res: &[f32]) -> Result<(f32, [f32; 4])> {
    if res.len() < 4 {
        return Err(one(format!(
            "step {step} 層 {il}: res_hc の長さ {} が 4 未満",
            res.len()
        )));
    }
    let mut ss = 0.0f64;
    for (i, &v) in res.iter().enumerate() {
        if !v.is_finite() {
            return Err(one(format!(
                "step {step} 層 {il}: res_hc[{i}] が有限でない ({v})"
            )));
        }
        ss += (v as f64) * (v as f64);
    }
    let norm = ss.sqrt() as f32;
    if !norm.is_finite() {
        return Err(one(format!(
            "step {step} 層 {il}: ||res_hc|| が有限でない ({norm})"
        )));
    }
    Ok((norm, [res[0], res[1], res[2], res[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_takes_smallest_index_on_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]).expect("argmax"), 1);
        assert_eq!(argmax(&[3.0, 3.0, 3.0]).expect("argmax"), 0);
        assert_eq!(argmax(&[-0.0, 0.0]).expect("argmax"), 0, "+0 と -0 は同値");
    }

    #[test]
    fn argmax_uses_raw_f32_bits() {
        // f32 で隣り合う 2 値は丸めずに比べる
        let a = 1.0f32;
        let b = f32::from_bits(a.to_bits() + 1);
        assert!(b > a);
        assert_eq!(argmax(&[a, b]).expect("argmax"), 1);
        assert_eq!(argmax(&[b, a]).expect("argmax"), 0);
    }

    #[test]
    fn argmax_rejects_nan_and_empty() {
        let e = argmax(&[]).expect_err("空で Ok が返った");
        assert!(e.to_string().contains("空"), "{e}");
        let e = argmax(&[0.0, f32::NAN, 1.0]).expect_err("NaN で Ok が返った");
        assert!(e.to_string().contains("NaN"), "{e}");
        assert!(e.to_string().contains("logits[1]"), "{e}");
        let e = argmax(&[f32::NAN]).expect_err("先頭 NaN で Ok が返った");
        assert!(e.to_string().contains("logits[0]"), "{e}");
    }
}
