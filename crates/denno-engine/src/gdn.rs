//! Gated DeltaNet (Qwen3.5 linear_attention 層) の CPU リファレンス実装 (P19-B)。
//!
//! llama.cpp `src/models/delta-net-base.cpp` の build_delta_net_autoregressive
//! (+ qwen35.cpp のグラフ構成) を 1:1 で移植した逐次形式。
//! - 参照実装: llama.cpp (https://github.com/ggml-org/llama.cpp)
//!   MIT License — Copyright (c) 2023-2026 The ggml authors。
//!   本ファイルは数学・演算順序の移植であり、コードの逐語コピーは含まない。
//! - 状態 S は head ごとに [head_k_dim × head_v_dim] の f32 (mamba_ssm_dtype=float32)。
//! - 逐次 AR 形式は chunked 形式と数学的に等価 (精度差のみ)。denno 内の契約は
//!   「同一入力→同一出力・決定論」で、llama.cpp とのビット一致は要求しない。
//!
//! 数学 (head h, 1 トークン):
//! 1. g[h] = ssm_a[h] * softplus(alpha[h] + ssm_dt[h])      (ssm_a は負値 = -exp(A_log))
//! 2. S ← S * exp(g[h])
//! 3. sk[c] = Σ_r k[r] * S[r][c]
//! 4. d[c] = beta[h] * (v[c] - sk[c])
//! 5. S[r][c] += k[r] * d[c]
//! 6. o[c] = Σ_r q[r] * S[r][c]
//! q,k は事前に L2 正規化 (x/sqrt(Σx²+eps))、q にのみ 1/sqrt(head_k_dim) を掛ける。
//! 出力は head ごとに gated RMS norm: rmsnorm(o)*w * silu(z)。

/// GDN ハイパーパラメータ (Qwen3.5 config の linear_* に対応)
#[derive(Debug, Clone)]
pub struct GdnParams {
    /// hidden (n_embd)
    pub n_embd: usize,
    /// linear_num_key_heads (ssm_n_group)
    pub n_k_heads: usize,
    /// linear_num_value_heads (ssm_dt_rank)
    pub n_v_heads: usize,
    /// linear_key_head_dim (ssm_d_state)
    pub head_k_dim: usize,
    /// linear_value_head_dim
    pub head_v_dim: usize,
    /// linear_conv_kernel_dim
    pub d_conv: usize,
    /// rms_norm_eps
    pub eps: f32,
}

impl GdnParams {
    /// 0.8B: hidden 1024 / K16 V16 / dk 128 / dv 128 / conv 4
    pub fn qwen35_08b() -> Self {
        GdnParams {
            n_embd: 1024,
            n_k_heads: 16,
            n_v_heads: 16,
            head_k_dim: 128,
            head_v_dim: 128,
            d_conv: 4,
            eps: 1e-6,
        }
    }

    pub fn key_dim(&self) -> usize {
        self.head_k_dim * self.n_k_heads
    }
    pub fn value_dim(&self) -> usize {
        self.head_v_dim * self.n_v_heads
    }
    /// q + k + v の投影次元
    pub fn qkv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
    pub fn conv_dim(&self) -> usize {
        self.qkv_dim()
    }
    pub fn conv_state_len(&self) -> usize {
        self.conv_dim() * (self.d_conv - 1)
    }
    pub fn ssm_state_len(&self) -> usize {
        self.n_v_heads * self.head_k_dim * self.head_v_dim
    }
}

/// GDN 重み (すべて f32・[out][in] 行優先。conv は [channel][tap])。
#[derive(Debug, Clone)]
pub struct GdnWeights {
    pub w_qkv: Vec<f32>,   // [qkv_dim][n_embd]
    pub w_gate: Vec<f32>,  // [value_dim][n_embd]   (z)
    pub w_beta: Vec<f32>,  // [n_v_heads][n_embd]
    pub w_alpha: Vec<f32>, // [n_v_heads][n_embd]
    pub conv1d: Vec<f32>,  // [conv_dim][d_conv]
    pub ssm_a: Vec<f32>,   // [n_v_heads] (負値)
    pub ssm_dt: Vec<f32>,  // [n_v_heads] (dt bias)
    pub ssm_norm: Vec<f32>, // [head_v_dim]
    pub w_out: Vec<f32>,   // [n_embd][value_dim]
}

/// レイヤー状態 (conv バッファ + SSM 行列)。
#[derive(Debug, Clone, PartialEq)]
pub struct GdnState {
    /// [conv_dim][d_conv-1] — チャネルごとに直前 d_conv-1 入力
    pub conv: Vec<f32>,
    /// [n_v_heads][head_k_dim][head_v_dim] — S[h][r][c]
    pub ssm: Vec<f32>,
}

impl GdnState {
    pub fn new(p: &GdnParams) -> Self {
        GdnState {
            conv: vec![0.0; p.conv_state_len()],
            ssm: vec![0.0; p.ssm_state_len()],
        }
    }
}

#[inline]
fn softplus(x: f32) -> f32 {
    // ggml_compute_softplus_f32 と同一規約
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// L2 正規化 (build_gdn_l2_norm と同一数学: x/sqrt(Σx²+eps))
#[inline]
fn l2_norm(x: &mut [f32], eps: f32) {
    let s: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (s + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// RMS 正規化 (build_norm LLM_NORM_RMS と同一数学: x/sqrt(mean(x²)+eps))
#[inline]
fn rms_norm(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let s: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (s / n + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// 行列ベクト�積: out[i] = Σ_j w[i*in_dim + j] * x[j]
#[inline]
fn matvec(w: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; out_dim];
    for (i, o) in out.iter_mut().enumerate() {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut acc = 0.0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            acc += a * b;
        }
        *o = acc;
    }
    out
}

impl GdnParams {
    /// 逐次 GDN 1 層 forward。x: [T][n_embd]、state を更新しながら全トークン処理。
    /// 戻り値: [T][n_embd]。
    pub fn forward(
        &self,
        w: &GdnWeights,
        x: &[Vec<f32>],
        state: &mut GdnState,
    ) -> Vec<Vec<f32>> {
        let t_len = x.len();
        let mut out = Vec::with_capacity(t_len);
        for xt in x {
            out.push(self.forward_token(w, xt, state));
        }
        out
    }

    /// 1 トークンの forward (AR パス・llama.cpp と同一順序)。
    pub fn forward_token(&self, w: &GdnWeights, x: &[f32], st: &mut GdnState) -> Vec<f32> {
        let qkv = matvec(&w.w_qkv, x, self.qkv_dim(), self.n_embd);
        let z = matvec(&w.w_gate, x, self.value_dim(), self.n_embd);
        let beta_raw = matvec(&w.w_beta, x, self.n_v_heads, self.n_embd);
        let alpha = matvec(&w.w_alpha, x, self.n_v_heads, self.n_embd);
        let cw = GdnCore {
            conv1d: &w.conv1d,
            ssm_a: &w.ssm_a,
            ssm_dt: &w.ssm_dt,
            ssm_norm: &w.ssm_norm,
        };
        let y = forward_token_core(self, &cw, &qkv, &z, &beta_raw, &alpha, st);
        matvec(&w.w_out, &y, self.n_embd, self.value_dim())
    }
}

/// 逆畳み込み/SSM 部の重み (小さい f32 テンソル群)。モデル側では
/// Linear (int8 再量子化) で持つ大きい行列 (qkv/gate/beta/alpha/out) と
/// 分離して受けるためのビュー。
pub struct GdnCore<'a> {
    pub conv1d: &'a [f32],  // [conv_dim][d_conv]
    pub ssm_a: &'a [f32],   // [n_v_heads] (負値)
    pub ssm_dt: &'a [f32],  // [n_v_heads]
    pub ssm_norm: &'a [f32], // [head_v_dim]
}

/// コア本体: 投影済み qkv/z/beta/alpha から gated-norm 済み出力 [value_dim] を返す。
/// forward_token と同じ演算順序 (bit 一致)。
pub fn forward_token_core(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
) -> Vec<f32> {
    // ---- causal depthwise conv + silu --------------------------------
    // 入力列 = [conv_state (d_conv-1), qkv_t] をチャネルごとに畳み込み
    let cs = p.d_conv - 1;
    let cd = p.conv_dim();
    let mut conv_in = vec![0.0f32; cd * p.d_conv];
    for c in 0..cd {
        conv_in[c * p.d_conv..c * p.d_conv + cs]
            .copy_from_slice(&st.conv[c * cs..(c + 1) * cs]);
        conv_in[c * p.d_conv + cs] = qkv[c];
    }
    // 状態をロール: 新 state = 直近 d_conv-1 入力
    for c in 0..cd {
        st.conv[c * cs..(c + 1) * cs]
            .copy_from_slice(&conv_in[c * p.d_conv + 1..(c + 1) * p.d_conv]);
    }
    let mut conv_out = vec![0.0f32; cd];
    for c in 0..cd {
        let row = &cw.conv1d[c * p.d_conv..(c + 1) * p.d_conv];
        let inp = &conv_in[c * p.d_conv..(c + 1) * p.d_conv];
        let mut acc = 0.0f32;
        for (a, b) in row.iter().zip(inp.iter()) {
            acc += a * b;
        }
        conv_out[c] = silu(acc);
    }

    // ---- split q/k/v ---------------------------------------------------
    let kd = p.key_dim();
    let vd = p.value_dim();
    let q_raw = &conv_out[0..kd];
    let k_raw = &conv_out[kd..2 * kd];
    let v_raw = &conv_out[2 * kd..2 * kd + vd];

    // ---- L2 正規化 (head ごと) + q スケール -----------------------------
    let q_scale = 1.0 / (p.head_k_dim as f32).sqrt();
    let mut q = vec![0.0f32; kd];
    let mut k = vec![0.0f32; kd];
    for h in 0..p.n_k_heads {
        let o = h * p.head_k_dim;
        q[o..o + p.head_k_dim].copy_from_slice(&q_raw[o..o + p.head_k_dim]);
        k[o..o + p.head_k_dim].copy_from_slice(&k_raw[o..o + p.head_k_dim]);
        l2_norm(&mut q[o..o + p.head_k_dim], p.eps);
        l2_norm(&mut k[o..o + p.head_k_dim], p.eps);
        for v in q[o..o + p.head_k_dim].iter_mut() {
            *v *= q_scale;
        }
    }

    // ---- デルタルール (head ごと・AR 形式) ------------------------------
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;
    let mut o = vec![0.0f32; vd];
    for h in 0..p.n_v_heads {
        // k/q は n_k_heads と n_v_heads が異なる場合は repeat_interleave 規約
        // (HF repeat_interleave / llama.cpp ggml_repeat = blocked タイリング:
        //  v-head h → k-head h/(nv/nk)。0.8B は nk==nv で同一・9B は nk=16/nv=32)
        let kh = h * p.n_k_heads / p.n_v_heads;
        let qh = &q[kh * dk..kh * dk + dk];
        let kh_v = &k[kh * dk..kh * dk + dk];
        let vh = &v_raw[h * dv..h * dv + dv];

        // decay: g = ssm_a * softplus(alpha + dt)
        let g = cw.ssm_a[h] * softplus(alpha[h] + cw.ssm_dt[h]);
        let eg = g.exp();

        // S ← S * exp(g)   (S[h] は [dk][dv] 行優先)
        let s_base = h * dk * dv;
        for r in 0..dk {
            for c in 0..dv {
                st.ssm[s_base + r * dv + c] *= eg;
            }
        }

        // sk[c] = Σ_r k[r] * S[r][c]
        let mut sk = vec![0.0f32; dv];
        for r in 0..dk {
            let kr = kh_v[r];
            if kr != 0.0 {
                let row = &st.ssm[s_base + r * dv..s_base + r * dv + dv];
                for (c, skc) in sk.iter_mut().enumerate() {
                    *skc += kr * row[c];
                }
            }
        }

        // d[c] = beta * (v[c] - sk[c])
        let b = sigmoid(beta_raw[h]);
        let mut d = vec![0.0f32; dv];
        for c in 0..dv {
            d[c] = b * (vh[c] - sk[c]);
        }

        // S[r][c] += k[r] * d[c]
        for r in 0..dk {
            let kr = kh_v[r];
            if kr != 0.0 {
                let row = &mut st.ssm[s_base + r * dv..s_base + r * dv + dv];
                for (c, dc) in d.iter().enumerate() {
                    row[c] += kr * dc;
                }
            }
        }

        // o[c] = Σ_r q[r] * S[r][c]
        let obase = h * dv;
        for r in 0..dk {
            let qr = qh[r];
            if qr != 0.0 {
                let row = &st.ssm[s_base + r * dv..s_base + r * dv + dv];
                for (c, oc) in o[obase..obase + dv].iter_mut().enumerate() {
                    *oc += qr * row[c];
                }
            }
        }
    }

    // ---- gated RMS norm (head ごと): rmsnorm(o)*w * silu(z) -------------
    let mut y = vec![0.0f32; vd];
    for h in 0..p.n_v_heads {
        let obase = h * dv;
        let mut seg: Vec<f32> = o[obase..obase + dv].to_vec();
        rms_norm(&mut seg, p.eps);
        for (c, v) in seg.iter_mut().enumerate() {
            *v *= cw.ssm_norm[c] * silu(z[obase + c]);
        }
        y[obase..obase + dv].copy_from_slice(&seg);
    }

    y
}

// ---------------------------------------------------------------------------
// A1-r2: GDN chunked scan (長 prefill 用・3-pass 構成)
// ---------------------------------------------------------------------------
//
// 長い prefill の GDN スキャンを chunk 単位に分割し、chunk 境界で状態を持ち越す。
// デルタルールの更新は S への**線形**写像
//   S ← eg·(I − beta·k kᵀ)·S + beta·k vᵀ   (eg = exp(g) はスカラー減衰)
// で表せるため、chunk をまたぐ依存を「ローカル遷移 (W, M) の計算」と
// 「状態の伝播」に分解できる:
//
//   pass 1 (chunk 並列): 各 chunk ci で
//     - W[ci] = S = 0 起点のローカル AR スキャンの終状態
//     - M[ci] = Π_t eg_t (I − beta_t k_t k_tᵀ)  (恒等行列から列ごとに伝播)
//   pass 2 (chunk 直列): S_init[0] = S_in;  S_init[ci+1] = M[ci]·S_init[ci] + W[ci]
//     (S_init[ci] = chunk ci 開始時点の真の状態)
//   pass 3 (chunk 並列): S_init[ci] から chunk 内を逐次 AR スキャン → y と最終状態
//
// 数学は逐次 AR (forward_token_core) と完全等価 (線形性)。fp32 の丸め順のみ
// 異なるため y は bit 一致しない (GDN は bit 一致不要契約・決定論のみ保証)。
// 単一 chunk (n_chunks == 1) では pass 3 が逐次 AR と同じ演算列になり bit 一致。
//
// 数値安全性: k は L2 正規化 (||k||≈1)・beta = sigmoid ∈ (0,1)・
// eg = exp(g) < 1 (ssm_a < 0 × softplus > 0 → g < 0) なので各因子のスペクトル
// ノルム ≤ 1・M は縮小写像となり丸め誤差が増幅しない。
//
// GPU 実装は kernels/gdn.cu の dno_gdn_chunk_pre / dno_gdn_chunk_state /
// dno_gdn_chunk_scan (本関数と同一の演算順序)。本関数はその参照実装であり
// CPU 経路 (qwen35 の prefill/decode) からは呼ばれない (既存契約「逐次 ==
// バッチ prefill bit 一致」を維持するため)。

/// chunk 分割計画: (n_chunks, chunk_len)。
/// n <= min_tokens なら 1 chunk (逐次相当)。そうでなければ
/// n_chunks = min(max_chunks, ceil(n / chunk_tokens))・
/// chunk_len = ceil(n / n_chunks) (最終 chunk に端数が集中・全 chunk 非空)。
pub fn chunk_plan(
    n: usize,
    min_tokens: usize,
    chunk_tokens: usize,
    max_chunks: usize,
) -> (usize, usize) {
    assert!(chunk_tokens >= 1 && max_chunks >= 1);
    if n == 0 {
        return (1, 0);
    }
    if n <= min_tokens {
        return (1, n);
    }
    let nc = ((n + chunk_tokens - 1) / chunk_tokens).min(max_chunks).max(1);
    let c = (n + nc - 1) / nc;
    (nc, c)
}

/// conv 入力 (時刻 τ = t − cs + back) の値。τ < 0 は初期 conv 状態
/// (conv_st[ch*cs + (τ+cs)])、τ >= 0 は qkv のグローバル行。chunk 境界をまたぐ
/// ウィンドウ (最大 cs トークン分) を chunk 内だけで再構築できるため、
/// pass 1 / pass 3 は chunk ごとに独立に動作できる。
#[inline]
fn conv_input_at(
    qkv: &[f32],
    conv_st: &[f32],
    ch: usize,
    t: usize,
    back: usize,
    conv_dim: usize,
    cs: usize,
) -> f32 {
    let idx = t as i64 - cs as i64 + back as i64;
    if idx >= 0 {
        qkv[idx as usize * conv_dim + ch]
    } else {
        conv_st[ch * cs + (t + back)]
    }
}

/// pass 1 の 1 ステップ: W スキャン (S の 1 列)。forward_token_core のデルタ
/// ルール部分と同一順序 (decay → sk (decay 後への内積) → d → 更新)。
#[inline]
fn w_step_col(dk: usize, k: &[f32], eg: f32, b: f32, v: f32, s_col: &mut [f32]) {
    for r in 0..dk {
        s_col[r] *= eg;
    }
    let mut skv = 0.0f32;
    for r in 0..dk {
        skv += k[r] * s_col[r];
    }
    let dlt = b * (v - skv);
    for r in 0..dk {
        s_col[r] += k[r] * dlt;
    }
}

/// pass 1 の 1 ステップ: M の 1 列。M[:, c] ← eg·(M[:, c] − b·k·(kᵀ M[:, c]))。
/// AR の「decay → decay 後の状態への内積 → 更新」と同一順序で丸めも揃える
/// (恒等行列から始めた M 列は AR の S 列の S0 係数の伝播と同じ演算列になる)。
#[inline]
fn m_step_col(dk: usize, k: &[f32], eg: f32, b: f32, m_col: &mut [f32]) {
    for j in 0..dk {
        m_col[j] *= eg;
    }
    let mut mk = 0.0f32;
    for r in 0..dk {
        mk += k[r] * m_col[r];
    }
    for r in 0..dk {
        m_col[r] -= b * k[r] * mk;
    }
}

/// 全トークンのフロントエンド (conv+SiLU・k の L2 正規化・ゲート) を先出しで
/// 計算する (pass 1 の列走査から共有 — GPU カーネルでは共有メモリに相当)。
/// 演算順序は forward_token_core と同一 (bit 一致)。戻り値:
/// (kall [n][kd] L2 正規化済, vall [n][vd], egall [n][nv], ball [n][nv])。
fn front_end_all(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    conv_st: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let cs = p.d_conv - 1;
    let cd = p.conv_dim();
    let kd = p.key_dim();
    let vd = p.value_dim();
    let nv = p.n_v_heads;
    let mut kall = vec![0.0f32; n * kd];
    let mut vall = vec![0.0f32; n * vd];
    let mut egall = vec![0.0f32; n * nv];
    let mut ball = vec![0.0f32; n * nv];
    let mut kbuf = vec![0.0f32; kd];
    for t in 0..n {
        let row = t * cd;
        // k チャネル: conv + SiLU → head ごとに L2 正規化 (forward_token_core と同一)
        for r in 0..kd {
            let ch = kd + r;
            let mut acc = 0.0f32;
            for j in 0..cs {
                acc += cw.conv1d[ch * p.d_conv + j] * conv_input_at(qkv, conv_st, ch, t, j, cd, cs);
            }
            acc += cw.conv1d[ch * p.d_conv + cs] * qkv[row + ch];
            kbuf[r] = silu(acc);
        }
        for h in 0..p.n_k_heads {
            let o = h * p.head_k_dim;
            l2_norm(&mut kbuf[o..o + p.head_k_dim], p.eps);
        }
        kall[t * kd..(t + 1) * kd].copy_from_slice(&kbuf);
        // v チャネル: conv + SiLU
        for c in 0..vd {
            let ch = 2 * kd + c;
            let mut acc = 0.0f32;
            for j in 0..cs {
                acc += cw.conv1d[ch * p.d_conv + j] * conv_input_at(qkv, conv_st, ch, t, j, cd, cs);
            }
            acc += cw.conv1d[ch * p.d_conv + cs] * qkv[row + ch];
            vall[t * vd + c] = silu(acc);
        }
        // ゲート: eg (指数減衰) と beta
        for h in 0..nv {
            let g = cw.ssm_a[h] * softplus(alpha[t * nv + h] + cw.ssm_dt[h]);
            egall[t * nv + h] = g.exp();
            ball[t * nv + h] = sigmoid(beta_raw[t * nv + h]);
        }
    }
    (kall, vall, egall, ball)
}

/// chunked scan 本体 (投影済み行入力)。`st` は入出力 (最終 conv + SSM 状態)。
/// 戻り値: y [n × value_dim] (gated RMS norm 済み・out_w 投影前)。
/// `n_chunks`/`chunk_len` は [`chunk_plan`] の出力を渡す。
#[allow(clippy::too_many_arguments)]
pub fn scan_chunked_core(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
    n_chunks: usize,
    chunk_len: usize,
) -> Vec<f32> {
    let cs = p.d_conv - 1;
    let cd = p.conv_dim();
    let kd = p.key_dim();
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;
    let vd = p.value_dim();
    let nv = p.n_v_heads;
    let n = z.len() / vd;
    if n == 0 {
        return Vec::new();
    }
    debug_assert_eq!(qkv.len(), n * cd);
    debug_assert_eq!(beta_raw.len(), n * nv);
    debug_assert_eq!(alpha.len(), n * nv);
    debug_assert_eq!(st.conv.len(), cd * cs);
    debug_assert_eq!(st.ssm.len(), nv * dk * dv);
    debug_assert!(n_chunks >= 1 && chunk_len >= 1);

    // ---- フロントエンド先出し (pass 1 / pass 3 で共有) ----------------------
    let conv_in = st.conv.clone();
    let (kall, vall, egall, ball) = front_end_all(p, cw, qkv, beta_raw, alpha, &conv_in, n);

    // ---- pass 1: chunk ごとのローカル遷移 (W[ci], M[ci]) --------------------
    // W: [nc][nv][dk][dv] (行優先 S[r][c])・M: [nc][nv][dk][dk] (行優先 M[r][j])
    let mut wbuf = vec![0.0f32; n_chunks * nv * dk * dv];
    let mut mbuf = vec![0.0f32; n_chunks * nv * dk * dk];
    let mut s_col = vec![0.0f32; dk];
    let mut m_col = vec![0.0f32; dk];
    for ci in 0..n_chunks {
        let t0 = ci * chunk_len;
        let t1 = n.min(t0 + chunk_len);
        for h in 0..nv {
            let kh = h * p.n_k_heads / nv; // repeat_interleave 規約
            let w_off = (ci * nv + h) * dk * dv;
            let m_off = (ci * nv + h) * dk * dk;
            // W スキャン: S = 0 起点の AR (列 c ごと)
            for c in 0..dv {
                for r in 0..dk {
                    s_col[r] = 0.0;
                }
                for t in t0..t1 {
                    let k = &kall[t * kd + kh * dk..t * kd + kh * dk + dk]; // トークン t・k-head kh
                    let b = ball[t * nv + h];
                    let eg = egall[t * nv + h];
                    let v = vall[t * vd + h * dv + c];
                    w_step_col(dk, k, eg, b, v, &mut s_col);
                }
                for r in 0..dk {
                    wbuf[w_off + r * dv + c] = s_col[r];
                }
            }
            // M 伝播: 恒等行列から列ごとに更新
            for c in 0..dk {
                for r in 0..dk {
                    m_col[r] = if r == c { 1.0 } else { 0.0 };
                }
                for t in t0..t1 {
                    let k = &kall[t * kd + kh * dk..t * kd + kh * dk + dk];
                    let b = ball[t * nv + h];
                    let eg = egall[t * nv + h];
                    m_step_col(dk, k, eg, b, &mut m_col);
                }
                for r in 0..dk {
                    mbuf[m_off + r * dk + c] = m_col[r];
                }
            }
        }
    }

    // ---- pass 2: chunk 先頭状態 S_init の直列伝播 ---------------------------
    let mut sinit = vec![0.0f32; n_chunks * nv * dk * dv];
    let mut running = st.ssm.clone(); // S_in
    for ci in 0..n_chunks {
        sinit[ci * nv * dk * dv..(ci + 1) * nv * dk * dv].copy_from_slice(&running);
        // running ← M[ci]·running + W[ci]  (j 昇順の内積 — カーネルと同一順序)
        let mut next = vec![0.0f32; nv * dk * dv];
        for h in 0..nv {
            let m_off = (ci * nv + h) * dk * dk;
            let w_off = (ci * nv + h) * dk * dv;
            for r in 0..dk {
                for c in 0..dv {
                    let mut acc = 0.0f32;
                    for j in 0..dk {
                        acc += mbuf[m_off + r * dk + j] * running[h * dk * dv + j * dv + c];
                    }
                    next[h * dk * dv + r * dv + c] = acc + wbuf[w_off + r * dv + c];
                }
            }
        }
        running = next;
    }

    // ---- pass 3: S_init[ci] からの逐次 AR スキャン (chunk 並列相当) ---------
    let mut y = vec![0.0f32; n * vd];
    for ci in 0..n_chunks {
        let t0 = ci * chunk_len;
        let t1 = n.min(t0 + chunk_len);
        // conv ウィンドウを chunk 開始時点まで巻き戻す (以後は通常どおりロール)。
        // 値は連続 AR のロール結果と一致するため演算は bit 一致する。
        for ch in 0..cd {
            for j in 0..cs {
                st.conv[ch * cs + j] = conv_input_at(qkv, &conv_in, ch, t0, j, cd, cs);
            }
        }
        st.ssm.copy_from_slice(&sinit[ci * nv * dk * dv..(ci + 1) * nv * dk * dv]);
        for t in t0..t1 {
            let y_t = forward_token_core(
                p,
                cw,
                &qkv[t * cd..(t + 1) * cd],
                &z[t * vd..(t + 1) * vd],
                &beta_raw[t * nv..(t + 1) * nv],
                &alpha[t * nv..(t + 1) * nv],
                st,
            );
            y[t * vd..(t + 1) * vd].copy_from_slice(&y_t);
        }
    }
    // 最終状態: st.conv = 最終 chunk のロール結果・st.ssm = 最終 chunk の
    // スキャン終状態 (真の最終状態 — M/W 合成 (pass 2 の running) と数学的に
    // 同一・丸めのみ異なる)
    y
}
#[cfg(test)]
mod tests {
    use super::*;

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// 小型パラメータ: hidden 3 / K1 V1 / dk 2 / dv 2 / conv 4
    fn tiny() -> (GdnParams, GdnWeights) {
        let p = GdnParams {
            n_embd: 3,
            n_k_heads: 1,
            n_v_heads: 1,
            head_k_dim: 2,
            head_v_dim: 2,
            d_conv: 4,
            eps: 1e-6,
        };
        // 決定論的な手作り重み (対称性の崩れた値)
        let w = GdnWeights {
            w_qkv: vec![
                0.1, -0.2, 0.3, // q[0]
                0.4, 0.5, -0.6, // q[1]
                -0.7, 0.8, 0.9, // k[0]
                1.1, -1.2, 1.3, // k[1]
                1.4, 1.5, -1.6, // v[0]
                -1.7, 1.8, 1.9, // v[1]
            ],
            w_gate: vec![0.2, -0.4, 0.6, 0.8, -1.0, 1.2],
            w_beta: vec![0.3, 0.5, -0.7],
            w_alpha: vec![-0.4, 0.6, 0.8],
            conv1d: vec![
                0.05, -0.15, 0.25, -0.35, // ch0 (q0)
                0.45, -0.55, 0.65, -0.75, // ch1 (q1)
                0.85, -0.95, 1.05, -1.15, // ch2 (k0)
                1.25, -1.35, 1.45, -1.55, // ch3 (k1)
                1.65, -1.75, 1.85, -1.95, // ch4 (v0)
                2.05, -2.15, 2.25, -2.35, // ch5 (v1)
            ],
            ssm_a: vec![-1.5],
            ssm_dt: vec![0.7],
            ssm_norm: vec![1.1, 0.9],
            w_out: vec![
                0.2, -0.4, // y0 -> out0
                0.6, 0.8, // y1 -> out1
                -1.0, 1.2, // y2 -> out2
            ],
        };
        (p, w)
    }

    /// 独立実装 (テスト内オラクル): 数式を直接書き下したもの
    fn oracle_step(p: &GdnParams, w: &GdnWeights, x: &[f32], conv_hist: &mut Vec<Vec<f32>>, s: &mut [[f32; 2]; 2]) -> Vec<f32> {
        // conv 履歴: conv_hist[c] は時間順の過去入力 (最大 d_conv-1 保持)
        let qkv: Vec<f32> = (0..p.qkv_dim())
            .map(|i| (0..p.n_embd).map(|j| w.w_qkv[i * p.n_embd + j] * x[j]).sum())
            .collect();
        for c in 0..p.conv_dim() {
            conv_hist[c].push(qkv[c]);
            if conv_hist[c].len() > p.d_conv {
                conv_hist[c].remove(0);
            }
        }
        let mut conv_out = vec![0.0f32; p.conv_dim()];
        for c in 0..p.conv_dim() {
            let n = conv_hist[c].len();
            let mut acc = 0.0;
            // 畳み込み: 直近 n 入力とカーネル末尾 n tap の内積 (パディング 0)
            for (i, &v) in conv_hist[c].iter().enumerate() {
                acc += w.conv1d[c * p.d_conv + (p.d_conv - n) + i] * v;
            }
            conv_out[c] = acc / (1.0 + (-acc).exp());
        }
        // l2 norm + q scale
        let q_raw = &conv_out[0..2];
        let k_raw = &conv_out[2..4];
        let v = &conv_out[4..6];
        let mut q = q_raw.to_vec();
        let mut k = k_raw.to_vec();
        {
            let s2: f32 = q.iter().map(|x| x * x).sum();
            let inv = 1.0 / (s2 + p.eps).sqrt();
            for v in q.iter_mut() {
                *v *= inv;
            }
        }
        for v in q.iter_mut() {
            *v *= 1.0 / (2.0f32).sqrt();
        }
        {
            let s2: f32 = k.iter().map(|x| x * x).sum();
            let inv = 1.0 / (s2 + p.eps).sqrt();
            for v in k.iter_mut() {
                *v *= inv;
            }
        }
        // beta / alpha / gate
        let beta = 1.0 / (1.0 + (-(0..p.n_embd).map(|j| w.w_beta[j] * x[j]).sum::<f32>()).exp());
        let alpha: f32 = (0..p.n_embd).map(|j| w.w_alpha[j] * x[j]).sum();
        let g = w.ssm_a[0] * (if alpha + w.ssm_dt[0] > 20.0 { alpha + w.ssm_dt[0] } else { (1.0 + (alpha + w.ssm_dt[0]).exp()).ln() });
        let eg = g.exp();
        // S ← S * eg
        for r in 0..2 {
            for c in 0..2 {
                s[r][c] *= eg;
            }
        }
        // sk[c] = Σ_r k[r] S[r][c]
        let mut sk = [0.0f32; 2];
        for r in 0..2 {
            for c in 0..2 {
                sk[c] += k[r] * s[r][c];
            }
        }
        // d = beta * (v - sk)
        let d = [beta * (v[0] - sk[0]), beta * (v[1] - sk[1])];
        // S += k ⊗ d
        for r in 0..2 {
            for c in 0..2 {
                s[r][c] += k[r] * d[c];
            }
        }
        // o[c] = Σ_r q[r] S[r][c]
        let mut o = [0.0f32; 2];
        for r in 0..2 {
            for c in 0..2 {
                o[c] += q[r] * s[r][c];
            }
        }
        // gated norm
        let z: Vec<f32> = (0..2).map(|i| (0..p.n_embd).map(|j| w.w_gate[i * p.n_embd + j] * x[j]).sum()).collect();
        let mut y = [0.0f32; 2];
        for c in 0..2 {
            let mean2 = (o[0] * o[0] + o[1] * o[1]) / 2.0;
            let inv = 1.0 / (mean2 + p.eps).sqrt();
            y[c] = o[c] * inv * w.ssm_norm[c] * (z[c] / (1.0 + (-z[c]).exp()));
        }
        (0..p.n_embd)
            .map(|i| (0..2).map(|c| w.w_out[i * 2 + c] * y[c]).sum())
            .collect()
    }

    #[test]
    fn gdn_matches_independent_oracle() {
        let (p, w) = tiny();
        // 3 トークン入力
        let xs = vec![
            vec![0.5, -0.25, 0.125],
            vec![-1.0, 0.75, 0.5],
            vec![0.3, 0.3, -0.9],
        ];
        let mut state = GdnState::new(&p);
        let got = p.forward(&w, &xs, &mut state);

        // オラクル: 独立に状態を再構成
        let mut conv_hist: Vec<Vec<f32>> = vec![Vec::new(); p.conv_dim()];
        let mut s = [[0.0f32; 2]; 2];
        for (t, x) in xs.iter().enumerate() {
            let expect = oracle_step(&p, &w, x, &mut conv_hist, &mut s);
            for (i, (&g, &e)) in got[t].iter().zip(expect.iter()).enumerate() {
                assert!(
                    (g - e).abs() < 1e-5,
                    "t={t} i={i}: got={g:.7} want={e:.7}"
                );
            }
        }
    }

    #[test]
    fn gdn_prefill_equals_token_by_token_bitexact() {
        let (p, w) = tiny();
        let xs = vec![
            vec![0.5, -0.25, 0.125],
            vec![-1.0, 0.75, 0.5],
            vec![0.3, 0.3, -0.9],
            vec![2.0, -1.0, 0.0],
        ];
        // 1 回の prefill
        let mut st1 = GdnState::new(&p);
        let whole = p.forward(&w, &xs, &mut st1);
        // トークンごとの逐次呼び出し
        let mut st2 = GdnState::new(&p);
        for (t, x) in xs.iter().enumerate() {
            let y = p.forward_token(&w, x, &mut st2);
            assert_eq!(bits(&y), bits(&whole[t]), "t={t}: prefill と逐次が不一致");
        }
        assert_eq!(bits(&st1.conv), bits(&st2.conv));
        assert_eq!(bits(&st1.ssm), bits(&st2.ssm));
    }

    #[test]
    fn gdn_deterministic_two_runs() {
        let (p, w) = tiny();
        let xs: Vec<Vec<f32>> = (0..8)
            .map(|i| vec![0.1 * i as f32, -0.2 * i as f32, 0.3 * i as f32])
            .collect();
        let mut s1 = GdnState::new(&p);
        let a = p.forward(&w, &xs, &mut s1);
        let mut s2 = GdnState::new(&p);
        let b = p.forward(&w, &xs, &mut s2);
        for (va, vb) in a.iter().zip(b.iter()) {
            assert_eq!(bits(va), bits(vb));
        }
        assert_eq!(bits(&s1.ssm), bits(&s2.ssm));
    }

    #[test]
    fn gdn_delta_rule_properties() {
        // 性質テスト: 初期状態 0 + v 固定 + q 極小 → 出力は過去 v の指数減衰重み付き和に漸近
        let (p, w) = tiny();
        // k を 0 に近づけると delta 項が消え、純粋な累積 (decay のみ) になる
        let mut w2 = w.clone();
        for v in w2.w_qkv[2..4].iter_mut() {
            *v = 0.0; // k 投影を 0 に → k ベクトル 0
        }
        let xs: Vec<Vec<f32>> = (0..6).map(|_| vec![0.7, 0.7, 0.7]).collect();
        let mut st = GdnState::new(&p);
        let out = p.forward(&w2, &xs, &mut st);
        assert!(out.iter().all(|o| o.iter().all(|v| v.is_finite())));
        // 出力が発散しない (decay あり)
        assert!(out.iter().all(|o| o.iter().all(|v| v.abs() < 1e3)));
    }

    // ==================== A1-r2: chunked scan テスト ========================

    /// LCG 疑似乱数 (決定論)。cuda 側テストと同じ定数。
    fn lcg_rng(seed: u64) -> impl FnMut() -> f32 {
        let mut rng = seed;
        move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 33) as f32) / (u32::MAX as f32) - 0.5
        }
    }

    /// ランダム重みの GDN (chunked テスト用)。ssm_a は必ず負 (減衰契約)。
    fn rand_params(
        nk: usize,
        nv: usize,
        dk: usize,
        dv: usize,
        n_embd: usize,
        d_conv: usize,
    ) -> (GdnParams, GdnWeights) {
        let mut rnd = lcg_rng(0x1234_5678_9abc_def0);
        let p = GdnParams {
            n_embd,
            n_k_heads: nk,
            n_v_heads: nv,
            head_k_dim: dk,
            head_v_dim: dv,
            d_conv,
            eps: 1e-6,
        };
        let w = GdnWeights {
            w_qkv: (0..p.qkv_dim() * n_embd).map(|_| rnd()).collect(),
            w_gate: (0..p.value_dim() * n_embd).map(|_| rnd()).collect(),
            w_beta: (0..nv * n_embd).map(|_| rnd()).collect(),
            w_alpha: (0..nv * n_embd).map(|_| rnd()).collect(),
            conv1d: (0..p.conv_dim() * d_conv).map(|_| rnd()).collect(),
            ssm_a: (0..nv).map(|_| -1.0 - rnd().abs()).collect(),
            ssm_dt: (0..nv).map(|_| rnd()).collect(),
            ssm_norm: (0..dv).map(|_| 0.5 + rnd().abs()).collect(),
            w_out: (0..n_embd * p.value_dim()).map(|_| rnd()).collect(),
        };
        (p, w)
    }

    /// forward_token の投影部分 (chunked テスト用)。列優先のフラット行で返す。
    fn project(
        p: &GdnParams,
        w: &GdnWeights,
        xs: &[Vec<f32>],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let cd = p.qkv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;
        let mut qkv = Vec::with_capacity(xs.len() * cd);
        let mut z = Vec::with_capacity(xs.len() * vd);
        let mut beta = Vec::with_capacity(xs.len() * nv);
        let mut alpha = Vec::with_capacity(xs.len() * nv);
        for x in xs {
            qkv.extend(matvec(&w.w_qkv, x, cd, p.n_embd));
            z.extend(matvec(&w.w_gate, x, vd, p.n_embd));
            beta.extend(matvec(&w.w_beta, x, nv, p.n_embd));
            alpha.extend(matvec(&w.w_alpha, x, nv, p.n_embd));
        }
        (qkv, z, beta, alpha)
    }

    fn maxabs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        dot / (na * nb)
    }

    fn core<'a>(p: &GdnParams, w: &'a GdnWeights) -> GdnCore<'a> {
        GdnCore {
            conv1d: &w.conv1d,
            ssm_a: &w.ssm_a,
            ssm_dt: &w.ssm_dt,
            ssm_norm: &w.ssm_norm,
        }
    }

    /// 投影済み行からの逐次 AR 参照 (scan_chunked_core と同じ入力セマンティクス・
    /// 戻り値は out_w 投影前の y [n × value_dim])。
    fn ar_rows(
        p: &GdnParams,
        cw: &GdnCore<'_>,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
        st: &mut GdnState,
    ) -> Vec<f32> {
        let cd = p.qkv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;
        let n = z.len() / vd;
        let mut y = vec![0.0f32; n * vd];
        for t in 0..n {
            let yt = forward_token_core(
                p,
                cw,
                &qkv[t * cd..(t + 1) * cd],
                &z[t * vd..(t + 1) * vd],
                &beta[t * nv..(t + 1) * nv],
                &alpha[t * nv..(t + 1) * nv],
                st,
            );
            y[t * vd..(t + 1) * vd].copy_from_slice(&yt);
        }
        y
    }

    #[test]
    fn gdn_chunk_plan_boundaries() {
        // n=0 / 閾値以下 / 端数・上限クランプ
        assert_eq!(chunk_plan(0, 2048, 1024, 16), (1, 0));
        assert_eq!(chunk_plan(1, 2048, 1024, 16), (1, 1));
        assert_eq!(chunk_plan(2048, 2048, 1024, 16), (1, 2048));
        assert_eq!(chunk_plan(2049, 2048, 1024, 16), (3, 683));
        assert_eq!(chunk_plan(4096, 2048, 1024, 16), (4, 1024));
        assert_eq!(chunk_plan(100_000, 2048, 1024, 16), (16, 6250));
        // 全 chunk 非空 (最終 chunk に端数)
        let (nc, c) = chunk_plan(10, 0, 4, 8);
        assert_eq!((nc, c), (3, 4)); // [0,4) [4,8) [8,10)
    }

    /// 単一 chunk (n_chunks == 1) は逐次 AR と bit 一致
    /// (pass 3 が forward_token_core そのもの・S_init[0] = S_in になるため)。
    #[test]
    fn gdn_chunked_single_chunk_bitexact() {
        let (p, w) = rand_params(2, 2, 8, 8, 16, 4);
        let n = 7usize; // d_conv - 1 = 3 未満も含む短列 (conv 初期状態経路をカバー)
        let xs: Vec<Vec<f32>> = (0..n).map(|i| vec![0.1 * i as f32; p.n_embd]).collect();
        let (qkv, z, beta, alpha) = project(&p, &w, &xs);
        let cw = core(&p, &w);
        // AR (投影済み行の逐次 — p.forward は out_w 投影後なので使わない)
        let mut st_ar = GdnState::new(&p);
        let ys_ar = ar_rows(&p, &cw, &qkv, &z, &beta, &alpha, &mut st_ar);
        // chunked (1 chunk)
        let mut st_ch = GdnState::new(&p);
        let ys_ch = scan_chunked_core(&p, &cw, &qkv, &z, &beta, &alpha, &mut st_ch, 1, n);
        let flat = ys_ar;
        assert_eq!(bits(&flat), bits(&ys_ch), "単一 chunk が逐次 AR と不一致");
        assert_eq!(bits(&st_ar.conv), bits(&st_ch.conv));
        assert_eq!(bits(&st_ar.ssm), bits(&st_ch.ssm));
    }

    /// 複数 chunk は数学的に逐次 AR と等価 (丸めのみ異なる)。
    /// **緩和契約 (GDN)**: bit 一致は要求しない。maxabs < 1e-3・cos > 0.999。
    /// (既存契約「逐次 == バッチ prefill bit 一致」は CPU 経路には適用されない —
    ///  chunked は GPU 専用で CPU forward 経路からは呼ばれない)
    #[test]
    fn gdn_chunked_matches_ar_relaxed() {
        let (p, w) = rand_params(2, 2, 8, 8, 16, 4);
        let n = 17usize;
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; p.n_embd];
                for (j, x) in v.iter_mut().enumerate() {
                    *x = ((i * 7 + j * 13) % 11) as f32 / 8.0 - 0.6;
                }
                v
            })
            .collect();
        let (qkv, z, beta, alpha) = project(&p, &w, &xs);
        let cw = core(&p, &w);
        // 非ゼロ初期状態 (conv 状態経路もカバー)
        let mut rnd = lcg_rng(0xfeed_beef);
        let st0 = GdnState {
            conv: (0..p.conv_dim() * (p.d_conv - 1)).map(|_| rnd()).collect(),
            ssm: (0..p.n_v_heads * p.head_k_dim * p.head_v_dim)
                .map(|_| rnd())
                .collect(),
        };
        // AR
        let mut st_ar = st0.clone();
        let ys_ar = ar_rows(&p, &cw, &qkv, &z, &beta, &alpha, &mut st_ar);
        // chunked (4 chunks × 5)
        let mut st_ch = st0.clone();
        let ys_ch = scan_chunked_core(&p, &cw, &qkv, &z, &beta, &alpha, &mut st_ch, 4, 5);
        let flat = ys_ar;
        let dy = maxabs_diff(&flat, &ys_ch);
        let ds = maxabs_diff(&st_ar.ssm, &st_ch.ssm);
        eprintln!("chunked vs AR: y maxabs={dy:.3e} ssm maxabs={ds:.3e} cos={:.6}", cosine(&flat, &ys_ch));
        assert!(dy < 1e-3, "y maxabs diff {dy}");
        assert!(ds < 1e-3, "ssm maxabs diff {ds}");
        assert!(cosine(&flat, &ys_ch) > 0.999);
        // conv 状態は整数値ロールで bit 一致のはず
        assert_eq!(bits(&st_ar.conv), bits(&st_ch.conv), "conv 状態が不一致");
    }

    /// chunked prefill の最終状態からの続き (decode 相当) も逐次 AR と近い。
    /// **緩和契約 (GDN)**: maxabs < 1e-3・cos > 0.999 (bit 一致不要)。
    #[test]
    fn gdn_chunked_state_continuity() {
        let (p, w) = rand_params(1, 2, 8, 8, 16, 4);
        let cw = core(&p, &w);
        let mk = |i: usize| vec![0.05 * i as f32 - 0.4; p.n_embd];
        let xs: Vec<Vec<f32>> = (0..12).map(mk).collect();
        let (qkv, z, beta, alpha) = project(&p, &w, &xs);
        // 真の AR: 12 トークン連続 (投影済み行の逐次)
        let mut st_true = GdnState::new(&p);
        let ys_true = ar_rows(&p, &cw, &qkv, &z, &beta, &alpha, &mut st_true);
        // chunked: 10 トークン chunked prefill (3 chunks: [0,4) [4,8) [8,10)・
        // 最終 chunk は端数切り詰め経路) + 残り 2 を逐次
        let mut st_ch = GdnState::new(&p);
        let cd = p.qkv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;
        let n_pre = 10usize;
        let y10 = scan_chunked_core(
            &p,
            &cw,
            &qkv[..n_pre * cd],
            &z[..n_pre * vd],
            &beta[..n_pre * nv],
            &alpha[..n_pre * nv],
            &mut st_ch,
            3,
            4,
        );
        let mut y_cont = vec![0.0f32; qkv.len() / cd * vd];
        y_cont[..n_pre * vd].copy_from_slice(&y10);
        for t in n_pre..12 {
            let y = forward_token_core(
                &p,
                &cw,
                &qkv[t * p.qkv_dim()..(t + 1) * p.qkv_dim()],
                &z[t * p.value_dim()..(t + 1) * p.value_dim()],
                &beta[t * p.n_v_heads..(t + 1) * p.n_v_heads],
                &alpha[t * p.n_v_heads..(t + 1) * p.n_v_heads],
                &mut st_ch,
            );
            y_cont[t * p.value_dim()..(t + 1) * p.value_dim()].copy_from_slice(&y);
        }
        let flat: Vec<f32> = ys_true;
        let dy = maxabs_diff(&flat, &y_cont);
        let ds = maxabs_diff(&st_true.ssm, &st_ch.ssm);
        assert!(dy < 1e-3, "継続 y maxabs diff {dy}");
        assert!(ds < 1e-3, "継続後 ssm maxabs diff {ds}");
        assert!(cosine(&flat, &y_cont) > 0.999);
    }

    /// chunk 分割の変え方で結果がほぼ不変 (丸め差のみ)。**緩和契約 (GDN)**:
    /// bit 一致不要・maxabs < 1e-4。
    #[test]
    fn gdn_chunked_split_invariance() {
        let (p, w) = rand_params(2, 3, 8, 8, 16, 4);
        let n = 23usize;
        let xs: Vec<Vec<f32>> = (0..n).map(|i| vec![(i % 5) as f32 / 4.0 - 0.5; p.n_embd]).collect();
        let (qkv, z, beta, alpha) = project(&p, &w, &xs);
        let cw = core(&p, &w);
        let mut base = None;
        for (nc, cl) in [(1usize, 23usize), (2, 12), (4, 6), (6, 4), (23, 1)] {
            let mut st = GdnState::new(&p);
            let y = scan_chunked_core(&p, &cw, &qkv, &z, &beta, &alpha, &mut st, nc, cl);
            match &base {
                None => base = Some((y, st.ssm.clone())),
                Some((y0, s0)) => {
                    let dy = maxabs_diff(y0, &y);
                    assert!(dy < 1e-4, "nc={nc}: y maxabs diff {dy}");
                    assert!(maxabs_diff(s0, &st.ssm) < 1e-4, "nc={nc}: ssm diff");
                    assert_eq!(bits(&y), bits(&y), "決定論 (同一入力)");
                }
            }
        }
    }

    /// chunked scan は決定論的 (同一入力 → bit 一致) — GDN の必須契約。
    #[test]
    fn gdn_chunked_deterministic_two_runs() {
        let (p, w) = rand_params(2, 2, 8, 8, 16, 4);
        let n = 37usize;
        let xs: Vec<Vec<f32>> = (0..n).map(|i| vec![0.03 * i as f32 - 0.5; p.n_embd]).collect();
        let (qkv, z, beta, alpha) = project(&p, &w, &xs);
        let cw = core(&p, &w);
        let mut s1 = GdnState::new(&p);
        let a = scan_chunked_core(&p, &cw, &qkv, &z, &beta, &alpha, &mut s1, 5, 8);
        let mut s2 = GdnState::new(&p);
        let b = scan_chunked_core(&p, &cw, &qkv, &z, &beta, &alpha, &mut s2, 5, 8);
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(bits(&s1.ssm), bits(&s2.ssm));
        assert_eq!(bits(&s1.conv), bits(&s2.conv));
    }
}
