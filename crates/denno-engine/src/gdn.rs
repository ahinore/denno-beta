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
}
