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
//! 出力は head ごとに gated RMS norm: rmsnorm(o)*w * g(z)。
//!
//! g はモデル系で違う。Qwen3.5 (`qwen35`) は `silu`、Qwen3.8-Flash-Next
//! (`qwen4exp`) は `sigmoid` (参照実装 `qwen4exp.cpp` の `build_norm_gated` に
//! "the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu"
//! と明記がある)。`GateKind` で選ぶ。既定 (`forward_token_core`) は `Silu`。

/// s90 (P1-1): 2 launch スケジュールの帳簿 (ホスト側・GPU 不要)
pub mod scan_schedule;

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

    /// P-MOE-F2: Qwen3.8-Flash-Next (qwen4exp) の ssm 層。
    ///
    /// GGUF メタデータで確定済みの値 (flashnext_meta.txt):
    /// `ssm.state_size=128` → head_k_dim = head_v_dim = 128、
    /// `ssm.group_count=16` → n_k_heads、`ssm.time_step_rank=48` → n_v_heads、
    /// `ssm.inner_size=6144` = 48 * 128、`ssm.conv_kernel=4`、
    /// `attention.layer_norm_rms_epsilon=1e-6`。n_embd は 2560
    /// (`attn_qkv.weight` の ne1)。
    pub fn qwen4exp() -> Self {
        GdnParams {
            n_embd: 2560,
            n_k_heads: 16,
            n_v_heads: 48,
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
/// gated RMS norm の出力ゲートの種類。
///
/// - `Silu`: Qwen3.5 (`qwen35`) の GDN。
/// - `Sigmoid`: Qwen3.8-Flash-Next (`qwen4exp`) の GDN。**ここだけが 3.5 と違う**。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateKind {
    Silu,
    Sigmoid,
}

pub fn forward_token_core(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
) -> Vec<f32> {
    forward_token_core_gate(p, cw, qkv, z, beta_raw, alpha, st, GateKind::Silu)
}

/// [`forward_token_core`] の出力ゲート種別つき版。
/// デルタルール 1 ヘッドぶん。`s_head` は `[dk][dv]` 行優先のヘッド専用スライス。
///
/// ヘッド内は逐次加算のまま (llama.cpp の fused カーネルとの比較対象)。
#[allow(clippy::too_many_arguments)]
fn gdn_head(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    q: &[f32],
    k: &[f32],
    v_raw: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    s_head: &mut [f32],
    o_head: &mut [f32],
    h: usize,
) {
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;

    // k/q は n_k_heads と n_v_heads が異なる場合はタイル repeat 規約
    // (GGUF / llama.cpp ggml_repeat / fused カーネル `iv1 % neq1` = strided:
    //  v-head h → k/q head h % nk。0.8B は nk==nv で同一・9B は nk=16/nv=32・
    //  27B は nk=16/nv=48)。HF transformers の repeat_interleave (blocked) と
    //  は異なる規約であり、第15/16セッションの lchk サブステージ比較で 9B/27B
    //  とも llama.cpp の o が strided のみ cos≈1 で一致することを確認済み。
    let kh = h % p.n_k_heads;
    let qh = &q[kh * dk..kh * dk + dk];
    let kh_v = &k[kh * dk..kh * dk + dk];
    let vh = &v_raw[h * dv..h * dv + dv];

    // decay: g = ssm_a * softplus(alpha + dt)
    let g = cw.ssm_a[h] * softplus(alpha[h] + cw.ssm_dt[h]);
    let eg = g.exp();

    // beta
    let b = sigmoid(beta_raw[h]);

    // 本体 (デルタ則)。作業域はヘッド 1 本ぶん (チャンク版と同じ関数を使う)
    let mut sk = vec![0.0f32; dv];
    let mut d = vec![0.0f32; dv];
    gdn_head_step(dk, dv, qh, kh_v, vh, eg, b, s_head, o_head, &mut sk, &mut d);
}

// ---------------------------------------------------------------------------
// 区間計測 (`DENNO_GDN_KPROF=1`・既定 OFF)
//
// decode の GDN コアは 1 トークン = 1 層 1 回なので、4 区間
// (conv / l2 / scan / rms) の ns を積むだけで「層あたりどこが重いか」が出る。
// 計測専用で数値経路からは読まない。`on()` が偽なら `Instant` も取らない
// (サイトあたり分岐 1 回)。
// ---------------------------------------------------------------------------
pub mod kprof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    /// 区間名 (添字が区間 ID)。
    pub const NAMES: [&str; 4] = ["conv", "l2", "scan", "rms"];
    /// 区間数。
    pub const N: usize = 4;

    static NS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
    static CALLS: AtomicU64 = AtomicU64::new(0);

    /// 計測するか (`DENNO_GDN_KPROF=1` のときだけ真)。既定 off。
    #[inline]
    pub fn on() -> bool {
        #[cfg(test)]
        if let Some(v) = forced() {
            return v;
        }
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| std::env::var("DENNO_GDN_KPROF").ok().as_deref() == Some("1"))
    }

    // env はプロセスで 1 回しか読めず、テストは並列なので上書き口を置く
    // (`cpu_gemv_par_from_value` と同じ流儀)。0 = env・1 = 強制 off・2 = 強制 on。
    #[cfg(test)]
    static FORCE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

    #[cfg(test)]
    fn forced() -> Option<bool> {
        match FORCE.load(Ordering::Relaxed) {
            1 => Some(false),
            2 => Some(true),
            _ => None,
        }
    }

    /// テストから env を介さずに切り替える (`None` で env の値に戻す)。
    #[cfg(test)]
    pub fn set_on_for_test(v: Option<bool>) {
        FORCE.store(
            match v {
                None => 0,
                Some(false) => 1,
                Some(true) => 2,
            },
            Ordering::Relaxed,
        );
    }

    /// 区間 `i` を閉じて次の時刻を返す。`t` が `None` (計測 off) なら no-op。
    #[inline]
    pub fn lap(t: Option<Instant>, i: usize) -> Option<Instant> {
        match t {
            Some(t0) => {
                NS[i].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                Some(Instant::now())
            }
            None => None,
        }
    }

    /// 1 回ぶんの呼び出しを数える。
    #[inline]
    pub fn bump_call() {
        CALLS.fetch_add(1, Ordering::Relaxed);
    }

    /// 積算を 0 に戻す (表示後に呼ぶ)。
    pub fn reset() {
        for c in NS.iter() {
            c.store(0, Ordering::Relaxed);
        }
        CALLS.store(0, Ordering::Relaxed);
    }

    /// (call 数, 区間ごとの ns) を読む。
    pub fn take() -> (u64, [u64; N]) {
        let mut ns = [0u64; N];
        for (d, s) in ns.iter_mut().zip(NS.iter()) {
            *d = s.load(Ordering::Relaxed);
        }
        (CALLS.load(Ordering::Relaxed), ns)
    }

    /// 表示行 (未計測・call 0 なら `None`)。
    pub fn line() -> Option<String> {
        if !on() {
            return None;
        }
        let (calls, ns) = take();
        if calls == 0 {
            return None;
        }
        let total: u64 = ns.iter().sum();
        let ms = |v: u64| v as f64 / 1e6;
        let pct = |v: u64| v as f64 / total.max(1) as f64 * 100.0;
        Some(format!(
            "[GDN_CORE_KPROF] conv={:.3} l2={:.3} scan={:.3} rms={:.3} total={:.3} calls={} \
             per-call={:.3} ms (conv {:.1}% l2 {:.1}% scan {:.1}% rms {:.1}%)",
            ms(ns[0]),
            ms(ns[1]),
            ms(ns[2]),
            ms(ns[3]),
            ms(total),
            calls,
            ms(total) / calls as f64,
            pct(ns[0]),
            pct(ns[1]),
            pct(ns[2]),
            pct(ns[3])
        ))
    }

    /// 1 行を stderr へ出す (未計測なら何もしない)。
    pub fn report() {
        if let Some(l) = line() {
            eprintln!("{l}");
        }
    }
}

pub fn forward_token_core_gate(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
    gate: GateKind,
) -> Vec<f32> {
    // 既定は並列版 (`forward_token_core_gate_par`)。次の 2 つだけ旧実装に落ちる:
    // - サブステージダンプ (`DENNO_DEBUG_GDN_DUMP_LAYER`) がこの層を指している
    //   → 追記順を llama.cpp の lchk と同一に保つため、トークン逐次のまま回す。
    // - `DENNO_GDN_CORE_SCALAR=1` → 実機で新旧のビット一致を A/B するための逃げ道。
    let dbg_layer = DBG_CUR_LAYER.with(|c| c.get());
    let dbg = matches!(gdump_cfg(), Some((_, l)) if l == dbg_layer && dbg_layer >= 0);
    if dbg || core_scalar() {
        return forward_token_core_gate_ref(p, cw, qkv, z, beta_raw, alpha, st, gate);
    }
    forward_token_core_gate_par(p, cw, qkv, z, beta_raw, alpha, st, gate)
}

/// `DENNO_GDN_CORE_SCALAR=1` で decode の GDN コアを旧実装 (トークン逐次) にする。
///
/// decode の GDN は 1 トークン 1 層 1 回なので、並列版と旧実装の出力は
/// ビット一致する (要素ごとの演算順序を変えない)。実機で A/B するための
/// 逃げ道として残す。既定 off。
fn core_scalar_from_value(v: Option<&str>) -> bool {
    v.map(|s| s.trim() == "1").unwrap_or(false)
}

/// [`core_scalar_from_value`] の実効値 (env はプロセスで 1 回)。
fn core_scalar() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| core_scalar_from_value(std::env::var("DENNO_GDN_CORE_SCALAR").ok().as_deref()))
}

/// 旧実装 (トークン逐次・スカラー)。値の基準であり、診断ダンプ経路でもある。
///
/// 並列版 [`forward_token_core_gate_par`] はこの関数と **f32 でビット一致**する。
pub fn forward_token_core_gate_ref(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
    gate: GateKind,
) -> Vec<f32> {
    // 恒久診断: GDN サブステージ per-token ダンプ (27B 初発散層の stage 内切り
    // 分け用)。DENNO_DEBUG_GDN_DUMP=<file> + DENNO_DEBUG_GDN_DUMP_LAYER=<li> で、
    // 指定層の全トークンのサブステージを llama.cpp tools/lchk の GDN detail モード
    // と同一順序 (qkv,z,beta,gate,conv,q,k,v,o,y) で raw f32 追記する。
    // env ゲート済み・未指定時は no-op (恒久保持)。
    let dbg_layer = DBG_CUR_LAYER.with(|c| c.get());
    let dbg = matches!(gdump_cfg(), Some((_, l)) if l == dbg_layer && dbg_layer >= 0);
    if dbg {
        gdump("qkv", qkv);
        gdump("z", z);
    }
    let (beta_sig, gate_dbg) = if dbg {
        let bs: Vec<f32> = beta_raw.iter().map(|&b| sigmoid(b)).collect();
        let g: Vec<f32> = (0..p.n_v_heads)
            .map(|h| cw.ssm_a[h] * softplus(alpha[h] + cw.ssm_dt[h]))
            .collect();
        (bs, g)
    } else {
        (Vec::new(), Vec::new())
    };

    // ---- causal depthwise conv + silu --------------------------------
    // 入力列 = [conv_state (d_conv-1), qkv_t] をチャネルごとに畳み込み
    let mut kt = if kprof::on() {
        Some(std::time::Instant::now())
    } else {
        None
    };
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
    if dbg {
        gdump("conv", &conv_out);
        gdump("beta", &beta_sig);
        gdump("gate", &gate_dbg);
    }
    kt = kprof::lap(kt, 0);

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
    if dbg {
        gdump("q", &q); // q_scale 適用済み (llama.cpp は scan 内で scale)
        gdump("k", &k);
        gdump("v", v_raw);
    }
    kt = kprof::lap(kt, 1);

    // ---- デルタルール (head ごと・AR 形式) ------------------------------
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;
    let mut o = vec![0.0f32; vd];
    // ヘッドは互いに独立 (状態 S[h] と出力 o[h] がヘッド専用のスライス) なので、
    // どのヘッドをどのスレッドが担当しても各要素の計算順序は変わらない。
    // 27B の「GDN は CPU/GPU でビット一致」契約はこの性質に依存している。
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        if p.n_v_heads >= 8 {
            st.ssm
                .par_chunks_mut(dk * dv)
                .zip(o.par_chunks_mut(dv))
                .enumerate()
                .for_each(|(h, (s_head, o_head))| {
                    gdn_head(
                        p,
                        cw,
                        &q,
                        &k,
                        v_raw,
                        beta_raw,
                        alpha,
                        s_head,
                        o_head,
                        h,
                    );
                });
        } else {
            for h in 0..p.n_v_heads {
                gdn_head(
                    p,
                    cw,
                    &q,
                    &k,
                    v_raw,
                    beta_raw,
                    alpha,
                    &mut st.ssm[h * dk * dv..(h + 1) * dk * dv],
                    &mut o[h * dv..(h + 1) * dv],
                    h,
                );
            }
        }
    }
    #[cfg(not(feature = "par"))]
    {
        for h in 0..p.n_v_heads {
            gdn_head(
                p,
                cw,
                &q,
                &k,
                v_raw,
                beta_raw,
                alpha,
                &mut st.ssm[h * dk * dv..(h + 1) * dk * dv],
                &mut o[h * dv..(h + 1) * dv],
                h,
            );
        }
    }

    // ---- gated RMS norm (head ごと): rmsnorm(o)*w * g(z) ----------------
    // g は `GateKind` (qwen35 = silu / qwen4exp = sigmoid)。
    kt = kprof::lap(kt, 2);
    let mut y = vec![0.0f32; vd];
    for h in 0..p.n_v_heads {
        let obase = h * dv;
        let mut seg: Vec<f32> = o[obase..obase + dv].to_vec();
        rms_norm(&mut seg, p.eps);
        for (c, v) in seg.iter_mut().enumerate() {
            let zc = z[obase + c];
            let g = match gate {
                GateKind::Silu => silu(zc),
                GateKind::Sigmoid => sigmoid(zc),
            };
            *v *= cw.ssm_norm[c] * g;
        }
        y[obase..obase + dv].copy_from_slice(&seg);
    }
    if dbg {
        gdump("o", &o); // scan 出力 (gated norm 前)
        gdump("y", &y); // gated norm 後
    }
    if kt.is_some() {
        kprof::lap(kt, 3);
        kprof::bump_call();
    }

    y
}

// ---------------------------------------------------------------------------
// P-PF-GDN: チャンク版 (トークン逐次のループをヘッド並列に置き換える)
// ---------------------------------------------------------------------------
//
// 現行のバッチ prefill は `for i in 0..nt { forward_token_core_gate(..) }` で、
// 1 トークンあたり 1 回の変換 + 1 回の rayon 分割 (ヘッド) を回す。状態
// S は 1 層 48 × 128 × 128 × 4 B = 3.1 MB あり、毎トークン全スレッドが触るので
// L3 帯域で止まる (実測: スキャン ≈215 µs/token = gdn 全体 279 µs/token の 8 割)。
//
// ここでは **ヘッド範囲をタスクに固定** し、そのタスクがチャンク全長の
// トークンループを回す。1 タスク 3 ヘッド = 状態 192 KB が L2 に載ったまま
// 走る。ヘッドどうしは独立 (S[h] / o[h] がヘッド専用) なので、要素ごとの
// 演算順序はトークン逐次と同一 = ビット一致。

thread_local! {
    /// conv 出力の作業域 (grow-only)。`vec![0.0; nt*conv_dim]` を毎回確保すると
    /// 初回タッチのページフォルトで 21 MB あたり数 ms かかる。
    static CONV_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    /// gated norm 出力の作業域 (ヘッド優先・grow-only)。
    static YH_SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// スレッドローカルの grow-only 作業域 (取得 / 返却)。
macro_rules! scratch_pair {
    ($get:ident, $put:ident, $tl:ident) => {
        fn $get(n: usize) -> Vec<f32> {
            $tl.with(|c| {
                let mut v = c.borrow_mut();
                if v.len() < n {
                    v.resize(n, 0.0);
                }
                std::mem::take(&mut *v)
            })
        }

        fn $put(v: Vec<f32>) {
            $tl.with(|c| {
                let mut s = c.borrow_mut();
                if s.capacity() < v.capacity() {
                    *s = v;
                }
            });
        }
    };
}

scratch_pair!(conv_scratch, conv_scratch_put, CONV_SCRATCH);
scratch_pair!(yh_scratch, yh_scratch_put, YH_SCRATCH);

// ---------------------------------------------------------------------------
// decode 1 トークンの GDN コア (並列版・既定)
// ---------------------------------------------------------------------------
//
// 旧実装 [`forward_token_core_gate_ref`] との違いは **並列化と作業域の再利用だけ**:
// - conv: チャネルごとの `acc = Σ_j w[j]*in[j]` (j 昇順) はそのまま。160 KB の
//   `conv_in` は作らず、状態 `st.conv` をその場で左ロールする (結果は同一)。
// - L2 正規化: k-head ごとに conv 出力をコピー → 2 乗和 (昇順) → `*= inv` →
//   q だけ `*= q_scale`。途中の丸めを変えない。
// - デルタ則: ヘッド独立なので [`gdn_head_step`] をそのまま呼び、担当ヘッドだけを
//   分ける。出力 o もヘッド専用スライス。
// - gated RMS norm: ヘッドごとに独立。
// よって 1 トークン 1 層の出力は f32 でビット一致する
// (テスト `par_core_matches_ref_bitexact`)。
//
// 旧実装の実測 (decode, 27B 形, 36 層, このコンテナ): conv 93 µs/層・l2 4・
// scan 95・rms 24。conv と rms は exp が 10240 + 6144 本ぶんスカラーで直列、
// これが全体の 5 割を占めていた。並列化で両方をスレッド数で割る。
thread_local! {
    /// 並列版 decode の conv 出力作業域 (grow-only, len = conv_dim)。
    static TOK_CV: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    /// 並列版 decode の q|k 作業域 (grow-only, len = 2*key_dim)。
    static TOK_QK: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    /// 並列版 decode の scan 出力作業域 (grow-only, len = value_dim)。
    static TOK_O: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

scratch_pair!(tok_cv, tok_cv_put, TOK_CV);
scratch_pair!(tok_qk, tok_qk_put, TOK_QK);
scratch_pair!(tok_o, tok_o_put, TOK_O);

/// 並列度 (`par` が無いビルドでは 1)。
#[inline]
fn par_threads() -> usize {
    #[cfg(feature = "par")]
    {
        rayon::current_num_threads().max(1)
    }
    #[cfg(not(feature = "par"))]
    {
        1
    }
}

/// 1 タスクが受け持つチャネル数 `bs` (= `head_k_dim` × 2^k)。
///
/// q/k のブロックが q と k をまたがないよう、`key_dim` を割り切る幅にする
/// (ブロック数がスレッド数の 4 倍程度になるまで倍にする)。
#[inline]
fn tok_block_channels(dk: usize, nk: usize, cd: usize, threads: usize) -> usize {
    if dk == 0 {
        return 1;
    }
    let mut hpb = 1usize;
    while nk % (hpb * 2) == 0 && cd / (dk * hpb) > 4 * threads.max(1) {
        hpb *= 2;
    }
    dk * hpb
}

/// decode 1 トークンの GDN コア (並列版)。[`forward_token_core_gate_ref`] とビット一致。
#[allow(clippy::too_many_arguments)]
pub fn forward_token_core_gate_par(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
    gate: GateKind,
) -> Vec<f32> {
    let cs = p.d_conv - 1;
    let dc = p.d_conv;
    let cd = p.conv_dim();
    let kd = p.key_dim();
    let vd = p.value_dim();
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;
    let nk = p.n_k_heads;
    let nv = p.n_v_heads;
    let eps = p.eps;
    // 退化形 (conv 幅 1・ヘッド無し・入力長不一致・状態長不一致) は旧実装に任せる。
    // 状態長が違うと zip が黙って短い方で回るので、ここで落とす (旧実装は panic する)。
    if cs == 0
        || dk == 0
        || dv == 0
        || nk == 0
        || nv == 0
        || qkv.len() != cd
        || z.len() != vd
        || st.conv.len() != cd * cs
        || st.ssm.len() != nv * dk * dv
    {
        return forward_token_core_gate_ref(p, cw, qkv, z, beta_raw, alpha, st, gate);
    }

    let q_scale = 1.0 / (dk as f32).sqrt();
    let threads = par_threads();
    let bs = tok_block_channels(dk, nk, cd, threads);
    let group = batch_head_group(nv);

    let mut cvv = tok_cv(cd);
    let mut qkv2 = tok_qk(2 * kd);
    let mut ovv = tok_o(vd);
    let cv = &mut cvv[..cd];
    let qk = &mut qkv2[..2 * kd];
    let o = &mut ovv[..vd];

    let mut kt = if kprof::on() {
        Some(std::time::Instant::now())
    } else {
        None
    };

    // ---- A: causal depthwise conv + silu + 状態ロール (チャネル並列) ----
    {
        let conv1d = cw.conv1d;
        let block = |bi: usize, cvb: &mut [f32], scb: &mut [f32]| {
            let c0 = bi * bs;
            for (i, out) in cvb.iter_mut().enumerate() {
                let c = c0 + i;
                let w = &conv1d[c * dc..c * dc + dc];
                let base = i * cs;
                let mut acc = 0.0f32;
                for (j, wj) in w[..cs].iter().enumerate() {
                    acc += *wj * scb[base + j];
                }
                acc += w[cs] * qkv[c];
                for j in 0..cs - 1 {
                    scb[base + j] = scb[base + j + 1];
                }
                scb[base + cs - 1] = qkv[c];
                *out = silu(acc);
            }
        };
        #[cfg(feature = "par")]
        {
            use rayon::prelude::*;
            cv.par_chunks_mut(bs)
                .zip(st.conv.par_chunks_mut(cs * bs))
                .enumerate()
                .for_each(|(bi, (cvb, scb))| block(bi, cvb, scb));
        }
        #[cfg(not(feature = "par"))]
        {
            for (bi, (cvb, scb)) in cv.chunks_mut(bs).zip(st.conv.chunks_mut(cs * bs)).enumerate() {
                block(bi, cvb, scb);
            }
        }
    }
    kt = kprof::lap(kt, 0);

    // ---- B: q/k の L2 正規化 (k-head ごと・conv 出力のコピーから) ----
    {
        let cv_qk: &[f32] = &cv[..2 * kd];
        let block = |bi: usize, dst: &mut [f32], src: &[f32]| {
            let is_q = bi * bs < kd;
            for (d, s) in dst.chunks_mut(dk).zip(src.chunks(dk)) {
                d.copy_from_slice(s);
                l2_norm(d, eps);
                if is_q {
                    for v in d.iter_mut() {
                        *v *= q_scale;
                    }
                }
            }
        };
        // `par` ビルドでもここは直列: 2*kd = 4096 要素 (16 KB) は 16 スレッドに
        // 割ると fork-join の方が高くつく (実測 9.3 µs/層 → 直列 3 µs/層)。
        for (bi, (dst, src)) in qk.chunks_mut(bs).zip(cv_qk.chunks(bs)).enumerate() {
            block(bi, dst, src);
        }
    }
    kt = kprof::lap(kt, 1);

    // ---- C: デルタ則 (ヘッド並列) ---------------------------------------
    {
        let qh_all: &[f32] = &qk[..kd];
        let kh_all: &[f32] = &qk[kd..2 * kd];
        let vraw: &[f32] = &cv[2 * kd..2 * kd + vd];
        let do_group = |gi: usize, sgrp: &mut [f32], ogrp: &mut [f32]| {
            let nh = sgrp.len() / (dk * dv);
            // 作業域はタスクの外で 1 回だけ (ヘッド×トークンごとの vec![] を消す)
            let mut sk = vec![0.0f32; dv];
            let mut dl = vec![0.0f32; dv];
            for hh in 0..nh {
                let h = gi * group + hh;
                let ob = &mut ogrp[hh * dv..(hh + 1) * dv];
                let kh = h % nk;
                let g = cw.ssm_a[h] * softplus(alpha[h] + cw.ssm_dt[h]);
                let eg = g.exp();
                let b = sigmoid(beta_raw[h]);
                let sh = &mut sgrp[hh * dk * dv..(hh + 1) * dk * dv];
                gdn_head_step_fused(
                    dk,
                    dv,
                    &qh_all[kh * dk..kh * dk + dk],
                    &kh_all[kh * dk..kh * dk + dk],
                    &vraw[h * dv..(h + 1) * dv],
                    eg,
                    b,
                    sh,
                    ob,
                    &mut sk,
                    &mut dl,
                );
            }
        };
        let par_ok = {
            #[cfg(feature = "par")]
            {
                nv >= 8
            }
            #[cfg(not(feature = "par"))]
            {
                false
            }
        };
        #[cfg(feature = "par")]
        if par_ok {
            use rayon::prelude::*;
            st.ssm
                .par_chunks_mut(group * dk * dv)
                .zip(o.par_chunks_mut(group * dv))
                .enumerate()
                .for_each(|(gi, (sgrp, ogrp))| do_group(gi, sgrp, ogrp));
        }
        if !par_ok {
            for (gi, (sgrp, ogrp)) in st
                .ssm
                .chunks_mut(group * dk * dv)
                .zip(o.chunks_mut(group * dv))
                .enumerate()
            {
                do_group(gi, sgrp, ogrp);
            }
        }
    }
    kt = kprof::lap(kt, 2);

    // ---- D: gated RMS norm (ヘッド並列): rmsnorm(o)*w * g(z) -------------
    let mut y = vec![0.0f32; vd];
    {
        let do_group = |gi: usize, ob: &[f32], yb: &mut [f32]| {
            let nh = ob.len() / dv;
            for hh in 0..nh {
                let h = gi * group + hh;
                let os = &ob[hh * dv..(hh + 1) * dv];
                let ys = &mut yb[hh * dv..(hh + 1) * dv];
                ys.copy_from_slice(os);
                rms_norm(ys, eps);
                let base = h * dv;
                for (c, v) in ys.iter_mut().enumerate() {
                    let zc = z[base + c];
                    let g = match gate {
                        GateKind::Silu => silu(zc),
                        GateKind::Sigmoid => sigmoid(zc),
                    };
                    *v *= cw.ssm_norm[c] * g;
                }
            }
        };
        let par_ok = {
            #[cfg(feature = "par")]
            {
                nv >= 8
            }
            #[cfg(not(feature = "par"))]
            {
                false
            }
        };
        #[cfg(feature = "par")]
        if par_ok {
            use rayon::prelude::*;
            o.par_chunks(group * dv)
                .zip(y.par_chunks_mut(group * dv))
                .enumerate()
                .for_each(|(gi, (ob, yb))| do_group(gi, ob, yb));
        }
        if !par_ok {
            for (gi, (ob, yb)) in o
                .chunks(group * dv)
                .zip(y.chunks_mut(group * dv))
                .enumerate()
            {
                do_group(gi, ob, yb);
            }
        }
    }
    if kt.is_some() {
        kprof::lap(kt, 3);
        kprof::bump_call();
    }

    tok_cv_put(cvv);
    tok_qk_put(qkv2);
    tok_o_put(ovv);
    y
}

/// 時刻 `tau` (チャンク先頭を 0 とし、負は conv 初期状態) の conv 入力。
#[inline]
fn conv_in_at(
    qkv: &[f32],
    conv_st: &[f32],
    cd: usize,
    cs: usize,
    tau: isize,
    ch: usize,
) -> f32 {
    if tau >= 0 {
        qkv[tau as usize * cd + ch]
    } else {
        conv_st[ch * cs + (tau + cs as isize) as usize]
    }
}

/// デルタ則 1 ヘッドぶんの中身 (減衰 → sk → d → 更新 → o)。
///
/// [`gdn_head`] から切り出しただけで演算順序は同一。作業域 `sk` / `d` を
/// 呼び出し側が持つので、ヘッド×トークンごとの `vec![]` が消える。
#[allow(clippy::too_many_arguments)]
fn gdn_head_step(
    dk: usize,
    dv: usize,
    qh: &[f32],
    kh_v: &[f32],
    vh: &[f32],
    eg: f32,
    b: f32,
    s_head: &mut [f32],
    o_head: &mut [f32],
    sk: &mut [f32],
    d: &mut [f32],
) {
    // すべて zip で回す: スライス添字だと要素ごとの境界チェックが残り、
    // LLVM がベクトル化を諦める (実測でスカラー 1 要素/cycle まで落ちる)。
    let s = &mut s_head[..dk * dv];

    // S ← S * exp(g)
    for v in s.iter_mut() {
        *v *= eg;
    }

    // sk[c] = Σ_r k[r] * S[r][c]
    for v in sk[..dv].iter_mut() {
        *v = 0.0;
    }
    for r in 0..dk {
        let kr = kh_v[r];
        if kr != 0.0 {
            let row = &s[r * dv..r * dv + dv];
            for (skc, rc) in sk.iter_mut().zip(row.iter()) {
                *skc += kr * rc;
            }
        }
    }

    // d[c] = beta * (v[c] - sk[c])
    for ((dc, &vc), &sc) in d.iter_mut().zip(vh.iter()).zip(sk.iter()) {
        *dc = b * (vc - sc);
    }

    // S[r][c] += k[r] * d[c]
    for r in 0..dk {
        let kr = kh_v[r];
        if kr != 0.0 {
            let row = &mut s[r * dv..r * dv + dv];
            for (rc, &dc) in row.iter_mut().zip(d.iter()) {
                *rc += kr * dc;
            }
        }
    }

    // o[c] = Σ_r q[r] * S[r][c]
    for r in 0..dk {
        let qr = qh[r];
        if qr != 0.0 {
            let row = &s[r * dv..r * dv + dv];
            for (oc, rc) in o_head.iter_mut().zip(row.iter()) {
                *oc += qr * rc;
            }
        }
    }
}

/// decode 用のデルタ則 1 ヘッドぶん (**走査 2 回**版)。
///
/// [`gdn_head_step`] と同じ値になる (テスト `par_core_matches_ref_bitexact`):
/// - pass 1+2 融合: ある要素の `S *= eg` と、その要素を使う `sk += k[r]*S` を
///   同じ走査で行う。行 `r` の昇順・列の昇順はそのまま (各要素の丸めは 1 回きり)。
/// - pass 3+4 融合: `S[r][c] += k[r]*d[c]` の直後に `o[c] += q[r]*S[r][c]` を足す。
///   列 `c` の総和は `r` 昇順のままで、各項は更新後の `S[r][c]`。
///
/// 4 走査 (decay / sk / 更新 / o) を 2 走査にすると状態 64 KB の読み書きが
/// 半減する (実測: decode 27B 形 36 層で scan 38.8 → 25 µs/層)。
#[allow(clippy::too_many_arguments)]
fn gdn_head_step_fused(
    dk: usize,
    dv: usize,
    qh: &[f32],
    kh_v: &[f32],
    vh: &[f32],
    eg: f32,
    b: f32,
    s_head: &mut [f32],
    o_head: &mut [f32],
    sk: &mut [f32],
    d: &mut [f32],
) {
    let (sk, d) = (&mut sk[..dv], &mut d[..dv]);
    let s = &mut s_head[..dk * dv];

    // ---- pass 1+2: S *= exp(g) と sk = Σ_r k[r]*S[r][c] ----
    for v in sk.iter_mut() {
        *v = 0.0;
    }
    for (r, row) in s.chunks_exact_mut(dv).enumerate() {
        let kr = kh_v[r];
        if kr != 0.0 {
            for (skc, rc) in sk.iter_mut().zip(row.iter_mut()) {
                let v = *rc * eg;
                *rc = v;
                *skc += kr * v;
            }
        } else {
            for rc in row.iter_mut() {
                *rc *= eg;
            }
        }
    }

    // d[c] = beta * (v[c] - sk[c])
    for ((dc, &vc), &sc) in d.iter_mut().zip(vh.iter()).zip(sk.iter()) {
        *dc = b * (vc - sc);
    }

    // ---- pass 3+4: S[r][c] += k[r]*d[c] と o[c] = Σ_r q[r]*S[r][c] ----
    for v in o_head[..dv].iter_mut() {
        *v = 0.0;
    }
    for (r, row) in s.chunks_exact_mut(dv).enumerate() {
        let kr = kh_v[r];
        let qr = qh[r];
        if kr != 0.0 {
            for (rc, &dc) in row.iter_mut().zip(d.iter()) {
                *rc += kr * dc;
            }
        }
        if qr != 0.0 {
            for (oc, rc) in o_head[..dv].iter_mut().zip(row.iter()) {
                *oc += qr * *rc;
            }
        }
    }
}

/// (a) conv1d + silu を全トークン一括で計算し `cv[t * conv_dim + c]` に書く。
///
/// 出力をトークン方向のブロックに切り、1 タスクがそのブロックを走査する
/// (書き込みは連続・作業域は「直前 3 トークン × ブロック幅」だけ)。
/// 1 要素の計算は `acc = 0; acc += w[j]*in[j] (j=0..d_conv)` の昇順そのままで、
/// [`forward_token_core_gate`] の `zip` 版と同一の丸めになる。
fn conv_silu_all(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    conv_st: &[f32],
    cv: &mut [f32],
    nt: usize,
) {
    let cs = p.d_conv - 1;
    let dc = p.d_conv;
    let cd = p.conv_dim();
    let w = cw.conv1d;
    if nt == 0 || cd == 0 {
        return;
    }
    let one_block = |t0: usize, out: &mut [f32]| {
        let ntb = out.len() / cd;
        // ブロック先頭の直前 3 行 (負の時刻は conv 初期状態) を 1 本に詰める
        let mut pre = vec![0.0f32; cs * cd];
        for i in 0..cs {
            let tau = t0 as isize - cs as isize + i as isize;
            let dst = &mut pre[i * cd..(i + 1) * cd];
            if tau >= 0 {
                dst.copy_from_slice(&qkv[tau as usize * cd..(tau as usize + 1) * cd]);
            } else {
                let si = (tau + cs as isize) as usize;
                for (ch, d) in dst.iter_mut().enumerate() {
                    *d = conv_st[ch * cs + si];
                }
            }
        }
        let mut r0 = &pre[0..cd];
        let mut r1 = &pre[cd..2 * cd];
        let mut r2 = &pre[2 * cd..3 * cd];
        for tt in 0..ntb {
            let t = t0 + tt;
            let r3 = &qkv[t * cd..(t + 1) * cd];
            let o = &mut out[tt * cd..(tt + 1) * cd];
            for c in 0..cd {
                let wr = &w[c * dc..c * dc + dc];
                let mut acc = 0.0f32;
                acc += wr[0] * r0[c];
                acc += wr[1] * r1[c];
                acc += wr[2] * r2[c];
                acc += wr[3] * r3[c];
                o[c] = silu(acc);
            }
            r0 = r1;
            r1 = r2;
            r2 = r3;
        }
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        let tb = (nt / rayon::current_num_threads().max(1)).max(1);
        cv.par_chunks_mut(tb * cd)
            .enumerate()
            .for_each(|(bi, out)| one_block(bi * tb, out));
    }
    #[cfg(not(feature = "par"))]
    {
        one_block(0, cv);
    }
}

/// (b) beta / alpha の変換 (sigmoid / softplus / exp) を全トークン一括。
/// トークン×ヘッドで独立なので、どの順に計算しても値は変わらない。
fn gate_table(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    beta_raw: &[f32],
    alpha: &[f32],
    nt: usize,
    egt: &mut [f32],
    bt: &mut [f32],
) {
    let nv = p.n_v_heads;
    if nv == 0 {
        return;
    }
    let one = |t: usize, e: &mut [f32], b: &mut [f32]| {
        for h in 0..nv {
            let g = cw.ssm_a[h] * softplus(alpha[t * nv + h] + cw.ssm_dt[h]);
            e[h] = g.exp();
            b[h] = sigmoid(beta_raw[t * nv + h]);
        }
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        egt.par_chunks_mut(nv)
            .zip(bt.par_chunks_mut(nv))
            .enumerate()
            .for_each(|(t, (e, b))| {
                if t < nt {
                    one(t, e, b);
                }
            });
    }
    #[cfg(not(feature = "par"))]
    {
        for t in 0..nt {
            one(t, &mut egt[t * nv..(t + 1) * nv], &mut bt[t * nv..(t + 1) * nv]);
        }
    }
}

/// タスクあたりのヘッド数。48 ヘッド / 16 スレッド = 3 (状態 192 KB が L2 に載る)。
fn batch_head_group(nv: usize) -> usize {
    #[cfg(feature = "par")]
    {
        let th = rayon::current_num_threads().max(1);
        nv.div_ceil(th).max(1)
    }
    #[cfg(not(feature = "par"))]
    {
        nv.max(1)
    }
}

/// (c)(d) スキャン + gated RMS norm。`h0` から `nh` ヘッドぶんを、
/// トークン順を保ったまま処理する (1 タスク = 1 グループ)。
///
/// 出力は **ヘッド優先** `yh[hh * nt * dv + t * dv + c]` に書く。タスクが
/// 連続した出力スライスを持てるようにするため (トークン優先の `y` は
/// ヘッド方向に切れない)。並べ替えは [`transpose_y`] で 1 回だけ行う。
#[allow(clippy::too_many_arguments)]
fn scan_gate_group(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    cv: &[f32],
    z: &[f32],
    egt: &[f32],
    bt: &[f32],
    nt: usize,
    h0: usize,
    sgrp: &mut [f32],
    yh: &mut [f32],
    gate: GateKind,
) {
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;
    let kd = p.key_dim();
    let cd = p.conv_dim();
    let vd = p.value_dim();
    let nv = p.n_v_heads;
    let nk = p.n_k_heads;
    let q_scale = 1.0 / (dk as f32).sqrt();
    let nh = sgrp.len() / (dk * dv);
    // ヘッド専用の作業域 (トークン×ヘッドごとの vec![] を消す)
    let mut qb = vec![0.0f32; dk];
    let mut kb = vec![0.0f32; dk];
    let mut sk = vec![0.0f32; dv];
    let mut dl = vec![0.0f32; dv];
    let mut o = vec![0.0f32; dv];
    for t in 0..nt {
        let row = &cv[t * cd..(t + 1) * cd];
        for hh in 0..nh {
            let h = h0 + hh;
            // q/k は k-head 単位に L2 正規化 + q のみスケール (現行と同一)
            let ko = (h % nk) * dk;
            qb.copy_from_slice(&row[ko..ko + dk]);
            kb.copy_from_slice(&row[kd + ko..kd + ko + dk]);
            l2_norm(&mut qb, p.eps);
            l2_norm(&mut kb, p.eps);
            for v in qb.iter_mut() {
                *v *= q_scale;
            }
            for v in o.iter_mut() {
                *v = 0.0;
            }
            let sh = &mut sgrp[hh * dk * dv..(hh + 1) * dk * dv];
            let vh = &row[2 * kd + h * dv..2 * kd + (h + 1) * dv];
            gdn_head_step(
                dk,
                dv,
                &qb,
                &kb,
                vh,
                egt[t * nv + h],
                bt[t * nv + h],
                sh,
                &mut o,
                &mut sk,
                &mut dl,
            );
            // gated RMS norm: rmsnorm(o) * w * g(z)
            rms_norm(&mut o, p.eps);
            let base = hh * nt * dv + t * dv;
            let yo = &mut yh[base..base + dv];
            let zo = t * vd + h * dv;
            for (c, v) in o.iter_mut().enumerate() {
                let zc = z[zo + c];
                let g = match gate {
                    GateKind::Silu => silu(zc),
                    GateKind::Sigmoid => sigmoid(zc),
                };
                yo[c] = *v * (cw.ssm_norm[c] * g);
            }
        }
    }
}

/// ヘッド優先 `yh[h * nt * dv + t * dv]` → トークン優先 `y[t * vd + h * dv]`。
fn transpose_y(yh: &[f32], y: &mut [f32], nt: usize, nv: usize, vd: usize, dv: usize) {
    let one = |t: usize, out: &mut [f32]| {
        for h in 0..nv {
            let s = h * nt * dv + t * dv;
            out[h * dv..(h + 1) * dv].copy_from_slice(&yh[s..s + dv]);
        }
    };
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        y.par_chunks_mut(vd).enumerate().for_each(|(t, out)| {
            if t < nt {
                one(t, out);
            }
        });
    }
    #[cfg(not(feature = "par"))]
    {
        for t in 0..nt {
            one(t, &mut y[t * vd..(t + 1) * vd]);
        }
    }
}

/// (c)(d) をヘッドグループ並列で回す。`group` はテスト・計測用に外から与える。
#[allow(clippy::too_many_arguments)]
fn scan_gate_all(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    cv: &[f32],
    z: &[f32],
    egt: &[f32],
    bt: &[f32],
    nt: usize,
    ssm: &mut [f32],
    yh: &mut [f32],
    y: &mut [f32],
    gate: GateKind,
    group: usize,
) {
    let dk = p.head_k_dim;
    let dv = p.head_v_dim;
    let vd = p.value_dim();
    let nv = p.n_v_heads;
    let group = group.max(1);
    #[cfg(feature = "par")]
    {
        use rayon::prelude::*;
        ssm.par_chunks_mut(group * dk * dv)
            .zip(yh.par_chunks_mut(group * nt * dv))
            .enumerate()
            .for_each(|(gi, (s, yy))| {
                scan_gate_group(p, cw, cv, z, egt, bt, nt, gi * group, s, yy, gate);
            });
    }
    #[cfg(not(feature = "par"))]
    {
        for (gi, (s, yy)) in ssm
            .chunks_mut(group * dk * dv)
            .zip(yh.chunks_mut(group * nt * dv))
            .enumerate()
        {
            scan_gate_group(p, cw, cv, z, egt, bt, nt, gi * group, s, yy, gate);
        }
    }
    transpose_y(yh, y, nt, nv, vd, dv);
}

/// チャンク版 GDN 本体。`qkv` / `z` / `beta_raw` / `alpha` は投影済みの
/// チャンク一括バッファ (`[nt][dim]`)。戻り値は `[nt][value_dim]`。
///
/// [`forward_token_core_gate`] を nt 回呼ぶのと **f32 でビット一致**する。
/// 根拠: (1) conv / L2 / beta・alpha 変換 / gated RMS norm は要素ごとに独立、
/// (2) デルタ則の状態と出力はヘッド専用で、ヘッドをまたぐ依存が無い。
/// どちらもトークン方向の順序を保てば丸め順が変わらない。
#[allow(clippy::too_many_arguments)]
pub fn forward_batch_core_gate(
    p: &GdnParams,
    cw: &GdnCore<'_>,
    qkv: &[f32],
    z: &[f32],
    beta_raw: &[f32],
    alpha: &[f32],
    st: &mut GdnState,
    gate: GateKind,
) -> Vec<f32> {
    let cd = p.conv_dim();
    let vd = p.value_dim();
    let nv = p.n_v_heads;
    let cs = p.d_conv - 1;
    let nt = if vd == 0 { 0 } else { z.len() / vd };
    if nt == 0 {
        return Vec::new();
    }
    debug_assert_eq!(qkv.len(), nt * cd);
    debug_assert_eq!(beta_raw.len(), nt * nv);
    debug_assert_eq!(alpha.len(), nt * nv);
    debug_assert_eq!(st.conv.len(), cd * cs);
    debug_assert_eq!(st.ssm.len(), nv * p.head_k_dim * p.head_v_dim);

    let conv_st = st.conv.clone();

    // ---- (a) conv1d + silu (全トークン一括) --------------------------------
    // 作業域は grow-only (前のチャンクの残りが入っている) ので、長さを明示する。
    let mut cv = conv_scratch(nt * cd);
    conv_silu_all(p, cw, qkv, &conv_st, &mut cv[..nt * cd], nt);

    // ---- conv 状態をチャンク末尾へ (入力の末尾 cs 本を写すだけ) ------------
    for i in 0..cs {
        let tau = nt as isize - cs as isize + i as isize;
        for c in 0..cd {
            st.conv[c * cs + i] = conv_in_at(qkv, &conv_st, cd, cs, tau, c);
        }
    }

    // ---- (b) beta / alpha の変換 -----------------------------------------
    let mut egt = vec![0.0f32; nt * nv];
    let mut bt = vec![0.0f32; nt * nv];
    gate_table(p, cw, beta_raw, alpha, nt, &mut egt, &mut bt);

    // ---- (c)(d) スキャン + gated RMS norm ---------------------------------
    let mut yh = yh_scratch(nt * vd);
    let mut y = vec![0.0f32; nt * vd];
    let group = batch_head_group(nv);
    scan_gate_all(
        p,
        cw,
        &cv[..nt * cd],
        z,
        &egt,
        &bt,
        nt,
        &mut st.ssm,
        &mut yh[..nt * vd],
        &mut y,
        gate,
        group,
    );

    conv_scratch_put(cv);
    yh_scratch_put(yh);
    y
}

// ---------------------------------------------------------------------------
// 恒久診断: GDN サブステージ per-token ダンプ (27B 初発散層の stage 内切り分け
// 用)。DENNO_DEBUG_GDN_DUMP=<file> + DENNO_DEBUG_GDN_DUMP_LAYER=<li> で、指定層
// の全トークンのサブステージを llama.cpp tools/lchk の GDN detail モードと同一
// 順序 (qkv,z,beta,gate,conv,q,k,v,o,y) で raw f32 追記する。
// 層インデックスは qwen35.rs gdn_scan から gdn_dbg_set_layer で通知する。
// env ゲート済み・未指定時は no-op (恒久保持)。
// ---------------------------------------------------------------------------
thread_local! {
    static DBG_CUR_LAYER: std::cell::Cell<i64> = const { std::cell::Cell::new(-1) };
}

/// 現在処理中の GDN 層インデックスを診断スレッドへ通知 (qwen35.rs の gdn_scan)。
pub fn gdn_dbg_set_layer(li: i64) {
    DBG_CUR_LAYER.with(|c| c.set(li));
}

/// 診断ダンプが指定層に対して有効か (qwen35.rs の出力投影後ダンプのゲート)。
pub fn gdn_dbg_enabled_for(li: i64) -> bool {
    matches!(gdump_cfg(), Some((_, l)) if l == li)
}

/// 層コンテキストなしで明示 stage 名をダンプ (qwen35.rs の出力投影後など)。
pub fn gdn_dbg_pub(stage: &str, v: &[f32]) {
    gdump(stage, v);
}

fn gdump_cfg() -> Option<(std::path::PathBuf, i64)> {
    static G: std::sync::OnceLock<Option<(std::path::PathBuf, i64)>> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let path = std::env::var("DENNO_DEBUG_GDN_DUMP").ok()?;
        let layer = std::env::var("DENNO_DEBUG_GDN_DUMP_LAYER")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0i64);
        Some((std::path::PathBuf::from(path), layer))
    })
    .clone()
}

fn gdump(stage: &str, v: &[f32]) {
    if let Some((path, _)) = gdump_cfg() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            let _ = f.write_all(&bytes);
        }
        let mut bitsum: u64 = 0;
        let mut maxabs = 0f32;
        let mut ssq = 0f64;
        for &x in v {
            bitsum = bitsum.wrapping_add(x.to_bits() as u64);
            maxabs = maxabs.max(x.abs());
            ssq += (x as f64) * (x as f64);
        }
        eprintln!(
            "[GDUMP] {stage} len={} bitsum={bitsum:016x} maxabs={maxabs:.6e} ssq={ssq:.10e}",
            v.len()
        );
    }
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
            // タイル repeat 規約 (forward_token_core と同一・llama.cpp ggml_repeat 相当)
            let kh = h % p.n_k_heads;
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

    /// GQA ヘッド対応の固定テスト: v-head h が参照する k/q head はタイル repeat 規約
    /// (h % nk・GGUF/llama.cpp ggml_repeat + fused カーネル `iv1 % neq1` 相当)。
    /// 旧実装 (blocked: h*nk/nv = HF repeat_interleave 相当) は 9B/27B とも llama.cpp
    /// の GDN scan 出力 o と乖離し 27B 出力破綻の原因だった (第15/16セッション確定)。
    /// 構成: dk=2/dv=2・nk=2/nv=4・2 トークン・decay なし (ssm_a=0)・β≈1。
    /// t0 と t1 で k head の向きを入れ替えることで、「どの k head をどの時刻で使ったか」
    /// が o (→ gated norm 後の y) の方向として観測できる。
    #[test]
    fn gdn_gqa_head_mapping_is_tiled() {
        let p = GdnParams {
            n_embd: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            head_k_dim: 2,
            head_v_dim: 2,
            d_conv: 4,
            eps: 1e-6,
        };
        let vd = p.value_dim(); // 8
        let cd = p.conv_dim(); // 16
        // conv1d: 最終 tap のみ 1.0 → conv_out[c] = silu(qkv[c]) (conv 状態は不問)
        let mut conv1d = vec![0.0f32; cd * p.d_conv];
        for c in 0..cd {
            conv1d[c * p.d_conv + p.d_conv - 1] = 1.0;
        }
        let cw = GdnCore {
            conv1d: &conv1d,
            ssm_a: &[0.0; 4], // decay なし (eg = 1)
            ssm_dt: &[0.0; 4],
            ssm_norm: &[1.0, 1.0],
        };
        // q チャネル (両トークン共通): kh0=[3,0] kh1=[0,3] → silu+L2 で [1,0]/[0,1] (×q_scale)
        let q_ch = [3.0f32, 0.0, 0.0, 3.0];
        // k チャネル: t0 は全 kh [1,0] 向き / t1 は全 kh [0,1] 向き
        // → (q_kh·k_t0kh, q_kh·k_t1kh) = (q_kh[0], q_kh[1]) となり o[t1] は q_kh に比例
        let k_ch = [
            [3.0f32, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 3.0],
        ];
        // v チャネル: t0 全 head [3,0] / t1 全 head [0,3] (d0=[s,0] d1=[0,s] 方向)
        let v_ch = [
            [3.0f32, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0, 3.0, 0.0, 3.0, 0.0, 3.0],
        ];
        let z = vec![1.0f32; vd];
        let beta_raw = vec![20.0f32; 4]; // sigmoid ≈ 1
        let alpha = vec![0.0f32; 4];

        let mut y = Vec::new();
        let mut st = GdnState::new(&p);
        for t in 0..2 {
            let mut qkv = Vec::with_capacity(cd);
            qkv.extend_from_slice(&q_ch);
            qkv.extend_from_slice(&k_ch[t]);
            qkv.extend_from_slice(&v_ch[t]);
            y.push(forward_token_core(&p, &cw, &qkv, &z, &beta_raw, &alpha, &mut st));
        }

        // 独立に期待値を計算 (decay なし・β≈1 → o[t] = Σ_τ (q_kh·k_τkh)·d_τ)
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let l2 = |mut x: Vec<f32>| {
            let s: f32 = x.iter().map(|t| t * t).sum();
            let inv = 1.0 / (s + p.eps).sqrt();
            for t in x.iter_mut() {
                *t *= inv;
            }
            x
        };
        let qscale = 1.0f32 / (p.head_k_dim as f32).sqrt();
        let qh: Vec<Vec<f32>> = (0..2)
            .map(|h| {
                l2(q_ch[h * 2..h * 2 + 2].to_vec())
                    .iter()
                    .map(|t| t * qscale)
                    .collect()
            })
            .collect();
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let beta = 1.0f32 / (1.0 + (-20.0f32).exp());
        let d: Vec<[f32; 2]> = (0..2)
            .map(|t| {
                let s = silu(3.0);
                if t == 0 {
                    [beta * s, 0.0]
                } else {
                    [0.0, beta * s]
                }
            })
            .collect();
        // k の L2 後の向き (両 kh 同一・トークンで異なる)
        let k_l2: Vec<[f32; 2]> = (0..2)
            .map(|t| {
                let v = l2(k_ch[t][0..2].to_vec());
                [v[0], v[1]]
            })
            .collect();
        // t=1 の o: (c0, c1) = (q_kh·k_t0kh, q_kh·k_t1kh) で d0/d1 を線形結合
        let o_of = |kh: usize| {
            let c0 = dot(&qh[kh], &k_l2[0]);
            let c1 = dot(&qh[kh], &k_l2[1]);
            [c0 * d[0][0], c1 * d[1][1]]
        };
        // gated rms norm (dv=2) 後の期待 y
        let y_of = |o: [f32; 2]| {
            let mean2 = (o[0] * o[0] + o[1] * o[1]) / 2.0;
            let inv = 1.0 / (mean2 + p.eps).sqrt();
            [(o[0] * inv) * silu(1.0), (o[1] * inv) * silu(1.0)]
        };
        let expect1: Vec<f32> = (0..4)
            .flat_map(|h| y_of(o_of(h % 2)))
            .collect();
        // 判別力: blocked (h*nk/nv) では h=2 が k-head 1 (=q [0,1] 向き) を使う
        let expect1_blocked_h2 = y_of(o_of(2 * 2 / 4));
        assert!(
            (expect1[4] - expect1_blocked_h2[0]).abs() > 1e-3
                || (expect1[5] - expect1_blocked_h2[1]).abs() > 1e-3,
            "テスト入力が blocked/strided を区別できていない"
        );
        for h in 0..4 {
            assert!(
                (y[1][h * 2] - expect1[h * 2]).abs() < 1e-4
                    && (y[1][h * 2 + 1] - expect1[h * 2 + 1]).abs() < 1e-4,
                "h={h}: got=({:.6},{:.6}) want(strided)=({:.6},{:.6})",
                y[1][h * 2],
                y[1][h * 2 + 1],
                expect1[h * 2],
                expect1[h * 2 + 1]
            );
        }
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

    /// チャンク版 ([`forward_batch_core_gate`]) がトークン逐次とビット一致すること。
    /// 形状は実モデル (qwen4exp の ssm 1 層: cd=10240 / vd=6144 / nv=48 / dk=dv=128)。
    /// 初期状態は**非ゼロの乱数**にして conv プロローグ (負の時刻) と状態の持ち越し
    /// の両方を通す。
    #[test]
    fn gdn_batch_matches_token_by_token_bitexact() {
        let p = GdnParams::qwen4exp();
        for nt in [1usize, 3, 64, 512] {
            let (w, qkv, z, beta, alpha) = real_case(nt);
            let cw = core(&p, &w);
            let mut rnd = lcg_rng(0xabcd_1234 ^ nt as u64);
            let st0 = GdnState {
                conv: (0..p.conv_state_len()).map(|_| rnd()).collect(),
                ssm: (0..p.ssm_state_len()).map(|_| rnd()).collect(),
            };

            // 参照: 現行のトークン逐次ループ
            let mut st_ref = st0.clone();
            let mut y_ref = vec![0.0f32; nt * p.value_dim()];
            seq_tokens(&p, &cw, nt, &qkv, &z, &beta, &alpha, &mut st_ref, &mut y_ref);

            // 新: チャンク一括
            let mut st_new = st0.clone();
            let y_new = forward_batch_core_gate(
                &p,
                &cw,
                &qkv,
                &z,
                &beta,
                &alpha,
                &mut st_new,
                GateKind::Sigmoid,
            );

            assert_eq!(y_new.len(), nt * p.value_dim(), "nt={nt}: 出力長");
            assert_eq!(bits(&y_ref), bits(&y_new), "nt={nt}: y がビット不一致");
            assert_eq!(bits(&st_ref.ssm), bits(&st_new.ssm), "nt={nt}: ssm 状態");
            assert_eq!(bits(&st_ref.conv), bits(&st_new.conv), "nt={nt}: conv 状態");
        }
    }

    /// チャンクを分割しても 1 回で回したのと同じ (状態の持ち越しが正しい)。
    /// conv 状態は「チャンク末尾の d_conv-1 本」で決まるので、ここがずれると
    /// 2 回目の先頭トークンだけが静かに壊れる。
    #[test]
    fn gdn_batch_chunk_split_matches_single_chunk() {
        let p = GdnParams::qwen4exp();
        let nt = 40usize;
        let (w, qkv, z, beta, alpha) = real_case(nt);
        let cw = core(&p, &w);
        let cd = p.conv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;

        let mut st_a = GdnState::new(&p);
        let y_all = forward_batch_core_gate(
            &p,
            &cw,
            &qkv,
            &z,
            &beta,
            &alpha,
            &mut st_a,
            GateKind::Sigmoid,
        );

        let mut st_b = GdnState::new(&p);
        let mut y_split = Vec::with_capacity(nt * vd);
        for (t0, t1) in [(0usize, 13usize), (13, 14), (14, 40)] {
            let n = t1 - t0;
            let y = forward_batch_core_gate(
                &p,
                &cw,
                &qkv[t0 * cd..t1 * cd],
                &z[t0 * vd..t1 * vd],
                &beta[t0 * nv..t1 * nv],
                &alpha[t0 * nv..t1 * nv],
                &mut st_b,
                GateKind::Sigmoid,
            );
            assert_eq!(y.len(), n * vd);
            y_split.extend_from_slice(&y);
        }
        assert_eq!(bits(&y_all), bits(&y_split), "分割チャンクで y が変わる");
        assert_eq!(bits(&st_a.ssm), bits(&st_b.ssm), "分割チャンクで ssm 状態");
        assert_eq!(bits(&st_a.conv), bits(&st_b.conv), "分割チャンクで conv 状態");
    }

    // -----------------------------------------------------------------------
    // P-PF-GDN: 実形状マイクロベンチ (qwen4exp の ssm 1 層・nt=512)
    // -----------------------------------------------------------------------

    /// 実形状 (`GdnParams::qwen4exp`) の投影済み入力一式を乱数で作る。
    ///
    /// 投影 (`w_qkv` 等) は使わないので空。時間は入力値に依存しない
    /// (分岐は `if kr != 0.0` だけで、乱数は 0 を踏まない)。
    fn real_case(
        nt: usize,
    ) -> (GdnWeights, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        case_for(&GdnParams::qwen4exp(), nt)
    }

    /// 任意のパラメータで投影済み入力一式を乱数で作る (`real_case` の一般形)。
    fn case_for(
        p: &GdnParams,
        nt: usize,
    ) -> (GdnWeights, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut rnd = lcg_rng(0x5eed_1234);
        let conv1d: Vec<f32> = (0..p.conv_dim() * p.d_conv).map(|_| rnd() * 0.4).collect();
        let w = GdnWeights {
            w_qkv: Vec::new(),
            w_gate: Vec::new(),
            w_beta: Vec::new(),
            w_alpha: Vec::new(),
            conv1d,
            ssm_a: (0..p.n_v_heads).map(|_| -0.02 - 0.05 * rnd().abs()).collect(),
            ssm_dt: (0..p.n_v_heads).map(|_| 0.1 * rnd()).collect(),
            ssm_norm: (0..p.head_v_dim).map(|_| 0.5 + rnd().abs()).collect(),
            w_out: Vec::new(),
        };
        let qkv: Vec<f32> = (0..nt * p.conv_dim()).map(|_| 2.0 * rnd()).collect();
        let z: Vec<f32> = (0..nt * p.value_dim()).map(|_| 4.0 * rnd()).collect();
        let beta: Vec<f32> = (0..nt * p.n_v_heads).map(|_| 2.0 * rnd()).collect();
        let alpha: Vec<f32> = (0..nt * p.n_v_heads).map(|_| 2.0 * rnd()).collect();
        (w, qkv, z, beta, alpha)
    }

    /// 現行のトークン逐次ループ (= `SsmLayer::forward_batch` が回しているもの)。
    fn seq_tokens(
        p: &GdnParams,
        cw: &GdnCore<'_>,
        nt: usize,
        qkv: &[f32],
        z: &[f32],
        beta: &[f32],
        alpha: &[f32],
        st: &mut GdnState,
        out: &mut [f32],
    ) {
        let cd = p.conv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;
        for t in 0..nt {
            let y = forward_token_core_gate(
                p,
                cw,
                &qkv[t * cd..(t + 1) * cd],
                &z[t * vd..(t + 1) * vd],
                &beta[t * nv..(t + 1) * nv],
                &alpha[t * nv..(t + 1) * nv],
                st,
                GateKind::Sigmoid,
            );
            out[t * vd..(t + 1) * vd].copy_from_slice(&y);
        }
    }

    fn best_of(reps: usize, mut f: impl FnMut()) -> f64 {
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            f();
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    }

    /// 内訳 (a) conv1d / (a2) L2 正規化 / (b) beta・alpha 変換 / (c) スキャン /
    /// (d) gated RMS norm を実形状で測る。合計は現行のトークン逐次 1 回ぶん。
    ///
    /// 実行: cargo test --release -p denno-engine --lib -- --ignored --nocapture bench_gdn
    #[test]
    #[ignore = "実形状 (nt=512) のマイクロベンチ。--ignored --nocapture で実行"]
    fn bench_gdn_core_real_shape() {
        let nt = 512usize;
        let p = GdnParams::qwen4exp();
        let (w, qkv, z, beta, alpha) = real_case(nt);
        let cw = core(&p, &w);
        let cd = p.conv_dim();
        let vd = p.value_dim();
        let nv = p.n_v_heads;
        let kd = p.key_dim();
        let dk = p.head_k_dim;
        let dv = p.head_v_dim;
        let dc = p.d_conv;
        let cs = p.d_conv - 1;
        let reps = 3;

        // ---- 0) 現行: トークン逐次 (全部入り) ----
        let mut ys = vec![0.0f32; nt * vd];
        let mut st = GdnState::new(&p);
        let total = best_of(reps, || {
            st = GdnState::new(&p);
            seq_tokens(&p, &cw, nt, &qkv, &z, &beta, &alpha, &mut st, &mut ys);
        });

        // ---- (a) conv + 状態ロール + silu (確保も現行と同じく内側) ----
        let mut st_a = GdnState::new(&p);
        let mut conv_out = vec![0.0f32; cd];
        let ta = best_of(reps, || {
            st_a = GdnState::new(&p);
            for t in 0..nt {
                let mut conv_in = vec![0.0f32; cd * dc];
                for c in 0..cd {
                    conv_in[c * dc..c * dc + cs].copy_from_slice(&st_a.conv[c * cs..(c + 1) * cs]);
                    conv_in[c * dc + cs] = qkv[t * cd + c];
                }
                for c in 0..cd {
                    st_a.conv[c * cs..(c + 1) * cs]
                        .copy_from_slice(&conv_in[c * dc + 1..(c + 1) * dc]);
                }
                for c in 0..cd {
                    let row = &cw.conv1d[c * dc..(c + 1) * dc];
                    let inp = &conv_in[c * dc..(c + 1) * dc];
                    let mut acc = 0.0f32;
                    for (a, b) in row.iter().zip(inp.iter()) {
                        acc += a * b;
                    }
                    conv_out[c] = silu(acc);
                }
            }
        });

        // ---- (a2) L2 正規化 (q/k, 全トークン先出し) ----
        let mut qall = vec![0.0f32; nt * kd];
        let mut kall = vec![0.0f32; nt * kd];
        let q_scale = 1.0 / (dk as f32).sqrt();
        let ta2 = best_of(reps, || {
            for t in 0..nt {
                for h in 0..p.n_k_heads {
                    let o = h * dk;
                    qall[t * kd + o..t * kd + o + dk]
                        .copy_from_slice(&conv_out[o..o + dk]);
                    kall[t * kd + o..t * kd + o + dk]
                        .copy_from_slice(&conv_out[kd + o..kd + o + dk]);
                    l2_norm(&mut qall[t * kd + o..t * kd + o + dk], p.eps);
                    l2_norm(&mut kall[t * kd + o..t * kd + o + dk], p.eps);
                    for v in qall[t * kd + o..t * kd + o + dk].iter_mut() {
                        *v *= q_scale;
                    }
                }
            }
        });

        // ---- (b) beta・alpha の変換 (sigmoid / softplus / exp) — (c) の内数 ----
        let mut egt = vec![0.0f32; nt * nv];
        let mut bt = vec![0.0f32; nt * nv];
        let tb = best_of(reps, || {
            for t in 0..nt {
                for h in 0..nv {
                    let g = cw.ssm_a[h] * softplus(alpha[t * nv + h] + cw.ssm_dt[h]);
                    egt[t * nv + h] = g.exp();
                    bt[t * nv + h] = sigmoid(beta[t * nv + h]);
                }
            }
        });

        // ---- (c) スキャン (gdn_head をトークン逐次・ヘッド逐次) ----
        let mut st_c = GdnState::new(&p);
        let mut o = vec![0.0f32; nt * vd];
        let tc = best_of(reps, || {
            st_c = GdnState::new(&p);
            for t in 0..nt {
                for h in 0..nv {
                    gdn_head(
                        &p,
                        &cw,
                        &qall[t * kd..(t + 1) * kd],
                        &kall[t * kd..(t + 1) * kd],
                        &conv_out[2 * kd..2 * kd + vd], // v は値に依らない
                        &beta[t * nv..(t + 1) * nv],
                        &alpha[t * nv..(t + 1) * nv],
                        &mut st_c.ssm[h * dk * dv..(h + 1) * dk * dv],
                        &mut o[t * vd + h * dv..t * vd + (h + 1) * dv],
                        h,
                    );
                }
            }
        });

        // ---- (d) gated RMS norm ----
        let mut y2 = vec![0.0f32; nt * vd];
        let td = best_of(reps, || {
            for t in 0..nt {
                for h in 0..nv {
                    let obase = t * vd + h * dv;
                    let mut seg: Vec<f32> = o[obase..obase + dv].to_vec();
                    rms_norm(&mut seg, p.eps);
                    for (c, v) in seg.iter_mut().enumerate() {
                        let zc = z[h * dv + c];
                        *v *= cw.ssm_norm[c] * sigmoid(zc);
                    }
                    y2[obase..obase + dv].copy_from_slice(&seg);
                }
            }
        });

        let per_tok = |s: f64| s * 1e6 / nt as f64;
        println!("GDN core 実形状 (nt={nt}, cd={cd}, vd={vd}, nv={nv}) — µs/token");
        println!("  現行 トークン逐次 合計 {:8.1}", per_tok(total));
        println!("  (a) conv+roll+silu   {:8.1}", per_tok(ta));
        println!("  (a2) L2 正規化 q/k   {:8.1}", per_tok(ta2));
        println!("  (b) 変換 (c の内数)  {:8.1}", per_tok(tb));
        println!("  (c) スキャン (逐次)  {:8.1}", per_tok(tc));
        println!("  (d) gated RMS norm   {:8.1}", per_tok(td));
        println!(
            "  現行のスキャン実効 (= 合計 - a - a2 - d) {:8.1}",
            per_tok(total - ta - ta2 - td)
        );

        // ---- ビット一致 (nt=512・実形状): 逐次 vs チャンク一括 ----
        let mut st_r = GdnState::new(&p);
        let mut y_ref = vec![0.0f32; nt * vd];
        seq_tokens(&p, &cw, nt, &qkv, &z, &beta, &alpha, &mut st_r, &mut y_ref);
        let mut st_b = GdnState::new(&p);
        let y_bat = forward_batch_core_gate(
            &p,
            &cw,
            &qkv,
            &z,
            &beta,
            &alpha,
            &mut st_b,
            GateKind::Sigmoid,
        );
        assert_eq!(bits(&y_ref), bits(&y_bat), "y がビット不一致");
        assert_eq!(bits(&st_r.ssm), bits(&st_b.ssm), "ssm 状態がビット不一致");
        assert_eq!(bits(&st_r.conv), bits(&st_b.conv), "conv 状態がビット不一致");

        // ---- 新実装 (チャンク一括・ヘッド範囲をタスクに固定) ----
        let mut st_n = GdnState::new(&p);
        let mut y_new = vec![0.0f32; nt * vd];
        let tnew = best_of(reps, || {
            st_n = GdnState::new(&p);
            y_new = forward_batch_core_gate(
                &p,
                &cw,
                &qkv,
                &z,
                &beta,
                &alpha,
                &mut st_n,
                GateKind::Sigmoid,
            );
        });
        assert_eq!(bits(&y_new), bits(&y_ref));
        println!("  新実装 合計          {:8.1}", per_tok(tnew));

        // ---- (c)(d) のグループ幅の掃引 (タスクあたりのヘッド数) ----
        let mut cv = vec![0.0f32; nt * cd];
        let conv_st0 = vec![0.0f32; cd * cs];
        conv_silu_all(&p, &cw, &qkv, &conv_st0, &mut cv, nt);
        for group in [1usize, 2, 3, 4, 6, 8, 12, 16] {
            let mut ssm = vec![0.0f32; p.ssm_state_len()];
            let mut yy = vec![0.0f32; nt * vd];
            let mut yyh = vec![0.0f32; nt * vd];
            let tg = best_of(reps, || {
                for v in ssm.iter_mut() {
                    *v = 0.0;
                }
                scan_gate_all(
                    &p,
                    &cw,
                    &cv,
                    &z,
                    &egt,
                    &bt,
                    nt,
                    &mut ssm,
                    &mut yyh,
                    &mut yy,
                    GateKind::Sigmoid,
                    group,
                );
            });
            println!("  group={group:<3} scan+gate{:8.1}", per_tok(tg));
        }
    }

    /// decode 実形 (nt=1 × 36 層 = 1 トークンぶん) の A/B と区間内訳。
    ///
    /// `DENNO_GDN_KPROF` をテストから強制し、4 区間 (conv / l2 / scan / rms) を
    /// 36 層積算で出す。壁時計は旧実装 (ref) と並列版 (par) を同一実行で比べる
    /// (ホストは他セッションと共有なので、別実行の比較はしない)。
    ///
    /// 実行: cargo test --release -p denno-engine --lib --features cuda \
    ///        -- --ignored --nocapture bench_gdn_decode
    #[test]
    #[ignore = "decode 実形 (nt=1) のマイクロベンチ。--ignored --nocapture で実行"]
    fn bench_gdn_decode_layers() {
        let _g = crate::prof::counters_guard();
        let layers = 36usize;
        let p = GdnParams::qwen4exp();
        let (w, qkv, z, beta, alpha) = real_case(1);
        let cw = core(&p, &w);

        // 壁時計 (状態は 0 から始まり、1 回目以降は値が入った状態で回る。
        // 時間は値に依存しない: 分岐は `if kr != 0.0` だけで q/k は L2 正規化後)。
        let one_rep = |f: &dyn Fn(&mut GdnState) -> Vec<f32>| {
            let mut st = GdnState::new(&p);
            best_of(5, || {
                for _ in 0..layers {
                    let y = f(&mut st);
                    std::hint::black_box(&y);
                }
            })
        };
        let total_ref = one_rep(&|st| {
            forward_token_core_gate_ref(&p, &cw, &qkv, &z, &beta, &alpha, st, GateKind::Sigmoid)
        });
        let total_par = one_rep(&|st| {
            forward_token_core_gate_par(&p, &cw, &qkv, &z, &beta, &alpha, st, GateKind::Sigmoid)
        });

        // 区間内訳 (計測 on で同じ 36 層)
        kprof::set_on_for_test(Some(true));
        kprof::reset();
        let mut st2 = GdnState::new(&p);
        for _ in 0..layers {
            let y = forward_token_core_gate_par(
                &p,
                &cw,
                &qkv,
                &z,
                &beta,
                &alpha,
                &mut st2,
                GateKind::Sigmoid,
            );
            std::hint::black_box(&y);
        }
        let (calls, ns) = kprof::take();
        let us_per_layer = |v: u64| v as f64 / 1e3 / layers as f64;
        kprof::set_on_for_test(None);
        kprof::reset();

        println!(
            "GDN decode 実形 (36 層, nv={}, dk={}, cd={})",
            p.n_v_heads,
            p.head_k_dim,
            p.conv_dim()
        );
        println!(
            "  旧 (ref)  {:8.3} ms/token ({:7.1} µs/層)  [壁時計]",
            total_ref * 1e3,
            total_ref * 1e6 / layers as f64
        );
        println!(
            "  並列 (par){:8.3} ms/token ({:7.1} µs/層)  比 {:.2}x",
            total_par * 1e3,
            total_par * 1e6 / layers as f64,
            total_ref / total_par
        );
        println!("  区間計測 (par) calls={calls} (36 層) [µs/層]");
        for (i, name) in kprof::NAMES.iter().enumerate() {
            println!(
                "    {name:<5} {:8.1}  ({:.1}%)",
                us_per_layer(ns[i]),
                ns[i] as f64 / ns.iter().sum::<u64>().max(1) as f64 * 100.0
            );
        }
        let sum: u64 = ns.iter().sum();
        println!(
            "    合計  {:8.1}  ({:.3} ms/token)",
            us_per_layer(sum),
            sum as f64 / 1e6
        );
    }

    /// 区間計測のゲートが実際に通ることを示す (緑 = 配線が生きている証拠)。
    ///
    /// - 強制 off: 1 回呼んでも call 数も区間も増えない・表示行は出ない。
    /// - 強制 on: 1 回呼ぶと call が 1 以上増え、4 区間すべて ns が増える。
    ///
    /// 他のテストが並列で `forward_token_core_gate` を呼ぶ可能性があるので、
    /// 「増えないこと」は差分 0、「増えること」は差分 > 0 で見る (>= 比較)。
    #[test]
    fn kprof_gate_covers_four_phases() {
        let _g = crate::prof::counters_guard();
        let p = GdnParams::qwen4exp();
        let (w, qkv, z, beta, alpha) = real_case(1);
        let cw = core(&p, &w);
        let mut st = GdnState::new(&p);

        // ---- off: 何も積まない ----
        kprof::set_on_for_test(Some(false));
        kprof::reset();
        let (c0, ns0) = kprof::take();
        assert!(kprof::line().is_none(), "off なのに表示行が出た");
        let y = forward_token_core_gate(
            &p,
            &cw,
            &qkv,
            &z,
            &beta,
            &alpha,
            &mut st,
            GateKind::Sigmoid,
        );
        assert_eq!(y.len(), p.value_dim());
        let (c1, ns1) = kprof::take();
        assert_eq!(c1, c0, "off なのに call を数えた");
        assert_eq!(ns1, ns0, "off なのに区間を積んだ");

        // ---- on: 4 区間すべてに ns が入る ----
        kprof::set_on_for_test(Some(true));
        kprof::reset();
        let (c2, ns2) = kprof::take();
        let _ = forward_token_core_gate(
            &p,
            &cw,
            &qkv,
            &z,
            &beta,
            &alpha,
            &mut st,
            GateKind::Sigmoid,
        );
        let (c3, ns3) = kprof::take();
        kprof::set_on_for_test(None);
        kprof::reset();
        assert!(c3 >= c2 + 1, "on なのに call が増えない ({c2} -> {c3})");
        for (i, name) in kprof::NAMES.iter().enumerate() {
            assert!(ns3[i] > ns2[i], "区間 {i} ({name}) に ns が入らない");
        }
    }

    /// 並列版 decode コアが旧実装と **f32 でビット一致**する。
    ///
    /// 実形 (nk=16 / nv=48 = 並列経路) と小型 (nk=1 / nv=2 = 直列経路) の両方を、
    /// 非ゼロ初期状態から 3 トークン進めて比較する (状態そのものも `to_bits` で比較)。
    /// 両方のゲート種別 (Silu / Sigmoid) を回す。
    #[test]
    fn par_core_matches_ref_bitexact() {
        let _g = crate::prof::counters_guard();
        let small = GdnParams {
            n_embd: 8,
            n_k_heads: 1,
            n_v_heads: 2,
            head_k_dim: 4,
            head_v_dim: 4,
            d_conv: 4,
            eps: 1e-6,
        };
        for p in [GdnParams::qwen4exp(), small] {
            let (w, qkv, z, beta, alpha) = case_for(&p, 1);
            let cw = core(&p, &w);
            for gate in [GateKind::Sigmoid, GateKind::Silu] {
                // 非ゼロ初期状態 (conv のロールと scan の両方を通す)
                let mut rnd = lcg_rng(0x0bad_c0de);
                let mut st_ref = GdnState::new(&p);
                for v in st_ref.conv.iter_mut() {
                    *v = rnd();
                }
                for v in st_ref.ssm.iter_mut() {
                    *v = rnd();
                }
                let mut st_par = GdnState {
                    conv: st_ref.conv.clone(),
                    ssm: st_ref.ssm.clone(),
                };
                for step in 0..3 {
                    let ya = forward_token_core_gate_ref(
                        &p, &cw, &qkv, &z, &beta, &alpha, &mut st_ref, gate,
                    );
                    let yb = forward_token_core_gate_par(
                        &p, &cw, &qkv, &z, &beta, &alpha, &mut st_par, gate,
                    );
                    assert_eq!(
                        bits(&ya),
                        bits(&yb),
                        "y がビット不一致 (dk={}, nv={}, step={step}, gate={gate:?})",
                        p.head_k_dim,
                        p.n_v_heads
                    );
                    assert_eq!(
                        bits(&st_ref.ssm),
                        bits(&st_par.ssm),
                        "ssm 状態がビット不一致 (step={step})"
                    );
                    assert_eq!(
                        bits(&st_ref.conv),
                        bits(&st_par.conv),
                        "conv 状態がビット不一致 (step={step})"
                    );
                }
            }
        }
    }
}

