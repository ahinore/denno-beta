//! P-MOE-F2: `model::qwen4exp_ssm::SsmLayer` の形状検査と切り分けのテスト。
//!
//! 実モデル (111 GB) は使わない。このファイル内で **合成 GGUF** を書き出し、
//! `SsmLayer::from_gguf` / `forward_token` / `project_qkv` を通す。
//!
//! 期待値の出どころ:
//! - qkv の切り分け `[q 2048][k 2048][v 6144]` は llama.cpp `qwen4exp.cpp`
//!   `build_layer_attn_linear` の `q_conv` / `k_conv` / `v_conv` の view オフセット
//!   (0 / head_k_dim*num_k_heads / 2*head_k_dim*num_k_heads)。
//! - GQA の繰り返しは `qwen4exp.cpp` の `ggml_repeat_4d` (循環 = `h % n_k_heads`)。
//!   HF transformers の `repeat_interleave` (blocked = `h / 3`) とは別規約。
//! - デルタ則の 1 ステップは参照実装の式を手で書き下した閉形式と比べる
//!   (gdn.rs のコードを呼ばずに済ませるため、テスト内で独立に計算する)。
//!
//! 出力ゲートは参照実装が `sigmoid(z)`、Qwen3.5 系 (`gdn.rs` の既定) が
//! `silu(z)` で**違う** (`build_norm_gated` のコメント参照)。P-MOE-F11 で
//! `GateKind` を足し、qwen4exp の層は `Sigmoid` を渡すようにした。この
//! テストの期待値も `sigmoid` 側で書く。

use std::path::{Path, PathBuf};

use denno_engine::gdn::{
    forward_token_core_gate, GateKind, GdnCore, GdnParams, GdnState,
};
use denno_engine::gguf::GgufFile;
use denno_engine::model::qwen4exp_ssm::SsmLayer;

// ---- 諸元 (flashnext_meta.txt) -------------------------------------------
const N_EMBD: usize = 2560;
const KEY_DIM: usize = 2048;
const VALUE_DIM: usize = 6144;
const QKV_DIM: usize = 10240;
const N_V_HEADS: usize = 48;
const D_CONV: usize = 4;
const HEAD_V_DIM: usize = 128;
const CONV_DIM: usize = QKV_DIM;

// ---- GGUF 書き出し -------------------------------------------------------
const ALIGN: usize = 32;
const GT_U32: u32 = 4;
const GT_STRING: u32 = 8;
const TY_F32: u32 = 0;
const TY_Q4_0: u32 = 2;
const TY_Q8_0: u32 = 8;

/// f16 の 1.0 (Q8_0 のスケール。1.0 は f16 で厳密)。
const F16_ONE: [u8; 2] = [0x00, 0x3c];

enum Kv {
    U32(u32),
    Str(&'static str),
}

impl Kv {
    fn write(&self, b: &mut Vec<u8>) {
        match self {
            Kv::U32(v) => {
                b.extend_from_slice(&GT_U32.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
            Kv::Str(s) => {
                b.extend_from_slice(&GT_STRING.to_le_bytes());
                push_str(b, s);
            }
        }
    }
}

struct TensorSpec {
    name: String,
    dims: Vec<u64>,
    ty: u32,
    data: Vec<u8>,
}

fn push_str(b: &mut Vec<u8>, s: &str) {
    b.extend_from_slice(&(s.len() as u64).to_le_bytes());
    b.extend_from_slice(s.as_bytes());
}

fn align_up(v: usize, a: usize) -> usize {
    (v + a - 1) & !(a - 1)
}

/// 最小 GGUF (v3) を書き出す。データ部は 32 B 整列、テンソルは与えた順に並ぶ。
fn write_gguf(path: &Path, kvs: &[(&'static str, Kv)], tensors: &[TensorSpec]) {
    let mut head = Vec::new();
    head.extend_from_slice(b"GGUF");
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&((kvs.len() + 1) as u64).to_le_bytes());
    for (k, v) in kvs {
        push_str(&mut head, k);
        v.write(&mut head);
    }
    push_str(&mut head, "general.alignment");
    Kv::U32(ALIGN as u32).write(&mut head);

    let mut offsets = Vec::with_capacity(tensors.len());
    let mut cur = 0usize;
    for t in tensors {
        offsets.push(cur);
        cur = align_up(cur + t.data.len(), ALIGN);
    }
    for (t, off) in tensors.iter().zip(&offsets) {
        push_str(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.ty.to_le_bytes());
        head.extend_from_slice(&(*off as u64).to_le_bytes());
    }

    let mut out = head;
    out.resize(align_up(out.len(), ALIGN), 0);
    for t in tensors {
        out.extend_from_slice(&t.data);
        out.resize(align_up(out.len(), ALIGN), 0);
    }
    std::fs::write(path, out).expect("テスト GGUF の書き出し");
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "denno_qwen4exp_ssm_{tag}_{}_{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("temp dir");
        TempDir(p)
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- テンソル生成 --------------------------------------------------------

/// Q8_0 の 1 ブロック = f16 スケール 2 B + int8 32 本 = 34 B (ggml の block_q8_0)。
const Q8_0_BLOCK: usize = 34;

/// Q8_0 行列 (k 列 n 行)。`ones` の (行, 列) だけ 1.0、残りは 0。
/// スケールは全ブロック f16 の 1.0 なので、1.0 の要素はデコードしても厳密に 1.0。
fn q8_0_matrix(k: usize, n: usize, ones: &[(usize, usize)]) -> Vec<u8> {
    assert_eq!(k % 32, 0, "Q8_0 の行は 32 の倍数");
    let nb = k / 32;
    let rb = nb * Q8_0_BLOCK;
    let mut out = vec![0u8; n * rb];
    for r in 0..n {
        for b in 0..nb {
            let o = r * rb + b * Q8_0_BLOCK;
            out[o..o + 2].copy_from_slice(&F16_ONE);
        }
    }
    for &(r, c) in ones {
        assert!(r < n && c < k, "ones の ({r}, {c}) が範囲外");
        let o = r * rb + (c / 32) * Q8_0_BLOCK;
        out[o + 2 + (c % 32)] = 1;
    }
    out
}

fn f32_data(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 1 層ぶんの合成重み。`from_gguf` が要求する 9 本をすべて持つ。
struct LayerSpec {
    qkv_ones: Vec<(usize, usize)>,
    gate_ones: Vec<(usize, usize)>,
    out_ones: Vec<(usize, usize)>,
    /// [conv_dim][d_conv]
    conv1d: Vec<f32>,
    ssm_a: Vec<f32>,
    ssm_dt: Vec<f32>,
    ssm_norm: Vec<f32>,
}

impl LayerSpec {
    /// conv1d は `tap` 番目だけ 1.0 (他は 0)、ssm_a = -1、dt = 0、norm = 1。
    fn with_conv_tap(tap: usize) -> Self {
        let mut conv1d = vec![0f32; CONV_DIM * D_CONV];
        for c in 0..CONV_DIM {
            conv1d[c * D_CONV + tap] = 1.0;
        }
        Self {
            qkv_ones: vec![(0, 0), (KEY_DIM, 0), (2 * KEY_DIM, 0)],
            gate_ones: (0..VALUE_DIM).map(|c| (c, 0)).collect(),
            out_ones: (0..8).map(|r| (r, r)).collect(),
            conv1d,
            ssm_a: vec![-1.0; N_V_HEADS],
            ssm_dt: vec![0.0; N_V_HEADS],
            ssm_norm: vec![1.0; HEAD_V_DIM],
        }
    }

    fn tensors(&self, il: usize) -> Vec<TensorSpec> {
        let n = |s: &str| format!("blk.{il}.{s}");
        vec![
            TensorSpec {
                name: n("attn_qkv.weight"),
                dims: vec![N_EMBD as u64, QKV_DIM as u64],
                ty: TY_Q8_0,
                data: q8_0_matrix(N_EMBD, QKV_DIM, &self.qkv_ones),
            },
            TensorSpec {
                name: n("attn_gate.weight"),
                dims: vec![N_EMBD as u64, VALUE_DIM as u64],
                ty: TY_Q8_0,
                data: q8_0_matrix(N_EMBD, VALUE_DIM, &self.gate_ones),
            },
            // beta/alpha は F32 の生値。0 行列 → beta_raw = 0 → sigmoid = 0.5。
            TensorSpec {
                name: n("ssm_beta.weight"),
                dims: vec![N_EMBD as u64, N_V_HEADS as u64],
                ty: TY_F32,
                data: f32_data(&vec![0.0; N_EMBD * N_V_HEADS]),
            },
            TensorSpec {
                name: n("ssm_alpha.weight"),
                dims: vec![N_EMBD as u64, N_V_HEADS as u64],
                ty: TY_F32,
                data: f32_data(&vec![0.0; N_EMBD * N_V_HEADS]),
            },
            TensorSpec {
                name: n("ssm_conv1d.weight"),
                dims: vec![D_CONV as u64, CONV_DIM as u64],
                ty: TY_F32,
                data: f32_data(&self.conv1d),
            },
            TensorSpec {
                name: n("ssm_a"),
                dims: vec![N_V_HEADS as u64],
                ty: TY_F32,
                data: f32_data(&self.ssm_a),
            },
            TensorSpec {
                name: n("ssm_dt.bias"),
                dims: vec![N_V_HEADS as u64],
                ty: TY_F32,
                data: f32_data(&self.ssm_dt),
            },
            TensorSpec {
                name: n("ssm_norm.weight"),
                dims: vec![HEAD_V_DIM as u64],
                ty: TY_F32,
                data: f32_data(&self.ssm_norm),
            },
            TensorSpec {
                name: n("ssm_out.weight"),
                dims: vec![VALUE_DIM as u64, N_EMBD as u64],
                ty: TY_Q8_0,
                data: q8_0_matrix(VALUE_DIM, N_EMBD, &self.out_ones),
            },
        ]
    }
}

fn kvs() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen4exp")),
        ("qwen4exp.block_count", Kv::U32(48)),
        ("qwen4exp.embedding_length", Kv::U32(N_EMBD as u32)),
        ("qwen4exp.ssm.state_size", Kv::U32(128)),
        ("qwen4exp.ssm.group_count", Kv::U32(16)),
        ("qwen4exp.ssm.time_step_rank", Kv::U32(48)),
        ("qwen4exp.ssm.inner_size", Kv::U32(VALUE_DIM as u32)),
        ("qwen4exp.ssm.conv_kernel", Kv::U32(D_CONV as u32)),
    ]
}

/// 1 層ぶんの合成 GGUF を書いて開く。
fn open_layer(dir: &TempDir, tag: &str, spec: &LayerSpec) -> GgufFile {
    let path = dir.path(&format!("{tag}.gguf"));
    write_gguf(&path, &kvs(), &spec.tensors(0));
    GgufFile::open(&path).expect("合成 GGUF を開けない")
}

fn onehot(i: usize, n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    v[i] = 1.0;
    v
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `from_gguf` が Err になることを確かめ、エラー文を返す (`SsmLayer` は Debug 無し)。
fn from_gguf_err(f: &GgufFile, il: usize, what: &str) -> String {
    match SsmLayer::from_gguf(f, il) {
        Ok(_) => panic!("{what}: Err のはずが Ok になった"),
        Err(e) => format!("{e}"),
    }
}

// ---- 1. qkv の切り分け ---------------------------------------------------

/// `attn_qkv` の出力行 0 / 2047 / 2048 / 4095 / 4096 / 10239 が、それぞれ
/// 入力の別々の要素から立つことを確かめる (投影の行→入力対応)。
/// そのうえで q/k/v の境界が 2048 / 4096 / 10240 であることを `qkv_split()` で固定する。
#[test]
fn qkv_projection_rows_and_split() {
    let dir = TempDir::new("qkv");
    let mut spec = LayerSpec::with_conv_tap(3);
    // 行 → 入力列の対応。q の端 (0, 2047)・k の端 (2048, 4095)・v の端 (4096, 10239)。
    let pairs = [(0usize, 1usize), (2047, 2), (2048, 3), (4095, 4), (4096, 5), (10239, 6)];
    spec.qkv_ones = pairs.to_vec();
    let f = open_layer(&dir, "qkv", &spec);
    let layer = SsmLayer::from_gguf(&f, 0).expect("from_gguf");

    // qkv の切り分け (q 2048 / k 2048 / v 6144)
    assert_eq!(layer.qkv_split(), (2048, 4096, 10240));
    assert_eq!(layer.p.n_embd, N_EMBD);
    assert_eq!(layer.p.n_k_heads, 16);
    assert_eq!(layer.p.n_v_heads, 48);
    assert_eq!(layer.p.qkv_dim(), QKV_DIM);

    for (row, col) in pairs {
        let got = layer.project_qkv(&onehot(col, N_EMBD)).expect("project_qkv");
        assert_eq!(got.len(), QKV_DIM);
        // 立っているのは 1 行だけ・値は厳密に 1.0 (Q8_0 のスケールが 1.0 のため)
        let live: Vec<usize> = (0..QKV_DIM).filter(|&i| got[i] != 0.0).collect();
        assert_eq!(live, vec![row], "入力 {col} が行 {row} 以外にも出ている");
        assert_eq!(got[row], 1.0, "入力 {col} → 行 {row} の値が 1.0 でない");
    }

    // どの行も読まない入力は全 0
    let got = layer.project_qkv(&onehot(0, N_EMBD)).expect("project_qkv");
    assert!(got.iter().all(|&v| v == 0.0), "未接続の入力で非零が出た");
}

// ---- 2. GQA の繰り返し + デルタ則 1 ステップ (小さい次元・手計算) --------

/// sigmoid (qwen4exp の出力ゲート)
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// silu (conv の活性化。こちらは Qwen3.5 と同じ)
fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// `n_k_heads=2 / n_v_heads=6` (比 3) で、v head h が k head **h % 2** を使うことを
/// 1 ステップのデルタ則の閉形式と比べて確かめる。
/// `h / 3` (blocked = HF の repeat_interleave) では h=4 で k head が変わるので、
/// その差が状態に出ることも同時に見る。
#[test]
fn gqa_repeat_is_strided_and_matches_closed_form_step() {
    let p = GdnParams {
        n_embd: 4,
        n_k_heads: 2,
        n_v_heads: 6,
        head_k_dim: 2,
        head_v_dim: 2,
        d_conv: 2,
        eps: 1e-6,
    };
    let conv_dim = p.conv_dim();
    assert_eq!(conv_dim, 20);

    // conv は最新 tap だけ 1.0 → conv_out[c] = silu(qkv[c])
    let mut conv1d = vec![0f32; conv_dim * p.d_conv];
    for c in 0..conv_dim {
        conv1d[c * p.d_conv + (p.d_conv - 1)] = 1.0;
    }
    let ssm_a = vec![-1.0f32; p.n_v_heads];
    let ssm_dt = vec![0.0f32; p.n_v_heads];
    let ssm_norm = vec![1.0f32; p.head_v_dim];
    let cw = GdnCore {
        conv1d: &conv1d,
        ssm_a: &ssm_a,
        ssm_dt: &ssm_dt,
        ssm_norm: &ssm_norm,
    };

    // q: head0 = [0.5, -1.5], head1 = [2.0, 0.25]
    // k: head0 = [1.0,  3.0], head1 = [-2.0, 0.5]
    // v: head h = [h+1, -(h+1)/2]
    let mut qkv = vec![0f32; conv_dim];
    qkv[0..4].copy_from_slice(&[0.5, -1.5, 2.0, 0.25]);
    qkv[4..8].copy_from_slice(&[1.0, 3.0, -2.0, 0.5]);
    for h in 0..p.n_v_heads {
        qkv[8 + h * 2] = (h + 1) as f32;
        qkv[8 + h * 2 + 1] = -((h + 1) as f32) * 0.5;
    }
    let z = vec![0.25f32; p.value_dim()];
    let beta_raw = vec![0.0f32; p.n_v_heads]; // sigmoid(0) = 0.5
    let alpha = vec![0.0f32; p.n_v_heads]; // softplus(0+0) = ln 2

    let mut st = GdnState::new(&p);
    let y = forward_token_core_gate(
        &p,
        &cw,
        &qkv,
        &z,
        &beta_raw,
        &alpha,
        &mut st,
        GateKind::Sigmoid,
    );

    // ---- 閉形式 (f64) で期待値を作る ----
    let c: Vec<f64> = qkv.iter().map(|&v| silu_f64(v as f64)).collect();
    let kd = p.head_k_dim;
    let vd = p.head_v_dim;
    let q_scale = 1.0 / (kd as f64).sqrt();
    let mut q = vec![0f64; p.n_k_heads * kd];
    let mut k = vec![0f64; p.n_k_heads * kd];
    for h in 0..p.n_k_heads {
        let o = h * kd;
        let qs: f64 = (0..kd).map(|r| c[o + r] * c[o + r]).sum();
        let ks: f64 = (0..kd).map(|r| c[4 + o + r] * c[4 + o + r]).sum();
        let qi = 1.0 / (qs + p.eps as f64).sqrt();
        let ki = 1.0 / (ks + p.eps as f64).sqrt();
        for r in 0..kd {
            q[o + r] = c[o + r] * qi * q_scale;
            k[o + r] = c[4 + o + r] * ki;
        }
    }
    let beta = 0.5f64;
    let mut want = vec![0f64; p.value_dim()];
    let mut want_state = vec![0f64; p.n_v_heads * kd * vd];
    for h in 0..p.n_v_heads {
        let kh = h % p.n_k_heads; // cyclic (ggml_repeat_4d)
        let mut o = vec![0f64; vd];
        for r in 0..kd {
            for cc in 0..vd {
                // S[r][c] = k[kh][r] * beta * v[h][c] (1 ステップ目は S=0 起点)
                let s = k[kh * kd + r] * beta * c[8 + h * vd + cc];
                want_state[(h * kd + r) * vd + cc] = s;
                o[cc] += q[kh * kd + r] * s;
            }
        }
        let mean: f64 = o.iter().map(|x| x * x).sum::<f64>() / vd as f64;
        let inv = 1.0 / (mean + p.eps as f64).sqrt();
        for cc in 0..vd {
            want[h * vd + cc] = o[cc] * inv * ssm_norm[cc] as f64 * sigmoid(z[h * vd + cc] as f64);
        }
    }

    for i in 0..y.len() {
        let d = (y[i] as f64 - want[i]).abs();
        assert!(
            d <= 1e-6 * want[i].abs().max(1.0),
            "y[{i}]: got={} want={} (差 {d:.3e})",
            y[i],
            want[i]
        );
    }

    // 状態は k head の取り違えを直接見る。h=4 は h%2=0 / h/3=1 で別の head。
    for h in [4usize, 5] {
        let kh = h % p.n_k_heads;
        let kh_blocked = h / 3;
        let s = st.ssm[(h * kd) * vd] as f64; // S[h][0][0]
        let s_strided = k[kh * kd] * beta * c[8 + h * vd] as f64;
        let s_blocked = k[kh_blocked * kd] * beta * c[8 + h * vd] as f64;
        assert!(
            (s - s_strided).abs() <= 1e-6 * s_strided.abs().max(1.0),
            "h={h}: S[0][0]={s} が strided (k head {kh}) の {s_strided} と違う"
        );
        if (s_blocked - s_strided).abs() > 1e-3 {
            assert!(
                (s - s_blocked).abs() > 1e-3,
                "h={h}: blocked (k head {kh_blocked}) と区別できていない"
            );
        }
    }
    // 状態全体も閉形式と一致する
    for (i, (got, want)) in st.ssm.iter().zip(want_state.iter()).enumerate() {
        let d = (*got as f64 - *want).abs();
        assert!(
            d <= 1e-6 * want.abs().max(1.0),
            "state[{i}]: got={got} want={want}"
        );
    }
}

// ---- 3. conv の履歴 ------------------------------------------------------

/// conv 重みを「1 つ前のトークン」の tap (= 2) だけ 1.0 にする。
/// 1 トークン目は履歴が 0 なので出力が厳密に 0、2 トークン目は 1 トークン目の
/// 入力で conv が立ち、出力が非零になることを見る (state が進んでいる証拠)。
/// 併せて conv 状態の中身が「直前 d_conv-1 入力」になっていることも直接見る。
#[test]
fn conv_state_carries_previous_token() {
    let dir = TempDir::new("conv");
    let spec = LayerSpec::with_conv_tap(2); // 2 = 1 つ前のトークン
    let f = open_layer(&dir, "conv", &spec);
    let mut layer = SsmLayer::from_gguf(&f, 0).expect("from_gguf");

    let x = onehot(0, N_EMBD);
    let qkv1 = layer.project_qkv(&x).expect("project_qkv");
    // 合成重みでは q/k/v の先頭要素だけが立つ
    assert_eq!(qkv1[0], 1.0);
    assert_eq!(qkv1[KEY_DIM], 1.0);
    assert_eq!(qkv1[2 * KEY_DIM], 1.0);

    let out1 = layer.forward_token(&x).expect("forward_token 1");
    assert!(
        out1.iter().all(|&v| v == 0.0),
        "1 トークン目は conv 履歴が 0 なので出力が 0 のはず"
    );

    // conv 状態 = [0, 0, qkv1] (チャネルごと・末尾が最新)
    let cs = D_CONV - 1;
    for c in [0usize, KEY_DIM, 2 * KEY_DIM] {
        let s = &layer.state.conv[c * cs..(c + 1) * cs];
        assert_eq!(s[cs - 1], qkv1[c], "ch {c}: conv 状態の最新が qkv1 と違う");
        assert!(
            s[..cs - 1].iter().all(|&v| v == 0.0),
            "ch {c}: 履歴の古い側が 0 でない"
        );
    }
    // 立っていないチャネルは状態も 0
    assert!(layer.state.conv.iter().filter(|&&v| v != 0.0).count() == 3);

    let out2 = layer.forward_token(&x).expect("forward_token 2");
    assert_eq!(out2.len(), N_EMBD);
    assert!(
        out2.iter().all(|v| v.is_finite()),
        "2 トークン目に非有限値が出た"
    );
    // 2 トークン目の conv は qkv1 を読む → q/k/v が立ち、出力が非零
    let mx = out2.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(mx > 1e-3, "2 トークン目の出力が 0 のまま (conv が履歴を読んでいない)");
    // ssm_out は行 0..8 が単位行列なので out2[0] は y[0]
    assert_ne!(out2[0], 0.0);
}

// ---- 4. 形状検査 ---------------------------------------------------------

#[test]
fn from_gguf_rejects_bad_shapes() {
    let dir = TempDir::new("bad");

    // (a) attn_qkv の次元が違う
    let spec = LayerSpec::with_conv_tap(3);
    let mut tensors = spec.tensors(0);
    tensors[0].dims = vec![32, 32];
    tensors[0].data = q8_0_matrix(32, 32, &[]);
    let p = dir.path("bad_qkv.gguf");
    write_gguf(&p, &kvs(), &tensors);
    let f = GgufFile::open(&p).expect("open");
    let e = from_gguf_err(&f, 0, "attn_qkv の形状違い");
    assert!(
        e.contains("attn_qkv"),
        "エラーにテンソル名が無い: {e}"
    );

    // (b) ssm_a が欠けている
    let mut tensors = spec.tensors(0);
    tensors.retain(|t| t.name != "blk.0.ssm_a");
    let p = dir.path("bad_missing.gguf");
    write_gguf(&p, &kvs(), &tensors);
    let f = GgufFile::open(&p).expect("open");
    let e = from_gguf_err(&f, 0, "ssm_a が欠けている");
    assert!(e.contains("ssm_a"), "{e}");

    // (c) ssm_norm の長さが違う
    let mut tensors = spec.tensors(0);
    let norm = tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.ssm_norm.weight")
        .expect("ssm_norm");
    norm.dims = vec![64];
    norm.data = f32_data(&vec![1.0; 64]);
    let p = dir.path("bad_norm.gguf");
    write_gguf(&p, &kvs(), &tensors);
    let f = GgufFile::open(&p).expect("open");
    let e = from_gguf_err(&f, 0, "ssm_norm の長さ違い");
    assert!(e.contains("ssm_norm"), "{e}");

    // (d) gemv が扱えない量子型 (Q4_0) の attn_gate
    let mut tensors = spec.tensors(0);
    let gate = tensors
        .iter_mut()
        .find(|t| t.name == "blk.0.attn_gate.weight")
        .expect("gate");
    gate.ty = TY_Q4_0;
    gate.data = vec![0u8; VALUE_DIM * (N_EMBD / 32) * 18];
    let p = dir.path("bad_ty.gguf");
    write_gguf(&p, &kvs(), &tensors);
    let f = GgufFile::open(&p).expect("open");
    let e = from_gguf_err(&f, 0, "未対応の量子型");
    assert!(e.contains("量子型"), "{e}");

    // (e) 層番号が違えば別テンソルを探す → 無いので Err
    let p2 = dir.path("ok.gguf");
    write_gguf(&p2, &kvs(), &spec.tensors(0));
    let f = GgufFile::open(&p2).expect("open");
    assert!(SsmLayer::from_gguf(&f, 1).is_err(), "blk.1 は無いので Err");
}

// ---- 5. state の非対称性 -------------------------------------------------

/// 同じ入力を 2 回流すと 2 回目は 1 回目と違う (state が効いている)。
#[test]
fn same_input_twice_differs_after_state_update() {
    let dir = TempDir::new("state");
    // 最新 tap (3) を 1.0 → 各トークンの conv はそのトークン自身の入力を見る。
    // 2 回目の差は SSM 状態だけに由来する。
    let spec = LayerSpec::with_conv_tap(3);
    let f = open_layer(&dir, "state", &spec);
    let mut layer = SsmLayer::from_gguf(&f, 0).expect("from_gguf");

    let x = onehot(0, N_EMBD);
    // 参考値: 合成重み (Q8_0) での 1 トークンあたりの CPU 所要時間
    let t0 = std::time::Instant::now();
    let out1 = layer.forward_token(&x).expect("1 回目");
    let t1 = std::time::Instant::now();
    let out2 = layer.forward_token(&x).expect("2 回目");
    let t2 = std::time::Instant::now();
    eprintln!(
        "[TIME] forward_token 1 回目 {:?} / 2 回目 {:?} (合成重み・1 層)",
        t1 - t0,
        t2 - t1
    );

    let mx1 = out1.iter().fold(0f32, |a, v| a.max(v.abs()));
    assert!(mx1 > 1e-3, "1 回目の出力が 0 (合成重みの想定違い)");
    assert!(out2.iter().all(|v| v.is_finite()), "2 回目に非有限値");
    assert_ne!(bits(&out1), bits(&out2), "同じ入力なのに出力が同一 (state が効いていない)");
    // 状態そのものも進んでいる
    assert!(
        layer.state.ssm.iter().any(|&v| v != 0.0),
        "SSM 状態が 0 のまま"
    );
}
